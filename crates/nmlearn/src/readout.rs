//! The readout learner around `rls` (S11 revision 4, redesigned for the mini from EAMM V108.1,
//! V82.5 judge per scope and V97.3 gate on measured benefit). Pure CPU logic, no engine:
//! - scopes: a question's kind (`scope_of`); a unit is judged and activated per scope;
//! - the paired test (3a): unit against base on the same held-out cases, exact sign test on
//!   the discordant pairs;
//! - the accept rule (`judge`): 3b (no loss on unseen cases) first, then 3a; a unit that does
//!   not beat the base near seen cases is Rejected by the rule itself, so a shuffled-label
//!   control fails here and not through a budget;
//! - the local gate (`Gate`): a tiny RLS fitted on measured benefit (unit right minus base
//!   right) from a calibration split; the unit's order is used only where the gate predicts a
//!   gain;
//! - decay and retire (`Life`): the benefit seen in use decays with a half-life; a unit whose
//!   benefit stays below zero, or that sits idle too long, is retired (its files are kept).
use crate::rls::Rls;
use nmjson::Value;

/// The reranker scopes the readout knows, by question kind.
pub const SCOPES: [&str; 3] = ["reranker:value", "reranker:where-fn", "reranker:where-type"];
/// Fewest held-out paired cases per scope before 3a can pass.
pub const MIN_NEAR: usize = 30;
/// One-sided level of the sign test.
pub const ALPHA: f64 = 0.05;
/// Benefit half-life in use, days.
pub const HALF_LIFE_DAYS: f64 = 14.0;
/// A unit with at least `MIN_USES` uses and a decayed benefit below this is retired.
pub const RETIRE_BELOW: f64 = -0.05;
/// Uses before the benefit can retire a unit.
pub const MIN_USES: u64 = 20;
/// Days without use before a unit is retired.
pub const IDLE_DAYS: f64 = 30.0;
const DAY_MS: f64 = 86_400_000.0;

/// The scope of a reranker question.
#[must_use]
pub fn scope_of(question: &str) -> &'static str {
    if question.starts_with("What value is the constant") {
        "reranker:value"
    } else if question.starts_with("Where is the function") {
        "reranker:where-fn"
    } else if question.starts_with("Where is the type") {
        "reranker:where-type"
    } else {
        "reranker:other"
    }
}

/// Is the positive (index 0) strictly first under `scores`?
#[must_use]
pub fn first(scores: &[f64]) -> bool {
    scores.first().is_some_and(|p| scores.iter().skip(1).all(|s| s < p))
}

/// The paired comparison of a unit and the base on the same cases.
#[derive(Clone, Debug, PartialEq)]
pub struct Paired {
    pub n: usize,
    pub unit: usize,
    pub base: usize,
    /// Unit right, base wrong.
    pub b: usize,
    /// Base right, unit wrong.
    pub c: usize,
    /// One-sided exact sign test, P(X >= b) for X ~ Bin(b + c, 1/2).
    pub p: f64,
}
impl Paired {
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("n", num(self.n)),
            ("unit_right", num(self.unit)),
            ("base_right", num(self.base)),
            ("b_unit_only", num(self.b)),
            ("c_base_only", num(self.c)),
            ("p_one_sided", Value::Number(format!("{:.6}", self.p))),
        ])
    }
}

/// Pair up per-case right/wrong flags (same cases, same order).
#[must_use]
pub fn paired(unit: &[bool], base: &[bool]) -> Paired {
    let n = unit.len().min(base.len());
    let both = unit.iter().zip(base);
    let b = both.clone().filter(|(u, s)| **u && !**s).count();
    let c = both.clone().filter(|(u, s)| !**u && **s).count();
    Paired {
        n,
        unit: unit.iter().take(n).filter(|x| **x).count(),
        base: base.iter().take(n).filter(|x| **x).count(),
        b,
        c,
        p: sign_p(b, c),
    }
}

fn ln_fact(k: usize) -> f64 {
    (2..=k).map(|i| f64::from(u32::try_from(i).unwrap_or(u32::MAX)).ln()).sum()
}

/// P(X >= b) for X ~ Bin(b + c, 1/2); 1 when there are no discordant pairs.
#[must_use]
pub fn sign_p(b: usize, c: usize) -> f64 {
    let m = b + c;
    if m == 0 {
        return 1.0;
    }
    let lm = ln_fact(m);
    let half = f64::from(u32::try_from(m).unwrap_or(u32::MAX)) * 0.5_f64.ln();
    (b..=m).map(|k| (lm - ln_fact(k) - ln_fact(m - k) + half).exp()).sum::<f64>().min(1.0)
}

/// A per-scope verdict of the accept rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScopeVerdict {
    pub scope: String,
    /// accept, candidate, reject or abstain (a scope the unit never trained on).
    pub verdict: String,
    pub reason: String,
}

/// The accept rule for one scope. `near`: the paired test on held-out cases of a scope the
/// unit trained on (None for a scope it never saw). `gated` and `best_ref`: the gated
/// pathway's and the best reference's held-out rank-first share in this scope (3b).
#[must_use]
pub fn judge(scope: &str, near: Option<&Paired>, gated: f64, best_ref: f64) -> ScopeVerdict {
    let v = |verdict: &str, reason: String| ScopeVerdict { scope: scope.to_owned(), verdict: verdict.to_owned(), reason };
    if gated + 1e-9 < best_ref {
        return v("reject", format!("3b: gated {gated:.3} below best reference {best_ref:.3}"));
    }
    let Some(p) = near else {
        return v("abstain", "never trained on this scope; 3b holds, the unit stays off here".into());
    };
    if p.b <= p.c {
        return v("reject", format!("3a: no gain near seen cases (unit only {}, base only {})", p.b, p.c));
    }
    if p.n < MIN_NEAR {
        return v("candidate", format!("3a: sample too small ({} cases, {MIN_NEAR} needed)", p.n));
    }
    if p.p >= ALPHA {
        return v("candidate", format!("3a: gain not significant (p {:.4})", p.p));
    }
    v("accept", format!("3a: unit only {} vs base only {}, p {:.4}; 3b holds", p.b, p.c, p.p))
}

/// The local gate: predicts the benefit of the unit's order over the base's for one set.
#[derive(Clone, Debug, PartialEq)]
pub struct Gate {
    pub w: Vec<f64>,
    pub fitted_on: usize,
}
impl Gate {
    /// Gate features of one set: unit margin, base margin, agreement on the top document, 1.
    #[must_use]
    pub fn features(unit: &[f64], base: &[f64]) -> Vec<f64> {
        let top2 = |s: &[f64]| -> (usize, f64) {
            let mut idx: Vec<usize> = (0..s.len()).collect();
            idx.sort_by(|a, b| s.get(*b).partial_cmp(&s.get(*a)).unwrap_or(std::cmp::Ordering::Equal));
            let a = idx.first().copied().unwrap_or(0);
            let margin = s.get(a).copied().unwrap_or(0.0) - idx.get(1).and_then(|j| s.get(*j)).copied().unwrap_or(0.0);
            (a, margin)
        };
        let (ua, um) = top2(unit);
        let (ba, bm) = top2(base);
        vec![um, bm, if ua == ba { 1.0 } else { 0.0 }, 1.0]
    }
    /// Fit on (features, measured benefit) pairs.
    #[must_use]
    pub fn fit(rows: &[(Vec<f64>, f64)]) -> Self {
        let mut r = Rls::new(4, 1.0);
        for (x, y) in rows {
            r.update(x, *y);
        }
        Self { w: r.w, fitted_on: rows.len() }
    }
    #[must_use]
    pub fn predict(&self, x: &[f64]) -> f64 {
        self.w.iter().zip(x).map(|(a, b)| a * b).sum()
    }
    /// Use the unit's order for this set?
    #[must_use]
    pub fn open(&self, x: &[f64]) -> bool {
        self.predict(x) > 0.0
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("features", Value::String("unit margin, base margin, top agreement, 1".into())),
            ("w", Value::Array(self.w.iter().map(|x| Value::Number(format!("{x}"))).collect())),
            ("fitted_on", num(self.fitted_on)),
        ])
    }
    /// # Errors
    /// A malformed gate file.
    pub fn from_value(v: &Value) -> Result<Self, String> {
        let w: Vec<f64> = v
            .get("w")
            .and_then(Value::array)
            .map_err(|e| e.to_string())?
            .iter()
            .filter_map(|x| if let Value::Number(s) = x { s.parse().ok() } else { None })
            .collect();
        if w.len() != 4 {
            return Err("gate: 4 weights expected".into());
        }
        Ok(Self { w, fitted_on: usize::try_from(v.get("fitted_on").and_then(Value::integer).unwrap_or(0)).unwrap_or(0) })
    }
}

/// A unit's life in use: decayed benefit, uses, times, status.
#[derive(Clone, Debug, PartialEq)]
pub struct Life {
    pub benefit: f64,
    pub uses: u64,
    pub last_use_ms: u64,
    pub decayed_ms: u64,
    pub status: String,
}
impl Life {
    #[must_use]
    pub fn new(now_ms: u64, benefit: f64, status: &str) -> Self {
        Self { benefit, uses: 0, last_use_ms: now_ms, decayed_ms: now_ms, status: status.to_owned() }
    }
    fn days(from: u64, to: u64) -> f64 {
        f64::from(u32::try_from(to.saturating_sub(from) / 1000).unwrap_or(u32::MAX)) * 1000.0 / DAY_MS
    }
    /// Decay the benefit to `now`; retire when the rule says so. Returns true if retired now.
    pub fn decay(&mut self, now_ms: u64) -> bool {
        self.benefit *= 0.5_f64.powf(Self::days(self.decayed_ms, now_ms) / HALF_LIFE_DAYS);
        self.decayed_ms = self.decayed_ms.max(now_ms);
        if self.status == "retired" {
            return false;
        }
        let bad = self.uses >= MIN_USES && self.benefit < RETIRE_BELOW;
        let idle = Self::days(self.last_use_ms, now_ms) > IDLE_DAYS;
        if bad || idle {
            self.status = "retired".into();
            return true;
        }
        false
    }
    /// One measured benefit in use (unit right minus base right on a verified outcome).
    pub fn observe(&mut self, benefit: f64, now_ms: u64) -> bool {
        self.uses += 1;
        self.last_use_ms = now_ms;
        let retired = self.decay(now_ms);
        self.benefit = 0.9 * self.benefit + 0.1 * benefit;
        retired || self.decay(now_ms)
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("benefit", Value::Number(format!("{}", self.benefit))),
            ("uses", num(self.uses)),
            ("last_use_ms", num(self.last_use_ms)),
            ("decayed_ms", num(self.decayed_ms)),
            ("status", Value::String(self.status.clone())),
            ("half_life_days", Value::Number(format!("{HALF_LIFE_DAYS}"))),
            ("retire_below", Value::Number(format!("{RETIRE_BELOW}"))),
            ("idle_days", Value::Number(format!("{IDLE_DAYS}"))),
        ])
    }
    /// # Errors
    /// A malformed life file.
    pub fn from_value(v: &Value) -> Result<Self, String> {
        let int = |k: &str| v.get(k).and_then(Value::integer).map_err(|e| format!("{k}: {e}"));
        let benefit = match v.get("benefit") {
            Ok(Value::Number(s)) => s.parse().map_err(|_| "benefit".to_owned())?,
            _ => return Err("benefit".into()),
        };
        Ok(Self {
            benefit,
            uses: int("uses")?,
            last_use_ms: int("last_use_ms")?,
            decayed_ms: int("decayed_ms")?,
            status: v.get("status").and_then(Value::string).map_err(|e| e.to_string())?.to_owned(),
        })
    }
}

/// The readout as the reranker uses it: only a unit Luis switched on (registry status
/// `active`), only in its accepted scopes, and its order only where the gate opens.
#[derive(Clone, Debug)]
pub struct Active {
    pub unit: String,
    pub w: Vec<f64>,
    pub gate: Gate,
    pub scopes: Vec<String>,
}
impl Active {
    /// The switched-on readout unit under `home`, if any (none means no change anywhere).
    #[must_use]
    pub fn open(home: &std::path::Path) -> Option<Self> {
        let units = crate::Units::open(home);
        let card = units.cards.iter().find(|c| c.status == "active" && c.target.ends_with("#final_norm"))?;
        let dir = units.unit_dir(&card.id);
        let (w, _, _) = crate::UnitWeights::decode_q8(&std::fs::read(dir.join("readout.q8")).ok()?).ok()?;
        let gate = Gate::from_value(&nmjson::parse(&std::fs::read(dir.join("gate.json")).ok()?).ok()?).ok()?;
        let v = nmjson::parse(&std::fs::read(dir.join("verdict.json")).ok()?).ok()?;
        let scopes =
            v.get("accepted_scopes").and_then(Value::array).ok()?.iter().filter_map(|s| s.string().ok().map(str::to_owned)).collect();
        Some(Self { unit: card.id.clone(), w: w.into_iter().map(f64::from).collect(), gate, scopes })
    }
    /// Does the unit serve this question (the scope gate)?
    #[must_use]
    pub fn serves(&self, question: &str) -> bool {
        self.scopes.iter().any(|s| s == scope_of(question))
    }
    /// The readout value of one (hidden state, base score) pair.
    #[must_use]
    pub fn score(&self, hidden: &[f32], base: f32) -> f64 {
        hidden.iter().map(|x| f64::from(*x)).chain([f64::from(base), 1.0]).zip(&self.w).map(|(a, b)| a * b).sum()
    }
    /// The order of one set: the unit's scores where the gate opens, else the base's.
    #[must_use]
    pub fn order(&self, unit: &[f64], base: &[f64]) -> Vec<f64> {
        if self.gate.open(&Gate::features(unit, base)) { unit.to_vec() } else { base.to_vec() }
    }
}

fn num(v: impl ToString) -> Value {
    Value::Number(v.to_string())
}
fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

/// Self-check of `readout`: scopes, the sign test, the accept rule (a shuffled-like unit is
/// rejected by 3a), the gate learns a benefit sign, and decay retires a losing unit.
pub(crate) fn check() -> Result<(), String> {
    let f = |m: &str| Err(format!("nmlearn.readout: {m}"));
    if scope_of("Where is the function run defined?") != "reranker:where-fn"
        || scope_of("What value is the constant X set to?") != "reranker:value"
    {
        return f("scope_of");
    }
    if (sign_p(10, 0) - 1.0 / 1024.0).abs() > 1e-9 || (sign_p(0, 0) - 1.0).abs() > 1e-12 || sign_p(5, 5) < 0.5 {
        return f("sign test");
    }
    let good = paired(&[true; 40], &[false; 40]);
    let noise = paired(&[false; 40], &[true; 40]);
    let small = paired(&[true; 8], &[false; 8]);
    let ok = judge("s", Some(&good), 1.0, 0.0).verdict == "accept"
        && judge("s", Some(&noise), 0.5, 0.5).verdict == "reject"
        && judge("s", Some(&small), 1.0, 0.0).verdict == "candidate"
        && judge("s", Some(&good), 0.2, 0.5).verdict == "reject"
        && judge("s", None, 0.5, 0.5).verdict == "abstain";
    if !ok {
        return f("accept rule");
    }
    let rows: Vec<(Vec<f64>, f64)> =
        (0..20).map(|i| if i % 2 == 0 { (vec![2.0, 0.1, 0.0, 1.0], 1.0) } else { (vec![0.1, 2.0, 1.0, 1.0], -1.0) }).collect();
    let g = Gate::fit(&rows);
    if !g.open(&[2.0, 0.1, 0.0, 1.0]) || g.open(&[0.1, 2.0, 1.0, 1.0]) || Gate::from_value(&g.to_value()).map(|x| x.w.len()) != Ok(4) {
        return f("gate");
    }
    let mut life = Life::new(0, 0.0, "accepted");
    let mut retired = false;
    for i in 0..30_u64 {
        retired |= life.observe(-1.0, i * 1000);
    }
    let mut idle = Life::new(0, 0.5, "accepted");
    let idled = idle.decay(31 * 86_400_000);
    if !retired || life.status != "retired" || !idled || idle.benefit > 0.25 || Life::from_value(&life.to_value()).map(|l| l.uses) != Ok(30)
    {
        return f("decay and retire");
    }
    Ok(())
}
