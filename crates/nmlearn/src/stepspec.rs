//! Step Checks (PLAN 4c `stepspec`; guide 10.5; Comprehensive Vision 9.4): a scope (nine
//! levels, neuron to Neuromorph), a rung (S0 static, S1 numeric), the metrics and baseline,
//! a budget, and the report with eight criteria and a verdict. Two specs whose scopes overlap
//! cannot both be accepted in one version. Shape from `nrntrn/src/spec.rs`, the scopes from
//! the mini's seven-level decomposition.
use crate::unit::{n, obj, s};
use nmjson::Value;
use nmtype::Budget;
use std::collections::BTreeMap;

/// What a Step Check may change.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Scope {
    Neuron(String, usize),
    Cluster(String),
    Component(String),
    Structure(String),
    Slot(String),
    Drone(String),
    Tentacle(String),
    Brain,
    Neuromorph,
}
impl Scope {
    /// The component or structure the scope lives in (a cluster id has no component).
    fn owner(&self) -> Option<&str> {
        match self {
            Self::Neuron(c, _) | Self::Component(c) | Self::Structure(c) | Self::Drone(c) | Self::Tentacle(c) => Some(c),
            Self::Cluster(c) | Self::Slot(c) => Some(c),
            Self::Brain | Self::Neuromorph => None,
        }
    }
    /// Do two scopes overlap? Equal scopes do; a Neuromorph scope overlaps everything; the
    /// Brain overlaps every slot; and two scopes on the same owner (a neuron, cluster,
    /// component or structure of one component) overlap.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        if self == other || matches!(self, Self::Neuromorph) || matches!(other, Self::Neuromorph) {
            return true;
        }
        if matches!((self, other), (Self::Brain, Self::Slot(_)) | (Self::Slot(_), Self::Brain)) {
            return true;
        }
        match (self.owner(), other.owner()) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let (level, id, index) = match self {
            Self::Neuron(c, i) => ("neuron", c.clone(), Some(*i)),
            Self::Cluster(c) => ("cluster", c.clone(), None),
            Self::Component(c) => ("component", c.clone(), None),
            Self::Structure(c) => ("structure", c.clone(), None),
            Self::Slot(c) => ("slot", c.clone(), None),
            Self::Drone(c) => ("drone", c.clone(), None),
            Self::Tentacle(c) => ("tentacle", c.clone(), None),
            Self::Brain => ("brain", String::new(), None),
            Self::Neuromorph => ("neuromorph", String::new(), None),
        };
        obj(vec![("level", s(level)), ("id", s(&id)), ("index", index.map_or(Value::Null, n))])
    }
}

/// S0 is the static rung (shapes, hashes, scope), S1 the numeric one (scores on a shard).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rung {
    S0,
    S1,
}

/// One Step Check as proposed, before it runs.
#[derive(Clone, Debug, PartialEq)]
pub struct StepCheckSpec {
    pub scope: Scope,
    pub rung: Rung,
    pub metrics: Vec<String>,
    pub baseline: Option<String>,
    pub budget: Budget,
    /// Scopes already open in this version; the spec is refused if it overlaps one.
    pub overlap: Vec<Scope>,
}
impl StepCheckSpec {
    /// The scopes in `overlap` this spec collides with.
    #[must_use]
    pub fn collisions(&self) -> Vec<&Scope> {
        self.overlap.iter().filter(|o| self.scope.overlaps(o)).collect()
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("scope", self.scope.to_value()),
            ("rung", s(if self.rung == Rung::S0 { "S0" } else { "S1" })),
            ("metrics", Value::Array(self.metrics.iter().map(|m| s(m)).collect())),
            ("baseline", self.baseline.as_deref().map_or(Value::Null, s)),
            ("budget", self.budget.to_value()),
        ])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Accept,
    Reject,
    Candidate,
}
impl Verdict {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Accept => "Accept",
            Self::Reject => "Reject",
            Self::Candidate => "Candidate",
        }
    }
}

/// The eight criteria of a Step Check and its verdict.
#[derive(Clone, Debug, PartialEq)]
pub struct StepReport {
    /// Score with the change minus without (positive is better).
    pub improvement: f32,
    /// Share of held-out items that got worse (0 when none).
    pub regression: f32,
    pub integrity: bool,
    pub compat: bool,
    /// Share of fit steps with finite losses.
    pub stability: f32,
    /// Only the scope changed.
    pub alignment: bool,
    pub cost_ms: u32,
    pub verdict: Verdict,
    pub detail: BTreeMap<String, Value>,
}
impl StepReport {
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut m: BTreeMap<String, Value> = [
            ("improvement".to_owned(), n(self.improvement)),
            ("regression".to_owned(), n(self.regression)),
            ("integrity".to_owned(), Value::Bool(self.integrity)),
            ("compat".to_owned(), Value::Bool(self.compat)),
            ("stability".to_owned(), n(self.stability)),
            ("alignment".to_owned(), Value::Bool(self.alignment)),
            ("cost_ms".to_owned(), n(self.cost_ms)),
            ("verdict".to_owned(), s(self.verdict.name())),
        ]
        .into_iter()
        .collect();
        m.insert("detail".into(), Value::Object(self.detail.clone()));
        Value::Object(m)
    }
}

fn f(v: &Value, k: &str) -> f32 {
    v.get(k).ok().and_then(crate::unit::float_of).unwrap_or(0.0)
}
fn b(v: &Value, k: &str) -> bool {
    matches!(v.get(k), Ok(Value::Bool(true)))
}

/// The verdict rule: Accept when the score improved and nothing regressed with every boolean
/// true; Candidate when the score is unchanged with every boolean true; Reject otherwise.
#[must_use]
#[expect(clippy::too_many_arguments, reason = "the seven criteria that decide a verdict")]
pub fn verdict(
    improvement: f32,
    regression: f32,
    integrity: bool,
    compat: bool,
    stability: f32,
    alignment: bool,
    within_budget: bool,
) -> Verdict {
    let sound = integrity && compat && alignment && within_budget && stability >= 1.0;
    if sound && improvement > 0.0 && regression == 0.0 {
        Verdict::Accept
    } else if sound && improvement == 0.0 && regression == 0.0 {
        Verdict::Candidate
    } else {
        Verdict::Reject
    }
}

/// Compare `before` and `after` (`{score, loss, ranks: [..]}` each) with the check's
/// booleans (`{integrity, compat, alignment, stability, cost_ms}` in `after`) under `spec`.
/// Regression is the share of items whose rank got worse; items are matched by position.
#[must_use]
pub fn compare(spec: &StepCheckSpec, before: &Value, after: &Value) -> StepReport {
    let ranks = |v: &Value| {
        v.get("ranks").and_then(Value::array).map(|a| a.iter().filter_map(|x| x.integer().ok()).collect::<Vec<_>>()).unwrap_or_default()
    };
    let (rb, ra) = (ranks(before), ranks(after));
    let worse = rb.iter().zip(&ra).filter(|(x, y)| y > x).count();
    #[expect(clippy::cast_precision_loss, reason = "item counts are small")]
    let regression = if rb.is_empty() { 0.0 } else { worse as f32 / rb.len() as f32 };
    let improvement = f(after, "score") - f(before, "score");
    let cost = after.get("cost_ms").and_then(Value::integer).ok().and_then(|x| u32::try_from(x).ok()).unwrap_or(0);
    let (integrity, compat, alignment, stability) =
        (b(after, "integrity"), b(after, "compat"), b(after, "alignment"), f(after, "stability"));
    let within = cost <= spec.budget.ms && spec.collisions().is_empty();
    let v = verdict(improvement, regression, integrity, compat, stability, alignment, within);
    let detail: BTreeMap<String, Value> = [
        ("score_before".to_owned(), n(f(before, "score"))),
        ("score_after".to_owned(), n(f(after, "score"))),
        ("loss_before".to_owned(), n(f(before, "loss"))),
        ("loss_after".to_owned(), n(f(after, "loss"))),
        ("items_worse".to_owned(), n(worse)),
        ("items".to_owned(), n(rb.len())),
        ("within_budget".to_owned(), Value::Bool(within)),
        ("scope".to_owned(), spec.scope.to_value()),
    ]
    .into_iter()
    .collect();
    StepReport { improvement, regression, integrity, compat, stability, alignment, cost_ms: cost, verdict: v, detail }
}

/// Promotion policy v2 (redesigned from EAMM V79.2, where policy v1 promoted a unit trained on
/// shuffled outcomes): the unit's held-out score must beat the BEST of the references, not
/// only the unit-free base. Accept above the best with no regression and sound booleans;
/// Candidate at a tie with the best; Reject otherwise, and whenever `compare` found the
/// booleans unsound.
#[must_use]
pub fn promote(report: &StepReport, score: f32, references: &[(String, f32)]) -> Verdict {
    let sound = report.integrity
        && report.compat
        && report.alignment
        && report.stability >= 1.0
        && report.detail.get("within_budget") == Some(&Value::Bool(true));
    let best = references.iter().map(|r| r.1).fold(f32::NEG_INFINITY, f32::max);
    if !sound {
        Verdict::Reject
    } else if score > best && report.regression == 0.0 {
        Verdict::Accept
    } else if (score - best).abs() < f32::EPSILON && report.regression == 0.0 {
        Verdict::Candidate
    } else {
        Verdict::Reject
    }
}

/// Self-check of `stepspec`: overlaps as defined; an improved shard with sound booleans is
/// Accept, unchanged is Candidate, a regression or a false boolean is Reject.
pub(crate) fn check() -> Result<(), String> {
    let a = Scope::Neuron("c#1".into(), 3);
    if !a.overlaps(&Scope::Component("c#1".into()))
        || a.overlaps(&Scope::Component("c#2".into()))
        || !Scope::Brain.overlaps(&Scope::Slot("writer".into()))
        || !Scope::Neuromorph.overlaps(&Scope::Cluster("x".into()))
    {
        return Err("nmlearn.stepspec: overlap rule".into());
    }
    let spec = StepCheckSpec {
        scope: Scope::Component("c#1".into()),
        rung: Rung::S1,
        metrics: vec!["rank-first".into()],
        baseline: Some("base".into()),
        budget: Budget { tokens: 0, ms: 60_000, depth: 0, units: 1 },
        overlap: Vec::new(),
    };
    let mk = |score: f32, ranks: &[i64], ok: bool| -> Value {
        obj(vec![
            ("score", n(score)),
            ("loss", n(1.0)),
            ("ranks", Value::Array(ranks.iter().map(n).collect())),
            ("integrity", Value::Bool(ok)),
            ("compat", Value::Bool(ok)),
            ("alignment", Value::Bool(ok)),
            ("stability", n(1.0)),
            ("cost_ms", n(100)),
        ])
    };
    let r = compare(&spec, &mk(0.5, &[1, 0], true), &mk(0.75, &[0, 0], true));
    if r.verdict != Verdict::Accept || r.regression != 0.0 {
        return Err(format!("nmlearn.stepspec: accept {:?}", r.to_value().encode()));
    }
    if compare(&spec, &mk(0.5, &[1, 0], true), &mk(0.5, &[1, 0], true)).verdict != Verdict::Candidate {
        return Err("nmlearn.stepspec: candidate".into());
    }
    // Policy v2: beating the base but not the best reference is not a promotion.
    let refs = |control: f32| vec![("base".to_owned(), 0.5), ("control".to_owned(), control), ("prior".to_owned(), 0.25)];
    if promote(&r, 0.75, &refs(0.875)) != Verdict::Reject
        || promote(&r, 0.75, &refs(0.5)) != Verdict::Accept
        || promote(&r, 0.75, &refs(0.75)) != Verdict::Candidate
    {
        return Err("nmlearn.stepspec: promotion policy v2".into());
    }
    if compare(&spec, &mk(0.5, &[0, 0], true), &mk(0.75, &[1, 0], true)).verdict != Verdict::Reject
        || compare(&spec, &mk(0.5, &[0], true), &mk(0.75, &[0], false)).verdict != Verdict::Reject
    {
        return Err("nmlearn.stepspec: reject".into());
    }
    let busy = StepCheckSpec { overlap: vec![Scope::Neuron("c#1".into(), 9)], ..spec };
    if busy.collisions().is_empty() || compare(&busy, &mk(0.5, &[0], true), &mk(0.75, &[0], true)).verdict != Verdict::Reject {
        return Err("nmlearn.stepspec: an overlapping spec was accepted".into());
    }
    Ok(())
}
