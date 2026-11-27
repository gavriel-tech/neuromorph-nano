//! L1 decision tuner (2026-11-12, build steps 4 and 5 of research file Solution 6). Learns the
//! Core Brain's second-look threshold in service.
//!
//! Signals (the design's own, research file Solution 6, L1: "re-read verification, agent use,
//! cost"; the router `crate::route` sends verification and cost to L1 only): each logged decision
//! carries the re-read verification of every chunk of its second look, so every arm is valued by
//! exact replay of the logged scores (`Decision::replay_look`), with no human label and no
//! propensity weighting; cost breaks ties. Verdicts do not train this layer (they belong to L2, L4
//! and L5); they are used only as its gate, the design's "never worse on right-first in paired
//! replay": an arm may replace the default only when, over decisions with a verdict, the cases
//! where it would move away from a choice known right are no more than the cases where it keeps a
//! choice known right that the default would leave. Fixed 2026-11-12: the first build trained the
//! tuner on verdicts, against the router.
//!
//! Policy: epsilon-greedy over a fixed set of thresholds (arms), every choice logged with its exact
//! propensity (the design's Thompson sampling replaced because its propensities are not exact).
//!
//! Also here (step 4): `combine`, the stacking rule for calibrated pieces: log-odds weighted by
//! each piece's reliability.
//!
//! `site_default` (layer 1 on the served path): the per-project default at one decision site, from
//! the verified outcomes of the site's choices on the same calls (as served and as replayed: the
//! re-read verification signal), kept only if it holds on calls it was not chosen on.
use crate::ledger::Decision;
use std::collections::BTreeMap;

/// The thresholds tried (the yes-or-no logit margin the second look needs to move first place);
/// `f32::INFINITY` never moves it.
pub const ARMS: [f32; 6] = [0.0, 0.1, 0.205, 0.35, 0.6, f32::INFINITY];
/// The default arm (CB-1's out-of-fold mean, `nmapi::decide::LOOK_MARGIN`).
pub const DEFAULT: usize = 2;
/// Exploration rate.
pub const EPS: f64 = 0.1;
/// Replayed decisions an arm needs before it may lead, and verdicts the gate needs.
pub const MIN_N: usize = 10;

/// What the ledger says about each arm.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tuner {
    /// Per arm, by replay: (verified choices, replayed decisions).
    pub value: Vec<(usize, usize)>,
    /// Per arm, decisions made with it: (summed cost in ms, count).
    pub cost: Vec<(u64, usize)>,
    /// Per arm against the default, over decisions with a verdict: (worse, better).
    pub paired: Vec<(usize, usize)>,
    /// Decisions with a verdict.
    pub verdicts: usize,
}

fn arm_of(t: f32) -> Option<usize> {
    ARMS.iter().position(|a| (a.is_infinite() && t.is_infinite()) || (a - t).abs() < 1e-6)
}

impl Tuner {
    /// Build from ledger decisions.
    #[must_use]
    pub fn from_ledger(ds: &[Decision]) -> Self {
        let k = ARMS.len();
        let (mut value, mut cost, mut paired, mut verdicts) = (vec![(0, 0); k], vec![(0, 0); k], vec![(0, 0); k], 0);
        let def = ARMS[DEFAULT];
        for d in ds {
            if let Some(a) = arm_of(d.threshold)
                && d.cost_ms > 0
                && let Some(c) = cost.get_mut(a)
            {
                c.0 += d.cost_ms;
                c.1 += 1;
            }
            for (a, v) in value.iter_mut().enumerate() {
                // Agent use (2026-11-13): the chunk the agent went on to use counts as verified.
                let verified = |c: usize| if d.agent_used == Some(true) && c == d.chosen { Some(true) } else { d.verified(c) };
                if let Some(ok) = ARMS.get(a).and_then(|t| verified(d.replay_look(*t))) {
                    v.0 += usize::from(ok);
                    v.1 += 1;
                }
            }
            // The gate: a verdict is known only for the choice actually made.
            if let Some(right) = d.correct {
                verdicts += 1;
                let base = d.replay_look(def);
                for (a, p) in paired.iter_mut().enumerate() {
                    let mine = ARMS.get(a).map_or(base, |t| d.replay_look(*t));
                    if mine == base || !right {
                        continue;
                    }
                    if base == d.chosen {
                        p.0 += 1;
                    } else if mine == d.chosen {
                        p.1 += 1;
                    }
                }
            }
        }
        Self { value, cost, paired, verdicts }
    }
    /// The replayed share of verified choices per arm (`None` without replays).
    #[must_use]
    pub fn estimate(&self, a: usize) -> Option<f64> {
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        self.value.get(a).filter(|v| v.1 > 0).map(|v| v.0 as f64 / v.1 as f64)
    }
    fn mean_cost(&self, a: usize) -> f64 {
        #[expect(clippy::cast_precision_loss, reason = "milliseconds")]
        self.cost.get(a).filter(|c| c.1 > 0).map_or(f64::MAX, |c| c.0 as f64 / c.1 as f64)
    }
    /// Whether the verdict gate lets arm `a` replace the default.
    #[must_use]
    pub fn gate(&self, a: usize) -> bool {
        a == DEFAULT || (self.verdicts >= MIN_N && self.paired.get(a).is_some_and(|(worse, better)| worse <= better))
    }
    /// The arm that leads: the best replayed verification among arms with `MIN_N` replays (ties
    /// to the lower mean cost, then the lower arm), if it beats the default and passes the gate.
    #[must_use]
    pub fn best(&self) -> usize {
        let enough = |a: usize| self.value.get(a).is_some_and(|v| v.1 >= MIN_N);
        if !enough(DEFAULT) {
            return DEFAULT;
        }
        let d = self.estimate(DEFAULT).unwrap_or(0.0);
        (0..ARMS.len())
            .filter(|a| enough(*a) && self.gate(*a))
            .max_by(|a, b| {
                self.estimate(*a)
                    .unwrap_or(0.0)
                    .total_cmp(&self.estimate(*b).unwrap_or(0.0))
                    .then(self.mean_cost(*b).total_cmp(&self.mean_cost(*a)))
                    .then(b.cmp(a))
            })
            .filter(|a| self.estimate(*a).unwrap_or(0.0) > d)
            .unwrap_or(DEFAULT)
    }
    /// Pick a threshold with draw `u` in [0, 1): (arm, threshold, propensity of that arm).
    #[must_use]
    pub fn pick(&self, u: f64) -> (usize, f32, f32) {
        let k = ARMS.len();
        let best = self.best();
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "arm index from a uniform draw"
        )]
        let a = if u < EPS { (((u / EPS) * k as f64) as usize).min(k - 1) } else { best };
        #[expect(clippy::cast_precision_loss, clippy::cast_possible_truncation, reason = "probabilities")]
        let p = (if a == best { 1.0 - EPS + EPS / k as f64 } else { EPS / k as f64 }) as f32;
        (a, ARMS.get(a).copied().unwrap_or(ARMS[DEFAULT]), p)
    }
}

/// A uniform draw in [0, 1) from a seed (xorshift64*).
#[must_use]
pub fn draw(seed: u64) -> f64 {
    let mut x = seed | 1;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    #[expect(clippy::cast_precision_loss, reason = "a uniform draw")]
    let u = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1_u64 << 53) as f64;
    u
}

/// Step 4: stack calibrated pieces: each a probability and a reliability weight in [0, 1];
/// the combined probability is the logistic of the reliability-weighted sum of log-odds.
#[must_use]
pub fn combine(pieces: &[(f64, f64)]) -> f64 {
    let z: f64 = pieces
        .iter()
        .map(|(p, w)| {
            let p = p.clamp(1e-6, 1.0 - 1e-6);
            w.clamp(0.0, 1.0) * (p / (1.0 - p)).ln()
        })
        .sum();
    1.0 / (1.0 + (-z).exp())
}

/// Layer 1 at one decision site: the default choice for this project and the rule's evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteDefault {
    /// The configured choice, or the learned one when `kept`.
    pub choice: String,
    pub kept: bool,
    /// Leave one call out: calls, mean paired gain of the choice made without that call over the
    /// configured choice on it, and the standard error.
    pub held: (usize, f64, f64),
    /// The choice's margin over each other choice, in standard errors of their paired difference
    /// over the calls holding both (0 where it does not lead): the deliberative evidence a pack's
    /// prior must exceed at the site.
    pub margins: BTreeMap<String, f64>,
}

fn paired_mean_se(d: &[f64]) -> (f64, f64) {
    if d.is_empty() {
        return (0.0, 0.0);
    }
    #[expect(clippy::cast_precision_loss, reason = "call counts")]
    let n = d.len() as f64;
    let m = d.iter().sum::<f64>() / n;
    let v = if d.len() > 1 { d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0) } else { 0.0 };
    (m, (v / n).sqrt())
}

/// The choice with the best mean paired gain over `configured` across `calls` (skipping call
/// `skip`); ties keep `configured`.
fn best_choice(calls: &[BTreeMap<String, f64>], configured: &str, skip: Option<usize>) -> String {
    let mut gains: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for (i, c) in calls.iter().enumerate() {
        if Some(i) == skip {
            continue;
        }
        let Some(base) = c.get(configured) else { continue };
        for (k, v) in c {
            gains.entry(k.as_str()).or_default().push(v - base);
        }
    }
    let mut best = (configured.to_owned(), 0.0_f64);
    for (k, g) in gains {
        let (m, _) = paired_mean_se(&g);
        if m > best.1 {
            best = (k.to_owned(), m);
        }
    }
    best.0
}

/// Layer 1's default at a site from `calls` (each: choice to verified outcome), against the
/// `configured` choice, with the gate `z` in standard errors.
#[must_use]
pub fn site_default(calls: &[BTreeMap<String, f64>], configured: &str, z: f64) -> SiteDefault {
    let mut held = Vec::new();
    for (i, c) in calls.iter().enumerate() {
        let pick = best_choice(calls, configured, Some(i));
        if let (Some(a), Some(b)) = (c.get(&pick), c.get(configured)) {
            held.push(a - b);
        }
    }
    let (hm, hse) = paired_mean_se(&held);
    let all = best_choice(calls, configured, None);
    let kept = all != configured && held.len() >= 2 && hm > 0.0 && hm > z * hse;
    let choice = if kept { all } else { configured.to_owned() };
    let mut margins = BTreeMap::new();
    let others: std::collections::BTreeSet<&String> = calls.iter().flat_map(BTreeMap::keys).filter(|k| **k != choice).collect();
    for o in others {
        let d: Vec<f64> = calls.iter().filter_map(|c| Some(c.get(&choice)? - c.get(o)?)).collect();
        let (m, se) = paired_mean_se(&d);
        let ev = if se > 0.0 { m / se } else if m > 0.0 { f64::MAX } else { 0.0 };
        margins.insert(o.clone(), ev.max(0.0));
    }
    SiteDefault { choice, kept, held: (held.len(), hm, hse), margins }
}

/// Section check of `site_default`: a choice better on every call is kept with a margin; a choice
/// better only by chance on the calls it was chosen on is not kept.
/// # Errors
/// One fault line.
pub fn check_site() -> Result<(), String> {
    let call = |a: f64, b: f64| BTreeMap::from([("nbv".to_owned(), a), ("uniform".to_owned(), b)]);
    let better: Vec<_> = (0..8_u8).map(|i| call(0.5, 0.7 + 0.01 * f64::from(i))).collect();
    let d = site_default(&better, "nbv", 1.96);
    if !d.kept || d.choice != "uniform" || d.margins.get("nbv").is_none_or(|m| *m < 1.96) {
        return Err(format!("nmlearn.tuner: a choice better on every call was not kept: {d:?}"));
    }
    let noise = [call(0.5, 0.6), call(0.6, 0.5), call(0.5, 0.55), call(0.55, 0.5)];
    let n = site_default(&noise, "nbv", 1.96);
    if n.kept || n.choice != "nbv" {
        return Err(format!("nmlearn.tuner: a choice better by chance was kept: {n:?}"));
    }
    Ok(())
}

/// Section check: propensities of all arms sum to one; with too few verdicts the default leads;
/// with enough verdicts favouring another arm it leads; a ledger replayed by its own threshold
/// keeps its choices; `combine` of one fully reliable piece is that piece, of opposite equal
/// pieces is one half.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let t = Tuner::from_ledger(&[]);
    let mut total = 0.0_f64;
    let k = ARMS.len();
    for a in 0..k {
        let p = if a == t.best() { 1.0 - EPS + EPS / k as f64 } else { EPS / k as f64 };
        total += p;
    }
    if (total - 1.0).abs() > 1e-9 || t.best() != DEFAULT || t.pick(0.5).0 != DEFAULT {
        return Err("nmlearn.tuner: default and propensities".into());
    }
    // Two-chunk looks: the reranker's first (chunk 1) and a challenger (chunk 2) ahead by `m`;
    // re-read verification says only the challenger holds the question's terms.
    let mk = |m: f32, chosen: usize, y: Option<bool>| Decision {
        look: vec![1, 2],
        look_scores: vec![0.0, m],
        look_verified: vec![false, true],
        threshold: 0.205,
        chosen,
        propensity: 0.9,
        cost_ms: 10,
        correct: y,
        ..Decision::default()
    };
    // Margin 0.15: the default (0.205) keeps chunk 1; arms 0 and 0.1 move to chunk 2.
    let ds: Vec<Decision> = (0..12).map(|_| mk(0.15, 1, None)).collect();
    let t1 = Tuner::from_ledger(&ds);
    if t1.estimate(0) != Some(1.0) || t1.estimate(DEFAULT) != Some(0.0) || t1.best() != DEFAULT {
        return Err(format!("nmlearn.tuner: without verdicts the default must stay {t1:?}"));
    }
    // Verdicts say the default's choice was wrong each time: nothing known right is left; the
    // gate opens and a verified arm leads.
    let mut ds: Vec<Decision> = (0..12).map(|_| mk(0.15, 1, Some(false))).collect();
    let t2 = Tuner::from_ledger(&ds);
    if !matches!(t2.best(), 0 | 1) || !t2.gate(1) {
        return Err(format!("nmlearn.tuner: a verified arm with an open gate did not lead {t2:?}"));
    }
    // Verdicts say the default's choice was right: moving away is worse; the gate shuts.
    for d in &mut ds {
        d.correct = Some(true);
    }
    let t3 = Tuner::from_ledger(&ds);
    if t3.best() != DEFAULT || t3.gate(0) {
        return Err(format!("nmlearn.tuner: the gate let an arm leave right answers {t3:?}"));
    }
    // Agent use: the default's chosen chunk 1, unverified by re-read, counts as verified once used.
    let used: Vec<Decision> = (0..12).map(|_| Decision { agent_used: Some(true), ..mk(0.15, 1, None) }).collect();
    if Tuner::from_ledger(&used).estimate(DEFAULT) != Some(1.0) {
        return Err("nmlearn.tuner: agent use did not count as verification".into());
    }
    if (combine(&[(0.8, 1.0)]) - 0.8).abs() > 1e-9 || (combine(&[(0.8, 1.0), (0.2, 1.0)]) - 0.5).abs() > 1e-9 {
        return Err("nmlearn.tuner: combine".into());
    }
    if !(0.0..1.0).contains(&draw(42)) {
        return Err("nmlearn.tuner: draw".into());
    }
    check_site()
}
