//! Gain gate: a chunk is promotable only with an outcome record showing that questions that
//! used it scored better than comparable ones that did not.
//!
//! First-form estimator `SplitMeans`: mean outcome of traces that contain the chunk against
//! traces that do not, with a Welch standard error; promotable when both groups have at least
//! two outcomes and the difference exceeds `z` standard errors (`designs/muscle.json`
//! `gate.z`, a fixed test gate). Statistical weakness: it is observational, not paired.
//! Questions that use a chunk may differ from those that do not (easier, other repository
//! area), so the difference can be confounded; small groups make the Welch error unstable;
//! testing many chunks against one log inflates false promotions. The proper measurement is
//! the paired replay of research test G3-a (same questions, pack applied against not); this
//! trait is where that estimator plugs in.
use crate::log::Trace;

/// Outcome record attached to a promotion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GainRecord {
    pub with_n: u64,
    pub with_mean: f64,
    pub without_n: u64,
    pub without_mean: f64,
    /// `with_mean - without_mean`.
    pub diff: f64,
    /// Standard error of `diff`.
    pub se: f64,
}
impl GainRecord {
    /// Gate: both groups have two or more outcomes and `diff > z * se`.
    #[must_use]
    pub fn promotable(&self, z: f64) -> bool {
        self.with_n >= 2 && self.without_n >= 2 && self.diff > 0.0 && self.diff > z * self.se
    }
}

/// Estimates the gain of using a chunk.
pub trait GainEstimator {
    /// `None` when the log holds no outcomes to compare.
    fn estimate(&self, traces: &[Trace], chunk: &[u32]) -> Option<GainRecord>;
}

/// First-form estimator (see module note for its weakness).
#[derive(Clone, Copy, Debug, Default)]
pub struct SplitMeans;

fn mean_var(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let m = xs.iter().sum::<f64>() / n;
    let v = if xs.len() > 1 { xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0) } else { 0.0 };
    (m, v)
}

impl GainEstimator for SplitMeans {
    fn estimate(&self, traces: &[Trace], chunk: &[u32]) -> Option<GainRecord> {
        let (mut with, mut without) = (Vec::new(), Vec::new());
        for t in traces {
            let Some(o) = t.outcome else { continue };
            let used = !chunk.is_empty() && t.symbols.windows(chunk.len()).any(|w| w == chunk);
            if used { with.push(o) } else { without.push(o) }
        }
        if with.is_empty() || without.is_empty() {
            return None;
        }
        let (mw, vw) = mean_var(&with);
        let (mo, vo) = mean_var(&without);
        let se = (vw / with.len() as f64 + vo / without.len() as f64).sqrt();
        Some(GainRecord { with_n: with.len() as u64, with_mean: mw, without_n: without.len() as u64, without_mean: mo, diff: mw - mo, se })
    }
}

/// The design's estimator (2026-11-13, research file Solution 3, MO7: "effect = outcome minus the
/// running mean for that question kind"): each trace's effect is its outcome minus the mean
/// outcome of the other traces of its kind (all other traces when it is alone in its kind); the
/// effects of traces that used the chunk are compared with those that did not, Welch error as
/// above. Removes the confound of a chunk used mostly on easier kinds of question.
pub struct KindMeans<'a> {
    pub kind_of: &'a dyn Fn(&str) -> String,
}

impl GainEstimator for KindMeans<'_> {
    fn estimate(&self, traces: &[Trace], chunk: &[u32]) -> Option<GainRecord> {
        let scored: Vec<(&Trace, f64, String)> = traces.iter().filter_map(|t| Some((t, t.outcome?, (self.kind_of)(&t.id)))).collect();
        let (mut with, mut without) = (Vec::new(), Vec::new());
        for (i, (t, o, k)) in scored.iter().enumerate() {
            let same: Vec<f64> = scored.iter().enumerate().filter(|(j, x)| *j != i && x.2 == *k).map(|(_, x)| x.1).collect();
            let others: Vec<f64> =
                if same.is_empty() { scored.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, x)| x.1).collect() } else { same };
            if others.is_empty() {
                continue;
            }
            let effect = o - others.iter().sum::<f64>() / others.len() as f64;
            let used = !chunk.is_empty() && t.symbols.windows(chunk.len()).any(|w| w == chunk);
            if used { with.push(effect) } else { without.push(effect) }
        }
        if with.is_empty() || without.is_empty() {
            return None;
        }
        let (mw, vw) = mean_var(&with);
        let (mo, vo) = mean_var(&without);
        let se = (vw / with.len() as f64 + vo / without.len() as f64).sqrt();
        Some(GainRecord { with_n: with.len() as u64, with_mean: mw, without_n: without.len() as u64, without_mean: mo, diff: mw - mo, se })
    }
}

/// Section check of `KindMeans`: a chunk used only on an easy kind of question, with no effect of
/// its own, shows a gain to `SplitMeans` and none to `KindMeans`; a chunk with a real effect within
/// each kind passes `KindMeans`.
/// # Errors
/// One fault line.
pub fn check_kinds() -> std::result::Result<(), String> {
    let mk = |i: usize, sym: u32, o: f64| Trace { id: format!("q{i}"), symbols: vec![1, sym, 2], outcome: Some(o) };
    // Kind "easy" (ids even) scores near 0.9, "hard" (odd) near 0.3; chunk [1, 7, 2] only on easy.
    let noise = |i: usize| ((i * 37 % 11) as f64 - 5.0) / 100.0;
    let confounded: Vec<Trace> = (0..40)
        .map(|i| if i % 2 == 0 { mk(i, if i % 4 == 0 { 7 } else { 8 }, 0.9 + noise(i)) } else { mk(i, 8, 0.3 + noise(i)) })
        .collect();
    let kind =
        |id: &str| if id.trim_start_matches('q').parse::<usize>().unwrap_or(0) % 2 == 0 { "easy".to_owned() } else { "hard".to_owned() };
    let km = KindMeans { kind_of: &kind };
    let split = SplitMeans.estimate(&confounded, &[1, 7, 2]).ok_or("no split estimate")?;
    let fair = km.estimate(&confounded, &[1, 7, 2]).ok_or("no kind estimate")?;
    if !split.promotable(2.0) || fair.promotable(2.0) {
        return Err(format!("nmmusc.gain.kinds: confound split {split:?} kind {fair:?}"));
    }
    // A real effect: chunk [1, 9, 2] adds 0.2 within either kind.
    let real: Vec<Trace> = (0..40)
        .map(|i| {
            let base = if i % 2 == 0 { 0.9 } else { 0.3 };
            if i % 4 < 2 { mk(i, 9, base - 0.1 + noise(i)) } else { mk(i, 8, base - 0.3 + noise(i)) }
        })
        .collect();
    let r = km.estimate(&real, &[1, 9, 2]).ok_or("no estimate on the real effect")?;
    if !r.promotable(2.0) {
        return Err(format!("nmmusc.gain.kinds: real effect not promotable {r:?}"));
    }
    Ok(())
}

/// Section check: a chunk with a planted positive effect passes; one with no effect does not.
/// # Errors
/// One fault line.
pub fn check() -> std::result::Result<(), String> {
    let good = [3u32, 9, 14, 1];
    let null = [30u32, 2, 18, 11];
    let (mut traces, users) = crate::synth::generate(200, 50, 40, crate::synth::Background::Uniform, &[(&good, 40), (&null, 40)], 11);
    let mut rng = crate::synth::Rng::new(5);
    let good_users = users.first().cloned().unwrap_or_default();
    for (i, t) in traces.iter_mut().enumerate() {
        let lift = if good_users.contains(&i) { 0.15 } else { 0.0 };
        t.outcome = Some(0.5 + lift + 0.1 * rng.normal());
    }
    let z = crate::Policy::default().gate_z;
    let g = SplitMeans.estimate(&traces, &good).ok_or("no record for good chunk")?;
    let n = SplitMeans.estimate(&traces, &null).ok_or("no record for null chunk")?;
    if !g.promotable(z) {
        return Err(format!("positive chunk not promotable: {g:?}"));
    }
    if n.promotable(z) {
        return Err(format!("null chunk promoted: {n:?}"));
    }
    Ok(())
}
