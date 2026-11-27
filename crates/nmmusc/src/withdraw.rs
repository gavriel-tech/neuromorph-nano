//! Withdrawal: each pack carries a running gain (one observed gain per use, for example the
//! outcome of a question with the pack applied minus its comparable baseline). A pack is
//! withdrawn (kept on disk, no longer applied) when its running mean falls below its
//! promotion gain by more than the derived margin
//!
//! `margin = z * sqrt(se_promotion^2 + se_running^2)`,
//!
//! where `se_running` is the standard error of the observed gains (the promotion error stands
//! in for it while fewer than two uses are recorded) and `z` is `withdrawal.z` in
//! `designs/muscle.json` (a fixed test gate). The margin therefore shrinks as evidence grows.
use crate::gain::GainRecord;

/// Welford running mean and spread of observed gains.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RunningGain {
    pub n: u64,
    pub mean: f64,
    /// Sum of squared deviations.
    pub m2: f64,
}
impl RunningGain {
    /// Start from the promotion record: the running mean begins at the promoted gain with no
    /// uses recorded.
    #[must_use]
    pub const fn from_promotion(g: &GainRecord) -> Self {
        Self { n: 0, mean: g.diff, m2: 0.0 }
    }
    pub fn push(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        if self.n == 0 {
            self.mean = 0.0;
        }
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (x - self.mean);
    }
    /// Standard error of the running mean; `None` with fewer than two uses.
    #[must_use]
    pub fn se(&self) -> Option<f64> {
        (self.n >= 2).then(|| (self.m2 / (self.n - 1) as f64 / self.n as f64).sqrt())
    }
}

/// Derived margin, `None` before the first use.
#[must_use]
pub fn margin(promo: &GainRecord, run: &RunningGain, z: f64) -> Option<f64> {
    if run.n == 0 {
        return None;
    }
    let rs = run.se().unwrap_or(promo.se);
    Some(z * (promo.se.powi(2) + rs.powi(2)).sqrt())
}

/// Whether the pack must be withdrawn.
#[must_use]
pub fn should_withdraw(promo: &GainRecord, run: &RunningGain, z: f64) -> bool {
    margin(promo, run, z).is_some_and(|m| run.mean < promo.diff - m)
}

/// Weight of the newest use in the recent estimator (2026-11-13, G3 method: ACE's helpful and
/// harmful counters with a fast estimator; research file Part 12). An effective window of
/// `(2 - ALPHA) / ALPHA` uses (9 at 0.2).
pub const ALPHA: f64 = 0.2;
/// The recent rule's gate (drift test 2026-11-13: z 1.96 doubles false withdrawals after 200
/// good uses, z 3 keeps them as the whole-history rule's with a delay of 9 to 10 uses).
pub const Z_RECENT: f64 = 3.0;
/// Uses before the recent estimator may withdraw a pack.
pub const MIN_RECENT: u64 = 5;

/// Helpful and harmful counts over a pack's life (gain above zero, or not) and an exponentially
/// weighted mean and variance of its recent gains.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Recent {
    pub helpful: u64,
    pub harmful: u64,
    pub n: u64,
    pub mean: f64,
    pub var: f64,
    /// Page's reflected sum of `promotion gain / 2 - gain` over the uses (`push_against`); the
    /// change rule scales it to a log likelihood ratio (2026-11-18; 0 in older packs).
    pub cusum: f64,
}
impl Recent {
    /// One use's gain, with the change rule's sum against the promotion record.
    pub fn push_against(&mut self, promo: &GainRecord, x: f64) {
        if !x.is_finite() {
            return;
        }
        self.cusum = (self.cusum + promo.diff / 2.0 - x).max(0.0);
        self.push(x);
    }
    pub fn push(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        if x > 0.0 {
            self.helpful += 1;
        } else {
            self.harmful += 1;
        }
        if self.n == 0 {
            self.mean = x;
            self.var = 0.0;
        } else {
            let d = x - self.mean;
            self.mean += ALPHA * d;
            self.var = (1.0 - ALPHA) * (self.var + ALPHA * d * d);
        }
        self.n += 1;
    }
    /// Effective number of uses behind the weighted mean.
    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "use counts")]
    pub fn eff_n(&self) -> f64 {
        ((2.0 - ALPHA) / ALPHA).min(self.n as f64).max(1.0)
    }
}

/// The recent rule: the whole-history rule, or (after `MIN_RECENT` uses) the recent mean below
/// the promotion gain by `z_recent` standard errors of the two together.
#[must_use]
pub fn should_withdraw_recent(promo: &GainRecord, run: &RunningGain, rec: &Recent, z: f64, z_recent: f64) -> bool {
    if should_withdraw(promo, run, z) {
        return true;
    }
    if rec.n < MIN_RECENT {
        return false;
    }
    let se = (rec.var / rec.eff_n()).sqrt();
    rec.mean < promo.diff - z_recent * (promo.se.powi(2) + se.powi(2)).sqrt()
}

/// One-sided upper tail of the standard normal, `P(Z > z)` (Abramowitz and Stegun 7.1.26, error
/// under 1.5e-7).
#[must_use]
pub fn normal_tail(z: f64) -> f64 {
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t * (0.254_829_592 + t * (-0.284_496_736 + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erfc = poly * (-x * x).exp();
    if z >= 0.0 { erfc / 2.0 } else { 1.0 - erfc / 2.0 }
}

/// Per-use variance of a pack's gain: the promotion's (its standard error times the square root
/// of its count, `promo_n - 1` degrees of freedom) pooled with the observed uses'.
#[must_use]
#[expect(clippy::cast_precision_loss, reason = "use counts")]
pub fn use_variance(promo: &GainRecord, run: &RunningGain) -> f64 {
    let k0 = promo.with_n.max(2) as f64 - 1.0;
    let prior = promo.se.powi(2) * promo.with_n.max(1) as f64;
    let df = run.n.saturating_sub(1) as f64;
    (k0 * prior + run.m2.max(0.0)) / (k0 + df)
}

/// The change rule (2026-11-18, T-042; research protocol in the build plan, `## Muscle memory and
/// G3`): a sequential test of "the pack still gains its promotion gain" against "it gains nothing",
/// with Page's CUSUM (the change-point test that is optimal for the delay at a given false-alarm
/// rate) so a long good history never hides a drop. The log likelihood ratio of the latest run of
/// uses is `diff / s^2 * cusum` (Gaussian gains of per-use variance `s^2`); the pack is withdrawn
/// when it exceeds `ln(n (n + 1) / alpha)` after `n` uses, with `alpha = P(Z > z)`. Each start
/// `k` of a run has `P(ever crossing ln(k (k + 1) / alpha)) <= alpha / (k (k + 1))` (Ville), and the
/// threshold at `n >= k` is higher, so a pack that still gains is withdrawn falsely with
/// probability at most `alpha` over its whole life, with no fixed window or count.
#[must_use]
#[expect(clippy::cast_precision_loss, reason = "use counts")]
pub fn should_withdraw_change(promo: &GainRecord, run: &RunningGain, rec: &Recent, z: f64) -> bool {
    if rec.n == 0 || promo.diff <= 0.0 {
        return false;
    }
    let s2 = use_variance(promo, run);
    if s2 <= 0.0 || !s2.is_finite() {
        return false;
    }
    let llr = promo.diff / s2 * rec.cusum;
    let n = rec.n as f64;
    let alpha = normal_tail(z).max(f64::MIN_POSITIVE);
    llr > (n * (n + 1.0) / alpha).ln()
}

/// Section check: holding gains keep the pack; dropped gains withdraw it.
/// # Errors
/// One fault line.
pub fn check() -> std::result::Result<(), String> {
    let promo = GainRecord { with_n: 20, with_mean: 0.65, without_n: 180, without_mean: 0.5, diff: 0.15, se: 0.03 };
    let z = crate::Policy::default().withdraw_z;
    let mut hold = RunningGain::from_promotion(&promo);
    for x in [0.16, 0.12, 0.17, 0.14, 0.15, 0.13] {
        hold.push(x);
    }
    if should_withdraw(&promo, &hold, z) {
        return Err(format!("holding pack withdrawn: {hold:?}"));
    }
    let mut drop = RunningGain::from_promotion(&promo);
    for x in [0.01, -0.02, 0.0, 0.02, -0.01, 0.01] {
        drop.push(x);
    }
    if !should_withdraw(&promo, &drop, z) {
        return Err(format!("dropped pack not withdrawn: {drop:?}"));
    }
    // The recent rule: after a long good history, a drop the whole-history mean hides is seen.
    let (mut run, mut rec) = (RunningGain::from_promotion(&promo), Recent::default());
    for _ in 0..200 {
        run.push(0.15);
        rec.push(0.15);
    }
    for _ in 0..8 {
        run.push(-0.05);
        rec.push(-0.05);
    }
    // The change rule: the same drop after the same history is seen; holding gains are not.
    let (mut crun, mut crec, mut hold_rec) = (RunningGain::from_promotion(&promo), Recent::default(), Recent::default());
    let mut hold_run = RunningGain::from_promotion(&promo);
    for i in 0..200 {
        let x = if i % 2 == 0 { 0.25 } else { 0.05 };
        crun.push(x);
        crec.push_against(&promo, x);
        hold_run.push(x);
        hold_rec.push_against(&promo, x);
        if should_withdraw_change(&promo, &hold_run, &hold_rec, z) {
            return Err(format!("change rule withdrew a holding pack after {} uses", i + 1));
        }
    }
    let mut seen = None;
    for i in 0..40 {
        let x = if i % 2 == 0 { 0.1 } else { -0.1 };
        crun.push(x);
        crec.push_against(&promo, x);
        if should_withdraw_change(&promo, &crun, &crec, z) {
            seen = Some(i + 1);
            break;
        }
    }
    if seen.is_none() || (normal_tail(1.96) - 0.025).abs() > 1e-4 {
        return Err(format!("change rule: drop not seen in 40 uses ({seen:?}), tail {}", normal_tail(1.96)));
    }
    if should_withdraw(&promo, &run, z) || !should_withdraw_recent(&promo, &run, &rec, z, 3.0) || rec.helpful != 200 || rec.harmful != 8 {
        return Err(format!(
            "recent rule: whole-history {}, recent {}, counts {} {}",
            should_withdraw(&promo, &run, z),
            should_withdraw_recent(&promo, &run, &rec, z, 3.0),
            rec.helpful,
            rec.harmful
        ));
    }
    Ok(())
}
