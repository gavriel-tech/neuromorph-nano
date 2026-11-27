//! Retention over run-days (2026-11-12, memory revision 2, section 7; numbers set by Luis Jake
//! Gabriel III: 1-core drops after 28 run-days unused, 2-active after 14, temporary notes are
//! cleared after 7; higher tiers harder to drop).
//!
//! Model: the FSRS forgetting curve and stability updates (open spaced repetition; FSRS-6 default
//! weights; fitted on about 350 million reviews, research memory Part 13): a memory has a stability
//! S in run-days and a difficulty D (1 to 10); its retrievability after t run-days unused is
//! R = (1 + f t / S)^(-w20). Each tier has a starting stability and a window N; its threshold is
//! the retrievability a never-used memory reaches after exactly N run-days, so the owner's numbers
//! hold exactly for a memory never used again, and use extends life beyond them.
//!
//! Updates (once per run-day, the first use of the day): a useful use raises S by the FSRS success
//! rule (more when R had fallen further: the spacing effect); a plain use raises it by the "hard"
//! factor; a wrong verdict or failed re-read applies the FSRS lapse rule and raises D. Pending
//! salience multiplies the success gain.

/// FSRS-6 default weights used here (w2, w3 starting stabilities; w4 to w7 difficulty; w8 to w10
/// success; w11 to w14 lapse; w15 hard; w16 easy; w20 curve decay).
pub const W: [f64; 21] = [
    0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001, 1.8722, 0.1666, 0.796, 1.4835, 0.0614, 0.2629, 1.6483, 0.6014, 1.8729,
    0.5425, 0.0912, 0.0658, 0.1542,
];

/// Run-day windows per tier (owner's numbers).
pub const CORE_DAYS: f64 = 28.0;
pub const ACTIVE_DAYS: f64 = 14.0;
pub const TEMP_DAYS: f64 = 7.0;

/// Starting stabilities per tier: temp w2, active w3, core twice active (design value, owner's call).
pub const S0_TEMP: f64 = W[2];
pub const S0_ACTIVE: f64 = W[3];
pub const S0_CORE: f64 = 2.0 * W[3];

/// Salience weight on the success gain (design value, owner's call).
pub const SALIENCE_GAIN: f64 = 1.0;

fn factor() -> f64 {
    0.9_f64.powf(-1.0 / W[20]) - 1.0
}

/// Retrievability after `t` run-days unused at stability `s`.
#[must_use]
pub fn retrievability(t: f64, s: f64) -> f64 {
    (1.0 + factor() * t / s.max(1e-6)).powf(-W[20])
}

/// The tier a retention rule applies to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Core,
    Active,
    Temp,
}

impl Level {
    #[must_use]
    pub const fn window(self) -> f64 {
        match self {
            Self::Core => CORE_DAYS,
            Self::Active => ACTIVE_DAYS,
            Self::Temp => TEMP_DAYS,
        }
    }
    #[must_use]
    pub const fn s0(self) -> f64 {
        match self {
            Self::Core => S0_CORE,
            Self::Active => S0_ACTIVE,
            Self::Temp => S0_TEMP,
        }
    }
    /// The retrievability below which a memory leaves this tier.
    #[must_use]
    pub fn threshold(self) -> f64 {
        retrievability(self.window(), self.s0())
    }
}

/// Whether a memory at stability `s`, unused for `t` run-days, leaves `level` now.
#[must_use]
pub fn leaves(level: Level, t: u64, s: f64) -> bool {
    #[expect(clippy::cast_precision_loss, reason = "run-days")]
    let t = t as f64;
    retrievability(t, s) <= level.threshold() + 1e-12
}

/// Starting difficulty (FSRS D0 for a "good" first use).
#[must_use]
pub fn d0() -> f64 {
    (W[4] - (W[5] * 2.0).exp() + 1.0).clamp(1.0, 10.0)
}

/// What happened at a use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Recalled, nothing more known.
    Plain,
    /// Its source part reached the answer, or the answer was accepted.
    Useful,
    /// A wrong verdict or a failed re-read.
    Lapse,
}

/// New (stability, difficulty) after a use `t` run-days after the last one.
#[must_use]
pub fn update(s: f64, d: f64, t: u64, outcome: Outcome, salience: f64) -> (f64, f64) {
    #[expect(clippy::cast_precision_loss, reason = "run-days")]
    let r = retrievability(t as f64, s);
    match outcome {
        Outcome::Plain | Outcome::Useful => {
            let grade = if outcome == Outcome::Useful { 1.0 } else { W[15] };
            let gain = W[8].exp()
                * (11.0 - d)
                * s.powf(-W[9])
                * ((W[10] * (1.0 - r)).exp() - 1.0)
                * grade
                * (1.0 + SALIENCE_GAIN * salience.clamp(0.0, 1.0));
            (s * (1.0 + gain), d)
        }
        Outcome::Lapse => {
            let s2 = (W[11] * d.powf(-W[12]) * ((s + 1.0).powf(W[13]) - 1.0) * (W[14] * (1.0 - r)).exp()).min(s);
            let d2 = (d + 2.0 * W[6] * (10.0 - d) / 9.0).clamp(1.0, 10.0);
            (s2.max(0.1), d2)
        }
    }
}

/// Section check: a never-used memory leaves each tier at exactly its window (not a run-day before);
/// higher tiers are harder to drop; a use on a later run-day raises stability more than one right
/// away (spacing); a useful use raises it more than a plain one; a lapse lowers it and raises
/// difficulty.
/// # Errors
/// One fault line.
pub(crate) fn check() -> Result<(), String> {
    for (lvl, n) in [(Level::Core, 28_u64), (Level::Active, 14), (Level::Temp, 7)] {
        if leaves(lvl, n - 1, lvl.s0()) || !leaves(lvl, n, lvl.s0()) {
            return Err(format!("nmmem.retention: {lvl:?} does not leave at exactly {n} run-days"));
        }
    }
    // Harder to drop higher up: after 14 run-days unused an active memory leaves, a core one stays.
    if !leaves(Level::Active, 14, S0_ACTIVE)
        || leaves(Level::Core, 14, S0_CORE)
        || leaves(Level::Core, 27, S0_CORE)
        || S0_CORE <= S0_ACTIVE
        || S0_ACTIVE <= S0_TEMP
    {
        return Err("nmmem.retention: higher tiers are not harder to drop".into());
    }
    let d = d0();
    let (early, _) = update(S0_ACTIVE, d, 0, Outcome::Useful, 0.0);
    let (spaced, _) = update(S0_ACTIVE, d, 10, Outcome::Useful, 0.0);
    let (plain, _) = update(S0_ACTIVE, d, 10, Outcome::Plain, 0.0);
    let (lapse, d2) = update(S0_ACTIVE, d, 10, Outcome::Lapse, 0.0);
    if !(spaced > early && spaced > plain && plain > S0_ACTIVE && lapse < S0_ACTIVE && d2 > d) {
        return Err(format!(
            "nmmem.retention: updates early {early:.2} spaced {spaced:.2} plain {plain:.2} lapse {lapse:.2} d {d:.2} to {d2:.2}"
        ));
    }
    Ok(())
}
