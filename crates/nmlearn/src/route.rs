//! The signal router of continuous learning (2026-11-12, build step 1 of research file Solution 6,
//! "Keeping the layers from corrupting each other"): each kind of signal has fixed destinations
//! among the six learning layers, and a signal offered to any other layer is refused. Corruption
//! happens when one signal trains two stores that should be independent (for example a right or
//! wrong verdict taken as taste); the router makes that impossible by construction.
//!
//! Destinations (the design's table):
//! - verdict and correction (was the answer right): L2 memory, L4 read-out calibration;
//! - behaviour gain (did a recurring behaviour help, from paired outcomes): L3 muscle memory;
//! - taste (accept, reject, edit of an offered result, given as taste): L6 preferences;
//! - cost and re-read verification (reads, seconds, tokens; a cited span re-read fresh): L1
//!   decision tuning.
//!
//! L5 (mapped in-model cartridges) takes no raw signal: it is fitted from verified spans through
//! its own gate, so it is listed as a destination of verdicts only when a fit is requested.
use nmtype::{Fault, FaultKind};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmlearn.route", FaultKind::Interface, m)
}

/// The kinds of learning signal available in service.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SignalKind {
    Verdict,
    Correction,
    BehaviourGain,
    Taste,
    Cost,
    Verification,
    /// The agent went on to use a cited location (2026-11-13; research file Solution 6, L1 signal).
    AgentUse,
}

/// The six learning layers, fastest to slowest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Layer {
    L1Decision,
    L2Memory,
    L3Muscle,
    L4Readout,
    L5Mapped,
    L6Preference,
}

impl Layer {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::L1Decision => "L1 decision tuning",
            Self::L2Memory => "L2 memory",
            Self::L3Muscle => "L3 muscle memory",
            Self::L4Readout => "L4 read-out calibration",
            Self::L5Mapped => "L5 mapped cartridges",
            Self::L6Preference => "L6 preferences",
        }
    }
}

/// The layers a signal kind may train.
#[must_use]
pub fn destinations(kind: SignalKind) -> &'static [Layer] {
    match kind {
        SignalKind::Verdict | SignalKind::Correction => &[Layer::L2Memory, Layer::L4Readout, Layer::L5Mapped],
        SignalKind::BehaviourGain => &[Layer::L3Muscle],
        SignalKind::Taste => &[Layer::L6Preference],
        SignalKind::Cost | SignalKind::Verification | SignalKind::AgentUse => &[Layer::L1Decision],
    }
}

/// Admit `kind` into `layer`, or refuse it by name.
/// # Errors
/// The layer is not a destination of the signal kind.
pub fn admit(kind: SignalKind, layer: Layer) -> Result<(), Fault> {
    if destinations(kind).contains(&layer) { Ok(()) } else { Err(bad(format!("a {kind:?} signal does not train {}", layer.name()))) }
}

/// Section check: every signal reaches exactly its design destinations; a verdict never reaches
/// preferences, taste never reaches memory or read-outs, cost only decision tuning.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let all = [
        SignalKind::Verdict,
        SignalKind::Correction,
        SignalKind::BehaviourGain,
        SignalKind::Taste,
        SignalKind::Cost,
        SignalKind::Verification,
    ];
    let layers = [Layer::L1Decision, Layer::L2Memory, Layer::L3Muscle, Layer::L4Readout, Layer::L5Mapped, Layer::L6Preference];
    for k in all {
        for l in layers {
            if admit(k, l).is_ok() != destinations(k).contains(&l) {
                return Err(format!("nmlearn.route: {k:?} into {l:?}"));
            }
        }
    }
    if admit(SignalKind::Verdict, Layer::L6Preference).is_ok()
        || admit(SignalKind::Taste, Layer::L2Memory).is_ok()
        || admit(SignalKind::Cost, Layer::L4Readout).is_ok()
    {
        return Err("nmlearn.route: a crossing was admitted".into());
    }
    // Every layer but none is reached by some signal; no signal reaches more than three layers.
    if layers.iter().any(|l| !all.iter().any(|k| destinations(*k).contains(l))) || all.iter().any(|k| destinations(*k).len() > 3) {
        return Err("nmlearn.route: coverage".into());
    }
    Ok(())
}
