//! nmmusc: G3 pattern recognition and muscle memory, first form (step P8; specification
//! sections 3 and 5.3; research gap 1). Gavriel Technologies; architecture by Luis Jake
//! Gabriel III.
//!
//! Components, each usable alone:
//! - `symbol`: a behavioural message as a compact symbol (`BehaviouralSymbol`, `Codebook`);
//!   sensory items are refused here first.
//! - `log`: the behaviour log in RAM, append-only symbol sequences per question with the
//!   outcome attached when known.
//! - `miner`: recurring contiguous chunks ranked by compression gain, with boundaries; the
//!   "repeats a lot" threshold is derived from the log's own statistics.
//! - `gain`: the gain gate (outcome with the chunk against without it).
//! - `pack`: the muscle-memory pack format (self-describing header, quantized payload,
//!   reserved vector section).
//! - `store`: packs on disk under `muscle/` with an index; refuses `memory/`, `weights/`,
//!   `units/` and sensory content.
//! - `reload`: load by fingerprint, verify, hand to the G3 reasoning layer, which returns the
//!   control messages to send downward.
//! - `withdraw`: running gain per pack and the derived withdrawal rule.
//! - `measure`: first measurements M1 and M2 on synthetic logs.
//! - `g3`: G3 as its own level: configuration records, negative selection, successive halving by
//!   paired replay, provisional biases, promotion to packs, priors, withdrawal; tests G3-T, MO-T.
//!
//! All GCX levels live in RAM; only packs are on disk. Muscle memory is not the memory system,
//! not a weight set and not a cartridge.
pub mod config;
pub mod g3;
pub mod gain;
pub mod log;
pub mod measure;
pub mod miner;
pub mod pack;
pub mod priors;
pub mod reload;
pub mod sites;
pub mod store;
pub mod symbol;
mod synth;
pub mod withdraw;

pub use gain::{GainEstimator, GainRecord, KindMeans, SplitMeans};
pub use log::{BehaviourLog, Trace};
pub use miner::{Candidate, Chance, Mined, mine, mine_with};
pub use pack::Pack;
pub use reload::{ControlMessage, G3Reasoning, ReplayLayer, reload};
pub use store::MuscleStore;
pub use symbol::{BehaviouralSymbol, Codebook, DataClass, Message, MessageKind};
pub use withdraw::RunningGain;

/// A named fault of this crate: a stable code plus one detail line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fault {
    /// Stable code, for example `nmmusc-sensory-refused`.
    pub code: &'static str,
    /// What happened.
    pub detail: String,
}
impl Fault {
    #[must_use]
    pub fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Self { code, detail: detail.into() }
    }
}
impl std::fmt::Display for Fault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.detail)
    }
}
impl std::error::Error for Fault {}

/// Result type of this crate.
pub type Result<T> = std::result::Result<T, Fault>;

/// Settings read from `designs/muscle.json`; a missing file or field keeps the stated
/// fallback.
#[derive(Clone, Debug, PartialEq)]
pub struct Policy {
    /// Two-sided gate on the outcome difference, in standard errors (test gate, fixed).
    pub gate_z: f64,
    /// Withdrawal margin, in combined standard errors (test gate, fixed).
    pub withdraw_z: f64,
    /// Store folder relative to the mini root.
    pub store_dir: String,
}
impl Default for Policy {
    fn default() -> Self {
        Self { gate_z: 1.96, withdraw_z: 1.96, store_dir: "muscle".to_owned() }
    }
}
impl Policy {
    /// Read `designs/muscle.json` under the mini root; fallback to the defaults.
    #[must_use]
    pub fn load() -> Self {
        let mut p = Self::default();
        let path = nmtype::nano_root().join("designs").join("muscle.json");
        let Ok(bytes) = std::fs::read(path) else { return p };
        let Ok(v) = nmjson::parse(&bytes) else { return p };
        if let Some(z) = v.get("gate").ok().and_then(|g| pack::num(g, "z").ok()) {
            p.gate_z = z;
        }
        if let Some(z) = v.get("withdrawal").ok().and_then(|g| pack::num(g, "z").ok()) {
            p.withdraw_z = z;
        }
        if let Ok(d) = v.get("store").and_then(|s| s.get("dir")).and_then(nmjson::Value::string) {
            d.clone_into(&mut p.store_dir);
        }
        p
    }
}

/// The working folder of one section check: `tests/section-checks/nmmusc-<name>/`.
pub(crate) fn scratch_dir(name: &str) -> Result<std::path::PathBuf> {
    nmtype::check_dir(&format!("nmmusc-{name}")).map_err(|e| Fault::new("nmmusc-io", format!("check folder: {e}")))
}

/// One section check: (sub-component id, check returning ok or one fault line).
pub type Check = (&'static str, fn() -> std::result::Result<(), String>);

/// Section checks of this crate, one per sub-component.
#[must_use]
pub fn checks() -> Vec<Check> {
    vec![
        ("nmmusc.symbol", symbol::check),
        ("nmmusc.miner", miner::check),
        ("nmmusc.gain", gain::check),
        ("nmmusc.gain.kinds", gain::check_kinds),
        ("nmmusc.pack", pack::check),
        ("nmmusc.store", store::check),
        ("nmmusc.reload", reload::check),
        ("nmmusc.withdraw", withdraw::check),
        ("nmmusc.priors", priors::check),
        ("nmmusc.g3.mo-t", g3::check_mot),
        ("nmmusc.g3.g3-t", g3::check_g3t),
        ("nmmusc.sites.g3-t", sites::check),
        ("nmmusc.sites.learning", sites::check_learning),
        ("nmmusc.config", config::check),
    ]
}

/// Run every section check; the first fault line, if any.
/// # Errors
/// The first failing check, prefixed with its id.
pub fn check() -> std::result::Result<(), String> {
    for (id, f) in checks() {
        f().map_err(|e| format!("{id}: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn all_section_checks_pass() {
        for (id, f) in super::checks() {
            assert_eq!(f(), Ok(()), "{id}");
        }
    }

    #[test]
    fn design_file_parses_to_policy() {
        let path = nmtype::nano_root().join("designs").join("muscle.json");
        let bytes = std::fs::read(path).expect("designs/muscle.json");
        nmjson::parse(&bytes).expect("strict JSON");
        assert_eq!(super::Policy::load(), super::Policy::default());
    }
}
