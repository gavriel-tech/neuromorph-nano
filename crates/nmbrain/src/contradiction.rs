//! The Tension Buffer (guide 6.4; PLAN 4c): conflicting packets stay available for later
//! evaluation; nothing is resolved by the reflex that suppressed it. Reflex suppression,
//! arbitration, structural response and resolution are four separate steps, logged as such.
//! Every change of a conclusion is recorded with why and on what evidence.
use crate::arbiter::HoldStep;
use nmtype::{Packet, Provenance, Status};
use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::PathBuf;

/// One recorded change of conclusion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Change {
    pub at: u64,
    pub key: String,
    pub from: String,
    pub to: String,
    pub why: String,
    pub evidence: Vec<String>,
}

/// Conflicts held open, the steps each has gone through, and the change log.
#[derive(Debug, Default)]
pub struct TensionBuffer {
    held: BTreeMap<String, Vec<Packet>>,
    steps: BTreeMap<String, Vec<HoldStep>>,
    pub changes: Vec<Change>,
    log: Option<PathBuf>,
}
impl TensionBuffer {
    /// Hold every contradicted group (arbitration has labelled them) and answer each with a
    /// structural response: a question packet asking for a re-read. Returns the questions.
    pub fn hold(&mut self, packets: &[Packet], version: u64) -> Vec<Packet> {
        let mut questions = Vec::new();
        for p in packets.iter().filter(|p| p.get_status() == Status::Contradicted) {
            let group = self.held.entry(p.key().to_owned()).or_default();
            if !group.iter().any(|g| g.text() == p.text() && g.provenance().source == p.provenance().source) {
                group.push(p.clone());
            }
        }
        for (key, group) in &self.held {
            let steps = self.steps.entry(key.clone()).or_default();
            if steps.contains(&HoldStep::StructuralResponse) || group.len() < 2 {
                continue;
            }
            steps.extend([HoldStep::Arbitration, HoldStep::StructuralResponse]);
            let sides: Vec<String> = group.iter().map(|g| format!("{} ({})", g.text(), g.provenance().source)).collect();
            questions.push(
                Packet::new(
                    "question",
                    &format!("conflict:{key}"),
                    &format!("conflicting observations kept for {key}: {}", sides.join(" vs ")),
                    Provenance::new("unit:nmbrain.contradiction", "nmbrain.contradiction#b0", version, nmtype::now_ms()),
                )
                .status(Status::Unknown)
                .evidence(group.iter().flat_map(|g| g.get_evidence().to_vec()).collect()),
            );
        }
        questions
    }
    /// Packets held for a key.
    #[must_use]
    pub fn held(&self, key: &str) -> &[Packet] {
        self.held.get(key).map_or(&[], Vec::as_slice)
    }
    /// The steps a key has gone through.
    #[must_use]
    pub fn steps(&self, key: &str) -> &[HoldStep] {
        self.steps.get(key).map_or(&[], Vec::as_slice)
    }
    /// Resolve a held conflict on new evidence; the change is logged, the held packets stay
    /// readable in the change record.
    pub fn resolve(&mut self, key: &str, to: &str, why: &str, evidence: Vec<String>) -> Option<Change> {
        let group = self.held.remove(key)?;
        let from = group.iter().map(Packet::text).collect::<Vec<_>>().join(" | ");
        self.steps.entry(key.to_owned()).or_default().push(HoldStep::Resolution);
        let c = Change { at: nmtype::now_ms(), key: key.to_owned(), from, to: to.to_owned(), why: why.to_owned(), evidence };
        if let Some(path) = &self.log
            && let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path)
        {
            let q = |t: &str| format!("\"{}\"", t.replace('\\', "\\\\").replace('"', "\\\""));
            let _ =
                writeln!(f, "{{\"at\":{},\"key\":{},\"from\":{},\"to\":{},\"why\":{}}}", c.at, q(&c.key), q(&c.from), q(&c.to), q(&c.why));
        }
        self.changes.push(c.clone());
        Some(c)
    }
}

/// Self-check of the `contradiction` sub-component: two contradicted packets are held with
/// one structural question; resolving logs the change and records the fourth step.
pub(crate) fn check() -> Result<(), String> {
    let mk = |t: &str, s: &str| {
        Packet::new("symbol", "value:L", t, Provenance::new(s, "x", 1, 0)).status(Status::Contradicted).evidence(vec![s.to_owned()])
    };
    let mut b = TensionBuffer::default();
    let q = b.hold(&[mk("L = 4", "a.rs:1"), mk("L = 9", "b.rs:1")], 1);
    if q.len() != 1 || b.held("value:L").len() != 2 {
        return Err("nmbrain.contradiction: conflict not held with one question".into());
    }
    if !b.hold(&[mk("L = 4", "a.rs:1")], 1).is_empty() {
        return Err("nmbrain.contradiction: asked twice for one conflict".into());
    }
    let c =
        b.resolve("value:L", "L = 9", "a.rs is a test copy", vec!["a.rs:1".into()]).ok_or("nmbrain.contradiction: nothing to resolve")?;
    if !c.from.contains("L = 4") || b.steps("value:L").last() != Some(&HoldStep::Resolution) {
        return Err("nmbrain.contradiction: change record or steps".into());
    }
    Ok(())
}
