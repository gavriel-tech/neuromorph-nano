//! Epistemic arbitration (guide 6.3, 6.4; PLAN 4d item 3): every packet is labelled from its
//! own fields and the other packets with the same key. Status: `Observed` from a `path:line`,
//! `git:` or `build:` source with evidence; `Inferred` from `crate:` sources and semantic or
//! call readers; `Hypothesized` from a slot with no agreeing reader; `Verified` when a
//! verification record exists for the key at this version; `Contradicted` when two observed
//! packets share a key but differ in text; `Unknown` otherwise. Stance: objective for
//! readers, subjective for memory, mixed when merged, unknown for slot output. Certainty is
//! absolute only for verified packets and for constraints from `rules.md`.
use nmtype::{Category, Certainty, Packet, Stance, Status};
use std::collections::BTreeSet;

/// The four distinct steps of a protective hold (guide 6.4), never merged into one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HoldStep {
    ReflexSuppression,
    Arbitration,
    StructuralResponse,
    Resolution,
}

/// A `path:line`, `git:` or `build:` source. `unit:` and `crate:` sources are not
/// observations. Fixed 2026-11-08 (context pack fix): the test was `!p.starts_with("crate")`
/// on the path, which also refused every file under `crates/`, so code lines were never
/// observed while plan and note documents were, and packs on code came back empty.
fn observed_source(src: &str) -> bool {
    src.starts_with("git:")
        || src.starts_with("build:")
        || (!src.starts_with("unit:")
            && !src.starts_with("crate:")
            && src.rsplit_once(':').is_some_and(|(p, l)| !p.is_empty() && l.parse::<usize>().is_ok()))
}

/// Labels for one packet given every packet of the task and the verified keys.
#[must_use]
pub fn label(p: &Packet, all: &[Packet], verified: &BTreeSet<String>) -> (Status, Stance, Certainty) {
    let src = p.provenance().source.as_str();
    let from_slot = src.starts_with("unit:nmbrain.slot");
    let peers: Vec<&Packet> = all.iter().filter(|o| o.key() == p.key() && o.id() != p.id()).collect();
    let observed = observed_source(src) && !p.get_evidence().is_empty() && p.kind() != "chunk";
    let status = if verified.contains(p.key()) {
        Status::Verified
    } else if observed && peers.iter().any(|o| observed_source(&o.provenance().source) && o.text() != p.text()) {
        Status::Contradicted
    } else if observed {
        Status::Observed
    } else if from_slot {
        if peers.iter().any(|o| !o.provenance().source.starts_with("unit:nmbrain.slot")) { Status::Inferred } else { Status::Hypothesized }
    } else if src.starts_with("crate:") || p.kind() == "chunk" {
        Status::Inferred
    } else {
        Status::Unknown
    };
    let stance = if from_slot {
        Stance::Unknown
    } else if src.starts_with("memory:") {
        Stance::Subjective
    } else if p.get_evidence().len() > 1 && peers.is_empty() {
        Stance::Mixed
    } else if status == Status::Unknown {
        Stance::Unknown
    } else {
        Stance::Objective
    };
    let certainty = if status == Status::Verified || (p.get_category() == Category::Constraint && src.starts_with("rules.md")) {
        Certainty::Absolute
    } else {
        Certainty::Relative
    };
    (status, stance, certainty)
}

/// Relabel every packet (new packets; the originals are untouched).
#[must_use]
pub fn arbitrate(packets: &[Packet], verified: &BTreeSet<String>) -> Vec<Packet> {
    packets
        .iter()
        .map(|p| {
            let (s, st, c) = label(p, packets, verified);
            p.clone().status(s).stance(st).certainty(c)
        })
        .collect()
}

/// An absolute constraint binds a plan only when its authority, scope and applicability
/// check out (guide 6.4): it comes from `rules.md`, it names a scope, and the scope covers
/// the task.
#[must_use]
pub fn constraint_binds(p: &Packet, task_scope: &str) -> bool {
    p.get_certainty() == Certainty::Absolute
        && p.provenance().source.starts_with("rules.md")
        && !p.get_scope().is_empty()
        && (p.get_scope() == "*" || task_scope.starts_with(p.get_scope()))
}

/// Self-check of the `arbiter` sub-component: two observed values under one key are both
/// contradicted, a lone slot answer is a hypothesis, a chunk is inferred, a verified key is
/// absolute, a line under `crates/` is observed.
pub(crate) fn check() -> Result<(), String> {
    use nmtype::Provenance;
    let mk = |key: &str, text: &str, src: &str, kind: &str| {
        Packet::new(kind, key, text, Provenance::new(src, "x", 1, 0)).evidence(vec![src.to_owned()])
    };
    let all = vec![
        mk("value:L", "L = 4", "a.rs:1", "symbol"),
        mk("value:L", "L = 9", "b.rs:1", "symbol"),
        mk("slot:q", "maybe", "unit:nmbrain.slot#writer", "finding"),
        mk("chunk:a.rs:1", "text", "a.rs:1", "chunk"),
        mk("symbol:S@a.rs", "struct S", "a.rs:3", "symbol"),
        mk("grep:crates/x/src/a.rs:7", "fn go()", "crates/x/src/a.rs:7", "grep"),
    ];
    let v = BTreeSet::from(["symbol:S@a.rs".to_owned()]);
    let out = arbitrate(&all, &v);
    let st: Vec<Status> = out.iter().map(Packet::get_status).collect();
    if st != [Status::Contradicted, Status::Contradicted, Status::Hypothesized, Status::Inferred, Status::Verified, Status::Observed] {
        return Err(format!("nmbrain.arbiter: labels {st:?}"));
    }
    if out.get(4).map(Packet::get_certainty) != Some(Certainty::Absolute) {
        return Err("nmbrain.arbiter: verified packet not absolute".into());
    }
    Ok(())
}
