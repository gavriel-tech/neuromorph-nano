//! Consolidation (guide 7, 17.3; PLAN 4c): at task end, a checked outcome becomes one
//! episode built from the emitted answer and the task's working notes; everything else in
//! the task's working scope is dropped (temporary and persistent purposes kept apart).
use crate::episodic::Episode;
use crate::working::Working;
use nmtype::Packet;

/// Turn a task's emitted packet and working notes into an episode; the scope is dropped.
///
/// No code is stored: every citation goes through `refer`, which turns `path:line text` or
/// `path:line lines a..b` into its stored shape (place, line range and content hash, see
/// `nmsubs::Substrate::cite_ref`). The answer keeps its first line (the reason) and its cited
/// places in the same shape; other lines of the emitted text are not kept.
#[must_use]
pub fn consolidate(
    working: &mut Working,
    task_scope: &str,
    query: &str,
    emitted: &Packet,
    version: u64,
    refer: &dyn Fn(&str) -> String,
) -> Episode {
    let notes = working.drop_scope(task_scope);
    let mut evidence: Vec<String> = Vec::new();
    for e in emitted.get_evidence() {
        let r = if is_cite(e) { refer(e) } else { e.clone() };
        if !evidence.contains(&r) {
            evidence.push(r);
        }
    }
    evidence.extend(notes.into_iter().map(|n| format!("note: {n}")));
    let mut lines = emitted.text().lines();
    let mut answer = lines.next().unwrap_or("").to_owned();
    for l in lines.filter(|l| is_cite(l)) {
        answer.push('\n');
        answer.push_str(&refer(l));
    }
    Episode::new("neuromorph", &emitted.provenance().origin, query, &answer, evidence, version)
}

/// A citation line: `path:<digits>`, optionally followed by a space and more.
fn is_cite(line: &str) -> bool {
    let loc = line.split_once(' ').map_or(line, |(a, _)| a);
    loc.rsplit_once(':').is_some_and(|(p, n)| !p.is_empty() && !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()))
}

/// Self-check of the `consolidation` sub-component: the episode carries the evidence and the
/// notes, and the working scope is gone afterwards.
pub(crate) fn check() -> Result<(), String> {
    use nmtype::Provenance;
    let mut w = Working::new();
    w.note("task:1", "8 tentacles reported");
    let p = Packet::new(
        "pack",
        "answer:q",
        "answered\na.rs:3 pub struct X",
        Provenance::new("unit:nmbrain.stack#b0", "nmbrain.stack#b0", 1, 0),
    )
    .evidence(vec!["a.rs:3 pub struct X".into()]);
    // The stored shape keeps the place and a hash, never the cited code.
    let refer = |c: &str| format!("{} lines 3..3 h:0123456789abcdef", c.split_once(' ').map_or(c, |x| x.0));
    let e = consolidate(&mut w, "task:1", "Where is X?", &p, 1, &refer);
    if e.evidence.len() != 2
        || !w.notes("task:1").is_empty()
        || e.answer != "answered\na.rs:3 lines 3..3 h:0123456789abcdef"
        || e.evidence.iter().any(|x| x.contains("pub struct"))
    {
        return Err(format!("nmmem.consolidation: episode fields or scope {:?} {:?}", e.answer, e.evidence));
    }
    Ok(())
}
