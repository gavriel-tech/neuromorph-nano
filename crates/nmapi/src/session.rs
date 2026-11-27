//! Cross-session memory for Claude Code (guide 7, 7.1, 17.2; PLAN 15.2): open reconstructs
//! the context (the organism's identity, objectives, latest episodes, the last structural
//! summary and whether it is stale); every answer is consolidated into an episode through
//! the gate; recall brings back earlier episodes as memory packets (subjective, never
//! observed); close saves the structural summary, the objectives and the identity.
use crate::forage::{MemoryState, Organism};
use nmjson::Value;
use nmmem::{Episodic, Gate, Hierarchy, HierarchyPolicy, Identity, Temp, TempPolicy, Working, WriterPolicy};
use nmtype::class::{Against, FromMaterial, FromRequest};
use nmtype::{Category, Pack, Packet, Provenance, Stance, Status, Written};
use std::path::Path;

fn s(t: &str) -> Value {
    Value::String(t.to_owned())
}
fn n(x: u64) -> Value {
    Value::Number(x.to_string())
}
fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

/// Hierarchy topic for an episode level: answers to questions are `context`, promoted temp
/// notes are `notes`, anything else keeps its level name.
fn topic_of(level: &str) -> String {
    match level {
        "neuromorph" | "brain" => "context".into(),
        "temp" => "notes".into(),
        other => other.into(),
    }
}

/// One memory sweep on the current run-day: save temp notes used on 3 run-days into episodic memory
/// and the hierarchy, clear those unused 7 run-days, then move hierarchy items by use. Returns
/// (promoted, cleared, up, down).
pub fn sweep(temp: &mut Temp, episodic: &mut Episodic, hierarchy: &mut Hierarchy) -> (Vec<String>, usize, usize, usize) {
    let r = temp.sweep(episodic).unwrap_or_default();
    for id in &r.promoted {
        if let Some(e) = episodic.get(id) {
            let (q, a) = (e.query.clone(), e.answer.clone());
            let _ = hierarchy.add(id, "notes", &q, &a, "nmmem.temp");
        }
    }
    let m = hierarchy.rebalance().unwrap_or_default();
    (r.promoted, r.cleared, m.up, m.down)
}

/// Memory state from a memory folder (`<home>/memory` by default) and `designs/memory.json`; the
/// memory's sources are read under `root`, the studied folder (anchors' part hashes, re-reads).
#[must_use]
/// With `writes` false (`serve --memory off`) the folder is only read: no adoption of the
/// older flat files, no hierarchy items added, no sweep.
pub fn memory_state(home: &Path, root: &Path, dir: &Path, writes: bool) -> MemoryState {
    let dir = dir.to_path_buf();
    let design = std::fs::read(home.join("designs/memory.json")).ok().and_then(|b| nmjson::parse(&b).ok()).unwrap_or(Value::Null);
    let mut episodic = Episodic::open(&dir);
    let (hp, tp) = (HierarchyPolicy::from_value(&design), TempPolicy::from_value(&design));
    // One writer per file: the journal's views own the tier folders and the index; the hierarchy
    // keeps `hierarchy.json`, the temporary notes `temp/notes.json`. A folder in the older mixed
    // layout is brought over first (a copy of its files kept under `migrated/`). Read only with
    // writes off.
    // Today counts as a run-day (with writes on), before anything counts a use or a lifetime.
    let mut rundays = nmmem::runday::RunDays::open(&dir);
    if writes {
        let _ = rundays.tick(nmtype::now_ms());
    }
    let (mut hierarchy, mut temp) = if writes {
        nmmem::hierarchy::adopt_flat(&dir);
        match nmmem::onewriter::migrate(&dir, hp, tp) {
            Ok((h, t, _)) => (h, t),
            Err(_) => (Hierarchy::open(&dir, hp), Temp::open(&dir, tp)),
        }
    } else {
        (Hierarchy::open(&dir, hp), Temp::open(&dir, tp))
    };
    hierarchy.set_day(rundays.count);
    temp.set_day(rundays.count);
    if writes {
        // Every episode is an item of the hierarchy (added once, by id; topic from its level).
        for e in episodic.latest(usize::MAX) {
            if hierarchy.get(&e.id).is_none() {
                let _ = hierarchy.add(&e.id, &topic_of(&e.level), &e.query, e.answer.lines().next().unwrap_or(""), &e.unit);
            }
        }
        // At every open: often-used temp notes are saved, stale ones go, then items move by use.
        let _ = sweep(&mut temp, &mut episodic, &mut hierarchy);
    }
    // Memory revision 2: open the journal (importing revision 1 when it is empty, running the
    // nightly pass on a new run-day); revision 1 keeps working beside it.
    let (mem2, mem2_fault) = if writes {
        match nmmem::mem2::Memory2::open(&dir, root) {
            Ok(mut m) => {
                // Verified and refuted outcomes the evidence ledger wrote since the last open become
                // memories through the journal (each once).
                let _ = nmmem::outcomes::import_ledger(&mut m, &dir);
                (Some(m), None)
            }
            Err(e) => (None, Some(e.to_string())),
        }
    } else {
        (None, None)
    };
    MemoryState {
        mem2,
        mem2_fault,
        rundays,
        vectors: std::collections::BTreeMap::new(),
        episodic,
        working: Working::new(),
        temp,
        hierarchy,
        gate: Gate::new(&dir, WriterPolicy::from_value(&design)),
        identity: Identity::load_or_new(&nmmem::hierarchy::core_dir(&dir), "nmapi#n0", "designs/anatomy.json"),
        design,
        dir,
        pending: Vec::new(),
    }
}

/// Session open within a token budget (characters / 4; guide 17.2, Comprehensive Vision
/// 18.3): the identity is summarized (id, origin, state, counts, last events) and the latest
/// episodes are dropped oldest first until the reply fits.
#[must_use]
pub fn open_budget(org: &Organism, tokens: usize) -> Value {
    let full = open(org);
    let get = |k: &str| full.get(k).ok().cloned().unwrap_or(Value::Null);
    let id = get("identity");
    let pick = |k: &str| id.get(k).ok().cloned().unwrap_or(Value::Null);
    let events: Vec<Value> =
        id.get("developmental_history").and_then(Value::array).map(|a| a.iter().rev().take(3).cloned().collect()).unwrap_or_default();
    let count = |k: &str| n(id.get(k).and_then(Value::array).map_or(0, <[Value]>::len) as u64);
    let summary = obj(vec![
        ("id", pick("id")),
        ("origin", pick("origin")),
        ("state", pick("state")),
        ("events", count("developmental_history")),
        ("last_events", Value::Array(events)),
        ("versions", count("version_history")),
    ]);
    let mut latest: Vec<Value> = get("latest").array().map(<[Value]>::to_vec).unwrap_or_default();
    loop {
        let v = obj(vec![
            ("identity", summary.clone()),
            ("sessions", get("sessions")),
            ("objectives", get("objectives")),
            ("episodes", get("episodes")),
            ("latest", Value::Array(latest.clone())),
            ("structural_summary_stale", get("structural_summary_stale")),
        ]);
        let size = v.encode().map_or(usize::MAX, |t| t.len() / 4);
        if size <= tokens || latest.is_empty() {
            return v;
        }
        latest.pop();
    }
}

/// Session open: record it in the organism's identity and return the reconstructed context.
#[must_use]
pub fn open(org: &Organism) -> Value {
    let version = org.nte.snapshot().version;
    let digest = org.sub().digest();
    let objectives = org.objectives.lock().map(|o| o.clone()).unwrap_or_default();
    let Ok(mut m) = org.memory.lock() else { return Value::Null };
    // The nightly pass's identity step (memory revision 2, section 12): at the first session of a
    // new run-day the identity's developmental history is compacted (each day's session lines
    // become one), so it does not grow by a line per session.
    if m.mem2.as_ref().is_some_and(|m2| matches!(m2.opened.get("nightly_pass"), Ok(Value::Bool(true)))) {
        m.identity.compact_history();
    }
    m.identity.event("session open", version);
    m.identity.state = "serving".into();
    if org.memory_on {
        let _ = m.identity.save(&nmmem::hierarchy::core_dir(&m.dir));
    }
    let summary = nmmem::structural::load_latest(&m.dir);
    let stale = summary.as_ref().is_none_or(|v| nmmem::structural::stale(v, &digest));
    obj(vec![
        ("identity", m.identity.to_value()),
        ("sessions", n(sessions(&m.identity))),
        (
            "objectives",
            Value::Array(objectives.iter().map(|(id, t, st)| obj(vec![("id", n(*id)), ("objective", s(t)), ("status", s(st))])).collect()),
        ),
        ("episodes", n(m.episodic.len() as u64)),
        (
            "latest",
            Value::Array(
                m.episodic
                    .latest(5)
                    .iter()
                    .map(|e| obj(vec![("id", s(&e.id)), ("query", s(&e.query)), ("correct", e.correct.map_or(Value::Null, Value::Bool))]))
                    .collect(),
            ),
        ),
        ("structural_summary_stale", Value::Bool(stale)),
    ])
}

/// The working-memory scope recall tags a question's surprises under, before the task that answers
/// it is known.
fn surprise_scope(question: &str) -> String {
    format!("question:{}", nmhash::sha256_bytes(question.as_bytes()).get(..16).unwrap_or_default())
}

/// Sessions opened over the identity's life: each `session open` line, and the count a compacted
/// day line carries (`<n> sessions on this run-day`).
fn sessions(id: &Identity) -> u64 {
    id.developmental_history
        .iter()
        .map(|(_, e)| {
            if e == "session open" {
                1
            } else if e.ends_with("on this run-day") {
                e.split_whitespace().next().and_then(|x| x.parse().ok()).unwrap_or(0)
            } else {
                0
            }
        })
        .sum()
}

/// Session close: save the structural summary for this substrate version, the objectives
/// and the identity. Nothing is saved with memory off.
pub fn close(org: &Organism) {
    if !org.memory_on {
        return;
    }
    settle(org);
    let sub = org.sub();
    let summary =
        obj(vec![("files", n(sub.files.len() as u64)), ("symbols", n(sub.symbols.len() as u64)), ("crates", n(sub.crates.len() as u64))]);
    let objectives = org.objectives.lock().map(|o| o.clone()).unwrap_or_default();
    if let Ok(mut m) = org.memory.lock() {
        let _ = nmmem::structural::save(&m.dir, sub.version, &sub.digest(), summary);
        let _ = nmmem::structural::save_objectives(&nmmem::hierarchy::core_dir(&m.dir), &objectives);
        let version = org.nte.snapshot().version;
        m.identity.event("session close", version);
        m.identity.state = "idle".into();
        let _ = m.identity.save(&nmmem::hierarchy::core_dir(&m.dir));
    }
}

/// The tier name of a hierarchy level (`1-core` is `core`).
fn tier_name(level: usize) -> &'static str {
    match level {
        0 => "core",
        1 => "active",
        _ => "archive",
    }
}

/// The memory cells a recall or a write touched, for the audit row: `<op>:episodes` for the
/// episodic store, `<op>:<tier>` for the tier holding each named item, `<op>:journal` when
/// memory revision 2 is open. `ids` are episode ids. The tier is the journal's (the one tier file
/// that lists the item); without the journal, the hierarchy's level.
#[must_use]
pub fn memory_cells(org: &Organism, op: &str, ids: &[String]) -> Vec<String> {
    let Ok(m) = org.memory.lock() else { return Vec::new() };
    let mut out = vec![format!("{op}:episodes")];
    let tier_of = |id: &String| -> Option<&'static str> {
        if let Some(m2) = m.mem2.as_ref() {
            return m2.book.records.get(&nmmem::mem2::episode_record_id(id)).map(|r| match r.tier {
                nmmem::record::Tier::Core => "core",
                nmmem::record::Tier::Active => "active",
                nmmem::record::Tier::Archive => "archive",
                nmmem::record::Tier::Temp => "temp",
            });
        }
        m.hierarchy.get(id).map(|i| tier_name(i.level))
    };
    let mut tiers: Vec<&'static str> = ids.iter().filter_map(tier_of).collect();
    tiers.sort_unstable();
    tiers.dedup();
    out.extend(tiers.iter().map(|t| format!("{op}:{t}")));
    if m.mem2.is_some() && (op == "write" || !ids.is_empty()) {
        out.push(format!("{op}:journal"));
    }
    out
}

/// Recall by meaning (2026-11-10), when the embedding and reranker engines are bound: the
/// embedding picks the stored questions whose nearness stands out from the rest, the reranker
/// judges each, and a floor derived from memory's own decoys decides (`meaning_floor`). This is the
/// stated fallback floor while there are too few decoys (set on the development set, offline: 7 of
/// 20 matching pairs, 1 wrong, no decoy at 0.7).
pub const MEANING_FLOOR: f32 = 0.7;
/// The embedding instruction for questions about the studied code.
pub const MEANING_TASK: &str = "Given a question about a code repository, retrieve earlier questions that ask about the same code";

/// Citations of each recalled episode read fresh from the current tree (M1).
pub const REREAD_PER_EPISODE: usize = 3;

/// The verified episodes word recall chooses for `question`, best first with their scores. Revision 1
/// (default): the rarity-weighted share of the question's key terms in each episode's question and
/// cited lines, standing out from the question's own null (`nmmem::evidence::recalled`), every one
/// that does (the null is the template guard: a phrase many stored questions share raises it).
/// Revision 2 (`Organism::recall_r2`, 2026-11-13): the same rule gates, and memory revision 2's
/// recall (BM25 over memories, links, retention; `nmmem::recall2`) orders the episodes that pass.
fn word_recall(
    org: &Organism,
    m: &crate::forage::MemoryState,
    question: &str,
    idf: &dyn Fn(&str) -> f32,
) -> Vec<(f32, nmmem::episodic::Episode)> {
    let r2 = org.recall_r2.load(std::sync::atomic::Ordering::Relaxed);
    let sub = org.sub();
    let fresh = |c: &str| sub.reread(c).map(|p| FromMaterial::new(p.text().to_owned()));
    // Only verified episodes are recalled (a positive verdict); every other answer stays on file.
    let verified = |(_, e): &(f32, &nmmem::episodic::Episode)| e.correct == Some(true);
    if r2 && let Some(m2) = m.mem2.as_ref() {
        let gated: Vec<(f32, &nmmem::episodic::Episode)> =
            m.episodic.recall_cited(question, usize::MAX, REREAD_PER_EPISODE, idf, &fresh).into_iter().filter(verified).collect();
        let hits = nmmem::recall2::recall(&m2.book, question, m2.day, &std::collections::BTreeMap::new(), m2.book.records.len());
        let place =
            |e: &nmmem::episodic::Episode| hits.iter().position(|h| h.id == nmmem::mem2::episode_record_id(&e.id)).unwrap_or(usize::MAX);
        let mut v: Vec<(usize, usize, f32, &nmmem::episodic::Episode)> =
            gated.into_iter().enumerate().map(|(i, (s, e))| (place(e), i, s, e)).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        return v.into_iter().map(|(_, _, s, e)| (s, e.clone())).collect();
    }
    m.episodic.recall_cited(question, usize::MAX, REREAD_PER_EPISODE, idf, &fresh).into_iter().filter(verified).map(|(x, e)| (x, e.clone())).collect()
}

/// What recall computes for one question, once per call: word recall's episodes and memory's
/// outcome evidence, with the state they were computed from (question, studied tree and version,
/// episodes stored, journal position).
type Shared = Option<(String, Vec<(f32, nmmem::episodic::Episode)>, nmmem::evidence::Evidence)>;
static SHARED: std::sync::Mutex<Shared> = std::sync::Mutex::new(None);

/// Word recall and outcome evidence for `question`, computed once per call: the window step
/// (`recalled_chunks`, `take` false) keeps its result, and the recall at G2 (`recall`, `take` true)
/// takes it when nothing changed in between, so the next call, after this call's write, computes
/// afresh.
fn recall_inputs(
    org: &Organism,
    m: &crate::forage::MemoryState,
    question: &str,
    idf: &dyn Fn(&str) -> f32,
    take: bool,
) -> (Vec<(f32, nmmem::episodic::Episode)>, nmmem::evidence::Evidence) {
    let sub = org.sub();
    let key = format!(
        "{question}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}",
        sub.root.display(),
        sub.version,
        m.episodic.len(),
        m.mem2.as_ref().map_or(0, |m2| m2.book.seq)
    );
    if let Ok(mut s) = SHARED.lock() {
        let hit = s.as_ref().is_some_and(|(k, _, _)| *k == key);
        if hit {
            let held = if take { s.take() } else { s.clone() };
            if let Some((_, w, e)) = held {
                return (w, e);
            }
        }
    }
    let words = word_recall(org, m, question, idf);
    let ev = outcome_evidence(org, m, question);
    if !take && let Ok(mut s) = SHARED.lock() {
        *s = Some((key, words.clone(), ev.clone()));
    }
    (words, ev)
}

/// Earlier episodes relevant to a question, as memory packets (subjective; never observed),
/// plus their citations read fresh from the current tree (M1, 2026-11-09). The fresh reads
/// are ordinary observations of the code as it is now: memory chooses where to look, the code
/// decides. Returns (memory packets, fresh reads, episodes recalled).
#[must_use]
pub fn recall(org: &Organism, question: &str) -> (Vec<Packet>, Vec<Packet>, usize) {
    let changed = org.changed_at.lock().map(|c| c.clone()).unwrap_or_default();
    let sub = org.sub();
    let idf = |t: &str| sub.idf(t);
    let Ok(mut m) = org.memory.lock() else { return (Vec::new(), Vec::new(), 0) };
    let (mut scored, ev) = recall_inputs(org, &m, question, &idf, true);
    let ids: Vec<String> = scored.iter().map(|(_, e)| e.id.clone()).collect();
    // A G3 replay (`crate::g3`) leaves no use behind.
    if !crate::g3::replaying() {
        let _ = m.hierarchy.touch(&ids);
    }
    // Recall by meaning: only with both engines bound; episodes the reranker judges relevant
    // join the word-recalled ones (the higher score kept).
    let by_meaning = meaning(org, &mut m, question);
    for (s, id) in by_meaning {
        match scored.iter_mut().find(|(_, e)| e.id == id) {
            Some(x) => x.0 = x.0.max(s),
            None => {
                if let Some(e) = m.episodic.get(&id) {
                    scored.push((s, e.clone()));
                }
            }
        }
    }
    let mut fresh: Vec<Packet> = Vec::new();
    // Re-read verification (2026-11-12, memory design): an episode none of whose cited lines
    // reads back counts a failed re-read; two demote it in the hierarchy. A cited span whose
    // content hash is found nowhere in its file is stale: the code it was about has changed.
    let mut stale: std::collections::BTreeMap<String, Vec<String>> = std::collections::BTreeMap::new();
    let mut failed: Vec<String> = Vec::new();
    for (_, e) in &scored {
        let mut any = false;
        for c in e.evidence.iter().filter(|c| !c.starts_with("note:")).take(REREAD_PER_EPISODE) {
            match sub.reread_state(c) {
                nmsubs::views::Reread::Fresh(_) => any = true,
                nmsubs::views::Reread::Stale { path, start, end } => stale
                    .entry(e.id.clone())
                    .or_default()
                    .push(format!("obsolete: stale citation {path}:{start}-{end}, its content hash no longer matches the file")),
                nmsubs::views::Reread::Gone => {}
            }
        }
        if !any {
            failed.push(e.id.clone());
        }
    }
    if !failed.is_empty() && !crate::g3::replaying() {
        let _ = m.hierarchy.mark_failed(&failed);
    }
    // A surprise (memory revision 2, section 10): a recalled memory whose cited part changed under
    // it is tagged in working memory for this question; `remember` carries the tag into the task
    // that answers it, so the note becomes a temporary note at task end.
    let scope = surprise_scope(question);
    for notes in stale.values() {
        for t in notes {
            m.working.note(&scope, t);
            m.working.tag(&scope, t, "surprise: a recalled memory's source changed under it");
        }
    }
    // Revision 2: every recalled memory is reconsolidated against its sources (moved, changed or
    // gone parts become journal events; a failed re-read is a lapse).
    // The views are written once for all of them.
    if let Some(m2) = m.mem2.as_mut() {
        m2.hold();
        for (_, e) in &scored {
            let _ = m2.reconsolidate(&e.id);
        }
        let _ = m2.release();
    }
    for (score, e) in &scored {
        // Revision 2: the recall score travels as the read's relevance (`nmbrain::synthesis`).
        for p in e
            .evidence
            .iter()
            .filter(|c| !c.starts_with("note:"))
            .take(REREAD_PER_EPISODE)
            .filter_map(|c| sub.reread(c))
            .map(|p| p.confidence(*score).evidence(vec![format!("recalled: cited by verified answer {}, read fresh", e.id)]))
        {
            if !fresh.iter().any(|x| x.key() == p.key()) {
                fresh.push(p);
            }
        }
    }
    // Verified outcomes of like questions: their places, read fresh, as recalled reads tagged with
    // the outcome; a refuted one as a memory note that names its places and serves none of them.
    let mut notes: Vec<Packet> = Vec::new();
    for piece in ev.served() {
        let Some(p) = sub.reread(&cite_of(&piece.place)) else { continue };
        let tag = match piece.kind {
            nmmem::evidence::Kind::CoCited => format!("recalled: used together with a verified answer ({})", piece.record),
            _ => format!(
                "recalled: verified answer {} to a like question (verified {}, {}), read fresh",
                piece.record,
                piece.verified,
                if piece.evidence.is_empty() { "weak" } else { piece.evidence.as_str() }
            ),
        };
        let p = p.confidence(piece.strength).evidence(vec![tag]);
        if !fresh.iter().any(|x| x.key() == p.key()) {
            fresh.push(p);
        }
    }
    set_family(org, question, &ev);
    let mut refuted: std::collections::BTreeMap<&str, Vec<&nmmem::evidence::Piece>> = std::collections::BTreeMap::new();
    for piece in ev.refuted() {
        refuted.entry(piece.record.as_str()).or_default().push(piece);
    }
    for (record, pieces) in refuted {
        let places: Vec<String> = pieces.iter().map(|p| p.place.label()).collect();
        let q = pieces.first().map_or("", |p| p.question.as_str());
        let text = format!(
            "refuted earlier: {} answered the like question \"{q}\" and was refuted (verified {}, refuted {}); memory serves none of these lines",
            places.join(", "),
            pieces.first().map_or(0, |p| p.verified),
            pieces.first().map_or(0, |p| p.refuted)
        );
        notes.push(
            Packet::new("memory", &format!("memory:{record}"), &text, Provenance::new(&format!("memory:{record}"), "nmmem.evidence#m0", 0, nmtype::now_ms()))
                .status(Status::Contradicted)
                .stance(Stance::Subjective)
                .category(Category::Support)
                .confidence(pieces.first().map_or(0.0, |p| p.strength))
                .evidence(places),
        );
    }
    let eps: Vec<&nmmem::episodic::Episode> = scored.iter().map(|(_, e)| e).collect();
    let mut packets: Vec<Packet> = eps
        .iter()
        .map(|e| {
            // The text carries the earlier question, so the reranker scores question to question
            // (what an episode is for); provenance stays memory, stance subjective.
            let text =
                format!("earlier question: {} | earlier answer: {}", e.query, e.answer.lines().find(|l| l.contains(':')).unwrap_or(""));
            // A claim citing a file changed after the episode is kept and marked obsolete
            // (Comprehensive Vision 6.3, 6.4), never silently dropped.
            let mut flags: Vec<String> = e
                .evidence
                .iter()
                .filter_map(|ev| ev.split(':').next())
                .filter_map(|f| changed.get(f).filter(|v| **v > e.version).map(|v| format!("obsolete: {f} changed at version {v}")))
                .take(1)
                .collect();
            flags.extend(stale.get(&e.id).into_iter().flatten().cloned());
            Packet::new(
                "memory",
                &format!("memory:{}", e.id),
                &text,
                Provenance::new(&format!("memory:{}", e.id), "nmmem.episodic#m0", e.version, nmtype::now_ms()),
            )
            .status(Status::Unknown)
            .stance(Stance::Subjective)
            .category(Category::Support)
            .evidence(e.evidence.iter().take(4).cloned().collect())
            .constraints(flags)
        })
        .collect();
    let k = packets.len() + ev.served().map(|p| p.record.as_str()).collect::<std::collections::BTreeSet<_>>().len();
    packets.extend(notes);
    (packets, fresh, k)
}

/// One piece of memory's family: the place (path, first and last line, as read fresh now) and the
/// strength of the request's match to the verified outcome it comes from (0 to 1).
pub type FamilyPiece = (String, usize, usize, f32);
/// Memory's family for the call in hand, with the state it was computed from (question, studied
/// tree and version).
type Family = Option<(String, Vec<FamilyPiece>)>;
static FAMILY: std::sync::Mutex<Family> = std::sync::Mutex::new(None);

fn family_key(org: &Organism, question: &str) -> String {
    let sub = org.sub();
    format!("{question}\u{1f}{}\u{1f}{}", sub.root.display(), sub.version)
}

/// Whether memory's family of evidence enters the drones' judgment of packets: on, unless
/// `NANO_MEMORY_FAMILY=off`.
#[must_use]
pub fn family_on() -> bool {
    std::env::var("NANO_MEMORY_FAMILY").ok().as_deref() != Some("off")
}

/// Keeps the served pieces of memory's evidence (verified outcomes of like requests and the places
/// used with them, each read fresh) for the drones' judgment of this call.
fn set_family(org: &Organism, question: &str, ev: &nmmem::evidence::Evidence) {
    let pieces: Vec<FamilyPiece> =
        ev.served().map(|p| (p.place.path.replace('\\', "/"), p.place.start, p.place.end, p.strength.clamp(0.0, 1.0))).collect();
    if let Ok(mut f) = FAMILY.lock() {
        *f = Some((family_key(org, question), pieces));
    }
}

/// Memory's family of evidence for `question`, as the recall of this call computed it: empty with
/// memory off, with the family switched off, or when no recall ran for this question and tree.
#[must_use]
pub fn memory_family(org: &Organism, question: &str) -> Vec<FamilyPiece> {
    if !org.memory_on || !family_on() {
        return Vec::new();
    }
    let key = family_key(org, question);
    FAMILY.lock().ok().and_then(|f| f.clone()).filter(|(k, _)| *k == key).map(|(_, v)| v).unwrap_or_default()
}

/// The strongest family piece over lines `a..=b` of `path`, if any.
#[must_use]
pub fn family_strength(family: &[FamilyPiece], path: &str, a: usize, b: usize) -> Option<f32> {
    let path = path.replace('\\', "/");
    family.iter().filter(|(p, s, e, _)| *p == path && *s <= b && a <= *e).map(|x| x.3).reduce(f32::max)
}

/// A packet's judgment with memory's family beside the drones' families: `accepted` and `share` as
/// the drones judged its piece, `fired` whether any drone signal fired on the piece for this
/// request, `memory` the family's strength there. Memory is never evidence by itself: it accepts a
/// piece only where the piece, read fresh, holds some evidence for the request. Then the families
/// combine as independent evidence (one minus the product of what each leaves unexplained).
/// With `NANO_MEMORY_FAMILY=accepted` (a measurement switch), memory joins only pieces the drones
/// accepted for an aspect of the request, never one they only fired on.
#[must_use]
pub fn with_family(accepted: bool, fired: bool, share: f32, memory: Option<f32>) -> (bool, f32) {
    family_in(accepted, fired && !family_accepted_only(), share, memory)
}

/// Whether a packet is the memory note of a refuted outcome (`recall`): kind `memory`, status
/// contradicted, from the outcome evidence.
#[must_use]
pub fn is_refuted_note(p: &Packet) -> bool {
    p.kind() == "memory" && p.get_status() == Status::Contradicted && p.provenance().origin.as_str() == "nmmem.evidence#m0"
}

/// Whether memory's family joins only pieces the drones accepted (`NANO_MEMORY_FAMILY=accepted`).
#[must_use]
pub fn family_accepted_only() -> bool {
    std::env::var("NANO_MEMORY_FAMILY").ok().as_deref() == Some("accepted")
}

fn family_in(accepted: bool, fired: bool, share: f32, memory: Option<f32>) -> (bool, f32) {
    match memory {
        Some(m) if accepted || fired => (true, 1.0 - (1.0 - share.clamp(0.0, 1.0)) * (1.0 - m.clamp(0.0, 1.0))),
        _ => (accepted, share),
    }
}

/// Section check of memory's family in the drones' judgment: a piece the drones accept gains by the
/// family; one they fired on but did not accept is accepted with the family; one they never fired
/// on is left as it was (memory is never evidence by itself); a place matches by path and
/// overlapping lines only.
/// # Errors
/// The fault, named.
pub fn check_family() -> Result<(), String> {
    let fam: Vec<FamilyPiece> = vec![("src/a.rs".into(), 10, 20, 0.8), ("src/b.rs".into(), 5, 5, 0.4)];
    if family_strength(&fam, "src\\a.rs", 18, 30) != Some(0.8) || family_strength(&fam, "src/a.rs", 21, 30).is_some() {
        return Err("a family place matched by path and overlapping lines only".into());
    }
    if family_strength(&fam, "src/c.rs", 10, 20).is_some() {
        return Err("a family place matched another file".into());
    }
    let (ok, s) = family_in(true, true, 0.5, Some(0.8));
    if !ok || (s - 0.9).abs() > 1e-6 {
        return Err(format!("an accepted piece with the family: expected 0.9, got {s}"));
    }
    if !family_in(false, true, 0.2, Some(0.5)).0 {
        return Err("a piece the drones fired on, with the family, was not accepted".into());
    }
    if family_in(false, false, 0.0, Some(1.0)) != (false, 0.0) {
        return Err("memory accepted a piece the drones never fired on".into());
    }
    if family_in(true, true, 0.3, None) != (true, 0.3) {
        return Err("a piece without the family changed".into());
    }
    Ok(())
}

/// The citation shape `reread` takes for an outcome's place (`path:line lines a..b h:<hash>`), so
/// the lines are found by their hash at their place or where they moved.
fn cite_of(p: &nmmem::outcomes::Place) -> String {
    format!("{}:{} lines {}..{} h:{}", p.path, p.start, p.start, p.end, p.hash.get(..nmsubs::views::CITE_HASH_HEX).unwrap_or(&p.hash))
}

/// Fresh reads of outcome places for one studied tree and version (root and version, then the
/// place's citation to its first line now and its text, or none when it no longer reads back).
/// RAM only, replaced when the version changes.
type PlaceReads = Option<(String, std::collections::BTreeMap<String, Option<(usize, String)>>)>;
static PLACE_READS: std::sync::Mutex<PlaceReads> = std::sync::Mutex::new(None);

/// Memory's evidence for a question from the outcomes in the open journal (verified and refuted),
/// on the processor, each place read fresh from the tree in RAM. It always completes (no deadline
/// skips it); the audit step `nmmem.evidence#m0` records its time. Empty with memory off or without
/// the journal.
#[must_use]
pub fn outcome_evidence(org: &Organism, m: &crate::forage::MemoryState, question: &str) -> nmmem::evidence::Evidence {
    let Some(m2) = m.mem2.as_ref() else { return nmmem::evidence::Evidence::default() };
    let t = std::time::Instant::now();
    let known = nmmem::outcomes::known_in(&m2.book);
    if known.is_empty() {
        return nmmem::evidence::Evidence::default();
    }
    let sub = org.sub();
    let idf = |t: &str| sub.idf(t);
    // Each place's fresh read is kept in RAM for the studied tree's version (`PLACE_READS`): the
    // same version holds the same lines, so a later call reads it again only after a change.
    let tree = format!("{}\u{1f}{}", sub.root.display(), sub.version);
    let read = |p: &nmmem::outcomes::Place| -> Option<(nmmem::outcomes::Place, FromMaterial<String>)> {
        let cite = cite_of(p);
        let held = PLACE_READS.lock().ok().and_then(|r| r.as_ref().filter(|(k, _)| *k == tree).and_then(|(_, m)| m.get(&cite).cloned()));
        let got = held.unwrap_or_else(|| {
            let got = match sub.reread_state(&cite) {
                nmsubs::views::Reread::Fresh(pk) => pk
                    .provenance()
                    .source
                    .rsplit_once(':')
                    .and_then(|(_, l)| l.parse::<usize>().ok())
                    .map(|start| (start, pk.text().to_owned())),
                _ => None,
            };
            if let Ok(mut r) = PLACE_READS.lock() {
                if r.as_ref().is_none_or(|(k, _)| *k != tree) {
                    *r = Some((tree.clone(), std::collections::BTreeMap::new()));
                }
                if let Some((_, m)) = r.as_mut() {
                    m.insert(cite.clone(), got.clone());
                }
            }
            got
        });
        let (start, text) = got?;
        Some((nmmem::outcomes::Place { start, end: start + (p.end - p.start), ..p.clone() }, FromMaterial::new(text)))
    };
    let ev = nmmem::evidence::evidence(&known, &nmmem::evidence::Request { text: question, idf: &idf, read: &read });
    let cells = vec!["read:outcomes".to_owned()];
    let (us, n) = (u64::try_from(t.elapsed().as_micros()).unwrap_or(u64::MAX), ev.served().count() as u64);
    crate::tools::note(crate::tools::Step {
        part: "nmmem.evidence#m0".into(),
        ms: Some(us / 1000),
        us: Some(us),
        cells,
        counts: vec![("outcomes", known.len() as u64), ("pieces", n), ("refuted", ev.refuted().count() as u64), ("stale", ev.stale as u64)],
        ..crate::tools::Step::default()
    });
    ev
}

/// Chunks memory recalls for a question, read only (refinement step 1, item 5: the memory lens's
/// list on the tentacle core path): the chunks holding the lines cited by the episodes word
/// recall chooses (as `recall` chooses them, without touching the hierarchy), in recall order.
#[must_use]
pub fn recalled_chunks(org: &Organism, question: &str) -> Vec<usize> {
    let sub = org.sub();
    let idf = |t: &str| sub.idf(t);
    let Ok(mut m) = org.memory.lock() else { return Vec::new() };
    // 2026-11-12: episodes recalled by meaning join the word-recalled ones (when both engines
    // are bound; empty otherwise).
    let by_meaning: Vec<String> = meaning(org, &mut m, question).into_iter().map(|(_, id)| id).collect();
    let (words, ev) = recall_inputs(org, &m, question, &idf, false);
    let mut scored: Vec<nmmem::episodic::Episode> = words.into_iter().map(|(_, e)| e).collect();
    for id in by_meaning {
        if !scored.iter().any(|e| e.id == id)
            && let Some(e) = m.episodic.get(&id)
        {
            scored.push(e.clone());
        }
    }
    let mut out: Vec<usize> = Vec::new();
    let mut add = |path: &str, line: usize| {
        for (i, ch) in sub.chunks.iter().enumerate() {
            if ch.start <= line && line <= ch.end && sub.files.get(ch.file).is_some_and(|f| f.path == path) && !out.contains(&i) {
                out.push(i);
            }
        }
    };
    for e in &scored {
        for c in e.evidence.iter().filter(|c| !c.starts_with("note:")).take(REREAD_PER_EPISODE) {
            let loc = c.split_once(' ').map_or(c.as_str(), |(a, _)| a);
            let Some((path, line)) = loc.rsplit_once(':') else { continue };
            let Ok(line) = line.parse::<usize>() else { continue };
            add(path, line);
        }
    }
    // The places of verified outcomes of like questions, read fresh.
    for p in ev.served() {
        add(&p.place.path, p.place.start);
    }
    out
}

/// The files of the places memory recalls for a question, in recall order, read only: the files of
/// `recalled_chunks`. Places of the material only; nothing of an earlier question. None with
/// `NANO_MEMORY_SEEDS=off` (a measurement switch: the recall at G2 then computes alone).
#[must_use]
pub fn recalled_files(org: &Organism, question: &str) -> Vec<String> {
    if std::env::var("NANO_MEMORY_SEEDS").ok().as_deref() == Some("off") {
        return Vec::new();
    }
    let chunks = recalled_chunks(org, question);
    let sub = org.sub();
    let mut out: Vec<String> = Vec::new();
    for f in chunks.iter().filter_map(|i| sub.chunks.get(*i)).filter_map(|ch| sub.files.get(ch.file)) {
        if !out.contains(&f.path) {
            out.push(f.path.clone());
        }
    }
    out
}

/// Co-citation links from memory (2026-11-12, memory design: "co-citation links act as priors"
/// at G1): chunks cited together by one verified episode or used together in one verified
/// outcome, by content key, weighted by how many cite the pair.
#[must_use]
pub fn cocitations(org: &Organism) -> std::collections::BTreeMap<String, Vec<(String, f32)>> {
    use std::collections::BTreeMap;
    let sub = org.sub();
    let mut pairs: BTreeMap<(String, String), f32> = BTreeMap::new();
    let Ok(m) = org.memory.lock() else { return BTreeMap::new() };
    let key_at = |path: &str, line: usize| {
        sub.chunks.iter().find(|ch| ch.start <= line && line <= ch.end && sub.files.get(ch.file).is_some_and(|f| f.path == path)).map(|ch| ch.key.clone())
    };
    let mut add = |keys: &[String], w: f32| {
        for (i, a) in keys.iter().enumerate() {
            for b in keys.iter().skip(i + 1).filter(|b| *b != a) {
                *pairs.entry((a.clone(), b.clone())).or_default() += w;
            }
        }
    };
    for e in m.episodic.live().into_iter().filter(|e| e.correct == Some(true)) {
        let mut keys: Vec<String> = Vec::new();
        for c in e.evidence.iter().filter(|c| !c.starts_with("note:")) {
            let loc = c.split_once(' ').map_or(c.as_str(), |(a, _)| a);
            let Some((path, line)) = loc.rsplit_once(':') else { continue };
            let Ok(line) = line.parse::<usize>() else { continue };
            if let Some(k) = key_at(path, line)
                && !keys.contains(&k)
            {
                keys.push(k);
            }
        }
        add(&keys, 1.0);
    }
    let known = m.mem2.as_ref().map(|m2| nmmem::outcomes::known_in(&m2.book)).unwrap_or_default();
    for (a, b, count) in nmmem::evidence::cocitations(&known) {
        if let (Some(x), Some(y)) = (key_at(&a.path, a.start), key_at(&b.path, b.start)) {
            #[expect(clippy::cast_precision_loss, reason = "small counts")]
            add(&[x, y], count as f32);
        }
    }
    let mut links: BTreeMap<String, Vec<(String, f32)>> = BTreeMap::new();
    for ((a, b), w) in pairs {
        links.entry(a.clone()).or_default().push((b.clone(), w));
        links.entry(b).or_default().push((a, w));
    }
    links
}

/// The meaning gate's two floors: one for judging the new request against an earlier request, one
/// for judging it against an earlier answer's lines read fresh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Floors {
    pub request: f32,
    pub material: f32,
}

/// One stored episode as the meaning gate sees it: its question (request data) and its first cited
/// place read fresh (material data; `None` when the lines no longer read back).
#[derive(Clone, Debug)]
pub struct Candidate {
    pub id: String,
    pub question: FromRequest<String>,
    pub lines: Option<FromMaterial<String>>,
}

/// How the meaning gate asks its judge: the new request against one item of one origin.
pub type Judge<'a> = &'a dyn Fn(&FromRequest<&str>, Against<'_, String>) -> Option<f32>;

/// The candidates memory's meaning gate judges: each episode named that has a cited place, in the
/// order given, its place read fresh.
fn candidates(org: &Organism, m: &crate::forage::MemoryState, ids: &[String]) -> Vec<Candidate> {
    let sub = org.sub();
    ids.iter()
        .filter_map(|id| {
            let e = m.episodic.get(id)?;
            let cite = e.evidence.iter().find(|c| !c.starts_with("note:"))?;
            Some(Candidate {
                id: id.clone(),
                question: FromRequest::new(e.query.clone()),
                lines: sub.reread(cite).map(|p| FromMaterial::new(p.text().to_owned())),
            })
        })
        .collect()
}

const MEANING_SUBDIR: &str = "meaning";
const FLOORS_FILE: &str = "floors.json";
const PAIRS_FILE: &str = "rerank_pairs.bin";
const PAIRS_MAGIC: &[u8; 8] = b"NMPARK1\0";
const PAIRS_KEY_LEN: usize = 64;
const PAIRS_CAP: usize = 4096;

#[derive(Clone)]
struct CachedFloors {
    dir: std::path::PathBuf,
    reranker_key: String,
    episodes_digest: String,
    sub_digest: String,
    candidates_digest: String,
    floors: Floors,
}

static FLOORS_CACHE: std::sync::Mutex<Option<CachedFloors>> = std::sync::Mutex::new(None);
static KEPT_PAIRS: std::sync::Mutex<std::collections::BTreeMap<(String, String), f32>> = std::sync::Mutex::new(std::collections::BTreeMap::new());

fn disk_cache_on(org: &Organism) -> bool {
    org.memory_on && nmsubs::dense::store::cache_on(&org.home) && crate::whole::recall_cache_control(&org.home)
}

fn prompt_hash(query: &str) -> String {
    nmhash::sha256_bytes(format!("Instruct: {MEANING_TASK}\nQuery:{query}").as_bytes())
}

fn pair_hash(q: &str, d: &str) -> String {
    let qh = nmhash::sha256_bytes(q.as_bytes());
    let dh = nmhash::sha256_bytes(d.as_bytes());
    nmhash::sha256_bytes(format!("{qh}:{dh}").as_bytes())
}

fn episodes_digest(live: &[&nmmem::episodic::Episode]) -> String {
    let mut s = String::new();
    for e in live {
        let cite = e.evidence.iter().find(|c| !c.starts_with("note:")).map_or("", String::as_str);
        let qh = nmhash::sha256_bytes(e.query.as_bytes());
        s.push_str(&format!("{}\t{:?}\t{cite}\t{qh}\n", e.id, e.correct));
    }
    nmhash::sha256_bytes(s.as_bytes())
}

fn candidates_digest(docs: &[Candidate]) -> String {
    let mut s = String::new();
    for c in docs {
        let qh = nmhash::sha256_bytes(c.question.get().as_bytes());
        let lh = c.lines.as_ref().map_or_else(|| "none".to_owned(), |l| nmhash::sha256_bytes(l.get().as_bytes()));
        s.push_str(&format!("{}\t{qh}\t{lh}\n", c.id));
    }
    nmhash::sha256_bytes(s.as_bytes())
}

fn write_atomic(dir: &Path, file: &str, bytes: &[u8]) {
    use std::io::Write as _;
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let tmp = dir.join(format!("{file}.tmp"));
    let Ok(mut f) = std::fs::File::create(&tmp) else { return };
    if f.write_all(bytes).and_then(|()| f.sync_all()).is_ok() {
        drop(f);
        let _ = std::fs::rename(&tmp, dir.join(file));
    }
}

fn load_pair_rows(dir: &Path, rkey: &str) -> Vec<(String, f32)> {
    let Ok(bytes) = std::fs::read(dir.join(PAIRS_FILE)) else { return Vec::new() };
    if bytes.get(..8) != Some(PAIRS_MAGIC.as_slice()) {
        return Vec::new();
    }
    let Some(hdr_len) = bytes.get(8..12).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize) else {
        return Vec::new();
    };
    let Some(hdr_bytes) = bytes.get(12..12 + hdr_len) else { return Vec::new() };
    let Ok(hdr) = std::str::from_utf8(hdr_bytes) else { return Vec::new() };
    let mut got_key = None;
    let mut rows = None;
    for l in hdr.lines() {
        if let Some((k, v)) = l.split_once('=') {
            if k == "reranker_key" {
                got_key = Some(v);
            } else if k == "rows" {
                rows = v.parse::<usize>().ok();
            }
        }
    }
    if got_key != Some(rkey) {
        return Vec::new();
    }
    let Some(rows) = rows else { return Vec::new() };
    let Some(sum) = bytes.get(12 + hdr_len..12 + hdr_len + PAIRS_KEY_LEN) else { return Vec::new() };
    let body = bytes.get(12 + hdr_len + PAIRS_KEY_LEN..).unwrap_or(&[]);
    let rec_len = PAIRS_KEY_LEN + 4;
    if body.len() != rows.saturating_mul(rec_len) || nmhash::sha256_bytes(body).as_bytes() != sum {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(rows);
    for rec in body.chunks_exact(rec_len) {
        let (k, v) = rec.split_at(PAIRS_KEY_LEN);
        let Ok(k) = std::str::from_utf8(k) else { return Vec::new() };
        if !k.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Vec::new();
        }
        let score = f32::from_le_bytes([v[0], v[1], v[2], v[3]]);
        if score.is_finite() {
            out.push((k.to_owned(), score));
        }
    }
    out
}

fn save_pair_rows(dir: &Path, rkey: &str, rows: &[(String, f32)]) {
    let start = rows.len().saturating_sub(PAIRS_CAP);
    let kept = &rows[start..];
    let mut body = Vec::with_capacity(kept.len() * (PAIRS_KEY_LEN + 4));
    for (k, v) in kept {
        if k.len() == PAIRS_KEY_LEN && v.is_finite() {
            body.extend_from_slice(k.as_bytes());
            body.extend_from_slice(&v.to_le_bytes());
        }
    }
    let count = body.len() / (PAIRS_KEY_LEN + 4);
    let hdr = format!("reranker_key={}\nrows={count}\n", rkey.replace('\n', " "));
    let sum = nmhash::sha256_bytes(&body);
    let mut out = Vec::with_capacity(8 + 4 + hdr.len() + PAIRS_KEY_LEN + body.len());
    out.extend_from_slice(PAIRS_MAGIC);
    out.extend_from_slice(&u32::try_from(hdr.len()).unwrap_or(0).to_le_bytes());
    out.extend_from_slice(hdr.as_bytes());
    out.extend_from_slice(sum.as_bytes());
    out.extend_from_slice(&body);
    write_atomic(dir, PAIRS_FILE, &out);
}

struct PairCache {
    dir: std::path::PathBuf,
    rkey: String,
    disk: bool,
    rows: Vec<(String, f32)>,
    index: std::collections::BTreeMap<String, usize>,
    dirty: bool,
}

impl PairCache {
    fn open(mem_dir: &Path, rkey: &str, disk: bool) -> Self {
        let dir = mem_dir.join(MEANING_SUBDIR);
        let rows = if disk { load_pair_rows(&dir, rkey) } else { Vec::new() };
        let mut index = std::collections::BTreeMap::new();
        for (i, (k, _)) in rows.iter().enumerate() {
            index.insert(k.clone(), i);
        }
        if !rows.is_empty()
            && let Ok(mut kept) = KEPT_PAIRS.lock()
        {
            if kept.len() > PAIRS_CAP {
                kept.clear();
            }
            for (k, v) in &rows {
                kept.insert((rkey.to_owned(), k.clone()), *v);
            }
        }
        Self { dir, rkey: rkey.to_owned(), disk, rows, index, dirty: false }
    }

    fn get(&self, ph: &str) -> Option<f32> {
        if let Ok(kept) = KEPT_PAIRS.lock()
            && let Some(v) = kept.get(&(self.rkey.clone(), ph.to_owned()))
        {
            return Some(*v);
        }
        self.index.get(ph).and_then(|i| self.rows.get(*i)).map(|(_, v)| *v)
    }

    fn insert(&mut self, ph: String, val: f32) {
        if !val.is_finite() {
            return;
        }
        if let Ok(mut kept) = KEPT_PAIRS.lock() {
            if kept.len() >= PAIRS_CAP {
                kept.clear();
            }
            kept.insert((self.rkey.clone(), ph.clone()), val);
        }
        if let Some(&i) = self.index.get(&ph) {
            if let Some(slot) = self.rows.get_mut(i) {
                slot.1 = val;
            }
        } else {
            let idx = self.rows.len();
            self.index.insert(ph.clone(), idx);
            self.rows.push((ph, val));
        }
        self.dirty = true;
    }

    fn flush(&self) {
        if self.disk && self.dirty {
            save_pair_rows(&self.dir, &self.rkey, &self.rows);
        }
    }
}

fn load_floors_disk(dir: &Path, rkey: &str) -> Option<CachedFloors> {
    let bytes = std::fs::read(dir.join(MEANING_SUBDIR).join(FLOORS_FILE)).ok()?;
    let v = nmjson::parse(&bytes).ok()?;
    if v.get("version").and_then(Value::integer).ok() != Some(1) {
        return None;
    }
    let got_key = v.get("reranker_key").and_then(Value::string).ok()?;
    if got_key != rkey {
        return None;
    }
    let ep = v.get("episodes_digest").and_then(Value::string).ok()?.to_owned();
    let sub = v.get("sub_digest").and_then(Value::string).ok()?.to_owned();
    let cand = v.get("candidates_digest").and_then(Value::string).ok()?.to_owned();
    let req_bits = v.get("request_bits").and_then(Value::integer).ok().and_then(|x| u32::try_from(x).ok())?;
    let mat_bits = v.get("material_bits").and_then(Value::integer).ok().and_then(|x| u32::try_from(x).ok())?;
    let (request, material) = (f32::from_bits(req_bits), f32::from_bits(mat_bits));
    if !request.is_finite() || !material.is_finite() {
        return None;
    }
    Some(CachedFloors {
        dir: dir.to_path_buf(),
        reranker_key: rkey.to_owned(),
        episodes_digest: ep,
        sub_digest: sub,
        candidates_digest: cand,
        floors: Floors { request, material },
    })
}

fn save_floors_disk(dir: &Path, entry: &CachedFloors, candidates: usize) {
    let v = obj(vec![
        ("version", n(1)),
        ("reranker_key", s(&entry.reranker_key)),
        ("episodes_digest", s(&entry.episodes_digest)),
        ("sub_digest", s(&entry.sub_digest)),
        ("candidates_digest", s(&entry.candidates_digest)),
        ("candidates", n(candidates as u64)),
        ("request_bits", n(u64::from(entry.floors.request.to_bits()))),
        ("material_bits", n(u64::from(entry.floors.material.to_bits()))),
        ("request", Value::Number(format!("{:.6}", entry.floors.request))),
        ("material", Value::Number(format!("{:.6}", entry.floors.material))),
    ]);
    if let Ok(encoded) = v.encode() {
        write_atomic(&dir.join(MEANING_SUBDIR), FLOORS_FILE, encoded.as_bytes());
    }
}

fn floor_from_scores(scores: Vec<f32>) -> f32 {
    nmmem::evidence::null_bar(&scores).filter(|b| *b < 1.0).unwrap_or(MEANING_FLOOR)
}

fn recompute_meaning_floor(org: &Organism, m: &crate::forage::MemoryState, docs: &[Candidate], rkey: &str, disk: bool) -> Floors {
    let mut cache = PairCache::open(&m.dir, rkey, disk);
    let mat_count = docs.iter().filter(|d| d.lines.is_some()).count();
    let include_material = mat_count >= nmmem::evidence::NULL_MIN;
    let mut req_pairs: Vec<(String, String, String)> = Vec::with_capacity(docs.len());
    let mut mat_pairs: Vec<(String, String, String)> = Vec::new();
    let mut missing: Vec<(String, String, String)> = Vec::new();
    let mut seen_missing: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for (i, c) in docs.iter().enumerate() {
        let Some(other) = docs.get((i + 1) % docs.len().max(1)).filter(|_| docs.len() > 1) else { continue };
        let q = c.question.get().clone();
        let req_doc = format!("earlier question: {}", other.question.get());
        let req_ph = pair_hash(&q, &req_doc);
        if cache.get(&req_ph).is_none() && seen_missing.insert(req_ph.clone()) {
            missing.push((req_ph.clone(), q.clone(), req_doc.clone()));
        }
        req_pairs.push((q.clone(), req_doc, req_ph));
        if include_material && let Some(lines) = other.lines.as_ref() {
            let mat_doc = lines.get().clone();
            let mat_ph = pair_hash(&q, &mat_doc);
            if cache.get(&mat_ph).is_none() && seen_missing.insert(mat_ph.clone()) {
                missing.push((mat_ph.clone(), q, mat_doc.clone()));
            }
            mat_pairs.push((c.question.get().clone(), mat_doc, mat_ph));
        }
    }
    if !missing.is_empty() {
        let pair_refs: Vec<(&str, &str)> = missing.iter().map(|(_, q, d)| (q.as_str(), d.as_str())).collect();
        if let Some(scores) = crate::neural::rerank_pairs(org, &pair_refs) {
            for ((ph, _, _), s) in missing.into_iter().zip(scores) {
                cache.insert(ph, s);
            }
        }
    }
    cache.flush();
    let on_request: Vec<f32> = req_pairs.iter().filter_map(|(_, _, ph)| cache.get(ph)).collect();
    let on_material: Vec<f32> = mat_pairs.iter().filter_map(|(_, _, ph)| cache.get(ph)).collect();
    Floors { request: floor_from_scores(on_request), material: floor_from_scores(on_material) }
}

fn cached_floors_fast(mem_dir: &Path, rkey: &str, ep_digest: &str, sub_digest: &str, disk: bool) -> Option<(Floors, Option<CachedFloors>)> {
    if let Ok(c) = FLOORS_CACHE.lock()
        && let Some(hit) = c.as_ref()
        && hit.dir == mem_dir
        && hit.reranker_key == rkey
        && hit.episodes_digest == ep_digest
        && hit.sub_digest == sub_digest
    {
        return Some((hit.floors, None));
    }
    let disk_hit = if disk { load_floors_disk(mem_dir, rkey) } else { None };
    if let Some(hit) = disk_hit.as_ref()
        && hit.episodes_digest == ep_digest
        && hit.sub_digest == sub_digest
    {
        if let Ok(mut c) = FLOORS_CACHE.lock() {
            *c = Some(hit.clone());
        }
        return Some((hit.floors, None));
    }
    None
        .or(Some((Floors { request: MEANING_FLOOR, material: MEANING_FLOOR }, disk_hit)))
        .filter(|(_, d)| d.is_some())
}

fn cached_floors_by_candidates(next: &CachedFloors, cand_count: usize, disk: bool, disk_hit: Option<CachedFloors>) -> Option<Floors> {
    if let Ok(mut c) = FLOORS_CACHE.lock()
        && let Some(hit) = c.as_mut()
        && hit.dir == next.dir
        && hit.reranker_key == next.reranker_key
        && hit.candidates_digest == next.candidates_digest
    {
        hit.episodes_digest.clone_from(&next.episodes_digest);
        hit.sub_digest.clone_from(&next.sub_digest);
        let updated = hit.clone();
        drop(c);
        if disk {
            save_floors_disk(&next.dir, &updated, cand_count);
        }
        return Some(updated.floors);
    }
    if let Some(mut hit) = disk_hit
        && hit.candidates_digest == next.candidates_digest
    {
        hit.episodes_digest.clone_from(&next.episodes_digest);
        hit.sub_digest.clone_from(&next.sub_digest);
        if let Ok(mut c) = FLOORS_CACHE.lock() {
            *c = Some(hit.clone());
        }
        save_floors_disk(&next.dir, &hit, cand_count);
        return Some(hit.floors);
    }
    None
}

/// The meaning gate's floors, derived at run time (2026-11-12, memory design step 7: "floor set
/// from development decoys at run time, fallback 0.7"), one per kind of judged input. The decoys
/// are memory's own mismatches: each stored question against a different episode's question (the
/// request floor) and against a different episode's cited lines read fresh (the material floor).
/// Each floor is the bar a judgment must reach to stand out from its decoys (median plus
/// `nmmem::evidence::NULL_BAND` robust deviations, `nmmem::evidence::null_bar`), over every live
/// episode's pair (each pair judged once per process), cached per model/episodes/candidates digest;
/// with fewer than `nmmem::evidence::NULL_MIN` decoys, or a bar no P(yes) can reach, the fallback.
pub fn meaning_floor(org: &Organism, m: &crate::forage::MemoryState) -> Floors {
    let fallback = Floors { request: MEANING_FLOOR, material: MEANING_FLOOR };
    let live = m.episodic.live();
    if live.len() < nmmem::evidence::NULL_MIN {
        return fallback;
    }
    let Some(rkey) = crate::neural::reranker_values_key(org) else { return fallback };
    let ep_digest = episodes_digest(&live);
    let sub_digest = org.sub().digest();
    let disk = disk_cache_on(org);
    let disk_hit = match cached_floors_fast(&m.dir, &rkey, &ep_digest, &sub_digest, disk) {
        Some((f, None)) => return f,
        Some((_, d)) => d,
        None => None,
    };
    let ids: Vec<String> = live.iter().map(|e| e.id.clone()).collect();
    let docs: Vec<Candidate> = candidates(org, m, &ids);
    if docs.len() < nmmem::evidence::NULL_MIN {
        return fallback;
    }
    let cand_digest = candidates_digest(&docs);
    let mut entry = CachedFloors {
        dir: m.dir.clone(),
        reranker_key: rkey.clone(),
        episodes_digest: ep_digest,
        sub_digest,
        candidates_digest: cand_digest,
        floors: fallback,
    };
    if let Some(f) = cached_floors_by_candidates(&entry, docs.len(), disk, disk_hit) {
        return f;
    }
    let floors = recompute_meaning_floor(org, m, &docs, &rkey, disk);
    entry.floors = floors;
    if let Ok(mut c) = FLOORS_CACHE.lock() {
        *c = Some(entry.clone());
    }
    if disk {
        save_floors_disk(&m.dir, &entry, docs.len());
    }
    floors
}

/// The floors from decoy pairs: each candidate's question against the next candidate's question and,
/// apart, against its lines.
fn decoy_floors(docs: &[Candidate], judge: Judge) -> Floors {
    let (mut on_request, mut on_material): (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
    for (i, c) in docs.iter().enumerate() {
        let Some(other) = docs.get((i + 1) % docs.len().max(1)).filter(|_| docs.len() > 1) else { continue };
        let q = FromRequest::new(c.question.get().as_str());
        if let Some(p) = judge(&q, Against::Request(&other.question)) {
            on_request.push(p);
        }
        if let Some(p) = other.lines.as_ref().and_then(|l| judge(&q, Against::Material(l))) {
            on_material.push(p);
        }
    }
    let floor = |scores: Vec<f32>| nmmem::evidence::null_bar(&scores).filter(|b| *b < 1.0).unwrap_or(MEANING_FLOOR);
    Floors { request: floor(on_request), material: floor(on_material) }
}

/// The meaning gate: the new request is judged against each candidate's question (request against
/// request) and, apart, against its lines read fresh (the material against the request). A candidate
/// is recalled when either judgment reaches its own floor, with the higher passing judgment as its
/// score. No judged input holds an earlier question and material lines together.
pub fn meaning_gate(question: &FromRequest<&str>, found: &[Candidate], floors: Floors, judge: Judge) -> Vec<(f32, String)> {
    found
        .iter()
        .filter_map(|c| {
            let on_request = judge(question, Against::Request(&c.question)).filter(|p| *p >= floors.request);
            let on_material = c.lines.as_ref().and_then(|l| judge(question, Against::Material(l))).filter(|p| *p >= floors.material);
            match (on_request, on_material) {
                (None, None) => None,
                (a, b) => Some((a.unwrap_or(0.0).max(b.unwrap_or(0.0)), c.id.clone())),
            }
        })
        .collect()
}

struct GateItem {
    id: String,
    req_ph: String,
    mat_ph: Option<String>,
}

fn meaning_gate_batched(org: &Organism, mem_dir: &Path, disk: bool, question: &FromRequest<&str>, found: &[Candidate], floors: Floors) -> Vec<(f32, String)> {
    if found.is_empty() {
        return Vec::new();
    }
    let Some(rkey) = crate::neural::reranker_values_key(org) else { return Vec::new() };
    let mut cache = PairCache::open(mem_dir, &rkey, disk);
    let q = question.get();
    let mut items: Vec<GateItem> = Vec::with_capacity(found.len());
    let mut missing: Vec<(String, String)> = Vec::new();
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for c in found {
        let req_doc = format!("earlier question: {}", c.question.get());
        let req_ph = pair_hash(q, &req_doc);
        if cache.get(&req_ph).is_none() && seen.insert(req_ph.clone()) {
            missing.push((req_ph.clone(), req_doc));
        }
        let mat_ph = c.lines.as_ref().map(|l| {
            let mat_doc = l.get().clone();
            let ph = pair_hash(q, &mat_doc);
            if cache.get(&ph).is_none() && seen.insert(ph.clone()) {
                missing.push((ph.clone(), mat_doc));
            }
            ph
        });
        items.push(GateItem { id: c.id.clone(), req_ph, mat_ph });
    }
    if !missing.is_empty() {
        let docs: Vec<String> = missing.iter().map(|(_, d)| d.clone()).collect();
        if let Some(scores) = crate::neural::rerank_batch(org, q, &docs) {
            for ((ph, _), s) in missing.into_iter().zip(scores) {
                cache.insert(ph, s);
            }
        }
    }
    cache.flush();
    items
        .into_iter()
        .filter_map(|it| {
            let on_request = cache.get(&it.req_ph).filter(|p| *p >= floors.request);
            let on_material = it.mat_ph.and_then(|ph| cache.get(&ph)).filter(|p| *p >= floors.material);
            match (on_request, on_material) {
                (None, None) => None,
                (a, b) => Some((a.unwrap_or(0.0).max(b.unwrap_or(0.0)), it.id)),
            }
        })
        .collect()
}

/// Episodes recalled by meaning with the key they were judged under.
type Judged = Option<(String, Vec<(f32, String)>)>;

/// Episodes recalled by meaning: (P(yes), episode id), best first; empty without both engines.
/// The last result is kept with what it was computed from (the studied tree, its version, the
/// question and the live episodes): the recalled chunks and the recall of one question ask the
/// same, so the second asks nothing of the engines.
fn meaning(org: &Organism, m: &mut crate::forage::MemoryState, question: &str) -> Vec<(f32, String)> {
    static LAST: std::sync::Mutex<Judged> = std::sync::Mutex::new(None);
    if !org.recall_meaning.load(std::sync::atomic::Ordering::SeqCst) {
        return Vec::new();
    }
    // Nothing to judge, or no reranker to judge it: no engine is asked (nor loaded).
    if org.slots.by_name.get("reranker").is_none_or(|s| s.engine().is_none())
        || org.slots.by_name.get("embedder").is_none_or(|s| s.engine().is_none())
        || !m.episodic.live().iter().any(|e| e.correct == Some(true) && e.evidence.iter().any(|c| !c.starts_with("note:")))
    {
        return Vec::new();
    }
    let rkey = crate::neural::reranker_values_key(org).unwrap_or_default();
    let ekey = org.slots.by_name.get("embedder").and_then(|s| s.engine()).map(|e| e.label()).unwrap_or_default();
    let key = format!(
        "{question}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{rkey}\u{1f}{ekey}",
        m.dir.display(),
        org.sub().root.display(),
        org.sub().version,
        m.episodic.live().iter().map(|e| e.id.as_str()).collect::<Vec<_>>().join(",")
    );
    if let Some((_, r)) = LAST.lock().ok().and_then(|l| l.clone()).filter(|(k, _)| *k == key) {
        return r;
    }
    let out = meaning_judged(org, m, question);
    if let Ok(mut l) = LAST.lock() {
        *l = Some((key, out.clone()));
    }
    out
}

fn load_episode_vectors(org: &Organism, m: &mut crate::forage::MemoryState, live: &[(String, String)], disk: bool) -> Option<(nmsubs::dense::store::Store, Vec<String>)> {
    if !disk || !live.iter().any(|(id, _)| !m.vectors.contains_key(id)) {
        return None;
    }
    let header = crate::neural::meaning_embed_header(org)?;
    let hashes: Vec<String> = live.iter().map(|(_, q)| prompt_hash(q)).collect();
    let hash_set: std::collections::BTreeSet<&str> = hashes.iter().map(String::as_str).collect();
    let (st, loaded, _) = nmsubs::dense::store::Store::open(&m.dir.join(MEANING_SUBDIR), header, &hash_set);
    let by_hash: std::collections::BTreeMap<String, Vec<f32>> = loaded.into_iter().collect();
    for ((id, _), h) in live.iter().zip(&hashes) {
        if !m.vectors.contains_key(id)
            && let Some(v) = by_hash.get(h)
        {
            m.vectors.insert(id.clone(), v.clone());
        }
    }
    Some((st, hashes))
}

fn embed_question_and_missing(org: &Organism, m: &mut crate::forage::MemoryState, question: &str, live: &[(String, String)], disk: bool) -> Option<Vec<f32>> {
    let mut store = load_episode_vectors(org, m, live, disk);
    let q_prompt = format!("Instruct: {MEANING_TASK}\nQuery:{question}");
    let mut to_embed = vec![q_prompt];
    let mut missing_prompts: Vec<(String, String, String)> = Vec::new();
    let mut seen_prompts: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for (id, query) in live {
        if !m.vectors.contains_key(id) {
            let ph = prompt_hash(query);
            let prompt = format!("Instruct: {MEANING_TASK}\nQuery:{query}");
            if seen_prompts.insert(ph.clone()) {
                to_embed.push(prompt.clone());
            }
            missing_prompts.push((id.clone(), ph, prompt));
        }
    }
    let embs = crate::neural::embed_batch(org, &to_embed)?;
    let qv = embs.first()?.clone();
    if !missing_prompts.is_empty() {
        let mut by_prompt: std::collections::BTreeMap<&str, &Vec<f32>> = std::collections::BTreeMap::new();
        for (p, v) in to_embed.iter().skip(1).zip(embs.iter().skip(1)) {
            by_prompt.insert(p.as_str(), v);
        }
        for (id, ph, prompt) in &missing_prompts {
            if let Some(v) = by_prompt.get(prompt.as_str()) {
                let stored = if disk { nmsubs::dense::store::rounded(v) } else { (*v).clone() };
                m.vectors.insert(id.clone(), stored);
                if let Some((st, _)) = store.as_mut() {
                    st.add(ph, v);
                }
            }
        }
    }
    if let Some((st, hashes)) = store.as_mut() {
        let hash_set: std::collections::BTreeSet<&str> = hashes.iter().map(String::as_str).collect();
        let _ = st.compact(&hash_set);
        st.flush();
    }
    Some(qv)
}

fn meaning_judged(org: &Organism, m: &mut crate::forage::MemoryState, question: &str) -> Vec<(f32, String)> {
    let live: Vec<(String, String)> = m
        .episodic
        .live()
        .iter()
        .filter(|e| e.correct == Some(true) && e.evidence.iter().any(|c| !c.starts_with("note:")))
        .map(|e| (e.id.clone(), e.query.clone()))
        .collect();
    let disk = disk_cache_on(org);
    let Some(qv) = embed_question_and_missing(org, m, question, &live, disk) else { return Vec::new() };
    let mut near: Vec<(f32, String)> = Vec::new();
    for (id, _) in &live {
        if let Some(v) = m.vectors.get(id) {
            near.push((qv.iter().zip(v).map(|(a, b)| a * b).sum(), id.clone()));
        }
    }
    near.sort_by(|a, b| b.0.total_cmp(&a.0));
    let all: Vec<f32> = near.iter().map(|x| x.0).collect();
    let ids: Vec<String> = near
        .iter()
        .enumerate()
        .filter(|(i, (x, _))| {
            let others: Vec<f32> = all.iter().enumerate().filter(|(j, _)| j != i).map(|(_, v)| *v).collect();
            nmmem::evidence::above_null(*x, &others).unwrap_or(true)
        })
        .map(|(_, (_, id))| id.clone())
        .collect();
    let floors = meaning_floor(org, m);
    meaning_gate_batched(org, &m.dir, disk, &FromRequest::new(question), &candidates(org, m, &ids), floors)
}

/// Section check of the meaning gate's inputs with a recording judge: every judged input is the new
/// request against one item of one origin; no material input holds an earlier question's words and
/// no request input holds material lines; a candidate passes on either judgment at its own floor;
/// the floors' decoys keep the origins apart too.
/// # Errors
/// One fault line, starting with `nm-origin-mismatch` when the request's and the material's data
/// met in one input.
pub fn check_meaning_origins() -> Result<(), String> {
    use nmtype::class::{Origin, Sourced, WRONG_ORIGIN};
    let c = |id: &str, q: &str, l: Option<&str>| Candidate {
        id: id.into(),
        question: FromRequest::new(q.into()),
        lines: l.map(|x| FromMaterial::new(x.to_owned())),
    };
    let found = vec![
        c("e1", "where is the order total computed", Some("pub fn order_total(items: &[Item]) -> u64 {")),
        c("e2", "which function parses the config file", Some("pub fn parse_config(text: &str) -> Config {")),
        c("e3", "where are refunds booked", None),
    ];
    let seen: std::cell::RefCell<Vec<(Origin, String)>> = std::cell::RefCell::new(Vec::new());
    // A judge that answers by word overlap and records what it was given.
    let judge = |q: &FromRequest<&str>, item: Against<'_, String>| -> Option<f32> {
        seen.borrow_mut().push((item.origin(), item.item().clone()));
        let words: Vec<String> = nmlex::key_terms(q.get());
        let text = item.item().to_lowercase();
        #[expect(clippy::cast_precision_loss, reason = "word counts")]
        Some(words.iter().filter(|w| text.contains(w.as_str())).count() as f32 / words.len().max(1) as f32)
    };
    let floors = Floors { request: 0.6, material: 0.3 };
    let got = meaning_gate(&FromRequest::new("which code computes the order total"), &found, floors, &judge);
    let questions: Vec<&str> = found.iter().map(|x| x.question.get().as_str()).collect();
    let lines: Vec<&str> = found.iter().filter_map(|x| x.lines.as_ref().map(|l| l.get().as_str())).collect();
    let crossed = |seen: &[(Origin, String)]| {
        seen.iter().find(|(origin, text)| match origin {
            Origin::Material => questions.iter().any(|q| text.contains(q)) || !lines.contains(&text.as_str()),
            Origin::Request => lines.iter().any(|l| text.contains(l)) || !questions.contains(&text.as_str()),
        })
        .map(|(o, t)| format!("{WRONG_ORIGIN}: nmapi.session meaning gate judged a {} input {t:?}", o.name()))
    };
    if let Some(e) = crossed(&seen.borrow()) {
        return Err(e);
    }
    // Five inputs: three questions, two readable places (the third place no longer reads back).
    let n = seen.borrow().len();
    if n != 5 || got.iter().map(|x| x.1.as_str()).collect::<Vec<_>>() != ["e1"] {
        return Err(format!("nmapi.session meaning gate: {n} inputs, recalled {got:?}"));
    }
    seen.borrow_mut().clear();
    let _ = decoy_floors(&found, &judge);
    if let Some(e) = crossed(&seen.borrow()) {
        return Err(e);
    }
    let count = |o: Origin| seen.borrow().iter().filter(|(x, _)| *x == o).count();
    if count(Origin::Request) != 3 || count(Origin::Material) != 2 {
        return Err("nmapi.session meaning floors: decoys not one origin per input".into());
    }
    Ok(())
}

/// A hierarchy item waiting for `settle`: the answer's episode id, its topic, question, first answer
/// line and unit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pending {
    pub id: String,
    pub topic: String,
    pub query: String,
    pub text: String,
    pub unit: String,
}

/// The memory write's second half, run after the reply has gone out: the hierarchy's items for the
/// answers stored since the last settle, and the journal's views held back since then. A process
/// that ends before this loses nothing: at the next open every episode is added to the hierarchy and
/// the views are written from the journal. Timed for the audit as `nmmem.gate#m0` with the cell
/// `after-reply`, in the next call's row; nothing to do, nothing noted.
pub fn settle(org: &Organism) {
    if !org.memory_on {
        return;
    }
    let ((done, ids), c) = crate::tools::clocked(|| {
        let Ok(mut m) = org.memory.lock() else { return (false, Vec::new()) };
        let pending = std::mem::take(&mut m.pending);
        for p in &pending {
            let _ = m.hierarchy.add(&p.id, &p.topic, &p.query, &p.text, &p.unit);
        }
        let ids: Vec<String> = pending.iter().map(|p| p.id.clone()).collect();
        let held = m.mem2.as_ref().is_some_and(nmmem::mem2::Memory2::is_held);
        if let Some(m2) = m.mem2.as_mut() {
            let _ = m2.release();
        }
        (!pending.is_empty() || held, ids)
    });
    if done {
        let mut cells = memory_cells(org, "write", &ids);
        cells.push("after-reply".into());
        crate::tools::note_after_reply(crate::tools::Step { cells, ..crate::tools::step_at("nmmem.gate#m0", c) });
    }
}

/// Consolidate an answered question into an episode through the gate.
pub fn remember(org: &Organism, question: &str, pack: &Pack, emitted: &Packet, version: u64) -> (Option<String>, String) {
    if !org.memory_on {
        return (None, "not stored: memory off".into());
    }
    // A G3 replay (`crate::g3`) writes nothing.
    if crate::g3::replaying() {
        return (None, "not stored: replay".into());
    }
    if pack.withheld {
        return (None, "not stored: withheld".into());
    }
    let Ok(mut m) = org.memory.lock() else { return (None, "memory lock".into()) };
    let scope = format!("task:{}", pack.task);
    m.working.note(&scope, &pack.reason);
    // Surprises recall tagged for this question join the task, tag and reason kept.
    for (n, why) in m.working.end_task(&surprise_scope(question), &[]) {
        m.working.note(&scope, &n);
        m.working.tag(&scope, &n, &why);
    }
    // The question is stored with the episode, so it passes the same never-save screen as the
    // answer.
    if let Err(why) = m.gate.screen(&format!("question:{question}"), question).and_then(|()| m.gate.admit(emitted)) {
        return (None, format!("refused: {why}"));
    }
    // Tag and capture (memory revision 2, section 10): the task's working notes worth keeping
    // (tagged, reached the answer, recurred) are taken before the scope closes; the notes go back
    // so the episode still carries them.
    let notes = m.working.notes(&scope);
    let kept = m.working.end_task(&scope, emitted.get_evidence());
    for n in &notes {
        m.working.note(&scope, n);
    }
    // The captures and the answer's record reach the journal one by one; the views are written
    // once, after the record.
    if let Some(m2) = m.mem2.as_mut() {
        m2.hold();
        for (note, why) in &kept {
            let _ = m2.capture(note, why);
        }
    }
    // Episodes keep each cited span as its place, line range and content hash, never its code.
    let sub = org.sub();
    let ep = nmmem::consolidation::consolidate(&mut m.working, &scope, question, emitted, version, &|c| sub.cite_ref(c));
    let id = ep.id.clone();
    let (q, a) = (ep.query.clone(), ep.answer.lines().next().unwrap_or("").to_owned());
    let unit = ep.unit.clone();
    let copy = ep.clone();
    // Before the reply: the two records the memory is rebuilt from, the episode (appended) and the
    // journal event (flushed to disk before it counts). After the reply (`settle`): the hierarchy's
    // item and the journal's views, both rebuilt from those records at the next open.
    m.pending.push(Pending { id: id.clone(), topic: "context".into(), query: q, text: a, unit });
    let written = m.episodic.write(ep);
    // Revision 2: the same answer as a journal record (a repeat counts one use).
    let r2 = match &written {
        Ok(Written::Stored(_) | Written::Duplicate(_)) => m.mem2.as_mut().map(|m2| m2.remember(&copy)),
        _ => None,
    };
    match written {
        Ok(w @ (Written::Stored(_) | Written::Duplicate(_))) => {
            let note = match r2 {
                Some(Err(e)) => format!("; revision 2: {e}"),
                _ => String::new(),
            };
            let head = if matches!(w, Written::Stored(_)) { "stored" } else { "duplicate: already stored, not written again" };
            (Some(id), format!("{head}{note}"))
        }
        Ok(Written::Refused(r)) => (None, format!("refused: {r}")),
        Err(e) => (None, format!("error: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct MockEngine {
        label: String,
        embed_calls: Arc<AtomicUsize>,
        rerank_calls: Arc<AtomicUsize>,
    }

    impl nmbrain::slot::Engine for MockEngine {
        fn embed(&self, text: &str) -> Result<Vec<f32>, nmtype::Fault> {
            self.embed_calls.fetch_add(1, Ordering::SeqCst);
            let h = nmhash::sha256_bytes(text.as_bytes());
            let mut v = vec![0.0_f32; 8];
            for (i, b) in h.bytes().take(8).enumerate() {
                v[i] = f32::from(b) / 255.0;
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            for x in &mut v {
                *x /= norm;
            }
            Ok(v)
        }
        fn rerank(&self, query: &str, doc: &str) -> Result<f32, nmtype::Fault> {
            self.rerank_calls.fetch_add(1, Ordering::SeqCst);
            let qterms: std::collections::BTreeSet<String> = nmlex::key_terms(query).into_iter().collect();
            let dlow = doc.to_lowercase();
            let hits = qterms.iter().filter(|t| dlow.contains(t.as_str())).count();
            #[expect(clippy::cast_precision_loss, reason = "test score")]
            Ok((hits as f32 / qterms.len().max(1) as f32).clamp(0.05, 0.95))
        }
        fn resident_mb(&self) -> (f32, f32) {
            (1.0, 0.0)
        }
        fn release_device(&self) {}
        fn label(&self) -> String {
            self.label.clone()
        }
    }

    fn copy_json_tree(src: &Path, dst: &Path) {
        std::fs::create_dir_all(dst).unwrap();
        for entry in std::fs::read_dir(src).unwrap().flatten() {
            let p = entry.path();
            let target = dst.join(entry.file_name());
            if p.is_dir() {
                copy_json_tree(&p, &target);
            } else if p.extension().is_some_and(|e| e == "json") {
                std::fs::copy(&p, &target).unwrap();
            }
        }
    }

    /// A working folder owned by this test process, `scratch/unit-tests/<tag>-<pid>-<ms>/`,
    /// removed when the test ends (also on a failed assertion). A fixed name under
    /// `tests/section-checks/` was cleared by any other process running the same tests in this
    /// copy (another lane, another target folder), which broke the copy into the test home.
    struct UnitDir(std::path::PathBuf);

    impl UnitDir {
        fn new(tag: &str) -> Self {
            let d = nmtype::nano_root().join("scratch").join("unit-tests").join(format!("{tag}-{}-{}", std::process::id(), nmtype::now_ms()));
            std::fs::create_dir_all(&d).unwrap();
            Self(d)
        }
    }

    impl std::ops::Deref for UnitDir {
        type Target = Path;
        fn deref(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for UnitDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn test_home(tag: &str, dense_off: bool) -> UnitDir {
        let dir = UnitDir::new(tag);
        let real = nmtype::nano_root();
        copy_json_tree(&real.join("designs"), &dir.join("designs"));
        std::fs::create_dir_all(dir.join("crates")).unwrap();
        for entry in std::fs::read_dir(real.join("crates")).unwrap().flatten() {
            let pj = entry.path().join("part.json");
            if pj.is_file() {
                let dest = dir.join("crates").join(entry.file_name());
                std::fs::create_dir_all(&dest).unwrap();
                std::fs::copy(&pj, dest.join("part.json")).unwrap();
            }
        }
        let settings = if dense_off { r#"{"dense_cache":"off"}"# } else { r#"{"dense_cache":"on"}"# };
        std::fs::write(dir.join("settings.json"), settings).unwrap();
        dir
    }

    fn setup_repo(dir: &Path, files: &[(&str, &str)]) {
        for (rel, body) in files {
            let p = dir.join(rel);
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(p, body).unwrap();
        }
    }

    fn bind_mocks(org: &Organism, label: &str) -> (Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let embed_calls = Arc::new(AtomicUsize::new(0));
        let rerank_calls = Arc::new(AtomicUsize::new(0));
        let eng = Arc::new(MockEngine {
            label: label.to_owned(),
            embed_calls: Arc::clone(&embed_calls),
            rerank_calls: Arc::clone(&rerank_calls),
        });
        org.slots.bind("embedder", Arc::clone(&eng) as Arc<dyn nmbrain::slot::Engine>).unwrap();
        org.slots.bind("reranker", eng as Arc<dyn nmbrain::slot::Engine>).unwrap();
        (embed_calls, rerank_calls)
    }

    fn add_episode(org: &Organism, id: &str, query: &str, path: &str, line: usize) {
        let sub = org.sub();
        let cite = sub
            .chunks
            .iter()
            .find(|c| c.start <= line && line <= c.end && sub.files.get(c.file).is_some_and(|f| f.path == path))
            .map(|c| format!("{path}:{} lines {}..{}", c.start, c.start, c.end))
            .unwrap_or_else(|| format!("{path}:{line} lines {line}..{line}"));
        let mut m = org.memory.lock().unwrap();
        let mut ep = nmmem::episodic::Episode::new("neuromorph", "test", query, &format!("see {path}:{line}"), vec![cite], sub.version);
        ep.id = id.to_owned();
        let _ = m.episodic.write(ep);
        let _ = m.episodic.verdict(id, true);
    }

    fn sample_files() -> Vec<(&'static str, &'static str)> {
        vec![
            ("src/alpha.rs", "pub fn alpha_calc() -> u32 {\n    10\n}\n"),
            ("src/beta.rs", "pub fn beta_parse() -> u32 {\n    20\n}\n"),
            ("src/gamma.rs", "pub fn gamma_store() -> u32 {\n    30\n}\n"),
            ("src/delta.rs", "pub fn delta_route() -> u32 {\n    40\n}\n"),
            ("src/epsilon.rs", "pub fn epsilon_flush() -> u32 {\n    50\n}\n"),
            ("src/zeta.rs", "pub fn zeta_compact() -> u32 {\n    60\n}\n"),
            ("src/uncited.rs", "pub fn uncited_helper() -> u32 {\n    70\n}\n"),
        ]
    }

    fn sample_episodes() -> Vec<(&'static str, &'static str, &'static str)> {
        vec![
            ("ep0", "where is alpha_calc SECRET_TOKEN_ALPHA defined?", "src/alpha.rs"),
            ("ep1", "where is beta_parse SECRET_TOKEN_BETA defined?", "src/beta.rs"),
            ("ep2", "where is gamma_store SECRET_TOKEN_GAMMA defined?", "src/gamma.rs"),
            ("ep3", "where is delta_route SECRET_TOKEN_DELTA defined?", "src/delta.rs"),
            ("ep4", "where is epsilon_flush SECRET_TOKEN_EPSILON defined?", "src/epsilon.rs"),
        ]
    }

    #[test]
    fn meaning_floor_short_circuits_below_null_min() {
        let dir = UnitDir::new("nmapi-session-null-min");
        let home = test_home("nmapi-session-null-min-home", false);
        let repo = dir.join("repo");
        let mem = dir.join("mem");
        setup_repo(&repo, &sample_files());
        let org = Organism::open_with(&home, &repo, &[], true, &mem).unwrap();
        let (_, rerank_calls) = bind_mocks(&org, "mock-null-min");
        for (id, q, path) in sample_episodes().into_iter().take(4) {
            add_episode(&org, id, q, path, 1);
        }
        let floors = {
            let m = org.memory.lock().unwrap();
            meaning_floor(&org, &m)
        };
        assert_eq!(floors, Floors { request: MEANING_FLOOR, material: MEANING_FLOOR });
        assert_eq!(rerank_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn meaning_caches_persist_and_reuse_across_fresh_organisms() {
        let dir = UnitDir::new("nmapi-session-persist");
        let home = test_home("nmapi-session-persist-home", false);
        let repo = dir.join("repo");
        let mem = dir.join("mem");
        setup_repo(&repo, &sample_files());

        let org1 = Organism::open_with(&home, &repo, &[], true, &mem).unwrap();
        let (e1, r1) = bind_mocks(&org1, "mock-persist-v1");
        for (id, q, path) in sample_episodes() {
            add_episode(&org1, id, q, path, 1);
        }
        let _ = recalled_chunks(&org1, "how does alpha_calc work?");
        assert!(e1.load(Ordering::SeqCst) >= 6);
        assert!(r1.load(Ordering::SeqCst) >= 10);

        let mdir = mem.join(MEANING_SUBDIR);
        assert!(mdir.join("vectors.bin").is_file());
        assert!(mdir.join("floors.json").is_file());
        assert!(mdir.join("rerank_pairs.bin").is_file());
        for fname in ["vectors.bin", "floors.json", "rerank_pairs.bin"] {
            let raw = std::fs::read(mdir.join(fname)).unwrap();
            let lossy = String::from_utf8_lossy(&raw);
            assert!(!lossy.contains("SECRET_TOKEN_"), "{fname} leaked query text");
            assert!(!lossy.contains("alpha_calc"), "{fname} leaked code text");
            assert!(!lossy.contains("src/alpha.rs"), "{fname} leaked file path");
        }

        // Fresh organism on the same repo & memory: embeds ONLY the new question, 0 decoy reranks.
        let org2 = Organism::open_with(&home, &repo, &[], true, &mem).unwrap();
        let (e2, r2) = bind_mocks(&org2, "mock-persist-v1");
        {
            let m = org2.memory.lock().unwrap();
            let _ = meaning_floor(&org2, &m);
        }
        assert_eq!(r2.load(Ordering::SeqCst), 0, "floors.json should avoid decoy reranks on fresh organism");
        let _ = recalled_chunks(&org2, "where is zeta_compact defined?");
        assert_eq!(e2.load(Ordering::SeqCst), 1, "only the new question should be embedded when episode vectors are loaded from disk");
        assert_eq!(r2.load(Ordering::SeqCst), 10, "floors.json skipped decoy floor reranks; only the new question's 10 candidate pairs ran");

        // Adding a 6th episode embeds 1 new episode (+1 question) and reranks <= 4 new boundary pairs.
        let org3 = Organism::open_with(&home, &repo, &[], true, &mem).unwrap();
        let (e3, r3) = bind_mocks(&org3, "mock-persist-v1");
        add_episode(&org3, "ep5", "where is zeta_compact SECRET_TOKEN_ZETA defined?", "src/zeta.rs", 1);
        {
            let m = org3.memory.lock().unwrap();
            let _ = meaning_floor(&org3, &m);
        }
        assert!(r3.load(Ordering::SeqCst) <= 4, "expected <= 4 boundary decoy reranks, got {}", r3.load(Ordering::SeqCst));
        r3.store(0, Ordering::SeqCst);
        let _ = recalled_chunks(&org3, "explain zeta_compact behavior");
        assert_eq!(e3.load(Ordering::SeqCst), 2, "1 new question + 1 new episode");
    }

    #[test]
    fn meaning_cache_invalidates_on_model_and_cited_file_change() {
        let dir = UnitDir::new("nmapi-session-invalidate");
        let home = test_home("nmapi-session-invalidate-home", false);
        let repo = dir.join("repo");
        let mem = dir.join("mem");
        setup_repo(&repo, &sample_files());

        let org1 = Organism::open_with(&home, &repo, &[], true, &mem).unwrap();
        bind_mocks(&org1, "mock-inv-A");
        for (id, q, path) in sample_episodes() {
            add_episode(&org1, id, q, path, 1);
        }
        let _ = recalled_chunks(&org1, "where is alpha_calc?");

        // Model label change invalidates caches.
        let org2 = Organism::open_with(&home, &repo, &[], true, &mem).unwrap();
        let (e2, r2) = bind_mocks(&org2, "mock-inv-B");
        let _ = recalled_chunks(&org2, "where is beta_parse?");
        assert!(e2.load(Ordering::SeqCst) >= 6);
        assert!(r2.load(Ordering::SeqCst) >= 10);

        // Uncited file change preserves floors (0 decoy reranks on meaning_floor).
        std::fs::write(repo.join("src/uncited.rs"), "pub fn uncited_helper() -> u32 {\n    999\n}\n").unwrap();
        let org3 = Organism::open_with(&home, &repo, &[], true, &mem).unwrap();
        let (_, r3) = bind_mocks(&org3, "mock-inv-B");
        {
            let m = org3.memory.lock().unwrap();
            let _ = meaning_floor(&org3, &m);
        }
        assert_eq!(r3.load(Ordering::SeqCst), 0, "uncited file edit must preserve floors");

        // Cited file change invalidates candidates_digest and triggers recomputation.
        std::fs::write(repo.join("src/alpha.rs"), "pub fn alpha_calc_v2() -> u32 {\n    11\n}\n").unwrap();
        let org4 = Organism::open_with(&home, &repo, &[], true, &mem).unwrap();
        let (_, r4) = bind_mocks(&org4, "mock-inv-B");
        {
            let m = org4.memory.lock().unwrap();
            let _ = meaning_floor(&org4, &m);
        }
        assert!(r4.load(Ordering::SeqCst) > 0, "cited file edit must recompute floors");

        // Corrupt cache files must not panic.
        let mdir = mem.join(MEANING_SUBDIR);
        std::fs::write(mdir.join("vectors.bin"), b"corrupt").unwrap();
        std::fs::write(mdir.join("floors.json"), b"{bad json").unwrap();
        std::fs::write(mdir.join("rerank_pairs.bin"), b"NMPARK1\0\x05\x00\x00\x00bad").unwrap();
        let org5 = Organism::open_with(&home, &repo, &[], true, &mem).unwrap();
        bind_mocks(&org5, "mock-inv-B");
        let _ = recalled_chunks(&org5, "where is gamma_store?");
    }

    #[test]
    fn meaning_cache_respects_memory_off_and_dense_cache_off() {
        let dir = UnitDir::new("nmapi-session-disabled");
        let home_on = test_home("nmapi-session-disabled-home-on", false);
        let home_off = test_home("nmapi-session-disabled-home-off", true);
        let repo = dir.join("repo");
        setup_repo(&repo, &sample_files());

        // Seed episodes once with memory on, then test memory_on = false.
        let mem1 = dir.join("mem1");
        let seed = Organism::open_with(&home_on, &repo, &[], true, &mem1).unwrap();
        for (id, q, path) in sample_episodes() {
            add_episode(&seed, id, q, path, 1);
        }
        drop(seed);
        let org_mem_off = Organism::open_with(&home_on, &repo, &[], false, &mem1).unwrap();
        bind_mocks(&org_mem_off, "mock-mem-off");
        let _ = recalled_chunks(&org_mem_off, "where is alpha_calc?");
        assert!(!mem1.join(MEANING_SUBDIR).exists(), "memory_on=false must not write meaning/");

        // Now test dense_cache = off with memory_on = true.
        let mem2 = dir.join("mem2");
        let org_dense_off = Organism::open_with(&home_off, &repo, &[], true, &mem2).unwrap();
        bind_mocks(&org_dense_off, "mock-dense-off");
        for (id, q, path) in sample_episodes() {
            add_episode(&org_dense_off, id, q, path, 1);
        }
        let _ = recalled_chunks(&org_dense_off, "where is alpha_calc?");
        assert!(!mem2.join(MEANING_SUBDIR).exists(), "dense_cache=off must not write meaning/");
    }

    #[test]
    fn meaning_floor_skips_material_decoys_below_null_min() {
        let dir = UnitDir::new("nmapi-session-mat-min");
        let home = test_home("nmapi-session-mat-min-home", false);
        let repo = dir.join("repo");
        let mem = dir.join("mem");
        setup_repo(&repo, &sample_files());
        let org = Organism::open_with(&home, &repo, &[], true, &mem).unwrap();
        let (_, rerank_calls) = bind_mocks(&org, "mock-mat-min");
        add_episode(&org, "ep0", "where is alpha_calc?", "src/alpha.rs", 1);
        add_episode(&org, "ep1", "where is beta_parse?", "src/beta.rs", 1);
        // Remaining 3 cite missing files so lines == None (< NULL_MIN readable citations).
        add_episode(&org, "ep2", "where is missing_one?", "src/missing1.rs", 1);
        add_episode(&org, "ep3", "where is missing_two?", "src/missing2.rs", 1);
        add_episode(&org, "ep4", "where is missing_three?", "src/missing3.rs", 1);

        let floors = {
            let m = org.memory.lock().unwrap();
            meaning_floor(&org, &m)
        };
        assert!((floors.material - MEANING_FLOOR).abs() < 1e-6);
        // Only 5 request decoy pairs should be scored (0 material decoy pairs).
        assert_eq!(rerank_calls.load(Ordering::SeqCst), 5);
    }
}
