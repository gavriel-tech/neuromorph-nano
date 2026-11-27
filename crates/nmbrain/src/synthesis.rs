//! Synthesis: reduce to a cited pack within the token budget (guide 3, 6.4, 17.4; PLAN 4c,
//! from `gcntxt` reduce). Rules, in order: packets without evidence are refused; packets the
//! reranker scores below the relevance floor for the question are dropped (it orders, and the
//! floor keeps unrelated material out); agreeing packets under one key merge their evidence;
//! conflicting ones are both kept; order by score times the evidence-kind weight times the
//! source weight (code before plan and note documents, 2026-11-08); fill the budget, skipping
//! a packet too large for what is left rather than stopping; withhold when nothing observed,
//! verified or conflicting-but-observed remains and no cited source chunk passed the floor
//! (such a pack answers from the chunks, labelled Inferred). A pack whose packets all fall below
//! the floor is served marked low (`floor_met` false) when any of them holds a question term;
//! only a pack with no such packet is withheld.
use crate::evidence::{EvidenceSet, SpanEvidence};
use crate::slot::Slots;
use nmtype::{Pack, Packet, Status};
use std::collections::BTreeMap;

/// Relevance floor on the reranker score (exact control: key-term coverage blend).
/// Applied before the evidence-kind weight below.
pub const FLOOR: f32 = 0.5;

/// Rarity-weighted share of a question's key terms that no file of the tree contains, at or
/// above which the pack is withheld (ported 2026-11-09 from the earlier brain's rule, which
/// held decoys on the sealed C1 sets; the live path lacked it, so a question mixing real
/// subject words with an absent core concept was answered from the subject words).
pub const ABSENT_SHARE: f32 = 0.25;

/// Withhold a pack when a weighty part of the question is absent from the whole tree:
/// `absent` are those terms, `share` their rarity-weighted share of the question.
#[must_use]
pub fn withhold_uncovered(mut pack: Pack, absent: &[String], share: f32) -> Pack {
    if pack.withheld || absent.is_empty() || share < ABSENT_SHARE {
        return pack;
    }
    pack.withheld = true;
    pack.packets.clear();
    pack.reason = format!("withheld: the repository does not contain [{}] ({share:.2} of the question by rarity)", absent.join(", "));
    pack
}

/// Ordering weight by evidence kind: definitions and dependency facts first, then verified
/// build records, line matches (grep, test lines, uses), history, and semantic chunks last. It
/// orders only; nothing is dropped by it. (2026-11-09: the `test`, `uses` and `build` kinds of
/// the new readers fell to the chunk weight.)
#[must_use]
pub fn weight(kind: &str) -> f32 {
    match kind {
        "symbol" | "deps" | "recall" => 1.0,
        "build" => 0.9,
        "grep" | "test" | "uses" | "rule" | "interface" => 0.85,
        "history" => 0.8,
        _ => 0.7,
    }
}

/// Evidence kinds that answer a focus job's kind of question.
#[must_use]
pub fn focus_kinds(job: &str) -> &'static [&'static str] {
    match job {
        "historical" => &["history"],
        "execution" => &["build"],
        "verification" => &["test"],
        "dependency" => &["uses", "deps"],
        "constraint" => &["rule"],
        "functional" => &["interface"],
        _ => &[],
    }
}

/// Ordering boost for evidence of the kind the question asks for (a history, build, test or
/// uses question, as the router reads it).
pub const FOCUS_BOOST: f32 = 1.5;
/// Least relevance of evidence a focus job picked for the question's kind.
pub const FOCUS_FLOOR: f32 = 0.8;

/// Key prefix of chunks placed by the dense lane (`nmsubs` views; named here so nmbrain does not
/// depend on nmsubs). The same chunk read lexically (`chunk:`) merges into it.
pub const DENSE_KEY: &str = "dense:";
/// Ordering weight of a dense lane chunk (a plain chunk weighs 0.7).
pub const DENSE_WEIGHT: f32 = 1.0;
/// Relevance of the lane's first fused place, and the fall per place (floored at `FLOOR`).
pub const DENSE_TOP: f32 = 0.95;
pub const DENSE_STEP: f32 = 0.02;

/// Key prefix of chunks the tentacle windows judged (`nmapi::tentacles`, 2026-11-10): the
/// place is the judged order over every window's candidates. The same chunk placed by the lane
/// or read lexically merges into it.
pub const TENTACLE_KEY: &str = "tentacle:";
/// Relevance of the tentacle pass's first judged place (one step above the lane's first, so a
/// chunk the windows read in full orders ahead of the lane's place for it); the fall per place
/// is `DENSE_STEP`.
pub const TENTACLE_TOP: f32 = 0.97;

/// The evidence prefix that ties a thin wrapper's callee to the wrapper (`nmsubs` views write
/// `called-by:<wrapper's source> <callee's first line>`; named here so nmbrain does not depend on
/// nmsubs).
pub const CALLED_BY: &str = "called-by";
/// Evidence prefix of a code piece's fields (`nmsubs::views::PIECE`; named here so nmbrain does
/// not depend on the substrate): read with the piece's text by the judge, never shown.
pub const PIECE: &str = "piece:";
/// Marks the `PIECE` line that holds the file's field (`nmsubs::views::FILE_FIELD`).
pub const FILE_FIELD: &str = "file ";

/// The wrapper source a packet's `called-by:` evidence names.
fn called_by(p: &Packet) -> Option<&str> {
    p.get_evidence().iter().find_map(|e| e.strip_prefix(CALLED_BY)?.strip_prefix(':')?.split(' ').next())
}

/// The place of a ranked-reader chunk key (`dense:<place>:<source>` or
/// `tentacle:<place>:<source>`) with the relevance of that reader's first place.
fn ranked_place(key: &str) -> Option<(usize, f32)> {
    let (rest, top) = match key.strip_prefix(DENSE_KEY) {
        Some(r) => (r, DENSE_TOP),
        None => (key.strip_prefix(TENTACLE_KEY)?, TENTACLE_TOP),
    };
    Some((rest.split_once(':')?.0.parse().ok()?, top))
}
/// The key a chunk merges under: a ranked-reader chunk and the same chunk read lexically are one.
fn chunk_key(key: &str) -> String {
    match key.strip_prefix(DENSE_KEY).or_else(|| key.strip_prefix(TENTACLE_KEY)).and_then(|r| r.split_once(':')) {
        Some((_, src)) => format!("chunk:{src}"),
        None => key.to_owned(),
    }
}

/// Rough token count of a packet as it would be delivered.
#[must_use]
pub fn tokens(p: &Packet) -> u32 {
    let chars = p.text().len() + p.key().len() + p.get_evidence().iter().map(String::len).sum::<usize>();
    u32::try_from(chars / 4 + 8).unwrap_or(u32::MAX)
}

/// Candidates the neural reranker orders per question (runbook S6: at most 12).
pub const NEURAL_CAP: usize = 12;

/// With a neural reranker bound, the first `NEURAL_CAP` admitted candidates (control order)
/// are scored by the neural reranker; when the slot is qualified they are reordered by that
/// score, otherwise the order is recorded beside the control's (shadow); the rest keep the
/// control order. The control's floor
/// already decided admission, so the neural score orders only (law 6).
fn neural_order(question: &str, ordered: Vec<(f32, Packet)>, slots: &Slots) -> Vec<(f32, Packet)> {
    let cap = slots.rerank_cap(NEURAL_CAP).min(ordered.len());
    if cap == 0 || !slots.by_name.get("reranker").is_some_and(|s| s.engine().is_some()) {
        return ordered;
    }
    let mut head: Vec<(f32, f32, Packet)> = Vec::with_capacity(cap);
    let mut rest = Vec::new();
    // A set-ordering engine (a switched-on learned readout) scores the head in one call.
    let docs: Vec<String> = ordered.iter().take(cap).map(|(_, p)| format!("{} {}", p.key(), p.text())).collect();
    let set = slots.rerank_set_neural(question, &docs);
    for (i, (s, p)) in ordered.into_iter().enumerate() {
        if i < cap {
            let n = match set.as_ref().and_then(|v| v.get(i)) {
                Some(x) => *x,
                None => slots.rerank_neural(question, &format!("{} {}", p.key(), p.text())).unwrap_or(f32::MIN),
            };
            head.push((n, s, p));
        } else {
            rest.push((s, p));
        }
    }
    let best = head.iter().map(|h| h.0).fold(f32::MIN, f32::max);
    if best > f32::MIN {
        slots.note_best(best);
    }
    let control_top = head.first().map(|h| h.2.id());
    let mut neural = head.clone();
    neural.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| b.1.total_cmp(&a.1)));
    slots.note_shadow("reranker", control_top == neural.first().map(|h| h.2.id()));
    // Law 8 (plan owner ruling, 2026-11-07 23:56 UTC): a neural part that does not beat its
    // exact control stays out of the default path. Unqualified, the neural order is only
    // recorded; the control order is kept.
    let kept = if slots.qualified("reranker") { neural } else { head };
    kept.into_iter().map(|(_, s, p)| (s, p)).chain(rest).collect()
}

/// Reduce labelled packets for one question.
#[must_use]
pub fn reduce(task: u64, question: &str, packets: &[Packet], budget_tokens: u32, slots: &Slots) -> Pack {
    reduce_inner(task, question, packets, budget_tokens, slots, None)
}

/// Relevance of a tentacle chunk from its evidence record (2026-11-10, refinement step 1, item 1):
/// G1's combined value as a share of the best forwarded record's, on the range a ranked reader's
/// chunk has (`FLOOR` to `TENTACLE_TOP`). Nothing here reads the chunk's place in a list.
#[must_use]
pub fn evidence_relevance(set: &EvidenceSet, rec: &SpanEvidence) -> f32 {
    let top = set.forwarded.iter().map(|e| e.combined).fold(0.0_f32, f32::max);
    if top > 0.0 && rec.combined.is_finite() { FLOOR + (TENTACLE_TOP - FLOOR) * (rec.combined / top).clamp(0.0, 1.0) } else { FLOOR }
}

/// The record a packet stands for: its source `path:line`, path with forward slashes.
fn record_of<'a>(set: &'a EvidenceSet, p: &Packet) -> Option<&'a SpanEvidence> {
    let (path, line) = p.provenance().source.rsplit_once(':')?;
    set.at(&path.replace('\\', "/"), line.parse().ok()?)
}

/// A packet as a low-priority queue record (a filter would have dropped it): its source span,
/// its key as the reference, the reducer's relevance as its score.
fn low_record(p: &Packet, relevance: f32) -> SpanEvidence {
    let (path, line) = p
        .provenance()
        .source
        .rsplit_once(':')
        .map_or((p.provenance().source.clone(), 0), |(a, b)| (a.replace('\\', "/"), b.parse().unwrap_or(0)));
    SpanEvidence {
        span: crate::g0::SpanRef { file: path, start: line, end: line, key: p.key().to_owned() },
        facts: crate::evidence::SpanFacts {
            tokens: tokens(p) as usize,
            scores: vec![("reducer".into(), relevance)],
            ..crate::evidence::SpanFacts::default()
        },
        found_by: vec![p.provenance().origin.to_string()],
        combined: relevance,
        ..SpanEvidence::default()
    }
}

/// `reduce` on the new path (2026-11-10, refinement step 1, items 1 and 5): a tentacle chunk is
/// ordered by its evidence record (`evidence_relevance`) instead of its place; a packet below the
/// relevance floor, or skipped for the token budget, is not dropped but held in the set's queue
/// with a low-priority mark (a forwarded record goes back to the queue, anything else is added).
/// Everything else is as `reduce`.
#[must_use]
pub fn reduce_evidence(task: u64, question: &str, packets: &[Packet], budget_tokens: u32, slots: &Slots, set: &mut EvidenceSet) -> Pack {
    reduce_inner(task, question, packets, budget_tokens, slots, Some(set))
}

#[expect(clippy::too_many_lines, reason = "one reduce step in order; the evidence branches are marked")]
fn reduce_inner(task: u64, question: &str, packets: &[Packet], budget_tokens: u32, slots: &Slots, ev: Option<&mut EvidenceSet>) -> Pack {
    let mut low: Vec<(f32, Packet, &'static str)> = Vec::new();
    let mut refused = 0;
    let mut scored: Vec<(f32, Packet)> = Vec::new();
    let mut cited: Vec<&Packet> = Vec::new();
    for p in packets.iter().filter(|p| p.kind() != "question") {
        if p.get_evidence().is_empty() {
            refused += 1;
        } else {
            cited.push(p);
        }
    }
    // A code piece is judged with its fields (`PIECE` evidence), in every part: its own (name
    // path and signature) and its file's (the file's notes, shared by the file's pieces).
    let fields = |p: &Packet| -> String {
        p.get_evidence().iter().filter_map(|e| e.strip_prefix(PIECE)).map(|f| f.strip_prefix(FILE_FIELD).unwrap_or(f)).collect::<Vec<_>>().join(" ")
    };
    let docs: Vec<String> = cited.iter().map(|p| format!("{} {} {}", p.key(), fields(p), p.text())).collect();

    let wanted: Vec<&str> = slots.route(question).iter().flat_map(|j| focus_kinds(j).iter().copied()).collect();
    // Packets below the floor that hold at least one of the question's terms: served with a
    // low-confidence mark when nothing reaches the floor, instead of withholding a real answer.
    let mut terms = crate::slot::coverage_terms(question);
    let material = slots.material();
    let unmatched = material.unmatched.clone();
    terms.retain(|t| !unmatched.contains(t));
    let mut below: Vec<(f32, Packet)> = Vec::new();
    // Document passages are scored with the question's document terms, code exactly as before.
    let is_doc: Vec<bool> = cited.iter().map(|p| nmtype::is_document(&p.provenance().source)).collect();
    let score_of = |docs: &[String], m: &crate::slot::Material| {
        let mut rer = slots.rerank_all_with(question, docs, false, m);
        if is_doc.iter().any(|d| *d) {
            let dd: Vec<String> = docs.iter().zip(&is_doc).filter(|(_, d)| **d).map(|(x, _)| x.clone()).collect();
            let mut ds = slots.rerank_all_with(question, &dd, true, m).into_iter();
            for (r, d) in rer.iter_mut().zip(&is_doc) {
                if *d && let Some(x) = ds.next() {
                    *r = x;
                }
            }
        }
        rer
    };
    let score = |m: &crate::slot::Material| score_of(&docs, m);
    // The drones' judgment of a packet, when they gave one: their calibrated evidence in place of
    // the share of the question's words its text holds.
    let judged = slots.judged();
    let by_drones = |p: &Packet, s: f32| judged.get(&p.id()).copied().unwrap_or(s);
    let rer: Vec<f32> = score(&material).into_iter().zip(&cited).map(|(s, p)| by_drones(p, s)).collect();
    // With terms the material holds in no form left out, the question as asked is still scored:
    // a pack none of whose packets reaches the floor on it is served marked low.
    let asked: Vec<f32> = if material.is_plain() || !judged.is_empty() {
        rer.clone()
    } else {
        score(&crate::slot::Material::default())
    };
    let mut asked_ok: std::collections::BTreeSet<(String, String)> = std::collections::BTreeSet::new();
    let doc_kind = nmlex::doc_kind(question).is_some();
    for (((p, s), doc), a) in cited.into_iter().zip(rer).zip(&docs).zip(asked) {
        let place = ranked_place(p.key()).filter(|_| p.kind() == "chunk");
        let dense = place.is_some();
        let lift = |s: f32| {
            // M1 revision 2 (2026-11-09): a `recall` packet is a fresh read of a line an earlier,
            // verified answer cited; the episode match that chose it is its relevance (carried as
            // its confidence), so a reworded question does not drop it for lack of shared words.
            // A table row chosen by the question's column and extreme carries that choice as its
            // confidence, as a recall does: its text holds values, not the question's words.
            let s = if p.kind() == "recall" || p.kind() == "row" { s.max(p.get_confidence()) } else { s };
            // The dense lane (2026-11-10): a chunk the question's fused ranking placed (embedding
            // and lexical order joined) is admitted and ordered by that place, carried as its
            // confidence, so code described in other words than the question's is not dropped for
            // lack of them, and the fused order is kept.
            // The tentacle windows (2026-11-10) are a second ranked reader, admitted the same way
            // by their judged place.
            #[expect(clippy::cast_precision_loss, reason = "places are small")]
            let s = place.map_or(s, |(n, top)| (top - DENSE_STEP * n as f32).max(FLOOR));
            // New path: a tentacle chunk takes its relevance from its evidence record, not its place.
            let s = match ev.as_deref() {
                Some(set) if p.key().starts_with(TENTACLE_KEY) && p.kind() == "chunk" => {
                    record_of(set, p).map_or(s, |r| evidence_relevance(set, r))
                }
                _ => s,
            };
            // Evidence a focus job picked for this kind of question (history of a file the question
            // names, its build records, test lines, uses) counts as relevant (`FOCUS_FLOOR`): the job
            // chose it by the question's names, which its text may not repeat (2026-11-09).
            if wanted.contains(&p.kind()) { s.max(FOCUS_FLOOR) } else { s }
        };
        let s = lift(s);
        // The question as asked (every term counted), lifted the same way.
        if lift(a) >= FLOOR {
            asked_ok.insert((chunk_key(p.key()), p.text().to_owned()));
        }
        if s >= FLOOR {
            let boost = if wanted.contains(&p.kind()) { FOCUS_BOOST } else { 1.0 };
            let w = if dense { DENSE_WEIGHT } else { weight(p.kind()) };
            scored.push((s * w * boost * nmtype::source_weight_q(&p.provenance().source, doc_kind), p.clone()));
        } else {
            let edge = p.kind() == "trace" || called_by(p).is_some();
            if edge || crate::slot::coverage_in(&terms, doc, material.form_weight) > 0.0 {
                below.push((s, p.clone()));
            }
            if ev.is_some() {
                low.push((s, p.clone(), "held low: below the reducer's relevance floor"));
            }
        }
    }
    let mut merged: BTreeMap<(String, String), (f32, Packet)> = BTreeMap::new();
    for (s, p) in scored {
        let slot = (chunk_key(p.key()), p.text().to_owned());
        match merged.remove(&slot) {
            Some((s0, p0)) => {
                // The higher-scored reading keeps its packet (a lane chunk over the same chunk
                // read lexically); evidence from both is kept.
                let (keep, other) = if s > s0 { (p, p0) } else { (p0, p) };
                let extra: Vec<String> = other.get_evidence().iter().filter(|e| !keep.get_evidence().contains(e)).cloned().collect();
                merged.insert(slot, (s0.max(s), keep.evidence(extra)));
            }
            None => {
                merged.insert(slot, (s, p));
            }
        }
    }
    let mut ordered: Vec<(f32, Packet)> = merged.into_values().collect();
    ordered.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.key().cmp(b.1.key())));
    slots.note_best(-1.0);
    let ordered = neural_order(question, ordered, slots);
    // 2026-11-10: with a withhold floor set (`serve --rerank-withhold`, its own decision,
    // qualified apart from ordering), a pack whose best candidate the bound reranker scores
    // below the floor is withheld (no candidate judged relevant by meaning). The scores come
    // from the engine whether or not it orders (shadow).
    // Below that floor the pack is served marked low; it is withheld only when it holds nothing.
    let weak = slots.withhold_floor().and_then(|floor| {
        let best = slots.last_best();
        (best >= 0.0 && best < floor).then(|| format!("no candidate judged relevant by the reranker (best {best:.3} below {floor})"))
    });
    let mut used = 0_u32;
    let mut kept = Vec::new();
    for (s, p) in ordered {
        let t = tokens(&p);
        if used + t > budget_tokens {
            // A packet larger than what is left is skipped; smaller ones after it still fit.
            if ev.is_some() {
                low.push((s, p, "held low: beyond the pack's token room"));
            }
            continue;
        }
        used += t;
        kept.push(p);
    }
    // New path: nothing the reducer passed over is lost; it goes to the queue, marked low.
    if let Some(set) = ev {
        for (s, p, why) in &low {
            let key = record_of(set, p).map(|r| r.span.key.clone());
            match key {
                Some(k) => {
                    set.mark_low(&|e: &SpanEvidence| e.span.key == k, why, Vec::new());
                }
                None => {
                    set.mark_low(&|_: &SpanEvidence| false, why, vec![low_record(p, *s)]);
                }
            }
        }
    }
    let observed = kept.iter().any(|p| matches!(p.get_status(), Status::Observed | Status::Verified | Status::Contradicted));
    // Never an empty pack for a question the repo answers (2026-11-08): with no observed line
    // above the floor, source chunks above it still answer, labelled Inferred. They are cited
    // file lines, so the evidence is there; the label says relevance was not confirmed by a
    // reader. Decoys stay withheld: none of their packets reaches the floor.
    let chunks = !observed && kept.iter().any(|p| p.kind() == "chunk" && p.get_status() == Status::Inferred);
    let grounded = observed || chunks;
    let conflicts = kept.iter().filter(|p| p.get_status() == Status::Contradicted).count();
    if let Some(why) = weak {
        return if kept.is_empty() {
            Pack { task, packets: Vec::new(), withheld: true, reason: format!("withheld: {why}"), tokens: 0, floor_met: true }
        } else {
            Pack {
                task,
                packets: kept,
                withheld: false,
                reason: format!("low confidence: {why}; read the cited lines before relying on them"),
                tokens: used,
                floor_met: false,
            }
        };
    }
    if grounded && !material.is_plain() && !kept.iter().any(|p| asked_ok.contains(&(chunk_key(p.key()), p.text().to_owned()))) {
        // Packets reached the floor only once measured against the material (terms it lacks left
        // out, stems counted): served, marked low, so the reader knows the question as asked is
        // not met word for word.
        let lacks = if unmatched.is_empty() { String::new() } else { format!("; the material holds no form of [{}]", unmatched.join(", ")) };
        let reason = format!(
            "low confidence: no packet reached the relevance floor {FLOOR} on the question as asked{lacks}; read the cited lines before relying on them"
        );
        return Pack { task, packets: kept, withheld: false, reason, tokens: used, floor_met: false };
    }
    if grounded {
        complete(&mut kept, &mut used, std::mem::take(&mut below), &terms, budget_tokens);
        pointer_last(&mut kept);
    }
    if !grounded {
        // Nothing observed reached the floor: what holds the question's terms is still served,
        // ordered by score and marked low; with no such packet the pack is withheld.
        below.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.key().cmp(b.1.key())));
        let best = below.first().map_or(0.0, |b| b.0);
        for (_, p) in below {
            let t = tokens(&p);
            if used + t <= budget_tokens {
                used += t;
                kept.push(p);
            }
        }
        if !kept.is_empty() {
            let reason = format!(
                "low confidence: no packet reached the relevance floor {FLOOR} (best {best:.2}); read the cited lines before relying on them"
            );
            return Pack { task, packets: kept, withheld: false, reason, tokens: used, floor_met: false };
        }
    }
    let reason = if !grounded {
        format!("withheld: nothing observed or verified for the question ({refused} uncited packets refused)")
    } else if chunks {
        "answered from source chunks only (inferred: read the cited lines before relying on them)".into()
    } else if conflicts > 0 {
        format!("answered with {conflicts} conflicting observations kept, not merged")
    } else {
        "answered from observed packets".into()
    };
    Pack { task, packets: if grounded { kept } else { Vec::new() }, withheld: !grounded, reason, tokens: used, floor_met: true }
}

/// Packets with their reducer scores.
type Scored = Vec<(f32, Packet)>;

/// The file a packet cites (its source without the line).
fn file_of(p: &Packet) -> &str {
    let s = p.provenance().source.as_str();
    s.rsplit_once(':').filter(|(_, l)| l.chars().all(|c| c.is_ascii_digit())).map_or(s, |(f, _)| f)
}

/// The lines a packet cites: the `lines a..b` of its evidence, else its source line.
fn lines_of(p: &Packet) -> (usize, usize) {
    let at = p.provenance().source.rsplit_once(':').and_then(|(_, l)| l.parse().ok()).unwrap_or(0);
    p.get_evidence()
        .iter()
        .find_map(|e| {
            let (a, b) = e.split_once(" lines ")?.1.split_whitespace().next()?.split_once("..")?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((at, at))
}

/// True when every line of `text` is a file note (an inner doc comment, `//!` or `/*!`, which
/// describes the whole file) or blank. A doc comment of one definition (`///`) is specific.
fn is_note(text: &str) -> bool {
    text.lines().map(str::trim_start).all(|l| l.is_empty() || l.starts_with("//!") || l.starts_with("/*!"))
}

/// A code file's opening window (its notes and imports) is a pointer to the file: it names what
/// the whole file is about, so it matches a question in words best, while the lines that decide
/// sit further down. When the pack holds a file's opening window and also a later, separate packet
/// of the same file, the best such packet takes the place of the first packet that lies inside the
/// opening window (the window itself, or a file note line in it that the reply merges into it; a
/// code line there, such as a constant, or a definition's doc comment, is specific and keeps its
/// place), and that packet follows. Documents are cut by their sections and keep their order.
fn pointer_last(kept: &mut Vec<Packet>) {
    let mut openings: BTreeMap<String, usize> = BTreeMap::new();
    for p in kept.iter() {
        let src = &p.provenance().source;
        if p.kind() == "chunk" && src.ends_with(":1") && !nmtype::is_document(src) {
            openings.insert(file_of(p).to_owned(), lines_of(p).1);
        }
    }
    let mut i = 0;
    while i < kept.len() {
        let Some(p) = kept.get(i) else { break };
        let file = file_of(p).to_owned();
        let opening = p.kind() == "chunk" && p.provenance().source.ends_with(":1");
        if let Some(end) = openings.get(&file).copied()
            && lines_of(p).1 <= end
            && (opening || is_note(p.text()))
        {
            openings.remove(&file);
            if let Some(j) = (i + 1..kept.len()).find(|j| kept.get(*j).is_some_and(|k| file_of(k) == file && lines_of(k).0 > end + 1)) {
                let deeper = kept.remove(j);
                kept.insert(i, deeper);
            }
        }
        i += 1;
    }
}

/// Packets below the floor that complete a grounded pack: from a file the pack already cites,
/// holding a question term that the pack's packets of that file do not hold. Each is placed right
/// after the last kept packet of its file, best score first, while the token room lasts. A
/// question's terms are often split over a file (its notes name the topic, a definition further
/// down holds the deciding lines); the floor scores each packet alone, so the second half fell
/// below it although the first half had found the file.
fn complete(kept: &mut Vec<Packet>, used: &mut u32, below: Vec<(f32, Packet)>, terms: &[String], budget: u32) {
    if terms.is_empty() || below.is_empty() {
        return;
    }
    // Call path hops keep the order the path met them, after the rest (best score first).
    let (trace, mut below): (Scored, Scored) = below.into_iter().partition(|(_, p)| p.kind() == "trace");
    below.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.key().cmp(b.1.key())));
    below.extend(trace);
    let holds = |p: &Packet| -> Vec<usize> {
        let d = format!("{} {}", p.key(), p.text()).to_lowercase();
        terms.iter().enumerate().filter(|(_, t)| d.contains(t.as_str())).map(|(i, _)| i).collect()
    };
    // What the pack holds per file; a file's opening window (its notes, which name the topic of
    // the whole file) marks the file as found but holds no terms, so the definitions that decide
    // can still complete it.
    let mut held: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for p in kept.iter() {
        let opening = p.kind() == "chunk" && p.provenance().source.rsplit_once(':').is_some_and(|(_, l)| l == "1");
        let e = held.entry(file_of(p).to_owned()).or_default();
        if !opening {
            e.extend(holds(p));
        }
    }
    for (_, p) in below {
        let file = file_of(&p).to_owned();
        // Documents are cut along their sections already, each passage with its heading path;
        // completion is for code, where notes and definitions sit apart.
        if nmtype::is_document(&p.provenance().source) || kept.iter().any(|k| k.key() == p.key() && k.text() == p.text()) {
            continue;
        }
        let has = holds(&p);
        let t = tokens(&p);
        // The function a kept thin wrapper calls (`called-by:` evidence naming the wrapper): the
        // wrapper's notes name the topic, the callee holds the lines that decide.
        let caller = called_by(&p).map(str::to_owned);
        let linked = caller.as_ref().is_some_and(|c| kept.iter().any(|k| k.provenance().source == *c));
        // A hop on a call path completes the pack it was followed for: the edge, not a shared word,
        // is its evidence. It goes after the packets it was followed from, in the order the path met
        // it (a wrapper's callee right after the wrapper); when the room is spent, the last-placed
        // line windows and passages (inferred, not on the path) give way to it.
        if p.kind() == "trace" || linked {
            while *used + t > budget {
                let Some(i) = kept.iter().rposition(|k| k.kind() == "chunk" && k.get_status() == Status::Inferred) else { break };
                *used = used.saturating_sub(tokens(&kept.remove(i)));
            }
            if *used + t > budget {
                continue;
            }
            held.entry(file).or_default().extend(has);
            match caller.and_then(|c| kept.iter().position(|k| k.provenance().source == c)) {
                Some(i) => kept.insert(i + 1, p),
                None => kept.push(p),
            }
            *used += t;
            continue;
        }
        if *used + t > budget {
            continue;
        }
        match held.get_mut(&file) {
            Some(have) => {
                let new: Vec<usize> = has.iter().copied().filter(|i| !have.contains(i)).collect();
                if new.is_empty() {
                    continue;
                }
                have.extend(new);
                let at = kept.iter().rposition(|k| file_of(k) == file).map_or(kept.len(), |i| i + 1);
                kept.insert(at, p);
            }
            None => continue,
        }
        *used += t;
    }
}

/// Self-check of the `synthesis` sub-component: a decoy is withheld, two conflicting values
/// both survive, an uncited packet is refused, agreeing packets merge.
pub(crate) fn check() -> Result<(), String> {
    use nmtype::Provenance;
    let slots = Slots::default();
    let mk = |key: &str, text: &str, src: &str, st| {
        Packet::new("symbol", key, text, Provenance::new(src, "x", 1, 0)).status(st).evidence(vec![src.to_owned()])
    };
    let all = vec![
        mk("value:LIMIT", "LIMIT = 4", "a.rs:1", Status::Contradicted),
        mk("value:LIMIT", "LIMIT = 9", "b.rs:1", Status::Contradicted),
        mk("value:LIMIT", "LIMIT = 9", "c.rs:1", Status::Contradicted),
        Packet::new("symbol", "value:LIMIT", "LIMIT = 1", Provenance::new("unit:x", "x", 1, 0)),
    ];
    let p = reduce(1, "What is LIMIT set to?", &all, 4000, &slots);
    if p.withheld || p.packets.len() != 2 || p.packets.iter().map(|x| x.get_evidence().len()).sum::<usize>() != 3 {
        return Err(format!("nmbrain.synthesis: conflict pack {:?} {}", p.packets.iter().map(Packet::text).collect::<Vec<_>>(), p.reason));
    }
    let d = reduce(2, "What is DRONE_FANOUT_CAP set to?", &all, 4000, &slots);
    if !d.withheld || !d.packets.is_empty() {
        return Err("nmbrain.synthesis: decoy not withheld".into());
    }
    // Chunks only: a cited chunk above the floor answers, labelled Inferred; one below it
    // does not.
    let chunk = |text: &str, src: &str| {
        Packet::new("chunk", &format!("chunk:{src}"), text, Provenance::new(src, "x", 1, 0))
            .status(Status::Inferred)
            .evidence(vec![src.to_owned()])
    };
    let c = reduce(3, "where is the LIMIT check", &[chunk("// the LIMIT check runs here", "a/x.rs:1")], 4000, &slots);
    if c.withheld || c.packets.first().map(Packet::get_status) != Some(Status::Inferred) {
        return Err(format!("nmbrain.synthesis: chunk-only pack withheld ({})", c.reason));
    }
    if !reduce(4, "where is the zqxv wobble", &[chunk("// the LIMIT check runs here", "a/x.rs:1")], 4000, &slots).withheld {
        return Err("nmbrain.synthesis: unrelated chunk answered".into());
    }
    // Below the floor but sharing a question term: served, marked low, not withheld.
    let weak =
        reduce(4, "where does the zqxv wobble frame meet the LIMIT", &[chunk("// the LIMIT check runs here", "a/x.rs:1")], 4000, &slots);
    if weak.withheld || weak.floor_met || weak.packets.len() != 1 || !weak.reason.starts_with("low confidence") {
        return Err(format!(
            "nmbrain.synthesis: below-floor pack withheld {} floor_met {} ({})",
            weak.withheld, weak.floor_met, weak.reason
        ));
    }
    if !c.floor_met || !p.floor_met {
        return Err("nmbrain.synthesis: a pack above the floor must say the floor was met".into());
    }
    // Completion: below the floor, a packet of a file the pack cites that holds a question term
    // the pack's packets of that file lack is kept right after them; one of an uncited file, or
    // one adding nothing, is not.
    let q = "where does the LIMIT check reject the wobble frame";
    let anchor = chunk("// the LIMIT check reject frame path runs here", "a/x.rs:5");
    let adds = chunk("fn wobble() {}", "a/x.rs:40");
    let nothing = chunk("// LIMIT", "a/x.rs:80");
    let elsewhere = chunk("fn wobble() {}", "b/y.rs:3");
    let k = reduce(6, q, &[anchor, elsewhere, nothing, adds], 4000, &slots);
    let srcs: Vec<&str> = k.packets.iter().map(|p| p.provenance().source.as_str()).collect();
    if k.withheld || srcs != ["a/x.rs:5", "a/x.rs:40"] {
        return Err(format!("nmbrain.synthesis: completion kept {srcs:?} ({})", k.reason));
    }
    // The function a kept wrapper calls is kept right after it although it holds no question term.
    let wrapper = chunk("// the LIMIT check reject frame path runs here", "a/w.rs:5");
    let callee = Packet::new("definition", "symbol:inner@a/w.rs", "fn inner() { let z = 1; }", Provenance::new("a/w.rs:40", "x", 1, 0))
        .status(Status::Observed)
        .evidence(vec!["a/w.rs:40 lines 40..40".into(), format!("{CALLED_BY}:a/w.rs:5 fn inner() {{ let z = 1; }}")]);
    let k = reduce(7, q, &[callee, wrapper], 4000, &slots);
    let srcs: Vec<&str> = k.packets.iter().map(|p| p.provenance().source.as_str()).collect();
    if srcs != ["a/w.rs:5", "a/w.rs:40"] {
        return Err(format!("nmbrain.synthesis: a wrapper's callee kept {srcs:?} ({})", k.reason));
    }
    // A code file's opening window gives its place to a later, separate packet of the same file;
    // a window that touches it, another file's packet and a document passage keep their places.
    let span_t = |src: &str, a: usize, b: usize, text: &str| {
        Packet::new("chunk", &format!("chunk:{src}"), text, Provenance::new(src, "x", 1, 0)).evidence(vec![format!("{src} lines {a}..{b}")])
    };
    let span = |src: &str, a: usize, b: usize| span_t(src, a, b, "x");
    let mut order = vec![
        span_t("a/x.rs:4", 4, 4, "//! the file's note"),
        span("a/x.rs:1", 1, 30),
        span("b/y.rs:5", 5, 9),
        span("a/x.rs:31", 31, 60),
        span("a/x.rs:80", 80, 90),
        span("d.md:1", 1, 9),
        span("d.md:20", 20, 29),
    ];
    pointer_last(&mut order);
    let srcs: Vec<&str> = order.iter().map(|p| p.provenance().source.as_str()).collect();
    if srcs != ["a/x.rs:80", "a/x.rs:4", "a/x.rs:1", "b/y.rs:5", "a/x.rs:31", "d.md:1", "d.md:20"] {
        return Err(format!("nmbrain.synthesis: opening window order {srcs:?}"));
    }
    // A code line inside the opening window (a constant) or its doc comment is specific and keeps
    // its place.
    let mut order = vec![span_t("a/x.rs:19", 19, 19, "/// The cap.\npub const X: u8 = 1;"), span("a/x.rs:1", 1, 30), span("a/x.rs:80", 80, 90)];
    pointer_last(&mut order);
    let srcs: Vec<&str> = order.iter().map(|p| p.provenance().source.as_str()).collect();
    if srcs != ["a/x.rs:19", "a/x.rs:80", "a/x.rs:1"] {
        return Err(format!("nmbrain.synthesis: a code line in the opening window moved {srcs:?}"));
    }
    // Budget: a packet too large for what is left is skipped, a smaller one after it fits.
    let big = mk("value:LIMIT", &format!("LIMIT = 4 {}", "x".repeat(400)), "a.rs:1", Status::Observed);
    let small = mk("symbol:LIMIT@b.rs", "LIMIT", "b.rs:1", Status::Observed);
    let b = reduce(5, "What is LIMIT set to?", &[big, small], 40, &slots);
    if b.withheld || b.packets.len() != 1 {
        return Err(format!("nmbrain.synthesis: budget skip kept {} ({})", b.packets.len(), b.reason));
    }
    // Ranked readers: a tentacle chunk and a lane chunk are admitted by place without sharing
    // the question's words; the same chunk from both is one packet with both citations, and
    // the tentacle's first place orders ahead of the lane's first.
    let ranked = |prefix: &str, place: usize, text: &str, src: &str, cite: &str| {
        Packet::new("chunk", &format!("{prefix}{place:02}:{src}"), text, Provenance::new(src, "x", 1, 0))
            .status(Status::Inferred)
            .evidence(vec![cite.to_owned()])
    };
    let set = [
        ranked(DENSE_KEY, 0, "fn wobble() {}", "a/x.rs:1", "a/x.rs:1 lines 1..9"),
        ranked(TENTACLE_KEY, 0, "fn other() {}", "a/y.rs:1", "a/y.rs:1 lines 1..9"),
        ranked(TENTACLE_KEY, 1, "fn wobble() {}", "a/x.rs:1", "a/x.rs:1 line 1"),
    ];
    let t = reduce(6, "which code steadies the frame", &set, 4000, &slots);
    if t.withheld
        || t.packets.len() != 2
        || t.packets.first().map(Packet::key) != Some("tentacle:00:a/y.rs:1")
        || t.packets.get(1).map(|x| x.get_evidence().len()) != Some(2)
    {
        return Err(format!(
            "nmbrain.synthesis: ranked readers kept {:?} ({})",
            t.packets.iter().map(Packet::key).collect::<Vec<_>>(),
            t.reason
        ));
    }
    evidence_check(&slots)
}

/// (g) The new path (refinement step 1): tentacle chunks are ordered by their evidence records,
/// not by the place in their keys (the place-1 chunk with the stronger record leads); a packet
/// below the floor and a packet past the token room are held low in the queue, not lost; the
/// old path on the same packets is unchanged.
fn evidence_check(slots: &Slots) -> Result<(), String> {
    use crate::evidence::{Priority, SpanFacts};
    use crate::g0::SpanRef;
    let ranked = |place: usize, text: &str, src: &str| {
        Packet::new("chunk", &format!("{TENTACLE_KEY}{place:02}:{src}"), text, Provenance::new(src, "x", 1, 0))
            .status(Status::Inferred)
            .evidence(vec![format!("{src} lines 1..9")])
    };
    let rec = |file: &str, key: &str, combined: f32| SpanEvidence {
        span: SpanRef { file: file.into(), start: 1, end: 9, key: key.into() },
        combined,
        priority: Priority::Forwarded,
        round: 1,
        facts: SpanFacts { tokens: 20, ..SpanFacts::default() },
        ..SpanEvidence::default()
    };
    use nmtype::Provenance;
    let mut set = EvidenceSet { forwarded: vec![rec("a/y.rs", "ky", 0.3), rec("a/x.rs", "kx", 0.9)], ..EvidenceSet::default() };
    let unrelated = Packet::new("symbol", "symbol:zz@q.rs", "fn zz() {}", Provenance::new("q.rs:4", "x", 1, 0))
        .status(Status::Observed)
        .evidence(vec!["q.rs:4".into()]);
    let packets = [ranked(0, "fn other() {}", "a/y.rs:1"), ranked(1, "fn wobble() {}", "a/x.rs:1"), unrelated];
    let before = set.all().count();
    let p = reduce_evidence(7, "which code steadies the frame", &packets, 4000, slots, &mut set);
    let old = reduce(7, "which code steadies the frame", &packets, 4000, slots);
    let low_added = set.queue.iter().any(|e| e.priority == Priority::Low && e.span.file == "q.rs");
    if p.packets.first().map(Packet::key) != Some("tentacle:01:a/x.rs:1")
        || old.packets.first().map(Packet::key) != Some("tentacle:00:a/y.rs:1")
        || !low_added
        || set.all().count() != before + 1
    {
        return Err(format!(
            "nmbrain.synthesis (g): evidence order {:?}, old order {:?}, low held {low_added}",
            p.packets.iter().map(Packet::key).collect::<Vec<_>>(),
            old.packets.iter().map(Packet::key).collect::<Vec<_>>()
        ));
    }
    // Token room: the second chunk does not fit and goes back to the queue, marked low.
    let mut set2 = EvidenceSet { forwarded: vec![rec("a/y.rs", "ky", 0.3), rec("a/x.rs", "kx", 0.9)], ..EvidenceSet::default() };
    let p2 = reduce_evidence(8, "which code steadies the frame", &packets[..2], tokens(&packets[1]), slots, &mut set2);
    if p2.packets.len() != 1
        || set2.forwarded.len() != 1
        || set2.queue.first().is_none_or(|e| e.span.key != "ky" || e.priority != Priority::Low)
    {
        return Err(format!(
            "nmbrain.synthesis (g): budget kept {} packets, forwarded {}, queue {:?}",
            p2.packets.len(),
            set2.forwarded.len(),
            set2.queue.iter().map(|e| (&e.span.key, e.priority)).collect::<Vec<_>>()
        ));
    }
    Ok(())
}
