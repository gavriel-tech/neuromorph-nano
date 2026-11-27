//! The whole path through the drones, first form (2026-11-10; Gavriel Technologies, architecture
//! and directions by Luis Jake Gabriel III: assemble the whole first, then fix the worst part).
//! Served by `serve --whole` (`nmapi::whole`, 2026-11-12); `run dx whole` measures it.
//!
//! One question, in the order of the specification's processing cycle
//! (`docs/2026-11-10-neumor-spec-octopus-to-neuromorph.md` section 8):
//! 1. Drones: every piece of the tree is sensed against the request, in parallel on the
//!    processor cores (`Keep::signals`). The flag strength is the BM25F sense (measured best of
//!    the drone senses, `runs/2026-11-10-D1-drone-intake-30q-1794334108354.txt`); every other
//!    sense travels beside it as a drone row. With the lenses, every sense is the drones'
//!    calibrated evidence (`nmpod::evidence`, `Keep::evidence`): its surprise under the
//!    content-free pose, per aspect of the request.
//! 2. Tentacles: eight windows of `nmsubs::lens::BUDGET` tokens, each composed by its own lens
//!    (`lenses`, `compose_lenses`): six lenses read one drone signal each, one follows the
//!    specific relations of the strongest finds, one takes what memory recalled and G2 seeded;
//!    no piece in two windows, every window full.
//! 3. `g0_tx`: one per tentacle, built from its own drones' readings of its own pieces
//!    (`nmbrain::g0::G0`, which is a `TentacleSide` without a model): held spans by reference,
//!    the quantized drone table, every drone signal as its own score set with the lens's first,
//!    and confidences on the scale shared by all tentacles (with the lenses, each piece's whole
//!    calibrated evidence as a share of the tree's strongest); with the lenses it also holds the
//!    drones' readout of its pieces (aspects held, epistemic tags, related pieces it does not
//!    hold) for its report. A re-read turns to a signal not yet used, so G1's later rounds receive
//!    new views.
//! 4. G1: the coordinating loop over the eight (`nmbrain::g1::G1::run`, one thread per
//!    tentacle), the only path between them; hand-over and queue, nothing dropped.
//! 5. G2, first form on this path: the capacity gate. What goes up to the Core Brain is the
//!    hand-over order, then the queue, up to the number of pieces the Core Brain can read.
//! 6. Core Brain: the reranker's order (2026-11-12; the three-judge agreement of
//!    `decide::agreement_from` did not hold on the unseen half), then the second look inside the
//!    winning file (`decide::second_look`, CB-1, 2026-11-12).
//!
//! Every stage's outcome is kept in `StackRun`, so a measurement can say at which stage the right
//! piece was lost.
use crate::drone_windows::{Keep, compose_with};
use crate::joint_read::{Evidence, Piece};
use crate::tentacle_core::{SHORTLIST_FALLBACK, SubGraph, codebase};
use nmbrain::g0::{G0, HeldChunk, SpanRef, TentacleSide};
use nmbrain::g1::{Budget, G1, StructureRef};
use nmbrain::gmap::WindowRecord;
use nmsubs::Substrate;
use nmtype::Fault;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

/// Pieces the memory lane's I11 release asks for (`forage`); nothing on this path reads it. What
/// goes up here follows the caller's token budget, else `PACK_TOKENS`.
pub const UP_FALLBACK: usize = 12;

/// The tokens a pack holds when the caller names no budget: the default of `nm_context_pack`'s
/// `tokens` argument, which is what Claude Code receives per call when it names none (two
/// percent of its 200,000-token context). The one place this fallback is written; with a caller's
/// budget, that budget governs.
pub const PACK_TOKENS: usize = 4_000;
/// Stage 5 WP-C1: the value fill's first pieces kept when G2's relation-aware selection joins the
/// pack set (the leader and its two nearest rivals).
const G2_PACK_PROTECT: usize = 3;

/// What G2 did on one request (`stack_core`): its gates before and after its orders, the
/// covering set, the aspects no connected set holds, the motor path's ledger, goals and replies,
/// the edges used by class, the pieces its orders had read, the joint read named, and its time.
#[derive(Debug, Default)]
pub struct G2Run {
    pub first_gate: String,
    pub gate: String,
    pub set: Vec<usize>,
    /// W5: G2's selection's own set (before the fused value filled the room); the pack's parts
    /// held and the gate read it, so the pack's confidence keeps its meaning. Empty without W5.
    pub selected: Vec<usize>,
    pub absent: Vec<String>,
    pub motor: nmbrain::motor::Down,
    pub up: nmbrain::motor::Up,
    pub edges: Vec<(&'static str, &'static str, nmtype::class::DataClass)>,
    /// Edges refused by the topology check (must stay 0).
    pub class_refusals: usize,
    pub lines: Vec<String>,
    pub read: usize,
    pub joint: Vec<usize>,
    pub ms: f64,
    /// F of the chosen set and the number of aspects and relations it is over.
    pub value: (f32, usize),
    /// Why the selection stopped, and the gains it evaluated.
    pub stop: String,
    pub evaluated: usize,
    /// Milliseconds of the tentacles' read on G2's orders (G1's round over them).
    pub read_ms: f64,
    /// The Core Brain's judge on the pieces at the inclusion edge of G2's selection: pieces
    /// judged, its milliseconds, and the pieces whose judgment moved them into or out of the set.
    pub judged: usize,
    pub judge_ms: f64,
    pub judge_moved: usize,
    /// Tokens the judge read (each piece once posed, and once per content-free form when its
    /// content-free label was not yet held).
    pub judge_tokens: usize,
    /// For the bench's account of what the judge decided: each judged piece with the shift it
    /// needed to cross the edge and the judge's calibrated evidence added to it; and the pieces
    /// the judgment moved into and out of the set.
    pub judge_pairs: Vec<(usize, f32, f32)>,
    pub judge_in: Vec<usize>,
    pub judge_out: Vec<usize>,
    /// The judge's duels at the edge: entrant, holder, the log-odds shift the entrant needed to
    /// cross the greedy's edge (the duels' order), and the judge's difference of their values
    /// (entrant less holder; above 0 the entrant took the holder's place).
    pub judge_duels: Vec<(usize, usize, f32, f32)>,
    /// The set's rivals: candidates left out whose gain in F, given the set, still stands above the
    /// noise band (what the greedy would take next with more room), by gain per token.
    pub rivals: Vec<usize>,
    /// For the replay bench: the noise band of the last selection, the room, the caller-fact
    /// columns (relation k, its fact pieces) and the relation terms between candidates
    /// (p, q, relation k: p settles relation k with q in the set), as `nmbrain::g2::select` saw them.
    pub band: f32,
    pub room: usize,
    pub facts: Vec<(usize, Vec<usize>)>,
    pub rel_pairs: Vec<(usize, usize, usize)>,
    /// With items on (`drone_windows::items_mode`): the items G2 chose, in G1's order.
    pub items: Vec<usize>,
    /// G2's margin at its edge: the log of the last set piece's gain per token over the best left
    /// out's (C9; above 0 the set's last choice stood clear). `MARGIN_CAP` when nothing is left out.
    pub margin: f32,
    /// The follow-up read (`NANO_FOLLOW=on`): its query (the anchor's names, held words masked)
    /// and the pieces the anchor's tentacle read on it; empty when it did not fire.
    pub follow_query: Vec<String>,
    pub follow: Vec<usize>,
    /// The follow-up's anchor (the set's first piece when it fired).
    pub follow_from: Option<usize>,
    /// Close calls routed by strength (on by default; `NANO_ROUTE=off`): duels the embedder's meaning read
    /// decided, duels the judge read in both orders, the judge's split verdicts (ties), and the
    /// ties the reader's joint read decided.
    pub routed: (usize, usize, usize, usize),
    /// G2's room rule (`NANO_SWAP_IN=on`): pieces read on its orders or follow-up that took a
    /// set piece's place (entrant, holder).
    pub swapped: Vec<(usize, usize)>,
    /// The pack's confidence signals (`PackSignals`), read by the flow's three states.
    pub signals: PackSignals,
    /// Meaning in G1's map (`NANO_MEANING_MAP=on`): orders aimed at a meaning cluster the set
    /// lacks, and claims refused because a near-duplicate by meaning was held or read.
    pub cluster_orders: usize,
    pub refused_twin: usize,
    /// The close-call arm (step J, `NANO_CLOSE=on`): the leader's margin in fused evidence over
    /// its strongest rival among the first `CLOSE_K` pieces sent up and that set's noise band;
    /// pieces the judge read (0: not fired), and its milliseconds.
    pub close_lead: (f32, f32),
    pub close_fired: usize,
    pub close_ms: f64,
    /// Phase H2: the judge's sufficiency read of the set's first piece (`PackSignals::sufficient`).
    pub sufficient: Option<f32>,
    /// The gate found no finite rival to the leader and the judge's sufficiency read did not say
    /// the set answers: a tie the pack's status serves as low (lead, 2026-11-19).
    pub unrivalled_tie: bool,
    /// Milliseconds of G2's reads by part (speed, lane T1 2026-11-19): the judge's sufficiency
    /// read, the own-pick reads (prose picks and the admission's judge read), the Core Brain's
    /// intake chain.
    pub part_ms: (f64, f64, f64),
}

/// The signals the flow's pack status is fitted on (research item 4, sufficient context and
/// CRAG): the share of the request's parts the pack holds, G2's margin at its edge (C9), the
/// meaning lane's agreement with the first piece sent up (1 / (1 + its lane place), 0 outside the
/// lane), and whether that piece holds a definition site (or a document section) of a part.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PackSignals {
    pub held: f32,
    pub margin: f32,
    pub lane: f32,
    pub defines: f32,
    /// Phase H2 (W7, 2026-11-19): the judge's sufficiency read of the set's first piece (its label
    /// log-odds under the question, content-free calibrated by the close-call read; above 0 the set
    /// answers); `None` when not read (no card, or the judge not ready).
    pub sufficient: Option<f32>,
}

impl PackSignals {
    /// The signals as the fit's columns, in order.
    #[must_use]
    pub fn row(&self) -> [f32; 4] {
        [self.held, self.margin, self.lane, self.defines]
    }
}

/// The largest margin `G2Run::margin` reports (nothing left out, or nothing close).
pub const MARGIN_CAP: f32 = 3.0;

/// The margin at or below which the follow-up read fires (`NANO_FOLLOW=on`): the 40th percentile of
/// G2's margins on the large code tuning set (71 calls, 2026-11-18: lower quartile -1.40, median
/// -0.46), inside FLARE's firing band of 20 to 50 percent; `NANO_FOLLOW_BAND` overrides it.
pub const FOLLOW_BAND: f32 = -0.8;

/// Pieces the follow-up read takes at most.
pub const FOLLOW_PIECES: usize = 3;

/// For the replay bench (guide section 13): chunks placed by hand at one stage of the next
/// `stack_core` call. Off unless the bench sets it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForceAt {
    Windows,
    Handover,
    Set,
}
static FORCE: std::sync::RwLock<Option<(ForceAt, Vec<usize>)>> = std::sync::RwLock::new(None);
/// Set the chunks forced in at one stage (`None`: nothing forced).
pub fn set_force(f: Option<(ForceAt, Vec<usize>)>) {
    if let Ok(mut g) = FORCE.write() {
        *g = f;
    }
}
fn forced(at: ForceAt) -> Vec<usize> {
    FORCE.read().ok().and_then(|g| g.as_ref().filter(|(a, _)| *a == at).map(|(_, v)| v.clone())).unwrap_or_default()
}

static JUDGE: std::sync::RwLock<bool> = std::sync::RwLock::new(false);
/// Set whether the Core Brain's judge reads the pieces at the inclusion edge of G2's selection
/// (round 1 of the Core Brain, guide 12.3), its judgment replacing their chance.
pub fn set_judge(on: bool) {
    if let Ok(mut g) = JUDGE.write() {
        *g = on;
    }
}

/// Stated fallback time budget of one call in seconds when the caller gives none: the median
/// call the guide sets for the default path (WP6). The judge's round fills what is left of it.
pub const CALL_BUDGET_FALLBACK_S: f64 = 2.0;
static CALL_BUDGET: std::sync::RwLock<Option<f64>> = std::sync::RwLock::new(None);
/// Set the caller's time budget per call in seconds (`None`: the stated fallback).
pub fn set_call_budget(s: Option<f64>) {
    if let Ok(mut g) = CALL_BUDGET.write() {
        *g = s;
    }
}
/// The call's time budget in seconds (`set_call_budget`, else `CALL_BUDGET_FALLBACK_S`).
#[must_use]
pub fn call_budget() -> f64 {
    call_budget_s()
}
fn call_budget_s() -> f64 {

    CALL_BUDGET.read().ok().and_then(|g| *g).unwrap_or(CALL_BUDGET_FALLBACK_S)
}
/// This machine's measured judge time: seconds and tokens judged in this process.
static JUDGE_RATE: std::sync::Mutex<(f64, f64)> = std::sync::Mutex::new((0.0, 0.0));
/// Whether the judge has read once in this process (its first read is not measured).
static JUDGE_WARM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
fn logit(p: f32) -> f32 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}
fn logistic(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
/// The log-odds shift `d`, added to every part a piece holds (`r` above 0), that brings its gain
/// over the misses `miss` to `edge` per token: `None` when no shift could (the piece holds no
/// part, or its gain even at certainty stays below the edge). Step 1i's order of the contested.
fn shift_to(miss: &[f32], r: &[f32], tokens: usize, edge: f32) -> Option<f32> {
    #[expect(clippy::cast_precision_loss, reason = "token counts")]
    let target = edge * tokens.max(1) as f32;
    let held: Vec<(f32, f32)> = miss.iter().zip(r).filter(|(m, x)| **x > 0.0 && **m > 0.0).map(|(m, x)| (*m, logit(*x))).collect();
    if held.is_empty() || target <= 0.0 || held.iter().map(|h| h.0).sum::<f32>() <= target {
        return None;
    }
    let gain = |d: f32| -> f32 { held.iter().map(|(m, l)| m * logistic(l + d)).sum() };
    let (mut lo, mut hi) = (-40.0_f32, 40.0_f32);
    for _ in 0..48 {
        let mid = (lo + hi) / 2.0;
        if gain(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some((lo + hi) / 2.0)
}
/// The entrants the judge may duel at G2's edge, in order of the smallest log-odds shift that
/// would carry each into the set (|`shift_to`| over the set's misses, to the greedy's edge: the
/// geometric mean of the last piece taken's gain per token and the best left out's).
fn shift_entrants(
    set: &[usize],
    out: &[usize],
    miss: &[f32],
    rs: &dyn Fn(usize) -> Vec<f32>,
    tk: &dyn Fn(usize) -> usize,
    last_in: f32,
) -> Vec<(f32, usize)> {
    #[expect(clippy::cast_precision_loss, reason = "token counts")]
    let per = |g: f32, c: usize| g / tk(c).max(1) as f32;
    let best_out = out.iter().map(|c| per(miss.iter().zip(rs(*c)).map(|(m, x)| m * x).sum(), *c)).fold(0.0_f32, f32::max);
    let edge = if last_in > 0.0 && best_out > 0.0 { (last_in * best_out).sqrt() } else { last_in.max(best_out) };
    let mut e: Vec<(f32, usize)> = out.iter().filter(|c| !set.contains(c)).filter_map(|c| shift_to(miss, &rs(*c), tk(*c), edge).map(|d| (d.abs(), *c))).collect();
    e.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    e
}

static G2COVER: std::sync::RwLock<bool> = std::sync::RwLock::new(false);
/// Set whether G2 comprehends by coverage and sends orders (on the common scale with G1's race),
/// or passes G1's order up to the capacity (the default until coverage passes its measure).
pub fn set_g2(on: bool) {
    if let Ok(mut g) = G2COVER.write() {
        *g = on;
    }
}

/// The drone signals, by name: each `g0_tx` holds every one as its own score set.
pub const SIGNALS: [(&str, nmpod::intake::Sig); 6] = [
    ("bm25f", |s| s.bm25f),
    ("bm25", |s| s.bm25),
    ("exact", |s| s.exact),
    ("structural", |s| s.structural),
    ("affinity", |s| s.affinity),
    ("ident", |s| s.ident),
];

/// One tentacle's lens on a question: the function it has now (its label), the statistic its
/// `g0_tx` reads by first, and its score of every chunk of the tree (0: not wanted).
#[derive(Clone, Debug)]
pub struct Lens {
    pub name: &'static str,
    pub statistic: &'static str,
    pub score: Vec<f32>,
}

/// Every chunk's drone evidence on the scale shared by all tentacles: each signal as a share of
/// the tree's strongest for this question, averaged over the signals.
#[must_use]
pub fn fused_shares(signals: &[nmpod::intake::Signals]) -> Vec<f32> {
    let tops: Vec<f32> = SIGNALS.iter().map(|(_, g)| signals.iter().map(*g).fold(0.0_f32, f32::max)).collect();
    #[expect(clippy::cast_precision_loss, reason = "six signals")]
    let k = SIGNALS.len() as f32;
    signals
        .iter()
        .map(|s| SIGNALS.iter().zip(&tops).map(|((_, g), top)| if *top > 0.0 { (g(s) / top).max(0.0) } else { 0.0 }).sum::<f32>() / k)
        .collect()
}

/// The calls whose stage times are kept for the live view.
const STAGE_RING: usize = 32;
/// The last calls' stage times, newest last: (time stamp in ms, stage and ms pairs).
static STAGE_TIMES_RING: std::sync::Mutex<std::collections::VecDeque<StageCall>> = std::sync::Mutex::new(std::collections::VecDeque::new());
/// One call's stage times: its time stamp (ms) and the stage and ms pairs.
type StageCall = (u64, Vec<(&'static str, f64)>);

/// Keeps one call's stage times in the ring (`STAGE_RING` calls).
fn record_stage_times(stages: &[(&'static str, f64)]) {
    // Roadmap Phase 4.5 (2026-11-24): the call's stage times on the live timeline too.
    crate::live::stage_times(stages);
    if let Ok(mut r) = STAGE_TIMES_RING.lock() {
        r.push_back((nmtype::now_ms(), stages.to_vec()));
        while r.len() > STAGE_RING {
            r.pop_front();
        }
    }
}

/// Roadmap Phase 3 (2026-11-24): the last calls' stage times for the live file, oldest first, each
/// an object of stage names to ms with `at_ms`.
#[must_use]
pub fn recent_stage_times() -> Vec<nmjson::Value> {
    let Ok(r) = STAGE_TIMES_RING.lock() else { return Vec::new() };
    r.iter()
        .map(|(at, st)| {
            let mut o: std::collections::BTreeMap<String, nmjson::Value> = st.iter().map(|(k, v)| ((*k).to_owned(), nmjson::Value::Number(format!("{v:.1}")))).collect();
            o.insert("at_ms".to_owned(), nmjson::Value::Number(at.to_string()));
            nmjson::Value::Object(o)
        })
        .collect()
}

/// Roadmap Phase 0 item 1: the tree prior's evidence in the team's belief. Each tree seed's belief
/// rises as a relation's does from a find (noisy-or), by the strongest find's belief times the hop
/// decay times the seed file's tree weight.
fn raise_by_tree(belief: &mut [f32], finds: &[usize], seeds: &[(usize, f32)]) {
    let bf = finds.iter().filter_map(|c| belief.get(*c).copied()).filter(|x| x.is_finite()).fold(0.0_f32, f32::max);
    for (c, w) in seeds {
        if let Some(x) = belief.get_mut(*c) {
            let p = (bf * nmsubs::lens::HOP_DECAY * w).clamp(0.0, 1.0);
            *x = 1.0 - (1.0 - x.clamp(0.0, 1.0)) * (1.0 - p);
        }
    }
}

/// Roadmap Phase 0 item 1: the tree prior's seeds into the frontier. They join the seeds of the
/// last owner-less follow aim in `aims` (the round's frontier walk), those not already there; with
/// no such aim they become one.
fn join_frontier(aims: &mut Vec<crate::coordinator::Aim>, ts: Vec<(usize, f32)>) {
    if ts.is_empty() {
        return;
    }
    if let Some(crate::coordinator::Aim::Follow { seeds, .. }) = aims.iter_mut().rev().find(|a| matches!(a, crate::coordinator::Aim::Follow { owner: None, .. })) {
        for t in ts {
            if !seeds.iter().any(|s| s.0 == t.0) {
                seeds.push(t);
            }
        }
    } else {
        aims.push(crate::coordinator::Aim::Follow { seeds: ts, owner: None });
    }
}

/// Each value as a share of the largest (0 where none is above 0).
#[must_use]
pub fn shares(v: &[f32]) -> Vec<f32> {
    let top = v.iter().copied().fold(0.0_f32, f32::max);
    v.iter().map(|x| if top > 0.0 { (x / top).max(0.0) } else { 0.0 }).collect()
}

/// The drones' calibrated evidence as one row of signals per piece, by the signals' names: each
/// signal's surprise under the content-free pose, added over the request's aspects.
#[must_use]
pub fn calibrated_signals(ev: &nmpod::evidence::Evidence) -> Vec<nmpod::intake::Signals> {
    let at = |name: &str, i: usize| nmpod::evidence::SIGNALS.iter().position(|n| *n == name).map_or(0.0, |s| ev.signal(s, i));
    (0..ev.len())
        .map(|i| nmpod::intake::Signals {
            bm25f: at("bm25f", i),
            bm25: at("bm25", i),
            exact: at("exact", i),
            structural: at("structural", i),
            ident: at("ident", i),
            affinity: at("affinity", i),
        })
        .collect()
}

/// The scale every `g0_tx` reports on and G1 values on, for one request: the pool of the
/// drones' independent families over every piece the drones found evidence in, each piece's
/// value on it, and the one report gate (the shortlist rule over those values: the effective
/// count of the request's evidence, its last value the gate).
#[derive(Clone, Debug)]
pub struct CommonScale {
    pub pool: nmbrain::g1::Pool,
    pub values: Vec<f32>,
    pub floor: f32,
    /// The noise band on this scale: the median absolute deviation of the positive values (the
    /// mean absolute deviation where that is 0).
    pub band: f32,
}

/// A piece's family values from its calibrated signals (each family's best signal).
#[must_use]
pub fn family_row(s: &nmpod::intake::Signals) -> [f32; 3] {
    let m = |v: &[f32]| v.iter().copied().filter(|x| x.is_finite()).fold(0.0_f32, f32::max);
    [m(&[s.bm25f, s.bm25, s.exact]), m(&[s.structural, s.ident]), m(&[s.affinity])]
}

/// The common scale of one request (`CommonScale`).
#[must_use]
pub fn common_scale(signals: &[nmpod::intake::Signals]) -> CommonScale {
    let rows: Vec<[f32; 3]> = signals.iter().map(family_row).collect();
    let fired: Vec<[f32; 3]> = rows.iter().copied().filter(|r| r.iter().any(|x| *x > 0.0)).collect();
    let pool = nmbrain::g1::Pool::over(&fired);
    let values: Vec<f32> = rows.iter().map(|r| pool.value_of(r)).collect();
    let mut pos: Vec<f32> = values.iter().copied().filter(|v| *v > 0.0).collect();
    pos.sort_by(|a, b| b.total_cmp(a));
    let (k, _) = nmbrain::g1::shortlist_by(nmbrain::g1::ShortlistRule::Entropy, &pos, SHORTLIST_FALLBACK);
    let floor = pos.get(k.saturating_sub(1)).copied().unwrap_or(f32::INFINITY);
    let band = crate::coordinator::noise_band(&pos);
    CommonScale { pool, values, floor, band }
}

/// The eight lenses for one question, in tie order. Six read one drone signal each (keywords
/// with definition lines and paths boosted, plain keywords, identifiers as written, definition
/// lines, meaning by token embedding, split identifier words); `relations` wants the chunks the
/// strongest finds relate to over the specific reference graph (each relation weighted by the
/// find's evidence and the edge's derived kind weight); `memory` wants what memory recalled and
/// G2 seeded, and with none of those reads the fused evidence (`fused_shares`).
#[must_use]
pub fn lenses(signals: &[nmpod::intake::Signals], fused: &[f32], graph: &nmsubs::refgraph::Specific, pinned: &[usize]) -> Vec<Lens> {
    let of = |g: nmpod::intake::Sig| -> Vec<f32> { signals.iter().map(g).collect() };
    let signal = |name: &str| SIGNALS.iter().find(|(n, _)| *n == name).map_or_else(Vec::new, |(_, g)| of(*g));
    let threshold = crate::drone_windows::strong_threshold(fused);
    let kw = graph.kind_weights();
    let mut related = vec![0.0_f32; fused.len()];
    for (f, v) in fused.iter().enumerate().filter(|(_, v)| **v > 0.0 && **v >= threshold) {
        let edges = graph.edges_of(f);
        let top = edges.iter().map(|e| e.weight * kw.get(e.kind.index()).copied().unwrap_or(1.0)).fold(0.0_f32, f32::max);
        if top > 0.0 {
            for e in edges {
                if let Some(r) = related.get_mut(e.to as usize) {
                    *r += v * e.weight * kw.get(e.kind.index()).copied().unwrap_or(1.0) / top;
                }
            }
        }
    }
    let memory = if pinned.is_empty() {
        Lens { name: "fused", statistic: "fused", score: fused.to_vec() }
    } else {
        let mut v = vec![0.0_f32; fused.len()];
        for c in pinned {
            if let Some(x) = v.get_mut(*c) {
                *x = 1.0;
            }
        }
        Lens { name: "memory", statistic: "memory", score: v }
    };
    vec![
        Lens { name: "fields", statistic: "bm25f", score: signal("bm25f") },
        Lens { name: "words", statistic: "bm25", score: signal("bm25") },
        Lens { name: "names", statistic: "exact", score: signal("exact") },
        Lens { name: "structure", statistic: "structural", score: signal("structural") },
        Lens { name: "meaning", statistic: "affinity", score: signal("affinity") },
        Lens { name: "parts", statistic: "ident", score: signal("ident") },
        Lens { name: "relations", statistic: "relations", score: related },
        memory,
    ]
}

/// One window per lens of at most `budget` tokens, no chunk in two: each lens lists the chunks it
/// wants, best first (ties by the fused evidence, then chunk), and a chunk goes to the lens that
/// ranks it highest (`nmsubs::lens::assign`). A window with room left then takes the unplaced
/// chunks in fused order, so every window is full and nothing is cut.
#[must_use]
pub fn compose_lenses(lenses: &[Lens], fused: &[f32], tokens: &dyn Fn(usize) -> usize, budget: usize) -> Vec<Vec<usize>> {
    let f = |c: usize| fused.get(c).copied().unwrap_or(0.0);
    let lists: Vec<Vec<usize>> = lenses
        .iter()
        .map(|l| {
            let s = |c: usize| l.score.get(c).copied().unwrap_or(0.0);
            let mut v: Vec<usize> = (0..l.score.len()).filter(|c| s(*c) > 0.0).collect();
            v.sort_by(|a, b| s(*b).total_cmp(&s(*a)).then(f(*b).total_cmp(&f(*a))).then(a.cmp(b)));
            v
        })
        .collect();
    let mut windows = nmsubs::lens::assign(&lists, tokens, budget);
    let mut placed = vec![false; fused.len()];
    for c in windows.iter().flatten() {
        if let Some(p) = placed.get_mut(*c) {
            *p = true;
        }
    }
    let mut order: Vec<usize> = (0..fused.len()).collect();
    order.sort_by(|a, b| f(*b).total_cmp(&f(*a)).then(a.cmp(b)));
    for w in &mut windows {
        let mut room = budget.saturating_sub(w.iter().map(|c| tokens(*c)).sum());
        for &c in &order {
            if room == 0 {
                break;
            }
            let t = tokens(c);
            if t <= room && placed.get(c) == Some(&false) {
                w.push(c);
                room -= t;
                if let Some(p) = placed.get_mut(c) {
                    *p = true;
                }
            }
        }
    }
    windows
}

/// How the tentacles' windows are formed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Windows {
    /// From the drones' flags, the strong files dealt across the tentacles (`compose_with`).
    Flags,
    /// One window per lens (`compose_lenses`).
    Lenses,
    /// The tentacles as one team: the Map Coordinator's assignment over the belief, applied by
    /// claim in G1's allocation map, over rounds (`coordinator::Team`).
    Team,
    /// Comparison: first fit down one ranked list (the drones' calibrated evidence).
    FirstFit,
    /// Comparison: one reader given the same total tokens down the same list.
    Single,
}
impl Windows {
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "flags" => Some(Self::Flags),
            "lenses" => Some(Self::Lenses),
            "team" => Some(Self::Team),
            "firstfit" => Some(Self::FirstFit),
            "single" => Some(Self::Single),
            _ => None,
        }
    }
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Flags => "flags",
            Self::Lenses => "lenses",
            Self::Team => "team",
            Self::FirstFit => "firstfit",
            Self::Single => "single",
        }
    }
}
static WINDOWS: std::sync::RwLock<Option<Windows>> = std::sync::RwLock::new(None);
/// Set how windows are formed for this process (`None`: by the `deal` argument, lenses or flags).
pub fn set_windows(w: Option<Windows>) {
    if let Ok(mut g) = WINDOWS.write() {
        *g = w;
    }
}
static POOLED: std::sync::RwLock<Option<bool>> = std::sync::RwLock::new(None);
/// Set whether G1 ranks by its reasoning layer over the merged set (`nmbrain::g1::Combine::Pooled`)
/// for this process (`None`: by the arm, pooled for the team).
pub fn set_pooled(on: Option<bool>) {
    if let Ok(mut g) = POOLED.write() {
        *g = on;
    }
}
static COMMON: std::sync::RwLock<bool> = std::sync::RwLock::new(true);
/// Set whether a pooling G1 values pieces on the scale shared by all tentacles (`CommonScale`,
/// the default) or pools over the merged set of each round's reports (the comparison form).
pub fn set_common(on: bool) {
    if let Ok(mut g) = COMMON.write() {
        *g = on;
    }
}
static RACE: std::sync::RwLock<bool> = std::sync::RwLock::new(true);
/// Set whether G1's race (`nmbrain::g1::Race`) decides what goes up on the common scale (the
/// default), or the round's capacity does (the comparison form).
pub fn set_race(on: bool) {
    if let Ok(mut g) = RACE.write() {
        *g = on;
    }
}
/// How windows are formed on a call with `deal`.
#[must_use]
pub fn windows_rule(deal: bool) -> Windows {
    WINDOWS.read().ok().and_then(|w| *w).unwrap_or(if deal { Windows::Lenses } else { Windows::Flags })
}

/// Windows filled first fit down one ranked list: each piece of `order` goes to the first window
/// with room for it.
#[must_use]
pub fn first_fit(order: &[usize], tokens: &dyn Fn(usize) -> usize, budget: usize, windows: usize) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = vec![Vec::new(); windows];
    let mut room = vec![budget; windows];
    for &c in order {
        let t = tokens(c);
        if let Some(w) = room.iter().position(|r| *r >= t)
            && let (Some(o), Some(r)) = (out.get_mut(w), room.get_mut(w))
        {
            o.push(c);
            *r -= t;
        }
        if room.iter().all(|r| *r == 0) {
            break;
        }
    }
    out
}

/// W1's stage ladder (`NANO_LADDER=<file>`): how deep each stage's order is written (a measurement
/// depth: past it a piece's place reads as beyond).
pub const LADDER_DEPTH: usize = 300;

/// What the Globalized Context Stack did with one question, stage by stage.
#[derive(Debug, Default)]
pub struct StackRun {
    /// Drone flag strength per chunk (BM25F).
    pub flags: Vec<f32>,
    /// The eight tentacle windows (chunks in the order taken).
    pub windows: Vec<Vec<usize>>,
    /// Per tentacle, its gate depth (how many it reports first).
    pub depths: Vec<usize>,
    /// Per tentacle, its held chunks in its `g0_tx`'s order after G1's last round.
    pub orders: Vec<Vec<usize>>,
    /// G1's hand-over (chunks, G1's order), then its queue.
    pub handover: Vec<usize>,
    pub queue: Vec<usize>,
    /// What G2 sends up to the Core Brain, in order.
    pub up: Vec<usize>,
    /// The answer-first pack written from what went up (`write_pack`), and its seconds.
    pub pack: String,
    pub pack_s: f64,
    pub rounds: u32,
    pub stop: String,
    /// Orders refused at the isolation gate (must stay 0).
    pub gate: usize,
    /// Seconds: drones, tentacles and `g0_tx`, G1, and all of it.
    pub sense_s: f64,
    pub compose_s: f64,
    pub g1_s: f64,
    pub total_s: f64,
    /// G2's instructions on this question (`nmbrain::instruct`): one line per instruction, the
    /// joint read named by I6, and chunks scheduled for the next window formation (I7, I8).
    pub g2: Vec<String>,
    pub joint: Vec<usize>,
    pub scheduled: Vec<usize>,
    /// Per tentacle, its last report's aspects with no information (no held piece holds them),
    /// and whether it reported no information at all.
    pub absent: Vec<Vec<String>>,
    pub no_information: Vec<bool>,
    /// How the windows were formed, and each tentacle's function on its last window (its label).
    pub arm: String,
    pub labels: Vec<String>,
    /// The tentacle of each window in `windows` (a tentacle reads one window per round).
    pub owners: Vec<usize>,
    /// Every round's windows, round 1 first (`windows` is the team's windows after the last).
    pub taken: Vec<Vec<Vec<usize>>>,
    /// Per round, each tentacle's yield (the belief its report carried; `None`: no new read).
    pub yields: Vec<Vec<Option<f32>>>,
    /// The request's aspects some piece in the team's windows holds, and all of them.
    pub aspects_held: (usize, usize),
    /// Design invariants, counted over every round (each must stay 0): pieces in two windows of
    /// one round; held pieces neither forwarded nor queued, or both; tentacles whose first report
    /// differs from the report of the same window read with the other tentacles absent.
    pub overlap: usize,
    pub unconserved: usize,
    pub solo: usize,
    /// G1's race at the end (on the common scale): committed, ties and uncovered aspects for G2.
    pub verdict: Option<nmbrain::g1::Verdict>,
    /// What G2 did (empty without its comprehension).
    pub g2run: G2Run,
    /// The drones' calibrated evidence for the request (every arm but the flags), and G2's
    /// candidates at its last selection (G1's hand-over, then its queue): kept for the board's
    /// diagnosis of the selection.
    pub evidence: Option<nmpod::evidence::Evidence>,
    pub cands: Vec<usize>,
    /// The answer-first pack as the pieces it serves (audit W13; empty when withheld or with items).
    pub pack_pieces: Vec<Piece>,
}

/// Run the Globalized Context Stack for `question`: drones, tentacles, `g0_tx`, G1, G2's gate. Processor only.
/// `up` is the number of pieces the Core Brain can read (`None`: `UP_FALLBACK`). `common`: every
/// `g0_tx` reports on the common scale (each piece's calibrated evidence with the lenses, the BM25F
/// share without) in place of its standing inside its own
/// window. `deal`: each tentacle's window is composed by its lens (`compose_lenses`) and every
/// drone signal is a score set; without it (the comparison arm, `serve --lenses off`) the windows
/// fill from the BM25F flags with the strong files dealt across the tentacles
/// (`drone_windows::compose_with`), BM25F the only score set and its share of the tree's
/// strongest the common scale.
/// # Errors
/// The tokenizer refuses the request.
pub fn stack_run(sub: &Substrate, keep: &Keep, question: &str, up: Option<usize>, common: bool, deal: bool) -> Result<StackRun, Fault> {
    stack_run_with(sub, keep, question, up, common, deal, &[], &BTreeMap::new())
}

/// `stack_run` with memory's recalled chunks (recalled spans are guaranteed a window place: the
/// memory lens wants them first); G1 still ranks them by their drones' own readings.
/// # Errors
/// As `stack_run`.
/// `cocited`: co-citation links from memory (`session::cocitations`), joined to the reference
/// graph G1 reads as structure (`tentacle_core::WithCoCitation`); empty leaves G1 as before.
#[expect(clippy::too_many_arguments, reason = "the question, its settings and memory's chunks and links")]
pub fn stack_run_with(
    sub: &Substrate,
    keep: &Keep,
    question: &str,
    up: Option<usize>,
    common: bool,
    deal: bool,
    recalled: &[usize],
    cocited: &BTreeMap<String, Vec<(String, f32)>>,
) -> Result<StackRun, Fault> {
    stack_run_g2(sub, keep, question, up, common, deal, recalled, cocited, &G2In::default())
}

/// G2's instructions for one question (`nmbrain::instruct`): this question's orders, the standing
/// gains per tentacle (I9, from the desk) and what stands inhibited (I10).
#[derive(Clone, Debug, Default)]
pub struct G2In {
    pub orders: Vec<nmbrain::instruct::Instr>,
    pub gains: Vec<f32>,
    pub inhibited: Vec<nmbrain::instruct::Target>,
    /// Reflexes G2 installed (`nmbrain::reflex`) and the files the organism itself just edited (R4).
    pub reflexes: Vec<nmbrain::reflex::Installed>,
    pub own_files: Vec<String>,
    /// Replay (SN3): the drones' signals decoded from their records, in place of a fresh scan.
    pub signals: Option<Vec<nmpod::intake::Signals>>,
    /// Stage 5 WP-C1: the caller's pack budget in tokens, G2's room when no piece count is given
    /// (`None`: `PACK_TOKENS`).
    pub room_tokens: Option<usize>,
    /// Stage 5 WP-C2: the call's refinement allocation (G3's `g1.allocation` site, the configured
    /// mode its rule): `nbv` (the rule: the team's rounds as designed), `uniform` (coverage aims only
    /// from round 2), `off` (no round after the first); `None` as `nbv`.
    pub refine: Option<String>,
}


/// `stack_run_with` carrying G2's instructions (2026-11-12): I8 seeds are raised before the windows
/// form; I1, I2 and I3 go to G1 to relay in its first round, I9 gains into its combination; after G1,
/// G2's gate carries out I4, I5, I7, I10, I11 and names I6's joint read.
/// # Errors
/// As `stack_run`.
#[expect(clippy::too_many_arguments, reason = "the question, its settings, memory's chunks and links, and G2's instructions")]
pub fn stack_run_g2(
    sub: &Substrate,
    keep: &Keep,
    question: &str,
    up: Option<usize>,
    common: bool,
    deal: bool,
    recalled: &[usize],
    cocited: &BTreeMap<String, Vec<(String, f32)>>,
    g2: &G2In,
) -> Result<StackRun, Fault> {
    stack_core(sub, keep, question, up, common, windows_rule(deal), None, recalled, cocited, g2)
}

/// The one flow's stack (`whole::Mode::Flow`): the tentacles as a team on the common scale, with
/// memory's chunks and links and G2's instructions, as `stack_run_g2`.
/// # Errors
/// As `stack_run`.
#[expect(clippy::too_many_arguments, reason = "the question, its settings, memory's chunks and links, and G2's instructions")]
pub fn stack_run_flow(
    sub: &Substrate,
    keep: &Keep,
    question: &str,
    up: Option<usize>,
    recalled: &[usize],
    cocited: &BTreeMap<String, Vec<(String, f32)>>,
    g2: &G2In,
) -> Result<StackRun, Fault> {
    // TEAM-2's comparison arms (the bench's `--windows firstfit|single`) reach the served flow;
    // anything else serves the team.
    let arm = match WINDOWS.read().ok().and_then(|w| *w) {
        Some(a @ (Windows::FirstFit | Windows::Single)) => a,
        _ => Windows::Team,
    };
    stack_core(sub, keep, question, up, true, arm, None, recalled, cocited, g2)
}

/// The stack with the windows formed by `arm` (the board's arms). `total`: the tokens the
/// comparison arms (`FirstFit`, `Single`) take in all (`None`: one round of the team's windows).
/// # Errors
/// As `stack_run`.
pub fn stack_run_arm(sub: &Substrate, keep: &Keep, question: &str, up: Option<usize>, arm: Windows, total: Option<usize>) -> Result<StackRun, Fault> {
    stack_core(sub, keep, question, up, true, arm, total, &[], &BTreeMap::new(), &G2In::default())
}

/// One tentacle window to read: the tentacle, its pieces, and the lens its `g0_tx` reads by.
struct Read {
    tentacle: usize,
    window: Vec<usize>,
    reader: Option<Lens>,
    label: String,
}

/// What G1 made of one round of reads.
struct Round {
    outcome: nmbrain::g1::Outcome,
    /// Per read (in the round's order), its gate depth and its held pieces in its `g0_tx` order.
    depths: Vec<usize>,
    orders: Vec<Vec<usize>>,
    /// Pieces each read reported, first and last report together.
    finds: Vec<Vec<usize>>,
    unconserved: usize,
    solo: usize,
    /// The round's `g0_tx`, in the round's order: what each tentacle holds, kept to answer orders.
    segments: Vec<Box<dyn TentacleSide>>,
}

#[expect(clippy::too_many_lines, clippy::too_many_arguments, reason = "one question through every level, stage by stage")]
fn stack_core(
    sub: &Substrate,
    keep: &Keep,
    question: &str,
    up: Option<usize>,
    common: bool,
    arm: Windows,
    total: Option<usize>,
    recalled: &[usize],
    cocited: &BTreeMap<String, Vec<(String, f32)>>,
    g2: &G2In,
) -> Result<StackRun, Fault> {
    use nmbrain::instruct::Instr;
    let t0 = Instant::now();
    // Roadmap Phase 4.5 (2026-11-24): the question's stages in the live view, each shown active
    // from its start: G0 intake and routing (the drones' signals and calibrated evidence), G1
    // search, G2 selection with its reads and judge, G3 the pack.
    let mut live_stage = crate::live::stage("G0", "");
    // Stage 5 WP-C4 E1: a configured note store's notes and likely names join the meaning lane
    // once per material version (nothing without a store), on their own thread between calls (the
    // call holds the card gate the embedder needs).
    crate::writer_idle::seed_aux_background(sub);
    // Every arm but the flags reads the drones' calibrated evidence (`nmpod::evidence`): each
    // signal's surprise under the content-free pose, added over the request's aspects; the common
    // scale is each piece's evidence over every family and aspect, as a share of the strongest.
    let rich = arm != Windows::Flags;
    // The meaning lane encodes the question while the drones sense (the two share nothing): its
    // fused shortlist for G1's hand-over and, with the meaning signal, every piece's cosines.
    let lane = if rich { sub.dense.read().ok().and_then(|d| d.clone()) } else { None };
    // Meaning in G1's map (`NANO_MEANING_MAP=on`): the material's clusters by the embedder's
    // stored vectors, formed once per material version on the processor.
    let mm = if meaning_map_on() { lane.as_ref().and_then(|l| meaning_map(sub, l)) } else { None };
    let (signals, meaning) = std::thread::scope(|sc| {
        let m = sc.spawn(|| {
            let l = lane.as_ref()?;
            let _dense = crate::live::stage("G1.dense", "embedder");
            if stack_lane_on() {
                let _ = l.ranks(sub, question);
            }
            if stack_meaning_on() { l.cosines_now(sub, question) } else { l.cosines(sub, question) }
        });
        let s = match &g2.signals {
            Some(s) => Ok(s.clone()),
            None => keep.signals(question),
        };
        (s, m.join().unwrap_or(None))
    });
    let signals = signals?;
    let flags: Vec<f32> = signals.iter().map(|s| s.bm25f).collect();
    // The tentacles' read (D7) forms its shortlists from the wording (plain BM25) and meaning.
    let words: Vec<f32> = signals.iter().map(|s| s.bm25).collect();
    let t_sig = t0.elapsed().as_secs_f64();
    let calibrated = if rich { Some(keep.evidence_with(question, meaning.as_deref())?) } else { None };
    if std::env::var_os("NANO_STAGE_TIMES").is_some() {
        eprintln!("STAGE_TIMES signals and meaning {t_sig:.3} s, evidence {:.3} s", t0.elapsed().as_secs_f64() - t_sig);
    }
    let signals = match &calibrated {
        Some(ev) => calibrated_signals(ev),
        None => signals,
    };
    let sense_s = t0.elapsed().as_secs_f64();
    live_stage.count("pieces", signals.len());
    live_stage.next("G1", "");
    let graph = sub.refs();
    let tk = |c: usize| keep.tokens(c).max(1);
    // G2's I8 seeds are placed like recalled chunks: each is guaranteed a window place.
    let seeds: Vec<usize> =
        g2.orders.iter().filter_map(|i| if let Instr::Reshape { seeds, .. } = i { Some(seeds.clone()) } else { None }).flatten().collect();
    let pinned: Vec<usize> = recalled.iter().chain(&seeds).copied().collect();
    let top = flags.iter().copied().fold(0.0_f32, f32::max);
    // The common scale G1 compares on: each piece's calibrated evidence as a share of the tree's
    // strongest; in the flags arm the BM25F flag's share alone.
    let fused = match &calibrated {
        Some(ev) => shares(&ev.pieces()),
        None => flags.iter().map(|f| if top > 0.0 { f / top } else { 0.0 }).collect(),
    };
    // The scale shared by all tentacles when G1 pools: every piece's families pooled over the
    // whole material for this request (ceilings the tree's maxima, weights and agreement from the
    // request's own distributions), so a piece's value does not move with the window or set it
    // is reported in; and one report gate on that scale for every tentacle.
    let pooled = POOLED.read().ok().and_then(|p| *p).unwrap_or(arm == Windows::Team);
    let shared = (rich && pooled && COMMON.read().is_ok_and(|g| *g)).then(|| common_scale(&signals));
    let lenses = lenses(&signals, &fused, &graph.specific, &pinned);
    let n_t = lenses.len();
    let budget_w = crate::whole::tentacle_tokens_control(&nmtype::nano_root());
    let file_name = |c: usize| sub.chunks.get(c).and_then(|x| sub.files.get(x.file)).map(|f| f.path.replace('\\', "/")).unwrap_or_default();
    let by_key: BTreeMap<&str, usize> = sub.chunks.iter().enumerate().map(|(i, c)| (c.key.as_str(), i)).collect();
    let chunk_of = |key: &str| by_key.get(key).copied();
    let span_of = |c: usize| sub.chunks.get(c).map(|ch| SpanRef { file: file_name(c), start: ch.start, end: ch.end, key: ch.key.clone() });

    // W2 (on by default; `NANO_W2=off`): one fused value per piece for this request, used by the g0's order, G1's
    // hand-over and G2's set order; the meaning lane is one of its kinds (no splice).
    let w2 = rich && w2_on();
    // W5 on by default with W2 (2026-11-18 21:2x, C129); `NANO_W5=off` restores G2's own selection.
    let w5 = w2 && !std::env::var("NANO_W5").is_ok_and(|v| v == "off");
    // W4 (leader only) on by default with W2 since 2026-11-18 (C130's quiet card timing: p95 2.03 s on
    // large code); `NANO_W4=off` restores the value's order.
    let w4 = w2 && !std::env::var("NANO_W4").is_ok_and(|v| v == "off");
    // W3 (second form) on by default since 2026-11-18; `NANO_W3=off` restores.
    let w3 = w2 && !std::env::var("NANO_W3").is_ok_and(|v| v == "off");
    // W6's judge half on by default since 2026-11-18 (Luis: speed is weighed, not a veto);
    // `NANO_W6=off` restores the embedder on prose duels. Its co-citation half now runs whenever
    // memory supplies co-citation links (`NANO_W6_COCITE=off` disables it).
    let w6 = rich && !std::env::var("NANO_W6").is_ok_and(|v| v == "off");
    let w6_cocite = rich && !std::env::var("NANO_W6_COCITE").is_ok_and(|v| v == "off");
    let lane_early: Vec<usize> = if w2 && stack_lane_on() { lane_shortlist(sub, question) } else { Vec::new() };
    // Lane DR's drone kind as G2's set membership evidence (computed asynchronously on GPU while
    // CPU runs `note_sense`, `compose`, and `G1` tentacle scans; joined right before G2's `fill_by`).
    let drone_member_bg: std::cell::RefCell<Option<std::thread::JoinHandle<Vec<f32>>>> = std::cell::RefCell::new(None);
    let rw = active_retrieval_weights();
    let one_rc = std::cell::RefCell::new(if w2 {
        let posed: Option<Vec<f32>> = meaning
            .as_ref()
            .and_then(|m| m.first().cloned())
            .or_else(|| lane.as_ref().and_then(|l| l.cosines_now(sub, question)).and_then(|m| m.into_iter().next()));
        let kinds = w2_kinds();
        let n = sub.chunks.len();
        let mut cols: Vec<(f32, Vec<f32>)> = Vec::new();
        if kinds.iter().any(|k| k == "wording") {
            cols.push((rw.lexical, (0..n).map(|c| words.get(c).copied().unwrap_or(0.0)).collect()));
        }
        if kinds.iter().any(|k| k == "meaning")
            && let Some(m) = &posed
        {
            cols.push((rw.semantic, (0..n).map(|c| m.get(c).copied().unwrap_or(f32::NAN)).collect()));
        }
        if kinds.iter().any(|k| k == "settle")
            && let Some(ev) = &calibrated
        {
            // Per aspect, the mean settle chance over its parts, added over the aspects: the
            // request drone's finer parts split an aspect into several parts, and the plain sum
            // then weighed split aspects more and reordered the value (closure c06 lost first
            // with the finer parts on, held it off, 2026-11-18 runs); with one part an aspect it
            // equals the plain sum.
            let mut by_aspect: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for (k, p) in ev.parts.iter().enumerate() {
                by_aspect.entry(p.0).or_default().push(k);
            }
            #[expect(clippy::cast_precision_loss, reason = "a count of parts")]
            let col: Vec<f32> = if by_aspect.is_empty() || std::env::var("NANO_SETTLE_ASPECT").is_ok_and(|v| v == "off") {
                (0..n).map(|c| ev.settle_sum(c)).collect()
            } else {
                (0..n).map(|c| by_aspect.values().map(|ks| ks.iter().map(|k| ev.settles(*k, c)).sum::<f32>() / ks.len().max(1) as f32).sum()).collect()
            };
            cols.push((rw.lexical, col));
        }
        if kinds.iter().any(|k| k == "lane") && !lane_early.is_empty() {
            let mut v = vec![0.0_f32; n];
            for (i, c) in lane_early.iter().enumerate() {
                if let Some(x) = v.get_mut(*c) {
                    #[expect(clippy::cast_precision_loss, reason = "a place")]
                    let place = i as f32;
                    *x = 1.0 / (1.0 + place);
                }
            }
            cols.push((rw.semantic, v));
        }
        // The scale: the request's leading pieces by belief (as a window holds them).
        let mut lead: Vec<usize> = (0..n).collect();
        lead.sort_by(|a, b| fused.get(*b).copied().unwrap_or(0.0).total_cmp(&fused.get(*a).copied().unwrap_or(0.0)).then(a.cmp(b)));
        lead.truncate(W2_POOL);
        // Roadmap Phase 4 (2026-11-24): `NANO_STACK_JOINT` as a kind (`joint`, with the switch or
        // named in `NANO_W2_KINDS`). Its old reorder of G2's set runs only without W2; here each
        // code piece of the lead takes its place in the joint order over the lead's code pieces
        // (`Evidence::joint_order`, fusion constant from `Evidence::plausible`, as the reorder),
        // in the lane's form: 1 / (1 + place), 0 for a piece outside the lead, weighed as settle
        // is (the joint value reads the same settle evidence). A lead piece that restates (a
        // document, notes) has no value here, so it keeps the place the other kinds give it, as
        // the reorder kept its place in G1's order.
        if kinds.iter().any(|k| k == "joint")
            && let Some(ev) = &calibrated
        {
            let restates = |c: usize| keep.record(c).is_some_and(|r| nmpod::evidence::tag(r, &[], nmtype::is_document(&file_name(c))) == nmpod::evidence::Tag::Documents);
            let file_of = |c: usize| sub.chunks.get(c).map_or(usize::MAX, |x| x.file);
            let code: Vec<usize> = lead.iter().copied().filter(|c| !restates(*c)).collect();
            if !code.is_empty() {
                #[expect(clippy::cast_precision_loss, reason = "a count of pieces")]
                let k = ev.plausible(&code) as f32;
                // Roadmap Phase 4.6 (2026-11-24): a partial column, as the lane's. With a value for
                // every code piece of the lead the column was full, so 1 / (1 + place) put the
                // joint's first ten or so pieces at the +3 clip each (the gap the close-call judge
                // lost to on closure c20 and c23) and pushed the answers other kinds had placed
                // out of the pack (dev-large-2 pack 91 to 87 at a half or a quarter weight).
                // Only the joint's plausible places (`Evidence::plausible`, the count the reorder
                // used) carry a value; every other piece has none, so the column lifts by at most
                // one unit and lowers no piece.
                // Below half the pool, so the fusion always reads it as partial.
                let plausible = ev.plausible(&code).min(lead.len().saturating_sub(1) / 2);
                let mut v = vec![f32::NAN; n];
                // 2026-11-25 (IN_TROUBLE fix, the dev-large-2 pack drop): a document of the lead
                // had no value, so every valued code piece rose past it. Now the column's order is
                // the lead's order with its code slots refilled in the joint order: a document keeps
                // its slot and takes that slot's value, so the column lifts documents and code alike
                // by place and no longer lifts code over documents. Values stop after the last of the
                // joint's plausible code places, as before (a partial column).
                // 2026-11-26 (IN_TROUBLE fix 2, diagnosed on the large code tuning set: the joint's
                // first slot outranked the value's own leader on val15, val19 and trc24, each answer
                // falling from first to second): the fused leader keeps slot 0, as a document keeps
                // its slot, so the joint reorders below the leader and never displaces it.
                let leader = lead.first().copied();
                let mut joint = ev.joint_order(&code, &file_of, k).into_iter().map(|(c, _)| c).filter(move |c| Some(*c) != leader);
                let (mut place, mut valued) = (0usize, 0usize);
                for (i, slot) in lead.iter().enumerate() {
                    if valued >= plausible {
                        break;
                    }
                    let c = if i == 0 {
                        if !restates(*slot) {
                            valued += 1;
                        }
                        *slot
                    } else if restates(*slot) {
                        *slot
                    } else {
                        let Some(c) = joint.next() else { break };
                        valued += 1;
                        c
                    };
                    if let Some(x) = v.get_mut(c) {
                        #[expect(clippy::cast_precision_loss, reason = "a place")]
                        let p = place as f32;
                        *x = 1.0 / (1.0 + p);
                    }
                    place += 1;
                }
                // Roadmap Phase 4.6 (2026-11-24): its own weight, a share of settle's (default
                // one half, `NANO_W2_JOINT_WEIGHT`), measured on the boards.
                let share = std::env::var("NANO_W2_JOINT_WEIGHT").ok().and_then(|x| x.parse::<f32>().ok()).filter(|x| x.is_finite() && *x >= 0.0).unwrap_or(0.5);
                cols.push((rw.lexical * share, v));
            }
        }
        // W6: memory's co-citation links as a kind (they were dropped under the Fixed forward
        // rule): a piece's prior is its strongest link, over the strongest, to the value's first
        // `CLOSE_K` pieces.
        // Lane DR's kind (`drone`, a default kind since 2026-11-19): each lead piece's depth-k read
        // by its own drone against the request's drone (`drone_dendrite::kind_column`), with the
        // intake's relevance keys when the pass has them; only with the reader on the card.
        if kinds.iter().any(|k| k == "drone")
            && warm_reader() == JudgeState::Card
            && let Ok(model) = crate::neural::model(&nmtype::nano_root())
        {
            let relevance = crate::tools::installed().and_then(|o| o.whole.read().ok().and_then(|w| w.clone())).and_then(|w| w.relevance_keys());
            let td = Instant::now();
            // A kind of G2's set membership (not of the order): as an order kind it moved the
            // close-call judge's leader off answers (closure c20); in the set it decides which
            // pieces get in beside the value's.
            // The read's pieces: the first `DRONE_READ` of the lead that are at most
            // `DRONE_TOKENS` long (one long piece made the read take 190 s on the Mini's tree;
            // the time grows faster than the tokens).
            let read_lead: Vec<usize> = lead.iter().copied().filter(|c| tk(*c) <= DRONE_TOKENS).take(DRONE_READ).collect();
            let items: Vec<(usize, String, String)> = read_lead
                .iter()
                .copied()
                .filter(|c| *c < n)
                .map(|c| {
                    let found = sub.chunks.get(c).and_then(|ch| sub.files.get(ch.file).map(|f| (ch, f)));
                    match found {
                        Some((ch, f)) => (c, nmsubs::dense::doc_text(&f.path, &ch.text), format!("{}:{}", f.path, ch.start)),
                        None => (c, String::new(), String::new()),
                    }
                })
                .collect();
            let lead_bg = lead.clone();
            let q_bg = question.to_owned();
            let rw_struct = rw.structural;
            let stage_times = std::env::var_os("NANO_STAGE_TIMES").is_some();
            *drone_member_bg.borrow_mut() = Some(std::thread::spawn(move || {
                let col = crate::drone_dendrite::kind_column_owned(&model, relevance.as_deref(), n, &items, &q_bg);
                let out = fuse_kinds_weighted(&[(rw_struct, col)], &lead_bg);
                if stage_times {
                    eprintln!("STAGE_TIMES drone kind (async bg) {:.3} s", td.elapsed().as_secs_f64());
                }
                out
            }));
        }
        // The writer's notes as a sense (lane B's notes written at idle; a kind only where notes
        // exist): the share of the request's content words a piece's note holds.
        {
            let _ = nmpod::intake::install_notes(Arc::new(crate::writer_idle::note));
            let qw = nmpod::intake::note_words(question);
            let col: Vec<f32> = (0..n).map(|c| sub.chunks.get(c).and_then(|ch| nmpod::intake::note_sense(&ch.key, &qw)).unwrap_or(f32::NAN)).collect();
            if col.iter().any(|x| x.is_finite() && *x > 0.0) {
                cols.push((rw.structural, col));
            }
        }
        if w6_cocite && !cocited.is_empty() {
            let base = fuse_kinds_weighted(&cols, &lead);
            let mut seeds: Vec<usize> = (0..n).collect();
            seeds.sort_by(|a, b| base.get(*b).copied().unwrap_or(f32::MIN).total_cmp(&base.get(*a).copied().unwrap_or(f32::MIN)).then(a.cmp(b)));
            seeds.truncate(CLOSE_K);
            let top = cocited.values().flatten().map(|(_, w)| *w).fold(0.0_f32, f32::max).max(1e-9);
            let mut co = vec![0.0_f32; n];
            for s in &seeds {
                let Some(links) = sub.chunks.get(*s).and_then(|ch| cocited.get(&ch.key)) else { continue };
                for (k, w) in links {
                    if let Some(c) = by_key.get(k.as_str())
                        && let Some(x) = co.get_mut(*c)
                    {
                        *x = x.max(w / top);
                    }
                }
            }
            cols.push((rw.relationship, co));
        }
        fuse_kinds_weighted(&cols, &lead)
    } else {
        Vec::new()
    });
    // W3: the value before any read, and the reads so far (piece, signed read score).
    let one_base: Vec<f32> = one_rc.borrow().clone();
    let read_all: std::cell::RefCell<BTreeMap<usize, f32>> = std::cell::RefCell::new(BTreeMap::new());
    // C132 (`NANO_OWN`, on by default; `NANO_OWN=off`): each tentacle's own best by its perspective
    // (its lens's score over its window; its read's order where it read), `OWN_K` a read, kept
    // for G2's set beside the shared value's top.
    let own_on = w2 && !std::env::var("NANO_OWN").is_ok_and(|v| v == "off");
    let own_all: std::cell::RefCell<Vec<usize>> = std::cell::RefCell::new(Vec::new());
    // Each read's own first two, by window (the own-pick read takes each tentacle's best answering).
    let own_by_t: std::cell::RefCell<Vec<Vec<usize>>> = std::cell::RefCell::new(Vec::new());
    // Each integration read's first piece and its read value (the tentacle's own read's find).
    let own_readfirst: std::cell::RefCell<Vec<(usize, f32)>> = std::cell::RefCell::new(Vec::new());
    // Each tentacle's single best by its own perspective, every round (the picks G2 keeps).
    let own_first: std::cell::RefCell<Vec<usize>> = std::cell::RefCell::new(Vec::new());
    // One window's `g0_tx`: held spans by reference, its score sets (its lens's statistic first),
    // the drone table, the drones' readout, G2's reflexes; and its gate depth.
    let make = |t: usize, w: &[usize], reader: Option<&Lens>, read: Option<&[f32]>| -> (G0, usize) {
        let held: Vec<HeldChunk> = w
            .iter()
            .filter_map(|c| {
                let ch = sub.chunks.get(*c)?;
                let file = sub.files.get(ch.file)?.path.replace('\\', "/");
                Some(HeldChunk { span: SpanRef { file, start: ch.start, end: ch.end, key: ch.key.clone() }, chunk: *c, tokens: tk(*c) })
            })
            .collect();
        let over = |v: &dyn Fn(usize) -> f32| -> Vec<f32> { held.iter().map(|h| v(h.chunk)).collect() };
        // Every drone signal is its own score set; the lens's statistic comes first, so the
        // `g0_tx` reads by it, and a re-read turns to a signal not yet used.
        let mut sets: Vec<(String, Vec<f32>)> = Vec::new();
        if w2 {
            sets.push(("one".to_owned(), over(&|c| one_rc.borrow().get(c).copied().unwrap_or(f32::MIN))));
        }
        if let Some(l) = reader {
            sets.push((l.statistic.to_owned(), over(&|c| l.score.get(c).copied().unwrap_or(0.0))));
        }
        for &(name, g) in SIGNALS.iter().take(if rich { SIGNALS.len() } else { 1 }) {
            if sets.iter().all(|(n, _)| n != name) {
                sets.push((name.to_owned(), over(&|c| signals.get(c).map_or(0.0, g))));
            }
        }
        // The tentacle's integration read (D7), last: only on windows that integrated, a value
        // on the pieces read (minus infinity elsewhere, which G1's `reading` family skips).
        if let Some(v) = read {
            sets.push(("read".to_owned(), v.to_vec()));
        }
        // The drone table: every sense of this tentacle's own drones over its own pieces.
        let rows: Vec<Vec<f32>> = SIGNALS.iter().map(|(_, g)| over(&|c| signals.get(c).map_or(0.0, *g))).collect();
        let scores = sets.first().map(|s| s.1.clone()).unwrap_or_default();
        let best = scores.iter().copied().fold(0.0_f32, f32::max);
        let mut files: Vec<String> = held.iter().map(|h| h.span.file.clone()).collect();
        files.sort_unstable();
        files.dedup();
        let shares = over(&|c| fused.get(c).copied().unwrap_or(0.0));
        let rec = WindowRecord {
            tentacle: u8::try_from(t + 1).unwrap_or(u8::MAX),
            lens: reader.map_or("drones", |l| l.name).into(),
            pose: "plain".into(),
            chunks: w.to_vec(),
            tokens: held.iter().map(|h| h.tokens).sum(),
            level: 1,
            findings: 0,
            best,
            baseline: 0.0,
            no_information: best <= 0.0,
            files,
        };
        let (depth, _) = nmbrain::g1::shortlist_by(nmbrain::g1::rules().shortlist, &scores, SHORTLIST_FALLBACK);
        // G2's reflexes with what they need: R2 companions (a held chunk defining an item this one
        // uses, heaviest by the reference graph) and R4 keys (held spans of the organism's own edits).
        let mut companions: BTreeMap<String, usize> = BTreeMap::new();
        let mut self_edits = std::collections::BTreeSet::new();
        if !g2.reflexes.is_empty() {
            for h in &held {
                if let Some(j) = graph
                    .neighbours(h.chunk)
                    .into_iter()
                    .filter(|(_, rel, _)| *rel == nmsubs::refgraph::Relation::Uses)
                    .find_map(|(j, _, _)| held.iter().position(|x| x.chunk == j))
                {
                    companions.insert(h.span.key.clone(), j);
                }
                if g2.own_files.iter().any(|f| f.replace('\\', "/") == h.span.file) {
                    self_edits.insert(h.span.key.clone());
                }
            }
        }
        // The drones' readout of each held piece: the aspects it holds, its epistemic tag for the
        // aspect it holds best, and the pieces related to it that this window does not hold.
        let readout = calibrated.as_ref().map(|ev| nmbrain::g0::Readout {
            aspects: ev.aspects.clone(),
            accepted: held.iter().map(|h| (0..ev.aspects.len()).any(|a| ev.accepted(a, h.chunk))).collect(),
            covers: held.iter().map(|h| ev.covers(h.chunk)).collect(),
            quorum: held.iter().map(|h| ev.families(h.chunk)).collect(),
            tags: held
                .iter()
                .map(|h| {
                    let a = (0..ev.aspects.len()).max_by(|x, y| ev.aspect(*x, h.chunk).total_cmp(&ev.aspect(*y, h.chunk))).unwrap_or(0);
                    let doc = nmtype::is_document(&h.span.file);
                    keep.record(h.chunk).map_or(String::new(), |r| {
                        nmpod::evidence::tag(r, ev.stems.get(a).map_or(&[][..], Vec::as_slice), doc).name().to_owned()
                    })
                })
                .collect(),
            related: held
                .iter()
                .map(|h| graph.neighbours(h.chunk).into_iter().map(|(j, _, _)| j).filter(|j| !w.contains(j)).filter_map(span_of).collect())
                .collect(),
        });
        let held_common = shared.as_ref().map(|cs| (over(&|c| cs.values.get(c).copied().unwrap_or(0.0)), cs.floor));
        let g0 = G0::from_window(&rec, held, sets, &rows, depth).with_reflexes(g2.reflexes.clone(), companions, self_edits);
        let g0 = match readout {
            Some(r) => g0.with_readout(r),
            None => g0,
        };
        let g0 = match held_common {
            Some((values, floor)) => g0.on_common_scale(values).with_floor(Some(floor)),
            None if common => g0.on_common_scale(shares),
            None => g0,
        };
        (g0, depth)
    };

    // Lane M's motor path on G1's ledger (Phase H wiring, 2026-11-19): one ledger for the call,
    // G1's (set on G1 below), so every envelope down is recorded by G1 first (G1-b); G2's I9 gains go
    // through `Down::gain` (bounded 0.25 to 4, with expiry) and G1 reads them by `gain_vector`.
    // Phase J (lane M's configuration record): the call's orders and chain as labels.
    let config_orders: std::cell::RefCell<Vec<(String, bool)>> = std::cell::RefCell::new(Vec::new());
    let config_chain: std::cell::RefCell<Vec<String>> = std::cell::RefCell::new(Vec::new());
    let config_lenses: std::cell::RefCell<Vec<String>> = std::cell::RefCell::new(Vec::new());
    let call_ledger = nmbrain::g1::Ledger::default();
    let down = std::cell::RefCell::new(nmbrain::motor::Down::with_ledger(call_ledger.clone()));
    type RoutedEdge = (&'static str, &'static str, nmtype::class::DataClass, Option<nmtype::message::Verb>);
    let routed_edges: std::cell::RefCell<Vec<RoutedEdge>> = std::cell::RefCell::new(Vec::new());
    // G2 to G1: gains (standing and this question's I9, I1's gain on the tentacles holding its
    // files) and the instructions G1 relays in its first round (I1, I2, I3), against `windows`.
    let g2_relay = |windows: &[Vec<usize>]| -> (Vec<f32>, Vec<(usize, nmbrain::g0::Instruction, String)>) {
        use nmtype::message::{Arg, MessageKind, Motor, Verb};
        let n = windows.len();
        let now = nmtype::now_ms();
        let expiry = now.saturating_add(60_000);
        let mut gains: Vec<f32> = (0..n).map(|t| g2.gains.get(t).copied().unwrap_or(1.0)).collect();
        let mut relay: Vec<(usize, nmbrain::g0::Instruction, String)> = Vec::new();
        let holder = |c: usize| windows.iter().position(|w| w.contains(&c));
        let ids: Vec<String> = (0..n).map(|t| nmbrain::g0::g0_id(u8::try_from(t + 1).unwrap_or(u8::MAX))).collect();
        for i in &g2.orders {
            match i {
                Instr::Gain { tentacle, g, .. } => {
                    let mut d = down.borrow_mut();
                    let id = d.id();
                    let sent = ids.get(*tentacle).and_then(|who| {
                        Motor::new(id, None, MessageKind::Instruction, Verb::Gain, nmbrain::motor::G2, "g1", 0.5, "G2's gain on one tentacle", expiry)
                            .and_then(|m| m.arg("tentacle", Arg::Word(who.clone())))
                            .and_then(|m| m.arg("gain", Arg::Scale(*g)))
                            .ok()
                    });
                    match sent.map(|m| d.gain(m, now)) {
                        Some(Ok(_)) => routed_edges.borrow_mut().push(("g2", "g1", nmtype::class::DataClass::Behavioural, Some(Verb::Gain))),
                        _ => {
                            if let Some(x) = gains.get_mut(*tentacle) {
                                *x *= g;
                            }
                        }
                    }
                }
                Instr::Focus { files, terms, gain } => {
                    let focus = nmbrain::g0::Focus {
                        lens: "drones".into(),
                        pose: "focus".into(),
                        statistic: String::new(),
                        files: files.clone(),
                        terms: terms.clone(),
                    };
                    let holders: Vec<usize> = windows.iter().enumerate().filter(|(_, w)| files.is_empty() || w.iter().any(|c| files.contains(&file_name(*c)))).map(|(t, _)| t).collect();
                    for (t, w) in windows.iter().enumerate() {
                        if files.is_empty() || w.iter().any(|c| files.contains(&file_name(*c))) {
                            // F21: a focus about one tentacle's own window is that tentacle's own
                            // focus (direct route, G1's copy recorded first); one about several
                            // coordinates them and goes through G1.
                            if holders.len() == 1 {
                                let mut d = down.borrow_mut();
                                let id = d.id();
                                if let Some(who) = ids.get(t)
                                    && let Ok(m) = Motor::new(id, None, MessageKind::Instruction, Verb::Focus, nmbrain::motor::G2, who, 0.5, "G2's focus on one tentacle's own window", expiry)
                                    && d.ledger.record(&m).is_ok()
                                {
                                    d.direct += 1;
                                    d.sent.push(m);
                                    routed_edges.borrow_mut().push(("g2", "g0", nmtype::class::DataClass::Behavioural, Some(Verb::Focus)));
                                }
                            }
                            relay.push((t, nmbrain::g0::Instruction::Reread(focus.clone()), "G2 I1 FOCUS".into()));
                            if let Some(x) = gains.get_mut(t) {
                                *x *= gain;
                            }
                        }
                    }
                }
                Instr::Reread { chunks, mode } => {
                    let mut by_t: BTreeMap<usize, Vec<String>> = BTreeMap::new();
                    for c in chunks {
                        if let Some(t) = holder(*c) {
                            by_t.entry(t).or_default().push(file_name(*c));
                        }
                    }
                    for (t, files) in by_t {
                        let focus =
                            nmbrain::g0::Focus { lens: "drones".into(), pose: "focus".into(), statistic: String::new(), files, terms: Vec::new() };
                        relay.push((t, nmbrain::g0::Instruction::Reread(focus), format!("G2 I2 REREAD ({mode:?})")));
                    }
                }
                Instr::Answer { chunk, about } => {
                    if let (Some(t), Some(span)) = (holder(*chunk), span_of(*chunk)) {
                        relay.push((t, nmbrain::g0::Instruction::Ask { span, about: about.clone() }, "G2 I3 ANSWER".into()));
                    }
                }
                _ => {}
            }
        }
        // G2's gains in force on each tentacle, recorded in G1's ledger (`Down::gain_vector`).
        let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        for (x, v) in gains.iter_mut().zip(down.borrow().gain_vector(&id_refs, now)) {
            *x *= v;
        }
        (gains, relay)
    };
    // The receiving room in tokens: the caller's count of pieces at the mean held size when it
    // names one (the whole path's judges), else the pack fallback.
    let room_of = |mean: usize| up.map_or(g2.room_tokens.unwrap_or(PACK_TOKENS).max(1), |n| n.max(1) * mean.max(1));
    let structure = || {
        (nmbrain::g1::rules().forward == nmbrain::g1::ForwardRule::Queue).then(|| {
            if cocited.is_empty() {
                StructureRef(Arc::new(SubGraph(sub.refs())) as Arc<dyn nmbrain::evidence::Structure>)
            } else {
                StructureRef(Arc::new(crate::tentacle_core::WithCoCitation { base: SubGraph(sub.refs()), links: cocited.clone() })
                    as Arc<dyn nmbrain::evidence::Structure>)
            }
        })
    };
    let has = |c: nmbrain::reflex::Code| g2.reflexes.iter().any(|r| r.code == c && r.live());
    // The tentacles' read (D7, `NANO_TENTACLE_READ=on`): the call's card tokens left, the round
    // count, the largest read score so far (the `reading` family's ceiling) and the trace lines.
    // The read runs only with the reader on the card (warmed in the background, never on the
    // processor; until it is ready, or without a card, the windows report without it).
    let read_on = rich && crate::tentacle_read::on() && warm_reader() == JudgeState::Card;
    let read_budget = std::env::var("NANO_READ_TOKENS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(crate::tentacle_read::CALL_TOKENS_FALLBACK);
    let read_left = std::cell::Cell::new(read_budget);
    let read_round = std::cell::Cell::new(0_usize);
    let read_top = std::cell::Cell::new(0.0_f32);
    let read_lines: std::cell::RefCell<Vec<String>> = std::cell::RefCell::new(Vec::new());
    let read_cos: std::cell::OnceCell<Option<Vec<f32>>> = std::cell::OnceCell::new();
    // The stage ladder (`NANO_LADDER=<file>`, a measurement): per window read, its tentacle, round,
    // pieces in fused order and the pieces the read integrated in read order.
    let ladder = std::env::var_os("NANO_LADDER").filter(|v| !v.is_empty());
    let ladder_windows: std::cell::RefCell<Vec<String>> = std::cell::RefCell::new(Vec::new());
    let ladder_stages: std::cell::RefCell<Vec<String>> = std::cell::RefCell::new(Vec::new());
    let ladder_round = std::cell::Cell::new(0_usize);
    let ladder_ref = |c: usize| sub.chunks.get(c).map_or_else(String::new, |ch| format!("{}:{}-{}", file_name(c), ch.start, ch.end));
    // One stage: its name and its order (the first `LADDER_DEPTH` pieces) as references.
    let lad = |name: &str, v: &[usize]| {
        if ladder.is_some() {
            let r: Vec<String> = v.iter().copied().take(LADDER_DEPTH).map(&ladder_ref).collect();
            ladder_stages.borrow_mut().push(format!("{name:?}:{r:?}"));
        }
    };
    // One stage's value that is not an order (a gate, a count, the request's parts), as JSON.
    let lad_value = |name: &str, json: String| {
        if ladder.is_some() {
            ladder_stages.borrow_mut().push(format!("{name:?}:{json}"));
        }
    };
    // W1, the drones: the request's parts and aspects, and the material in the order of each
    // scale the flow uses (B belief, V pool value, S settle sum, M the meaning lane's cosine).
    if ladder.is_some() {
        let by = |v: &dyn Fn(usize) -> f32| -> Vec<usize> {
            let mut o: Vec<usize> = (0..sub.chunks.len()).collect();
            o.sort_by(|a, b| v(*b).total_cmp(&v(*a)).then(a.cmp(b)));
            o
        };
        if let Some(ev) = &calibrated {
            let parts: Vec<&str> = ev.parts.iter().map(|p| p.1.as_str()).collect();
            lad_value("request", format!("{{\"kind\":{:?},\"aspects\":{:?},\"parts\":{parts:?}}}", format!("{:?}", nmpod::intake::Request::read(question, &[]).kind), ev.aspects));
            lad("drones.S", &by(&|c| ev.settle_sum(c)));
        }
        lad("drones.B", &by(&|c| fused.get(c).copied().unwrap_or(0.0)));
        if w2 {
            lad("drones.F", &by(&|c| one_rc.borrow().get(c).copied().unwrap_or(f32::MIN)));
        }
        if let Some(cs) = &shared {
            lad("drones.V", &by(&|c| cs.values.get(c).copied().unwrap_or(0.0)));
        }
        if let Some(m) = meaning.as_ref().and_then(|m| m.first()) {
            lad("drones.M", &by(&|c| m.get(c).copied().filter(|x| x.is_finite()).unwrap_or(f32::MIN)));
        }
    }
    let read_solo = std::env::var("NANO_TENTACLE_READ_SOLO").is_ok_and(|v| v == "on");
    let g0_resid = !std::env::var("NANO_G0_RESID").is_ok_and(|v| v == "off");
    // One tentacle system (lane DR's assembled core, `designs/tentacle-core.json`; `NANO_CORE=off`):
    // one core per tentacle for the call, kept across rounds (its recurrent state); each round it
    // takes its window's pieces with their log-odds (the read's where it read, else the fused value),
    // makes its local decisions (report depth by posterior mass, no information, the deep check when
    // its gate calls and the reader is on the card within the call's time) and its depth is the g0's.
    let core_on = rich && !std::env::var("NANO_CORE").is_ok_and(|v| v == "off");
    // Seconds the call spent on deep checks and on integration reads (the stage times' record).
    let deep_spent: std::cell::Cell<f64> = std::cell::Cell::new(0.0);
    let read_spent: std::cell::Cell<f64> = std::cell::Cell::new(0.0);
    let deep_fallbacks: std::cell::Cell<usize> = std::cell::Cell::new(0);
    let core0: Option<crate::tentacle_core::Core> = if core_on { crate::tentacle_core::Core::from_design(&nmtype::nano_root()).ok() } else { None };
    let cores: std::cell::RefCell<BTreeMap<usize, crate::tentacle_core::Core>> = std::cell::RefCell::new(BTreeMap::new());
    let core_depth: std::cell::RefCell<BTreeMap<usize, usize>> = std::cell::RefCell::new(BTreeMap::new());
    let core_reports: std::cell::RefCell<BTreeMap<usize, nmbrain::g0::Report>> = std::cell::RefCell::new(BTreeMap::new());

    let drone_sets = if read_on { Some(crate::tentacle_pass::drones(&nmtype::nano_root())) } else { None };
    // Each read's window evidence for the read's shortlist: wording (plain BM25) and the meaning
    // lane's posed cosine (its first row: the stack's, else read now), each standardised in the
    // window, summed.
    let window_fused = |reads: &[Read]| -> Vec<Vec<f32>> {
        let posed = read_cos.get_or_init(|| {
            meaning.as_ref().and_then(|m| m.first().cloned()).or_else(|| lane.as_ref().and_then(|l| l.cosines_now(sub, question)).and_then(|m| m.into_iter().next()))
        });
        let cos = |c: usize| posed.as_ref().and_then(|m| m.get(c).copied()).filter(|x| x.is_finite()).unwrap_or(-1.0);
        reads
            .iter()
            .map(|r| {
                let b: Vec<f32> = r.window.iter().map(|c| words.get(*c).copied().unwrap_or(0.0)).collect();
                if posed.is_some() {
                    let m: Vec<f32> = r.window.iter().map(|c| cos(*c)).collect();
                    crate::tentacle_read::fuse(&[&b, &m])
                } else {
                    crate::tentacle_read::fuse(&[&b])
                }
            })
            .collect()
    };
    // The ladder's record of one round's windows: fused order, and read order where `sets` holds one.
    let ladder_record = |reads: &[Read], sets: &[Option<Vec<f32>>], orders: &[Vec<usize>], finds: &[Vec<usize>]| {
        if ladder.is_none() {
            return;
        }
        ladder_round.set(ladder_round.get() + 1);
        let fused = window_fused(reads);
        let mut out = ladder_windows.borrow_mut();
        for (i, ((r, f), set)) in reads.iter().zip(&fused).zip(sets).enumerate() {
            let mut by: Vec<usize> = (0..r.window.len()).collect();
            by.sort_by(|a, b| f.get(*b).copied().unwrap_or(f32::MIN).total_cmp(&f.get(*a).copied().unwrap_or(f32::MIN)).then(a.cmp(b)));
            let fo: Vec<usize> = by.iter().filter_map(|k| r.window.get(*k).copied()).collect();
            let ro: Vec<usize> = set.as_ref().map_or_else(Vec::new, |v| {
                let mut k: Vec<usize> = (0..v.len()).filter(|i| v.get(*i).is_some_and(|x| x.is_finite())).collect();
                k.sort_by(|a, b| v.get(*b).copied().unwrap_or(f32::MIN).total_cmp(&v.get(*a).copied().unwrap_or(f32::MIN)).then(a.cmp(b)));
                k.iter().filter_map(|i| r.window.get(*i).copied()).collect()
            });
            let fo: Vec<String> = fo.into_iter().map(&ladder_ref).collect();
            let ro: Vec<String> = ro.into_iter().map(&ladder_ref).collect();
            let g0: Vec<String> = orders.get(i).map_or_else(Vec::new, |o| o.iter().copied().map(&ladder_ref).collect());
            let rep: Vec<String> = finds.get(i).map_or_else(Vec::new, |o| o.iter().copied().map(&ladder_ref).collect());
            out.push(format!(
                "{{\"t\":{},\"round\":{},\"lens\":{:?},\"label\":{:?},\"g0\":{g0:?},\"reported\":{rep:?},\"fused\":{fo:?},\"read\":{ro:?}}}",
                r.tentacle + 1,
                ladder_round.get(),
                r.reader.as_ref().map_or("drones", |l| l.name),
                r.label
            ));
        }
    };
    // One round's reads: per read its `read` score set (window order) when it integrated, and the
    // count of reads whose solo read differs from the packed one.
    let tentacle_reads = |reads: &[Read]| -> (Vec<Option<Vec<f32>>>, usize) {
        let mut sets: Vec<Option<Vec<f32>>> = vec![None; reads.len()];
        let Some(ds) = drone_sets.as_ref() else { return (sets, 0) };
        read_round.set(read_round.get() + 1);
        let round = read_round.get();
        let fused = window_fused(reads);
        let toks: Vec<Vec<usize>> = reads.iter().map(|r| r.window.iter().map(|c| tk(*c)).collect()).collect();
        // Homeostasis: the read runs while its expected cost fits the call's time after the close-call
        // judge's reserve (the judge's value per second is the higher); else the windows report
        // without it.
        // The read runs on every call where its gate (the fused margin in the band) applies: the
        // call's time budget observes it but never withholds it (Luis, 2026-11-19: no part of the
        // architecture is effectively switched off; its time is brought down as its own work).
        let read_ok = true;
        // Integrations granted are a learnable site: G3's promoted bucket when it has one (`none`
        // grants nothing), else the rule (the budget's grant).
        let rule = if read_ok && read_left.get() > 0 { "some" } else { "none" };
        let grant_site = crate::g3::decide_site(nmmusc::priors::Site::Integrations, rule, false);
        let grant = if read_ok && grant_site != "none" { read_left.get() } else { 0 };
        let (plans, used) = crate::tentacle_read::plan(&fused, &toks, crate::tentacle_read::GATE_BAND, grant, crate::tentacle_read::TENTACLE_TOKENS_FALLBACK);
        read_left.set(read_left.get().saturating_sub(used));
        for (r, p) in reads.iter().zip(&plans) {
            let choice = if p.list.is_empty() { "granted:0".to_owned() } else { format!("granted:{}", p.list.len()) };
            let _ = crate::g3::note_site(nmmusc::sites::SiteKind::Integration, &format!("t{}", r.tentacle + 1), &format!("round{round}"), &choice, None);
        }
        let doc = |c: usize| sub.chunks.get(c).and_then(|ch| sub.files.get(ch.file).map(|f| nmsubs::dense::doc_text(&f.path, &ch.text))).unwrap_or_default();
        let heads_of = |r: &Read| -> Vec<(usize, usize)> {
            let name = r.reader.as_ref().map_or_else(|| nmsubs::lens::NAMES.get(r.tentacle).copied().unwrap_or("lens"), |l| l.name);
            ds.for_tentacle(name).to_vec()
        };
        // Integrated reads: (read index, chunks in relation order).
        let live: Vec<(usize, Vec<usize>)> = plans
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.list.is_empty())
            .filter_map(|(i, p)| {
                let r = reads.get(i)?;
                let list: Vec<usize> = p.list.iter().filter_map(|k| r.window.get(*k).copied()).collect();
                Some((i, crate::tentacle_read::relation_order(sub, &list)))
            })
            .collect();
        let items: Vec<crate::tentacle_read::Item> =
            live.iter().filter_map(|(i, order)| Some((order.iter().map(|c| doc(*c)).collect(), heads_of(reads.get(*i)?), question.to_owned()))).collect();
        let t = Instant::now();
        // Roadmap Phase 4.5 (2026-11-24): the tentacles' head reads in the live view.
        let mut head_reads = crate::live::stage("tentacle.read", "reader");
        head_reads.count("reads", items.len());
        // Which tentacles read, one bit each (bit 0 is the first), so the live view lights those only.
        head_reads.count("tentacle_mask", live.iter().filter_map(|(i, _)| reads.get(*i)).fold(0_usize, |m, r| if r.tentacle < 32 { m | (1 << r.tentacle) } else { m }));
        let got = if items.is_empty() {
            Ok((Vec::new(), 0))
        } else {
            crate::tentacle_read::served(&crate::tentacle_read::anchor_text(sub)).and_then(|ir| {
                let v = ir.read_packed(&items)?;
                // Solo equivalence with the assignment fixed: each read alone gives the same scores.
                let differ = if read_solo {
                    items
                        .iter()
                        .zip(&v)
                        .filter(|(it, rd)| {
                            ir.read_packed(std::slice::from_ref(*it)).ok().and_then(|x| x.into_iter().next()).is_none_or(|a| {
                                a.scores.len() != rd.scores.len() || a.scores.iter().zip(&rd.scores).any(|(x, y)| (x - y).abs() > 1e-3)
                            })
                        })
                        .count()
                } else {
                    0
                };
                Ok((v, differ))
            })
        };
        drop(head_reads);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        read_spent.set(read_spent.get() + ms / 1000.0);
        // Roadmap Phase 0 item 5 (WP-C5) probe (`NANO_HEAD_PROBE=<file>`, a measurement): every head's
        // calibrated score per piece for these same windows, with each window's lens.
        // Each tentacle's whole window is probed (in relation order), not only the contested list the
        // read was granted: the lists held an answer piece in 27 of 571 windows (six tuning sets).
        // Round 1 only (every lens has its own window there; later rounds made the probe about 112 s
        // a question).
        if let Some(pf) = std::env::var_os("NANO_HEAD_PROBE").filter(|_| round == 1) {
            // Each window's first pieces as it was filled (`NANO_HEAD_PROBE_PIECES`, default 8): whole
            // windows cost about 110 s a question to encode at every head.
            let cap = std::env::var("NANO_HEAD_PROBE_PIECES").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(8).max(2);
            let orders: Vec<Vec<usize>> = reads.iter().map(|r| crate::tentacle_read::relation_order(sub, &r.window.iter().copied().take(cap).collect::<Vec<usize>>())).collect();
            let pitems: Vec<crate::tentacle_read::Item> = reads.iter().zip(&orders).map(|(r, o)| (o.iter().map(|c| doc(*c)).collect(), heads_of(r), question.to_owned())).collect();
            let names: Vec<String> = reads.iter().map(|r| r.reader.as_ref().map_or_else(|| nmsubs::lens::NAMES.get(r.tentacle).copied().unwrap_or("lens"), |l| l.name).to_owned()).collect();
            let refs: Vec<Vec<String>> = orders.iter().map(|o| o.iter().map(|c| sub.chunks.get(*c).map_or_else(String::new, |ch| format!("{}:{}-{}", file_name(*c), ch.start, ch.end))).collect()).collect();
            if !pitems.is_empty()
                && let Ok(ir) = crate::tentacle_read::served(&crate::tentacle_read::anchor_text(sub))
            {
                let _ = ir.probe_heads(&pitems, &names, &refs, 16, std::path::Path::new(&pf));
            }
        }
        let mut lines = Vec::new();
        let mut solo = 0;
        let mut reports: BTreeMap<usize, (crate::tentacle_read::Report, usize)> = BTreeMap::new();
        match got {
            Ok((v, differ)) => {
                solo = differ;
                for ((i, order), mut rd) in live.iter().zip(v) {
                    // Phase E1 (`NANO_G0_RESID=off`): the read's scores residualised on its joined
                    // read's nuisance basis (place in the joined order, tokens to its end, log
                    // length, newline share) by a robust fit, so each tentacle's report is
                    // comparable with the others' (G0-R; the last-position penalty).
                    if g0_resid {
                        let toks: Vec<usize> = order.iter().map(|c| tk(*c)).collect();
                        let nl: Vec<f32> = order
                            .iter()
                            .map(|c| {
                                let t = sub.chunks.get(*c).map_or("", |ch| ch.text.as_str());
                                #[expect(clippy::cast_precision_loss, reason = "character counts")]
                                let share = t.matches('\n').count() as f32 / t.len().max(1) as f32;
                                share
                            })
                            .collect();
                        rd.scores = nmbrain::g0::residualise(&rd.scores, &nmbrain::g0::Nuisance::of_window(&toks, &nl));
                    }
                    let rep = crate::tentacle_read::report(order, &rd.scores, &rd.ratio);
                    let top = rd.scores.iter().copied().filter(|x| x.is_finite()).fold(f32::NEG_INFINITY, f32::max);
                    if top > read_top.get() {
                        read_top.set(top);
                    }
                    if let (Some(slot), Some(r)) = (sets.get_mut(*i), reads.get(*i)) {
                        let mut set = vec![f32::NEG_INFINITY; r.window.len()];
                        for (c, x) in order.iter().zip(&rd.scores) {
                            if let Some(k) = r.window.iter().position(|w| w == c)
                                && let Some(y) = set.get_mut(k)
                            {
                                *y = *x;
                            }
                        }
                        *slot = Some(set);
                    }
                    reports.insert(*i, (rep, rd.tokens));
                }
            }
            Err(e) => lines.push(format!("TENTACLE_READ round {round}: the read failed ({}); every window reports without it", e.message)),
        }
        // Labels: each tentacle's perspective, aspect, mode and round, whether it integrated, its
        // margin and no-information flag.
        for (i, (r, p)) in reads.iter().zip(&plans).enumerate() {
            let perspective = r.reader.as_ref().map_or("drones", |l| l.name);
            let lead = fused.get(i).and_then(|f| {
                let k = (0..f.len()).max_by(|a, b| f.get(*a).copied().unwrap_or(f32::MIN).total_cmp(&f.get(*b).copied().unwrap_or(f32::MIN)))?;
                r.window.get(k).copied()
            });
            let aspect = calibrated
                .as_ref()
                .zip(lead)
                .and_then(|(ev, c)| {
                    let a = (0..ev.aspects.len()).max_by(|x, y| ev.aspect(*x, c).total_cmp(&ev.aspect(*y, c)))?;
                    ev.aspects.get(a).cloned()
                })
                .unwrap_or_default();
            let head = format!("TENTACLE_READ t{} round {round}: perspective {perspective}, aspect \"{aspect}\", mode {}", r.tentacle + 1, r.label);
            match reports.get(&i) {
                Some((rep, tokens)) => lines.push(format!(
                    "{head}; integrated {} pieces, {tokens} tokens; read margin {:.3}, spread {:.3}, no-information {}; fused margin {:.3}",
                    rep.best.len(),
                    rep.margin,
                    rep.spread,
                    if rep.no_information { "yes" } else { "no" },
                    p.margin
                )),
                None => lines.push(format!(
                    "{head}; not integrated ({}), fused margin {:.3}; reports without the read",
                    if p.list.is_empty() { p.why } else { "read failed" },
                    p.margin
                )),
            }
        }
        if !items.is_empty() {
            lines.push(format!(
                "TENTACLE_READ round {round}: {} of {} windows integrated in one card run, {used} tokens, {ms:.0} ms; {} tokens left",
                items.len(),
                reads.len(),
                read_left.get()
            ));
        }
        if std::env::var_os("NANO_TENTACLE_READ_TRACE").is_some() {
            for l in &lines {
                eprintln!("{l}");
            }
        }
        read_lines.borrow_mut().extend(lines);
        (sets, solo)
    };
    // G1, the only path between the tentacles: one coordinating loop over one round's reads.
    let g1_round = |g1: &mut G1, reads: &[Read], relay_g2: bool| -> Round {
        let mut boxes: Vec<Box<dyn TentacleSide>> = Vec::new();
        let mut solos: Vec<G0> = Vec::new();
        let mut depths = Vec::new();
        let (read_sets, read_solo) = if read_on { tentacle_reads(reads) } else { (vec![None; reads.len()], 0) };
        // W3: the round's reads join the value as one more kind (signed, standardised over every
        // piece read in this call, clipped), before the g0s are made.
        // W3's second form (default; `NANO_W3_FORM=kind` the first): the read reorders only the
        // pieces it read, inside its own tentacle's shortlist: their fused values are dealt out to
        // them in the read's order (the best value to the best read), so the g0's report, G1's
        // hand-over and G2's set carry the read's order among those pieces and nothing else moves.
        let w3_kind = std::env::var("NANO_W3_FORM").is_ok_and(|v| v == "kind");
        if read_on && read_sets.iter().any(Option::is_some) {
            // Every read so far, kept for the Core Brain's reader hop (Phase I).
            let mut ra = read_all.borrow_mut();
            for (r, rs) in reads.iter().zip(&read_sets) {
                if let Some(v) = rs {
                    for (c, x) in r.window.iter().zip(v) {
                        if x.is_finite() {
                            ra.insert(*c, *x);
                        }
                    }
                }
            }
        }
        if w3 && read_on && !w3_kind && read_sets.iter().any(Option::is_some) {
            // Bounded additive evidence from the tentacles' integration reads: standardised over
            // every read piece and scaled to half a kind (0.5), added onto `one_base` so it breaks
            // close ties without overwriting a strong consensus of wording, meaning and settle
            // (replacing the earlier in-shortlist rank permutation that lost 13 win.g0 firsts).
            let ra = read_all.borrow();
            let keys: Vec<usize> = ra.keys().copied().collect();
            let mut col = vec![f32::NAN; one_base.len()];
            for (c, x) in ra.iter() {
                if let Some(y) = col.get_mut(*c) {
                    *y = *x;
                }
            }
            let add = fuse_kinds(&[col], &keys);
            let mut o = one_rc.borrow_mut();
            for ((y, b), a) in o.iter_mut().zip(&one_base).zip(&add) {
                *y = b + 0.5 * a;
            }
        }
        if w3 && read_on && w3_kind && read_sets.iter().any(Option::is_some) {
            let mut ra = read_all.borrow_mut();
            for (r, rs) in reads.iter().zip(&read_sets) {
                if let Some(v) = rs {
                    for (c, x) in r.window.iter().zip(v) {
                        if x.is_finite() {
                            ra.insert(*c, *x);
                        }
                    }
                }
            }
            let keys: Vec<usize> = ra.keys().copied().collect();
            let mut col = vec![f32::NAN; one_base.len()];
            for (c, x) in ra.iter() {
                if let Some(y) = col.get_mut(*c) {
                    *y = *x;
                }
            }
            let add = fuse_kinds(&[col], &keys);
            let mut o = one_rc.borrow_mut();
            for ((y, b), a) in o.iter_mut().zip(&one_base).zip(&add) {
                *y = b + a;
            }
        }
        if read_on
            && let Some(p) = g1.common.as_mut()
        {
            p.read_up_to(read_top.get());
        }
        if own_on {
            let mut own = own_all.borrow_mut();
            for (r, rs) in reads.iter().zip(&read_sets) {
                let mut by: Vec<(f32, usize)> = match rs {
                    Some(v) => r.window.iter().zip(v).filter(|(_, x)| x.is_finite()).map(|(c, x)| (*x, *c)).collect(),
                    None => r.reader.as_ref().map_or_else(Vec::new, |l| r.window.iter().map(|c| (l.score.get(*c).copied().unwrap_or(0.0), *c)).filter(|x| x.0 > 0.0).collect()),
                };
                by.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                if let Some((_, c)) = by.first() {
                    let mut f = own_first.borrow_mut();
                    if !f.contains(c) {
                        f.push(*c);
                    }
                }
                own_by_t.borrow_mut().push(by.iter().take(OWN_K).map(|x| x.1).collect());
                if rs.is_some()
                    && let Some((x, c)) = by.first()
                {
                    own_readfirst.borrow_mut().push((*c, *x));
                }
                for (_, c) in by.into_iter().take(OWN_K) {
                    if !own.contains(&c) {
                        own.push(c);
                    }
                }
            }
        }
        if let Some(c0) = core0.as_ref() {
            struct FlowIo<'a> {
                read: BTreeMap<usize, f32>,
                deep: Option<crate::tentacle_core::ReaderIo<'a>>,
                spent: &'a std::cell::Cell<f64>,
                /// This round's packed deep checks of this tentacle, by chunk (lane S's packed read).
                packed: BTreeMap<usize, f32>,
                fallbacks: &'a std::cell::Cell<usize>,
            }
            // The packer's probe (lane T1, 2026-11-19, speed): a dry round on a copy of the
            // tentacle's core with the same integration values records which pieces its deep check
            // asks for, so the round's packed run reads exactly those (the top two by the read
            // missed what the core asked on most calls: a fallback read each).
            struct Probe {
                read: BTreeMap<usize, f32>,
                asked: Option<Vec<usize>>,
            }
            impl crate::tentacle_core::CoreIo for Probe {
                fn integrate(&mut self, chunks: &[usize], _pose: &str) -> Option<Vec<f32>> {
                    let v: Vec<f32> = chunks.iter().filter_map(|c| self.read.get(c).copied()).collect();
                    (v.len() == chunks.len() && !v.is_empty()).then_some(v)
                }
                fn deep_check(&mut self, chunks: &[usize]) -> Option<Vec<f32>> {
                    if self.asked.is_none() {
                        self.asked = Some(chunks.to_vec());
                    }
                    None
                }
            }
            impl crate::tentacle_core::CoreIo for FlowIo<'_> {
                fn integrate(&mut self, chunks: &[usize], _pose: &str) -> Option<Vec<f32>> {
                    let v: Vec<f32> = chunks.iter().filter_map(|c| self.read.get(c).copied()).collect();
                    (v.len() == chunks.len() && !v.is_empty()).then_some(v)
                }
                fn deep_check(&mut self, chunks: &[usize]) -> Option<Vec<f32>> {
                    // Each deep check within the call's time (homeostasis, its own measured cost).
                    // Every deep check its gate calls for runs (the budget observes, it does not withhold).
                    let t = Instant::now();
                    let pk: Vec<f32> = chunks.iter().filter_map(|c| self.packed.get(c).copied()).collect();
                    if !chunks.is_empty() && pk.len() == chunks.len() {
                        return Some(pk);
                    }
                    // A fallback: a check the packed read did not cover, read on its own (a tentacle
                    // whose gate did not call for the deep check reads nothing and is not counted).
                    if self.deep.is_some() {
                        self.fallbacks.set(self.fallbacks.get() + 1);
                    }
                    let v = self.deep.as_mut().and_then(|d| d.deep_check(chunks));
                    self.spent.set(self.spent.get() + t.elapsed().as_secs_f64());
                    if let Ok(mut b) = nmbrain::homeostasis::TIME_BUDGET.lock() {
                        b.observe(nmbrain::homeostasis::Stage::Deep, t.elapsed().as_secs_f64(), v.as_ref().is_some_and(|x| x.iter().any(|y| *y > 0.0)));
                    }
                    v
                }
            }
            let reader = if warm_reader() == JudgeState::Card {
                crate::neural::model(&nmtype::nano_root()).ok()

            } else {
                None
            };
            // The deep check's design gate (master plan 3.3 step 7): the tentacle's read leaves its
            // leader contested, inside the noise band (its margin over the second at most the read
            // values' median absolute deviation), or an order formed the window. Bounded to at most
            // one packed card pass per call (`DEEP_CAP_S`, override `NANO_DEEP_CAP_S`) so
            // multi-round calls and G2 order rounds do not burn 0.8-1.8 s on repeated deep checks.
            // 2026-11-25: 0.35 s to 0.2 s. Full sets: large code tuning 26, 39, 47, 44, 42, 12/12 both,
            // median 2.358 to 2.307 s, worst 5.005 to 4.451 s; dev-large-2 verification 62, 80, 89, 83,
            // 84 to 63, 80, 89, 83, 84, 20/20, worst 5.302 to 4.713 s. With the check off the tuning set
            // looked free but dev-large-2 fell to 59 first (below the gate): the check stays on.
            let deep_cap_s: f64 = std::env::var("NANO_DEEP_CAP_S").ok().and_then(|v| v.parse().ok()).unwrap_or(0.2);
            let contested = |rs: Option<&[f32]>, label: &str| -> bool {
                if deep_spent.get() >= deep_cap_s {
                    return false;
                }
                if label.contains("order") {
                    return deep_spent.get() == 0.0;
                }
                let Some(v) = rs else { return false };
                let mut f: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
                if f.len() < 2 {
                    return false;
                }
                f.sort_by(|a, b| b.total_cmp(a));
                let margin = f[0] - f[1];
                margin <= crate::coordinator::noise_band(&f)
            };
            let heads = crate::tentacle_pass::drones(&nmtype::nano_root());
            let mut depths_now = core_depth.borrow_mut();
            depths_now.clear();
            // The round's deep checks as one packed card run (master plan 3.3 step 7 with lane S's
            // read_group_labels): for every tentacle its design gate admits, its top two pieces by
            // their log-odds (the read's where it read, else the value), cut to their first lines.
            let mut packed_by_t: BTreeMap<usize, BTreeMap<usize, f32>> = BTreeMap::new();
            let input_of = |r: &Read, rs: Option<&[f32]>| -> (BTreeMap<usize, f32>, crate::tentacle_core::RoundInput) {
                let one = one_rc.borrow();
                let read: BTreeMap<usize, f32> = rs.map_or_else(BTreeMap::new, |v| r.window.iter().zip(v).filter(|(_, x)| x.is_finite()).map(|(c, x)| (*c, *x)).collect());
                let pieces: Vec<crate::tentacle_core::Piece> = r
                    .window
                    .iter()
                    .map(|c| crate::tentacle_core::Piece {
                        chunk: *c,
                        tokens: tk(*c),
                        // Keep every piece of the window on one comparable scale (`one`, which
                        // already folds in the calibrated integration read), never mixing raw
                        // read logits with multi-kind fused sums.
                        log_odds: one.get(*c).copied().filter(|x| x.is_finite()).unwrap_or(0.0),
                        companion: graph.specific.edges_of(*c).iter().max_by(|a, b| a.weight.total_cmp(&b.weight)).map(|e| e.to as usize),
                        ..crate::tentacle_core::Piece::default()
                    })
                    .collect();
                (read, crate::tentacle_core::RoundInput { pieces, grant: None, band: None, pose: question.to_owned() })
            };
            if let Some(m) = reader.as_ref() {
                let tp = Instant::now();
                let max_groups: usize = std::env::var("NANO_DEEP_MAX_GROUPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
                // Rank contested tentacles by tightest margin (most uncertain leader first) so
                // `deep_check_packed` spends its single card pass on the highest-value tie and
                // deduplicates identical chunk pairs across tentacles (cutting 0.8-1.2 s multi-span
                // deep passes down to one ~0.40 s pass).
                let mut cand_t: Vec<(f32, usize, Vec<usize>)> = Vec::new();
                for (r, rs) in reads.iter().zip(&read_sets) {
                    if !contested(rs.as_deref(), &r.label) {
                        continue;
                    }
                    // `NANO_DEEP_PROBE=off`: the earlier prediction, the top two by the read.
                    let ask = if std::env::var("NANO_DEEP_PROBE").is_ok_and(|v| v == "off") {
                        let one = one_rc.borrow();
                        let lo = |c: usize, k: usize| rs.as_ref().and_then(|v| v.get(k).copied()).filter(|x| x.is_finite()).or_else(|| one.get(c).copied().filter(|x| x.is_finite())).unwrap_or(f32::MIN);
                        let mut by: Vec<(f32, usize)> = r.window.iter().enumerate().map(|(k, c)| (lo(*c, k), *c)).collect();
                        by.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                        by.iter().take(2).map(|x| x.1).collect::<Vec<usize>>()
                    } else {
                        let (read, input) = input_of(r, rs.as_deref());
                        let mut probe = Probe { read, asked: None };
                        let mut dry = cores.borrow().get(&r.tentacle).cloned().unwrap_or_else(|| c0.clone());
                        let _ = dry.round(input, &mut probe);
                        probe.asked.unwrap_or_default()
                    };
                    if ask.is_empty() {
                        continue;
                    }
                    let margin = rs.as_deref().map_or(0.0_f32, |v| {
                        let mut f: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
                        f.sort_by(|a, b| b.total_cmp(a));
                        if f.len() >= 2 { f[0] - f[1] } else { f32::INFINITY }
                    });
                    cand_t.push((margin, r.tentacle, ask));
                }
                cand_t.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                let mut uniq_asks: Vec<Vec<usize>> = Vec::new();
                let mut reqs: Vec<Vec<crate::joint_read::Piece>> = Vec::new();
                let mut who_idx: Vec<(usize, usize)> = Vec::new();
                for (_, tentacle, ask) in cand_t {
                    if let Some(pos) = uniq_asks.iter().position(|u| u == &ask) {
                        who_idx.push((tentacle, pos));
                        continue;
                    }
                    if reqs.len() >= max_groups {
                        continue;
                    }
                    let ps = pieces(sub, &ask, &[]);
                    if ps.len() == ask.len() {
                        let pos = reqs.len();
                        uniq_asks.push(ask);
                        reqs.push(ps);
                        who_idx.push((tentacle, pos));
                    }
                }
                if !reqs.is_empty() {
                    let out = crate::tentacle_core::deep_check_packed(m, question, &reqs);
                    for (t, pos) in who_idx {
                        if let (Some(req), Some(v)) = (reqs.get(pos), out.get(pos)) {
                            packed_by_t.insert(t, req.iter().map(|p| p.chunk).zip(v.iter().copied()).collect());
                        }
                    }
                    deep_spent.set(deep_spent.get() + tp.elapsed().as_secs_f64());
                }
            }
            for (r, rs) in reads.iter().zip(&read_sets) {
                let (read, input) = input_of(r, rs.as_deref());
                // The deep check by its design gate (master plan 3.3 step 7): a tentacle whose
                // margin lies in the noise band (its integration read was granted: the fused margin
                // in the band) or whose window an order from G1 or G2 formed; never every tentacle
                // every round (that took 15 s a call on large code).
                let deep_gate = contested(rs.as_deref(), &r.label);
                let deep = reader.as_ref().filter(|_| deep_gate).map(|m| crate::tentacle_core::ReaderIo {
                    model: m,
                    dev: if m.pop.gpu.is_some() { nmengn::placement::Device::Gpu } else { nmengn::placement::Device::Cpu },
                    question,
                    pieces: pieces_by_chunk(sub, &r.window),
                    heads: heads.heads.clone(),
                    last: heads.last_layer(),
                    integrate: None,
                    faults: Vec::new(),
                });
                let mut io = FlowIo { read, deep, spent: &deep_spent, packed: packed_by_t.remove(&r.tentacle).unwrap_or_default(), fallbacks: &deep_fallbacks };
                let mut cs = cores.borrow_mut();
                let core = cs.entry(r.tentacle).or_insert_with(|| c0.clone());
                let rep = core.round(input, &mut io);
                if !rep.best.is_empty() {
                    depths_now.insert(r.tentacle, rep.best.len());
                }
                let focus = nmbrain::g0::Focus { lens: r.reader.as_ref().map_or("drones", |l| l.name).into(), pose: "plain".into(), ..nmbrain::g0::Focus::default() };
                let id = nmbrain::g0::g0_id(u8::try_from(r.tentacle + 1).unwrap_or(u8::MAX));
                core_reports.borrow_mut().insert(r.tentacle, rep.to_g0(&id, u8::try_from(r.tentacle + 1).unwrap_or(u8::MAX), &focus, &span_of));
                let _ = crate::g3::note_site(nmmusc::sites::SiteKind::Integration, &format!("t{}", r.tentacle + 1), "core", if rep.deep.is_empty() { "deep:0" } else { "deep:1" }, None);
            }
        }
        for (r, rs) in reads.iter().zip(&read_sets) {
            let (mut g0, mut depth) = make(r.tentacle, &r.window, r.reader.as_ref(), rs.as_deref());
            if let Some(d) = core_depth.borrow().get(&r.tentacle) {
                depth = *d;
            }
            if let Some(rep) = core_reports.borrow_mut().remove(&r.tentacle) {
                g0 = g0.with_core(rep);
            }

            solos.push(g0.clone());
            boxes.push(Box::new(g0));
            depths.push(depth);
        }
        let windows: Vec<Vec<usize>> = reads.iter().map(|r| r.window.clone()).collect();
        let all: Vec<usize> = windows.iter().flatten().map(|c| tk(*c)).collect();
        let mean = all.iter().sum::<usize>().checked_div(all.len()).unwrap_or(1).max(1);
        let room = room_of(mean);
        let budget = Budget::derive(windows.len(), Some(room), Some(mean), None).with_room(Some(room));
        let (gains, relay) = if relay_g2 { g2_relay(&windows) } else { (vec![1.0; windows.len()], Vec::new()) };
        g1.structure = structure();
        g1.g2_gains = gains;
        g1.relay = relay;
        g1.reflex_skip = has(nmbrain::reflex::Code::G1Skip);
        g1.reflex_low = has(nmbrain::reflex::Code::G1RereadLow);
        let outcome = g1.run(question, &mut boxes, &budget);
        // Solo equivalence: each window read with the other tentacles absent gives the same first report.
        let solo = solos.iter_mut().zip(&outcome.first).map(|(g, r)| format!("{:?}", g.report()) != format!("{r:?}")).filter(|x| *x).count();
        // Nothing discarded: every held piece is forwarded or queued, exactly once.
        let held: std::collections::BTreeSet<String> = boxes.iter().flat_map(|b| b.held()).map(|s| s.key).collect();
        let fwd: Vec<&str> = outcome.evidence.forwarded.iter().map(|e| e.span.key.as_str()).collect();
        let que: Vec<&str> = outcome.evidence.queue.iter().map(|e| e.span.key.as_str()).collect();
        let mut seen = std::collections::BTreeSet::new();
        let twice = fwd.iter().chain(&que).filter(|k| !seen.insert(**k)).count();
        let missing = held.iter().filter(|k| !seen.contains(k.as_str())).count();
        let extra = seen.iter().filter(|k| !held.contains(**k)).count();
        let orders: Vec<Vec<usize>> = outcome
            .last
            .iter()
            .map(|r| {
                let mut v: Vec<(usize, usize)> = r.facts.iter().filter_map(|(k, f)| Some((f.window_rank, chunk_of(k)?))).collect();
                v.sort_unstable();
                v.into_iter().map(|(_, c)| c).collect()
            })
            .collect();
        let finds: Vec<Vec<usize>> = outcome
            .first
            .iter()
            .zip(&outcome.last)
            .map(|(a, b)| {
                let mut v: Vec<usize> = a.best.iter().chain(&b.best).filter_map(|s| chunk_of(&s.span.key)).collect();
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect();
        ladder_record(reads, &read_sets, &orders, &finds);
        Round { outcome, depths, orders, finds, unconserved: twice + missing + extra, solo: solo + read_solo, segments: boxes }
    };

    // The windows of every arm, and G1 over them.
    let mut ranked: Vec<usize> = pinned.iter().copied().filter(|c| *c < fused.len()).collect();
    let mut rest: Vec<usize> = (0..fused.len()).filter(|c| !ranked.contains(c)).collect();
    rest.sort_by(|a, b| fused.get(*b).copied().unwrap_or(0.0).total_cmp(&fused.get(*a).copied().unwrap_or(0.0)).then(a.cmp(b)));
    ranked.extend(rest);
    let mut g1 = G1::new(codebase(sub));
    g1.ledger = call_ledger.clone();
    if pooled {
        g1.combine = nmbrain::g1::Combine::Pooled;
        if let Some(cs) = &shared {
            g1.common = Some(cs.pool.clone());
            // Phase G1-a (lane M's race, 2026-11-19; `NANO_G1_ACCUM=off` restores the mean race):
            // fused evidence accumulated across rounds, runners not re-read fading by half a round
            // (honeybee dance expiry, stated), the quorum counted by measured independence.
            g1.race = if std::env::var("NANO_G1_ACCUM").is_ok_and(|v| v == "off") {
                nmbrain::g1::Race::new(calibrated.as_ref().map_or(0, |ev| ev.aspects.len()), cs.band, cs.pool.rho)
            } else {
                nmbrain::g1::Race::accumulating(calibrated.as_ref().map_or(0, |ev| ev.aspects.len()), cs.band, cs.pool.rho, G1_FADE)
            };
            g1.racing = RACE.read().is_ok_and(|g| *g);
        }
    }
    let mut rounds: Vec<(Vec<Read>, Round)> = Vec::new();
    let mut yields: Vec<Vec<Option<f32>>> = Vec::new();
    let mut compose_s = 0.0_f64;
    let mut g1_s = 0.0_f64;
    if arm == Windows::Team {
        // The perspectives on offer: every lens (one drone signal each, the relations of the
        // strongest finds, memory) and the whole evidence.
        let mut readers: Vec<Lens> = lenses.clone();
        if !readers.iter().any(|l| l.statistic == "fused") {
            readers.push(Lens { name: "evidence", statistic: "fused", score: fused.clone() });
        }
        let evidence = readers.iter().position(|l| l.statistic == "fused").unwrap_or(0);
        let perspectives: Vec<crate::coordinator::Perspective> = readers
            .iter()
            .map(|l| crate::coordinator::Perspective { name: if l.statistic == "fused" { "evidence".into() } else { l.name.into() }, statistic: l.statistic.into(), score: l.score.clone() })
            .collect();
        let mut team = crate::coordinator::Team::new(fused.clone(), &tk, &graph.specific, &perspectives, budget_w, evidence);
        if let Some(m) = &mm {
            team.meaning.clone_from(&m.clusters);
        }
        // Round 1 covers the request's aspects: each aspect's evidence as the chance each piece
        // answers it (its share of the aspect's total; roadmap Phase 0 item 1, 2026-11-23). The
        // earlier share of the strongest (`NANO_ASPECT_ROWS=max`, diagnosis only) made the strongest
        // piece certain, so the first window found every aspect and the other seven windows of
        // round 1 were worth 0 and taken in list order (every traced question). With the
        // perspectives' diminishing returns, the whole evidence counted once and reading last
        // (`Team::assign`) and the tree prior: dev-large-2 code first 60 to 61 or 62, pack 87 to 90,
        // every 78 to 80, decoys 20 of 20; large code first 26 to 27 to 31, pack 47 to 45 (trc05,
        // trc06: their answers are read in round 1 but lost in G1's joining of the lens reports;
        // IN_TROUBLE, open), decoy dec16 at the confidence cut (0.69 against 0.70, same pieces).
        let max_rows = std::env::var("NANO_ASPECT_ROWS").is_ok_and(|v| v == "max");
        if let Some(ev) = &calibrated {
            team.aspects = (0..ev.aspects.len())
                .map(|a| {
                    let row: Vec<f32> = (0..fused.len()).map(|i| ev.aspect(a, i).max(0.0)).collect();
                    if max_rows {
                        shares(&row)
                    } else {
                        let total: f32 = row.iter().filter(|x| x.is_finite()).sum();
                        row.iter().map(|x| if total > 0.0 && x.is_finite() { x / total } else { 0.0 }).collect()
                    }
                })
                .filter(|r| r.iter().any(|x| *x > 0.0))
                .collect();
            team.missing = vec![1.0; team.aspects.len()];
            team.cover = true;
        }
        g1.map = nmbrain::allocation::AllocationMap::new(fused.len(), n_t);
        if let Some(m) = &mm {
            g1.map.set_meaning(m.clusters.clone(), m.twins.clone());
        }
        let tentacles: Vec<usize> = (0..n_t).collect();
        // Phase F (lane T1, 2026-11-19; each on by default with its off switch): no two tentacles
        // through the same perspective and request aspect in a round (`NANO_TEAM_DISTINCT=off`);
        // round 1 per request aspect and lead kind (`Aim::Aspect`) with Burgard's discount over
        // relation distance between windows (`NANO_BURGARD=off`); from round 2 relation mode on
        // each tentacle's finds, the frontier by personalised PageRank from the round's finds and
        // recall seeding walks (`NANO_FRONTIER=off`); on large material, scanning mode: regions
        // proposed by the embedder index and the wording, taken along relations (`NANO_SCAN=off`).
        let off = |k: &str| std::env::var(k).is_ok_and(|v| v == "off");
        team.distinct = !off("NANO_TEAM_DISTINCT");
        let burgard = !off("NANO_BURGARD");
        let frontier_on = !off("NANO_FRONTIER");
        let n_aspects = team.aspects.len();
        let cover_aims = |n_p: usize| -> Vec<crate::coordinator::Aim> {
            if n_aspects > 0 && !off("NANO_TEAM_DISTINCT") {
                (0..n_p).flat_map(|p| (0..n_aspects).map(move |a| crate::coordinator::Aim::Aspect { p, a })).collect()
            } else {
                (0..n_p).map(crate::coordinator::Aim::Cover).collect()
            }
        };
        // Round 1: coverage, every tentacle by the perspective and aspect that adds most given the others.
        let mut aims: Vec<crate::coordinator::Aim> = cover_aims(perspectives.len());
        let material_tokens: usize = (0..fused.len()).map(&tk).sum();
        if !off("NANO_SCAN") && material_tokens > 4 * n_t * budget_w {
            let top_by = |v: &dyn Fn(usize) -> f32| -> Vec<(usize, f32)> {
                let mut o: Vec<usize> = (0..fused.len()).filter(|c| v(*c).is_finite() && v(*c) > 0.0).collect();
                o.sort_by(|a, b| v(*b).total_cmp(&v(*a)).then(a.cmp(b)));
                o.into_iter().take(SCAN_SEEDS).map(|c| (c, fused.get(c).copied().unwrap_or(0.0).max(f32::MIN_POSITIVE))).collect()
            };
            if let Some(m) = meaning.as_ref().and_then(|m| m.first()) {
                let seeds = top_by(&|c| m.get(c).copied().unwrap_or(f32::NAN));
                if !seeds.is_empty() {
                    aims.push(crate::coordinator::Aim::Follow { seeds, owner: None });
                }
            }
            let seeds = top_by(&|c| words.get(c).copied().unwrap_or(0.0));
            if !seeds.is_empty() {
                aims.push(crate::coordinator::Aim::Follow { seeds, owner: None });
            }
        }
        // Roadmap Phase 0 item 1 (2026-11-23): the tree prior in round 1. The provisional finds are
        // the pieces the fused evidence ranks first; the files beside them in the structure tree
        // give one owner-less follow aim (each file's piece the question's evidence ranks highest),
        // and those pieces' belief rises by the tree's weight (noisy-or, as a relation's), so the
        // aim competes on worth with the others. Round 2 adds the same from the round's real finds.
        let mut tree_first: Option<usize>;
        {
            let mut prov: Vec<usize> = (0..fused.len()).filter(|c| fused.get(*c).is_some_and(|x| x.is_finite() && *x > 0.0)).collect();
            prov.sort_by(|a, b| fused.get(*b).copied().unwrap_or(0.0).total_cmp(&fused.get(*a).copied().unwrap_or(0.0)).then(a.cmp(b)));
            prov.truncate(SCAN_SEEDS);
            let ts = {
                let _tree = crate::live::stage("G1.tree", "");
                crate::coordinator::tree_seeds(sub, &prov, SCAN_SEEDS, &|_| true, &|c| fused.get(c).copied().unwrap_or(f32::MIN))
            };
            raise_by_tree(&mut team.belief, &prov, &ts);
            tree_first = ts.first().map(|s| s.0);
            // Into the frontier (the roadmap's "Round 1 / G1 frontier"): the tree's pieces join the
            // seeds of round 1's own frontier walk (the scan's owner-less follow aim), so they compete
            // inside that window and never take a tentacle from a lens. A separate aim (inserted
            // first on questions of two or more aspects) displaced the lens window holding the answer
            // on large code exp22, trc05 and trc06. Without a scan aim, the tree aim competes alone.
            join_frontier(&mut aims, ts);
        }
        let mut allowed: Vec<Vec<bool>> = vec![vec![true; aims.len()]; n_t];
        let mut floors: Vec<f32> = vec![f32::NEG_INFINITY; n_t];
        let mut all_yields: Vec<f32> = Vec::new();
        // `tree_first`: the first seed of the round's tree aim (`NANO_TREE_TRACE=<file>`, diagnosis).
        loop {
            let tc = Instant::now();
            let ok = |t: usize, k: usize| allowed.get(t).and_then(|a| a.get(k)).copied().unwrap_or(false);
            let fl = |t: usize| floors.get(t).copied().unwrap_or(f32::INFINITY);
            team.begin_round(burgard);
            let picks = team.assign(&mut g1.map, &tentacles, &aims, &ok, &fl);
            if let Some(path) = std::env::var_os("NANO_TREE_TRACE") {
                let is_tree = |a: &crate::coordinator::Aim| matches!(a, crate::coordinator::Aim::Follow { owner: None, seeds } if tree_first.is_some_and(|t| seeds.iter().any(|s| s.0 == t)));
                let offered = aims.iter().any(is_tree);
                let taken: Vec<String> = picks.iter().filter(|a| is_tree(&a.aim)).map(|a| format!("t{} worth {:.3} window {}", a.tentacle + 1, a.worth, a.window.len())).collect();
                let file_of = |c: &usize| sub.chunks.get(*c).and_then(|ch| sub.files.get(ch.file)).map_or("?".to_owned(), |f| f.path.replace('\\', "/"));
                let worths: Vec<String> = picks
                    .iter()
                    .map(|a| {
                        let mut fs: Vec<String> = a.window.iter().map(file_of).collect();
                        fs.dedup();
                        format!("{:.3}:{} {{{}}}", a.worth, a.label, fs.join(","))
                    })
                    .collect();
                let line = format!("tree round {} aims {} picks {} worths [{}] tree offered {offered} taken [{}] q {}\n", rounds.len() + 1, aims.len(), picks.len(), worths.join(" "), taken.join("; "), question.replace('\n', " "));
                let _ = std::fs::OpenOptions::new().create(true).append(true).open(path).and_then(|mut h| std::io::Write::write_all(&mut h, line.as_bytes()));
            }
            team.cover = false;
            compose_s += tc.elapsed().as_secs_f64();
            if picks.is_empty() {
                break;
            }
            // Phase J1 (lane M's sites): each tentacle's perspective and aspect, its pose, the
            // round's mode and moving on, as behavioural decision sites for G3 (labels only).
            for a in &picks {
                let unit = format!("t{}", a.tentacle + 1);
                let pname = perspectives.get(a.perspective).map_or("evidence", |p| p.name.as_str());
                let (aspect, mode) = match &a.aim {
                    crate::coordinator::Aim::Aspect { a: k, .. } => (format!("aspect{}", k + 1), "coverage"),
                    crate::coordinator::Aim::Cover(_) => ("all".to_owned(), "coverage"),
                    crate::coordinator::Aim::Follow { owner: Some(o), .. } if *o == a.tentacle => ("finds".to_owned(), "relation"),
                    crate::coordinator::Aim::Follow { owner: Some(_), .. } => ("finds".to_owned(), "relation"),
                    crate::coordinator::Aim::Follow { owner: None, .. } => ("frontier".to_owned(), "frontier"),
                };
                let _ = crate::g3::note_site(nmmusc::sites::SiteKind::Perspective, &unit, "-", &format!("{pname}+{aspect}"), None);
                {
                    let mut ls = config_lenses.borrow_mut();
                    if !ls.iter().any(|x| x == pname) {
                        ls.push(pname.to_owned());
                    }
                }
                let _ = crate::g3::note_site(nmmusc::sites::SiteKind::Pose, &unit, "-", "question+aspect", None);
                let _ = crate::g3::note_site(nmmusc::sites::SiteKind::Mode, &unit, &format!("round{}", rounds.len() + 1), mode, None);
                if !rounds.is_empty() {
                    let stays = matches!(&a.aim, crate::coordinator::Aim::Follow { owner: Some(o), .. } if *o == a.tentacle);
                    let _ = crate::g3::note_site(nmmusc::sites::SiteKind::MovingOn, &unit, &format!("round{}", rounds.len() + 1), if stays { "stay" } else { "move" }, None);
                }
            }
            let mut reads: Vec<Read> = picks
                .iter()
                .map(|a| Read { tentacle: a.tentacle, window: a.window.clone(), reader: readers.get(a.perspective).cloned(), label: a.label.clone() })
                .collect();
            if rounds.is_empty() {
                let held: std::collections::BTreeSet<usize> = reads.iter().flat_map(|r| r.window.iter().copied()).collect();
                if let Some(r0) = reads.first_mut() {
                    r0.window.extend(forced(ForceAt::Windows).into_iter().filter(|c| !held.contains(c)));
                }
            }
            let tg = Instant::now();
            let round = g1_round(&mut g1, &reads, rounds.is_empty());
            g1_s += tg.elapsed().as_secs_f64();
            // Yields: the belief each read's report carried.
            let mut ys: Vec<Option<f32>> = vec![None; n_t];
            for (r, f) in reads.iter().zip(&round.finds) {
                let y: f32 = f.iter().map(|c| team.belief.get(*c).copied().unwrap_or(0.0)).sum();
                if let Some(x) = ys.get_mut(r.tentacle) {
                    *x = Some(y);
                }
                all_yields.push(y);
            }
            for e in &round.outcome.evidence.forwarded {
                if let Some(c) = chunk_of(&e.span.key) {
                    g1.map.mark(c, nmbrain::allocation::Status::Forwarded);
                }
            }
            for e in &round.outcome.evidence.queue {
                if let Some(c) = chunk_of(&e.span.key) {
                    g1.map.mark(c, nmbrain::allocation::Status::Queued);
                }
            }
            // The belief after the round: a piece read and not reported loses belief by the
            // chance the read would have reported it; the unread relations of a find gain the
            // find's belief through the relation (noisy-or), so the frontier carries worth.
            let mut b = team.belief.clone();
            for ((r, f), (order, depth)) in reads.iter().zip(&round.finds).zip(round.orders.iter().zip(&round.depths)) {
                let _ = r;
                for (rank, c) in order.iter().enumerate() {
                    if f.contains(c) {
                        continue;
                    }
                    if let Some(x) = b.get_mut(*c) {
                        let miss = 1.0 - crate::coordinator::reported_chance(rank, *depth);
                        let num = *x * miss;
                        *x = if num + (1.0 - *x) > 0.0 { num / (num + (1.0 - *x)) } else { 0.0 };
                    }
                }
            }
            let kw = graph.specific.kind_weights();
            for f in round.finds.iter().flatten() {
                let bf = team.belief.get(*f).copied().unwrap_or(0.0);
                let edges = graph.specific.edges_of(*f);
                let topw = edges.iter().map(|e| e.weight * kw.get(e.kind.index()).copied().unwrap_or(1.0)).fold(0.0_f32, f32::max);
                if topw <= 0.0 {
                    continue;
                }
                for e in edges {
                    let j = e.to as usize;
                    if g1.map.cell(j).is_some_and(|x| x.holder.is_some() || x.visited) {
                        continue;
                    }
                    if let Some(x) = b.get_mut(j) {
                        let p = bf * nmsubs::lens::HOP_DECAY * e.weight * kw.get(e.kind.index()).copied().unwrap_or(1.0) / topw;
                        *x = 1.0 - (1.0 - *x) * (1.0 - p.clamp(0.0, 1.0));
                    }
                }
            }
            team.belief = b;
            // Moving on: a tentacle stays and follows the relations of its own finds while its
            // yield holds against the request's average less the noise band; otherwise it moves
            // to the frontier. A new window must be worth at least that level.
            let average = all_yields.iter().sum::<f32>() / all_yields.len().max(1) as f32;
            let level = average - crate::coordinator::noise_band(&all_yields);
            let one = std::env::var("NANO_TEAM_ROUNDS").is_ok_and(|v| v == "one");
            let mut next_aims: Vec<crate::coordinator::Aim> = cover_aims(perspectives.len());
            for (r, f) in reads.iter().zip(&round.finds) {
                let finds: Vec<(usize, f32)> = f.iter().map(|c| (*c, team.belief.get(*c).copied().unwrap_or(0.0))).filter(|x| x.1 > 0.0).collect();
                if !finds.is_empty() {
                    next_aims.push(crate::coordinator::Aim::Follow { seeds: finds, owner: Some(r.tentacle) });
                }
            }
            if frontier_on {
                // The frontier: unread pieces the round's finds point to, by personalised PageRank
                // from the finds (it proposes where to look, never ranks), walked from the best.
                let seeds: Vec<(usize, f32)> =
                    round.finds.iter().flatten().map(|c| (*c, team.belief.get(*c).copied().unwrap_or(0.0))).filter(|x| x.1 > 0.0).collect();
                let ppr = crate::coordinator::personalised_pagerank(&graph.specific, fused.len(), &seeds, PPR_STEPS);
                let mut fr: Vec<usize> = (0..fused.len()).filter(|c| ppr.get(*c).is_some_and(|x| *x > 0.0) && g1.map.cell(*c).is_some_and(|x| x.holder.is_none() && !x.visited)).collect();
                fr.sort_by(|a, b| ppr.get(*b).copied().unwrap_or(0.0).total_cmp(&ppr.get(*a).copied().unwrap_or(0.0)).then(a.cmp(b)));
                let fr: Vec<(usize, f32)> = fr.into_iter().take(SCAN_SEEDS).map(|c| (c, team.belief.get(c).copied().unwrap_or(0.0).max(f32::MIN_POSITIVE))).collect();
                if !fr.is_empty() {
                    next_aims.push(crate::coordinator::Aim::Follow { seeds: fr, owner: None });
                }
                // The structure tree's neighbour files of this round's finds join the frontier's
                // seeds (each file's piece the question's evidence ranks highest, belief raised by
                // the tree's weight), so a cross-file question reaches the files beside its finds.
                let finds: Vec<usize> = round.finds.iter().flatten().copied().collect();
                let ts = {
                    let _tree = crate::live::stage("G1.tree", "");
                    crate::coordinator::tree_seeds(sub, &finds, SCAN_SEEDS, &|c| g1.map.cell(c).is_some_and(|x| x.holder.is_none() && !x.visited), &|c| fused.get(c).copied().unwrap_or(f32::MIN))
                };
                raise_by_tree(&mut team.belief, &finds, &ts);
                tree_first = ts.first().map(|s| s.0);
                join_frontier(&mut next_aims, ts);
                // Recall seeds relation walks from round 2.
                if rounds.is_empty() && !pinned.is_empty() {
                    let rs: Vec<(usize, f32)> = pinned.iter().map(|c| (*c, team.belief.get(*c).copied().unwrap_or(0.0).max(f32::MIN_POSITIVE))).collect();
                    next_aims.push(crate::coordinator::Aim::Follow { seeds: rs, owner: None });
                }
            }
            // Marginal value: any tentacle takes the window that adds most, its own finds' walk
            // first among equals (it stays and elaborates), else a frontier or coverage target (it
            // moves on); a new window must be worth the request's level, the average yield so far
            // less its noise band.
            // The team's mode is a learnable site: G3's promoted mode for the next round when it has
            // one (only that mode's aims are offered), else the rule (every aim, relation mode leading).
            let rule_mode = if next_aims.iter().any(|a| matches!(a, crate::coordinator::Aim::Follow { owner: Some(_), .. })) { "relation" } else { "coverage" };
            let mode = crate::g3::decide_site(nmmusc::priors::Site::TeamMode, rule_mode, false);
            let _ = crate::g3::note_site(nmmusc::sites::SiteKind::Mode, "g1", &format!("round{}", rounds.len() + 2), &mode, None);
            if mode != rule_mode {
                let keep_aim = |a: &crate::coordinator::Aim| match mode.as_str() {
                    "coverage" => matches!(a, crate::coordinator::Aim::Cover(_) | crate::coordinator::Aim::Aspect { .. }),
                    "relation" => matches!(a, crate::coordinator::Aim::Follow { owner: Some(_), .. }),
                    _ => matches!(a, crate::coordinator::Aim::Follow { owner: None, .. }),
                };
                let kept: Vec<crate::coordinator::Aim> = next_aims.iter().filter(|a| keep_aim(a)).cloned().collect();
                if !kept.is_empty() {
                    next_aims = kept;
                }
            }
            // Stage 5 WP-C2: the call's refinement allocation (G3's allocation site): `uniform` keeps
            // coverage aims only for the next round; `nbv` (the rule) leaves the aims as they are.
            if g2.refine.as_deref() == Some("uniform") {
                let kept: Vec<crate::coordinator::Aim> =
                    next_aims.iter().filter(|a| matches!(a, crate::coordinator::Aim::Cover(_) | crate::coordinator::Aim::Aspect { .. })).cloned().collect();
                if !kept.is_empty() {
                    next_aims = kept;
                }
            }
            allowed = vec![vec![true; next_aims.len()]; n_t];

            floors = vec![if one { f32::INFINITY } else { level.max(f32::MIN_POSITIVE) }; n_t];
            aims = next_aims;
            yields.push(ys);
            rounds.push((reads, round));
            g1.map.next_round();
            // The stop (master plan 3.5): no unread region's upper bound on evidence can change the
            // committed set: the best value of any piece no window has read is below the value of
            // the room's last piece among those read (`STOP_ROOM` pieces); then no round can bring
            // a piece that overtakes (`NANO_TEAM_STOP=off` keeps the yield rule alone).
            if w2 && !std::env::var("NANO_TEAM_STOP").is_ok_and(|v| v == "off") {
                let o = one_rc.borrow();
                let v = |c: usize| o.get(c).copied().filter(|x| x.is_finite()).unwrap_or(f32::MIN);
                let mut read_vals: Vec<f32> = (0..fused.len()).filter(|c| g1.map.cell(*c).is_some_and(|x| x.visited || x.holder.is_some())).map(v).collect();
                read_vals.sort_by(|a, b| b.total_cmp(a));
                let bar = read_vals.get(STOP_ROOM.saturating_sub(1)).copied().unwrap_or(f32::MIN);
                let bound = (0..fused.len()).filter(|c| g1.map.cell(*c).is_some_and(|x| !x.visited && x.holder.is_none())).map(v).fold(f32::MIN, f32::max);
                if read_vals.len() >= STOP_ROOM && bound <= bar {
                    break;
                }
            }
            // Stage 5 WP-C2: refinement `off` (G3's allocation site) ends the team after its first round.
            if g2.refine.as_deref() == Some("off") {
                break;
            }
        }
    } else {
        let tc = Instant::now();
        let one = n_t * budget_w;
        let (windows, reader): (Vec<Vec<usize>>, Vec<Option<Lens>>) = match arm {
            Windows::Flags => {
                let mut placed = flags.clone();
                for c in &pinned {
                    if let Some(f) = placed.get_mut(*c) {
                        *f = f.max(top);
                    }
                }
                let file_of = |c: usize| sub.chunks.get(c).map_or(usize::MAX, |x| x.file);
                (compose_with(&placed, &graph.specific, &tk, budget_w, n_t, Some(&file_of)).windows, vec![None; n_t])
            }
            Windows::Lenses => (compose_lenses(&lenses, &fused, &tk, budget_w), lenses.iter().cloned().map(Some).collect()),
            Windows::FirstFit | Windows::Single => {
                let all = total.unwrap_or(one);
                let w = if arm == Windows::Single {
                    first_fit(&ranked, &tk, all, 1)
                } else {
                    first_fit(&ranked, &tk, budget_w, all.div_ceil(budget_w).max(1))
                };
                let k = w.len();
                (w, vec![Some(Lens { name: "evidence", statistic: "fused", score: fused.clone() }); k])
            }
            Windows::Team => (Vec::new(), Vec::new()),
        };
        let label = |l: &Option<Lens>| match (arm, l) {
            (Windows::FirstFit, _) => "first fit".to_owned(),
            (Windows::Single, _) => "single reader".to_owned(),
            (_, Some(l)) => l.name.to_owned(),
            (_, None) => "drones".to_owned(),
        };
        let reads: Vec<Read> = windows
            .into_iter()
            .zip(reader)
            .enumerate()
            .map(|(t, (window, reader))| Read { tentacle: t, label: label(&reader), window, reader })
            .collect();
        compose_s = tc.elapsed().as_secs_f64();
        let tg = Instant::now();
        let round = g1_round(&mut g1, &reads, true);
        g1_s = tg.elapsed().as_secs_f64();
        yields.push(vec![None; reads.len()]);
        rounds.push((reads, round));
    }

    // G1's hand-over and queue: one round as G1 gave them; over several rounds every round's
    // records together, by their combined value.
    let race_hand = |g1: &G1, rounds: &[(Vec<Read>, Round)]| -> (nmbrain::g1::Verdict, Vec<usize>, Vec<usize>) {
        let v = g1.race.verdict();
        let handover: Vec<usize> = v.forward(&g1.race).iter().filter_map(|k| chunk_of(k)).collect();
        let mut rest: Vec<(f32, usize)> = Vec::new();
        for (_, r) in rounds {
            for e in r.outcome.evidence.forwarded.iter().chain(&r.outcome.evidence.queue) {
                if let Some(c) = chunk_of(&e.span.key) {
                    rest.push((e.combined, c));
                }
            }
        }
        rest.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut queue: Vec<usize> = Vec::new();
        for (_, c) in rest {
            if !handover.contains(&c) && !queue.contains(&c) {
                queue.push(c);
            }
        }
        (v, handover, queue)
    };
    let mut verdict = (g1.common.is_some() && g1.racing).then(|| g1.race.verdict());
    let (mut handover, mut queue, stop, gate, g1_rounds) = if verdict.is_some() {
        // On the common scale: G1's race over every round. Up: the committed and both sides of
        // each tie; queued: every other record of every round, by its value.
        let (v, handover, queue) = race_hand(&g1, &rounds);
        // On the common scale: G1's race over every round. Up: the committed and both sides of
        // each tie; queued: every other record of every round, by its value.
        let stop = format!(
            "{} rounds of windows; race: {} committed, {} ties, {} aspects uncovered; {}",
            rounds.len(),
            v.committed.len(),
            v.ties.len(),
            v.uncovered.len(),
            rounds.last().map_or(String::new(), |(_, r)| r.outcome.handover.stop.clone())
        );
        (handover, queue, stop, rounds.iter().map(|(_, r)| r.outcome.handover.gate_direct).sum(), rounds.iter().map(|(_, r)| r.outcome.handover.rounds).sum())
    } else if rounds.len() == 1 {
        let Some((_, r)) = rounds.first() else { return Err(nmtype::Fault::new("nmapi.drone_path", nmtype::FaultKind::Execution, "no round")) };
        let handover: Vec<usize> = r.outcome.handover.items.iter().filter_map(|f| chunk_of(&f.span.key)).collect();
        let mut queue: Vec<usize> = Vec::new();
        for e in r.outcome.evidence.forwarded.iter().chain(&r.outcome.evidence.queue) {
            if let Some(c) = chunk_of(&e.span.key)
                && !handover.contains(&c)
                && !queue.contains(&c)
            {
                queue.push(c);
            }
        }
        (handover, queue, r.outcome.handover.stop.clone(), r.outcome.handover.gate_direct, r.outcome.handover.rounds)
    } else {
        let mut fwd: Vec<(f32, usize)> = Vec::new();
        let mut rest: Vec<(f32, usize)> = Vec::new();
        for (_, r) in &rounds {
            for item in &r.outcome.handover.items {
                if let Some(c) = chunk_of(&item.span.key) {
                    fwd.push((item.combined, c));
                }
            }
            for e in r.outcome.evidence.forwarded.iter().chain(&r.outcome.evidence.queue) {
                if let Some(c) = chunk_of(&e.span.key) {
                    rest.push((e.combined, c));
                }
            }
        }
        let by = |a: &(f32, usize), b: &(f32, usize)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
        fwd.sort_by(by);
        rest.sort_by(by);
        let held: Vec<usize> = rounds.iter().flat_map(|(rs, _)| rs.iter().flat_map(|r| r.window.iter().copied())).collect();
        let room = room_of(held.iter().map(|c| tk(*c)).sum::<usize>().checked_div(held.len()).unwrap_or(1));
        let mut handover: Vec<usize> = Vec::new();
        let mut used = 0;
        for (_, c) in fwd {
            if !handover.contains(&c) && used + tk(c) <= room {
                used += tk(c);
                handover.push(c);
            }
        }
        let mut queue: Vec<usize> = Vec::new();
        for (_, c) in rest {
            if !handover.contains(&c) && !queue.contains(&c) {
                queue.push(c);
            }
        }
        let stop = format!("{} rounds of windows; {}", rounds.len(), rounds.last().map_or(String::new(), |(_, r)| r.outcome.handover.stop.clone()));
        (handover, queue, stop, rounds.iter().map(|(_, r)| r.outcome.handover.gate_direct).sum(), rounds.iter().map(|(_, r)| r.outcome.handover.rounds).sum())
    };
    for c in forced(ForceAt::Handover) {
        queue.retain(|x| *x != c);
        if !handover.contains(&c) {
            handover.push(c);
        }
    }
    let lane = if stack_lane_on() { lane_shortlist(sub, question) } else { Vec::new() };
    if let Some(v) = &verdict {
        lad("g1.committed", &v.committed.iter().filter_map(|k| chunk_of(k)).collect::<Vec<_>>());
        lad("g1.ties", &v.ties.iter().flat_map(|(_, a, b)| [chunk_of(a), chunk_of(b)]).flatten().collect::<Vec<_>>());
    }
    lad("g1.handover", &handover);
    lad("g1.queue", &queue);
    lad("lane", &lane);
    if w2 {
        one_forward(&one_rc.borrow(), &mut handover, &mut queue);
    } else {
        settle_forward(calibrated.as_ref(), &mut handover, &mut queue);
    }
    // C132, G1's union: every tentacle's own picks go up with the shared value's hand-over (none
    // is left in the queue), in the value's order; G2 and the judge order the union.
    let own_union = |handover: &mut Vec<usize>, queue: &mut Vec<usize>| {
        if !own_on {
            return;
        }
        let own = own_all.borrow();
        let moved: Vec<usize> = queue.iter().copied().filter(|c| own.contains(c)).collect();
        if moved.is_empty() {
            return;
        }
        queue.retain(|c| !own.contains(c));
        // After the shared hand-over, in the value's order among themselves (the section test on the
        // dump: re-sorting the whole union moved an own pick of higher value but wrong ahead of G1's
        // first, forwarded first 12 to 9 of 30); G2 and the judge order the union.
        let mut moved = moved;
        if w2 {
            let o = one_rc.borrow();
            let v = |c: usize| o.get(c).copied().unwrap_or(f32::MIN);
            moved.sort_by(|a, b| v(*b).total_cmp(&v(*a)).then(a.cmp(b)));
        }
        handover.extend(moved);
    };

    own_union(&mut handover, &mut queue);
    lad("g1.handover.settle", &handover);
    if !w2 {
        join_lane(&mut handover, &mut queue, &lane);
    }
    lad("g1.handover.lane", &handover);

    // G2 (on the common scale, after G1's race): the set that maximises the request's coverage F
    // within the caller's budget (`nmbrain::g2::select`); orders down the motor nervous system,
    // by value of information, for what the set leaves open; the tentacles read what they have
    // never seen and answer for the chosen pieces; G2 selects again and its gate decides.
    // G2's motor path is the call's, on G1's ledger (lane M's G1-b).
    live_stage.count("handover", handover.len());
    live_stage.count("queue", queue.len());
    live_stage.next("G2", "");
    let mut g2_run = G2Run { motor: down.borrow().clone(), ..G2Run::default() };

    let mut g2_up: Option<Vec<usize>> = None;
    let mut g2_cands: Vec<usize> = Vec::new();
    if let (Some(ev), Some(cs), true) = (&calibrated, &shared, verdict.is_some() && G2COVER.read().is_ok_and(|g| *g)) {
        use nmbrain::g2::{Choice, Gate as Comprehended, select};
        use nmpod::intake::Asks;
        use nmsubs::refgraph::{EdgeKind, Relation as Dir};
        use nmtype::class::DataClass::{Behavioural, Sensory};
        use nmtype::message::{Arg, MessageKind, Motor, Verb};
        let t2 = Instant::now();
        // The word aspects G2 covers are the parts of the request's aspects.
        let n_a = ev.parts.len();
        let chance = Chance::new(ev);
        // The drones' chance; the judge never adds to it (it decides swaps at the edge, below).
        let r = |k: usize, c: usize| -> f32 { chance.r(ev, k, c) };
        let asks: Vec<Asks> = nmpod::intake::Request::read(question, &[]).relations;
        // A relation settled as a fact of the code's structure (`Evidence::facts`: every piece
        // whose own lines hold it) is covered piece by piece: one column per fact piece, each
        // worth its share of the relation (it is answered whole when every fact piece is in the
        // set). The other relations are settled through the set (`nmbrain::g2::select`).
        let facts_on = FACTS.read().is_ok_and(|g| *g);
        let fact_of = |k: usize| ev.facts.iter().filter(|_| facts_on).find(|f| f.0 == k).map(|f| f.1.as_slice());
        #[expect(clippy::cast_precision_loss, reason = "a count of pieces")]
        let fact_cols: Vec<(usize, usize, f32)> =
            (0..asks.len()).filter_map(|k| fact_of(k).map(|v| (k, v))).flat_map(|(k, v)| v.iter().map(move |p| (k, *p, 1.0 / v.len().max(1) as f32))).collect();
        let n_w = n_a + fact_cols.len();
        let kept: Vec<usize> = (0..asks.len()).filter(|k| fact_of(*k).is_none()).collect();
        let rel_col = |k: usize| kept.iter().position(|x| *x == k).map(|i| n_w + i);
        let rw = |c: usize| -> Vec<f32> { (0..n_a).map(|a| r(a, c)).chain(fact_cols.iter().map(|(_, p, w)| if *p == c { *w } else { 0.0 })).collect() };
        let value = |c: usize| cs.values.get(c).copied().unwrap_or(0.0);
        let held_all: Vec<usize> = rounds.iter().flat_map(|(rs, _)| rs.iter().flat_map(|r| r.window.iter().copied())).collect();
        let mean = held_all.iter().map(|c| tk(*c)).sum::<usize>().checked_div(held_all.len()).unwrap_or(1).max(1);
        // The pack budget: the caller's, else the stated fallback.
        let room = room_of(mean);
        // The relations each piece `p` at the far end of a typed relation from `q` settles with
        // `q` in the set, of the kinds asked, confirmed on the two pieces' own records: one pass
        // over `q`'s relations.
        let rel_from = |q: usize| -> Vec<(usize, usize)> {
            let mut out: std::collections::BTreeSet<(usize, usize)> = std::collections::BTreeSet::new();
            let Some(rq) = keep.record(q) else { return Vec::new() };
            for e in graph.specific.edges_of(q) {
                let p = e.to as usize;
                let Some(rp) = keep.record(p) else { continue };
                for (k, a) in asks.iter().enumerate() {
                    let ok = match a {
                        Asks::Callees => e.kind == EdgeKind::Call && e.dir == Dir::Uses && rp.defines_used_by(rq),
                        Asks::Callers => e.kind == EdgeKind::Call && e.dir == Dir::UsedBy && rq.defines_used_by(rp),
                        Asks::Reads => matches!(e.kind, EdgeKind::DefUse | EdgeKind::TypeUse | EdgeKind::DefinesUsed) && e.dir == Dir::Uses && rp.defines_used_by(rq),
                    };
                    if ok {
                        out.insert((p, k));
                    }
                }
            }
            out.into_iter().collect()
        };
        let choice = |c: usize| Choice { chunk: c, tokens: tk(c), r: rw(c) };
        // G2's selection over the candidates: what G1 handed over and queued, and (after orders)
        // what the tentacles read on them. The relations between candidates are taken once.
        let run_select = |cands: &[usize]| {
            let index: BTreeMap<usize, usize> = cands.iter().enumerate().map(|(i, c)| (*c, i)).collect();
            let mut pairs: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
            let mut linked: Vec<Vec<usize>> = vec![Vec::new(); cands.len()];
            if !kept.is_empty() {
                for (qi, q) in cands.iter().enumerate() {
                    for (p, k) in rel_from(*q) {
                        let (Some(&pi), Some(kk)) = (index.get(&p), kept.iter().position(|x| *x == k)) else { continue };
                        pairs.entry((pi, qi)).or_default().push(kk);
                        for (a, b) in [(qi, pi), (pi, qi)] {
                            if let Some(l) = linked.get_mut(a)
                                && !l.contains(&b)
                            {
                                l.push(b);
                            }
                        }
                    }
                }
            }
            let choices: Vec<Choice> = cands.iter().map(|c| choice(*c)).collect();
            let s = select(&choices, kept.len(), &|p, q| pairs.get(&(p, q)).cloned().unwrap_or_default(), &|q| linked.get(q).cloned().unwrap_or_default(), room);
            let set: Vec<usize> = s.set.iter().filter_map(|i| cands.get(*i).copied()).collect();
            (s, set)
        };
        let mut cands: Vec<usize> = handover.iter().chain(&queue).copied().collect();
        let (first, first_set) = run_select(&cands);
        lad("g2.first", &first_set);
        g2_run.first_gate = format!("F {:.2} of {}", first.value, n_a + asks.len());
        // Orders, by value of information: for each relation and aspect the set leaves open, the
        // piece no tentacle holds whose expected gain in F is largest (a far end of a typed
        // relation from the set, or the strongest holder of an open aspect); ordered only when
        // that gain stands above the noise band and the piece fits the room left.
        let used: usize = first_set.iter().map(|c| tk(*c)).sum();
        let held_set: std::collections::BTreeSet<usize> = cands.iter().copied().collect();
        let held = |c: usize| held_set.contains(&c);
        let rho = |c: usize| (0..n_a).map(|a| r(a, c)).fold(0.0_f32, f32::max);
        let mut targets: Vec<(usize, f32, String, usize)> = Vec::new();
        for (i, (_, p, w)) in fact_cols.iter().enumerate() {
            let open = 1.0 - first.covered.get(n_a + i).copied().unwrap_or(0.0);
            if !held(*p) && w * open > first.band {
                let near = first_set.iter().copied().find(|s| graph.specific.edges_of(*s).iter().any(|e| e.to as usize == *p)).or_else(|| first_set.first().copied()).unwrap_or(*p);
                targets.push((*p, w * open, format!("its lines hold {}", asks.get(fact_cols.get(i).map_or(0, |f| f.0)).map_or("", |a| a.name())), near));
            }
        }
        for (k, a) in asks.iter().enumerate() {
            let Some(col) = rel_col(k) else { continue };
            let open = 1.0 - first.covered.get(col).copied().unwrap_or(0.0);
            if open <= first.band {
                continue;
            }
            let best = first_set
                .iter()
                .flat_map(|q| rel_from(*q).into_iter().filter(|x| x.1 == k).map(move |(p, _)| (p, *q)))
                .filter(|(p, _)| !held(*p))
                .map(|(p, q)| (p, open * rho(q), q))
                .max_by(|x, y| x.1.total_cmp(&y.1).then(y.0.cmp(&x.0)));
            if let Some((p, g, q)) = best {
                targets.push((p, g, format!("{} of {}", a.name(), file_name(q)), q));
            }
        }
        for a in 0..n_a {
            let open = 1.0 - first.covered.get(a).copied().unwrap_or(0.0);
            if open <= first.band {
                continue;
            }
            let best = (0..ev.len()).filter(|c| !held(*c)).map(|c| (c, open * r(a, c))).max_by(|x, y| x.1.total_cmp(&y.1).then(y.0.cmp(&x.0)));
            if let Some((p, g)) = best {
                let near = first_set.iter().copied().find(|s| graph.specific.edges_of(*s).iter().any(|e| e.to as usize == p)).or_else(|| first_set.first().copied()).unwrap_or(p);
                targets.push((p, g, format!("holds the part {} the set leaves open", ev.parts.get(a).map_or("", |p| p.1.as_str())), near));
            }
        }
        // Orders aimed at a meaning cluster the set lacks: of the clusters of the pieces the
        // question's meaning ranks highest (`CLUSTER_LOOK`), each one no set piece is in gives
        // its best piece no tentacle holds, worth its expected gain over the open aspects.
        if let (Some(m), Some(posed)) = (&mm, meaning.as_ref().and_then(|v| v.first())) {
            let cl = |c: usize| m.clusters.get(c).copied().filter(|k| *k != u32::MAX);
            let in_set: std::collections::BTreeSet<u32> = first_set.iter().filter_map(|c| cl(*c)).collect();
            let cos = |c: usize| posed.get(c).copied().filter(|x| x.is_finite()).unwrap_or(f32::NEG_INFINITY);
            let mut top: Vec<usize> = (0..posed.len()).filter(|c| cos(*c).is_finite()).collect();
            top.sort_by(|a, b| cos(*b).total_cmp(&cos(*a)).then(a.cmp(b)));
            let mut lacking: Vec<u32> = Vec::new();
            for k in top.iter().take(CLUSTER_LOOK).filter_map(|c| cl(*c)) {
                if !in_set.contains(&k) && !lacking.contains(&k) {
                    lacking.push(k);
                }
            }
            for k in lacking {
                let best = top.iter().copied().filter(|c| cl(*c) == Some(k) && !held(*c)).max_by(|a, b| cos(*a).total_cmp(&cos(*b)).then(b.cmp(a)));
                let Some(p) = best else { continue };
                let g = (0..n_a).map(|a| (1.0 - first.covered.get(a).copied().unwrap_or(0.0)) * r(a, p)).fold(0.0_f32, f32::max);
                let near = first_set.first().copied().unwrap_or(p);
                targets.push((p, g, format!("meaning cluster {k} the set lacks"), near));
            }
        }
        targets.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        // W5: the room is reserved before the orders (they may fill any of it) and filled after.
        let mut room_left = if w5 { room } else { room.saturating_sub(used) };
        let mut ordered: Vec<(usize, String, usize)> = Vec::new();
        for (p, g, why, s) in targets {
            // An order aimed at a meaning cluster is a read in a tentacle's window (its budget),
            // not a place in the set: G2's selection after the read decides what fits the room
            // (try 2: under the room rule no order fired, the set filling the room).
            let cluster = why.starts_with("meaning cluster");
            if cluster && g > first.band && tk(p) <= budget_w && !ordered.iter().any(|o| o.0 == p) {
                g2_run.cluster_orders += 1;
                ordered.push((p, why, s));
            } else if g > first.band && tk(p) <= room_left && !ordered.iter().any(|o| o.0 == p) {
                room_left -= tk(p);
                ordered.push((p, why, s));
            }
        }
        let now = nmtype::now_ms();
        let expiry = now.saturating_add(1000 * 60);
        let place = |c: usize, rounds: &[(Vec<Read>, Round)]| -> Option<(usize, usize)> {
            rounds.iter().enumerate().find_map(|(ri, (rs, _))| rs.iter().position(|r| r.window.contains(&c)).map(|k| (ri, k)))
        };
        // The never-seen pieces: one read per ordered tentacle (the one holding the set piece it
        // relates to), each piece claimed in G1's map, as a long order with a goal.
        let mut unseen: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
        // The pieces read on G2's orders and its follow-up, for the swap-in rule below.
        let mut read_on_order: Vec<usize> = Vec::new();
        for (c, why, s) in &ordered {
            let t = place(*s, &rounds).and_then(|(ri, k)| rounds.get(ri).and_then(|(rs, _)| rs.get(k))).map_or(0, |r| r.tentacle);
            unseen.entry(t).or_default().push((*c, *s));
            g2_run.lines.push(format!("RESHAPE_WINDOW t{} reads {} ({why})", t + 1, file_name(*c)));
        }
        if !unseen.is_empty() {
            if g1.map.tentacles() == 0 {
                g1.map = nmbrain::allocation::AllocationMap::new(fused.len(), n_t.max(rounds.iter().map(|(rs, _)| rs.len()).max().unwrap_or(0)));
                for (rs, _) in &rounds {
                    for r in rs {
                        for c in &r.window {
                            let _ = g1.map.claim(r.tentacle.min(g1.map.tentacles().saturating_sub(1)), *c, nmbrain::allocation::Mode::Coverage, None, tk(*c), true);
                        }
                    }
                }
            }
            let mut reads: Vec<Read> = Vec::new();
            for (t, list) in &unseen {
                let window: Vec<usize> = list.iter().filter(|(c, s)| g1.map.claim(*t, *c, nmbrain::allocation::Mode::Relation, Some(*s), tk(*c), true)).map(|(c, _)| *c).collect();
                if window.is_empty() {
                    continue;
                }
                let goal = g2_run.motor.id();
                let keys: Vec<String> = window.iter().filter_map(|c| span_of(*c)).map(|s| s.key).collect();
                // I8 (lane M's `Down::reshape_order`, `NANO_I8=off` restores the ordered pieces
                // alone): the envelope recorded in G1's ledger, then the tentacle's next window full
                // and seeded by the ordered pieces, their relations next, no piece another window holds.
                let i8_on = !std::env::var("NANO_I8").is_ok_and(|v| v == "off");
                let sent_ok = Motor::new(goal, None, MessageKind::Instruction, Verb::Reshape, nmbrain::motor::G2, &nmbrain::g0::g0_id(u8::try_from(t + 1).unwrap_or(u8::MAX)), 0.5, "read pieces never seen", expiry)
                    .and_then(|m| m.arg("pieces", Arg::Pieces(keys)))
                    .ok()
                    .and_then(|m| {
                        let m = m.goal(goal);
                        if i8_on {
                            let around: Vec<usize> = window.iter().flat_map(|c| graph.specific.edges_of(*c).iter().map(|e| e.to as usize).collect::<Vec<_>>()).collect();
                            let map = &g1.map;
                            g2_run.motor.reshape_order(m, &window, &around, budget_w, &tk, &|c| map.holder(c).is_some_and(|h| h != *t) || map.cell(c).is_some_and(|x| x.visited)).ok()
                        } else if g2_run.motor.ledger.record(&m).is_ok() {
                            g2_run.motor.sent.push(m);
                            Some(window.clone())
                        } else {
                            None
                        }
                    });
                if let Some(full) = sent_ok {
                    let mut window = window;
                    for c in full {
                        if !window.contains(&c) && g1.map.claim(*t, c, nmbrain::allocation::Mode::Relation, window.first().copied(), tk(c), true) {
                            window.push(c);
                        }
                    }
                    g2_run.edges.extend([("g2", "g1", Behavioural), ("g1", "g0", Behavioural)]);
                    read_on_order.extend(window.iter().copied());
                    reads.push(Read { tentacle: *t, window, reader: None, label: "read on G2's order".into() });
                }
            }
            if !reads.is_empty() {
                let tg = Instant::now();
                let round = g1_round(&mut g1, &reads, false);
                g1_s += tg.elapsed().as_secs_f64();
                g2_run.read_ms = tg.elapsed().as_secs_f64() * 1000.0;
                g2_run.edges.extend([("g0", "g1", Behavioural), ("g0", "g1", Sensory)]);
                g2_run.read = reads.iter().map(|r| r.window.len()).sum();
                yields.push(vec![None; n_t]);
                rounds.push((reads, round));
            }
            let (v, h, q) = race_hand(&g1, &rounds);
            verdict = Some(v);
            handover = h;
            queue = q;
            if w2 {
                one_forward(&one_rc.borrow(), &mut handover, &mut queue);
            } else {
                settle_forward(Some(ev), &mut handover, &mut queue);
                join_lane(&mut handover, &mut queue, &lane);
            }
            own_union(&mut handover, &mut queue);
            cands = handover.iter().chain(&queue).copied().collect();
        }
        let first_evaluated = first.evaluated;
        let (mut last, mut set) = if g2_run.read > 0 { run_select(&cands) } else { (first, first_set) };
        // W5: the set is the fused value's order filled to the room, then each part no piece of it
        // holds gets its best holder (the weakest pieces not alone on a part leave to fit).
        // The selection's own set: G2's edge margin (a signal of the pack's confidence) reads it.
        let selected = set.clone();
        if w5 {
            // The set's membership by the value before the read (the read orders the pieces it read
            // among themselves, it does not push a piece out of the room: W3's pack and every-range
            // losses, 2026-11-18); its order below by the value with the read.
            let mut fill_by: Vec<f32> = if w3 && read_on { one_base.clone() } else { one_rc.borrow().clone() };
            if let Some(handle) = drone_member_bg.borrow_mut().take()
                && let Ok(dm) = handle.join()
                && dm.len() == fill_by.len()
            {
                for (f, d) in fill_by.iter_mut().zip(dm.iter()) {
                    *f += d;
                }
            }
            set = value_fill(&cands, &fill_by, &tk, room, n_a, &|a, c| ev.part_holds(a, c));
            // Phase G2-u (lane M's belief per item; on by default as a design part, `NANO_ITEM_BELIEF=off`):
            // the candidates' items (a definition with its comment), each valued by its best piece's
            // fused value (so a long item gains nothing by length, C65), length-normalised over the
            // request (`item_beliefs`), the room filled in tokens by item (`fill_items`); the set is
            // then the candidate pieces of the chosen items in item order, while they fit the room,
            // each part still held (the pieces stay the unit served).
            if !std::env::var("NANO_ITEM_BELIEF").is_ok_and(|v| v == "off") {
                let items = keep.items();
                let mut by_item: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
                for c in &cands {
                    if let Some(its) = items.of_piece.get(*c) {
                        for i in its {
                            by_item.entry(*i).or_default().push(*c);
                        }
                    }
                }
                let v = |c: usize| fill_by.get(c).copied().filter(|x| x.is_finite()).unwrap_or(f32::MIN);
                let ins: Vec<nmbrain::g1::ItemIn> = by_item
                    .iter()
                    .map(|(i, ps)| nmbrain::g1::ItemIn {
                        key: i.to_string(),
                        tokens: keep.item_tokens(*i).max(1),
                        value: ps.iter().map(|c| v(*c)).fold(f32::MIN, f32::max),
                        neighbours: Vec::new(),
                    })
                    .filter(|x| x.value > f32::MIN)
                    .collect();
                if !ins.is_empty() {
                    let mut beliefs = nmbrain::g1::item_beliefs(&ins);
                    beliefs.sort_by(|a, b| b.value.total_cmp(&a.value).then(a.key.cmp(&b.key)));
                    // The item belief joins the fused value as one more kind (standardised over the
                    // candidates and clipped, as every kind is): a piece takes its best item's
                    // length-normalised belief. Restricting the set to the chosen items' pieces
                    // (the first wiring) dropped answers the value held (first card smoke: the
                    // answer piece of a long item left the set), so the items weigh, they do not
                    // gate. `fill_items` still gives the items' room order for the record.
                    let (chosen, _, _) = nmbrain::g2::fill_items(&beliefs, room.max(1));
                    let bel: BTreeMap<String, f32> = beliefs.iter().map(|b| (b.key.clone(), b.value)).collect();
                    let mut col = vec![f32::NAN; fill_by.len()];
                    for (i, ps) in &by_item {
                        let Some(b) = bel.get(&i.to_string()).copied() else { continue };
                        for c in ps {
                            if let Some(x) = col.get_mut(*c) {
                                *x = if x.is_finite() { x.max(b) } else { b };
                            }
                        }
                    }
                    let add = fuse_kinds(&[col], &cands);
                    let adj: Vec<f32> = fill_by.iter().zip(&add).map(|(a, b)| a + b).collect();
                    let item_set = value_fill(&cands, &adj, &tk, room, n_a, &|a, c| ev.part_holds(a, c));
                    if !item_set.is_empty() {
                        lad("g2.items", &item_set);
                        g2_run.lines.push(format!("ITEMS: {} items valued, {} fill the room by belief", beliefs.len(), chosen.len()));
                        set = item_set;
                    }
                }
            }
            // Stage 5 WP-C1 (gap 7): G2's relation-aware selection is kept beside the value fill,
            // for its relation partners only. A piece G2 chose joins when it is at the far end of a
            // typed relation the question asks (caller, callee, read) from one of the value fill's
            // first pieces; it takes the place of the weakest value pieces, never of the first
            // ones (so first place and top 3 are not displaced). G2's part-coverage picks do not
            // join (merging the whole selection cost every range 235 to 227 of 315 and one decoy on
            // the eight sets, runs/STAGE-BENCH-1795342598619.txt against 1795344634259).
            // Arms (`NANO_G2_PACK`): `full` (default), G2's whole selection under the value gate with
            // relation partners and the leaders' neighbour lines kept (eight sets, same binary: first
            // 185 to 186, top 3 242, pack 264 to 265, decoys 48 to 49 of 50; dev-large-2 first 60 to
            // 61, pack 85 to 86, every 75 to 77, decoys 20 of 20; runs/STAGE-BENCH-1795355549548.txt
            // against 1795353789737); `partners`, relation partners only; `full-ungated`, the first
            // design as built; `off`, the value fill alone (the paired A/B arm).
            let g2_pack = std::env::var("NANO_G2_PACK").unwrap_or_else(|_| "full".to_owned());
            lad("g2.selected", &selected);
            let lead_of = |set: &[usize]| -> Vec<usize> { set.iter().take(G2_PACK_PROTECT).copied().collect() };
            if g2_pack == "full-ungated" {
                // The first design as built, kept as a measurable arm.
                let merged = nmbrain::g2::merge_relation_set(&set, &selected, &tk, room, G2_PACK_PROTECT);
                lad("g2.merged", &merged);
                set = merged;
            } else if g2_pack == "full" {
                // The full design with its dip's cause fixed: G2's pieces join under the fused value's
                // gate (a piece displaces a value piece only within the noise band of its value) and
                // relation partners of the leading pieces bypass it.
                let lead = lead_of(&set);
                let vals: Vec<f32> = cands.iter().filter_map(|c| fill_by.get(*c).copied()).filter(|x| x.is_finite()).collect();
                let band = {
                    let med = |x: &mut Vec<f32>| {
                        x.sort_by(f32::total_cmp);
                        x.get(x.len() / 2).copied().unwrap_or(0.0)
                    };
                    let m = med(&mut vals.clone());
                    med(&mut vals.iter().map(|x| (x - m).abs()).collect())
                };
                let value = |c: usize| fill_by.get(c).copied().unwrap_or(f32::NAN);
                let partner = |c: usize| !asks.is_empty() && lead.iter().any(|q| rel_from(*q).iter().any(|(p, _)| *p == c) || rel_from(c).iter().any(|(p, _)| p == q));
                // The lines right beside a leading piece in its file stay with it.
                let beside = |c: usize| {
                    let Some(x) = sub.chunks.get(c) else { return false };
                    lead.iter().filter_map(|l| sub.chunks.get(*l)).any(|l| l.file == x.file && (x.end + 1 >= l.start && x.start <= l.end + 1))
                };
                let merged = nmbrain::g2::merge_gated(&set, &selected, &value, &partner, &beside, &tk, room, G2_PACK_PROTECT, band);
                lad("g2.merged", &merged);
                set = merged;
            } else if g2_pack != "off" && !asks.is_empty() {
                let lead: Vec<usize> = set.iter().take(G2_PACK_PROTECT).copied().collect();
                let partners: Vec<usize> = selected
                    .iter()
                    .copied()
                    .filter(|c| !set.contains(c))
                    .filter(|c| lead.iter().any(|q| rel_from(*q).iter().any(|(p, _)| p == c) || rel_from(*c).iter().any(|(p, _)| p == q)))
                    .collect();
                if !partners.is_empty() {
                    let merged = nmbrain::g2::merge_relation_set(&set, &partners, &tk, room, G2_PACK_PROTECT);
                    lad("g2.partners", &partners);
                    g2_run.lines.push(format!("G2 PACK: {} relation partners of the leading pieces join the set", partners.len()));
                    set = merged;
                }
            }

            // C132: every tentacle's own best enters beside the shared value's top; the weakest
            // by the value that are neither own bests nor alone on a part leave to fit the room.
            if own_on {
                let own = own_all.borrow();
                lad("g2.valueset", &set);
                lad("own", &own);
                g2_run.lines.push(format!(
                    "OWN: {} distinct own picks from the tentacles, {} of them outside the shared value's set",
                    own.len(),
                    own.iter().filter(|c| !set.contains(c)).count()
                ));
                // The union is ordered by G2 and the judge (Luis, 2026-11-18 23:35): the tentacles'
                // own picks do not displace the shared value's set by rule (that cost answers on the
                // first card smoke); they join the close-call judge's read beside the value's first
                // pieces (W4 below), and one enters the set when the judge's read with the value
                // makes it the leader.
            }
        }
        lad("g2.orders", &set);
        lad("g2.cands", &cands);
        // G2's margin at its edge (C9): the last set piece's gain per token against the best
        // left out's, over the misses the set leaves.
        let edge_margin = |last: &nmbrain::g2::Selected, set: &[usize], cands: &[usize]| -> f32 {
            #[expect(clippy::cast_precision_loss, reason = "token counts")]
            let per = |g: f32, c: usize| g / tk(c).max(1) as f32;
            let miss: Vec<f32> = (0..n_a).map(|a| 1.0 - last.covered.get(a).copied().unwrap_or(0.0)).collect();
            let last_in = set.last().zip(last.gains.last()).map_or(0.0, |(c, g)| per(*g, *c));
            let best_out = cands.iter().filter(|c| !set.contains(c)).map(|c| per(miss.iter().enumerate().map(|(a, m)| m * r(a, *c)).sum(), *c)).fold(0.0_f32, f32::max);
            if last_in > 0.0 && best_out > 0.0 { (last_in / best_out).ln().clamp(-MARGIN_CAP, MARGIN_CAP) } else if last_in > 0.0 { MARGIN_CAP } else { 0.0 }
        };
        g2_run.margin = edge_margin(&last, &selected, &cands);
        if std::env::var_os("NANO_FOLLOW_DEBUG").is_some() {
            eprintln!("follow: kind {} margin {:.3} set {} cands {}", nmpod::intake::Request::read(question, &[]).kind, g2_run.margin, set.len(), cands.len());
        }
        // At most one follow-up read by G2's motor order (research item 2, RepoCoder's one round,
        // FLARE's trigger and masked query; `NANO_FOLLOW=on`, off by default): on a trace or explain
        // request (the request's drone reads most trace questions as explain)
        // whose margin is in the low band, the query is the names the set's first piece (the
        // anchor) calls and defines, less those holding a part the set already holds with
        // confidence; the pieces outside the candidates that define a called name or call a
        // defined one (at most `FOLLOW_PIECES`, within the room left) are read by the tentacle
        // whose window holds the anchor, in relation mode, and G2 selects again.
        if follow_on()
            && g2_run.margin <= follow_band()
            && matches!(nmpod::intake::Request::read(question, &[]).kind, "trace" | "explain")
            && let Some(&anchor) = set.first()
            && let Some(ach) = sub.chunks.get(anchor)
        {
            let confident: Vec<&str> = (0..n_a)
                .filter(|k| last.covered.get(*k).is_some_and(|c| *c >= 0.5))
                .filter_map(|k| ev.parts.get(k).and_then(|p| ev.stems.get(p.0)))
                .flatten()
                .map(String::as_str)
                .collect();
            let (calls, defs) = anchor_names(&ach.low);
            let masked = |n: &String| !confident.iter().any(|s| !s.is_empty() && n.contains(*s));
            let calls: Vec<String> = calls.into_iter().filter(masked).collect();
            let defs: Vec<String> = defs.into_iter().filter(masked).collect();
            g2_run.follow_query = calls.iter().chain(&defs).cloned().collect();
            // The read is a window of the anchor's tentacle (its budget), not a place in the set:
            // G2's selection after it decides what fits the room.
            let mut left = budget_w;
            let mut found: Vec<usize> = Vec::new();
            let defines = |low: &str, n: &str| ["fn ", "def ", "function ", "class ", "struct ", "enum ", "trait "].iter().any(|k| names_at(low, &format!("{k}{n}"), false));
            let calls_name = |low: &str, n: &str| names_at(low, &format!("{n}("), true) && !defines(low, n);
            // A name counts only when the material defines it in at most two pieces (an
            // unambiguous name; `new`, `get` and the like say nothing).
            let rare = |n: &String| sub.chunks.iter().filter(|c| defines(&c.low, n)).take(3).count() <= 2;
            let calls: Vec<String> = calls.into_iter().filter(rare).collect();
            let defs: Vec<String> = defs.into_iter().filter(rare).collect();
            g2_run.follow_query = calls.iter().chain(&defs).cloned().collect();
            for (i, c) in sub.chunks.iter().enumerate() {
                if found.len() >= FOLLOW_PIECES {
                    break;
                }
                if i == anchor || cands.contains(&i) || tk(i) > left {
                    continue;
                }
                if calls.iter().any(|n| defines(&c.low, n)) || defs.iter().any(|n| calls_name(&c.low, n)) {
                    left -= tk(i);
                    found.push(i);
                }
            }
            let t = place(anchor, &rounds).and_then(|(ri, k)| rounds.get(ri).and_then(|(rs, _)| rs.get(k))).map_or(0, |r| r.tentacle);
            if std::env::var_os("NANO_FOLLOW_DEBUG").is_some() {
                eprintln!("follow: calls {calls:?} defs {defs:?} found {found:?} room left {left}");
            }
            if !found.is_empty() {
                if g1.map.tentacles() == 0 {
                    g1.map = nmbrain::allocation::AllocationMap::new(fused.len(), n_t.max(rounds.iter().map(|(rs, _)| rs.len()).max().unwrap_or(0)));
                    for (rs, _) in &rounds {
                        for r in rs {
                            for c in &r.window {
                                let _ = g1.map.claim(r.tentacle.min(g1.map.tentacles().saturating_sub(1)), *c, nmbrain::allocation::Mode::Coverage, None, tk(*c), true);
                            }
                        }
                    }
                }
                let window: Vec<usize> = found.iter().copied().filter(|c| g1.map.claim(t, *c, nmbrain::allocation::Mode::Relation, Some(anchor), tk(*c), true)).collect();
                let goal = g2_run.motor.id();
                let keys: Vec<String> = window.iter().filter_map(|c| span_of(*c)).map(|s| s.key).collect();
                if !window.is_empty()
                    && let Ok(m) = Motor::new(goal, None, MessageKind::Instruction, Verb::Reshape, nmbrain::motor::G2, &nmbrain::g0::g0_id(u8::try_from(t + 1).unwrap_or(u8::MAX)), 0.5, "follow-up read on a low margin", expiry)
                        .and_then(|m| m.arg("pieces", Arg::Pieces(keys)))
                    && g2_run.motor.ledger.record(&m).is_ok()
                {
                    g2_run.motor.sent.push(m.goal(goal));
                    g2_run.edges.extend([("g2", "g1", Behavioural), ("g1", "g0", Behavioural)]);
                    g2_run.lines.push(format!("FOLLOW_UP t{} reads {} pieces on {}", t + 1, window.len(), g2_run.follow_query.join(" ")));
                    g2_run.follow.clone_from(&window);
                    g2_run.follow_from = Some(anchor);
                    let reads = vec![Read { tentacle: t, window, reader: None, label: "follow-up on G2's order".into() }];
                    let tg = Instant::now();
                    let round = g1_round(&mut g1, &reads, false);
                    g1_s += tg.elapsed().as_secs_f64();
                    g2_run.read_ms += tg.elapsed().as_secs_f64() * 1000.0;
                    g2_run.edges.extend([("g0", "g1", Behavioural), ("g0", "g1", Sensory)]);
                    g2_run.read += reads.iter().map(|r| r.window.len()).sum::<usize>();
                    yields.push(vec![None; n_t]);
                    rounds.push((reads, round));
                    let (v, h, q) = race_hand(&g1, &rounds);
                    verdict = Some(v);
                    handover = h;
                    queue = q;
                    if w2 {
                        one_forward(&one_rc.borrow(), &mut handover, &mut queue);
                    } else {
                        settle_forward(Some(ev), &mut handover, &mut queue);
                        join_lane(&mut handover, &mut queue, &lane);
                    }
                    cands = handover.iter().chain(&queue).copied().collect();
                    (last, set) = run_select(&cands);
                    g2_run.margin = edge_margin(&last, &set, &cands);
                }
            }
        }
        // G2's room rule (C107; `NANO_SWAP_IN`, on by default): a piece read on an order or the
        // follow-up that the selection left out competes with the set's weakest piece by its
        // evidence, as the judge's duels pair them: its gain per token over the misses the set
        // leaves without the holder against the holder's own; it takes the holder's place when
        // the larger and the swap fits the caller's room. Nothing is added past the room; each
        // holder is displaced at most once.
        // The value's first three set pieces are never pushed out by the edge's swaps and duels
        // (large code: exp18's answer, third by the value, left on a swap; val03's, second, on a
        // routed duel); those act at the set's edge.
        let lead3: Vec<usize> = if w2 {
            let o = one_rc.borrow();
            let mut s = set.clone();
            s.sort_by(|a, b| o.get(*b).copied().unwrap_or(f32::MIN).total_cmp(&o.get(*a).copied().unwrap_or(f32::MIN)).then(a.cmp(b)));
            s.truncate(3);
            s
        } else {
            Vec::new()
        };
        if swap_in_on() && !set.is_empty() {
            read_on_order.extend(g2_run.follow.iter().copied());
            // Try 3 (T-064): by marginal gain over the set's misses every entrant read 0, the
            // set already covering the request's parts (the coverage objective is saturated, so
            // nothing outside it can add); the comparison is now each piece's own evidence as
            // G2's routed duels read it: the embedder's calibrated meaning (cosine with the
            // question less its content-free mean), a difference inside the noise band a tie.
            let meant = |c: usize| -> Option<f32> {
                let m = meaning.as_ref()?;
                let posed = m.first()?.get(c).copied().filter(|x| x.is_finite())?;
                let cf: Vec<f32> = m.iter().skip(1).filter_map(|v| v.get(c).copied()).filter(|x| x.is_finite()).collect();
                #[expect(clippy::cast_precision_loss, reason = "a count of forms")]
                let mean = (!cf.is_empty()).then(|| cf.iter().sum::<f32>() / cf.len() as f32)?;
                Some(posed - mean)
            };
            let band = crate::coordinator::noise_band(&cands.iter().filter_map(|c| meant(*c)).collect::<Vec<f32>>());
            #[expect(clippy::cast_precision_loss, reason = "token counts")]
            let per = |g: f32, c: usize| g / tk(c).max(1) as f32;
            let mut used: usize = set.iter().map(|c| tk(*c)).sum();
            let mut moved: Vec<usize> = Vec::new();
            for x in read_on_order.iter().copied().filter(|c| cands.contains(c)) {
                if set.contains(&x) {
                    continue;
                }
                let miss: Vec<f32> = (0..n_a).map(|a| 1.0 - last.covered.get(a).copied().unwrap_or(0.0)).collect();
                let rs = |c: usize| -> Vec<f32> { (0..n_a).map(|a| r(a, c)).collect() };
                let without = |y: usize| -> Vec<f32> { miss.iter().zip(rs(y)).map(|(m, v)| if v < 1.0 { m / (1.0 - v) } else { *m }).collect() };
                // The weakest holder: the set piece adding least per token (the judge's holder).
                let weakest = set
                    .iter()
                    .copied()
                    .filter(|y| !moved.contains(y) && !lead3.contains(y) && used - tk(*y).min(used) + tk(x) <= room)
                    .map(|y| (per(without(y).iter().zip(rs(y)).map(|(a, b)| a * b).sum(), y), y))
                    .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                let Some((_, y)) = weakest else { continue };
                let (Some(vx), Some(vy)) = (meant(x), meant(y)) else { continue };
                if std::env::var_os("NANO_FOLLOW_DEBUG").is_some() {
                    eprintln!("swap: entrant {x} meaning {vx:.4} holder {y} meaning {vy:.4} band {band:.4}");
                }
                if vx - vy > band
                    && let Some(at) = set.iter().position(|c| *c == y)
                {
                    let (rx, ry) = (rs(x), rs(y));
                    for a in 0..n_a {
                        let (Some(cov), Some(gx), Some(gy)) = (last.covered.get_mut(a), rx.get(a), ry.get(a)) else { continue };
                        let m = 1.0 - *cov;
                        let m_out = if *gy < 1.0 { m / (1.0 - gy) } else { m };
                        let new = 1.0 - m_out * (1.0 - gx);
                        last.value += new - *cov;
                        *cov = new;
                    }
                    if let Some(sl) = set.get_mut(at) {
                        *sl = x;
                    }
                    used = used - tk(y).min(used) + tk(x);
                    moved.push(x);
                    g2_run.swapped.push((x, y));
                }
            }
        }
        lad("g2.swap", &set);
        for c in forced(ForceAt::Set) {
            if !set.contains(&c) {
                set.push(c);
            }
            if !cands.contains(&c) {
                cands.push(c);
            }
        }
        // The meaning lane's first fused place is in the set: when the selection left it out, it
        // enters and the set's last choices leave until the set fits the room again; it is placed
        // after the set's own pieces.
        let entered = if w2 {
            let one = one_rc.borrow();
            cands.iter().copied().max_by(|a, b| one.get(*a).copied().unwrap_or(f32::MIN).total_cmp(&one.get(*b).copied().unwrap_or(f32::MIN)).then(b.cmp(a))).filter(|h| !set.contains(h))
        } else if stack_leader_on() {
            lane.first().copied().filter(|h| !set.contains(h))
        } else {
            None
        };
        if let Some(h) = entered {
            set.push(h);
            while set.len() > 1 && set.iter().map(|c| tk(*c)).sum::<usize>() > room {
                let Some(i) = set.iter().rposition(|c| *c != h) else { break };
                set.remove(i);
            }
        }
        lad("g2.leader", &set);
        // Round 1 of the Core Brain, the judge as a duel at G2's edge (guide 12.3; Core Brain note
        // 2.4: compare, do not only score). Entrants left out are taken in order of the smallest
        // log-odds shift that would carry each into the set (`shift_entrants`, step 1i's order,
        // which reaches the answer pieces), each against the set piece it would push out (the
        // smallest gain per token among the holders its swap fits), and each duel is read by the
        // judge: both pieces under the same question in one packed call, each value its label
        // log-odds less its held content-free label (`neural::judge_edge`), so no piece, a lone one
        // included, is read on the raw scale. The entrant takes the holder's place when its value
        // is the larger; nothing is added to any part's chance and nothing else in the set moves,
        // so no piece leaves the set unread. Each piece duels once. Duels are planned while their
        // reads fit the caller's budget less the time already spent and less one read of the
        // largest planned piece, at this machine's measured seconds per judged token (none
        // measured yet: the first duel only, which measures it).
        // Close calls routed by measured strength (on by default, `NANO_ROUTE=off`; research
        // item 1): the same entrants and holders as the judge's duels below, each duel read by
        // the MicroModel the strengths table names strongest for its kind (`chain_choice`): code
        // against code to the embedder's meaning read (each piece's cosine less its mean over
        // the content-free questions), any duel involving prose to the judge, which reads the
        // pair in both orders in one packed call, each order calibrated by its content-free
        // reads (PRP; ICR's equation 2); orders that disagree are a tie and move nothing, unless
        // the reader's joint read of the two (both orders, label readout) then decides it.
        let judging = JUDGE.read().is_ok_and(|g| *g);
        // Routing serves by default for the embedder half; with the judge on, the routed judge
        // half (both orders, the joint read on ties) only with `NANO_ROUTE_JUDGE=on`, else the
        // judge's own duels below, as before.
        let routing = route_on() && (!judging || route_judge_on());
        if routing && !set.is_empty() {
            let tj = Instant::now();
            let miss: Vec<f32> = (0..n_a).map(|a| 1.0 - last.covered.get(a).copied().unwrap_or(0.0)).collect();
            let rs = |c: usize| -> Vec<f32> { (0..n_a).map(|a| r(a, c)).collect() };
            let without = |y: usize| -> Vec<f32> { miss.iter().zip(rs(y)).map(|(m, x)| if x < 1.0 { m / (1.0 - x) } else { *m }).collect() };
            let mut used: usize = set.iter().map(|c| tk(*c)).sum();
            let out: Vec<usize> = cands.iter().filter(|c| !set.contains(c)).copied().collect();
            #[expect(clippy::cast_precision_loss, reason = "token counts")]
            let per = |g: f32, c: usize| g / tk(c).max(1) as f32;
            let last_in = set.last().zip(last.gains.last()).map_or(0.0, |(c, g)| per(*g, *c));
            let entrants = shift_entrants(&set, &out, &miss, &rs, &tk, last_in);
            let hold: Vec<(f32, usize)> = set.iter().map(|y| (per(without(*y).iter().zip(rs(*y)).map(|(m, x)| m * x).sum(), *y), *y)).collect();
            let root = nmtype::nano_root();
            // W6 (`NANO_W6=on`): the judge is loaded for the routed duels, so prose duels go to it.
            // W6 (on by default, `NANO_W6=off`): the judge reads the routed prose duels when it is on
            // the card (never on the processor; it warms in the background with W4's).
            let w6_fits = w6 && warm_close_judge() == JudgeState::Card;
            let model = if (JUDGE.read().is_ok_and(|g| *g) || w6_fits) && !entrants.is_empty() { crate::neural::reranker_model(&root).ok() } else { None };
            let prose = |c: usize| nmtype::is_document(&file_name(c));
            // The embedder's meaning read of a piece: its cosine with the question less its mean
            // cosine with the content-free questions (`None` without a vector).
            let meant = |c: usize| -> Option<f32> {
                let m = meaning.as_ref()?;
                let posed = m.first()?.get(c).copied().filter(|x| x.is_finite())?;
                let cf: Vec<f32> = m.iter().skip(1).filter_map(|v| v.get(c).copied()).filter(|x| x.is_finite()).collect();
                #[expect(clippy::cast_precision_loss, reason = "a count of forms")]
                let mean = (!cf.is_empty()).then(|| cf.iter().sum::<f32>() / cf.len() as f32)?;
                Some(posed - mean)
            };
            let terms = crate::whole::term_weights(sub, question);
            // The embedder's tie band: the noise band (median absolute deviation) of its
            // calibrated values over the candidates; a difference inside it is a tie and moves
            // nothing (try 2: try 1 swapped on any difference and lost answers).
            let meant_band = crate::coordinator::noise_band(&cands.iter().filter_map(|c| meant(*c)).collect::<Vec<f32>>());
            // Each duel: entrant, holder, the shift the entrant needed, the verdict's difference
            // (entrant less holder) and whether the entrant takes the holder's place.
            let mut verdicts: Vec<(usize, usize, f32, f32, bool)> = Vec::new();
            let mut pairs_planned: Vec<(f32, usize, usize, String, String, String, String)> = Vec::new();
            let mut dueled: Vec<usize> = Vec::new();
            // Homeostasis of the call's time (`NANO_HOMEOSTASIS=off`): the judge's duels are planned
            // only while their expected cost fits after the time the close-call judge (higher value
            // per second) is expected to need, and within what is left after that reserve.
            let budget_on = homeostasis_on();
            let duels_ok = !budget_on || nmbrain::homeostasis::TIME_BUDGET.lock().is_ok_and(|b| b.allow(nmbrain::homeostasis::Stage::Duels, call_budget_s() - t0.elapsed().as_secs_f64(), &[nmbrain::homeostasis::Stage::Judge]));
            let reserve = if budget_on && w4 { nmbrain::homeostasis::TIME_BUDGET.lock().map_or(0.0, |b| b.stats.get(&nmbrain::homeostasis::Stage::Judge).map_or(0.75, |s| s.0)) } else { 0.0 };
            let left = call_budget_s() - t0.elapsed().as_secs_f64() - reserve;
            let rate = JUDGE_RATE.lock().ok().and_then(|g| (g.1 > 0.0).then(|| g.0 / g.1));
            let mut cost = 0_usize;
            // Stage 5 WP-C2: the measured routing record's calibration. Each MicroModel's slope turns
            // its raw difference into calibrated log-odds (a negative slope reads it the other way
            // round: the judge's pair prompt on code, measured); a dual-strong job (`combine`) adds
            // both MicroModels' calibrated evidence. Without a valid record nothing changes.
            let rec = measured_record();
            let job_of = |k: PieceKind| if k == PieceKind::Code { nmbrain::chain::Job::CodeClose } else { nmbrain::chain::Job::ProseClose };
            let slope_of = |k: PieceKind, m: nmbrain::chain::MicroModel| rec.and_then(|r| r.row(job_of(k), m)).filter(|r| r.n >= RECORD_MIN_DUELS && r.slope != 0.0).map(|r| r.slope);
            let combine = |k: PieceKind| {
                rec.is_some_and(|r| r.combine.contains(&job_of(k))) && slope_of(k, nmbrain::chain::MicroModel::Embedder).is_some() && slope_of(k, nmbrain::chain::MicroModel::Judge).is_some()
            };
            // Per planned judge pair: its kind and, for a combined duel, the embedder's difference.
            let mut pair_meta: Vec<(PieceKind, Option<f32>)> = Vec::new();
            for (need, x) in &entrants {
                let Some(y) = hold
                    .iter()
                    .filter(|(_, y)| !dueled.contains(y) && !lead3.contains(y) && used - tk(*y).min(used) + tk(*x) <= room)
                    .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
                    .map(|h| h.1)
                else {
                    continue;
                };
                let kind = if prose(*x) || prose(y) { PieceKind::Prose } else { PieceKind::Code };
                // A dual-strong duel needs the judge's read only when it can change the decision: when
                // the embedder's calibrated evidence alone is decisive (`COMBINE_DECISIVE`), the duel is
                // the embedder's (no judge read, no added time).
                let decisive = combine(kind)
                    && meant(*x).zip(meant(y)).zip(slope_of(kind, nmbrain::chain::MicroModel::Embedder)).is_some_and(|((vx, vy), ke)| {
                        // Roadmap Phase 0 item 4: the record's derived threshold, else ln 9.
                        let t = rec.and_then(|r| r.decisive.iter().find(|(j, _)| *j == job_of(kind)).map(|(_, t)| *t)).unwrap_or(COMBINE_DECISIVE);
                        (ke * (vx - vy)).abs() >= t
                    });
                let route = if decisive {
                    Some(MicroModel::Embedder)
                } else if combine(kind) && model.is_some() {
                    Some(MicroModel::Judge)
                } else {
                    chain_choice(kind, model.is_some())
                };
                match route {
                    Some(MicroModel::Embedder) => {
                        let (Some(vx), Some(vy)) = (meant(*x), meant(y)) else { continue };
                        dueled.push(y);
                        g2_run.routed.0 += 1;
                        // The embedder's difference read through its measured slope's sign.
                        let e = if slope_of(kind, nmbrain::chain::MicroModel::Embedder).is_some_and(|k| k < 0.0) { vy - vx } else { vx - vy };
                        verdicts.push((*x, y, *need, e, e > meant_band));
                    }
                    Some(MicroModel::Judge) => {
                        let Some(m) = model.as_ref() else { continue };
                        let count = |s: &str| m.pop.tokenizer.encode(s).map_or(0, |v| v.len());
                        let w = crate::whole::judge_window_control(&nmtype::nano_root()).unwrap_or(usize::MAX);
                        let view = |c: usize| -> Option<(String, String, String)> {
                            let p = pieces(sub, &[c], &[]).into_iter().next()?;
                            let ch = sub.chunks.get(c)?;
                            let (v, _) = crate::whole::judge_view(&p, &terms, w, &count);
                            let (u, _) = crate::whole::judge_view(&p, &[], w, &count);
                            Some((ch.key.clone(), nmsubs::dense::doc_text_with(&v.path, &ch.heading, &v.text), nmsubs::dense::doc_text_with(&u.path, &ch.heading, &u.text)))
                        };
                        let (Some((kx, px, ux)), Some((ky, py, uy))) = (view(*x), view(y)) else { continue };
                        let held = pair_unheld(&kx, &ky) == 0;
                        let posed = 2 * (count(&px) + count(&py));
                        let t = posed + if held { 0 } else { crate::neural::CONTENT_FREE.len() * 2 * (count(&ux) + count(&uy)) };
                        #[expect(clippy::cast_precision_loss, reason = "token counts")]
                        let fits = match rate {
                            Some(s) => (cost + t + posed) as f64 * s <= left,
                            None => pairs_planned.is_empty() && duels_ok,
                        };
                        if !fits {
                            // A combined duel that does not fit the time left is the embedder's alone.
                            if combine(kind)
                                && let (Some(vx), Some(vy)) = (meant(*x), meant(y))
                            {
                                dueled.push(y);
                                g2_run.routed.0 += 1;
                                verdicts.push((*x, y, *need, vx - vy, vx - vy > meant_band));
                            }
                            continue;
                        }
                        cost += t;
                        dueled.push(y);
                        pair_meta.push((kind, if combine(kind) { meant(*x).zip(meant(y)).map(|(a, b)| a - b) } else { None }));
                        pairs_planned.push((*need, *x, y, format!("{kx}|{ky}"), px, py, format!("{ux}\u{0}{uy}")));
                    }
                    None => {}
                }
            }
            if let (Some(m), false) = (model.as_ref(), pairs_planned.is_empty()) {
                let duels: Vec<PairDuel> = pairs_planned
                    .iter()
                    .map(|(_, _, _, key, px, py, plain)| {
                        let (ux, uy) = plain.split_once('\u{0}').unwrap_or(("", ""));
                        PairDuel { key: key.clone(), a: px.clone(), b: py.clone(), plain_a: ux.to_owned(), plain_b: uy.to_owned() }
                    })
                    .collect();
                let tr = Instant::now();
                if let Ok((v, tokens)) = judge_pairs(m, question, &duels) {
                    let read_s = tr.elapsed().as_secs_f64();
                    g2_run.judge_tokens += tokens;
                    g2_run.judged += 2 * v.len();
                    if tokens > 0
                        && JUDGE_WARM.swap(true, std::sync::atomic::Ordering::Relaxed)
                        && let Ok(mut g) = JUDGE_RATE.lock()
                    {
                        g.0 += read_s;
                        #[expect(clippy::cast_precision_loss, reason = "token counts")]
                        let t = tokens as f64;
                        g.1 += t;
                    }
                    let mut ties: Vec<(f32, usize, usize, f32)> = Vec::new();
                    let mut combined = 0_usize;
                    for (((need, x, y, ..), (ab, ba)), (kind, emb)) in pairs_planned.iter().zip(&v).zip(&pair_meta) {
                        g2_run.routed.1 += 1;
                        let kj = slope_of(*kind, nmbrain::chain::MicroModel::Judge);
                        // Dual-strong: both MicroModels' calibrated log-odds added.
                        if let (Some(e), Some(kj), Some(ke)) = (emb, kj, slope_of(*kind, nmbrain::chain::MicroModel::Embedder)) {
                            let z = kj * (ab - ba) / 2.0 + ke * e;
                            combined += 1;
                            verdicts.push((*x, *y, *need, z, z > 0.0));
                            continue;
                        }
                        // The judge's two orders read through its measured slope's sign.
                        let (ab, ba) = if kj.is_some_and(|k| k < 0.0) { (-*ab, -*ba) } else { (*ab, *ba) };
                        let d = (ab - ba) / 2.0;
                        if ab > 0.0 && ba < 0.0 {
                            verdicts.push((*x, *y, *need, d, true));
                        } else if ab < 0.0 && ba > 0.0 {
                            verdicts.push((*x, *y, *need, d, false));
                        } else {
                            g2_run.routed.2 += 1;
                            ties.push((*need, *x, *y, d));
                        }
                    }
                    if combined > 0 {
                        g2_run.lines.push(format!("ROUTED: {combined} duels decided by both MicroModels' calibrated log-odds (routing record)"));
                    }
                    // A still-split tie goes to the reader's joint read of the two pieces (both
                    // orders, the label readout); its best takes the place.
                    // The reader on the card only (never on the processor; warmed in the background).
                    let base = if ties.is_empty() || warm_reader() != JudgeState::Card { None } else { crate::neural::model(&root).ok() };

                    let drones = base.as_ref().map(|_| crate::tentacle_pass::drones(&root));
                    // Stage 5 WP-C3 J6: every still-split tie of the call read in one packed card run
                    // (both orders of each pair, the label readout), in place of one joint read per
                    // tie; a failed packed read leaves each tie to its own read as before.
                    let packed: Option<Vec<f32>> = base
                        .as_ref()
                        .filter(|b| b.pop.gpu.is_some() && ties.len() > 1 && crate::joint_read::packed_labels_on() && t0.elapsed().as_secs_f64() < call_budget_s())
                        .and_then(|b| {
                            let groups: Vec<Vec<Piece>> = ties.iter().map(|(_, x, y, _)| pieces(sub, &[*x, *y], &[])).collect();
                            if groups.iter().any(|g| g.len() != 2) {
                                return None;
                            }
                            let refs: Vec<Vec<&Piece>> = groups.iter().map(|g| g.iter().collect()).collect();
                            let (v, _) = crate::joint_read::read_labels_many(b, question, &refs, true).ok()?;
                            Some(v.iter().map(|s| s.first().copied().unwrap_or(0.0) - s.get(1).copied().unwrap_or(0.0)).collect())
                        });
                    for (ti, (need, x, y, d)) in ties.into_iter().enumerate() {
                        // Within the call's budget only (lane T1, 2026-11-19: unbudgeted tie reads
                        // cost up to 6 s a call once the team's distinct windows brought more prose
                        // duels); past it the tie moves nothing.
                        let in_time = t0.elapsed().as_secs_f64() < call_budget_s();
                        let pre = packed.as_ref().and_then(|v| v.get(ti).copied());
                        let decided = pre.or_else(|| base.as_ref().zip(drones.as_ref()).filter(|_| in_time).and_then(|(b, set_d)| {

                            let two: Vec<Piece> = pieces(sub, &[x, y], &[]);
                            let dev = if b.pop.gpu.is_some() { nmengn::placement::Device::Gpu } else { nmengn::placement::Device::Cpu };
                            let jo = crate::joint_read::read_with(b, dev, question, &two, &set_d.heads, set_d.last_layer(), usize::MAX, true, true, false).ok()?;
                            let s = &jo.yes_no.group_score;
                            Some(s.first().copied().unwrap_or(0.0) - s.get(1).copied().unwrap_or(0.0))
                        }));
                        match decided {
                            Some(j) if j.is_finite() && j != 0.0 => {
                                g2_run.routed.3 += 1;
                                verdicts.push((x, y, need, j, j > 0.0));
                            }
                            _ => verdicts.push((x, y, need, d, false)),
                        }
                    }
                }
            }
            // Stage 5 WP-C2 ability audit (`NANO_DUEL_AUDIT=<file>`, a measurement, off by default):
            // every duel of this call read by both MicroModels on the same pair (the embedder's
            // meaning difference, entrant less holder, and the judge's both-orders pair read, each
            // order calibrated by its content-free reads), one JSON line per duel appended to the
            // file; nothing is decided from it. `nano-lab stage-bench routing-record` scores the lines
            // against the questions' answers into `designs/routing-record.json`.
            // Roadmap Phase 0 item 4 (2026-11-23): `NANO_DUEL_AUDIT_WIDE=<k>` adds every pair among
            // the first `k` of G2's candidates to the audited duels (the planned duels alone gave 35
            // informative code duels over the six tuning sets, too few to derive the judge-skip
            // threshold); the judge is loaded for them when no duel was planned. Measurement only.
            let wide = std::env::var("NANO_DUEL_AUDIT_WIDE").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
            let mut apairs: Vec<(usize, usize)> = verdicts.iter().map(|v| (v.0, v.1)).collect();
            if wide > 1 {
                let top: Vec<usize> = cands.iter().copied().take(wide).collect();
                for (i, a) in top.iter().enumerate() {
                    for b in top.iter().skip(i + 1) {
                        if !apairs.iter().any(|(x, y)| (x == a && y == b) || (x == b && y == a)) {
                            apairs.push((*a, *b));
                        }
                    }
                }
            }
            if let Some(audit) = std::env::var_os("NANO_DUEL_AUDIT").filter(|v| !v.is_empty())
                && !apairs.is_empty()
            {
                let refer = |c: usize| sub.chunks.get(c).map_or_else(String::new, |ch| format!("{}:{}-{}", file_name(c), ch.start, ch.end));
                let mut judged: Vec<Option<f32>> = vec![None; apairs.len()];
                let audit_model = if model.is_some() { model.clone() } else if wide > 1 { crate::neural::reranker_model(&root).ok() } else { None };
                if let Some(m) = audit_model.as_ref() {
                    let count = |s: &str| m.pop.tokenizer.encode(s).map_or(0, |v| v.len());
                    let w = crate::whole::judge_window_control(&nmtype::nano_root()).unwrap_or(usize::MAX);
                    let view = |c: usize| -> Option<(String, String, String)> {
                        let p = pieces(sub, &[c], &[]).into_iter().next()?;
                        let ch = sub.chunks.get(c)?;
                        let (v, _) = crate::whole::judge_view(&p, &terms, w, &count);
                        let (u, _) = crate::whole::judge_view(&p, &[], w, &count);
                        Some((ch.key.clone(), nmsubs::dense::doc_text_with(&v.path, &ch.heading, &v.text), nmsubs::dense::doc_text_with(&u.path, &ch.heading, &u.text)))
                    };
                    let mut at: Vec<usize> = Vec::new();
                    let mut duels: Vec<PairDuel> = Vec::new();
                    for (i, (x, y)) in apairs.iter().enumerate() {
                        if let (Some((kx, px, ux)), Some((ky, py, uy))) = (view(*x), view(*y)) {
                            at.push(i);
                            duels.push(PairDuel { key: format!("{kx}|{ky}"), a: px, b: py, plain_a: ux, plain_b: uy });
                        }
                    }
                    if let Ok((v, _)) = judge_pairs(m, question, &duels) {
                        for (i, (ab, ba)) in at.iter().zip(&v) {
                            if let Some(slot) = judged.get_mut(*i) {
                                *slot = Some((ab - ba) / 2.0);
                            }
                        }
                    }
                }
                let num = |x: Option<f32>| x.filter(|v| v.is_finite()).map_or_else(|| "null".to_owned(), |v| format!("{v:.5}"));
                let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('\r', "\\r").replace('\t', "\\t");
                let mut lines = String::new();
                for ((x, y), j) in apairs.iter().zip(&judged) {
                    let kind = if prose(*x) || prose(*y) { PieceKind::Prose } else { PieceKind::Code };
                    let routed = match chain_choice(kind, model.is_some()) {
                        Some(MicroModel::Embedder) => "embedder",
                        Some(MicroModel::Judge) => "judge",
                        None => "none",
                    };
                    let emb = meant(*x).zip(meant(*y)).map(|(a, b)| a - b);
                    // Roadmap Phase 0 item 4: the lens in force, so the record built from these
                    // lines is used only with the same lens (its embedder slope depends on it).
                    let lens_key = crate::distil::lens_key();
                    lines.push_str(&format!(
                        "{{\"question\":\"{}\",\"a\":\"{}\",\"b\":\"{}\",\"kind\":\"{}\",\"routed\":\"{routed}\",\"embedder\":{},\"judge\":{},\"lens\":\"{}\"}}\n",
                        esc(question),
                        esc(&refer(*x)),
                        esc(&refer(*y)),
                        if kind == PieceKind::Prose { "prose" } else { "code" },
                        num(emb),
                        num(*j),
                        esc(&lens_key)
                    ));
                }
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&audit) {
                    let _ = std::io::Write::write_all(&mut f, lines.as_bytes());
                }
            }
            for (x, y, close, d, wins) in verdicts {
                g2_run.judge_duels.push((x, y, close, d));
                if wins
                    && used - tk(y).min(used) + tk(x) <= room
                    && let Some(at) = set.iter().position(|c| *c == y)
                {
                    let (rx, ry) = (rs(x), rs(y));
                    let mut delta = 0.0_f32;
                    for a in 0..n_a {
                        let (Some(cov), Some(gx), Some(gy)) = (last.covered.get_mut(a), rx.get(a), ry.get(a)) else { continue };
                        let m = 1.0 - *cov;
                        let m_out = if *gy < 1.0 { m / (1.0 - gy) } else { m };
                        let new = 1.0 - m_out * (1.0 - gx);
                        delta += new - *cov;
                        *cov = new;
                    }
                    last.value += delta;
                    if let Some(s) = set.get_mut(at) {
                        *s = x;
                    }
                    used = used - tk(y).min(used) + tk(x);
                    g2_run.judge_in.push(x);
                    g2_run.judge_out.push(y);
                }
            }
            g2_run.judge_moved = g2_run.judge_in.len() + g2_run.judge_out.len();
            g2_run.judge_ms = tj.elapsed().as_secs_f64() * 1000.0;
            if g2_run.judge_tokens > 0
                && let Ok(mut b) = nmbrain::homeostasis::TIME_BUDGET.lock()
            {
                b.observe(nmbrain::homeostasis::Stage::Duels, g2_run.judge_ms / 1000.0, g2_run.judge_moved > 0);
            }
        }
        if judging && !set.is_empty() && !routing {
            let tj = Instant::now();
            let miss: Vec<f32> = (0..n_a).map(|a| 1.0 - last.covered.get(a).copied().unwrap_or(0.0)).collect();
            let rs = |c: usize| -> Vec<f32> { (0..n_a).map(|a| r(a, c)).collect() };
            // The misses the set leaves without its piece `y`.
            let without = |y: usize| -> Vec<f32> { miss.iter().zip(rs(y)).map(|(m, x)| if x < 1.0 { m / (1.0 - x) } else { *m }).collect() };
            let mut used: usize = set.iter().map(|c| tk(*c)).sum();
            let out: Vec<usize> = cands.iter().filter(|c| !set.contains(c)).copied().collect();
            #[expect(clippy::cast_precision_loss, reason = "token counts")]
            let per = |g: f32, c: usize| g / tk(c).max(1) as f32;
            let last_in = set.last().zip(last.gains.last()).map_or(0.0, |(c, g)| per(*g, *c));
            let entrants = shift_entrants(&set, &out, &miss, &rs, &tk, last_in);
            // Each set piece's gain per token over the misses the set leaves without it: the
            // holder an entrant would push out is the one with the smallest.
            let hold: Vec<(f32, usize)> = set.iter().map(|y| (per(without(*y).iter().zip(rs(*y)).map(|(m, x)| m * x).sum(), *y), *y)).collect();
            if let (false, Ok(model)) = (entrants.is_empty(), crate::neural::reranker_model(&nmtype::nano_root())) {
                let terms = crate::whole::term_weights(sub, question);
                let count = |s: &str| model.pop.tokenizer.encode(s).map_or(0, |v| v.len());
                let left = call_budget_s() - t0.elapsed().as_secs_f64();
                let rate = JUDGE_RATE.lock().ok().and_then(|g| (g.1 > 0.0).then(|| g.0 / g.1));
                let forms = crate::neural::CONTENT_FREE.len();
                let reads = |posed: &str, plain: &str, held: bool| count(posed) + if held { 0 } else { forms * count(plain) };
                let stems: Vec<&str> = ev.stems.iter().flatten().map(String::as_str).collect();
                let make = |c: usize| -> Option<crate::neural::EdgePiece> {
                    let p = pieces(sub, &[c], &[]).into_iter().next()?;
                    let ch = sub.chunks.get(c)?;
                    let w = crate::whole::judge_window_control(&nmtype::nano_root()).unwrap_or(usize::MAX);
                    // In pack form the judge reads, posed and content-free, the lines the pack
                    // would show of the piece (`pack_form`), its label held by the piece and that
                    // text; else the request's window of the piece, and its first lines with no
                    // request for the label.
                    Some(match pack_form(sub, c, &stems).filter(|_| PACK_READ.load(std::sync::atomic::Ordering::Relaxed)) {
                        Some(t) => {
                            let doc = nmsubs::dense::doc_text_with(&p.path, &ch.heading, &t);
                            crate::neural::EdgePiece { key: format!("{}#{}", ch.key, nmhash::sha256_bytes(doc.as_bytes())), posed: doc.clone(), plain: doc }
                        }
                        None => {
                            let (v, _) = crate::whole::judge_view(&p, &terms, w, &count);
                            let (u, _) = crate::whole::judge_view(&p, &[], w, &count);
                            crate::neural::EdgePiece {
                                key: ch.key.clone(),
                                posed: nmsubs::dense::doc_text_with(&v.path, &ch.heading, &v.text),
                                plain: nmsubs::dense::doc_text_with(&u.path, &ch.heading, &u.text),
                            }
                        }
                    })
                };
                // Entrants in order of the shift they need, each against the set piece it would
                // push out (the smallest gain per token among holders not yet in a duel whose
                // swap fits the room), while the reads fit the time left less one read of the
                // largest planned piece.
                let mut docs: Vec<(usize, crate::neural::EdgePiece, usize)> = Vec::new();
                let mut planned: Vec<(f32, usize, usize)> = Vec::new();
                let (mut cost, mut largest) = (0_usize, 0_usize);
                for (need, x) in &entrants {
                    let Some(y) = hold
                        .iter()
                        .filter(|(_, y)| !planned.iter().any(|d| d.2 == *y) && used - tk(*y).min(used) + tk(*x) <= room)
                        .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
                        .map(|h| h.1)
                    else {
                        continue;
                    };
                    let (close, x, y) = (need, x, &y);
                    let (Some(px), Some(py)) = (make(*x), make(*y)) else { continue };
                    let held = |p: &crate::neural::EdgePiece| crate::neural::edge_unheld(&model, &[p.key.as_str()]) == 0;
                    let (tx, ty) = (reads(&px.posed, &px.plain, held(&px)), reads(&py.posed, &py.plain, held(&py)));
                    // One read: the posed pair of the largest planned piece.
                    let big = largest.max(count(&px.posed)).max(count(&py.posed));
                    #[expect(clippy::cast_precision_loss, reason = "token counts")]
                    let fits = match rate {
                        Some(s) => (cost + tx + ty + big) as f64 * s <= left,
                        None => planned.is_empty(),
                    };
                    if !fits {
                        // A duel that does not fit is skipped; a later one whose labels are held
                        // may still fit (`NANO_JUDGE_SKIP=on`); else planning ends here.
                        if judge_skip() {
                            continue;
                        }
                        break;
                    }
                    cost += tx + ty;
                    largest = big;
                    docs.push((*x, px, tx));
                    docs.push((*y, py, ty));
                    planned.push((*close, *x, *y));
                }
                let tr = Instant::now();
                // One packed read: every posed pair and the content-free pairs of pieces seen first.
                let pieces_read: Vec<crate::neural::EdgePiece> =
                    docs.iter().map(|d| crate::neural::EdgePiece { key: d.1.key.clone(), posed: d.1.posed.clone(), plain: d.1.plain.clone() }).collect();
                let read = if pieces_read.is_empty() { Ok((Vec::new(), crate::neural::EdgeRead::default())) } else { crate::neural::judge_edge(&model, question, &pieces_read) };
                // At idle (`NANO_JUDGE_IDLE=on`): the content-free labels of this call's entrants
                // and holders not yet held are read after the call, in the background, so a later
                // duel on them costs only its two posed reads.
                if judge_idle() {
                    let idle: Vec<crate::neural::EdgePiece> = entrants
                        .iter()
                        .map(|(_, x)| *x)
                        .chain(set.iter().copied())
                        .filter(|c| !docs.iter().any(|d| d.0 == *c))
                        .filter_map(make)
                        .filter(|p| crate::neural::edge_unheld(&model, &[p.key.as_str()]) > 0)
                        .collect();
                    if !idle.is_empty() {
                        let m = std::sync::Arc::clone(&model);
                        std::thread::spawn(move || {
                            let _ = crate::neural::judge_edge(&m, "", &idle);
                        });
                    }
                }
                let read_s = tr.elapsed().as_secs_f64();
                if let Ok((v, _)) = read {
                    let value: BTreeMap<usize, f32> = docs.iter().zip(&v).map(|(d, y)| (d.0, *y)).collect();
                    for (close, x, y) in &planned {
                        let (Some(vx), Some(vy)) = (value.get(x), value.get(y)) else { continue };
                        let d = vx - vy;
                        g2_run.judge_pairs.extend([(*x, *close, *vx), (*y, *close, *vy)]);
                        g2_run.judge_duels.push((*x, *y, *close, d));
                        // The duel's verdict decides that one swap: the entrant takes the
                        // holder's place, and the set's coverage of the word aspects follows.
                        if d > 0.0
                            && used - tk(*y).min(used) + tk(*x) <= room
                            && let Some(at) = set.iter().position(|c| c == y)
                        {
                            let (rx, ry) = (rs(*x), rs(*y));
                            let mut delta = 0.0_f32;
                            for a in 0..n_a {
                                let (Some(cov), Some(gx), Some(gy)) = (last.covered.get_mut(a), rx.get(a), ry.get(a)) else { continue };
                                let m = 1.0 - *cov;
                                let m_out = if *gy < 1.0 { m / (1.0 - gy) } else { m };
                                let new = 1.0 - m_out * (1.0 - gx);
                                delta += new - *cov;
                                *cov = new;
                            }
                            last.value += delta;
                            if let Some(s) = set.get_mut(at) {
                                *s = *x;
                            }
                            used = used - tk(*y).min(used) + tk(*x);
                            g2_run.judge_in.push(*x);
                            g2_run.judge_out.push(*y);
                        }
                    }
                    g2_run.judged = v.len();
                    g2_run.judge_tokens = docs.iter().map(|d| d.2).sum();
                    // The measured rate: the judge's reads only (not the model's opening), and not
                    // the first read of the process, which also loads the card's routines.
                    if g2_run.judge_tokens > 0
                        && JUDGE_WARM.swap(true, std::sync::atomic::Ordering::Relaxed)
                        && let Ok(mut g) = JUDGE_RATE.lock()
                    {
                        g.0 += read_s;
                        #[expect(clippy::cast_precision_loss, reason = "token counts")]
                        let t = g2_run.judge_tokens as f64;
                        g.1 += t;
                    }
                }
            }
            g2_run.judge_moved = g2_run.judge_in.len() + g2_run.judge_out.len();
            g2_run.judge_ms = tj.elapsed().as_secs_f64() * 1000.0;
        }
        // Each chosen piece G1 did not hand over is answered by its holder from what it holds
        // (ANSWER through G1's ledger) and goes up regardless of rank (FORWARD).
        let mut replies: Vec<(SpanRef, String, f32, usize, Vec<usize>)> = Vec::new();
        let mut forwarded: Vec<usize> = Vec::new();
        for c in set.iter().filter(|c| !handover.contains(c)) {
            let Some(span) = span_of(*c) else { continue };
            let Some((ri, k)) = place(*c, &rounds) else { continue };
            let id = g2_run.motor.id();
            let Some((_, round)) = rounds.get_mut(ri) else { continue };
            // The piece's facts as its `g0_tx` last reported them (no new report is built).
            let f = round.outcome.last.get(k).and_then(|rep| rep.facts.get(&span.key)).cloned().unwrap_or_default();
            let Some(seg) = round.segments.get_mut(k) else { continue };
            let m = Motor::new(id, None, MessageKind::Instruction, Verb::Answer, nmbrain::motor::G2, seg.id(), 0.5, "chosen by G2's selection", expiry)
                .and_then(|m| m.arg("pieces", Arg::Pieces(vec![span.key.clone()])))
                .and_then(|m| m.reference(nmtype::Reference::Span { path: span.file.clone(), start: u32::try_from(span.start.max(1)).unwrap_or(1), end: u32::try_from(span.end.max(span.start).max(1)).unwrap_or(1) }));
            let Ok(m) = m else { continue };
            g2_run.edges.extend([("g2", "g1", Behavioural), ("g1", "g0", Behavioural)]);
            if g2_run.motor.send(m, seg.as_mut(), nmbrain::g0::Instruction::Ask { span: span.clone(), about: question.to_owned() }, now).is_err() {
                continue;
            }
            for reply in seg.process() {
                if let nmbrain::g0::Reply::Answer(a) = reply
                    && a.held
                {
                    g2_run.edges.extend([("g0", "g1", Behavioural), ("g0", "g1", Sensory)]);
                    replies.push((a.span.clone(), a.from.clone(), value(*c), f.quorum, f.aspects.clone()));
                    // Stage 5 WP-C2: the reply is matched against the efference copy of the order
                    // G2 sent (`Down::send` registered it), closing the motor loop.
                    let _ = g2_run.up.reply_seen(&mut g2_run.motor.efference, id, a);
                    forwarded.push(*c);
                }
            }
        }
        {
            let e = &g2_run.motor.efference;
            g2_run.lines.push(format!("EFFERENCE: {} replies matched to orders sent, {} from outside, {} orders awaiting a reply", e.matched, e.unexpected, e.expected.len()));
            let (hits, misses) = crate::cache_reads::judge_stats();
            g2_run.lines.push(format!("JUDGE CACHE: {hits} pieces served from the cache, {misses} read (process totals; NANO_JUDGE_CACHE={})", if crate::cache_reads::judge_cache_on() { "on" } else { "off" }));
        }
        if !forwarded.is_empty() {
            let id = g2_run.motor.id();
            let keys: Vec<String> = forwarded.iter().filter_map(|c| span_of(*c)).map(|s| s.key).collect();
            if let Ok(m) = Motor::new(id, None, MessageKind::Instruction, Verb::Forward, nmbrain::motor::G2, "g1", 0.5, "chosen pieces go up regardless of rank", expiry)
                .and_then(|m| m.arg("pieces", Arg::Pieces(keys)))
                && g2_run.motor.ledger.record(&m).is_ok()
            {
                g2_run.motor.sent.push(m);
                g2_run.edges.push(("g2", "g1", Behavioural));
            }
        }
        // G1-a: each reply with the kinds of evidence that stand above the content-free band on it
        // (the quorum then counts kinds by their measured independence).
        let kinds_of = |key: &str| -> Vec<String> {
            let Some(c) = chunk_of(key) else { return Vec::new() };
            let Some(s) = signals.get(c) else { return Vec::new() };
            let mut k = Vec::new();
            if s.bm25f > 0.0 || s.bm25 > 0.0 {
                k.push("wording".to_owned());
            }
            if s.exact > 0.0 || s.ident > 0.0 {
                k.push("naming".to_owned());
            }
            if s.affinity > 0.0 {
                k.push("meaning".to_owned());
            }
            if s.structural > 0.0 {
                k.push("structure".to_owned());
            }
            if read_all.borrow().get(&c).is_some_and(|x| *x > 0.0) {
                k.push("reading".to_owned());
            }
            k
        };
        let reply_kinds: Vec<Vec<String>> = replies.iter().map(|r| kinds_of(&r.0.key)).collect();
        g1.race.feed_kinds(&replies, &reply_kinds);

        // Up: the chosen set in G1's order (its hand-over, then its queue; a piece read on an
        // order after them); then the room the caller gave is filled in G1's order: the pieces
        // that rival the set and what the pack lists after it.
        lad("g2.duels", &set);
        // W4: the judge's close-call evidence as a kind. On a close call by the fused value (the
        // best candidate's margin over the next within the noise band of the first `CLOSE_K`), the
        // judge reads those candidates (about 128 tokens each, one packed read); its log-odds,
        // standardised over them and clipped as every kind is, join their value; the best then
        // enters the set if it is not in it (the weakest by the value leave to fit the room).
        let mut judged: BTreeMap<usize, f32> = BTreeMap::new();
        // The judge's raw label log-odds of the close call's pieces (Phase I's judge hop).
        let mut w4_z: BTreeMap<usize, f32> = BTreeMap::new();
        // W4's second form (default; `NANO_W4_FORM=kind` the first): only the leader can change.
        let w4_kind = std::env::var("NANO_W4_FORM").is_ok_and(|v| v == "kind");
        let mut w4_leader: Option<usize> = None;
        // C132: a tentacle's own pick the judge reads above the value's first pieces takes the
        // second place of G2's order (behind the leader).
        let mut own_second: Option<usize> = None;
        let mut suff_need: Option<usize> = None;
        // W4 safety: the arm fires only when the judge is already on the card (warmed in the
        // background, `warm_close_judge`); never on the processor, and no call waits for its load.
        let judge_state = if w4 { warm_close_judge() } else { JudgeState::Card };
        if w4 && judge_state != JudgeState::Card {
            g2_run.lines.push(match judge_state {
                JudgeState::NoCard => "W4: no card; the close-call judge stays off".to_owned(),
                _ => "W4: the close-call judge is loading on the card; not used on this call".to_owned(),
            });
        }
        // Pre-compute tentacle own-pick admission lists so when the close-call judge fires below,
        // it scores `top` + any unjudged own-picks / lead3 in ONE batched reranker call instead of
        // paying the full reranker prefix + 28-layer weight read twice (~300 ms saved per call,
        // bit-identical per-document scores).
        let own_admit_on = own_on && w2 && !std::env::var("NANO_OWN_ADMIT").is_ok_and(|v| v == "off");
        let per_t = if own_admit_on { own_by_t.borrow().clone() } else { Vec::new() };
        let prose = |c: usize| nmtype::is_document(&file_name(c));
        let prose_picks: Vec<usize> = if own_admit_on {
            per_t.iter().flatten().copied().filter(|c| prose(*c) && cands.contains(c)).collect::<std::collections::BTreeSet<usize>>().into_iter().collect()
        } else {
            Vec::new()
        };
        let mut firsts: Vec<(usize, f32)> = if own_admit_on {
            let mut v: Vec<(usize, f32)> =
                own_readfirst.borrow().iter().copied().filter(|(c, x)| *x > 0.0 && cands.contains(c) && !lead3.contains(c)).collect();
            v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            v.dedup_by_key(|x| x.0);
            // Cap unjudged tentacle own-picks sent to the reranker to the top 4 strongest reads
            // (at most `OWN_ADMIT = 2` can be admitted anyway), cutting `g2 own picks` latency.
            v.truncate(OWN_ADMIT.saturating_mul(2).max(4));
            v
        } else {
            Vec::new()
        };
        if w4 && judge_state == JudgeState::Card {
            let one = one_rc.borrow().clone();
            let val = |c: usize| one.get(c).copied().unwrap_or(f32::MIN);
            let mut top: Vec<usize> = cands.clone();
            top.sort_by(|a, b| val(*b).total_cmp(&val(*a)).then(a.cmp(b)));
            top.dedup();
            top.truncate(CLOSE_K);
            let vals: Vec<f32> = top.iter().map(|c| val(*c)).collect();
            let lead = match vals.as_slice() {
                [a, b, ..] => a - b,
                _ => f32::INFINITY,
            };
            let band = crate::coordinator::noise_band(&vals);
            g2_run.close_lead = (lead, band);
            let gate_k = if w4_kind { 1.0 } else { W4_GATE_K };
            // The tentacles' own picks (each tentacle's best by its perspective) join the judge's
            // read beside the value's first pieces: G2 and the judge order the union (C132).
            let k0 = top.len();
            // The tentacles' own picks are read by the own-pick read below (the job map's model for
            // their kind), not added to the close call's judge read (on code the judge is below the
            // embedder, C102: the judge's own-pick reads moved answers down on large code).
            if own_on && std::env::var("NANO_OWN_IN_W4").is_ok_and(|v| v == "on") {
                // Each tentacle's own first two (`OWN_K`): a relations tentacle's second find held
                // large code trc09's answer that no other arm held (its first was elsewhere).
                for c in own_all.borrow().iter().copied().take(OWN_READ) {
                    if cands.contains(&c) && !top.contains(&c) {
                        top.push(c);
                    }
                }
            }
            if top.len() >= 2 && lead <= gate_k * band {
                let tc = Instant::now();
                if let Ok(m) = crate::neural::reranker_model(&nmtype::nano_root()) {
                    let mut judge_list: Vec<usize> = top.clone();
                    for c in &prose_picks {
                        if !judge_list.contains(c) {
                            judge_list.push(*c);
                        }
                    }
                    if !firsts.is_empty() {
                        for c in &lead3 {
                            if !judge_list.contains(c) {
                                judge_list.push(*c);
                            }
                        }
                        for (c, _) in &firsts {
                            if !judge_list.contains(c) {
                                judge_list.push(*c);
                            }
                        }
                    }
                    let docs: Vec<String> = judge_list
                        .iter()
                        .map(|c| {
                            let ch = sub.chunks.get(*c);
                            let place = format!("{}:{}-{}", file_name(*c), ch.map_or(0, |x| x.start), ch.map_or(0, |x| x.end));
                            crate::neural::close_doc(&m, &place, ch.map_or("", |x| x.text.as_str()))
                        })
                        .collect();
                    if let Ok(z_all) = crate::cache_reads::judge_close(&m, question, &docs) {
                        w4_z = judge_list.iter().copied().zip(z_all.iter().copied()).filter(|(_, x)| x.is_finite()).collect();
                        let z = z_all.get(..top.len()).unwrap_or(&z_all);
                        let mut col = vec![f32::NAN; sub.chunks.len()];
                        for (c, x) in top.iter().zip(z) {
                            if let Some(y) = col.get_mut(*c) {
                                *y = *x;
                            }
                        }
                        // The judge's scale and the leader from the value's first pieces only (W4's
                        // measured form); the tentacles' own picks are read on that same scale.
                        let add = fuse_kinds(&[col], top.get(..k0).unwrap_or(&top));
                        let jz: BTreeMap<usize, f32> = top.iter().map(|c| (*c, add.get(*c).copied().unwrap_or(0.0))).collect();
                        g2_run.close_fired = top.len();
                        let best = top.iter().take(k0).copied().max_by(|a, b| {
                            (val(*a) + jz.get(a).copied().unwrap_or(0.0)).total_cmp(&(val(*b) + jz.get(b).copied().unwrap_or(0.0))).then(b.cmp(a))
                        });
                        // An own pick the judge reads at least as answering as the leader enters the
                        // set behind it (the weakest by the value leave to fit); it does not lead
                        // (taking the lead cost first places on dev d01, d05, d28).
                        // The judge must read the pick above every one of the value's first pieces,
                        // and at most `OWN_IN` enter a call (picks over the leader's read alone
                        // pushed the value's own second and third pieces out on large code: val03,
                        // exp18).
                        let jtop = top.iter().take(k0).filter_map(|c| jz.get(c).copied()).fold(f32::NEG_INFINITY, f32::max);
                        let mut own_in: Vec<usize> = top.iter().skip(k0).copied().filter(|c| jz.get(c).is_some_and(|z| *z > jtop) && !set.contains(c)).collect();
                        own_in.sort_by(|a, b| jz.get(b).copied().unwrap_or(0.0).total_cmp(&jz.get(a).copied().unwrap_or(0.0)).then(a.cmp(b)));
                        own_in.truncate(OWN_IN);
                        own_second = own_in.first().copied();
                        for h in own_in {
                            set.push(h);
                            let mut used: usize = set.iter().map(|c| tk(*c)).sum();
                            while used > room && set.len() > 1 {
                                let Some(i) = set.iter().enumerate().filter(|(_, c)| **c != h && Some(**c) != best).min_by(|a, b| val(*a.1).total_cmp(&val(*b.1))).map(|(i, _)| i) else { break };
                                used -= tk(set.remove(i));
                            }
                            g2_run.lines.push(format!("OWN: {} entered by the judge's read", file_name(h)));
                        }
                        g2_run.sufficient = best.and_then(|b| top.iter().position(|c| *c == b)).and_then(|i| z.get(i).copied()).filter(|x| x.is_finite());

                        if w4_kind {
                            judged = jz;
                        } else {
                            w4_leader = best;
                        }
                        if let Some(h) = best.filter(|h| !set.contains(h)) {
                            set.push(h);
                            let mut used: usize = set.iter().map(|c| tk(*c)).sum();
                            while used > room && set.len() > 1 {
                                let Some(i) = set
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, c)| **c != h)
                                    .min_by(|a, b| val(*a.1).total_cmp(&val(*b.1)))
                                    .map(|(i, _)| i)
                                else {
                                    break;
                                };
                                used -= tk(set.remove(i));
                            }
                        }
                    }
                }
                g2_run.close_ms = tc.elapsed().as_secs_f64() * 1000.0;
            } else if own_on && top.len() > k0 && std::env::var("NANO_OWN_JUDGE").is_ok_and(|v| v == "on") {
                // No close call: the judge still reads the tentacles' own picks beside the
                // value's first three (Luis, 2026-11-19: only the judge tells a right own find from
                // a wrong one), and the pick it reads above all three takes the second place.
                let tc = Instant::now();
                let mut read: Vec<usize> = top.iter().take(3.min(k0)).copied().collect();
                let n_value = read.len();
                read.extend(top.iter().skip(k0).copied());
                if let Ok(m) = crate::neural::reranker_model(&nmtype::nano_root()) {
                    let docs: Vec<String> = read
                        .iter()
                        .map(|c| {
                            let ch = sub.chunks.get(*c);
                            let place = format!("{}:{}-{}", file_name(*c), ch.map_or(0, |x| x.start), ch.map_or(0, |x| x.end));
                            crate::neural::close_doc(&m, &place, ch.map_or("", |x| x.text.as_str()))
                        })
                        .collect();
                    if let Ok(z) = crate::cache_reads::judge_close(&m, question, &docs) {
                        let ztop = z.iter().take(n_value).copied().filter(|x| x.is_finite()).fold(f32::NEG_INFINITY, f32::max);
                        let best_own = read.iter().zip(&z).skip(n_value).filter(|(_, x)| x.is_finite() && **x > ztop).max_by(|a, b| a.1.total_cmp(b.1)).map(|(c, _)| *c);
                        if let Some(h) = best_own {
                            own_second = Some(h);
                            if !set.contains(&h) {
                                set.push(h);
                                let lead_keep: Vec<usize> = read.iter().take(n_value).copied().collect();
                                let mut used: usize = set.iter().map(|c| tk(*c)).sum();
                                while used > room && set.len() > 1 {
                                    let Some(i) = set.iter().enumerate().filter(|(_, c)| **c != h && !lead_keep.contains(c)).min_by(|a, b| val(*a.1).total_cmp(&val(*b.1))).map(|(i, _)| i) else { break };
                                    used -= tk(set.remove(i));
                                }
                            }
                            g2_run.lines.push(format!("OWN: {} read by the judge above the value's first three", file_name(h)));
                        }
                    }
                }
                if let Ok(mut b) = nmbrain::homeostasis::TIME_BUDGET.lock() {
                    b.observe(nmbrain::homeostasis::Stage::Judge, tc.elapsed().as_secs_f64(), own_second.is_some());
                }
                lad("g2.ownjudge", &own_second.into_iter().collect::<Vec<_>>());
            }
            if std::env::var_os("NANO_CLOSE_TRACE").is_some() {
                eprintln!("w4 close: lead {lead:.4} band {band:.4} fired {} ms {:.0}", g2_run.close_fired, g2_run.close_ms);
            }
            lad("g2.judge", &set);
            if g2_run.close_fired > 0
                && let Ok(mut b) = nmbrain::homeostasis::TIME_BUDGET.lock()
            {
                b.observe(nmbrain::homeostasis::Stage::Judge, g2_run.close_ms / 1000.0, w4_leader.is_some() && w4_leader != top.first().copied());
            }
            // Phase H2, the sufficiency read (W7 three states; `NANO_SUFFICIENCY=off`): when the
            // close call did not fire, queue the value's leader for the single batched judge pass
            // below (shared with own-pick reads so the judge runs at most once per G2 pass).
            if g2_run.sufficient.is_none()
                && !std::env::var("NANO_SUFFICIENCY").is_ok_and(|v| v == "off")
                && let Some(lead_c) = top.first().copied()
            {
                suff_need = Some(lead_c);
            }
        }
        // C132, the team's finds reach G2's set (lead, 2026-11-19, from the design): each tentacle's
        // best own find that the own-pick read rates as answering enters the set (only into free
        // room, never evicting a value piece). When the close-call judge fired above, `w4_z`
        // already holds `prose_picks`, `lead3`, and `firsts`, so `batch_need` is empty (`0 ms`).
        let mut batch_need: Vec<usize> = Vec::new();
        let mut push_need = |c: usize| {
            if !w4_z.contains_key(&c) && !batch_need.contains(&c) {
                batch_need.push(c);
            }
        };
        if let Some(c) = suff_need {
            push_need(c);
        }
        for c in &prose_picks {
            push_need(*c);
        }
        if !firsts.is_empty() {
            for c in &lead3 {
                push_need(*c);
            }
            for (c, _) in &firsts {
                push_need(*c);
            }
        }
        let tj_batch = Instant::now();
        if !batch_need.is_empty()
            && warm_close_judge() == JudgeState::Card
            && let Ok(m) = crate::neural::reranker_model(&nmtype::nano_root())
        {
            let docs: Vec<String> = batch_need
                .iter()
                .map(|c| {
                    let ch = sub.chunks.get(*c);
                    let place = format!("{}:{}-{}", file_name(*c), ch.map_or(0, |x| x.start), ch.map_or(0, |x| x.end));
                    crate::neural::close_doc(&m, &place, ch.map_or("", |x| x.text.as_str()))
                })
                .collect();
            if let Ok(z) = crate::cache_reads::judge_close(&m, question, &docs) {
                for (c, x) in batch_need.iter().copied().zip(z) {
                    if x.is_finite() {
                        w4_z.insert(c, x);
                    }
                }
            }
        }
        let batch_ms = tj_batch.elapsed().as_secs_f64() * 1000.0;
        if let Some(lead_c) = suff_need {
            g2_run.sufficient = w4_z.get(&lead_c).copied().filter(|x| x.is_finite());
            if !own_admit_on || (prose_picks.is_empty() && firsts.is_empty()) {
                g2_run.part_ms.0 += batch_ms;
            }
        }
        let mut own_promoted: Option<usize> = None;
        if own_admit_on {
            let to = Instant::now();
            let reads_now = read_all.borrow().clone();
            let jz: BTreeMap<usize, f32> = prose_picks.iter().copied().filter_map(|c| w4_z.get(&c).copied().map(|x| (c, x))).collect();
            if !firsts.is_empty() {
                let ztop = lead3.iter().filter_map(|c| w4_z.get(c).copied()).filter(|x| x.is_finite()).fold(f32::NEG_INFINITY, f32::max);
                firsts.retain(|(c, _)| w4_z.get(c).is_some_and(|x| x.is_finite() && *x > ztop));
            }
            firsts.sort_by(|a, b| {
                let za = w4_z.get(&a.0).copied().unwrap_or(f32::NEG_INFINITY);
                let zb = w4_z.get(&b.0).copied().unwrap_or(f32::NEG_INFINITY);
                zb.total_cmp(&za).then(b.1.total_cmp(&a.1)).then(a.0.cmp(&b.0))
            });
            firsts.dedup_by_key(|x| x.0);
            // The strongest reads first, at most `OWN_ADMIT` a call.
            firsts.truncate(OWN_ADMIT);
            let promote_on = !std::env::var("NANO_OWN_PROMOTE").is_ok_and(|v| v == "off");
            let mut admitted: Vec<usize> = Vec::new();
            // Mini E3.3 team mechanism: when a tentacle's own perspective find strictly beats
            // ALL three global leaders on the judge (`z > ztop`), give it a real seat near the top
            // (evicting non-essential tail pieces past index 2 if room is full, never dropping a
            // top-3 piece or a sole part holder) and preserve its rank past W2/intake re-sorts.
            for (c, _) in &firsts {
                if !cands.contains(c) || set.contains(c) || admitted.contains(c) {
                    continue;
                }
                let need = tk(*c);
                let mut used: usize = set.iter().map(|x| tk(*x)).sum();
                if used + need > room && promote_on {
                    let sole_holder = |idx: usize, cur: &[usize]| -> bool {
                        let Some(&ch) = cur.get(idx) else { return false };
                        (0..ev.parts.len()).any(|k| ev.part_holds(k, ch) && cur.iter().filter(|&&x| ev.part_holds(k, x)).count() <= 1)
                    };
                    while used + need > room && set.len() > 3 {
                        if let Some(pos) = (3..set.len()).rev().find(|&i| !sole_holder(i, &set)) {
                            let drop_c = set.remove(pos);
                            used = used.saturating_sub(tk(drop_c));
                        } else {
                            break;
                        }
                    }
                }
                if used + need > room {
                    continue;
                }
                if promote_on {
                    // Place winning tentacle own-picks at second place (index 1), never
                    // displacing the #1 leader (preserving #1 precision and decoy gating while
                    // lifting top-3/pack recall with the team's own finds).
                    let at = 1.min(set.len());
                    set.insert(at, *c);
                    if own_promoted.is_none() {
                        own_promoted = Some(*c);
                    }
                } else {
                    set.push(*c);
                }
                admitted.push(*c);
            }
            // Prose picks only enter into free room, never evicting a value piece.
            let prose_in: Vec<Vec<usize>> = per_t
                .iter()
                .map(|p| p.iter().copied().filter(|c| prose(*c) && !reads_now.contains_key(c) && jz.get(c).is_some_and(|z| *z > 0.0)).take(1).collect())
                .collect();
            for picks in &prose_in {
                let Some(c) = picks.iter().copied().find(|c| cands.contains(c)) else { continue };
                if set.contains(&c) || admitted.contains(&c) {
                    continue;
                }
                let used: usize = set.iter().map(|x| tk(*x)).sum();
                if used + tk(c) > room {
                    continue;
                }
                set.push(c);
                admitted.push(c);
            }
            let own_ms = batch_ms + to.elapsed().as_secs_f64() * 1000.0;
            g2_run.lines.push(format!("OWN: {} tentacles' finds admitted to G2's set by the own-pick read ({} prose judged) in {:.0} ms", admitted.len(), jz.len(), own_ms));
            g2_run.part_ms.1 += own_ms;
            lad("g2.ownadmit", &admitted);
        }
        let g1_place: BTreeMap<usize, usize> = handover.iter().chain(&queue).enumerate().map(|(i, c)| (*c, i)).collect();
        let mut ranked = set.clone();
        if w2 {
            let one = one_rc.borrow().clone();
            let val = |c: usize| one.get(c).copied().unwrap_or(f32::MIN) + judged.get(&c).copied().unwrap_or(0.0);
            ranked.sort_by(|a, b| val(*b).total_cmp(&val(*a)).then(a.cmp(b)));
            // W4 leader-only: the judge's winner takes the first place; every other keeps its order.
            if let Some(h) = w4_leader
                && let Some(i) = ranked.iter().position(|c| *c == h)
            {
                let c = ranked.remove(i);
                ranked.insert(0, c);
            }
            if let Some(h) = own_second.filter(|h| Some(*h) != w4_leader)
                && let Some(i) = ranked.iter().position(|c| *c == h)
                && i > 1
            {
                let c = ranked.remove(i);
                ranked.insert(1, c);
            }
            if let Some(h) = own_promoted
                && Some(h) != w4_leader
                && let Some(i) = ranked.iter().position(|c| *c == h)
            {
                let target = 1.min(ranked.len().saturating_sub(1));
                if i > target {
                    let c = ranked.remove(i);
                    ranked.insert(target, c);
                }
            }
        } else {
            ranked.sort_by_key(|c| (Some(*c) == entered, g1_place.get(c).copied().unwrap_or(usize::MAX), *c));
        }
        // The set's pieces that can settle a part by structure (code) take the places code holds
        // in G1's order, in the joint order over G2's candidates (`Evidence::joint_order`, fusion
        // constant from `Evidence::plausible`); a piece that restates (a document, notes) keeps its
        // place in G1's order, where the drones' graded evidence put it (the joint value reads
        // which of the request's words a piece holds, which in prose does not find the answer).
        if stack_joint_on() && !w2 {
            let restates = |c: usize| keep.record(c).is_some_and(|r| nmpod::evidence::tag(r, &[], nmtype::is_document(&file_name(c))) == nmpod::evidence::Tag::Documents);
            let file_of = |c: usize| sub.chunks.get(c).map_or(usize::MAX, |x| x.file);
            #[expect(clippy::cast_precision_loss, reason = "a count of pieces")]
            let k = ev.plausible(&cands) as f32;
            let joint: BTreeMap<usize, usize> = ev.joint_order(&cands, &file_of, k).into_iter().enumerate().map(|(r, (c, _))| (c, r)).collect();
            let slots: Vec<usize> = (0..ranked.len()).filter(|i| ranked.get(*i).is_some_and(|c| !restates(*c))).collect();
            let mut code: Vec<usize> = slots.iter().filter_map(|i| ranked.get(*i).copied()).collect();
            code.sort_by_key(|c| (joint.get(c).copied().unwrap_or(usize::MAX), g1_place.get(c).copied().unwrap_or(usize::MAX), *c));
            for (i, c) in slots.into_iter().zip(code) {
                if let Some(x) = ranked.get_mut(i) {
                    *x = c;
                }
            }
        }
        let mut sent = ranked.clone();
        let mut filled: usize = sent.iter().map(|c| tk(*c)).sum();
        for c in handover.iter().chain(&queue) {
            if !sent.contains(c) && filled + tk(*c) <= room {
                filled += tk(*c);
                sent.push(*c);
            }
        }
        // The gate: settled when the set holds every aspect and relation and its leader stands
        // clear of its strongest rival (a candidate outside the set holding one of its aspects)
        // by the noise band of the common scale; a tie when it does not; absent, naming what no
        // piece of the set holds.
        let held_by: &[usize] = if w5 { &selected } else { &ranked };
        let mut lacks: Vec<String> = (0..n_a).filter(|a| !held_by.iter().any(|c| ev.part_holds(*a, *c))).filter_map(|a| ev.parts.get(a).map(|p| p.1.clone())).collect();
        lacks.extend(
            asks.iter()
                .enumerate()
                .filter(|(k, _)| match fact_of(*k) {
                    Some(v) => !v.iter().any(|p| ranked.contains(p)),
                    None => rel_col(*k).is_none_or(|col| last.covered.get(col).copied().unwrap_or(0.0) <= 0.0),
                })
                .map(|(_, a)| a.name().to_owned()),
        );
        let gate = match ranked.first() {
            None => Comprehended::Absent((0..n_a).collect()),
            Some(_) if !lacks.is_empty() => Comprehended::Absent(Vec::new()),
            Some(l) => {
                let la = ev.covers(*l);
                let rival = cands.iter().filter(|c| !ranked.contains(c) && ev.covers(**c).iter().any(|a| la.contains(a))).map(|c| value(*c)).fold(f32::NEG_INFINITY, f32::max);
                // Phase G1-a: the commit margin, fitted on the tuning sets' decoys once the final
                // run's ladder gives them (`NANO_COMMIT_MARGIN`, in the common scale's units); until
                // then the common scale's noise band (stated fallback).
                let commit = std::env::var("NANO_COMMIT_MARGIN").ok().and_then(|v| v.parse::<f32>().ok()).filter(|x| x.is_finite()).unwrap_or(cs.band);
                let lead_m = value(*l) - rival;
                lad_value("g2.commit", format!("{{\"lead\":{},\"margin\":{commit:.4}}}", if lead_m.is_finite() { format!("{lead_m:.4}") } else { "null".to_owned() }));

                // A leader with no finite rival (no candidate outside the set shares its aspects)
                // is not settled by default (an infinite lead settled large2-code's decoy dec101,
                // lead 2026-11-19): it settles only when the judge's sufficiency read says the set
                // answers; otherwise it is a tie (low).
                if !rival.is_finite() {
                    if g2_run.sufficient.is_some_and(|z| z > 0.0) {
                        Comprehended::Settled
                    } else {
                        g2_run.unrivalled_tie = true;
                        Comprehended::Tie
                    }
                } else if lead_m > commit {
                    Comprehended::Settled
                } else {
                    Comprehended::Tie
                }

            }
        };
        // Phase I1 wiring (lane M's chain internals, 2026-11-19; `NANO_INTAKE_CHAIN=off`): the Core
        // Brain's intake reads G2's gate and evidence: each set piece with its fused value as
        // log-odds, the plan from the gate (settled: no hop), hops by the job map, each adding
        // calibrated evidence: the embedder's meaning read, the judge's close-call read (W4's
        // leader-only form: only the judge's read pieces carry it, and W4's leader keeps the first
        // place), the tentacles' read; stopping at the evidence bound or the call's budget. Its
        // order is served.
        if w2 && !std::env::var("NANO_INTAKE_CHAIN").is_ok_and(|v| v == "off") && ranked.len() >= 2 {
            let one = one_rc.borrow().clone();
            let key_of = |c: usize| sub.chunks.get(c).map_or_else(String::new, |ch| ch.key.clone());
            let pieces_ev: Vec<nmbrain::chain::PieceEvidence> = ranked
                .iter()
                .map(|c| nmbrain::chain::PieceEvidence { key: key_of(*c), prose: nmtype::is_document(&file_name(*c)), logodds: one.get(*c).copied().filter(|x| x.is_finite()).unwrap_or(0.0) })
                .collect();
            let meant_of = |c: usize| -> Option<f32> {
                let m = meaning.as_ref()?;
                let posed = m.first()?.get(c).copied().filter(|x| x.is_finite())?;
                let cf: Vec<f32> = m.iter().skip(1).filter_map(|v| v.get(c).copied()).filter(|x| x.is_finite()).collect();
                #[expect(clippy::cast_precision_loss, reason = "a count of forms")]
                let mean = (!cf.is_empty()).then(|| cf.iter().sum::<f32>() / cf.len() as f32)?;
                Some(posed - mean)
            };
            let emb_vals = |_: nmbrain::chain::Job, k: &str| chunk_of(k).and_then(meant_of);
            let emb_scale = crate::coordinator::noise_band(&ranked.iter().filter_map(|c| meant_of(*c)).collect::<Vec<f32>>());
            let judge_vals = |_: nmbrain::chain::Job, k: &str| chunk_of(k).and_then(|c| w4_z.get(&c).copied());
            let judge_scale = crate::coordinator::noise_band(&w4_z.values().copied().collect::<Vec<f32>>());
            let reads_now = read_all.borrow().clone();
            // The tentacles' integration read is one read: its evidence enters the intake once, at the
            // joint hop; the one-word hop has no read of its own here (both hops served the same
            // values, counting the read twice: answers first by the judge moved down at the intake
            // on 10 questions of tonight's run, lane T1 2026-11-19).
            let read_vals = |j: nmbrain::chain::Job, k: &str| (j == nmbrain::chain::Job::Joint).then(|| chunk_of(k).and_then(|c| reads_now.get(&c).copied())).flatten();
            let read_scale = crate::coordinator::noise_band(&reads_now.values().copied().collect::<Vec<f32>>());
            let mut emb = crate::brain_side::ValueHop { model: nmbrain::chain::MicroModel::Embedder, value: &emb_vals, scale: emb_scale };
            let mut jud = crate::brain_side::ValueHop { model: nmbrain::chain::MicroModel::Judge, value: &judge_vals, scale: judge_scale };
            let mut rdr = crate::brain_side::ValueHop { model: nmbrain::chain::MicroModel::Reader, value: &read_vals, scale: read_scale };
            let deadline = Instant::now() + std::time::Duration::from_secs_f64((call_budget_s() - t0.elapsed().as_secs_f64()).max(0.0));
            // The close call's model is a learnable site (lane M's G3 learning): G3's promoted choice
            // when it has one, else the rule: the stronger MicroModel for the kind by the strengths
            // table in force (the measured routing record when one is valid, Stage 5 WP-C2; else the
            // published table: code to the embedder, prose to the judge).
            let rule_of = |kind: PieceKind| if chain_choice(kind, true) == Some(MicroModel::Judge) { "judge" } else { "embedder" };
            let priors = |job: nmbrain::chain::Job| -> Option<(nmbrain::chain::MicroModel, f64)> {
                let rule = match job {
                    nmbrain::chain::Job::CodeClose => rule_of(PieceKind::Code),
                    nmbrain::chain::Job::ProseClose => rule_of(PieceKind::Prose),
                    _ => return None,
                };
                let d = crate::g3::decide_site(nmmusc::priors::Site::IntakeChain, rule, false);
                (d != rule).then(|| (if d == "judge" { nmbrain::chain::MicroModel::Judge } else { nmbrain::chain::MicroModel::Embedder }, f64::MAX))
            };
            let ti = Instant::now();
            let intake_out = crate::brain_side::intake(gate.name(), pieces_ev, &mut [&mut emb, &mut jud, &mut rdr], &priors, None, deadline);
            g2_run.part_ms.2 += ti.elapsed().as_secs_f64() * 1000.0;
            if let Ok(intake) = intake_out {
                let mut order: Vec<usize> = intake.out.order.iter().filter_map(|i| ranked.get(*i).copied()).collect();
                if order.len() == ranked.len() {
                    if let Some(h) = w4_leader
                        && let Some(i) = order.iter().position(|c| *c == h)
                    {
                        let c = order.remove(i);
                        order.insert(0, c);
                    }
                    if let Some(h) = own_promoted
                        && Some(h) != w4_leader
                        && let Some(i) = order.iter().position(|c| *c == h)
                    {
                        let target = 1.min(order.len().saturating_sub(1));
                        if i > target {
                            let c = order.remove(i);
                            order.insert(target, c);
                        }
                    }
                    g2_run.lines.push(format!(
                        "INTAKE: gate {}, {} hops ({}), stop {:?}, bound {:.2}; first {}",
                        gate.name(),
                        intake.out.hops.len(),
                        intake.out.hops.iter().map(|h| format!("{}:{}", h.job.name(), h.model.map_or("none", nmbrain::chain::MicroModel::name))).collect::<Vec<_>>().join(" "),
                        intake.out.stop,
                        intake.bound,
                        if order.first() == ranked.first() { "kept" } else { "changed" }
                    ));
                    config_chain.borrow_mut().extend(intake.out.hops.iter().map(|h| format!("{}:{}", h.job.name(), h.model.map_or("none", nmbrain::chain::MicroModel::name))));
                    let rest: Vec<usize> = sent.iter().copied().filter(|c| !order.contains(c)).collect();
                    ranked = order;
                    sent = ranked.iter().copied().chain(rest).collect();
                }
            }
            lad("brain.intake", &ranked);
        }
        // A tie in G1's race: both sides named for a joint read in the Core Brain (JOINT).
        let ties: Vec<(usize, usize)> = verdict
            .as_ref()
            .map(|v| v.ties.iter().filter_map(|(_, a, b)| Some((chunk_of(a)?, chunk_of(b)?))).collect())
            .unwrap_or_default();
        for (a, b) in &ties {
            let id = g2_run.motor.id();
            let keys: Vec<String> = [*a, *b].iter().filter_map(|c| span_of(*c)).map(|s| s.key).collect();
            if let Ok(m) = Motor::new(id, None, MessageKind::Instruction, Verb::Joint, nmbrain::motor::G2, "core-brain", 0.5, "tie: read both sides jointly", expiry)
                .and_then(|m| m.arg("pieces", Arg::Pieces(keys)))
                && g2_run.motor.ledger.record(&m).is_ok()
            {
                g2_run.motor.sent.push(m.goal(id));
                g2_run.edges.extend([("g2", "g1", Behavioural), ("g2", "core-brain", Behavioural)]);
                g2_run.joint.extend([*a, *b]);
            }
        }
        g2_run.edges.extend([("g1", "g2", Sensory), ("g2", "core-brain", Sensory)]);
        g2_run.class_refusals = usize::from(nmbrain::motor::topology(&g2_run.edges).is_err()) + usize::from(nmbrain::motor::topology_routed(&routed_edges.borrow()).is_err());

        g2_run.gate = gate.name().to_owned();
        g2_run.set = set.clone();
        // Phase J1: each G2 order with whether its read entered the set.
        for (c, why, _) in &ordered {
            let verb = if why.starts_with("meaning cluster") { "reshape-meaning" } else { "reshape" };
            let _ = crate::g3::note_site(nmmusc::sites::SiteKind::Order, "g2", "-", verb, Some(set.contains(c)));
            config_orders.borrow_mut().push((verb.to_owned(), set.contains(c)));
        }

        if w5 {
            g2_run.selected.clone_from(&selected);
        }
        lad("g2.ranked", &ranked);
        lad("g2.sent", &sent);
        lad_value("g2.gate", format!("{:?}", gate.name()));
        {
            let miss: Vec<f32> = (0..n_w).map(|a| 1.0 - last.covered.get(a).copied().unwrap_or(0.0)).collect();
            let gain = |c: usize| -> f32 { rw(c).iter().zip(&miss).map(|(x, m)| x * m).sum() };
            #[expect(clippy::cast_precision_loss, reason = "token counts")]
            let mut rv: Vec<(f32, usize)> = cands.iter().filter(|c| !set.contains(c)).map(|c| (gain(*c), *c)).filter(|g| g.0 > last.band).map(|(g, c)| (g / tk(c) as f32, c)).collect();
            rv.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            g2_run.rivals = rv.into_iter().map(|x| x.1).collect();
        }
        g2_run.absent = lacks;
        g2_run.value = (last.value, n_a + asks.len());
        g2_run.stop = last.stop.to_owned();
        g2_run.band = last.band;
        g2_run.room = room;
        g2_run.facts = (0..asks.len()).filter_map(|k| fact_of(k).map(|v| (k, v.to_vec()))).collect();
        g2_run.rel_pairs = cands.iter().flat_map(|q| rel_from(*q).into_iter().filter(|(p, _)| cands.contains(p)).map(move |(p, k)| (p, *q, k))).collect();
        g2_run.evaluated = first_evaluated + if g2_run.read > 0 { last.evaluated } else { 0 };
        // G2's own time: its selections, orders and gate; the tentacles' read on its orders is
        // theirs (G1 with the window reads), kept apart in `read_ms`.
        g2_run.ms = (t2.elapsed().as_secs_f64() * 1000.0 - g2_run.read_ms - g2_run.judge_ms).max(0.0);
        // Step J (tentacle master plan 7.5, C119): the close-call arm. On a close call (G2's gate
        // not settled, and the leader of what goes up stands within the noise band of the fused
        // evidence (`Evidence::settle_sum`, the plain sum of the drones' settle chances) over the
        // first `CLOSE_K` pieces, against the strongest of them) the judge MicroModel reads each of
        // those pieces cut to about 128 tokens, all in one packed card run, and its label log-odds
        // are added to each piece's fused evidence; the best by that sum takes the first place and
        // every other place keeps its order (`NANO_CLOSE_FORM=all`: all ten reordered, the first
        // form). The judge never replaces the evidence, and nothing enters or leaves what goes up.
        // Off unless `NANO_CLOSE=on`.
        // `NANO_CLOSE_PLACES` (lane G package 3, trace only): one `places:` line a call with G1's
        // hand-over, G2's set before the arm and after it (keys `path:first-last`), and the judge's
        // own card time, so the flow's losses can be put on a stage.
        let places_before: Option<Vec<usize>> = std::env::var_os("NANO_CLOSE_PLACES").map(|_| sent.clone());
        let mut judge_only_ms = 0.0_f64;
        if close_on() && gate != Comprehended::Settled {
            let top: Vec<usize> = sent.iter().take(CLOSE_K).copied().collect();
            let vals: Vec<f32> = top.iter().map(|c| ev.settle_sum(*c)).collect();
            let lead = vals.first().copied().unwrap_or(0.0) - vals.iter().skip(1).copied().fold(f32::NEG_INFINITY, f32::max);
            let band = crate::coordinator::noise_band(&vals);
            g2_run.close_lead = (lead, band);
            if top.len() >= 2 && lead <= band {
                let tc = Instant::now();
                if let Ok(m) = crate::neural::reranker_model(&nmtype::nano_root()) {
                    let docs: Vec<String> = top
                        .iter()
                        .map(|c| {
                            let ch = sub.chunks.get(*c);
                            let place = format!("{}:{}-{}", file_name(*c), ch.map_or(0, |x| x.start), ch.map_or(0, |x| x.end));
                            crate::neural::close_doc(&m, &place, ch.map_or("", |x| x.text.as_str()))
                        })
                        .collect();
                    let tj = Instant::now();
                    let judged = crate::cache_reads::judge_close(&m, question, &docs);
                    judge_only_ms = tj.elapsed().as_secs_f64() * 1000.0;
                    if let Ok(z) = judged {
                        let mut scored: Vec<(f32, usize, usize)> = top.iter().zip(&vals).zip(&z).enumerate().map(|(i, ((c, f), z))| (f + z, i, *c)).collect();
                        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                        if std::env::var("NANO_CLOSE_FORM").is_ok_and(|v| v == "all") {
                            // Step J's first form: all ten places reordered (top 3 fell, T-080).
                            for (slot, (_, _, c)) in sent.iter_mut().zip(scored) {
                                *slot = c;
                            }
                        } else if let Some((_, i, _)) = scored.first().copied() {
                            // The leader only (package 2): the best by fused evidence plus the
                            // judge goes first; every other place keeps its order.
                            let c = sent.remove(i);
                            sent.insert(0, c);
                        }
                        g2_run.close_fired = top.len();
                    }
                }
                g2_run.close_ms = tc.elapsed().as_secs_f64() * 1000.0;
            }
            if std::env::var_os("NANO_CLOSE_TRACE").is_some() {
                eprintln!("close: gate {} lead {lead:.4} band {band:.4} fired {} ms {:.0}", gate.name(), g2_run.close_fired, g2_run.close_ms);
            }
        } else if close_on() && std::env::var_os("NANO_CLOSE_TRACE").is_some() {
            eprintln!("close: gate {} fired 0", gate.name());
        }
        if let Some(before) = places_before {
            let keys = |l: &[usize], n: usize| -> String {
                l.iter().take(n).map(|c| sub.chunks.get(*c).map_or_else(String::new, |x| format!("{}:{}-{}", file_name(*c), x.start, x.end))).collect::<Vec<_>>().join("|")
            };
            eprintln!(
                "places: fired {} judge_ms {judge_only_ms:.0} close_ms {:.0}	g1 {}	g2 {}	up {}",
                g2_run.close_fired,
                g2_run.close_ms,
                keys(&handover, 40),
                keys(&before, 40),
                keys(&sent, 40)
            );
        }
        lad("close", &sent);
        g2_up = Some(sent);
        g2_cands = cands;
    }

    // Every round's reads, flattened in order: one entry per window read.
    let windows: Vec<Vec<usize>> = rounds.iter().flat_map(|(rs, _)| rs.iter().map(|r| r.window.clone())).collect();
    let taken: Vec<Vec<Vec<usize>>> = rounds.iter().map(|(rs, _)| rs.iter().map(|r| r.window.clone()).collect()).collect();
    let labels: Vec<String> = rounds.iter().flat_map(|(rs, _)| rs.iter().map(|r| r.label.clone())).collect();
    let owners: Vec<usize> = rounds.iter().flat_map(|(rs, _)| rs.iter().map(|r| r.tentacle)).collect();
    let depths: Vec<usize> = rounds.iter().flat_map(|(_, r)| r.depths.clone()).collect();
    let orders: Vec<Vec<usize>> = rounds.iter().flat_map(|(_, r)| r.orders.clone()).collect();
    let mut seen = std::collections::BTreeSet::new();
    let overlap = windows.iter().flatten().filter(|c| !seen.insert(**c)).count();
    let unconserved = rounds.iter().map(|(_, r)| r.unconserved).sum();
    let solo = rounds.iter().map(|(_, r)| r.solo).sum();
    let aspects_held = calibrated.as_ref().map_or((0, 0), |ev| {
        let held = (0..ev.aspects.len()).filter(|a| windows.iter().flatten().any(|c| ev.holds(*a, *c))).count();
        (held, ev.aspects.len())
    });
    // G2's gate over G1's order, with its instructions (I4, I5, I7, I10, I11; I6 named).
    let tokens = |c: usize| tk(c);
    let file_of_c = |c: usize| file_name(c);
    let related = |c: usize, r: nmbrain::instruct::Relation| -> Vec<usize> {
        use nmbrain::instruct::Relation as R;
        graph
            .neighbours(c)
            .into_iter()
            .filter(|(_, rel, _)| match r {
                R::Callers => *rel == nmsubs::refgraph::Relation::UsedBy,
                R::Callees | R::Definitions | R::Imports => *rel == nmsubs::refgraph::Relation::Uses,
                R::Any => true,
            })
            .map(|(j, _, _)| j)
            .collect()
    };
    // With G2's selection, what goes up is its chosen set (no fixed count); else G1's order, each
    // piece that fits the receiving room in tokens.
    let held_n = windows.iter().map(Vec::len).sum::<usize>();
    let mean = windows.iter().flatten().map(|c| tk(*c)).sum::<usize>().checked_div(held_n).unwrap_or(1);
    let (gate_hand, gate_queue, gate_room) = match &g2_up {
        Some(u) => (u.clone(), handover.iter().chain(&queue).filter(|c| !u.contains(c)).copied().collect::<Vec<usize>>(), u.iter().map(|c| tk(*c)).sum()),
        None => (handover.clone(), queue.clone(), room_of(mean)),
    };
    let gate_in =
        nmbrain::instruct::Gate { handover: &gate_hand, queue: &gate_queue, windows: &windows, room: gate_room, file_of: &file_of_c, tokens: &tokens, related: &related };
    let mut routed = nmbrain::instruct::route(&gate_in, &g2.orders, &g2.inhibited);
    routed.done.extend(g2_run.lines.iter().map(|l| format!("G2 {l}")));
    if !g2_run.joint.is_empty() {
        routed.joint.clone_from(&g2_run.joint);
    }
    // Reflexes that fired: each tentacle's on its last report, G1's per round.
    let mut absent = Vec::new();
    let mut no_information = Vec::new();
    for (_, r) in &rounds {
        for rep in &r.outcome.last {
            routed.done.extend(rep.reflexes.iter().map(|x| format!("{}: {x}", rep.from)));
            absent.push(rep.absent.clone());
            no_information.push(rep.no_information);
        }
        for l in &r.outcome.log {
            routed.done.extend(l.reflexes.iter().map(|x| format!("round {}: {x}", l.round)));
        }
    }
    live_stage.count("selected", g2_run.set.len());
    live_stage.next("G3", "");
    let tp = Instant::now();
    // With items on and G2's selection run, belief, the set and the pack are per item
    // (`item_stage`); else the pieces' pack.
    let item_ev = (crate::drone_windows::items_mode() == crate::drone_windows::ItemsMode::On && g2_up.is_some())
        .then(|| keep.item_evidence(question, meaning.as_deref()).ok())
        .flatten()
        .filter(|e| e.len() == keep.items().items.len() && !e.is_empty());
    let mut pack_pieces: Vec<Piece> = Vec::new();
    let pack = calibrated.as_ref().map_or_else(String::new, |ev| {
        let request = nmpod::intake::Request::read(question, &[]);
        let held: Vec<usize> = windows.iter().flatten().copied().collect();
        let room = room_of(held.iter().map(|c| tk(*c)).sum::<usize>().checked_div(held.len()).unwrap_or(1));
        match &item_ev {
            Some(iev) => {
                let (pack, chosen) = item_stage(sub, keep, question, iev, &request, &g2_cands, &routed.sent, &graph.specific, room);
                if std::env::var("NANO_ITEMS_DEBUG").is_ok() {
                    eprintln!("items pack: {question}\n{pack}");
                }
                g2_run.items = chosen;
                pack
            }
            None => {
                // The status line from G2's own selection (W5: the gate's meaning kept, decoys 39 to
                // 37 otherwise); the served pieces from the set that is served (W13).
                let pack = write_pack(sub, keep, ev, &request, &routed.sent, if w5 { &g2_run.selected } else { &g2_run.set }, &graph.specific, &g2_run.rivals, room);
                if !std::env::var("NANO_PACK_SERVE").is_ok_and(|v| v == "off") {
                    pack_pieces = write_pack_served(sub, keep, ev, &request, &routed.sent, &g2_run.set, &graph.specific, &g2_run.rivals, room).1;
                }
                pack

            }

        }
    });
    let pack_s = tp.elapsed().as_secs_f64();
    g2_run.refused_twin = g1.map.refused_twin;
    // The pack's confidence signals, for the flow's three states.
    if let Some(ev) = calibrated.as_ref() {
        let top = routed.sent.first().copied();
        let held = pack
            .lines()
            .next()
            .and_then(|l| l.split_whitespace().skip_while(|w| *w != "confidence").nth(1))
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.0);
        #[expect(clippy::cast_precision_loss, reason = "a place in a short list")]
        let lane_at = top.and_then(|c| lane.iter().position(|x| *x == c)).map_or(0.0, |p| 1.0 / (1.0 + p as f32));
        let defines = top.is_some_and(|c| {
            let doc = nmtype::is_document(&file_name(c));
            (0..ev.parts.len()).any(|k| {
                ev.part_holds(k, c)
                    && keep.record(c).is_some_and(|rec| {
                        let a = ev.parts.get(k).map_or(0, |p| p.0);
                        matches!(nmpod::evidence::tag(rec, ev.stems.get(a).map_or(&[][..], Vec::as_slice), doc), nmpod::evidence::Tag::Defines | nmpod::evidence::Tag::Documents)
                    })
            })
        });
        g2_run.signals = PackSignals { held, margin: g2_run.margin, lane: lane_at, defines: if defines { 1.0 } else { 0.0 }, sufficient: g2_run.sufficient };

    }
    routed.done.extend(read_lines.into_inner());
    if let Some(path) = &ladder {
        use std::io::Write;
        lad("up", &routed.sent);
        lad("served", &routed.sent.iter().copied().filter(|c| g2_run.set.contains(c)).collect::<Vec<_>>());
        let line = format!(
            "{{\"question\":{:?},\"read_on\":{read_on},{},\"windows\":[{}]}}\n",
            question,
            ladder_stages.borrow().join(","),
            ladder_windows.borrow().join(",")
        );
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = f.write_all(line.as_bytes());
        }
    }
    // Phase J: the call's configuration record for G3 (labels only; inert with G3 off).
    {
        let req = nmpod::intake::Request::read(question, &[]);
        let label = |s: String| -> String { s.chars().map(|c| if c.is_ascii_alphanumeric() || "-_.:+#".contains(c) { c.to_ascii_lowercase() } else { '-' }).take(40).collect() };
        let cfg = nmmusc::config::CallConfig {
            call: format!("call-{}", nmtype::now_ms()),
            kind: label(format!("{:?}", req.kind)),
            aspects: req.relations.iter().map(|a| label(a.name().to_owned())).collect(),
            lenses: config_lenses.borrow().clone(),
            orders: config_orders.borrow().clone(),
            chain: config_chain.borrow().iter().map(|c| label(c.clone())).collect(),
            cartridges: Vec::new(),
            budget_tokens: up.map_or(PACK_TOKENS, |n| n * 300),
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a positive budget in ms")]
            budget_ms: (call_budget_s() * 1000.0) as u64,
            outcome: None,
        };
        let _ = crate::g3::note_config(cfg);
    }
    if std::env::var_os("NANO_STAGE_TIMES").is_some() {

        eprintln!(
            "STAGE_TIMES sense {sense_s:.3} compose {compose_s:.3} g1 {g1_s:.3} pack {pack_s:.3} total {:.3} rounds {}; deep {:.3} (fallbacks {}) read {:.3}; g2 own {:.0} ms, order reads {:.0} ms, duels judge {:.0} ms, close {:.0} ms; g2 parts: sufficiency {:.0} ms, own picks {:.0} ms, intake {:.0} ms",
            t0.elapsed().as_secs_f64(),
            g1_rounds,
            deep_spent.get(),
            deep_fallbacks.get(),
            read_spent.get(),
            g2_run.ms,
            g2_run.read_ms,
            g2_run.judge_ms,
            g2_run.close_ms,
            g2_run.part_ms.0,
            g2_run.part_ms.1,
            g2_run.part_ms.2
        );

    }
    drop(live_stage);
    // Roadmap Phase 3 (2026-11-24): every call's stage times (ms) in a ring of the last
    // `STAGE_RING` calls, read by the live file for the interface's stage-times view.
    record_stage_times(&[
        ("sense", sense_s * 1000.0),
        ("compose", compose_s * 1000.0),
        ("g1", g1_s * 1000.0),
        ("tentacle_read", read_spent.get() * 1000.0),
        ("deep_check", deep_spent.get() * 1000.0),
        ("g2_own", g2_run.ms),
        ("g2_order_reads", g2_run.read_ms),
        ("judge_duels", g2_run.judge_ms),
        ("judge_close", g2_run.close_ms),
        ("pack", pack_s * 1000.0),
        ("total", t0.elapsed().as_secs_f64() * 1000.0),
    ]);
    Ok(StackRun {
        pack,
        pack_s,
        flags,
        depths,
        orders,
        handover,
        queue: routed.queue,
        up: routed.sent,
        rounds: g1_rounds,
        stop,
        gate,
        sense_s,
        compose_s,
        g1_s,
        total_s: t0.elapsed().as_secs_f64(),
        g2: routed.done,
        joint: routed.joint,
        scheduled: routed.scheduled,
        absent,
        no_information,
        windows,
        arm: arm.name().to_owned(),
        labels,
        owners,
        taken,
        yields,
        aspects_held,
        overlap,
        unconserved,
        solo,
        verdict,
        g2run: g2_run,
        evidence: calibrated,
        cands: g2_cands,
        pack_pieces,
    })
}

static FACTS: std::sync::RwLock<bool> = std::sync::RwLock::new(true);
/// Set whether G2 reads the relations the code's structure settles as facts
/// (`nmpod::evidence::Evidence::facts`); on by default.
pub fn set_facts(on: bool) {
    if let Ok(mut g) = FACTS.write() {
        *g = on;
    }
}

/// Whether the stack places the code pieces of G2's set by the joint order (pieces that restate
/// keep G1's places): off, unless `NANO_STACK_JOINT=on` (measured 2026-11-17: first 92 to 96,
/// top 3 122 to 129 of 188, no fall, not significant; build plan "Documents and the surprise
/// ceiling"). Roadmap Phase 4 (2026-11-24): with W2 on the reorder does not run, so the switch
/// adds the `joint` kind to W2's value instead (`w2_kinds_from`).
#[must_use]
pub fn stack_joint_on() -> bool {
    std::env::var("NANO_STACK_JOINT").is_ok_and(|v| v == "on")
}

/// Whether the meaning lane's fused shortlist (`nmsubs::dense::DenseLane::ranks`, the pieces the
/// served plain path adds as candidates) heads G1's hand-over on the stack, so G2 chooses among
/// them and the room is filled from them first: on, unless `NANO_STACK_LANE=off` (measured
/// 2026-11-17 over eight tuning sets, 315 questions with answers: first 143 to 162, p 0.027; top 3 201
/// to 228, p below 0.001; nothing fell; build plan "WP10: one flow for every call").
#[must_use]
pub fn stack_lane_on() -> bool {
    !std::env::var("NANO_STACK_LANE").is_ok_and(|v| v == "off")
}

/// Whether the stack's evidence holds the meaning signal (each piece's cosine with the question,
/// calibrated against the content-free questions'): on, unless `NANO_STACK_MEANING=off`; the plain
/// path's `NANO_DENSE_SIGNAL` is a separate switch. Measured 2026-11-18 over eight tuning sets (315
/// questions with answers): pack 246 to 259, p 0.015; every range 216 to 229, p 0.029; nothing fell;
/// seconds a call in quiet timing within 3 percent of the stack without it.
#[must_use]
pub fn stack_meaning_on() -> bool {
    !std::env::var("NANO_STACK_MEANING").is_ok_and(|v| v == "off")
}

/// Whether G1's map, the Map Coordinator and G2's orders read the material's meaning clusters
/// (W7: on by default since 2026-11-19, `NANO_MEANING_MAP=off` restores).
fn meaning_map_on() -> bool {
    !std::env::var("NANO_MEANING_MAP").is_ok_and(|v| v == "off")
}

/// Pieces, best by the question's meaning, whose clusters G2's orders look at.
pub const CLUSTER_LOOK: usize = 20;
/// Cosine at or above which two pieces of one cluster are near-duplicates by meaning.
pub const TWIN_COS: f32 = 0.97;
/// k-means passes.
const MEANING_PASSES: usize = 6;

/// The material's meaning: each piece's cluster and near-duplicate group (`u32::MAX`: none or no
/// vector), the cluster count, and the material version and vectors held it was formed on.
#[derive(Debug, Default)]
pub struct MeaningMap {
    pub clusters: Vec<u32>,
    pub twins: Vec<u32>,
    pub k: usize,
    pub version: u64,
    pub vectors: usize,
}

static MEANING: std::sync::Mutex<Option<Arc<MeaningMap>>> = std::sync::Mutex::new(None);

/// The material's meaning clusters from the embedder's stored vectors (processor only, in RAM,
/// once per material version and count of vectors held): spherical k-means (Lloyd's passes on
/// unit vectors; k the square root of the pieces with a vector, 8 to 128; seeds by farthest
/// point from the first piece, so the same material gives the same clusters), then within each
/// cluster the near-duplicate groups (cosine at least `TWIN_COS` with the group's first piece).
/// `None` with fewer than 16 vectors.
#[must_use]
pub fn meaning_map(sub: &Substrate, lane: &nmsubs::dense::DenseLane) -> Option<Arc<MeaningMap>> {
    let held: Vec<(usize, Arc<Vec<f32>>)> = sub.chunks.iter().enumerate().filter_map(|(i, c)| lane.vector(&c.key).map(|v| (i, v))).collect();
    if let Ok(g) = MEANING.lock()
        && let Some(m) = g.as_ref().filter(|m| m.version == sub.version && m.vectors == held.len())
    {
        return Some(Arc::clone(m));
    }
    if held.len() < 16 {
        return None;
    }
    let unit = |v: &[f32]| -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        v.iter().map(|x| x / n).collect()
    };
    let vs: Vec<Vec<f32>> = held.iter().map(|(_, v)| unit(v)).collect();
    let dim = vs.first().map_or(0, Vec::len);
    if dim == 0 || vs.iter().any(|v| v.len() != dim) {
        return None;
    }
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss, reason = "a square root of a count")]
    let k = ((vs.len() as f64).sqrt() as usize).clamp(8, 128);
    // Seeds: farthest point (least best cosine to the seeds so far), from the first piece.
    let mut seeds: Vec<Vec<f32>> = vec![vs.first()?.clone()];
    let mut best: Vec<f32> = vs.iter().map(|v| dot(v, &seeds[0])).collect();
    while seeds.len() < k {
        let (far, _) = best.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1).then(a.0.cmp(&b.0)))?;
        let s = vs.get(far)?.clone();
        for (b, v) in best.iter_mut().zip(&vs) {
            *b = b.max(dot(v, &s));
        }
        seeds.push(s);
    }
    let threads = std::thread::available_parallelism().map_or(1, usize::from).clamp(1, 4);
    let assign = |cents: &[Vec<f32>]| -> Vec<usize> {
        let part = vs.len().div_ceil(threads);
        std::thread::scope(|sc| {
            let hs: Vec<_> = vs
                .chunks(part.max(1))
                .map(|block| {
                    sc.spawn(move || {
                        block
                            .iter()
                            .map(|v| cents.iter().enumerate().map(|(j, c)| (dot(v, c), j)).max_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1))).map_or(0, |x| x.1))
                            .collect::<Vec<usize>>()
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
        })
    };
    let mut cents = seeds;
    let mut of = assign(&cents);
    for _ in 0..MEANING_PASSES {
        let mut sum = vec![vec![0.0_f32; dim]; k];
        for (v, j) in vs.iter().zip(&of) {
            if let Some(s) = sum.get_mut(*j) {
                for (a, x) in s.iter_mut().zip(v) {
                    *a += x;
                }
            }
        }
        for (c, s) in cents.iter_mut().zip(sum) {
            if s.iter().any(|x| *x != 0.0) {
                *c = unit(&s);
            }
        }
        let next = assign(&cents);
        if next == of {
            break;
        }
        of = next;
    }
    let mut clusters = vec![u32::MAX; sub.chunks.len()];
    let mut twins = vec![u32::MAX; sub.chunks.len()];
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (n, ((i, _), j)) in held.iter().zip(&of).enumerate() {
        if let Some(c) = clusters.get_mut(*i) {
            *c = u32::try_from(*j).unwrap_or(u32::MAX);
        }
        if let Some(m) = members.get_mut(*j) {
            m.push(n);
        }
    }
    for m in &members {
        let mut firsts: Vec<usize> = Vec::new();
        for &n in m {
            let Some(v) = vs.get(n) else { continue };
            match firsts.iter().copied().find(|f| vs.get(*f).is_some_and(|w| dot(v, w) >= TWIN_COS)) {
                Some(f) => {
                    let (Some((fi, _)), Some((ni, _))) = (held.get(f), held.get(n)) else { continue };
                    let g = u32::try_from(*fi).unwrap_or(u32::MAX);
                    for x in [*fi, *ni] {
                        if let Some(t) = twins.get_mut(x) {
                            *t = g;
                        }
                    }
                }
                None => firsts.push(n),
            }
        }
    }
    let m = Arc::new(MeaningMap { clusters, twins, k, version: sub.version, vectors: held.len() });
    if let Ok(mut g) = MEANING.lock() {
        *g = Some(Arc::clone(&m));
    }
    Some(m)
}

/// Whether G2's close calls are routed by measured strength (on by default since 2026-11-18, the
/// embedder half: eight sets level, tokens down 1.7 percent; `NANO_ROUTE=off` turns it off).
fn route_on() -> bool {
    !std::env::var("NANO_ROUTE").is_ok_and(|v| v == "off")
}

/// Whether the routed judge half (pairs in both orders, the joint read on ties) serves when the
/// judge is on (`NANO_ROUTE_JUDGE=on`; off by default, not yet measured on the card).
fn route_judge_on() -> bool {
    std::env::var("NANO_ROUTE_JUDGE").is_ok_and(|v| v == "on")
}

/// Whether a piece read on G2's orders or follow-up may swap in for the set's weakest piece
/// (W7: on by default since 2026-11-19, `NANO_SWAP_IN=off` restores).
fn swap_in_on() -> bool {
    !std::env::var("NANO_SWAP_IN").is_ok_and(|v| v == "off")
}

/// The kind of a close call: both pieces code, or at least one prose (a document or note).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieceKind {
    Code,
    Prose,
}

/// The MicroModels a close call can be routed to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MicroModel {
    Embedder,
    Judge,
}

/// The strengths table the chain choice reads: each MicroModel's measured strength on each kind
/// of close call. Seeded from its makers' published table for the released weights
/// (Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B reranking its top 100: MTEB-Code 75.41 against
/// 73.42, MLDR 50.26 against 67.28), the Nano's own C73 to C79 result seen from outside; to be
/// replaced by the Nano's own duel measurements when they exist. `NANO_STRENGTHS` names a file
/// of lines `embedder|judge code|prose <strength>` that replaces any row it gives.
pub const STRENGTHS: [(MicroModel, PieceKind, f32); 4] =
    [(MicroModel::Embedder, PieceKind::Code, 75.41), (MicroModel::Judge, PieceKind::Code, 73.42), (MicroModel::Embedder, PieceKind::Prose, 50.26), (MicroModel::Judge, PieceKind::Prose, 67.28)];

/// A dual-strong duel's embedder evidence this decisive (calibrated log-odds, ln 9: 90 percent) is
/// decided by the embedder alone, without the judge's read (Stage 5 WP-C2).
const COMBINE_DECISIVE: f32 = 2.197_224_6;

/// The fewest fit duels a measured routing row needs to replace a `STRENGTHS` row.
const RECORD_MIN_DUELS: u64 = 20;

/// The measured routing record in force (Stage 5 WP-C2): `designs/routing-record.json` when it was
/// measured on the base cartridges now serving (`designs/base-cartridges.json`), read once per
/// process; `None` (the published table) without it or with `NANO_ROUTING_RECORD=off`.
pub fn measured_record() -> Option<&'static nmbrain::chain::RoutingRecord> {
    static REC: std::sync::OnceLock<Option<nmbrain::chain::RoutingRecord>> = std::sync::OnceLock::new();
    REC.get_or_init(|| {
        let home = nmtype::nano_root();
        // 2026-11-25: the `routing.record` control (interface, MCP) sets the record aside too.
        if std::env::var("NANO_ROUTING_RECORD").is_ok_and(|v| v == "off") || crate::whole::control_str(&home, "routing.record").is_some_and(|v| v == "off") {
            return None;
        }
        let rec = nmbrain::chain::RoutingRecord::load(&home.join("designs").join("routing-record.json"))?;
        let served = nmjson::parse(&std::fs::read(home.join("designs").join("base-cartridges.json")).ok()?).ok()?;
        let serving: Vec<(String, String)> = served
            .get("serving")
            .ok()?
            .object()
            .ok()?
            .iter()
            .filter_map(|(k, v)| v.get("sha256").ok().and_then(|s| s.string().ok()).map(|s| (k.clone(), s.to_owned())))
            .collect();
        // Roadmap Phase 0 item 4 (2026-11-23): the record holds only for the lenses its duels were
        // read with (a list; `none` among them for projects without one).
        let lens_key = crate::distil::lens_key();
        (rec.valid_for(&serving) && rec.lens.split(',').any(|l| l == lens_key)).then_some(rec)
    })
    .as_ref()
}

/// The strengths table in force: `STRENGTHS`, with each row the measured routing record measures
/// (on at least `RECORD_MIN_DUELS` fit duels) replaced by its measured accuracy, then the rows of
/// `NANO_STRENGTHS` laid over it.
#[must_use]
pub fn strengths() -> Vec<(MicroModel, PieceKind, f32)> {
    let mut t = STRENGTHS.to_vec();
    if let Some(rec) = measured_record() {
        for row in &mut t {
            let job = if row.1 == PieceKind::Code { nmbrain::chain::Job::CodeClose } else { nmbrain::chain::Job::ProseClose };
            let model = if row.0 == MicroModel::Embedder { nmbrain::chain::MicroModel::Embedder } else { nmbrain::chain::MicroModel::Judge };
            if let Some(m) = rec.row(job, model).filter(|m| m.n >= RECORD_MIN_DUELS) {
                row.2 = m.strength;
            }
        }
    }
    let Some(text) = std::env::var_os("NANO_STRENGTHS").and_then(|p| std::fs::read_to_string(p).ok()) else { return t };
    for l in text.lines() {
        let w: Vec<&str> = l.split_whitespace().collect();
        let (Some(m), Some(k), Some(v)) = (w.first(), w.get(1), w.get(2).and_then(|v| v.parse::<f32>().ok())) else { continue };
        let m = match *m {
            "embedder" => MicroModel::Embedder,
            "judge" => MicroModel::Judge,
            _ => continue,
        };
        let k = match *k {
            "code" => PieceKind::Code,
            "prose" => PieceKind::Prose,
            _ => continue,
        };
        if let Some(row) = t.iter_mut().find(|r| r.0 == m && r.1 == k) {
            row.2 = v;
        }
    }
    t
}

/// One routing control, read from `settings.json` at most once a second (close calls are many).
fn routing_choice(key: &str) -> Option<String> {
    type Cache = Option<(std::time::Instant, Vec<(String, Option<String>)>)>;
    static CACHE: std::sync::Mutex<Cache> = std::sync::Mutex::new(None);
    let mut g = CACHE.lock().ok()?;
    let fresh = g.as_ref().is_some_and(|(at, _)| at.elapsed() < std::time::Duration::from_secs(1));
    if !fresh {
        let home = nmtype::nano_root();
        let read = ["routing.code", "routing.prose"].iter().map(|k| ((*k).to_owned(), crate::whole::control_str(&home, k))).collect();
        *g = Some((std::time::Instant::now(), read));
    }
    g.as_ref().and_then(|(_, v)| v.iter().find(|(k, _)| k == key).and_then(|(_, x)| x.clone()))
}

/// G2's chain choice for a close call of `kind`: the MicroModel the strengths table names
/// strongest for it among those available (the judge only when `judge` is true).
#[must_use]
pub fn chain_choice(kind: PieceKind, judge: bool) -> Option<MicroModel> {
    // 2026-11-25 (directions of Luis Jake Gabriel III: routing customizable): the user's choice
    // for this kind of close call (`routing.code`, `routing.prose`) wins over the table; the judge
    // only where it runs.
    match routing_choice(if kind == PieceKind::Code { "routing.code" } else { "routing.prose" }).as_deref() {
        Some("embedder") => return Some(MicroModel::Embedder),
        Some("judge") if judge => return Some(MicroModel::Judge),
        _ => {}
    }
    strengths()
        .into_iter()
        .filter(|r| r.1 == kind && (judge || r.0 != MicroModel::Judge))
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .map(|r| r.0)
}

/// The instruction the judge reads a duel's pair under (pairwise ranking prompting).
pub const PAIR_INSTRUCTION: &str = "Given a question about a code repository and two passages, judge whether Passage A answers the question better than Passage B";

/// One duel for `judge_pairs`: the pair's content key, the two pieces' texts as read for the
/// question, and their request-free texts for the content-free reads.
pub struct PairDuel {
    pub key: String,
    pub a: String,
    pub b: String,
    pub plain_a: String,
    pub plain_b: String,
}

/// Each ordered pair's content-free label (mean log-odds over `neural::CONTENT_FREE`), by pair
/// key and order, held in RAM for the process.
static PAIR_NULL: std::sync::Mutex<BTreeMap<String, f32>> = std::sync::Mutex::new(BTreeMap::new());

/// Pairs (of the two orders of the pair `ka|kb`) whose content-free label is not yet held.
fn pair_unheld(ka: &str, kb: &str) -> usize {
    PAIR_NULL.lock().map_or(2, |h| [format!("{ka}|{kb}"), format!("{kb}|{ka}")].iter().filter(|k| !h.contains_key(*k)).count())
}

/// The text the judge reads for one order of a duel.
fn pair_doc(a: &str, b: &str) -> String {
    format!("Passage A:\n{a}\n\nPassage B:\n{b}")
}

/// The judge's duels in both orders in one packed read (PRP): for each duel, the label log-odds
/// that A is the better with the pair as given and with it swapped, each less that order's
/// content-free label (read once per ordered pair, then held). Above 0 in the first and below 0 in
/// the second, the first piece wins; the reverse, the second; anything else is a split verdict.
/// Returns each duel's two values and the prompt tokens read.
/// # Errors
/// Tokenizer or engine failure.
pub fn judge_pairs(m: &nmengn::model::Model, question: &str, duels: &[PairDuel]) -> Result<(Vec<(f32, f32)>, usize), Fault> {
    // Roadmap Phase 4.5 (2026-11-24): the close-duel judge in the live view.
    let mut live = crate::live::stage("G2.duel", crate::live::role_of(&m.pop.card.id));
    live.count("duels", duels.len());
    let forms = crate::neural::CONTENT_FREE;
    let docs: Vec<[String; 2]> = duels.iter().map(|d| [pair_doc(&d.a, &d.b), pair_doc(&d.b, &d.a)]).collect();
    let plains: Vec<[String; 2]> = duels.iter().map(|d| [pair_doc(&d.plain_a, &d.plain_b), pair_doc(&d.plain_b, &d.plain_a)]).collect();
    let keys: Vec<[String; 2]> = duels
        .iter()
        .map(|d| {
            let (a, b) = d.key.split_once('|').unwrap_or((d.key.as_str(), ""));
            [format!("{a}|{b}"), format!("{b}|{a}")]
        })
        .collect();
    let held: Vec<[Option<f32>; 2]> = {
        let h = PAIR_NULL.lock().map_err(|_| nmtype::Fault::new("nmapi.drone_path", nmtype::FaultKind::Execution, "pair labels: lock poisoned"))?;
        keys.iter().map(|k| [h.get(&k[0]).copied(), h.get(&k[1]).copied()]).collect()
    };
    let mut pairs: Vec<(&str, &str)> = docs.iter().flat_map(|d| [(question, d[0].as_str()), (question, d[1].as_str())]).collect();
    for (p, h) in plains.iter().zip(&held) {
        for o in 0..2 {
            if h[o].is_none() {
                pairs.extend(forms.iter().map(|f| (*f, p[o].as_str())));
            }
        }
    }
    let dev = if m.pop.gpu.is_some() { nmengn::placement::Device::Gpu } else { nmengn::placement::Device::Cpu };
    let tokens = pairs.iter().map(|(q, d)| m.pop.tokenizer.encode(&nmengn::roles::card_prompt(PAIR_INSTRUCTION, q, d)).map_or(0, |v| v.len())).sum();
    let p = nmengn::roles::card_rerank_pairs(m, dev, PAIR_INSTRUCTION, &pairs)?;
    let mut fresh = p.get(2 * duels.len()..).unwrap_or(&[]).chunks(forms.len());
    let mut h = PAIR_NULL.lock().map_err(|_| nmtype::Fault::new("nmapi.drone_path", nmtype::FaultKind::Execution, "pair labels: lock poisoned"))?;
    let mut out = Vec::with_capacity(duels.len());
    for (i, (k, was)) in keys.iter().zip(&held).enumerate() {
        let mut v = [0.0_f32; 2];
        for o in 0..2 {
            let null = match was[o] {
                Some(n) => n,
                None => {
                    #[expect(clippy::cast_precision_loss, reason = "a count of forms")]
                    let n = fresh.next().unwrap_or(&[]).iter().map(|x| logit(*x)).sum::<f32>() / forms.len() as f32;
                    h.insert(k[o].clone(), n);
                    n
                }
            };
            v[o] = logit(p.get(2 * i + o).copied().unwrap_or(0.5)) - null;
        }
        out.push((v[0], v[1]));
    }
    Ok((out, tokens))
}

/// Whether a duel that does not fit the time left is skipped rather than ending the planning
/// (`NANO_JUDGE_SKIP=on`; off by default).
fn judge_skip() -> bool {
    std::env::var("NANO_JUDGE_SKIP").is_ok_and(|v| v == "on")
}

/// Whether the judge reads, after a call, the content-free labels of that call's entrants and
/// holders it did not read (`NANO_JUDGE_IDLE=on`; off by default).
fn judge_idle() -> bool {
    std::env::var("NANO_JUDGE_IDLE").is_ok_and(|v| v == "on")
}

/// Whether G2 may send one follow-up read on a low margin (W7: on by default since 2026-11-19,
/// `NANO_FOLLOW=off` restores).
/// Whether the call's time is spent by expected value (homeostasis, on by default since 2026-11-19;
/// `NANO_HOMEOSTASIS=off` restores the earlier fixed rules).
fn homeostasis_on() -> bool {
    !std::env::var("NANO_HOMEOSTASIS").is_ok_and(|v| v == "off")
}
fn follow_on() -> bool {

    !std::env::var("NANO_FOLLOW").is_ok_and(|v| v == "off")
}


/// The follow-up read's margin band (`NANO_FOLLOW_BAND=<log ratio>`, else `FOLLOW_BAND`).
fn follow_band() -> f32 {
    std::env::var("NANO_FOLLOW_BAND").ok().and_then(|v| v.parse().ok()).unwrap_or(FOLLOW_BAND)
}

/// Whether `low` holds `pat` as whole words: not followed by a name character, and (with `start`)
/// not preceded by one.
fn names_at(low: &str, pat: &str, start: bool) -> bool {
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    low.match_indices(pat).any(|(i, _)| {
        let after = low.get(i + pat.len()..).and_then(|r| r.chars().next());
        let before = low.get(..i).and_then(|r| r.chars().last());
        (pat.ends_with('(') || after.is_none_or(|c| !ident(c))) && (!start || before.is_none_or(|c| !ident(c)))
    })
}

/// The names a piece's lines call (`name(`, not its definition, no control words) and the names
/// they define (`fn`, `def`, `function`), each of three characters or more, once each, lower case.
#[must_use]
pub fn anchor_names(low: &str) -> (Vec<String>, Vec<String>) {
    const CONTROL: [&str; 16] = ["if", "for", "while", "match", "return", "some", "ok", "err", "vec", "box", "print", "println", "format", "assert", "assert_eq", "len"];
    const DEF: [&str; 3] = ["fn ", "def ", "function "];
    let ident = |ch: char| ch.is_alphanumeric() || ch == '_';
    let (mut calls, mut defs): (Vec<String>, Vec<String>) = (Vec::new(), Vec::new());
    for (i, _) in low.match_indices('(') {
        let before = low.get(..i).unwrap_or("");
        let name: String = before.chars().rev().take_while(|c| ident(*c)).collect::<Vec<char>>().into_iter().rev().collect();
        let rest = before.get(..before.len() - name.len()).unwrap_or("");
        if name.chars().count() < 3 || name.starts_with(|c: char| c.is_ascii_digit()) || CONTROL.contains(&name.as_str()) || rest.ends_with('!') {
            continue;
        }
        let list = if DEF.iter().any(|d| rest.ends_with(d)) { &mut defs } else { &mut calls };
        if !list.contains(&name) {
            list.push(name);
        }
    }
    calls.retain(|n| !defs.contains(n));
    (calls, defs)
}

/// Whether the meaning lane's first fused place always enters G2's set: on, unless
/// `NANO_STACK_LEADER=off` (measured 2026-11-18 over eight tuning sets, 315 questions with answers:
/// pack 259 to 268, p 0.023; every range 229 to 238, p 0.004; nothing fell).
#[must_use]
pub fn stack_leader_on() -> bool {
    !std::env::var("NANO_STACK_LEADER").is_ok_and(|v| v == "off")
}

/// The meaning lane's fused shortlist for `question`, best place first; empty without the lane.
fn lane_shortlist(sub: &Substrate, question: &str) -> Vec<usize> {
    let Some(r) = sub.dense.read().ok().and_then(|d| d.clone()).and_then(|l| l.ranks(sub, question)) else {
        return Vec::new();
    };
    let mut places: Vec<(usize, usize)> = r.place.iter().map(|(c, p)| (*p, *c)).collect();
    places.sort_unstable();
    places.into_iter().map(|(_, c)| c).collect()
}

/// W2: one fused value per piece (on by default since 2026-11-18, C129; `NANO_W2=off` restores the
/// lane splice and the per-stage orders): the kinds named in `NANO_W2_KINDS`
/// (default wording, meaning, settle, lane, drone; `joint` with `NANO_STACK_JOINT=on`), each
/// standardised over the request's leading pieces
/// and clipped to `W2_CLIP` standard units, summed; a missing value counts 0.
#[must_use]
pub fn w2_on() -> bool {
    !std::env::var("NANO_W2").is_ok_and(|v| v == "off")
}
fn w2_kinds() -> Vec<String> {
    w2_kinds_from(std::env::var("NANO_W2_KINDS").ok().as_deref(), stack_joint_on())
}
/// W2's kinds from the named list (`None`: the default list) and the joint switch.
/// Roadmap Phase 4 (2026-11-24): `NANO_STACK_JOINT=on` adds the `joint` kind (each piece's place
/// in the joint order), so the part acts under W2, where its own reorder of G2's set does not run;
/// naming `joint` in `NANO_W2_KINDS` does the same. Not in the default list until measured.
fn w2_kinds_from(named: Option<&str>, joint: bool) -> Vec<String> {
    let mut kinds: Vec<String> = named.unwrap_or("wording,meaning,settle,lane,drone").split(',').map(|k| k.trim().to_owned()).collect();
    if joint && !kinds.iter().any(|k| k == "joint") {
        kinds.push("joint".to_owned());
    }
    kinds
}
/// The joint read's pieces of `chunks`, by chunk (the tentacle core's deep check reads them).
fn pieces_by_chunk(sub: &Substrate, chunks: &[usize]) -> BTreeMap<usize, Piece> {
    pieces(sub, chunks, &[]).into_iter().map(|p| (p.chunk, p)).collect()
}

/// W2's scale: the leading pieces by belief (about a team round's windows) the kinds are
/// standardised over; and the clip, so no one kind (the lane's few places) decides alone.
pub const W2_POOL: usize = 500;
/// C132: each tentacle's own best pieces kept for G2's set (the Mini's E3.3: own top 2).
pub const OWN_K: usize = 2;
/// G1's hand-over holds at least this many of the value's first pieces (the old race's usual count).
pub const HANDOVER_MIN: usize = 3;
/// C132: the tentacles' read-first finds G2's set may admit on one call.
pub const OWN_ADMIT: usize = 3;
/// The team's stop: the room's pieces whose last value an unread piece must beat to matter.
pub const STOP_ROOM: usize = 12;
/// C132: the tentacles' own picks the judge reads on one call (each tentacle's first two).
pub const OWN_READ: usize = 16;
/// C132: own picks the judge may let into G2's set on one call.
pub const OWN_IN: usize = 1;
/// Lane DR's drone kind in the flow: pieces read a call, and the longest piece read (tokens).
pub const DRONE_READ: usize = 8;
pub const DRONE_TOKENS: usize = 600;
/// Phase G1-a: the share of a runner's accumulated evidence kept per round it is not re-read.
pub const G1_FADE: f32 = 0.5;
/// Phase F: seeds of a frontier, scanning or recall walk (the best few, a window's start).
pub const SCAN_SEEDS: usize = 5;
/// Phase F: personalised PageRank's power steps (0.5 damping halves the rest each step).
pub const PPR_STEPS: usize = 12;
/// W4's gate (second form): the fused leader's margin over the next at most this share of the
/// noise band of the first `CLOSE_K` (stated from the W4 run's traced margins, 2026-11-18: 141 of
/// 315 answered questions fire, a minority, against 224 at the whole band).
pub const W4_GATE_K: f32 = 0.25;
pub const W2_CLIP: f32 = 3.0;

/// Live scaling weights fed from `nmlearn::engine::RetrievalAdaptation` into W2 fusion.
/// When unadapted (all defaults), every scale is exactly `1.0` so baseline ranking is unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetrievalWeights {
    pub lexical: f32,
    pub semantic: f32,
    pub structural: f32,
    pub relationship: f32,
}
impl Default for RetrievalWeights {
    fn default() -> Self {
        Self { lexical: 1.0, semantic: 1.0, structural: 1.0, relationship: 1.0 }
    }
}
impl RetrievalWeights {
    /// Converts `nmlearn::engine::RetrievalAdaptation` into bounded `[0.5, 2.0]` fusion scales
    /// normalized against `RetrievalAdaptation::default()`.
    #[must_use]
    pub fn from_adaptation(a: &nmlearn::engine::RetrievalAdaptation) -> Self {
        let d = nmlearn::engine::RetrievalAdaptation::default();
        let scale = |v: f64, base: f64| -> f32 {
            if !v.is_finite() || base <= 0.0 {
                return 1.0;
            }
            #[expect(clippy::cast_possible_truncation, reason = "clamped ratio in [0.5, 2.0]")]
            let r = (v / base).clamp(0.5, 2.0) as f32;
            r
        };
        Self {
            lexical: scale(a.lexical_weight, d.lexical_weight),
            semantic: scale(a.semantic_weight, d.semantic_weight),
            structural: scale(a.structural_weight, d.structural_weight),
            relationship: scale(a.relationship_weight, d.relationship_weight),
        }
    }
}

thread_local! {
    static ACTIVE_RETRIEVAL_WEIGHTS: std::cell::Cell<RetrievalWeights> = std::cell::Cell::new(RetrievalWeights::default());
}

/// Active retrieval fusion weights for the current call thread.
#[must_use]
pub fn active_retrieval_weights() -> RetrievalWeights {
    ACTIVE_RETRIEVAL_WEIGHTS.with(std::cell::Cell::get)
}

/// Runs `f` with `weights` installed for W2 fusion on the current thread, restoring the previous
/// weights on return or unwind.
pub fn with_retrieval_weights<T>(weights: RetrievalWeights, f: impl FnOnce() -> T) -> T {
    struct Guard(RetrievalWeights);
    impl Drop for Guard {
        fn drop(&mut self) {
            ACTIVE_RETRIEVAL_WEIGHTS.with(|c| c.set(self.0));
        }
    }
    let prev = ACTIVE_RETRIEVAL_WEIGHTS.with(|c| c.replace(weights));
    let _guard = Guard(prev);
    f()
}

/// The fused value of every piece: each column standardised (mean and deviation over `pool`,
/// finite values only), clipped to `W2_CLIP`, summed; a non-finite value adds 0.
#[must_use]
pub fn fuse_kinds(cols: &[Vec<f32>], pool: &[usize]) -> Vec<f32> {
    let weighted: Vec<(f32, Vec<f32>)> = cols.iter().cloned().map(|c| (1.0, c)).collect();
    fuse_kinds_weighted(&weighted, pool)
}

/// Weighted W2 fusion: each `(weight, column)` is standardised over `pool`, clipped, multiplied
/// by `weight`, and summed. When all weights are `1.0`, this matches `fuse_kinds` bit-for-bit.
#[must_use]
pub fn fuse_kinds_weighted(cols: &[(f32, Vec<f32>)], pool: &[usize]) -> Vec<f32> {
    let n = cols.first().map_or(0, |(_, c)| c.len());
    let mut out = vec![0.0_f32; n];
    for (w, col) in cols {
        let w = if w.is_finite() { w.clamp(0.0, 4.0) } else { 1.0 };
        if w <= 0.0 {
            continue;
        }
        let xs: Vec<f32> = pool.iter().filter_map(|c| col.get(*c).copied()).filter(|x| x.is_finite()).collect();
        if xs.len() < 2 {
            continue;
        }
        #[expect(clippy::cast_precision_loss, reason = "a pool's size")]
        let m = xs.iter().sum::<f32>() / xs.len() as f32;
        #[expect(clippy::cast_precision_loss, reason = "a pool's size")]
        let sd = (xs.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / xs.len() as f32).sqrt();
        if sd <= 0.0 {
            continue;
        }
        // A kind read on only part of the pool (the drones' depth read of the first pieces, the
        // writer's notes) lifts the pieces that stand out on it and lowers none: centred on its own
        // pieces, it would put a read piece below the mean under every unread one (the first card
        // smoke of the drone kind dropped answers from G2's set that way).
        // It lifts by at most one standard unit: a full three opened gaps at the top that kept the
        // close-call judge from firing (closure c20 and c23 lost the judge's first place).
        let partial = xs.len() * 2 < pool.len();
        let (lo, hi) = if partial { (0.0, 1.0) } else { (-W2_CLIP, W2_CLIP) };
        for (o, x) in out.iter_mut().zip(col) {
            if x.is_finite() {
                *o += w * ((x - m) / sd).clamp(lo, hi);
            }
        }
    }
    out
}

/// Where W4's close-call judge stands: on the card and ready, loading, or no card here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JudgeState {
    Card,
    Loading,
    NoCard,
}
static JUDGE_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
/// Starts the close-call judge's load in the background on the first call (serve calls it at start)
/// and says where it stands: `NoCard` when no card opens (the judge is then never loaded) or it
/// bound on the processor, `Loading` until it is on the card, then `Card`.
pub fn warm_close_judge() -> JudgeState {
    use std::sync::atomic::Ordering::SeqCst;
    if JUDGE_STATE.compare_exchange(0, 1, SeqCst, SeqCst).is_ok() {
        std::thread::spawn(|| {
            // Roadmap Phase 4.6 (2026-11-25): the process's one card (`nmengn::device`).
            let card = nmengn::device::open().is_ok();
            let on_card = card && crate::neural::reranker_model(&nmtype::nano_root()).is_ok_and(|m| m.pop.gpu.is_some());
            JUDGE_STATE.store(if on_card { 2 } else { 3 }, SeqCst);
        });
    }
    match JUDGE_STATE.load(SeqCst) {
        2 => JudgeState::Card,
        3 => JudgeState::NoCard,
        _ => JudgeState::Loading,
    }
}

static READER_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
/// As `warm_close_judge`, for the reader the tentacles' integration read runs on.
pub fn warm_reader() -> JudgeState {
    use std::sync::atomic::Ordering::SeqCst;
    if READER_STATE.compare_exchange(0, 1, SeqCst, SeqCst).is_ok() {
        std::thread::spawn(|| {
            // Roadmap Phase 4.6 (2026-11-25): the process's one card (`nmengn::device`).
            let card = nmengn::device::open().is_ok();
            let on_card = card && crate::neural::model(&nmtype::nano_root()).is_ok_and(|m| m.pop.gpu.is_some());
            READER_STATE.store(if on_card { 2 } else { 3 }, SeqCst);
        });
    }
    match READER_STATE.load(SeqCst) {
        2 => JudgeState::Card,
        3 => JudgeState::NoCard,
        _ => JudgeState::Loading,
    }
}

/// W5: G2's set by the fused value `one`: candidates in its order, each taken while it fits
/// `room` tokens; then every part (of `parts`) no set piece holds gets its best holder by the
/// value, and the weakest set pieces that are not the only holder of a part leave until the set
/// fits again.
#[must_use]
pub fn value_fill(cands: &[usize], one: &[f32], tk: &dyn Fn(usize) -> usize, room: usize, parts: usize, holds: &dyn Fn(usize, usize) -> bool) -> Vec<usize> {
    let v = |c: usize| one.get(c).copied().unwrap_or(f32::MIN);
    let mut order: Vec<usize> = cands.to_vec();
    order.sort_by(|a, b| v(*b).total_cmp(&v(*a)).then(a.cmp(b)));
    order.dedup();
    let mut set: Vec<usize> = Vec::new();
    let mut used = 0;
    let trace = std::env::var_os("NANO_FILL_TRACE").is_some();
    // Room counted fairly in tokens (lane T1, 2026-11-19; large code loc10): when a piece does not
    // fit, a longer piece past the value's first three whose value lies within the noise band of
    // it gives its room to the pieces left out within that band, if two or more of them fit in
    // it (two near-equal pieces for one; loc10's answer, 67 tokens short, stayed out while a
    // 540-token piece of the same value held its room). A piece that gave its room does not return.
    let band = crate::coordinator::noise_band(&order.iter().take(24).map(|c| v(*c)).collect::<Vec<f32>>());
    let mut gone: Vec<usize> = Vec::new();
    for c in &order {
        if set.contains(c) || gone.contains(c) {
            continue;
        }
        let take = used + tk(*c) <= room;
        if trace {
            eprintln!("FILL c{c} value {:.3} tokens {} used {used} room {room} band {band:.3} {}", v(*c), tk(*c), if take { "in" } else { "out" });
        }
        if take {
            used += tk(*c);
            set.push(*c);
            continue;
        }
        let Some(x) = set.iter().skip(3).copied().filter(|x| v(*x) - v(*c) <= band && tk(*x) > tk(*c)).max_by(|a, b| tk(*a).cmp(&tk(*b)).then(b.cmp(a))) else { continue };
        let mut left = room - (used - tk(x));
        let mut came: Vec<usize> = Vec::new();
        for y in order.iter().copied().filter(|y| !set.contains(y) && !gone.contains(y) && v(x) - v(*y) <= band && v(*y) <= v(*c)) {
            if tk(y) <= left {
                left -= tk(y);
                came.push(y);
            }
        }
        if came.len() >= 2 && came.contains(c) {
            if trace {
                eprintln!("FILL fair: c{x} ({} tokens) gives its room to {}", tk(x), came.iter().map(|y| format!("c{y}")).collect::<Vec<_>>().join(" "));
            }
            set.retain(|y| *y != x);
            gone.push(x);
            used = used - tk(x) + came.iter().map(|y| tk(*y)).sum::<usize>();
            set.extend(came);
        }
    }
    set.sort_by(|a, b| v(*b).total_cmp(&v(*a)).then(a.cmp(b)));
    for a in 0..parts {
        if set.iter().any(|c| holds(a, *c)) {
            continue;
        }
        let Some(h) = order.iter().copied().find(|c| holds(a, *c) && !set.contains(c)) else { continue };
        set.push(h);
        used += tk(h);
        while used > room && set.len() > 1 {
            let alone = |c: usize| (0..parts).any(|k| holds(k, c) && set.iter().filter(|x| holds(k, **x)).count() == 1);
            let Some(i) = set.iter().rposition(|c| *c != h && !alone(*c)) else { break };
            used -= tk(set.remove(i));
        }
    }
    set.sort_by(|a, b| v(*b).total_cmp(&v(*a)).then(a.cmp(b)));
    set
}

/// W2: G1's hand-over and queue in the fused value's order; the hand-over keeps its count.
fn one_forward(one: &[f32], handover: &mut Vec<usize>, queue: &mut Vec<usize>) {
    // G1 hands over at least the value's first `HANDOVER_MIN` pieces: the accumulating race with
    // its quorum by independence commits none on many calls, and the hand-over was then only the
    // tentacles' own picks (G1-T 14 to 12 of 30 on the dump: t15, t17).
    let n = handover.len().max(HANDOVER_MIN).min(handover.len() + queue.len());
    let mut pool: Vec<(f32, usize, usize)> = handover.iter().chain(queue.iter()).enumerate().map(|(i, c)| (one.get(*c).copied().unwrap_or(f32::MIN), i, *c)).collect();
    pool.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let order: Vec<usize> = pool.into_iter().map(|(_, _, c)| c).collect();
    *handover = order.iter().take(n).copied().collect();
    *queue = order.iter().skip(n).copied().collect();
}

/// `lane` at the head of G1's hand-over, in its order (taken out of the queue).
fn join_lane(handover: &mut Vec<usize>, queue: &mut Vec<usize>, lane: &[usize]) {
    if lane.is_empty() {
        return;
    }
    queue.retain(|c| !lane.contains(c));
    handover.retain(|c| !lane.contains(c));
    handover.splice(0..0, lane.iter().copied());
}

/// Pieces the close-call arm's judge reads: the first ten of what goes up (A3 T2's candidates).
pub const CLOSE_K: usize = 10;

/// Whether the close-call arm serves (step J, C119): off unless `NANO_CLOSE=on`.
#[must_use]
pub fn close_on() -> bool {
    std::env::var("NANO_CLOSE").is_ok_and(|v| v == "on")
}

static G1_SETTLE: std::sync::RwLock<Option<bool>> = std::sync::RwLock::new(None);
/// Set whether G1 forwards by the plain sum of the drones' settle chances for this process
/// (`None`: by `NANO_G1_SETTLE`).
pub fn set_g1_settle(on: Option<bool>) {
    if let Ok(mut g) = G1_SETTLE.write() {
        *g = on;
    }
}

/// Whether G1 forwards by the plain sum of the drones' settle chances (`settle_forward`): off,
/// unless `NANO_G1_SETTLE=on` or `set_g1_settle` (step G1 of the tentacle master plan, C120: the
/// best measured fusion of the kinds on G2's pool, 77 of 180 against G1's order 55, p 0.0002).
#[must_use]
pub fn g1_settle_on() -> bool {
    G1_SETTLE.read().ok().and_then(|g| *g).unwrap_or_else(|| std::env::var("NANO_G1_SETTLE").is_ok_and(|v| v == "on"))
}

/// G1's hand-over by the plain sum of the drones' settle chances (C120): every piece G1 holds
/// (its hand-over and its queue) is kept, every evidence kind with it (wording, naming and
/// meaning all enter the settle chance); the pieces are ordered by `Evidence::settle_sum`, ties
/// by their place before; the hand-over keeps its count and takes the best, the queue the rest.
/// Nothing is dropped. The meaning lane's shortlist is joined at the head after it (try 2: with
/// the lane ordered inside the sum, try 1, first fell 167 to 135 of 315, T entry of step G1).
/// No-op when off or without the drones' calibrated evidence (the flags arm).
fn settle_forward(ev: Option<&nmpod::evidence::Evidence>, handover: &mut Vec<usize>, queue: &mut Vec<usize>) {
    let Some(ev) = ev.filter(|_| g1_settle_on()) else { return };
    let n = handover.len();
    let mut pool: Vec<(f32, usize, usize)> = handover.iter().chain(queue.iter()).enumerate().map(|(i, c)| (ev.settle_sum(*c), i, *c)).collect();
    pool.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let order: Vec<usize> = pool.into_iter().map(|(_, _, c)| c).collect();
    *handover = order.iter().take(n).copied().collect();
    *queue = order.iter().skip(n).copied().collect();
}

static SETTLE: std::sync::RwLock<bool> = std::sync::RwLock::new(false);
/// Set which chance G2's selection reads: the settle chance (`nmpod::evidence::Evidence::settle`)
/// or the first form (`Chance::first_form`).
pub fn set_settle(on: bool) {
    if let Ok(mut g) = SETTLE.write() {
        *g = on;
    }
}

/// G2's chance that a piece settles a part of the request (the word parts; a relation is settled
/// through the set, `nmbrain::g2::select`).
#[derive(Clone, Debug, Default)]
pub struct Chance {
    /// The material's strongest evidence for each aspect.
    top: Vec<f32>,
    settle: bool,
}
impl Chance {
    #[must_use]
    pub fn new(ev: &nmpod::evidence::Evidence) -> Self {
        Self {
            top: (0..ev.aspects.len()).map(|a| (0..ev.len()).map(|i| ev.aspect(a, i)).fold(0.0_f32, f32::max)).collect(),
            settle: SETTLE.read().is_ok_and(|g| *g),
        }
    }
    /// The chance piece `c` settles part `k`, as G2 reads it.
    #[must_use]
    pub fn r(&self, ev: &nmpod::evidence::Evidence, k: usize, c: usize) -> f32 {
        if self.settle { ev.settles(k, c) } else { self.first_form(ev, k, c) }
    }
    /// The first form: the drones' part chance, times what the piece holds of the part's aspect
    /// as a share of the material's strongest.
    #[must_use]
    pub fn first_form(&self, ev: &nmpod::evidence::Evidence, k: usize, c: usize) -> f32 {
        let a = ev.parts.get(k).map_or(0, |p| p.0);
        let t = self.top.get(a).copied().unwrap_or(0.0);
        if t > 0.0 { ev.part_chance(k, c) * ev.aspect(a, c) / t } else { 0.0 }
    }
}

/// The pack that leads with the answer (guide WP5), written from what went up (`up`, in order):
/// the answer (the pieces of G2's chosen set when it chose, else the pieces of `up` that hold a
/// part of the request), each whole with its epistemic tag and the parts it covers; what is needed
/// with it (pieces of `up` joined to an answer piece by a typed relation), whole; then the rivals
/// as one outline line each (its place and its first code line, or its heading): the `rivals` G2
/// named, candidates left out whose gain in F still stands above its noise band (with no
/// selection of G2's, the rest of `up`). Everything is written
/// within the caller's `room` in tokens, the outline lines counted (a piece needed with the answer
/// or an outline line that would pass the room is left out; the answer itself was chosen within
/// it). Every line shown is the file's own line; nothing is generated. The head names the status
/// (answered when the answer and what is needed with it hold every part of the request, low naming
/// the parts they do not hold, withheld when no piece holds a part), the confidence (the share of
/// parts held) and the tokens.
#[must_use]
#[expect(clippy::too_many_arguments, reason = "the material, the drones, their evidence, the request, what went up, what G2 chose, the queue and the room")]
pub fn write_pack(
    sub: &Substrate,
    keep: &Keep,
    ev: &nmpod::evidence::Evidence,
    request: &nmpod::intake::Request,
    up: &[usize],
    chosen: &[usize],
    graph: &nmsubs::refgraph::Specific,
    rivals: &[usize],
    room: usize,
) -> String {
    write_pack_served(sub, keep, ev, request, up, chosen, graph, rivals, room).0
}

/// `write_pack`, and the pack as the pieces it serves (audit W13, lane T1 2026-11-19): each answer
/// span (grown toward its definition) as one piece over its lines, then what is needed with it,
/// then each rival as its one outline line; in the pack's order.
#[must_use]
#[expect(clippy::too_many_arguments, reason = "as write_pack")]
pub fn write_pack_served(
    sub: &Substrate,
    keep: &Keep,
    ev: &nmpod::evidence::Evidence,
    request: &nmpod::intake::Request,
    up: &[usize],
    chosen: &[usize],
    graph: &nmsubs::refgraph::Specific,
    rivals: &[usize],
    room: usize,
) -> (String, Vec<Piece>) {
    use std::fmt::Write;
    let n_p = ev.parts.len();
    let holds = |c: usize| (0..n_p).filter(|k| ev.part_holds(*k, c)).collect::<Vec<usize>>();
    let answer: Vec<usize> = if chosen.is_empty() { up.iter().copied().filter(|c| !holds(*c).is_empty()).collect() } else { up.iter().copied().filter(|c| chosen.contains(c)).collect() };
    let path = |c: usize| sub.chunks.get(c).and_then(|ch| sub.files.get(ch.file)).map(|f| f.path.replace('\\', "/")).unwrap_or_default();
    // A piece written whole: its head line (place, heading, tag, the parts it covers) and its text.
    let block = |c: usize| -> String {
        let mut out = String::new();
        let Some(ch) = sub.chunks.get(c) else { return out };
        let p = path(c);
        let doc = nmtype::is_document(&p);
        let hs = holds(c);
        let a = hs.first().and_then(|k| ev.parts.get(*k)).map_or(0, |x| x.0);
        let tag = keep.record(c).map_or("", |r| nmpod::evidence::tag(r, ev.stems.get(a).map_or(&[][..], Vec::as_slice), doc).name());
        let covers: Vec<&str> = hs.iter().filter_map(|k| ev.parts.get(*k).map(|x| x.1.as_str())).collect();
        let covers = if covers.is_empty() { String::new() } else { format!("  (covers: {})", covers.join("; ")) };
        let head = if ch.heading.is_empty() { String::new() } else { format!("  {}", ch.heading) };
        let _ = writeln!(out, "@@ {p}:{}-{}{head}  [{tag}]{covers}", ch.start, ch.end);
        out.push_str(&ch.text);
        if !ch.text.ends_with('\n') {
            out.push('\n');
        }
        out
    };
    // What a text costs in the room: the pack's own count (four characters a token).
    let cost = |t: &str| t.chars().count().div_ceil(4);
    // The head line at its longest (every part named as not held) is counted before anything else.
    let parts_named: usize = ev.parts.iter().map(|p| cost(&p.1) + 1).sum();
    let mut used = cost("nm-pack v2  status answered  confidence 0.00  kind explain  parts 00 of 00  tokens 00000\nconfidence: low; not held: \n== answer\n") + parts_named;
    // The answer as the definitions that settle its parts: answer pieces of one file whose lines
    // meet are one span; each span's goal is every definition (or document section) holding one of
    // its pieces' lines that hold a part, its pieces' own lines always kept.
    struct Span {
        chunk: usize,
        file: usize,
        lo: usize,
        hi: usize,
        goal: (usize, usize),
    }
    let mut spans: Vec<Span> = Vec::new();
    // The definitions of the files the answer pieces are in, found once.
    let answer_files: Vec<&str> = answer.iter().filter_map(|c| sub.chunks.get(*c).and_then(|ch| sub.files.get(ch.file)).map(|f| f.path.as_str())).collect();
    let file_defs: Vec<&nmsubs::repo::Symbol> = sub.symbols.iter().filter(|s| answer_files.contains(&s.file.as_str())).collect();
    for c in &answer {
        let Some(ch) = sub.chunks.get(*c) else { continue };
        let stems: Vec<&str> = holds(*c).iter().filter_map(|k| request_stems(request, *k)).flatten().map(String::as_str).collect();
        let (s, e) = definition_of(sub, &file_defs, *c, &stems);
        let goal = (s.min(ch.start), e.max(ch.end));
        match spans.iter_mut().find(|w| w.file == ch.file && ch.start <= w.hi + 1 && ch.end + 1 >= w.lo) {
            Some(w) => {
                w.lo = w.lo.min(ch.start);
                w.hi = w.hi.max(ch.end);
                w.goal = (w.goal.0.min(goal.0), w.goal.1.max(goal.1));
            }
            None => spans.push(Span { chunk: *c, file: ch.file, lo: ch.start, hi: ch.end, goal }),
        }
    }
    // The lines of each file an answer span is in, split once.
    let file_lines: BTreeMap<usize, Vec<&str>> = spans.iter().filter_map(|w| sub.files.get(w.file).map(|f| (w.file, f.text.lines().collect()))).collect();
    let line_cost = |file: usize, n: usize| file_lines.get(&file).and_then(|v| v.get(n.saturating_sub(1))).map_or(1, |l| cost(l) + 1);
    let span_text = |w: &Span| -> String {
        let b = block(w.chunk);
        let Some(ch) = sub.chunks.get(w.chunk) else { return b };
        let head = b.lines().next().unwrap_or("").replacen(&format!(":{}-{}", ch.start, ch.end), &format!(":{}-{}", w.lo, w.hi), 1);
        let lines: String = file_lines.get(&w.file).map_or(String::new(), |v| v.iter().skip(w.lo.saturating_sub(1)).take((w.hi + 1).saturating_sub(w.lo)).map(|l| format!("{l}\n")).collect());
        format!("{head}\n{lines}")
    };
    used += spans.iter().map(|w| cost(&span_text(w))).sum::<usize>();
    // Each span grows toward its goal, a line above then a line below, while the room holds; a line
    // another span already shows costs nothing, and spans that meet become one.
    let grow = |spans: &mut Vec<Span>, used: &mut usize| {
        for i in 0..spans.len() {
            while let Some(w) = spans.get(i) {
                let (file, lo, hi, goal) = (w.file, w.lo, w.hi, w.goal);
                let shown = |n: usize| spans.iter().any(|x| x.file == file && x.lo <= n && n <= x.hi);
                let mut step = |n: usize| -> bool {
                    let c = if shown(n) { 0 } else { line_cost(file, n) };
                    if *used + c <= room {
                        *used += c;
                        true
                    } else {
                        false
                    }
                };
                let up = lo > goal.0 && step(lo - 1);
                let down = hi < goal.1 && step(hi + 1);
                if let Some(w) = spans.get_mut(i) {
                    if up {
                        w.lo -= 1;
                    }
                    if down {
                        w.hi += 1;
                    }
                }
                if !up && !down {
                    break;
                }
            }
        }
        let mut merged: Vec<Span> = Vec::new();
        for w in spans.drain(..) {
            match merged.iter_mut().find(|m| m.file == w.file && w.lo <= m.hi + 1 && w.hi + 1 >= m.lo) {
                Some(m) => {
                    m.lo = m.lo.min(w.lo);
                    m.hi = m.hi.max(w.hi);
                }
                None => merged.push(w),
            }
        }
        *spans = merged;
    };
    let mut needed: Vec<(usize, String)> = Vec::new();
    for c in up.iter().copied().filter(|c| !answer.contains(c) && graph.edges_of(*c).iter().any(|e| answer.contains(&(e.to as usize)))) {
        let b = block(c);
        if used + cost(&b) <= room {
            used += cost(&b);
            needed.push((c, b));
        }
    }
    // The rivals G2 named (with no selection of G2's: the rest of what went up).
    let taken: Vec<usize> = answer.iter().copied().chain(needed.iter().map(|n| n.0)).collect();
    let listed: Vec<usize> = if chosen.is_empty() { up.to_vec() } else { rivals.to_vec() };
    let mut lines: Vec<String> = Vec::new();
    let mut rival_of: Vec<usize> = Vec::new();
    for c in listed.into_iter().filter(|c| !taken.contains(c)) {
        let Some(ch) = sub.chunks.get(c) else { continue };
        let first = if ch.heading.is_empty() { ch.text.lines().map(str::trim).find(|l| !l.is_empty() && !l.starts_with("//")).unwrap_or("") } else { ch.heading.as_str() };
        let line = format!("{}:{}-{}  {first}\n", path(c), ch.start, ch.end);
        if lines.contains(&line) || used + cost(&line) > room {
            continue;
        }
        used += cost(&line);
        lines.push(line);
        rival_of.push(c);
    }
    // The definitions grow last, into the room the answer pieces, what is needed with them and
    // the rivals' lines leave.
    grow(&mut spans, &mut used);
    let answer_text: String = spans.iter().map(&span_text).collect();
    // The pack as served pieces, in its order.
    let mut served: Vec<Piece> = Vec::new();
    let mk = |c: usize, start: usize, end: usize, text: String, rank: usize| Piece {
        chunk: c,
        path: path(c),
        start,
        end,
        text,
        evidence: crate::joint_read::Evidence { rank, ..crate::joint_read::Evidence::default() },
    };
    // Each span's text is its file lines joined, with no newline after the last (the packet's
    // verbatim check splits on every newline: a trailing one reads as one more line, and the span
    // is then shown as a note without its place, the eight-set run's closure first 22 to 4).
    for w in &spans {
        let text: String = file_lines.get(&w.file).map_or(String::new(), |v| v.iter().skip(w.lo.saturating_sub(1)).take((w.hi + 1).saturating_sub(w.lo)).copied().collect::<Vec<&str>>().join("\n"));
        served.push(mk(w.chunk, w.lo, w.hi, text, served.len()));
    }
    // What is needed with the answer is marked (`evidence.tentacle` 1) so the server can place it
    // after the set's other pieces.
    for (c, _) in &needed {
        if let Some(ch) = sub.chunks.get(*c) {
            let mut p = mk(*c, ch.start, ch.end, ch.text.clone(), served.len());
            p.evidence.tentacle = 1;
            served.push(p);
        }
    }
    for (c, l) in rival_of.iter().zip(&lines) {
        if let Some(ch) = sub.chunks.get(*c) {
            served.push(mk(*c, ch.start, ch.start, l.trim_end().to_owned(), served.len()));
        }
    }

    let held: std::collections::BTreeSet<usize> = answer.iter().chain(needed.iter().map(|n| &n.0)).flat_map(|c| holds(*c)).collect();
    let open: Vec<&str> = (0..n_p).filter(|k| !held.contains(k)).filter_map(|k| ev.parts.get(k).map(|p| p.1.as_str())).collect();
    let status = if answer.is_empty() {
        "withheld"
    } else if open.is_empty() {
        "answered"
    } else {
        "low"
    };
    #[expect(clippy::cast_precision_loss, reason = "part counts")]
    let confidence = if n_p == 0 { 0.0 } else { held.len() as f32 / n_p as f32 };
    let mut body = String::new();
    if status == "withheld" {
        let _ = writeln!(body, "no piece holds a part of the request: {}", (0..n_p).filter_map(|k| ev.parts.get(k).map(|p| p.1.as_str())).collect::<Vec<_>>().join("; "));
    } else {
        let _ = writeln!(body, "== answer");
        body.push_str(&answer_text);
        if !needed.is_empty() {
            let _ = writeln!(body, "== needed with it");
            for (_, b) in &needed {
                body.push_str(b);
            }
        }
        if !lines.is_empty() {
            let _ = writeln!(body, "== also relevant");
            for l in lines {
                body.push_str(&l);
            }
        }
    }
    let low = if open.is_empty() { String::new() } else { format!("\nconfidence: low; not held: {}", open.join("; ")) };
    let mut out = format!(
        "nm-pack v2  status {status}  confidence {confidence:.2}  kind {}  parts {} of {n_p}  tokens {}{low}\n",
        request.kind,
        held.len(),
        body.chars().count() / 4
    );
    out.push_str(&body);
    if status == "withheld" {
        served.clear();
    }
    (out, served)
}


/// The stack's item stage (belief per item; lead's design 1.1, 1.3, 1.9): G1's order per item is
/// the order of G2's candidates (`cands`, pieces in G1's order) carried to the items their lines
/// meet, each item at its first piece's place (the items of one piece by their own evidence);
/// G2's set is chosen over those items by the same selection (`nmbrain::g2::select`) on the items'
/// own evidence (`iev`: the chance each item settles each part, the relations of the request
/// settled through typed relations between the items' pieces, confirmed on the items' records,
/// and the facts of the code's structure); each item costs what the pack writes of it. The pack
/// writes each chosen item whole when it fits the room left, else as one line citing its whole
/// range and its attached comment, definition line and lines holding the request's parts; then
/// the items of what went up joined to a chosen item by a typed relation (whole, when they fit);
/// then the rivals as one line each; then the shown lines of each item written in part grow
/// toward its whole, a line above and a line below, while the room holds. Returns the pack and
/// the chosen items.
#[expect(clippy::too_many_arguments, clippy::too_many_lines, reason = "the material, the drones, the items' evidence, the request, G2's candidates, what went up, the graph and the room")]
fn item_stage(
    sub: &Substrate,
    keep: &Keep,
    question: &str,
    iev: &nmpod::evidence::Evidence,
    request: &nmpod::intake::Request,
    cands: &[usize],
    up: &[usize],
    graph: &nmsubs::refgraph::Specific,
    room: usize,
) -> (String, Vec<usize>) {
    use nmpod::intake::Asks;
    use nmsubs::refgraph::{EdgeKind, Relation as Dir};
    use std::fmt::Write;
    let items = keep.items();
    let n_p = iev.parts.len();
    let item = |i: usize| items.items.get(i);
    let path = |i: usize| item(i).and_then(|it| sub.files.get(it.file)).map(|f| f.path.replace('\\', "/")).unwrap_or_default();
    // The lines of every file an item of the candidates or of what went up is in, split once.
    let files_used: std::collections::BTreeSet<usize> =
        cands.iter().chain(up).flat_map(|c| items.of_piece.get(*c).map_or(&[][..], Vec::as_slice)).filter_map(|i| item(*i).map(|it| it.file)).collect();
    let split: BTreeMap<usize, Vec<&str>> = files_used.into_iter().filter_map(|f| sub.files.get(f).map(|x| (f, x.text.lines().collect()))).collect();
    let lines_of = |i: usize| item(i).and_then(|it| split.get(&it.file)).map_or(&[][..], Vec::as_slice);
    let cost = |t: &str| t.chars().count().div_ceil(4);
    // G1's order per item.
    let mut icands: Vec<usize> = Vec::new();
    for c in cands {
        let mut of: Vec<usize> = items.of_piece.get(*c).cloned().unwrap_or_default();
        of.sort_by(|a, b| iev.piece(*b).total_cmp(&iev.piece(*a)).then(a.cmp(b)));
        for i in of {
            if !icands.contains(&i) {
                icands.push(i);
            }
        }
    }
    let holds = |i: usize| (0..n_p).filter(|k| iev.part_holds(*k, i)).collect::<Vec<usize>>();
    let part_stems: Vec<&Vec<String>> = request.aspects.iter().flat_map(|a| a.parts.iter()).map(|p| &p.stems).collect();
    // The lines of an item shown when the whole does not fit: its attached comment, its definition
    // line, and its own lines holding a stem of a part it holds (of any part when it holds none).
    let shown = |i: usize| -> Vec<(usize, usize)> {
        let Some(it) = item(i) else { return Vec::new() };
        let lines = lines_of(i);
        let held = holds(i);
        let stems: Vec<&str> = if held.is_empty() {
            part_stems.iter().copied().flatten().map(String::as_str).collect()
        } else {
            held.iter().filter_map(|k| part_stems.get(*k).copied()).flatten().map(String::as_str).collect()
        };
        let has = |l: &str| {
            l.split(|x: char| !(x.is_alphanumeric() || x == '_')).filter(|w| !w.is_empty()).flat_map(nmlex::name_parts).any(|p| stems.contains(&nmlex::stem(&p).as_str()))
        };
        let mut keep_lines: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        if let Some((a, b)) = it.described {
            keep_lines.extend(a..=b);
        }
        if it.def > 0 {
            keep_lines.insert(it.def);
        }
        for (a, b) in &it.runs {
            keep_lines.extend((*a..=*b).filter(|n| lines.get(n.saturating_sub(1)).is_some_and(|l| has(l))));
        }
        if keep_lines.is_empty() {
            keep_lines.insert(it.start());
        }
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for n in keep_lines {
            match runs.last_mut() {
                Some(r) if r.1 + 1 == n => r.1 = n,
                _ => runs.push((n, n)),
            }
        }
        runs
    };
    let line_cost = |i: usize, n: usize| lines_of(i).get(n.saturating_sub(1)).map_or(1, |l| cost(l) + 1);
    let run_cost = |i: usize, runs: &[(usize, usize)]| -> usize { runs.iter().map(|(a, b)| cost(&format!("@@ {}:{a}-{b}\n", path(i))) + (*a..=*b).map(|n| line_cost(i, n)).sum::<usize>()).sum() };
    let label = |i: usize| item(i).map_or(String::new(), |it| it.label.clone());
    let whole_cost = |i: usize| item(i).map_or(0, |it| cost(&format!("@@ {}:{}-{}  {}  [mentions]\n", path(i), it.start(), it.end(), label(i))) + (it.start()..=it.end()).map(|n| line_cost(i, n)).sum::<usize>());
    let part_cost = |i: usize| item(i).map_or(0, |it| cost(&format!("{}:{}-{}  {}\n", path(i), it.start(), it.end(), label(i))) + run_cost(i, &shown(i)));
    let costs: BTreeMap<usize, (usize, usize)> = icands.iter().map(|i| (*i, (whole_cost(*i), part_cost(*i)))).collect();
    let whole_of = |i: usize| costs.get(&i).map_or_else(|| whole_cost(i), |c| c.0);
    let written_cost = |i: usize| if whole_of(i) <= room { whole_of(i) } else { costs.get(&i).map_or_else(|| part_cost(i), |c| c.1) };
    // G2's selection over the items.
    let chance = Chance::new(iev);
    let asks: Vec<Asks> = request.relations.clone();
    let facts_on = FACTS.read().is_ok_and(|g| *g);
    let fact_of = |k: usize| iev.facts.iter().filter(|_| facts_on).find(|f| f.0 == k).map(|f| f.1.as_slice());
    #[expect(clippy::cast_precision_loss, reason = "a count of items")]
    let fact_cols: Vec<(usize, usize, f32)> =
        (0..asks.len()).filter_map(|k| fact_of(k).map(|v| (k, v))).flat_map(|(k, v)| v.iter().map(move |p| (k, *p, 1.0 / v.len().max(1) as f32))).collect();
    let kept: Vec<usize> = (0..asks.len()).filter(|k| fact_of(*k).is_none()).collect();
    let rw = |i: usize| -> Vec<f32> { (0..n_p).map(|k| chance.r(iev, k, i)).chain(fact_cols.iter().map(|(_, p, w)| if *p == i { *w } else { 0.0 })).collect() };
    // The relations item `q` settles with the items at the far ends of its pieces' typed relations.
    let rel_from = |q: usize| -> Vec<(usize, usize)> {
        let mut out: std::collections::BTreeSet<(usize, usize)> = std::collections::BTreeSet::new();
        let Some(rq) = keep.item_record(q) else { return Vec::new() };
        for pq in items.pieces_of.get(q).map_or(&[][..], Vec::as_slice) {
            for e in graph.edges_of(*pq) {
                for p in items.of_piece.get(e.to as usize).map_or(&[][..], Vec::as_slice) {
                    if *p == q {
                        continue;
                    }
                    let Some(rp) = keep.item_record(*p) else { continue };
                    for (k, a) in asks.iter().enumerate() {
                        let ok = match a {
                            Asks::Callees => e.kind == EdgeKind::Call && e.dir == Dir::Uses && rp.defines_used_by(rq),
                            Asks::Callers => e.kind == EdgeKind::Call && e.dir == Dir::UsedBy && rq.defines_used_by(rp),
                            Asks::Reads => matches!(e.kind, EdgeKind::DefUse | EdgeKind::TypeUse | EdgeKind::DefinesUsed) && e.dir == Dir::Uses && rp.defines_used_by(rq),
                        };
                        if ok {
                            out.insert((*p, k));
                        }
                    }
                }
            }
        }
        out.into_iter().collect()
    };
    let index: BTreeMap<usize, usize> = icands.iter().enumerate().map(|(x, c)| (*c, x)).collect();
    let mut pairs: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    let mut linked: Vec<Vec<usize>> = vec![Vec::new(); icands.len()];
    if !kept.is_empty() {
        for (qi, q) in icands.iter().enumerate() {
            for (p, k) in rel_from(*q) {
                let (Some(&pi), Some(kk)) = (index.get(&p), kept.iter().position(|x| *x == k)) else { continue };
                pairs.entry((pi, qi)).or_default().push(kk);
                for (a, b) in [(qi, pi), (pi, qi)] {
                    if let Some(l) = linked.get_mut(a)
                        && !l.contains(&b)
                    {
                        l.push(b);
                    }
                }
            }
        }
    }
    let choices: Vec<nmbrain::g2::Choice> = icands.iter().map(|i| nmbrain::g2::Choice { chunk: *i, tokens: written_cost(*i).max(1), r: rw(*i) }).collect();
    let sel = nmbrain::g2::select(&choices, kept.len(), &|p, q| pairs.get(&(p, q)).cloned().unwrap_or_default(), &|q| linked.get(q).cloned().unwrap_or_default(), room);
    let mut set: Vec<usize> = sel.set.iter().filter_map(|x| icands.get(*x).copied()).collect();
    set.sort_by_key(|i| index.get(i).copied().unwrap_or(usize::MAX));
    // The rivals: candidates left out whose gain on what the set leaves open stands above the band.
    let n_w = n_p + fact_cols.len();
    let miss: Vec<f32> = (0..n_w).map(|a| 1.0 - sel.covered.get(a).copied().unwrap_or(0.0)).collect();
    #[expect(clippy::cast_precision_loss, reason = "token counts")]
    let mut rivals: Vec<(f32, usize)> = icands
        .iter()
        .filter(|i| !set.contains(i))
        .map(|i| (rw(*i).iter().zip(&miss).map(|(x, m)| x * m).sum::<f32>(), *i))
        .filter(|g| g.0 > sel.band)
        .map(|(g, i)| (g / written_cost(i).max(1) as f32, i))
        .collect();
    rivals.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    // The pack.
    let parts_named: usize = iev.parts.iter().map(|p| cost(&p.1) + 1).sum();
    let mut used = cost("nm-pack v2  status answered  confidence 0.00  kind explain  parts 00 of 00  tokens 00000\nconfidence: low; not held: \n== answer\n") + parts_named;
    // Each chosen item: whole, or (its whole range, the runs shown) when the whole does not fit.
    enum Written {
        Whole(usize),
        Part(usize, Vec<(usize, usize)>),
    }
    let mut answer: Vec<Written> = Vec::new();
    for i in &set {
        let w = whole_of(*i);
        if used + w <= room {
            used += w;
            answer.push(Written::Whole(*i));
        } else {
            let runs = shown(*i);
            let c = costs.get(i).map_or_else(|| part_cost(*i), |c| c.1);
            if used + c <= room {
                used += c;
                answer.push(Written::Part(*i, runs));
            } else if let Some(it) = item(*i) {
                // The line naming the item still cites it when its lines do not fit.
                let l = cost(&format!("{}:{}-{}  {}\n", path(*i), it.start(), it.end(), label(*i)));
                if used + l <= room {
                    used += l;
                    answer.push(Written::Part(*i, Vec::new()));
                }
            }
        }
    }
    let shown_items: Vec<usize> = answer.iter().map(|w| match w {
        Written::Whole(i) | Written::Part(i, _) => *i,
    }).collect();
    // What is needed with the answer: items of what went up joined to a chosen item by a typed
    // relation, whole, when they fit.
    let mut needed: Vec<usize> = Vec::new();
    for c in up {
        for i in items.of_piece.get(*c).map_or(&[][..], Vec::as_slice) {
            if set.contains(i) || needed.contains(i) {
                continue;
            }
            let joined = items.pieces_of.get(*i).is_some_and(|ps| {
                ps.iter().any(|p| graph.edges_of(*p).iter().any(|e| items.of_piece.get(e.to as usize).is_some_and(|to| to.iter().any(|t| set.contains(t)))))
            });
            if joined && iev.covers(*i).len() + holds(*i).len() > 0 && used + whole_cost(*i) <= room {
                used += whole_cost(*i);
                needed.push(*i);
            }
        }
    }
    // The rivals, one line each: its whole range and its label (or first code line).
    let mut lines: Vec<String> = Vec::new();
    for (_, i) in &rivals {
        let Some(it) = item(*i) else { continue };
        let first = if label(*i).is_empty() {
            lines_of(*i).get(it.start().saturating_sub(1)..it.end()).and_then(|v| v.iter().map(|l| l.trim()).find(|l| !l.is_empty() && !l.starts_with("//"))).unwrap_or("").to_owned()
        } else {
            label(*i)
        };
        let line = format!("{}:{}-{}  {first}\n", path(*i), it.start(), it.end());
        if lines.contains(&line) || used + cost(&line) > room {
            continue;
        }
        used += cost(&line);
        lines.push(line);
    }
    // The shown lines of each item written in part grow toward its whole while the room holds.
    for w in &mut answer {
        let Written::Part(i, runs) = w else { continue };
        let Some(it) = item(*i) else { continue };
        if runs.is_empty() {
            continue;
        }
        let (lo_all, hi_all) = (it.start(), it.end());
        loop {
            let mut grew = false;
            for r in 0..runs.len() {
                let Some(&(a, b)) = runs.get(r) else { continue };
                let prev_end = if r == 0 { lo_all.saturating_sub(1) } else { runs.get(r - 1).map_or(0, |x| x.1) };
                let next_start = runs.get(r + 1).map_or(hi_all + 1, |x| x.0);
                if a > lo_all && a - 1 > prev_end && used + line_cost(*i, a - 1) <= room {
                    used += line_cost(*i, a - 1);
                    if let Some(x) = runs.get_mut(r) {
                        x.0 = a - 1;
                    }
                    grew = true;
                }
                if b < hi_all && b + 1 < next_start && used + line_cost(*i, b + 1) <= room {
                    used += line_cost(*i, b + 1);
                    if let Some(x) = runs.get_mut(r) {
                        x.1 = b + 1;
                    }
                    grew = true;
                }
            }
            // Runs that meet become one (their header line no longer counts).
            let mut merged: Vec<(usize, usize)> = Vec::new();
            for (a, b) in runs.drain(..) {
                match merged.last_mut() {
                    Some(m) if a <= m.1 + 1 => {
                        m.1 = m.1.max(b);
                        used = used.saturating_sub(cost(&format!("@@ {}:{a}-{b}\n", path(*i))));
                    }
                    _ => merged.push((a, b)),
                }
            }
            *runs = merged;
            if !grew {
                break;
            }
        }
    }
    let tag = |i: usize| -> &'static str {
        let hs = holds(i);
        let a = hs.first().and_then(|k| iev.parts.get(*k)).map_or(0, |x| x.0);
        keep.item_record(i).map_or("", |r| nmpod::evidence::tag(r, iev.stems.get(a).map_or(&[][..], Vec::as_slice), nmtype::is_document(&path(i))).name())
    };
    let covers = |i: usize| -> String {
        let c: Vec<&str> = holds(i).iter().filter_map(|k| iev.parts.get(*k).map(|x| x.1.as_str())).collect();
        if c.is_empty() { String::new() } else { format!("  (covers: {})", c.join("; ")) }
    };
    let text_of = |i: usize, a: usize, b: usize| -> String { lines_of(i).get(a.saturating_sub(1)..b.min(lines_of(i).len())).map_or(String::new(), |v| v.iter().map(|l| format!("{l}\n")).collect()) };
    let mut answer_text = String::new();
    for w in &answer {
        match w {
            Written::Whole(i) => {
                let Some(it) = item(*i) else { continue };
                let lab = if label(*i).is_empty() { String::new() } else { format!("  {}", label(*i)) };
                let _ = writeln!(answer_text, "@@ {}:{}-{}{lab}  [{}]{}", path(*i), it.start(), it.end(), tag(*i), covers(*i));
                answer_text.push_str(&text_of(*i, it.start(), it.end()));
            }
            Written::Part(i, runs) => {
                let Some(it) = item(*i) else { continue };
                let _ = writeln!(answer_text, "{}:{}-{}  {}  [{}]{}  (lines shown below; the whole range holds the rest)", path(*i), it.start(), it.end(), label(*i), tag(*i), covers(*i));
                for (a, b) in runs {
                    let _ = writeln!(answer_text, "@@ {}:{a}-{b}", path(*i));
                    answer_text.push_str(&text_of(*i, *a, *b));
                }
            }
        }
    }
    let held: std::collections::BTreeSet<usize> = shown_items.iter().chain(&needed).flat_map(|i| holds(*i)).collect();
    let open: Vec<&str> = (0..n_p).filter(|k| !held.contains(k)).filter_map(|k| iev.parts.get(k).map(|p| p.1.as_str())).collect();
    let status = if shown_items.is_empty() {
        "withheld"
    } else if open.is_empty() {
        "answered"
    } else {
        "low"
    };
    #[expect(clippy::cast_precision_loss, reason = "part counts")]
    let confidence = if n_p == 0 { 0.0 } else { held.len() as f32 / n_p as f32 };
    let mut body = String::new();
    if status == "withheld" {
        let _ = writeln!(body, "no item holds a part of the request: {}", (0..n_p).filter_map(|k| iev.parts.get(k).map(|p| p.1.as_str())).collect::<Vec<_>>().join("; "));
    } else {
        let _ = writeln!(body, "== answer");
        body.push_str(&answer_text);
        if !needed.is_empty() {
            let _ = writeln!(body, "== needed with it");
            for i in &needed {
                let Some(it) = item(*i) else { continue };
                let lab = if label(*i).is_empty() { String::new() } else { format!("  {}", label(*i)) };
                let _ = writeln!(body, "@@ {}:{}-{}{lab}  [{}]{}", path(*i), it.start(), it.end(), tag(*i), covers(*i));
                body.push_str(&text_of(*i, it.start(), it.end()));
            }
        }
        if !lines.is_empty() {
            let _ = writeln!(body, "== also relevant");
            for l in lines {
                body.push_str(&l);
            }
        }
    }
    let _ = question;
    let low = if open.is_empty() { String::new() } else { format!("\nconfidence: low; not held: {}", open.join("; ")) };
    let mut out = format!("nm-pack v2  status {status}  confidence {confidence:.2}  kind {}  parts {} of {n_p}  tokens {}{low}\n", request.kind, held.len(), body.chars().count() / 4);
    out.push_str(&body);
    (out, set)
}

/// The stems of the request's part `k` (its parts in order over its aspects, as the drones' evidence
/// numbers them).
fn request_stems(request: &nmpod::intake::Request, k: usize) -> Option<&Vec<String>> {
    request.aspects.iter().flat_map(|a| a.parts.iter()).nth(k).map(|p| &p.stems)
}

/// The lines of the definition that settles a piece's parts: for code, every innermost definition
/// (with its doc comments and attributes) holding one of the piece's lines whose words hold one of
/// `stems`; for a document, the section the piece belongs to. The piece's own lines when there is
/// none.
fn definition_of(sub: &Substrate, symbols: &[&nmsubs::repo::Symbol], c: usize, stems: &[&str]) -> (usize, usize) {
    let Some(ch) = sub.chunks.get(c) else { return (0, 0) };
    let Some(f) = sub.files.get(ch.file) else { return (ch.start, ch.end) };
    if !f.is_code() {
        return ch.section.map_or((ch.start, ch.end), |(s, e)| (s.min(ch.start), e.max(ch.end)));
    }
    // Every line read here is at or above the piece's last line.
    let lines: Vec<&str> = f.text.lines().take(ch.end).collect();
    let holds = |l: &str| {
        l.split(|x: char| !(x.is_alphanumeric() || x == '_')).filter(|w| !w.is_empty()).flat_map(nmlex::name_parts).any(|p| stems.contains(&nmlex::stem(&p).as_str()))
    };
    let defs: Vec<&&nmsubs::repo::Symbol> = symbols.iter().filter(|s| s.file == f.path && s.line <= ch.end && s.end >= ch.start).collect();
    let (mut lo, mut hi) = (usize::MAX, 0);
    for n in ch.start..=ch.end {
        if !lines.get(n.saturating_sub(1)).is_some_and(|l| holds(l)) {
            continue;
        }
        if let Some(d) = defs.iter().filter(|s| s.line <= n && n <= s.end).min_by_key(|s| s.end - s.line) {
            // Its doc comments and attributes: the comment and attribute lines right above it.
            let mut top = d.line;
            while top > 1
                && lines.get(top - 2).is_some_and(|l| {
                    let t = l.trim_start();
                    t.starts_with("///") || t.starts_with("//!") || t.starts_with("#[") || t.starts_with("//") || t.starts_with("/*") || t.starts_with("* ") || t.starts_with('@')
                })
            {
                top -= 1;
            }
            lo = lo.min(top);
            hi = hi.max(d.end);
        }
    }
    if hi == 0 { (ch.start, ch.end) } else { (lo, hi) }
}

/// Whether the judge at G2's edge reads each contested piece in pack form (`pack_form`); off by
/// default, `NANO_JUDGE_READ=pack` for a measurement (read where the judge first runs).
static PACK_READ: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static PACK_READ_SET: std::sync::Once = std::sync::Once::new();

/// Piece `c` as the pack shows its evidence: for each of its lines holding a stem of `stems`, the
/// doc comment and attribute lines above the innermost definition holding it, that definition's
/// first line, and the line itself; in file order, each once. `None` when no line holds a stem.
fn pack_form(sub: &Substrate, c: usize, stems: &[&str]) -> Option<String> {
    PACK_READ_SET.call_once(|| PACK_READ.store(std::env::var("NANO_JUDGE_READ").is_ok_and(|v| v == "pack"), std::sync::atomic::Ordering::Relaxed));
    let ch = sub.chunks.get(c)?;
    let f = sub.files.get(ch.file)?;
    let lines: Vec<&str> = f.text.lines().take(ch.end).collect();
    let holds = |l: &str| {
        l.split(|x: char| !(x.is_alphanumeric() || x == '_')).filter(|w| !w.is_empty()).flat_map(nmlex::name_parts).any(|p| stems.contains(&nmlex::stem(&p).as_str()))
    };
    let defs: Vec<&nmsubs::repo::Symbol> = sub.symbols.iter().filter(|s| s.file == f.path && s.line <= ch.end && s.end >= ch.start).collect();
    let mut keep: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for n in ch.start..=ch.end {
        if !lines.get(n.saturating_sub(1)).is_some_and(|l| holds(l)) {
            continue;
        }
        keep.insert(n);
        if let Some(d) = defs.iter().filter(|s| s.line <= n && n <= s.end).min_by_key(|s| s.end - s.line) {
            keep.insert(d.line);
            let mut top = d.line;
            while top > 1
                && lines.get(top - 2).is_some_and(|l| {
                    let t = l.trim_start();
                    t.starts_with("///") || t.starts_with("//!") || t.starts_with("#[") || t.starts_with("//") || t.starts_with("/*") || t.starts_with("* ") || t.starts_with('@')
                })
            {
                top -= 1;
                keep.insert(top);
            }
        }
    }
    if keep.is_empty() {
        return None;
    }
    Some(keep.iter().filter_map(|n| lines.get(n.saturating_sub(1))).map(|l| format!("{l}
")).collect())
}

/// The strips of piece `c`: each line that holds one of `stems` with the whole of the smallest
/// definition it sits in (its comment and attributes above it, its body to its end, within the
/// piece), or, outside any definition, with two lines either side; the piece's other lines are
/// left out, and a gap between kept lines is marked `...`. `None` when no line holds one.
/// Try 2 (T-061): the first form kept only the holding line and its definition's header, and lost
/// the deciding lines of the definition (pack 44 to 27 of 59 on large code).
#[must_use]
pub fn strips(sub: &Substrate, c: usize, stems: &[&str]) -> Option<String> {
    let ch = sub.chunks.get(c)?;
    let f = sub.files.get(ch.file)?;
    let lines: Vec<&str> = f.text.lines().take(ch.end).collect();
    let holds = |l: &str| {
        l.split(|x: char| !(x.is_alphanumeric() || x == '_')).filter(|w| !w.is_empty()).flat_map(nmlex::name_parts).any(|p| stems.contains(&nmlex::stem(&p).as_str()))
    };
    let defs: Vec<&nmsubs::repo::Symbol> = sub.symbols.iter().filter(|s| s.file == f.path && s.line <= ch.end && s.end >= ch.start).collect();
    let mut keep: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for n in ch.start..=ch.end {
        if !lines.get(n.saturating_sub(1)).is_some_and(|l| holds(l)) {
            continue;
        }
        match defs.iter().filter(|s| s.line <= n && n <= s.end).min_by_key(|s| s.end - s.line) {
            Some(d) => {
                let mut top = d.line;
                while top > ch.start
                    && lines.get(top - 2).is_some_and(|l| {
                        let t = l.trim_start();
                        t.starts_with("///") || t.starts_with("//!") || t.starts_with("#[") || t.starts_with("//") || t.starts_with("/*") || t.starts_with("* ") || t.starts_with('@')
                    })
                {
                    top -= 1;
                }
                keep.extend(top.max(ch.start)..=d.end.min(ch.end));
            }
            None => keep.extend(n.saturating_sub(2).max(ch.start)..=(n + 2).min(ch.end)),
        }
    }
    if keep.is_empty() {
        return None;
    }
    let mut out = String::new();
    let mut last = 0_usize;
    for n in &keep {
        if last > 0 && *n > last + 1 {
            out.push_str("...\n");
        }
        if let Some(l) = lines.get(n.saturating_sub(1)) {
            out.push_str(l);
            out.push('\n');
        }
        last = *n;
    }
    Some(out)
}

/// The pieces of `chunks` for the Core Brain, in the order given (the Globalized Context Stack's order).
#[must_use]
pub fn pieces(sub: &Substrate, chunks: &[usize], windows: &[Vec<usize>]) -> Vec<Piece> {
    chunks
        .iter()
        .enumerate()
        .filter_map(|(rank, c)| {
            let ch = sub.chunks.get(*c)?;
            let tentacle = windows.iter().position(|w| w.contains(c)).unwrap_or(0);
            Some(Piece {
                chunk: *c,
                path: sub.files.get(ch.file)?.path.replace('\\', "/"),
                start: ch.start,
                end: ch.end,
                text: ch.text.clone(),
                evidence: Evidence { tentacle, rank, ..Evidence::default() },
            })
        })
        .collect()
}

/// Section check: the path is not on by default anywhere (it has no switch of its own) and the
/// stated fallback is positive.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    if UP_FALLBACK == 0 {
        return Err("nmapi.drone_path: fallback".into());
    }
    let sig = |bm25f: f32, exact: f32| nmpod::intake::Signals { bm25f, exact, ..nmpod::intake::Signals::default() };
    let signals = [sig(3.0, 0.0), sig(1.0, 1.0), sig(0.0, 0.0), sig(2.0, 0.0)];
    let fused = fused_shares(&signals);
    if fused.iter().any(|f| !(0.0..=1.0).contains(f)) || fused.first() <= fused.get(3) {
        return Err(format!("nmapi.drone_path: fused shares {fused:?}"));
    }
    let ls = lenses(&signals, &fused, &nmsubs::refgraph::Specific::default(), &[2]);
    let w = compose_lenses(&ls, &fused, &|_| 10, 20);
    let mut all: Vec<usize> = w.iter().flatten().copied().collect();
    let n = all.len();
    all.sort_unstable();
    all.dedup();
    if all.len() != n || all.len() != signals.len() || w.iter().any(|x| x.len() > 2) {
        return Err(format!("nmapi.drone_path: lens windows {w:?}"));
    }
    if w.first().and_then(|x| x.first()) != Some(&0) || w.get(2).and_then(|x| x.first()) != Some(&1) || w.last().and_then(|x| x.first()) != Some(&2) {
        return Err(format!("nmapi.drone_path: lens order {w:?}"));
    }
    // The follow-up's names: calls and definitions, control words and macros left out.
    let (calls, defs) = anchor_names("fn route_call(x: u8) { if ok(x) { let y = parse_line(x); println!(\"{}\", y); } }");
    if calls != vec!["parse_line".to_owned()] || defs != vec!["route_call".to_owned()] {
        return Err(format!("nmapi.drone_path: anchor names {calls:?} {defs:?}"));
    }
    // W2 and W5: the fused value standardises and clips each kind; the set fills by it and covers parts.
    let f = fuse_kinds(&[vec![1.0, 2.0, 3.0, 100.0]], &[0, 1, 2, 3]);
    if f.iter().any(|x| x.abs() > W2_CLIP) || f.first() >= f.get(3) {
        return Err(format!("nmapi.drone_path: fused value {f:?}"));
    }
    // Closed-loop continuous learning retrieval weights: neutral defaults are 1.0; adapted weights shift ranking.
    let neutral = RetrievalWeights::from_adaptation(&nmlearn::engine::RetrievalAdaptation::default());
    if neutral != RetrievalWeights::default() {
        return Err(format!("nmapi.drone_path: default adaptation was not neutral: {neutral:?}"));
    }
    let adapted = nmlearn::engine::RetrievalAdaptation { lexical_weight: 0.48, semantic_weight: 0.20, ..Default::default() };
    let rw_adapted = RetrievalWeights::from_adaptation(&adapted);
    let lex_col = vec![10.0, 1.0, 0.0];
    let sem_col = vec![1.0, 10.0, 0.0];
    let eq = fuse_kinds(&[lex_col.clone(), sem_col.clone()], &[0, 1, 2]);
    let wt = fuse_kinds_weighted(&[(rw_adapted.lexical, lex_col), (rw_adapted.semantic, sem_col)], &[0, 1, 2]);
    if (eq[0] - eq[1]).abs() > 1e-5 || wt[0] <= wt[1] {
        return Err(format!("nmapi.drone_path: retrieval adaptation failed to shift ranking: eq={eq:?} wt={wt:?}"));
    }
    let set = value_fill(&[0, 1, 2], &[3.0, 2.0, 1.0], &|_| 10, 20, 1, &|_, c| c == 2);
    if set != vec![0, 2] {
        return Err(format!("nmapi.drone_path: value fill {set:?}"));
    }
    // The chain choice: with the published table, code to the embedder and prose to the judge; with
    // a measured routing record in force (Stage 5 WP-C2), each kind to its stronger measured
    // MicroModel; without the judge, prose to the embedder either way.
    let stronger = |kind: PieceKind| strengths().into_iter().filter(|r| r.1 == kind).max_by(|a, b| a.2.total_cmp(&b.2)).map(|r| r.0);
    let expect_code = if measured_record().is_some() { stronger(PieceKind::Code) } else { Some(MicroModel::Embedder) };
    let expect_prose = if measured_record().is_some() { stronger(PieceKind::Prose) } else { Some(MicroModel::Judge) };
    if chain_choice(PieceKind::Code, true) != expect_code || chain_choice(PieceKind::Prose, true) != expect_prose || chain_choice(PieceKind::Prose, false) != Some(MicroModel::Embedder) {
        return Err("nmapi.drone_path: chain choice".into());
    }
    let unset = |k: &str| std::env::var(k).is_err();
    if !route_on()
        || route_judge_on()
        || (unset("NANO_SWAP_IN") && !swap_in_on())
        || (unset("NANO_FOLLOW") && !follow_on())
        || (unset("NANO_MEANING_MAP") && !meaning_map_on())
    {
        return Err("nmapi.drone_path: defaults (routing on, its judge half off; W7: swap-in, follow-up and meaning map on)".into());

    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Roadmap Phase 4 (2026-11-24): the `joint` kind is in W2's list with `NANO_STACK_JOINT=on` or
    /// when named, once, and absent by default.
    #[test]
    fn w2_joint_kind_follows_its_switch() {
        let default = w2_kinds_from(None, false);
        assert_eq!(default, ["wording", "meaning", "settle", "lane", "drone"]);
        assert!(!default.iter().any(|k| k == "joint"));
        let switched = w2_kinds_from(None, true);
        assert_eq!(switched, ["wording", "meaning", "settle", "lane", "drone", "joint"]);
        let named = w2_kinds_from(Some("wording, settle,joint"), false);
        assert_eq!(named, ["wording", "settle", "joint"]);
        let both = w2_kinds_from(Some("wording,joint"), true);
        assert_eq!(both.iter().filter(|k| *k == "joint").count(), 1);
        assert!(!w2_kinds_from(Some("wording,meaning"), false).iter().any(|k| k == "joint"));
    }
}
