//! The Core Brain's decision step, first form (2026-11-10; architecture by Luis Jake Gabriel III;
//! Gavriel Technologies). Selectable and off by default: nothing in the served path calls it
//! unless `serve --brain-chains --brain-decide` (or `trace` with the same switches) is given, and
//! `run dx d4t` tests it alone.
//!
//! Input: the union of the tentacles' shortlists (references with evidence, `joint_read::Piece`).
//! Round 1: the released Qwen3-Reranker-0.6B scores every candidate (full depth, as released, the
//! served card prompt). Then a tournament of small joint reads with the base Qwen3-0.6B at full
//! depth: groups of `GROUP` candidates taken in score order, one joint read per group (question,
//! then every piece whole), a yes or no label per piece; the group's best goes to the next round,
//! until one remains. Nothing is dropped: candidates the joint rounds do not reach stay queued as
//! references, in score order, after the read ones.
//!
//! Group size: 4, from the measured curve `runs/DX-j3a-kcurve-1794320274135.txt` (full-depth yes
//! or no joint read right on every read of 2 and 4 chunks, wrong at 8; joint encoding slows from
//! 1.1 to 5.9 ms per token from 0.9k to 12.6k tokens). How far the first round goes is derived at
//! run time from the measured seconds per read against the time budget (stated fallback 30 s, the
//! target time per question); the first group is always read.
use crate::joint_read::{self, Piece};
use nmengn::model::Model;
use nmengn::placement::Device;
use nmtype::Fault;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Candidates per joint read, from the measured curve (see the module note).
pub const GROUP: usize = 4;
/// Stated fallback time budget in seconds for the joint rounds of one decision.
pub const BUDGET_FALLBACK_S: f64 = 30.0;
/// The served switch (with `--brain-chains`).
pub const FLAG: &str = "--brain-decide";

static ON: AtomicBool = AtomicBool::new(false);
/// Switch the decision step on or off for this process (off by default).
pub fn set_enabled(on: bool) {
    ON.store(on, Ordering::SeqCst);
}
/// Whether the decision step is on.
#[must_use]
pub fn enabled() -> bool {
    ON.load(Ordering::SeqCst)
}

/// One joint read of the tournament.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Read {
    pub round: usize,
    /// Candidates read (indices into the candidates given), as placed.
    pub group: Vec<usize>,
    /// Yes minus no label logit per candidate of `group`.
    pub yes_no: Vec<f32>,
    pub winner: usize,
    pub tokens: usize,
    pub seconds: f64,
}

/// The outcome of a decision.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Decision {
    /// Indices into the candidates, best first: the winner, then the read candidates by the
    /// round they reached and their label, then the queued ones in score order.
    pub order: Vec<usize>,
    /// The reranker's P(yes) per candidate.
    pub rerank: Vec<f32>,
    /// Candidates in score order.
    pub by_score: Vec<usize>,
    pub reads: Vec<Read>,
    /// Candidates not reached by the joint rounds (references, score order).
    pub queued: Vec<usize>,
    pub rerank_s: f64,
    pub joint_s: f64,
    /// Agreement rule only: the embedder's cosine per candidate, and its seconds.
    pub embed: Vec<f32>,
    pub embed_s: f64,
}
impl Decision {
    #[must_use]
    pub fn winner(&self) -> Option<usize> {
        self.order.first().copied()
    }
}

fn dev_of(m: &Model) -> Device {
    if m.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu }
}

/// Indices best first by `s`, ties by index.
fn best_first(s: &[f32], idx: &[usize]) -> Vec<usize> {
    let mut v = idx.to_vec();
    v.sort_by(|&a, &b| {
        s.get(b).copied().unwrap_or(f32::NEG_INFINITY).total_cmp(&s.get(a).copied().unwrap_or(f32::NEG_INFINITY)).then(a.cmp(&b))
    });
    v
}

/// Joint reads still needed to finish a tournament that starts with `field_groups` group winners.
#[must_use]
pub fn reads_after(field_groups: usize) -> usize {
    let (mut n, mut reads) = (field_groups, 0);
    while n > 1 {
        let g = n.div_ceil(GROUP);
        reads += g;
        n = g;
    }
    reads
}

/// Decide over `cands`: reranker scores (`reranker`), then the tournament with the base model
/// (`base`, drone `heads` through `last` for the joint read's attention readout, which is kept but
/// not used for the decision). `budget_s` bounds the joint rounds (`None`: the fallback).
/// # Errors
/// Tokenizer or engine faults.
#[expect(clippy::too_many_arguments, reason = "one decision's inputs, passed once")]
pub fn decide(
    base: &Model,
    reranker: &Model,
    question: &str,
    cands: &[Piece],
    heads: &[(usize, usize)],
    last: usize,
    budget_s: Option<f64>,
) -> Result<Decision, Fault> {
    let mut d = Decision::default();
    if cands.is_empty() {
        return Ok(d);
    }
    let t0 = Instant::now();
    let docs: Vec<String> = cands.iter().map(|p| nmsubs::dense::doc_text(&p.path, &p.text)).collect();
    d.rerank = nmengn::roles::card_rerank_set(reranker, dev_of(reranker), nmengn::roles::CODE_INSTRUCTION, question, &docs)?;
    d.rerank_s = t0.elapsed().as_secs_f64();
    d.by_score = best_first(&d.rerank, &(0..cands.len()).collect::<Vec<_>>());
    let budget = budget_s.unwrap_or(BUDGET_FALLBACK_S);
    let t1 = Instant::now();
    let dev = dev_of(base);
    let read = |group: &[usize], round: usize, d: &mut Decision| -> Result<usize, Fault> {
        let t = Instant::now();
        let refs: Vec<&Piece> = group.iter().filter_map(|i| cands.get(*i)).collect();
        let order: Vec<usize> = (0..refs.len()).collect();
        let o = joint_read::pass(base, dev, question, &refs, &order, heads, last, true)?;
        nmengn::residency::get().release_windows();
        let w = best_first(&o.yes_no, &order).first().and_then(|j| group.get(*j)).copied().unwrap_or(group[0]);
        d.reads.push(Read {
            round,
            group: group.to_vec(),
            yes_no: o.yes_no,
            winner: w,
            tokens: o.tokens,
            seconds: t.elapsed().as_secs_f64(),
        });
        Ok(w)
    };
    // Round 1 over groups in score order, as far as the budget reaches.
    let groups: Vec<Vec<usize>> = d.by_score.chunks(GROUP).map(<[usize]>::to_vec).collect();
    let mut winners = Vec::new();
    let mut reached = 0;
    for (g, grp) in groups.iter().enumerate() {
        if g > 0 {
            #[expect(clippy::cast_precision_loss, reason = "counts")]
            let per = t1.elapsed().as_secs_f64() / d.reads.len().max(1) as f64;
            if t1.elapsed().as_secs_f64() + per * (1 + reads_after(g + 1)) as f64 > budget {
                break;
            }
        }
        winners.push(if grp.len() > 1 { read(grp, 1, &mut d)? } else { grp[0] });
        reached += grp.len();
    }
    let mut round = 1;
    while winners.len() > 1 {
        round += 1;
        let mut next = Vec::new();
        for grp in winners.chunks(GROUP) {
            next.push(if grp.len() > 1 { read(grp, round, &mut d)? } else { grp[0] });
        }
        winners = next;
    }
    d.joint_s = t1.elapsed().as_secs_f64();
    d.queued = d.by_score.get(reached..).map(<[usize]>::to_vec).unwrap_or_default();
    // Order: by the furthest round reached (winner first), then label in that round.
    let mut best: std::collections::BTreeMap<usize, (usize, f32)> = std::collections::BTreeMap::new();
    for r in &d.reads {
        for (i, y) in r.group.iter().zip(&r.yes_no) {
            let e = best.entry(*i).or_insert((0, f32::NEG_INFINITY));
            if r.round >= e.0 {
                *e = (r.round + usize::from(*i == r.winner), *y);
            }
        }
    }
    let mut read_ones: Vec<usize> = best.keys().copied().collect();
    read_ones.sort_by(|a, b| {
        let (x, y) = (best[a], best[b]);
        y.0.cmp(&x.0).then(y.1.total_cmp(&x.1)).then(a.cmp(b))
    });
    let mut order = Vec::new();
    if let Some(&w) = winners.first() {
        order.push(w);
    }
    order.extend(read_ones.into_iter().filter(|i| Some(i) != winners.first()));
    let rest: Vec<usize> = d.by_score.iter().copied().filter(|i| !order.contains(i)).collect();
    order.extend(rest);
    d.order = order;
    Ok(d)
}

/// Mean reciprocal rank over several orders of the same candidates (each a list of candidate
/// indices, best first); a candidate missing from an order adds nothing for it. Best first, ties
/// by index. LocAgent's rule for merging independent rankings (Chen et al. 2025,
/// https://arxiv.org/html/2503.09089v2, method only): no weights.
#[must_use]
pub fn agree(orders: &[Vec<usize>], n: usize) -> Vec<usize> {
    let mut score = vec![0.0_f64; n];
    for o in orders {
        for (r, i) in o.iter().enumerate() {
            if let Some(x) = score.get_mut(*i) {
                #[expect(clippy::cast_precision_loss, reason = "ranks")]
                {
                    *x += 1.0 / (r + 1) as f64;
                }
            }
        }
    }
    let mut v: Vec<usize> = (0..n).collect();
    v.sort_by(|a, b| score.get(*b).copied().unwrap_or(0.0).total_cmp(&score.get(*a).copied().unwrap_or(0.0)).then(a.cmp(b)));
    v
}

/// The agreement decision from scores already made: the reranker's and the embedder's score per
/// candidate, candidates in arrival order.
#[must_use]
pub fn agreement_from(rerank: Vec<f32>, embed: Vec<f32>) -> Decision {
    let all: Vec<usize> = (0..rerank.len()).collect();
    let by_score = best_first(&rerank, &all);
    let order = agree(&[by_score.clone(), best_first(&embed, &all), all], rerank.len());
    Decision { order, rerank, by_score, embed, ..Decision::default() }
}

/// Pieces the second look reads: the winner's file's first pieces in the decision order.
pub const LOOK_PIECES: usize = 3;
/// Stated fallback for the second look's switch threshold (yes-or-no logit margin over the
/// winner): the mean of the two out-of-fold choices of CB-1 (0.176 and 0.234,
/// `runs/DX-2026-11-12-cb1-1794493324761.txt`), until more questions refit it.
pub const LOOK_MARGIN: f32 = 0.205;

/// What the second look did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Look {
    /// Candidates read together (indices into the candidates), the winner first.
    pub read: Vec<usize>,
    /// Yes-or-no score per read candidate, in `read` order.
    pub scores: Vec<f32>,
    /// The joint read's best minus the winner's score.
    pub margin: f32,
    /// Whether first place moved.
    pub switched: bool,
    pub seconds: f64,
    /// Tokens encoded over every pass of the joint read, and the passes.
    pub tokens: usize,
    pub passes: usize,
}

/// The candidates the second look reads: the winner's file's first `k` candidates in `order`.
#[must_use]
pub fn look_set(cands: &[Piece], order: &[usize], k: usize) -> Vec<usize> {
    let Some(path) = order.first().and_then(|w| cands.get(*w)).map(|p| p.path.as_str()) else { return Vec::new() };
    order.iter().copied().filter(|i| cands.get(*i).is_some_and(|p| p.path == path)).take(k).collect()
}

/// The Core Brain's second look inside the winning file (2026-11-12, CB-1, research file Part
/// 10): when the winner's file has two or more of the candidates, the base model reads its first
/// `LOOK_PIECES` together, in both orders, with a full-depth yes-or-no label per piece
/// (`joint_read::read`). First place moves to the joint read's best only when its margin over the
/// winner reaches `min_margin`; no other place changes. Measured: right code first 19 to 22 of 30
/// out of fold, 1 broken (`runs/DX-2026-11-12-cb1-1794493324761.txt`). Encodings in RAM only.
/// # Errors
/// Tokenizer or engine faults.
#[expect(
    clippy::too_many_arguments,
    reason = "the model, the read's heads and depth, the question, the candidates, the order and the threshold"
)]
pub fn second_look(
    base: &Model,
    heads: &[(usize, usize)],
    last: usize,
    question: &str,
    cands: &[Piece],
    order: &mut Vec<usize>,
    min_margin: f32,
) -> Result<Look, Fault> {
    let read = look_set(cands, order, LOOK_PIECES);
    let mut look = Look { read: read.clone(), ..Look::default() };
    if read.len() < 2 {
        return Ok(look);
    }
    let pieces: Vec<Piece> = read.iter().filter_map(|i| cands.get(*i).cloned()).collect();
    let t = Instant::now();
    // Only the label readout decides here, so the attention readouts are not run.
    let jo = joint_read::read_with(base, dev_of(base), question, &pieces, heads, last, usize::MAX, true, true, false)?;
    look.seconds = t.elapsed().as_secs_f64();
    look.tokens = jo.tokens;
    look.passes = jo.passes;
    look.scores.clone_from(&jo.yes_no.group_score);
    let best = jo.yes_no.order.first().copied().unwrap_or(0);
    look.margin = look.scores.get(best).copied().unwrap_or(0.0) - look.scores.first().copied().unwrap_or(0.0);
    if best != 0
        && look.margin >= min_margin
        && let Some(&c) = read.get(best)
    {
        order.retain(|x| *x != c);
        order.insert(0, c);
        look.switched = true;
    }
    Ok(look)
}

/// Section check (processor only): off by default; the tournament's remaining reads.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    if enabled() {
        return Err("nmapi.decide: on by default".into());
    }
    // Agreement: a candidate second in two orders and first in none loses to one first in two.
    if agree(&[vec![0, 1, 2], vec![0, 2, 1], vec![1, 0, 2]], 3) != vec![0, 1, 2] || agree(&[], 2) != vec![0, 1] {
        return Err("nmapi.decide: agreement order".into());
    }
    // Second look: the winner's file's first three in decision order, the winner first.
    let p = |c: usize, path: &str| Piece { chunk: c, path: path.into(), ..Piece::default() };
    let cands = [p(0, "a.rs"), p(1, "b.rs"), p(2, "a.rs"), p(3, "a.rs"), p(4, "a.rs")];
    if look_set(&cands, &[2, 1, 4, 0, 3], LOOK_PIECES) != vec![2, 4, 0]
        || look_set(&cands, &[1, 0], LOOK_PIECES) != vec![1]
        || !look_set(&cands, &[], LOOK_PIECES).is_empty()
    {
        return Err("nmapi.decide: second look set".into());
    }
    if reads_after(1) != 0 || reads_after(2) != 1 || reads_after(4) != 1 || reads_after(5) != 3 || reads_after(16) != 5 {
        return Err("nmapi.decide: reads after".into());
    }
    Ok(())
}
