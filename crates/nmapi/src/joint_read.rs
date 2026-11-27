//! The Core Brain's joint read (2026-11-10, step 3a; architecture by Luis Jake Gabriel III;
//! specification `docs/2026-11-10-neumor-analysis-d-core-brain.md` gap 10.1 f, option C; faults
//! F14, F15 and F07 of `docs/2026-11-10-neumor-design-parts-and-gaps.md`). A separable component,
//! selectable and off by default (`JointOptions::default()` is off); nothing in the served path
//! calls it unless the option is set.
//!
//! Input: the shortlists the tentacles already produce (their best chunks by attention mass, each
//! with its evidence record), joined across tentacles and ordered by evidence (`union_order`). The
//! pieces pass as references resolved to text; their tokens are encoded here by the one central
//! base model (encodings of one population are not usable by another, so tokens pass).
//!
//! One pass (`pass`): the question, then every piece labelled with its path and lines and whole
//! (nothing truncated), encoded as one causal sequence through every layer and held in RAM only;
//! from that one encoding the readout tails are read without encoding the pieces again:
//! - R1, attention: at the question repeated after the pieces, the attention mass of the drone
//!   heads on each piece's tokens, minus the same with a content-free question (`Kind::Mass` of
//!   `core_read`; `Kind::Calibrated` reported beside it);
//! - R2, label logits of a yes or no question per piece: a short tail per piece asking whether
//!   piece `[k]` answers the question, logit of "yes" minus logit of "no" at its last position.
//!
//! `read` runs the pass in two piece orders (evidence order and reversed) and averages each
//! readout per piece, which cancels the first-order position effect. When the pieces exceed the
//! group length (the length the model reads reliably, measured by the caller; `group_by`), they
//! are read in consecutive groups and a final joint read over the group winners decides first
//! place, so nothing is dropped and every piece is read jointly at least once.
//!
//! Method sources (no code copied): In-Context Reranking (Chen, Jimenez Gutierrez, Su, arXiv
//! 2410.02642) for the attention readout with a content-free calibration; FIRST (Reddy et al. 2024)
//! and the reranker model card's yes or no prompt for the label readout; permutation
//! self-consistency for the two orders; the RankGPT sliding window in joint form for the groups.
use crate::core_read::{self, Kind, RawRead};
use nmengn::kvstore::{self, Window};
use nmengn::model::{Model, RunOpts};
use nmengn::placement::Device;
use nmtype::{Fault, FaultKind};
use std::sync::Arc;
use std::time::Instant;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmapi.joint_read", FaultKind::Execution, m)
}

/// The label tails read the whole output head instead of its yes and no rows (`serve
/// --full-head`, comparison only; off by default). Both give the same logits; the full head needs
/// a device copy of every vocabulary row.
pub static FULL_HEAD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Whether `FULL_HEAD` is set.
#[must_use]
pub fn full_head() -> bool {
    FULL_HEAD.load(std::sync::atomic::Ordering::Relaxed)
}

/// Stated fallback group length in tokens until the reliable length is measured (analysis D gap
/// 10.1 f, "start value until measured").
pub const GROUP_FALLBACK: usize = 4_000;

/// Whether the joint read is on, and its group length (`None`: the stated fallback).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JointOptions {
    pub enabled: bool,
    pub group_tokens: Option<usize>,
}
impl JointOptions {
    /// `off`, `on`, or `on:<tokens>`.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "off" => Some(Self::default()),
            "on" => Some(Self { enabled: true, group_tokens: None }),
            x => x.strip_prefix("on:").and_then(|n| n.parse().ok()).map(|n| Self { enabled: true, group_tokens: Some(n) }),
        }
    }
    #[must_use]
    pub fn group(&self) -> usize {
        self.group_tokens.unwrap_or(GROUP_FALLBACK)
    }
}

/// What a tentacle reported about a piece: its tentacle, its place in the tentacle's order after
/// the second look (0 first), the full read's mass and the second look's score.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Evidence {
    pub tentacle: usize,
    pub rank: usize,
    pub mass: f32,
    pub look: f32,
}

/// One piece: a reference (substrate chunk, path, first and last line), its text and evidence.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Piece {
    pub chunk: usize,
    pub path: String,
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub evidence: Evidence,
}

/// The union of the tentacles' shortlists ordered by evidence: by place in its tentacle's order
/// (every tentacle's first, then every second, ...), then by the full read's mass, then by chunk.
/// A chunk listed twice is kept once (its better place).
#[must_use]
pub fn union_order(lists: &[Vec<Piece>]) -> Vec<Piece> {
    let mut all: Vec<Piece> = lists.iter().flatten().cloned().collect();
    all.sort_by(|a, b| a.evidence.rank.cmp(&b.evidence.rank).then(b.evidence.mass.total_cmp(&a.evidence.mass)).then(a.chunk.cmp(&b.chunk)));
    let mut seen = std::collections::BTreeSet::new();
    all.retain(|p| seen.insert(p.chunk));
    all
}

/// Consecutive groups of at most `limit` tokens (a longer piece alone), in the given order.
#[must_use]
pub fn group_by(tokens: &[usize], limit: usize) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut sum = 0;
    for (i, &t) in tokens.iter().enumerate() {
        if !cur.is_empty() && sum + t > limit {
            out.push(std::mem::take(&mut cur));
            sum = 0;
        }
        cur.push(i);
        sum += t;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// The reliable length from a measured curve of (joint length, accuracy): the largest length
/// whose accuracy keeps at least two thirds of the best (analysis D gap 10.1 f); the fallback when
/// the curve is empty or holds no positive accuracy.
#[must_use]
pub fn reliable_length(curve: &[(usize, f64)]) -> usize {
    let best = curve.iter().map(|x| x.1).fold(0.0_f64, f64::max);
    if best <= 0.0 {
        return GROUP_FALLBACK;
    }
    curve.iter().filter(|x| x.1 * 3.0 >= best * 2.0 - 1e-9).map(|x| x.0).max().unwrap_or(GROUP_FALLBACK)
}

pub const SYSTEM: &str =
    "<|im_start|>system\nYou read pieces of one code repository and judge which pieces answer a question.<|im_end|>\n<|im_start|>user\n";
pub const ASSISTANT: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// The label line and body of piece `k` (1-based label) as placed in the joint context.
#[must_use]
pub fn piece_text(k: usize, p: &Piece) -> String {
    format!("[{k}] {}:{}-{}\n```\n{}\n```\n", p.path, p.start, p.end, p.text.trim_end_matches('\n'))
}

/// Token count of a piece as placed (label, fence and body).
/// # Errors
/// Tokenizer failure.
pub fn piece_tokens(m: &Model, k: usize, p: &Piece) -> Result<usize, Fault> {
    Ok(m.pop.tokenizer.encode(&piece_text(k, p)).map_err(|e| bad(e.to_string()))?.len())
}

/// The readouts of one pass, per piece in the order the pieces were given (not as placed).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PassOut {
    pub mass: Vec<f32>,
    pub calibrated: Vec<f32>,
    pub yes_no: Vec<f32>,
    pub tokens: usize,
    pub encode_s: f64,
    pub tails_s: f64,
}

/// One joint pass over `pieces` placed in `order` (indices into `pieces`). `heads` are the drone
/// heads (layers up to `last`); `yes_no` switches the per-piece label tails on.
/// # Errors
/// Tokenizer or engine faults.
#[expect(clippy::too_many_arguments, reason = "one pass's inputs, passed once")]
pub fn pass(
    m: &Model,
    dev: Device,
    question: &str,
    pieces: &[&Piece],
    order: &[usize],
    heads: &[(usize, usize)],
    last: usize,
    yes_no: bool,
) -> Result<PassOut, Fault> {
    pass_with(m, dev, question, pieces, order, heads, last, yes_no, true)
}

/// `pass` with the attention readouts (R1) on or off. Off, `mass` and `calibrated` are zeros and
/// the label tails (R2) are the same values `pass` gives: they read the same encoding.
/// # Errors
/// Tokenizer or engine faults.
#[expect(clippy::too_many_arguments, reason = "one pass's inputs, passed once")]
pub fn pass_with(
    m: &Model,
    dev: Device,
    question: &str,
    pieces: &[&Piece],
    order: &[usize],
    heads: &[(usize, usize)],
    last: usize,
    yes_no: bool,
    attention: bool,
) -> Result<PassOut, Fault> {
    let _job = nmengn::chanrec::job("joint");
    // The joint read runs on the served path's faster kernels when their switches are on
    // (`nmengn::gpu::ServedScope`); the engine's reference path is unchanged.
    let _served = nmengn::gpu::ServedScope::new();
    let tok = |s: &str| m.pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
    let t0 = Instant::now();
    let mut ids = tok(&format!("{SYSTEM}Question: {question}\n\n"))?;
    let mut parts = vec![(0, ids.len())];
    for (k, &i) in order.iter().enumerate() {
        let p = pieces.get(i).ok_or_else(|| bad("piece index out of range"))?;
        let t = tok(&piece_text(k + 1, p))?;
        parts.push((ids.len(), t.len()));
        ids.extend(t);
    }
    // Every layer when the label tails are read; through the drone layers otherwise.
    let depth = if yes_no { m.pop.dims.layers.saturating_sub(1) } else { last };
    let kv = Arc::new(kvstore::encode_through(m, dev, None, &ids, depth)?);
    let w = Arc::new(Window::build(m, &[kv])?);
    let encode_s = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let reads_span = nmengn::prof::Span::new("joint.reads");
    let (placed_mass, placed_cal) = if attention {
        let stop = kvstore::score_targets(m, last)?;
        let posed: RawRead = core_read::raw_read(m, dev, &w, &stop, heads, &format!("\nQuestion: {question}"))?;
        let null: RawRead = core_read::raw_read(m, dev, &w, &stop, heads, &core_read::plain_null(""))?;
        (core_read::chunk_scores(&posed, &null, &parts, Kind::Mass, None), core_read::chunk_scores(&posed, &null, &parts, Kind::Calibrated, None))
    } else {
        (vec![0.0; order.len()], vec![0.0; order.len()])
    };
    drop(reads_span);
    let _tails_span = nmengn::prof::Span::new("joint.tails");
    let mut placed_yn = vec![0.0_f32; order.len()];
    if yes_no {
        let first = |s: &str| -> Result<usize, Fault> {
            tok(s)?.first().and_then(|x| usize::try_from(*x).ok()).ok_or_else(|| bad(format!("no token for {s:?}")))
        };
        let (yes, no) = (first("yes")?, first("no")?);
        let target = (m.role_node("head")?, "logits".to_owned());
        // The tail reads two rows of the output head (yes, no); each logit equals the full head's.
        let full = full_head();
        let rows = if full { Arc::new(Vec::new()) } else { Arc::new(vec![yes, no]) };
        let (at_yes, at_no) = if full { (yes, no) } else { (0, 1) };
        let tails: Vec<Vec<i32>> = (0..order.len())
            .map(|k| tok(&format!("\nQuestion: {question}\nDoes piece [{}] answer the question? Answer yes or no.{ASSISTANT}", k + 1)))
            .collect::<Result<_, _>>()?;
        if dev == Device::Gpu && tails.len() > 1 {
            // On the card the tails run in one forward as packed sequences, each reading the
            // encoding as its prefix and none reading another tail; one row of logits each.
            let lens: Vec<usize> = tails.iter().map(Vec::len).collect();
            let all: Vec<i32> = tails.concat();
            let out = m.forward_at(
                dev,
                &all,
                w.len,
                std::slice::from_ref(&target),
                &RunOpts {
                    window: Some(Arc::clone(&w)),
                    head_rows: Arc::clone(&rows),
                    segments: Some(Arc::new(lens)),
                    ..RunOpts::default()
                },
            )?;
            let l = nmengn::model::values(out.get(&target).ok_or_else(|| bad("no logits"))?)?;
            let width = l.len() / tails.len();
            for (slot, row) in placed_yn.iter_mut().zip(l.chunks_exact(width.max(1))) {
                *slot = row.get(at_yes).copied().unwrap_or(0.0) - row.get(at_no).copied().unwrap_or(0.0);
            }
        } else {
            for (slot, tail) in placed_yn.iter_mut().zip(&tails) {
                let out = m.forward_at(
                    dev,
                    tail,
                    w.len,
                    std::slice::from_ref(&target),
                    &RunOpts { window: Some(Arc::clone(&w)), head_rows: Arc::clone(&rows), ..RunOpts::default() },
                )?;
                let l = nmengn::model::values(out.get(&target).ok_or_else(|| bad("no logits"))?)?;
                *slot = l.get(at_yes).copied().unwrap_or(0.0) - l.get(at_no).copied().unwrap_or(0.0);
            }
        }
    }
    w.release_device();
    let tails_s = t1.elapsed().as_secs_f64();
    // Back to the order the pieces were given.
    let mut out = PassOut {
        mass: vec![0.0; pieces.len()],
        calibrated: vec![0.0; pieces.len()],
        yes_no: vec![0.0; pieces.len()],
        tokens: ids.len(),
        encode_s,
        tails_s,
    };
    for (k, &i) in order.iter().enumerate() {
        if let (Some(a), Some(b), Some(c)) = (out.mass.get_mut(i), out.calibrated.get_mut(i), out.yes_no.get_mut(i)) {
            *a = placed_mass.get(k).copied().unwrap_or(0.0);
            *b = placed_cal.get(k).copied().unwrap_or(0.0);
            *c = placed_yn.get(k).copied().unwrap_or(0.0);
        }
    }
    Ok(out)
}

/// Whether label-only joint reads on the card run packed (Stage 5 WP-C3 J6; `NANO_PACKED_LABELS=off`
/// restores one pass per order).
#[must_use]
pub fn packed_labels_on() -> bool {
    !std::env::var("NANO_PACKED_LABELS").is_ok_and(|v| v == "off")
}

/// The yes and no label tokens and the tails of `pass_with`'s label read for `n` placed pieces.
fn label_tails(m: &Model, question: &str, n: usize) -> Result<(usize, usize, Vec<Vec<i32>>), Fault> {
    let tok = |s: &str| m.pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
    let first = |s: &str| -> Result<usize, Fault> { tok(s)?.first().and_then(|x| usize::try_from(*x).ok()).ok_or_else(|| bad(format!("no token for {s:?}"))) };
    let (yes, no) = (first("yes")?, first("no")?);
    let tails = (0..n).map(|k| tok(&format!("\nQuestion: {question}\nDoes piece [{}] answer the question? Answer yes or no.{ASSISTANT}", k + 1))).collect::<Result<Vec<_>, _>>()?;
    Ok((yes, no, tails))
}

/// The encoded span of `pass_with`: the system and question head, then each placed piece's label
/// and body, tokenized part by part exactly as `pass_with` joins them.
fn label_span(m: &Model, question: &str, pieces: &[&Piece], order: &[usize]) -> Result<Vec<i32>, Fault> {
    let tok = |s: &str| m.pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
    let mut ids = tok(&format!("{SYSTEM}Question: {question}\n\n"))?;
    for (k, &i) in order.iter().enumerate() {
        let p = pieces.get(i).ok_or_else(|| bad("piece index out of range"))?;
        ids.extend(tok(&piece_text(k + 1, p))?);
    }
    Ok(ids)
}

/// Label-only joint reads of several groups in one card run (Stage 5 WP-C3 J6): each group read in
/// the evidence order and, with `two`, the reverse order, every order one span with its per-piece
/// yes or no tails, all packed into one `nmengn::kvstore::read_group_labels` run. Returns per group
/// the per-piece label (logit of yes less no), the two orders averaged, in the order the pieces were
/// given: the values `read_with(.., yes_no = true, two, attention = false)` gives a single group,
/// within the card's rounding. On the card only.
/// # Errors
/// Tokenizer or engine faults, or a model not on the card.
pub fn read_labels_many(m: &Model, question: &str, groups: &[Vec<&Piece>], two: bool) -> Result<(Vec<Vec<f32>>, usize), Fault> {
    if m.pop.gpu.is_none() {
        return Err(bad("packed label reads run on the card only"));
    }
    let _job = nmengn::chanrec::job("joint");
    let _served = nmengn::gpu::ServedScope::new();
    let mut packed: Vec<(Vec<i32>, Vec<Vec<i32>>)> = Vec::new();
    let mut plan: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut yes_no = (0_usize, 0_usize);
    for (g, pieces) in groups.iter().enumerate() {
        let fwd: Vec<usize> = (0..pieces.len()).collect();
        let mut orders = vec![fwd.clone()];
        if two && pieces.len() > 1 {
            orders.push(fwd.iter().rev().copied().collect());
        }
        let (yes, no, tails) = label_tails(m, question, pieces.len())?;
        yes_no = (yes, no);
        for order in orders {
            packed.push((label_span(m, question, pieces, &order)?, tails.clone()));
            plan.push((g, order));
        }
    }
    if packed.is_empty() {
        return Ok((vec![Vec::new(); groups.len()], 0));
    }
    let tokens = packed.iter().map(|(s, t)| s.len() + t.iter().map(Vec::len).sum::<usize>()).sum();
    let rows = kvstore::read_group_labels(m, None, &packed, &[yes_no.0, yes_no.1])?;
    let mut sums: Vec<Vec<f32>> = groups.iter().map(|p| vec![0.0; p.len()]).collect();
    let mut counts = vec![0_u32; groups.len()];
    for ((g, order), per_tail) in plan.iter().zip(&rows) {
        if let Some(slot) = sums.get_mut(*g) {
            for (k, &i) in order.iter().enumerate() {
                let r = per_tail.get(k);
                let v = r.and_then(|r| r.first()).copied().unwrap_or(0.0) - r.and_then(|r| r.get(1)).copied().unwrap_or(0.0);
                if let Some(x) = slot.get_mut(i) {
                    *x += v;
                }
            }
        }
        if let Some(c) = counts.get_mut(*g) {
            *c += 1;
        }
    }
    for (s, c) in sums.iter_mut().zip(&counts) {
        if *c > 0 {
            #[expect(clippy::cast_precision_loss, reason = "one or two orders")]
            let d = *c as f32;
            for x in s.iter_mut() {
                *x /= d;
            }
        }
    }
    Ok((sums, tokens))
}

/// One readout's outcome over every piece.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ranked {
    /// Pieces best first (indices into the pieces given to `read`): the final read's order of the
    /// group winners, then every other piece by its group score.
    pub order: Vec<usize>,
    /// Score of every piece in its group (two orders averaged).
    pub group_score: Vec<f32>,
}

/// The outcome of a joint read.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct JointOut {
    pub mass: Ranked,
    pub calibrated: Ranked,
    pub yes_no: Ranked,
    pub groups: usize,
    pub passes: usize,
    pub tokens: usize,
    pub seconds: f64,
    pub encode_s: f64,
    pub tails_s: f64,
}

#[expect(clippy::too_many_arguments, reason = "one read's inputs, passed once")]
fn averaged(
    m: &Model,
    dev: Device,
    question: &str,
    pieces: &[&Piece],
    heads: &[(usize, usize)],
    last: usize,
    yes_no: bool,
    two: bool,
    attention: bool,
    acc: &mut JointOut,
) -> Result<[Vec<f32>; 3], Fault> {
    // Stage 5 WP-C3 J6: a label-only read on the card packs both orders into one run.
    if yes_no && !attention && dev == Device::Gpu && packed_labels_on() {
        let t0 = Instant::now();
        if let Ok((mut v, tokens)) = read_labels_many(m, question, &[pieces.to_vec()], two) {
            let n = pieces.len();
            acc.passes += if two && n > 1 { 2 } else { 1 };
            acc.tokens += tokens;
            acc.encode_s += t0.elapsed().as_secs_f64();
            return Ok([vec![0.0; n], vec![0.0; n], v.pop().unwrap_or_default()]);
        }
    }
    let fwd: Vec<usize> = (0..pieces.len()).collect();
    let rev: Vec<usize> = fwd.iter().rev().copied().collect();
    let a = pass_with(m, dev, question, pieces, &fwd, heads, last, yes_no, attention)?;
    if std::env::var_os("NM_JOINT_TRACE").is_some() {
        eprintln!("joint pass: {} pieces, {} tokens, encode {:.3} s, tails {:.3} s", pieces.len(), a.tokens, a.encode_s, a.tails_s);
    }
    let mut sums = [a.mass.clone(), a.calibrated.clone(), a.yes_no.clone()];
    acc.passes += 1;
    acc.tokens += a.tokens;
    acc.encode_s += a.encode_s;
    acc.tails_s += a.tails_s;
    if two && pieces.len() > 1 {
        let b = pass_with(m, dev, question, pieces, &rev, heads, last, yes_no, attention)?;
        for (s, v) in sums.iter_mut().zip([&b.mass, &b.calibrated, &b.yes_no]) {
            for (x, y) in s.iter_mut().zip(v) {
                *x = 0.5 * (*x + y);
            }
        }
        acc.passes += 1;
        acc.tokens += b.tokens;
        acc.encode_s += b.encode_s;
        acc.tails_s += b.tails_s;
    }
    Ok(sums)
}

fn best_first(s: &[f32], idx: &[usize]) -> Vec<usize> {
    let mut v = idx.to_vec();
    v.sort_by(|&a, &b| {
        s.get(b).copied().unwrap_or(f32::NEG_INFINITY).total_cmp(&s.get(a).copied().unwrap_or(f32::NEG_INFINITY)).then(a.cmp(&b))
    });
    v
}

/// The joint read of `pieces` (in evidence order): groups of at most `group` tokens, each read in
/// two orders (one, evidence order, when `two` is false); with more than one group, a final joint read (two orders) over each group's best
/// piece under every readout decides first place.
/// # Errors
/// Tokenizer or engine faults.
#[expect(clippy::too_many_arguments, reason = "one read's inputs, passed once")]
pub fn read(
    m: &Model,
    dev: Device,
    question: &str,
    pieces: &[Piece],
    heads: &[(usize, usize)],
    last: usize,
    group: usize,
    yes_no: bool,
    two: bool,
) -> Result<JointOut, Fault> {
    read_with(m, dev, question, pieces, heads, last, group, yes_no, two, true)
}

/// `read` with the attention readouts on or off (`pass_with`). Off, the `mass` and `calibrated`
/// outcomes carry zeros and the label outcome is the one `read` gives.
/// # Errors
/// Tokenizer or engine faults.
#[expect(clippy::too_many_arguments, reason = "one read's inputs, passed once")]
pub fn read_with(
    m: &Model,
    dev: Device,
    question: &str,
    pieces: &[Piece],
    heads: &[(usize, usize)],
    last: usize,
    group: usize,
    yes_no: bool,
    two: bool,
    attention: bool,
) -> Result<JointOut, Fault> {
    let _job = nmengn::chanrec::job("joint");
    let t0 = Instant::now();
    let mut acc = JointOut::default();
    let sizes: Vec<usize> = pieces.iter().enumerate().map(|(k, p)| piece_tokens(m, k + 1, p)).collect::<Result<_, _>>()?;
    let groups = group_by(&sizes, group);
    acc.groups = groups.len();
    let mut score = [vec![f32::NEG_INFINITY; pieces.len()], vec![f32::NEG_INFINITY; pieces.len()], vec![f32::NEG_INFINITY; pieces.len()]];
    let mut winners: Vec<std::collections::BTreeSet<usize>> = vec![Default::default(); 3];
    for g in &groups {
        let refs: Vec<&Piece> = g.iter().filter_map(|i| pieces.get(*i)).collect();
        let s = averaged(m, dev, question, &refs, heads, last, yes_no, two, attention, &mut acc)?;
        for r in 0..3 {
            for (j, &i) in g.iter().enumerate() {
                if let (Some(d), Some(v)) = (score[r].get_mut(i), s[r].get(j)) {
                    *d = *v;
                }
            }
            if let Some(&b) = best_first(&score[r], g).first() {
                winners[r].insert(b);
            }
        }
    }
    let mut finals: Vec<Vec<usize>> = (0..3).map(|r| best_first(&score[r], &(0..pieces.len()).collect::<Vec<_>>())).collect();
    if groups.len() > 1 {
        let w: Vec<usize> = winners.iter().flatten().copied().collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        let refs: Vec<&Piece> = w.iter().filter_map(|i| pieces.get(*i)).collect();
        let s = averaged(m, dev, question, &refs, heads, last, yes_no, two, attention, &mut acc)?;
        for r in 0..3 {
            let mut fin = vec![f32::NEG_INFINITY; pieces.len()];
            for (j, &i) in w.iter().enumerate() {
                if let (Some(d), Some(v)) = (fin.get_mut(i), s[r].get(j)) {
                    *d = *v;
                }
            }
            let top: Vec<usize> = winners[r].iter().copied().collect();
            let mut order = best_first(&fin, &top);
            order.extend(best_first(&score[r], &(0..pieces.len()).filter(|i| !winners[r].contains(i)).collect::<Vec<_>>()));
            finals[r] = order;
        }
    }
    let [sm, sc, sy] = score;
    let mut fs = finals.into_iter();
    acc.mass = Ranked { order: fs.next().unwrap_or_default(), group_score: sm };
    acc.calibrated = Ranked { order: fs.next().unwrap_or_default(), group_score: sc };
    acc.yes_no = Ranked { order: fs.next().unwrap_or_default(), group_score: sy };
    acc.seconds = t0.elapsed().as_secs_f64();
    Ok(acc)
}

/// Section check (processor only): option parsing (off by default), grouping keeps every piece
/// once in order with no group over the limit except a single long piece, the union keeps each
/// chunk once ordered by place then mass, and the reliable length follows the two-thirds rule.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    if JointOptions::default().enabled || JointOptions::parse("on:3000") != Some(JointOptions { enabled: true, group_tokens: Some(3000) }) {
        return Err("nmapi.joint_read: options".into());
    }
    let g = group_by(&[1500, 1500, 1500, 5000, 100], 3000);
    if g != vec![vec![0, 1], vec![2], vec![3], vec![4]] {
        return Err(format!("nmapi.joint_read: groups {g:?}"));
    }
    let p = |chunk: usize, rank: usize, mass: f32| Piece {
        chunk,
        evidence: Evidence { rank, mass, ..Evidence::default() },
        ..Piece::default()
    };
    let u = union_order(&[vec![p(5, 0, 0.1), p(6, 1, 0.9)], vec![p(7, 0, 0.5), p(5, 1, 0.3)]]);
    let ids: Vec<usize> = u.iter().map(|x| x.chunk).collect();
    if ids != vec![7, 5, 6] {
        return Err(format!("nmapi.joint_read: union {ids:?}"));
    }
    if reliable_length(&[(2000, 0.9), (4000, 0.6), (8000, 0.5)]) != 4000 || reliable_length(&[]) != GROUP_FALLBACK {
        return Err("nmapi.joint_read: reliable length".into());
    }
    Ok(())
}
