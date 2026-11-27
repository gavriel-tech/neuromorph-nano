//! Components of the tentacle core's read (2026-11-10, refinement step 2a; architecture by Luis
//! Jake Gabriel III; derivation `docs/2026-11-10-neumor-analysis-b-sensory.md` gaps 3.4, 4.1, 4.2,
//! 4.5 and 5.1; faults F08, F09 and F13 of `docs/2026-11-10-neumor-design-parts-and-gaps.md`).
//! Three separable, selectable parts, each with its own section check; with every option off the
//! core's read is the first form, bit for bit. The full window is always read first and nothing
//! is reduced or discarded; every read goes through the one central model by reference and over
//! encodings already held in RAM (nothing is encoded again, nothing is written to disk).
//!
//! 1. Second look (`look_size`, `look_pick`, `look_order`): after the full-window read the
//!    tentacle reads its own best chunks once more as one short window with the question. The
//!    shortlist is taken by the attention-mass score of the full read; its size is derived from
//!    those scores (`look_size`), with `LOOK_FALLBACK` as the stated fallback. First place inside
//!    the tentacle is decided by the short read, whose scores are reported unscaled, the full
//!    read's scores kept beside them.
//! 2. Token votes (`chunk_scores` with `Kind::Weighted`, `low_signal`): a chunk score in which each
//!    question token and drone head votes with the share of its attention that lands on the
//!    window's chunks, in place of an equal vote per token; and positions that draw attention with
//!    no question (found in the content-free read, per head) are left out of every chunk sum.
//! 3. Length-matched content-free pose (`matched_null`): the content-free pose in the same frame
//!    and with the same token count as the posed question, so the two reads differ in content
//!    and not in length or position.
//!
//! Method sources (no code copied): the content-free calibration and the low-score token filter
//! of In-Context Reranking (Chen, Jimenez Gutierrez, Su, arXiv 2410.02642, MIT licence) are the
//! references for the content-free read and for leaving positions out; the weighting of each
//! token and head by where its attention lands is the gain-weighted aggregation of analysis B
//! gap 4.2 row 1; the short second read follows the fovea re-read of gap 4.2 row 7.
use crate::tentacle_pass;
use nmengn::kvstore::{Probe, Window};
use nmengn::model::{Model, RunOpts};
use nmengn::placement::Device;
use nmtype::{Fault, FaultKind};
use std::sync::Arc;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmapi.core_read", FaultKind::Execution, m)
}

/// Which components of the core's read are on. All off: the first form.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadOptions {
    /// Component 1: the second look decides first place inside the tentacle, reported unscaled.
    pub second_look: bool,
    /// Component 2: attention-weighted token votes with low-signal positions left out.
    pub votes: bool,
    /// Component 3: the content-free pose has the token length of the posed question.
    pub matched_pose: bool,
}
impl ReadOptions {
    /// Every component on.
    #[must_use]
    pub fn all() -> Self {
        Self { second_look: true, votes: true, matched_pose: true }
    }
    /// From a flag value: `off`, `all`, or a comma list of `look`, `votes`, `pose`.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "off" | "first" => return Some(Self::default()),
            "all" => return Some(Self::all()),
            _ => {}
        }
        let mut o = Self::default();
        for p in s.split(',') {
            match p.trim() {
                "look" => o.second_look = true,
                "votes" => o.votes = true,
                "pose" => o.matched_pose = true,
                _ => return None,
            }
        }
        Some(o)
    }
    /// What is on, as carried in the core's report and the evidence record (empty: first form).
    #[must_use]
    pub fn text(&self) -> String {
        let mut v = Vec::new();
        if self.second_look {
            v.push("second look (mass shortlist, unscaled)");
        }
        if self.votes {
            v.push("attention-weighted votes, low-signal positions left out");
        }
        if self.matched_pose {
            v.push("length-matched content-free pose");
        }
        v.join("; ")
    }
    /// True when any component is on.
    #[must_use]
    pub fn any(&self) -> bool {
        self.second_look || self.votes || self.matched_pose
    }
}

// ---------------------------------------------------------------------------------------------
// Component 1: the second look.
// ---------------------------------------------------------------------------------------------

/// Stated fallback size of the second look's shortlist (analysis B gap 4.2 row 7; G1 rule D2).
pub const LOOK_FALLBACK: usize = tentacle_pass::SERVED_SHORTLIST;

/// Size of the second look's shortlist from the full read's mass scores: the smallest number of
/// best chunks that hold the mass a single read cannot tell from the best chunk's. The noise of
/// one read is taken from the scores themselves: chunks with a negative mass are pure read noise
/// (the question drew less attention there than no question), so the largest magnitude among the
/// negative scores is the noise floor, and every chunk whose mass is within that floor of the
/// best one is in the shortlist. With no negative score or no positive one the size is the stated
/// fallback (`LOOK_FALLBACK`). Always at least 2 (a second look over one chunk decides nothing)
/// and at most the chunk count. Returns the size and how it was set.
#[must_use]
pub fn look_size(mass: &[f32]) -> (usize, String) {
    let n = mass.len();
    let fin: Vec<f32> = mass.iter().copied().filter(|x| x.is_finite()).collect();
    let best = fin.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let floor = fin.iter().copied().filter(|x| *x < 0.0).map(f32::abs).fold(0.0_f32, f32::max);
    if n == 0 || best.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) || floor <= 0.0 {
        let k = LOOK_FALLBACK.min(n.max(1));
        return (k, format!("fallback {k}: no positive mass or no noise floor in the full read"));
    }
    let near = fin.iter().filter(|x| **x >= best - floor && **x > 0.0).count();
    let k = near.max(2).min(n);
    (k, format!("{k} chunks within the read noise ({floor:.4}, the largest negative mass) of the best mass ({best:.4}), of {n}"))
}

/// The shortlist: the `k` best chunks by `mass` (ties in window order), returned in window order.
#[must_use]
pub fn look_pick(mass: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..mass.len()).collect();
    idx.sort_by(|&a, &b| {
        mass.get(b).copied().unwrap_or(f32::NEG_INFINITY).total_cmp(&mass.get(a).copied().unwrap_or(f32::NEG_INFINITY)).then(a.cmp(&b))
    });
    idx.truncate(k);
    idx.sort_unstable();
    idx
}

/// Order of a window's chunks after a second look: the picked chunks by their second-look scores
/// (`look`, in the order of `pick`), best first, then every other chunk by its full-read score.
/// Ties in window order. Nothing leaves the order.
#[must_use]
pub fn look_order(full: &[f32], pick: &[usize], look: &[f32]) -> Vec<usize> {
    let at = |v: &[f32], i: usize| v.get(i).copied().unwrap_or(f32::NEG_INFINITY);
    let mut first: Vec<(usize, f32)> = pick.iter().enumerate().map(|(j, &i)| (i, at(look, j))).collect();
    first.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut rest: Vec<usize> = (0..full.len()).filter(|i| !pick.contains(i)).collect();
    rest.sort_by(|&a, &b| at(full, b).total_cmp(&at(full, a)).then(a.cmp(&b)));
    first.into_iter().map(|x| x.0).chain(rest).collect()
}

// ---------------------------------------------------------------------------------------------
// Component 2: token votes.
// ---------------------------------------------------------------------------------------------

/// The raw scores of one read, kept for the length of one scoring (transient, RAM only): per
/// drone head `[tokens, len]` scaled dot products before softmax, and per head and token the log
/// of the softmax normaliser over every position the token sees.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RawRead {
    pub tokens: usize,
    pub len: usize,
    pub logits: Vec<Vec<f32>>,
    pub lse: Vec<Vec<f32>>,
}
impl RawRead {
    /// Drone heads read.
    #[must_use]
    pub fn heads(&self) -> usize {
        self.logits.len()
    }
    /// Bytes held.
    #[must_use]
    pub fn bytes(&self) -> usize {
        (self.logits.iter().map(Vec::len).sum::<usize>() + self.lse.iter().map(Vec::len).sum::<usize>()) * 4
    }
    /// The attention weight of token `t` of head `h` on position `p`.
    fn weight(&self, h: usize, t: usize, p: usize) -> f32 {
        let l = self.logits.get(h).and_then(|v| v.get(t * self.len + p)).copied().unwrap_or(f32::NEG_INFINITY);
        let z = self.lse.get(h).and_then(|v| v.get(t)).copied().unwrap_or(0.0);
        if l.is_finite() { (l - z).exp() } else { 0.0 }
    }
    /// Per head, the attention on every window position averaged over the tokens (`heads x len`):
    /// the mass per question token the pass's read records.
    #[must_use]
    pub fn mass(&self) -> Vec<Vec<f32>> {
        #[expect(clippy::cast_precision_loss, reason = "token counts are small")]
        let per = 1.0 / self.tokens.max(1) as f32;
        (0..self.heads())
            .map(|h| (0..self.len).map(|p| (0..self.tokens).map(|t| self.weight(h, t, p)).sum::<f32>() * per).collect())
            .collect()
    }
}

/// One raw read of a window: `text` posed after `w`, the raw scores of every pair of `heads`.
/// # Errors
/// Tokenizer or engine faults, or a head the run did not read.
pub fn raw_read(
    model: &Model,
    dev: Device,
    w: &Arc<Window>,
    stop: &[(String, String)],
    heads: &[(usize, usize)],
    text: &str,
) -> Result<RawRead, Fault> {
    let _job = nmengn::chanrec::job("tentacle-core");
    let q = model.pop.tokenizer.encode(text).map_err(|e| bad(e.to_string()))?;
    let probe = Arc::new(Probe::of_heads(heads).with_logits());
    model.forward_at(
        dev,
        &q,
        w.len,
        stop,
        &RunOpts { window: Some(Arc::clone(w)), probe: Some(Arc::clone(&probe)), ..RunOpts::default() },
    )?;
    let mut out = RawRead { tokens: q.len(), len: w.len, ..RawRead::default() };
    for &(l, h) in heads {
        let lg = probe.logits(l, h).ok_or_else(|| bad(format!("layer {l} head {h}: no raw scores recorded")))?;
        out.logits.push(lg.logits);
        out.lse.push(lg.lse);
    }
    Ok(out)
}

/// Rounding tolerance of the even-share comparison in `low_signal` (an exactly even read must
/// leave nothing out; not an operating number).
const EVEN_TOLERANCE: f64 = 1e-6;

/// Positions left out of the chunk sums, per head (`true`: left out), and how the rule ran.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mask {
    pub out: Vec<Vec<bool>>,
    /// Positions left out summed over heads, and positions considered (heads x chunk positions).
    pub left_out: usize,
    pub considered: usize,
    pub why: String,
}
impl Mask {
    fn is_out(&self, h: usize, p: usize) -> bool {
        self.out.get(h).and_then(|v| v.get(p)).copied().unwrap_or(false)
    }
    /// True when any head leaves position `p` out.
    #[must_use]
    pub fn any_head(&self, p: usize) -> bool {
        self.out.iter().any(|v| v.get(p).copied().unwrap_or(false))
    }
}

/// Low-signal positions from the content-free read (no hand list): per drone head, a position of
/// the window's chunks is left out when the head's attention on it with no question is above the
/// level that head would give it if the content-free attention on the chunks were spread evenly
/// over their positions times the head's own concentration, that is when it is one of the
/// positions that carry the head's content-free attention. The head's concentration is measured,
/// not typed: with `p` the head's content-free attention over the chunk positions normalised to 1
/// and `N_eff = exp(entropy(p))` the effective number of positions carrying it, the positions left
/// out are those with `p > 1 / N_eff` (above the even share among the positions that carry
/// attention at all). A head whose content-free attention is even (`N_eff` near the position
/// count) leaves nothing out. `parts` is the window's `(first position, tokens)` list, anchor
/// first; the anchor is never left out (it is the window's baseline).
#[must_use]
pub fn low_signal(null: &RawRead, parts: &[(usize, usize)]) -> Mask {
    let mass = null.mass();
    let chunk_pos: Vec<usize> = parts.iter().skip(1).flat_map(|&(at, n)| at..at + n).collect();
    let mut out = Vec::with_capacity(mass.len());
    let (mut left, mut eff_sum) = (0_usize, 0.0_f64);
    for m in &mass {
        let total: f64 = chunk_pos.iter().map(|&p| f64::from(m.get(p).copied().unwrap_or(0.0))).sum();
        let mut row = vec![false; null.len];
        if total > 0.0 {
            let h: f64 = chunk_pos
                .iter()
                .map(|&p| f64::from(m.get(p).copied().unwrap_or(0.0)) / total)
                .filter(|x| *x > 0.0)
                .map(|x| -x * x.ln())
                .sum();
            let n_eff = h.exp();
            eff_sum += n_eff;
            for &p in &chunk_pos {
                if f64::from(m.get(p).copied().unwrap_or(0.0)) / total * n_eff > 1.0 + EVEN_TOLERANCE
                    && let Some(x) = row.get_mut(p)
                {
                    *x = true;
                    left += 1;
                }
            }
        }
        out.push(row);
    }
    let considered = mass.len() * chunk_pos.len();
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let (share, eff) = (left as f64 / considered.max(1) as f64, eff_sum / mass.len().max(1) as f64);
    Mask {
        out,
        left_out: left,
        considered,
        why: format!(
            "positions above the even share among the positions carrying each head's content-free attention: {left} of {considered} head positions left out ({:.1}%), mean effective positions per head {eff:.0} of {}",
            share * 100.0,
            chunk_pos.len()
        ),
    }
}

/// A chunk score taken from raw reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The first form: per head, log-sum-exp over the chunk, an equal vote per token, posed minus
    /// content-free, summed over heads, above the anchor (`tentacle_pass::Score::Calibrated`).
    Calibrated,
    /// Posed minus content-free attention mass on the chunk, summed over heads.
    Mass,
    /// Attention-weighted votes: as `Calibrated`, but token `t` of head `h` votes with weight
    /// `w(t, h)`, the share of its posed attention that lands on the window's chunks; the
    /// content-free side is weighted the same way by its own read; the heads' sums are divided by
    /// the total weight and multiplied by the head count, so the unit stays nats summed over the
    /// heads. Above the anchor.
    Weighted,
}

/// The score of every chunk of `parts` (anchor first in `parts`, excluded from the result) from a
/// posed and a content-free raw read of the same window. `mask` leaves positions out of every
/// chunk sum of both reads (the anchor's positions are never left out).
#[must_use]
pub fn chunk_scores(posed: &RawRead, null: &RawRead, parts: &[(usize, usize)], kind: Kind, mask: Option<&Mask>) -> Vec<f32> {
    let heads = posed.heads().min(null.heads());
    let keep = |h: usize, k: usize, p: usize| k == 0 || mask.is_none_or(|m| !m.is_out(h, p));
    // Log-sum-exp of head `h`, token `t` over the kept positions of part `k` (`None`: none kept).
    let lse = |r: &RawRead, h: usize, t: usize, k: usize| -> Option<f32> {
        let &(at, n) = parts.get(k)?;
        let row = r.logits.get(h)?;
        let mut max = f32::NEG_INFINITY;
        for p in at..at + n {
            if keep(h, k, p) {
                max = max.max(row.get(t * r.len + p).copied().unwrap_or(f32::NEG_INFINITY));
            }
        }
        if !max.is_finite() {
            return None;
        }
        let s: f32 = (at..at + n)
            .filter(|p| keep(h, k, *p))
            .map(|p| (row.get(t * r.len + p).copied().unwrap_or(f32::NEG_INFINITY) - max).exp())
            .sum();
        Some(max + s.ln())
    };
    // Share of token `t`'s attention (head `h`) on the kept chunk positions.
    let gain = |r: &RawRead, h: usize, t: usize| -> f32 {
        parts
            .iter()
            .enumerate()
            .skip(1)
            .map(|(k, &(at, n))| (at..at + n).filter(|p| keep(h, k, *p)).map(|p| r.weight(h, t, p)).sum::<f32>())
            .sum()
    };
    let n = parts.len();
    let per_part: Vec<f32> = match kind {
        Kind::Mass => (0..n)
            .map(|k| {
                let Some(&(at, len)) = parts.get(k) else { return 0.0 };
                (0..heads)
                    .map(|h| {
                        let side = |r: &RawRead| -> f32 {
                            #[expect(clippy::cast_precision_loss, reason = "token counts are small")]
                            let per = 1.0 / r.tokens.max(1) as f32;
                            (at..at + len)
                                .filter(|p| keep(h, k, *p))
                                .map(|p| (0..r.tokens).map(|t| r.weight(h, t, p)).sum::<f32>())
                                .sum::<f32>()
                                * per
                        };
                        side(posed) - side(null)
                    })
                    .sum()
            })
            .collect(),
        Kind::Calibrated | Kind::Weighted => {
            let weighted = kind == Kind::Weighted;
            // Per head: token weights of each read (equal votes: 1 per token).
            let weights =
                |r: &RawRead, h: usize| -> Vec<f32> { (0..r.tokens).map(|t| if weighted { gain(r, h, t) } else { 1.0 }).collect() };
            let mut acc = vec![0.0_f32; n];
            let mut total = 0.0_f32;
            for h in 0..heads {
                let (wp, wn) = (weights(posed, h), weights(null, h));
                let (sp, sn): (f32, f32) = (wp.iter().sum(), wn.iter().sum());
                // The head's weight is its posed weight; an unweighted head counts once.
                let head_w = if weighted { sp } else { 1.0 };
                total += head_w;
                for (k, a) in acc.iter_mut().enumerate() {
                    let mean = |r: &RawRead, w: &[f32], s: f32| -> f32 {
                        if s <= 0.0 {
                            return 0.0;
                        }
                        (0..r.tokens).filter_map(|t| lse(r, h, t, k).map(|x| x * w.get(t).copied().unwrap_or(0.0))).sum::<f32>() / s
                    };
                    *a += head_w * (mean(posed, &wp, sp) - mean(null, &wn, sn));
                }
            }
            #[expect(clippy::cast_precision_loss, reason = "a head count")]
            let scale = if weighted && total > 0.0 { heads as f32 / total } else { 1.0 };
            acc.into_iter().map(|x| x * scale).collect()
        }
    };
    let anchor = if kind == Kind::Mass { 0.0 } else { per_part.first().copied().unwrap_or(0.0) };
    per_part.into_iter().skip(1).map(|x| x - anchor).collect()
}

// ---------------------------------------------------------------------------------------------
// Component 3: the length-matched content-free pose.
// ---------------------------------------------------------------------------------------------

/// The frame every pose starts with, and the content-free word (the first form's pose is
/// `FRAME` followed by one `FILLER`).
pub const FRAME: &str = "\nQuestion:";
pub const FILLER: &str = " N/A";

/// The first form's content-free pose for a posed text ending in `suffix` (the focus line, or
/// empty).
#[must_use]
pub fn plain_null(suffix: &str) -> String {
    format!("{FRAME}{FILLER}{suffix}")
}

/// A content-free pose with the token count of `posed`: the same frame (`FRAME`), the neutral
/// filler (`FILLER`, repeated, the last repeat cut short when needed) and the same `suffix` (the
/// focus line of a re-read, or empty). `count` gives a text's token count. Returns the pose, its
/// token count and how it was made; when no filler text reaches the exact count the nearest
/// shorter one is returned and said so (the plain pose when even that fails).
#[must_use]
pub fn matched_null(count: &dyn Fn(&str) -> Option<usize>, posed: &str, suffix: &str) -> (String, usize, String) {
    let plain = plain_null(suffix);
    let (Some(target), Some(base)) = (count(posed), count(&plain)) else {
        return (plain, 0, "plain pose: the tokenizer gave no count".to_owned());
    };
    if target <= base {
        return (plain, base, format!("plain pose: the posed text has {target} tokens, the plain pose {base}"));
    }
    // Pieces the filler can grow by, longest first: the whole word, then its prefixes.
    let cuts: Vec<&str> = (1..=FILLER.len())
        .rev()
        .filter(|i| FILLER.is_char_boundary(*i))
        .filter_map(|i| FILLER.get(..i))
        .filter(|s| !s.trim().is_empty())
        .collect();
    let mut body = FILLER.to_owned();
    let mut have = base;
    while have < target {
        let mut grown = false;
        for c in &cuts {
            let cand = format!("{body}{c}");
            if let Some(n) = count(&format!("{FRAME}{cand}{suffix}")).filter(|n| *n > have && *n <= target) {
                body = cand;
                have = n;
                grown = true;
                break;
            }
        }
        if !grown {
            break;
        }
    }
    let text = format!("{FRAME}{body}{suffix}");
    let why = if have == target {
        format!("length-matched pose: {have} tokens, equal to the posed text")
    } else {
        format!("nearest shorter pose: {have} tokens against {target} posed (no filler text gave the exact count)")
    };
    (text, have, why)
}

// ---------------------------------------------------------------------------------------------
// Section checks.
// ---------------------------------------------------------------------------------------------

fn write_report(name: &str, rep: &str, r: &Result<(), String>) -> Result<(), String> {
    let dir = nmtype::check_dir(name).map_err(|e| e.to_string())?;
    let text = format!("{rep}result: {}\n", r.as_ref().map_or_else(|e| format!("FAIL {e}"), |()| "ok".to_owned()));
    std::fs::write(dir.join("report.txt"), text).map_err(|e| e.to_string())
}

/// Section check of component 1 (processor only, hand values): the shortlist size rule (noise
/// floor from the negative scores, fallback, bounds), the pick in window order, and the order
/// after a second look (picked chunks first by their own scores, the rest by the full read,
/// nothing lost). Values: `tests/section-checks/nmapi-core-read-look/report.txt`.
/// # Errors
/// One fault line.
pub fn check_look() -> Result<(), String> {
    use std::fmt::Write as _;
    let mut rep = String::from("nmapi.core_read.look (second look: shortlist size, pick, order)\n");
    let r = (|| -> Result<(), String> {
        let mass = [0.30_f32, 0.05, -0.04, 0.28, 0.27, 0.01, -0.02];
        let (k, why) = look_size(&mass);
        let _ = writeln!(rep, "size on {mass:?}: {k} ({why})");
        let (kf, whyf) = look_size(&[0.3, 0.2, 0.1]);
        let (kn, _) = look_size(&[-0.1, -0.2]);
        let (k2, _) = look_size(&[0.9, 0.1, -0.01, 0.0]);
        let _ = writeln!(rep, "no negative score: {kf} ({whyf}); none positive: {kn}; one clear best: {k2} (at least 2)");
        if k != 3 || kf != 3 || kn != 2 || k2 != 2 {
            return Err(format!("look_size gave {k}, {kf}, {kn}, {k2}; expected 3, 3, 2, 2"));
        }
        let pick = look_pick(&mass, k);
        let order = look_order(&[5.0, 9.0, 1.0, 4.0, 3.0, 2.0, 0.0], &pick, &[0.1, 0.7, 0.4]);
        let _ = writeln!(rep, "pick {pick:?} (window order); order after the look {order:?}");
        if pick != [0, 3, 4] || order != [3, 4, 0, 1, 5, 2, 6] {
            return Err(format!("pick {pick:?} or order {order:?} wrong"));
        }
        let mut all = order.clone();
        all.sort_unstable();
        if all != (0..7).collect::<Vec<_>>() {
            return Err("the order after a look lost or doubled a chunk".into());
        }
        Ok(())
    })();
    write_report("nmapi-core-read-look", &rep, &r)?;
    r
}

/// A hand raw read: one head, `tokens x len` scores, each token seeing only the window.
fn hand(tokens: usize, len: usize, logits: Vec<f32>) -> RawRead {
    let lse = (0..tokens)
        .map(|t| {
            let row = logits.get(t * len..(t + 1) * len).unwrap_or(&[]);
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            m + row.iter().map(|x| (x - m).exp()).sum::<f32>().ln()
        })
        .collect();
    RawRead { tokens, len, logits: vec![logits], lse: vec![lse] }
}

/// Section check of component 2 (processor only, hand values): (a) the equal-vote score from raw
/// reads equals `tentacle_pass::part_lse` and `calibrated`; (b) with one token whose attention
/// lands on the chunks and one that parks on the anchor, the weighted vote follows the first and
/// the equal vote does not; (c) a position that draws the content-free attention is left out by
/// the rule and a flat content-free read leaves nothing out; (d) leaving it out changes the mass
/// order as expected. Values: `tests/section-checks/nmapi-core-read-votes/report.txt`.
/// # Errors
/// One fault line.
pub fn check_votes() -> Result<(), String> {
    use std::fmt::Write as _;
    let mut rep = String::from("nmapi.core_read.votes (attention-weighted votes, low-signal positions)\n");
    let r = (|| -> Result<(), String> {
        // Window: anchor at 0..2, chunk A at 2..5, chunk B at 5..8.
        let parts = [(0_usize, 2_usize), (2, 3), (5, 3)];
        // Token 0 reads the chunks and prefers B; token 1 parks on the anchor and, among its
        // small remainder, prefers A strongly in log space.
        let posed = hand(2, 8, vec![0.0, 0.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 9.0, 9.0, 2.0, 2.0, 2.0, -3.0, -3.0, -3.0]);
        let null = hand(2, 8, vec![0.0; 16]);
        let eq = chunk_scores(&posed, &null, &parts, Kind::Calibrated, None);
        let wt = chunk_scores(&posed, &null, &parts, Kind::Weighted, None);
        let ms = chunk_scores(&posed, &null, &parts, Kind::Mass, None);
        let _ = writeln!(rep, "(b) equal votes {eq:?}; weighted votes {wt:?}; mass {ms:?} (chunks A, B)");
        let old_p = tentacle_pass::part_lse(posed.logits.first().map_or(&[], Vec::as_slice), 2, 8, &parts);
        let old_n = tentacle_pass::part_lse(null.logits.first().map_or(&[], Vec::as_slice), 2, 8, &parts);
        let old = tentacle_pass::calibrated(&old_p, &old_n, 3);
        let old: Vec<f32> = old.iter().skip(1).map(|x| x - old.first().copied().unwrap_or(0.0)).collect();
        let close = |a: &[f32], b: &[f32]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4);
        let _ = writeln!(rep, "(a) equal votes from raw reads against the pass's calibrated score: {eq:?} and {old:?}");
        if !close(&eq, &old) {
            return Err("the equal-vote score from raw reads differs from the pass's calibrated score".into());
        }
        let first = |v: &[f32]| v.first().copied().unwrap_or(0.0) > v.get(1).copied().unwrap_or(0.0);
        if !first(&eq) || first(&wt) || first(&ms) {
            return Err(
                "expected the equal vote to prefer A (the parked token's vote) and the weighted vote and the mass to prefer B".into()
            );
        }
        // (c) Content-free read: position 3 (in A) draws the attention; the rule leaves it out.
        let sink = hand(1, 8, vec![0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
        let m = low_signal(&sink, &parts);
        let flat = low_signal(&null, &parts);
        let _ = writeln!(rep, "(c) sink read: {}; flat read: {}", m.why, flat.why);
        let outs: Vec<usize> = (0..8).filter(|p| m.any_head(*p)).collect();
        if outs != [3] || flat.left_out != 0 {
            return Err(format!("low-signal rule left out {outs:?} (expected [3]) and {} on a flat read (expected 0)", flat.left_out));
        }
        // (d) A posed read that also loads position 3: with it left out, A scores nothing above
        // the anchor and B stays first.
        let posed2 = hand(1, 8, vec![0.0, 0.0, 0.0, 4.5, 0.0, 1.0, 1.0, 1.0]);
        let with = chunk_scores(&posed2, &sink, &parts, Kind::Weighted, None);
        let without = chunk_scores(&posed2, &sink, &parts, Kind::Weighted, Some(&m));
        let _ = writeln!(rep, "(d) weighted votes with the position {with:?}, with it left out {without:?}");
        if first(&without) || !without.iter().all(|x| x.is_finite()) {
            return Err("with the low-signal position left out the score should prefer B".into());
        }
        Ok(())
    })();
    write_report("nmapi-core-read-votes", &rep, &r)?;
    r
}

/// Section check of component 3 (the base model's tokenizer, no weights read, no device): for
/// posed texts of several lengths, with and without a focus line, the matched pose has exactly the
/// posed token count, starts with the frame, ends with the suffix and holds only filler between; a
/// posed text no longer than the plain pose keeps the plain pose; the pose text differs with the
/// length (so a kept no-question read keyed by its text is keyed by the length). Values:
/// `tests/section-checks/nmapi-core-read-pose/report.txt`.
/// # Errors
/// One fault line.
pub fn check_pose() -> Result<(), String> {
    use std::fmt::Write as _;
    let mut rep = String::from("nmapi.core_read.pose (length-matched content-free pose)\n");
    let r = (|| -> Result<(), String> {
        let home = nmtype::nano_root();
        let pop = nmengn::model::Population::open(&home, crate::neural::POPULATION, false, false).map_err(|e| e.to_string())?;
        let count = |s: &str| pop.tokenizer.encode(s).ok().map(|v| v.len());
        let words = [
            "where", "is", "the", "free", "room", "of", "a", "limit", "computed", "when", "the", "window", "is", "full", "and", "which",
            "function", "returns", "it", "to", "the", "caller", "?",
        ];
        let mut seen = std::collections::BTreeSet::new();
        for n in [1_usize, 4, 9, 15, 23] {
            for suffix in ["", "\nFocus: files a/limit.rs"] {
                let posed = format!("{FRAME} {}{suffix}", words.iter().take(n).copied().collect::<Vec<_>>().join(" "));
                let target = count(&posed).ok_or("no token count")?;
                let (text, have, why) = matched_null(&count, &posed, suffix);
                let _ = writeln!(
                    rep,
                    "{n} words{}: posed {target} tokens, content-free {have} ({why})",
                    if suffix.is_empty() { "" } else { " with a focus line" }
                );
                let body = text.strip_prefix(FRAME).and_then(|t| t.strip_suffix(suffix)).ok_or("the pose lost its frame or suffix")?;
                // A posed text shorter than the plain pose keeps the plain pose (nothing shorter
                // exists in the frame).
                if target < count(&plain_null(suffix)).unwrap_or(0) {
                    if text != plain_null(suffix) {
                        return Err(format!("{n} words: a posed text shorter than the plain pose should keep the plain pose"));
                    }
                    continue;
                }
                if have != target || count(&text) != Some(target) {
                    return Err(format!("{n} words: content-free pose has {have} tokens, posed {target}"));
                }
                if !body.chars().all(|c| FILLER.contains(c)) {
                    return Err("the pose holds something other than filler".into());
                }
                if suffix.is_empty() {
                    seen.insert(text);
                }
            }
        }
        if seen.len() != 4 {
            return Err(format!("{} distinct poses for 4 lengths at or above the plain pose", seen.len()));
        }
        let (short, n, why) = matched_null(&count, &plain_null(""), "");
        let _ = writeln!(rep, "posed text of the plain pose's length: {n} tokens ({why})");
        if short != plain_null("") {
            return Err("a posed text no longer than the plain pose should keep the plain pose".into());
        }
        Ok(())
    })();
    write_report("nmapi-core-read-pose", &rep, &r)?;
    r
}
