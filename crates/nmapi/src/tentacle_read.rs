//! The tentacle's read as decided by Phase A (2026-11-18; tentacle master plan 3.3 steps 4 and 5,
//! section 7 "Decided by Phase A"; architecture by Luis Jake Gabriel III). Lane T1, Phases D3 and D4.
//! - `shortlist` (D3): per tentacle, the pieces of its window by the fused wording and meaning
//!   evidence (each kind standardised within the window, summed), the softmax of that sum, the
//!   smallest set holding `mass` of it in fused order, cut to the card tokens the call grants the
//!   tentacle (the leader always kept). On the TC1 dump (20 questions, 123 windows) mass 0.99 and
//!   1,500 tokens reach the answer on 19 of 19 (median 4 pieces, 1,456 tokens).
//! - `in_band` (the gate): a tentacle integrates when its fused leader's margin over the second is
//!   inside the noise band.
//! - `Integrator::read` (D4): the shortlist read on the card through the relevance layers behind
//!   the anchor, pieces in relation order (`relation_order`), joined from layer 0 (each piece encoded
//!   behind the anchor and the pieces before it: the plain joint read, in-context read form), scored
//!   per piece at the tentacle's drone heads by the log-sum-exp of the raw scores, posed minus
//!   content-free, summed over heads, above the anchor. TN-1 on the TC1 dump: the joint form leads
//!   the pieces-alone form (first in window 14 against 13, across windows 7 against 6 of 20); the
//!   drones' design form (layers 0 to k alone, then k+1 to d joined) needs an engine entry at layer
//!   k+1 and is not built.
//!
//! - `plan` and `served` (D7): the read in the served flow (`drone_path::stack_core`), behind
//!   `NANO_TENTACLE_READ=on` (off by default): after a round's windows each tentacle forms its
//!   shortlist; those whose leader's margin is inside `GATE_BAND` integrate, closest first, in one
//!   packed card run under `CALL_TOKENS_FALLBACK` tokens a call; the scores enter G1 as the score
//!   set `read` (family `reading`).
//!
//! RAM only; nothing is written.
use crate::tentacle_pass;
use nmengn::kvstore::{self, ChunkKv, Window};
use nmengn::model::Model;
use nmengn::placement::Device;
use nmtype::{Fault, FaultKind};
use std::collections::BTreeSet;
use std::sync::Arc;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmapi.tentacle_read", FaultKind::Execution, m)
}

/// Shortlist mass and the grant used when the allocator gives none (D3 on the TC1 dump).
pub const MASS: f32 = 0.99;
pub const GRANT_FALLBACK: usize = 1_500;

fn standardised(v: &[f32]) -> Vec<f32> {
    #[expect(clippy::cast_precision_loss, reason = "a window's piece count")]
    let n = v.len().max(1) as f32;
    let m = v.iter().sum::<f32>() / n;
    let s = (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / n).sqrt();
    let s = if s > 0.0 { s } else { 1.0 };
    v.iter().map(|x| (x - m) / s).collect()
}

/// The fused evidence of a window's pieces: each kind (one list per kind, in piece order)
/// standardised within the window, summed.
#[must_use]
pub fn fuse(kinds: &[&[f32]]) -> Vec<f32> {
    let n = kinds.first().map_or(0, |k| k.len());
    let mut out = vec![0.0_f32; n];
    for k in kinds {
        for (o, x) in out.iter_mut().zip(standardised(k)) {
            *o += x;
        }
    }
    out
}

/// D3: indices (into the window's pieces) of the shortlist, best first: by `fused`, the smallest
/// set whose softmax mass reaches `mass`, skipping a piece that would pass `grant` tokens (the
/// leader always kept).
#[must_use]
pub fn shortlist(fused: &[f32], tokens: &[usize], mass: f32, grant: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..fused.len()).collect();
    order.sort_by(|a, b| fused.get(*b).copied().unwrap_or(f32::MIN).total_cmp(&fused.get(*a).copied().unwrap_or(f32::MIN)));
    let mx = fused.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = fused.iter().map(|x| (x - mx).exp()).collect();
    let tot: f32 = e.iter().sum();
    let (mut out, mut held, mut used) = (Vec::new(), 0.0_f32, 0_usize);
    for i in order {
        if held >= mass && !out.is_empty() {
            break;
        }
        let t = tokens.get(i).copied().unwrap_or(0);
        if !out.is_empty() && used + t > grant {
            continue;
        }
        out.push(i);
        held += e.get(i).copied().unwrap_or(0.0) / tot.max(f32::MIN_POSITIVE);
        used += t;
    }
    out
}

/// The gate: the fused leader's margin over the second is inside the noise band `band`.
#[must_use]
pub fn in_band(fused: &[f32], band: f32) -> bool {
    let mut v: Vec<f32> = fused.to_vec();
    v.sort_by(|a, b| b.total_cmp(a));
    matches!(v.as_slice(), [a, b, ..] if a - b < band)
}

/// Relation order of `list` (substrate chunk indices, best first): the leader first, then pieces
/// related to those already placed (breadth first over the specific reference graph), then the rest
/// by path and first line.
#[must_use]
pub fn relation_order(sub: &nmsubs::Substrate, list: &[usize]) -> Vec<usize> {
    let g = sub.refs();
    let want: BTreeSet<usize> = list.iter().copied().collect();
    let mut out: Vec<usize> = list.first().copied().into_iter().collect();
    let mut i = 0;
    while let Some(&c) = out.get(i) {
        for e in g.specific.edges_of(c) {
            let t = e.to as usize;
            if want.contains(&t) && !out.contains(&t) {
                out.push(t);
            }
        }
        i += 1;
    }
    let mut rest: Vec<usize> = list.iter().copied().filter(|c| !out.contains(c)).collect();
    rest.sort_by_key(|c| sub.chunks.get(*c).map(|ch| (sub.files.get(ch.file).map(|f| f.path.clone()).unwrap_or_default(), ch.start)).unwrap_or_default());
    out.extend(rest);
    out
}

/// One tentacle's integration read request: its pieces' texts in relation order, its relevance
/// heads, its pose.
pub type Item = (Vec<String>, Vec<(usize, usize)>, String);

/// D4: the integration reader of one model and anchor.
pub struct Integrator {
    pub model: Arc<Model>,
    pub dev: Device,
    last: usize,
    stop: Vec<(String, String)>,
    ak: Arc<ChunkKv>,
    aw: Arc<Window>,
}
impl std::fmt::Debug for Integrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Integrator").field("device", &self.dev.name()).field("last", &self.last).finish_non_exhaustive()
    }
}

/// One integration read: per piece (in the order read) the calibrated score above the anchor, the
/// tokens read and the milliseconds of the encode and of the two question reads.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Read {
    pub scores: Vec<f32>,
    /// Per piece the log of the posed-to-content-free ratio averaged over the heads (D6's
    /// no-information statistic, the anchor not taken off).
    pub ratio: Vec<f32>,
    pub tokens: usize,
    pub enc_ms: f64,
    pub read_ms: f64,
}

impl Integrator {
    /// The reader `model` on `dev` with `anchor` encoded through the deepest relevance layer `last`.
    /// # Errors
    /// Engine faults.
    pub fn new(model: Arc<Model>, dev: Device, anchor: &str, last: usize) -> Result<Self, Fault> {
        let stop = kvstore::score_targets(&model, last)?;
        let ids = model.pop.tokenizer.encode(anchor).map_err(|e| bad(e.to_string()))?;
        let ak = Arc::new(kvstore::encode_through(&model, dev, None, &ids, last)?);
        let aw = Arc::new(Window::build(&model, &[Arc::clone(&ak)])?);
        Ok(Self { model, dev, last, stop, ak, aw })
    }
    /// The window of `texts` behind the anchor: joined from layer 0 (`joint`), or each piece alone
    /// behind the anchor. Returns the window, tokens and encode milliseconds.
    /// # Errors
    /// Engine or tokenizer faults.
    pub fn window(&self, texts: &[String], joint: bool) -> Result<(Arc<Window>, usize, f64), Fault> {
        let t = std::time::Instant::now();
        let mut parts = vec![Arc::clone(&self.ak)];
        let mut toks = 0;
        for s in texts {
            let ids = self.model.pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()))?;
            toks += ids.len();
            let prefix = if joint { Arc::new(Window::build(&self.model, &parts)?) } else { Arc::clone(&self.aw) };
            let gate = (self.dev == Device::Gpu).then(|| nmengn::placement::gpu_gate().enter("tentacle integration read", true));
            let kv = kvstore::encode_through(&self.model, self.dev, Some(&prefix), &ids, self.last)?;
            drop(gate);
            parts.push(Arc::new(kv));
        }
        Ok((Arc::new(Window::build(&self.model, &parts)?), toks, t.elapsed().as_secs_f64() * 1000.0))
    }
    /// The integration reads of several tentacles in one packed card run (lane S's
    /// `kvstore::read_group_parts`, 2026-11-18): per tentacle one group, its joined pieces run once
    /// after the anchor and the posed and content-free questions as two tails over them, at that
    /// tentacle's own heads, the part log-sum-exp reduced on the card (parts: the anchor's, then each
    /// piece's position and tokens), inside a served scope; scored as `read` scores the joint form.
    /// On the card only; elsewhere each tentacle is read with `read` (joint form).
    /// # Errors
    /// Engine or tokenizer faults.
    pub fn read_packed(&self, items: &[Item]) -> Result<Vec<Read>, Fault> {
        // The packed run needs the one-launch packed attention (`serve` turns it on); without it
        // each tentacle is read in the sequential joint form.
        if self.dev != Device::Gpu || !nmengn::gpu::SEGMENT_ATTENTION.load(std::sync::atomic::Ordering::Relaxed) {
            return items.iter().map(|(t, h, p)| self.read(t, h, p, true)).collect();
        }
        let enc = |s: &str| self.model.pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
        let null = enc("
Question: N/A")?;
        let mut groups = Vec::with_capacity(items.len());
        let mut tokens = Vec::with_capacity(items.len());
        for (texts, heads, pose) in items {
            let mut pieces = Vec::new();
            let mut parts = self.aw.parts.clone();
            let mut at = self.aw.len;
            for t in texts {
                let x = enc(t)?;
                parts.push((at, x.len()));
                at += x.len();
                pieces.extend(x);
            }
            tokens.push(pieces.len());
            groups.push(kvstore::PartGroup { pieces, questions: vec![enc(&format!("
Question: {pose}"))?, null.clone()], heads: heads.clone(), parts });
        }
        let t = std::time::Instant::now();
        let gate = nmengn::placement::gpu_gate().enter("tentacle integration reads (packed)", true);
        let rows = {
            let _scope = nmengn::gpu::ServedScope::new();
            kvstore::read_group_parts(&self.model, Some(&self.aw), &self.stop, &groups)?
        };
        drop(gate);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        #[expect(clippy::cast_precision_loss, reason = "a share of the run's time")]
        let share = ms / items.len().max(1) as f64;
        let mut out = Vec::with_capacity(items.len());
        for ((g, r), n) in groups.iter().zip(&rows).zip(&tokens) {
            let (Some(posed), Some(null)) = (r.first(), r.get(1)) else { return Err(bad("a group's rows were lost")) };
            let scores = above_anchor_parts(posed, null, g.parts.len(), self.aw.parts.len());
            let ratio = mean_ratio(posed, null, g.parts.len(), self.aw.parts.len());
            out.push(Read { scores, ratio, tokens: *n, enc_ms: 0.0, read_ms: share });
        }
        Ok(out)
    }
    /// Roadmap Phase 0 item 5 (WP-C5) probe (`NANO_HEAD_PROBE`, a measurement): the same packed
    /// read of the same windows at every head of the layers they are encoded through, each head's
    /// calibrated score per piece above the anchor (as `above_anchor` per head); one JSON line per
    /// window with its lens `names[i]` and pieces `refs[i]`, appended to `file`. On the card with
    /// the packed attention only; elsewhere nothing.
    /// # Errors
    /// Engine or tokenizer faults.
    pub fn probe_heads(&self, items: &[Item], names: &[String], refs: &[Vec<String>], heads_per_layer: usize, file: &std::path::Path) -> Result<(), Fault> {
        if self.dev != Device::Gpu || !nmengn::gpu::SEGMENT_ATTENTION.load(std::sync::atomic::Ordering::Relaxed) || items.is_empty() {
            return Ok(());
        }
        // Layers 8 to the deepest encoded (every head of the default set and of the Mini's E2 probe
        // lies in layers 8 to 17); `NANO_HEAD_PROBE_FROM` sets the first layer.
        let from = std::env::var("NANO_HEAD_PROBE_FROM").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(8).min(self.last);
        let all: Vec<(usize, usize)> = (from..=self.last).flat_map(|l| (0..heads_per_layer).map(move |h| (l, h))).collect();
        let enc = |s: &str| self.model.pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
        let null = enc("\nQuestion: N/A")?;
        let mut groups = Vec::with_capacity(items.len());
        for (texts, _, pose) in items {
            let mut pieces = Vec::new();
            let mut parts = self.aw.parts.clone();
            let mut at = self.aw.len;
            for t in texts {
                let x = enc(t)?;
                parts.push((at, x.len()));
                at += x.len();
                pieces.extend(x);
            }
            groups.push(kvstore::PartGroup { pieces, questions: vec![enc(&format!("\nQuestion: {pose}"))?, null.clone()], heads: all.clone(), parts });
        }
        let gate = nmengn::placement::gpu_gate().enter("tentacle head probe", true);
        let rows = {
            let _scope = nmengn::gpu::ServedScope::new();
            kvstore::read_group_parts(&self.model, Some(&self.aw), &self.stop, &groups)?
        };
        drop(gate);
        let mut out = String::new();
        let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
        for (i, (g, r)) in groups.iter().zip(&rows).enumerate() {
            let (Some(posed), Some(nul)) = (r.first(), r.get(1)) else { continue };
            let parts = g.parts.len();
            let anchor = self.aw.parts.len();
            let pose = items.get(i).map_or("", |x| x.2.as_str());
            let mut mass = Vec::with_capacity(all.len());
            for h in 0..all.len() {
                let cal = |k: usize| posed.get(h * parts + k).copied().unwrap_or(0.0) - nul.get(h * parts + k).copied().unwrap_or(0.0);
                let base = cal(0);
                let row: Vec<String> = (anchor..parts).map(|k| format!("{:.5}", cal(k) - base)).collect();
                mass.push(format!("[{}]", row.join(",")));
            }
            out.push_str(&format!(
                "{{\"question\":\"{}\",\"lens\":\"{}\",\"parts\":[{}],\"heads\":[{}],\"mass\":[{}]}}\n",
                esc(pose),
                esc(names.get(i).map_or("", String::as_str)),
                refs.get(i).map_or_else(String::new, |v| v.iter().map(|x| format!("\"{}\"", esc(x))).collect::<Vec<_>>().join(",")),
                all.iter().map(|(l, h)| format!("[{l},{h}]")).collect::<Vec<_>>().join(","),
                mass.join(",")
            ));
        }
        std::fs::OpenOptions::new().create(true).append(true).open(file).and_then(|mut f| std::io::Write::write_all(&mut f, out.as_bytes())).map_err(|e| bad(e.to_string()))
    }
    /// The integration read of `texts` (in relation order) under `pose` at `heads`, one window and
    /// two question reads (the sequential form; `read_packed` packs several on the card).
    /// # Errors
    /// Engine or tokenizer faults.
    pub fn read(&self, texts: &[String], heads: &[(usize, usize)], pose: &str, joint: bool) -> Result<Read, Fault> {
        let (w, tokens, enc_ms) = self.window(texts, joint)?;
        let t = std::time::Instant::now();
        let gate = (self.dev == Device::Gpu).then(|| nmengn::placement::gpu_gate().enter("tentacle integration read", true));
        let (_, posed) = tentacle_pass::drone_read_logits(&self.model, self.dev, &w, &self.stop, heads, &format!("\nQuestion: {pose}"))?;
        let (_, null) = tentacle_pass::drone_read_logits(&self.model, self.dev, &w, &self.stop, heads, "\nQuestion: N/A")?;
        drop(gate);
        let ratio = mean_ratio(&posed, &null, w.parts.len(), 1);
        Ok(Read { scores: above_anchor(&posed, &null, w.parts.len()), ratio, tokens, enc_ms, read_ms: t.elapsed().as_secs_f64() * 1000.0 })
    }
}

/// As `above_anchor` with the anchor spread over its first `anchor_parts` parts (their scores
/// summed in log space would differ; the anchor of `Integrator` is one part, so this is the same).
#[must_use]
pub fn above_anchor_parts(posed: &[f32], null: &[f32], parts: usize, anchor_parts: usize) -> Vec<f32> {
    let skip = anchor_parts.max(1) - 1;
    let all = above_anchor(posed, null, parts);
    all.into_iter().skip(skip).collect()
}

/// Per part after the first `anchor_parts`: the mean over heads of posed minus content-free (the
/// log of the posed-to-null ratio over the heads).
#[must_use]
pub fn mean_ratio(posed: &[f32], null: &[f32], parts: usize, anchor_parts: usize) -> Vec<f32> {
    let heads = posed.len() / parts.max(1);
    #[expect(clippy::cast_precision_loss, reason = "a head count")]
    let per = 1.0 / heads.max(1) as f32;
    (anchor_parts..parts)
        .map(|k| (0..heads).map(|h| posed.get(h * parts + k).copied().unwrap_or(0.0) - null.get(h * parts + k).copied().unwrap_or(0.0)).sum::<f32>() * per)
        .collect()
}

/// D6's no-information threshold on `Read::ratio` (the best piece's mean log posed-to-null ratio
/// over the heads): measured on the TC1 dump's joint reads (20 questions, 123 windows,
/// 2026-11-18): the largest value that leaves at most 1 question with every answer-holding window
/// flagged; it flags 32 of the 100 windows whose shortlist lacks the answer. REP-1's 5 of 6 is not
/// reached by any threshold on these reads (`scratch/lane-t1/d6_threshold.py`).
pub const NO_INFORMATION: f32 = 1.25;

/// Per part after the anchor: the sum over heads of posed minus content-free, minus the anchor's.
#[must_use]
pub fn above_anchor(posed: &[f32], null: &[f32], parts: usize) -> Vec<f32> {
    let heads = posed.len() / parts.max(1);
    let cal: Vec<f32> = (0..parts)
        .map(|k| (0..heads).map(|h| posed.get(h * parts + k).copied().unwrap_or(0.0) - null.get(h * parts + k).copied().unwrap_or(0.0)).sum())
        .collect();
    let base = cal.first().copied().unwrap_or(0.0);
    cal.iter().skip(1).map(|x| x - base).collect()
}

/// D6: a tentacle's report of one integration read (master plan 3.3 steps 7 and 8): its pieces
/// best first with their calibrated evidence above the anchor, the margin of the first over the
/// second (a lone piece: over 0), the spread (best minus worst) and the no-information flag (master
/// plan 3.3 step 7): no piece's posed-to-null ratio over its heads passes `NO_INFORMATION`, so the
/// tentacle says it holds nothing for its aspect.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    pub best: Vec<(usize, f32)>,
    pub margin: f32,
    pub spread: f32,
    pub no_information: bool,
}

/// The report of a read: `pieces` (in the order read, e.g. substrate chunk indices), their
/// `scores` and `ratio` from `Read`; no information when no piece's ratio passes `NO_INFORMATION`.
#[must_use]
pub fn report(pieces: &[usize], scores: &[f32], ratio: &[f32]) -> Report {
    let mut best: Vec<(usize, f32)> = pieces.iter().copied().zip(scores.iter().copied()).collect();
    best.sort_by(|a, b| b.1.total_cmp(&a.1));
    let top = best.first().map_or(f32::NEG_INFINITY, |x| x.1);
    let second = best.get(1).map_or(0.0, |x| x.1);
    let low = best.last().map_or(top, |x| x.1);
    let best_ratio = ratio.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    Report { margin: top - second, spread: top - low, no_information: best_ratio.is_nan() || best_ratio <= NO_INFORMATION, best }
}

/// D7: integration tokens one call may spend on the card, over every round, when the call's
/// allocator grants none (master plan 3.3 step 5: one to three contested tentacles, 0.3 to 0.8 s
/// on the card; `NANO_READ_TOKENS` overrides for measurement).
pub const CALL_TOKENS_FALLBACK: usize = 2_250;
/// D7: the most one tentacle is granted of them in a round (a shortlist of about six pieces).
pub const TENTACLE_TOKENS_FALLBACK: usize = 750;
/// D7: the noise band of a window's fused leader margin (two standardised kinds summed): the
/// median margin over the TC1 dump's 123 windows, which integrated about half of them (stated
/// fallback until the margin's band is measured on the served flow).
pub const GATE_BAND: f32 = 1.106;

static SWITCH: std::sync::RwLock<Option<bool>> = std::sync::RwLock::new(None);
/// Sets the read's switch for this process (`None`: `NANO_TENTACLE_READ`).
pub fn set_on(on: Option<bool>) {
    if let Ok(mut g) = SWITCH.write() {
        *g = on;
    }
}
/// Whether the served flow runs the tentacles' read: `set_on`, else on unless
/// `NANO_TENTACLE_READ=off` (on by default since 2026-11-18, Luis: parts that advance the
/// architecture are not left off). The flow reads only with the reader on the card.
#[must_use]
pub fn on() -> bool {
    SWITCH.read().ok().and_then(|g| *g).unwrap_or_else(|| !std::env::var("NANO_TENTACLE_READ").is_ok_and(|v| v == "off"))
}

/// One window's decision in a round: its fused leader's margin, whether it is in the band, the
/// shortlist it integrates (window positions, best first; empty when it does not) and why.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plan {
    pub margin: f32,
    pub in_band: bool,
    pub list: Vec<usize>,
    pub why: &'static str,
}

/// D7's gate and grant over one round's windows: each window's `fused` evidence and piece
/// `tokens` (window order). Windows whose leader's margin is under `band` integrate, the
/// smallest margin first, each shortlist cut to `per` tokens or what is left of `budget` (at
/// least two pieces, else it does not integrate). Returns the plans and the tokens granted.
#[must_use]
pub fn plan(fused: &[Vec<f32>], tokens: &[Vec<usize>], band: f32, budget: usize, per: usize) -> (Vec<Plan>, usize) {
    let margin = |f: &[f32]| {
        let mut v: Vec<f32> = f.to_vec();
        v.sort_by(|a, b| b.total_cmp(a));
        match v.as_slice() {
            [a, b, ..] => a - b,
            _ => f32::INFINITY,
        }
    };
    let mut out: Vec<Plan> = fused
        .iter()
        .map(|f| {
            let m = margin(f);
            Plan { margin: m, in_band: m < band, list: Vec::new(), why: if m < band { "in band" } else { "clear leader" } }
        })
        .collect();
    let mut order: Vec<usize> = (0..out.len()).filter(|i| out.get(*i).is_some_and(|p| p.in_band)).collect();
    order.sort_by(|a, b| out.get(*a).map_or(0.0, |p| p.margin).total_cmp(&out.get(*b).map_or(0.0, |p| p.margin)).then(a.cmp(b)));
    let mut left = budget;
    for i in order {
        let (Some(f), Some(t)) = (fused.get(i), tokens.get(i)) else { continue };
        let grant = left.min(per);
        let list = shortlist(f, t, MASS, grant);
        let used: usize = list.iter().map(|k| t.get(*k).copied().unwrap_or(0)).sum();
        let Some(p) = out.get_mut(i) else { continue };
        if list.len() < 2 {
            p.why = if used > grant { "no card tokens left" } else { "shortlist of one piece" };
        } else if used > grant {
            p.why = "no card tokens left";
        } else {
            left -= used;
            p.list = list;
            p.why = "integrated";
        }
    }
    (out, budget - left)
}

/// The anchor the served reads are encoded behind, by `tentacle_pass`'s rule.
#[must_use]
pub fn anchor_text(sub: &nmsubs::Substrate) -> String {
    if sub.files.iter().any(|f| f.path.replace('\\', "/") == "crates/nmcore/Cargo.toml") {
        tentacle_pass::SELF_ANCHOR.to_owned()
    } else {
        format!("Repository of {} files, read in parts.\n", sub.files.len())
    }
}

static SERVED: std::sync::Mutex<Option<(String, Arc<Integrator>)>> = std::sync::Mutex::new(None);
/// The served integrator (the shared reader, the drone sets' last layer) for `anchor`, built on
/// first use and kept while the anchor is unchanged (RAM only).
/// # Errors
/// The reader cannot be opened, or engine faults on the anchor.
pub fn served(anchor: &str) -> Result<Arc<Integrator>, Fault> {
    let mut g = SERVED.lock().map_err(|_| bad("served integrator: lock poisoned"))?;
    if let Some((a, i)) = g.as_ref()
        && a == anchor
    {
        return Ok(Arc::clone(i));
    }
    let home = nmtype::nano_root();
    let model = crate::neural::model(&home)?;
    let dev = if model.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let last = tentacle_pass::drones(&home).last_layer();
    let gate = (dev == Device::Gpu).then(|| nmengn::placement::gpu_gate().enter("tentacle integration anchor", true));
    let i = Arc::new(Integrator::new(model, dev, anchor, last)?);
    drop(gate);
    *g = Some((anchor.to_owned(), Arc::clone(&i)));
    Ok(i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortlist_gate_and_scores() {
        let f = fuse(&[&[1.0, 5.0, 2.0, 0.0], &[0.5, 0.9, 0.1, 0.0]]);
        let s = shortlist(&f, &[100, 200, 300, 400], 0.99, 450);
        assert_eq!(s.first(), Some(&1), "the leader first");
        assert!(s.iter().map(|i| [100, 200, 300, 400][*i]).sum::<usize>() <= 450, "within the grant");
        assert_eq!(shortlist(&f, &[900, 900, 900, 900], 0.99, 450), vec![1], "the leader kept past the grant");
        assert!(in_band(&[1.0, 0.95, 0.0], 0.1) && !in_band(&[2.0, 0.5], 0.1));
        assert_eq!(above_anchor(&[1.0, 3.0, 2.0, 0.0, 1.0, 1.0], &[0.0; 6], 3), vec![3.0, 2.0]);
        let r = report(&[7, 8, 9], &[0.5, 2.0, -1.0], &[0.1, 3.0, 0.0]);
        assert_eq!((r.best.first().map(|x| x.0), r.margin, r.spread, r.no_information), (Some(8), 1.5, 3.0, false));
        assert!(report(&[1, 2], &[0.5, 0.1], &[1.0, 0.2]).no_information);
        assert_eq!(mean_ratio(&[1.0, 3.0, 2.0, 0.0, 1.0, 1.0], &[0.0; 6], 3, 1), vec![2.0, 1.5]);
    }

    #[test]
    fn plan_gates_and_grants() {
        // Window 0 clear; window 1 close (margin 0.1); window 2 closer (0.05) but its pieces
        // fill the budget first; window 1 then has no tokens left.
        let fused = vec![vec![5.0, 0.0, 0.0], vec![1.0, 0.9, 0.0], vec![1.0, 0.95, 0.9]];
        let tokens = vec![vec![100; 3], vec![400, 400, 400], vec![300, 300, 300]];
        let (p, used) = plan(&fused, &tokens, 1.0, 1_000, 1_000);
        assert_eq!((p[0].in_band, p[0].list.is_empty(), p[0].why), (false, true, "clear leader"));
        assert_eq!((p[2].list.len(), p[2].why), (3, "integrated"));
        assert_eq!((p[1].list.is_empty(), p[1].why), (true, "no card tokens left"));
        assert_eq!(used, 900);
        let (p, used) = plan(&fused, &tokens, 1.0, 2_000, 1_000);
        assert!(!p[1].list.is_empty() && used <= 2_000);
        // At 500 a tentacle both close windows are cut to their leader (a second piece passes
        // 500), so neither integrates.
        let (p, used) = plan(&fused, &tokens, 1.0, 1_000, 500);
        assert_eq!((p[2].list.len(), p[2].why, p[1].why, used), (0, "shortlist of one piece", "shortlist of one piece", 0));
    }
}
