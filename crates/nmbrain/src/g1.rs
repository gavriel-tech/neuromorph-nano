//! G1, the coordinating communicator across all tentacles (step P6, first form, 2026-11-10;
//! design by Luis Jake Gabriel III, `docs/2026-11-10-neumor-guide-how-it-works.md` section 3).
//! G1 receives and processes the output of every `g0_tx`, exchanges with each in both
//! directions, and decides what goes up to G2. It is the sole path between tentacles: a
//! tentacle holds no handle to another, and every instruction a `g0_tx` takes comes from G1
//! (the inbox refuses any other sender and counts the attempt; that count is the isolation
//! gate and must stay 0).
//!
//! Two internal layers (provisional in the specification), kept as separate types:
//! - `Coordination`: one gain per tentacle, which files the windows hold and what is unread,
//!   the rounds, the orders sent and the isolation gate.
//! - `Reasoning`: compares the reports (agreement on a file across tentacles, conflicts through
//!   the existing arbiter and Tension Buffer, coverage against the whole-code-base layer),
//!   decides per round which tentacle gets a re-read request or a specific question, and which
//!   spans go up, each marked with why.
//!
//! The whole-code-base layer (`WholeCodebase`) is compact by design: the file tree and the item
//! signatures `nmsubs` already extracts (`Substrate::files`, `Substrate::symbols`), passed in by
//! the caller as plain data so this crate keeps its dependencies. Full encodings of the whole
//! code base do not fit in RAM (measured 41.6 GiB for the Rust source); this layer is a few
//! megabytes at most.
//!
//! The loop: round 1 asks every tentacle for its report (all tentacles in parallel), later
//! rounds deliver G1's orders and process the replies (in parallel). It ends when a round
//! changes nothing, when there is nothing left to ask, when the time budget cannot fit another
//! round (mean round time measured), or at the ceiling derived from the tentacles' remaining
//! re-reads and held spans. No fixed round count.
//!
//! Research basis (`docs/2026-11-10-neumor-research-closing-the-gaps.md`, gap 2 and the
//! cross-analysis): gain per tentacle and a re-read request when a tentacle's confidence is
//! low or it reports no information (confidence-triggered re-reading, FLARE, arXiv 2305.06983,
//! abstract level; cascades with a confidence stop, FrugalGPT, arXiv 2305.05176); structured
//! reports with calibrated confidence and an explicit no-information answer. Methods only; no
//! code was taken.
use crate::arbiter;
use crate::contradiction::TensionBuffer;
use crate::evidence::{Capacity, EvidenceSet, Priority, SpanEvidence, SpanFacts, Structure};
use crate::g0::{Answer, Focus, G1_ID, Instruction, Order, Reply, Report, SpanRef, TentacleSide};
use nmjson::Value;
use nmtype::{Packet, Provenance, Status};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

/// G2's identity, the target of the hand-over messages.
pub const G2_ID: &str = "g2";

// ---------------------------------------------------------------------------------------------
// Decision rules (2026-11-10, Phase B): how a tentacle core turns its calibrated scores into a
// confidence and a shortlist, and how G1 combines a candidate's evidence. The first-form rules
// stay available so every comparison is repeatable; the served rules are one `Rules` record.
// ---------------------------------------------------------------------------------------------

/// How a tentacle core expresses the confidence of a chunk from its calibrated scores (scores
/// above the anchor, the window's content-free baseline).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfidenceRule {
    /// First form: `logistic(score / heads)`. Every chunk of a long window scores 11 to 18 nats
    /// over 16 heads, so confidences sit near 0.7 and do not discriminate.
    Logistic,
    /// The chunk's share of a softmax over the anchor (score 0) and every held chunk, at the
    /// window's own temperature (the standard deviation of its scores; fallback the head count
    /// when they do not spread). It reads rank and margin inside the window against the
    /// content-free baseline, on one probability scale in every tentacle.
    Window,
    /// `logistic((score - mean) / sd)` over the window's held chunks: the chunk's standing
    /// inside its own window in units of the window's spread, independent of how many chunks
    /// the window holds (the `Window` share falls as a window grows, so a short window's chunk
    /// outranks a long window's best). Fallback spread: the head count. Added 2026-11-10 after
    /// the first replay study showed that size effect. Test T3: within-window z-score.
    Standard,
    /// Test T3: `(score - median) / robust spread` (median absolute deviation) in the window.
    Robust,
    /// Test T3: rank plus margin: the window's first by its lead over the second, every other
    /// chunk by its distance below the first, in robust spreads of the window.
    Margin,
    /// Test T3: margin against the content-free read: the score above the anchor divided by the
    /// spread of the window's content-free read over its parts (a matched-filter noise floor).
    NullMargin,
}
/// How a tentacle core sets its shortlist size (gate 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShortlistRule {
    /// First form: `ceil(exp(entropy))` of the positive scores (7 to 15 chunks measured).
    Entropy,
    /// Cut at the largest gap between consecutive scores sorted best first, inside the chunks
    /// above the anchor (at least one).
    LargestGap,
}
/// Learned weights of the linear combination (non-negative, summing to 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Weights {
    pub confidence: f32,
    pub support: f32,
    pub agreement: f32,
    /// Weight of the tentacle's own rank, `1 / (1 + place)`.
    pub rank: f32,
}
impl Weights {
    #[must_use]
    pub fn text(&self) -> String {
        format!("confidence {:.3}, support {:.3}, agreement {:.3}, rank {:.3}", self.confidence, self.support, self.agreement, self.rank)
    }
}
/// The discriminating evidence G1 can combine (2026-11-10, refinement step 1, item 4;
/// analysis C sections 9.2 and 7.1), in weight order. Each feature is on 0 to 1:
/// `confidence` the core's confidence; `null-margin` the score above the window's content-free
/// baseline in units of the content-free spread, through the logistic; `support` the
/// whole-code-base layer's term support (normalised by the round's largest); `agreement` the share
/// of other tentacles reporting the same file; `structure` the reference-graph rank received from
/// the tentacles' first choices (normalised by the round's largest); `window-rank`, `mass-rank`
/// and `lens-rank` as `1 / (1 + rank)` in the span's window by the reported score, by the
/// attention-mass score, and in its lens's assignment order (0 when not recorded).
pub const EV_FEATURES: [&str; 8] =
    ["confidence", "null-margin", "support", "agreement", "structure", "window-rank", "mass-rank", "lens-rank"];
/// Weights of the evidence combination (non-negative, summing to 1), in `EV_FEATURES` order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvWeights(pub [f32; 8]);
impl EvWeights {
    #[must_use]
    pub fn text(&self) -> String {
        EV_FEATURES.iter().zip(self.0).filter(|(_, w)| *w > 0.0).map(|(n, w)| format!("{n} {w:.3}")).collect::<Vec<_>>().join(", ")
    }
    /// Simplex grid with step `1 / m` over the features named by index in `active` (the others
    /// weigh 0).
    #[must_use]
    pub fn grid(m: usize, active: &[usize]) -> Vec<Self> {
        let m = m.max(1);
        let mut out = Vec::new();
        let mut cur = vec![0_usize; active.len()];
        if !active.is_empty() {
            grid_rec(0, m, &mut cur, active, m, &mut out);
        }
        out
    }
}
fn grid_rec(i: usize, left: usize, cur: &mut Vec<usize>, active: &[usize], m: usize, out: &mut Vec<EvWeights>) {
    if i + 1 == cur.len() {
        if let Some(x) = cur.get_mut(i) {
            *x = left;
        }
        let mut w = [0.0_f32; 8];
        for (k, &f) in active.iter().enumerate() {
            if let Some(slot) = w.get_mut(f) {
                *slot = cur.get(k).copied().unwrap_or(0) as f32 / m as f32;
            }
        }
        out.push(EvWeights(w));
        return;
    }
    for v in 0..=left {
        if let Some(x) = cur.get_mut(i) {
            *x = v;
        }
        grid_rec(i + 1, left - v, cur, active, m, out);
    }
}
/// Weights of the evidence combination (`Rules::evidence`), in `EV_FEATURES` order: chosen by
/// cross-validation on the 30 saved development scans in two folds by question-index parity
/// (`crates/nmapi/tests/decision_rules.rs`, `stack_by_replay`, record
/// `runs/TR1-replay-1794302617209.txt`), then refitted on all 30. Out of fold it tied the first
/// form on first place (12 of 30) and lost presence (24 against 26), so it is not the served rule;
/// it stays selectable for comparison.
pub const STACK_WEIGHTS: [f32; 8] = [0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0];
/// How G1 combines a candidate's evidence into one score.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Combine {
    /// First form: `gain * confidence * (1 + support) * (1 + agreement share)`.
    #[default]
    Product,
    /// `gain * (w_c * confidence + w_s * support + w_a * agreement share + w_r * rank)`, the
    /// weights chosen by cross-validation on recorded questions.
    Linear(Weights),
    /// `gain * sum_f w_f * feature_f` over `EV_FEATURES` (2026-11-10, refinement step 1), the
    /// weights chosen by cross-validation on the saved scans.
    Evidence(EvWeights),
    /// The reasoning layer over the merged set of reports (`Pool`): each independent family of
    /// the drones' evidence normalised over the merged set and summed, weighted by how much it
    /// separates the candidates; agreement counted over the families, at its effective count.
    Pooled,
}
/// How G1 forwards (2026-11-10, refinement step 1, item 3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ForwardRule {
    /// First form: the best `Budget::forward` spans go up; every other reported span stays in the
    /// evidence set's queue (built under both rules).
    #[default]
    Fixed,
    /// The round's capacity in tokens (`evidence::Capacity`) goes up by evidence; everything
    /// else stays queued with its evidence (`evidence::EvidenceSet`); nothing is dropped.
    Queue,
}
/// The decision rules of the tentacle cores and G1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rules {
    pub confidence: ConfidenceRule,
    pub shortlist: ShortlistRule,
    pub combine: Combine,
    pub forward: ForwardRule,
}
impl Rules {
    /// The first-form rules (the comparison arm).
    #[must_use]
    pub fn first_form() -> Self {
        Self {
            confidence: ConfidenceRule::Logistic,
            shortlist: ShortlistRule::Entropy,
            combine: Combine::Product,
            forward: ForwardRule::Fixed,
        }
    }
    /// The served Globalized Context Stack rules of the new path (refinement step 1): first-form confidence, gate 1
    /// and combination (the best out of fold on the saved scans; nothing fitted), with the queue
    /// forward rule, so nothing reported is dropped and the evidence set travels on.
    #[must_use]
    pub fn stack() -> Self {
        Self { forward: ForwardRule::Queue, ..Self::first_form() }
    }
    /// The evidence combination with `STACK_WEIGHTS` under the queue rule (selectable).
    #[must_use]
    pub fn evidence() -> Self {
        Self { combine: Combine::Evidence(EvWeights(STACK_WEIGHTS)), forward: ForwardRule::Queue, ..Self::first_form() }
    }
    #[must_use]
    pub fn text(&self) -> String {
        let c = match self.combine {
            Combine::Product => "product".to_owned(),
            Combine::Linear(w) => format!("linear ({})", w.text()),
            Combine::Evidence(w) => format!("evidence ({})", w.text()),
            Combine::Pooled => "pooled".to_owned(),
        };
        format!("confidence {:?}, shortlist {:?}, combine {c}, forward {:?}", self.confidence, self.shortlist, self.forward)
    }
}
impl Default for Rules {
    fn default() -> Self {
        Self::first_form()
    }
}
static RULES: std::sync::RwLock<Option<Rules>> = std::sync::RwLock::new(None);
/// The rules in service for this process (the first form until `set_rules` says otherwise).
#[must_use]
pub fn rules() -> Rules {
    RULES.read().ok().and_then(|r| *r).unwrap_or_default()
}
/// Set the rules in service for this process (served setting, replay arms).
pub fn set_rules(r: Rules) {
    if let Ok(mut g) = RULES.write() {
        *g = Some(r);
    }
}

/// The window temperature of the `Window` rule: the standard deviation of the scores, fallback
/// `heads` when they do not spread (stated fallback).
#[must_use]
pub fn window_temperature(scores: &[f32], heads: usize) -> f32 {
    let v: Vec<f64> = scores.iter().filter(|x| x.is_finite()).map(|x| f64::from(*x)).collect();
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n.max(1.0);
    let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n.max(1.0)).sqrt();
    #[expect(clippy::cast_possible_truncation, reason = "a score spread")]
    let sd = sd as f32;
    if sd.is_finite() && sd > 0.0 { sd } else { heads.max(1) as f32 }
}
fn finite(scores: &[f32]) -> Vec<f32> {
    scores.iter().copied().filter(|x| x.is_finite()).collect()
}
fn median(v: &[f32]) -> f32 {
    let mut v = v.to_vec();
    v.sort_by(f32::total_cmp);
    match v.len() {
        0 => 0.0,
        n if n % 2 == 1 => v.get(n / 2).copied().unwrap_or(0.0),
        n => (v.get(n / 2 - 1).copied().unwrap_or(0.0) + v.get(n / 2).copied().unwrap_or(0.0)) / 2.0,
    }
}
/// Robust spread of the window: the median absolute deviation scaled to a standard deviation
/// (1.4826, the normal-consistency factor), fallback the standard deviation, then the head count.
#[must_use]
pub fn robust_spread(scores: &[f32], heads: usize) -> f32 {
    let v = finite(scores);
    let m = median(&v);
    let mad = 1.4826 * median(&v.iter().map(|x| (x - m).abs()).collect::<Vec<f32>>());
    if mad.is_finite() && mad > 0.0 { mad } else { window_temperature(scores, heads) }
}
/// The report step's normalisation component (test T3 of
/// `docs/2026-11-10-neumor-research-why-reads-degrade.md`): each chunk's score as a
/// dimensionless value inside its own window, comparable across windows. `None` for the rules
/// that are not a normalisation (`Logistic`, `Window`). `null_spread` is the spread of the
/// window's content-free read over its parts (used by `NullMargin`; fallback the window spread).
#[must_use]
pub fn normalised(rule: ConfidenceRule, scores: &[f32], heads: usize, null_spread: Option<f32>) -> Option<Vec<f32>> {
    let v = finite(scores);
    let map = |f: &dyn Fn(f32) -> f32| scores.iter().map(|s| if s.is_finite() { f(*s) } else { f32::NEG_INFINITY }).collect::<Vec<f32>>();
    match rule {
        ConfidenceRule::Logistic | ConfidenceRule::Window => None,
        ConfidenceRule::Standard => {
            let t = window_temperature(scores, heads);
            let mean = v.iter().sum::<f32>() / v.len().max(1) as f32;
            Some(map(&|s| (s - mean) / t))
        }
        ConfidenceRule::Robust => {
            let (m, t) = (median(&v), robust_spread(scores, heads));
            Some(map(&|s| (s - m) / t))
        }
        ConfidenceRule::Margin => {
            // Margin over the best other chunk: the window's first gets its lead over the second,
            // every other chunk its distance below the first, in robust spreads.
            let t = robust_spread(scores, heads);
            let mut sorted = v.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let (first, second) = (sorted.first().copied().unwrap_or(0.0), sorted.get(1).copied().unwrap_or(0.0));
            Some(map(&|s| if s >= first { (s - second) / t } else { (s - first) / t }))
        }
        ConfidenceRule::NullMargin => {
            let t = null_spread.filter(|x| x.is_finite() && *x > 0.0).unwrap_or_else(|| window_temperature(scores, heads));
            Some(map(&|s| s / t))
        }
    }
}
/// Confidence of every held chunk from its calibrated score, by the rule (normalisations pass
/// through the logistic, which keeps their order and puts them on 0 to 1).
#[must_use]
pub fn confidences_with(rule: ConfidenceRule, scores: &[f32], heads: usize, null_spread: Option<f32>) -> Vec<f32> {
    if let Some(z) = normalised(rule, scores, heads, null_spread) {
        return z.iter().map(|x| if x.is_finite() { 1.0 / (1.0 + (-x).exp()) } else { 0.0 }).collect();
    }
    match rule {
        ConfidenceRule::Window => {
            let t = window_temperature(scores, heads);
            let top = scores.iter().copied().filter(|x| x.is_finite()).fold(0.0_f32, f32::max);
            let e: Vec<f32> = scores.iter().map(|s| if s.is_finite() { ((s - top) / t).exp() } else { 0.0 }).collect();
            let z = (-top / t).exp() + e.iter().sum::<f32>();
            e.iter().map(|x| if z > 0.0 { x / z } else { 0.0 }).collect()
        }
        _ => {
            let unit = heads.max(1) as f32;
            scores.iter().map(|s| 1.0 / (1.0 + (-s / unit).exp())).collect()
        }
    }
}
/// `confidences_with` without a content-free spread.
#[must_use]
pub fn confidences(rule: ConfidenceRule, scores: &[f32], heads: usize) -> Vec<f32> {
    confidences_with(rule, scores, heads, None)
}
/// Confidence of one fresh score (an answer to a specific question) on the same scale as the
/// window's: the score takes the place of chunk `i` in the window.
#[must_use]
pub fn confidence_of(rule: ConfidenceRule, scores: &[f32], i: usize, score: f32, heads: usize, null_spread: Option<f32>) -> f32 {
    let mut s = scores.to_vec();
    match s.get_mut(i) {
        Some(x) => *x = score,
        None => s.push(score),
    }
    let at = i.min(s.len().saturating_sub(1));
    confidences_with(rule, &s, heads, null_spread).get(at).copied().unwrap_or(0.0)
}
/// Shortlist size by the rule, at most the chunk count, `fallback` (at most the chunk count) when
/// no score is above the anchor. Returns the size and how it was set.
#[must_use]
pub fn shortlist_by(rule: ShortlistRule, scores: &[f32], fallback: usize) -> (usize, String) {
    let n = scores.len();
    let mut pos: Vec<f32> = scores.iter().copied().filter(|x| x.is_finite() && *x > 0.0).collect();
    if n == 0 || pos.is_empty() {
        let k = fallback.min(n.max(1));
        return (k, format!("fallback {k}: no chunk above the anchor"));
    }
    match rule {
        ShortlistRule::Entropy => {
            let total: f64 = pos.iter().map(|x| f64::from(*x)).sum();
            let h: f64 = pos.iter().map(|x| f64::from(*x) / total).map(|p| -p * p.ln()).sum();
            let eff = h.exp();
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a small positive count")]
            let k = (eff.ceil() as usize).clamp(1, n);
            (k, format!("{k} = ceil(exp(entropy)) of the positive scores ({eff:.2} effective chunks of {n})"))
        }
        ShortlistRule::LargestGap => {
            pos.sort_by(|a, b| b.total_cmp(a));
            let mut k = 1;
            let mut best = f32::NEG_INFINITY;
            for (i, w) in pos.windows(2).enumerate() {
                let g = w.first().copied().unwrap_or(0.0) - w.get(1).copied().unwrap_or(0.0);
                if g > best {
                    best = g;
                    k = i + 1;
                }
            }
            let gap = if best.is_finite() { best } else { 0.0 };
            (k, format!("{k} = cut at the largest gap ({gap:.2}) among the {} scores above the anchor", pos.len()))
        }
    }
}
/// Simplex grid of `Weights` with step `1 / m` (every non-negative combination of four weights
/// in steps of `1 / m` summing to 1).
#[must_use]
pub fn weight_grid(m: usize) -> Vec<Weights> {
    let m = m.max(1);
    let mut out = Vec::new();
    #[expect(clippy::cast_precision_loss, reason = "small grid counts")]
    let f = |x: usize| x as f32 / m as f32;
    for a in 0..=m {
        for b in 0..=m - a {
            for c in 0..=m - a - b {
                let d = m - a - b - c;
                out.push(Weights { confidence: f(a), support: f(b), agreement: f(c), rank: f(d) });
            }
        }
    }
    out
}

/// One item signature of the whole-code-base layer (from `nmsubs::repo::Symbol`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ItemSig {
    pub name: String,
    pub kind: String,
    pub file: String,
    pub line: usize,
    /// The item's declaration line, trimmed.
    pub signature: String,
}

/// The whole-code-base layer: sorted file paths and the item signatures of each file.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WholeCodebase {
    pub files: Vec<String>,
    /// Sorted by (file, line).
    pub items: Vec<ItemSig>,
    /// Per file, the range of its items in `items`.
    ranges: Vec<(usize, usize)>,
}

/// Relevance of the layer to one question: per file and per item (idf-weighted term hits).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Relevance {
    pub terms: Vec<String>,
    pub file: Vec<f32>,
    pub item: Vec<f32>,
}

/// Question terms: key terms and their identifier parts, stop words dropped.
#[must_use]
pub fn question_terms(q: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for k in nmlex::key_terms(q) {
        for w in nmlex::words(&k) {
            if (w.len() >= 3 || nmlex::short_name(&w)) && !nmlex::STOP.contains(&w.as_str()) && !out.contains(&w) {
                out.push(w);
            }
        }
    }
    out
}

impl WholeCodebase {
    /// Build from the file paths and item signatures (any order).
    #[must_use]
    pub fn build(mut files: Vec<String>, mut items: Vec<ItemSig>) -> Self {
        files.sort();
        files.dedup();
        items.sort_by(|a, b| a.file.cmp(&b.file).then(a.line.cmp(&b.line)));
        let mut ranges = Vec::with_capacity(files.len());
        let mut at = 0;
        for f in &files {
            while items.get(at).is_some_and(|i| i.file < *f) {
                at += 1;
            }
            let start = at;
            while items.get(at).is_some_and(|i| i.file == *f) {
                at += 1;
            }
            ranges.push((start, at));
        }
        Self { files, items, ranges }
    }
    #[must_use]
    pub fn file_index(&self, f: &str) -> Option<usize> {
        self.files.binary_search_by(|x| x.as_str().cmp(f)).ok()
    }
    /// Items of file `i`.
    #[must_use]
    pub fn items_of(&self, i: usize) -> &[ItemSig] {
        self.ranges.get(i).and_then(|&(a, b)| self.items.get(a..b)).unwrap_or(&[])
    }
    /// Relevance to a question: an item scores the idf of each question term among its name
    /// and signature words; a file scores its items' sum plus the idf of terms in its path.
    #[must_use]
    pub fn relevance(&self, question: &str) -> Relevance {
        let terms = question_terms(question);
        let item_words: Vec<Vec<String>> = self.items.iter().map(|i| nmlex::words(&format!("{} {}", i.name, i.signature))).collect();
        let path_words: Vec<Vec<String>> = self.files.iter().map(|f| nmlex::words(f)).collect();
        let n = (self.items.len() + self.files.len()).max(1) as f32;
        let idf: Vec<f32> = terms
            .iter()
            .map(|t| {
                let df = item_words.iter().chain(&path_words).filter(|w| w.contains(t)).count();
                if df == 0 { 0.0 } else { (1.0 + n / df as f32).ln() }
            })
            .collect();
        let hit = |w: &[String]| terms.iter().zip(&idf).filter(|(t, _)| w.contains(t)).map(|(_, x)| *x).sum::<f32>();
        let item: Vec<f32> = item_words.iter().map(|w| hit(w)).collect();
        let file = (0..self.files.len())
            .map(|i| {
                let (a, b) = self.ranges.get(i).copied().unwrap_or((0, 0));
                item.get(a..b).map_or(0.0, |s| s.iter().sum::<f32>()) + path_words.get(i).map_or(0.0, |w| hit(w))
            })
            .collect();
        Relevance { terms, file, item }
    }
    /// Raw support of the layer for one span: the relevance of the items declared inside it plus
    /// its file's mean item relevance. 0 for a file outside the layer.
    #[must_use]
    pub fn support(&self, rel: &Relevance, span: &SpanRef) -> f32 {
        let Some(i) = self.file_index(&span.file) else { return 0.0 };
        let (a, b) = self.ranges.get(i).copied().unwrap_or((0, 0));
        let inside: f32 = (a..b)
            .filter(|&k| self.items.get(k).is_some_and(|it| span.contains(&it.file, it.line)))
            .map(|k| rel.item.get(k).copied().unwrap_or(0.0))
            .sum();
        let file = rel.file.get(i).copied().unwrap_or(0.0) / (b - a).max(1) as f32;
        inside + file
    }
    /// The most relevant item of file `i`, when any item there has relevance.
    #[must_use]
    pub fn best_item(&self, rel: &Relevance, i: usize) -> Option<&ItemSig> {
        let (a, b) = self.ranges.get(i).copied()?;
        (a..b)
            .filter(|k| rel.item.get(*k).copied().unwrap_or(0.0) > 0.0)
            .max_by(|x, y| rel.item.get(*x).copied().unwrap_or(0.0).total_cmp(&rel.item.get(*y).copied().unwrap_or(0.0)))
            .and_then(|k| self.items.get(k))
    }
    /// Files by relevance, most relevant first, relevance above 0 only.
    #[must_use]
    pub fn ranked_files(&self, rel: &Relevance) -> Vec<usize> {
        let mut v: Vec<usize> = (0..self.files.len()).filter(|i| rel.file.get(*i).copied().unwrap_or(0.0) > 0.0).collect();
        v.sort_by(|a, b| rel.file.get(*b).copied().unwrap_or(0.0).total_cmp(&rel.file.get(*a).copied().unwrap_or(0.0)).then(a.cmp(b)));
        v
    }
    /// The layer as compact text: one line per file, one indented line per item
    /// (`line kind signature`). This is what its size in tokens is measured on.
    #[must_use]
    pub fn render(&self) -> String {
        let mut s = String::new();
        for (i, f) in self.files.iter().enumerate() {
            s.push_str(f);
            s.push('\n');
            for it in self.items_of(i) {
                s.push_str(&format!("  {} {} {}\n", it.line, it.kind, it.signature));
            }
        }
        s
    }
    /// Bytes held in RAM (string bytes plus the fixed part of each record).
    #[must_use]
    pub fn bytes(&self) -> usize {
        let files: usize = self.files.iter().map(|f| f.len() + std::mem::size_of::<String>()).sum();
        let items: usize = self
            .items
            .iter()
            .map(|i| i.name.len() + i.kind.len() + i.file.len() + i.signature.len() + std::mem::size_of::<ItemSig>())
            .sum();
        files + items + self.ranges.len() * std::mem::size_of::<(usize, usize)>()
    }
}

/// Why a span was forwarded (or held).
#[derive(Clone, Debug, PartialEq)]
pub enum Reason {
    /// The tentacle's first choice.
    FirstChoice,
    /// Normalized support of the whole-code-base layer (0 to 1).
    Corroborated(f32),
    /// Other tentacles reported spans in the same file.
    Agreement(usize),
    /// A specific question to the tentacle was answered with this confidence.
    Answered(f32),
    /// Reported after this many reads (a re-read found it).
    AfterReread(u32),
    /// G1 aimed a question at it from the whole-code-base layer (it was not reported).
    Aimed,
    /// Two observations of the same key disagree; both kept, a question sent.
    ConflictHeld(String),
}
impl Reason {
    #[must_use]
    pub fn text(&self) -> String {
        match self {
            Self::FirstChoice => "first choice of its tentacle".into(),
            Self::Corroborated(s) => format!("corroborated by the whole-code-base layer ({s:.2})"),
            Self::Agreement(n) => format!("{n} other tentacle(s) reported the same file"),
            Self::Answered(c) => format!("answered a specific question with confidence {c:.2}"),
            Self::AfterReread(r) => format!("reported after {r} reads"),
            Self::Aimed => "aimed at by G1 from the whole-code-base layer".into(),
            Self::ConflictHeld(k) => format!("conflict held for {k}"),
        }
    }
}

/// One span under consideration in a round.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub span: SpanRef,
    pub tentacle: usize,
    pub from: String,
    pub place: usize,
    pub score: f32,
    pub confidence: f32,
    /// Normalized support of the whole-code-base layer this round (0 to 1).
    pub support: f32,
    pub agreement: usize,
    pub combined: f32,
    pub reasons: Vec<Reason>,
    /// The tentacle's evidence for the span (refinement step 1; default when none was given).
    pub facts: SpanFacts,
    /// Reference-graph rank received from the tentacles' first choices, normalised (0 to 1).
    pub structure: f32,
}

/// The features of `EV_FEATURES` for one candidate; `n` is the number of tentacles reporting.
#[must_use]
pub fn features(c: &Candidate, n: usize) -> [f32; 8] {
    let inv = |r: usize| if r == usize::MAX { 0.0 } else { 1.0 / (1.0 + r as f32) };
    let share = c.agreement as f32 / (n.max(2) as f32 - 1.0);
    let has = c.facts.tokens > 0 || !c.facts.scores.is_empty();
    let window = if has { c.facts.window_rank } else { c.place };
    let mass = if has { c.facts.mass_rank } else { c.place };
    [c.confidence, c.facts.null_margin(), c.support, share, c.structure, inv(window), inv(mass), c.facts.lens_rank.map_or(0.0, inv)]
}
/// The combined value of a candidate under `combine`, with its tentacle's gain.
#[must_use]
pub fn combined_value(c: &Candidate, combine: Combine, gain: f32, n: usize) -> f32 {
    let share = c.agreement as f32 / (n.max(2) as f32 - 1.0);
    match combine {
        Combine::Product => gain * c.confidence * (1.0 + c.support) * (1.0 + share),
        Combine::Linear(w) => {
            let rank = if c.place == usize::MAX { 0.0 } else { 1.0 / (1.0 + c.place as f32) };
            gain * (w.confidence * c.confidence + w.support * c.support + w.agreement * share + w.rank * rank)
        }
        Combine::Evidence(w) => gain * features(c, n).iter().zip(w.0).map(|(f, w)| f * w).sum::<f32>(),
        Combine::Pooled => gain * c.confidence,
    }
}

/// The independent families of the drones' evidence and the score sets each reads (the
/// calibrated signals a `g0_tx` holds per piece).
/// Since 2026-11-18 (lane T1, D7) a fourth: `reading`, the tentacles' integration read (score set
/// `read`), present only on pieces a tentacle integrated (`NANO_TENTACLE_READ=on`).
pub const FAMILIES: [(&str, &[&str]); 4] =
    [("wording", &["bm25f", "bm25", "exact"]), ("naming", &["structural", "ident"]), ("meaning", &["affinity"]), ("reading", &["read"])];
/// The drones' own families (the first three of `FAMILIES`): the pool's weights from their
/// distributions, their correlation and the quorum are over these; `reading` joins a piece's
/// value only where it was read (`Pool::reading`).
pub const DRONE_FAMILIES: usize = 3;
/// The weight of the `reading` family in a pooled value: an even share of the drones' three
/// (stated, not fitted; lane T1 D7, 2026-11-18).
pub const READING_WEIGHT: f32 = 1.0 / 3.0;

/// G1's reasoning layer over the merged set of every report (the protocol pass, Solution 4; spec
/// 4.4): every family's value (its best calibrated signal) is normalised over the merged set, its
/// floor the content-free value (0) and its ceiling the set's largest; the families are summed
/// with weights from their own distributions (one less the normalised entropy of a family's
/// values over the set, so a family that picks out few candidates counts more and one that is
/// flat counts little; equal when none separates); agreement is the number of families that
/// fired on a candidate, taken at its effective count `q / (1 + (q - 1) rho)` with `rho` the mean
/// correlation between the families over the set.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pool {
    pub ceiling: [f32; 3],
    pub weight: [f32; 3],
    pub rho: f32,
    /// The `reading` family's ceiling (the largest read score of the request so far) and its
    /// weight; 0 while no tentacle has integrated, so the value is the drones' alone.
    pub reading: f32,
    pub reading_weight: f32,
}
impl Pool {
    fn family(c: &Candidate, k: usize) -> f32 {
        FAMILIES.get(k).map_or(0.0, |(_, sigs)| sigs.iter().filter_map(|s| c.facts.score(s)).filter(|x| x.is_finite()).fold(0.0_f32, f32::max))
    }
    /// A candidate's family values (each family's best calibrated signal).
    #[must_use]
    pub fn families(c: &Candidate) -> [f32; 3] {
        [Self::family(c, 0), Self::family(c, 1), Self::family(c, 2)]
    }
    /// The pool of a merged set.
    #[must_use]
    pub fn of(cands: &[Candidate]) -> Self {
        Self::over(&cands.iter().map(Self::families).collect::<Vec<_>>())
    }
    /// The pool over any set of family values: one row per piece. Over every piece of the
    /// material for a request it is the scale shared by all tentacles: each family's ceiling is
    /// the tree's maximum, so a piece's value does not move with the set it is reported in.
    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "piece counts")]
    pub fn over(rows: &[[f32; 3]]) -> Self {
        let mut ceiling = [0.0_f32; 3];
        for r in rows {
            for (x, v) in ceiling.iter_mut().zip(r) {
                if v.is_finite() {
                    *x = x.max(*v);
                }
            }
        }
        let share = |c: &[f32; 3], k: usize| {
            let top = ceiling.get(k).copied().unwrap_or(0.0);
            let v = c.get(k).copied().filter(|x| x.is_finite()).unwrap_or(0.0);
            if top > 0.0 { (v / top).max(0.0) } else { 0.0 }
        };
        let cands = rows;
        let m = cands.len();
        let mut weight = [0.0_f32; 3];
        for (k, w) in weight.iter_mut().enumerate() {
            let total: f32 = cands.iter().map(|c| share(c, k)).sum();
            if total <= 0.0 || m < 2 {
                continue;
            }
            let h: f32 = cands.iter().map(|c| share(c, k) / total).filter(|p| *p > 0.0).map(|p| -p * p.ln()).sum();
            *w = (1.0 - h / (m as f32).ln()).max(0.0);
        }
        let sum: f32 = weight.iter().sum();
        if sum > 0.0 {
            for w in &mut weight {
                *w /= sum;
            }
        } else {
            weight = [1.0 / 3.0; 3];
        }
        // Mean pairwise correlation of the families' shares over the set.
        let col = |k: usize| -> Vec<f32> { cands.iter().map(|c| share(c, k)).collect() };
        let corr = |a: &[f32], b: &[f32]| -> Option<f32> {
            let n = a.len() as f32;
            if a.len() < 2 {
                return None;
            }
            let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
            let cov: f32 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
            let va: f32 = a.iter().map(|x| (x - ma) * (x - ma)).sum();
            let vb: f32 = b.iter().map(|y| (y - mb) * (y - mb)).sum();
            (va > 0.0 && vb > 0.0).then(|| cov / (va * vb).sqrt())
        };
        let cols: Vec<Vec<f32>> = (0..3).map(col).collect();
        let pairs: Vec<f32> = [(0, 1), (0, 2), (1, 2)].iter().filter_map(|(i, j)| corr(cols.get(*i)?, cols.get(*j)?)).collect();
        let rho = if pairs.is_empty() { 1.0 } else { (pairs.iter().sum::<f32>() / pairs.len() as f32).clamp(0.0, 1.0) };
        Self { ceiling, weight, rho, reading: 0.0, reading_weight: 0.0 }
    }
    /// The pool with the `reading` family's ceiling raised to `top` (a read score), at
    /// `READING_WEIGHT`.
    pub fn read_up_to(&mut self, top: f32) {
        if top.is_finite() && top > self.reading {
            self.reading = top;
            self.reading_weight = READING_WEIGHT;
        }
    }
    /// A candidate's pooled value: the weighted sum of its family shares, times one plus its
    /// families' effective agreement over the families' count.
    #[must_use]
    pub fn value(&self, c: &Candidate) -> f32 {
        self.value_with(&Self::families(c), Self::family(c, DRONE_FAMILIES))
    }
    /// The pooled value of one piece's family values.
    #[must_use]
    pub fn value_of(&self, fam: &[f32; 3]) -> f32 {
        self.value_with(fam, 0.0)
    }
    /// The pooled value of one piece's drone family values and its read score (0: not read):
    /// the read adds its share at `reading_weight` and counts as one more agreeing family.
    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "three families")]
    pub fn value_with(&self, fam: &[f32; 3], read: f32) -> f32 {
        let mut sum = 0.0;
        let mut q = 0_usize;
        for k in 0..3 {
            let top = self.ceiling.get(k).copied().unwrap_or(0.0);
            let v = fam.get(k).copied().filter(|x| x.is_finite()).unwrap_or(0.0);
            let s = if top > 0.0 { (v / top).clamp(0.0, 1.0) } else { 0.0 };
            if s > 0.0 {
                q += 1;
            }
            sum += self.weight.get(k).copied().unwrap_or(0.0) * s;
        }
        if self.reading > 0.0 && self.reading_weight > 0.0 && read.is_finite() {
            let s = (read / self.reading).clamp(0.0, 1.0);
            if s > 0.0 {
                q += 1;
                sum += self.reading_weight * s;
            }
        }
        let eff = if q == 0 { 0.0 } else { q as f32 / (1.0 + (q as f32 - 1.0) * self.rho) };
        sum * (1.0 + eff) / (1.0 + 3.0)
    }
}

/// One candidate in G1's race: its running total over every read that reported it, the drones'
/// quorum and the aspects it holds.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Runner {
    pub span: SpanRef,
    pub from: String,
    /// The running total: the mean of every reading the reads gave it (each new reading weighs
    /// as much as the ones before together decay to; a repeated equal reading leaves it as it is).
    pub total: f32,
    pub readings: u32,
    pub quorum: usize,
    pub aspects: Vec<usize>,
    /// The round it was first reported in.
    pub round: u32,
    /// The round it was last read in (accumulation fades a runner not re-read).
    pub last: u32,
    /// The evidence kinds above their content-free band on it (`Race::feed_kinds`), for the quorum
    /// by measured independence.
    pub kinds: Vec<String>,
}

/// The measured independence of the evidence kinds (tentacle master plan 3.6: the quorum counts
/// kinds only by measured independence; wording and naming are nearly one kind on code; structure
/// proposes only). The effective number of kinds among `k` is `n * n / sum_ij rho_ij` (the
/// effective count of correlated signals; equal kinds count once, independent ones fully).
#[derive(Clone, Debug, PartialEq)]
pub struct Independence {
    /// The kinds that may vote, in order.
    pub kinds: Vec<String>,
    /// Measured correlations between pairs of kinds (either order); others take `rho`.
    pub corr: Vec<(String, String, f32)>,
    /// The request's mean correlation, for pairs not measured.
    pub rho: f32,
    /// Kinds that propose candidates but never vote.
    pub proposes_only: Vec<String>,
}
impl Independence {
    /// On code (stated until A3's correlations are written per request): wording and naming nearly
    /// one kind (0.9), meaning and reading at the request's `rho`, structure proposes only.
    #[must_use]
    pub fn code(rho: f32) -> Self {
        Self {
            kinds: ["wording", "naming", "meaning", "reading"].iter().map(|s| (*s).to_owned()).collect(),
            corr: vec![("wording".into(), "naming".into(), 0.9)],
            rho: rho.clamp(0.0, 1.0),
            proposes_only: vec!["structure".into()],
        }
    }
    fn rho_of(&self, a: &str, b: &str) -> f32 {
        if a == b {
            return 1.0;
        }
        self.corr.iter().find(|(x, y, _)| (x == a && y == b) || (x == b && y == a)).map_or(self.rho, |c| c.2.clamp(0.0, 1.0))
    }
    /// The effective number of the voting kinds among `kinds`.
    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "kind counts")]
    pub fn effective(&self, kinds: &[String]) -> f32 {
        let mut seen: Vec<&String> = Vec::new();
        for k in kinds.iter().filter(|k| self.kinds.contains(k) && !self.proposes_only.contains(k)) {
            if !seen.contains(&k) {
                seen.push(k);
            }
        }
        let n = seen.len() as f32;
        let s: f32 = seen.iter().flat_map(|a| seen.iter().map(move |b| (a, b))).map(|(a, b)| self.rho_of(a, b)).sum();
        if s > 0.0 { n * n / s } else { 0.0 }
    }
    /// A quorum: the kinds above band on a candidate count for more than half of all voting kinds,
    /// both at their effective numbers, and for more than one independent kind (at least 1.5
    /// effective): one kind, however many signals it splits into, is never a quorum.
    #[must_use]
    pub fn quorum(&self, kinds: &[String]) -> bool {
        let e = self.effective(kinds);
        2.0 * e > self.effective(&self.kinds) && e >= 1.5
    }
}

/// What G1's race says at a moment: every runner's state, and what G2 must settle.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Verdict {
    /// Runners that commit upward, by total, best first: a quorum of independent families, or
    /// the leader of an aspect clear of its rival by the noise band.
    pub committed: Vec<String>,
    /// Aspects whose leader neither has a quorum nor stands clear of its rival: (aspect, leader,
    /// rival). Both go up and G2 decides.
    pub ties: Vec<(usize, String, String)>,
    /// Aspects no runner holds.
    pub uncovered: Vec<usize>,
    /// Every other runner, by total (queued, by reference).
    pub queued: Vec<String>,
}
impl Verdict {
    /// What goes up to G2, in order: the committed and both sides of every tie, by total.
    #[must_use]
    pub fn forward(&self, race: &Race) -> Vec<String> {
        let mut v: Vec<String> = self.committed.clone();
        for (_, a, b) in &self.ties {
            for k in [a, b] {
                if !v.contains(k) {
                    v.push(k.clone());
                }
            }
        }
        race.order(&mut v);
        v
    }
}

/// G1's race over the candidates of every round (spec 4.4, protocol pass Solution 4): one running
/// total per candidate; within each aspect the candidates holding it compete, and the leader
/// stands against its rival (the strongest other holder); a candidate commits upward on a quorum
/// of the drones' independent families (each above its content-free band, so the threshold is set
/// on null poses, not fitted) or on a lead beyond the noise band; a tie goes to G2 with both
/// sides; an aspect nobody holds goes to G2 as uncovered. Nothing is dropped: every runner not
/// forwarded stays queued.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Race {
    pub runners: Vec<Runner>,
    /// The noise band on the common scale (the request's median absolute deviation).
    pub band: f32,
    /// The request's aspects.
    pub aspects: usize,
    /// The families' mean correlation over the request (for effective counts).
    pub rho: f32,
    pub round: u32,
    /// Accumulation across rounds (tentacle master plan 3.6): when on, a runner's total is the sum
    /// of its fused evidence per round (its best reading of that round), not the mean of readings.
    pub accumulate: bool,
    /// Fading per round on runners not re-read that round (0: none, the totals of the old form).
    pub fade: f32,
    /// The quorum by measured independence of kinds; `None`: the drones' family count at `rho`.
    pub independence: Option<Independence>,
}
impl Race {
    /// A race for a request with `aspects` aspects, its noise band and its families' correlation.
    #[must_use]
    pub fn new(aspects: usize, band: f32, rho: f32) -> Self {
        Self { aspects, band, rho, ..Self::default() }
    }
    /// The accumulating race (master plan 3.6): fused evidence summed across rounds, runners not
    /// re-read fading by `fade` a round, the quorum by measured independence of kinds.
    #[must_use]
    pub fn accumulating(aspects: usize, band: f32, rho: f32, fade: f32) -> Self {
        Self { aspects, band, rho, accumulate: true, fade: fade.clamp(0.0, 1.0), independence: Some(Independence::code(rho)), ..Self::default() }
    }
    /// Feed one round's readings with the kinds above band on each (one list per item).
    pub fn feed_kinds(&mut self, items: &[(SpanRef, String, f32, usize, Vec<usize>)], kinds: &[Vec<String>]) {
        self.feed(items);
        for ((span, ..), ks) in items.iter().zip(kinds) {
            if let Some(r) = self.runners.iter_mut().find(|r| r.span.key == span.key) {
                for k in ks {
                    if !r.kinds.contains(k) {
                        r.kinds.push(k.clone());
                    }
                }
            }
        }
    }
    /// Feed one round's readings: (span, from, value on the common scale, quorum, aspects).
    pub fn feed(&mut self, items: &[(SpanRef, String, f32, usize, Vec<usize>)]) {
        self.round += 1;
        let round = self.round;
        if self.accumulate {
            // One fused value per candidate per round: its best reading of the round.
            let mut best: BTreeMap<String, usize> = BTreeMap::new();
            for (i, (span, _, value, ..)) in items.iter().enumerate() {
                if value.is_finite() && best.get(&span.key).is_none_or(|j| items.get(*j).is_some_and(|x| x.2 < *value)) {
                    best.insert(span.key.clone(), i);
                }
            }
            for i in best.into_values() {
                let Some((span, from, value, quorum, aspects)) = items.get(i) else { continue };
                match self.runners.iter_mut().find(|r| r.span.key == span.key) {
                    Some(r) => {
                        r.total += value;
                        r.readings += 1;
                        r.last = round;
                        r.quorum = r.quorum.max(*quorum);
                        for a in aspects {
                            if !r.aspects.contains(a) {
                                r.aspects.push(*a);
                            }
                        }
                    }
                    None => self.runners.push(Runner {
                        span: span.clone(),
                        from: from.clone(),
                        total: *value,
                        readings: 1,
                        quorum: *quorum,
                        aspects: aspects.clone(),
                        round,
                        last: round,
                        kinds: Vec::new(),
                    }),
                }
            }
            self.fade_unread();
            return;
        }
        for (span, from, value, quorum, aspects) in items {
            if !value.is_finite() {
                continue;
            }
            match self.runners.iter_mut().find(|r| r.span.key == span.key) {
                Some(r) => {
                    r.last = round;
                    #[expect(clippy::cast_precision_loss, reason = "a reading count")]
                    let n = r.readings as f32;
                    r.total = (r.total * n + value) / (n + 1.0);
                    r.readings += 1;
                    r.quorum = r.quorum.max(*quorum);
                    for a in aspects {
                        if !r.aspects.contains(a) {
                            r.aspects.push(*a);
                        }
                    }
                }
                None => self.runners.push(Runner {
                    span: span.clone(),
                    from: from.clone(),
                    total: *value,
                    readings: 1,
                    quorum: *quorum,
                    aspects: aspects.clone(),
                    round: self.round,
                    last: round,
                    kinds: Vec::new(),
                }),
            }
        }
        self.fade_unread();
    }
    /// Runners not re-read this round fade toward 0 by `fade` (nothing with `fade` 0).
    fn fade_unread(&mut self) {
        if self.fade > 0.0 {
            let (round, keep) = (self.round, 1.0 - self.fade);
            for r in self.runners.iter_mut().filter(|r| r.last != round) {
                r.total *= keep;
            }
        }
    }
    /// Whether a runner holds a quorum: by measured independence of its kinds when set, else by
    /// the drones' family count at `rho`.
    fn has_quorum(&self, r: &Runner) -> bool {
        self.independence.as_ref().map_or_else(|| self.quorum(r.quorum), |ind| ind.quorum(&r.kinds))
    }
    fn total(&self, key: &str) -> f32 {
        self.runners.iter().find(|r| r.span.key == key).map_or(f32::NEG_INFINITY, |r| r.total)
    }
    /// Order keys by total, best first (ties by key).
    pub fn order(&self, keys: &mut [String]) {
        keys.sort_by(|a, b| self.total(b).total_cmp(&self.total(a)).then(a.cmp(b)));
    }
    /// Effective count of `q` families at the request's correlation.
    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "family counts")]
    pub fn effective(&self, q: usize) -> f32 {
        if q == 0 { 0.0 } else { q as f32 / (1.0 + (q as f32 - 1.0) * self.rho) }
    }
    /// A quorum: more than half of the independent families, both at their effective counts.
    #[must_use]
    pub fn quorum(&self, q: usize) -> bool {
        2.0 * self.effective(q) > self.effective(DRONE_FAMILIES)
    }
    /// The race's state now.
    #[must_use]
    pub fn verdict(&self) -> Verdict {
        let mut by: Vec<&Runner> = self.runners.iter().collect();
        by.sort_by(|a, b| b.total.total_cmp(&a.total).then(a.span.key.cmp(&b.span.key)));
        let mut committed: Vec<String> = by.iter().filter(|r| r.total > 0.0 && self.has_quorum(r)).map(|r| r.span.key.clone()).collect();
        let mut ties = Vec::new();
        let mut uncovered = Vec::new();
        for a in 0..self.aspects {
            let holders: Vec<&&Runner> = by.iter().filter(|r| r.aspects.contains(&a)).collect();
            let Some(leader) = holders.first() else {
                uncovered.push(a);
                continue;
            };
            let rival = holders.get(1);
            let lead = rival.map_or(f32::INFINITY, |r| leader.total - r.total);
            if lead > self.band || self.has_quorum(leader) {
                if !committed.contains(&leader.span.key) {
                    committed.push(leader.span.key.clone());
                }
            } else if let Some(r) = rival {
                ties.push((a, leader.span.key.clone(), r.span.key.clone()));
            }
        }
        self.order(&mut committed);
        let forward: Vec<&String> = committed.iter().chain(ties.iter().flat_map(|(_, a, b)| [a, b])).collect();
        let queued = by.iter().map(|r| r.span.key.clone()).filter(|k| !forward.contains(&k)).collect();
        Verdict { committed, ties, uncovered, queued }
    }
}

/// The reference graph handed to G1, behind a name for its debug view.
#[derive(Clone)]
pub struct StructureRef(pub std::sync::Arc<dyn Structure>);
impl std::fmt::Debug for StructureRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0.describe())
    }
}

/// A decision of the reasoning layer for the next round.
#[derive(Clone, Debug, PartialEq)]
pub enum Decision {
    Reread { tentacle: usize, focus: Focus, why: String },
    Ask { tentacle: usize, span: SpanRef, why: String },
}

/// The budget of one loop: time and how many spans go up.
#[derive(Clone, Debug, PartialEq)]
pub struct Budget {
    pub time_ms: f64,
    pub forward: usize,
    /// How the two were set.
    pub why: String,
    /// Route capacity of one round in tokens (`ForwardRule::Queue`; refinement step 1).
    pub capacity: Capacity,
}
impl Budget {
    /// Forward count from the pack's token room over the mean span tokens (fallback: one per
    /// tentacle); time from the caller (fallback: none, the derived ceiling bounds the loop).
    #[must_use]
    pub fn derive(tentacles: usize, pack_tokens: Option<usize>, mean_span_tokens: Option<usize>, time_ms: Option<f64>) -> Self {
        let (forward, f_why) = match (pack_tokens, mean_span_tokens) {
            (Some(p), Some(m)) if m > 0 => ((p / m).max(1), format!("forward {} = pack {p} tokens / mean span {m} tokens", (p / m).max(1))),
            _ => (tentacles.max(1), format!("forward {} = one per tentacle (fallback)", tentacles.max(1))),
        };
        let (time_ms, t_why) = match time_ms {
            Some(t) if t > 0.0 => (t, format!("time {t:.0} ms from the caller")),
            _ => (f64::INFINITY, "no time limit given (fallback): the ceiling from re-reads and held spans bounds the loop".to_owned()),
        };
        Self { time_ms, forward, why: format!("{f_why}; {t_why}"), capacity: Capacity::derive(None) }
    }
    /// The same budget with the round capacity from the receiving room (`Capacity::derive`).
    #[must_use]
    pub fn with_room(mut self, room: Option<usize>) -> Self {
        self.capacity = Capacity::derive(room);
        self
    }
}

/// The coordination layer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Coordination {
    pub ids: Vec<String>,
    pub gains: Vec<f32>,
    /// G2's gain per tentacle (I9); not changed by G1's own gain update.
    pub g2_gains: Vec<f32>,
    /// G1 reflexes installed by G2 (`crate::reflex`): skip no-information tentacles; log re-read-low.
    pub reflex_skip: bool,
    pub reflex_low: bool,
    /// Layer file index to the tentacle whose window holds it.
    pub held_by: BTreeMap<usize, usize>,
    /// Held spans per tentacle (references).
    pub held: Vec<Vec<SpanRef>>,
    /// Spans reported so far (labels).
    pub reported: BTreeSet<String>,
    pub rereads: Vec<u32>,
    /// (tentacle, span key) already asked.
    pub asked: BTreeSet<(usize, String)>,
    pub sent: usize,
    /// Orders G1 relayed to one tentacle about a span another tentacle reported.
    pub relayed: usize,
    /// Orders G1 itself failed to stamp as its own (must stay 0).
    pub misstamped: usize,
    pub round: u32,
}
impl Coordination {
    fn new(cb: &WholeCodebase, ts: &[Box<dyn TentacleSide>]) -> Self {
        let held: Vec<Vec<SpanRef>> = ts.iter().map(|t| t.held()).collect();
        let mut held_by = BTreeMap::new();
        for (t, h) in held.iter().enumerate() {
            for s in h {
                if let Some(i) = cb.file_index(&s.file) {
                    held_by.entry(i).or_insert(t);
                }
            }
        }
        Self {
            ids: ts.iter().map(|t| t.id().to_owned()).collect(),
            gains: vec![1.0; ts.len()],
            held_by,
            held,
            rereads: vec![0; ts.len()],
            ..Self::default()
        }
    }
    /// Gain per tentacle from its report confidence: `(c + mean) / (2 mean)`, so the mean gain
    /// is 1 and a tentacle with no information keeps half weight (all 1 when no confidence).
    fn update_gains(&mut self, reports: &[Report]) {
        let n = reports.len().max(1) as f32;
        let mean = reports.iter().map(|r| r.confidence).sum::<f32>() / n;
        self.gains = reports.iter().map(|r| if mean > 0.0 { (r.confidence + mean) / (2.0 * mean) } else { 1.0 }).collect();
    }
    /// Send one order through G1 (stamped with G1's id).
    fn send(&mut self, t: &mut dyn TentacleSide, round: u32, confidence: f32, why: &str, ins: Instruction) -> bool {
        let Ok(o) = Order::new(G1_ID, t.id(), round, confidence, why, ins) else { return false };
        if o.msg.sender() != G1_ID {
            self.misstamped += 1;
            return false;
        }
        let ok = t.deliver(o).is_ok();
        self.sent += usize::from(ok);
        ok
    }
    /// The isolation gate: envelopes any `g0_tx` refused from a non-G1 sender, plus G1's own
    /// misstamped orders. Must stay 0.
    #[must_use]
    pub fn gate(&self, ts: &[Box<dyn TentacleSide>]) -> usize {
        ts.iter().map(|t| t.inbox().refused_direct).sum::<usize>() + self.misstamped
    }
    /// Layer files no window holds.
    #[must_use]
    pub fn unread(&self, cb: &WholeCodebase) -> Vec<usize> {
        (0..cb.files.len()).filter(|i| !self.held_by.contains_key(i)).collect()
    }
}

/// The reasoning layer.
#[derive(Debug, Default)]
pub struct Reasoning {
    pub buffer: TensionBuffer,
    /// Answers by (tentacle, span key).
    pub answers: BTreeMap<(usize, String), Answer>,
    pub conflicts: Vec<String>,
}
impl Reasoning {
    /// Candidates of one round from every report and every answer.
    #[expect(clippy::too_many_arguments, reason = "one private step of the loop; the inputs are the round's separate states")]
    fn candidates(
        &mut self,
        reports: &[Report],
        co: &Coordination,
        cb: &WholeCodebase,
        rel: &Relevance,
        combine: Combine,
        structure: Option<&dyn Structure>,
        common: Option<&Pool>,
    ) -> Vec<Candidate> {
        let mut out: Vec<Candidate> = Vec::new();
        for (t, r) in reports.iter().enumerate() {
            for b in &r.best {
                let mut reasons = Vec::new();
                if b.place == 0 {
                    reasons.push(Reason::FirstChoice);
                }
                if r.reads > 1 {
                    reasons.push(Reason::AfterReread(r.reads));
                }
                let facts = r.facts.get(&b.span.key).cloned().unwrap_or_default();
                out.push(Candidate {
                    span: b.span.clone(),
                    tentacle: t,
                    from: r.from.clone(),
                    place: b.place,
                    score: b.score,
                    confidence: b.confidence,
                    support: 0.0,
                    agreement: 0,
                    combined: 0.0,
                    reasons,
                    facts,
                    structure: 0.0,
                });
            }
        }
        // Answers: a confident answer about an unreported span makes it a candidate (aimed).
        for ((t, key), a) in &self.answers {
            if let Some(c) = out.iter_mut().find(|c| c.tentacle == *t && c.span.key == *key) {
                c.confidence = a.confidence;
                c.reasons.push(Reason::Answered(a.confidence));
            } else if a.held && !a.no_information {
                let facts = reports.get(*t).and_then(|r| r.facts.get(&a.span.key)).cloned().unwrap_or_default();
                out.push(Candidate {
                    span: a.span.clone(),
                    tentacle: *t,
                    from: a.from.clone(),
                    place: a.rank.unwrap_or(usize::MAX),
                    score: a.score,
                    confidence: a.confidence,
                    support: 0.0,
                    agreement: 0,
                    combined: 0.0,
                    reasons: vec![Reason::Aimed, Reason::Answered(a.confidence)],
                    facts,
                    structure: 0.0,
                });
            }
        }
        // Reference-graph rank received from every tentacle's first choice (refinement step 1).
        if let Some(st) = structure {
            let seeds: Vec<(String, f32)> = out.iter().filter(|c| c.place == 0).map(|c| (c.span.key.clone(), 1.0)).collect();
            let ask: Vec<String> = out.iter().map(|c| c.span.key.clone()).collect();
            let r = st.rank(&seeds, &ask);
            let top = r.iter().copied().fold(0.0_f32, f32::max);
            for (c, v) in out.iter_mut().zip(r) {
                c.structure = if top > 0.0 { v / top } else { 0.0 };
            }
        }
        // Support from the whole-code-base layer, normalized by the round's largest.
        let raw: Vec<f32> = out.iter().map(|c| cb.support(rel, &c.span)).collect();
        let top = raw.iter().copied().fold(0.0_f32, f32::max);
        let files: Vec<(String, usize)> = out.iter().map(|c| (c.span.file.clone(), c.tentacle)).collect();
        for (c, r) in out.iter_mut().zip(raw) {
            c.support = if top > 0.0 { r / top } else { 0.0 };
            if c.support > 0.0 {
                c.reasons.push(Reason::Corroborated(c.support));
            }
            let mut others: Vec<usize> = files.iter().filter(|(f, t)| *f == c.span.file && *t != c.tentacle).map(|(_, t)| *t).collect();
            others.sort_unstable();
            others.dedup();
            c.agreement = others.len();
            if c.agreement > 0 {
                c.reasons.push(Reason::Agreement(c.agreement));
            }
            let gain = co.gains.get(c.tentacle).copied().unwrap_or(1.0) * co.g2_gains.get(c.tentacle).copied().unwrap_or(1.0);
            c.combined = combined_value(c, combine, gain, reports.len());
        }
        if combine == Combine::Pooled {
            // On the scale shared by all tentacles (`common`) a piece's value is its own and
            // G2's gain alone: no factor that depends on which window or set reported it.
            let pool = common.cloned().unwrap_or_else(|| Pool::of(&out));
            for c in &mut out {
                let own = if common.is_some() { 1.0 } else { co.gains.get(c.tentacle).copied().unwrap_or(1.0) };
                let gain = own * co.g2_gains.get(c.tentacle).copied().unwrap_or(1.0);
                c.combined = gain * pool.value(c);
            }
        }
        // Conflicts: the same key observed with different content by two tentacles.
        let packets: Vec<Packet> = out
            .iter()
            .map(|c| {
                let src = format!("{}:{}", c.span.file, c.span.start);
                Packet::new("span", &format!("span:{src}"), &c.span.key, Provenance::new(&src, &c.from, 0, 0))
                    .confidence(c.confidence)
                    .evidence(vec![src.clone()])
            })
            .collect();
        let labelled = arbiter::arbitrate(&packets, &BTreeSet::new());
        for q in self.buffer.hold(&labelled, 0) {
            let key = q.key().trim_start_matches("conflict:").to_owned();
            if !self.conflicts.contains(&key) {
                self.conflicts.push(key);
            }
        }
        for (c, p) in out.iter_mut().zip(&labelled) {
            if p.get_status() == Status::Contradicted {
                c.reasons.push(Reason::ConflictHeld(p.key().to_owned()));
            }
        }
        out
    }
    /// The spans that go up: best combined first, no two overlapping, at most `k`.
    fn select(cands: &[Candidate], k: usize) -> Vec<Candidate> {
        let mut v: Vec<&Candidate> = cands.iter().filter(|c| c.combined > 0.0).collect();
        v.sort_by(|a, b| b.combined.total_cmp(&a.combined).then(a.tentacle.cmp(&b.tentacle)).then(a.place.cmp(&b.place)));
        let mut out: Vec<Candidate> = Vec::new();
        for c in v {
            if out.len() >= k {
                break;
            }
            if !out.iter().any(|o| o.span.overlaps(&c.span)) {
                out.push(c.clone());
            }
        }
        out
    }
    /// Queue rule: the best candidates by combined, no two overlapping, while their tokens fit
    /// the round's capacity (at least one). Tokens from the span's facts, else `fallback`.
    fn select_capacity(cands: &[Candidate], cap: &Capacity, fallback: usize) -> Vec<Candidate> {
        let mut v: Vec<&Candidate> = cands.iter().filter(|c| c.combined > 0.0).collect();
        v.sort_by(|a, b| b.combined.total_cmp(&a.combined).then(a.tentacle.cmp(&b.tentacle)).then(a.place.cmp(&b.place)));
        let mut out: Vec<Candidate> = Vec::new();
        let mut used = 0;
        for c in v {
            if out.iter().any(|o| o.span.overlaps(&c.span)) {
                continue;
            }
            let t = if c.facts.tokens > 0 { c.facts.tokens } else { fallback.max(1) };
            if out.is_empty() || used + t <= cap.tokens {
                used += t;
                out.push(c.clone());
            }
        }
        out
    }
    /// Decisions for the next round.
    #[expect(clippy::too_many_arguments, reason = "one private step of the loop; the inputs are the round's separate states")]
    fn decide(
        &self,
        reports: &[Report],
        cands: &[Candidate],
        co: &Coordination,
        cb: &WholeCodebase,
        rel: &Relevance,
        options: &[usize],
        k: usize,
    ) -> Vec<Decision> {
        let mut out = Vec::new();
        // Low confidence: below half the median of the informative reports' confidences.
        let mut confs: Vec<f32> = reports.iter().filter(|r| !r.no_information).map(|r| r.confidence).collect();
        confs.sort_by(f32::total_cmp);
        let median = confs.get(confs.len() / 2).copied().unwrap_or(0.0);
        let low = median / 2.0;
        let ranked = cb.ranked_files(rel);
        let mut rereading = BTreeSet::new();
        for (t, r) in reports.iter().enumerate() {
            if !(r.no_information || r.confidence < low) || options.get(t).copied().unwrap_or(0) == 0 {
                continue;
            }
            // G1 skip reflex: a tentacle that reported no information is not re-read this round.
            if r.no_information && co.reflex_skip {
                continue;
            }
            let held = co.held.get(t).map_or(&[][..], Vec::as_slice);
            let files: Vec<String> = ranked
                .iter()
                .filter_map(|i| cb.files.get(*i))
                .filter(|f| held.iter().any(|h| h.file == **f))
                .take(r.best.len().max(1))
                .cloned()
                .collect();
            let why = if r.no_information {
                format!("{} reported no information against baseline {:.3}", r.from, r.baseline)
            } else {
                format!("{} confidence {:.2} below {low:.2}", r.from, r.confidence)
            };
            out.push(Decision::Reread { tentacle: t, focus: Focus { files, terms: rel.terms.clone(), ..Focus::default() }, why });
            rereading.insert(t);
        }
        // Specific questions: a span below first place with more layer support than its
        // tentacle's first choice.
        for c in cands.iter().filter(|c| c.place > 0 && c.place != usize::MAX && c.support > 0.0) {
            let first = cands.iter().filter(|o| o.tentacle == c.tentacle && o.place == 0).map(|o| o.support).fold(0.0_f32, f32::max);
            if c.support > first && !co.asked.contains(&(c.tentacle, c.span.key.clone())) && !rereading.contains(&c.tentacle) {
                out.push(Decision::Ask {
                    tentacle: c.tentacle,
                    span: c.span.clone(),
                    why: format!("place {} corroborated by the whole-code-base layer ({:.2} over {:.2})", c.place + 1, c.support, first),
                });
            }
        }
        // Coverage: the most relevant files held by a tentacle that did not report them.
        for i in ranked.iter().take(k) {
            let Some(&t) = co.held_by.get(i) else { continue };
            let Some(file) = cb.files.get(*i) else { continue };
            if rereading.contains(&t) || cands.iter().any(|c| c.tentacle == t && c.span.file == *file) {
                continue;
            }
            let held = co.held.get(t).map_or(&[][..], Vec::as_slice);
            let item = cb.best_item(rel, *i);
            let span = held
                .iter()
                .filter(|h| h.file == *file)
                .find(|h| item.is_some_and(|it| h.contains(&it.file, it.line)))
                .or_else(|| held.iter().find(|h| h.file == *file));
            if let Some(s) = span
                && !co.asked.contains(&(t, s.key.clone()))
                && !out.iter().any(|d| matches!(d, Decision::Ask { span, .. } if span.key == s.key))
            {
                out.push(Decision::Ask { tentacle: t, span: s.clone(), why: format!("relevant file {file} held but not reported") });
            }
        }
        // Conflicts: ask each holder about its side once.
        for c in cands.iter().filter(|c| c.reasons.iter().any(|r| matches!(r, Reason::ConflictHeld(_)))) {
            if !co.asked.contains(&(c.tentacle, c.span.key.clone()))
                && !out
                    .iter()
                    .any(|d| matches!(d, Decision::Ask { tentacle, span, .. } if *tentacle == c.tentacle && span.key == c.span.key))
            {
                out.push(Decision::Ask { tentacle: c.tentacle, span: c.span.clone(), why: "conflicting observation held".into() });
            }
        }
        out
    }
}

/// One forwarded span.
#[derive(Clone, Debug, PartialEq)]
pub struct Forwarded {
    pub span: SpanRef,
    pub from: String,
    pub confidence: f32,
    pub combined: f32,
    pub reasons: Vec<Reason>,
}

/// The typed hand-over from G1 to G2.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Handover {
    pub question: String,
    pub items: Vec<Forwarded>,
    /// Mean confidence of the forwarded spans (0 when none).
    pub confidence: f32,
    /// Layer files no window holds, and their share of the layer.
    pub unread_files: usize,
    pub unread_share: f32,
    /// The most relevant unread files (G2 may direct a tentacle there).
    pub unread_relevant: Vec<String>,
    /// Held spans no tentacle reported or was asked about.
    pub held_unreported: usize,
    pub rounds: u32,
    pub stop: String,
    pub gate_direct: usize,
    pub relayed: usize,
    pub conflicts: Vec<String>,
}
impl Handover {
    /// The hand-over as in-house report messages (one per forwarded span, G1 to G2), the
    /// reasons as the description and the span as the reference.
    #[must_use]
    pub fn messages(&self) -> Vec<nmtype::Envelope> {
        self.items
            .iter()
            .filter_map(|f| {
                let why = f.reasons.iter().map(Reason::text).collect::<Vec<_>>().join("; ");
                let why = if why.is_empty() { "forwarded".to_owned() } else { why };
                let r = nmtype::Reference::Span {
                    path: f.span.file.clone(),
                    start: u32::try_from(f.span.start.max(1)).unwrap_or(u32::MAX),
                    end: u32::try_from(f.span.end.max(f.span.start).max(1)).unwrap_or(u32::MAX),
                };
                nmtype::Envelope::new(nmtype::MessageKind::Report, G1_ID, G2_ID, f.confidence, &why, vec![r], Vec::new()).ok()
            })
            .collect()
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let n = |x: usize| Value::Number(x.to_string());
        let f = |x: f32| Value::Number(format!("{x:.3}"));
        let s = |t: &str| Value::String(t.to_owned());
        let obj = |v: Vec<(&str, Value)>| Value::Object(v.into_iter().map(|(k, x)| (k.to_owned(), x)).collect());
        let items = self
            .items
            .iter()
            .map(|i| {
                obj(vec![
                    ("span", s(&i.span.label())),
                    ("key", s(&i.span.key)),
                    ("from", s(&i.from)),
                    ("confidence", f(i.confidence)),
                    ("combined", f(i.combined)),
                    ("reasons", Value::Array(i.reasons.iter().map(|r| s(&r.text())).collect())),
                ])
            })
            .collect();
        obj(vec![
            ("question", s(&self.question)),
            ("items", Value::Array(items)),
            ("confidence", f(self.confidence)),
            ("unread_files", n(self.unread_files)),
            ("unread_share", f(self.unread_share)),
            ("unread_relevant", Value::Array(self.unread_relevant.iter().map(|x| s(x)).collect())),
            ("held_unreported", n(self.held_unreported)),
            ("rounds", n(self.rounds as usize)),
            ("stop", s(&self.stop)),
            ("gate_direct", n(self.gate_direct)),
            ("relayed", n(self.relayed)),
            ("conflicts", Value::Array(self.conflicts.iter().map(|x| s(x)).collect())),
        ])
    }
}

/// What one round did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RoundLog {
    pub round: u32,
    pub ms: f64,
    pub rereads: Vec<String>,
    pub asks: Vec<String>,
    pub forwarded: Vec<String>,
    pub changed: bool,
    /// G1 reflexes that fired this round, one line each.
    pub reflexes: Vec<String>,
}

/// The result of one loop.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outcome {
    pub handover: Handover,
    pub log: Vec<RoundLog>,
    pub ceiling: u32,
    /// The evidence set of the question (refinement step 1): the forwarded spans' records and
    /// the queue of everything else (reported spans not forwarded, then held spans below gate 1),
    /// with the round capacity. Built under both forward rules; the bridge passes it on only
    /// under `ForwardRule::Queue`.
    pub evidence: EvidenceSet,
    /// Every tentacle's first report as G1 received it (for the solo-equivalence check).
    pub first: Vec<Report>,
    /// Every tentacle's last report.
    pub last: Vec<Report>,
}

// ---------------------------------------------------------------------------------------------
// The order ledger (master design 3.6 rule 7; tentacle master plan 3.6 and Phase G1-b, lane M
// 2026-11-19): G1's coordination layer keeps it, moved from `G2Run.motor`. Every `nm-msg/2`
// envelope down is recorded here first; recording returns an `Ack`, a token only this module can
// make, and every delivery on the motor path takes that token and checks it against the ledger
// (`Ledger::verify`), so the copy to G1 cannot be skipped by construction. The ledger is a shared
// handle: a clone is the same ledger (G1 owns it; the motor path holds a handle to it).
// ---------------------------------------------------------------------------------------------

/// The ledger's content: acknowledged envelopes (digest to id) and one line per envelope, in order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LedgerBook {
    pub acked: BTreeMap<String, u64>,
    pub lines: Vec<String>,
}

/// G1's acknowledgement of one recorded envelope. Its fields are private: only `Ledger::record`
/// makes one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ack {
    digest: String,
    id: u64,
}
impl Ack {
    #[must_use]
    pub fn digest(&self) -> &str {
        &self.digest
    }
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }
}

/// G1's order ledger (a shared handle).
#[derive(Clone, Debug, Default)]
pub struct Ledger(std::sync::Arc<std::sync::Mutex<LedgerBook>>);
impl PartialEq for Ledger {
    fn eq(&self, o: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &o.0) || self.book() == o.book()
    }
}
fn ledger_fault(m: impl Into<String>) -> nmtype::Fault {
    nmtype::Fault::new("nmbrain.g1.ledger", nmtype::FaultKind::Policy, m)
}
impl Ledger {
    fn lock(&self) -> std::sync::MutexGuard<'_, LedgerBook> {
        self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    /// Record an envelope (G1's copy) and acknowledge it.
    /// # Errors
    /// An envelope whose copy to G1 is off.
    pub fn record(&self, m: &nmtype::message::Motor) -> Result<Ack, nmtype::Fault> {
        if !m.copy_to_g1() {
            return Err(ledger_fault(format!("order {} without its copy to G1", m.id())));
        }
        let d = m.digest();
        let mut b = self.lock();
        b.acked.insert(d.clone(), m.id());
        b.lines.push(format!("{} {} {} -> {}: {}", m.id(), m.verb().name(), m.sender(), m.receiver(), m.description()));
        Ok(Ack { digest: d, id: m.id() })
    }
    /// Whether an envelope with this digest is recorded.
    #[must_use]
    pub fn acknowledged(&self, digest: &str) -> bool {
        self.lock().acked.contains_key(digest)
    }
    /// The admission check of an inbox: the token is this envelope's and the ledger holds it.
    #[must_use]
    pub fn verify(&self, ack: &Ack, m: &nmtype::message::Motor) -> bool {
        ack.id == m.id() && ack.digest == m.digest() && self.lock().acked.get(&ack.digest) == Some(&ack.id)
    }
    /// A copy of the content.
    #[must_use]
    pub fn book(&self) -> LedgerBook {
        self.lock().clone()
    }
    #[must_use]
    pub fn lines(&self) -> Vec<String> {
        self.lock().lines.clone()
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().lines.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------------------------
// Belief per item in G1 (Phase G2-u; direction 2026-11-17 07:12; lane M 2026-11-19), behind the
// switch `NANO_ITEM_BELIEF=on` (off by default), with the measured causes of C64 fixed (C65,
// T-031 to T-034): (1) length normalised: an item's value less the request's own slope of value on
// log length (the nuisance fit, conflict 31), so long items no longer win by length; (2) the
// item's neighbour lines go with it by reference (they are kept, never counted as its value);
// (3) room is counted in tokens, in G2 (`crate::g2::fill_items`).
// ---------------------------------------------------------------------------------------------

/// Whether belief per item is on: on by default as a design part, `NANO_ITEM_BELIEF=off` for
/// diagnosis (the same default the served flow reads in `nmapi`; Phase 4 inventory 2026-11-24 found
/// this function reading the opposite default).
#[must_use]
pub fn item_belief_on() -> bool {
    !std::env::var("NANO_ITEM_BELIEF").is_ok_and(|v| v == "off")
}

/// One item (a definition with its comment) as G1 holds it.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemIn {
    pub key: String,
    pub tokens: usize,
    /// The item's evidence on the log-odds scale (its fused value).
    pub value: f32,
    /// The lines beside it (the piece's neighbour lines), by reference.
    pub neighbours: Vec<String>,
}

/// An item's belief: its length-normalised value and its neighbours by reference.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemBelief {
    pub key: String,
    pub tokens: usize,
    pub value: f32,
    pub neighbours: Vec<String>,
}

/// The request's slope of value on log length (least squares over its items; 0 when fewer than
/// three items or no spread in length; never below 0, so short items are not favoured either).
#[must_use]
#[expect(clippy::cast_precision_loss, reason = "token and item counts")]
pub fn length_slope(items: &[ItemIn]) -> f32 {
    let xs: Vec<(f32, f32)> = items.iter().filter(|i| i.value.is_finite() && i.tokens > 0).map(|i| ((i.tokens as f32).ln(), i.value)).collect();
    if xs.len() < 3 {
        return 0.0;
    }
    let n = xs.len() as f32;
    let (mx, my) = (xs.iter().map(|p| p.0).sum::<f32>() / n, xs.iter().map(|p| p.1).sum::<f32>() / n);
    let sxx: f32 = xs.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
    if sxx <= f32::EPSILON {
        return 0.0;
    }
    (xs.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f32>() / sxx).max(0.0)
}

/// Belief per item: each item's value less `slope * ln(tokens / median tokens)`, best first (ties
/// by key); neighbours carried by reference.
#[must_use]
#[expect(clippy::cast_precision_loss, reason = "token counts")]
pub fn item_beliefs(items: &[ItemIn]) -> Vec<ItemBelief> {
    let slope = length_slope(items);
    let mut t: Vec<usize> = items.iter().map(|i| i.tokens.max(1)).collect();
    t.sort_unstable();
    let med = t.get(t.len() / 2).copied().unwrap_or(1) as f32;
    let mut out: Vec<ItemBelief> = items
        .iter()
        .map(|i| ItemBelief {
            key: i.key.clone(),
            tokens: i.tokens,
            value: if i.value.is_finite() { i.value - slope * ((i.tokens.max(1) as f32) / med).ln() } else { f32::NEG_INFINITY },
            neighbours: i.neighbours.clone(),
        })
        .collect();
    out.sort_by(|a, b| b.value.total_cmp(&a.value).then(a.key.cmp(&b.key)));
    out
}

/// G1: the whole-code-base layer and the two internal layers.
#[derive(Debug, Default)]
pub struct G1 {
    pub codebase: WholeCodebase,
    pub coordination: Coordination,
    pub reasoning: Reasoning,
    /// How candidates are combined (from the process rules at `new`).
    pub combine: Combine,
    /// How spans go up (from the process rules at `new`; refinement step 1).
    pub forward: ForwardRule,
    /// The reference graph over items (refinement step 1), when the caller gives one.
    pub structure: Option<StructureRef>,
    /// G2's gain per tentacle (I9, `crate::instruct`), multiplied into G1's own; empty: all 1.
    pub g2_gains: Vec<f32>,
    /// G2's instructions G1 relays to the tentacles in round 1 (I1, I2, I3): tentacle, instruction, reason.
    pub relay: Vec<(usize, Instruction, String)>,
    /// G1 reflexes installed by G2: skip and re-read-low.
    pub reflex_skip: bool,
    pub reflex_low: bool,
    /// The allocation map of the question (`crate::allocation`): who holds which piece, how and
    /// since which round, what was read, reported, forwarded, queued or passed over. G1 is its one
    /// writer: every allocation decision is applied here by claim, and a claim on a held piece is
    /// refused.
    pub map: crate::allocation::AllocationMap,
    /// The scale shared by all tentacles (`Pool::over` every piece of the material for this
    /// request): with `Combine::Pooled`, every reported and held piece is valued on it. `None`:
    /// the pool of the merged set.
    pub common: Option<Pool>,
    /// The race over every round's candidates (on the common scale only); kept across the rounds
    /// of one request. `racing`: the race decides what goes up (else the round's capacity does).
    pub race: Race,
    pub racing: bool,
    /// The order ledger of G1's coordination layer (G1-b): every envelope down recorded first.
    pub ledger: Ledger,
}

/// Run `f` on every tentacle in parallel (one scoped thread each). A tentacle that panics is
/// quarantined for the round: its slot is `None`.
fn each<R: Send>(ts: &mut [Box<dyn TentacleSide>], f: impl Fn(&mut dyn TentacleSide) -> R + Sync) -> Vec<Option<R>> {
    let f = &f;
    std::thread::scope(|s| {
        let hs: Vec<_> = ts.iter_mut().map(|t| s.spawn(move || f(t.as_mut()))).collect();
        hs.into_iter().map(|h| h.join().ok()).collect()
    })
}

/// What a report says for change detection: its spans and the no-information flag.
fn gist(r: &Report) -> (Vec<String>, bool) {
    (r.best.iter().map(|b| b.span.key.clone()).collect(), r.no_information)
}

impl G1 {
    #[must_use]
    pub fn new(codebase: WholeCodebase) -> Self {
        let r = rules();
        Self { codebase, combine: r.combine, forward: r.forward, ..Self::default() }
    }
    /// Record an order envelope in G1's ledger (the enforced copy); the `Ack` is what an inbox
    /// admits it by.
    /// # Errors
    /// As `Ledger::record`.
    pub fn record_order(&self, m: &nmtype::message::Motor) -> Result<Ack, nmtype::Fault> {
        self.ledger.record(m)
    }
    /// A handle to G1's ledger for the motor path (`crate::motor::Down::with_ledger`).
    #[must_use]
    pub fn ledger_handle(&self) -> Ledger {
        self.ledger.clone()
    }
    /// The same G1 with the reference graph (refinement step 1).
    #[must_use]
    pub fn with_structure(mut self, s: Option<StructureRef>) -> Self {
        self.structure = s;
        self
    }
    /// G1 reflexes installed by G2 (`crate::reflex`): skip no-information tentacles; log re-read-low.
    #[must_use]
    pub fn with_reflexes(mut self, skip: bool, low: bool) -> Self {
        self.reflex_skip = skip;
        self.reflex_low = low;
        self
    }
    /// G2's gains (I9) and the instructions to relay in round 1 (I1, I2, I3).
    #[must_use]
    pub fn with_g2(mut self, gains: Vec<f32>, relay: Vec<(usize, Instruction, String)>) -> Self {
        self.g2_gains = gains;
        self.relay = relay;
        self
    }
    /// The evidence set at the end of a loop: the forwarded spans (round 1), then every other
    /// candidate queued with why, then every held span no tentacle reported (below its gate 1),
    /// each with its evidence and its combined value under the rule in service; relations among
    /// the records from the reference graph.
    #[expect(clippy::too_many_arguments, reason = "one private step of the loop; the inputs are the loop's separate states")]
    fn evidence(
        &self,
        reports: &[Report],
        cands: &[Candidate],
        forwarded: &[Candidate],
        rel: &Relevance,
        budget: &Budget,
        structure: Option<&dyn Structure>,
    ) -> EvidenceSet {
        let rec = |c: &Candidate, priority: Priority, round: u32, why: Vec<String>| SpanEvidence {
            span: c.span.clone(),
            kept: vec![format!("kv:{}", c.span.key)],
            found_by: vec![c.from.clone()],
            lenses: if c.facts.lens.is_empty() { Vec::new() } else { vec![c.facts.lens.clone()] },
            agreement: c.agreement,
            support: c.support,
            structure: c.structure,
            confidence: c.confidence,
            combined: c.combined,
            reasons: c.reasons.iter().map(Reason::text).chain(why).collect(),
            priority,
            round,
            facts: c.facts.clone(),
            relations: Vec::new(),
        };
        let mut set = EvidenceSet {
            capacity: budget.capacity.clone(),
            rounds: 1,
            rule: format!("combine {:?}, forward {:?}", self.combine, self.forward),
            ..EvidenceSet::default()
        };
        let mut seen: BTreeMap<String, ()> = BTreeMap::new();
        for c in forwarded {
            seen.insert(c.span.key.clone(), ());
            set.forwarded.push(rec(c, Priority::Forwarded, 1, vec!["forwarded in round 1".into()]));
        }
        let mut sorted: Vec<&Candidate> = cands.iter().collect();
        sorted.sort_by(|a, b| b.combined.total_cmp(&a.combined).then(a.tentacle.cmp(&b.tentacle)).then(a.place.cmp(&b.place)));
        for c in sorted {
            if let Some(e) = set.forwarded.iter_mut().chain(set.queue.iter_mut()).find(|e| e.span.key == c.span.key) {
                if !e.found_by.contains(&c.from) {
                    e.found_by.push(c.from.clone());
                }
                continue;
            }
            seen.insert(c.span.key.clone(), ());
            let why = if c.combined <= 0.0 {
                "queued: combined value not above 0"
            } else if forwarded.iter().any(|f| f.span.overlaps(&c.span)) {
                "queued: overlaps a forwarded span"
            } else {
                "queued: beyond the round capacity"
            };
            set.queue.push(rec(c, Priority::Queued, 0, vec![why.into()]));
        }
        // Held spans below gate 1: same evidence, same rule.
        let n = reports.len();
        let mut held: Vec<Candidate> = Vec::new();
        for (t, r) in reports.iter().enumerate() {
            for (key, f) in &r.facts {
                if seen.contains_key(key) {
                    continue;
                }
                let Some(span) = self.coordination.held.get(t).and_then(|h| h.iter().find(|s| s.key == *key)).cloned() else { continue };
                seen.insert(key.clone(), ());
                let confidence = f.score("confidence").unwrap_or(0.0);
                held.push(Candidate {
                    span,
                    tentacle: t,
                    from: r.from.clone(),
                    place: f.window_rank,
                    score: f.score("calibrated").unwrap_or(0.0),
                    confidence,
                    support: 0.0,
                    agreement: 0,
                    combined: 0.0,
                    reasons: Vec::new(),
                    facts: f.clone(),
                    structure: 0.0,
                });
            }
        }
        if !held.is_empty() {
            let top_support = cands.iter().map(|c| self.codebase.support(rel, &c.span)).fold(0.0_f32, f32::max);
            let ranks = structure.map(|st| {
                let seeds: Vec<(String, f32)> = cands.iter().filter(|c| c.place == 0).map(|c| (c.span.key.clone(), 1.0)).collect();
                let ask: Vec<String> = held.iter().chain(cands.iter()).map(|c| c.span.key.clone()).collect();
                st.rank(&seeds, &ask)
            });
            let top_rank = ranks.as_ref().map_or(0.0, |r| r.iter().copied().fold(0.0_f32, f32::max));
            for (i, c) in held.iter_mut().enumerate() {
                let raw = self.codebase.support(rel, &c.span);
                c.support = if top_support > 0.0 { (raw / top_support).min(1.0) } else { 0.0 };
                let mut others: Vec<usize> =
                    cands.iter().filter(|o| o.span.file == c.span.file && o.tentacle != c.tentacle).map(|o| o.tentacle).collect();
                others.sort_unstable();
                others.dedup();
                c.agreement = others.len();
                c.structure = ranks.as_ref().and_then(|r| r.get(i)).map_or(0.0, |v| if top_rank > 0.0 { v / top_rank } else { 0.0 });
                let gain = self.coordination.gains.get(c.tentacle).copied().unwrap_or(1.0);
                c.combined = combined_value(c, self.combine, gain, n);
            }
            if self.combine == Combine::Pooled {
                let pool = self.common.clone().unwrap_or_else(|| Pool::of(cands));
                for c in &mut held {
                    let own = if self.common.is_some() { 1.0 } else { self.coordination.gains.get(c.tentacle).copied().unwrap_or(1.0) };
                    let gain = own * self.coordination.g2_gains.get(c.tentacle).copied().unwrap_or(1.0);
                    c.combined = gain * pool.value(c);
                }
            }
            held.sort_by(|a, b| b.combined.total_cmp(&a.combined).then(a.tentacle.cmp(&b.tentacle)).then(a.place.cmp(&b.place)));
            for c in &held {
                set.queue.push(rec(c, Priority::Queued, 0, vec!["queued: held below gate 1 of its tentacle".into()]));
            }
        }
        // Relations among the forwarded records and to the queue, from the reference graph.
        if let Some(st) = structure {
            let keys: BTreeSet<String> = set.all().map(|e| e.span.key.clone()).collect();
            for e in &mut set.forwarded {
                e.relations = st
                    .related(&e.span.key)
                    .into_iter()
                    .filter(|(k, _, _)| keys.contains(k) && *k != e.span.key)
                    .map(|(k, r, _)| (k, r))
                    .collect();
            }
        }
        set
    }
    #[expect(clippy::too_many_arguments, reason = "one private step of the loop; the inputs are the round's separate states")]
    fn handover(
        &self,
        question: &str,
        ts: &[Box<dyn TentacleSide>],
        items: Vec<Forwarded>,
        rel: &Relevance,
        k: usize,
        rounds: u32,
        stop: &str,
    ) -> Handover {
        let co = &self.coordination;
        let unread = co.unread(&self.codebase);
        let unread_relevant = self
            .codebase
            .ranked_files(rel)
            .into_iter()
            .filter(|i| unread.contains(i))
            .take(k)
            .filter_map(|i| self.codebase.files.get(i).cloned())
            .collect();
        let touched: BTreeSet<&String> = co.asked.iter().map(|(_, k)| k).collect();
        let held_unreported = co.held.iter().flatten().filter(|s| !co.reported.contains(&s.label()) && !touched.contains(&s.key)).count();
        let confidence = if items.is_empty() { 0.0 } else { items.iter().map(|i| i.confidence).sum::<f32>() / items.len() as f32 };
        Handover {
            question: question.to_owned(),
            items,
            confidence,
            unread_files: unread.len(),
            unread_share: if self.codebase.files.is_empty() { 0.0 } else { unread.len() as f32 / self.codebase.files.len() as f32 },
            unread_relevant,
            held_unreported,
            rounds,
            stop: stop.to_owned(),
            gate_direct: co.gate(ts),
            relayed: co.relayed,
            conflicts: self.reasoning.conflicts.clone(),
        }
    }
    /// The one-shot baseline: each tentacle's first choice goes up, nothing else.
    pub fn one_shot(&mut self, question: &str, ts: &mut [Box<dyn TentacleSide>]) -> Handover {
        self.coordination = Coordination::new(&self.codebase, ts);
        let rel = self.codebase.relevance(question);
        let reports: Vec<Report> = each(ts, |t| t.report()).into_iter().flatten().collect();
        let items: Vec<Forwarded> = reports
            .iter()
            .filter_map(|r| {
                r.best.first().map(|b| Forwarded {
                    span: b.span.clone(),
                    from: r.from.clone(),
                    confidence: b.confidence,
                    combined: b.confidence,
                    reasons: vec![Reason::FirstChoice],
                })
            })
            .collect();
        for r in &reports {
            self.coordination.reported.extend(r.best.iter().map(|b| b.span.label()));
        }
        let k = ts.len();
        self.handover(question, ts, items, &rel, k, 1, "one-shot")
    }
    /// The coordinating loop over the tentacles' `g0_tx`.
    pub fn run(&mut self, question: &str, ts: &mut [Box<dyn TentacleSide>], budget: &Budget) -> Outcome {
        let started = Instant::now();
        self.coordination = Coordination::new(&self.codebase, ts);
        self.coordination.g2_gains.clone_from(&self.g2_gains);
        self.coordination.reflex_skip = self.reflex_skip;
        self.coordination.reflex_low = self.reflex_low;
        self.reasoning = Reasoning::default();
        let rel = self.codebase.relevance(question);
        let ceiling = 1 + ts.iter().map(|t| t.options() + t.held().len()).sum::<usize>();
        let ceiling = u32::try_from(ceiling).unwrap_or(u32::MAX);
        let mut reports: Vec<Report> = Vec::new();
        let mut log: Vec<RoundLog> = Vec::new();
        let mut prev: Vec<String> = Vec::new();
        let mut forwarded: Vec<Candidate>;
        let mut last_cands: Vec<Candidate> = Vec::new();
        let mut first: Vec<Report> = Vec::new();
        let stop;
        loop {
            let t0 = Instant::now();
            self.coordination.round += 1;
            let round = self.coordination.round;
            let mut changed = round == 1;
            if round == 1 {
                reports = each(ts, |t| t.report())
                    .into_iter()
                    .enumerate()
                    .map(|(i, r)| {
                        r.unwrap_or_else(|| Report {
                            from: self.coordination.ids.get(i).cloned().unwrap_or_default(),
                            no_information: true,
                            ..Report::default()
                        })
                    })
                    .collect();
                first.clone_from(&reports);
            } else {
                let replies = each(ts, |t| t.process());
                for (t, rs) in replies.into_iter().enumerate() {
                    for reply in rs.unwrap_or_default() {
                        match reply {
                            Reply::Report(r) => {
                                if let Some(old) = reports.get_mut(t) {
                                    changed |= gist(old) != gist(&r);
                                    *old = *r;
                                }
                            }
                            Reply::Answer(a) => {
                                let key = (t, a.span.key.clone());
                                changed |= self.reasoning.answers.get(&key).is_none_or(|o| o != &a);
                                self.reasoning.answers.insert(key, a);
                            }
                        }
                    }
                }
            }
            for r in &reports {
                self.coordination.reported.extend(r.best.iter().map(|b| b.span.label()));
            }
            self.coordination.update_gains(&reports);
            let st = self.structure.as_ref().map(|s| &*s.0);
            let cands = self.reasoning.candidates(&reports, &self.coordination, &self.codebase, &rel, self.combine, st, self.common.as_ref());
            forwarded = match self.forward {
                ForwardRule::Fixed => Reasoning::select(&cands, budget.forward),
                ForwardRule::Queue => {
                    let toks: Vec<usize> = cands.iter().map(|c| c.facts.tokens).filter(|t| *t > 0).collect();
                    let mean = toks.iter().sum::<usize>().checked_div(toks.len()).unwrap_or(1);
                    Reasoning::select_capacity(&cands, &budget.capacity, mean)
                }
            };
            last_cands.clone_from(&cands);
            let labels: Vec<String> = forwarded.iter().map(|c| c.span.label()).collect();
            changed |= labels != prev;
            prev = labels.clone();
            let options: Vec<usize> = ts.iter().map(|t| t.options()).collect();
            let mut decisions = self.reasoning.decide(&reports, &cands, &self.coordination, &self.codebase, &rel, &options, budget.forward);
            // G2's instructions to relay go first in round 1 (I1 FOCUS and I2 REREAD as re-reads,
            // I3 ANSWER as a point read); only G1 sends to a tentacle.
            if round == 1 {
                let relayed: Vec<Decision> = self
                    .relay
                    .drain(..)
                    .map(|(tentacle, i, why)| match i {
                        Instruction::Reread(focus) => Decision::Reread { tentacle, focus, why },
                        Instruction::Ask { span, .. } => Decision::Ask { tentacle, span, why },
                    })
                    .collect();
                decisions.splice(0..0, relayed);
            }
            let mut entry = RoundLog { round, forwarded: labels, changed, ..RoundLog::default() };
            if self.reflex_skip {
                for r in reports.iter().filter(|r| r.no_information) {
                    entry.reflexes.push(format!("G1 skip: {} reported no information; not re-read this round", r.from));
                }
            }
            if self.reflex_low {
                for d in &decisions {
                    if let Decision::Reread { tentacle, why, .. } = d
                        && why.contains(" below ")
                    {
                        entry.reflexes.push(format!("G1 re-read low: tentacle {} ({why})", tentacle + 1));
                    }
                }
            }
            let mean_ms = (started.elapsed().as_secs_f64() * 1000.0) / f64::from(round);
            let halt = if !changed {
                Some("a round changed nothing")
            } else if decisions.is_empty() {
                Some("nothing left to ask")
            } else if round >= ceiling {
                Some("ceiling from re-reads and held spans")
            } else if started.elapsed().as_secs_f64() * 1000.0 + mean_ms > budget.time_ms {
                Some("time budget")
            } else {
                None
            };
            if halt.is_none() {
                for d in decisions {
                    match d {
                        Decision::Reread { tentacle, focus, why } => {
                            if let Some(t) = ts.get_mut(tentacle)
                                && self.coordination.send(
                                    t.as_mut(),
                                    round,
                                    reports.get(tentacle).map_or(0.0, |r| r.confidence),
                                    &why,
                                    Instruction::Reread(focus),
                                )
                            {
                                if let Some(n) = self.coordination.rereads.get_mut(tentacle) {
                                    *n += 1;
                                }
                                entry.rereads.push(self.coordination.ids.get(tentacle).cloned().unwrap_or_default());
                            }
                        }
                        Decision::Ask { tentacle, span, why } => {
                            let source = cands.iter().find(|c| c.span.key == span.key).map(|c| c.tentacle);
                            if let Some(t) = ts.get_mut(tentacle)
                                && self.coordination.send(
                                    t.as_mut(),
                                    round,
                                    0.5,
                                    &why,
                                    Instruction::Ask { span: span.clone(), about: question.to_owned() },
                                )
                            {
                                if source.is_some_and(|s| s != tentacle) {
                                    self.coordination.relayed += 1;
                                }
                                entry.asks.push(format!(
                                    "{} {}",
                                    self.coordination.ids.get(tentacle).cloned().unwrap_or_default(),
                                    span.label()
                                ));
                                self.coordination.asked.insert((tentacle, span.key));
                            }
                        }
                    }
                }
            }
            entry.ms = t0.elapsed().as_secs_f64() * 1000.0;
            log.push(entry);
            if let Some(h) = halt {
                stop = h;
                break;
            }
        }
        // On the common scale G1's race decides what goes up: this round's readings join every
        // earlier round's runners, and the round's candidates the race forwards go up (committed,
        // and both sides of each tie); every other candidate stays queued.
        if self.common.is_some() && self.racing {
            let items: Vec<(SpanRef, String, f32, usize, Vec<usize>)> =
                last_cands.iter().map(|c| (c.span.clone(), c.from.clone(), c.combined, c.facts.quorum, c.facts.aspects.clone())).collect();
            self.race.feed(&items);
            let up = self.race.verdict().forward(&self.race);
            let mut chosen: Vec<Candidate> = Vec::new();
            for k in &up {
                if let Some(c) = last_cands.iter().find(|c| c.span.key == *k)
                    && !chosen.iter().any(|o| o.span.key == *k)
                {
                    chosen.push(c.clone());
                }
            }
            forwarded = chosen;
        }
        let st = self.structure.as_ref().map(|s| &*s.0);
        let evidence = self.evidence(&reports, &last_cands, &forwarded, &rel, budget, st);
        let items = forwarded
            .into_iter()
            .map(|c| Forwarded { span: c.span, from: c.from, confidence: c.confidence, combined: c.combined, reasons: c.reasons })
            .collect();
        let rounds = self.coordination.round;
        let handover = self.handover(question, ts, items, &rel, budget.forward, rounds, stop);
        Outcome { handover, log, ceiling, evidence, first, last: reports }
    }
}

/// The question every constructed case asks.
pub const CASE_QUESTION: &str = "where is derive_limit computed for the kept chunks";

fn item(name: &str, file: &str, line: usize) -> ItemSig {
    ItemSig { name: name.into(), kind: "fn".into(), file: file.into(), line, signature: format!("pub fn {name}() -> usize") }
}

/// One constructed case: name, layer, tentacles, the right span's file.
pub type Case = (&'static str, WholeCodebase, Vec<Box<dyn TentacleSide>>, &'static str);

/// Constructed cases for the first measurement (each: name, layer, tentacles, the right span's
/// file). Case 0: right first in its tentacle. Case 1: right second and corroborated by the
/// layer. Case 2: right second, not corroborated, beside a tentacle with no information.
/// Case 3: right third (not reported) in a file the layer marks relevant. Case 4: right in a
/// file no window holds.
#[must_use]
pub fn cases() -> Vec<Case> {
    use crate::g0::scripted;
    let layer = |right: &str, extra: &[&str]| {
        let mut files: Vec<String> = ["a/wrong.rs", "a/z.rs", "b/x.rs", "b/y.rs", "c/p.rs", "c/q.rs", "d/r.rs", "d/s.rs", "e/far.rs"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        files.extend(extra.iter().map(|s| (*s).to_owned()));
        let mut items = vec![
            item("open_file", "a/wrong.rs", 5),
            item("tidy", "a/z.rs", 5),
            item("parse_args", "c/p.rs", 5),
            item("render_page", "d/r.rs", 5),
        ];
        if !right.is_empty() {
            files.push(right.to_owned());
            items.push(item("derive_limit", right, 5));
        }
        WholeCodebase::build(files, items)
    };
    let mut out: Vec<Case> = Vec::new();
    let rest = || -> Vec<Box<dyn TentacleSide>> {
        vec![
            Box::new(scripted(1, &["b/x.rs", "b/y.rs"], &[("mass", &[-0.2, -0.5]), ("evidence", &[-0.1, -0.4])])),
            Box::new(scripted(2, &["c/p.rs", "c/q.rs"], &[("mass", &[0.7, 0.2]), ("evidence", &[0.6, 0.1])])),
            Box::new(scripted(3, &["d/r.rs", "d/s.rs"], &[("mass", &[0.8, 0.3]), ("evidence", &[0.7, 0.2])])),
        ]
    };
    let with = |first: Box<dyn TentacleSide>| {
        let mut v = vec![first];
        v.extend(rest());
        v
    };
    out.push((
        "right first",
        layer("a/right.rs", &[]),
        with(Box::new(scripted(0, &["a/right.rs", "a/wrong.rs", "a/z.rs"], &[("mass", &[0.9, 0.6, 0.1]), ("evidence", &[0.8, 0.5, 0.0])]))),
        "a/right.rs",
    ));
    out.push((
        "right second, corroborated",
        layer("a/right.rs", &[]),
        with(Box::new(scripted(0, &["a/wrong.rs", "a/right.rs", "a/z.rs"], &[("mass", &[0.9, 0.6, 0.1]), ("evidence", &[0.8, 0.7, 0.0])]))),
        "a/right.rs",
    ));
    let mut uncorroborated = layer("", &["a/right.rs"]);
    uncorroborated.items.retain(|i| i.name != "derive_limit");
    out.push((
        "right second, not corroborated",
        uncorroborated,
        with(Box::new(scripted(0, &["a/wrong.rs", "a/right.rs", "a/z.rs"], &[("mass", &[0.9, 0.6, 0.1]), ("evidence", &[0.8, 0.5, 0.0])]))),
        "a/right.rs",
    ));
    out.push((
        "right third, relevant file",
        layer("a/right.rs", &[]),
        with(Box::new(scripted(
            0,
            &["a/wrong.rs", "a/z.rs", "a/right.rs"],
            &[("mass", &[0.9, 0.6, 0.4]), ("evidence", &[0.8, 0.5, 0.45])],
        ))),
        "a/right.rs",
    ));
    out.push((
        "right in no window",
        layer("e/right.rs", &[]),
        with(Box::new(scripted(0, &["a/wrong.rs", "a/z.rs"], &[("mass", &[0.9, 0.6]), ("evidence", &[0.8, 0.5])]))),
        "e/right.rs",
    ));
    out
}

/// First measurement: per constructed case, whether the loop and the one-shot baseline forward
/// the right span, the loop's rounds and its time per round on the processor.
#[must_use]
pub fn measure() -> Value {
    let n = |x: usize| Value::Number(x.to_string());
    let obj = |v: Vec<(&str, Value)>| Value::Object(v.into_iter().map(|(k, x)| (k.to_owned(), x)).collect());
    let mut rows = Vec::new();
    let (mut loop_reach, mut shot_reach, mut total_ms, mut total_rounds) = (0_usize, 0_usize, 0.0_f64, 0_u32);
    let mut count = 0_usize;
    for (name, cb, ts, right) in cases() {
        count += 1;
        let k = ts.len();
        let mut ts = ts;
        let mut g = G1::new(cb.clone());
        let out = g.run(CASE_QUESTION, &mut ts, &Budget::derive(k, None, None, None));
        let (_, _, ts2, _) = cases().into_iter().find(|c| c.0 == name).unwrap_or(("", WholeCodebase::default(), Vec::new(), ""));
        let mut ts2 = ts2;
        let mut g2 = G1::new(cb);
        let shot = g2.one_shot(CASE_QUESTION, &mut ts2);
        let hit_loop = out.handover.items.iter().any(|i| i.span.file == right);
        let hit_shot = shot.items.iter().any(|i| i.span.file == right);
        loop_reach += usize::from(hit_loop);
        shot_reach += usize::from(hit_shot);
        let ms: f64 = out.log.iter().map(|r| r.ms).sum();
        total_ms += ms;
        total_rounds += out.handover.rounds;
        rows.push(obj(vec![
            ("case", Value::String(name.into())),
            ("loop_forwards_right", Value::Bool(hit_loop)),
            ("one_shot_forwards_right", Value::Bool(hit_shot)),
            ("rounds", n(out.handover.rounds as usize)),
            ("stop", Value::String(out.handover.stop.clone())),
            ("ms_per_round", Value::Number(format!("{:.3}", ms / f64::from(out.handover.rounds.max(1))))),
            ("unread_relevant", Value::Array(out.handover.unread_relevant.iter().map(|f| Value::String(f.clone())).collect())),
            ("gate_direct", n(out.handover.gate_direct)),
        ]));
    }
    obj(vec![
        ("cases", n(count)),
        ("loop_reach", n(loop_reach)),
        ("one_shot_reach", n(shot_reach)),
        ("rounds_total", n(total_rounds as usize)),
        ("ms_per_round_mean", Value::Number(format!("{:.3}", total_ms / f64::from(total_rounds.max(1))))),
        ("rows", Value::Array(rows)),
    ])
}

/// Self-check of the `g1` sub-component with scripted first-form tentacles:
/// (a) a tentacle reporting no information gets a re-read request and the others do not;
/// (b) nothing passes between two `g0_tx` directly (gate 0 through the loop; a direct attempt
///     is refused and counted);
/// (c) the loop ends when a round changes nothing, before its derived ceiling;
/// (d) the right span, second in one tentacle's report and corroborated by the whole-code-base
///     layer, is forwarded by the loop and not by one-shot forwarding of first choices.
pub(crate) fn check() -> Result<(), String> {
    use crate::g0::scripted;
    let cb = WholeCodebase::build(
        ["a/cache.rs", "a/other.rs", "b/x.rs", "b/y.rs", "c/p.rs", "c/q.rs"].iter().map(|s| (*s).to_owned()).collect(),
        vec![item("derive_limit", "a/cache.rs", 5), item("parse_args", "c/p.rs", 5)],
    );
    let mut ts: Vec<Box<dyn TentacleSide>> = vec![
        Box::new(scripted(0, &["a/cache.rs", "a/other.rs"], &[("mass", &[0.9, 0.3]), ("evidence", &[0.8, 0.1])])),
        Box::new(scripted(1, &["b/x.rs", "b/y.rs"], &[("mass", &[-0.2, -0.5]), ("evidence", &[-0.1, -0.4]), ("peak", &[-0.3, -0.6])])),
        Box::new(scripted(2, &["c/p.rs", "c/q.rs"], &[("mass", &[0.7, 0.2]), ("evidence", &[0.6, 0.1])])),
    ];
    let mut g = G1::new(cb);
    let out = g.run(CASE_QUESTION, &mut ts, &Budget::derive(3, None, None, None));
    let first = out.log.first().ok_or("nmbrain.g1: no round")?;
    if first.rereads != ["g0_t2"] {
        return Err(format!("nmbrain.g1 (a): re-read requests {:?}, want only g0_t2", first.rereads));
    }
    if out.handover.gate_direct != 0 || g.coordination.misstamped != 0 {
        return Err(format!("nmbrain.g1 (b): gate {}", out.handover.gate_direct));
    }
    if out.handover.stop != "a round changed nothing" || out.handover.rounds >= out.ceiling || ts.get(1).map_or(0, |t| t.options()) == 0 {
        return Err(format!("nmbrain.g1 (c): stop {:?} after {} rounds, ceiling {}", out.handover.stop, out.handover.rounds, out.ceiling));
    }
    if out.log.last().is_some_and(|r| r.changed) {
        return Err("nmbrain.g1 (c): last round marked changed".into());
    }
    // (b) the gate is live: an envelope from one g0_tx straight into another is refused and counted.
    let direct = Order::new("g0_t1", "g0_t3", 9, 0.5, "direct attempt", Instruction::Reread(Focus::default()))?;
    if let Some(t) = ts.get_mut(2)
        && t.deliver(direct).is_ok()
    {
        return Err("nmbrain.g1 (b): a g0_tx accepted an envelope from another g0_tx".into());
    }
    if g.coordination.gate(&ts) != 1 {
        return Err(format!("nmbrain.g1 (b): gate after a direct attempt {}", g.coordination.gate(&ts)));
    }
    // (d) loop against one-shot on the corroborated-second case.
    let (_, cb, ts, right) = cases().into_iter().nth(1).ok_or("nmbrain.g1: no case")?;
    let mut ts = ts;
    let k = ts.len();
    let mut g = G1::new(cb.clone());
    let out = g.run(CASE_QUESTION, &mut ts, &Budget::derive(k, None, None, None));
    let (_, _, ts2, _) = cases().into_iter().nth(1).ok_or("nmbrain.g1: no case")?;
    let mut ts2 = ts2;
    let shot = G1::new(cb).one_shot(CASE_QUESTION, &mut ts2);
    let hit = out.handover.items.iter().find(|i| i.span.file == right);
    if hit.is_none() || shot.items.iter().any(|i| i.span.file == right) || out.handover.items.len() > k {
        return Err(format!(
            "nmbrain.g1 (d): loop {:?} one-shot {:?}",
            out.handover.items.iter().map(|i| i.span.label()).collect::<Vec<_>>(),
            shot.items.iter().map(|i| i.span.label()).collect::<Vec<_>>()
        ));
    }
    if !hit.is_some_and(|h| h.reasons.iter().any(|r| matches!(r, Reason::Corroborated(_))))
        || out.handover.messages().len() != out.handover.items.len()
    {
        return Err("nmbrain.g1 (d): forwarded span not marked corroborated, or messages missing".into());
    }
    queue_check()?;
    rules_check()
}

/// (f) The queue rule (refinement step 1): with a round capacity of one and a half spans, one
/// span goes up and every other held span of every tentacle stays in the evidence set (nothing
/// is dropped), each record carrying its facts and why it was queued; the first-form rule on the
/// same case forwards several and its evidence set still lists the rest.
fn queue_check() -> Result<(), String> {
    let (_, cb, ts, _) = cases().into_iter().nth(1).ok_or("nmbrain.g1: no case")?;
    let held: usize = ts.iter().map(|t| t.held().len()).sum();
    let k = ts.len();
    let mut ts = ts;
    let mut g = G1::new(cb);
    g.forward = ForwardRule::Queue;
    g.combine = Combine::Evidence(EvWeights([0.5, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]));
    let out = g.run(CASE_QUESTION, &mut ts, &Budget::derive(k, None, None, None).with_room(Some(150)));
    let ev = &out.evidence;
    let n = ev.forwarded.len() + ev.queue.len();
    if out.handover.items.len() != 1
        || ev.forwarded.len() != 1
        || n != held
        || ev.queue.iter().any(|e| e.reasons.iter().all(|r| !r.starts_with("queued")))
        || ev.forwarded.iter().any(|e| e.facts.tokens != 100)
    {
        return Err(format!("nmbrain.g1 (f): queue rule forwarded {} and held {n} of {held} spans", ev.forwarded.len()));
    }
    Ok(())
}

/// (e) The decision rules on hand values: the window confidence discriminates where the logistic
/// one does not; the largest-gap shortlist cuts where the scores break; the grid sums to 1.
fn rules_check() -> Result<(), String> {
    let scores = [16.0_f32, 12.0, 11.5, 11.0, 10.8];
    let lo = confidences(ConfidenceRule::Logistic, &scores, 16);
    let wi = confidences(ConfidenceRule::Window, &scores, 16);
    let spread = |v: &[f32]| v.first().copied().unwrap_or(0.0) - v.get(1).copied().unwrap_or(0.0);
    if !(spread(&wi) > spread(&lo) && wi.iter().sum::<f32>() < 1.0 && wi.windows(2).all(|w| w.first() >= w.get(1))) {
        return Err(format!("nmbrain.g1 (e): window confidences {wi:?} against logistic {lo:?}"));
    }
    let a = confidence_of(ConfidenceRule::Window, &scores, 1, 16.0, 16, None);
    if a <= wi.get(1).copied().unwrap_or(1.0) || a >= wi.first().copied().unwrap_or(0.0) {
        return Err(format!("nmbrain.g1 (e): answer confidence {a}"));
    }
    let (k, why) = shortlist_by(ShortlistRule::LargestGap, &scores, 6);
    let (k0, _) = shortlist_by(ShortlistRule::Entropy, &scores, 6);
    let (kf, _) = shortlist_by(ShortlistRule::LargestGap, &[-1.0, -2.0], 6);
    if k != 1 || k0 != 5 || kf != 2 {
        return Err(format!("nmbrain.g1 (e): shortlists gap {k} ({why}), entropy {k0}, fallback {kf}"));
    }
    // T3 normalisations: shift-invariant inside the window (z, robust, margin), the margin
    // positive for the window's first only, the content-free margin on the given noise floor.
    let shifted: Vec<f32> = scores.iter().map(|x| x + 100.0).collect();
    for r in [ConfidenceRule::Standard, ConfidenceRule::Robust, ConfidenceRule::Margin] {
        let (a, b) = (normalised(r, &scores, 16, None).unwrap_or_default(), normalised(r, &shifted, 16, None).unwrap_or_default());
        if a.len() != scores.len() || a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-3) {
            return Err(format!("nmbrain.g1 (e): {r:?} not shift-invariant: {a:?} {b:?}"));
        }
    }
    let m = normalised(ConfidenceRule::Margin, &scores, 16, None).unwrap_or_default();
    let nm = normalised(ConfidenceRule::NullMargin, &scores, 16, Some(2.0)).unwrap_or_default();
    if !(m.first().is_some_and(|x| *x > 0.0) && m.iter().skip(1).all(|x| *x < 0.0))
        || nm.first() != Some(&8.0)
        || normalised(ConfidenceRule::Logistic, &scores, 16, None).is_some()
    {
        return Err(format!("nmbrain.g1 (e): margin {m:?}, content-free margin {nm:?}"));
    }
    let g = weight_grid(4);
    if g.len() != 35 || g.iter().any(|w| (w.confidence + w.support + w.agreement + w.rank - 1.0).abs() > 1e-5) {
        return Err(format!("nmbrain.g1 (e): weight grid of {} points", g.len()));
    }
    Ok(())
}

#[cfg(test)]
mod lane_m_tests {
    use super::{Independence, Race, SpanRef};

    fn sp(k: &str) -> SpanRef {
        SpanRef { file: format!("{k}.rs"), start: 1, end: 2, key: k.to_owned() }
    }

    /// Accumulation sums one fused value per round; a runner not re-read fades; the old form is
    /// unchanged by default.
    #[test]
    fn accumulation_and_fading() {
        let mut r = Race::accumulating(1, 0.1, 0.3, 0.5);
        r.feed(&[(sp("a"), "t1".into(), 1.0, 1, vec![0]), (sp("a"), "t2".into(), 2.0, 1, vec![0]), (sp("b"), "t1".into(), 1.0, 1, vec![0])]);
        r.feed(&[(sp("a"), "t1".into(), 1.0, 1, vec![0])]);
        let t = |k: &str| r.runners.iter().find(|x| x.span.key == k).map(|x| x.total);
        assert_eq!(t("a"), Some(3.0));
        assert_eq!(t("b"), Some(0.5));
        let mut old = Race::new(1, 0.1, 0.3);
        old.feed(&[(sp("a"), "t1".into(), 1.0, 1, vec![0])]);
        old.feed(&[(sp("a"), "t1".into(), 3.0, 1, vec![0])]);
        assert_eq!(old.runners[0].total, 2.0);
    }

    /// Wording and naming together are nearly one kind: no quorum; wording, meaning and reading are.
    #[test]
    fn quorum_by_independence() {
        let ind = Independence::code(0.2);
        let k = |v: &[&str]| v.iter().map(|s| (*s).to_owned()).collect::<Vec<String>>();
        assert!(!ind.quorum(&k(&["wording", "naming"])));
        assert!(ind.quorum(&k(&["wording", "meaning", "reading"])));
        assert!(!ind.quorum(&k(&["structure", "wording"])));
    }
}
