//! G2 selection, first form (guide 3, G2 row; guide 4.1; plan step P7; research gap 2,
//! 2026-11-10). Designs: Luis Jake Gabriel III, Gavriel Technologies.
//!
//! G2 belongs to the context system: it decides what goes upstream to the Core Brain and
//! which MicroModel chain receives it. The chains themselves live in the Core Brain
//! (`crate::chain`). Method (research note gap 2, "When Simple kNN Beats Complex Learned
//! Routers", arXiv 2505.12601; cascade escalation as in FrugalGPT, arXiv 2305.05176; method
//! only, no code taken): embed the task description through the embedder slot, vote among
//! the k nearest labelled past cases weighted by similarity, calibrate the vote share on the
//! labelled set's own leave-one-out outcomes, and escalate when the calibrated confidence is
//! below a threshold derived from that set.
//!
//! Adaptive numbers (no fixed operating values): k is chosen by leave-one-out accuracy over
//! odd k up to 2 * floor(sqrt(n)) + 1 (capped at n - 1); the calibration window is
//! round(sqrt(n)) nearest leave-one-out outcomes, Laplace smoothed; the escalation threshold
//! is the median calibrated confidence of the leave-one-out errors (stated fallback when the
//! set produced no error: the share of the most common label, the accuracy of always guessing
//! it).
//!
//! Past cases come through `CaseSource`, so the memory system can supply them later; the
//! in-crate set (`BuiltinCases`) is written from task kinds only, from no locked test question.
use crate::chain::ChainBook;
use crate::slot::Slots;
use nmtype::{Fault, FaultKind};
use std::sync::Arc;
use std::time::Instant;

pub const ID: &str = "nmbrain.g2";

/// One labelled past case: the task description and the chain kind that served it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Case {
    pub text: String,
    pub label: String,
}
/// Where labelled past cases come from (memory later; the built-in set now).
pub trait CaseSource {
    fn cases(&self) -> Vec<Case>;
}

/// G2's decision for one task.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// Send upstream to this chain; `nodes` are its roles in order.
    Chosen { kind: String, nodes: Vec<String>, confidence: f32, raw: f32 },
    /// Confidence below the derived threshold: escalate (a longer chain, more reading, or the
    /// decision node), with the best guess attached.
    Escalate { best_guess: String, confidence: f32, threshold: f32 },
}
/// The selection with the parts to send upstream.
#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    pub outcome: Outcome,
    /// Parts handed up that share any term with the task, in the incoming order (order contract).
    pub upstream: Vec<String>,
    /// Parts held back (scored zero against the task).
    pub held_back: usize,
    /// The nearest cases: (label, similarity).
    pub neighbours: Vec<(String, f32)>,
    pub ms: f64,
}

/// The fitted selector: embedded cases, the derived k, calibration and threshold.
pub struct Selector {
    slots: Arc<Slots>,
    cases: Vec<(Case, Vec<f32>)>,
    pub k: usize,
    pub threshold: f32,
    /// Leave-one-out accuracy at the chosen k.
    pub loo_accuracy: f32,
    /// Share of the most common label.
    pub majority_rate: f32,
    /// Leave-one-out (raw vote share, correct), sorted by raw.
    calib: Vec<(f32, bool)>,
    window: usize,
    /// Calibrated confidence of each leave-one-out case (same order as `calib` is built).
    loo: Vec<(f32, bool)>,
    /// Measured track record of the part ordering (none for the exact reranker): with none,
    /// parts go up in the incoming order.
    pub track: Option<crate::chain::Track>,
    /// First-form behaviour: parts go up in the exact reranker's order (no order contract).
    pub first_form_order: bool,
}
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
impl Selector {
    /// Embed the cases through the embedder slot and derive k, calibration and threshold.
    /// # Errors
    /// Fewer than three cases or fewer than two labels.
    pub fn fit(slots: Arc<Slots>, source: &dyn CaseSource) -> Result<Self, Fault> {
        let cases: Vec<(Case, Vec<f32>)> = source
            .cases()
            .into_iter()
            .map(|c| {
                let v = slots.embed(&c.text);
                (c, v)
            })
            .collect();
        let n = cases.len();
        let mut labels: Vec<&str> = cases.iter().map(|(c, _)| c.label.as_str()).collect();
        labels.sort_unstable();
        labels.dedup();
        if n < 3 || labels.len() < 2 {
            return Err(Fault::new(ID, FaultKind::Config, format!("need at least 3 cases and 2 labels, have {n} and {}", labels.len())));
        }
        let majority = labels.iter().map(|l| cases.iter().filter(|(c, _)| c.label == *l).count()).max().unwrap_or(0);
        let mut s = Self {
            slots,
            cases,
            k: 1,
            threshold: 0.0,
            loo_accuracy: 0.0,
            majority_rate: majority as f32 / n as f32,
            calib: Vec::new(),
            window: 1,
            loo: Vec::new(),
            track: None,
            first_form_order: false,
        };
        let kmax = (2 * n.isqrt() + 1).min(n - 1);
        let mut best = (0usize, 1usize);
        for k in (1..=kmax).step_by(2) {
            let hits = (0..n).filter(|&i| s.loo_vote(i, k).0 == s.label_of(i)).count();
            if hits > best.0 {
                best = (hits, k);
            }
        }
        s.k = best.1;
        s.loo_accuracy = best.0 as f32 / n as f32;
        let raw: Vec<(f32, bool)> = (0..n)
            .map(|i| {
                let (l, r, _) = s.loo_vote(i, s.k);
                (r, l == s.label_of(i))
            })
            .collect();
        s.calib = raw.clone();
        s.calib.sort_by(|a, b| a.0.total_cmp(&b.0));
        s.window = (n as f32).sqrt().round().max(1.0) as usize;
        s.loo = raw.iter().map(|&(r, ok)| (s.calibrate(r), ok)).collect();
        let mut wrong: Vec<f32> = s.loo.iter().filter(|(_, ok)| !ok).map(|(c, _)| *c).collect();
        wrong.sort_by(f32::total_cmp);
        s.threshold = wrong.get(wrong.len() / 2).copied().unwrap_or(s.majority_rate);
        Ok(s)
    }
    fn label_of(&self, i: usize) -> &str {
        self.cases.get(i).map_or("", |(c, _)| c.label.as_str())
    }
    /// Similarity-weighted vote of the k nearest cases (one case excluded for leave-one-out):
    /// (label, raw vote share, neighbours).
    fn vote_from(&self, qv: &[f32], skip: Option<usize>, k: usize) -> (String, f32, Vec<(String, f32)>) {
        let mut near: Vec<(f32, usize)> =
            self.cases.iter().enumerate().filter(|(i, _)| Some(*i) != skip).map(|(i, (_, v))| (dot(qv, v), i)).collect();
        near.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        near.truncate(k);
        let total: f32 = near.iter().map(|(s, _)| s.max(0.0)).sum();
        let weight = |s: f32| if total > 0.0 { s.max(0.0) } else { 1.0 };
        let mut tally: Vec<(String, f32)> = Vec::new();
        for (sim, i) in &near {
            let l = self.label_of(*i).to_owned();
            match tally.iter_mut().find(|(x, _)| *x == l) {
                Some(t) => t.1 += weight(*sim),
                None => tally.push((l, weight(*sim))),
            }
        }
        tally.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let sum: f32 = tally.iter().map(|t| t.1).sum();
        let (label, w) = tally.first().cloned().unwrap_or_default();
        let neighbours = near.iter().map(|(s, i)| (self.label_of(*i).to_owned(), *s)).collect();
        (label, if sum > 0.0 { w / sum } else { 0.0 }, neighbours)
    }
    fn loo_vote(&self, i: usize, k: usize) -> (String, f32, Vec<(String, f32)>) {
        let qv = self.cases.get(i).map(|(_, v)| v.clone()).unwrap_or_default();
        self.vote_from(&qv, Some(i), k)
    }
    /// Calibrated confidence for a raw vote share: Laplace-smoothed leave-one-out accuracy of
    /// the `window` cases with the nearest raw share.
    #[must_use]
    pub fn calibrate(&self, raw: f32) -> f32 {
        let mut by: Vec<&(f32, bool)> = self.calib.iter().collect();
        by.sort_by(|a, b| (a.0 - raw).abs().total_cmp(&(b.0 - raw).abs()));
        let w: Vec<&&(f32, bool)> = by.iter().take(self.window).collect();
        let ok = w.iter().filter(|x| x.1).count();
        (ok as f32 + 1.0) / (w.len() as f32 + 2.0)
    }
    /// Choose the chain for a task and the parts to send upstream.
    #[must_use]
    pub fn select(&self, task: &str, parts: &[String], book: &ChainBook) -> Selection {
        let t = Instant::now();
        let qv = self.slots.embed(task);
        let (kind, raw, neighbours) = self.vote_from(&qv, None, self.k);
        let confidence = self.calibrate(raw);
        let outcome = match book.get(&kind) {
            Some(c) if confidence >= self.threshold => Outcome::Chosen { nodes: c.roles(), kind, confidence, raw },
            _ => Outcome::Escalate { best_guess: kind, confidence, threshold: self.threshold },
        };
        // Parts sharing no term with the task are held back. The order is the order contract of
        // the chain runner (`chain::hold_order`): the exact reranker's order is a proposal and,
        // with no track record behind it, the incoming (reducer's) order is kept.
        let scores = self.slots.rerank_all(task, parts);
        let mut ix: Vec<usize> = (0..parts.len()).filter(|&i| scores.get(i).is_some_and(|s| *s > 0.0)).collect();
        ix.sort_by(|&a, &b| scores.get(b).unwrap_or(&0.0).total_cmp(scores.get(a).unwrap_or(&0.0)));
        let proposed: Vec<String> = ix.iter().filter_map(|&i| parts.get(i).cloned()).collect();
        let upstream =
            if self.first_form_order { proposed } else { crate::chain::hold_order(parts, &proposed, None, self.track.as_ref()).0 };
        Selection { outcome, held_back: parts.len() - upstream.len(), upstream, neighbours, ms: t.elapsed().as_secs_f64() * 1000.0 }
    }
    /// G2 on the new path (2026-11-10, refinement step 1, items 2, 3 and 5): the chain is chosen
    /// as `select` chooses it (unchanged in this step), but the parts are worked on with the
    /// question's evidence set from G1 (`evidence::EvidenceSet`), not on the pack lines and the
    /// question text alone:
    /// - a part sharing no term with the task is not dropped: it leaves the upstream list and its
    ///   record goes to the queue marked low (a part with no record is added to the queue as one);
    /// - when the records of the parts going up carry less than one round's capacity in tokens,
    ///   G2 sends NEXT to G1's queue and the released spans go up as further parts (citation
    ///   lines `path:start lines a..b`), each released record marked with the round.
    ///
    /// `held_back` then counts parts moved to the queue (none is lost).
    pub fn select_evidence(&self, task: &str, parts: &[String], book: &ChainBook, set: &mut crate::evidence::EvidenceSet) -> Selection {
        let mut sel = self.select(task, parts, book);
        let t = Instant::now();
        let scores = self.slots.rerank_all(task, parts);
        let loc = |l: &str| -> Option<(String, usize)> {
            let head = l.split_once(' ').map_or(l, |(a, _)| a);
            let (p, n) = head.rsplit_once(':')?;
            Some((p.replace('\\', "/"), n.split('-').next()?.parse().ok()?))
        };
        // Parts the word-overlap rule would hold back: kept, marked low in the queue.
        let mut upstream: Vec<String> = Vec::new();
        for (i, part) in parts.iter().enumerate() {
            if scores.get(i).is_some_and(|s| *s > 0.0) {
                upstream.push(part.clone());
                continue;
            }
            let rec = loc(part).and_then(|(p, n)| set.at(&p, n).map(|e| e.span.key.clone()));
            match rec {
                Some(k) => {
                    set.mark_low(&|e| e.span.key == k, "held low by G2: no term shared with the task", Vec::new());
                }
                None => {
                    let (file, line) = loc(part).unwrap_or_else(|| (part.clone(), 0));
                    let e = crate::evidence::SpanEvidence {
                        span: crate::g0::SpanRef { file, start: line, end: line, key: part.clone() },
                        found_by: vec![ID.into()],
                        ..crate::evidence::SpanEvidence::default()
                    };
                    set.mark_low(&|_| false, "held low by G2: no term shared with the task", vec![e]);
                }
            }
        }
        // Keep the order contract over what goes up (the reducer's order, now from evidence).
        let order: Vec<String> = sel.upstream.iter().filter(|p| upstream.contains(p)).cloned().collect();
        let mut up: Vec<String> = order;
        for p in &upstream {
            if !up.contains(p) {
                up.push(p.clone());
            }
        }
        // NEXT: fill to one round's capacity from the queue when the parts going up carry less.
        let load: usize = up.iter().filter_map(|l| loc(l)).filter_map(|(p, n)| set.at(&p, n).map(|e| e.facts.tokens)).sum();
        if load < set.capacity.tokens && set.queue.iter().any(|e| e.priority != crate::evidence::Priority::Low) {
            let released = set.request(&crate::evidence::Request::Next(0), None);
            for e in released {
                let line = format!("{}:{} lines {}..{}", e.span.file, e.span.start, e.span.start, e.span.end);
                if !up.contains(&line) {
                    up.push(line);
                }
            }
        }
        sel.held_back = parts.len().saturating_sub(upstream.len());
        sel.upstream = up;
        sel.ms += t.elapsed().as_secs_f64() * 1000.0;
        sel
    }
    /// Share of leave-one-out cases that would escalate, and the accuracy of the rest.
    #[must_use]
    pub fn loo_escalation(&self) -> (f32, f32) {
        let n = self.loo.len().max(1) as f32;
        let kept: Vec<&(f32, bool)> = self.loo.iter().filter(|(c, _)| *c >= self.threshold).collect();
        let acc = if kept.is_empty() { 0.0 } else { kept.iter().filter(|x| x.1).count() as f32 / kept.len() as f32 };
        (1.0 - kept.len() as f32 / n, acc)
    }
}

/// G2's gate on one request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Gate {
    /// The covering set holds every aspect and its leader stands clear of its rival.
    Settled,
    /// Every aspect is held, but the leader does not stand clear of its rival.
    Tie,
    /// These aspects are held by no connected set: the pack is withheld or marked low, naming them.
    Absent(Vec<usize>),
}
impl Gate {
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Settled => "settled",
            Self::Tie => "tie",
            Self::Absent(_) => "absent",
        }
    }
}

/// One piece G2 may choose: its chunk, its tokens, and `r[a]`, the chance it settles aspect `a`
/// the request's drone read from the request's words (from the drones' calibrated evidence).
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    pub chunk: usize,
    pub tokens: usize,
    pub r: Vec<f32>,
}

/// What G2's selection chose.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Selected {
    /// Indices into the choices, in the order taken.
    pub set: Vec<usize>,
    /// The gain in F each brought.
    pub gains: Vec<f32>,
    /// F of the set, and of every aspect (the word aspects, then the relations) its coverage.
    pub value: f32,
    pub covered: Vec<f32>,
    /// The noise band the last gain was held against, and why the selection stopped.
    pub band: f32,
    pub stop: &'static str,
    /// Gains evaluated (the lazy greedy's work).
    pub evaluated: usize,
}

/// A heap entry of the lazy greedy: a bound on gain per token, the choice, and the step it was
/// computed at.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Bound(f32, usize, usize);
impl Eq for Bound {}
impl PartialOrd for Bound {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Bound {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0).then(other.1.cmp(&self.1))
    }
}

/// The median absolute deviation of `v` (0 for an empty list).
fn mad(v: &[f32]) -> f32 {
    let med = |x: &mut Vec<f32>| -> f32 {
        x.sort_by(f32::total_cmp);
        let n = x.len();
        if n == 0 {
            0.0
        } else if n % 2 == 1 {
            x.get(n / 2).copied().unwrap_or(0.0)
        } else {
            (x.get(n / 2 - 1).copied().unwrap_or(0.0) + x.get(n / 2).copied().unwrap_or(0.0)) / 2.0
        }
    };
    let m = med(&mut v.to_vec());
    med(&mut v.iter().map(|x| (x - m).abs()).collect())
}

/// G2's selection as one objective (guide section 12.1): the set S of pieces that maximises
/// F(S) = sum over aspects a of w_a (1 - product over p in S of (1 - r_pa)), every w_a equal,
/// within `room` tokens. The aspects are the request's word aspects (`r` of each choice) and its
/// `relations`: relation k is settled by a piece at the far end of a typed relation from a piece
/// of S (`settles(p, q)` lists the relations p settles with q in S, read on p's own record), with
/// the chance that q holds what is named (q's largest r): the reference graph only brings p into
/// the candidates. F is monotone with diminishing returns; the budgeted greedy takes the largest
/// gain per token, lazily (a bound is recomputed only when it reaches the top; the pieces joined
/// to a newly taken one, `linked(q)`, are recomputed at once, since a relation can raise their
/// gain), and keeps the better of its set and the best single piece. It stops when the room is
/// full or the best gain is inside the noise band: the median absolute deviation of the pieces'
/// single values over the request's own candidates. No count is fixed anywhere.
#[must_use]
pub fn select(choices: &[Choice], relations: usize, settles: &dyn Fn(usize, usize) -> Vec<usize>, linked: &dyn Fn(usize) -> Vec<usize>, room: usize) -> Selected {
    let nw = choices.iter().map(|c| c.r.len()).max().unwrap_or(0);
    let n = nw + relations;
    let rho: Vec<f32> = choices.iter().map(|c| c.r.iter().copied().fold(0.0_f32, f32::max)).collect();
    let single: Vec<f32> = choices.iter().map(|c| c.r.iter().sum::<f32>()).collect();
    let positive: Vec<f32> = single.iter().copied().filter(|v| *v > 0.0).collect();
    let band = mad(&positive);
    let mut miss = vec![1.0_f32; n];
    let mut taken: Vec<usize> = Vec::new();
    let mut in_set = vec![false; choices.len()];
    // The relation terms adding `x` to `set` brings: (relation aspect, r).
    let rel_terms = |x: usize, set: &[usize]| -> Vec<(usize, f32)> {
        let mut out = Vec::new();
        for &s in set {
            for k in settles(x, s) {
                out.push((nw + k, rho.get(s).copied().unwrap_or(0.0)));
            }
            for k in settles(s, x) {
                out.push((nw + k, rho.get(x).copied().unwrap_or(0.0)));
            }
        }
        out
    };
    let gain = |x: usize, set: &[usize], miss: &[f32]| -> f32 {
        let word: f32 = choices.get(x).map_or(0.0, |c| c.r.iter().enumerate().map(|(a, r)| miss.get(a).copied().unwrap_or(0.0) * r).sum());
        // Several terms on one relation multiply its miss, as F does.
        let mut m: Vec<f32> = miss.to_vec();
        for (k, r) in rel_terms(x, set) {
            if let Some(v) = m.get_mut(k) {
                *v *= 1.0 - r.clamp(0.0, 1.0);
            }
        }
        let rel: f32 = (nw..n).map(|k| miss.get(k).copied().unwrap_or(0.0) - m.get(k).copied().unwrap_or(0.0)).sum();
        word + rel
    };
    #[expect(clippy::cast_precision_loss, reason = "token counts")]
    let per = |g: f32, x: usize| g / choices.get(x).map_or(1, |c| c.tokens.max(1)) as f32;
    let mut heap: std::collections::BinaryHeap<Bound> = std::collections::BinaryHeap::new();
    let mut evaluated = 0;
    for x in 0..choices.len() {
        if choices.get(x).is_some_and(|c| c.tokens <= room) {
            heap.push(Bound(per(gain(x, &[], &miss), x), x, 0));
            evaluated += 1;
        }
    }
    let mut used = 0_usize;
    let mut gains: Vec<f32> = Vec::new();
    let mut stop = "nothing left";
    while let Some(Bound(_, x, step)) = heap.pop() {
        if in_set.get(x).copied().unwrap_or(true) {
            continue;
        }
        let t = choices.get(x).map_or(0, |c| c.tokens);
        if used + t > room {
            stop = "budget";
            continue;
        }
        if step != taken.len() {
            heap.push(Bound(per(gain(x, &taken, &miss), x), x, taken.len()));
            evaluated += 1;
            continue;
        }
        let g = gain(x, &taken, &miss);
        // Inside the noise band: with diminishing returns it stays there (a relation that could
        // raise it is recomputed when its anchor is taken).
        if g <= band {
            stop = "noise band";
            continue;
        }
        if let Some(c) = choices.get(x) {
            for (a, r) in c.r.iter().enumerate() {
                if let Some(m) = miss.get_mut(a) {
                    *m *= 1.0 - r.clamp(0.0, 1.0);
                }
            }
        }
        for (k, r) in rel_terms(x, &taken) {
            if let Some(m) = miss.get_mut(k) {
                *m *= 1.0 - r.clamp(0.0, 1.0);
            }
        }
        taken.push(x);
        gains.push(g);
        used += t;
        if let Some(f) = in_set.get_mut(x) {
            *f = true;
        }
        for y in linked(x) {
            if !in_set.get(y).copied().unwrap_or(true) && choices.get(y).is_some_and(|c| used + c.tokens <= room) {
                heap.push(Bound(per(gain(y, &taken, &miss), y), y, taken.len()));
                evaluated += 1;
            }
        }
    }
    let value: f32 = miss.iter().map(|m| 1.0 - m).sum();
    // The better of the greedy set and the best single piece that fits.
    let best = (0..choices.len()).filter(|x| choices.get(*x).is_some_and(|c| c.tokens <= room)).max_by(|a, b| {
        single.get(*a).copied().unwrap_or(0.0).total_cmp(&single.get(*b).copied().unwrap_or(0.0)).then(b.cmp(a))
    });
    if let Some(b) = best
        && single.get(b).copied().unwrap_or(0.0) > value
    {
        let covered: Vec<f32> = (0..n).map(|a| choices.get(b).and_then(|c| c.r.get(a)).copied().unwrap_or(0.0)).collect();
        return Selected { set: vec![b], gains: vec![single.get(b).copied().unwrap_or(0.0)], value: covered.iter().sum(), covered, band, stop: "best single piece", evaluated };
    }
    Selected { set: taken, gains, value, covered: miss.iter().map(|m| 1.0 - m).collect(), band, stop, evaluated }
}

/// The existing keyword router of `slot.rs` read as a chain choice (the baseline). Its table
/// has jobs, not chains: verification or execution jobs map to verify-answer; any other focus
/// job (dependency, functional, historical, constraint) and no focus job map to locate-code.
/// It has no job for explaining or editing, which bounds it on those kinds.
#[must_use]
pub fn keyword_choice(slots: &Slots, task: &str) -> String {
    keyword_choice_inner(slots, task)
}

/// Belief per item in G2 (Phase G2-u, behind `crate::g1::item_belief_on`): fill the room in tokens
/// from G1's item beliefs, best first; an item that does not fit is queued by reference, never
/// cut; each chosen item's neighbour lines go with it by reference (not counted in the room, and
/// never as their own places, T-034). Returns (chosen keys, their neighbours, queued keys).
#[must_use]
pub fn fill_items(beliefs: &[crate::g1::ItemBelief], room_tokens: usize) -> (Vec<String>, Vec<String>, Vec<String>) {
    let (mut used, mut chosen, mut near, mut queued) = (0_usize, Vec::new(), Vec::new(), Vec::new());
    for b in beliefs {
        if chosen.is_empty() || used + b.tokens <= room_tokens {
            used += b.tokens;
            chosen.push(b.key.clone());
            for n in &b.neighbours {
                if !near.contains(n) {
                    near.push(n.clone());
                }
            }
        } else {
            queued.push(b.key.clone());
        }
    }
    near.retain(|n| !chosen.contains(n));
    (chosen, near, queued)
}

/// The pack set with G2's relation-aware selection kept (Stage 5 WP-C1, gap 7 of the coordination
/// diagnosis). `value_set` is the fused value's fill (best first); `relation_set` is what
/// `select` chose for the question's parts and relations. The first `protect` pieces of the value
/// fill always stay (the leader and its nearest rivals, so first place is never displaced); every
/// piece of the relation set joins (it settles a part or a relation the question asks); the rest
/// of the value fill follows in its order while it fits `room` tokens. When protected and
/// relation pieces alone exceed the room, relation pieces are taken in their own order while they
/// fit (the protected pieces always go).
#[must_use]
pub fn merge_relation_set(value_set: &[usize], relation_set: &[usize], tokens: &dyn Fn(usize) -> usize, room: usize, protect: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::with_capacity(value_set.len() + relation_set.len());
    let mut used = 0_usize;
    for c in value_set.iter().take(protect) {
        if !out.contains(c) {
            used += tokens(*c);
            out.push(*c);
        }
    }
    for c in relation_set {
        if !out.contains(c) && used + tokens(*c) <= room {
            used += tokens(*c);
            out.push(*c);
        }
    }
    for c in value_set.iter().skip(protect) {
        if !out.contains(c) && used + tokens(*c) <= room {
            used += tokens(*c);
            out.push(*c);
        }
    }
    out
}

/// The pack set with G2's relation-aware selection joined under a value gate (Stage 5 WP-C1, the
/// fix of the full merge's dip). Diagnosis (closure and large code, 2026-11-22): on every lost
/// range the answer piece was in the value fill below its first pieces, outside G2's selection, and
/// G2's selection was mostly pieces the fused value ranks low; joining them all pushed answers out.
/// Here a piece of `relation_set` takes the place of the weakest unprotected piece of the set only
/// when its fused `value` is within `band` of that piece's value, or when it is a relation partner
/// of a leading piece (`partner`), which bypasses the gate. The first `protect` pieces of
/// `value_set` never leave. The set stays within `room` tokens; a joining piece that cannot fit even
/// after the weakest unprotected pieces leave is skipped. The result keeps the value order, each
/// joined piece placed by its value. Pieces for which `pinned` holds never leave either: the lines
/// right beside a leading piece in its file, which score low alone but are read with it (diagnosis
/// on large docs, doc22: dropping the answer's two neighbour pieces cost its first place).
#[must_use]
#[allow(clippy::too_many_arguments, reason = "the merge's inputs are each a distinct quantity of G2's set")]
pub fn merge_gated(
    value_set: &[usize],
    relation_set: &[usize],
    value: &dyn Fn(usize) -> f32,
    partner: &dyn Fn(usize) -> bool,
    pinned: &dyn Fn(usize) -> bool,
    tokens: &dyn Fn(usize) -> usize,
    room: usize,
    protect: usize,
    band: f32,
) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::with_capacity(value_set.len() + relation_set.len());
    let mut used = 0_usize;
    for (i, c) in value_set.iter().enumerate() {
        if !out.contains(c) && (i < protect || used + tokens(*c) <= room) {
            used += tokens(*c);
            out.push(*c);
        }
    }
    let protected: Vec<usize> = value_set.iter().take(protect).copied().collect();
    let v = |c: usize| {
        let x = value(c);
        if x.is_finite() { x } else { f32::MIN }
    };
    for g in relation_set {
        if out.contains(g) {
            continue;
        }
        // The weakest unprotected pieces, weakest first.
        let mut weak: Vec<usize> = out.iter().copied().filter(|c| !protected.contains(c) && !pinned(*c)).collect();
        weak.sort_by(|a, b| v(*a).total_cmp(&v(*b)));
        // Free room costs nothing: a piece that fits without any piece leaving joins.
        let fits = used + tokens(*g) <= room;
        let admit = fits || partner(*g) || weak.first().is_none_or(|w| v(*g) >= v(*w) - band);
        if !admit {
            continue;
        }
        // Make room: drop the weakest unprotected pieces (only those weaker than the joining one,
        // unless it is a partner) until it fits.
        let need = tokens(*g);
        let mut free = room.saturating_sub(used);
        let mut drop: Vec<usize> = Vec::new();
        for w in &weak {
            if free >= need {
                break;
            }
            if !partner(*g) && v(*w) > v(*g) + band {
                break;
            }
            free += tokens(*w);
            drop.push(*w);
        }
        if free < need {
            continue;
        }
        out.retain(|c| !drop.contains(c));
        used = used + need - drop.iter().map(|c| tokens(*c)).sum::<usize>();
        let at = out.iter().position(|c| !protected.contains(c) && v(*c) < v(*g)).unwrap_or(out.len()).max(protected.iter().filter(|p| out.contains(p)).count());
        out.insert(at.min(out.len()), *g);
    }
    out
}

/// Section check of belief per item: a long item no longer wins by length alone; the room is
/// counted in tokens (an item that does not fit is queued, the first always goes); neighbours
/// travel by reference.
/// # Errors
/// One fault line.
pub fn check_items() -> Result<(), String> {
    use crate::g1::{ItemIn, item_beliefs};
    let it = |k: &str, t: usize, v: f32| ItemIn { key: k.into(), tokens: t, value: v, neighbours: vec![format!("{k}-near")] };
    // Value grows with length across the request (the C65 cause); the right item is short and
    // above its length's trend, the long one only on the trend.
    let items = vec![it("short-right", 40, 1.2), it("long", 2000, 2.0), it("mid", 300, 1.1), it("tiny", 10, 0.2), it("big", 1200, 1.7)];
    let b = item_beliefs(&items);
    if b.first().map(|x| x.key.as_str()) != Some("short-right") {
        return Err(format!("nmbrain.g2.items: order {:?}", b.iter().map(|x| (&x.key, x.value)).collect::<Vec<_>>()));
    }
    let (chosen, near, queued) = fill_items(&b, 400);
    if chosen.first().map(String::as_str) != Some("short-right") || !queued.iter().any(|k| k == "long") || !near.contains(&"short-right-near".to_owned()) {
        return Err(format!("nmbrain.g2.items: fill {chosen:?} {near:?} {queued:?}"));
    }
    Ok(())
}

fn keyword_choice_inner(slots: &Slots, task: &str) -> String {
    let jobs = slots.route(task);
    if jobs.iter().any(|j| j == "verification" || j == "execution") { "verify-answer".into() } else { "locate-code".into() }
}

/// The in-crate labelled set: task descriptions written per task kind (from the kinds'
/// meaning, not from any locked test question), labelled with the chain that serves them.
pub struct BuiltinCases;
const BUILTIN: &[(&str, &[&str])] = &[
    (
        "locate-code",
        &[
            "Where is the configuration loader defined?",
            "Which file holds the struct for the packet header?",
            "Find the function that opens the database connection",
            "Locate the module that parses command line flags",
            "In which crate is the tokenizer implemented?",
            "Where are the default budget values declared?",
            "Which functions call the cache eviction routine?",
            "Show me where the error type for the loader lives",
            "Find every place the retry limit constant is used",
            "Where does the server register its request handlers?",
            "Which source file defines the trait for storage backends?",
            "Locate the test helper that builds a temporary folder",
            "What file contains the main entry point of the binary?",
            "Where is the hash function for file contents written?",
            "Find the definition of the scheduler queue type",
            "Which modules import the logging setup?",
            "Where is the path resolution helper located?",
            "List the callers of the parse header function",
        ],
    ),
    (
        "explain-code",
        &[
            "Explain how the scheduler decides which job runs next",
            "How does the cache decide what to evict?",
            "Why does the loader read the file twice?",
            "What does the retry wrapper do when the connection drops?",
            "Explain the purpose of the state machine in the parser",
            "How do the request handlers relate to the router table?",
            "Describe how errors propagate from the storage layer to the caller",
            "Why is the buffer flushed before the lock is released?",
            "How does the tokenizer split words with underscores?",
            "What is the role of the version counter in invalidation?",
            "Explain the flow of a request through the middleware",
            "How are budgets divided between the workers?",
            "Why does the merge step keep both conflicting values?",
            "Walk me through what happens at server start-up",
            "What does the ranking function weigh most heavily?",
            "How does the build step detect stale outputs?",
            "Explain what the reducer does with packets lacking evidence",
            "Describe the relationship between the mailbox and the subscribers",
        ],
    ),
    (
        "propose-edit",
        &[
            "Add a timeout parameter to the connection opener",
            "Fix the off-by-one error in the line counter",
            "Refactor the loader so it reads the file only once",
            "Rename the helper function to a clearer name and update callers",
            "Change the default retry limit from three to five",
            "Remove the unused import warnings in the parser module",
            "Implement a size cap for the in-memory cache",
            "Replace the manual loop with an iterator chain in the reducer",
            "Patch the handler so empty requests return an error",
            "Modify the scheduler to prefer short jobs",
            "Add logging when a packet is refused for missing evidence",
            "Make the path helper accept relative folders",
            "Split the large function in the server into smaller pieces",
            "Write a new command that prints the queue length",
            "Update the struct to derive Clone and Debug",
            "Introduce a configuration flag to disable the cache",
            "Edit the tokenizer to keep dashes inside identifiers",
            "Propose a change that makes the merge step deterministic",
        ],
    ),
    (
        "verify-answer",
        &[
            "Verify that the cache never holds more than the cap",
            "Check whether the loader handles a missing file without crashing",
            "Is it true that the scheduler is fair across workers?",
            "Confirm the build passes after the last change",
            "Do the tests cover the error path of the parser?",
            "Check that the reducer refuses packets without evidence",
            "Is the claim that the writer only copies evidence lines correct?",
            "Verify the retry wrapper stops after the limit",
            "Does clippy report any warnings in the storage crate?",
            "Confirm that the hash is stable across runs",
            "Check if this answer about the router table is supported by the code",
            "Is the statement that the mailbox drops old packets accurate?",
            "Validate that the server returns an error on empty requests",
            "Does the test suite pass on the current version?",
            "Check that the budget is never exceeded during a run",
            "Is it correct that the parser rejects trailing input?",
            "Verify the proposed fix compiles and keeps behaviour",
            "Confirm the configuration flag actually disables the cache",
        ],
    ),
];
impl CaseSource for BuiltinCases {
    fn cases(&self) -> Vec<Case> {
        BUILTIN
            .iter()
            .flat_map(|(label, texts)| texts.iter().map(move |t| Case { text: (*t).to_owned(), label: (*label).to_owned() }))
            .collect()
    }
}

/// First measurement: G2 leave-one-out accuracy beside the keyword router on the same set,
/// with the derived numbers and the time per selection.
#[derive(Clone, Debug, PartialEq)]
pub struct Measurement {
    pub n: usize,
    pub k: usize,
    pub g2_loo_accuracy: f32,
    pub keyword_accuracy: f32,
    /// The Core Brain's exact decision node (chain descriptions scored by the reranker
    /// control) on the same set; it uses no labels, so it is not leave-one-out.
    pub decision_node_accuracy: f32,
    pub majority_rate: f32,
    pub threshold: f32,
    pub escalate_rate: f32,
    pub accepted_accuracy: f32,
    pub fit_ms: f64,
    pub ms_per_selection: f64,
}
impl std::fmt::Display for Measurement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "G2 first measurement: n={} cases, k={} (leave-one-out), G2 kNN leave-one-out accuracy {:.3}, keyword router accuracy {:.3}, exact decision node accuracy {:.3}, majority-label rate {:.3}, escalation threshold {:.3}, leave-one-out escalation rate {:.3}, accuracy of non-escalated {:.3}, fit {:.2} ms, {:.4} ms per selection",
            self.n,
            self.k,
            self.g2_loo_accuracy,
            self.keyword_accuracy,
            self.decision_node_accuracy,
            self.majority_rate,
            self.threshold,
            self.escalate_rate,
            self.accepted_accuracy,
            self.fit_ms,
            self.ms_per_selection
        )
    }
}
/// Fit on a case source and measure.
/// # Errors
/// As `Selector::fit` or the chain file.
pub fn measure(source: &dyn CaseSource) -> Result<Measurement, Fault> {
    let slots = Arc::new(Slots::default());
    let book = ChainBook::from_mini().map_err(Fault::from)?;
    let t = Instant::now();
    let sel = Selector::fit(Arc::clone(&slots), source)?;
    let fit_ms = t.elapsed().as_secs_f64() * 1000.0;
    let cases = source.cases();
    let kw = cases.iter().filter(|c| keyword_choice(&slots, &c.text) == c.label).count();
    let nodes = crate::chain::NodeMap::exact(&slots, &book.roles);
    let dn = cases
        .iter()
        .filter(|c| {
            crate::chain::Former::form(&crate::chain::DecisionFormer, &c.text, &[], &book, &nodes).is_ok_and(|f| f.chain.kind == c.label)
        })
        .count();
    let t = Instant::now();
    for c in &cases {
        let _ = sel.select(&c.text, &[], &book);
    }
    let per = t.elapsed().as_secs_f64() * 1000.0 / cases.len().max(1) as f64;
    let (escalate_rate, accepted_accuracy) = sel.loo_escalation();
    Ok(Measurement {
        n: cases.len(),
        k: sel.k,
        g2_loo_accuracy: sel.loo_accuracy,
        keyword_accuracy: kw as f32 / cases.len().max(1) as f32,
        decision_node_accuracy: dn as f32 / cases.len().max(1) as f32,
        majority_rate: sel.majority_rate,
        threshold: sel.threshold,
        escalate_rate,
        accepted_accuracy,
        fit_ms,
        ms_per_selection: per,
    })
}

/// Self-check of the `g2` sub-component: the selector fits on the built-in set with derived
/// k and threshold in range; clear tasks select a listed chain with its roles and a
/// confidence in [0, 1]; parts that share no term with the task are held back; the escalate
/// outcome is reachable; the measurement runs.
pub(crate) fn check() -> Result<(), String> {
    let e = |m: String| format!("{ID}: {m}");
    let slots = Arc::new(Slots::default());
    let book = ChainBook::from_mini().map_err(|f| e(f.to_string()))?;
    let mut sel = Selector::fit(Arc::clone(&slots), &BuiltinCases).map_err(|f| e(f.to_string()))?;
    if sel.k == 0 || !(0.0..=1.0).contains(&sel.threshold) || !(0.0..=1.0).contains(&sel.loo_accuracy) {
        return Err(e(format!("derived values out of range: k {} threshold {} accuracy {}", sel.k, sel.threshold, sel.loo_accuracy)));
    }
    let parts = vec!["src/cache.rs:10 fn evict(&mut self)".to_owned(), "zzz qqq".to_owned()];
    let s = sel.select("How does the cache decide what to evict?", &parts, &book);
    let (kind, conf) = match &s.outcome {
        Outcome::Chosen { kind, nodes, confidence, .. } => {
            if book.get(kind).map(crate::chain::ChainDef::roles).as_ref() != Some(nodes) {
                return Err(e("chosen nodes are not the chain's roles".into()));
            }
            (kind.clone(), *confidence)
        }
        Outcome::Escalate { best_guess, confidence, .. } => (best_guess.clone(), *confidence),
    };
    if book.get(&kind).is_none() || !(0.0..=1.0).contains(&conf) {
        return Err(e(format!("selection {kind} confidence {conf}")));
    }
    if s.upstream.len() != 1 || s.held_back != 1 {
        return Err(e(format!("upstream {:?} held back {}", s.upstream, s.held_back)));
    }
    let derived = sel.threshold;
    sel.threshold = 1.01;
    if !matches!(sel.select("Where is the loader?", &[], &book).outcome, Outcome::Escalate { .. }) {
        return Err(e("escalate outcome not reachable".into()));
    }
    sel.threshold = derived;
    // (h) The new path: the part sharing no term is not lost (its record goes to the queue,
    // marked low), and with less than a round's capacity going up G2 sends NEXT, whose released
    // span goes up as a further part.
    {
        use crate::evidence::{Capacity, EvidenceSet, Priority, SpanEvidence, SpanFacts};
        use crate::g0::SpanRef;
        let rec = |file: &str, a: usize, key: &str, combined: f32, priority: Priority| SpanEvidence {
            span: SpanRef { file: file.into(), start: a, end: a + 9, key: key.into() },
            combined,
            priority,
            facts: SpanFacts { tokens: 100, ..SpanFacts::default() },
            ..SpanEvidence::default()
        };
        let mut set = EvidenceSet {
            capacity: Capacity::derive(Some(250)),
            rounds: 1,
            forwarded: vec![rec("src/cache.rs", 10, "k1", 0.9, Priority::Forwarded), rec("src/zz.rs", 1, "k2", 0.4, Priority::Forwarded)],
            queue: vec![rec("src/next.rs", 1, "k3", 0.5, Priority::Queued)],
            ..EvidenceSet::default()
        };
        let parts = vec!["src/cache.rs:10 fn evict(&mut self)".to_owned(), "src/zz.rs:1 zzz qqq".to_owned()];
        let before = set.all().count();
        let s2 = sel.select_evidence("How does the cache decide what to evict?", &parts, &book, &mut set);
        let low = set.queue.iter().any(|x| x.span.key == "k2" && x.priority == Priority::Low);
        let next = s2.upstream.iter().any(|l| l.starts_with("src/next.rs:1"));
        if s2.held_back != 1 || !low || !next || set.all().count() != before || set.log.len() != 1 {
            return Err(e(format!(
                "evidence selection: upstream {:?} held back {}, low {low}, next {next}, log {:?}",
                s2.upstream, s2.held_back, set.log
            )));
        }
    }
    // Selection: by gain per token A then B (each settles one aspect); C repeats A and does not
    // fit the room; D holds both but is too large; with a relation asked, E (at the far end of a
    // typed relation from A) settles it on A's chance and joins.
    {
        let ch = |chunk: usize, tokens: usize, r: &[f32]| Choice { chunk, tokens, r: r.to_vec() };
        let cs = vec![ch(1, 100, &[0.9, 0.0]), ch(2, 100, &[0.0, 0.8]), ch(3, 100, &[0.9, 0.0]), ch(4, 1000, &[0.95, 0.85]), ch(5, 50, &[0.0, 0.0])];
        let none = |_: usize, _: usize| Vec::new();
        let no_links = |_: usize| Vec::new();
        let s = select(&cs, 0, &none, &no_links, 250);
        if s.set != vec![0, 1] || (s.value - 1.7).abs() > 1e-4 {
            return Err(e(format!("select {s:?}")));
        }
        let settles = |p: usize, q: usize| if (p, q) == (4, 0) { vec![0] } else { Vec::new() };
        let links = |q: usize| if q == 0 { vec![4] } else { Vec::new() };
        let s = select(&cs, 1, &settles, &links, 400);
        if !s.set.contains(&4) || s.covered.get(2).is_none_or(|c| (c - 0.9).abs() > 1e-4) {
            return Err(e(format!("select with a relation {s:?}")));
        }
    }
    measure(&BuiltinCases).map_err(|f| e(f.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    /// Prints the first measurement (`cargo test --release -p nmbrain g2_measure -- --nocapture`).
    #[test]
    fn g2_measure() {
        let m = super::measure(&super::BuiltinCases).expect("measure");
        println!("{m}");
    }
    /// WP-C1: a multi-range answer whose second piece is fourth by value enters beside the leader,
    /// the leader stays first, and the room is respected.
    #[test]
    fn merge_keeps_leader_and_relation_partner() {
        let tk = |_c: usize| 1000_usize;
        // Value fill: A=10 first, then 11, 12, 13; the relation set holds A and its partner B=40
        // (fourth or later by value, not in the value fill's room).
        let merged = super::merge_relation_set(&[10, 11, 12, 13], &[10, 40], &tk, 4000, 1);
        assert_eq!(merged.first(), Some(&10), "the leader stays first");
        assert!(merged.contains(&40), "the relation partner enters the pack");
        assert_eq!(merged.len(), 4, "four pieces of 1,000 tokens fill a 4,000-token room");
        assert!(!merged.contains(&13), "the weakest value piece leaves to make room");
    }
    /// WP-C1: the room clamps the set; protected pieces always go.
    #[test]
    fn merge_respects_room_and_protection() {
        let tk = |c: usize| if c == 99 { 5000 } else { 1000 };
        let merged = super::merge_relation_set(&[1, 2, 3], &[99, 4], &tk, 2500, 2);
        assert_eq!(merged, vec![1, 2], "a relation piece larger than the room left is not taken");
        let merged = super::merge_relation_set(&[1, 2, 3], &[4], &tk, 16_000, 1);
        assert_eq!(merged, vec![1, 4, 2, 3]);
        let merged = super::merge_relation_set(&[1, 2], &[], &tk, 1000, 1);
        assert_eq!(merged, vec![1], "an empty relation set gives the value fill to the room");
        assert!(super::merge_relation_set(&[], &[], &tk, 1000, 1).is_empty());
    }
    /// WP-C1 gated merge: a weak G2 piece does not displace a stronger answer piece; a strong one
    /// or a relation partner does; protected pieces stay; the room holds.
    #[test]
    fn gated_merge_keeps_strong_value_pieces() {
        let tk = |_c: usize| 1000_usize;
        // Values: value pieces 1..4 strong to weak; G2 offers 50 (weak) and 60 (strong).
        let val = |c: usize| match c {
            1 => 9.0,
            2 => 8.0,
            3 => 7.0,
            4 => 6.0,
            50 => 1.0,
            60 => 6.5,
            70 => 0.5,
            _ => 0.0,
        };
        let none = |_c: usize| false;
        // The weak piece 50 is refused: answer piece 4 stays.
        let m = super::merge_gated(&[1, 2, 3, 4], &[50], &val, &none, &none, &tk, 4000, 3, 0.25);
        assert_eq!(m, vec![1, 2, 3, 4]);
        // The strong piece 60 displaces the weakest unprotected piece (4).
        let m = super::merge_gated(&[1, 2, 3, 4], &[60], &val, &none, &none, &tk, 4000, 3, 0.25);
        assert_eq!(m, vec![1, 2, 3, 60]);
        // A relation partner of a leader joins even when weak.
        let partner = |c: usize| c == 70;
        let m = super::merge_gated(&[1, 2, 3, 4], &[70], &val, &partner, &none, &tk, 4000, 3, 0.25);
        assert!(m.contains(&70) && m.contains(&1) && m.contains(&2) && m.contains(&3), "{m:?}");
        assert_eq!(m.len(), 4);
        // A pinned neighbour of a leader is never the piece that leaves: the next weakest goes.
        let pin4 = |c: usize| c == 4;
        let m = super::merge_gated(&[1, 2, 3, 5, 4], &[60], &|c| if c == 5 { 6.2 } else { val(c) }, &none, &pin4, &tk, 5000, 3, 0.25);
        assert!(m.contains(&4) && m.contains(&60) && !m.contains(&5), "{m:?}");
        // Free room: a G2 piece joins without displacing anything, placed by its value.
        let m = super::merge_gated(&[1, 2], &[60], &val, &none, &none, &tk, 4000, 1, 0.25);
        assert_eq!(m, vec![1, 2, 60]);
    }
    /// The two new section checks (`neuromorph-nano check nmbrain` runs them with the rest).
    #[test]
    fn chain_and_g2_checks() {
        for (id, f) in crate::checks().into_iter().filter(|(id, _)| *id == "nmbrain.chain" || *id == "nmbrain.g2") {
            let t = std::time::Instant::now();
            assert_eq!(f(), Ok(()), "{id}");
            println!("{id} ok in {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
        }
    }
}
