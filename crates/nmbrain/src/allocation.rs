//! Allocation map and allocation rule (2026-11-10, refinement step 4a, fault F28; Gavriel
//! Technologies, architecture by Luis Jake Gabriel III; analysis
//! `docs/2026-11-10-neumor-analysis-h-natural-systems-for-the-globalized-context-stack.md` sections F and G).
//!
//! Directions served (2026-11-10 09:17 and 09:19 UTC): the tentacles each cover different data;
//! what each one looks at is decided and mapped in coordination, with layers of its own; on a
//! very large code base they take in specific, spread-out, relation-connected information and do
//! not cover it blindly chunk after chunk.
//!
//! Two parts, both in RAM only, both holding references, flags and counts and nothing encoded:
//! - `AllocationMap`, held by G1: who holds what (per tentacle its chunk set, its lens or
//!   relation path, its tokens), what has been read this session, what is uncovered, and what
//!   was passed over (kept as references; nothing is discarded). A chunk has one holder: a claim
//!   on a chunk another tentacle holds is refused, so no chunk is in two windows.
//! - `assign`, used by G2: from the map, the candidate targets (relation seeds and coverage
//!   regions) and the previous round's yields it decides each tentacle's next intake, coverage
//!   mode or relation mode, with which region or seed.
//!
//! The rules, each a pure function:
//! - Move on (marginal value rule): a tentacle moves on when its last read yielded less than
//!   the question's running average, the mean of every yield recorded for the question so far
//!   (`moves_on`). With no yield recorded (round 1) nothing has been found, so every tentacle
//!   takes a coverage target (analysis G.4).
//! - Stay and elaborate (area-restricted search): a tentacle that does not move on follows the
//!   relations of its own finds in relation mode.
//! - Different data (coordinated assignment with a utility discount, Burgard et al. 2005): the
//!   (tentacle, target) pair of highest utility is assigned first; every other target then loses
//!   the share of its utility that the chosen tentacle's walk is expected to cover, 1 - d /
//!   (reach + 1) for a target d hops away over the specific relations and nothing beyond `reach`
//!   hops. `reach` is the depth relation walks have reached this session (`AllocationMap::
//!   reach`); the stated fallback before any walk is `REACH_FALLBACK`.
//! - More room for lenses that yielded (`lens_shares`): a lens's share of the coverage targets
//!   is its yield above zero plus the mean such yield, over the sum, so a lens that found
//!   something is offered more regions and a lens that found nothing keeps a floor. With no
//!   yields the shares are equal (the stated fallback).
//! - Mixture of the modes (`coverage_weight`): coverage targets are weighed by the share of the
//!   still uncovered tree that one round of windows can take, at most 1. When a round can take
//!   in all that is left, coverage counts in full; on a very large code base it counts little
//!   and relation mode leads after round 1 (analysis G.4), while the discount still sends a
//!   tentacle to a coverage region when the open seeds lie close to seeds already assigned.
//!
//! No tentacle sees another: the discount and the claims are computed here from the map, by G1
//! and G2. Sensory material (chunk references) and behavioural material (orders) stay separate:
//! an `Order` names a mode and a target and carries no content.
//!
//! This module is a data structure and pure functions with checks. It is not wired into the
//! live pass; the calls the pass and G1 make are set out in
//! `runs/2026-11-10-neumor-note-step-4a-relations-allocation.md`.
use std::collections::{BTreeMap, BTreeSet};

/// Walk depth assumed before any relation walk has run this session: two hops, a relation and
/// its neighbour (the depth at which the analysis measured the walk; stated fallback).
pub const REACH_FALLBACK: usize = 2;

/// How a chunk was taken in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Mode {
    /// A lens choosing from the whole tree.
    #[default]
    Coverage,
    /// A walk over specific relations from seeds.
    Relation,
    /// A keys-only scan (the periphery).
    Scan,
}
impl Mode {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Coverage => "coverage",
            Self::Relation => "relation",
            Self::Scan => "scan",
        }
    }
}

/// Where a chunk stands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Status {
    #[default]
    Unread,
    /// In a tentacle's window this round.
    Held,
    /// Read in an earlier round of this session.
    Read,
    Reported,
    Forwarded,
    Queued,
    /// Reached by a walk or a lens and not taken: kept as a reference.
    PassedOver,
}

/// The map's record of one chunk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cell {
    /// The tentacle holding it this round.
    pub holder: Option<u8>,
    pub mode: Mode,
    pub status: Status,
    /// The round it was last taken in (0: never).
    pub round: u8,
    /// Read this session and unchanged since (cleared when the content changes).
    pub visited: bool,
    /// The seed or region anchor it was taken from (`u32::MAX`: none).
    pub source: u32,
}

/// What one tentacle holds this round.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Holding {
    pub chunks: BTreeSet<usize>,
    pub tokens: usize,
    pub mode: Mode,
    /// Lens name, or `relation`.
    pub lens: String,
    /// Seeds of its relation walk and the relations followed (from, to).
    pub seeds: Vec<usize>,
    pub path: Vec<(usize, usize)>,
    /// Yield of each read this question, oldest first.
    pub yields: Vec<f32>,
}

/// The allocation map (module notes). Held by G1, in RAM only.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AllocationMap {
    cells: Vec<Cell>,
    holdings: Vec<Holding>,
    /// The round being formed (1 first).
    pub round: u8,
    /// Passed-over chunks with their best priority (references in G1's queue).
    passed: BTreeMap<usize, f32>,
    /// Claims refused because another tentacle held the chunk, and because it was read before.
    pub refused_held: usize,
    pub refused_visited: usize,
    /// Deepest hop of each relation walk this session.
    depths: Vec<usize>,
    /// Meaning in the map (`set_meaning`; empty: off): each chunk's meaning cluster and its
    /// near-duplicate group (the group's first chunk; `u32::MAX`: none), the tentacle holding a
    /// member of each group this round, and the groups read this session.
    meaning: Vec<u32>,
    twin: Vec<u32>,
    twin_holder: BTreeMap<u32, u8>,
    twin_read: BTreeSet<u32>,
    /// Claims refused because a near-duplicate by meaning was held by another tentacle or read.
    pub refused_twin: usize,
}

impl AllocationMap {
    /// A map over `chunks` chunks for `tentacles` tentacles, nothing held, round 1.
    #[must_use]
    pub fn new(chunks: usize, tentacles: usize) -> Self {
        Self {
            cells: vec![Cell { source: u32::MAX, ..Cell::default() }; chunks],
            holdings: vec![Holding::default(); tentacles],
            round: 1,
            ..Self::default()
        }
    }
    #[must_use]
    pub fn tentacles(&self) -> usize {
        self.holdings.len()
    }
    #[must_use]
    pub fn cell(&self, chunk: usize) -> Option<&Cell> {
        self.cells.get(chunk)
    }
    #[must_use]
    pub fn holding(&self, tentacle: usize) -> Option<&Holding> {
        self.holdings.get(tentacle)
    }
    /// The tentacle holding a chunk this round.
    #[must_use]
    pub fn holder(&self, chunk: usize) -> Option<usize> {
        self.cells.get(chunk).and_then(|c| c.holder).map(usize::from)
    }
    /// Whether a chunk can be claimed by `tentacle`: not held by another, and not read before
    /// (inhibition of return) unless `requested` (a request from above names it).
    #[must_use]
    pub fn free_for(&self, tentacle: usize, chunk: usize, requested: bool) -> bool {
        self.cells.get(chunk).is_some_and(|c| c.holder.is_none_or(|h| usize::from(h) == tentacle) && (requested || !c.visited))
            && (requested || !self.twin_covered(tentacle, chunk))
    }
    /// Give the map the material's meaning: each chunk's cluster and near-duplicate group (both
    /// `u32::MAX` for none; a group is named by its first chunk). With it, a chunk whose
    /// near-duplicate another tentacle holds, or whose group was read this session, counts as
    /// covered and is not claimed unrequested (the claim rule by meaning).
    pub fn set_meaning(&mut self, clusters: Vec<u32>, twins: Vec<u32>) {
        self.meaning = clusters;
        self.twin = twins;
        self.twin_holder.clear();
        self.twin_read.clear();
    }
    /// A chunk's meaning cluster (`None` without meaning or a vector).
    #[must_use]
    pub fn cluster(&self, chunk: usize) -> Option<u32> {
        self.meaning.get(chunk).copied().filter(|c| *c != u32::MAX)
    }
    /// The meaning clusters of the chunks held now or read this session.
    #[must_use]
    pub fn clusters_seen(&self) -> BTreeSet<u32> {
        self.cells.iter().enumerate().filter(|(_, c)| c.holder.is_some() || c.visited).filter_map(|(i, _)| self.cluster(i)).collect()
    }
    fn twin_of(&self, chunk: usize) -> Option<u32> {
        self.twin.get(chunk).copied().filter(|t| *t != u32::MAX)
    }
    /// Whether a near-duplicate by meaning of `chunk` covers it for `tentacle`: another tentacle
    /// holds a member of its group, or the group was read this session (the chunk itself unread).
    fn twin_covered(&self, tentacle: usize, chunk: usize) -> bool {
        let Some(g) = self.twin_of(chunk) else { return false };
        let other = self.twin_holder.get(&g).is_some_and(|h| usize::from(*h) != tentacle);
        let read = self.twin_read.contains(&g) && self.cells.get(chunk).is_some_and(|c| !c.visited);
        other || read
    }
    /// Claim a chunk for a tentacle (compare and set on the holder): refused when another
    /// tentacle holds it, or when it was read this session and is unchanged and no request
    /// names it. `source` is the seed or region anchor it comes from.
    #[expect(clippy::too_many_arguments, reason = "one claim: who, what, how, from where, its size, and whether requested")]
    pub fn claim(&mut self, tentacle: usize, chunk: usize, mode: Mode, source: Option<usize>, tokens: usize, requested: bool) -> bool {
        let Ok(t) = u8::try_from(tentacle) else { return false };
        let round = self.round;
        let Some(cell) = self.cells.get_mut(chunk) else { return false };
        if tentacle >= self.holdings.len() {
            return false;
        }
        match cell.holder {
            Some(h) if h == t => return true,
            Some(_) => {
                self.refused_held += 1;
                return false;
            }
            None => {}
        }
        if cell.visited && !requested {
            self.refused_visited += 1;
            return false;
        }
        if !requested && self.twin_covered(tentacle, chunk) {
            self.refused_twin += 1;
            return false;
        }
        if let Some(g) = self.twin_of(chunk) {
            self.twin_holder.entry(g).or_insert(t);
        }
        let Some(cell) = self.cells.get_mut(chunk) else { return false };
        *cell = Cell {
            holder: Some(t),
            mode,
            status: Status::Held,
            round,
            visited: cell.visited,
            source: source.and_then(|s| u32::try_from(s).ok()).unwrap_or(u32::MAX),
        };
        self.passed.remove(&chunk);
        if let Some(h) = self.holdings.get_mut(tentacle) {
            h.chunks.insert(chunk);
            h.tokens += tokens;
            h.mode = mode;
        }
        true
    }
    /// Name what a tentacle's window is: its lens (or `relation`) and the seeds of its walk.
    pub fn label(&mut self, tentacle: usize, lens: &str, seeds: &[usize]) {
        if let Some(h) = self.holdings.get_mut(tentacle) {
            lens.clone_into(&mut h.lens);
            h.seeds = seeds.to_vec();
        }
    }
    /// Record one relation followed by a tentacle's walk, and the walk's depth.
    pub fn follow(&mut self, tentacle: usize, from: usize, to: usize) {
        if let Some(h) = self.holdings.get_mut(tentacle)
            && !h.path.contains(&(from, to))
        {
            h.path.push((from, to));
        }
    }
    pub fn walked(&mut self, depth: usize) {
        if depth > 0 {
            self.depths.push(depth);
        }
    }
    /// Walk depth for the utility discount: the mean deepest hop of this session's relation
    /// walks, rounded up; `REACH_FALLBACK` before any walk.
    #[must_use]
    pub fn reach(&self) -> usize {
        if self.depths.is_empty() { REACH_FALLBACK } else { self.depths.iter().sum::<usize>().div_ceil(self.depths.len()).max(1) }
    }
    /// A chunk reached and not taken: kept as a reference with its best priority.
    pub fn pass_over(&mut self, chunk: usize, priority: f32) {
        let Some(cell) = self.cells.get_mut(chunk) else { return };
        if cell.holder.is_some() {
            return;
        }
        if cell.status == Status::Unread {
            cell.status = Status::PassedOver;
        }
        let p = self.passed.entry(chunk).or_insert(priority);
        *p = p.max(priority);
    }
    /// Passed-over references, best first.
    #[must_use]
    pub fn passed(&self) -> Vec<(usize, f32)> {
        let mut v: Vec<(usize, f32)> = self.passed.iter().map(|(c, p)| (*c, *p)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        v
    }
    /// Set a held or read chunk's status (reported, forwarded, queued).
    pub fn mark(&mut self, chunk: usize, status: Status) {
        if let Some(c) = self.cells.get_mut(chunk) {
            c.status = status;
        }
    }
    /// The yield of a tentacle's read this round.
    pub fn record_yield(&mut self, tentacle: usize, value: f32) {
        if let Some(h) = self.holdings.get_mut(tentacle)
            && value.is_finite()
        {
            h.yields.push(value);
        }
    }
    /// The question's running average yield: the mean of every yield recorded so far.
    #[must_use]
    pub fn running_average(&self) -> Option<f32> {
        let all: Vec<f32> = self.holdings.iter().flat_map(|h| h.yields.iter().copied()).collect();
        #[expect(clippy::cast_precision_loss, reason = "a small count")]
        let mean = all.iter().sum::<f32>() / all.len().max(1) as f32;
        (!all.is_empty()).then_some(mean)
    }
    /// A tentacle's last yield.
    #[must_use]
    pub fn last_yield(&self, tentacle: usize) -> Option<f32> {
        self.holdings.get(tentacle).and_then(|h| h.yields.last().copied())
    }
    /// Close the round: every held chunk becomes read (visited) and loses its holder, each
    /// tentacle's chunk set is cleared for its next window (its yields, seeds and path stay),
    /// and the next round begins. Nothing is removed from the map: read chunks stay referenced.
    pub fn next_round(&mut self) {
        let held: Vec<u32> = self.twin_holder.keys().copied().collect();
        self.twin_read.extend(held);
        self.twin_holder.clear();
        for c in &mut self.cells {
            if c.holder.take().is_some() {
                c.visited = true;
                if c.status == Status::Held {
                    c.status = Status::Read;
                }
            }
        }
        for h in &mut self.holdings {
            h.chunks.clear();
            h.tokens = 0;
        }
        self.round = self.round.saturating_add(1);
    }
    /// A new question: holders, yields, paths and passed-over references are cleared; what was
    /// read this session stays marked visited, so `read()` still answers.
    pub fn next_question(&mut self) {
        self.next_round();
        for h in &mut self.holdings {
            *h = Holding::default();
        }
        self.passed.clear();
        self.round = 1;
    }
    /// The content of a chunk changed (from sync): it is unread again and held by no one.
    pub fn content_changed(&mut self, chunk: usize) {
        if let Some(c) = self.cells.get_mut(chunk) {
            if let Some(h) = c.holder.and_then(|h| self.holdings.get_mut(usize::from(h))) {
                h.chunks.remove(&chunk);
            }
            *c = Cell { source: u32::MAX, ..Cell::default() };
        }
    }
    /// Chunks read this session and unchanged.
    #[must_use]
    pub fn read(&self) -> Vec<usize> {
        (0..self.cells.len()).filter(|i| self.cells.get(*i).is_some_and(|c| c.visited)).collect()
    }
    /// Chunks neither held now nor read this session.
    #[must_use]
    pub fn uncovered(&self) -> Vec<usize> {
        (0..self.cells.len()).filter(|i| self.cells.get(*i).is_some_and(|c| c.holder.is_none() && !c.visited)).collect()
    }
    /// Chunks in more than one tentacle's set (zero by construction; counted from the holdings,
    /// not from the holder cells, as an independent check).
    #[must_use]
    pub fn overlap(&self) -> usize {
        let mut seen = BTreeSet::new();
        self.holdings.iter().flat_map(|h| h.chunks.iter()).filter(|c| !seen.insert(**c)).count()
    }
    /// Share of the tree's tokens held now or read this session.
    #[must_use]
    pub fn covered_share(&self, tokens: &dyn Fn(usize) -> usize) -> f64 {
        let (mut all, mut got) = (0_usize, 0_usize);
        for (i, c) in self.cells.iter().enumerate() {
            let t = tokens(i);
            all += t;
            if c.holder.is_some() || c.visited {
                got += t;
            }
        }
        #[expect(clippy::cast_precision_loss, reason = "token counts are far below 2^52")]
        let share = got as f64 / all.max(1) as f64;
        share
    }
    /// Bytes held in RAM (cells, holdings, references).
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.cells.len() * std::mem::size_of::<Cell>()
            + self.holdings.iter().map(|h| h.chunks.len() * 8 + h.path.len() * 16 + h.seeds.len() * 8 + h.yields.len() * 4).sum::<usize>()
            + self.passed.len() * 12
    }
    /// One line: the state of the map.
    #[must_use]
    pub fn describe(&self) -> String {
        let held: usize = self.holdings.iter().map(|h| h.chunks.len()).sum();
        format!(
            "allocation map: round {}, {} chunks, {} held by {} tentacles, {} read this session, {} uncovered, {} passed over (kept as references), overlap {}, claims refused {} (held) {} (read before), walk reach {}, {} bytes in RAM",
            self.round,
            self.cells.len(),
            held,
            self.holdings.iter().filter(|h| !h.chunks.is_empty()).count(),
            self.read().len(),
            self.uncovered().len(),
            self.passed.len(),
            self.overlap(),
            self.refused_held,
            self.refused_visited,
            self.reach(),
            self.bytes()
        )
    }
}

/// The move-on rule: a tentacle moves on when its last read yielded less than the question's
/// running average.
#[must_use]
pub fn moves_on(last: f32, average: f32) -> bool {
    last < average
}

/// Each lens's share of the coverage targets from its yield (module notes): yield above zero
/// plus the mean such yield, over the sum; equal shares when nothing yielded.
#[must_use]
pub fn lens_shares(yields: &[Option<f32>]) -> Vec<f32> {
    let n = yields.len();
    if n == 0 {
        return Vec::new();
    }
    let pos: Vec<f32> = yields.iter().map(|y| y.unwrap_or(0.0).max(0.0)).collect();
    #[expect(clippy::cast_precision_loss, reason = "a small count")]
    let (sum, count) = (pos.iter().sum::<f32>(), n as f32);
    if sum <= 0.0 || !sum.is_finite() {
        return vec![1.0 / count; n];
    }
    let mean = sum / count;
    pos.iter().map(|y| (y + mean) / (2.0 * sum)).collect()
}

/// Weight of the coverage targets against the relation targets (module notes): the tokens one
/// round of windows holds over the tokens still uncovered, at most 1.
#[must_use]
pub fn coverage_weight(round_tokens: usize, uncovered_tokens: usize) -> f32 {
    if uncovered_tokens == 0 {
        return 1.0;
    }
    #[expect(clippy::cast_precision_loss, reason = "token counts are far below 2^52")]
    let w = (round_tokens as f64 / uncovered_tokens as f64).min(1.0) as f32;
    w
}

/// A candidate intake.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Target {
    pub mode: Mode,
    /// Relation mode: the seed chunk. Coverage mode: the first chunk of the region.
    pub anchor: usize,
    /// Expected value before any discount (relevance of the seed; lens share over region rank).
    pub utility: f32,
    /// Coverage mode: the lens and the chunks of the region, in the lens's order.
    pub lens: Option<usize>,
    pub region: Vec<usize>,
    /// Relation mode: the tentacle whose find the seed is (none: a seed from the question).
    pub owner: Option<usize>,
}

/// G2's order to one tentacle (behavioural: a mode and a target, no content).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Order {
    pub tentacle: usize,
    /// Index into the targets given to `assign`.
    pub target: usize,
    pub mode: Mode,
    pub anchor: usize,
    pub lens: Option<usize>,
    /// Utility after the discounts of the targets assigned before it, of the strongest's 1.
    pub utility: f32,
    /// The tentacle moved on (its last yield fell below the running average).
    pub moved: bool,
    pub reason: String,
}

/// Coverage targets from the lens lists: per lens, successive regions of chunks no tentacle
/// holds and no earlier round read, each region one window's worth (`budget` tokens), in the
/// lens's own order. The r-th next region of a lens is worth its share over r + 1, so a lens
/// that yielded offers more regions before another lens's first is passed. As many regions per
/// lens as there are tentacles at most (no more can be assigned in a round).
#[must_use]
pub fn coverage_targets(
    map: &AllocationMap,
    lists: &[Vec<usize>],
    shares: &[f32],
    tokens: &dyn Fn(usize) -> usize,
    budget: usize,
) -> Vec<Target> {
    let mut out = Vec::new();
    for (l, list) in lists.iter().enumerate() {
        let share = shares.get(l).copied().unwrap_or(0.0);
        let mut region: Vec<usize> = Vec::new();
        let (mut used, mut r) = (0_usize, 0_usize);
        let mut seen = BTreeSet::new();
        let close = |region: &mut Vec<usize>, r: &mut usize, out: &mut Vec<Target>| {
            if let Some(&anchor) = region.first() {
                #[expect(clippy::cast_precision_loss, reason = "a small count")]
                let utility = share / (*r as f32 + 1.0);
                out.push(Target { mode: Mode::Coverage, anchor, utility, lens: Some(l), region: std::mem::take(region), owner: None });
                *r += 1;
            }
        };
        for &c in list {
            if r >= map.tentacles() {
                break;
            }
            if !seen.insert(c) || map.cell(c).is_none_or(|x| x.holder.is_some() || x.visited) {
                continue;
            }
            let t = tokens(c);
            if t > budget {
                continue;
            }
            if used + t > budget {
                close(&mut region, &mut r, &mut out);
                used = 0;
                if r >= map.tentacles() {
                    break;
                }
            }
            region.push(c);
            used += t;
        }
        if r < map.tentacles() {
            close(&mut region, &mut r, &mut out);
        }
    }
    out
}

/// The allocation rule (module notes): each tentacle's next intake from the map, the targets
/// and the yields the map holds. `distance(a, b)` is the hop count between two anchors over the
/// specific relations when it is within the map's reach (none: farther, or unrelated). Returns
/// one order per tentacle that got a target, in the order assigned.
#[must_use]
pub fn assign(map: &AllocationMap, targets: &[Target], distance: &dyn Fn(usize, usize) -> Option<usize>) -> Vec<Order> {
    let n = map.tentacles();
    let average = map.running_average();
    let reach = map.reach();
    let top = targets.iter().map(|t| t.utility).fold(0.0_f32, f32::max);
    // Utilities of the strongest's 1, then discounted as targets are assigned.
    let mut utility: Vec<f32> = targets.iter().map(|t| if top > 0.0 { (t.utility / top).max(0.0) } else { 1.0 }).collect();
    let mut open: Vec<bool> = vec![true; targets.len()];
    // Per tentacle: (moved, stays with finds).
    let state: Vec<(bool, bool)> = (0..n)
        .map(|t| match (map.last_yield(t), average) {
            (Some(y), Some(a)) => (moves_on(y, a), !moves_on(y, a)),
            _ => (false, false),
        })
        .collect();
    let allowed = |open: &[bool], t: usize, k: usize| -> bool {
        let Some(target) = targets.get(k) else { return false };
        let (_, stays) = state.get(t).copied().unwrap_or((false, false));
        let owns_one = targets
            .iter()
            .enumerate()
            .any(|(j, x)| open.get(j).copied().unwrap_or(false) && x.mode == Mode::Relation && x.owner == Some(t));
        match target.mode {
            // Round 1 (no yields): coverage only. A staying tentacle elaborates its own finds
            // while it has any; a find belongs to its finder unless the finder moved on.
            Mode::Relation => {
                average.is_some()
                    && target.owner.is_none_or(|o| o == t || state.get(o).is_none_or(|s| s.0))
                    && (!stays || !owns_one || target.owner == Some(t))
            }
            Mode::Coverage | Mode::Scan => !(stays && owns_one),
        }
    };
    let mut free: Vec<bool> = vec![true; n];
    let mut orders = Vec::new();
    loop {
        let mut best: Option<(f32, usize, usize)> = None;
        for t in (0..n).filter(|t| free.get(*t).copied().unwrap_or(false)) {
            for k in (0..targets.len()).filter(|k| open.get(*k).copied().unwrap_or(false)) {
                let u = utility.get(k).copied().unwrap_or(0.0);
                if allowed(&open, t, k) && best.is_none_or(|b| u > b.0) {
                    best = Some((u, t, k));
                }
            }
        }
        let Some((u, t, k)) = best else { break };
        let Some(target) = targets.get(k) else { break };
        if let Some(x) = free.get_mut(t) {
            *x = false;
        }
        if let Some(x) = open.get_mut(k) {
            *x = false;
        }
        // Discount the targets near the chosen one.
        for (j, other) in targets.iter().enumerate() {
            if j == k || !open.get(j).copied().unwrap_or(false) {
                continue;
            }
            let near = if other.anchor == target.anchor { Some(0) } else { distance(target.anchor, other.anchor) };
            if let Some(d) = near.filter(|d| *d <= reach)
                && let Some(x) = utility.get_mut(j)
            {
                #[expect(clippy::cast_precision_loss, reason = "hop counts are small")]
                let covered = 1.0 - d as f32 / (reach as f32 + 1.0);
                *x *= 1.0 - covered;
            }
        }
        let (moved, stays) = state.get(t).copied().unwrap_or((false, false));
        let why = match (average, map.last_yield(t)) {
            (Some(a), Some(y)) if moved => format!("moved on: last yield {y:.3} below the running average {a:.3}"),
            (Some(a), Some(y)) if stays && target.owner == Some(t) => {
                format!("stays and elaborates its own find: last yield {y:.3} at or above the running average {a:.3}")
            }
            (Some(a), Some(y)) => format!("last yield {y:.3}, running average {a:.3}; no find of its own left"),
            (Some(_), None) => "no read yet this question".to_owned(),
            (None, _) => "round 1: nothing found yet, coverage".to_owned(),
        };
        orders.push(Order {
            tentacle: t,
            target: k,
            mode: target.mode,
            anchor: target.anchor,
            lens: target.lens,
            utility: u,
            moved,
            reason: why,
        });
    }
    orders
}

/// Self-check of `allocation`: one holder per chunk; read chunks are not taken again until
/// their content changes; passed-over chunks stay as references; the move-on rule; round 1 is
/// coverage; a tentacle that yielded elaborates its own find and one that did not moves on;
/// the discount sends the next tentacle to a farther target; yielding lenses get more room.
#[expect(clippy::too_many_lines, reason = "one check per rule of the module")]
pub(crate) fn check() -> Result<(), String> {
    let mut m = AllocationMap::new(12, 3);
    if !m.claim(0, 1, Mode::Coverage, None, 10, false)
        || m.claim(1, 1, Mode::Relation, Some(4), 10, false)
        || !m.claim(1, 2, Mode::Relation, Some(4), 10, false)
        || m.overlap() != 0
        || m.holder(1) != Some(0)
        || m.refused_held != 1
    {
        return Err(format!("nmbrain.allocation: claims: {}", m.describe()));
    }
    m.label(1, "relation", &[4]);
    m.follow(1, 4, 2);
    m.walked(3);
    m.pass_over(5, 0.4);
    m.pass_over(5, 0.2);
    m.pass_over(1, 0.9);
    if m.passed() != vec![(5, 0.4)]
        || m.cell(5).map(|c| c.status) != Some(Status::PassedOver)
        || m.reach() != 3
        || m.holding(1).is_none_or(|h| h.path != vec![(4, 2)] || h.tokens != 10 || h.lens != "relation")
    {
        return Err(format!("nmbrain.allocation: references and paths: {:?} {:?}", m.passed(), m.holding(1)));
    }
    m.record_yield(0, 0.2);
    m.record_yield(1, 0.8);
    let avg = m.running_average().unwrap_or(0.0);
    if (avg - 0.5).abs() > 1e-6 || !moves_on(0.2, avg) || moves_on(0.8, avg) {
        return Err(format!("nmbrain.allocation: running average {avg}"));
    }
    m.next_round();
    if m.round != 2
        || m.holder(1).is_some()
        || m.claim(2, 1, Mode::Coverage, None, 10, false)
        || m.refused_visited != 1
        || !m.claim(2, 1, Mode::Coverage, None, 10, true)
        || m.read() != vec![1, 2]
        || m.uncovered().len() != 10
    {
        return Err(format!("nmbrain.allocation: after the round: {}", m.describe()));
    }
    m.content_changed(2);
    if !m.claim(0, 2, Mode::Coverage, None, 10, false) || m.overlap() != 0 || (m.covered_share(&|_| 10) - 2.0 / 12.0).abs() > 1e-9 {
        return Err(format!("nmbrain.allocation: after a content change: {}", m.describe()));
    }
    // Lens shares: more room for the lens that yielded; a floor for the others; equal with none.
    let s = lens_shares(&[Some(0.9), Some(0.0), None]);
    if !(s.first() > s.get(1) && s.get(1) == s.get(2) && s.get(1).is_some_and(|x| *x > 0.0))
        || (s.iter().sum::<f32>() - 1.0).abs() > 1e-5
        || lens_shares(&[None, None]) != vec![0.5, 0.5]
    {
        return Err(format!("nmbrain.allocation: lens shares {s:?}"));
    }
    // Round 1: coverage only, each tentacle a different region, the yielding lens first.
    let fresh = AllocationMap::new(12, 3);
    let lists = vec![vec![0, 1, 2, 3], vec![1, 6, 7, 8], vec![]];
    let cover = coverage_targets(&fresh, &lists, &lens_shares(&[None, None, None]), &|_| 10, 20);
    let mut targets = cover.clone();
    targets.push(Target { mode: Mode::Relation, anchor: 9, utility: 5.0, owner: None, ..Target::default() });
    let far = |_: usize, _: usize| None;
    let o = assign(&fresh, &targets, &far);
    if o.len() != 3
        || o.iter().any(|x| x.mode != Mode::Coverage)
        || o.iter().map(|x| x.target).collect::<BTreeSet<_>>().len() != 3
        || cover.len() != 4
    {
        return Err(format!("nmbrain.allocation: round 1 orders {o:?} over {} coverage targets", cover.len()));
    }
    // Round 2: tentacle 1 yielded and elaborates its own find; tentacles 0 and 2 move on; of
    // two question seeds one hop apart the second mover takes the far coverage region instead.
    let mut m2 = AllocationMap::new(12, 3);
    for (t, c) in [(0, 0), (1, 4), (2, 5)] {
        let _ = m2.claim(t, c, Mode::Coverage, None, 10, false);
    }
    m2.record_yield(0, 0.1);
    m2.record_yield(1, 0.9);
    m2.record_yield(2, 0.2);
    m2.next_round();
    let targets = vec![
        Target { mode: Mode::Relation, anchor: 4, utility: 0.9, owner: Some(1), ..Target::default() },
        Target { mode: Mode::Relation, anchor: 9, utility: 1.0, owner: None, ..Target::default() },
        Target { mode: Mode::Relation, anchor: 10, utility: 0.95, owner: None, ..Target::default() },
        Target { mode: Mode::Coverage, anchor: 7, utility: 0.6, lens: Some(0), region: vec![7, 8], owner: None },
    ];
    let near = |a: usize, b: usize| ((a == 9 && b == 10) || (a == 10 && b == 9)).then_some(1);
    let o = assign(&m2, &targets, &near);
    let of = |t: usize| o.iter().find(|x| x.tentacle == t);
    let own = of(1).is_some_and(|x| x.target == 0 && !x.moved);
    let movers: BTreeSet<usize> = [0, 2].iter().filter_map(|t| of(*t)).filter(|x| x.moved).map(|x| x.target).collect();
    if !own || movers != BTreeSet::from([1, 3]) {
        return Err(format!("nmbrain.allocation: round 2 orders {o:?}"));
    }
    // Without the discount the second mover would have taken the near seed.
    let o = assign(&m2, &targets, &far);
    if [0, 2].iter().filter_map(|t| o.iter().find(|x| x.tentacle == *t)).map(|x| x.target).collect::<BTreeSet<_>>()
        != BTreeSet::from([1, 2])
    {
        return Err(format!("nmbrain.allocation: undiscounted orders {o:?}"));
    }
    if coverage_weight(96, 960) > 0.11 || coverage_weight(96, 50) < 1.0 || coverage_weight(96, 0) < 1.0 {
        return Err("nmbrain.allocation: coverage weight".into());
    }
    m2.next_question();
    if m2.round != 1 || m2.running_average().is_some() || m2.read().len() != 3 {
        return Err(format!("nmbrain.allocation: next question: {}", m2.describe()));
    }
    // The claim rule by meaning: chunks 1 and 2 are near-duplicates (group 1); with 1 held by
    // tentacle 0, tentacle 1 does not claim 2 unrequested, may on request, and after the round
    // the group counts as read.
    let mut m3 = AllocationMap::new(4, 2);
    m3.set_meaning(vec![0, 1, 1, u32::MAX], vec![u32::MAX, 1, 1, u32::MAX]);
    if !m3.claim(0, 1, Mode::Coverage, None, 10, false) || m3.claim(1, 2, Mode::Coverage, None, 10, false) || m3.refused_twin != 1 || m3.free_for(1, 2, false) {
        return Err("nmbrain.allocation: twin held".into());
    }
    if !m3.free_for(1, 2, true) || m3.cluster(3).is_some() || m3.clusters_seen() != BTreeSet::from([1]) {
        return Err("nmbrain.allocation: meaning clusters".into());
    }
    m3.next_round();
    if m3.free_for(0, 2, false) || !m3.free_for(0, 3, false) {
        return Err("nmbrain.allocation: twin read".into());
    }
    Ok(())
}
