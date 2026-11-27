//! The Map Coordinator (CM1, 2026-11-09; design by Luis Jake Gabriel III: the context map and
//! parameter control coordinate live, each keeping its own map). Per question it reads one
//! summary from each side, the context side from the GCX context map (`nmbrain::gmap`) and the model side from
//! the slots (engines bound, qualified, resident memory), and sets each side's budget by four
//! ordered rules. Each side clamps the budget to its own safe range, so a coordinator fault
//! cannot break either. It never reads or writes the Parameter Map (`nmnpcs::parmap`); the
//! only shared field is the task family.
use nmjson::Value;

/// The context side's summary of one question.
#[derive(Clone, Debug, PartialEq)]
pub struct ContextSummary {
    pub task_family: String,
    pub coverage: f32,
    pub scope_files: usize,
    pub salient: usize,
    pub conflicts: usize,
}

/// The model side's summary: what the Neuromorph's neural slots can contribute now.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelSummary {
    pub task_family: String,
    /// The reranker slot has a neural engine bound.
    pub reranker_engine: bool,
    /// That engine passed its qualification (law 8); otherwise the exact control decides.
    pub reranker_qualified: bool,
    /// Resident megabytes of the slot engines (RAM plus GPU).
    pub resident_mb: f32,
    /// The same split by device (2026-11-09).
    pub ram_mb: f32,
    pub gpu_mb: f32,
    /// GPU megabytes at which homeostasis suspends slots (`designs/homeostasis.json`).
    pub gpu_mb_max: f32,
    /// Slots with a neural engine bound (router, writer, embedder, reranker).
    pub engines: Vec<String>,
    /// Shadow comparisons of the reranker engine against its exact control, and the share whose
    /// first candidates agreed (none yet: 0 and 1.0).
    pub shadow: u64,
    pub shadow_agree: f32,
}

/// The budgets the coordinator sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budgets {
    /// Context side: segments refined by drones in the second round.
    pub refine: usize,
    /// Context side: tokens of the pack.
    pub pack_tokens: u32,
    /// Model side: candidates the neural reranker orders.
    pub rerank_cap: usize,
}

/// Context-side safe ranges.
pub const REFINE_MAX: usize = 8;
pub const PACK_MIN: u32 = 1000;
pub const PACK_MAX: u32 = 8000;
/// Model-side safe range.
pub const RERANK_MAX: usize = 12;
/// Refinement when no rule asks for more.
pub const REFINE_BASE: usize = 3;

/// The coordinator's decision: budgets after each side's clamp, and the rules that fired.
#[must_use]
pub fn coordinate(c: &ContextSummary, m: &ModelSummary, base: Budgets) -> (Budgets, Vec<&'static str>) {
    let mut b = base;
    let mut fired = Vec::new();
    // 1. Low model-side help (no qualified neural reranker): more evidence from the context
    //    side, never lower context precision.
    if !(m.reranker_engine && m.reranker_qualified) {
        b.refine += 1;
        fired.push("model side exact only: one more refined segment");
    }
    // 2. A wide scope (many salient segments): more pack room for the evidence.
    if c.salient > 6 {
        b.pack_tokens = b.pack_tokens.saturating_add(b.pack_tokens / 4);
        fired.push("wide scope: pack budget +25%");
    }
    // 3. Disagreement: tighten both sides, refine more on the context side and order fewer
    //    candidates on the model side.
    if c.conflicts > 0 {
        b.refine += 1;
        b.rerank_cap = b.rerank_cap.saturating_sub(b.rerank_cap / 3);
        fired.push("conflicts: one more refined segment, neural ordering narrowed");
    }
    // 4. GPU memory near the homeostasis limit: the model side orders half as many candidates.
    if m.gpu_mb_max > 0.0 && m.gpu_mb >= 0.9 * m.gpu_mb_max {
        b.rerank_cap /= 2;
        fired.push("GPU memory near its limit: neural ordering halved");
    }
    // 5. Nothing salient: refinement has no target.
    if c.salient == 0 {
        b.refine = 0;
        fired.push("nothing salient: no refinement");
    }
    // Each side clamps to its own range (allowing an explicit pack token allowance up to 16,000).
    let pack_cap = if base.pack_tokens <= 16_000 { PACK_MAX.max(base.pack_tokens) } else { PACK_MAX };
    b.refine = b.refine.min(REFINE_MAX);
    b.pack_tokens = b.pack_tokens.clamp(PACK_MIN, pack_cap);
    b.rerank_cap = b.rerank_cap.min(RERANK_MAX);
    (b, fired)
}

/// The decision as JSON, with both summaries, for the context map and the trace.
#[must_use]
pub fn to_value(c: &ContextSummary, m: &ModelSummary, b: Budgets, fired: &[&str]) -> Value {
    let n = |x: usize| Value::Number(x.to_string());
    let s = |t: &str| Value::String(t.to_owned());
    let o = |v: Vec<(&str, Value)>| Value::Object(v.into_iter().map(|(k, v)| (k.to_owned(), v)).collect());
    o(vec![
        ("task_family", s(&c.task_family)),
        (
            "context",
            o(vec![
                ("coverage", Value::Number(format!("{:.3}", c.coverage))),
                ("scope_files", n(c.scope_files)),
                ("salient", n(c.salient)),
                ("conflicts", n(c.conflicts)),
            ]),
        ),
        (
            "model",
            o(vec![
                ("reranker_engine", Value::Bool(m.reranker_engine)),
                ("reranker_qualified", Value::Bool(m.reranker_qualified)),
                ("resident_mb", Value::Number(format!("{:.1}", m.resident_mb))),
                ("ram_mb", Value::Number(format!("{:.1}", m.ram_mb))),
                ("gpu_mb", Value::Number(format!("{:.1}", m.gpu_mb))),
                ("gpu_mb_max", Value::Number(format!("{:.0}", m.gpu_mb_max))),
                ("engines", Value::Array(m.engines.iter().map(|e| s(e)).collect())),
                ("shadow", n(usize::try_from(m.shadow).unwrap_or(usize::MAX))),
                ("shadow_agree", Value::Number(format!("{:.3}", m.shadow_agree))),
            ]),
        ),
        ("budgets", o(vec![("refine", n(b.refine)), ("pack_tokens", n(b.pack_tokens as usize)), ("rerank_cap", n(b.rerank_cap))])),
        ("rules", Value::Array(fired.iter().map(|r| s(r)).collect())),
    ])
}

// ---------------------------------------------------------------------------------------------
// The tentacles as one team: the allocation layer. The Map Coordinator decides where each
// tentacle looks; G1's allocation map applies every decision by claim, so no piece is in two
// windows. The team maximises one objective, the evidence about the request its windows yield
// together: a window is worth the belief of each piece it holds, times the chance its read ranks
// that piece well against the rest of the window (the observation model). Assignments are chosen
// one at a time by best gain over what is already claimed, so pieces another tentacle holds add
// nothing and the worth of every other target falls with what was taken (coverage with
// diminishing returns). Nothing here is a share or a quota: every number comes from the belief,
// the window's own read and the round's yields.
// ---------------------------------------------------------------------------------------------

/// One way a tentacle reads: the evidence it leads with, one score per piece of the material.
#[derive(Clone, Debug, Default)]
pub struct Perspective {
    pub name: String,
    /// The score set its `g0_tx` reads by.
    pub statistic: String,
    pub score: Vec<f32>,
}

/// What a tentacle is sent to read.
#[derive(Clone, Debug, PartialEq)]
pub enum Aim {
    /// Coverage mode: the material in perspective `0`'s order.
    Cover(usize),
    /// Relation mode: a best-first walk over the specific relations from these seeds (piece,
    /// strength). `owner`: the tentacle whose finds the seeds are (none: the frontier's).
    Follow { seeds: Vec<(usize, f32)>, owner: Option<usize> },
    /// Coverage of one request aspect through one perspective (Phase F2, 2026-11-19): the best
    /// unclaimed pieces for aspect `a` under perspective `p`'s lead kind (the perspective's score
    /// times the aspect's belief), then the perspective's own order.
    Aspect { p: usize, a: usize },
}

/// One tentacle's intake for a round.
#[derive(Clone, Debug)]
pub struct Assignment {
    pub tentacle: usize,
    pub aim: Aim,
    /// The perspective its `g0_tx` reads the window by.
    pub perspective: usize,
    pub window: Vec<usize>,
    /// The window's worth under the belief when it was chosen.
    pub worth: f32,
    /// The function the tentacle has now (its label) and why it was given it.
    pub label: String,
    pub why: String,
}

/// The material as the team sees it for one request.
pub struct Team<'a> {
    /// How likely each piece is to hold part of the answer (0 to 1).
    pub belief: Vec<f32>,
    pub tokens: &'a dyn Fn(usize) -> usize,
    pub graph: &'a nmsubs::refgraph::Specific,
    pub perspectives: &'a [Perspective],
    /// Window capacity in tokens.
    pub budget: usize,
    /// The perspective relation windows are read by (the whole evidence).
    pub evidence: usize,
    /// Per aspect of the request, each piece's belief for that aspect (0 to 1), and the chance
    /// each aspect is still unfound by the windows taken so far. With `cover`, a window is worth
    /// what it adds to the chance every aspect is found (coverage with diminishing returns: an
    /// aspect the team already holds adds little more); without, the belief it carries.
    pub aspects: Vec<Vec<f32>>,
    pub missing: Vec<f32>,
    pub cover: bool,
    /// Meaning in the map (empty: off): each piece's meaning cluster (`u32::MAX`: none). With it,
    /// a piece adds half as much again for each piece of its cluster the team already took, so
    /// windows spread over the material's meanings (coverage with diminishing returns by meaning).
    pub meaning: Vec<u32>,
    taken_by_cluster: std::collections::BTreeMap<u32, usize>,
    /// Each perspective's pieces in its own order (score, then belief, then piece).
    orders: Vec<Vec<usize>>,
    /// Lane T1 (Luis, 2026-11-18 23:35 and 23:39, firm): every aim at most once a round, so no two
    /// tentacles read through the same perspective and request aspect in one round.
    pub distinct: bool,
    /// Burgard's utility discount (Burgard et al. 2005; `nmbrain::allocation`): per piece, the share
    /// of its worth left after this round's windows, `d / (reach + 1)` at `d` relation hops from a
    /// piece a window of this round took (1 beyond `reach`); empty: off.
    pub discount: Vec<f32>,
    /// The relation depth of the discount (`nmbrain::allocation::REACH_FALLBACK` until walks ran).
    pub reach: usize,
}

/// The noise band of a set of values: their median absolute deviation, the mean absolute
/// deviation where that is 0, 0 for fewer than two values.
#[must_use]
pub fn noise_band(v: &[f32]) -> f32 {
    let mut f: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if f.len() < 2 {
        return 0.0;
    }
    f.sort_by(f32::total_cmp);
    let mid = |w: &[f32]| {
        let n = w.len();
        if n % 2 == 1 { w.get(n / 2).copied().unwrap_or(0.0) } else { (w.get(n / 2 - 1).copied().unwrap_or(0.0) + w.get(n / 2).copied().unwrap_or(0.0)) / 2.0 }
    };
    let m = mid(&f);
    let mut dev: Vec<f32> = f.iter().map(|x| (x - m).abs()).collect();
    dev.sort_by(f32::total_cmp);
    let mad = mid(&dev);
    #[expect(clippy::cast_precision_loss, reason = "a small count")]
    let mean = dev.iter().sum::<f32>() / dev.len() as f32;
    if mad > 0.0 { mad } else { mean }
}

/// Personalised PageRank over the specific relations (HippoRAG, Gutierrez et al. 2024: damping
/// 0.5, the seeds' strengths as the restart distribution), `iters` power steps, edges by weight
/// over the node's total. It proposes where to look (the frontier); it never ranks or votes
/// (master design 3.4 rule 15; C39, C40).
#[must_use]
pub fn personalised_pagerank(graph: &nmsubs::refgraph::Specific, n: usize, seeds: &[(usize, f32)], iters: usize) -> Vec<f32> {
    const DAMP: f32 = 0.5;
    let total: f32 = seeds.iter().map(|s| s.1.max(0.0)).sum();
    let mut restart = vec![0.0_f32; n];
    if total <= 0.0 {
        return restart;
    }
    for (c, w) in seeds {
        if let Some(x) = restart.get_mut(*c) {
            *x += w.max(0.0) / total;
        }
    }
    let mut p = restart.clone();
    for _ in 0..iters {
        let mut next: Vec<f32> = restart.iter().map(|r| (1.0 - DAMP) * r).collect();
        for (c, mass) in p.iter().enumerate().filter(|(_, m)| **m > 1e-9) {
            let edges = graph.edges_of(c);
            let sum: f32 = edges.iter().map(|e| e.weight.max(0.0)).sum();
            if sum <= 0.0 {
                if let Some(x) = next.get_mut(c) {
                    *x += DAMP * mass;
                }
                continue;
            }
            for e in edges {
                if let Some(x) = next.get_mut(e.to as usize) {
                    *x += DAMP * mass * e.weight.max(0.0) / sum;
                }
            }
        }
        p = next;
    }
    p
}

/// The observation model: the chance a read reports a piece it holds at place `rank` of its
/// window when its shortlist gate is `depth` (1 at the top, one half at the gate, falling
/// beyond it as the gate's own size against the place).
#[must_use]
pub fn reported_chance(rank: usize, depth: usize) -> f32 {
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    let p = depth.max(1) as f32 / (depth.max(1) + rank) as f32;
    p
}

impl<'a> Team<'a> {
    #[must_use]
    pub fn new(
        belief: Vec<f32>,
        tokens: &'a dyn Fn(usize) -> usize,
        graph: &'a nmsubs::refgraph::Specific,
        perspectives: &'a [Perspective],
        budget: usize,
        evidence: usize,
    ) -> Self {
        let b = |c: usize| belief.get(c).copied().unwrap_or(0.0);
        let orders = perspectives
            .iter()
            .map(|p| {
                let s = |c: usize| p.score.get(c).copied().unwrap_or(0.0);
                let mut v: Vec<usize> = (0..belief.len()).collect();
                v.sort_by(|x, y| s(*y).total_cmp(&s(*x)).then(b(*y).total_cmp(&b(*x))).then(x.cmp(y)));
                v
            })
            .collect();
        Self {
            belief,
            tokens,
            graph,
            perspectives,
            budget,
            evidence,
            orders,
            aspects: Vec::new(),
            missing: Vec::new(),
            cover: false,
            meaning: Vec::new(),
            taken_by_cluster: std::collections::BTreeMap::new(),
            distinct: false,
            discount: Vec::new(),
            reach: nmbrain::allocation::REACH_FALLBACK,
        }
    }
    fn b(&self, c: usize) -> f32 {
        self.belief.get(c).copied().unwrap_or(0.0) * self.spread(c) * self.discount.get(c).copied().unwrap_or(1.0)
    }
    /// Starts a round's discount: every piece worth its whole belief again (`on`), or none (off).
    pub fn begin_round(&mut self, on: bool) {
        self.discount = if on { vec![1.0; self.belief.len()] } else { Vec::new() };
    }
    /// Burgard's discount after a window: each piece within `reach` relation hops of the window
    /// keeps `d / (reach + 1)` of its worth (the share the window's walk is not expected to cover).
    fn discount_from(&mut self, window: &[usize]) {
        if self.discount.is_empty() {
            return;
        }
        let mut frontier: Vec<usize> = window.to_vec();
        let mut seen: std::collections::BTreeSet<usize> = window.iter().copied().collect();
        for d in 1..=self.reach {
            let mut next = Vec::new();
            for &c in &frontier {
                for e in self.graph.edges_of(c) {
                    let j = e.to as usize;
                    if seen.insert(j) {
                        next.push(j);
                    }
                }
            }
            #[expect(clippy::cast_precision_loss, reason = "small hop counts")]
            let keep = d as f32 / (self.reach + 1) as f32;
            for &j in &next {
                if let Some(x) = self.discount.get_mut(j) {
                    *x = x.min(keep);
                }
            }
            frontier = next;
            if frontier.is_empty() {
                break;
            }
        }
    }
    /// The order of an aspect aim: the perspective's score (at least 0) times the aspect's
    /// belief, then the perspective's own order for the rest.
    fn aspect_order(&self, p: usize, a: usize) -> Vec<usize> {
        let Some(base) = self.orders.get(p) else { return Vec::new() };
        let s = |c: usize| self.perspectives.get(p).and_then(|x| x.score.get(c)).copied().unwrap_or(0.0).max(0.0);
        let r = |c: usize| self.aspects.get(a).and_then(|row| row.get(c)).copied().unwrap_or(0.0).max(0.0);
        let mut lead: Vec<usize> = base.iter().copied().filter(|c| s(*c) * r(*c) > 0.0).collect();
        lead.sort_by(|x, y| (s(*y) * r(*y)).total_cmp(&(s(*x) * r(*x))).then(x.cmp(y)));
        let set: std::collections::BTreeSet<usize> = lead.iter().copied().collect();
        lead.extend(base.iter().copied().filter(|c| !set.contains(c)));
        lead
    }
    /// A piece's share of its worth by meaning: 0.5 to the power of the pieces of its cluster
    /// the team already took (1 without meaning or a cluster).
    fn spread(&self, c: usize) -> f32 {
        let Some(k) = self.meaning.get(c).copied().filter(|k| *k != u32::MAX) else { return 1.0 };
        let n = self.taken_by_cluster.get(&k).copied().unwrap_or(0);
        0.5_f32.powi(i32::try_from(n).unwrap_or(i32::MAX))
    }
    /// The window `aim` would take now: only pieces `free` allows, never more than the budget,
    /// and full (a relation walk that runs out of relations fills its room by the belief).
    #[must_use]
    pub fn window(&self, aim: &Aim, free: &dyn Fn(usize) -> bool) -> Vec<usize> {
        let tk = |c: usize| (self.tokens)(c).max(1);
        let mut out: Vec<usize> = Vec::new();
        let mut room = self.budget;
        let fill = |order: &[usize], out: &mut Vec<usize>, room: &mut usize| {
            for &c in order {
                if *room == 0 {
                    break;
                }
                let t = tk(c);
                if t <= *room && free(c) && !out.contains(&c) {
                    out.push(c);
                    *room -= t;
                }
            }
        };
        match aim {
            Aim::Cover(p) => {
                if let Some(o) = self.orders.get(*p) {
                    fill(o, &mut out, &mut room);
                }
            }
            Aim::Aspect { p, a } => {
                let o = self.aspect_order(*p, *a);
                fill(&o, &mut out, &mut room);
            }
            Aim::Follow { seeds, .. } => {
                let walk = nmsubs::lens::relation_window_with(
                    self.graph,
                    seeds,
                    &|c| !free(c),
                    &tk,
                    self.budget,
                    nmsubs::lens::Stop::MarginalGain,
                    nmsubs::lens::KindWeights::Derived,
                    &|c| self.b(c),
                );
                room = self.budget.saturating_sub(walk.tokens);
                out = walk.chunks;
                if let Some(o) = self.orders.get(self.evidence) {
                    fill(o, &mut out, &mut room);
                }
            }
        }
        out
    }
    /// The perspective a window taken for `aim` is read by.
    #[must_use]
    pub fn reader(&self, aim: &Aim) -> usize {
        match aim {
            Aim::Cover(p) | Aim::Aspect { p, .. } => *p,
            Aim::Follow { .. } => self.evidence,
        }
    }
    /// A window's place order under perspective `p` and its shortlist gate.
    #[must_use]
    pub fn read(&self, window: &[usize], p: usize) -> (Vec<usize>, usize) {
        let s = |c: usize| self.perspectives.get(p).and_then(|x| x.score.get(c)).copied().unwrap_or(0.0);
        let mut v = window.to_vec();
        v.sort_by(|x, y| s(*y).total_cmp(&s(*x)).then(self.b(*y).total_cmp(&self.b(*x))).then(x.cmp(y)));
        let scores: Vec<f32> = v.iter().map(|c| s(*c)).collect();
        let (depth, _) = nmbrain::g1::shortlist_by(nmbrain::g1::rules().shortlist, &scores, crate::tentacle_core::SHORTLIST_FALLBACK);
        (v, depth)
    }
    /// What a window read by perspective `p` is worth: each piece's belief times the chance the
    /// read reports it from its place.
    #[must_use]
    pub fn worth(&self, window: &[usize], p: usize) -> f32 {
        let (order, depth) = self.read(window, p);
        if self.cover && !self.aspects.is_empty() {
            return self
                .aspects
                .iter()
                .zip(&self.missing)
                .map(|(row, m)| m * (1.0 - self.unfound(row, &order, depth)))
                .sum();
        }
        order.iter().enumerate().map(|(r, c)| self.b(*c) * reported_chance(r, depth)).sum()
    }
    /// The chance a read of `order` (its gate `depth`) reports no piece of one aspect's `row`.
    fn unfound(&self, row: &[f32], order: &[usize], depth: usize) -> f32 {
        order.iter().enumerate().map(|(r, c)| 1.0 - (row.get(*c).copied().unwrap_or(0.0).clamp(0.0, 1.0) * self.spread(*c)) * reported_chance(r, depth)).product()
    }
    /// One round's assignment: repeatedly the (tentacle, aim) pair whose window adds most to
    /// the team's worth, its pieces claimed in `map` before the next choice, so later windows
    /// hold only what no tentacle holds. `allowed(t, k)` says whether tentacle `t` may take aim
    /// `k` (a tentacle elaborating its own finds takes its own walk); `floor` is the least worth
    /// a new window must add (none in round 1). A coverage aim can be taken by several
    /// tentacles in turn (the next region of the same perspective); a walk once.
    pub fn assign(
        &mut self,
        map: &mut nmbrain::allocation::AllocationMap,
        tentacles: &[usize],
        aims: &[Aim],
        allowed: &dyn Fn(usize, usize) -> bool,
        floor: &dyn Fn(usize) -> f32,
    ) -> Vec<Assignment> {
        let mut free_t: Vec<usize> = tentacles.to_vec();
        let mut used = vec![false; aims.len()];
        let mut out: Vec<Assignment> = Vec::new();
        // Diminishing returns by perspective (roadmap Phase 0 item 1, 2026-11-23), the meaning
        // clusters' rule applied to the lenses: in a coverage round, a coverage or aspect window
        // through a perspective that already has `n` windows this round counts 0.5 to the power `n`
        // (`NANO_PERSPECTIVE_SPREAD=off` for diagnosis). Without it the aspect worth piled windows
        // onto a few perspectives (large code: three each through parts and structure) and lenses
        // whose own first pieces hold the answer went unread (first 26 to 25, trc05 and trc06
        // withheld). Follow aims (relations, scan, tree) are not discounted.
        let spread_p = self.cover && !std::env::var("NANO_PERSPECTIVE_SPREAD").is_ok_and(|v| v == "off");
        let mut by_p: std::collections::BTreeMap<usize, i32> = std::collections::BTreeMap::new();
        // The whole evidence's view is already in the coverage model (the aspect rows are its own
        // calibrated evidence), so its worth there partly measures agreement with itself: it starts
        // the round as if it held one window (`NANO_EVIDENCE_COUNTED=off` for diagnosis). Without
        // this it claimed the shared first pieces before the lenses whose reads place them.
        if spread_p && !self.aspects.is_empty() && !std::env::var("NANO_EVIDENCE_COUNTED").is_ok_and(|v| v == "off") {
            by_p.insert(self.evidence, 1);
        }
        while !free_t.is_empty() {
            let mut best: Option<(f32, usize, usize, Vec<usize>)> = None;
            // The whole evidence reads last in a coverage round: when one tentacle is left and the
            // evidence has no window yet, only its coverage or aspect aims are considered, so it
            // reads the pieces the lenses left (its place in the old round 1, where its window held
            // large code trc06's answer; counted once above, it otherwise lost round 1 entirely).
            let evidence_last = spread_p
                && free_t.len() == 1
                && aims.iter().enumerate().any(|(k, a)| !used.get(k).copied().unwrap_or(true) && matches!(a, Aim::Cover(_) | Aim::Aspect { .. }) && self.reader(a) == self.evidence)
                && !out.iter().any(|x| x.perspective == self.evidence && matches!(x.aim, Aim::Cover(_) | Aim::Aspect { .. }));
            for (k, aim) in aims.iter().enumerate() {
                if used.get(k).copied().unwrap_or(true) {
                    continue;
                }
                if evidence_last && !(matches!(aim, Aim::Cover(_) | Aim::Aspect { .. }) && self.reader(aim) == self.evidence) {
                    continue;
                }
                // The window does not depend on which free tentacle takes it (nothing is held
                // by a free tentacle this round), so it is formed once per aim.
                let Some(&t0) = free_t.iter().find(|t| allowed(**t, k)) else { continue };
                let w = self.window(aim, &|c| map.free_for(t0, c, false));
                if w.is_empty() {
                    continue;
                }
                let v = self.worth(&w, self.reader(aim));
                let v = if spread_p && matches!(aim, Aim::Cover(_) | Aim::Aspect { .. }) { v * 0.5_f32.powi(by_p.get(&self.reader(aim)).copied().unwrap_or(0)) } else { v };
                for &t in free_t.iter().filter(|t| allowed(**t, k)) {
                    if v >= floor(t) && best.as_ref().is_none_or(|b| v > b.0) {
                        best = Some((v, t, k, w.clone()));
                    }
                }
            }
            let Some((v, mut t, k, w)) = best else { break };
            // A walk over a tentacle's own finds goes to that tentacle when it is free: it stays
            // and elaborates while that walk is worth the most it could do.
            if let Some(Aim::Follow { owner: Some(o), .. }) = aims.get(k)
                && free_t.contains(o)
                && allowed(*o, k)
            {
                t = *o;
            }
            let Some(aim) = aims.get(k) else { break };
            let mode = if matches!(aim, Aim::Follow { .. }) { nmbrain::allocation::Mode::Relation } else { nmbrain::allocation::Mode::Coverage };
            let source = match aim {
                Aim::Follow { seeds, .. } => seeds.first().map(|s| s.0),
                Aim::Cover(_) | Aim::Aspect { .. } => None,
            };
            let mut taken = Vec::new();
            for &c in &w {
                if map.claim(t, c, mode, source, (self.tokens)(c).max(1), false) {
                    taken.push(c);
                }
            }
            let p = self.reader(aim);
            if matches!(aim, Aim::Cover(_) | Aim::Aspect { .. }) {
                *by_p.entry(p).or_insert(0) += 1;
            }
            if self.cover && !self.aspects.is_empty() {
                let (order, depth) = self.read(&taken, p);
                let left: Vec<f32> = self.aspects.iter().map(|row| self.unfound(row, &order, depth)).collect();
                for (m, l) in self.missing.iter_mut().zip(left) {
                    *m *= l;
                }
            }
            self.discount_from(&taken);
            for c in &taken {
                if let Some(k) = self.meaning.get(*c).copied().filter(|k| *k != u32::MAX) {
                    *self.taken_by_cluster.entry(k).or_insert(0) += 1;
                }
            }
            let pname = self.perspectives.get(p).map_or("evidence", |x| x.name.as_str());
            let (label, why) = match aim {
                Aim::Cover(_) => (format!("{pname}, coverage"), format!("coverage by {pname}: worth {v:.3}")),
                Aim::Aspect { a, .. } => (format!("{pname}, aspect {}", a + 1), format!("aspect {} by {pname}: worth {v:.3}", a + 1)),
                Aim::Follow { owner, .. } => (
                    match owner {
                        Some(o) if *o == t => "relations of its own finds".to_owned(),
                        Some(_) => "relations of another's finds".to_owned(),
                        None => "relations of the frontier".to_owned(),
                    },
                    match owner {
                        Some(o) if *o == t => format!("stays and follows the relations of its own finds: worth {v:.3}"),
                        Some(o) => format!("follows the relations of t{}'s finds: worth {v:.3}", o + 1),
                        None => format!("follows the frontier: worth {v:.3}"),
                    },
                ),
            };
            map.label(t, &label, &source.into_iter().collect::<Vec<_>>());
            if (self.distinct || matches!(aim, Aim::Follow { .. }))
                && let Some(x) = used.get_mut(k)
            {
                *x = true;
            }
            free_t.retain(|x| *x != t);
            out.push(Assignment { tentacle: t, aim: aim.clone(), perspective: p, window: taken, worth: v, label, why });
        }
        out
    }
}

// ---------------------------------------------------------------------------------------------
// Tree prior (WP-C4 item 4, 2026-11-23): files near the found ones in the shaped structural
// tree (`nmsubs::tree_shape`) as owner-less follow seeds, so cross-file architectural questions
// reach sibling files. Off by default; the lead measures it before it is turned on.
// ---------------------------------------------------------------------------------------------

/// The tree prior switch: on by default (part states, 2026-11-23); `NANO_TREE_PRIOR=off` only for
/// a diagnosis run.
#[must_use]
pub fn tree_prior_on() -> bool {
    !std::env::var("NANO_TREE_PRIOR").is_ok_and(|v| v == "off")
}

/// What one cached tree was built for: root, version and file count of the substrate, and the
/// world model's note count (its summaries follow the written notes).
type TreeKey = (std::path::PathBuf, u64, usize, usize);
/// The cached tree with what it was built for.
type TreeSlot = Option<(TreeKey, std::sync::Arc<nmsubs::tree_shape::Tree>)>;
static TREE: std::sync::OnceLock<std::sync::Mutex<TreeSlot>> = std::sync::OnceLock::new();

/// The shaped tree of `sub`, built once per substrate version and held in RAM (default shape,
/// structural summaries; written notes can be passed to `nmsubs::tree_shape::build` later).
#[must_use]
pub fn tree_for(sub: &nmsubs::Substrate) -> std::sync::Arc<nmsubs::tree_shape::Tree> {
    let key: TreeKey = (sub.root.clone(), sub.version, sub.files.len(), crate::writer_idle::notes_held());
    let cell = TREE.get_or_init(|| std::sync::Mutex::new(None));
    if let Ok(g) = cell.lock()
        && let Some((k, t)) = g.as_ref()
        && *k == key
    {
        return std::sync::Arc::clone(t);
    }
    // Roadmap Phase 1 (2026-11-24): the world model's notes (L1 to L3) as the tree's summaries,
    // structural signatures where none is written.
    let mut tree = nmsubs::tree_shape::build(sub, Some(&crate::writer_idle::tree_note), nmsubs::tree_shape::Shape::default());
    // The dependency edges on the structure: a piece that uses an item another file's piece
    // defines makes its file depend on that file (the reference graph's specific relations).
    let refs = sub.refs();
    let path_of = |c: usize| sub.chunks.get(c).and_then(|ch| sub.files.get(ch.file)).map(|f| f.path.replace('\\', "/"));
    let mut edges: Vec<(String, String)> = Vec::new();
    for i in 0..sub.chunks.len() {
        let Some(a) = path_of(i) else { continue };
        for e in refs.specific.edges_of(i) {
            if e.dir == nmsubs::refgraph::Relation::Uses
                && let Some(b) = path_of(e.to as usize)
            {
                edges.push((a.clone(), b));
            }
        }
    }
    tree.set_dependencies(edges);
    let t = std::sync::Arc::new(tree);
    if let Ok(mut g) = cell.lock() {
        *g = Some((key, std::sync::Arc::clone(&t)));
    }
    t
}

/// The top piece of each of the `k` files the tree prior ranks highest around the files of
/// `found_chunks`, with that file's prior weight: per file the chunk `free` allows and the finds do
/// not hold with the highest `score` (the question's evidence for it), then the largest, then the
/// lowest index. Best first. Roadmap Phase 0 item 1 (2026-11-23): the first form took each file's
/// largest chunk whatever the question asked, so its windows carried no evidence and lost every
/// assignment (large code trace: offered in 15 rounds, taken in none).
fn tree_pieces(sub: &nmsubs::Substrate, tree: &nmsubs::tree_shape::Tree, found_chunks: &[usize], k: usize, free: &dyn Fn(usize) -> bool, score: &dyn Fn(usize) -> f32) -> Vec<(usize, f32)> {
    let found: std::collections::BTreeSet<usize> = found_chunks.iter().copied().collect();
    let files: std::collections::BTreeSet<String> =
        found_chunks.iter().filter_map(|c| sub.chunks.get(*c)).filter_map(|c| sub.files.get(c.file)).map(|f| f.path.clone()).collect();
    let files: Vec<String> = files.into_iter().collect();
    let mut out: Vec<(usize, f32)> = Vec::new();
    for (path, w) in nmsubs::tree_shape::prior(tree, &files, k) {
        let best = tree
            .file_chunks(&path)
            .iter()
            .copied()
            .filter(|c| !found.contains(c) && free(*c))
            .max_by(|a, b| {
                let t = |c: usize| sub.chunks.get(c).map_or(0, |x| x.tokens);
                let s = |c: usize| score(c).max(f32::MIN);
                s(*a).total_cmp(&s(*b)).then(t(*a).cmp(&t(*b))).then(b.cmp(a))
            });
        if let Some(c) = best
            && !out.iter().any(|x| x.0 == c)
        {
            out.push((c, w.max(f32::MIN_POSITIVE)));
        }
    }
    out
}

/// Chunk indices of the top pieces of the files nearest the found ones in `tree` (one per
/// neighbour file, `k` files at most, best first). Pure: no switch, no cache.
#[must_use]
pub fn tree_aims(sub: &nmsubs::Substrate, tree: &nmsubs::tree_shape::Tree, found_chunks: &[usize], k: usize) -> Vec<usize> {
    tree_pieces(sub, tree, found_chunks, k, &|_| true, &|_| 0.0).into_iter().map(|x| x.0).collect()
}

/// Owner-less follow seeds from the tree prior for the served flow (empty with
/// `NANO_TREE_PRIOR=off`): per neighbour file of the finds in the cached tree of `sub`, its piece
/// with the highest `score` (the question's fused evidence) that `free` allows (pass the allocation
/// map's unheld, unvisited test as the frontier does), with the file's prior weight.
///
/// The lead wires it into `drone_path.rs` (`stack_core`, team arm, the round loop) beside the
/// frontier seeds, as
/// `let finds: Vec<usize> = round.finds.iter().flatten().copied().collect();`
/// `let ts = crate::coordinator::tree_seeds(sub, &finds, SCAN_SEEDS, &|c| g1.map.cell(c).is_some_and(|x| x.holder.is_none() && !x.visited));`
/// `if !ts.is_empty() { next_aims.push(crate::coordinator::Aim::Follow { seeds: ts, owner: None }); }`.
/// Pieces are substrate chunk indices, as the team's pieces are with `NANO_ITEMS` off.
#[must_use]
pub fn tree_seeds(sub: &nmsubs::Substrate, found_chunks: &[usize], k: usize, free: &dyn Fn(usize) -> bool, score: &dyn Fn(usize) -> f32) -> Vec<(usize, f32)> {
    if !tree_prior_on() || found_chunks.is_empty() {
        return Vec::new();
    }
    let tree = tree_for(sub);
    tree_pieces(sub, &tree, found_chunks, k, free, score)
}

/// Self-check of the `coordinator` sub-component: rules fire in order and every budget stays
/// inside its side's range.
pub fn check() -> Result<(), String> {
    let base = Budgets { refine: REFINE_BASE, pack_tokens: 4000, rerank_cap: 12 };
    let c = ContextSummary { task_family: "context".into(), coverage: 0.5, scope_files: 100, salient: 9, conflicts: 1 };
    let m = ModelSummary {
        task_family: "context".into(),
        reranker_engine: false,
        reranker_qualified: false,
        resident_mb: 0.0,
        ram_mb: 0.0,
        gpu_mb: 0.0,
        gpu_mb_max: 3500.0,
        engines: Vec::new(),
        shadow: 0,
        shadow_agree: 1.0,
    };
    let (b, fired) = coordinate(&c, &m, base);
    if b != (Budgets { refine: 5, pack_tokens: 5000, rerank_cap: 8 }) || fired.len() != 3 {
        return Err(format!("nmapi.coordinator: {b:?} {fired:?}"));
    }
    let (z, _) = coordinate(&ContextSummary { salient: 0, ..c.clone() }, &m, Budgets { refine: 99, pack_tokens: 99_999, rerank_cap: 99 });
    if z.refine != 0 || z.pack_tokens != PACK_MAX || z.rerank_cap != RERANK_MAX {
        return Err(format!("nmapi.coordinator: clamps {z:?}"));
    }
    let (g, _) = coordinate(&ContextSummary { salient: 1, conflicts: 0, ..c }, &ModelSummary { gpu_mb: 3400.0, ..m.clone() }, base);
    if g.rerank_cap != 6 {
        return Err(format!("nmapi.coordinator: GPU pressure rule {g:?}"));
    }
    // The team: three tentacles, six pieces of 10 tokens, windows of 20; every piece claimed
    // once, no window over its budget, and the strongest piece in the first window taken.
    let graph = nmsubs::refgraph::Specific::default();
    let tk = |_: usize| 10;
    let belief = vec![1.0, 0.2, 0.9, 0.1, 0.5, 0.05];
    let ps = vec![
        Perspective { name: "evidence".into(), statistic: "fused".into(), score: belief.clone() },
        Perspective { name: "other".into(), statistic: "other".into(), score: vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0] },
    ];
    let mut team = Team::new(belief, &tk, &graph, &ps, 20, 0);
    let mut map = nmbrain::allocation::AllocationMap::new(6, 3);
    let aims = vec![Aim::Cover(0), Aim::Cover(1)];
    let picks = team.assign(&mut map, &[0, 1, 2], &aims, &|_, _| true, &|_| f32::NEG_INFINITY);
    let mut all: Vec<usize> = picks.iter().flat_map(|a| a.window.clone()).collect();
    let n = all.len();
    all.sort_unstable();
    all.dedup();
    if all.len() != n || n != 6 || map.overlap() != 0 || picks.iter().any(|a| a.window.len() > 2) || picks.first().is_none_or(|a| !a.window.contains(&0)) {
        return Err(format!("nmapi.coordinator: team windows {:?}", picks.iter().map(|a| &a.window).collect::<Vec<_>>()));
    }
    // Meaning: after one window, a piece of a cluster already taken counts half; none counts whole.
    let mut team = Team::new(vec![1.0; 6], &tk, &graph, &ps, 20, 0);
    team.meaning = vec![0, 0, 1, 1, u32::MAX, 2];
    let mut map = nmbrain::allocation::AllocationMap::new(6, 1);
    let _ = team.assign(&mut map, &[0], &[Aim::Cover(0)], &|_, _| true, &|_| f32::NEG_INFINITY);
    // The window takes pieces 0 and 2 (clusters 0 and 1).
    if (team.b(1) - 0.5).abs() > 1e-6 || (team.b(3) - 0.5).abs() > 1e-6 || (team.b(4) - 1.0).abs() > 1e-6 || (team.b(5) - 1.0).abs() > 1e-6 {
        return Err(format!("nmapi.coordinator: meaning spread {} {} {} {}", team.b(1), team.b(3), team.b(4), team.b(5)));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_aims_reach_sibling_files() {
        let embed: nmsubs::Embed = std::sync::Arc::new(|_: &str| vec![1.0]);
        let body = |n: &str| format!("pub fn {n}(x: usize) -> usize {{\n    x + 1\n}}\n\npub fn {n}_more(x: usize) -> usize {{\n    x + 2\n}}\n");
        let (one, two, far) = (body("one"), body("two"), body("far"));
        let sub = nmsubs::from_files(&[("a/Cargo.toml", "[package]\nname = \"a\"\n"), ("a/src/one.rs", &one), ("a/src/two.rs", &two), ("b/src/far.rs", &far)], embed);
        let tree = tree_for(&sub);
        // Cached per version: the same tree comes back.
        assert!(std::sync::Arc::ptr_eq(&tree, &tree_for(&sub)));
        let path = |c: usize| sub.chunks.get(c).and_then(|x| sub.files.get(x.file)).map(|f| f.path.clone()).unwrap_or_default();
        let found: Vec<usize> = (0..sub.chunks.len()).filter(|c| path(*c) == "a/src/one.rs").collect();
        assert!(!found.is_empty());
        let aims = tree_aims(&sub, &tree, &found, 2);
        assert!(aims.iter().all(|c| !found.contains(c)), "{aims:?}");
        assert_eq!(aims.first().map(|c| path(*c)).as_deref(), Some("a/src/two.rs"), "{aims:?}\n{}", tree.outline());
        // A piece `free` refuses is never seeded.
        let blocked = aims.first().copied();
        let seeds = tree_pieces(&sub, &tree, &found, 2, &|c| Some(c) != blocked, &|_| 0.0);
        assert!(seeds.iter().all(|x| Some(x.0) != blocked && x.1 > 0.0));
        // Off by default.
        if !tree_prior_on() {
            assert!(tree_seeds(&sub, &found, 2, &|_| true, &|_| 0.0).is_empty());
        }
    }
}
