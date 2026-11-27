//! Lenses (E3.2, 2026-11-09; design `docs/2026-11-10-neumor-design-e3-tentacles.md`, Luis Jake
//! Gabriel III): what each tentacle's full window holds. Every lens chooses from the whole tree
//! by its own reading of the question; no lens sees another lens's choice. `assign` then gives
//! each chunk to one window only (no crossover, Luis Jake Gabriel III 2026-11-09): when lenses want the same
//! chunk, the lens ranking it higher takes it and the other goes on to its next pick, so the
//! windows hold different code and context is expanded, never doubled.
use crate::Substrate;
use crate::refgraph::{EdgeKind, Specific};
use std::collections::{BTreeMap, BTreeSet};

/// The lenses, one per tentacle, in tie order.
pub const NAMES: [&str; 8] = ["meaning", "words", "names", "structure", "history", "crate", "files", "memory"];
/// Chunk tokens per tentacle window (Luis Jake Gabriel III, 2026-11-09).
pub const BUDGET: usize = 12_000;
/// How deep each lens's list is read.
const LIST: usize = 600;
/// Symbols the structure lens follows.
const SYMBOLS: usize = 20;
/// Recently changed files the history lens reads.
const RECENT: usize = 40;
/// Files the files lens reads whole.
const FILES: usize = 12;

/// Rank of every chunk in `order` (absent: `usize::MAX`).
fn rank_of(order: &[usize], n: usize) -> Vec<usize> {
    let mut r = vec![usize::MAX; n];
    for (p, &i) in order.iter().enumerate() {
        if let Some(x) = r.get_mut(i) {
            *x = (*x).min(p);
        }
    }
    r
}

/// Chunks of file `path` holding line `line`.
fn chunks_at<'a>(sub: &'a Substrate, path: &'a str, line: usize) -> impl Iterator<Item = usize> + 'a {
    sub.chunks
        .iter()
        .enumerate()
        .filter(move |(_, c)| c.start <= line && c.end >= line && sub.files.get(c.file).is_some_and(|f| f.path == path))
        .map(|(i, _)| i)
}

/// Every lens's list for `question`, best first, in `NAMES` order. `embedding` is the dense
/// lane's embedding order of the whole tree (`DenseLane::embedding_order`), when the lane runs;
/// `recalled` are chunks memory chose for this question (empty when memory is off).
#[must_use]
pub fn orders(sub: &Substrate, question: &str, embedding: Option<&[usize]>, recalled: &[usize]) -> Vec<Vec<usize>> {
    let n = sub.chunks.len();
    let lexical = sub.lexical_order(question);
    let lex_rank = rank_of(&lexical, n);
    let emb_rank = embedding.map_or_else(|| vec![usize::MAX; n], |e| rank_of(e, n));
    let by_lex = |v: &mut Vec<usize>| v.sort_by_key(|i| (lex_rank[*i], *i));
    let dedup = |v: Vec<usize>| -> Vec<usize> {
        let mut seen = BTreeSet::new();
        v.into_iter().filter(|i| seen.insert(*i)).take(LIST).collect()
    };

    // Names: definitions of the identifiers the question names, then chunks that hold them.
    let names: Vec<String> = nmlex::names(question);
    let mut defs = Vec::new();
    let mut uses = Vec::new();
    for name in &names {
        for s in sub.symbols.iter().filter(|s| s.name.eq_ignore_ascii_case(name)) {
            defs.extend(chunks_at(sub, &s.file, s.line));
        }
        let low = name.to_lowercase();
        uses.extend(sub.chunks.iter().enumerate().filter(|(_, c)| c.low.contains(&low)).map(|(i, _)| i));
    }
    by_lex(&mut uses);
    let names_list = dedup(defs.into_iter().chain(uses).collect());

    // Structure: symbols whose names carry the question's rarer key terms, their definitions,
    // then the chunks that use them.
    let terms: Vec<(String, f32)> = nmlex::key_terms(question)
        .into_iter()
        .filter(|t| t.len() >= 4)
        .map(|t| {
            let w = sub.idf(&t);
            (t, w)
        })
        .collect();
    let mut syms: Vec<(f32, usize)> = sub
        .symbols
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            let low = s.name.to_lowercase();
            let w: f32 = terms.iter().filter(|(t, _)| low.contains(t.as_str())).map(|(_, w)| w).sum();
            (w > 0.0).then_some((w, i))
        })
        .collect();
    syms.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let picked: Vec<usize> = syms.iter().take(SYMBOLS).map(|x| x.1).collect();
    let mut s_defs = Vec::new();
    let mut s_uses: Vec<(usize, usize, usize)> = Vec::new();
    for (k, &si) in picked.iter().enumerate() {
        let s = &sub.symbols[si];
        s_defs.extend(chunks_at(sub, &s.file, s.line));
        let low = s.name.to_lowercase();
        s_uses.extend(sub.chunks.iter().enumerate().filter(|(_, c)| c.low.contains(&low)).map(|(i, _)| (k, lex_rank[i], i)));
    }
    s_uses.sort_unstable();
    let structure = dedup(s_defs.into_iter().chain(s_uses.into_iter().map(|x| x.2)).collect());

    // History: the most recently changed files (working-tree changes first, then by the
    // newest commit's date), their chunks in lexical order.
    let mut recent: Vec<(bool, String, &str)> = sub
        .history
        .iter()
        .map(|(path, log)| {
            let working = log.iter().any(|l| l.starts_with("- "));
            let date = log.iter().find(|l| !l.starts_with("- ")).and_then(|l| l.split_whitespace().nth(1)).unwrap_or("").to_owned();
            (!working, date, path.as_str())
        })
        .collect();
    recent.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)).then(a.2.cmp(b.2)));
    let recent_files: BTreeSet<&str> = recent.iter().take(RECENT).map(|x| x.2).collect();
    let mut history: Vec<usize> = sub
        .chunks
        .iter()
        .enumerate()
        .filter(|(_, c)| sub.files.get(c.file).is_some_and(|f| recent_files.contains(f.path.as_str())))
        .map(|(i, _)| i)
        .collect();
    by_lex(&mut history);
    let history = dedup(history);

    // Crate and files: where the question's best chunks gather, read in line order.
    #[expect(clippy::cast_precision_loss, reason = "ranks are small")]
    let f = |r: usize| if r == usize::MAX { 0.0 } else { 1.0 / (60.0 + r as f32) };
    let fused = |i: usize| -> f32 { f(lex_rank[i]) + f(emb_rank[i]) };
    let top: Vec<usize> = {
        let mut v: Vec<usize> = (0..n).filter(|i| lex_rank[*i] < 50 || emb_rank[*i] < 50).collect();
        v.sort_by(|a, b| fused(*b).total_cmp(&fused(*a)).then(a.cmp(b)));
        v
    };
    let path = |i: usize| sub.files.get(sub.chunks[i].file).map_or("", |f| f.path.as_str());
    let mut per_crate: BTreeMap<&str, f32> = BTreeMap::new();
    for &i in &top {
        if let Some(c) = sub.crates.iter().find(|c| !c.dir.is_empty() && path(i).starts_with(c.dir.as_str())) {
            *per_crate.entry(c.dir.as_str()).or_default() += fused(i);
        }
    }
    let mut crates: Vec<(&str, f32)> = per_crate.into_iter().collect();
    crates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(b.0)));
    let in_line_order = |keep: &dyn Fn(&str) -> bool, key: &dyn Fn(&str) -> usize| -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).filter(|i| keep(path(*i))).collect();
        v.sort_by(|a, b| (key(path(*a)), path(*a), sub.chunks[*a].start).cmp(&(key(path(*b)), path(*b), sub.chunks[*b].start)));
        v
    };
    let crate_dirs: Vec<&str> = crates.iter().take(2).map(|x| x.0).collect();
    let crate_list = dedup(in_line_order(&|p| crate_dirs.iter().any(|d| p.starts_with(d)), &|p| {
        crate_dirs.iter().position(|d| p.starts_with(d)).unwrap_or(usize::MAX)
    }));
    let mut files: Vec<&str> = Vec::new();
    for &i in &top {
        if !files.contains(&path(i)) && files.len() < FILES {
            files.push(path(i));
        }
    }
    let files_list = dedup(in_line_order(&|p| files.contains(&p), &|p| files.iter().position(|f| *f == p).unwrap_or(usize::MAX)));

    let meaning = embedding.map_or_else(Vec::new, |e| e.iter().copied().take(LIST).collect());
    vec![meaning, dedup(lexical), names_list, structure, history, crate_list, files_list, dedup(recalled.to_vec())]
}

/// Give each chunk to one window (no crossover): every (rank, lens, chunk) in rank order, ties
/// by lens order; a chunk goes to the first lens that wants it and still has room for it. Each
/// window holds at most `budget` tokens (`tokens(chunk)`).
#[must_use]
pub fn assign(lists: &[Vec<usize>], tokens: &dyn Fn(usize) -> usize, budget: usize) -> Vec<Vec<usize>> {
    let mut wants: Vec<(usize, usize, usize)> =
        lists.iter().enumerate().flat_map(|(l, v)| v.iter().enumerate().map(move |(r, &c)| (r, l, c))).collect();
    wants.sort_unstable();
    let mut taken = BTreeSet::new();
    let mut used = vec![0_usize; lists.len()];
    let mut out = vec![Vec::new(); lists.len()];
    for (_, l, c) in wants {
        let t = tokens(c);
        if taken.contains(&c) || used[l] + t > budget {
            continue;
        }
        taken.insert(c);
        used[l] += t;
        out[l].push(c);
    }
    out
}

/// Fallback fill of empty lens lists (2026-11-10, refinement step 1, item 5; architecture by Luis
/// Jake Gabriel III; context is never reduced, direction of 2026-11-10 08:04 UTC). A lens with no
/// list (the names lens when the question names no identifier, the memory lens with nothing
/// recalled) would leave its tentacle's window empty. Its fallback list is the other lenses'
/// lists read in lens order, rank by rank (rank 0 of every lens in `NAMES` order, then rank 1,
/// and so on), each chunk once. Under `assign` a chunk goes to the lens ranking it higher, and a
/// fallback rank is always behind the chunk's own rank in the lens that listed it, so the filled
/// window takes what the other windows could not hold: their next picks beyond their budgets.
/// Returns the lists and the indices of the lenses that were filled.
#[must_use]
pub fn fill_empty(lists: &[Vec<usize>]) -> (Vec<Vec<usize>>, Vec<usize>) {
    let empty: Vec<usize> = (0..lists.len()).filter(|l| lists.get(*l).is_some_and(Vec::is_empty)).collect();
    if empty.is_empty() || empty.len() == lists.len() {
        return (lists.to_vec(), Vec::new());
    }
    let depth = lists.iter().map(Vec::len).max().unwrap_or(0);
    let mut seen = BTreeSet::new();
    let mut fallback = Vec::new();
    for r in 0..depth {
        for l in lists {
            if let Some(&c) = l.get(r)
                && seen.insert(c)
            {
                fallback.push(c);
            }
        }
    }
    let mut out = lists.to_vec();
    for &l in &empty {
        if let Some(x) = out.get_mut(l) {
            x.clone_from(&fallback);
        }
    }
    (out, empty)
}

/// `orders` with the empty lists filled (`fill_empty`): every window carries context.
#[must_use]
pub fn orders_filled(sub: &Substrate, question: &str, embedding: Option<&[usize]>, recalled: &[usize]) -> (Vec<Vec<usize>>, Vec<usize>) {
    fill_empty(&orders(sub, question, embedding, recalled))
}

/// Name of the relation-guided window as a lens kind (2026-11-10, refinement step 4a, fault F28;
/// architecture by Luis Jake Gabriel III; analysis H section G). It is the second intake mode
/// beside the eight coverage lenses of `NAMES`: a tentacle given this kind takes a window
/// assembled by `relation_window`. No default uses it: `orders` and `assign` are unchanged, and
/// a caller selects it through `assign_with_relation`.
pub const RELATION: &str = "relation";
/// Decay of a relation's contribution per hop from its seed: one half, the form measured in
/// analysis H section 0.2 (priority halved per hop). A stated fallback: no measurement on this
/// tree yet separates other values.
pub const HOP_DECAY: f32 = 0.5;

/// When a relation walk stops following a seed's path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// Marginal gain: a seed's path ends when its last addition scored below the running
    /// average of every addition of the walk so far; the walk then goes on to the next seed,
    /// and when every path has ended the window is filled best first from all that is left
    /// (context is never reduced: the window stays a full window).
    MarginalGain,
    /// First form: best first over all seeds at once until the budget is full.
    BudgetOnly,
}

/// Weights of the edge kinds in a relation walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KindWeights {
    /// From the graph (`Specific::kind_weights`).
    Derived,
    /// Every kind counts one (first form).
    Uniform,
}

/// One chunk a relation walk took.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Step {
    pub chunk: usize,
    /// The chunk whose relation contributed most (itself for a seed taken into the window).
    pub from: usize,
    pub kind: EdgeKind,
    /// Hops from the seed (0 for a seed).
    pub hop: usize,
    /// Priority when taken.
    pub score: f32,
    /// Index of the seed whose path took it (`usize::MAX` in the best-first fill).
    pub seed: usize,
}

/// A window assembled by a relation walk.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RelationWindow {
    /// Chunks taken, in the order taken (the caller lays them out in file order).
    pub chunks: Vec<usize>,
    pub tokens: usize,
    pub steps: Vec<Step>,
    /// Neighbours reached and not taken (no room, or the walk ended first), with their
    /// priority, best first: kept as references, nothing is discarded.
    pub passed: Vec<(usize, f32)>,
    /// Seed paths ended by the marginal-gain rule, and chunks added by the best-first fill.
    pub paths_ended: usize,
    pub filled: usize,
    /// Mean priority of the additions (seeds taken excluded).
    pub mean_score: f32,
    /// Deepest hop taken and the mean hop of the additions.
    pub depth: usize,
    pub mean_hop: f32,
    /// Kind weights used (`EdgeKind::ALL` order).
    pub kind_weights: [f32; crate::refgraph::KINDS],
}
impl RelationWindow {
    /// Distinct relations followed: (from, to) pairs of the additions.
    #[must_use]
    pub fn relations(&self) -> usize {
        self.steps.iter().filter(|s| s.from != s.chunk).map(|s| (s.from, s.chunk)).collect::<BTreeSet<_>>().len()
    }
}

/// A frontier candidate of a relation walk.
#[derive(Clone, Copy)]
struct Cand {
    score: f32,
    best: f32,
    from: usize,
    kind: EdgeKind,
    hop: usize,
    seed: usize,
}

/// Assemble one window by a budgeted best-first walk over the specific relations (the second
/// intake mode). `seeds` are (chunk, relevance) pairs: the question's terms matched to items
/// (`question_seeds`) and spans already found. A seed that is not `blocked` and fits is taken
/// into the window; a blocked seed (held by another tentacle) is a source only. From every
/// source the walk scores its neighbours: relation weight x kind weight x the seed's relevance
/// (relative to the strongest seed) x `HOP_DECAY` per hop, summed over the sources reaching the
/// neighbour. Blocked chunks are never taken, so windows stay disjoint. The pieces may be far
/// apart in the files. `budget` is the window's tokens (a tentacle window, `BUDGET`); the walk
/// never passes it. Stopping: `Stop`. Everything reached and not taken is returned in `passed`.
#[must_use]
#[expect(clippy::too_many_arguments, reason = "one walk: seeds, paths, fill, record; each argument is one selectable part")]
pub fn relation_window(
    graph: &Specific,
    seeds: &[(usize, f32)],
    blocked: &dyn Fn(usize) -> bool,
    tokens: &dyn Fn(usize) -> usize,
    budget: usize,
    stop: Stop,
    weights: KindWeights,
) -> RelationWindow {
    relation_window_with(graph, seeds, blocked, tokens, budget, stop, weights, &|_| 0.0)
}

/// `relation_window` with a relevance prior on the pieces reached: a relation's contribution is
/// multiplied by one plus the reached piece's prior (0 to 1), so among the relations of a seed
/// the pieces the drones also found come first, and a piece the drones did not flag is still
/// reached through its relation.
#[must_use]
#[expect(
    clippy::too_many_lines,
    clippy::too_many_arguments,
    reason = "one walk: seeds, paths, fill, record; each argument is one selectable part"
)]
pub fn relation_window_with(
    graph: &Specific,
    seeds: &[(usize, f32)],
    blocked: &dyn Fn(usize) -> bool,
    tokens: &dyn Fn(usize) -> usize,
    budget: usize,
    stop: Stop,
    weights: KindWeights,
    prior: &dyn Fn(usize) -> f32,
) -> RelationWindow {
    let kw = if weights == KindWeights::Derived { graph.kind_weights() } else { [1.0; crate::refgraph::KINDS] };
    let mut out = RelationWindow { kind_weights: kw, ..RelationWindow::default() };
    let top = seeds.iter().map(|s| s.1).fold(0.0_f32, f32::max);
    // Seeds, strongest first, each once.
    let mut order: Vec<(usize, f32)> = Vec::new();
    for &(c, w) in seeds {
        if c < graph.len() && !order.iter().any(|x| x.0 == c) {
            order.push((c, if top > 0.0 { (w / top).max(0.0) } else { 1.0 }));
        }
    }
    order.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut taken: BTreeSet<usize> = BTreeSet::new();
    let mut sources: BTreeSet<usize> = BTreeSet::new();
    let mut frontier: BTreeMap<usize, Cand> = BTreeMap::new();
    // A source spreads priority to its neighbours.
    let spread = |frontier: &mut BTreeMap<usize, Cand>,
                  taken: &BTreeSet<usize>,
                  sources: &BTreeSet<usize>,
                  src: usize,
                  hop: usize,
                  seed: usize,
                  strength: f32| {
        #[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap, reason = "hop counts are small")]
        let decay = HOP_DECAY.powi(hop as i32);
        for e in graph.edges_of(src) {
            let to = e.to as usize;
            if taken.contains(&to) || sources.contains(&to) || blocked(to) {
                continue;
            }
            let c = e.weight * kw.get(e.kind.index()).copied().unwrap_or(1.0) * strength * decay * (1.0 + prior(to).clamp(0.0, 1.0));
            let x = frontier.entry(to).or_insert(Cand { score: 0.0, best: 0.0, from: src, kind: e.kind, hop: hop + 1, seed });
            x.score += c;
            if c > x.best {
                x.best = c;
                x.from = src;
                x.kind = e.kind;
                x.seed = seed;
            }
            x.hop = x.hop.min(hop + 1);
        }
    };
    let mut strength: Vec<f32> = Vec::new();
    for (k, &(c, w)) in order.iter().enumerate() {
        strength.push(w);
        sources.insert(c);
        frontier.remove(&c);
        let t = tokens(c);
        if !blocked(c) && out.tokens + t <= budget {
            taken.insert(c);
            out.tokens += t;
            out.chunks.push(c);
            out.steps.push(Step { chunk: c, from: c, kind: EdgeKind::Names, hop: 0, score: w, seed: k });
        }
    }
    for (k, &(c, w)) in order.iter().enumerate() {
        spread(&mut frontier, &taken, &sources, c, 0, k, w);
    }
    let (mut sum, mut count, mut hops) = (0.0_f32, 0_usize, 0_usize);
    // Take the best fitting candidate (of one seed's path, or of all).
    let mut take = |frontier: &mut BTreeMap<usize, Cand>,
                    taken: &mut BTreeSet<usize>,
                    out: &mut RelationWindow,
                    of_seed: Option<usize>|
     -> Option<f32> {
        let room = budget.saturating_sub(out.tokens);
        let (&c, &cand) = frontier
            .iter()
            .filter(|(c, x)| of_seed.is_none_or(|s| x.seed == s) && tokens(**c) <= room)
            .max_by(|a, b| a.1.score.total_cmp(&b.1.score).then(b.0.cmp(a.0)))?;
        frontier.remove(&c);
        taken.insert(c);
        out.tokens += tokens(c);
        out.chunks.push(c);
        out.steps.push(Step {
            chunk: c,
            from: cand.from,
            kind: cand.kind,
            hop: cand.hop,
            score: cand.score,
            seed: of_seed.unwrap_or(usize::MAX),
        });
        out.depth = out.depth.max(cand.hop);
        hops += cand.hop;
        let seed = if cand.seed < strength.len() { cand.seed } else { 0 };
        spread(frontier, taken, &sources, c, cand.hop, seed, strength.get(seed).copied().unwrap_or(1.0));
        Some(cand.score)
    };
    if stop == Stop::MarginalGain {
        for k in 0..order.len() {
            let mut on_path = 0_usize;
            while let Some(score) = take(&mut frontier, &mut taken, &mut out, Some(k)) {
                #[expect(clippy::cast_precision_loss, reason = "a small count")]
                let below = count > 0 && score < sum / count as f32;
                sum += score;
                count += 1;
                on_path += 1;
                if below && on_path > 1 {
                    out.paths_ended += 1;
                    break;
                }
            }
        }
    }
    while let Some(score) = take(&mut frontier, &mut taken, &mut out, None) {
        sum += score;
        count += 1;
        if stop == Stop::MarginalGain {
            out.filled += 1;
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    if count > 0 {
        out.mean_score = sum / count as f32;
        out.mean_hop = hops as f32 / count as f32;
    }
    out.passed = frontier.into_iter().map(|(c, x)| (c, x.score)).collect();
    out.passed.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

/// Seeds of a relation walk from the question alone: the chunks defining the items the question
/// names (`nmlex::names`, exact) and the items whose names carry its rarer key terms (the
/// structure lens's reading), each with the summed rarity of the terms matched; best first.
/// The count followed is the structure lens's `SYMBOLS`.
#[must_use]
pub fn question_seeds(sub: &Substrate, question: &str) -> Vec<(usize, f32)> {
    let terms: Vec<(String, f32)> = nmlex::key_terms(question)
        .into_iter()
        .filter(|t| t.len() >= 4)
        .map(|t| {
            let w = sub.idf(&t);
            (t, w)
        })
        .collect();
    let named: Vec<String> = nmlex::names(question);
    let top = terms.iter().map(|t| t.1).fold(0.0_f32, f32::max).max(1.0);
    let mut syms: Vec<(f32, usize)> = sub
        .symbols
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            let low = s.name.to_lowercase();
            let mut w: f32 = terms.iter().filter(|(t, _)| low.contains(t.as_str())).map(|(_, w)| w).sum();
            if named.iter().any(|n| n.eq_ignore_ascii_case(&s.name)) {
                w += top;
            }
            (w > 0.0).then_some((w, i))
        })
        .collect();
    syms.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut out: Vec<(usize, f32)> = Vec::new();
    for &(w, si) in syms.iter().take(SYMBOLS) {
        let Some(s) = sub.symbols.get(si) else { continue };
        for c in chunks_at(sub, &s.file, s.line) {
            match out.iter_mut().find(|x| x.0 == c) {
                Some(x) => x.1 = x.1.max(w),
                None => out.push((c, w)),
            }
        }
    }
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

/// `assign` with one tentacle on the relation kind: the other lenses are assigned as `assign`
/// does (the relation tentacle's own lens list is not used), then the relation tentacle's
/// window is walked from `seeds` with every chunk the others hold blocked, so no chunk is in
/// two windows. Returns the windows (the relation tentacle's in the order taken) and the walk.
#[must_use]
pub fn assign_with_relation(
    lists: &[Vec<usize>],
    relation: usize,
    graph: &Specific,
    seeds: &[(usize, f32)],
    tokens: &dyn Fn(usize) -> usize,
    budget: usize,
) -> (Vec<Vec<usize>>, RelationWindow) {
    let mut others = lists.to_vec();
    if let Some(l) = others.get_mut(relation) {
        l.clear();
    }
    let mut out = assign(&others, tokens, budget);
    let held: BTreeSet<usize> = out.iter().flatten().copied().collect();
    let walk = relation_window(graph, seeds, &|c| held.contains(&c), tokens, budget, Stop::MarginalGain, KindWeights::Derived);
    if let Some(w) = out.get_mut(relation) {
        w.clone_from(&walk.chunks);
    }
    (out, walk)
}

/// Self-check of `lens`: a chunk two lenses want goes to the one ranking it higher, the other
/// takes its next pick, and no window passes its budget.
pub(crate) fn check() -> Result<(), String> {
    let lists = vec![vec![1, 2, 3], vec![2, 1, 4]];
    let got = assign(&lists, &|_| 10, 20);
    if got != vec![vec![1, 3], vec![2, 4]] {
        return Err(format!("nmsubs.lens: assignment {got:?}"));
    }
    let got = assign(&lists, &|c| if c == 1 { 30 } else { 10 }, 20);
    if got[0].contains(&1) || got[1].contains(&1) {
        return Err("nmsubs.lens: a window passed its budget".into());
    }
    // Fallback fill: the empty list takes the others' next picks; nothing doubles across windows.
    let lists = vec![vec![1, 2, 3, 5], vec![], vec![2, 4, 6]];
    let (filled, which) = fill_empty(&lists);
    let got = assign(&filled, &|_| 10, 20);
    let all: Vec<usize> = got.iter().flatten().copied().collect();
    let mut uniq = all.clone();
    uniq.sort_unstable();
    uniq.dedup();
    if which != vec![1] || got.get(1).is_none_or(Vec::is_empty) || uniq.len() != all.len() || got.first() != Some(&vec![1, 3]) {
        return Err(format!("nmsubs.lens: fallback fill {filled:?} assigned {got:?}"));
    }
    relation_check()
}

/// Self-check of the relation-guided window: the walk follows specific relations to chunks far
/// apart in the files, never takes a blocked chunk, never passes its budget, keeps what it
/// passed over as references, and `assign_with_relation` leaves no chunk in two windows.
fn relation_check() -> Result<(), String> {
    let embed: crate::Embed = std::sync::Arc::new(|_: &str| vec![0.0]);
    let mut files: Vec<(String, String)> = vec![
        ("a/seed.rs".into(), "pub fn derive_window_limit() -> usize { pick_reader_span() }\n".into()),
        ("z/far.rs".into(), "pub fn pick_reader_span() -> usize { settle_span_floor() }\n".into()),
        ("m/deep.rs".into(), "pub fn settle_span_floor() -> usize { 4 }\n".into()),
        ("q/held.rs".into(), "pub fn held_elsewhere() -> usize { derive_window_limit() }\n".into()),
    ];
    for k in 0..6 {
        files.push((format!("c/other{k}.rs"), format!("pub fn unrelated_item_{k}() -> u8 {{ 0 }}\n")));
    }
    let borrowed: Vec<(&str, &str)> = files.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let sub = crate::from_files(&borrowed, embed);
    let g = sub.refs();
    let s = g.specific_with(&sub, crate::refgraph::Limit::Fixed(crate::refgraph::RARE_FALLBACK));
    let idx = |p: &str| sub.chunks.iter().position(|c| sub.files.get(c.file).is_some_and(|f| f.path == p)).unwrap_or(usize::MAX);
    let (seed, far, deep, held) = (idx("a/seed.rs"), idx("z/far.rs"), idx("m/deep.rs"), idx("q/held.rs"));
    for stop in [Stop::MarginalGain, Stop::BudgetOnly] {
        let w = relation_window(&s, &[(seed, 1.0)], &|c| c == held, &|_| 10, 30, stop, KindWeights::Derived);
        if w.chunks != vec![seed, far, deep] || w.tokens != 30 || w.depth != 2 || w.relations() != 2 || w.chunks.contains(&held) {
            return Err(format!("nmsubs.lens: relation walk ({stop:?}) took {:?}, {} tokens, depth {}", w.chunks, w.tokens, w.depth));
        }
        let small = relation_window(&s, &[(seed, 1.0)], &|c| c == held, &|_| 10, 20, stop, KindWeights::Uniform);
        if small.chunks != vec![seed, far] || !small.passed.iter().any(|p| p.0 == deep) {
            return Err(format!(
                "nmsubs.lens: relation walk ({stop:?}) at a smaller budget took {:?}, passed over {:?}",
                small.chunks, small.passed
            ));
        }
    }
    // A blocked seed is a source only.
    let w = relation_window(&s, &[(seed, 1.0)], &|c| c == seed, &|_| 10, 30, Stop::MarginalGain, KindWeights::Derived);
    if w.chunks.contains(&seed) || !w.chunks.contains(&far) {
        return Err(format!("nmsubs.lens: walk from a held seed took {:?}", w.chunks));
    }
    let qs = question_seeds(&sub, "where is derive_window_limit set");
    if qs.first().map(|x| x.0) != Some(seed) {
        return Err(format!("nmsubs.lens: question seeds {qs:?}"));
    }
    let lists = vec![vec![held, seed], vec![far], vec![deep, held]];
    let (windows, walk) = assign_with_relation(&lists, 1, &s, &[(seed, 1.0)], &|_| 10, 30);
    let all: Vec<usize> = windows.iter().flatten().copied().collect();
    let uniq: BTreeSet<usize> = all.iter().copied().collect();
    if uniq.len() != all.len() || windows.get(1) != Some(&vec![far]) || walk.chunks != vec![far] {
        return Err(format!("nmsubs.lens: relation tentacle among lenses {windows:?}"));
    }
    Ok(())
}
