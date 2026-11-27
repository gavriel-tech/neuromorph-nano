//! Specific relations, the relation-guided window and the allocation rule by replay of a scan
//! dump (2026-11-10, refinement step 4a, faults F27 and F28). Gavriel Technologies; architecture
//! by Luis Jake Gabriel III.
//!
//! Processor only, no model weights: the saved scans under `scratch/tc1-dump/` give each
//! question's windows and reading scores; chunk sizes come from the base tokenizer file (a
//! vocabulary, not weights). Nothing is read on the card, so every figure here is about which
//! code a window holds, never about how well it is read. Question text is held in memory (the
//! seeds and the lens lists need it) and never written: the record names questions by index.
//! The sealed claim half is never read (only even positions of the sorted question files, the
//! development half, as the locked harness orders it).
//!
//! Run: `cargo test --release -p nmapi --test allocation_replay -- --nocapture`.
//! Writes `runs/A4-allocation-replay-<ms>.txt`.
use nmapi::scan_dump;
use nmapi::tentacle_pass::{Scan, Score, WindowScan};
use nmbrain::allocation::{self, AllocationMap, Mode, Target};
use nmsubs::lens::{self, KindWeights, RelationWindow, Stop};
use nmsubs::refgraph::{Limit, RARE_FALLBACK, Specific};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The dump the analysis measured; the newest dump when it is absent.
const DUMP: &str = "1794288995065";

/// The development half: (id, question text) by index, as the locked harness orders it.
fn dev_half(home: &Path) -> Vec<(String, String)> {
    const TYPES: [&str; 5] = ["behaviour", "name", "value", "explain", "relation"];
    let mut out = Vec::new();
    for area in ["context", "engine", "tools"] {
        let p = home.join(format!("tests/capability-gates/cx1-context/questions-{area}.json"));
        let v = nmjson::parse(&std::fs::read(&p).expect("question file")).expect("json");
        let mut qs: Vec<(usize, String, String)> = v
            .get("questions")
            .and_then(|y| y.array())
            .expect("questions")
            .iter()
            .map(|x| {
                let s = |k: &str| x.get(k).and_then(|y| y.string()).expect("field").to_owned();
                let t = s("type");
                (TYPES.iter().position(|y| *y == t).expect("type"), s("id"), s("q"))
            })
            .collect();
        qs.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
        out.extend(qs.into_iter().enumerate().filter(|(pos, _)| pos % 2 == 0).map(|(_, (_, id, q))| (id, q)));
    }
    out
}

fn dump_dir(home: &Path) -> Option<PathBuf> {
    let root = home.join("scratch").join("tc1-dump");
    let named = root.join(DUMP);
    if named.join("substrate.json").is_file() {
        return Some(named);
    }
    let mut v: Vec<PathBuf> =
        std::fs::read_dir(root).ok()?.filter_map(Result::ok).map(|e| e.path()).filter(|p| p.join("substrate.json").is_file()).collect();
    v.sort();
    v.pop()
}

/// A window's yield: its best chunk's calibrated score above the anchor's (the same meaning in
/// every tentacle), or its best mass when the scan recorded no raw scores.
fn window_yield(w: &WindowScan) -> f32 {
    let cal = w.calibrated_parts();
    match cal.first() {
        Some(a) => cal.iter().skip(1).map(|x| x - a).fold(f32::NEG_INFINITY, f32::max),
        None => w.scores(Score::Mass).into_iter().fold(f32::NEG_INFINITY, f32::max),
    }
}

/// A window's chunks, best first by mass (the reading statistic the served pass ranks by).
fn finds(w: &WindowScan) -> Vec<(usize, f32)> {
    w.ranked(Score::Mass).into_iter().filter_map(|(k, s)| w.place.get(k).map(|c| (*c, s))).collect()
}

/// Sums over questions for one arm.
#[derive(Clone, Copy, Debug, Default)]
struct Arm {
    tokens: usize,
    chunks: usize,
    files: usize,
    share: f64,
    reach: usize,
    windows: usize,
    n: usize,
}
impl Arm {
    fn add(&mut self, sub: &nmsubs::Substrate, windows: &[Vec<usize>], tk: &dyn Fn(usize) -> usize, tree: usize, gold: &[usize]) {
        let all: BTreeSet<usize> = windows.iter().flatten().copied().collect();
        let tokens: usize = all.iter().map(|c| tk(*c)).sum();
        self.tokens += tokens;
        self.chunks += all.len();
        self.files += all.iter().filter_map(|c| sub.chunks.get(*c).map(|x| x.file)).collect::<BTreeSet<_>>().len();
        #[expect(clippy::cast_precision_loss, reason = "token counts are far below 2^52")]
        let share = tokens as f64 / tree.max(1) as f64;
        self.share += share;
        self.reach += usize::from(gold.iter().any(|g| all.contains(g)));
        self.windows += windows.iter().filter(|w| !w.is_empty()).count();
        self.n += 1;
    }
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    fn text(&self) -> String {
        let n = self.n.max(1) as f64;
        format!(
            "windows {:.1}, chunks {:.1}, tokens {:.0}, distinct files {:.1}, share of the tree {:.2}%, right code in some window {} of {}",
            self.windows as f64 / n,
            self.chunks as f64 / n,
            self.tokens as f64 / n,
            self.files as f64 / n,
            100.0 * self.share / n,
            self.reach,
            self.n
        )
    }
}

/// Chunks in more than one window.
fn overlap(windows: &[Vec<usize>]) -> usize {
    let mut seen = BTreeSet::new();
    windows.iter().flatten().filter(|c| !seen.insert(**c)).count()
}

fn median(v: &mut [usize]) -> usize {
    v.sort_unstable();
    v.get(v.len() / 2).copied().unwrap_or(0)
}

/// 1-based rank of the best right chunk by `score` among the chunks that are not seeds; none
/// when no right chunk has a score above zero (the relations do not reach it).
fn rank_of_gold(score: &[f32], seeds: &BTreeSet<usize>, gold: &[usize]) -> Option<usize> {
    let best = gold.iter().filter(|g| !seeds.contains(g)).filter_map(|g| score.get(*g).copied()).fold(0.0_f32, f32::max);
    (best > 0.0).then(|| 1 + score.iter().enumerate().filter(|(i, s)| !seeds.contains(i) && **s > best).count())
}

/// One relation walk for a tentacle, applied to the map: every chunk taken is claimed (a claim
/// cannot fail: the walk is blocked on what the map does not free), the path and the passed-over
/// references are recorded.
fn walk_into(
    map: &mut AllocationMap,
    graph: &Specific,
    tentacle: usize,
    seeds: &[(usize, f32)],
    tk: &dyn Fn(usize) -> usize,
) -> RelationWindow {
    let walk = {
        let m: &AllocationMap = map;
        lens::relation_window(
            graph,
            seeds,
            &|c| !m.free_for(tentacle, c, false),
            tk,
            lens::BUDGET,
            Stop::MarginalGain,
            KindWeights::Derived,
        )
    };
    for s in &walk.steps {
        assert!(
            map.claim(tentacle, s.chunk, Mode::Relation, seeds.get(s.seed).map(|x| x.0), tk(s.chunk), false),
            "a walked chunk was not free"
        );
        if s.from != s.chunk {
            map.follow(tentacle, s.from, s.chunk);
        }
    }
    for &(c, p) in &walk.passed {
        map.pass_over(c, p);
    }
    map.walked(walk.depth);
    map.label(tentacle, lens::RELATION, &seeds.iter().map(|s| s.0).collect::<Vec<_>>());
    walk
}

#[test]
#[ignore = "needs a scan dump under scratch/tc1-dump/ (nano-lab run tc1 dump)"]
#[expect(clippy::too_many_lines, reason = "one replay study with its report")]
fn allocation_by_replay() {
    let home = nmtype::nano_root();
    // Part checks first (seconds): the three parts this step built.
    for (name, check) in nmsubs::checks()
        .into_iter()
        .filter(|c| c.0 == "nmsubs.refgraph" || c.0 == "nmsubs.lens")
        .chain(nmbrain::checks().into_iter().filter(|c| c.0 == "nmbrain.allocation"))
    {
        assert_eq!(check(), Ok(()), "{name}");
    }
    let Some(dir) = dump_dir(&home) else {
        panic!("needs a scan dump under scratch/tc1-dump/ (write one with `nano-lab run tc1 dump`); none found");
    };
    let started = Instant::now();
    let items = dev_half(&home);
    let sub = scan_dump::load_substrate(&dir).expect("substrate");
    let mut qs: Vec<(usize, scan_dump::QuestionDump, Scan)> = Vec::new();
    for idx in scan_dump::indices(&dir) {
        let Some((id, text)) = items.get(idx) else { continue };
        let (q, scan) = scan_dump::load_question(&dir, idx, text).expect("question");
        assert_eq!(&q.id, id, "dump question {idx} does not match the development half");
        qs.push((idx, q, scan));
    }
    let n = qs.len();
    assert!(n > 0, "empty dump");
    let nc = sub.chunks.len();
    // Chunk sizes: the dump's own counts where a window held the chunk, the base tokenizer for
    // the rest (the tokenizer file is a vocabulary; no weights are loaded).
    let mut tokens: Vec<usize> = vec![0; nc];
    let mut known = 0_usize;
    for (_, _, scan) in &qs {
        for w in &scan.windows {
            for (k, &c) in w.place.iter().enumerate() {
                if let (Some(x), Some(p)) = (tokens.get_mut(c), w.parts.get(k + 1))
                    && *x == 0
                {
                    *x = p.1;
                    known += 1;
                }
            }
        }
    }
    let t0 = Instant::now();
    let tok = nmengn::tokenizer::Tokenizer::open(&nmtype::models_dir(&home).join("qwen3-0.6b").join("files").join("tokenizer.json"));
    let (known_tokens, known_chars): (usize, usize) = (0..nc)
        .filter(|c| tokens.get(*c).is_some_and(|t| *t > 0))
        .fold((0, 0), |a, c| (a.0 + tokens.get(c).copied().unwrap_or(0), a.1 + sub.chunks.get(c).map_or(0, |x| x.text.len())));
    let size_note = match &tok {
        Ok(tok) => {
            for (c, x) in tokens.iter_mut().enumerate() {
                if *x == 0 {
                    *x = sub
                        .chunks
                        .get(c)
                        .and_then(|ch| {
                            sub.files.get(ch.file).map(|f| tok.encode(&nmsubs::dense::doc_text(&f.path, &ch.text)).map_or(0, |v| v.len()))
                        })
                        .unwrap_or(0);
                }
            }
            format!(
                "chunk sizes: {known} chunks from the dump's windows, {} by the base tokenizer file ({:.1} s)",
                nc - known,
                t0.elapsed().as_secs_f64()
            )
        }
        Err(_) => {
            // Stated fallback: tokens per character measured on the chunks the dump sized.
            #[expect(clippy::cast_precision_loss, reason = "sizes are far below 2^52")]
            let ratio = known_tokens as f64 / known_chars.max(1) as f64;
            for (c, x) in tokens.iter_mut().enumerate() {
                if *x == 0 {
                    #[expect(
                        clippy::cast_precision_loss,
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "a chunk size"
                    )]
                    let est = (sub.chunks.get(c).map_or(0, |ch| ch.text.len()) as f64 * ratio).ceil() as usize;
                    *x = est;
                }
            }
            format!(
                "chunk sizes: {known} chunks from the dump's windows, the rest estimated at {ratio:.3} tokens per character (tokenizer file not readable)"
            )
        }
    };
    let tree: usize = tokens.iter().sum();
    let tk = |c: usize| tokens.get(c).copied().unwrap_or(usize::MAX);
    let file_of = |c: usize| sub.chunks.get(c).map_or(usize::MAX, |x| x.file);

    let mut rep = String::new();
    let _ = writeln!(
        rep,
        "A4 allocation replay (refinement step 4a, faults F27 and F28): {n} development questions, dump {}, processor only, no model weights loaded",
        dir.file_name().and_then(|x| x.to_str()).unwrap_or("")
    );
    let _ = writeln!(rep, "part checks passed: nmsubs.refgraph, nmsubs.lens, nmbrain.allocation");
    let _ = writeln!(rep, "tree: {} files, {nc} chunks, {} symbols, {tree} tokens; {size_note}", sub.files.len(), sub.symbols.len());
    let _ = writeln!(
        rep,
        "window budget {} tokens (a tentacle window); hop decay {} (stated fallback, analysis H 0.2); walk reach fallback {} hops",
        lens::BUDGET,
        lens::HOP_DECAY,
        allocation::REACH_FALLBACK
    );

    // 1. Specific relations: edges and degrees before and after.
    let g = sub.refs();
    let mut d0: Vec<usize> = (0..nc).map(|i| g.degree(i)).collect();
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    let mean0 = d0.iter().sum::<usize>() as f64 / nc.max(1) as f64;
    let max0 = d0.iter().copied().max().unwrap_or(0);
    let _ = writeln!(rep, "\n1. SPECIFIC RELATIONS (F27)");
    let _ = writeln!(
        rep,
        "before (first form, any shared item name): {} item names, {} chunk pairs ({} directed), degree mean {mean0:.1} median {} largest {max0}",
        g.names,
        g.pairs(),
        g.edges,
        median(&mut d0)
    );
    let derived = &*g.specific;
    let fixed10 = g.specific_with(&sub, Limit::Fixed(RARE_FALLBACK));
    let fixed30 = g.specific_with(&sub, Limit::Fixed(30));
    let unlimited = g.specific_with(&sub, Limit::Unlimited);
    let _ = writeln!(rep, "after, limit derived from the tree: {}", derived.describe());
    let _ = writeln!(rep, "after, stated fallback limit {RARE_FALLBACK}: {}", fixed10.describe());
    let _ = writeln!(rep, "comparison, limit 30: {}", fixed30.describe());
    let _ = writeln!(rep, "comparison, no limit (specificity weights only): {}", unlimited.describe());
    let kw = derived.kind_weights();
    let _ = writeln!(
        rep,
        "edge-kind weights derived from the graph (1 + ln(all / of the kind) / ln(all); names 1): {}",
        nmsubs::refgraph::EdgeKind::ALL
            .iter()
            .map(|k| format!("{} {:.2}", k.name(), kw.get(k.index()).copied().unwrap_or(1.0)))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Per question preparation.
    struct Prep {
        windows: Vec<Vec<usize>>,
        tentacles: Vec<usize>,
        yields: Vec<f32>,
        found: Vec<(usize, f32)>,
        qseeds: Vec<(usize, f32)>,
    }
    let preps: Vec<Prep> = qs
        .iter()
        .map(|(idx, _, scan)| {
            let text = items.get(*idx).map_or("", |x| x.1.as_str());
            let top_q = lens::question_seeds(&sub, text);
            let top = top_q.iter().map(|x| x.1).fold(0.0_f32, f32::max).max(f32::MIN_POSITIVE);
            Prep {
                windows: scan.windows.iter().map(|w| w.place.clone()).collect(),
                tentacles: scan.windows.iter().map(|w| w.tentacle).collect(),
                yields: scan.windows.iter().map(window_yield).collect(),
                found: scan.windows.iter().filter_map(|w| finds(w).first().map(|f| (f.0, 1.0))).collect(),
                qseeds: top_q.into_iter().map(|(c, w)| (c, w / top)).collect(),
            }
        })
        .collect();

    // Rank of the right chunk by relation score, before and after (seeds: each window's first
    // chunk by mass, as the analysis; the right chunk ranked among the chunks that are not seeds).
    let _ = writeln!(rep, "rank of the right chunk by relation score (personalised rank from the windows' first chunks):");
    type ScoreOf<'a> = dyn Fn(&[(usize, f32)]) -> Vec<f32> + 'a;
    let mut rank_line = |name: &str, score_of: &ScoreOf<'_>| {
        let (mut ranks, mut top25, mut unreached, mut asked) = (Vec::new(), 0, 0, 0);
        for ((_, q, _), p) in qs.iter().zip(&preps) {
            let seeds: BTreeSet<usize> = p.found.iter().map(|s| s.0).collect();
            if q.gold.iter().all(|g| seeds.contains(g)) {
                continue;
            }
            asked += 1;
            match rank_of_gold(&score_of(&p.found), &seeds, &q.gold) {
                Some(r) => {
                    ranks.push(r);
                    top25 += usize::from(r <= 25);
                }
                None => {
                    ranks.push(nc);
                    unreached += 1;
                }
            }
        }
        let _ = writeln!(
            rep,
            "  {name}: median rank {} over the {asked} questions whose right chunk is not itself a seed, in the first 25 on {top25}, not reached by any relation on {unreached}",
            median(&mut ranks)
        );
    };
    rank_line("before, first form", &|s| g.rank(s).0);
    rank_line("after, derived limit", &|s| derived.rank(s).0);
    rank_line("after, fallback limit 10", &|s| fixed10.rank(s).0);
    rank_line("comparison, no limit", &|s| unlimited.rank(s).0);

    // 2. Relation-guided window, as in the analysis: one walk window from the windows' first
    // chunks, beside the seeds (dense measure: seeds plus the one window).
    let _ = writeln!(rep, "\n2. RELATION-GUIDED WINDOW (second intake mode)");
    let _ = writeln!(
        rep,
        "dense measure (the windows' first chunks plus one relation window walked from them, nothing else): right code in the seeds / in the window / in either, mean tokens of seeds plus window, mean chunks and files of the window, relations followed, hop depth"
    );
    let seeds_hold = qs.iter().zip(&preps).filter(|((_, q, _), p)| p.found.iter().any(|s| q.gold.contains(&s.0))).count();
    for (gname, graph) in [("derived limit", derived), ("fallback limit 10", &fixed10), ("limit 30", &fixed30), ("no limit", &unlimited)] {
        for (wname, stop, weights, with_q) in [
            ("first form: best first to the budget, uniform kinds", Stop::BudgetOnly, KindWeights::Uniform, false),
            ("marginal-gain paths, derived kinds", Stop::MarginalGain, KindWeights::Derived, false),
            ("marginal-gain paths, derived kinds, question seeds added", Stop::MarginalGain, KindWeights::Derived, true),
        ] {
            let (mut in_w, mut either, mut toks, mut chunks, mut files, mut rels, mut depth, mut passed) = (0, 0, 0, 0, 0, 0, 0, 0);
            for ((_, q, _), p) in qs.iter().zip(&preps) {
                let mut seeds = p.found.clone();
                if with_q {
                    seeds.extend(p.qseeds.iter().copied());
                }
                let held: BTreeSet<usize> = p.found.iter().map(|s| s.0).collect();
                let w = lens::relation_window(graph, &seeds, &|c| held.contains(&c), &tk, lens::BUDGET, stop, weights);
                let hit = q.gold.iter().any(|x| w.chunks.contains(x));
                in_w += usize::from(hit);
                either += usize::from(hit || p.found.iter().any(|s| q.gold.contains(&s.0)));
                toks += w.tokens + held.iter().map(|c| tk(*c)).sum::<usize>();
                chunks += w.chunks.len();
                files += w.chunks.iter().map(|c| file_of(*c)).collect::<BTreeSet<_>>().len();
                rels += w.relations();
                depth += w.depth;
                passed += w.passed.len();
            }
            #[expect(clippy::cast_precision_loss, reason = "small counts")]
            let per = |x: usize| x as f64 / n as f64;
            let _ = writeln!(
                rep,
                "  {gname}, {wname}: {seeds_hold} / {in_w} / {either} of {n}; tokens {:.0}; window chunks {:.1}, files {:.1}; relations {:.1}; depth {:.1}; passed over and kept as references {:.1}",
                per(toks),
                per(chunks),
                per(files),
                per(rels),
                per(depth),
                per(passed)
            );
        }
    }

    // 3. The three arms.
    let (mut a, mut b_switch, mut b_added, mut c_two, mut c_round2) =
        (Arm::default(), Arm::default(), Arm::default(), Arm::default(), Arm::default());
    let (mut b_overlap, mut b_rel, mut b_tokens, mut b_window_hit, mut b_lost, mut b_free) = (0, 0, 0, 0, 0, 0);
    let (
        mut c_overlap,
        mut c_rel,
        mut c_relation_orders,
        mut c_coverage_orders,
        mut c_moved,
        mut c_orders,
        mut c_refused,
        mut c_passed,
        mut c_short,
        mut c_new_hit,
    ) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    let mut per_q = String::new();
    for ((idx, q, _), p) in qs.iter().zip(&preps) {
        let text = items.get(*idx).map_or("", |x| x.1.as_str());
        // (a) today's windows.
        a.add(&sub, &p.windows, &tk, tree, &q.gold);
        assert_eq!(overlap(&p.windows), 0, "today's windows overlap on question {idx}");
        // (b) one tentacle switched to the relation-guided window: the window that yielded
        // least; the others unchanged. Seeds: the other windows' first chunks and the question's
        // items. The others' chunks are blocked, so nothing is in two windows.
        let low = (0..p.windows.len())
            .min_by(|x, y| p.yields.get(*x).copied().unwrap_or(0.0).total_cmp(&p.yields.get(*y).copied().unwrap_or(0.0)))
            .unwrap_or(0);
        let others: BTreeSet<usize> =
            p.windows.iter().enumerate().filter(|(k, _)| *k != low).flat_map(|(_, w)| w.iter().copied()).collect();
        let mut seeds: Vec<(usize, f32)> = p.found.iter().enumerate().filter(|(k, _)| *k != low).map(|(_, s)| *s).collect();
        seeds.extend(p.qseeds.iter().copied());
        let walk =
            lens::relation_window(derived, &seeds, &|c| others.contains(&c), &tk, lens::BUDGET, Stop::MarginalGain, KindWeights::Derived);
        let mut switched = p.windows.clone();
        if let Some(w) = switched.get_mut(low) {
            w.clone_from(&walk.chunks);
        }
        b_switch.add(&sub, &switched, &tk, tree, &q.gold);
        b_overlap += overlap(&switched);
        b_rel += walk.relations();
        b_tokens += walk.tokens;
        b_window_hit += usize::from(q.gold.iter().any(|x| walk.chunks.contains(x)));
        let before_hit = p.windows.iter().flatten().any(|c| q.gold.contains(c));
        let after_hit = switched.iter().flatten().any(|c| q.gold.contains(c));
        b_lost += usize::from(before_hit && !after_hit);
        // (b, added) the same walk on a tentacle that had no window in the dump (all eight
        // lens windows unchanged), when one is free.
        let mut added = p.windows.clone();
        if p.windows.len() < lens::NAMES.len() {
            b_free += 1;
            let all: BTreeSet<usize> = p.windows.iter().flatten().copied().collect();
            let mut s2 = p.found.clone();
            s2.extend(p.qseeds.iter().copied());
            added.push(
                lens::relation_window(derived, &s2, &|c| all.contains(&c), &tk, lens::BUDGET, Stop::MarginalGain, KindWeights::Derived)
                    .chunks,
            );
        }
        b_added.add(&sub, &added, &tk, tree, &q.gold);
        b_overlap += overlap(&added);

        // (c) the allocation rule choosing modes. Round 1 is coverage for every tentacle (no
        // findings yet): today's windows, entered in the map with their yields. Round 2 is
        // decided by `allocation::assign` from the map.
        let slots = lens::NAMES.len();
        let mut map = AllocationMap::new(nc, slots);
        for (k, w) in p.windows.iter().enumerate() {
            let t = p.tentacles.get(k).copied().unwrap_or(k);
            for &c in w {
                assert!(map.claim(t, c, Mode::Coverage, None, tk(c), false), "round 1 claim refused");
            }
            map.label(t, lens::NAMES.get(t).copied().unwrap_or(""), &[]);
            map.record_yield(t, p.yields.get(k).copied().unwrap_or(0.0));
        }
        map.next_round();
        // Targets. Relation: each window's first find, owned by its tentacle, worth its yield
        // of the best yield; the question's items, owned by no one. Coverage: the lens lists'
        // next regions, shared by yield. Each family is scaled to its strongest; coverage is
        // then weighed by the share of the uncovered tree one round can take.
        let best_yield = p.yields.iter().copied().fold(f32::MIN_POSITIVE, f32::max);
        let mut targets: Vec<Target> = Vec::new();
        for (k, f) in p.found.iter().enumerate() {
            let t = p.tentacles.get(k).copied().unwrap_or(k);
            targets.push(Target {
                mode: Mode::Relation,
                anchor: f.0,
                utility: (p.yields.get(k).copied().unwrap_or(0.0) / best_yield).max(0.0),
                owner: Some(t),
                ..Target::default()
            });
        }
        for &(c, w) in &p.qseeds {
            if !targets.iter().any(|x| x.anchor == c) {
                targets.push(Target { mode: Mode::Relation, anchor: c, utility: w, owner: None, ..Target::default() });
            }
        }
        let lists = lens::orders(&sub, text, None, &[]);
        let lens_yield: Vec<Option<f32>> = (0..slots).map(|t| map.last_yield(t)).collect();
        let mut cover = allocation::coverage_targets(&map, &lists, &allocation::lens_shares(&lens_yield), &tk, lens::BUDGET);
        let uncovered: usize = map.uncovered().iter().map(|c| tk(*c)).sum();
        let weight = allocation::coverage_weight(slots * lens::BUDGET, uncovered);
        let top_c = cover.iter().map(|x| x.utility).fold(f32::MIN_POSITIVE, f32::max);
        for x in &mut cover {
            x.utility = x.utility / top_c * weight;
        }
        targets.extend(cover);
        let reach = map.reach();
        let mut hops: BTreeMap<usize, BTreeMap<usize, usize>> = BTreeMap::new();
        for x in &targets {
            hops.entry(x.anchor).or_insert_with(|| derived.hops(x.anchor, reach));
        }
        let orders = allocation::assign(&map, &targets, &|x, y| hops.get(&x).and_then(|h| h.get(&y)).copied());
        let mut round2: Vec<Vec<usize>> = vec![Vec::new(); slots];
        for o in &orders {
            c_orders += 1;
            c_moved += usize::from(o.moved);
            match o.mode {
                Mode::Relation => {
                    c_relation_orders += 1;
                    // Seeds: the ordered seed first; a staying tentacle adds its own window's
                    // further finds, a seed from the question adds the question's other items,
                    // each at its own lower relevance.
                    let mut seeds = vec![(o.anchor, 1.0_f32)];
                    let own = p
                        .tentacles
                        .iter()
                        .position(|t| *t == o.tentacle)
                        .filter(|_| targets.get(o.target).is_some_and(|x| x.owner == Some(o.tentacle)));
                    match own.and_then(|k| qs.iter().find(|x| x.0 == *idx).and_then(|x| x.2.windows.get(k))) {
                        Some(w) => {
                            let f = finds(w);
                            let top = f.first().map_or(1.0, |x| x.1).max(f32::MIN_POSITIVE);
                            seeds.extend(f.iter().skip(1).filter(|x| x.1 > 0.0).map(|x| (x.0, (x.1 / top).min(1.0))));
                        }
                        None => seeds.extend(p.qseeds.iter().filter(|x| x.0 != o.anchor).map(|x| (x.0, x.1.min(1.0)))),
                    }
                    let walk = walk_into(&mut map, derived, o.tentacle, &seeds, &tk);
                    c_rel += walk.relations();
                    c_passed += walk.passed.len();
                    if let Some(w) = round2.get_mut(o.tentacle) {
                        w.clone_from(&walk.chunks);
                    }
                }
                Mode::Coverage | Mode::Scan => {
                    c_coverage_orders += 1;
                    let l = o.lens.unwrap_or(0);
                    let mut used = 0;
                    // The region first, then on down the lens's list while the window has room.
                    let region = targets.get(o.target).map_or(&[][..], |x| x.region.as_slice());
                    for &c in region.iter().chain(lists.get(l).map_or(&[][..], Vec::as_slice)) {
                        if used + tk(c) <= lens::BUDGET
                            && map.holder(c).is_none()
                            && map.claim(o.tentacle, c, Mode::Coverage, Some(o.anchor), tk(c), false)
                        {
                            used += tk(c);
                            if let Some(w) = round2.get_mut(o.tentacle) {
                                w.push(c);
                            }
                        }
                    }
                    map.label(o.tentacle, lens::NAMES.get(l).copied().unwrap_or(""), &[]);
                }
            }
        }
        // Windows a relation walk could not fill (its relations ran out below the budget) are
        // topped up from the passed-over references and then from the uncovered lens lists, so
        // every window is a full window.
        for (t, w) in round2.iter_mut().enumerate() {
            let mut used: usize = w.iter().map(|c| tk(*c)).sum();
            if w.is_empty() || used * 10 >= lens::BUDGET * 9 {
                continue;
            }
            c_short += 1;
            let refs: Vec<usize> = map.passed().into_iter().map(|x| x.0).collect();
            for c in refs
                .into_iter()
                .chain((0..lists.iter().map(Vec::len).max().unwrap_or(0)).flat_map(|r| lists.iter().filter_map(move |l| l.get(r).copied())))
            {
                if used + tk(c) <= lens::BUDGET && map.holder(c).is_none() && map.claim(t, c, Mode::Coverage, None, tk(c), false) {
                    used += tk(c);
                    w.push(c);
                }
            }
        }
        let both: Vec<Vec<usize>> = p.windows.iter().cloned().chain(round2.iter().cloned()).collect();
        c_two.add(&sub, &both, &tk, tree, &q.gold);
        c_round2.add(&sub, &round2, &tk, tree, &q.gold);
        c_overlap += overlap(&both) + map.overlap();
        c_refused += map.refused_held;
        let new_hit = round2.iter().flatten().any(|c| q.gold.contains(c));
        c_new_hit += usize::from(new_hit);
        let _ = writeln!(
            per_q,
            "  q{idx:02}: today {} windows, right code {}; switched window: tentacle {} to relation, {} chunks, right code in it {}, anywhere {}; rule: {} orders ({} relation, {} coverage, {} moved on), round 2 right code {}, covered after two rounds {:.2}%",
            p.windows.len(),
            u8::from(before_hit),
            p.tentacles.get(low).copied().unwrap_or(low),
            walk.chunks.len(),
            u8::from(q.gold.iter().any(|x| walk.chunks.contains(x))),
            u8::from(after_hit),
            orders.len(),
            orders.iter().filter(|o| o.mode == Mode::Relation).count(),
            orders.iter().filter(|o| o.mode != Mode::Relation).count(),
            orders.iter().filter(|o| o.moved).count(),
            u8::from(new_hit),
            100.0 * map.covered_share(&tk)
        );
    }
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    let per = |x: usize| x as f64 / n as f64;
    let _ = writeln!(rep, "\n3. WINDOWS BY REPLAY (which code the windows hold; no read on the card)");
    let _ = writeln!(rep, "(a) today's windows: {}; overlap 0", a.text());
    let _ = writeln!(
        rep,
        "(b) one tentacle (the window that yielded least) switched to the relation-guided window, the others unchanged: {}; relation window tokens {:.0}, right code in the relation window {b_window_hit}, questions that lost the right code by the switch {b_lost}, distinct relations followed {:.1}",
        b_switch.text(),
        per(b_tokens),
        per(b_rel)
    );
    let _ = writeln!(
        rep,
        "(b, added) the relation-guided window on a tentacle with no window in the dump ({b_free} of {n} questions had one free), all lens windows unchanged: {}",
        b_added.text()
    );
    let _ = writeln!(rep, "    overlap between windows in (b) and (b, added): {b_overlap}");
    let _ = writeln!(rep, "(c) allocation rule choosing modes; round 1 is coverage (today's windows), round 2 by the rule:");
    let _ = writeln!(rep, "    round 2 alone: {}", c_round2.text());
    let _ = writeln!(rep, "    after two rounds: {}", c_two.text());
    let _ = writeln!(
        rep,
        "    orders per question {:.1}: relation mode {:.1}, coverage mode {:.1}; tentacles moved on {:.1}; relation windows topped up because their relations ran out below nine tenths of the budget {:.1}",
        per(c_orders),
        per(c_relation_orders),
        per(c_coverage_orders),
        per(c_moved),
        per(c_short)
    );
    let _ = writeln!(
        rep,
        "    overlap between windows (counted from the windows and from the map; must be 0): {c_overlap}; distinct relations followed {:.1}; right code in a round 2 window {c_new_hit} of {n}; passed over and kept as references {:.1}; claims refused because another tentacle held the chunk {:.1}",
        per(c_rel),
        per(c_passed),
        per(c_refused)
    );
    let _ = writeln!(rep, "\nper question (index only):\n{per_q}");
    let _ = writeln!(rep, "total {:.1} s", started.elapsed().as_secs_f64());
    assert_eq!(b_overlap, 0, "overlap in the relation arms");
    assert_eq!(c_overlap, 0, "overlap under the allocation rule");
    let path = home.join("runs").join(format!("A4-allocation-replay-{}.txt", nmtype::now_ms()));
    std::fs::write(&path, &rep).expect("record");
    println!("{rep}\nrecord: {}", path.display());
}
