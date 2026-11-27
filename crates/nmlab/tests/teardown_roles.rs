//! Short isolated tests of the role model patches from the open-source teardown (2026-11-10,
//! `docs/2026-11-10-neumor-analysis-j-open-source-teardown.md`, sections D and E).
//! Gavriel Technologies; architecture by Luis Jake Gabriel III.
//!
//! One variable per test, on the graphics card, minutes each. Inputs come from the scan dump
//! (`scratch/tc1-dump/<stamp>/`, `NM_DUMP` relative to the mini, else the newest): its substrate
//! for chunk texts, each question's right chunks, and the 36 chunks the control arm's reranker
//! scored per question. Question text is held in memory only. Every encoding and score lives in
//! RAM in this one process; records hold numbers, question indices and ids only.
//! `NM_LIMIT=<n>` gives the smoke run (n questions, no record written).
//!
//! - `te1_reranker_instruction` (T-E1, patch 1): P(yes) under the card's web-search instruction
//!   and two code instructions, documents at the earlier cap of 384 tokens; pooled AUC of P(yes)
//!   for right against wrong, and the right chunk's place among the question's scored chunks.
//! - `td3_query_instruction` (T-D3, patch 9): about 200 chunks holding every right chunk, document
//!   vectors kept; three query instructions; right chunk in the first 6 and the first 50.
//! - `td1_embed_cap` (T-D1, patch 3): the same chunks embedded at the cap of 384 and at the
//!   derived cap; the right chunk's place by embedding order; build seconds.
//! - `te2_rerank_cap` (T-E2, patch 3): the T-E1 pairs at 384 and at the derived cap; AUC, place,
//!   seconds per document (and per document longer than 384 tokens).
//! - `p7_label_rows` (patch 7): P(yes) with the full output head and with its two label rows only,
//!   on a fixed set; device memory after each.
//!
//! Run one: `cargo test --release -p nmapi --test teardown_roles <name> -- --ignored --nocapture`.
//! Writes `runs/TR-<name>-<ms>.txt`.
use nmapi::scan_dump;
use nmengn::model::{Model, Population};
use nmengn::placement::Device;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Questions of the T-E1 and T-E2 pair set (the teardown's 5 questions of 36 documents).
const PAIR_QUESTIONS: usize = 5;
/// Chunks of the T-D1 and T-D3 set.
const CHUNKS: usize = 200;

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

/// One dump question: index, id, text, right chunks, the chunks the control arm scored.
struct Q {
    idx: usize,
    id: String,
    text: String,
    gold: Vec<usize>,
    scored: Vec<usize>,
}

struct Setup {
    home: PathBuf,
    sub: nmsubs::Substrate,
    qs: Vec<Q>,
    limit: Option<usize>,
}

fn setup() -> Setup {
    let home = nmtype::nano_root();
    let dir = std::env::var("NM_DUMP").map_or_else(
        |_| {
            let mut v: Vec<PathBuf> = std::fs::read_dir(home.join("scratch").join("tc1-dump"))
                .expect("dumps")
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.join("substrate.json").is_file())
                .collect();
            v.sort();
            v.pop().expect("a dump")
        },
        |d| home.join(d),
    );
    let limit = std::env::var("NM_LIMIT").ok().and_then(|s| s.parse().ok());
    let items = dev_half(&home);
    let sub = scan_dump::load_substrate(&dir).expect("substrate");
    let mut qs = Vec::new();
    for idx in scan_dump::indices(&dir) {
        let Some((id, text)) = items.get(idx) else { continue };
        let (q, _) = scan_dump::load_question(&dir, idx, text).expect("question");
        let scored: Vec<usize> = scan_dump::load_rerank(&dir, idx).expect("rerank").into_keys().collect();
        qs.push(Q { idx, id: id.clone(), text: text.clone(), gold: q.gold, scored });
    }
    Setup { home, sub, qs, limit }
}

fn doc(sub: &nmsubs::Substrate, c: usize) -> String {
    sub.chunks.get(c).and_then(|ch| sub.files.get(ch.file).map(|f| nmsubs::dense::doc_text(&f.path, &ch.text))).unwrap_or_default()
}

fn open(home: &Path, id: &str, asm: &str) -> (Arc<Model>, Device) {
    let pop = Population::open(home, id, true, true).expect("population");
    let dev = if pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    (Arc::new(Model::bind(&pop, nmengn::Assembly::load(&home.join("designs/assemblies"), asm).expect("assembly")).expect("bind")), dev)
}

/// Pooled AUC of scores for positives against negatives (ties count half).
#[expect(clippy::cast_precision_loss, reason = "small counts")]
fn auc(pos: &[f32], neg: &[f32]) -> f64 {
    let mut s = 0.0;
    for p in pos {
        for n in neg {
            s += if p > n {
                1.0
            } else if (p - n).abs() < f32::EPSILON {
                0.5
            } else {
                0.0
            };
        }
    }
    s / (pos.len() * neg.len()).max(1) as f64
}

/// Place (0 best) of the best-placed right chunk in `order`.
fn place(order: &[usize], gold: &[usize]) -> Option<usize> {
    order.iter().position(|c| gold.contains(c))
}

fn record(s: &Setup, name: &str, rep: &str) {
    eprintln!("{rep}");
    if s.limit.is_none() {
        let p = s.home.join("runs").join(format!("TR-{name}-{}.txt", nmtype::now_ms()));
        std::fs::write(&p, rep).expect("record");
        eprintln!("record {}", p.display());
    }
}

/// The pair set: the first questions whose right chunk is among the scored chunks.
/// In the smoke run each question keeps its right chunks and 5 others.
fn pair_questions(s: &Setup) -> Vec<Q> {
    s.qs.iter()
        .filter(|q| q.scored.iter().any(|c| q.gold.contains(c)))
        .take(s.limit.unwrap_or(PAIR_QUESTIONS))
        .map(|q| {
            let mut scored = q.scored.clone();
            if s.limit.is_some() {
                let (g, o): (Vec<usize>, Vec<usize>) = scored.iter().partition(|c| q.gold.contains(c));
                scored = g.into_iter().chain(o.into_iter().take(5)).collect();
            }
            Q { idx: q.idx, id: q.id.clone(), text: q.text.clone(), gold: q.gold.clone(), scored }
        })
        .collect()
}

/// P(yes) for every pair of the pair set under one instruction and cap; (scores per question,
/// seconds per document, seconds per document longer than the earlier cap).
fn score_pairs(
    m: &Model,
    dev: Device,
    sub: &nmsubs::Substrate,
    qs: &[&Q],
    instruction: &str,
    cap: Option<usize>,
) -> (Vec<Vec<f32>>, f64, Option<f64>) {
    let (mut out, mut secs, mut n, mut long_s, mut long_n) = (Vec::new(), 0.0, 0_usize, 0.0, 0_usize);
    for q in qs {
        let mut v = Vec::new();
        for c in &q.scored {
            let d = doc(sub, *c);
            let toks = m.pop.tokenizer.encode(&d).map_or(0, |x| x.len());
            let t = Instant::now();
            let p = nmengn::roles::card_rerank_set_capped(m, dev, instruction, &q.text, std::slice::from_ref(&d), cap).expect("rerank");
            let e = t.elapsed().as_secs_f64();
            secs += e;
            n += 1;
            if toks > nmengn::roles::EARLIER_DOC_CAP {
                long_s += e;
                long_n += 1;
            }
            v.push(p.first().copied().unwrap_or(0.0));
        }
        out.push(v);
    }
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    let per = |s: f64, n: usize| s / n.max(1) as f64;
    (out, per(secs, n), (long_n > 0).then(|| per(long_s, long_n)))
}

/// AUC and mean place of the right chunk among each question's scored chunks.
fn pair_measures(qs: &[&Q], scores: &[Vec<f32>]) -> (f64, f64, usize) {
    let (mut pos, mut neg, mut places, mut firsts) = (Vec::new(), Vec::new(), Vec::new(), 0);
    for (q, v) in qs.iter().zip(scores) {
        for (c, p) in q.scored.iter().zip(v) {
            if q.gold.contains(c) { pos.push(*p) } else { neg.push(*p) }
        }
        let mut order: Vec<usize> = (0..v.len()).collect();
        order.sort_by(|a, b| v.get(*b).unwrap_or(&0.0).total_cmp(v.get(*a).unwrap_or(&0.0)));
        let ord: Vec<usize> = order.iter().filter_map(|i| q.scored.get(*i).copied()).collect();
        if let Some(p) = place(&ord, &q.gold) {
            places.push(p);
            firsts += usize::from(p == 0);
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    let mean = places.iter().sum::<usize>() as f64 / places.len().max(1) as f64;
    (auc(&pos, &neg), mean, firsts)
}

#[test]
#[ignore = "runs the reranker on the graphics card (a few minutes)"]
fn te1_reranker_instruction() {
    let s = setup();
    let started = Instant::now();
    let (m, dev) = open(&s.home, "qwen3-reranker-0.6b", "qwen3-reranker");
    let owned = pair_questions(&s);
    let qs: Vec<&Q> = owned.iter().collect();
    let pairs: usize = qs.iter().map(|q| q.scored.len()).sum();
    let mut rep = format!(
        "T-E1 reranker instruction (teardown patch 1), {} questions {:?}, {pairs} pairs, documents at the earlier cap of {} tokens, on {}\n",
        qs.len(),
        qs.iter().map(|q| (q.idx + 1, q.id.as_str())).collect::<Vec<_>>(),
        nmengn::roles::EARLIER_DOC_CAP,
        dev.name()
    );
    let instr = [
        ("card web search (present)", nmengn::roles::CARD_INSTRUCTION),
        ("code judge", nmengn::roles::CODE_INSTRUCTIONS[0]),
        ("code retrieve (lane task)", nmengn::roles::CODE_INSTRUCTIONS[1]),
    ];
    for (name, text) in instr {
        let (sc, per, _) = score_pairs(&m, dev, &s.sub, &qs, text, Some(nmengn::roles::EARLIER_DOC_CAP));
        let (a, mp, f) = pair_measures(&qs, &sc);
        let _ = writeln!(
            rep,
            "  {name}: AUC {a:.4}, right chunk mean place {mp:.2} (0 best), first in {f} of {}; {per:.3} s per document",
            qs.len()
        );
    }
    let _ = writeln!(rep, "total {:.1} s", started.elapsed().as_secs_f64());
    record(&s, "te1", &rep);
}

#[test]
#[ignore = "runs the reranker on the graphics card (a few minutes)"]
fn te2_rerank_cap() {
    let s = setup();
    let started = Instant::now();
    let (m, dev) = open(&s.home, "qwen3-reranker-0.6b", "qwen3-reranker");
    let owned = pair_questions(&s);
    let qs: Vec<&Q> = owned.iter().collect();
    let fixed = m.pop.tokenizer.encode(&nmengn::roles::card_prompt(nmengn::roles::CODE_INSTRUCTION, "q", "")).map_or(0, |x| x.len());
    let derived = nmengn::roles::input_cap(&m, dev, fixed);
    let long = qs
        .iter()
        .flat_map(|q| q.scored.iter())
        .filter(|c| m.pop.tokenizer.encode(&doc(&s.sub, **c)).map_or(0, |x| x.len()) > nmengn::roles::EARLIER_DOC_CAP)
        .count();
    let mut rep = format!(
        "T-E2 reranker document cap (teardown patch 3), {} questions, {} pairs ({long} documents longer than {} tokens), instruction CODE_INSTRUCTION, on {}; derived cap now {derived}\n",
        qs.len(),
        qs.iter().map(|q| q.scored.len()).sum::<usize>(),
        nmengn::roles::EARLIER_DOC_CAP,
        dev.name()
    );
    for (name, cap) in [("cap 384 (earlier)", Some(nmengn::roles::EARLIER_DOC_CAP)), ("derived cap", None)] {
        let (sc, per, per_long) = score_pairs(&m, dev, &s.sub, &qs, nmengn::roles::CODE_INSTRUCTION, cap);
        let (a, mp, f) = pair_measures(&qs, &sc);
        let _ = writeln!(
            rep,
            "  {name}: AUC {a:.4}, right chunk mean place {mp:.2}, first in {f} of {}; {per:.3} s per document, {} s per document longer than 384 tokens",
            qs.len(),
            per_long.map_or_else(|| "-".to_owned(), |x| format!("{x:.3}"))
        );
    }
    let _ = writeln!(rep, "total {:.1} s", started.elapsed().as_secs_f64());
    record(&s, "te2", &rep);
}

/// The T-D1 and T-D3 chunk set: every right chunk of the questions, then other chunks in tree order.
fn chunk_set(s: &Setup, qs: &[&Q]) -> Vec<usize> {
    let mut set: Vec<usize> = qs.iter().flat_map(|q| q.gold.iter().copied()).collect();
    set.sort_unstable();
    set.dedup();
    let want = if s.limit.is_some() { set.len() + 10 } else { CHUNKS.max(set.len()) };
    for c in 0..s.sub.chunks.len() {
        if set.len() >= want {
            break;
        }
        if !set.contains(&c) {
            set.push(c);
        }
    }
    set
}

fn embed_set(m: &Model, dev: Device, s: &Setup, set: &[usize], cap: Option<usize>) -> (BTreeMap<usize, Vec<f32>>, f64, usize) {
    let t = Instant::now();
    let mut toks = 0;
    let v = set
        .iter()
        .map(|c| {
            let d = doc(&s.sub, *c);
            toks += nmengn::roles::card_embed_ids_capped(m, &d, cap).map_or(0, |x| x.len());
            (*c, nmengn::roles::card_embed_on_capped(m, dev, &d, cap).expect("embed"))
        })
        .collect();
    (v, t.elapsed().as_secs_f64(), toks)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Places (0 best) of each question's right chunk by embedding order over the kept vectors.
fn embed_places(m: &Model, dev: Device, qs: &[&Q], vecs: &BTreeMap<usize, Vec<f32>>, task: &str) -> Vec<usize> {
    qs.iter()
        .map(|q| {
            let qv = nmengn::roles::card_embed_on(m, dev, &nmengn::roles::card_query(task, &q.text)).expect("query");
            let mut order: Vec<(f32, usize)> = vecs.iter().map(|(c, v)| (dot(&qv, v), *c)).collect();
            order.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            place(&order.iter().map(|x| x.1).collect::<Vec<_>>(), &q.gold).unwrap_or(usize::MAX)
        })
        .collect()
}

fn median(v: &[usize]) -> usize {
    let mut w = v.to_vec();
    w.sort_unstable();
    w.get(w.len() / 2).copied().unwrap_or(0)
}

#[test]
#[ignore = "runs the embedder on the graphics card (a few minutes)"]
fn td3_query_instruction() {
    let s = setup();
    let started = Instant::now();
    let (m, dev) = open(&s.home, "qwen3-embedding-0.6b", "qwen3-embedding");
    let qs: Vec<&Q> = s.qs.iter().take(s.limit.unwrap_or(usize::MAX)).collect();
    let set = chunk_set(&s, &qs);
    let (vecs, build, _) = embed_set(&m, dev, &s, &set, Some(nmengn::roles::EARLIER_DOC_CAP));
    let mut rep = format!(
        "T-D3 query instruction of the embedder (teardown patch 9), {} questions, {} chunks (every right chunk included), document vectors at cap 384 kept in RAM (built in {build:.1} s), on {}\n",
        qs.len(),
        set.len(),
        dev.name()
    );
    let tasks = [
        ("lane task (present)", nmsubs::dense::QUERY_TASK),
        ("card default", nmengn::roles::CARD_INSTRUCTION),
        ("behaviour", "Given a description of what some code does, retrieve the code that does it"),
    ];
    for (name, task) in tasks {
        let p = embed_places(&m, dev, &qs, &vecs, task);
        let fold = |f: usize| p.iter().zip(&qs).filter(|(x, q)| q.idx % 2 == f && **x < 6).count();
        let _ = writeln!(
            rep,
            "  {name}: right chunk in first 6 {} of {} (fold A {}, fold B {}), in first 50 {}, median place {}",
            p.iter().filter(|x| **x < 6).count(),
            qs.len(),
            fold(0),
            fold(1),
            p.iter().filter(|x| **x < 50).count(),
            median(&p)
        );
    }
    let _ = writeln!(rep, "total {:.1} s", started.elapsed().as_secs_f64());
    record(&s, "td3", &rep);
}

#[test]
#[ignore = "runs the embedder on the graphics card (a few minutes)"]
fn td1_embed_cap() {
    let s = setup();
    let started = Instant::now();
    let (m, dev) = open(&s.home, "qwen3-embedding-0.6b", "qwen3-embedding");
    let qs: Vec<&Q> = s.qs.iter().take(s.limit.unwrap_or(usize::MAX)).collect();
    let set = chunk_set(&s, &qs);
    let derived = nmengn::roles::input_cap(&m, dev, 1);
    let mut rep = format!(
        "T-D1 embedder document cap (teardown patch 3), {} questions, {} chunks (every right chunk included), query task QUERY_TASK, on {}; derived cap now {derived}\n",
        qs.len(),
        set.len(),
        dev.name()
    );
    let mut places = Vec::new();
    for (name, cap) in [("cap 384 (earlier)", Some(nmengn::roles::EARLIER_DOC_CAP)), ("derived cap", None)] {
        let (vecs, build, toks) = embed_set(&m, dev, &s, &set, cap);
        let p = embed_places(&m, dev, &qs, &vecs, nmsubs::dense::QUERY_TASK);
        #[expect(clippy::cast_precision_loss, reason = "small counts")]
        let per_tok = build * 1000.0 / toks.max(1) as f64;
        let _ = writeln!(
            rep,
            "  {name}: right chunk median place {}, first 6 {} of {}, first 50 {}; build {build:.1} s for {toks} tokens ({per_tok:.3} ms per token)",
            median(&p),
            p.iter().filter(|x| **x < 6).count(),
            qs.len(),
            p.iter().filter(|x| **x < 50).count()
        );
        places.push(p);
    }
    if let [a, b] = places.as_slice() {
        let fell = a.iter().zip(b).filter(|(x, y)| **x < 50 && **y >= 50).count();
        let (better, worse) = (a.iter().zip(b).filter(|(x, y)| y < x).count(), a.iter().zip(b).filter(|(x, y)| y > x).count());
        let _ = writeln!(rep, "  per question: derived cap better on {better}, worse on {worse}; fell out of the first 50: {fell}");
    }
    let tree_tokens: usize = s
        .sub
        .chunks
        .iter()
        .map(|c| {
            s.sub
                .files
                .get(c.file)
                .map_or(0, |f| m.pop.tokenizer.encode(&nmsubs::dense::doc_text(&f.path, &c.text)).map_or(0, |x| x.len() + 1))
        })
        .sum();
    let tree_capped: usize = s
        .sub
        .chunks
        .iter()
        .map(|c| {
            s.sub.files.get(c.file).map_or(0, |f| {
                m.pop
                    .tokenizer
                    .encode(&nmsubs::dense::doc_text(&f.path, &c.text))
                    .map_or(0, |x| x.len().min(nmengn::roles::EARLIER_DOC_CAP) + 1)
            })
        })
        .sum();
    let _ = writeln!(
        rep,
        "  whole tree of the dump: {} chunks, {tree_capped} tokens at cap 384, {tree_tokens} uncapped (ratio {:.2})",
        s.sub.chunks.len(),
        tree_tokens as f64 / tree_capped.max(1) as f64
    );
    let _ = writeln!(rep, "total {:.1} s", started.elapsed().as_secs_f64());
    record(&s, "td1", &rep);
}

#[test]
#[ignore = "runs the reranker on the graphics card (a few minutes)"]
fn p7_label_rows() {
    let s = setup();
    let started = Instant::now();
    let owned = pair_questions(&s);
    let qs: Vec<&Q> = owned.iter().take(s.limit.unwrap_or(2)).collect();
    let pop = Population::open(&s.home, "qwen3-reranker-0.6b", true, true).expect("population");
    let dev = if pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let asm = || nmengn::Assembly::load(&s.home.join("designs/assemblies"), "qwen3-reranker").expect("assembly");
    let mut rep = format!("P7 reranker output head: two label rows against the full head (teardown patch 7), {} questions, {} pairs, on {}; arms in the order rows, full, rows (each from a released device)
", qs.len(), qs.iter().map(|q| q.scored.len()).sum::<usize>(), dev.name());
    let mut all: Vec<Vec<Vec<f32>>> = Vec::new();
    for (name, full) in [("two label rows", false), ("full head", true), ("two label rows again", false)] {
        pop.release_device();
        let free0 = pop.gpu_free().unwrap_or(0);
        let m = Model::bind(&pop, asm()).expect("bind");
        let m = if full { m.full_head() } else { m.label_rows() };
        let (sc, per, _) = score_pairs(&m, dev, &s.sub, &qs, nmengn::roles::CODE_INSTRUCTION, Some(nmengn::roles::EARLIER_DOC_CAP));
        let used = free0.saturating_sub(pop.gpu_free().unwrap_or(0));
        let (_, _, held) = m.holdings();
        let _ = writeln!(
            rep,
            "  {name}: device memory taken during the arm {} MB; weight copies resident {} MB; {per:.3} s per document",
            used >> 20,
            held >> 20
        );
        all.push(sc);
        drop(m);
    }
    pop.release_device();
    if let [a, b, _] = all.as_slice() {
        let diffs: Vec<f32> = a.iter().flatten().zip(b.iter().flatten()).map(|(x, y)| (x - y).abs()).collect();
        let same = a.iter().flatten().zip(b.iter().flatten()).filter(|(x, y)| x.to_bits() == y.to_bits()).count();
        let _ = writeln!(
            rep,
            "  P(yes) bit-identical on {same} of {} pairs; largest difference {:.2e}",
            diffs.len(),
            diffs.iter().copied().fold(0.0, f32::max)
        );
    }
    let _ = writeln!(rep, "total {:.1} s", started.elapsed().as_secs_f64());
    record(&s, "p7", &rep);
}
