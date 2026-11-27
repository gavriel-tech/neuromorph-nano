//! E2 (runbook `runs/2026-11-10-E2-runbook.md`; design by Luis Jake Gabriel III,
//! `docs/2026-11-10-neumor-design-context-architecture.md`): the released Qwen3-0.6B's attention
//! heads as drone tips. Per CX1 development question, the dense lane's candidates fill one
//! window of encoded code (E1, by reference); every head's calibrated attention mass orders the
//! candidates (QRHead arXiv 2506.09944, ICR arXiv 2410.02642), against the lane's order, the
//! lexical order and the released reranker on the same candidates. Heads are chosen on one fold
//! and scored on the other. Records hold question ids only.
use nmengn::assembly::Assembly;
use nmengn::kvstore::{ChunkKv, Probe, Window, encode};
use nmengn::model::{Model, Population, RunOpts};
use nmengn::placement::Device;
use nmengn::roles;
use nmtype::{Fault, FaultKind};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt::Write as _;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

/// Chunk tokens per window (about the GPU's key and value room beside one resident model).
const BUDGET: usize = 4_096;
/// How deep the lane's fused order is read for candidates.
const DEPTH: usize = 50;
/// Heads summed after selection (QRHead's count).
const PICK: usize = 16;
/// Encoded chunks kept between questions (least recently used out).
const CACHE_BYTES: usize = 4 << 30;
const TYPES: [&str; 5] = ["behaviour", "name", "value", "explain", "relation"];
const ANCHOR: &str = "Repository neuromorph-mini: a Rust workspace of 18 crates (nmsubs substrate, nmpod tentacles, nmbrain Core Brain, nmengn engine, nmapi tools).\n";
const NULL_QUERY: &str = "\nQuestion: N/A";

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("e2", FaultKind::Execution, m)
}

pub(crate) struct Item {
    pub(crate) id: String,
    pub(crate) q: String,
    pub(crate) keys: Vec<(String, usize, usize)>,
}

/// The CX1 development half, in the locked harness's order.
pub(crate) fn suite(home: &Path) -> Result<Vec<Item>, Fault> {
    let mut out = Vec::new();
    for area in ["context", "engine", "tools"] {
        let p = home.join(format!("tests/capability-gates/cx1-context/questions-{area}.json"));
        let v = nmjson::parse(&std::fs::read(&p).map_err(|e| bad(format!("{}: {e}", p.display())))?).map_err(|e| bad(e.to_string()))?;
        let s = |x: &nmjson::Value, k: &str| -> Result<String, Fault> {
            Ok(x.get(k).and_then(|y| y.string()).map_err(|e| bad(e.to_string()))?.to_owned())
        };
        let range = |x: &nmjson::Value| -> Result<(String, usize, usize), Fault> {
            let l = x.get("lines").and_then(|y| y.array()).map_err(|e| bad(e.to_string()))?;
            let n = |y: Option<&nmjson::Value>| -> Result<usize, Fault> {
                usize::try_from(y.ok_or_else(|| bad("empty lines"))?.integer().map_err(|e| bad(e.to_string()))?)
                    .map_err(|e| bad(e.to_string()))
            };
            Ok((s(x, "file")?, n(l.first())?, n(l.last())?))
        };
        let mut qs = Vec::new();
        for x in v.get("questions").and_then(|y| y.array()).map_err(|e| bad(e.to_string()))? {
            let t = s(x, "type")?;
            let ti = TYPES.iter().position(|y| *y == t).ok_or_else(|| bad(format!("unknown type {t}")))?;
            let mut keys = vec![range(x)?];
            if let Ok(also) = x.get("also").and_then(|y| y.array()) {
                for a in also {
                    keys.push(range(a)?);
                }
            }
            qs.push((ti, s(x, "id")?, s(x, "q")?, keys));
        }
        qs.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
        out.extend(qs.into_iter().enumerate().filter(|(pos, _)| pos % 2 == 0).map(|(_, (_, id, q, keys))| Item { id, q, keys }));
    }
    Ok(out)
}

/// One question's readings: candidates in lane order, which are right, each arm's order.
struct Read {
    id: String,
    right: Vec<bool>,
    /// `[layer][head][candidate]` calibrated scores.
    heads: Vec<Vec<Vec<f32>>>,
    lexical: Vec<usize>,
    rerank: Vec<f32>,
    tokens: usize,
}

/// Rank (0 first) of the first right candidate when ordered by `score` (ties: lane order).
fn first_right(score: &[f32], right: &[bool]) -> Option<usize> {
    let mut o: Vec<usize> = (0..score.len()).collect();
    o.sort_by(|a, b| score[*b].total_cmp(&score[*a]).then(a.cmp(b)));
    o.iter().position(|i| right[*i])
}
fn by_order(order: &[usize], right: &[bool]) -> Option<usize> {
    order.iter().position(|i| right[*i])
}
#[expect(clippy::cast_precision_loss, reason = "ranks are small")]
fn rr(r: Option<usize>) -> f64 {
    r.map_or(0.0, |x| 1.0 / (x + 1) as f64)
}

/// Encoded chunks by content key, least recently used out past `CACHE_BYTES`.
#[derive(Default)]
struct Cache {
    map: HashMap<String, Arc<ChunkKv>>,
    order: VecDeque<String>,
    bytes: usize,
}
impl Cache {
    fn get(&mut self, key: &str) -> Option<Arc<ChunkKv>> {
        let c = self.map.get(key).cloned()?;
        self.order.retain(|k| k != key);
        self.order.push_back(key.to_owned());
        Some(c)
    }
    fn put(&mut self, key: &str, c: Arc<ChunkKv>) {
        self.bytes += c.bytes();
        self.map.insert(key.to_owned(), c);
        self.order.push_back(key.to_owned());
        while self.bytes > CACHE_BYTES {
            let Some(k) = self.order.pop_front() else { break };
            if let Some(old) = self.map.remove(&k) {
                self.bytes -= old.bytes();
            }
        }
    }
}

fn device_of(pop: &Population, want: Device) -> Device {
    if want == Device::Gpu && pop.gpu.is_some() { Device::Gpu } else { Device::Cpu }
}

#[expect(clippy::too_many_lines, reason = "one run, phases in order")]
fn e2(home: &Path, stub: bool, want: Device) -> Result<(String, String), Fault> {
    let mut items = suite(home)?;
    if stub {
        items.truncate(2);
    }
    let mut out = String::new();
    let org = nmapi::forage::Organism::open_with(home, home, &["neuromorph-nano".to_owned()], false, &home.join("memory"))?;
    let sub = org.sub();
    let doc = |i: usize| -> String {
        sub.files.get(sub.chunks[i].file).map_or_else(String::new, |f| nmsubs::dense::doc_text(&f.path, &sub.chunks[i].text))
    };
    let path_of = |i: usize| -> String { sub.files.get(sub.chunks[i].file).map_or_else(String::new, |f| f.path.replace('\\', "/")) };

    // Phase 1: the lane's fused order per question (embedder only on the device).
    let t = Instant::now();
    let orders: Vec<Vec<usize>> = {
        let pop = Population::open(home, "qwen3-embedding-0.6b", true, want == Device::Gpu)?;
        let dev = device_of(&pop, want);
        let m = Arc::new(Model::bind(&pop, Assembly::load(&home.join("designs/assemblies"), "qwen3-embedding")?)?);
        let root_tag = nmhash::sha256_bytes(sub.root.to_string_lossy().as_bytes());
        let file = home.join(format!(
            "scratch/dense/{}-{}.vec",
            pop.card.sha_main.get(..12).unwrap_or("model"),
            root_tag.get(..12).unwrap_or("root")
        ));
        let (m1, m2) = (Arc::clone(&m), Arc::clone(&m));
        let lane = nmsubs::dense::DenseLane::open(
            Some(file),
            Arc::new(move |s: &str| roles::card_embed_on(&m1, dev, s).ok()),
            Arc::new(move |s: &str| roles::card_embed_on(&m2, dev, s).ok()),
        );
        let held = lane.len();
        let filled = lane.fill(&sub, &|| true);
        let _ = writeln!(out, "lane: {} chunks, {held} vectors on disk, {filled} embedded now ({})", sub.chunks.len(), dev.name());
        items
            .iter()
            .map(|x| lane.order(&sub, &x.q, DEPTH).ok_or_else(|| bad(format!("{}: no lane order", x.id))))
            .collect::<Result<_, _>>()?
    };
    let _ = writeln!(out, "phase 1 (lane orders): {:.1} s", t.elapsed().as_secs_f64());

    // Phase 2: windows, encoded once per chunk, read by the question and the content-free query.
    let pop = Population::open(home, "qwen3-0.6b", true, want == Device::Gpu)?;
    let dev = device_of(&pop, want);
    let m = Model::bind(&pop, Assembly::load(&home.join("designs/assemblies"), "qwen3-reference")?)?;
    let d = pop.dims;
    let tok = |s: &str| pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
    let anchor_ids = tok(ANCHOR)?;
    let ak = Arc::new(encode(&m, dev, None, &anchor_ids)?);
    let anchor = Arc::new(Window::build(&m, &[Arc::clone(&ak)])?);
    let layers: Vec<usize> = (0..d.layers).collect();
    let final_norm = (nmengn::assembly::nid("final_norm"), "y".to_owned());
    let mut cache = Cache::default();
    let (mut enc_tokens, mut enc_ms, mut read_ms, mut build_ms) = (0_usize, 0.0_f64, 0.0_f64, 0.0_f64);
    let mut reads: Vec<Read> = Vec::new();
    let mut cands_all: Vec<Vec<usize>> = Vec::new();
    for (x, order) in items.iter().zip(&orders) {
        // Candidates in lane order up to the budget; chunk ids per candidate.
        let (mut cand, mut ids, mut used) = (Vec::new(), Vec::new(), 0_usize);
        for &i in order {
            let t_ids = tok(&doc(i))?;
            if used + t_ids.len() > BUDGET {
                continue;
            }
            used += t_ids.len();
            cand.push(i);
            ids.push(t_ids);
        }
        let right: Vec<bool> = cand
            .iter()
            .map(|&i| {
                let (p, c) = (path_of(i), &sub.chunks[i]);
                x.keys.iter().any(|(f, a, b)| *f == p && c.start <= *b && c.end >= *a)
            })
            .collect();
        // Window in repository order.
        let mut place: Vec<usize> = (0..cand.len()).collect();
        place.sort_by(|a, b| (path_of(cand[*a]), sub.chunks[cand[*a]].start).cmp(&(path_of(cand[*b]), sub.chunks[cand[*b]].start)));
        let mut parts = vec![Arc::clone(&ak)];
        for &j in &place {
            let key = sub.chunks[cand[j]].key.clone();
            let c = if let Some(c) = cache.get(&key) {
                c
            } else {
                let t = Instant::now();
                let c = Arc::new(encode(&m, dev, Some(&anchor), &ids[j])?);
                enc_ms += t.elapsed().as_secs_f64() * 1000.0;
                enc_tokens += ids[j].len();
                cache.put(&key, Arc::clone(&c));
                c
            };
            parts.push(c);
        }
        let t = Instant::now();
        let w = Arc::new(Window::build(&m, &parts)?);
        build_ms += t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let read = |text: &str| -> Result<(BTreeMap<usize, Vec<f32>>, usize), Fault> {
            let q = tok(text)?;
            let probe = Arc::new(Probe::new(layers.clone()));
            m.forward_at(
                dev,
                &q,
                w.len,
                std::slice::from_ref(&final_norm),
                &RunOpts { window: Some(Arc::clone(&w)), probe: Some(Arc::clone(&probe)), ..RunOpts::default() },
            )?;
            let mut got = BTreeMap::new();
            for &l in &layers {
                got.insert(l, probe.get(l).ok_or_else(|| bad(format!("{}: layer {l} not read", x.id)))?);
            }
            Ok((got, q.len()))
        };
        let (mq, nq) = read(&format!("\nQuestion: {}", x.q))?;
        let (mn, nn) = read(NULL_QUERY)?;
        read_ms += t.elapsed().as_secs_f64() * 1000.0;
        // Part k of the window (k >= 1) holds candidate place[k - 1].
        #[expect(clippy::cast_precision_loss, reason = "token counts are small")]
        let (fq, fnn) = (1.0 / nq as f32, 1.0 / nn as f32);
        let mut heads = vec![vec![vec![0.0_f32; cand.len()]; d.heads]; d.layers];
        for (l, per_l) in heads.iter_mut().enumerate() {
            let (a, b) = (&mq[&l], &mn[&l]);
            for (h, per_h) in per_l.iter_mut().enumerate() {
                for (k, &(start, n)) in w.parts.iter().enumerate().skip(1) {
                    let row = h * w.len;
                    let s: f32 = (start..start + n).map(|p| a[row + p] * fq - b[row + p] * fnn).sum();
                    per_h[place[k - 1]] = s;
                }
            }
        }
        let lex = sub.lexical_order(&x.q);
        let mut lexical: Vec<usize> = (0..cand.len()).collect();
        lexical.sort_by_key(|j| (lex.iter().position(|i| *i == cand[*j]).unwrap_or(usize::MAX), *j));
        reads.push(Read { id: x.id.clone(), right, heads, lexical, rerank: Vec::new(), tokens: w.len });
        cands_all.push(cand);
    }
    let window_kb = cache.bytes >> 10;
    drop(m);
    drop(pop);

    // Phase 3: the released reranker on the same candidates.
    let t = Instant::now();
    let mut n_docs = 0_usize;
    {
        let pop = Population::open(home, "qwen3-reranker-0.6b", true, want == Device::Gpu)?;
        let rdev = device_of(&pop, want);
        let m = Model::bind(&pop, Assembly::load(&home.join("designs/assemblies"), "qwen3-reranker")?)?;
        for ((r, x), cand) in reads.iter_mut().zip(&items).zip(&cands_all) {
            let docs: Vec<String> = cand.iter().map(|&i| doc(i)).collect();
            n_docs += docs.len();
            r.rerank = roles::card_rerank_set(&m, rdev, roles::CARD_INSTRUCTION, &x.q, &docs)?;
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "counts for a report")]
    let rerank_ms = t.elapsed().as_secs_f64() * 1000.0 / n_docs.max(1) as f64;

    // Scoring.
    let n = reads.len();
    let fold = |i: usize| i % 2;
    let head_rank = |r: &Read, l: usize, h: usize| first_right(&r.heads[l][h], &r.right);
    let sel_heads = |f: usize| -> Vec<(usize, usize)> {
        let mut all: Vec<(usize, f64, usize, usize)> = Vec::new();
        for l in 0..d.layers {
            for h in 0..d.heads {
                let (mut top, mut mrr) = (0_usize, 0.0_f64);
                for (_, r) in reads.iter().enumerate().filter(|(i, _)| fold(*i) == f) {
                    let k = head_rank(r, l, h);
                    top += usize::from(k == Some(0));
                    mrr += rr(k);
                }
                all.push((top, mrr, l, h));
            }
        }
        all.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.total_cmp(&a.1)).then((a.2, a.3).cmp(&(b.2, b.3))));
        all.into_iter().take(PICK).map(|x| (x.2, x.3)).collect()
    };
    let sum_of = |r: &Read, hs: &[(usize, usize)]| -> Vec<f32> {
        (0..r.right.len()).map(|c| hs.iter().map(|&(l, h)| r.heads[l][h][c]).sum()).collect()
    };
    let picks = [sel_heads(0), sel_heads(1)];
    let all_heads: Vec<(usize, usize)> = (0..d.layers).flat_map(|l| (0..d.heads).map(move |h| (l, h))).collect();
    let mut arms: BTreeMap<&str, Vec<Option<usize>>> = BTreeMap::new();
    for (i, r) in reads.iter().enumerate() {
        let lane: Vec<usize> = (0..r.right.len()).collect();
        arms.entry("1 lane").or_default().push(by_order(&lane, &r.right));
        arms.entry("2 lexical").or_default().push(by_order(&r.lexical, &r.right));
        arms.entry("3 reranker").or_default().push(first_right(&r.rerank, &r.right));
        arms.entry("4 heads selected out of sample").or_default().push(first_right(&sum_of(r, &picks[1 - fold(i)]), &r.right));
        arms.entry("5 all heads").or_default().push(first_right(&sum_of(r, &all_heads), &r.right));
    }
    // Best single head in sample (reported only).
    let mut best = (0_usize, 0.0_f64, 0_usize, 0_usize);
    for l in 0..d.layers {
        for h in 0..d.heads {
            let ks: Vec<Option<usize>> = reads.iter().map(|r| head_rank(r, l, h)).collect();
            let top = ks.iter().filter(|k| **k == Some(0)).count();
            let mrr: f64 = ks.iter().map(|k| rr(*k)).sum();
            if (top, mrr) > (best.0, best.1) {
                best = (top, mrr, l, h);
            }
        }
    }
    let in_window = reads.iter().filter(|r| r.right.iter().any(|x| *x)).count();
    let _ = writeln!(
        out,
        "E2 {} on {}: {n} development questions, right code in the window for {in_window}/{n}",
        if stub { "stub" } else { "dev" },
        dev.name()
    );
    for (name, ks) in &arms {
        let top = ks.iter().filter(|k| **k == Some(0)).count();
        let top3 = ks.iter().filter(|k| k.is_some_and(|x| x < 3)).count();
        #[expect(clippy::cast_precision_loss, reason = "counts for a report")]
        let mrr = ks.iter().map(|k| rr(*k)).sum::<f64>() / n.max(1) as f64;
        let _ = writeln!(out, "{name:<32} top {top}/{n}, first three {top3}/{n}, mean reciprocal rank {mrr:.3}");
    }
    #[expect(clippy::cast_precision_loss, reason = "counts for a report")]
    let best_mrr = best.1 / n.max(1) as f64;
    let _ = writeln!(
        out,
        "{:<32} top {}/{n} (layer {} head {}, mean reciprocal rank {best_mrr:.3}; chosen on the same questions, reported only)",
        "6 best single head in sample", best.0, best.2, best.3
    );
    for (f, p) in picks.iter().enumerate() {
        let _ = writeln!(
            out,
            "heads chosen on fold {} (layer.head): {}",
            if f == 0 { "A" } else { "B" },
            p.iter().map(|(l, h)| format!("{l}.{h}")).collect::<Vec<_>>().join(" ")
        );
    }
    let top_of = |k: &str| arms.get(k).map_or(0, |ks| ks.iter().filter(|x| **x == Some(0)).count());
    let b = top_of("1 lane").max(top_of("2 lexical")).max(top_of("3 reranker"));
    let h = top_of("4 heads selected out of sample");
    let verdict = if h >= b + 2 {
        "heads at least 2 above the best ordering: heads read and judge in E3 (next: a locked claim run)"
    } else if h + 1 >= b {
        "heads within 1 of the best ordering: heads read and cover, the best ordering orders; E3 keeps both"
    } else {
        "heads 2 or more below the best ordering: E3 built around the lane and the reranker; heads do not judge"
    };
    let _ = writeln!(out, "decision (runbook): heads {h}, best ordering {b}: {verdict}");
    #[expect(clippy::cast_precision_loss, reason = "counts for a report")]
    let (per_tok, per_q, per_b, mean_tokens) = (
        enc_ms / enc_tokens.max(1) as f64,
        read_ms / n.max(1) as f64,
        build_ms / n.max(1) as f64,
        reads.iter().map(|r| r.tokens).sum::<usize>() as f64 / n.max(1) as f64,
    );
    let _ = writeln!(
        out,
        "time: encode {per_tok:.2} ms per token ({enc_tokens} tokens), window build {per_b:.0} ms, two reads with 28 layers probed {per_q:.0} ms per question, reranker {rerank_ms:.0} ms per candidate; mean window {mean_tokens:.0} tokens; encoded chunks held {} MB",
        window_kb >> 10
    );
    // Data: ids, which candidates are right, each arm's rank, every head's rank.
    let mut js = String::from("{\"runbook\":\"runs/2026-11-10-E2-runbook.md\",\"questions\":[\n");
    for (i, r) in reads.iter().enumerate() {
        let f = |k: Option<usize>| k.map_or_else(|| "null".to_owned(), |x| x.to_string());
        let hr: Vec<String> = (0..d.layers).flat_map(|l| (0..d.heads).map(move |h| (l, h))).map(|(l, h)| f(head_rank(r, l, h))).collect();
        let arm = |k: &str| arms.get(k).and_then(|v| v.get(i).copied()).flatten();
        let _ = write!(
            js,
            "{}{{\"id\":\"{}\",\"fold\":\"{}\",\"window_tokens\":{},\"candidates\":{},\"right\":[{}],\"lane\":{},\"lexical\":{},\"reranker\":{},\"heads_selected\":{},\"all_heads\":{},\"head_ranks\":[{}]}}",
            if i == 0 { "" } else { ",\n" },
            r.id,
            if fold(i) == 0 { "A" } else { "B" },
            r.tokens,
            r.right.len(),
            r.right.iter().enumerate().filter(|(_, x)| **x).map(|(j, _)| j.to_string()).collect::<Vec<_>>().join(","),
            f(arm("1 lane")),
            f(arm("2 lexical")),
            f(arm("3 reranker")),
            f(arm("4 heads selected out of sample")),
            f(arm("5 all heads")),
            hr.join(",")
        );
    }
    js.push_str("\n]}\n");
    Ok((out, js))
}

/// `run e2 <stub|dev> [gpu|cpu]`.
pub fn run(args: &[String], home: &Path) -> ExitCode {
    let part = args.get(2).map_or("stub", String::as_str);
    let want = if args.get(3).map(String::as_str) == Some("cpu") { Device::Cpu } else { Device::Gpu };
    if part != "stub" && part != "dev" {
        eprintln!("run e2 <stub|dev> [gpu|cpu]");
        return ExitCode::FAILURE;
    }
    let stamp = nmtype::now_ms();
    let txt = home.join(format!("runs/E2-{part}-{stamp}.txt"));
    match e2(home, part == "stub", want) {
        Ok((text, js)) => {
            let _ = std::fs::write(&txt, &text);
            let _ = std::fs::write(home.join(format!("runs/E2-{part}-{stamp}.json")), js);
            print!("{text}");
            println!("report: {}", txt.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            let _ = std::fs::write(&txt, format!("FAULT {e}\n"));
            eprintln!("e2 {part}: {e} (report {})", txt.display());
            ExitCode::FAILURE
        }
    }
}
