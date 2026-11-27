//! E3.3 (runbook `runs/2026-11-10-E3-3-runbook.md`; design by Luis Jake Gabriel III,
//! `docs/2026-11-10-neumor-design-e3-tentacles.md`): eight tentacles, each reading its own
//! 12,000-token lens window (no crossover) with the E2 drone heads, plain and with its own
//! focus; each forwards its two best chunks; the reranker and the heads judge alone and
//! together; the drones' per-line mass picks lines inside the top chunk. Records hold question
//! ids only.
use nmengn::assembly::{Assembly, nid};
use nmengn::kvstore::{ChunkCache, DRONE_HEADS_QWEN3_0_6B as DRONES, Probe, Window, encode};
use nmengn::model::{Model, Population, RunOpts};
use nmengn::placement::Device;
use nmengn::roles;
use nmsubs::lens::{self, BUDGET};
use nmtype::{Fault, FaultKind};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Each tentacle's focus, in `lens::NAMES` order (runbook).
const FOCUS: [&str; 8] = [
    "the code that does this",
    "the lines that use these words",
    "where these names are defined",
    "the functions that call or use this",
    "what changed recently for this",
    "how this module is put together",
    "the part of the file that answers this",
    "what was found before for this",
];
const PER_TENTACLE: usize = 2;
const CACHE_BYTES: usize = 8 << 30;
const ANCHOR: &str = "Repository neuromorph-mini: a Rust workspace of 18 crates (nmsubs substrate, nmpod tentacles, nmbrain Core Brain, nmengn engine, nmapi tools).\n";

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("e3.judge", FaultKind::Execution, m)
}

/// One candidate: chunk, its drone score in its tentacle, the tentacle, its place there (0 best).
#[derive(Clone, Copy)]
struct Cand {
    chunk: usize,
    score: f32,
    place: usize,
}

struct Rec {
    id: String,
    keys: Vec<(String, usize, usize)>,
    lane: Vec<usize>,
    reach: bool,
    plain: Vec<Cand>,
    focused: Vec<Cand>,
    /// Per focused candidate: (source line, score, tokens) of its lines.
    lines: BTreeMap<usize, Vec<(usize, f32, usize)>>,
    rerank: Vec<f32>,
}

/// Rank (0 first) of the first right item of `order`.
fn first(order: &[usize], right: &dyn Fn(usize) -> bool) -> Option<usize> {
    order.iter().position(|c| right(*c))
}
#[expect(clippy::cast_precision_loss, reason = "ranks are small")]
fn rr(r: Option<usize>) -> f64 {
    r.map_or(0.0, |x| 1.0 / (x + 1) as f64)
}
/// Candidates ordered by `key` descending (ties: candidate order).
fn by(c: &[Cand], key: &dyn Fn(usize) -> f32) -> Vec<usize> {
    let mut o: Vec<usize> = (0..c.len()).collect();
    o.sort_by(|a, b| key(*b).total_cmp(&key(*a)).then(a.cmp(b)));
    o.into_iter().map(|i| c[i].chunk).collect()
}
/// The three combination rules of the runbook, over the focused candidates.
fn rule(k: usize, r: &Rec) -> Vec<usize> {
    let c = &r.focused;
    let p = |i: usize| r.rerank.get(i).copied().unwrap_or(0.0);
    let rank_of = |key: &dyn Fn(usize) -> f32| -> Vec<usize> {
        let mut o: Vec<usize> = (0..c.len()).collect();
        o.sort_by(|a, b| key(*b).total_cmp(&key(*a)).then(a.cmp(b)));
        let mut rk = vec![0; c.len()];
        for (pos, i) in o.into_iter().enumerate() {
            rk[i] = pos;
        }
        rk
    };
    match k {
        0 => {
            let (rp, rh) = (rank_of(&p), rank_of(&|i| c[i].score));
            #[expect(clippy::cast_precision_loss, reason = "ranks are small")]
            let f = |i: usize| 1.0 / (60.0 + rp[i] as f32) + 1.0 / (60.0 + rh[i] as f32);
            by(c, &f)
        }
        1 => {
            let top = c.iter().map(|x| x.score).fold(f32::NEG_INFINITY, f32::max).max(1e-9);
            by(c, &|i| p(i) + c[i].score / top)
        }
        _ => {
            #[expect(clippy::cast_precision_loss, reason = "places are small")]
            let f = |i: usize| p(i) - c[i].place as f32 * 10.0;
            by(c, &f)
        }
    }
}

#[expect(clippy::too_many_lines, reason = "one run, phases in order")]
pub fn judge(home: &Path, stub: bool, want: Device) -> Result<(String, String), Fault> {
    let mut items = crate::e2::suite(home)?;
    if stub {
        items.truncate(2);
    }
    let n = items.len();
    let mut out = String::new();
    let org = nmapi::forage::Organism::open_with(home, home, &["neuromorph-nano".to_owned()], false, &home.join("memory"))?;
    let sub = org.sub();
    let path = |i: usize| sub.files.get(sub.chunks[i].file).map_or_else(String::new, |f| f.path.replace('\\', "/"));
    let doc =
        |i: usize| sub.files.get(sub.chunks[i].file).map_or_else(String::new, |f| nmsubs::dense::doc_text(&f.path, &sub.chunks[i].text));
    let tok0 =
        nmtokz::Tokenizer::open(&nmtype::models_dir(home).join("qwen3-0.6b/files/tokenizer.json")).map_err(|e| bad(e.to_string()))?;
    let tokens: Vec<usize> = (0..sub.chunks.len()).map(|i| tok0.encode(&doc(i)).map(|v| v.len()).unwrap_or(usize::MAX)).collect();

    // Phase 1: lens windows and the lane's first choices (embedder only on the device).
    let t = Instant::now();
    let mut plans: Vec<(Vec<Vec<usize>>, Vec<usize>)> = Vec::new();
    {
        let (lane, line) = nmapi::neural::open_lane(home, &sub, want == Device::Gpu)?;
        let _ = writeln!(out, "{line}");
        for x in &items {
            let emb = lane.embedding_order(&sub, &x.q);
            let lists = lens::orders(&sub, &x.q, emb.as_deref(), &[]);
            let windows = lens::assign(&lists, &|c| tokens.get(c).copied().unwrap_or(usize::MAX), BUDGET);
            plans.push((windows, lane.order(&sub, &x.q, 3).unwrap_or_default()));
        }
    }
    let _ = writeln!(out, "phase 1 (lens windows): {:.1} s", t.elapsed().as_secs_f64());

    // Phase 2: every tentacle reads its own window, plain and focused.
    let pop = Population::open(home, "qwen3-0.6b", true, want == Device::Gpu)?;
    let dev = if want == Device::Gpu && pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let m = Model::bind(&pop, Assembly::load(&home.join("designs/assemblies"), "qwen3-reference")?)?;
    let tok = |s: &str| pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
    let anchor_ids = tok(ANCHOR)?;
    let ak = Arc::new(encode(&m, dev, None, &anchor_ids)?);
    let anchor = Arc::new(Window::build(&m, &[Arc::clone(&ak)])?);
    let mut cache = ChunkCache::new(CACHE_BYTES);
    let mut line_of: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let probe_heads = Probe::of_heads(&DRONES);
    let last = probe_heads.last_layer().unwrap_or(0);
    let stop = (nid(&format!("L{last}.score")), "y".to_owned());
    let (mut enc_tokens, mut enc_ms, mut read_ms, mut reads_n) = (0_usize, 0.0_f64, 0.0_f64, 0_usize);
    let mut recs: Vec<Rec> = Vec::new();
    for (x, (windows, lane)) in items.iter().zip(&plans) {
        let right = |c: usize| {
            let (p, ch) = (path(c), &sub.chunks[c]);
            x.keys.iter().any(|(f, a, b)| *f == p && ch.start <= *b && ch.end >= *a)
        };
        let reach = windows.iter().flatten().any(|c| right(*c));
        let (mut plain, mut focused) = (Vec::new(), Vec::new());
        let mut lines = BTreeMap::new();
        for (t_i, win) in windows.iter().enumerate() {
            if win.is_empty() {
                continue;
            }
            let mut place: Vec<usize> = win.clone();
            place.sort_by(|a, b| (path(*a), sub.chunks[*a].start).cmp(&(path(*b), sub.chunks[*b].start)));
            let mut parts = vec![Arc::clone(&ak)];
            for &c in &place {
                let key = sub.chunks[c].key.clone();
                let kv = if let Some(kv) = cache.get(&key) {
                    kv
                } else {
                    let ids = tok(&doc(c))?;
                    let t = Instant::now();
                    let kv = Arc::new(encode(&m, dev, Some(&anchor), &ids)?);
                    enc_ms += t.elapsed().as_secs_f64() * 1000.0;
                    enc_tokens += ids.len();
                    cache.put(&key, Arc::clone(&kv));
                    kv
                };
                if let std::collections::btree_map::Entry::Vacant(e) = line_of.entry(c) {
                    // Source line of every token: the path line is 0, then the chunk's lines.
                    let ids = tok(&doc(c))?;
                    let mut li = 0_usize;
                    let start = sub.chunks[c].start;
                    let mut v = Vec::with_capacity(ids.len());
                    for id in ids {
                        v.push(if li == 0 { 0 } else { start + li - 1 });
                        li += pop.tokenizer.decode(&[id]).matches('\n').count();
                    }
                    e.insert(v);
                }
                parts.push(kv);
            }
            let w = Arc::new(Window::build(&m, &parts)?);
            let t = Instant::now();
            let read = |text: &str| -> Result<(Vec<f32>, usize), Fault> {
                let q = tok(text)?;
                let probe = Arc::new(Probe::of_heads(&DRONES));
                m.forward_at(
                    dev,
                    &q,
                    w.len,
                    std::slice::from_ref(&stop),
                    &RunOpts { window: Some(Arc::clone(&w)), probe: Some(Arc::clone(&probe)), ..RunOpts::default() },
                )?;
                let mut s = vec![0.0_f32; w.len];
                for &(l, h) in &DRONES {
                    let mass = probe.get(l).ok_or_else(|| bad(format!("{}: layer {l} not read", x.id)))?;
                    for (p, v) in s.iter_mut().enumerate() {
                        *v += mass.get(h * w.len + p).copied().unwrap_or(0.0);
                    }
                }
                Ok((s, q.len()))
            };
            let calibrated = |pose: &str, null: &str| -> Result<Vec<f32>, Fault> {
                let ((a, na), (b, nb)) = (read(pose)?, read(null)?);
                #[expect(clippy::cast_precision_loss, reason = "token counts are small")]
                let (fa, fb) = (1.0 / na as f32, 1.0 / nb as f32);
                Ok(a.iter().zip(&b).map(|(x, y)| x * fa - y * fb).collect())
            };
            let sp = calibrated(&format!("\nQuestion: {}", x.q), "\nQuestion: N/A")?;
            let sf =
                calibrated(&format!("\nQuestion: {}\nFocus: {}", x.q, FOCUS[t_i]), &format!("\nQuestion: N/A\nFocus: {}", FOCUS[t_i]))?;
            read_ms += t.elapsed().as_secs_f64() * 1000.0;
            reads_n += 4;
            let chunk_scores = |s: &[f32]| -> Vec<(usize, f32)> {
                let mut v: Vec<(usize, f32)> =
                    w.parts.iter().enumerate().skip(1).map(|(k, &(a, len))| (place[k - 1], s[a..a + len].iter().sum())).collect();
                v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                v
            };
            for (k, (c, s)) in chunk_scores(&sp).into_iter().take(PER_TENTACLE).enumerate() {
                plain.push(Cand { chunk: c, score: s, place: k });
            }
            for (k, (c, s)) in chunk_scores(&sf).into_iter().take(PER_TENTACLE).enumerate() {
                focused.push(Cand { chunk: c, score: s, place: k });
                // Per-line scores of this candidate under its focus.
                let &(a, len) = w
                    .parts
                    .iter()
                    .enumerate()
                    .skip(1)
                    .find(|(k2, _)| place[k2 - 1] == c)
                    .map(|(_, x)| x)
                    .ok_or_else(|| bad("candidate not placed"))?;
                let mut per: BTreeMap<usize, (f32, usize)> = BTreeMap::new();
                for (j, li) in line_of.get(&c).map_or(&[][..], Vec::as_slice).iter().enumerate().take(len) {
                    let e = per.entry(*li).or_default();
                    e.0 += sf[a + j];
                    e.1 += 1;
                }
                lines.insert(c, per.into_iter().filter(|(li, _)| *li > 0).map(|(li, (s, t))| (li, s, t)).collect());
            }
        }
        recs.push(Rec { id: x.id.clone(), keys: x.keys.clone(), lane: lane.clone(), reach, plain, focused, lines, rerank: Vec::new() });
    }
    let held_mb = cache.bytes() >> 20;
    drop(cache);
    drop(m);
    drop(pop);

    // Phase 3: the released reranker on the focused candidates.
    let t = Instant::now();
    let mut n_docs = 0_usize;
    {
        let pop = Population::open(home, "qwen3-reranker-0.6b", true, want == Device::Gpu)?;
        let rdev = if want == Device::Gpu && pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
        let m = Model::bind(&pop, Assembly::load(&home.join("designs/assemblies"), "qwen3-reranker")?)?;
        for (r, x) in recs.iter_mut().zip(&items) {
            let docs: Vec<String> = r.focused.iter().map(|c| doc(c.chunk)).collect();
            n_docs += docs.len();
            r.rerank = roles::card_rerank_set(&m, rdev, roles::CARD_INSTRUCTION, &x.q, &docs)?;
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "counts for a report")]
    let rerank_ms = t.elapsed().as_secs_f64() * 1000.0 / n_docs.max(1) as f64;

    // Scoring.
    let right_of = |r: &Rec, c: usize| {
        let (p, ch) = (path(c), &sub.chunks[c]);
        r.keys.iter().any(|(f, a, b)| *f == p && ch.start <= *b && ch.end >= *a)
    };
    let fold = |i: usize| i % 2;
    let top_mrr = |orders: &[(usize, Vec<usize>)]| -> (usize, f64) {
        orders.iter().fold((0, 0.0), |(t, m), (i, o)| {
            let k = first(o, &|c| right_of(&recs[*i], c));
            (t + usize::from(k == Some(0)), m + rr(k))
        })
    };
    let pick_rule = |f: usize| -> usize {
        let mut best = (0, 0, 0.0);
        for k in 0..3 {
            let o: Vec<(usize, Vec<usize>)> =
                recs.iter().enumerate().filter(|(i, _)| fold(*i) == f).map(|(i, r)| (i, rule(k, r))).collect();
            let (t, m) = top_mrr(&o);
            if (t, m) > (best.1, best.2) {
                best = (k, t, m);
            }
        }
        best.0
    };
    let picks = [pick_rule(0), pick_rule(1)];
    let mut arms: BTreeMap<&str, Vec<Option<usize>>> = BTreeMap::new();
    let mut chosen: Vec<Vec<usize>> = Vec::new();
    for (i, r) in recs.iter().enumerate() {
        let rt = |c: usize| right_of(r, c);
        arms.entry("1 lane first choice").or_default().push(first(&r.lane, &rt));
        arms.entry("2 reranker alone").or_default().push(first(&by(&r.focused, &|k| r.rerank.get(k).copied().unwrap_or(0.0)), &rt));
        arms.entry("3 heads alone, plain").or_default().push(first(&by(&r.plain, &|k| r.plain[k].score), &rt));
        arms.entry("4 heads alone, focused").or_default().push(first(&by(&r.focused, &|k| r.focused[k].score), &rt));
        let o = rule(picks[1 - fold(i)], r);
        arms.entry("5 combined, rule out of sample").or_default().push(first(&o, &rt));
        chosen.push(o);
    }
    let reach = recs.iter().filter(|r| r.reach).count();
    let in_cands = recs.iter().filter(|r| r.focused.iter().any(|c| right_of(r, c.chunk))).count();
    let _ = writeln!(
        out,
        "E3.3 {} on {}: {n} development questions; right code in a window {reach}/{n}, among the focused candidates {in_cands}/{n}",
        if stub { "stub" } else { "dev" },
        dev.name()
    );
    let top_of = |k: &str| arms.get(k).map_or(0, |v| v.iter().filter(|x| **x == Some(0)).count());
    for (name, ks) in &arms {
        let top3 = ks.iter().filter(|k| k.is_some_and(|x| x < 3)).count();
        #[expect(clippy::cast_precision_loss, reason = "counts for a report")]
        let mrr = ks.iter().map(|k| rr(*k)).sum::<f64>() / n.max(1) as f64;
        let _ = writeln!(out, "{name:<32} top {}/{n}, first three {top3}/{n}, mean reciprocal rank {mrr:.3}", top_of(name));
    }
    let _ = writeln!(out, "rules chosen: on fold A R{}, on fold B R{}", picks[0] + 1, picks[1] + 1);
    // Lines inside arm 5's top chunk.
    let (mut line_hits, mut chunk_hits, mut line_tok, mut chunk_tok) = (0_usize, 0_usize, 0_usize, 0_usize);
    let mut line_rows = Vec::new();
    for (r, o) in recs.iter().zip(&chosen) {
        let Some(&c) = o.first() else {
            line_rows.push((false, 0));
            continue;
        };
        let ls = r.lines.get(&c).cloned().unwrap_or_default();
        let total: f32 = ls.iter().map(|x| x.1.max(0.0)).sum();
        let mut order = ls.clone();
        order.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut picked = Vec::new();
        let mut acc = 0.0;
        for (li, s, t) in order {
            if !picked.is_empty() && acc >= total * 0.5 {
                break;
            }
            acc += s.max(0.0);
            picked.push((li, t));
        }
        let p = path(c);
        let hit = picked.iter().any(|(li, _)| r.keys.iter().any(|(f, a, b)| *f == p && li >= a && li <= b));
        line_hits += usize::from(hit);
        chunk_hits += usize::from(right_of(r, c));
        line_tok += picked.iter().map(|x| x.1).sum::<usize>();
        chunk_tok += tokens[c];
        line_rows.push((hit, picked.len()));
    }
    #[expect(clippy::cast_precision_loss, reason = "counts for a report")]
    let (lt, ct) = (line_tok as f64 / n.max(1) as f64, chunk_tok as f64 / n.max(1) as f64);
    let _ = writeln!(
        out,
        "lines inside arm 5's top chunk (fewest lines with half the drone mass): key lines hit {line_hits}/{n} (whole chunk {chunk_hits}/{n}); mean {lt:.0} tokens against {ct:.0} for the chunk"
    );
    // Decisions (runbook).
    let (h3, h4, a2, a5) = (
        top_of("3 heads alone, plain"),
        top_of("4 heads alone, focused"),
        top_of("2 reranker alone"),
        top_of("5 combined, rule out of sample"),
    );
    let focus = if h4 >= h3 + 2 {
        "focused poses kept"
    } else if h4 + 1 >= h3 && h3 + 1 >= h4 {
        "within 1: focused poses kept only if line hits are higher (compare runs)"
    } else {
        "plain poses"
    };
    let b = a2.max(h4);
    let judge = if a5 >= b + 2 {
        "the combined judge goes to the claim run"
    } else if a5 + 1 >= b {
        "within 1: the reranker judges, heads order within tentacles"
    } else {
        "the better of reranker and focused heads judges"
    };
    let lines_v =
        if line_hits >= chunk_hits && line_tok < chunk_tok { "the coding agent gets lines" } else { "the coding agent gets chunks" };
    let _ = writeln!(
        out,
        "decision (runbook): focus: heads plain {h3}, focused {h4}: {focus}; judge: combined {a5}, best single {b}: {judge}; lines: {lines_v}"
    );
    #[expect(clippy::cast_precision_loss, reason = "counts for a report")]
    let (per_tok, per_read) = (enc_ms / enc_tokens.max(1) as f64, read_ms / reads_n.max(1) as f64);
    let _ = writeln!(
        out,
        "time: encode {per_tok:.2} ms per token ({enc_tokens} tokens), {per_read:.0} ms per read ({reads_n} reads, drones only, stop after layer {last}), reranker {rerank_ms:.0} ms per candidate; encoded chunks held at the end {held_mb} MB"
    );
    let mut js = String::from("{\"runbook\":\"runs/2026-11-10-E3-3-runbook.md\",\"questions\":[\n");
    for (i, r) in recs.iter().enumerate() {
        let f = |k: Option<usize>| k.map_or_else(|| "null".to_owned(), |x| x.to_string());
        let arm = |k: &str| arms.get(k).and_then(|v| v.get(i).copied()).flatten();
        let _ = write!(
            js,
            "{}{{\"id\":\"{}\",\"fold\":\"{}\",\"reach\":{},\"lane\":{},\"reranker\":{},\"heads_plain\":{},\"heads_focused\":{},\"combined\":{},\"line_hit\":{},\"lines\":{}}}",
            if i == 0 { "" } else { ",\n" },
            r.id,
            if fold(i) == 0 { "A" } else { "B" },
            r.reach,
            f(arm("1 lane first choice")),
            f(arm("2 reranker alone")),
            f(arm("3 heads alone, plain")),
            f(arm("4 heads alone, focused")),
            f(arm("5 combined, rule out of sample")),
            line_rows.get(i).is_some_and(|x| x.0),
            line_rows.get(i).map_or(0, |x| x.1)
        );
    }
    js.push_str("\n]}\n");
    Ok((out, js))
}
