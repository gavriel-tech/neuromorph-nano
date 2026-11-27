//! Phase A3 decisive tests, card parts (2026-11-18; lane A3; tentacle master plan
//! the 2026-11-18 tentacle master plan, 3.3, 3.6, 7, 8; architecture by Luis Jake
//! Gabriel III). Two measurement jobs, each writing raw per-piece scores for the analysis scripts in
//! `scratch/a3/` (no question text and no piece text is written; pieces by index or key):
//!
//! - `t1_periphery`: on the Mini's TC1 dump (development half only, read as
//!   `allocation_replay.rs` reads it), per question and window: the keys-only scan
//!   (`nmapi::key_scan::cal_rows` over `nmengn::drone::scan_keys`, uncompressed drone keys in RAM,
//!   posed minus content-free) for three poses (plain question; question plus "the definition of";
//!   question plus "the callers of"), and the embedder's cosine of every window piece. The exact
//!   read's tables come from the dump. Env: `A3_FROM`, `A3_N` (question indices), `A3_OUT`.
//! - `t2_rerank`: on the top 10 of G2's sent-up order from the stack dumps (list prepared by
//!   `scratch/a3/t2_prep.py`, keys and lines only; texts read here from the frozen material, in
//!   RAM): ICR (`nmapi::joint_read::read`, attention of the question to each piece, content-free
//!   subtraction, two orders) on the pieces cut to about 140 tokens (one read, about 1,500 tokens)
//!   and on the whole pieces in groups of 1,500 tokens; the judge (Qwen3-Reranker) on each piece cut
//!   to about 128 tokens, raw and content-free calibrated; the embedder's cosine. Env: `A3_IN`,
//!   `A3_OUT`, `A3_FROM`, `A3_N`.
//!
//! Run (card, under `scratch/LOCKS/card.lock`):
//! `cargo test --release -p nmlab --test a3_decisive -- --ignored --nocapture t1_periphery`.
use nmapi::key_scan::{Job, Keys};
use nmapi::joint_read::{self, Piece};
use nmapi::scan_dump;
use nmapi::tentacle_pass;
use nmengn::kvstore::{self, Window};
use nmengn::placement::Device;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

type R<T> = Result<T, String>;

const MINI_ANCHOR: &str = "Repository neuromorph-mini: a Rust workspace of 18 crates (nmsubs substrate, nmpod tentacles, nmbrain Core Brain, nmengn engine, nmapi tools).\n";
const DUMP: &str = "1794288995065";
const POSES: [(&str, &str); 3] = [("plain", ""), ("definition", "the definition of"), ("callers", "the callers of")];

/// One block of the judged material: its name, path, kind, text and the pieces in it.
type Block = (String, String, String, String, Vec<(String, String, usize, usize)>);

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// The development half of the Mini's CX1 questions: (id, text) by index, as the locked harness
/// orders it (even positions only; the sealed half is never kept).
fn dev_half(mini: &Path) -> R<Vec<(String, String)>> {
    const TYPES: [&str; 5] = ["behaviour", "name", "value", "explain", "relation"];
    let mut out = Vec::new();
    for area in ["context", "engine", "tools"] {
        let p = mini.join(format!("tests/capability-gates/cx1-context/questions-{area}.json"));
        let v = nmjson::parse(&std::fs::read(&p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let mut qs: Vec<(usize, String, String)> = Vec::new();
        for x in v.get("questions").and_then(|y| y.array()).map_err(|e| e.to_string())? {
            let s = |k: &str| x.get(k).and_then(|y| y.string()).map(str::to_owned).map_err(|e| e.to_string());
            let t = s("type")?;
            qs.push((TYPES.iter().position(|y| *y == t).ok_or("type")?, s("id")?, s("q")?));
        }
        qs.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
        out.extend(qs.into_iter().enumerate().filter(|(pos, _)| pos % 2 == 0).map(|(_, (_, id, q))| (id, q)));
    }
    Ok(out)
}

fn floats(v: &[f32]) -> String {
    let mut s = String::from("[");
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        if x.is_finite() {
            let _ = write!(s, "{x}");
        } else {
            s.push_str("null");
        }
    }
    s.push(']');
    s
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0.0_f32, 0.0_f32, 0.0_f32);
    for (x, y) in a.iter().zip(b) {
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    if na > 0.0 && nb > 0.0 { d / (na.sqrt() * nb.sqrt()) } else { f32::NAN }
}

fn out_dir(default: &str) -> PathBuf {
    let home = nmtype::nano_root();
    let d = std::env::var("A3_OUT").map_or_else(|_| home.join(default), PathBuf::from);
    let _ = std::fs::create_dir_all(&d);
    d
}

#[test]
#[ignore = "card job: needs the reader, the embedder and the Mini's TC1 dump"]
fn t1_periphery() {
    let r = t1();
    if let Err(e) = r {
        panic!("t1: {e}");
    }
}

#[expect(clippy::too_many_lines, reason = "one measurement, in the order it runs")]
fn t1() -> R<()> {
    let home = nmtype::nano_root();
    let mini = home.join("..").join("neuromorph-mini");
    let dir = mini.join("scratch").join("tc1-dump").join(DUMP);
    let out = out_dir("scratch/a3/t1");
    let (from, n) = (env_usize("A3_FROM", 0), env_usize("A3_N", 2));
    let items = dev_half(&mini)?;
    let sub = scan_dump::load_substrate(&dir).map_err(|e| e.message)?;
    let t_load = Instant::now();
    let model = nmapi::neural::model(&home).map_err(|e| e.message)?;
    let emb = nmapi::neural::embedder_model(&home).map_err(|e| e.message)?;
    let dev = if model.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let edev = if emb.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    eprintln!("models on {} and {} in {:.1} s", dev.name(), edev.name(), t_load.elapsed().as_secs_f64());
    let d = model.pop.dims;
    let group = d.heads / d.kv_heads.max(1);
    let sets = tentacle_pass::drones(&home);
    let last = sets.last_layer();
    let mut all_heads = sets.heads.clone();
    for v in sets.tentacles.values() {
        all_heads.extend(v.iter().copied());
    }
    all_heads.sort_unstable();
    all_heads.dedup();
    let tok = |s: &str| model.pop.tokenizer.encode(s).map_err(|e| e.to_string());
    let ak = Arc::new(kvstore::encode_through(&model, dev, None, &tok(MINI_ANCHOR)?, last).map_err(|e| e.message)?);
    let aw = Arc::new(Window::build(&model, &[Arc::clone(&ak)]).map_err(|e| e.message)?);
    let keys = Keys::new(&all_heads, group, d.head_dim, 8 << 30);
    let mut docs: Vec<(String, String)> = Vec::new();
    let mut summary = format!(
        "T1 keys-only periphery on the TC1 dump {DUMP}: drone set {} heads ({}), {} pairs, last layer {last}, {} bytes per token\n",
        all_heads.len(),
        sets.source,
        keys.pairs.len(),
        keys.bytes_per_token
    );
    for idx in scan_dump::indices(&dir).into_iter().skip(from).take(n) {
        let Some((id, text)) = items.get(idx) else { continue };
        let (q, scan) = scan_dump::load_question(&dir, idx, text).map_err(|e| e.message)?;
        if &q.id != id {
            return Err(format!("dump question {idx} is {} not {id}", q.id));
        }
        // 1. Packets of this question's window pieces (held ones reused).
        let t0 = Instant::now();
        for w in &scan.windows {
            for c in &w.place {
                if let Some(ch) = sub.chunks.get(*c)
                    && !docs.iter().any(|x| x.0 == ch.key)
                {
                    docs.push((ch.key.clone(), sub.files.get(ch.file).map(|f| nmsubs::dense::doc_text(&f.path, &ch.text)).unwrap_or_default()));
                }
            }
        }
        keys.pass(&model, dev, &aw, &docs, sub.version);
        let st = keys.status();
        let enc_s = t0.elapsed().as_secs_f64();
        // 2. Question paths per window and pose (posed and content-free), then the scan.
        let t1 = Instant::now();
        let mut jobs: Vec<Vec<Job>> = (0..POSES.len()).map(|_| Vec::new()).collect();
        for w in &scan.windows {
            let lens = nmsubs::lens::NAMES.get(w.tentacle).copied().unwrap_or("lens");
            let heads = sets.for_tentacle(lens);
            let mut packets = vec![Arc::new(nmengn::drone::DroneKeys::from_chunk(&ak, &keys.pairs, d.kv_heads, d.head_dim).map_err(|e| e.message)?)];
            for c in &w.place {
                let key = sub.chunks.get(*c).map(|ch| ch.key.clone()).unwrap_or_default();
                packets.push(keys.get(&key).ok_or_else(|| format!("question {idx}: no packet for chunk {c}"))?);
            }
            for (p, (_, focus)) in POSES.iter().enumerate() {
                let (posed, null) = if focus.is_empty() {
                    (format!("\nQuestion: {text}"), "\nQuestion: N/A".to_owned())
                } else {
                    (format!("\nQuestion: {text}\nFocus: {focus}"), format!("\nQuestion: N/A\nFocus: {focus}"))
                };
                let gate = (dev == Device::Gpu).then(|| nmengn::placement::gpu_gate().enter("a3 question path", true));
                let qp = nmengn::drone::question_path(&model, dev, &aw, &tok(&posed)?, w.len, heads).map_err(|e| e.message)?;
                let qn = nmengn::drone::question_path(&model, dev, &aw, &tok(&null)?, w.len, heads).map_err(|e| e.message)?;
                drop(gate);
                if let Some(j) = jobs.get_mut(p) {
                    j.push(Job { packets: packets.clone(), parts: w.parts.clone(), len: w.len, posed: qp, null: qn });
                }
            }
        }
        let qpath_s = t1.elapsed().as_secs_f64();
        let mut tables = Vec::new();
        let mut score_ms = Vec::new();
        for j in &jobs {
            let (t, ms) = nmapi::key_scan::score_windows(j, d.head_dim, group, d.theta);
            tables.push(t);
            score_ms.push(ms);
        }
        // 3. The embedder's cosine of every window piece.
        let t2 = Instant::now();
        let qv = nmengn::roles::card_embed_on(&emb, edev, &format!("Instruct: {}\nQuery:{text}", nmsubs::dense::QUERY_TASK)).map_err(|e| e.message)?;
        // Batched as the served lane embeds (first `MAX_TOKENS` tokens of each piece).
        let mut want: Vec<usize> = scan.windows.iter().flat_map(|w| w.place.iter().copied()).collect();
        want.sort_unstable();
        want.dedup();
        let texts: Vec<String> = want
            .iter()
            .map(|c| sub.chunks.get(*c).map(|ch| nmsubs::dense::embed_text(sub.files.get(ch.file).map_or("", |f| f.path.as_str()), ch)).unwrap_or_default())
            .collect();
        let vs = nmengn::roles::card_embed_batch(&emb, &texts, nmengn::roles::MAX_TOKENS, 4096).map_err(|e| e.message)?;
        let cos: std::collections::BTreeMap<usize, f32> = want.iter().zip(&vs).map(|(c, v)| (*c, cosine(&qv, v))).collect();
        let emb_s = t2.elapsed().as_secs_f64();
        // 4. Record (indices only).
        let mut js = format!(
            "{{\"index\":{idx},\"id\":\"{}\",\"gold\":{:?},\"enc_s\":{enc_s},\"encoded_tokens\":{},\"qpath_s\":{qpath_s},\"score_ms\":{},\"embed_s\":{emb_s},\"exact_scan_s\":{},\"windows\":[",
            q.id,
            q.gold,
            st.tokens,
            floats(&score_ms.iter().map(|x| *x as f32).collect::<Vec<_>>()),
            q.scan_s
        );
        for (wi, w) in scan.windows.iter().enumerate() {
            if wi > 0 {
                js.push(',');
            }
            let _ = write!(
                js,
                "{{\"tentacle\":{},\"len\":{},\"place\":{:?},\"parts\":{:?},\"exact_posed\":{},\"exact_null\":{},\"cos\":{}",
                w.tentacle,
                w.len,
                w.place,
                w.parts.iter().map(|p| p.1).collect::<Vec<_>>(),
                floats(&w.cal_posed),
                floats(&w.cal_null),
                floats(&w.place.iter().map(|c| cos.get(c).copied().unwrap_or(f32::NAN)).collect::<Vec<_>>())
            );
            for (p, (name, _)) in POSES.iter().enumerate() {
                let tb = tables.get(p).and_then(|t| t.get(wi)).ok_or("no table")?;
                let (kp, kn) = tb.as_ref().map_err(|e| e.message.clone())?;
                let _ = write!(js, ",\"{name}_posed\":{},\"{name}_null\":{}", floats(kp), floats(kn));
            }
            js.push('}');
        }
        js.push_str("]}\n");
        std::fs::write(out.join(format!("q{idx:02}.json")), js).map_err(|e| e.to_string())?;
        let line = format!(
            "q{idx:02} {}: packets {} tokens encoded in {enc_s:.1} s; question paths {qpath_s:.2} s; scan {:.0} ms ({} windows x {} poses); embedder {emb_s:.1} s",
            q.id,
            st.tokens,
            score_ms.iter().sum::<f64>(),
            scan.windows.len(),
            POSES.len()
        );
        eprintln!("{line}");
        summary.push_str(&line);
        summary.push('\n');
    }
    let ram = nmffi::dynlib::process_memory().map_or_else(|e| e, |(peak, now)| format!("RAM peak {} MB, now {} MB", peak >> 20, now >> 20));
    let (kb, kn, _) = keys.held();
    let _ = writeln!(summary, "store {kn} packets, {} MB; {ram}", kb >> 20);
    std::fs::write(out.join(format!("run-{from}-{n}-{}.txt", nmtype::now_ms())), summary).map_err(|e| e.to_string())?;
    Ok(())
}

/// The first lines of `text` holding at least `want` tokens (whole lines; at least one line).
fn cut(m: &nmengn::model::Model, text: &str, want: usize) -> String {
    let mut out = String::new();
    for line in text.lines() {
        out.push_str(line);
        out.push('\n');
        if m.pop.tokenizer.encode(&out).map_or(0, |v| v.len()) >= want {
            break;
        }
    }
    out
}

#[test]
#[ignore = "card job: needs the reader, the judge, the embedder and scratch/a3/t2-in.txt"]
fn t2_rerank() {
    if let Err(e) = t2() {
        panic!("t2: {e}");
    }
}

#[expect(clippy::too_many_lines, reason = "one measurement, in the order it runs")]
fn t2() -> R<()> {
    let home = nmtype::nano_root();
    let input = std::env::var("A3_IN").map_or_else(|_| home.join("scratch/a3/t2-in.txt"), PathBuf::from);
    let out = out_dir("scratch/a3/t2");
    let (from, n) = (env_usize("A3_FROM", 0), env_usize("A3_N", 2));
    let model = nmapi::neural::model(&home).map_err(|e| e.message)?;
    let judge = nmapi::neural::reranker_model(&home).map_err(|e| e.message)?;
    let emb = nmapi::neural::embedder_model(&home).map_err(|e| e.message)?;
    let dev = if model.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let jdev = if judge.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let edev = if emb.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let sets = tentacle_pass::drones(&home);
    let (heads, last) = (sets.heads.clone(), sets.last_layer());
    // Input: blocks "Q <set> <id> <root>" then "P <key>\t<path>\t<first>\t<last>" lines, then a
    // line "T <question text>" (the tuning sets' question, held in RAM only).
    let text = std::fs::read_to_string(&input).map_err(|e| format!("{}: {e}", input.display()))?;
    let mut blocks: Vec<Block> = Vec::new();
    for line in text.lines() {
        if let Some(r) = line.strip_prefix("Q ") {
            let mut it = r.splitn(3, ' ');
            let (s, id, root) = (it.next().unwrap_or(""), it.next().unwrap_or(""), it.next().unwrap_or(""));
            blocks.push((s.to_owned(), id.to_owned(), root.to_owned(), String::new(), Vec::new()));
        } else if let Some(r) = line.strip_prefix("T ") {
            if let Some(b) = blocks.last_mut() {
                b.3 = r.to_owned();
            }
        } else if let Some(r) = line.strip_prefix("P ") {
            let f: Vec<&str> = r.split('\t').collect();
            if let (Some(b), [k, p, a, z]) = (blocks.last_mut(), f.as_slice()) {
                b.4.push(((*k).to_owned(), (*p).to_owned(), a.parse().unwrap_or(1), z.parse().unwrap_or(1)));
            }
        }
    }
    let mut summary = format!("T2 rerank arms: {} questions in the input; from {from}, {n}\n", blocks.len());
    for (set, id, root, question, cands) in blocks.into_iter().skip(from).take(n) {
        let root = PathBuf::from(&root);
        let mut pieces: Vec<Piece> = Vec::new();
        for (ci, (_, path, first, lastl)) in cands.iter().enumerate() {
            let body = std::fs::read_to_string(root.join(path)).unwrap_or_default();
            let lines: Vec<&str> = body.lines().collect();
            let a = first.saturating_sub(1).min(lines.len());
            let z = (*lastl).min(lines.len()).max(a);
            let t = lines.get(a..z).map(|s| s.join("\n")).unwrap_or_default();
            pieces.push(Piece { chunk: ci, path: path.clone(), start: *first, end: *lastl, text: t, evidence: joint_read::Evidence::default() });
        }
        let cut_pieces: Vec<Piece> = pieces.iter().map(|p| Piece { text: cut(&model, &p.text, 140), ..p.clone() }).collect();
        // (b) ICR on the cut pieces, one read (about 1,500 tokens), two orders, label tails too.
        let t0 = Instant::now();
        let on = |k: &str| std::env::var(k).is_ok_and(|v| v == "on");
        let icr_cut = joint_read::read(&model, dev, &question, &cut_pieces, &heads, last, usize::MAX, on("A3_YESNO"), true).map_err(|e| e.message)?;
        let icr_cut_s = t0.elapsed().as_secs_f64();
        // (b') ICR on whole pieces in groups of 1,500 tokens with a final read of the winners.
        let t1 = Instant::now();
        let icr_whole = if on("A3_WHOLE") {
            joint_read::read(&model, dev, &question, &pieces, &heads, last, 1_500, false, true).map_err(|e| e.message)?
        } else {
            joint_read::JointOut { mass: joint_read::Ranked { order: (0..pieces.len()).collect(), group_score: vec![0.0; pieces.len()] }, calibrated: joint_read::Ranked { order: (0..pieces.len()).collect(), group_score: vec![0.0; pieces.len()] }, ..joint_read::JointOut::default() }
        };
        let icr_whole_s = t1.elapsed().as_secs_f64();
        // (c) The judge on each piece cut to about 128 tokens: raw P(yes), and the content-free
        // calibrated label log-odds (`judge_edge`).
        let docs: Vec<String> = pieces.iter().map(|p| format!("{}:{}-{}\n{}", p.path, p.start, p.end, cut(&judge, &p.text, 128))).collect();
        let t2 = Instant::now();
        let pairs: Vec<(&str, &str)> = docs.iter().map(|d| (question.as_str(), d.as_str())).collect();
        let raw = nmengn::roles::card_rerank_pairs(&judge, jdev, nmengn::roles::CODE_INSTRUCTION, &pairs).map_err(|e| e.message)?;
        let judge_s = t2.elapsed().as_secs_f64();
        let edge: Vec<nmapi::neural::EdgePiece> =
            cands.iter().zip(&docs).map(|(c, d)| nmapi::neural::EdgePiece { key: c.0.clone(), posed: d.clone(), plain: d.clone() }).collect();
        let t3 = Instant::now();
        let cal = if on("A3_JUDGECAL") { nmapi::neural::judge_edge(&judge, &question, &edge).map_err(|e| e.message)?.0 } else { vec![0.0; edge.len()] };
        let judge_cal_s = t3.elapsed().as_secs_f64();
        // (d) inputs: the embedder's cosine of each whole piece.
        let t4 = Instant::now();
        let qv = nmengn::roles::card_embed_on(&emb, edev, &format!("Instruct: {}\nQuery:{question}", nmsubs::dense::QUERY_TASK)).map_err(|e| e.message)?;
        let texts: Vec<String> = pieces.iter().map(|p| nmsubs::dense::doc_text(&p.path, &p.text)).collect();
        let vs = nmengn::roles::card_embed_batch(&emb, &texts, nmengn::roles::MAX_TOKENS, 4096).map_err(|e| e.message)?;
        let cos: Vec<f32> = vs.iter().map(|v| cosine(&qv, v)).collect();
        let emb_s = t4.elapsed().as_secs_f64();
        let ord = |v: &[usize]| format!("{v:?}");
        let js = format!(
            "{{\"set\":\"{set}\",\"id\":\"{id}\",\"keys\":[{}],\"icr_cut\":{{\"mass\":{},\"cal\":{},\"yes_no\":{},\"tokens\":{},\"s\":{icr_cut_s}}},\"icr_whole\":{{\"mass_order\":{},\"cal_order\":{},\"mass\":{},\"cal\":{},\"groups\":{},\"tokens\":{},\"s\":{icr_whole_s}}},\"judge_raw\":{},\"judge_cal\":{},\"judge_s\":{judge_s},\"judge_cal_s\":{judge_cal_s},\"cos\":{},\"embed_s\":{emb_s}}}\n",
            cands.iter().map(|c| format!("\"{}\"", c.0.replace('\\', "/").replace('"', ""))).collect::<Vec<_>>().join(","),
            floats(&icr_cut.mass.group_score),
            floats(&icr_cut.calibrated.group_score),
            floats(&icr_cut.yes_no.group_score),
            icr_cut.tokens,
            ord(&icr_whole.mass.order),
            ord(&icr_whole.calibrated.order),
            floats(&icr_whole.mass.group_score),
            floats(&icr_whole.calibrated.group_score),
            icr_whole.groups,
            icr_whole.tokens,
            floats(&raw),
            floats(&cal),
            floats(&cos)
        );
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(out.join("t2.jsonl")).map_err(|e| e.to_string())?;
        std::io::Write::write_all(&mut f, js.as_bytes()).map_err(|e| e.to_string())?;
        let line = format!(
            "{set} {id}: icr cut {icr_cut_s:.2} s ({} tokens over 2 orders), icr whole {icr_whole_s:.2} s ({} groups), judge {judge_s:.2} s, judge calibrated {judge_cal_s:.2} s, embedder {emb_s:.2} s",
            icr_cut.tokens, icr_whole.groups
        );
        eprintln!("{line}");
        summary.push_str(&line);
        summary.push('\n');
    }
    std::fs::write(out.join(format!("run-{from}-{n}-{}.txt", nmtype::now_ms())), summary).map_err(|e| e.to_string())?;
    Ok(())
}
