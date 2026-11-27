//! B1 (tentacle master plan section 8; Phase 0 audit B1; lane M, 2026-11-19): the Core Brain's
//! schedule on what G2 sends up. 5 questions, the right piece plus the 7 hardest wrong (G2's
//! highest-ranked wrong pieces); one packed read of the 8 by the judge (`neural::judge_close`, the
//! cheap wide stage), then full depth over the best 4: judge duels in both orders, each less its
//! content-free label (`nmbrain::chain::duel` with `nmapi::brain_side::JudgeReader`, labels held at
//! idle before the timed call), evidence added on the log-odds scale. Pass: the right piece first
//! on at least 4 of 5, under 30 s each.
//!
//! Data: the Nano's own dump (the CX1 dump is not in the Nano): A3 T2's G2 top 10 per question
//! (`scratch/a3/t2-in.txt`, `t2-meta.json` flags). Selection, fixed before the run: tuning sets only
//! (`closure`, `dev`), in file order, the first 5 questions whose right piece is in G2's top 8 but
//! not first. Writes keys and numbers only to `scratch/lane-m/b1.jsonl`.
//!
//! Run (card, under `scratch/LOCKS/card.lock`):
//! `cargo test --release -p nmlab --test b1_intake -- --ignored --nocapture`.
use nmbrain::chain::{Labels, duel};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

type R<T> = Result<T, String>;

#[test]
#[ignore = "card job: needs the judge and scratch/a3/t2-in.txt"]
fn b1_core_brain_schedule() {
    match run() {
        Ok((first, n, worst)) => {
            eprintln!("B1: first {first} of {n}, slowest {worst:.1} s");
            assert!(first >= 4 && worst < 30.0, "B1 not met: first {first} of {n}, slowest {worst:.1} s");
        }
        Err(e) => panic!("b1: {e}"),
    }
}

/// B1 with the chain's real routing (T-083 rerun, 2026-11-19): the same five questions and eight
/// candidates; G2's gate from the dump; the pieces' starting evidence the wide judge read of the 8;
/// then `brain_side::intake`, whose job map sends a code close call to the embedder (its meaning read
/// of each piece: cosine with the question less the mean over the content-free questions, over the
/// candidates' noise band) and a prose one to the judge. Control arms printed: the wide read alone,
/// the embedder's meaning read alone. Writes `scratch/lane-m/b1-routed.jsonl`.
#[test]
#[ignore = "card job: needs the judge, the embedder and scratch/a3/t2-in.txt"]
fn b1_routed_chain() {
    match run_routed() {
        Ok((first, n, worst)) => {
            eprintln!("B1 routed: first {first} of {n}, slowest {worst:.1} s");
            assert!(first >= 4 && worst < 30.0, "B1 routed not met: first {first} of {n}, slowest {worst:.1} s");
        }
        Err(e) => panic!("b1 routed: {e}"),
    }
}

#[allow(clippy::type_complexity)]
fn chosen_five(home: &std::path::Path) -> R<(Vec<(String, String, String, String, Vec<(String, String, usize, usize)>)>, nmjson::Value)> {
    let input = std::fs::read_to_string(home.join("scratch/a3/t2-in.txt")).map_err(|e| e.to_string())?;
    let meta = nmjson::parse(&std::fs::read(home.join("scratch/a3/t2-meta.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let mut blocks: Vec<(String, String, String, String, Vec<(String, String, usize, usize)>)> = Vec::new();
    for line in input.lines() {
        if let Some(r) = line.strip_prefix("Q ") {
            let mut it = r.splitn(3, ' ');
            let (s, id, root) = (it.next().unwrap_or(""), it.next().unwrap_or(""), it.next().unwrap_or(""));
            blocks.push((s.to_owned(), id.to_owned(), root.to_owned(), String::new(), Vec::new()));
        } else if let Some(r) = line.strip_prefix("T ") {
            if let Some(b) = blocks.last_mut() {
                r.clone_into(&mut b.3);
            }
        } else if let Some(r) = line.strip_prefix("P ") {
            let f: Vec<&str> = r.split('\t').collect();
            if let (Some(b), [k, p, a, z]) = (blocks.last_mut(), f.as_slice()) {
                b.4.push(((*k).to_owned(), (*p).to_owned(), a.parse().unwrap_or(1), z.parse().unwrap_or(1)));
            }
        }
    }
    let mut chosen = Vec::new();
    for b in blocks {
        if !(b.0 == "closure" || b.0 == "dev") {
            continue;
        }
        let flags: Vec<bool> = meta
            .get(&format!("{}/{}", b.0, b.1))
            .and_then(|m| m.get("flags"))
            .and_then(nmjson::Value::array)
            .map(|a| a.iter().map(|x| matches!(x, nmjson::Value::Bool(true))).collect())
            .unwrap_or_default();
        let Some(gold) = flags.iter().take(8).position(|f| *f) else { continue };
        if gold == 0 {
            continue;
        }
        let mut cands = vec![b.4[gold].clone()];
        cands.extend(b.4.iter().zip(&flags).filter(|(_, f)| !**f).map(|(c, _)| c.clone()).take(7));
        chosen.push((b.0, b.1, b.2, b.3, cands));
        if chosen.len() == 5 {
            break;
        }
    }
    Ok((chosen, meta))
}

fn run_routed() -> R<(usize, usize, f64)> {
    use nmbrain::chain::{MicroModel, PieceEvidence};
    let home = nmtype::nano_root();
    let (chosen, meta) = chosen_five(&home)?;
    nmengn::gpu::HALF_MATH.store(true, std::sync::atomic::Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, std::sync::atomic::Ordering::Relaxed);
    let judge = nmapi::neural::reranker_model(&home).map_err(|e| e.message)?;
    let emb = nmapi::neural::embedder_model(&home).map_err(|e| e.message)?;
    let reader = nmapi::neural::model(&home).map_err(|e| e.message)?;
    let drones = nmapi::tentacle_pass::drones(&home);
    let dev = if emb.pop.gpu.is_some() { nmengn::placement::Device::Gpu } else { nmengn::placement::Device::Cpu };
    eprintln!("judge on {}, embedder on {}", if judge.pop.gpu.is_some() { "card" } else { "processor" }, if emb.pop.gpu.is_some() { "card" } else { "processor" });
    let enc = |t: &str| nmengn::roles::card_embed_on(&emb, dev, t).map_err(|e| e.message);
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let (mut first, mut wide_first, mut emb_first, mut worst) = (0_usize, 0_usize, 0_usize, 0.0_f64);
    let mut lines = String::new();
    for (set, id, root, question, cands) in &chosen {
        let root = PathBuf::from(root);
        let gate = meta.get(&format!("{set}/{id}")).and_then(|m| m.get("gate")).and_then(nmjson::Value::string).unwrap_or("low").to_owned();
        let mut order: Vec<usize> = (0..cands.len()).collect();
        order.sort_by(|a, b| cands[*a].0.cmp(&cands[*b].0));
        let texts: BTreeMap<String, String> = cands.iter().map(|(k, p, a, z)| (k.clone(), piece(&root, p, *a, *z))).collect();
        let t0 = Instant::now();
        let docs: Vec<String> = order
            .iter()
            .map(|&i| {
                let (k, p, a, z) = &cands[i];
                nmapi::neural::close_doc(&judge, &format!("{p}:{a}-{z}"), texts.get(k).map_or("", String::as_str))
            })
            .collect();
        let z = nmapi::neural::judge_close(&judge, question, &docs).map_err(|e| e.message)?;
        // The embedder's meaning read, as the served lane poses it.
        let task = nmsubs::dense::task_for(question);
        let q = enc(&format!("Instruct: {task}\nQuery:{question}"))?;
        let nulls: Vec<Vec<f32>> = nmsubs::dense::CONTENT_FREE.iter().map(|f| enc(&format!("Instruct: {task}\nQuery:{f}"))).collect::<R<_>>()?;
        let mut meant: BTreeMap<String, f32> = BTreeMap::new();
        for &i in &order {
            let (k, p, ..) = &cands[i];
            let d = enc(&nmsubs::dense::doc_text_with(p, "", texts.get(k).map_or("", String::as_str)))?;
            #[expect(clippy::cast_precision_loss, reason = "three forms")]
            let cf = nulls.iter().map(|n| dot(n, &d)).sum::<f32>() / nulls.len() as f32;
            meant.insert(k.clone(), dot(&q, &d) - cf);
        }
        let band = nmapi::coordinator::noise_band(&meant.values().copied().collect::<Vec<f32>>()).max(1e-3);
        let pieces: Vec<PieceEvidence> = order
            .iter()
            .zip(&z)
            .map(|(&i, v)| PieceEvidence { key: cands[i].0.clone(), prose: nmtype::is_document(&cands[i].1), logodds: *v })
            .collect();
        let value = |_j: nmbrain::chain::Job, k: &str| meant.get(k).copied();
        let mut emb_hop = nmapi::brain_side::ValueHop { model: MicroModel::Embedder, value: &value, scale: band };
        let text_of = |k: &str| texts.get(k).map(|t| (t.clone(), t.clone()));
        let mut jr = nmapi::brain_side::JudgeReader { model: &judge, question: question.clone(), text: &text_of };
        let mut labels = Labels::default();
        let mut judge_hop = nmbrain::chain::DuelHop { reader: &mut jr, labels: &mut labels, band: 0.0, reads: 0 };
        let piece_of = |k: &str| -> Option<nmapi::joint_read::Piece> {
            let (_, p, a, z) = cands.iter().find(|c| c.0 == k)?;
            Some(nmapi::joint_read::Piece { path: p.clone(), start: *a, end: *z, text: texts.get(k)?.clone(), ..Default::default() })
        };
        let mut reader_hop = nmapi::brain_side::ReaderHop { model: &reader, question: question.clone(), heads: drones.heads.clone(), last: drones.last_layer(), piece: &piece_of };
        let deadline = t0 + std::time::Duration::from_secs(30);
        // Each arm alone for the trace: the reader's joint read over the best 4 of the wide read.
        let mut best4: Vec<usize> = (0..z.len()).collect();
        best4.sort_by(|a, b| z[*b].total_cmp(&z[*a]));
        best4.truncate(4);
        let all: Vec<PieceEvidence> = order.iter().map(|&i| PieceEvidence { key: cands[i].0.clone(), prose: false, logodds: 0.0 }).collect();
        let named: Vec<&PieceEvidence> = all.iter().collect();
        let joint_all = nmbrain::chain::HopReader::read(&mut reader_hop, nmbrain::chain::Job::Joint, &named).unwrap_or_default();
        let reader_won = (0..joint_all.len()).max_by(|a, b| joint_all[*a].total_cmp(&joint_all[*b])) == Some(order.iter().position(|&i| i == 0).unwrap_or(usize::MAX));
        eprintln!("  reader's joint read of all 8 alone: first {reader_won}");
        let r = nmapi::brain_side::intake(&gate, pieces, &mut [&mut emb_hop, &mut judge_hop, &mut reader_hop], &|_| None, None, deadline).map_err(|f| f.to_string())?;
        let s = t0.elapsed().as_secs_f64();
        let gold_at = order.iter().position(|&i| i == 0).unwrap_or(usize::MAX);
        let won = r.out.order.first() == Some(&gold_at);
        let top = |v: &[f32]| (0..v.len()).max_by(|a, b| v[*a].total_cmp(&v[*b]).then(b.cmp(a))).unwrap_or(0);
        let wide_won = top(&z) == gold_at;
        let mv: Vec<f32> = order.iter().map(|&i| meant.get(&cands[i].0).copied().unwrap_or(f32::NEG_INFINITY)).collect();
        let emb_won = top(&mv) == gold_at;
        first += usize::from(won);
        wide_first += usize::from(wide_won);
        emb_first += usize::from(emb_won);
        worst = worst.max(s);
        let hops: Vec<String> = r.out.hops.iter().map(|h| format!("{}:{}", h.job.name(), h.model.map_or("none", MicroModel::name))).collect();
        eprintln!("{set} {id} (gate {gate}): first {won} (wide {wide_won}, embedder {emb_won}); hops {hops:?}; stop {:?}; bound {:.2}; {s:.2} s", r.out.stop, r.bound);
        lines.push_str(&format!("{{\"set\":\"{set}\",\"id\":\"{id}\",\"gate\":\"{gate}\",\"first\":{won},\"wide_first\":{wide_won},\"embedder_first\":{emb_won},\"s\":{s:.3},\"hops\":{}}}\n", hops.len()));
    }
    let out = home.join("scratch/lane-m");
    std::fs::write(out.join("b1-routed.jsonl"), lines).map_err(|e| e.to_string())?;
    eprintln!("B1 routed controls: wide alone {wide_first} of {n}, embedder alone {emb_first} of {n}", n = chosen.len());
    Ok((first, chosen.len(), worst))
}

fn piece(root: &std::path::Path, path: &str, first: usize, last: usize) -> String {
    let body = std::fs::read_to_string(root.join(path)).unwrap_or_default();
    let ls: Vec<&str> = body.lines().collect();
    let a = first.saturating_sub(1).min(ls.len());
    let z = last.min(ls.len()).max(a);
    ls.get(a..z).map(|s| s.join("\n")).unwrap_or_default()
}

#[allow(clippy::type_complexity)]
fn run() -> R<(usize, usize, f64)> {
    let home = nmtype::nano_root();
    let input = std::fs::read_to_string(home.join("scratch/a3/t2-in.txt")).map_err(|e| e.to_string())?;
    let meta = nmjson::parse(&std::fs::read(home.join("scratch/a3/t2-meta.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let out = home.join("scratch/lane-m");
    let _ = std::fs::create_dir_all(&out);
    nmengn::gpu::HALF_MATH.store(true, std::sync::atomic::Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, std::sync::atomic::Ordering::Relaxed);
    // Blocks: "Q <set> <id> <root>", "T <question>", "P <key>\t<path>\t<first>\t<last>".
    let mut blocks: Vec<(String, String, String, String, Vec<(String, String, usize, usize)>)> = Vec::new();
    for line in input.lines() {
        if let Some(r) = line.strip_prefix("Q ") {
            let mut it = r.splitn(3, ' ');
            let (s, id, root) = (it.next().unwrap_or(""), it.next().unwrap_or(""), it.next().unwrap_or(""));
            blocks.push((s.to_owned(), id.to_owned(), root.to_owned(), String::new(), Vec::new()));
        } else if let Some(r) = line.strip_prefix("T ") {
            if let Some(b) = blocks.last_mut() {
                r.clone_into(&mut b.3);
            }
        } else if let Some(r) = line.strip_prefix("P ") {
            let f: Vec<&str> = r.split('\t').collect();
            if let (Some(b), [k, p, a, z]) = (blocks.last_mut(), f.as_slice()) {
                b.4.push(((*k).to_owned(), (*p).to_owned(), a.parse().unwrap_or(1), z.parse().unwrap_or(1)));
            }
        }
    }
    let mut chosen = Vec::new();
    for b in blocks {
        if !(b.0 == "closure" || b.0 == "dev") {
            continue;
        }
        let flags: Vec<bool> = meta
            .get(&format!("{}/{}", b.0, b.1))
            .and_then(|m| m.get("flags"))
            .and_then(nmjson::Value::array)
            .map(|a| a.iter().map(|x| matches!(x, nmjson::Value::Bool(true))).collect())
            .unwrap_or_default();
        let Some(gold) = flags.iter().take(8).position(|f| *f) else { continue };
        if gold == 0 {
            continue;
        }
        let mut cands = vec![b.4[gold].clone()];
        cands.extend(b.4.iter().zip(&flags).filter(|(_, f)| !**f).map(|(c, _)| c.clone()).take(7));
        chosen.push((b.0, b.1, b.2, b.3, cands));
        if chosen.len() == 5 {
            break;
        }
    }
    let t_load = Instant::now();
    let judge = nmapi::neural::reranker_model(&home).map_err(|e| e.message)?;
    eprintln!("judge on {} in {:.1} s", if judge.pop.gpu.is_some() { "card" } else { "processor" }, t_load.elapsed().as_secs_f64());
    let (mut first, mut wide_first, mut worst) = (0_usize, 0_usize, 0.0_f64);
    let mut lines = String::new();
    for (set, id, root, question, cands) in &chosen {
        let root = PathBuf::from(root);
        // Candidates in a fixed order by key, so the right piece's place in the list says nothing.
        let mut order: Vec<usize> = (0..cands.len()).collect();
        order.sort_by(|a, b| cands[*a].0.cmp(&cands[*b].0));
        let texts: BTreeMap<String, String> = cands.iter().map(|(k, p, a, z)| (k.clone(), piece(&root, p, *a, *z))).collect();
        let text_of = |k: &str| texts.get(k).map(|t| (t.clone(), t.clone()));
        let docs: Vec<String> = order
            .iter()
            .map(|&i| {
                let (k, p, a, z) = &cands[i];
                nmapi::neural::close_doc(&judge, &format!("{p}:{a}-{z}"), texts.get(k).map_or("", String::as_str))
            })
            .collect();
        // Labels at idle (not timed): the content-free reads of every ordered pair among the 8.
        let mut reader = nmapi::brain_side::JudgeReader { model: &judge, question: question.clone(), text: &text_of };
        let mut labels = Labels::default();
        let t_idle = Instant::now();
        let keys: Vec<String> = order.iter().map(|&i| cands[i].0.clone()).collect();
        let t = Instant::now();
        let z = nmapi::neural::judge_close(&judge, question, &docs).map_err(|e| e.message)?;
        let mut best: Vec<usize> = (0..z.len()).collect();
        best.sort_by(|a, b| z[*b].total_cmp(&z[*a]));
        best.truncate(4);
        let wide_s = t.elapsed().as_secs_f64();
        let pairs: Vec<(String, String)> = best.iter().enumerate().flat_map(|(n, a)| best.iter().skip(n + 1).map(|b| (keys[*a].clone(), keys[*b].clone()))).collect();
        let idle = {
            let ti = Instant::now();
            let n = labels.prepare_idle(&mut reader, &pairs);
            (n, ti.elapsed().as_secs_f64())
        };
        let _ = t_idle;
        let td = Instant::now();
        let mut total: Vec<f32> = z.clone();
        let mut reads = 1_usize;
        for (a, b) in &pairs {
            let d = duel(&mut reader, &mut labels, a, b, 0.0)?;
            reads += d.reads;
            if !d.tie {
                let ia = keys.iter().position(|k| k == a).unwrap_or(0);
                let ib = keys.iter().position(|k| k == b).unwrap_or(0);
                total[ia] += d.a_over_b / 2.0;
                total[ib] -= d.a_over_b / 2.0;
            }
        }
        let duel_s = td.elapsed().as_secs_f64();
        let s = wide_s + duel_s;
        let gold_at = order.iter().position(|&i| i == 0).unwrap_or(usize::MAX);
        let top = |v: &[f32]| (0..v.len()).max_by(|a, b| v[*a].total_cmp(&v[*b]).then(b.cmp(a))).unwrap_or(0);
        let (won, wide_won) = (top(&total) == gold_at && best.contains(&gold_at), top(&z) == gold_at);
        first += usize::from(won);
        wide_first += usize::from(wide_won);
        worst = worst.max(s);
        eprintln!("{set} {id}: first {won} (wide only {wide_won}); right piece in best 4 {}; {reads} reads, wide {wide_s:.2} s, duels {duel_s:.2} s, labels at idle {} in {:.2} s", best.contains(&gold_at), idle.0, idle.1);
        let zs: Vec<String> = total.iter().map(|x| format!("{x:.4}")).collect();
        lines.push_str(&format!("{{\"set\":\"{set}\",\"id\":\"{id}\",\"first\":{won},\"wide_first\":{wide_won},\"s\":{s:.3},\"reads\":{reads},\"total\":[{}],\"gold_at\":{gold_at}}}\n", zs.join(",")));
    }
    std::fs::write(out.join("b1.jsonl"), lines).map_err(|e| e.to_string())?;
    eprintln!("B1 wide-only control: first {wide_first} of {}", chosen.len());
    Ok((first, chosen.len(), worst))
}
