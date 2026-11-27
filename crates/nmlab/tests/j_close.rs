//! Step J test (tentacle master plan 7.5 and section 8, A3 T2 form; lane G, 2026-11-18): the
//! close-call arm's judge as served (`nmapi::neural::close_doc`, `judge_close`: each candidate cut
//! to about 128 tokens, all candidates of a question in one packed card run with the prompt head
//! read once) on A3 T2's candidates (G2's top 10 on the six stack dumps, `scratch/a3/t2-in.txt`,
//! 145 questions). Writes each candidate's log-odds and the card time per question to
//! `scratch/lane-g/j/j.jsonl` (keys and numbers only, no question or piece text); the gate and the
//! fusion with the settle sum are scored by `scratch/lane-g/j_test.py`.
//!
//! Run (card, under `scratch/LOCKS/card.lock`):
//! `cargo test --release -p nmlab --test j_close -- --ignored --nocapture` (`J_N=2` for the smoke).
use std::path::PathBuf;
use std::time::Instant;

type R<T> = Result<T, String>;

/// One block of the judged material: its name, path, kind, text and the pieces in it.
type Block = (String, String, String, String, Vec<(String, String, usize, usize)>);

#[test]
#[ignore = "card job: needs the judge and scratch/a3/t2-in.txt"]
fn j_close_judge() {
    if let Err(e) = run() {
        panic!("j: {e}");
    }
}

fn run() -> R<()> {
    let home = nmtype::nano_root();
    let input = home.join("scratch/a3/t2-in.txt");
    let out = home.join("scratch/lane-g/j");
    let _ = std::fs::create_dir_all(&out);
    let (from, n) = (std::env::var("J_FROM").ok().and_then(|v| v.parse().ok()).unwrap_or(0_usize), std::env::var("J_N").ok().and_then(|v| v.parse().ok()).unwrap_or(2_usize));
    // The served switches as `serve` and `bench` set them, before the first bind (lane S, T-079:
    // without them every kernel is the reference one and the call takes 1.9 s, not 0.78).
    // `J_REF=1` keeps the reference kernels.
    if std::env::var_os("J_REF").is_none() {
        nmengn::gpu::HALF_MATH.store(true, std::sync::atomic::Ordering::Relaxed);
        nmengn::gpu::SEGMENT_ATTENTION.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let t_load = Instant::now();
    let judge = nmapi::neural::reranker_model(&home).map_err(|e| e.message)?;
    eprintln!("judge on {} in {:.1} s", if judge.pop.gpu.is_some() { "card" } else { "processor" }, t_load.elapsed().as_secs_f64());
    let text = std::fs::read_to_string(&input).map_err(|e| format!("{}: {e}", input.display()))?;
    // Blocks: "Q <set> <id> <root>", "T <question>", "P <key>\t<path>\t<first>\t<last>".
    let mut blocks: Vec<Block> = Vec::new();
    for line in text.lines() {
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
    let mut lines = String::new();
    let mut total = 0.0_f64;
    let mut done = 0_usize;
    for (set, id, root, question, cands) in blocks.into_iter().skip(from).take(n) {
        let root = PathBuf::from(&root);
        let t_doc = Instant::now();
        let docs: Vec<String> = cands
            .iter()
            .map(|(_, path, first, last)| {
                let body = std::fs::read_to_string(root.join(path)).unwrap_or_default();
                let ls: Vec<&str> = body.lines().collect();
                let a = first.saturating_sub(1).min(ls.len());
                let z = (*last).min(ls.len()).max(a);
                let t = ls.get(a..z).map(|s| s.join("\n")).unwrap_or_default();
                nmapi::neural::close_doc(&judge, &format!("{path}:{first}-{last}"), &t)
            })
            .collect();
        let doc_s = t_doc.elapsed().as_secs_f64();
        let t = Instant::now();
        let z = nmapi::neural::judge_close(&judge, &question, &docs).map_err(|e| e.message)?;
        let s = t.elapsed().as_secs_f64();
        total += s;
        done += 1;
        let tokens: usize = docs.iter().map(|d| judge.pop.tokenizer.encode(d).map_or(0, |v| v.len())).sum();
        let keys: Vec<String> = cands.iter().map(|c| format!("\"{}\"", c.0.replace('\\', "/").replace('"', ""))).collect();
        let zs: Vec<String> = z.iter().map(|x| format!("{x}")).collect();
        lines.push_str(&format!("{{\"set\":\"{set}\",\"id\":\"{id}\",\"keys\":[{}],\"z\":[{}],\"s\":{s},\"doc_tokens\":{tokens}}}\n", keys.join(","), zs.join(",")));
        eprintln!("{set} {id}: {} candidates, {tokens} document tokens, {s:.3} s (pieces read and cut {doc_s:.3} s)", cands.len());
    }
    let path = out.join(std::env::var("J_FILE").unwrap_or_else(|_| "j.jsonl".to_owned()));
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path).map_err(|e| e.to_string())?;
    std::io::Write::write_all(&mut f, lines.as_bytes()).map_err(|e| e.to_string())?;
    #[expect(clippy::cast_precision_loss, reason = "a question count")]
    let mean = if done > 0 { total / done as f64 } else { 0.0 };
    eprintln!("{done} questions, mean card time {mean:.3} s a question; record {}", path.display());
    Ok(())
}
