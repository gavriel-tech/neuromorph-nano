//! G1 fusion, first form in the drone path (2026-11-10; Gavriel Technologies, architecture by
//! Luis Jake Gabriel III; spec `docs/2026-11-10-neumor-spec-octopus-to-neuromorph.md` 4.4). By
//! replay on the saved development scans, processor only: the drones' BM25 order over the whole
//! tree (`Keep::sense_with`); the strong flags (mean + 1 sd of the flag strengths,
//! `drone_windows::strong_threshold`) seed a personalised PageRank over the code's specific
//! reference relations (`refgraph::Specific::rank`, the method of the Aider repository map,
//! Apache-2.0, method only); G1 fuses the two evidence families by reciprocal-rank fusion (k 60).
//! Arms, right chunk first / top 6 / top 12 / top 50: BM25 alone, graph alone, G1 fusion.
//! Questions are named by index and id; no question text is written. Encodings stay in RAM.
//!
//! Run: `cargo test --release -p nmapi --test g1_fusion -- --ignored --nocapture`
//! (`NM_G1_N=2` for the smoke run). Writes `runs/2026-11-10-G1F-fusion-<n>q-<ms>.txt`.
use nmapi::drone_windows::{Keep, strong_threshold};
use nmapi::scan_dump;
use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

type R<T> = Result<T, String>;

/// The development half: (id, question text) by index, as the locked harness orders it.
fn dev_half(home: &Path) -> R<Vec<(String, String)>> {
    const TYPES: [&str; 5] = ["behaviour", "name", "value", "explain", "relation"];
    let mut out = Vec::new();
    for area in ["context", "engine", "tools"] {
        let p = home.join(format!("tests/capability-gates/cx1-context/questions-{area}.json"));
        let v = nmjson::parse(&std::fs::read(&p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let mut qs = Vec::new();
        for x in v.get("questions").and_then(|y| y.array()).map_err(|e| e.to_string())? {
            let s = |k: &str| x.get(k).and_then(|y| y.string()).map(str::to_owned).map_err(|e| e.to_string());
            let t = s("type")?;
            qs.push((TYPES.iter().position(|y| *y == t).unwrap_or(usize::MAX), s("id")?, s("q")?));
        }
        qs.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
        out.extend(qs.into_iter().enumerate().filter(|(pos, _)| pos % 2 == 0).map(|(_, (_, id, q))| (id, q)));
    }
    Ok(out)
}

/// Place (1-based) of the best right chunk in descending order of `score`, ties by index.
fn place(score: &[f32], gold: &[usize]) -> usize {
    let mut idx: Vec<usize> = (0..score.len()).collect();
    idx.sort_by(|a, b| score.get(*b).copied().unwrap_or(0.0).total_cmp(&score.get(*a).copied().unwrap_or(0.0)).then(a.cmp(b)));
    idx.iter().position(|i| gold.contains(i)).map_or(usize::MAX, |p| p + 1)
}

/// Reciprocal-rank fusion (k 60) of score columns; a zero score adds nothing.
fn rrf(cols: &[&[f32]]) -> Vec<f32> {
    let n = cols.first().map_or(0, |c| c.len());
    let mut out = vec![0.0f32; n];
    for col in cols {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|a, b| col.get(*b).copied().unwrap_or(0.0).total_cmp(&col.get(*a).copied().unwrap_or(0.0)).then(a.cmp(b)));
        for (r, i) in idx.into_iter().enumerate() {
            if col.get(i).copied().unwrap_or(0.0) > 0.0
                && let Some(x) = out.get_mut(i)
            {
                *x += 1.0 / (60.0 + (r + 1) as f32);
            }
        }
    }
    out
}

#[test]
#[ignore = "G1 fusion replay: needs the saved scans and the release tokenizer and table; run by hand"]
fn g1_fusion_replay() -> R<()> {
    let home = nmtype::nano_root();
    let dir = home.join("scratch/tc1-dump/1794288995065");
    let n_want: usize = std::env::var("NM_G1_N").ok().and_then(|x| x.parse().ok()).unwrap_or(usize::MAX);
    let started = Instant::now();
    let items = dev_half(&home)?;
    let sub = scan_dump::load_substrate(&dir).map_err(|e| e.to_string())?;
    let mut keep = Keep::open(&nmtype::models_dir(&home).join("qwen3-0.6b/files")).map_err(|e| e.to_string())?;
    keep.sync(&sub).map_err(|e| e.to_string())?;
    let t_graph = Instant::now();
    let graph = sub.refs();
    let graph_s = t_graph.elapsed().as_secs_f64();
    let setup_s = started.elapsed().as_secs_f64();

    let mut out = String::new();
    let _ = writeln!(
        out,
        "G1F G1 fusion by replay, processor only; dump 1794288995065; {} chunks; specific relations {} pairs (graph {graph_s:.2} s); setup {setup_s:.1} s",
        sub.chunks.len(),
        graph.specific.edges
    );
    let _ = writeln!(
        out,
        "seeds: drone strong flags (BM25 order, flag strength at or above mean + 1 sd), weight = flag strength; graph = personalised PageRank over the specific relations (damping 0.85); G1 fusion = RRF k 60 of BM25 and graph"
    );
    let _ = writeln!(out, "q | id | seeds | bm25 | graph | g1 fusion | seconds | fused places of the BM25 top 12 | gold");
    let (mut b, mut g, mut f, mut secs, mut n) = (Vec::new(), Vec::new(), Vec::new(), 0.0f64, 0usize);
    for idx in scan_dump::indices(&dir) {
        if n >= n_want {
            break;
        }
        let Some((id, text)) = items.get(idx) else { continue };
        let (q, _) = scan_dump::load_question(&dir, idx, text).map_err(|e| e.to_string())?;
        if &q.id != id {
            return Err(format!("dump question {idx} does not match the development half"));
        }
        let t = Instant::now();
        let bm = keep.sense_with(text, &[|s| s.bm25]).map_err(|e| e.to_string())?;
        let cut = strong_threshold(&bm);
        let seeds: Vec<(usize, f32)> = bm.iter().enumerate().filter(|(_, x)| **x > 0.0 && **x >= cut).map(|(i, x)| (i, *x)).collect();
        let (pr, _) = graph.specific.rank(&seeds);
        let fused = rrf(&[&bm, &pr]);
        let s = t.elapsed().as_secs_f64();
        secs += s;
        n += 1;
        let (pb, pg, pf) = (place(&bm, &q.gold), place(&pr, &q.gold), place(&fused, &q.gold));
        b.push(pb);
        g.push(pg);
        f.push(pf);
        // Fused place of every chunk in the BM25 top 12 (the judge's candidates), for the
        // tie-break replay against the recorded judge orders.
        let mut fo: Vec<usize> = (0..fused.len()).collect();
        fo.sort_by(|a, b| fused.get(*b).copied().unwrap_or(0.0).total_cmp(&fused.get(*a).copied().unwrap_or(0.0)).then(a.cmp(b)));
        let mut fplace = vec![usize::MAX; fused.len()];
        for (r, i) in fo.iter().enumerate() {
            if let Some(x) = fplace.get_mut(*i) {
                *x = r + 1;
            }
        }
        let mut bo: Vec<usize> = (0..bm.len()).collect();
        bo.sort_by(|a, b| bm.get(*b).copied().unwrap_or(0.0).total_cmp(&bm.get(*a).copied().unwrap_or(0.0)).then(a.cmp(b)));
        let top: Vec<String> = bo.iter().take(12).map(|i| format!("{i}:{}", fplace.get(*i).copied().unwrap_or(usize::MAX))).collect();
        let _ = writeln!(
            out,
            "{idx} | {id} | {} | {pb} | {pg} | {pf} | {s:.2} | fused places of the BM25 top 12: {} | gold {:?}",
            seeds.len(),
            top.join(" "),
            q.gold
        );
    }
    let within = |v: &[usize], m: usize| v.iter().filter(|r| **r <= m).count();
    let four = |v: &[usize]| format!("{} / {} / {} / {}", within(v, 1), within(v, 6), within(v, 12), within(v, 50));
    let _ = writeln!(
        out,
        "summary over {n} questions (first / top 6 / top 12 / top 50):\nbm25 alone {}\ngraph alone {}\ng1 fusion {}",
        four(&b),
        four(&g),
        four(&f)
    );
    let _ = writeln!(out, "per question {:.2} s; whole test {:.1} s", secs / n.max(1) as f64, started.elapsed().as_secs_f64());
    print!("{out}");
    let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    std::fs::write(home.join(format!("runs/2026-11-10-G1F-fusion-{n}q-{ms}.txt")), &out).map_err(|e| e.to_string())?;
    Ok(())
}
