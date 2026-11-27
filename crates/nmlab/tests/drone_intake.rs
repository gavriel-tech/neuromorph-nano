//! Drone intake part test D1 (2026-11-10; architecture by Luis Jake Gabriel III). Processor only:
//! the tokenizer and the token embedding table, nothing else. Replays the saved scan folder
//! `scratch/tc1-dump/1794288995065/` (substrate chunk texts and each question's right chunks):
//! every chunk of the studied tree enters through a drone, then per development question the
//! drone's first sensing ranks all chunks. Reports the right chunk's rank by drone signals alone,
//! within the top 6 and top 50, and time per 1,000 chunks. Questions are named by index and id;
//! no question text is written. Encodings stay in RAM.
//!
//! Run: `cargo test --release -p nmpod --test drone_intake -- --ignored --nocapture`
//! (`NM_DRONE_N=2` for the smoke run).
use nmpod::intake::{Bf16Table, Drone, Kind, Probe, Rarity, Record, Signals, Source, WITH_BM25, fuse, fuse_with};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

type R<T> = Result<T, String>;

fn json(p: &Path) -> R<nmjson::Value> {
    nmjson::parse(&std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?).map_err(|e| e.to_string())
}
fn s(v: &nmjson::Value, k: &str) -> R<String> {
    Ok(v.get(k).and_then(|x| x.string()).map_err(|e| format!("{k}: {e}"))?.to_owned())
}
fn u(v: &nmjson::Value, k: &str) -> R<usize> {
    usize::try_from(v.get(k).and_then(|x| x.integer()).map_err(|e| format!("{k}: {e}"))?).map_err(|e| e.to_string())
}

/// Position (1-based) of the best right chunk in descending order of `score`, ties by index.
fn rank(score: &[f32], gold: &[usize]) -> usize {
    let mut idx: Vec<usize> = (0..score.len()).collect();
    idx.sort_by(|a, b| score.get(*b).copied().unwrap_or(0.0).total_cmp(&score.get(*a).copied().unwrap_or(0.0)).then(a.cmp(b)));
    idx.iter().position(|i| gold.contains(i)).map_or(usize::MAX, |p| p + 1)
}

#[test]
#[ignore = "part test D1: needs the saved scans and the release tokenizer and table; run by hand"]
fn drone_intake_d1() -> R<()> {
    let home = nmtype::nano_root();
    let dir = home.join("scratch/tc1-dump/1794288995065");
    let n_want: usize = std::env::var("NM_DRONE_N").ok().and_then(|x| x.parse().ok()).unwrap_or(usize::MAX);
    let t_all = Instant::now();

    // Saved tree: files and chunk texts.
    let sub = json(&dir.join("substrate.json"))?;
    let files: Vec<String> =
        sub.get("files").and_then(|x| x.array()).map_err(|e| e.to_string())?.iter().map(|f| s(f, "path")).collect::<R<_>>()?;
    let mut chunks = Vec::new();
    for c in sub.get("chunks").and_then(|x| x.array()).map_err(|e| e.to_string())? {
        chunks.push((files.get(u(c, "file")?).cloned().unwrap_or_default(), u(c, "start")?, u(c, "end")?, s(c, "text")?));
    }
    // Question texts by id, held in RAM only.
    let mut texts = BTreeMap::new();
    for area in ["context", "engine", "tools"] {
        let v = json(&home.join(format!("tests/capability-gates/cx1-context/questions-{area}.json")))?;
        for q in v.get("questions").and_then(|x| x.array()).map_err(|e| e.to_string())? {
            texts.insert(s(q, "id")?, s(q, "q")?);
        }
    }
    let mut qs = Vec::new();
    for i in 0..30usize {
        let p = dir.join(format!("q{i:02}.json"));
        if !p.exists() || qs.len() >= n_want {
            continue;
        }
        let v = json(&p)?;
        let gold: Vec<usize> = v
            .get("gold")
            .and_then(|x| x.array())
            .map_err(|e| e.to_string())?
            .iter()
            .filter_map(|g| g.integer().ok().and_then(|x| usize::try_from(x).ok()))
            .collect();
        qs.push((u(&v, "index")?, s(&v, "id")?, gold));
    }

    // The drone: release tokenizer and the central token embedding table.
    let wdir = nmtype::models_dir(&home).join("qwen3-0.6b/files");
    let t0 = Instant::now();
    let tok = nmtokz::Tokenizer::open(&wdir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let table = Bf16Table::open(&wdir).map_err(|e| e.to_string())?;
    let open_s = t0.elapsed().as_secs_f64();
    let drone = Drone::new(Arc::new(tok), table);

    // Intake of every chunk: transduction once (the code is unchanged across questions).
    let t1 = Instant::now();
    let mut pieces: Vec<Record> = Vec::with_capacity(chunks.len());
    let mut ntok = 0usize;
    for (i, (path, a, b, text)) in chunks.iter().enumerate() {
        let r =
            drone.take_in(Kind::Code, Source::Lines { path: path.clone(), start: *a, end: *b }, i + 1, text).map_err(|e| e.to_string())?;
        ntok += r.get().len();
        pieces.push(r.into_inner());
    }
    let rarity = Rarity::from_records(pieces.iter());
    let intake_s = t1.elapsed().as_secs_f64();
    let n = pieces.len();
    let per_k = |secs: f64| secs * 1000.0 / f64::from(u32::try_from(n.max(1)).unwrap_or(1));

    let mut out = String::new();
    let _ = writeln!(
        out,
        "D1 drone intake, processor only; {n} chunks, {ntok} tokens; open tokenizer and table {open_s:.2} s; intake (tokens, words, rarity) {intake_s:.2} s = {:.3} s per 1,000 chunks",
        per_k(intake_s)
    );
    let _ = writeln!(
        out,
        "affinity: per request token the max cosine over the piece's distinct tokens, rarity-weighted mean over request tokens; combined: reciprocal-rank fusion of exact, ident, structural, affinity (k 60)"
    );
    let _ = writeln!(out, "q | id | combined | affinity | ident | structural | exact | bm25 | combined plus bm25 | sensing s");
    let mut combined = Vec::new();
    let mut aff = Vec::new();
    let (mut bm, mut plus) = (Vec::new(), Vec::new());
    // Fusion shapes (2026-11-10): bm25 with structural; bm25 with exact and structural; the five
    // signals with bm25 counted twice (weighted RRF).
    let shapes: [(&str, &[nmpod::intake::Sig]); 4] = [
        ("bm25 + structural", &[|x| x.bm25, |x| x.structural]),
        ("bm25 + exact + structural", &[|x| x.bm25, |x| x.exact, |x| x.structural]),
        ("five signals, bm25 twice", &[|x| x.exact, |x| x.ident, |x| x.structural, |x| x.affinity, |x| x.bm25, |x| x.bm25]),
        ("bm25f alone (Sourcegraph, boost 5)", &[|x| x.bm25f]),
    ];
    let mut shaped: Vec<Vec<usize>> = vec![Vec::new(); shapes.len()];
    let mut sense_total = 0.0f64;
    for (idx, id, gold) in &qs {
        let text = texts.get(id).ok_or_else(|| format!("question {idx}: id {id} not in the suite"))?;
        let t = Instant::now();
        let req = drone.take_in(Kind::Request, Source::Message { label: format!("q{idx:02}") }, 0, text).map_err(|e| e.to_string())?;
        let mut probe: Probe = drone.probe(req.get(), &rarity);
        let rows: Vec<Signals> = pieces.iter().map(|p| probe.sense(p)).collect();
        let fused = fuse(&rows);
        let fused5 = fuse_with(&rows, &WITH_BM25);
        let secs = t.elapsed().as_secs_f64();
        sense_total += secs;
        let col = |g: fn(&Signals) -> f32| rank(&rows.iter().map(g).collect::<Vec<_>>(), gold);
        let (rc, ra) = (rank(&fused, gold), col(|x| x.affinity));
        combined.push(rc);
        aff.push(ra);
        let (rb, rp) = (col(|x| x.bm25), rank(&fused5, gold));
        bm.push(rb);
        for (v, (_, g)) in shaped.iter_mut().zip(shapes.iter()) {
            v.push(rank(&fuse_with(&rows, g), gold));
        }
        // 2026-11-10: the BM25F top 12 by chunk index (for the judge replay).
        let mut fo: Vec<usize> = (0..rows.len()).collect();
        fo.sort_by(|a, b| rows.get(*b).map_or(0.0, |x| x.bm25f).total_cmp(&rows.get(*a).map_or(0.0, |x| x.bm25f)).then(a.cmp(b)));
        let _ = writeln!(out, "bm25f top 12 q{idx} {id}: {}", fo.iter().take(12).map(ToString::to_string).collect::<Vec<_>>().join(" "));
        plus.push(rp);
        let _ = writeln!(
            out,
            "{idx} | {id} | {rc} | {ra} | {} | {} | {} | {rb} | {rp} | {secs:.2}",
            col(|x| x.ident),
            col(|x| x.structural),
            col(|x| x.exact)
        );
    }
    let k = qs.len();
    let within = |v: &[usize], m: usize| v.iter().filter(|r| **r <= m).count();
    let median = |v: &[usize]| {
        let mut w = v.to_vec();
        w.sort_unstable();
        w.get(w.len() / 2).copied().unwrap_or(0)
    };
    let row = |name: &str, v: &[usize]| {
        format!(
            "{name}: first {} top 6 {} top 12 {} top 50 {} median {}",
            within(v, 1),
            within(v, 6),
            within(v, 12),
            within(v, 50),
            median(v)
        )
    };
    let _ = writeln!(
        out,
        "{}
{}
{}
{}",
        row("first form (4 signals)", &combined),
        row("affinity alone", &aff),
        row("bm25 alone", &bm),
        row("first form plus bm25 (5 signals)", &plus)
    );
    for (v, (nm, _)) in shaped.iter().zip(shapes.iter()) {
        let _ = writeln!(out, "{}", row(nm, v));
    }
    let mean_sense = sense_total / f64::from(u32::try_from(k.max(1)).unwrap_or(1));
    let _ = writeln!(
        out,
        "summary over {k} questions: combined first {} top 6 {} top 50 {} median {}; affinity alone first {} top 6 {} top 50 {} median {}",
        within(&combined, 1),
        within(&combined, 6),
        within(&combined, 50),
        median(&combined),
        within(&aff, 1),
        within(&aff, 6),
        within(&aff, 50),
        median(&aff)
    );
    let _ = writeln!(
        out,
        "sensing per question {mean_sense:.2} s = {:.3} s per 1,000 chunks; whole test {:.1} s",
        per_k(mean_sense),
        t_all.elapsed().as_secs_f64()
    );
    let _ = writeln!(
        out,
        "references: lens windows hold the right chunk 29 of 30; old layer 8 to 17 head reading first inside its window 10 to 18 of 30"
    );
    print!("{out}");
    let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    let p = home.join(format!("runs/2026-11-10-D1-drone-intake-{k}q-{ms}.txt"));
    std::fs::write(&p, &out).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok(())
}
