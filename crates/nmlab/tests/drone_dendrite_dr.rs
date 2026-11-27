//! Dendrite drone part tests DR-1 and DR-2 (2026-11-10; architecture by Luis Jake Gabriel III; spec
//! `docs/2026-11-10-neumor-spec-octopus-to-neuromorph.md` 4.1). Replays the saved scan folder
//! `scratch/tc1-dump/1794288995065/`. Questions are named by index and id; no question text is
//! written. Every encoding stays in RAM.
//!
//! DR-1: a drone's output (the reference assembly cut after layer k, bound to the shared
//! population) equals the reference assembly's own hidden state after layer k on the same piece
//! alone, bit for bit, for k = 0, 2, 4, 8 on the card and k = 2 on the processor.
//! DR-2: every chunk of the tree goes through its drone once (depths 0, 2, 4, 8 read in one pass
//! to layer 8); per question the request goes through its own drone, and the right chunk's rank
//! among all chunks is taken by the drone affinity alone and by the D1 fusion with the drone
//! affinity added, against the D1 fusion; time per 1,000 chunks.
//!
//! Run: `cargo test --release -p nmapi --test drone_dendrite_dr -- --ignored --nocapture --test-threads 1`
//! (`NM_DR_N=2` questions and `NM_DR_LIMIT=<chunks>` for the smoke run; `NM_DR_DEPTHS=0,2` to
//! shrink the depths).
use nmapi::drone_dendrite::{Affine, Local, Pool, Standard, Dendrite, accumulate, calibrated, output_node};
use nmengn::{Assembly, Device, Model, Population, RunOpts};
use nmpod::intake::{Bf16Table, Drone, Kind, Rarity, Record, Signals, Source, fuse};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::{Arc, Mutex, mpsc};
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
fn rank(score: &[f32], gold: &[usize]) -> usize {
    let mut idx: Vec<usize> = (0..score.len()).collect();
    idx.sort_by(|a, b| score.get(*b).copied().unwrap_or(0.0).total_cmp(&score.get(*a).copied().unwrap_or(0.0)).then(a.cmp(b)));
    idx.iter().position(|i| gold.contains(i)).map_or(usize::MAX, |p| p + 1)
}
/// Reciprocal-rank fusion (k 60, as `nmpod::intake::fuse`) over score columns.
fn rrf(cols: &[&[f32]]) -> Vec<f32> {
    let n = cols.first().map_or(0, |c| c.len());
    let mut out = vec![0.0f32; n];
    for c in cols {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|a, b| c.get(*b).copied().unwrap_or(0.0).total_cmp(&c.get(*a).copied().unwrap_or(0.0)).then(a.cmp(b)));
        for (r, i) in idx.into_iter().enumerate() {
            if c.get(i).copied().unwrap_or(0.0) > 0.0
                && let Some(f) = out.get_mut(i)
            {
                *f += 1.0 / (60.0 + r as f32 + 1.0);
            }
        }
    }
    out
}
fn depths() -> Vec<usize> {
    std::env::var("NM_DR_DEPTHS").ok().map_or_else(|| vec![0, 2, 4, 8], |x| x.split(',').filter_map(|t| t.trim().parse().ok()).collect())
}

struct Tree {
    chunks: Vec<(String, usize, usize, String)>,
    qs: Vec<(usize, String, Vec<usize>)>,
    /// Per question, the chunks of its saved lens windows.
    windows: Vec<Vec<usize>>,
    texts: BTreeMap<String, String>,
}
fn tree(home: &Path, n_want: usize) -> R<Tree> {
    let dir = home.join("scratch/tc1-dump/1794288995065");
    let sub = json(&dir.join("substrate.json"))?;
    let files: Vec<String> =
        sub.get("files").and_then(|x| x.array()).map_err(|e| e.to_string())?.iter().map(|f| s(f, "path")).collect::<R<_>>()?;
    let mut chunks = Vec::new();
    for c in sub.get("chunks").and_then(|x| x.array()).map_err(|e| e.to_string())? {
        chunks.push((files.get(u(c, "file")?).cloned().unwrap_or_default(), u(c, "start")?, u(c, "end")?, s(c, "text")?));
    }
    let mut texts = BTreeMap::new();
    for area in ["context", "engine", "tools"] {
        let v = json(&home.join(format!("tests/capability-gates/cx1-context/questions-{area}.json")))?;
        for q in v.get("questions").and_then(|x| x.array()).map_err(|e| e.to_string())? {
            texts.insert(s(q, "id")?, s(q, "q")?);
        }
    }
    let mut qs = Vec::new();
    let mut windows = Vec::new();
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
        let mut w = std::collections::BTreeSet::new();
        for x in v.get("scan").and_then(|x| x.get("windows")).and_then(|x| x.array()).map_err(|e| e.to_string())? {
            for c in x.get("place").and_then(|x| x.array()).map_err(|e| e.to_string())? {
                w.insert(usize::try_from(c.integer().map_err(|e| e.to_string())?).map_err(|e| e.to_string())?);
            }
        }
        windows.push(w.into_iter().collect());
    }
    Ok(Tree { chunks, qs, windows, texts })
}
fn stamp() -> u128 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}

#[test]
#[ignore = "part test DR-1: needs the release weights and the card; run by hand"]
fn dr1_parity() -> R<()> {
    let home = nmtype::nano_root();
    // 2026-11-19 (lane DR): the pieces of the Nano's frozen development tree (the old dump exists
    // only in the Mini).
    let root = nmsubs::repo::canon(&home.join("scratch").join("frozen-p2"));
    let sub = nmsubs::Substrate::scan(&root, &["neuromorph-nano".to_owned()], Arc::new(|_: &str| Vec::new()));
    let chunks = sub
        .chunks
        .iter()
        .map(|c| (sub.files.get(c.file).map(|f| f.path.clone()).unwrap_or_default(), c.start, c.end, c.text.clone()))
        .collect();
    let t = Tree { chunks, qs: Vec::new(), windows: Vec::new(), texts: BTreeMap::new() };
    let reference = Assembly::load(&home.join("designs/assemblies"), "qwen3-reference").map_err(|e| e.to_string())?;
    let mut out = String::new();
    // Three pieces: the first chunk, the middle one and the longest under 1,200 tokens.
    let mut cases: Vec<(&str, usize, Device)> = Vec::new();
    for dev in [Device::Gpu, Device::Cpu] {
        let pop = Population::open(&home, "qwen3-0.6b", true, dev == Device::Gpu).map_err(|e| e.to_string())?;
        if dev == Device::Gpu && pop.gpu.is_none() {
            let _ = writeln!(out, "gpu: none open, skipped");
            continue;
        }
        let tok = |x: &str| pop.tokenizer.encode(x).map_err(|e| e.to_string());
        let lens: Vec<usize> = t.chunks.iter().map(|c| tok(&c.3).map_or(0, |v| v.len())).collect();
        let longest = lens.iter().enumerate().filter(|(_, l)| **l < 1200).max_by_key(|(_, l)| **l).map_or(0, |x| x.0);
        let picks = if dev == Device::Gpu { vec![0, t.chunks.len() / 2, longest] } else { vec![t.chunks.len() / 2] };
        let ks: Vec<usize> = if dev == Device::Gpu { depths() } else { vec![2] };
        let refm = Model::bind(&pop, reference.clone()).map_err(|e| e.to_string())?;
        for &k in &ks {
            let dendrite = Dendrite::new(&pop, &reference, k, dev).map_err(|e| e.to_string())?;
            for &p in &picks {
                let ids = tok(&t.chunks.get(p).ok_or("chunk")?.3)?;
                let target = output_node(k);
                let r = refm.forward(dev, &ids, std::slice::from_ref(&target), &RunOpts::default()).map_err(|e| e.to_string())?;
                let rv = nmengn::model::values(r.get(&target).ok_or("no reference output")?).map_err(|e| e.to_string())?;
                let d = dendrite.sense(&ids).map_err(|e| e.to_string())?;
                let same = d.states.len() == rv.len() && d.states.iter().zip(rv).all(|(a, b)| a.to_bits() == b.to_bits());
                let dmax = d.states.iter().zip(rv).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                // 2026-11-19: the intake's reading pass (`sense_states` on the whole model).
                let s = nmapi::drone_dendrite::sense_states(&refm, dev, &ids, k).map_err(|e| e.to_string())?;
                let same_s = s.states.len() == rv.len() && s.states.iter().zip(rv).all(|(a, b)| a.to_bits() == b.to_bits());
                let _ = writeln!(
                    out,
                    "{} k {k} chunk {p} tokens {} drone nodes {} of {}: bit-identical {same}, max abs diff {dmax:e}; intake reading pass bit-identical {same_s}",
                    dev.name(),
                    ids.len(),
                    dendrite.model.asm.nodes.len(),
                    reference.nodes.len()
                );
                if !same_s {
                    cases.push(("reading differs", p, dev));
                }
                if !same {
                    cases.push(("differs", p, dev));
                }
            }
        }
        pop.release_device();
    }
    let _ = writeln!(
        out,
        "DR-1 {}",
        if cases.is_empty() { "pass: every drone output bit-identical to the reference path through layer k" } else { "FAIL" }
    );
    print!("{out}");
    std::fs::write(home.join(format!("runs/2026-11-19-DR1-drone-parity-{}.txt", stamp())), &out).map_err(|e| e.to_string())?;
    if cases.is_empty() { Ok(()) } else { Err("DR-1 failed".into()) }
}

#[test]
#[ignore = "part test DR-2: needs the saved scans, the release weights and the card; run by hand"]
#[expect(clippy::too_many_lines, reason = "one part test, read top to bottom")]
fn dr2_rank() -> R<()> {
    let home = nmtype::nano_root();
    let n_want: usize = std::env::var("NM_DR_N").ok().and_then(|x| x.parse().ok()).unwrap_or(usize::MAX);
    let limit: usize = std::env::var("NM_DR_LIMIT").ok().and_then(|x| x.parse().ok()).unwrap_or(usize::MAX);
    let ks = depths();
    let scope_windows = std::env::var("NM_DR_SCOPE").is_ok_and(|x| x == "windows");
    let kmax = ks.iter().copied().max().ok_or("no depths")?;
    let t_all = Instant::now();
    let t = tree(&home, n_want)?;
    // Lexical first sensing (D1) and the pieces' tokens.
    let wdir = nmtype::models_dir(&home).join("qwen3-0.6b/files");
    let tokz = nmtokz::Tokenizer::open(&wdir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let drone = Drone::new(Arc::new(tokz), Bf16Table::open(&wdir).map_err(|e| e.to_string())?);
    let mut pieces: Vec<Record> = Vec::new();
    for (i, (path, a, b, text)) in t.chunks.iter().enumerate().take(limit) {
        pieces.push(
            drone
                .take_in(Kind::Code, Source::Lines { path: path.clone(), start: *a, end: *b }, i + 1, text)
                .map_err(|e| e.to_string())?
                .into_inner(),
        );
    }
    let n = pieces.len();
    let ntok: usize = pieces.iter().map(Record::len).sum();
    let rarity = Rarity::from_records(pieces.iter());
    let mut lex: Vec<Vec<Signals>> = Vec::new();
    let mut reqs: Vec<Record> = Vec::new();
    for (idx, id, _) in &t.qs {
        let text = t.texts.get(id).ok_or_else(|| format!("question {idx}: id not in the suite"))?;
        let req =
            drone.take_in(Kind::Request, Source::Message { label: format!("q{idx:02}") }, 0, text).map_err(|e| e.to_string())?.into_inner();
        let mut probe = drone.probe(&req, &rarity);
        lex.push(pieces.iter().map(|p| probe.sense(p)).collect());
        reqs.push(req);
    }

    // The dendrite drones: one population on the card, the reference cut after layer kmax.
    let pop = Population::open(&home, "qwen3-0.6b", true, true).map_err(|e| e.to_string())?;
    let dev = if pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let reference = Assembly::load(&home.join("designs/assemblies"), "qwen3-reference").map_err(|e| e.to_string())?;
    let dendrite = Dendrite::new(&pop, &reference, kmax, dev).map_err(|e| e.to_string())?;
    let width = pop.dims.hidden;
    let piece_ids: Vec<Vec<i32>> = pieces.iter().map(|p| p.tokens.clone()).collect();
    let pool = Pool::sized(&dendrite, &piece_ids);
    let free0 = pop.gpu_free();
    // Warm-up on the request drones (weights uploaded once, shared).
    let t_req = Instant::now();
    let req_locals: Vec<Vec<Local>> =
        reqs.iter().map(|r| dendrite.sense_at(&r.tokens, &ks)).collect::<Result<_, _>>().map_err(|e| e.to_string())?;
    let req_s = t_req.elapsed().as_secs_f64();

    // Standardisation from a spread sample (every 16th piece), encoded first; the rest stream to
    // scoring workers on the processor while the card encodes.
    let sel: Vec<usize> = if scope_windows {
        t.windows.iter().flatten().copied().filter(|i| *i < n).collect::<std::collections::BTreeSet<_>>().into_iter().collect()
    } else {
        (0..n).collect()
    };
    let stride = 16usize;
    let sample: Vec<usize> = sel.iter().copied().step_by(stride).collect();
    let rest: Vec<usize> = sel.iter().enumerate().filter(|(j, _)| j % stride != 0).map(|(_, i)| *i).collect();
    let enc_tok: usize = sel.iter().filter_map(|i| piece_ids.get(*i)).map(Vec::len).sum();
    let prof = std::env::var("NM_DR_PROF").is_ok();
    nmffi::prof::set(prof);
    let t_enc = Instant::now();
    let mut held: Vec<(usize, Vec<Local>)> = Vec::new();
    let sample_ids: Vec<Vec<i32>> = sample.iter().map(|i| piece_ids.get(*i).cloned().unwrap_or_default()).collect();
    pool.run(&sample_ids, &ks, |j, out| {
        held.push((sample.get(j).copied().unwrap_or(0), out));
        Ok(())
    })
    .map_err(|e| e.to_string())?;
    let mut stds = Vec::new();
    for di in 0..ks.len() {
        let (mut sum, mut sq, mut cnt) = (vec![0.0f64; width], vec![0.0f64; width], 0u64);
        for (_, out) in &held {
            if let Some(l) = out.get(di) {
                cnt += accumulate(l, &mut sum, &mut sq);
            }
        }
        stds.push(Standard::from_sums(&sum, &sq, cnt));
    }
    // Request affinity: unit rows (position 0 left out), weights = rarity of each row's token.
    let affines: Vec<Vec<Affine>> = (0..ks.len())
        .map(|di| {
            reqs.iter()
                .zip(&req_locals)
                .map(|(r, ls)| {
                    let st = stds.get(di).cloned().unwrap_or_default();
                    let l = ls.get(di).cloned().unwrap_or_default();
                    let rows = st.units(&l);
                    let skip = usize::from(r.tokens.len() > 1);
                    let w: Vec<f32> = r.tokens.iter().skip(skip).map(|tk| rarity.token(*tk)).collect();
                    Affine::new(width, rows, w)
                })
                .collect()
        })
        .collect();
    let affines = Arc::new(affines);
    let stds = Arc::new(stds);
    let scores = Arc::new(Mutex::new(vec![vec![vec![0.0f32; n]; t.qs.len()]; ks.len()]));
    let held_bytes = Arc::new(Mutex::new(0usize));
    // 2026-11-10: per depth, question and piece, the per-request-row maxima, and each piece's
    // unit-row count, for the background-relative affinity (`calibrated`).
    let maxes = Arc::new(Mutex::new(vec![vec![vec![Vec::<f32>::new(); n]; t.qs.len()]; ks.len()]));
    let nrows = Arc::new(Mutex::new(vec![0usize; n]));
    let workers = std::thread::available_parallelism().map_or(4, std::num::NonZero::get).saturating_sub(1).max(1);
    let (tx, rx) = mpsc::sync_channel::<(usize, Vec<Local>)>(workers * 4);
    let rx = Arc::new(Mutex::new(rx));
    let mut handles = Vec::new();
    for _ in 0..workers {
        let (rx, affines, stds, scores, held_bytes) =
            (Arc::clone(&rx), Arc::clone(&affines), Arc::clone(&stds), Arc::clone(&scores), Arc::clone(&held_bytes));
        let (maxes, nrows) = (Arc::clone(&maxes), Arc::clone(&nrows));
        handles.push(std::thread::spawn(move || {
            while let Ok((i, out)) = rx.lock().map_err(|_| ()).and_then(|g| g.recv().map_err(|_| ())) {
                if let Ok(mut b) = held_bytes.lock() {
                    *b += out.iter().map(|l| l.states.len() * 4).sum::<usize>();
                }
                for (di, l) in out.iter().enumerate() {
                    let (Some(st), Some(afs)) = (stds.get(di), affines.get(di)) else { continue };
                    let units = st.units(l);
                    let row: Vec<f32> = afs.iter().map(|a| a.score(&units)).collect();
                    let mrow: Vec<Vec<f32>> = afs.iter().map(|a| a.row_max(&units)).collect();
                    if let Ok(mut nr) = nrows.lock()
                        && let Some(x) = nr.get_mut(i)
                    {
                        *x = units.len() / l.width.max(1);
                    }
                    if let Ok(mut mx) = maxes.lock() {
                        for (qi, v) in mrow.into_iter().enumerate() {
                            if let Some(x) = mx.get_mut(di).and_then(|d| d.get_mut(qi)).and_then(|q| q.get_mut(i)) {
                                *x = v;
                            }
                        }
                    }
                    if let Ok(mut sc) = scores.lock() {
                        for (qi, v) in row.into_iter().enumerate() {
                            if let Some(x) = sc.get_mut(di).and_then(|d| d.get_mut(qi)).and_then(|q| q.get_mut(i)) {
                                *x = v;
                            }
                        }
                    }
                }
            }
        }));
    }
    for (i, out) in held.drain(..) {
        tx.send((i, out)).map_err(|e| e.to_string())?;
    }
    let rest_ids: Vec<Vec<i32>> = rest.iter().map(|i| piece_ids.get(*i).cloned().unwrap_or_default()).collect();
    pool.run(&rest_ids, &ks, |j, out| {
        tx.send((rest.get(j).copied().unwrap_or(0), out))
            .map_err(|e| nmtype::Fault::new("dr2", nmtype::FaultKind::Execution, e.to_string()))
    })
    .map_err(|e| e.to_string())?;
    let enc_s = t_enc.elapsed().as_secs_f64();
    drop(tx);
    for h in handles {
        h.join().map_err(|_| "worker")?;
    }
    let all_s = t_enc.elapsed().as_secs_f64();
    let per_k = |secs: f64| secs * 1000.0 / sel.len().max(1) as f64;

    let scores = scores.lock().map_err(|_| "lock")?.clone();
    let maxes = maxes.lock().map_err(|_| "lock")?.clone();
    let nrows = nrows.lock().map_err(|_| "lock")?.clone();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "DR-2 dendrite drones; ranked among {}; tree {n} chunks, {ntok} tokens; encoded {} chunks, {enc_tok} tokens; device {}, batch {} (free device bytes {:?}); depths {ks:?} read in one pass to layer {kmax}",
        if scope_windows {
            "each question's saved lens-window chunks"
        } else if n == t.chunks.len() {
            "all chunks of the whole tree"
        } else {
            "a prefix of the tree (smoke)"
        },
        sel.len(),
        dev.name(),
        pool.batch,
        free0
    );
    let _ = writeln!(
        out,
        "encode {enc_s:.1} s = {:.2} s per 1,000 chunks; with scoring {all_s:.1} s = {:.2} s per 1,000 chunks; requests {req_s:.2} s; drone states seen {:.2} GB (RAM, dropped after scoring)",
        per_k(enc_s),
        per_k(all_s),
        *held_bytes.lock().map_err(|_| "lock")? as f64 / 1e9
    );
    let _ = writeln!(
        out,
        "affinity rule: states after layer k standardised per dimension (mean and spread over every 16th piece), unit-normalised, position 0 left out; per request position max cosine over piece positions, rarity-weighted mean; fused = RRF k 60 of D1's four signals plus the drone affinity"
    );
    let _ = writeln!(
        out,
        "2026-11-10 arms: calibrated = background-relative affinity (per request row, residual of the piece's max cosine over a fit a + b ln(rows) across the pieces scored, in residual spreads, floored at 0, rarity-weighted mean); bm25 = the drones' BM25 sense; bm25 + cal = RRF k 60 of the two"
    );
    let _ = writeln!(
        out,
        "q | id | D1 fused | bm25 | {}",
        ks.iter().map(|k| format!("k{k} drone | k{k} fused | k{k} calibrated | k{k} bm25 + cal")).collect::<Vec<_>>().join(" | ")
    );
    let mut bmr = Vec::new();
    let mut per2: Vec<(Vec<usize>, Vec<usize>)> = vec![(Vec::new(), Vec::new()); ks.len()];
    let mut d1 = Vec::new();
    let mut per: Vec<(Vec<usize>, Vec<usize>)> = vec![(Vec::new(), Vec::new()); ks.len()];
    for (qi, (idx, id, gold)) in t.qs.iter().enumerate() {
        let cand: Vec<usize> = if scope_windows {
            t.windows.get(qi).cloned().unwrap_or_default().into_iter().filter(|i| *i < n).collect()
        } else {
            (0..n).collect()
        };
        let gold: Vec<usize> = cand.iter().enumerate().filter(|(_, c)| gold.contains(c)).map(|(j, _)| j).collect();
        let gold = &gold;
        let all_rows = lex.get(qi).ok_or("lex")?;
        let rows: Vec<Signals> = cand.iter().filter_map(|c| all_rows.get(*c).copied()).collect();
        let rows = &rows;
        let r1 = rank(&fuse(rows), gold);
        d1.push(r1);
        let cols: [Vec<f32>; 4] = [
            rows.iter().map(|x| x.exact).collect(),
            rows.iter().map(|x| x.ident).collect(),
            rows.iter().map(|x| x.structural).collect(),
            rows.iter().map(|x| x.affinity).collect(),
        ];
        let bm: Vec<f32> = rows.iter().map(|x| x.bm25).collect();
        let rb = rank(&bm, gold);
        bmr.push(rb);
        let mut line = format!("{idx} | {id} | {r1} | {rb}");
        for (di, p) in per.iter_mut().enumerate() {
            let full = scores.get(di).and_then(|d| d.get(qi)).ok_or("scores")?;
            let dsc: Vec<f32> = cand.iter().filter_map(|c| full.get(*c).copied()).collect();
            let rd = rank(&dsc, gold);
            let rf = rank(&rrf(&[&cols[0], &cols[1], &cols[2], &cols[3], &dsc]), gold);
            p.0.push(rd);
            p.1.push(rf);
            let mq = maxes.get(di).and_then(|d| d.get(qi)).ok_or("maxes")?;
            let cm: Vec<Vec<f32>> = cand.iter().map(|c| mq.get(*c).cloned().unwrap_or_default()).collect();
            let cr: Vec<usize> = cand.iter().map(|c| nrows.get(*c).copied().unwrap_or(0)).collect();
            let w = affines.get(di).and_then(|a| a.get(qi)).map(|a| a.weights.clone()).unwrap_or_default();
            let cal = calibrated(&cm, &cr, &w);
            let rc = rank(&cal, gold);
            let rbc = rank(&rrf(&[&bm, &cal]), gold);
            if let Some(p2) = per2.get_mut(di) {
                p2.0.push(rc);
                p2.1.push(rbc);
            }
            let _ = write!(line, " | {rd} | {rf} | {rc} | {rbc}");
        }
        let _ = writeln!(out, "{line} | candidates {}", cand.len());
    }
    let within = |v: &[usize], m: usize| v.iter().filter(|r| **r <= m).count();
    let q = t.qs.len();
    let _ = writeln!(out, "(rank {} = the right chunk is not among the candidates)", usize::MAX);
    let _ = writeln!(out, "summary over {q} questions (first / top 6 / top 50):");
    let _ = writeln!(out, "D1 fused (first form) {} / {} / {}", within(&d1, 1), within(&d1, 6), within(&d1, 50));
    let four = |v: &[usize]| format!("{} / {} / {} / {}", within(v, 1), within(v, 6), within(v, 12), within(v, 50));
    let _ = writeln!(out, "(first / top 6 / top 12 / top 50 below)");
    let _ = writeln!(out, "bm25 alone {}", four(&bmr));
    for ((k, (dr, fu)), (ca, bc)) in ks.iter().zip(&per).zip(&per2) {
        let _ = writeln!(
            out,
            "k {k}: drone affinity alone {}; D1 plus drone affinity {}; calibrated alone {}; bm25 plus calibrated {}",
            four(dr),
            four(fu),
            four(ca),
            four(bc)
        );
    }
    if prof {
        let mut tb = nmffi::prof::table();
        tb.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (k, secs, c) in tb.iter().take(14) {
            let _ = writeln!(out, "prof {k} {secs:.2} s x{c}");
        }
    }
    let _ = writeln!(out, "whole test {:.1} s", t_all.elapsed().as_secs_f64());
    print!("{out}");
    std::fs::write(home.join(format!("runs/2026-11-10-DR2-drone-rank-{q}q-{}.txt", stamp())), &out).map_err(|e| e.to_string())?;
    pop.release_device();
    Ok(())
}

/// One question of the Nano's development data: id, text, the chunks that answer it.
type DevQuestion = (String, String, Vec<usize>);

/// The tuning questions of `tests/dev/closure.json` on `scratch/frozen-p2` with their right
/// chunks (a chunk of the answer's file overlapping the answer's range, located by its anchor
/// line; `match: "file"` takes every chunk of the file), at most `n_want`; and the decoys skipped.
fn dev_closure(home: &Path, sub: &nmsubs::Substrate, root: &Path, n_want: usize) -> R<(Vec<DevQuestion>, usize)> {
    let v = json(&home.join("tests/dev/closure.json"))?;
    let norm = |p: &str| p.replace('\\', "/");
    let mut qs = Vec::new();
    let mut decoys = 0;
    for q in v.get("questions").and_then(|x| x.array()).map_err(|e| e.to_string())? {
        let answers = q.get("answers").and_then(|x| x.array()).map_err(|e| e.to_string())?;
        if answers.is_empty() {
            decoys += 1;
            continue;
        }
        if qs.len() >= n_want {
            break;
        }
        let mut gold = Vec::new();
        for a in answers {
            let file = norm(&s(a, "file")?);
            let anchor = s(a, "anchor")?;
            let lines: Vec<usize> = a
                .get("lines")
                .and_then(|x| x.array())
                .map_err(|e| e.to_string())?
                .iter()
                .filter_map(|x| x.integer().ok().and_then(|n| usize::try_from(n).ok()))
                .collect();
            let span = match lines.as_slice() {
                [x, y] if y >= x => y - x,
                _ => 0,
            };
            let text = String::from_utf8_lossy(&std::fs::read(root.join(&file)).map_err(|e| format!("{file}: {e}"))?).into_owned();
            let first = text.lines().position(|l| l.contains(anchor.as_str())).map_or(0, |i| i + 1);
            let whole = a.get("match").and_then(|x| x.string()).is_ok_and(|m| m == "file");
            for (ci, ch) in sub.chunks.iter().enumerate() {
                let same = sub.files.get(ch.file).is_some_and(|f| norm(&f.path) == file);
                if same && (whole || (first > 0 && ch.start <= first + span && ch.end >= first)) && !gold.contains(&ci) {
                    gold.push(ci);
                }
            }
        }
        qs.push((s(q, "id")?, s(q, "question")?, gold));
    }
    Ok((qs, decoys))
}

/// DR-2 in the Nano (2026-11-19, lane DR; master plan 3.2, Phase C3 and section 8): the request
/// drone's attention to every piece of the whole material at depth k (`drone_dendrite::attention`'s
/// rule: the request's layer-k queries against the piece's layer-k keys, the piece encoded alone and
/// placed just before the request, log-mean-exp over its positions, posed minus content-free, mean
/// over the heads), for each k of `NM_DR_DEPTHS` (default 1, 2, 3, 4, 6, 8; every depth read from
/// one encode per piece to the deepest), on the development data (`tests/dev/closure.json` on
/// `scratch/frozen-p2`, decoys left out). Pass: right piece in the top 50 on at least 26 of 30 and
/// in the top 6 on more than 18 of 30 (scaled to the questions with an answer). Per k: first, top
/// 6, top 50 by the drone alone and by RRF with the drones' BM25; encode ms a token; the bytes a
/// token of layer-k keys. Questions are named by id only; every encoding stays in RAM.
///
/// Run (card, under `scratch/LOCKS/card.lock`): `cargo test --release -p nmlab --test
/// drone_dendrite_dr -- --ignored --nocapture --test-threads 1 dr2_depth` (`NM_DR_N=2` and
/// `NM_DR_LIMIT=<pieces>` for the smoke run).
#[test]
#[ignore = "part test DR-2 (Nano): needs the release weights and scratch/frozen-p2; run by hand under the card lock"]
#[expect(clippy::too_many_lines, reason = "one part test, read top to bottom")]
fn dr2_depth() -> R<()> {
    use nmapi::drone_dendrite::{depth_rows_with, head_mean, request_queries};
    use nmengn::drone::Reduce;
    // 2026-11-19 (troubleshooting after the first full run): four scoring variants from the same
    // keys: the reduction over the piece's positions (log-mean-exp, log-sum-exp) and the request
    // tokens that query (every content token, or the rarer half by the material's rarity).
    const VARIANTS: [(&str, Reduce, bool); 4] =
        [("lme all", Reduce::Mean, false), ("lse all", Reduce::Sum, false), ("lme rare", Reduce::Mean, true), ("lse rare", Reduce::Sum, true)];
    let nv = VARIANTS.len();
    let home = nmtype::nano_root();
    let n_want: usize = std::env::var("NM_DR_N").ok().and_then(|x| x.parse().ok()).unwrap_or(usize::MAX);
    let limit: usize = std::env::var("NM_DR_LIMIT").ok().and_then(|x| x.parse().ok()).unwrap_or(usize::MAX);
    let ks: Vec<usize> = std::env::var("NM_DR_DEPTHS")
        .unwrap_or_else(|_| "1,2,3,4,6,8".into())
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let kmax = ks.iter().copied().max().ok_or("no depths")?;
    let t_all = Instant::now();
    let root = nmsubs::repo::canon(&home.join("scratch").join("frozen-p2"));
    let sub = nmsubs::Substrate::scan(&root, &["neuromorph-nano".to_owned()], Arc::new(|_: &str| Vec::new()));
    let (qs, decoys) = dev_closure(&home, &sub, &root, n_want)?;
    let n = sub.chunks.len().min(limit);
    // The drones' lexical sense (BM25) for the RRF arm.
    let wdir = nmtype::models_dir(&home).join("qwen3-0.6b/files");
    let tokz = nmtokz::Tokenizer::open(&wdir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let drone = Drone::new(Arc::new(tokz), Bf16Table::open(&wdir).map_err(|e| e.to_string())?);
    let mut records: Vec<Record> = Vec::with_capacity(n);
    for (i, ch) in sub.chunks.iter().enumerate().take(n) {
        let path = sub.files.get(ch.file).map(|f| f.path.clone()).unwrap_or_default();
        records.push(
            drone.take_in(Kind::Code, Source::Lines { path, start: ch.start, end: ch.end }, i + 1, &ch.text).map_err(|e| e.to_string())?.into_inner(),
        );
    }
    let rarity = Rarity::from_records(records.iter());
    let mut bm: Vec<Vec<f32>> = Vec::new();
    for (id, text, _) in &qs {
        let req = drone.take_in(Kind::Request, Source::Message { label: id.clone() }, 0, text).map_err(|e| e.to_string())?.into_inner();
        let mut probe = drone.probe(&req, &rarity);
        bm.push(records.iter().map(|p| probe.sense(p).bm25).collect());
    }
    // The reader the drones share; every piece's ids as the served path reads it.
    let model = nmapi::neural::model(&home).map_err(|e| e.message)?;
    let dev = if model.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let d = model.pop.dims;
    let geometry = (d.head_dim, d.heads / d.kv_heads.max(1));
    let tok = |x: &str| model.pop.tokenizer.encode(x).map_err(|e| e.to_string());
    let ids: Vec<Vec<i32>> = sub
        .chunks
        .iter()
        .take(n)
        .map(|ch| tok(&nmsubs::dense::doc_text(&sub.files.get(ch.file).map(|f| f.path.clone()).unwrap_or_default(), &ch.text)))
        .collect::<R<_>>()?;
    let ntok: usize = ids.iter().map(Vec::len).sum();
    let start = ids.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let table = Arc::new(nmengn::cpu::rope_table(start, d.head_dim, d.theta));
    // The request drones: per depth and question, the posed and content-free queries.
    let t_req = Instant::now();
    let null = tok("\nQuestion: N/A")?;
    let mut queries = Vec::new();
    for &k in &ks {
        let mut per_q = Vec::new();
        for (_, text, _) in &qs {
            let posed = tok(&format!("\nQuestion: {text}"))?;
            let p = request_queries(&model, dev, &posed, start, k).map_err(|e| e.message)?;
            let z = request_queries(&model, dev, &null, start, k).map_err(|e| e.message)?;
            let tp: Vec<usize> = (usize::from(posed.len() > 1)..posed.len()).collect();
            let tn: Vec<usize> = (usize::from(null.len() > 1)..null.len()).collect();
            let mut rar: Vec<f32> = tp.iter().filter_map(|t| posed.get(*t)).map(|id| rarity.token(*id)).collect();
            rar.sort_by(f32::total_cmp);
            let med = rar.get(rar.len() / 2).copied().unwrap_or(0.0);
            let tr: Vec<usize> = tp.iter().copied().filter(|t| posed.get(*t).is_some_and(|id| rarity.token(*id) >= med)).collect();
            per_q.push((p, z, tp, tn, tr));
        }
        queries.push(per_q);
    }
    let req_s = t_req.elapsed().as_secs_f64();
    let queries = Arc::new(queries);
    let scores = Arc::new(Mutex::new(vec![vec![vec![f32::NAN; n]; qs.len()]; ks.len() * nv]));
    let faults = Arc::new(Mutex::new(Vec::<String>::new()));
    let workers = std::thread::available_parallelism().map_or(4, std::num::NonZero::get).saturating_sub(1).max(1);
    type Job = (usize, Vec<Arc<nmengn::drone::RelevanceKeys>>);
    let (tx, rx) = mpsc::sync_channel::<Job>(workers * 2);
    let rx = Arc::new(Mutex::new(rx));
    let mut handles = Vec::new();
    for _ in 0..workers {
        let (rx, queries, scores, table, faults) = (Arc::clone(&rx), Arc::clone(&queries), Arc::clone(&scores), Arc::clone(&table), Arc::clone(&faults));
        handles.push(std::thread::spawn(move || {
            while let Ok((i, keys)) = rx.lock().map_err(|_| ()).and_then(|g| g.recv().map_err(|_| ())) {
                for (di, pk) in keys.iter().enumerate() {
                    let Some(per_q) = queries.get(di) else { continue };
                    for (vi, (_, reduce, rare)) in VARIANTS.iter().enumerate() {
                        let row: Vec<f32> = per_q
                            .iter()
                            .map(|(p, z, tp, tn, tr)| {
                                let toks = if *rare { tr } else { tp };
                                match (depth_rows_with(p, pk, &table, start, toks, geometry, *reduce), depth_rows_with(z, pk, &table, start, tn, geometry, *reduce)) {
                                    (Ok(a), Ok(b)) => head_mean(&a, &b),
                                    (Err(e), _) | (_, Err(e)) => {
                                        if let Ok(mut f) = faults.lock() {
                                            f.push(format!("piece {i}: {}", e.message));
                                        }
                                        f32::NAN
                                    }
                                }
                            })
                            .collect();
                        if let Ok(mut sc) = scores.lock() {
                            for (qi, v) in row.into_iter().enumerate() {
                                if let Some(x) = sc.get_mut(di * nv + vi).and_then(|q| q.get_mut(qi)).and_then(|c| c.get_mut(i)) {
                                    *x = v;
                                }
                            }
                        }
                    }
                }
            }
        }));
    }
    // Every piece alone through layers 0 to the deepest k once; the keys of each k from it.
    let t_enc = Instant::now();
    let mut enc_ms = 0.0_f64;
    let mut most_held = 0usize;
    for (i, piece) in ids.iter().enumerate() {
        if piece.is_empty() {
            continue;
        }
        let t = Instant::now();
        let kv = nmengn::kvstore::encode_through(&model, dev, None, piece, kmax).map_err(|e| e.message)?;
        enc_ms += t.elapsed().as_secs_f64() * 1000.0;
        let mut keys = Vec::with_capacity(ks.len());
        for &k in &ks {
            let pairs: Vec<(usize, usize)> = (0..d.kv_heads).map(|h| (k, h)).collect();
            keys.push(Arc::new(nmengn::drone::RelevanceKeys::from_chunk(&kv, &pairs, d.kv_heads, d.head_dim).map_err(|e| e.message)?));
        }
        most_held = most_held.max(keys.iter().map(|p| p.bytes()).sum::<usize>());
        tx.send((i, keys)).map_err(|e| e.to_string())?;
    }
    drop(tx);
    for h in handles {
        h.join().map_err(|_| "worker")?;
    }
    let all_s = t_enc.elapsed().as_secs_f64();
    let scores = scores.lock().map_err(|_| "lock")?.clone();
    let faults = faults.lock().map_err(|_| "lock")?.clone();
    let q = qs.len();
    let within = |v: &[usize], m: usize| v.iter().filter(|r| **r <= m).count();
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let scale = |x: usize| x as f64 * q as f64 / 30.0;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "DR-2 (Nano) request drone attention at depth k; development data tests/dev/closure.json on scratch/frozen-p2: {q} questions with an answer ({decoys} decoys left out); ranked among {} of {} pieces ({ntok} tokens); device {}; depths {ks:?} from one encode per piece to layer {kmax}",
        n,
        sub.chunks.len(),
        dev.name()
    );
    #[expect(clippy::cast_precision_loss, reason = "a token count")]
    let per_tok = enc_ms / ntok.max(1) as f64;
    let _ = writeln!(
        out,
        "encode to layer {kmax}: {enc_ms:.0} ms = {per_tok:.3} ms a token (layers 0 to k cost about (k+1)/{} of a forward); with scoring {all_s:.1} s; requests {req_s:.2} s; layer-k keys {} bytes a token per k (all {} key-value heads), largest piece's keys of every k held at once {:.1} MB; scoring faults {}",
        d.layers,
        d.kv_heads * d.head_dim * 2,
        d.kv_heads,
        most_held as f64 / 1e6,
        faults.len()
    );
    let _ = writeln!(out, "pass rule (of 30, scaled to {q}): top 50 at least {:.1}, top 6 above {:.1}", scale(26), scale(18));
    let mut line = String::from("id | bm25");
    for k in &ks {
        for (name, _, _) in VARIANTS {
            let _ = write!(line, " | k{k} {name} drone | k{k} {name} bm25+drone");
        }
    }
    let _ = writeln!(out, "{line}");
    let mut bmr = Vec::new();
    let mut per: Vec<(Vec<usize>, Vec<usize>)> = vec![(Vec::new(), Vec::new()); ks.len() * nv];
    for (qi, (id, _, gold)) in qs.iter().enumerate() {
        let b = bm.get(qi).ok_or("bm25")?;
        let rb = rank(b, gold);
        bmr.push(rb);
        let mut line = format!("{id} | {rb}");
        for (di, p) in per.iter_mut().enumerate() {
            let dsc: Vec<f32> = scores.get(di).and_then(|x| x.get(qi)).ok_or("scores")?.iter().map(|x| if x.is_finite() { *x } else { f32::MIN }).collect();
            let rd = rank(&dsc, gold);
            let shifted: Vec<f32> = {
                let lo = dsc.iter().copied().filter(|x| *x > f32::MIN).fold(f32::INFINITY, f32::min);
                dsc.iter().map(|x| if *x > f32::MIN { x - lo + 1.0 } else { 0.0 }).collect()
            };
            let rf = rank(&rrf(&[b, &shifted]), gold);
            p.0.push(rd);
            p.1.push(rf);
            let _ = write!(line, " | {rd} | {rf}");
        }
        let _ = writeln!(out, "{line}{}", if gold.is_empty() { " (no right chunk found in the material)" } else { "" });
    }
    let three = |v: &[usize]| format!("{} / {} / {}", within(v, 1), within(v, 6), within(v, 50));
    let pass = |v: &[usize]| (within(v, 50) as f64) >= scale(26) && (within(v, 6) as f64) > scale(18);
    let _ = writeln!(out, "summary over {q} questions (first / top 6 / top 50):");
    let _ = writeln!(out, "bm25 alone {}", three(&bmr));
    for (col, (dr, fu)) in per.iter().enumerate() {
        let k = ks.get(col / nv).copied().unwrap_or(0);
        let name = VARIANTS.get(col % nv).map_or("", |v| v.0);
        let _ = writeln!(
            out,
            "k {k} {name}: drone alone {} ({}); bm25 plus drone {} ({})",
            three(dr),
            if pass(dr) { "pass" } else { "not passed" },
            three(fu),
            if pass(fu) { "pass" } else { "not passed" }
        );
    }
    let first_pass = per.iter().enumerate().find(|(_, (dr, fu))| pass(dr) || pass(fu)).and_then(|(col, _)| ks.get(col / nv).copied());
    let _ = writeln!(out, "k by the rule (the smallest depth that passes): {}", first_pass.map_or_else(|| format!("none; fallback {}", nmapi::drone_dendrite::FALLBACK_DEPTH), |k| k.to_string()));
    if let Some(f) = faults.first() {
        let _ = writeln!(out, "first scoring fault: {f}");
    }
    let _ = writeln!(out, "whole test {:.1} s", t_all.elapsed().as_secs_f64());
    print!("{out}");
    std::fs::write(home.join(format!("runs/2026-11-19-DR2-depth-{q}q-{}.txt", stamp())), &out).map_err(|e| e.to_string())?;
    Ok(())
}

#[test]
fn section_check() -> R<()> {
    nmapi::drone_dendrite::check()
}

#[test]
fn key_scan_section_check() -> R<()> {
    nmapi::key_scan::check()
}

/// The tentacle core's and the tentacle pass's section checks (lane DR's coding package of
/// 2026-11-19: the assembled core, its decisions, the deep check, the Team's windows).
#[test]
fn tentacle_core_section_check() -> R<()> {
    nmapi::tentacle_core::check()
}

#[test]
fn tentacle_pass_section_check() -> R<()> {
    nmapi::tentacle_pass::check()
}
