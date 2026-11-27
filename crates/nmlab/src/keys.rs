//! `nano-lab run keys [dump folder] [n]` (2026-11-10): first measurement of the keys-only
//! scan (tentacle core design `docs/2026-11-10-neumor-design-tentacle-core-v1.md` section 2.2,
//! additions A6 and A7; architecture by Luis Jake Gabriel III) against the exact scan saved by
//! `run tc1 dump`. One graphics-card job:
//! 1. The background pass of `nmapi::key_scan::Keys` over every chunk of the dump's tree (the code
//!    base as the dump saw it), through the device gate, keeping only the drone-key packets: its
//!    time on this machine, bytes per token, bytes held, and the bytes the Rust source alone needs.
//! 2. Per dumped question (development half, by index; the first `n`, default all), per window:
//!    the question path for the posed and the content-free text, and the keys-only raw-score table
//!    on the processor (one thread per tentacle); compared with the exact scan's table from the
//!    dump: right chunk first inside its window, right chunk in the best 6, same chunk first, rank
//!    correlation of the chunk scores per window, right chunk first across all windows, and per
//!    drone layer the rank correlation of each head's chunk readings (which layers lose most).
//! 3. Seconds per question, keys-only against the exact scan's, and peak RAM.
//!
//! Records hold indices and ids only. Writes `runs/KEYS-scan-<ms>.txt`.
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use crate::MINI_ANCHOR;
use nmapi::key_scan::{Job, Keys, spearman};
use nmapi::scan_dump;
use nmapi::tentacle_pass::{self, calibrated};
use nmengn::kvstore::{self, Window};
use nmengn::placement::Device;
use nmtype::{Fault, FaultKind};

use crate::e2::Item;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("keys", FaultKind::Execution, m)
}

/// Chunk scores above the anchor from a table (heads x parts): calibrated, anchor subtracted.
fn above(posed: &[f32], null: &[f32], parts: usize) -> Vec<f32> {
    let c = calibrated(posed, null, parts);
    let a = c.first().copied().unwrap_or(0.0);
    c.into_iter().skip(1).map(|x| x - a).collect()
}

/// One head's reading per chunk above the anchor (row `h` of a heads x parts table).
fn head_row(posed: &[f32], null: &[f32], parts: usize, h: usize) -> Vec<f32> {
    let at = |k: usize| posed.get(h * parts + k).copied().unwrap_or(0.0) - null.get(h * parts + k).copied().unwrap_or(0.0);
    (1..parts).map(|k| at(k) - at(0)).collect()
}

/// Place of the best gold chunk under `scores` (index order of `gold`), if any.
fn gold_place(scores: &[f32], gold: &[bool]) -> Option<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|a, b| {
        scores.get(*b).copied().unwrap_or(f32::NEG_INFINITY).total_cmp(&scores.get(*a).copied().unwrap_or(f32::NEG_INFINITY)).then(a.cmp(b))
    });
    idx.iter().position(|i| gold.get(*i).copied().unwrap_or(false))
}

fn first_of(scores: &[f32]) -> Option<usize> {
    (0..scores.len()).max_by(|a, b| {
        scores.get(*a).copied().unwrap_or(f32::NEG_INFINITY).total_cmp(&scores.get(*b).copied().unwrap_or(f32::NEG_INFINITY)).then(b.cmp(a))
    })
}

fn mean(v: &[f64]) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let n = v.len().max(1) as f64;
    v.iter().sum::<f64>() / n
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

fn show(k: Option<usize>) -> String {
    k.map_or_else(|| "-".to_owned(), |x| x.to_string())
}

#[derive(Default)]
struct QRow {
    id: String,
    /// (exact, keys): best gold place inside its window; across all windows.
    inside: (Option<usize>, Option<usize>),
    across: (Option<usize>, Option<usize>),
    /// Windows, windows whose first chunk agrees, rank correlations.
    windows: usize,
    same_first: usize,
    rho: Vec<f64>,
    /// Seconds: exact scan (dump, encode plus reads) and its reads; keys-only question path and
    /// scoring.
    exact_s: f64,
    exact_read_s: f64,
    qpath_s: f64,
    score_s: f64,
}

#[expect(clippy::too_many_lines, reason = "one measurement, in the order it runs")]
fn measure(home: &Path, dir: &Path, n: usize, rec: &Path) -> Result<String, Fault> {
    let started = Instant::now();
    let mut out = String::new();
    let _ = writeln!(out, "KEYS first measurement of the keys-only scan against the exact scan of a dump: {}", dir.display());
    let items: Vec<Item> = crate::e2::suite(home)?;
    let sub = scan_dump::load_substrate(dir)?;
    let model = nmapi::neural::model(home)?;
    let dev = if model.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let d = model.pop.dims;
    let group = d.heads / d.kv_heads.max(1);
    let sets = tentacle_pass::drones(home);
    let last = sets.last_layer();
    let mut all_heads = sets.heads.clone();
    for v in sets.tentacles.values() {
        all_heads.extend(v.iter().copied());
    }
    all_heads.sort_unstable();
    all_heads.dedup();
    let anchor_text = if sub.files.iter().any(|f| f.path == "crates/nmcore/Cargo.toml") {
        MINI_ANCHOR.to_owned()
    } else {
        format!("Repository of {} files, read in parts.\n", sub.files.len())
    };
    let tok = |s: &str| model.pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
    let ak = Arc::new(kvstore::encode_through(&model, dev, None, &tok(&anchor_text)?, last)?);
    let aw = Arc::new(Window::build(&model, &[Arc::clone(&ak)])?);
    let limit = kvstore::derive_ram_limit();
    let keys = Keys::new(&all_heads, group, d.head_dim, limit.bytes);
    let _ = writeln!(
        out,
        "model on {}; drone set {} heads ({}), {} pairs, {} bytes per token (full keys and values of layers 0..={last}: {}); store limit {} MB ({}; free RAM read {} MB)",
        dev.name(),
        all_heads.len(),
        sets.source,
        keys.pairs.len(),
        keys.bytes_per_token,
        kvstore::bytes_per_token(&model, last + 1),
        limit.bytes >> 20,
        if limit.fallback { "fallback" } else { "half of the free RAM" },
        limit.free >> 20
    );
    let _ = std::fs::write(rec, &out);

    // 1. The background pass over the whole tree.
    let docs: Vec<(String, String)> = sub
        .chunks
        .iter()
        .map(|c| (c.key.clone(), sub.files.get(c.file).map(|f| nmsubs::dense::doc_text(&f.path, &c.text)).unwrap_or_default()))
        .collect();
    keys.pass(&model, dev, &aw, &docs, sub.version);
    let st = keys.status();
    let (kb, kn, _) = keys.held();
    // Bytes the Rust source needs: packets of chunks of `.rs` files.
    let mut rs_bytes = 0_usize;
    let mut rs_tokens = 0_usize;
    for c in &sub.chunks {
        if sub.files.get(c.file).is_some_and(|f| f.path.ends_with(".rs"))
            && let Some(p) = keys.get(&c.key)
        {
            rs_bytes += p.bytes();
            rs_tokens += p.tokens;
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "a rate")]
    let per_tok = st.enc_ms / st.tokens.max(1) as f64;
    let _ = writeln!(
        out,
        "background pass: {} chunks, {} encoded ({} tokens) in {:.1} s wall ({:.1} s encoding, {per_tok:.3} ms per token), waited {:.1} s at the device gate; faults {}; store {} packets, {} MB ({:.2} GB); Rust source alone (.rs chunks): {rs_tokens} tokens, {} MB ({:.2} GB); full keys and values of the same tokens would be {:.1} GB",
        st.chunks,
        st.encoded,
        st.tokens,
        st.wall_s,
        st.enc_ms / 1000.0,
        st.waited_s,
        if st.faults.is_empty() { "none".to_owned() } else { st.faults.join("; ") },
        kn,
        kb >> 20,
        kb as f64 / 1e9,
        rs_bytes >> 20,
        rs_bytes as f64 / 1e9,
        (rs_tokens * kvstore::bytes_per_token(&model, last + 1)) as f64 / 1e9
    );
    let _ = std::fs::write(rec, &out);

    // 2. The dumped questions.
    let mut rows: Vec<QRow> = Vec::new();
    let mut by_layer: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for idx in scan_dump::indices(dir).into_iter().take(n) {
        let Some(x) = items.get(idx) else { continue };
        let (q, scan) = scan_dump::load_question(dir, idx, &x.q)?;
        let mut row = QRow { id: q.id.clone(), exact_s: q.scan_s, exact_read_s: scan.read_ms / 1000.0, ..QRow::default() };
        let mut jobs = Vec::new();
        let t = Instant::now();
        for w in &scan.windows {
            let lens = nmsubs::lens::NAMES.get(w.tentacle).copied().unwrap_or("lens");
            let heads = sets.for_tentacle(lens);
            let mut packets = vec![Arc::new(nmengn::drone::DroneKeys::from_chunk(&ak, &keys.pairs, d.kv_heads, d.head_dim)?)];
            for c in &w.place {
                let key = sub.chunks.get(*c).map(|ch| ch.key.clone()).unwrap_or_default();
                packets.push(keys.get(&key).ok_or_else(|| bad(format!("question {idx}: no packet for chunk {c}")))?);
            }
            let (posed, null) = if scan.focused {
                let focus = tentacle_pass::FOCUS.get(w.tentacle).copied().unwrap_or("the code that does this");
                (format!("\nQuestion: {}\nFocus: {focus}", x.q), format!("\nQuestion: N/A\nFocus: {focus}"))
            } else {
                (format!("\nQuestion: {}", x.q), "\nQuestion: N/A".to_owned())
            };
            let gate = (dev == Device::Gpu).then(|| nmengn::placement::gpu_gate().enter("keys-only question path", true));
            let qp = nmengn::drone::question_path(&model, dev, &aw, &tok(&posed)?, w.len, heads)?;
            let qn = nmengn::drone::question_path(&model, dev, &aw, &tok(&null)?, w.len, heads)?;
            drop(gate);
            jobs.push(Job { packets, parts: w.parts.clone(), len: w.len, posed: qp, null: qn });
        }
        row.qpath_s = t.elapsed().as_secs_f64();
        let (tables, score_ms) = nmapi::key_scan::score_windows(&jobs, d.head_dim, group, d.theta);
        row.score_s = score_ms / 1000.0;
        let mut all_exact: Vec<(f32, bool)> = Vec::new();
        let mut all_keys: Vec<(f32, bool)> = Vec::new();
        for (w, tb) in scan.windows.iter().zip(tables) {
            let (kp, kn) = tb?;
            let parts = w.parts.len();
            let gold: Vec<bool> = w.place.iter().map(|c| q.gold.contains(c)).collect();
            let ex = above(&w.cal_posed, &w.cal_null, parts);
            let ks = above(&kp, &kn, parts);
            if ex.len() != ks.len() || ex.is_empty() {
                return Err(bad(format!("question {idx}: tables of {} and {} chunks", ex.len(), ks.len())));
            }
            row.windows += 1;
            row.same_first += usize::from(first_of(&ex) == first_of(&ks));
            if let Some(r) = spearman(&ex, &ks) {
                row.rho.push(r);
            }
            let (pe, pk) = (gold_place(&ex, &gold), gold_place(&ks, &gold));
            row.inside.0 = match (row.inside.0, pe) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            row.inside.1 = match (row.inside.1, pk) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            all_exact.extend(ex.iter().zip(&gold).map(|(s, g)| (*s, *g)));
            all_keys.extend(ks.iter().zip(&gold).map(|(s, g)| (*s, *g)));
            let heads_n = w.cal_posed.len() / parts.max(1);
            let lens = nmsubs::lens::NAMES.get(w.tentacle).copied().unwrap_or("lens");
            for (h, &(l, _)) in sets.for_tentacle(lens).iter().enumerate().take(heads_n) {
                if let Some(r) = spearman(&head_row(&w.cal_posed, &w.cal_null, parts, h), &head_row(&kp, &kn, parts, h)) {
                    by_layer.entry(l).or_default().push(r);
                }
            }
        }
        let place = |v: &mut Vec<(f32, bool)>| {
            v.sort_by(|a, b| b.0.total_cmp(&a.0));
            v.iter().position(|y| y.1)
        };
        row.across = (place(&mut all_exact), place(&mut all_keys));
        let _ = writeln!(
            out,
            "q{} ({}): inside its window exact {} keys {}; across windows exact {} keys {}; same first chunk {} of {} windows; rank correlation per window mean {:.3}; seconds exact scan {:.1} (reads {:.1}), keys-only {:.2} (question path {:.2}, scoring {:.2})",
            idx + 1,
            row.id,
            show(row.inside.0),
            show(row.inside.1),
            show(row.across.0),
            show(row.across.1),
            row.same_first,
            row.windows,
            mean(&row.rho),
            row.exact_s,
            row.exact_read_s,
            row.qpath_s + row.score_s,
            row.qpath_s,
            row.score_s
        );
        let _ = std::fs::write(rec, &out);
        rows.push(row);
    }

    // 3. Summary.
    let k = rows.len();
    let count = |f: &dyn Fn(&QRow) -> bool| rows.iter().filter(|r| f(r)).count();
    let mut rhos: Vec<f64> = rows.iter().flat_map(|r| r.rho.iter().copied()).collect();
    let windows: usize = rows.iter().map(|r| r.windows).sum();
    let same: usize = rows.iter().map(|r| r.same_first).sum();
    let held = count(&|r| r.inside.0.is_some());
    let _ = writeln!(out, "\nsummary over {k} questions ({held} with the right chunk in some window), {windows} windows");
    let _ = writeln!(out, "| measure | exact scan | keys-only scan |");
    let _ = writeln!(out, "|---|---|---|");
    let _ = writeln!(
        out,
        "| right chunk first inside its window | {} | {} |",
        count(&|r| r.inside.0 == Some(0)),
        count(&|r| r.inside.1 == Some(0))
    );
    let _ = writeln!(
        out,
        "| right chunk in the best 6 of its window | {} | {} |",
        count(&|r| r.inside.0.is_some_and(|p| p < 6)),
        count(&|r| r.inside.1.is_some_and(|p| p < 6))
    );
    let _ = writeln!(
        out,
        "| right chunk first across all windows | {} | {} |",
        count(&|r| r.across.0 == Some(0)),
        count(&|r| r.across.1 == Some(0))
    );
    let _ = writeln!(out, "| same chunk first in a window (keys against exact) | - | {same} of {windows} |");
    let _ =
        writeln!(out, "| rank correlation of chunk scores per window (mean, median) | - | {:.3}, {:.3} |", mean(&rhos), median(&mut rhos));
    let _ = writeln!(
        out,
        "| seconds per question (mean): exact scan as dumped (encode plus reads) | {:.1} | - |",
        mean(&rows.iter().map(|r| r.exact_s).collect::<Vec<_>>())
    );
    let _ = writeln!(
        out,
        "| seconds per question (mean): exact reads only (no encoding) | {:.1} | - |",
        mean(&rows.iter().map(|r| r.exact_read_s).collect::<Vec<_>>())
    );
    let _ = writeln!(
        out,
        "| seconds per question (mean): keys-only, packets held (question path plus scoring) | - | {:.2} ({:.2} + {:.2}) |",
        mean(&rows.iter().map(|r| r.qpath_s + r.score_s).collect::<Vec<_>>()),
        mean(&rows.iter().map(|r| r.qpath_s).collect::<Vec<_>>()),
        mean(&rows.iter().map(|r| r.score_s).collect::<Vec<_>>())
    );
    let _ = writeln!(
        out,
        "\nper drone layer: rank correlation of each head's chunk readings, keys-only against exact (mean over heads and windows)"
    );
    for (l, v) in &by_layer {
        let _ = writeln!(out, "layer {l}: {:.3} over {} head-windows", mean(v), v.len());
    }
    let ram = nmffi::dynlib::process_memory().map_or_else(|e| e, |(peak, now)| format!("RAM peak {} MB, now {} MB", peak >> 20, now >> 20));
    let _ = writeln!(out, "\ntotal {:.0} s; {ram}", started.elapsed().as_secs_f64());
    Ok(out)
}

/// `run keys [dump folder] [n]`.
pub fn run(args: &[String], home: &Path) -> ExitCode {
    let stamp = nmtype::now_ms();
    let path = home.join("runs").join(format!("KEYS-scan-{stamp}.txt"));
    let dir = args
        .get(2)
        .filter(|a| a.parse::<usize>().is_err())
        .map(PathBuf::from)
        .map(|p| if p.is_absolute() { p } else { home.join(p) })
        .or_else(|| {
            let mut v: Vec<PathBuf> = std::fs::read_dir(home.join("scratch/tc1-dump"))
                .ok()?
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.join("substrate.json").is_file())
                .collect();
            v.sort();
            v.pop()
        });
    let n = args.iter().skip(2).find_map(|a| a.parse::<usize>().ok()).unwrap_or(usize::MAX);
    let result = dir.map_or_else(|| Err(bad("no dump under scratch/tc1-dump")), |d| measure(home, &d, n, &path));
    let (text, ok) = match result {
        Ok(t) => (t, true),
        Err(e) => (format!("{}FAULT {}: {}\n", std::fs::read_to_string(&path).unwrap_or_default(), e.component, e.message), false),
    };
    if let Err(e) = std::fs::write(&path, &text) {
        eprintln!("keys: {}: {e}", path.display());
        return ExitCode::FAILURE;
    }
    println!("{text}\nreport: {}", path.display());
    if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}
