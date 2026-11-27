//! Fitting commands for the project's adapter cartridges (lane B, 2026-11-19; `nmapi::adapter`,
//! `nmapi::adapter_fit`): the same fit `serve` runs at idle with `NANO_ADAPTER_FIT=on`, run once over
//! a material root from the lab.
//!
//! - `run adapter labels [--root R] [--pairs N]`: the labels only (processor, no model): pairs from
//!   commit messages and from doc comments hidden from their definitions, by fold, with examples.
//! - `run adapter fit [--root R] [--pairs N] [--judge-pairs N] [--no-lens] [--no-judge] [--write]
//!   [--force] [--wait S]`: the organism opened on the root with memory off, the embedder bound and
//!   the meaning lane started (`NANO_DENSE_DIR` may name a store of its vectors), waiting up to S
//!   seconds (default 3600) for the lane to be full; then the lens fit and the judge's code read-out
//!   fit, each reported on its held-out report half against the base and a shuffled-label control.
//!   Accepted adapters are written to the shelf only with `--write`.
//!
//! Reports: `runs/ADAPTER-<step>-<ms>.txt`, a new file per run.
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use nmapi::adapter_fit::{self, Opts};
use nmapi::forage::Organism;
use nmtype::{Fault, FaultKind};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("adapter", FaultKind::Execution, m)
}

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2).find(|w| w.first().map(String::as_str) == Some(name)).and_then(|w| w.get(1)).map(String::as_str)
}
fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn root_of(home: &Path, args: &[String]) -> PathBuf {
    arg(args, "--root").map_or_else(|| home.to_path_buf(), PathBuf::from)
}

fn labels(home: &Path, args: &[String]) -> Result<String, Fault> {
    let root = root_of(home, args);
    let org = Organism::open_with(home, &root, &[], false, &home.join("memory"))?;
    let sub = org.sub();
    let cap = arg(args, "--pairs").and_then(|v| v.parse().ok()).unwrap_or(Opts::default().pairs);
    let pairs = adapter_fit::labels(&sub, cap);
    let mut out = format!("adapter labels over {} ({} pieces, {} files)\n", root.display(), sub.chunks.len(), sub.files.len());
    let by = |src: &str, f: u8| pairs.iter().filter(|p| p.source == src && adapter_fit::fold(&p.file) == f).count();
    for src in ["commit", "doc"] {
        let _ = writeln!(out, "{src}: fit {}, choose {}, report {}", by(src, 0), by(src, 1), by(src, 2));
    }
    let _ = writeln!(out, "labels key {}", adapter_fit::labels_key(&sub, &pairs));
    for p in pairs.iter().step_by((pairs.len() / 12).max(1)).take(12) {
        let place = sub.chunks.get(p.target).map_or(0, |c| c.start);
        let q: String = p.query.chars().take(120).collect();
        let _ = writeln!(out, "  [{}] {}:{place}  <-  {q}", p.source, p.file);
    }
    Ok(out)
}

fn seat_served_base_and_q8(home: &Path) {
    // Match `serve`'s GPU math switches so dense vector stores written by `nano-lab run adapter fit`
    // are 100% header-compatible with `serve` and `bench --mode board` (no 30-minute re-embedding).
    nmengn::gpu::HALF_MATH.store(true, std::sync::atomic::Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, std::sync::atomic::Ordering::Relaxed);
    if let Ok(mut p) = nmapi::neural::PRECISION.write() {
        p.base = p.base.or(Some(8));
        p.embedder = p.embedder.or(Some(8));
        p.reranker = p.reranker.or(Some(8));
        p.writer = p.writer.or(Some(8));
    }
    if let Ok(lib) = nmlearn::library::Library::open(home)
        && let Ok(mut offered) = nmapi::neural::BASE_CARTRIDGES.write()
    {
        for (model, id) in [
            ("qwen3-0.6b", "reader-base"),
            ("qwen3-embedding-0.6b", "embedder-base"),
            ("qwen3-reranker-0.6b", "reranker-base"),
            ("qwen3-1.7b", "writer-base"),
        ] {
            if let Some(c) = lib.cartridge(id)
                && let Some(first) = c.files.first()
            {
                offered.insert(model.to_owned(), vec![lib.cartridge_dir(c).join(&first.name)]);
            }
        }
    }
}

fn fit(home: &Path, args: &[String]) -> Result<String, Fault> {
    seat_served_base_and_q8(home);
    let root = root_of(home, args);
    let org = Organism::open_with(home, &root, &[], false, &home.join("memory"))?;
    let mut out = String::new();
    let _ = writeln!(out, "embedder: {}", nmapi::neural::bind_embedder(&org)?);
    let _ = writeln!(out, "dense: {}", nmapi::neural::start_dense(&org)?);
    let wait = arg(args, "--wait").and_then(|v| v.parse::<u64>().ok()).unwrap_or(3600);
    let started = std::time::Instant::now();
    loop {
        let sub = org.sub();
        let lane = sub.dense.read().ok().and_then(|d| d.clone()).ok_or_else(|| bad("no meaning lane"))?;
        if lane.ready(&sub) {
            break;
        }
        if started.elapsed().as_secs() > wait {
            let (covered, total) = lane.coverage(&sub);
            return Err(bad(format!("the meaning lane is not full after {wait} s ({covered} of {total})")));
        }
        std::thread::park_timeout(std::time::Duration::from_secs(2));
    }
    let _ = writeln!(out, "meaning lane full after {:.0} s", started.elapsed().as_secs_f64());
    let d = Opts::default();
    let opts = Opts {
        pairs: arg(args, "--pairs").and_then(|v| v.parse().ok()).unwrap_or(d.pairs),
        judge_pairs: arg(args, "--judge-pairs").and_then(|v| v.parse().ok()).unwrap_or(d.judge_pairs),
        lens: !flag(args, "--no-lens"),
        judge: !flag(args, "--no-judge"),
        write: flag(args, "--write"),
        force: flag(args, "--force"),
        questions: arg(args, "--questions").map(|p| tuning_answers(home, p)).transpose()?.unwrap_or_default(),
        check_only: flag(args, "--check-only"),
        beta: arg(args, "--beta").and_then(|v| v.parse::<f32>().ok()),
    };
    out.push_str(&adapter_fit::fit(&org, &opts, &|| false)?);
    Ok(out)
}

/// The tuning questions of `path` with their first answer place each, for the lens fit's question
/// check (roadmap Phase 0 item 3). The unseen verification set is refused.
fn tuning_answers(home: &Path, path: &str) -> Result<Vec<(String, String, usize, usize)>, Fault> {
    if path.contains("dev-large-2") || path.contains("real-use-2") {
        return Err(bad("--questions: the unseen verification and held-out sets are never used in a fit"));
    }
    let text = std::fs::read(home.join(path)).map_err(|e| bad(format!("{path}: {e}")))?;
    let v = nmjson::parse(&text).map_err(|e| bad(format!("{path}: {e}")))?;
    let list = v.get("questions").ok().and_then(|q| q.array().ok()).or_else(|| v.array().ok()).ok_or_else(|| bad(format!("{path}: no questions")))?;
    let mut out = Vec::new();
    for q in list {
        let Some(question) = q.get("question").ok().and_then(|x| x.string().ok()) else { continue };
        // Every answer place of the question (pack and every range need them all, not only the first).
        let Some(answers) = q.get("answers").ok().and_then(|a| a.array().ok()) else { continue };
        for ans in answers {
            let Some(file) = ans.get("file").ok().and_then(|x| x.string().ok()) else { continue };
            let lines = ans.get("lines").ok().and_then(|l| l.array().ok());
            let at = |i: usize| lines.and_then(|l| l.get(i)).and_then(|x| x.integer().ok()).and_then(|x| usize::try_from(x).ok());
            let (Some(a), Some(b)) = (at(0), at(1)) else { continue };
            out.push((question.to_owned(), file.replace('\\', "/"), a, b));
        }
    }
    Ok(out)
}

pub fn run(args: &[String], home: &Path) -> ExitCode {
    let step = args.get(2).map_or("labels", String::as_str);
    let stamp = nmtype::now_ms();
    let result = match step {
        "labels" => labels(home, args),
        "fit" => fit(home, args),
        other => Err(bad(format!("unknown step {other} (labels, fit)"))),
    };
    match result {
        Ok(t) => {
            let p = home.join(format!("runs/ADAPTER-{step}-{stamp}.txt"));
            let _ = std::fs::write(&p, format!("nano-lab {}\n{t}", args.get(1..).unwrap_or_default().join(" ")));
            print!("{t}");
            println!("report: {}", p.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("adapter {step}: {e}");
            ExitCode::FAILURE
        }
    }
}
