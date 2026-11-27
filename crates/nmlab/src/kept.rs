//! Part test of the kept encoded chunks (2026-11-10; direction of Luis Jake Gabriel III: each
//! part gets a short isolated test before any long whole-system run). One process runs the
//! tentacle pass on the first three development questions by index (the question file `g1.rs`
//! reads, `crate::e2::suite`), the served settings, and prints per question: seconds, tokens
//! encoded (and how many of them had been encoded before and dropped at the limit), tokens
//! held, bytes held, the RAM limit and how it was set.
//!
//! Pass rule (printed with the result): question 1 may encode everything; questions 2 and 3
//! encode 0 tokens and take seconds, not minutes (under 60 s each).
//!
//! `run kept`: report `runs/kept-chunks-<ms>.txt`, a new file per run. Since 2026-11-10 (step
//! P2) each question's row is followed by a memory line: the process's peak and current RAM
//! (working set), the device bytes the process holds (kept weights and windows, working
//! buffers, idle pool buffers), the device's free memory, and the device budget's figures.
//! `run kept layout`: the same three questions' windows only (no encoding, no read): tokens per
//! question, tokens of chunks an earlier question's windows already held, the union, the whole
//! tree, and the RAM each needs at the bytes per token kept; report
//! `runs/kept-chunks-layout-<ms>.txt`, a new file per run.
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use nmapi::forage::Organism;
use nmapi::tentacle_pass::{Settings, TentaclePass, drone_last_layer};
use nmtype::{Fault, FaultKind};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("kept", FaultKind::Execution, m)
}

/// Questions read, by index from the start of the development half.
const QUESTIONS: usize = 3;
/// Seconds a later question may take ("seconds, not minutes"): the test gate.
const LATER_S: f64 = 60.0;

/// The organism (memory off: nothing is written to the memory folder), with the qualified
/// embedder, reranker and dense lane bound as served.
fn open(home: &Path, out: &mut String) -> Result<Organism, Fault> {
    let org = Organism::open_with(home, home, &["neuromorph-nano".to_owned()], false, &home.join("memory"))?;
    let _ = writeln!(out, "neural: {}", nmapi::neural::bind(&org)?);
    let _ = writeln!(out, "dense: {}", nmapi::neural::start_dense(&org)?);
    Ok(org)
}

fn test(home: &Path) -> Result<(String, bool), Fault> {
    let mut items = crate::e2::suite(home)?;
    items.truncate(QUESTIONS);
    let mut out = String::new();
    let _ = writeln!(out, "kept encoded chunks part test, questions 1 to {} of the development half (by index)", items.len());
    let org = open(home, &mut out)?;
    let sub = org.sub();
    let settings = Settings::default();
    let pass = TentaclePass::new(home, settings);
    let _ = writeln!(
        out,
        "settings: served defaults (judge {}, score {}, shortlist {}, cache_bytes {} = derived)",
        settings.judge.name(),
        settings.score.name(),
        settings.shortlist,
        settings.cache_bytes
    );
    let _ = writeln!(
        out,
        "q  seconds  tokens_encoded  encoded_again  tokens_held  bytes_held  chunks_held  limit_bytes  free_ram_read  bytes_per_token  last_layer"
    );
    let mut ok = true;
    let mut notes = Vec::new();
    for (i, x) in items.iter().enumerate() {
        let t = Instant::now();
        let p = pass.read(&sub, &org.slots, &x.q)?;
        let secs = t.elapsed().as_secs_f64();
        let scan = pass.last_scan().ok_or_else(|| bad("no scan kept"))?;
        let _ = writeln!(
            out,
            "{}  {secs:.1}  {}  {}  {}  {}  {}  {}{}  {}  {}  {}",
            i + 1,
            scan.enc_tokens,
            scan.enc_again_tokens,
            scan.window_tokens,
            scan.held_bytes,
            scan.held_chunks,
            scan.limit.bytes,
            if scan.limit.fallback { " (fallback)" } else { "" },
            scan.limit.free,
            scan.bytes_per_token,
            scan.last_layer
        );
        if i > 0 && (scan.enc_tokens != 0 || secs >= LATER_S) {
            ok = false;
        }
        notes.push(p.note);
        let _ = writeln!(out, "   memory after question {}: {}", i + 1, memory_line());
    }
    let _ = writeln!(out, "pass notes:");
    for (i, n) in notes.iter().enumerate() {
        let _ = writeln!(out, "  {}: {n}", i + 1);
    }
    let _ = writeln!(out, "rule: question 1 may encode everything; questions 2 and 3 encode 0 tokens and take under {LATER_S:.0} s each");
    let _ = writeln!(out, "verdict: {}", if ok { "pass" } else { "FAIL" });
    Ok((out, ok))
}

/// RAM and device memory now (step P2): peak and current working set of this process, device
/// bytes held by it through the pool (kept, working, idle) and their peak working bytes, the
/// device's free and total memory.
fn memory_line() -> String {
    let ram = nmffi::dynlib::process_memory().map_or_else(|e| e, |(peak, now)| format!("RAM peak {} MB, now {} MB", peak >> 20, now >> 20));
    let dev = nmapi::neural::model(&nmtype::nano_root()).ok().and_then(|m| m.pop.gpu.clone()).map_or_else(
        || "no device".to_owned(),
        |o| {
            let s = o.gpu.pool_stats();
            let free = o.gpu.memory().map_or_else(|e| e, |(f, t)| format!("device free {} MB of {} MB", f >> 20, t >> 20));
            format!("device held by the process {} MB (kept weights and windows {} MB, working {} MB, idle pool {} MB; working peak since the last run start {} MB), {free}", (s.live_kept + s.live_work + s.idle_bytes) >> 20, s.live_kept >> 20, s.live_work >> 20, s.idle_bytes >> 20, s.peak_work >> 20)
        },
    );
    format!("{ram}; {dev}; {}", nmengn::residency::get().report())
}

fn layout(home: &Path) -> Result<String, Fault> {
    let mut items = crate::e2::suite(home)?;
    items.truncate(QUESTIONS);
    let mut out = String::new();
    let _ = writeln!(out, "kept encoded chunks, window layout only, questions 1 to {} of the development half (by index)", items.len());
    let org = open(home, &mut out)?;
    let sub = org.sub();
    let pass = TentaclePass::new(home, Settings::default());
    let model = nmapi::neural::model(home)?;
    let per = nmengn::kvstore::bytes_per_token(&model, drone_last_layer() + 1);
    let limit = nmengn::kvstore::derive_ram_limit();
    let _ = writeln!(
        out,
        "bytes per token kept {per} (layers 0..={}); RAM limit now {} bytes (free RAM read {} of {})",
        drone_last_layer(),
        limit.bytes,
        limit.free,
        limit.total
    );
    let _ = writeln!(out, "q  windows_with_chunks  chunks  tokens  tokens_in_earlier_windows  tokens_new  union_tokens  union_bytes");
    let mut seen: BTreeSet<usize> = BTreeSet::new();
    let mut tree = 0_usize;
    let mut union_tokens = 0_usize;
    for (i, x) in items.iter().enumerate() {
        let (tokens, windows) = pass.layout(&sub, &x.q)?;
        tree = tokens.iter().filter(|t| **t != usize::MAX).sum();
        let tk = |c: &usize| tokens.get(*c).copied().unwrap_or(0);
        let chunks: Vec<usize> = windows.iter().flatten().copied().collect();
        let all: usize = chunks.iter().map(tk).sum();
        let earlier: usize = chunks.iter().filter(|c| seen.contains(c)).map(tk).sum();
        for c in &chunks {
            if seen.insert(*c) {
                union_tokens += tk(c);
            }
        }
        let _ = writeln!(
            out,
            "{}  {}  {}  {all}  {earlier}  {}  {union_tokens}  {}",
            i + 1,
            windows.iter().filter(|w| !w.is_empty()).count(),
            chunks.len(),
            all - earlier,
            union_tokens * per
        );
    }
    let _ = writeln!(
        out,
        "whole tree: {} chunks, {tree} tokens, {} bytes at {per} bytes per token",
        sub.chunks.len(),
        tree.saturating_mul(per)
    );
    Ok(out)
}

/// `run kept [layout]`.
pub fn run(args: &[String], home: &Path) -> ExitCode {
    let stamp = nmtype::now_ms();
    if args.get(2).map(String::as_str) == Some("layout") {
        let txt = home.join(format!("runs/kept-chunks-layout-{stamp}.txt"));
        return match layout(home) {
            Ok(text) => {
                let _ = std::fs::write(&txt, &text);
                print!("{text}");
                println!("report: {}", txt.display());
                ExitCode::SUCCESS
            }
            Err(e) => {
                let _ = std::fs::write(&txt, format!("FAULT {e}\n"));
                eprintln!("kept layout: {e} (report {})", txt.display());
                ExitCode::FAILURE
            }
        };
    }
    let txt = home.join(format!("runs/kept-chunks-{stamp}.txt"));
    match test(home) {
        Ok((text, ok)) => {
            let _ = std::fs::write(&txt, &text);
            print!("{text}");
            println!("report: {}", txt.display());
            if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE }
        }
        Err(e) => {
            let _ = std::fs::write(&txt, format!("FAULT {e}\n"));
            eprintln!("kept: {e} (report {})", txt.display());
            ExitCode::FAILURE
        }
    }
}
