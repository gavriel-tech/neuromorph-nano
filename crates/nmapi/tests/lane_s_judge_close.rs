//! Lane S (2026-11-18, T-079): the close-call arm's judge (`nmapi::neural::judge_close`) as lane
//! G's step J test runs it (the process's served switches off, as in `crates/nmlab/tests/j_close.rs`)
//! and as `serve` runs it (half math and packed attention on, `nmcore` `serve` defaults): seconds per
//! question for 10 candidates cut by `close_doc` (about 128 tokens), and the largest change of the
//! clamped log-odds between the two. Candidates are pieces of this tree's engine sources.
//!
//! Run (card, under `scratch/LOCKS/card.lock`):
//! `cargo test --release -p nmapi --test lane_s_judge_close -- --ignored --nocapture`.
use std::sync::atomic::Ordering;
use std::time::Instant;

#[test]
#[ignore = "card job: needs the judge"]
fn judge_close_switches() {
    if let Err(e) = run() {
        panic!("lane S judge_close: {e}");
    }
}

fn run() -> Result<(), String> {
    let home = nmtype::nano_root();
    // `NANO_LANE_S_Q8=1`: the judge bound at 8 bits (as `serve --reranker-q8`).
    let q8 = std::env::var("NANO_LANE_S_Q8").is_ok();
    if q8 {
        nmapi::neural::RERANK_Q8.store(true, Ordering::Relaxed);
    }
    let judge = nmapi::neural::reranker_model(&home).map_err(|e| e.message)?;
    if judge.pop.gpu.is_none() {
        return Err("no card".to_owned());
    }
    let mut pieces: Vec<(String, String)> = Vec::new();
    for f in ["crates/nmengn/src/model.rs", "crates/nmengn/src/kvstore.rs", "crates/nmengn/src/gpu.rs", "crates/nmengn/src/roles.rs"] {
        let text = std::fs::read_to_string(home.join(f)).map_err(|e| format!("{f}: {e}"))?;
        let lines: Vec<&str> = text.lines().collect();
        for k in 0..15 {
            let first = 40 + k * 60;
            let body = lines.iter().skip(first).take(40).copied().collect::<Vec<_>>().join("\n");
            pieces.push((format!("{f}:{}-{}", first + 1, first + 40), body));
        }
    }
    let questions = [
        "Where is an activation released after its last reader has run?",
        "How does the engine choose the kernel for an 8-bit linear?",
        "Where are packed sequences read in one attention launch?",
        "How is the key and value cache kept between decode steps?",
        "Where is the judge's prompt head prefilled once?",
    ];
    let docs_for = |q: usize| -> Vec<String> {
        (0..10).map(|i| {
            let (place, text) = &pieces[(q * 7 + i * 5) % pieces.len()];
            nmapi::neural::close_doc(&judge, place, text)
        }).collect()
    };
    let pass = |label: &str| -> Result<(f64, Vec<f32>), String> {
        let _ = nmapi::neural::judge_close(&judge, questions[0], &docs_for(0)).map_err(|e| e.message)?;
        let mut all = Vec::new();
        let t = Instant::now();
        for (q, question) in questions.iter().enumerate() {
            all.extend(nmapi::neural::judge_close(&judge, question, &docs_for(q)).map_err(|e| e.message)?);
        }
        #[expect(clippy::cast_precision_loss, reason = "a mean for a report")]
        let per = t.elapsed().as_secs_f64() / questions.len() as f64;
        eprintln!("{label}: {per:.3} s a question (10 candidates, {} questions)", questions.len());
        Ok((per, all))
    };
    nmengn::gpu::HALF_MATH.store(false, Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(false, Ordering::Relaxed);
    let (_, off) = pass("switches off (as the step J test runs)")?;
    nmengn::gpu::HALF_MATH.store(true, Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
    let (_, on) = pass("serve's switches on (half math, packed attention)")?;
    let mut widths: Vec<&str> = judge.asm.nodes.iter().filter(|n| n.op == "linear").map(|n| n.precision.as_str()).collect();
    widths.sort_unstable();
    widths.dedup();
    eprintln!("judge linears at {widths:?}; card bytes {}", judge.pop.device_bytes());
    nmffi::prof::reset();
    nmffi::prof::set(true);
    nmffi::prof::set_kernels(true);
    let _ = nmapi::neural::judge_close(&judge, questions[1], &docs_for(1)).map_err(|e| e.message)?;
    nmffi::prof::set_kernels(false);
    nmffi::prof::set(false);
    let mut k: Vec<(String, f64, u64)> = nmffi::prof::side().into_iter().filter(|r| r.0.starts_with("kernel ")).collect();
    k.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (name, secs, c) in k.iter().take(8) {
        eprintln!("    {:7.2} ms {c:6}  {name}", secs * 1000.0);
    }
    let worst = off.iter().zip(&on).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    let rank = |v: &[f32]| {
        let mut o: Vec<usize> = (0..v.len()).collect();
        o.sort_by(|a, b| v[*b].total_cmp(&v[*a]));
        o
    };
    let same_first = off.chunks(10).zip(on.chunks(10)).filter(|(a, b)| rank(a).first() == rank(b).first()).count();
    eprintln!("largest log-odds change {worst:.3}; first candidate the same on {same_first} of {} questions", questions.len());
    let dir = home.join("scratch/lane-s");
    let name = if q8 { "judge-close-q8.f32" } else { "judge-close-16.f32" };
    let _ = std::fs::write(dir.join(name), on.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    if q8 && let Ok(b) = std::fs::read(dir.join("judge-close-16.f32")) {
        let r: Vec<f32> = b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        let w = r.iter().zip(&on).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        let sf = r.chunks(10).zip(on.chunks(10)).filter(|(a, b)| rank(a).first() == rank(b).first()).count();
        let so = r.chunks(10).zip(on.chunks(10)).filter(|(a, b)| rank(a) == rank(b)).count();
        eprintln!("8 bits against 16 bits (served switches): largest log-odds change {w:.3}; first the same on {sf}, whole order on {so} of {}", questions.len());
    }
    Ok(())
}
