//! Stage 5 WP-C3 J3: the cross-call judge score cache tested directly (not through the whole
//! pack's answer cache): the close-call judge (`nmapi::cache_reads::judge_close`) asked twice for
//! the same question and pieces returns identical values, the second time from the cache (hits rise
//! by the batch, no new misses) and faster; a batch with one new piece is read whole; with
//! `NANO_JUDGE_CACHE=off` semantics checked by `nmapi::neural::judge_close` giving the same values.
//! The served switches are set as `serve` sets them.
//!
//! Run (card, under `scratch/LOCKS/card.lock`):
//! `cargo test --release -p nmapi --test judge_cache -- --ignored --nocapture`.
use std::sync::atomic::Ordering;
use std::time::Instant;

#[test]
#[ignore = "card job: needs the judge"]
fn judge_cache_serves_repeats_exactly() {
    if let Err(e) = run() {
        panic!("judge cache: {e}");
    }
}

fn run() -> Result<(), String> {
    let home = nmtype::nano_root();
    nmengn::gpu::HALF_MATH.store(true, Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
    let judge = nmapi::neural::reranker_model(&home).map_err(|e| e.message)?;
    if judge.pop.gpu.is_none() {
        return Err("no card".to_owned());
    }
    let src = std::fs::read_to_string(home.join("crates/nmapi/src/cache_reads.rs")).map_err(|e| e.to_string())?;
    let lines: Vec<&str> = src.lines().collect();
    let docs: Vec<String> = lines.chunks(14).take(6).map(|c| c.join("\n")).collect();
    let question = "Where is the least recently used entry dropped when the cache is full?";

    // The uncached reference.
    let reference = nmapi::neural::judge_close(&judge, question, &docs).map_err(|e| e.message)?;

    let (h0, m0) = nmapi::cache_reads::judge_stats();
    let t1 = Instant::now();
    let first = nmapi::cache_reads::judge_close(&judge, question, &docs).map_err(|e| e.message)?;
    let first_s = t1.elapsed().as_secs_f64();
    let (h1, m1) = nmapi::cache_reads::judge_stats();
    let t2 = Instant::now();
    let second = nmapi::cache_reads::judge_close(&judge, question, &docs).map_err(|e| e.message)?;
    let second_s = t2.elapsed().as_secs_f64();
    let (h2, m2) = nmapi::cache_reads::judge_stats();
    println!("first {first_s:.4} s ({} misses), second {second_s:.6} s ({} hits)", m1 - m0, h2 - h1);
    if first != reference {
        return Err(format!("first cached read differs from the uncached read: {first:?} against {reference:?}"));
    }
    if second != first {
        return Err("the repeat differs from the first read".to_owned());
    }
    let n = docs.len() as u64;
    if h1 != h0 || m1 - m0 != n || h2 - h1 != n || m2 != m1 {
        return Err(format!("counters: hits {h0} to {h1} to {h2}, misses {m0} to {m1} to {m2}"));
    }
    if second_s >= first_s {
        return Err(format!("the repeat was not faster: {second_s} against {first_s}"));
    }
    // One new piece: the whole batch is read (exact), the old pieces' values unchanged.
    let mut more = docs.clone();
    more.push(lines.iter().take(10).copied().collect::<Vec<_>>().join("\n"));
    let third = nmapi::cache_reads::judge_close(&judge, question, &more).map_err(|e| e.message)?;
    let (h3, m3) = nmapi::cache_reads::judge_stats();
    if h3 != h2 || m3 - m2 != more.len() as u64 {
        return Err(format!("a partly new batch must be read whole: hits {h2} to {h3}, misses {m2} to {m3}"));
    }
    let uncached = nmapi::neural::judge_close(&judge, question, &more).map_err(|e| e.message)?;
    if third != uncached {
        return Err("the partly new batch differs from its uncached read".to_owned());
    }
    Ok(())
}
