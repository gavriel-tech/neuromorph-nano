//! Phase C1 and D2 tests of the tentacle master plan (2026-11-18), on synthetic relevance keys of
//! the default drone set's geometry (16 heads over 13 (layer, key-value head) pairs, head width
//! 128, two query heads per key-value head, rope theta 1,000,000) and 12,000-token windows:
//! - `d2_kernel_equals_old`: the periphery kernel (`scan_pieces`, log-sum-exp form, every token)
//!   equals the old path (`scan_keys` then the per-part log-sum-exp of `nmapi::tentacle_pass`)
//!   within tolerance; its log-mean-exp form is the log-sum-exp form minus the log of the count;
//!   timing per window per thread for both (run with `--release --nocapture`).
//! - `proc1_budget`: a window scan inside the one processor budget while another parallel stage
//!   runs, against the scan alone, and against the same load outside the budget (the five pools
//!   before Phase C1); scores identical across loads.
use nmengn::drone::{Queries, Reduce, RelevanceKeys, scan_keys, scan_pieces};
use std::sync::Arc;
use std::time::Instant;

const D: usize = 128;
const GROUP: usize = 2;
const WINDOW: usize = 12_000;

/// A small deterministic generator (xorshift) giving roughly normal values (sum of four uniforms).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        let mut t = 0.0;
        for _ in 0..4 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            #[expect(clippy::cast_precision_loss, reason = "a test value")]
            let u = (self.0 >> 40) as f32 / (1u64 << 24) as f32;
            t += u - 0.5;
        }
        t * 1.7
    }
}

fn heads() -> Vec<(usize, usize)> {
    // 13 pairs; the first three read both query heads of their key-value head: 16 heads.
    let mut h = Vec::new();
    for p in 0..13 {
        let (l, g) = (3 + p, p % 8);
        h.push((l, g * GROUP));
        if p < 3 {
            h.push((l, g * GROUP + 1));
        }
    }
    h
}

/// A window of pieces (anchor of 64 tokens, then pieces of 50 to 450 tokens) filling `WINDOW`.
fn window(seed: u64) -> (Vec<Arc<RelevanceKeys>>, Vec<(usize, usize)>) {
    let pairs = nmengn::drone::pairs(&heads(), GROUP);
    let mut r = Rng(seed);
    let mut parts = Vec::new();
    let mut at_list = Vec::new();
    let mut at = 0;
    let mut i = 0_usize;
    while at < WINDOW {
        let want = if i == 0 { 64 } else { 50 + (i * 97) % 400 };
        let n = want.min(WINDOW - at);
        let rows = pairs.iter().map(|_| (0..n * D).map(|_| nmengn::half::bits16(r.next())).collect()).collect();
        parts.push(Arc::new(RelevanceKeys { tokens: n, pairs: pairs.clone(), rows }));
        at_list.push((at, n));
        at += n;
        i += 1;
    }
    (parts, at_list)
}

fn queries(seed: u64, tokens: usize) -> Queries {
    let mut r = Rng(seed);
    let h = heads();
    let q = h.iter().map(|_| (0..tokens * D).map(|_| r.next() * 0.6).collect()).collect();
    Queries { tokens, heads: h, q }
}

/// The old per-part reduction (`nmapi::tentacle_pass::part_lse`, copied).
fn part_lse(logits: &[f32], tokens: usize, span: usize, parts: &[(usize, usize)]) -> Vec<f32> {
    #[expect(clippy::cast_precision_loss, reason = "token counts are small")]
    let per = 1.0 / tokens.max(1) as f32;
    parts
        .iter()
        .map(|&(at, len)| {
            let mut acc = 0.0_f32;
            for t in 0..tokens {
                let row = logits.get(t * span + at..t * span + at + len).unwrap_or(&[]);
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if max.is_finite() {
                    acc += max + row.iter().map(|x| (x - max).exp()).sum::<f32>().ln();
                }
            }
            acc * per
        })
        .collect()
}

fn old_rows(q: &Queries, parts: &[Arc<RelevanceKeys>], at: &[(usize, usize)], table: &[(f32, f32)]) -> Vec<f32> {
    let logits = scan_keys(q, parts, WINDOW, table, D, GROUP).expect("old scan");
    logits.iter().flat_map(|lg| part_lse(lg, q.tokens, WINDOW, at)).collect()
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

#[test]
fn d2_kernel_equals_old() {
    let table = nmengn::cpu::rope_table(WINDOW, D, 1_000_000.0);
    let (parts, at) = window(7);
    let q = queries(11, 20);
    // Best of three each (the machine is shared with other lanes).
    let (mut old, mut new) = (Vec::new(), Vec::new());
    let (mut old_ms, mut new_ms) = (f64::MAX, f64::MAX);
    for _ in 0..3 {
        let t = Instant::now();
        old = old_rows(&q, &parts, &at, &table);
        old_ms = old_ms.min(ms(t));
        let t = Instant::now();
        new = scan_pieces(&q, &parts, &at, WINDOW, &table, (D, GROUP), None, Reduce::Sum).expect("new scan");
        new_ms = new_ms.min(ms(t));
    }
    assert_eq!(old.len(), new.len());
    let worst = old.iter().zip(&new).map(|(a, b)| (a - b).abs() / a.abs().max(1.0)).fold(0.0_f32, f32::max);
    // Same order of pieces per head.
    let mean = scan_pieces(&q, &parts, &at, WINDOW, &table, (D, GROUP), None, Reduce::Mean).expect("mean scan");
    #[expect(clippy::cast_precision_loss, reason = "piece lengths are small")]
    let shift_ok = new
        .iter()
        .zip(&mean)
        .enumerate()
        .all(|(k, (s, m))| at.get(k % at.len()).is_some_and(|&(_, n)| ((s - (n as f32).ln()) - m).abs() < 1e-4));
    // Content tokens only: the same as the old path on the queries cut to those tokens.
    let content = [0_usize, 3, 4, 9, 15, 19];
    let sub = Queries {
        tokens: content.len(),
        heads: q.heads.clone(),
        q: q.q.iter().map(|r| content.iter().flat_map(|t| r.get(t * D..(t + 1) * D).unwrap_or(&[]).to_vec()).collect()).collect(),
    };
    let old_sub = old_rows(&sub, &parts, &at, &table);
    let new_sub = scan_pieces(&q, &parts, &at, WINDOW, &table, (D, GROUP), Some(&content), Reduce::Sum).expect("content scan");
    let worst_sub = old_sub.iter().zip(&new_sub).map(|(a, b)| (a - b).abs() / a.abs().max(1.0)).fold(0.0_f32, f32::max);
    println!(
        "D2 kernel: {} heads x {} pieces over {WINDOW} positions, {} query tokens; largest relative difference from the old path {worst:.2e} (content tokens {worst_sub:.2e}); log-mean-exp = log-sum-exp - ln(count): {shift_ok}; old {old_ms:.1} ms, new {new_ms:.1} ms per window per thread ({:.1} times)",
        q.heads.len(),
        at.len(),
        q.tokens,
        old_ms / new_ms.max(1e-9)
    );
    assert!(worst < 1e-4 && worst_sub < 1e-4, "kernel differs from the old path: {worst} {worst_sub}");
    assert!(shift_ok, "log-mean-exp is not log-sum-exp minus ln(count)");
}

/// A stage of `parts` partitions each spinning arithmetic for about `each_ms` milliseconds.
fn busy(each_ms: f64) -> f32 {
    let t = Instant::now();
    let mut x = 1.0_f32;
    while ms(t) < each_ms {
        for _ in 0..10_000 {
            x = (x * 1.000_001).sin() + 1.0;
        }
    }
    x
}

#[test]
fn proc1_budget() {
    let table = nmengn::cpu::rope_table(WINDOW, D, 1_000_000.0);
    let tentacles: usize = 8;
    let wins: Vec<_> = (0..tentacles as u64).map(|i| window(100 + i)).collect();
    let q = queries(11, 12);
    let scan = |i: usize| -> (Vec<f32>, f64) {
        let (parts, at) = wins.get(i).expect("window");
        let t = Instant::now();
        let r = scan_pieces(&q, parts, at, WINDOW, &table, (D, GROUP), None, Reduce::Mean).expect("scan");
        (r, ms(t))
    };
    // Alone: window 0, nothing else running (best of three).
    let mut alone = f64::MAX;
    let mut alone_rows = Vec::new();
    for _ in 0..3 {
        let (r, t) = scan(0);
        alone = alone.min(t);
        alone_rows = r;
    }
    let total = nmtype::budget::total();
    let stop = std::sync::atomic::AtomicBool::new(false);
    let median = |v: &[(Vec<f32>, f64)]| {
        let mut t: Vec<f64> = v.iter().map(|x| x.1).collect();
        t.sort_by(f64::total_cmp);
        t.get(t.len() / 2).copied().unwrap_or(0.0)
    };
    // The eight tentacles' scans as one foreground stage, nothing else running.
    let stage = || -> Vec<(Vec<f32>, f64)> { nmtype::budget::run(tentacles, scan).into_iter().map(|x| x.expect("scan lost")).collect() };
    let mut quiet = stage();
    for _ in 0..2 {
        let again = stage();
        if median(&again) < median(&quiet) {
            quiet = again;
        }
    }
    // The same stage while background work (the intake pass's shape: kernels of about 10 ms on
    // every thread the budget lends it) runs; the load starts first and holds every thread.
    let under_load = |inside: bool| -> Vec<(Vec<f32>, f64)> {
        stop.store(false, std::sync::atomic::Ordering::Relaxed);
        std::thread::scope(|sc| {
            let load: Vec<_> = if inside {
                vec![sc.spawn(|| {
                    nmtype::budget::background(|| {
                        nmtype::budget::run(10_000, |_| if stop.load(std::sync::atomic::Ordering::Relaxed) { 0.0 } else { busy(10.0) })
                    });
                })]
            } else {
                (0..total)
                    .map(|_| {
                        sc.spawn(|| {
                            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                                let _ = busy(10.0);
                            }
                        })
                    })
                    .collect()
            };
            std::thread::park_timeout(std::time::Duration::from_millis(50));
            let got = stage();
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            for h in load {
                let _ = h.join();
            }
            got
        })
    };
    let in_budget = under_load(true);
    let outside = under_load(false);
    let pct = |v: &[(Vec<f32>, f64)]| (median(v) / median(&quiet) - 1.0) * 100.0;
    let same = [&quiet, &in_budget, &outside].iter().all(|v| v.first().is_some_and(|x| x.0 == alone_rows));
    println!(
        "PROC-1: {total} threads; one 12,000-token window scan alone {alone:.1} ms; the eight tentacles' scans as one stage: quiet median {:.1} ms; beside background work in the budget median {:.1} ms ({:+.0} percent of quiet); beside the same work outside the budget median {:.1} ms ({:+.0} percent); scores identical across loads: {same}",
        median(&quiet),
        median(&in_budget),
        pct(&in_budget),
        median(&outside),
        pct(&outside),
    );
    assert!(pct(&in_budget) <= 20.0, "PROC-1: scans under background load {:+.0} percent of quiet", pct(&in_budget));
    assert!(same, "scores differ across loads");
}

#[test]
#[ignore = "timing by query tokens, run by hand"]
fn d2_tokens() {
    let table = nmengn::cpu::rope_table(WINDOW, D, 1_000_000.0);
    let (parts, at) = window(7);
    let q = queries(11, 20);
    for nt in [1_usize, 5, 20] {
        let toks: Vec<usize> = (0..nt).collect();
        let t = Instant::now();
        let _ = scan_pieces(&q, &parts, &at, WINDOW, &table, (D, GROUP), Some(&toks), Reduce::Sum).expect("scan");
        println!("{nt} tokens: {:.1} ms", ms(t));
    }
}
