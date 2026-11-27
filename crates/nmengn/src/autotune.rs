//! Per-chip CUDA kernel auto-tuner cached in `<home>/memory/machine.json`.
//!
//! All four compiled `dp4a` tile kernels in `fast.fatbin` produce bit-identical results
//! (`gpu::check_dp4a_tile`), but different GPU architectures (`sm_75` Turing, `sm_80`/`sm_86`
//! Ampere, `sm_89` Ada, `sm_90` Hopper, Blackwell) peak on different tile shapes at small
//! (`<= 64` tokens) vs large (`> 64` tokens) prefill sizes. `tune_on_device` benchmarks all four
//! once per GPU (under 100 ms total) and persists the winning indices in `machine.json` so every
//! future run selects the fastest bit-identical kernel automatically with zero manual flags.
use nmffi::{Gpu, Param};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::Instant;

/// Bit-identical `dp4a` kernel variants in `fast.fatbin`: `(symbol, tile_rows, tile_tokens)`.
pub const DP4A_VARIANTS: [(&str, usize, usize); 4] = [
    ("_ZN4emma16dp4a_tile_kernelE12emma_op_args", 64, 128),
    ("_ZN4emma14dp4a_tt_kernelILi8ELi4ELi2EEEv12emma_op_args", 64, 128),
    ("_ZN4emma14dp4a_tt_kernelILi8ELi8ELi1EEEv12emma_op_args", 128, 128),
    ("_ZN4emma14dp4a_tt_kernelILi4ELi8ELi2EEEv12emma_op_args", 128, 64),
];

/// Default small-token (`<= 64`) winner (`dp4a_tt<4,8,2>`, measured on Turing `sm_75`).
pub const DEFAULT_SMALL: u8 = 3;
/// Default large-token (`> 64`) winner (`dp4a_tile`, measured on Turing `sm_75`).
pub const DEFAULT_LARGE: u8 = 0;

static SMALL: AtomicU8 = AtomicU8::new(DEFAULT_SMALL);
static LARGE: AtomicU8 = AtomicU8::new(DEFAULT_LARGE);
static SEEDED: AtomicBool = AtomicBool::new(false);

/// Selected bit-identical `dp4a` kernel variants for this GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub small: u8,
    pub large: u8,
}

/// The active `dp4a` kernel plan.
#[must_use]
pub fn current() -> Plan {
    Plan { small: SMALL.load(Ordering::Relaxed), large: LARGE.load(Ordering::Relaxed) }
}

/// Active bit-identical `dp4a` kernel `(symbol, tile_rows, tile_tokens)` for `small` (`tokens <= 64`)
/// vs large (`tokens > 64`).
#[must_use]
pub fn dp4a_kernel(small: bool) -> (&'static str, usize, usize) {
    let idx = usize::from(if small { SMALL.load(Ordering::Relaxed) } else { LARGE.load(Ordering::Relaxed) });
    DP4A_VARIANTS.get(idx).copied().unwrap_or(DP4A_VARIANTS[0])
}

/// Apply saved per-machine kernel picks from `<home>/memory/machine.json` when present.
pub fn seed_from_machine(home: &Path, machine_key: &str) -> bool {
    if std::env::var("NANO_AUTOTUNE").is_ok_and(|v| v == "off") {
        return false;
    }
    let s = crate::capacity::recall(home, machine_key, "dp4a_small").and_then(|x| u8::try_from(x).ok()).filter(|&i| usize::from(i) < DP4A_VARIANTS.len());
    let l = crate::capacity::recall(home, machine_key, "dp4a_large").and_then(|x| u8::try_from(x).ok()).filter(|&i| usize::from(i) < DP4A_VARIANTS.len());
    if let (Some(s), Some(l)) = (s, l) {
        SMALL.store(s, Ordering::Relaxed);
        LARGE.store(l, Ordering::Relaxed);
        SEEDED.store(true, Ordering::Relaxed);
        return true;
    }
    false
}

/// Benchmark all four bit-identical `dp4a` kernels on `gpu` at `tokens = 64` (small) and
/// `tokens = 256` (large), install the fastest pair, and save them in `<home>/memory/machine.json`.
/// Skips immediately when already seeded for this machine or when `NANO_AUTOTUNE=off`.
/// # Errors
/// CUDA allocation or kernel launch failure.
pub fn tune_on_device(home: &Path, gpu: &Gpu) -> Result<Plan, String> {
    let info = gpu.info();
    let key = crate::capacity::machine_key(Some((info.name.as_str(), info.total_bytes >> 20)));
    if std::env::var("NANO_AUTOTUNE").is_ok_and(|v| v == "off") {
        return Ok(current());
    }
    if SEEDED.load(Ordering::Relaxed) || seed_from_machine(home, &key) {
        return Ok(current());
    }
    let o = crate::gpu::Ops { gpu: gpu.clone() };
    let rows = 2048_usize;
    let k = 1024_usize;
    let wt = o.alloc(vec![rows, k], crate::gpu::I8)?;
    let ws = o.alloc(vec![rows], crate::gpu::F16)?;
    let best_for = |tokens: usize| -> Result<u8, String> {
        let xq = o.alloc(vec![tokens, k], crate::gpu::I8)?;
        let xs = o.alloc(vec![tokens, k / 32], crate::gpu::F16)?;
        let y = o.alloc(vec![tokens, rows], crate::gpu::F16)?;
        let a = crate::gpu::Ops::args(&[&xq, &xs], &[&wt, &ws], &[&y]);
        let mut best_i = 0_u8;
        let mut best_us = u128::MAX;
        for (idx, &(name, br, bt)) in DP4A_VARIANTS.iter().enumerate() {
            let grid = (u32::try_from(rows.div_ceil(br)).unwrap_or(1), u32::try_from(tokens.div_ceil(bt)).unwrap_or(1), 1);
            gpu.launch(name, grid, (256, 1, 1), 0, &[Param::Op(Box::new(a))])?;
            gpu.sync()?;
            // 2026-11-25: the best of three rounds of five launches (the pick is kept for good,
            // so one noisy round must not decide it).
            let mut us = u128::MAX;
            for _ in 0..3 {
                let t0 = Instant::now();
                for _ in 0..5 {
                    gpu.launch(name, grid, (256, 1, 1), 0, &[Param::Op(Box::new(a))])?;
                }
                gpu.sync()?;
                us = us.min(t0.elapsed().as_micros());
            }
            if us < best_us {
                best_us = us;
                best_i = u8::try_from(idx).unwrap_or(0);
            }
        }
        Ok(best_i)
    };
    let small = best_for(64)?;
    let large = best_for(256)?;
    SMALL.store(small, Ordering::Relaxed);
    LARGE.store(large, Ordering::Relaxed);
    SEEDED.store(true, Ordering::Relaxed);
    crate::capacity::remember(home, &key, "dp4a_small", u64::from(small));
    crate::capacity::remember(home, &key, "dp4a_large", u64::from(large));
    let name = |i: u8| DP4A_VARIANTS.get(usize::from(i)).map_or("?", |v| v.0);
    eprintln!("autotune: {} measured once: up to 64 tokens {}, longer {} (kept in machine.json)", info.name, name(small), name(large));
    Ok(Plan { small, large })
}

/// Section check (no GPU required): default plan resolves valid symbols and round-trips through
/// `machine.json` in a temporary directory.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let p = current();
    if usize::from(p.small) >= DP4A_VARIANTS.len() || usize::from(p.large) >= DP4A_VARIANTS.len() {
        return Err(format!("nmengn.autotune: invalid plan {p:?}"));
    }
    let (s_name, s_br, s_bt) = dp4a_kernel(true);
    let (l_name, l_br, l_bt) = dp4a_kernel(false);
    if s_name.is_empty() || l_name.is_empty() || s_br == 0 || s_bt == 0 || l_br == 0 || l_bt == 0 {
        return Err("nmengn.autotune: empty kernel descriptor".to_owned());
    }
    let dir = std::env::temp_dir().join(format!("nm-autotune-check-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let prev = current();
    let prev_seeded = SEEDED.load(Ordering::Relaxed);
    let key = "TestCard|8192";
    crate::capacity::remember(&dir, key, "dp4a_small", 2);
    crate::capacity::remember(&dir, key, "dp4a_large", 1);
    let ok = seed_from_machine(&dir, key);
    let got = current();
    SMALL.store(prev.small, Ordering::Relaxed);
    LARGE.store(prev.large, Ordering::Relaxed);
    SEEDED.store(prev_seeded, Ordering::Relaxed);
    let _ = std::fs::remove_dir_all(&dir);
    if !ok || got != (Plan { small: 2, large: 1 }) {
        return Err(format!("nmengn.autotune: machine.json round-trip failed: ok={ok} got={got:?}"));
    }
    Ok(())
}
