//! First measurements of the first form (tests M1 and M2 of the G3 design, first form), on
//! synthetic behaviour logs, processor only.
//!
//! - M1: the repeat count at which a planted chunk becomes recurring under the derived miner
//!   rule, and at which it also passes the gain gate, for logs of realistic size.
//! - M2: bytes per pack by chunk length.
//! - Mining time for 1,000 sequences of 50 symbols.
//!
//! Synthetic log: alphabet 40 behavioural message types (stated assumption: eight tentacles,
//! GCX levels and a few actions each); background either uniform or first-order Markov with 3
//! preferred successors taken with probability 0.6 (control traffic that repeats routes);
//! planted chunk of length 4; outcomes 0.5 plus noise of standard deviation 0.1, with a lift
//! of 0.1 for questions that used the planted chunk.
use crate::gain::{GainEstimator, SplitMeans};
use crate::pack::Pack;
use crate::symbol::{Codebook, Message, MessageKind};
use crate::synth::{Background, Rng, generate};
use std::fmt::Write;
use std::time::Instant;

const ALPHABET: usize = 40;
const LEN: usize = 50;
const PLANT: [u32; 4] = [7, 19, 3, 33];
const KS: [usize; 17] = [2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 160, 200];

/// M1 result for one log shape.
#[derive(Clone, Debug)]
pub struct M1Row {
    pub traces: usize,
    pub background: &'static str,
    /// Smallest repeat count at which the plant was recurring, per seed.
    pub recurring_k: Vec<Option<usize>>,
    /// Smallest repeat count at which the plant was recurring and promotable, per seed.
    pub promotable_k: Vec<Option<usize>>,
    /// Rank of the plant at its first recurring count, per seed (1 = first).
    pub rank_at_recurring: Vec<Option<usize>>,
    /// Smallest repeat count at which the plant ranked first by compression gain, per seed.
    pub top_k: Vec<Option<usize>>,
    /// Derived threshold for the plant at the smallest recurring count (first seed).
    pub threshold: Option<f64>,
}

#[derive(Default)]
struct One {
    rec: Option<usize>,
    pro: Option<usize>,
    rank: Option<usize>,
    top: Option<usize>,
    thr: Option<f64>,
}

fn bg_name(bg: Background) -> &'static str {
    match bg {
        Background::Uniform => "uniform",
        Background::Markov { .. } => "markov(3 successors, p 0.6)",
    }
}

fn m1_one(traces: usize, bg: Background, seed: u64, z: f64) -> One {
    let mut o = One::default();
    for &k in &KS {
        if k > traces {
            break;
        }
        let (mut log, users) = generate(traces, LEN, ALPHABET, bg, &[(&PLANT, k)], seed);
        let users = users.first().cloned().unwrap_or_default();
        let mut rng = Rng::new(seed ^ 0xA5A5);
        for (i, t) in log.iter_mut().enumerate() {
            let lift = if users.contains(&i) { 0.1 } else { 0.0 };
            t.outcome = Some(0.5 + lift + 0.1 * rng.normal());
        }
        let m = crate::miner::mine(&log);
        let pos = m.candidates.iter().position(|c| c.symbols == PLANT);
        if let Some(p) = pos {
            if o.rec.is_none() {
                o.rec = Some(k);
                o.rank = Some(p + 1);
                o.thr = m.candidates.get(p).map(|c| c.threshold);
            }
            if o.pro.is_none() && SplitMeans.estimate(&log, &PLANT).is_some_and(|g| g.promotable(z)) {
                o.pro = Some(k);
            }
            if o.top.is_none() && p == 0 {
                o.top = Some(k);
            }
        }
        if o.rec.is_some() && o.pro.is_some() && o.top.is_some() {
            break;
        }
    }
    o
}

/// Run M1 for the given log sizes over `seeds` seeds.
#[must_use]
pub fn m1(sizes: &[usize], seeds: u64) -> Vec<M1Row> {
    let z = crate::Policy::load().gate_z;
    let mut out = Vec::new();
    for &n in sizes {
        for bg in [Background::Uniform, Background::Markov { succ: 3, p: 0.6 }] {
            let mut row = M1Row {
                traces: n,
                background: bg_name(bg),
                recurring_k: Vec::new(),
                promotable_k: Vec::new(),
                rank_at_recurring: Vec::new(),
                top_k: Vec::new(),
                threshold: None,
            };
            for s in 0..seeds {
                let o = m1_one(n, bg, 1000 + s, z);
                row.recurring_k.push(o.rec);
                row.promotable_k.push(o.pro);
                row.rank_at_recurring.push(o.rank);
                row.top_k.push(o.top);
                if s == 0 {
                    row.threshold = o.thr;
                }
            }
            out.push(row);
        }
    }
    out
}

/// M2: (chunk length, file bytes, payload bytes) for packs with distinct realistic
/// descriptors.
#[must_use]
pub fn m2(lengths: &[usize]) -> Vec<(usize, usize, usize)> {
    let kinds = [MessageKind::Instruction, MessageKind::Route, MessageKind::Reflex];
    let mut out = Vec::new();
    for &len in lengths {
        let mut cb = Codebook::default();
        let mut chunk = Vec::new();
        for i in 0..len {
            let m = Message::behavioural(
                kinds.get(i % 3).copied().unwrap_or(MessageKind::Instruction),
                &format!("g0_t{}", 1 + i % 8),
                &format!("focus:area{}", i % 5),
            );
            if let Ok(c) = cb.code(&m) {
                chunk.push(c);
            }
        }
        let g = crate::gain::GainRecord { with_n: 12, with_mean: 0.6, without_n: 188, without_mean: 0.5, diff: 0.1, se: 0.03 };
        if let Ok(p) = Pack::build(&chunk, &cb, "", "0".repeat(64).as_str(), g)
            && let Ok(b) = p.encode()
        {
            out.push((len, b.len(), 5 + len * usize::from(p.header.width)));
        }
    }
    out
}

/// Median mining time in milliseconds for `traces` sequences of 50 symbols, over `runs` runs.
#[must_use]
pub(crate) fn mine_time_ms(traces: usize, bg: Background, runs: usize) -> (f64, usize) {
    let (log, _) = generate(traces, LEN, ALPHABET, bg, &[(&PLANT, traces / 20)], 42);
    let mut ts = Vec::new();
    let mut found = 0;
    for _ in 0..runs.max(1) {
        let t = Instant::now();
        let m = crate::miner::mine(&log);
        ts.push(t.elapsed().as_secs_f64() * 1000.0);
        found = m.candidates.len();
    }
    ts.sort_by(f64::total_cmp);
    (ts.get(ts.len() / 2).copied().unwrap_or(0.0), found)
}

/// All first measurements as a report text.
#[must_use]
pub fn report() -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "M1 (plant length 4, alphabet {ALPHABET}, {LEN} symbols per question, lift 0.1, noise sd 0.1, 5 seeds; counts tried {KS:?})"
    );
    for r in m1(&[200, 1000], 5) {
        let _ = writeln!(
            s,
            "  {} questions, {}: recurring at k = {:?}; promotable at k = {:?}; rank when first recurring {:?}; ranked first at k = {:?}; derived threshold (seed 0) {:?}",
            r.traces,
            r.background,
            r.recurring_k,
            r.promotable_k,
            r.rank_at_recurring,
            r.top_k,
            r.threshold.map(|t| (t * 100.0).round() / 100.0)
        );
    }
    let _ = writeln!(s, "M2 (chunk length, pack file bytes, payload bytes)");
    for (l, b, p) in m2(&[4, 8, 16, 32]) {
        let _ = writeln!(s, "  {l}: {b} bytes, payload {p}");
    }
    for bg in [Background::Uniform, Background::Markov { succ: 3, p: 0.6 }] {
        let (ms, n) = mine_time_ms(1000, bg, 5);
        let _ = writeln!(s, "Mining 1,000 x 50, {}: median {ms:.1} ms over 5 runs, {n} recurring candidates", bg_name(bg));
    }
    s
}

/// The drift test (2026-11-13, G3 method; research file Part 12, smallest test): a promoted pack
/// whose gains hold (normal around the promotion gain) for `d` uses and then fall to zero. For
/// each rule: the share of seeds withdrawn before the drift (false withdrawals) and the median
/// uses after the drift until withdrawal (detection delay; 400 when never within 400).
#[must_use]
pub fn drift_report(seeds: u64) -> String {
    use crate::gain::GainRecord;
    use crate::withdraw::{Recent, RunningGain, should_withdraw, should_withdraw_change, should_withdraw_recent};
    let promo = GainRecord { with_n: 20, with_mean: 0.65, without_n: 180, without_mean: 0.5, diff: 0.15, se: 0.03 };
    let z = crate::Policy::default().withdraw_z;
    let sd = 0.1;
    let mut out = String::from(
        "G3 drift test: gain sd 0.1, promotion gain 0.15 (se 0.03), after the drift the gain is 0\nrule | uses before drift | false withdrawals | median delay after drift\n",
    );
    let rules: [(&str, f64); 5] = [("whole history", 0.0), ("recent, z 1.96", 1.96), ("recent, z 3", 3.0), ("recent, z 4", 4.0), ("change (today)", -1.0)];
    for d in [10_usize, 50, 200] {
        for (name, zr) in rules {
            let (mut false_w, mut delays) = (0_u64, Vec::new());
            for s in 0..seeds {
                let mut rng = Rng::new(s * 7919 + d as u64);
                let (mut run, mut rec) = (RunningGain::from_promotion(&promo), Recent::default());
                let mut at = None;
                for i in 0..d + 400 {
                    let g = if i < d { promo.diff } else { 0.0 } + sd * rng.normal();
                    run.push(g);
                    rec.push_against(&promo, g);
                    let w = if zr < 0.0 {
                        should_withdraw_change(&promo, &run, &rec, z)
                    } else if zr == 0.0 {
                        should_withdraw(&promo, &run, z)
                    } else {
                        should_withdraw_recent(&promo, &run, &rec, z, zr)
                    };
                    if w {
                        at = Some(i);
                        break;
                    }
                }
                match at {
                    Some(i) if i < d => false_w += 1,
                    Some(i) => delays.push(i - d + 1),
                    None => delays.push(400),
                }
            }
            delays.sort_unstable();
            let med = delays.get(delays.len() / 2).copied().unwrap_or(400);
            #[expect(clippy::cast_precision_loss, reason = "seed counts")]
            let fw = false_w as f64 / seeds as f64;
            let _ = writeln!(out, "{name} | {d} | {fw:.3} | {med}");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    /// First measurements; run with `cargo test --release -p nmmusc -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, run on demand"]
    fn first_measurements() {
        println!("{}", super::report());
    }

    #[test]
    #[ignore = "measurement, run on demand"]
    fn drift_rules() {
        println!("{}", super::drift_report(200));
    }
}
