//! Pattern miner: recurring contiguous chunks across the log's sequences, scored by
//! compression gain, with start and stop boundaries.
//!
//! Method (idea ported from Stitch, Bowers et al., POPL 2023, arXiv 2211.16605, paper CC BY
//! 4.0; no code read or copied): the best abstraction is the one whose use most shortens the
//! whole corpus. Here the corpus is flat symbol sequences, so an abstraction is a contiguous
//! chunk. Rewriting each non-overlapping occurrence of a chunk of length `n` as one new
//! symbol saves `n - 1` symbols per occurrence and costs `n + 1` to define it (its `n`
//! symbols plus the new symbol), so
//!
//! `gain = occurrences * (n - 1) - (n + 1)`.
//!
//! Search is bottom-up by length (an `n`-chunk is counted only where its `n - 1` prefix
//! already repeated), the sequence form of Stitch's pruning of non-repeating partial
//! abstractions.
//!
//! "Repeats a lot", derived from the log's own statistics: under independent draws from the
//! log's own symbol frequencies, a chunk is expected `E = positions(n) * prod p(s_i)` times,
//! and its count is close to Poisson with mean `E`. With `M` distinct chunks examined in the
//! pass, a chunk counts as recurring when the expected number of chunks reaching its count by
//! chance is below one, `M * P(Poisson(E) >= occurrences) < 1` (a Bonferroni bound over the
//! pass), and `gain > 0`. No fixed count is used; the floor of 2 is the definition of a
//! repeat. The threshold reported per candidate is the smallest count meeting the rule.
//! (2026-11-10: replaces a first rule `E + sqrt(2 ln M) * sqrt(E)`, which floored at 2 for
//! every chunk of length 4 or more because the normal tail understates the Poisson tail at
//! small `E`; measured as thousands of chance chunks on a Markov background.)
use crate::log::Trace;
use std::collections::{HashMap, HashSet};

/// One occurrence: trace index and start position; stop is `start + len` (exclusive).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Occurrence {
    pub trace: usize,
    pub start: usize,
}

/// One ranked candidate chunk.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub symbols: Vec<u32>,
    /// Non-overlapping occurrences, with boundaries.
    pub occurrences: Vec<Occurrence>,
    /// Compression gain in symbols.
    pub gain: i64,
    /// Expected occurrences by chance.
    pub expected: f64,
    /// Derived recurrence threshold this chunk passed.
    pub threshold: f64,
}
impl Candidate {
    #[must_use]
    pub fn count(&self) -> usize {
        self.occurrences.len()
    }
}

/// Statistics of one mining pass (recorded with every result).
#[derive(Clone, Debug, PartialEq)]
pub struct MineStats {
    pub traces: usize,
    pub symbols: usize,
    pub alphabet: usize,
    /// Distinct chunks examined (`M`).
    pub examined: usize,
    /// Longest chunk length that still repeated.
    pub max_len: usize,
}

/// Result of a mining pass.
#[derive(Clone, Debug, PartialEq)]
pub struct Mined {
    /// Recurring candidates, best compression gain first, nested repeats folded.
    pub candidates: Vec<Candidate>,
    pub stats: MineStats,
}

struct Entry {
    occ: Vec<Occurrence>,
    last_end: Option<(usize, usize)>,
}

fn count_level(traces: &[Trace], n: usize, keep: Option<&HashSet<Vec<u32>>>) -> HashMap<Vec<u32>, Entry> {
    let mut map: HashMap<Vec<u32>, Entry> = HashMap::new();
    for (ti, t) in traces.iter().enumerate() {
        if t.symbols.len() < n {
            continue;
        }
        for start in 0..=t.symbols.len() - n {
            let Some(w) = t.symbols.get(start..start + n) else { continue };
            if let (Some(k), Some(prefix)) = (keep, w.get(..n - 1))
                && !k.contains(prefix)
            {
                continue;
            }
            let e = map.entry(w.to_vec()).or_insert(Entry { occ: Vec::new(), last_end: None });
            let overlaps = matches!(e.last_end, Some((lt, le)) if lt == ti && start < le);
            if !overlaps {
                e.occ.push(Occurrence { trace: ti, start });
                e.last_end = Some((ti, start + n));
            }
        }
    }
    map
}

/// `P(Poisson(mean) >= k)`, summed in log space from term `k` upward.
fn poisson_tail(mean: f64, k: usize) -> f64 {
    if mean <= 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    let mut ln_term = -mean + k as f64 * mean.ln() - (1..=k).map(|i| (i as f64).ln()).sum::<f64>();
    let mut sum = 0.0;
    let mut i = k;
    loop {
        let t = ln_term.exp();
        sum += t;
        i += 1;
        ln_term += mean.ln() - (i as f64).ln();
        if (t < sum * 1e-12 && i as f64 > mean) || i > k + 10_000 {
            break;
        }
    }
    sum.min(1.0)
}

/// Smallest count `c >= 2` with `m * P(Poisson(mean) >= c) < 1`.
fn poisson_threshold(mean: f64, m: f64) -> usize {
    let mut c = 2usize;
    while m * poisson_tail(mean, c) >= 1.0 && c < 1_000_000 {
        c += 1;
    }
    c
}

fn contains(hay: &[u32], needle: &[u32]) -> bool {
    needle.len() <= hay.len() && hay.windows(needle.len()).any(|w| w == needle)
}

/// Mine the traces. The alphabet size is taken from the symbols present.
/// The chance model a chunk's expected count is taken from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chance {
    /// Independent draws from the log's own symbol frequencies (the first form).
    Independent,
    /// First-order Markov from the log's own transitions (2026-11-13, research file Solution 3,
    /// MO6, analysis E 12.1 to 12.4: control traffic repeats routes, so a chunk that only follows
    /// common transitions is expected often and is not a pattern of its own).
    Markov,
}

/// Mine with the design's chance model, first-order Markov.
#[must_use]
pub fn mine(traces: &[Trace]) -> Mined {
    mine_with(traces, Chance::Markov)
}

/// Mine with the chance model `chance`.
#[must_use]
pub fn mine_with(traces: &[Trace], chance: Chance) -> Mined {
    let mut freq: HashMap<u32, usize> = HashMap::new();
    // Transitions a to b, and how often a is followed by anything.
    let mut trans: HashMap<(u32, u32), usize> = HashMap::new();
    let mut from: HashMap<u32, usize> = HashMap::new();
    for t in traces {
        for w in t.symbols.windows(2) {
            if let [a, b] = w {
                *trans.entry((*a, *b)).or_insert(0) += 1;
                *from.entry(*a).or_insert(0) += 1;
            }
        }
    }
    let mut total = 0usize;
    for t in traces {
        for &s in &t.symbols {
            *freq.entry(s).or_insert(0) += 1;
            total += 1;
        }
    }
    let mut repeated: Vec<(Vec<u32>, Vec<Occurrence>)> = Vec::new();
    let mut examined = 0usize;
    let mut keep: Option<HashSet<Vec<u32>>> = None;
    let mut n = 2usize;
    let mut max_len = 0usize;
    loop {
        let level = count_level(traces, n, keep.as_ref());
        examined += level.len();
        let mut next = HashSet::new();
        for (k, e) in level {
            if e.occ.len() >= 2 {
                next.insert(k.clone());
                repeated.push((k, e.occ));
            }
        }
        if next.is_empty() {
            break;
        }
        max_len = n;
        keep = Some(next);
        n += 1;
    }
    let m_tests = examined.max(1) as f64;
    let tot = total.max(1) as f64;
    let mut cands: Vec<Candidate> = repeated
        .into_iter()
        .filter_map(|(symbols, occurrences)| {
            let len = symbols.len();
            let positions: usize = traces.iter().map(|t| t.symbols.len().saturating_sub(len - 1)).sum();
            let first = symbols.first().map_or(0.0, |s| freq.get(s).copied().unwrap_or(0) as f64 / tot);
            let p: f64 = match chance {
                Chance::Independent => symbols.iter().map(|s| freq.get(s).copied().unwrap_or(0) as f64 / tot).product(),
                Chance::Markov => {
                    first
                        * symbols
                            .windows(2)
                            .map(|w| match w {
                                [a, b] => {
                                    trans.get(&(*a, *b)).copied().unwrap_or(0) as f64 / from.get(a).copied().unwrap_or(0).max(1) as f64
                                }
                                _ => 1.0,
                            })
                            .product::<f64>()
                }
            };
            let expected = positions as f64 * p;
            let passes = occurrences.len() >= 2 && m_tests * poisson_tail(expected, occurrences.len()) < 1.0;
            let occ = occurrences.len() as i64;
            let l = len as i64;
            let gain = occ * (l - 1) - (l + 1);
            (passes && gain > 0).then(|| {
                let threshold = poisson_threshold(expected, m_tests) as f64;
                Candidate { symbols, occurrences, gain, expected, threshold }
            })
        })
        .collect();
    cands.sort_by(|a, b| b.gain.cmp(&a.gain).then_with(|| a.symbols.cmp(&b.symbols)));
    // Fold nested repeats: a chunk inside a better one that occurs no more often than the
    // better one plus its sampling spread (sqrt of its count) adds nothing of its own.
    let mut kept: Vec<Candidate> = Vec::new();
    for c in cands {
        let nested =
            kept.iter().any(|k| contains(&k.symbols, &c.symbols) && (c.count() as f64) <= k.count() as f64 + (k.count() as f64).sqrt());
        if !nested {
            kept.push(c);
        }
    }
    Mined { candidates: kept, stats: MineStats { traces: traces.len(), symbols: total, alphabet: freq.len(), examined, max_len } }
}

/// Section check: two chunks planted in uniform random sequences are found and ranked first
/// and second, with correct boundaries.
/// # Errors
/// One fault line.
pub fn check() -> std::result::Result<(), String> {
    let a = [3u32, 9, 14, 1, 22, 7];
    let b = [30u32, 2, 18, 11, 5];
    let traces = crate::synth::uniform_with_plants(200, 50, 40, &[(&a, 30), (&b, 25)], 7);
    let m = mine(&traces);
    let top: Vec<&[u32]> = m.candidates.iter().take(2).map(|c| c.symbols.as_slice()).collect();
    if top != [&a[..], &b[..]] {
        return Err(format!("planted chunks not ranked first: top {top:?}"));
    }
    for c in m.candidates.iter().take(2) {
        for o in &c.occurrences {
            let w = traces.get(o.trace).and_then(|t| t.symbols.get(o.start..o.start + c.symbols.len()));
            if w != Some(c.symbols.as_slice()) {
                return Err(format!("boundary wrong at trace {} start {}", o.trace, o.start));
            }
        }
    }
    if m.candidates.first().map(Candidate::count) != Some(30) {
        return Err("planted chunk count wrong".to_owned());
    }
    // On a first-order Markov background (preferred successors), the independent model counts
    // common routes as patterns; the Markov model keeps far fewer and still finds the plant.
    let (mk, _) = crate::synth::generate(200, 50, 40, crate::synth::Background::Markov { succ: 3, p: 0.6 }, &[(&a, 30)], 11);
    let ind = mine_with(&mk, Chance::Independent);
    let mar = mine_with(&mk, Chance::Markov);
    if !mar.candidates.iter().any(|c| c.symbols == a) || mar.candidates.len() * 2 > ind.candidates.len() {
        return Err(format!(
            "Markov chance model: {} candidates against {} independent, plant found {}",
            mar.candidates.len(),
            ind.candidates.len(),
            mar.candidates.iter().any(|c| c.symbols == a)
        ));
    }
    Ok(())
}
