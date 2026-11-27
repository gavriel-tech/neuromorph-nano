//! Synthetic behaviour logs for checks and measurements (deterministic, seeded).
use crate::log::Trace;

/// xorshift64* generator; deterministic per seed.
pub(crate) struct Rng(u64);
impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub(crate) fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next() % (n.max(1) as u64)) as usize
    }
    pub(crate) fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Standard normal (Box-Muller).
    pub(crate) fn normal(&mut self) -> f64 {
        let u = self.unit().max(f64::MIN_POSITIVE);
        let v = self.unit();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// Background generator: uniform, or first-order Markov with a few preferred successors per
/// symbol (closer to real control traffic, which repeats routes).
#[derive(Clone, Copy)]
pub(crate) enum Background {
    Uniform,
    /// Each symbol has `succ` preferred successors taken with probability `p`.
    Markov {
        succ: usize,
        p: f64,
    },
}

fn background(rng: &mut Rng, len: usize, alphabet: usize, bg: Background, table: &[Vec<u32>]) -> Vec<u32> {
    let mut out = Vec::with_capacity(len);
    let mut prev = rng.below(alphabet) as u32;
    for _ in 0..len {
        let s = match bg {
            Background::Uniform => rng.below(alphabet) as u32,
            Background::Markov { p, .. } => {
                let row = table.get(prev as usize);
                match row {
                    Some(r) if !r.is_empty() && rng.unit() < p => r.get(rng.below(r.len())).copied().unwrap_or(0),
                    _ => rng.below(alphabet) as u32,
                }
            }
        };
        out.push(s);
        prev = s;
    }
    out
}

/// Traces with background symbols and each plant inserted into `count` distinct traces at a
/// random position (overwriting background). Returns traces with no outcome set.
pub(crate) fn generate(
    traces: usize,
    len: usize,
    alphabet: usize,
    bg: Background,
    plants: &[(&[u32], usize)],
    seed: u64,
) -> (Vec<Trace>, Vec<Vec<usize>>) {
    let mut rng = Rng::new(seed);
    let table: Vec<Vec<u32>> = match bg {
        Background::Uniform => Vec::new(),
        Background::Markov { succ, .. } => (0..alphabet).map(|_| (0..succ).map(|_| rng.below(alphabet) as u32).collect()).collect(),
    };
    let mut out: Vec<Trace> = (0..traces)
        .map(|i| Trace { id: format!("q{i}"), symbols: background(&mut rng, len, alphabet, bg, &table), outcome: None })
        .collect();
    let mut used_by = Vec::new();
    let mut slot = 0usize;
    for (chunk, count) in plants {
        let mut users = Vec::new();
        for _ in 0..*count {
            let ti = slot % traces.max(1);
            slot += 1;
            if let Some(t) = out.get_mut(ti) {
                let room = t.symbols.len().saturating_sub(chunk.len());
                let at = rng.below(room + 1);
                if let Some(w) = t.symbols.get_mut(at..at + chunk.len()) {
                    w.copy_from_slice(chunk);
                }
            }
            users.push(ti);
        }
        used_by.push(users);
    }
    (out, used_by)
}

pub(crate) fn uniform_with_plants(traces: usize, len: usize, alphabet: usize, plants: &[(&[u32], usize)], seed: u64) -> Vec<Trace> {
    generate(traces, len, alphabet, Background::Uniform, plants, seed).0
}
