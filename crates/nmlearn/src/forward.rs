//! Forward-only fitting (PLAN 4c `forward`, 4d item 7; guide 10.6): MeZO (Malladi et al.
//! 2023, from `nrntrn/src/mezo.rs` and `neuron.rs`) on the scoped rows of a unit's `up`
//! buffer: perturb by `+eps z` and `-eps z`, read the loss twice, step along `z` by the
//! projected gradient; `z` is regenerated from the seed, never stored. The budget rule: the
//! first pair's wall time times the pair cap is the estimate, and an estimate over the
//! ceiling refuses the fit before it continues; the loop stops at the wall limit or after
//! `patience` pairs without improvement. The loss is a trait object the engine implements.
use crate::unit::{Normals, narrow};
use nmtype::{Fault, FaultKind};
use std::time::Instant;

fn bad(kind: FaultKind, m: impl Into<String>) -> Fault {
    Fault::new("nmlearn.forward", kind, m)
}

/// Add `factor x z` to the rows `rows` of `buffer` (`rows x width`, row-major) only; `z`
/// covers just those rows, in row order, so other rows never change.
/// # Errors
/// A row outside the buffer.
pub fn perturb_rows(buffer: &mut [f32], width: usize, rows: &[usize], seed: u64, step: u64, factor: f64) -> Result<(), Fault> {
    let count = buffer.len() / width.max(1);
    if let Some(r) = rows.iter().find(|r| **r >= count) {
        return Err(bad(FaultKind::Config, format!("row {r} outside the {count}-row buffer")));
    }
    let mut z = Normals::new(seed, step);
    for r in rows {
        for v in buffer.get_mut(r * width..(r + 1) * width).unwrap_or(&mut []) {
            *v = narrow(f64::from(*v) + factor * z.next_normal());
        }
    }
    Ok(())
}

/// The loss over a batch of outcomes for a given `up` buffer (the engine runs the forwards).
pub trait Loss {
    /// Mean loss of `batch` (outcome indices) with `up` in place.
    /// # Errors
    /// An engine fault.
    fn loss(&mut self, up: &[f32], batch: &[usize]) -> Result<f64, Fault>;
}

/// One pair's record.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pair {
    pub plus: f64,
    pub minus: f64,
    pub projected: f64,
    pub ms: f64,
}

/// What a fit did.
#[derive(Clone, Debug, PartialEq)]
pub struct FitLog {
    pub pairs: Vec<Pair>,
    pub wall_ms: u64,
    pub estimate_ms: u64,
    /// Pair cap in force after the first pair (S8 defect 1: also bounded by the wall limit
    /// divided by the first pair's time).
    pub cap: usize,
    pub first_loss: f64,
    pub last_loss: f64,
    pub best_loss: f64,
    pub stopped: String,
}
impl FitLog {
    /// Share of pairs whose two losses were finite (1.0 for a clean fit).
    #[must_use]
    pub fn stability(&self) -> f32 {
        if self.pairs.is_empty() {
            return 1.0;
        }
        #[expect(clippy::cast_precision_loss, reason = "pair counts are small")]
        let s = self.pairs.iter().filter(|p| p.plus.is_finite() && p.minus.is_finite()).count() as f32 / self.pairs.len() as f32;
        s
    }
}

/// A fitter over a unit's `up` buffer.
pub trait Fitter {
    /// Fit `up` (`rows x width`) on `rows` with `batches` of outcome indices, one batch per
    /// pair, cycling.
    /// # Errors
    /// A refused estimate (policy), a non-finite loss, or an engine fault.
    fn fit(&self, up: &mut [f32], width: usize, rows: &[usize], batches: &[Vec<usize>], loss: &mut dyn Loss) -> Result<FitLog, Fault>;
}

/// MeZO under the one-minute rule.
#[derive(Clone, Debug, PartialEq)]
pub struct Mezo {
    pub eps: f64,
    pub lr: f64,
    pub seed: u64,
    /// Pair cap (PLAN 4d item 7: `min(64, 50_000 / (outcomes x tokens))`).
    pub max_pairs: usize,
    /// Wall limit for the loop (50 s).
    pub wall_ms: u64,
    /// Ceiling the estimate is checked against (60 s).
    pub budget_ms: u64,
    pub patience: usize,
}
impl Mezo {
    /// The pair cap for `outcomes` per pair of `tokens` each.
    #[must_use]
    pub fn pair_cap(outcomes: usize, tokens: usize) -> usize {
        (50_000 / (outcomes * tokens).max(1)).clamp(1, 64)
    }
    /// The S8 settings: eps 1e-3, lr 1e-3, 50 s loop, 60 s ceiling, patience 8.
    #[must_use]
    pub fn standard(seed: u64, max_pairs: usize) -> Self {
        Self { eps: 1e-3, lr: 1e-3, seed, max_pairs, wall_ms: 50_000, budget_ms: 60_000, patience: 8 }
    }
}
impl Fitter for Mezo {
    fn fit(&self, up: &mut [f32], width: usize, rows: &[usize], batches: &[Vec<usize>], loss: &mut dyn Loss) -> Result<FitLog, Fault> {
        if batches.is_empty() {
            return Err(bad(FaultKind::Config, "no batches"));
        }
        let start = up.to_vec();
        let t0 = Instant::now();
        let mut pairs = Vec::new();
        let mut best = f64::INFINITY;
        let mut since_best = 0;
        let mut estimate_ms = 0;
        let mut stopped = String::from("pair cap");
        let mut cap = self.max_pairs;
        let mut i = 0;
        while i < cap {
            let batch = &batches[i % batches.len()];
            let step = u64::try_from(i).unwrap_or(0);
            let tp = Instant::now();
            perturb_rows(up, width, rows, self.seed, step, self.eps)?;
            let plus = loss.loss(up, batch)?;
            perturb_rows(up, width, rows, self.seed, step, -2.0 * self.eps)?;
            let minus = loss.loss(up, batch)?;
            perturb_rows(up, width, rows, self.seed, step, self.eps)?;
            if !plus.is_finite() || !minus.is_finite() {
                up.copy_from_slice(&start);
                return Err(bad(
                    FaultKind::Execution,
                    format!("loss not finite at pair {i} ({plus}, {minus}); the fit stops and the unit is restored"),
                ));
            }
            let projected = (plus - minus) / (2.0 * self.eps);
            perturb_rows(up, width, rows, self.seed, step, -self.lr * projected)?;
            let ms = tp.elapsed().as_secs_f64() * 1000.0;
            pairs.push(Pair { plus, minus, projected, ms });
            if i == 0 {
                // The cap is bounded by what the wall limit allows at the first pair's pace.
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_precision_loss,
                    reason = "pair counts and milliseconds are small"
                )]
                let fits = (self.wall_ms as f64 / ms.max(1e-3)).floor() as usize;
                cap = self.max_pairs.min(fits.max(1));
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_precision_loss,
                    reason = "milliseconds fit u64"
                )]
                let est = (ms * cap as f64) as u64;
                estimate_ms = est;
                if est > self.budget_ms {
                    up.copy_from_slice(&start);
                    return Err(bad(
                        FaultKind::Policy,
                        format!("estimated {est} ms for {cap} pairs exceeds the {} ms ceiling; refused before running", self.budget_ms),
                    ));
                }
            }
            let now = plus.min(minus);
            if now < best {
                best = now;
                since_best = 0;
            } else {
                since_best += 1;
            }
            let wall = u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX);
            if since_best >= self.patience {
                stopped = format!("no improvement for {} pairs", self.patience);
                break;
            }
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "milliseconds fit u64")]
            let pair_ms = ms.max(0.0).round() as u64;
            if wall + pair_ms >= self.wall_ms {
                stopped = "wall limit".into();
                break;
            }
            i += 1;
        }
        let first = pairs.first().map_or(0.0, |p| p.plus.min(p.minus));
        let last = pairs.last().map_or(0.0, |p| p.plus.min(p.minus));
        Ok(FitLog {
            pairs,
            wall_ms: u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX),
            estimate_ms,
            cap,
            first_loss: first,
            last_loss: last,
            best_loss: best,
            stopped,
        })
    }
}

/// Self-check of `forward`: only scoped rows move; forward-only steps lower a least-squares
/// loss on the scoped rows; a non-finite loss stops the fit and restores the buffer; an
/// estimate over the ceiling is refused with the buffer restored.
pub(crate) fn check() -> Result<(), String> {
    let width = 4;
    let start: Vec<f32> = (0..8 * width).map(|i| narrow(f64::from(u32::try_from(i).unwrap_or(0)) / 10.0)).collect();
    let mut buf = start.clone();
    perturb_rows(&mut buf, width, &[1, 5], 3, 0, 0.1).map_err(|e| e.to_string())?;
    for row in 0..8 {
        let same = buf[row * width..row * width + width] == start[row * width..row * width + width];
        if same == (row == 1 || row == 5) {
            return Err(format!("nmlearn.forward: row {row}"));
        }
    }
    struct Ls(Vec<f32>);
    impl Loss for Ls {
        fn loss(&mut self, up: &[f32], _: &[usize]) -> Result<f64, Fault> {
            Ok(up.iter().zip(&self.0).map(|(x, t)| (f64::from(*x) - f64::from(*t)).powi(2)).sum::<f64>())
        }
    }
    let target: Vec<f32> = (0..8 * width).map(|i| if i / width == 2 || i / width == 6 { 0.5 } else { 0.0 }).collect();
    let mut w = vec![0.0_f32; 8 * width];
    let mut l = Ls(target);
    let first = l.loss(&w, &[]).map_err(|e| e.to_string())?;
    let m = Mezo { eps: 1e-3, lr: 0.05, seed: 21, max_pairs: 300, wall_ms: 50_000, budget_ms: 600_000, patience: 300 };
    let log = m.fit(&mut w, width, &[2, 6], &[vec![0]], &mut l).map_err(|e| e.to_string())?;
    let last = l.loss(&w, &[]).map_err(|e| e.to_string())?;
    if last >= first * 0.3 || log.stability() < 1.0 || w[0..width].iter().any(|v| *v != 0.0) {
        return Err(format!("nmlearn.forward: loss {first} to {last}"));
    }
    struct Nan;
    impl Loss for Nan {
        fn loss(&mut self, _: &[f32], _: &[usize]) -> Result<f64, Fault> {
            Ok(f64::NAN)
        }
    }
    let mut w2 = vec![0.25_f32; 8];
    if m.fit(&mut w2, 4, &[0], &[vec![0]], &mut Nan).is_ok() || w2 != vec![0.25_f32; 8] {
        return Err("nmlearn.forward: a non-finite loss did not stop and restore".into());
    }
    struct Slow;
    impl Loss for Slow {
        fn loss(&mut self, _: &[f32], _: &[usize]) -> Result<f64, Fault> {
            let t = Instant::now();
            while t.elapsed().as_millis() < 3 {}
            Ok(1.0)
        }
    }
    let tight = Mezo { max_pairs: 64, wall_ms: 60_000, budget_ms: 100, ..m };
    let mut w3 = vec![0.0_f32; 8];
    match tight.fit(&mut w3, 4, &[0], &[vec![0]], &mut Slow) {
        Err(f) if f.kind == FaultKind::Policy && w3 == vec![0.0_f32; 8] => {}
        other => return Err(format!("nmlearn.forward: estimate over the ceiling was not refused: {other:?}")),
    }
    if Mezo::pair_cap(4, 60) != 64 || Mezo::pair_cap(100, 100) != 5 {
        return Err("nmlearn.forward: pair cap".into());
    }
    Ok(())
}
