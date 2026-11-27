//! L4 read-out calibrator (2026-11-12, build step 3 of research file Solution 6: "the first
//! piece a calibrator fitted by Newton's method on the processor"; "Gate: leave-one-question-out,
//! shuffled-label control, paired sign test"). A logistic map from a few features of a Core Brain
//! decision to the probability that its first place is right. Fitted in closed form steps
//! (Newton's method with a small ridge) in microseconds; held in RAM and refitted from the
//! ledger's verdicts, so nothing derived from code reads is written to disk (the disk rule for
//! L4 awaits the word of Luis Jake Gabriel III).
//!
//! Features of a decision (from the reranker's P(yes) over the candidates sent up): the log-odds
//! of the best, the log-odds gap to the second, and the share of candidates above one half.
use crate::readout::sign_p;

/// Features of a decision from the reranker's P(yes) per candidate.
#[must_use]
pub fn features(rerank: &[f32]) -> Vec<f64> {
    let logit = |p: f32| {
        let p = f64::from(p).clamp(1e-6, 1.0 - 1e-6);
        (p / (1.0 - p)).ln()
    };
    let mut s: Vec<f64> = rerank.iter().map(|p| logit(*p)).collect();
    s.sort_by(|a, b| b.total_cmp(a));
    let top = s.first().copied().unwrap_or(0.0);
    let gap = top - s.get(1).copied().unwrap_or(top - 10.0);
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let above = rerank.iter().filter(|p| **p > 0.5).count() as f64 / rerank.len().max(1) as f64;
    vec![top, gap, above]
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// A fitted calibrator: weights over the features plus a bias (last), with the feature means
/// and spreads it standardises by.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibrator {
    pub w: Vec<f64>,
    pub mean: Vec<f64>,
    pub sd: Vec<f64>,
}

impl Calibrator {
    /// Fit by Newton's method (iteratively reweighted least squares) with ridge `lambda`.
    #[must_use]
    pub fn fit(xs: &[Vec<f64>], ys: &[bool], lambda: f64) -> Self {
        let d = xs.first().map_or(0, Vec::len);
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let n = xs.len().max(1) as f64;
        let mean: Vec<f64> = (0..d).map(|j| xs.iter().map(|x| x.get(j).copied().unwrap_or(0.0)).sum::<f64>() / n).collect();
        let sd: Vec<f64> = (0..d)
            .map(|j| {
                (xs.iter().map(|x| (x.get(j).copied().unwrap_or(0.0) - mean.get(j).copied().unwrap_or(0.0)).powi(2)).sum::<f64>() / n)
                    .sqrt()
                    .max(1e-9)
            })
            .collect();
        let z: Vec<Vec<f64>> = xs.iter().map(|x| Self::scale(x, &mean, &sd)).collect();
        let k = d + 1;
        let mut w = vec![0.0; k];
        for _ in 0..25 {
            let mut g = vec![0.0; k];
            let mut h = vec![vec![0.0; k]; k];
            for (x, &y) in z.iter().zip(ys) {
                let p = sigmoid(dot(&w, x));
                let r = p - f64::from(u8::from(y));
                for i in 0..k {
                    let xi = x.get(i).copied().unwrap_or(0.0);
                    if let Some(gi) = g.get_mut(i) {
                        *gi += r * xi;
                    }
                    for j in 0..k {
                        if let Some(hij) = h.get_mut(i).and_then(|row| row.get_mut(j)) {
                            *hij += p * (1.0 - p) * xi * x.get(j).copied().unwrap_or(0.0);
                        }
                    }
                }
            }
            // Ridge on the feature weights (not the bias).
            for i in 0..d {
                if let (Some(gi), Some(wi)) = (g.get_mut(i), w.get(i)) {
                    *gi += lambda * wi;
                }
                if let Some(hii) = h.get_mut(i).and_then(|row| row.get_mut(i)) {
                    *hii += lambda;
                }
            }
            let Some(step) = solve(h, g) else { break };
            let mut moved = 0.0_f64;
            for (wi, si) in w.iter_mut().zip(&step) {
                *wi -= si;
                moved = moved.max(si.abs());
            }
            if moved < 1e-9 {
                break;
            }
        }
        Self { w, mean, sd }
    }
    fn scale(x: &[f64], mean: &[f64], sd: &[f64]) -> Vec<f64> {
        let mut v: Vec<f64> = x.iter().zip(mean.iter().zip(sd)).map(|(a, (m, s))| (a - m) / s).collect();
        v.push(1.0);
        v
    }
    /// Probability that the decision's first place is right.
    #[must_use]
    pub fn predict(&self, x: &[f64]) -> f64 {
        sigmoid(dot(&self.w, &Self::scale(x, &self.mean, &self.sd)))
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Solve `h s = g` by Gaussian elimination with partial pivoting.
fn solve(mut h: Vec<Vec<f64>>, mut g: Vec<f64>) -> Option<Vec<f64>> {
    let n = g.len();
    for c in 0..n {
        let at = |r: usize| h.get(r).and_then(|row| row.get(c)).map_or(0.0, |v| v.abs());
        let p = (c..n).max_by(|a, b| at(*a).total_cmp(&at(*b)))?;
        if h.get(p)?.get(c)?.abs() < 1e-12 {
            return None;
        }
        h.swap(c, p);
        g.swap(c, p);
        let pivot = *h.get(c)?.get(c)?;
        let row_c = h.get(c)?.clone();
        let gc = *g.get(c)?;
        for r in 0..n {
            if r == c {
                continue;
            }
            let f = h.get(r)?.get(c)? / pivot;
            if let Some(row) = h.get_mut(r) {
                for (x, y) in row.iter_mut().zip(&row_c) {
                    *x -= f * y;
                }
            }
            if let Some(gr) = g.get_mut(r) {
                *gr -= f * gc;
            }
        }
    }
    Some((0..n).map(|i| g.get(i).copied().unwrap_or(0.0) / h.get(i).and_then(|r| r.get(i)).copied().unwrap_or(1.0)).collect())
}

/// Log-loss of one prediction.
#[must_use]
pub fn log_loss(p: f64, y: bool) -> f64 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    if y { -p.ln() } else { -(1.0 - p).ln() }
}

/// The gate's figures: mean log-loss of the base probability, of the calibrator left out one
/// question at a time, of the same with shuffled labels; the paired sign test of the calibrator
/// against the base; and the verdict (keep only when it beats both the base and the control).
#[derive(Clone, Debug, PartialEq)]
pub struct Gate {
    pub base: f64,
    pub calibrated: f64,
    pub shuffled: f64,
    pub better: usize,
    pub worse: usize,
    pub p: f64,
    pub keep: bool,
}

/// Run the gate. `base[i]` is the uncalibrated probability of decision `i` (the reranker's P(yes)
/// of its first place).
#[must_use]
pub fn gate(xs: &[Vec<f64>], ys: &[bool], base: &[f64], lambda: f64) -> Gate {
    let n = xs.len();
    let loqo = |labels: &[bool]| -> Vec<f64> {
        (0..n)
            .map(|i| {
                let (tx, ty): (Vec<Vec<f64>>, Vec<bool>) =
                    (0..n).filter(|j| *j != i).filter_map(|j| Some((xs.get(j)?.clone(), *labels.get(j)?))).unzip();
                let c = Calibrator::fit(&tx, &ty, lambda);
                xs.get(i).map_or(0.5, |x| c.predict(x))
            })
            .collect()
    };
    let cal = loqo(ys);
    // Shuffled-label control: labels rotated by a fixed stride, so the control is reproducible.
    let stride = (n / 3).max(1);
    let rot: Vec<bool> = (0..n).map(|i| ys.get((i + stride) % n.max(1)).copied().unwrap_or(false)).collect();
    let shuf = loqo(&rot);
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let mean = |ps: &[f64]| ps.iter().zip(ys).map(|(p, y)| log_loss(*p, *y)).sum::<f64>() / n.max(1) as f64;
    let (mut better, mut worse) = (0, 0);
    for ((c, b), y) in cal.iter().zip(base).zip(ys) {
        let (lc, lb) = (log_loss(*c, *y), log_loss(*b, *y));
        if lc + 1e-12 < lb {
            better += 1;
        } else if lb + 1e-12 < lc {
            worse += 1;
        }
    }
    let (b, c, s) = (mean(base), mean(&cal), mean(&shuf));
    let p = sign_p(better, worse);
    Gate { base: b, calibrated: c, shuffled: s, better, worse, p, keep: c < b && c < s && p < 0.05 }
}

/// Section check: on separable synthetic data the calibrator ranks and is calibrated better than
/// a flat base; Newton converges; with random labels the gate does not keep it.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let xs: Vec<Vec<f64>> = (0..40).map(|i| vec![f64::from(i) / 10.0 - 2.0, 0.5, 0.25]).collect();
    let ys: Vec<bool> = (0..40).map(|i| i >= 20).collect();
    let c = Calibrator::fit(&xs, &ys, 0.1);
    if !(c.predict(&[1.5, 0.5, 0.25]) > 0.8 && c.predict(&[-1.5, 0.5, 0.25]) < 0.2) {
        return Err("nmlearn.calibrate: fit".into());
    }
    let g = gate(&xs, &ys, &vec![0.5; 40], 0.1);
    if !g.keep {
        return Err(format!("nmlearn.calibrate: gate refused a real signal {g:?}"));
    }
    let noise: Vec<bool> = (0..40).map(|i| (i * 7919 + 13) % 5 < 2).collect();
    let g = gate(&xs, &noise, &vec![0.4; 40], 0.1);
    if g.keep {
        return Err(format!("nmlearn.calibrate: gate kept noise {g:?}"));
    }
    if features(&[0.9, 0.1, 0.6]).len() != 3 {
        return Err("nmlearn.calibrate: features".into());
    }
    Ok(())
}
