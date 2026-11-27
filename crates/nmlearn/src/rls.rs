//! Recursive least squares readout (redesigned for the mini from EAMM V108.1, where RLS with
//! no forgetting factor passed its retention check: the exact least-squares solution over every
//! outcome seen, updated online at O(D^2) per step). Forward-only: no gradient, no backward
//! pass; the frozen weights are never touched. The readout sits beside a component's output.
//! State: `w` (D) and `P` (D x D), in f64 for stability.

/// One RLS readout.
#[derive(Clone, Debug, PartialEq)]
pub struct Rls {
    pub d: usize,
    pub w: Vec<f64>,
    p: Vec<f64>,
    pub updates: u64,
}
impl Rls {
    /// A readout of `d` features with `P = I / delta`.
    #[must_use]
    pub fn new(d: usize, delta: f64) -> Self {
        let mut p = vec![0.0; d * d];
        for i in 0..d {
            if let Some(x) = p.get_mut(i * d + i) {
                *x = 1.0 / delta;
            }
        }
        Self { d, w: vec![0.0; d], p, updates: 0 }
    }
    /// The readout value for features `x`.
    #[must_use]
    pub fn predict(&self, x: &[f64]) -> f64 {
        self.w.iter().zip(x).map(|(a, b)| a * b).sum()
    }
    /// One update towards target `y`.
    pub fn update(&mut self, x: &[f64], y: f64) {
        let d = self.d;
        let px: Vec<f64> =
            (0..d).map(|i| self.p.get(i * d..i * d + d).map_or(0.0, |row| row.iter().zip(x).map(|(a, b)| a * b).sum())).collect();
        let denom = 1.0 + x.iter().zip(&px).map(|(a, b)| a * b).sum::<f64>();
        let err = y - self.predict(x);
        for (i, wi) in self.w.iter_mut().enumerate() {
            *wi += px.get(i).copied().unwrap_or(0.0) / denom * err;
        }
        for i in 0..d {
            let ki = px.get(i).copied().unwrap_or(0.0) / denom;
            if let Some(row) = self.p.get_mut(i * d..i * d + d) {
                for (r, pj) in row.iter_mut().zip(&px) {
                    *r -= ki * pj;
                }
            }
        }
        self.updates += 1;
    }
    /// Numbers held as state (`w` plus `P`).
    #[must_use]
    pub const fn state_numbers(&self) -> usize {
        self.d + self.d * self.d
    }
}

/// Self-check of `rls`: a noiseless linear target is recovered from a few samples, and the
/// state size is D + D^2.
pub(crate) fn check() -> Result<(), String> {
    let truth = [0.5, -2.0, 1.25];
    let mut r = Rls::new(3, 1e-6);
    let xs = [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, -1.0, 1.0], [0.5, 0.25, 1.0]];
    for x in &xs {
        let y: f64 = truth.iter().zip(x).map(|(a, b)| a * b).sum();
        r.update(x, y);
    }
    let probe = [3.0, 2.0, 1.0];
    let want: f64 = truth.iter().zip(&probe).map(|(a, b)| a * b).sum();
    if (r.predict(&probe) - want).abs() > 0.05 || r.state_numbers() != 12 || r.updates != 5 {
        return Err(format!("nmlearn.rls: predicted {} for {want}", r.predict(&probe)));
    }
    Ok(())
}
