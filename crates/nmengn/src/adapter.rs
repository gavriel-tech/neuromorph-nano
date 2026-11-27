//! Learning-weight unit as an additive low-rank adapter on one assembly node (PLAN 4c
//! `adapter`, guide 10.1; from `framework/stacks/blkstk/src/adapter.rs` and LoRA, Hu et al.
//! 2021): `y += scale * up(down(x))` after a `linear` component, applied unmerged, so the
//! node's own clusters are never written and unloading the unit is the exact rollback. The
//! small matrices run on the CPU with the engine's binary16 rounding points; on the GPU the
//! delta is added on the device, so the kernels see no new shapes.
use crate::half::r16;
use nmtype::{Fault, FaultKind};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmengn.adapter", FaultKind::Interface, m)
}

/// One unit's weights bound to one node: `down [rank, k]` (the node's input width), `up
/// [n, rank]` (its output width), both binary16-exact f32 values.
#[derive(Clone, Debug, PartialEq)]
pub struct Adapter {
    pub unit: String,
    pub node: String,
    pub rank: usize,
    pub k: usize,
    pub n: usize,
    pub down: Vec<f32>,
    pub up: Vec<f32>,
    pub scale: f32,
}
impl Adapter {
    /// Build and check the shapes.
    /// # Errors
    /// A matrix that does not fit `rank`, `k` and `n`, or a non-finite scale.
    #[expect(clippy::too_many_arguments, reason = "a unit is its names, its three widths, two matrices and a scale")]
    pub fn new(unit: &str, node: &str, rank: usize, k: usize, n: usize, down: Vec<f32>, up: Vec<f32>, scale: f32) -> Result<Self, Fault> {
        if rank == 0 || down.len() != rank * k || up.len() != n * rank || !scale.is_finite() {
            return Err(bad(format!("{unit}: down {} (want {rank}x{k}), up {} (want {n}x{rank}), scale {scale}", down.len(), up.len())));
        }
        Ok(Self { unit: unit.into(), node: node.into(), rank, k, n, down, up, scale })
    }
    /// The delta `[tokens, n]` for `x` `[tokens, k]`: `r16(scale * up(r16(down(x))))`.
    #[must_use]
    pub fn delta(&self, x: &[f32], tokens: usize) -> Vec<f32> {
        let (r, k, n) = (self.rank, self.k, self.n);
        let mut out = vec![0.0_f32; tokens * n];
        let mut h = vec![0.0_f32; r];
        for t in 0..tokens {
            let xs = x.get(t * k..(t + 1) * k).unwrap_or(&[]);
            for (j, hv) in h.iter_mut().enumerate() {
                let row = self.down.get(j * k..(j + 1) * k).unwrap_or(&[]);
                *hv = r16(xs.iter().zip(row).map(|(a, b)| a * b).sum::<f32>());
            }
            for i in 0..n {
                let row = self.up.get(i * r..(i + 1) * r).unwrap_or(&[]);
                let v = self.scale * row.iter().zip(&h).map(|(a, b)| a * b).sum::<f32>();
                if let Some(o) = out.get_mut(t * n + i) {
                    *o = r16(v);
                }
            }
        }
        out
    }
    /// Is the unit the identity (every `up` value zero)?
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.up.iter().all(|v| *v == 0.0)
    }
}

/// Self-check of `adapter`: a zero `up` gives an exact zero delta; a rank-1 unit reproduces
/// a hand computation; wrong shapes are refused.
pub(crate) fn check() -> Result<(), String> {
    let z = Adapter::new("u", "n", 2, 3, 2, vec![1.0; 6], vec![0.0; 4], 1.0).map_err(|e| e.to_string())?;
    if !z.is_zero() || z.delta(&[1.0, 2.0, 3.0], 1) != [0.0, 0.0] {
        return Err("nmengn.adapter: a zero unit must add nothing".into());
    }
    // down = [1, 0, 0; 0, 1, 0], up = [1, 2; 0, 0.5], x = [1, 2, 3]: h = [1, 2], y = [5, 1].
    let a =
        Adapter::new("u", "n", 2, 3, 2, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![1.0, 2.0, 0.0, 0.5], 1.0).map_err(|e| e.to_string())?;
    if a.delta(&[1.0, 2.0, 3.0], 1) != [5.0, 1.0] {
        return Err(format!("nmengn.adapter: delta {:?}", a.delta(&[1.0, 2.0, 3.0], 1)));
    }
    if Adapter::new("u", "n", 2, 3, 2, vec![1.0; 5], vec![0.0; 4], 1.0).is_ok() {
        return Err("nmengn.adapter: a wrong shape was accepted".into());
    }
    Ok(())
}
