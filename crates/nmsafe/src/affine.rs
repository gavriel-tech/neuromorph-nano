//! MLX affine four-bit groups, independent of the signed symmetric int4 format.
use crate::{Error, Result};

/// Borrowed row-major MLX weights: low nibble first, eight codes per LE uint32.
#[derive(Debug, Clone, Copy)]
pub struct AffineInt4<'a> {
    /// Packed unsigned codes, with no zero-point subtraction.
    pub codes: &'a [u8],
    /// One float scale per consecutive group of 64 codes.
    pub scales: &'a [f32],
    /// One additive float bias per group, not an integer zero point.
    pub biases: &'a [f32],
}
impl AffineInt4<'_> {
    /// Fixed group size of this released affine format.
    pub const GROUP: usize = 64;

    /// Decode complete groups using `scale * unsigned_code + bias` in float32.
    /// # Errors
    /// Rejects partial groups, mismatched metadata and non-finite parameters/results.
    pub fn dequantize(&self) -> Result<Vec<f32>> {
        let groups = self.codes.len() / (Self::GROUP / 2);
        if self.codes.is_empty()
            || !self.codes.len().is_multiple_of(Self::GROUP / 2)
            || self.scales.len() != groups
            || self.biases.len() != groups
            || self.scales.iter().chain(self.biases).any(|x| !x.is_finite())
        {
            return Err(Error::Invalid(format!(
                "MLX affine int4 requires complete 64-code groups: bytes={}, scales={}, biases={}",
                self.codes.len(),
                self.scales.len(),
                self.biases.len()
            )));
        }
        let mut out = Vec::with_capacity(self.codes.len().saturating_mul(2));
        for (group, bytes) in self.codes.as_chunks::<32>().0.iter().enumerate() {
            for &byte in bytes {
                for code in [byte & 15, byte >> 4] {
                    let value = self.scales[group] * f32::from(code) + self.biases[group];
                    if !value.is_finite() {
                        return Err(Error::Invalid("affine int4 result exceeds float32".into()));
                    }
                    out.push(value);
                }
            }
        }
        Ok(out)
    }
}
