//! Explicit IEEE conversions and float32 row arithmetic.
use crate::{Error, Result};

/// Expand the high 16 bits of an IEEE float32 value.
pub fn bfloat16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Decode an IEEE binary16 value, preserving special values.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = (bits >> 10) & 31;
    let fraction = u32::from(bits & 1023);
    let payload = match exp {
        0 if fraction == 0 => 0,
        0 => {
            let shift = fraction.leading_zeros() - 21;
            ((113 - shift) << 23) | ((fraction << shift & 1023) << 13)
        }
        31 => 0x7f80_0000 | (fraction << 13),
        _ => ((u32::from(exp) + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(sign | payload)
}
fn rounded(value: u32, shift: u32) -> u32 {
    let base = value >> shift;
    let rest = value & ((1 << shift) - 1);
    let midpoint = 1 << (shift - 1);
    base + u32::from(rest > midpoint || (rest == midpoint && base & 1 != 0))
}
/// Round float32 to IEEE binary16 using nearest, ties to even.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = u16::try_from((bits >> 16) & 0x8000).unwrap_or_default();
    let exp = (bits >> 23) & 255;
    let fraction = bits & 0x007f_ffff;
    let payload = if exp == 255 {
        0x7c00 | if fraction == 0 { 0 } else { 0x0200 }
    } else if exp > 142 {
        0x7c00
    } else if exp < 102 {
        0
    } else if exp < 113 {
        rounded(fraction | 0x0080_0000, 126 - exp)
    } else {
        ((exp - 112) << 10) + rounded(fraction, 13)
    };
    sign | u16::try_from(payload).unwrap_or(0x7c00)
}

/// Quantised row-major weights and one little-endian binary16 scale per row.
#[derive(Debug)]
pub struct Quantized {
    /// Signed int8 values as their raw two's-complement bytes.
    pub values: Vec<u8>,
    /// Binary16 scale bits in row order.
    pub scales: Vec<u16>,
}
#[expect(
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "bounded rounded float is explicitly converted to its signed int8 representation"
)]
fn int8(value: f32) -> u8 {
    (value.round_ties_even().clamp(-127.0, 127.0) as i8).to_ne_bytes()[0]
}

/// Quantise finite float32 weights with the oracle's per-row order of operations.
/// # Errors
/// Rejects zero width, incomplete rows and non-finite values.
pub fn quantize(weights: &[f32], columns: usize) -> Result<Quantized> {
    if columns == 0 || !weights.len().is_multiple_of(columns) || weights.iter().any(|x| !x.is_finite()) {
        return Err(Error::Invalid(format!("invalid quantisation shape: {} values, {columns} columns, finite required", weights.len())));
    }
    let mut values = Vec::with_capacity(weights.len());
    let mut scales = Vec::new();
    for row in weights.chunks_exact(columns) {
        let maximum = row.iter().fold(0.0_f32, |v, x| v.max(x.abs()));
        let scale = maximum.max(1e-8) / 127.0;
        values.extend(row.iter().map(|x| int8(*x / scale)));
        scales.push(f32_to_f16(scale));
    }
    Ok(Quantized { values, scales })
}
