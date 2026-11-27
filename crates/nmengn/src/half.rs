//! Binary16 rounding as PyTorch float16 does it (research note section 3). Activations are
//! carried as f32 values that are exactly representable in binary16; `r16` rounds an f32 to
//! the nearest binary16 value (ties to even). Weights are stored bfloat16 and become binary16
//! once (what `model.half()` does): `w16` widens a bfloat16 and rounds it to binary16.
use nmsafe::{f16_to_f32, f32_to_f16};

/// Round to the nearest binary16 value, returned as f32. Normal-range values take the fast
/// mantissa rounding; subnormal, overflow and special values take the exact converter.
#[inline]
#[must_use]
pub fn r16(x: f32) -> f32 {
    let b = x.to_bits();
    let e = (b >> 23) & 0xff;
    if (113..=141).contains(&e) {
        let lsb = (b >> 13) & 1;
        f32::from_bits((b + 0x0fff + lsb) & !0x1fff)
    } else {
        f16_to_f32(f32_to_f16(x))
    }
}

/// A bfloat16 weight as the binary16 value PyTorch would hold after `.half()`.
#[inline]
#[must_use]
pub fn w16(bits: u16) -> f32 {
    let x = f32::from_bits(u32::from(bits) << 16);
    let e = (x.to_bits() >> 23) & 0xff;
    if (113..=142).contains(&e) { x } else { f16_to_f32(f32_to_f16(x)) }
}

/// Binary16 bits of an f32 (for device uploads).
#[inline]
#[must_use]
pub fn bits16(x: f32) -> u16 {
    f32_to_f16(x)
}

/// f32 from binary16 bits (for device downloads).
#[inline]
#[must_use]
pub fn from16(b: u16) -> f32 {
    f16_to_f32(b)
}

/// Self-check of `half`: the fast path equals the exact converter over a sweep of values,
/// including a tie and the edges of the normal range.
pub(crate) fn check() -> Result<(), String> {
    let mut x = -70000.0_f32;
    while x < 70000.0 {
        for v in [x, x * 1e-5, x * 1.000_244_1, 1.0 + 1.0 / 2048.0, 65504.0, 65519.0, 6.1e-5] {
            if r16(v).to_bits() != f16_to_f32(f32_to_f16(v)).to_bits() {
                return Err(format!("nmengn.half: r16({v}) = {} not {}", r16(v), f16_to_f32(f32_to_f16(v))));
            }
        }
        x += 13.37;
    }
    if w16(0x3f80) != 1.0 || w16(0x0001) != 0.0 {
        return Err("nmengn.half: w16".into());
    }
    Ok(())
}
