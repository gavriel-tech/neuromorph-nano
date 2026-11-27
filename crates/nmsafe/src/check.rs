//! Section check of the safetensors reader (added in the mini; the rest of `src/` is a byte
//! copy of emma-labs `native/rscore/loader`, build record runs/2026-11-07-build-record-copied.md).
use crate::{TensorIndex, bfloat16_to_f32, f16_to_f32, f32_to_f16};

/// One section check: (sub-component id, check returning ok or one fault line).
pub type Check = (&'static str, fn() -> Result<(), String>);

/// Section checks of this crate.
#[must_use]
pub fn checks() -> Vec<Check> {
    vec![("nmsafe.tensors", tensors), ("nmsafe.numeric", numeric)]
}

/// A two-tensor safetensors file written to the mini's scratch folder: the header parses,
/// shapes and ranges are right, a range read returns the stored bytes, a range past the end
/// is refused.
fn tensors() -> Result<(), String> {
    let dir = nmtype::tests_dir("section-checks/nmsafe");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let header = r#"{"a":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}"#;
    let mut bytes = u64::try_from(header.len()).map_err(|e| e.to_string())?.to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0x80, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40]);
    bytes.extend_from_slice(&1.5_f32.to_le_bytes());
    std::fs::write(dir.join("tiny.safetensors"), &bytes).map_err(|e| e.to_string())?;
    let index = TensorIndex::open(&dir).map_err(|e| e.to_string())?;
    let a = index.tensors.get("a").ok_or("nmsafe.tensors: tensor a missing")?;
    let b = index.tensors.get("b").ok_or("nmsafe.tensors: tensor b missing")?;
    if a.shape != [2, 2] || a.dtype != "BF16" || a.end - a.start != 8 || b.end - b.start != 4 {
        return Err(format!("nmsafe.tensors: descriptors {a:?} {b:?}"));
    }
    let raw = a.read_range(0, 2).map_err(|e| e.to_string())?;
    if bfloat16_to_f32(u16::from_le_bytes([raw[0], raw[1]])).to_bits() != 1.0_f32.to_bits() {
        return Err("nmsafe.tensors: range read".into());
    }
    if a.read_range(6, 4).is_ok() {
        return Err("nmsafe.tensors: a range past the tensor was read".into());
    }
    Ok(())
}

/// Binary16 conversions round to nearest even and keep special values.
fn numeric() -> Result<(), String> {
    let one = 1.0_f32.to_bits();
    if f32_to_f16(1.0) != 0x3c00
        || f16_to_f32(0x3c00).to_bits() != one
        || f32_to_f16(65520.0) != 0x7c00
        || f16_to_f32(f32_to_f16(1.0 + 1.0 / 4096.0)).to_bits() != one
    {
        return Err("nmsafe.numeric: binary16 rounding".into());
    }
    Ok(())
}
