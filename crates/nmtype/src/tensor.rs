//! Tensor payload for neural components (guide 8.1, 9.5). Data is either owned f32 values
//! or a shared, read-once byte range (`Arc<[u8]>`) viewed without copying.
use crate::component::{Fault, FaultKind};
use crate::wire::{self, named_enum};
use nmjson::Value;
use std::borrow::Cow;
use std::sync::Arc;

named_enum! {
    /// Element type. Q8 and Q4 need their scale set to dequantize (nmengn::safetensors).
    pub enum DType in "tensor" { F32 => "f32", F16 => "f16", BF16 => "bf16", Q8 => "q8", Q4 => "q4" }
}

/// Owned values or a shared byte range.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorData {
    Owned(Vec<f32>),
    Shared(Arc<[u8]>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data: TensorData,
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let frac = u32::from(h & 0x3ff);
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: normalise the fraction.
            let mut e = 127 - 15 + 1;
            let mut m = frac;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, _) => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

impl Tensor {
    /// An owned f32 tensor of zeros.
    #[must_use]
    pub fn zeros(shape: &[usize]) -> Self {
        Self { shape: shape.to_vec(), dtype: DType::F32, data: TensorData::Owned(vec![0.0; shape.iter().product()]) }
    }
    /// Number of elements the shape describes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The values as f32: borrowed when owned, decoded when shared (F32, F16, BF16).
    /// # Errors
    /// Q8 and Q4 need a scale set (dequantize in nmengn); a byte length that does not match
    /// the shape is an integrity fault.
    pub fn as_f32(&self) -> Result<Cow<'_, [f32]>, Fault> {
        let fault = |msg: String| Fault::new("nmtype.tensor", FaultKind::Integrity, msg);
        match &self.data {
            TensorData::Owned(v) if v.len() == self.len() => Ok(Cow::Borrowed(v)),
            TensorData::Owned(v) => Err(fault(format!("{} values for {} elements", v.len(), self.len()))),
            TensorData::Shared(bytes) => {
                let width = match self.dtype {
                    DType::F32 => 4,
                    DType::F16 | DType::BF16 => 2,
                    DType::Q8 | DType::Q4 => {
                        return Err(Fault::new(
                            "nmtype.tensor",
                            FaultKind::Interface,
                            "quantized data needs its scale set; dequantize in nmengn",
                        ));
                    }
                };
                if bytes.len() != self.len() * width {
                    return Err(fault(format!("{} bytes for {} elements", bytes.len(), self.len())));
                }
                let out = match self.dtype {
                    DType::F32 => bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
                    DType::F16 => bytes.as_chunks::<2>().0.iter().map(|c| f16_to_f32(u16::from_le_bytes(*c))).collect(),
                    _ => bytes.as_chunks::<2>().0.iter().map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16)).collect(),
                };
                Ok(Cow::Owned(out))
            }
        }
    }
    /// JSON form for telemetry and payload logs: owned values as numbers, shared bytes as
    /// a byte array.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let data = match &self.data {
            TensorData::Owned(v) => wire::obj(vec![("owned", Value::Array(v.iter().map(|x| wire::f(*x)).collect()))]),
            TensorData::Shared(b) => wire::obj(vec![("shared", Value::Array(b.iter().map(|x| wire::u(u64::from(*x))).collect()))]),
        };
        wire::obj(vec![
            ("shape", Value::Array(self.shape.iter().map(|d| wire::u(*d as u64)).collect())),
            ("dtype", self.dtype.to_value()),
            ("data", data),
        ])
    }
    /// # Errors
    /// Rejects a malformed value.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let m = "tensor";
        let num = |x: &Value| x.integer().map_err(|e| wire::bad(m, e.to_string()));
        let shape = wire::get_a(v, m, "shape")?
            .iter()
            .map(|d| num(d).and_then(|n| usize::try_from(n).map_err(|_| wire::bad(m, "dim"))))
            .collect::<Result<_, _>>()?;
        let data = wire::get_v(v, m, "data")?;
        let data = if let Ok(items) = data.get("owned").and_then(Value::array) {
            TensorData::Owned(
                items
                    .iter()
                    .map(|x| match x {
                        Value::Number(t) => t.parse().map_err(|_| wire::bad(m, "value")),
                        _ => Err(wire::bad(m, "value")),
                    })
                    .collect::<Result<_, _>>()?,
            )
        } else {
            let items = wire::get_a(data, m, "shared")?;
            let bytes: Vec<u8> =
                items.iter().map(|x| num(x).and_then(|n| u8::try_from(n).map_err(|_| wire::bad(m, "byte")))).collect::<Result<_, _>>()?;
            TensorData::Shared(Arc::from(bytes))
        };
        Ok(Self { shape, dtype: wire::get_e(v, m, "dtype", DType::from_value)?, data })
    }
}

/// Self-check of the `tensor` sub-component: zeros, f16 and bf16 decoding of a shared
/// range, quantized data refused without a scale set, round trip.
pub(crate) fn check() -> Result<(), String> {
    let z = Tensor::zeros(&[2, 3]);
    if z.as_f32().map_err(|e| e.to_string())?.len() != 6 {
        return Err("nmtype.tensor: zeros length".into());
    }
    // 1.0 and -2.0 in f16 (0x3C00, 0xC000), little-endian.
    let h = Tensor { shape: vec![2], dtype: DType::F16, data: TensorData::Shared(Arc::from(vec![0x00, 0x3C, 0x00, 0xC0])) };
    if h.as_f32().map_err(|e| e.to_string())?.as_ref() != [1.0, -2.0] {
        return Err("nmtype.tensor: f16 decode".into());
    }
    let b = Tensor { dtype: DType::BF16, data: TensorData::Shared(Arc::from(vec![0x80, 0x3F, 0x00, 0x40])), ..h.clone() };
    if b.as_f32().map_err(|e| e.to_string())?.as_ref() != [1.0, 2.0] {
        return Err("nmtype.tensor: bf16 decode".into());
    }
    let q = Tensor { dtype: DType::Q8, ..h.clone() };
    if q.as_f32().is_ok() {
        return Err("nmtype.tensor: Q8 accepted without a scale set".into());
    }
    if Tensor::from_value(&h.to_value()).map_err(|e| e.to_string())? != h {
        return Err("nmtype.tensor: round trip".into());
    }
    Ok(())
}
