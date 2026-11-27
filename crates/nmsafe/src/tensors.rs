//! Validated safetensors descriptors and selective read-only I/O.
use crate::{Error, Result, json};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// A validated tensor's identity and absolute byte interval.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    /// Tensor name from the header.
    pub name: String,
    /// Safetensors dtype code.
    pub dtype: String,
    /// Tensor dimensions, including scalar and empty tensors.
    pub shape: Vec<u64>,
    /// Original file read in place.
    pub file: PathBuf,
    /// Absolute inclusive start offset.
    pub start: u64,
    /// Absolute exclusive end offset.
    pub end: u64,
}
impl Tensor {
    /// Read a relative byte interval without loading any other payload.
    /// # Errors
    /// Returns an error for out-of-bounds intervals or unreadable/truncated files.
    pub fn read_range(&self, offset: u64, length: usize) -> Result<Vec<u8>> {
        let length64 = u64::try_from(length).map_err(|e| Error::Invalid(e.to_string()))?;
        if self.start > self.end || offset.checked_add(length64).is_none_or(|n| n > self.end - self.start) {
            return Err(Error::Invalid(format!(
                "{} range offset={offset}, length={length} outside {}..{}",
                self.name, self.start, self.end
            )));
        }
        let mut stream = File::open(&self.file).map_err(|source| Error::Io { context: self.file.display().to_string(), source })?;
        stream.seek(SeekFrom::Start(self.start + offset))?;
        let mut bytes = vec![0; length];
        stream.read_exact(&mut bytes)?;
        Ok(bytes)
    }
}

/// All descriptors of a local weight set, with no payload cached.
#[derive(Debug)]
pub struct TensorIndex {
    /// Descriptors sorted by tensor name.
    pub tensors: BTreeMap<String, Tensor>,
}
impl TensorIndex {
    /// Read headers of all safetensors in a directory.
    /// # Errors
    /// Rejects missing files, malformed headers, duplicate names and invalid ranges.
    pub fn open(folder: &Path) -> Result<Self> {
        let mut files = std::fs::read_dir(folder)?.map(|p| p.map(|e| e.path())).collect::<std::io::Result<Vec<_>>>()?;
        files.retain(|p| p.extension().is_some_and(|x| x == "safetensors"));
        files.sort();
        if files.is_empty() {
            return Err(Error::Invalid(format!("no safetensors in {}", folder.display())));
        }
        let mut tensors = BTreeMap::new();
        for file in files {
            for tensor in header(&file)? {
                if tensors.insert(tensor.name.clone(), tensor).is_some() {
                    return Err(Error::Invalid("duplicate tensor across files".into()));
                }
            }
        }
        Ok(Self { tensors })
    }
}
fn width(dtype: &str) -> Result<u64> {
    match dtype {
        "F64" | "I64" | "U64" => Ok(8),
        "F32" | "I32" | "U32" => Ok(4),
        "BF16" | "F16" | "I16" | "U16" => Ok(2),
        "I8" | "U8" | "BOOL" | "F8_E4M3" | "F8_E5M2" | "F8_E4M3FNUZ" | "F8_E5M2FNUZ" | "F8_E8M0" => Ok(1),
        _ => Err(Error::Invalid(format!("unsupported safetensors dtype {dtype}"))),
    }
}
fn header(file: &Path) -> Result<Vec<Tensor>> {
    let mut stream = File::open(file)?;
    let size = stream.metadata()?.len();
    let mut bytes = [0; 8];
    stream.read_exact(&mut bytes)?;
    let length = u64::from_le_bytes(bytes);
    if length == 0 || length > 100_000_000 || length > size.saturating_sub(8) {
        return Err(Error::Invalid(format!("{} header length {length}, file size {size}", file.display())));
    }
    let start = 8 + length;
    let mut raw = vec![0; usize::try_from(length).map_err(|e| Error::Invalid(e.to_string()))?];
    stream.read_exact(&mut raw)?;
    if raw.first() != Some(&b'{') {
        return Err(Error::Invalid("header must start with {".into()));
    }
    let value = json::parse(&raw)?;
    let mut out = Vec::new();
    for (name, item) in value.object()? {
        if name == "__metadata__" {
            continue;
        }
        let dtype = item.get("dtype")?.string()?.to_owned();
        let shape = item.get("shape")?.array()?.iter().map(json::Value::integer).collect::<Result<Vec<_>>>()?;
        let range = item.get("data_offsets")?.array()?;
        if range.len() != 2 {
            return Err(Error::Invalid(format!("{name}: expected two offsets")));
        }
        let a = range[0].integer()?;
        let b = range[1].integer()?;
        let expected = shape
            .iter()
            .try_fold(width(&dtype)?, |v, n| v.checked_mul(*n).ok_or_else(|| Error::Invalid(format!("{name}: shape overflow"))))?;
        if a > b || b > size - start || b - a != expected {
            return Err(Error::Invalid(format!("{name}: invalid range {a}..{b}, expected {expected} bytes")));
        }
        out.push(Tensor { name: name.clone(), dtype, shape, file: file.to_owned(), start: start + a, end: start + b });
    }
    Ok(out)
}
