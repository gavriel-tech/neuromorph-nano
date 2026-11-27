//! Immutable expert byte maps over original tensors or aligned packed files.
use crate::{Error, Result, Tensor};
use std::collections::BTreeMap;

/// Whether ranges address an original safetensors payload or an aligned expert pack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeLayout {
    /// Original offsets need no alignment and never cause a payload copy.
    Original,
    /// Each expert starts on a 4 KiB file boundary.
    Packed,
}
/// An expert's independently mountable portion of a validated tensor descriptor.
#[derive(Debug, Clone)]
pub struct ExpertRange {
    /// Expert id local to this bank's block.
    pub expert: usize,
    /// Projection identity, for example gate codes, scales or biases.
    pub part: String,
    /// Validated source descriptor retained by reference when mounted.
    pub tensor: Tensor,
    /// Byte offset relative to the tensor payload.
    pub offset: u64,
    /// Number of bytes in this expert part.
    pub length: u64,
}
impl ExpertRange {
    /// Read just this mounted part from its original file.
    /// # Errors
    /// Rejects invalid ranges, address-space overflow or local I/O errors.
    pub fn read(&self) -> Result<Vec<u8>> {
        self.tensor.read_range(self.offset, usize::try_from(self.length).map_err(|e| Error::Invalid(e.to_string()))?)
    }
}
/// A block's descriptor-only expert bank; construction does not read any payload.
#[derive(Debug)]
pub struct ExpertBank {
    ranges: BTreeMap<usize, Vec<ExpertRange>>,
}
impl ExpertBank {
    /// Validate byte maps once, preserving the original files and immutable metadata.
    /// # Errors
    /// Rejects invalid/overlapping ranges, duplicate parts and unaligned packed starts.
    pub fn new(ranges: Vec<ExpertRange>, layout: RangeLayout) -> Result<Self> {
        let mut bank: BTreeMap<usize, Vec<ExpertRange>> = BTreeMap::new();
        for range in ranges {
            validate(&range)?;
            for prior in bank.values().flatten() {
                if prior.tensor.file == range.tensor.file {
                    let a = prior.tensor.start + prior.offset;
                    let b = range.tensor.start + range.offset;
                    if a < b + range.length && b < a + prior.length {
                        return Err(Error::Invalid("overlapping expert byte intervals".into()));
                    }
                }
            }
            let parts = bank.entry(range.expert).or_default();
            if parts.iter().any(|p| p.part == range.part) {
                return Err(Error::Invalid(format!("duplicate expert part {}", range.part)));
            }
            parts.push(range);
        }
        if layout == RangeLayout::Packed {
            for parts in bank.values() {
                let mut starts: BTreeMap<_, u64> = BTreeMap::new();
                for p in parts {
                    let start = p.tensor.start + p.offset;
                    starts.entry(&p.tensor.file).and_modify(|v| *v = (*v).min(start)).or_insert(start);
                }
                if starts.values().any(|n| !n.is_multiple_of(4096)) {
                    return Err(Error::Invalid("packed expert must start at 4 KiB boundary".into()));
                }
            }
        }
        Ok(Self { ranges: bank })
    }
    /// Borrow an expert's descriptors without reading or copying its weights.
    /// # Errors
    /// Rejects an expert absent from this block map.
    pub fn mount(&self, expert: usize) -> Result<&[ExpertRange]> {
        self.ranges.get(&expert).map(Vec::as_slice).ok_or_else(|| Error::Invalid(format!("expert {expert} absent from block map")))
    }
}
fn validate(range: &ExpertRange) -> Result<()> {
    let t = &range.tensor;
    if range.part.is_empty()
        || range.length == 0
        || t.start > t.end
        || range.offset.checked_add(range.length).is_none_or(|n| n > t.end - t.start)
    {
        return Err(Error::Invalid(format!(
            "invalid expert {} part {} range {}+{} in {}..{}",
            range.expert, range.part, range.offset, range.length, t.start, t.end
        )));
    }
    Ok(())
}
