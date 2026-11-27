//! Decompose (guide 9.3 responsibility 4, 9.4; PLAN 4c): a checkpoint is a parameter
//! population cut into clusters. A cluster is one region of one tensor: whole, a row range or
//! a column range. Granularity `heads` (the S6 cut, research note section 5): per layer, query
//! rows and output-projection columns per query head (group `layer{n}.head{h}`), key and value
//! rows per key-value head (group `layer{n}.kv{g}`), the feed-forward and its norm whole
//! (group `layer{n}.ffn`), the attention norms whole (group `layer{n}.attn`), the embedding
//! (tied head) and the final norm whole. Every byte of every tensor lies in exactly one
//! cluster. Identity: `model/tensor/region`; the content hash is taken when the bytes are read
//! (provision).
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::collections::BTreeMap;
use std::ops::Range;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmnpcs.decompose", FaultKind::Config, m)
}

/// Region of a tensor a cluster covers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Region {
    Whole,
    Rows(Range<usize>),
    Cols(Range<usize>),
}
impl Region {
    fn label(&self) -> String {
        match self {
            Self::Whole => "whole".into(),
            Self::Rows(r) => format!("rows{}-{}", r.start, r.end),
            Self::Cols(r) => format!("cols{}-{}", r.start, r.end),
        }
    }
}

/// One cluster: identity, where its bytes are, its inseparable group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParameterCluster {
    pub id: String,
    pub model: String,
    pub tensor: String,
    pub region: Region,
    pub group: String,
    /// Element type as stored (`BF16` for the Qwen3 release).
    pub dtype: String,
    /// Shape of the whole tensor.
    pub shape: Vec<usize>,
    /// The released file that holds the tensor (its name in the population folder; a model
    /// released in several files keeps each tensor in its own file, read in place).
    pub file: String,
    /// Absolute file offset and byte length of the whole tensor.
    pub offset: u64,
    pub len: u64,
    /// Bytes of this cluster.
    pub bytes: u64,
    /// SHA-256 of the whole tensor's bytes when its source states it (a cartridge part), else
    /// empty; tensors with the same hash are held once in RAM (`provision`).
    pub sha256: String,
}
impl ParameterCluster {
    /// Rows and columns of the cluster's own block.
    #[must_use]
    pub fn dims(&self) -> (usize, usize) {
        let (r, c) = match self.shape.as_slice() {
            [n] => (1, *n),
            [r, c] => (*r, *c),
            _ => (1, self.shape.iter().product()),
        };
        match &self.region {
            Region::Whole => (r, c),
            Region::Rows(x) => (x.len(), c),
            Region::Cols(x) => (r, x.len()),
        }
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut m = BTreeMap::new();
        for (k, v) in [("id", &self.id), ("tensor", &self.tensor), ("group", &self.group), ("dtype", &self.dtype)] {
            m.insert(k.to_owned(), Value::String(v.clone()));
        }
        m.insert("region".into(), Value::String(self.region.label()));
        m.insert("bytes".into(), Value::Number(self.bytes.to_string()));
        Value::Object(m)
    }
}

/// A tensor as the safetensors header describes it (name, dtype, shape, absolute range).
#[derive(Clone, Debug)]
pub struct TensorDesc {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// The released file holding the tensor (name in the population folder).
    pub file: String,
    pub start: u64,
    pub end: u64,
    /// SHA-256 of the tensor's bytes when the source states it (a cartridge part), else empty.
    pub sha256: String,
}

fn width(dtype: &str) -> u64 {
    match dtype {
        "F32" | "I32" | "U32" => 4,
        "I8" | "U8" | "BOOL" => 1,
        _ => 2,
    }
}

/// Cut a Qwen3-family population into clusters at granularity heads.
/// # Errors
/// A tensor the cut does not know, or a missing dimension.
pub fn decompose(
    model: &str,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    tensors: &[TensorDesc],
) -> Result<Vec<ParameterCluster>, Fault> {
    let mut out = Vec::new();
    for t in tensors {
        let w = width(&t.dtype);
        let cols = t.shape.get(1).copied().unwrap_or(1);
        let rows = t.shape.first().copied().unwrap_or(1);
        let mut push = |region: Region, group: String| {
            let bytes = match &region {
                Region::Whole => t.end - t.start,
                Region::Rows(r) => (r.len() * cols) as u64 * w,
                Region::Cols(c) => (rows * c.len()) as u64 * w,
            };
            out.push(ParameterCluster {
                id: format!("{model}/{}/{}", t.name, region.label()),
                model: model.to_owned(),
                tensor: t.name.clone(),
                region,
                group,
                dtype: t.dtype.clone(),
                shape: t.shape.clone(),
                file: t.file.clone(),
                offset: t.start,
                len: t.end - t.start,
                bytes,
                sha256: t.sha256.clone(),
            });
        };
        if t.name == "model.embed_tokens.weight" {
            push(Region::Whole, "embed".into());
            continue;
        }
        if t.name == "model.norm.weight" {
            push(Region::Whole, "final".into());
            continue;
        }
        if t.name == "lm_head.weight" {
            push(Region::Whole, "head".into());
            continue;
        }
        let Some(rest) = t.name.strip_prefix("model.layers.") else {
            return Err(bad(format!("unknown tensor {}", t.name)));
        };
        let (n, part) = rest.split_once('.').ok_or_else(|| bad(format!("tensor {}", t.name)))?;
        let layer = format!("layer{n}");
        match part {
            "self_attn.q_proj.weight" => {
                (0..heads).for_each(|h| push(Region::Rows(h * head_dim..(h + 1) * head_dim), format!("{layer}.head{h}")))
            }
            "self_attn.o_proj.weight" => {
                (0..heads).for_each(|h| push(Region::Cols(h * head_dim..(h + 1) * head_dim), format!("{layer}.head{h}")))
            }
            "self_attn.k_proj.weight" | "self_attn.v_proj.weight" => {
                (0..kv_heads).for_each(|g| push(Region::Rows(g * head_dim..(g + 1) * head_dim), format!("{layer}.kv{g}")))
            }
            "self_attn.q_norm.weight" | "self_attn.k_norm.weight" | "input_layernorm.weight" => {
                push(Region::Whole, format!("{layer}.attn"))
            }
            "post_attention_layernorm.weight" | "mlp.gate_proj.weight" | "mlp.up_proj.weight" | "mlp.down_proj.weight" => {
                push(Region::Whole, format!("{layer}.ffn"))
            }
            _ => return Err(bad(format!("unknown tensor {}", t.name))),
        }
    }
    Ok(out)
}

/// Coverage audit: per tensor, the clusters' bytes add up to the tensor's bytes and their
/// row or column ranges partition the tensor (no overlap, no gap). Returns (tensors, cluster
/// bytes, tensor bytes) or the first tensor that fails.
/// # Errors
/// A tensor whose clusters overlap, leave a gap or miss bytes.
pub fn coverage(tensors: &[TensorDesc], clusters: &[ParameterCluster]) -> Result<(usize, u64, u64), Fault> {
    let mut by: BTreeMap<&str, Vec<&ParameterCluster>> = BTreeMap::new();
    for c in clusters {
        by.entry(c.tensor.as_str()).or_default().push(c);
    }
    let (mut cb, mut tb) = (0_u64, 0_u64);
    for t in tensors {
        let cs = by.get(t.name.as_str()).ok_or_else(|| bad(format!("{} has no cluster", t.name)))?;
        let sum: u64 = cs.iter().map(|c| c.bytes).sum();
        let mut ranges: Vec<Range<usize>> = cs
            .iter()
            .filter_map(|c| match &c.region {
                Region::Whole => None,
                Region::Rows(r) | Region::Cols(r) => Some(r.clone()),
            })
            .collect();
        ranges.sort_by_key(|r| r.start);
        let whole = cs.iter().filter(|c| c.region == Region::Whole).count();
        let span = match cs.first().map(|c| &c.region) {
            Some(Region::Rows(_)) => t.shape.first().copied().unwrap_or(0),
            Some(Region::Cols(_)) => t.shape.get(1).copied().unwrap_or(0),
            _ => 0,
        };
        let tiled = whole == 1 && ranges.is_empty()
            || whole == 0
                && ranges.first().is_some_and(|r| r.start == 0)
                && ranges.windows(2).all(|w| w[0].end == w[1].start)
                && ranges.last().is_some_and(|r| r.end == span);
        if sum != t.end - t.start || !tiled {
            return Err(Fault::new(
                "nmnpcs.decompose",
                FaultKind::Integrity,
                format!("{}: clusters cover {sum} of {} bytes or overlap", t.name, t.end - t.start),
            ));
        }
        cb += sum;
        tb += t.end - t.start;
    }
    Ok((tensors.len(), cb, tb))
}

/// Self-check of `decompose`: a two-layer toy header cuts into the expected groups and
/// passes the coverage audit; an overlapping cut fails it.
pub(crate) fn check() -> Result<(), String> {
    let mut ts = Vec::new();
    let mut at = 0_u64;
    let mut add = |name: &str, shape: Vec<usize>| {
        let len = shape.iter().product::<usize>() as u64 * 2;
        ts.push(TensorDesc { name: name.into(), dtype: "BF16".into(), shape, file: "toy.safetensors".into(), start: at, end: at + len, sha256: String::new() });
        at += len;
    };
    add("model.embed_tokens.weight", vec![10, 8]);
    for l in 0..2 {
        for (p, s) in [
            ("self_attn.q_proj.weight", vec![8, 8]),
            ("self_attn.k_proj.weight", vec![4, 8]),
            ("self_attn.v_proj.weight", vec![4, 8]),
            ("self_attn.o_proj.weight", vec![8, 8]),
            ("self_attn.q_norm.weight", vec![4]),
            ("self_attn.k_norm.weight", vec![4]),
            ("input_layernorm.weight", vec![8]),
            ("post_attention_layernorm.weight", vec![8]),
            ("mlp.gate_proj.weight", vec![12, 8]),
            ("mlp.up_proj.weight", vec![12, 8]),
            ("mlp.down_proj.weight", vec![8, 12]),
        ] {
            add(&format!("model.layers.{l}.{p}"), s);
        }
    }
    add("model.norm.weight", vec![8]);
    let cs = decompose("toy", 2, 1, 4, &ts).map_err(|e| e.to_string())?;
    let heads = cs.iter().filter(|c| c.group.contains(".head")).map(|c| c.group.clone()).collect::<std::collections::BTreeSet<_>>().len();
    let ffn = cs.iter().filter(|c| c.group.ends_with(".ffn")).map(|c| c.group.clone()).collect::<std::collections::BTreeSet<_>>().len();
    if heads != 4 || ffn != 2 {
        return Err(format!("nmnpcs.decompose: {heads} head groups, {ffn} ffn groups"));
    }
    coverage(&ts, &cs).map_err(|e| e.to_string())?;
    let mut bad_cut = cs.clone();
    if let Some(c) = bad_cut.iter_mut().find(|c| matches!(c.region, Region::Rows(_))) {
        c.region = Region::Rows(0..5);
    }
    if coverage(&ts, &bad_cut).is_ok() {
        return Err("nmnpcs.decompose: an overlapping cut passed the audit".into());
    }
    Ok(())
}
