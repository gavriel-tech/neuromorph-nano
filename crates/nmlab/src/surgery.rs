//! Precision all-layer surgical parameter mapping, adding, removing, swapping, and two-pass
//! anchor-then-cascade restoration (`nano-lab run pmap surgery <probe|add|remove|swap|cascade>`).
//!
//! Designed to be **fully adaptable** across all 4+ active base cartridges (`reader-base`,
//! `embedder-base`, `reranker-base`, `writer-base`, and any future MicroModel in
//! `designs/base-cartridges.json`), arbitrary layer/block counts (6, 12, 24, 28, 32, 40+ layers),
//! graph-discovered subblock topologies (via `Assembly` nodes/edges + `Population` clusters),
//! and multiple raw weight formats (`BF16`, `F16`, `F32`).
//!
//! Uses **exact signed first-order Taylor L2 error-reduction scoring** on every graph-discovered
//! residual-writing projection (`gain[r,c] = W[r,c] * (E^T X)[r,c] - 0.5 * W[r,c]^2 * ||X[:,c]||^2`,
//! where `E = Y_open - Y_cand`), only restoring pruned weights that actively move the cartridge
//! toward the untouched open reference, and automatically protecting super-weight coordinates in
//! `library/parameter-maps/<role>/super-weights.json`.

use nmengn::assembly::Assembly;
use nmengn::model::{Model, Population, RunOpts};
use nmengn::placement::Device;
use nmengn::pmap::{self, BitMask, ParamMask, ParamRecord, Stream};
use nmjson::Value;
use nmnpcs::TensorDesc;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

struct ResolvedSpec {
    short_role: String,
    pop_id: String,
    asm_id: String,
    cart_id: String,
}

/// Dynamically resolves any role alias (`reader`, `embedder`, `reranker`, `writer`), cartridge ID
/// (`writer-base`, etc.), or population ID (`qwen3-1.7b`, or any future MicroModel listed in
/// `designs/base-cartridges.json`) without hardcoding a fixed 3- or 4-model table.
fn resolve_model_spec(home: &Path, query: &str) -> Result<ResolvedSpec, String> {
    let q_low = query.trim().to_ascii_lowercase();
    let known_asm = |pop_id: &str, cart_id: &str| -> String {
        let asm_dir = home.join("designs/assemblies");
        for cand in [pop_id, cart_id] {
            if asm_dir.join(format!("{cand}.json")).exists() {
                return cand.to_owned();
            }
        }
        if pop_id.contains("embedding") || cart_id.contains("embed") {
            "qwen3-embedding".to_owned()
        } else if pop_id.contains("reranker") || cart_id.contains("rerank") || cart_id.contains("judge") {
            "qwen3-reranker".to_owned()
        } else {
            "qwen3-reference".to_owned()
        }
    };

    // 1. Read `designs/base-cartridges.json` so all 4 active cartridges and any future MicroModel
    //    registered there resolve automatically.
    let reg_path = home.join("designs/base-cartridges.json");
    if let Ok(text) = std::fs::read_to_string(&reg_path)
        && let Ok(Value::Object(root)) = nmjson::parse(text.as_bytes())
        && let Some(Value::Object(serving)) = root.get("serving")
    {
        for (pop_id, entry) in serving {
            let cart_id = entry.get("cartridge").and_then(Value::string).unwrap_or("").to_owned();
            let short = cart_id.strip_suffix("-base").unwrap_or(&cart_id).to_owned();
            let matches_alias = q_low == short.to_ascii_lowercase()
                || q_low == cart_id.to_ascii_lowercase()
                || q_low == pop_id.to_ascii_lowercase()
                || (q_low == "judge" && short == "reranker")
                || (q_low == "embed" && short == "embedder")
                || (q_low == "rerank" && short == "reranker")
                || (q_low == "read" && short == "reader")
                || (q_low == "write" && short == "writer");
            if matches_alias {
                return Ok(ResolvedSpec {
                    short_role: short,
                    pop_id: pop_id.clone(),
                    asm_id: known_asm(pop_id, &cart_id),
                    cart_id,
                });
            }
        }
    }

    // 2. Fallback defaults if registry file is absent or query is an unregistered custom population.
    let (short, pop, asm, cart) = match q_low.as_str() {
        "embedder" | "embed" => ("embedder", "qwen3-embedding-0.6b", "qwen3-embedding", "embedder-base"),
        "reranker" | "rerank" | "judge" => ("reranker", "qwen3-reranker-0.6b", "qwen3-reranker", "reranker-base"),
        "writer" | "write" => ("writer", "qwen3-1.7b", "qwen3-reference", "writer-base"),
        "reader" | "read" => ("reader", "qwen3-0.6b", "qwen3-reference", "reader-base"),
        other => return Err(format!("unknown model/role '{other}' (not found in designs/base-cartridges.json)")),
    };
    Ok(ResolvedSpec {
        short_role: short.to_owned(),
        pop_id: pop.to_owned(),
        asm_id: asm.to_owned(),
        cart_id: cart.to_owned(),
    })
}

/// Format-adaptive raw weight decoder/masker supporting BF16, F16, and F32 tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WeightFormat {
    Bf16,
    F16,
    F32,
}

impl WeightFormat {
    fn parse(dtype: &str) -> Option<Self> {
        match dtype {
            "BF16" => Some(Self::Bf16),
            "F16" | "FP16" => Some(Self::F16),
            "F32" | "FP32" => Some(Self::F32),
            _ => None,
        }
    }

    fn decode_f32(self, raw: &[u8]) -> Vec<f32> {
        match self {
            Self::Bf16 => pmap::decode_bf16(raw),
            Self::F16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| nmengn::half::from16(u16::from_le_bytes(*c)))
                .collect(),
            Self::F32 => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
        }
    }

    fn extract_zero_mask(self, raw: &[u8], rows: usize, cols: usize) -> BitMask {
        let mut bm = BitMask::new(rows, cols, true);
        let total = rows * cols;
        match self {
            Self::Bf16 | Self::F16 => {
                for idx in 0..total {
                    let lo = raw[idx * 2];
                    let hi = raw[idx * 2 + 1];
                    if lo == 0 && (hi & 0x7f) == 0 {
                        bm.set(idx / cols, idx % cols, false);
                    }
                }
            }
            Self::F32 => {
                for idx in 0..total {
                    let off = idx * 4;
                    let bits = u32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                    if (bits & 0x7fff_ffff) == 0 {
                        bm.set(idx / cols, idx % cols, false);
                    }
                }
            }
        }
        bm
    }

    fn apply_mask(self, raw: &[u8], mask: &BitMask) -> Vec<u8> {
        match self {
            Self::Bf16 => pmap::apply_mask_bf16(raw, mask),
            Self::F16 => {
                let mut out = raw.to_vec();
                let total = mask.total();
                for idx in 0..total {
                    if !mask.get(idx / mask.cols, idx % mask.cols) {
                        let off = idx * 2;
                        if let Some(s) = out.get_mut(off..off + 2) {
                            s[0] = 0;
                            s[1] = 0;
                        }
                    }
                }
                out
            }
            Self::F32 => {
                let mut out = raw.to_vec();
                let total = mask.total();
                for idx in 0..total {
                    if !mask.get(idx / mask.cols, idx % mask.cols) {
                        let off = idx * 4;
                        if let Some(s) = out.get_mut(off..off + 4) {
                            s.fill(0);
                        }
                    }
                }
                out
            }
        }
    }
}

/// Graph-discovered residual-writing linear projection (e.g. `o_proj` or `down_proj`, or any
/// equivalent linear projection feeding a residual `add` join in a future MicroModel).
#[derive(Clone, Debug)]
struct ResidualWriter {
    linear_node: String,
    input_node: String,
    input_port: String,
    residual_add_node: String,
}

/// Graph-discovered linear tensor belonging to a specific layer and subblock.
#[derive(Clone, Debug)]
struct LinearEntry {
    tensor_name: String,
    layer: usize,
    is_ffn: bool,
    stream: Option<Stream>,
    writer_idx: Option<usize>,
}

/// Complete dynamically discovered layer/block topology of any MicroModel `Assembly`.
#[derive(Clone, Debug)]
struct ModelTopology {
    n_layers: usize,
    final_node: String,
    writers: Vec<ResidualWriter>,
    /// Per-layer `(attn_writer_idx, ffn_writer_idx)` for summary reporting & drift weighting.
    layer_writers: Vec<(Option<usize>, Option<usize>)>,
    /// Every 2D float linear tensor discovered across all layers.
    linears: Vec<LinearEntry>,
}

/// Discovers the model's exact layer count, residual-writing projections, residual joins, and
/// per-layer linear weight tensors directly from `open.asm` + `open.pop.clusters` (no hardcoded
/// layer counts or Qwen3-only node/tensor names).
fn discover_topology(open: &Model) -> Result<ModelTopology, String> {
    let asm = &open.asm;
    let pop = &open.pop;

    let prefix = format!("{}/", pop.card.id);
    let mut cluster_tensor: BTreeMap<&str, &str> = BTreeMap::new();
    for c in &pop.clusters {
        cluster_tensor.insert(c.id.as_str(), c.tensor.as_str());
        if let Some(stripped) = c.id.strip_prefix(&prefix) {
            cluster_tensor.insert(stripped, c.tensor.as_str());
        }
    }
    let tensor_is_2d_float: BTreeMap<&str, bool> = pop
        .tensors
        .iter()
        .map(|t| (t.name.as_str(), t.shape.len() == 2 && WeightFormat::parse(&t.dtype).is_some()))
        .collect();
    let resolve_cluster_tensor = |cid: &str| -> Option<&str> {
        if let Some(&tn) = cluster_tensor.get(cid) {
            return Some(tn);
        }
        let head = cid.split('/').next().unwrap_or(cid);
        tensor_is_2d_float.get_key_value(head).map(|(&k, _)| k)
    };

    let node_by_id: BTreeMap<&str, &nmengn::assembly::Node> = asm.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
    // Map `(to_node, to_port) -> (from_node, from_port)`
    let mut incoming: BTreeMap<(&str, &str), (&str, &str)> = BTreeMap::new();
    // Map `from_node -> Vec<to_node>`
    let mut outgoing: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for e in &asm.edges {
        incoming.insert((e.to.as_str(), e.to_port.as_str()), (e.from.as_str(), e.from_port.as_str()));
        outgoing.entry(e.from.as_str()).or_default().push(e.to.as_str());
    }

    let max_node_layer = asm.nodes.iter().filter_map(|n| n.layer).max().map_or(0, |m| m + 1);
    let n_layers = pop.dims.layers.max(max_node_layer);

    let is_ffn_node = |n: &nmengn::assembly::Node, tensors: &[String]| -> bool {
        if n.kind == "feedforward" {
            return true;
        }
        let id_low = n.id.to_ascii_lowercase();
        if id_low.contains("mlp") || id_low.contains("ffn") || id_low.contains("gate") || id_low.contains("up") || id_low.contains("down") {
            return true;
        }
        if let Some(outs) = outgoing.get(n.id.as_str())
            && outs.iter().any(|&dst| node_by_id.get(dst).is_some_and(|dn| dn.op == "swigate"))
        {
            return true;
        }
        if let Some(&(src, _)) = incoming.get(&(n.id.as_str(), "x"))
            && node_by_id.get(src).is_some_and(|sn| sn.op == "swigate")
        {
            return true;
        }
        tensors.iter().any(|tn| matches!(Stream::for_tensor(tn), Some((_, Stream::Mlp | Stream::Down))))
    };

    // 1. Discover residual-writing linear nodes (any `linear` node feeding an `add` residual join).
    let mut writers: Vec<ResidualWriter> = Vec::new();
    let mut tensor_to_writer: BTreeMap<String, usize> = BTreeMap::new();
    let mut layer_writers: Vec<(Option<usize>, Option<usize>)> = vec![(None, None); n_layers];

    for node in &asm.nodes {
        if node.op != "linear" {
            continue;
        }
        let Some(layer) = node.layer else { continue };
        if layer >= n_layers {
            continue;
        }
        let Some(outs) = outgoing.get(node.id.as_str()) else { continue };
        let Some(&add_id) = outs.iter().find(|&&dst| node_by_id.get(dst).is_some_and(|dn| dn.op == "add")) else {
            continue;
        };
        let Some(&(in_node, in_port)) = incoming.get(&(node.id.as_str(), "x")) else { continue };
        let mut tensors: Vec<String> = Vec::new();
        for cid in &node.clusters {
            if let Some(tname) = resolve_cluster_tensor(cid)
                && tensor_is_2d_float.get(tname).copied().unwrap_or(false)
                && !tensors.iter().any(|x| x == tname)
            {
                tensors.push(tname.to_owned());
            }
        }
        if tensors.is_empty() {
            continue;
        }
        let is_ffn = is_ffn_node(node, &tensors);
        let w_idx = writers.len();
        for tn in &tensors {
            tensor_to_writer.insert(tn.clone(), w_idx);
        }
        if is_ffn {
            if layer_writers[layer].1.is_none() {
                layer_writers[layer].1 = Some(w_idx);
            }
        } else if layer_writers[layer].0.is_none() {
            layer_writers[layer].0 = Some(w_idx);
        }
        writers.push(ResidualWriter {
            linear_node: node.id.clone(),
            input_node: in_node.to_owned(),
            input_port: in_port.to_owned(),
            residual_add_node: add_id.to_owned(),
        });
    }

    // 2. Discover EVERY 2D float linear tensor across all layers from the assembly graph.
    let mut seen_tensors: BTreeSet<String> = BTreeSet::new();
    let mut linears: Vec<LinearEntry> = Vec::new();
    for node in &asm.nodes {
        if node.op != "linear" {
            continue;
        }
        let Some(layer) = node.layer else { continue };
        if layer >= n_layers {
            continue;
        }
        let mut tensors: Vec<String> = Vec::new();
        for cid in &node.clusters {
            if let Some(tname) = resolve_cluster_tensor(cid)
                && tensor_is_2d_float.get(tname).copied().unwrap_or(false)
                && seen_tensors.insert(tname.to_owned())
            {
                tensors.push(tname.to_owned());
            }
        }
        let is_ffn = is_ffn_node(node, &tensors);
        for tn in tensors {
            let stream = Stream::for_tensor(&tn).map(|(_, s)| s);
            let writer_idx = tensor_to_writer.get(&tn).copied();
            linears.push(LinearEntry { tensor_name: tn, layer, is_ffn, stream, writer_idx });
        }
    }

    // Fallback: include any 2D float tensor recognized by `Stream::for_tensor` not already linked.
    for t in &pop.tensors {
        if let Some((l, s)) = Stream::for_tensor(&t.name)
            && l < n_layers
            && t.shape.len() == 2
            && WeightFormat::parse(&t.dtype).is_some()
            && seen_tensors.insert(t.name.clone())
        {
            let is_ffn = matches!(s, Stream::Mlp | Stream::Down);
            let writer_idx = tensor_to_writer.get(&t.name).copied();
            linears.push(LinearEntry { tensor_name: t.name.clone(), layer: l, is_ffn, stream: Some(s), writer_idx });
        }
    }

    let final_node = open
        .role_node("final_norm")
        .or_else(|_| {
            asm.nodes
                .iter()
                .rev()
                .find(|n| n.op == "norm" || n.op == "add")
                .map(|n| n.id.clone())
                .ok_or_else(|| nmtype::Fault::new("nmlab.surgery", nmtype::FaultKind::Interface, "no final norm node"))
        })
        .map_err(|e| e.to_string())?;

    Ok(ModelTopology { n_layers, final_node, writers, layer_writers, linears })
}

fn default_prompts(role: &str) -> Vec<String> {
    match role {
        "embedder" => vec![
            "Instruct: Given a codebase question, retrieve the source lines that answer it\nQuery:How does MCP Tool Control block unselected tools when Dynamic tool activation is off?<|endoftext|>".to_owned(),
            "Instruct: Given a codebase question, retrieve the source lines that answer it\nQuery:Which function writes the dead-skin vector store segments and revives vectors on undo?<|endoftext|>".to_owned(),
            "crates/nmapi/src/mcp.rs\npub fn tool_allowed(limit: usize, tool_control_on: bool, dynamic_on: bool, custom_enabled: Option<&[String]>, tool_name: &str) -> bool<|endoftext|>".to_owned(),
            "crates/nmsubs/src/dense_store.rs\npub fn shed(&mut self, live: &BTreeSet<&str>, live_file_tags: Option<&BTreeSet<String>>, now_ms: u64) -> ShedReport<|endoftext|>".to_owned(),
        ],
        "reranker" => vec![
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: Given a codebase question, retrieve the source lines that answer it\n<Query>: How does MCP Tool Control block unselected tools when Dynamic tool activation is off?\n<Document>: crates/nmapi/src/mcp.rs:179-238\npub fn tool_allowed(limit: usize, tool_control_on: bool, dynamic_on: bool, custom_enabled: Option<&[String]>, tool_name: &str) -> bool { if !tool_control_on || dynamic_on { return true; } }<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".to_owned(),
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: Given a codebase question, retrieve the source lines that answer it\n<Query>: What is the exact VRAM coexistence free-memory threshold in writer_idle.rs?\n<Document>: crates/nmapi/src/writer_idle.rs:701-720\nconst COEXIST_FREE_MB: usize = 1_500; let free_mb = nmengn::residency::get().memory_now().map_or(0, |(free, _)| free >> 20);<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".to_owned(),
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: Given a codebase question, retrieve the source lines that answer it\n<Query>: Which key is used to encrypt the memory journal on disk?\n<Document>: crates/nmmem/src/journal.rs:1-40\n// The journal chains lines with SHA-256 hashes only; no encryption key is used.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".to_owned(),
        ],
        "writer" => vec![
            "<|im_start|>user\ncrates/nmapi/src/writer_idle.rs lines 701-720:\n```\nconst COEXIST_FREE_MB: usize = 1_500;\nlet released = if free_mb >= COEXIST_FREE_MB { Vec::new() } else { finders };\n```\nWrite one line of at most 25 words, plain text, no markdown: first what this code or text does, then \"Defines:\" and the names it defines.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".to_owned(),
            "<|im_start|>user\nIn at most two short sentences of plain words, on one line, say what crates/nmsubs/src/cache.rs does and when someone working on the project would need to read it.\n\n```\npub fn load(root: &Path, skip: &[String], embed: &Embed, cache_path: &Path) -> Option<(Substrate, String)>\n```<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".to_owned(),
        ],
        _ => vec![
            "Repository neuromorph-nano: crates/nmapi/src/mcp.rs implements MCP tools/list, tool_allowed(), and trigger_dynamic_tools_with().\nQuestion: how does MCP Tool Control block unselected tools when dynamic activation is off?".to_owned(),
            "Repository neuromorph-nano: crates/nmsubs/src/dense_store.rs stores 512-row NMDENSE1 segments and skin.json hash barriers for undo revival.\nQuestion: how does the dead-skin vector store revive vectors on undo without GPU re-embedding?".to_owned(),
            "Repository neuromorph-nano: crates/nmapi/src/writer_idle.rs keeps reader and reranker resident when free VRAM >= 1500 MiB.\nQuestion: when does writer_idle skip evicting reader and reranker?".to_owned(),
        ],
    }
}

fn load_prompts(role: &str, file: Option<&str>) -> Result<Vec<String>, String> {
    let Some(path) = file else {
        return Ok(default_prompts(role));
    };
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut out: Vec<String> = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if !t.is_empty() && !t.starts_with('#') {
            out.push(t.replace("\\n", "\n"));
        }
    }
    if out.is_empty() {
        return Err(format!("{path}: no non-empty prompts"));
    }
    Ok(out)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockFilter {
    All,
    Attn,
    Ffn,
}

impl BlockFilter {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "all" => Ok(Self::All),
            "attn" | "attention" => Ok(Self::Attn),
            "ffn" | "mlp" => Ok(Self::Ffn),
            other => Err(format!("invalid --blocks '{other}' (use all, attn, or ffn)")),
        }
    }

    fn matches_ffn(self, is_ffn: bool) -> bool {
        match self {
            Self::All => true,
            Self::Attn => !is_ffn,
            Self::Ffn => is_ffn,
        }
    }
}

fn parse_layers(spec: &str, n_layers: usize) -> Result<(BTreeSet<usize>, bool), String> {
    if spec == "all" {
        return Ok(((0..n_layers).collect(), false));
    }
    if spec == "auto" {
        return Ok(((0..n_layers).collect(), true));
    }
    let mut set = BTreeSet::new();
    for part in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if let Some((a, b)) = part.split_once('-') {
            let lo: usize = a.trim().parse().map_err(|_| format!("invalid layer '{a}'"))?;
            let hi: usize = b.trim().parse().map_err(|_| format!("invalid layer '{b}'"))?;
            if lo > hi || hi >= n_layers {
                return Err(format!("layer range {lo}-{hi} outside 0..{}", n_layers.saturating_sub(1)));
            }
            for l in lo..=hi {
                set.insert(l);
            }
        } else {
            let l: usize = part.parse().map_err(|_| format!("invalid layer '{part}'"))?;
            if l >= n_layers {
                return Err(format!("layer {l} outside 0..{}", n_layers.saturating_sub(1)));
            }
            set.insert(l);
        }
    }
    if set.is_empty() {
        return Err(format!("empty --layers spec '{spec}'"));
    }
    Ok((set, false))
}

fn read_range(path: &Path, start: u64, end: u64) -> Result<Vec<u8>, String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    f.seek(SeekFrom::Start(start)).map_err(|e| format!("{}: {e}", path.display()))?;
    let len = usize::try_from(end.saturating_sub(start)).map_err(|e| e.to_string())?;
    let mut buf = vec![0_u8; len];
    f.read_exact(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(buf)
}

/// Captured signed output residual `E = Y_open - Y_cand` (`[tokens * rows]`) and candidate input
/// `X_cand` (`[tokens * cols]`) for one residual-writing linear projection, enabling exact
/// analytical first-order Taylor L2 error-reduction scoring.
struct SignedSignal {
    tokens: usize,
    rows: usize,
    cols: usize,
    /// `Y_open - Y_cand`, concatenated across prompts (`tokens * rows`).
    err: Vec<f32>,
    /// `X_cand`, concatenated across prompts (`tokens * cols`).
    x_cand: Vec<f32>,
}

#[derive(Clone, Debug)]
struct LayerProbe {
    layer: usize,
    attn_sp: f64,
    ffn_sp: f64,
    o_drift: f64,
    down_drift: f64,
    res1_drift: f64,
    res2_drift: f64,
    jump: f64,
    o_row_err: Vec<f32>,
    down_row_err: Vec<f32>,
}

struct ProbeResult {
    layers: Vec<LayerProbe>,
    final_cos: f64,
    final_max_diff: f32,
    probe_ms: f64,
    vram_bytes: usize,
    /// Exact signed `(E, X_cand)` signals indexed by `writer_idx` in `ModelTopology::writers`.
    signed_signals: Vec<SignedSignal>,
}

struct OpenProbeRef {
    open_targets: Vec<(String, String)>,
    cand_targets: Vec<(String, String)>,
    encoded: Vec<Vec<i32>>,
    /// Per prompt: `[w0.y, w0.res, w1.y, w1.res, ..., final_norm.y]`.
    open_outs: Vec<Vec<Vec<f32>>>,
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (&x, &y) in a.iter().zip(b) {
        let d = f64::from(x) - f64::from(y);
        num += d * d;
        den += f64::from(x) * f64::from(x);
    }
    num.sqrt() / den.sqrt().max(1e-12)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += f64::from(x) * f64::from(y);
        na += f64::from(x) * f64::from(x);
        nb += f64::from(y) * f64::from(y);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-12)
}

fn row_err_accum(a: &[f32], b: &[f32], hidden: usize, acc: &mut [f32]) {
    if hidden == 0 || acc.len() != hidden {
        return;
    }
    let tokens = a.len().min(b.len()) / hidden;
    for t in 0..tokens {
        let off = t * hidden;
        for c in 0..hidden {
            acc[c] += (a[off + c] - b[off + c]).abs();
        }
    }
}

/// Loads the active single-copy base cartridge's exact `ParamMask` AND pre-held `base_parts`
/// directly from the `.cartridge` file in parallel across CPU cores, adapting to `BF16`, `F16`,
/// or `F32` tensors and any dynamically discovered layer/block layout.
#[expect(clippy::type_complexity, reason = "internal cartridge loader tuple")]
fn active_cartridge_mask(
    home: &Path,
    spec: &ResolvedSpec,
    topo: &ModelTopology,
    open: &Model,
) -> Result<(ParamMask, u64, String, Vec<TensorDesc>, Vec<Arc<[u8]>>), String> {
    let lib = nmlearn::library::Library::open(home).map_err(|e| e.to_string())?;
    let entry = lib
        .cartridge(&spec.cart_id)
        .ok_or_else(|| format!("no active base cartridge '{}' in the cartridge library", spec.cart_id))?;
    let file_name = entry.files.first().map(|f| f.name.as_str()).ok_or("cartridge entry has no files")?;
    let cart_path = lib.cartridge_dir(entry).join(file_name);
    let offer = nmapi::neural::offer_of(&cart_path).map_err(|e| e.to_string())?;

    let known_linears: BTreeSet<&str> = topo.linears.iter().map(|e| e.tensor_name.as_str()).collect();
    let mut targets: Vec<(&TensorDesc, WeightFormat)> = Vec::new();
    let mut mask = ParamMask::default();
    for td in &offer.parts {
        let Some(fmt) = WeightFormat::parse(&td.dtype) else { continue };
        if td.shape.len() == 2 && known_linears.contains(td.name.as_str()) {
            if offer.open.contains_key(&td.name) {
                mask.tensors.insert(td.name.clone(), BitMask::new(td.shape[0], td.shape[1], true));
            } else {
                targets.push((td, fmt));
            }
        }
    }
    let decoded = nmtype::budget::run(targets.len(), |i| -> Result<(String, BitMask, TensorDesc, Arc<[u8]>), String> {
        let (td, fmt) = targets[i];
        let raw = read_range(Path::new(&td.file), td.start, td.end)?;
        let bm = fmt.extract_zero_mask(&raw, td.shape[0], td.shape[1]);
        let arc: Arc<[u8]> = Arc::from(raw);
        nmnpcs::provision::hold(&td.sha256, &arc);
        let held_desc = TensorDesc {
            file: format!("(surgery:base:{})", td.name),
            start: 0,
            end: arc.len() as u64,
            ..td.clone()
        };
        Ok((td.name.clone(), bm, held_desc, arc))
    });
    let mut held_by_name: BTreeMap<String, TensorDesc> = BTreeMap::new();
    let mut base_arcs: Vec<Arc<[u8]>> = Vec::with_capacity(targets.len());
    for item in decoded.into_iter().flatten() {
        let (name, bm, desc, arc) = item?;
        mask.tensors.insert(name.clone(), bm);
        held_by_name.insert(name, desc);
        base_arcs.push(arc);
    }
    if mask.tensors.is_empty() {
        return Err(format!("could not extract any linear masks from {}", cart_path.display()));
    }
    let mut base_parts = Vec::with_capacity(open.pop.tensors.len());
    for t in &open.pop.tensors {
        if let Some(desc) = held_by_name.remove(&t.name) {
            base_parts.push(desc);
        } else {
            base_parts.push(t.clone());
        }
    }
    Ok((mask, entry.version, entry.key(), base_parts, base_arcs))
}

fn model_from_parts(open: &Model, parts: Vec<TensorDesc>, tag: &str, q8: bool) -> Result<Model, String> {
    let gpu = open.pop.gpu.is_some();
    let pop = Population::open_cartridge(&open.pop.home, &open.pop.card.id, gpu, tag, parts).map_err(|e| e.to_string())?;
    let mut asm = open.asm.clone();
    if q8 {
        for node in &mut asm.nodes {
            if node.op == "linear" {
                node.precision = "q8".to_owned();
            }
        }
    }
    Model::bind(&pop, asm).map_err(|e| e.to_string())
}

/// Fast incremental candidate builder: ONLY re-masks tensors in `touched`, reusing `base_parts`
/// verbatim for every untouched layer/matrix, and adapting to `BF16`/`F16`/`F32`.
type BuiltModel = (Model, Vec<Arc<[u8]>>);
type BuiltPart = (TensorDesc, Option<Arc<[u8]>>);

fn masked_model_delta(
    open: &Model,
    base_parts: &[TensorDesc],
    mask: &ParamMask,
    touched: &BTreeSet<String>,
    tag: &str,
    q8: bool,
) -> Result<BuiltModel, String> {
    let open_by_name: BTreeMap<&str, &TensorDesc> = open.pop.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let open_dir = &open.pop.card.dir;
    let built = nmtype::budget::run(base_parts.len(), |i| -> Result<BuiltPart, String> {
        let base_t = &base_parts[i];
        if !touched.contains(&base_t.name) {
            return Ok((base_t.clone(), None));
        }
        let Some(bm) = mask.tensors.get(&base_t.name) else {
            return Ok((base_t.clone(), None));
        };
        let open_t = open_by_name.get(base_t.name.as_str()).copied().unwrap_or(base_t);
        let fmt = WeightFormat::parse(&open_t.dtype).ok_or_else(|| format!("unsupported dtype {}", open_t.dtype))?;
        let raw = read_range(&open_dir.join(&open_t.file), open_t.start, open_t.end)?;
        let masked = fmt.apply_mask(&raw, bm);
        let sha = nmhash::sha256_bytes(&masked);
        let arc: Arc<[u8]> = Arc::from(masked);
        nmnpcs::provision::hold(&sha, &arc);
        let desc = TensorDesc {
            file: format!("(surgery:{tag}:{})", open_t.name),
            start: 0,
            end: arc.len() as u64,
            sha256: sha,
            ..open_t.clone()
        };
        Ok((desc, Some(arc)))
    });
    let mut parts = Vec::with_capacity(base_parts.len());
    let mut held_arcs = Vec::new();
    for item in built.into_iter().flatten() {
        let (desc, arc) = item?;
        parts.push(desc);
        if let Some(a) = arc {
            held_arcs.push(a);
        }
    }
    let m = model_from_parts(open, parts, tag, q8)?;
    Ok((m, held_arcs))
}

fn masked_model(open: &Model, mask: &ParamMask, tag: &str, q8: bool) -> Result<BuiltModel, String> {
    let touched: BTreeSet<String> = mask.tensors.keys().cloned().collect();
    masked_model_delta(open, &open.pop.tensors, mask, &touched, tag, q8)
}

fn layer_sparsities(mask: &ParamMask, topo: &ModelTopology) -> Vec<(f64, f64)> {
    let n_layers = topo.n_layers;
    let mut attn_rem = vec![0_usize; n_layers];
    let mut attn_tot = vec![0_usize; n_layers];
    let mut ffn_rem = vec![0_usize; n_layers];
    let mut ffn_tot = vec![0_usize; n_layers];
    for entry in &topo.linears {
        let Some(bm) = mask.tensors.get(&entry.tensor_name) else { continue };
        let tot = bm.total();
        let rem = bm.removed_count();
        if entry.is_ffn {
            ffn_rem[entry.layer] += rem;
            ffn_tot[entry.layer] += tot;
        } else {
            attn_rem[entry.layer] += rem;
            attn_tot[entry.layer] += tot;
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "percentage display")]
    (0..n_layers)
        .map(|l| {
            let a = if attn_tot[l] > 0 { 100.0 * attn_rem[l] as f64 / attn_tot[l] as f64 } else { 0.0 };
            let f = if ffn_tot[l] > 0 { 100.0 * ffn_rem[l] as f64 / ffn_tot[l] as f64 } else { 0.0 };
            (a, f)
        })
        .collect()
}

/// Single-pass full-depth reference capture on the untouched `open` model across all graph-discovered
/// residual writers + `final_node`. Reused across BEFORE, PASS 1, and AFTER probes so `open` is
/// evaluated strictly ONCE.
fn run_open_ref(
    open: &Model,
    topo: &ModelTopology,
    prompts: &[String],
    dev: Device,
    record_open: bool,
) -> Result<(OpenProbeRef, Option<ParamRecord>), String> {
    let mut open_targets: Vec<(String, String)> = Vec::with_capacity(topo.writers.len() * 2 + 1);
    let mut cand_targets: Vec<(String, String)> = Vec::with_capacity(topo.writers.len() * 3 + 1);
    for w in &topo.writers {
        open_targets.push((w.linear_node.clone(), "y".to_owned()));
        open_targets.push((w.residual_add_node.clone(), "y".to_owned()));
        cand_targets.push((w.linear_node.clone(), "y".to_owned()));
        cand_targets.push((w.residual_add_node.clone(), "y".to_owned()));
        cand_targets.push((w.input_node.clone(), w.input_port.clone()));
    }
    open_targets.push((topo.final_node.clone(), "y".to_owned()));
    cand_targets.push((topo.final_node.clone(), "y".to_owned()));

    let encoded: Vec<Vec<i32>> = prompts
        .iter()
        .map(|p| open.pop.tokenizer.encode(p).map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;

    if record_open {
        pmap::start_recording(&open.pop.card.id, None);
    }
    let mut open_outs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(encoded.len());
    {
        let _guard = nmengn::chanrec::job("surgery");
        for ids in &encoded {
            let r = open.forward(dev, ids, &open_targets, &RunOpts::default()).map_err(|e| e.to_string())?;
            let mut row = Vec::with_capacity(open_targets.len());
            for key in &open_targets {
                let act = r.get(key).ok_or_else(|| format!("missing tap {}:{}", key.0, key.1))?;
                row.push(nmengn::model::values(act).map_err(|e| e.to_string())?.to_vec());
            }
            open_outs.push(row);
        }
    }
    let rec = if record_open { pmap::stop_recording() } else { None };
    open.pop.release_device();
    Ok((OpenProbeRef { open_targets, cand_targets, encoded, open_outs }, rec))
}

/// Single-pass full-depth fMRI probe comparing a candidate cartridge against a cached `OpenProbeRef`
/// while simultaneously capturing exact signed `(Y_open - Y_cand, X_cand)` signals for every
/// graph-discovered residual-writing linear projection.
fn probe_cand_against_ref(
    open_ref: &OpenProbeRef,
    topo: &ModelTopology,
    cand: &Model,
    cand_mask: &ParamMask,
    dev: Device,
) -> Result<ProbeResult, String> {
    let t0 = Instant::now();
    let n_layers = topo.n_layers;
    let hidden = cand.pop.dims.hidden;
    let n_writers = topo.writers.len();

    let mut cand_outs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(open_ref.encoded.len());
    for ids in &open_ref.encoded {
        let r = cand.forward(dev, ids, &open_ref.cand_targets, &RunOpts::default()).map_err(|e| e.to_string())?;
        let mut row = Vec::with_capacity(open_ref.cand_targets.len());
        for key in &open_ref.cand_targets {
            let act = r.get(key).ok_or_else(|| format!("missing tap {}:{}", key.0, key.1))?;
            row.push(nmengn::model::values(act).map_err(|e| e.to_string())?.to_vec());
        }
        cand_outs.push(row);
    }
    let vram_bytes = cand.pop.device_bytes();
    cand.pop.release_device();

    // Build exact signed `(E = Y_open - Y_cand, X_cand)` signals + per-writer drifts.
    let n_prompts = open_ref.encoded.len().max(1);
    #[expect(clippy::cast_precision_loss, reason = "prompt average")]
    let inv_p = 1.0 / n_prompts as f64;
    let mut writer_y_drift = vec![0.0_f64; n_writers];
    let mut writer_res_drift = vec![0.0_f64; n_writers];
    let mut writer_row_err: Vec<Vec<f32>> = (0..n_writers).map(|_| vec![0.0_f32; hidden]).collect();
    let mut signed_signals: Vec<SignedSignal> = Vec::with_capacity(n_writers);

    for wi in 0..n_writers {
        let mut err_all: Vec<f32> = Vec::new();
        let mut x_all: Vec<f32> = Vec::new();
        let mut total_tokens: usize = 0;
        let mut rows_dim: usize = 0;
        let mut cols_dim: usize = 0;
        let o_y_idx = wi * 2;
        let o_r_idx = wi * 2 + 1;
        let c_y_idx = wi * 3;
        let c_r_idx = wi * 3 + 1;
        let c_x_idx = wi * 3 + 2;
        for (p_i, (oa, ca)) in open_ref.open_outs.iter().zip(&cand_outs).enumerate() {
            let oy = &oa[o_y_idx];
            let cy = &ca[c_y_idx];
            let cx = &ca[c_x_idx];
            writer_y_drift[wi] += rel_l2(oy, cy);
            writer_res_drift[wi] += rel_l2(&oa[o_r_idx], &ca[c_r_idx]);
            row_err_accum(oy, cy, hidden, &mut writer_row_err[wi]);
            let t_cnt = open_ref.encoded[p_i].len().max(1);
            let r_d = oy.len() / t_cnt;
            let c_d = cx.len() / t_cnt;
            if r_d > 0 && c_d > 0 && oy.len() == cy.len() && oy.len() == t_cnt * r_d && cx.len() == t_cnt * c_d {
                rows_dim = r_d;
                cols_dim = c_d;
                total_tokens += t_cnt;
                err_all.reserve(oy.len());
                for (&yo, &yc) in oy.iter().zip(cy) {
                    err_all.push(yo - yc);
                }
                x_all.extend_from_slice(cx);
            }
        }
        writer_y_drift[wi] *= inv_p;
        writer_res_drift[wi] *= inv_p;
        signed_signals.push(SignedSignal {
            tokens: total_tokens,
            rows: rows_dim,
            cols: cols_dim,
            err: err_all,
            x_cand: x_all,
        });
    }

    let sps = layer_sparsities(cand_mask, topo);
    let mut layer_probes = Vec::with_capacity(n_layers);
    let mut prev_res2 = 0.0_f64;
    for (l, &(attn_sp, ffn_sp)) in sps.iter().enumerate().take(n_layers) {
        let (aw, fw) = topo.layer_writers[l];
        let o_d = aw.map_or(0.0, |i| writer_y_drift[i]);
        let r1_d = aw.map_or(0.0, |i| writer_res_drift[i]);
        let dn_d = fw.map_or(0.0, |i| writer_y_drift[i]);
        let r2_d = fw.map_or(r1_d, |i| writer_res_drift[i]);
        let o_err = aw.map_or_else(|| vec![0.0_f32; hidden], |i| writer_row_err[i].clone());
        let dn_err = fw.map_or_else(|| vec![0.0_f32; hidden], |i| writer_row_err[i].clone());
        layer_probes.push(LayerProbe {
            layer: l,
            attn_sp,
            ffn_sp,
            o_drift: o_d,
            down_drift: dn_d,
            res1_drift: r1_d,
            res2_drift: r2_d,
            jump: (r2_d - prev_res2).max(0.0),
            o_row_err: o_err,
            down_row_err: dn_err,
        });
        prev_res2 = r2_d;
    }

    let o_fin_idx = open_ref.open_targets.len() - 1;
    let c_fin_idx = open_ref.cand_targets.len() - 1;
    let mut fin_cos = 0.0_f64;
    let mut fin_max = 0.0_f32;
    for (oa, ca) in open_ref.open_outs.iter().zip(&cand_outs) {
        fin_cos += cosine(&oa[o_fin_idx], &ca[c_fin_idx]);
        let d = oa[o_fin_idx]
            .iter()
            .zip(&ca[c_fin_idx])
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        fin_max = fin_max.max(d);
    }
    Ok(ProbeResult {
        layers: layer_probes,
        final_cos: fin_cos * inv_p,
        final_max_diff: fin_max,
        probe_ms: t0.elapsed().as_secs_f64() * 1000.0,
        vram_bytes,
        signed_signals,
    })
}

fn format_probe_table(p: &ProbeResult) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "  Lyr | AttnSp%  FFNSp% | o_drift%  dn_drift% | res1_d%  res2_d%  jump%");
    let _ = writeln!(out, "  ----+-----------------+--------------------+------------------------");
    for lp in &p.layers {
        let _ = writeln!(
            out,
            "  L{:02} | {:6.2}% {:6.2}% |  {:6.2}%    {:6.2}% | {:6.2}%  {:6.2}% {:+6.2}%",
            lp.layer,
            lp.attn_sp,
            lp.ffn_sp,
            lp.o_drift * 100.0,
            lp.down_drift * 100.0,
            lp.res1_drift * 100.0,
            lp.res2_drift * 100.0,
            lp.jump * 100.0,
        );
    }
    let _ = writeln!(
        out,
        "  final_norm cosine: {:.6} | max_diff: {:.4} | {}-layer probe time: {:.0} ms",
        p.final_cos,
        p.final_max_diff,
        p.layers.len(),
        p.probe_ms
    );
    out
}

/// Exact analytical first-order Taylor L2 error-reduction surgical editor for residual-writing
/// projections (`o_proj`, `down_proj`, or any graph-discovered residual writer).
///
/// For `is_add = true`: selects up to `k` pruned weights with strictly positive exact L2 gain:
///   `gain[r,c] = W_open[r,c] * (E^T X_cand)[r,c] - 0.5 * W_open[r,c]^2 * ||X_cand[:,c]||_2^2`
/// Never restores weights that move `Y_cand` away from `Y_open`.
///
/// For `is_add = false`: removes up to `k` kept, non-pinned weights with the smallest exact L2
/// removal cost `cost[r,c] = 0.5 * W[r,c]^2 * ||X_cand[:,c]||_2^2 + W[r,c] * (E^T X_cand)[r,c]`.
///
/// Returns `(changed_count, top_restored_coords)` where `top_restored_coords` captures up to 64
/// highest-gain super-weight coordinates for `super-weights.json`.
#[expect(clippy::too_many_arguments, reason = "precision matrix editor")]
fn edit_matrix_signed_taylor(
    weights: &[f32],
    rows: usize,
    cols: usize,
    sig: &SignedSignal,
    pin_rows: &BTreeSet<usize>,
    pin_cols: &BTreeSet<usize>,
    pinned_coords: Option<&BTreeSet<(usize, usize)>>,
    bm: &mut BitMask,
    k: usize,
    is_add: bool,
) -> (usize, Vec<(usize, usize, f32)>) {
    if k == 0 || weights.len() < rows * cols || sig.tokens == 0 || sig.rows != rows || sig.cols != cols {
        return (0, Vec::new());
    }
    // Precompute `x_sq[c] = sum_t X_cand[t, c]^2`.
    let mut x_sq = vec![0.0_f32; cols];
    for t in 0..sig.tokens {
        let x_t = &sig.x_cand[t * cols..(t + 1) * cols];
        for c in 0..cols {
            let v = x_t[c];
            x_sq[c] += v * v;
        }
    }

    let mut cands: Vec<(f32, u32)> = Vec::new();
    let mut g_row = vec![0.0_f32; cols];
    for r in 0..rows {
        let row_pinned = !is_add && pin_rows.contains(&r);
        if row_pinned {
            continue;
        }
        g_row.fill(0.0);
        for t in 0..sig.tokens {
            let er = sig.err[t * rows + r];
            if er == 0.0 {
                continue;
            }
            let x_t = &sig.x_cand[t * cols..(t + 1) * cols];
            for c in 0..cols {
                g_row[c] += er * x_t[c];
            }
        }
        let r_off = r * cols;
        for c in 0..cols {
            let kept = bm.get(r, c);
            if is_add {
                if kept {
                    continue;
                }
                let w = weights[r_off + c];
                if w == 0.0 {
                    continue;
                }
                let gain = w * g_row[c] - 0.5 * w * w * x_sq[c];
                if gain > 0.0 {
                    #[expect(clippy::cast_possible_truncation, reason = "matrix index fits u32")]
                    cands.push((gain, (r_off + c) as u32));
                }
            } else {
                if !kept || pin_cols.contains(&c) {
                    continue;
                }
                if pinned_coords.is_some_and(|s| s.contains(&(r, c))) {
                    continue;
                }
                let w = weights[r_off + c];
                let cost = 0.5 * w * w * x_sq[c] + w * g_row[c];
                #[expect(clippy::cast_possible_truncation, reason = "matrix index fits u32")]
                cands.push((cost, (r_off + c) as u32));
            }
        }
    }
    if cands.is_empty() {
        return (0, Vec::new());
    }
    let pick = k.min(cands.len());
    if pick < cands.len() {
        if is_add {
            cands.select_nth_unstable_by(pick, |a, b| b.0.total_cmp(&a.0));
        } else {
            cands.select_nth_unstable_by(pick, |a, b| a.0.total_cmp(&b.0));
        }
        cands.truncate(pick);
    }
    let mut top_coords: Vec<(usize, usize, f32)> = Vec::new();
    if is_add {
        cands.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        for &(score, idx) in cands.iter().take(64) {
            let i = idx as usize;
            top_coords.push((i / cols, i % cols, score));
        }
    }
    for &(_, idx) in &cands {
        let i = idx as usize;
        bm.set(i / cols, i % cols, is_add);
    }
    (cands.len(), top_coords)
}

/// Fast O(N) activation-norm + error-weighted surgical editor (balanced sign, zero overshoot,
/// ~20 ms per matrix across CPU threads). Returns `(changed_count, top_restored_coords)` where
/// `top_restored_coords` records up to 64 highest-impact restored coordinates for `super-weights.json`.
#[expect(clippy::too_many_arguments, reason = "fast matrix surgical editor")]
fn edit_matrix_wanda(
    weights: &[f32],
    rows: usize,
    cols: usize,
    col_norms: &[f32],
    row_err: Option<&[f32]>,
    pin_rows: &BTreeSet<usize>,
    pin_cols: &BTreeSet<usize>,
    pinned_coords: Option<&BTreeSet<(usize, usize)>>,
    bm: &mut BitMask,
    k: usize,
    is_add: bool,
) -> (usize, Vec<(usize, usize, f32)>) {
    if k == 0 || weights.len() < rows * cols {
        return (0, Vec::new());
    }
    let err_scale: Vec<f32> = match row_err {
        Some(re) if re.len() == rows => {
            let mean = re.iter().copied().sum::<f32>() / (rows.max(1) as f32);
            let inv = if mean > 1e-6 { 1.0 / mean } else { 0.0 };
            re.iter().map(|&v| 1.0 + v * inv).collect()
        }
        _ => vec![1.0_f32; rows],
    };
    let mut cands: Vec<(f32, u32)> = Vec::new();
    for (r, &r_mult) in err_scale.iter().enumerate().take(rows) {
        let row_pinned = !is_add && pin_rows.contains(&r);
        let r_off = r * cols;
        for c in 0..cols {
            let kept = bm.get(r, c);
            if is_add {
                if !kept {
                    let w = weights[r_off + c].abs();
                    if w > 0.0 {
                        let cn = col_norms.get(c).copied().unwrap_or(1.0);
                        #[expect(clippy::cast_possible_truncation, reason = "matrix index fits u32")]
                        cands.push((w * cn * r_mult, (r_off + c) as u32));
                    }
                }
            } else if kept && !row_pinned && !pin_cols.contains(&c) && !pinned_coords.is_some_and(|s| s.contains(&(r, c))) {
                let w = weights[r_off + c].abs();
                let cn = col_norms.get(c).copied().unwrap_or(1.0);
                #[expect(clippy::cast_possible_truncation, reason = "matrix index fits u32")]
                cands.push((w * cn * r_mult, (r_off + c) as u32));
            }
        }
    }
    if cands.is_empty() {
        return (0, Vec::new());
    }
    let pick = k.min(cands.len());
    if pick < cands.len() {
        if is_add {
            cands.select_nth_unstable_by(pick, |a, b| b.0.total_cmp(&a.0));
        } else {
            cands.select_nth_unstable_by(pick, |a, b| a.0.total_cmp(&b.0));
        }
        cands.truncate(pick);
    }
    let mut top_coords: Vec<(usize, usize, f32)> = Vec::new();
    if is_add {
        cands.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        for &(score, idx) in cands.iter().take(64) {
            let i = idx as usize;
            top_coords.push((i / cols, i % cols, score));
        }
    }
    for &(_, idx) in &cands {
        let i = idx as usize;
        bm.set(i / cols, i % cols, is_add);
    }
    (cands.len(), top_coords)
}

/// Loads protected super-weight coordinates from `library/parameter-maps/<role>/super-weights.json`
/// so no future `remove` or `swap` can accidentally cut critical anchor weights.
fn load_super_weights(home: &Path, role: &str) -> BTreeMap<String, BTreeSet<(usize, usize)>> {
    let path = home.join(format!("library/parameter-maps/{role}/super-weights.json"));
    let mut out: BTreeMap<String, BTreeSet<(usize, usize)>> = BTreeMap::new();
    let Ok(text) = std::fs::read_to_string(&path) else { return out };
    let Ok(Value::Object(root)) = nmjson::parse(text.as_bytes()) else { return out };
    let Some(Value::Object(tensors)) = root.get("tensors") else { return out };
    for (tname, val) in tensors {
        if let Value::Array(arr) = val {
            let entry = out.entry(tname.clone()).or_default();
            for pair in arr {
                if let Value::Array(rc) = pair
                    && rc.len() >= 2
                    && let (Some(r), Some(c)) = (
                        rc[0].integer().ok().and_then(|x| usize::try_from(x).ok()),
                        rc[1].integer().ok().and_then(|x| usize::try_from(x).ok()),
                    )
                {
                    entry.insert((r, c));
                }
            }
        }
    }
    out
}

fn save_super_weights(home: &Path, role: &str, coords: &BTreeMap<String, BTreeSet<(usize, usize)>>) {
    if coords.is_empty() {
        return;
    }
    let path = home.join(format!("library/parameter-maps/{role}/super-weights.json"));
    if let Some(p) = path.parent() {
        let _ = std::fs::create_dir_all(p);
    }
    let mut s = String::from("{\n  \"tensors\": {\n");
    for (ti, (tname, set)) in coords.iter().enumerate() {
        let _ = write!(s, "    \"{tname}\": [");
        for (ci, &(r, c)) in set.iter().enumerate() {
            let comma = if ci + 1 < set.len() { "," } else { "" };
            let _ = write!(s, "[{r},{c}]{comma}");
        }
        let comma = if ti + 1 < coords.len() { "," } else { "" };
        let _ = writeln!(s, "]{comma}");
    }
    s.push_str("  }\n}\n");
    if std::fs::write(&path, s).is_ok()
        && let Some(dir) = path.parent()
    {
        // 2026-11-26: lineage as it is written (nmapi::lineage).
        let _ = nmapi::lineage::record_made(dir, "super-weights.json", "", "nano-lab run pmap surgery <op> --write", "base-build");
    }
}

#[expect(clippy::too_many_arguments, reason = "internal cartridge writer helper")]
fn write_edited_cartridge(
    home: &Path,
    spec: &ResolvedSpec,
    open: &Model,
    asm_bf16: &Assembly,
    new_mask: &ParamMask,
    parent_key: &str,
    next_version: u64,
    note: &str,
) -> Result<(String, f32, f32, usize), String> {
    let saved_sw = (
        nmengn::gpu::HALF_MATH.load(std::sync::atomic::Ordering::Relaxed),
        nmengn::gpu::SEGMENT_ATTENTION.load(std::sync::atomic::Ordering::Relaxed),
    );
    let mut largest_diff: f32 = 0.0;
    let mut bmq8_vram: usize = 0;
    let is_writer = spec.short_role == "writer";
    let precisions: &[bool] = if is_writer { &[true] } else { &[false, true] };
    for served in [false, true] {
        nmengn::gpu::HALF_MATH.store(served, std::sync::atomic::Ordering::Relaxed);
        nmengn::gpu::SEGMENT_ATTENTION.store(served, std::sync::atomic::Ordering::Relaxed);
        for &q8 in precisions {
            let (open_m, _h0) = masked_model(open, &ParamMask::default(), "surg-open-chk", q8)?;
            let a = nmengn::base::check_outputs(&open_m).map_err(|e| e.to_string())?;
            drop(open_m);
            let (cand_m, _h1) = masked_model(open, new_mask, "surg-cand-chk", q8)?;
            let b = nmengn::base::check_outputs(&cand_m).map_err(|e| e.to_string())?;
            if served && q8 {
                bmq8_vram = cand_m.pop.device_bytes();
            }
            drop(cand_m);
            let d = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (f32::from_bits(*x) - f32::from_bits(*y)).abs())
                .fold(0.0_f32, f32::max);
            largest_diff = largest_diff.max(d);
        }
    }
    if is_writer {
        nmengn::gpu::HALF_MATH.store(false, std::sync::atomic::Ordering::Relaxed);
        nmengn::gpu::SEGMENT_ATTENTION.store(false, std::sync::atomic::Ordering::Relaxed);
        let cpu_pop = Population::open(home, &spec.pop_id, false, false).map_err(|e| e.to_string())?;
        let cpu_open = Model::bind(&cpu_pop, asm_bf16.clone()).map_err(|e| e.to_string())?;
        let a = nmengn::base::check_outputs(&cpu_open).map_err(|e| e.to_string())?;
        let (cpu_cand, _hc) = masked_model(&cpu_open, new_mask, "surg-cpu-chk", false)?;
        let b = nmengn::base::check_outputs(&cpu_cand).map_err(|e| e.to_string())?;
        let d = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (f32::from_bits(*x) - f32::from_bits(*y)).abs())
            .fold(0.0_f32, f32::max);
        largest_diff = largest_diff.max(d);
    }
    nmengn::gpu::HALF_MATH.store(saved_sw.0, std::sync::atomic::Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(saved_sw.1, std::sync::atomic::Ordering::Relaxed);
    let check_bound = (largest_diff * 1.25).max(largest_diff + 1.0);

    let needed = nmengn::base::read_by(asm_bf16);
    let open_dir = &open.pop.card.dir;
    let mut tensors = Vec::with_capacity(needed.len());
    let mut masked_n = 0_usize;
    let mut removed_w = 0_usize;
    let mut total_w = 0_usize;
    for name in &needed {
        let (layer, block) = nmengn::base::address(name).ok_or_else(|| format!("{name}: no layer and block"))?;
        let td = open.pop.tensors.iter().find(|t| &t.name == name).ok_or_else(|| format!("missing tensor {name}"))?;
        let raw = read_range(&open_dir.join(&td.file), td.start, td.end)?;
        let bytes = if let Some(bm) = new_mask.tensors.get(name) {
            let fmt = WeightFormat::parse(&td.dtype).ok_or_else(|| format!("unsupported dtype {}", td.dtype))?;
            masked_n += 1;
            removed_w += bm.removed_count();
            total_w += bm.total();
            fmt.apply_mask(&raw, bm)
        } else {
            raw
        };
        tensors.push(nmlearn::library::Tensor {
            name: name.clone(),
            layer,
            block: block.to_owned(),
            dtype: td.dtype.clone(),
            shape: td.shape.clone(),
            bytes,
        });
    }
    #[expect(clippy::cast_precision_loss, reason = "display percentage")]
    let actual_sp = 100.0 * removed_w as f64 / total_w.max(1) as f64;
    let layers_max = open.pop.dims.layers.saturating_sub(1) as u64;
    let weights_sha = open.pop.card.sha_main.clone();

    let mut lib = nmlearn::library::Library::open(home).map_err(|e| e.to_string())?;
    let domain = match spec.short_role.as_str() {
        "reranker" => "job:judge-code",
        "writer" => "job:write",
        _ => "job:base",
    };
    let cspec = nmlearn::library::CartridgeSpec {
        id: spec.cart_id.clone(),
        version: next_version,
        family: String::new(),
        domain: domain.to_owned(),
        scope: "general".to_owned(),
        model: spec.pop_id.clone(),
        weights: weights_sha,
        layers: Some((0, layers_max)),
        name: format!("{}, surgically refined base cartridge", spec.short_role),
        description: format!(
            "Precision layer-by-layer surgically refined base cartridge ({actual_sp:.2}% sparse across {masked_n} linear matrices) for {}; auto-compresses on GPU with bmq8.cu.",
            spec.pop_id
        ),
        made_from: format!("{parent_key} + open weights via `nano-lab run pmap surgery` ({note})"),
        effect: format!("Surgically refined ({actual_sp:.2}% sparse, {note})"),
    };
    let mut notes = BTreeMap::new();
    notes.insert("content".to_owned(), Value::String("base".to_owned()));
    notes.insert("exact".to_owned(), Value::Bool(false));
    notes.insert("check_bound".to_owned(), Value::Number(format!("{check_bound:e}")));
    notes.insert("parent".to_owned(), Value::String(parent_key.to_owned()));
    notes.insert("mapped".to_owned(), Value::String(format!("yes: precision-surgery ({note})")));
    notes.insert("pruned".to_owned(), Value::String(format!("yes: {actual_sp:.2}% zeroed across {masked_n} linear matrices (bmq8 on GPU)")));
    notes.insert("edits".to_owned(), Value::String(note.to_owned()));
    notes.insert("measured".to_owned(), Value::String("pending full board benchmark".to_owned()));
    let filter = nmlearn::library::Filter::Tensors(tensors);
    let (key, _removed) = if crate::KEEP_BESIDE.load(std::sync::atomic::Ordering::Relaxed) {
        lib.add_base_cartridge(&cspec, &filter, Value::Object(notes), Vec::new(), false)
    } else {
        lib.replace_base_cartridge(&cspec, &filter, Value::Object(notes), Vec::new(), false)
    }
    .map_err(|e| e.to_string())?;
    lib.save().map_err(|e| e.to_string())?;
    Ok((key, largest_diff, check_bound, bmq8_vram))
}

fn save_probe_json(path: &Path, role: &str, key: &str, p: &ProbeResult) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut s = String::new();
    let _ = writeln!(s, "{{");
    let _ = writeln!(s, "  \"role\": \"{role}\",");
    let _ = writeln!(s, "  \"cartridge\": \"{key}\",");
    let _ = writeln!(s, "  \"overlay\": \"{}\",", crate::overlay_line().trim_start_matches("overlay: "));
    let _ = writeln!(s, "  \"n_layers\": {},", p.layers.len());
    let _ = writeln!(s, "  \"final_cos\": {:.6},", p.final_cos);
    let _ = writeln!(s, "  \"final_max_diff\": {:.4},", p.final_max_diff);
    let _ = writeln!(s, "  \"probe_ms\": {:.1},", p.probe_ms);
    let _ = writeln!(s, "  \"layers\": [");
    for (i, lp) in p.layers.iter().enumerate() {
        let comma = if i + 1 < p.layers.len() { "," } else { "" };
        let _ = writeln!(
            s,
            "    {{\"layer\":{},\"attn_sp\":{:.2},\"ffn_sp\":{:.2},\"o_drift\":{:.5},\"down_drift\":{:.5},\"res1_drift\":{:.5},\"res2_drift\":{:.5},\"jump\":{:.5}}}{comma}",
            lp.layer, lp.attn_sp, lp.ffn_sp, lp.o_drift, lp.down_drift, lp.res1_drift, lp.res2_drift, lp.jump
        );
    }
    let _ = writeln!(s, "  ]");
    let _ = writeln!(s, "}}");
    let _ = std::fs::write(path, s);
}

#[expect(clippy::too_many_arguments, reason = "ledger record")]
fn append_ledger(
    home: &Path,
    role: &str,
    op: &str,
    from_key: &str,
    to_version: u64,
    layers_touched: usize,
    weights_edited: usize,
    before: &ProbeResult,
    after: &ProbeResult,
) {
    let dir = home.join(format!("library/parameter-maps/{role}"));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("surgery-ledger.jsonl");
    let line = format!(
        "{{\"ts_ms\":{},\"role\":\"{role}\",\"op\":\"{op}\",\"from\":\"{from_key}\",\"to_version\":{to_version},\"layers_touched\":{layers_touched},\"weights_edited\":{weights_edited},\"cos_before\":{:.6},\"cos_after\":{:.6},\"max_diff_before\":{:.4},\"max_diff_after\":{:.4}}}\n",
        nmtype::now_ms(),
        before.final_cos,
        after.final_cos,
        before.final_max_diff,
        after.final_max_diff,
    );
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        use std::io::Write as _;
        let _ = f.write_all(line.as_bytes());
    }
}

fn flag_val<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.windows(2).find(|w| w[0] == key).map(|w| w[1].as_str())
}

/// Parses a per-layer/per-block surgical spec like `"0:8,2.ffn:4,16-25:6,27.attn:15"` into
/// `(layer, is_ffn) -> pct` deltas.
fn parse_precision_spec(spec: &str, n_layers: usize) -> Result<BTreeMap<(usize, bool), f64>, String> {
    let mut out: BTreeMap<(usize, bool), f64> = BTreeMap::new();
    for part in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let Some((lhs, rhs)) = part.split_once(':') else {
            return Err(format!("invalid precision spec entry '{part}' (expected '<layers>[.<attn|ffn>]:<pct>')"));
        };
        let pct: f64 = rhs.trim().parse().map_err(|_| format!("invalid pct '{rhs}' in '{part}'"))?;
        let (layer_part, blk) = match lhs.split_once('.') {
            Some((l, b)) => (l.trim(), BlockFilter::parse(b.trim())?),
            None => (lhs.trim(), BlockFilter::All),
        };
        let (layers, _) = parse_layers(layer_part, n_layers)?;
        for l in layers {
            if blk.matches_ffn(false) {
                *out.entry((l, false)).or_insert(0.0) += pct;
            }
            if blk.matches_ffn(true) {
                *out.entry((l, true)).or_insert(0.0) += pct;
            }
        }
    }
    Ok(out)
}

struct EditJob {
    tensor_name: String,
    layer: usize,
    is_ffn: bool,
    stream: Option<Stream>,
    writer_idx: Option<usize>,
    k: usize,
    is_add: bool,
}

type EditSummary = (usize, usize, BTreeSet<usize>, BTreeSet<String>, BTreeMap<String, BTreeSet<(usize, usize)>>, f64);
type JobEditOut = (String, BitMask, usize, usize, bool, Vec<(usize, usize, f32)>);

#[expect(clippy::too_many_arguments, reason = "internal parallel edit executor")]
fn execute_edit_jobs(
    jobs: &[EditJob],
    pop: &Population,
    probe: &ProbeResult,
    js: Option<&pmap::JobStats>,
    pinned_heads: &BTreeMap<usize, Vec<usize>>,
    super_weights: &BTreeMap<String, BTreeSet<(usize, usize)>>,
    use_taylor: bool,
    cur_mask: &mut ParamMask,
) -> Result<EditSummary, String> {
    let t_edit = Instant::now();
    let head_dim = pop.dims.head_dim;
    let open_dir = &pop.card.dir;
    let tensor_by_name: BTreeMap<&str, &TensorDesc> = pop.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let cur_mask_ref = &*cur_mask;

    let edited = nmtype::budget::run(
        jobs.len(),
        |i| -> Result<JobEditOut, String> {
            let job = &jobs[i];
            let t = tensor_by_name
                .get(job.tensor_name.as_str())
                .copied()
                .ok_or_else(|| format!("missing tensor {}", job.tensor_name))?;
            let fmt = WeightFormat::parse(&t.dtype).ok_or_else(|| format!("unsupported dtype {}", t.dtype))?;
            let mut bm = cur_mask_ref
                .tensors
                .get(&t.name)
                .cloned()
                .ok_or_else(|| format!("missing mask {}", t.name))?;
            let raw = read_range(&open_dir.join(&t.file), t.start, t.end)?;
            let weights = fmt.decode_f32(&raw);
            let (pin_r, pin_c) = pmap::pinned_axes_for_tensor(&t.name, head_dim, pinned_heads);
            let p_coords = super_weights.get(&t.name);

            if use_taylor
                && let Some(wi) = job.writer_idx
                && let Some(sig) = probe.signed_signals.get(wi)
                && sig.tokens > 0
                && sig.rows == t.shape[0]
                && sig.cols == t.shape[1]
            {
                let (changed, top_sw) = edit_matrix_signed_taylor(
                    &weights, t.shape[0], t.shape[1], sig, &pin_r, &pin_c, p_coords, &mut bm, job.k, job.is_add,
                );
                return Ok((t.name.clone(), bm, changed, job.layer, job.is_add, top_sw));
            }

            let norms = job
                .stream
                .and_then(|st| js.and_then(|j| j.layers.get(&job.layer)).and_then(|m| m.get(&st)))
                .map_or_else(|| vec![1.0_f32; t.shape[1]], |ss| ss.l2_norms());
            let row_err = if job.is_ffn && probe.layers[job.layer].down_row_err.len() == t.shape[0] {
                Some(probe.layers[job.layer].down_row_err.as_slice())
            } else if !job.is_ffn && probe.layers[job.layer].o_row_err.len() == t.shape[0] {
                Some(probe.layers[job.layer].o_row_err.as_slice())
            } else {
                None
            };
            let (changed, top_sw) = edit_matrix_wanda(
                &weights, t.shape[0], t.shape[1], &norms, row_err, &pin_r, &pin_c, p_coords, &mut bm, job.k, job.is_add,
            );
            Ok((t.name.clone(), bm, changed, job.layer, job.is_add, top_sw))
        },
    );

    let mut total_added: usize = 0;
    let mut total_removed: usize = 0;
    let mut touched_layers: BTreeSet<usize> = BTreeSet::new();
    let mut touched_names: BTreeSet<String> = BTreeSet::new();
    let mut new_super: BTreeMap<String, BTreeSet<(usize, usize)>> = BTreeMap::new();
    for item in edited.into_iter().flatten() {
        let (name, bm, changed, l, is_add, top_sw) = item?;
        if changed > 0 {
            if is_add {
                total_added += changed;
            } else {
                total_removed += changed;
            }
            touched_layers.insert(l);
            touched_names.insert(name.clone());
            if !top_sw.is_empty() && probe.layers[l].jump >= 0.02 {
                let entry = new_super.entry(name.clone()).or_default();
                for (r, c, _) in top_sw {
                    entry.insert((r, c));
                }
            }
        }
        cur_mask.tensors.insert(name, bm);
    }
    let edit_ms = t_edit.elapsed().as_secs_f64() * 1000.0;
    Ok((total_added, total_removed, touched_layers, touched_names, new_super, edit_ms))
}

#[expect(clippy::too_many_arguments, reason = "internal drift-weighted job builder")]
fn build_drift_weighted_jobs(
    topo: &ModelTopology,
    pop: &Population,
    cur_mask: &ParamMask,
    probe: &ProbeResult,
    selected_layers: &BTreeSet<usize>,
    blocks: BlockFilter,
    budget_pct: f64,
    is_add: bool,
    jump_boost: f64,
    auto_weight: bool,
) -> Vec<EditJob> {
    let tensor_by_name: BTreeMap<&str, &TensorDesc> = pop.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let mut sub_weights: BTreeMap<(usize, bool), f64> = BTreeMap::new();
    for &l in selected_layers {
        let lp = &probe.layers[l];
        let j_bonus = lp.jump * jump_boost;
        let aw = if is_add {
            (lp.o_drift + j_bonus * 0.5).max(if auto_weight { 0.0 } else { 1e-4 })
        } else {
            1.0 / (lp.o_drift + j_bonus + 0.01)
        };
        let fw = if is_add {
            (lp.down_drift + j_bonus).max(if auto_weight { 0.0 } else { 1e-4 })
        } else {
            1.0 / (lp.down_drift + j_bonus + 0.01)
        };
        if blocks.matches_ffn(false) {
            sub_weights.insert((l, false), aw);
        }
        if blocks.matches_ffn(true) {
            sub_weights.insert((l, true), fw);
        }
    }
    let w_sum: f64 = sub_weights.values().sum::<f64>().max(1e-12);

    let mut selected_total_w: usize = 0;
    let mut bucket_tot: BTreeMap<(usize, bool), usize> = BTreeMap::new();
    let mut elig: Vec<(&LinearEntry, usize)> = Vec::new();
    for entry in &topo.linears {
        if !selected_layers.contains(&entry.layer) || !blocks.matches_ffn(entry.is_ffn) {
            continue;
        }
        if !cur_mask.tensors.contains_key(&entry.tensor_name) {
            continue;
        }
        let Some(td) = tensor_by_name.get(entry.tensor_name.as_str()) else { continue };
        let tw = td.shape[0] * td.shape[1];
        selected_total_w += tw;
        *bucket_tot.entry((entry.layer, entry.is_ffn)).or_insert(0) += tw;
        elig.push((entry, tw));
    }
    #[expect(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "budget count")]
    let total_budget_w = ((selected_total_w as f64) * (budget_pct / 100.0)).round() as usize;

    let mut jobs = Vec::new();
    for (entry, tw) in elig {
        let bw = sub_weights.get(&(entry.layer, entry.is_ffn)).copied().unwrap_or(0.0);
        let btot = bucket_tot.get(&(entry.layer, entry.is_ffn)).copied().unwrap_or(1).max(1);
        #[expect(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "per-tensor budget")]
        let k = ((total_budget_w as f64) * (bw / w_sum) * (tw as f64 / btot as f64)).round() as usize;
        if k > 0 {
            jobs.push(EditJob {
                tensor_name: entry.tensor_name.clone(),
                layer: entry.layer,
                is_ffn: entry.is_ffn,
                stream: entry.stream,
                writer_idx: entry.writer_idx,
                k,
                is_add,
            });
        }
    }
    jobs
}

/// The job filter file a `job-filter --check-file` run measures (set by `run` for its job-filter path).
static CHECK_FILE: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

pub fn run(args: &[String], home: &Path) -> Result<String, String> {
    if let Ok(mut g) = CHECK_FILE.lock() {
        *g = flag_val(args, "--check-file").map(str::to_owned);
    }
    let t_total = Instant::now();
    let op = args.get(3).map(String::as_str).unwrap_or("probe");
    let role_arg = args.get(4).map(String::as_str).unwrap_or("reranker");
    let use_gpu = !args.iter().any(|a| a == "--cpu");
    let use_taylor = args.iter().any(|a| a == "--taylor");
    let write_out = args.iter().any(|a| a == "--write");
    let split_gate = args.iter().any(|a| a == "--split-gate");
    let prompts_file = flag_val(args, "--prompts");
    let check_prompts_file = flag_val(args, "--check-prompts");
    let job_name = flag_val(args, "--job").unwrap_or("default");
    let ridge_lambda: f64 = flag_val(args, "--lambda").and_then(|v| v.parse::<f64>().ok()).unwrap_or(5.0);
    let layers_spec = flag_val(args, "--layers").unwrap_or("all");
    let blocks_spec = flag_val(args, "--blocks").unwrap_or("all");
    let budget_pct: f64 = flag_val(args, "--budget").and_then(|v| v.parse::<f64>().ok()).unwrap_or(if op == "cascade" { 3.0 } else { 1.0 });
    let anchor_budget_pct: f64 = flag_val(args, "--anchor-budget").and_then(|v| v.parse::<f64>().ok()).unwrap_or(6.0);
    let spec_arg = flag_val(args, "--spec");
    let add_spec_arg = flag_val(args, "--add");
    let rem_spec_arg = flag_val(args, "--rem");

    let spec = resolve_model_spec(home, role_arg)?;
    let all_prompts = load_prompts(&spec.short_role, prompts_file)?;
    let (prompts, check_prompts): (Vec<String>, Vec<String>) = if let Some(cf) = check_prompts_file {
        (all_prompts, load_prompts(&spec.short_role, Some(cf))?)
    } else if split_gate && all_prompts.len() >= 4 {
        let mut fit = Vec::new();
        let mut chk = Vec::new();
        for (i, p) in all_prompts.into_iter().enumerate() {
            if i % 2 == 0 {
                fit.push(p);
            } else {
                chk.push(p);
            }
        }
        (fit, chk)
    } else {
        (all_prompts, Vec::new())
    };
    let blocks = BlockFilter::parse(blocks_spec)?;

    let pop = Population::open(home, &spec.pop_id, false, use_gpu).map_err(|e| e.to_string())?;
    let asm_bf16 = Assembly::load(&home.join("designs/assemblies"), &spec.asm_id).map_err(|e| e.to_string())?;
    let mut asm_q8 = asm_bf16.clone();
    for node in &mut asm_q8.nodes {
        if node.op == "linear" {
            node.precision = "q8".to_owned();
        }
    }
    let open = Model::bind(&pop, asm_q8).map_err(|e| e.to_string())?;
    let dev = if use_gpu && pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let topo = discover_topology(&open)?;
    let n_layers = topo.n_layers;

    let (mut cur_mask, cur_ver, cur_key, base_parts, base_arcs) = active_cartridge_mask(home, &spec, &topo, &open)?;
    if op == "job-filter" {
        return run_job_filter(
            home,
            &spec,
            &open,
            &topo,
            &cur_key,
            base_parts,
            &prompts,
            &check_prompts,
            layers_spec,
            job_name,
            ridge_lambda,
            dev,
            write_out,
            t_total,
        );
    }
    let (open_ref, rec_opt) = run_open_ref(&open, &topo, &prompts, dev, op != "probe")?;
    let (check_ref_opt, check_before_opt) = if !check_prompts.is_empty() {
        let (cr, _) = run_open_ref(&open, &topo, &check_prompts, dev, false)?;
        let cb_model = model_from_parts(&open, base_parts.clone(), "surg-chk-before", true)?;
        let cb = probe_cand_against_ref(&cr, &topo, &cb_model, &cur_mask, dev)?;
        drop(cb_model);
        (Some(cr), Some(cb))
    } else {
        (None, None)
    };
    let cand_before = model_from_parts(&open, base_parts.clone(), "surg-before", true)?;
    let probe_before = probe_cand_against_ref(&open_ref, &topo, &cand_before, &cur_mask, dev)?;
    let before_vram = probe_before.vram_bytes;
    drop(cand_before);

    let probe_path = home.join(format!("library/parameter-maps/{}/surgery-probe.json", spec.short_role));
    save_probe_json(&probe_path, &spec.short_role, &cur_key, &probe_before);

    let mut out = String::new();
    let _ = writeln!(
        out,
        "=== Precision {n_layers}-Layer Surgery ({op}): {} ({cur_key}, {:?}, {} prompts, {} writers) ===",
        spec.short_role,
        dev,
        prompts.len(),
        topo.writers.len()
    );
    let _ = writeln!(out, "BEFORE:");
    out.push_str(&format_probe_table(&probe_before));

    if op == "probe" {
        let _ = writeln!(out, "probe json: {}", probe_path.display());
        let _ = writeln!(out, "elapsed   : {:.2} s", t_total.elapsed().as_secs_f64());
        return Ok(out);
    }

    let js = rec_opt.as_ref().and_then(|r| r.jobs.get("surgery"));
    let mut pinned_heads: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    if spec.short_role == "reader" {
        let lo = (n_layers * 3) / 7;
        let hi = (n_layers * 5) / 7;
        for l in lo..hi {
            pinned_heads.insert(l, vec![0, 1]);
        }
    }
    let mut super_weights = load_super_weights(home, &spec.short_role);
    let tensor_by_name: BTreeMap<&str, &TensorDesc> = pop.tensors.iter().map(|t| (t.name.as_str(), t)).collect();

    let mut total_added: usize = 0;
    let mut total_removed: usize = 0;
    let mut touched_layers: BTreeSet<usize> = BTreeSet::new();
    let mut touched_names: BTreeSet<String> = BTreeSet::new();
    let mut total_edit_ms: f64 = 0.0;
    let summary_desc: String;

    if op == "cascade" {
        // PASS 1 (Anchor Pass): heal cliff layers (`jump >= 1.5%`, e.g., L00, L02, L27) first so
        // downstream layers receive clean activations before Pass 2 scores them!
        let mut anchor_layers: BTreeSet<usize> = probe_before
            .layers
            .iter()
            .filter(|lp| lp.jump >= 0.015)
            .map(|lp| lp.layer)
            .collect();
        if anchor_layers.is_empty() {
            anchor_layers.insert(0);
            if n_layers > 2 {
                anchor_layers.insert(2);
            }
            anchor_layers.insert(n_layers.saturating_sub(1));
        }
        let pass1_jobs = build_drift_weighted_jobs(
            &topo,
            &pop,
            &cur_mask,
            &probe_before,
            &anchor_layers,
            blocks,
            anchor_budget_pct,
            true,
            4.0,
            false,
        );
        let (a1, r1, tl1, tn1, sw1, ms1) =
            execute_edit_jobs(&pass1_jobs, &pop, &probe_before, js, &pinned_heads, &super_weights, use_taylor, &mut cur_mask)?;
        total_added += a1;
        total_removed += r1;
        touched_layers.extend(tl1);
        touched_names.extend(tn1);
        total_edit_ms += ms1;
        for (k, v) in sw1 {
            super_weights.entry(k).or_default().extend(v);
        }

        // Re-probe candidate after Anchor Pass (zero extra open-model passes!).
        let (cand_mid, _hm) = masked_model_delta(&open, &base_parts, &cur_mask, &touched_names, "surg-mid", true)?;
        let probe_mid = probe_cand_against_ref(&open_ref, &topo, &cand_mid, &cur_mask, dev)?;
        drop(cand_mid);

        let anchor_list: Vec<String> = anchor_layers.iter().map(|l| format!("L{l:02}")).collect();
        let _ = writeln!(
            out,
            "\nPASS 1 (ANCHOR on {}, budget {:.2}%): +{a1} weights in {:.0} ms | final_cos {:.6} -> {:.6} ({:+.6}) | max_diff {:.4} -> {:.4}",
            anchor_list.join(","),
            anchor_budget_pct,
            ms1,
            probe_before.final_cos,
            probe_mid.final_cos,
            probe_mid.final_cos - probe_before.final_cos,
            probe_before.final_max_diff,
            probe_mid.final_max_diff
        );

        // PASS 2 (Clean Semantic + Decision Cascade across ALL layers on clean post-anchor signal).
        let all_layers: BTreeSet<usize> = (0..n_layers).collect();
        let pass2_jobs = build_drift_weighted_jobs(
            &topo,
            &pop,
            &cur_mask,
            &probe_mid,
            &all_layers,
            blocks,
            budget_pct,
            true,
            5.0,
            false,
        );
        let (a2, r2, tl2, tn2, sw2, ms2) =
            execute_edit_jobs(&pass2_jobs, &pop, &probe_mid, js, &pinned_heads, &super_weights, use_taylor, &mut cur_mask)?;
        total_added += a2;
        total_removed += r2;
        touched_layers.extend(tl2);
        touched_names.extend(tn2);
        total_edit_ms += ms2;
        for (k, v) in sw2 {
            super_weights.entry(k).or_default().extend(v);
        }
        summary_desc = format!("cascade anchors={} anchor_budget={anchor_budget_pct:.2}% budget={budget_pct:.2}%", anchor_list.join(","));
    } else {
        let mut jobs: Vec<EditJob> = Vec::new();
        match op {
            "swap" => {
                let add_map = match add_spec_arg {
                    Some(s) => parse_precision_spec(s, n_layers)?,
                    None => BTreeMap::new(),
                };
                let rem_map = match rem_spec_arg {
                    Some(s) => parse_precision_spec(s, n_layers)?,
                    None => BTreeMap::new(),
                };
                if add_map.is_empty() && rem_map.is_empty() {
                    return Err("swap requires --add \"<spec>\" and/or --rem \"<spec>\" (e.g. --add \"0:8,2.ffn:4,27:12\" --rem \"3-15:4\")".to_owned());
                }
                summary_desc = format!(
                    "swap --add \"{}\" --rem \"{}\"",
                    add_spec_arg.unwrap_or(""),
                    rem_spec_arg.unwrap_or("")
                );
                for entry in &topo.linears {
                    if !cur_mask.tensors.contains_key(&entry.tensor_name) {
                        continue;
                    }
                    let Some(td) = tensor_by_name.get(entry.tensor_name.as_str()) else { continue };
                    let add_p = add_map.get(&(entry.layer, entry.is_ffn)).copied().unwrap_or(0.0);
                    let rem_p = rem_map.get(&(entry.layer, entry.is_ffn)).copied().unwrap_or(0.0);
                    let net = add_p - rem_p;
                    if net.abs() < 1e-6 {
                        continue;
                    }
                    #[expect(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "per-tensor budget")]
                    let k = (((td.shape[0] * td.shape[1]) as f64) * (net.abs() / 100.0)).round() as usize;
                    if k > 0 {
                        jobs.push(EditJob {
                            tensor_name: entry.tensor_name.clone(),
                            layer: entry.layer,
                            is_ffn: entry.is_ffn,
                            stream: entry.stream,
                            writer_idx: entry.writer_idx,
                            k,
                            is_add: net > 0.0,
                        });
                    }
                }
            }
            "add" | "remove" => {
                let is_add = op == "add";
                if let Some(sp) = spec_arg {
                    let map = parse_precision_spec(sp, n_layers)?;
                    summary_desc = format!("{op} --spec \"{sp}\"");
                    for entry in &topo.linears {
                        if !cur_mask.tensors.contains_key(&entry.tensor_name) {
                            continue;
                        }
                        let Some(td) = tensor_by_name.get(entry.tensor_name.as_str()) else { continue };
                        let pct = map.get(&(entry.layer, entry.is_ffn)).copied().unwrap_or(0.0);
                        if pct <= 0.0 {
                            continue;
                        }
                        #[expect(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "per-tensor budget")]
                        let k = (((td.shape[0] * td.shape[1]) as f64) * (pct / 100.0)).round() as usize;
                        if k > 0 {
                            jobs.push(EditJob {
                                tensor_name: entry.tensor_name.clone(),
                                layer: entry.layer,
                                is_ffn: entry.is_ffn,
                                stream: entry.stream,
                                writer_idx: entry.writer_idx,
                                k,
                                is_add,
                            });
                        }
                    }
                } else {
                    let (selected_layers, auto_weight) = parse_layers(layers_spec, n_layers)?;
                    summary_desc = format!("{op} --layers {layers_spec} --blocks {blocks_spec} --budget {budget_pct:.2}%");
                    jobs = build_drift_weighted_jobs(
                        &topo,
                        &pop,
                        &cur_mask,
                        &probe_before,
                        &selected_layers,
                        blocks,
                        budget_pct,
                        is_add,
                        if auto_weight { 4.0 } else { 0.0 },
                        auto_weight,
                    );
                }
            }
            other => {
                return Err(format!(
                    "usage: nano-lab run pmap surgery <probe|add|remove|swap|cascade> <role|model> [--layers all|auto|0-N] [--blocks all|attn|ffn] [--budget pct] [--anchor-budget pct] [--spec \"...\"] [--add \"...\"] [--rem \"...\"] [--prompts file] [--gpu|--cpu] [--write], got {other}"
                ));
            }
        }
        let (a, r, tl, tn, sw, ms) =
            execute_edit_jobs(&jobs, &pop, &probe_before, js, &pinned_heads, &super_weights, use_taylor, &mut cur_mask)?;
        total_added = a;
        total_removed = r;
        touched_layers = tl;
        touched_names = tn;
        total_edit_ms = ms;
        for (k, v) in sw {
            super_weights.entry(k).or_default().extend(v);
        }
    }

    let total_changed = total_added + total_removed;
    let (cand_after, ha) = masked_model_delta(&open, &base_parts, &cur_mask, &touched_names, "surg-after", true)?;
    let probe_after = probe_cand_against_ref(&open_ref, &topo, &cand_after, &cur_mask, dev)?;
    let check_after_opt = match &check_ref_opt {
        Some(cr) => Some(probe_cand_against_ref(cr, &topo, &cand_after, &cur_mask, dev)?),
        None => None,
    };
    let after_vram = probe_after.vram_bytes;
    drop(cand_after);
    // Release multi-GB temporary weight buffers and cached reference activations before serializing
    // large cartridges (e.g. 1.7B+ models) to disk.
    drop(ha);
    drop(base_arcs);
    drop(base_parts);
    drop(open_ref);
    drop(check_ref_opt);
    drop(rec_opt);

    let _ = writeln!(out, "\nSURGERY APPLIED ({summary_desc}):");
    let mode_lbl = if use_taylor { "signed Taylor + parallel O(N)" } else { "balanced Wanda+err + parallel O(N)" };
    let _ = writeln!(
        out,
        "  added +{total_added} / removed -{total_removed} weights ({total_changed} total) across {} of {} layers in {:.0} ms ({mode_lbl})",
        touched_layers.len(),
        n_layers,
        total_edit_ms
    );
    let _ = writeln!(out, "\nAFTER:");
    out.push_str(&format_probe_table(&probe_after));
    #[expect(clippy::cast_precision_loss, reason = "display delta")]
    let vram_delta_mb = (after_vram as f64 - before_vram as f64) / (1024.0 * 1024.0);
    let _ = writeln!(
        out,
        "DELTA: final_cos {:.6} -> {:.6} ({:+.6}) | max_diff {:.4} -> {:.4} ({:+.4}) | GPU Q8 VRAM {:+.2} MiB",
        probe_before.final_cos,
        probe_after.final_cos,
        probe_after.final_cos - probe_before.final_cos,
        probe_before.final_max_diff,
        probe_after.final_max_diff,
        probe_after.final_max_diff - probe_before.final_max_diff,
        vram_delta_mb,
    );

    let mut check_gate_passed = true;
    if let (Some(cb), Some(ca)) = (&check_before_opt, &check_after_opt) {
        let cos_d = ca.final_cos - cb.final_cos;
        let md_ratio = if cb.final_max_diff > 1e-6 {
            ca.final_max_diff / cb.final_max_diff
        } else {
            1.0
        };
        check_gate_passed = cos_d >= -0.0005 && md_ratio <= 1.05;
        let _ = writeln!(
            out,
            "HELD-OUT CHECK ({} prompts): final_cos {:.6} -> {:.6} ({:+.6}) | max_diff {:.4} -> {:.4} ({:+.4}) | gate: {}",
            check_prompts.len(),
            cb.final_cos,
            ca.final_cos,
            cos_d,
            cb.final_max_diff,
            ca.final_max_diff,
            ca.final_max_diff - cb.final_max_diff,
            if check_gate_passed { "PASS" } else { "REFUSED (held-out regression)" }
        );
    }

    if write_out {
        if !check_gate_passed {
            return Err(format!("{out}\nREFUSED WRITE: held-out check split regressed; base cartridge left untouched."));
        }
        save_super_weights(home, &spec.short_role, &super_weights);
        let next_ver = cur_ver + 1;
        let next_key = format!("{}@{next_ver}", spec.cart_id);
        save_probe_json(&probe_path, &spec.short_role, &next_key, &probe_after);
        let note = format!("{summary_desc} added=+{total_added} removed=-{total_removed}");
        let (path, largest_diff, check_bound, bmq8_vram) =
            write_edited_cartridge(home, &spec, &open, &asm_bf16, &cur_mask, &cur_key, next_ver, &note)?;
        append_ledger(home, &spec.short_role, op, &cur_key, next_ver, touched_layers.len(), total_changed, &probe_before, &probe_after);
        #[expect(clippy::cast_precision_loss, reason = "display MiB")]
        let mib = bmq8_vram as f64 / (1024.0 * 1024.0);
        let _ = writeln!(
            out,
            "WRITTEN: {path} (v{next_ver}, max_diff={largest_diff:.4}, check_bound={check_bound:.4}, GPU Q8={mib:.1} MiB)"
        );
    } else {
        let _ = writeln!(out, "DRY RUN: pass --write to save the new single-copy version in place and log to surgery-ledger.jsonl.");
    }
    let _ = writeln!(out, "elapsed: {:.2} s", t_total.elapsed().as_secs_f64());
    Ok(out)
}

/// WP-A2 Phase 1: builds and held-out verifies a lightweight FFN-only per-channel `gain-bias`
/// job-path filter (`L{l}.swigate`, `nmengn-sockets/1`) against the untouched open reference,
/// applying it purely through per-hop `RunOpts::filters` with ZERO base cartridge weight edits.
#[expect(clippy::too_many_arguments, reason = "internal surgery job-filter command handler")]
#[expect(clippy::too_many_lines, reason = "closed-form fit + held-out RunOpts::filters verification + json export")]
fn run_job_filter(
    home: &Path,
    spec: &ResolvedSpec,
    open: &Model,
    topo: &ModelTopology,
    cur_key: &str,
    base_parts: Vec<TensorDesc>,
    fit_prompts: &[String],
    check_prompts: &[String],
    layers_spec: &str,
    job_name: &str,
    ridge_lambda: f64,
    dev: Device,
    write_out: bool,
    t_total: Instant,
) -> Result<String, String> {
    let n_layers = topo.n_layers;
    let (selected_layers, _) = parse_layers(layers_spec, n_layers)?;
    let cand = model_from_parts(open, base_parts, "job-filter-cand", true)?;
    let fin_id = open.role_node("final_norm").map_err(|e| e.to_string())?;
    let fin_target = (fin_id.clone(), "y".to_owned());

    // Discover FFN `swigate` nodes in the selected layers dynamically (adapts to any layer count).
    let mut swi_targets: Vec<(usize, String, String)> = Vec::new();
    for &l in &selected_layers {
        let short = format!("L{l}.swigate");
        let nid = nmengn::assembly::nid(&short);
        if open.asm.node(&nid).is_some() {
            swi_targets.push((l, short, nid));
        }
    }
    if swi_targets.is_empty() {
        return Err(format!("no swigate nodes found for layers '{layers_spec}' on {}", spec.pop_id));
    }

    let encode_list = |ps: &[String]| -> Result<Vec<Vec<i32>>, String> {
        let mut out = Vec::with_capacity(ps.len());
        for p in ps {
            let mut ids = open.pop.tokenizer.encode(p).map_err(|e| e.to_string())?;
            ids.truncate(256);
            if ids.is_empty() {
                ids.push(0);
            }
            out.push(ids);
        }
        Ok(out)
    };
    let fit_ids = encode_list(fit_prompts)?;
    let eval_prompts = if check_prompts.is_empty() { fit_prompts } else { check_prompts };
    let eval_ids = encode_list(eval_prompts)?;

    let width = open.pop.dims.inter;
    let lam = ridge_lambda.max(1e-4);

    // Precompute reference `final_norm` outputs on eval prompts once (reused for exact sequential
    // layer gating with zero extra open-reference passes!).
    let mut eval_ref_fin: Vec<Vec<f32>> = Vec::with_capacity(eval_ids.len());
    for ids in &eval_ids {
        let ro = open.forward(dev, ids, std::slice::from_ref(&fin_target), &RunOpts::default()).map_err(|e| e.to_string())?;
        eval_ref_fin.push(nmengn::model::values(ro.get(&fin_target).ok_or("missing ref fin")?).map_err(|e| e.to_string())?.to_vec());
    }

    // Precompute open-reference `swigate` outputs on fit prompts once across all selected layers.
    let all_swi_targets: Vec<(String, String)> = swi_targets.iter().map(|(_, _, nid)| (nid.clone(), "y".to_owned())).collect();
    let mut fit_ref_swi: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(fit_ids.len()); swi_targets.len()];
    for ids in &fit_ids {
        let ro = open.forward(dev, ids, &all_swi_targets, &RunOpts::default()).map_err(|e| e.to_string())?;
        for (si, key) in all_swi_targets.iter().enumerate() {
            let yv = nmengn::model::values(ro.get(key).ok_or("missing ref swigate")?).map_err(|e| e.to_string())?;
            fit_ref_swi[si].push(yv.to_vec());
        }
    }

    let eval_with = |filt: Option<Arc<nmengn::filter::Plugged>>| -> Result<(f64, f32), String> {
        let mut cos_sum = 0.0_f64;
        let mut worst = 0.0_f32;
        let opts = RunOpts { filters: filt, ..RunOpts::default() };
        for (pi, ids) in eval_ids.iter().enumerate() {
            let co = cand.forward(dev, ids, std::slice::from_ref(&fin_target), &opts).map_err(|e| e.to_string())?;
            let rv = &eval_ref_fin[pi];
            let cv = nmengn::model::values(co.get(&fin_target).ok_or("missing cand fin")?).map_err(|e| e.to_string())?;
            cos_sum += cosine(rv, cv);
            for (&a, &b) in rv.iter().zip(cv) {
                worst = worst.max((a - b).abs());
            }
        }
        #[expect(clippy::cast_precision_loss, reason = "prompt count")]
        Ok((cos_sum / eval_ids.len().max(1) as f64, worst))
    };

    let (base_cos, base_md) = eval_with(None)?;
    let filter_name = format!("{}-{job_name}@1", spec.short_role);
    // Check only (`--check-file <job-filter.json>`): an existing job filter, an edited version say,
    // measured on the held-out prompts against the same reference; nothing is fitted or written.
    let check_file = CHECK_FILE.lock().ok().and_then(|g| g.clone());
    if let Some(f) = check_file.as_deref() {
        let mut doc = nmjson::parse(&std::fs::read(f).map_err(|e| format!("{f}: {e}"))?).map_err(|e| format!("{f}: {e}"))?;
        let model = doc.get("model").and_then(nmjson::Value::string).unwrap_or("").to_owned();
        // A filter written plugged out is measured as if plugged in.
        if let nmjson::Value::Object(o) = &mut doc {
            o.insert("enabled".into(), nmjson::Value::Bool(true));
        }
        let bytes = doc.encode().map_err(|e| e.to_string())?;
        let cf = nmengn::filter::parse_job_filter_json(bytes.as_bytes(), "")?.ok_or_else(|| format!("{f}: not a job filter"))?;
        let plugged = Arc::new(nmengn::filter::build(&model, &[cf])?);
        let (c, md) = eval_with(Some(plugged))?;
        let pass = c > base_cos + 1e-5 && md <= base_md * 1.02;
        return Ok(format!(
            "held-out check of {f} on {} prompts: final cosine {c:.6} against {base_cos:.6} without it, max difference {md:.4} against {base_md:.4}: {}
",
            eval_ids.len(),
            if pass { "PASSES" } else { "DOES NOT PASS" }
        ));
    }

    // Sequential cumulative fitting: each layer `si` sees the EXACT post-filter activations of
    // all previously accepted layers (zero compounding multi-layer overshoot!) and is kept ONLY
    // if it strictly improves held-out `final_cos` without regressing `max_diff`.
    let mut parts: Vec<nmengn::filter::Part> = Vec::new();
    let mut best_cos = base_cos;
    let mut best_md = base_md;
    let mut cur_plugged: Option<Arc<nmengn::filter::Plugged>> = None;

    for (si, (_, short, nid)) in swi_targets.iter().enumerate() {
        let key = (nid.clone(), "y".to_owned());
        let mut sum_x = vec![0.0_f64; width];
        let mut sum_y = vec![0.0_f64; width];
        let mut sum_xx = vec![0.0_f64; width];
        let mut sum_xy = vec![0.0_f64; width];
        let mut total_tok: usize = 0;

        let fit_opts = RunOpts { filters: cur_plugged.clone(), ..RunOpts::default() };
        for (pi, ids) in fit_ids.iter().enumerate() {
            let co = cand.forward(dev, ids, std::slice::from_ref(&key), &fit_opts).map_err(|e| e.to_string())?;
            let xv = nmengn::model::values(co.get(&key).ok_or("missing cand swigate")?).map_err(|e| e.to_string())?;
            let yv = &fit_ref_swi[si][pi];
            let rows = xv.len().min(yv.len()) / width.max(1);
            total_tok += rows;
            for r in 0..rows {
                let off = r * width;
                for c in 0..width {
                    let x = f64::from(xv[off + c]);
                    let y = f64::from(yv[off + c]);
                    sum_x[c] += x;
                    sum_y[c] += y;
                    sum_xx[c] += x * x;
                    sum_xy[c] += x * y;
                }
            }
        }

        #[expect(clippy::cast_precision_loss, reason = "token count denominator")]
        let n_f = (total_tok.max(1)) as f64;
        let mut gain = vec![1.0_f32; width];
        let mut bias = vec![0.0_f32; width];
        for c in 0..width {
            let mx = sum_x[c] / n_f;
            let my = sum_y[c] / n_f;
            let var_x = (sum_xx[c] / n_f - mx * mx).max(0.0);
            let cov_xy = sum_xy[c] / n_f - mx * my;
            // Strong ridge shrinkage toward identity (gain = 1.0, bias = 0.0).
            let g = ((cov_xy + lam) / (var_x + lam)).clamp(0.85, 1.15);
            let b = ((my - g * mx) * (var_x / (var_x + lam))).clamp(-0.1, 0.1);
            #[expect(clippy::cast_possible_truncation, reason = "filter stored in f32")]
            {
                gain[c] = if (g - 1.0).abs() < 0.005 { 1.0 } else { g as f32 };
                bias[c] = if b.abs() < 0.002 { 0.0 } else { b as f32 };
            }
        }

        let mut trial_parts = parts.clone();
        trial_parts.push(nmengn::filter::Part { target: short.clone(), gain, bias });
        let trial_cf = nmengn::filter::CartridgeFilter { cartridge: filter_name.clone(), parts: trial_parts.clone() };
        let trial_plugged = Arc::new(nmengn::filter::build(&spec.pop_id, &[trial_cf])?);
        let (t_cos, t_md) = eval_with(Some(Arc::clone(&trial_plugged)))?;
        if t_cos > best_cos + 1e-6 && t_md <= best_md * 1.02 {
            parts = trial_parts;
            best_cos = t_cos;
            best_md = t_md;
            cur_plugged = Some(trial_plugged);
        }
    }

    let n_swi = parts.len();
    let plugged = cur_plugged.unwrap_or_else(|| Arc::new(nmengn::filter::Plugged::default()));
    let (filt_cos, filt_md) = (best_cos, best_md);
    let improved = !parts.is_empty() && filt_cos > base_cos + 1e-5 && filt_md <= base_md * 1.02;

    let mut out = String::new();
    let split_lbl = if check_prompts.is_empty() { "fit-only" } else { "held-out check" };
    let _ = writeln!(
        out,
        "=== Job-Path FFN Filter ({filter_name} on {cur_key}, {n_swi}/{n_layers} swigate layers, lambda={lam:.3}, {split_lbl}) ==="
    );
    #[expect(clippy::cast_precision_loss, reason = "display MiB")]
    let filt_mib = (plugged.values() as f64 * 4.0) / (1024.0 * 1024.0);
    let _ = writeln!(
        out,
        "EVAL ({} prompts, zero base weight edits): final_cos {:.6} -> {:.6} ({:+.6}) | max_diff {:.4} -> {:.4} ({:+.4}) | values={} ({filt_mib:.2} MiB f32)",
        eval_ids.len(),
        base_cos,
        filt_cos,
        filt_cos - base_cos,
        base_md,
        filt_md,
        filt_md - base_md,
        plugged.values()
    );

    if write_out {
        if !improved {
            return Err(format!("{out}\nREFUSED WRITE: job-path filter did not improve evaluation metrics."));
        }
        let dir = home.join(format!("library/parameter-maps/{}", spec.short_role));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join(format!("job-filter-{job_name}.json"));
        let mut json = String::new();
        let _ = writeln!(json, "{{");
        let _ = writeln!(json, "  \"contract\": \"{}\",", nmengn::filter::SOCKETS);
        let _ = writeln!(json, "  \"cartridge\": \"{filter_name}\",");
        let _ = writeln!(json, "  \"base\": \"{cur_key}\",");
        let _ = writeln!(json, "  \"model\": \"{}\",", spec.pop_id);
        let _ = writeln!(json, "  \"job\": \"{job_name}\",");
        let _ = writeln!(json, "  \"base_cos\": {base_cos:.6},");
        let _ = writeln!(json, "  \"filtered_cos\": {filt_cos:.6},");
        let _ = writeln!(json, "  \"base_max_diff\": {base_md:.4},");
        let _ = writeln!(json, "  \"filtered_max_diff\": {filt_md:.4},");
        let _ = writeln!(json, "  \"parts\": [");
        for (pi, p) in parts.iter().enumerate() {
            let comma = if pi + 1 < parts.len() { "," } else { "" };
            let gs: Vec<String> = p.gain.iter().map(|v| format!("{v:.6}")).collect();
            let bs: Vec<String> = p.bias.iter().map(|v| format!("{v:.6}")).collect();
            let _ = writeln!(
                json,
                "    {{\"target\": \"{}\", \"gain\": [{}], \"bias\": [{}]}}{comma}",
                p.target,
                gs.join(","),
                bs.join(",")
            );
        }
        let _ = writeln!(json, "  ]\n}}");
        std::fs::write(&path, json).map_err(|e| format!("{}: {e}", path.display()))?;
        // 2026-11-26: the filter's lineage, as it is written (nmapi::lineage).
        let _ = nmapi::lineage::record_made(&dir, &format!("job-filter-{job_name}.json"), cur_key, &format!("nano-lab run pmap surgery job-filter {} --job {job_name} --write", spec.short_role), "serving");
        let _ = writeln!(out, "WRITTEN: {}", path.display());
    } else {
        let _ = writeln!(out, "DRY RUN: pass --write to save library/parameter-maps/{}/job-filter-{job_name}.json.", spec.short_role);
    }
    let _ = writeln!(out, "elapsed: {:.2} s", t_total.elapsed().as_secs_f64());
    Ok(out)
}
