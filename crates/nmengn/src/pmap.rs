//! Bidirectional surgical parameter mapping, fine-grained editing, compressed bitmask-q8 packing,
//! exact token-row trimming, and path add-back for Neuromorph Nano base cartridges.
//!
//! Architecture by Luis Jake Gabriel III (Gavriel Technologies, 2026-11-19 & 2026-11-21):
//! - Maps individual parameter usage per job across all linear projections (attention `q/k/v/o`
//!   with tentacle relevance heads pinned + FFN `gate/up/down`) and exact vocabulary token rows.
//! - Supports bidirectional surgical cartridge operations:
//!   - **Surgical removal** (`M_next = M_current AND NOT M_drop`) to remove unused parameter paths.
//!   - **Surgical addition** (`M_next = M_current OR M_target`, `Delta = M_target AND NOT M_current`)
//!     pulling exact parameter values from untouched reference `weights/` whenever pruning went
//!     too far or a cartridge needs paths for a new job/capability.
//! - Never modifies released open weights (`weights/` stays read-only reference and fallback).
use nmjson::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// The four shared linear input streams inside each Transformer layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stream {
    /// Input to `self_attn.q_proj`, `k_proj`, `v_proj` (shape `[hidden]`).
    Attn,
    /// Input to `self_attn.o_proj` (shape `[heads * head_dim]`).
    OutProj,
    /// Input to `mlp.gate_proj`, `up_proj` (shape `[hidden]`).
    Mlp,
    /// Input to `mlp.down_proj` (shape `[inter]`).
    Down,
}

impl Stream {
    /// Canonical JSON key for this input stream.
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            Self::Attn => "attn_in",
            Self::OutProj => "o_in",
            Self::Mlp => "mlp_in",
            Self::Down => "down_in",
        }
    }

    /// Parse a stream key.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "attn_in" => Some(Self::Attn),
            "o_in" => Some(Self::OutProj),
            "mlp_in" => Some(Self::Mlp),
            "down_in" => Some(Self::Down),
            _ => None,
        }
    }

    /// Which input stream feeds a given Qwen3 linear tensor name.
    #[must_use]
    pub fn for_tensor(name: &str) -> Option<(usize, Self)> {
        let (layer_str, part) = name.strip_prefix("model.layers.")?.split_once('.')?;
        let layer = layer_str.parse::<usize>().ok()?;
        let stream = match part {
            "self_attn.q_proj.weight" | "self_attn.k_proj.weight" | "self_attn.v_proj.weight" => Self::Attn,
            "self_attn.o_proj.weight" => Self::OutProj,
            "mlp.gate_proj.weight" | "mlp.up_proj.weight" => Self::Mlp,
            "mlp.down_proj.weight" => Self::Down,
            _ => return None,
        };
        Some((layer, stream))
    }

    /// Representative linear node suffix used to record each shared stream exactly once per layer.
    #[must_use]
    pub fn from_node_id(node_id: &str) -> Option<(usize, Self)> {
        let short = node_id.rsplit_once('#').map_or(node_id, |x| x.1);
        let rest = short.strip_prefix('L')?;
        let (layer_str, suffix) = rest.split_once('.')?;
        let layer = layer_str.parse::<usize>().ok()?;
        let stream = match suffix {
            "q" => Self::Attn,
            "o" => Self::OutProj,
            "gate" => Self::Mlp,
            "down" => Self::Down,
            _ => return None,
        };
        Some((layer, stream))
    }
}

/// Accumulated activation statistics for one input stream of one layer under one job.
#[derive(Clone, Debug, Default)]
pub struct StreamStats {
    pub tokens: u64,
    pub sum: Vec<f64>,
    pub sq: Vec<f64>,
}

impl StreamStats {
    /// Accumulate a `[tokens, width]` row-major activation slice.
    pub fn add_rows(&mut self, width: usize, x: &[f32]) {
        if width == 0 || !x.len().is_multiple_of(width) {
            return;
        }
        if self.sum.is_empty() {
            self.sum.resize(width, 0.0);
            self.sq.resize(width, 0.0);
        }
        if self.sum.len() != width {
            return;
        }
        for row in x.chunks_exact(width) {
            for ((s, q), v) in self.sum.iter_mut().zip(self.sq.iter_mut()).zip(row) {
                let d = f64::from(*v);
                *s += d;
                *q += d * d;
            }
            self.tokens = self.tokens.saturating_add(1);
        }
    }

    /// Per-feature L2 norm `sqrt(sum x_j^2)` used by Wanda scoring.
    #[must_use]
    pub fn l2_norms(&self) -> Vec<f32> {
        #[expect(clippy::cast_possible_truncation, reason = "activation norm stored as f32 for Wanda scoring")]
        self.sq.iter().map(|q| q.max(0.0).sqrt() as f32).collect()
    }

    /// Per-feature mean `E[x_j]` used for per-output bias compensation.
    #[must_use]
    pub fn means(&self) -> Vec<f32> {
        #[expect(clippy::cast_precision_loss, reason = "token count to f64 mean denominator")]
        let denom = self.tokens.max(1) as f64;
        #[expect(clippy::cast_possible_truncation, reason = "feature mean stored as f32")]
        self.sum.iter().map(|s| (*s / denom) as f32).collect()
    }
}

/// Recorded per-job parameter usage statistics for one job.
#[derive(Clone, Debug, Default)]
pub struct JobStats {
    pub layers: BTreeMap<usize, BTreeMap<Stream, StreamStats>>,
    pub tokens_in: BTreeSet<u32>,
    pub tokens_out: BTreeSet<u32>,
}

/// Complete recorded per-job parameter statistics for one MicroModel.
#[derive(Clone, Debug, Default)]
pub struct ParamRecord {
    pub model: String,
    pub jobs: BTreeMap<String, JobStats>,
}

impl ParamRecord {
    /// Encode the compact parameter-recording JSON.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let num = |x: f64| Value::Number(format!("{x:e}"));
        let jobs_obj: BTreeMap<String, Value> = self
            .jobs
            .iter()
            .map(|(job, js)| {
                let layers_obj: BTreeMap<String, Value> = js
                    .layers
                    .iter()
                    .map(|(layer, streams)| {
                        let s_obj: BTreeMap<String, Value> = streams
                            .iter()
                            .map(|(st, ss)| {
                                let entry = Value::Object(
                                    [
                                        ("tokens".to_owned(), Value::Number(ss.tokens.to_string())),
                                        ("sum".to_owned(), Value::Array(ss.sum.iter().map(|x| num(*x)).collect())),
                                        ("sq".to_owned(), Value::Array(ss.sq.iter().map(|x| num(*x)).collect())),
                                    ]
                                    .into_iter()
                                    .collect(),
                                );
                                (st.key().to_owned(), entry)
                            })
                            .collect();
                        (format!("L{layer}"), Value::Object(s_obj))
                    })
                    .collect();
                let jval = Value::Object(
                    [
                        ("tokens_in".to_owned(), Value::Array(js.tokens_in.iter().map(|id| Value::Number(id.to_string())).collect())),
                        ("tokens_out".to_owned(), Value::Array(js.tokens_out.iter().map(|id| Value::Number(id.to_string())).collect())),
                        ("layers".to_owned(), Value::Object(layers_obj)),
                    ]
                    .into_iter()
                    .collect(),
                );
                (job.clone(), jval)
            })
            .collect();
        Value::Object(
            [
                ("schema".to_owned(), Value::String("nm-pmap/1".to_owned())),
                ("model".to_owned(), Value::String(self.model.clone())),
                ("jobs".to_owned(), Value::Object(jobs_obj)),
            ]
            .into_iter()
            .collect(),
        )
    }

    /// Decode a recorded parameter-map JSON file.
    /// # Errors
    /// Schema or parse error.
    pub fn from_value(v: &Value) -> Result<Self, String> {
        let model = v.get("model").and_then(Value::string).unwrap_or_default().to_owned();
        let mut jobs = BTreeMap::new();
        if let Ok(jmap) = v.get("jobs").and_then(Value::object) {
            for (jname, jv) in jmap {
                let mut js = JobStats::default();
                if let Ok(arr) = jv.get("tokens_in").and_then(Value::array) {
                    for x in arr {
                        if let Ok(id) = x.integer()
                            && let Ok(u) = u32::try_from(id)
                        {
                            js.tokens_in.insert(u);
                        }
                    }
                }
                if let Ok(arr) = jv.get("tokens_out").and_then(Value::array) {
                    for x in arr {
                        if let Ok(id) = x.integer()
                            && let Ok(u) = u32::try_from(id)
                        {
                            js.tokens_out.insert(u);
                        }
                    }
                }
                if let Ok(lmap) = jv.get("layers").and_then(Value::object) {
                    for (lname, lv) in lmap {
                        let Some(layer) = lname.strip_prefix('L').and_then(|s| s.parse::<usize>().ok()) else {
                            continue;
                        };
                        let Ok(smap) = lv.object() else { continue };
                        let mut streams = BTreeMap::new();
                        for (sname, sv) in smap {
                            let Some(st) = Stream::parse(sname) else { continue };
                            let tokens = sv.get("tokens").and_then(Value::integer).unwrap_or(0);
                            let parse_vec = |k: &str| -> Vec<f64> {
                                sv.get(k)
                                    .and_then(Value::array)
                                    .map(|a| {
                                        a.iter()
                                            .map(|n| match n {
                                                Value::Number(t) => t.parse::<f64>().unwrap_or(0.0),
                                                _ => 0.0,
                                            })
                                            .collect()
                                    })
                                    .unwrap_or_default()
                            };
                            streams.insert(st, StreamStats { tokens, sum: parse_vec("sum"), sq: parse_vec("sq") });
                        }
                        js.layers.insert(layer, streams);
                    }
                }
                jobs.insert(jname.clone(), js);
            }
        }
        Ok(Self { model, jobs })
    }

    /// Save atomically to `path`.
    /// # Errors
    /// Encoding or filesystem I/O error.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        let encoded = self.to_value().encode().map_err(|e| e.to_string())?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, encoded).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Load from `path`.
    /// # Errors
    /// Read or JSON parse error.
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let v = nmjson::parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_value(&v)
    }
}

// ---------------------------------------------------------------------------------------------
// Live process-wide recorder (off unless NANO_PARAM_MAP is set or enabled programmatically).

struct LiveRec {
    file: Option<PathBuf>,
    pop: String,
    data: Mutex<(ParamRecord, Instant)>,
}

static ENV_REC: OnceLock<Option<LiveRec>> = OnceLock::new();
static PROG_REC: Mutex<Option<LiveRec>> = Mutex::new(None);

fn env_rec() -> Option<&'static LiveRec> {
    ENV_REC
        .get_or_init(|| {
            let file = std::env::var("NANO_PARAM_MAP").ok().filter(|f| !f.is_empty())?;
            let pop = std::env::var("NANO_PARAM_POP").unwrap_or_else(|_| "qwen3-0.6b".to_owned());
            Some(LiveRec {
                file: Some(PathBuf::from(file)),
                pop: pop.clone(),
                data: Mutex::new((ParamRecord { model: pop, jobs: BTreeMap::new() }, Instant::now())),
            })
        })
        .as_ref()
}

/// Enable programmatic recording for `pop` (used by `nano-lab run pmap` without env-var restarts).
pub fn start_recording(pop: &str, file: Option<PathBuf>) {
    if let Ok(mut g) = PROG_REC.lock() {
        *g = Some(LiveRec {
            file,
            pop: pop.to_owned(),
            data: Mutex::new((ParamRecord { model: pop.to_owned(), jobs: BTreeMap::new() }, Instant::now())),
        });
    }
}

/// Stop programmatic recording and return the collected `ParamRecord`.
pub fn stop_recording() -> Option<ParamRecord> {
    let rec = PROG_REC.lock().ok()?.take()?;
    let out = rec.data.lock().ok().map(|g| g.0.clone());
    if let (Some(f), Some(r)) = (rec.file.as_ref(), out.as_ref()) {
        let _ = r.save(f);
    }
    out
}

/// Whether parameter-level recording is active for any population.
#[must_use]
pub fn any_on() -> bool {
    env_rec().is_some() || PROG_REC.lock().ok().is_some_and(|g| g.is_some())
}

/// Whether parameter-level recording is active for population `pop`.
#[must_use]
pub fn on(pop: &str) -> bool {
    if env_rec().is_some_and(|r| r.pop == pop) {
        return true;
    }
    PROG_REC.lock().ok().is_some_and(|g| g.as_ref().is_some_and(|r| r.pop == pop))
}

fn with_live(pop: &str, mut f: impl FnMut(&mut ParamRecord)) {
    if let Some(r) = env_rec().filter(|r| r.pop == pop)
        && let Ok(mut g) = r.data.lock()
    {
        f(&mut g.0);
        if g.1.elapsed().as_secs() >= 20 {
            g.1 = Instant::now();
            if let Some(path) = r.file.as_ref() {
                let _ = g.0.save(path);
            }
        }
    }
    if let Ok(guard) = PROG_REC.lock()
        && let Some(r) = guard.as_ref().filter(|r| r.pop == pop)
        && let Ok(mut g) = r.data.lock()
    {
        f(&mut g.0);
    }
}

/// Record one linear input activation stream `[tokens, width]` at `(layer, stream)` for the active job.
pub fn note_input(pop: &str, layer: usize, stream: Stream, width: usize, x: &[f32]) {
    let job = crate::chanrec::current_job();
    with_live(pop, |rec| {
        rec.jobs
            .entry(job.to_owned())
            .or_default()
            .layers
            .entry(layer)
            .or_default()
            .entry(stream)
            .or_default()
            .add_rows(width, x);
    });
}

/// Record active vocabulary token IDs (`output == false` for `embed_tokens`, `true` for `lm_head`).
pub fn note_tokens(pop: &str, ids: &[i32], output: bool) {
    let job = crate::chanrec::current_job();
    with_live(pop, |rec| {
        let js = rec.jobs.entry(job.to_owned()).or_default();
        let set = if output { &mut js.tokens_out } else { &mut js.tokens_in };
        for &id in ids {
            if let Ok(u) = u32::try_from(id) {
                set.insert(u);
            }
        }
    });
}

/// Flush any env-configured live parameter map to disk.
pub fn flush() {
    if let Some(r) = env_rec()
        && let Some(path) = r.file.as_ref()
        && let Ok(g) = r.data.lock()
    {
        let _ = g.0.save(path);
    }
}

// ---------------------------------------------------------------------------------------------
// Compact 1-bit-per-weight BitMask and bidirectional surgical mask operations.

/// Compact 1-bit-per-weight mask for a `[rows, cols]` linear weight matrix (1 = kept, 0 = removed).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitMask {
    pub rows: usize,
    pub cols: usize,
    pub bits: Vec<u8>,
}

impl BitMask {
    /// Create a mask with all `rows * cols` weights either kept (`filled == true`) or empty.
    #[must_use]
    pub fn new(rows: usize, cols: usize, filled: bool) -> Self {
        let total = rows.saturating_mul(cols);
        let bytes = total.div_ceil(8);
        let mut bits = vec![if filled { 0xFF } else { 0x00 }; bytes];
        if filled && !total.is_multiple_of(8)
            && let Some(last) = bits.last_mut()
        {
            *last = (1_u8 << (total % 8)) - 1;
        }
        Self { rows, cols, bits }
    }

    #[must_use]
    pub const fn total(&self) -> usize {
        self.rows.saturating_mul(self.cols)
    }

    #[must_use]
    pub fn get(&self, r: usize, c: usize) -> bool {
        if r >= self.rows || c >= self.cols {
            return false;
        }
        let idx = r * self.cols + c;
        self.bits.get(idx / 8).is_some_and(|b| (b & (1_u8 << (idx % 8))) != 0)
    }

    pub fn set(&mut self, r: usize, c: usize, kept: bool) {
        if r >= self.rows || c >= self.cols {
            return;
        }
        let idx = r * self.cols + c;
        if let Some(b) = self.bits.get_mut(idx / 8) {
            if kept {
                *b |= 1_u8 << (idx % 8);
            } else {
                *b &= !(1_u8 << (idx % 8));
            }
        }
    }

    /// Number of kept weights (1-bits).
    #[must_use]
    pub fn kept_count(&self) -> usize {
        self.bits.iter().map(|b| b.count_ones() as usize).sum()
    }

    /// Number of removed weights (0-bits).
    #[must_use]
    pub fn removed_count(&self) -> usize {
        self.total().saturating_sub(self.kept_count())
    }

    /// Fraction of weights removed (`0.0..=1.0`).
    #[must_use]
    pub fn sparsity(&self) -> f64 {
        let tot = self.total();
        if tot == 0 {
            return 0.0;
        }
        #[expect(clippy::cast_precision_loss, reason = "sparsity ratio")]
        {
            self.removed_count() as f64 / tot as f64
        }
    }

    /// Logical OR (`self = self OR other`): returns count of newly added weights (`other AND NOT self`).
    pub fn add_with(&mut self, other: &Self) -> usize {
        if self.rows != other.rows || self.cols != other.cols {
            return 0;
        }
        let mut added = 0;
        for (a, b) in self.bits.iter_mut().zip(&other.bits) {
            let delta = *b & !*a;
            added += delta.count_ones() as usize;
            *a |= *b;
        }
        added
    }

    /// Logical AND NOT (`self = self AND NOT drop`): returns count of newly removed weights.
    pub fn remove_with(&mut self, drop: &Self) -> usize {
        if self.rows != drop.rows || self.cols != drop.cols {
            return 0;
        }
        let mut removed = 0;
        for (a, d) in self.bits.iter_mut().zip(&drop.bits) {
            let gone = *a & *d;
            removed += gone.count_ones() as usize;
            *a &= !*d;
        }
        removed
    }

    /// Jaccard similarity `|A AND B| / |A OR B|` over kept weights.
    #[must_use]
    pub fn jaccard(&self, other: &Self) -> f64 {
        if self.rows != other.rows || self.cols != other.cols {
            return 0.0;
        }
        let (mut inter, mut uni) = (0_usize, 0_usize);
        for (a, b) in self.bits.iter().zip(&other.bits) {
            inter += (*a & *b).count_ones() as usize;
            uni += (*a | *b).count_ones() as usize;
        }
        if uni == 0 {
            return 1.0;
        }
        #[expect(clippy::cast_precision_loss, reason = "Jaccard ratio")]
        {
            inter as f64 / uni as f64
        }
    }

    /// Share of `1 x tile_k` input tiles that are 100% zero (skippable by tile-skipping kernels).
    #[must_use]
    pub fn k_tile_skip_ratio(&self, tile_k: usize) -> f64 {
        if tile_k == 0 || self.rows == 0 || self.cols < tile_k {
            return 0.0;
        }
        let tiles_per_row = self.cols / tile_k;
        let total_tiles = self.rows * tiles_per_row;
        let mut zero_tiles = 0_usize;
        for r in 0..self.rows {
            for t in 0..tiles_per_row {
                let c0 = t * tile_k;
                let any = (c0..c0 + tile_k).any(|c| self.get(r, c));
                if !any {
                    zero_tiles += 1;
                }
            }
        }
        #[expect(clippy::cast_precision_loss, reason = "tile skip ratio")]
        {
            zero_tiles as f64 / total_tiles.max(1) as f64
        }
    }
}

/// Network-wide parameter mask across all linear tensors and vocabulary tables.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParamMask {
    pub tensors: BTreeMap<String, BitMask>,
    pub tokens_in: BTreeSet<u32>,
    pub tokens_out: BTreeSet<u32>,
    /// Pinned query head indices per layer (`layer -> [head_idx]`) that must never be pruned.
    pub pinned_heads: BTreeMap<usize, Vec<usize>>,
}

/// Summary of a bidirectional surgical `remove` or `add` operation on a `ParamMask`.
#[derive(Clone, Debug, Default)]
pub struct SurgicalDelta {
    pub changed_weights: usize,
    pub changed_tokens_in: usize,
    pub changed_tokens_out: usize,
    pub per_tensor: BTreeMap<String, usize>,
}

/// Convert a BF16 little-endian 2-byte slice to f32.
#[inline]
#[must_use]
pub fn bf16_to_f32(lo: u8, hi: u8) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([lo, hi])) << 16)
}

/// Decode a BF16 tensor buffer into f32 values.
#[must_use]
pub fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<2>().0.iter().map(|c| bf16_to_f32(c[0], c[1])).collect()
}

/// Build a Wanda per-weight mask for a `[rows, cols]` matrix:
/// `score(i, j) = |w_ij| * input_l2[j]`, keeping the top `(1 - sparsity)` per output row, while
/// strictly keeping every weight in `pinned.0` (rows) or `pinned.1` (cols), e.g. tentacle
/// relevance heads.
#[must_use]
pub fn wanda_mask_for_matrix(
    weights: &[f32],
    rows: usize,
    cols: usize,
    input_l2: &[f32],
    sparsity: f64,
    pinned: (&BTreeSet<usize>, &BTreeSet<usize>),
) -> BitMask {
    let (pinned_rows, pinned_cols) = pinned;
    let s = sparsity.clamp(0.0, 0.99);
    let mut mask = BitMask::new(rows, cols, true);
    if s <= 0.0 || rows == 0 || cols == 0 || weights.len() < rows * cols {
        return mask;
    }
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss, reason = "per-row prune target count")]
    let target_drop = ((cols as f64) * s).floor() as usize;
    if target_drop == 0 {
        return mask;
    }
    let mut scored: Vec<(f32, usize)> = Vec::with_capacity(cols);
    for r in 0..rows {
        if pinned_rows.contains(&r) {
            continue;
        }
        scored.clear();
        let row = &weights[r * cols..(r + 1) * cols];
        for (c, &w) in row.iter().enumerate() {
            if pinned_cols.contains(&c) {
                continue;
            }
            let xnorm = input_l2.get(c).copied().unwrap_or(1.0);
            scored.push((w.abs() * xnorm, c));
        }
        let drop_n = target_drop.min(scored.len());
        if drop_n == 0 {
            continue;
        }
        if drop_n < scored.len() {
            scored.select_nth_unstable_by(drop_n, |a, b| a.0.total_cmp(&b.0));
        }
        for &(_, c) in &scored[..drop_n] {
            mask.set(r, c, false);
        }
    }
    mask
}

/// Pinned row/col index sets for a tensor given `pinned_heads` and `head_dim` (protects tentacle
/// relevance heads in `q_proj` rows and `o_proj` columns).
#[must_use]
pub fn pinned_axes_for_tensor(name: &str, head_dim: usize, pinned_heads: &BTreeMap<usize, Vec<usize>>) -> (BTreeSet<usize>, BTreeSet<usize>) {
    let mut rows = BTreeSet::new();
    let mut cols = BTreeSet::new();
    let Some((layer_str, part)) = name.strip_prefix("model.layers.").and_then(|r| r.split_once('.')) else {
        return (rows, cols);
    };
    let Ok(layer) = layer_str.parse::<usize>() else {
        return (rows, cols);
    };
    let Some(heads) = pinned_heads.get(&layer) else {
        return (rows, cols);
    };
    for &h in heads {
        let lo = h * head_dim;
        let hi = lo + head_dim;
        if part == "self_attn.q_proj.weight" {
            for idx in lo..hi {
                rows.insert(idx);
            }
        } else if part == "self_attn.o_proj.weight" {
            for idx in lo..hi {
                cols.insert(idx);
            }
        }
    }
    (rows, cols)
}

/// Union of multiple job masks (`M = OR_t M_t`). A parameter is kept if any job needs it.
#[must_use]
pub fn union_masks(masks: &[&ParamMask]) -> ParamMask {
    let mut out = ParamMask::default();
    for m in masks {
        for &id in &m.tokens_in {
            out.tokens_in.insert(id);
        }
        for &id in &m.tokens_out {
            out.tokens_out.insert(id);
        }
        for (l, hs) in &m.pinned_heads {
            let e = out.pinned_heads.entry(*l).or_default();
            for h in hs {
                if !e.contains(h) {
                    e.push(*h);
                }
            }
        }
        for (name, bm) in &m.tensors {
            out.tensors
                .entry(name.clone())
                .and_modify(|cur| {
                    let _ = cur.add_with(bm);
                })
                .or_insert_with(|| bm.clone());
        }
    }
    out
}

/// Bidirectional surgical removal: `M_next = M_current AND NOT M_drop`, re-enforcing pinned heads.
#[must_use]
pub fn surgical_remove(current: &ParamMask, drop: &ParamMask, head_dim: usize) -> (ParamMask, SurgicalDelta) {
    let mut next = current.clone();
    let mut delta = SurgicalDelta::default();
    for (name, drop_bm) in &drop.tensors {
        if let Some(cur_bm) = next.tensors.get_mut(name) {
            let (pin_r, pin_c) = pinned_axes_for_tensor(name, head_dim, &current.pinned_heads);
            let mut safe_drop = drop_bm.clone();
            for &r in &pin_r {
                for c in 0..safe_drop.cols {
                    safe_drop.set(r, c, false);
                }
            }
            for &c in &pin_c {
                for r in 0..safe_drop.rows {
                    safe_drop.set(r, c, false);
                }
            }
            let n = cur_bm.remove_with(&safe_drop);
            if n > 0 {
                delta.changed_weights += n;
                delta.per_tensor.insert(name.clone(), n);
            }
        }
    }
    for id in &drop.tokens_in {
        if next.tokens_in.remove(id) {
            delta.changed_tokens_in += 1;
        }
    }
    for id in &drop.tokens_out {
        if next.tokens_out.remove(id) {
            delta.changed_tokens_out += 1;
        }
    }
    (next, delta)
}

/// Bidirectional surgical addition: `Delta = M_target AND NOT M_current`, `M_next = M_current OR M_target`.
/// Used both to recover over-pruned quality from untouched reference `weights/` and to graft exact
/// parameter paths needed for a new job/capability into a cartridge.
#[must_use]
pub fn surgical_add(current: &ParamMask, target: &ParamMask) -> (ParamMask, SurgicalDelta) {
    let mut next = current.clone();
    let mut delta = SurgicalDelta::default();
    for (name, tgt_bm) in &target.tensors {
        match next.tensors.get_mut(name) {
            Some(cur_bm) => {
                let n = cur_bm.add_with(tgt_bm);
                if n > 0 {
                    delta.changed_weights += n;
                    delta.per_tensor.insert(name.clone(), n);
                }
            }
            None => {
                let n = tgt_bm.kept_count();
                next.tensors.insert(name.clone(), tgt_bm.clone());
                if n > 0 {
                    delta.changed_weights += n;
                    delta.per_tensor.insert(name.clone(), n);
                }
            }
        }
    }
    for &id in &target.tokens_in {
        if next.tokens_in.insert(id) {
            delta.changed_tokens_in += 1;
        }
    }
    for &id in &target.tokens_out {
        if next.tokens_out.insert(id) {
            delta.changed_tokens_out += 1;
        }
    }
    (next, delta)
}

/// Per-output bias compensation `b_i = sum_{removed j} w_ij * E[x_j]` for a masked linear matrix.
#[must_use]
pub fn bias_compensation(weights: &[f32], rows: usize, cols: usize, mask: &BitMask, input_mean: &[f32]) -> Vec<f32> {
    let mut bias = vec![0.0_f32; rows];
    if weights.len() < rows * cols {
        return bias;
    }
    for (r, slot) in bias.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        let row = &weights[r * cols..(r + 1) * cols];
        for (c, &w) in row.iter().enumerate() {
            if !mask.get(r, c) {
                let m = f64::from(input_mean.get(c).copied().unwrap_or(0.0));
                acc += f64::from(w) * m;
            }
        }
        #[expect(clippy::cast_possible_truncation, reason = "compensation bias stored as f32")]
        {
            *slot = acc as f32;
        }
    }
    bias
}

/// Apply `mask` to a BF16 open-weights byte buffer (returns a new dense-masked copy; never edits source).
#[must_use]
pub fn apply_mask_bf16(open_bf16: &[u8], mask: &BitMask) -> Vec<u8> {
    let mut out = open_bf16.to_vec();
    let total = mask.total();
    for idx in 0..total {
        let r = idx / mask.cols;
        let c = idx % mask.cols;
        if !mask.get(r, c) {
            let off = idx * 2;
            if let Some(pair) = out.get_mut(off..off + 2) {
                pair[0] = 0;
                pair[1] = 0;
            }
        }
    }
    out
}

/// Compressed `bitmask-q8` weight representation: 1-bit mask + per-row f32 scale + packed kept `i8` values.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedBitmaskQ8 {
    pub mask: BitMask,
    pub row_scales: Vec<f32>,
    pub packed_i8: Vec<i8>,
}

impl PackedBitmaskQ8 {
    /// Total compressed payload bytes (bitmask + f16 row scales + packed kept int8 values).
    #[must_use]
    pub fn compressed_bytes(&self) -> usize {
        self.mask.bits.len() + self.row_scales.len() * 2 + self.packed_i8.len()
    }

    /// Equivalent dense 8-bit bytes (`rows * cols + rows * 2`).
    #[must_use]
    pub fn dense_q8_bytes(&self) -> usize {
        self.mask.total() + self.mask.rows * 2
    }

    /// Equivalent dense BF16 bytes (`rows * cols * 2`).
    #[must_use]
    pub fn dense_bf16_bytes(&self) -> usize {
        self.mask.total() * 2
    }

    /// Build the GPU-ready `bitmask-q8` layout for `kernels/cukern/bmq8.cu` (`__popc()` O(1) unpack).
    #[must_use]
    pub fn to_gpu_layout(&self) -> GpuBitmaskQ8 {
        let rows = self.mask.rows;
        let cols = self.mask.cols;
        let words_per_row = cols.div_ceil(32);
        let groups_per_row = cols.div_ceil(64);
        let mut mask_words = vec![0_u32; rows.saturating_mul(words_per_row)];
        let mut group_offs = vec![0_u16; rows.saturating_mul(groups_per_row)];
        let mut row_offs = vec![0_u32; rows];
        let mut global_off: u32 = 0;
        for r in 0..rows {
            row_offs[r] = global_off;
            let mut in_row: u32 = 0;
            for g in 0..groups_per_row {
                group_offs[r * groups_per_row + g] = u16::try_from(in_row).unwrap_or(u16::MAX);
                for half in 0..2 {
                    let w_idx = g * 2 + half;
                    if w_idx >= words_per_row {
                        break;
                    }
                    let mut word: u32 = 0;
                    let c0 = w_idx * 32;
                    for bit in 0..32 {
                        let c = c0 + bit;
                        if c < cols && self.mask.get(r, c) {
                            word |= 1_u32 << bit;
                            in_row += 1;
                        }
                    }
                    mask_words[r * words_per_row + w_idx] = word;
                }
            }
            global_off = global_off.saturating_add(in_row);
        }
        let row_scales_f16: Vec<u16> = self.row_scales.iter().map(|&s| crate::half::bits16(s)).collect();
        GpuBitmaskQ8 {
            rows,
            cols,
            words_per_row,
            groups_per_row,
            mask_words,
            group_offs,
            row_offs,
            packed_i8: self.packed_i8.clone(),
            row_scales_f16,
        }
    }
}

/// GPU-ready `bitmask-q8` layout matching `kernels/cukern/bmq8.cu`.
/// Metadata cost is `0.15625` bytes/weight (`uint32` bitmask + `uint16` 64-weight group offset)
/// plus `4` bytes/row for `row_offs` and `2` bytes/row for `row_scales_f16`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuBitmaskQ8 {
    pub rows: usize,
    pub cols: usize,
    pub words_per_row: usize,
    pub groups_per_row: usize,
    pub mask_words: Vec<u32>,
    pub group_offs: Vec<u16>,
    pub row_offs: Vec<u32>,
    pub packed_i8: Vec<i8>,
    pub row_scales_f16: Vec<u16>,
}

impl GpuBitmaskQ8 {
    /// Total device VRAM bytes required for this compressed matrix.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.mask_words.len() * 4
            + self.group_offs.len() * 2
            + self.row_offs.len() * 4
            + self.packed_i8.len()
            + self.row_scales_f16.len() * 2
    }

    /// Equivalent dense Q8 device bytes (`rows * cols + rows * 2`).
    #[must_use]
    pub fn dense_q8_bytes(&self) -> usize {
        self.rows.saturating_mul(self.cols) + self.rows * 2
    }

    /// Compress an already-quantised dense Q8 weight matrix (`rows * cols` `i8` bytes and `rows * 2`
    /// f16 scale bytes) into `GpuBitmaskQ8` iff exact zero fraction is at least `min_zero_pct`% and
    /// compressed device bytes are strictly smaller than dense Q8 bytes.
    #[must_use]
    pub fn from_dense_q8(
        rows: usize,
        cols: usize,
        q8_bytes: &[u8],
        scales_f16_bytes: &[u8],
        min_zero_pct: usize,
    ) -> Option<Self> {
        let total = rows.saturating_mul(cols);
        if total == 0 || q8_bytes.len() < total || scales_f16_bytes.len() < rows * 2 || cols > usize::from(u16::MAX) {
            return None;
        }
        let zeros = q8_bytes[..total].iter().filter(|&&b| b == 0).count();
        if zeros.saturating_mul(100) < total.saturating_mul(min_zero_pct) {
            return None;
        }
        let words_per_row = cols.div_ceil(32);
        let groups_per_row = cols.div_ceil(64);
        let kept = total - zeros;
        let est_bytes = rows * words_per_row * 4 + rows * groups_per_row * 2 + rows * 4 + kept + rows * 2;
        if est_bytes >= total + rows * 2 {
            return None;
        }
        let mut mask_words = vec![0_u32; rows * words_per_row];
        let mut group_offs = vec![0_u16; rows * groups_per_row];
        let mut row_offs = vec![0_u32; rows];
        let mut packed_i8 = Vec::with_capacity(kept);
        let mut global_off: u32 = 0;
        for r in 0..rows {
            row_offs[r] = global_off;
            let row_slice = &q8_bytes[r * cols..(r + 1) * cols];
            let mut in_row: u32 = 0;
            for g in 0..groups_per_row {
                group_offs[r * groups_per_row + g] = u16::try_from(in_row).unwrap_or(u16::MAX);
                for half in 0..2 {
                    let w_idx = g * 2 + half;
                    if w_idx >= words_per_row {
                        break;
                    }
                    let mut word: u32 = 0;
                    let c0 = w_idx * 32;
                    for bit in 0..32 {
                        let c = c0 + bit;
                        if c < cols {
                            let b = row_slice[c];
                            if b != 0 {
                                word |= 1_u32 << bit;
                                packed_i8.push(i8::from_ne_bytes([b]));
                                in_row += 1;
                            }
                        }
                    }
                    mask_words[r * words_per_row + w_idx] = word;
                }
            }
            global_off = global_off.saturating_add(in_row);
        }
        let row_scales_f16: Vec<u16> = scales_f16_bytes[..rows * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b))
            .collect();
        Some(Self {
            rows,
            cols,
            words_per_row,
            groups_per_row,
            mask_words,
            group_offs,
            row_offs,
            packed_i8,
            row_scales_f16,
        })
    }
}

/// Outlier-Weighted Layerwise Sparsity (OWL, ICML 2024): allocates per-layer sparsity budgets
/// centered on `target_sparsity` with spread `lambda` (e.g. `0.08`) so high-outlier sensitive
/// layers stay denser and redundant low-outlier layers prune harder while preserving the exact
/// mean `target_sparsity`.
#[must_use]
pub fn owl_layer_sparsities(layer_outlier_ratios: &[f64], target_sparsity: f64, lambda: f64) -> Vec<f64> {
    let n = layer_outlier_ratios.len();
    if n == 0 {
        return Vec::new();
    }
    let target = target_sparsity.clamp(0.0, 0.90);
    let lam = lambda.clamp(0.0, 0.25);
    #[expect(clippy::cast_precision_loss, reason = "layer count is small")]
    let mean_r = layer_outlier_ratios.iter().copied().sum::<f64>() / n as f64;
    let max_dev = layer_outlier_ratios.iter().map(|r| (r - mean_r).abs()).fold(0.0_f64, f64::max);
    if max_dev <= 1e-12 || lam <= 0.0 {
        return vec![target; n];
    }
    // Higher outlier ratio -> lower sparsity (more weights kept); lower outlier ratio -> higher sparsity.
    let mut budgets: Vec<f64> = layer_outlier_ratios
        .iter()
        .map(|&r| {
            let norm = (mean_r - r) / max_dev; // [-1, +1]
            (target + lam * norm).clamp(0.05_f64.min(target), 0.85_f64.max(target))
        })
        .collect();
    // Re-center so the mean equals `target` cleanly.
    #[expect(clippy::cast_precision_loss, reason = "layer count is small")]
    let cur_mean = budgets.iter().sum::<f64>() / n as f64;
    let shift = target - cur_mean;
    for b in &mut budgets {
        *b = (*b + shift).clamp(0.0, 0.90);
    }
    budgets
}

/// Pack a BF16 linear weight matrix under `mask` into compressed `bitmask-q8`.
#[must_use]
pub fn pack_bitmask_q8(open_bf16: &[u8], mask: &BitMask) -> PackedBitmaskQ8 {
    let vals = decode_bf16(open_bf16);
    let mut row_scales = vec![0.0_f32; mask.rows];
    let mut packed_i8 = Vec::with_capacity(mask.kept_count());
    for (r, scale_slot) in row_scales.iter_mut().enumerate() {
        let mut amax = 0.0_f32;
        for c in 0..mask.cols {
            if mask.get(r, c) {
                let v = vals.get(r * mask.cols + c).copied().unwrap_or(0.0).abs();
                if v > amax {
                    amax = v;
                }
            }
        }
        let scale = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        *scale_slot = scale;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for c in 0..mask.cols {
            if mask.get(r, c) {
                let v = vals.get(r * mask.cols + c).copied().unwrap_or(0.0);
                #[expect(clippy::cast_possible_truncation, reason = "clamped symmetric int8 quantization")]
                let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
                packed_i8.push(q);
            }
        }
    }
    PackedBitmaskQ8 { mask: mask.clone(), row_scales, packed_i8 }
}

/// Unpack a compressed `PackedBitmaskQ8` matrix back to dense f32 for parity / reference verification.
#[must_use]
pub fn unpack_bitmask_q8(p: &PackedBitmaskQ8) -> Vec<f32> {
    let mut out = vec![0.0_f32; p.mask.total()];
    let mut cursor = 0_usize;
    for r in 0..p.mask.rows {
        let scale = p.row_scales.get(r).copied().unwrap_or(0.0);
        for c in 0..p.mask.cols {
            if p.mask.get(r, c) {
                let q = p.packed_i8.get(cursor).copied().unwrap_or(0);
                cursor += 1;
                if let Some(slot) = out.get_mut(r * p.mask.cols + c) {
                    *slot = f32::from(q) * scale;
                }
            }
        }
    }
    out
}

/// Exact job-conditioned trimmed vocabulary table (`embed_tokens` or `lm_head`).
#[derive(Clone, Debug)]
pub struct TrimmedVocab {
    pub vocab_size: usize,
    pub hidden: usize,
    pub kept_ids: Vec<u32>,
    pub row_index: BTreeMap<u32, usize>,
    pub bf16_bytes: Vec<u8>,
}

impl TrimmedVocab {
    #[must_use]
    pub const fn original_bytes(&self) -> usize {
        self.vocab_size * self.hidden * 2
    }
    #[must_use]
    pub fn trimmed_bytes(&self) -> usize {
        self.bf16_bytes.len()
    }
    #[must_use]
    pub fn saved_bytes(&self) -> usize {
        self.original_bytes().saturating_sub(self.trimmed_bytes())
    }
    /// Look up token `id` from the trimmed cartridge table, falling back seamlessly to `open_ref`
    /// (untouched reference open weights on host) if `id` was outside the trimmed active set.
    #[must_use]
    pub fn lookup_row_with_fallback(&self, id: u32, open_bf16: &[u8]) -> Option<Vec<f32>> {
        let row_bytes = self.hidden * 2;
        if let Some(&idx) = self.row_index.get(&id) {
            let off = idx * row_bytes;
            return self.bf16_bytes.get(off..off + row_bytes).map(decode_bf16);
        }
        let off = (id as usize).checked_mul(row_bytes)?;
        open_bf16.get(off..off + row_bytes).map(decode_bf16)
    }
}

/// Trim a `[vocab_size, hidden]` BF16 token table down to `active_ids` (exact, zero approximation).
#[must_use]
pub fn trim_vocab_bf16(open_bf16: &[u8], vocab_size: usize, hidden: usize, active_ids: &BTreeSet<u32>) -> TrimmedVocab {
    let row_bytes = hidden * 2;
    let mut kept_ids = Vec::with_capacity(active_ids.len());
    let mut row_index = BTreeMap::new();
    let mut bf16_bytes = Vec::with_capacity(active_ids.len() * row_bytes);
    for &id in active_ids {
        let uid = id as usize;
        if uid >= vocab_size {
            continue;
        }
        let off = uid * row_bytes;
        if let Some(slice) = open_bf16.get(off..off + row_bytes) {
            let idx = kept_ids.len();
            kept_ids.push(id);
            row_index.insert(id, idx);
            bf16_bytes.extend_from_slice(slice);
        }
    }
    TrimmedVocab { vocab_size, hidden, kept_ids, row_index, bf16_bytes }
}

/// Memory and sparsity summary across a `ParamMask`.
#[derive(Clone, Debug, Default)]
pub struct StorageReport {
    pub total_weights: usize,
    pub kept_weights: usize,
    pub removed_weights: usize,
    pub sparsity: f64,
    pub dense_bf16_bytes: usize,
    pub dense_q8_bytes: usize,
    pub bitmask_q8_bytes: usize,
    pub mean_k32_tile_skip: f64,
    pub vocab_orig_bytes: usize,
    pub vocab_trimmed_bytes: usize,
}

/// Compute exact storage and `1x32` K-tile skip metrics for `mask`.
#[must_use]
pub fn estimate_storage(mask: &ParamMask, vocab_size: usize, hidden: usize, tied_head: bool) -> StorageReport {
    let mut rep = StorageReport::default();
    let mut skip_sum = 0.0_f64;
    let mut skip_n = 0_usize;
    for bm in mask.tensors.values() {
        let tot = bm.total();
        let kept = bm.kept_count();
        rep.total_weights += tot;
        rep.kept_weights += kept;
        rep.removed_weights += tot.saturating_sub(kept);
        rep.dense_bf16_bytes += tot * 2;
        rep.dense_q8_bytes += tot + bm.rows * 2;
        rep.bitmask_q8_bytes += bm.bits.len() + bm.rows * 2 + kept;
        skip_sum += bm.k_tile_skip_ratio(32);
        skip_n += 1;
    }
    #[expect(clippy::cast_precision_loss, reason = "report ratios")]
    {
        rep.sparsity = if rep.total_weights > 0 { rep.removed_weights as f64 / rep.total_weights as f64 } else { 0.0 };
        rep.mean_k32_tile_skip = if skip_n > 0 { skip_sum / skip_n as f64 } else { 0.0 };
    }
    let tables = if tied_head { 1 } else { 2 };
    rep.vocab_orig_bytes = tables * vocab_size * hidden * 2;
    let active_in = if mask.tokens_in.is_empty() { vocab_size } else { mask.tokens_in.len().min(vocab_size) };
    let active_out = if mask.tokens_out.is_empty() { active_in } else { mask.tokens_out.len().min(vocab_size) };
    rep.vocab_trimmed_bytes = if tied_head {
        let mut union = mask.tokens_in.clone();
        union.extend(&mask.tokens_out);
        let n = if union.is_empty() { vocab_size } else { union.len().min(vocab_size) };
        n * hidden * 2
    } else {
        (active_in + active_out) * hidden * 2
    };
    rep
}

/// Section check (`nmengn.pmap`): verifies Wanda scoring, relevance-head pinning, bidirectional
/// surgical removal and addition (`remove` + `add` round-trip), `bitmask-q8` pack/unpack parity,
/// per-output bias compensation, exact vocabulary trimming with open-weights fallback, and JSON
/// round-trip of `ParamRecord`.
/// # Errors
/// First invariant failure.
pub fn check() -> Result<(), String> {
    // 1. Wanda scoring + attention relevance head pinning.
    // 4 rows x 4 cols (`head_dim = 2`, head 0 = rows 0..2 pinned, head 1 = rows 2..4 unpinned).
    let weights = vec![
        0.1, 0.2, 0.3, 0.4, // row 0 (pinned head 0)
        0.5, 0.6, 0.7, 0.8, // row 1 (pinned head 0)
        0.01, 2.0, 0.02, 3.0, // row 2 (unpinned: cols 0 & 2 lowest)
        4.0, 0.01, 5.0, 0.02, // row 3 (unpinned: cols 1 & 3 lowest)
    ];
    let norms = vec![1.0, 1.0, 1.0, 1.0];
    let mut pinned = BTreeMap::new();
    pinned.insert(0, vec![0]);
    let tname = "model.layers.0.self_attn.q_proj.weight";
    let (pin_r, pin_c) = pinned_axes_for_tensor(tname, 2, &pinned);
    let m1 = wanda_mask_for_matrix(&weights, 4, 4, &norms, 0.5, (&pin_r, &pin_c));
    // Pinned head 0 (rows 0, 1) must be 100% kept.
    for r in 0..2 {
        for c in 0..4 {
            if !m1.get(r, c) {
                return Err(format!("nmengn.pmap: pinned relevance head weight ({r},{c}) was pruned"));
            }
        }
    }
    // Unpinned rows 2 & 3 should prune their 2 lowest weights each.
    if m1.get(2, 0) || m1.get(2, 2) || !m1.get(2, 1) || !m1.get(2, 3) {
        return Err("nmengn.pmap: wanda row 2 selection mismatch".into());
    }
    if !m1.get(3, 0) || m1.get(3, 1) || !m1.get(3, 2) || m1.get(3, 3) {
        return Err("nmengn.pmap: wanda row 3 selection mismatch".into());
    }

    // 2. Bidirectional surgical remove + add round-trip.
    let mut full_mask = ParamMask { pinned_heads: pinned.clone(), ..ParamMask::default() };
    full_mask.tensors.insert(tname.to_owned(), BitMask::new(4, 4, true));
    let mut drop_mask = ParamMask::default();
    let mut drop_bm = BitMask::new(4, 4, false);
    drop_bm.set(0, 0, true); // pinned: must be protected!
    drop_bm.set(2, 0, true); // unpinned: should be removed
    drop_bm.set(2, 2, true); // unpinned: should be removed
    drop_mask.tensors.insert(tname.to_owned(), drop_bm);

    let (after_remove, rem_delta) = surgical_remove(&full_mask, &drop_mask, 2);
    if rem_delta.changed_weights != 2 || !after_remove.tensors[tname].get(0, 0) || after_remove.tensors[tname].get(2, 0) {
        return Err(format!("nmengn.pmap: surgical_remove failed (changed={})", rem_delta.changed_weights));
    }
    let (after_add, add_delta) = surgical_add(&after_remove, &full_mask);
    if add_delta.changed_weights != 2 || after_add.tensors[tname] != full_mask.tensors[tname] {
        return Err(format!("nmengn.pmap: surgical_add recovery failed (changed={})", add_delta.changed_weights));
    }

    // 3. Per-output bias compensation.
    let mean = vec![2.0, 1.0, 3.0, 1.0];
    let comp = bias_compensation(&weights, 4, 4, &after_remove.tensors[tname], &mean);
    let expected_r2 = 0.01 * 2.0 + 0.02 * 3.0;
    if (comp[2] - expected_r2).abs() > 1e-4 || comp[0].abs() > 1e-6 {
        return Err(format!("nmengn.pmap: bias compensation mismatch: {comp:?}"));
    }

    // 4. Bitmask-Q8 pack/unpack parity and compression check.
    let bf16_bytes: Vec<u8> = weights.iter().flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes()).collect();
    let packed = pack_bitmask_q8(&bf16_bytes, &m1);
    let unpacked = unpack_bitmask_q8(&packed);
    if unpacked[2 * 4] != 0.0 || (unpacked[2 * 4 + 1] - 2.0).abs() > 0.05 || (unpacked[3 * 4 + 2] - 5.0).abs() > 0.08 {
        return Err(format!("nmengn.pmap: bitmask-q8 unpack mismatch: {unpacked:?}"));
    }
    let gpu_bm = packed.to_gpu_layout();
    if gpu_bm.rows != 4 || gpu_bm.cols != 4 || gpu_bm.packed_i8.len() != packed.packed_i8.len() {
        return Err("nmengn.pmap: GpuBitmaskQ8 layout mismatch".into());
    }
    let owl = owl_layer_sparsities(&[0.10, 0.02, 0.06], 0.35, 0.08);
    if owl.len() != 3 || owl[0] >= owl[1] || (owl.iter().sum::<f64>() / 3.0 - 0.35).abs() > 1e-6 {
        return Err(format!("nmengn.pmap: owl_layer_sparsities failed: {owl:?}"));
    }

    // 5. Exact vocabulary trimming + open-weights fallback.
    let vocab_bf16: Vec<u8> = (0_u16..16)
        .flat_map(|i| {
            let f = f32::from(i) + 1.0;
            ((f.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect(); // vocab=4, hidden=4
    let active: BTreeSet<u32> = [0, 2].into_iter().collect();
    let trimmed = trim_vocab_bf16(&vocab_bf16, 4, 4, &active);
    if trimmed.trimmed_bytes() != 16 || trimmed.saved_bytes() != 16 {
        return Err("nmengn.pmap: vocab trim byte count mismatch".into());
    }
    let hit = trimmed.lookup_row_with_fallback(2, &vocab_bf16).ok_or("nmengn.pmap: active vocab lookup failed")?;
    let fb = trimmed.lookup_row_with_fallback(3, &vocab_bf16).ok_or("nmengn.pmap: fallback vocab lookup failed")?;
    if (hit[0] - 9.0).abs() > 1e-3 || (fb[0] - 13.0).abs() > 1e-3 {
        return Err(format!("nmengn.pmap: vocab lookup values wrong hit={hit:?} fb={fb:?}"));
    }

    // 6. ParamRecord JSON round-trip & node-id stream parsing.
    if Stream::from_node_id("nmengn.components#L0.q") != Some((0, Stream::Attn))
        || Stream::from_node_id("nmengn.components#L27.down") != Some((27, Stream::Down))
    {
        return Err("nmengn.pmap: Stream::from_node_id failed on qualified node id".into());
    }
    let mut rec = ParamRecord { model: "qwen3-0.6b".into(), jobs: BTreeMap::new() };
    let js = rec.jobs.entry("drone".into()).or_default();
    js.tokens_in.insert(42);
    js.tokens_out.insert(7);
    js.layers.entry(0).or_default().entry(Stream::Attn).or_default().add_rows(4, &[1.0, 2.0, 3.0, 4.0]);
    let round = ParamRecord::from_value(&rec.to_value())?;
    if round.model != "qwen3-0.6b" || !round.jobs["drone"].tokens_in.contains(&42) || round.jobs["drone"].layers[&0][&Stream::Attn].tokens != 1 {
        return Err("nmengn.pmap: ParamRecord JSON round-trip failed".into());
    }
    Ok(())
}
