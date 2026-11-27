//! Drones as dendrites, first form (2026-11-10; architecture and directions by Luis Jake Gabriel III,
//! guide entry of 14:52 UTC; spec `docs/2026-11-10-neumor-spec-octopus-to-neuromorph.md` 4.1).
//! A drone takes one piece of raw input (a chunk of code, or the request) and runs the released
//! Qwen3 components in order on that piece alone: token embedding, then layers 0 to k (input
//! RMSNorm, Q, K, V, QK-Norm, RoPE at positions within the piece, causal grouped-query attention
//! over the piece, output projection, residual, post-attention RMSNorm, SwiGLU, residual).
//!
//! 1. The drone assembly (`drone_assembly`) is the reference assembly cut after layer k: the
//!    sources, the embedding and every component of layers 0 to k, nothing above (no final norm,
//!    no head). It binds the same population, so every cluster and every device copy is shared
//!    by reference with every other model of that population; a drone holds no weights of its own
//!    and loads only the clusters of layers 0 to k.
//! 2. Output (`Local`): the hidden states after layer k's second residual, one row per token of the
//!    piece, held in RAM only (never written to disk; redone when the code changes).
//! 3. Signals (`Affine`): affinity of a piece to the request from the drone representations, by
//!    late interaction (the `MaxSim` rule of `ColBERT`, Khattab and Zaharia 2020) over contextual
//!    states: every state is standardised per dimension with the mean and spread over all states
//!    of the pieces taken in (removes the shared direction and the few very large dimensions of
//!    early-layer states, Mu and Viswanath 2018), then unit-normalised; for each request position
//!    the maximum cosine over the piece's positions; then the rarity-weighted mean over request
//!    positions (rarity of the position's token id, as in `nmpod::intake`). Position 0 of every
//!    piece and of the request is left out of the affinity when the piece is longer than one
//!    token: from the early layers on its state is the attention sink (Xiao et al. 2023) and
//!    carries no content of its own. The lexical signals of `nmpod::intake` stay beside it.
//! 4. The pool (`Pool`): one drone per piece; pieces run in batches whose size is derived from the
//!    free device memory and the measured piece lengths (`batch_from_free`). In this form the
//!    engine runs one sequence per call, so the pieces of a batch go to the card one after another
//!    and the batch is the unit of residency (outputs held before reduction).
//!
//! Depth k: derived by measurement (part test DR-2, `crates/nmapi/tests/drone_dendrite_dr.rs`);
//! `FALLBACK_DEPTH` when no measurement is given. Off by default (`Mode::Off`); selected with
//! `Mode::parse("dendrites")`.
use nmengn::assembly::{Assembly, nid};
use nmengn::model::values;
use nmengn::{Device, Model, Population, RunOpts};
use nmtype::{Fault, FaultKind};
use std::collections::BTreeSet;
use std::sync::Arc;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmapi.drone_dendrite", FaultKind::Execution, m)
}

/// Drone depth used when no measurement is given (layers 0 to 2). Measured choice: see the
/// dated DR-2 line in `docs/2026-11-10-neumor-plan-whole-architecture.md`.
pub const FALLBACK_DEPTH: usize = 2;

/// Whether dendrite drones run. Off unless selected.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    #[default]
    Off,
    Dendrites,
}
impl Mode {
    /// `off` or `dendrites` (also `on`).
    /// # Errors
    /// An unknown word.
    pub fn parse(s: &str) -> Result<Self, Fault> {
        match s {
            "off" => Ok(Self::Off),
            "dendrites" | "on" => Ok(Self::Dendrites),
            x => Err(Fault::new("nmapi.drone_dendrite", FaultKind::Config, format!("drone mode {x}: off or dendrites"))),
        }
    }
}

/// The node whose output is the hidden state after layer `k`.
#[must_use]
pub fn output_node(k: usize) -> (String, String) {
    (nid(&format!("L{k}.res2")), "y".to_owned())
}

/// The reference assembly cut after layer `k`: sources, embedding and layers 0 to k only.
/// # Errors
/// `k` past the assembly's layers, or the cut fails the port check.
pub fn drone_assembly(reference: &Assembly, k: usize) -> Result<Assembly, Fault> {
    let keep = |n: &nmengn::assembly::Node| match n.layer {
        Some(l) => l <= k,
        None => n.op == "source" || n.op == "embed",
    };
    let nodes: Vec<_> = reference.nodes.iter().filter(|n| keep(n)).cloned().collect();
    let out = output_node(k).0;
    if !nodes.iter().any(|n| n.id == out) {
        return Err(bad(format!("assembly {} has no layer {k} output", reference.name)));
    }
    let ids: BTreeSet<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
    let edges = reference.edges.iter().filter(|e| ids.contains(e.from.as_str()) && ids.contains(e.to.as_str())).cloned().collect();
    let a = Assembly { name: format!("{}-drone-{k}", reference.name), nodes, edges, names: std::collections::BTreeMap::new() };
    a.check_ports()?;
    Ok(a)
}

/// A drone's local representation: hidden states after layer k, `tokens` rows of `width`.
#[derive(Clone, Debug, Default)]
pub struct Local {
    pub tokens: usize,
    pub width: usize,
    pub states: Vec<f32>,
}

/// A drone: the cut assembly bound to the shared population, run on one device.
#[derive(Debug)]
pub struct Dendrite {
    pub model: Model,
    pub k: usize,
    pub device: Device,
}
impl Dendrite {
    /// Bind the drone assembly of depth `k` (cut from `reference`) to `pop`.
    /// # Errors
    /// The cut or the binding fails.
    pub fn new(pop: &Arc<Population>, reference: &Assembly, k: usize, device: Device) -> Result<Self, Fault> {
        Ok(Self { model: Model::bind(pop, drone_assembly(reference, k)?)?, k, device })
    }
    /// Run the piece `ids` alone (positions 0..) through layer k; the hidden states after each
    /// layer in `depths` (each at most k), in that order.
    /// # Errors
    /// Engine faults or a depth past k.
    pub fn sense_at(&self, ids: &[i32], depths: &[usize]) -> Result<Vec<Local>, Fault> {
        if let Some(d) = depths.iter().find(|d| **d > self.k) {
            return Err(bad(format!("depth {d} past the drone's {}", self.k)));
        }
        let targets: Vec<(String, String)> = depths.iter().map(|d| output_node(*d)).collect();
        let out = self.model.forward(self.device, ids, &targets, &RunOpts::default())?;
        let width = self.model.pop.dims.hidden;
        targets
            .iter()
            .map(|t| Ok(Local { tokens: ids.len(), width, states: values(out.get(t).ok_or_else(|| bad("no drone output"))?)?.to_vec() }))
            .collect()
    }
    /// Run the piece through layer k (the drone's output).
    /// # Errors
    /// As `sense_at`.
    pub fn sense(&self, ids: &[i32]) -> Result<Local, Fault> {
        self.sense_at(ids, &[self.k])?.pop().ok_or_else(|| bad("no drone output"))
    }
}

/// Device bytes one piece of `tokens` needs while it runs (activations of one layer at a time:
/// hidden, normed, residuals, query, key, value, attention output, gate, up, product, plus the
/// score matrix), at f32.
#[must_use]
pub fn piece_bytes(d: &nmengn::Dims, tokens: usize) -> u64 {
    let per_tok = 6 * d.hidden + (d.heads + 2 * d.kv_heads) * d.head_dim + d.heads * d.head_dim + 3 * d.inter;
    let t = tokens as u64;
    4 * (t * per_tok as u64 + d.heads as u64 * t * t)
}

/// Batch size: how many pieces of the given lengths fit half of `free` device bytes (the other
/// half is left to weights brought in on demand and other users of the card); at least 1. With
/// no device (`None`) the batch is the processor core count.
#[must_use]
pub fn batch_from_free(d: &nmengn::Dims, free: Option<u64>, lengths: &[usize]) -> usize {
    let Some(free) = free else { return nmtype::budget::total() };
    let mut sorted = lengths.to_vec();
    sorted.sort_unstable();
    let typical = sorted.get(sorted.len() / 2).copied().unwrap_or(1).max(1);
    usize::try_from((free / 2) / piece_bytes(d, typical).max(1)).unwrap_or(1).max(1)
}

/// The drone pool of one tentacle: one drone per piece, run in batches.
#[derive(Debug)]
pub struct Pool<'a> {
    pub dendrite: &'a Dendrite,
    pub batch: usize,
}
impl<'a> Pool<'a> {
    /// A pool over `pieces`, its batch derived from the free device memory now.
    #[must_use]
    pub fn sized(dendrite: &'a Dendrite, pieces: &[Vec<i32>]) -> Self {
        let free = if dendrite.device == Device::Gpu { dendrite.model.pop.gpu_free() } else { None };
        let lengths: Vec<usize> = pieces.iter().map(Vec::len).collect();
        Self { dendrite, batch: batch_from_free(&dendrite.model.pop.dims, free, &lengths) }
    }
    /// Every piece through its drone; `each` receives (piece index, outputs at `depths`) per
    /// piece as each batch completes. Nothing is dropped.
    /// # Errors
    /// The first engine fault.
    pub fn run(
        &self,
        pieces: &[Vec<i32>],
        depths: &[usize],
        mut each: impl FnMut(usize, Vec<Local>) -> Result<(), Fault>,
    ) -> Result<(), Fault> {
        for (b, group) in pieces.chunks(self.batch.max(1)).enumerate() {
            let mut held = Vec::with_capacity(group.len());
            for ids in group {
                held.push(self.dendrite.sense_at(ids, depths)?);
            }
            for (j, out) in held.into_iter().enumerate() {
                each(b * self.batch.max(1) + j, out)?;
            }
        }
        Ok(())
    }
}

/// Per-dimension mean and inverse spread over all states taken in (RAM only).
#[derive(Clone, Debug, Default)]
pub struct Standard {
    pub mean: Vec<f32>,
    pub inv: Vec<f32>,
}
impl Standard {
    /// From running sums over `n` rows: `sum` and `sq` per dimension.
    #[must_use]
    pub fn from_sums(sum: &[f64], sq: &[f64], n: u64) -> Self {
        let n = (n.max(1)) as f64;
        let mean: Vec<f64> = sum.iter().map(|s| s / n).collect();
        let inv = sq
            .iter()
            .zip(&mean)
            .map(|(q, m)| {
                let var = (q / n - m * m).max(0.0);
                if var > 0.0 { (1.0 / var.sqrt()) as f32 } else { 0.0 }
            })
            .collect();
        Self { mean: mean.into_iter().map(|m| m as f32).collect(), inv }
    }
    /// Standardised, unit-normalised rows of `l`, leaving out position 0 when there are more.
    #[must_use]
    pub fn units(&self, l: &Local) -> Vec<f32> {
        let skip = usize::from(l.tokens > 1);
        let mut out = Vec::with_capacity((l.tokens - skip.min(l.tokens)) * l.width);
        for row in l.states.chunks_exact(l.width).skip(skip) {
            let z: Vec<f32> = row.iter().zip(&self.mean).zip(&self.inv).map(|((x, m), i)| (x - m) * i).collect();
            let n = z.iter().map(|x| x * x).sum::<f32>().sqrt();
            out.extend(z.iter().map(|x| if n > 0.0 { x / n } else { 0.0 }));
        }
        out
    }
}

/// Add the rows of `l` to running sums (for `Standard`).
pub fn accumulate(l: &Local, sum: &mut [f64], sq: &mut [f64]) -> u64 {
    for row in l.states.chunks_exact(l.width) {
        for ((s, q), x) in sum.iter_mut().zip(sq.iter_mut()).zip(row) {
            let x = f64::from(*x);
            *s += x;
            *q += x * x;
        }
    }
    l.tokens as u64
}

/// The request prepared for the drone affinity: unit rows and their rarity weights.
#[derive(Clone, Debug)]
pub struct Affine {
    pub width: usize,
    pub rows: Vec<f32>,
    pub weights: Vec<f32>,
}
impl Affine {
    /// From the request's unit rows (`Standard::units`) and the weight of each row.
    #[must_use]
    pub fn new(width: usize, rows: Vec<f32>, weights: Vec<f32>) -> Self {
        Self { width, rows, weights }
    }
    /// Affinity of a piece given by its unit rows: per request row the maximum cosine over the
    /// piece rows (floored at 0), weighted mean over request rows.
    #[must_use]
    pub fn score(&self, piece: &[f32]) -> f32 {
        let q = self.weights.len();
        let mut best = vec![f32::MIN; q];
        for p in piece.chunks_exact(self.width) {
            for (b, r) in best.iter_mut().zip(self.rows.chunks_exact(self.width)) {
                let c = dot(p, r);
                if c > *b {
                    *b = c;
                }
            }
        }
        let w: f32 = self.weights.iter().sum();
        if w <= 0.0 || piece.is_empty() {
            return 0.0;
        }
        best.iter().zip(&self.weights).map(|(b, w)| b.max(0.0) * w).sum::<f32>() / w
    }
}

impl Affine {
    /// Per request row, the maximum cosine over the piece rows (`f32::MIN` for an empty piece).
    #[must_use]
    pub fn row_max(&self, piece: &[f32]) -> Vec<f32> {
        let mut best = vec![f32::MIN; self.weights.len()];
        for p in piece.chunks_exact(self.width) {
            for (b, r) in best.iter_mut().zip(self.rows.chunks_exact(self.width)) {
                let c = dot(p, r);
                if c > *b {
                    *b = c;
                }
            }
        }
        best
    }
}

/// Background-relative affinity (2026-11-10), the dendrite's sensory adaptation: a receptor reports
/// its deviation from the ambient level, not the level. For each request row, the expected
/// maximum cosine of a piece is fitted over every piece scored as `a + b ln(rows)` (the maximum
/// of more draws is larger, so long pieces would otherwise win by chance), and a piece scores the
/// residual in units of the residual spread, floored at 0 (only where it stands out); then the
/// weighted mean over request rows. Engineering reference: background calibration of similarity
/// against hubness, CSLS (Conneau et al. 2018, `MUSE`, method only). `maxes[i]` holds piece i's
/// `row_max`, `rows[i]` its unit-row count; empty pieces score 0.
#[must_use]
pub fn calibrated(maxes: &[Vec<f32>], rows: &[usize], weights: &[f32]) -> Vec<f32> {
    let n = maxes.len();
    let mut out = vec![0.0f32; n];
    let wsum: f32 = weights.iter().sum();
    if n < 3 || wsum <= 0.0 {
        return out;
    }
    let x: Vec<f64> = rows.iter().map(|r| ((*r).max(1) as f64).ln()).collect();
    let live = |i: usize, r: usize| -> Option<(f64, f64)> {
        let m = maxes.get(i)?.get(r).copied()?;
        (rows.get(i).copied().unwrap_or(0) > 0 && m > f32::MIN).then(|| (x.get(i).copied().unwrap_or(0.0), f64::from(m)))
    };
    for (r, w) in weights.iter().enumerate() {
        let pts: Vec<(f64, f64)> = (0..n).filter_map(|i| live(i, r)).collect();
        if pts.len() < 3 {
            continue;
        }
        let k = pts.len() as f64;
        let (mx, my) = (pts.iter().map(|p| p.0).sum::<f64>() / k, pts.iter().map(|p| p.1).sum::<f64>() / k);
        let sxx: f64 = pts.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
        let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
        let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
        let fit = |xv: f64| my + slope * (xv - mx);
        let sd = (pts.iter().map(|p| (p.1 - fit(p.0)).powi(2)).sum::<f64>() / k).sqrt();
        if sd <= 0.0 {
            continue;
        }
        for (i, o) in out.iter_mut().enumerate() {
            if let Some((xv, m)) = live(i, r) {
                *o += w * (((m - fit(xv)) / sd).max(0.0) as f32) / wsum;
            }
        }
    }
    out
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (ca, cb) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    let (ra, rb) = (ca.1, cb.1);
    for (x, y) in ca.0.iter().zip(cb.0) {
        for i in 0..8 {
            if let (Some(s), Some(u), Some(v)) = (acc.get_mut(i), x.get(i), y.get(i)) {
                *s += u * v;
            }
        }
    }
    acc.iter().sum::<f32>() + ra.iter().zip(rb).map(|(x, y)| x * y).sum::<f32>()
}

// ---------------------------------------------------------------------------------------------
// The drone signals (2026-11-19, lane DR; master plan 3.2 and Phase C2; design as written 01 A5:
// sense, recognise context, classify evidence, attend locally, check reflexively, detect
// anomalies, judge confidence, local memory). Two halves:
// - `Reading`: request-independent, made once per piece and material version at intake from the
//   piece's own pass through layers 0 to k (`sense_states`) beside its lexical record; kept in
//   the piece's local memory (`key_scan::PieceMemory::reading`). Off unless `NANO_DRONE_SIGNALS=on`.
// - `attention`: request-dependent, the request drone's attention to a piece at depth k (the
//   request's queries at layer k against the piece's layer-k keys, the piece encoded alone and
//   placed just before the request, log-mean-exp over the piece's positions, content tokens of
//   the request, posed minus content-free, mean over layer k's query heads). Layer-k states are
//   not stored (master plan 3.2): the piece's layers 0 to k are recomputed for the pieces scored.
// `piece_values` joins both into one value per piece on the log-odds scale (`kind_column` is the
// hook the fused value W2 calls). Constants below are stated, not fitted; DR-2 and the flow
// measurement settle k and them.
// ---------------------------------------------------------------------------------------------

/// The drone signals' switch for the intake: on by default since 2026-11-19 (lead's word, with the
/// `drone` kind in the fused value); `NANO_DRONE_SIGNALS=off` restores.
#[must_use]
pub fn signals_on() -> bool {
    !std::env::var("NANO_DRONE_SIGNALS").is_ok_and(|v| v == "off")
}

/// The drone depth k in use: `NANO_DRONE_K`, else `FALLBACK_DEPTH` (Luis 2026-11-19: build with
/// the fallback; DR-2 fixes k after the build).
#[must_use]
pub fn depth() -> usize {
    std::env::var("NANO_DRONE_K").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(FALLBACK_DEPTH)
}

/// How many of the fused value's leading pieces the request drone's attention reads per request
/// (`NANO_DRONE_POOL`, default 48: DR-2's rule looks at the top 50).
#[must_use]
pub fn attention_pool() -> usize {
    std::env::var("NANO_DRONE_POOL").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(48)
}

/// A token stands out when the norm of its standardised layer-k state, over the square root of
/// the width (about 1 on average), exceeds this.
pub const STANDOUT_Z: f32 = 2.0;
/// Standout positions kept per piece.
pub const STANDOUT_KEEP: usize = 8;
/// Pieces shorter than this many tokens are read with less confidence (in proportion).
pub const CONFIDENT_TOKENS: usize = 32;
/// Confidence of a piece read lexically only (cold: no pass through layers 0 to k yet).
pub const COLD_CONFIDENCE: f32 = 0.3;
/// The clip of the attention's standard units in the per-piece value.
pub const VALUE_CLIP: f32 = 3.0;

/// The epistemic class of a piece (classify evidence): what kind of evidence it can give.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Epistemic {
    #[default]
    Unknown,
    /// Code whose first statement opens a definition (function, type, constant, module).
    Definition,
    /// Code that uses what is defined elsewhere.
    Use,
    /// Tests and their fixtures.
    Test,
    /// Prose: documents, notes, comments only.
    Documentation,
    /// Settings and data files.
    Configuration,
    /// Repetitive or generated text (the anomaly signal high).
    Generated,
}
impl Epistemic {
    /// The class's name in traces.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Definition => "definition",
            Self::Use => "use",
            Self::Test => "test",
            Self::Documentation => "documentation",
            Self::Configuration => "configuration",
            Self::Generated => "generated",
        }
    }
    /// The class from the piece's path, text, document flag and anomaly.
    #[must_use]
    pub fn classify(path: &str, text: &str, document: bool, anomaly: f32) -> Self {
        const OPENS: [&str; 22] = [
            "fn ", "struct ", "enum ", "trait ", "impl ", "impl<", "mod ", "type ", "const ", "static ", "macro_rules!", "union ", "class ",
            "def ", "async def ", "interface ", "func ", "function ", "export ", "module ", "package ", "namespace ",
        ];
        const PREFIX: [&str; 12] =
            ["pub(crate) ", "pub(super) ", "pub ", "async ", "unsafe ", "extern \"C\" ", "const ", "export ", "default ", "public ", "private ", "static "];
        let p = path.replace('\\', "/").to_ascii_lowercase();
        let ext = p.rsplit_once('.').map_or("", |x| x.1);
        let file = p.rsplit('/').next().unwrap_or("");
        if anomaly >= 0.8 {
            return Self::Generated;
        }
        if matches!(ext, "toml" | "json" | "yaml" | "yml" | "ini" | "cfg" | "lock" | "xml" | "csv") {
            return Self::Configuration;
        }
        if document || matches!(ext, "md" | "txt" | "rst" | "adoc" | "org") {
            return Self::Documentation;
        }
        if p.contains("/tests/")
            || p.starts_with("tests/")
            || file.starts_with("test_")
            || file.contains("_test.")
            || file.contains(".test.")
            || text.contains("#[test]")
            || text.contains("#[cfg(test)]")
            || text.contains("def test_")
            || text.contains("@Test")
        {
            return Self::Test;
        }
        let first = text.lines().map(str::trim).find(|l| {
            !(l.is_empty()
                || l.starts_with("//")
                || l.starts_with('#')
                || l.starts_with("/*")
                || l.starts_with('*')
                || l.starts_with("--")
                || l.starts_with('@'))
        });
        let Some(first) = first else { return Self::Documentation };
        let bare = PREFIX.iter().fold(first, |l, w| l.strip_prefix(w).unwrap_or(l));
        if OPENS.iter().any(|o| bare.starts_with(o) || first.starts_with(o)) { Self::Definition } else { Self::Use }
    }
}

/// Running per-dimension sums of every layer-k state the intake has seen (for `Standard`; the
/// drones' adaptive processing: a state is read against the material's own background).
#[derive(Clone, Debug, Default)]
pub struct Background {
    pub sum: Vec<f64>,
    pub sq: Vec<f64>,
    pub rows: u64,
}
impl Background {
    /// Add a piece's states and return the standardisation over everything seen so far.
    pub fn take_in(&mut self, l: &Local) -> Standard {
        if self.sum.len() != l.width {
            self.sum = vec![0.0; l.width];
            self.sq = vec![0.0; l.width];
            self.rows = 0;
        }
        self.rows += accumulate(l, &mut self.sum, &mut self.sq);
        Standard::from_sums(&self.sum, &self.sq, self.rows)
    }
}

/// One piece's drone reading (request-independent; the piece's last reading in its local
/// memory). Signals: kind, epistemic class, standout flags, anomaly, confidence.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Reading {
    /// The depth read (layers 0 to `depth`), and whether that pass ran (false: lexical only, cold).
    pub depth: usize,
    pub sensed: bool,
    pub tokens: usize,
    /// The piece's kind as the material names it (code, document, ...), and its class.
    pub kind: String,
    pub class: Epistemic,
    /// Positions (within the piece) of the tokens that stand out most, at most `STANDOUT_KEEP`,
    /// and the share of the piece's tokens that stand out.
    pub standout: Vec<u16>,
    pub standout_share: f32,
    /// Mean cosine of consecutive standardised states (high: repetitive), the share of distinct
    /// tokens, and the anomaly in 0 to 1 from both.
    pub repetition: f32,
    pub distinct_share: f32,
    pub anomaly: f32,
    /// How far the drone trusts its own reading, 0 to 1.
    pub confidence: f32,
}

/// What a piece is, for its reading: file path, text, kind, share of distinct tokens, whether it
/// is prose, and its token count.
#[derive(Clone, Copy, Debug)]
pub struct PieceFacts<'a> {
    pub path: &'a str,
    pub text: &'a str,
    pub kind: &'a str,
    pub distinct_share: f32,
    pub document: bool,
    pub tokens: usize,
}

/// The piece's hidden states after layer `k`, run alone (positions from 0) on the model in use:
/// the drone's pass over layers 0 to k (the reference cut's output node; the engine runs only
/// what that node needs).
/// # Errors
/// Engine faults, or a model without that node.
pub fn sense_states(model: &Model, dev: Device, ids: &[i32], k: usize) -> Result<Local, Fault> {
    let target = output_node(k);
    let mut out = model.forward(dev, ids, std::slice::from_ref(&target), &RunOpts::default())?;
    let a = out.remove(&target).ok_or_else(|| bad(format!("no layer {k} output")))?;
    let states = host(model, a)?;
    let width = model.pop.dims.hidden;
    if states.len() != ids.len() * width {
        return Err(bad(format!("layer {k} output: {} values for {} tokens", states.len(), ids.len())));
    }
    Ok(Local { tokens: ids.len(), width, states })
}

fn host(model: &Model, a: nmengn::model::Act) -> Result<Vec<f32>, Fault> {
    match a {
        nmengn::model::Act::Cpu(x) => Ok(x),
        nmengn::model::Act::Gpu(t) => model.pop.gpu.as_ref().ok_or_else(|| bad("a device result without a device"))?.download(&t).map_err(bad),
        nmengn::model::Act::Ids(_) => Err(bad("an output is not a tensor")),
    }
}

impl Reading {
    /// The reading of one piece: `local` its states after layer `depth` (None: cold, lexical
    /// only), `background` the material's running standardisation (the piece is added to it).
    #[must_use]
    pub fn read(local: Option<&Local>, background: &mut Background, depth: usize, f: &PieceFacts<'_>) -> Self {
        let lex = ((0.35 - f.distinct_share) / 0.35).clamp(0.0, 1.0);
        #[expect(clippy::cast_precision_loss, reason = "a token count")]
        let size = (f.tokens as f32 / CONFIDENT_TOKENS as f32).min(1.0);
        let Some(l) = local.filter(|l| l.tokens > 0 && l.width > 0) else {
            return Self {
                depth,
                sensed: false,
                tokens: f.tokens,
                kind: f.kind.to_owned(),
                class: Epistemic::classify(f.path, f.text, f.document, lex),
                distinct_share: f.distinct_share,
                anomaly: lex,
                confidence: COLD_CONFIDENCE * size * (1.0 - 0.5 * lex),
                ..Self::default()
            };
        };
        let st = background.take_in(l);
        // Standardised rows (position 0, the attention sink, left out when there are more).
        let skip = usize::from(l.tokens > 1);
        #[expect(clippy::cast_precision_loss, reason = "a width")]
        let root = (l.width as f32).sqrt().max(1.0);
        let mut z: Vec<(usize, f32)> = Vec::with_capacity(l.tokens);
        let mut prev: Option<(Vec<f32>, f32)> = None;
        let (mut cos_sum, mut cos_n) = (0.0_f32, 0_usize);
        for (t, row) in l.states.chunks_exact(l.width).enumerate().skip(skip) {
            let v: Vec<f32> = row.iter().zip(&st.mean).zip(&st.inv).map(|((x, m), i)| (x - m) * i).collect();
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            z.push((t, n / root));
            if let Some((p, pn)) = &prev
                && n > 0.0
                && *pn > 0.0
            {
                cos_sum += dot(p, &v) / (n * pn);
                cos_n += 1;
            }
            prev = Some((v, n));
        }
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let repetition = if cos_n == 0 { 0.0 } else { cos_sum / cos_n as f32 };
        let mut top: Vec<(usize, f32)> = z.iter().copied().filter(|x| x.1 > STANDOUT_Z).collect();
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let standout_share = if z.is_empty() { 0.0 } else { top.len() as f32 / z.len() as f32 };
        top.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let standout: Vec<u16> = top.iter().take(STANDOUT_KEEP).filter_map(|x| u16::try_from(x.0).ok()).collect();
        let anomaly = lex.max(((repetition - 0.6) / 0.4).clamp(0.0, 1.0));
        Self {
            depth,
            sensed: true,
            tokens: l.tokens,
            kind: f.kind.to_owned(),
            class: Epistemic::classify(f.path, f.text, f.document, anomaly),
            standout,
            standout_share,
            repetition,
            distinct_share: f.distinct_share,
            anomaly,
            confidence: size * (1.0 - 0.5 * anomaly),
        }
    }
}

/// The request's queries at layer `k` (every query head of that layer, after the head norm and
/// rope) at positions `start..`, the request run alone with no prefix (its own drone).
/// # Errors
/// Engine faults.
pub fn request_queries(model: &Model, dev: Device, ids: &[i32], start: usize, k: usize) -> Result<nmengn::drone::Queries, Fault> {
    let d = model.pop.dims;
    let heads: Vec<(usize, usize)> = (0..d.heads).map(|h| (k, h)).collect();
    let targets = nmengn::drone::query_targets(model, &heads)?;
    let want: Vec<(String, String)> = targets.iter().map(|t| (t.1.clone(), t.2.clone())).collect();
    let acts = model.forward_at(dev, ids, start, &want, &RunOpts::default())?;
    let mut got: std::collections::BTreeMap<(String, String), Vec<f32>> = std::collections::BTreeMap::new();
    for (key, a) in acts {
        if want.contains(&key) {
            got.insert(key, host(model, a)?);
        }
    }
    let tokens = ids.len();
    let mut q = Vec::with_capacity(heads.len());
    for &(l, h) in &heads {
        let (_, id, port, range) =
            targets.iter().find(|t| t.0 == l && h >= t.3.0 && h < t.3.1).ok_or_else(|| bad(format!("no query output covers layer {l} head {h}")))?;
        let v = got.get(&(id.clone(), port.clone())).ok_or_else(|| bad(format!("{id}.{port} not returned")))?;
        let n = range.1 - range.0;
        if v.len() != tokens * n * d.head_dim {
            return Err(bad(format!("{id}.{port}: {} values for {tokens} tokens of {n} heads", v.len())));
        }
        let local = h - range.0;
        let mut rows = Vec::with_capacity(tokens * d.head_dim);
        for t in 0..tokens {
            rows.extend_from_slice(v.get((t * n + local) * d.head_dim..(t * n + local + 1) * d.head_dim).unwrap_or(&[]));
        }
        q.push(rows);
    }
    Ok(nmengn::drone::Queries { tokens, heads, q })
}

/// A piece's layer-k keys (every key-value head, before rope), the piece run alone through layers
/// 0 to k (positions from 0).
/// # Errors
/// Engine faults.
pub fn piece_keys(model: &Model, dev: Device, ids: &[i32], k: usize) -> Result<nmengn::drone::RelevanceKeys, Fault> {
    let d = model.pop.dims;
    let kv = nmengn::kvstore::encode_through(model, dev, None, ids, k)?;
    let pairs: Vec<(usize, usize)> = (0..d.kv_heads).map(|h| (k, h)).collect();
    nmengn::drone::RelevanceKeys::from_chunk(&kv, &pairs, d.kv_heads, d.head_dim)
}

/// Per query head, the log-mean-exp of a piece's scores (position 0 left out when there are more)
/// averaged over `tokens` of `q`; the piece's keys roped at positions `start - n ..start`.
/// # Errors
/// As `nmengn::drone::scan_pieces`.
pub fn depth_rows(
    q: &nmengn::drone::Queries,
    keys: &Arc<nmengn::drone::RelevanceKeys>,
    table: &[(f32, f32)],
    start: usize,
    tokens: &[usize],
    geometry: (usize, usize),
) -> Result<Vec<f32>, Fault> {
    depth_rows_with(q, keys, table, start, tokens, geometry, nmengn::drone::Reduce::Mean)
}

/// `depth_rows` with the reduction over the piece's positions given (log-mean-exp or
/// log-sum-exp; DR-2 compares both, 2026-11-19).
/// # Errors
/// As `depth_rows`.
#[expect(clippy::too_many_arguments, reason = "queries, keys, rotary table, start, tokens, geometry and reduction are independent inputs")]
pub fn depth_rows_with(
    q: &nmengn::drone::Queries,
    keys: &Arc<nmengn::drone::RelevanceKeys>,
    table: &[(f32, f32)],
    start: usize,
    tokens: &[usize],
    geometry: (usize, usize),
    reduce: nmengn::drone::Reduce,
) -> Result<Vec<f32>, Fault> {
    let n = keys.tokens;
    if n == 0 || n > start {
        return Err(bad(format!("a piece of {n} tokens placed before position {start}")));
    }
    let skip = usize::from(n > 1);
    let half = geometry.0 / 2;
    let tb = table.get((start - n) * half..).unwrap_or(&[]);
    nmengn::drone::scan_pieces(q, std::slice::from_ref(keys), &[(skip, n - skip)], n, tb, geometry, Some(tokens), reduce)
}

/// The request drone's attention to each piece at depth `k`: `request` and `null` (content-free)
/// token ids, `pieces` each piece's ids. Every piece is encoded alone through layers 0 to k; the
/// request's queries are taken at positions after the longest piece (`start`), and each piece's
/// keys roped at the positions just before `start` (as if the piece came right before the
/// request, the exact read's order); per query head the log-mean-exp of the scores over the
/// piece's positions (position 0, the sink, left out) averaged over the request's tokens after
/// its first; posed minus content-free; mean over the heads. NaN for an empty piece. `gate` is
/// called around each device job (the caller enters the device gate in it).
/// # Errors
/// Engine faults.
#[expect(clippy::too_many_arguments, reason = "the model, the request pair, the pieces, the depth and the gate")]
pub fn attention(
    model: &Model,
    dev: Device,
    request: &[i32],
    null: &[i32],
    pieces: &[Vec<i32>],
    k: usize,
    gate: &mut dyn FnMut() -> Option<nmengn::placement::GateGuard>,
) -> Result<Vec<f32>, Fault> {
    let d = model.pop.dims;
    let geometry = (d.head_dim, d.heads / d.kv_heads.max(1));
    let mut keys = Vec::with_capacity(pieces.len());
    for ids in pieces {
        if ids.is_empty() {
            keys.push(None);
            continue;
        }
        let g = gate();
        let r = piece_keys(model, dev, ids, k);
        drop(g);
        keys.push(Some(Arc::new(r?)));
    }
    let start = pieces.iter().map(Vec::len).max().unwrap_or(0).max(1);
    let g = gate();
    let posed = request_queries(model, dev, request, start, k);
    let none = request_queries(model, dev, null, start, k);
    drop(g);
    let (posed, none) = (posed?, none?);
    let table = nmengn::cpu::rope_table(start, d.head_dim, d.theta);
    let tok_p: Vec<usize> = (usize::from(request.len() > 1)..request.len()).collect();
    let tok_n: Vec<usize> = (usize::from(null.len() > 1)..null.len()).collect();
    let mut out = Vec::with_capacity(pieces.len());
    for pk in &keys {
        let Some(pk) = pk else {
            out.push(f32::NAN);
            continue;
        };
        let a = depth_rows(&posed, pk, &table, start, &tok_p, geometry)?;
        let b = depth_rows(&none, pk, &table, start, &tok_n, geometry)?;
        out.push(head_mean(&a, &b));
    }
    Ok(out)
}

/// Mean over heads of posed minus content-free (finite rows only; NaN with none).
#[must_use]
pub fn head_mean(posed: &[f32], null: &[f32]) -> f32 {
    let diffs: Vec<f32> = posed.iter().zip(null).map(|(x, y)| x - y).filter(|x| x.is_finite()).collect();
    #[expect(clippy::cast_precision_loss, reason = "a head count")]
    if diffs.is_empty() { f32::NAN } else { diffs.iter().sum::<f32>() / diffs.len() as f32 }
}

/// One value per piece on the log-odds scale (nats of evidence against the scored pieces'
/// average): the request drone's attention in standard units over the finite scores (clipped to
/// `VALUE_CLIP`), times the piece's confidence (its reading's, `COLD_CONFIDENCE` with none),
/// less its anomaly. NaN where the attention is not finite (the fused value counts it 0); all
/// NaN with fewer than three finite scores.
#[must_use]
pub fn piece_values(readings: &[Option<&Reading>], attention: &[f32]) -> Vec<f32> {
    let xs: Vec<f32> = attention.iter().copied().filter(|x| x.is_finite()).collect();
    if xs.len() < 3 {
        return vec![f32::NAN; attention.len()];
    }
    #[expect(clippy::cast_precision_loss, reason = "a count")]
    let m = xs.iter().sum::<f32>() / xs.len() as f32;
    #[expect(clippy::cast_precision_loss, reason = "a count")]
    let sd = (xs.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / xs.len() as f32).sqrt();
    attention
        .iter()
        .enumerate()
        .map(|(i, a)| {
            if !a.is_finite() {
                return f32::NAN;
            }
            let z = if sd > 0.0 { ((a - m) / sd).clamp(-VALUE_CLIP, VALUE_CLIP) } else { 0.0 };
            let r = readings.get(i).copied().flatten();
            r.map_or(COLD_CONFIDENCE, |r| r.confidence) * z - r.map_or(0.0, |r| r.anomaly)
        })
        .collect()
}

/// The hook for the fused value (lane T1, W2 kind `drone`): one value per piece of `sub` on the
/// log-odds scale, NaN outside the first `attention_pool()` pieces of `lead` (the fused value's
/// leading pieces, best first) or on any fault. Reads the request drone's attention at depth
/// `depth()` (on the card under the device gate as a foreground job) and each piece's reading
/// from `keys` (the intake's local memory; cold pieces count `COLD_CONFIDENCE`).
#[must_use]
pub fn kind_column(model: &Model, keys: Option<&crate::key_scan::Keys>, sub: &nmsubs::Substrate, question: &str, lead: &[usize]) -> Vec<f32> {
    let n = sub.chunks.len();
    let items: Vec<(usize, String, String)> = lead
        .iter()
        .copied()
        .filter(|c| *c < n)
        .take(attention_pool())
        .map(|c| {
            let found = sub.chunks.get(c).and_then(|ch| sub.files.get(ch.file).map(|f| (ch, f)));
            match found {
                Some((ch, f)) => (c, nmsubs::dense::doc_text(&f.path, &ch.text), format!("{}:{}", f.path, ch.start)),
                None => (c, String::new(), String::new()),
            }
        })
        .collect();
    kind_column_owned(model, keys, n, &items, question)
}

/// Owned-input variant of [`kind_column`] so `drone_path` can run the GPU depth read concurrently
/// with CPU `compose` and `tentacle_read` stages without borrowing `&Substrate` across threads.
#[must_use]
pub fn kind_column_owned(
    model: &Model,
    keys: Option<&crate::key_scan::Keys>,
    total_chunks: usize,
    items: &[(usize, String, String)],
    question: &str,
) -> Vec<f32> {
    let mut col = vec![f32::NAN; total_chunks];
    let items = &items[..items.len().min(attention_pool())];
    let dev = if model.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let tok = |s: &str| model.pop.tokenizer.encode(s).ok();
    let (Some(req), Some(null)) = (tok(&format!("\nQuestion: {question}")), tok("\nQuestion: N/A")) else { return col };
    let mut ids = Vec::with_capacity(items.len());
    for (_, text, _) in items {
        if text.is_empty() {
            ids.push(Vec::new());
        } else {
            ids.push(tok(text).unwrap_or_default());
        }
    }
    let mut gate = || (dev == Device::Gpu).then(|| nmengn::placement::gpu_gate().enter("drone depth read", true));
    let Ok(att) = attention(model, dev, &req, &null, &ids, depth(), &mut gate) else { return col };
    let mems: Vec<Option<crate::key_scan::PieceMemory>> = items.iter().map(|(_, _, ident)| keys.and_then(|k| k.piece(ident))).collect();
    let readings: Vec<Option<&Reading>> = mems.iter().map(|m| m.as_ref().and_then(|m| m.reading.as_ref())).collect();
    for ((c, _, _), v) in items.iter().zip(piece_values(&readings, &att)) {
        if let Some(x) = col.get_mut(*c) {
            *x = v;
        }
    }
    col
}

/// Section check: the cut of a two-layer generated assembly at k = 0 keeps the embedding and
/// layer 0 only, passes the port check, and names its output; a cut past the layers fails; the
/// affinity of a piece holding the request row exceeds that of one that does not; off by default.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let a = nmengn::assembly::generate("t", nmengn::Shape { layers: 2, heads: 4, kv_heads: 2, head_dim: 2 }, 1);
    let c = drone_assembly(&a, 0).map_err(|e| e.to_string())?;
    if c.nodes.iter().any(|n| n.layer.is_some_and(|l| l > 0) || n.op == "head" || n.op == "norm" && n.layer.is_none()) {
        return Err("nmapi.drone_dendrite: the cut holds a component above layer 0".into());
    }
    if c.node(&output_node(0).0).is_none() || c.node(&nid("embed")).is_none() {
        return Err("nmapi.drone_dendrite: the cut lacks the embedding or the layer 0 output".into());
    }
    if drone_assembly(&a, 2).is_ok() {
        return Err("nmapi.drone_dendrite: a cut past the layers passed".into());
    }
    let af = Affine::new(2, vec![1.0, 0.0], vec![1.0]);
    if af.score(&[0.0, 1.0, 1.0, 0.0]) <= af.score(&[0.0, 1.0, -1.0, 0.0]) {
        return Err("nmapi.drone_dendrite: affinity order".into());
    }
    if Mode::default() != Mode::Off {
        return Err("nmapi.drone_dendrite: not off by default".into());
    }
    signals_check()
}

/// The drone signals' part of the section check (lane DR, 2026-11-19): classes on hand pieces; a
/// reading of hand states flags the standing-out token, finds a repeated piece anomalous and
/// trusts it less; a cold reading is less confident; the per-piece value orders by attention,
/// counts the anomaly against a piece, is NaN where the attention is missing; off by default.
fn signals_check() -> Result<(), String> {
    let classes = [
        (Epistemic::classify("crates/a/src/x.rs", "/// doc
pub fn read(x: u32) {}", false, 0.0), Epistemic::Definition),
        (Epistemic::classify("crates/a/src/x.rs", "    let y = read(3);
    y + 1", false, 0.0), Epistemic::Use),
        (Epistemic::classify("crates/a/tests/t.rs", "fn t() {}", false, 0.0), Epistemic::Test),
        (Epistemic::classify("README.md", "Notes", true, 0.0), Epistemic::Documentation),
        (Epistemic::classify("Cargo.toml", "[package]", false, 0.0), Epistemic::Configuration),
        (Epistemic::classify("crates/a/src/x.rs", "fn a() {}", false, 0.9), Epistemic::Generated),
    ];
    if let Some((got, want)) = classes.iter().find(|(g, w)| g != w) {
        return Err(format!("nmapi.drone_dendrite: class {} where {} is right", got.name(), want.name()));
    }
    // Hand states, width 4: a varied piece with one standing-out token, and a repeated piece.
    let varied: Vec<f32> = (0..40).flat_map(|t| {
        let x = f32::from(u8::try_from(t % 7).unwrap_or(0));
        if t == 20 { vec![40.0, -40.0, 40.0, -40.0] } else { vec![x, 1.0 - x, (x * 0.7).sin(), (x * 1.3).cos()] }
    }).collect();
    let same: Vec<f32> = (0..40).flat_map(|_| vec![1.0, 2.0, 3.0, 4.0]).collect();
    let mut bg = Background::default();
    let facts = |share: f32| PieceFacts { path: "crates/a/src/x.rs", text: "fn a() {}", kind: "code", distinct_share: share, document: false, tokens: 40 };
    let a = Reading::read(Some(&Local { tokens: 40, width: 4, states: varied }), &mut bg, 2, &facts(0.6));
    let b = Reading::read(Some(&Local { tokens: 40, width: 4, states: same }), &mut bg, 2, &facts(0.05));
    let cold = Reading::read(None, &mut bg, 2, &facts(0.6));
    if !a.sensed || a.standout.first() != Some(&20) || a.class != Epistemic::Definition {
        return Err(format!("nmapi.drone_dendrite: standout {:?} (want 20 first), class {}", a.standout, a.class.name()));
    }
    if b.anomaly <= a.anomaly || b.confidence >= a.confidence || b.class != Epistemic::Generated {
        return Err(format!("nmapi.drone_dendrite: repeated piece anomaly {} confidence {} against {} {}", b.anomaly, b.confidence, a.anomaly, a.confidence));
    }
    if cold.sensed || cold.confidence >= a.confidence {
        return Err("nmapi.drone_dendrite: a cold reading is trusted as much as a sensed one".into());
    }
    let v = piece_values(&[Some(&a), Some(&a), Some(&b), None], &[1.0, 3.0, 3.0, f32::NAN]);
    let (v0, v1, v2) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0), v.get(2).copied().unwrap_or(0.0));
    if v1 <= v0 || v2 >= v1 || v.get(3).is_none_or(|x| !x.is_nan()) {
        return Err(format!("nmapi.drone_dendrite: piece values {v:?}"));
    }
    if (head_mean(&[2.0, 4.0, f32::NAN], &[1.0, 1.0, 0.0]) - 2.0).abs() > 1e-6 || !head_mean(&[], &[]).is_nan() {
        return Err("nmapi.drone_dendrite: head mean".into());
    }
    if !signals_on() && std::env::var("NANO_DRONE_SIGNALS").is_err() {
        return Err("nmapi.drone_dendrite: signals not on by default".into());
    }

    Ok(())
}
