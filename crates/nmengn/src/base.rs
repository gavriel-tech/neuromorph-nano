//! The base source of a MicroModel's parameters: the open weights, or one complete base cartridge.
//! Never a mixture of the two.
//!
//! The blocks (attention, feed-forward, norms, the order they run in) are the MicroModel's
//! assembly and hold no parameters of their own. A base cartridge holds parameter sets, each
//! addressed to the layer and block it loads onto by the tensor name the open weights use for that
//! place (`model.layers.13.mlp.down_proj.weight`: layer 13, feed-forward). Seating one is a single
//! step: the sets are checked complete and connected for the job's assembly (`connects`), the
//! job's model is bound over them, a check run compares its outputs with the open weights' on the
//! same input, and only then does the caller drop the open weights' model, which releases their
//! tensors from RAM and the card. A refusal names its reason and leaves the open weights in
//! service. Pulling is binding the open weights again. Additional cartridges (`filter`) stack on
//! whichever base the job runs on; the base's identity is recorded here so one made for another
//! base is refused.
use crate::assembly::Assembly;
use crate::model::{Model, Population, RunOpts, values};
use crate::placement::Device;
use nmnpcs::TensorDesc;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Why a cartridge was refused as a base source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refused {
    /// A parameter set a block of the job reads is not in the cartridge.
    Incomplete(String),
    /// A parameter set addressed to no block, to a layer the MicroModel lacks, twice to one place,
    /// or to a block other than its name says.
    Misaddressed(String),
    /// A parameter set whose shape or element type its block does not accept.
    Shape(String),
    /// Made for other open weights.
    Weights(String),
    /// The check run against the open weights failed.
    Check(String),
}
impl Refused {
    /// The short name of the reason.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Incomplete(_) => "incomplete",
            Self::Misaddressed(_) => "misaddressed",
            Self::Shape(_) => "shape",
            Self::Weights(_) => "weights",
            Self::Check(_) => "check",
        }
    }
}
impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (Self::Incomplete(m) | Self::Misaddressed(m) | Self::Shape(m) | Self::Weights(m) | Self::Check(m)) = self;
        write!(f, "base cartridge refused ({}): {m}", self.code())
    }
}

/// The block a parameter set loads onto: (layer, block), from its name. `None` for a name that is
/// no parameter set of a Qwen3 MicroModel's blocks.
#[must_use]
pub fn address(name: &str) -> Option<(Option<usize>, &'static str)> {
    match name {
        "model.embed_tokens.weight" => return Some((None, "table")),
        "model.norm.weight" => return Some((None, "final norm")),
        "lm_head.weight" => return Some((None, "head")),
        _ => {}
    }
    let (n, part) = name.strip_prefix("model.layers.")?.split_once('.')?;
    let layer = n.parse::<usize>().ok()?;
    let block = match part {
        "input_layernorm.weight"
        | "self_attn.q_proj.weight"
        | "self_attn.k_proj.weight"
        | "self_attn.v_proj.weight"
        | "self_attn.o_proj.weight"
        | "self_attn.q_norm.weight"
        | "self_attn.k_norm.weight" => "attention",
        "post_attention_layernorm.weight" | "mlp.gate_proj.weight" | "mlp.up_proj.weight" | "mlp.down_proj.weight" => "feed-forward",
        _ => return None,
    };
    Some((Some(layer), block))
}

/// Bytes a parameter set of `shape` takes in element type `dtype`.
#[must_use]
pub fn stored_bytes(dtype: &str, shape: &[usize]) -> Option<u64> {
    let n: usize = shape.iter().product();
    let b = match dtype {
        "BF16" | "F16" => n * 2,
        "F32" => n * 4,
        _ => return None,
    };
    Some(b as u64)
}

/// The tensors every block of an assembly reads (cluster references with the region taken off).
#[must_use]
pub fn read_by(asm: &Assembly) -> BTreeSet<String> {
    asm.nodes.iter().flat_map(|n| n.clusters.iter().map(|r| r.rsplit_once('/').map_or(r.as_str(), |x| x.0).to_owned())).collect()
}

/// Whether a cartridge's parameter sets are complete and connect for a job: every tensor a block of
/// the job reads (`needed`) is present exactly once, each set is addressed to a block and layer of
/// the MicroModel (`stated` is the label's own address per set, which must agree with its name),
/// its bytes fit its element type and shape, and its shape is the one its block accepts: the open
/// weights' shape (`open`) for attention, norms, table and head (never narrowed), and for a
/// feed-forward block the same stream width with an inner width of at most the open weights' and
/// one inner width for its three matrices. Values are held whole (BF16, F16 or F32), so the
/// MicroModel serves them at whatever precision is selected for it.
/// # Errors
/// The first reason found, named.
pub fn connects(
    needed: &BTreeSet<String>,
    open: &BTreeMap<String, Vec<usize>>,
    layers: usize,
    parts: &[TensorDesc],
    stated: &BTreeMap<String, (Option<usize>, String)>,
) -> Result<(), Refused> {
    let mut seen = BTreeSet::new();
    let mut inner: BTreeMap<usize, Vec<(String, usize)>> = BTreeMap::new();
    for p in parts {
        let n = p.name.as_str();
        let Some((layer, block)) = address(n) else {
            return Err(Refused::Misaddressed(format!("{n} is no parameter set of this MicroModel's blocks")));
        };
        if layer.is_some_and(|l| l >= layers) {
            return Err(Refused::Misaddressed(format!("{n} is addressed to layer {}, the MicroModel has layers 0 to {}", layer.unwrap_or(0), layers - 1)));
        }
        if let Some((sl, sb)) = stated.get(n)
            && (*sl != layer || sb != block)
        {
            return Err(Refused::Misaddressed(format!("{n} loads onto layer {layer:?} {block}, its label addresses it to layer {sl:?} {sb}")));
        }
        if !seen.insert(n.to_owned()) {
            return Err(Refused::Misaddressed(format!("{n} is given twice")));
        }
        if !needed.contains(n) {
            return Err(Refused::Misaddressed(format!("{n} is read by no block of this job")));
        }
        match stored_bytes(&p.dtype, &p.shape) {
            Some(b) if b == p.end - p.start => {}
            Some(b) => return Err(Refused::Shape(format!("{n}: {} {:?} takes {b} bytes, the cartridge gives {}", p.dtype, p.shape, p.end - p.start))),
            None => return Err(Refused::Shape(format!("{n}: element type {} is not held (BF16, F16, F32)", p.dtype))),
        }
        let want = open.get(n).ok_or_else(|| Refused::Misaddressed(format!("{n} is not in the open weights")))?;
        let ffn_matrix = n.ends_with(".mlp.gate_proj.weight") || n.ends_with(".mlp.up_proj.weight") || n.ends_with(".mlp.down_proj.weight");
        if ffn_matrix {
            let (stream, width, open_width) = if n.ends_with(".mlp.down_proj.weight") {
                (p.shape.first(), p.shape.get(1), want.get(1))
            } else {
                (p.shape.get(1), p.shape.first(), want.first())
            };
            let open_stream = if n.ends_with(".mlp.down_proj.weight") { want.first() } else { want.get(1) };
            if p.shape.len() != 2 || stream != open_stream || width.is_none_or(|w| *w == 0) || width > open_width {
                return Err(Refused::Shape(format!("{n}: {:?} does not fit the feed-forward block (open weights {want:?}; the stream width is kept, the inner width at most the open one)", p.shape)));
            }
            inner.entry(layer.unwrap_or(0)).or_default().push((n.to_owned(), width.copied().unwrap_or(0)));
        } else if &p.shape != want {
            return Err(Refused::Shape(format!("{n}: {:?}, its block takes {want:?} (attention, norms and the table are never narrowed)", p.shape)));
        }
    }
    for (l, v) in &inner {
        if let Some((a, x)) = v.first()
            && let Some((b, y)) = v.iter().find(|(_, w)| w != x)
        {
            return Err(Refused::Shape(format!("layer {l}: the feed-forward block's inner width differs between {a} ({x}) and {b} ({y})")));
        }
    }
    let missing: Vec<&String> = needed.iter().filter(|n| !seen.contains(*n)).collect();
    if let Some(first) = missing.first() {
        let place = address(first).map(|(l, b)| l.map_or_else(|| b.to_owned(), |l| format!("layer {l} {b}"))).unwrap_or_default();
        return Err(Refused::Incomplete(format!("{} parameter sets the job reads are missing, first {first} ({place})", missing.len())));
    }
    Ok(())
}

/// Fixed input of the check run.
const CHECK_TEXT: &str = "Judge whether the Document meets the requirements based on the Query.\nQuery: where does the ring write a new item\nDocument: crates/x/src/ring.rs\npub struct Ring { items: Vec<u32>, head: usize }\nimpl Ring { pub fn push(&mut self, x: u32) { let n = self.items.len(); self.items[self.head % n] = x; self.head += 1; } }\n";

/// The check run's outputs as bits: the final norm's state at every position and the job's last
/// output, on the population's own device.
/// # Errors
/// A run that fails.
pub fn check_outputs(m: &Model) -> Result<Vec<u32>, nmtype::Fault> {
    let ids = m.pop.tokenizer.encode(CHECK_TEXT).map_err(|e| nmtype::Fault::new("nmengn.base", nmtype::FaultKind::Execution, e.to_string()))?;
    let norm = m.role_node("final_norm")?;
    let last = m.asm.nodes.last().filter(|n| n.id != norm).map(|n| (n.id.clone(), if n.op == "head" { "logits" } else { "y" }.to_owned()));
    let mut targets = vec![(norm, "y".to_owned())];
    targets.extend(last);
    let dev = if m.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let out = m.forward(dev, &ids, &targets, &RunOpts::default())?;
    let mut bits = Vec::new();
    for t in &targets {
        if let Some(a) = out.get(t) {
            bits.extend(values(a)?.iter().map(|x| x.to_bits()));
        }
    }
    Ok(bits)
}

/// A cartridge offered as a job's base source.
#[derive(Clone, Debug)]
pub struct Offer {
    /// `id@version`, for the population's source label.
    pub key: String,
    /// SHA-256 of the open weights it was made from (the label's `weights`).
    pub weights: String,
    /// SHA-256 of its parameters (the label's `parameters_sha256`).
    pub parameters: String,
    /// Its parameter sets, each with the place of its bytes in the cartridge file.
    pub parts: Vec<TensorDesc>,
    /// The label's address of each set.
    pub stated: BTreeMap<String, (Option<usize>, String)>,
    /// The label says its values are the open weights' own: the check must then give
    /// bit-identical outputs.
    pub exact: bool,
    /// For a cartridge that is not exact, the largest output change its label states was measured;
    /// the check passes within it.
    pub bound: Option<f32>,
    /// The sets taken from the open weights at the seat, by tensor name: whole (`None`), or the
    /// rows or columns a recipe keeps. Their `parts` entries carry no file place.
    pub open: BTreeMap<String, Option<Recipe>>,
}

/// The rows (`cols` false) or columns of an open-weights tensor a set keeps, as element index
/// ranges along that axis of its `[rows, cols]` shape, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recipe {
    pub cols: bool,
    pub ranges: Vec<(usize, usize)>,
}

/// Sets built from the open weights for each seated MicroModel, held while it is seated so the
/// engine always finds them by content hash and never reads them from a file.
static BUILT: std::sync::Mutex<BTreeMap<String, Vec<Arc<[u8]>>>> = std::sync::Mutex::new(BTreeMap::new());

/// The cartridge's parameter sets as places the engine reads: its own sets as they are; a whole set
/// from the open weights at its place in the open weights' file; a narrowed one built in RAM from
/// the open weights' bytes by its recipe, checked against the content hash its label states, and
/// offered under that hash. Returns the places and the built bytes (to be held while seated).
type Built = (Vec<TensorDesc>, Vec<Arc<[u8]>>);
fn from_open(open: &Model, offer: &Offer) -> Result<Built, Refused> {
    let by_name: BTreeMap<&str, &TensorDesc> = open.pop.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let mut out = Vec::with_capacity(offer.parts.len());
    let mut built = Vec::new();
    for p in &offer.parts {
        let Some(recipe) = offer.open.get(&p.name) else {
            out.push(p.clone());
            continue;
        };
        let od = by_name.get(p.name.as_str()).ok_or_else(|| Refused::Incomplete(format!("{}: not in the open weights", p.name)))?;
        let Some(r) = recipe else {
            if od.shape != p.shape || od.dtype != p.dtype {
                return Err(Refused::Shape(format!("{}: whole from the open weights is {} {:?}, its label says {} {:?}", p.name, od.dtype, od.shape, p.dtype, p.shape)));
            }
            out.push(TensorDesc { file: od.file.clone(), start: od.start, end: od.end, ..p.clone() });
            continue;
        };
        let bytes = read_range(&open.pop.card.dir.join(&od.file), od.start, od.end).map_err(Refused::Shape)?;
        let width = match od.dtype.as_str() {
            "F32" => 4,
            _ => 2,
        };
        let (rows, cols) = (od.shape.first().copied().unwrap_or(0), od.shape.get(1).copied().unwrap_or(1));
        let mut b = Vec::new();
        if r.cols {
            for row in 0..rows {
                for (x, y) in &r.ranges {
                    b.extend_from_slice(bytes.get(width * (row * cols + x)..width * (row * cols + y)).ok_or_else(|| Refused::Shape(format!("{}: columns {x}..{y} outside {cols}", p.name)))?);
                }
            }
        } else {
            for (x, y) in &r.ranges {
                b.extend_from_slice(bytes.get(width * x * cols..width * y * cols).ok_or_else(|| Refused::Shape(format!("{}: rows {x}..{y} outside {rows}", p.name)))?);
            }
        }
        let want: usize = p.shape.iter().product::<usize>() * width;
        let sha = nmhash::sha256_bytes(&b);
        if b.len() != want || (!p.sha256.is_empty() && sha != p.sha256) {
            return Err(Refused::Shape(format!("{}: its recipe over the open weights gives other values than its label states", p.name)));
        }
        let arc: Arc<[u8]> = Arc::from(b);
        nmnpcs::provision::hold(&sha, &arc);
        built.push(Arc::clone(&arc));
        // No file holds these bytes: a read would fail by name rather than give other values.
        out.push(TensorDesc { file: format!("(built from the open weights: {})", p.name), start: 0, end: want as u64, sha256: sha, ..p.clone() });
    }
    Ok((out, built))
}

fn read_range(path: &std::path::Path, start: u64, end: u64) -> Result<Vec<u8>, String> {
    use std::io::{Read as _, Seek as _};
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    f.seek(std::io::SeekFrom::Start(start)).map_err(|e| e.to_string())?;
    let mut b = vec![0_u8; usize::try_from(end.saturating_sub(start)).map_err(|e| e.to_string())?];
    f.read_exact(&mut b).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(b)
}

/// What a seat did.
#[derive(Clone, Debug, Default)]
pub struct Seated {
    /// Check outputs compared, and how many were bit-identical; the largest difference.
    pub outputs: usize,
    pub identical: usize,
    pub largest: f32,
    /// Seconds to check, open and bind the cartridge, and run both checks.
    pub secs: f64,
}

/// Seat `offer` as the base source of the job `open` runs on the open weights: connects, binds the
/// job's assembly over the cartridge (at the precision selected for the MicroModel, as the open
/// weights' model was), runs the check against the open weights and returns the cartridge's model. The caller
/// then drops `open`, which releases the open weights from RAM and the card. On a refusal nothing
/// of the cartridge stays and `open` is untouched.
/// # Errors
/// The reason, named.
pub fn seat(open: &Model, offer: &Offer) -> Result<(Model, Seated), Refused> {
    let t = std::time::Instant::now();
    if offer.weights != open.pop.card.sha_main {
        return Err(Refused::Weights(format!("{} was made for open weights {:.12}, {} is {:.12}", offer.key, offer.weights, open.pop.card.id, open.pop.card.sha_main)));
    }
    let needed = read_by(&open.asm);
    let shapes: BTreeMap<String, Vec<usize>> = open.pop.tensors.iter().map(|t| (t.name.clone(), t.shape.clone())).collect();
    connects(&needed, &shapes, open.pop.dims.layers, &offer.parts, &offer.stated)?;
    let gpu = open.pop.gpu.is_some();
    let (parts, built) = from_open(open, offer)?;
    let pop = Population::open_cartridge(&open.pop.home, &open.pop.card.id, gpu, &offer.key, parts).map_err(|e| Refused::Shape(e.to_string()))?;
    let rows = open.bound.iter().any(|b| !b.rows.is_empty());
    let bind = |p: &Arc<Population>, a: Assembly| -> Result<Model, Refused> {
        let m = Model::bind(p, a).map_err(|e| Refused::Shape(e.to_string()))?;
        Ok(if rows { m.label_rows() } else { m })
    };
    let cart = bind(&pop, open.asm.clone())?;
    let a = check_outputs(open).map_err(|e| Refused::Check(format!("open weights: {e}")))?;
    let b = check_outputs(&cart).map_err(|e| Refused::Check(format!("cartridge: {e}")))?;
    // The check read the full output head; the jobs read their rows or none, so the seated model
    // holds on the card what the open weights' model held.
    let _ = cart.release_full_heads();
    let identical = a.iter().zip(&b).filter(|(x, y)| x == y).count();
    let largest = a.iter().zip(&b).map(|(x, y)| (f32::from_bits(*x) - f32::from_bits(*y)).abs()).fold(0.0_f32, f32::max);
    let seated = Seated { outputs: a.len(), identical, largest, secs: t.elapsed().as_secs_f64() };
    if a.len() != b.len() || a.is_empty() {
        return Err(Refused::Check(format!("{} outputs against the open weights' {}", b.len(), a.len())));
    }
    if offer.exact && identical != a.len() {
        return Err(Refused::Check(format!("its label says exact, yet {} of {} check outputs differ from the open weights (largest {largest:e})", a.len() - identical, a.len())));
    }
    if !offer.exact {
        let within = offer.bound.is_some_and(|bd| largest.is_finite() && largest <= bd);
        if !within {
            return Err(Refused::Check(format!(
                "not exact, check outputs differ by up to {largest:e}, its label states {}",
                offer.bound.map_or_else(|| "no bound".to_owned(), |bd| format!("a bound of {bd:e}"))
            )));
        }
    }
    set_base(&open.pop.card.id, if offer.exact { &open.pop.card.sha_main } else { &offer.parameters });
    if let Ok(mut g) = BUILT.lock() {
        g.insert(open.pop.card.id.clone(), built);
    }
    Ok((cart, seated))
}

/// Expected Qwen3 tensor shapes derived from `weights/<id>/files/config.json` so a self-contained
/// base cartridge can validate and seat without downloading the 8 GB upstream `.safetensors`.
fn expected_shapes_from_card(card: &nmnpcs::WeightCard) -> Result<BTreeMap<String, Vec<usize>>, Refused> {
    let get = |k: &str| card.int(k).map_err(|e| Refused::Shape(e.to_string()));
    let layers = get("num_hidden_layers")?;
    let hidden = get("hidden_size")?;
    let heads = get("num_attention_heads")?;
    let kv_heads = get("num_key_value_heads")?;
    let head_dim = get("head_dim")?;
    let inter = get("intermediate_size")?;
    let vocab = get("vocab_size")?;
    let mut shapes = BTreeMap::new();
    shapes.insert("model.embed_tokens.weight".to_owned(), vec![vocab, hidden]);
    shapes.insert("model.norm.weight".to_owned(), vec![hidden]);
    shapes.insert("lm_head.weight".to_owned(), vec![vocab, hidden]);
    for l in 0..layers {
        let p = |s: &str| format!("model.layers.{l}.{s}.weight");
        shapes.insert(p("input_layernorm"), vec![hidden]);
        shapes.insert(p("post_attention_layernorm"), vec![hidden]);
        shapes.insert(p("self_attn.q_proj"), vec![heads * head_dim, hidden]);
        shapes.insert(p("self_attn.k_proj"), vec![kv_heads * head_dim, hidden]);
        shapes.insert(p("self_attn.v_proj"), vec![kv_heads * head_dim, hidden]);
        shapes.insert(p("self_attn.o_proj"), vec![hidden, heads * head_dim]);
        shapes.insert(p("self_attn.q_norm"), vec![head_dim]);
        shapes.insert(p("self_attn.k_norm"), vec![head_dim]);
        shapes.insert(p("mlp.gate_proj"), vec![inter, hidden]);
        shapes.insert(p("mlp.up_proj"), vec![inter, hidden]);
        shapes.insert(p("mlp.down_proj"), vec![hidden, inter]);
    }
    Ok(shapes)
}

/// Seat a self-contained base cartridge directly without upstream open-weight `.safetensors` on
/// disk: validates completeness, addressing, and shapes against `config.json` + `part.json`, opens
/// the cartridge population, and binds `asm`. Upstream open weights remain optional for fallback,
/// reference parity checks, parameter mapping, and surgery.
/// # Errors
/// The reason, named.
pub fn seat_standalone(
    home: &std::path::Path,
    id: &str,
    asm: Assembly,
    gpu: bool,
    label_rows: bool,
    offer: &Offer,
) -> Result<(Model, Seated), Refused> {
    let t = std::time::Instant::now();
    if !offer.open.is_empty() {
        return Err(Refused::Incomplete(format!("{} depends on upstream open-weight tensor recipes; download the optional upstream open weights to seat it", offer.key)));
    }
    let card = nmnpcs::WeightCard::load(home, id).map_err(|e| Refused::Shape(e.to_string()))?;
    if offer.weights != card.sha_main {
        return Err(Refused::Weights(format!("{} was made for open weights {:.12}, {} is {:.12}", offer.key, offer.weights, card.id, card.sha_main)));
    }
    let needed = read_by(&asm);
    let shapes = expected_shapes_from_card(&card)?;
    let layers = card.int("num_hidden_layers").map_err(|e| Refused::Shape(e.to_string()))?;
    connects(&needed, &shapes, layers, &offer.parts, &offer.stated)?;
    let pop = Population::open_cartridge(home, id, gpu, &offer.key, offer.parts.clone()).map_err(|e| Refused::Shape(e.to_string()))?;
    let m = Model::bind(&pop, asm).map_err(|e| Refused::Shape(e.to_string()))?;
    let cart = if label_rows { m.label_rows() } else { m };
    set_base(id, if offer.exact { &card.sha_main } else { &offer.parameters });
    Ok((cart, Seated { outputs: 0, identical: 0, largest: 0.0, secs: t.elapsed().as_secs_f64() }))
}

/// Bases in service per MicroModel: the identity an additional cartridge must have been made for
/// (the open weights' SHA-256, also for a base cartridge exact to them; else the cartridge's
/// parameters' SHA-256). A MicroModel not listed runs on its open weights.
static BASES: std::sync::RwLock<BTreeMap<String, String>> = std::sync::RwLock::new(BTreeMap::new());

/// Record the base identity a MicroModel runs on.
pub fn set_base(model: &str, identity: &str) {
    if let Ok(mut g) = BASES.write() {
        g.insert(model.to_owned(), identity.to_owned());
    }
}
/// The MicroModel is back on its open weights.
pub fn pull_base(model: &str) {
    if let Ok(mut g) = BASES.write() {
        g.remove(model);
    }
    if let Ok(mut g) = BUILT.lock() {
        g.remove(model);
    }
}
/// The identity of the base a MicroModel runs on, if a base cartridge is seated.
#[must_use]
pub fn base_of(model: &str) -> Option<String> {
    BASES.read().ok().and_then(|g| g.get(model).cloned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(name: &str, dtype: &str, shape: &[usize]) -> TensorDesc {
        let len = stored_bytes(dtype, shape).unwrap_or(0);
        TensorDesc { name: name.to_owned(), dtype: dtype.to_owned(), shape: shape.to_vec(), file: String::new(), start: 0, end: len, sha256: String::new() }
    }
    fn layer(l: usize, inner: usize) -> Vec<TensorDesc> {
        let p = |x: &str| format!("model.layers.{l}.{x}.weight");
        vec![
            t(&p("input_layernorm"), "BF16", &[8]),
            t(&p("self_attn.q_proj"), "BF16", &[8, 8]),
            t(&p("self_attn.k_proj"), "BF16", &[4, 8]),
            t(&p("post_attention_layernorm"), "BF16", &[8]),
            t(&p("mlp.gate_proj"), "BF16", &[inner, 8]),
            t(&p("mlp.up_proj"), "BF16", &[inner, 8]),
            t(&p("mlp.down_proj"), "BF16", &[8, inner]),
        ]
    }
    fn open() -> (BTreeSet<String>, BTreeMap<String, Vec<usize>>) {
        let all: Vec<TensorDesc> = (0..2).flat_map(|l| layer(l, 12)).chain([t("model.norm.weight", "BF16", &[8])]).collect();
        (all.iter().map(|x| x.name.clone()).collect(), all.iter().map(|x| (x.name.clone(), x.shape.clone())).collect())
    }

    #[test]
    fn a_complete_cartridge_connects_and_each_fault_is_refused_by_name() {
        let (needed, shapes) = open();
        let none = BTreeMap::new();
        let full: Vec<TensorDesc> = (0..2).flat_map(|l| layer(l, 12)).chain([t("model.norm.weight", "BF16", &[8])]).collect();
        assert!(connects(&needed, &shapes, 2, &full, &none).is_ok());
        let narrowed: Vec<TensorDesc> = layer(0, 12).into_iter().chain(layer(1, 4)).chain([t("model.norm.weight", "BF16", &[8])]).collect();
        assert!(connects(&needed, &shapes, 2, &narrowed, &none).is_ok());
        let missing: Vec<TensorDesc> = full.iter().filter(|x| !x.name.contains("layers.1.mlp.up")).cloned().collect();
        assert_eq!(connects(&needed, &shapes, 2, &missing, &none).map_err(|e| e.code()), Err("incomplete"));
        let mut moved = full.clone();
        moved[0].name = "model.layers.1.input_layernorm.weight".into();
        assert_eq!(connects(&needed, &shapes, 2, &moved, &none).map_err(|e| e.code()), Err("misaddressed"));
        let mut beyond = full.clone();
        beyond.push(t("model.layers.7.input_layernorm.weight", "BF16", &[8]));
        assert_eq!(connects(&needed, &shapes, 2, &beyond, &none).map_err(|e| e.code()), Err("misaddressed"));
        let stated: BTreeMap<String, (Option<usize>, String)> = [("model.layers.0.input_layernorm.weight".to_owned(), (Some(1), "attention".to_owned()))].into();
        assert_eq!(connects(&needed, &shapes, 2, &full, &stated).map_err(|e| e.code()), Err("misaddressed"));
        let mut narrow_attention = full.clone();
        narrow_attention[1] = t("model.layers.0.self_attn.q_proj.weight", "BF16", &[4, 8]);
        assert_eq!(connects(&needed, &shapes, 2, &narrow_attention, &none).map_err(|e| e.code()), Err("shape"));
        let mut mixed = full.clone();
        mixed[5] = t("model.layers.0.mlp.up_proj.weight", "BF16", &[10, 8]);
        assert_eq!(connects(&needed, &shapes, 2, &mixed, &none).map_err(|e| e.code()), Err("shape"));
        let mut wide = full.clone();
        wide[6] = t("model.layers.0.mlp.down_proj.weight", "BF16", &[6, 12]);
        assert_eq!(connects(&needed, &shapes, 2, &wide, &none).map_err(|e| e.code()), Err("shape"));
        let mut int8 = full.clone();
        int8[1] = TensorDesc { dtype: "I8".into(), ..int8[1].clone() };
        assert_eq!(connects(&needed, &shapes, 2, &int8, &none).map_err(|e| e.code()), Err("shape"));
    }
}
