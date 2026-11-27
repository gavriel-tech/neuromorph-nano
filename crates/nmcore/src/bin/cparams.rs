//! Cartridge parameters: one read format and one edit path for every object that holds learned
//! parameters, whether or not it is a cartridge in the library's catalogue.
//!
//! Objects read here, each by the key the interface, MCP and the API use:
//! - library cartridges (`id@version`): socket cartridges (gain and bias points per node, the
//!   `nmengn-sockets/1` contract; adapter-family cartridges of the same form), preference
//!   cartridges (their learned `<good>/<signals>` counts), base cartridges (parameter sets, read by
//!   shape; values read only for parts at most `READ_CAP` bytes, and paged from the file for any
//!   part). An operating settings cartridge holds no parameters and says so;
//! - job filters (`filter:<role>/<job>`): `library/parameter-maps/<role>/job-filter-<job>.json`,
//!   gain and bias points per node;
//! - project adapters in the adapter library (`<id>@<version>`, `library/cartridges/adapter/<id>/`):
//!   the embedder's lens (`flat`, `beta`, low-rank `down` and `up`) and the judge's code read-out
//!   (`w`);
//! - learning units of the registry (`units/registry.json`, by id): low-rank `down` and `up`.
//!
//! The read format (`FORMAT`) is one object, `parameters`: whether the object holds parameters,
//! its kind and MicroModel, what may be edited, a row per layer (tensor count and the L2 norm of
//! what the parameters change, `effect_l2`), and a row per tensor or point: name, target node,
//! layer, shape, element type, size, bytes, L2 and largest magnitude, `effect_l2` (the norm of the
//! difference from the value that changes nothing: 1 for a gain, 0 otherwise), and a summary
//! (minimum, maximum, mean, a histogram of `BINS` bins). Values are never sent whole: a tensor's
//! values are read by page (`page`), from memory for the small objects and by a ranged read of the
//! file for base cartridge parts, so a large part is never loaded whole.
//!
//! Edits (`edit`): a strength (one scale for every scalable tensor, `identity + s * (x -
//! identity)`), per-layer switches (`off` drops a layer's points where the form is a list of
//! points, and zeroes the change elsewhere; a number scales that layer; `on` keeps it), and direct
//! values for tensors of at most `SMALL` elements, shape checked. Every edit writes a new object
//! beside the source, which is never written: a new cartridge version on trial (library), a new job
//! filter file plugged out (job filters), the next adapter version (the adapter library), or a new
//! unit as a candidate (the registry). The new version's compatibility with the installed
//! MicroModels is checked as `library::compat` does, and the edit is one `cartridge-parameters`
//! audit row; undo sets the new object aside (`library/set-aside/`, the role's map archive) or
//! retires the new unit, deleting nothing. Base cartridges and the upstream open weights are only
//! read.
use nmjson::Value;
use nmlearn::library::{CartridgeSpec, Filter, Library, Point};
use std::collections::BTreeMap;
use std::io::{Read as _, Seek as _};
use std::path::{Path, PathBuf};

/// The read format's name.
pub const FORMAT: &str = "nm-parameters/1";
/// Elements a tensor may have for its values to be edited directly.
pub const SMALL: usize = 65_536;
/// Bytes of one base cartridge part read for its norms and summary (the norms and biases of a
/// layer; a matrix is larger): larger parts are listed by shape and read by page only, so an
/// inspection reads a small part of a base cartridge's file.
pub const READ_CAP: u64 = 256 << 10;
/// The most values one page returns.
pub const PAGE_MAX: usize = 4096;
/// Bins of a tensor's histogram.
pub const BINS: usize = 8;
/// The largest strength an edit may set.
pub const SCALE_MAX: f32 = 4.0;
/// The audit row kind of a parameter edit.
pub const AUDIT_KIND: &str = "cartridge-parameters";
/// The key prefix of a job filter.
pub const FILTER_PREFIX: &str = "filter:";
/// Where an undone edit's file is set aside, under the Nano's folder.
pub const SET_ASIDE: &str = "library/cartridges/set-aside";
const ROLES: [&str; 4] = ["embedder", "reader", "reranker", "writer"];

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}
/// A figure at the precision parameters are held in (f32), so a stored 0.2 reads as 0.2.
fn f(x: f64) -> Value {
    let y = narrow(x);
    if y.is_finite() { Value::Number(format!("{y:e}")) } else { Value::Null }
}
fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn text(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default()
}
fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Number(t) => t.parse().ok(),
        Value::String(t) => t.trim().parse().ok(),
        _ => None,
    }
}
fn shape_v(shape: &[usize]) -> Value {
    Value::Array(shape.iter().map(|x| n(*x as u64)).collect())
}
#[expect(clippy::cast_possible_truncation, reason = "parameters and their figures are held in f32")]
const fn narrow(x: f64) -> f32 {
    x as f32
}

/// The layer a node id names (`L27.swigate`, `nmengn.components#L27.down`), if any.
#[must_use]
pub fn layer_of(target: &str) -> Option<usize> {
    target.split(['.', '#', '/', ':']).find_map(|seg| seg.strip_prefix('L').filter(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit())).and_then(|d| d.parse().ok()))
}

// ---------------------------------------------------------------------------------------------
// The objects read.

/// One tensor or point of an object.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    /// Unique within the object (`L27.swigate.gain`, `down`, `pref.global`).
    pub name: String,
    /// The node it acts at.
    pub target: String,
    pub layer: Option<usize>,
    /// `gain`, `bias`, `flat`, `beta`, `down`, `up`, `w`, `preference`, or the block of a base set.
    pub part: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    /// The value at which the tensor changes nothing (1 for a gain, 0 otherwise).
    pub identity: f32,
    /// Whether a strength scales it (the factor that carries the change, not its direction).
    pub scales: bool,
    /// The values, when read; `None` for a base part above `READ_CAP` or one taken from the open
    /// weights.
    pub values: Option<Vec<f32>>,
    pub bytes: u64,
    /// Why the values were not read.
    pub unread: Option<String>,
    /// For a base part: its file and the byte offset of its values, for paged reads.
    pub at: Option<(PathBuf, u64)>,
}
impl Tensor {
    fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

pub(crate) fn point_tensors(target: &str, gain: Vec<f32>, bias: Vec<f32>, dtype: &str) -> [Tensor; 2] {
    let layer = layer_of(target);
    let one = |part: &str, identity: f32, values: Vec<f32>, per: u64| Tensor {
        name: format!("{target}.{part}"),
        target: target.to_owned(),
        layer,
        part: part.to_owned(),
        shape: vec![values.len()],
        dtype: dtype.to_owned(),
        identity,
        scales: true,
        bytes: values.len() as u64 * per,
        values: Some(values),
        unread: None,
        at: None,
    };
    let per = if dtype == "q8" { 1 } else { 4 };
    [one("gain", 1.0, gain, per), one("bias", 0.0, bias, per)]
}

/// Where an object came from, enough to write its next version.
#[derive(Clone, Debug)]
enum Origin {
    Library { c: Box<nmlearn::library::Cartridge>, q8: bool },
    Job { role: String, file: String, doc: Value },
    Adapter { id: String, version: u64, label: Value },
    Unit { card: Box<nmlearn::UnitCard> },
    /// A base cartridge: an edit is written as a new base version beside the others, on trial.
    Base { c: Box<nmlearn::library::Cartridge> },
    ReadOnly,
}

/// One object holding parameters, read.
#[derive(Clone, Debug)]
pub struct Held {
    pub key: String,
    /// `library`, `job-filter`, `adapter` or `unit`.
    pub source: &'static str,
    /// `filter`, `adapter`, `lens`, `readout`, `unit`, `preference`, `base` or `settings`.
    pub kind: &'static str,
    pub model: String,
    pub name: String,
    pub tensors: Vec<Tensor>,
    /// `Ok`: edits are taken; `Err`: why not.
    pub editable: Result<(), String>,
    /// Whether a layer switched off drops its points (a list of points) or zeroes its change.
    pub drops: bool,
    /// The object's state (`trial`, `served`, `plugged in`, `plugged out`, a unit's status).
    pub state: String,
    pub path: String,
    pub note: String,
    origin: Origin,
}

/// The cartridge file's label and where its parameters begin, read from its head alone.
fn head(file: &Path) -> Result<(Value, u64), String> {
    let io = |e: std::io::Error| format!("{}: {e}", file.display());
    let mut fh = std::fs::File::open(file).map_err(io)?;
    let mut h = [0_u8; 12];
    fh.read_exact(&mut h).map_err(io)?;
    if h.get(..8) != Some(nmlearn::library::CARTRIDGE_MARK.as_slice()) {
        return Err(format!("{}: not a cartridge file", file.display()));
    }
    let len = h.get(8..12).and_then(|x| <[u8; 4]>::try_from(x).ok()).map_or(0, u32::from_le_bytes);
    let mut b = vec![0_u8; len as usize];
    fh.read_exact(&mut b).map_err(io)?;
    let label = nmjson::parse(&b).map_err(|e| format!("{}: label: {e}", file.display()))?;
    Ok((label, 12 + u64::from(len)))
}

/// Element bytes of a stored type.
fn elem(dtype: &str) -> Option<u64> {
    match dtype {
        "BF16" | "F16" => Some(2),
        "F32" => Some(4),
        _ => None,
    }
}

/// Stored values as f32.
fn decode(dtype: &str, raw: &[u8]) -> Vec<f32> {
    match dtype {
        "BF16" => raw.as_chunks::<2>().0.iter().map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16)).collect(),
        "F16" => raw.as_chunks::<2>().0.iter().map(|c| nmengn::half::from16(u16::from_le_bytes(*c))).collect(),
        "F32" => raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
        _ => Vec::new(),
    }
}

/// `count` values of a base part from `offset`, by a ranged read of its file.
fn read_range(file: &Path, at: u64, dtype: &str, offset: usize, count: usize) -> Result<Vec<f32>, String> {
    let e = elem(dtype).ok_or_else(|| format!("{dtype}: not a stored type read here"))?;
    let io = |x: std::io::Error| format!("{}: {x}", file.display());
    let mut fh = std::fs::File::open(file).map_err(io)?;
    fh.seek(std::io::SeekFrom::Start(at + offset as u64 * e)).map_err(io)?;
    let mut b = vec![0_u8; count * usize::try_from(e).unwrap_or(4)];
    fh.read_exact(&mut b).map_err(io)?;
    Ok(decode(dtype, &b))
}

fn base_tensors(file: &Path) -> Result<Vec<Tensor>, String> {
    let (label, body) = head(file)?;
    let mut out = Vec::new();
    for p in label.get("parts").and_then(Value::array).unwrap_or_default() {
        let form = text(p, "form");
        let name = text(p, "socket");
        let shape: Vec<usize> = p.get("shape").and_then(Value::array).unwrap_or_default().iter().filter_map(|x| x.integer().ok().and_then(|v| usize::try_from(v).ok())).collect();
        let layer = p.get("layer").and_then(Value::integer).ok().and_then(|x| usize::try_from(x).ok());
        let dtype = text(p, "encoding");
        let at = p.get("at").and_then(Value::integer).unwrap_or(0);
        let bytes = p.get("bytes").and_then(Value::integer).unwrap_or(0);
        let mut t = Tensor { name: name.clone(), target: name, layer, part: text(p, "block"), shape, dtype: dtype.clone(), identity: 0.0, scales: false, values: None, bytes, unread: None, at: None };
        match form.as_str() {
            "tensor" if elem(&dtype).is_some() => {
                t.at = Some((file.to_path_buf(), body + at));
                if bytes <= READ_CAP {
                    t.values = Some(read_range(file, body + at, &dtype, 0, t.size())?);
                } else {
                    t.unread = Some(format!("{} KiB, above the {} KiB read for norms; read it by page", bytes >> 10, READ_CAP >> 10));
                }
            }
            "open" => t.unread = Some("taken from the open weights at the seat; this cartridge does not hold its values".to_owned()),
            _ => continue,
        }
        out.push(t);
    }
    Ok(out)
}

fn library_held(lib: &Library, c: &nmlearn::library::Cartridge) -> Held {
    let mut h = Held {
        key: c.key(),
        source: "library",
        kind: "settings",
        model: c.model.clone(),
        name: c.name.clone(),
        tensors: Vec::new(),
        editable: Ok(()),
        drops: true,
        state: if c.trial { "trial" } else { "served" }.to_owned(),
        path: c.files.first().map(|x| lib.cartridge_dir(c).join(&x.name).display().to_string()).unwrap_or_default(),
        note: String::new(),
        origin: Origin::Library { c: Box::new(c.clone()), q8: false },
    };
    match c.kind.as_str() {
        "tensors" => {
            h.kind = "base";
            h.drops = false;
            h.origin = Origin::Base { c: Box::new(c.clone()) };
            match c.files.first().map(|x| lib.cartridge_dir(c).join(&x.name)) {
                Some(file) if file.is_file() => match base_tensors(&file) {
                    Ok(t) => {
                        h.tensors = t;
                        h.note = "a base cartridge: every parameter set its blocks read; sets above the read cap are listed by shape and read by page. An edit is written as a new base version beside the others, on trial, and is checked against the upstream open weights when it is seated; the version serving now and the open weights are never written".to_owned();
                    }
                    Err(e) => h.note = e,
                },
                _ => h.note = "the cartridge's file is missing".to_owned(),
            }
        }
        "affine" => {
            h.kind = if c.family == nmlearn::library::ADAPTER { "adapter" } else { "filter" };
            match lib.filter(c) {
                Ok(Filter::Points { points, q8 }) => {
                    let dtype = if q8.is_some() { "q8" } else { "F32" };
                    h.tensors = points.into_iter().flat_map(|p| point_tensors(&p.target, p.gain, p.bias, dtype)).collect();
                    h.origin = Origin::Library { c: Box::new(c.clone()), q8: q8.is_some() };
                    h.note = "gain and bias per channel at each socket (x * gain + bias); an edit makes a new version on trial".to_owned();
                }
                Ok(_) => {
                    h.editable = Err("the label says a socket cartridge but the file holds another kind of filter".to_owned());
                    h.origin = Origin::ReadOnly;
                }
                Err(e) => {
                    h.editable = Err(format!("the cartridge did not pass its checks: {e}"));
                    h.origin = Origin::ReadOnly;
                    h.note = e.to_string();
                }
            }
        }
        _ if c.family == nmlearn::library::PREFERENCE => {
            h.kind = "preference";
            h.drops = false;
            if let Ok(Filter::Settings(m)) = lib.filter(c) {
                for (k, v) in m.iter().filter(|(k, _)| k.starts_with("pref.")) {
                    let Some((g, sg)) = v.split_once('/').and_then(|(a, b)| Some((a.trim().parse::<f32>().ok()?, b.trim().parse::<f32>().ok()?))) else { continue };
                    h.tensors.push(Tensor { name: k.clone(), target: k.clone(), layer: None, part: "preference".to_owned(), shape: vec![2], dtype: "count".to_owned(), identity: 0.0, scales: false, values: Some(vec![g, sg]), bytes: 0, unread: None, at: None });
                }
            }
            h.note = "learned preference values: per label, good signals of all signals".to_owned();
        }
        _ => {
            h.editable = Err("this cartridge holds no parameters: it holds operating settings (edit them as values)".to_owned());
            h.origin = Origin::ReadOnly;
            h.note = "this cartridge holds no parameters".to_owned();
        }
    }
    h
}

fn parse_points(doc: &Value) -> Result<Vec<Point>, String> {
    let mut out = Vec::new();
    for p in doc.get("parts").and_then(Value::array).unwrap_or_default() {
        let list = |k: &str| -> Result<Vec<f32>, String> { p.get(k).and_then(Value::array).map_err(|e| format!("{k}: {e}"))?.iter().map(|x| num(x).map(narrow).ok_or_else(|| format!("{k}: a value that is not a number"))).collect() };
        let target = text(p, "target");
        let (gain, bias) = (list("gain")?, list("bias")?);
        if target.is_empty() || gain.len() != bias.len() {
            return Err(format!("part {target}: a target with a gain and a bias of one width"));
        }
        out.push(Point { target, gain, bias });
    }
    Ok(out)
}

fn job_dir(home: &Path, role: &str) -> PathBuf {
    home.join("library/parameter-maps").join(role)
}

fn job_held(home: &Path, role: &str, job: &str) -> Result<Held, String> {
    if !ROLES.contains(&role) || job.is_empty() || job.contains(['/', '\\']) || job.starts_with('.') {
        return Err(format!("{FILTER_PREFIX}{role}/{job}: give a job filter as {FILTER_PREFIX}<role>/<job> (roles: reader, writer, embedder, reranker)"));
    }
    let file = format!("job-filter-{job}.json");
    let path = job_dir(home, role).join(&file);
    let doc = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display())).and_then(|b| nmjson::parse(&b).map_err(|e| format!("{file}: {e}")))?;
    let points = parse_points(&doc)?;
    let enabled = !matches!(doc.get("enabled"), Ok(Value::Bool(false)));
    Ok(Held {
        key: format!("{FILTER_PREFIX}{role}/{job}"),
        source: "job-filter",
        kind: "filter",
        model: text(&doc, "model"),
        name: text(&doc, "cartridge"),
        tensors: points.into_iter().flat_map(|p| point_tensors(&p.target, p.gain, p.bias, "F32")).collect(),
        editable: Ok(()),
        drops: true,
        state: if enabled { "plugged in" } else { "plugged out" }.to_owned(),
        path: path.display().to_string(),
        note: "a job filter: gain and bias per channel at each node; an edit makes a new job filter file, plugged out".to_owned(),
        origin: Origin::Job { role: role.to_owned(), file, doc },
    })
}

/// The job filter a key names: `filter:<role>/<job>`, or `filter:<cartridge>` as `compat` lists it.
fn job_of(home: &Path, rest: &str) -> Result<Held, String> {
    if let Some((role, job)) = rest.split_once('/') {
        return job_held(home, role, job);
    }
    for role in ROLES {
        for (job, _) in job_files(home, role) {
            if let Ok(h) = job_held(home, role, &job)
                && h.name == rest
            {
                return Ok(h);
            }
        }
    }
    Err(format!("no job filter {FILTER_PREFIX}{rest}"))
}

/// The job filters of a role: (job, file path), in name order.
fn job_files(home: &Path, role: &str) -> Vec<(String, PathBuf)> {
    let mut v: Vec<(String, PathBuf)> = std::fs::read_dir(job_dir(home, role))
        .map(|rd| rd.flatten().filter_map(|e| Some((e.file_name().to_str()?.strip_prefix("job-filter-")?.strip_suffix(".json")?.to_owned(), e.path()))).collect())
        .unwrap_or_default();
    v.sort();
    v
}

fn adapter_held(home: &Path, id: &str, version: u64, file: &Path) -> Result<Held, String> {
    let (label, parts) = nmapi::adapter::read_file(file).map_err(|e| e.message)?;
    let kind = text(&label, "kind");
    let dtype = label.get("parts").and_then(Value::array).ok().and_then(|a| a.first().map(|p| text(p, "encoding"))).unwrap_or_else(|| "F16".to_owned());
    let (k, target) = match kind.as_str() {
        nmapi::adapter::LENS_KIND => ("lens", "the embedder's request vector"),
        nmapi::adapter::READOUT_KIND => ("readout", "final_norm"),
        _ => ("adapter", "adapter"),
    };
    let tensors = parts
        .into_iter()
        .map(|(name, shape, xs)| Tensor {
            target: target.to_owned(),
            layer: None,
            part: name.clone(),
            scales: matches!(name.as_str(), "up" | "beta" | "w"),
            dtype: dtype.clone(),
            identity: 0.0,
            bytes: xs.len() as u64 * elem(&dtype).unwrap_or(2),
            shape,
            values: Some(xs),
            unread: None,
            at: None,
            name,
        })
        .collect();
    let plugged = nmapi::plugs::acts(home, &text(label.get("notes").unwrap_or(&Value::Null), "project"), id);
    let newest = nmapi::adapter::versions(home, id).last().is_some_and(|(v, _)| *v == version);
    Ok(Held {
        key: format!("{id}@{version}"),
        source: "adapter",
        kind: k,
        model: text(&label, "model"),
        name: id.to_owned(),
        tensors,
        editable: Ok(()),
        drops: false,
        state: format!("{}{}", if newest { "newest version" } else { "earlier version" }, if plugged { "" } else { ", unplugged" }),
        path: file.display().to_string(),
        note: if k == "lens" {
            "the embedder's per-project lens: directions flattened (flat, beta) and a low-rank map (down, up); a strength scales beta and up".to_owned()
        } else {
            "the judge's code read-out at its final norm (w); a strength scales w".to_owned()
        },
        origin: Origin::Adapter { id: id.to_owned(), version, label },
    })
}

/// The adapter library file a key names (`<id>@<version>`, or `<id>` for its newest version).
fn adapter_of(home: &Path, key: &str) -> Option<Result<Held, String>> {
    let (id, ver) = match key.rsplit_once('@') {
        Some((i, v)) => (i, v.parse::<u64>().ok()),
        None => (key, None),
    };
    if id.is_empty() || id.contains(['/', '\\']) || id.starts_with('.') {
        return None;
    }
    let all = nmapi::adapter::versions(home, id);
    let (v, file) = match ver {
        Some(v) => all.into_iter().find(|(x, _)| *x == v)?,
        None => all.into_iter().last()?,
    };
    Some(adapter_held(home, id, v, &file))
}

fn unit_held(home: &Path, card: &nmlearn::UnitCard) -> Held {
    let units = nmlearn::Units::open(home);
    let dir = units.unit_dir(&card.id);
    let layer = layer_of(&card.target);
    let mut h = Held {
        key: card.id.clone(),
        source: "unit",
        kind: "unit",
        model: card.base.clone(),
        name: card.id.clone(),
        tensors: Vec::new(),
        editable: Ok(()),
        drops: false,
        state: card.status.clone(),
        path: dir.display().to_string(),
        note: format!("a learning unit on {}: down [rank, k] stays as drawn, up [n, rank] carries the change; scale {} on the card", card.target, card.scale),
        origin: Origin::Unit { card: Box::new(card.clone()) },
    };
    if card.files.is_empty() {
        h.editable = Err("an unfitted unit holds no files: its up factor is zero until it is fitted".to_owned());
        h.note = "an unfitted unit: no parameters held yet".to_owned();
        return h;
    }
    match nmlearn::UnitWeights::read(card, &dir) {
        Ok(w) => {
            let t = |name: &str, shape: Vec<usize>, xs: Vec<f32>, scales: bool| Tensor {
                name: name.to_owned(),
                target: card.target.clone(),
                layer,
                part: name.to_owned(),
                bytes: xs.len() as u64,
                shape,
                dtype: "q8".to_owned(),
                identity: 0.0,
                scales,
                values: Some(xs),
                unread: None,
                at: None,
            };
            h.tensors = vec![t("down", vec![w.rank, w.k], w.down, false), t("up", vec![w.n, w.rank], w.up, true)];
        }
        Err(e) => {
            h.editable = Err(format!("the unit's files did not pass their checks: {}", e.message));
            h.note = e.message;
        }
    }
    h
}

/// Reads the object a key names: a library cartridge, a job filter, an adapter library version or a
/// learning unit.
/// # Errors
/// Nothing of that key, or a file that does not read.
/// The key prefix of a MicroModel's upstream open weights in the parameter editor (`open:<model>`).
pub const OPEN_PREFIX: &str = "open:";

/// The block a tensor of the open weights belongs to, from its name.
fn open_block(name: &str) -> String {
    if name.contains("embed_tokens") {
        "table"
    } else if name.contains("self_attn") {
        "attention"
    } else if name.contains(".mlp.") {
        "mlp"
    } else if name.contains("norm") {
        "norm"
    } else if name.contains("lm_head") {
        "head"
    } else {
        "other"
    }
    .to_owned()
}

/// A MicroModel's upstream open weights, read in place from their safetensors files: every tensor
/// by name, layer, block, shape and type; values of small tensors read for their norms, larger ones
/// read by page. Never written: the open weights stay as released.
fn open_held(home: &Path, model: &str) -> Result<Held, String> {
    let folder = nmtype::models_dir(home).join(model).join("files");
    let idx = nmsafe::TensorIndex::open(&folder).map_err(|e| format!("{model}: the open weights are not on this machine or do not read ({e})"))?;
    let mut tensors = Vec::new();
    for t in idx.tensors.values() {
        let shape: Vec<usize> = t.shape.iter().filter_map(|x| usize::try_from(*x).ok()).collect();
        let layer = t.name.strip_prefix("model.layers.").and_then(|r| r.split('.').next()).and_then(|x| x.parse::<usize>().ok());
        let bytes = t.end.saturating_sub(t.start);
        let mut x = Tensor { name: t.name.clone(), target: t.name.clone(), layer, part: open_block(&t.name), shape, dtype: t.dtype.clone(), identity: 0.0, scales: false, values: None, bytes, unread: None, at: None };
        if elem(&t.dtype).is_some() {
            x.at = Some((t.file.clone(), t.start));
            if bytes <= READ_CAP {
                x.values = Some(read_range(&t.file, t.start, &t.dtype, 0, x.size())?);
            } else {
                x.unread = Some(format!("{} KiB, above the {} KiB read for norms; read it by page", bytes >> 10, READ_CAP >> 10));
            }
        } else {
            x.unread = Some(format!("{}: a stored type not read here", t.dtype));
        }
        tensors.push(x);
    }
    Ok(Held {
        key: format!("{OPEN_PREFIX}{model}"),
        source: "open-weights",
        kind: "open",
        model: model.to_owned(),
        name: format!("{model} open weights"),
        tensors,
        editable: Err("the upstream open weights are never written: edit, mask and freeze them in their overlay (POST /api/parameters/overlay), applied when the engine seats them; a new version is made from a base cartridge of this MicroModel, or built on Cartridges, Build".to_owned()),
        drops: false,
        state: "open weights".to_owned(),
        path: folder.display().to_string(),
        note: "the upstream open weights as released, read in place: every tensor its blocks read; tensors above the read cap are listed by shape and read by page".to_owned(),
        origin: Origin::ReadOnly,
    })
}

/// The MicroModels whose open weights are on this machine.
fn open_models(home: &Path) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(nmtype::models_dir(home))
        .map(|rd| rd.flatten().filter(|e| e.path().join("files").is_dir()).filter_map(|e| e.file_name().to_str().map(str::to_owned)).collect())
        .unwrap_or_default();
    out.sort();
    out
}

pub fn held(home: &Path, key: &str) -> Result<Held, String> {
    if let Some(rest) = key.strip_prefix(FILTER_PREFIX) {
        return job_of(home, rest);
    }
    if let Some(model) = key.strip_prefix(OPEN_PREFIX) {
        return open_held(home, model);
    }
    if let Ok(lib) = Library::open(home)
        && let Some(c) = lib.cartridge(key)
    {
        return Ok(library_held(&lib, c));
    }
    if let Some(r) = adapter_of(home, key) {
        return r;
    }
    if let Some(card) = nmlearn::Units::open(home).get(key) {
        return Ok(unit_held(home, card));
    }
    Err(format!("no cartridge, job filter, adapter or learning unit {key}"))
}

// ---------------------------------------------------------------------------------------------
// The read format.

/// Norms and a summary of a tensor's values.
#[derive(Clone, Debug, PartialEq)]
pub struct Stats {
    pub l2: f64,
    pub max_abs: f64,
    pub effect_l2: f64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub hist: Vec<u64>,
}

/// The norms and summary of `xs` (empty: all zero).
#[must_use]
pub fn stats(xs: &[f32], identity: f32) -> Stats {
    let (mut l2, mut eff, mut max_abs, mut sum) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for x in xs.iter().map(|x| f64::from(*x)) {
        l2 += x * x;
        let d = x - f64::from(identity);
        eff += d * d;
        max_abs = max_abs.max(x.abs());
        sum += x;
        lo = lo.min(x);
        hi = hi.max(x);
    }
    let mut hist = vec![0_u64; BINS];
    if xs.is_empty() {
        return Stats { l2: 0.0, max_abs: 0.0, effect_l2: 0.0, min: 0.0, max: 0.0, mean: 0.0, hist };
    }
    let width = (hi - lo) / BINS as f64;
    for x in xs.iter().map(|x| f64::from(*x)) {
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a bin index from a non-negative ratio, clamped to the bins")]
        let b = if width > 0.0 { (((x - lo) / width) as usize).min(BINS - 1) } else { 0 };
        if let Some(c) = hist.get_mut(b) {
            *c += 1;
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "a mean over a count of values")]
    let mean = sum / xs.len() as f64;
    Stats { l2: l2.sqrt(), max_abs, effect_l2: eff.sqrt(), min: lo, max: hi, mean, hist }
}

fn tensor_row(t: &Tensor, editable: bool) -> Value {
    let st = t.values.as_deref().map(|v| stats(v, t.identity));
    let mut row = vec![
        ("name", s(&t.name)),
        ("target", s(&t.target)),
        ("layer", t.layer.map_or(Value::Null, |l| n(l as u64))),
        ("part", s(&t.part)),
        ("shape", shape_v(&t.shape)),
        ("dtype", s(&t.dtype)),
        ("size", n(t.size() as u64)),
        ("bytes", n(t.bytes)),
        ("identity", f(f64::from(t.identity))),
        ("scales", Value::Bool(t.scales)),
        ("read", Value::Bool(t.values.is_some())),
        ("paged", Value::Bool(t.values.is_some() || t.at.is_some())),
        ("values_editable", Value::Bool(editable && t.values.is_some() && t.size() <= SMALL && t.dtype != "count")),
    ];
    match st {
        Some(x) => row.extend([
            ("l2", f(x.l2)),
            ("max_abs", f(x.max_abs)),
            ("effect_l2", f(x.effect_l2)),
            ("summary", obj(vec![("min", f(x.min)), ("max", f(x.max)), ("mean", f(x.mean)), ("histogram", obj(vec![("lo", f(x.min)), ("hi", f(x.max)), ("counts", Value::Array(x.hist.iter().map(|c| n(*c)).collect()))]))])),
        ]),
        None => row.extend([("l2", Value::Null), ("max_abs", Value::Null), ("effect_l2", Value::Null), ("summary", Value::Null), ("unread", t.unread.as_deref().map_or(Value::Null, s))]),
    }
    obj(row)
}

/// Per layer: the tensors at it, how many were read, and the L2 norm of what they change.
fn layer_rows(ts: &[Tensor]) -> Vec<Value> {
    let mut by: BTreeMap<usize, (u64, u64, f64, f64)> = BTreeMap::new();
    for t in ts {
        let Some(l) = t.layer else { continue };
        let e = by.entry(l).or_default();
        e.0 += 1;
        if let Some(v) = &t.values {
            let st = stats(v, t.identity);
            e.1 += 1;
            e.2 += st.effect_l2 * st.effect_l2;
            e.3 += st.l2 * st.l2;
        }
    }
    by.into_iter().map(|(l, (count, read, eff, l2))| obj(vec![("layer", n(l as u64)), ("tensors", n(count)), ("read", n(read)), ("effect_l2", f(eff.sqrt())), ("l2", f(l2.sqrt()))])).collect()
}

/// The role a MicroModel serves, by the roles in effect, else by its upstream id.
fn role_for(home: &Path, model: &str) -> String {
    nmtype::role_models::resolve(home).into_iter().find(|(_, m)| m == model).map_or_else(|| nmlearn::library::role_of(model).to_owned(), |(r, _)| r)
}

/// The `parameters` section of one object (`FORMAT`).
#[must_use]
pub fn section_of(home: &Path, h: &Held) -> Value {
    let ok = h.editable.is_ok();
    let values: u64 = h.tensors.iter().map(|t| t.size() as u64).sum();
    let read: Vec<&Tensor> = h.tensors.iter().filter(|t| t.values.is_some()).collect();
    let total = read.iter().filter_map(|t| t.values.as_deref().map(|v| stats(v, t.identity).l2.powi(2))).sum::<f64>().sqrt();
    let layered = h.tensors.iter().any(|t| t.layer.is_some());
    obj(vec![
        ("format", s(FORMAT)),
        ("key", s(&h.key)),
        ("root", s(&home.display().to_string())),
        ("source", s(h.source)),
        ("kind", s(h.kind)),
        ("model", s(&h.model)),
        ("role", s(&role_for(home, &h.model))),
        ("name", s(&h.name)),
        ("state", s(&h.state)),
        ("path", s(&h.path)),
        ("holds", Value::Bool(!h.tensors.is_empty())),
        ("note", s(if h.tensors.is_empty() && h.note.is_empty() { "this cartridge holds no parameters" } else { &h.note })),
        (
            "editable",
            obj(vec![
                ("scale", Value::Bool(ok && h.tensors.iter().any(|t| t.scales))),
                ("layers", Value::Bool(ok && layered)),
                ("values", Value::Bool(ok && h.tensors.iter().any(|t| t.values.is_some() && t.size() <= SMALL))),
                ("why", h.editable.as_ref().err().map_or(Value::Null, |w| s(w))),
                ("layer_off", s(if h.drops { "drops the layer's points" } else { "zeroes the layer's change" })),
                ("scale_max", f(f64::from(SCALE_MAX))),
                ("small", n(SMALL as u64)),
            ]),
        ),
        ("totals", obj(vec![("tensors", n(h.tensors.len() as u64)), ("read", n(read.len() as u64)), ("values", n(values)), ("l2", f(total))])),
        ("layers", Value::Array(layer_rows(&h.tensors))),
        ("tensors", Value::Array(h.tensors.iter().map(|t| tensor_row(t, ok)).collect())),
        ("page", obj(vec![("route", s("/api/cartridges/parameters?key=<key>&tensor=<name>&offset=<n>&limit=<n>")), ("max", n(PAGE_MAX as u64))])),
        ("held_out", held_out(h)),
        // The object's parameter overlay (`crate::overlay`): entries, the marks per tensor and layer,
        // and when a change takes effect. Edits of the open weights are taken here.
        ("overlay", crate::overlay::section(home, h)),
    ])
}

/// What a held-out check of this object needs.
fn held_out(h: &Held) -> Value {
    match h.kind {
        "filter" => obj(vec![("runs_here", Value::Bool(false)), ("needs", s("the graphics card")), ("how", s("a job filter's held-out split gate runs in its build (nm_cartridges filter_build) on the graphics card"))]),
        "lens" | "readout" => obj(vec![("runs_here", Value::Bool(false)), ("needs", s("the bound MicroModel")), ("how", s("an adapter's control runs in its idle fit in a serving process"))]),
        _ => Value::Null,
    }
}

/// `GET /api/cartridges/parameters?key=<key>&head=1`: how many tensors opening the object reads,
/// from its header or label alone (no value is read), so a view can say what it is reading while the
/// section is made. `tensors` is null where only opening the object tells.
#[must_use]
pub fn head_count(home: &Path, key: &str) -> Value {
    let count = if let Some(model) = key.strip_prefix(OPEN_PREFIX) {
        nmsafe::TensorIndex::open(&nmtype::models_dir(home).join(model).join("files")).ok().map(|i| i.tensors.len())
    } else if key.starts_with(FILTER_PREFIX) {
        None
    } else {
        Library::open(home).ok().and_then(|lib| {
            let c = lib.cartridge(key)?;
            if c.kind != "tensors" {
                return None;
            }
            let file = lib.cartridge_dir(c).join(&c.files.first()?.name);
            head(&file).ok().map(|(l, _)| l.get("parts").and_then(Value::array).map_or(0, <[Value]>::len))
        })
    };
    obj(vec![("key", s(key)), ("tensors", count.map_or(Value::Null, |c| n(c as u64))), ("root", s(&home.display().to_string()))])
}

/// `parameters` of the object a key names; an object of `error` when it does not read.
#[must_use]
pub fn section(home: &Path, key: &str) -> Value {
    match held(home, key) {
        Ok(h) => section_of(home, &h),
        Err(e) => obj(vec![("format", s(FORMAT)), ("key", s(key)), ("holds", Value::Bool(false)), ("error", s(&e))]),
    }
}

/// One page of a tensor's values.
/// # Errors
/// An unknown object or tensor, values that were not read and have no file, or a read failure.
pub fn page(home: &Path, key: &str, tensor: &str, offset: usize, limit: usize) -> Result<Value, String> {
    let h = held(home, key)?;
    let t = h.tensors.iter().find(|t| t.name == tensor).ok_or_else(|| format!("{key} has no tensor {tensor}"))?;
    let size = t.size();
    let offset = offset.min(size);
    let limit = limit.clamp(1, PAGE_MAX).min(size - offset);
    let vals = match (&t.values, &t.at) {
        (Some(v), _) => v.get(offset..offset + limit).map(<[f32]>::to_vec).unwrap_or_default(),
        (None, Some((file, at))) => read_range(file, *at, &t.dtype, offset, limit)?,
        (None, None) => return Err(format!("{tensor}: {}", t.unread.as_deref().unwrap_or("its values are not held here"))),
    };
    Ok(obj(vec![
        ("key", s(&h.key)),
        ("tensor", s(tensor)),
        ("shape", shape_v(&t.shape)),
        ("dtype", s(&t.dtype)),
        ("size", n(size as u64)),
        ("offset", n(offset as u64)),
        ("limit", n(limit as u64)),
        ("values", Value::Array(vals.iter().map(|x| f(f64::from(*x))).collect())),
    ]))
}

/// The label fields of a job filter file, read from the text before its `parts` (the files are
/// written with their keys in name order, so the label comes first); the gain and bias values are not
/// read. `None` when the head does not read as a label.
fn job_label(path: &Path) -> Option<Value> {
    let mut b = Vec::new();
    std::fs::File::open(path).ok()?.take(64 << 10).read_to_end(&mut b).ok()?;
    let text = String::from_utf8_lossy(&b);
    let head = text.find("\"parts\"").map_or(&text[..0], |i| &text[..i]);
    let head = head.trim_end().trim_end_matches(',');
    if head.trim().is_empty() {
        return None;
    }
    nmjson::parse(format!("{head}}}").as_bytes()).ok()
}

/// Bytes on disk of the files directly in a folder.
fn folder_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir).map(|rd| rd.flatten().filter_map(|e| e.metadata().ok()).filter(|m| m.is_file()).map(|m| m.len()).sum()).unwrap_or(0)
}

/// `GET /api/cartridges/parameters`: every object that holds parameters, per MicroModel, listed from
/// its label or catalogue record alone: key, kind, MicroModel, state, the layer range where the label
/// records it, and its size on disk. No tensor, value or norm is read for the list; an object's
/// parameters are read only when it is opened (`section`, `page`, `region`). The overlay held for each
/// object (`memory/overlays/`) is counted by its entries, never applied here.
#[must_use]
pub fn list(home: &Path) -> Value {
    let mut rows = Vec::new();
    let range_v = |r: Option<(u64, u64)>| r.map_or(Value::Null, |(a, b)| Value::Array(vec![n(a), n(b)]));
    let row = |key: &str, source: &str, kind: &str, model: &str, name: &str, state: &str, editable: bool, range: Value, bytes: u64| {
        obj(vec![
            ("key", s(key)),
            ("source", s(source)),
            ("kind", s(kind)),
            ("model", s(model)),
            ("role", s(&role_for(home, model))),
            ("name", s(name)),
            ("state", s(state)),
            ("editable", Value::Bool(editable)),
            ("layer_range", range),
            ("bytes", n(bytes)),
            ("read", Value::Bool(false)),
            ("overlay", crate::overlay::counts(home, key)),
        ])
    };
    // The upstream open weights of each MicroModel, by name and size on disk; their header is not read.
    for model in open_models(home) {
        let bytes = folder_bytes(&nmtype::models_dir(home).join(&model).join("files"));
        rows.push(row(&format!("{OPEN_PREFIX}{model}"), "open-weights", "open", &model, &format!("{model} open weights"), "open weights", false, Value::Null, bytes));
    }
    let lib = Library::open(home).ok();
    if let Some(lib) = &lib {
        for c in &lib.cartridges {
            let kind = match c.kind.as_str() {
                "tensors" => "base",
                "affine" if c.family == nmlearn::library::ADAPTER => "adapter",
                "affine" => "filter",
                _ if c.family == nmlearn::library::PREFERENCE => "preference",
                // Operating settings hold no parameters.
                _ => continue,
            };
            let bytes = c.files.iter().map(|f| f.bytes).sum();
            rows.push(row(&c.key(), "library", kind, &c.model, &c.name, if c.trial { "trial" } else { "served" }, true, range_v(c.layers), bytes));
        }
    }
    for role in ROLES {
        for (job, path) in job_files(home, role) {
            let label = job_label(&path);
            let lt = |k: &str| label.as_ref().map(|l| text(l, k)).unwrap_or_default();
            let model = Some(lt("model")).filter(|m| !m.is_empty()).unwrap_or_else(|| nmtype::role_models::model(role));
            let state = match &label {
                Some(l) if matches!(l.get("enabled"), Ok(Value::Bool(false))) => "plugged out",
                Some(_) => "plugged in",
                None => "label not read",
            };
            let bytes = std::fs::metadata(&path).map_or(0, |m| m.len());
            rows.push(row(&format!("{FILTER_PREFIX}{role}/{job}"), "job-filter", "filter", &model, &lt("cartridge"), state, true, Value::Null, bytes));
        }
    }
    if let Ok(rd) = std::fs::read_dir(home.join("library/cartridges/adapter")) {
        let mut ids: Vec<String> = rd.flatten().filter(|e| e.path().is_dir()).filter_map(|e| e.file_name().to_str().map(str::to_owned)).collect();
        ids.sort();
        for id in ids {
            let all = nmapi::adapter::versions(home, &id);
            let newest = all.last().map(|(v, _)| *v);
            for (v, file) in &all {
                if lib.as_ref().is_some_and(|l| l.cartridge(&format!("{id}@{v}")).is_some()) {
                    continue;
                }
                let Ok(label) = nmlearn::library::label_of(file) else { continue };
                let kind = match text(&label, "kind").as_str() {
                    nmapi::adapter::LENS_KIND => "lens",
                    nmapi::adapter::READOUT_KIND => "readout",
                    _ => "adapter",
                };
                let state = if newest == Some(*v) { "newest version" } else { "earlier version" };
                let bytes = std::fs::metadata(file).map_or(0, |m| m.len());
                rows.push(row(&format!("{id}@{v}"), "adapter", kind, &text(&label, "model"), &id, state, true, Value::Null, bytes));
            }
        }
    }
    let units = nmlearn::Units::open(home);
    for card in &units.cards {
        let range = layer_of(&card.target).map(|l| (l as u64, l as u64));
        let bytes = if card.files.is_empty() { 0 } else { folder_bytes(&units.unit_dir(&card.id)) };
        rows.push(row(&card.id, "unit", "unit", &card.base, &card.id, &card.status, !card.files.is_empty(), range_v(range), bytes));
    }
    let models: Vec<Value> = nmtype::role_models::resolve(home).into_iter().map(|(r, m)| obj(vec![("role", s(&r)), ("model", s(&m))])).collect();
    obj(vec![
        ("format", s(FORMAT)),
        ("objects", Value::Array(rows)),
        ("roles", Value::Array(models)),
        ("read", s("labels only: an object's tensors, values and norms are read when it is opened (key)")),
        ("edit", s("POST /api/cartridges/parameters {key, scale?, layers?: {n: on|off|scale}, values?: {tensor: [..] | {offset, values}}}: a new version; the source is never written")),
        ("overlay", s("POST /api/parameters/overlay {key, action: edit|mask|unmask|freeze|unfreeze|clear|undo, ...}: held beside the object, applied when the engine seats it")),
    ])
}

// ---------------------------------------------------------------------------------------------
// Edits: requests read into operations of the selection engine (`pedit`).

/// An edit of an object's parameters: operations in order, the name and description of the new
/// version, and whether only the difference is wanted (nothing written).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Edit {
    pub ops: Vec<crate::pedit::Op>,
    pub name: Option<String>,
    pub description: Option<String>,
    pub preview: bool,
}
impl Edit {
    /// What the edit does, in short phrases.
    #[must_use]
    pub fn summary(&self) -> Vec<String> {
        let mut out: Vec<String> = self.ops.iter().map(crate::pedit::Op::phrase).collect();
        if let Some(x) = &self.name {
            out.push(format!("name {x}"));
        }
        if self.description.is_some() {
            out.push("description".to_owned());
        }
        out
    }
}

fn scale_ok(x: f64, what: &str) -> Result<f32, String> {
    if x.is_finite() && (0.0..=f64::from(SCALE_MAX)).contains(&x) { Ok(narrow(x)) } else { Err(format!("{what}: a scale from 0 to {SCALE_MAX}")) }
}

fn values_of(v: &Value, what: &str) -> Result<Vec<f32>, String> {
    let a = v.array().map_err(|_| format!("{what}: give the values as a list of numbers"))?;
    a.iter()
        .map(|x| match num(x) {
            Some(y) if y.is_finite() && y.abs() <= f64::from(f32::MAX) => Ok(narrow(y)),
            _ => Err(format!("{what}: every value a finite number")),
        })
        .collect()
}

/// Reads an edit from a request as the API, MCP and the command line send it: one operation (`op`
/// with `selection` or `path`), a list of them (`ops`), and the short forms `scale` (a strength for
/// every scalable tensor), `layers` (`{n: on | off | scale}`) and `values` (`{tensor: [every value]
/// | {offset, values}}`); `name` and `description` label the new version; `preview: true` asks for
/// the difference only.
/// # Errors
/// An operation or a field that does not read, or nothing to do.
pub fn parse_edit(v: &Value) -> Result<Edit, String> {
    use crate::pedit::{Op, OpKind, Sel};
    let mut e = Edit { preview: matches!(v.get("preview"), Ok(Value::Bool(true))), ..Edit::default() };
    match v.get("scale") {
        Ok(Value::Null) | Err(_) => {}
        Ok(x) => e.ops.push(Op { sel: Sel::default(), kind: OpKind::Scale(scale_ok(num(x).ok_or("scale: a number")?, "scale")?) }),
    }
    if let Ok(Value::Object(m)) = v.get("layers") {
        for (k, x) in m {
            let l = k.trim().trim_start_matches('L').parse::<usize>().map_err(|_| format!("layers: {k} is not a layer number"))?;
            let sel = Sel { layer: Some(l), ..Sel::default() };
            match x {
                Value::String(t) if t == "on" => {}
                Value::Bool(true) => {}
                Value::String(t) if t == "off" => e.ops.push(Op { sel, kind: OpKind::Drop }),
                Value::Bool(false) => e.ops.push(Op { sel, kind: OpKind::Drop }),
                other => e.ops.push(Op { sel, kind: OpKind::Scale(scale_ok(num(other).ok_or_else(|| format!("layers: {k} is on, off or a scale"))?, &format!("layer {l}"))?) }),
            }
        }
    }
    if let Ok(Value::Object(m)) = v.get("values") {
        for (k, x) in m {
            let (range, vals) = match x {
                Value::Array(_) => (None, values_of(x, k)?),
                Value::Object(_) => {
                    let offset = x.get("offset").ok().and_then(num).filter(|o| *o >= 0.0 && o.fract() == 0.0).ok_or_else(|| format!("{k}: offset a whole number"))?;
                    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a whole non-negative offset")]
                    let offset = offset as usize;
                    let vals = values_of(x.get("values").map_err(|_| format!("{k}: values"))?, k)?;
                    (Some((offset, offset + vals.len())), vals)
                }
                _ => return Err(format!("{k}: give a list of values, or {{offset, values}}")),
            };
            e.ops.push(Op { sel: Sel { tensor: Some(k.clone()), range, ..Sel::default() }, kind: OpKind::Paste(vals) });
        }
    }
    let one = |o: &Value| if o.get("op").and_then(Value::string).ok() == Some("import") { import_op(o) } else { crate::pedit::parse_op(o) };
    if v.get("op").is_ok() {
        e.ops.push(one(v)?);
    }
    if let Ok(Value::Array(list)) = v.get("ops") {
        for o in list {
            e.ops.push(one(o)?);
        }
    }
    let label = |k: &str| v.get(k).and_then(Value::string).ok().map(str::trim).filter(|x| !x.is_empty()).map(str::to_owned);
    e.name = label("name");
    e.description = label("description");
    if e.name.as_ref().is_some_and(|x| x.len() > 120) || e.description.as_ref().is_some_and(|x| x.len() > 1000 || x.contains('\u{2014}')) {
        return Err("name: at most 120 characters; description: at most 1,000, no em dashes".to_owned());
    }
    if e.ops.is_empty() && e.name.is_none() && e.description.is_none() {
        return Err("give op (with selection or path), ops, scale, layers, values, name or description".to_owned());
    }
    Ok(e)
}

/// Values of another object, or of a parameter map, at a tensor's place.
pub(crate) fn source_values(home: &Path, src: &crate::pedit::Source, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
    let ts = match src {
        crate::pedit::Source::Object(k) => held(home, k)?.tensors,
        crate::pedit::Source::Map(m) => {
            let (role, file) = m.split_once('/').ok_or_else(|| format!("map {m}: give <role>/<file>"))?;
            if !ROLES.contains(&role) || file.contains(['/', '\\']) || !file.starts_with("job-filter-") {
                return Err(format!("map {m}: values at the same place come from a job filter map (<role>/job-filter-<job>.json)"));
            }
            let doc = std::fs::read(job_dir(home, role).join(file)).map_err(|e| format!("{m}: {e}")).and_then(|b| nmjson::parse(&b).map_err(|e| format!("{m}: {e}")))?;
            parse_points(&doc)?.into_iter().flat_map(|p| point_tensors(&p.target, p.gain, p.bias, "F32")).collect()
        }
    };
    let t = ts.into_iter().find(|t| t.name == name).ok_or_else(|| format!("{} holds no values at {name}", match src { crate::pedit::Source::Object(k) => k.clone(), crate::pedit::Source::Map(m) => format!("map {m}") }))?;
    if t.shape != shape {
        return Err(format!("shape check: {name} is {:?} there, {shape:?} here", t.shape));
    }
    // A tensor above the read cap (a base cartridge's part, a tensor of the open weights) is read
    // whole from its file, up to the overlay's capture limit.
    match (t.values, &t.at) {
        (Some(v), _) => Ok(v),
        (None, Some((file, at))) if t.shape.iter().product::<usize>() <= crate::overlay::CAPTURE_MAX => read_range(file, *at, &t.dtype, 0, t.shape.iter().product()),
        (None, Some(_)) => Err(format!("{name}: {} values, more than the {} read from a source at once", t.shape.iter().product::<usize>(), crate::overlay::CAPTURE_MAX)),
        (None, None) => Err(format!("{name}: its values are not held there")),
    }
}

/// The edited tensors (the source's are untouched).
/// # Errors
/// As `pedit::run`, or a layer edit on an object without layers.
pub fn apply(home: &Path, h: &Held, e: &Edit) -> Result<Vec<Tensor>, String> {
    use crate::pedit::OpKind;
    let layered = h.tensors.iter().any(|t| t.layer.is_some());
    let mut ops = e.ops.clone();
    for op in &mut ops {
        if op.sel.layer.is_some() && !layered {
            return Err(format!("{} has no layers: it acts on {}", h.key, h.tensors.first().map_or("one node", |t| t.target.as_str())));
        }
        // A layer switched off where the form holds a fixed set of tensors zeroes its change.
        if op.kind == OpKind::Drop && !h.drops && op.sel.layer.is_some() && op.sel.tensor.is_none() {
            op.kind = OpKind::Scale(0.0);
        }
    }
    let src = |s: &crate::pedit::Source, name: &str, shape: &[usize]| source_values(home, s, name, shape);
    let ts = crate::pedit::run(&h.tensors, &ops, h.drops, SMALL, &src)?;
    if h.kind == "preference" {
        for t in &ts {
            let v = t.values.as_deref().unwrap_or_default();
            if t.shape != [2] || v.iter().any(|x| *x < 0.0 || x.fract() != 0.0) || v.first() > v.get(1) {
                return Err(format!("{}: a preference is [good, signals], whole numbers with good at most signals", t.name));
            }
        }
    }
    Ok(ts)
}

/// The assembly name of a MicroModel's open weights (`qwen3-reranker-0.6b` to `qwen3-reranker`).
fn assembly_of(model: &str) -> &str {
    model.strip_suffix("-0.6b").or_else(|| model.strip_suffix("-1.7b")).unwrap_or(model)
}

/// Checks new points against the MicroModel's sockets: a node the model's assembly does not have is
/// refused. `None` when the assembly could not be read (the check is then reported as not run).
fn sockets_check(home: &Path, model: &str, added: &[String]) -> Result<Option<String>, String> {
    if added.is_empty() {
        return Ok(None);
    }
    let Ok(asm) = nmengn::Assembly::load(&home.join("designs").join("assemblies"), assembly_of(model)) else {
        return Ok(Some(format!("sockets not checked: no assembly for {model} here")));
    };
    if let Some(t) = added.iter().find(|t| !asm.nodes.iter().any(|n| n.id == nmengn::assembly::nid(t))) {
        return Err(format!("{model} has no socket {t}: the point is refused"));
    }
    Ok(Some(format!("sockets checked: {} on {model}", added.join(", "))))
}

/// The points of edited point tensors, in the source's order.
fn points_of(ts: &[Tensor]) -> Vec<Point> {
    let mut out: Vec<Point> = Vec::new();
    for t in ts {
        let vals = t.values.clone().unwrap_or_default();
        let i = match out.iter().position(|p| p.target == t.target) {
            Some(i) => i,
            None => {
                out.push(Point { target: t.target.clone(), gain: Vec::new(), bias: Vec::new() });
                out.len() - 1
            }
        };
        if let Some(p) = out.get_mut(i) {
            if t.part == "gain" {
                p.gain = vals;
            } else {
                p.bias = vals;
            }
        }
    }
    out
}

fn compat_row(home: &Path, key: &str) -> Value {
    crate::library::compat(home).get("cartridges").and_then(Value::array).unwrap_or_default().iter().find(|r| text(r, "key") == key).cloned().unwrap_or(Value::Null)
}

/// Makes the edited object's new version. Returns (new key, its file, compatibility).
/// Most values of base parts one edit may read and change (32 Mi values, 128 MiB as f32).
const BASE_EDIT_CAP: usize = 32 << 20;

/// Reads the values of the base parts the edit's operations select, so the edit runs on them; a part
/// taken from the open weights is refused (this cartridge holds no values of its own for it), and
/// so is an edit over more than `BASE_EDIT_CAP` values (select a tensor, a layer or a range).
fn load_for_edit(h: &mut Held, e: &Edit) -> Result<(), String> {
    let mut need = 0_usize;
    let mut picked = Vec::new();
    for (i, t) in h.tensors.iter().enumerate() {
        if !e.ops.iter().any(|op| crate::pedit::picks(&op.sel, t)) {
            continue;
        }
        if t.values.is_some() {
            continue;
        }
        if t.at.is_none() {
            return Err(format!("{}: taken from the open weights at the seat; this base cartridge holds no values of its own for it to edit", t.name));
        }
        need += t.size();
        picked.push(i);
    }
    if need > BASE_EDIT_CAP {
        return Err(format!("the edit selects {need} values of base parameter sets; at most {BASE_EDIT_CAP} are edited at once: select a tensor, a layer or a range"));
    }
    for i in picked {
        if let Some(t) = h.tensors.get_mut(i)
            && let Some((file, at)) = t.at.clone()
        {
            t.values = Some(read_range(&file, at, &t.dtype, 0, t.size())?);
            t.unread = None;
        }
    }
    Ok(())
}

/// f32 to IEEE half, rounded to nearest even (the engine's `half::from16` reads it back).
fn to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1_u32 << (shift - 1);
        let rest = m & ((1 << shift) - 1);
        let mut v = m >> shift;
        if rest > half || (rest == half && v & 1 == 1) {
            v += 1;
        }
        return sign | v as u16;
    }
    let mut v = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && v & 1 == 1) {
        v += 1;
    }
    sign | v as u16
}

/// Values as the stored element type (`BF16` rounded to nearest even, `F16`, `F32`).
fn encode(dtype: &str, v: &[f32]) -> Result<Vec<u8>, String> {
    Ok(match dtype {
        "BF16" => v
            .iter()
            .flat_map(|x| {
                let b = x.to_bits();
                let r = if x.is_nan() { (b >> 16) | 0x40 } else { (b + 0x7fff + ((b >> 16) & 1)) >> 16 };
                (r as u16).to_le_bytes()
            })
            .collect(),
        "F16" => v.iter().flat_map(|x| to_f16(*x).to_le_bytes()).collect(),
        "F32" => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        other => return Err(format!("{other}: not a stored type written here")),
    })
}

/// Writes an edited base cartridge as the next version beside the others, on trial: the source
/// version's parameter sets with the edited ones replaced, through the library's own base writer.
fn write_base(home: &Path, h: &Held, c: &nmlearn::library::Cartridge, ts: &[Tensor], e: &Edit, by: &str) -> Result<(String, PathBuf, Value), String> {
    let summary = e.summary().join(", ");
    let source_bytes: u64 = c.files.iter().map(|f| f.bytes).sum();
    if let Some((free, _)) = nmffi::dynlib::disk_space(home) {
        let floor = nmapi::storage::floor_gb(home) << 30;
        if free < source_bytes.saturating_add(floor) {
            return Err(format!(
                "a new base version takes {:.1} GB; the drive has {:.1} GB free and keeps a {} GB floor: prune rebuildable storage first",
                source_bytes as f64 / f64::from(1 << 30),
                free as f64 / f64::from(1 << 30),
                floor >> 30
            ));
        }
    }
    let changed: Vec<&Tensor> = ts.iter().filter(|t| h.tensors.iter().find(|o| o.name == t.name).is_none_or(|o| o.values != t.values)).collect();
    let (mut lib, _g) = crate::library::for_change(home)?;
    let mut filter = lib.filter(c).map_err(|x| x.to_string())?;
    let sets: &mut Vec<nmlearn::library::Tensor> = match &mut filter {
        Filter::Tensors(v) => v,
        Filter::Open { edited, .. } => edited,
        _ => return Err(format!("{}: not a base cartridge of parameter sets", c.key())),
    };
    for t in &changed {
        let values = t.values.as_deref().ok_or_else(|| format!("{}: its values were not read", t.name))?;
        let set = sets.iter_mut().find(|x| x.name == t.target && x.layer == t.layer && x.block == t.part).ok_or_else(|| format!("{}: this base cartridge holds no values of its own for it (taken from the open weights)", t.name))?;
        let bytes = encode(&set.dtype, values)?;
        if bytes.len() != set.bytes.len() {
            return Err(format!("{}: {} values do not fit its {:?} shape", t.name, values.len(), set.shape));
        }
        set.bytes = bytes;
    }
    let spec = CartridgeSpec {
        id: c.id.clone(),
        version: crate::library::next_version(&lib, &c.id, false),
        family: c.family.clone(),
        domain: c.domain.clone(),
        scope: c.scope.clone(),
        model: c.model.clone(),
        weights: c.weights.clone(),
        layers: c.layers,
        name: e.name.clone().unwrap_or_else(|| c.name.clone()),
        description: e.description.clone().unwrap_or_else(|| c.description.clone()),
        made_from: format!("{} with {summary} ({by})", c.key()),
        effect: "parameters edited by hand; not measured; checked against the upstream open weights when it is seated".to_owned(),
    };
    let notes = obj(vec![("edited_from", s(&c.key())), ("edited_by", s(by)), ("parameter_edit", s(&summary)), ("parent", s(&c.key()))]);
    let (key, _) = lib.add_base_cartridge(&spec, &filter, notes, Vec::new(), true).map_err(|x| x.to_string())?;
    lib.save().map_err(|x| x.to_string())?;
    let file = lib.cartridge(&key).and_then(|nc| nc.files.first().map(|x| lib.cartridge_dir(nc).join(&x.name))).unwrap_or_default();
    drop(lib);
    let compat = compat_row(home, &key);
    Ok((key.clone(), file, if compat == Value::Null { obj(vec![("key", s(&key))]) } else { compat }))
}

fn write_new(home: &Path, h: &Held, ts: &[Tensor], e: &Edit, by: &str) -> Result<(String, PathBuf, Value), String> {
    if let Origin::Base { c } = &h.origin {
        return write_base(home, h, c, ts, e, by);
    }
    let summary = e.summary().join(", ");
    // A point added at a node the MicroModel does not have is refused before anything is written.
    let sockets = sockets_check(home, &h.model, &added_targets(h, ts))?;
    let with_sockets = |mut v: Value| {
        if let (Value::Object(o), Some(x)) = (&mut v, &sockets) {
            o.insert("sockets".into(), s(x));
        }
        v
    };
    match &h.origin {
        Origin::Library { c, q8 } => {
            let (mut lib, _g) = crate::library::for_change(home)?;
            let points = points_of(ts);
            let filter = if *q8 { Filter::points_q8(points) } else { Filter::Points { points, q8: None } };
            let spec = CartridgeSpec {
                id: c.id.clone(),
                version: crate::library::next_version(&lib, &c.id, false),
                family: c.family.clone(),
                domain: c.domain.clone(),
                scope: c.scope.clone(),
                model: c.model.clone(),
                weights: c.weights.clone(),
                layers: c.layers,
                name: e.name.clone().unwrap_or_else(|| c.name.clone()),
                description: e.description.clone().unwrap_or_else(|| c.description.clone()),
                made_from: format!("the parameters of {} with {summary} ({by})", c.key()),
                effect: "parameters edited by hand; not measured".to_owned(),
            };
            let notes = obj(vec![("edited_from", s(&c.key())), ("edited_by", s(by)), ("parameter_edit", s(&summary))]);
            let key = lib.add_cartridge(&spec, &filter, notes, Vec::new(), true).map_err(|x| x.to_string())?;
            lib.save().map_err(|x| x.to_string())?;
            let file = lib.cartridge(&key).and_then(|nc| nc.files.first().map(|x| lib.cartridge_dir(nc).join(&x.name))).unwrap_or_default();
            drop(lib);
            let compat = compat_row(home, &key);
            Ok((key.clone(), file, with_sockets(if compat == Value::Null { obj(vec![("key", s(&key))]) } else { compat })))
        }
        Origin::Job { role, file, doc } => {
            let dir = job_dir(home, role);
            let job = file.strip_prefix("job-filter-").and_then(|x| x.strip_suffix(".json")).unwrap_or(file);
            let taken = |j: &str| {
                let name = format!("job-filter-{j}.json");
                dir.join(&name).exists() || dir.join(crate::pmaps::ARCHIVE).join(&name).exists()
            };
            let p = (2..1000).find(|p| !taken(&format!("{job}-p{p}"))).ok_or("no free name for the new job filter")?;
            let new_job = format!("{job}-p{p}");
            let id = text(doc, "cartridge").split_once('@').map_or_else(|| format!("{role}-{job}"), |(i, _)| i.to_owned());
            let cart = format!("{id}-p{p}@1");
            let mut d = doc.clone();
            if let Value::Object(o) = &mut d {
                o.insert("job".into(), s(&new_job));
                o.insert("cartridge".into(), s(&cart));
                o.insert("enabled".into(), Value::Bool(false));
                o.insert("edited_from".into(), s(file));
                o.insert("parameter_edit".into(), s(&summary));
                o.insert("edited_by".into(), s(by));
                if let Some(x) = &e.name {
                    o.insert("name".into(), s(x));
                }
                if let Some(x) = &e.description {
                    o.insert("description".into(), s(x));
                }
                o.insert(
                    "parts".into(),
                    Value::Array(points_of(ts).into_iter().map(|pt| obj(vec![("target", s(&pt.target)), ("gain", Value::Array(pt.gain.iter().map(|x| f(f64::from(*x))).collect())), ("bias", Value::Array(pt.bias.iter().map(|x| f(f64::from(*x))).collect()))])).collect()),
                );
            }
            let path = dir.join(format!("job-filter-{new_job}.json"));
            let bytes = nmjson::pretty::pretty(&d);
            write_once(&path, bytes.as_bytes())?;
            Ok((format!("{FILTER_PREFIX}{role}/{new_job}"), path, with_sockets(compat_row(home, &format!("filter:{cart}")))))
        }
        Origin::Adapter { id, version, label } => {
            let next = nmapi::adapter::versions(home, id).last().map_or(1, |(v, _)| v + 1);
            let mut l = label.clone();
            if let Value::Object(o) = &mut l {
                o.insert("version".into(), n(next));
                o.insert("made".into(), s(&nmlearn::cartridge::date_of(nmtype::now_ms())));
                if let Some(x) = &e.name {
                    o.insert("name".into(), s(x));
                }
                if let Some(x) = &e.description {
                    o.insert("does".into(), s(x));
                }
                if let Some(Value::Object(notes)) = o.get_mut("notes") {
                    notes.insert("parent".into(), s(&format!("{id}@{version}")));
                    notes.insert("edited_from".into(), s(&format!("{id}@{version}")));
                    notes.insert("parameter_edit".into(), s(&summary));
                    notes.insert("edited_by".into(), s(by));
                }
            }
            let parts: Vec<nmapi::adapter::Part> = ts.iter().map(|t| (t.name.clone(), t.shape.clone(), t.values.clone().unwrap_or_default())).collect();
            let bytes = nmapi::adapter_fit::encode_parts(l, &parts);
            let (label2, parts2) = nmapi::adapter::read_bytes(&bytes).map_err(|x| x.message)?;
            let values = text(label2.get("notes").unwrap_or(&Value::Null), "values");
            let check = if h.kind == "lens" { nmapi::adapter::lens_from(&label2, &parts2, &values).map(|_| ()) } else { nmapi::adapter::readout_from(&label2, &parts2, &values).map(|_| ()) };
            let path = nmapi::adapter::shelf_file(home, id, next);
            write_once(&path, &bytes)?;
            nmapi::adapter::look_again();
            let compat = obj(vec![("key", s(&format!("{id}@{next}"))), ("made_for", s(&h.model)), ("values", s(&values)), ("shapes", s(&check.map_or_else(|x| x.message, |()| "fit".to_owned()))), ("loads", s("only on the base whose values key it carries; the newest version loads at the next look of a serving process"))]);
            Ok((format!("{id}@{next}"), path, compat))
        }
        Origin::Unit { card } => {
            if e.name.is_some() || e.description.is_some() {
                return Err("a learning unit's card has no name or description to set; it is named by its id".to_owned());
            }
            let mut units = nmlearn::Units::open(home);
            let p = (2..1000).find(|p| units.get(&format!("{}-p{p}", card.id)).is_none()).ok_or("no free id for the new unit")?;
            let id = format!("{}-p{p}", card.id);
            let rank = ts.iter().find(|t| t.name == "down").and_then(|t| t.shape.first().copied()).unwrap_or(card.rank);
            let mut nc = nmlearn::UnitCard::new(&id, &card.base, &card.base_sha256, &card.slot, &card.target, rank, card.seed);
            nc.scale = card.scale;
            nc.lineage = vec![card.id.clone()];
            units.register(nc).map_err(|x| x.message)?;
            let get = |name: &str| ts.iter().find(|t| t.name == name).and_then(|t| t.values.clone().map(|v| (v, t.shape.clone())));
            let ((down, ds), (up, us)) = (get("down").ok_or("no down")?, get("up").ok_or("no up")?);
            let mut files = Vec::new();
            for (name, m, sh) in [("down.q8", down, ds), ("up.q8", up, us)] {
                let (r, c) = (sh.first().copied().unwrap_or(0), sh.get(1).copied().unwrap_or(0));
                let b = nmlearn::UnitWeights::encode_q8(&m, r, c).map_err(|x| x.message)?;
                files.push((name.to_owned(), units.commit_file(&id, name, &b).map_err(|x| x.message)?));
            }
            units.set_files(&id, files, "candidate").map_err(|x| x.message)?;
            let model_sha = nmnpcs::WeightCard::load(home, &card.base).ok().map(|w| w.sha_main);
            let compat = obj(vec![("key", s(&id)), ("made_for", s(&card.base)), ("base_compatible", model_sha.as_deref().map_or(Value::Null, |m| Value::Bool(m == card.base_sha256))), ("loads", s("a candidate: it loads when a server starts with serve --unit"))]);
            Ok((id.clone(), units.unit_dir(&id), compat))
        }
        Origin::Base { .. } | Origin::ReadOnly => Err(h.editable.clone().err().unwrap_or_else(|| "read only".to_owned())),
    }
}

/// Writes a file that must not exist yet (through a temporary file, read back, renamed).
fn write_once(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if path.exists() {
        return Err(format!("{} exists: a file is never written over", path.display()));
    }
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).map_err(|e| format!("{}: {e}", tmp.display()))?;
    if std::fs::read(&tmp).map_err(|e| e.to_string())? != bytes {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("{}: read-back differs", tmp.display()));
    }
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}

/// `POST /api/cartridges/parameters {key, scale?, layers?, values?}`: edits an object's parameters
/// into a new version beside it (the source is never written), checks the new version's
/// compatibility, and writes one `cartridge-parameters` audit row (undo sets the new version aside).
/// # Errors
/// An unknown or read-only object, an edit that does not read or does not fit, or a write failure.
pub fn edit(home: &Path, key: &str, e: &Edit, by: &str) -> Result<Value, String> {
    let mut h = held(home, key)?;
    h.editable.clone()?;
    if matches!(h.origin, Origin::Base { .. }) {
        load_for_edit(&mut h, e)?;
    }
    let ts = apply(home, &h, e)?;
    let meta = e.name.is_some() || e.description.is_some();
    if e.preview {
        let sockets = match sockets_check(home, &h.model, &added_targets(&h, &ts)) {
            Ok(x) => x.map_or(Value::Null, |x| s(&x)),
            Err(x) => s(&format!("refused: {x}")),
        };
        let writes = match h.source {
            "library" if h.kind == "preference" => "a new version of the preference cartridge through its values (every use that ran it switches to it)",
            "library" if h.kind == "base" => "a new base cartridge version beside the others, on trial; checked against the upstream open weights when it is seated, chosen to serve on Base versions",
            "library" => "a new cartridge version on trial",
            "job-filter" => "a new job filter file, plugged out",
            "adapter" => "the next adapter version",
            _ => "a new candidate unit",
        };
        return Ok(obj(vec![
            ("preview", Value::Bool(true)),
            ("key", s(&h.key)),
            ("edit", Value::Array(e.summary().iter().map(|x| s(x)).collect())),
            ("diff", crate::pedit::diff(&h.tensors, &ts)),
            ("sockets", sockets),
            ("writes", s(writes)),
        ]));
    }
    if ts == h.tensors && !meta {
        return Ok(obj(vec![("changed", Value::Bool(false)), ("key", s(&h.key)), ("audit_id", Value::Null)]));
    }
    // Preference counts are values of a settings cartridge: they go through its own values path
    // (a new version, its uses switched to it, its own audit row and undo).
    if let (Origin::Library { c, .. }, "preference") = (&h.origin, h.kind) {
        return preference_edit(home, &h, c, &ts, e, by);
    }
    let (new_key, file, compat) = write_new(home, &h, &ts, e, by)?;
    let summary = e.summary();
    let label = format!("Parameters of {} edited ({}): {new_key}", h.key, summary.join(", "));
    let audit = crate::controls::audit(
        home,
        AUDIT_KIND,
        vec![
            ("action", s("edit")),
            ("source", s(h.source)),
            ("from", s(&h.key)),
            ("new", s(&new_key)),
            ("file", s(&file.display().to_string())),
            ("edit", Value::Array(summary.iter().map(|x| s(x)).collect())),
            ("by", s(by)),
            ("label", s(&label)),
        ],
    )?;
    let state = match h.source {
        "library" => "on trial: put it in a stack (nm_cartridges stack_save) or try it for one run (--try-cartridge); nothing that serves changed",
        "job-filter" => "plugged out: plug it in (nm_cartridges filter_on) and the source out to use it; nothing that serves changed",
        "adapter" => "the newest version: a serving process loads it at its next look when its values key fits; undo sets it aside",
        _ => "a candidate unit: it loads when a server starts with serve --unit; nothing that serves changed",
    };
    Ok(obj(vec![
        ("changed", Value::Bool(true)),
        ("audit_id", s(&audit)),
        ("from", s(&h.key)),
        ("key", s(&new_key)),
        ("file", s(&file.display().to_string())),
        ("edit", Value::Array(summary.iter().map(|x| s(x)).collect())),
        ("state", s(state)),
        ("compat", compat),
        ("parameters", section(home, &new_key)),
    ]))
}

/// The targets of points an edit adds.
fn added_targets(h: &Held, ts: &[Tensor]) -> Vec<String> {
    ts.iter().filter(|t| t.part == "gain" && !h.tensors.iter().any(|o| o.target == t.target)).map(|t| t.target.clone()).collect()
}

/// A preference edit, written through the cartridge's values (`library::set_values`).
fn preference_edit(home: &Path, h: &Held, c: &nmlearn::library::Cartridge, ts: &[Tensor], e: &Edit, by: &str) -> Result<Value, String> {
    let mut changes = BTreeMap::new();
    for t in ts.iter().filter(|t| h.tensors.iter().find(|o| o.name == t.name) != Some(*t)) {
        let v = t.values.clone().unwrap_or_default();
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "whole non-negative counts, checked in apply")]
        let (g, sg) = (v.first().copied().unwrap_or(0.0) as u64, v.get(1).copied().unwrap_or(0.0) as u64);
        changes.insert(t.name.clone(), s(&format!("{g}/{sg}")));
    }
    let mut r = crate::library::set_values(home, &c.key(), &changes, by)?;
    if let Value::Object(o) = &mut r {
        o.insert("edit".into(), Value::Array(e.summary().iter().map(|x| s(x)).collect()));
        o.insert("state".into(), s("a new version of the preference cartridge; every use that ran it now runs it; undo switches them back"));
        let k = o.get("key").and_then(|x| x.string().ok()).unwrap_or_default().to_owned();
        o.insert("parameters".into(), section(home, &k));
    }
    Ok(r)
}

/// Moves a file into `to` (made when absent), keeping its name.
fn set_aside(file: &Path, to: &Path) -> Result<PathBuf, String> {
    let name = file.file_name().ok_or("no file name")?;
    std::fs::create_dir_all(to).map_err(|e| format!("{}: {e}", to.display()))?;
    let mut dest = to.join(name);
    if dest.exists() {
        dest = to.join(format!("{}.{}", name.to_string_lossy(), nmtype::now_ms()));
    }
    std::fs::rename(file, &dest).map_err(|e| format!("{}: {e}", file.display()))?;
    Ok(dest)
}

/// Undoes a parameter edit: the new version is set aside (a library cartridge taken out of the
/// catalogue and its file moved to `library/set-aside/`, a job filter moved into its role's map
/// archive, an adapter version moved to `library/set-aside/`) or, for a unit, retired. Nothing is
/// deleted; the source was never written.
/// # Errors
/// A row without its new version, a stack holding the new cartridge, or a move failing.
pub fn undo(home: &Path, row: &Value, id: &str, by: &str) -> Result<Option<String>, String> {
    let (source, new) = (text(row, "source"), text(row, "new"));
    let file = PathBuf::from(text(row, "file"));
    let aside = home.join(SET_ASIDE);
    let moved = match source.as_str() {
        "library" => {
            let (mut lib, _g) = crate::library::for_change(home)?;
            if let Some(st) = lib.stacks.iter().find(|st| st.cartridges.contains(&new)) {
                return Err(format!("{new} is in stack {}; take it out of the stack first, then undo", st.key()));
            }
            let Some(c) = lib.cartridge(&new).cloned() else { return Err(format!("{new} is no longer in the library")) };
            let f = c.files.first().map(|x| lib.cartridge_dir(&c).join(&x.name)).ok_or("no file")?;
            lib.unlist(&new);
            lib.save().map_err(|e| e.to_string())?;
            if f.is_file() { set_aside(&f, &aside)?.display().to_string() } else { String::new() }
        }
        "job-filter" => {
            let role = new.strip_prefix(FILTER_PREFIX).and_then(|r| r.split_once('/')).map(|x| x.0.to_owned()).unwrap_or_default();
            if !file.is_file() {
                return Err(format!("{}: not there", file.display()));
            }
            set_aside(&file, &job_dir(home, &role).join(crate::pmaps::ARCHIVE))?.display().to_string()
        }
        "adapter" => {
            if !file.is_file() {
                return Err(format!("{}: not there", file.display()));
            }
            let d = set_aside(&file, &aside.join("adapter"))?;
            nmapi::adapter::look_again();
            d.display().to_string()
        }
        "unit" => {
            nmlearn::Units::open(home).set_status(&new, "retired").map_err(|e| e.message)?;
            String::new()
        }
        o => return Err(format!("{o}: not a parameter edit source")),
    };
    crate::controls::audit(
        home,
        AUDIT_KIND,
        vec![("action", s("undo")), ("source", s(&source)), ("new", s(&new)), ("moved_to", s(&moved)), ("undo_of", s(id)), ("by", s(by)), ("label", s(&format!("Undone: parameter edit {new} set aside")))],
    )
    .map(Some)
}


// ---------------------------------------------------------------------------------------------
// Regions and mapping: the values a selection names, and the parameter maps beside an object.

/// The values a selection names in its first tensor, read by region: from memory for the objects
/// held whole, by ranged reads of the file for a base cartridge part, so a large part is never
/// loaded whole.
pub(crate) fn region_values(t: &Tensor, idx: &[usize]) -> Result<Vec<f32>, String> {
    if let Some(v) = &t.values {
        return Ok(idx.iter().map(|i| v.get(*i).copied().unwrap_or(f32::NAN)).collect());
    }
    let Some((file, at)) = &t.at else { return Err(format!("{}: {}", t.name, t.unread.as_deref().unwrap_or("its values are not held here"))) };
    let mut out = Vec::with_capacity(idx.len());
    let mut k = 0;
    while let Some(&start) = idx.get(k) {
        let mut end = k + 1;
        while idx.get(end).is_some_and(|x| *x == start + (end - k)) {
            end += 1;
        }
        out.extend(read_range(file, *at, &t.dtype, start, end - k)?);
        k = end;
    }
    Ok(out)
}

/// `GET /api/cartridges/parameters?key=<key>&path=<path>[&offset=<n>]`: the values a selection
/// names (at most `PAGE_MAX`, from `offset` within the selection) in its first tensor, with their
/// flat indices, the tensor's shape and summary (the place of each value in its distribution), a
/// summary of the selection, and the values parameter maps hold at the same place.
/// # Errors
/// An unknown object, a path that does not read or selects nothing, or a read failure.
pub fn region(home: &Path, key: &str, path: &str, offset: usize) -> Result<Value, String> {
    let h = held(home, key)?;
    let sel = crate::pedit::parse_path(path)?;
    let t = h.tensors.iter().find(|t| crate::pedit::picks(&sel, t)).ok_or_else(|| format!("{path} selects no tensor of {key}"))?;
    let idx = crate::pedit::indices(&sel, t)?;
    let window: Vec<usize> = idx.iter().skip(offset).take(PAGE_MAX).copied().collect();
    let vals = region_values(t, &window)?;
    let st = stats(&vals, t.identity);
    let whole = t.values.as_deref().map(|v| stats(v, t.identity));
    Ok(obj(vec![
        ("key", s(&h.key)),
        ("path", s(path)),
        ("tensor", s(&t.name)),
        ("target", s(&t.target)),
        ("layer", t.layer.map_or(Value::Null, |l| n(l as u64))),
        ("shape", shape_v(&t.shape)),
        ("dtype", s(&t.dtype)),
        ("identity", f(f64::from(t.identity))),
        ("selected", n(idx.len() as u64)),
        ("offset", n(offset as u64)),
        ("indices", Value::Array(window.iter().map(|i| n(*i as u64)).collect())),
        ("values", Value::Array(vals.iter().map(|x| f(f64::from(*x))).collect())),
        ("summary", obj(vec![("l2", f(st.l2)), ("max_abs", f(st.max_abs)), ("effect_l2", f(st.effect_l2)), ("min", f(st.min)), ("max", f(st.max)), ("mean", f(st.mean))])),
        (
            "distribution",
            whole.map_or(Value::Null, |w| obj(vec![("lo", f(w.min)), ("hi", f(w.max)), ("mean", f(w.mean)), ("counts", Value::Array(w.hist.iter().map(|c| n(*c)).collect()))])),
        ),
        ("maps", Value::Array(map_values(home, &h, t, &window))),
        // `[value, masked, frozen]` per index with the object's overlay applied; null when the
        // overlay changes none of them.
        ("overlay", crate::overlay::region_marks(home, &h, t, &window, &vals)),
    ]))
}

/// The role folder of an object's parameter maps.
fn role_dir_of(home: &Path, h: &Held) -> PathBuf {
    let role = h.key.strip_prefix(FILTER_PREFIX).and_then(|r| r.split_once('/')).map_or_else(|| role_for(home, &h.model), |(r, _)| r.to_owned());
    job_dir(home, &role)
}

/// Values parameter maps hold at a tensor's place: a job filter map's values at the same point and
/// indices, and a per-layer map's value at the tensor's layer.
fn map_values(home: &Path, h: &Held, t: &Tensor, idx: &[usize]) -> Vec<Value> {
    let dir = role_dir_of(home, h);
    let own = PathBuf::from(&h.path);
    let mut out = Vec::new();
    for (file, path, bytes) in map_files(&dir) {
        if path == own {
            continue;
        }
        if file.starts_with("job-filter-") {
            let Some(doc) = (bytes <= MAP_READ).then(|| std::fs::read(&path).ok().and_then(|b| nmjson::parse(&b).ok())).flatten() else { continue };
            let Ok(points) = parse_points(&doc) else { continue };
            let Some(m) = points.into_iter().flat_map(|p| point_tensors(&p.target, p.gain, p.bias, "F32")).find(|m| m.name == t.name && m.shape == t.shape) else { continue };
            let vals = m.values.unwrap_or_default();
            out.push(obj(vec![("map", s(&file)), ("at", s("same point")), ("values", Value::Array(idx.iter().map(|i| vals.get(*i).map_or(Value::Null, |x| f(f64::from(*x)))).collect()))]));
        } else if let Some(l) = t.layer
            && let Some(v) = layer_values(&path, bytes).get(&l)
        {
            out.push(obj(vec![("map", s(&file)), ("at", s(&format!("layer {l}"))), ("value", f(*v))]));
        }
    }
    out
}

/// Bytes of a map read to overlay its values; larger maps are related by their lineage alone.
const MAP_READ: u64 = 4 << 20;

/// The map files of a role folder: (name, path, bytes), in name order.
fn map_files(dir: &Path) -> Vec<(String, PathBuf, u64)> {
    let mut v: Vec<(String, PathBuf, u64)> = std::fs::read_dir(dir)
        .map(|rd| rd.flatten().filter(|e| e.path().is_file()).filter_map(|e| Some((e.file_name().to_str()?.to_owned(), e.path(), e.metadata().ok()?.len()))).filter(|(n, _, _)| n.ends_with(".json") && n != "lineage.json").collect())
        .unwrap_or_default();
    v.sort();
    v
}

/// A per-layer map's value at each layer: keys `L<n>` (to a depth of three) holding a number, or a
/// list of numbers (their mean magnitude).
fn layer_values(path: &Path, bytes: u64) -> BTreeMap<usize, f64> {
    let mut out = BTreeMap::new();
    if bytes > MAP_READ {
        return out;
    }
    let Some(v) = std::fs::read(path).ok().and_then(|b| nmjson::parse(&b).ok()) else { return out };
    fn walk(v: &Value, depth: usize, out: &mut BTreeMap<usize, f64>) {
        let Value::Object(o) = v else { return };
        for (k, x) in o {
            if let Some(l) = k.strip_prefix('L').and_then(|d| d.parse::<usize>().ok()) {
                let val = match x {
                    Value::Array(a) => {
                        let xs: Vec<f64> = a.iter().filter_map(num).collect();
                        #[expect(clippy::cast_precision_loss, reason = "a mean over a count of values")]
                        let m = (!xs.is_empty()).then(|| xs.iter().map(|y| y.abs()).sum::<f64>() / xs.len() as f64);
                        m
                    }
                    other => num(other),
                };
                if let Some(val) = val {
                    out.entry(l).or_insert(val);
                }
            } else if depth < 3 {
                walk(x, depth + 1, out);
            }
        }
    }
    walk(&v, 0, &mut out);
    out
}

/// `GET /api/cartridges/parameters?key=<key>&maps=1`: the parameter maps beside an object, each with
/// why it relates (its lineage names the object or its base, a job filter map at the same points, a
/// map covering the same layers), its lineage, and its values on the same layers and points.
#[must_use]
pub fn maps(home: &Path, key: &str) -> Value {
    let h = match held(home, key) {
        Ok(h) => h,
        Err(e) => return obj(vec![("key", s(key)), ("error", s(&e))]),
    };
    let dir = role_dir_of(home, &h);
    let lineage = nmapi::lineage::read(&dir);
    let base = match &h.origin {
        Origin::Job { doc, .. } => text(doc, "base"),
        _ => String::new(),
    };
    let layers: std::collections::BTreeSet<usize> = h.tensors.iter().filter_map(|t| t.layer).collect();
    let targets: std::collections::BTreeSet<&str> = h.tensors.iter().map(|t| t.target.as_str()).collect();
    let own = PathBuf::from(&h.path);
    let mut rows = Vec::new();
    for (file, path, bytes) in map_files(&dir) {
        if path == own {
            continue;
        }
        let rec = lineage.get(&file);
        let mut why = Vec::new();
        if let Some(r) = rec {
            if r.made_on == h.key || (!base.is_empty() && r.made_on == base) {
                why.push(format!("measured on {}", r.made_on));
            }
            if r.used_by.iter().any(|u| *u == h.key || *u == h.name || u.split('@').next() == h.name.split('@').next()) {
                why.push("used by this object".to_owned());
            }
        }
        let mut overlay: Vec<Value> = Vec::new();
        let mut shared: Vec<String> = Vec::new();
        if file.starts_with("job-filter-") {
            if let Some(points) = (bytes <= MAP_READ).then(|| std::fs::read(&path).ok().and_then(|b| nmjson::parse(&b).ok()).and_then(|d| parse_points(&d).ok())).flatten() {
                let ts: Vec<Tensor> = points.into_iter().flat_map(|p| point_tensors(&p.target, p.gain, p.bias, "F32")).collect();
                shared = ts.iter().filter(|t| t.part == "gain" && targets.contains(t.target.as_str())).map(|t| t.target.clone()).collect();
                if !shared.is_empty() {
                    why.push(format!("job filter at {} of the same points", shared.len()));
                }
                overlay = layer_rows(&ts).into_iter().filter(|r| r.get("layer").and_then(Value::integer).ok().and_then(|l| usize::try_from(l).ok()).is_some_and(|l| layers.contains(&l))).map(|r| obj(vec![("layer", r.get("layer").cloned().unwrap_or(Value::Null)), ("value", r.get("effect_l2").cloned().unwrap_or(Value::Null))])).collect();
            }
        } else {
            let lv = layer_values(&path, bytes);
            overlay = lv.iter().filter(|(l, _)| layers.contains(l)).map(|(l, v)| obj(vec![("layer", n(*l as u64)), ("value", f(*v))])).collect();
            if !overlay.is_empty() {
                why.push(format!("covers {} of its layers", overlay.len()));
            }
        }
        if why.is_empty() {
            continue;
        }
        rows.push(obj(vec![
            ("file", s(&file)),
            ("kind", s(file.split('-').next().unwrap_or("map"))),
            ("bytes", n(bytes)),
            ("made_on", rec.map_or(Value::Null, |r| s(&r.made_on))),
            ("used_by", Value::Array(rec.map(|r| r.used_by.iter().map(|u| s(u)).collect()).unwrap_or_default())),
            ("purpose", rec.map_or(Value::Null, |r| s(&r.purpose))),
            ("why", Value::Array(why.iter().map(|x| s(x)).collect())),
            ("targets", Value::Array(shared.iter().map(|x| s(x)).collect())),
            ("layers", Value::Array(overlay)),
            ("read", Value::Bool(bytes <= MAP_READ)),
        ]));
    }
    obj(vec![("key", s(&h.key)), ("role_dir", s(&dir.display().to_string())), ("base", if base.is_empty() { Value::Null } else { s(&base) }), ("maps", Value::Array(rows))])
}

/// The read format of exported values.
pub const VALUES_FORMAT: &str = "nm-parameter-values/1";

/// `POST /api/cartridges/parameters/export {key, path}`: writes the values a selection names (in
/// its first tensor, at most `SMALL`) to `exports/parameters/` as `nm-parameter-values/1` (the key,
/// path, tensor, shape, flat indices and values), which `op: import` reads back. Nothing of the
/// object is written.
/// # Errors
/// As `region`, a selection of more than `SMALL` values, or a write failure.
pub fn export(home: &Path, key: &str, path: &str) -> Result<Value, String> {
    let h = held(home, key)?;
    let sel = crate::pedit::parse_path(path)?;
    let t = h.tensors.iter().find(|t| crate::pedit::picks(&sel, t)).ok_or_else(|| format!("{path} selects no tensor of {key}"))?;
    let idx = crate::pedit::indices(&sel, t)?;
    if idx.len() > SMALL {
        return Err(format!("{path} selects {} values; an export holds at most {SMALL}", idx.len()));
    }
    let vals = region_values(t, &idx)?;
    let doc = obj(vec![
        ("format", s(VALUES_FORMAT)),
        ("key", s(&h.key)),
        ("path", s(path)),
        ("tensor", s(&t.name)),
        ("shape", shape_v(&t.shape)),
        ("indices", Value::Array(idx.iter().map(|i| n(*i as u64)).collect())),
        ("values", Value::Array(vals.iter().map(|x| f(f64::from(*x))).collect())),
    ]);
    let safe: String = format!("{}-{}", h.key, t.name).chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '_' }).collect();
    let file = home.join("exports").join("parameters").join(format!("{safe}-{}.json", nmtype::now_ms()));
    write_once(&file, nmjson::pretty::pretty(&doc).as_bytes())?;
    Ok(obj(vec![("file", s(&file.display().to_string())), ("tensor", s(&t.name)), ("values", n(vals.len() as u64)), ("format", s(VALUES_FORMAT))]))
}

/// The paste a values file gives (`nm-parameter-values/1`, or `{values: [...]}`): on the request's
/// selection, else on the file's own path.
fn import_op(v: &Value) -> Result<crate::pedit::Op, String> {
    let file = v.get("file").and_then(Value::string).map_err(|_| "import: give file (a values file on this machine)".to_owned())?;
    let src = crate::pmaps::local_file(file, 64 << 20)?;
    let doc = std::fs::read(&src).map_err(|e| format!("{}: {e}", src.display())).and_then(|b| nmjson::parse(&b).map_err(|e| format!("{file}: {e}")))?;
    let vals = values_of(doc.get("values").map_err(|_| format!("{file}: no values"))?, file)?;
    let mut sel = crate::pedit::parse_sel(v)?;
    if sel == crate::pedit::Sel::default()
        && let Ok(p) = doc.get("path").and_then(Value::string)
    {
        sel = crate::pedit::parse_path(p)?;
    }
    Ok(crate::pedit::Op { sel, kind: crate::pedit::OpKind::Paste(vals) })
}

// ---------------------------------------------------------------------------------------------
// The command line.

const USAGE: &str = "usage: neuromorph-nano library parameters [<key>] [--tensor <name> [--offset <n>] [--limit <n>]]
       neuromorph-nano library parameters-edit <key> [--scale <s>] [--layer <n>=on|off|<s>]... [--values <json object>]";

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2).find(|w| w.first().is_some_and(|a| a == name)).and_then(|w| w.get(1)).map(String::as_str)
}

/// `library parameters` and `library parameters-edit`: the same reads and edit as the API.
/// # Errors
/// As `page` and `edit`, or arguments that do not read.
pub fn run(home: &Path, args: &[String]) -> Result<Value, String> {
    let sub = args.get(1).map_or("", String::as_str);
    let key = args.get(2).filter(|a| !a.starts_with("--")).cloned();
    match (sub, key) {
        ("parameters", None) => Ok(list(home)),
        ("parameters", Some(k)) => match flag(args, "--tensor") {
            Some(t) => page(home, &k, t, flag(args, "--offset").and_then(|x| x.parse().ok()).unwrap_or(0), flag(args, "--limit").and_then(|x| x.parse().ok()).unwrap_or(256)),
            None => Ok(section(home, &k)),
        },
        ("parameters-edit", Some(k)) => {
            let mut req = BTreeMap::new();
            if let Some(x) = flag(args, "--scale") {
                req.insert("scale".to_owned(), s(x));
            }
            let mut layers = BTreeMap::new();
            for w in args.windows(2).filter(|w| w.first().is_some_and(|a| a == "--layer")) {
                let (l, set) = w.get(1).and_then(|x| x.split_once('=')).ok_or("--layer <n>=on|off|<scale>")?;
                layers.insert(l.to_owned(), s(set));
            }
            if !layers.is_empty() {
                req.insert("layers".to_owned(), Value::Object(layers));
            }
            if let Some(x) = flag(args, "--values") {
                req.insert("values".to_owned(), nmjson::parse(x.as_bytes()).map_err(|e| format!("--values: a JSON object ({e})"))?);
            }
            edit(home, &k, &parse_edit(&Value::Object(req))?, "command")
        }
        _ => Err(USAGE.to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nmlearn::library::Stack;

    struct Home(PathBuf);
    impl Drop for Home {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn home(name: &str) -> Home {
        let d = std::env::temp_dir().join(format!("nano-cparams-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("memory")).unwrap();
        Home(d)
    }
    fn spec(id: &str, family: &str) -> CartridgeSpec {
        CartridgeSpec {
            id: id.into(),
            version: 1,
            family: family.into(),
            domain: "test".into(),
            scope: "general".into(),
            model: "qwen3-reranker-0.6b".into(),
            weights: String::new(),
            layers: Some((3, 5)),
            name: id.into(),
            description: "a test cartridge".into(),
            made_from: "the test".into(),
            effect: "not measured".into(),
        }
    }
    fn points() -> Vec<Point> {
        vec![Point { target: "L3.swigate".into(), gain: vec![1.5, 0.5, 1.0, 1.25], bias: vec![0.1, -0.2, 0.0, 0.3] }, Point { target: "L5.o".into(), gain: vec![2.0, 1.0], bias: vec![0.0, 0.5] }]
    }
    fn library_with(h: &Path, filter: &Filter, family: &str) -> String {
        let mut lib = Library::open(h).unwrap();
        let k = lib.add_cartridge(&spec("pt-socket", family), filter, Value::Null, Vec::new(), false).unwrap();
        lib.add_stack(Stack { id: "pt-stack".into(), version: 1, use_for: "job:pt".into(), piece: None, cartridges: vec![k.clone()], description: "test".into(), trial: false, hash: String::new() }).unwrap();
        lib.save().unwrap();
        k
    }
    fn file_bytes(h: &Path, key: &str) -> Vec<u8> {
        let lib = Library::open(h).unwrap();
        let c = lib.cartridge(key).unwrap();
        std::fs::read(lib.cartridge_dir(c).join(&c.files[0].name)).unwrap()
    }
    fn get<'a>(v: &'a Value, k: &str) -> &'a Value {
        v.get(k).unwrap()
    }
    fn fnum(v: &Value, k: &str) -> f64 {
        num(get(v, k)).unwrap()
    }
    fn req(json: &str) -> Edit {
        parse_edit(&nmjson::parse(json.as_bytes()).unwrap()).unwrap()
    }

    #[test]
    fn layers_are_read_from_node_ids() {
        assert_eq!(layer_of("L27.swigate"), Some(27));
        assert_eq!(layer_of("nmengn.components#L3.down"), Some(3));
        assert_eq!(layer_of("final_norm"), None);
        assert_eq!(layer_of("Lx.o"), None);
    }

    #[test]
    fn stats_give_norms_and_a_histogram() {
        let st = stats(&[3.0, -4.0], 0.0);
        assert!((st.l2 - 5.0).abs() < 1e-9 && (st.max_abs - 4.0).abs() < 1e-9 && (st.mean + 0.5).abs() < 1e-9);
        assert_eq!(st.hist.iter().sum::<u64>(), 2);
        assert_eq!(st.hist.first(), Some(&1));
        assert_eq!(st.hist.last(), Some(&1));
        let g = stats(&[1.0, 1.0], 1.0);
        assert!(g.effect_l2 == 0.0 && (g.l2 - 2.0_f64.sqrt()).abs() < 1e-9, "a gain of one changes nothing");
    }

    #[test]
    fn socket_cartridge_section_lists_gain_and_bias_per_point() {
        let h = home("section");
        let key = library_with(&h.0, &Filter::Points { points: points(), q8: None }, "learning");
        let v = section(&h.0, &key);
        assert_eq!(text(&v, "kind"), "filter");
        assert!(matches!(get(&v, "holds"), Value::Bool(true)));
        let ts = get(&v, "tensors").array().unwrap();
        assert_eq!(ts.len(), 4);
        let g = &ts[0];
        assert_eq!(text(g, "name"), "L3.swigate.gain");
        assert_eq!(get(g, "layer").integer().unwrap(), 3);
        assert_eq!(get(g, "size").integer().unwrap(), 4);
        assert!((fnum(g, "max_abs") - 1.5).abs() < 1e-6);
        assert_eq!(get(get(get(g, "summary"), "histogram"), "counts").array().unwrap().len(), BINS);
        let layers = get(&v, "layers").array().unwrap();
        assert_eq!(layers.iter().map(|l| get(l, "layer").integer().unwrap()).collect::<Vec<_>>(), vec![3, 5]);
        // Paged reads.
        let p = page(&h.0, &key, "L3.swigate.bias", 1, 2).unwrap();
        let vals: Vec<f64> = get(&p, "values").array().unwrap().iter().filter_map(num).collect();
        assert_eq!(vals.len(), 2);
        assert!((vals[0] + 0.2).abs() < 1e-6 && (vals[1]).abs() < 1e-9);
        assert!(page(&h.0, &key, "nope", 0, 2).is_err());
    }

    #[test]
    fn settings_say_they_hold_no_parameters_and_preferences_show_counts() {
        let h = home("settings");
        let mut lib = Library::open(&h.0).unwrap();
        let mut m = BTreeMap::new();
        m.insert("fast_k".to_owned(), "5".to_owned());
        let sk = lib.add_cartridge(&CartridgeSpec { model: "settings".into(), layers: None, ..spec("pt-settings", "fundamentals") }, &Filter::Settings(m), Value::Null, Vec::new(), false).unwrap();
        let mut p = BTreeMap::new();
        p.insert("pref.global".to_owned(), "3/4".to_owned());
        let pk = lib.add_cartridge(&CartridgeSpec { model: "settings".into(), layers: None, ..spec("pt-pref", nmlearn::library::PREFERENCE) }, &Filter::Settings(p), Value::Null, Vec::new(), false).unwrap();
        lib.save().unwrap();
        let v = section(&h.0, &sk);
        assert!(matches!(get(&v, "holds"), Value::Bool(false)));
        assert_eq!(text(&v, "note"), "this cartridge holds no parameters");
        assert!(v.get("error").is_err(), "not an error");
        let pv = section(&h.0, &pk);
        assert_eq!(text(&pv, "kind"), "preference");
        let t = &get(&pv, "tensors").array().unwrap()[0];
        assert_eq!(text(t, "name"), "pref.global");
        assert!((fnum(t, "max_abs") - 4.0).abs() < 1e-6);
        assert!(edit(&h.0, &pk, &req(r#"{"scale":0.5}"#), "test").is_err(), "preference counts are set as values");
        assert!(edit(&h.0, &sk, &req(r#"{"scale":0.5}"#), "test").is_err());
    }

    #[test]
    fn strength_makes_a_new_version_on_trial_and_leaves_the_source() {
        let h = home("strength");
        let key = library_with(&h.0, &Filter::Points { points: points(), q8: None }, "learning");
        let before = file_bytes(&h.0, &key);
        let r = edit(&h.0, &key, &req(r#"{"scale":0.5}"#), "test").unwrap();
        let nk = text(&r, "key");
        assert_eq!(nk, "pt-socket@2");
        assert!(!text(&r, "audit_id").is_empty());
        assert_eq!(file_bytes(&h.0, &key), before, "the source is never written");
        let lib = Library::open(&h.0).unwrap();
        let nc = lib.cartridge(&nk).unwrap();
        assert!(nc.trial, "the new version is on trial");
        let Ok(Filter::Points { points: p, .. }) = lib.filter(nc) else { panic!("points") };
        assert!((p[0].gain[0] - 1.25).abs() < 1e-6 && (p[0].bias[1] + 0.1).abs() < 1e-6, "gain toward 1, bias toward 0");
        assert!(lib.stack("pt-stack").unwrap().cartridges.contains(&key), "stacks are untouched");
        // Undo sets the new version aside; nothing is deleted.
        undo(&h.0, &audit_row(&h.0, &text(&r, "audit_id")), &text(&r, "audit_id"), "test").unwrap();
        let lib = Library::open(&h.0).unwrap();
        assert!(lib.cartridge(&nk).is_none());
        assert!(lib.cartridge(&key).is_some());
        assert!(std::fs::read_dir(h.0.join(SET_ASIDE)).unwrap().count() == 1);
    }

    #[test]
    fn layers_off_drop_points_and_values_are_shape_checked() {
        let h = home("layers");
        let key = library_with(&h.0, &Filter::Points { points: points(), q8: None }, "learning");
        let r = edit(&h.0, &key, &req(r#"{"layers":{"5":"off","3":0.0}}"#), "test").unwrap();
        let lib = Library::open(&h.0).unwrap();
        let Ok(Filter::Points { points: p, .. }) = lib.filter(lib.cartridge(&text(&r, "key")).unwrap()) else { panic!("points") };
        assert_eq!(p.len(), 1, "layer 5 dropped");
        assert!(p[0].gain.iter().all(|g| (*g - 1.0).abs() < 1e-6) && p[0].bias.iter().all(|b| *b == 0.0), "layer 3 at scale 0 is the identity");
        // The shape check refuses a bad edit, and nothing is written.
        let n_before = Library::open(&h.0).unwrap().cartridges.len();
        let bad = edit(&h.0, &key, &req(r#"{"values":{"L3.swigate.gain":[1,2,3]}}"#), "test").unwrap_err();
        assert!(bad.contains("shape check"), "{bad}");
        let past = edit(&h.0, &key, &req(r#"{"values":{"L5.o.bias":{"offset":1,"values":[1,2]}}}"#), "test").unwrap_err();
        assert!(past.contains("shape check"), "{past}");
        assert!(edit(&h.0, &key, &req(r#"{"layers":{"9":"off"}}"#), "test").is_err());
        assert!(parse_edit(&nmjson::parse(br#"{"scale":9}"#).unwrap()).is_err());
        assert_eq!(Library::open(&h.0).unwrap().cartridges.len(), n_before);
        // A direct value edit.
        let ok = edit(&h.0, &key, &req(r#"{"values":{"L5.o.bias":{"offset":1,"values":[0.25]}}}"#), "test").unwrap();
        let lib = Library::open(&h.0).unwrap();
        let Ok(Filter::Points { points: p, .. }) = lib.filter(lib.cartridge(&text(&ok, "key")).unwrap()) else { panic!("points") };
        assert!((p[1].bias[1] - 0.25).abs() < 1e-6);
        // Every layer off is refused.
        assert!(edit(&h.0, &key, &req(r#"{"layers":{"3":"off","5":"off"}}"#), "test").is_err());
    }

    #[test]
    fn q8_cartridges_stay_at_8_bits() {
        let h = home("q8");
        let key = library_with(&h.0, &Filter::points_q8(points()), nmlearn::library::ADAPTER);
        let v = section(&h.0, &key);
        assert_eq!(text(&v, "kind"), "adapter");
        let r = edit(&h.0, &key, &req(r#"{"scale":2}"#), "test").unwrap();
        let lib = Library::open(&h.0).unwrap();
        assert!(matches!(lib.filter(lib.cartridge(&text(&r, "key")).unwrap()), Ok(Filter::Points { q8: Some(_), .. })));
    }

    fn audit_row(h: &Path, id: &str) -> Value {
        let all = std::fs::read_to_string(h.join("memory/audit.jsonl")).unwrap();
        all.lines().filter_map(|l| nmjson::parse(l.as_bytes()).ok()).find(|v| text(v, "id") == id).unwrap()
    }

    #[test]
    fn job_filters_edit_into_a_new_file_plugged_out() {
        let h = home("job");
        let dir = h.0.join("library/parameter-maps/reranker");
        std::fs::create_dir_all(&dir).unwrap();
        let src = r#"{"contract":"nmengn-sockets/1","cartridge":"reranker-code@1","model":"qwen3-reranker-0.6b","job":"code","enabled":true,"parts":[{"target":"L2.o","gain":[1.5,0.5],"bias":[0.1,0.2]},{"target":"L4.o","gain":[2.0],"bias":[1.0]}]}"#;
        std::fs::write(dir.join("job-filter-code.json"), src).unwrap();
        let v = section(&h.0, "filter:reranker/code");
        assert_eq!(text(&v, "kind"), "filter");
        assert_eq!(get(&v, "tensors").array().unwrap().len(), 4);
        assert!(section(&h.0, "filter:reranker-code@1").get("error").is_err(), "the compat key reads too");
        let r = edit(&h.0, "filter:reranker/code", &req(r#"{"layers":{"4":"off"},"scale":0}"#), "test").unwrap();
        assert_eq!(text(&r, "key"), "filter:reranker/code-p2");
        assert_eq!(std::fs::read_to_string(dir.join("job-filter-code.json")).unwrap(), src, "the source is never written");
        let new = nmjson::parse(&std::fs::read(dir.join("job-filter-code-p2.json")).unwrap()).unwrap();
        assert!(matches!(new.get("enabled"), Ok(Value::Bool(false))));
        let pts = parse_points(&new).unwrap();
        assert_eq!(pts.len(), 1);
        assert!(pts[0].gain.iter().all(|g| (*g - 1.0).abs() < 1e-6));
        // Bad edits write nothing.
        assert!(edit(&h.0, "filter:reranker/code", &req(r#"{"values":{"L2.o.gain":[1]}}"#), "test").is_err());
        assert!(!dir.join("job-filter-code-p3.json").exists());
        let id = text(&r, "audit_id");
        crate::controls::undo_by(&h.0, &id, "test").unwrap();
        assert!(!dir.join("job-filter-code-p2.json").exists());
        assert!(dir.join("archive/job-filter-code-p2.json").is_file(), "set aside in the archive, not deleted");
    }

    #[test]
    fn adapter_library_versions_are_read_and_edited_into_the_next_version() {
        let h = home("adapter");
        let id = "judge-code-000000000000";
        let l = nmapi::adapter_fit::LabelIn { id, version: 1, model: nmapi::adapter::JUDGE, weights: "", kind: nmapi::adapter::READOUT_KIND, does: "test read-out", made_from: "test", notes: obj(vec![("values", s("vk")), ("alpha", f(1.0)), ("bias", f(0.0))]) };
        let file = nmapi::adapter::shelf_file(&h.0, id, 1);
        nmapi::adapter_fit::write_parts(&file, nmapi::adapter_fit::label(&l), &[("w".to_owned(), vec![4], vec![0.5, -0.5, 0.25, 1.0])]).unwrap();
        let before = std::fs::read(&file).unwrap();
        let v = section(&h.0, &format!("{id}@1"));
        assert_eq!(text(&v, "kind"), "readout");
        assert!(matches!(get(get(&v, "editable"), "layers"), Value::Bool(false)));
        assert!(edit(&h.0, id, &req(r#"{"layers":{"1":"off"}}"#), "test").unwrap_err().contains("no layers"));
        let r = edit(&h.0, id, &req(r#"{"scale":0.5}"#), "test").unwrap();
        assert_eq!(text(&r, "key"), format!("{id}@2"));
        assert_eq!(std::fs::read(&file).unwrap(), before);
        let (_, parts) = nmapi::adapter::read_file(&nmapi::adapter::shelf_file(&h.0, id, 2)).unwrap();
        assert!((parts[0].2[0] - 0.25).abs() < 1e-3);
        assert_eq!(text(get(&r, "compat"), "shapes"), "fit");
        crate::controls::undo_by(&h.0, &text(&r, "audit_id"), "test").unwrap();
        assert!(!nmapi::adapter::shelf_file(&h.0, id, 2).exists());
        assert!(h.0.join(SET_ASIDE).join("adapter").read_dir().unwrap().count() == 1);
    }

    #[test]
    fn learning_units_show_their_factors() {
        let h = home("unit");
        let mut units = nmlearn::Units::open(&h.0);
        let card = nmlearn::UnitCard::new("lwu-pt", "qwen3-reranker-0.6b", "aa", "reranker", "nmengn.components#L7.down", 2, 3);
        units.register(card.clone()).unwrap();
        let v = section(&h.0, "lwu-pt");
        assert!(matches!(get(&v, "holds"), Value::Bool(false)), "unfitted: no files yet");
        let w = nmlearn::UnitWeights { rank: 2, k: 3, n: 2, down: vec![0.1, 0.2, 0.3, -0.1, -0.2, -0.3], up: vec![0.5, 0.0, 0.0, 0.5], scale: 1.0 };
        let mut files = Vec::new();
        for (name, b) in w.files().unwrap() {
            files.push((name.clone(), units.commit_file("lwu-pt", &name, &b).unwrap()));
        }
        units.set_files("lwu-pt", files, "fitted").unwrap();
        let v = section(&h.0, "lwu-pt");
        assert_eq!(text(&v, "kind"), "unit");
        let ts = get(&v, "tensors").array().unwrap();
        assert_eq!(ts.iter().map(|t| text(t, "name")).collect::<Vec<_>>(), vec!["down", "up"]);
        assert_eq!(get(&ts[1], "layer").integer().unwrap(), 7);
        // The registry holds one open unit per node, so an edit of an open unit is refused by it.
        assert!(edit(&h.0, "lwu-pt", &req(r#"{"scale":0.5}"#), "test").unwrap_err().contains("overlaps"));
        units = nmlearn::Units::open(&h.0);
        units.set_status("lwu-pt", "retired").unwrap();
        let r = edit(&h.0, "lwu-pt", &req(r#"{"layers":{"7":0.5}}"#), "test").unwrap();
        assert_eq!(text(&r, "key"), "lwu-pt-p2");
        let nw = nmlearn::UnitWeights::read(nmlearn::Units::open(&h.0).get("lwu-pt-p2").unwrap(), &nmlearn::Units::open(&h.0).unit_dir("lwu-pt-p2")).unwrap();
        assert!((nw.up[0] - 0.25).abs() < 0.01);
        crate::controls::undo_by(&h.0, &text(&r, "audit_id"), "test").unwrap();
        assert_eq!(nmlearn::Units::open(&h.0).get("lwu-pt-p2").unwrap().status, "retired");
    }

    #[test]
    fn the_list_covers_every_object_that_holds_parameters() {
        let h = home("list");
        library_with(&h.0, &Filter::Points { points: points(), q8: None }, "learning");
        let v = list(&h.0);
        let objs = get(&v, "objects").array().unwrap();
        let row = objs.iter().find(|o| text(o, "key") == "pt-socket@1").unwrap();
        assert_eq!(text(row, "role"), "reranker");
        assert_eq!(get(row, "layer_range").array().unwrap().iter().map(|x| x.integer().unwrap()).collect::<Vec<_>>(), vec![3, 5]);
        // Labels only: no tensor, value count or per-layer norm is read for the list.
        assert!(row.get("tensors").is_err() && row.get("values").is_err() && row.get("layers").is_err());
        assert!(matches!(get(row, "read"), Value::Bool(false)));
        assert!(fnum(row, "bytes") > 0.0);
        let dir = h.0.join("library/parameter-maps/reranker");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("job-filter-lj.json"), r#"{"cartridge":"reranker-lj@1","contract":"nmengn-sockets/1","enabled":false,"job":"lj","model":"qwen3-reranker-0.6b","parts":[{"target":"L2.o","gain":[1.5],"bias":[0.1]}]}"#).unwrap();
        let v = list(&h.0);
        let j = get(&v, "objects").array().unwrap().iter().find(|o| text(o, "key") == "filter:reranker/lj").unwrap().clone();
        assert_eq!((text(&j, "state"), text(&j, "model"), text(&j, "name")), ("plugged out".to_owned(), "qwen3-reranker-0.6b".to_owned(), "reranker-lj@1".to_owned()));
        assert!(held(&h.0, "nothing-here").is_err());
    }

    fn ops(h: &Path, key: &str, json: &str) -> Result<Value, String> {
        edit(h, key, &req(json), "test")
    }

    #[test]
    fn a_preview_gives_the_difference_and_writes_nothing() {
        let h = home("preview");
        let key = library_with(&h.0, &Filter::Points { points: points(), q8: None }, "learning");
        let n_before = Library::open(&h.0).unwrap().cartridges.len();
        let r = ops(&h.0, &key, r#"{"op":"set","value":2,"path":"L3.swigate.gain[0:2]","preview":true}"#).unwrap();
        assert!(matches!(get(&r, "preview"), Value::Bool(true)));
        let d = get(&r, "diff");
        assert_eq!(get(d, "changed").integer().unwrap(), 2);
        let row = &get(d, "tensors").array().unwrap()[0];
        assert_eq!(text(row, "name"), "L3.swigate.gain");
        assert!(fnum(row, "new_l2") > fnum(row, "old_l2"));
        assert_eq!(Library::open(&h.0).unwrap().cartridges.len(), n_before, "a preview writes nothing");
        assert!(!h.0.join("memory/audit.jsonl").exists());
    }

    #[test]
    fn operations_on_a_selection_make_a_new_version_with_its_label() {
        let h = home("ops");
        let key = library_with(&h.0, &Filter::Points { points: points(), q8: None }, "learning");
        let r = ops(&h.0, &key, r#"{"ops":[{"op":"clamp","lo":0.75,"hi":1.25,"path":"L3.swigate.gain"},{"op":"negate","selection":{"tensor":"L5.o.bias","index":1}}],"name":"pt clamped","description":"gains clamped by hand"}"#).unwrap();
        let lib = Library::open(&h.0).unwrap();
        let nc = lib.cartridge(&text(&r, "key")).unwrap();
        assert_eq!((nc.name.as_str(), nc.description.as_str()), ("pt clamped", "gains clamped by hand"));
        let Ok(Filter::Points { points: p, .. }) = lib.filter(nc) else { panic!("points") };
        assert_eq!(p[0].gain, vec![1.25, 0.75, 1.0, 1.25]);
        assert!((p[1].bias[1] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn points_are_added_only_at_sockets_the_micromodel_has() {
        let h = home("sockets");
        let asm = h.0.join("designs/assemblies");
        std::fs::create_dir_all(&asm).unwrap();
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../designs/assemblies");
        for f in ["qwen3-reranker.json", "qwen3-reference.json"] {
            std::fs::copy(src.join(f), asm.join(f)).unwrap();
        }
        let key = library_with(&h.0, &Filter::Points { points: points(), q8: None }, "learning");
        let bad = ops(&h.0, &key, r#"{"op":"add_point","target":"L4.nothing","width":2}"#).unwrap_err();
        assert!(bad.contains("no socket L4.nothing"), "{bad}");
        let ok = ops(&h.0, &key, r#"{"op":"add_point","target":"L4.o","width":2}"#).unwrap();
        assert!(text(get(&ok, "compat"), "sockets").contains("L4.o"));
        assert_eq!(get(get(&ok, "parameters"), "tensors").array().unwrap().len(), 6);
    }

    #[test]
    fn preference_counts_are_edited_through_their_values() {
        let h = home("prefedit");
        let mut lib = Library::open(&h.0).unwrap();
        let mut p = BTreeMap::new();
        p.insert("pref.global".to_owned(), "3/4".to_owned());
        let pk = lib.add_cartridge(&CartridgeSpec { model: "settings".into(), layers: None, ..spec("pt-pref", nmlearn::library::PREFERENCE) }, &Filter::Settings(p), Value::Null, Vec::new(), false).unwrap();
        lib.save().unwrap();
        assert!(ops(&h.0, &pk, r#"{"op":"set","value":0.5,"path":"pref.global[0]"}"#).unwrap_err().contains("whole numbers"));
        assert!(ops(&h.0, &pk, r#"{"op":"set","value":9,"path":"pref.global[0]"}"#).is_err(), "good at most signals");
        let r = ops(&h.0, &pk, r#"{"op":"add","value":1,"path":"pref.global"}"#).unwrap();
        let lib = Library::open(&h.0).unwrap();
        let Ok(Filter::Settings(m)) = lib.filter(lib.cartridge(&text(&r, "key")).unwrap()) else { panic!("settings") };
        assert_eq!(m.get("pref.global").map(String::as_str), Some("4/5"));
    }

    fn job(h: &Path, name: &str, parts: &str) {
        let dir = h.join("library/parameter-maps/reranker");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(format!("job-filter-{name}.json")), format!(r#"{{"contract":"nmengn-sockets/1","cartridge":"reranker-{name}@1","model":"qwen3-reranker-0.6b","job":"{name}","enabled":false,"base":"reranker-base@1","parts":{parts}}}"#)).unwrap();
    }

    #[test]
    fn regions_maps_copies_exports_and_imports_meet_at_the_same_place() {
        let h = home("maps");
        job(&h.0, "code", r#"[{"target":"L2.o","gain":[1.5,0.5,1.25],"bias":[0.1,0.2,0.3]}]"#);
        job(&h.0, "docs", r#"[{"target":"L2.o","gain":[2.0,2.0,2.0],"bias":[0.0,0.0,0.0]}]"#);
        let dir = h.0.join("library/parameter-maps/reranker");
        std::fs::write(dir.join("tol-pt.json"), r#"{"tolerances":{"L2":0.05,"L9":0.1}}"#).unwrap();
        std::fs::write(dir.join("lineage.json"), r#"{"maps":{"pmap-far.json":{"made_on":"reranker-base@1","used_by":[]}}}"#).unwrap();
        std::fs::write(dir.join("pmap-far.json"), r#"{"note":"no layers"}"#).unwrap();
        let k = "filter:reranker/code";
        let m = maps(&h.0, k);
        let files: Vec<String> = get(&m, "maps").array().unwrap().iter().map(|x| text(x, "file")).collect();
        assert_eq!(files, vec!["job-filter-docs.json", "pmap-far.json", "tol-pt.json"], "same points, the same base, the same layers");
        let r = region(&h.0, k, "L2.o.gain[1:3]", 0).unwrap();
        let vals: Vec<f64> = get(&r, "values").array().unwrap().iter().filter_map(num).collect();
        assert_eq!(vals, vec![0.5, 1.25]);
        let mv = get(&r, "maps").array().unwrap();
        assert!(mv.iter().any(|x| text(x, "map") == "job-filter-docs.json" && get(x, "values").array().unwrap().len() == 2));
        assert!(mv.iter().any(|x| text(x, "map") == "tol-pt.json" && (fnum(x, "value") - 0.05).abs() < 1e-6));
        assert!(get(&r, "distribution").get("counts").is_ok());
        // Copy and interpolate from a map at the same place.
        let c = ops(&h.0, k, r#"{"op":"interpolate","t":0.5,"source":{"map":"reranker/job-filter-docs.json"},"path":"L2.o.gain[0:1]"}"#).unwrap();
        let p = region(&h.0, &text(&c, "key"), "L2.o.gain[0]", 0).unwrap();
        assert!((get(&p, "values").array().unwrap().iter().filter_map(num).next().unwrap() - 1.75).abs() < 1e-6);
        assert!(ops(&h.0, k, r#"{"op":"copy","source":{"map":"reranker/tol-pt.json"},"path":"L2.o.gain"}"#).is_err(), "a per-layer map holds no values at a point");
        // Export a selection and import it into another place of the same width.
        let x = export(&h.0, k, "L2.o.bias").unwrap();
        let file = text(&x, "file");
        let imp = edit(&h.0, k, &parse_edit(&obj(vec![("op", s("import")), ("file", s(&file)), ("path", s("L2.o.gain"))])).unwrap(), "test").unwrap();
        let g = region(&h.0, &text(&imp, "key"), "L2.o.gain", 0).unwrap();
        let gv: Vec<f64> = get(&g, "values").array().unwrap().iter().filter_map(num).collect();
        assert!((gv[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn a_unit_rank_is_truncated_into_a_new_candidate() {
        let h = home("rank");
        let mut units = nmlearn::Units::open(&h.0);
        units.register(nmlearn::UnitCard::new("lwu-r", "qwen3-reranker-0.6b", "aa", "reranker", "nmengn.components#L7.down", 2, 3)).unwrap();
        let w = nmlearn::UnitWeights { rank: 2, k: 3, n: 2, down: vec![0.1, 0.2, 0.3, -0.1, -0.2, -0.3], up: vec![0.5, 0.25, 0.125, 0.5], scale: 1.0 };
        let mut files = Vec::new();
        for (name, b) in w.files().unwrap() {
            files.push((name.clone(), units.commit_file("lwu-r", &name, &b).unwrap()));
        }
        units.set_files("lwu-r", files, "retired").unwrap();
        let r = ops(&h.0, "lwu-r", r#"{"op":"rank","rank":1}"#).unwrap();
        let card = nmlearn::Units::open(&h.0).get(&text(&r, "key")).cloned().unwrap();
        assert_eq!((card.rank, card.status.as_str()), (1, "candidate"));
        assert!(ops(&h.0, "lwu-r", r#"{"op":"drop","path":"down"}"#).is_err(), "a unit's tensors are a fixed set");
    }

    #[test]
    fn stored_types_round_trip_through_encode_and_decode() {
        let v = [1.5_f32, -0.25, 0.0, 3.0];
        assert_eq!(decode("BF16", &encode("BF16", &v).unwrap()), v.to_vec());
        assert_eq!(decode("F32", &encode("F32", &v).unwrap()), v.to_vec());
        assert_eq!(decode("F16", &encode("F16", &v).unwrap()), v.to_vec());
        assert_eq!(to_f16(65504.0), 0x7bff);
        assert_eq!(to_f16(1e9), 0x7c00, "overflow is infinity");
        assert_eq!(to_f16(1e-9), 0, "underflow is zero");
        assert!((nmengn::half::from16(to_f16(0.1)) - 0.1).abs() < 1e-4);
        assert!(encode("Q4", &v).is_err());
    }

    #[test]
    fn a_base_edit_is_a_new_version_on_trial_and_the_source_is_unchanged() {
        let h = home("base-edit");
        let name = "model.layers.0.mlp.down_proj.weight";
        let base = Filter::Tensors(vec![nmlearn::library::Tensor { name: name.into(), layer: Some(0), block: "mlp".into(), dtype: "BF16".into(), shape: vec![2, 2], bytes: encode("BF16", &[1.0, 2.0, 3.0, 4.0]).unwrap() }]);
        let mut lib = Library::open(&h.0).unwrap();
        let bspec = CartridgeSpec { id: "reader-base".into(), family: String::new(), model: "qwen3-0.6b".into(), layers: Some((0, 0)), ..spec("reader-base", "") };
        let (k1, _) = lib.add_base_cartridge(&bspec, &base, Value::Null, Vec::new(), false).unwrap();
        lib.save().unwrap();
        drop(lib);
        let before = file_bytes(&h.0, &k1);
        let held1 = held(&h.0, &k1).unwrap();
        assert_eq!(held1.kind, "base");
        assert!(held1.editable.is_ok(), "a base cartridge's parameters are editable into a new version");
        let preview = ops(&h.0, &k1, &format!(r#"{{"op":"set","value":9,"selection":{{"tensor":"{name}","index":1}},"preview":true}}"#)).unwrap();
        assert!(text(&preview, "writes").contains("new base cartridge version"));
        assert_eq!(Library::open(&h.0).unwrap().cartridges.iter().filter(|c| c.id == "reader-base").count(), 1, "a preview writes nothing");
        let r = ops(&h.0, &k1, &format!(r#"{{"op":"set","value":9,"selection":{{"tensor":"{name}","index":1}}}}"#)).unwrap();
        let k2 = text(&r, "key");
        assert_eq!(k2, "reader-base@2");
        let lib = Library::open(&h.0).unwrap();
        assert!(lib.cartridge(&k2).is_some_and(|c| c.trial), "the new version arrives on trial");
        assert!(lib.cartridge(&k1).is_some_and(|c| !c.trial), "the source version stays as it was");
        drop(lib);
        assert_eq!(file_bytes(&h.0, &k1), before, "the source file is never written");
        let held2 = held(&h.0, &k2).unwrap();
        assert_eq!(held2.tensors[0].values.as_deref(), Some([1.0, 9.0, 3.0, 4.0].as_slice()));
    }
}
