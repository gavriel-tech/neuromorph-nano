//! The cartridge library. A cartridge is a self-contained unit of learned parameters on one
//! knowledge domain (machine learning architecture, Rust fundamentals, memory and hardware use,
//! the Nano's own system, preferences, the structure of a software library), much as a game
//! cartridge is to a console, except that cartridges stack and hold parameters for MicroModels.
//! The engine is the console: it has sockets, named places in a MicroModel's layers, and does not
//! change when a cartridge goes in or out. A plugged cartridge filters the values passing its
//! sockets; the bare weights (the released weights) are only read, so pulling a cartridge returns
//! the engine exactly to how it was. A settings cartridge filters the values the Nano's settings
//! take. Cartridges are stacked for the work at hand, in a defined order. A cartridge is never the
//! code being worked on: that changes and is read live.
//!
//! One cartridge is one file, `cartridges/<id>-v<version>.cartridge`: an 8-byte mark, the length
//! of its label, the label (JSON), then its parameters. The label says everything needed to seat
//! it: the MicroModel and the SHA-256 of the bare weights it was made for, the socket contract,
//! its family, domain, scope and version, and each part (the socket, the form, the shape, the
//! encoding and where its bytes are), the SHA-256 of the parameters, and any notes (what it was
//! fitted from). Forms: `tensor` (a parameter set of a MicroModel's blocks, addressed by the tensor
//! name the open weights use for its place, with its layer and block, element type, shape and
//! content hash: a base cartridge is all the sets a job's blocks read, and holds no structure),
//! `gain-bias` (an additional cartridge at a socket: per-channel `x * gain + bias`, at 8 bits or
//! float32) and `settings` (new values for named settings).
//!
//! The library's own record, `catalogue.json`, is a shelf: which cartridges exist (file and file
//! hash, and whether each is on trial), the stacks (a use, the hardware piece it is for, its
//! cartridges in order) and which stack is switched on for each use. Every load checks the file
//! against the shelf's hash and the parameters against the label's. A cartridge written before
//! this form (a folder of files listed in full on the shelf) is still read.
//!
//! What a cartridge is for is open: the label's descriptive fields (`family`, `domain`, `scope`,
//! `description`, `notes`) change nothing in how it loads, with one rule: a `preference` cartridge
//! (the user's ways) is kept in `preferences/` and plugged only by stacks for a preference use,
//! which apply after the others. A stack applies its cartridges in its own order. A cartridge or
//! stack on `trial` is loaded and recorded and never changes what is served. `evidence` rows (on the
//! label) are tagged with the exact hardware piece they were measured on. Stacks are exported and imported as one file; an import is checked and
//! arrives on trial. The library sends nothing anywhere.
use crate::hardware::{Kind, Machine, Piece};
use nmjson::Value;
use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// Schema of `catalogue.json`.
pub const SCHEMA: &str = "nm-library/1";
/// Schema of a cartridge file's label.
pub const CARTRIDGE_SCHEMA: &str = "nm-cartridge/1";
/// The first bytes of a cartridge file.
pub const CARTRIDGE_MARK: &[u8; 8] = b"NMCART1\n";
/// The ending of a cartridge file's name.
pub const CARTRIDGE_EXT: &str = "cartridge";
/// The socket contract a `gain-bias` part is written for (the engine's `nmengn::filter::SOCKETS`).
pub const SOCKETS: &str = "nmengn-sockets/1";
/// Schema of an exported stack file.
pub const EXPORT_SCHEMA: &str = "nm-stack-export/1";
/// The catalogue file: the cartridge library's own index, kept inside the cartridge library.
pub const CATALOGUE: &str = "catalogue.json";
/// The release catalogue the installer leaves for `library merge`, inside the cartridge library.
pub const RELEASE_CATALOGUE: &str = "release-catalogue.json";
/// The list of base cartridge files the installer fetches, inside the cartridge library.
pub const RELEASE_FILES: &str = "release-files.json";

/// The catalogue of the cartridge library under `lib_dir` (`<home>/library`).
#[must_use]
pub fn catalogue_file(lib_dir: &Path) -> PathBuf {
    lib_dir.join("cartridges").join(CATALOGUE)
}

/// This machine's stack state, kept beside the Nano's memory and outside the libraries (the
/// `library/` folder holds only the libraries): what was seated at the last start, and the pieces
/// stacks are keyed by.
#[must_use]
pub fn stack_state_dir(home: &Path) -> PathBuf {
    home.join("memory").join("stacks")
}

/// What was seated at the last start (`memory/stacks/in-use.json`).
#[must_use]
pub fn in_use_file(home: &Path) -> PathBuf {
    stack_state_dir(home).join("in-use.json")
}

/// This machine's pieces as the stacks are keyed by (`memory/stacks/machine.json`).
#[must_use]
pub fn machine_file(home: &Path) -> PathBuf {
    stack_state_dir(home).join(crate::hardware::MACHINE_FILE)
}

/// Moves the records an earlier version kept loose at the root of `library/` to their places, so
/// `library/` holds only the libraries. A record already in its new place is kept and the loose copy
/// is left where it is; nothing is deleted. Cheap enough to run before every read.
pub fn tidy(home: &Path) {
    let lib = home.join("library");
    let cart = lib.join("cartridges");
    let moves = [
        (lib.join(CATALOGUE), catalogue_file(&lib)),
        (lib.join(RELEASE_CATALOGUE), cart.join(RELEASE_CATALOGUE)),
        (lib.join(RELEASE_FILES), cart.join(RELEASE_FILES)),
        (lib.join("in-use.json"), in_use_file(home)),
        (lib.join(crate::hardware::MACHINE_FILE), machine_file(home)),
    ];
    for (from, to) in moves {
        if from.is_file() && !to.exists() {
            if let Some(d) = to.parent() {
                let _ = std::fs::create_dir_all(d);
            }
            let _ = std::fs::rename(&from, &to);
        }
    }
    for (from, to) in [(lib.join("set-aside"), cart.join("set-aside")), (lib.join("incoming"), home.join("incoming"))] {
        if from.is_dir() && !to.exists() {
            if let Some(d) = to.parent() {
                let _ = std::fs::create_dir_all(d);
            }
            let _ = std::fs::rename(&from, &to);
        }
    }
}
/// Verified rows from one piece needed before a stack for it comes off trial.
pub const READY_ROWS: usize = 5;

/// A named reason the library refused something.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LibFault {
    /// A file's bytes do not match the hash recorded for it.
    FileHash { cartridge: String, file: String },
    /// A file recorded for a cartridge is missing.
    FileMissing { cartridge: String, file: String },
    /// A cartridge's content hash does not match its files.
    ContentHash { cartridge: String },
    /// A stack's hash does not match its members.
    StackHash { stack: String },
    /// An exported file's whole-file hash does not match its body.
    ExportHash,
    /// Not the expected schema, or not JSON.
    Schema(String),
    /// A filter whose length does not match the vector it is applied to.
    Shape(String),
    /// The same id and version exists with other content, or already.
    Conflict(String),
    /// No such cartridge or stack.
    Unknown(String),
    /// Reading or writing the folder failed.
    Io(String),
}
impl LibFault {
    /// The short name of the reason.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::FileHash { .. } => "file-hash",
            Self::FileMissing { .. } => "file-missing",
            Self::ContentHash { .. } => "content-hash",
            Self::StackHash { .. } => "stack-hash",
            Self::ExportHash => "export-hash",
            Self::Schema(_) => "schema",
            Self::Shape(_) => "shape",
            Self::Conflict(_) => "conflict",
            Self::Unknown(_) => "unknown",
            Self::Io(_) => "io",
        }
    }
}
impl std::fmt::Display for LibFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let what = match self {
            Self::FileHash { cartridge, file } => format!("{cartridge}: {file} does not match its recorded SHA-256"),
            Self::FileMissing { cartridge, file } => format!("{cartridge}: {file} is missing"),
            Self::ContentHash { cartridge } => format!("{cartridge}: content hash does not match its files"),
            Self::StackHash { stack } => format!("{stack}: stack hash does not match its cartridges"),
            Self::ExportHash => "the file's hash does not match its contents".to_owned(),
            Self::Schema(m) | Self::Shape(m) | Self::Conflict(m) | Self::Unknown(m) | Self::Io(m) => m.clone(),
        };
        write!(f, "library refused ({}): {what}", self.code())
    }
}

fn io(e: impl std::fmt::Display) -> LibFault {
    LibFault::Io(e.to_string())
}
fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}
fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn text(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default()
}
fn need(v: &Value, k: &str) -> Result<String, LibFault> {
    v.get(k).and_then(Value::string).map(str::to_owned).map_err(|e| LibFault::Schema(format!("{k}: {e}")))
}
fn flag(v: &Value, k: &str) -> bool {
    matches!(v.get(k), Ok(Value::Bool(true)))
}

// ---------------------------------------------------------------------------------------------
// Filters.

/// One point of an affine filter: the socket (a node of the MicroModel by short id) and the gain
/// and bias over it.
#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    pub target: String,
    pub gain: Vec<f32>,
    pub bias: Vec<f32>,
}

/// The learned parameters of a cartridge, used as a filter.
#[derive(Clone, Debug, PartialEq)]
pub enum Filter {
    /// New values for named settings.
    Settings(BTreeMap<String, String>),
    /// `x * gain + bias` at each point, in order. With `q8` (one gain scale and one bias scale per
    /// point) the numbers are stored at 8 bits, gain as `1 + gain_scale * q` and bias as
    /// `bias_scale * q`, and the values held are exactly those the stored integers give.
    Points { points: Vec<Point>, q8: Option<Vec<(f32, f32)>> },
    /// Parameter sets of a MicroModel's blocks, each exactly as the engine reads it: a base
    /// cartridge, seated as the job's one source of parameters in place of the open weights.
    Tensors(Vec<Tensor>),
    /// A base cartridge held as what differs from the open weights: every set its blocks read is
    /// named; `sets` are taken from the open weights (whole, or the rows or columns its `keep`
    /// names) and `edited` carry values of their own. Built at the seat from the open weights,
    /// which stay the one full copy on disk.
    Open { sets: Vec<OpenSet>, edited: Vec<Tensor> },
}

/// The rows or columns of an open-weights tensor a set keeps, as element index ranges along that
/// axis (`[rows, cols]` order), in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Keep {
    pub cols: bool,
    pub ranges: Vec<(usize, usize)>,
}

/// One parameter set taken from the open weights: its tensor name, place, element type, the shape
/// it has once taken, the content hash of those bytes, and what it keeps (`None`: whole).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenSet {
    pub name: String,
    pub layer: Option<usize>,
    pub block: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub sha256: String,
    pub keep: Option<Keep>,
}

/// One parameter set a cartridge carries: the tensor name the open weights use for the place it
/// loads onto, that place (layer, absent for the table and the final norm, and block), its element
/// type (`BF16`, `F16` or `F32`: stored whole, so the MicroModel serves it at whatever precision
/// is selected for it), its shape, and its bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub name: String,
    pub layer: Option<usize>,
    pub block: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

/// Where a cartridge file holds one parameter set, for seating it without reading the file into
/// memory first: its name, place (layer and block), element type, shape, content hash, the file,
/// and the bytes' offset and length in it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorPart {
    pub name: String,
    pub layer: Option<usize>,
    pub block: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub sha256: String,
    pub file: PathBuf,
    pub offset: u64,
    pub len: u64,
    /// Taken from the open weights at the seat (`keep`: the rows or columns; `None`: whole); the
    /// file, offset and length then say nothing.
    pub open: bool,
    pub keep: Option<Keep>,
}

/// Bytes a parameter set of `shape` takes in element type `dtype`, if the type is one a cartridge
/// holds.
#[must_use]
pub fn tensor_bytes(dtype: &str, shape: &[usize]) -> Option<usize> {
    let n: usize = shape.iter().product();
    match dtype {
        "BF16" | "F16" => Some(n * 2),
        "F32" => Some(n * 4),
        _ => None,
    }
}

/// The scale that maps the largest magnitude of `xs` to 127 (zero when all are zero).
#[must_use]
pub fn scale8(xs: &[f32]) -> f32 {
    xs.iter().fold(0.0_f32, |a, x| a.max(x.abs())) / 127.0
}

/// `x` as an 8-bit integer at `scale` (rounded, clamped; zero at scale zero).
#[must_use]
pub fn int8(x: f32, scale: f32) -> i8 {
    if scale == 0.0 {
        return 0;
    }
    #[expect(clippy::cast_possible_truncation, reason = "rounded and clamped to the 8-bit range")]
    let q = (x / scale).round().clamp(-127.0, 127.0) as i8;
    q
}

fn f32_text(v: &Value, k: &str) -> Result<f32, LibFault> {
    match v.get(k) {
        Ok(Value::Number(t)) => t.parse::<f32>().map_err(|e| LibFault::Schema(format!("{k}: {e}"))),
        _ => Err(LibFault::Schema(format!("{k} missing"))),
    }
}

impl Filter {
    /// One point at float32.
    #[must_use]
    pub fn point(target: &str, gain: Vec<f32>, bias: Vec<f32>) -> Self {
        Self::Points { points: vec![Point { target: target.to_owned(), gain, bias }], q8: None }
    }
    /// Points stored at 8 bits: each point's gain offsets from 1 and biases get one scale each, and
    /// the values held become exactly those the integers give.
    #[must_use]
    pub fn points_q8(points: Vec<Point>) -> Self {
        let mut scales = Vec::with_capacity(points.len());
        let points = points
            .into_iter()
            .map(|p| {
                let gs = scale8(&p.gain.iter().map(|g| g - 1.0).collect::<Vec<_>>());
                let bs = scale8(&p.bias);
                scales.push((gs, bs));
                Point {
                    gain: p.gain.iter().map(|g| 1.0 + gs * f32::from(int8(g - 1.0, gs))).collect(),
                    bias: p.bias.iter().map(|b| bs * f32::from(int8(*b, bs))).collect(),
                    target: p.target,
                }
            })
            .collect();
        Self::Points { points, q8: Some(scales) }
    }
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Settings(_) => "settings",
            Self::Points { .. } => "affine",
            Self::Tensors { .. } | Self::Open { .. } => "tensors",
        }
    }
    /// What it filters: `settings`, or the sockets joined by commas.
    #[must_use]
    pub fn target(&self) -> String {
        match self {
            Self::Settings(_) => "settings".to_owned(),
            Self::Tensors(tensors) => tensors.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(","),
            Self::Open { sets, edited } => sets.iter().map(|t| t.name.as_str()).chain(edited.iter().map(|t| t.name.as_str())).collect::<Vec<_>>().join(","),
            Self::Points { points, .. } => points.iter().map(|p| p.target.as_str()).collect::<Vec<_>>().join(","),
        }
    }
    /// The points it filters, with the numbers as used; none for settings.
    #[must_use]
    pub fn points(&self) -> Vec<Point> {
        match self {
            Self::Settings(_) | Self::Tensors(_) | Self::Open { .. } => Vec::new(),
            Self::Points { points, .. } => points.clone(),
        }
    }
    /// The parts of a cartridge file's label and its parameter bytes.
    fn encode(&self) -> (Vec<Value>, Vec<u8>) {
        let mut body: Vec<u8> = Vec::new();
        let mut parts = Vec::new();
        match self {
            Self::Settings(m) => {
                let b = Value::Object(m.iter().map(|(k, v)| (k.clone(), s(v))).collect()).encode().unwrap_or_default().into_bytes();
                parts.push(obj(vec![("form", s("settings")), ("encoding", s("json")), ("at", n(0)), ("bytes", n(b.len() as u64))]));
                body = b;
            }
            Self::Open { sets, edited } => {
                for t in sets {
                    let keep = t.keep.as_ref().map_or(Value::Null, |k| {
                        obj(vec![("axis", s(if k.cols { "cols" } else { "rows" })), ("ranges", Value::Array(k.ranges.iter().map(|(a, b)| Value::Array(vec![n(*a as u64), n(*b as u64)])).collect()))])
                    });
                    parts.push(obj(vec![
                        ("socket", s(&t.name)),
                        ("form", s("open")),
                        ("layer", t.layer.map_or(Value::Null, |l| n(l as u64))),
                        ("block", s(&t.block)),
                        ("shape", Value::Array(t.shape.iter().map(|x| n(*x as u64)).collect())),
                        ("encoding", s(&t.dtype)),
                        ("sha256", s(&t.sha256)),
                        ("keep", keep),
                        ("at", n(0)),
                        ("bytes", n(0)),
                    ]));
                }
                return Self::Tensors(edited.clone()).encode_more(parts, body);
            }
            Self::Tensors(tensors) => {
                for t in tensors {
                    let at = body.len() as u64;
                    body.extend_from_slice(&t.bytes);
                    parts.push(obj(vec![
                        ("socket", s(&t.name)),
                        ("form", s("tensor")),
                        ("layer", t.layer.map_or(Value::Null, |l| n(l as u64))),
                        ("block", s(&t.block)),
                        ("shape", Value::Array(t.shape.iter().map(|x| n(*x as u64)).collect())),
                        ("encoding", s(&t.dtype)),
                        ("sha256", s(&nmhash::sha256_bytes(&t.bytes))),
                        ("at", n(at)),
                        ("bytes", n(t.bytes.len() as u64)),
                    ]));
                }
            }
            Self::Points { points, q8 } => {
                for (i, p) in points.iter().enumerate() {
                    let at = body.len() as u64;
                    let mut row = vec![("socket", s(&p.target)), ("form", s("gain-bias")), ("shape", Value::Array(vec![n(p.gain.len() as u64)]))];
                    match q8.as_ref().and_then(|v| v.get(i)) {
                        Some((gs, bs)) => {
                            body.extend(p.gain.iter().map(|g| int8(g - 1.0, *gs)).chain(p.bias.iter().map(|b| int8(*b, *bs))).map(|v| v.to_le_bytes()[0]));
                            row.extend([("encoding", s("q8")), ("gain_scale", Value::Number(format!("{gs:e}"))), ("bias_scale", Value::Number(format!("{bs:e}")))]);
                        }
                        None => {
                            body.extend(p.gain.iter().chain(&p.bias).flat_map(|x| x.to_le_bytes()));
                            row.push(("encoding", s("f32")));
                        }
                    }
                    row.extend([("at", n(at)), ("bytes", n(body.len() as u64 - at))]);
                    parts.push(obj(row));
                }
            }
        }
        (parts, body)
    }
    /// `encode` of these sets after the parts already written.
    fn encode_more(&self, mut parts: Vec<Value>, mut body: Vec<u8>) -> (Vec<Value>, Vec<u8>) {
        let (more, b) = self.encode();
        for mut p in more {
            if let Value::Object(o) = &mut p
                && let Some(Value::Number(at)) = o.get("at").cloned()
            {
                o.insert("at".into(), n(at.parse::<u64>().unwrap_or(0) + body.len() as u64));
            }
            parts.push(p);
        }
        body.extend(b);
        (parts, body)
    }
    /// Reads the filter from a label's parts and the parameter bytes.
    fn decode(parts: &[Value], body: &[u8]) -> Result<Self, LibFault> {
        let mut tensors = Vec::new();
        let mut opens = Vec::new();
        let mut points = Vec::new();
        let mut scales = Vec::new();
        for p in parts {
            let at = usize::try_from(p.get("at").and_then(Value::integer).unwrap_or(0)).unwrap_or(usize::MAX);
            let len = usize::try_from(p.get("bytes").and_then(Value::integer).unwrap_or(0)).unwrap_or(0);
            let raw = body.get(at..at.saturating_add(len)).ok_or_else(|| LibFault::Shape("a part lies outside the parameters".into()))?;
            match (text(p, "form").as_str(), text(p, "encoding").as_str()) {
                ("settings", "json") => {
                    let v = nmjson::parse(raw).map_err(|e| LibFault::Schema(format!("settings: {e}")))?;
                    let o = v.object().map_err(|e| LibFault::Schema(format!("settings: {e}")))?;
                    return Ok(Self::Settings(o.iter().map(|(k, v)| v.string().map(|x| (k.clone(), x.to_owned())).map_err(|e| LibFault::Schema(format!("{k}: {e}")))).collect::<Result<_, _>>()?));
                }
                ("gain-bias", enc) => {
                    let w = p.get("shape").and_then(Value::array).ok().and_then(|a| a.first()?.integer().ok()).and_then(|x| usize::try_from(x).ok()).unwrap_or(0);
                    let target = need(p, "socket")?;
                    let (gain, bias): (Vec<f32>, Vec<f32>) = match enc {
                        "q8" if raw.len() == 2 * w => {
                            let (gs, bs) = (f32_text(p, "gain_scale")?, f32_text(p, "bias_scale")?);
                            scales.push((gs, bs));
                            let q: Vec<i8> = raw.iter().map(|x| i8::from_le_bytes([*x])).collect();
                            let (gq, bq) = q.split_at(w);
                            (gq.iter().map(|v| 1.0 + gs * f32::from(*v)).collect(), bq.iter().map(|v| bs * f32::from(*v)).collect())
                        }
                        "f32" if raw.len() == 8 * w => {
                            let xs: Vec<f32> = raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
                            let (g, b) = xs.split_at(w);
                            (g.to_vec(), b.to_vec())
                        }
                        e => return Err(LibFault::Shape(format!("{target}: {e} parameters of {} bytes for a width of {w}", raw.len()))),
                    };
                    points.push(Point { target, gain, bias });
                }
                ("tensor", dtype) => {
                    let shape = shape_of(p);
                    if tensor_bytes(dtype, &shape) != Some(raw.len()) {
                        return Err(LibFault::Shape(format!("{}: {dtype} {shape:?} in {} bytes", text(p, "socket"), raw.len())));
                    }
                    tensors.push(Tensor { name: need(p, "socket")?, layer: layer_of(p), block: text(p, "block"), dtype: dtype.to_owned(), shape, bytes: raw.to_vec() });
                }
                ("open", dtype) => opens.push(OpenSet {
                    name: need(p, "socket")?,
                    layer: layer_of(p),
                    block: text(p, "block"),
                    dtype: dtype.to_owned(),
                    shape: shape_of(p),
                    sha256: text(p, "sha256"),
                    keep: keep_of(p)?,
                }),
                ("assembly", _) => return Err(LibFault::Schema("a cartridge holds parameters, never structure: an assembly part is refused".into())),
                (form, enc) => return Err(LibFault::Schema(format!("form {form} ({enc}) is not seated by this build (it seats tensor, gain-bias and settings)"))),
            }
        }
        if !opens.is_empty() {
            return if points.is_empty() { Ok(Self::Open { sets: opens, edited: tensors }) } else { Err(LibFault::Schema("a cartridge holds parameter sets or gain-bias parts, not both".into())) };
        }
        if !tensors.is_empty() {
            return if points.is_empty() { Ok(Self::Tensors(tensors)) } else { Err(LibFault::Schema("a cartridge holds tensors or gain-bias parts, not both".into())) };
        }
        let q8 = (!scales.is_empty()).then_some(scales);
        if q8.as_ref().is_some_and(|v| v.len() != points.len()) {
            return Err(LibFault::Schema("points mix 8-bit and float32 encodings".into()));
        }
        Ok(Self::Points { points, q8 })
    }
    /// A filter from a cartridge written before the one-file form: its kind, target and files.
    fn from_folder(kind: &str, target: &str, files: &BTreeMap<String, Vec<u8>>) -> Result<Self, LibFault> {
        match kind {
            "settings" => {
                let b = files.get("settings.json").ok_or_else(|| LibFault::Schema("settings.json missing".into()))?;
                let v = nmjson::parse(b).map_err(|e| LibFault::Schema(format!("settings.json: {e}")))?;
                let o = v.get("values").and_then(Value::object).map_err(|e| LibFault::Schema(format!("settings.json values: {e}")))?;
                let m = o.iter().map(|(k, v)| v.string().map(|x| (k.clone(), x.to_owned())).map_err(|e| LibFault::Schema(format!("{k}: {e}")))).collect::<Result<_, _>>()?;
                Ok(Self::Settings(m))
            }
            "affine" => {
                let b = files.get("numbers.f32").ok_or_else(|| LibFault::Schema("numbers.f32 missing".into()))?;
                if !b.len().is_multiple_of(8) {
                    return Err(LibFault::Shape("numbers.f32 does not hold gain and bias of one length".into()));
                }
                let xs: Vec<f32> = b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
                let (g, bi) = xs.split_at(xs.len() / 2);
                Ok(Self::point(target, g.to_vec(), bi.to_vec()))
            }
            o => Err(LibFault::Schema(format!("{o} is not a filter kind (settings, affine)"))),
        }
    }
}

fn layer_of(p: &Value) -> Option<usize> {
    p.get("layer").and_then(Value::integer).ok().and_then(|x| usize::try_from(x).ok())
}

fn shape_of(p: &Value) -> Vec<usize> {
    p.get("shape").and_then(Value::array).unwrap_or_default().iter().filter_map(|x| x.integer().ok().and_then(|v| usize::try_from(v).ok())).collect()
}

/// A cartridge file: the mark, the label's length, the label, the parameters.
#[must_use]
pub fn cartridge_file(label: &Value, body: &[u8]) -> Vec<u8> {
    let l = label.encode().unwrap_or_default();
    let mut out = CARTRIDGE_MARK.to_vec();
    out.extend_from_slice(&u32::try_from(l.len()).unwrap_or(u32::MAX).to_le_bytes());
    out.extend_from_slice(l.as_bytes());
    out.extend_from_slice(body);
    out
}

/// Splits a cartridge file into its label and parameters, the parameters checked against the
/// label's hash.
/// # Errors
/// Not a cartridge file, or parameters that do not match the label.
pub fn read_cartridge(b: &[u8]) -> Result<(Value, &[u8]), LibFault> {
    if b.get(..8) != Some(CARTRIDGE_MARK.as_slice()) {
        return Err(LibFault::Schema("not a cartridge file".into()));
    }
    let len = b.get(8..12).and_then(|x| <[u8; 4]>::try_from(x).ok()).map_or(0, |x| u32::from_le_bytes(x) as usize);
    let label = nmjson::parse(b.get(12..12 + len).ok_or_else(|| LibFault::Schema("label cut short".into()))?).map_err(|e| LibFault::Schema(format!("label: {e}")))?;
    if text(&label, "schema") != CARTRIDGE_SCHEMA {
        return Err(LibFault::Schema(format!("label schema is not {CARTRIDGE_SCHEMA}")));
    }
    let body = b.get(12 + len..).unwrap_or_default();
    if nmhash::sha256_bytes(body) != text(&label, "parameters_sha256") {
        return Err(LibFault::ContentHash { cartridge: format!("{}@{}", text(&label, "id"), label.get("version").and_then(Value::integer).unwrap_or(0)) });
    }
    Ok((label, body))
}

/// The MicroModel a set of open weights serves, by what it does: `reranker` (the judge), `reader`
/// (the drones' and tentacles' reading and the second look), `embedder`, `writer`; any other open
/// weights by their own id.
#[must_use]
pub fn role_of(model: &str) -> &str {
    match model {
        "qwen3-reranker-0.6b" => "reranker",
        "qwen3-0.6b" => "reader",
        "qwen3-embedding-0.6b" => "embedder",
        "qwen3-1.7b" => "writer",
        other => other,
    }
}

/// Where a cartridge file is kept under `library/cartridges/`, by kind, then by MicroModel or
/// hardware (owner direction 2026-11-17), its version in its name: a base cartridge (parameter sets)
/// `base/<role>/<id>-v<n>.cartridge`; a settings cartridge for hardware
/// `hardware/<id>/<id>-v<n>.cartridge`; an adapter `adapter/<id>/`; a preference cartridge
/// `preference/<id>/`; any other learned cartridge `learned/<id>/`. One file per folder on the drive,
/// the current version; earlier versions live in the backup repository.
#[must_use]
pub fn shelf_path(family: &str, kind: &str, domain: &str, model: &str, id: &str, version: u64) -> String {
    let file = format!("{id}-v{version}.{CARTRIDGE_EXT}");
    match (family, kind) {
        (ADAPTER, _) => format!("adapter/{id}/{file}"),
        (PREFERENCE, _) => format!("preference/{id}/{file}"),
        (_, "tensors") => format!("base/{}/{file}", role_of(model)),
        (_, "settings") if domain.starts_with("hardware") => format!("hardware/{id}/{file}"),
        _ => format!("learned/{id}/{file}"),
    }
}

/// A base cartridge version's record, kept in its label's notes when it is written and on its
/// catalogue row (owner direction 2026-11-17): the version it was made over, whether and how it was
/// mapped and pruned, any other edits (compensation, detail edits), and its measured result on its
/// role. A row's value stands in for a label written before the field existed.
pub const RECORD: [&str; 5] = ["parent", "mapped", "pruned", "edits", "measured"];

/// Sets `key` in a cartridge's notes (made an object when it is not one); `over` replaces a value
/// already there.
fn note(notes: &mut Value, key: &str, value: &str, over: bool) {
    if !matches!(notes, Value::Object(_)) {
        *notes = Value::Object(BTreeMap::new());
    }
    if let Value::Object(o) = notes
        && (over || !o.contains_key(key))
    {
        o.insert(key.to_owned(), s(value));
    }
}

/// The folder a cartridge file of the earlier flat layout is kept in: `preferences/` for preference
/// cartridges, `adapters/` for adapter cartridges, else `cartridges/`. A file named by a path
/// (`base/reranker/...`) is under `cartridges/` in the layout by kind (`shelf_path`).
fn shelf_folder(family: &str) -> &'static str {
    match family {
        PREFERENCE => "preferences",
        ADAPTER => "adapters",
        _ => "cartridges",
    }
}

/// The family of adapter cartridges: small learned sets that stack on a base (the socket form),
/// kept on their own shelf (`adapters/`, the catalogue's `adapters` section), never with the base
/// cartridges. Can specialize either a base model (`BASE_SPECIALIZATION_SCOPE`) or a project
/// (`PROJECT_SCOPE`).
pub const ADAPTER: &str = "adapter";
/// Adapter scope for base-model specialization (role, job, or domain specialization layered on a
/// base cartridge).
pub const BASE_SPECIALIZATION_SCOPE: &str = "base_specialization";
/// Adapter scope for per-project specialization (lenses and project code adapters).
pub const PROJECT_SCOPE: &str = "project";

/// Whether a cartridge of `family` and `kind` (`tensors`, `affine`, `settings`) may exist: a base
/// cartridge (parameter sets) is never an adapter or a preference, and an adapter holds learned
/// sets at sockets only.
/// # Errors
/// The mismatch, named.
pub fn shelf_rule(key: &str, family: &str, kind: &str) -> Result<(), LibFault> {
    match kind {
        "tensors" if family == ADAPTER || family == PREFERENCE => Err(LibFault::Conflict(format!("{key}: a base cartridge (parameter sets) never goes on the {family} shelf"))),
        "settings" if family == ADAPTER => Err(LibFault::Conflict(format!("{key}: a settings cartridge is not an adapter"))),
        _ => Ok(()),
    }
}

/// The parameter sets a cartridge file carries, as places in the file: the loading path seats
/// each by its name and address. The file is read once, as a stream: the label, then the
/// parameters hashed against the label's `parameters_sha256`. A label without a name, what it
/// does, what it was made from or when, or a part that carries structure, is refused.
/// # Errors
/// An unreadable file, not a cartridge, parameters that do not match, a missing label, an assembly
/// part, or a parameter set whose bytes do not fit its type and shape.
pub fn tensor_parts_of(file: &Path) -> Result<Vec<TensorPart>, LibFault> {
    use std::io::{Read as _, Seek as _};
    let io_err = |e: std::io::Error| LibFault::Io(format!("{}: {e}", file.display()));
    let mut f = std::fs::File::open(file).map_err(io_err)?;
    let mut head = [0_u8; 12];
    f.read_exact(&mut head).map_err(io_err)?;
    if head.get(..8) != Some(CARTRIDGE_MARK.as_slice()) {
        return Err(LibFault::Schema("not a cartridge file".into()));
    }
    let len = head.get(8..12).and_then(|x| <[u8; 4]>::try_from(x).ok()).map_or(0, |x| u32::from_le_bytes(x) as usize);
    let mut lb = vec![0_u8; len];
    f.read_exact(&mut lb).map_err(io_err)?;
    let label = nmjson::parse(&lb).map_err(|e| LibFault::Schema(format!("label: {e}")))?;
    if text(&label, "schema") != CARTRIDGE_SCHEMA {
        return Err(LibFault::Schema(format!("label schema is not {CARTRIDGE_SCHEMA}")));
    }
    let start = 12 + len as u64;
    let total = f.metadata().map_err(io_err)?.len();
    f.seek(std::io::SeekFrom::Start(start)).map_err(io_err)?;
    if nmhash::sha256(&mut f).map_err(io_err)? != text(&label, "parameters_sha256") {
        return Err(LibFault::ContentHash { cartridge: format!("{}@{}", text(&label, "id"), label.get("version").and_then(Value::integer).unwrap_or(0)) });
    }
    let name = file.file_name().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default();
    Cartridge::from_label(&label, FileRec { name, sha256: String::new(), bytes: total }, false)?;
    let file = std::fs::canonicalize(file).map_err(|e| LibFault::Io(e.to_string()))?;
    let mut out = Vec::new();
    for p in label.get("parts").and_then(Value::array).unwrap_or_default() {
        let at = p.get("at").and_then(Value::integer).unwrap_or(0);
        let bytes = p.get("bytes").and_then(Value::integer).unwrap_or(0);
        if start + at + bytes > total {
            return Err(LibFault::Shape("a part lies outside the parameters".into()));
        }
        match text(p, "form").as_str() {
            "assembly" => return Err(LibFault::Schema("a cartridge holds parameters, never structure: an assembly part is refused".into())),
            "open" => {
                let (dtype, shape) = (text(p, "encoding"), shape_of(p));
                let len = tensor_bytes(&dtype, &shape).ok_or_else(|| LibFault::Schema(format!("{}: element type {dtype} is not seated (BF16, F16, F32)", text(p, "socket"))))?;
                out.push(TensorPart {
                    name: need(p, "socket")?,
                    layer: layer_of(p),
                    block: text(p, "block"),
                    dtype,
                    shape,
                    sha256: text(p, "sha256"),
                    file: PathBuf::new(),
                    offset: 0,
                    len: len as u64,
                    open: true,
                    keep: keep_of(p)?,
                });
            }
            "tensor" => {
                let (dtype, shape) = (text(p, "encoding"), shape_of(p));
                match tensor_bytes(&dtype, &shape) {
                    Some(b) if b as u64 == bytes => {}
                    Some(_) => return Err(LibFault::Shape(format!("{}: {dtype} {shape:?} in {bytes} bytes", text(p, "socket")))),
                    None => return Err(LibFault::Schema(format!("{}: element type {dtype} is not seated (BF16, F16, F32)", text(p, "socket")))),
                }
                out.push(TensorPart {
                    name: need(p, "socket")?,
                    layer: layer_of(p),
                    block: text(p, "block"),
                    dtype,
                    shape,
                    sha256: text(p, "sha256"),
                    file: file.clone(),
                    offset: start + at,
                    len: bytes,
                    open: false,
                    keep: None,
                });
            }
            _ => {}
        }
    }
    Ok(out)
}

/// A cartridge file's label, read from its head alone (the parameters are not read or checked).
/// # Errors
/// An unreadable file, or not a cartridge.
pub fn label_of(file: &Path) -> Result<Value, LibFault> {
    use std::io::Read as _;
    let io_err = |e: std::io::Error| LibFault::Io(format!("{}: {e}", file.display()));
    let mut f = std::fs::File::open(file).map_err(io_err)?;
    let mut head = [0_u8; 12];
    f.read_exact(&mut head).map_err(io_err)?;
    let len = head.get(8..12).and_then(|x| <[u8; 4]>::try_from(x).ok()).map_or(0, |x| u32::from_le_bytes(x) as usize);
    let mut b = head.to_vec();
    b.resize(12 + len, 0);
    f.read_exact(b.get_mut(12..).unwrap_or_default()).map_err(io_err)?;
    read_label(&b).ok_or_else(|| LibFault::Schema(format!("{}: not a cartridge file", file.display())))
}

/// The `keep` of an `open` part: absent or null for a whole set, else its axis and ranges.
fn keep_of(p: &Value) -> Result<Option<Keep>, LibFault> {
    let Ok(k) = p.get("keep") else { return Ok(None) };
    if matches!(k, Value::Null) {
        return Ok(None);
    }
    let cols = match text(k, "axis").as_str() {
        "cols" => true,
        "rows" => false,
        a => return Err(LibFault::Schema(format!("{}: keep axis {a:?} (rows or cols)", text(p, "socket")))),
    };
    let int = |v: &Value| v.integer().ok().and_then(|x| usize::try_from(x).ok());
    let ranges = k
        .get("ranges")
        .and_then(Value::array)
        .unwrap_or_default()
        .iter()
        .map(|r| match r.array().unwrap_or_default() {
            [a, b] => match (int(a), int(b)) {
                (Some(a), Some(b)) if a < b => Ok((a, b)),
                _ => Err(LibFault::Schema(format!("{}: keep range {a:?}..{b:?}", text(p, "socket")))),
            },
            _ => Err(LibFault::Schema(format!("{}: a keep range is two numbers", text(p, "socket")))),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(Keep { cols, ranges }))
}

/// A cartridge file's label alone (no parameter check), for listing.
fn read_label(b: &[u8]) -> Option<Value> {
    let len = b.get(8..12).and_then(|x| <[u8; 4]>::try_from(x).ok()).map(|x| u32::from_le_bytes(x) as usize)?;
    (b.get(..8) == Some(CARTRIDGE_MARK.as_slice())).then(|| nmjson::parse(b.get(12..12 + len)?).ok()).flatten()
}

/// Applies settings filters in order to a copy of `base`: later values replace earlier ones.
#[must_use]
pub fn apply_settings(base: &BTreeMap<String, String>, filters: &[&Filter]) -> BTreeMap<String, String> {
    let mut out = base.clone();
    for f in filters {
        if let Filter::Settings(m) = f {
            out.extend(m.iter().map(|(k, v)| (k.clone(), v.clone())));
        }
    }
    out
}

/// Applies the affine filters for `target` in order to a copy of `base`. `base` is only read.
/// # Errors
/// A filter whose length differs from the vector's.
pub fn apply_numbers(target: &str, base: &[f32], filters: &[&Filter]) -> Result<Vec<f32>, LibFault> {
    let mut x = base.to_vec();
    for f in filters {
        for Point { target: t, gain, bias } in f.points().iter().filter(|p| p.target == target) {
            if gain.len() != x.len() || bias.len() != x.len() {
                return Err(LibFault::Shape(format!("{t}: filter of {} values over a vector of {}", gain.len(), x.len())));
            }
            for ((v, g), b) in x.iter_mut().zip(gain).zip(bias) {
                *v = v.mul_add(*g, *b);
            }
        }
    }
    Ok(x)
}

// ---------------------------------------------------------------------------------------------
// Evidence.

/// How strong a piece of evidence is (the evidence ledger's labels).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strength {
    Direct,
    Indirect,
    Weak,
}
impl Strength {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Indirect => "indirect",
            Self::Weak => "weak",
        }
    }
    #[must_use]
    pub fn parse(t: &str) -> Option<Self> {
        match t {
            "direct" => Some(Self::Direct),
            "indirect" => Some(Self::Indirect),
            "weak" => Some(Self::Weak),
            _ => None,
        }
    }
}

/// One measurement about a cartridge: on which hardware piece (its exact identity), with which
/// outcome (`verified`, `refuted`, or another ledger state that does not count) and strength.
#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceRow {
    /// The cartridge it is about (`id@version`).
    pub subject: String,
    pub hardware: String,
    pub outcome: String,
    pub strength: Option<Strength>,
    pub measure: String,
    pub source: String,
    pub at: u64,
}
impl EvidenceRow {
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("subject", s(&self.subject)),
            ("hardware", s(&self.hardware)),
            ("outcome", s(&self.outcome)),
            ("strength", self.strength.map_or(Value::Null, |x| s(x.name()))),
            ("measure", s(&self.measure)),
            ("source", s(&self.source)),
            ("at", n(self.at)),
        ])
    }
    /// # Errors
    /// A missing subject, hardware or outcome.
    pub fn from_value(v: &Value) -> Result<Self, LibFault> {
        Ok(Self {
            subject: need(v, "subject")?,
            hardware: need(v, "hardware")?,
            outcome: need(v, "outcome")?,
            strength: v.get("strength").and_then(Value::string).ok().and_then(Strength::parse),
            measure: text(v, "measure"),
            source: text(v, "source"),
            at: v.get("at").and_then(Value::integer).unwrap_or(0),
        })
    }
}

/// Where evidence comes from (the evidence ledger, recorded rows): the rows about one cartridge.
pub trait EvidenceSource {
    fn rows_for(&self, subject: &str) -> Vec<EvidenceRow>;
}

/// Counts for one piece: (verified, refuted) among rows measured on exactly `hardware` with
/// strength direct or indirect. Rows from other hardware and weak rows never count.
#[must_use]
pub fn counts(rows: &[EvidenceRow], hardware: &str) -> (usize, usize) {
    let c = |o: &str| rows.iter().filter(|r| r.hardware == hardware && r.outcome == o && matches!(r.strength, Some(Strength::Direct | Strength::Indirect))).count();
    (c("verified"), c("refuted"))
}

/// Whether the rows let something come off trial for `hardware`: at least `READY_ROWS` verified,
/// and at least twice as many verified as refuted.
#[must_use]
pub fn ready(rows: &[EvidenceRow], hardware: &str) -> bool {
    let (v, r) = counts(rows, hardware);
    v >= READY_ROWS && v >= 2 * r
}

// ---------------------------------------------------------------------------------------------
// Catalogue entries.

/// One file of a cartridge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileRec {
    pub name: String,
    pub sha256: String,
    pub bytes: u64,
}

/// The content hash of a file list (`name sha256` lines in name order).
#[must_use]
pub fn content_hash(files: &[FileRec]) -> String {
    let mut f: Vec<&FileRec> = files.iter().collect();
    f.sort_by(|a, b| a.name.cmp(&b.name));
    nmhash::sha256_bytes(f.iter().map(|x| format!("{} {}\n", x.name, x.sha256)).collect::<String>().as_bytes())
}

/// The one family the library treats apart: preference cartridges (the user's ways) are kept in
/// their own folder and plugged only by stacks for a `preference` use, never mixed with others.
pub const PREFERENCE: &str = "preference";

/// Whether a stack's use is a preference use (`preference` or `preference:<scope>`).
#[must_use]
pub fn preference_use(use_for: &str) -> bool {
    use_for == PREFERENCE || use_for.starts_with("preference:")
}

/// What a new cartridge is: its identity, its family and knowledge domain, and which MicroModel
/// and layers it filters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CartridgeSpec {
    pub id: String,
    pub version: u64,
    /// A plain description of what it is (`fundamentals`, `learning`, ...); only `preference`
    /// changes anything: such a cartridge is stored and switched apart.
    pub family: String,
    /// The knowledge domain it holds (`rust-fundamentals`, `nano-system`, `hardware:gpu`,
    /// `preferences`, ...).
    pub domain: String,
    /// Where it applies: `general`, `hardware:<piece>`, `project:<key>` or `user`.
    pub scope: String,
    /// The MicroModel it filters, or `settings` for a settings filter.
    pub model: String,
    /// SHA-256 of the bare weights it was made for (empty for settings).
    pub weights: String,
    /// First and last layer it filters, when it filters weights at layers.
    pub layers: Option<(u64, u64)>,
    /// A short plain name.
    pub name: String,
    /// One or two plain sentences: what it does and what it is for.
    pub description: String,
    /// What it was made from: the material or evidence, and the bare weights it was made over.
    pub made_from: String,
    /// Its measured effect, where one exists (for example counts with it plugged against pulled).
    pub effect: String,
}

/// A cartridge in the catalogue: learned parameters on one knowledge domain.
#[derive(Clone, Debug, PartialEq)]
pub struct Cartridge {
    pub id: String,
    pub version: u64,
    /// `fundamentals`, `learning` or `preference`.
    pub family: String,
    pub domain: String,
    pub scope: String,
    pub model: String,
    /// SHA-256 of the bare weights it was made for (empty for settings).
    pub weights: String,
    /// The socket contract its parts are written for (empty for settings).
    pub sockets: String,
    pub layers: Option<(u64, u64)>,
    /// `settings` or `affine`.
    pub kind: String,
    /// What it filters: `settings`, or the vector's name.
    pub target: String,
    /// The label: name, what it does, what it was made from, when, its measured effect.
    pub name: String,
    pub description: String,
    pub made_from: String,
    pub made: String,
    pub effect: String,
    pub files: Vec<FileRec>,
    pub hash: String,
    pub evidence: Vec<EvidenceRow>,
    /// Notes on the label (details of how it was made and measured).
    pub notes: Value,
    pub trial: bool,
}
impl Cartridge {
    /// One file in the one-file form (`<id>-v<version>.cartridge`), not a folder of files.
    #[must_use]
    pub fn one_file(&self) -> bool {
        self.files.len() == 1 && self.files.first().is_some_and(|f| f.name.ends_with(&format!(".{CARTRIDGE_EXT}")))
    }
    /// The entry a cartridge file's label gives, with its file record and trial flag.
    /// # Errors
    /// A label field missing.
    pub fn from_label(label: &Value, file: FileRec, trial: bool) -> Result<Self, LibFault> {
        if let Some(k) = ["name", "does", "made_from", "made", "model"].into_iter().find(|k| text(label, k).trim().is_empty()) {
            return Err(LibFault::Schema(format!("{}: the label has no {k}; a cartridge without a label is refused", file.name)));
        }
        let parts = label.get("parts").and_then(Value::array).unwrap_or_default();
        let settings = parts.iter().any(|p| text(p, "form") == "settings");
        let tensors = parts.iter().any(|p| text(p, "form") == "tensor");
        let files = vec![file];
        Ok(Self {
            id: need(label, "id")?,
            version: label.get("version").and_then(Value::integer).map_err(|e| LibFault::Schema(format!("version: {e}")))?,
            family: text(label, "family"),
            domain: text(label, "domain"),
            scope: text(label, "scope"),
            model: need(label, "model")?,
            weights: text(label, "weights"),
            sockets: text(label, "sockets"),
            layers: label.get("layers").and_then(Value::array).ok().and_then(|a| Some((a.first()?.integer().ok()?, a.get(1)?.integer().ok()?))),
            kind: (if settings { "settings" } else if tensors { "tensors" } else { "affine" }).to_owned(),
            target: if settings { "settings".to_owned() } else { parts.iter().map(|p| text(p, "socket")).filter(|x| !x.is_empty()).collect::<Vec<_>>().join(",") },
            name: text(label, "name"),
            description: text(label, "does"),
            made_from: text(label, "made_from"),
            made: text(label, "made"),
            effect: text(label, "effect"),
            hash: content_hash(&files),
            files,
            evidence: label.get("evidence").and_then(Value::array).unwrap_or_default().iter().map(EvidenceRow::from_value).collect::<Result<_, _>>()?,
            notes: label.get("notes").cloned().unwrap_or(Value::Null),
            trial,
        })
    }
    /// The label as plain lines for a person: name, what it does, where it loads, what it was made
    /// from and when, its size, and its measured effect.
    #[must_use]
    pub fn label_text(&self) -> String {
        let place = match (self.model.as_str(), self.layers) {
            ("settings", _) => "the Nano's settings".to_owned(),
            (m, Some((a, b))) if a == b => format!("{m}, layer {a}, at {}", self.target),
            (m, Some((a, b))) => format!("{m}, layers {a} to {b}, at {}", self.target),
            (m, None) => format!("{m}, at {}", self.target),
        };
        let mut out = format!(
            "{} ({})\n  {}\n  loads into: {place}\n  made from: {}\n  made: {}; size {} bytes\n",
            if self.name.is_empty() { "unlabelled" } else { &self.name },
            self.key(),
            self.description,
            self.made_from,
            self.made,
            self.bytes()
        );
        if !self.effect.is_empty() {
            out.push_str(&format!("  measured effect: {}\n", self.effect));
        }
        out
    }
    /// The shelf's record of it: which file holds it and its hash, its trial flag, and the
    /// MicroModel and kind its label states, so a cartridge whose file is gone is still refused for
    /// the right MicroModel by name (a cartridge in the earlier folder form keeps its full entry).
    #[must_use]
    pub fn shelf_value(&self) -> Value {
        match self.files.first().filter(|_| self.one_file()) {
            Some(f) => {
                let mut row = vec![("id", s(&self.id)), ("version", n(self.version)), ("file", s(&f.name)), ("sha256", s(&f.sha256)), ("bytes", n(f.bytes)), ("trial", Value::Bool(self.trial))];
                if !self.model.is_empty() {
                    row.push(("model", s(&self.model)));
                }
                if !self.kind.is_empty() {
                    row.push(("kind", s(&self.kind)));
                }
                if !self.scope.is_empty() {
                    row.push(("scope", s(&self.scope)));
                }
                if !self.domain.is_empty() {
                    row.push(("domain", s(&self.domain)));
                }
                for k in RECORD {
                    if let Ok(p) = self.notes.get(k).and_then(Value::string) {
                        row.push((k, s(p)));
                    }
                }
                obj(row)
            }
            None => self.to_value(),
        }
    }
    /// `id@version`.
    #[must_use]
    pub fn key(&self) -> String {
        format!("{}@{}", self.id, self.version)
    }
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.files.iter().map(|f| f.bytes).sum()
    }
    fn fixed(&self) -> Vec<(&'static str, Value)> {
        vec![
            ("id", s(&self.id)),
            ("version", n(self.version)),
            ("family", s(&self.family)),
            ("domain", s(&self.domain)),
            ("scope", s(&self.scope)),
            ("model", s(&self.model)),
            ("weights", s(&self.weights)),
            ("layers", self.layers.map_or(Value::Null, |(a, b)| Value::Array(vec![n(a), n(b)]))),
            ("kind", s(&self.kind)),
            ("target", s(&self.target)),
            ("name", s(&self.name)),
            ("description", s(&self.description)),
            ("made_from", s(&self.made_from)),
            ("made", s(&self.made)),
            ("effect", s(&self.effect)),
            ("files", Value::Array(self.files.iter().map(|f| obj(vec![("name", s(&f.name)), ("sha256", s(&f.sha256)), ("bytes", n(f.bytes))])).collect())),
            ("hash", s(&self.hash)),
            ("evidence", Value::Array(self.evidence.iter().map(EvidenceRow::to_value).collect())),
            ("notes", self.notes.clone()),
        ]
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut f = self.fixed();
        f.push(("trial", Value::Bool(self.trial)));
        obj(f)
    }
    /// Reads a full entry (an earlier catalogue, or an exported stack).
    /// # Errors
    /// A missing field.
    pub fn from_value(v: &Value) -> Result<Self, LibFault> {
        let domain = text(v, "domain");
        let family = text(v, "family");
        let files = v
            .get("files")
            .and_then(Value::array)
            .map_err(|e| LibFault::Schema(format!("files: {e}")))?
            .iter()
            .map(|f| Ok(FileRec { name: need(f, "name")?, sha256: need(f, "sha256")?, bytes: f.get("bytes").and_then(Value::integer).unwrap_or(0) }))
            .collect::<Result<Vec<_>, LibFault>>()?;
        Ok(Self {
            id: need(v, "id")?,
            version: v.get("version").and_then(Value::integer).map_err(|e| LibFault::Schema(format!("version: {e}")))?,
            family,
            domain,
            scope: text(v, "scope"),
            model: need(v, "model")?,
            weights: text(v, "weights"),
            sockets: text(v, "sockets"),
            layers: v.get("layers").and_then(Value::array).ok().and_then(|a| Some((a.first()?.integer().ok()?, a.get(1)?.integer().ok()?))),
            kind: need(v, "kind")?,
            target: need(v, "target")?,
            name: text(v, "name"),
            description: text(v, "description"),
            made_from: text(v, "made_from"),
            made: text(v, "made"),
            effect: text(v, "effect"),
            files,
            hash: need(v, "hash")?,
            evidence: v.get("evidence").and_then(Value::array).unwrap_or_default().iter().map(EvidenceRow::from_value).collect::<Result<_, _>>()?,
            notes: v.get("notes").cloned().unwrap_or(Value::Null),
            trial: flag(v, "trial"),
        })
    }
}

/// A stack: the cartridges plugged in for one use (the work at hand), in order, and (for a hardware
/// use) the piece it is for.
#[derive(Clone, Debug, PartialEq)]
pub struct Stack {
    pub id: String,
    pub version: u64,
    pub use_for: String,
    pub piece: Option<Piece>,
    pub cartridges: Vec<String>,
    pub description: String,
    pub hash: String,
    pub trial: bool,
}
impl Stack {
    #[must_use]
    pub fn key(&self) -> String {
        format!("{}@{}", self.id, self.version)
    }
    fn fixed(&self) -> Vec<(&'static str, Value)> {
        vec![
            ("id", s(&self.id)),
            ("version", n(self.version)),
            ("use", s(&self.use_for)),
            ("piece", self.piece.as_ref().map_or(Value::Null, Piece::to_value)),
            ("cartridges", Value::Array(self.cartridges.iter().map(|c| s(c)).collect())),
            ("description", s(&self.description)),
            ("hash", s(&self.hash)),
        ]
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut f = self.fixed();
        f.push(("trial", Value::Bool(self.trial)));
        obj(f)
    }
    /// # Errors
    /// A missing field or a piece that does not parse.
    pub fn from_value(v: &Value) -> Result<Self, LibFault> {
        let piece = match v.get("piece") {
            Ok(Value::Null) | Err(_) => None,
            Ok(p) => Some(Piece::from_value(p).map_err(LibFault::Schema)?),
        };
        Ok(Self {
            id: need(v, "id")?,
            version: v.get("version").and_then(Value::integer).map_err(|e| LibFault::Schema(format!("version: {e}")))?,
            use_for: need(v, "use")?,
            piece,
            // A stack written before the name was settled lists its members under `books`.
            cartridges: v.get("cartridges").or_else(|_| v.get("books")).and_then(Value::array).unwrap_or_default().iter().filter_map(|c| c.string().ok().map(str::to_owned)).collect(),
            description: text(v, "description"),
            hash: need(v, "hash")?,
            trial: flag(v, "trial"),
        })
    }
}

/// A stack's hash: its fixed fields (hash left out) and each cartridge's content hash in order.
#[must_use]
pub fn stack_hash(st: &Stack, members: &[&Cartridge]) -> String {
    let mut fixed: BTreeMap<String, Value> = st.fixed().into_iter().map(|(k, v)| (k.to_owned(), v)).collect();
    fixed.remove("hash");
    let body = Value::Object(fixed).encode().unwrap_or_default();
    let mem: String = members.iter().map(|e| format!("{} {}\n", e.key(), e.hash)).collect();
    nmhash::sha256_bytes(format!("{body}\n{mem}").as_bytes())
}

/// Settled cartridge stacking hierarchy (order of authority, 2026-11-17 07:12 UTC):
/// `0: base` -> `1: hardware/settings` -> `2: adapter` (base specialization & project)
/// -> `3: learned` -> `4: preference`.
#[must_use]
pub fn family_rank(family: &str) -> u8 {
    match family {
        "base" => 0,
        "hardware" | "settings" => 1,
        ADAPTER => 2,
        PREFERENCE => 4,
        _ => 3, // learned / fundamentals / compensation filters
    }
}

/// Order of uses when stacks for several uses apply together: general first, then hardware from
/// the most general piece to the most specific, then project, then job.
#[must_use]
pub fn use_rank(use_for: &str) -> (u8, u8) {
    if preference_use(use_for) {
        return (5, 0);
    }
    match use_for.split_once(':') {
        None if use_for == "general" => (0, 0),
        Some(("hardware", k)) => (1, Kind::parse(k).map_or(9, |k| k as u8)),
        Some(("project", _)) => (2, 0),
        Some(("job", _)) => (3, 0),
        _ => (4, 0),
    }
}

// ---------------------------------------------------------------------------------------------
// The library on disk.

/// The library: its folder, catalogue, and the stack switched on for each use.
#[derive(Debug)]
pub struct Library {
    pub dir: PathBuf,
    pub cartridges: Vec<Cartridge>,
    pub stacks: Vec<Stack>,
    /// Use to stack key, as switched; `off` switches a use off.
    pub active: BTreeMap<String, String>,
}

/// A cartridge's files as (name, bytes).
pub type Files = Vec<(String, Vec<u8>)>;

/// Writes `bytes` to `path` through a temporary file, read back and compared, then renamed.
fn write_checked(path: &Path, bytes: &[u8]) -> Result<(), LibFault> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(io)?;
    }
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp).map_err(io)?;
        f.write_all(bytes).map_err(io)?;
        f.sync_all().map_err(io)?;
    }
    if std::fs::read(&tmp).map_err(io)? != bytes {
        let _ = std::fs::remove_file(&tmp);
        return Err(LibFault::Io(format!("{}: read-back differs", tmp.display())));
    }
    std::fs::rename(&tmp, path).map_err(io)
}

fn valid_id(id: &str) -> bool {
    !id.is_empty() && !id.starts_with('.') && id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// A stack loaded for use: its filters in order, each with its key and whether it is on trial.
#[derive(Clone, Debug, PartialEq)]
pub struct Loaded {
    pub stack: String,
    pub use_for: String,
    pub filters: Vec<(String, Filter, bool)>,
    /// Cartridges refused at load, with the named reason.
    pub refused: Vec<(String, String)>,
}

impl Library {
    /// Opens the library in `<home>/library/` (an absent catalogue is an empty library).
    /// # Errors
    /// A catalogue that exists but does not parse.
    pub fn open(home: &Path) -> Result<Self, LibFault> {
        tidy(home);
        let dir = home.join("library");
        let mut lib = Self { dir, cartridges: Vec::new(), stacks: Vec::new(), active: BTreeMap::new() };
        let file = catalogue_file(&lib.dir);
        let Ok(bytes) = std::fs::read(&file) else { return Ok(lib) };
        let v = nmjson::parse(&bytes).map_err(|e| LibFault::Schema(format!("{}: {e}", file.display())))?;
        if text(&v, "schema") != SCHEMA {
            return Err(LibFault::Schema(format!("{}: schema is not {SCHEMA}", file.display())));
        }
        // A catalogue written before the name was settled lists its cartridges under `books`.
        // Adapters are listed in their own section; an entry in the other section than its label's
        // family says is kept and refused by name when loaded.
        let base = lib.dir.clone();
        let main = v.get("cartridges").or_else(|_| v.get("books")).and_then(Value::array).unwrap_or_default();
        let adapters = v.get("adapters").and_then(Value::array).unwrap_or_default();
        lib.cartridges = main
            .iter()
            .map(|e| (e, false))
            .chain(adapters.iter().map(|e| (e, true)))
            .map(|(e, in_adapters)| {
                let mut c = Self::entry(&base, e)?;
                if (c.family == ADAPTER) != in_adapters {
                    let why = format!("{} is listed in the {} section, its label says family {:?}", c.key(), if in_adapters { "adapters" } else { "cartridges" }, c.family);
                    c.notes = obj(vec![("misplaced", s(&why))]);
                }
                Ok(c)
            })
            .collect::<Result<_, LibFault>>()?;
        lib.stacks = v.get("stacks").and_then(Value::array).unwrap_or_default().iter().map(Stack::from_value).collect::<Result<_, _>>()?;
        lib.active = v.get("active").and_then(Value::object).map(|o| o.iter().filter_map(|(k, v)| v.string().ok().map(|x| (k.clone(), x.to_owned()))).collect()).unwrap_or_default();
        Ok(lib)
    }
    /// One catalogue entry: a shelf entry whose file's label says the rest, or an earlier full entry.
    fn entry(base: &Path, e: &Value) -> Result<Cartridge, LibFault> {
        {
            match e.get("file").and_then(Value::string) {
                // A shelf entry: the label inside the file says the rest (an unreadable file leaves
                // the entry with its name only, and loading it is refused by name).
                Ok(f) => {
                    let rec = FileRec { name: f.to_owned(), sha256: text(e, "sha256"), bytes: e.get("bytes").and_then(Value::integer).unwrap_or(0) };
                    // Only the file's head and label are read here; its parameters are hashed when
                    // it is seated or loaded (`tensor_parts_of`, `Library::filter`).
                    let label = if f.contains('/') {
                        label_of(&base.join(shelf_folder("")).join(f)).ok()
                    } else {
                        [shelf_folder(""), shelf_folder(PREFERENCE), shelf_folder(ADAPTER)].iter().find_map(|d| label_of(&base.join(d).join(f)).ok())
                    };
                    match label {
                        Some(l) => {
                            let mut c = Cartridge::from_label(&l, rec, flag(e, "trial"))?;
                            // The catalogue names it (id and version) where it was renamed after its
                            // file was written (the file, and so its hash, is left as it is); the
                            // label's own name is kept in its notes, with a parent the row records.
                            let (id, version) = (text(e, "id"), e.get("version").and_then(Value::integer).ok());
                            let label_key = c.key();
                            if !id.is_empty() && version.is_some() && (id != c.id || version != Some(c.version)) {
                                c.id = id;
                                c.version = version.unwrap_or(c.version);
                                note(&mut c.notes, "label_key", &label_key, true);
                            }
                            for k in RECORD {
                                if let Ok(p) = e.get(k).and_then(Value::string) {
                                    note(&mut c.notes, k, p, false);
                                }
                            }
                            Ok(c)
                        }
                        None => Ok(Cartridge { id: text(e, "id"), version: e.get("version").and_then(Value::integer).unwrap_or(0), family: String::new(), domain: String::new(), scope: String::new(), model: text(e, "model"), weights: String::new(), sockets: String::new(), layers: None, kind: text(e, "kind"), target: String::new(), name: String::new(), description: String::new(), made_from: String::new(), made: String::new(), effect: String::new(), hash: content_hash(std::slice::from_ref(&rec)), files: vec![rec], evidence: Vec::new(), notes: Value::Null, trial: flag(e, "trial") }),
                    }
                }
                Err(_) => Cartridge::from_value(e),
            }
        }
    }
    /// Refuses, by name, a cartridge listed in the section its family does not belong to, or of a
    /// family its kind may not have (`shelf_rule`).
    /// # Errors
    /// The reason.
    pub fn placed(&self, c: &Cartridge) -> Result<(), LibFault> {
        if let Ok(why) = c.notes.get("misplaced").and_then(Value::string) {
            return Err(LibFault::Conflict(why.to_owned()));
        }
        shelf_rule(&c.key(), &c.family, &c.kind)
    }
    #[must_use]
    pub fn exists(&self) -> bool {
        catalogue_file(&self.dir).is_file()
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let (adapters, base): (Vec<&Cartridge>, Vec<&Cartridge>) = self.cartridges.iter().partition(|c| c.family == ADAPTER);
        obj(vec![
            ("schema", s(SCHEMA)),
            ("cartridges", Value::Array(base.into_iter().map(Cartridge::to_value).collect())),
            ("adapters", Value::Array(adapters.into_iter().map(Cartridge::to_value).collect())),
            ("stacks", Value::Array(self.stacks.iter().map(Stack::to_value).collect())),
            ("active", Value::Object(self.active.iter().map(|(k, v)| (k.clone(), s(v))).collect())),
            ("note", s("The cartridge library: learned parameters used as filters, stacked for a specific use. Written by the Neuromorph Nano; files are checked against these hashes on every load.")),
        ])
    }
    /// Writes the shelf (each cartridge's file and hash, the stacks, the uses switched on).
    /// # Errors
    /// The folder cannot be written.
    pub fn save(&self) -> Result<(), LibFault> {
        let mut v = self.to_value();
        if let Value::Object(o) = &mut v {
            let (adapters, base): (Vec<&Cartridge>, Vec<&Cartridge>) = self.cartridges.iter().partition(|c| c.family == ADAPTER);
            o.insert("cartridges".into(), Value::Array(base.into_iter().map(Cartridge::shelf_value).collect()));
            o.insert("adapters".into(), Value::Array(adapters.into_iter().map(Cartridge::shelf_value).collect()));
        }
        write_checked(&catalogue_file(&self.dir), nmjson::pretty::pretty(&v).as_bytes())
    }
    /// A cartridge's folder: `cartridges/<id>-v<version>/` (a folder written before the name was
    /// settled, `books/<id>-v<version>/`, is read where it is).
    #[must_use]
    pub fn cartridge_dir(&self, c: &Cartridge) -> PathBuf {
        if c.one_file() {
            // A file named by a path is in the layout by kind under `cartridges/`.
            let by_kind = c.files.first().is_some_and(|f| f.name.contains('/'));
            return self.dir.join(if by_kind { shelf_folder("") } else { shelf_folder(&c.family) });
        }
        let name = format!("{}-v{}", c.id, c.version);
        let old = self.dir.join("books").join(&name);
        let now = self.dir.join("cartridges").join(&name);
        if !now.exists() && old.is_dir() { old } else { now }
    }
    /// A cartridge by `id@version`, or by id (its newest version).
    #[must_use]
    pub fn cartridge(&self, name: &str) -> Option<&Cartridge> {
        match name.split_once('@') {
            Some((id, v)) => self.cartridges.iter().find(|e| e.id == id && v.parse() == Ok(e.version)),
            None => self.cartridges.iter().filter(|e| e.id == name).max_by_key(|e| e.version),
        }
    }
    /// A stack by `id@version`, or by id (its newest version).
    #[must_use]
    pub fn stack(&self, name: &str) -> Option<&Stack> {
        match name.split_once('@') {
            Some((id, v)) => self.stacks.iter().find(|e| e.id == id && v.parse() == Ok(e.version)),
            None => self.stacks.iter().filter(|e| e.id == name).max_by_key(|e| e.version),
        }
    }
    /// Adapter cartridges registered for base-model specialization on `model` (or all models if empty).
    #[must_use]
    pub fn base_specialization_adapters(&self, model: &str) -> Vec<&Cartridge> {
        self.cartridges
            .iter()
            .filter(|c| c.family == ADAPTER && c.scope == BASE_SPECIALIZATION_SCOPE && (model.is_empty() || c.model == model))
            .collect()
    }
    /// Adapter cartridges registered for per-project specialization on `model` (or all models if empty).
    #[must_use]
    pub fn project_adapters(&self, model: &str) -> Vec<&Cartridge> {
        self.cartridges
            .iter()
            .filter(|c| c.family == ADAPTER && (c.scope.is_empty() || c.scope == PROJECT_SCOPE) && (model.is_empty() || c.model == model))
            .collect()
    }

    /// Adds a cartridge holding `filter` as one file, written once, with `notes` on its label (what
    /// it was fitted from, how it measured). Returns its key.
    /// # Errors
    /// A bad id or family, an id and version already present, or a write failure.
    pub fn add_cartridge(&mut self, spec: &CartridgeSpec, filter: &Filter, notes: Value, evidence: Vec<EvidenceRow>, trial: bool) -> Result<String, LibFault> {
        if !valid_id(&spec.id) {
            return Err(LibFault::Schema(format!("{} is not a valid id (letters, digits, - _ .)", spec.id)));
        }
        if spec.name.trim().is_empty() || spec.description.trim().is_empty() || spec.made_from.trim().is_empty() {
            return Err(LibFault::Schema(format!("{}: a cartridge needs a label (a name, what it does, what it was made from)", spec.id)));
        }
        let key = format!("{}@{}", spec.id, spec.version);
        if self.cartridge(&key).is_some() {
            return Err(LibFault::Conflict(format!("{key} is already in the library")));
        }
        shelf_rule(&key, &spec.family, filter.kind())?;
        let (parts, body) = filter.encode();
        let label = obj(vec![
            ("schema", s(CARTRIDGE_SCHEMA)),
            ("id", s(&spec.id)),
            ("version", n(spec.version)),
            ("family", s(&spec.family)),
            ("domain", s(&spec.domain)),
            ("scope", s(&spec.scope)),
            ("model", s(&spec.model)),
            ("weights", s(&spec.weights)),
            ("sockets", s(if filter.kind() == "affine" { SOCKETS } else { "" })),
            ("layers", spec.layers.map_or(Value::Null, |(a, b)| Value::Array(vec![n(a), n(b)]))),
            ("name", s(&spec.name)),
            ("does", s(&spec.description)),
            ("made_from", s(&spec.made_from)),
            ("made", s(&crate::cartridge::date_of(nmtype::now_ms()))),
            ("effect", s(&spec.effect)),
            ("parameters_bytes", n(body.len() as u64)),
            ("parts", Value::Array(parts)),
            ("evidence", Value::Array(evidence.iter().map(EvidenceRow::to_value).collect())),
            ("notes", notes),
            ("parameters_sha256", s(&nmhash::sha256_bytes(&body))),
        ]);
        let bytes = cartridge_file(&label, &body);
        let name = shelf_path(&spec.family, filter.kind(), &spec.domain, &spec.model, &spec.id, spec.version);
        let path = self.dir.join(shelf_folder("")).join(&name);
        if path.exists() {
            return Err(LibFault::Conflict(format!("{} exists: cartridge files are never written over", path.display())));
        }
        write_checked(&path, &bytes)?;
        let c = Cartridge::from_label(&label, FileRec { name, sha256: nmhash::sha256_bytes(&bytes), bytes: bytes.len() as u64 }, trial)?;
        self.cartridges.push(c);
        Ok(key)
    }

    /// Single-copy base-cartridge replacement (Luis Jake Gabriel III, 2026-11-21 03:49 UTC):
    /// encodes the replacement base cartridge (`Filter::Tensors`), removes any older local copy of
    /// `spec.id` under `base/<role>/` to keep strictly one active file per base type on disk
    /// (Hugging Face commit history holds the rollback archive), writes and hash-verifies the new
    /// file, and updates matching stacks in the catalogue. Returns `(new_key, removed_paths)`.
    /// # Errors
    /// Non-base filter, invalid schema, or disk write failure.
    pub fn replace_base_cartridge(
        &mut self,
        spec: &CartridgeSpec,
        filter: &Filter,
        notes: Value,
        evidence: Vec<EvidenceRow>,
        trial: bool,
    ) -> Result<(String, Vec<String>), LibFault> {
        self.write_base_cartridge(spec, filter, notes, evidence, trial, false)
    }

    /// Adds a new version of a base cartridge beside the versions already on the shelf: nothing is
    /// removed and no stack is pointed at it; the user chooses what serves (Base versions,
    /// `library.upgrade` keep). 2026-11-26 (Luis Jake Gabriel III: "This should all be part of the
    /// neuromorph nano and integrated fully into the mcp the api and the interface"): a build started
    /// from the interface, MCP or the API never takes away the version serving now.
    /// # Errors
    /// As `replace_base_cartridge`.
    pub fn add_base_cartridge(&mut self, spec: &CartridgeSpec, filter: &Filter, notes: Value, evidence: Vec<EvidenceRow>, trial: bool) -> Result<(String, Vec<String>), LibFault> {
        self.write_base_cartridge(spec, filter, notes, evidence, trial, true)
    }

    #[expect(clippy::too_many_arguments, reason = "the two public entry points pass the same six values and differ only in keep")]
    fn write_base_cartridge(&mut self, spec: &CartridgeSpec, filter: &Filter, notes: Value, evidence: Vec<EvidenceRow>, trial: bool, keep: bool) -> Result<(String, Vec<String>), LibFault> {
        if filter.kind() != "tensors" {
            return Err(LibFault::Schema(format!("{}: replace_base_cartridge is only for base tensor cartridges, got {}", spec.id, filter.kind())));
        }
        if !valid_id(&spec.id) {
            return Err(LibFault::Schema(format!("{} is not a valid id (letters, digits, - _ .)", spec.id)));
        }
        if spec.name.trim().is_empty() || spec.description.trim().is_empty() || spec.made_from.trim().is_empty() {
            return Err(LibFault::Schema(format!("{}: a cartridge needs a label (a name, what it does, what it was made from)", spec.id)));
        }
        let key = format!("{}@{}", spec.id, spec.version);
        shelf_rule(&key, &spec.family, filter.kind())?;
        let (parts, body) = filter.encode();
        let label = obj(vec![
            ("schema", s(CARTRIDGE_SCHEMA)),
            ("id", s(&spec.id)),
            ("version", n(spec.version)),
            ("family", s(&spec.family)),
            ("domain", s(&spec.domain)),
            ("scope", s(&spec.scope)),
            ("model", s(&spec.model)),
            ("weights", s(&spec.weights)),
            ("sockets", s("")),
            ("layers", spec.layers.map_or(Value::Null, |(a, b)| Value::Array(vec![n(a), n(b)]))),
            ("name", s(&spec.name)),
            ("does", s(&spec.description)),
            ("made_from", s(&spec.made_from)),
            ("made", s(&crate::cartridge::date_of(nmtype::now_ms()))),
            ("effect", s(&spec.effect)),
            ("parameters_bytes", n(body.len() as u64)),
            ("parts", Value::Array(parts)),
            ("evidence", Value::Array(evidence.iter().map(EvidenceRow::to_value).collect())),
            ("notes", notes),
            ("parameters_sha256", s(&nmhash::sha256_bytes(&body))),
        ]);
        let bytes = cartridge_file(&label, &body);
        let name = shelf_path(&spec.family, filter.kind(), &spec.domain, &spec.model, &spec.id, spec.version);
        let target = self.dir.join(shelf_folder("")).join(&name);

        // Remove older local copies of this base cartridge id before writing the new file so disk
        // space stays strictly single-copy.
        let mut removed = Vec::new();
        if keep {
            if target.exists() {
                return Err(LibFault::Schema(format!("{key} is already on the shelf; a kept build needs a new version")));
            }
            write_checked(&target, &bytes)?;
            let c = Cartridge::from_label(&label, FileRec { name, sha256: nmhash::sha256_bytes(&bytes), bytes: bytes.len() as u64 }, trial)?;
            self.cartridges.push(c);
            return Ok((key, removed));
        }
        let old_keys: Vec<String> = self.cartridges.iter().filter(|c| c.id == spec.id).map(Cartridge::key).collect();
        for old in self.cartridges.iter().filter(|c| c.id == spec.id) {
            let dir = self.cartridge_dir(old);
            for f in &old.files {
                let p = dir.join(&f.name);
                if p.exists() && std::fs::remove_file(&p).is_ok() {
                    removed.push(f.name.clone());
                }
            }
        }
        if target.exists() && std::fs::remove_file(&target).is_ok() {
            removed.push(name.clone());
        }
        write_checked(&target, &bytes)?;
        let c = Cartridge::from_label(&label, FileRec { name, sha256: nmhash::sha256_bytes(&bytes), bytes: bytes.len() as u64 }, trial)?;
        self.cartridges.retain(|x| x.id != spec.id);
        self.cartridges.push(c);
        for i in 0..self.stacks.len() {
            let touches = self.stacks.get(i).is_some_and(|st| st.cartridges.iter().any(|k| old_keys.contains(k)));
            if !touches {
                continue;
            }
            if let Some(st) = self.stacks.get_mut(i) {
                for k in &mut st.cartridges {
                    if old_keys.contains(k) {
                        k.clone_from(&key);
                    }
                }
            }
            let h = self.stacks.get(i).map(|st| stack_hash(st, &st.cartridges.iter().filter_map(|k| self.cartridge(k)).collect::<Vec<_>>())).unwrap_or_default();
            if let Some(st) = self.stacks.get_mut(i) {
                st.hash = h;
            }
        }
        Ok((key, removed))
    }

    /// Adds a stack over cartridges already in the library. Returns its key.
    /// # Errors
    /// A bad id, a cartridge not in the library, or an id and version already present.
    pub fn add_stack(&mut self, mut st: Stack) -> Result<String, LibFault> {
        if !valid_id(&st.id) {
            return Err(LibFault::Schema(format!("{} is not a valid id", st.id)));
        }
        if self.stack(&st.key()).is_some() {
            return Err(LibFault::Conflict(format!("stack {} is already in the library", st.key())));
        }
        let members: Vec<&Cartridge> = st.cartridges.iter().map(|c| self.cartridge(c).ok_or_else(|| LibFault::Unknown(format!("no cartridge {c}")))).collect::<Result<_, _>>()?;
        if let Some(c) = members.iter().find(|c| (c.family == PREFERENCE) != preference_use(&st.use_for)) {
            return Err(LibFault::Conflict(format!("{}: preference cartridges are plugged only by preference stacks, and only they ({} is for {})", c.key(), st.key(), st.use_for)));
        }
        st.cartridges = members.iter().map(|e| e.key()).collect();
        st.hash = stack_hash(&st, &members);
        let key = st.key();
        self.stacks.push(st);
        Ok(key)
    }

    /// Every file of a cartridge is on disk at the size its record states (a record without a size
    /// asks only that the file is there). Nothing is read: the hashes are checked where the files
    /// are read to be seated or loaded (`read_files`, `tensor_parts_of`).
    #[must_use]
    pub fn files_present(&self, c: &Cartridge) -> bool {
        let dir = self.cartridge_dir(c);
        !c.files.is_empty() && c.files.iter().all(|f| std::fs::metadata(dir.join(&f.name)).is_ok_and(|m| m.is_file() && (f.bytes == 0 || m.len() == f.bytes)))
    }

    /// Reads every file of a cartridge, each checked against its hash.
    /// # Errors
    /// A file missing or not matching its hash, or a content hash that does not match.
    pub fn read_files(&self, c: &Cartridge) -> Result<BTreeMap<String, Vec<u8>>, LibFault> {
        let dir = self.cartridge_dir(c);
        let mut out = BTreeMap::new();
        for f in &c.files {
            let b = std::fs::read(dir.join(&f.name)).map_err(|_| LibFault::FileMissing { cartridge: c.key(), file: f.name.clone() })?;
            if nmhash::sha256_bytes(&b) != f.sha256 {
                return Err(LibFault::FileHash { cartridge: c.key(), file: f.name.clone() });
            }
            out.insert(f.name.clone(), b);
        }
        if content_hash(&c.files) != c.hash {
            return Err(LibFault::ContentHash { cartridge: c.key() });
        }
        Ok(out)
    }
    /// Loads a cartridge's filter, hash-checked.
    /// # Errors
    /// As `read_files`, or a filter that does not parse.
    pub fn filter(&self, c: &Cartridge) -> Result<Filter, LibFault> {
        self.placed(c)?;
        let files = self.read_files(c)?;
        match files.values().next().filter(|_| c.one_file()) {
            Some(b) => {
                let (label, body) = read_cartridge(b)?;
                Filter::decode(label.get("parts").and_then(Value::array).unwrap_or_default(), body)
            }
            None => Err(LibFault::Schema(format!("{}: written before cartridges carried a label in their own file; it is not loaded", c.key()))),
        }
    }
    /// The parameter sets a cartridge carries, as places in its file, for seating it as a job's
    /// base source (`tensor_parts_of`, which checks its parameters against its label).
    /// # Errors
    /// A file missing, or as `tensor_parts_of`.
    pub fn tensor_parts(&self, c: &Cartridge) -> Result<Vec<TensorPart>, LibFault> {
        self.placed(c)?;
        let f = c.files.first().ok_or_else(|| LibFault::Unknown(c.key()))?;
        let file = self.cartridge_dir(c).join(&f.name);
        if !file.exists() {
            return Err(LibFault::FileMissing { cartridge: c.key(), file: f.name.clone() });
        }
        tensor_parts_of(&file)
    }
    /// The filter of a cartridge in the earlier folder form, read so it can be written again as a
    /// labelled cartridge file.
    /// # Errors
    /// As `read_files`, or files that do not parse.
    pub fn earlier_filter(&self, c: &Cartridge) -> Result<Filter, LibFault> {
        Filter::from_folder(&c.kind, &c.target, &self.read_files(c)?)
    }
    /// Names a cartridge `id@version` and moves its file to `path` under `cartridges/` (a rename on
    /// the same drive: the file is not copied or rewritten, so its hash stays); the stacks holding it
    /// name it so and are hashed again, and a use switched to a stack keeps it. A `parent` (the
    /// version it was made over) is recorded on its row. Returns the new key.
    /// # Errors
    /// Unknown cartridge, the new key or path already taken, or the move fails.
    pub fn rename(&mut self, from: &str, id: &str, version: u64, path: &str, parent: Option<&str>) -> Result<String, LibFault> {
        if !valid_id(id) {
            return Err(LibFault::Schema(format!("{id} is not a valid id")));
        }
        let old = self.cartridge(from).cloned().ok_or_else(|| LibFault::Unknown(format!("no cartridge {from}")))?;
        let new_key = format!("{id}@{version}");
        if new_key != old.key() && self.cartridge(&new_key).is_some() {
            return Err(LibFault::Conflict(format!("{new_key} is already in the library")));
        }
        let rec = old.files.first().filter(|_| old.one_file()).cloned().ok_or_else(|| LibFault::Schema(format!("{from}: only a one-file cartridge is renamed")))?;
        let src = self.cartridge_dir(&old).join(&rec.name);
        let dst = self.dir.join(shelf_folder("")).join(path);
        if src != dst {
            if dst.exists() {
                return Err(LibFault::Conflict(format!("{} exists", dst.display())));
            }
            if let Some(d) = dst.parent() {
                std::fs::create_dir_all(d).map_err(io)?;
            }
            std::fs::rename(&src, &dst).map_err(|e| LibFault::Io(format!("{} to {}: {e}", src.display(), dst.display())))?;
        }
        let was = old.key();
        if let Some(c) = self.cartridges.iter_mut().find(|c| c.key() == was) {
            let label_key = c.notes.get("label_key").and_then(Value::string).map_or_else(|_| was.clone(), str::to_owned);
            c.id = id.to_owned();
            c.version = version;
            if let Some(f) = c.files.first_mut() {
                path.clone_into(&mut f.name);
            }
            c.hash = content_hash(&c.files);
            if label_key != new_key {
                note(&mut c.notes, "label_key", &label_key, true);
            }
            if let Some(p) = parent {
                note(&mut c.notes, "parent", p, true);
            }
        }
        for i in 0..self.stacks.len() {
            if !self.stacks.get(i).is_some_and(|st| st.cartridges.contains(&was)) {
                continue;
            }
            if let Some(st) = self.stacks.get_mut(i) {
                for k in &mut st.cartridges {
                    if *k == was {
                        k.clone_from(&new_key);
                    }
                }
            }
            let h = self.stacks.get(i).map(|st| stack_hash(st, &st.cartridges.iter().filter_map(|k| self.cartridge(k)).collect::<Vec<_>>())).unwrap_or_default();
            if let Some(st) = self.stacks.get_mut(i) {
                st.hash = h;
            }
        }
        Ok(new_key)
    }

    /// Takes a cartridge off the shelf listing (its file stays where it is). Returns whether it was
    /// listed.
    pub fn unlist(&mut self, key: &str) -> bool {
        let before = self.cartridges.len();
        self.cartridges.retain(|c| c.key() != key);
        before != self.cartridges.len()
    }
    /// Loads a stack: each cartridge's filter in order, hash-checked; a cartridge that fails is
    /// refused (named) and left out. A cartridge is on trial when it or its stack is.
    /// # Errors
    /// Unknown stack.
    pub fn load(&self, name: &str) -> Result<Loaded, LibFault> {
        let st = self.stack(name).ok_or_else(|| LibFault::Unknown(format!("no stack {name}")))?;
        let mut l = Loaded { stack: st.key(), use_for: st.use_for.clone(), filters: Vec::new(), refused: Vec::new() };
        for k in &st.cartridges {
            // A base cartridge's parameters are read and checked when it is seated; here only its
            // file's presence at its recorded size, and an empty set marks its kind.
            let base = |c: &Cartridge| -> Result<Filter, LibFault> {
                self.placed(c)?;
                if self.files_present(c) { Ok(Filter::Tensors(Vec::new())) } else { Err(LibFault::FileMissing { cartridge: c.key(), file: c.files.first().map(|f| f.name.clone()).unwrap_or_default() }) }
            };
            match self.cartridge(k).ok_or_else(|| LibFault::Unknown(format!("no cartridge {k}"))).and_then(|c| if c.kind == "tensors" { base(c) } else { self.filter(c) }.map(|f| (c, f))) {
                Ok((c, f)) => l.filters.push((c.key(), f, c.trial || st.trial)),
                Err(e) => l.refused.push((k.clone(), e.to_string())),
            }
        }
        Ok(l)
    }

    /// Switches `stack` on for `use_for` (`None` switches the use off). Returns the previous
    /// value, or none when nothing changed.
    /// # Errors
    /// Unknown stack, or a stack for another use.
    pub fn switch_use(&mut self, use_for: &str, stack: Option<&str>) -> Result<Option<Option<String>>, LibFault> {
        let want = match stack {
            Some(name) => {
                let st = self.stack(name).ok_or_else(|| LibFault::Unknown(format!("no stack {name}")))?;
                if st.use_for != use_for {
                    return Err(LibFault::Conflict(format!("stack {} is for {}, not {use_for}", st.key(), st.use_for)));
                }
                st.key()
            }
            None => "off".to_owned(),
        };
        let before = self.active.get(use_for).cloned();
        if before.as_deref() == Some(want.as_str()) {
            return Ok(None);
        }
        self.active.insert(use_for.to_owned(), want);
        Ok(Some(before))
    }
    /// Puts a stack (and its cartridges) on trial or takes it off. Returns whether it changed.
    /// # Errors
    /// Unknown stack.
    pub fn set_trial(&mut self, name: &str, trial: bool) -> Result<bool, LibFault> {
        let key = self.stack(name).map(Stack::key).ok_or_else(|| LibFault::Unknown(format!("no stack {name}")))?;
        let mut changed = false;
        let members = self.stack(&key).map(|s| s.cartridges.clone()).unwrap_or_default();
        if let Some(st) = self.stacks.iter_mut().find(|s| s.key() == key) {
            changed |= st.trial != trial;
            st.trial = trial;
        }
        for c in self.cartridges.iter_mut().filter(|c| members.contains(&c.key())) {
            changed |= c.trial != trial;
            c.trial = trial;
        }
        Ok(changed)
    }

    /// Exports a stack and its cartridges as one file: canonical JSON with every file's bytes
    /// (hex), the evidence, and a hash over the whole body. Trial flags are not exported.
    /// # Errors
    /// Unknown stack, or a cartridge that fails its hash.
    pub fn export_stack(&self, name: &str) -> Result<Vec<u8>, LibFault> {
        let st = self.stack(name).ok_or_else(|| LibFault::Unknown(format!("no stack {name}")))?;
        let mut carts = Vec::new();
        for k in &st.cartridges {
            let c = self.cartridge(k).ok_or_else(|| LibFault::Unknown(format!("no cartridge {k}")))?;
            let files = self.read_files(c)?;
            let data = files.iter().map(|(name, b)| obj(vec![("name", s(name)), ("hex", s(&hex(b)))])).collect();
            carts.push(obj(vec![("cartridge", obj(c.fixed())), ("data", Value::Array(data))]));
        }
        let body = obj(vec![("schema", s(EXPORT_SCHEMA)), ("stack", obj(st.fixed())), ("cartridges", Value::Array(carts))]);
        let hash = nmhash::sha256_bytes(body.encode().map_err(io)?.as_bytes());
        let mut out = obj(vec![("body", body), ("hash", s(&hash))]).encode().map_err(io)?;
        out.push('\n');
        Ok(out.into_bytes())
    }

    /// Imports a stack file. Every check passes before anything is written; the stack and its new
    /// cartridges arrive on trial; a cartridge already present with the same content is reused.
    /// Returns (stack key, cartridges added).
    /// # Errors
    /// The first named fault: schema, file hash, content hash, stack hash, whole-file hash, or a
    /// conflict with what the library holds.
    pub fn import(&mut self, bytes: &[u8]) -> Result<(String, Vec<String>), LibFault> {
        let v = nmjson::parse(bytes).map_err(|e| LibFault::Schema(format!("not a stack file: {e}")))?;
        let body = v.get("body").map_err(|_| LibFault::Schema("not a stack file: no body".into()))?;
        if text(body, "schema") != EXPORT_SCHEMA {
            return Err(LibFault::Schema(format!("not a stack file: schema is not {EXPORT_SCHEMA}")));
        }
        let mut entries: Vec<(Cartridge, Files)> = Vec::new();
        // A file exported before the name was settled holds `books`, each with its `book`.
        for item in body.get("cartridges").or_else(|_| body.get("books")).and_then(Value::array).map_err(|e| LibFault::Schema(format!("cartridges: {e}")))? {
            let mut c = Cartridge::from_value(item.get("cartridge").or_else(|_| item.get("book")).map_err(|x| LibFault::Schema(x.to_string()))?)?;
            let mut files = Vec::new();
            for f in item.get("data").and_then(Value::array).map_err(|x| LibFault::Schema(format!("data: {x}")))? {
                let name = need(f, "name")?;
                let rec = c.files.iter().find(|r| r.name == name).ok_or_else(|| LibFault::Schema(format!("{}: {name} is not in its file list", c.key())))?;
                let b = unhex(&need(f, "hex")?).filter(|b| nmhash::sha256_bytes(b) == rec.sha256).ok_or_else(|| LibFault::FileHash { cartridge: c.key(), file: name.clone() })?;
                files.push((name, b));
            }
            if files.len() != c.files.len() {
                return Err(LibFault::FileMissing { cartridge: c.key(), file: "(file list)".into() });
            }
            if content_hash(&c.files) != c.hash {
                return Err(LibFault::ContentHash { cartridge: c.key() });
            }
            if c.one_file() {
                let (rec, b) = (c.files.first().cloned().ok_or(LibFault::ExportHash)?, files.first().map(|f| f.1.clone()).unwrap_or_default());
                let (label, _) = read_cartridge(&b)?;
                c = Cartridge::from_label(&label, rec, true)?;
            }
            c.trial = true;
            entries.push((c, files));
        }
        let mut st = Stack::from_value(body.get("stack").map_err(|x| LibFault::Schema(x.to_string()))?)?;
        let members: Vec<&Cartridge> = st.cartridges.iter().map(|k| entries.iter().map(|x| &x.0).find(|e| &e.key() == k).ok_or_else(|| LibFault::Schema(format!("stack cartridge {k} is not in the file")))).collect::<Result<_, _>>()?;
        if stack_hash(&st, &members) != st.hash {
            return Err(LibFault::StackHash { stack: st.key() });
        }
        if nmhash::sha256_bytes(body.encode().map_err(io)?.as_bytes()) != text(&v, "hash") {
            return Err(LibFault::ExportHash);
        }
        if self.stack(&st.key()).is_some() {
            return Err(LibFault::Conflict(format!("stack {} is already in the library", st.key())));
        }
        if let Some((c, _)) = entries.iter().find(|(c, _)| self.cartridge(&c.key()).is_some_and(|h| h.hash != c.hash)) {
            return Err(LibFault::Conflict(format!("cartridge {} is in the library with other content", c.key())));
        }
        let mut added = Vec::new();
        for (c, files) in entries {
            if self.cartridge(&c.key()).is_some() {
                continue;
            }
            let dir = self.cartridge_dir(&c);
            for (name, b) in &files {
                write_checked(&dir.join(name), b)?;
            }
            added.push(c.key());
            self.cartridges.push(c);
        }
        st.trial = true;
        let key = st.key();
        self.stacks.push(st);
        Ok((key, added))
    }

    /// Verifies that the cartridges in `st` respect the settled stacking hierarchy
    /// (`base` -> `hardware` -> `adapter` -> `learned` -> `preference`), i.e., non-decreasing
    /// `family_rank`.
    /// # Errors
    /// Names the first pair of cartridges that violates the stacking hierarchy.
    pub fn validate_stack_hierarchy(&self, st: &Stack) -> Result<(), String> {
        let mut prev: Option<(u8, String, String)> = None;
        for key in &st.cartridges {
            let Some(c) = self.cartridge(key) else { continue };
            let r = family_rank(&c.family);
            if let Some((pr, pkey, pfam)) = &prev
                && r < *pr
            {
                return Err(format!(
                    "stack {} violates hierarchy (base -> hardware -> adapter -> learned -> preference): {} ({}) cannot come after {} ({})",
                    st.key(),
                    c.key(),
                    c.family,
                    pkey,
                    pfam
                ));
            }
            prev = Some((r, c.key(), c.family.clone()));
        }
        Ok(())
    }

    /// The stack for each use on `machine`, with why: a use switched by hand gets its stack; a
    /// hardware use not switched gets the stack whose piece identity equals the machine's piece,
    /// else the nearest stack of the same family, on trial; other uses not switched get nothing.
    #[must_use]
    pub fn choose(&self, machine: &Machine) -> Vec<Choice> {
        let mut out = Vec::new();
        let mut uses: Vec<String> = self.active.keys().cloned().collect();
        for p in &machine.pieces {
            let u = format!("hardware:{}", p.kind.name());
            if !uses.contains(&u) {
                uses.push(u);
            }
        }
        uses.sort_by_key(|u| use_rank(u));
        for u in uses {
            let piece = u.strip_prefix("hardware:").and_then(|k| Kind::parse(k).ok()).and_then(|k| machine.piece(k));
            if let Some(k) = self.active.get(&u) {
                let (stack, why) = if k == "off" { (None, "switched off".to_owned()) } else { (Some(k.clone()), "switched on for this use".to_owned()) };
                out.push(Choice { use_for: u, piece: piece.map(Piece::identity), stack, why, trial: false });
                continue;
            }
            let Some(piece) = piece else { continue };
            let cands: Vec<&Stack> = self.stacks.iter().filter(|s| s.use_for == u && s.piece.is_some()).collect();
            let exact = cands.iter().filter(|s| s.piece.as_ref().is_some_and(|p| p.identity() == piece.identity())).max_by_key(|s| s.version);
            let near = cands
                .iter()
                .filter_map(|s| s.piece.as_ref().filter(|p| p.family_key() == piece.family_key()).map(|p| (s, piece.distance(p))))
                .min_by(|a, b| a.1.total_cmp(&b.1));
            let c = match (exact, near) {
                (Some(s), _) => Choice { use_for: u, piece: Some(piece.identity()), stack: Some(s.key()), why: format!("stack for this exact piece, {}", piece.identity()), trial: false },
                (None, Some((s, _))) => Choice {
                    use_for: u,
                    piece: Some(piece.identity()),
                    stack: Some(s.key()),
                    why: format!("no stack for {}; nearest of family {}: the stack for {}, on trial until measured here", piece.identity(), piece.family_key(), s.piece.as_ref().map(Piece::identity).unwrap_or_default()),
                    trial: true,
                },
                (None, None) => Choice { use_for: u, piece: Some(piece.identity()), stack: None, why: format!("no stack for {} or its family", piece.identity()), trial: false },
            };
            out.push(c);
        }
        out
    }
}

/// Which stack serves a use, and why.
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    pub use_for: String,
    /// The machine's piece, for a hardware use.
    pub piece: Option<String>,
    pub stack: Option<String>,
    pub why: String,
    /// Loaded on trial only (a relative standing in for a piece the library has not seen).
    pub trial: bool,
}

/// Lowercase hex of bytes.
#[must_use]
pub fn hex(b: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(b.len() * 2);
    for x in b {
        let _ = write!(out, "{x:02x}");
    }
    out
}
/// Bytes of hex, or none.
#[must_use]
pub fn unhex(t: &str) -> Option<Vec<u8>> {
    if !t.len().is_multiple_of(2) {
        return None;
    }
    (0..t.len()).step_by(2).map(|i| t.get(i..i + 2).and_then(|x| u8::from_str_radix(x, 16).ok())).collect()
}

/// Self-check of `library`: two filters stacked in both orders give their defined results, the
/// base is unchanged, a tampered file is refused, export and import round trip.
pub(crate) fn check() -> Result<(), String> {
    let home = nmtype::check_dir("nmlearn-library").map_err(|e| e.to_string())?;
    let mut lib = Library::open(&home).map_err(|e| e.to_string())?;
    let a = Filter::point("v", vec![2.0, 2.0], vec![1.0, 0.0]);
    let b = Filter::point("v", vec![1.0, 3.0], vec![0.0, -1.0]);
    let spec = |id: &str| CartridgeSpec {
        id: id.into(),
        version: 1,
        family: "fundamentals".into(),
        domain: "test".into(),
        scope: "general".into(),
        model: "m".into(),
        weights: String::new(),
        layers: Some((27, 27)),
        name: format!("test {id}"),
        description: "A test filter.".into(),
        made_from: "numbers written for the check".into(),
        effect: String::new(),
    };
    lib.add_cartridge(&spec("a"), &a, Value::Null, Vec::new(), false).map_err(|e| e.to_string())?;
    lib.add_cartridge(&spec("b"), &b, Value::Null, Vec::new(), false).map_err(|e| e.to_string())?;
    if lib.filter(lib.cartridge("a").ok_or("nmlearn.library: lost a")?).map_err(|e| e.to_string())? != a {
        return Err("nmlearn.library: a cartridge file did not read back its filter".into());
    }
    let base = [1.0_f32, 1.0];
    let ab = apply_numbers("v", &base, &[&a, &b]).map_err(|e| e.to_string())?;
    let ba = apply_numbers("v", &base, &[&b, &a]).map_err(|e| e.to_string())?;
    if ab != [3.0, 5.0] || ba != [3.0, 4.0] || base != [1.0, 1.0] {
        return Err(format!("nmlearn.library: stacked {ab:?} and {ba:?}"));
    }
    // Hierarchy ordering check: base -> hardware -> adapter -> learned -> preference.
    if !(family_rank("base") < family_rank("hardware")
        && family_rank("hardware") < family_rank(ADAPTER)
        && family_rank(ADAPTER) < family_rank("learned")
        && family_rank("learned") < family_rank(PREFERENCE))
    {
        return Err("nmlearn.library: family_rank hierarchy invariant violated".into());
    }
    let st = Stack { id: "s".into(), version: 1, use_for: "job:x".into(), piece: None, cartridges: vec!["a".into(), "b".into()], description: String::new(), hash: String::new(), trial: false };
    lib.validate_stack_hierarchy(&st)?;
    lib.add_stack(st).map_err(|e| e.to_string())?;
    // Single-copy base cartridge replacement check: v2 replaces v1 in place, removing v1 file.
    let base_tensor = Filter::Tensors(vec![Tensor {
        name: "model.layers.0.mlp.down_proj.weight".into(),
        layer: Some(0),
        block: "mlp".into(),
        dtype: "BF16".into(),
        shape: vec![2, 2],
        bytes: vec![0; 8],
    }]);
    let mut bspec = CartridgeSpec {
        id: "reader-base".into(),
        version: 1,
        family: String::new(),
        domain: "job:base".into(),
        scope: "general".into(),
        model: "qwen3-0.6b".into(),
        weights: String::new(),
        layers: Some((0, 0)),
        name: "reader base".into(),
        description: "Test base cartridge.".into(),
        made_from: "test weights".into(),
        effect: String::new(),
    };
    let (k1, _) = lib.replace_base_cartridge(&bspec, &base_tensor, Value::Null, Vec::new(), false).map_err(|e| e.to_string())?;
    bspec.version = 2;
    let (k2, removed) = lib.replace_base_cartridge(&bspec, &base_tensor, Value::Null, Vec::new(), false).map_err(|e| e.to_string())?;
    let base_count = lib.cartridges.iter().filter(|c| c.id == "reader-base").count();
    if k1 != "reader-base@1" || k2 != "reader-base@2" || removed.len() != 1 || base_count != 1 {
        return Err(format!("nmlearn.library: single-copy replace failed ({k1} -> {k2}, removed {removed:?}, count {base_count})"));
    }
    // 2026-11-26: a build kept beside (interface, MCP, API): v3 is added, v2 stays on the shelf and
    // its file stays on disk; adding v3 again is refused.
    bspec.version = 3;
    let (k3, removed3) = lib.add_base_cartridge(&bspec, &base_tensor, Value::Null, Vec::new(), false).map_err(|e| e.to_string())?;
    let kept: Vec<String> = lib.cartridges.iter().filter(|c| c.id == "reader-base").map(Cartridge::key).collect();
    let v2_file = lib.cartridge("reader-base@2").is_some_and(|c| c.files.iter().all(|f| lib.cartridge_dir(c).join(&f.name).is_file()));
    if k3 != "reader-base@3" || !removed3.is_empty() || kept.len() != 2 || !v2_file {
        return Err(format!("nmlearn.library: kept build failed ({k3}, removed {removed3:?}, on the shelf {kept:?}, v2 file kept {v2_file})"));
    }
    if lib.add_base_cartridge(&bspec, &base_tensor, Value::Null, Vec::new(), false).is_ok() {
        return Err("nmlearn.library: the same version was added twice".into());
    }
    lib.save().map_err(|e| e.to_string())?;
    let out = lib.export_stack("s").map_err(|e| e.to_string())?;
    let home2 = nmtype::check_dir("nmlearn-library-b").map_err(|e| e.to_string())?;
    let mut lib2 = Library::open(&home2).map_err(|e| e.to_string())?;
    lib2.import(&out).map_err(|e| e.to_string())?;
    if lib2.export_stack("s").map_err(|e| e.to_string())? != out || !lib2.stack("s").is_some_and(|s| s.trial) {
        return Err("nmlearn.library: round trip not byte-equal or not on trial".into());
    }
    let c = lib.cartridge("a").ok_or("nmlearn.library: lost a")?.clone();
    let f = lib.cartridge_dir(&c).join(&c.files.first().ok_or("nmlearn.library: no file")?.name);
    let mut bytes = std::fs::read(&f).map_err(|e| e.to_string())?;
    if let Some(x) = bytes.last_mut() {
        *x ^= 1;
    }
    std::fs::write(&f, bytes).map_err(|e| e.to_string())?;
    match lib.filter(&c) {
        Err(LibFault::FileHash { .. }) => Ok(()),
        o => Err(format!("nmlearn.library: a tampered cartridge loaded: {o:?}")),
    }
}
