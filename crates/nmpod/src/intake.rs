//! Drone intake, first form (2026-11-10; architecture and directions by Luis Jake Gabriel III,
//! guide entries of 14:15, 14:19 and 14:26 UTC): the drones are the point of intake. Whatever comes
//! in (the request, each piece of code taken in for a tentacle's window; later images and other
//! kinds) enters through a drone first, ahead of the tentacle core, the Globalized Context Stack and the Core Brain.
//!
//! 1. Transduction (`Drone::take_in`): text becomes tokens (the release tokenizer) and token
//!    embedding rows held by reference (the one central table, `EmbeddingRows`; a record keeps token
//!    ids, never a copy of rows; nothing is encoded through layers). The result is a `Record`
//!    wrapped as `Sensory`, tagged with its kind, its source (path and lines, or a message label,
//!    never the message text) and its position in the intake order. Other kinds enter through the
//!    `Encoder` slot (an interface only; no image encoder is built).
//! 2. First sensing (`Probe::sense`): per piece, fast processor-only signals of what stands out
//!    relative to the request: exact identifier matches, split-identifier matches (camelCase,
//!    `snake_case`, kebab-case, rarity weighted), structural hits (definition and signature lines
//!    naming request words) and a token-embedding affinity. Affinity rule: for each request token
//!    the maximum cosine over the piece's distinct tokens, then the rarity-weighted mean over
//!    request tokens (late interaction, the MaxSim rule of `ColBERT`, Khattab and Zaharia 2020).
//!    Max, not mean, per request token: a mean over the piece's tokens is dominated by piece length
//!    and filler tokens, while the max asks whether the piece holds something close to each request
//!    token; rarity weighting keeps filler request tokens from counting.
//! 3. Hand-off (`hand_off`, `window_pieces`): every piece is passed on with its signals and a
//!    combined order (reciprocal-rank fusion of the four signals, k = 60, the usual constant), tagged
//!    sensory. Drones flag and pass on; they never drop a piece (context is never reduced).
//!
//! Off by default (`Mode::Off`); selected with `Mode::parse("drones")`. No weights beyond the
//! tokenizer and the token embedding table; nothing is written to disk.
use nmtype::class::{FromRequest, Origin, Sensory, Sourced, inlets};
use nmtype::{Fault, FaultKind};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, RwLock};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmpod.intake", FaultKind::Execution, m)
}

/// Whether drone intake runs. Off unless selected.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    #[default]
    Off,
    Drones,
}
impl Mode {
    /// `off` or `drones` (also `on`).
    /// # Errors
    /// An unknown word.
    pub fn parse(s: &str) -> Result<Self, Fault> {
        match s {
            "off" => Ok(Self::Off),
            "drones" | "on" => Ok(Self::Drones),
            x => Err(Fault::new("nmpod.intake", FaultKind::Config, format!("intake mode {x}: off or drones"))),
        }
    }
}

/// What kind of input a record is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    Request,
    Code,
    /// Reserved for the encoder slot; no encoder is built in this form.
    Image,
}

/// Where an input came from. A message is named by a label, never by its text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    Message { label: String },
    Lines { path: String, start: usize, end: usize },
    Bytes { label: String },
}

/// Text to token ids (the release tokenizer behind it).
pub trait Tokens: Send + Sync {
    /// # Errors
    /// Text the tokenizer refuses.
    fn tokens(&self, text: &str) -> Result<Vec<i32>, Fault>;
}
impl Tokens for nmtokz::Tokenizer {
    fn tokens(&self, text: &str) -> Result<Vec<i32>, Fault> {
        self.encode(text).map_err(|e| bad(format!("tokenizer: {e}")))
    }
}

/// The token embedding table, held once centrally and read by row.
pub trait EmbeddingRows: Send + Sync {
    fn width(&self) -> usize;
    fn rows(&self) -> usize;
    /// Row `id` as f32 into `out` (length `width`); false for an id outside the table.
    fn row(&self, id: usize, out: &mut [f32]) -> bool;
}

/// Slot for other kinds of input (an image encoder later): raw bytes to a sequence of vectors in
/// the embedding width. Interface only in this form.
pub trait Encoder: Send + Sync {
    fn kind(&self) -> Kind;
    /// # Errors
    /// Input the encoder cannot read.
    fn transduce(&self, bytes: &[u8]) -> Result<Vec<Vec<f32>>, Fault>;
}

/// The release's token embedding table read once into RAM from its safetensors file (binary16
/// bfloat values, converted per row on read). One instance is shared by every drone (`Arc`).
#[derive(Debug)]
pub struct Bf16Table {
    bytes: Vec<u8>,
    rows: usize,
    width: usize,
}
impl Bf16Table {
    /// Open `model.embed_tokens.weight` (or `embed_tokens.weight`) in the weight folder `dir`.
    /// # Errors
    /// A missing tensor, an unexpected type or shape, an unreadable file.
    pub fn open(dir: &Path) -> Result<Arc<Self>, Fault> {
        let index = nmsafe::TensorIndex::open(dir).map_err(|e| bad(format!("{}: {e}", dir.display())))?;
        let t = index
            .tensors
            .get("model.embed_tokens.weight")
            .or_else(|| index.tensors.get("embed_tokens.weight"))
            .ok_or_else(|| bad("no embed_tokens tensor"))?;
        if t.dtype != "BF16" || t.shape.len() != 2 {
            return Err(bad(format!("embed_tokens: type {} shape {:?}", t.dtype, t.shape)));
        }
        let rows = usize::try_from(t.shape.first().copied().unwrap_or(0)).map_err(|e| bad(e.to_string()))?;
        let width = usize::try_from(t.shape.get(1).copied().unwrap_or(0)).map_err(|e| bad(e.to_string()))?;
        let bytes = t.read_range(0, rows * width * 2).map_err(|e| bad(format!("embed_tokens: {e}")))?;
        Ok(Arc::new(Self { bytes, rows, width }))
    }
    /// Open `model.embed_tokens.weight` directly from a BF16 byte range in a standalone base
    /// cartridge file (`file`, starting at `start`), without needing upstream `.safetensors`.
    /// # Errors
    /// Unreadable file or unexpected byte length.
    pub fn open_range(file: &Path, start: u64, rows: usize, width: usize) -> Result<Arc<Self>, Fault> {
        use std::io::{Read as _, Seek as _};
        let want = rows.checked_mul(width).and_then(|n| n.checked_mul(2)).ok_or_else(|| bad("embed_tokens: size overflow"))?;
        let mut f = std::fs::File::open(file).map_err(|e| bad(format!("{}: {e}", file.display())))?;
        f.seek(std::io::SeekFrom::Start(start)).map_err(|e| bad(format!("{}: {e}", file.display())))?;
        let mut bytes = vec![0_u8; want];
        f.read_exact(&mut bytes).map_err(|e| bad(format!("{}: {e}", file.display())))?;
        Ok(Arc::new(Self { bytes, rows, width }))
    }
    /// A table from f32 rows (section checks).
    #[must_use]
    pub fn from_rows(rows: &[Vec<f32>]) -> Arc<Self> {
        let width = rows.first().map_or(0, Vec::len);
        let bytes = rows.iter().flat_map(|r| r.iter().flat_map(|x| u16::try_from(x.to_bits() >> 16).unwrap_or(0).to_le_bytes())).collect();
        Arc::new(Self { bytes, rows: rows.len(), width })
    }
}
impl EmbeddingRows for Bf16Table {
    fn width(&self) -> usize {
        self.width
    }
    fn rows(&self) -> usize {
        self.rows
    }
    fn row(&self, id: usize, out: &mut [f32]) -> bool {
        let Some(src) = self.bytes.get(id * self.width * 2..(id + 1) * self.width * 2) else {
            return false;
        };
        for (o, s) in out.iter_mut().zip(src.as_chunks::<2>().0) {
            *o = f32::from_bits(u32::from(u16::from_le_bytes(*s)) << 16);
        }
        true
    }
}

/// Lower-case identifier parts of `text`: split on anything not alphanumeric, then at camelCase
/// boundaries (`ParMap` to `par`, `map`; `HTTPServer` to `http`, `server`). Each whole word is
/// kept too, lower-cased.
#[must_use]
pub fn split_words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for w in text.split(|c: char| !c.is_alphanumeric()) {
        if w.is_empty() {
            continue;
        }
        let cs: Vec<char> = w.chars().collect();
        let mut part = String::new();
        let mut parts = Vec::new();
        for (i, c) in cs.iter().enumerate() {
            let prev = i.checked_sub(1).and_then(|j| cs.get(j));
            let next = cs.get(i + 1);
            let boundary = match prev {
                Some(p) => {
                    (p.is_lowercase() && c.is_uppercase())
                        || (p.is_uppercase() && c.is_uppercase() && next.is_some_and(|n| n.is_lowercase()))
                }
                None => false,
            };
            if boundary && !part.is_empty() {
                parts.push(std::mem::take(&mut part));
            }
            part.extend(c.to_lowercase());
        }
        if !part.is_empty() {
            parts.push(part);
        }
        if parts.len() > 1 {
            out.push(w.to_lowercase());
        }
        out.extend(parts);
    }
    out
}

/// Definition lines of a piece in a language of the language table other than Rust: lines that
/// start, after modifier words, with a definition head word.
fn def_lines_lang(text: &str) -> Vec<&str> {
    const HEADS: [&str; 15] = [
        "function ", "function* ", "class ", "interface ", "type ", "enum ", "namespace ", "def ", "struct ", "union ", "typedef ", "#define ",
        "const ", "filter ", "@keyframes ",
    ];
    const MODIFIERS: [&str; 16] = [
        "export ", "default ", "async ", "public ", "private ", "protected ", "static ", "abstract ", "declare ", "readonly ", "__global__ ",
        "__device__ ", "__host__ ", "inline ", "extern ", "override ",
    ];
    text.lines()
        .filter(|l| {
            let mut t = l.trim_start();
            while let Some(r) = MODIFIERS.iter().find_map(|m| t.strip_prefix(m)) {
                t = r;
            }
            HEADS.iter().any(|h| t.starts_with(h))
        })
        .collect()
}

/// Definition and signature lines of a piece (code items and Markdown headings); for a piece of
/// a language of the language table other than Rust, that language's definition lines.
fn def_lines_at<'a>(text: &'a str, path: &str) -> Vec<&'a str> {
    match nmlex::lang::lang_of(path) {
        Some(l) if l.id != "rust" => def_lines_lang(text),
        _ => def_lines(text),
    }
}

/// Definition and signature lines of a piece (code items and Markdown headings).
fn def_lines(text: &str) -> Vec<&str> {
    const HEADS: [&str; 15] = [
        "fn ",
        "struct ",
        "enum ",
        "trait ",
        "impl ",
        "impl<",
        "type ",
        "const ",
        "static ",
        "mod ",
        "macro_rules!",
        "def ",
        "class ",
        "union ",
        "#",
    ];
    text.lines()
        .filter(|l| {
            let mut t = l.trim_start();
            for p in ["pub(crate) ", "pub(super) ", "pub ", "async ", "unsafe ", "const fn ", "extern \"C\" "] {
                if let Some(r) = t.strip_prefix(p) {
                    t = r;
                }
            }
            if t.starts_with("#[") || t.starts_with("#!") {
                return false;
            }
            t.starts_with("const fn") || HEADS.iter().any(|h| t.starts_with(h))
        })
        .collect()
}

/// One transduced input: tags, the text, its tokens, and the words the first sensing uses. Rows
/// are not copied: the drone's central table gives the row of any token id.
#[derive(Clone, Debug)]
pub struct Record {
    pub kind: Kind,
    pub source: Source,
    pub position: usize,
    pub text: String,
    pub tokens: Vec<i32>,
    /// Distinct token ids, sorted.
    pub distinct: Vec<i32>,
    pub(crate) words: HashSet<String>,
    pub(crate) def_words: HashSet<String>,
    /// BM25 bag of the piece (2026-11-10): counts of its split words of three or more letters
    /// not in `nmlex::STOP`, over the text and the path, and their total.
    pub(crate) bag: HashMap<String, u32>,
    pub(crate) bag_len: u32,
    /// BM25F bag (2026-11-10, Sourcegraph's form): as `bag`, but a word on a definition line or
    /// in the path counts `BM25F_BOOST` times instead of once, inside the one term count.
    pub(crate) bagf: HashMap<String, u32>,
    pub(crate) bagf_len: u32,
    /// What the evidence families read (`crate::evidence`): the text lower-cased (exact phrase),
    /// the stems of every identifier part in the piece, of those on its definition lines (what it
    /// defines), of those on its other code lines (what it uses), and whether the piece is a test
    /// or holds only notes.
    pub(crate) low: String,
    /// Its definition lines (code items, a document's headings), lower-cased.
    pub(crate) def_low: String,
    pub(crate) stems: HashSet<String>,
    pub(crate) def_stems: HashSet<String>,
    pub(crate) use_stems: HashSet<String>,
    /// Stems of the words of its file's path (`path_stems`): a file named for a thing is where
    /// that thing is defined.
    pub(crate) path_stems: HashSet<String>,
    /// Its file's path, lower-cased.
    pub(crate) path_low: String,
    pub(crate) test: bool,
    pub(crate) notes: bool,
    /// Stems of the comment attached directly above the item this record is (an item's doc
    /// comment describes it: the describes role, lead's design 1.2), and that comment lower-cased;
    /// empty for a piece.
    pub(crate) desc_stems: HashSet<String>,
    pub(crate) desc_low: String,
}

/// True for a comment line in any of the languages the Nano reads (line comments, block comment
/// lines, doc comments); a Rust attribute is code.
fn comment_line(line: &str) -> bool {
    let t = line.trim_start();
    (t.starts_with('#') && !t.starts_with("#[") && !t.starts_with("#!") && !t.starts_with("#define") && !t.starts_with("#include"))
        || ["//", "/*", "*", "--", ";", "<!--", "\"\"\"", "'''"].iter().any(|p| t.starts_with(p))
}

/// Stems of every identifier part of `text`: split on anything not a letter, digit or
/// underscore, then on underscores and case changes, each part stemmed.
fn stems_of(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_')).filter(|w| !w.is_empty()).flat_map(nmlex::name_parts).map(|p| nmlex::stem(&p))
}
/// Stems of the words of a file's path: each folder and the file's name without its extension,
/// split into identifier parts at underscores, dashes, dots and case changes, and at a change
/// between letters and digits, each part stemmed.
#[must_use]
pub fn path_stems(path: &str) -> HashSet<String> {
    let segs: Vec<&str> = path.split(['/', '\\']).filter(|x| !x.is_empty()).collect();
    let last = segs.len().saturating_sub(1);
    let mut out = HashSet::new();
    for (i, seg) in segs.iter().enumerate() {
        let seg = if i == last { seg.rsplit_once('.').map_or(*seg, |(name, _)| name) } else { seg };
        for w in seg.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()) {
            for part in nmlex::name_parts(w) {
                let mut run = String::new();
                let mut digit = None;
                for c in part.chars() {
                    let d = c.is_ascii_digit();
                    if digit.is_some_and(|x| x != d) && !run.is_empty() {
                        out.insert(if digit == Some(true) { std::mem::take(&mut run) } else { nmlex::stem(&std::mem::take(&mut run)) });
                    }
                    digit = Some(d);
                    run.push(c);
                }
                if !run.is_empty() {
                    out.insert(if digit == Some(true) { run } else { nmlex::stem(&run) });
                }
            }
        }
    }
    out
}

/// A request record comes from the request; every other record from the material.
impl Sourced for Record {
    fn origin(&self) -> Origin {
        if self.kind == Kind::Request { Origin::Request } else { Origin::Material }
    }
}
impl Record {
    /// The bytes this record holds in RAM, estimated from its parts (memory areas, 2026-11-27:
    /// the drones' intake counts in the `content` area): each heap block its capacity rounded up
    /// to 16 bytes plus 16 bytes of allocator header, each hash table its buckets (capacity / 7 *
    /// 8) of slot plus control byte, each string kept in a table its own block too.
    #[must_use]
    pub fn held_bytes(&self) -> usize {
        const HEAD: usize = 16;
        let block = |cap: usize| if cap == 0 { 0 } else { cap.div_ceil(16) * 16 + HEAD };
        let string = |s: &String| block(s.capacity());
        let table = |cap: usize, slot: usize| if cap == 0 { 0 } else { block((cap * 8 / 7 + 1).next_power_of_two() * (slot + 1)) };
        let set = |s: &HashSet<String>| table(s.capacity(), 24) + s.iter().map(string).sum::<usize>();
        let bag = |b: &HashMap<String, u32>| table(b.capacity(), 32) + b.keys().map(string).sum::<usize>();
        std::mem::size_of::<Self>()
            + string(&self.text)
            + block(self.tokens.capacity() * 4)
            + block(self.distinct.capacity() * 4)
            + set(&self.words)
            + set(&self.def_words)
            + bag(&self.bag)
            + bag(&self.bagf)
            + string(&self.low)
            + string(&self.def_low)
            + set(&self.stems)
            + set(&self.def_stems)
            + set(&self.use_stems)
            + set(&self.path_stems)
            + string(&self.path_low)
            + set(&self.desc_stems)
            + string(&self.desc_low)
    }
    /// Whether this piece defines a name `other` uses in code: an identifier part on its
    /// definition lines among `other`'s uses.
    #[must_use]
    pub fn defines_used_by(&self, other: &Record) -> bool {
        self.def_stems.iter().any(|s| other.use_stems.contains(s))
    }
    /// Whether a code line of this piece calls `name` (lower-cased, the last part of a path
    /// such as `module::name`): the name standing alone, then `(`, `!(` or `::<`, on a line that
    /// is not a comment and not the name's own definition.
    #[must_use]
    pub fn calls(&self, name: &str) -> bool {
        if name.is_empty() || !self.low.contains(name) {
            return false;
        }
        let word = |c: char| c.is_alphanumeric() || c == '_';
        self.text.lines().filter(|l| !comment_line(l)).any(|line| {
            let l = line.to_lowercase();
            let mut from = 0;
            while let Some(at) = l.get(from..).and_then(|s| s.find(name)).map(|i| i + from) {
                from = at + name.len();
                let before = l.get(..at).unwrap_or("");
                if before.chars().next_back().is_some_and(word) || l.get(from..).and_then(|s| s.chars().next()).is_some_and(word) {
                    continue;
                }
                let head = before.trim_end();
                if head.len() < before.len() && matches!(head.rsplit(|c: char| !word(c)).next(), Some("fn" | "def" | "function")) {
                    continue;
                }
                let after = l.get(from..).unwrap_or("").trim_start();
                if after.starts_with('(') || after.starts_with("!(") || after.starts_with("::<") {
                    return true;
                }
            }
            false
        })
    }
    /// Number of tokens.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Sentinel in `RowCache::slots`: token id not decoded yet.
const SLOT_EMPTY: u32 = u32::MAX;
/// Sentinel in `RowCache::slots`: token id missing from the embedding table or zero-norm.
const SLOT_MISSING: u32 = u32::MAX - 1;

/// Decoded token embedding rows and their exact L2 norms, populated lazily once per seen token id
/// so repeated questions never re-decode BF16/F16 table rows or recompute norms.
struct RowCache {
    width: usize,
    slots: Vec<u32>,
    data: Vec<f32>,
    norms: Vec<f32>,
}
impl RowCache {
    fn new(width: usize) -> Self {
        Self { width, slots: Vec::new(), data: Vec::new(), norms: Vec::new() }
    }
    fn get(&self, id: usize) -> Option<(&[f32], f32)> {
        let slot = *self.slots.get(id)?;
        if slot >= SLOT_MISSING {
            return None;
        }
        let k = slot as usize;
        let off = k.checked_mul(self.width)?;
        let row = self.data.get(off..off + self.width)?;
        let norm = *self.norms.get(k)?;
        Some((row, norm))
    }
}

/// Precomputed cosines between every token id in a set and every query row, indexed directly by
/// token id for O(1) slice lookups with zero hashing inside per-piece loops.
pub(crate) struct TokenCosines {
    nq: usize,
    slots: Vec<u32>,
    data: Vec<f32>,
}
impl TokenCosines {
    #[inline]
    pub(crate) fn get(&self, t: i32) -> Option<&[f32]> {
        let id = usize::try_from(t).ok()?;
        let slot = *self.slots.get(id)?;
        if slot == SLOT_EMPTY {
            return None;
        }
        let off = (slot as usize).checked_mul(self.nq)?;
        self.data.get(off..off + self.nq)
    }
}

/// A drone: the point of intake. Holds the tokenizer and the central table by reference.
#[derive(Clone)]
pub struct Drone {
    pub(crate) tokens: Arc<dyn Tokens>,
    pub(crate) table: Arc<dyn EmbeddingRows>,
    encoders: Vec<Arc<dyn Encoder>>,
    rows: Arc<RwLock<RowCache>>,
}
impl std::fmt::Debug for Drone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Drone")
            .field("width", &self.table.width())
            .field("rows", &self.table.rows())
            .field("encoders", &self.encoders.len())
            .finish()
    }
}
impl Drone {
    #[must_use]
    pub fn new(tokens: Arc<dyn Tokens>, table: Arc<dyn EmbeddingRows>) -> Self {
        let width = table.width();
        Self { tokens, table, encoders: Vec::new(), rows: Arc::new(RwLock::new(RowCache::new(width))) }
    }
    /// Ensure every token id in `ids` has its decoded row and exact L2 norm cached in RAM.
    pub(crate) fn ensure_rows(&self, ids: &[i32]) {
        let width = self.table.width();
        if width == 0 || ids.is_empty() {
            return;
        }
        let missing: Vec<usize> = {
            let Ok(cache) = self.rows.read() else { return };
            let mut m = Vec::new();
            for &t in ids {
                let Ok(id) = usize::try_from(t) else { continue };
                if cache.slots.get(id).copied().unwrap_or(SLOT_EMPTY) == SLOT_EMPTY {
                    m.push(id);
                }
            }
            m
        };
        if missing.is_empty() {
            return;
        }
        let mut missing = missing;
        missing.sort_unstable();
        missing.dedup();
        let cores = nmtype::budget::total();
        let run = missing.len().div_ceil(cores.max(1)).max(1);
        let runs = missing.len().div_ceil(run);
        let table = &self.table;
        let missing_ref = &missing;
        type DecodedRow = (usize, Option<(Vec<f32>, f32)>);
        let decoded: Vec<Vec<DecodedRow>> = nmtype::budget::run(runs, |p| {
            let slice = missing_ref.get(p * run..((p + 1) * run).min(missing_ref.len())).unwrap_or(&[]);
            let mut buf = vec![0.0_f32; width];
            let mut out = Vec::with_capacity(slice.len());
            for &id in slice {
                if table.row(id, &mut buf) {
                    let n = buf.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if n > 0.0 {
                        out.push((id, Some((buf.clone(), n))));
                    } else {
                        out.push((id, None));
                    }
                } else {
                    out.push((id, None));
                }
            }
            out
        })
        .into_iter()
        .flatten()
        .collect();
        let Ok(mut cache) = self.rows.write() else { return };
        if let Some(&max_id) = missing.last()
            && max_id >= cache.slots.len()
        {
            cache.slots.resize(max_id + 1, SLOT_EMPTY);
        }
        for chunk in decoded {
            for (id, row) in chunk {
                if cache.slots.get(id).copied().unwrap_or(SLOT_EMPTY) != SLOT_EMPTY {
                    continue;
                }
                match row {
                    Some((v, n)) => {
                        if let Ok(slot) = u32::try_from(cache.norms.len())
                            && slot < SLOT_MISSING
                        {
                            cache.data.extend_from_slice(&v);
                            cache.norms.push(n);
                            if let Some(s) = cache.slots.get_mut(id) {
                                *s = slot;
                            }
                        }
                    }
                    None => {
                        if let Some(s) = cache.slots.get_mut(id) {
                            *s = SLOT_MISSING;
                        }
                    }
                }
            }
        }
    }
    /// Compute cosines of every token id in `ids` against every `width`-wide query row in `queries`
    /// once across processor cores, reusing cached decoded rows and returning an O(1) lookup table.
    /// # Errors
    /// A worker partition failed to complete.
    pub(crate) fn cosines_for(&self, ids: &[i32], queries: &[f32], missing_fill: f32) -> Result<TokenCosines, Fault> {
        let width = self.table.width();
        let nq = queries.len() / width.max(1);
        if ids.is_empty() || nq == 0 || width == 0 {
            return Ok(TokenCosines { nq, slots: Vec::new(), data: Vec::new() });
        }
        self.ensure_rows(ids);
        let cache = self.rows.read().map_err(|_| bad("row cache lock poisoned"))?;
        let cores = nmtype::budget::total();
        let run = ids.len().div_ceil(cores.max(1)).max(1);
        let runs = ids.len().div_ceil(run);
        let cache_ref: &RowCache = &cache;
        let parts = nmtype::budget::run(runs, |p| {
            let slice = ids.get(p * run..((p + 1) * run).min(ids.len())).unwrap_or(&[]);
            let mut out = Vec::with_capacity(slice.len().saturating_mul(nq));
            for &t in slice {
                if let Some((row, n)) = usize::try_from(t).ok().and_then(|id| cache_ref.get(id)) {
                    for q in queries.chunks_exact(width) {
                        out.push(q.iter().zip(row).map(|(a, b)| a * b).sum::<f32>() / n);
                    }
                } else {
                    out.resize(out.len() + nq, missing_fill);
                }
            }
            out
        });
        let mut data = Vec::with_capacity(ids.len().saturating_mul(nq));
        let mut completed = 0_usize;
        for part in parts {
            let Some(v) = part else {
                return Err(bad("drone sensing: a run was lost"));
            };
            data.extend_from_slice(&v);
            completed += 1;
        }
        if completed != runs || data.len() != ids.len().saturating_mul(nq) {
            return Err(bad("drone sensing: a run was lost"));
        }
        let max_id = ids.iter().filter_map(|&t| usize::try_from(t).ok()).max().unwrap_or(0);
        let mut slots = vec![SLOT_EMPTY; max_id.saturating_add(1)];
        for (k, &t) in ids.iter().enumerate() {
            if let (Ok(id), Ok(slot)) = (usize::try_from(t), u32::try_from(k))
                && let Some(s) = slots.get_mut(id)
            {
                *s = slot;
            }
        }
        Ok(TokenCosines { nq, slots, data })
    }
    /// Attach an encoder for another kind of input (the slot; none is built in this form).
    pub fn attach(&mut self, encoder: Arc<dyn Encoder>) {
        self.encoders.push(encoder);
    }
    /// The encoder for `kind`, when one is attached.
    #[must_use]
    pub fn encoder(&self, kind: &Kind) -> Option<&Arc<dyn Encoder>> {
        self.encoders.iter().find(|e| &e.kind() == kind)
    }
    /// Take one text input in: tokens, words, definition lines, tagged.
    /// # Errors
    /// The tokenizer refuses the text.
    pub fn take_in(&self, kind: Kind, source: Source, position: usize, text: &str) -> Result<Sensory<Record>, Fault> {
        self.take_in_with(kind, source, position, text, "")
    }
    /// `take_in` for a document piece with its heading path: the heading's words count once in
    /// both bags (a heading names what its section is about even where the text never repeats
    /// it). Not boosted: every block of a section repeats its heading, and at `BM25F_BOOST` times
    /// the small blocks of one document filled the drones' shortlists ahead of code (any-data
    /// stage 1, served path). An empty heading (every code piece) gives exactly `take_in`.
    /// # Errors
    /// The tokenizer refuses the text.
    pub fn take_in_with(&self, kind: Kind, source: Source, position: usize, text: &str, heading: &str) -> Result<Sensory<Record>, Fault> {
        self.take_in_described(kind, source, position, text, heading, "")
    }
    /// `take_in_with` for an item of the material (`nmsubs::items`) whose attached comment is
    /// `described` (its lines, as they stand in `text`): the comment's stems describe the item and
    /// count where its definition lines' do (the describes role), and its lines weigh as
    /// definition lines in the BM25F bag. An empty `described` gives exactly `take_in_with`.
    /// # Errors
    /// The tokenizer refuses the text.
    #[expect(clippy::too_many_arguments, reason = "take_in_with's arguments and the attached comment")]
    pub fn take_in_described(&self, kind: Kind, source: Source, position: usize, text: &str, heading: &str, described: &str) -> Result<Sensory<Record>, Fault> {
        let tokens = self.tokens.tokens(text)?;
        let distinct: Vec<i32> = tokens.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();
        let words = split_words(text).into_iter().collect();
        let path = match &source {
            Source::Lines { path, .. } => path.as_str(),
            _ => "",
        };
        let def_words = def_lines_at(text, path).iter().flat_map(|l| split_words(l)).collect();
        let mut bag: HashMap<String, u32> = HashMap::new();
        let mut bag_len = 0_u32;
        for w in bm25_words(&format!("{text} {path}")) {
            *bag.entry(w).or_insert(0) += 1;
            bag_len += 1;
        }
        let mut bagf: HashMap<String, u32> = HashMap::new();
        let mut bagf_len = 0_u32;
        let defs: HashSet<&str> = def_lines_at(text, path).into_iter().collect();
        let desc: HashSet<&str> = described.lines().collect();
        for line in text.lines() {
            let w8 = if defs.contains(line) || desc.contains(line) { BM25F_BOOST } else { 1 };
            for w in bm25_words(line) {
                *bagf.entry(w).or_insert(0) += w8;
                bagf_len += w8;
            }
        }
        for w in bm25_words(path) {
            *bagf.entry(w).or_insert(0) += BM25F_BOOST;
            bagf_len += BM25F_BOOST;
        }
        for w in bm25_words(heading) {
            *bag.entry(w.clone()).or_insert(0) += 1;
            bag_len += 1;
            *bagf.entry(w).or_insert(0) += 1;
            bagf_len += 1;
        }
        let stems: HashSet<String> = stems_of(text).chain(stems_of(heading)).collect();
        let def_stems: HashSet<String> = defs.iter().flat_map(|l| stems_of(l)).collect();
        let use_stems: HashSet<String> =
            text.lines().filter(|l| !defs.contains(l) && !comment_line(l)).flat_map(stems_of).filter(|s| !def_stems.contains(s)).collect();
        let notes = text.lines().filter(|l| !l.trim().is_empty()).all(comment_line);
        let path_stems = path_stems(path);
        let path_low = path.to_lowercase();
        let lowpath = path.to_lowercase();
        let test = lowpath.split(['/', '\\']).any(|p| p == "tests" || p == "test" || p.starts_with("test_") || p.contains("_test"))
            || text.contains("#[test]")
            || text.contains("#[cfg(test)]");
        Ok(Sensory::new(Record {
            kind,
            source,
            position,
            text: text.to_owned(),
            tokens,
            distinct,
            words,
            def_words,
            bag,
            bag_len,
            bagf,
            bagf_len,
            low: text.to_lowercase(),
            def_low: defs.iter().map(|l| l.to_lowercase()).collect::<Vec<_>>().join("
"),
            stems,
            def_stems,
            use_stems,
            path_stems,
            path_low,
            test,
            notes,
            desc_stems: stems_of(described).collect(),
            desc_low: described.to_lowercase(),
        }))
    }
    /// Take the request in through its own drone: a request record, typed as taken from the
    /// request, so it can never be handed on as a piece of the material.
    /// # Errors
    /// The tokenizer refuses the text.
    pub fn take_request(&self, label: &str, text: &str) -> Result<FromRequest<Record>, Fault> {
        let r = self.take_in(Kind::Request, Source::Message { label: label.to_owned() }, 0, text)?.into_inner();
        inlets::REQUEST_DRONE.admit(&r)?;
        Ok(FromRequest::new(r))
    }
    /// `probe` on the request as its own drone took it in (`take_request`).
    #[must_use]
    pub fn probe_request(&self, request: &FromRequest<Record>, rarity: &Rarity) -> Probe {
        self.probe(request.get(), rarity)
    }
    /// Prepare the first sensing of pieces against `request`, with rarity from `rarity`. The
    /// served paths call `probe_request`, which takes only a record from the request's drone.
    #[must_use]
    pub fn probe(&self, request: &Record, rarity: &Rarity) -> Probe {
        let names = nmlex::names(&request.text);
        let split: String = split_words(&request.text).join(" ");
        let mut terms: Vec<(String, f32)> =
            nmlex::key_terms(&split).into_iter().map(|w| (rarity.word(&w), w)).map(|(r, w)| (w, r)).collect();
        terms.retain(|(_, r)| *r > 0.0);
        let bm25: Vec<(String, f32)> = bm25_words(&request.text)
            .into_iter()
            .map(|w| {
                let i = rarity.bm25_idf(&w);
                (w, i)
            })
            .collect();
        let width = self.table.width();
        let mut qv = Vec::new();
        let mut qw = Vec::new();
        let mut buf = vec![0.0f32; width];
        for t in &request.distinct {
            let Ok(id) = usize::try_from(*t) else { continue };
            if !self.table.row(id, &mut buf) {
                continue;
            }
            let n = buf.iter().map(|x| x * x).sum::<f32>().sqrt();
            let w = rarity.token(*t);
            if n > 0.0 && w > 0.0 {
                qv.extend(buf.iter().map(|x| x / n));
                qw.push(w);
            }
        }
        let names_low: Vec<String> = names.iter().map(|n| n.to_lowercase()).collect();
        let terms_total: f32 = terms.iter().map(|t| t.1).sum();
        let wsum: f32 = qw.iter().sum();
        Probe {
            table: Arc::clone(&self.table),
            names,
            names_low,
            terms,
            terms_total,
            bm25,
            avg_len: rarity.avg_len(),
            avg_len_f: rarity.avg_len_f(),
            qv,
            qw,
            wsum,
            width,
            cache: HashMap::new(),
        }
    }
    /// Sense every piece in `records` against `request` in parallel across processor cores,
    /// precomputing token cosines once over `rarity.token_ids()` instead of per worker thread.
    /// # Errors
    /// A worker partition failed to complete.
    pub fn sense_records(&self, request: &FromRequest<Record>, records: &[Record], rarity: &Rarity) -> Result<Vec<Signals>, Fault> {
        let probe = self.probe_request(request, rarity);
        let ids = rarity.token_ids();
        let cos = self.cosines_for(&ids, &probe.qv, -1.0)?;
        let cores = nmtype::budget::total();
        let run = records.len().div_ceil(cores.max(1)).max(1);
        let runs = records.len().div_ceil(run);
        let probe_ref = &probe;
        let cos_ref = &cos;
        let parts = nmtype::budget::run(runs, |p| {
            let slice = records.get(p * run..((p + 1) * run).min(records.len())).unwrap_or(&[]);
            let mut best = vec![-1.0_f32; probe_ref.qw.len()];
            let mut fallback: HashMap<i32, Option<Vec<f32>>> = HashMap::new();
            let mut buf = Vec::new();
            slice.iter().map(|rec| probe_ref.sense_fast(rec, cos_ref, &mut best, &mut fallback, &mut buf)).collect::<Vec<_>>()
        });
        let out: Vec<Signals> = parts.into_iter().flatten().flatten().collect();
        if out.len() != records.len() {
            return Err(bad("drone sensing: a run was lost"));
        }
        Ok(out)
    }
}

/// Rarity of words and tokens over the pieces taken in (inverse document frequency,
/// `ln((n + 1) / (df + 1))`), so common words and filler tokens weigh little.
#[derive(Clone, Debug, Default)]
pub struct Rarity {
    n: usize,
    words: HashMap<String, u32>,
    tokens: HashMap<i32, u32>,
    /// BM25 document frequencies over the pieces' bags, and the bags' total length.
    bag_df: HashMap<String, u32>,
    bag_total: u64,
    bagf_total: u64,
    /// Pieces holding each identifier-part stem (the naming family's rarity).
    stem_df: HashMap<String, u32>,
    /// Pieces holding each stem in their text or in their file's path (what the settle roles read).
    held_df: HashMap<String, u32>,
    /// The material's BM25 words by how many pieces hold them, then by word: where the
    /// content-free pose takes words of like frequency from.
    by_df: Vec<(u32, String)>,
}
impl Rarity {
    #[must_use]
    pub fn from_records<'a>(records: impl IntoIterator<Item = &'a Record>) -> Self {
        let mut r = Self::default();
        // Lane R: with the request's drone in its model form, the material's symbols are indexed
        // here for its tagging (material to request only).
        let mut ix = fine_parts_on().then(SymbolIndex::default);
        // A project's own hidden configuration is scored but not counted: the material's
        // rarities stay those of the files read before it was (`nmtype::in_hidden_folder`).
        for rec in records.into_iter().filter(|rec| !nmtype::in_hidden_folder(&rec.path_low)) {
            if let Some(ix) = ix.as_mut() {
                ix.add_record(rec);
            }
            r.n += 1;
            for w in &rec.words {
                *r.words.entry(w.clone()).or_insert(0) += 1;
            }
            for t in &rec.distinct {
                *r.tokens.entry(*t).or_insert(0) += 1;
            }
            for w in rec.bag.keys() {
                *r.bag_df.entry(w.clone()).or_insert(0) += 1;
            }
            for s in &rec.stems {
                *r.stem_df.entry(s.clone()).or_insert(0) += 1;
                *r.held_df.entry(s.clone()).or_insert(0) += 1;
            }
            for s in rec.path_stems.iter().filter(|s| !rec.stems.contains(*s)) {
                *r.held_df.entry(s.clone()).or_insert(0) += 1;
            }
            r.bag_total += u64::from(rec.bag_len);
            r.bagf_total += u64::from(rec.bagf_len);
        }
        r.by_df = r.bag_df.iter().filter(|(w, _)| w.chars().all(|c| c.is_alphabetic())).map(|(w, d)| (*d, w.clone())).collect();
        r.by_df.sort_unstable();
        if let Some(ix) = ix {
            publish_symbols(&ix);
        }
        r
    }
    /// Every token id the pieces hold, in order.
    #[must_use]
    pub fn token_ids(&self) -> Vec<i32> {
        let mut v: Vec<i32> = self.tokens.keys().copied().collect();
        v.sort_unstable();
        v
    }
    /// Pieces holding word `w` in their BM25 bag.
    #[must_use]
    pub fn bag_df(&self, w: &str) -> u32 {
        self.bag_df.get(w).copied().unwrap_or(0)
    }
    /// Rarity of an identifier-part stem.
    #[must_use]
    pub fn stem(&self, s: &str) -> f32 {
        self.idf(self.stem_df.get(s).copied().unwrap_or(0))
    }
    /// Rarity of a stem over the pieces holding it in their text or their file's path.
    #[must_use]
    pub fn held(&self, s: &str) -> f32 {
        self.idf(self.held_df.get(s).copied().unwrap_or(0))
    }
    /// The material's words of like frequency to a word held by `df` pieces (at least one): the
    /// words held by the nearest count the material has, widened to the nearest counts on both
    /// sides until at least `need` words are there (or the material has no more).
    #[must_use]
    pub fn like(&self, df: u32, need: usize) -> &[(u32, String)] {
        let want = df.max(1);
        let at = self.by_df.partition_point(|(d, _)| *d < want);
        let pick = match (self.by_df.get(at), at.checked_sub(1).and_then(|i| self.by_df.get(i))) {
            (Some((hi, _)), Some((lo, _))) => {
                if hi - want <= want - lo {
                    *hi
                } else {
                    *lo
                }
            }
            (Some((hi, _)), None) => *hi,
            (None, Some((lo, _))) => *lo,
            (None, None) => return &[],
        };
        let mut a = self.by_df.partition_point(|(d, _)| *d < pick);
        let mut b = self.by_df.partition_point(|(d, _)| *d <= pick);
        while b - a < need && (a > 0 || b < self.by_df.len()) {
            // Widen towards the side whose next count is nearer.
            let lo = a.checked_sub(1).and_then(|i| self.by_df.get(i)).map(|x| pick - x.0);
            let hi = self.by_df.get(b).map(|x| x.0 - pick);
            match (lo, hi) {
                (Some(l), Some(h)) if l <= h => a -= 1,
                (Some(_), None) => a -= 1,
                _ => b += 1,
            }
        }
        self.by_df.get(a..b).unwrap_or(&[])
    }
    /// Lucene's BM25 inverse document frequency, `ln(1 + (n - df + 0.5) / (df + 0.5))`.
    #[must_use]
    pub fn bm25_idf(&self, w: &str) -> f32 {
        let n = f64::from(u32::try_from(self.n).unwrap_or(u32::MAX));
        let df = f64::from(self.bag_df.get(w).copied().unwrap_or(0));
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln() as f32
    }
    /// Mean BM25F bag length over the pieces.
    #[must_use]
    pub fn avg_len_f(&self) -> f32 {
        if self.n == 0 { 1.0 } else { (self.bagf_total as f64 / self.n as f64) as f32 }
    }
    /// Mean bag length over the pieces.
    #[must_use]
    pub fn avg_len(&self) -> f32 {
        if self.n == 0 { 1.0 } else { (self.bag_total as f64 / self.n as f64) as f32 }
    }
    fn idf(&self, df: u32) -> f32 {
        let n = f64::from(u32::try_from(self.n).unwrap_or(u32::MAX));
        ((n + 1.0) / (f64::from(df) + 1.0)).ln().max(0.0) as f32
    }
    #[must_use]
    pub fn word(&self, w: &str) -> f32 {
        self.idf(self.words.get(w).copied().unwrap_or(0))
    }
    #[must_use]
    pub fn token(&self, t: i32) -> f32 {
        self.idf(self.tokens.get(&t).copied().unwrap_or(0))
    }
}

/// The first-sensing signals of one piece against the request.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Signals {
    /// Share of the request's identifiers (as written) found verbatim in the piece.
    pub exact: f32,
    /// Rarity-weighted share of the request's split words present in the piece.
    pub ident: f32,
    /// Rarity-weighted share of the request's split words on the piece's definition lines, plus
    /// one when a request identifier is defined there.
    pub structural: f32,
    /// Token-embedding affinity (see the module text).
    pub affinity: f32,
    /// Okapi BM25 of the request's words in the piece (2026-11-10; Robertson and Zaragoza 2009,
    /// the Lucene form: k1 1.2, b 0.75, idf `ln(1 + (n - df + 0.5) / (df + 0.5))`), the same
    /// scoring as the CX1 conventional baseline.
    pub bm25: f32,
    /// BM25F (2026-11-10): BM25 over the BM25F bag (definition-line and path hits boosted inside
    /// the term count; Sourcegraph, "Keeping it boring and relevant with BM25F"; boost 5 as
    /// published there, not tuned here). Same idf (document frequency does not change).
    pub bm25f: f32,
}

/// The request prepared for sensing: its identifiers, weighted words, unit token rows, and a
/// cache of cosines per piece token (RAM only, for this request).
pub struct Probe {
    table: Arc<dyn EmbeddingRows>,
    names: Vec<String>,
    names_low: Vec<String>,
    terms: Vec<(String, f32)>,
    terms_total: f32,
    bm25: Vec<(String, f32)>,
    avg_len: f32,
    avg_len_f: f32,
    qv: Vec<f32>,
    qw: Vec<f32>,
    wsum: f32,
    width: usize,
    cache: HashMap<i32, Vec<f32>>,
}
impl std::fmt::Debug for Probe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Probe")
            .field("names", &self.names.len())
            .field("terms", &self.terms.len())
            .field("tokens", &self.qw.len())
            .finish_non_exhaustive()
    }
}
impl Probe {
    fn cosines(&mut self, t: i32) -> Option<&Vec<f32>> {
        if !self.cache.contains_key(&t) {
            let mut buf = vec![0.0f32; self.width];
            let id = usize::try_from(t).ok()?;
            if !self.table.row(id, &mut buf) {
                return None;
            }
            let n = buf.iter().map(|x| x * x).sum::<f32>().sqrt();
            let c: Vec<f32> = self
                .qv
                .chunks_exact(self.width)
                .map(|q| if n > 0.0 { q.iter().zip(&buf).map(|(a, b)| a * b).sum::<f32>() / n } else { 0.0 })
                .collect();
            self.cache.insert(t, c);
        }
        self.cache.get(&t)
    }
    /// Sense one piece using precomputed token cosines and caller-owned scratch buffers.
    fn sense_fast(
        &self,
        piece: &Record,
        cos: &TokenCosines,
        best: &mut [f32],
        fallback: &mut HashMap<i32, Option<Vec<f32>>>,
        buf: &mut Vec<f32>,
    ) -> Signals {
        let exact = if self.names.is_empty() {
            0.0
        } else {
            u16::try_from(self.names.iter().filter(|n| piece.text.contains(n.as_str())).count()).map_or(0.0, f32::from)
                / u16::try_from(self.names.len()).map_or(1.0, f32::from)
        };
        let total = self.terms_total;
        let share = |set: &HashSet<String>| {
            if total > 0.0 { self.terms.iter().filter(|(w, _)| set.contains(w)).map(|t| t.1).sum::<f32>() / total } else { 0.0 }
        };
        let ident = share(&piece.words);
        let defined = self.names_low.iter().any(|n| piece.def_words.contains(n));
        let structural = share(&piece.def_words) + if defined { 1.0 } else { 0.0 };
        let affinity = if self.wsum > 0.0 && !piece.distinct.is_empty() {
            best.fill(-1.0_f32);
            for &t in &piece.distinct {
                if let Some(c) = cos.get(t) {
                    for (b, x) in best.iter_mut().zip(c) {
                        if *x > *b {
                            *b = *x;
                        }
                    }
                } else {
                    let entry = fallback.entry(t).or_insert_with(|| {
                        let id = usize::try_from(t).ok()?;
                        if buf.len() != self.width {
                            buf.resize(self.width, 0.0_f32);
                        }
                        if !self.table.row(id, buf) {
                            return None;
                        }
                        let n = buf.iter().map(|x| x * x).sum::<f32>().sqrt();
                        Some(
                            self.qv
                                .chunks_exact(self.width)
                                .map(|q| if n > 0.0 { q.iter().zip(buf.iter()).map(|(a, b)| a * b).sum::<f32>() / n } else { 0.0 })
                                .collect(),
                        )
                    });
                    if let Some(c) = entry.as_deref() {
                        for (b, x) in best.iter_mut().zip(c) {
                            if *x > *b {
                                *b = *x;
                            }
                        }
                    }
                }
            }
            best.iter().zip(&self.qw).map(|(b, w)| b.max(0.0) * w).sum::<f32>() / self.wsum
        } else {
            0.0
        };
        let (k1, b) = (BM25_K1, BM25_B);
        let norm = k1 * (1.0 - b + b * f32::from(u16::try_from(piece.bag_len).unwrap_or(u16::MAX)) / self.avg_len.max(1.0));
        let bm25 = self
            .bm25
            .iter()
            .filter_map(|(w, idf)| {
                piece.bag.get(w).map(|f| {
                    let f = f32::from(u16::try_from(*f).unwrap_or(u16::MAX));
                    idf * f * (k1 + 1.0) / (f + norm)
                })
            })
            .sum();
        let normf = k1 * (1.0 - b + b * f32::from(u16::try_from(piece.bagf_len).unwrap_or(u16::MAX)) / self.avg_len_f.max(1.0));
        let bm25f = self
            .bm25
            .iter()
            .filter_map(|(w, idf)| {
                piece.bagf.get(w).map(|f| {
                    let f = f32::from(u16::try_from(*f).unwrap_or(u16::MAX));
                    idf * f * (k1 + 1.0) / (f + normf)
                })
            })
            .sum();
        Signals { exact, ident, structural, affinity, bm25, bm25f }
    }
    /// Sense one piece.
    pub fn sense(&mut self, piece: &Record) -> Signals {
        let exact = if self.names.is_empty() {
            0.0
        } else {
            u16::try_from(self.names.iter().filter(|n| piece.text.contains(n.as_str())).count()).map_or(0.0, f32::from)
                / u16::try_from(self.names.len()).map_or(1.0, f32::from)
        };
        let total = self.terms_total;
        let share = |set: &HashSet<String>| {
            if total > 0.0 { self.terms.iter().filter(|(w, _)| set.contains(w)).map(|t| t.1).sum::<f32>() / total } else { 0.0 }
        };
        let ident = share(&piece.words);
        let defined = self.names_low.iter().any(|n| piece.def_words.contains(n));
        let structural = share(&piece.def_words) + if defined { 1.0 } else { 0.0 };
        let q = self.qw.len();
        let mut best = vec![-1.0f32; q];
        for t in &piece.distinct {
            if let Some(c) = self.cosines(*t) {
                for (b, x) in best.iter_mut().zip(c) {
                    if *x > *b {
                        *b = *x;
                    }
                }
            }
        }
        let affinity = if self.wsum > 0.0 && !piece.distinct.is_empty() {
            best.iter().zip(&self.qw).map(|(b, w)| b.max(0.0) * w).sum::<f32>() / self.wsum
        } else {
            0.0
        };
        let (k1, b) = (BM25_K1, BM25_B);
        let norm = k1 * (1.0 - b + b * f32::from(u16::try_from(piece.bag_len).unwrap_or(u16::MAX)) / self.avg_len.max(1.0));
        let bm25 = self
            .bm25
            .iter()
            .filter_map(|(w, idf)| {
                piece.bag.get(w).map(|f| {
                    let f = f32::from(u16::try_from(*f).unwrap_or(u16::MAX));
                    idf * f * (k1 + 1.0) / (f + norm)
                })
            })
            .sum();
        let normf = k1 * (1.0 - b + b * f32::from(u16::try_from(piece.bagf_len).unwrap_or(u16::MAX)) / self.avg_len_f.max(1.0));
        let bm25f = self
            .bm25
            .iter()
            .filter_map(|(w, idf)| {
                piece.bagf.get(w).map(|f| {
                    let f = f32::from(u16::try_from(*f).unwrap_or(u16::MAX));
                    idf * f * (k1 + 1.0) / (f + normf)
                })
            })
            .sum();
        Signals { exact, ident, structural, affinity, bm25, bm25f }
    }
}

/// A piece as handed to its tentacle: the sensory record, its signals, and its place in the
/// drone's combined order (0 = stands out most). Every piece taken in is handed on.
#[derive(Clone, Debug)]
pub struct Handed {
    pub record: Sensory<Record>,
    pub signals: Signals,
    pub fused: f32,
    pub place: usize,
}

/// BM25F weight of a definition-line or path hit (Sourcegraph's published value; they report the
/// ranking is not very sensitive to it).
pub const BM25F_BOOST: u32 = 5;

/// BM25 saturation and length normalisation (the Lucene defaults).
pub const BM25_K1: f32 = 1.2;
pub const BM25_B: f32 = 0.75;

/// Words a BM25 bag holds: split words of three or more letters, not in `nmlex::STOP`.
#[must_use]
pub fn bm25_words(text: &str) -> Vec<String> {
    split_words(text).into_iter().filter(|w| w.len() > 2 && !nmlex::STOP.contains(&w.as_str())).collect()
}

/// Reciprocal-rank fusion constant (the usual value, Cormack, Clarke and Buettcher 2009).
pub const RRF_K: f32 = 60.0;

/// Combined order of signal rows by reciprocal-rank fusion over the four signals (a signal at
/// zero for a piece adds nothing for it). Returns per row its fused score.
#[must_use]
pub fn fuse(rows: &[Signals]) -> Vec<f32> {
    fuse_with(rows, &FIRST_FORM)
}

/// A signal of a sensed piece, for fusion.
pub type Sig = fn(&Signals) -> f32;
/// The four signals of the first form.
pub const FIRST_FORM: [Sig; 4] = [|s| s.exact, |s| s.ident, |s| s.structural, |s| s.affinity];
/// The first form plus BM25 (2026-11-10).
pub const WITH_BM25: [Sig; 5] = [|s| s.exact, |s| s.ident, |s| s.structural, |s| s.affinity, |s| s.bm25];

/// Reciprocal-rank fusion over the given signals.
#[must_use]
pub fn fuse_with(rows: &[Signals], get: &[Sig]) -> Vec<f32> {
    let mut fused = vec![0.0f32; rows.len()];
    for &g in get {
        let mut idx: Vec<usize> = (0..rows.len()).collect();
        idx.sort_by(|a, b| rows.get(*b).map_or(0.0, g).total_cmp(&rows.get(*a).map_or(0.0, g)).then(a.cmp(b)));
        for (rank, i) in idx.into_iter().enumerate() {
            if rows.get(i).map_or(0.0, g) > 0.0
                && let Some(f) = fused.get_mut(i)
            {
                *f += 1.0 / (RRF_K + u16::try_from(rank + 1).map_or(f32::from(u16::MAX), f32::from));
            }
        }
    }
    fused
}

/// Sense every piece against the request and hand all of them on, in the drone's combined order.
/// Only material reaches the material's drones: a request record among the pieces is refused.
/// # Errors
/// `WRONG_ORIGIN` when a piece came from the request.
pub fn hand_off(drone: &Drone, request: &FromRequest<Record>, rarity: &Rarity, pieces: Vec<Sensory<Record>>) -> Result<Vec<Handed>, Fault> {
    for p in &pieces {
        inlets::MATERIAL_DRONES.admit(p.get())?;
    }
    let mut probe = drone.probe_request(request, rarity);
    let rows: Vec<Signals> = pieces.iter().map(|p| probe.sense(p.get())).collect();
    let fused = fuse(&rows);
    let mut out: Vec<Handed> =
        pieces.into_iter().zip(rows).zip(fused).map(|((record, signals), fused)| Handed { record, signals, fused, place: 0 }).collect();
    out.sort_by(|a, b| b.fused.total_cmp(&a.fused).then(a.record.get().position.cmp(&b.record.get().position)));
    for (i, h) in out.iter_mut().enumerate() {
        h.place = i;
    }
    Ok(out)
}

/// The call a tentacle pass makes for its window: the request text enters a drone first, then each
/// candidate piece (path, first line, last line, text); all are handed back, sensed and ordered,
/// none dropped. Rarity is taken over the pieces given.
/// # Errors
/// The tokenizer refuses a text.
pub fn window_pieces(drone: &Drone, request: &str, pieces: &[(String, usize, usize, String)]) -> Result<Vec<Handed>, Fault> {
    let req = drone.take_request("request", request)?;
    let mut taken = Vec::with_capacity(pieces.len());
    for (i, (path, start, end, text)) in pieces.iter().enumerate() {
        taken.push(drone.take_in(Kind::Code, Source::Lines { path: path.clone(), start: *start, end: *end }, i + 1, text)?);
    }
    let rarity = Rarity::from_records(taken.iter().map(Sensory::get));
    hand_off(drone, &req, &rarity, taken)
}

/// A question asking for the callers of something: who or which code calls it, its caller, where
/// it is started, invoked or called from.
#[must_use]
pub fn asks_callers(question: &str) -> bool {
    let w: Vec<String> = question.split(|c: char| !c.is_alphanumeric()).filter(|x| !x.is_empty()).map(str::to_lowercase).collect();
    let has = |x: &str| w.iter().any(|y| y == x);
    let pair = |a: &str, b: &str| w.windows(2).any(|p| p.first().is_some_and(|x| x == a) && p.get(1).is_some_and(|x| x == b));
    ["calls", "caller", "callers", "invoked", "invokes"].iter().any(|x| has(x)) || pair("called", "from") || pair("called", "by") || pair("started", "from")
}

/// The question's wording kind: `relation` (asks for callers), `value` (a default, limit,
/// threshold or count), `explain` (why, or how something works), `name` (where something is
/// defined, or which function or type does something), else `behaviour`. The whole path's
/// arrangements are keyed by it.
#[must_use]
pub fn wording_kind(question: &str) -> &'static str {
    let w: Vec<String> = question.split(|c: char| !c.is_alphanumeric() && c != '_').filter(|x| !x.is_empty()).map(str::to_lowercase).collect();
    let has = |x: &str| w.iter().any(|y| y == x);
    let starts = |a: &str, b: &[&str]| w.first().is_some_and(|f| f == a) && w.get(1).is_some_and(|s| b.contains(&s.as_str()));
    if asks_callers(question) {
        "relation"
    } else if ["value", "default", "limit", "threshold", "maximum", "minimum", "timeout", "budget", "constant"].iter().any(|x| has(x))
        || starts("how", &["many", "deeply", "long", "much"])
    {
        "value"
    } else if has("why") || starts("how", &["does", "do", "is", "are"]) || has("explain") {
        "explain"
    } else if (starts("where", &["is", "are"]) && (has("defined") || has("implemented") || has("declared")))
        || starts("which", &["function", "type", "struct", "enum", "method", "trait", "routine", "module"])
        || starts("what", &["function", "type", "struct", "enum", "method", "trait", "routine"])
    {
        "name"
    } else {
        "behaviour"
    }
}

/// How an aspect is written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// A group of adjacent content words.
    Words,
    /// One identifier as written (`make_room`, `tools/call`, `SHA-256`).
    Identifier,
    /// One quantity (`384`, `1.99.1`).
    Quantity,
}

/// One thing a request asks about: its words as written, their stems (an identifier's stems are
/// those of its parts) and its form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Aspect {
    pub text: String,
    pub words: Vec<String>,
    pub stems: Vec<String>,
    pub form: Form,
    /// The separate things the clause names and what it asks of them.
    pub parts: Vec<Part>,
}

/// One separate thing within an aspect: an identifier, a quantity, or a run of adjacent content
/// words (a thing named), or a plain word standing alone in its clause (what is asked of it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Part {
    pub text: String,
    pub words: Vec<String>,
    pub stems: Vec<String>,
    pub form: Form,
    pub role: Role,
}

/// What a part is to the request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// A thing the request names: an identifier, a quantity, or a run of adjacent content words.
    Named,
    /// What the request asks of a thing it names: a plain word standing alone in its clause.
    Asked,
}

/// A relation the request asks about. It is settled by a piece at the far end of a typed
/// relation from a piece holding what the request names, on that piece's own record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Asks {
    /// Who calls what is named.
    Callers,
    /// What it calls, down a path.
    Callees,
    /// The definitions it reads: a constant, a default, a type.
    Reads,
}
impl Asks {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Callers => "its callers",
            Self::Callees => "what it calls",
            Self::Reads => "what it reads",
        }
    }
}

/// The request as its own drone reads it: the kind of request (`locate`, `value`, `explain`,
/// `trace`, `absent-check`), its aspects (the separate things it names and what it asks of them)
/// and the relations it asks about. A type of its own, taken from the request only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    pub kind: &'static str,
    pub aspects: Vec<Aspect>,
    pub relations: Vec<Asks>,
}
impl Sourced for Request {
    fn origin(&self) -> Origin {
        Origin::Request
    }
}

/// Words that carry no aspect: function words, question words, and the words that say what kind
/// of answer is wanted rather than what it is about.
const FRAME: &[&str] = &[
    "a", "an", "the", "of", "in", "on", "at", "by", "for", "with", "without", "from", "to", "into", "onto", "through", "over", "under", "between",
    "as", "and", "or", "but", "if", "then", "than", "so", "that", "this", "these", "those", "it", "its", "it's", "they", "them", "their", "there",
    "what", "where", "which", "who", "whom", "whose", "how", "why", "when", "whether", "is", "are", "was", "were", "be", "been", "being", "do", "does",
    "did", "done", "has", "have", "had", "can", "could", "may", "might", "must", "shall", "should", "will", "would", "code", "function", "functions",
    "method", "methods", "file", "files", "line", "lines", "defined", "declared", "implemented", "located", "set", "computed", "happen", "happens",
    "trace", "call", "calls", "called", "caller", "callers", "invoked", "invokes", "find", "show", "get", "gets", "each", "any", "all", "some",
    "one's", "own", "still", "again", "also", "just", "including", "include", "includes", "happening", "what's", "exactly", "way", "ways",
    "many", "much", "large", "far", "long", "often", "one",
];
/// Question words and auxiliaries that open a new clause after `and`, `or` or a comma.
const OPENERS: &[&str] = &[
    "where", "which", "what", "how", "when", "why", "who", "whether", "is", "are", "does", "do", "did", "may", "must", "can", "could", "should", "will",
    "would", "has", "have", "including", "then",
];

/// True for a word written as an identifier: an underscore, a path or member separator between
/// letters, an inner capital after a lower-case letter, a dash joining letters and digits.
fn identifier(w: &str) -> bool {
    let cs: Vec<char> = w.chars().collect();
    let inner = |c: char| cs.windows(3).any(|x| x.get(1) == Some(&c) && x.first().is_some_and(|a| a.is_alphanumeric()) && x.get(2).is_some_and(|b| b.is_alphanumeric()));
    w.contains('_')
        || w.contains("::")
        || inner('/')
        || inner('.') && w.chars().any(char::is_alphabetic)
        || cs.windows(2).any(|x| x.first().is_some_and(|a| a.is_lowercase()) && x.get(1).is_some_and(|b| b.is_uppercase()))
        || (inner('-') && w.chars().any(|c| c.is_ascii_digit()) && w.chars().any(char::is_alphabetic))
}

/// True for a quantity: digits, with dots, commas or a percent sign.
fn quantity(w: &str) -> bool {
    w.chars().any(|c| c.is_ascii_digit()) && w.chars().all(|c| c.is_ascii_digit() || matches!(c, '.' | ',' | '%'))
}

/// Phrases that name a multi-hop trace or end-to-end flow outright.
const TRACE_WORDS: &[&str] = &[
    "trace",
    "call path",
    "call chain",
    "end to end",
    "end-to-end",
    "flow from",
    "flows from",
    "flows through",
    "flows into",
    "passes through",
    "data flow",
    "execution path",
    "step by step",
    "step-by-step",
    "lifecycle",
    "pipeline",
];

fn has_from_to_flow(words: &[String]) -> bool {
    let Some(f) = words.iter().position(|w| w == "from") else { return false };
    words[f + 1..].iter().any(|w| matches!(w.as_str(), "to" | "through" | "into"))
}

/// The word rules' kind of a request, and whether the rules are sure of it: sure on an explicit cue
/// (an existence question, a trace phrase, asked callers, a dependency job, a value word, a named
/// definition, `why`); unsure where the kind is the rules' default (`how does` read as explain, a
/// plain behaviour read as explain or locate).
#[must_use]
pub fn word_kind_of(q: &str, jobs: &[String]) -> (&'static str, bool) {
    let low = q.to_lowercase();
    let words: Vec<String> = low.split(|c: char| !c.is_alphanumeric()).filter(|x| !x.is_empty()).map(str::to_owned).collect();
    let first = words.first().map_or("", String::as_str);
    let starts = |a: &str| low.trim_start().starts_with(a);
    let wording = wording_kind(q);
    let absent = starts("is there") || starts("are there") || starts("is any") || starts("does any") || low.contains(" exist");
    let trace_words = TRACE_WORDS.iter().any(|w| low.contains(w)) || (matches!(first, "how" | "what" | "trace") && has_from_to_flow(&words));
    if absent {
        ("absent-check", true)
    } else if trace_words || wording == "relation" || (jobs.iter().any(|j| j == "dependency") && wording != "value") {
        ("trace", true)
    } else if wording == "value" || (first == "how" && words.get(1).is_some_and(|w| matches!(w.as_str(), "large" | "far" | "big" | "small" | "often" | "old"))) {
        ("value", true)
    } else if wording == "explain" || (wording == "behaviour" && !matches!(first, "where" | "which")) {
        ("explain", words.iter().any(|w| w == "why"))
    } else {
        ("locate", wording == "name")
    }
}

impl Request {
    /// Read a request: its kind by the wording rules (and the router slot's jobs, when given),
    /// its aspects by clauses. A clause ends at `, and`, `, or`, `;`, a mid-sentence `?`, at `and`
    /// or `or` before a question word or auxiliary, and at `and` or `or` joining two phrases of
    /// two content words or more each; in a trace also at `from`, `to`, `through`, `into` and
    /// `ends with`, so its first and last aspects are where the path starts and ends. Each
    /// clause's content words (with any identifier or quantity among them) are one aspect; a
    /// clause of only frame words gives none.
    ///
    /// Lane R (2026-11-18, master plan 3.1, Phase B): with `NANO_REQ_ORACLE=<file>` a question
    /// found in the file gets its hand read (B1); with the finer parts on (`NANO_REQ_PARTS=fine`,
    /// off by default) or the model form on (`NANO_REQ_MODEL=on`, off by default; it also reads
    /// the kind and the asked relations with the reader, `RequestModel`) named things are tagged against the material's symbol index and parts
    /// are formed by the finer word rules; each read is kept in the request's own store
    /// (`RequestStore`). Otherwise, and as the fallback, the word rules as before.
    #[must_use]
    pub fn read(q: &str, jobs: &[String]) -> Self {
        if let Some(r) = oracle_read(q) {
            return r;
        }
        if fine_parts_on() || embed_kind_on() {
            return read_model_form(q, jobs);
        }
        Self::read_words(q, jobs, false)
    }
    /// The word rules. `fine` (the model form's parts, lane R): closed-class words, pronouns and
    /// kind-of-answer words are frame; a kind noun next to a content word stays (`file size`,
    /// `test code`); a plain verb is split from the thing it is glued to and stands as what is
    /// asked; a comma ends a run; a lone word joins no neighbour; a lone clause carries forward
    /// only when it opens the question.
    #[must_use]
    pub fn read_words(q: &str, jobs: &[String], fine: bool) -> Self {
        Self::read_with(q, jobs, fine, None)
    }
    /// The word rules with the kind given (the model form's kind), else the wording's kind.
    fn read_with(q: &str, jobs: &[String], fine: bool, kind_in: Option<&'static str>) -> Self {
        let low = q.to_lowercase();
        let trace_words = TRACE_WORDS.iter().any(|w| low.contains(w));
        let (word_kind, _) = word_kind_of(q, jobs);
        let kind = kind_in.unwrap_or(word_kind);
        // Tokens as written, each with whether a clause boundary follows it.
        let raw: Vec<&str> = q.split_whitespace().collect();
        let mut toks: Vec<(String, bool)> = Vec::new();
        for (i, r) in raw.iter().enumerate() {
            let ends = r.ends_with(',') || r.ends_with(';') || r.ends_with(':') || (r.ends_with('?') && i + 1 < raw.len()) || (r.ends_with('.') && i + 1 < raw.len() && !identifier(r.trim_end_matches('.')));
            let w = r.trim_matches(|c: char| !(c.is_alphanumeric() || c == '_'));
            let w = w.strip_suffix("'s").unwrap_or(w);
            if !w.is_empty() {
                toks.push((w.to_owned(), ends));
            } else if let Some(last) = toks.last_mut() {
                last.1 |= ends;
            }
        }
        // The word a "how many" or "how much" asks to count names the thing counted, even when it
        // is a frame word elsewhere ("how many lines").
        let counted: Vec<String> = toks
            .windows(3)
            .filter(|w| w.first().is_some_and(|x| x.0.eq_ignore_ascii_case("how")) && w.get(1).is_some_and(|x| x.0.eq_ignore_ascii_case("many") || x.0.eq_ignore_ascii_case("much")))
            .filter_map(|w| w.get(2).map(|x| x.0.to_lowercase()))
            .collect();
        let frame = |w: &str| {
            let l = w.to_lowercase();
            (FRAME.contains(&l.as_str()) || (fine && FINE_FRAME.contains(&l.as_str()))) && !counted.contains(&l)
        };
        let base = |w: &str| identifier(w) || quantity(w) || (w.chars().any(char::is_alphabetic) && !frame(w));
        // A kind noun (`file`, `code`, `line`, `call`) is content in the finer rules when it
        // forms a compound with the content word before it, or (not `call`) after it.
        let content_at = |i: usize| {
            toks.get(i).is_some_and(|t| {
                base(&t.0)
                    || (fine && KIND_NOUNS.contains(&t.0.to_lowercase().as_str()) && {
                        let before = i > 0 && toks.get(i - 1).is_some_and(|p| !p.1 && base(&p.0) && !OPENERS.contains(&p.0.to_lowercase().as_str()));
                        let after = !t.1 && !t.0.eq_ignore_ascii_case("call") && toks.get(i + 1).is_some_and(|n| base(&n.0));
                        before || after
                    })
            })
        };
        let opener = |i: usize| toks.get(i).is_some_and(|t| OPENERS.contains(&t.0.to_lowercase().as_str()));
        let count = |a: usize, b: usize| (a..b.min(toks.len())).filter(|i| content_at(*i)).count();
        let trace_cut = ["from", "to", "through", "into"];
        let mut clauses: Vec<(usize, usize)> = Vec::new();
        let mut s = 0;
        let mut i = 0;
        while i < toks.len() {
            let w = toks.get(i).map_or(String::new(), |t| t.0.to_lowercase());
            let comma_before = i > 0 && toks.get(i - 1).is_some_and(|t| t.1);
            let conj = w == "and" || w == "or";
            let next_end = (i + 1..toks.len()).find(|j| toks.get(*j).is_some_and(|t| t.1)).map_or(toks.len(), |j| j + 1);
            let cut_here = (comma_before && (conj || opener(i)))
                || (conj && opener(i + 1))
                || (conj && count(s, i) >= 2 && count(i + 1, next_end) >= 2)
                || (kind == "trace" && (trace_cut.contains(&w.as_str()) || (w == "ends" && toks.get(i + 1).is_some_and(|t| t.0 == "with"))));
            // A comma alone does not end a clause (a list, an opening phrase); one before a
            // conjunction or a question word does.
            if cut_here && i > s {
                clauses.push((s, i));
                s = i;
            }
            i += 1;
        }
        clauses.push((s, toks.len()));
        // Within a clause, the separate things it names: each identifier or quantity alone, and
        // each run of adjacent plain content words (a frame word ends a run). A run of one plain
        // word joins the plain run after it in its clause, else the one before; with neither, it
        // stands alone as what the clause asks of the thing it names.
        let ident = |w: &str| identifier(w) || quantity(w);
        let lone = |g: &[String]| matches!(g, [w] if !ident(w));
        let clause_groups = |a: usize, b: usize| -> Vec<Vec<String>> {
            let mut runs: Vec<Vec<String>> = Vec::new();
            let mut cur: Vec<String> = Vec::new();
            for i in a..b.min(toks.len()) {
                let Some(t) = toks.get(i) else { break };
                if ident(&t.0) {
                    if !cur.is_empty() {
                        runs.push(std::mem::take(&mut cur));
                    }
                    runs.push(vec![t.0.clone()]);
                } else if content_at(i) {
                    if fine {
                        // A plain verb stands alone; a participle closing a run is a verb too
                        // (`session saved`), one opening it is an adjective (`learned pieces`).
                        let next_in_run = !t.1 && i + 1 < b && content_at(i + 1) && toks.get(i + 1).is_some_and(|n| !ident(&n.0));
                        let prev_ly = cur.last().is_some_and(|p| p.to_lowercase().ends_with("ly"));
                        // A listed verb in participle form (`saved`, `kept`) inside a run is an
                        // adjective too (`old saved prompt text`): split only where it ends one.
                        let run_end = !next_in_run && !cur.is_empty() && !prev_ly;
                        let base_verb = verb_word(&t.0) && !participle(&t.0) && !PARTICIPLES.contains(&t.0.to_lowercase().as_str());
                        let mid_run = !cur.is_empty() && next_in_run;
                        if base_verb || (verb_word(&t.0) && !mid_run) || (participle(&t.0) && run_end) {
                            if !cur.is_empty() {
                                runs.push(std::mem::take(&mut cur));
                            }
                            runs.push(vec![t.0.clone()]);
                            continue;
                        }
                    }
                    cur.push(t.0.clone());
                    if fine && t.1 {
                        runs.push(std::mem::take(&mut cur));
                    }
                } else if !cur.is_empty() {
                    runs.push(std::mem::take(&mut cur));
                }
            }
            if !cur.is_empty() {
                runs.push(cur);
            }
            if fine {
                return runs;
            }
            let plain = |r: &[String]| r.first().is_some_and(|w| !ident(w));
            let mut out: Vec<Vec<String>> = Vec::new();
            let mut i = 0;
            while i < runs.len() {
                let Some(r) = runs.get(i) else { break };
                if lone(r) {
                    if let Some(next) = runs.get(i + 1).filter(|n| plain(n)) {
                        let mut g = r.clone();
                        g.extend(next.iter().cloned());
                        out.push(g);
                        i += 2;
                        continue;
                    }
                    if let Some(prev) = out.last_mut().filter(|p| plain(p)) {
                        prev.extend(r.iter().cloned());
                        i += 1;
                        continue;
                    }
                }
                out.push(r.clone());
                i += 1;
            }
            out
        };
        let stems_of = |ws: &[String]| -> Vec<String> {
            let mut stems: Vec<String> = Vec::new();
            for w in ws {
                let parts = if identifier(w) { nmlex::name_parts(&w.replace(['/', '.'], "_")) } else { vec![w.to_lowercase()] };
                for p in parts {
                    let st = if quantity(&p) { p } else { nmlex::stem(&p) };
                    if !stems.contains(&st) {
                        stems.push(st);
                    }
                }
            }
            stems
        };
        let form_of = |ws: &[String]| match ws {
            [one] if identifier(one) => Form::Identifier,
            [one] if quantity(one) => Form::Quantity,
            _ => Form::Words,
        };
        // Each clause's content words are one aspect (the drones pose each aspect against the
        // content-free pose). A clause naming one plain word only (an opening phrase such as "By
        // default,") belongs to the clause after it, or else to the one before; its word joins
        // that clause's first plain part.
        let mut joined: Vec<(Vec<String>, Vec<Vec<String>>)> = Vec::new();
        let mut carry: Vec<String> = Vec::new();
        for (a, b) in &clauses {
            let mut gs = clause_groups(*a, *b);
            if gs.is_empty() {
                continue;
            }
            if let [g] = gs.as_slice()
                && lone(g)
                && (!fine || joined.is_empty())
            {
                carry.extend(g.iter().cloned());
                continue;
            }
            // Finer rules: a clause naming only what is asked (`and where are they checked`) asks
            // it of the clause before, so its verbs join that aspect.
            if fine
                && gs.iter().all(|g| matches!(g.as_slice(), [w] if !ident(w) && (verb_word(w) || participle(w))))
                && let Some((words, prev)) = joined.last_mut()
            {
                words.extend(gs.iter().flatten().cloned());
                prev.extend(gs);
                continue;
            }
            let mut words = std::mem::take(&mut carry);
            words.extend((*a..(*b).min(toks.len())).filter(|i| content_at(*i)).filter_map(|i| toks.get(i).map(|t| t.0.clone())));
            let lead: Vec<String> = words.iter().take_while(|w| !gs.iter().flatten().any(|x| x == *w)).cloned().collect();
            if !lead.is_empty() {
                match gs.iter_mut().find(|g| g.first().is_some_and(|w| !ident(w))) {
                    Some(g) => {
                        let mut c = lead;
                        c.append(g);
                        *g = c;
                    }
                    None => gs.insert(0, lead),
                }
            }
            joined.push((words, gs));
        }
        if !carry.is_empty() {
            match joined.last_mut() {
                Some((words, gs)) => {
                    words.extend(carry.iter().cloned());
                    match gs.iter_mut().rev().find(|g| g.first().is_some_and(|w| !ident(w))) {
                        Some(g) => g.extend(carry),
                        None => gs.push(carry),
                    }
                }
                None => joined.push((carry.clone(), vec![carry])),
            }
        }
        let mut aspects: Vec<Aspect> = Vec::new();
        for (ws, gs) in joined {
            let parts: Vec<Part> = gs
                .into_iter()
                .map(|g| {
                    let asked = if fine { matches!(g.as_slice(), [w] if !ident(w) && (verb_word(w) || (participle(w) && w.to_lowercase().ends_with("ed")))) } else { lone(&g) };
                    Part { text: g.join(" "), stems: stems_of(&g), form: form_of(&g), role: if asked { Role::Asked } else { Role::Named }, words: g }
                })
                .collect();
            aspects.push(Aspect { text: ws.join(" "), stems: stems_of(&ws), form: form_of(&ws), words: ws, parts });
        }
        // The relations asked about, each settled by a piece at the far end of a typed relation
        // from a piece holding what is named: its callers when the request asks who calls it; its
        // callees when it asks for a path or how something works; the definitions it reads (a
        // constant, a default, a type) when it asks for a value or where one is set.
        // The request's own words, not the parts of an identifier it names (`.._set_..`).
        let plain: Vec<String> = toks.iter().filter(|t| !identifier(&t.0)).map(|t| t.0.to_lowercase()).collect();
        let has = |x: &str| plain.iter().any(|w| w == x);
        let mut relations: Vec<Asks> = Vec::new();
        if asks_callers(q) {
            relations.push(Asks::Callers);
        }
        if matches!(kind, "trace" | "explain") && !(kind == "trace" && relations.contains(&Asks::Callers) && !trace_words) {
            relations.push(Asks::Callees);
        }
        if kind == "value" || ["set", "default", "defaults", "configured", "limit"].iter().any(|w| has(w)) {
            relations.push(Asks::Reads);
        }
        Self { kind, aspects, relations }
    }
    /// The aspects one line each, for the audit: `kind <kind>`, then `aspect <text>` per aspect.
    #[must_use]
    pub fn cells(&self) -> Vec<String> {
        let mut out = vec![format!("kind {}", self.kind)];
        out.extend(self.aspects.iter().map(|a| format!("aspect {}", a.text)));
        out.extend(self.relations.iter().map(|r| format!("relation {}", r.name())));
        out
    }
    /// The aspects a text covers: an aspect is covered when the text holds at least half of its
    /// stems (an identifier or quantity: when it holds it whole, or every part's stem).
    #[must_use]
    pub fn covered_by(&self, text: &str) -> Vec<bool> {
        let low = text.to_lowercase();
        let stems: std::collections::BTreeSet<String> =
            low.split(|c: char| !(c.is_alphanumeric() || c == '_')).flat_map(nmlex::name_parts).map(|p| nmlex::stem(&p)).collect();
        self.aspects
            .iter()
            .map(|a| match a.form {
                Form::Identifier | Form::Quantity => {
                    a.words.first().is_some_and(|w| low.contains(&w.to_lowercase())) || a.stems.iter().all(|s| stems.contains(s))
                }
                Form::Words => 2 * a.stems.iter().filter(|s| stems.contains(*s)).count() >= a.stems.len(),
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------------------------
// Lane R (2026-11-18; the 2026-11-18 tentacle master plan, 3.1, Phase B):
// the request's drone, model form. The kind is read in one forward pass of the reader per short
// prompt (next-token probabilities over the kind labels, no generation); the asked relations by
// yes/no reads on short templates, batched with the kind read; named things by lexical tagging
// against the material's symbol index (exact identifiers, split identifiers, file paths); the
// parts by the finer word rules. Off by default: `NANO_REQ_PARTS=fine` for the parts and tags alone,
// `NANO_REQ_MODEL=on` for the reader's kind and relations with them. The word rules stay as the
// fallback. Every read is kept in the request's own store (`RequestStore`), a type of its own that
// no material type converts from or into (test ISO-R, `check_iso_r`).
// ---------------------------------------------------------------------------------------------

/// Words the finer rules count as frame besides `FRAME`: pronouns, closed-class words and
/// kind-of-answer words (hand reading of 20 questions, C14: `I`, `am`, `supposed`, `before`,
/// `after`, `up`, `out`, `anyway`, `always`, `ones`, `time`, `work`).
const FINE_FRAME: &[&str] = &[
    "i", "me", "my", "mine", "we", "our", "us", "you", "your", "am", "before", "after", "up", "out", "not", "cannot", "anyway", "always", "ones",
    "too", "about", "together", "time", "times", "work", "works", "supposed", "want", "wants", "need", "needs", "both", "either", "whenever",
    "while", "once", "later", "kind", "kinds",
];
/// Kind nouns: frame alone (`which file`), content inside a compound (`file size`, `test code`,
/// `command line`, `tool call`).
const KIND_NOUNS: &[&str] = &["code", "file", "files", "line", "lines", "function", "functions", "method", "methods", "call"];
/// Plain verbs of questions: what is asked of a thing, never part of its name.
const VERBS: &[&str] = &[
    "decide", "decides", "decided", "raise", "raises", "lower", "lowers", "report", "reports", "reported", "reach", "reaches", "hold", "holds",
    "read", "reads", "look", "looks", "open", "opens", "learn", "learns", "run", "runs", "make", "makes", "made", "keep", "keeps", "kept", "send",
    "sends", "sent", "connect", "connects", "connected", "started", "walk", "walks", "walking", "put", "puts", "putting", "switch", "switched",
    "checked", "fill", "fills", "filled", "save", "saves", "saved", "clean", "cleaned", "give", "gives", "given", "worked", "wake", "wakes", "left",
    "leave", "leaves", "dropped", "dropping", "distribute", "distributed", "use", "uses", "return", "returns", "returned", "pick", "picks",
    "picked", "choose", "chooses", "chosen", "stopped", "skip", "skips", "skipped", "turn", "turns", "turned", "write", "writes", "written",
    "loaded", "stored", "handle", "handles", "handled", "passed", "reject", "rejects", "rejected", "refuse", "refuses", "refused", "selected",
    "sorted", "removed", "added", "replaced", "reached",
];

/// Listed verbs in a participle form without `-ed` (adjectives inside a run: `kept copies`).
const PARTICIPLES: &[&str] = &["kept", "made", "sent", "given", "left", "written", "chosen"];

fn verb_word(w: &str) -> bool {
    VERBS.contains(&w.to_lowercase().as_str())
}
/// A word ending in `-ed` or `-ing` (five letters or more): a verb when it closes a run.
fn participle(w: &str) -> bool {
    let l = w.to_lowercase();
    l.len() >= 5 && l.chars().all(char::is_alphabetic) && (l.ends_with("ed") || l.ends_with("ing"))
}

fn words_stems(ws: &[String]) -> Vec<String> {
    let mut stems: Vec<String> = Vec::new();
    for w in ws {
        let parts = if identifier(w) { nmlex::name_parts(&w.replace(['/', '.'], "_")) } else { vec![w.to_lowercase()] };
        for p in parts {
            let st = if quantity(&p) { p } else { nmlex::stem(&p) };
            if !stems.contains(&st) {
                stems.push(st);
            }
        }
    }
    stems
}
fn words_form(ws: &[String]) -> Form {
    match ws {
        [one] if identifier(one) => Form::Identifier,
        [one] if quantity(one) => Form::Quantity,
        _ => Form::Words,
    }
}
fn part_of(ws: Vec<String>, role: Role) -> Part {
    Part { text: ws.join(" "), stems: words_stems(&ws), form: words_form(&ws), role, words: ws }
}

// ---- B1: the oracle (hand reads injected by question) ----

static ORACLE: std::sync::OnceLock<HashMap<String, Request>> = std::sync::OnceLock::new();

fn static_kind(k: &str) -> &'static str {
    match k {
        "locate" => "locate",
        "trace" => "trace",
        "change" => "change",
        "value" => "value",
        "prose" => "prose",
        "absent-check" | "absent" => "absent-check",
        _ => "explain",
    }
}

/// Parse hand reads: one line per question, `question<TAB>kind<TAB>relations<TAB>aspects`;
/// relations comma-separated (`callers`, `callees`, `reads`); aspects separated by `||`, parts by
/// `|`, an asked part written with a leading `?`. Lines starting with `#` are notes.
#[must_use]
pub fn parse_oracle(text: &str) -> HashMap<String, Request> {
    let mut out = HashMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')) {
        let f: Vec<&str> = line.split('\t').collect();
        let [q, kind, rels, aspects] = f.as_slice() else { continue };
        let relations = rels
            .split(',')
            .filter_map(|r| match r.trim() {
                "callers" => Some(Asks::Callers),
                "callees" => Some(Asks::Callees),
                "reads" => Some(Asks::Reads),
                _ => None,
            })
            .collect();
        let aspects = aspects
            .split("||")
            .map(|a| {
                let parts: Vec<Part> = a
                    .split('|')
                    .map(str::trim)
                    .filter(|p| !p.is_empty())
                    .map(|p| {
                        let (role, p) = p.strip_prefix('?').map_or((Role::Named, p), |r| (Role::Asked, r));
                        part_of(p.split_whitespace().map(str::to_owned).collect(), role)
                    })
                    .collect();
                let words: Vec<String> = parts.iter().flat_map(|p| p.words.iter().cloned()).collect();
                Aspect { text: words.join(" "), stems: words_stems(&words), form: words_form(&words), words, parts }
            })
            .filter(|a| !a.parts.is_empty())
            .collect();
        out.insert(q.trim().to_owned(), Request { kind: static_kind(kind.trim()), aspects, relations });
    }
    out
}

/// The hand read of `q` when `NANO_REQ_ORACLE` names a file holding it (B1, a measurement only).
/// With `NANO_REQ_ORACLE_PARTS=only` only the aspects are taken; kind and relations stay the word
/// rules'.
fn oracle_read(q: &str) -> Option<Request> {
    let path = std::env::var_os("NANO_REQ_ORACLE")?;
    let map = ORACLE.get_or_init(|| std::fs::read_to_string(&path).map(|t| parse_oracle(&t)).unwrap_or_default());
    let hand = map.get(q.trim())?;
    if std::env::var("NANO_REQ_ORACLE_PARTS").as_deref() == Ok("only") {
        let mut r = Request::read_words(q, &[], false);
        r.aspects.clone_from(&hand.aspects);
        return Some(r);
    }
    Some(hand.clone())
}

// ---- B2: the model form ----

/// The kind read, second form (lane R, 2026-11-18, the lead's package 2): one yes/no prompt per
/// kind, as the relations; each kind scored by its yes log-odds less its mean yes log-odds over the
/// content-free questions (`NEUTRAL`); the highest wins. Locate and absent-check are asked apart
/// explicitly. (First form, a softmax over one-word labels after one prompt: T-065.)
///
/// Roadmap Phase 4 (2026-11-24), T-065 and T-070: each kind now has three contrastive templates,
/// each asking for what the asker wants back (a reason, a place, a path, an edit, a value, what the
/// guides say, whether a thing exists) and naming what the kind is not, rather than the topic words
/// a question uses. A kind's read is the mean yes log-odds over its templates, taken relative to the
/// mean over kinds (the reader says yes to nearly anything, T-070, so only the difference between
/// kinds carries signal); the content-free prior is taken the same way and capped
/// (`MODEL_PRIOR_CAP`), so no kind wins by a low prior alone (change in T-065). The label set (yes,
/// no) and the read's output (one score per kind) are as before.
pub const KIND_TEMPLATES: [(&str, &[&str]); 7] = [
    (
        "explain",
        &[
            "Does the question ask why the code does something, wanting a reason rather than a place, a path of calls or a single value?",
            "Does the question ask how one piece of code works out its result inside itself, rather than what runs before or after it?",
            "Would the best answer explain the idea or rule behind one piece of code, not list steps across several places?",
        ],
    ),
    (
        "locate",
        &[
            "Does the question ask where something is in the code, so the answer is a file or function name rather than an explanation?",
            "Would the best answer be just a place in the code, such as a file, a function or a line, and little else?",
            "Does the question ask which file or function does something, rather than how or why it does it?",
        ],
    ),
    (
        "trace",
        &[
            "Does the question ask for the path from one point in the code to another, such as what calls what or where a value flows?",
            "Does the question ask what happens step by step when something runs, starts or fails, rather than why or where?",
            "Would the best answer be a sequence of calls or events across several places in the code, in order?",
        ],
    ),
    (
        "change",
        &[
            "Does the question ask how to modify, add to or fix the code, rather than how the code behaves now?",
            "Is the asker planning to change the code and asking what to edit?",
            "Would the best answer be instructions for an edit, not a description of what the code already does?",
        ],
    ),
    (
        "value",
        &[
            "Does the question ask for one number, limit, version, name or setting, rather than how it is used?",
            "Would the best answer be a single value, such as a default, a size or a count, and little else?",
            "Does the question ask what a setting or constant is set to, rather than where it is or what happens with it?",
        ],
    ),
    (
        "prose",
        &[
            "Does the question ask what the project's documentation tells a reader, rather than about the code itself?",
            "Is the question about using the product, its licence, its guides or its policies, as a reader of the manual would ask?",
            "Would the best answer come from a guide or README page, not from reading the source code?",
        ],
    ),
    (
        "absent-check",
        &[
            "Does the question ask whether something exists at all, so the answer may be that there is none?",
            "Would the best answer start with yes or no on whether the code has such a thing at all?",
            "Does the question ask if anything anywhere handles a case, rather than how or where it is handled?",
        ],
    ),
];
/// The most the content-free prior may move one kind's model read, in yes log-odds, after the
/// prior is taken relative to its median over kinds (Roadmap Phase 4 (2026-11-24), T-065: an
/// uncapped prior lifted change on 25 of 32 questions). The median, not the mean, so one kind's
/// outlying prior does not move the others.
pub const MODEL_PRIOR_CAP: f32 = 1.0;
/// The yes/no templates of the asked relations, one forward each.
pub const RELATION_TEMPLATES: [(Asks, &str); 3] = [
    (Asks::Callers, "Does the question ask which code calls, starts or uses something?"),
    (Asks::Callees, "Does the question ask what happens in order when something runs, or how one part reaches another?"),
    (Asks::Reads, "Does the question ask for a value, a default, a limit, a version or where a setting is set?"),
];
/// The content-free questions of the calibration (as `nmapi::neural::CONTENT_FREE`).
pub const NEUTRAL: [&str; 3] = ["N/A", "", "[MASK]"];
const PROMPT_HEAD: &str = "<|im_start|>system\nYou sort questions about a code repository.<|im_end|>\n<|im_start|>user\nQuestion: ";
const PROMPT_TAIL: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// The shared head of every prompt of `q` (prefilled once on the card).
#[must_use]
pub fn request_prefix(q: &str) -> String {
    format!("{PROMPT_HEAD}{q}\n")
}
/// The part of a yes/no prompt after the shared head.
#[must_use]
pub fn yes_no_suffix(template: &str) -> String {
    format!("{template} Answer yes or no.{PROMPT_TAIL}")
}
/// A whole yes/no prompt for `q`.
#[must_use]
pub fn relation_prompt(q: &str, template: &str) -> String {
    format!("{}{}", request_prefix(q), yes_no_suffix(template))
}

/// The reader's side of the model form: for each `(prompt, labels)` item, the probability of each
/// label as the next token after the prompt (a softmax over the labels' first tokens), and the
/// milliseconds the reads took. Implemented over the reader MicroModel in `nmapi::request_read`.
pub trait RequestModel: Send + Sync {
    /// # Errors
    /// Tokenizer or engine failure, or labels sharing a first token.
    fn label_probs(&self, items: &[(String, Vec<&'static str>)]) -> Result<(Vec<Vec<f32>>, f64), Fault>;
    /// As `label_probs` for prompts that all start with `prefix` (each item gives the rest); a
    /// reader may compute the prefix once.
    /// # Errors
    /// As `label_probs`.
    fn label_probs_shared(&self, prefix: &str, items: &[(String, Vec<&'static str>)]) -> Result<(Vec<Vec<f32>>, f64), Fault> {
        let whole: Vec<(String, Vec<&'static str>)> = items.iter().map(|(s, l)| (format!("{prefix}{s}"), l.clone())).collect();
        self.label_probs(&whole)
    }
}

static MODEL: std::sync::OnceLock<Arc<dyn RequestModel>> = std::sync::OnceLock::new();
/// The reader's mean yes log-odds per prompt over `NEUTRAL` (one reader per process, keyed by it).
static PRIOR: std::sync::Mutex<Option<(usize, Vec<f32>)>> = std::sync::Mutex::new(None);
/// 0 unread, 1 off, 2 on.
static MODEL_FORM: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
/// The finer parts and the symbol tags alone (`NANO_REQ_PARTS=fine`): 0 unread, 1 off, 2 on.
static FINE_PARTS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
/// `NANO_REQ_PARTS=off` read once: 0 unread, 1 not off, 2 off.
static FINE_PARTS_OFF: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// The writer's notes as a sense (lane B's `writer_idle::note`, wired by lane T1 2026-11-19): a
/// piece's one-line note by its content key, given by the serving crate (nmpod cannot see nmapi).
pub type NoteFn = dyn Fn(&str) -> Option<String> + Send + Sync;
static NOTES: std::sync::OnceLock<Arc<NoteFn>> = std::sync::OnceLock::new();
/// Give the drones the writer's notes (once per process). Returns false when given before.
pub fn install_notes(f: Arc<NoteFn>) -> bool {
    NOTES.set(f).is_ok()
}
/// The note sense of one piece for a request: the share of the request's content words (lower
/// case, three letters or more) its writer's note holds; `None` without a note.
#[must_use]
pub fn note_sense(key: &str, request_words: &[String]) -> Option<f32> {
    let note = NOTES.get().and_then(|f| f(key))?.to_lowercase();
    if request_words.is_empty() {
        return None;
    }
    let words: std::collections::BTreeSet<&str> = note.split(|c: char| !c.is_alphanumeric() && c != '_').filter(|w| w.len() >= 3).collect();
    #[expect(clippy::cast_precision_loss, reason = "word counts")]
    let share = request_words.iter().filter(|w| words.contains(w.as_str())).count() as f32 / request_words.len() as f32;
    Some(share)
}
/// The request's content words for `note_sense`.
#[must_use]
pub fn note_words(question: &str) -> Vec<String> {
    let mut v: Vec<String> = question.to_lowercase().split(|c: char| !c.is_alphanumeric() && c != '_').filter(|w| w.len() >= 3).map(str::to_owned).collect();
    v.sort();
    v.dedup();
    v
}

/// Give the request's drone the reader (once per process). Returns false when one was given.
pub fn install_model(m: Arc<dyn RequestModel>) -> bool {

    MODEL.set(m).is_ok()
}
/// Whether a reader was given.
#[must_use]
pub fn model_installed() -> bool {
    MODEL.get().is_some()
}
/// Turn the model form (the reader's kind and relations, with the finer parts and tags) on or off
/// for this process (otherwise `NANO_REQ_MODEL=on`).
pub fn set_model_form(on: bool) {
    MODEL_FORM.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
}
/// Turn the finer parts and the symbol tags on or off for this process (otherwise
/// `NANO_REQ_PARTS=fine`); the kind and relations stay the word rules' unless the model form is on.
pub fn set_fine_parts(on: bool) {
    FINE_PARTS.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
}
fn switch(cell: &std::sync::atomic::AtomicU8, var: &str, on_value: &str) -> bool {
    match cell.load(std::sync::atomic::Ordering::Relaxed) {
        2 => true,
        1 => false,
        _ => {
            let on = std::env::var(var).as_deref() == Ok(on_value);
            cell.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
            on
        }
    }
}
/// Whether the model form is on (`NANO_REQ_MODEL=on`; off by default).
#[must_use]
pub fn model_form_on() -> bool {
    switch(&MODEL_FORM, "NANO_REQ_MODEL", "on")
}
/// Whether the finer parts and symbol tags are on: on by default since 2026-11-18 (Luis: parts that
/// advance the architecture are not left off; lane T1 turned it on), `NANO_REQ_PARTS=off` (or
/// `set_fine_parts(false)`) restores the word rules' parts; always with the model form.
#[must_use]
pub fn fine_parts_on() -> bool {
    (!switch(&FINE_PARTS_OFF, "NANO_REQ_PARTS", "off") && FINE_PARTS.load(std::sync::atomic::Ordering::Relaxed) != 1) || model_form_on()
}

// ---- The kind read by the embedder (lane R, 2026-11-18, the lead's third package) ----

/// The embedder's side of the kind read: the L2-normalised embedding of each text, and the
/// milliseconds it took. Implemented over the embedder MicroModel in `nmapi::request_read`.
pub trait TextEmbedder: Send + Sync {
    /// # Errors
    /// Tokenizer or engine failure.
    fn embed(&self, texts: &[String]) -> Result<(Vec<Vec<f32>>, f64), Fault>;
}

/// The instruction every text of the kind read is embedded under (questions and descriptions
/// alike: a symmetric comparison of what kind of answer is asked for).
pub const KIND_TASK: &str = "Given a question about a code repository, retrieve questions that ask for the same kind of answer";
/// Written descriptions of each kind (lane R, from the kinds' definitions in master plan 3.1; no
/// test question used). X, Y, A and B stand for any named thing.
///
/// Roadmap Phase 4 (2026-11-24), T-071: rewritten so the kinds part by what the asker wants back,
/// not by topic words. Trace asks for the path of calls or data from one point to another, or what
/// happens in order; locate asks where a thing is; explain asks why, or how one piece works out its
/// result; value asks what a setting or number is; change asks how to modify; prose asks what the
/// guides tell a user; absent-check asks whether a thing exists at all. Five or more short
/// descriptions per kind, each a prototype; see `embed_kind_scores` for how they are combined.
pub const KIND_DESCRIPTIONS: [(&str, &[&str]); 7] = [
    (
        "explain",
        &[
            "Why does the code do it this way?",
            "How does this function work out its result?",
            "What is the reasoning behind this rule?",
            "What is this check for?",
            "Explain the idea behind this algorithm.",
            "How does this formula decide its answer?",
        ],
    ),
    (
        "locate",
        &[
            "Where is X defined?",
            "Which file implements this feature?",
            "Where in the code is this done?",
            "Which function handles this?",
            "In which module does this check live?",
        ],
    ),
    (
        "trace",
        &[
            "What calls X, and what calls that?",
            "How does a request get from the entry point to Y?",
            "Where does this value flow after it is read, and which functions pass it on?",
            "What happens, in order, when this command starts?",
            "After this event fires, what runs next?",
            "When this fails, which code handles it and what does it do then?",
            "Follow the path of the data from A to B.",
            "How does an option given on the command line reach the part that uses it?",
        ],
    ),
    (
        "change",
        &[
            "How do I add a new option to this?",
            "What must I edit to change this behaviour?",
            "How would I extend this to support another case?",
            "What should I modify to fix this bug?",
            "Which lines would I rewrite to make it do this instead?",
        ],
    ),
    (
        "value",
        &[
            "What is the default value of this setting?",
            "How many items does it keep at most?",
            "What is this limit set to?",
            "Which version number does it use?",
            "What is the largest size it accepts?",
            "How long is the timeout, in seconds?",
        ],
    ),
    (
        "prose",
        &[
            "What does the documentation say about this topic?",
            "What do the project's guides tell a new user?",
            "What does the README explain about installing it?",
            "What licence does the project use?",
            "How should a user contact the maintainers?",
            "Which page of the manual covers this?",
        ],
    ),
    (
        "absent-check",
        &[
            "Is there any code that does this?",
            "Does the project have a feature for this at all?",
            "Is this ever handled anywhere?",
            "Does anything support this case, or is it missing?",
            "Is there a setting for this, or none?",
        ],
    ),
];
/// The most the content-free baseline may move one kind's embedder read, in cosine, after the
/// baseline is taken relative to its median over kinds (Roadmap Phase 4 (2026-11-24): as
/// `MODEL_PRIOR_CAP`, so no kind wins by a low baseline alone).
pub const EMBED_PRIOR_CAP: f32 = 0.02;

static EMBEDDER: std::sync::OnceLock<Arc<dyn TextEmbedder>> = std::sync::OnceLock::new();
/// Per embedder (keyed by it): each kind's description vectors, each description's weight
/// (`description_weights`) and each kind's content-free baseline.
#[expect(clippy::type_complexity, reason = "one cache cell")]
static KIND_TABLE: std::sync::Mutex<Option<(usize, Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>, Vec<f32>)>> = std::sync::Mutex::new(None);
/// The kind read by the embedder (`NANO_REQ_KIND=embed`): 0 unread, 1 off, 2 on.
static EMBED_KIND: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Give the request's drone the embedder (once per process). Returns false when one was given.
pub fn install_embedder(e: Arc<dyn TextEmbedder>) -> bool {
    EMBEDDER.set(e).is_ok()
}
/// Whether an embedder was given.
#[must_use]
pub fn embedder_installed() -> bool {
    EMBEDDER.get().is_some()
}
/// Turn the embedder's kind read on or off for this process (otherwise `NANO_REQ_KIND=embed`).
pub fn set_embed_kind(on: bool) {
    EMBED_KIND.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
}
/// Whether the embedder's kind read is on (`NANO_REQ_KIND=embed`; off by default).
#[must_use]
pub fn embed_kind_on() -> bool {
    switch(&EMBED_KIND, "NANO_REQ_KIND", "embed")
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0.0_f32, 0.0_f32, 0.0_f32);
    for (x, y) in a.iter().zip(b) {
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    d / (na.sqrt() * nb.sqrt()).max(f32::MIN_POSITIVE)
}
fn mean_of(xs: &[f32]) -> f32 {
    #[expect(clippy::cast_precision_loss, reason = "a mean over a few values")]
    let n = xs.len().max(1) as f32;
    xs.iter().sum::<f32>() / n
}
fn median_of(xs: &[f32]) -> f32 {
    let mut s = xs.to_vec();
    s.sort_by(f32::total_cmp);
    let mid = s.len() / 2;
    if s.len() % 2 == 1 {
        s.get(mid).copied().unwrap_or(0.0)
    } else {
        (s.get(mid.wrapping_sub(1)).copied().unwrap_or(0.0) + s.get(mid).copied().unwrap_or(0.0)) / 2.0
    }
}
/// Roadmap Phase 4 (2026-11-24): each kind's read less the mean over kinds, so what every kind
/// shares (how close the question is to any description, or how readily the reader says yes)
/// carries no weight.
fn relative_to_kinds(xs: &[f32]) -> Vec<f32> {
    let m = mean_of(xs);
    xs.iter().map(|x| x - m).collect()
}
/// Roadmap Phase 4 (2026-11-24): a content-free prior per kind taken relative to its median over
/// kinds and capped at `cap` either way, so a kind can be moved by its prior by at most `cap` and
/// cannot win by a low prior alone.
fn capped_prior(prior: &[f32], cap: f32) -> Vec<f32> {
    let m = median_of(prior);
    prior.iter().map(|p| (p - m).clamp(-cap, cap)).collect()
}

/// Roadmap Phase 4 (2026-11-24), T-071: each description's weight, from the description vectors
/// alone (no question). A description's weight is its mean cosine to the other descriptions of its
/// own kind less its highest mean cosine to another kind's descriptions, floored at zero: a
/// description as close to another kind as to its own carries no weight. A kind whose weights are
/// all zero falls back to equal weights.
#[must_use]
pub fn description_weights(descs: &[Vec<Vec<f32>>]) -> Vec<Vec<f32>> {
    descs
        .iter()
        .enumerate()
        .map(|(k, own)| {
            let w: Vec<f32> = own
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    let mates: Vec<f32> = own.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, o)| cosine(d, o)).collect();
                    let near = if mates.is_empty() { 1.0 } else { mean_of(&mates) };
                    let other = descs
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != k)
                        .map(|(_, set)| mean_of(&set.iter().map(|o| cosine(d, o)).collect::<Vec<_>>()))
                        .fold(f32::NEG_INFINITY, f32::max);
                    (near - if other.is_finite() { other } else { 0.0 }).max(0.0)
                })
                .collect();
            if w.iter().any(|x| *x > 0.0) { w } else { vec![1.0; own.len()] }
        })
        .collect()
}

/// Roadmap Phase 4 (2026-11-24), T-071: each kind's read of one embedded text `v`, relative to the
/// other kinds. Per kind: half the highest cosine to its weighted descriptions plus half the
/// weighted mean cosine (several prototypes, max and mean); then less the mean over kinds.
#[must_use]
pub fn kind_read(v: &[f32], descs: &[Vec<Vec<f32>>], weights: &[Vec<f32>]) -> Vec<f32> {
    let raw: Vec<f32> = descs
        .iter()
        .zip(weights)
        .map(|(set, w)| {
            let sims: Vec<(f32, f32)> = set.iter().zip(w).map(|(d, wt)| (cosine(v, d), *wt)).collect();
            let top = sims.iter().filter(|(_, wt)| *wt > 0.0).map(|(s, _)| *s).fold(f32::NEG_INFINITY, f32::max);
            let top = if top.is_finite() { top } else { 0.0 };
            let total: f32 = sims.iter().map(|(_, wt)| wt).sum();
            let mean = if total > 0.0 { sims.iter().map(|(s, wt)| s * wt).sum::<f32>() / total } else { 0.0 };
            0.5 * (top + mean)
        })
        .collect();
    relative_to_kinds(&raw)
}

/// Each kind's score for `q` by the embedder, all texts under `KIND_TASK`; with the milliseconds of
/// the question's embedding. The descriptions, their weights and the baseline are embedded once per
/// embedder.
///
/// Roadmap Phase 4 (2026-11-24), T-071: the score is the question's `kind_read` (weighted
/// prototypes, max and mean, relative to the other kinds) less the content-free baseline (the mean
/// `kind_read` over `NEUTRAL`), the baseline relative to its median over kinds and capped at
/// `EMBED_PRIOR_CAP`. Before, a kind's score was its plain mean cosine less an uncapped baseline,
/// and the trace descriptions scored the same on every kind.
#[must_use]
pub fn embed_kind_scores(q: &str, e: &dyn TextEmbedder) -> Option<(Vec<(&'static str, f32)>, f64)> {
    let posed = |t: &str| format!("Instruct: {KIND_TASK}\nQuery:{t}");
    let who = std::ptr::from_ref(e).cast::<()>() as usize;
    let held = KIND_TABLE.lock().ok().and_then(|g| g.as_ref().filter(|(w, _, _, _)| *w == who).map(|(_, d, wt, b)| (d.clone(), wt.clone(), b.clone())));
    let (descs, weights, base) = match held {
        Some(x) => x,
        None => {
            let texts: Vec<String> = KIND_DESCRIPTIONS.iter().flat_map(|(_, ds)| ds.iter().map(|d| posed(d))).chain(NEUTRAL.iter().map(|n| posed(n))).collect();
            let (vs, _) = e.embed(&texts).ok()?;
            let mut it = vs.into_iter();
            let descs: Vec<Vec<Vec<f32>>> = KIND_DESCRIPTIONS.iter().map(|(_, ds)| it.by_ref().take(ds.len()).collect()).collect();
            let neutral: Vec<Vec<f32>> = it.collect();
            let weights = description_weights(&descs);
            let reads: Vec<Vec<f32>> = neutral.iter().map(|n| kind_read(n, &descs, &weights)).collect();
            let mean_read: Vec<f32> = (0..descs.len()).map(|k| mean_of(&reads.iter().filter_map(|r| r.get(k).copied()).collect::<Vec<_>>())).collect();
            let base = capped_prior(&mean_read, EMBED_PRIOR_CAP);
            if let Ok(mut g) = KIND_TABLE.lock() {
                *g = Some((who, descs.clone(), weights.clone(), base.clone()));
            }
            (descs, weights, base)
        }
    };
    let (v, ms) = e.embed(&[posed(q)]).ok()?;
    let v = v.into_iter().next()?;
    let read = kind_read(&v, &descs, &weights);
    Some((KIND_DESCRIPTIONS.iter().zip(read.iter().zip(&base)).map(|((k, _), (r, b))| (*k, r - b)).collect(), ms))
}

/// How a named thing was tagged against the material's symbol index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TagHow {
    /// The request writes the identifier as defined.
    Exact,
    /// Adjacent words of the request are a defined identifier's parts (`first fit`, `first_fit`).
    Split,
    /// Words of the request are a file's stem (`coordinator`, `coordinator.rs`).
    Path,
}
/// One named thing of the request found in the material: the words as the request writes them,
/// the symbol or path they name, how. A seed for the search.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tag {
    pub words: Vec<String>,
    pub symbol: String,
    pub how: TagHow,
}

/// The material's symbol index: defined names (lower-cased, to the name as written), defined
/// names by their split parts (`first fit` to `first_fit`), and file stems (to the path). Built
/// from material records only; nothing of a request enters it (ISO-R).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SymbolIndex {
    names: HashMap<String, String>,
    split: HashMap<String, String>,
    stems: HashMap<String, String>,
}
const DEF_HEADS: [&str; 13] = ["fn ", "struct ", "enum ", "trait ", "type ", "const ", "static ", "mod ", "class ", "def ", "function ", "interface ", "union "];
/// Most names the index holds (memory bound; names past it are not added).
const SYMBOL_CAP: usize = 400_000;
impl SymbolIndex {
    /// The index of `records` (pieces of code and documents with a path).
    #[must_use]
    pub fn from_records<'a>(records: impl IntoIterator<Item = &'a Record>) -> Self {
        let mut ix = Self::default();
        for rec in records {
            ix.add_record(rec);
        }
        ix
    }
    fn add_record(&mut self, rec: &Record) {
        let Source::Lines { path, .. } = &rec.source else { return };
        if self.names.len() >= SYMBOL_CAP {
            return;
        }
        let file = path.rsplit(['/', '\\']).next().unwrap_or(path);
        let stem = file.split('.').next().unwrap_or(file).to_lowercase();
        if stem.len() >= 4 {
            self.stems.entry(stem.replace(['_', '-'], " ")).or_insert_with(|| path.clone());
        }
        for line in def_lines_at(&rec.text, path) {
            let mut t = line.trim_start();
            for p in ["pub(crate) ", "pub(super) ", "pub ", "export ", "default ", "async ", "unsafe ", "const fn ", "public ", "private "] {
                if let Some(r) = t.strip_prefix(p) {
                    t = r;
                }
            }
            let Some(rest) = DEF_HEADS.iter().find_map(|h| t.strip_prefix(h)) else { continue };
            let name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
            if name.len() < 3 || !name.chars().any(char::is_alphabetic) {
                continue;
            }
            self.names.entry(name.to_lowercase()).or_insert_with(|| name.clone());
            let parts = nmlex::name_parts(&name);
            if parts.len() >= 2 {
                self.split.entry(parts.join(" ")).or_insert(name);
            }
        }
    }
    /// Add another index's names (the same material read in parts).
    pub fn merge(&mut self, other: &Self) {
        for (k, v) in &other.names {
            if self.names.len() < SYMBOL_CAP {
                self.names.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
        for (k, v) in &other.split {
            self.split.entry(k.clone()).or_insert_with(|| v.clone());
        }
        for (k, v) in &other.stems {
            self.stems.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }
    /// Defined names, split names and file stems held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.names.len() + self.split.len() + self.stems.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Whether any key or symbol of the index holds `word` (lower-cased), for the ISO-R test.
    #[must_use]
    pub fn mentions(&self, word: &str) -> bool {
        let w = word.to_lowercase();
        [&self.names, &self.split, &self.stems].iter().any(|m| m.iter().any(|(k, v)| k.contains(&w) || v.to_lowercase().contains(&w)))
    }
    /// The request's named things found in the index, over its words as written: the longest
    /// match first (three words, two, one). Plain single words match a file stem only;
    /// identifiers match a defined name.
    #[must_use]
    pub fn tag(&self, words: &[String]) -> Vec<Tag> {
        let mut tags = Vec::new();
        let mut i = 0;
        while i < words.len() {
            let mut hit = None;
            for n in (2..=3).rev() {
                let Some(g) = words.get(i..i + n) else { continue };
                if g.iter().any(|w| identifier(w) || FRAME.contains(&w.to_lowercase().as_str())) {
                    continue;
                }
                let key = g.iter().map(|w| w.to_lowercase()).collect::<Vec<_>>().join(" ");
                if let Some(s) = self.split.get(&key) {
                    hit = Some((n, Tag { words: g.to_vec(), symbol: s.clone(), how: TagHow::Split }));
                    break;
                }
                if let Some(p) = self.stems.get(&key) {
                    hit = Some((n, Tag { words: g.to_vec(), symbol: p.clone(), how: TagHow::Path }));
                    break;
                }
            }
            if hit.is_none()
                && let Some(w) = words.get(i)
            {
                let l = w.to_lowercase();
                if identifier(w) {
                    if let Some(s) = self.names.get(&l) {
                        hit = Some((1, Tag { words: vec![w.clone()], symbol: s.clone(), how: TagHow::Exact }));
                    }
                } else if l.len() >= 4
                    && !FRAME.contains(&l.as_str())
                    && !verb_word(&l)
                    && let Some(p) = self.stems.get(&l)
                {
                    hit = Some((1, Tag { words: vec![w.clone()], symbol: p.clone(), how: TagHow::Path }));
                }
            }
            match hit {
                Some((n, t)) => {
                    tags.push(t);
                    i += n;
                }
                None => i += 1,
            }
        }
        tags
    }
}

static SYMBOLS: std::sync::RwLock<Option<Arc<SymbolIndex>>> = std::sync::RwLock::new(None);
/// Publish material symbols for the request's drone: merged into the index the process holds
/// (material only; called where the material's rarities are built). A stopgap until the served
/// path hands the index to `Request::read` itself (hook asked of the drone path's lane).
pub fn publish_symbols(ix: &SymbolIndex) {
    if ix.is_empty() {
        return;
    }
    if let Ok(mut g) = SYMBOLS.write() {
        let mut next = g.as_deref().cloned().unwrap_or_default();
        next.merge(ix);
        *g = Some(Arc::new(next));
    }
}
/// The material symbol index the process holds, if any.
#[must_use]
pub fn material_symbols() -> Option<Arc<SymbolIndex>> {
    SYMBOLS.read().ok().and_then(|g| g.clone())
}

/// One read of the request's drone, kept in the request's store: the request as read, the
/// reader's kind probabilities and relation yes probabilities (when the model read it), the named
/// things tagged against the material, and the milliseconds of the reads.
#[derive(Clone, Debug)]
pub struct RequestRead {
    /// A hash of the question and the router's jobs (the key).
    pub key: u64,
    pub request: Request,
    /// Each kind's score: its yes log-odds less its content-free mean.
    pub kind_probs: Vec<(&'static str, f32)>,
    /// Each relation's score, as the kinds'.
    pub relation_yes: Vec<(Asks, f32)>,
    pub tags: Vec<Tag>,
    /// The reader was used (false: the word rules' fallback).
    pub model: bool,
    /// Milliseconds of the reader's forwards (device time as the reader reports it).
    pub model_ms: f64,
    /// Milliseconds of the whole read.
    pub read_ms: f64,
}
impl Sourced for RequestRead {
    fn origin(&self) -> Origin {
        Origin::Request
    }
}

/// The request's own store: the latest reads, in order (bounded; a request is read once and its
/// read reused by every later call for it). Holds request data only.
#[derive(Debug, Default)]
pub struct RequestStore {
    reads: std::collections::VecDeque<RequestRead>,
}
/// Reads kept.
const STORE_CAP: usize = 256;
impl RequestStore {
    #[must_use]
    pub fn get(&self, key: u64) -> Option<&RequestRead> {
        self.reads.iter().rev().find(|r| r.key == key)
    }
    pub fn put(&mut self, r: RequestRead) {
        self.reads.retain(|x| x.key != r.key);
        if self.reads.len() >= STORE_CAP {
            self.reads.pop_front();
        }
        self.reads.push_back(r);
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.reads.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.reads.is_empty()
    }
}
static STORE: std::sync::Mutex<RequestStore> = std::sync::Mutex::new(RequestStore { reads: std::collections::VecDeque::new() });

/// The key of a question and its jobs (FNV-1a, 64 bits).
#[must_use]
pub fn request_key(q: &str, jobs: &[String]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in q.trim().bytes().chain(jobs.iter().flat_map(|j| std::iter::once(0u8).chain(j.bytes()))) {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}
/// The stored read of a question, if it was read in this process.
#[must_use]
pub fn stored_read(q: &str, jobs: &[String]) -> Option<RequestRead> {
    STORE.lock().ok().and_then(|s| s.get(request_key(q, jobs)).cloned())
}

/// Split each part that holds a tagged run as only part of its words: the tagged run becomes its
/// own named part, the words before and after stay parts of their own.
fn split_by_tags(r: &mut Request, tags: &[Tag]) {
    for a in &mut r.aspects {
        let mut parts: Vec<Part> = Vec::new();
        for p in std::mem::take(&mut a.parts) {
            // A single plain word naming a file is a seed only: file stems are common words here
            // (`pack`, `window`, `drone`), and splitting on them breaks named things apart.
            let hit = tags.iter().filter(|t| !(t.how == TagHow::Path && t.words.len() == 1)).find_map(|t| {
                let n = t.words.len();
                if n >= p.words.len() {
                    return None;
                }
                p.words.windows(n).position(|w| w.iter().zip(&t.words).all(|(x, y)| x.eq_ignore_ascii_case(y))).map(|at| (at, n))
            });
            match hit {
                Some((at, n)) => {
                    let w = &p.words;
                    if at > 0 {
                        parts.push(part_of(w.get(..at).unwrap_or(&[]).to_vec(), Role::Named));
                    }
                    parts.push(part_of(w.get(at..at + n).unwrap_or(&[]).to_vec(), Role::Named));
                    if at + n < w.len() {
                        parts.push(part_of(w.get(at + n..).unwrap_or(&[]).to_vec(), Role::Named));
                    }
                }
                None => parts.push(p),
            }
        }
        a.parts = parts;
    }
}

/// The yes log-odds of each `(prompt rest, yes/no)` item after `prefix`.
fn yes_log_odds(m: &dyn RequestModel, prefix: &str, items: &[(String, Vec<&'static str>)]) -> Option<(Vec<f32>, f64)> {
    let (probs, ms) = m.label_probs_shared(prefix, items).ok()?;
    (probs.len() == items.len()).then(|| {
        (probs.iter().map(|p| (p.first().copied().unwrap_or(0.5).max(1e-6) / p.get(1).copied().unwrap_or(0.5).max(1e-6)).ln()).collect(), ms)
    })
}

/// Roadmap Phase 4 (2026-11-24), T-065 and T-070: each kind's model score from the yes log-odds of
/// the kind templates (`lo`, in `KIND_TEMPLATES` order, every kind's templates in turn) and their
/// content-free means (`prior`, the same order). Per kind the mean over its templates, relative to
/// the mean over kinds, less the prior taken the same way, relative to its median over kinds and
/// capped at `MODEL_PRIOR_CAP`.
#[must_use]
pub fn model_kind_scores(lo: &[f32], prior: &[f32]) -> Vec<(&'static str, f32)> {
    let per_kind = |xs: &[f32]| {
        let mut at = 0;
        KIND_TEMPLATES
            .iter()
            .map(|(_, ts)| {
                let m = mean_of(xs.get(at..at + ts.len()).unwrap_or(&[]));
                at += ts.len();
                m
            })
            .collect::<Vec<f32>>()
    };
    let read = relative_to_kinds(&per_kind(lo));
    let base = capped_prior(&relative_to_kinds(&per_kind(prior)), MODEL_PRIOR_CAP);
    KIND_TEMPLATES.iter().zip(read.iter().zip(&base)).map(|((k, _), (r, b))| (static_kind(k), r - b)).collect()
}

/// The model form's read of `q` with the reader and the symbol index given (none: the word rules'
/// kind and relations; no tags). Every kind template and relation is one yes/no prompt after the
/// shared head. Kind: the highest `model_kind_scores` (Roadmap Phase 4 (2026-11-24): three
/// contrastive templates a kind, read relative to the other kinds, prior capped). Relations: yes
/// log-odds less the content-free mean above zero, and those the kind implies (trace and explain:
/// what it calls; value: what it reads).
#[must_use]
pub fn read_model_with(q: &str, jobs: &[String], model: Option<&dyn RequestModel>, symbols: Option<&SymbolIndex>) -> RequestRead {
    read_with_parts(q, jobs, model, None, symbols, true)
}

/// As `read_model_with`, with the embedder's kind read (`embedder`, used where the word rules are
/// unsure and no reader is given) and the finer parts on or off (`fine`).
#[must_use]
pub fn read_with_parts(q: &str, jobs: &[String], model: Option<&dyn RequestModel>, embedder: Option<&dyn TextEmbedder>, symbols: Option<&SymbolIndex>, fine: bool) -> RequestRead {
    let t = std::time::Instant::now();
    let mut kind_probs = Vec::new();
    let mut relation_yes = Vec::new();
    let mut model_ms = 0.0;
    if let Some(m) = model {
        let items: Vec<(String, Vec<&'static str>)> = KIND_TEMPLATES
            .iter()
            .flat_map(|(_, ts)| ts.iter().copied())
            .chain(RELATION_TEMPLATES.iter().map(|(_, t)| *t))
            .map(|t| (yes_no_suffix(t), vec!["yes", "no"]))
            .collect();
        let kind_items = items.len() - RELATION_TEMPLATES.len();
        let who = std::ptr::from_ref(m).cast::<()>() as usize;
        let held = PRIOR.lock().ok().and_then(|g| g.as_ref().filter(|(w, _)| *w == who).map(|(_, p)| p.clone()));
        let prior = held.or_else(|| {
            let mut sum = vec![0.0_f32; items.len()];
            for n in NEUTRAL {
                let (lo, _) = yes_log_odds(m, &request_prefix(n), &items)?;
                sum.iter_mut().zip(&lo).for_each(|(s, x)| *s += x / NEUTRAL.len() as f32);
            }
            if let Ok(mut g) = PRIOR.lock() {
                *g = Some((who, sum.clone()));
            }
            Some(sum)
        });
        if let Some((lo, ms)) = yes_log_odds(m, &request_prefix(q), &items) {
            model_ms = ms;
            let off = std::env::var("NANO_REQ_CALIB").as_deref() == Ok("off");
            let p: Vec<f32> = (0..lo.len()).map(|i| if off { 0.0 } else { prior.as_ref().and_then(|p| p.get(i)).copied().unwrap_or(0.0) }).collect();
            kind_probs = model_kind_scores(lo.get(..kind_items).unwrap_or(&[]), p.get(..kind_items).unwrap_or(&[]));
            relation_yes = RELATION_TEMPLATES.iter().map(|(a, _)| *a).zip(lo.iter().zip(&p).skip(kind_items).map(|(x, b)| x - b)).collect();
        }
    }
    let used = !kind_probs.is_empty();
    let mut kind = kind_probs.iter().copied().max_by(|a, b| a.1.total_cmp(&b.1)).map(|(k, _)| k);
    let mut embedded = false;
    if !used
        && let Some(e) = embedder
        && let Some((scores, ms)) = embed_kind_scores(q, e)
    {
        let (wk, sure) = word_kind_of(q, jobs);
        kind = Some(if sure { wk } else { scores.iter().copied().max_by(|a, b| a.1.total_cmp(&b.1)).map_or(wk, |(k, _)| k) });
        kind_probs = scores;
        model_ms = ms;
        embedded = true;
    }
    let mut request = Request::read_with(q, jobs, fine, kind);
    if used {
        let mut rels: Vec<Asks> = Vec::new();
        for a in [Asks::Callers, Asks::Callees, Asks::Reads] {
            let yes = relation_yes.iter().any(|(r, s)| *r == a && *s > 0.0);
            let implied = match a {
                Asks::Callers => false,
                Asks::Callees => matches!(request.kind, "trace" | "explain"),
                Asks::Reads => request.kind == "value",
            };
            if yes || implied {
                rels.push(a);
            }
        }
        request.relations = rels;
    }
    let mut tags = Vec::new();
    if let Some(ix) = symbols.filter(|_| fine) {
        let words: Vec<String> = q
            .split_whitespace()
            .map(|r| r.trim_matches(|c: char| !(c.is_alphanumeric() || c == '_')))
            .map(|w| w.strip_suffix("'s").unwrap_or(w).to_owned())
            .filter(|w| !w.is_empty())
            .collect();
        tags = ix.tag(&words);
        split_by_tags(&mut request, &tags);
    }
    RequestRead { key: request_key(q, jobs), request, kind_probs, relation_yes, tags, model: used || embedded, model_ms, read_ms: t.elapsed().as_secs_f64() * 1000.0 }
}

/// The model form as served: the stored read when the request was read before, else a read with
/// the process's reader and material index, then stored.
fn read_model_form(q: &str, jobs: &[String]) -> Request {
    let key = request_key(q, jobs);
    if let Some(r) = STORE.lock().ok().and_then(|s| s.get(key).filter(|r| r.model || (MODEL.get().is_none() || !model_form_on()) && (EMBEDDER.get().is_none() || !embed_kind_on())).map(|r| r.request.clone())) {
        return r;
    }
    let symbols = material_symbols();
    let model = MODEL.get().filter(|_| model_form_on()).map(|m| m.as_ref() as &dyn RequestModel);
    let embedder = EMBEDDER.get().filter(|_| embed_kind_on()).map(|e| e.as_ref() as &dyn TextEmbedder);
    let read = read_with_parts(q, jobs, model, embedder, symbols.as_deref(), fine_parts_on());
    let out = read.request.clone();
    if let Ok(mut s) = STORE.lock() {
        s.put(read);
    }
    out
}

/// Test ISO-R: request-derived data never appears in material stores, indexes or maps. Material
/// records, their rarities and their symbol index are built; a request naming an unseen word is
/// read in the model form (a stand-in reader) and stored; the material's index, rarities and
/// records are then compared with copies taken before, and searched for the request's word.
/// # Errors
/// One fault line naming the violation.
pub fn check_iso_r() -> Result<(), String> {
    struct Chars;
    impl Tokens for Chars {
        fn tokens(&self, text: &str) -> Result<Vec<i32>, Fault> {
            Ok(text.bytes().map(|b| i32::from(b % 4)).collect())
        }
    }
    struct Stand;
    impl RequestModel for Stand {
        fn label_probs(&self, items: &[(String, Vec<&'static str>)]) -> Result<(Vec<Vec<f32>>, f64), Fault> {
            // For the test's question: yes to the trace prompts and every relation; else even.
            // Roadmap Phase 4 (2026-11-24): keyed on the trace kind's templates, now three.
            let trace = KIND_TEMPLATES.iter().filter(|(k, _)| *k == "trace").flat_map(|(_, ts)| ts.iter());
            let yes = |p: &str| p.contains("zqxv") && (trace.clone().any(|t| p.contains(t)) || RELATION_TEMPLATES.iter().any(|(_, t)| p.contains(t)));
            Ok((items.iter().map(|(p, _)| if yes(p) { vec![0.9, 0.1] } else { vec![0.5, 0.5] }).collect(), 0.0))
        }
    }
    let table = Bf16Table::from_rows(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![-1.0, 0.0]]);
    let drone = Drone::new(Arc::new(Chars), table);
    let mut recs = Vec::new();
    for (i, (path, text)) in [("src/first_fit.rs", "pub fn first_fit(order: &[usize]) -> Vec<usize> { vec![] }\n"), ("src/coordinator.rs", "pub struct Budgets { pack: usize }\n")]
        .iter()
        .enumerate()
    {
        let s = drone.take_in(Kind::Code, Source::Lines { path: (*path).into(), start: 1, end: 1 }, i, text).map_err(|e| e.to_string())?;
        recs.push(s.into_inner());
    }
    let ix = SymbolIndex::from_records(recs.iter());
    let rarity = Rarity::from_records(recs.iter());
    let (ix0, words0, bag0) = (ix.clone(), rarity.words.clone(), rarity.bag_df.clone());
    let texts0: Vec<String> = recs.iter().map(|r| r.text.clone()).collect();
    let q = "How does the coordinator reach zqxvwobble through first fit?";
    let read = read_model_with(q, &[], Some(&Stand), Some(&ix));
    if read.request.kind != "trace" || !read.request.relations.contains(&Asks::Callers) || !read.model {
        return Err(format!("nmpod.intake ISO-R: model form read {} {:?}", read.request.kind, read.request.relations));
    }
    let tagged: Vec<&str> = read.tags.iter().map(|t| t.symbol.as_str()).collect();
    if tagged != ["src/coordinator.rs", "first_fit"] {
        return Err(format!("nmpod.intake ISO-R: tags {tagged:?}"));
    }
    let mut store = RequestStore::default();
    store.put(read);
    let mut violations = Vec::new();
    if ix != ix0 || ix.mentions("zqxvwobble") || ix.mentions("reach") {
        violations.push("symbol index");
    }
    if rarity.words != words0 || rarity.bag_df != bag0 || rarity.words.contains_key("zqxvwobble") {
        violations.push("rarities");
    }
    if recs.iter().map(|r| r.text.clone()).collect::<Vec<_>>() != texts0 || recs.iter().any(|r| r.words.contains("zqxvwobble")) {
        violations.push("records");
    }
    if store.get(request_key(q, &[])).is_none_or(|r| r.origin() != Origin::Request) {
        violations.push("request store");
    }
    // The material inlet refuses the stored read's request as it refuses any request.
    if store.get(request_key(q, &[])).is_some_and(|r| inlets::MATERIAL_DRONES.admit(&r.request).is_ok()) {
        violations.push("material inlet admitted the request");
    }
    if violations.is_empty() { Ok(()) } else { Err(format!("nmpod.intake ISO-R: request data in {}", violations.join(", "))) }
}

/// Section check: a two-word table and a character tokenizer; the piece that defines the
/// request's identifier comes first and nothing is dropped.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    struct Chars;
    impl Tokens for Chars {
        fn tokens(&self, text: &str) -> Result<Vec<i32>, Fault> {
            Ok(text.bytes().map(|b| i32::from(b % 4)).collect())
        }
    }
    if split_words("ParMap HTTPServer snake_case") != ["parmap", "par", "map", "httpserver", "http", "server", "snake", "case"] {
        return Err(format!("nmpod.intake: split_words {:?}", split_words("ParMap HTTPServer snake_case")));
    }
    let table = Bf16Table::from_rows(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![-1.0, 0.0]]);
    let drone = Drone::new(Arc::new(Chars), table);
    let pieces = vec![
        ("a.rs".to_owned(), 1, 1, "fn other() {}".to_owned()),
        ("b.rs".to_owned(), 1, 2, "pub struct ParMap { x: u8 }\n".to_owned()),
        ("c.rs".to_owned(), 1, 1, "let y = 2;".to_owned()),
    ];
    let h = window_pieces(&drone, "Where is ParMap defined?", &pieces).map_err(|e| e.to_string())?;
    if h.len() != 3 {
        return Err("nmpod.intake: a piece was dropped".into());
    }
    if h.first().map(|x| x.record.get().position) != Some(2) {
        return Err("nmpod.intake: the defining piece is not first".into());
    }
    if Mode::default() != Mode::Off {
        return Err("nmpod.intake: not off by default".into());
    }
    // The request and the material never cross: a request record among the pieces is refused,
    // and the request's drone takes no material record.
    let req = drone.take_request("request", "Where is ParMap defined?").map_err(|e| e.to_string())?;
    let rarity = Rarity::default();
    let stray = drone.take_in(Kind::Request, Source::Message { label: "q".into() }, 1, "Where is ParMap defined?").map_err(|e| e.to_string())?;
    match hand_off(&drone, &req, &rarity, vec![stray]) {
        Err(f) if f.message.starts_with(nmtype::class::WRONG_ORIGIN) => {}
        other => return Err(format!("nmpod.intake: a request record was taken as material: {:?}", other.map(|h| h.len()))),
    }
    let piece = drone.take_in(Kind::Code, Source::Lines { path: "b.rs".into(), start: 1, end: 1 }, 1, "pub struct ParMap;").map_err(|e| e.to_string())?;
    if inlets::REQUEST_DRONE.admit(piece.get()).is_ok() || inlets::MATERIAL_DRONES.admit(&Request::read("Where is ParMap?", &[])).is_ok() {
        return Err("nmpod.intake: a material record was taken as the request, or the request as material".into());
    }
    // The request's drone: kind and aspects.
    let r = Request::read_words("Where is the growth ratio limit set, and which function reads max_ratio?", &[], false);
    let texts: Vec<&str> = r.aspects.iter().map(|a| a.text.as_str()).collect();
    let parts: Vec<(&str, Role)> = r.aspects.iter().flat_map(|a| a.parts.iter().map(|p| (p.text.as_str(), p.role))).collect();
    if r.kind != "value"
        || texts != ["growth ratio limit", "reads max_ratio"]
        || parts != [("growth ratio limit", Role::Named), ("reads", Role::Asked), ("max_ratio", Role::Named)]
        || r.relations != [Asks::Reads]
        || !r.aspects.get(1).is_some_and(|a| a.stems.contains(&"ratio".to_owned()))
    {
        return Err(format!("nmpod.intake: request {} {texts:?} {parts:?} {:?}", r.kind, r.relations));
    }
    let t = Request::read_words("Trace an nm_patch_propose call from the finding index to the recorded proposal.", &[], false);
    let texts: Vec<&str> = t.aspects.iter().map(|a| a.text.as_str()).collect();
    if t.kind != "trace"
        || texts != ["nm_patch_propose", "finding index", "recorded proposal"]
        || t.aspects.first().is_none_or(|a| a.form != Form::Identifier)
        || t.relations != [Asks::Callees]
    {
        return Err(format!("nmpod.intake: trace request {} {texts:?} {:?}", t.kind, t.relations));
    }
    let m = Request::read_words("By default, where does the MCP server answer the initialize request?", &[], false);
    let parts: Vec<&str> = m.aspects.iter().flat_map(|a| a.parts.iter().map(|p| p.text.as_str())).collect();
    if m.aspects.len() != 1 || parts != ["default MCP server answer", "initialize request"] {
        return Err(format!("nmpod.intake: parts {parts:?}"));
    }
    if r.covered_by("fn guard(l: &Config) { if l.max_ratio > 2 {} }") != [false, true] {
        return Err("nmpod.intake: aspect coverage".into());
    }
    check_iso_r()
}

// Roadmap Phase 4 (2026-11-24): unit checks of the request drone's two kind reads (T-065, T-070,
// T-071). No question of any fold is used; the vectors and reads are made up.
#[cfg(test)]
mod kind_read_tests {
    use super::*;

    const KINDS: [&str; 7] = ["explain", "locate", "trace", "change", "value", "prose", "absent-check"];

    #[test]
    fn kind_tables_have_the_same_kinds_and_three_entries_or_more_each() {
        let model: Vec<&str> = KIND_TEMPLATES.iter().map(|(k, _)| *k).collect();
        let embed: Vec<&str> = KIND_DESCRIPTIONS.iter().map(|(k, _)| *k).collect();
        assert_eq!(model, KINDS);
        assert_eq!(embed, KINDS);
        for (k, ts) in &KIND_TEMPLATES {
            assert!(ts.len() >= 3, "{k} has {} templates", ts.len());
            assert!(ts.iter().all(|t| t.ends_with('?')), "{k}: a template is not a question");
        }
        for (k, ds) in &KIND_DESCRIPTIONS {
            assert!(ds.len() >= 3, "{k} has {} descriptions", ds.len());
        }
    }

    #[test]
    fn a_low_prior_alone_does_not_win_the_model_read() {
        // Flat reads except trace, 1.5 above the rest on its every template; change has a far
        // lower content-free prior (the T-065 case).
        let mut lo = Vec::new();
        let mut prior = Vec::new();
        for (k, ts) in &KIND_TEMPLATES {
            for _ in *ts {
                lo.push(if *k == "trace" { 6.5 } else { 5.0 });
                prior.push(if *k == "change" { -6.0 } else { 3.0 });
            }
        }
        let s = model_kind_scores(&lo, &prior);
        let best = s.iter().copied().max_by(|a, b| a.1.total_cmp(&b.1)).map(|(k, _)| k);
        assert_eq!(best, Some("trace"), "{s:?}");
        // The prior moves change by at most the cap.
        let flat = model_kind_scores(&lo, &vec![0.0; lo.len()]);
        let change = |v: &[(&str, f32)]| v.iter().find(|(k, _)| *k == "change").map_or(0.0, |x| x.1);
        assert!(change(&s) - change(&flat) <= MODEL_PRIOR_CAP + 1e-5, "{s:?} {flat:?}");
        // Reads equal on every kind give equal scores, whatever the reader's yes level.
        let even = model_kind_scores(&vec![9.0; lo.len()], &vec![9.0; lo.len()]);
        assert!(even.iter().all(|(_, x)| x.abs() < 1e-5), "{even:?}");
    }

    #[test]
    fn a_description_shared_by_every_kind_carries_no_weight() {
        // Three kinds on three axes; the last description of each kind is the same vector.
        let shared = vec![1.0, 1.0, 1.0];
        let axis = |i: usize| (0..3).map(|j| if j == i { 1.0 } else { 0.0 }).collect::<Vec<f32>>();
        let descs: Vec<Vec<Vec<f32>>> = (0..3).map(|i| vec![axis(i), axis(i), shared.clone()]).collect();
        let w = description_weights(&descs);
        for k in &w {
            assert!(k.first().is_some_and(|x| *x > 0.0) && k.get(2).is_some_and(|x| *x == 0.0), "{w:?}");
        }
        // A question on the second axis reads as the second kind; the shared vector reads even.
        let r = kind_read(&axis(1), &descs, &w);
        assert!(r.get(1).is_some_and(|x| r.iter().all(|y| x >= y)), "{r:?}");
        let flat = kind_read(&shared, &descs, &w);
        assert!(flat.iter().all(|x| x.abs() < 1e-5), "{flat:?}");
    }

    #[test]
    fn the_baseline_is_capped_and_median_centred() {
        let b = capped_prior(&[0.0, 0.0, -5.0, 0.0, 4.0], 1.0);
        assert_eq!(b, vec![0.0, 0.0, -1.0, 0.0, 1.0]);
    }

    #[test]
    fn section_check_holds() {
        assert_eq!(check(), Ok(()));
    }

    #[test]
    fn sense_records_matches_sequential_probe_and_caches_rows() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct ByteTok;
        impl Tokens for ByteTok {
            fn tokens(&self, text: &str) -> Result<Vec<i32>, Fault> {
                Ok(text.bytes().map(|b| i32::from(b % 6)).collect())
            }
        }
        struct CountingRows {
            inner: Arc<Bf16Table>,
            reads: AtomicUsize,
        }
        impl EmbeddingRows for CountingRows {
            fn width(&self) -> usize {
                self.inner.width()
            }
            fn rows(&self) -> usize {
                self.inner.rows()
            }
            fn row(&self, id: usize, out: &mut [f32]) -> bool {
                self.reads.fetch_add(1, Ordering::Relaxed);
                self.inner.row(id, out)
            }
        }
        let table = Arc::new(CountingRows {
            inner: Bf16Table::from_rows(&[
                vec![1.0, 0.25, -0.5],
                vec![0.0, 1.0, 0.5],
                vec![0.5, 0.5, 0.5],
                vec![-1.0, 0.2, 0.8],
                vec![0.0, 0.0, 0.0],
                vec![0.3, -0.7, 0.9],
            ]),
            reads: AtomicUsize::new(0),
        });
        let drone = Drone::new(Arc::new(ByteTok), Arc::clone(&table) as Arc<dyn EmbeddingRows>);
        let items = [
            ("src/alpha.rs", "pub fn derive_window_limit() -> usize { 42 }"),
            ("src/beta.rs", "fn other_helper() { let x = 7; }"),
            (".hidden/cfg.toml", "secret_key = 999"),
        ];
        let records: Vec<Record> = items
            .iter()
            .enumerate()
            .map(|(i, (path, text))| {
                drone
                    .take_in(Kind::Code, Source::Lines { path: (*path).into(), start: 1, end: 1 }, i + 1, text)
                    .map(Sensory::into_inner)
                    .unwrap_or_else(|e| panic!("take_in failed: {e}"))
            })
            .collect();
        let rarity = Rarity::from_records(records.iter());
        let req = drone.take_request("request", "Where is derive_window_limit defined?").unwrap_or_else(|e| panic!("take_request failed: {e}"));
        let mut seq_probe = drone.probe_request(&req, &rarity);
        let expected: Vec<Signals> = records.iter().map(|r| seq_probe.sense(r)).collect();
        let fast1 = drone.sense_records(&req, &records, &rarity).unwrap_or_else(|e| panic!("sense_records 1 failed: {e}"));
        assert_eq!(fast1, expected);
        let after_first = table.reads.load(Ordering::Relaxed);
        let fast2 = drone.sense_records(&req, &records, &rarity).unwrap_or_else(|e| panic!("sense_records 2 failed: {e}"));
        assert_eq!(fast2, expected);
        // Only the request tokens read inside `probe_request` + any hidden fallback token may read;
        // the cached material rows in `rarity.token_ids()` are never decoded again.
        let delta = table.reads.load(Ordering::Relaxed) - after_first;
        assert!(delta <= req.get().distinct.len() + 2, "expected cached material rows on warm call, got {delta} extra reads");
    }
}
