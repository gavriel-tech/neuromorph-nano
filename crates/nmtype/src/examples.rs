//! The training store's contract: where the Nano keeps, per project, the data future cartridges are
//! fitted from, the shape of its rows, the fixed split, the reader a fitting job calls, and the tap a
//! serving path hands the reranker's numbers to.
//!
//! The store is `<project memory>/training/`. Learning examples (`examples.jsonl`), preference
//! signals (`preferences.jsonl`) and hardware measurements (`hardware.json`) are separate files with
//! separate family tags and are never mixed. Rows hold places, content hashes, labels, the models'
//! numbers and provenance; text (questions and passages) only when the `examples` setting is `full`,
//! passages then in `passages.jsonl`, once per content hash. The models' numbers are tied to an
//! engine identity (`engines.json`: weight hash, precision, arithmetic switches); last-layer values,
//! when a serving path supplies them, sit in `views/<engine>.bin`, one file per engine, so numbers
//! from two engines are never in one file. Reference material for fundamentals cartridges is listed
//! (never copied) in the Nano's own memory folder, `memory/training/references.json`.
//!
//! The writer is the program (`neuromorph-nano`); this module only reads, plus the tap and the split.
use nmjson::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering};

/// The store's folder in a project memory folder.
pub const DIR: &str = "training";
pub const EXAMPLES: &str = "examples.jsonl";
pub const JUDGED: &str = "judged.jsonl";
pub const PREFERENCES: &str = "preferences.jsonl";
pub const HARDWARE: &str = "hardware.json";
pub const PASSAGES: &str = "passages.jsonl";
pub const ENGINES: &str = "engines.json";
pub const VIEWS: &str = "views";
pub const REFERENCES: &str = "references.json";
/// The row format version written in every row (`v`).
pub const FORMAT: u64 = 1;
/// The first bytes of a views file.
pub const VIEWS_MAGIC: &[u8; 8] = b"NMVIEW1\n";

/// The store folder of a project memory folder.
#[must_use]
pub fn dir(memory: &Path) -> PathBuf {
    memory.join(DIR)
}

// ---------------------------------------------------------------------------------------------
// The split.

/// Which use an example is kept for, fixed when it is captured.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Split {
    Fit,
    Tune,
    Held,
}

impl Split {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Fit => "fit",
            Self::Tune => "tune",
            Self::Held => "held-out",
        }
    }
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "fit" => Some(Self::Fit),
            "tune" => Some(Self::Tune),
            "held-out" | "held" => Some(Self::Held),
            _ => None,
        }
    }
}

/// The split of an example from the hash of its question and its anchor place (a path): 80 in 100
/// fit, 10 tune, 10 held out. The same question and place always land in the same split, so a
/// held-out question is never seen by a fit, whichever run captures it.
#[must_use]
pub fn split_of(ask_sha: &str, anchor: &str) -> Split {
    let h = nmhash::sha256_bytes(format!("{ask_sha}|{anchor}").as_bytes());
    let k = u32::from_str_radix(h.get(..8).unwrap_or("0"), 16).unwrap_or(0) % 100;
    match k {
        0..80 => Split::Fit,
        80..90 => Split::Tune,
        _ => Split::Held,
    }
}

/// Rank of an evidence strength: direct 3, indirect 2, weak 1, anything else 0.
#[must_use]
pub fn strength_rank(s: &str) -> u8 {
    match s {
        "direct" => 3,
        "indirect" => 2,
        "weak" => 1,
        _ => 0,
    }
}

/// The content hash of a place's lines: the first 16 hex digits of the SHA-256 of the lines joined
/// by line ends (the prefix of the hash memory keeps for the same lines).
#[must_use]
pub fn content_hash(lines: &[&str]) -> String {
    nmhash::sha256_bytes(lines.join("\n").as_bytes()).chars().take(16).collect()
}

// ---------------------------------------------------------------------------------------------
// The tap: what a serving path hands over from a call it already made.

/// Capture levels, as the `examples` setting names them.
pub const OFF: u8 = 0;
pub const FEATURES: u8 = 1;
pub const FULL: u8 = 2;

static LEVEL: AtomicU8 = AtomicU8::new(OFF);
static TAPPED: Mutex<Option<Tapped>> = Mutex::new(None);

/// One judged candidate as the serving path saw it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TapCand {
    pub path: String,
    pub start: usize,
    pub end: usize,
    /// The reranker's P(yes), exactly as computed.
    pub score: f32,
    /// The passage text the judge read, kept only at level `FULL`.
    pub text: Option<String>,
}

/// The judged set of the last call: the question, every candidate with its score, whether the
/// scores are the reranker's (false: it failed and arrival order stood in), and when.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tapped {
    pub question: String,
    pub cands: Vec<TapCand>,
    pub scored: bool,
    pub at_ms: u64,
}

/// Sets the capture level (`OFF`, `FEATURES`, `FULL`); the serving process sets it from the setting.
pub fn set_level(level: u8) {
    LEVEL.store(level.min(FULL), Ordering::Relaxed);
}

/// The capture level now.
#[must_use]
pub fn level() -> u8 {
    LEVEL.load(Ordering::Relaxed)
}

/// Hands over the judged set of a call from values the call already computed: `cands` are (path,
/// first line, last line, text) in the order of `scores`. Nothing is kept at level `OFF`; the text
/// only at level `FULL`. A set not taken before the next call is replaced by it.
pub fn tap<'a, I: IntoIterator<Item = (&'a str, usize, usize, &'a str)>>(question: &str, cands: I, scores: &[f32], scored: bool) {
    let lv = level();
    if lv == OFF {
        return;
    }
    let cands = cands
        .into_iter()
        .zip(scores)
        .map(|((path, start, end, text), score)| TapCand {
            path: path.replace('\\', "/"),
            start,
            end,
            score: *score,
            text: (lv == FULL).then(|| text.to_owned()),
        })
        .collect();
    if let Ok(mut g) = TAPPED.lock() {
        *g = Some(Tapped { question: question.to_owned(), cands, scored, at_ms: crate::now_ms() });
    }
}

/// Takes the judged set the last call handed over, if any.
#[must_use]
pub fn take_tap() -> Option<Tapped> {
    TAPPED.lock().ok().and_then(|mut g| g.take())
}

// ---------------------------------------------------------------------------------------------
// Binary16 numbers for the views files.

/// The binary16 bits nearest to `x` (round to nearest, ties to even; overflow to infinity).
#[must_use]
pub fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = u16::try_from((b >> 16) & 0x8000).unwrap_or(0);
    let exp = i32::try_from((b >> 23) & 0xff).unwrap_or(0);
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man == 0 { 0 } else { 0x200 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = u32::try_from(14 - e).unwrap_or(31);
        let half = 1u32 << (shift - 1);
        let rest = m & ((1u32 << shift) - 1);
        let mut v = m >> shift;
        if rest > half || (rest == half && v & 1 == 1) {
            v += 1;
        }
        return sign | u16::try_from(v).unwrap_or(0);
    }
    let rest = man & 0x1fff;
    let mut v = (u32::try_from(e).unwrap_or(0) << 10) | (man >> 13);
    if rest > 0x1000 || (rest == 0x1000 && v & 1 == 1) {
        v += 1;
    }
    sign | u16::try_from(v.min(0x7c00)).unwrap_or(0x7c00)
}

/// The value of binary16 bits.
#[must_use]
pub fn f16_value(h: u16) -> f32 {
    let sign = if h & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let man = f32::from(h & 0x3ff);
    match exp {
        0 => sign * man * 2f32.powi(-24),
        0x1f if man == 0.0 => sign * f32::INFINITY,
        0x1f => f32::NAN,
        e => sign * (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    }
}

// ---------------------------------------------------------------------------------------------
// The reader.

fn text(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::string).ok().map(str::to_owned)
}
fn strs(v: &Value, k: &str) -> Vec<String> {
    v.get(k).and_then(Value::array).unwrap_or_default().iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect()
}
fn float(v: &Value, k: &str) -> Option<f64> {
    match v.get(k) {
        Ok(Value::Number(n)) => n.parse().ok(),
        _ => None,
    }
}

/// Every row of one store file (`examples.jsonl`, `judged.jsonl`, `preferences.jsonl`,
/// `passages.jsonl`); an empty list when the file is not there. A line that does not parse (a write
/// cut off by a stop) is passed over.
/// # Errors
/// The file exists and cannot be read.
pub fn read_rows(memory: &Path, file: &str) -> Result<Vec<Value>, String> {
    let p = dir(memory).join(file);
    match std::fs::read_to_string(&p) {
        Ok(t) => Ok(t.lines().filter(|l| !l.trim().is_empty()).filter_map(|l| nmjson::parse(l.as_bytes()).ok()).collect()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(format!("{}: {e}", p.display())),
    }
}

/// One candidate of an example: a place, its content hash, whether the judge read it and whether it
/// was served, the reranker's score when it ran, the label the evidence gave it and its strength,
/// and the passage text when it was kept (`full`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Candidate {
    pub path: String,
    pub start: u64,
    pub end: u64,
    pub hash: Option<String>,
    pub judged: bool,
    pub served: bool,
    /// Place in the reply (0 first) when served.
    pub rank: Option<u64>,
    /// The reranker's P(yes), exact (from its recorded bits) when the serving path handed it over.
    pub score: Option<f32>,
    /// `verified`, `refuted`, `used`, `unused`, `passed-over`, `went`, or `judged` (read by the judge,
    /// not served, no evidence about it).
    pub label: String,
    pub strength: Option<String>,
    pub text: Option<String>,
}

/// One learning example.
#[derive(Clone, Debug)]
pub struct Example {
    pub id: String,
    pub pack: String,
    pub family: String,
    pub domains: Vec<String>,
    pub split: Option<Split>,
    /// `verified`, `refuted`, `corrected` (the Nano was wrong; the place Claude Code went to was then
    /// verified) or `passed-over` (went elsewhere, nothing verified).
    pub verdict: String,
    pub strength: String,
    pub ask_sha: String,
    pub question: Option<String>,
    /// The engine identity the scores were computed under (`engines.json`); `None` on the plain path.
    pub engine: Option<String>,
    /// The serving path taken: `plain`, `fast` or `full`.
    pub path_taken: String,
    /// Cartridges switched on and cartridge library stacks in use when the pack was served.
    pub cartridges: Vec<String>,
    pub stacks: Vec<String>,
    pub at: u64,
    pub project: String,
    pub candidates: Vec<Candidate>,
    /// The row as stored, for fields not typed here (settings, hardware).
    pub raw: Value,
}

impl Example {
    /// The typed form of one `examples.jsonl` row.
    #[must_use]
    pub fn from_value(v: &Value, passages: &BTreeMap<String, String>) -> Self {
        let candidates = v
            .get("candidates")
            .and_then(Value::array)
            .unwrap_or_default()
            .iter()
            .map(|c| {
                let (path, span) = place(&text(c, "place").unwrap_or_default());
                let hash = text(c, "hash");
                let bits = text(c, "bits").and_then(|b| u32::from_str_radix(&b, 16).ok()).map(f32::from_bits);
                #[expect(clippy::cast_possible_truncation, reason = "a score stored from an f32")]
                let score = bits.or_else(|| float(c, "score").map(|x| x as f32));
                Candidate {
                    path,
                    start: span.0,
                    end: span.1,
                    text: hash.as_ref().and_then(|h| passages.get(h).cloned()),
                    hash,
                    judged: matches!(c.get("judged"), Ok(Value::Bool(true))),
                    served: matches!(c.get("served"), Ok(Value::Bool(true))),
                    rank: c.get("rank").and_then(Value::integer).ok(),
                    score,
                    label: text(c, "label").unwrap_or_default(),
                    strength: text(c, "strength"),
                }
            })
            .collect();
        Self {
            id: text(v, "id").unwrap_or_default(),
            pack: text(v, "pack").unwrap_or_default(),
            family: text(v, "family").unwrap_or_default(),
            domains: strs(v, "domains"),
            split: text(v, "split").and_then(|s| Split::parse(&s)),
            verdict: text(v, "verdict").unwrap_or_default(),
            strength: text(v, "strength").unwrap_or_default(),
            ask_sha: text(v, "ask_sha").unwrap_or_default(),
            question: text(v, "question"),
            engine: text(v, "engine"),
            path_taken: text(v, "path").unwrap_or_default(),
            cartridges: strs(v, "cartridges"),
            stacks: strs(v, "stacks"),
            at: v.get("at").and_then(Value::integer).unwrap_or(0),
            project: text(v, "project").unwrap_or_default(),
            candidates,
            raw: v.clone(),
        }
    }
    /// The candidates a fit takes as right answers: verified places, and the place Claude Code went
    /// to when that was verified.
    #[must_use]
    pub fn positives(&self) -> Vec<&Candidate> {
        self.candidates.iter().filter(|c| c.label == "verified" || (c.label == "went" && c.strength.as_deref() == Some("indirect"))).collect()
    }
    /// The candidates a fit takes as hard negatives: served places that were refuted or passed over.
    #[must_use]
    pub fn hard_negatives(&self) -> Vec<&Candidate> {
        self.candidates.iter().filter(|c| c.served && matches!(c.label.as_str(), "refuted" | "passed-over")).collect()
    }
}

/// A place `path:a-b` (or `path:a`) as its path and lines; a whole file is (0, 0).
#[must_use]
pub fn place(p: &str) -> (String, (u64, u64)) {
    if let Some((path, r)) = p.rsplit_once(':') {
        let span = match r.split_once('-') {
            Some((a, b)) => a.parse().ok().zip(b.parse().ok()),
            None => r.parse().ok().map(|a| (a, a)),
        };
        if let Some(sp) = span {
            return (path.to_owned(), sp);
        }
    }
    (p.to_owned(), (0, 0))
}

/// Which examples a fitting job wants.
#[derive(Clone, Debug, Default)]
pub struct Filter {
    /// `learning` for learning cartridges (the only family in `examples.jsonl`).
    pub family: Option<String>,
    pub split: Option<Split>,
    /// Only examples whose scores were computed under this engine identity. Examples of two engines
    /// are never mixed in one fit: pass the engine the cartridge is fitted for.
    pub engine: Option<String>,
    /// The weakest example strength taken (`strength_rank`); 0 takes all.
    pub min_strength: u8,
    /// Only examples with this domain tag (`code:rust`, `ml-architecture`, `project:<key>`).
    pub domain: Option<String>,
    /// Verdicts taken; empty takes all.
    pub verdicts: Vec<String>,
}

/// The learning examples of a project memory folder that pass `f`, oldest first, with the passage
/// text joined in where it was kept.
/// # Errors
/// A store file exists and cannot be read.
pub fn read_examples(memory: &Path, f: &Filter) -> Result<Vec<Example>, String> {
    let passages: BTreeMap<String, String> = read_rows(memory, PASSAGES)?.iter().filter_map(|p| Some((text(p, "hash")?, text(p, "text")?))).collect();
    let mut out: Vec<Example> = read_rows(memory, EXAMPLES)?
        .iter()
        .map(|v| Example::from_value(v, &passages))
        .filter(|e| f.family.as_ref().is_none_or(|x| &e.family == x))
        .filter(|e| f.split.is_none_or(|s| e.split == Some(s)))
        .filter(|e| f.engine.is_none() || e.engine == f.engine)
        .filter(|e| strength_rank(&e.strength) >= f.min_strength)
        .filter(|e| f.domain.as_ref().is_none_or(|d| e.domains.contains(d)))
        .filter(|e| f.verdicts.is_empty() || f.verdicts.contains(&e.verdict))
        .collect();
    out.sort_by_key(|e| e.at);
    Ok(out)
}

/// The preference signals of a project memory folder (family `preference`), oldest first.
/// # Errors
/// As `read_rows`.
pub fn read_preferences(memory: &Path) -> Result<Vec<Value>, String> {
    let mut v = read_rows(memory, PREFERENCES)?;
    v.sort_by_key(|r| r.get("at").and_then(Value::integer).unwrap_or(0));
    Ok(v)
}

/// The hardware measurements of a project memory folder (family `fundamentals`, domain `hardware`).
/// # Errors
/// The file exists and cannot be read or parsed.
pub fn read_hardware(memory: &Path) -> Result<Option<Value>, String> {
    let p = dir(memory).join(HARDWARE);
    match std::fs::read(&p) {
        Ok(b) => nmjson::parse(&b).map(Some).map_err(|e| format!("{}: {e}", p.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", p.display())),
    }
}

/// The engine identities of a project's store: id to its description (weight hash, precision,
/// arithmetic switches).
/// # Errors
/// The file exists and cannot be read or parsed.
pub fn read_engines(memory: &Path) -> Result<BTreeMap<String, Value>, String> {
    let p = dir(memory).join(ENGINES);
    match std::fs::read(&p) {
        Ok(b) => match nmjson::parse(&b).map_err(|e| format!("{}: {e}", p.display()))? {
            Value::Object(m) => Ok(m),
            _ => Err(format!("{}: not an object", p.display())),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
        Err(e) => Err(format!("{}: {e}", p.display())),
    }
}

/// One views record: the pack, the candidate's index in its judged set, the last-layer values.
pub type View = (String, u32, Vec<f32>);

/// Encodes one views record: pack id length (u16) and bytes, candidate index (u32), width (u32),
/// then the values as binary16, all little-endian.
#[must_use]
pub fn view_record(pack: &str, index: u32, values: &[f32]) -> Vec<u8> {
    let id = pack.as_bytes();
    let mut b = Vec::with_capacity(10 + id.len() + values.len() * 2);
    b.extend_from_slice(&u16::try_from(id.len()).unwrap_or(u16::MAX).to_le_bytes());
    b.extend_from_slice(id.get(..usize::from(u16::MAX).min(id.len())).unwrap_or_default());
    b.extend_from_slice(&index.to_le_bytes());
    b.extend_from_slice(&u32::try_from(values.len()).unwrap_or(0).to_le_bytes());
    for x in values {
        b.extend_from_slice(&f16_bits(*x).to_le_bytes());
    }
    b
}

/// Decodes a views file's bytes; a record cut off at the end is passed over.
/// # Errors
/// The file does not start with the views magic.
pub fn parse_views(b: &[u8]) -> Result<Vec<View>, String> {
    if b.get(..8) != Some(VIEWS_MAGIC.as_slice()) {
        return Err("not a views file".into());
    }
    let mut out = Vec::new();
    let mut at = 8;
    let u16_at = |i: usize| b.get(i..i + 2).and_then(|x| <[u8; 2]>::try_from(x).ok()).map(u16::from_le_bytes);
    let u32_at = |i: usize| b.get(i..i + 4).and_then(|x| <[u8; 4]>::try_from(x).ok()).map(u32::from_le_bytes);
    while let Some(n) = u16_at(at) {
        let n = usize::from(n);
        let Some(id) = b.get(at + 2..at + 2 + n) else { break };
        let (Some(index), Some(width)) = (u32_at(at + 2 + n), u32_at(at + 6 + n)) else { break };
        let start = at + 10 + n;
        let Some(raw) = b.get(start..start + width as usize * 2) else { break };
        let values = raw.as_chunks::<2>().0.iter().map(|c| f16_value(u16::from_le_bytes(*c))).collect();
        out.push((String::from_utf8_lossy(id).into_owned(), index, values));
        at = start + width as usize * 2;
    }
    Ok(out)
}

/// The last-layer values kept for one engine identity, keyed by (pack, candidate index in its judged
/// set). Empty when none were kept (no serving path supplies them yet).
/// # Errors
/// The file exists and cannot be read, or is not a views file.
pub fn read_views(memory: &Path, engine: &str) -> Result<BTreeMap<(String, u32), Vec<f32>>, String> {
    let p = dir(memory).join(VIEWS).join(format!("{engine}.bin"));
    match std::fs::read(&p) {
        Ok(b) => Ok(parse_views(&b).map_err(|e| format!("{}: {e}", p.display()))?.into_iter().map(|(k, i, v)| ((k, i), v)).collect()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
        Err(e) => Err(format!("{}: {e}", p.display())),
    }
}

// ---------------------------------------------------------------------------------------------
// The distil family: curated teacher pairs for the Writer's distilled cartridges.

/// The distil store file (family `distil`). It lives in the Nano's own memory folder
/// (`memory/training/distil.jsonl`): a distilled cartridge serves the Writer in every project, so
/// its data is not one project's. Rows hold text (the prompt and the teacher's completion) because
/// the user imported them as curated material by explicit request; they are never a read of a
/// studied code base, and the file they came from is listed in `references.json` with its licence
/// and hash.
pub const DISTIL: &str = "distil.jsonl";
/// The family tag of a distil row.
pub const DISTIL_FAMILY: &str = "distil";
/// The longest prompt and completion a distil row may hold (characters).
pub const DISTIL_PROMPT_MAX: usize = 16_000;
pub const DISTIL_COMPLETION_MAX: usize = 4_000;
/// The most teacher log-probabilities kept per completion token.
pub const DISTIL_TOPK_MAX: usize = 20;

/// One curated teacher pair.
#[derive(Clone, Debug, PartialEq)]
pub struct DistilRow {
    /// `d-` and the first 16 hex digits of the SHA-256 of prompt, completion and teacher.
    pub id: String,
    /// The set it was imported in (the import's name; a distilled cartridge is fitted per set).
    pub set: String,
    pub prompt: String,
    pub completion: String,
    /// The teacher that wrote the completion (a model id, or a person or tool).
    pub teacher: String,
    pub licence: String,
    /// The reference the import listed (`references.json`), empty when none.
    pub reference: String,
    /// `chat` (the Writer's chat template around the prompt, the default) or `raw`.
    pub template: String,
    /// Per completion token, the teacher's top log-probabilities as (token id, natural log
    /// probability), most likely first. Empty when the teacher gave text only. Token ids are the
    /// Qwen3 tokenizer's (the Writer's); a fit checks their alignment before it reads them.
    pub topk: Vec<Vec<(u32, f32)>>,
    pub split: Split,
    /// One of the map outcomes (`used`, `superseded`, `failed`, `duplicate`, `control`, `unused`):
    /// only `used` and `superseded` rows are read by a fit.
    pub outcome: String,
    pub at: u64,
    /// A teacher's judgment of the completion, for distillation into a preference cartridge:
    /// `accept`, `reject`, or empty for a plain teacher pair.
    pub verdict: String,
    /// The circumstance of a judged completion: the paths it concerns (their file kind and top
    /// folder are the circumstance labels) and the tool, as a preference signal carries them.
    pub paths: Vec<String>,
    pub tool: String,
    /// A teacher's score of the completion for the prompt, from 0 to 1, for distillation into a
    /// MicroModel whose output is a score rather than next tokens: for the Reranker the teacher's
    /// probability that the completion (a document) answers the prompt (a query); for the Embedder
    /// the teacher's similarity of the two texts. None for a plain teacher pair.
    pub score: Option<f64>,
}

impl DistilRow {
    /// Whether a fit reads the row: healthy data only (`used`, `superseded`).
    #[must_use]
    pub fn healthy(&self) -> bool {
        matches!(self.outcome.as_str(), "used" | "superseded")
    }
    /// The typed form of one stored row; None when it lacks a prompt or a completion.
    #[must_use]
    pub fn from_value(v: &Value) -> Option<Self> {
        let prompt = text(v, "prompt").filter(|p| !p.is_empty())?;
        let completion = text(v, "completion").filter(|c| !c.is_empty())?;
        let teacher = text(v, "teacher").unwrap_or_default();
        let id = text(v, "id").unwrap_or_else(|| distil_id(&prompt, &completion, &teacher));
        let topk = parse_topk(v.get("topk").ok());
        Some(Self {
            split: text(v, "split").and_then(|s| Split::parse(&s)).unwrap_or_else(|| distil_split(&prompt)),
            set: text(v, "set").unwrap_or_default(),
            licence: text(v, "licence").unwrap_or_default(),
            reference: text(v, "reference").unwrap_or_default(),
            template: text(v, "template").filter(|t| t == "raw").unwrap_or_else(|| "chat".to_owned()),
            outcome: text(v, "outcome").unwrap_or_default(),
            at: v.get("at").and_then(Value::integer).unwrap_or(0),
            verdict: text(v, "verdict").filter(|x| x == "accept" || x == "reject").unwrap_or_default(),
            paths: v.get("paths").and_then(Value::array).unwrap_or_default().iter().filter_map(|p| p.string().ok().map(str::to_owned)).take(16).collect(),
            tool: text(v, "tool").unwrap_or_default(),
            score: score_of(v.get("score").ok()),
            id,
            prompt,
            completion,
            teacher,
            topk,
        })
    }
    /// The row as stored.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let s = |x: &str| Value::String(x.to_owned());
        let mut m: BTreeMap<String, Value> = [
            ("v", Value::Number(FORMAT.to_string())),
            ("family", s(DISTIL_FAMILY)),
            ("id", s(&self.id)),
            ("set", s(&self.set)),
            ("prompt", s(&self.prompt)),
            ("completion", s(&self.completion)),
            ("teacher", s(&self.teacher)),
            ("licence", s(&self.licence)),
            ("reference", s(&self.reference)),
            ("template", s(&self.template)),
            ("split", s(self.split.name())),
            ("outcome", s(&self.outcome)),
            ("at", Value::Number(self.at.to_string())),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect();
        if !self.verdict.is_empty() {
            m.insert("verdict".into(), s(&self.verdict));
            m.insert("paths".into(), Value::Array(self.paths.iter().map(|p| s(p)).collect()));
            m.insert("tool".into(), s(&self.tool));
        }
        if let Some(x) = self.score {
            m.insert("score".into(), Value::Number(format!("{x:.6}")));
        }
        if !self.topk.is_empty() {
            let rows = self
                .topk
                .iter()
                .map(|pos| Value::Array(pos.iter().map(|(t, lp)| Value::Array(vec![Value::Number(t.to_string()), Value::Number(format!("{lp:.5}"))])).collect()))
                .collect();
            m.insert("topk".into(), Value::Array(rows));
        }
        Value::Object(m)
    }
}

/// A teacher's score as stored: a finite number from 0 to 1 (a number or its text); anything else
/// reads as none.
fn score_of(v: Option<&Value>) -> Option<f64> {
    let x = match v? {
        Value::Number(n) | Value::String(n) => n.trim().parse::<f64>().ok()?,
        _ => return None,
    };
    (x.is_finite() && (0.0..=1.0).contains(&x)).then_some(x)
}

/// Per-token teacher log-probabilities: `[[[id, logprob], ...], ...]`; anything else reads empty.
fn parse_topk(v: Option<&Value>) -> Vec<Vec<(u32, f32)>> {
    let Some(Value::Array(rows)) = v else { return Vec::new() };
    rows.iter()
        .map(|pos| {
            pos.array()
                .unwrap_or_default()
                .iter()
                .take(DISTIL_TOPK_MAX)
                .filter_map(|pair| {
                    let p = pair.array().ok()?;
                    let id = u32::try_from(p.first()?.integer().ok()?).ok()?;
                    #[expect(clippy::cast_possible_truncation, reason = "a log-probability stored as text")]
                    let lp = match p.get(1)? {
                        Value::Number(n) => n.parse::<f64>().ok()? as f32,
                        _ => return None,
                    };
                    (lp.is_finite() && lp <= 0.0).then_some((id, lp))
                })
                .collect()
        })
        .collect()
}

/// A distil row's id from what it holds.
#[must_use]
pub fn distil_id(prompt: &str, completion: &str, teacher: &str) -> String {
    let h = nmhash::sha256_bytes(format!("{prompt}\u{1f}{completion}\u{1f}{teacher}").as_bytes());
    format!("d-{}", h.get(..16).unwrap_or(""))
}

/// The split of a distil row, fixed by its prompt alone: every completion of one prompt (two
/// teachers, two versions) lands in the same split, so a held-out prompt is never fitted.
#[must_use]
pub fn distil_split(prompt: &str) -> Split {
    split_of(&nmhash::sha256_bytes(prompt.as_bytes()), DISTIL_FAMILY)
}

/// Every distil row of a memory folder (the Nano's own: `memory/`), oldest first; rows that do not
/// parse are passed over.
/// # Errors
/// The file exists and cannot be read.
pub fn read_distil(memory: &Path) -> Result<Vec<DistilRow>, String> {
    let mut v: Vec<DistilRow> = read_rows(memory, DISTIL)?.iter().filter_map(DistilRow::from_value).collect();
    v.sort_by_key(|r| r.at);
    Ok(v)
}

/// Section check: the split is fixed and near its shares over many keys; binary16 round trips;
/// a views record reads back; the tap keeps nothing when off and the text only at full; a distil
/// row round trips with its teacher log-probabilities and only healthy rows are read.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let row = DistilRow {
        id: distil_id("p", "c", "t"),
        set: "s".into(),
        prompt: "p".into(),
        completion: "c".into(),
        teacher: "t".into(),
        licence: "Apache-2.0".into(),
        reference: String::new(),
        template: "chat".into(),
        topk: vec![vec![(12, -0.25), (7, -2.0)]],
        split: distil_split("p"),
        outcome: "used".into(),
        at: 5,
        verdict: String::new(),
        paths: Vec::new(),
        tool: String::new(),
        score: Some(0.75),
    };
    let back = DistilRow::from_value(&row.to_value()).ok_or("examples: a distil row did not read back")?;
    let out_of_range = DistilRow::from_value(&DistilRow { score: None, ..row.clone() }.to_value()).map(|mut r| {
        r.score = score_of(Some(&Value::Number("1.5".into())));
        r.score
    });
    if out_of_range != Some(None) {
        return Err("examples: a teacher score outside 0 to 1 was kept".into());
    }
    if back != row || !back.healthy() || (DistilRow { outcome: "failed".into(), ..row.clone() }).healthy() || distil_split("p") != distil_split("p") {
        return Err(format!("examples: distil row round trip {back:?}"));
    }
    if split_of("abc", "src/a.rs") != split_of("abc", "src/a.rs") {
        return Err("examples: split not fixed".into());
    }
    let mut n = [0_u32; 3];
    for i in 0..2000 {
        let k = match split_of(&format!("q{i}"), "p") {
            Split::Fit => 0,
            Split::Tune => 1,
            Split::Held => 2,
        };
        n[k] += 1;
    }
    if !(1500..1700).contains(&n[0]) || !(140..260).contains(&n[1]) || !(140..260).contains(&n[2]) {
        return Err(format!("examples: split shares {n:?}"));
    }
    for x in [0.0_f32, 1.0, -2.5, 0.333_333_34, 65504.0, 1e-5, -1e-7] {
        let back = f16_value(f16_bits(x));
        let tol = (x.abs() * 1e-3).max(6e-8);
        if (back - x).abs() > tol {
            return Err(format!("examples: binary16 {x} came back {back}"));
        }
    }
    let mut file = VIEWS_MAGIC.to_vec();
    file.extend(view_record("a1", 3, &[0.5, -1.0, 2.0]));
    file.extend(view_record("a2", 0, &[1.0]));
    file.extend_from_slice(&[1, 0]);
    let got = parse_views(&file)?;
    if got != vec![("a1".to_owned(), 3, vec![0.5, -1.0, 2.0]), ("a2".to_owned(), 0, vec![1.0])] {
        return Err(format!("examples: views read back {got:?}"));
    }
    let before = level();
    set_level(OFF);
    tap("q", [("a.rs", 1, 2, "t")], &[0.5], true);
    let off = take_tap();
    set_level(FEATURES);
    tap("q", [("a.rs", 1, 2, "t")], &[0.5], true);
    let feat = take_tap();
    set_level(FULL);
    tap("q", [("a.rs", 1, 2, "t")], &[0.5], true);
    let full = take_tap();
    set_level(before);
    if off.is_some() || feat.as_ref().and_then(|t| t.cands.first()).is_none_or(|c| c.text.is_some()) || full.as_ref().and_then(|t| t.cands.first()).and_then(|c| c.text.clone()).as_deref() != Some("t") {
        return Err("examples: tap levels".into());
    }
    Ok(())
}
