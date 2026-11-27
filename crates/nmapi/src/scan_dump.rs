//! Scan dump for processor-only replay (2026-11-10; architecture by Luis Jake Gabriel III, plan
//! `docs/2026-11-10-neumor-plan-whole-architecture.md` section 11, phase B: the decision rules of the
//! tentacle cores and the G1 loop are improved by replay instead of a graphics-card run per trial).
//!
//! One exact scan per question is saved with everything the tentacle cores and G1 need to decide:
//! per window its lens, chunks (substrate index; path and line span in the substrate file), token
//! counts, the drone heads read, the per-position attention mass of the posed and content-free
//! reads, the per-head per-part log-sum-exp of the raw attention scores of both reads, every
//! per-chunk score kind, the window record, and the gold flags by the run's gold rule; plus what
//! the control arm forwarded (and the reranker's P(yes) it judged by) and the live core arm's
//! results, and a tape of every device read the live core loop made (re-reads and answers), keyed
//! by a hash of what was read, so a replay with the same rules gives the same numbers bit for bit.
//! A replay with changed rules that asks for a read the tape lacks gets a named fault, which the
//! core already handles (a re-read keeps the scan's scores; an answer takes the held score); the
//! misses are counted and reported.
//!
//! Format (one folder per dump run, under `scratch/tc1-dump/`): `substrate.json` (files, chunks
//! with their text, symbols: the tree as the run saw it, so chunk indices and G1's whole-code-base
//! layer replay exactly), and per question `q<index>.json` plus `q<index>.bin` (per-position mass,
//! little-endian f32). Small float arrays are hexadecimal strings of the f32 bits (exact, compact,
//! non-finite values kept). No question text is written: questions are named by index and id, and
//! read texts appear only inside SHA-256 keys. Written as each question finishes.
use crate::tentacle_pass::{Scan, Score, WindowScan};
use nmbrain::gmap::WindowRecord;
use nmengn::kvstore::{ChunkKv, Window};
use nmjson::Value;
use nmsubs::Substrate;
use nmtype::{Fault, FaultKind};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmapi.scan_dump", FaultKind::Execution, m)
}

/// Format version written into every file.
pub const VERSION: u64 = 1;

/// The tape of device reads made by the tentacle cores of one question: recorded in a live run,
/// answered from in a replay.
#[derive(Debug, Default)]
pub struct Tape {
    replay: bool,
    entries: Mutex<BTreeMap<String, Vec<f32>>>,
    /// (reads answered from the tape, reads the tape lacked).
    counts: Mutex<(usize, usize)>,
}
impl Tape {
    /// A tape that records every read.
    #[must_use]
    pub fn recorder() -> Self {
        Self::default()
    }
    /// A tape that answers reads from `entries` (a replay: no model, no device).
    #[must_use]
    pub fn replayer(entries: BTreeMap<String, Vec<f32>>) -> Self {
        Self { replay: true, entries: Mutex::new(entries), counts: Mutex::new((0, 0)) }
    }
    /// True for a replay tape.
    #[must_use]
    pub fn replaying(&self) -> bool {
        self.replay
    }
    /// The recorded result of a read, counting hits and misses.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<Vec<f32>> {
        let v = self.entries.lock().ok().and_then(|e| e.get(key).cloned());
        if let Ok(mut c) = self.counts.lock() {
            if v.is_some() {
                c.0 += 1;
            } else {
                c.1 += 1;
            }
        }
        v
    }
    /// Record a read.
    pub fn put(&self, key: &str, v: Vec<f32>) {
        if let Ok(mut e) = self.entries.lock() {
            e.insert(key.to_owned(), v);
        }
    }
    /// Everything recorded.
    #[must_use]
    pub fn entries(&self) -> BTreeMap<String, Vec<f32>> {
        self.entries.lock().map(|e| e.clone()).unwrap_or_default()
    }
    /// (hits, misses) of a replay.
    #[must_use]
    pub fn counts(&self) -> (usize, usize) {
        self.counts.lock().map(|c| *c).unwrap_or((0, 0))
    }
}

/// The address of a kept encoding, used only as a name lookup within one process.
#[must_use]
pub fn addr(c: &Arc<ChunkKv>) -> usize {
    Arc::as_ptr(c).addr()
}

/// Content key of every kept encoding of a scan's windows, by address (`anchor` for each
/// window's first part): a read is named by what it reads, the same in a live run and a replay.
#[must_use]
pub fn names_of(sub: &Substrate, scan: &Scan) -> BTreeMap<usize, String> {
    let mut out = BTreeMap::new();
    for w in &scan.windows {
        let Some(win) = w.window.as_ref() else { continue };
        if let Some(a) = win.chunks.first() {
            out.insert(addr(a), "anchor".to_owned());
        }
        for (c, kv) in w.place.iter().zip(win.chunks.iter().skip(1)) {
            if let Some(ch) = sub.chunks.get(*c) {
                out.insert(addr(kv), ch.key.clone());
            }
        }
    }
    out
}

/// The tape key of one read: SHA-256 over the content keys of the chunks read (after the
/// anchor), the heads, and the posed and content-free texts (so no text is stored).
#[must_use]
pub fn read_key(names: &BTreeMap<usize, String>, chunks: &[Arc<ChunkKv>], heads: &[(usize, usize)], posed: &str, null: &str) -> String {
    let mut s = String::new();
    for c in chunks {
        s.push_str(names.get(&addr(c)).map_or("?", String::as_str));
        s.push('\n');
    }
    s.push_str(&format!("{heads:?}\n{posed}\n\u{1}\n{null}"));
    nmhash::sha256_bytes(s.as_bytes())
}

/// f32 values as one hexadecimal string of their bits (8 characters each).
#[must_use]
pub fn hex(v: &[f32]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(v.len() * 8);
    for x in v {
        let _ = write!(s, "{:08x}", x.to_bits());
    }
    s
}

/// The inverse of `hex`.
/// # Errors
/// A length that is not a multiple of 8 or a non-hexadecimal digit.
pub fn unhex(s: &str) -> Result<Vec<f32>, Fault> {
    if !s.len().is_multiple_of(8) {
        return Err(bad("float string length is not a multiple of 8"));
    }
    (0..s.len() / 8)
        .map(|i| {
            s.get(i * 8..i * 8 + 8)
                .and_then(|d| u32::from_str_radix(d, 16).ok())
                .map(f32::from_bits)
                .ok_or_else(|| bad("bad hexadecimal float"))
        })
        .collect()
}

fn n(x: usize) -> Value {
    Value::Number(x.to_string())
}
fn f(x: f64) -> Value {
    if x.is_finite() { Value::Number(format!("{x}")) } else { Value::String(format!("{x}")) }
}
fn s(x: &str) -> Value {
    Value::String(x.to_owned())
}
fn obj(items: Vec<(&str, Value)>) -> Value {
    Value::Object(items.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn arr<T>(v: &[T], g: impl Fn(&T) -> Value) -> Value {
    Value::Array(v.iter().map(g).collect())
}

fn get<'a>(v: &'a Value, k: &str) -> Result<&'a Value, Fault> {
    v.get(k).map_err(|e| bad(format!("{k}: {e}")))
}
fn gu(v: &Value, k: &str) -> Result<usize, Fault> {
    usize::try_from(get(v, k)?.integer().map_err(|e| bad(format!("{k}: {e}")))?).map_err(|e| bad(e.to_string()))
}
fn gs(v: &Value, k: &str) -> Result<String, Fault> {
    Ok(get(v, k)?.string().map_err(|e| bad(format!("{k}: {e}")))?.to_owned())
}
fn gf(v: &Value, k: &str) -> Result<f64, Fault> {
    match get(v, k)? {
        Value::Number(x) | Value::String(x) => x.parse::<f64>().map_err(|e| bad(format!("{k}: {e}"))),
        _ => Err(bad(format!("{k}: not a number"))),
    }
}
fn gb(v: &Value, k: &str) -> Result<bool, Fault> {
    match get(v, k)? {
        Value::Bool(b) => Ok(*b),
        _ => Err(bad(format!("{k}: not a boolean"))),
    }
}
fn ga<'a>(v: &'a Value, k: &str) -> Result<&'a [Value], Fault> {
    get(v, k)?.array().map_err(|e| bad(format!("{k}: {e}")))
}
fn us(v: &Value, k: &str) -> Result<Vec<usize>, Fault> {
    ga(v, k)?.iter().map(|x| usize::try_from(x.integer().map_err(|e| bad(e.to_string()))?).map_err(|e| bad(e.to_string()))).collect()
}
fn pairs(v: &Value, k: &str) -> Result<Vec<(usize, usize)>, Fault> {
    ga(v, k)?
        .iter()
        .map(|p| {
            let a = p.array().map_err(|e| bad(e.to_string()))?;
            let at = |i: usize| -> Result<usize, Fault> {
                usize::try_from(a.get(i).ok_or_else(|| bad("short pair"))?.integer().map_err(|e| bad(e.to_string()))?)
                    .map_err(|e| bad(e.to_string()))
            };
            Ok((at(0)?, at(1)?))
        })
        .collect()
}

/// Write the substrate the dump run read: files (path, hash), chunks (file, lines, content key,
/// text) and symbols, so a replay builds the same chunk indices and G1 layer.
/// # Errors
/// Write faults.
pub fn write_substrate(dir: &Path, sub: &Substrate) -> Result<PathBuf, Fault> {
    let v = obj(vec![
        ("format", n(usize::try_from(VERSION).unwrap_or(1))),
        ("version", Value::Number(sub.version.to_string())),
        ("files", arr(&sub.files, |x| obj(vec![("path", s(&x.path)), ("hash", s(&x.hash))]))),
        (
            "chunks",
            arr(&sub.chunks, |c| {
                obj(vec![("file", n(c.file)), ("start", n(c.start)), ("end", n(c.end)), ("key", s(&c.key)), ("text", s(&c.text))])
            }),
        ),
        (
            "symbols",
            arr(&sub.symbols, |x| {
                obj(vec![
                    ("name", s(&x.name)),
                    ("kind", s(&x.kind)),
                    ("file", s(&x.file)),
                    ("line", n(x.line)),
                    ("end", n(x.end)),
                    ("parent", s(&x.parent)),
                    ("text", s(&x.text)),
                ])
            }),
        ),
    ]);
    let p = dir.join("substrate.json");
    std::fs::write(&p, v.encode().map_err(|e| bad(e.to_string()))?).map_err(|e| bad(format!("{}: {e}", p.display())))?;
    Ok(p)
}

/// Read a dump's substrate back (processor only: no embedding, no dense lane, no model).
/// # Errors
/// Missing or malformed file.
pub fn load_substrate(dir: &Path) -> Result<Substrate, Fault> {
    let p = dir.join("substrate.json");
    let v = nmjson::parse(&std::fs::read(&p).map_err(|e| bad(format!("{}: {e}", p.display())))?).map_err(|e| bad(e.to_string()))?;
    let embed: nmsubs::Embed = Arc::new(|_: &str| Vec::new());
    let mut sub = nmsubs::from_files(&[], embed);
    sub.version = get(&v, "version")?.integer().map_err(|e| bad(e.to_string()))?;
    sub.files = ga(&v, "files")?
        .iter()
        .map(|x| {
            let mut f = nmsubs::repo::FileInfo::new(&gs(x, "path")?, "");
            f.hash = gs(x, "hash")?;
            Ok(f)
        })
        .collect::<Result<_, Fault>>()?;
    sub.chunks = ga(&v, "chunks")?
        .iter()
        .map(|x| {
            let text = gs(x, "text")?;
            Ok(nmsubs::Chunk {
                file: gu(x, "file")?,
                start: gu(x, "start")?,
                end: gu(x, "end")?,
                low: text.to_lowercase(),
                text,
                vec: Vec::new(),
                key: gs(x, "key")?,
                kind: "window",
                heading: String::new(),
                section: None,
                tokens: 0,
                piece: nmsubs::piece::Meta::default(),
                lead: String::new(),
            })
        })
        .collect::<Result<_, Fault>>()?;
    sub.symbols = ga(&v, "symbols")?
        .iter()
        .map(|x| {
            Ok(nmsubs::repo::Symbol {
                name: gs(x, "name")?,
                kind: gs(x, "kind")?,
                file: gs(x, "file")?,
                line: gu(x, "line")?,
                // Dumps written before symbols had spans: the first line only, no parent.
                end: gu(x, "end").or_else(|_| gu(x, "line"))?,
                parent: gs(x, "parent").unwrap_or_default(),
                text: gs(x, "text")?,
            })
        })
        .collect::<Result<_, Fault>>()?;
    // Term counts per file from the saved chunks (2026-11-13, found by SN5: the saved tree had none,
    // so every question word read as absent); the same word split the live index uses.
    let mut per_file: Vec<std::collections::BTreeSet<String>> = vec![std::collections::BTreeSet::new(); sub.files.len()];
    // A project's own hidden configuration is kept out of the counts (`Substrate::hidden_df`).
    let hidden: Vec<bool> = sub.files.iter().map(|f| nmtype::in_hidden_folder(&f.path)).collect();
    for c in &sub.chunks {
        if let Some(set) = per_file.get_mut(c.file) {
            let w = nmlex::words(&c.text);
            if !hidden.get(c.file).copied().unwrap_or(false) {
                sub.words += w.len();
            }
            set.extend(w);
        }
    }
    sub.hidden_files = hidden.iter().filter(|h| **h).count();
    for (set, h) in per_file.into_iter().zip(&hidden) {
        let counts = if *h { &mut sub.hidden_df } else { &mut sub.df };
        for w in set {
            *counts.entry(w).or_default() += 1;
        }
    }
    Ok(sub)
}

/// The control arm of one question: today's served pass on the same scan.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ControlArm {
    /// Forwarded chunk indices, best first.
    pub order: Vec<usize>,
    pub seconds: f64,
    pub note: String,
}

/// The live core arm of one question (what a replay must reproduce).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CoreArm {
    /// G1's hand-over as chunk indices, best first.
    pub order: Vec<usize>,
    /// Per core (g0 id), its first ranking: (chunk index, reported score), best first.
    pub ranked: Vec<(String, Vec<(usize, f32)>)>,
    pub rounds: u32,
    pub gate: usize,
    pub stop: String,
    pub no_info: usize,
    pub reads: usize,
    pub seconds: f64,
    pub note: String,
    /// SHA-256 of G1's whole hand-over as encoded (`Handover::to_value`), so a replay can be
    /// compared field for field without the question text being stored.
    pub handover: String,
}

/// One question of a dump.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct QuestionDump {
    /// Index in the development half (0-based) and the question id.
    pub index: usize,
    pub id: String,
    /// Substrate chunk indices that hold the answer by the run's gold rule.
    pub gold: Vec<usize>,
    pub scan_s: f64,
    pub control: ControlArm,
    pub core: CoreArm,
    /// The core loop's device reads: tape key to the read's scores.
    pub tape: BTreeMap<String, Vec<f32>>,
    /// Settings of both arms and the memory line at the question's end.
    pub settings: String,
    pub memory: String,
}

/// The file stem of question `index`.
#[must_use]
pub fn stem(index: usize) -> String {
    format!("q{index:02}")
}

fn record_value(r: &WindowRecord) -> Value {
    obj(vec![
        ("tentacle", n(usize::from(r.tentacle))),
        ("lens", s(&r.lens)),
        ("pose", s(&r.pose)),
        ("chunks", arr(&r.chunks, |c| n(*c))),
        ("tokens", n(r.tokens)),
        ("level", n(usize::from(r.level))),
        ("findings", n(r.findings)),
        ("best", s(&hex(&[r.best]))),
        ("baseline", s(&hex(&[r.baseline]))),
        ("no_information", Value::Bool(r.no_information)),
        ("files", arr(&r.files, |x| s(x))),
    ])
}
fn record_of(v: &Value) -> Result<WindowRecord, Fault> {
    let one = |k: &str| -> Result<f32, Fault> { unhex(&gs(v, k)?)?.first().copied().ok_or_else(|| bad(format!("{k}: empty"))) };
    Ok(WindowRecord {
        tentacle: u8::try_from(gu(v, "tentacle")?).map_err(|e| bad(e.to_string()))?,
        lens: gs(v, "lens")?,
        pose: gs(v, "pose")?,
        chunks: us(v, "chunks")?,
        tokens: gu(v, "tokens")?,
        level: u8::try_from(gu(v, "level")?).map_err(|e| bad(e.to_string()))?,
        findings: gu(v, "findings")?,
        best: one("best")?,
        baseline: one("baseline")?,
        no_information: gb(v, "no_information")?,
        files: ga(v, "files")?
            .iter()
            .map(|x| x.string().map(str::to_owned).map_err(|e| bad(e.to_string())))
            .collect::<Result<_, Fault>>()?,
    })
}

/// Write one question: `q<index>.json` and `q<index>.bin`. `heads` gives the drone heads a
/// window was read with (by tentacle). Returns the bytes written.
/// # Errors
/// Write faults.
pub fn write_question(dir: &Path, q: &QuestionDump, scan: &Scan, heads: &dyn Fn(usize) -> Vec<(usize, usize)>) -> Result<usize, Fault> {
    let mut bin: Vec<u8> = Vec::new();
    let mut windows = Vec::new();
    for w in &scan.windows {
        let at = bin.len() / 4;
        for x in w.posed.iter().chain(&w.null) {
            bin.extend_from_slice(&x.to_le_bytes());
        }
        let scores: Vec<(String, Value)> =
            Score::ALL.into_iter().chain(Score::EXTRA).map(|k| (k.name().to_owned(), Value::String(hex(&w.scores(k))))).collect();
        windows.push(obj(vec![
            ("tentacle", n(w.tentacle)),
            ("lens", s(nmsubs::lens::NAMES.get(w.tentacle).copied().unwrap_or("lens"))),
            ("heads", Value::Array(heads(w.tentacle).iter().map(|(l, h)| Value::Array(vec![n(*l), n(*h)])).collect())),
            ("place", arr(&w.place, |c| n(*c))),
            ("parts", Value::Array(w.parts.iter().map(|(a, b)| Value::Array(vec![n(*a), n(*b)])).collect())),
            ("len", n(w.len)),
            ("mass_at", n(at)),
            ("mass_values", n(w.posed.len())),
            ("null_values", n(w.null.len())),
            ("cal_posed", s(&hex(&w.cal_posed))),
            ("cal_null", s(&hex(&w.cal_null))),
            ("scores", Value::Object(scores.into_iter().collect())),
            ("gold", Value::Array(w.place.iter().map(|c| Value::Bool(q.gold.contains(c))).collect())),
            // Refinement step 1: lens ranks when the scan recorded them (absent in older dumps).
            ("lens_rank", arr(&w.lens_rank, |c| n(*c))),
        ]));
    }
    let tape: BTreeMap<String, Value> = q.tape.iter().map(|(k, v)| (k.clone(), Value::String(hex(v)))).collect();
    let rerank: BTreeMap<String, Value> =
        scan.rerank_scores().into_iter().map(|(c, p)| (c.to_string(), Value::String(hex(&[p])))).collect();
    let v = obj(vec![
        ("format", n(usize::try_from(VERSION).unwrap_or(1))),
        ("index", n(q.index)),
        ("id", s(&q.id)),
        ("gold", arr(&q.gold, |c| n(*c))),
        ("settings", s(&q.settings)),
        ("memory", s(&q.memory)),
        (
            "scan",
            obj(vec![
                ("seconds", f(q.scan_s)),
                ("focused", Value::Bool(scan.focused)),
                ("logits", Value::Bool(scan.logits)),
                ("version", Value::Number(scan.version.to_string())),
                ("chunks", n(scan.chunks)),
                ("device", s(&scan.device)),
                ("window_tokens", n(scan.window_tokens)),
                ("enc_tokens", n(scan.enc_tokens)),
                ("enc_ms", f(scan.enc_ms)),
                ("bytes_per_token", n(scan.bytes_per_token)),
                ("last_layer", n(scan.last_layer)),
                ("drones", s(&scan.drones)),
                ("held_bytes", n(scan.held_bytes)),
                ("held_chunks", n(scan.held_chunks)),
                ("enc_again_tokens", n(scan.enc_again_tokens)),
                ("read_ms", f(scan.read_ms)),
                ("null_kept", n(scan.null_kept)),
                ("null_read", n(scan.null_read)),
                ("null_dropped", n(scan.null_dropped)),
                ("budget", s(&scan.budget)),
                ("windows", Value::Array(windows)),
                ("records", arr(&scan.records, record_value)),
            ]),
        ),
        (
            "control",
            obj(vec![
                ("order", arr(&q.control.order, |c| n(*c))),
                ("seconds", f(q.control.seconds)),
                ("note", s(&q.control.note)),
                ("rerank", Value::Object(rerank)),
            ]),
        ),
        (
            "core",
            obj(vec![
                ("order", arr(&q.core.order, |c| n(*c))),
                (
                    "ranked",
                    Value::Array(
                        q.core
                            .ranked
                            .iter()
                            .map(|(id, l)| {
                                obj(vec![
                                    ("id", s(id)),
                                    ("chunks", arr(l, |x| n(x.0))),
                                    ("scores", s(&hex(&l.iter().map(|x| x.1).collect::<Vec<f32>>()))),
                                ])
                            })
                            .collect(),
                    ),
                ),
                ("rounds", n(usize::try_from(q.core.rounds).unwrap_or(0))),
                ("gate", n(q.core.gate)),
                ("stop", s(&q.core.stop)),
                ("no_info", n(q.core.no_info)),
                ("reads", n(q.core.reads)),
                ("seconds", f(q.core.seconds)),
                ("note", s(&q.core.note)),
                ("handover", s(&q.core.handover)),
            ]),
        ),
        ("tape", Value::Object(tape)),
    ]);
    let stem = stem(q.index);
    let jp = dir.join(format!("{stem}.json"));
    let bp = dir.join(format!("{stem}.bin"));
    let text = v.encode().map_err(|e| bad(e.to_string()))?;
    std::fs::write(&bp, &bin).map_err(|e| bad(format!("{}: {e}", bp.display())))?;
    std::fs::write(&jp, &text).map_err(|e| bad(format!("{}: {e}", jp.display())))?;
    Ok(bin.len() + text.len())
}

/// Question indices present in a dump folder, in order.
#[must_use]
pub fn indices(dir: &Path) -> Vec<usize> {
    let mut v: Vec<usize> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .filter_map(|e| {
                    e.file_name()
                        .to_str()
                        .and_then(|n| n.strip_prefix('q'))
                        .and_then(|n| n.strip_suffix(".json"))
                        .and_then(|n| n.parse().ok())
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort_unstable();
    v
}

/// Read one question back. `question` is its text (read by the caller from the question file;
/// the dump holds none): the replayed scan carries it so G1 and the tape keys see what the live
/// run saw. The windows are outlines (`Window::outline`): chunk references and positions, no
/// encodings, so the cores run on the processor and every device read comes from the tape.
/// Returns the question's record and the scan.
/// # Errors
/// Missing or malformed files.
pub fn load_question(dir: &Path, index: usize, question: &str) -> Result<(QuestionDump, Scan), Fault> {
    let stem = stem(index);
    let jp = dir.join(format!("{stem}.json"));
    let bp = dir.join(format!("{stem}.bin"));
    let v = nmjson::parse(&std::fs::read(&jp).map_err(|e| bad(format!("{}: {e}", jp.display())))?).map_err(|e| bad(e.to_string()))?;
    let bin = std::fs::read(&bp).map_err(|e| bad(format!("{}: {e}", bp.display())))?;
    let floats = |at: usize, k: usize| -> Result<Vec<f32>, Fault> {
        bin.get(at * 4..(at + k) * 4)
            .ok_or_else(|| bad(format!("{}: short", bp.display())))?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| Ok(f32::from_le_bytes(*b)))
            .collect()
    };
    let sv = get(&v, "scan")?;
    let mut scan = Scan::default();
    scan.question = question.to_owned();
    scan.focused = gb(sv, "focused")?;
    scan.logits = gb(sv, "logits")?;
    scan.version = get(sv, "version")?.integer().map_err(|e| bad(e.to_string()))?;
    scan.chunks = gu(sv, "chunks")?;
    scan.device = gs(sv, "device")?;
    scan.window_tokens = gu(sv, "window_tokens")?;
    scan.enc_tokens = gu(sv, "enc_tokens")?;
    scan.enc_ms = gf(sv, "enc_ms")?;
    scan.bytes_per_token = gu(sv, "bytes_per_token")?;
    scan.last_layer = gu(sv, "last_layer")?;
    scan.drones = gs(sv, "drones")?;
    scan.held_bytes = gu(sv, "held_bytes")?;
    scan.held_chunks = gu(sv, "held_chunks")?;
    scan.enc_again_tokens = gu(sv, "enc_again_tokens")?;
    scan.read_ms = gf(sv, "read_ms")?;
    scan.null_kept = gu(sv, "null_kept")?;
    scan.null_read = gu(sv, "null_read")?;
    scan.null_dropped = gu(sv, "null_dropped")?;
    scan.budget = gs(sv, "budget")?;
    scan.records = ga(sv, "records")?.iter().map(record_of).collect::<Result<_, Fault>>()?;
    for w in ga(sv, "windows")? {
        let parts = pairs(w, "parts")?;
        let at = gu(w, "mass_at")?;
        let (np, nn) = (gu(w, "mass_values")?, gu(w, "null_values")?);
        let anchor = parts.first().map_or(0, |p| p.1);
        let kv: Vec<Arc<ChunkKv>> = parts
            .iter()
            .enumerate()
            .map(|(k, p)| Arc::new(ChunkKv { tokens: p.1, anchor: if k == 0 { 0 } else { anchor }, ..ChunkKv::default() }))
            .collect();
        let window = Window::outline(&kv);
        if window.parts != parts {
            return Err(bad(format!("{}: window outline differs from the recorded parts", jp.display())));
        }
        scan.windows.push(WindowScan {
            tentacle: gu(w, "tentacle")?,
            place: us(w, "place")?,
            len: gu(w, "len")?,
            posed: floats(at, np)?,
            null: floats(at + np, nn)?,
            cal_posed: unhex(&gs(w, "cal_posed")?)?,
            cal_null: unhex(&gs(w, "cal_null")?)?,
            window: Some(Arc::new(window)),
            parts,
            lens_rank: if w.get("lens_rank").is_ok() { us(w, "lens_rank")? } else { Vec::new() },
        });
    }
    let cv = get(&v, "control")?;
    let kv = get(&v, "core")?;
    let tape = get(&v, "tape")?
        .object()
        .map_err(|e| bad(e.to_string()))?
        .iter()
        .map(|(k, x)| Ok((k.clone(), unhex(x.string().map_err(|e| bad(e.to_string()))?)?)))
        .collect::<Result<BTreeMap<_, _>, Fault>>()?;
    let ranked = ga(kv, "ranked")?
        .iter()
        .map(|r| {
            let cs = us(r, "chunks")?;
            let sc = unhex(&gs(r, "scores")?)?;
            Ok((gs(r, "id")?, cs.into_iter().zip(sc).collect()))
        })
        .collect::<Result<_, Fault>>()?;
    let q = QuestionDump {
        index: gu(&v, "index")?,
        id: gs(&v, "id")?,
        gold: us(&v, "gold")?,
        scan_s: gf(sv, "seconds")?,
        control: ControlArm { order: us(cv, "order")?, seconds: gf(cv, "seconds")?, note: gs(cv, "note")? },
        core: CoreArm {
            order: us(kv, "order")?,
            ranked,
            rounds: u32::try_from(gu(kv, "rounds")?).map_err(|e| bad(e.to_string()))?,
            gate: gu(kv, "gate")?,
            stop: gs(kv, "stop")?,
            no_info: gu(kv, "no_info")?,
            reads: gu(kv, "reads")?,
            seconds: gf(kv, "seconds")?,
            note: gs(kv, "note")?,
            handover: gs(kv, "handover")?,
        },
        tape,
        settings: gs(&v, "settings")?,
        memory: gs(&v, "memory")?,
    };
    Ok((q, scan))
}

/// The reranker's P(yes) per chunk the control arm judged by, from a dump question file.
/// # Errors
/// Missing or malformed file.
pub fn load_rerank(dir: &Path, index: usize) -> Result<BTreeMap<usize, f32>, Fault> {
    let jp = dir.join(format!("{}.json", stem(index)));
    let v = nmjson::parse(&std::fs::read(&jp).map_err(|e| bad(format!("{}: {e}", jp.display())))?).map_err(|e| bad(e.to_string()))?;
    get(get(&v, "control")?, "rerank")?
        .object()
        .map_err(|e| bad(e.to_string()))?
        .iter()
        .map(|(k, x)| {
            Ok((
                k.parse::<usize>().map_err(|e| bad(e.to_string()))?,
                unhex(x.string().map_err(|e| bad(e.to_string()))?)?.first().copied().unwrap_or(0.0),
            ))
        })
        .collect()
}

/// Section check of the dump format (processor only): float strings round-trip every bit pattern
/// class (finite, signed zero, infinities, NaN payloads); a synthetic question with two windows
/// and a tape round-trips through the files: every score, mass value, record field and tape entry
/// equal bit for bit, the outlined windows carry the recorded positions, and the scores the loaded
/// windows compute equal the written ones; read keys depend on content, heads and texts, not on
/// where the encodings live. Values: `tests/section-checks/nmapi-scan-dump/report.txt`.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    use std::fmt::Write as _;
    let mut rep = String::from("nmapi.scan_dump (scan dump for processor-only replay)\n");
    let r = dump_check(&mut rep);
    let _ = writeln!(rep, "result: {}", r.as_ref().map_or_else(|e| format!("FAIL {e}"), |()| "ok".to_owned()));
    let dir = nmtype::check_dir("nmapi-scan-dump").map_err(|e| e.to_string())?;
    std::fs::write(dir.join("report.txt"), rep).map_err(|e| e.to_string())?;
    r
}

fn dump_check(rep: &mut String) -> Result<(), String> {
    use std::fmt::Write as _;
    let vals = [
        0.0_f32,
        -0.0,
        1.5,
        -2.25e-30,
        f32::MAX,
        f32::MIN_POSITIVE,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_1234),
        3.4e38,
        1e-45,
    ];
    let back = unhex(&hex(&vals)).map_err(|e| e.to_string())?;
    let same = back.iter().zip(&vals).all(|(a, b)| a.to_bits() == b.to_bits()) && back.len() == vals.len();
    let _ = writeln!(rep, "(a) {} float bit patterns round-trip through the hexadecimal form: {same}", vals.len());
    if !same || unhex("abc").is_ok() {
        return Err("float strings do not round-trip, or a bad length was accepted".into());
    }
    // A synthetic tree and question.
    let embed: nmsubs::Embed = Arc::new(|_: &str| vec![0.0; 4]);
    let files = [("a/one.rs", "pub fn one() -> u32 { 1 }\n"), ("a/two.rs", "pub fn two() -> u32 { 2 }\npub fn three() -> u32 { 3 }\n")];
    let sub = nmsubs::from_files(&files, embed);
    let dir = nmtype::check_dir("nmapi-scan-dump").map_err(|e| e.to_string())?.join("dump");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    write_substrate(&dir, &sub).map_err(|e| e.to_string())?;
    let sub2 = load_substrate(&dir).map_err(|e| e.to_string())?;
    let same_sub = sub2.chunks.len() == sub.chunks.len()
        && sub2
            .chunks
            .iter()
            .zip(&sub.chunks)
            .all(|(a, b)| a.key == b.key && a.start == b.start && a.end == b.end && a.file == b.file && a.text == b.text)
        && sub2.files.len() == sub.files.len()
        && sub2.symbols == sub.symbols;
    let _ = writeln!(
        rep,
        "(b) substrate: {} files, {} chunks, {} symbols round-trip: {same_sub}",
        sub.files.len(),
        sub.chunks.len(),
        sub.symbols.len()
    );
    if !same_sub || sub.chunks.is_empty() {
        return Err("substrate does not round-trip".into());
    }
    let mut scan = Scan::default();
    scan.question = "synthetic".into();
    scan.logits = true;
    scan.version = sub.version;
    scan.chunks = sub.chunks.len();
    scan.device = "cpu".into();
    scan.enc_ms = 1.25;
    scan.read_ms = 0.5;
    let nc = sub.chunks.len();
    for t in 0..2_usize {
        let place: Vec<usize> = (0..nc).collect();
        let mut parts = vec![(0, 3)];
        let mut at = 3;
        for k in 0..nc {
            parts.push((at, 4 + k));
            at += 4 + k;
        }
        let len = at;
        let heads = 2;
        #[expect(clippy::cast_precision_loss, reason = "small synthetic values")]
        let posed: Vec<f32> = (0..heads * len).map(|i| ((i * 7 + t) % 11) as f32 / 37.0).collect();
        #[expect(clippy::cast_precision_loss, reason = "small synthetic values")]
        let null: Vec<f32> = (0..heads * len).map(|i| ((i * 5 + 3) % 13) as f32 / 41.0).collect();
        #[expect(clippy::cast_precision_loss, reason = "small synthetic values")]
        let cal_posed: Vec<f32> = (0..heads * parts.len()).map(|i| 1.0 + i as f32 * 0.3).collect();
        #[expect(clippy::cast_precision_loss, reason = "small synthetic values")]
        let cal_null: Vec<f32> = (0..heads * parts.len()).map(|i| 0.5 + i as f32 * 0.1).collect();
        scan.windows.push(WindowScan {
            tentacle: t,
            place: place.clone(),
            parts,
            len,
            posed,
            null,
            cal_posed,
            cal_null,
            window: None,
            lens_rank: (0..nc).rev().collect(),
        });
        #[expect(clippy::cast_possible_truncation, reason = "two tentacles")]
        scan.records.push(WindowRecord {
            tentacle: t as u8,
            lens: "l".into(),
            pose: "plain".into(),
            chunks: place,
            tokens: len,
            level: 2,
            findings: 1,
            best: 0.25,
            baseline: -0.0,
            no_information: false,
            files: vec!["a/one.rs".into()],
        });
    }
    let mut tape = BTreeMap::new();
    tape.insert("k1".to_owned(), vec![1.0_f32, f32::NEG_INFINITY]);
    let q = QuestionDump {
        index: 3,
        id: "x01".into(),
        gold: vec![1],
        scan_s: 2.5,
        control: ControlArm { order: vec![1, 0], seconds: 0.75, note: "control".into() },
        core: CoreArm {
            order: vec![0],
            ranked: vec![("g0_t1".into(), vec![(1, 0.5), (0, -0.25)])],
            rounds: 2,
            gate: 0,
            stop: "done".into(),
            no_info: 0,
            reads: 4,
            seconds: 0.125,
            note: "core".into(),
            handover: "h".into(),
        },
        tape,
        settings: "s".into(),
        memory: "m".into(),
    };
    let bytes = write_question(&dir, &q, &scan, &|_| vec![(13, 6), (14, 6)]).map_err(|e| e.to_string())?;
    let (q2, s2) = load_question(&dir, 3, "synthetic").map_err(|e| e.to_string())?;
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    let windows_same = s2.windows.len() == scan.windows.len()
        && s2.windows.iter().zip(&scan.windows).all(|(a, b)| {
            a.place == b.place
                && a.parts == b.parts
                && a.len == b.len
                && bits(&a.posed) == bits(&b.posed)
                && bits(&a.null) == bits(&b.null)
                && bits(&a.cal_posed) == bits(&b.cal_posed)
                && bits(&a.cal_null) == bits(&b.cal_null)
                && a.lens_rank == b.lens_rank
                && Score::ALL.into_iter().chain(Score::EXTRA).all(|k| bits(&a.scores(k)) == bits(&b.scores(k)))
                && a.window.as_ref().is_some_and(|w| w.parts == b.parts && w.len == b.len && w.chunks.len() == b.parts.len())
        });
    let q_same = q2 == q && s2.records == scan.records && s2.enc_ms.to_bits() == scan.enc_ms.to_bits() && s2.question == "synthetic";
    let _ = writeln!(
        rep,
        "(c) question {} ({bytes} bytes): windows, every score kind, mass values, records, tape and arms equal bit for bit: {}",
        q.id,
        windows_same && q_same
    );
    let listed = indices(&dir);
    if !windows_same || !q_same || listed != vec![3] {
        return Err(format!("question does not round-trip (windows {windows_same}, record {q_same}, listed {listed:?})"));
    }
    // Read keys name content, not addresses.
    let a = Arc::new(ChunkKv { tokens: 4, ..ChunkKv::default() });
    let b = Arc::new(ChunkKv { tokens: 4, ..ChunkKv::default() });
    let mut names = BTreeMap::new();
    names.insert(addr(&a), "key-one".to_owned());
    names.insert(addr(&b), "key-one".to_owned());
    let k1 = read_key(&names, std::slice::from_ref(&a), &[(13, 6)], "p", "n");
    let k2 = read_key(&names, std::slice::from_ref(&b), &[(13, 6)], "p", "n");
    let k3 = read_key(&names, std::slice::from_ref(&a), &[(13, 7)], "p", "n");
    let k4 = read_key(&names, std::slice::from_ref(&a), &[(13, 6)], "p2", "n");
    let _ = writeln!(
        rep,
        "(d) read keys: same content at two addresses equal {}; other heads differ {}; other text differs {}",
        k1 == k2,
        k1 != k3,
        k1 != k4
    );
    if k1 != k2 || k1 == k3 || k1 == k4 {
        return Err("read keys do not follow content, heads and texts".into());
    }
    let t = Tape::replayer(q.tape);
    let hit = t.get("k1").is_some();
    let miss = t.get("absent").is_none();
    let _ = writeln!(rep, "(e) replay tape: hit {hit}, miss counted {miss}, counts {:?}", t.counts());
    if !hit || !miss || t.counts() != (1, 1) || !t.replaying() {
        return Err("replay tape counts".into());
    }
    Ok(())
}
