//! The training store: what the Nano keeps, as it works, for fitting cartridges later. It belongs to
//! learning, not to memory; its contract (row shapes, split, reader) is `nmtype::examples`.
//!
//! Two moments write it:
//! - Capture, in the serving process, after a context pack's reply is sent: the judged set (every
//!   candidate the judge read, served or not, with its place, the content hash of its lines, its
//!   rank in the reply and the reranker's exact score) goes to `judged.jsonl`, tied to the audit id
//!   of the call and to the engine identity the scores came from. The serving path hands the scores
//!   over through `nmtype::examples::tap`, from values it already computed; no model runs again. The
//!   writing runs on its own thread so the reply waits for nothing but taking the tap.
//! - Settle, when the evidence ledger closes a pack's window (a hook at a stop, or the `examples`
//!   command): the judged set is joined to what followed and becomes one learning example with a
//!   label per candidate, the verdict, its strength, the provenance (settings, cartridges and stacks
//!   in use, path taken, hardware, time, project), domain tags and a split fixed by hash. The same
//!   pass writes preference signals (separate file, family `preference`) and the hardware
//!   measurements (family `fundamentals`, domain `hardware`).
//!
//! The setting `examples` decides what is kept: `off` nothing; `features` (the default) hashes,
//! places, labels, the models' numbers and provenance, no question or passage text; `full` also the
//! question text and each judged passage's text, secrets dropped first. Nothing is kept with memory
//! off. Caps on rows, age and bytes prune the weakest and oldest first.
use crate::ledger::PackState;
use crate::sessions::Rec;
use nmjson::Value;
use nmtype::examples as ex;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::OnceLock;

/// When the store was last cleared (in the project memory folder, outside the store).
const CLEARED: &str = "training.cleared";
/// Packs, preference signals and example contents already handled, one key per line.
const SEEN: &str = "seen.txt";
const LOCK: &str = "lock";
const STALE_LOCK_MS: u64 = 60_000;
/// A session with no record for this long is over, for the edit-kept rule.
const IDLE_MS: u64 = 60 * 60 * 1000;

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}
fn obj(f: Vec<(&str, Value)>) -> Value {
    Value::Object(f.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn opt(v: Option<&str>) -> Value {
    v.map_or(Value::Null, s)
}
fn text(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::string).ok().map(str::to_owned)
}
fn sha16(t: &str) -> String {
    nmhash::sha256_bytes(t.as_bytes()).chars().take(16).collect()
}
fn norm(t: &str) -> String {
    t.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

/// The capture level the `examples` setting names (`nmtype::examples::OFF`, `FEATURES`, `FULL`).
#[must_use]
pub fn level_of(home: &Path) -> u8 {
    match crate::controls::current(home).get("examples").map(String::as_str) {
        Some("off") => ex::OFF,
        Some("full") => ex::FULL,
        _ => ex::FEATURES,
    }
}
fn level_name(l: u8) -> &'static str {
    match l {
        ex::OFF => "off",
        ex::FULL => "full",
        _ => "features",
    }
}

// ---------------------------------------------------------------------------------------------
// The store's lock and files.

struct Lock(PathBuf);
impl Drop for Lock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn lock(tdir: &Path, wait_ms: u64) -> Result<Lock, String> {
    std::fs::create_dir_all(tdir).map_err(|e| format!("{}: {e}", tdir.display()))?;
    let p = tdir.join(LOCK);
    let start = std::time::Instant::now();
    loop {
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&p) {
            Ok(mut f) => {
                let _ = writeln!(f, "{}", std::process::id());
                return Ok(Lock(p));
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let age = std::fs::metadata(&p).and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).map_or(0, |d| d.as_millis());
                if age > u128::from(STALE_LOCK_MS) {
                    let _ = std::fs::remove_file(&p);
                    continue;
                }
                if start.elapsed().as_millis() >= u128::from(wait_ms) {
                    return Err("the training store is being written; try again".into());
                }
                std::thread::park_timeout(std::time::Duration::from_millis(10));
            }
            Err(e) => return Err(format!("{}: {e}", p.display())),
        }
    }
}

fn append_lines(file: &Path, rows: &[Value]) -> Result<(), String> {
    if rows.is_empty() {
        return Ok(());
    }
    let mut out = String::new();
    for r in rows {
        out.push_str(&r.encode().map_err(|e| e.to_string())?);
        out.push('\n');
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(file).map_err(|e| format!("{}: {e}", file.display()))?;
    f.write_all(out.as_bytes()).map_err(|e| format!("{}: {e}", file.display()))
}

fn rewrite_lines(file: &Path, rows: &[Value]) -> Result<(), String> {
    let mut out = String::new();
    for r in rows {
        out.push_str(&r.encode().map_err(|e| e.to_string())?);
        out.push('\n');
    }
    let tmp = file.with_extension("tmp");
    std::fs::write(&tmp, out).and_then(|()| std::fs::rename(&tmp, file)).map_err(|e| format!("{}: {e}", file.display()))
}

fn seen(tdir: &Path) -> BTreeSet<String> {
    std::fs::read_to_string(tdir.join(SEEN)).unwrap_or_default().lines().map(str::to_owned).collect()
}

fn cleared_at(memory: &Path) -> u64 {
    std::fs::read_to_string(memory.join(CLEARED)).ok().and_then(|t| t.lines().next().and_then(|l| l.trim().parse().ok())).unwrap_or(0)
}

/// Passage hashes already in `passages.jsonl` (each line starts with its hash, keys being sorted).
fn passage_hashes(tdir: &Path) -> BTreeSet<String> {
    std::fs::read_to_string(tdir.join(ex::PASSAGES))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| l.strip_prefix("{\"hash\":\"").and_then(|r| r.get(..16)).map(str::to_owned))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Capture in the serving process.

struct Ctx {
    home: PathBuf,
    root: PathBuf,
    memory: PathBuf,
    label: String,
    memory_on: bool,
}
static CTX: OnceLock<Ctx> = OnceLock::new();

/// Called when serving starts: where captures go, and the tap's level from the setting.
pub fn start(home: &Path, root: &Path, memory: &Path, memory_on: bool) {
    let label = nmtype::project::label(home, root);
    let _ = CTX.set(Ctx { home: home.to_path_buf(), root: root.to_path_buf(), memory: memory.to_path_buf(), label, memory_on });
    ex::set_level(if memory_on { level_of(home) } else { ex::OFF });
}

/// The observer installed on the served contract: the audit row (`project::observe`), then the
/// training store's capture of a context pack, handed to a thread of its own.
pub fn observe(tool: &str, input: &Value, out: &Result<Value, nmapi::ApiError>, ms: u128) {
    crate::project::observe(tool, input, out, ms);
    if tool != "nm_context_pack" {
        return;
    }
    // Taken on every call, so one call's judged set never reaches another call's row.
    let tap = ex::take_tap();
    let Some(c) = CTX.get() else { return };
    if !c.memory_on || out.is_err() {
        return;
    }
    let question = input.get("question").and_then(Value::string).unwrap_or_default().to_owned();
    let pid = std::process::id();
    let _ = std::thread::Builder::new().name("training-capture".into()).spawn(move || {
        let lv = level_of(&c.home);
        ex::set_level(lv);
        if lv == ex::OFF {
            return;
        }
        let Some(row) = newest_pack_row(&c.home, &c.label, pid) else { return };
        let engine = if tap.is_some() { engine_identity() } else { None };
        let _ = capture(&c.memory, &c.root, &row, tap.as_ref(), &question, engine, &[], lv);
    });
}

/// The newest context pack row this process wrote to the audit log for `label` (read from the
/// log's tail; the observer writes it before the capture runs).
fn newest_pack_row(home: &Path, label: &str, pid: u32) -> Option<Value> {
    use std::io::{Read, Seek, SeekFrom};
    let p = home.join("memory").join("audit.jsonl");
    let mut f = std::fs::File::open(&p).ok()?;
    let len = f.metadata().map_or(0, |m| m.len());
    let mut b = Vec::new();
    f.seek(SeekFrom::Start(len.saturating_sub(512 * 1024))).ok()?;
    f.read_to_end(&mut b).ok()?;
    let t = String::from_utf8_lossy(&b);
    let me = format!("\"session\":\"{pid}-");
    let project = format!("\"project\":\"{label}\"");
    t.lines()
        .rev()
        .filter(|l| l.contains("\"tool\":\"nm_context_pack\"") && l.contains(&me) && l.contains(&project) && l.contains("\"state\":\"done\""))
        .find_map(|l| nmjson::parse(l.as_bytes()).ok())
}

/// The reranker's engine identity when it is bound in this process: an id and what makes its
/// numbers what they are (model, weight hash, assembly, precision of its linear layers, the
/// arithmetic switches, batching and early drop). Numbers under two identities are never mixed.
fn engine_identity() -> Option<(String, Value)> {
    use std::sync::atomic::Ordering::Relaxed;
    let m = nmapi::neural::bound_models().into_iter().find(|(r, _)| *r == "reranker")?.1;
    let mut widths: Vec<&str> = m.asm.nodes.iter().filter(|x| x.op == "linear").map(|x| x.precision.as_str()).collect();
    widths.sort_unstable();
    widths.dedup();
    let keeps_f32 = m.no_half.load(Relaxed);
    let drop = nmengn::roles::RERANK_DROP.lock().ok().and_then(|d| *d).map_or_else(|| "off".to_owned(), |d| format!("{}:{}:{}", d.layer, d.margin, d.keep));
    let fields: Vec<(&str, String)> = vec![
        ("model", m.pop.card.id.clone()),
        ("weights", m.pop.card.sha_main.clone()),
        ("assembly", m.asm.name.clone()),
        ("precision", widths.join(",")),
        ("binary32_math", keeps_f32.to_string()),
        ("half_math", (nmengn::gpu::HALF_MATH.load(Relaxed) && !keeps_f32).to_string()),
        ("segment_attention", nmengn::gpu::SEGMENT_ATTENTION.load(Relaxed).to_string()),
        ("batched", nmengn::roles::RERANK_BATCH.load(Relaxed).to_string()),
        ("early_drop", drop),
        ("device", if m.pop.gpu.is_some() { "gpu" } else { "cpu" }.to_owned()),
    ];
    engine_of(&fields)
}

/// An engine identity from its fields: `e` and 15 hex digits of the hash of the fields in order.
fn engine_of(fields: &[(&str, String)]) -> Option<(String, Value)> {
    let canon: String = fields.iter().map(|(k, v)| format!("{k}={v}\n")).collect();
    let v = Value::Object(fields.iter().map(|(k, v)| ((*k).to_owned(), s(v))).collect());
    Some((format!("e{}", sha16(&canon).get(..15).unwrap_or("")), v))
}

/// Reads the lines `start..=end` of `path` under `root`, caching whole files in `cache`.
fn lines_of(root: &Path, path: &str, start: u64, end: u64, cache: &mut BTreeMap<String, Option<String>>) -> Option<Vec<String>> {
    if path.contains("..") || path == "<outside>" {
        return None;
    }
    let body = cache.entry(path.to_owned()).or_insert_with(|| std::fs::read_to_string(path.split('/').fold(root.to_path_buf(), |a, c| a.join(c))).ok());
    let body = body.as_deref()?;
    let (a, b) = if (start, end) == (0, 0) { (1, u64::try_from(body.lines().count()).unwrap_or(0)) } else { (start, end) };
    if a == 0 || b < a {
        return None;
    }
    let lines: Vec<String> = body.lines().skip(usize::try_from(a - 1).ok()?).take(usize::try_from(b - a + 1).ok()?).map(str::to_owned).collect();
    (lines.len() as u64 == b - a + 1).then_some(lines)
}

fn place_text(path: &str, a: u64, b: u64) -> String {
    match (a, b) {
        (0, 0) => path.to_owned(),
        (a, b) if a == b => format!("{path}:{a}"),
        (a, b) => format!("{path}:{a}-{b}"),
    }
}

/// Writes the judged set of one served pack: from the tap when the serving path handed one over
/// (exact scores, the judge's whole set), else from the audit row's recorded scores (six decimals)
/// or, on the plain path, its served places alone. `views` are last-layer values per candidate in
/// the tap's order, when a serving path supplies them. Returns the candidates written.
/// # Errors
/// The store cannot be written.
#[allow(clippy::too_many_arguments)]
pub fn capture(memory: &Path, root: &Path, audit: &Value, tap: Option<&ex::Tapped>, question: &str, engine: Option<(String, Value)>, views: &[Option<Vec<f32>>], lv: u8) -> Result<usize, String> {
    if lv == ex::OFF {
        return Ok(0);
    }
    let pack = text(audit, "id").ok_or("the audit row has no id")?;
    let sources: Vec<String> = audit.get("sources").and_then(Value::array).unwrap_or_default().iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect();
    let cmap = audit.get("context_map").ok();
    let path_taken = cmap.and_then(|m| text(m, "mode")).unwrap_or_else(|| text(audit, "mode").unwrap_or_default());
    // (path, first line, last line, score, read by the judge, text from the judge)
    type Cand = (String, u64, u64, Option<f32>, bool, Option<String>);
    let mut cands: Vec<Cand> = Vec::new();
    let source = if let Some(t) = tap {
        for c in &t.cands {
            cands.push((c.path.clone(), c.start as u64, c.end as u64, t.scored.then_some(c.score), true, c.text.clone()));
        }
        "tap"
    } else if let Some(sc) = cmap.and_then(|m| m.get("scores").ok()).and_then(|x| x.array().ok()).filter(|a| !a.is_empty()) {
        for line in sc.iter().filter_map(|x| x.string().ok()) {
            let mut it = line.split(' ');
            if it.next() != Some("rr") {
                continue;
            }
            let (Some(at), Some(p)) = (it.next(), it.next()) else { continue };
            let (path, (a, b)) = ex::place(at);
            cands.push((path, a, b, p.parse().ok(), true, None));
        }
        "audit"
    } else {
        for src in &sources {
            let (path, (a, b)) = ex::place(src);
            cands.push((path, a, b, None, false, None));
        }
        "served"
    };
    let mut cache = BTreeMap::new();
    let mut passages: Vec<Value> = Vec::new();
    let mut rows = Vec::new();
    for (i, (path, a, b, score, judged, jtext)) in cands.iter().enumerate() {
        let at = place_text(path, *a, *b);
        let lines = lines_of(root, path, *a, *b, &mut cache);
        let hash = match (&lines, jtext) {
            (Some(l), _) => Some(ex::content_hash(&l.iter().map(String::as_str).collect::<Vec<_>>())),
            (None, Some(t)) => Some(ex::content_hash(&t.lines().collect::<Vec<_>>())),
            _ => None,
        };
        if lv == ex::FULL
            && let Some(h) = &hash
        {
            let raw = jtext.clone().or_else(|| lines.as_ref().map(|l| l.join("\n"))).unwrap_or_default();
            passages.push(obj(vec![("hash", s(h)), ("text", s(&crate::secrets::drop_secrets(&raw))), ("v", n(ex::FORMAT))]));
        }
        let mut f = vec![("place", s(&at)), ("hash", opt(hash.as_deref())), ("judged", Value::Bool(*judged)), ("index", n(i as u64))];
        let rank = sources.iter().position(|x| x == &at);
        f.push(("served", Value::Bool(rank.is_some())));
        f.push(("rank", rank.map_or(Value::Null, |r| n(r as u64))));
        if let Some(p) = score {
            f.push(("score", Value::Number(format!("{p:.6}"))));
            if source == "tap" {
                f.push(("bits", s(&format!("{:08x}", p.to_bits()))));
            }
        }
        rows.push(obj(f));
    }
    // Served places the judged set does not hold (definitions, crate folders) join it as served.
    for (k, src) in sources.iter().enumerate() {
        if !rows.iter().any(|r| text(r, "place").as_deref() == Some(src.as_str())) {
            let (path, (a, b)) = ex::place(src);
            let hash = lines_of(root, &path, a, b, &mut cache).map(|l| ex::content_hash(&l.iter().map(String::as_str).collect::<Vec<_>>()));
            rows.push(obj(vec![("place", s(src)), ("hash", opt(hash.as_deref())), ("judged", Value::Bool(false)), ("served", Value::Bool(true)), ("rank", n(k as u64))]));
        }
    }
    let q = crate::secrets::drop_secrets(question);
    let count = rows.len();
    let mut row = vec![
        ("v", n(ex::FORMAT)),
        ("row", s("judged")),
        ("pack", s(&pack)),
        ("server", opt(text(audit, "session").as_deref())),
        ("at", n(audit.get("time_ms").and_then(Value::integer).unwrap_or(0).saturating_add(audit.get("ms").and_then(Value::integer).unwrap_or(0)))),
        ("ask_sha", s(&sha16(&norm(&q)))),
        ("path", s(&path_taken)),
        ("source", s(source)),
        ("scored", Value::Bool(tap.is_none_or(|t| t.scored) && source != "served")),
        ("engine", opt(engine.as_ref().map(|e| e.0.as_str()))),
        ("level", s(level_name(lv))),
        ("candidates", Value::Array(rows)),
    ];
    if lv == ex::FULL {
        row.push(("question", s(&q)));
    }
    let tdir = ex::dir(memory);
    let _g = lock(&tdir, 2000)?;
    append_lines(&tdir.join(ex::JUDGED), &[obj(row)])?;
    if !passages.is_empty() {
        let have = passage_hashes(&tdir);
        let mut fresh: BTreeSet<String> = BTreeSet::new();
        passages.retain(|p| text(p, "hash").is_some_and(|h| !have.contains(&h) && fresh.insert(h)));
        append_lines(&tdir.join(ex::PASSAGES), &passages)?;
    }
    if let Some((id, header)) = &engine {
        let mut all = ex::read_engines(memory).unwrap_or_default();
        if !all.contains_key(id) {
            all.insert(id.clone(), header.clone());
            std::fs::write(tdir.join(ex::ENGINES), nmjson::pretty::pretty(&Value::Object(all))).map_err(|e| e.to_string())?;
        }
        if views.iter().any(Option::is_some) {
            let vdir = tdir.join(ex::VIEWS);
            std::fs::create_dir_all(&vdir).map_err(|e| e.to_string())?;
            let file = vdir.join(format!("{id}.bin"));
            let mut bytes = if file.exists() { Vec::new() } else { ex::VIEWS_MAGIC.to_vec() };
            for (i, v) in views.iter().enumerate() {
                if let Some(v) = v {
                    bytes.extend(ex::view_record(&pack, u32::try_from(i).unwrap_or(u32::MAX), v));
                }
            }
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&file).map_err(|e| e.to_string())?;
            f.write_all(&bytes).map_err(|e| e.to_string())?;
        }
    }
    Ok(count)
}

// ---------------------------------------------------------------------------------------------
// Settle: examples from closed packs, preference signals, hardware measurements.

/// What one settle wrote.
#[derive(Default, Debug)]
pub struct Report {
    pub examples: usize,
    pub preferences: usize,
    pub duplicates: usize,
    pub no_verdict: usize,
    pub pruned: usize,
}

/// One touch of a place in a window: when, read or edit, its lines.
struct Touch {
    at: (u64, usize),
    edit: bool,
    spans: Vec<(u64, u64)>,
}

fn overlaps(c: (u64, u64), sp: (u64, u64)) -> bool {
    c == (0, 0) || (sp.0 <= c.1 && c.0 <= sp.1)
}

/// Domain tags of an example from its places: language or kind by extension, `ml-architecture` by
/// path words, and the project.
fn domains(paths: &[&str], label: &str) -> Vec<String> {
    let mut d: BTreeSet<String> = BTreeSet::new();
    for p in paths {
        let lower = p.to_lowercase();
        let ext = lower.rsplit_once('.').map_or("", |x| x.1);
        let tag = match ext {
            "rs" => "code:rust",
            "py" | "pyi" => "code:python",
            "ts" | "tsx" | "js" | "jsx" | "mjs" => "code:javascript",
            "go" => "code:go",
            "java" | "kt" => "code:jvm",
            "c" | "h" | "cc" | "cpp" | "hpp" => "code:c-cpp",
            "cu" | "cuh" => "code:cuda",
            "md" | "txt" | "rst" | "pdf" => "documents",
            "toml" | "json" | "yaml" | "yml" => "config",
            _ => "other",
        };
        d.insert(tag.to_owned());
        if ["model", "attention", "kernel", "tensor", "layer", "train", "infer", "embed", "rerank", "cuda", "quant"].iter().any(|w| lower.contains(w)) {
            d.insert("ml-architecture".to_owned());
        }
    }
    d.insert(format!("project:{label}"));
    d.into_iter().collect()
}

/// The library stacks and cartridges a serving process put in use at start, from its
/// `library-stacks` audit row (the latest of that process at or before `at`).
fn stacks_of(starts: &[(String, u64, Value)], server: &str, at: u64) -> (Vec<String>, Vec<String>) {
    let pid = server.split('-').next().unwrap_or("");
    let Some((_, _, v)) = starts.iter().filter(|(p, t, _)| p == pid && *t <= at.saturating_add(1000)).max_by_key(|x| x.1) else {
        return (Vec::new(), Vec::new());
    };
    let uses = v
        .get("uses")
        .and_then(Value::array)
        .unwrap_or_default()
        .iter()
        .filter_map(|u| {
            let st = text(u, "stack")?;
            Some(format!("{}={st}{}", text(u, "use").unwrap_or_default(), if matches!(u.get("trial"), Ok(Value::Bool(true))) { " (trial)" } else { "" }))
        })
        .collect();
    let loaded = v.get("loaded").and_then(Value::array).unwrap_or_default().iter().filter_map(|x| text(x, "key")).collect();
    (uses, loaded)
}

/// Reads the audit log once for what settle needs: `library-stacks` rows (pid, time, row) and this
/// project's done context pack rows.
fn audit_rows(home: &Path, label: &str) -> (Vec<(String, u64, Value)>, Vec<Value>) {
    let all = std::fs::read_to_string(home.join("memory").join("audit.jsonl")).unwrap_or_default();
    let project = format!("\"project\":\"{label}\"");
    let (mut starts, mut packs) = (Vec::new(), Vec::new());
    for l in all.lines() {
        if l.contains("\"kind\":\"library-stacks\"") {
            if let Ok(v) = nmjson::parse(l.as_bytes()) {
                starts.push((text(&v, "pid").unwrap_or_default(), v.get("time_ms").and_then(Value::integer).unwrap_or(0), v));
            }
        } else if l.contains("\"tool\":\"nm_context_pack\"") && l.contains(&project) && l.contains("\"state\":\"done\"")
            && let Ok(v) = nmjson::parse(l.as_bytes())
        {
            packs.push(v);
        }
    }
    (starts, packs)
}

/// The example of one closed pack, or `None` when nothing verified or refuted it (and Claude Code
/// did not go elsewhere).
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn example_of(
    st: &PackState,
    recs: &[Rec],
    judged: Option<&Value>,
    idx: &crate::ledger::AuditIndex,
    starts: &[(String, u64, Value)],
    hw: &Value,
    root: Option<&Path>,
    label: &str,
    lv: u8,
    now: u64,
) -> Option<Value> {
    let p = &st.pack;
    let sv = p.served.as_ref()?;
    let w = crate::ledger::window(recs, p, now);
    let pre: BTreeMap<&str, Vec<(String, u64, u64)>> =
        recs.iter().filter(|r| r.event == "PreToolUse" && r.session == p.rec.session).filter_map(|r| Some((r.use_id.as_deref()?, r.lines.clone()))).collect();
    let turn: Vec<&Rec> = w.turn.iter().copied().filter(|r| r.done() && r.ok == Some(true)).collect();
    let touches = |path: &str| -> Vec<Touch> {
        turn.iter()
            .filter(|r| matches!(r.kind.as_str(), "read" | "edit" | "write") && r.paths.iter().any(|x| x == path))
            .map(|r| {
                let mut spans: Vec<(u64, u64)> = r.span_of(path).into_iter().collect();
                if let Some(extra) = r.use_id.as_deref().and_then(|u| pre.get(u)) {
                    spans.extend(extra.iter().filter(|e| e.0 == path).map(|e| (e.1, e.2)));
                }
                Touch { at: (r.t, r.i), edit: r.kind != "read", spans }
            })
            .collect()
    };
    let checks: Vec<&Rec> = w.turn.iter().copied().filter(|r| r.done() && matches!(r.kind.as_str(), "build" | "test")).collect();
    let last_check = checks.last().copied();
    let undos: Vec<&Rec> = w
        .turn
        .iter()
        .copied()
        .filter(|r| r.done() && matches!(r.cmd_class.as_deref(), Some("git-revert" | "git-restore" | "git-checkout" | "git-reset" | "git-stash")))
        .collect();
    let ev = |rule: &str| st.evs.iter().find(|e| e.rule == rule);
    let elsewhere = ev("elsewhere").is_some_and(|e| e.state == "passed-over");
    let pack_refuted = ev("asked-again").filter(|e| e.state == "refuted").or_else(|| ev("next-prompt").filter(|e| e.state == "refuted"));
    let pack_accepted = ev("next-prompt").filter(|e| e.state == "verified");
    // The edit-then-check verdict of a place edited at `first_edit`.
    let checked = |first_edit: (u64, usize)| -> Option<(&'static str, &'static str)> {
        let c = last_check.filter(|c| (c.t, c.i) > first_edit)?;
        match c.ok {
            Some(true) => Some(("verified", "indirect")),
            Some(false) => Some(("refuted", "indirect")),
            None => None,
        }
    };
    let reverted = |path: &str, first_edit: (u64, usize)| -> Option<&'static str> {
        let after: Vec<&&Rec> = undos.iter().filter(|r| (r.t, r.i) > first_edit).collect();
        if after.iter().any(|r| r.paths.iter().any(|x| x == path)) {
            Some("direct")
        } else if after.iter().any(|r| r.paths.is_empty() || r.paths.iter().any(|x| x == ".")) {
            Some("weak")
        } else {
            None
        }
    };
    // The judged set, else the served places.
    let mut cands: Vec<BTreeMap<String, Value>> = match judged.and_then(|j| j.get("candidates").and_then(Value::array).ok()) {
        Some(list) => list.iter().filter_map(|c| c.object().ok().cloned()).collect(),
        None => sv
            .sources
            .iter()
            .enumerate()
            .map(|(k, src)| {
                [("place", s(src)), ("hash", Value::Null), ("judged", Value::Bool(false)), ("served", Value::Bool(true)), ("rank", n(k as u64))]
                    .into_iter()
                    .map(|(a, b)| (a.to_owned(), b))
                    .collect()
            })
            .collect(),
    };
    let served_paths: BTreeSet<String> = cands.iter().filter(|c| matches!(c.get("served"), Some(Value::Bool(true)))).filter_map(|c| c.get("place").and_then(|x| x.string().ok()).map(|x| ex::place(x).0)).collect();
    let mut any_used = false;
    for c in &mut cands {
        let at = c.get("place").and_then(|x| x.string().ok()).unwrap_or_default().to_owned();
        let (path, span) = ex::place(&at);
        let served = matches!(c.get("served"), Some(Value::Bool(true)));
        let t = if served { touches(&path) } else { Vec::new() };
        let direct = t.iter().any(|x| x.spans.iter().any(|sp| overlaps(span, *sp)));
        let first_edit = t.iter().filter(|x| x.edit).map(|x| x.at).min();
        let used_strength = if direct { "direct" } else { "indirect" };
        let (label, strength): (&str, Option<&str>) = if let Some(fe) = first_edit {
            if let Some(r) = reverted(&path, fe) {
                ("refuted", Some(r))
            } else if let Some((l, sg)) = checked(fe) {
                (l, Some(sg))
            } else if let Some(e) = pack_refuted {
                ("refuted", e.strength)
            } else if pack_accepted.is_some() {
                ("verified", Some("weak"))
            } else {
                ("used", Some(used_strength))
            }
        } else if !t.is_empty() {
            if let Some(e) = pack_refuted {
                ("refuted", e.strength)
            } else if pack_accepted.is_some() {
                ("verified", Some("weak"))
            } else {
                ("used", Some(used_strength))
            }
        } else if served && elsewhere {
            ("passed-over", Some("indirect"))
        } else if served {
            ("unused", None)
        } else {
            ("judged", None)
        };
        any_used |= !t.is_empty();
        c.insert("label".into(), s(label));
        c.insert("strength".into(), opt(strength));
    }
    // Where Claude Code went instead: the true place when it was then verified.
    if elsewhere && !any_used {
        let mut went: BTreeMap<String, Vec<Touch>> = BTreeMap::new();
        for r in &turn {
            if !matches!(r.kind.as_str(), "read" | "edit" | "write") {
                continue;
            }
            for x in r.paths.iter().filter(|x| !served_paths.contains(*x) && !matches!(x.as_str(), "." | "<outside>")) {
                went.entry(x.clone()).or_default().extend(touches(x));
            }
        }
        let mut cache = BTreeMap::new();
        for (path, t) in went.iter().take(8) {
            let first_edit = t.iter().filter(|x| x.edit).map(|x| x.at).min();
            let strength = match first_edit.and_then(checked) {
                Some(("verified", _)) if first_edit.and_then(|fe| reverted(path, fe)).is_none() => "indirect",
                _ => "weak",
            };
            let spans = crate::sessions::merge(t.iter().flat_map(|x| x.spans.iter().copied()).collect());
            let mut marked = false;
            for c in &mut cands {
                let at = c.get("place").and_then(|x| x.string().ok()).unwrap_or_default().to_owned();
                let (cp, cs) = ex::place(&at);
                if &cp == path && (spans.is_empty() || spans.iter().any(|sp| overlaps(cs, *sp))) {
                    c.insert("label".into(), s("went"));
                    c.insert("strength".into(), s(strength));
                    marked = true;
                }
            }
            if !marked {
                let (a, b) = spans.first().copied().unwrap_or((0, 0));
                let (a, b) = spans.iter().fold((a, b), |acc, sp| (acc.0.min(sp.0), acc.1.max(sp.1)));
                // The lines as they are now (after any edit); marked as hashed at settle.
                let hash = root.and_then(|r| lines_of(r, path, a, b, &mut cache)).map(|l| ex::content_hash(&l.iter().map(String::as_str).collect::<Vec<_>>()));
                let mut m: BTreeMap<String, Value> = BTreeMap::new();
                m.insert("place".into(), s(&place_text(path, a, b)));
                m.insert("hash".into(), opt(hash.as_deref()));
                m.insert("hash_at".into(), s("settle"));
                m.insert("judged".into(), Value::Bool(false));
                m.insert("served".into(), Value::Bool(false));
                m.insert("label".into(), s("went"));
                m.insert("strength".into(), s(strength));
                cands.push(m);
            }
        }
    }
    let label_of = |c: &BTreeMap<String, Value>| c.get("label").and_then(|x| x.string().ok()).unwrap_or_default().to_owned();
    let strength_of = |c: &BTreeMap<String, Value>| c.get("strength").and_then(|x| x.string().ok()).unwrap_or_default().to_owned();
    let path_of = |c: &BTreeMap<String, Value>| ex::place(c.get("place").and_then(|x| x.string().ok()).unwrap_or_default()).0;
    let best = |l: &str| cands.iter().filter(|c| label_of(c) == l).map(strength_of).max_by_key(|x| ex::strength_rank(x));
    let went_true = cands.iter().find(|c| label_of(c) == "went" && strength_of(c) == "indirect");
    let answered = root.and_then(|r| crate::outcomes::answered(r, recs, st));
    let (verdict, strength, by) = if let Some(_c) = went_true {
        ("corrected", "indirect".to_owned(), "went".to_owned())
    } else if let Some(o) = &answered {
        (if o.verified { "verified" } else { "refuted" }, o.strength.clone(), o.rule.clone())
    } else if let Some(sg) = best("refuted") {
        ("refuted", sg, "labels".to_owned())
    } else if let Some(sg) = best("verified") {
        ("verified", sg, "labels".to_owned())
    } else if elsewhere {
        ("passed-over", "weak".to_owned(), "elsewhere".to_owned())
    } else {
        return None;
    };
    let anchor = match verdict {
        "corrected" => went_true.map(path_of),
        "verified" => cands.iter().find(|c| label_of(c) == "verified").or_else(|| cands.iter().find(|c| label_of(c) == "used")).map(path_of),
        "refuted" => cands.iter().find(|c| label_of(c) == "refuted").map(path_of),
        _ => cands.iter().find(|c| label_of(c) == "went").or_else(|| cands.iter().find(|c| label_of(c) == "passed-over")).map(path_of),
    }
    .unwrap_or_default();
    let ask = p.rec.nano.as_ref().and_then(|x| x.1.clone());
    let ask_sha = ask.as_deref().map(|a| sha16(&norm(a))).or_else(|| judged.and_then(|j| text(j, "ask_sha"))).unwrap_or_default();
    let split = ex::split_of(&ask_sha, &anchor);
    let labelled: Vec<String> = cands.iter().filter(|c| label_of(c) != "judged").map(path_of).collect();
    let paths: Vec<&str> = labelled.iter().map(String::as_str).collect();
    let mut key_parts: Vec<String> = cands
        .iter()
        .filter(|c| label_of(c) != "judged")
        .map(|c| format!("{}:{}", c.get("hash").and_then(|x| x.string().ok()).map_or_else(|| c.get("place").and_then(|x| x.string().ok()).unwrap_or_default().to_owned(), str::to_owned), label_of(c)))
        .collect();
    key_parts.sort();
    let content_key = sha16(&format!("{ask_sha}|{verdict}|{}", key_parts.join(",")));
    let settings = crate::ledger::settings_at(idx, sv);
    let cartridges = settings.get("cartridges").ok().cloned().unwrap_or(Value::Array(Vec::new()));
    let (uses, loaded) = stacks_of(starts, &sv.server, sv.time_ms);
    let has_gpu = !matches!(hw.get("gpu"), Ok(Value::Null) | Err(_));
    let mut used = vec![s("cpu")];
    if has_gpu && sv.mode != "plain" && !sv.mode.is_empty() {
        used.push(s("gpu"));
    }
    let hardware = obj(vec![
        ("id", opt(text(hw, "id").as_deref())),
        ("cpu", opt(text(hw, "cpu").as_deref())),
        ("threads", hw.get("threads").ok().cloned().unwrap_or(Value::Null)),
        ("gpu", opt(text(hw, "gpu").as_deref())),
        ("gpu_mib", hw.get("gpu_mib").ok().cloned().unwrap_or(Value::Null)),
        ("ram_mib", hw.get("ram_mib").ok().cloned().unwrap_or(Value::Null)),
        ("used", Value::Array(used)),
    ]);
    let mut row = vec![
        ("v", n(ex::FORMAT)),
        ("row", s("example")),
        ("id", s(&format!("x-{}", p.id))),
        ("family", s("learning")),
        ("domains", Value::Array(domains(&paths, label).iter().map(|d| s(d)).collect())),
        ("split", s(split.name())),
        ("verdict", s(verdict)),
        ("strength", s(&strength)),
        ("decided_by", s(&by)),
        ("pack", s(&p.id)),
        ("session", s(&p.rec.session)),
        ("at", n(p.at)),
        ("ask_sha", s(&ask_sha)),
        ("anchor", s(&anchor)),
        ("key", s(&content_key)),
        ("engine", judged.and_then(|j| text(j, "engine")).map_or(Value::Null, |e| s(&e))),
        ("path", s(&judged.and_then(|j| text(j, "path")).unwrap_or_else(|| sv.mode.clone()))),
        ("serving", s(&sv.mode)),
        ("server", s(&sv.server)),
        ("source", s(&judged.and_then(|j| text(j, "source")).unwrap_or_else(|| "ledger".into()))),
        ("settings", settings),
        ("cartridges", cartridges),
        ("stacks", Value::Array(uses.iter().map(|x| s(x)).collect())),
        ("loaded", Value::Array(loaded.iter().map(|x| s(x)).collect())),
        ("hardware", hardware),
        ("project", s(label)),
        ("level", s(level_name(lv))),
        ("candidates", Value::Array(cands.into_iter().map(Value::Object).collect())),
    ];
    if lv == ex::FULL
        && let Some(q) = ask.or_else(|| judged.and_then(|j| text(j, "question")))
    {
        row.push(("question", s(&q)));
    }
    Some(obj(row))
}

/// Words that mark a prompt as stating a rule of the user's way of working.
const RULE_MARKERS: &[&str] = &["always ", "never ", "don't ", "do not ", "please use ", "prefer ", "from now on", "make sure ", "avoid ", "instead of "];
/// Words in a failed tool call's message that say the user refused it.
const REFUSAL_MARKERS: &[&str] = &["doesn't want to proceed", "does not want to proceed", "rejected", "denied", "declined"];

/// The preference signals in a project's records: an id and the row of each, decided now.
fn preference_signals(memory: &Path, recs: &[Rec], prompts: &BTreeMap<String, String>, label: &str, lv: u8, now: u64) -> Vec<(String, Value)> {
    let raw = crate::situation::read(memory, None, None);
    let mut out: Vec<(String, Value)> = Vec::new();
    let scope = format!("project:{label}");
    let mut push = |kind: &str, strength: &str, r: &Rec, paths: Vec<String>, extra: Vec<(&str, Value)>, key: &str| {
        let id = format!("p-{}", sha16(&format!("{kind}|{}|{key}", r.session)));
        let mut f = vec![
            ("v", n(ex::FORMAT)),
            ("row", s("preference")),
            ("family", s("preference")),
            ("id", s(&id)),
            ("kind", s(kind)),
            ("strength", s(strength)),
            ("session", s(&r.session)),
            ("at", n(r.t)),
            ("paths", Value::Array(paths.iter().map(|x| s(x)).collect())),
            ("scope", s(&scope)),
            ("split", s(ex::split_of(&r.session, kind).name())),
        ];
        f.extend(extra);
        out.push((id, obj(f)));
    };
    let mut by_session: BTreeMap<&str, Vec<&Rec>> = BTreeMap::new();
    for r in recs.iter().filter(|r| !r.session.is_empty()) {
        by_session.entry(r.session.as_str()).or_default().push(r);
    }
    for list in by_session.values() {
        let last_t = list.iter().map(|r| r.t).max().unwrap_or(0);
        let over = list.iter().any(|r| r.event == "SessionEnd") || now >= last_t.saturating_add(IDLE_MS);
        // Edits kept or reverted, per file: decided by an undo after the last edit, a passing check
        // after it, or the session being over.
        let mut edits: BTreeMap<&str, &Rec> = BTreeMap::new();
        for r in list.iter().filter(|r| r.done() && r.ok == Some(true) && matches!(r.kind.as_str(), "edit" | "write")) {
            for p in r.paths.iter().filter(|p| !matches!(p.as_str(), "." | "<outside>")) {
                edits.insert(p.as_str(), r);
            }
        }
        for (path, last) in edits {
            let after: Vec<&&Rec> = list.iter().filter(|r| (r.t, r.i) > (last.t, last.i) && r.done() && r.ok == Some(true)).collect();
            let undo = |named: bool| {
                after.iter().find(|r| {
                    matches!(r.cmd_class.as_deref(), Some("git-revert" | "git-restore" | "git-checkout" | "git-reset" | "git-stash"))
                        && if named { r.paths.iter().any(|x| x == path) } else { r.paths.is_empty() || r.paths.iter().any(|x| x == ".") }
                })
            };
            let key = format!("{path}|{}", last.t);
            if let Some(r) = undo(true) {
                push("edit-reverted", "direct", r, vec![path.to_owned()], vec![], &key);
            } else if let Some(r) = undo(false) {
                push("edit-reverted", "weak", r, vec![path.to_owned()], vec![], &key);
            } else if let Some(r) = after.iter().find(|r| matches!(r.kind.as_str(), "build" | "test")) {
                push("edit-kept", "indirect", r, vec![path.to_owned()], vec![], &key);
            } else if over {
                push("edit-kept", "weak", last, vec![path.to_owned()], vec![], &key);
            }
        }
        for r in list.iter() {
            if let Some((sha, _, _)) = &r.prompt
                && let Some(t) = prompts.get(sha)
            {
                let lower = format!(" {}", t.to_lowercase());
                let with_text = |mut v: Vec<(&'static str, Value)>| {
                    v.push(("prompt_sha", s(sha)));
                    if lv == ex::FULL {
                        v.push(("text", s(t)));
                    }
                    v
                };
                match crate::ledger::markers(t) {
                    (Some(m), None) => push("correction", "weak", r, vec![], with_text(vec![("marker", s(m))]), &r.t.to_string()),
                    (None, Some(m)) => push("acceptance", "weak", r, vec![], with_text(vec![("marker", s(m))]), &r.t.to_string()),
                    _ => {}
                }
                if let Some(m) = RULE_MARKERS.iter().find(|m| lower.contains(&format!(" {m}"))) {
                    push("stated-rule", "weak", r, vec![], with_text(vec![("marker", s(m.trim()))]), &format!("rule|{}", r.t));
                }
            }
            let v = raw.get(r.i);
            let interrupted = v.is_some_and(|v| matches!(v.get("interrupted"), Ok(Value::Bool(true))));
            let said = v.and_then(|v| text(v, "msg").or_else(|| v.get("digest").ok().and_then(|d| text(d, "first")))).map(|m| m.to_lowercase()).unwrap_or_default();
            if r.done() && r.ok == Some(false) && (interrupted || REFUSAL_MARKERS.iter().any(|m| said.contains(m))) {
                push("refusal", "direct", r, r.paths.clone(), vec![("tool", s(&r.tool)), ("by", s(if interrupted { "interrupted" } else { "refused" }))], &format!("{}|{}", r.t, r.i));
            }
        }
    }
    out
}

/// Which hardware piece a measured part ran on.
fn piece_of(part: &str, path: &str, hw: &Value) -> String {
    let gpu = text(hw, "gpu");
    let on_card = path != "plain"
        && ["nmbrain.slot#reranker", "nmbrain.slot#embedder", "nmengn", "nmapi.joint_read", "nmapi.decide", "stage:reranker", "stage:embedding", "stage:joint read", "stage:second look", "stage:load"]
            .iter()
            .any(|p| part.starts_with(p));
    match (on_card, gpu) {
        (true, Some(g)) => format!("gpu:{g}"),
        _ => format!("cpu:{}", text(hw, "cpu").unwrap_or_default()),
    }
}

/// Aggregates per hardware piece, settings and part over a project's context pack rows: count,
/// mean, spread, least and most (milliseconds; sizes in their own units).
fn hardware_measures(packs: &[Value], idx: &crate::ledger::AuditIndex, hw: &Value, now: u64) -> Value {
    #[derive(Default)]
    struct Acc {
        n: u64,
        sum: f64,
        sq: f64,
        lo: f64,
        hi: f64,
    }
    let mut acc: BTreeMap<(String, String, String), Acc> = BTreeMap::new();
    let mut through = 0;
    let by_id: BTreeMap<&str, &crate::ledger::Served> = idx.packs.iter().map(|p| (p.id.as_str(), p)).collect();
    for v in packs {
        let id = text(v, "id").unwrap_or_default();
        through = through.max(v.get("time_ms").and_then(Value::integer).unwrap_or(0));
        let cmap = v.get("context_map").ok();
        let path = cmap.and_then(|m| text(m, "mode")).unwrap_or_else(|| text(v, "mode").unwrap_or_default());
        let settings = by_id.get(id.as_str()).map(|sv| crate::ledger::settings_at(idx, sv));
        let setting_key = settings.as_ref().map_or_else(String::new, |st| {
            ["precision.base", "precision.reranker", "precision.embedder"].iter().map(|k| format!("{k}={}", text(st, k).unwrap_or_default())).collect::<Vec<_>>().join(" ")
        });
        let key = format!("path={path} {setting_key}");
        let mut add = |part: String, x: f64| {
            let a = acc.entry((piece_of(&part, &path, hw), key.clone(), part)).or_default();
            if a.n == 0 {
                a.lo = x;
                a.hi = x;
            }
            a.n += 1;
            a.sum += x;
            a.sq += x * x;
            a.lo = a.lo.min(x);
            a.hi = a.hi.max(x);
        };
        for st in v.get("steps").and_then(Value::array).unwrap_or_default() {
            let Some(part) = text(st, "part") else { continue };
            #[expect(clippy::cast_precision_loss, reason = "times for a measure")]
            let ms = st.get("us").and_then(Value::integer).map(|u| u as f64 / 1000.0).or_else(|_| st.get("ms").and_then(Value::integer).map(|m| m as f64));
            if let Ok(ms) = ms {
                add(part.split('#').next().unwrap_or(&part).to_owned(), ms);
            }
        }
        if let Some(Value::Object(stages)) = cmap.and_then(|m| m.get("stages").ok()) {
            for (k, x) in stages {
                if let Value::Number(t) = x
                    && let Ok(secs) = t.parse::<f64>()
                {
                    add(format!("stage:{k}"), secs * 1000.0);
                }
            }
        }
        for k in ["judge_tokens", "rerank_docs", "look_tokens", "judge_pieces"] {
            #[expect(clippy::cast_precision_loss, reason = "sizes for a measure")]
            if let Some(x) = cmap.and_then(|m| m.get(k).and_then(Value::integer).ok()) {
                add(format!("size:{k}"), x as f64);
            }
        }
        if let Ok(ms) = v.get("ms").and_then(Value::integer) {
            #[expect(clippy::cast_precision_loss, reason = "times for a measure")]
            add("call".to_owned(), ms as f64);
        }
    }
    let rows = acc
        .into_iter()
        .map(|((piece, settings, part), a)| {
            #[expect(clippy::cast_precision_loss, reason = "counts for a mean")]
            let mean = a.sum / a.n as f64;
            #[expect(clippy::cast_precision_loss, reason = "counts for a spread")]
            let sd = (a.sq / a.n as f64 - mean * mean).max(0.0).sqrt();
            let f = |x: f64| Value::Number(format!("{x:.3}"));
            obj(vec![("piece", s(&piece)), ("settings", s(&settings)), ("part", s(&part)), ("n", n(a.n)), ("mean", f(mean)), ("sd", f(sd)), ("min", f(a.lo)), ("max", f(a.hi))])
        })
        .collect();
    obj(vec![
        ("v", n(ex::FORMAT)),
        ("family", s("fundamentals")),
        ("domain", s("hardware")),
        ("machine", hw.clone()),
        ("units", s("parts and stages in milliseconds; size: rows in their own units (tokens, documents, pieces)")),
        ("memory", s("not measured per step: the serving path's audit rows carry times and sizes, not memory")),
        ("rows", Value::Array(rows)),
        ("through", n(through)),
        ("written", n(now)),
    ])
}

/// Brings the training store up to date: an example for every closed pack not yet handled, the
/// preference signals decided by now, the hardware measurements, then retention by run-days. Nothing with memory
/// off or the setting `off`. Called by the ledger after it settles.
/// # Errors
/// The store cannot be read or written.
#[allow(clippy::too_many_arguments)]
pub fn settle(home: &Path, memory: &Path, label: &str, root: Option<&Path>, recs: &[Rec], states: &[PackState], now: u64, wait_ms: u64) -> Result<Report, String> {
    let mut rep = Report::default();
    let lv = level_of(home);
    if crate::controls::memory_off(home) || lv == ex::OFF {
        return Ok(rep);
    }
    let tdir = ex::dir(memory);
    let cleared = cleared_at(memory);
    let done = seen(&tdir);
    let todo: Vec<&PackState> =
        states.iter().filter(|st| st.closed.is_some() && st.pack.served.is_some() && st.pack.rec.t > cleared && !done.contains(&format!("pack:{}", st.pack.id))).collect();
    let prompts = crate::sessions::kept_prompts(home, memory);
    let prefs: Vec<(String, Value)> = preference_signals(memory, recs, &prompts, label, lv, now)
        .into_iter()
        .filter(|(id, v)| !done.contains(&format!("pref:{id}")) && v.get("at").and_then(Value::integer).unwrap_or(0) > cleared)
        .collect();
    if todo.is_empty() && prefs.is_empty() {
        return Ok(rep);
    }
    let _g = lock(&tdir, wait_ms)?;
    let judged: BTreeMap<String, Value> = ex::read_rows(memory, ex::JUDGED)?.into_iter().filter_map(|j| Some((text(&j, "pack")?, j))).collect();
    let idx = crate::ledger::audit_index(home, label);
    let (starts, pack_rows) = audit_rows(home, label);
    let hw = crate::ledger::hardware(home, false);
    let mut keys = done;
    let mut new_seen: Vec<String> = Vec::new();
    let mut new_rows: Vec<Value> = Vec::new();
    for st in todo {
        new_seen.push(format!("pack:{}", st.pack.id));
        let Some(e) = example_of(st, recs, judged.get(&st.pack.id), &idx, &starts, &hw, root, label, lv, now) else {
            rep.no_verdict += 1;
            continue;
        };
        let key = format!("key:{}", text(&e, "key").unwrap_or_default());
        if keys.contains(&key) {
            rep.duplicates += 1;
            continue;
        }
        keys.insert(key.clone());
        new_seen.push(key);
        new_rows.push(e);
    }
    rep.examples = new_rows.len();
    append_lines(&tdir.join(ex::EXAMPLES), &new_rows)?;
    rep.preferences = prefs.len();
    new_seen.extend(prefs.iter().map(|(id, _)| format!("pref:{id}")));
    append_lines(&tdir.join(ex::PREFERENCES), &prefs.into_iter().map(|x| x.1).collect::<Vec<_>>())?;
    if !new_seen.is_empty() {
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(tdir.join(SEEN)).map_err(|e| e.to_string())?;
        f.write_all(format!("{}\n", new_seen.join("\n")).as_bytes()).map_err(|e| e.to_string())?;
    }
    if !pack_rows.is_empty() {
        let measures = hardware_measures(&pack_rows, &idx, &hw, now);
        std::fs::write(tdir.join(ex::HARDWARE), nmjson::pretty::pretty(&measures)).map_err(|e| e.to_string())?;
    }
    rep.pruned = prune(home, memory, now)?;
    // 2026-11-25: every settle brings in what the store holds and the engine has not taken yet,
    // not only the rows this settle wrote, so stored examples never wait for a new one.
    if nmapi::whole::learning_control(home) && engine_backlog(memory, &nmlearn::engine::ContinuousLearningEngine::open(memory)) > 0 {
        let mut eng = nmlearn::engine::ContinuousLearningEngine::open(memory);
        if sync_engine_from_store(home, memory, &mut eng) {
            let _ = eng.save(memory);
        }
        // In `auto` mode a cycle runs once enough verified examples have settled since the last
        // one, through the gated and recorded idle cycle (G3 is off in a hook's process, so its
        // replay is recorded as not run there).
        if nmapi::whole::learning_cycle_control(home) == "auto" && nmlearn::idle::due(memory, nmlearn::idle::threshold()) {
            let key = project_key(home, memory);
            let _ = run_unified_cycle(home, memory, &key, "settle", "threshold");
        }
    }
    Ok(rep)
}

/// Keeps the store by the run-day rule of retention (`nmmem::keep::training_keeps`): judged sets
/// already used, or unused past 7 run-days, go; examples and preference signals past 28 run-days go,
/// except a verified example, which is kept. Nothing goes for size. Passages and views no longer
/// referenced go with them. Returns the rows removed. The caller holds the lock.
fn prune(home: &Path, memory: &Path, now: u64) -> Result<usize, String> {
    let tdir = ex::dir(memory);
    let days = nmmem::keep::run_days(&home.join("memory").join("audit.jsonl"), now);
    let at = |v: &Value| v.get("at").and_then(Value::integer).unwrap_or(0);
    let keeps = |file: &str, v: &Value| nmmem::keep::training_keeps(&days, file, at(v), &text(v, "verdict").unwrap_or_default());
    let done = seen(&tdir);
    let mut removed = 0;
    let judged = ex::read_rows(memory, ex::JUDGED)?;
    let keep_j: Vec<Value> = judged.iter().filter(|j| !text(j, "pack").is_some_and(|p| done.contains(&format!("pack:{p}"))) && keeps(ex::JUDGED, j)).cloned().collect();
    if keep_j.len() != judged.len() {
        removed += judged.len() - keep_j.len();
        rewrite_lines(&tdir.join(ex::JUDGED), &keep_j)?;
    }
    let mut kept = |file: &str| -> Result<Vec<Value>, String> {
        let rows = ex::read_rows(memory, file)?;
        let keep: Vec<Value> = rows.iter().filter(|r| keeps(file, r)).cloned().collect();
        if keep.len() != rows.len() {
            removed += rows.len() - keep.len();
            rewrite_lines(&tdir.join(file), &keep)?;
        }
        Ok(keep)
    };
    let examples = kept(ex::EXAMPLES)?;
    kept(ex::PREFERENCES)?;
    // Passages and views referenced by what remains.
    let live_packs: BTreeSet<String> = examples.iter().chain(&keep_j).filter_map(|r| text(r, "pack")).collect();
    let live_hashes: BTreeSet<String> =
        examples.iter().chain(&keep_j).flat_map(|r| r.get("candidates").and_then(Value::array).unwrap_or_default().iter().filter_map(|c| text(c, "hash")).collect::<Vec<_>>()).collect();
    let passages = ex::read_rows(memory, ex::PASSAGES)?;
    let keep_p: Vec<Value> = passages.iter().filter(|p| text(p, "hash").is_some_and(|h| live_hashes.contains(&h))).cloned().collect();
    if keep_p.len() != passages.len() {
        rewrite_lines(&tdir.join(ex::PASSAGES), &keep_p)?;
    }
    if let Ok(rd) = std::fs::read_dir(tdir.join(ex::VIEWS)) {
        for e in rd.flatten() {
            let Ok(b) = std::fs::read(e.path()) else { continue };
            let Ok(all) = ex::parse_views(&b) else { continue };
            let keep: Vec<&ex::View> = all.iter().filter(|v| live_packs.contains(&v.0)).collect();
            if keep.len() != all.len() {
                let mut out = ex::VIEWS_MAGIC.to_vec();
                for (p, i, v) in keep {
                    out.extend(ex::view_record(p, *i, v));
                }
                std::fs::write(e.path(), out).map_err(|x| x.to_string())?;
            }
        }
    }
    Ok(removed)
}

// ---------------------------------------------------------------------------------------------
// Clear, export, import, references.

/// Removes a project's training store and marks the time, so packs and records before it are not
/// written again from what remains. Logged as a `privacy` audit row. Returns the rows removed and
/// the audit id.
/// # Errors
/// Another writer holds the store, or a file cannot be removed or written.
pub fn clear(home: &Path, memory: &Path, label: &str, by: &str) -> Result<(usize, String), String> {
    let tdir = ex::dir(memory);
    let rows = if tdir.is_dir() {
        let g = lock(&tdir, 3000)?;
        let count = [ex::EXAMPLES, ex::PREFERENCES, ex::JUDGED, ex::PASSAGES].iter().map(|f| ex::read_rows(memory, f).map_or(0, |r| r.len())).sum();
        drop(g);
        std::fs::remove_dir_all(&tdir).map_err(|e| format!("{}: {e}", tdir.display()))?;
        count
    } else {
        0
    };
    if memory.is_dir() {
        std::fs::write(memory.join(CLEARED), format!("{}\nTraining store cleared at this time (ms); packs and records before it are not written again.\n", nmtype::now_ms()))
            .map_err(|e| e.to_string())?;
    }
    let id = crate::controls::audit(
        home,
        "privacy",
        vec![("label", s(&format!("Training store cleared: {rows} rows"))), ("project", s(label)), ("cleared", n(rows as u64)), ("by", s(by))],
    )?;
    Ok((rows, id))
}

/// Writes a project's whole store to one file, for moving it between the owner's own machines: a
/// header line (project, time, row count, SHA-256 of the lines after it), then one line per row
/// naming its file. Views go as hexadecimal. Returns the rows written.
/// # Errors
/// The store cannot be read or the file written.
pub fn export(home: &Path, memory: &Path, label: &str, file: &Path) -> Result<usize, String> {
    let tdir = ex::dir(memory);
    let mut lines: Vec<String> = Vec::new();
    for f in [ex::EXAMPLES, ex::PREFERENCES, ex::PASSAGES, ex::JUDGED] {
        for r in ex::read_rows(memory, f)? {
            lines.push(obj(vec![("file", s(f)), ("row", r)]).encode().map_err(|e| e.to_string())?);
        }
    }
    for f in [ex::ENGINES, ex::HARDWARE] {
        if let Ok(b) = std::fs::read(tdir.join(f))
            && let Ok(v) = nmjson::parse(&b)
        {
            lines.push(obj(vec![("file", s(f)), ("row", v)]).encode().map_err(|e| e.to_string())?);
        }
    }
    if let Ok(rd) = std::fs::read_dir(tdir.join(ex::VIEWS)) {
        for e in rd.flatten() {
            if let Ok(b) = std::fs::read(e.path()) {
                let hex: String = b.iter().map(|x| format!("{x:02x}")).collect();
                lines.push(obj(vec![("file", s(&format!("{}/{}", ex::VIEWS, e.file_name().to_string_lossy()))), ("hex", s(&hex))]).encode().map_err(|e| e.to_string())?);
            }
        }
    }
    let body = if lines.is_empty() { String::new() } else { format!("{}\n", lines.join("\n")) };
    let head = obj(vec![
        ("kind", s("neuromorph-nano-training-export")),
        ("v", n(ex::FORMAT)),
        ("project", s(label)),
        ("created", n(nmtype::now_ms())),
        ("rows", n(lines.len() as u64)),
        ("sha256", s(&nmhash::sha256_bytes(body.as_bytes()))),
    ]);
    std::fs::write(file, format!("{}\n{body}", head.encode().map_err(|e| e.to_string())?)).map_err(|e| format!("{}: {e}", file.display()))?;
    let _ = crate::controls::audit(home, "training", vec![("label", s(&format!("Training store exported: {} rows", lines.len()))), ("project", s(label)), ("rows", n(lines.len() as u64))]);
    Ok(lines.len())
}

/// Reads an export into a project's store: the hash is checked first; rows already there (by id,
/// pack or passage hash) are passed over. Returns the rows added.
/// # Errors
/// The file is not an export, its hash does not match, or the store cannot be written.
pub fn import(home: &Path, memory: &Path, label: &str, file: &Path) -> Result<usize, String> {
    let t = std::fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?;
    let (head, body) = t.split_once('\n').ok_or("not an export: no header line")?;
    let h = nmjson::parse(head.as_bytes()).map_err(|e| format!("header: {e}"))?;
    if text(&h, "kind").as_deref() != Some("neuromorph-nano-training-export") {
        return Err("not a training store export".into());
    }
    if text(&h, "sha256").as_deref() != Some(nmhash::sha256_bytes(body.as_bytes()).as_str()) {
        return Err("the export's hash does not match its rows; nothing was imported".into());
    }
    let tdir = ex::dir(memory);
    let _g = lock(&tdir, 3000)?;
    let ident = |file: &str, r: &Value| match file {
        f if f == ex::PASSAGES => text(r, "hash"),
        f if f == ex::JUDGED => text(r, "pack"),
        _ => text(r, "id"),
    };
    let mut have: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for f in [ex::EXAMPLES, ex::PREFERENCES, ex::PASSAGES, ex::JUDGED] {
        have.insert(f.to_owned(), ex::read_rows(memory, f)?.iter().filter_map(|r| ident(f, r)).collect());
    }
    let mut add: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    let mut added = 0;
    for l in body.lines().filter(|l| !l.trim().is_empty()) {
        let v = nmjson::parse(l.as_bytes()).map_err(|e| format!("row: {e}"))?;
        let f = text(&v, "file").unwrap_or_default();
        if let Some(hex) = text(&v, "hex") {
            let name = f.strip_prefix(&format!("{}/", ex::VIEWS)).filter(|x| !x.contains('/') && !x.contains("..") && x.ends_with(".bin")).ok_or("views: bad name")?;
            let bytes: Vec<u8> = (0..hex.len() / 2).filter_map(|i| hex.get(i * 2..i * 2 + 2).and_then(|x| u8::from_str_radix(x, 16).ok())).collect();
            let incoming = ex::parse_views(&bytes)?;
            let vdir = tdir.join(ex::VIEWS);
            std::fs::create_dir_all(&vdir).map_err(|e| e.to_string())?;
            let path = vdir.join(name);
            let existing = std::fs::read(&path).ok().and_then(|b| ex::parse_views(&b).ok()).unwrap_or_default();
            let mut out = if path.exists() { std::fs::read(&path).map_err(|e| e.to_string())? } else { ex::VIEWS_MAGIC.to_vec() };
            for (p, i, vals) in incoming.iter().filter(|x| !existing.iter().any(|y| y.0 == x.0 && y.1 == x.1)) {
                out.extend(ex::view_record(p, *i, vals));
                added += 1;
            }
            std::fs::write(&path, out).map_err(|e| e.to_string())?;
            continue;
        }
        let row = v.get("row").map_err(|e| e.to_string())?.clone();
        if f == ex::ENGINES {
            let mut all = ex::read_engines(memory).unwrap_or_default();
            if let Value::Object(m) = row {
                all.extend(m);
            }
            std::fs::write(tdir.join(ex::ENGINES), nmjson::pretty::pretty(&Value::Object(all))).map_err(|e| e.to_string())?;
            continue;
        }
        if f == ex::HARDWARE {
            if !tdir.join(ex::HARDWARE).exists() {
                std::fs::write(tdir.join(ex::HARDWARE), nmjson::pretty::pretty(&row)).map_err(|e| e.to_string())?;
            }
            continue;
        }
        if ![ex::EXAMPLES, ex::PREFERENCES, ex::PASSAGES, ex::JUDGED].contains(&f.as_str()) {
            return Err(format!("unknown file {f} in the export"));
        }
        let id = ident(&f, &row).unwrap_or_default();
        if have.entry(f.clone()).or_default().insert(id) {
            add.entry(f).or_default().push(row);
            added += 1;
        }
    }
    for (f, rows) in &add {
        append_lines(&tdir.join(f), rows)?;
    }
    let _ = crate::controls::audit(home, "training", vec![("label", s(&format!("Training store imported: {added} rows"))), ("project", s(label)), ("rows", n(added as u64))]);
    Ok(added)
}

/// The registry of reference material for fundamentals cartridges, in the Nano's own memory folder.
fn references_file(home: &Path) -> PathBuf {
    ex::dir(&home.join("memory")).join(ex::REFERENCES)
}

fn read_references(home: &Path) -> Vec<Value> {
    std::fs::read(references_file(home)).ok().and_then(|b| nmjson::parse(&b).ok()).and_then(|v| v.get("references").and_then(Value::array).ok().map(<[Value]>::to_vec)).unwrap_or_default()
}

fn write_references(home: &Path, refs: Vec<Value>) -> Result<(), String> {
    let f = references_file(home);
    if let Some(d) = f.parent() {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    let v = obj(vec![
        ("v", n(ex::FORMAT)),
        ("family", s("fundamentals")),
        ("note", s("Reference material for fundamentals cartridges: where it is on this machine, its licence and hash, what was fitted from it. The material itself is never copied here.")),
        ("references", Value::Array(refs)),
    ]);
    std::fs::write(&f, nmjson::pretty::pretty(&v)).map_err(|e| format!("{}: {e}", f.display()))
}

/// The SHA-256 of a file, or of a folder as the hash of its sorted `relative-path sha256` lines,
/// with the number of files.
fn material_hash(p: &Path) -> Result<(String, u64), String> {
    if p.is_file() {
        let h = std::fs::File::open(p).and_then(nmhash::sha256).map_err(|e| format!("{}: {e}", p.display()))?;
        return Ok((h, 1));
    }
    let mut lines: Vec<String> = Vec::new();
    let mut stack = vec![p.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).map_err(|e| format!("{}: {e}", d.display()))?.flatten() {
            let path = e.path();
            if path.is_dir() {
                stack.push(path);
            } else if lines.len() < 100_000 {
                let h = std::fs::File::open(&path).and_then(nmhash::sha256).map_err(|e| format!("{}: {e}", path.display()))?;
                lines.push(format!("{} {h}", path.strip_prefix(p).unwrap_or(&path).display().to_string().replace('\\', "/")));
            }
        }
    }
    lines.sort();
    Ok((nmhash::sha256_bytes(lines.join("\n").as_bytes()), lines.len() as u64))
}

/// Adds reference material to the registry (path, licence, source, domain, hash); audited.
/// # Errors
/// The path cannot be read or the registry written.
pub fn reference_add(home: &Path, path: &Path, licence: &str, source: &str, domain: &str) -> Result<String, String> {
    let abs = nmsubs::repo::canon(path);
    let (hash, files) = material_hash(&abs)?;
    let id = format!("r-{}", hash.get(..12).unwrap_or(""));
    let mut refs: Vec<Value> = read_references(home).into_iter().filter(|r| text(r, "id").as_deref() != Some(id.as_str())).collect();
    refs.push(obj(vec![
        ("id", s(&id)),
        ("source", s(source)),
        ("path", s(&abs.display().to_string())),
        ("kind", s(if abs.is_file() { "file" } else { "folder" })),
        ("files", n(files)),
        ("licence", s(licence)),
        ("domain", s(domain)),
        ("sha256", s(&hash)),
        ("added", n(nmtype::now_ms())),
        ("fitted", Value::Array(Vec::new())),
    ]));
    write_references(home, refs)?;
    crate::controls::audit(home, "training", vec![("label", s(&format!("Reference material listed: {source} ({licence})"))), ("reference", s(&id)), ("sha256", s(&hash))])?;
    Ok(id)
}

/// Records that a cartridge was fitted from a listed reference; audited.
/// # Errors
/// No such reference, or the registry cannot be written.
pub fn reference_fitted(home: &Path, id: &str, cartridge: &str) -> Result<(), String> {
    let mut refs = read_references(home);
    let r = refs.iter_mut().find(|r| text(r, "id").as_deref() == Some(id)).ok_or_else(|| format!("no reference {id}"))?;
    if let Value::Object(m) = r {
        let list = m.entry("fitted".to_owned()).or_insert_with(|| Value::Array(Vec::new()));
        if let Value::Array(a) = list
            && !a.iter().any(|x| x.string().ok() == Some(cartridge))
        {
            a.push(s(cartridge));
        }
    }
    write_references(home, refs)?;
    crate::controls::audit(home, "training", vec![("label", s(&format!("Cartridge {cartridge} fitted from reference {id}"))), ("reference", s(id)), ("cartridge", s(cartridge))]).map(|_| ())
}

/// `GET /api/training/references` (2026-11-25, Luis Jake Gabriel III: "Allow users to prep their
/// own air gapped data for their own training"): the reference material listed on this machine,
/// each with whether its path is still there. Nothing is copied; the material stays where it is.
#[must_use]
pub fn references_view(home: &Path) -> Value {
    let refs: Vec<Value> = read_references(home)
        .into_iter()
        .map(|mut r| {
            let present = text(&r, "path").is_some_and(|p| Path::new(&p).exists());
            if let Value::Object(m) = &mut r {
                m.insert("present".into(), Value::Bool(present));
            }
            r
        })
        .collect();
    obj(vec![("references", Value::Array(refs)), ("registry", s(&references_file(home).display().to_string()))])
}

/// The folder training store exports are written to (`<home>/exports/`).
#[must_use]
pub fn exports_dir(home: &Path) -> PathBuf {
    home.join("exports")
}

/// Exports a project's training store to `<home>/exports/<name>` (a plain name ending in
/// `.jsonl`, else one made from the project and the time), for carrying to an air-gapped machine.
/// Returns the file written and the rows.
/// # Errors
/// A bad name, or the store cannot be read or the file written.
pub fn export_to_exports(home: &Path, memory: &Path, label: &str, name: Option<&str>) -> Result<(PathBuf, usize), String> {
    let name = match name.filter(|n| !n.is_empty()) {
        Some(n) if n.ends_with(".jsonl") && n.len() <= 120 && !n.contains(['/', '\\', ':']) && !n.starts_with('.') => n.to_owned(),
        Some(n) => return Err(format!("{n}: give a plain file name ending in .jsonl")),
        None => format!("{}-training-{}.jsonl", label.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '-' }).collect::<String>(), nmtype::now_ms()),
    };
    let dir = exports_dir(home);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let file = dir.join(name);
    let rows = export(home, memory, label, &file)?;
    Ok((file, rows))
}

// ---------------------------------------------------------------------------------------------
// Views.

fn count_into(m: &mut BTreeMap<String, u64>, k: &str) {
    *m.entry(k.to_owned()).or_default() += 1;
}

/// Counts and figures of a project's store, as JSON.
#[must_use]
pub fn summary(home: &Path, memory: &Path, key: &str) -> Value {
    let examples = ex::read_rows(memory, ex::EXAMPLES).unwrap_or_default();
    let prefs = ex::read_rows(memory, ex::PREFERENCES).unwrap_or_default();
    let judged = ex::read_rows(memory, ex::JUDGED).unwrap_or_default();
    let (mut verdict, mut strength, mut split, mut domain, mut label, mut engine, mut pkind) =
        (BTreeMap::new(), BTreeMap::new(), BTreeMap::new(), BTreeMap::new(), BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
    for e in &examples {
        count_into(&mut verdict, &text(e, "verdict").unwrap_or_default());
        count_into(&mut strength, &text(e, "strength").unwrap_or_default());
        count_into(&mut split, &text(e, "split").unwrap_or_default());
        count_into(&mut engine, &text(e, "engine").unwrap_or_else(|| "none (plain path)".into()));
        for d in e.get("domains").and_then(Value::array).unwrap_or_default() {
            count_into(&mut domain, d.string().unwrap_or_default());
        }
        for c in e.get("candidates").and_then(Value::array).unwrap_or_default() {
            count_into(&mut label, &text(c, "label").unwrap_or_default());
        }
    }
    for p in &prefs {
        count_into(&mut pkind, &text(p, "kind").unwrap_or_default());
    }
    let hw_rows = ex::read_hardware(memory).ok().flatten().and_then(|h| h.get("rows").and_then(Value::array).ok().map(<[Value]>::len)).unwrap_or(0);
    let tdir = ex::dir(memory);
    let bytes: u64 = std::fs::read_dir(&tdir).map(|rd| rd.flatten().map(|e| e.metadata().map_or(0, |m| m.len())).sum()).unwrap_or(0)
        + std::fs::read_dir(tdir.join(ex::VIEWS)).map(|rd| rd.flatten().map(|e| e.metadata().map_or(0, |m| m.len())).sum()).unwrap_or(0);
    let m = |x: BTreeMap<String, u64>| Value::Object(x.into_iter().map(|(k, v)| (k, n(v))).collect());
    obj(vec![
        ("project", s(key)),
        ("folder", s(&tdir.display().to_string())),
        ("setting", s(level_name(level_of(home)))),
        ("memory", s(if crate::controls::memory_off(home) { "off" } else { "on" })),
        (
            "families",
            obj(vec![("learning", n(examples.len() as u64)), ("preference", n(prefs.len() as u64)), ("fundamentals", obj(vec![("hardware_rows", n(hw_rows as u64)), ("references", n(read_references(home).len() as u64))]))]),
        ),
        ("judged_waiting", n(judged.len() as u64)),
        ("verdicts", m(verdict)),
        ("strengths", m(strength)),
        ("splits", m(split)),
        ("domains", m(domain)),
        ("labels", m(label)),
        ("engines", m(engine)),
        ("preference_kinds", m(pkind)),
        ("bytes", n(bytes)),
    ])
}

/// `/api/examples?project=<key>[&id=<example id>][&limit=<n>]`: the store's counts and its newest
/// examples (or one). Read only.
#[must_use]
pub fn api_view(home: &Path, query: &str) -> Value {
    let Some((key, memory)) = crate::ledger::project_memory(home, query) else {
        return obj(vec![("error", s("project: give the key of a project as /api/projects lists it"))]);
    };
    let mut v = summary(home, &memory, &key);
    let rows = ex::read_rows(&memory, ex::EXAMPLES).unwrap_or_default();
    let picked: Vec<Value> = match crate::project::query_arg(query, "id").filter(|x| !x.is_empty()) {
        Some(id) => rows.into_iter().filter(|r| text(r, "id").as_deref() == Some(id.as_str())).collect(),
        None => {
            let limit = crate::project::query_arg(query, "limit").and_then(|l| l.parse::<usize>().ok()).unwrap_or(50).min(2000);
            let skip = rows.len().saturating_sub(limit);
            rows.into_iter().skip(skip).collect()
        }
    };
    if let Value::Object(m) = &mut v {
        m.insert("examples".into(), Value::Array(picked));
        m.insert("engines_known".into(), Value::Object(ex::read_engines(&memory).unwrap_or_default()));
    }
    v
}

/// Rows of the training store the project's learning engine has not taken in yet.
fn engine_backlog(memory: &Path, eng: &nmlearn::engine::ContinuousLearningEngine) -> u64 {
    nmlearn::idle::backlog(memory, eng)
}

/// Feeds settled training examples (`training/examples.jsonl`) and preference signals
/// (`training/preferences.jsonl`) into the project's `ContinuousLearningEngine` when `learning`
/// is `"on"` and memory is on (`nmlearn::idle::sync_engine`).
fn sync_engine_from_store(home: &Path, memory: &Path, eng: &mut nmlearn::engine::ContinuousLearningEngine) -> bool {
    if !nmapi::whole::learning_control(home) || crate::controls::memory_off(home) {
        return false;
    }
    nmlearn::idle::sync_engine(memory, eng)
}

/// `/api/learning?project=<key>`: the project's 3-level Continuous Learning Engine & Supervisor
/// state (episodic, semantic, procedural, preference and adaptive memory, extracted patterns,
/// scoped behavioral policies, 6 continual learning cartridges, 8 evaluation dimensions, and
/// 10 benchmark metrics), with the idle learning cycle's standing (verified examples since the
/// last cycle, the threshold, the cycles recorded). Read only: it never syncs, cycles or saves.
#[must_use]
pub fn learning_api_view(home: &Path, query: &str) -> Value {
    let Some((key, memory)) = crate::ledger::project_memory(home, query) else {
        return obj(vec![("error", s("project: give the key of a project as /api/projects lists it"))]);
    };
    let eng = nmlearn::engine::ContinuousLearningEngine::open(&memory);
    let mut v = eng.view_value();
    if let Value::Object(m) = &mut v {
        m.insert("backlog".into(), n(engine_backlog(&memory, &eng)));
        m.insert("project".into(), s(&key));
        m.insert("learning".into(), s(if nmapi::whole::learning_control(home) { "on" } else { "off" }));
        m.insert("cycle_mode".into(), s(&nmapi::whole::learning_cycle_control(home)));
        m.insert("idle_cycle".into(), idle_standing(&memory));
    }
    v
}

/// The idle learning cycle's standing for a project: verified examples since the last cycle, the
/// threshold, whether a cycle is due, and the newest cycles (with whether each was rolled back).
fn idle_standing(memory: &Path) -> Value {
    let threshold = nmlearn::idle::threshold();
    let done = nmlearn::idle::rolled_back(memory);
    let cycles = nmlearn::idle::cycles(memory);
    let skip = cycles.len().saturating_sub(20);
    obj(vec![
        ("verified_since_last_cycle", n(nmlearn::idle::verified_since_last_cycle(memory) as u64)),
        ("threshold", n(threshold as u64)),
        ("due", Value::Bool(nmlearn::idle::due(memory, threshold))),
        (
            "cycles",
            Value::Array(
                cycles
                    .iter()
                    .skip(skip)
                    .map(|c| {
                        let mut v = c.to_value();
                        if let Value::Object(m) = &mut v {
                            m.insert("rolled_back".into(), Value::Bool(done.contains(&c.id)));
                        }
                        v
                    })
                    .collect(),
            ),
        ),
    ])
}

/// The key `/api/projects` lists a project memory folder under: `nano` for the Nano's own memory,
/// else the folder's name.
fn project_key(home: &Path, memory: &Path) -> String {
    if memory == home.join("memory") {
        return "nano".into();
    }
    memory.file_name().map_or_else(|| "nano".into(), |f| f.to_string_lossy().into_owned())
}

/// The idle cycle's learners in their order: G3's prior replay (runs only where G3 is on), the
/// preference cartridge fit, the continual learning engine (fed from the training store when
/// learning and memory are on).
fn cycle_learners(home: &Path, memory: &Path, key: &str) -> Vec<Box<dyn nmlearn::idle::Learner>> {
    let g3 = Box::new(nmapi::g3::G3Learner { audit: home.join("memory").join("audit.jsonl"), memory: memory.to_path_buf() });
    let sync = nmapi::whole::learning_control(home) && !crate::controls::memory_off(home);
    nmlearn::idle::standard_learners(home, memory, key, Some(g3), sync)
}

/// One unified learning cycle (`nmlearn::idle::run_cycle`) with its audit row. Returns the cycle's
/// record.
/// # Errors
/// Another cycle runs for the project, or the records cannot be written.
pub fn run_unified_cycle(home: &Path, memory: &Path, key: &str, by: &str, trigger: &str) -> Result<nmlearn::idle::CycleRecord, String> {
    let mut learners = cycle_learners(home, memory, key);
    let mut audit = |rec: &Value| {
        let field = |k: &str| rec.get(k).cloned().unwrap_or(Value::Null);
        crate::controls::audit(
            home,
            "learning-cycle",
            vec![("project", s(key)), ("action", s("cycle")), ("cycle", field("id")), ("ran", field("ran")), ("trigger", s(trigger)), ("by", s(by))],
        )
        .ok()
    };
    // Roadmap Phase 4.5 (2026-11-24): the unified learning cycle in the live view (the process
    // that runs it: the serving process at idle, or the interface server on request).
    let mut live = nmapi::live::stage_as("learn", "learn.cycle", "", if trigger == "manual" { "manual" } else { "threshold" });
    let out = nmlearn::idle::run_cycle(memory, &mut learners, by, trigger, &mut audit);
    match &out {
        Ok(rec) => {
            live.count("verified", rec.verified);
            live.count("learners", rec.learners.len());
            live.count("ran", rec.learners.iter().filter(|l| l.ran).count());
        }
        Err(_) => live.label("failed"),
    }
    out
}

/// What `POST /api/learning/cycle` asks for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CycleAsk {
    /// Run one unified cycle.
    Cycle,
    /// Bring the store's examples the engine has not taken yet into it (Cycle A), without a cycle.
    Sync,
    /// Roll back `cycle` (else the newest cycle not yet rolled back); with `version` and no
    /// `cycle`, the engine alone to that supervisor snapshot (the earlier form).
    Rollback { version: Option<u64>, cycle: Option<String> },
}

impl CycleAsk {
    /// The ask a request's fields make (`action`, `version`, `cycle`).
    #[must_use]
    pub fn of(action: &str, version: Option<u64>, cycle: Option<&str>) -> Self {
        match action {
            "rollback" => Self::Rollback { version, cycle: cycle.filter(|c| !c.is_empty()).map(str::to_owned) },
            "sync" => Self::Sync,
            _ => Self::Cycle,
        }
    }
}

/// `POST /api/learning/cycle`: runs one unified learning cycle (G3's prior replay, the preference
/// cartridge fit and the engine's Cycle B, each promoted only when its held-out score beats the
/// current one), or, with `action == "rollback"`, rolls back one cycle (`cycle`, else the newest
/// not yet rolled back), restoring every learner it touched. A `version` with a rollback and no
/// `cycle` rolls back the engine alone to that supervisor snapshot (the earlier form). Each is
/// recorded with an audit id. Returns the engine's state with the cycle's record.
/// # Errors
/// If rollback fails, another cycle runs, or the state cannot be written.
pub fn learning_cycle_api_at(home: &Path, memory: &Path, key: &str, ask: &CycleAsk, by: &str) -> Result<Value, String> {
    let (record, audit_id) = match ask {
        CycleAsk::Rollback { version: Some(ver), cycle: None } => {
            let ver = *ver;
            let mut eng = nmlearn::engine::ContinuousLearningEngine::open(memory);
            let got = eng.rollback(Some(ver)).map_err(|e| e.to_string())?;
            eng.save(memory).map_err(|e| e.to_string())?;
            let label = format!("Continuous learning engine rolled back to v{got}");
            let audit = crate::controls::audit(home, "learning-cycle", vec![("project", s(key)), ("action", s("rollback")), ("version", n(got)), ("by", s(by)), ("label", s(&label))])?;
            (obj(vec![("row", s("engine-rollback")), ("version", n(got)), ("label", s(&label))]), audit)
        }
        CycleAsk::Rollback { cycle, .. } => {
            let mut learners = cycle_learners(home, memory, key);
            let mut audit = |row: &Value| {
                let field = |k: &str| row.get(k).cloned().unwrap_or(Value::Null);
                crate::controls::audit(home, "learning-cycle", vec![("project", s(key)), ("action", s("rollback")), ("cycle", field("rollback_of")), ("restored", field("restored")), ("by", s(by))]).ok()
            };
            let row = nmlearn::idle::rollback(memory, cycle.as_deref(), &mut learners, by, &mut audit)?;
            let aid = text(&row, "audit_id").unwrap_or_default();
            (row, aid)
        }
        CycleAsk::Sync => {
            if !nmapi::whole::learning_control(home) || crate::controls::memory_off(home) {
                return Err("learning or memory is off: nothing is brought into the engine".into());
            }
            let mut eng = nmlearn::engine::ContinuousLearningEngine::open(memory);
            let (before, rows_before) = (eng.episodic.len(), eng.synced.len());
            if nmlearn::idle::sync_engine(memory, &mut eng) {
                eng.save(memory).map_err(|e| e.to_string())?;
            }
            let added = (eng.episodic.len() - before) as u64;
            let rows = (eng.synced.len() - rows_before) as u64;
            let label = format!("{rows} stored rows taken into the learning engine: {added} new experiences, {} merged into identical ones", rows.saturating_sub(added));
            let audit = crate::controls::audit(home, "learning-cycle", vec![("project", s(key)), ("action", s("sync")), ("added", n(added)), ("by", s(by)), ("label", s(&label))])?;
            (obj(vec![("row", s("engine-sync")), ("added", n(added)), ("label", s(&label))]), audit)
        }
        CycleAsk::Cycle => {
            let rec = run_unified_cycle(home, memory, key, by, "manual")?;
            let aid = rec.audit_id.clone().unwrap_or_default();
            (rec.to_value(), aid)
        }
    };
    let eng = nmlearn::engine::ContinuousLearningEngine::open(memory);
    let mut out = eng.view_value();
    if let Value::Object(m) = &mut out {
        m.insert("backlog".into(), n(engine_backlog(memory, &eng)));
        m.insert("audit_id".into(), s(&audit_id));
        m.insert("cycle".into(), record);
        m.insert("project".into(), s(key));
        m.insert("learning".into(), s(if nmapi::whole::learning_control(home) { "on" } else { "off" }));
        m.insert("cycle_mode".into(), s(&nmapi::whole::learning_cycle_control(home)));
        m.insert("idle_cycle".into(), idle_standing(memory));
    }
    Ok(out)
}

/// Quiet time the serving process needs before an idle cycle starts (ms), by default; the variable
/// `NANO_LEARN_QUIET_MS` changes it.
pub const IDLE_QUIET_MS: u64 = 30_000;

fn idle_quiet_ms() -> u64 {
    std::env::var("NANO_LEARN_QUIET_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|x| *x > 0).unwrap_or(IDLE_QUIET_MS)
}

/// Whether an idle cycle should start now: learning and memory on, `learning.cycle` `auto` (2026-11-25:
/// `immediate` waits for a cycle asked for by hand, as its help says),
/// enough verified examples since the last cycle (`nmlearn::idle::due`), and the process quiet for
/// at least `quiet` ms (`quiet_for`, `None` while a call is in flight).
fn idle_ready(learning_on: bool, memory_on: bool, mode: &str, due: bool, quiet_for: Option<u64>, quiet: u64) -> bool {
    learning_on && memory_on && mode == "auto" && due && quiet_for.is_some_and(|q| q >= quiet)
}

/// One look of the idle loop at a project: runs a unified cycle when `idle_ready`. Returns the
/// cycle's record when one ran.
pub fn idle_learning_tick(home: &Path, memory: &Path, key: &str) -> Option<Result<nmlearn::idle::CycleRecord, String>> {
    let ready = idle_ready(
        nmapi::whole::learning_control(home),
        !crate::controls::memory_off(home),
        &nmapi::whole::learning_cycle_control(home),
        nmlearn::idle::due(memory, nmlearn::idle::threshold()),
        nmapi::g3::quiet_ms(),
        idle_quiet_ms(),
    );
    ready.then(|| run_unified_cycle(home, memory, key, "idle", "threshold"))
}

/// Whether the idle release should run now: switched on (`minutes`) and the process quiet that long
/// (`quiet`, `None` while a call is in flight).
fn idle_release_due(minutes: Option<u64>, quiet: Option<u64>) -> Option<u64> {
    let age = minutes?.saturating_mul(60_000);
    quiet.filter(|q| *q >= age).map(|_| age)
}

/// The idle KV window release (roadmap Phase 4.6; `residency.idle_release`, off by default): once
/// the serving process has been quiet for the set minutes, window copies unused that long and idle
/// scratch buffers go back to the driver; the seated models stay. Logged when it frees anything.
///
/// 2026-11-25 (IN_TROUBLE fix, measured by two soaks: questions p50 280 to 803 ms with 96 releases in
/// 16 idle phases): the tick released every 5 s once the quiet time was reached, trimming the card's
/// scratch buffers while the idle writer was using them. Now it releases once per quiet period (the
/// period is known by when the last call ended) and only while the writer has no session and no
/// notes left to write.
fn idle_release_tick(home: &Path) {
    static DONE_FOR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let quiet = nmapi::g3::quiet_ms();
    let Some(age) = idle_release_due(nmapi::whole::idle_release_minutes(home), quiet) else { return };
    // The quiet period by its start (the last call's end), to the second.
    let period = nmtype::now_ms().saturating_sub(quiet.unwrap_or(0)) / 1000;
    if DONE_FOR.load(std::sync::atomic::Ordering::SeqCst) == period || nmapi::writer_idle::busy_or_pending() {
        return;
    }
    DONE_FOR.store(period, std::sync::atomic::Ordering::SeqCst);
    let freed = nmengn::residency::get().release_idle_windows(age);
    if freed > 0 {
        eprintln!("residency: idle release gave back {} MB of card memory", freed >> 20);
    }
}

/// Starts the idle learning loop in the serving process (once): every few seconds it looks at the
/// served project's memory folder and runs a unified cycle when `idle_learning_tick` says so.
/// Returns `false` when it already runs.
pub fn start_idle_learning(home: PathBuf) -> bool {
    static STARTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if STARTED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return false;
    }
    std::thread::spawn(move || {
        loop {
            std::thread::park_timeout(std::time::Duration::from_secs(5));
            let Some(o) = nmapi::tools::installed() else { continue };
            idle_release_tick(&home);
            if !o.memory_on {
                continue;
            }
            let Some(memory) = o.memory.lock().ok().map(|m| m.dir.clone()) else { continue };
            // 2026-11-26: memory deletions asked for in the interface while this process holds
            // the project's memory journal are applied here.
            if memory.join(crate::forget::REQUESTS).is_file()
                && let Ok(mut m) = o.memory.lock()
                && let Some(m2) = m.mem2.as_mut()
            {
                let done = crate::forget::apply_requests(&memory, m2);
                if done > 0 {
                    eprintln!("memory: {done} memories deleted at the owner's request");
                }
            }
            // 2026-11-26 (Phase B): memory notes, pins, sweeps, objectives and a world-model refresh
            // asked for in the interface while this process holds the journal, applied with its
            // own state through the same functions as `nm_memory` and `nm_goal`.
            if memory.join(nmapi::memory_write::REQUESTS).is_file() {
                let (done, refused) = nmapi::memory_write::apply_queued(o);
                if done + refused > 0 {
                    eprintln!("memory: {done} interface requests applied, {refused} refused");
                }
            }
            let key = project_key(&home, &memory);
            match idle_learning_tick(&home, &memory, &key) {
                Some(Ok(rec)) => eprintln!("learning: idle cycle {}: ran {:?}", rec.id, rec.ran()),
                Some(Err(e)) => eprintln!("learning: idle cycle: {e}"),
                None => {}
            }
        }
    });
    true
}

fn print_example(e: &Value, passages: &BTreeMap<String, String>) {
    println!(
        "{} {} {} ({}) split {} engine {} path {}",
        text(e, "id").unwrap_or_default(),
        text(e, "verdict").unwrap_or_default(),
        text(e, "strength").unwrap_or_default(),
        text(e, "decided_by").unwrap_or_default(),
        text(e, "split").unwrap_or_default(),
        text(e, "engine").unwrap_or_else(|| "-".into()),
        text(e, "path").unwrap_or_default()
    );
    let ds: Vec<String> = e.get("domains").and_then(Value::array).unwrap_or_default().iter().filter_map(|d| d.string().ok().map(str::to_owned)).collect();
    println!("  question hash {}  domains {}", text(e, "ask_sha").unwrap_or_default(), ds.join(" "));
    if let Some(q) = text(e, "question") {
        println!("  question: {q}");
    }
    let stacks: Vec<String> = e.get("stacks").and_then(Value::array).unwrap_or_default().iter().filter_map(|d| d.string().ok().map(str::to_owned)).collect();
    let carts: Vec<String> = e.get("cartridges").and_then(Value::array).unwrap_or_default().iter().filter_map(|d| d.string().ok().map(str::to_owned)).collect();
    println!("  cartridges on: {}  stacks: {}", if carts.is_empty() { "none".into() } else { carts.join(" ") }, if stacks.is_empty() { "none".into() } else { stacks.join(" ") });
    for c in e.get("candidates").and_then(Value::array).unwrap_or_default() {
        let score = c.get("score").ok().and_then(|x| if let Value::Number(t) = x { Some(t.clone()) } else { None }).unwrap_or_else(|| "-".into());
        println!(
            "  {:<12} {:<9} {} {}{} score {score} hash {}",
            text(c, "label").unwrap_or_default(),
            text(c, "strength").unwrap_or_else(|| "-".into()),
            text(c, "place").unwrap_or_default(),
            if matches!(c.get("served"), Ok(Value::Bool(true))) { "[served] " } else { "" },
            if matches!(c.get("judged"), Ok(Value::Bool(true))) { "[judged]" } else { "" },
            text(c, "hash").unwrap_or_else(|| "-".into())
        );
        if let Some(t) = text(c, "hash").and_then(|h| passages.get(&h)) {
            let first = t.lines().next().unwrap_or_default();
            println!("               text: {first}{}", if t.lines().count() > 1 { " ..." } else { "" });
        }
    }
}

fn print_summary(v: &Value) {
    let field = |k: &str| -> String {
        match v.get(k) {
            Ok(Value::Object(m)) if m.is_empty() => "none".into(),
            Ok(Value::Object(m)) => m.iter().map(|(a, b)| format!("{a} {}", if let Value::Number(x) = b { x.clone() } else { b.encode().unwrap_or_default() })).collect::<Vec<_>>().join(", "),
            Ok(x) => x.string().map_or_else(|_| x.encode().unwrap_or_default(), str::to_owned),
            Err(_) => String::new(),
        }
    };
    println!("training store: {} (setting {}, memory {}, {} bytes)", field("folder"), field("setting"), field("memory"), field("bytes"));
    println!("  families: {}", field("families"));
    println!("  verdicts: {}", field("verdicts"));
    println!("  strengths: {}", field("strengths"));
    println!("  splits: {}", field("splits"));
    println!("  domains: {}", field("domains"));
    println!("  candidate labels: {}", field("labels"));
    println!("  engines: {}", field("engines"));
    println!("  preference signals: {}", field("preference_kinds"));
    println!("  judged sets waiting for their window: {}", field("judged_waiting"));
}

/// `neuromorph-nano examples --project <repo> [--json] [--show <id>] [--export <file>] [--import
/// <file>] [--clear] [--as-of <ms>] [--capture <file>]`, and `examples --references`,
/// `--reference-add <path> --licence <id> [--source <name>] [--domain <tag>]`, `--reference-fitted
/// <id> --cartridge <id>`.
#[allow(clippy::too_many_lines)]
pub fn run(home: &Path, args: &[String]) -> ExitCode {
    let arg = |k: &str| args.windows(2).find(|w| w.first().map(String::as_str) == Some(k)).and_then(|w| w.get(1)).cloned();
    let has = |k: &str| args.iter().any(|a| a == k);
    let fail = |e: String| {
        eprintln!("examples: {e}");
        ExitCode::FAILURE
    };
    if has("--references") {
        let refs = read_references(home);
        println!("reference material for fundamentals cartridges ({}): {}", refs.len(), references_file(home).display());
        for r in refs {
            let fitted: Vec<String> = r.get("fitted").and_then(Value::array).unwrap_or_default().iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect();
            println!(
                "  {} {} [{}] {} ({} files) licence {} sha256 {} fitted: {}",
                text(&r, "id").unwrap_or_default(),
                text(&r, "source").unwrap_or_default(),
                text(&r, "domain").unwrap_or_default(),
                text(&r, "path").unwrap_or_default(),
                r.get("files").and_then(Value::integer).unwrap_or(0),
                text(&r, "licence").unwrap_or_default(),
                text(&r, "sha256").unwrap_or_default().get(..16).unwrap_or(""),
                if fitted.is_empty() { "none".into() } else { fitted.join(" ") }
            );
        }
        return ExitCode::SUCCESS;
    }
    if let Some(p) = arg("--reference-add") {
        let Some(licence) = arg("--licence") else { return fail("--reference-add needs --licence <id>".into()) };
        let source = arg("--source").unwrap_or_else(|| Path::new(&p).file_name().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default());
        return match reference_add(home, Path::new(&p), &licence, &source, &arg("--domain").unwrap_or_else(|| "fundamentals".into())) {
            Ok(id) => {
                println!("listed {id}: {p} ({licence}); nothing was copied");
                ExitCode::SUCCESS
            }
            Err(e) => fail(e),
        };
    }
    if let Some(id) = arg("--reference-fitted") {
        let Some(c) = arg("--cartridge") else { return fail("--reference-fitted needs --cartridge <id>".into()) };
        return match reference_fitted(home, &id, &c) {
            Ok(()) => {
                println!("recorded: {c} fitted from {id}");
                ExitCode::SUCCESS
            }
            Err(e) => fail(e),
        };
    }
    let Some(repo) = arg("--project") else {
        eprintln!("usage: neuromorph-nano examples --project <repo> [--json] [--show <id>] [--export <file>] [--import <file>] [--clear]");
        eprintln!("       neuromorph-nano examples --references | --reference-add <path> --licence <id> [--source <name>] [--domain <tag>] | --reference-fitted <id> --cartridge <id>");
        return ExitCode::FAILURE;
    };
    let root = nmsubs::repo::canon(Path::new(&repo));
    let memory = nmtype::project::memory_dir(home, &root);
    let label = nmtype::project::label(home, &root);
    if has("--clear") {
        return match clear(home, &memory, &label, "command line") {
            Ok((rows, id)) => {
                println!("cleared the training store of {label}: {rows} rows removed from {}; audit {id}", ex::dir(&memory).display());
                ExitCode::SUCCESS
            }
            Err(e) => fail(e),
        };
    }
    if let Some(f) = arg("--import") {
        return match import(home, &memory, &label, Path::new(&f)) {
            Ok(k) => {
                println!("imported {k} rows into {}", ex::dir(&memory).display());
                ExitCode::SUCCESS
            }
            Err(e) => fail(e),
        };
    }
    if let Some(f) = arg("--capture") {
        // Records one judged set from a file, in the shape a serving path hands over (used by tests
        // and to replay a recorded call): {"pack", "question", "scored", "candidates": [{"place",
        // "score", "text"}], "engine": {...}, "views": [[...] or null]}.
        let lv = level_of(home);
        if crate::controls::memory_off(home) || lv == ex::OFF {
            println!("nothing captured: {}", if lv == ex::OFF { "the setting examples is off" } else { "memory is off" });
            return ExitCode::SUCCESS;
        }
        let r = (|| -> Result<usize, String> {
            let v = nmjson::parse(&std::fs::read(&f).map_err(|e| format!("{f}: {e}"))?).map_err(|e| format!("{f}: {e}"))?;
            let pack = text(&v, "pack").ok_or("pack: give the audit id of the call")?;
            let audit = std::fs::read_to_string(home.join("memory").join("audit.jsonl")).unwrap_or_default();
            let needle = format!("\"id\":\"{pack}\"");
            let row = audit.lines().rev().filter(|l| l.contains(&needle)).find_map(|l| nmjson::parse(l.as_bytes()).ok()).ok_or_else(|| format!("no audit row {pack}"))?;
            let question = text(&v, "question").unwrap_or_default();
            let mut cands = Vec::new();
            let mut scores = Vec::new();
            for c in v.get("candidates").and_then(Value::array).unwrap_or_default() {
                let (path, (a, b)) = ex::place(&text(c, "place").unwrap_or_default());
                #[expect(clippy::cast_possible_truncation, reason = "a score given as a number")]
                let sc = match c.get("score") {
                    Ok(Value::Number(x)) => x.parse::<f64>().unwrap_or(0.0) as f32,
                    _ => 0.0,
                };
                cands.push((path, usize::try_from(a).unwrap_or(0), usize::try_from(b).unwrap_or(0), text(c, "text").unwrap_or_default()));
                scores.push(sc);
            }
            let tap = (!cands.is_empty()).then(|| ex::Tapped {
                question: question.clone(),
                cands: cands
                    .iter()
                    .zip(&scores)
                    .map(|((p, a, b, t), sc)| ex::TapCand { path: p.clone(), start: *a, end: *b, score: *sc, text: (lv == ex::FULL && !t.is_empty()).then(|| t.clone()) })
                    .collect(),
                scored: !matches!(v.get("scored"), Ok(Value::Bool(false))),
                at_ms: nmtype::now_ms(),
            });
            let engine = match v.get("engine") {
                Ok(Value::Object(m)) => {
                    let fields: Vec<(&str, String)> = m.iter().map(|(k, x)| (k.as_str(), x.string().map_or_else(|_| x.encode().unwrap_or_default(), str::to_owned))).collect();
                    engine_of(&fields)
                }
                _ => None,
            };
            let views: Vec<Option<Vec<f32>>> = v
                .get("views")
                .and_then(Value::array)
                .unwrap_or_default()
                .iter()
                .map(|x| {
                    #[expect(clippy::cast_possible_truncation, reason = "values given as numbers")]
                    x.array().ok().map(|a| a.iter().map(|y| if let Value::Number(t) = y { t.parse::<f64>().unwrap_or(0.0) as f32 } else { 0.0 }).collect())
                })
                .collect();
            capture(&memory, &root, &row, tap.as_ref(), &question, engine, &views, lv)
        })();
        return match r {
            Ok(k) => {
                println!("captured {k} candidates");
                ExitCode::SUCCESS
            }
            Err(e) => fail(e),
        };
    }
    let now = match arg("--as-of").map(|t| t.parse::<u64>()) {
        Some(Ok(t)) => t,
        Some(Err(_)) => return fail("--as-of takes a time in milliseconds since 1970".into()),
        None => nmtype::now_ms(),
    };
    // The ledger settles first and settles the store after it.
    if !crate::controls::memory_off(home)
        && let Err(e) = crate::ledger::settle(home, &memory, &label, Some(&root), now, 3000, true)
    {
        return fail(e);
    }
    if let Some(f) = arg("--export") {
        return match export(home, &memory, &label, Path::new(&f)) {
            Ok(k) => {
                println!("exported {k} rows to {f}");
                ExitCode::SUCCESS
            }
            Err(e) => fail(e),
        };
    }
    let passages: BTreeMap<String, String> = ex::read_rows(&memory, ex::PASSAGES).unwrap_or_default().iter().filter_map(|p| Some((text(p, "hash")?, text(p, "text")?))).collect();
    if let Some(id) = arg("--show") {
        let rows = ex::read_rows(&memory, ex::EXAMPLES).unwrap_or_default();
        let Some(e) = rows.iter().find(|r| text(r, "id").as_deref() == Some(id.as_str()) || text(r, "pack").as_deref() == Some(id.as_str())) else {
            return fail(format!("no example {id}"));
        };
        if has("--json") {
            println!("{}", e.encode().unwrap_or_default());
        } else {
            print_example(e, &passages);
        }
        return ExitCode::SUCCESS;
    }
    let v = summary(home, &memory, &label);
    if has("--json") {
        println!("{}", v.encode().unwrap_or_default());
        return ExitCode::SUCCESS;
    }
    print_summary(&v);
    let rows = ex::read_rows(&memory, ex::EXAMPLES).unwrap_or_default();
    if let Some(last) = rows.last() {
        println!("newest example:");
        print_example(last, &passages);
    }
    ExitCode::SUCCESS
}

/// Section check (temporary folders): a judged set captured from a tap keeps exact score bits and no
/// text at `features`; at `full` passages are kept once per hash with secrets dropped; domain tags;
/// engine ids differ when a switch differs; retention counts run-days and keeps verified examples.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let base = std::env::temp_dir().join(format!("nano-examples-check-{}-{}", std::process::id(), nmtype::now_ms()));
    let r = check_in(&base);
    let _ = std::fs::remove_dir_all(&base);
    r
}

fn check_in(base: &Path) -> Result<(), String> {
    let e = |x: std::io::Error| x.to_string();
    let root = base.join("repo");
    let memory = base.join("memory");
    std::fs::create_dir_all(root.join("src")).map_err(e)?;
    std::fs::create_dir_all(&memory).map_err(e)?;
    std::fs::write(root.join("src/a.rs"), "fn a() {}\nlet password = \"hunter2hunter2\";\nfn b() {}\n").map_err(e)?;
    let audit = nmjson::parse(br#"{"id":"a1","time_ms":10,"ms":5,"session":"1-1","mode":"whole","sources":["src/a.rs:1-2"],"context_map":{"mode":"full"}}"#).map_err(|x| x.to_string())?;
    let tap = ex::Tapped {
        question: "where is a".into(),
        cands: vec![
            ex::TapCand { path: "src/a.rs".into(), start: 1, end: 2, score: 0.912_345_7, text: None },
            ex::TapCand { path: "src/a.rs".into(), start: 3, end: 3, score: 0.1, text: None },
        ],
        scored: true,
        at_ms: 1,
    };
    let eng = engine_of(&[("weights", "w1".into()), ("half_math", "true".into())]);
    let eng2 = engine_of(&[("weights", "w1".into()), ("half_math", "false".into())]);
    if eng.as_ref().map(|x| &x.0) == eng2.as_ref().map(|x| &x.0) {
        return Err("examples: engine ids equal across a switch".into());
    }
    capture(&memory, &root, &audit, Some(&tap), "where is a", eng.clone(), &[], ex::FEATURES)?;
    let j = ex::read_rows(&memory, ex::JUDGED)?;
    let first = j.first().and_then(|r| r.get("candidates").ok()).and_then(|c| c.array().ok()).and_then(|a| a.first()).cloned().ok_or("examples: no judged row")?;
    if text(&first, "bits").as_deref() != Some(format!("{:08x}", 0.912_345_7_f32.to_bits()).as_str()) || !matches!(first.get("served"), Ok(Value::Bool(true))) {
        return Err(format!("examples: judged candidate {first:?}"));
    }
    let all = std::fs::read_to_string(ex::dir(&memory).join(ex::JUDGED)).map_err(e)?;
    if all.contains("hunter2") || all.contains("where is a") || ex::dir(&memory).join(ex::PASSAGES).exists() {
        return Err("examples: text kept at features".into());
    }
    capture(&memory, &root, &audit, Some(&tap), "where is a", eng, &[], ex::FULL)?;
    capture(&memory, &root, &audit, Some(&tap), "where is a", None, &[], ex::FULL)?;
    let p = std::fs::read_to_string(ex::dir(&memory).join(ex::PASSAGES)).map_err(e)?;
    if p.lines().count() != 2 || p.contains("hunter2") || !p.contains("fn a()") {
        return Err(format!("examples: passages at full: {p}"));
    }
    if domains(&["src/model/attention.rs", "README.md"], "p") != ["code:rust", "documents", "ml-architecture", "project:p"] {
        return Err("examples: domain tags".into());
    }
    // Retention by run-days, nothing for size: the audit log of `base` ran on 30 days with idle days
    // between; an unverified example from the first of them is past 28 run-days and goes, a verified
    // one of the same day stays, as does an unverified one of the last run-day.
    let day = 86_400_000_u64;
    let now = nmtype::now_ms();
    let today = now / day * day;
    std::fs::create_dir_all(base.join("memory")).map_err(e)?;
    let audit_rows: Vec<String> = (0..30_u64).map(|i| format!(r#"{{"kind":"tool","time_ms":{}}}"#, today - (29 - i) * 2 * day)).collect();
    std::fs::write(base.join("memory").join("audit.jsonl"), audit_rows.join("
") + "
").map_err(e)?;
    let first = today - 58 * day + 5;
    let rows: Vec<Value> = [("old-passed", "passed-over", first), ("old-verified", "verified", first), ("new-passed", "passed-over", today + 5)]
        .iter()
        .map(|(id, v, at)| obj(vec![("id", s(id)), ("pack", s(id)), ("strength", s("indirect")), ("verdict", s(v)), ("at", n(*at))]))
        .collect();
    rewrite_lines(&ex::dir(&memory).join(ex::EXAMPLES), &rows)?;
    let removed = prune(base, &memory, now)?;
    let left: Vec<String> = ex::read_rows(&memory, ex::EXAMPLES)?.iter().filter_map(|r| text(r, "id")).collect();
    if removed < 1 || left != ["old-verified", "new-passed"] {
        return Err(format!("examples: retention by run-days kept {left:?} ({removed} rows removed)"));
    }
    Ok(())
}

#[cfg(test)]
mod learning_cycle_tests {
    use super::{idle_ready, idle_release_due, project_key};
    use std::path::Path;

    #[test]
    fn the_idle_cycle_waits_for_learning_memory_the_threshold_and_quiet() {
        assert!(idle_ready(true, true, "auto", true, Some(40_000), 30_000));
        assert!(!idle_ready(true, true, "immediate", true, Some(30_000), 30_000), "immediate waits for a cycle asked for by hand");
        assert!(!idle_ready(false, true, "auto", true, Some(40_000), 30_000), "learning off");
        assert!(!idle_ready(true, false, "auto", true, Some(40_000), 30_000), "memory off");
        assert!(!idle_ready(true, true, "manual", true, Some(40_000), 30_000), "manual cycles only");
        assert!(!idle_ready(true, true, "auto", false, Some(40_000), 30_000), "below the threshold");
        assert!(!idle_ready(true, true, "auto", true, Some(10), 30_000), "not quiet long enough");
        assert!(!idle_ready(true, true, "auto", true, None, 30_000), "a call in flight");
        // The idle release: off by default, waits for the quiet it is set to, never mid-call.
        assert_eq!(idle_release_due(None, Some(10_000_000)), None, "off");
        assert_eq!(idle_release_due(Some(5), Some(299_999)), None, "not quiet long enough");
        assert_eq!(idle_release_due(Some(5), Some(300_000)), Some(300_000));
        assert_eq!(idle_release_due(Some(15), None), None, "a call in flight");
    }

    #[test]
    fn the_cycle_ask_reads_the_request() {
        use super::CycleAsk;
        assert_eq!(CycleAsk::of("cycle", Some(3), Some("lc-1")), CycleAsk::Cycle);
        assert_eq!(CycleAsk::of("rollback", None, None), CycleAsk::Rollback { version: None, cycle: None });
        assert_eq!(CycleAsk::of("rollback", Some(4), Some("")), CycleAsk::Rollback { version: Some(4), cycle: None });
        assert_eq!(CycleAsk::of("rollback", None, Some("lc-9-abc")), CycleAsk::Rollback { version: None, cycle: Some("lc-9-abc".into()) });
    }

    #[test]
    fn project_keys_follow_the_projects_listing() {
        let home = Path::new("H");
        assert_eq!(project_key(home, &home.join("memory")), "nano");
        assert_eq!(project_key(home, Path::new("H/projects/my-repo")), "my-repo");
    }
}
