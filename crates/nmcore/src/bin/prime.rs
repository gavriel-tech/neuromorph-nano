//! Priming: primes are short pieces of context the Nano offers Claude Code without being asked, through the
//! hooks that accept added context, at a moment the session shows they are likely to help. Silence
//! is the default; a prime is given only when one of the rules below fires, its source is read
//! fresh and still holds, and the budget allows it.
//!
//! Kinds, each a rule with a trigger and a source:
//! - `prompt-recall` (UserPromptSubmit): the prompt matches the question of an answered outcome in
//!   memory that was verified; source: that memory, its cited lines read fresh.
//! - `failure-fix` (PostToolUseFailure, or PostToolUse of a build or test whose output shows a
//!   failure): the failure has the signature of one fixed before (same check, same error code and
//!   file); source: that memory, the fix's lines read fresh.
//! - `edit-impact` (PreToolUse of Edit or MultiEdit): the lines about to change hold a verified fix
//!   or rest under a refuted outcome in memory; or the edit changes a function's signature line and
//!   the function is called elsewhere in the project; source: that memory, or the calling lines
//!   read now.
//! - `resume` (SessionStart after `compact`; also after `startup` with `priming` on): what was in
//!   play, from this session's record (or, at a start, what the project's previous session left
//!   open); source: the situation record, each file checked present.
//! - `core` (SessionStart, when `resume` has nothing to say): the memory system hands the agent its
//!   most used memories at session open; source: the `1-core` memories of the project's memory,
//!   each place read fresh (`nmmem::session_open`); a memory whose lines changed is held back.
//! - `open-loop` (Stop, only with `priming` on): files changed this turn and not built or tested
//!   since, or tested and failing; source: the situation record of the turn.
//!
//! Budget: a prime is given whole (never cut) and states only its located facts; a session has no
//! count or character quota: the same prime (same kind and source) is never given twice in a
//! session, and each prime is given only when its rule fires. Time: a prime may take as long again
//! as the hook's own work takes on this machine (`deadline`); one not ready by then is dropped. Every prime given is kept on the hook's
//! own situation record (kind, id, places, sources) and judged by the evidence ledger like a pack.
//! Nothing here blocks, denies or asks: replies carry only `additionalContext`.
use crate::sessions::{Rec, records};
use nmjson::Value;
use nmmem::outcomes::{Known, Place, fresh, known, rank, recall};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// The stated fallback deadline, from the hook's start, while this project has too few measured
/// hooks to derive one (fewer than `nmmem::evidence::NULL_MIN`).
pub const FALLBACK_DEADLINE: Duration = Duration::from_millis(25);

/// The hook's own work on this machine: the median time, from process creation (`whole_ms`) when
/// available else from the hook's start (`hook_ms`), of this project's hooks on which no prime
/// rule runs (a tool event other than a build or test). What the host already waits for on every
/// event; `None` with too few measured. Floored at 1 ms so sub-millisecond integer truncation
/// never collapses the prime deadline to 0 ms.
#[must_use]
pub fn own_work(memory: &Path) -> Option<Duration> {
    let mut ms: Vec<u64> = crate::situation::read(memory, None, None)
        .iter()
        .filter(|v| v.get("event").and_then(Value::string).ok() == Some("PostToolUse"))
        .filter(|v| !matches!(v.get("kind").and_then(Value::string).ok(), Some("build" | "test")))
        .filter(|v| v.get("prime").is_err() && v.get("prime_late").is_err())
        .filter_map(|v| v.get("whole_ms").or_else(|_| v.get("hook_ms")).and_then(Value::integer).ok())
        .collect();
    if ms.len() < nmmem::evidence::NULL_MIN {
        return None;
    }
    ms.sort_unstable();
    ms.get(ms.len() / 2).map(|m| Duration::from_millis((*m).max(1)))
}

/// The deadline in force, from the hook's start: the milliseconds in `NANO_PRIME_DEADLINE_MS` when
/// set (scripted tests on a loaded machine, and the test of the drop itself); else the hook's own
/// work twice over (the prime may take as long again as the hook itself, `own_work`); else the
/// stated fallback.
fn deadline(memory: &Path) -> Duration {
    static D: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *D.get_or_init(|| {
        let set = std::env::var("NANO_PRIME_DEADLINE_MS").or_else(|_| std::env::var("NANO_NUDGE_DEADLINE_MS")).ok().and_then(|v| v.trim().parse::<u64>().ok());
        set.map(Duration::from_millis).or_else(|| own_work(memory).map(|d| d * 2)).unwrap_or(FALLBACK_DEADLINE)
    })
}

/// The `priming` setting: off, quiet, on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Off,
    Quiet,
    On,
}

impl Mode {
    #[must_use]
    pub fn parse(v: Option<&String>) -> Self {
        match v.map(String::as_str) {
            Some("off") => Self::Off,
            Some("on") => Self::On,
            _ => Self::Quiet,
        }
    }
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Quiet => "quiet",
            Self::On => "on",
        }
    }
    /// Least strength of the verification a memory needs before it is offered.
    const fn min_rank(self) -> u8 {
        match self {
            Self::On => 1,
            _ => 2,
        }
    }
}


/// A prime ready to give.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Given {
    pub kind: &'static str,
    pub id: String,
    pub text: String,
    /// The places it cites (`path:a-b`, or a file).
    pub places: Vec<String>,
    /// The memories it came from (record ids), or `situation` for the session's own record.
    pub from: Vec<String>,
}

impl Given {
    /// The `prime` field of the hook's situation record: what was given and from where, without
    /// the text.
    #[must_use]
    pub fn record_value(&self, mode: Mode) -> Value {
        let s = |t: &str| Value::String(t.to_owned());
        Value::Object(
            [
                ("kind", s(self.kind)),
                ("id", s(&self.id)),
                ("places", Value::Array(self.places.iter().map(|p| s(p)).collect())),
                ("from", Value::Array(self.from.iter().map(|p| s(p)).collect())),
                ("chars", Value::Number(self.text.chars().count().to_string())),
                ("mode", s(mode.label())),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect(),
        )
    }
    /// The hook's reply: one JSON object with `hookSpecificOutput.additionalContext` only.
    #[must_use]
    pub fn reply(&self, event: &str) -> String {
        let s = |t: &str| Value::String(t.to_owned());
        let inner = Value::Object([("hookEventName".to_owned(), s(event)), ("additionalContext".to_owned(), s(&self.text))].into_iter().collect());
        Value::Object([("hookSpecificOutput".to_owned(), inner)].into_iter().collect()).encode().unwrap_or_default()
    }
}

/// Why no prime was given when one was being made.
#[derive(Debug, PartialEq, Eq)]
pub enum Silent {
    /// Nothing fired, or the budget or a stale source stopped it.
    Nothing,
    /// A rule's trigger was being looked at when the deadline passed; its answer is dropped (the kind is named).
    Late(&'static str),
}

/// What the hook knows when it asks for a prime.
pub struct Ask<'a> {
    pub home: &'a Path,
    pub memory: &'a Path,
    pub root: &'a Path,
    pub event: &'a str,
    pub input: &'a Value,
    /// The hook's own record of this event (not yet appended).
    pub rec: &'a BTreeMap<String, Value>,
    pub mode: Mode,
    /// When the hook started.
    pub start: Instant,
}

impl Ask<'_> {
    fn late(&self) -> bool {
        self.start.elapsed() > deadline(self.memory)
    }
    fn session(&self) -> Option<&str> {
        self.rec.get("session").and_then(|v| v.string().ok()).filter(|x| !x.is_empty())
    }
    fn field(&self, k: &str) -> Option<&str> {
        self.rec.get(k).and_then(|v| v.string().ok())
    }
}

fn sha16(t: &str) -> String {
    nmhash::sha256_bytes(t.as_bytes()).chars().take(16).collect()
}

/// The file that keeps one session's primes (id, kind, size), for the never-twice rule. Holds no
/// text.
#[must_use]
pub fn memo_file(memory: &Path, session: &str) -> PathBuf {
    let name: String = session.chars().filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_')).take(80).collect();
    let name = if name.is_empty() || name.len() != session.len() { sha16(session) } else { name };
    memory.join("primes").join(format!("{name}.jsonl"))
}

/// The ids of the primes already given in a session.
fn memo(memory: &Path, session: &str) -> BTreeSet<String> {
    let text = std::fs::read_to_string(memo_file(memory, session)).unwrap_or_default();
    text.lines()
        .filter_map(|l| nmjson::parse(l.as_bytes()).ok())
        .filter_map(|v| v.get("id").and_then(Value::string).ok().map(str::to_owned))
        .collect()
}

/// Keeps a given prime in the session's memo (one appended line).
/// # Errors
/// The file cannot be written.
pub fn remember(memory: &Path, session: &str, g: &Given, event: &str) -> Result<(), String> {
    let file = memo_file(memory, session);
    if let Some(d) = file.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let s = |t: &str| Value::String(t.to_owned());
    let v = Value::Object(
        [
            ("t", Value::Number(nmtype::now_ms().to_string())),
            ("id", s(&g.id)),
            ("kind", s(g.kind)),
            ("event", s(event)),
            ("chars", Value::Number(g.text.chars().count().to_string())),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect(),
    );
    let mut line = v.encode().map_err(|e| e.to_string())?;
    line.push('\n');
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&file).map_err(|e| format!("{}: {e}", file.display()))?;
    std::io::Write::write_all(&mut f, line.as_bytes()).map_err(|e| format!("{}: {e}", file.display()))
}

/// The prime for this hook, if any rule fires and it was not given before in the session.
/// # Errors
/// `Silent::Late` when the deadline passed while a rule's trigger was being looked at; `Silent::Nothing`
/// otherwise.
pub fn offer(a: &Ask) -> Result<Given, Silent> {
    if a.mode == Mode::Off {
        return Err(Silent::Nothing);
    }
    let session = a.session().ok_or(Silent::Nothing)?;
    let ids = memo(a.memory, session);
    let kind = a.field("kind").unwrap_or_default();
    let (rule, made): (&'static str, Option<Given>) = match a.event {
        "UserPromptSubmit" => ("prompt-recall", prompt_recall(a)),
        "PostToolUse" | "PostToolUseFailure" if matches!(kind, "build" | "test") && matches!(a.rec.get("ok"), Some(Value::Bool(false))) => {
            ("failure-fix", failure_fix(a))
        }
        "PreToolUse" if matches!(a.field("tool"), Some("Edit" | "MultiEdit")) => ("edit-impact", edit_impact(a)),
        "SessionStart" => match resume(a, session) {
            Some(g) => ("resume", Some(g)),
            None => ("core", core(a)),
        },
        "Stop" if a.mode == Mode::On && !matches!(a.input.get("stop_hook_active"), Ok(Value::Bool(true))) => ("open-loop", open_loop(a, session)),
        _ => return Err(Silent::Nothing),
    };
    // Past the deadline the rule's answer is dropped, whatever it was, and the record says so.
    if a.late() {
        return Err(Silent::Late(rule));
    }
    let g = made.ok_or(Silent::Nothing)?;
    if ids.contains(&g.id) {
        return Err(Silent::Nothing);
    }
    Ok(g)
}

fn id_of(kind: &str, parts: &[String]) -> String {
    format!("n{}", sha16(&format!("{kind}\n{}", parts.join("\n"))))
}

fn labels(places: &[Place]) -> Vec<String> {
    places.iter().map(Place::label).collect()
}

/// Every place of a memory reads as it did when verified.
fn all_fresh(root: &Path, k: &Known) -> bool {
    !k.places.is_empty() && k.places.iter().all(|p| fresh(root, p))
}

fn times(n: u32) -> String {
    if n == 1 { "once".into() } else { format!("{n} times") }
}

fn prompt_recall(a: &Ask) -> Option<Given> {
    let prompt = a.input.get("prompt").and_then(Value::string).ok()?.trim();
    if prompt.starts_with('/') {
        return None;
    }
    let all = known(a.memory);
    let usable: Vec<Known> = all.into_iter().filter(|k| k.kind == "answered" && k.standing && rank(&k.strength) >= a.mode.min_rank()).collect();
    let hits = recall(&usable, prompt);
    let (_, k) = hits.first()?;
    if a.late() || !all_fresh(a.root, k) {
        return None;
    }
    let places = labels(&k.places);
    let text = format!(
        "Neuromorph Nano memory: an earlier question like this one, \"{}\", was answered at {} (verified {}{}, strongest evidence {}). Those lines are unchanged since.",
        k.question.as_deref().unwrap_or_default(),
        places.join(", "),
        times(k.verified),
        if k.refuted > 0 { format!(", refuted {}", times(k.refuted)) } else { String::new() },
        k.strength
    );
    let mut parts = vec![k.record.clone()];
    parts.extend(k.places.iter().map(|p| format!("{} {}", p.label(), p.hash)));
    Some(Given { kind: "prompt-recall", id: id_of("prompt-recall", &parts), text, places, from: vec![k.record.clone()] })
}

/// The first located error of the hook's record: code, file, line, message.
fn first_error(rec: &BTreeMap<String, Value>) -> Option<(String, String, u64, String)> {
    let e = rec.get("errors")?.array().ok()?.first()?.clone();
    let t = |k: &str| e.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default();
    Some((t("code"), t("file"), e.get("line").and_then(Value::integer).unwrap_or(0), t("msg")))
}

fn failure_fix(a: &Ask) -> Option<Given> {
    let (code, file, line, msg) = first_error(a.rec)?;
    let check = a.field("cmd_class").unwrap_or("check");
    let pkg = a.field("pkg").unwrap_or_default();
    let all = known(a.memory);
    let mut cands: Vec<&Known> = all
        .iter()
        .filter(|k| k.kind == "fixed" && k.standing && rank(&k.strength) >= a.mode.min_rank())
        .filter(|k| {
            k.failure.as_ref().is_some_and(|f| {
                f.check == check
                    && f.pkg == pkg
                    && f.file == file
                    && if code.is_empty() { f.line.abs_diff(line) <= crate::outcomes::NEAR_LINES } else { f.code == code }
            })
        })
        .collect();
    cands.sort_by(|x, y| y.verified.cmp(&x.verified).then(rank(&y.strength).cmp(&rank(&x.strength))).then(x.record.cmp(&y.record)));
    let k = cands.into_iter().find(|k| all_fresh(a.root, k))?;
    let f = k.failure.as_ref()?;
    let places = labels(&k.places);
    let label = if pkg.is_empty() { check.to_owned() } else { format!("{check} {pkg}") };
    let text = format!(
        "Neuromorph Nano memory: {label} failed the same way before ({}at {}:{}{}). The change that fixed it then is at {}; {label} passed after it (verified {}, evidence {}). Those lines are unchanged since.",
        if code.is_empty() { String::new() } else { format!("{code} ") },
        file,
        line,
        if msg.is_empty() { if f.msg.is_empty() { String::new() } else { format!(", {}", f.msg) } } else { format!(", {msg}") },
        places.join(", "),
        times(k.verified),
        k.strength
    );
    let mut parts = vec![k.record.clone(), format!("{label} {code} {file}")];
    parts.extend(k.places.iter().map(|p| format!("{} {}", p.label(), p.hash)));
    Some(Given { kind: "failure-fix", id: id_of("failure-fix", &parts), text, places, from: vec![k.record.clone()] })
}

/// The path and lines of the edit's before-lines (the hook's own record of a PreToolUse).
fn edit_lines(rec: &BTreeMap<String, Value>) -> Option<(String, u64, u64)> {
    let l = rec.get("lines")?.array().ok()?.first()?.array().ok()?.to_vec();
    Some((l.first()?.string().ok()?.to_owned(), l.get(1)?.integer().ok()?, l.get(2)?.integer().ok()?))
}

fn edit_impact(a: &Ask) -> Option<Given> {
    let (path, start, end) = edit_lines(a.rec)?;
    if path == "<outside>" {
        return None;
    }
    let overlaps = |p: &Place| p.path == path && (p.start as u64) <= end && start <= p.end as u64;
    let all = known(a.memory);
    // A verified fix in these lines.
    if let Some(k) = all.iter().find(|k| k.kind == "fixed" && k.standing && rank(&k.strength) >= a.mode.min_rank() && k.places.iter().any(overlaps) && all_fresh(a.root, k))
        && let Some(f) = &k.failure
    {
        let at: Vec<&Place> = k.places.iter().filter(|p| overlaps(p)).collect();
        let places: Vec<String> = at.iter().map(|p| p.label()).collect();
        let text = format!(
            "Neuromorph Nano memory: {} holds the change that fixed {} {}at {}:{} (verified {}).",
            places.join(", "),
            f.check_label(),
            if f.code.is_empty() { String::new() } else { format!("{} ", f.code) },
            f.file,
            f.line,
            times(k.verified)
        );
        let parts = vec![k.record.clone(), places.join(" ")];
        return Some(Given { kind: "edit-impact", id: id_of("edit-impact", &parts), text, places, from: vec![k.record.clone()] });
    }
    // A refuted outcome resting on these lines.
    if let Some(k) = all.iter().find(|k| !k.standing && k.places.iter().any(overlaps) && all_fresh(a.root, k)) {
        let places: Vec<String> = k.places.iter().filter(|p| overlaps(p)).map(Place::label).collect();
        let what = match (&k.question, &k.failure) {
            (Some(q), _) => format!("the answer to \"{q}\""),
            (None, Some(f)) => format!("a fix of {} {}", f.check_label(), f.code),
            _ => k.kind.clone(),
        };
        let text = format!("Neuromorph Nano memory: an earlier change at {} ({what}) was refuted afterwards.", places.join(", "));
        let parts = vec![k.record.clone(), places.join(" ")];
        return Some(Given { kind: "edit-impact", id: id_of("edit-impact", &parts), text, places, from: vec![k.record.clone()] });
    }
    if a.late() {
        return None;
    }
    callers(a, &path)
}

/// The name of a function defined on a line (`fn`, `def`, `function`, `func`), if any.
fn defined(line: &str) -> Option<String> {
    let t = line.trim_start();
    for kw in ["fn ", "def ", "function ", "func "] {
        if let Some(i) = t.find(kw)
            && (i == 0 || t.get(..i).is_some_and(|b| b.ends_with(' ') || b.ends_with('(')))
        {
            let name: String = t.get(i + kw.len()..)?.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
            if name.chars().count() >= 3 {
                return Some(name);
            }
        }
    }
    None
}

fn callers(a: &Ask, path: &str) -> Option<Given> {
    let ti = a.input.get("tool_input").ok()?;
    let old = ti.get("old_string").and_then(Value::string).ok()?;
    let new = ti.get("new_string").and_then(Value::string).unwrap_or_default();
    // A signature line of the old text that the new text does not keep word for word.
    let (sig, name) = old.lines().find_map(|l| defined(l).map(|n| (l.trim().to_owned(), n)))?;
    if new.lines().any(|l| l.trim() == sig) {
        return None;
    }
    let ext = Path::new(path).extension()?.to_string_lossy().into_owned();
    let call = format!("{name}(");
    let mut found: Vec<String> = Vec::new();
    // The search covers the edited file's own package (the nearest folder above it with a
    // manifest), so it can finish within the deadline on a large tree; the reply names it. The
    // deadline bounds it; a file the material's reader would not read (`nmdoc::format`, which
    // skips oversized files and what holds nothing to read) is not read here either.
    let scope = package_of(a.root, path);
    let mut stack = vec![scope.iter().fold(a.root.to_path_buf(), |p, c| p.join(c))];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        let mut entries: Vec<std::fs::DirEntry> = rd.flatten().collect();
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for e in entries {
            if a.late() {
                return None;
            }
            let p = e.path();
            let fname = e.file_name().to_string_lossy().into_owned();
            if p.is_dir() {
                // The folders the material's reader leaves out (`nmsubs::repo`): hidden ones, its
                // skip list, and the Neuromorph folder's own state when the project is one.
                let own = nmsubs::repo::is_own(a.root) && nmsubs::repo::SKIP_OWN.contains(&fname.as_str());
                if !fname.starts_with('.') && !nmsubs::repo::SKIP.contains(&fname.as_str()) && !own {
                    stack.push(p);
                }
                continue;
            }
            let size = e.metadata().map_or(u64::MAX, |m| m.len());
            if p.extension().is_none_or(|x| x.to_string_lossy() != ext) || matches!(nmdoc::format::by_name(&fname, size), nmdoc::Class::Skip(_)) {
                continue;
            }
            let rel = p.strip_prefix(a.root).map(|r| r.to_string_lossy().replace('\\', "/")).unwrap_or_default();
            if rel.is_empty() || rel == path {
                continue;
            }
            let Ok(body) = std::fs::read_to_string(&p) else { continue };
            if !body.contains(&call) {
                continue;
            }
            for (i, l) in body.lines().enumerate() {
                let at = l.find(&call);
                let word_start = at.is_some_and(|x| x == 0 || l.get(..x).and_then(|b| b.chars().last()).is_none_or(|c| !(c.is_alphanumeric() || c == '_')));
                if word_start && defined(l).is_none() && found.len() < 3 {
                    found.push(format!("{rel}:{}", i + 1));
                }
            }
        }
    }
    if found.is_empty() {
        return None;
    }
    let searched = if scope.is_empty() { "the project".to_owned() } else { scope.join("/") };
    let text = format!("Neuromorph Nano: this edit changes the signature line of {name} in {path}; {name} is called at {} (searched {searched}, read now).", found.join(", "));
    let parts = vec![format!("{path} {sig}"), found.join(" ")];
    Some(Given { kind: "edit-impact", id: id_of("edit-impact", &parts), text, places: found, from: vec!["source".into()] })
}

/// The folders, from the project root, of the nearest package above a project-relative file: the
/// first folder holding `Cargo.toml`, `package.json`, `go.mod`, `pyproject.toml` or `setup.py`.
/// Empty (the whole project) when none is found below the root.
fn package_of(root: &Path, path: &str) -> Vec<String> {
    let mut parts: Vec<String> = path.split('/').map(str::to_owned).collect();
    parts.pop();
    while !parts.is_empty() {
        let dir = parts.iter().fold(root.to_path_buf(), |p, c| p.join(c));
        if ["Cargo.toml", "package.json", "go.mod", "pyproject.toml", "setup.py"].iter().any(|m| dir.join(m).is_file()) {
            return parts;
        }
        parts.pop();
    }
    Vec::new()
}

/// Files with code, for the loop rules (documents and data are not built or tested).
fn is_code(path: &str) -> bool {
    let ext = Path::new(path).extension().map(|x| x.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    matches!(
        ext.as_str(),
        "rs" | "py" | "js" | "jsx" | "ts" | "tsx" | "go" | "java" | "kt" | "c" | "h" | "cc" | "cpp" | "hpp" | "cs" | "rb" | "swift" | "scala" | "php" | "m" | "mm"
    )
}

fn exists(root: &Path, rel: &str) -> bool {
    rel.split('/').fold(root.to_path_buf(), |a, c| a.join(c)).is_file()
}

fn spans(v: &[(u64, u64)]) -> String {
    v.iter().map(|(a, b)| if a == b { a.to_string() } else { format!("{a}-{b}") }).collect::<Vec<_>>().join(",")
}

fn check_label(r: &Rec) -> String {
    format!("{}{}", r.cmd_class.as_deref().unwrap_or("check"), r.pkg.as_deref().map(|p| format!(" {p}")).unwrap_or_default())
}

fn first_error_text(r: &Rec) -> String {
    r.errors.first().map(|(c, f, l)| format!(" ({}at {f}:{l})", if c.is_empty() { String::new() } else { format!("{c} ") })).unwrap_or_default()
}

fn resume(a: &Ask, session: &str) -> Option<Given> {
    let source = a.field("source").unwrap_or_default();
    match source {
        "compact" => resume_compact(a, session),
        "startup" if a.mode == Mode::On => resume_startup(a, session),
        _ => None,
    }
}

/// The core memories of the project at session open, each place read fresh.
fn core(a: &Ask) -> Option<Given> {
    let all = nmmem::session_open::for_session(a.memory, a.root);
    if all.is_empty() || a.late() {
        return None;
    }
    // Whole memories, most used first: every core memory whose places read back unchanged (core
    // holds at most 32), none cut short.
    let head = "Neuromorph Nano memory, the most used in this project (places read now, unchanged): ";
    let mut parts: Vec<String> = Vec::new();
    let mut given: Vec<&nmmem::session_open::Core> = Vec::new();
    for c in &all {
        parts.push(format!("{} ({})", c.text.replace('\n', " "), c.places.join(", ")));
        given.push(c);
    }
    if given.is_empty() {
        return None;
    }
    let text = format!("{head}{}.", parts.join("; "));
    let places: Vec<String> = given.iter().flat_map(|c| c.places.iter().cloned()).collect();
    let from: Vec<String> = given.iter().map(|c| c.id.clone()).collect();
    let mut ids = from.clone();
    ids.extend(places.iter().cloned());
    Some(Given { kind: "core", id: id_of("core", &ids), text, places, from })
}

fn resume_compact(a: &Ask, session: &str) -> Option<Given> {
    let recs = records(a.memory);
    if a.late() {
        return None;
    }
    let prompts = crate::sessions::kept_prompts(a.home, a.memory);
    let st = crate::sessions::state(&recs, session, &prompts);
    let edited: Vec<(String, Vec<(u64, u64)>)> =
        st.files.iter().filter(|(p, f)| f.edits + f.writes > 0 && exists(a.root, p)).map(|(p, f)| (p.clone(), f.edit_lines.clone())).collect();
    let packs: Vec<String> = st.served.iter().filter_map(|r| r.nano.as_ref().and_then(|n| n.2.clone())).collect();
    if edited.is_empty() && st.failing.is_empty() && packs.is_empty() {
        return None;
    }
    let mut parts: Vec<String> = Vec::new();
    if let Some((_, _, _, Some(t))) = st.prompts.last() {
        parts.push(format!("task: \"{}\"", t.replace('\n', " ")));
    }
    if !edited.is_empty() {
        parts.push(format!(
            "edited: {}",
            edited.iter().map(|(p, l)| if l.is_empty() { p.clone() } else { format!("{p}:{}", spans(l)) }).collect::<Vec<_>>().join(", ")
        ));
    }
    for c in &st.failing {
        parts.push(format!("failing: {}{}", check_label(c), first_error_text(c)));
    }
    let unchecked: Vec<&String> = st.unchecked.iter().filter(|p| is_code(p)).collect();
    if !unchecked.is_empty() {
        parts.push(format!("no passing build or test since the last change: {}", unchecked.iter().map(|x| x.as_str()).collect::<Vec<_>>().join(", ")));
    }
    if !packs.is_empty() {
        parts.push(format!("Nano context packs earlier in this session: {}", packs.join(", ")));
    }
    let text = format!("Neuromorph Nano, from this session's record before the conversation was compacted: {}.", parts.join("; "));
    let places: Vec<String> = edited.iter().map(|(p, _)| p.clone()).chain(st.failing.iter().flat_map(|c| c.errors.first().map(|e| e.1.clone()))).collect();
    let id = id_of("resume", &[session.to_owned(), text.clone()]);
    Some(Given { kind: "resume", id, text, places, from: vec!["situation".into()] })
}

fn resume_startup(a: &Ask, session: &str) -> Option<Given> {
    let recs = records(a.memory);
    let all = crate::sessions::sessions(&recs);
    let now = nmtype::now_ms();
    // The newest other session in which something happened (a start with nothing after it is passed by).
    let worked = |id: &str| recs.iter().any(|r| r.session == id && r.kind != "session");
    let (prev, _, last, _) = all.iter().rev().find(|x| x.0 != session && worked(&x.0))?.clone();
    // Only when that session ran on the last run-day before today, or today: no day on which the
    // system ran (a day with any record) lies between.
    let day = |t: u64| t / 86_400_000;
    let between = recs.iter().map(|r| day(r.t)).filter(|d| *d > day(last) && *d < day(now)).collect::<BTreeSet<u64>>().len();
    if between > 0 || a.late() {
        return None;
    }
    let st = crate::sessions::state(&recs, &prev, &BTreeMap::new());
    let unchecked: Vec<&String> = st.unchecked.iter().filter(|p| is_code(p) && exists(a.root, p)).collect();
    if st.failing.is_empty() && unchecked.is_empty() {
        return None;
    }
    let mut parts: Vec<String> = st.failing.iter().map(|c| format!("failing {}{}", check_label(c), first_error_text(c))).collect();
    if !unchecked.is_empty() {
        parts.push(format!("changed with no passing build or test since: {}", unchecked.iter().map(|x| x.as_str()).collect::<Vec<_>>().join(", ")));
    }
    let text = format!("Neuromorph Nano: the previous Claude Code session in this project left open: {}.", parts.join("; "));
    let places: Vec<String> = unchecked.iter().map(|x| (*x).clone()).chain(st.failing.iter().flat_map(|c| c.errors.first().map(|e| e.1.clone()))).collect();
    if places.is_empty() {
        return None;
    }
    let id = id_of("resume", &[prev, text.clone()]);
    Some(Given { kind: "resume", id, text, places, from: vec!["situation".into()] })
}

fn open_loop(a: &Ask, session: &str) -> Option<Given> {
    let recs = records(a.memory);
    let mut mine: Vec<&Rec> = recs.iter().filter(|r| r.session == session).collect();
    mine.sort_by_key(|r| (r.t, r.i));
    let from = mine.iter().rposition(|r| r.kind == "prompt").unwrap_or(0);
    let turn: Vec<&Rec> = mine.into_iter().skip(from).filter(|r| r.done()).collect();
    let mut edited: BTreeMap<String, Vec<(u64, u64)>> = BTreeMap::new();
    let mut last_edit = None;
    for r in turn.iter().filter(|r| matches!(r.kind.as_str(), "edit" | "write") && r.ok == Some(true)) {
        for p in r.paths.iter().filter(|p| is_code(p)) {
            edited.entry(p.clone()).or_default().extend(r.span_of(p));
            last_edit = Some((r.t, r.i));
        }
    }
    let last_edit = last_edit?;
    let last_check = turn.iter().rev().find(|r| matches!(r.kind.as_str(), "build" | "test"));
    let after = last_check.filter(|c| (c.t, c.i) > last_edit);
    if after.is_some_and(|c| c.ok == Some(true)) {
        return None;
    }
    let files: Vec<String> = edited
        .iter()
        .filter(|(p, _)| exists(a.root, p))
        .take(5)
        .map(|(p, l)| {
            let m = crate::sessions::merge(l.clone());
            if m.is_empty() { p.clone() } else { format!("{p} ({} {})", if m.len() == 1 && m[0].0 == m[0].1 { "line" } else { "lines" }, spans(&m)) }
        })
        .collect();
    if files.is_empty() {
        return None;
    }
    let tail = match after {
        Some(c) => format!(" The last check after them, {}, failed{}.", check_label(c), first_error_text(c)),
        None => " No build or test ran after them.".to_owned(),
    };
    let text = format!("Neuromorph Nano: changed this turn and not verified since: {}.{tail}", files.join(", "));
    let places: Vec<String> = edited.keys().filter(|p| exists(a.root, p)).take(5).cloned().collect();
    let id = id_of("open-loop", &[session.to_owned(), last_edit.0.to_string(), places.join(" ")]);
    Some(Given { kind: "open-loop", id, text, places, from: vec!["situation".into()] })
}

/// Section check (processor only, a temporary folder): a fixed outcome in a ledger recalls on a
/// matching failure and not on another; a stale fix line gives nothing; the same prime is not
/// given twice; `off` gives nothing; a signature change finds its caller; the budget stops a
/// session at its cap.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let dir = std::env::temp_dir().join(format!("nano-prime-check-{}-{}", std::process::id(), nmtype::now_ms()));
    let r = check_in(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    r
}

fn check_in(dir: &Path) -> Result<(), String> {
    let e = |x: std::io::Error| x.to_string();
    let root = dir.join("repo");
    let memory = dir.join("memory");
    std::fs::create_dir_all(root.join("src")).map_err(e)?;
    std::fs::create_dir_all(&memory).map_err(e)?;
    let body = "pub fn total(n: u64) -> u64 {\n    n * 2\n}\n";
    std::fs::write(root.join("src/lib.rs"), body).map_err(e)?;
    std::fs::write(root.join("src/main.rs"), "fn main() {\n    let x = total(3);\n}\n").map_err(e)?;
    let place = crate::outcomes::place_now(&root, "src/lib.rs", 1, 3).ok_or("prime: place")?;
    let o = nmmem::outcomes::Outcome {
        id: "fixed:x".into(),
        kind: "fixed".into(),
        verified: true,
        strength: "direct".into(),
        rule: "check".into(),
        question: None,
        ask_sha: None,
        failure: Some(nmmem::outcomes::Failure { check: "cargo-test".into(), pkg: "a".into(), code: "E0308".into(), file: "src/lib.rs".into(), line: 2, msg: String::new() }),
        places: vec![place],
        session: "s0".into(),
        at: 1,
    };
    let row = Value::Object(o.fields().into_iter().map(|(k, v)| (k.to_owned(), v)).collect());
    std::fs::write(memory.join(nmmem::outcomes::LEDGER), format!("{}\n", row.encode().map_err(|x| x.to_string())?)).map_err(e)?;
    let rec_of = |j: &str| -> Result<BTreeMap<String, Value>, String> {
        match nmjson::parse(j.as_bytes()).map_err(|x| x.to_string())? {
            Value::Object(m) => Ok(m),
            _ => Err("prime: record".into()),
        }
    };
    let fail = rec_of(r#"{"session":"s1","kind":"test","tool":"Bash","cmd_class":"cargo-test","pkg":"a","ok":false,"errors":[{"code":"E0308","file":"src/lib.rs","line":2}]}"#)?;
    let other = rec_of(r#"{"session":"s1","kind":"test","tool":"Bash","cmd_class":"cargo-test","pkg":"a","ok":false,"errors":[{"code":"E0599","file":"src/lib.rs","line":2}]}"#)?;
    let input = Value::Object(BTreeMap::new());
    let ask = |rec: &BTreeMap<String, Value>, event: &'static str, mode: Mode| -> Result<Given, Silent> {
        offer(&Ask { home: dir, memory: &memory, root: &root, event, input: &input, rec, mode, start: Instant::now() })
    };
    let g = ask(&fail, "PostToolUseFailure", Mode::Quiet).map_err(|x| format!("prime: no failure-fix ({x:?})"))?;
    if g.kind != "failure-fix" || g.places != ["src/lib.rs:1-3"] || !g.text.contains("E0308") || g.text.ends_with("...") {
        return Err(format!("prime: failure-fix {g:?}"));
    }
    if ask(&other, "PostToolUseFailure", Mode::Quiet).is_ok() || ask(&fail, "PostToolUseFailure", Mode::Off).is_ok() {
        return Err("prime: a prime for another error, or with priming off".into());
    }
    remember(&memory, "s1", &g, "PostToolUseFailure")?;
    if ask(&fail, "PostToolUseFailure", Mode::Quiet).is_ok() {
        return Err("prime: the same prime was given twice".into());
    }
    // No session quota: a session that was given other primes still gets this one, once.
    for i in 0..3 {
        let x = Given { id: format!("filler{i}"), ..g.clone() };
        remember(&memory, "s2", &x, "PostToolUseFailure")?;
    }
    let mut s2 = fail.clone();
    s2.insert("session".into(), Value::String("s2".into()));
    if ask(&s2, "PostToolUseFailure", Mode::Quiet).is_err() {
        return Err("prime: a session that was given other primes got nothing".into());
    }
    // A signature change finds the caller in another file.
    let pre = rec_of(r#"{"session":"s3","kind":"edit","tool":"Edit","lines":[["src/lib.rs",1,1]]}"#)?;
    let edit_in = nmjson::parse(br#"{"tool_input":{"old_string":"pub fn total(n: u64) -> u64 {","new_string":"pub fn total(n: u64, k: u64) -> u64 {"}}"#).map_err(|x| x.to_string())?;
    let g = offer(&Ask { home: dir, memory: &memory, root: &root, event: "PreToolUse", input: &edit_in, rec: &pre, mode: Mode::Quiet, start: Instant::now() });
    // The fix's own lines overlap the edit, so memory speaks first.
    if !g.as_ref().is_ok_and(|g| g.kind == "edit-impact" && g.text.contains("fixed cargo-test a")) {
        return Err(format!("prime: edit-impact from memory {g:?}"));
    }
    std::fs::write(root.join("src/lib.rs"), body.replace("n * 2", "n * 3")).map_err(e)?;
    if ask(&fail, "PostToolUseFailure", Mode::Quiet).is_ok() || ask(&s2, "PostToolUseFailure", Mode::On).is_ok_and(|g| g.kind == "failure-fix") {
        return Err("prime: a stale fix gave a prime".into());
    }
    let g = offer(&Ask { home: dir, memory: &memory, root: &root, event: "PreToolUse", input: &edit_in, rec: &pre, mode: Mode::Quiet, start: Instant::now() });
    if !g.as_ref().is_ok_and(|g| g.places == ["src/main.rs:2"]) {
        return Err(format!("prime: caller of a changed signature {g:?}"));
    }
    if defined("    pub fn total(n: u64) -> u64 {").as_deref() != Some("total") || defined("let x = total(3);").is_some() {
        return Err("prime: definition lines".into());
    }
    // Session open: nothing without core memories; with one in `1-core` whose lines read back, it is
    // handed on at its place; once its lines change, it is held back.
    let start = rec_of(r#"{"session":"s4","kind":"session","source":"startup"}"#)?;
    if ask(&start, "SessionStart", Mode::Quiet).is_ok() {
        return Err("prime: a session open with no core memory gave a prime".into());
    }
    let ep = nmmem::episodic::Episode::new("neuromorph", "check", "where is main", "src/main.rs:1", vec!["src/main.rs:1 lines 1..3".into()], 1);
    let rec = nmmem::migrate::record_of(&ep, nmmem::record::Tier::Core, 1, &root).to_value();
    let view = Value::Object([("items".to_owned(), Value::Array(vec![rec]))].into_iter().collect());
    std::fs::create_dir_all(memory.join("1-core")).map_err(e)?;
    std::fs::write(memory.join("1-core/items.json"), view.encode().map_err(|x| x.to_string())?).map_err(e)?;
    let g = ask(&start, "SessionStart", Mode::Quiet).map_err(|x| format!("prime: no core prime at session open ({x:?})"))?;
    if g.kind != "core" || g.places != ["src/main.rs:1-3"] || !g.text.contains("where is main") || g.text.ends_with("...") {
        return Err(format!("prime: core at session open {g:?}"));
    }
    std::fs::write(root.join("src/main.rs"), "fn main() {\n    let x = total(4);\n}\n").map_err(e)?;
    if ask(&start, "SessionStart", Mode::Quiet).is_ok() {
        return Err("prime: a core memory whose lines changed was handed on".into());
    }
    Ok(())
}
