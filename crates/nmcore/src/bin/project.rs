//! Serving any repository and the audit of real work (Neuromorph Nano, 2026-11-15, build plan E1
//! and E2). Memory per project (`nmtype::project`), the `setup` command that prints or writes a
//! repository's `.mcp.json` entry, the project list for the interface, and the observer that
//! appends one row per tool call to the one audit log, `<nano>/memory/audit.jsonl`.
//! Arguments are kept short in the log and file contents are never logged.
use nmjson::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Longest argument text kept in an audit row.
const ARG_CHARS: usize = 200;
/// Most sources listed in an audit row.
const MAX_SOURCES: usize = 24;
/// The server's name in `.mcp.json`.
pub const SERVER: &str = "neuromorph-nano";

struct Ctx {
    home: PathBuf,
    project: String,
    memory_on: bool,
    mode: String,
    via: &'static str,
}
static CTX: OnceLock<Ctx> = OnceLock::new();

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn num(v: u128) -> Value {
    Value::Number(v.to_string())
}
fn short(t: &str, max: usize) -> String {
    let one = t.split_whitespace().collect::<Vec<_>>().join(" ");
    if one.chars().count() <= max { one } else { format!("{}...", one.chars().take(max).collect::<String>()) }
}

/// Starts the audit of this process's tool calls: `via` is `serve` or `call`, `mode` names the
/// path that answers (plain, neural, whole).
pub fn start_audit(home: &Path, root: &Path, memory_on: bool, mode: &str, via: &'static str) {
    let project = nmtype::project::label(home, root);
    if via == "serve" {
        crate::live::start(home, root, &project, mode);
        // Running rows left by processes that ended mid-call, once per serving process.
        mark_stale(home);
    }
    let _ = CTX.set(Ctx { home: home.to_path_buf(), project, memory_on, mode: mode.to_owned(), via });
}

/// A tool call's running row older than this with no end row is shown and marked as `stale`
/// (a process that crashed or was killed mid-call). Longer than the longest measured call.
const STALE_MS: u64 = 10 * 60 * 1000;
/// Bytes at the end of the audit log read when looking for running rows to mark stale.
const STALE_TAIL: u64 = 512 * 1024;

/// This process's session id in running rows: process id and start time.
fn session() -> &'static str {
    static S: OnceLock<String> = OnceLock::new();
    S.get_or_init(|| format!("{}-{}", std::process::id(), nmtype::now_ms()))
}

/// The running row of the call in progress: (audit id, start time, tool).
static PENDING: std::sync::Mutex<Option<(String, u64, String)>> = std::sync::Mutex::new(None);

/// The start hook installed on the MCP server (`nmapi::mcp::on_start`): writes this call's row
/// with `state: "running"`, the parts the tool goes through and no steps yet; the row is written
/// again under the same id and time when the call ends. A row of this process still open from an
/// earlier call (its end never came) is marked stale first.
pub fn begin(tool: &str, input: &Value) {
    let Some(c) = CTX.get() else { return };
    if PENDING.lock().ok().and_then(|mut p| p.take()).is_some() {
        mark_stale(&c.home);
    }
    let now = nmtype::now_ms();
    let clock = std::time::Instant::now();
    let id = crate::controls::new_id(now);
    let mut f: Vec<(&str, Value)> = vec![
        ("tool", s(tool)),
        ("project", s(&c.project)),
        ("mode", s(&c.mode)),
        ("via", s(c.via)),
        ("client", s(&nmapi::mcp::client())),
        ("session", s(session())),
        ("state", s("running")),
        ("parts", Value::Array(parts_of(tool, input).iter().map(|p| s(p)).collect())),
        ("steps", Value::Array(Vec::new())),
    ];
    if c.memory_on {
        f.push(("label", s(&label(tool, main_arg(input).as_ref().map(|a| a.1.as_str())))));
    } else {
        f.push(("label", s(&format!("{}: memory off", label(tool, None)))));
        f.push(("memory", s("off")));
    }
    if crate::controls::audit_row(&c.home, "tool", &id, now, f).is_ok()
        && let Ok(mut p) = PENDING.lock()
    {
        *p = Some((id, now, tool.to_owned()));
    }
    crate::live::call(Some(tool));
    // The start hook's own time (the running row written) is the call's first step.
    if let Ok(mut b) = BEGUN.lock() {
        *b = Some((now, u64::try_from(clock.elapsed().as_micros()).unwrap_or(u64::MAX)));
    }
}

/// The start hook's clock for the call in progress: when the call started and the microseconds
/// the hook spent writing its running row.
static BEGUN: std::sync::Mutex<Option<(u64, u64)>> = std::sync::Mutex::new(None);

/// Appends a `stale` end row for every running row in the log's tail that has no end row: at
/// once for this session's own (its call never ended), after `STALE_MS` for other sessions'.
/// Run when serving starts and when this process finds its own previous call still open; the
/// audit view also shows a running row older than `STALE_MS` as stale.
fn mark_stale(home: &Path) {
    use std::io::{Read, Seek, SeekFrom};
    let p = home.join("memory").join("audit.jsonl");
    let Ok(mut f) = std::fs::File::open(&p) else { return };
    let len = f.metadata().map_or(0, |m| m.len());
    let mut bytes = Vec::new();
    if f.seek(SeekFrom::Start(len.saturating_sub(STALE_TAIL))).is_err() || f.read_to_end(&mut bytes).is_err() {
        return;
    }
    // The first line of the tail may be cut; it does not parse and is passed over.
    let text = String::from_utf8_lossy(&bytes);
    let now = nmtype::now_ms();
    let mut open: Vec<(String, Value)> = Vec::new();
    for v in text.lines().filter(|l| l.contains("\"state\":")).filter_map(|l| nmjson::parse(l.as_bytes()).ok()) {
        let Ok(id) = v.get("id").and_then(Value::string).map(str::to_owned) else { continue };
        open.retain(|(i, _)| *i != id);
        if v.get("state").and_then(Value::string).ok() == Some("running") {
            open.push((id, v));
        }
    }
    for (id, v) in open {
        let time = v.get("time_ms").and_then(Value::integer).unwrap_or(0);
        let mine = v.get("session").and_then(Value::string).ok() == Some(session());
        if !mine && now.saturating_sub(time) < STALE_MS {
            continue;
        }
        let Value::Object(mut o) = v else { continue };
        o.insert("state".into(), s("stale"));
        o.insert("ended".into(), s(if mine { "no end recorded before the next call" } else { "no end recorded within ten minutes" }));
        o.remove("id");
        o.remove("kind");
        o.remove("time_ms");
        let fields: Vec<(&str, Value)> = o.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        let _ = crate::controls::audit_row(home, "tool", &id, time, fields);
    }
}

/// One step of an audit row: the part, its start relative to the call's start when measured, `ms`
/// (null with `timed: false` when the part was not timed), `us` when measured that finely, and
/// what else the step reported (shared pass, step it ran inside, layers, job and lens, cells,
/// counts).
fn step_value(st: &nmapi::tools::Step, call_start: u64) -> Value {
    let mut o = BTreeMap::new();
    if let Some(j) = st.job {
        o.insert("job".to_owned(), s(j));
        if let Some(l) = &st.lens {
            o.insert("lens".to_owned(), s(l));
        }
    }
    o.insert("part".to_owned(), s(&st.part));
    if let Some(t) = st.start_ms {
        o.insert("start_ms".to_owned(), num(u128::from(t.saturating_sub(call_start))));
    }
    o.insert("ms".to_owned(), st.ms.map_or(Value::Null, |m| num(u128::from(m))));
    o.insert("timed".to_owned(), Value::Bool(st.ms.is_some()));
    if let Some(u) = st.us {
        o.insert("us".to_owned(), num(u128::from(u)));
    }
    if let Some(g) = st.shared {
        o.insert("shared".to_owned(), s(g));
    }
    if let Some(i) = st.inside {
        o.insert("inside".to_owned(), s(i));
    }
    if let Some((a, b)) = st.layers {
        o.insert("layers".to_owned(), Value::Array(vec![num(u128::from(a)), num(u128::from(b))]));
    }
    if !st.cells.is_empty() {
        o.insert("cells".to_owned(), Value::Array(st.cells.iter().map(|c| s(c)).collect()));
    }
    if !st.counts.is_empty() {
        o.insert("counts".to_owned(), Value::Object(st.counts.iter().map(|(k, v)| ((*k).to_owned(), num(u128::from(*v)))).collect()));
    }
    Value::Object(o)
}

/// The audit row's `steps` from the steps a call noted: in run order (by start; a step without a
/// start goes after every step that started before it was noted), each start relative to the call's
/// start. Without noted steps, the tool's handler (timed by the dispatcher) and the parts it goes
/// through, which run inside the handler and have no clock of their own (`timed: false`).
fn steps_value(noted: Vec<nmapi::tools::Step>, call_start: u64, tool: &str, input: &Value, ms: u128) -> Value {
    use nmapi::tools::Step;
    if noted.is_empty() {
        let parts = parts_of(tool, input);
        return Value::Array(
            parts
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let st = if i == 0 {
                        Step { part: (*p).to_owned(), start_ms: Some(call_start), ms: Some(u64::try_from(ms).unwrap_or(u64::MAX)), ..Step::default() }
                    } else {
                        Step { part: (*p).to_owned(), inside: Some("nmapi.tools"), ..Step::default() }
                    };
                    step_value(&st, call_start)
                })
                .collect(),
        );
    }
    let mut latest = 0_u64;
    let mut keyed: Vec<(u64, usize, Step)> = noted
        .into_iter()
        .enumerate()
        .map(|(i, st)| match st.start_ms {
            Some(t) => {
                latest = latest.max(t);
                (t, i, st)
            }
            None => (latest, i, st),
        })
        .collect();
    keyed.sort_by_key(|(k, i, _)| (*k, *i));
    Value::Array(keyed.iter().map(|(_, _, st)| step_value(st, call_start)).collect())
}

/// The question or main argument of a call, as (field, short text). A diff is never logged:
/// only its line count and the files it touches.
fn main_arg(input: &Value) -> Option<(&'static str, String)> {
    if let Ok(d) = input.get("diff").and_then(Value::string) {
        let files: Vec<&str> =
            d.lines().filter_map(|l| l.strip_prefix("+++ ")).map(|f| f.trim().trim_start_matches("b/")).take(6).collect();
        return Some(("diff", short(&format!("{} lines; files {}", d.lines().count(), files.join(", ")), ARG_CHARS)));
    }
    for k in ["question", "name", "crate", "query", "objective", "text", "package", "episode", "view", "op", "filter", "run", "id"] {
        if let Ok(v) = input.get(k).and_then(Value::string) {
            // Text the memory gate would refuse (a credential, key, token or e-mail address) is
            // not copied into the audit log either; the row says why it was left out.
            if let Some(why) = nmmem::classes::never_save(v, &[]) {
                return Some((k, format!("[left out: {why}]")));
            }
            return Some((k, short(v, ARG_CHARS)));
        }
    }
    if let Ok(Value::Number(i)) = input.get("index") {
        return Some(("index", i.clone()));
    }
    None
}

/// Short human text for a row.
fn label(tool: &str, arg: Option<&str>) -> String {
    let title = match tool {
        "nm_context_pack" => "Context pack",
        "nm_repo_map" => "Repo map",
        "nm_symbol_find" => "Symbol find",
        "nm_deps" => "Dependencies",
        "nm_view" => "View",
        "nm_status" => "Status",
        "nm_anatomy" => "Anatomy",
        "nm_sync" => "Sync",
        "nm_memory" => "Memory",
        "nm_feedback" => "Feedback",
        "nm_goal" => "Goal",
        "nm_weights" => "Weights",
        "nm_build_run" => "Build run",
        "nm_build_explain" => "Build explain",
        "nm_improve_scan" => "Improve scan",
        "nm_patch_propose" => "Patch propose",
        "nm_verify" => "Verify",
        "nm_stepcheck" => "Step Check",
        "nm_unit" => "Learning unit",
        other => other,
    };
    match arg.filter(|a| !a.is_empty()) {
        Some(a) => format!("{title}: {}", short(a, 60)),
        None => title.to_owned(),
    }
}

/// The sources a reply sent to the agent: packet sources, definition places, crate folders.
fn sources(v: &Value) -> Vec<Value> {
    let mut out: Vec<String> = Vec::new();
    for p in v.get("packets").and_then(Value::array).unwrap_or_default() {
        if let Ok(src) = p.get("source").and_then(Value::string) {
            out.push(src.to_owned());
        }
    }
    for d in v.get("definitions").and_then(Value::array).unwrap_or_default() {
        if let (Ok(f), Ok(l)) = (d.get("file").and_then(Value::string), d.get("line").and_then(Value::integer)) {
            out.push(format!("{f}:{l}"));
        }
    }
    for c in v.get("crates").and_then(Value::array).unwrap_or_default() {
        if let Ok(dir) = c.get("dir").and_then(Value::string) {
            out.push(if dir.is_empty() { "Cargo.toml".to_owned() } else { format!("{dir}/Cargo.toml") });
        }
    }
    if let Ok(src) = v.get("source").and_then(Value::string) {
        out.push(src.to_owned());
    }
    out.dedup();
    out.into_iter().take(MAX_SOURCES).map(|x| s(&x)).collect()
}

/// The sources a plain-text reply sent: its `@@ path:first-last` and `~~ source` headers. Each
/// header's body is skipped by its length (the lines first to last after `@@`; after `~~`, the
/// count shown as `(N lines)`, else one line), so a quoted code line that itself starts with
/// `@@ ` or `~~ ` is never taken for a header.
fn text_sources(t: &str) -> Vec<Value> {
    let mut out = Vec::new();
    let mut lines = t.lines();
    while let Some(l) = lines.next() {
        let (place, body) = if let Some(h) = l.strip_prefix("@@ ") {
            let place = h.split_whitespace().next().unwrap_or_default().trim_end_matches(';');
            let span = place
                .rsplit_once(':')
                .and_then(|(_, r)| r.split_once('-'))
                .and_then(|(a, b)| Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?)));
            (place, span.map_or(0, |(a, b)| b.saturating_sub(a) + 1))
        } else if let Some(h) = l.strip_prefix("~~ ") {
            let count = h.split_once(" lines)").and_then(|(a, _)| a.rsplit_once('(')).and_then(|(_, n)| n.parse::<usize>().ok());
            (h.split_whitespace().next().unwrap_or_default().trim_end_matches(';'), count.unwrap_or(1))
        } else {
            continue;
        };
        if out.len() < MAX_SOURCES {
            out.push(s(place));
        }
        for _ in 0..body {
            lines.next();
        }
    }
    out
}

/// The observer installed on the served contract (`nmapi::Api::observed`): one audit row per
/// call. With memory off the row holds only the tool, project, mode, time and outcome.
pub fn observe(tool: &str, input: &Value, out: &Result<Value, nmapi::ApiError>, ms: u128) {
    // Taken on every call so one call's diagnostics never reach another call's row.
    let diag = nmapi::tools::take_diagnostics();
    let mut noted = nmapi::tools::take_steps();
    if let Some((at, us)) = BEGUN.lock().ok().and_then(|mut b| b.take())
        && !noted.is_empty()
    {
        noted.insert(
            0,
            nmapi::tools::Step {
                part: "nmapi.mcp".into(),
                start_ms: Some(at),
                ms: Some(us / 1000),
                us: Some(us),
                cells: vec!["running row".into()],
                ..nmapi::tools::Step::default()
            },
        );
    }
    let pending = PENDING.lock().ok().and_then(|mut p| p.take()).filter(|p| p.2 == tool);
    crate::live::call(None);
    let Some(c) = CTX.get() else { return };
    let call_start = pending.as_ref().map_or_else(|| nmtype::now_ms().saturating_sub(u64::try_from(ms).unwrap_or(0)), |p| p.1);
    let mut f: Vec<(&str, Value)> = vec![
        ("tool", s(tool)),
        ("project", s(&c.project)),
        ("mode", s(&c.mode)),
        ("via", s(c.via)),
        ("client", s(&nmapi::mcp::client())),
        ("ms", num(ms)),
        ("ok", Value::Bool(out.is_ok())),
        ("state", s("done")),
        ("session", s(session())),
        ("steps", steps_value(noted, call_start, tool, input, ms)),
    ];
    if let Err(e) = out {
        f.push(("error", s(e.code)));
    }
    if c.memory_on {
        let arg = main_arg(input);
        f.push(("label", s(&label(tool, arg.as_ref().map(|a| a.1.as_str())))));
        if let Some((k, v)) = arg {
            f.push(("argument_field", s(k)));
            f.push(("argument", s(&v)));
        }
        for k in ["form", "kind"] {
            if let Ok(v) = input.get(k).and_then(Value::string) {
                f.push((k, s(&short(v, 40))));
            }
        }
        if let Ok(v) = out {
            let (chars, src) = match v {
                Value::String(t) => (t.len(), text_sources(t)),
                _ => (v.encode().map_or(0, |t| t.len()), sources(v)),
            };
            f.push(("reply_tokens", num(chars as u128 / 4)));
            // 2026-11-27 (C3): the request by the same count (its arguments, characters / 4).
            f.push(("request_tokens", num(input.encode().map_or(0, |t| t.len()) as u128 / 4)));
            if !src.is_empty() {
                f.push(("sources", Value::Array(src)));
            }
            let d = diag.as_ref();
            if let Some(e) =
                v.get("episode").and_then(Value::string).ok().or_else(|| d.and_then(|d| d.get("episode").and_then(Value::string).ok()))
            {
                f.push(("episode", s(e)));
            }
            if let Some(w) =
                [v.get("withheld").ok(), d.and_then(|d| d.get("withheld").ok())].into_iter().flatten().find(|w| matches!(w, Value::Bool(_)))
            {
                f.push(("withheld", w.clone()));
            }
        }
        // Diagnostics that hold no code go to the row, not to the agent.
        if let Some(Value::Object(d)) = &diag {
            for k in ["parts", "floor_met", "context_map", "rerank_best", "memory_write", "memory_recalled", "embed_check", "config"] {
                if let Some(x) = d.get(k) {
                    f.push((k, x.clone()));
                }
            }
        }
        if !f.iter().any(|x| x.0 == "parts") {
            f.push(("parts", Value::Array(parts_of(tool, input).iter().map(|p| s(p)).collect())));
        }
    } else {
        f.push(("label", s(&format!("{}: memory off", label(tool, None)))));
        f.push(("memory", s("off")));
    }
    let row = match pending {
        Some((id, time, _)) => crate::controls::audit_row(&c.home, "tool", &id, time, f).map(|()| id),
        None => crate::controls::audit(&c.home, "tool", f),
    };
    if c.memory_on
        && let (Ok(v), Ok(row)) = (out, row)
    {
        outcome(&c.home, &c.project, tool, input, v, &row);
    }
}

/// The parts a tool call goes through, for tools whose reply does not say: the crates and
/// stages its handler calls.
fn parts_of(tool: &str, input: &Value) -> Vec<&'static str> {
    let mut p = vec!["nmapi.tools"];
    p.extend_from_slice(match tool {
        "nm_status" => &["nmnte", "nmsubs", "nmapi.delta"],
        "nm_anatomy" if input.get("level").is_ok() => &["nmapi.anatomy", "nmnte"],
        "nm_anatomy" => &["nmnte"],
        "nm_repo_map" | "nm_symbol_find" | "nm_deps" => &["nmsubs"],
        "nm_view" if input.get("zoom").is_ok() => &["nmsubs.zoom"],
        "nm_view" => &["nmsubs.views"],
        "nm_sync" => &["nmsubs", "nmapi.delta"],
        "nm_memory" => &["nmmem"],
        "nm_feedback" => &["nmmem.episodic", "nmlearn.ledger", "nmapi.brain_side"],
        "nm_goal" => &["nmmem.structural"],
        "nm_weights" => &["nmapi.neural", "nmengn"],
        "nm_build_run" | "nm_build_explain" | "nm_improve_scan" | "nm_patch_propose" | "nm_stepcheck" => &["nmapi.build", "nmtool"],
        "nm_verify" => &["nmtool.verify"],
        "nm_unit" => &["nmapi.learn", "nmlearn"],
        _ => &[],
    });
    p
}

/// The audit id of the newest context pack row that produced `episode`, if any.
fn row_of(home: &Path, episode: &str) -> Option<String> {
    let text = std::fs::read_to_string(home.join("memory").join("audit.jsonl")).ok()?;
    let needle = format!("\"episode\":\"{episode}\"");
    text.lines()
        .rev()
        .filter(|l| l.contains(&needle))
        .filter_map(|l| nmjson::parse(l.as_bytes()).ok())
        .find(|v| {
            v.get("kind").and_then(Value::string).ok() == Some("tool")
                && v.get("tool").and_then(Value::string).ok() == Some("nm_context_pack")
        })
        .and_then(|v| v.get("id").and_then(Value::string).ok().map(str::to_owned))
}

/// An `outcome` row when a call reports the outcome of an earlier episode: `nm_feedback` (right
/// or wrong) and `nm_verify` given an `episode` (its findings counted), linked by episode id and,
/// when found, by the audit id of the row that produced the episode.
fn outcome(home: &Path, project: &str, tool: &str, input: &Value, out: &Value, row: &str) {
    let Ok(episode) = input.get("episode").and_then(Value::string) else { return };
    let mut f: Vec<(&str, Value)> = vec![("episode", s(episode)), ("by", s(tool)), ("project", s(project)), ("reported_in", s(row))];
    match tool {
        "nm_feedback" => {
            let correct = matches!(out.get("correct"), Ok(Value::Bool(true)));
            f.push(("label", s(&format!("Outcome: episode {} {}", short(episode, 16), if correct { "right" } else { "wrong" }))));
            f.push(("correct", Value::Bool(correct)));
            if let Ok(t) = input.get("taste").and_then(Value::string) {
                f.push(("taste", s(&short(t, 20))));
            }
        }
        "nm_verify" => {
            let found = out.array().unwrap_or_default();
            let open = found.iter().filter(|x| matches!(x.get("open"), Ok(Value::Bool(true)))).count();
            let mut dims: Vec<&str> = found.iter().filter_map(|x| x.get("dimension").and_then(Value::string).ok()).collect();
            dims.dedup();
            f.push(("label", s(&format!("Outcome: episode {} verified, {} findings, {open} open", short(episode, 16), found.len()))));
            f.push(("findings", num(found.len() as u128)));
            f.push(("open", num(open as u128)));
            f.push(("dimensions", Value::Array(dims.iter().map(|d| s(d)).collect())));
        }
        _ => return,
    }
    if let Some(of) = row_of(home, episode) {
        f.push(("of", s(&of)));
    }
    let _ = crate::controls::audit(home, "outcome", f);
}

/// Writes `project.json` in a project memory folder: the studied root and when it was first and
/// last served. Only with memory on, and only for project folders (not the Nano's own memory).
pub fn note_project(home: &Path, root: &Path, memory_dir: &Path) {
    if nmtype::project::is_home(home, root) || memory_dir != nmtype::project::memory_dir(home, root) {
        return;
    }
    let p = memory_dir.join("project.json");
    let now = nmtype::now_ms();
    let first = std::fs::read(&p)
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .and_then(|v| v.get("first_ms").ok().cloned())
        .unwrap_or_else(|| Value::Number(now.to_string()));
    let mut o = BTreeMap::new();
    o.insert("key".to_owned(), s(&nmtype::project::key(root)));
    o.insert("root".to_owned(), s(&root.display().to_string()));
    o.insert("first_ms".to_owned(), first);
    o.insert("last_ms".to_owned(), Value::Number(now.to_string()));
    o.insert("note".to_owned(), s("Memory of one studied repository, kept by the Neuromorph Nano (written by serve)."));
    if std::fs::create_dir_all(memory_dir).is_ok() {
        let _ = std::fs::write(&p, nmjson::pretty::pretty(&Value::Object(o)));
    }
}

/// Every project memory folder for the interface: key, root, first and last served, bytes.
#[must_use]
pub fn list(home: &Path) -> Value {
    let dir = nmtype::project::projects_dir(home);
    let mut rows: Vec<(String, Value)> = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().is_dir())
                .map(|e| {
                    let key = e.file_name().to_string_lossy().into_owned();
                    let info = std::fs::read(e.path().join("project.json")).ok().and_then(|b| nmjson::parse(&b).ok());
                    let field = |k: &str| info.as_ref().and_then(|v| v.get(k).ok().cloned()).unwrap_or(Value::Null);
                    let episodes = std::fs::read_to_string(e.path().join("episodes.jsonl"))
                        .map(|t| t.lines().filter(|l| !l.trim().is_empty()).count())
                        .unwrap_or(0);
                    let mut o = BTreeMap::new();
                    o.insert("key".to_owned(), s(&key));
                    o.insert("root".to_owned(), field("root"));
                    o.insert("first_ms".to_owned(), field("first_ms"));
                    o.insert("last_ms".to_owned(), field("last_ms"));
                    o.insert("episodes".to_owned(), Value::Number(episodes.to_string()));
                    o.insert("bytes".to_owned(), Value::Number(bytes(&e.path()).to_string()));
                    o.insert("tiers".to_owned(), crate::activity::tiers(&e.path()));
                    o.insert("tier_topics".to_owned(), crate::activity::tier_topics(&e.path()));
                    o.insert("tiers_overlap".to_owned(), Value::Number(crate::activity::tiers_overlap(&e.path()).to_string()));
                    o.insert("dense".to_owned(), dense_size(&e.path()));
                    o.insert("dir".to_owned(), s(&e.path().display().to_string()));
                    (key, Value::Object(o))
                })
                .collect()
        })
        .unwrap_or_default();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    // Served now: the serving processes with a fresh live file (`crate::live`).
    let live = crate::live::read_all(home);
    let key_of = |v: &Value| v.get("project_key").and_then(Value::string).ok().map(str::to_owned);
    let served: Vec<String> = live.iter().filter_map(key_of).collect();
    for (key, row) in &mut rows {
        if let Value::Object(o) = row {
            o.insert("serving".to_owned(), Value::Bool(served.contains(key)));
        }
    }
    let serving: Vec<Value> = live
        .iter()
        .map(|v| {
            let mut o = BTreeMap::new();
            for k in ["project", "project_key", "root", "mode", "pid", "started_ms", "beat_ms", "call"] {
                if let Ok(x) = v.get(k) {
                    o.insert(k.to_owned(), x.clone());
                }
            }
            Value::Object(o)
        })
        .collect();
    let mut out = BTreeMap::new();
    out.insert("serving".to_owned(), Value::Array(serving));
    out.insert(
        "current".to_owned(),
        live.first().and_then(|v| v.get("project").ok().cloned()).unwrap_or(Value::Null),
    );
    out.insert("dir".to_owned(), s(&dir.display().to_string()));
    out.insert("nano_dense".to_owned(), dense_size(&home.join("memory")));
    out.insert("projects".to_owned(), Value::Array(rows.into_iter().map(|r| r.1).collect()));
    Value::Object(out)
}
/// The dense index kept on disk in a memory folder (`dense/` segments + `skin.json`):
/// `{rows, bytes}`, summed across all segments; null when there is none.
fn dense_size(memory: &Path) -> Value {
    let dir = memory.join(nmsubs::dense::store::DIR);
    let Some((rows, bytes)) = nmsubs::dense::store::disk_usage(&dir) else { return Value::Null };
    let mut o = BTreeMap::new();
    o.insert("rows".to_owned(), Value::Number(rows.to_string()));
    o.insert("bytes".to_owned(), Value::Number(bytes.to_string()));
    Value::Object(o)
}
fn bytes(p: &Path) -> u64 {
    std::fs::read_dir(p)
        .map(|rd| {
            rd.flatten()
                .map(|e| match e.metadata() {
                    Ok(m) if m.is_dir() => bytes(&e.path()),
                    Ok(m) => m.len(),
                    Err(_) => 0,
                })
                .sum()
        })
        .unwrap_or(0)
}

/// The program the hook entries run: the launcher copy in the Nano's `bin` folder.
fn hook_exe(home: &Path) -> String {
    crate::launch::launcher_path(home).display().to_string()
}

/// The user's home folder, where Claude Code keeps its own user settings: setup never writes a
/// project file there.
pub fn is_user_home(repo: &Path) -> bool {
    ["USERPROFILE", "HOME"]
        .iter()
        .filter_map(std::env::var_os)
        .filter(|h| !h.is_empty())
        .any(|h| nmtype::project::normal(&nmsubs::repo::canon(Path::new(&h))) == nmtype::project::normal(repo))
}

/// Reads a settings file as text: `None` when it does not exist.
pub fn read_text(file: &Path) -> Result<Option<String>, String> {
    match std::fs::read(file) {
        Ok(b) => String::from_utf8(b).map(Some).map_err(|_| format!("{} is not UTF-8 text", file.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", file.display())),
    }
}

/// `neuromorph-nano setup --project <repo> ...`:
/// - no mode flag: prints the `.mcp.json` entry (the whole path; `--plain` for the plain path)
///   and the file it would write; `--write` merges it in;
/// - `--hooks`: prints the hook entries and `.claude/settings.local.json` as it would be;
///   `--write` merges them in;
/// - `--remove`: takes out the Nano's MCP entry and hook entries and says what it removed;
/// - `--status`: reports what is installed.
///
/// Every write keeps the other entries byte for byte, writes a backup copy of the file as it was
/// into the project's memory folder (`setup/backups/`) and logs an audit row.
pub fn setup(home: &Path, args: &[String]) -> std::process::ExitCode {
    use std::process::ExitCode;
    let Some(repo) = args.windows(2).find(|w| w.first().map(String::as_str) == Some("--project")).and_then(|w| w.get(1)) else {
        eprintln!(
            "usage: neuromorph-nano setup --project <repository> [--plain] [--lean] [--write]\n       neuromorph-nano setup --project <repository> --hooks [--write]\n       neuromorph-nano setup --project <repository> --antigravity|--agy [--plain] [--hooks] [--write]\n       neuromorph-nano setup --project <repository> [--antigravity|--agy] --remove\n       neuromorph-nano setup --project <repository> --status"
        );
        return ExitCode::FAILURE;
    };
    let repo = Path::new(repo);
    if !repo.is_dir() {
        eprintln!("setup: {} is not a folder", repo.display());
        return ExitCode::FAILURE;
    }
    let repo = nmsubs::repo::canon(repo);
    let has = |f: &str| args.iter().any(|a| a == f);
    let agy = has("--agy") || has("--antigravity");
    let r = if has("--status") {
        status(home, &repo)
    } else if agy && has("--remove") {
        remove_agy(home, &repo)
    } else if agy {
        setup_agy(home, &repo, has("--plain"), has("--hooks"), has("--write"))
    } else if has("--remove") {
        remove(home, &repo)
    } else if has("--hooks") {
        setup_hooks(home, &repo, has("--write"))
    } else {
        setup_mcp(home, &repo, has("--plain"), has("--lean"), has("--write"))
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("setup: {e}");
            ExitCode::FAILURE
        }
    }
}

/// The Claude Code project entry: `.mcp.json`, `mcpServers`, `"type": "stdio"` with `command`
/// and `args`, as documented at <https://code.claude.com/docs/en/mcp> (read 2026-11-25; Roadmap
/// Phase 4.6). The same entry can be written through Claude Code itself with the printed
/// `claude mcp add --transport stdio --scope project` line.
pub fn setup_mcp(home: &Path, repo: &Path, plain: bool, lean_flag: bool, write: bool) -> Result<(), String> {
    use crate::connect;
    use nmcore::mcp_clients::{self as clients, Client};
    let launcher = crate::launch::launcher_path(home);
    let file = repo.join(".mcp.json");
    let existing = read_text(&file)?;
    let e = connect::mcp_entry(&launcher, repo, plain);
    let plan = connect::plan_mcp(existing.as_deref(), &e).map_err(|x| format!("refused, nothing changed: {x}"))?;
    let st = crate::jsonc::Style { nl: "\n", unit: "  ".into() };
    println!("entry for {} (server \"{SERVER}\"):\n{}", repo.display(), crate::jsonc::pretty(&e, "", &st));
    println!("the same entry through Claude Code itself: {}", clients::claude_add_line(&launcher, repo, plain));
    println!("memory folder: {}", nmtype::project::memory_dir(home, repo).display());
    if let Some(t) = &plan.text {
        // Roadmap Phase 4.6 (2026-11-25): checked against the documented shape before anything is
        // written; the dry run shows the line difference.
        clients::check_shape(Client::Claude, t).map_err(|x| format!("refused, nothing changed: the merged .mcp.json would not fit Claude Code's format ({x})"))?;
    }
    match &plan.text {
        None => println!("{} already holds this entry; nothing to change", file.display()),
        Some(t) if !write => println!(
            "{} {} (line difference; - taken out, + put in):\n{}",
            if existing.is_some() { "would merge into" } else { "would write" },
            file.display(),
            clients::diff(existing.as_deref(), t)
        ),
        Some(_) => {}
    }
    if !write {
        println!("launcher: {} (copied from the newest build on --write)", launcher.display());
        println!("nothing written; add --write to write it");
        return Ok(());
    }
    println!("{}", crate::launch::write_launcher(home)?);
    let lean = lean_flag || nmapi::whole::claude_lean_control(home);
    approve_mcp(home, repo, lean)?;
    let Some(text) = plan.text else { return Ok(()) };
    let fresh = existing.as_deref().and_then(connect::mcp_installed).is_none();
    let bak = connect::backup(home, repo, &file, existing.as_deref().map(str::as_bytes))?;
    connect::write_file(&file, &text)?;
    if fresh {
        connect::save_record(home, repo, "mcp", Some(&plan.created))?;
    }
    let _ = crate::controls::audit(
        home,
        "setup",
        vec![
            ("label", s(&format!("MCP entry written: {}", short(&repo.display().to_string(), 60)))),
            ("project", s(&nmtype::project::label(home, repo))),
            ("file", s(&file.display().to_string())),
            ("merged", Value::Bool(existing.is_some())),
            ("backup", s(&bak.display().to_string())),
        ],
    );
    println!(
        "wrote {}: {} the server entry \"{SERVER}\" shown above; backup of the file as it was: {}",
        file.display(),
        if existing.is_some() { "merged in, other entries kept," } else { "a new file holding" },
        bak.display()
    );
    Ok(())
}

/// The approval of the Nano's project server in the project's personal settings file
/// (`enabledMcpjsonServers` in `.claude/settings.local.json`), which Claude Code in the terminal and
/// in the desktop app read before starting a server of the project's `.mcp.json`; written with the
/// entry, with a backup, removed by `--remove`.
fn approve_mcp(home: &Path, repo: &Path, lean: bool) -> Result<(), String> {
    use crate::connect;
    let file = repo.join(".claude").join("settings.local.json");
    let existing = read_text(&file)?;
    let plan = connect::plan_approval_with(existing.as_deref(), lean).map_err(|x| format!("refused, nothing changed: {x}"))?;
    let Some(text) = plan.text else {
        println!("{} already approves the server \"{SERVER}\"", file.display());
        return Ok(());
    };
    let bak = connect::backup(home, repo, &file, existing.as_deref().map(str::as_bytes))?;
    let made_dir = connect::write_file(&file, &text)?;
    let mut c = plan.created.clone();
    c.dir = made_dir;
    connect::save_record(home, repo, "mcp-approval", Some(&c))?;
    println!(
        "wrote {}: the server \"{SERVER}\" approved ({} added), so Claude Code starts it without asking; backup: {}",
        file.display(),
        connect::APPROVAL,
        bak.display()
    );
    Ok(())
}

pub fn setup_hooks(home: &Path, repo: &Path, write: bool) -> Result<(), String> {
    use crate::connect;
    let file = repo.join(".claude").join("settings.local.json");
    let shown = ".claude/settings.local.json";
    let existing = read_text(&file)?;
    let exe = hook_exe(home);
    let (timeout, why) = connect::timeout_s(&nmtype::project::memory_dir(home, repo), &exe);
    let plan = connect::plan_hooks(existing.as_deref(), &exe, shown, timeout).map_err(|x| format!("refused, nothing changed: {x}"))?;
    let st = crate::jsonc::Style { nl: "\n", unit: "  ".into() };
    println!(
        "hook entries for {} (each runs {exe} hook <Event>, records what Claude Code did, prints nothing and never blocks):\n{}",
        repo.display(),
        crate::jsonc::pretty(&crate::jsonc::Out::Obj(vec![("hooks".into(), connect::hooks_object(&exe, timeout))]), "", &st)
    );
    println!("hook timeout {timeout} s: {why}; run setup again to write it anew");
    println!("situation record: {}", crate::hook::situation_file(&nmtype::project::memory_dir(home, repo)).display());
    match &plan.text {
        None => println!("{} already holds these entries; nothing to change", file.display()),
        Some(t) if !write => println!("{} {}:\n{t}", if existing.is_some() { "would merge into" } else { "would write" }, file.display()),
        Some(_) => {}
    }
    if !write {
        println!("nothing written; add --write to write it (only this personal settings file is written, never .claude/settings.json)");
        return Ok(());
    }
    if is_user_home(repo) {
        return Err(format!("refused: {} is your home folder, where Claude Code keeps its user settings; nothing changed", repo.display()));
    }
    println!("{}", crate::launch::write_launcher(home)?);
    let Some(text) = plan.text else { return Ok(()) };
    let fresh = existing.as_deref().is_none_or(|t| connect::installed(t).0.is_empty());
    let bak = connect::backup(home, repo, &file, existing.as_deref().map(str::as_bytes))?;
    let made_dir = connect::write_file(&file, &text)?;
    if fresh {
        let mut c = plan.created.clone();
        c.dir = made_dir;
        connect::save_record(home, repo, "settings-local", Some(&c))?;
    }
    let _ = crate::controls::audit(
        home,
        "setup",
        vec![
            ("label", s(&format!("Hook entries written: {}", short(&repo.display().to_string(), 60)))),
            ("project", s(&nmtype::project::label(home, repo))),
            ("file", s(&file.display().to_string())),
            ("events", Value::Array(connect::events().iter().map(|e| s(e)).collect())),
            ("backup", s(&bak.display().to_string())),
        ],
    );
    println!(
        "wrote {}: {} the {} hook handlers shown above ({}); backup of the file as it was: {}",
        file.display(),
        if existing.is_some() { "merged in, other entries kept," } else { "a new file holding" },
        connect::GROUPS.len(),
        connect::events().join(", "),
        bak.display()
    );
    Ok(())
}

fn agy_plugin_dir(repo: &Path) -> PathBuf {
    repo.join(".agents").join("plugins").join(SERVER)
}

/// The plugin's `mcp_config.json`: `mcpServers` with `command` and `args` (no `type`), at the
/// plugin root, as documented at <https://antigravity.google/docs/build-plugins> and
/// <https://antigravity.google/docs/mcp> (read 2026-11-25; Roadmap Phase 4.6). The plugin folder
/// is the Nano's own, so the file is written whole (with a backup of what was there).
fn agy_mcp_text(launcher: &Path, repo: &Path, plain: bool) -> String {
    nmcore::mcp_clients::new_file_text(nmcore::mcp_clients::Client::Antigravity, launcher, repo, plain)
}

fn agy_hooks_text(exe: &str, timeout: i64) -> String {
    use crate::jsonc::{Out, Style, pretty};
    let cmd = |ev: &str| {
        let q = if exe.contains(' ') { format!("\"{exe}\" hook {ev}") } else { format!("{exe} hook {ev}") };
        Out::Obj(vec![("type".into(), Out::Str("command".into())), ("command".into(), Out::Str(q)), ("timeout".into(), Out::Num(timeout))])
    };
    let group = |matcher: &str, ev: &str| Out::Obj(vec![("matcher".into(), Out::Str(matcher.into())), ("hooks".into(), Out::Arr(vec![cmd(ev)]))]);
    // PreToolUse is intentionally omitted: Antigravity PreToolUse requires a permission decision
    // and does not support injectSteps, while PostToolUse records every edit and write safely.
    let nano = Out::Obj(vec![
        ("SessionStart".into(), Out::Arr(vec![cmd("SessionStart")])),
        ("PreInvocation".into(), Out::Arr(vec![cmd("PreInvocation")])),
        ("PostToolUse".into(), Out::Arr(vec![group("*", "PostToolUse")])),
        ("Stop".into(), Out::Arr(vec![cmd("Stop")])),
    ]);
    let root = Out::Obj(vec![(SERVER.into(), nano)]);
    let st = Style { nl: "\n", unit: "  ".into() };
    format!("{}\n", pretty(&root, "", &st))
}

fn agy_rules_text() -> &'static str {
    "# Neuromorph Nano\n\nWhen answering questions about this codebase or locating symbols, call `nm_context_pack` first before broad grep or multi-file reads. Use `nm_symbol_find`, `nm_deps`, `nm_view`, `nm_build_run`, and `nm_stepcheck` when applicable, and report outcomes with `nm_feedback` when verified.\n"
}

/// Connects a repository to Antigravity (CLI, IDE, or desktop app) via a self-contained project
/// plugin under `.agents/plugins/neuromorph-nano/` holding `plugin.json`, `mcp_config.json`,
/// `rules/AGENTS.md`, and optionally `hooks.json`.
pub fn setup_agy(home: &Path, repo: &Path, plain: bool, hooks: bool, write: bool) -> Result<(), String> {
    use crate::connect;
    let dir = agy_plugin_dir(repo);
    let launcher = crate::launch::launcher_path(home);
    let plugin_json = format!("{{\n  \"name\": \"{SERVER}\",\n  \"version\": \"{}\",\n  \"author\": \"Gavriel Technologies\"\n}}\n", env!("CARGO_PKG_VERSION"));
    let mcp_json = agy_mcp_text(&launcher, repo, plain);
    let exe = hook_exe(home);
    let (timeout, why) = connect::timeout_s(&nmtype::project::memory_dir(home, repo), &exe);
    let hooks_json = hooks.then(|| agy_hooks_text(&exe, timeout));
    println!("Antigravity plugin folder: {}", dir.display());
    println!("memory folder: {}", nmtype::project::memory_dir(home, repo).display());
    println!("mcp_config.json:\n{mcp_json}");
    if let Some(h) = &hooks_json {
        println!("hooks.json (timeout {timeout} s: {why}):\n{h}");
    }
    if !write {
        println!("nothing written; add --write to write the Antigravity plugin");
        return Ok(());
    }
    if is_user_home(repo) {
        return Err(format!("refused: {} is your home folder; nothing changed", repo.display()));
    }
    println!("{}", crate::launch::write_launcher(home)?);
    let write_part = |rel: &str, body: &str| -> Result<(), String> {
        let p = dir.join(rel);
        if let Some(prev) = read_text(&p)? {
            let _ = connect::backup(home, repo, &p, Some(prev.as_bytes()))?;
        }
        connect::write_file(&p, body).map(|_| ())
    };
    write_part("plugin.json", &plugin_json)?;
    write_part("mcp_config.json", &mcp_json)?;
    write_part("rules/AGENTS.md", agy_rules_text())?;
    if let Some(h) = &hooks_json {
        write_part("hooks.json", h)?;
    }
    let _ = crate::controls::audit(
        home,
        "setup",
        vec![
            ("label", s(&format!("Antigravity plugin written: {}", short(&repo.display().to_string(), 60)))),
            ("project", s(&nmtype::project::label(home, repo))),
            ("file", s(&dir.display().to_string())),
            ("hooks", Value::Bool(hooks)),
        ],
    );
    println!("wrote Antigravity plugin at {} (MCP{}, rules/AGENTS.md)", dir.display(), if hooks { ", hooks.json" } else { "" });
    Ok(())
}

pub fn remove_agy(home: &Path, repo: &Path) -> Result<(), String> {
    let dir = agy_plugin_dir(repo);
    if !dir.exists() {
        println!("{}: not present", dir.display());
        return Ok(());
    }
    std::fs::remove_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for parent in [repo.join(".agents").join("plugins"), repo.join(".agents")] {
        let _ = std::fs::remove_dir(&parent);
    }
    let _ = crate::controls::audit(
        home,
        "setup",
        vec![
            ("label", s(&format!("Antigravity plugin removed: {}", short(&repo.display().to_string(), 60)))),
            ("project", s(&nmtype::project::label(home, repo))),
            ("file", s(&dir.display().to_string())),
        ],
    );
    println!("removed Antigravity plugin {}", dir.display());
    Ok(())
}

pub fn remove(home: &Path, repo: &Path) -> Result<(), String> {
    use crate::connect;
    let mut said = Vec::new();
    let mcp = repo.join(".mcp.json");
    let hooks = repo.join(".claude").join("settings.local.json");
    // Both files are checked before either is changed, so a refusal changes nothing.
    let mcp_text = read_text(&mcp)?;
    let hooks_text = read_text(&hooks)?;
    let mcp_rec = connect::record(home, repo, "mcp");
    let hooks_rec = connect::record(home, repo, "settings-local");
    let mcp_plan = match &mcp_text {
        Some(t) => Some(connect::unplan_mcp(t, mcp_rec.as_ref()).map_err(|x| format!("refused, nothing changed: {x}"))?),
        None => None,
    };
    // The approval of the server comes out of the same file first, then the hook handlers.
    let appr_rec = connect::record(home, repo, "mcp-approval");
    let (mid, approved) = match &hooks_text {
        Some(t) => connect::unplan_approval(t, appr_rec.as_ref()).map_err(|x| format!("refused, nothing changed: {x}"))?,
        None => (None, false),
    };
    let hooks_plan = match (&hooks_text, &mid) {
        (Some(_), Some(t)) => {
            let (after, events) =
                connect::unplan_hooks(t, hooks_rec.as_ref(), ".claude/settings.local.json").map_err(|x| format!("refused, nothing changed: {x}"))?;
            let gone = after.is_none() || (appr_rec.as_ref().is_some_and(|c| c.file) && after.as_deref().is_some_and(connect::is_empty_object));
            Some((if gone { None } else { after }, events))
        }
        (Some(_), None) => Some((None, Vec::new())),
        (None, _) => None,
    };
    let mut apply = |file: &Path, before: &str, after: Option<&String>, what: String, which: &str| -> Result<(), String> {
        let bak = connect::backup(home, repo, file, Some(before.as_bytes()))?;
        match after {
            Some(t) => connect::write_file(file, t).map(|_| ())?,
            None => std::fs::remove_file(file).map_err(|e| format!("{}: {e}", file.display()))?,
        }
        connect::save_record(home, repo, which, None)?;
        let removed_file = if after.is_none() { " (setup had created the file, so it was removed)" } else { "" };
        said.push(format!("{what} from {}{removed_file}; backup of the file as it was: {}", file.display(), bak.display()));
        Ok(())
    };
    match (&mcp_text, &mcp_plan) {
        (Some(before), Some((after, true))) => apply(&mcp, before, after.as_ref(), format!("removed the MCP server entry \"{SERVER}\""), "mcp")?,
        (Some(_), _) => println!("{}: no Nano entry", mcp.display()),
        (None, _) => println!("{}: not present", mcp.display()),
    }
    let mut dir_note = None;
    match (&hooks_text, &hooks_plan) {
        (Some(before), Some((after, events))) if !events.is_empty() || approved => {
            let mut what = Vec::new();
            if approved {
                what.push(format!("removed the approval of \"{SERVER}\""));
            }
            if !events.is_empty() {
                what.push(format!("removed {} hook handlers ({})", events.len(), events.join(", ")));
            }
            apply(&hooks, before, after.as_ref(), what.join(" and "), "settings-local")?;
            connect::save_record(home, repo, "mcp-approval", None)?;
            let made_dir = hooks_rec.as_ref().is_some_and(|c| c.dir) || appr_rec.as_ref().is_some_and(|c| c.dir);
            if made_dir && after.is_none() {
                // Only an empty folder is removed; anything else in it stays.
                if let Some(d) = hooks.parent()
                    && std::fs::remove_dir(d).is_ok()
                {
                    dir_note = Some(format!("removed the folder {} (setup had created it and it is empty)", d.display()));
                }
            }
        }
        (Some(_), _) => println!("{}: no Nano hook entries", hooks.display()),
        (None, _) => println!("{}: not present", hooks.display()),
    }
    said.extend(dir_note);
    for l in &said {
        println!("{l}");
    }
    if said.is_empty() {
        println!("nothing to remove");
    } else {
        let _ = crate::controls::audit(
            home,
            "setup",
            vec![
                ("label", s(&format!("Claude Code entries removed: {}", short(&repo.display().to_string(), 60)))),
                ("project", s(&nmtype::project::label(home, repo))),
                ("removed", Value::Array(said.iter().map(|l| s(&short(l, 300))).collect())),
            ],
        );
    }
    Ok(())
}

pub fn status(home: &Path, repo: &Path) -> Result<(), String> {
    use crate::connect;
    let memory = nmtype::project::memory_dir(home, repo);
    println!("project: {}", repo.display());
    println!("memory folder: {}", memory.display());
    let mcp = repo.join(".mcp.json");
    match read_text(&mcp)? {
        None => println!("MCP entry: none ({} not present)", mcp.display()),
        Some(t) => match connect::mcp_installed(&t) {
            Some(c) => println!("MCP entry: \"{SERVER}\" in {} runs {c}", mcp.display()),
            None => println!("MCP entry: none in {}", mcp.display()),
        },
    }
    let exe = hook_exe(home);
    for (name, shared) in [("settings.local.json", false), ("settings.json", true)] {
        let f = repo.join(".claude").join(name);
        let Some(t) = read_text(&f)? else {
            if !shared {
                println!("hook entries: none ({} not present)", f.display());
            }
            continue;
        };
        let (ours, other, cmds) = connect::installed(&t);
        if ours.is_empty() && other == 0 {
            if !shared {
                println!("hook entries: none in {}", f.display());
            }
            continue;
        }
        let mut ev = ours.clone();
        ev.dedup();
        println!("hook entries in {}: {} handlers ({})", f.display(), ours.len(), ev.join(", "));
        for c in &cmds {
            let note = if *c == exe {
                if Path::new(c).is_file() { "this Nano's launcher" } else { "this Nano's launcher, NOT PRESENT: run setup --hooks --write again" }
            } else {
                "another Nano program"
            };
            println!("  runs {c} ({note})");
        }
        if other > 0 {
            println!("  {other} other Nano hooks written by hand (not managed by setup, left as they are)");
        }
    }
    let c = crate::controls::current(home);
    println!(
        "prompt text: {} (kept {} days; settings prompts and prompts.days)",
        c.get("prompts").map_or("keep", String::as_str),
        c.get("prompts.days").map_or("30", String::as_str)
    );
    let rows = crate::situation::read(&memory, None, None);
    let kept = std::fs::read_to_string(crate::hook::prompts_file(&memory)).map(|t| t.lines().filter(|l| !l.trim().is_empty()).count()).unwrap_or(0);
    let last = rows.last().and_then(|r| r.get("t").and_then(Value::integer).ok()).map(nmmusc::pack::utc_stamp);
    println!("situation record: {} records, last {}; prompts with kept text: {kept}", rows.len(), last.as_deref().unwrap_or("never"));
    let backups = std::fs::read_dir(connect::records_dir(home, repo).join("backups")).map(Iterator::count).unwrap_or(0);
    println!("setup backups: {backups} in {}", connect::records_dir(home, repo).join("backups").display());
    let agy_dir = agy_plugin_dir(repo);
    if agy_dir.join("plugin.json").is_file() {
        let hooks = if agy_dir.join("hooks.json").is_file() { "with hooks.json" } else { "MCP only" };
        println!("Antigravity plugin: present at {} ({hooks})", agy_dir.display());
    } else {
        println!("Antigravity plugin: none ({} not present)", agy_dir.display());
    }
    Ok(())
}

/// Rows returned by `/api/audit` when no `limit` is given, and the most it returns.
const AUDIT_ROWS: usize = 200;
const AUDIT_MAX: usize = 10_000;

/// The percent-decoded value of `key` in a URL query (`a=1&b=x%20y`).
#[must_use]
pub fn query_arg(query: &str, key: &str) -> Option<String> {
    let raw = query.split('&').find_map(|kv| kv.split_once('=').filter(|(k, _)| *k == key).map(|x| x.1))?;
    let b = raw.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while let Some(&c) = b.get(i) {
        let hex = b.get(i + 1..i + 3).and_then(|h| std::str::from_utf8(h).ok()).and_then(|h| u8::from_str_radix(h, 16).ok());
        match (c, hex) {
            (b'%', Some(x)) => {
                out.push(x);
                i += 3;
            }
            (b'+', _) => {
                out.push(b' ');
                i += 1;
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8(out).ok()
}

/// The newest `limit` rows of a JSON-lines log whose `kind` and `project` match the filters given.
/// Unfiltered, a line that does not parse is kept as text, as before.
/// A row written again under the same id (a tool call's running row, then its end) is one row:
/// at the place of its first writing, with the content of its last. A running row older than
/// `STALE_MS` with no end is shown as `stale`.
fn filter_rows(text: &str, kind: Option<&str>, project: Option<&str>, limit: usize) -> Vec<Value> {
    // 2026-11-26: a kind filter may name several kinds, comma separated (the Audit page's groups:
    // "library,library-base,library-seat,..."); a project filter names one project.
    let want = |v: &Value, k: &str, f: Option<&str>| {
        f.is_none_or(|f| v.get(k).and_then(Value::string).ok().is_some_and(|x| if k == "kind" { f.split(',').any(|w| w.trim() == x) } else { x == f }))
    };
    let now = nmtype::now_ms();
    let mut all: Vec<Result<Value, &str>> = Vec::new();
    let mut at: BTreeMap<String, usize> = BTreeMap::new();
    for l in text.lines().filter(|l| !l.trim().is_empty()) {
        match nmjson::parse(l.as_bytes()) {
            Ok(v) => {
                let id = v.get("id").and_then(Value::string).ok().map(str::to_owned);
                match id.as_ref().and_then(|i| at.get(i).copied()) {
                    Some(i) => {
                        if let Some(slot) = all.get_mut(i) {
                            *slot = Ok(v);
                        }
                    }
                    None => {
                        if let Some(i) = id {
                            at.insert(i, all.len());
                        }
                        all.push(Ok(v));
                    }
                }
            }
            Err(_) => all.push(Err(l)),
        }
    }
    let mut rows: Vec<Value> = all
        .into_iter()
        .rev()
        .filter_map(|r| match r {
            Ok(v) => (want(&v, "kind", kind) && want(&v, "project", project)).then_some(v),
            Err(l) => (kind.is_none() && project.is_none()).then(|| s(l)),
        })
        .take(limit)
        .map(|v| {
            let old = v.get("time_ms").and_then(Value::integer).ok().is_some_and(|t| now.saturating_sub(t) >= STALE_MS);
            match v {
                Value::Object(mut o) if old && o.get("state").and_then(|x| x.string().ok()) == Some("running") => {
                    o.insert("state".into(), s("stale"));
                    Value::Object(o)
                }
                other => other,
            }
        })
        .collect();
    rows.reverse();
    rows
}

/// The newest `limit` rows of a JSON-lines log matching the filters (`filter_rows`), read from the
/// end of the file (2026-11-26: the audit log is tens of MB and the interface polls it every
/// second from two places; reading it whole cost about 0.5 s of processor time a call and held
/// hundreds of MB). The read starts with the last few kilobytes per row wanted and grows four times
/// at a time until `limit` rows match or the whole file is read; a partial first line is dropped.
/// A row written again under one id keeps its newest content; its place is its first writing inside
/// what was read.
fn tail_rows(path: &Path, kind: Option<&str>, project: Option<&str>, limit: usize) -> Vec<Value> {
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut f) = std::fs::File::open(path) else { return Vec::new() };
    let len = f.metadata().map_or(0, |m| m.len());
    let mut want = u64::try_from(limit).unwrap_or(u64::MAX).saturating_mul(4096).max(256 * 1024);
    loop {
        let start = len.saturating_sub(want);
        let mut buf = Vec::new();
        if f.seek(SeekFrom::Start(start)).is_err() || f.read_to_end(&mut buf).is_err() {
            return Vec::new();
        }
        let text = String::from_utf8_lossy(&buf);
        let body = if start > 0 { text.split_once('\n').map_or("", |(_, rest)| rest) } else { &text };
        let rows = filter_rows(body, kind, project, limit);
        if rows.len() >= limit || start == 0 {
            return rows;
        }
        want = want.saturating_mul(4);
    }
}

/// `/api/audit[?kind=<kind>][&project=<id>][&limit=<n>]`: the `audit` rows matching the filters
/// and the `episodes` of that project (the Nano's own when none is named), newest `limit` of each.
#[must_use]
pub fn audit_view(home: &Path, query: &str) -> Value {
    let kind = query_arg(query, "kind").filter(|k| !k.is_empty());
    let project = query_arg(query, "project").filter(|p| !p.is_empty());
    let limit = query_arg(query, "limit").and_then(|l| l.parse::<usize>().ok()).unwrap_or(AUDIT_ROWS).min(AUDIT_MAX);
    let audit = tail_rows(&home.join("memory").join("audit.jsonl"), kind.as_deref(), project.as_deref(), limit);
    // A project id names a folder under memory/projects; anything else that could climb out is refused.
    let safe = project
        .as_deref()
        .filter(|p| *p != "nano" && !p.contains("..") && p.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)));
    let episodes = match safe {
        Some(p) => tail_rows(&nmtype::project::projects_dir(home).join(p).join("episodes.jsonl"), None, None, limit),
        None if project.is_some() && project.as_deref() != Some("nano") => Vec::new(),
        None => tail_rows(&home.join("memory").join("episodes.jsonl"), None, None, limit),
    };
    let mut o = BTreeMap::new();
    o.insert("audit".to_owned(), Value::Array(audit));
    o.insert("episodes".to_owned(), Value::Array(episodes));
    Value::Object(o)
}

/// Section check (processor only): the `.mcp.json` entry runs the launcher with `launch serve
/// --root <repo>` (merging is checked in `connect`); the audit filters, query decoding, the sources
/// of a text reply and the parts of a tool.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let args_of = |e: &Value| -> Vec<String> {
        e.get("args").and_then(Value::array).unwrap_or_default().iter().filter_map(|a| a.string().ok().map(str::to_owned)).collect()
    };
    let as_value = |o: &crate::jsonc::Out| {
        let st = crate::jsonc::Style { nl: "\n", unit: "  ".into() };
        nmjson::parse(crate::jsonc::pretty(o, "", &st).as_bytes()).unwrap_or(Value::Null)
    };
    let plain = as_value(&crate::connect::mcp_entry(Path::new("/x/nano/bin/neuromorph-nano.exe"), Path::new("/w/shop"), true));
    if args_of(&plain) != ["launch", "serve", "--root", "/w/shop", "--flow", "off"] {
        return Err(format!("project.setup: plain entry {}", plain.encode().unwrap_or_default()));
    }
    let e = as_value(&crate::connect::mcp_entry(Path::new("/x/nano/bin/neuromorph-nano.exe"), Path::new("/w/shop"), false));
    if args_of(&e) != ["launch", "serve", "--root", "/w/shop", "--flow"]
        || !e.get("command").and_then(Value::string).is_ok_and(|c| c.contains("bin"))
    {
        return Err(format!("project.setup: entry {}", e.encode().unwrap_or_default()));
    }
    let agy_default = nmjson::parse(agy_mcp_text(Path::new("/x/nano/bin/neuromorph-nano.exe"), Path::new("/w/shop"), false).as_bytes())
        .unwrap_or(Value::Null);
    let agy_default_args = agy_default.get("mcpServers").and_then(|s| s.get(SERVER)).map(args_of).unwrap_or_default();
    if agy_default_args != ["launch", "serve", "--root", "/w/shop", "--flow"] {
        return Err(format!("project.setup: agy default entry {agy_default_args:?}"));
    }
    let agy_plain = nmjson::parse(agy_mcp_text(Path::new("/x/nano/bin/neuromorph-nano.exe"), Path::new("/w/shop"), true).as_bytes())
        .unwrap_or(Value::Null);
    let agy_plain_args = agy_plain.get("mcpServers").and_then(|s| s.get(SERVER)).map(args_of).unwrap_or_default();
    if agy_plain_args != ["launch", "serve", "--root", "/w/shop", "--flow", "off"] {
        return Err(format!("project.setup: agy plain entry {agy_plain_args:?}"));
    }
    // Audit filters: kind and project select rows, limit keeps the newest, order is kept.
    let log = "{\"id\":\"a1\",\"kind\":\"setting\"}
{\"id\":\"a2\",\"kind\":\"tool\",\"project\":\"shop-1\"}
{\"id\":\"a3\",\"kind\":\"tool\",\"project\":\"nano\"}
{\"id\":\"a4\",\"kind\":\"outcome\",\"project\":\"shop-1\"}
";
    let ids =
        |v: Vec<Value>| v.iter().filter_map(|r| r.get("id").and_then(Value::string).ok().map(str::to_owned)).collect::<Vec<_>>().join(",");
    let all = ids(filter_rows(log, None, None, 200));
    let settings = ids(filter_rows(log, Some("setting"), None, 200));
    let shop = ids(filter_rows(log, None, Some("shop-1"), 200));
    let last = ids(filter_rows(log, None, None, 2));
    if all != "a1,a2,a3,a4" || settings != "a1" || shop != "a2,a4" || last != "a3,a4" {
        return Err(format!("project.audit: filters gave [{all}] [{settings}] [{shop}] [{last}]"));
    }
    // A running row written again at its end is one row, in its first place, with its last
    // content; a running row with no end older than the stale limit is shown as stale.
    let calls = "{\"id\":\"t1\",\"kind\":\"tool\",\"time_ms\":5,\"state\":\"running\"}
{\"id\":\"s1\",\"kind\":\"setting\",\"time_ms\":6}
{\"id\":\"t1\",\"kind\":\"tool\",\"time_ms\":5,\"state\":\"done\",\"steps\":[]}
{\"id\":\"t2\",\"kind\":\"tool\",\"time_ms\":7,\"state\":\"running\"}
";
    let got = filter_rows(calls, None, None, 200);
    let states: Vec<String> = got
        .iter()
        .map(|r| format!("{}:{}", r.get("id").and_then(Value::string).unwrap_or(""), r.get("state").and_then(Value::string).unwrap_or("-")))
        .collect();
    if states != ["t1:done", "s1:-", "t2:stale"] {
        return Err(format!("project.audit: running rows collapse to {states:?}"));
    }
    let noted = vec![
        nmapi::tools::Step { part: "b".into(), start_ms: Some(1010), ms: Some(5), us: Some(5300), ..nmapi::tools::Step::default() },
        nmapi::tools::Step { part: "a".into(), start_ms: Some(1000), ms: Some(9), ..nmapi::tools::Step::default() },
        nmapi::tools::Step { part: "u".into(), ..nmapi::tools::Step::default() },
    ];
    // An untimed step says so (`ms` null, `timed` false); it is never shown as 0 ms.
    let order = steps_value(noted, 1000, "nm_context_pack", &Value::Null, 20).encode().unwrap_or_default();
    if order != r#"[{"ms":9,"part":"a","start_ms":0,"timed":true},{"ms":5,"part":"b","start_ms":10,"timed":true,"us":5300},{"ms":null,"part":"u","timed":false}]"# {
        return Err(format!("project.audit: steps in run order gave {order}"));
    }
    let cells = vec![nmapi::tools::Step {
        part: "nmmem.gate#m0".into(),
        start_ms: Some(3),
        ms: Some(0),
        us: Some(410),
        inside: Some("nmapi.whole#assembly"),
        cells: vec!["read:episodes".into(), "read:active".into()],
        counts: vec![("episodes", 1)],
        ..nmapi::tools::Step::default()
    }];
    let cv = steps_value(cells, 0, "nm_context_pack", &Value::Null, 1).encode().unwrap_or_default();
    if cv != r#"[{"cells":["read:episodes","read:active"],"counts":{"episodes":1},"inside":"nmapi.whole#assembly","ms":0,"part":"nmmem.gate#m0","start_ms":3,"timed":true,"us":410}]"# {
        return Err(format!("project.audit: a step with cells and counts gave {cv}"));
    }
    // A tentacle step names the job and lens it carried; a step without them names neither.
    let t = vec![nmapi::tools::Step {
        part: "nmpod.unit#t2".into(),
        job: Some("dependency"),
        lens: Some("structure".into()),
        ..nmapi::tools::Step::default()
    }];
    let tv = steps_value(t, 0, "nm_context_pack", &Value::Null, 1).encode().unwrap_or_default();
    if tv != r#"[{"job":"dependency","lens":"structure","ms":null,"part":"nmpod.unit#t2","timed":false}]"# {
        return Err(format!("project.audit: a tentacle step with its job gave {tv}"));
    }
    if query_arg("kind=tool&project=my%20shop+x&limit=5", "project").as_deref() != Some("my shop x")
        || query_arg("kind=tool", "limit").is_some()
    {
        return Err("project.audit: query decoding".into());
    }
    // Quoted lines that start like headers (a diff hunk, a struck line) stay body lines.
    let pack = "nm-pack v1 cited 3 of 3 packets
answer: x
@@ src/a.rs:3-4
fn a() {}
}
~~ src/b.rs:9 symbol
fn b defined at src/b.rs:9
@@ docs/c.md:1-3
@@ -1,2 +1,2 @@
~~ old line
end
~~ src/d.rs:2 note (2 lines)
~~ first
@@ second
";
    if text_sources(pack).iter().filter_map(|v| v.string().ok()).collect::<Vec<_>>()
        != ["src/a.rs:3-4", "src/b.rs:9", "docs/c.md:1-3", "src/d.rs:2"]
    {
        return Err("project.audit: sources of a text reply".into());
    }
    if parts_of("nm_verify", &Value::Null) != ["nmapi.tools", "nmtool.verify"] {
        return Err("project.audit: parts of a tool".into());
    }
    // A note holding a secret is named in the audit row by the reason only, never by its text.
    let secret = nmjson::parse(br#"{"op":"store","text":"the key is sk-ABCdef1234567890abcdefGHIJ"}"#).map_err(|x| x.to_string())?;
    match main_arg(&secret) {
        Some(("text", t)) if t.starts_with("[left out:") && !t.contains("ABCdef") => {}
        other => return Err(format!("project.audit: secret argument logged as {other:?}")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn project_check_passes_with_flow_entries() {
        super::check().unwrap();
    }

    /// 2026-11-26: reading the log from its end gives the rows reading it whole gives: every
    /// limit, with and without the kind filter, a row written again under its id (running, then
    /// its end) kept once with its last content, and a log far larger than the first read.
    /// 2026-11-26: a kind filter naming several kinds keeps the rows of each and no other.
    #[test]
    fn kind_filter_takes_a_comma_list() {
        let log = "{\"id\":\"a1\",\"kind\":\"tool\",\"time_ms\":1}\n{\"id\":\"a2\",\"kind\":\"library-seat\",\"time_ms\":2}\n{\"id\":\"a3\",\"kind\":\"setting\",\"time_ms\":3}\n";
        let ids = |kind: &str| super::filter_rows(log, Some(kind), None, 10).iter().filter_map(|v| v.get("id").and_then(nmjson::Value::string).ok().map(str::to_owned)).collect::<Vec<_>>();
        assert_eq!(ids("library,library-seat,setting"), vec!["a2", "a3"]);
        assert_eq!(ids("tool"), vec!["a1"]);
        assert!(ids("library").is_empty(), "a kind is matched whole, never as a prefix");
    }

    #[test]
    fn tail_rows_match_the_whole_log() {
        use std::fmt::Write as _;
        let dir = std::env::temp_dir().join(format!("nano-tail-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("audit.jsonl");
        let mut log = String::new();
        for i in 0..6000_u64 {
            let kind = if i % 3 == 0 { "tool" } else { "setting" };
            let pad = "x".repeat(200);
            writeln!(log, "{{\"id\":\"a{i}\",\"kind\":\"{kind}\",\"time_ms\":{i},\"pad\":\"{pad}\"}}").unwrap();
            if i % 50 == 0 {
                writeln!(log, "{{\"id\":\"a{i}\",\"kind\":\"{kind}\",\"time_ms\":{i},\"state\":\"done\"}}").unwrap();
            }
        }
        std::fs::write(&path, &log).unwrap();
        for limit in [1_usize, 14, 20, 300, 1000, 10_000] {
            for kind in [None, Some("tool")] {
                let whole = super::filter_rows(&log, kind, None, limit);
                let tail = super::tail_rows(&path, kind, None, limit);
                assert_eq!(tail, whole, "limit {limit} kind {kind:?}");
            }
        }
        assert!(super::tail_rows(&dir.join("absent.jsonl"), None, None, 20).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
