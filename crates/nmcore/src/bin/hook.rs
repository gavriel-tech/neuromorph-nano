//! Hook intake: `neuromorph-nano hook <Event>` reads one Claude Code hook's JSON on standard input
//! and appends one compact record to the project's situation record,
//! `<project memory>/situation.jsonl`. It always exits 0, so it can never fail Claude Code
//! (and it blocks only for a loop, below); a fault is logged in the Nano's own audit (`kind: "hook-fault"`). It prints nothing,
//! except a prime (`prime.rs`) and the reminders and step nudges the user or the agent set
//! (`guidance.rs`): one JSON object carrying `additionalContext`, when a rule fires. The one
//! exception to never blocking is a loop the user or the agent set (`guidance.rs`): at Stop, while
//! its condition is unmet and it is under its cap, the reply is `decision: block` with the loop's
//! prompt as the reason, which continues the agent; the `loops` setting switches it off.
//! A record holds what later steps need and no more: time, session, turn, event, tool, the files
//! and line ranges touched, pass or fail with the exit code, a short digest of long output, and,
//! when the `prompts` control is `keep`, the prompt text after secret-dropping (kept apart in
//! `prompts.jsonl`, so it can be cleared on its own). File contents and full outputs are never kept.
use crate::secrets::drop_secrets;
use nmjson::Value;
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;

/// Most bytes read from standard input; larger input is recorded as oversize.
const MAX_INPUT: u64 = 8 << 20;
/// Most bytes of a file read to find an edit's lines.
const MAX_FILE: u64 = 4 << 20;
/// Longest command kept, after secret-dropping.
const CMD_CHARS: usize = 200;
/// Longest message line kept (a digest's first and last lines, an error message).
const MSG_CHARS: usize = 120;
/// Longest prompt text kept.
const PROMPT_CHARS: usize = 8000;
/// Most paths in one record.
const MAX_PATHS: usize = 16;
/// Most errors in one record.
const MAX_ERRORS: usize = 5;

/// The situation record's file in a project memory folder.
#[must_use]
pub fn situation_file(memory: &Path) -> PathBuf {
    memory.join("situation.jsonl")
}
/// The kept prompt text's file in a project memory folder.
#[must_use]
pub fn prompts_file(memory: &Path) -> PathBuf {
    memory.join("prompts.jsonl")
}

/// Runs the intake for `args[0]` (the event name). Always exits 0.
pub fn run(args: &[String]) -> ExitCode {
    std::panic::set_hook(Box::new(|_| {}));
    let start = std::time::Instant::now();
    let event = args.first().cloned().unwrap_or_default();
    let home = nmtype::find_root().ok().map(|h| nmsubs::repo::canon(&h));
    let r = std::panic::catch_unwind(|| intake(&event, home.as_deref(), start));
    let fault = match r {
        Ok(Ok(())) => None,
        Ok(Err(e)) => Some(e),
        Err(_) => Some("the intake stopped unexpectedly (panic)".to_owned()),
    };
    if let (Some(e), Some(h)) = (fault, home.as_deref()) {
        let _ = crate::controls::audit(
            h,
            "hook-fault",
            vec![("event", s(&event)), ("label", s(&format!("Hook fault ({event}): {}", short(&e, 80)))), ("error", s(&short(&e, 300)))],
        );
    }
    ExitCode::SUCCESS
}

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}
fn short(t: &str, max: usize) -> String {
    let one = t.split_whitespace().collect::<Vec<_>>().join(" ");
    if one.chars().count() <= max { one } else { format!("{}...", one.chars().take(max).collect::<String>()) }
}
/// Secret-dropped first, then cut, so a cut never leaves part of a secret.
fn clean(t: &str, max: usize) -> String {
    short(&drop_secrets(t), max)
}
fn sha16(t: &str) -> String {
    nmhash::sha256_bytes(t.as_bytes()).chars().take(16).collect()
}

fn intake(event: &str, home: Option<&Path>, start: std::time::Instant) -> Result<(), String> {
    let home = home.ok_or("Nano folder not found")?;
    let mut raw = Vec::new();
    std::io::stdin().take(MAX_INPUT + 1).read_to_end(&mut raw).map_err(|e| format!("standard input: {e}"))?;
    let oversize = raw.len() as u64 > MAX_INPUT;
    let parsed = if oversize { Value::Object(BTreeMap::new()) } else { nmjson::parse(&raw).map_err(|e| format!("input is not JSON: {e}"))? };
    let agy = is_agy(&parsed);
    let agy_event = text(&parsed, "hookEventName").unwrap_or(event).to_owned();
    let controls = crate::controls::current(home);
    if controls.get("memory").is_some_and(|v| v == "off") {
        if agy {
            write_stdout(&agy_reply(&agy_event, None));
        }
        return Ok(());
    }
    // Antigravity fires PreInvocation before every model call within a turn (invocationNum 0, 1, 2...).
    // Only invocationNum == 0 (or omitted) starts the turn; mid-turn invocations are no-ops so
    // situation.jsonl keeps one prompt row per turn and ledger turn windows stay intact.
    if agy && agy_event == "PreInvocation" && parsed.get("invocationNum").and_then(Value::integer).is_ok_and(|k| k > 0) {
        write_stdout("{}");
        return Ok(());
    }
    let (norm_event, input) = if agy { normalize_agy(&agy_event, &parsed) } else { (event.to_owned(), parsed) };
    let root = project_root(&input, agy);
    let memory = nmtype::project::memory_dir(home, &root);
    let keep = controls.get("prompts").is_none_or(|v| v == "keep");
    let (mut rec, prompt) = record(&norm_event, &input, &root, keep);
    if agy {
        rec.insert("host".into(), s("antigravity"));
    }
    if oversize {
        rec.insert("oversize".into(), Value::Bool(true));
    }
    let ev_name = rec.get("event").and_then(|v| v.string().ok()).unwrap_or(&norm_event).to_owned();
    // A prime, when a rule fires and the budget allows it; it rides on this record. A fault or a
    // stop inside the prime rules costs the prime only, never the record. On Antigravity, only
    // SessionStart and PreInvocation accept injectSteps, so skip offering primes on other events
    // rather than recording or memoising a prime the host cannot receive.
    let mode = crate::prime::Mode::parse(controls.get("priming"));
    let can_prime = !agy || matches!(agy_event.as_str(), "SessionStart" | "PreInvocation");
    let given = if oversize || mode == crate::prime::Mode::Off || !can_prime {
        None
    } else {
        let ask = crate::prime::Ask { home, memory: &memory, root: &root, event: &ev_name, input: &input, rec: &rec, mode, start };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| crate::prime::offer(&ask))) {
            Ok(Ok(g)) => Some(g),
            Ok(Err(crate::prime::Silent::Late(kind))) => {
                rec.insert("prime_late".into(), s(kind));
                None
            }
            _ => None,
        }
    };
    if let Some(g) = &given {
        rec.insert("prime".into(), g.record_value(mode));
    }
    // Custom primes configured per project (event, interval, and schedule triggers).
    let custom_fired = if oversize || mode == crate::prime::Mode::Off || !can_prime {
        Vec::new()
    } else {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let ctx = crate::primes_ctl::HookCtx {
                home,
                memory: &memory,
                root: &root,
                event: &ev_name,
                rec: &rec,
                now_ms: nmtype::now_ms(),
                agy,
            };
            crate::primes_ctl::at_hook(&ctx)
        }))
        .unwrap_or_default()
    };
    if !custom_fired.is_empty() {
        rec.insert(
            "custom_primes".into(),
            Value::Array(custom_fired.iter().map(|(id, _, _)| s(id)).collect()),
        );
    }
    // Reminders, step nudges and loops the user or the agent set (`guidance.rs`); they ride with
    // the prime, never count against it, and a fault costs the guidance only.
    let guided = if oversize {
        None
    } else {
        let at = crate::guidance::At { home, memory: &memory, event: &ev_name, rec: &rec, agy, inject: can_prime, now: nmtype::now_ms() };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| crate::guidance::at_hook(&at))) {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => {
                rec.insert("guidance_fault".into(), s(&short(&e, 120)));
                None
            }
            Err(_) => None,
        }
    };
    if let Some(o) = &guided {
        rec.insert("guidance".into(), o.record.clone());
    }
    // The hook's own time to its record, from which the prime deadline is derived.
    rec.insert("hook_ms".into(), Value::Number(start.elapsed().as_millis().to_string()));
    // The whole hook, from the process's creation (its launch included) to this record, from
    // which setup derives the hook timeout; left out when the system cannot say.
    if let Ok(created) = nmffi::dynlib::process_started_ms() {
        rec.insert("whole_ms".into(), Value::Number(nmtype::now_ms().saturating_sub(created).to_string()));
    }
    let session = rec.get("session").and_then(|v| v.string().ok()).unwrap_or_default().to_owned();
    std::fs::create_dir_all(&memory).map_err(|e| format!("{}: {e}", memory.display()))?;
    append(&situation_file(&memory), &Value::Object(rec))?;
    if let Some(g) = &given {
        crate::prime::remember(&memory, &session, g, &ev_name)?;
        let proj_key = memory.file_name().and_then(|x| x.to_str()).unwrap_or_default();
        let places = Value::Array(g.places.iter().map(|p| s(p)).collect());
        let label = format!("Prime ({}) at {} ({} {})", g.kind, ev_name, g.places.len(), if g.places.len() == 1 { "place" } else { "places" });
        let _ = crate::controls::audit(
            home,
            "prime",
            vec![
                ("project", s(proj_key)),
                ("prime_kind", s(g.kind)),
                ("event", s(&ev_name)),
                ("prime_id", s(&g.id)),
                ("places", places),
                ("chars", n(g.text.chars().count() as u64)),
                ("mode", s(mode.label())),
                ("via", s(if agy { "antigravity" } else { "claude-code" })),
                ("ok", Value::Bool(true)),
                ("label", s(&label)),
            ],
        );
    }
    let combined_prime: Option<String> = {
        let mut parts: Vec<&str> = Vec::new();
        if let Some(g) = &given {
            parts.push(g.text.as_str());
        }
        for (_, _, line) in &custom_fired {
            parts.push(line.as_str());
        }
        if parts.is_empty() { None } else { Some(parts.join("\n")) }
    };
    let prime_text = combined_prime.as_deref();
    if agy {
        write_stdout(&agy_reply(&agy_event, crate::guidance::agy_text(prime_text, guided.as_ref()).as_deref()));
    } else if guided.is_none()
        && let Some(text) = prime_text
    {
        write_stdout(&crate::prime::Given::reply_text(&ev_name, text));
    } else if let Some(line) = crate::guidance::reply(&ev_name, prime_text, guided.as_ref()) {
        write_stdout(&line);
    }
    if let Some(p) = prompt {
        append(&prompts_file(&memory), &p)?;
        // Expired text is looked for at most once an hour (a marker file's time, read without
        // opening it), so a prompt costs one append to each file and no more.
        let marker = memory.join("prompts.pruned");
        let due = std::fs::metadata(&marker).and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).is_none_or(|age| age.as_secs() >= 3600);
        if due {
            let days = controls.get("prompts.days").and_then(|d| d.parse::<u64>().ok()).unwrap_or(30);
            prune(&prompts_file(&memory), days)?;
            std::fs::write(&marker, b"Time of the last removal of expired prompt text.\n").map_err(|e| format!("{}: {e}", marker.display()))?;
        }
    }
    // At a stop or a session start or end, the evidence ledger is brought up to date.
    crate::ledger::after_hook(home, &memory, &root, &ev_name)
}

fn write_stdout(line: &str) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
}

/// Whether `input` is an Antigravity lifecycle hook payload (camelCase keys: `hookEventName`,
/// `conversationId`, or `toolCall`).
fn is_agy(input: &Value) -> bool {
    input.get("hookEventName").is_ok() || input.get("conversationId").is_ok() || input.get("toolCall").is_ok()
}

/// Normalises an Antigravity camelCase hook payload into the canonical snake_case shape `record`
/// and `prime` expect, so situation tracking, the evidence ledger and priming work across both
/// Claude Code and Antigravity hosts.
fn normalize_agy(agy_event: &str, input: &Value) -> (String, Value) {
    let mut out = BTreeMap::new();
    let has_err = text(input, "error").is_some_and(|e| !e.trim().is_empty());
    let first_inv = !input.get("invocationNum").and_then(Value::integer).is_ok_and(|k| k > 0);
    let ev = match agy_event {
        "PreInvocation" if first_inv => "UserPromptSubmit",
        "PostToolUse" if has_err => "PostToolUseFailure",
        other => other,
    };
    out.insert("hook_event_name".into(), s(ev));
    if agy_event == "SessionStart" {
        out.insert("source".into(), s(text(input, "source").unwrap_or("startup")));
    }
    if let Some(id) = text(input, "conversationId") {
        out.insert("session_id".into(), s(id));
    }
    if let Some(m) = text(input, "modelName") {
        out.insert("model".into(), s(m));
    }
    if let Ok(step) = input.get("stepIdx").and_then(Value::integer) {
        out.insert("prompt_id".into(), s(&format!("step-{step}")));
    } else if let Ok(init) = input.get("initialNumSteps").and_then(Value::integer) {
        out.insert("prompt_id".into(), s(&format!("turn-{init}")));
    } else if let Ok(inv) = input.get("invocationNum").and_then(Value::integer) {
        out.insert("prompt_id".into(), s(&format!("inv-{inv}")));
    }
    let cwd = text(input, "cwd").or_else(|| input.get("workspacePaths").ok().and_then(|v| v.array().ok()).and_then(|a| a.first()).and_then(|v| v.string().ok()));
    if let Some(c) = cwd {
        out.insert("cwd".into(), s(c));
    }
    if let Some(p) = text(input, "lastUserInput") {
        out.insert("prompt".into(), s(p));
    }
    if matches!(input.get("continuationActive"), Ok(Value::Bool(true))) {
        out.insert("stop_hook_active".into(), Value::Bool(true));
    }
    if has_err && let Some(e) = text(input, "error") {
        out.insert("error".into(), s(e));
    }
    if let Ok(tc) = input.get("toolCall") {
        if let Some(id) = text(tc, "id") {
            out.insert("tool_use_id".into(), s(id));
        }
        let raw_name = text(tc, "name").unwrap_or_default();
        let args = tc.get("args").cloned().unwrap_or(Value::Null);
        let (tool_name, tool_input) = normalize_agy_tool(raw_name, &args);
        out.insert("tool_name".into(), s(&tool_name));
        out.insert("tool_input".into(), tool_input);
    }
    (ev.to_owned(), Value::Object(out))
}

fn unwrap_agy_mcp_args(args: &Value) -> Value {
    match args.get("Arguments").or_else(|_| args.get("arguments")) {
        Ok(Value::Object(m)) => Value::Object(m.clone()),
        Ok(Value::String(raw)) => nmjson::parse(raw.as_bytes()).unwrap_or(Value::Null),
        _ => Value::Null,
    }
}

fn normalize_agy_tool(name: &str, args: &Value) -> (String, Value) {
    let mut ti = BTreeMap::new();
    let copy_str = |dst: &mut BTreeMap<String, Value>, to: &str, keys: &[&str]| {
        if let Some(v) = keys.iter().find_map(|k| text(args, k)) {
            dst.insert(to.into(), s(v));
        }
    };
    let low = name.to_ascii_lowercase();
    if low == "call_mcp_tool" {
        let server = ["ServerName", "serverName", "server_name"].iter().find_map(|k| text(args, k)).unwrap_or_default().trim_matches('"');
        let mcp_tool = ["ToolName", "toolName", "tool_name"].iter().find_map(|k| text(args, k)).unwrap_or_default().trim_matches('"');
        let inner = unwrap_agy_mcp_args(args);
        if !mcp_tool.is_empty() {
            let s_low = server.to_ascii_lowercase();
            let t_low = mcp_tool.to_ascii_lowercase();
            if s_low.contains("neuromorph-nano") || t_low.starts_with("nm_") {
                return (format!("mcp__neuromorph-nano__{mcp_tool}"), inner);
            }
            return (format!("mcp__{server}__{mcp_tool}"), inner);
        }
    }
    if low.contains("nm_") || low.contains("neuromorph-nano") {
        let short_tool = name.rsplit([':', '/', '.']).next().unwrap_or(name);
        let canonical = if short_tool.starts_with("mcp__") { short_tool.to_owned() } else { format!("mcp__neuromorph-nano__{short_tool}") };
        return (canonical, args.clone());
    }
    match low.as_str() {
        "read_file" | "view_file" | "view_file_outline" | "view_code_item" => {
            copy_str(&mut ti, "file_path", &["Path", "AbsolutePath", "File", "TargetFile", "file_path"]);
            if let Ok(a) = args.get("StartLine").and_then(Value::integer) {
                ti.insert("offset".into(), n(a));
                if let Ok(b) = args.get("EndLine").and_then(Value::integer) {
                    ti.insert("limit".into(), n(b.saturating_sub(a).saturating_add(1)));
                }
            }
            ("Read".into(), Value::Object(ti))
        }
        "edit_file" | "replace_file_content" => {
            copy_str(&mut ti, "file_path", &["Path", "TargetFile", "file_path"]);
            copy_str(&mut ti, "old_string", &["Target", "TargetContent", "old_string"]);
            copy_str(&mut ti, "new_string", &["Replacement", "ReplacementContent", "new_string"]);
            ("Edit".into(), Value::Object(ti))
        }
        "multi_replace_file_content" => {
            copy_str(&mut ti, "file_path", &["Path", "TargetFile", "file_path"]);
            if let Ok(chunks) = args.get("ReplacementChunks").and_then(Value::array) {
                let edits: Vec<Value> = chunks
                    .iter()
                    .map(|c| {
                        let mut e = BTreeMap::new();
                        if let Some(o) = text(c, "TargetContent") {
                            e.insert("old_string".into(), s(o));
                        }
                        if let Some(r) = text(c, "ReplacementContent") {
                            e.insert("new_string".into(), s(r));
                        }
                        Value::Object(e)
                    })
                    .collect();
                if !edits.is_empty() {
                    ti.insert("edits".into(), Value::Array(edits));
                }
            }
            ("MultiEdit".into(), Value::Object(ti))
        }
        "write_file" | "write_to_file" => {
            copy_str(&mut ti, "file_path", &["Path", "TargetFile", "file_path"]);
            copy_str(&mut ti, "content", &["Content", "CodeContent", "content"]);
            ("Write".into(), Value::Object(ti))
        }
        "run_command" => {
            copy_str(&mut ti, "command", &["CommandLine", "command"]);
            ("PowerShell".into(), Value::Object(ti))
        }
        "grep_search" => {
            copy_str(&mut ti, "pattern", &["Query", "pattern"]);
            copy_str(&mut ti, "path", &["SearchPath", "path"]);
            ("Grep".into(), Value::Object(ti))
        }
        "find_by_name" | "list_dir" => {
            copy_str(&mut ti, "pattern", &["Pattern", "pattern"]);
            copy_str(&mut ti, "path", &["SearchDirectory", "DirectoryPath", "path"]);
            ("Glob".into(), Value::Object(ti))
        }
        _ => (name.to_owned(), args.clone()),
    }
}

/// Antigravity JSON stdout reply contract: SessionStart and PreInvocation carry any prime and
/// reminders inside `injectSteps[0].systemMessage.systemMessage`, and other events return `{}`.
/// Passive hooks never emit `decision: "allow"` so they cannot bypass the host's permission
/// prompts, and no loop continues the agent there (`guidance.rs`).
fn agy_reply(agy_event: &str, given: Option<&str>) -> String {
    match agy_event {
        "SessionStart" | "PreInvocation" => match given {
            Some(t) => {
                let inner = Value::Object([("systemMessage".to_owned(), s(t))].into_iter().collect());
                let step = Value::Object([("systemMessage".to_owned(), inner)].into_iter().collect());
                Value::Object([("injectSteps".to_owned(), Value::Array(vec![step]))].into_iter().collect())
                    .encode()
                    .unwrap_or_else(|_| "{}".to_owned())
            }
            None => "{}".to_owned(),
        },
        _ => "{}".to_owned(),
    }
}

/// Appends one line with a single write, so lines from hooks running at once stay whole.
fn append(file: &Path, v: &Value) -> Result<(), String> {
    let mut line = v.encode().map_err(|e| e.to_string())?;
    line.push('\n');
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(file).map_err(|e| format!("{}: {e}", file.display()))?;
    f.write_all(line.as_bytes()).map_err(|e| format!("{}: {e}", file.display()))
}

/// Removes kept prompt text older than `days`. Lines are in time order, so only the first line is
/// looked at unless it has expired.
/// # Errors
/// The file cannot be rewritten.
pub fn prune(file: &Path, days: u64) -> Result<usize, String> {
    let cutoff = nmtype::now_ms().saturating_sub(days.saturating_mul(86_400_000));
    let t_of = |l: &str| nmjson::parse(l.as_bytes()).ok().and_then(|v| v.get("t").and_then(Value::integer).ok()).unwrap_or(0);
    let Ok(f) = std::fs::File::open(file) else { return Ok(0) };
    let mut first = String::new();
    let _ = std::io::BufRead::read_line(&mut std::io::BufReader::new(f), &mut first);
    if first.trim().is_empty() || t_of(&first) >= cutoff {
        return Ok(0);
    }
    let text = std::fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?;
    let keep: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty() && t_of(l) >= cutoff).collect();
    let gone = text.lines().filter(|l| !l.trim().is_empty()).count() - keep.len();
    let tmp = file.with_extension("jsonl.tmp");
    let body = if keep.is_empty() { String::new() } else { format!("{}\n", keep.join("\n")) };
    std::fs::write(&tmp, body).and_then(|()| std::fs::rename(&tmp, file)).map_err(|e| format!("{}: {e}", file.display()))?;
    Ok(gone)
}

/// The project: for Antigravity, the payload's `cwd` (from `cwd` or `workspacePaths[0]`), else
/// this process's folder; for Claude Code, `CLAUDE_PROJECT_DIR` when set, else `cwd`, else this
/// process's folder.
fn project_root(input: &Value, agy: bool) -> PathBuf {
    let env = (!agy).then(|| std::env::var_os("CLAUDE_PROJECT_DIR").map(PathBuf::from).filter(|p| p.is_dir())).flatten();
    let cwd = input.get("cwd").and_then(Value::string).ok().map(PathBuf::from).filter(|p| p.is_dir());
    let p = env.or(cwd).or_else(|| std::env::current_dir().ok()).unwrap_or_default();
    nmsubs::repo::canon(&p)
}

fn text<'a>(v: &'a Value, k: &str) -> Option<&'a str> {
    v.get(k).and_then(Value::string).ok()
}

/// Builds the record of one hook input, and the prompt text row when it is kept.
fn record(event: &str, input: &Value, root: &Path, keep: bool) -> (BTreeMap<String, Value>, Option<Value>) {
    let mut r = BTreeMap::new();
    let now = nmtype::now_ms();
    r.insert("v".into(), n(1));
    r.insert("t".into(), n(now));
    let ev = text(input, "hook_event_name").unwrap_or(event);
    r.insert("event".into(), s(ev));
    for (k, from) in [("session", "session_id"), ("turn", "prompt_id")] {
        if let Some(x) = text(input, from) {
            r.insert(k.into(), s(&short(x, 80)));
        }
    }
    if let Some(a) = text(input, "agent_type") {
        let id = text(input, "agent_id").map(|i| format!(" {}", short(i, 40))).unwrap_or_default();
        r.insert("agent".into(), s(&format!("{}{id}", short(a, 40))));
    }
    let cwd = text(input, "cwd").map_or_else(|| root.to_path_buf(), PathBuf::from);
    let mut prompt_row = None;
    match ev {
        "SessionStart" => {
            r.insert("kind".into(), s("session"));
            if let Some(x) = text(input, "source") {
                r.insert("source".into(), s(&short(x, 20)));
            }
            if let Some(m) = text(input, "model") {
                r.insert("model".into(), s(&short(m, 60)));
            }
        }
        "UserPromptSubmit" => {
            r.insert("kind".into(), s("prompt"));
            let p = text(input, "prompt").unwrap_or_default();
            let dropped = crate::secrets::kinds(p).len();
            let safe = drop_secrets(p);
            let safe: String = safe.chars().take(PROMPT_CHARS).collect();
            let mut o = BTreeMap::new();
            o.insert("chars".into(), n(p.chars().count() as u64));
            o.insert("sha".into(), s(&sha16(&safe)));
            o.insert("kept".into(), Value::Bool(keep && !safe.trim().is_empty()));
            if dropped > 0 {
                o.insert("secrets_dropped".into(), n(dropped as u64));
            }
            r.insert("prompt".into(), Value::Object(o));
            if keep && !safe.trim().is_empty() {
                let mut row = BTreeMap::new();
                row.insert("t".into(), n(now));
                row.insert("sha".into(), s(&sha16(&safe)));
                for k in ["session", "turn"] {
                    if let Some(x) = r.get(k) {
                        row.insert(k.into(), x.clone());
                    }
                }
                row.insert("text".into(), s(&safe));
                prompt_row = Some(Value::Object(row));
            }
        }
        "PreToolUse" | "PostToolUse" | "PostToolUseFailure" => tool(ev, input, root, &cwd, &mut r),
        "Stop" | "SubagentStop" => {
            r.insert("kind".into(), s(if ev == "Stop" { "stop" } else { "subagent" }));
            if matches!(input.get("stop_hook_active"), Ok(Value::Bool(true))) {
                r.insert("again".into(), Value::Bool(true));
            }
        }
        "PreCompact" | "PostCompact" => {
            r.insert("kind".into(), s("compact"));
            if let Some(x) = text(input, "trigger") {
                r.insert("trigger".into(), s(&short(x, 20)));
            }
            if let Some(x) = text(input, "custom_instructions").filter(|x| !x.is_empty()) {
                r.insert("instructions_chars".into(), n(x.chars().count() as u64));
            }
        }
        _ => {
            r.insert("kind".into(), s("other"));
            if let Some(x) = text(input, "reason") {
                r.insert("reason".into(), s(&clean(x, MSG_CHARS)));
            }
        }
    }
    (r, prompt_row)
}

/// The record of a tool event: tool, kind, paths and lines, command, outcome, digest.
fn tool(ev: &str, input: &Value, root: &Path, cwd: &Path, r: &mut BTreeMap<String, Value>) {
    let name = text(input, "tool_name").unwrap_or_default();
    let ti = input.get("tool_input").cloned().unwrap_or(Value::Null);
    let resp = input.get("tool_response").cloned().unwrap_or(Value::Null);
    r.insert("tool".into(), s(&short(name, 80)));
    if let Some(u) = text(input, "tool_use_id") {
        r.insert("use".into(), s(&short(u, 60)));
    }
    if let Ok(ms) = input.get("duration_ms").and_then(Value::integer) {
        r.insert("ms".into(), n(ms));
    }
    let mut paths: Vec<String> = Vec::new();
    let mut lines: Vec<Value> = Vec::new();
    let mut sensitive = false;
    let mut add_path = |p: &str, span: Option<(u64, u64)>, paths: &mut Vec<String>, lines: &mut Vec<Value>| {
        let rel = relative(root, cwd, p);
        sensitive |= is_sensitive(&rel) || nmsubs::repo::is_nanoignored(root, &rel);
        if paths.len() < MAX_PATHS && !paths.contains(&rel) {
            paths.push(rel.clone());
        }
        if let Some((a, b)) = span
            && lines.len() < MAX_PATHS
        {
            lines.push(Value::Array(vec![s(&rel), n(a), n(b)]));
        }
    };
    let file_path = text(&ti, "file_path").or_else(|| text(&ti, "notebook_path"));
    let kind = match name {
        "Read" => {
            if let Some(p) = file_path {
                add_path(p, read_span(&ti, &resp), &mut paths, &mut lines);
            }
            "read"
        }
        "Grep" | "Glob" => {
            if let Some(p) = text(&ti, "path") {
                add_path(p, None, &mut paths, &mut lines);
            }
            if let Some(q) = text(&ti, "pattern") {
                r.insert("query".into(), s(&clean(q, MSG_CHARS)));
            }
            "search"
        }
        "Edit" | "MultiEdit" | "NotebookEdit" | "Write" => {
            if let Some(p) = file_path {
                let span = edit_span(name, ev, &ti, root, cwd, p);
                add_path(p, span, &mut paths, &mut lines);
                if let Some((old, new)) = edit_size(name, &ti) {
                    r.insert("edit".into(), Value::Array(vec![n(old), n(new)]));
                }
            }
            if name == "Write" { "write" } else { "edit" }
        }
        "Bash" | "PowerShell" => {
            let cmd = text(&ti, "command").unwrap_or_default();
            r.insert("cmd".into(), s(&clean(cmd, CMD_CHARS)));
            let class = command_class(cmd);
            r.insert("cmd_class".into(), s(class.0));
            if let Some(p) = class.2 {
                r.insert("pkg".into(), s(&short(&p, 60)));
            }
            for (p, span) in shell_paths(cmd, root, cwd) {
                add_path(&p, span, &mut paths, &mut lines);
            }
            class.1
        }
        _ if name.contains("neuromorph-nano__") => {
            let mut o = BTreeMap::new();
            o.insert("tool".into(), s(name.rsplit("__").next().unwrap_or(name)));
            if let Some(q) = ["question", "query", "name", "crate", "package"].iter().find_map(|k| text(&ti, k)) {
                o.insert("ask".into(), s(&clean(q, CMD_CHARS)));
            }
            if let Some(e) = episode(&resp).or_else(|| text(&ti, "episode").map(str::to_owned)) {
                o.insert("episode".into(), s(&short(&e, 80)));
            }
            r.insert("nano".into(), Value::Object(o));
            "nano"
        }
        _ if name.starts_with("mcp__") => "mcp",
        "Agent" | "Task" => "subagent",
        _ => "other",
    };
    r.insert("kind".into(), s(kind));
    if !paths.is_empty() {
        r.insert("paths".into(), Value::Array(paths.iter().map(|p| s(p)).collect()));
    }
    if !lines.is_empty() {
        r.insert("lines".into(), Value::Array(lines));
    }
    if sensitive {
        r.insert("sensitive".into(), Value::Bool(true));
    }
    outcome(ev, name, kind, input, &resp, root, cwd, r);
}

/// Pass or fail, exit code, test counts, located errors and a digest of the output.
#[allow(clippy::too_many_arguments)]
fn outcome(ev: &str, name: &str, kind: &str, input: &Value, resp: &Value, root: &Path, cwd: &Path, r: &mut BTreeMap<String, Value>) {
    if ev == "PreToolUse" {
        return;
    }
    let shell = matches!(name, "Bash" | "PowerShell");
    let out = if ev == "PostToolUseFailure" { text(input, "error").unwrap_or_default().to_owned() } else { output_text(resp) };
    let interrupted = matches!(input.get("is_interrupt"), Ok(Value::Bool(true))) || matches!(resp.get("interrupted"), Ok(Value::Bool(true)));
    if interrupted {
        r.insert("interrupted".into(), Value::Bool(true));
    }
    let failed_marker = matches!(kind, "build" | "test") && failure_marker(&out);
    let ok = ev != "PostToolUseFailure" && !interrupted && !failed_marker;
    r.insert("ok".into(), Value::Bool(ok));
    let exit = if ev == "PostToolUseFailure" {
        out.lines()
            .next()
            .and_then(|l| {
                let t = l.trim();
                t.strip_prefix("Exit code ").or_else(|| t.strip_prefix("exit status "))
            })
            .and_then(|c| c.split_whitespace().next())
            .and_then(|c| c.trim_end_matches([':', ',', '.']).parse::<i64>().ok())
    } else if shell {
        ["exitCode", "exit_code", "returnCode", "code"].iter().find_map(|k| resp.get(k).ok().and_then(|v| match v {
            Value::Number(x) => x.parse::<i64>().ok(),
            _ => None,
        }))
        .or(Some(0))
    } else {
        None
    };
    if let Some(c) = exit {
        r.insert("exit".into(), Value::Number(c.to_string()));
    }
    if matches!(kind, "build" | "test") {
        if let Some((p, f)) = test_counts(&out) {
            r.insert("tests".into(), Value::Array(vec![n(p), n(f)]));
        }
        let errs = errors(&out, root, cwd);
        if !errs.is_empty() {
            r.insert("errors".into(), Value::Array(errs));
        }
    }
    if ev == "PostToolUseFailure"
        && !shell
        && let Some(l) = out.lines().find(|l| !l.trim().is_empty())
    {
        r.insert("msg".into(), s(&clean(l, MSG_CHARS)));
    }
    if (shell || ev == "PostToolUseFailure") && !out.is_empty() {
        r.insert("digest".into(), digest(&out));
    }
}

/// The output text of a tool response: Bash `stdout` and `stderr`, a plain string, or MCP text
/// content.
fn output_text(resp: &Value) -> String {
    match resp {
        Value::String(t) => t.clone(),
        Value::Object(_) => {
            let mut t = String::new();
            for k in ["stdout", "stderr", "output"] {
                if let Some(x) = text(resp, k).filter(|x| !x.is_empty()) {
                    if !t.is_empty() {
                        t.push('\n');
                    }
                    t.push_str(x);
                }
            }
            t
        }
        _ => String::new(),
    }
}

/// A short digest of an output: its size, a hash, its first and last non-empty lines.
fn digest(out: &str) -> Value {
    let mut o = BTreeMap::new();
    o.insert("chars".into(), n(out.chars().count() as u64));
    o.insert("lines".into(), n(out.lines().count() as u64));
    o.insert("sha".into(), s(&sha16(out)));
    let mut non_empty = out.lines().filter(|l| !l.trim().is_empty());
    let first = non_empty.next();
    let last = non_empty.next_back();
    if let Some(f) = first {
        o.insert("first".into(), s(&clean(f, MSG_CHARS)));
    }
    if let Some(l) = last {
        o.insert("last".into(), s(&clean(l, MSG_CHARS)));
    }
    Value::Object(o)
}

/// Signs of a failed build or test in an output that came back as a success (a pipe that ends in
/// a command that exits 0).
fn failure_marker(out: &str) -> bool {
    out.contains("error[E")
        || out.contains("error: could not compile")
        || out.contains("test result: FAILED")
        || out.contains("error: test failed")
        || out.contains("npm ERR!")
        || out.lines().any(|l| l.starts_with("FAIL") || l.starts_with("--- FAIL"))
        || test_counts(out).is_some_and(|(_, f)| f > 0)
}

/// Passed and failed counts from cargo (`test result:`), pytest (`N passed, M failed`) and jest
/// (`Tests: ...`) summaries.
fn test_counts(out: &str) -> Option<(u64, u64)> {
    let count = |l: &str, word: &str| -> Option<u64> {
        let i = l.find(word)?;
        l.get(..i)?.trim_end().rsplit(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
    };
    let mut found = false;
    let (mut p, mut f) = (0, 0);
    for l in out.lines() {
        let t = l.trim();
        if t.starts_with("test result:") || t.starts_with("Tests:") || (t.starts_with('=') && t.ends_with('=') && (t.contains(" passed") || t.contains(" failed"))) {
            found = true;
            p += count(t, " passed").unwrap_or(0);
            f += count(t, " failed").unwrap_or(0);
        }
    }
    found.then_some((p, f))
}

/// Located errors: rustc (`error[E...]: msg` then `--> file:line:col`, and the short form),
/// panics (`panicked at file:line:col`), and `file:line:` lines of Go, Python and JavaScript tools.
fn errors(out: &str, root: &Path, cwd: &Path) -> Vec<Value> {
    let mut found: Vec<(Option<String>, String, u64, String)> = Vec::new();
    let mut pending: Option<(Option<String>, String)> = None;
    let loc = |t: &str| -> Option<(String, u64)> {
        let mut it = t.trim().splitn(3, ':');
        let mut file = it.next()?.to_owned();
        let mut rest = it.next()?;
        // A Windows drive letter: `C:\x\y.rs:12:3`.
        if file.len() == 1 && rest.starts_with('\\') {
            let mut more = t.trim().get(2..)?.splitn(3, ':');
            file = format!("{}:{}", file, more.next()?);
            rest = more.next()?;
        }
        let line: u64 = rest.trim().parse().ok()?;
        (file.contains('.') && !file.contains(' ')).then_some((file, line))
    };
    for l in out.lines() {
        if found.len() >= MAX_ERRORS {
            break;
        }
        let t = l.trim();
        if let Some(rest) = t.strip_prefix("error") {
            if let Some(code_end) = rest.strip_prefix('[').and_then(|x| x.find(']')) {
                let code = rest.get(1..=code_end).unwrap_or_default().to_owned();
                let msg = rest.get(code_end + 2..).unwrap_or_default().trim_start_matches(':').trim().to_owned();
                pending = Some((Some(code), msg));
            } else if let Some(msg) = rest.strip_prefix(':') {
                pending = Some((None, msg.trim().to_owned()));
            }
            continue;
        }
        if let Some(at) = t.strip_prefix("--> ") {
            if let (Some((code, msg)), Some((file, line))) = (pending.take(), loc(at)) {
                found.push((code, file, line, msg));
            }
            continue;
        }
        if let Some(i) = t.find("panicked at ") {
            if let Some((file, line)) = t.get(i + 12..).and_then(|x| loc(x.trim_end_matches(':'))) {
                found.push((Some("panic".into()), file, line, String::new()));
            }
            continue;
        }
        if let Some(i) = t.find(": error")
            && let Some((file, line)) = t.get(..i).and_then(loc)
        {
            let rest = t.get(i + 2..).unwrap_or_default();
            let code = rest.strip_prefix("error[").and_then(|x| x.split(']').next()).map(str::to_owned);
            let msg = rest.split_once(": ").map_or("", |x| x.1).to_owned();
            found.push((code, file, line, msg));
        }
    }
    found.dedup_by(|a, b| a.1 == b.1 && a.2 == b.2);
    found
        .into_iter()
        .map(|(code, file, line, msg)| {
            let mut o = BTreeMap::new();
            if let Some(c) = code {
                o.insert("code".into(), s(&short(&c, 20)));
            }
            o.insert("file".into(), s(&relative(root, cwd, &file)));
            o.insert("line".into(), n(line));
            if !msg.is_empty() {
                o.insert("msg".into(), s(&clean(&msg, MSG_CHARS)));
            }
            Value::Object(o)
        })
        .collect()
}

/// A path as the project sees it: relative to the project root with forward slashes, or
/// `<outside>` for a path outside the project.
fn relative(root: &Path, cwd: &Path, p: &str) -> String {
    let given = Path::new(p.trim_matches(['"', '\'']));
    let abs = if given.is_absolute() { given.to_path_buf() } else { cwd.join(given) };
    let mut parts: Vec<String> = Vec::new();
    for c in abs.components() {
        match c {
            Component::ParentDir => {
                parts.pop();
            }
            Component::CurDir => {}
            other => parts.push(other.as_os_str().to_string_lossy().trim_start_matches(r"\\?\").to_owned()),
        }
    }
    let root_parts: Vec<String> = root.components().map(|c| c.as_os_str().to_string_lossy().trim_start_matches(r"\\?\").to_owned()).collect();
    let eq = |a: &str, b: &str| if cfg!(windows) { a.eq_ignore_ascii_case(b) || a.trim_end_matches('\\') == b.trim_end_matches('\\') } else { a == b };
    if parts.len() >= root_parts.len() && parts.iter().zip(&root_parts).all(|(a, b)| eq(a, b)) {
        let rest: Vec<&str> = parts.iter().skip(root_parts.len()).map(String::as_str).collect();
        if rest.is_empty() { ".".into() } else { rest.join("/") }
    } else {
        "<outside>".into()
    }
}

/// Files whose name says they hold secrets: kept as a path, the record marked `sensitive`.
fn is_sensitive(rel: &str) -> bool {
    let name = rel.rsplit('/').next().unwrap_or(rel).to_ascii_lowercase();
    name.starts_with(".env")
        || name.starts_with("id_rsa")
        || name.starts_with("id_ed25519")
        || [".pem", ".key", ".pfx", ".p12"].iter().any(|e| name.ends_with(e))
        || name == "credentials"
        || name == ".npmrc"
        || name == ".netrc"
}

/// The lines a Read covered: from the response's `file` block when given, else from the input's
/// `offset` and `limit`.
fn read_span(ti: &Value, resp: &Value) -> Option<(u64, u64)> {
    let f = resp.get("file").ok();
    let start = f.and_then(|f| f.get("startLine").and_then(Value::integer).ok());
    let count = f.and_then(|f| f.get("numLines").and_then(Value::integer).ok());
    if let (Some(a), Some(c)) = (start, count)
        && c > 0
    {
        return Some((a, a + c - 1));
    }
    let off = ti.get("offset").and_then(Value::integer).ok();
    let lim = ti.get("limit").and_then(Value::integer).ok();
    match (off, lim) {
        (Some(o), Some(l)) if l > 0 => Some((o.max(1), o.max(1) + l - 1)),
        (None, Some(l)) if l > 0 => Some((1, l)),
        _ => None,
    }
}

/// The lines an edit touches: before the edit, where `old_string` is; after it, where
/// `new_string` is; a Write covers its whole content. The file is read once, in memory only.
fn edit_span(name: &str, ev: &str, ti: &Value, root: &Path, cwd: &Path, p: &str) -> Option<(u64, u64)> {
    if name == "Write" {
        let c = text(ti, "content")?;
        return Some((1, c.lines().count().max(1) as u64));
    }
    if name == "NotebookEdit" {
        return None;
    }
    let needle = if name == "MultiEdit" {
        let first = ti.get("edits").and_then(Value::array).ok()?.first()?.clone();
        text(&first, if ev == "PreToolUse" { "old_string" } else { "new_string" })?.to_owned()
    } else {
        text(ti, if ev == "PreToolUse" { "old_string" } else { "new_string" })?.to_owned()
    };
    if needle.is_empty() {
        return None;
    }
    let path = if Path::new(p).is_absolute() { PathBuf::from(p) } else { cwd.join(p) };
    if relative(root, cwd, p) == "<outside>" {
        return None;
    }
    let mut buf = Vec::new();
    std::fs::File::open(&path).ok()?.take(MAX_FILE).read_to_end(&mut buf).ok()?;
    let body = String::from_utf8_lossy(&buf);
    let at = body.find(&needle).or_else(|| body.find(&needle.replace('\n', "\r\n")))?;
    let start = body.get(..at)?.matches('\n').count() as u64 + 1;
    Some((start, start + needle.trim_end_matches('\n').matches('\n').count() as u64))
}

/// Lines of the old and new text of an edit (sizes only; the text is never kept).
fn edit_size(name: &str, ti: &Value) -> Option<(u64, u64)> {
    let count = |t: &str| t.lines().count() as u64;
    match name {
        "Edit" => Some((count(text(ti, "old_string")?), count(text(ti, "new_string")?))),
        "Write" => Some((0, count(text(ti, "content")?))),
        "MultiEdit" => {
            let e = ti.get("edits").and_then(Value::array).ok()?;
            Some(e.iter().fold((0, 0), |(a, b), x| (a + text(x, "old_string").map_or(0, count), b + text(x, "new_string").map_or(0, count))))
        }
        _ => None,
    }
}

/// The episode id in a reply of the Nano's own tools: a JSON reply's `episode`, or `episode` in a
/// text reply.
fn episode(resp: &Value) -> Option<String> {
    let mut texts: Vec<String> = Vec::new();
    let content = match resp {
        Value::Array(a) => a.clone(),
        Value::Object(_) => resp.get("content").and_then(Value::array).map(<[Value]>::to_vec).unwrap_or_default(),
        Value::String(t) => {
            texts.push(t.clone());
            Vec::new()
        }
        _ => Vec::new(),
    };
    for c in &content {
        if let Some(t) = text(c, "text") {
            texts.push(t.to_owned());
        }
    }
    if let Ok(e) = resp.get("episode").and_then(Value::string) {
        return Some(e.to_owned());
    }
    for t in texts {
        if let Ok(v) = nmjson::parse(t.as_bytes())
            && let Ok(e) = v.get("episode").and_then(Value::string)
        {
            return Some(e.to_owned());
        }
        if let Some(i) = t.find("episode") {
            let rest = t.get(i + 7..).unwrap_or_default().trim_start_matches(['"', ':', '=', ' ']);
            let id: String = rest.chars().take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')).collect();
            if id.len() >= 4 {
                return Some(id);
            }
        }
    }
    None
}

/// Splits a shell command into segments (at `;`, `|`, `&&`, `||`) of words, honouring quotes.
fn segments(cmd: &str) -> Vec<Vec<String>> {
    let mut segs: Vec<Vec<String>> = vec![Vec::new()];
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let mut word = false;
    let flush = |cur: &mut String, word: &mut bool, segs: &mut Vec<Vec<String>>| {
        if *word {
            if let Some(last) = segs.last_mut() {
                last.push(std::mem::take(cur));
            }
            *word = false;
        }
    };
    for c in cmd.chars() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), c) => cur.push(c),
            (None, '"' | '\'') => {
                quote = Some(c);
                word = true;
            }
            (None, c) if c.is_whitespace() => flush(&mut cur, &mut word, &mut segs),
            (None, ';' | '|' | '&' | '\n') => {
                flush(&mut cur, &mut word, &mut segs);
                if segs.last().is_some_and(|l| !l.is_empty()) {
                    segs.push(Vec::new());
                }
            }
            (None, c) => {
                cur.push(c);
                word = true;
            }
        }
    }
    flush(&mut cur, &mut word, &mut segs);
    segs.retain(|s| !s.is_empty());
    segs
}

/// The class of a shell command, the record kind it gives, and the package it names. Across
/// segments the strongest class wins: test, build, version control, read, search.
fn command_class(cmd: &str) -> (&'static str, &'static str, Option<String>) {
    let mut best: (&str, &str, Option<String>, u8) = ("other", "shell", None, 0);
    for seg in segments(cmd) {
        // Skip `VAR=value` prefixes and wrappers.
        let words: Vec<&str> = seg
            .iter()
            .map(String::as_str)
            .skip_while(|w| (w.contains('=') && !w.starts_with('-')) || matches!(*w, "time" | "env" | "npx" | "sudo" | "&"))
            .collect();
        let Some(first) = words.first() else { continue };
        let prog = first.rsplit(['/', '\\']).next().unwrap_or(first).to_ascii_lowercase();
        let prog = prog.trim_end_matches(".exe").trim_end_matches(".cmd");
        let sub = words.iter().skip(1).find(|w| !w.starts_with('-') && !w.starts_with('+')).map(|w| w.to_ascii_lowercase()).unwrap_or_default();
        let pkg = words.windows(2).find(|w| matches!(w.first(), Some(&"-p" | &"--package"))).and_then(|w| w.get(1)).map(|p| (*p).to_owned());
        let c: (&str, &str, u8) = match (prog, sub.as_str()) {
            ("cargo", "test" | "nextest") => ("cargo-test", "test", 5),
            ("cargo", "check") => ("cargo-check", "build", 4),
            ("cargo", "build") => ("cargo-build", "build", 4),
            ("cargo", "clippy") => ("cargo-clippy", "build", 4),
            ("npm" | "pnpm" | "yarn" | "bun", "test" | "t") => ("npm-test", "test", 5),
            ("npm" | "pnpm" | "yarn" | "bun", "run") if words.iter().any(|w| w.starts_with("test")) => ("npm-test", "test", 5),
            ("npm" | "pnpm" | "yarn" | "bun", "run") if words.iter().any(|w| w.starts_with("build") || *w == "lint") => ("npm-build", "build", 4),
            ("pytest", _) => ("pytest", "test", 5),
            ("python" | "python3" | "py", _) if words.windows(2).any(|w| w == ["-m", "pytest"]) => ("pytest", "test", 5),
            ("go", "test") => ("go-test", "test", 5),
            ("go", "build" | "vet") => ("go-build", "build", 4),
            ("dotnet", "test") => ("dotnet-test", "test", 5),
            ("dotnet", "build") => ("dotnet-build", "build", 4),
            ("make", "test" | "check") => ("make-test", "test", 5),
            ("jest" | "vitest" | "mocha", _) => ("js-test", "test", 5),
            ("tsc", _) => ("tsc", "build", 4),
            ("git", "revert") => ("git-revert", "vcs", 3),
            ("git", "restore") => ("git-restore", "vcs", 3),
            ("git", "checkout") => ("git-checkout", "vcs", 3),
            ("git", "reset") => ("git-reset", "vcs", 3),
            ("git", "commit") => ("git-commit", "vcs", 3),
            ("git", "stash") => ("git-stash", "vcs", 3),
            ("cat" | "head" | "tail" | "sed" | "less" | "more" | "type" | "get-content" | "gc" | "wc" | "nl", _) => ("read-like", "read", 2),
            ("grep" | "rg" | "findstr" | "select-string" | "sls" | "find" | "ls" | "dir" | "get-childitem" | "gci" | "tree", _) => {
                ("search-like", "search", 1)
            }
            _ => continue,
        };
        if c.2 > best.3 {
            best = (c.0, c.1, if prog == "cargo" { pkg } else { None }, c.2);
        }
    }
    (best.0, best.1, best.2)
}

/// Files and folders a shell command names that exist inside the project (or `<outside>`), with
/// the lines of `sed -n 'a,bp' file` and `head -n N file`.
fn shell_paths(cmd: &str, root: &Path, cwd: &Path) -> Vec<(String, Option<(u64, u64)>)> {
    let mut out: Vec<(String, Option<(u64, u64)>)> = Vec::new();
    let mut checked = 0;
    for seg in segments(cmd) {
        let prog = seg.first().map(|w| w.rsplit(['/', '\\']).next().unwrap_or(w).to_ascii_lowercase()).unwrap_or_default();
        let mut span: Option<(u64, u64)> = None;
        for (i, w) in seg.iter().enumerate().skip(1) {
            if prog == "sed" && i >= 2 && seg.get(i - 1).is_some_and(|f| f == "-n") {
                span = w.trim_end_matches('p').split_once(',').and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)));
                continue;
            }
            if prog == "head" && seg.get(i - 1).is_some_and(|f| f == "-n") {
                span = w.parse::<u64>().ok().map(|k| (1, k));
                continue;
            }
            if w.starts_with('-') || w.len() < 2 || checked >= 48 {
                continue;
            }
            let looks = w.contains('/') || w.contains('\\') || w.contains('.');
            if !looks || w.contains("://") {
                continue;
            }
            checked += 1;
            let p = Path::new(w.as_str());
            let abs = if p.is_absolute() { p.to_path_buf() } else { cwd.join(p) };
            if abs.exists() {
                let rel = relative(root, cwd, w);
                if !out.iter().any(|(x, _)| *x == rel) {
                    out.push((rel, if abs.is_file() { span } else { None }));
                }
            }
        }
    }
    out
}

/// Section check (processor only): the record of each event from a fixture input holds the
/// expected fields, no planted secret and no file content.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let root = Path::new(if cfg!(windows) { "C:\\w\\shop" } else { "/w/shop" });
    let sep = if cfg!(windows) { "\\" } else { "/" };
    let file = format!("{}{sep}crates{sep}shop{sep}src{sep}lib.rs", root.display());
    let cases: Vec<(String, &str, &str)> = vec![
        (r#"{"hook_event_name":"SessionStart","session_id":"s1","source":"startup"}"#.into(), "session", "\"source\":\"startup\""),
        (
            r#"{"hook_event_name":"UserPromptSubmit","session_id":"s1","prompt_id":"p1","prompt":"fix it, token=sk-abcdefghijklmnopqrstuvwxyz123"}"#.into(),
            "prompt",
            "\"secrets_dropped\":",
        ),
        (
            format!(
                r#"{{"hook_event_name":"PostToolUse","session_id":"s1","tool_name":"Write","tool_input":{{"file_path":{},"content":"PLANTED CONTENT\nline2"}}}}"#,
                crate::jsonc::quote(&file)
            ),
            "write",
            "\"paths\":[\"crates/shop/src/lib.rs\"]",
        ),
        (
            r#"{"hook_event_name":"PostToolUseFailure","session_id":"s1","tool_name":"Bash","tool_input":{"command":"API_TOKEN=sk-test1234567890abcdefghij cargo test -p shop"},"error":"Exit code 101\nerror[E0308]: mismatched types\n --> crates/shop/src/lib.rs:12:5\ntest result: FAILED. 3 passed; 1 failed"}"#.into(),
            "test",
            "\"exit\":101",
        ),
        (r#"{"hook_event_name":"Stop","session_id":"s1","stop_hook_active":false}"#.into(), "stop", "\"kind\":\"stop\""),
        (r#"{"hook_event_name":"PreCompact","session_id":"s1","trigger":"auto"}"#.into(), "compact", "\"trigger\":\"auto\""),
    ];
    for (input, kind, want) in &cases {
        let v = nmjson::parse(input.as_bytes()).map_err(|e| e.to_string())?;
        let (r, _) = record("x", &v, root, true);
        let line = Value::Object(r).encode().map_err(|e| e.to_string())?;
        if !line.contains(&format!("\"kind\":\"{kind}\"")) || !line.contains(want) {
            return Err(format!("hook: record {line} lacks {kind} or {want}"));
        }
        if line.contains("PLANTED") || line.contains("sk-test") || line.contains("sk-abcdef") {
            return Err(format!("hook: record kept planted text: {line}"));
        }
    }
    let f = cases.get(3).and_then(|c| nmjson::parse(c.0.as_bytes()).ok()).ok_or("hook: case")?;
    let line = Value::Object(record("x", &f, root, true).0).encode().map_err(|e| e.to_string())?;
    if !line.contains("\"code\":\"E0308\"") || !line.contains("\"line\":12") || !line.contains("\"tests\":[3,1]") || !line.contains("\"pkg\":\"shop\"") {
        return Err(format!("hook: failed test record {line}"));
    }
    if relative(root, root, "/elsewhere/x.rs") != "<outside>" || command_class("git status && cargo check -p a").0 != "cargo-check" {
        return Err("hook: outside path or command class".into());
    }
    // Antigravity camelCase hook normalisation and stdout reply contract.
    let agy_cases: Vec<(String, &str, &str, &str)> = vec![
        (r#"{"hookEventName":"SessionStart","conversationId":"c1","modelName":"gemini-4-neon-high"}"#.into(), "SessionStart", "session", "\"source\":\"startup\""),
        (r#"{"hookEventName":"PreInvocation","conversationId":"c1","invocationNum":0,"initialNumSteps":1,"lastUserInput":"explain nm_context_pack"}"#.into(), "PreInvocation", "prompt", "\"turn\":\"turn-1\""),
        (r#"{"hookEventName":"PreInvocation","conversationId":"c1","invocationNum":2,"initialNumSteps":1,"lastUserInput":"explain nm_context_pack"}"#.into(), "PreInvocation", "other", "\"event\":\"PreInvocation\""),
        (
            format!(
                r#"{{"hookEventName":"PostToolUse","conversationId":"c1","stepIdx":3,"toolCall":{{"id":"tc1","name":"read_file","args":{{"Path":{},"StartLine":1,"EndLine":20}}}}}}"#,
                crate::jsonc::quote(&file)
            ),
            "PostToolUse",
            "read",
            "\"paths\":[\"crates/shop/src/lib.rs\"]",
        ),
        (
            format!(
                r#"{{"hookEventName":"PostToolUse","conversationId":"c1","stepIdx":3,"toolCall":{{"id":"tc1b","name":"multi_replace_file_content","args":{{"TargetFile":{},"ReplacementChunks":[{{"TargetContent":"a","ReplacementContent":"b\nc"}}]}}}}}}"#,
                crate::jsonc::quote(&file)
            ),
            "PostToolUse",
            "edit",
            "\"edit\":[1,2]",
        ),
        (
            r#"{"hookEventName":"PostToolUse","conversationId":"c1","stepIdx":4,"toolCall":{"id":"tc2","name":"call_mcp_tool","args":{"ServerName":"neuromorph-nano_neuromorph-nano","ToolName":"nm_context_pack","Arguments":{"question":"where is shop?"}}}}"#.into(),
            "PostToolUse",
            "nano",
            "\"ask\":\"where is shop?\"",
        ),
        (
            r#"{"hookEventName":"PostToolUse","conversationId":"c1","stepIdx":4,"toolCall":{"id":"tc2b","name":"call_mcp_tool","args":{"ServerName":"neuromorph-nano_neuromorph-nano","ToolName":"nm_context_pack","Arguments":"{\"question\":\"where is shop?\"}"}}}"#.into(),
            "PostToolUse",
            "nano",
            "\"tool\":\"nm_context_pack\"",
        ),
        (
            r#"{"hookEventName":"PostToolUse","conversationId":"c1","stepIdx":4,"toolCall":{"id":"tc2c","name":"call_mcp_tool","args":{"ServerName":"other-server","ToolName":"list_items","Arguments":{}}}}"#.into(),
            "PostToolUse",
            "mcp",
            "\"tool\":\"mcp__other-server__list_items\"",
        ),
        (
            r#"{"hookEventName":"PostToolUse","conversationId":"c1","stepIdx":5,"error":"exit status 1\nerror[E0308]: mismatched types","toolCall":{"id":"tc3","name":"run_command","args":{"CommandLine":"cargo test -p shop"}}}"#.into(),
            "PostToolUse",
            "test",
            "\"exit\":1",
        ),
    ];
    for (raw_json, agy_ev, kind, want) in &agy_cases {
        let v = nmjson::parse(raw_json.as_bytes()).map_err(|e| e.to_string())?;
        if !is_agy(&v) {
            return Err(format!("hook: agy not detected on {agy_ev}"));
        }
        let (norm_ev, norm_v) = normalize_agy(agy_ev, &v);
        let (r, _) = record(&norm_ev, &norm_v, root, true);
        let line = Value::Object(r).encode().map_err(|e| e.to_string())?;
        if !line.contains(&format!("\"kind\":\"{kind}\"")) || !line.contains(want) {
            return Err(format!("hook: agy {agy_ev} record {line} lacks {kind} or {want}"));
        }
    }
    let prime_sample = crate::prime::Given { kind: "core", id: "id1".into(), text: "note".into(), places: Vec::new(), from: Vec::new() };
    let injected = agy_reply("SessionStart", Some(prime_sample.text.as_str()));
    if !injected.contains("\"injectSteps\"") || !injected.contains("\"systemMessage\":\"note\"") || agy_reply("PostToolUse", None) != "{}" || agy_reply("Stop", None) != "{}" {
        return Err(format!("hook: agy reply contract mismatch: {injected}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn records() {
        super::check().unwrap();
    }
    #[test]
    fn counts_and_markers() {
        assert_eq!(super::test_counts("test result: ok. 12 passed; 0 failed; 0 ignored"), Some((12, 0)));
        assert_eq!(super::test_counts("===== 3 passed, 2 failed in 0.1s ====="), Some((3, 2)));
        assert!(super::failure_marker("running\ntest result: FAILED. 1 passed; 1 failed"));
        assert!(!super::failure_marker("Finished `release` profile"));
        assert_eq!(super::command_class("cd x && npm run test:unit").1, "test");
        assert_eq!(super::command_class("sed -n '1,20p' a.rs").0, "read-like");
    }
}
