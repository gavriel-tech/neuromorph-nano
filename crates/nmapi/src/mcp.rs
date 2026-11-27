//! MCP over stdio (JSON-RPC 2.0, one message per line): initialize, ping, tools/list and
//! tools/call, through the same dispatcher as HTTP.
use crate::Api;
use nmjson::{self as json, Value};
use std::collections::BTreeMap;
use std::io::{BufRead, Write};

/// Standing guidance for the agent (MCP `instructions`, shown by the host to the model): which tool
/// for which need, and the loop to close before finishing.
///
/// The guidance every listing shares (`INSTRUCTIONS`, `instructions_for`).
const CORE: &str = "This server has already read this project's folder (code, documents, notes, tables, logs). For every question or task about the project, call nm_context_pack with the question as your first step, before Grep, Glob or Read: one call searches the whole folder and returns the lines that answer it, each cited as path:first-last. Then read only the ranges it cites; search further only if they are not enough. For an exact name, nm_symbol_find; for package relations, nm_deps or nm_repo_map. Before finishing a Rust change, run nm_build_run (check, then test) on the changed package and fix what it reports.";

/// The instructions with the default listing, where every tool is in `tools/list` (2026-11-27:
/// a client such as Claude Code calls only listed tools, so the tools these instructions name
/// are all listed by default; `mcp.tool_count` defaults to every tool).
pub const INSTRUCTIONS: &str = "This server has already read this project's folder (code, documents, notes, tables, logs). For every question or task about the project, call nm_context_pack with the question as your first step, before Grep, Glob or Read: one call searches the whole folder and returns the lines that answer it, each cited as path:first-last. Then read only the ranges it cites; search further only if they are not enough. For an exact name, nm_symbol_find; for package relations, nm_deps or nm_repo_map. Before finishing a Rust change, run nm_build_run (check, then test) on the changed package and fix what it reports. Every tool is in the tool list: also nm_view, nm_feedback, nm_build_explain, nm_stepcheck, nm_verify, nm_status, nm_sync, nm_memory, nm_goal, nm_anatomy, nm_weights, nm_improve_scan, nm_patch_propose, nm_unit, and, to inspect and control the Nano itself as its web interface does, nm_storage, nm_cartridges, nm_learning, nm_comprehension, nm_settings, nm_live.";

/// The instructions for a listing of `listed` tools: `INSTRUCTIONS` when every tool is listed;
/// otherwise the shared guidance, the tools listed now, and how the others come into the list,
/// so the instructions never name a tool the client cannot call.
#[must_use]
pub fn instructions_for(listed: &[&str]) -> String {
    if PRIORITY_TOOLS.iter().all(|t| listed.contains(t)) {
        return INSTRUCTIONS.to_owned();
    }
    let dynamic = crate::whole::mcp_dynamic_tools_control(&nmtype::nano_root());
    format!(
        "{CORE} Use only the tools in the tool list; listed now: {}. The Nano's tool settings (mcp.tools, mcp.tool_count, mcp.enabled_tools) keep the others out of the list{}",
        listed.join(", "),
        if dynamic { "; dynamic tool activation brings one in when a call needs it, and the list then changes (tools/list_changed)." } else { "." }
    )
}

/// Protocol revision this server speaks.
pub const PROTOCOL: &str = "2025-06-18";

fn text(s: &str) -> Value {
    Value::String(s.to_owned())
}
fn object(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn ok(id: Value, result: Value) -> Value {
    object(vec![("jsonrpc", text("2.0")), ("id", id), ("result", result)])
}
fn err(id: Value, code: i64, message: &str) -> Value {
    let error = object(vec![("code", Value::Number(code.to_string())), ("message", text(message))]);
    object(vec![("jsonrpc", text("2.0")), ("id", id), ("error", error)])
}
/// The 5 core coding tools advertised in `tools/list` when `mcp.tools` is `"lean"`. Every tool in
/// the contract remains callable by name in `tools/call`.
pub const LEAN_TOOLS: &[&str] = &["nm_context_pack", "nm_symbol_find", "nm_view", "nm_deps", "nm_feedback"];

/// The Neuromorph Nano MCP tools ordered by coding priority so `mcp.tool_count` (1 to all 25, all by default)
/// selects the highest-leverage subset of size `k`. Roadmap Phase 4.5 (2026-11-24): the tools
/// after the first 19 match the web interface's inspections and controls (`crate::parity`); they
/// are callable on demand and promoted into `tools/list` by dynamic activation or `mcp.enabled_tools`.
pub const PRIORITY_TOOLS: &[&str] = &[
    "nm_context_pack",
    "nm_symbol_find",
    "nm_view",
    "nm_deps",
    "nm_feedback",
    "nm_repo_map",
    "nm_build_run",
    "nm_build_explain",
    "nm_stepcheck",
    "nm_verify",
    "nm_status",
    "nm_sync",
    "nm_memory",
    "nm_goal",
    "nm_anatomy",
    "nm_weights",
    "nm_improve_scan",
    "nm_patch_propose",
    "nm_unit",
    "nm_storage",
    "nm_cartridges",
    "nm_learning",
    "nm_comprehension",
    "nm_settings",
    // Roadmap Phase 4.5 (2026-11-24): the live activity (`crate::live`), on demand.
    "nm_live",
];

/// Tools dynamically promoted into `tools/list` during the current session when
/// `mcp.dynamic_tools` is `"on"`.
static DYNAMIC_ACTIVATED: std::sync::Mutex<std::collections::BTreeSet<&'static str>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());
/// True when `DYNAMIC_ACTIVATED` gained a tool outside the baseline limit and a
/// `notifications/tools/list_changed` line should be emitted after the current reply.
static LIST_CHANGED_PENDING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Dynamically activates tools relevant to `question` or `tool_name` when `dynamic_on` is true.
/// Returns the newly activated tool names that were outside the baseline tool set.
pub fn trigger_dynamic_tools(limit: usize, dynamic_on: bool, tool_name: &str, question: Option<&str>) -> Vec<&'static str> {
    trigger_dynamic_tools_with(limit, dynamic_on, None, tool_name, question)
}

/// As [`trigger_dynamic_tools`], honoring an explicit per-tool baseline when `custom_enabled` is
/// `Some` (non-empty).
pub fn trigger_dynamic_tools_with(
    limit: usize,
    dynamic_on: bool,
    custom_enabled: Option<&[String]>,
    tool_name: &str,
    question: Option<&str>,
) -> Vec<&'static str> {
    if !dynamic_on {
        return Vec::new();
    }
    let cap = limit.clamp(1, PRIORITY_TOOLS.len());
    let in_baseline = |name: &str| match custom_enabled {
        Some(list) if !list.is_empty() => list.iter().any(|t| t == name),
        _ => PRIORITY_TOOLS[..cap].contains(&name),
    };
    let mut candidates: Vec<&'static str> = Vec::new();
    if let Some(&exact) = PRIORITY_TOOLS.iter().find(|&&t| t == tool_name) {
        candidates.push(exact);
    }
    if let Some(q) = question {
        let low = q.to_lowercase();
        let has_any = |words: &[&str]| words.iter().any(|w| low.contains(w));
        if has_any(&["build", "test", "cargo", "compile", "clippy", "check", "fail", "error", "verify", "stepcheck"]) {
            candidates.extend(["nm_build_run", "nm_build_explain", "nm_stepcheck", "nm_verify"]);
        }
        if has_any(&["anatomy", "architecture", "repo map", "crate", "weight", "model", "cartridge", "unit", "layer"]) {
            candidates.extend(["nm_repo_map", "nm_anatomy", "nm_weights", "nm_unit"]);
        }
        if has_any(&["memory", "episode", "recall", "goal", "objective", "sync", "improve", "scan", "patch", "status"]) {
            candidates.extend(["nm_memory", "nm_goal", "nm_sync", "nm_improve_scan", "nm_patch_propose", "nm_status"]);
        }
        // Roadmap Phase 4.5 (2026-11-24): questions about the Nano's own storage, cartridges,
        // learning, comprehension or settings bring the interface-parity tools.
        if has_any(&["disk", "storage", "prune", "cartridge", "job filter", "learning", "preference", "rollback", "comprehension", "world model", "setting", "vram", "precision"]) {
            candidates.extend(["nm_storage", "nm_cartridges", "nm_learning", "nm_comprehension", "nm_settings"]);
        }
    }
    let mut added = Vec::new();
    if let Ok(mut set) = DYNAMIC_ACTIVATED.lock() {
        for c in candidates {
            if !in_baseline(c) && set.insert(c) {
                added.push(c);
            }
        }
    }
    if !added.is_empty() {
        LIST_CHANGED_PENDING.store(true, std::sync::atomic::Ordering::SeqCst);
    }
    added
}

/// Clears dynamically activated tools (used by tests and session resets).
pub fn reset_dynamic_tools() {
    if let Ok(mut set) = DYNAMIC_ACTIVATED.lock() {
        set.clear();
    }
    LIST_CHANGED_PENDING.store(false, std::sync::atomic::Ordering::SeqCst);
}

/// Build the `tools/list` result for `api`, filtering to [`LEAN_TOOLS`] when `lean` is true.
#[must_use]
pub fn tools_with(api: &Api, lean: bool) -> Value {
    tools_with_limit(api, if lean { LEAN_TOOLS.len() } else { api.operations().len() })
}

/// Build the `tools/list` result for `api`, advertising the top `limit` tools in [`PRIORITY_TOOLS`]
/// order (all operations when `limit >= api.operations().len()`).
#[must_use]
pub fn tools_with_limit(api: &Api, limit: usize) -> Value {
    tools_with_dynamic(api, limit, false)
}

/// Build the `tools/list` result for `api`, advertising the top `limit` tools in [`PRIORITY_TOOLS`]
/// order plus any dynamically activated tools when `dynamic_on` is true.
#[must_use]
pub fn tools_with_dynamic(api: &Api, limit: usize, dynamic_on: bool) -> Value {
    tools_with_config(api, limit, dynamic_on, None)
}

/// Build the `tools/list` result for `api`, using `custom_enabled` as the explicit default tool
/// set when `Some` (non-empty), else the top `limit` tools in [`PRIORITY_TOOLS`] order, plus any
/// dynamically activated tools when `dynamic_on` is true.
#[must_use]
pub fn tools_with_config(api: &Api, limit: usize, dynamic_on: bool, custom_enabled: Option<&[String]>) -> Value {
    let total = api.operations().len();
    let cap = limit.clamp(1, total);
    let allowed: &[&str] = &PRIORITY_TOOLS[..cap.min(PRIORITY_TOOLS.len())];
    let dyn_set: std::collections::BTreeSet<&'static str> = if dynamic_on {
        DYNAMIC_ACTIVATED.lock().map(|s| s.clone()).unwrap_or_default()
    } else {
        std::collections::BTreeSet::new()
    };
    let list = api
        .operations()
        .iter()
        .filter(|op| {
            let in_base = match custom_enabled {
                Some(explicit) if !explicit.is_empty() => explicit.iter().any(|t| t == op.name),
                _ => cap >= total || allowed.contains(&op.name),
            };
            in_base || dyn_set.contains(op.name)
        })
        .map(|op| object(vec![("name", text(op.name)), ("description", text(op.summary)), ("inputSchema", Api::input_schema(op))]))
        .collect();
    object(vec![("tools", Value::Array(list))])
}

/// Returns whether `tool_name` is permitted to execute under the active tool control settings.
/// When `tool_control_on` is true and `dynamic_on` is false, only the user's selected baseline
/// tools (`custom_enabled`, or top `limit` in [`PRIORITY_TOOLS`] when `None`) are permitted.
/// Unknown tool names pass through so normal unknown-tool errors and observers fire unchanged.
#[must_use]
pub fn tool_allowed(
    limit: usize,
    tool_control_on: bool,
    dynamic_on: bool,
    custom_enabled: Option<&[String]>,
    tool_name: &str,
) -> bool {
    if !tool_control_on || dynamic_on {
        return true;
    }
    if !PRIORITY_TOOLS.contains(&tool_name) {
        return true;
    }
    match custom_enabled {
        Some(explicit) if !explicit.is_empty() => explicit.iter().any(|t| t == tool_name),
        _ => {
            let cap = limit.clamp(1, PRIORITY_TOOLS.len());
            PRIORITY_TOOLS[..cap].contains(&tool_name)
        }
    }
}

fn tools(api: &Api) -> Value {
    let home = nmtype::nano_root();
    let tool_control_on = crate::whole::mcp_tool_control(&home);
    let custom = if tool_control_on { crate::whole::mcp_enabled_tools_control(&home) } else { None };
    tools_with_config(api, crate::whole::mcp_tool_count_control(&home), crate::whole::mcp_dynamic_tools_control(&home), custom.as_deref())
}
/// Told when a tool call starts, before it is dispatched (the audit's running row): tool name and
/// arguments.
pub type StartHook = fn(&str, &Value);
static START: std::sync::OnceLock<StartHook> = std::sync::OnceLock::new();
/// Installs the start hook (first call wins).
pub fn on_start(f: StartHook) {
    let _ = START.set(f);
}
fn call(api: &Api, params: &Value) -> Value {
    let name = params.get("name").and_then(Value::string).unwrap_or_default();
    // With G3's idle cycle running (`serve --muscle on`) a served call holds the gate for its whole
    // length, so no replay runs beside it; otherwise this takes nothing. 2026-11-27: a light call
    // (`LIGHT_TOOLS`, read-only state) takes neither this gate nor the writer's mark, so it answers
    // beside a call in progress instead of waiting for its whole length.
    let light = LIGHT_TOOLS.contains(&name);
    let _busy = (!light).then(crate::g3::busy);
    // A call preempts the writer at idle (`writer_idle`) and marks the time it ends.
    let _writer = (!light).then(crate::writer_idle::call);
    // Roadmap Phase 4.5 (2026-11-24): the call in the live view, by its registered name only.
    let mut live = crate::live::call(api.operations().iter().find(|o| o.name == name).map_or("unknown", |o| o.name));
    let empty = Value::Object(BTreeMap::new());
    let args = params.get("arguments").unwrap_or(&empty);
    let home = nmtype::nano_root();
    let q_arg = args.get("question").and_then(Value::string).ok().or_else(|| args.get("objective").and_then(Value::string).ok());
    let tool_control_on = crate::whole::mcp_tool_control(&home);
    let dynamic_on = crate::whole::mcp_dynamic_tools_control(&home);
    let limit = crate::whole::mcp_tool_count_control(&home);
    let custom = if tool_control_on { crate::whole::mcp_enabled_tools_control(&home) } else { None };
    if !tool_allowed(limit, tool_control_on, dynamic_on, custom.as_deref(), name) {
        let msg = format!(
            "disabled_by_tool_control: tool '{name}' is turned off in Tool Control. Enable it in Settings or turn Dynamic tool activation on."
        );
        return object(vec![
            ("content", Value::Array(vec![object(vec![("type", text("text")), ("text", text(&msg))])])),
            ("isError", Value::Bool(true)),
        ]);
    }
    let _ = trigger_dynamic_tools_with(
        limit,
        dynamic_on,
        custom.as_deref(),
        name,
        q_arg,
    );
    if let Some(f) = START.get() {
        f(name, args);
    }
    let (body, is_error) = match api.dispatch(name, args) {
        // A text reply (the context pack's plain-text form) goes out as it is, not JSON-quoted.
        Ok(Value::String(t)) => (t, false),
        Ok(v) => (v.encode().unwrap_or_default(), false),
        Err(e) => (format!("{}: {}", e.code, e.message), true),
    };
    // 2026-11-27 (C3): tokens in (the request's arguments) and out (the reply), by the call
    // record's own count (characters / 4, as `reply_tokens` in the audit row); no tokenizer pass.
    live.count("tokens_in", args.encode().map_or(0, |t| t.len()) / 4);
    live.count("tokens_out", body.len() / 4);
    object(vec![
        ("content", Value::Array(vec![object(vec![("type", text("text")), ("text", text(&body))])])),
        ("isError", Value::Bool(is_error)),
    ])
}
/// The client this server speaks to, as its `initialize` named itself (`clientInfo.name`, e.g.
/// "claude-code", "cursor", "Visual Studio Code"); 2026-11-26: written on every tool call's audit
/// row, so the interface says which coding agent made a call instead of assuming one.
static CLIENT: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

/// The connected client's name, or empty before its `initialize`.
#[must_use]
pub fn client() -> String {
    CLIENT.lock().map(|c| c.clone()).unwrap_or_default()
}

/// Answer one JSON-RPC line; notifications get no reply.
pub fn handle_line(api: &Api, line: &str) -> Option<String> {
    let line = line.trim_start_matches('\u{feff}');
    let Ok(msg) = json::parse(line.as_bytes()) else {
        return err(Value::Null, -32700, "parse error").encode().ok();
    };
    let Ok(id) = msg.get("id").cloned() else {
        return None;
    };
    let method = msg.get("method").and_then(Value::string).unwrap_or_default();
    let empty = Value::Object(BTreeMap::new());
    let params = msg.get("params").unwrap_or(&empty);
    if method == "initialize"
        && let Ok(name) = params.get("clientInfo").and_then(|c| c.get("name")).and_then(Value::string)
        && let Ok(mut c) = CLIENT.lock()
    {
        *c = name.chars().filter(|ch| !ch.is_control()).take(64).collect();
    }
    let reply = match method {
        "initialize" => ok(
            id,
            object(vec![
                ("protocolVersion", text(PROTOCOL)),
                ("capabilities", object(vec![("tools", object(vec![("listChanged", Value::Bool(true))]))])),
                ("instructions", text(&{
                    let list = tools(api);
                    let names: Vec<&str> = list
                        .get("tools")
                        .and_then(Value::array)
                        .map(|t| t.iter().filter_map(|x| x.get("name").and_then(Value::string).ok()).collect())
                        .unwrap_or_default();
                    instructions_for(&names)
                })),
                ("serverInfo", object(vec![("name", text("neuromorph-nano")), ("version", text(env!("CARGO_PKG_VERSION")))])),
            ]),
        ),
        "ping" => ok(id, Value::Object(BTreeMap::new())),
        "tools/list" => ok(id, tools(api)),
        "tools/call" => ok(id, call(api, params)),
        _ => err(id, -32601, "method not found"),
    };
    reply.encode().ok()
}
/// Serve stdio until end of input.
/// # Errors
/// Read or write failures.
pub fn run(api: &Api, input: impl BufRead, output: impl Write + Send) -> std::io::Result<()> {
    run_as(api, input, output, false)
}

/// Tool calls that read the process's state only and answer at once, beside a call in progress
/// (2026-11-27): `nm_status` and `nm_live` take no call gate.
pub const LIGHT_TOOLS: &[&str] = &["nm_status", "nm_live"];

/// One tool call at a time on the engine (2026-11-27): the calls of every connection (this
/// process's own stdio and the clients attached to its shared engine) take this gate, so the
/// engine sees the same one-call-at-a-time order it always had, while the reader of each
/// connection goes on reading (a `notifications/cancelled` for a waiting call, an `nm_status`).
static CALL_GATE: std::sync::Mutex<()> = std::sync::Mutex::new(());
/// Request ids the client cancelled while they waited for the gate, as `<connection>:<id
/// encoded>`; such a call is not run and gets no reply (MCP: a cancelled request's receiver
/// should not answer it).
static CONNECTIONS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static CANCELLED: std::sync::Mutex<std::collections::BTreeSet<String>> = std::sync::Mutex::new(std::collections::BTreeSet::new());
/// Calls waiting for the gate or running now, over every connection.
static IN_FLIGHT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Clients attached to this process's shared engine now, and since the start.
static ATTACHED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static ATTACHED_TOTAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// The loopback port of this process's shared engine (0: not shared).
static ENGINE_PORT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
thread_local! {
    /// Set on the threads that serve a client attached to this process's shared engine.
    static ATTACHED_CALL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Records the loopback port this process shares its engine on (`serve`, `crate::engine`).
pub fn set_engine_port(port: u16) {
    ENGINE_PORT.store(usize::from(port), std::sync::atomic::Ordering::SeqCst);
}
/// Clients attached to this process's shared engine now.
#[must_use]
pub fn attached() -> usize {
    ATTACHED.load(std::sync::atomic::Ordering::SeqCst)
}

/// The engine block of `nm_status`: whether this process shares its engine, how many clients are
/// attached, and how the asking client reaches it.
#[must_use]
pub fn engine_value() -> Value {
    use std::sync::atomic::Ordering::SeqCst;
    let port = ENGINE_PORT.load(SeqCst);
    let attached_call = ATTACHED_CALL.with(std::cell::Cell::get);
    let this = if attached_call {
        format!("attached: this client's own serve process forwards to the shared engine of process {} on 127.0.0.1:{port}; one engine, one set of MicroModels on the card", std::process::id())
    } else if port > 0 {
        format!("own: this process holds the engine and shares it on 127.0.0.1:{port} with later clients of the same folder and options")
    } else {
        "own: this process holds the engine (not shared)".to_owned()
    };
    object(vec![
        ("pid", Value::Number(std::process::id().to_string())),
        ("shared_port", Value::Number(port.to_string())),
        ("clients_attached", Value::Number(ATTACHED.load(SeqCst).to_string())),
        ("clients_attached_since_start", Value::Number(ATTACHED_TOTAL.load(SeqCst).to_string())),
        ("calls_in_flight", Value::Number(IN_FLIGHT.load(SeqCst).saturating_sub(1).to_string())),
        ("this_client", text(&this)),
    ])
}

/// The request id of a `notifications/cancelled` line, encoded, when `line` is one.
fn cancelled_id(line: &str) -> Option<String> {
    let msg = json::parse(line.as_bytes()).ok()?;
    if msg.get("method").and_then(Value::string).ok()? != "notifications/cancelled" {
        return None;
    }
    msg.get("params").and_then(|p| p.get("requestId")).ok().and_then(|v| v.encode().ok())
}

/// The (id encoded, tool name) of a `tools/call` line.
fn call_of(line: &str) -> Option<(String, String)> {
    let msg = json::parse(line.as_bytes()).ok()?;
    if msg.get("method").and_then(Value::string).ok()? != "tools/call" {
        return None;
    }
    let id = msg.get("id").ok()?.encode().ok()?;
    let name = msg.get("params").and_then(|p| p.get("name")).and_then(Value::string).unwrap_or_default().to_owned();
    Some((id, name))
}

fn write_reply(output: &std::sync::Mutex<impl Write>, reply: &str) -> std::io::Result<()> {
    let mut out = output.lock().map_err(|_| std::io::Error::other("output lock poisoned"))?;
    writeln!(out, "{reply}")?;
    if LIST_CHANGED_PENDING.swap(false, std::sync::atomic::Ordering::SeqCst) {
        writeln!(out, r#"{{"jsonrpc":"2.0","method":"notifications/tools/list_changed"}}"#)?;
    }
    out.flush()
}

/// Runs one `tools/call` line: through the call gate unless it is a light tool, skipped when it
/// was cancelled while it waited, its reply written, then the memory write's second half.
fn serve_call(api: &Api, line: &str, key: &str, name: &str, output: &std::sync::Mutex<impl Write>, attached: bool) -> std::io::Result<()> {
    ATTACHED_CALL.with(|c| c.set(attached));
    let light = LIGHT_TOOLS.contains(&name);
    let _gate = if light { None } else { Some(CALL_GATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner)) };
    if CANCELLED.lock().is_ok_and(|mut c| c.remove(key)) {
        return Ok(());
    }
    let Some(reply) = handle_line(api, line) else { return Ok(()) };
    write_reply(output, &reply)?;
    // The reply is out; the memory write's second half follows (`session::settle`). For the
    // test that the write is recoverable, `NANO_TEST_STOP_AFTER_REPLY` ends the process here.
    if std::env::var_os("NANO_TEST_STOP_AFTER_REPLY").is_some() {
        #[expect(clippy::disallowed_methods, reason = "the test needs the process to end with no further work, as a crash would")]
        std::process::exit(0);
    }
    if !light && let Some(org) = crate::tools::installed() {
        crate::session::settle(org);
    }
    Ok(())
}

/// Serve one connection until end of input: `attached` when it is a client attached to this
/// process's shared engine (`crate::engine`). 2026-11-27 (MCP stability fix): each tool call runs
/// on its own thread behind the call gate, so the connection's reader goes on while a call runs:
/// an `nm_status` answers at once, and a call the client cancelled while it waited (its 60 s
/// timeout) is dropped instead of running after the client gave up on it. The engine still
/// runs one call at a time.
/// # Errors
/// Read or write failures.
pub fn run_as(api: &Api, input: impl BufRead, output: impl Write + Send, attached: bool) -> std::io::Result<()> {
    use std::sync::atomic::Ordering::SeqCst;
    let output = std::sync::Mutex::new(output);
    let conn = CONNECTIONS.fetch_add(1, SeqCst);
    if attached {
        ATTACHED.fetch_add(1, SeqCst);
        ATTACHED_TOTAL.fetch_add(1, SeqCst);
    }
    let result = std::thread::scope(|scope| -> std::io::Result<()> {
        // 2026-11-24: lines are read as bytes and decoded leniently. `lines()` ended the whole
        // connection on the first line that was not valid UTF-8, leaving the client's data
        // unread, and on Windows closing a socket with unread data resets it ("forcibly closed by
        // the remote host", one call in the 2-hour soak).
        for raw in input.split(b'\n') {
            let raw = raw?;
            let line = String::from_utf8_lossy(&raw);
            let line = line.trim_end_matches('\r').trim_start_matches('\u{feff}').to_owned();
            if line.trim().is_empty() {
                continue;
            }
            if let Some(id) = cancelled_id(&line) {
                if let Ok(mut c) = CANCELLED.lock() {
                    c.insert(format!("{conn}:{id}"));
                }
                continue;
            }
            if let Some((id, name)) = call_of(&line) {
                let out = &output;
                let key = format!("{conn}:{id}");
                IN_FLIGHT.fetch_add(1, SeqCst);
                scope.spawn(move || {
                    if let Err(e) = serve_call(api, &line, &key, &name, out, attached) {
                        eprintln!("mcp: a reply could not be written: {e}");
                    }
                    IN_FLIGHT.fetch_sub(1, SeqCst);
                });
                continue;
            }
            if let Some(reply) = handle_line(api, &line) {
                write_reply(&output, &reply)?;
            }
        }
        Ok(())
    });
    if attached {
        ATTACHED.fetch_sub(1, SeqCst);
    }
    if let Ok(mut c) = CANCELLED.lock() {
        let mine = format!("{conn}:");
        c.retain(|k| !k.starts_with(&mine));
    }
    result
}

/// Section check (2026-11-13, review: the outward interface had no check of its own): over the
/// served tool table, `initialize` names the protocol, a notification gets no reply, a broken line
/// gets -32700, an unknown method -32601, `tools/list` lists every operation once with an input
/// schema, `ping` answers, a call to an unknown tool is an error result (not a fault), and a
/// leading byte-order mark is accepted.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let api = crate::tools::api();
    let reply = |l: &str| handle_line(&api, l).and_then(|r| json::parse(r.as_bytes()).ok());
    let init = reply(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#).ok_or("nmapi.mcp: no initialize reply")?;
    if init.get("result").and_then(|r| r.get("protocolVersion")).and_then(Value::string).ok() != Some(PROTOCOL) {
        return Err("nmapi.mcp: initialize".into());
    }
    if handle_line(&api, r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#).is_some() {
        return Err("nmapi.mcp: a notification was answered".into());
    }
    let broken = reply("{not json").ok_or("nmapi.mcp: no parse-error reply")?;
    let unknown = reply(r#"{"jsonrpc":"2.0","id":2,"method":"no/such"}"#).ok_or("nmapi.mcp: no reply")?;
    // Error codes are negative numbers; compared as encoded text.
    let text_of = |v: &Value| v.encode().unwrap_or_default();
    if !text_of(&broken).contains("-32700") || !text_of(&unknown).contains("-32601") {
        return Err("nmapi.mcp: error codes".into());
    }
    let list = reply("\u{feff}{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/list\"}")
        .ok_or("nmapi.mcp: no tools/list reply (byte-order mark)")?;
    let tools = list.get("result").and_then(|r| r.get("tools")).and_then(Value::array).map_err(|_| "nmapi.mcp: tools/list shape")?;
    let mut names: Vec<&str> = tools.iter().filter_map(|t| t.get("name").and_then(Value::string).ok()).collect();
    let all_schemas = tools.iter().all(|t| t.get("inputSchema").is_ok_and(|s| matches!(s, Value::Object(_))));
    let n = names.len();
    names.sort_unstable();
    names.dedup();
    // Roadmap Phase 4.5 (2026-11-24): with more tools than `mcp.tool_count` allows, the default
    // listing is the configured count (the rest are on demand), or the user's explicit list.
    let home = nmtype::nano_root();
    let configured = crate::whole::mcp_tool_count_control(&home).min(api.operations().len());
    let explicit = crate::whole::mcp_enabled_tools_control(&home).filter(|_| crate::whole::mcp_tool_control(&home)).map(|l| l.len());
    let expected = [Some(api.operations().len()), Some(LEAN_TOOLS.len()), Some(configured), explicit];
    if !expected.contains(&Some(n)) || names.len() != n || !all_schemas {
        return Err(format!("nmapi.mcp: tools/list {n} tools, {} distinct, schemas {all_schemas}", names.len()));
    }
    let lean_list = tools_with(&api, true);
    let lean_tools = lean_list.get("tools").and_then(Value::array).map_err(|_| "nmapi.mcp: lean tools/list shape")?;
    if lean_tools.len() != LEAN_TOOLS.len() {
        return Err(format!("nmapi.mcp: lean tools/list count {}, expected {}", lean_tools.len(), LEAN_TOOLS.len()));
    }
    reset_dynamic_tools();
    let added = trigger_dynamic_tools(5, true, "nm_context_pack", Some("why did cargo build and clippy fail"));
    let dyn_list = tools_with_dynamic(&api, 5, true);
    let dyn_tools = dyn_list.get("tools").and_then(Value::array).map_err(|_| "nmapi.mcp: dynamic tools/list shape")?;
    let pending = LIST_CHANGED_PENDING.swap(false, std::sync::atomic::Ordering::SeqCst);
    reset_dynamic_tools();
    if added.len() != 4 || dyn_tools.len() != 9 || !pending {
        return Err(format!("nmapi.mcp: dynamic tool activation added {:?}, count {}, pending {pending}", added, dyn_tools.len()));
    }
    // Custom per-tool control: strict mode (dynamic off) blocks unselected tools; dynamic on
    // starts from the user's selected baseline and promotes additional tools on demand.
    let explicit = vec!["nm_context_pack".to_owned(), "nm_build_run".to_owned()];
    if !tool_allowed(19, false, false, Some(&explicit), "nm_status") {
        return Err("nmapi.mcp: tool_control=off must allow default tools".into());
    }
    if !tool_allowed(19, true, false, Some(&explicit), "nm_build_run")
        || tool_allowed(19, true, false, Some(&explicit), "nm_status")
    {
        return Err("nmapi.mcp: tool_control=on + dynamic=off must allow only selected tools".into());
    }
    if !tool_allowed(19, true, true, Some(&explicit), "nm_status") {
        return Err("nmapi.mcp: tool_control=on + dynamic=on must allow dynamic tool activation".into());
    }
    let custom_strict = tools_with_config(&api, 19, false, Some(&explicit));
    let strict_tools = custom_strict.get("tools").and_then(Value::array).map_err(|_| "nmapi.mcp: custom strict tools/list shape")?;
    let custom_list = tools_with_config(&api, 19, true, Some(&explicit));
    let custom_tools = custom_list.get("tools").and_then(Value::array).map_err(|_| "nmapi.mcp: custom tools/list shape")?;
    let added2 = trigger_dynamic_tools_with(19, true, Some(&explicit), "nm_status", None);
    let after_dyn = tools_with_config(&api, 19, true, Some(&explicit));
    let after_tools = after_dyn.get("tools").and_then(Value::array).map_err(|_| "nmapi.mcp: custom+dyn tools/list shape")?;
    reset_dynamic_tools();
    if strict_tools.len() != 2 || custom_tools.len() != 2 || added2 != ["nm_status"] || after_tools.len() != 3 {
        return Err(format!("nmapi.mcp: custom per-tool baseline failed (strict {}, base {}, added {added2:?}, after {})", strict_tools.len(), custom_tools.len(), after_tools.len()));
    }
    if reply(r#"{"jsonrpc":"2.0","id":4,"method":"ping"}"#).and_then(|v| v.get("result").ok().cloned()).is_none() {
        return Err("nmapi.mcp: ping".into());
    }
    let call = reply(r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"nm_no_such_tool","arguments":{}}}"#)
        .ok_or("nmapi.mcp: no tools/call reply")?;
    if call.get("result").and_then(|r| r.get("isError")).ok() != Some(&Value::Bool(true)) {
        return Err("nmapi.mcp: an unknown tool must give an error result".into());
    }
    // 2026-11-15 (Nano build plan E2, audit of real work): an observed contract reports every
    // dispatched call exactly once, a failed one too.
    static SEEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    fn watch(_: &str, _: &Value, out: &Result<Value, crate::ApiError>, _: u128) {
        if out.is_err() {
            SEEN.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
    let observed = crate::tools::api().observed(watch);
    let before = SEEN.load(std::sync::atomic::Ordering::SeqCst);
    let _ = handle_line(&observed, r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"nm_no_such_tool","arguments":{}}}"#);
    if SEEN.load(std::sync::atomic::Ordering::SeqCst) != before + 1 {
        return Err("nmapi.mcp: the observer must see each dispatched call once".into());
    }
    Ok(())
}
