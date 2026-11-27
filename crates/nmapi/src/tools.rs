//! The MCP tools Claude Code calls (PLAN section 7; guide 17, 17.1, 17.4), S3 set: status,
//! anatomy, context pack, repo map, symbol find, deps, view, sync, goal. Each handler reads
//! the one organism this process serves (set once by `serve`) and stays short.
use crate::forage::Organism;
use crate::{Api, ApiError, Field, FieldKind, Operation};
use nmjson::Value;
use nmtype::{Packet, SubstrateView};
use std::sync::OnceLock;

static ORG: OnceLock<Organism> = OnceLock::new();
/// The start of the organism in the background: `None` when no start is under way (or none was
/// begun), `Some(None)` while it runs, `Some(Some(fault))` when it failed.
static START: std::sync::Mutex<Option<Option<String>>> = std::sync::Mutex::new(None);
static STARTED: std::sync::Condvar = std::sync::Condvar::new();

/// Install the organism this process serves (first call wins); a tool call waiting for the start
/// goes on.
/// # Errors
/// An organism was already installed.
pub fn install(org: Organism) -> Result<(), ApiError> {
    let r = ORG.set(org).map_err(|_| ApiError::new("nm-409", "organism already installed"));
    if let Ok(mut s) = START.lock() {
        *s = None;
    }
    STARTED.notify_all();
    r
}
/// Marks the organism's start as under way in the background: until `install` (or
/// `start_failed`), a tool call that needs the organism waits for it, while the handshake and the
/// tool list answer at once.
pub fn starting() {
    if let Ok(mut s) = START.lock() {
        *s = Some(None);
    }
}
/// The background start failed: waiting and later tool calls get this fault.
pub fn start_failed(fault: &str) {
    if let Ok(mut s) = START.lock() {
        *s = Some(Some(fault.to_owned()));
    }
    STARTED.notify_all();
}
/// The organism this process serves, if installed.
#[must_use]
pub fn installed() -> Option<&'static Organism> {
    ORG.get()
}
fn org() -> Result<&'static Organism, ApiError> {
    if let Some(o) = ORG.get() {
        return Ok(o);
    }
    let mut s = START.lock().map_err(|_| ApiError::new("nm-500", "start state poisoned"))?;
    while ORG.get().is_none() && matches!(*s, Some(None)) {
        s = STARTED.wait(s).map_err(|_| ApiError::new("nm-500", "start state poisoned"))?;
    }
    if let Some(o) = ORG.get() {
        return Ok(o);
    }
    match s.as_ref() {
        Some(Some(fault)) => Err(ApiError::new("nm-503", fault.clone())),
        _ => Err(ApiError::new("nm-503", "no organism installed; start with `serve --root <repo>`")),
    }
}
fn s(t: &str) -> Value {
    Value::String(t.to_owned())
}
fn n(x: usize) -> Value {
    Value::Number(x.to_string())
}
fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn arg<'a>(v: &'a Value, k: &str) -> Option<&'a str> {
    v.get(k).and_then(Value::string).ok()
}
fn fault(e: &nmtype::Fault) -> ApiError {
    ApiError::new("nm-500", e.to_string())
}
fn packet(p: &Packet) -> Value {
    obj(vec![
        ("key", s(p.key())),
        ("kind", s(p.kind())),
        ("text", s(p.text())),
        ("status", s(p.get_status().name())),
        ("source", s(&p.provenance().source)),
        ("version", Value::Number(p.provenance().version.to_string())),
        ("evidence", Value::Array(p.get_evidence().iter().map(|e| s(e)).collect())),
    ])
}

/// Roadmap Phase 4.5 (2026-11-24): `nm_live`, the live activity of this process as the interface
/// shows it (`crate::live::snapshot`): stages active, MicroModels executing, card memory, the
/// machine's capacity, the idle writer and the recent timeline. Behavioural fields only.
fn live(_: &Value) -> Result<Value, ApiError> {
    Ok(crate::live::snapshot())
}

/// The Core Brain's held changes as `nm_status` last read them.
static LAST_HELD: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn status(_: &Value) -> Result<Value, ApiError> {
    let o = org()?;
    let sub = o.sub();
    let snap = o.nte.snapshot();
    // 2026-11-27: `nm_status` answers beside a call in progress (it takes no call gate), so it
    // does not wait for the Core Brain a running call holds: the count is the last one read then.
    let held = match o.brain.try_lock() {
        Ok(b) => {
            let n = b.buffer.changes.len();
            LAST_HELD.store(n, std::sync::atomic::Ordering::Relaxed);
            n
        }
        Err(_) => LAST_HELD.load(std::sync::atomic::Ordering::Relaxed),
    };
    // 2026-11-27: `models_loaded` counts the MicroModels held on the card or seated, from the same
    // view as `micromodels.held` ("N of 4"), so the two never disagree.
    let micromodels = crate::preload::view();
    let models_loaded = micromodels
        .get("roles")
        .and_then(Value::object)
        .map_or(0, |r| r.values().filter(|x| x.get("state").and_then(Value::string).is_ok_and(|st| st == "held")).count());
    Ok(obj(vec![
        ("snapshot_version", Value::Number(snap.version.to_string())),
        ("nodes", n(snap.nodes().len())),
        ("components", n(snap.components.len())),
        ("substrate_version", Value::Number(sub.version.to_string())),
        ("substrate_digest", s(&sub.digest())),
        ("files", n(sub.files.len())),
        ("read_by", {
            let mut by: std::collections::BTreeMap<String, Value> = std::collections::BTreeMap::new();
            for f in &sub.files {
                let c = by.get(f.decoder).and_then(|v| v.integer().ok()).unwrap_or(0);
                by.insert(f.decoder.to_owned(), Value::Number((c + 1).to_string()));
            }
            Value::Object(by)
        }),
        // Files read generally (no specific reader fits them), by how: container listings,
        // container members, gzip data, raw text runs.
        ("read_generally", {
            let mut by: std::collections::BTreeMap<String, Value> = std::collections::BTreeMap::new();
            for how in sub.files.iter().filter_map(|f| f.general) {
                let c = by.get(how).and_then(|v| v.integer().ok()).unwrap_or(0);
                by.insert(how.to_owned(), Value::Number((c + 1).to_string()));
            }
            Value::Object(by)
        }),
        (
            "skipped",
            Value::Object(nmsubs::repo::skip_counts(&sub.skipped).into_iter().map(|(r, c)| (r.to_owned(), n(c))).collect()),
        ),
        ("symbols", n(sub.symbols.len())),
        ("crates", n(sub.crates.len())),
        ("changes_logged", n(held)),
        ("max_run_ms", Value::Number(o.max_run_ms.to_string())),
        ("models_loaded", n(models_loaded)),
        ("slots", o.slots.qualification()),
        ("engine_last", s(&crate::neural::last_note())),
        // Roadmap Phase 4.6 (2026-11-25): where the MicroModels run, in plain English: the card
        // in use and why it was chosen, or why the processor (no driver, no card, a card older
        // than the kernels, a driver too old, or NANO_GPU=cpu).
        ("backend", s(&nmengn::device::status_line())),
        ("delta", crate::delta::status()),
        // 2026-11-27 (MCP stability fix): the four MicroModels as loaded at start (`preload`),
        // the host memory rule with its figures now, and the shared engine and its clients.
        ("micromodels", micromodels),
        ("host_memory", host_memory(o)),
        // 2026-11-27: each memory area's use against its budget, and the drones' intake.
        ("memory_areas", crate::memory_areas::refresh(o)),
        ("engine", crate::mcp::engine_value()),
    ]))
}
/// The host memory rule with the figures now and what the larger holders keep in RAM (MiB): each
/// MicroModel's weights read into RAM (distinct tensors) and the relevance-key store.
fn host_memory(o: &Organism) -> Value {
    let mb = |b: u64| Value::Number((b >> 20).to_string());
    let mut held = std::collections::BTreeMap::new();
    for (slot, m) in crate::neural::bound_models() {
        let role = if slot == "base" { "reader" } else { slot };
        held.insert(format!("{role}_weights"), mb(m.pop.prov.totals().0));
    }
    if let Some(ram) = crate::writer_idle::resident_writer_view(|w| w.pop.prov.totals().0) {
        held.insert("writer_weights".to_owned(), mb(ram));
    }
    if let Some(k) = o.whole.read().ok().and_then(|w| w.as_ref().and_then(|w| w.relevance_keys())) {
        held.insert("relevance_keys".to_owned(), mb(k.held().0 as u64));
    }
    if let Some(p) = o.tentacle_pass.read().ok().and_then(|p| p.clone()) {
        held.insert("encoded_chunks".to_owned(), mb(p.held_bytes() as u64));
    }
    let mut v = nmengn::limits::host_rule_value();
    if let Value::Object(m) = &mut v {
        m.insert("held_mb".to_owned(), Value::Object(held));
    }
    v
}
fn anatomy(v: &Value) -> Result<Value, ApiError> {
    if let Some(level) = arg(v, "level") {
        return Ok(crate::anatomy::zoom(org()?, level, arg(v, "id")));
    }
    let snap = org()?.nte.snapshot();
    let focus = arg(v, "node");
    let nodes = snap.nodes().iter().filter(|x| focus.is_none_or(|f| x.id == f || x.parent.as_deref() == Some(f)));
    let edges = snap.edges().iter().filter(|e| focus.is_none_or(|f| e.from == f || e.to == f));
    let bypass = snap.edges().iter().filter(|e| nmnte::topology::bypasses_brain(&snap.graph, &e.from, &e.to)).count();
    Ok(obj(vec![
        ("version", Value::Number(snap.version.to_string())),
        (
            "nodes",
            Value::Array(
                nodes
                    .map(|x| {
                        obj(vec![("id", s(&x.id)), ("kind", s(x.kind.name())), ("parent", x.parent.as_deref().map_or(Value::Null, s))])
                    })
                    .collect(),
            ),
        ),
        ("edges", Value::Array(edges.map(|e| s(&format!("{} -> {}", e.from, e.to))).collect())),
        ("waves", Value::Array(snap.plan.waves.iter().map(|w| n(w.len())).collect())),
        ("edges_bypassing_core_brain", n(bypass)),
    ]))
}
/// Diagnostics of the last tool call that hold no code (parts that ran, the context map and
/// other figures), kept for the audit row instead of the agent's reply; the observer takes them.
static LAST: std::sync::Mutex<Option<Value>> = std::sync::Mutex::new(None);

/// Takes the diagnostics the last call left, if any.
#[must_use]
pub fn take_diagnostics() -> Option<Value> {
    LAST.lock().ok().and_then(|mut g| g.take())
}
fn set_diagnostics(v: Value) {
    if let Ok(mut g) = LAST.lock() {
        *g = Some(v);
    }
}

/// One part a call used, for the audit row: the part id (anatomy id, `crate.sub#instance`,
/// sub-component id or crate id), when it started (milliseconds since 1970) and how long it ran,
/// each only when that part's own clock measured it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Step {
    pub part: String,
    pub start_ms: Option<u64>,
    pub ms: Option<u64>,
    /// Steps that ran as one shared pass (the eight drones' intake, say) carry the pass's name.
    pub shared: Option<&'static str>,
    /// For a MicroModel step, the first and last model layer it ran.
    pub layers: Option<(u64, u64)>,
    /// For a tentacle step, the job the Core Brain's plan gave it for this call
    /// (`nmpod::Job::name`), when the path assigns jobs.
    pub job: Option<&'static str>,
    /// For a tentacle step, the lens its window was filled with for this call, when it has one.
    pub lens: Option<String>,
    /// The same length in microseconds, when the part's clock measured it that finely.
    pub us: Option<u64>,
    /// The step ran inside another step (named here) and its time is part of that step's.
    pub inside: Option<&'static str>,
    /// The cells of the part this step touched, by name (memory tiers read or written, the Core
    /// Brain components a reduce ran through).
    pub cells: Vec<String>,
    /// Counts this step reported (chunks in a tentacle's window, packets in, episodes recalled).
    pub counts: Vec<(&'static str, u64)>,
}

/// Steps of the call in progress, in the order they were noted. Calls on one served contract run
/// one at a time; the observer takes them when the call ends.
static STEPS: std::sync::Mutex<Vec<Step>> = std::sync::Mutex::new(Vec::new());
/// Most steps kept for one call (a caller that never takes them cannot grow the list past this).
const STEPS_CAP: usize = 1024;

/// Notes one step of the call in progress.
pub fn note(st: Step) {
    if let Ok(mut g) = STEPS.lock()
        && g.len() < STEPS_CAP
    {
        g.push(st);
    }
}
/// Notes one step of the call in progress.
pub fn note_step(part: &str, start_ms: Option<u64>, ms: Option<u64>) {
    note_shared(part, start_ms, ms, None);
}
/// Notes one step that ran inside a shared pass named `shared`.
pub fn note_shared(part: &str, start_ms: Option<u64>, ms: Option<u64>, shared: Option<&'static str>) {
    note(Step { part: part.to_owned(), start_ms, ms, shared, ..Step::default() });
}
/// Notes one tentacle step with the job and lens it carried for this call.
pub fn note_job(part: &str, start_ms: Option<u64>, ms: Option<u64>, job: Option<&'static str>, lens: Option<String>) {
    note(Step { part: part.to_owned(), start_ms, ms, job, lens, ..Step::default() });
}
/// The clock of one step: when it started (milliseconds since 1970), whole milliseconds and
/// microseconds spent.
#[derive(Clone, Copy, Debug)]
pub struct Clocked {
    pub start_ms: u64,
    pub ms: u64,
    pub us: u64,
}
/// Runs `f` and measures it.
pub fn clocked<T>(f: impl FnOnce() -> T) -> (T, Clocked) {
    let start_ms = nmtype::now_ms();
    let t = std::time::Instant::now();
    let out = f();
    let e = t.elapsed();
    (out, Clocked { start_ms, ms: u64::try_from(e.as_millis()).unwrap_or(u64::MAX), us: u64::try_from(e.as_micros()).unwrap_or(u64::MAX) })
}
/// A step for `part` with the clock `c`.
#[must_use]
pub fn step_at(part: &str, c: Clocked) -> Step {
    Step { part: part.to_owned(), start_ms: Some(c.start_ms), ms: Some(c.ms), us: Some(c.us), ..Step::default() }
}
/// Notes a MicroModel step timed by `f`, with the first and last model layer it runs.
pub fn timed_layers<T>(part: &str, layers: (u64, u64), f: impl FnOnce() -> T) -> T {
    let (out, c) = clocked(f);
    note(Step { layers: Some(layers), ..step_at(part, c) });
    out
}
/// Notes a MicroModel step with the first and last model layer it ran.
pub fn note_layers(part: &str, start_ms: Option<u64>, ms: Option<u64>, layers: (u64, u64)) {
    note(Step { part: part.to_owned(), start_ms, ms, layers: Some(layers), ..Step::default() });
}
/// Notes a step timed by `f`: its start now and its measured length.
pub fn timed_step<T>(part: &str, f: impl FnOnce() -> T) -> T {
    let (out, c) = clocked(f);
    note(step_at(part, c));
    out
}
/// Takes the steps noted since the last take.
#[must_use]
pub fn take_steps() -> Vec<Step> {
    STEPS.lock().map(|mut g| std::mem::take(&mut *g)).unwrap_or_default()
}
/// The step of the work a call did after its reply went out (`session::settle`); the next call's
/// row carries it, since the call's own row was written before it ran.
static AFTER_REPLY: std::sync::Mutex<Option<Step>> = std::sync::Mutex::new(None);
/// Keeps the step of work done after a reply for the next call's row.
pub fn note_after_reply(st: Step) {
    if let Ok(mut a) = AFTER_REPLY.lock() {
        *a = Some(st);
    }
}
fn clear_steps() {
    if let Ok(mut g) = STEPS.lock() {
        g.clear();
        if let Some(st) = AFTER_REPLY.lock().ok().and_then(|mut a| a.take()) {
            g.push(st);
        }
    }
    RUN_END.store(0, std::sync::atomic::Ordering::Relaxed);
}
/// When the last whole-path run returned (milliseconds since 1970; 0 before any).
static RUN_END: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Notes when a whole-path run returned (its assembly starts there).
pub fn note_run_end(at_ms: u64) {
    RUN_END.store(at_ms, std::sync::atomic::Ordering::Relaxed);
}
/// The steps an answer's path leaves to be noted after it returns: on the whole path its assembly
/// (timed by the path, starting where the whole run ended); the MicroModels' engine time per slot
/// (their own clocks give the length, not the start). The paths note their Core Brain and memory
/// steps themselves, with their clocks.
fn note_after_ask(o: &Organism, a: &crate::forage::Answer, engine_before: &std::collections::BTreeMap<String, f64>) {
    let whole = a.ran.iter().any(|r| r == "nmapi.whole");
    if whole {
        let run_end = Some(RUN_END.load(std::sync::atomic::Ordering::Relaxed)).filter(|t| *t > 0);
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "positive milliseconds")]
        let assembly = a
            .context_map
            .get("stages")
            .and_then(|s| s.get("assembly"))
            .ok()
            .and_then(|v| match v {
                Value::Number(t) => t.parse::<f64>().ok(),
                _ => None,
            })
            .map(|secs| (secs * 1000.0).round() as u64);
        note_step("nmapi.whole#assembly", run_end, assembly);
    }
    // A slot's engine time comes from its own clock (length only, no start); the reranker and the
    // embedder run every layer of their model.
    let bound = crate::neural::bound_models();
    for (id, us) in engine_spent(o, engine_before) {
        let role = id.strip_prefix("nmbrain.slot#").unwrap_or("");
        let layers = bound.iter().find(|(r, _)| *r == role).map(|(_, m)| (0, m.pop.dims.layers.saturating_sub(1) as u64));
        note(Step { part: id, ms: Some(us / 1000), us: Some(us), layers, ..Step::default() });
    }
}

/// Microseconds an engine spent per slot since `before` (the slot's own engine clock).
fn engine_spent(o: &Organism, before: &std::collections::BTreeMap<String, f64>) -> Vec<(String, u64)> {
    o.slots
        .engine_ms()
        .into_iter()
        .filter_map(|(id, ms)| {
            let spent = ms - before.get(&id).copied().unwrap_or(0.0);
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "positive microseconds")]
            (spent > 0.0).then(|| (id, (spent * 1000.0).round() as u64))
        })
        .collect()
}

/// The embedder engine runs beside the control (qualification record): the cosine between the
/// question and the first packet under the neural embedding; it decides nothing. With the dense
/// lane running the question is already embedded once by the lane, so the lane's own figures are
/// reported instead of embedding again (about 0.7 s).
fn embed_check(o: &Organism, q: &str, a: &crate::forage::Answer) -> Value {
    let lane = o.sub().dense.read().ok().and_then(|d| d.clone());
    if let Some(l) = lane {
        let st = l.stats.lock().map(|s| s.clone()).unwrap_or_default();
        #[expect(clippy::cast_precision_loss, reason = "counts for a report")]
        let per_query = if st.queries == 0 { 0.0 } else { st.query_ms / st.queries as f64 };
        obj(vec![
            ("dense_lane_vectors", n(st.vectors)),
            ("embedded_this_session", n(st.embedded_now)),
            ("query_ms_mean", Value::Number(format!("{per_query:.1}"))),
            // The lane's follow of the embedder (`neural::follow_embedder`): why its vectors are
            // stale (empty while current), questions run without them, and the last step taken.
            ("dense_lane_stale", s(&st.stale)),
            ("dense_lane_withheld", n(st.stale_skips)),
            ("dense_lane_follow", s(&crate::neural::dense_follow_line())),
        ])
    } else {
        match (o.slots.embed_neural(q), a.pack.packets.first().and_then(|p| o.slots.embed_neural(&format!("{} {}", p.key(), p.text())))) {
            (Some(x), Some(y)) => Value::Number(format!("{:.4}", x.iter().zip(&y).map(|(a, b)| a * b).sum::<f32>())),
            _ => Value::Null,
        }
    }
}

/// The diagnostics of one pack: parts that ran, memory figures, the context map, the best
/// reranker score (-1 when no engine ordered it), and the embed check when it was computed.
fn pack_diagnostics(o: &Organism, a: &crate::forage::Answer, embed: Option<Value>) -> Value {
    let mut parts: Vec<Value> = a.ran.iter().map(|r| s(r)).collect();
    parts.extend(["nmbrain.synthesis", "nmapi.assemble"].map(s));
    let mut f = vec![
        ("parts", Value::Array(parts)),
        ("withheld", Value::Bool(a.pack.withheld)),
        ("floor_met", Value::Bool(a.pack.floor_met)),
        ("episode", a.episode.as_deref().map_or(Value::Null, s)),
        ("memory_write", s(&a.written)),
        ("memory_recalled", n(a.recalled)),
        ("context_map", a.context_map.clone()),
        ("rerank_best", Value::Number(format!("{:.4}", o.slots.last_best()))),
    ];
    if let Some(e) = embed {
        f.push(("embed_check", e));
    }
    obj(f)
}

/// One reply's key: everything that shapes it (question with its spacing normalised, budget,
/// form, format, since, debug, mode).
type ReplyKey = Vec<String>;
/// Replies already given on the current code version, with the diagnostics they left. Cleared
/// when the code version changes.
type Replies = std::collections::BTreeMap<ReplyKey, (Value, Option<Value>)>;
static ANSWERS: std::sync::Mutex<(u64, Replies)> = std::sync::Mutex::new((u64::MAX, std::collections::BTreeMap::new()));
/// Replies kept per code version at most (all are dropped past it, then kept again).
const ANSWERS_CAP: usize = 256;

fn context_pack(v: &Value) -> Result<Value, ApiError> {
    clear_steps();
    // A stack switched since the last call reaches the sockets here, between calls.
    nmengn::filter::settle().map_err(|e| ApiError::new("nm-500", format!("cartridge sockets: {e}")))?;
    let o = org()?;
    // G3 and muscle memory (`serve --muscle on`, `crate::g3`): the call's allocation, its chain
    // when the stack serves, and its configuration record; with it off nothing is set or recorded.
    let call = crate::g3::begin(o, v);
    let input = call.as_ref().and_then(|c| c.input.as_ref()).unwrap_or(v);
    let out = context_pack_in(o, input, call.as_ref().map_or("", |c| c.allocation.as_str()));
    if let Some(c) = call {
        let diag = LAST.lock().ok().and_then(|d| d.clone());
        let cfg = crate::g3::end(o, c, out.as_ref().ok().and_then(|r| r.string().ok()), diag.as_ref());
        if let Ok(mut g) = LAST.lock()
            && let Some(Value::Object(d)) = g.as_mut()
        {
            d.insert("config".into(), cfg);
        }
    }
    out
}

fn context_pack_in(o: &'static Organism, v: &Value, allocation: &str) -> Result<Value, ApiError> {
    let mode = match arg(v, "mode") {
        Some(m) => Some(crate::whole::Mode::parse(m).map_err(|e| ApiError::new("nm-400", e))?),
        None => None,
    };
    let whole = o.whole.read().ok().and_then(|w| w.as_ref().map(|p| p.default_mode));
    let mode = whole.map(|default| mode.or(default).unwrap_or_else(|| crate::whole::mode_control(&o.home)));
    let q = arg(v, "question").unwrap_or_default();
    let version = o.sub().version;
    let default_pack_tokens = crate::whole::pack_tokens_control(&o.home);
    let key: ReplyKey = vec![
        q.split_whitespace().collect::<Vec<_>>().join(" "),
        v.get("tokens").and_then(Value::integer).map_or_else(|_| default_pack_tokens.to_string(), |t| t.to_string()),
        arg(v, "form").unwrap_or("").to_owned(),
        arg(v, "format").unwrap_or("").to_owned(),
        v.get("since").and_then(Value::integer).map_or(String::new(), |x| x.to_string()),
        matches!(v.get("debug"), Ok(Value::Bool(true))).to_string(),
        mode.map_or("plain", crate::whole::Mode::name).to_owned(),
        // The cartridges plugged into the MicroModels' sockets shape the reply too.
        nmengn::filter::generation().to_string(),
        allocation.to_owned(),
        // So does the preference cartridge in use.
        crate::prefs::generation().to_string(),
        // 2026-11-27: and the drones' intake while it is not complete (memory areas).
        crate::memory_areas::intake_mark(o),
    ];
    if let Ok(g) = ANSWERS.lock()
        && g.0 == version
        && !crate::g3::replaying()
        && let Some((reply, diag)) = g.1.get(&key)
    {
        let mut d = diag.clone().unwrap_or(Value::Object(std::collections::BTreeMap::new()));
        if let Value::Object(m) = &mut d {
            m.insert("answer_cache".into(), s("hit"));
        }
        set_diagnostics(d);
        return Ok(reply.clone());
    }
    let reply = context_pack_fresh(o, v, mode)?;
    if !crate::g3::replaying()
        && let Ok(mut g) = ANSWERS.lock()
    {
        if g.0 != version || g.1.len() >= ANSWERS_CAP {
            *g = (version, std::collections::BTreeMap::new());
        }
        let diag = LAST.lock().ok().and_then(|d| d.clone());
        g.1.insert(key, (reply.clone(), diag));
    }
    Ok(reply)
}

/// The line a fast reply carries: which mode produced it and what full mode adds.
pub const FAST_NOTE: &str = "mode: fast (the stack's first pieces and the dense lane's, judged by the reranker); call again with mode \"full\" for the whole path: more candidates, the code around the first place, callers and a second look";

fn context_pack_fresh(o: &Organism, v: &Value, mode: Option<crate::whole::Mode>) -> Result<Value, ApiError> {
    use crate::assemble::{Form, Format};
    let q = arg(v, "question").unwrap_or_default();
    let raw_tokens = v
        .get("tokens")
        .and_then(Value::integer)
        .ok()
        .and_then(|t| u32::try_from(t).ok())
        .unwrap_or_else(|| crate::whole::pack_tokens_control(&o.home));
    let quota_gate = crate::whole::evaluate_quota(&o.home, raw_tokens);
    let (tokens, mode) = match &quota_gate {
        crate::whole::QuotaGate::Normal => (raw_tokens, mode),
        crate::whole::QuotaGate::NearQuota { budget } => (*budget, mode),
        crate::whole::QuotaGate::Reached { action, budget, .. } => {
            let next_mode = if action == "fast" && mode.is_some() { Some(crate::whole::Mode::Fast) } else { mode };
            (*budget, next_mode)
        }
    };
    let form = Form::parse(arg(v, "form").unwrap_or("")).map_err(|e| ApiError::new("nm-400", e))?;
    let format = Format::parse(arg(v, "format").unwrap_or("")).map_err(|e| ApiError::new("nm-400", e))?;
    let debug = matches!(v.get("debug"), Ok(Value::Bool(true)));
    let since = v.get("since").and_then(Value::integer).ok();
    let engine_before = o.slots.engine_ms();
    let mut a = o.ask_in(q, tokens, mode).map_err(|e| fault(&e))?;
    if let crate::whole::QuotaGate::Reached { action, reason, .. } = &quota_gate
        && action == "withhold"
    {
        a.pack.withheld = true;
        a.pack.reason.clone_from(reason);
        a.pack.packets.clear();
        a.pack.tokens = 0;
        a.pack.floor_met = false;
    }
    if !a.pack.withheld {
        crate::whole::SESSION_PACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        crate::whole::SESSION_TOKENS.fetch_add(u64::from(a.pack.tokens), std::sync::atomic::Ordering::Relaxed);
    }
    let learned_guidance: Vec<String> = if o.memory_on && crate::whole::learning_control(&o.home) {
        let mdir = mem_dir(o);
        let eng = nmlearn::engine::ContinuousLearningEngine::open(&mdir);
        eng.active_guidance_for(q, "project")
    } else {
        Vec::new()
    };
    // Roadmap Phase 2 item 2: the preference cartridge in use, read at serve time.
    let pref_guidance = if a.pack.withheld { Vec::new() } else { crate::prefs::guidance(&a.pack.packets) };
    // The user's stated preferences whose scope fits this project and the cited files: the agent
    // follows them in what it writes. Evidence lines are never touched.
    let stated_block = if a.pack.withheld {
        Vec::new()
    } else {
        let root = &o.sub().root;
        let key = if nmtype::project::is_home(&o.home, root) { "nano".to_owned() } else { nmtype::project::key(root) };
        crate::prefs::stated_block(&a.pack.packets, Some(&key))
    };
    note_after_ask(o, &a, &engine_before);
    let fast = mode == Some(crate::whole::Mode::Fast);
    if format == Format::Text {
        let embed = debug.then(|| embed_check(o, q, &a));
        let diag = pack_diagnostics(o, &a, embed);
        let sub = o.sub();
        let mut text = timed_step("nmapi.assemble", || {
            crate::assemble::reply_text(q, &a, form, since, sub.version, &sub.files, debug.then_some(&diag))
        });
        // The mode line goes after the head line, so the agent sees it before any code.
        if fast && let Some(i) = text.find('\n') {
            text.insert_str(i + 1, &format!("{FAST_NOTE}\n"));
        }
        if !learned_guidance.is_empty()
            && let Some(i) = text.find('\n')
        {
            text.insert_str(i + 1, &format!("{}\n", learned_guidance.join("\n")));
        }
        if !pref_guidance.is_empty()
            && let Some(i) = text.find('\n')
        {
            text.insert_str(i + 1, &format!("{}\n", pref_guidance.join("\n")));
        }
        if !stated_block.is_empty()
            && let Some(i) = text.find('\n')
        {
            text.insert_str(i + 1, &format!("{}\n", stated_block.join("\n")));
        }
        set_diagnostics(diag);
        return Ok(Value::String(text));
    }
    if form != Form::Full || since.is_some() {
        set_diagnostics(pack_diagnostics(o, &a, None));
        return Ok(timed_step("nmapi.assemble", || crate::assemble::reply(q, &a, form, since, o.sub().version, &packet)));
    }
    let embed = embed_check(o, q, &a);
    set_diagnostics(pack_diagnostics(o, &a, Some(embed.clone())));
    let mut low = if a.pack.floor_met { Vec::new() } else { vec![("confidence", s("low")), ("floor_met", Value::Bool(false))] };
    if let Some(m) = mode {
        low.push(("mode", s(m.name())));
        if fast {
            low.push(("mode_note", s(FAST_NOTE)));
        }
    }
    if !learned_guidance.is_empty() {
        low.push(("learned_guidance", Value::Array(learned_guidance.iter().map(|g| s(g)).collect())));
    }
    if !pref_guidance.is_empty() {
        low.push(("preference_guidance", Value::Array(pref_guidance.iter().map(|g| s(g)).collect())));
    }
    if !stated_block.is_empty() {
        low.push(("preferences_in_force", Value::Array(stated_block.iter().map(|g| s(g)).collect())));
    }
    Ok(obj([
        low,
        vec![
            ("embed_check", embed),
            ("withheld", Value::Bool(a.pack.withheld)),
            ("reason", s(&a.pack.reason)),
            ("tokens", n(a.pack.tokens as usize)),
            ("answer", s(a.emitted.text())),
            ("packets", Value::Array(a.pack.packets.iter().map(packet).collect())),
            ("questions", Value::Array(a.questions.iter().map(|p| s(p.text())).collect())),
            ("episode", a.episode.as_deref().map_or(Value::Null, s)),
            ("memory_write", s(&a.written)),
            ("memory_recalled", n(a.recalled)),
            ("context_map", a.context_map.clone()),
            ("rerank_best", Value::Number(format!("{:.4}", o.slots.last_best()))),
        ],
    ]
    .concat()))
}
fn weights(v: &Value) -> Result<Value, ApiError> {
    // With `model`, any installed model's weight view, as the interface's Models page reads it.
    if arg(v, "model").is_some_and(|m| !m.is_empty()) {
        return crate::parity::weights_of(v);
    }
    let limit = v.get("limit").and_then(Value::integer).ok().and_then(|x| usize::try_from(x).ok()).unwrap_or(40);
    crate::neural::weights(&org()?.home, limit).map_err(|e| fault(&e))
}
fn repo_map(v: &Value) -> Result<Value, ApiError> {
    let sub = org()?.sub();
    let want = arg(v, "crate");
    let crates = sub.crates.iter().filter(|c| want.is_none_or(|w| sub.crate_named(w).is_some_and(|x| x.name == c.name)));
    Ok(obj(vec![
        ("root_digest", s(&sub.digest())),
        ("files", n(sub.files.len())),
        (
            "crates",
            Value::Array(
                crates
                    .map(|c| {
                        obj(vec![
                            ("name", s(&c.name)),
                            ("dir", s(&c.dir)),
                            ("ecosystem", s(c.ecosystem)),
                            ("manifest", s(&c.manifest)),
                            ("deps", Value::Array(c.deps.iter().map(|d| s(d)).collect())),
                        ])
                    })
                    .collect(),
            ),
        ),
    ]))
}
fn symbol_find(v: &Value) -> Result<Value, ApiError> {
    if let Ok(o) = org() {
        let _ = crate::whole::note_use(o, &v.encode().unwrap_or_default(), "nm_symbol_find");
    }
    let sub = org()?.sub();
    let name = arg(v, "name").unwrap_or_default();
    let mut hits: Vec<&nmsubs::repo::Symbol> = sub.symbols.iter().filter(|x| x.name == name).collect();
    if hits.is_empty() {
        hits = sub.symbols.iter().filter(|x| x.name.eq_ignore_ascii_case(name)).collect();
    }
    // `Class.method` (or `Class::method`): the member of that class or namespace.
    if hits.is_empty()
        && let Some((owner, member)) = name.rsplit_once("::").or_else(|| name.rsplit_once('.'))
    {
        hits = sub.symbols.iter().filter(|x| x.name.eq_ignore_ascii_case(member) && x.parent.eq_ignore_ascii_case(owner)).collect();
    }
    // Library source first, then tests, examples and benches.
    hits.sort_by_key(|x| (["/tests/", "/examples/", "/benches/"].iter().any(|t| x.file.contains(t)), x.file.clone(), x.line));
    Ok(obj(vec![(
        "definitions",
        Value::Array(
            hits.iter()
                .map(|x| {
                    obj(vec![
                        ("kind", s(&x.kind)),
                        ("file", s(&x.file)),
                        ("line", n(x.line)),
                        ("end", n(x.end)),
                        ("parent", s(&x.parent)),
                        ("text", s(&x.text)),
                    ])
                })
                .collect(),
        ),
    )]))
}
fn deps(v: &Value) -> Result<Value, ApiError> {
    if let Ok(o) = org() {
        let _ = crate::whole::note_use(o, &v.encode().unwrap_or_default(), "nm_deps");
    }
    let sub = org()?.sub();
    let inward = arg(v, "direction") == Some("in");
    // A file: the files it imports (out) or the files importing it (in), and outside packages.
    if let Some(file) = arg(v, "file") {
        let file = file.replace('\\', "/");
        let edges: Vec<&nmsubs::imports::Edge> =
            sub.import_edges.iter().filter(|e| if inward { e.to.as_deref() == Some(file.as_str()) } else { e.from == file }).collect();
        if edges.is_empty() && !sub.files.iter().any(|f| f.path == file) {
            return Err(ApiError::new("nm-404", format!("no file {file} in the studied repository")));
        }
        let mut files: Vec<String> = edges.iter().filter_map(|e| if inward { Some(e.from.clone()) } else { e.to.clone() }).collect();
        files.sort();
        files.dedup();
        let mut outside: Vec<String> = if inward { Vec::new() } else { edges.iter().filter_map(|e| e.external.clone()).collect() };
        outside.sort();
        outside.dedup();
        let sources: Vec<Value> = edges.iter().map(|e| s(&format!("{}:{} {}", e.from, e.line, e.spec))).collect();
        return Ok(obj(vec![
            ("file", s(&file)),
            ("direction", s(if inward { "in" } else { "out" })),
            ("files", Value::Array(files.iter().map(|d| s(d)).collect())),
            ("outside", Value::Array(outside.iter().map(|d| s(d)).collect())),
            ("sources", Value::Array(sources)),
        ]));
    }
    let name = arg(v, "crate").or_else(|| arg(v, "package")).unwrap_or_default();
    if name.is_empty() {
        return Err(ApiError::new("nm-400", "nm_deps needs a crate (or package) name, or a file"));
    }
    let c = sub.crate_named(name).ok_or_else(|| ApiError::new("nm-404", format!("no crate or package {name} in the studied repository")))?;
    let list = if inward { sub.dependents(&c.name) } else { c.deps.clone() };
    Ok(obj(vec![
        ("crate", s(&c.name)),
        ("ecosystem", s(c.ecosystem)),
        ("direction", s(arg(v, "direction").unwrap_or("out"))),
        ("crates", Value::Array(list.iter().map(|d| s(d)).collect())),
        ("source", s(&c.manifest)),
    ]))
}
fn view(v: &Value) -> Result<Value, ApiError> {
    if let Ok(o) = org() {
        let _ = crate::whole::note_use(o, &v.encode().unwrap_or_default(), "nm_view");
    }
    let sub = org()?.sub();
    let (name, q) = (arg(v, "view").unwrap_or("global"), arg(v, "query").unwrap_or_default());
    let packets = match arg(v, "zoom") {
        Some(z) => {
            let level = match z {
                "symbol" => nmsubs::zoom::Level::Symbol,
                "file" => nmsubs::zoom::Level::File,
                "crate" => nmsubs::zoom::Level::Crate,
                _ => nmsubs::zoom::Level::Workspace,
            };
            sub.zoom(q, level)
        }
        None => SubstrateView::view(sub.as_ref(), name, q),
    };
    Ok(obj(vec![("view", s(name)), ("packets", Value::Array(packets.iter().map(packet).collect()))]))
}
fn sync(_: &Value) -> Result<Value, ApiError> {
    let (r, inv) = org()?.sync().map_err(|e| fault(&e))?;
    let (own, outside) = org()?.last_sync_origin.lock().map(|g| g.clone()).unwrap_or_default();
    Ok(obj(vec![
        ("version", Value::Number(r.version.to_string())),
        ("changed", n(r.changed.len())),
        ("added", n(r.added.len())),
        ("removed", n(r.removed.len())),
        ("invalidations", n(inv.len())),
        ("own_changes", n(own.len())),
        ("outside_changes", n(outside.len())),
    ]))
}
fn goal(v: &Value) -> Result<Value, ApiError> {
    match arg(v, "op") {
        // 2026-11-26 (Phase B): one path with the interface's `/api/goals` (`memory_write`).
        None | Some("goal") => crate::memory_write::goal_on(org()?, v),
        // The reminders, loops and step graph that guide the agent, through the interface's
        // `/api/guidance` (`crate::parity::guidance`), on the project this process serves.
        Some(op) if crate::parity::GUIDANCE_OPS.contains(&op) => {
            let served = installed().map_or_else(|| "nano".to_owned(), served_key);
            crate::parity::guidance(v, &served)
        }
        Some(op) => Err(ApiError::new("nm-400", format!("op {op}: use goal (the default) or {}", crate::parity::GUIDANCE_OPS.join(", ")))),
    }
}

/// The key `/api/projects` names the served project by: `nano` for the Nano's own folder, else
/// its memory folder's name.
fn served_key(o: &Organism) -> String {
    let dir = mem_dir(o);
    if dir == o.home.join("memory") {
        return "nano".to_owned();
    }
    dir.file_name().map_or_else(|| "nano".to_owned(), |f| f.to_string_lossy().into_owned())
}

fn memory(v: &Value) -> Result<Value, ApiError> {
    let o = org()?;
    let mut m = o.memory.lock().map_err(|_| ApiError::new("nm-500", "memory lock"))?;
    let text = arg(v, "text").unwrap_or_default();
    let op = arg(v, "op").unwrap_or("recall");
    // Memory off: only read-only views; anything that recalls, counts a use or saves is refused.
    if !o.memory_on && !matches!(op, "note" | "temp_list" | "hierarchy" | "identity" | "tiers" | "health" | "record" | "recall2") {
        return Err(ApiError::new("nm-409", "memory is off for this session (serve --memory off): nothing is recalled or written"));
    }
    match op {
        // 2026-11-26 (Phase B): the writes run one path with the interface's `/api/memory/*`
        // routes (`memory_write`); a stored claim admitted by the gate is now kept as a memory.
        "note" | "store" | "temp" | "sweep" | "pin" | "unpin" => crate::memory_write::memory_op(&mut m, op, v),
        "temp_list" => Ok(Value::Array(
            m.temp
                .list()
                .iter()
                .map(|t| {
                    obj(vec![("id", s(&t.id)), ("scope", s(&t.scope)), ("text", s(&t.text)), ("uses", Value::Number(t.uses.to_string()))])
                })
                .collect(),
        )),
        "hierarchy" => Ok(obj(vec![("levels", levels(&m.hierarchy)), ("file", s("hierarchy.json")), ("tiers", s("index.json"))])),
        "identity" => Ok(m.identity.to_value()),
        // Memory revision 2 views and owner controls.
        "health" => Ok(m.mem2.as_ref().map_or_else(
            || obj(vec![("revision_2", Value::Bool(false)), ("why", m.mem2_fault.as_deref().map_or(Value::Null, s))]),
            nmmem::mem2::Memory2::health,
        )),
        "record" => {
            let m2 = m.mem2.as_ref().ok_or_else(|| ApiError::new("nm-409", "memory revision 2 is not open"))?;
            let id = m2.resolve(text).map_err(|e| fault(&e))?;
            Ok(m2.book.records.get(&id).map_or(Value::Null, nmmem::record::Record::to_value))
        }
        "recall2" => {
            // Revision 2 recall (words, links, retention), read only: shown beside revision 1.
            let m2 = m.mem2.as_ref().ok_or_else(|| ApiError::new("nm-409", "memory revision 2 is not open"))?;
            let hits = nmmem::recall2::recall(&m2.book, text, m2.day, &std::collections::BTreeMap::new(), 5);
            Ok(Value::Array(
                hits.iter()
                    .map(|h| {
                        obj(vec![
                            ("id", s(&h.id)),
                            ("score", s(&format!("{:.4}", h.score))),
                            ("by", Value::Array(h.by.iter().map(|b| s(b)).collect())),
                        ])
                    })
                    .collect(),
            ))
        }
        "undo" => {
            let m2 = m.mem2.as_mut().ok_or_else(|| ApiError::new("nm-409", "memory revision 2 is not open"))?;
            let seq = text.trim().parse::<u64>().map_err(|_| ApiError::new("nm-400", "undo needs a journal sequence number in text"))?;
            m2.undo(seq).map_err(|e| fault(&e))?;
            Ok(obj(vec![("undone", Value::Number(seq.to_string()))]))
        }
        "tiers" => {
            let ram = nmffi::dynlib::system_memory().ok().map(|(available, _)| available);
            let use_ = nmmem::tiers::usage(&m.dir, m.working.bytes(), ram);
            Ok(Value::Array(
                use_.iter()
                    .map(|t| {
                        obj(vec![
                            ("tier", s(t.tier)),
                            ("bytes", Value::Number(t.bytes.to_string())),
                            ("budget", t.budget.map_or(Value::Null, |b| Value::Number(b.to_string()))),
                            ("over", Value::Bool(t.over())),
                        ])
                    })
                    .collect(),
            ))
        }
        _ => {
            // A temp note recalled here counts as one use toward its promotion.
            let temp = m.temp.recall(text, 3).unwrap_or_default();
            let saved = m.hierarchy.recall(text, 5).unwrap_or_default();
            let eps = m.episodic.recall(text, 5);
            Ok(obj(vec![
                (
                    "saved",
                    Value::Array(
                        saved
                            .iter()
                            .map(|i| {
                                obj(vec![
                                    ("id", s(&i.id)),
                                    ("level", s(nmmem::hierarchy::LEVELS.get(i.level).copied().unwrap_or(""))),
                                    ("topic", s(&i.topic)),
                                    ("text", s(&i.text)),
                                    ("uses", Value::Number(i.uses.to_string())),
                                ])
                            })
                            .collect(),
                    ),
                ),
                (
                    "temp",
                    Value::Array(
                        temp.iter()
                            .map(|t| obj(vec![("id", s(&t.id)), ("text", s(&t.text)), ("uses", Value::Number(t.uses.to_string()))]))
                            .collect(),
                    ),
                ),
                ("episodes_total", n(m.episodic.len())),
                (
                    "recalled",
                    Value::Array(
                        eps.iter()
                            .map(|e| {
                                obj(vec![
                                    ("id", s(&e.id)),
                                    ("query", s(&e.query)),
                                    ("correct", e.correct.map_or(Value::Null, Value::Bool)),
                                    ("answer", s(e.answer.lines().next().unwrap_or(""))),
                                ])
                            })
                            .collect(),
                    ),
                ),
                ("refused_writes", n(m.gate.refused)),
            ]))
        }
    }
}
fn unit(v: &Value) -> Result<Value, ApiError> {
    let o = org()?;
    crate::learn::tool(o, v).map_err(|e| fault(&e))
}
fn feedback(v: &Value) -> Result<Value, ApiError> {
    let o = org()?;
    let id = arg(v, "episode").unwrap_or_default();
    let correct = matches!(v.get("correct"), Ok(Value::Bool(true)));
    if !o.memory_on {
        return Err(ApiError::new("nm-409", "memory is off for this session (serve --memory off): no verdict is written"));
    }
    let mut m = o.memory.lock().map_err(|_| ApiError::new("nm-500", "memory lock"))?;
    let ep_info = m.episodic.latest(usize::MAX).into_iter().find(|e| e.id == id).map(|e| (e.query.clone(), e.evidence.clone()));
    m.episodic.verdict(id, correct).map_err(|e| fault(&e))?;
    if let Some(m2) = m.mem2.as_mut() {
        m2.verdict(id, correct, "nm_feedback").map_err(|e| fault(&e))?;
    }
    let mdir = m.dir.clone();
    // The checked outcome is also the training data path for later fits (S8).
    let line = obj(vec![("at", Value::Number(nmtype::now_ms().to_string())), ("episode", s(id)), ("correct", Value::Bool(correct))])
        .encode()
        .unwrap_or_default();
    let path = m.dir.join("outcomes.jsonl");
    // The verdict also joins the decision ledger by episode (continuous learning step 2).
    let _ = nmlearn::ledger::append_verdict(&crate::whole::ledger_path(&o.home, &m.dir), id, correct, nmtype::now_ms());
    let ok = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut f| std::io::Write::write_all(&mut f, format!("{line}\n").as_bytes()))
        .is_ok();
    drop(m);
    if crate::whole::learning_control(&o.home) {
        let mut eng = nmlearn::engine::ContinuousLearningEngine::open(&mdir);
        let (query, places) = ep_info.unwrap_or_else(|| (id.to_owned(), Vec::new()));
        let cands: Vec<String> = nmlex::key_terms(&query).into_iter().take(4).collect();
        let exp_type = if correct {
            nmlearn::engine::ExperienceType::SuccessfulOutcome
        } else {
            nmlearn::engine::ExperienceType::FailedOutcome
        };
        let rec = nmlearn::engine::ExperienceRecord {
            experience_id: format!("exp_{id}"),
            timestamp_ms: nmtype::now_ms(),
            experience_type: exp_type,
            signal: nmlearn::engine::FeedbackSignal::Explicit,
            task: "context_verification".to_owned(),
            scope: "project".to_owned(),
            context: nmlearn::engine::ExperienceContext {
                problem: query,
                initial_approach: "nm_context_pack".to_owned(),
                correction: arg(v, "note").unwrap_or(if correct { "verified answer" } else { "refuted answer" }).to_owned(),
                places,
                provenance: "nm_feedback".to_owned(),
            },
            outcome: nmlearn::engine::ExperienceOutcome {
                initial_success: correct,
                corrected_success: correct,
                delayed_updated: false,
                tokens_used: 0,
                ms_used: 0,
            },
            learning_candidates: if cands.is_empty() { vec!["context_verification".to_owned()] } else { cands },
            confidence: if correct { 0.92 } else { 0.85 },
            status: nmlearn::engine::SupervisionState::Observed,
            occurrences: 1,
        };
        // Cycle A only (2026-11-25): the consolidation cycle runs through the gated, recorded idle
        // cycle (held-out gate, audit row, rollback) when enough verified evidence has settled in
        // `auto` mode, or when the user asks; never ungated here on each verdict.
        let _ = eng.observe(rec);
        let _ = eng.save(&mdir);
    }
    // Core Brain side (`--brain-chains`, 2026-11-10): outcome to the behaviour log, verdict to
    // the preference feed, verdicts to the cartridge track record. Off: the reply is unchanged.
    if let Some(side) = crate::brain_side::side(o) {
        let taste = match arg(v, "taste") {
            Some("accept") => side.on_taste(id, true),
            Some("reject") => side.on_taste(id, false),
            _ => Value::Null,
        };
        return Ok(obj(vec![
            ("episode", s(id)),
            ("correct", Value::Bool(correct)),
            ("outcome_logged", Value::Bool(ok)),
            ("brain_chains", side.on_feedback(id, correct)),
            ("taste", taste),
        ]));
    }
    Ok(obj(vec![("episode", s(id)), ("correct", Value::Bool(correct)), ("outcome_logged", Value::Bool(ok))]))
}

fn levels(h: &nmmem::Hierarchy) -> Value {
    let c = h.counts();
    obj(nmmem::hierarchy::LEVELS.iter().zip(c).map(|(l, k)| (*l, n(k))).collect())
}
fn mem_dir(o: &Organism) -> std::path::PathBuf {
    o.memory.lock().map_or_else(|_| o.home.join("memory"), |m| m.dir.clone())
}
fn build_run(v: &Value) -> Result<Value, ApiError> {
    let o = org()?;
    let r = crate::build::tool_run(&o.home, &o.sub().root, &mem_dir(o), v);
    o.reload_operational();
    r
}
fn build_explain(v: &Value) -> Result<Value, ApiError> {
    let o = org()?;
    crate::build::tool_explain(&o.home, &o.sub().root, &mem_dir(o), v)
}
fn improve_scan(v: &Value) -> Result<Value, ApiError> {
    let o = org()?;
    crate::build::tool_improve_scan(&o.home, &o.sub().root, v)
}
fn patch_propose(v: &Value) -> Result<Value, ApiError> {
    if let Ok(o) = org() {
        let _ = crate::whole::note_use(o, &v.encode().unwrap_or_default(), "nm_patch_propose");
    }
    let o = org()?;
    crate::build::tool_patch_propose(&mem_dir(o), v)
}
/// `nm_verify` (2026-11-10): the verification dimensions of a diff, read from the diff and the
/// studied tree before anything runs (`nmtool::verify`).
fn verify(v: &Value) -> Result<Value, ApiError> {
    let diff = arg(v, "diff").ok_or_else(|| ApiError::new("nm-400", "`diff` (unified diff against the studied tree) is required"))?;
    let sub = org()?.sub();
    let files: Vec<(String, String)> = sub.files.iter().map(|f| (f.path.clone(), f.text.clone())).collect();
    Ok(nmtool::verify::to_value(&nmtool::verify::verify(diff, &files)))
}
fn stepcheck(v: &Value) -> Result<Value, ApiError> {
    if let Ok(o) = org() {
        let _ = crate::whole::note_use(o, &v.encode().unwrap_or_default(), "nm_stepcheck");
    }
    let o = org()?;
    let r = crate::build::tool_stepcheck(&o.home, &o.sub().root, &mem_dir(o), v);
    if let (Ok(out), Ok(mut led)) = (r.as_ref(), o.own.lock())
        && let Ok(list) = out.get("predicted").and_then(Value::array)
    {
        for p in list {
            if let (Ok(f), Ok(h)) = (p.get("file").and_then(Value::string), p.get("sha256").and_then(Value::string)) {
                led.predict(f, h);
            }
        }
    }
    o.reload_operational();
    r
}

const fn field(name: &'static str, kind: FieldKind, required: bool, doc: &'static str) -> Field {
    Field { name, kind, required, doc, default: None }
}

const IN_ANATOMY: &[Field] = &[
    field("node", FieldKind::String, false, "node id to focus on"),
    field("level", FieldKind::String, false, "anatomy zoom: organism, slot, structure, component or cluster"),
    field("id", FieldKind::String, false, "component or cluster id for those levels"),
];
const IN_WEIGHTS: &[Field] = &[
    field("limit", FieldKind::Integer, false, "clusters listed, default 40 (counts are always complete)"),
    field("model", FieldKind::String, false, "an installed model id (as nm_settings op=roles lists them): that model's weight card, tensors and clusters; without it, the served base model"),
];
const IN_PACK: &[Field] = &[
    field("question", FieldKind::String, true, "the question"),
    field("tokens", FieldKind::Integer, false, "token budget").defaulting(crate::drone_path::PACK_TOKENS),
    field("form", FieldKind::String, false, "adaptive (default), answer, cited or full"),
    field("since", FieldKind::Integer, false, "substrate version of an earlier reply: only newer packets come back"),
    field(
        "format",
        FieldKind::String,
        false,
        "text (default) or json",
    ),
    field("debug", FieldKind::Bool, false, "add the diagnostics (parts that ran, context map, memory figures) to a text reply"),
    field("mode", FieldKind::String, false, "fast (seconds) or full (the whole path); default from the mode control"),
];
const IN_MAP: &[Field] = &[field("crate", FieldKind::String, false, "one crate or package")];
const IN_SYMBOL: &[Field] = &[field("name", FieldKind::String, true, "symbol name, or Class.method for a member")];
const IN_DEPS: &[Field] =
    &[
        field("crate", FieldKind::String, false, "crate or package name (Cargo, npm, Python)"),
        field("package", FieldKind::String, false, "same as crate"),
        field("file", FieldKind::String, false, "a file: the files it imports (out) or that import it (in)"),
        field("direction", FieldKind::String, false, "out (default) or in"),
    ];
const IN_VIEW: &[Field] = &[
    field("view", FieldKind::String, false, "view name"),
    field("query", FieldKind::String, false, "query or subject"),
    field("zoom", FieldKind::String, false, "symbol, file, crate or workspace"),
];
const IN_MEMORY: &[Field] = &[
    field(
        "op",
        FieldKind::String,
        false,
        "recall (default), note, temp, temp_list, sweep, hierarchy, store, identity, tiers, health, record, recall2, pin, unpin or undo",
    ),
    field("text", FieldKind::String, false, "query, note, claim, record id or journal sequence number"),
    field("scope", FieldKind::String, false, "working scope for a note"),
    field("source", FieldKind::String, false, "source of a claim to store"),
];
const IN_FEEDBACK: &[Field] = &[
    field("episode", FieldKind::String, true, "episode id from nm_context_pack"),
    field("correct", FieldKind::Bool, true, "was the answer right"),
    field(
        "taste",
        FieldKind::String,
        false,
        "accept or reject: your taste for the answer's form",
    ),
    field(
        "note",
        FieldKind::String,
        false,
        "optional correction or observation note for continuous learning",
    ),
];
const IN_GOAL: &[Field] = &[
    field(
        "op",
        FieldKind::String,
        false,
        "goal (default: the objectives), guidance (reminders, loops, steps, recent deliveries with the evidence ledger's verdict, and the outcome rows of ended loops), reminder_add, reminder_list, reminder_edit, loop_add, loop_edit, loop_end, loop_outcomes (the ended loops, newest first), step_add, step_done, step_edit, remove (two calls: a preview with a confirm_token, then the same call with it), enable, disable or reset",
    ),
    field("objective", FieldKind::String, false, "objective text to add"),
    field("id", FieldKind::Integer, false, "id of an objective to close or open again"),
    field("status", FieldKind::String, false, "with id: done (default), dropped or open"),
    field("item", FieldKind::String, false, "the reminder, loop or step id (r1, l2, s3) to edit, end, mark done, remove, switch or reset; step_add: an id for the new step (optional)"),
    field("kind", FieldKind::String, false, "remove, enable, disable, reset: reminder, loop or step"),
    field("text", FieldKind::String, false, "reminder_add, reminder_edit: the reminder's words"),
    field("trigger", FieldKind::String, false, "reminder: session-start, prompt, compact, stop, edit, failure, tools or error-repeats (the same recorded error in count tool results)"),
    field("error", FieldKind::String, false, "reminder with trigger error-repeats: the error code (E0308) or a piece of its message, as the hook records it"),
    field("count", FieldKind::Integer, false, "reminder with trigger error-repeats: tool results carrying the error before it fires (2 to 5, default 2)"),
    field("every", FieldKind::Integer, false, "reminder with trigger prompt or tools: every Nth one (default 1)"),
    field("glob", FieldKind::String, false, "reminder with trigger edit: only edits of paths matching this glob"),
    field("repeat", FieldKind::String, false, "reminder: once, always (default) or max (at most max times per session)"),
    field("max", FieldKind::Integer, false, "reminder with repeat max: times per session; loop: most iterations (default 3, at most 20)"),
    field("files", FieldKind::String, false, "reminder: a file scope glob (the event's or the session's files must match)"),
    field("prompt", FieldKind::String, false, "loop_add: what to do while the condition is unmet (given at each continued stop)"),
    field("until", FieldKind::String, false, "loop_add: check-passes, goal-done, steps-done, manual, all-of or any-of (with parts)"),
    field("parts", FieldKind::Array, false, "loop with until all-of or any-of: 2 to 4 distinct conditions (check-passes, goal-done, steps-done, manual); all_of or any_of with the list also sets until"),
    field("all_of", FieldKind::Array, false, "loop: 2 to 4 conditions that must all be met (sets until all-of)"),
    field("any_of", FieldKind::Array, false, "loop: 2 to 4 conditions of which one must be met (sets until any-of)"),
    field("stall", FieldKind::Integer, false, "loop: the same unmet reason this many times in a row ends it stalled (1 to 10, default 2; a manual loop does not stall)"),
    field("budget_ms", FieldKind::Integer, false, "loop: wall time in ms since it was set after which it ends budget-spent (1000 to 86400000; 0 clears)"),
    field("budget_tools", FieldKind::Integer, false, "loop: tool calls since it was set after which it ends budget-spent (1 to 10000; 0 clears)"),
    field("limit", FieldKind::Integer, false, "loop_outcomes: how many ended loops, newest first (default 20, at most 200)"),
    field("goal", FieldKind::Integer, false, "loop goal-done: the objective's id; loop steps-done or a step: the objective the steps belong to"),
    field("title", FieldKind::String, false, "step_add, step_edit: what the step is"),
    field("depends_on", FieldKind::Array, false, "step_add, step_edit: the ids of the steps it waits for, all of them (an AND-join; a cycle is refused)"),
    field("depends_any", FieldKind::Array, false, "step_add, step_edit: an OR-join: at least one of these steps must be done as well (a cycle over both lists is refused)"),
    field("probe", FieldKind::Bool, false, "step with a check: true makes it a probe, done when its check passes while its dependencies are met, with no edit needed"),
    field("check", FieldKind::String, false, "step: build or test must pass after edits for the Nano to mark it done; none to clear"),
    field("paths", FieldKind::Array, false, "step: globs the edits must be under for its check"),
    field("enabled", FieldKind::Bool, false, "reminder or loop: on or off when added or edited"),
    field("project", FieldKind::String, false, "project key (default: the project this server serves)"),
    field("confirm_token", FieldKind::String, false, "remove's second call: the token its preview returned"),
];
const IN_UNIT: &[Field] = &[
    field("op", FieldKind::String, false, "list (default), register, unregister, load or unload"),
    field("id", FieldKind::String, false, "unit id"),
    field("target", FieldKind::String, false, "component node the unit stacks on (register)"),
    field("slot", FieldKind::String, false, "slot the unit serves (register; default reranker)"),
    field("rank", FieldKind::Integer, false, "adapter rank (register; default 8)"),
    field("seed", FieldKind::Integer, false, "seed for the fixed down factor (register; default 8)"),
];

const IN_BUILD: &[Field] = &[
    field("package", FieldKind::String, false, "one package (-p); the whole workspace when absent"),
    field("kind", FieldKind::String, false, "check (default), clippy, test or build"),
    field("warnings", FieldKind::Bool, false, "list warnings too (default: counted only)"),
    field("budget", FieldKind::Integer, false, "items listed, default 12"),
    field("form", FieldKind::String, false, "lines (default: one line per error with its source line and help) or json (the item list)"),
];
const IN_EXPLAIN: &[Field] = &[
    field("run", FieldKind::String, false, "run id from nm_build_run"),
    field("package", FieldKind::String, false, "run a fresh check of this package and explain it"),
    field("kind", FieldKind::String, false, "kind for a fresh run"),
];
const IN_IMPROVE: &[Field] = &[
    field("filter", FieldKind::String, false, "only paths containing this text"),
    field("package", FieldKind::String, false, "also collect clippy machine-applicable suggestions for this package"),
    field("budget", FieldKind::Integer, false, "items listed, default 20"),
];
const IN_PROPOSE: &[Field] = &[field("index", FieldKind::Integer, true, "finding index from the last nm_improve_scan")];
const IN_VERIFY: &[Field] = &[
    field("diff", FieldKind::String, true, "unified diff against the studied tree"),
    field("episode", FieldKind::String, false, "episode id from nm_context_pack this diff acts on: the findings are logged as its outcome"),
];
const IN_STEPCHECK: &[Field] = &[
    field("diff", FieldKind::String, true, "unified diff against the studied tree"),
    field("package", FieldKind::String, false, "limit check, clippy and tests to one package"),
    field("tests", FieldKind::Bool, false, "also run tests"),
];

/// The tool table.
#[must_use]
pub fn api() -> Api {
    let op = |name, summary, inputs, handler| Operation { name, summary, inputs, handler };
    Api::default()
        .with(op("nm_status", "Returns the server's state: the folder it reads, its version, file and symbol counts. Call it to check the server is up.", &[], status))
        .with(op("nm_live", "Returns what the Nano is doing now: active stages, MicroModels executing, card memory, the idle writer and recent events. Call it only to watch the Nano itself.", &[], live))
        .with(op("nm_anatomy", "Returns the Nano's own running parts. Call it only to debug the Nano, never for questions about the material.", IN_ANATOMY, anatomy))
        .with(op(
            "nm_context_pack",
            "Searches this project's whole folder (code, documents, notes, tables) for a question and returns the lines that answer it, each cited as path:first-last; low or withheld when the folder lacks the answer. Call it first for every question about the project, before Grep, Glob or Read, then read only the cited ranges you still need.",
            IN_PACK,
            context_pack,
        ))
        .with(op(
            "nm_repo_map",
            "Returns the folder's packages (Cargo, npm, Python) and which depend on which. Call it for the layout of an unfamiliar repository.",
            IN_MAP,
            repo_map,
        ))
        .with(op(
            "nm_symbol_find",
            "Returns where a named symbol is defined: kind, file, lines, enclosing class, defining text. Call it when you know the exact name.",
            IN_SYMBOL,
            symbol_find,
        ))
        .with(op(
            "nm_deps",
            "Returns what a package or file depends on (out) or what depends on it (in). Call it before changing a package or file.",
            IN_DEPS,
            deps,
        ))
        .with(op(
            "nm_view",
            "Returns one view of the folder (an outline, a file's history, and others) or a zoom at symbol, file, crate or workspace level. Call it when a pack is too broad.",
            IN_VIEW,
            view,
        ))
        .with(op("nm_sync", "Re-reads changed files and returns what changed. Call it after editing files when a later answer must see the edits.", &[], sync))
        .with(op("nm_memory", "Recalls or saves this project's notes and earlier answers. Call it to keep or look up a fact that should outlive the session.", IN_MEMORY, memory))
        .with(op("nm_feedback", "Records whether an earlier nm_context_pack answer was right, by its episode id. Call it after checking a pack against the files.", IN_FEEDBACK, feedback))
        .with(op("nm_goal", "Sets, closes or returns the session's objectives, and the reminders, loops and step graph that guide you (op). Call it to say what the session is working towards, to set yourself a reminder for a later prompt, stop or edit, to set a loop that continues you at a stop until a check passes, a goal closes or every step is done (capped), or to lay out steps and mark them done.", IN_GOAL, goal))
        .with(op("nm_weights", "Returns the model weights the Nano reads (tensors, shapes, precision). Call it only to inspect the Nano itself.", IN_WEIGHTS, weights))
        .with(op("nm_build_run", "Runs one cargo check, clippy, test or build on a scratch copy and returns the errors as short cited lines. Call it to check a Rust change.", IN_BUILD, build_run))
        .with(op("nm_build_explain", "Returns the first error of the last build with its lines, cause and fix. Call it after nm_build_run reports an error.", IN_EXPLAIN, build_explain))
        .with(op("nm_improve_scan", "Returns concrete defects (rule words, long files, clippy fixes); applies nothing. Call it when asked for cleanups.", IN_IMPROVE, improve_scan))
        .with(op("nm_patch_propose", "Returns the diff for one finding of the last nm_improve_scan; applies nothing. Call it to turn a finding into a patch.", IN_PROPOSE, patch_propose))
        .with(op("nm_verify", "Returns what a diff touches (interfaces, manifests, tests, rule words) before it runs. Call it on a diff before applying it.", IN_VERIFY, verify))
        .with(op("nm_stepcheck", "Applies a diff to a scratch copy and returns check, clippy, test, size and rule results; the source is untouched. Call it to prove a diff builds.", IN_STEPCHECK, stepcheck))
        .with(op("nm_unit", "Lists, registers, loads or unloads learning units. Call it only to manage the Nano's own learning units.", IN_UNIT, unit))
        // Roadmap Phase 4.5 (2026-11-24): the web interface's inspections and controls, one for
        // one, through the same functions (`crate::parity`); on demand, not in the lean set.
        .with(op("nm_storage", "Returns the Nano's disk use by category with exact paths, or prunes rebuildable caches, deletes one removable item (confirm: true) or sets the disk floor. Call it only to manage the Nano's own storage.", crate::parity::IN_STORAGE, crate::parity::storage))
        .with(op("nm_cartridges", "Lists cartridges and job filters (each with the base it was built on and whether that base serves now), switches a cartridge or use on or off, builds a job filter through the split gate or removes one (confirm: true), maps a MicroModel's parameters and cuts a base cartridge version from the map (map_build), or builds a base cartridge version by restoration (base_build; new versions are kept beside the others by default), or trains curated teacher pairs into a learned cartridge of the chosen kind for a MicroModel through the held-out gate (distil_build, distils). Call it only to manage the Nano's own cartridges.", crate::parity::IN_CARTRIDGES, crate::parity::cartridges))
        .with(op("nm_learning", "Returns a project's learning state, examples, ledger and outcomes, runs one learning cycle, or rolls one back or clears a store (confirm: true); turns the user's stated preferences (their words, passed as they said them) into a preference cartridge of rules (preference_say: a preview, then confirm: true), lists, edits, plugs or removes those cartridges, and checks a draft against the rules in force (preference_apply); lists, imports and labels the teacher data for distillation (distil, distil_import, distil_label). Call it only to manage the Nano's own learning and the user's stated preferences.", crate::parity::IN_LEARNING, crate::parity::learning))
        .with(op("nm_comprehension", "Returns a project's 4-level world model (piece, file, module, system notes), routing record, lens heads and memory tiers. Call it only to inspect the Nano's own comprehension.", crate::parity::IN_COMPREHENSION, crate::parity::comprehension))
        .with(op("nm_settings", "Returns or sets the Nano's controls (each change logged and undoable), its hardware limits and this machine's recommended values, MicroModel and card status, projects and the audit log. Call it only to manage the Nano's own settings.", crate::parity::IN_SETTINGS, crate::parity::settings))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_diagnostics_reports_slots_last_best_and_unscored() {
        let dir = nmtype::check_dir("nmapi-tools-rerank-best").unwrap();
        let tree = dir.join("tree");
        std::fs::create_dir_all(&tree).unwrap();
        std::fs::write(tree.join("lib.rs"), "pub fn item() -> u32 { 1 }\n").unwrap();
        let org = Organism::open_with(&nmtype::nano_root(), &tree, &[], false, &dir.join("mem")).unwrap();
        let pk = nmtype::Pack {
            task: 1,
            packets: Vec::new(),
            withheld: false,
            reason: "whole path".to_owned(),
            tokens: 0,
            floor_met: true,
        };
        let emitted = nmtype::Packet::new("answer", "answer:1", "", nmtype::Provenance::new("answer:1", "nmbrain", 1, 1));
        let ans = crate::forage::Answer::for_check(pk, emitted);

        org.slots.note_best(0.1981);
        let d1 = pack_diagnostics(&org, &ans, None);
        let best1 = d1.get("rerank_best").unwrap().encode().unwrap();
        assert_eq!(best1, "0.1981");

        org.slots.note_best(-1.0);
        let d2 = pack_diagnostics(&org, &ans, None);
        let best2 = d2.get("rerank_best").unwrap().encode().unwrap();
        assert_eq!(best2, "-1.0000");
    }
}
