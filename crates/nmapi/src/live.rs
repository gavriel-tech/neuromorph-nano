//! Roadmap Phase 4.5 (2026-11-24): the live activity recorder of this process. A bounded ring of
//! behavioural events and a "now" state, fed only by hooks where the work actually happens:
//!
//! - the stages of a question (`G0` intake and routing, `G1` search with its dense lane and tree
//!   prior, `G2` with the tentacles' head reads and the close-duel judge, `G3` pack assembly), from
//!   `drone_path`, and each served tool call from `mcp`;
//! - which MicroModel is executing now: every forward of a bound model passes the engine's hook
//!   (`nmengn::model::EXEC_HOOK`), installed here on the first bind;
//! - the idle writer (`writer_idle`): seated, session, results by level, queue depth, preemption,
//!   release;
//! - the dense lane's build progress, job-filter builds and learning cycles;
//! - memory writes (`memory_write`), a parameter overlay's rebind of a MicroModel (`neural`) and a
//!   backup (the interface server), each as one timed event.
//!
//! Events carry behavioural fields only: a stage, a MicroModel role, a kind and a label (all fixed
//! names from the code), milliseconds, counts and at most one file path, which must have the shape
//! of a path (`clean_path`). Never question text or code text: the recorder is a behavioural store
//! (`BOUNDARY`, `nmtype::class`) and the types it takes admit nothing else. Recording is one short
//! mutex push with no I/O; the ring is allocated once and only a path is copied.
//!
//! `snapshot` gives the whole live state (the JSON of `GET /api/live` and `nm_live`): the stages
//! and models active now, the four MicroModels with their card bytes, precision, plugged job
//! filters and lens, the card's memory now, the machine's capacity profile, the idle writer, the
//! recent timeline and the backend.
use nmjson::Value;
use nmtype::class::{Boundary, Classed, DataClass};
use std::collections::{BTreeMap, VecDeque};
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

/// Events kept in the ring.
pub const RING: usize = 512;
/// Events a snapshot's timeline carries (the newest).
pub const TIMELINE: usize = 200;
/// The longest path an event may carry (the classic Windows path limit).
pub const PATH_MAX: usize = 260;
/// Counts an event carries at most.
pub const COUNTS: usize = 4;
/// The recorder's boundary: behavioural only.
pub const BOUNDARY: Boundary = Boundary::new("live-activity", DataClass::Behavioural);
/// The four MicroModel roles, in the order the live view shows them.
pub const ROLES: [&str; 4] = ["embedder", "reranker", "reader", "writer"];
/// Progress events of one kind are told at most this often (ms); the now state is always current.
const PROGRESS_TELL_MS: u64 = 1_000;

/// One behavioural event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    /// Order of the event in this process (1, 2, ...).
    pub seq: u64,
    pub at_ms: u64,
    /// `stage`, `call`, `writer`, `progress`, `learn`, `filter`, `memory`, `rebind`, `backup`.
    pub kind: &'static str,
    /// `G0`, `G1.dense`, `G2.close`, `L1`, `dense`, ... (empty when none).
    pub stage: &'static str,
    /// The MicroModel role (empty when none).
    pub model: &'static str,
    /// A fixed name: a tool, a writer job, an outcome (empty when none).
    pub label: &'static str,
    pub ms: u64,
    /// Named counts; unused slots have an empty name.
    pub counts: [(&'static str, u64); COUNTS],
    /// A file path, only when it has the shape of one (`clean_path`).
    pub path: Option<Box<str>>,
}
impl Classed for Event {
    fn data_class(&self) -> DataClass {
        DataClass::Behavioural
    }
}

/// A stage in flight.
#[derive(Clone, Copy, Debug)]
struct Active {
    id: u64,
    stage: &'static str,
    model: &'static str,
    label: &'static str,
    since_ms: u64,
}

/// One MicroModel role's execution now and so far.
#[derive(Clone, Copy, Debug)]
struct Exec {
    depth: u32,
    since_ms: u64,
    forwards: u64,
    busy_us: u64,
}
const EXEC0: Exec = Exec { depth: 0, since_ms: 0, forwards: 0, busy_us: 0 };

/// The idle writer now and so far.
#[derive(Clone, Copy, Debug)]
struct Writer {
    queue: u64,
    held: bool,
    in_session: bool,
    sessions: u64,
    results: u64,
    preempted: u64,
    kept_resident: bool,
    headroom_mb: u64,
    levels: [u64; 4],
    last_kind: &'static str,
    last_at_ms: u64,
    /// What holds the writer back now (`call`, `quiet`, `index`, `memory`), empty when nothing does.
    waiting: &'static str,
}

/// A piece of background work with a measurable end (the dense lane's build).
#[derive(Clone, Copy, Debug)]
struct Progress {
    kind: &'static str,
    done: u64,
    total: u64,
    since_ms: u64,
    told_ms: u64,
}

struct State {
    ring: VecDeque<Event>,
    seq: u64,
    next_id: u64,
    active: Vec<Active>,
    exec: [Exec; 5],
    writer: Writer,
    progress: Vec<Progress>,
}
impl State {
    const fn new() -> Self {
        Self {
            ring: VecDeque::new(),
            seq: 0,
            next_id: 1,
            active: Vec::new(),
            exec: [EXEC0; 5],
            writer: Writer {
                queue: 0,
                held: false,
                in_session: false,
                sessions: 0,
                results: 0,
                preempted: 0,
                kept_resident: false,
                headroom_mb: 0,
                levels: [0; 4],
                last_kind: "",
                last_at_ms: 0,
                waiting: "",
            },
            progress: Vec::new(),
        }
    }
}

static STATE: Mutex<State> = Mutex::new(State::new());
/// Moves on every change of the ring or the now state (a publisher writes only when it moved).
static VERSION: AtomicU64 = AtomicU64::new(0);

fn lock() -> MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(PoisonError::into_inner)
}
fn bump() {
    VERSION.fetch_add(1, Ordering::Relaxed);
}
fn ms_since(t: Instant) -> u64 {
    u64::try_from(t.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// The change counter (`VERSION`).
#[must_use]
pub fn version() -> u64 {
    VERSION.load(Ordering::Relaxed)
}

/// The MicroModel role of a weight card or population id (`weight-opnwgt-basewt-qwen3-0.6b` is
/// the reader); `other` for a model none of the four uses.
#[must_use]
pub fn role_of(id: &str) -> &'static str {
    // 2026-11-25: the role a chosen model runs as (`nmtype::role_models`) first.
    if let Some(r) = nmtype::role_models::role_of(id) {
        return r;
    }
    if id.contains("embedding") {
        "embedder"
    } else if id.contains("reranker") {
        "reranker"
    } else if id.ends_with(crate::neural::WRITER) {
        "writer"
    } else if id.ends_with(crate::neural::POPULATION) {
        "reader"
    } else {
        "other"
    }
}
/// A role name as the fixed name the recorder keeps (`other` for any other text).
#[must_use]
pub fn role_name(role: &str) -> &'static str {
    ROLES.iter().find(|r| **r == role).copied().unwrap_or("other")
}
fn role_index(role: &str) -> usize {
    ROLES.iter().position(|r| *r == role).unwrap_or(ROLES.len())
}

/// `p` when it has the shape of a file path: at most `PATH_MAX` characters of letters, digits,
/// spaces and path punctuation, with no line break; otherwise none. Code and sentences are
/// refused by their characters (braces, semicolons, quotes, operators), so no text rides in it.
#[must_use]
pub fn clean_path(p: &str) -> Option<Box<str>> {
    let ok = !p.is_empty()
        && p.chars().count() <= PATH_MAX
        && p.chars().all(|c| c.is_alphanumeric() || " /\\._-+@#:~()[]".contains(c))
        && !p.contains("  ");
    ok.then(|| p.into())
}

/// An event of fixed names, milliseconds and counts (`seq` and `at_ms` are set when it is pushed).
fn mk(kind: &'static str, stage: &'static str, model: &'static str, label: &'static str, ms: u64, counts: &[(&'static str, u64)]) -> Event {
    let mut c = [("", 0_u64); COUNTS];
    for (slot, v) in c.iter_mut().zip(counts) {
        *slot = *v;
    }
    Event { seq: 0, at_ms: 0, kind, stage, model, label, ms, counts: c, path: None }
}

/// Pushes one event (the caller holds the lock).
fn push(st: &mut State, mut e: Event) {
    if BOUNDARY.admit(&e).is_err() {
        return;
    }
    st.seq += 1;
    e.seq = st.seq;
    e.at_ms = nmtype::now_ms();
    if st.ring.len() >= RING {
        st.ring.pop_front();
    }
    st.ring.push_back(e);
    bump();
}

/// Records one event.
pub fn event(kind: &'static str, stage: &'static str, model: &'static str, label: &'static str, ms: u64, counts: &[(&'static str, u64)]) {
    push(&mut lock(), mk(kind, stage, model, label, ms, counts));
}

/// A stage in flight: shown as active from its start, recorded with its milliseconds and counts
/// when it ends (dropped) or moves on (`next`).
#[must_use = "the stage ends when this is dropped"]
pub struct Stage {
    id: u64,
    kind: &'static str,
    stage: &'static str,
    model: &'static str,
    label: &'static str,
    start: Instant,
    counts: [(&'static str, u64); COUNTS],
}
fn open(kind: &'static str, stage: &'static str, model: &'static str, label: &'static str) -> Stage {
    let mut st = lock();
    let id = st.next_id;
    st.next_id += 1;
    st.active.push(Active { id, stage, model, label, since_ms: nmtype::now_ms() });
    bump();
    Stage { id, kind, stage, model, label, start: Instant::now(), counts: [("", 0); COUNTS] }
}
/// Opens a stage (`G0`, `G2.close`, `dense`, `learn.cycle`, ...) run by `model` (a role or empty).
pub fn stage(stage: &'static str, model: &'static str) -> Stage {
    open("stage", stage, model, "")
}
/// Opens a stage of another kind with a fixed label (a tool's call, a learning cycle).
pub fn stage_as(kind: &'static str, stage: &'static str, model: &'static str, label: &'static str) -> Stage {
    open(kind, stage, model, label)
}
/// A served tool call, by its registered name.
pub fn call(tool: &'static str) -> Stage {
    open("call", "call", "", tool)
}
impl Stage {
    /// Sets a count recorded with the stage when it ends (at most `COUNTS`; a later value of the
    /// same name replaces the earlier).
    pub fn count(&mut self, name: &'static str, v: usize) {
        let v = v as u64;
        if let Some(slot) = self.counts.iter_mut().find(|(n, _)| *n == name || n.is_empty()) {
            *slot = (name, v);
        }
    }
    /// Sets the label recorded when it ends (an outcome).
    pub fn label(&mut self, label: &'static str) {
        self.label = label;
    }
    /// Ends this stage and opens `stage` in its place, in one lock.
    pub fn next(&mut self, stage: &'static str, model: &'static str) {
        let mut st = lock();
        self.close(&mut st);
        let id = st.next_id;
        st.next_id += 1;
        st.active.push(Active { id, stage, model, label: "", since_ms: nmtype::now_ms() });
        self.id = id;
        self.stage = stage;
        self.model = model;
        self.label = "";
        self.start = Instant::now();
        self.counts = [("", 0); COUNTS];
    }
    fn close(&self, st: &mut State) {
        st.active.retain(|a| a.id != self.id);
        let counts = self.counts;
        let n = counts.iter().take_while(|(k, _)| !k.is_empty()).count();
        push(st, mk(self.kind, self.stage, self.model, self.label, ms_since(self.start), counts.get(..n).unwrap_or(&[])));
    }
}
impl Drop for Stage {
    fn drop(&mut self) {
        self.close(&mut lock());
    }
}

/// The engine's hook (`nmengn::model::EXEC_HOOK`): a forward of the MicroModel of card `id`
/// starts (`None`) or ends (`Some(start)`). Updates the now state only; no event per forward.
pub fn exec(id: &str, started: Option<Instant>) {
    let i = role_index(role_of(id)).min(ROLES.len());
    let mut st = lock();
    let Some(x) = st.exec.get_mut(i) else { return };
    match started {
        None => {
            if x.depth == 0 {
                x.since_ms = nmtype::now_ms();
            }
            x.depth += 1;
        }
        Some(t) => {
            x.depth = x.depth.saturating_sub(1);
            x.forwards += 1;
            x.busy_us = x.busy_us.saturating_add(u64::try_from(t.elapsed().as_micros()).unwrap_or(u64::MAX));
        }
    }
    bump();
}

/// Installs the engine's hook (once; every model bind calls this).
pub fn install() {
    let _ = nmengn::model::EXEC_HOOK.set(exec);
}

/// A question's stage times (`drone_path::record_stage_times`, ms per stage with its `total`): one
/// timeline event with the total and the four stages that took longest.
pub fn stage_times(stages: &[(&'static str, f64)]) {
    // A float to integer cast saturates; negative and non-finite times count as zero.
    let ms = |v: f64| if v.is_finite() && v > 0.0 { v.round() as u64 } else { 0 };
    let total = stages.iter().find(|(k, _)| *k == "total").map_or(0, |(_, v)| ms(*v));
    let mut parts: Vec<(&'static str, u64)> = stages.iter().filter(|(k, _)| *k != "total").map(|(k, v)| (*k, ms(*v))).collect();
    parts.sort_by_key(|(_, v)| std::cmp::Reverse(*v));
    parts.truncate(COUNTS);
    push(&mut lock(), mk("times", "question", "", "", total, &parts));
}

/// The world-model level (1 to 4) of a writer job tag (`note`, `names`, `sum` are pieces, `file`
/// files, `module` modules, `outline` the system).
#[must_use]
pub fn level_of(tag: &str) -> usize {
    match tag {
        "file" => 2,
        "module" => 3,
        "outline" => 4,
        _ => 1,
    }
}
const LEVEL_STAGES: [&str; 4] = ["L1", "L2", "L3", "L4"];

/// The writer was seated on the card (`preseat` at start, or loaded for a session): its card MiB
/// and the load's ms.
pub fn writer_seated(label: &'static str, mb: usize, ms: u64) {
    let mut st = lock();
    st.writer.held = true;
    writer_push(&mut st, label, ms, &[("vram_mb", mb as u64)]);
}
/// A writer session starts: whether the other three MicroModels stay seated beside the writer,
/// and the live calls' card room kept free (the queue is told when the session fills it).
pub fn writer_session(kept_resident: bool, headroom_mb: usize) {
    let mut st = lock();
    st.writer.sessions += 1;
    st.writer.in_session = true;
    st.writer.waiting = "";
    st.writer.kept_resident = kept_resident;
    st.writer.headroom_mb = headroom_mb as u64;
    writer_push(&mut st, "session", 0, &[("headroom_mb", headroom_mb as u64), ("kept_resident", u64::from(kept_resident))]);
}
/// What holds the idle writer back now: `call` (a call is in flight), `quiet` (the quiet time after
/// the last call has not passed), `index` (the dense lane is not ready), `memory` (not enough free
/// commit memory to load it), or empty when nothing does. Part of the now state only (no timeline
/// event: the writer is not working while it waits).
pub fn writer_waiting(reason: &'static str) {
    let mut st = lock();
    if st.writer.waiting != reason {
        st.writer.waiting = reason;
        bump();
    }
}
/// The writer's queue was refilled from the material (`pending` tasks in the batch).
pub fn writer_queue(pending: usize) {
    let mut st = lock();
    st.writer.queue = pending as u64;
    bump();
}
/// One result written: its job tag, its ms and its file.
pub fn writer_result(tag: &'static str, ms: u64, path: &str) {
    let level = level_of(tag);
    let p = clean_path(path);
    let mut st = lock();
    st.writer.results += 1;
    st.writer.queue = st.writer.queue.saturating_sub(1);
    if let Some(l) = st.writer.levels.get_mut(level - 1) {
        *l += 1;
    }
    let stage = LEVEL_STAGES.get(level - 1).copied().unwrap_or("L1");
    st.writer.last_kind = tag;
    st.writer.last_at_ms = nmtype::now_ms();
    let results = st.writer.results;
    push(&mut st, Event { path: p, ..mk("writer", stage, "writer", tag, ms, &[("results", results)]) });
}
/// A call preempted the writer's session.
pub fn writer_preempted() {
    let mut st = lock();
    st.writer.preempted += 1;
    writer_push(&mut st, "preempted", 0, &[]);
}
/// The session ended: results written, and whether the writer stays on the card (`kept`) or was
/// let go (`released`).
pub fn writer_session_end(written: usize, kept: bool, ms: u64) {
    let mut st = lock();
    st.writer.in_session = false;
    st.writer.held = kept;
    writer_push(&mut st, if kept { "kept" } else { "released" }, ms, &[("written", written as u64)]);
}
fn writer_push(st: &mut State, label: &'static str, ms: u64, counts: &[(&'static str, u64)]) {
    st.writer.last_kind = label;
    st.writer.last_at_ms = nmtype::now_ms();
    push(st, mk("writer", "writer", "writer", label, ms, counts));
}

/// Background work of `kind` starts with `done` of `total` already there.
pub fn progress_start(kind: &'static str, done: usize, total: usize) {
    let mut st = lock();
    let now = nmtype::now_ms();
    st.progress.retain(|p| p.kind != kind);
    st.progress.push(Progress { kind, done: done as u64, total: total as u64, since_ms: now, told_ms: now });
    push(&mut st, mk("progress", kind, "", "start", 0, &[("done", done as u64), ("total", total as u64)]));
}
/// `n` more of the work of `kind` are done (told on the timeline at most once a second).
pub fn progress_add(kind: &'static str, n: usize) {
    let mut st = lock();
    let now = nmtype::now_ms();
    let Some(p) = st.progress.iter_mut().find(|p| p.kind == kind) else { return };
    p.done = p.done.saturating_add(n as u64);
    if p.total > 0 {
        p.done = p.done.min(p.total);
    }
    let (done, total) = (p.done, p.total);
    if now.saturating_sub(p.told_ms) >= PROGRESS_TELL_MS {
        p.told_ms = now;
        push(&mut st, mk("progress", kind, "", "step", 0, &[("done", done), ("total", total)]));
    } else {
        bump();
    }
}
/// The work of `kind` ended with `done` of `total`.
pub fn progress_end(kind: &'static str, done: usize, total: usize) {
    let mut st = lock();
    let since = st.progress.iter().find(|p| p.kind == kind).map_or(0, |p| p.since_ms);
    st.progress.retain(|p| p.kind != kind);
    let ms = nmtype::now_ms().saturating_sub(since);
    push(&mut st, mk("progress", kind, "", "end", ms, &[("done", done as u64), ("total", total as u64)]));
}

/// The events after `seq`, oldest first (a stream reads only what is new).
#[must_use]
pub fn events_since(seq: u64) -> Vec<Event> {
    lock().ring.iter().filter(|e| e.seq > seq).cloned().collect()
}

fn num(v: u64) -> Value {
    Value::Number(v.to_string())
}
fn text(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn object(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn mb(bytes: usize) -> u64 {
    (bytes >> 20) as u64
}

/// One short line on an event, from its fixed names and numbers only (no path, no text).
#[must_use]
pub fn summary(e: &Event) -> String {
    let mut s = String::new();
    for part in [e.kind, e.stage, e.label, e.model] {
        if !part.is_empty() && !s.split(' ').any(|w| w == part) {
            if !s.is_empty() {
                s.push(' ');
            }
            s.push_str(part);
        }
    }
    if e.ms > 0 {
        let _ = write!(s, " {} ms", e.ms);
    }
    for (k, v) in e.counts.iter().filter(|(k, _)| !k.is_empty()) {
        let _ = write!(s, " {k} {v}");
    }
    s
}

/// An event as JSON: `{seq, at_ms, kind, stage, model, label, ms, counts, path, summary}`.
#[must_use]
pub fn event_value(e: &Event) -> Value {
    let counts: BTreeMap<String, Value> = e.counts.iter().filter(|(k, _)| !k.is_empty()).map(|(k, v)| ((*k).to_owned(), num(*v))).collect();
    object(vec![
        ("seq", num(e.seq)),
        ("at_ms", num(e.at_ms)),
        ("kind", text(e.kind)),
        ("stage", text(e.stage)),
        ("model", text(e.model)),
        ("label", text(e.label)),
        ("ms", num(e.ms)),
        ("counts", Value::Object(counts)),
        ("path", e.path.as_deref().map_or(Value::Null, text)),
        ("summary", text(&summary(e))),
    ])
}

/// The precision most of a model's linear layers are held at (`q4`, `q8`, `f16`, `bf16`, `f32`).
fn precision_of(m: &nmengn::model::Model) -> String {
    let mut by: BTreeMap<&str, usize> = BTreeMap::new();
    for node in m.asm.nodes.iter().filter(|x| x.op == "linear") {
        *by.entry(node.precision.as_str()).or_default() += 1;
    }
    by.into_iter().max_by_key(|(_, c)| *c).map_or_else(|| "default".to_owned(), |(p, _)| p.to_owned())
}

/// What the recorder holds now, copied out under one lock.
struct Now {
    active: Vec<Active>,
    exec: [Exec; 5],
    writer: Writer,
    progress: Vec<Progress>,
    timeline: Vec<Event>,
}
fn now_state() -> Now {
    let st = lock();
    let skip = st.ring.len().saturating_sub(TIMELINE);
    Now { active: st.active.clone(), exec: st.exec, writer: st.writer, progress: st.progress.clone(), timeline: st.ring.iter().skip(skip).cloned().collect() }
}

/// One MicroModel's row: card id, card bytes, precision.
struct Seat {
    card: String,
    bytes: usize,
    precision: String,
}
fn seats() -> BTreeMap<&'static str, Seat> {
    let mut out: BTreeMap<&'static str, Seat> = BTreeMap::new();
    for (slot, m) in crate::neural::bound_models() {
        let role = if slot == "base" { "reader" } else { role_name(slot) };
        out.insert(role, Seat { card: m.pop.card.id.clone(), bytes: m.pop.device_bytes(), precision: precision_of(&m) });
    }
    // 2026-11-27: the writer a session runs counts too, not only the one kept between sessions.
    if let Some(w) = crate::writer_idle::writer_on_card_view(|w| Seat { card: w.pop.card.id.clone(), bytes: w.pop.device_bytes(), precision: precision_of(w) }) {
        out.insert("writer", w);
    }
    out
}

/// The whole live state (see the module note); the JSON of `GET /api/live` and `nm_live`.
#[must_use]
pub fn snapshot() -> Value {
    let now = now_state();
    let t = nmtype::now_ms();
    let res = nmengn::residency::get();
    // Roadmap Phase 4.5 (2026-11-25): the driver's figures, so used is the card's real use; the
    // effective free memory under the user's card limit is under `capacity.limits`.
    let memory = res.memory_driver();
    let seats = seats();
    let mut filters = nmengn::filter::plugged();
    filters.extend(nmengn::filter::job_filters_loaded());
    let (lens, readout) = crate::adapter::seated();
    // 2026-11-27: one source for what is held: the MicroModels seated (`preload::view`, "N of 4"),
    // so each model's `resident` and `micromodels.held` never disagree. A seated model whose card
    // copies were released for room (the writer's session, say) is still held; `vram_mb` says
    // what it has on the card now.
    let micromodels = crate::preload::view();
    let held: Vec<String> = micromodels
        .get("roles")
        .and_then(Value::object)
        .map(|r| r.iter().filter(|(_, x)| x.get("state").and_then(Value::string).is_ok_and(|st| st == "held")).map(|(k, _)| k.clone()).collect())
        .unwrap_or_default();
    let models = models_value(&now, &seats, &held, &filters, lens.as_deref(), readout.as_deref());
    let busy = !now.active.is_empty() || now.exec.iter().any(|x| x.depth > 0) || now.writer.in_session || !now.progress.is_empty();
    let active: Vec<Value> = now
        .active
        .iter()
        .map(|a| object(vec![("stage", text(a.stage)), ("model", text(a.model)), ("label", text(a.label)), ("since_ms", num(a.since_ms)), ("elapsed_ms", num(t.saturating_sub(a.since_ms)))]))
        .collect();
    let progress: Vec<Value> = now
        .progress
        .iter()
        .map(|p| object(vec![("kind", text(p.kind)), ("done", num(p.done)), ("total", num(p.total)), ("since_ms", num(p.since_ms))]))
        .collect();
    let filter_bytes: usize = filters.iter().map(|p| p.device_bytes()).sum();
    object(vec![
        ("state", text(if busy { "busy" } else { "idle" })),
        ("at_ms", num(t)),
        ("pid", num(u64::from(std::process::id()))),
        ("active", Value::Array(active)),
        ("progress", Value::Array(progress)),
        ("models", Value::Array(models)),
        // 2026-11-27: the four MicroModels as loaded at start ("N of 4" held, and per role why).
        ("micromodels", micromodels),
        ("vram", vram_value(memory, res, &seats, filter_bytes)),
        ("capacity", crate::limits::capacity_value()),
        // 2026-11-27: each memory area's use against its budget, and the drones' intake.
        ("memory_areas", crate::memory_areas::current()),
        ("writer", writer_value(&now.writer)),
        ("timeline", Value::Array(now.timeline.iter().map(event_value).collect())),
        ("backend", text(if res.card().is_some() { "cuda" } else { "cpu" })),
        // Roadmap Phase 4.6 (2026-11-25): the backend in plain English (which card and why, or
        // why the processor), as this process already knows it; nothing is opened to say it.
        ("backend_note", text(&nmengn::device::status_cached().unwrap_or_else(|| "No graphics card has been opened yet in this process.".to_owned()))),
    ])
}

fn models_value(now: &Now, seats: &BTreeMap<&'static str, Seat>, held: &[String], filters: &[std::sync::Arc<nmengn::filter::Plugged>], lens: Option<&str>, readout: Option<&str>) -> Vec<Value> {
    ROLES
        .iter()
        .enumerate()
        .map(|(i, role)| {
            let x = now.exec.get(i).copied().unwrap_or(EXEC0);
            let seat = seats.get(role);
            let plugged: Vec<Value> = filters.iter().filter(|p| role_of(&p.model) == *role).flat_map(|p| p.cartridges.iter().map(|c| text(c))).collect();
            let lens_now = match *role {
                "embedder" => lens,
                "reranker" => readout,
                _ => None,
            };
            object(vec![
                ("role", text(role)),
                ("card", seat.map_or(Value::Null, |s| text(&s.card))),
                ("resident", Value::Bool(held.iter().any(|h| h == role) || seat.is_some_and(|s| s.bytes > 0))),
                ("executing", Value::Bool(x.depth > 0)),
                ("since_ms", if x.depth > 0 { num(x.since_ms) } else { Value::Null }),
                ("forwards", num(x.forwards)),
                ("busy_ms", num(x.busy_us / 1000)),
                ("vram_mb", num(seat.map_or(0, |s| mb(s.bytes)))),
                ("precision", seat.map_or(Value::Null, |s| text(&s.precision))),
                ("filters", Value::Array(plugged)),
                ("lens", lens_now.map_or(Value::Null, text)),
            ])
        })
        .collect()
}

fn vram_value(memory: Option<(usize, usize)>, res: &nmengn::residency::Residency, seats: &BTreeMap<&'static str, Seat>, filter_bytes: usize) -> Value {
    let stats = res.stats();
    let mut by: BTreeMap<String, Value> = ROLES.iter().map(|r| ((*r).to_owned(), num(seats.get(r).map_or(0, |s| mb(s.bytes))))).collect();
    by.insert("filters".to_owned(), num(mb(filter_bytes)));
    by.insert("kv_windows".to_owned(), num(mb(stats.window_bytes)));
    // 2026-11-27: card memory in use that no MicroModel, filter or window accounts for (buffers,
    // the driver's context, another process), named as such rather than left out.
    let attributed: usize = seats.values().map(|s| s.bytes).sum::<usize>() + filter_bytes + stats.window_bytes;
    by.insert("unattributed".to_owned(), memory.map_or(Value::Null, |(free, total)| num(mb(total.saturating_sub(free).saturating_sub(attributed)))));
    let opt = |v: Option<u64>| v.map_or(Value::Null, num);
    object(vec![
        ("total_mb", opt(memory.map(|(_, total)| mb(total)))),
        ("used_mb", opt(memory.map(|(free, total)| mb(total.saturating_sub(free))))),
        ("free_mb", opt(memory.map(|(free, _)| mb(free)))),
        ("reserve_mb", num(mb(stats.reserve))),
        ("reserve_measured", Value::Bool(stats.measured)),
        ("by_model", Value::Object(by)),
        (
            "residency",
            object(vec![
                ("windows_held", num(stats.windows_held as u64)),
                ("window_hits", num(stats.window_hits)),
                ("window_misses", num(stats.window_misses)),
                ("windows_released", num(stats.windows_released)),
                ("weights_released", num(stats.weights_released)),
                ("bytes_released_mb", num(stats.bytes_released >> 20)),
            ]),
        ),
    ])
}

fn writer_value(w: &Writer) -> Value {
    object(vec![
        ("queue", num(w.queue)),
        ("held", Value::Bool(w.held)),
        ("in_session", Value::Bool(w.in_session)),
        ("sessions", num(w.sessions)),
        ("results", num(w.results)),
        ("preempted", num(w.preempted)),
        ("kept_resident", Value::Bool(w.kept_resident)),
        ("headroom_mb", num(w.headroom_mb)),
        ("waiting", if w.waiting.is_empty() { Value::Null } else { text(w.waiting) }),
        ("levels", object(LEVEL_STAGES.iter().zip(w.levels).map(|(l, v)| (*l, num(v))).collect())),
        (
            "last_event",
            if w.last_kind.is_empty() { Value::Null } else { object(vec![("kind", text(w.last_kind)), ("at_ms", num(w.last_at_ms))]) },
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The recorder is process-wide; the tests take turns.
    static TURN: Mutex<()> = Mutex::new(());

    fn strings(v: &Value, out: &mut Vec<String>) {
        match v {
            Value::String(s) => out.push(s.clone()),
            Value::Array(a) => a.iter().for_each(|x| strings(x, out)),
            Value::Object(o) => o.values().for_each(|x| strings(x, out)),
            _ => {}
        }
    }

    #[test]
    fn the_ring_is_bounded_and_ordered() {
        let _t = TURN.lock().unwrap_or_else(PoisonError::into_inner);
        let before = lock().seq;
        for i in 0..(RING + 40) {
            event("stage", "G1", "", "", i as u64, &[("n", i as u64)]);
        }
        let st = lock();
        assert_eq!(st.ring.len(), RING);
        assert!(st.ring.iter().zip(st.ring.iter().skip(1)).all(|(a, b)| b.seq == a.seq + 1));
        assert_eq!(st.ring.back().map(|e| e.seq), Some(before + (RING + 40) as u64));
        drop(st);
        assert_eq!(events_since(before + (RING + 40) as u64 - 3).len(), 3);
    }

    #[test]
    fn a_stage_is_active_until_it_ends_and_moves_on() {
        let _t = TURN.lock().unwrap_or_else(PoisonError::into_inner);
        let seq = lock().seq;
        {
            let mut s = stage("G0", "");
            assert!(lock().active.iter().any(|a| a.stage == "G0"));
            s.count("pieces", 7);
            s.next("G1", "embedder");
            let st = lock();
            assert!(st.active.iter().any(|a| a.stage == "G1") && !st.active.iter().any(|a| a.stage == "G0"));
        }
        assert!(!lock().active.iter().any(|a| a.stage == "G1" || a.stage == "G0"));
        let ev = events_since(seq);
        let g0 = ev.iter().find(|e| e.stage == "G0").expect("G0 recorded");
        assert_eq!(g0.counts.first(), Some(&("pieces", 7)));
        assert!(ev.iter().any(|e| e.stage == "G1" && e.model == "embedder"));
    }

    #[test]
    fn executing_follows_the_forward_hook() {
        let _t = TURN.lock().unwrap_or_else(PoisonError::into_inner);
        let i = role_index("reranker");
        let f0 = lock().exec.get(i).map_or(0, |x| x.forwards);
        exec("weight-opnwgt-basewt-qwen3-reranker-0.6b", None);
        assert!(lock().exec.get(i).is_some_and(|x| x.depth == 1));
        exec("weight-opnwgt-basewt-qwen3-reranker-0.6b", Some(Instant::now()));
        let x = lock().exec.get(i).copied().unwrap_or(EXEC0);
        assert_eq!((x.depth, x.forwards), (0, f0 + 1));
        assert_eq!(role_of("weight-opnwgt-basewt-qwen3-0.6b"), "reader");
        assert_eq!(role_of("weight-opnwgt-basewt-qwen3-1.7b"), "writer");
        assert_eq!(role_of("qwen3-embedding-0.6b"), "embedder");
    }

    #[test]
    fn the_snapshot_has_the_live_shape() {
        let _t = TURN.lock().unwrap_or_else(PoisonError::into_inner);
        writer_session(true, 512);
        writer_queue(12);
        writer_result("file", 900, "crates/nmapi/src/live.rs");
        writer_session_end(1, true, 1);
        let v = snapshot();
        for k in ["state", "active", "models", "vram", "capacity", "writer", "timeline", "backend"] {
            assert!(v.get(k).is_ok(), "missing {k}");
        }
        let state = v.get("state").and_then(Value::string).unwrap_or_default();
        assert!(state == "idle" || state == "busy");
        let backend = v.get("backend").and_then(Value::string).unwrap_or_default();
        assert!(backend == "cuda" || backend == "cpu");
        let models = v.get("models").and_then(Value::array).unwrap_or_default();
        assert_eq!(models.len(), 4);
        for m in models {
            for k in ["role", "card", "resident", "executing", "vram_mb", "precision", "filters", "lens"] {
                assert!(m.get(k).is_ok(), "model missing {k}");
            }
        }
        let vram = v.get("vram").expect("vram");
        for k in ["total_mb", "used_mb", "free_mb", "reserve_mb", "by_model"] {
            assert!(vram.get(k).is_ok(), "vram missing {k}");
        }
        let w = v.get("writer").expect("writer");
        for k in ["queue", "held", "sessions", "results", "preempted", "kept_resident", "headroom_mb", "levels", "last_event"] {
            assert!(w.get(k).is_ok(), "writer missing {k}");
        }
        let levels = w.get("levels").expect("levels");
        assert!(["L1", "L2", "L3", "L4"].iter().all(|l| levels.get(l).is_ok()));
        assert!(levels.get("L2").and_then(Value::integer).unwrap_or(0) >= 1);
        let tl = v.get("timeline").and_then(Value::array).unwrap_or_default();
        let last = tl.iter().rev().find(|e| e.get("stage").and_then(Value::string).ok() == Some("L2")).expect("L2 result on the timeline");
        for k in ["at_ms", "kind", "stage", "model", "ms", "summary"] {
            assert!(last.get(k).is_ok(), "event missing {k}");
        }
        assert_eq!(last.get("path").and_then(Value::string).ok(), Some("crates/nmapi/src/live.rs"));
        assert!(v.encode().is_ok());
    }

    #[test]
    fn nm_live_is_in_the_table_and_on_demand() {
        let api = crate::tools::api();
        assert!(api.operations().iter().any(|o| o.name == "nm_live"));
        assert!(crate::mcp::PRIORITY_TOOLS.contains(&"nm_live") && !crate::mcp::LEAN_TOOLS.contains(&"nm_live"));
        let v = api.dispatch("nm_live", &Value::Object(BTreeMap::new())).expect("nm_live answers");
        assert!(v.get("timeline").is_ok() && v.get("capacity").is_ok());
    }

    #[test]
    fn no_event_field_carries_text_longer_than_a_path() {
        let _t = TURN.lock().unwrap_or_else(PoisonError::into_inner);
        let code = "fn main() { let x = 1; println!(\"{x}\"); }";
        let sentence = "where is the ring buffer pushed, and why?";
        let long = "a/".repeat(PATH_MAX);
        assert!(clean_path(code).is_none());
        assert!(clean_path(sentence).is_none());
        assert!(clean_path(&long).is_none());
        assert!(clean_path("src/a.rs\nfn x() {}").is_none());
        assert!(clean_path("crates/nmapi/src/live.rs").is_some());
        assert!(clean_path("C:\\Users\\dev\\My Project\\src\\main.rs").is_some());
        writer_result("note", 3, code);
        writer_result("note", 3, &long);
        writer_result("note", 5, sentence);
        let seq = lock().seq;
        assert!(events_since(seq.saturating_sub(3)).iter().all(|e| e.path.is_none()));
        let v = snapshot();
        let mut all = Vec::new();
        for e in v.get("timeline").and_then(Value::array).unwrap_or_default() {
            strings(e, &mut all);
        }
        assert!(!all.is_empty());
        for s in &all {
            assert!(s.chars().count() <= PATH_MAX, "an event field of {} characters", s.chars().count());
            assert!(!s.contains('\n') && !s.contains('{') && !s.contains(';'), "text in an event field: {s}");
        }
    }
}
