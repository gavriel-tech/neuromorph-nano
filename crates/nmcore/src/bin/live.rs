//! The live state of a serving process, for the interface server (a separate process): which
//! project it serves, its path, the call in progress, and the models it holds now with their bytes
//! in RAM and on the card, the precision in effect and the layers on the card. Written to
//! `memory/live/<project key>-<pid>.json` when serving starts, every `BEAT_MS`, and soon after a call
//! starts or ends (by the writing thread, so the call does not wait); the interface reads only files whose beat is newer than `FRESH_MS`. The file holds no
//! file contents and no question text. Standard library and nmjson only.
use nmjson::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// How often a serving process rewrites its live file.
const BEAT_MS: u64 = 10_000;
/// A live file older than this is a process that ended or hangs, and is not shown.
pub const FRESH_MS: u64 = 3 * BEAT_MS;

struct Live {
    home: PathBuf,
    file: PathBuf,
    fixed: BTreeMap<String, Value>,
    call: Mutex<Option<(String, u64)>>,
}
static LIVE: OnceLock<Live> = OnceLock::new();

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}

/// The folder of live files.
#[must_use]
pub fn dir(home: &Path) -> PathBuf {
    home.join("memory").join("live")
}

/// Starts publishing this process's live state (serving processes only).
pub fn start(home: &Path, root: &Path, project: &str, mode: &str) {
    let key = nmtype::project::key(root);
    let started = nmtype::now_ms();
    let fixed: BTreeMap<String, Value> = [
        ("pid", n(u64::from(std::process::id()))),
        ("project", s(project)),
        ("project_key", s(&key)),
        ("root", s(&root.display().to_string())),
        ("mode", s(mode)),
        ("started_ms", n(started)),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v))
    .collect();
    // 2026-11-27: one file per serving process (`<project key>-<pid>.json`), so two engines on
    // one project (a `daemon` beside a coding agent's `serve`, say) never overwrite each other's
    // readings; the interface picks the one that holds the MicroModels (`merged`).
    sweep(home);
    let live = Live { home: home.to_path_buf(), file: dir(home).join(format!("{key}-{}.json", std::process::id())), fixed, call: Mutex::new(None) };
    if LIVE.set(live).is_err() {
        return;
    }
    write();
    let beat = std::thread::spawn(|| {
        loop {
            std::thread::park_timeout(std::time::Duration::from_millis(BEAT_MS));
            write();
        }
    });
    let _ = BEAT.set(beat.thread().clone());
    // Roadmap Phase 4.5 (2026-11-24): the live activity (`nmapi::live::snapshot`) is published
    // beside the live file, checked every `ACTIVITY_MS` and written only when something changed
    // (or every beat, so the interface knows it is fresh). The recorder itself does no I/O.
    std::thread::spawn(|| {
        let mut seen = u64::MAX;
        let mut wrote = 0_u64;
        loop {
            std::thread::park_timeout(std::time::Duration::from_millis(ACTIVITY_MS));
            let v = nmapi::live::version();
            let now = nmtype::now_ms();
            if v != seen || now.saturating_sub(wrote) >= BEAT_MS {
                seen = v;
                wrote = now;
                write_activity();
            }
        }
    });
}

/// The thread that rewrites the live file; woken when a call starts or ends, so the call itself
/// never waits for the write.
static BEAT: OnceLock<std::thread::Thread> = OnceLock::new();

/// A call started (`Some`) or ended (`None`).
pub fn call(tool: Option<&str>) {
    if let Some(l) = LIVE.get() {
        if let Ok(mut c) = l.call.lock() {
            *c = tool.map(|t| (t.to_owned(), nmtype::now_ms()));
        }
        if let Some(th) = BEAT.get() {
            th.unpark();
        }
    }
}

/// Serving ended: this process's files go, so it is no longer shown.
pub fn end() {
    if let Some(l) = LIVE.get() {
        let _ = std::fs::remove_file(&l.file);
        let _ = std::fs::remove_file(activity_file(l));
    }
}

/// Live and activity files a process left when it ended without `end` (killed, say): removed
/// once they are a day old. Fresh files of other processes are never touched.
fn sweep(home: &Path) {
    const OLD: std::time::Duration = std::time::Duration::from_secs(24 * 3600);
    for d in [dir(home), activity_dir(home)] {
        for e in std::fs::read_dir(d).into_iter().flatten().flatten() {
            let old = e.metadata().ok().and_then(|m| m.modified().ok()).and_then(|t| t.elapsed().ok()).is_some_and(|age| age > OLD);
            if old && e.path().extension().is_some_and(|x| x == "json") {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
}

fn write() {
    let Some(l) = LIVE.get() else { return };
    let mut o = l.fixed.clone();
    o.insert("beat_ms".into(), n(nmtype::now_ms()));
    if let Some((tool, since)) = l.call.lock().ok().and_then(|c| c.clone()) {
        o.insert("call".into(), Value::Object([("tool".to_owned(), s(&tool)), ("since_ms".to_owned(), n(since))].into_iter().collect()));
    }
    o.insert("loaded".into(), Value::Array(loaded(&l.home)));
    if let Some(org) = nmapi::tools::installed() {
        let sub = org.sub();
        o.insert(
            "substrate".into(),
            Value::Object([("chunks".to_owned(), n(sub.chunks.len() as u64)), ("files".to_owned(), n(sub.files.len() as u64))].into_iter().collect()),
        );
    }
    // Roadmap Phase 3 (2026-11-24): the last calls' stage times for the interface's view.
    o.insert("stage_times".into(), Value::Array(nmapi::drone_path::recent_stage_times()));
    // 2026-11-25 (nothing pseudo): the hardware limits in effect and the residency mode as THIS
    // serving process has them, so the interface shows the serving process's, not its own.
    o.insert("limits".into(), nmapi::limits::effective_value());
    let text = nmjson::pretty::pretty(&Value::Object(o));
    if std::fs::create_dir_all(dir(&l.home)).is_ok() {
        let tmp = l.file.with_extension("json.tmp");
        if std::fs::write(&tmp, text).is_ok() {
            let _ = std::fs::rename(&tmp, &l.file);
        }
    }
}

/// Bits of a linear's precision name (`q4`, `q8`, `f16`, `bf16`, `f32`).
fn bits(p: &str) -> Option<u64> {
    match p {
        "q4" => Some(4),
        "q8" => Some(8),
        "f16" | "bf16" => Some(16),
        "f32" => Some(32),
        _ => None,
    }
}

/// The models folder name a card id belongs to (`weight-opnwgt-basewt-qwen3-0.6b` is `qwen3-0.6b`).
fn folder_of(home: &Path, card_id: &str) -> String {
    std::fs::read_dir(nmtype::models_dir(home))
        .ok()
        .and_then(|rd| {
            rd.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|f| card_id == f || card_id.ends_with(&format!("-{f}")))
                .max_by_key(String::len)
        })
        .unwrap_or_else(|| card_id.to_owned())
}

/// Every model this process holds now, one row per population: the roles using it, bytes of its
/// parameter views in RAM and of its weight copies on the card, the precision asked and the one
/// in effect (the widths of its linear layers), its layer and head-group counts, and the layers
/// with weights on the card.
fn loaded(home: &Path) -> Vec<Value> {
    let asked = nmengn::model::PRECISION.read().map(|p| *p).unwrap_or_default();
    let mut by_pop: BTreeMap<String, (Vec<&'static str>, std::sync::Arc<nmengn::model::Model>)> = BTreeMap::new();
    // 2026-11-27: the writer on the card (kept, or run by an idle session) is a held model too.
    for (role, m) in nmapi::neural::bound_models().into_iter().chain(nmapi::writer_idle::writer_on_card().map(|w| ("writer", w))) {
        by_pop.entry(m.pop.card.id.clone()).or_insert_with(|| (Vec::new(), std::sync::Arc::clone(&m))).0.push(role);
    }
    by_pop
        .into_iter()
        .map(|(card, (roles, m))| {
            let (ram, gpu) = m.pop.prov.totals();
            let mut widths: BTreeMap<u64, usize> = BTreeMap::new();
            for node in m.asm.nodes.iter().filter(|x| x.op == "linear") {
                if let Some(b) = bits(&node.precision) {
                    *widths.entry(b).or_default() += 1;
                }
            }
            let effective = widths.iter().max_by_key(|(_, c)| **c).map(|(b, _)| *b);
            // Cluster ids name their tensor (`<model>/model.layers.<n>.<...>/<region>`).
            let layers_on_card: BTreeSet<usize> = m
                .pop
                .prov
                .residency()
                .into_iter()
                .filter(|(_, st, _)| matches!(st, nmnpcs::Storage::Device { .. }))
                .filter_map(|(id, _, _)| {
                    let rest = id.split_once("layers.")?.1;
                    rest.split('.').next()?.parse::<usize>().ok()
                })
                .collect();
            let device = if m.pop.gpu.is_some() && gpu > 0 { "gpu" } else { "cpu" };
            let mut o: BTreeMap<String, Value> = BTreeMap::new();
            o.insert("id".into(), s(&folder_of(home, &card)));
            o.insert("card".into(), s(&card));
            o.insert("roles".into(), Value::Array(roles.iter().map(|r| s(r)).collect()));
            o.insert("ram_bytes".into(), n(ram));
            o.insert("gpu_bytes".into(), n(gpu));
            o.insert("bytes".into(), n(ram + gpu));
            o.insert("device".into(), s(device));
            o.insert("precision".into(), asked.of(&card).map_or_else(|| s("default"), |b| n(u64::from(b))));
            o.insert("effective_bits".into(), effective.map_or(Value::Null, n));
            o.insert("mixed_precision".into(), Value::Bool(widths.len() > 1));
            o.insert("layers".into(), n(m.pop.dims.layers as u64));
            o.insert("heads".into(), n(m.pop.dims.heads as u64));
            o.insert("head_groups".into(), n(m.pop.dims.kv_heads as u64));
            o.insert("kv_heads".into(), n(m.pop.dims.kv_heads as u64));
            let on_card = Value::Array(layers_on_card.iter().map(|l| n(*l as u64)).collect());
            o.insert("layers_on_card".into(), on_card.clone());
            o.insert("on_card".into(), on_card);
            Value::Object(o)
        })
        .collect()
}

/// The live files of serving processes whose beat is fresh, newest beat first.
#[must_use]
pub fn read_all(home: &Path) -> Vec<Value> {
    let now = nmtype::now_ms();
    let mut rows: Vec<(u64, Value)> = std::fs::read_dir(dir(home))
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
                .filter_map(|e| std::fs::read(e.path()).ok().and_then(|b| nmjson::parse(&b).ok()))
                .filter_map(|v| {
                    let beat = v.get("beat_ms").and_then(Value::integer).ok()?;
                    (beat > 0 && now.saturating_sub(beat) <= FRESH_MS).then_some((beat, v))
                })
                .collect()
        })
        .unwrap_or_default();
    rows.sort_by_key(|r| std::cmp::Reverse(r.0));
    rows.into_iter().map(|r| r.1).collect()
}

// Roadmap Phase 4.5 (2026-11-24): live activity. Each serving process publishes its recorder's
// state (`nmapi::live::snapshot`) to `memory/live/activity/<project key>-<pid>.json`; the interface
// server (a separate process) serves the freshest as `GET /api/live`, merged with its own
// recorder (job-filter builds and learning cycles it runs), and streams new timeline events as
// Server-Sent Events on `GET /api/live/stream`. The files hold behavioural fields only.

/// How often the serving process checks whether its live activity changed (ms).
const ACTIVITY_MS: u64 = 200;
/// How often a stream looks for new events (ms).
const STREAM_MS: u64 = 200;
/// A stream sends a comment line this often when nothing happens (ms), so a closed tab is noticed.
const STREAM_BEAT_MS: u64 = 15_000;
/// Events `GET /api/live` returns (the newest, all processes together).
const TIMELINE: usize = 200;

fn activity_dir(home: &Path) -> PathBuf {
    dir(home).join("activity")
}
fn activity_file(l: &Live) -> PathBuf {
    let name = l.file.file_name().map(std::ffi::OsStr::to_os_string).unwrap_or_default();
    activity_dir(&l.home).join(name)
}

/// Writes this serving process's live activity (by the publishing thread only).
fn write_activity() {
    let Some(l) = LIVE.get() else { return };
    let mut v = nmapi::live::snapshot();
    if let Value::Object(o) = &mut v {
        for k in ["project", "project_key"] {
            if let Some(x) = l.fixed.get(k) {
                o.insert(k.to_owned(), x.clone());
            }
        }
        o.insert("beat_ms".into(), n(nmtype::now_ms()));
    }
    let Ok(text) = v.encode() else { return };
    let file = activity_file(l);
    if std::fs::create_dir_all(activity_dir(&l.home)).is_ok() {
        let tmp = file.with_extension("json.tmp");
        if std::fs::write(&tmp, text).is_ok() {
            let _ = std::fs::rename(&tmp, &file);
        }
    }
}

/// The fresh activity files of serving processes, newest beat first.
fn read_activity(home: &Path) -> Vec<Value> {
    let now = nmtype::now_ms();
    let mut rows: Vec<(u64, Value)> = std::fs::read_dir(activity_dir(home))
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
                .filter_map(|e| std::fs::read(e.path()).ok().and_then(|b| nmjson::parse(&b).ok()))
                .filter_map(|v| {
                    let beat = v.get("beat_ms").and_then(Value::integer).ok()?;
                    (now.saturating_sub(beat) <= FRESH_MS).then_some((beat, v))
                })
                .collect()
        })
        .unwrap_or_default();
    rows.sort_by_key(|r| std::cmp::Reverse(r.0));
    rows.into_iter().map(|r| r.1).collect()
}

fn pid_of(v: &Value) -> u64 {
    v.get("pid").and_then(Value::integer).unwrap_or(0)
}
fn arr(v: &Value, k: &str) -> Vec<Value> {
    v.get(k).and_then(Value::array).map(<[Value]>::to_vec).unwrap_or_default()
}
fn busy(v: &Value) -> bool {
    v.get("state").and_then(Value::string).ok() == Some("busy")
}
/// `rows` with each tagged by its process id.
fn tagged(rows: Vec<Value>, pid: u64) -> Vec<Value> {
    rows.into_iter()
        .map(|mut r| {
            if let Value::Object(o) = &mut r {
                o.insert("pid".into(), n(pid));
            }
            r
        })
        .collect()
}

// 2026-11-27 (C1): guidance deliveries. A hook (`neuromorph-nano hook <Event>`) is its own short
// process, so the reminder, loop continuation or step nudge it gives the agent cannot wait in a
// serving process's recorder. It appends one event line to `memory/live/deliveries.jsonl` at the
// delivery; `GET /api/live` and the stream read the file together with the serving processes'
// activity files (no loop of its own). The lines hold behavioural fields only: fixed names, a
// time and counts, never the guidance text.
const DELIVERIES: &str = "deliveries.jsonl";
/// Delivery events kept in the file once it passes `DELIVERIES_TRIM` bytes.
const DELIVERIES_KEEP: usize = 64;
const DELIVERIES_TRIM: u64 = 32 * 1024;
/// The process id the delivery events are tagged with (they come from many short processes).
const DELIVERIES_PID: u64 = 0;

/// Records one guidance delivery to the agent (`stage` `guidance.reminder`, `guidance.loop` or
/// `guidance.step`; `label` a fixed name), one event per hook reply. A failed write costs the
/// reading only, never the delivery.
pub fn delivered(home: &Path, stage: &'static str, label: &'static str, counts: &[(&'static str, u64)]) {
    let at = nmtype::now_ms();
    let mut c = [("", 0_u64); nmapi::live::COUNTS];
    for (slot, v) in c.iter_mut().zip(counts) {
        *slot = *v;
    }
    // The sequence grows with time across hook processes (the stream sends what is newer), and
    // within one process with each delivery.
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let by_time = at.saturating_mul(1000).saturating_add(u64::from(std::process::id()) % 1000);
    let seq = by_time.max(LAST.load(std::sync::atomic::Ordering::Relaxed).saturating_add(1));
    LAST.store(seq, std::sync::atomic::Ordering::Relaxed);
    let e = nmapi::live::Event { seq, at_ms: at, kind: "guidance", stage, model: "", label, ms: 0, counts: c, path: None };
    if nmapi::live::BOUNDARY.admit(&e).is_err() {
        return;
    }
    let Ok(line) = nmapi::live::event_value(&e).encode() else { return };
    let file = dir(home).join(DELIVERIES);
    if std::fs::create_dir_all(dir(home)).is_err() {
        return;
    }
    {
        use std::io::Write;
        let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&file) else { return };
        let _ = writeln!(f, "{line}");
    }
    if std::fs::metadata(&file).map_or(0, |m| m.len()) > DELIVERIES_TRIM {
        let all = std::fs::read_to_string(&file).unwrap_or_default();
        let lines: Vec<&str> = all.lines().filter(|l| !l.trim().is_empty()).collect();
        let keep = lines.get(lines.len().saturating_sub(DELIVERIES_KEEP)..).unwrap_or_default();
        let tmp = file.with_extension(format!("jsonl.{}.tmp", std::process::id()));
        if std::fs::write(&tmp, format!("{}\n", keep.join("\n"))).is_ok() && std::fs::rename(&tmp, &file).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }
}

/// The guidance deliveries as one process row (`pid` 0, idle, its timeline the newest delivery
/// events), merged beside the serving processes; `None` when there are none.
fn deliveries(home: &Path) -> Option<Value> {
    let text = std::fs::read_to_string(dir(home).join(DELIVERIES)).ok()?;
    let mut tl: Vec<Value> = text.lines().filter_map(|l| nmjson::parse(l.as_bytes()).ok()).filter(|v| v.get("kind").and_then(Value::string).ok() == Some("guidance")).collect();
    tl = tl.split_off(tl.len().saturating_sub(DELIVERIES_KEEP));
    (!tl.is_empty()).then(|| Value::Object([("pid".to_owned(), n(DELIVERIES_PID)), ("state".to_owned(), s("idle")), ("timeline".to_owned(), Value::Array(tl))].into_iter().collect()))
}

/// `GET /api/live`: the live state of the busiest (else the freshest) serving process, with the
/// stages active and the timeline of every serving process and of this one merged; with no
/// serving process, this process's own state and `backend` null (no engine runs to say which).
/// Guidance deliveries join the timeline.
#[must_use]
pub fn api_view(home: &Path) -> Value {
    merged(&read_activity(home), &nmapi::live::snapshot(), deliveries(home).as_ref())
}

/// MicroModels a process's activity says it holds (`micromodels.held`, "N of 4").
fn held_of(v: &Value) -> u64 {
    v.get("micromodels")
        .and_then(|m| m.get("held"))
        .and_then(Value::string)
        .ok()
        .and_then(|t| t.split_whitespace().next())
        .and_then(|x| x.parse().ok())
        .unwrap_or(0)
}

fn merged(procs: &[Value], own: &Value, extra: Option<&Value>) -> Value {
    // 2026-11-27: the primary is the serving engine: the process that holds the most MicroModels,
    // then a busy one, then the newest beat (`procs` is newest first), so the readings shown are
    // never those of a process that holds no model while an engine serves.
    let primary = procs.iter().enumerate().max_by_key(|(i, p)| (held_of(p), busy(p), std::cmp::Reverse(*i))).map(|(_, p)| p);
    let mut out = primary.cloned().unwrap_or_else(|| own.clone());
    let all: Vec<&Value> = procs.iter().chain(std::iter::once(own)).chain(extra).collect();
    let mut active = Vec::new();
    let mut timeline = Vec::new();
    for p in &all {
        active.extend(tagged(arr(p, "active"), pid_of(p)));
        timeline.extend(tagged(arr(p, "timeline"), pid_of(p)));
    }
    timeline.sort_by_key(|e| e.get("at_ms").and_then(Value::integer).unwrap_or(0));
    let skip = timeline.len().saturating_sub(TIMELINE);
    let processes: Vec<Value> = procs
        .iter()
        .map(|p| {
            let field = |k: &str| p.get(k).cloned().unwrap_or(Value::Null);
            Value::Object([("pid".to_owned(), field("pid")), ("project".to_owned(), field("project")), ("state".to_owned(), field("state"))].into_iter().collect())
        })
        .collect();
    if let Value::Object(o) = &mut out {
        o.insert("state".into(), s(if all.iter().any(|p| busy(p)) { "busy" } else { "idle" }));
        o.insert("active".into(), Value::Array(active));
        o.insert("timeline".into(), Value::Array(timeline.into_iter().skip(skip).collect()));
        o.insert("serving".into(), n(procs.len() as u64));
        o.insert("processes".into(), Value::Array(processes));
        if primary.is_none() {
            o.insert("backend".into(), Value::Null);
        }
        o.remove("beat_ms");
    }
    out
}

/// What a stream shows as "now": the state, the stages active, the MicroModels executing and the
/// writer.
fn now_line(v: &Value) -> Value {
    let executing: Vec<Value> = arr(v, "models")
        .into_iter()
        .filter(|m| m.get("executing").ok() == Some(&Value::Bool(true)))
        .filter_map(|m| m.get("role").ok().cloned())
        .collect();
    Value::Object(
        [
            ("state".to_owned(), v.get("state").cloned().unwrap_or(Value::Null)),
            ("active".to_owned(), v.get("active").cloned().unwrap_or(Value::Array(Vec::new()))),
            ("executing".to_owned(), Value::Array(executing)),
            ("writer".to_owned(), v.get("writer").cloned().unwrap_or(Value::Null)),
        ]
        .into_iter()
        .collect(),
    )
}

/// The events of `procs` newer than `seen` (by process id and sequence), oldest first; `seen`
/// moves past them.
fn new_events(procs: &[Value], seen: &mut BTreeMap<u64, u64>) -> Vec<Value> {
    let mut out = Vec::new();
    for p in procs {
        let pid = pid_of(p);
        let tl = arr(p, "timeline");
        let top = tl.iter().filter_map(|e| e.get("seq").and_then(Value::integer).ok()).max().unwrap_or(0);
        let last = seen.entry(pid).or_insert(0);
        // A process that started again under the same id begins its sequence anew.
        if top < *last {
            *last = 0;
        }
        let from = *last;
        out.extend(tagged(tl.into_iter().filter(|e| e.get("seq").and_then(Value::integer).unwrap_or(0) > from).collect(), pid));
        *last = top.max(from);
    }
    out.sort_by_key(|e| e.get("at_ms").and_then(Value::integer).unwrap_or(0));
    out
}

/// One Server-Sent Event; false when the client is gone.
fn send_event(stream: &mut std::net::TcpStream, event: &str, v: &Value) -> bool {
    use std::io::Write;
    v.encode().is_ok_and(|body| stream.write_all(format!("event: {event}\ndata: {body}\n\n").as_bytes()).is_ok())
}

/// `GET /api/live/stream`: Server-Sent Events. First `state` with the whole `GET /api/live`
/// state, then `live` for each new timeline event as it is recorded, and `now` whenever the
/// active stages, executing MicroModels or the writer change. Looks every `STREAM_MS` on this
/// connection's own thread; ends when the client goes.
pub fn stream(home: &Path, stream: &mut std::net::TcpStream, cors: Option<&str>) {
    use std::io::Write;
    let allow = cors.map_or_else(String::new, |o| format!("Access-Control-Allow-Origin: {o}\r\nVary: Origin\r\n"));
    let head = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n{allow}Connection: close\r\n\r\n");
    let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(5)));
    let _ = stream.set_nodelay(true);
    if stream.write_all(head.as_bytes()).is_err() {
        return;
    }
    let mut seen: BTreeMap<u64, u64> = BTreeMap::new();
    let mut last_now = String::new();
    let mut quiet_since = nmtype::now_ms();
    let mut first = true;
    loop {
        let procs = read_activity(home);
        let own = nmapi::live::snapshot();
        let extra = deliveries(home);
        let all: Vec<Value> = procs.iter().cloned().chain(std::iter::once(own.clone())).chain(extra.clone()).collect();
        let fresh = new_events(&all, &mut seen);
        let view = merged(&procs, &own, extra.as_ref());
        let now = now_line(&view);
        let text = now.encode().unwrap_or_default();
        let mut sent = false;
        if first {
            // The whole state once; the events already in it are not sent again.
            if !send_event(stream, "state", &view) {
                return;
            }
            first = false;
            sent = true;
        } else {
            for e in &fresh {
                if !send_event(stream, "live", e) {
                    return;
                }
                sent = true;
            }
            if text != last_now {
                if !send_event(stream, "now", &now) {
                    return;
                }
                sent = true;
            }
        }
        last_now = text;
        let t = nmtype::now_ms();
        if sent {
            quiet_since = t;
        } else if t.saturating_sub(quiet_since) >= STREAM_BEAT_MS {
            if stream.write_all(b": beat\n\n").is_err() {
                return;
            }
            quiet_since = t;
        }
        std::thread::park_timeout(std::time::Duration::from_millis(STREAM_MS));
    }
}

#[cfg(test)]
mod live_tests {
    use super::*;

    fn proc_row(pid: u64, state: &str, seqs: &[u64]) -> Value {
        let tl: Vec<Value> = seqs
            .iter()
            .map(|q| Value::Object([("seq".to_owned(), n(*q)), ("at_ms".to_owned(), n(1000 + q)), ("kind".to_owned(), s("stage"))].into_iter().collect()))
            .collect();
        Value::Object(
            [
                ("pid".to_owned(), n(pid)),
                ("state".to_owned(), s(state)),
                ("backend".to_owned(), s("cuda")),
                ("active".to_owned(), Value::Array(Vec::new())),
                ("timeline".to_owned(), Value::Array(tl)),
            ]
            .into_iter()
            .collect(),
        )
    }

    #[test]
    fn merged_prefers_the_busy_process_and_joins_timelines() {
        let own = proc_row(9, "idle", &[1]);
        let v = merged(&[proc_row(1, "idle", &[1, 2]), proc_row(2, "busy", &[5])], &own, None);
        assert_eq!(v.get("pid").and_then(Value::integer).ok(), Some(2));
        assert_eq!(v.get("state").and_then(Value::string).ok(), Some("busy"));
        assert_eq!(v.get("timeline").and_then(Value::array).map(<[Value]>::len).ok(), Some(4));
        assert_eq!(v.get("serving").and_then(Value::integer).ok(), Some(2));
        let none = merged(&[], &own, None);
        assert_eq!(none.get("backend").ok(), Some(&Value::Null));
        assert_eq!(none.get("state").and_then(Value::string).ok(), Some("idle"));
    }

    #[test]
    fn a_stream_sends_each_event_once() {
        let mut seen = BTreeMap::new();
        assert_eq!(new_events(&[proc_row(1, "idle", &[1, 2])], &mut seen).len(), 2);
        assert_eq!(new_events(&[proc_row(1, "idle", &[1, 2, 3])], &mut seen).len(), 1);
        assert!(new_events(&[proc_row(1, "idle", &[2, 3])], &mut seen).is_empty());
        // The same id started again: its sequence begins anew.
        assert_eq!(new_events(&[proc_row(1, "idle", &[1])], &mut seen).len(), 1);
    }

    /// A guidance delivery (a hook process) is one event in `GET /api/live`'s timeline, beside the
    /// serving processes, without becoming a serving process; a stream sends it once.
    #[test]
    fn a_guidance_delivery_joins_the_timeline_once() {
        let home = std::env::temp_dir().join(format!("nano-live-deliveries-{}-{}", std::process::id(), nmtype::now_ms()));
        assert!(deliveries(&home).is_none());
        delivered(&home, "guidance.reminder", "reminder", &[("reminders", 1), ("chars", 40)]);
        let extra = deliveries(&home).unwrap();
        let own = proc_row(9, "idle", &[1]);
        let v = merged(&[proc_row(1, "idle", &[1])], &own, Some(&extra));
        let tl = v.get("timeline").and_then(Value::array).unwrap();
        let g: Vec<&Value> = tl.iter().filter(|e| e.get("kind").and_then(Value::string).ok() == Some("guidance")).collect();
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].get("stage").and_then(Value::string).ok(), Some("guidance.reminder"));
        assert_eq!(g[0].get("counts").and_then(|c| c.get("reminders")).and_then(Value::integer).ok(), Some(1));
        assert_eq!(v.get("serving").and_then(Value::integer).ok(), Some(1), "a delivery is not a serving process");
        let mut seen = BTreeMap::new();
        assert_eq!(new_events(std::slice::from_ref(&extra), &mut seen).len(), 1);
        assert!(new_events(std::slice::from_ref(&deliveries(&home).unwrap()), &mut seen).is_empty());
        delivered(&home, "guidance.loop", "continuation", &[("loops", 1)]);
        assert_eq!(new_events(std::slice::from_ref(&deliveries(&home).unwrap()), &mut seen).len(), 1);
        let _ = std::fs::remove_dir_all(&home);
    }
}
