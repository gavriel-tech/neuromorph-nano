//! The memory writes MCP `nm_memory` and `nm_goal` make, as one path the interface's API uses too
//! (2026-11-26, Phase B: every feature in the interface, MCP and the API, in sync).
//!
//! `memory_op` runs a write on an open memory state (`MemoryState`): a working note, a stored claim
//! (through the relevance gate and its never-save screen, then kept as a memory), a temporary note,
//! a sweep of the temporary notes, a pin or unpin. `goal_op` changes the session's objectives. The
//! MCP tools call these on the serving process's own state; the interface's server, a process of
//! its own, calls them on the project's memory opened the way a serving process opens it
//! (`open_state`), and only when no serving process holds the project's memory journal.
//!
//! One writer per memory folder: when a serving process holds the journal (`journal_held`), the
//! interface's request waits in `memory-requests.jsonl` in that folder (`queue`) and the serving
//! process applies it with its own state within seconds (`apply_queued`, called from its idle
//! loop), as the memory deletions of `forget-requests.jsonl` are.
use crate::forage::{MemoryState, Organism};
use crate::ApiError;
use nmjson::Value;
use nmtype::Packet;
use std::path::{Path, PathBuf};

/// The memory writes a running serving process applies for the interface.
pub const REQUESTS: &str = "memory-requests.jsonl";

/// The write ops of `nm_memory` (`memory_op`).
pub const MEMORY_OPS: [&str; 6] = ["note", "store", "temp", "sweep", "pin", "unpin"];

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

/// The counts of the hierarchy's levels, by level name.
#[must_use]
pub fn levels(h: &nmmem::Hierarchy) -> Value {
    let c = h.counts();
    obj(nmmem::hierarchy::LEVELS.iter().zip(c).map(|(l, k)| (*l, n(k))).collect())
}

/// One write of `nm_memory` on an open memory state: `note` (working memory, this session only),
/// `store` (a claim through the relevance gate; admitted, it is kept as a memory when the memory
/// journal is open), `temp` (a temporary note, saved when used on three run-days), `sweep`, `pin`
/// and `unpin` (by record or episode id in `text`).
/// # Errors
/// An op that is not a write, never-save text, an unknown id, the journal not open for a pin, or a
/// write fault.
pub fn memory_op(m: &mut MemoryState, op: &str, v: &Value) -> Result<Value, ApiError> {
    let started = std::time::Instant::now();
    let out = memory_write(m, op, v);
    // The live view shows a memory write on the memory part: its fixed op name and its time only.
    if out.is_ok() {
        let label = MEMORY_OPS.iter().copied().find(|o| *o == op).unwrap_or("");
        crate::live::event("memory", "memory.write", "", label, u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX), &[]);
    }
    out
}

fn memory_write(m: &mut MemoryState, op: &str, v: &Value) -> Result<Value, ApiError> {
    let text = arg(v, "text").unwrap_or_default();
    match op {
        "note" => {
            m.working.note(arg(v, "scope").unwrap_or("brain"), text);
            Ok(obj(vec![("noted", Value::Bool(true))]))
        }
        "store" => {
            // A claim offered for storage goes through the gate like any other write.
            let src = arg(v, "source").unwrap_or("unit:claude-code");
            let p = Packet::new("memory", &format!("note:{text}"), text, nmtype::Provenance::new(src, "nmapi.tools", 0, nmtype::now_ms()))
                .status(nmtype::Status::Observed)
                .confidence(1.0)
                .evidence(vec![src.to_owned()]);
            Ok(match m.gate.admit(&p) {
                // 2026-11-26: an admitted claim is kept as a memory (journal, tier views) so it is
                // recalled and listed; before, the gate's answer was the whole of `store`.
                Ok(()) => match m.mem2.as_mut() {
                    Some(m2) => {
                        let id = m2.keep(text, src).map_err(|e| fault(&e))?;
                        obj(vec![("admitted", Value::Bool(true)), ("stored", s(&id))])
                    }
                    None => obj(vec![
                        ("admitted", Value::Bool(true)),
                        ("stored", Value::Null),
                        ("reason", s(&format!("the memory journal is not open here ({}); nothing was kept", m.mem2_fault.as_deref().unwrap_or("memory writes are off")))),
                    ]),
                },
                Err(why) => obj(vec![("admitted", Value::Bool(false)), ("reason", s(&why))]),
            })
        }
        "temp" => {
            let id = m.temp.note(arg(v, "scope").unwrap_or("brain"), text).map_err(|e| fault(&e))?;
            Ok(obj(vec![("temp_note", s(&id)), ("held", n(m.temp.len()))]))
        }
        "sweep" => {
            let st = &mut *m;
            let (promoted, cleared, up, down) = crate::session::sweep(&mut st.temp, &mut st.episodic, &mut st.hierarchy);
            Ok(obj(vec![
                ("promoted", Value::Array(promoted.iter().map(|p| s(p)).collect())),
                ("cleared", n(cleared)),
                ("kept", n(st.temp.len())),
                ("moved_up", n(up)),
                ("moved_down", n(down)),
                ("levels", levels(&st.hierarchy)),
            ]))
        }
        "pin" | "unpin" => {
            let m2 = m.mem2.as_mut().ok_or_else(|| ApiError::new("nm-409", "memory revision 2 is not open"))?;
            let id = m2.pin(text, op == "pin").map_err(|e| fault(&e))?;
            Ok(obj(vec![("id", s(&id)), ("class", s(if op == "pin" { "pinned" } else { "normal" }))]))
        }
        other => Err(ApiError::new("nm-400", format!("{other} is not a memory write (use {})", MEMORY_OPS.join(", ")))),
    }
}

/// The objectives as `nm_goal` returns them.
#[must_use]
pub fn objectives_value(g: &[(u64, String, String)]) -> Value {
    Value::Array(g.iter().map(|(id, t, st)| obj(vec![("id", Value::Number(id.to_string())), ("objective", s(t)), ("status", s(st))])).collect())
}

/// One change of `nm_goal` to a list of objectives: `objective` adds one (open), `id` with
/// `status` (`done` by default, `dropped` or `open`) closes or opens one again. Returns whether
/// anything changed.
/// # Errors
/// A bad status or an unknown id.
pub fn goal_op(g: &mut Vec<(u64, String, String)>, v: &Value) -> Result<bool, ApiError> {
    let mut changed = false;
    if let Some(text) = arg(v, "objective") {
        if text.trim().is_empty() {
            return Err(ApiError::new("nm-400", "objective: give the objective's text"));
        }
        // Ids are the time in milliseconds; two added in one millisecond still differ.
        let id = g.iter().map(|(i, _, _)| *i + 1).max().unwrap_or(0).max(nmtype::now_ms());
        g.push((id, text.to_owned(), "open".into()));
        changed = true;
    }
    // 2026-11-25: an objective can be closed (done or dropped) or opened again by its id.
    if let Ok(id) = v.get("id").and_then(Value::integer) {
        let status = arg(v, "status").unwrap_or("done");
        if !matches!(status, "open" | "done" | "dropped") {
            return Err(ApiError::new("nm-400", "status: open, done or dropped"));
        }
        let row = g.iter_mut().find(|(i, _, _)| *i == id).ok_or_else(|| ApiError::new("nm-404", format!("no objective {id}")))?;
        row.2 = status.to_owned();
        changed = true;
    }
    Ok(changed)
}

/// `nm_goal` on the serving process's organism: the change (if any), saved in the project's memory
/// when memory is on, and the objectives.
/// # Errors
/// As `goal_op`, or the objectives cannot be saved.
pub fn goal_on(o: &Organism, v: &Value) -> Result<Value, ApiError> {
    let mut g = o.objectives.lock().map_err(|_| ApiError::new("nm-500", "goal lock"))?;
    let changed = goal_op(&mut g, v)?;
    if changed && o.memory_on {
        let dir = o.memory.lock().map(|m| m.dir.clone()).map_err(|_| ApiError::new("nm-500", "memory lock"))?;
        nmmem::structural::save_objectives(&nmmem::hierarchy::core_dir(&dir), &g).map_err(|e| fault(&e))?;
    }
    Ok(obj(vec![
        ("objectives", objectives_value(&g)),
        ("note", s("objectives are kept in this project's memory; close one with its id and status done or dropped")),
    ]))
}

/// The objectives saved in a memory folder (what a serving process starts with).
#[must_use]
pub fn saved_objectives(memory: &Path) -> Vec<(u64, String, String)> {
    nmmem::structural::load_objectives(&nmmem::hierarchy::core_dir(memory))
}

/// Saves objectives in a memory folder.
/// # Errors
/// The file cannot be written.
pub fn save_objectives(memory: &Path, g: &[(u64, String, String)]) -> Result<(), ApiError> {
    nmmem::structural::save_objectives(&nmmem::hierarchy::core_dir(memory), g).map_err(|e| fault(&e))
}

/// Whether another process holds the memory journal of `memory` (a serving process on that
/// project). On Windows the journal's lock file is opened without sharing, as the journal itself
/// opens it; elsewhere the lock does not exclude other processes and this reads false.
#[must_use]
pub fn journal_held(memory: &Path) -> bool {
    let lock = memory.join("journal").join("LOCK");
    if !lock.is_file() {
        return false;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        let mut o = std::fs::OpenOptions::new();
        o.write(true).share_mode(0);
        o.open(&lock).is_err()
    }
    #[cfg(not(windows))]
    {
        false
    }
}

/// The studied folder of a memory folder (its `project.json` `root`), else `fallback`.
#[must_use]
pub fn root_of(memory: &Path, fallback: &Path) -> PathBuf {
    std::fs::read(memory.join("project.json"))
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .and_then(|v| v.get("root").and_then(Value::string).ok().map(PathBuf::from))
        .unwrap_or_else(|| fallback.to_path_buf())
}

/// Opens a memory folder for writing as a serving process opens it (`session::memory_state`),
/// when no serving process holds its journal; `None` when one does (queue the request instead).
#[must_use]
pub fn open_state(home: &Path, memory: &Path) -> Option<MemoryState> {
    if journal_held(memory) {
        return None;
    }
    let st = crate::session::memory_state(home, &root_of(memory, home), memory, true);
    // A serving process that took the journal between the check and the open: queue.
    if st.mem2.is_none() && st.mem2_fault.as_deref().is_some_and(|f| f.contains("held")) {
        return None;
    }
    Some(st)
}

/// The organism's memory state when this process serves `memory` itself (an MCP call into the
/// interface's routes in the serving process).
#[must_use]
pub fn served_here(memory: &Path) -> Option<&'static Organism> {
    let o = crate::tools::installed()?;
    let dir = o.memory.lock().ok()?.dir.clone();
    (o.memory_on && same_dir(&dir, memory)).then_some(o)
}

fn same_dir(a: &Path, b: &Path) -> bool {
    a == b || matches!((std::fs::canonicalize(a), std::fs::canonicalize(b)), (Ok(x), Ok(y)) if x == y)
}

/// Queues a request for the serving process that holds `memory`'s journal.
/// # Errors
/// The request file cannot be written.
pub fn queue(memory: &Path, v: &Value, by: &str) -> Result<(), String> {
    let mut row = match v {
        Value::Object(m) => m.clone(),
        _ => std::collections::BTreeMap::new(),
    };
    row.insert("at".into(), Value::Number(nmtype::now_ms().to_string()));
    row.insert("by".into(), s(by));
    let line = Value::Object(row).encode().map_err(|e| e.to_string())?;
    let path = memory.join(REQUESTS);
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    std::io::Write::write_all(&mut f, format!("{line}\n").as_bytes()).map_err(|e| format!("{}: {e}", path.display()))
}

/// The serving process applies the interface's queued memory writes with its own state (its idle
/// loop calls this every few seconds): memory writes (`op` one of `MEMORY_OPS`), objectives
/// (`op` `goal`) and a world-model refresh (`op` `world_model_refresh`, which empties the idle
/// writer's notes so it writes L1 to L4 again). Returns (applied, refused).
pub fn apply_queued(o: &Organism) -> (usize, usize) {
    let Some(memory) = o.memory.lock().ok().map(|m| m.dir.clone()) else { return (0, 0) };
    let path = memory.join(REQUESTS);
    let Ok(raw) = std::fs::read_to_string(&path) else { return (0, 0) };
    let _ = std::fs::remove_file(&path);
    let (mut done, mut refused) = (0, 0);
    for line in raw.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(v) = nmjson::parse(line.as_bytes()) else {
            refused += 1;
            continue;
        };
        let op = arg(&v, "op").unwrap_or("");
        let r = match op {
            "goal" => goal_on(o, &v).map(|_| ()),
            "world_model_refresh" => {
                crate::writer_idle::clear_store();
                Ok(())
            }
            op if MEMORY_OPS.contains(&op) => match o.memory.lock() {
                Ok(mut m) => memory_op(&mut m, op, &v).and_then(|r| if matches!(r.get("admitted"), Ok(Value::Bool(false))) { Err(ApiError::new("nm-400", "refused by the gate")) } else { Ok(()) }),
                Err(_) => Err(ApiError::new("nm-500", "memory lock")),
            },
            _ => Err(ApiError::new("nm-400", "unknown request")),
        };
        if r.is_ok() { done += 1 } else { refused += 1 }
    }
    (done, refused)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(pairs: &[(&str, Value)]) -> Value {
        Value::Object(pairs.iter().map(|(k, v)| ((*k).to_owned(), v.clone())).collect())
    }

    /// Objectives: one is added open, closed by its id, a bad status and an unknown id refused.
    #[test]
    fn goals_add_close_and_refuse() {
        let mut g = Vec::new();
        assert!(goal_op(&mut g, &input(&[("objective", s("ship v0.1.0"))])).unwrap());
        let id = g[0].0;
        assert_eq!(g[0].2, "open");
        assert!(goal_op(&mut g, &input(&[("id", Value::Number(id.to_string())), ("status", s("done"))])).unwrap());
        assert_eq!(g[0].2, "done");
        assert!(goal_op(&mut g, &input(&[("id", Value::Number(id.to_string())), ("status", s("later"))])).is_err());
        assert!(goal_op(&mut g, &input(&[("id", Value::Number("7".into()))])).is_err());
        assert!(!goal_op(&mut g, &input(&[])).unwrap());
        assert!(goal_op(&mut g, &input(&[("objective", s(" "))])).is_err());
    }

    /// A memory folder opened for writing stores an admitted claim as a memory listed in its tier
    /// view, refuses a secret, and pins the stored memory; a queued request is written as one line.
    #[test]
    fn store_keeps_a_memory_and_refuses_a_secret() {
        let home = std::env::temp_dir().join(format!("nm-memwrite-{}", std::process::id()));
        let memory = home.join("memory");
        std::fs::create_dir_all(&memory).unwrap();
        assert!(!journal_held(&memory));
        {
            let mut st = open_state(&home, &memory).expect("not held");
            let r = memory_op(&mut st, "store", &input(&[("text", s("the release ships on the 27th")), ("source", s("user:interface"))])).unwrap();
            assert!(matches!(r.get("admitted"), Ok(Value::Bool(true))), "{r:?}");
            let id = r.get("stored").and_then(Value::string).unwrap().to_owned();
            let secret = memory_op(&mut st, "store", &input(&[("text", s("the key is sk-ABCdef1234567890abcdefGHIJ"))])).unwrap();
            assert!(matches!(secret.get("admitted"), Ok(Value::Bool(false))));
            let p = memory_op(&mut st, "pin", &input(&[("text", s(&id))])).unwrap();
            assert_eq!(p.get("class").and_then(Value::string).unwrap(), "pinned");
            assert!(memory_op(&mut st, "recall", &Value::Null).is_err());
        }
        let listed = std::fs::read_to_string(memory.join("2-active/notes/items.json")).unwrap_or_default();
        assert!(listed.contains("the release ships on the 27th"), "{listed}");
        queue(&memory, &input(&[("op", s("temp")), ("text", s("x"))]), "test").unwrap();
        let q = std::fs::read_to_string(memory.join(REQUESTS)).unwrap();
        assert_eq!(q.lines().count(), 1);
        let _ = std::fs::remove_dir_all(&home);
    }
}
