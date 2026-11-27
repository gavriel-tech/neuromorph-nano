//! Memory authoring, objectives and the world-model refresh over the interface's API (2026-11-26,
//! Phase B items B1, B2 and B3; directions of Luis Jake Gabriel III: every feature is in the
//! interface, MCP and the API, in sync).
//!
//! Every write runs the function MCP runs (`nmapi::memory_write`): a stored note passes the same
//! relevance gate and never-save screen as `nm_memory` `store`, objectives change as `nm_goal`
//! changes them. One writer per memory folder: when this process serves the project itself (an MCP
//! call into these routes) the write runs on its own memory at once; when another process serves
//! it (it holds the memory journal) the request waits in `memory-requests.jsonl` and that process
//! applies it within seconds (the reply says `queued`); otherwise the memory folder is opened as a
//! serving process opens it and the write runs at once (`now`). Audit rows name what was done and
//! the ids, never the text.
use nmapi::memory_write as mw;
use nmjson::Value;
use std::path::{Path, PathBuf};

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}
fn obj(f: Vec<(&str, Value)>) -> Value {
    Value::Object(f.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn text(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::string).ok().map(str::to_owned)
}

/// The longest note the interface stores.
const MAX_TEXT: usize = 4000;

/// A project's key and memory folder.
fn memory_of(home: &Path, key: &str) -> Result<(String, PathBuf), String> {
    crate::ledger::project_memory(home, &format!("project={key}")).ok_or_else(|| format!("project: no project {key} (give a key as /api/projects lists it, or nano)"))
}

/// Where a write runs and what it answered: (`now` or `queued`, the reply of the shared function).
fn run(home: &Path, memory: &Path, req: &Value, by: &str) -> Result<(&'static str, Value), String> {
    let op = text(req, "op").unwrap_or_default();
    if let Some(o) = mw::served_here(memory) {
        let r = match op.as_str() {
            "goal" => mw::goal_on(o, req),
            "world_model_refresh" => Ok(obj(vec![("notes_dropped", n(nmapi::writer_idle::clear_store() as u64))])),
            _ => {
                let mut m = o.memory.lock().map_err(|_| "memory lock poisoned".to_owned())?;
                mw::memory_op(&mut m, &op, req)
            }
        };
        return r.map(|v| ("now", v)).map_err(|e| e.message);
    }
    if mw::journal_held(memory) {
        // A working note belongs to the serving process's session; it is queued like the rest.
        mw::queue(memory, req, by)?;
        return Ok(("queued", Value::Null));
    }
    match op.as_str() {
        "note" => Err("a working note lives in a serving process's memory for its session, and no serving process holds this project now; keep it as a temp note or store it".into()),
        "goal" => {
            let mut g = mw::saved_objectives(memory);
            if mw::goal_op(&mut g, req).map_err(|e| e.message)? {
                mw::save_objectives(memory, &g).map_err(|e| e.message)?;
            }
            Ok(("now", obj(vec![("objectives", mw::objectives_value(&g))])))
        }
        _ => {
            let mut st = match mw::open_state(home, memory) {
                Some(st) => st,
                None => {
                    mw::queue(memory, req, by)?;
                    return Ok(("queued", Value::Null));
                }
            };
            mw::memory_op(&mut st, &op, req).map(|v| ("now", v)).map_err(|e| e.message)
        }
    }
}

fn queued_note(applied: &str) -> &'static str {
    if applied == "queued" {
        "a running Nano holds this project's memory: the request is queued and that process applies it within seconds"
    } else {
        "done"
    }
}

/// `POST /api/memory/add {project, kind: store|note|temp, text[, scope]}`: a note through the same
/// write path as `nm_memory`: `store` passes the relevance gate and its never-save screen
/// (credentials, keys, tokens, e-mail addresses are refused) and is kept as a memory; `temp` is a
/// temporary note (saved when used on three run-days, cleared when unused seven); `note` is a
/// working note of the serving process's session. Audited (kind and ids, never the text).
/// # Errors
/// An unknown project or kind, no text, or the write failing.
pub fn add(home: &Path, v: &Value, by: &str) -> Result<Value, String> {
    let key = text(v, "project").ok_or("project: give a project key")?;
    let kind = text(v, "kind").unwrap_or_else(|| "store".into());
    if !["store", "note", "temp"].contains(&kind.as_str()) {
        return Err(format!("kind {kind}: use store, note or temp"));
    }
    let body = text(v, "text").map(|t| t.trim().to_owned()).filter(|t| !t.is_empty()).ok_or("text: give the note's text")?;
    if body.chars().count() > MAX_TEXT {
        return Err(format!("text: at most {MAX_TEXT} characters"));
    }
    let (key, memory) = memory_of(home, &key)?;
    let mut req = vec![("op", s(&kind)), ("text", s(&body)), ("source", s(&format!("user:{by}")))];
    if let Some(sc) = text(v, "scope").filter(|x| !x.is_empty()) {
        req.push(("scope", s(&sc)));
    }
    let (applied, result) = run(home, &memory, &obj(req), by)?;
    let admitted = !matches!(result.get("admitted"), Ok(Value::Bool(false)));
    let stored = text(&result, "stored").or_else(|| text(&result, "temp_note"));
    let audit = crate::controls::audit(
        home,
        "memory",
        vec![
            ("action", s(&kind)),
            ("project", s(&key)),
            ("applied", s(applied)),
            ("admitted", Value::Bool(admitted)),
            ("memory", stored.as_deref().map_or(Value::Null, s)),
            ("by", s(by)),
            ("label", s(&format!("Memory {kind} in {key}{}", if admitted { "" } else { ": refused by the gate" }))),
        ],
    )?;
    Ok(obj(vec![("project", s(&key)), ("kind", s(&kind)), ("applied", s(applied)), ("result", result), ("audit_id", s(&audit)), ("note", s(queued_note(applied)))]))
}

/// `POST /api/memory/pin {project, id, pin: bool}`: pins a memory (it never drops, merges or leaves
/// its tier by itself) or unpins it, through `nm_memory`'s pin. Audited; the Audit page undoes it.
/// # Errors
/// An unknown project, no id, or the pin failing (an unknown memory).
pub fn pin(home: &Path, v: &Value, by: &str) -> Result<Value, String> {
    let key = text(v, "project").ok_or("project: give a project key")?;
    let id = text(v, "id").filter(|x| !x.is_empty()).ok_or("id: give the memory's id")?;
    let on = match v.get("pin") {
        Ok(Value::Bool(b)) => *b,
        _ => return Err("pin: true to pin, false to unpin".into()),
    };
    let (key, memory) = memory_of(home, &key)?;
    pin_in(home, &key, &memory, &id, on, None, by)
}

#[expect(clippy::too_many_arguments, reason = "a pin names its project, folder, memory, direction, the row it undoes and who asked")]
fn pin_in(home: &Path, key: &str, memory: &Path, id: &str, on: bool, undo_of: Option<&str>, by: &str) -> Result<Value, String> {
    let op = if on { "pin" } else { "unpin" };
    let (applied, result) = run(home, memory, &obj(vec![("op", s(op)), ("text", s(id))]), by)?;
    let mut fields = vec![
        ("action", s(op)),
        ("project", s(key)),
        ("memory", s(id)),
        ("applied", s(applied)),
        ("by", s(by)),
        ("label", s(&format!("Memory {id} in {key} {}", if on { "pinned" } else { "unpinned" }))),
    ];
    if let Some(u) = undo_of {
        fields.push(("undo_of", s(u)));
    }
    let audit = crate::controls::audit(home, "memory", fields)?;
    Ok(obj(vec![("project", s(key)), ("id", s(id)), ("pinned", Value::Bool(on)), ("applied", s(applied)), ("result", result), ("audit_id", s(&audit)), ("note", s(queued_note(applied)))]))
}

/// Undoes a pin or unpin by its audit row: the other one.
/// # Errors
/// A row that is not a pin, or the pin failing.
pub fn undo(home: &Path, row: &Value, id: &str, by: &str) -> Result<Option<String>, String> {
    let (action, key, mid) = (text(row, "action").unwrap_or_default(), text(row, "project").unwrap_or_default(), text(row, "memory").unwrap_or_default());
    if !matches!(action.as_str(), "pin" | "unpin") {
        return Err(format!("memory {action} is not undone here (a stored memory is deleted from the memory list, in two steps)"));
    }
    let (key, memory) = memory_of(home, &key)?;
    let r = pin_in(home, &key, &memory, &mid, action == "unpin", Some(id), by)?;
    Ok(text(&r, "audit_id"))
}

/// `POST /api/memory/sweep {project}`: one sweep of the temporary notes, as `nm_memory` `sweep`:
/// notes used on three run-days are saved, notes unused seven run-days are cleared, then saved items
/// move between tiers by use. Audited.
/// # Errors
/// An unknown project, or the sweep failing.
pub fn sweep(home: &Path, v: &Value, by: &str) -> Result<Value, String> {
    let key = text(v, "project").ok_or("project: give a project key")?;
    let (key, memory) = memory_of(home, &key)?;
    let (applied, result) = run(home, &memory, &obj(vec![("op", s("sweep"))]), by)?;
    let num = |k: &str| result.get(k).and_then(Value::integer).unwrap_or(0);
    let promoted = result.get("promoted").and_then(Value::array).map_or(0, <[Value]>::len) as u64;
    let audit = crate::controls::audit(
        home,
        "memory",
        vec![
            ("action", s("sweep")),
            ("project", s(&key)),
            ("applied", s(applied)),
            ("promoted", n(promoted)),
            ("cleared", n(num("cleared"))),
            ("by", s(by)),
            ("label", s(&format!("Temp notes of {key} swept"))),
        ],
    )?;
    Ok(obj(vec![("project", s(&key)), ("applied", s(applied)), ("result", result), ("audit_id", s(&audit)), ("note", s(queued_note(applied)))]))
}

/// `GET /api/goals?project=<key>`: the session objectives (`nm_goal`) as kept in the project's
/// memory, or as this process holds them when it serves the project.
#[must_use]
pub fn goals_view(home: &Path, query: &str) -> Value {
    let Some((key, memory)) = crate::ledger::project_memory(home, query) else {
        return obj(vec![("error", s("project: give the key of a project as /api/projects lists it, or nano"))]);
    };
    let g = match mw::served_here(&memory) {
        Some(o) => o.objectives.lock().map(|g| g.clone()).unwrap_or_default(),
        None => mw::saved_objectives(&memory),
    };
    let open = g.iter().filter(|(_, _, st)| st == "open").count() as u64;
    obj(vec![
        ("project", s(&key)),
        ("objectives", mw::objectives_value(&g)),
        ("open", n(open)),
        ("total", n(g.len() as u64)),
        ("served", Value::Bool(mw::journal_held(&memory) || mw::served_here(&memory).is_some())),
        ("file", s(&nmmem::hierarchy::core_dir(&memory).display().to_string())),
    ])
}

/// `POST /api/goals {project, objective}` adds an objective (open); `{project, id, status}` closes
/// one (`done`, `dropped`) or opens it again (`open`), as `nm_goal`. Audited; the Audit page undoes
/// it (an added objective is dropped, a status set back).
/// # Errors
/// An unknown project or objective, a bad status, or the write failing.
pub fn goals_set(home: &Path, v: &Value, by: &str) -> Result<Value, String> {
    let key = text(v, "project").ok_or("project: give a project key")?;
    let (key, memory) = memory_of(home, &key)?;
    let mut req = vec![("op", s("goal"))];
    let added = text(v, "objective").filter(|t| !t.trim().is_empty());
    if let Some(t) = &added {
        if t.chars().count() > MAX_TEXT {
            return Err(format!("objective: at most {MAX_TEXT} characters"));
        }
        req.push(("objective", s(t)));
    }
    let id = v.get("id").and_then(Value::integer).ok();
    let before_list = match mw::served_here(&memory) {
        Some(o) => o.objectives.lock().map(|g| g.clone()).unwrap_or_default(),
        None => mw::saved_objectives(&memory),
    };
    if let Some(i) = id {
        req.push(("id", n(i)));
        req.push(("status", s(&text(v, "status").unwrap_or_else(|| "done".into()))));
    }
    if added.is_none() && id.is_none() {
        return Err("give objective (to add one) or id with status (to close or open one)".into());
    }
    let from = id.and_then(|i| before_list.iter().find(|(x, _, _)| *x == i).map(|(_, _, st)| st.clone()));
    if id.is_some() && from.is_none() {
        return Err(format!("no objective {}", id.unwrap_or(0)));
    }
    let (applied, result) = run(home, &memory, &obj(req.clone()), by)?;
    let new_id = added.as_ref().and_then(|_| {
        result.get("objectives").and_then(Value::array).ok().and_then(|a| a.iter().filter_map(|o| o.get("id").and_then(Value::integer).ok()).filter(|i| !before_list.iter().any(|(x, _, _)| x == i)).max())
    });
    let mut fields = vec![("project", s(&key)), ("applied", s(applied)), ("by", s(by))];
    match (id, &added) {
        (Some(i), _) => {
            let to = text(v, "status").unwrap_or_else(|| "done".into());
            fields.extend([("action", s("status")), ("objective", n(i)), ("from", s(&from.unwrap_or_default())), ("to", s(&to)), ("label", s(&format!("Objective {i} of {key}: {to}")))]);
        }
        (None, _) => fields.extend([("action", s("add")), ("objective", new_id.map_or(Value::Null, n)), ("label", s(&format!("Objective added in {key}")))]),
    }
    let audit = crate::controls::audit(home, "goal", fields)?;
    Ok(obj(vec![("project", s(&key)), ("applied", s(applied)), ("result", result), ("audit_id", s(&audit)), ("goals", goals_view(home, &format!("project={key}"))), ("note", s(queued_note(applied)))]))
}

/// Undoes an objective change by its audit row: an added objective is dropped; a status change is
/// set back.
/// # Errors
/// A row without an id (an add that was queued), or the change failing.
pub fn undo_goal(home: &Path, row: &Value, id: &str, by: &str) -> Result<Option<String>, String> {
    let key = text(row, "project").unwrap_or_default();
    let gid = row.get("objective").and_then(Value::integer).map_err(|_| "this objective's id was not known when it was added (it was queued); close it from the goals list".to_owned())?;
    let status = match text(row, "action").as_deref() {
        Some("add") => "dropped".to_owned(),
        Some("status") => text(row, "from").filter(|f| !f.is_empty()).ok_or("the row has no earlier status")?,
        _ => return Err("not an objective change".into()),
    };
    let (key, memory) = memory_of(home, &key)?;
    let (applied, _) = run(home, &memory, &obj(vec![("op", s("goal")), ("id", n(gid)), ("status", s(&status))]), by)?;
    crate::controls::audit(
        home,
        "goal",
        vec![("action", s("undo")), ("project", s(&key)), ("objective", n(gid)), ("to", s(&status)), ("applied", s(applied)), ("undo_of", s(id)), ("by", s(by)), ("label", s(&format!("Undone: objective {gid} of {key} set to {status}")))],
    )
    .map(Some)
}

/// What a refresh removes: (project key, note files with their bytes, notes, served).
pub type RefreshPlan = (String, Vec<(PathBuf, u64)>, u64, bool);

/// The world-model notes of a project a refresh removes: the note store and the older one it is
/// carried over from (both, or a start would bring the older back), with their bytes, the notes
/// counted, and whether a serving process holds the project (then it empties its own notes).
/// # Errors
/// An unknown project.
pub fn refresh_plan(home: &Path, key: &str) -> Result<RefreshPlan, String> {
    let (key, memory) = memory_of(home, key)?;
    let files: Vec<(PathBuf, u64)> = ["world-model/notes.jsonl", "writer/notes.jsonl"].iter().map(|r| memory.join(r)).filter(|p| p.is_file()).map(|p| {
        let b = std::fs::metadata(&p).map_or(0, |m| m.len());
        (p, b)
    }).collect();
    let notes = files.first().and_then(|(p, _)| std::fs::read_to_string(p).ok()).map_or(0, |t| t.lines().filter(|l| l.contains("\"k\"")).map(|l| l.split("\"k\"").nth(1).unwrap_or("")).collect::<std::collections::BTreeSet<_>>().len() as u64);
    let served = mw::served_here(&memory).is_some() || mw::journal_held(&memory);
    if files.is_empty() && !served {
        return Err(format!("{key} has no world-model notes yet; nothing to refresh"));
    }
    Ok((key, files, notes, served))
}

/// `POST /api/comprehension/refresh {project}` (after its confirmation): the project's world-model
/// notes go so the idle writer writes L1 to L4 again from the material. With no serving process on
/// the project the note files are deleted now; a serving process empties its own notes (the request
/// waits for it, or runs at once when this process serves it). Audited; it cannot be undone.
/// # Errors
/// An unknown project, or a removal failing.
pub fn refresh(home: &Path, key: &str, by: &str) -> Result<Value, String> {
    let (key, files, notes, served) = refresh_plan(home, key)?;
    let (_, memory) = memory_of(home, &key)?;
    let bytes: u64 = files.iter().map(|(_, b)| b).sum();
    let applied = if served {
        run(home, &memory, &obj(vec![("op", s("world_model_refresh"))]), by)?.0
    } else {
        for (f, _) in &files {
            std::fs::remove_file(f).map_err(|e| format!("{}: {e}", f.display()))?;
        }
        "now"
    };
    let audit = crate::controls::audit(
        home,
        "world-model-refresh",
        vec![("project", s(&key)), ("notes", n(notes)), ("bytes", n(bytes)), ("applied", s(applied)), ("by", s(by)), ("label", s(&format!("World model of {key} refreshed: {notes} notes to write again")))],
    )?;
    Ok(obj(vec![
        ("project", s(&key)),
        ("notes", n(notes)),
        ("bytes", n(bytes)),
        ("applied", s(applied)),
        ("audit_id", s(&audit)),
        ("note", s(if served { "the serving process empties its notes (within seconds when queued) and its idle writer writes L1 to L4 again" } else { "the note files are gone; the idle writer writes L1 to L4 again the next time the project is served" })),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stored note goes through the gate into the memory list, a secret is refused, a pin is
    /// undone, an objective is added, closed and undone, and a refresh plan lists the note store.
    #[test]
    fn memory_and_goals_through_the_shared_path() {
        let home = std::env::temp_dir().join(format!("nm-memapi-{}", std::process::id()));
        std::fs::create_dir_all(home.join("memory")).unwrap();
        let body = |pairs: &[(&str, Value)]| obj(pairs.iter().map(|(k, v)| (*k, v.clone())).collect());
        let r = add(&home, &body(&[("project", s("nano")), ("kind", s("store")), ("text", s("ship on the 27th"))]), "test").unwrap();
        assert_eq!(text(&r, "applied").as_deref(), Some("now"));
        let id = r.get("result").and_then(|x| x.get("stored")).and_then(Value::string).unwrap().to_owned();
        let bad = add(&home, &body(&[("project", s("nano")), ("kind", s("store")), ("text", s("the key is sk-ABCdef1234567890abcdefGHIJ"))]), "test").unwrap();
        assert!(matches!(bad.get("result").and_then(|x| x.get("admitted")), Ok(Value::Bool(false))));
        assert!(add(&home, &body(&[("project", s("nano")), ("kind", s("note")), ("text", s("x"))]), "test").is_err(), "no session to hold a working note");
        assert!(add(&home, &body(&[("project", s("nope")), ("text", s("x"))]), "test").is_err());
        let p = pin(&home, &body(&[("project", s("nano")), ("id", s(&id)), ("pin", Value::Bool(true))]), "test").unwrap();
        let log = std::fs::read_to_string(home.join("memory/audit.jsonl")).unwrap();
        let row = log.lines().filter_map(|l| nmjson::parse(l.as_bytes()).ok()).find(|v| text(v, "id") == text(&p, "audit_id")).unwrap();
        assert!(undo(&home, &row, &text(&p, "audit_id").unwrap(), "test").unwrap().is_some());
        let g = goals_set(&home, &body(&[("project", s("nano")), ("objective", s("finish phase B"))]), "test").unwrap();
        let gid = g.get("goals").and_then(|x| x.get("objectives")).and_then(Value::array).unwrap()[0].get("id").and_then(Value::integer).unwrap();
        goals_set(&home, &body(&[("project", s("nano")), ("id", n(gid)), ("status", s("done"))]), "test").unwrap();
        assert!(goals_set(&home, &body(&[("project", s("nano")), ("id", n(7)), ("status", s("done"))]), "test").is_err());
        let v = goals_view(&home, "project=nano");
        assert_eq!(v.get("open").and_then(Value::integer).unwrap(), 0);
        let log = std::fs::read_to_string(home.join("memory/audit.jsonl")).unwrap();
        let row = log.lines().filter_map(|l| nmjson::parse(l.as_bytes()).ok()).rfind(|v| text(v, "kind").as_deref() == Some("goal")).unwrap();
        undo_goal(&home, &row, &text(&row, "id").unwrap_or_default(), "test").unwrap();
        assert_eq!(goals_view(&home, "project=nano").get("open").and_then(Value::integer).unwrap(), 1, "the close is undone");
        std::fs::create_dir_all(home.join("memory/world-model")).unwrap();
        std::fs::write(home.join("memory/world-model/notes.jsonl"), "{\"k\":\"note:a\",\"by\":\"w\",\"t\":\"x\"}\n").unwrap();
        let (_, files, notes, served) = refresh_plan(&home, "nano").unwrap();
        assert_eq!((files.len(), notes, served), (1, 1, false));
        refresh(&home, "nano", "test").unwrap();
        assert!(!home.join("memory/world-model/notes.jsonl").exists());
        let _ = std::fs::remove_dir_all(&home);
    }
}
