//! The audit log writer (`<home>/memory/audit.jsonl`), one JSON row per change.
//!
//! Roadmap Phase 4.5 (2026-11-24): moved here from the program's controls (`nmcore`
//! `controls.rs`, which now re-exports it) so the storage controls and the MCP tools in this
//! library log their actions through the same writer as the interface, with the same ids.
use nmjson::Value;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

fn audit_path(home: &Path) -> PathBuf {
    home.join("memory").join("audit.jsonl")
}

/// A fresh audit id for a row written at `now`: `a<ms>-<pid>`, with `.<n>` added when this process
/// already gave out that id in the same millisecond, so two rows never share an id.
#[must_use]
pub fn new_id(now: u64) -> String {
    static LAST: Mutex<(u64, u32)> = Mutex::new((0, 0));
    let n = LAST.lock().map_or(0, |mut g| {
        if g.0 == now {
            g.1 += 1;
        } else {
            *g = (now, 0);
        }
        g.1
    });
    let base = format!("a{now}-{}", std::process::id() % 10_000);
    if n == 0 { base } else { format!("{base}.{n}") }
}

/// Appends one row to the audit log and returns its id.
/// # Errors
/// The memory folder or the log cannot be written.
pub fn audit(home: &Path, kind: &str, fields: Vec<(&str, Value)>) -> Result<String, String> {
    let now = nmtype::now_ms();
    let id = new_id(now);
    audit_row(home, kind, &id, now, fields).map(|()| id)
}

/// Appends one row with a given id and time (a row written again under the same id, as a tool
/// call's running row is when the call ends).
/// # Errors
/// The memory folder or the log cannot be written.
pub fn audit_row(home: &Path, kind: &str, id: &str, now: u64, fields: Vec<(&str, Value)>) -> Result<(), String> {
    let mut o: BTreeMap<String, Value> = fields.into_iter().map(|(k, v)| (k.to_owned(), v)).collect();
    o.insert("id".to_owned(), Value::String(id.to_owned()));
    o.insert("kind".to_owned(), Value::String(kind.to_owned()));
    o.insert("time_ms".to_owned(), Value::Number(now.to_string()));
    let line = Value::Object(o).encode().map_err(|e| e.to_string())?;
    let p = audit_path(home);
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    writeln!(f, "{line}").map_err(|e| format!("{}: {e}", p.display()))
}
