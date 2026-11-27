//! The approval queue (PLAN 4c `approval`): every proposed diff gets an id and is recorded,
//! append-only, with its finding. Nothing here applies a diff; the record says who may apply it
//! (Claude Code, after a Step Check) and the Step Check verdict when one is attached.
use nmjson::Value;
use std::path::Path;

fn s(t: &str) -> Value {
    Value::String(t.to_owned())
}

/// Record a proposal in `<dir>/proposals.jsonl`; returns its id (`p` + the diff's hash prefix).
/// # Errors
/// The file cannot be written.
pub fn record(dir: &Path, finding: &str, diff: &str) -> Result<String, String> {
    let id = format!("p{}", nmtype::now_ms());
    let v = Value::Object(
        [
            ("id", s(&id)),
            ("finding", s(finding)),
            ("diff", s(diff)),
            ("applied_by_mini", Value::Bool(false)),
            ("applies", s("Claude Code, after nm_stepcheck")),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect(),
    );
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let line = v.encode().map_err(|e| e.to_string())?;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("proposals.jsonl"))
        .and_then(|mut f| std::io::Write::write_all(&mut f, format!("{line}\n").as_bytes()))
        .map_err(|e| e.to_string())?;
    Ok(id)
}

/// Section check: two proposals get two ids and two records; none is applied.
/// # Errors
/// One line naming what failed.
pub fn check() -> Result<(), String> {
    let dir = crate::scratch_dir("approval")?;
    let a = record(&dir, "rule word", "--- a/x\n+++ b/x\n")?;
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let _ = rx.recv_timeout(std::time::Duration::from_millis(2));
    drop(tx);
    let b = record(&dir, "clippy", "--- a/y\n+++ b/y\n")?;
    let text = std::fs::read_to_string(dir.join("proposals.jsonl")).map_err(|e| e.to_string())?;
    if a == b || text.lines().count() != 2 || text.contains("\"applied_by_mini\":true") {
        return Err(format!("approval records: {text}"));
    }
    Ok(())
}
