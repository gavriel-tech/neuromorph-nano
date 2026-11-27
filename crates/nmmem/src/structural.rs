//! Structural memory (guide 7, 17.2; PLAN 4c): one summary of the studied repository per
//! substrate version (`structural/v<n>.json`), the latest loaded at session start, older ones
//! marked stale but kept. Also the objectives record (`objectives.json`), which the session
//! reconstructs until `nmbrain::objectives` owns it at S5.
use crate::store::{self, obj, s, u};
use nmjson::Value;
use nmtype::Fault;
use std::path::{Path, PathBuf};

fn dir(home: &Path) -> PathBuf {
    home.join("structural")
}

/// Save a summary for a substrate version (a new file per version; earlier ones stay).
/// # Errors
/// The file could not be written.
pub fn save(home: &Path, version: u64, digest: &str, summary: Value) -> Result<PathBuf, Fault> {
    let path = dir(home).join(format!("v{version}-{}.json", digest.get(..12).unwrap_or(digest)));
    if path.exists() {
        return Ok(path);
    }
    store::put(&path, &obj(vec![("version", u(version)), ("digest", s(digest)), ("at", u(nmtype::now_ms())), ("summary", summary)]))?;
    Ok(path)
}
/// The most recent summary, if any.
#[must_use]
pub fn load_latest(home: &Path) -> Option<Value> {
    let mut files: Vec<PathBuf> =
        std::fs::read_dir(dir(home)).ok()?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
    files.sort_by_key(|p| store::get(p).map_or(0, |v| store::num(&v, "at")));
    files.last().and_then(|p| store::get(p))
}
/// Is a stored summary stale against the current substrate digest?
#[must_use]
pub fn stale(summary: &Value, digest: &str) -> bool {
    store::text(summary, "digest") != digest
}
/// Save the objectives list (id, text, status).
/// # Errors
/// The file could not be written.
pub fn save_objectives(home: &Path, items: &[(u64, String, String)]) -> Result<(), Fault> {
    let list = items.iter().map(|(id, t, st)| obj(vec![("id", u(*id)), ("objective", s(t)), ("status", s(st))])).collect();
    store::put(&home.join("objectives.json"), &obj(vec![("objectives", Value::Array(list))]))
}
/// Load the objectives list.
#[must_use]
pub fn load_objectives(home: &Path) -> Vec<(u64, String, String)> {
    store::get(&home.join("objectives.json"))
        .and_then(|v| {
            v.get("objectives")
                .and_then(Value::array)
                .ok()
                .map(|a| a.iter().map(|o| (store::num(o, "id"), store::text(o, "objective"), store::text(o, "status"))).collect())
        })
        .unwrap_or_default()
}

/// Self-check of the `structural` sub-component: a saved summary loads back as the latest and
/// is stale against another digest; objectives round-trip.
pub(crate) fn check() -> Result<(), String> {
    let home = crate::scratch_dir("structural")?;
    save(&home, 3, "abcdef0123456789", obj(vec![("crates", u(69))])).map_err(|e| e.to_string())?;
    let latest = load_latest(&home).ok_or("nmmem.structural: nothing loaded")?;
    if store::num(&latest, "version") != 3 || !stale(&latest, "ffff") || stale(&latest, "abcdef0123456789") {
        return Err("nmmem.structural: latest or stale".into());
    }
    let goals = vec![(1, "map the context crates".to_owned(), "open".to_owned())];
    save_objectives(&home, &goals).map_err(|e| e.to_string())?;
    if load_objectives(&home) != goals {
        return Err("nmmem.structural: objectives round trip".into());
    }
    Ok(())
}
