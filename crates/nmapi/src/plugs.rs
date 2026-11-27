//! Learned units: plugged in or out, one by one and by compartment (2026-11-26, Luis Jake Gabriel
//! III, his words: "We should be able to control and compartmentalized and plug in and unplug the
//! learned units").
//!
//! A learned unit is anything the Nano fitted from use or from the user's material and keeps beside
//! a frozen base: the project adapters on the adapter shelf (the embedder's lens, the judge's code
//! read-out), the learned pieces of `nm_unit` (the L4 calibrator, the L1 settings) and the
//! registered learning units (`units/registry.json`). None of them is ever merged into a base.
//!
//! A compartment is the scope a unit was fitted in and the only scope it loads in: `nano` for the
//! Nano's own folder, a project's key (`<folder>-<8 hex>`) for a project, `shared` for the
//! registered learning units (`units/`, loaded by `serve --unit`). A unit acts only when it is
//! plugged in and its compartment is switched on; unplugging either way takes it out at the next
//! call of the serving process (the state file is read again when it changes), and its outputs are
//! the base's bit for bit, since every unit is applied beside the base and never merged.
//!
//! State: `<home>/units/plugs.json`, `{"units": {"<compartment>/<id>": "off"}, "compartments":
//! {"<compartment>": "off"}}`. Absent means plugged in, so a unit that passed its own gate acts as
//! before this file existed. Every change is a `setting` row in the audit log (keys `learned:` and
//! `compartment:`), undone like any other setting.
use nmjson::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// The compartment of the registered learning units (home-wide, loaded on request).
pub const SHARED: &str = "shared";
/// The compartment of the Nano's own folder.
pub const NANO: &str = "nano";

/// The state file under the home.
#[must_use]
pub fn path(home: &Path) -> PathBuf {
    home.join("units").join("plugs.json")
}

/// Unplugged units (`<compartment>/<id>`) and switched-off compartments.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct State {
    pub units_off: std::collections::BTreeSet<String>,
    pub compartments_off: std::collections::BTreeSet<String>,
}
impl State {
    /// Whether the unit `key` (`<compartment>/<id>`) is plugged in.
    #[must_use]
    pub fn plugged(&self, key: &str) -> bool {
        !self.units_off.contains(key)
    }
    /// Whether the compartment is switched on.
    #[must_use]
    pub fn compartment_on(&self, compartment: &str) -> bool {
        !self.compartments_off.contains(compartment)
    }
    /// Whether the unit acts: plugged in, in a compartment that is on.
    #[must_use]
    pub fn acts(&self, compartment: &str, id: &str) -> bool {
        self.compartment_on(compartment) && self.plugged(&key(compartment, id))
    }
    fn to_value(&self) -> Value {
        let off = |set: &std::collections::BTreeSet<String>| {
            Value::Object(set.iter().map(|k| (k.clone(), Value::String("off".into()))).collect::<BTreeMap<_, _>>())
        };
        Value::Object(
            [
                ("note".to_owned(), Value::String("Learned units unplugged and compartments switched off, written by the interface and MCP. Every change is in memory/audit.jsonl.".into())),
                ("units".to_owned(), off(&self.units_off)),
                ("compartments".to_owned(), off(&self.compartments_off)),
            ]
            .into_iter()
            .collect(),
        )
    }
    fn from_value(v: &Value) -> Self {
        let off = |field: &str| {
            v.get(field)
                .ok()
                .and_then(|o| if let Value::Object(m) = o { Some(m) } else { None })
                .map(|m| m.iter().filter(|(_, x)| x.string().is_ok_and(|s| s == "off")).map(|(k, _)| k.clone()).collect())
                .unwrap_or_default()
        };
        Self { units_off: off("units"), compartments_off: off("compartments") }
    }
}

/// A unit's key in the state file.
#[must_use]
pub fn key(compartment: &str, id: &str) -> String {
    format!("{compartment}/{id}")
}

/// The file's state read now (absent or unreadable: everything plugged in).
#[must_use]
pub fn read_now(home: &Path) -> State {
    std::fs::read(path(home)).ok().and_then(|b| nmjson::parse(&b).ok()).map_or_else(State::default, |v| State::from_value(&v))
}

/// The last read: (file, modified time, length, state). Read again when the file changes.
type Seen = (PathBuf, Option<std::time::SystemTime>, u64, State);
static CACHE: Mutex<Option<Seen>> = Mutex::new(None);

/// The state, read again only when the file changed (one metadata look per call).
#[must_use]
pub fn read(home: &Path) -> State {
    let p = path(home);
    let meta = std::fs::metadata(&p).ok();
    let (mtime, len) = (meta.as_ref().and_then(|m| m.modified().ok()), meta.as_ref().map_or(0, std::fs::Metadata::len));
    if let Ok(g) = CACHE.lock()
        && let Some((cp, cm, cl, st)) = g.as_ref()
        && *cp == p
        && *cm == mtime
        && *cl == len
    {
        return st.clone();
    }
    let st = if meta.is_some() { read_now(home) } else { State::default() };
    if let Ok(mut g) = CACHE.lock() {
        *g = Some((p, mtime, len, st.clone()));
    }
    st
}

/// Whether the unit `id` of `compartment` acts now.
#[must_use]
pub fn acts(home: &Path, compartment: &str, id: &str) -> bool {
    read(home).acts(compartment, id)
}

static WRITE: Mutex<()> = Mutex::new(());

/// Changes the state under one writer lock and writes it through a temporary file. Returns the
/// state before and after.
/// # Errors
/// The folder or the file cannot be written.
pub fn change(home: &Path, f: impl FnOnce(&mut State)) -> Result<(State, State), String> {
    let _g = WRITE.lock().map_err(|_| "plugs lock poisoned".to_owned())?;
    let before = read_now(home);
    let mut after = before.clone();
    f(&mut after);
    if after != before {
        let p = path(home);
        if let Some(dir) = p.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let tmp = p.with_extension("json.tmp");
        std::fs::write(&tmp, nmjson::pretty::pretty(&after.to_value())).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &p).map_err(|e| format!("{}: {e}", p.display()))?;
        if let Ok(mut g) = CACHE.lock() {
            *g = None;
        }
    }
    Ok((before, after))
}

/// The compartment of a memory folder: `nano` for the home memory, the project key for a project
/// memory, else `session:<folder name>` (a memory folder given by hand).
#[must_use]
pub fn compartment_of_memory(home: &Path, memory_dir: &Path) -> String {
    if memory_dir == home.join("memory") {
        NANO.to_owned()
    } else if memory_dir.parent() == Some(nmtype::project::projects_dir(home).as_path()) {
        memory_dir.file_name().and_then(|n| n.to_str()).unwrap_or("project").to_owned()
    } else {
        format!("session:{}", memory_dir.file_name().and_then(|n| n.to_str()).unwrap_or("memory"))
    }
}

/// The compartment of a learned-piece store (`learn::learned_dir`'s folders): `units/learned` is
/// `nano`, `<project memory>/learned` the project, `<name>-learned` beside a hand-given memory
/// `session:<name>`.
#[must_use]
pub fn compartment_of_store(home: &Path, store: &Path) -> String {
    if store == nmlearn::learned::store(home) {
        return NANO.to_owned();
    }
    if let Some(memory) = store.parent()
        && memory.parent() == Some(nmtype::project::projects_dir(home).as_path())
        && store.file_name().is_some_and(|n| n == "learned")
    {
        return compartment_of_memory(home, memory);
    }
    let name = store.file_name().and_then(|n| n.to_str()).unwrap_or("session");
    format!("session:{}", name.strip_suffix("-learned").unwrap_or(name))
}

/// The compartment of a studied root (`nano` or its project key).
#[must_use]
pub fn compartment_of_root(home: &Path, root: &Path) -> String {
    let r = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    nmtype::project::label(home, &r)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Home(PathBuf);
    impl Drop for Home {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn units_act_until_unplugged_and_compartments_gate_them() {
        let home = Home(std::env::temp_dir().join(format!("nm-plugs-{}", std::process::id())));
        std::fs::create_dir_all(&home.0).unwrap();
        assert!(acts(&home.0, "shop-1234abcd", "embedder-lens-1234abcd5678"), "absent file: plugged in");
        change(&home.0, |s| {
            s.units_off.insert(key("shop-1234abcd", "embedder-lens-1234abcd5678"));
        })
        .unwrap();
        assert!(!acts(&home.0, "shop-1234abcd", "embedder-lens-1234abcd5678"));
        assert!(acts(&home.0, "shop-1234abcd", "judge-code-1234abcd5678"), "only that unit is out");
        assert!(acts(&home.0, "other-00000000", "embedder-lens-1234abcd5678"), "keys are per compartment");
        change(&home.0, |s| {
            s.units_off.clear();
            s.compartments_off.insert("shop-1234abcd".into());
        })
        .unwrap();
        assert!(!acts(&home.0, "shop-1234abcd", "judge-code-1234abcd5678"), "a compartment off takes all its units out");
        assert!(acts(&home.0, NANO, "l4-calibrator-001"));
        let (before, after) = change(&home.0, |s| s.compartments_off.clear()).unwrap();
        assert!(!before.compartment_on("shop-1234abcd") && after.compartment_on("shop-1234abcd"));
        assert_eq!(read_now(&home.0), State::default());
    }

    #[test]
    fn memory_folders_name_their_compartment() {
        let home = Path::new("/h");
        assert_eq!(compartment_of_memory(home, &home.join("memory")), NANO);
        assert_eq!(compartment_of_memory(home, &home.join("memory/projects/shop-1234abcd")), "shop-1234abcd");
        assert_eq!(compartment_of_memory(home, Path::new("/elsewhere/m1")), "session:m1");
        assert_eq!(compartment_of_store(home, &home.join("units/learned")), NANO);
        assert_eq!(compartment_of_store(home, &home.join("memory/projects/shop-1234abcd/learned")), "shop-1234abcd");
        assert_eq!(compartment_of_store(home, Path::new("/elsewhere/m1-learned")), "session:m1");
    }
}
