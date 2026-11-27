//! Unit registry (PLAN 4c `registry`; guide 10.1, 10.4): `units/registry.json` holds every
//! card in registration order; `units/<id>/` holds a unit's files, written once through a
//! temporary file, read back and compared, never edited in place (the `nrntrn` commit rule).
//! Rollback is unloading: the base never changes. Two open units on the same target are an
//! overlapping scope and the second is refused (PLAN 4c `stepspec`: two specs whose scopes
//! overlap cannot both be accepted in one version).
use crate::unit::UnitCard;
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::io::Write as _;
use std::path::{Path, PathBuf};

fn bad(kind: FaultKind, m: impl Into<String>) -> Fault {
    Fault::new("nmlearn.registry", kind, m)
}

/// Statuses that no longer hold a scope open.
const CLOSED: &[&str] = &["rejected", "rolled-back", "retired"];

/// The registry under `<home>/units/`.
#[derive(Debug)]
pub struct Units {
    pub dir: PathBuf,
    pub cards: Vec<UnitCard>,
}
impl Units {
    /// Open the registry (an absent file is an empty registry).
    #[must_use]
    pub fn open(home: &Path) -> Self {
        let dir = home.join("units");
        let cards = std::fs::read(dir.join("registry.json"))
            .ok()
            .and_then(|b| nmjson::parse(&b).ok())
            .and_then(|v| v.array().ok().map(|a| a.iter().filter_map(|c| UnitCard::from_value(c).ok()).collect()))
            .unwrap_or_default();
        Self { dir, cards }
    }
    fn save(&self) -> Result<(), Fault> {
        std::fs::create_dir_all(&self.dir).map_err(|e| bad(FaultKind::Resource, e.to_string()))?;
        let text = Value::Array(self.cards.iter().map(UnitCard::to_value).collect())
            .encode()
            .map_err(|e| bad(FaultKind::Resource, e.to_string()))?;
        std::fs::write(self.dir.join("registry.json"), text).map_err(|e| bad(FaultKind::Resource, e.to_string()))
    }
    /// The unit folder of `id`.
    #[must_use]
    pub fn unit_dir(&self, id: &str) -> PathBuf {
        self.dir.join(id)
    }
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&UnitCard> {
        self.cards.iter().find(|c| c.id == id)
    }
    fn get_mut(&mut self, id: &str) -> Result<&mut UnitCard, Fault> {
        self.cards.iter_mut().find(|c| c.id == id).ok_or_else(|| bad(FaultKind::Config, format!("no unit {id}")))
    }
    /// Register a card. Refused: a duplicate id, a lineage base that is not registered, or an
    /// open unit on the same target (overlapping scope).
    /// # Errors
    /// As stated, or the registry file cannot be written.
    pub fn register(&mut self, card: UnitCard) -> Result<(), Fault> {
        if self.cards.iter().any(|c| c.id == card.id) {
            return Err(bad(FaultKind::Config, format!("unit {} exists", card.id)));
        }
        if let Some(b) = card.lineage.iter().find(|b| !self.cards.iter().any(|c| c.id == **b)) {
            return Err(bad(FaultKind::Config, format!("lineage base {b} is not registered")));
        }
        if let Some(open) = self.cards.iter().find(|c| c.target == card.target && !CLOSED.contains(&c.status.as_str())) {
            return Err(bad(
                FaultKind::Policy,
                format!("unit {} overlaps the open scope of {} on {}; refused", card.id, open.id, card.target),
            ));
        }
        self.cards.push(card);
        self.save()
    }
    /// Remove an unfitted unit (no files) from the registry.
    /// # Errors
    /// Unknown, or a unit with files (those are retired, never removed).
    pub fn unregister(&mut self, id: &str) -> Result<(), Fault> {
        let c = self.get_mut(id)?;
        if !c.files.is_empty() {
            return Err(bad(FaultKind::Policy, format!("unit {id} has files; retire it instead")));
        }
        self.cards.retain(|c| c.id != id);
        self.save()
    }
    /// Write one unit file once: temporary file, fsync, read back, hash compared, rename.
    /// # Errors
    /// An existing file (never in place), or a read-back that differs.
    pub fn commit_file(&self, id: &str, name: &str, bytes: &[u8]) -> Result<String, Fault> {
        let dir = self.unit_dir(id);
        std::fs::create_dir_all(&dir).map_err(|e| bad(FaultKind::Resource, e.to_string()))?;
        let path = dir.join(name);
        if path.exists() {
            return Err(bad(
                FaultKind::Policy,
                format!("{} exists: units are never edited in place; register a new unit in the lineage", path.display()),
            ));
        }
        let tmp = dir.join(format!("{name}.tmp"));
        {
            let mut f = std::fs::File::create(&tmp).map_err(|e| bad(FaultKind::Resource, e.to_string()))?;
            f.write_all(bytes).map_err(|e| bad(FaultKind::Resource, e.to_string()))?;
            f.sync_all().map_err(|e| bad(FaultKind::Resource, e.to_string()))?;
        }
        let want = nmhash::sha256_bytes(bytes);
        let back = std::fs::read(&tmp).map_err(|e| bad(FaultKind::Resource, e.to_string()))?;
        if nmhash::sha256_bytes(&back) != want {
            let _ = std::fs::remove_file(&tmp);
            return Err(bad(FaultKind::Integrity, format!("{}: read-back hash differs", tmp.display())));
        }
        std::fs::rename(&tmp, &path).map_err(|e| bad(FaultKind::Resource, e.to_string()))?;
        Ok(want)
    }
    /// Record a unit's committed files and its new status (candidate until its Step Check).
    /// # Errors
    /// Unknown unit or a registry write failure.
    pub fn set_files(&mut self, id: &str, files: Vec<(String, String)>, status: &str) -> Result<(), Fault> {
        let c = self.get_mut(id)?;
        c.files = files;
        c.status = status.into();
        self.save()
    }
    /// Change a unit's status.
    /// # Errors
    /// Unknown unit or a registry write failure.
    pub fn set_status(&mut self, id: &str, status: &str) -> Result<(), Fault> {
        self.get_mut(id)?.status = status.into();
        self.save()
    }
    /// Mark a unit loaded: every file must match its hash first.
    /// # Errors
    /// Unknown unit or a file that does not match.
    pub fn mark_loaded(&mut self, id: &str) -> Result<(), Fault> {
        let dir = self.unit_dir(id);
        let c = self.get_mut(id)?;
        for (p, h) in &c.files {
            let bytes = std::fs::read(dir.join(p)).map_err(|e| bad(FaultKind::Resource, format!("{p}: {e}")))?;
            if nmhash::sha256_bytes(&bytes) != *h {
                return Err(bad(FaultKind::Integrity, format!("unit {id} file {p} does not match its hash")));
            }
        }
        c.loaded = true;
        self.save()
    }
    /// Mark a unit unloaded.
    /// # Errors
    /// Unknown unit or a registry write failure.
    pub fn mark_unloaded(&mut self, id: &str) -> Result<(), Fault> {
        self.get_mut(id)?.loaded = false;
        self.save()
    }
    /// Rollback: unload and mark rolled-back; lineage and files are kept.
    /// # Errors
    /// Unknown unit or a registry write failure.
    pub fn rollback(&mut self, id: &str) -> Result<(), Fault> {
        let c = self.get_mut(id)?;
        c.loaded = false;
        c.status = "rolled-back".into();
        self.save()
    }
    /// Units with an open scope (unfitted, candidate or fitted).
    #[must_use]
    pub fn candidates(&self) -> Vec<&UnitCard> {
        self.cards.iter().filter(|c| !CLOSED.contains(&c.status.as_str())).collect()
    }
    /// Ids of the loaded units.
    #[must_use]
    pub fn loaded(&self) -> Vec<String> {
        self.cards.iter().filter(|c| c.loaded).map(|c| c.id.clone()).collect()
    }
    /// The registry as JSON.
    #[must_use]
    pub fn to_value(&self) -> Value {
        Value::Array(self.cards.iter().map(UnitCard::to_value).collect())
    }
}

/// Self-check of `registry`: register, an overlap refused, a second target accepted, a file
/// committed once and refused twice, load after the hash check, rollback keeps the lineage.
pub(crate) fn check() -> Result<(), String> {
    let home = nmtype::check_dir("nmlearn-registry").map_err(|e| e.to_string())?;
    let mut u = Units::open(&home);
    let a = UnitCard::new("lwu-a", "pop", "aa", "reranker", "n#1", 8, 1);
    u.register(a.clone()).map_err(|e| e.to_string())?;
    if u.register(UnitCard::new("lwu-b", "pop", "aa", "reranker", "n#1", 8, 2)).is_ok() {
        return Err("nmlearn.registry: an overlapping scope was accepted".into());
    }
    u.register(UnitCard::new("lwu-c", "pop", "aa", "writer", "n#2", 8, 3)).map_err(|e| e.to_string())?;
    let h = u.commit_file("lwu-a", "up.q8", b"NMQ8").map_err(|e| e.to_string())?;
    if u.commit_file("lwu-a", "up.q8", b"NMQ8").is_ok() {
        return Err("nmlearn.registry: a unit file was written twice".into());
    }
    u.set_files("lwu-a", vec![("up.q8".into(), h)], "candidate").map_err(|e| e.to_string())?;
    u.mark_loaded("lwu-a").map_err(|e| e.to_string())?;
    let mut d = UnitCard::new("lwu-d", "pop", "aa", "reranker", "n#3", 8, 4);
    d.lineage = vec!["lwu-a".into()];
    u.register(d).map_err(|e| e.to_string())?;
    u.rollback("lwu-a").map_err(|e| e.to_string())?;
    let again = Units::open(&home);
    let a2 = again.get("lwu-a").ok_or("nmlearn.registry: lost on reopen")?;
    if a2.loaded
        || a2.status != "rolled-back"
        || again.loaded() != Vec::<String>::new()
        || again.get("lwu-d").is_none_or(|c| c.lineage != ["lwu-a"])
    {
        return Err("nmlearn.registry: rollback or lineage".into());
    }
    if u.register(UnitCard::new("lwu-e", "pop", "aa", "reranker", "n#1", 8, 5)).is_err() {
        return Err("nmlearn.registry: a closed scope still blocks".into());
    }
    Ok(())
}
