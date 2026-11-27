//! Muscle-memory store: pack files `<fingerprint>.nmmusc` plus `index.json` in one folder,
//! by default `muscle/` under the mini root. It is its own store: a folder path with a
//! `memory`, `weights` or `units` component is refused before anything is created, and a
//! pack whose content is not behavioural is refused.
use crate::pack::{CLASS, KIND, Pack};
use crate::withdraw;
use crate::{Fault, Result};
use nmjson::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Folder names muscle memory never writes under.
pub const FORBIDDEN: [&str; 3] = ["memory", "weights", "units"];

/// One row of `index.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct IndexRow {
    pub fingerprint: String,
    pub file: String,
    pub description: String,
    pub base_hash: String,
    pub date: String,
    pub withdrawn: bool,
    pub promotion_gain: f64,
    pub running_mean: f64,
    pub running_n: u64,
}

/// Outcome of recording one observed gain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GainUpdate {
    pub running_mean: f64,
    pub running_n: u64,
    pub margin: Option<f64>,
    /// True when this update withdrew the pack.
    pub withdrawn_now: bool,
    pub withdrawn: bool,
}

/// The store.
#[derive(Clone, Debug)]
pub struct MuscleStore {
    dir: PathBuf,
}

fn io(e: &std::io::Error) -> Fault {
    Fault::new("nmmusc-io", e.to_string())
}

impl MuscleStore {
    /// The live store, `designs/muscle.json` `store.dir` under the mini root (fallback
    /// `muscle`).
    /// # Errors
    /// As `open`.
    pub fn live() -> Result<Self> {
        Self::open(&nmtype::nano_root().join(crate::Policy::load().store_dir))
    }

    /// Open (and create) a store folder.
    /// # Errors
    /// `nmmusc-forbidden-path` when any path component is `memory`, `weights` or `units`;
    /// `nmmusc-io` when the folder cannot be created.
    pub fn open(dir: &Path) -> Result<Self> {
        for c in dir.components() {
            let name = c.as_os_str().to_string_lossy().to_ascii_lowercase();
            if FORBIDDEN.contains(&name.as_str()) {
                return Err(Fault::new(
                    "nmmusc-forbidden-path",
                    format!("muscle memory is its own store; refused under {name}/: {}", dir.display()),
                ));
            }
        }
        std::fs::create_dir_all(dir).map_err(|e| io(&e))?;
        Ok(Self { dir: dir.to_path_buf() })
    }

    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn file_name(fp: &str) -> Result<String> {
        if fp.len() != 64 || !fp.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(Fault::new("nmmusc-bad-fingerprint", format!("not a SHA-256 hex digest: {fp:?}")));
        }
        Ok(format!("{fp}.nmmusc"))
    }

    /// Write a pack and update the index. The same fingerprint overwrites its own file only.
    /// # Errors
    /// `nmmusc-sensory-refused` when the pack is not behavioural muscle memory; I/O faults.
    pub fn write(&self, pack: &Pack) -> Result<PathBuf> {
        let bytes = pack.encode()?;
        // Re-read what will be written: refuses anything whose header is not behavioural.
        let check = Pack::decode(&bytes)?;
        let name = Self::file_name(&check.header.fingerprint)?;
        let path = self.dir.join(&name);
        std::fs::write(&path, &bytes).map_err(|e| io(&e))?;
        let mut rows = self.index()?;
        let row = row_of(pack, &name);
        if let Some(r) = rows.iter_mut().find(|r| r.fingerprint == row.fingerprint) {
            *r = row;
        } else {
            rows.push(row);
        }
        self.write_index(&rows)?;
        Ok(path)
    }

    /// Read the index (empty when absent).
    /// # Errors
    /// An unreadable or malformed index.
    pub fn index(&self) -> Result<Vec<IndexRow>> {
        let p = self.dir.join("index.json");
        if !p.exists() {
            return Ok(Vec::new());
        }
        let bytes = std::fs::read(&p).map_err(|e| io(&e))?;
        let v = nmjson::parse(&bytes).map_err(|e| Fault::new("nmmusc-bad-index", e.to_string()))?;
        let bad = |k: &str| Fault::new("nmmusc-bad-index", format!("{k} missing"));
        let mut out = Vec::new();
        for r in v.get("packs").and_then(Value::array).map_err(|_| bad("packs"))? {
            let t = |k: &str| r.get(k).and_then(Value::string).map(str::to_owned).map_err(|_| bad(k));
            out.push(IndexRow {
                fingerprint: t("fingerprint")?,
                file: t("file")?,
                description: t("description")?,
                base_hash: t("base_hash")?,
                date: t("date")?,
                withdrawn: matches!(r.get("withdrawn"), Ok(Value::Bool(true))),
                promotion_gain: crate::pack::num(r, "promotion_gain")?,
                running_mean: crate::pack::num(r, "running_mean")?,
                running_n: r.get("running_n").and_then(Value::integer).map_err(|_| bad("running_n"))?,
            });
        }
        Ok(out)
    }

    fn write_index(&self, rows: &[IndexRow]) -> Result<()> {
        let s = |t: &str| Value::String(t.to_owned());
        let n = |x: f64| Value::Number(if x.is_finite() { format!("{x}") } else { "0".to_owned() });
        let packs = rows
            .iter()
            .map(|r| {
                Value::Object(BTreeMap::from([
                    ("fingerprint".to_owned(), s(&r.fingerprint)),
                    ("file".to_owned(), s(&r.file)),
                    ("description".to_owned(), s(&r.description)),
                    ("base_hash".to_owned(), s(&r.base_hash)),
                    ("date".to_owned(), s(&r.date)),
                    ("withdrawn".to_owned(), Value::Bool(r.withdrawn)),
                    ("promotion_gain".to_owned(), n(r.promotion_gain)),
                    ("running_mean".to_owned(), n(r.running_mean)),
                    ("running_n".to_owned(), Value::Number(r.running_n.to_string())),
                ]))
            })
            .collect();
        let v = Value::Object(BTreeMap::from([
            ("kind".to_owned(), s(KIND)),
            ("data_class".to_owned(), s(CLASS)),
            ("packs".to_owned(), Value::Array(packs)),
        ]));
        let text = nmjson::pretty::pretty(&v);
        std::fs::write(self.dir.join("index.json"), text).map_err(|e| io(&e))
    }

    /// Load a pack by fingerprint and verify it against the base-weights hash in use.
    /// # Errors
    /// `nmmusc-not-found`, `nmmusc-bad-pack`, `nmmusc-base-mismatch`.
    pub fn load(&self, fp: &str, base_hash: &str) -> Result<Pack> {
        let path = self.dir.join(Self::file_name(fp)?);
        let bytes = std::fs::read(&path).map_err(|_| Fault::new("nmmusc-not-found", format!("no pack {fp}")))?;
        let p = Pack::decode(&bytes)?;
        if p.header.fingerprint != fp {
            return Err(Fault::new("nmmusc-bad-pack", "file name and fingerprint differ"));
        }
        if p.header.base_hash != base_hash {
            return Err(Fault::new("nmmusc-base-mismatch", format!("pack valid for base {}, not {base_hash}", p.header.base_hash)));
        }
        Ok(p)
    }

    /// Record one observed gain for a pack; withdraws it when the derived rule says so.
    /// # Errors
    /// As `load` and `write`.
    pub fn record_gain(&self, fp: &str, base_hash: &str, observed: f64, z: f64) -> Result<GainUpdate> {
        let mut p = self.load(fp, base_hash)?;
        p.header.running.push(observed);
        let promo = p.header.promotion;
        p.header.recent.push_against(&promo, observed);
        let was = p.header.withdrawn;
        let m = withdraw::margin(&p.header.promotion, &p.header.running, z);
        // 2026-11-18 (T-042): the change rule (Page's CUSUM against an always-valid threshold)
        // replaces the recent rule, whose repeated looks withdrew 16 of 60 packs that still gained.
        if withdraw::should_withdraw_change(&promo, &p.header.running, &p.header.recent, z) {
            p.header.withdrawn = true;
        }
        self.write(&p)?;
        Ok(GainUpdate {
            running_mean: p.header.running.mean,
            running_n: p.header.running.n,
            margin: m,
            withdrawn_now: !was && p.header.withdrawn,
            withdrawn: p.header.withdrawn,
        })
    }
}

fn row_of(p: &Pack, file: &str) -> IndexRow {
    let h = &p.header;
    IndexRow {
        fingerprint: h.fingerprint.clone(),
        file: file.to_owned(),
        description: h.description.clone(),
        base_hash: h.base_hash.clone(),
        date: h.date.clone(),
        withdrawn: h.withdrawn,
        promotion_gain: h.promotion.diff,
        running_mean: h.running.mean,
        running_n: h.running.n,
    }
}

/// Section check: forbidden folders refused before creation; write, reload and verify
/// round-trip exactly; wrong base hash refused; a sensory-tagged pack file refused.
/// # Errors
/// One fault line.
pub fn check() -> std::result::Result<(), String> {
    let root = nmtype::nano_root();
    for f in FORBIDDEN {
        let probe = root.join(f).join("nmmusc-probe");
        match MuscleStore::open(&probe) {
            Err(e) if e.code == "nmmusc-forbidden-path" => {}
            other => return Err(format!("{f}/ not refused: {other:?}")),
        }
        if probe.exists() {
            return Err(format!("refused path was created: {}", probe.display()));
        }
    }
    let dir = crate::scratch_dir("store").map_err(|e| e.to_string())?;
    let st = MuscleStore::open(&dir).map_err(|e| e.to_string())?;
    let p = crate::pack::sample("base-a").map_err(|e| e.to_string())?;
    st.write(&p).map_err(|e| e.to_string())?;
    let q = st.load(&p.header.fingerprint, "base-a").map_err(|e| e.to_string())?;
    if q != p {
        return Err("reloaded pack differs".to_owned());
    }
    match st.load(&p.header.fingerprint, "base-b") {
        Err(e) if e.code == "nmmusc-base-mismatch" => {}
        other => return Err(format!("wrong base accepted: {other:?}")),
    }
    let idx = st.index().map_err(|e| e.to_string())?;
    if idx.len() != 1 || idx.first().map(|r| r.fingerprint.as_str()) != Some(p.header.fingerprint.as_str()) {
        return Err(format!("index wrong: {idx:?}"));
    }
    // A file claiming the sensory class is refused on load.
    let mut bytes = p.encode().map_err(|e| e.to_string())?;
    let from = b"\"data_class\":\"behavioural\"";
    let to = b"\"data_class\":\"sensory\"    "; // same length, JSON whitespace pad
    let at = bytes.windows(from.len()).position(|w| w == from).ok_or("data_class not found")?;
    if let Some(w) = bytes.get_mut(at..at + to.len()) {
        w.copy_from_slice(to);
    }
    let path = dir.join(MuscleStore::file_name(&p.header.fingerprint).map_err(|e| e.to_string())?);
    std::fs::write(&path, &bytes).map_err(|e| e.to_string())?;
    match st.load(&p.header.fingerprint, "base-a") {
        Err(e) if e.code == "nmmusc-sensory-refused" => Ok(()),
        other => Err(format!("sensory-tagged pack accepted: {other:?}")),
    }
}
