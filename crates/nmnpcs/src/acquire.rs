//! Acquire (guide 9.3 responsibilities 1, 2, 13: discovery, registration and identity,
//! validation; PLAN 4c, shape from `wgtadp/wgtreg`): weight cards from `weights/<id>/` in the
//! mini's folder, verified by SHA-256 against the card. A verified file leaves a stamp
//! (size, modification time, hash) under `scratch/verified/`, so the 1.5 GB hash is recomputed
//! only when the file changed.
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmnpcs.acquire", FaultKind::Config, m)
}

/// One parameter population as released (a checkpoint), with its identity.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightCard {
    pub id: String,
    pub family: String,
    pub tokenizer: String,
    /// Folder holding the files (inside the mini).
    pub dir: PathBuf,
    /// Main file name, relative to `dir`.
    pub main: String,
    pub sha_main: String,
    pub bytes: u64,
    pub revision: String,
    pub config: Value,
}
impl WeightCard {
    /// Read `weights/<id>/part.json` and `files/config.json` under `home`.
    /// # Errors
    /// Missing or malformed card or config.
    pub fn load(home: &Path, id: &str) -> Result<Self, Fault> {
        let base = nmtype::models_dir(home).join(id);
        let card = nmjson::parse(&std::fs::read(base.join("part.json")).map_err(|e| bad(format!("{id}: part.json: {e}")))?)
            .map_err(|e| bad(e.to_string()))?;
        let w = card.get("weights").map_err(|e| bad(e.to_string()))?;
        let s = |v: &Value, k: &str| v.get(k).and_then(Value::string).map(str::to_owned).map_err(|e| bad(format!("{id}: {k}: {e}")));
        let main_rel = s(w, "main_file")?;
        let main = main_rel.trim_start_matches("files/").to_owned();
        let dir = base.join("files");
        let config = nmjson::parse(&std::fs::read(dir.join("config.json")).map_err(|e| bad(format!("{id}: config.json: {e}")))?)
            .map_err(|e| bad(e.to_string()))?;
        let family = config.get("model_type").and_then(Value::string).unwrap_or("unknown").to_owned();
        let vocab = config.get("vocab_size").and_then(Value::integer).unwrap_or(0);
        let bytes = match w.get("bytes") {
            Ok(Value::Number(n)) => n.parse().unwrap_or(0),
            _ => 0,
        };
        Ok(Self {
            id: id.to_owned(),
            family: family.clone(),
            tokenizer: format!("{family}-{vocab}"),
            dir,
            main,
            sha_main: s(w, "sha256")?,
            bytes,
            revision: s(w, "revision").unwrap_or_default(),
            config,
        })
    }
    /// Path of the main parameter file.
    #[must_use]
    pub fn main_path(&self) -> PathBuf {
        self.dir.join(&self.main)
    }
    /// An integer from config.json.
    /// # Errors
    /// The key is missing.
    pub fn int(&self, key: &str) -> Result<usize, Fault> {
        self.config
            .get(key)
            .and_then(Value::integer)
            .ok()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| bad(format!("{}: config {key}", self.id)))
    }
    /// A number from config.json.
    /// # Errors
    /// The key is missing.
    pub fn float(&self, key: &str) -> Result<f32, Fault> {
        match self.config.get(key) {
            Ok(Value::Number(n)) => n.parse().map_err(|_| bad(format!("{}: config {key}", self.id))),
            _ => Err(bad(format!("{}: config {key}", self.id))),
        }
    }
    /// The released parameter files of the population (every `.safetensors` in its folder).
    #[must_use]
    pub fn shards(&self) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(&self.dir)
            .map(|d| d.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.ends_with(".safetensors")).collect())
            .unwrap_or_default();
        v.sort();
        v
    }
    /// Recompute the SHA-256 of every released parameter file (or reuse a stamp for the same
    /// size and time): the main file against the card, any other file of a model released in
    /// several files against its row in `INTEGRITY.md` (`models/<id>/files/<name>`, a path in
    /// the models folder; a row written as `weights/<id>/files/<name>` is read too).
    /// # Errors
    /// A file is missing, has no recorded hash, or its hash differs.
    pub fn verify(&self, home: &Path) -> Result<(), Fault> {
        let rows = std::fs::read_to_string(home.join("INTEGRITY.md")).unwrap_or_default();
        for name in self.shards() {
            let want = if name == self.main {
                self.sha_main.clone()
            } else {
                let keys = [format!("`{MODELS_PREFIX}{}/files/{name}`", self.id), format!("`weights/{}/files/{name}`", self.id)];
                rows.lines()
                    .find(|l| keys.iter().any(|k| l.contains(k.as_str())))
                    .and_then(|l| l.split('`').nth(3))
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        Fault::new("nmnpcs.acquire", FaultKind::Integrity, format!("{}: no recorded hash for {name}", self.id))
                    })?
            };
            self.verify_file(home, &name, &want)?;
        }
        Ok(())
    }
    fn verify_file(&self, home: &Path, name: &str, sha: &str) -> Result<(), Fault> {
        let path = self.dir.join(name);
        let meta = std::fs::metadata(&path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
        let mtime = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
        let stamp_dir = home.join("scratch/verified");
        let stamp = stamp_dir.join(if name == self.main { format!("{}.json", self.id) } else { format!("{}--{name}.json", self.id) });
        let want = format!("{{\"bytes\":{},\"mtime\":{mtime},\"sha256\":\"{sha}\"}}", meta.len());
        if std::fs::read_to_string(&stamp).is_ok_and(|s| s == want) {
            return Ok(());
        }
        let file = std::fs::File::open(&path).map_err(|e| bad(e.to_string()))?;
        let got = nmhash::sha256(std::io::BufReader::with_capacity(1 << 20, file)).map_err(|e| bad(e.to_string()))?;
        if got != sha {
            return Err(Fault::new(
                "nmnpcs.acquire",
                FaultKind::Integrity,
                format!("{}: {name}: SHA-256 {got} differs from the record {sha}", self.id),
            ));
        }
        let _ = std::fs::create_dir_all(&stamp_dir);
        let _ = std::fs::write(&stamp, want);
        Ok(())
    }
}

/// Every weight card in `weights/` (discovery).
/// # Errors
/// An unreadable folder; cards that do not parse are skipped and named.
pub fn scan(home: &Path) -> Result<(Vec<WeightCard>, Vec<String>), Fault> {
    let mut ok = Vec::new();
    let mut skipped = Vec::new();
    // Only folders are populations; files such as the folder's README.md are skipped.
    let mut ids: Vec<String> = std::fs::read_dir(nmtype::models_dir(home))
        .map_err(|e| bad(e.to_string()))?
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    ids.sort();
    for id in ids {
        match WeightCard::load(home, &id) {
            Ok(c) => ok.push(c),
            Err(e) => skipped.push(format!("{id}: {e}")),
        }
    }
    Ok((ok, skipped))
}

/// Population ids actually used by the mini (law 9: one base population, nothing fine-tuned).
pub const IN_USE: &[&str] = &["qwen3-0.6b"];

/// The prefix of an `INTEGRITY.md` row whose path is relative to the models folder.
pub const MODELS_PREFIX: &str = "models/";

/// The card as JSON, for `nm_weights`.
#[must_use]
pub fn to_value(c: &WeightCard) -> Value {
    let mut m = BTreeMap::new();
    for (k, v) in [
        ("id", &c.id),
        ("family", &c.family),
        ("tokenizer", &c.tokenizer),
        ("sha256", &c.sha_main),
        ("revision", &c.revision),
        ("main_file", &c.main),
    ] {
        m.insert(k.to_owned(), Value::String(v.clone()));
    }
    m.insert("bytes".into(), Value::Number(c.bytes.to_string()));
    m.insert("in_use".into(), Value::Bool(IN_USE.contains(&c.id.as_str())));
    Value::Object(m)
}

/// Self-check of `acquire`: the in-use card loads from the mini's folder with its family,
/// tokenizer label, dimensions and hash (the full-file hash is a gate run, not this check).
pub(crate) fn check() -> Result<(), String> {
    let home = nmtype::nano_root();
    let c = WeightCard::load(&home, "qwen3-0.6b").map_err(|e| e.to_string())?;
    if c.family != "qwen3"
        || c.tokenizer != "qwen3-151936"
        || c.sha_main.len() != 64
        || c.int("num_hidden_layers").ok() != Some(28)
        || !c.main_path().is_file()
    {
        return Err(format!("nmnpcs.acquire: card {c:?}"));
    }
    Ok(())
}
