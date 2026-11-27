//! Where each parameter map came from and what it is for (2026-11-26, Luis Jake Gabriel III: "Also
//! are all those maps up to date? We need to keep track of what they belong to and organize them
//! well. In the system visually and through the back ends").
//!
//! One file per MicroModel role, `library/parameter-maps/<role>/lineage.json`, beside the maps it
//! describes (so it travels with them into a backup or a library): for each map file, what it was
//! measured on (a base cartridge `id@version`, or the open weights by model id), the command that
//! made it, when, its purpose, and what was built from it. A tool that makes a map records it here
//! as it writes the map; maps made before this file existed are filled in from what can be proven
//! (`recorded: "backfill"`) and the rest say their origin is not recorded. Nothing here changes a map.
use nmjson::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// The lineage file of a role folder.
pub const FILE: &str = "lineage.json";

/// What a map is for.
pub const PURPOSES: [&str; 6] = ["serving", "base-build", "job-filter", "probe", "measurement", "experiment"];

/// How a map turned out (2026-11-26, Luis Jake Gabriel III: "Make sure we list when a map fails. I
/// don't want to use trash maps in our data ... Like if we train on mapping later I don't want
/// fucked up maps to be used"; "We at least need to keep dialed data from healthy data"):
/// `used` (serves or builds what serves now), `superseded` (built an earlier version that worked),
/// `failed` (did not pass its gate or caused a dip, with the evidence), `duplicate` (the same bytes
/// as a map on record), `control` (a deliberate baseline, such as a random selection), `unused` (a
/// variant or sweep nothing read). Empty: not stated yet.
pub const OUTCOMES: [&str; 6] = ["used", "superseded", "failed", "duplicate", "control", "unused"];

/// One map's record.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Record {
    /// What it was measured on: a base cartridge `id@version`, or `open:<model id>` for the open
    /// weights; empty when not recorded.
    pub made_on: String,
    /// The command that made it; empty when not recorded.
    pub made_by: String,
    pub made_ms: u64,
    /// One of `PURPOSES`; empty when not recorded.
    pub purpose: String,
    /// What was built from it (cartridges, other maps).
    pub used_by: Vec<String>,
    /// `tool` (written as the map was made), `backfill` (proven from the map or the cartridges
    /// after the fact), or `user` (stated in the interface or over MCP).
    pub recorded: String,
    pub note: String,
    /// One of `OUTCOMES`; empty when not stated.
    pub outcome: String,
    /// Why, in words: the gate, the dip, the measurement or the fact behind the outcome.
    pub outcome_why: String,
}

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn text(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default()
}

impl Record {
    /// As JSON.
    #[must_use]
    pub fn to_value(&self) -> Value {
        Value::Object(
            [
                ("made_on", s(&self.made_on)),
                ("made_by", s(&self.made_by)),
                ("made_ms", Value::Number(self.made_ms.to_string())),
                ("purpose", s(&self.purpose)),
                ("used_by", Value::Array(self.used_by.iter().map(|u| s(u)).collect())),
                ("recorded", s(&self.recorded)),
                ("note", s(&self.note)),
                ("outcome", s(&self.outcome)),
                ("outcome_why", s(&self.outcome_why)),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect(),
        )
    }
    /// From JSON (missing fields empty).
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        Self {
            made_on: text(v, "made_on"),
            made_by: text(v, "made_by"),
            made_ms: v.get("made_ms").and_then(Value::integer).unwrap_or(0),
            purpose: text(v, "purpose"),
            used_by: v.get("used_by").and_then(Value::array).map(|a| a.iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect()).unwrap_or_default(),
            recorded: text(v, "recorded"),
            note: text(v, "note"),
            outcome: text(v, "outcome"),
            outcome_why: text(v, "outcome_why"),
        }
    }

    /// Healthy data: a map that is in use or built an earlier working version. Only these are
    /// read by anything that learns from maps unless negatives are asked for by name.
    #[must_use]
    pub fn healthy(&self) -> bool {
        matches!(self.outcome.as_str(), "used" | "superseded")
    }

    /// Whether a map may be learned from: healthy maps always; with `negatives`, also the labelled
    /// failures and controls (as negative examples). Duplicates, unused variants and maps whose
    /// outcome is not stated never are.
    #[must_use]
    pub fn trainable(&self, negatives: bool) -> bool {
        self.healthy() || (negatives && matches!(self.outcome.as_str(), "failed" | "control"))
    }
}

fn path(role_dir: &Path) -> PathBuf {
    role_dir.join(FILE)
}

/// Every record of a role folder, by map file name (an absent or unreadable file: none).
#[must_use]
pub fn read(role_dir: &Path) -> BTreeMap<String, Record> {
    let Some(v) = std::fs::read(path(role_dir)).ok().and_then(|b| nmjson::parse(&b).ok()) else { return BTreeMap::new() };
    match v.get("maps") {
        Ok(Value::Object(m)) => m.iter().map(|(k, r)| (k.clone(), Record::from_value(r))).collect(),
        _ => BTreeMap::new(),
    }
}

static WRITE: Mutex<()> = Mutex::new(());

/// Changes the records of a role folder under one writer lock and writes them through a
/// temporary file. Returns what `f` returns.
/// # Errors
/// The folder or the file cannot be written.
pub fn change<T>(role_dir: &Path, f: impl FnOnce(&mut BTreeMap<String, Record>) -> T) -> Result<T, String> {
    let _g = WRITE.lock().map_err(|_| "lineage lock poisoned".to_owned())?;
    let mut all = read(role_dir);
    let out = f(&mut all);
    let v = Value::Object(
        [
            ("note".to_owned(), s("Where each parameter map in this folder came from and what it is for (nmapi::lineage). Written by the tools that make maps, filled in from proof for older maps, or stated in the interface; it never changes a map.")),
            ("maps".to_owned(), Value::Object(all.iter().map(|(k, r)| (k.clone(), r.to_value())).collect())),
        ]
        .into_iter()
        .collect(),
    );
    std::fs::create_dir_all(role_dir).map_err(|e| format!("{}: {e}", role_dir.display()))?;
    let p = path(role_dir);
    let tmp = p.with_extension("json.tmp");
    std::fs::write(&tmp, nmjson::pretty::pretty(&v)).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &p).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok(out)
}

/// Records a map as its tool writes it (`recorded: tool`), keeping what it already had for the
/// fields left empty here.
/// # Errors
/// As `change`.
pub fn record_made(role_dir: &Path, file: &str, made_on: &str, made_by: &str, purpose: &str) -> Result<(), String> {
    change(role_dir, |all| {
        let r = all.entry(file.to_owned()).or_default();
        if !made_on.is_empty() {
            r.made_on = made_on.to_owned();
        }
        if !made_by.is_empty() {
            r.made_by = made_by.to_owned();
        }
        if !purpose.is_empty() {
            r.purpose = purpose.to_owned();
        }
        r.made_ms = nmtype::now_ms();
        r.recorded = "tool".into();
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_round_trip_and_keep_what_they_had() {
        let dir = std::env::temp_dir().join(format!("nm-lineage-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(read(&dir).is_empty(), "no file: no records");
        change(&dir, |all| {
            all.insert("pmap-v1.json".into(), Record { made_on: "open:qwen3-0.6b".into(), purpose: "base-build".into(), used_by: vec!["reader-base@2".into()], recorded: "backfill".into(), ..Record::default() });
        })
        .unwrap();
        record_made(&dir, "pmap-v1.json", "", "nano-lab pmap", "").unwrap();
        let r = read(&dir).remove("pmap-v1.json").unwrap();
        assert_eq!((r.made_on.as_str(), r.made_by.as_str(), r.purpose.as_str(), r.recorded.as_str()), ("open:qwen3-0.6b", "nano-lab pmap", "base-build", "tool"));
        assert_eq!(r.used_by, vec!["reader-base@2".to_owned()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 2026-11-26: only healthy maps are learned from by default; labelled failures and controls
    /// only as negatives; duplicates, unused variants and unstated outcomes never; the outcome
    /// survives a write and a read.
    #[test]
    fn failed_maps_are_kept_apart_from_healthy_ones() {
        let rec = |o: &str| Record { outcome: o.into(), ..Record::default() };
        for (o, healthy, plain, neg) in [
            ("used", true, true, true),
            ("superseded", true, true, true),
            ("failed", false, false, true),
            ("control", false, false, true),
            ("duplicate", false, false, false),
            ("unused", false, false, false),
            ("", false, false, false),
        ] {
            let r = rec(o);
            assert_eq!((r.healthy(), r.trainable(false), r.trainable(true)), (healthy, plain, neg), "outcome {o:?}");
        }
        let r = Record { outcome: "failed".into(), outcome_why: "held-out gate: cosine 0.91 below 0.95".into(), ..Record::default() };
        assert_eq!(Record::from_value(&r.to_value()), r);
    }
}
