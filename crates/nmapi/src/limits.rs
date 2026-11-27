//! The user's hardware limits as the interface and the MCP tools see them (Roadmap Phase 4.5
//! (2026-11-25), item 3): the saved values, the values in effect in this process, this
//! machine's recommended values with their reasons, setting one limit, and applying the
//! recommended values. The engine's side (`nmengn::limits`) reads the same `settings.json` live,
//! so a limit set here applies with no restart (processor only, a card limit of 0, at the next
//! start). Writes go through the program's controls (`set` there writes `settings.json` and an
//! audit row per change), lent here as a function, as the interface's routes are to `parity`.
use nmengn::capacity::{self, Recommendation};
use nmjson::Value;
use std::collections::BTreeMap;
use std::path::Path;

/// Sets one control and logs it: (key, value) to the audit id, `None` when it was already set.
pub type Setter<'a> = &'a mut dyn FnMut(&str, &str) -> Result<Option<String>, String>;

fn s(t: &str) -> Value {
    Value::String(t.to_owned())
}
fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect::<BTreeMap<_, _>>())
}

/// How the MicroModels sit on the card now, in words: all four seated (the writer kept beside
/// the other three), or the writer loaded per idle session (evict and restore), or none.
#[must_use]
pub fn residency_mode() -> &'static str {
    if nmengn::residency::get().card().is_none() {
        "processor"
    } else if crate::writer_idle::resident_writer_view(|_| ()).is_some() {
        "four seated"
    } else if crate::neural::bound_models().is_empty() {
        "none seated"
    } else {
        "evict and restore (the writer loads per idle session)"
    }
}

/// The limits in effect in this process, with the residency mode: the `limits` field of the
/// live capacity block.
#[must_use]
pub fn effective_value() -> Value {
    let mut v = nmengn::limits::to_value(&nmengn::limits::effective());
    if let Value::Object(o) = &mut v {
        o.insert("mode".to_owned(), s(residency_mode()));
    }
    v
}

/// The live capacity block (`nmengn::capacity::to_value`) with the limits in effect under
/// `limits` (`nmapi::live::snapshot`, `nm_live`).
#[must_use]
pub fn capacity_value() -> Value {
    let mut v = capacity::to_value(&capacity::profile());
    if let Value::Object(o) = &mut v {
        o.insert("limits".to_owned(), effective_value());
    }
    v
}

/// This machine's recommendation (the profile now and the free space on the Nano's drive).
#[must_use]
pub fn recommendation(home: &Path) -> Recommendation {
    capacity::recommend(&capacity::profile(), crate::storage::disk_space(home).map(|(free, _)| free))
}

/// The saved values under `home` (`auto` or a number for each key).
#[must_use]
pub fn saved(home: &Path) -> Value {
    let l = nmengn::limits::read_at(home);
    let a = nmengn::limits::read_areas_at(home);
    // 2026-11-27: the memory areas' sizes beside the three limits.
    Value::Object(
        nmengn::limits::KEYS
            .iter()
            .map(|k| ((*k).to_owned(), s(&l.setting(k))))
            .chain(nmengn::limits::AREA_KEYS.iter().map(|k| ((*k).to_owned(), s(&a.setting(k)))))
            .collect(),
    )
}

/// `GET /api/limits` and `nm_settings` `limits`: the saved values, the values in effect, the
/// recommendation and the profile.
#[must_use]
pub fn view(home: &Path) -> Value {
    obj(vec![
        ("current", saved(home)),
        ("effective", effective_value()),
        // 2026-11-27: each memory area's setting, budget and use now.
        ("memory_areas", crate::memory_areas::current()),
        ("recommendation", capacity::recommendation_value(&recommendation(home))),
        ("profile", capacity::to_value(&capacity::profile())),
        (
            "applies",
            s("live, with no restart (batches tile within the limits at the next call); a card limit of 0 (processor only) applies at the next start"),
        ),
    ])
}

/// Sets one limit (`vram_mb`, `ram_mb`, `cpu_threads`, or the full `limits.` key) to `auto` or a
/// number, through `set`.
/// # Errors
/// Not a limit key, a value out of bounds, or the write failing.
pub fn set_one(home: &Path, key: &str, value: &str, set: Setter<'_>) -> Result<Value, String> {
    let full = nmengn::limits::full_key(key).ok_or_else(|| format!("{key}: give vram_mb, ram_mb, cpu_threads, or a memory area (content_mb, caches_mb, weights_mb)"))?;
    let normalized = nmengn::limits::normalize(full, value).ok_or_else(|| format!("{full}: not a limit"))??;
    let audit = set(full, &normalized)?;
    let note = if full == nmengn::limits::VRAM_KEY && normalized == "0" {
        "processor only applies at the next start; the MicroModels seated now stay seated until then"
    } else {
        "applies live, with no restart"
    };
    Ok(obj(vec![
        ("audit_id", audit.map_or(Value::Null, |i| s(&i))),
        ("key", s(full)),
        ("value", s(&normalized)),
        ("note", s(note)),
        ("current", saved(home)),
    ]))
}

/// One applied value: the key, the value, the audit id (`None`: it was already set), the reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Applied {
    pub key: &'static str,
    pub value: String,
    pub audit: Option<String>,
    pub reason: String,
}

/// Writes the three recommended values through `set` (one audit row per changed value).
/// # Errors
/// The first write that fails.
pub fn apply(r: &Recommendation, set: Setter<'_>) -> Result<Vec<Applied>, String> {
    [&r.vram, &r.ram, &r.cpu]
        .into_iter()
        .map(|a| Ok(Applied { key: a.key, value: a.value.clone(), audit: set(a.key, &a.value)?, reason: a.reason.clone() }))
        .collect()
}

/// `POST /api/limits/apply-recommended` and `nm_settings` `apply_recommended`: this machine's
/// recommended values written through `set`, only with `confirm`.
/// # Errors
/// No `confirm`, or a write failing.
pub fn apply_recommended(home: &Path, confirm: bool, set: Setter<'_>) -> Result<Value, String> {
    if !confirm {
        return Err("apply-recommended changes three settings; send confirm: true".to_owned());
    }
    let r = recommendation(home);
    let applied = apply(&r, set)?;
    Ok(obj(vec![
        ("tier", s(r.tier.key())),
        (
            "applied",
            Value::Array(
                applied
                    .iter()
                    .map(|a| obj(vec![("key", s(a.key)), ("value", s(&a.value)), ("audit_id", a.audit.as_deref().map_or(Value::Null, s)), ("reason", s(&a.reason))]))
                    .collect(),
            ),
        ),
        ("notes", Value::Array(r.notes.iter().map(|n| s(n)).collect())),
        ("current", saved(home)),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nmengn::capacity::{Card, Profile};

    fn dir() -> std::path::PathBuf {
        // A counter beside the process and the millisecond: tests run in parallel in one process,
        // and two folders made in the same millisecond were one folder (they overwrote each other's
        // settings and audit log, and one removed the other's folder).
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let k = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let d = nmtype::nano_root().join("scratch").join("unit-tests").join(format!("nmapi-limits-{}-{}-{k}", std::process::id(), nmtype::now_ms()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A stand-in for the program's controls (`nmcore` `controls::set_by`, which the binary
    /// lends): merges the value into `settings.json` and appends a `setting` audit row on change.
    fn setter(home: &Path) -> impl FnMut(&str, &str) -> Result<Option<String>, String> + '_ {
        move |key: &str, value: &str| {
            let p = home.join("settings.json");
            let mut m: BTreeMap<String, Value> = match std::fs::read(&p).ok().and_then(|b| nmjson::parse(&b).ok()) {
                Some(Value::Object(m)) => m,
                _ => BTreeMap::new(),
            };
            let from = m.get(key).and_then(|v| v.string().ok()).unwrap_or("auto").to_owned();
            if from == value {
                return Ok(None);
            }
            m.insert(key.to_owned(), s(value));
            std::fs::write(&p, Value::Object(m).encode().map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            crate::audit::audit(home, "setting", vec![("key", s(key)), ("from", s(&from)), ("to", s(value)), ("by", s("test"))]).map(Some)
        }
    }

    #[test]
    fn limits_apply_recommended_writes_the_controls() {
        let home = dir();
        let cpu_only = Profile { card: None, host_mb: (4_096, 8_192), threads: 8, hw_threads: 8 };
        let r = capacity::recommend(&cpu_only, None);
        let mut set = setter(&home);
        let applied = apply(&r, &mut set).unwrap();
        assert_eq!(applied.iter().map(|a| (a.key, a.value.as_str())).collect::<Vec<_>>(), vec![
            (nmengn::limits::VRAM_KEY, "0"),
            (nmengn::limits::RAM_KEY, "6144"),
            (nmengn::limits::CPU_KEY, "7")
        ]);
        assert!(applied.iter().all(|a| a.audit.is_some() && !a.reason.is_empty()));
        let l = nmengn::limits::read_at(&home);
        assert_eq!(l, nmengn::limits::Limits { vram_mb: Some(0), ram_mb: Some(6_144), cpu_threads: Some(7) });
        let log = std::fs::read_to_string(home.join("memory").join("audit.jsonl")).unwrap();
        assert_eq!(log.lines().count(), 3);
        assert!(log.contains("\"key\":\"limits.ram_mb\"") && log.contains("\"to\":\"6144\""));
        // Applying again changes nothing and logs nothing.
        assert!(apply(&r, &mut set).unwrap().iter().all(|a| a.audit.is_none()));
        // The 4 GB card: all auto, so all four MicroModels stay seated.
        let small = Profile {
            card: Some(Card { name: "GTX 1650".to_owned(), total_mb: 4_096, free_mb: 3_500, compute: (7, 5), reserve_mb: (300, true) }),
            host_mb: (16_000, 32_768),
            threads: 12,
            hw_threads: 12,
        };
        apply(&capacity::recommend(&small, None), &mut set).unwrap();
        assert_eq!(nmengn::limits::read_at(&home), nmengn::limits::Limits::AUTO);
        drop(set);
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn limits_set_one_and_confirm() {
        let home = dir();
        let mut set = setter(&home);
        let v = set_one(&home, "vram_mb", "0", &mut set).unwrap();
        assert!(v.get("note").and_then(Value::string).is_ok_and(|n| n.contains("next start")));
        let v = set_one(&home, "limits.cpu_threads", " AUTO ", &mut set).unwrap();
        assert_eq!(v.get("value").and_then(Value::string).ok(), Some("auto"));
        assert!(set_one(&home, "gpu_mb_max", "4096", &mut set).is_err());
        assert!(set_one(&home, "ram_mb", "12", &mut set).is_err());
        assert!(apply_recommended(&home, false, &mut set).unwrap_err().contains("confirm"));
        drop(set);
        let _ = std::fs::remove_dir_all(&home);
    }
}
