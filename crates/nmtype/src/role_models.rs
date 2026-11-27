//! The MicroModel each role runs on (directions of Luis Jake Gabriel III, 2026-11-25: the
//! MicroModels are choosable, changeable and swappable per role, in the code and the interface).
//!
//! Four roles: the reader (the drones, the tentacles, the judge's reads), the writer (the idle
//! notes), the embedder (the meaning lane) and the reranker (the judge). Each defaults to the model
//! every measurement was made on; `settings.json` `role_models` (`{"writer": "<model id>"}`, written
//! by the library's role choice) names another installed model for a role, read once per process so
//! a choice applies from the next start. A choice is kept only when that model is installed (its
//! weight card is in the models folder) and is of the kind the role needs. The reader is locked:
//! its drones read attention heads mapped for one model, so another reader needs its own head map.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;

/// The roles and the model each runs on by default.
pub const ROLES: [(&str, &str); 4] = [("reader", "qwen3-0.6b"), ("writer", "qwen3-1.7b"), ("embedder", "qwen3-embedding-0.6b"), ("reranker", "qwen3-reranker-0.6b")];

/// The roles that cannot take another model yet, with the reason the interface shows.
pub const LOCKED: [(&str, &str); 1] = [("reader", "its drones read attention heads mapped for qwen3-0.6b; another reader needs its own head map first")];

/// The settings key that holds the choices.
pub const SETTINGS_KEY: &str = "role_models";

/// What a model is, read from its id and weight card: an embedding model, a reranker, or a
/// generative model (the reader and the writer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Embedding,
    Reranker,
    Generative,
}

impl Kind {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::Reranker => "reranker",
            Self::Generative => "generative",
        }
    }
}

/// The default model of `role`.
#[must_use]
pub fn default_model(role: &str) -> Option<&'static str> {
    ROLES.iter().find(|(r, _)| *r == role).map(|(_, m)| *m)
}

/// Why `role` is locked, when it is.
#[must_use]
pub fn locked(role: &str) -> Option<&'static str> {
    LOCKED.iter().find(|(r, _)| *r == role).map(|(_, why)| *why)
}

/// The kind a role needs.
#[must_use]
pub fn kind_for_role(role: &str) -> Option<Kind> {
    match role {
        "embedder" => Some(Kind::Embedding),
        "reranker" => Some(Kind::Reranker),
        "reader" | "writer" => Some(Kind::Generative),
        _ => None,
    }
}

/// The kind of the installed model `id`: its id names it (`embedding`, `rerank`), else the Hugging
/// Face repository its weight card names as its origin does; otherwise generative. (The card's free
/// text is not read: a generative model's description may name the roles it once served.)
#[must_use]
pub fn kind_of(home: &Path, id: &str) -> Kind {
    let origin = std::fs::read(crate::models_dir(home).join(id).join("part.json"))
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .and_then(|v| v.get("weights").and_then(|w| w.get("origin")).ok().and_then(|o| o.string().ok()).map(str::to_owned))
        .unwrap_or_default();
    let names = format!("{id} {}", origin.rsplit('/').next().unwrap_or("")).to_ascii_lowercase();
    if names.contains("embedding") || names.contains("embed") {
        Kind::Embedding
    } else if names.contains("rerank") {
        Kind::Reranker
    } else {
        Kind::Generative
    }
}

/// "a" or "an" before a kind's name.
fn article(word: &str) -> &'static str {
    if word.starts_with(['a', 'e', 'i', 'o', 'u']) { "an" } else { "a" }
}

/// Every installed model: the folders of the models folder that hold a weight card, by id.
#[must_use]
pub fn installed(home: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(crate::models_dir(home))
        .map(|rd| rd.flatten().filter(|e| e.path().join("part.json").is_file()).filter_map(|e| e.file_name().to_str().map(str::to_owned)).collect())
        .unwrap_or_default();
    v.sort();
    v
}

/// Whether `id` may run as `role` on this machine: installed, of the kind the role needs, and the
/// role not locked. A model serves one role, except that the writer may share the reader's model
/// (both generative; 2026-11-25, directions of Luis Jake Gabriel III: swap among the models the
/// system already has): the shared model keeps the reader's precision and job filters, and the
/// writer runs with that model's cartridges while its own stay plugged into its default model,
/// untouched. `Err` says why not.
/// # Errors
/// The reason, in one line.
pub fn may_run(home: &Path, role: &str, id: &str, others: &BTreeMap<String, String>) -> Result<(), String> {
    let need = kind_for_role(role).ok_or_else(|| format!("{role}: not a role (reader, writer, embedder or reranker)"))?;
    if let Some(why) = locked(role) {
        if default_model(role) != Some(id) {
            return Err(format!("the {role} is locked: {why}"));
        }
        return Ok(());
    }
    if !installed(home).iter().any(|m| m == id) {
        return Err(format!("{id} is not installed (no weight card in the models folder)"));
    }
    let kind = kind_of(home, id);
    if kind != need {
        let why = match role {
            "embedder" => ": its vectors fill the dense index, and another model's vectors do not match the index",
            "reranker" => ": the judge reads a yes or no through a reranker's trained head",
            _ => "",
        };
        return Err(format!("{id} is {} {} model; the {role} needs {} {} model{why}", article(kind.name()), kind.name(), article(need.name()), need.name()));
    }
    if let Some((r, _)) = others.iter().find(|(r, m)| r.as_str() != role && m.as_str() == id) {
        let shareable = (role == "writer" && r == "reader") || (role == "reader" && r == "writer");
        if !shareable {
            return Err(format!("{id} already runs as the {r}; only the writer may share a model (the reader's)"));
        }
    }
    Ok(())
}

/// The choices in `settings.json`, each kept only when it may run (else the default stands).
#[must_use]
pub fn resolve(home: &Path) -> BTreeMap<String, String> {
    let saved: BTreeMap<String, String> = std::fs::read(home.join("settings.json"))
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .and_then(|v| v.get(SETTINGS_KEY).ok().cloned())
        .and_then(|v| if let nmjson::Value::Object(o) = v { Some(o) } else { None })
        .map(|o| o.into_iter().filter_map(|(k, v)| v.string().ok().map(|s| (k, s.to_owned()))).collect())
        .unwrap_or_default();
    let mut out: BTreeMap<String, String> = ROLES.iter().map(|(r, m)| ((*r).to_owned(), (*m).to_owned())).collect();
    for (role, id) in &saved {
        if !out.contains_key(role) {
            continue;
        }
        let mut others = out.clone();
        others.remove(role);
        if may_run(home, role, id, &others).is_ok() {
            out.insert(role.clone(), id.clone());
        }
    }
    out
}

fn map() -> &'static BTreeMap<String, String> {
    static MAP: OnceLock<BTreeMap<String, String>> = OnceLock::new();
    MAP.get_or_init(|| resolve(&crate::nano_root()))
}

/// The model `role` runs on in this process (read once; a new choice applies from the next start).
#[must_use]
pub fn model(role: &str) -> String {
    map().get(role).cloned().or_else(|| default_model(role).map(str::to_owned)).unwrap_or_default()
}

/// The role the model `id` runs as in this process, matched by the end of the id (card ids may carry
/// a prefix), or `None` for a model no role runs on.
#[must_use]
pub fn role_of(id: &str) -> Option<&'static str> {
    let m = map();
    ROLES.iter().map(|(r, _)| *r).find(|r| m.get(*r).is_some_and(|x| id == x || id.ends_with(&format!("-{x}")) || id.ends_with(&format!("/{x}"))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_kinds() {
        assert_eq!(default_model("writer"), Some("qwen3-1.7b"));
        assert!(locked("reader").is_some() && locked("writer").is_none());
        let home = std::env::temp_dir().join(format!("nm-role-models-{}", std::process::id()));
        assert_eq!(kind_of(&home, "qwen3-embedding-4b"), Kind::Embedding);
        assert_eq!(kind_of(&home, "qwen3-reranker-4b"), Kind::Reranker);
        assert_eq!(kind_of(&home, "qwen3-4b"), Kind::Generative);
    }

    #[test]
    fn choices_need_an_installed_model_of_the_right_kind() {
        let home = std::env::temp_dir().join(format!("nm-role-models-choice-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        for id in ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-embedding-0.6b", "qwen3-reranker-0.6b"] {
            let d = home.join("weights").join(id);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("part.json"), "{}").unwrap();
        }
        std::fs::write(
            home.join("settings.json"),
            r#"{"role_models": {"writer": "qwen3-4b", "embedder": "qwen3-4b", "reranker": "qwen3-9b", "reader": "qwen3-4b"}}"#,
        )
        .unwrap();
        let r = resolve(&home);
        assert_eq!(r.get("writer").map(String::as_str), Some("qwen3-4b"));
        assert_eq!(r.get("embedder").map(String::as_str), Some("qwen3-embedding-0.6b"), "a generative model cannot embed");
        assert_eq!(r.get("reranker").map(String::as_str), Some("qwen3-reranker-0.6b"), "not installed");
        assert_eq!(r.get("reader").map(String::as_str), Some("qwen3-0.6b"), "the reader is locked");
        let mut others = r.clone();
        others.remove("writer");
        assert!(may_run(&home, "writer", "qwen3-0.6b", &others).is_ok(), "the writer may share the reader's model");
        assert!(may_run(&home, "embedder", "qwen3-0.6b", &others).is_err(), "a generative model cannot embed");
        let _ = std::fs::remove_dir_all(&home);
    }
}
