//! Read-only views a unit is given (PLAN 4c nmpod `Unit::run`): the substrate it reads and
//! the Core Brain slots it may ask. Defined here, in the contract crate, so nmpod never
//! depends on nmsubs or nmbrain, and a unit holds no handle to anything but these views.
use crate::component::Fault;
use crate::packet::Packet;
use nmjson::Value;

/// One line found in the substrate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hit {
    pub file: String,
    pub line: usize,
    pub text: String,
}

/// What a unit may read of the Global Context Substrate.
pub trait SubstrateView: Send + Sync {
    /// The substrate version every packet from it carries.
    fn version(&self) -> u64;
    /// Every file path, sorted (territories are cut from this list).
    fn files(&self) -> Vec<String>;
    /// One of the eight views for a query.
    fn view(&self, name: &str, query: &str) -> Vec<Packet>;
    /// Lines containing every term, inside the given paths (empty means everywhere).
    fn grep(&self, terms: &[String], paths: &[String], limit: usize) -> Vec<Hit>;
    /// Definitions of a symbol by exact name; `text` is the defining line.
    fn symbols(&self, name: &str) -> Vec<Hit>;
    /// The `k` nearest semantic chunks inside the given paths (empty means everywhere). The
    /// default filters the global semantic view; a substrate ranks inside the paths, so every
    /// territory gets its own nearest chunks rather than a share of the global top four.
    fn semantic_in(&self, query: &str, paths: &[String], k: usize) -> Vec<Packet> {
        self.view("semantic", query)
            .into_iter()
            .filter(|p| paths.is_empty() || paths.iter().any(|x| p.provenance().source.starts_with(x.as_str())))
            .take(k)
            .collect()
    }
}

/// What a unit may ask of a Core Brain slot (through the brain, never directly of a model).
pub trait SlotView: Send + Sync {
    /// # Errors
    /// Unknown slot, or the slot's own fault.
    fn call(&self, slot: &str, input: &Value) -> Result<Value, Fault>;
}

/// Self-check of the `view` sub-component: both traits are usable as trait objects over a
/// built-in fixture.
/// Ordering weight of a cited source by what kind of file it is (context pack fix,
/// 2026-11-08): code first, then build and data files, then prose. Plan and note documents
/// describe the code and can be older than it (S12 part C t5, self gate C1), so they order
/// after the code they describe; they still answer when no code matches; run records order
/// after the current docs (2026-11-09). Orders only.
#[must_use]
pub fn source_weight(source: &str) -> f32 {
    let path = source.split(':').next().unwrap_or(source);
    let ext = path.rsplit_once('.').map_or("", |(_, e)| e);
    // Run records (`runs/`) describe the code as it was at that run; they order after the
    // current docs (CM1 revision 2, 2026-11-09: they led packs on "how is X done" questions).
    if path.starts_with("runs/") || path.contains("/runs/") {
        return 0.6;
    }
    match ext {
        "rs" | "cu" | "cuh" | "ps1" | "psm1" | "py" | "pyi" | "cmd" | "ts" | "tsx" | "mts" | "cts" | "js" | "jsx" | "mjs" | "cjs" | "c" | "h"
        | "cpp" | "cc" | "cxx" | "hpp" | "hh" | "hxx" | "css" => 1.0,
        "toml" | "json" | "jsonl" => 0.95,
        "md" | "txt" => 0.75,
        _ if ["unit:", "git:", "memory"].iter().any(|p| source.starts_with(p)) => 1.0,
        _ => 0.9,
    }
}

/// True when a read path lies inside a hidden folder (a folder whose name starts with a dot): a
/// project's own configuration, read since 2026-11-17 (`nmsubs::repo`), whose files are scored
/// but kept out of the material's rarities, so reading them shifts no other piece's weights.
#[must_use]
pub fn in_hidden_folder(path: &str) -> bool {
    let disk = path.split('!').next().unwrap_or(path).replace('\\', "/");
    let mut parts: Vec<&str> = disk.split('/').collect();
    parts.pop();
    parts.iter().any(|p| p.starts_with('.') && *p != "." && *p != "..")
}

/// True when a cited source is a document or data file (prose, tables, records, logs), not code.
#[must_use]
pub fn is_document(source: &str) -> bool {
    let path = source.split(':').next().unwrap_or(source);
    let name = path.rsplit('/').next().unwrap_or(path).to_lowercase();
    let ext = name.rsplit_once('.').map_or("", |(_, e)| e);
    matches!(
        ext,
        "md" | "markdown" | "mdx" | "txt" | "text" | "rst" | "adoc" | "asciidoc" | "org" | "csv" | "tsv" | "tab" | "html" | "htm" | "xhtml" | "xml"
            | "json" | "jsonl" | "ndjson" | "ipynb" | "yaml" | "yml" | "log" | "pdf" | "rtf" | "doc" | "docx" | "xls" | "xlsx" | "ppt" | "pptx" | "odt"
            | "ods" | "odp" | "epub"
    ) || matches!(name.as_str(), "readme" | "license" | "licence" | "changelog" | "notice" | "authors" | "copying" | "todo" | "install")
}

/// `source_weight` for a question: when the question asks for something documents hold (a
/// definition, a procedure, a summary, a comparison: `document_kind`) and names no code
/// identifier, a document weighs as much as code; otherwise the weights are `source_weight`'s.
/// Run records keep their lower weight.
#[must_use]
pub fn source_weight_q(source: &str, document_kind: bool) -> f32 {
    let path = source.split(':').next().unwrap_or(source);
    if document_kind && is_document(source) && !(path.starts_with("runs/") || path.contains("/runs/")) {
        1.0
    } else {
        source_weight(source)
    }
}

pub(crate) fn check() -> Result<(), String> {
    struct Fixture;
    impl SubstrateView for Fixture {
        fn version(&self) -> u64 {
            3
        }
        fn files(&self) -> Vec<String> {
            vec!["a.rs".into()]
        }
        fn view(&self, _: &str, _: &str) -> Vec<Packet> {
            Vec::new()
        }
        fn grep(&self, terms: &[String], _: &[String], _: usize) -> Vec<Hit> {
            terms.iter().map(|t| Hit { file: "a.rs".into(), line: 1, text: t.clone() }).collect()
        }
        fn symbols(&self, _: &str) -> Vec<Hit> {
            Vec::new()
        }
    }
    impl SlotView for Fixture {
        fn call(&self, slot: &str, _: &Value) -> Result<Value, Fault> {
            Ok(Value::String(slot.to_owned()))
        }
    }
    let (s, k): (&dyn SubstrateView, &dyn SlotView) = (&Fixture, &Fixture);
    if s.version() != 3 || s.grep(&["x".into()], &[], 1).len() != 1 {
        return Err("nmtype.view: substrate view".into());
    }
    if k.call("router", &Value::Null).ok() != Some(Value::String("router".into())) {
        return Err("nmtype.view: slot view".into());
    }
    if !(source_weight("crates/a/src/x.rs:3") > source_weight("Cargo.toml:1")
        && source_weight("Cargo.toml:1") > source_weight("PLAN.md:400"))
    {
        return Err("nmtype.view: source weight does not order code, then config, then prose".into());
    }
    Ok(())
}
