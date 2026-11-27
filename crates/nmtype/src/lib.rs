//! nmtype: the contract crate of the Neuromorph Nano (guide 8, 8.1, 20, 21; PLAN 4c).
//! Context Packet, component identity and lifecycle, typed ports, cards, hops, tensors,
//! tasks and memory references. Depends only on nmjson and nmhash; every other crate
//! depends on it. All traits are synchronous and `Send`; concurrency is the NTE's job.
pub mod budget;
pub mod card;
pub mod class;
pub mod component;
pub mod doc;
pub mod examples;
pub mod hop;
pub mod limits;
pub mod memory;
pub mod message;
pub mod role_models;
pub mod order;
pub mod packet;
pub mod port;
pub mod project;
pub mod settings;
pub mod task;
pub mod tensor;
pub mod view;
mod wire;

pub use card::{Card, PartRef, card_hash};
pub use class::{Behavioural, Boundary, Classed, DataClass, Sensory, WRONG_CLASS, guard};
pub use component::{
    Component, ComponentDescriptor, ComponentId, ComponentType, ContextProcessor, Fault, FaultKind, FromCard, Health, MemoryComponent,
    Neuropod, Resolve, Telemetry, TensorProcessor,
};
pub use hop::{Hop, HopLog, now_ms};
pub use memory::{MemoryRef, Query, Written};
pub use message::{ActionLedger, Envelope, MessageKind, Reference, Resolver};
pub use packet::{Category, Certainty, Packet, Provenance, Stance, Status};
pub use port::{Payload, PortDesc, PortKind, compatible};
pub use settings::{SettingValue, Settings};
pub use task::{Budget, FrameView, Pack, Plan, Region, Task, TaskKind};
pub use tensor::{DType, Tensor, TensorData};
pub use view::{Hit, SlotView, SubstrateView, in_hidden_folder, is_document, source_weight, source_weight_q};

/// One section check: (sub-component id, check returning ok or one fault line).
pub type Check = (&'static str, fn() -> Result<(), String>);

/// The marker file at the top of a Neuromorph Nano folder (name, version, layout version).
pub const MARKER: &str = "nano.json";

/// The Neuromorph Nano folder, found at run time so the folder works wherever it sits. Order:
/// the `NANO_HOME` environment variable when it names a folder; else the nearest folder above the
/// running program, then above the working directory, that holds the marker file `nano.json`;
/// else, for a development tree, the nearest such folder that holds `rules.md` and
/// `crates/nmcore/Cargo.toml`.
///
/// # Errors
/// A message naming what was looked for when no folder qualifies.
pub fn find_root() -> Result<std::path::PathBuf, String> {
    use std::path::{Path, PathBuf};
    fn above(p: &Path, found: fn(&Path) -> bool) -> Option<PathBuf> {
        p.ancestors().find(|d| found(d)).map(Path::to_path_buf)
    }
    fn marked(d: &Path) -> bool {
        d.join(MARKER).is_file()
    }
    fn dev_tree(d: &Path) -> bool {
        d.join("rules.md").is_file() && d.join("crates/nmcore/Cargo.toml").is_file()
    }
    if let Some(h) = std::env::var_os("NANO_HOME").filter(|h| !h.is_empty()) {
        let h = PathBuf::from(h);
        return if h.is_dir() { Ok(h) } else { Err(format!("NANO_HOME={} is not a folder", h.display())) };
    }
    let exe = std::env::current_exe().ok();
    let cwd = std::env::current_dir().ok();
    for found in [marked as fn(&Path) -> bool, dev_tree] {
        if let Some(d) = exe.as_deref().and_then(|e| above(e, found)).or_else(|| cwd.as_deref().and_then(|c| above(c, found))) {
            return Ok(d);
        }
    }
    Err(format!("Neuromorph Nano folder not found: set NANO_HOME, or run the program from inside the folder that holds {MARKER}"))
}

/// The Neuromorph Nano folder (`find_root`). When no folder qualifies, the working directory is
/// returned; the programs check `find_root` at start and stop with its message first.
#[must_use]
pub fn nano_root() -> std::path::PathBuf {
    find_root().unwrap_or_else(|_| std::env::current_dir().unwrap_or_default())
}

/// The models folder (Neuromorph Nano, 2026-11-15): where the released open weights are read
/// from, never written. Order: the `NANO_MODELS` environment variable; then `dir` in
/// `designs/models.json` under `home` (relative paths are taken from `home`); then
/// `<home>/weights`. Each model sits in `<models>/<id>/` with `part.json` and `files/`.
#[must_use]
pub fn models_dir(home: impl AsRef<std::path::Path>) -> std::path::PathBuf {
    let home = home.as_ref();
    if let Some(d) = std::env::var_os("NANO_MODELS").filter(|d| !d.is_empty()) {
        return std::path::PathBuf::from(d);
    }
    if let Ok(bytes) = std::fs::read(home.join("designs").join("models.json"))
        && let Ok(v) = nmjson::parse(&bytes)
        && let Ok(d) = v.get("dir").and_then(nmjson::Value::string)
        && !d.is_empty()
    {
        let d = std::path::Path::new(d);
        return if d.is_absolute() { d.to_path_buf() } else { home.join(d) };
    }
    home.join("weights")
}

/// The mini's test folder for one category, `tests/<category>` (Luis Jake Gabriel III, 2026-11-08: tests
/// organized and categorized by what they test): `memory-system`, `capability-gates/<cN-...>`,
/// `stage-gates/<sN>`, `claude-code-sessions`, `mcp-launcher`, `section-checks`,
/// `removed-from-live-memory`. The mini never studies `tests/`.
#[must_use]
pub fn tests_dir(category: &str) -> std::path::PathBuf {
    nano_root().join("tests").join(category)
}

/// The one working folder of a section check, `tests/section-checks/<name>/` (Luis
/// 2026-11-09: no doubles). The check's own output from its previous run is cleared first,
/// so each check keeps exactly one folder instead of a new one per run.
/// # Errors
/// The folder could not be cleared or created.
pub fn check_dir(name: &str) -> std::io::Result<std::path::PathBuf> {
    let d = tests_dir("section-checks").join(name);
    if d.exists() {
        std::fs::remove_dir_all(&d)?;
    }
    std::fs::create_dir_all(&d)?;
    Ok(d)
}

/// Section checks of this crate, one per sub-component, in dependency order.
#[must_use]
pub fn checks() -> Vec<Check> {
    vec![
        ("nmtype.hop", hop::check),
        ("nmtype.packet", packet::check),
        ("nmtype.tensor", tensor::check),
        ("nmtype.memory", memory::check),
        ("nmtype.port", port::check),
        ("nmtype.component", component::check),
        ("nmtype.card", card::check),
        ("nmtype.task", task::check),
        ("nmtype.view", view::check),
        ("nmtype.settings", settings::check),
        ("nmtype.message", message::check),
        ("nmtype.class", class::check),
        ("nmtype.order", order::check),
        ("nmtype.project", project::check),
        ("nmtype.doc", doc::check),
        ("nmtype.examples", examples::check),
    ]
}
