//! nmmem: memory at recursive levels and identity (guide 7, 7.1, 17.2; PLAN 4c). Working
//! (bounded, RAM), episodic (append-only verified outcomes), structural (per substrate
//! version), the saved-memory hierarchy (1-core, 2-active, 3-archive), temp notes (promoted when used often, else cleared), identity records, the relevance gate with the provenance cap, consolidation,
//! and tier budgets. State lives in the mini's `memory/` folder, versioned and append-only
//! where it records history. Depends on nmjson, nmhash, nmlex, nmtype.
pub mod book;
pub mod classes;
pub mod consolidation;
pub mod episodic;
pub mod evidence;
pub mod gate;
pub mod hierarchy;
pub mod identity;
pub mod journal;
pub mod keep;
pub mod mem2;
pub mod migrate;
pub mod onewriter;
pub mod outcomes;
pub mod pass;
pub mod recall2;
pub mod record;
pub mod retention;
pub mod runday;
pub mod session_open;
mod store;
pub mod structural;
pub mod temp;
pub mod tiers;
pub mod working;

pub use episodic::{Episode, Episodic};
pub use gate::{Gate, WriterPolicy};
pub use hierarchy::{Hierarchy, HierarchyPolicy};
pub use identity::Identity;
pub use temp::{Temp, TempPolicy};
pub use working::Working;

/// The working folder of one section check: `tests/section-checks/nmmem-<name>/`, reused
/// and cleared at each run (`nmtype::check_dir`).
pub(crate) fn scratch_dir(name: &str) -> Result<std::path::PathBuf, String> {
    nmtype::check_dir(&format!("nmmem-{name}")).map_err(|e| format!("scratch: {e}"))
}

/// One section check: (sub-component id, check returning ok or one fault line).
pub type Check = (&'static str, fn() -> Result<(), String>);

/// Section checks of this crate, one per sub-component.
#[must_use]
pub fn checks() -> Vec<Check> {
    vec![
        ("nmmem.working", working::check),
        ("nmmem.episodic", episodic::check),
        ("nmmem.structural", structural::check),
        ("nmmem.identity", identity::check),
        ("nmmem.gate", gate::check),
        ("nmmem.consolidation", consolidation::check),
        ("nmmem.tiers", tiers::check),
        ("nmmem.temp", temp::check),
        ("nmmem.hierarchy", hierarchy::check),
        ("nmmem.runday", runday::check),
        ("nmmem.journal", journal::check),
        ("nmmem.record", record::check),
        ("nmmem.record.reconsolidate", record::check_reconsolidate),
        ("nmmem.book", book::check),
        ("nmmem.retention", retention::check),
        ("nmmem.classes", classes::check),
        ("nmmem.pass", pass::check),
        ("nmmem.recall2", recall2::check),
        ("nmmem.migrate", migrate::check),
        ("nmmem.mem2", mem2::check),
        ("nmmem.onewriter", onewriter::check),
        ("nmmem.outcomes", outcomes::check),
        ("nmmem.evidence", evidence::check),
        ("nmmem.keep", keep::check),
        ("nmmem.session_open", session_open::check),
    ]
}
