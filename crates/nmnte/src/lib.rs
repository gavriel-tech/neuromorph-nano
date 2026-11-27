//! nmnte: the Neural Topology Executor and the RNAM foundations (guide 11.1, 14, 14.1 to
//! 14.3; PLAN 4c). The anatomy is data (`graph`), validated (`integrity`, `topology`,
//! `registry`), published as immutable snapshots (`snapshot`), and executed in frames
//! (`frame`) by the scheduler over a bounded worker pool (`scheduler`, `runtime`).
//! NTE executes the anatomy, RNAM develops it, NPCS provisions its parameters; these stay
//! separate. No transformer block, layer count or forward order lives here: the executor
//! runs whatever graph a snapshot holds.
pub mod binding;
pub mod compose;
pub mod frame;
pub mod graph;
pub mod homeo;
pub mod integrity;
pub mod lifecycle;
pub mod plasticity;
pub mod registry;
pub mod runtime;
pub mod scheduler;
pub mod snapshot;
pub mod topology;

pub use frame::{ExecutionFrame, ExecutionState, MAX_RUN_MS};
pub use graph::{ConnectionMode, Edge, Graph, Node};
pub use lifecycle::{Phase, PhaseLog};
pub use plasticity::Change;
pub use registry::{CardState, Registry};
pub use runtime::{ContextExec, Exec, FnExec, Runtime, TensorExec};
pub use scheduler::{ExecutionPlan, RunResult, plan, resume, run};
pub use snapshot::{Components, Nte, Predicate, Snapshot, SnapshotExec};

/// One section check: (sub-component id, check returning ok or one fault line).
pub type Check = (&'static str, fn() -> Result<(), String>);

/// Section checks of this crate, one per built sub-component, in dependency order.
#[must_use]
pub fn checks() -> Vec<Check> {
    vec![
        ("nmnte.graph", graph::check),
        ("nmnte.topology", topology::check),
        ("nmnte.registry", registry::check),
        ("nmnte.integrity", integrity::check),
        ("nmnte.runtime", runtime::check),
        ("nmnte.frame", frame::check),
        ("nmnte.snapshot", snapshot::check),
        ("nmnte.scheduler", scheduler::check),
        ("nmnte.compose", compose::check),
        ("nmnte.binding", binding::check),
        ("nmnte.plasticity", plasticity::check),
        ("nmnte.lifecycle", lifecycle::check),
        ("nmnte.homeo", homeo::check),
    ]
}
