//! nmpod: the Neuropod, the distributed contextual nervous system (guide 4, 4.1 to 4.3,
//! 12, 17.1; PLAN 4c). One unit type for tentacles and drone tips, any job; readers at the
//! tip; local reflexes, confidence, anomaly and distillation in every unit; territories and
//! demand budgets; recursion only through the Core Brain. Depends on nmjson, nmtype, nmlex.
//!
//! 2026-11-10 (step P4 first form, one tentacle system; specification 2.2 to 2.4, design by
//! Luis Jake Gabriel III): a tentacle owns its window (`window`, laid out per question over the
//! whole code base; it replaces the territory as what the tentacle reads), its drone tip (every
//! reader behind the one `tip::Reader` interface: attention-head drone readers of a bound neural
//! core and the exact readers), its core (`core`: the neural core when bound, else the stated
//! exact fallback) and its `g0_tx` (held by the adapter in `nmapi::tentacle_unit`, so this crate
//! does not depend on the engine). Territories stay for the served control path and G's
//! coverage segments.
//!
//! 2026-11-10 (drone intake, first form, off by default; direction of Luis Jake Gabriel III,
//! 14:15 to 14:26 UTC): `intake`, the drones as the point of intake (tokens, token embedding rows
//! by reference, sensory records, first sensing, hand-off to the tentacle). Adds nmtokz and nmsafe.
pub mod attention;
pub mod core;
pub mod evidence;
mod fixture;
pub mod intake;
pub mod pipeline;
pub mod pod;
pub mod record;
pub mod recursion;
pub mod territory;
pub mod tip;
pub mod unit;
pub mod window;

pub use crate::core::{CoreKind, ExactReport};
pub use pipeline::{Reflex, ReflexTable, Signals};
pub use pod::{DEFAULT_JOBS, Neuropod};
pub use tip::{Finding, Reach, ReadCtx, Reader};
pub use unit::{Assignment, Job, Level, Outcome, Unit};
pub use window::{Span, Window};

/// One section check: (sub-component id, check returning ok or one fault line).
pub type Check = (&'static str, fn() -> Result<(), String>);

/// Section checks of this crate, one per sub-component, in dependency order.
#[must_use]
pub fn checks() -> Vec<Check> {
    vec![
        ("nmpod.tip", tip::check),
        ("nmpod.intake", intake::check),
        ("nmpod.evidence", evidence::check),
        ("nmpod.pipeline", pipeline::check),
        ("nmpod.record", record::check),
        ("nmpod.territory", territory::check),
        ("nmpod.window", window::check),
        ("nmpod.attention", attention::check),
        ("nmpod.recursion", recursion::check),
        ("nmpod.unit", unit::check),
        ("nmpod.core", core::check),
        ("nmpod.pod", pod::check),
    ]
}
