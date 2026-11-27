//! nmlearn: real-time modular learning as learning-weight units (guide 10, 10.1 to 10.6;
//! Comprehensive Vision 9, 9.1 to 9.5, 12.6; PLAN 4c `nmlearn`, 4d item 7). A unit is a new,
//! labelled, quantized low-rank adapter stacked beside the frozen clusters of one component;
//! it is fitted forward-only (MeZO) from checked outcomes in runs under one minute, promoted
//! only through a Step Check, loaded and unloaded as the exact rollback, never merged in place.
//! Sub-components: `unit` (card, weights, quantized files), `registry` (register, load,
//! unload, rollback, overlap refusal, write-once commit), `isolation` (bit-for-bit check),
//! `outcomes` (the checked-output log, the only training data), `stepspec` (scopes, rungs,
//! the eight-criterion report), `forward` (the fitter under a budget), `adapt` (parameter,
//! functional and structural adaptations routed to their mechanism), `rls` (a recursive least
//! squares readout beside a component output, redesigned from EAMM V108.1), `readout` (the
//! readout's scopes, paired 3a test, accept rule, benefit gate, decay and retire). Depends on nmjson,
//! nmhash, nmsafe, nmtype; the engine is reached through a trait object from nmapi.
pub mod adapt;
pub mod calibrate;
pub mod cartridge;
pub mod engine;
pub mod forward;
pub mod hardware;
pub mod idle;
pub mod isolation;
pub mod learned;
pub mod ledger;
pub mod library;
pub mod outcomes;
pub mod pref_fit;
pub mod preference;
pub mod readout;
pub mod registry;
pub mod rls;
pub mod route;
pub mod stated;
pub mod stepspec;
pub mod tuner;
pub mod unit;

pub use adapt::{Adaptation, Applied};
pub use forward::{FitLog, Fitter, Loss, Mezo};
pub use outcomes::Outcome;
pub use registry::Units;
pub use stepspec::{Rung, Scope, StepCheckSpec, StepReport, Verdict};
pub use unit::{UnitCard, UnitWeights};

/// One section check: (sub-component id, check returning ok or one fault line).
pub type Check = (&'static str, fn() -> Result<(), String>);

/// Section checks of this crate, one per sub-component, in dependency order.
#[must_use]
pub fn checks() -> Vec<Check> {
    vec![
        ("nmlearn.unit", unit::check),
        ("nmlearn.outcomes", outcomes::check),
        ("nmlearn.stepspec", stepspec::check),
        ("nmlearn.registry", registry::check),
        ("nmlearn.isolation", isolation::check),
        ("nmlearn.forward", forward::check),
        ("nmlearn.rls", rls::check),
        ("nmlearn.readout", readout::check),
        ("nmlearn.adapt", adapt::check),
        ("nmlearn.cartridge", cartridge::check),
        ("nmlearn.preference", preference::check),
        ("nmlearn.stated", stated::check),
        ("nmlearn.route", route::check),
        ("nmlearn.ledger", ledger::check),
        ("nmlearn.calibrate", calibrate::check),
        ("nmlearn.tuner", tuner::check),
        ("nmlearn.learned", learned::check),
        ("nmlearn.hardware", hardware::check),
        ("nmlearn.library", library::check),
        ("nmlearn.engine", engine::check),
    ]
}
