//! nmbrain: the Core Brain (guide 3, 6, 6.1 to 6.4; PLAN 4c), S2 form: the four slots on
//! their exact controls, epistemic arbitration, the Tension Buffer, synthesis, the river
//! mailbox and the Context Stack as a Neuropod interface; at S5 objectives, homeostasis and
//! development. Engines bind to the slots at S6 and S7. Depends on nmjson, nmtype, nmlex.
pub mod allocation;
pub mod arbiter;
pub mod chain;
pub mod contradiction;
pub mod development;
pub mod evidence;
pub mod g0;
pub mod g1;
pub mod g2;
pub mod gmap;
pub mod homeostasis;
pub mod instruct;
pub mod motor;
pub mod objectives;
pub mod reflex;
pub mod rivers;
pub mod slot;
pub mod stack;
pub mod synthesis;

pub use slot::{Controls, Exact, Slot, Slots};
pub use stack::{CoreBrain, Forager};

/// One section check: (sub-component id, check returning ok or one fault line).
pub type Check = (&'static str, fn() -> Result<(), String>);

/// Section checks of this crate, one per built sub-component, in dependency order.
#[must_use]
pub fn checks() -> Vec<Check> {
    vec![
        ("nmbrain.slot", slot::check),
        ("nmbrain.arbiter", arbiter::check),
        ("nmbrain.contradiction", contradiction::check),
        ("nmbrain.rivers", rivers::check),
        ("nmbrain.synthesis", synthesis::check),
        ("nmbrain.stack", stack::check),
        ("nmbrain.objectives", objectives::check),
        ("nmbrain.homeostasis", homeostasis::check),
        ("nmbrain.development", development::check),
        ("nmbrain.gmap", gmap::check),
        ("nmbrain.g0", g0::check),
        ("nmbrain.g0.reflexes", g0::check_reflexes),
        ("nmbrain.evidence", evidence::check),
        ("nmbrain.g1", g1::check),
        ("nmbrain.chain", chain::check),
        ("nmbrain.chain.intake", chain::check_intake),
        ("nmbrain.g2", g2::check),
        ("nmbrain.g2.items", g2::check_items),
        ("nmbrain.instruct", instruct::check),
        ("nmbrain.motor", motor::check),
        ("nmbrain.reflex", reflex::check),
        ("nmbrain.allocation", allocation::check),
    ]
}
