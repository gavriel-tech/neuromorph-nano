//! nmengn: the neural engine over the one Qwen3-0.6B population (guide 8, 9.4, 9.5, 13.1;
//! PLAN 4c `nmengn`). No Block type: an execution assembly (data) connects neural components,
//! each binding its own parameter clusters through NPCS. Sub-components: `safetensors` (crate
//! `nmsafe`, a byte copy of emma-labs `native/rscore/loader`), `tokenizer` (crate `nmtokz`, a
//! byte copy of `native/rscore/tokenz`), `half` (binary16 rounding), `cpu` and `gpu` (the
//! backend: CPU kernels and the copied cukern kernels through `nmffi`), `placement`,
//! `assembly`, `model` (population and bound assemblies), `base` (the base source of a
//! MicroModel's parameters: the open weights or one complete base cartridge, seated and pulled), `components`, `roles` (embed and
//! rerank over the base), `reference` (the engine reference recorded by this engine, and parity against it), `decode` (greedy
//! stepper, S7), `adapter` (learning-weight units on linear nodes, S8), `filter` (the sockets
//! where cartridges plug into a MicroModel's layers and filter them), `kvstore` (encoded
//! chunks and shared key and value windows passed by reference, E1 2026-11-10), `residency` (the
//! shared device budget for window copies, weights and idle pool buffers, step P2 2026-11-10), `drone`
//! (drone-key packets, keys-only scan and question path, tentacle core A6 and A7, 2026-11-10), `tentacle`
//! (a tentacle's own assembly generated from its drone set: sensory segments through its last drone
//! layer and its drone tip, 2026-11-10).
//! Depends on nmjson,
//! nmtype, nmnpcs, nmsafe, nmtokz, nmffi, nmhash (window content keys, step P2).
pub mod adapter;
pub mod assembly;
pub mod autotune;
pub mod base;
pub mod capacity;
pub mod chanrec;
pub mod components;
pub mod cpu;
pub mod decode;
/// Roadmap Phase 4.6 (2026-11-25): the card choice and the processor fallback line.
pub mod device;
pub mod drone;
pub mod filter;
pub mod gpu;
pub mod half;
pub mod kvstore;
pub mod limits;
pub mod model;
/// Parameter overlays: edits, masks and freezes held beside an object and applied when it is seated.
pub mod overlay;
pub mod placement;
pub mod pmap;
pub mod reference;
pub mod residency;
pub mod roles;
pub mod speculate;
pub mod selftest;
pub mod tentacle;

pub use adapter::Adapter;
pub use assembly::{Assembly, Shape};
pub use model::{Act, Cache, Dims, Model, Population, RunOpts};
/// Stage timing spans (off unless switched on), for callers of the engine.
pub use nmffi::prof;
pub use nmsafe as safetensors;
pub use nmtokz as tokenizer;
pub use placement::{Device, Placement};

/// One section check: (sub-component id, check returning ok or one fault line).
pub type Check = (&'static str, fn() -> Result<(), String>);

/// Section checks of this crate (the copied reader and tokenizer have their own crates).
#[must_use]
pub fn checks() -> Vec<Check> {
    vec![
        ("nmengn.half", half::check),
        ("nmengn.cpu", cpu::check),
        ("nmengn.assembly", assembly::check),
        ("nmengn.placement", placement::check),
        ("nmengn.placement.gate", placement::check_gate),
        ("nmengn.adapter", adapter::check),
        ("nmengn.pmap", pmap::check),
        ("nmengn.filter", filter::check),
        ("nmengn.filter.model", filter::check_model),
        ("nmengn.kvstore", kvstore::check),
        ("nmengn.residency", residency::check),
        ("nmengn.kvstore.upload", kvstore::check_upload),
        ("nmengn.kvstore.capture16", kvstore::check_capture16),
        ("nmengn.kvstore.logits", kvstore::check_logits),
        ("nmengn.drone", drone::check),
        ("nmengn.tentacle", tentacle::check),
        ("nmengn.residency.weights", residency::check_weights),
        ("nmengn.model.kv_groups", model::check_kv_groups),
        ("nmengn.model.by_node", model::check_by_node),
        ("nmengn.decode", decode::check),
        ("nmengn.gpu.prefill2", gpu::check_prefill2),
        ("nmengn.gpu.flash", gpu::check_flash),
        ("nmengn.gpu.h2q8", gpu::check_h2q8),
        ("nmengn.gpu.dp4a_tile", gpu::check_dp4a_tile),
        ("nmengn.gpu.h2q8_tile", gpu::check_h2q8_tile),
        ("nmengn.gpu.fastel", gpu::check_fastel),
        ("nmengn.gpu.segments", gpu::check_segments),
        ("nmengn.gpu.wide_q8f", gpu::check_wide_q8f),
        ("nmengn.gpu.flash_segments", gpu::check_flash_segments),
        ("nmengn.gpu.flash_h", gpu::check_flash_h),
        ("nmengn.gpu.h2_fetch", gpu::check_h2_fetch),
        ("nmengn.gpu.bmq8", gpu::check_bmq8),
        ("nmengn.autotune", autotune::check),
        ("nmengn.speculate", speculate::check),
        ("nmengn.roles", roles::check),
    ]
}
