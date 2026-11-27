//! nmnpcs: the Neural Parameter Compatibility System and the Parameter Map (guide 9, 9.1 to
//! 9.6, 10.4; PLAN 4c, 6d 9.3 map). `acquire` (weight cards, discovery, verification by
//! hash), `decompose` (a population cut into clusters, coverage audit), `compat` (shape, type,
//! family, unit base), `provision` (one read per tensor, one `Arc<View>` per cluster,
//! references, residency, reclamation), `lifecycle` (five checked states), `inspect` (read
//! only, for `nm_weights`) and `parmap` (S5). Depends on nmjson, nmhash, nmtype.
pub mod acquire;
pub mod atlas;
pub mod cartridge;
pub mod compare;
pub mod compat;
pub mod decompose;
pub mod inspect;
pub mod lifecycle;
pub mod parmap;
pub mod provision;

pub use acquire::WeightCard;
pub use decompose::{ParameterCluster, Region, TensorDesc};
pub use parmap::ParMap;
pub use provision::{Provisioner, Storage, View};

/// One section check: (sub-component id, check returning ok or one fault line).
pub type Check = (&'static str, fn() -> Result<(), String>);

/// Section checks of this crate.
#[must_use]
pub fn checks() -> Vec<Check> {
    vec![
        ("nmnpcs.acquire", acquire::check),
        ("nmnpcs.decompose", decompose::check),
        ("nmnpcs.compat", compat::check),
        ("nmnpcs.lifecycle", lifecycle::check),
        ("nmnpcs.provision", provision::check),
        ("nmnpcs.inspect", inspect::check),
        ("nmnpcs.parmap", parmap::check),
        ("nmnpcs.atlas", atlas::check),
        ("nmnpcs.compare", compare::check),
        ("nmnpcs.cartridge", cartridge::check),
    ]
}
