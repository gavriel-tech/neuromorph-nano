//! Compatibility (guide 9.3 responsibility 5; PLAN 4c): before a component binds a cluster,
//! its family, element type and block shape must be what the component's port expects; a
//! learning-weight unit made for another base population is refused.
use crate::decompose::ParameterCluster;
use nmtype::{Fault, FaultKind};

/// What a component's parameter port wants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Want {
    pub family: String,
    pub dtypes: Vec<String>,
    /// Block shape (rows, columns); `None` in a place accepts any size.
    pub rows: Option<usize>,
    pub cols: Option<usize>,
}

/// Check one cluster against a port.
/// # Errors
/// A mismatch, named.
pub fn check_cluster(c: &ParameterCluster, family: &str, want: &Want) -> Result<(), Fault> {
    let f = |m: String| Err(Fault::new("nmnpcs.compat", FaultKind::Interface, format!("{}: {m}", c.id)));
    if family != want.family {
        return f(format!("family {family}, port wants {}", want.family));
    }
    if !want.dtypes.iter().any(|d| d == &c.dtype) {
        return f(format!("dtype {}, port wants {:?}", c.dtype, want.dtypes));
    }
    let (r, k) = c.dims();
    if want.rows.is_some_and(|w| w != r) || want.cols.is_some_and(|w| w != k) {
        return f(format!("block {r}x{k}, port wants {:?}x{:?}", want.rows, want.cols));
    }
    Ok(())
}

/// A unit card's base hash must equal the population it adapts.
/// # Errors
/// The hashes differ.
pub fn check_unit_base(unit: &str, unit_base_sha: &str, population_sha: &str) -> Result<(), Fault> {
    if unit_base_sha == population_sha {
        Ok(())
    } else {
        Err(Fault::new(
            "nmnpcs.compat",
            FaultKind::Interface,
            format!("unit {unit} was made for base {unit_base_sha}, not {population_sha}; refused"),
        ))
    }
}

pub(crate) fn check() -> Result<(), String> {
    use crate::decompose::Region;
    let c = ParameterCluster {
        id: "m/q/rows0-128".into(),
        model: "m".into(),
        tensor: "q".into(),
        region: Region::Rows(0..128),
        group: "g".into(),
        dtype: "BF16".into(),
        shape: vec![2048, 1024],
        file: "toy.safetensors".into(),
        offset: 0,
        len: 0,
        bytes: 0,
        sha256: String::new(),
    };
    let want = Want { family: "qwen3".into(), dtypes: vec!["BF16".into(), "F16".into()], rows: Some(128), cols: Some(1024) };
    check_cluster(&c, "qwen3", &want).map_err(|e| e.to_string())?;
    if check_cluster(&c, "llama", &want).is_ok()
        || check_cluster(&c, "qwen3", &Want { cols: Some(512), ..want }).is_ok()
        || check_unit_base("u", "aa", "bb").is_ok()
    {
        return Err("nmnpcs.compat: a mismatch passed".into());
    }
    Ok(())
}
