//! SHA-256 for cards, clusters, units and copies. `sha256.rs` is a byte-identical copy of
//! emma-labs `native/rscore/owners/src/hash.rs` (see build record runs/2026-11-07-build-record-copied.md); this file adds only the
//! byte-slice helper and the section check.
mod sha256;
pub use sha256::sha256;

/// Hash an in-memory byte slice. Reading from a slice cannot fail, so the error branch
/// returns an empty string, which never equals a real digest.
#[must_use]
pub fn sha256_bytes(bytes: &[u8]) -> String {
    sha256(bytes).unwrap_or_default()
}

/// The one self-check of the `sha256` sub-component: two NIST vectors.
/// # Errors
/// Returns one fault line naming the vector that failed.
pub fn check() -> Result<(), String> {
    for (raw, want) in [
        ("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        ("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
    ] {
        if sha256_bytes(raw.as_bytes()) != want {
            return Err(format!("nmhash.sha256: wrong digest for {raw:?}"));
        }
    }
    Ok(())
}

/// One section check: (sub-component id, check returning ok or one fault line).
pub type Check = (&'static str, fn() -> Result<(), String>);

/// Section checks of this crate, in order: (sub-component id, check).
#[must_use]
pub fn checks() -> Vec<Check> {
    vec![("nmhash.sha256", check)]
}
