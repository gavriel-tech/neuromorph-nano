//! Runs every section check of nmpod (2026-11-10, step P4 first form), so
//! `cargo test --release -p nmpod` covers the territory path, the window path, the exact core
//! and the reader reach rules on the in-crate fixture.
#[test]
fn section_checks() {
    let mut failed = Vec::new();
    for (id, check) in nmpod::checks() {
        if let Err(e) = check() {
            failed.push(format!("{id}: {e}"));
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
}
