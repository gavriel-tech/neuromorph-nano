# nmhash

SHA-256 for the Neuromorph Nano: card hashes, cluster and unit identity, file verification.
`src/sha256.rs` is in-house engine code of Gavriel Technologies (see `THIRD_PARTY_NOTICES.md`);
`src/lib.rs` adds `sha256_bytes` and the section check (two NIST vectors). Bottom of the
dependency order; the one hash module of the workspace.

Section checks: `neuromorph-nano check nmhash`.
