# nmjson

Strict JSON value, parser and canonical encoder (sorted keys, no spaces). `json.rs`, `encode.rs`
and `error.rs` are in-house engine code of Gavriel Technologies (see `THIRD_PARTY_NOTICES.md`);
`check.rs` holds the section checks. Bottom of the dependency order; used by every crate.

Section checks: `neuromorph-nano check nmjson`.
