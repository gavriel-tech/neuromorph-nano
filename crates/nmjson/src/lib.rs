//! Strict JSON value, parser and canonical encoder. Copied from emma-labs `prtdef` (see build record runs/2026-11-07-build-record-copied.md).
pub mod check;
mod encode;
mod error;
pub mod json;
pub mod pretty;
pub use error::{Error, Result};
pub use json::{Value, parse};
