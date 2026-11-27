//! Recoverable contract and framework errors.
use std::fmt;
/// A contextual failure with a stable code and retained underlying cause.
#[derive(Debug)]
#[non_exhaustive]
pub struct Error {
    /// Stable crate-prefixed diagnostic code.
    pub code: &'static str,
    /// What failed, including identities and observed values.
    pub detail: String,
    /// Exact input or action needed to recover.
    pub repair: String,
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}
impl Error {
    /// Construct an input or contract failure with no side effects.
    pub fn new(code: &'static str, detail: impl Into<String>, repair: impl Into<String>) -> Self {
        Self { code, detail: detail.into(), repair: repair.into(), source: None }
    }
    pub(crate) fn invalid(detail: impl Into<String>) -> Self {
        Self::new("prtdef-001", detail, "Supply a valid part card, cgraph or typed value")
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {}. {}. Recovery: fix then retry. Validation side effects: none; completed calls/log appends remain committed",
            self.code, self.detail, self.repair
        )?;
        if let Some(source) = &self.source {
            write!(f, ". Cause: {source}")?;
        }
        Ok(())
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_deref().map(|source| -> &(dyn std::error::Error + 'static) { source })
    }
}
/// Result shared by the stable part interface.
pub type Result<T> = std::result::Result<T, Error>;
