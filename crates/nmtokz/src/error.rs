//! Contextual errors without third-party dependencies.
use std::fmt;

/// A recoverable tokenz input or I/O failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Invalid input with its underlying context.
    Invalid(String),
    /// File operation failure retaining its I/O cause.
    Io {
        /// File or operation context.
        context: String,
        /// Original operating-system error.
        source: std::io::Error,
    },
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(detail) => {
                write!(f, "tokenz-001: {detail}. Supply valid tokenz input. Recovery: fix then retry. Side effects: none.")
            }
            Self::Io { context, source } => {
                write!(f, "tokenz-002: {context}: {source}. Supply a readable local file. Recovery: fix then retry. Side effects: none.")
            }
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Invalid(_) => None,
        }
    }
}
/// Result returned by this crate.
pub type Result<T> = std::result::Result<T, Error>;
impl From<std::io::Error> for Error {
    fn from(source: std::io::Error) -> Self {
        Self::Io { context: "local file operation".into(), source }
    }
}
