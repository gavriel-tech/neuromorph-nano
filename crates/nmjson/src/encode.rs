//! Canonical JSON output shared by cards and hop records.
use crate::{Error, Result, json::Value};
use std::fmt::Write;
impl Value {
    /// Encode compact deterministic JSON, sorting object keys.
    /// # Errors
    /// Rejects a caller-constructed invalid number or formatting failure.
    pub fn encode(&self) -> Result<String> {
        let mut out = String::new();
        self.emit(&mut out)?;
        Ok(out)
    }
    fn emit(&self, out: &mut String) -> Result<()> {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            Self::String(text) => quoted(text, out)?,
            Self::Number(text) => {
                if !matches!(crate::json::parse(text.as_bytes())?, Self::Number(_)) {
                    return Err(Error::invalid("invalid JSON number"));
                }
                out.push_str(text);
            }
            Self::Array(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index != 0 {
                        out.push(',');
                    }
                    item.emit(out)?;
                }
                out.push(']');
            }
            Self::Object(items) => {
                out.push('{');
                for (index, (key, value)) in items.iter().enumerate() {
                    if index != 0 {
                        out.push(',');
                    }
                    quoted(key, out)?;
                    out.push(':');
                    value.emit(out)?;
                }
                out.push('}');
            }
        }
        Ok(())
    }
}
fn quoted(text: &str, out: &mut String) -> Result<()> {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if c.is_control() => {
                write!(out, "\\u{:04x}", u32::from(c)).map_err(|e| Error::invalid(e.to_string()))?;
            }
            c => out.push(c),
        }
    }
    out.push('"');
    Ok(())
}
