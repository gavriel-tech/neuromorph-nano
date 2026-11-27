//! Readable JSON for design files people edit: two-space indent, one field per line, short
//! scalar arrays on one line. Parses back to the same value as the canonical encoder.
use crate::Value;

fn scalar(v: &Value) -> bool {
    !matches!(v, Value::Object(_) | Value::Array(_))
}

/// Pretty-print a value (keys in their stored, sorted order).
#[must_use]
pub fn pretty(v: &Value) -> String {
    let mut out = String::new();
    write(v, 0, &mut out);
    out.push('\n');
    out
}

fn write(v: &Value, depth: usize, out: &mut String) {
    let pad = |d: usize| "  ".repeat(d);
    match v {
        Value::Object(m) if m.is_empty() => out.push_str("{}"),
        Value::Object(m) if m.values().all(scalar) && m.len() <= 6 => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&Value::String(k.clone()).encode().unwrap_or_default());
                out.push_str(": ");
                out.push_str(&x.encode().unwrap_or_default());
            }
            out.push('}');
        }
        Value::Object(m) => {
            out.push_str("{\n");
            for (i, (k, x)) in m.iter().enumerate() {
                out.push_str(&pad(depth + 1));
                out.push_str(&Value::String(k.clone()).encode().unwrap_or_default());
                out.push_str(": ");
                write(x, depth + 1, out);
                out.push_str(if i + 1 < m.len() { ",\n" } else { "\n" });
            }
            out.push_str(&pad(depth));
            out.push('}');
        }
        Value::Array(a) if a.is_empty() => out.push_str("[]"),
        Value::Array(a) if a.iter().all(scalar) => out.push_str(&v.encode().unwrap_or_default()),
        Value::Array(a) => {
            out.push_str("[\n");
            for (i, x) in a.iter().enumerate() {
                out.push_str(&pad(depth + 1));
                write(x, depth + 1, out);
                out.push_str(if i + 1 < a.len() { ",\n" } else { "\n" });
            }
            out.push_str(&pad(depth));
            out.push(']');
        }
        other => out.push_str(&other.encode().unwrap_or_default()),
    }
}
