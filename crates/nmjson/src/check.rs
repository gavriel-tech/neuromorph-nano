//! Section checks for the three copied sub-components (`json`, `encode`, `error`).
//! Kept in their own file so the copied files stay byte-identical to their source.
use crate::{Error, Value, parse};

/// `json`: strict parse of a small object, and rejection of a duplicate key.
fn json() -> Result<(), String> {
    let v = parse(br#"{"a":[1,"x",true,null],"b":{"c":"d"}}"#).map_err(|e| e.to_string())?;
    let a = v.get("a").and_then(Value::array).map_err(|e| e.to_string())?;
    if a.len() != 4 {
        return Err("nmjson.json: array length".into());
    }
    if parse(br#"{"a":1,"a":2}"#).is_ok() {
        return Err("nmjson.json: duplicate key accepted".into());
    }
    Ok(())
}

/// `encode`: canonical output sorts keys and round-trips through `parse`.
fn encode() -> Result<(), String> {
    let v = parse(br#"{"z":1,"a":"x"}"#).map_err(|e| e.to_string())?;
    let text = v.encode().map_err(|e| e.to_string())?;
    if !text.starts_with("{\"a\"") {
        return Err(format!("nmjson.encode: keys not sorted: {text}"));
    }
    let back = parse(text.as_bytes()).map_err(|e| e.to_string())?;
    if back != v {
        return Err("nmjson.encode: round trip changed the value".into());
    }
    Ok(())
}

/// `error`: the stable code and the repair text reach the display line.
fn error() -> Result<(), String> {
    let line = Error::new("nmjson-check", "detail", "repair").to_string();
    if line.contains("nmjson-check") && line.contains("repair") {
        Ok(())
    } else {
        Err("nmjson.error: display misses code or repair".into())
    }
}

/// `pretty`: readable output parses back to the same value.
fn pretty() -> Result<(), String> {
    let v = parse(br#"{"a":[1,2],"b":{"c":[{"d":null}]},"e":{}}"#).map_err(|e| e.to_string())?;
    let back = parse(crate::pretty::pretty(&v).as_bytes()).map_err(|e| e.to_string())?;
    if back == v { Ok(()) } else { Err("nmjson.pretty: round trip changed the value".into()) }
}

/// One section check: (sub-component id, check returning ok or one fault line).
pub type Check = (&'static str, fn() -> Result<(), String>);

/// Section checks of this crate, in order: (sub-component id, check).
#[must_use]
pub fn checks() -> Vec<Check> {
    vec![("nmjson.json", json), ("nmjson.encode", encode), ("nmjson.error", error), ("nmjson.pretty", pretty)]
}
