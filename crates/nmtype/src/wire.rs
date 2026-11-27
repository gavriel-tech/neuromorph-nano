//! Private JSON helpers shared by the nmtype sub-components: building values, reading
//! typed fields, and the enum-name macro. Not a sub-component (no card, no check).
use crate::component::{Fault, FaultKind};
use nmjson::Value;
use std::collections::BTreeMap;

pub fn obj(fields: Vec<(&str, Value)>) -> Value {
    Value::Object(fields.into_iter().map(|(k, v)| (k.to_owned(), v)).collect::<BTreeMap<_, _>>())
}
pub fn s(text: &str) -> Value {
    Value::String(text.to_owned())
}
pub fn u(n: u64) -> Value {
    Value::Number(n.to_string())
}
/// Shortest round-trip spelling; non-finite values are stored as 0 (JSON has no NaN).
pub fn f(x: f32) -> Value {
    Value::Number(if x.is_finite() { x.to_string() } else { "0".into() })
}
pub fn strings(items: &[String]) -> Value {
    Value::Array(items.iter().map(|t| s(t)).collect())
}
pub fn opt_s(text: Option<&str>) -> Value {
    text.map_or(Value::Null, s)
}
pub fn opt_u(n: Option<u64>) -> Value {
    n.map_or(Value::Null, u)
}

pub fn bad(module: &str, detail: impl Into<String>) -> Fault {
    Fault::new(&format!("nmtype.{module}"), FaultKind::Interface, detail)
}
fn field<'a>(v: &'a Value, module: &str, key: &str) -> Result<&'a Value, Fault> {
    v.get(key).map_err(|e| bad(module, format!("{key}: {e}")))
}
pub fn get_s(v: &Value, module: &str, key: &str) -> Result<String, Fault> {
    field(v, module, key)?.string().map(str::to_owned).map_err(|e| bad(module, format!("{key}: {e}")))
}
pub fn get_u(v: &Value, module: &str, key: &str) -> Result<u64, Fault> {
    field(v, module, key)?.integer().map_err(|e| bad(module, format!("{key}: {e}")))
}
pub fn get_n<T: TryFrom<u64>>(v: &Value, module: &str, key: &str) -> Result<T, Fault> {
    T::try_from(get_u(v, module, key)?).map_err(|_| bad(module, format!("{key}: out of range")))
}
pub fn get_f(v: &Value, module: &str, key: &str) -> Result<f32, Fault> {
    match field(v, module, key)? {
        Value::Number(t) => t.parse().map_err(|_| bad(module, format!("{key}: not a number"))),
        _ => Err(bad(module, format!("{key}: expected number"))),
    }
}
pub fn get_b(v: &Value, module: &str, key: &str) -> Result<bool, Fault> {
    match field(v, module, key)? {
        Value::Bool(b) => Ok(*b),
        _ => Err(bad(module, format!("{key}: expected bool"))),
    }
}
pub fn get_a<'a>(v: &'a Value, module: &str, key: &str) -> Result<&'a [Value], Fault> {
    field(v, module, key)?.array().map_err(|e| bad(module, format!("{key}: {e}")))
}
pub fn get_strings(v: &Value, module: &str, key: &str) -> Result<Vec<String>, Fault> {
    get_a(v, module, key)?.iter().map(|i| i.string().map(str::to_owned).map_err(|e| bad(module, format!("{key}: {e}")))).collect()
}
pub fn get_opt_s(v: &Value, module: &str, key: &str) -> Result<Option<String>, Fault> {
    match field(v, module, key)? {
        Value::Null => Ok(None),
        Value::String(t) => Ok(Some(t.clone())),
        _ => Err(bad(module, format!("{key}: expected string or null"))),
    }
}
pub fn get_opt_u(v: &Value, module: &str, key: &str) -> Result<Option<u64>, Fault> {
    match field(v, module, key)? {
        Value::Null => Ok(None),
        other => other.integer().map(Some).map_err(|e| bad(module, format!("{key}: {e}"))),
    }
}
pub fn get_v<'a>(v: &'a Value, module: &str, key: &str) -> Result<&'a Value, Fault> {
    field(v, module, key)
}

/// Declare a closed enum with stable lowercase names, `name()`, `from_name()`,
/// `to_value()`, `from_value()` and `ALL`.
macro_rules! named_enum {
    ($(#[$m:meta])* $vis:vis enum $name:ident in $module:literal { $($(#[$vm:meta])* $var:ident => $text:literal),+ $(,)? }) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        $vis enum $name { $($(#[$vm])* $var),+ }
        impl $name {
            /// Every variant, in declaration order.
            pub const ALL: &'static [Self] = &[$(Self::$var),+];
            /// The stable lowercase name.
            #[must_use]
            pub const fn name(self) -> &'static str {
                match self { $(Self::$var => $text),+ }
            }
            /// Parse a stable name.
            #[must_use]
            pub fn from_name(text: &str) -> Option<Self> {
                match text { $($text => Some(Self::$var),)+ _ => None }
            }
            /// JSON form: the stable name as a string.
            #[must_use]
            pub fn to_value(self) -> nmjson::Value {
                nmjson::Value::String(self.name().to_owned())
            }
            /// Read the JSON form.
            /// # Errors
            /// Rejects a non-string or an unknown name.
            pub fn from_value(v: &nmjson::Value) -> Result<Self, crate::component::Fault> {
                let text = v.string().map_err(|e| crate::wire::bad($module, e.to_string()))?;
                Self::from_name(text).ok_or_else(|| {
                    crate::wire::bad($module, format!("unknown {} {text:?}", stringify!($name)))
                })
            }
        }
    };
}
pub(crate) use named_enum;

/// Read an enum field by key.
pub fn get_e<T>(v: &Value, module: &str, key: &str, read: fn(&Value) -> Result<T, Fault>) -> Result<T, Fault> {
    read(field(v, module, key)?)
}
