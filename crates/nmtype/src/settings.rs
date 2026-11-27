//! The broadcast settings record (research note, gap 3 cross-analysis, 2026-11-10): a small
//! typed set of global values every part may read, with a version counter, carried on its
//! own channel apart from addressed in-house messages. Behavioural class. Immutable: `set`
//! returns a new record with the version raised by one, so a reader can tell at once
//! whether what it holds is current.
use crate::class::{Classed, DataClass};
use crate::component::Fault;
use crate::wire;
use nmjson::Value;
use std::collections::BTreeMap;

/// One typed setting value.
#[derive(Clone, Debug, PartialEq)]
pub enum SettingValue {
    Flag(bool),
    Count(u64),
    Level(f32),
    Word(String),
}
impl SettingValue {
    #[must_use]
    pub fn to_value(&self) -> Value {
        let (t, v) = match self {
            Self::Flag(b) => ("flag", Value::Bool(*b)),
            Self::Count(n) => ("count", wire::u(*n)),
            Self::Level(x) => ("level", wire::f(*x)),
            Self::Word(w) => ("word", wire::s(w)),
        };
        wire::obj(vec![("type", wire::s(t)), ("value", v)])
    }
    /// # Errors
    /// Rejects an unknown type or a value of the wrong shape.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let m = "settings";
        Ok(match wire::get_s(v, m, "type")?.as_str() {
            "flag" => Self::Flag(wire::get_b(v, m, "value")?),
            "count" => Self::Count(wire::get_u(v, m, "value")?),
            "level" => Self::Level(wire::get_f(v, m, "value")?),
            "word" => Self::Word(wire::get_s(v, m, "value")?),
            other => return Err(wire::bad(m, format!("unknown setting type {other:?}"))),
        })
    }
}

/// The broadcast record: version plus named values. No sender or target: it goes to all.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Settings {
    version: u64,
    values: BTreeMap<String, SettingValue>,
}
impl Settings {
    /// A new record with `key` set and the version raised by one. Setting a value equal to
    /// the current one returns the record unchanged (no version churn).
    #[must_use]
    pub fn set(&self, key: &str, value: SettingValue) -> Self {
        if self.values.get(key) == Some(&value) {
            return self.clone();
        }
        let mut values = self.values.clone();
        values.insert(key.to_owned(), value);
        Self { version: self.version.saturating_add(1), values }
    }
    #[must_use]
    pub const fn version(&self) -> u64 {
        self.version
    }
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&SettingValue> {
        self.values.get(key)
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    /// Whether `other` is newer than this record.
    #[must_use]
    pub const fn is_stale_against(&self, other: &Self) -> bool {
        other.version > self.version
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let values = self.values.iter().map(|(k, v)| (k.clone(), v.to_value())).collect::<BTreeMap<_, _>>();
        wire::obj(vec![("class", DataClass::Behavioural.to_value()), ("version", wire::u(self.version)), ("values", Value::Object(values))])
    }
    /// # Errors
    /// Rejects a malformed record or one tagged with any class but behavioural.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let m = "settings";
        let class = wire::get_e(v, m, "class", DataClass::from_value)?;
        crate::class::guard("broadcast", DataClass::Behavioural, class)?;
        let values = match wire::get_v(v, m, "values")? {
            Value::Object(map) => map.iter().map(|(k, x)| SettingValue::from_value(x).map(|s| (k.clone(), s))).collect::<Result<_, _>>()?,
            _ => return Err(wire::bad(m, "values: expected object")),
        };
        Ok(Self { version: wire::get_u(v, m, "version")?, values })
    }
}
impl Classed for Settings {
    fn data_class(&self) -> DataClass {
        DataClass::Behavioural
    }
}

/// Self-check of the `settings` sub-component: versions rise on change only, the record
/// round-trips, and a record tagged sensory is refused.
pub(crate) fn check() -> Result<(), String> {
    let a = Settings::default();
    let b = a.set("comprehension-first", SettingValue::Flag(true));
    let c = b.set("comprehension-first", SettingValue::Flag(true));
    let d = c
        .set("tentacles", SettingValue::Count(8))
        .set("ram-share", SettingValue::Level(0.5))
        .set("mode", SettingValue::Word("standalone".into()));
    if b.version() != 1 || c.version() != 1 || d.version() != 4 || !b.is_stale_against(&d) {
        return Err("nmtype.settings: version counter".into());
    }
    if Settings::from_value(&d.to_value()).map_err(|e| e.to_string())? != d {
        return Err("nmtype.settings: round trip".into());
    }
    let mut tagged = d.to_value();
    if let Value::Object(map) = &mut tagged {
        map.insert("class".into(), DataClass::Sensory.to_value());
    }
    if Settings::from_value(&tagged).is_ok() {
        return Err("nmtype.settings: sensory-tagged record accepted".into());
    }
    Ok(())
}
