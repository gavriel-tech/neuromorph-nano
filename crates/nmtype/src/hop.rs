//! One hop of a packet or frame trace, and the append-only hop log (from `fwcore/packet`
//! `HopLog`, written fresh): hops are added, never edited or removed.
//! Since 2026-11-10 (step P5) a hop also carries the data class of what it moved: sensory
//! by default (`Hop::new`, and `from_value` when the field is absent), behavioural through
//! `with_class`. The JSON form writes `class` only for behavioural hops, so every record
//! written before the tag existed reads back unchanged.
use crate::class::{Classed, DataClass};
use crate::component::{ComponentId, Fault};
use crate::wire;
use nmjson::Value;

/// Who touched it, when (ms since the Unix epoch), what it did, and what it cost.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hop {
    pub component: ComponentId,
    pub at: u64,
    pub action: String,
    pub cost_ms: u32,
    /// Data class of the item this hop moved (default sensory).
    pub class: DataClass,
}
impl Hop {
    #[must_use]
    pub fn new(component: &str, at: u64, action: &str, cost_ms: u32) -> Self {
        Self { component: component.to_owned(), at, action: action.to_owned(), cost_ms, class: DataClass::Sensory }
    }
    /// The same hop with its data class set.
    #[must_use]
    pub fn with_class(self, class: DataClass) -> Self {
        Self { class, ..self }
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut fields = vec![
            ("component", wire::s(&self.component)),
            ("at", wire::u(self.at)),
            ("action", wire::s(&self.action)),
            ("cost_ms", wire::u(u64::from(self.cost_ms))),
        ];
        if self.class != DataClass::Sensory {
            fields.push(("class", self.class.to_value()));
        }
        wire::obj(fields)
    }
    /// # Errors
    /// Rejects a value missing any field.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        Ok(Self {
            component: wire::get_s(v, "hop", "component")?,
            at: wire::get_u(v, "hop", "at")?,
            action: wire::get_s(v, "hop", "action")?,
            cost_ms: wire::get_n(v, "hop", "cost_ms")?,
            class: match v.get("class") {
                Ok(c) => DataClass::from_value(c)?,
                Err(_) => DataClass::Sensory,
            },
        })
    }
}
impl Classed for Hop {
    fn data_class(&self) -> DataClass {
        self.class
    }
}

/// Append-only list of hops. No method removes or rewrites an entry.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HopLog {
    hops: Vec<Hop>,
}
impl HopLog {
    pub fn append(&mut self, hop: Hop) {
        self.hops.push(hop);
    }
    #[must_use]
    pub fn hops(&self) -> &[Hop] {
        &self.hops
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        Value::Array(self.hops.iter().map(Hop::to_value).collect())
    }
    /// # Errors
    /// Rejects a non-array or a malformed hop.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let items = v.array().map_err(|e| wire::bad("hop", e.to_string()))?;
        Ok(Self { hops: items.iter().map(Hop::from_value).collect::<Result<_, _>>()? })
    }
}

/// Milliseconds since the Unix epoch (0 if the clock is before it).
#[must_use]
pub fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// Self-check of the `hop` sub-component: append keeps order, the log round-trips.
pub(crate) fn check() -> Result<(), String> {
    let mut log = HopLog::default();
    log.append(Hop::new("nmpod.unit#d1", 1, "read", 2));
    log.append(Hop::new("nmbrain.synthesis#b0", 3, "reduce", 1));
    if log.hops().len() != 2 || log.hops()[1].action != "reduce" {
        return Err("nmtype.hop: append order".into());
    }
    if HopLog::from_value(&log.to_value()).map_err(|e| e.to_string())? != log {
        return Err("nmtype.hop: round trip".into());
    }
    let b = Hop::new("nmbrain.g2#0", 4, "send", 0).with_class(DataClass::Behavioural);
    if Hop::from_value(&b.to_value()).map_err(|e| e.to_string())? != b || log.hops()[0].to_value().get("class").is_ok() {
        return Err("nmtype.hop: class tag round trip or default form".into());
    }
    Ok(())
}
