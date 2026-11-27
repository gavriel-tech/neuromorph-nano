//! Memory references, queries and write outcomes shared by every memory level (guide 7).
use crate::component::Fault;
use crate::wire;
use nmjson::Value;

/// A stored record: tier (1 working, 2 episodic, 3 structural), id, substrate version.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryRef {
    pub tier: u8,
    pub id: String,
    pub version: u64,
}
impl MemoryRef {
    #[must_use]
    pub fn to_value(&self) -> Value {
        wire::obj(vec![("tier", wire::u(u64::from(self.tier))), ("id", wire::s(&self.id)), ("version", wire::u(self.version))])
    }
    /// # Errors
    /// Rejects a value missing any field.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        Ok(Self {
            tier: wire::get_n(v, "memory", "tier")?,
            id: wire::get_s(v, "memory", "id")?,
            version: wire::get_u(v, "memory", "version")?,
        })
    }
}

/// Recall request: text, scope key (`t3`, `d1`, `task:7`, `brain`), top k, minimum overlap.
#[derive(Clone, Debug, PartialEq)]
pub struct Query {
    pub text: String,
    pub scope: String,
    pub k: usize,
    pub min_overlap: f32,
}
impl Query {
    #[must_use]
    pub fn to_value(&self) -> Value {
        wire::obj(vec![
            ("text", wire::s(&self.text)),
            ("scope", wire::s(&self.scope)),
            ("k", wire::u(self.k as u64)),
            ("min_overlap", wire::f(self.min_overlap)),
        ])
    }
    /// # Errors
    /// Rejects a value missing any field.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        Ok(Self {
            text: wire::get_s(v, "memory", "text")?,
            scope: wire::get_s(v, "memory", "scope")?,
            k: wire::get_n(v, "memory", "k")?,
            min_overlap: wire::get_f(v, "memory", "min_overlap")?,
        })
    }
}

/// Outcome of a gated write: stored, already present (same hash identity), or refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Written {
    Stored(MemoryRef),
    Duplicate(MemoryRef),
    Refused(String),
}
impl Written {
    #[must_use]
    pub fn to_value(&self) -> Value {
        match self {
            Self::Stored(r) => wire::obj(vec![("stored", r.to_value())]),
            Self::Duplicate(r) => wire::obj(vec![("duplicate", r.to_value())]),
            Self::Refused(why) => wire::obj(vec![("refused", wire::s(why))]),
        }
    }
    /// # Errors
    /// Rejects a value that is not one of the three forms.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        if let Ok(r) = v.get("stored") {
            return Ok(Self::Stored(MemoryRef::from_value(r)?));
        }
        if let Ok(r) = v.get("duplicate") {
            return Ok(Self::Duplicate(MemoryRef::from_value(r)?));
        }
        Ok(Self::Refused(wire::get_s(v, "memory", "refused")?))
    }
}

/// Self-check of the `memory` sub-component: the three write outcomes and a query round-trip.
pub(crate) fn check() -> Result<(), String> {
    let r = MemoryRef { tier: 2, id: "ep-1".into(), version: 3 };
    for w in [Written::Stored(r.clone()), Written::Duplicate(r), Written::Refused("no evidence".into())] {
        if Written::from_value(&w.to_value()).map_err(|e| e.to_string())? != w {
            return Err("nmtype.memory: written round trip".into());
        }
    }
    let q = Query { text: "Packet".into(), scope: "brain".into(), k: 5, min_overlap: 0.2 };
    if Query::from_value(&q.to_value()).map_err(|e| e.to_string())? != q {
        return Err("nmtype.memory: query round trip".into());
    }
    Ok(())
}
