//! Typed ports and payloads (guide 8.1, 14.1): components exchange tensors, packets,
//! memory references, JSON or text, never one universal float vector.
use crate::component::Fault;
use crate::memory::MemoryRef;
use crate::packet::Packet;
use crate::tensor::Tensor;
use crate::wire::{self, named_enum};
use nmjson::Value;

named_enum! {
    /// The eight prtdef kinds plus three typed payloads.
    pub enum PortKind in "port" {
        Text => "text", Documents => "documents", Json => "json", Choice => "choice",
        Path => "path", Run => "run", Bool => "bool", Any => "any",
        Packets => "packets", Tensor => "tensor", Refs => "refs",
    }
}

/// One named, typed port.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortDesc {
    pub name: String,
    pub kind: PortKind,
    pub required: bool,
}
impl PortDesc {
    #[must_use]
    pub fn new(name: &str, kind: PortKind, required: bool) -> Self {
        Self { name: name.to_owned(), kind, required }
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        wire::obj(vec![("name", wire::s(&self.name)), ("kind", self.kind.to_value()), ("required", Value::Bool(self.required))])
    }
    /// # Errors
    /// Rejects a value missing any field.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        Ok(Self {
            name: wire::get_s(v, "port", "name")?,
            kind: wire::get_e(v, "port", "kind", PortKind::from_value)?,
            required: wire::get_b(v, "port", "required")?,
        })
    }
}

/// Can an output port feed an input port? Same kind, or an `Any` input. An `Any` output
/// feeds only an `Any` input, because nothing guarantees what it carries.
#[must_use]
pub fn compatible(out: &PortDesc, inp: &PortDesc) -> bool {
    inp.kind == PortKind::Any || out.kind == inp.kind
}

/// What travels on an edge.
#[derive(Clone, Debug, PartialEq)]
pub enum Payload {
    Packets(Vec<Packet>),
    Tensor(Tensor),
    Refs(Vec<MemoryRef>),
    Json(Value),
    Text(String),
}
impl Payload {
    /// The port kind this payload satisfies.
    #[must_use]
    pub const fn kind(&self) -> PortKind {
        match self {
            Self::Packets(_) => PortKind::Packets,
            Self::Tensor(_) => PortKind::Tensor,
            Self::Refs(_) => PortKind::Refs,
            Self::Json(_) => PortKind::Json,
            Self::Text(_) => PortKind::Text,
        }
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let (tag, body) = match self {
            Self::Packets(p) => ("packets", Value::Array(p.iter().map(Packet::to_value).collect())),
            Self::Tensor(t) => ("tensor", t.to_value()),
            Self::Refs(r) => ("refs", Value::Array(r.iter().map(MemoryRef::to_value).collect())),
            Self::Json(j) => ("json", j.clone()),
            Self::Text(t) => ("text", wire::s(t)),
        };
        wire::obj(vec![("payload", wire::s(tag)), ("body", body)])
    }
    /// # Errors
    /// Rejects an unknown tag or a malformed body.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let body = wire::get_v(v, "port", "body")?;
        let list = || body.array().map_err(|e| wire::bad("port", e.to_string()));
        match wire::get_s(v, "port", "payload")?.as_str() {
            "packets" => Ok(Self::Packets(list()?.iter().map(Packet::from_value).collect::<Result<_, _>>()?)),
            "tensor" => Ok(Self::Tensor(Tensor::from_value(body)?)),
            "refs" => Ok(Self::Refs(list()?.iter().map(MemoryRef::from_value).collect::<Result<_, _>>()?)),
            "json" => Ok(Self::Json(body.clone())),
            "text" => Ok(Self::Text(body.string().map_err(|e| wire::bad("port", e.to_string()))?.to_owned())),
            other => Err(wire::bad("port", format!("unknown payload {other:?}"))),
        }
    }
}

/// Self-check of the `port` sub-component: the compatibility table and a payload round trip.
pub(crate) fn check() -> Result<(), String> {
    let p = |k| PortDesc::new("p", k, true);
    let table = [
        (PortKind::Packets, PortKind::Packets, true),
        (PortKind::Tensor, PortKind::Any, true),
        (PortKind::Any, PortKind::Tensor, false),
        (PortKind::Tensor, PortKind::Packets, false),
        (PortKind::Refs, PortKind::Refs, true),
    ];
    for (out, inp, want) in table {
        if compatible(&p(out), &p(inp)) != want {
            return Err(format!("nmtype.port: {} -> {} should be {want}", out.name(), inp.name()));
        }
    }
    let pay = Payload::Text("hello".into());
    if Payload::from_value(&pay.to_value()).map_err(|e| e.to_string())? != pay {
        return Err("nmtype.port: payload round trip".into());
    }
    Ok(())
}
