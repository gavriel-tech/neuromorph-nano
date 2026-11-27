//! Neural Topology Graph as data (guide 14.1): nodes with typed ports, directed edges with
//! a connection mode. Topology is data, never code (PLAN 2b). Read from and written to the
//! `designs/*.json` form: `{"nodes": [...], "edges": [...]}` with modes
//! `{"kind": "direct" | "buffered" | "asynchronous" | "feedback" | "conditional", ...}`.
use nmjson::Value;
use nmtype::{ComponentId, ComponentType, Fault, FaultKind, PortDesc};
use std::collections::BTreeMap;

fn bad(detail: impl Into<String>) -> Fault {
    Fault::new("nmnte.graph", FaultKind::Interface, detail)
}
fn text(v: &Value, key: &str) -> Result<String, Fault> {
    v.get(key).and_then(Value::string).map(str::to_owned).map_err(|e| bad(format!("{key}: {e}")))
}
fn small(v: &Value, key: &str) -> Result<u64, Fault> {
    v.get(key).and_then(Value::integer).map_err(|e| bad(format!("{key}: {e}")))
}
fn obj(fields: Vec<(&str, Value)>) -> Value {
    Value::Object(fields.into_iter().map(|(k, v)| (k.to_owned(), v)).collect::<BTreeMap<_, _>>())
}
fn s(t: &str) -> Value {
    Value::String(t.to_owned())
}

/// How an edge carries payloads (guide 14.1; PLAN 4d item 1 for feedback).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConnectionMode {
    /// The consumer waits for the producer in the same frame.
    Direct,
    /// Like direct, but up to `bound` payloads queue on the edge; the oldest is dropped
    /// beyond the bound, and the drop is traced.
    Buffered(usize),
    /// The consumer does not wait; it sees the latest payload the producer has made in this
    /// frame, if any. Not an ordering dependency.
    Asynchronous,
    /// A recurrent edge: re-runs the destination at most `max_iter` times per frame.
    Feedback { max_iter: u8 },
    /// Carries the payload only when the named predicate holds.
    Conditional(String),
}
impl ConnectionMode {
    /// Does this edge order execution (producer before consumer)?
    #[must_use]
    pub const fn orders(&self) -> bool {
        matches!(self, Self::Direct | Self::Buffered(_) | Self::Conditional(_))
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let num = |n: u64| Value::Number(n.to_string());
        match self {
            Self::Direct => obj(vec![("kind", s("direct"))]),
            Self::Buffered(b) => obj(vec![("kind", s("buffered")), ("bound", num(*b as u64))]),
            Self::Asynchronous => obj(vec![("kind", s("asynchronous"))]),
            Self::Feedback { max_iter } => obj(vec![("kind", s("feedback")), ("max_iter", num(u64::from(*max_iter)))]),
            Self::Conditional(p) => obj(vec![("kind", s("conditional")), ("predicate", s(p))]),
        }
    }
    /// # Errors
    /// Unknown kind or a missing field.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        match text(v, "kind")?.as_str() {
            "direct" => Ok(Self::Direct),
            "buffered" => Ok(Self::Buffered(usize::try_from(small(v, "bound")?).map_err(|_| bad("bound out of range"))?)),
            "asynchronous" => Ok(Self::Asynchronous),
            "feedback" => Ok(Self::Feedback { max_iter: u8::try_from(small(v, "max_iter")?).map_err(|_| bad("max_iter over 255"))? }),
            "conditional" => Ok(Self::Conditional(text(v, "predicate")?)),
            other => Err(bad(format!("unknown connection mode {other:?}"))),
        }
    }
}

/// One node: a registered component instance placed in the anatomy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node {
    pub id: ComponentId,
    pub kind: ComponentType,
    pub parent: Option<ComponentId>,
    pub ports_in: Vec<PortDesc>,
    pub ports_out: Vec<PortDesc>,
    pub config: Value,
}
impl Node {
    /// A node with no parent, no ports and an empty config.
    #[must_use]
    pub fn new(id: &str, kind: ComponentType) -> Self {
        Self { id: id.to_owned(), kind, parent: None, ports_in: Vec::new(), ports_out: Vec::new(), config: Value::Object(BTreeMap::new()) }
    }
    #[must_use]
    pub fn input(mut self, port: PortDesc) -> Self {
        self.ports_in.push(port);
        self
    }
    #[must_use]
    pub fn output(mut self, port: PortDesc) -> Self {
        self.ports_out.push(port);
        self
    }
    #[must_use]
    pub fn parent(mut self, parent: &str) -> Self {
        self.parent = Some(parent.to_owned());
        self
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let ports = |p: &[PortDesc]| Value::Array(p.iter().map(PortDesc::to_value).collect());
        obj(vec![
            ("id", s(&self.id)),
            ("kind", self.kind.to_value()),
            ("parent", self.parent.as_deref().map_or(Value::Null, s)),
            ("ports_in", ports(&self.ports_in)),
            ("ports_out", ports(&self.ports_out)),
            ("config", self.config.clone()),
        ])
    }
    /// # Errors
    /// Missing field or unknown kind.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let ports = |key: &str| -> Result<Vec<PortDesc>, Fault> {
            v.get(key).and_then(Value::array).map_err(|e| bad(format!("{key}: {e}")))?.iter().map(PortDesc::from_value).collect()
        };
        let parent = match v.get("parent") {
            Ok(Value::String(p)) => Some(p.clone()),
            _ => None,
        };
        Ok(Self {
            id: text(v, "id")?,
            kind: ComponentType::from_value(v.get("kind").map_err(|e| bad(e.to_string()))?)?,
            parent,
            ports_in: ports("ports_in")?,
            ports_out: ports("ports_out")?,
            config: v.get("config").cloned().unwrap_or_else(|_| Value::Object(BTreeMap::new())),
        })
    }
}

/// One directed edge from an output port to an input port.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edge {
    pub from: ComponentId,
    pub from_port: String,
    pub to: ComponentId,
    pub to_port: String,
    pub mode: ConnectionMode,
}
impl Edge {
    #[must_use]
    pub fn new(from: &str, from_port: &str, to: &str, to_port: &str, mode: ConnectionMode) -> Self {
        Self { from: from.to_owned(), from_port: from_port.to_owned(), to: to.to_owned(), to_port: to_port.to_owned(), mode }
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("from", s(&self.from)),
            ("from_port", s(&self.from_port)),
            ("to", s(&self.to)),
            ("to_port", s(&self.to_port)),
            ("mode", self.mode.to_value()),
        ])
    }
    /// # Errors
    /// Missing field or unknown mode.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        Ok(Self {
            from: text(v, "from")?,
            from_port: text(v, "from_port")?,
            to: text(v, "to")?,
            to_port: text(v, "to_port")?,
            mode: ConnectionMode::from_value(v.get("mode").map_err(|e| bad(e.to_string()))?)?,
        })
    }
}

/// Nodes and edges, before validation. A snapshot is a validated graph.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}
impl Graph {
    #[must_use]
    pub fn node(&self, id: &str) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("nodes", Value::Array(self.nodes.iter().map(Node::to_value).collect())),
            ("edges", Value::Array(self.edges.iter().map(Edge::to_value).collect())),
        ])
    }
    /// Read the `designs/*.json` form (extra keys such as `version` and `note` are ignored).
    /// # Errors
    /// Malformed nodes or edges.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let list = |key: &str| v.get(key).and_then(Value::array).map_err(|e| bad(format!("{key}: {e}")));
        Ok(Self {
            nodes: list("nodes")?.iter().map(Node::from_value).collect::<Result<_, _>>()?,
            edges: list("edges")?.iter().map(Edge::from_value).collect::<Result<_, _>>()?,
        })
    }
}

/// Self-check of the `graph` sub-component: every mode and a small graph round-trip.
pub(crate) fn check() -> Result<(), String> {
    use nmtype::PortKind;
    let g = Graph {
        nodes: vec![
            Node::new("a#0", ComponentType::Other).output(PortDesc::new("out", PortKind::Json, true)),
            Node::new("b#0", ComponentType::Other).input(PortDesc::new("in", PortKind::Json, true)).parent("a#0"),
        ],
        edges: [
            ConnectionMode::Direct,
            ConnectionMode::Buffered(4),
            ConnectionMode::Asynchronous,
            ConnectionMode::Feedback { max_iter: 2 },
            ConnectionMode::Conditional("has_unknown".into()),
        ]
        .into_iter()
        .map(|m| Edge::new("a#0", "out", "b#0", "in", m))
        .collect(),
    };
    let back = Graph::from_value(&g.to_value()).map_err(|e| e.to_string())?;
    if back != g {
        return Err("nmnte.graph: round trip changed the graph".into());
    }
    Ok(())
}
