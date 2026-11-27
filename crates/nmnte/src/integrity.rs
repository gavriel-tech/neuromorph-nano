//! Integrity and Compatibility Validator (guide 11.1 item 7, 14, 14.1): the anatomy Step
//! Check every snapshot passes before it is published. Rules, in order: unique ids, parents
//! exist, edges name real nodes and ports, port kinds compatible, mode parameters valid, no
//! edge bypasses the Core Brain (no tentacle-to-tentacle link), cycles only through feedback
//! edges, every node resolves in the registry. The first broken rule is the fault.
use crate::graph::{ConnectionMode, Graph};
use crate::registry::Registry;
use crate::topology;
use nmtype::{Fault, FaultKind, compatible};
use std::collections::BTreeSet;

fn broken(detail: impl Into<String>) -> Fault {
    Fault::new("nmnte.integrity", FaultKind::Integrity, detail)
}

/// Validate a graph. `registry` is optional so a bare graph can be checked in isolation.
/// # Errors
/// The first rule the graph breaks.
pub fn validate(g: &Graph, registry: Option<&Registry>) -> Result<(), Fault> {
    let mut ids = BTreeSet::new();
    for n in &g.nodes {
        if !ids.insert(n.id.as_str()) {
            return Err(broken(format!("duplicate node id {}", n.id)));
        }
    }
    for n in &g.nodes {
        if let Some(p) = &n.parent
            && !ids.contains(p.as_str())
        {
            return Err(broken(format!("{}: parent {p} is not in the graph", n.id)));
        }
    }
    for e in &g.edges {
        let at = format!("edge {}.{} -> {}.{}", e.from, e.from_port, e.to, e.to_port);
        let (Some(a), Some(b)) = (g.node(&e.from), g.node(&e.to)) else {
            return Err(broken(format!("{at}: names a node that is not in the graph")));
        };
        let out = a.ports_out.iter().find(|p| p.name == e.from_port);
        let inp = b.ports_in.iter().find(|p| p.name == e.to_port);
        let (Some(out), Some(inp)) = (out, inp) else {
            return Err(broken(format!("{at}: names a port the node does not declare")));
        };
        if !compatible(out, inp) {
            return Err(broken(format!("{at}: {} does not fit {}", out.kind.name(), inp.kind.name())));
        }
        match &e.mode {
            ConnectionMode::Buffered(0) => return Err(broken(format!("{at}: buffered bound must be at least 1"))),
            ConnectionMode::Feedback { max_iter: 0 } => {
                return Err(broken(format!("{at}: feedback edge needs max_iter of at least 1")));
            }
            ConnectionMode::Conditional(p) if p.trim().is_empty() => {
                return Err(broken(format!("{at}: conditional edge without a predicate")));
            }
            _ => {}
        }
        if topology::bypasses_brain(g, &e.from, &e.to) {
            return Err(broken(format!("{at}: bypasses the Core Brain (no tentacle-to-tentacle links)")));
        }
    }
    crate::scheduler::plan(g).map(|_| ())?;
    if let Some(reg) = registry {
        for n in &g.nodes {
            reg.resolve_node(&n.id)?;
        }
    }
    Ok(())
}

/// Self-check of the `integrity` sub-component: a lawful anatomy passes; a tentacle-to-
/// tentacle edge, a port mismatch, a zero-iteration feedback edge and an ordering cycle are
/// each refused.
pub(crate) fn check() -> Result<(), String> {
    use crate::graph::{Edge, Node};
    use nmtype::{ComponentType, PortDesc, PortKind};
    let p = |k| PortDesc::new("p", k, true);
    let node = |id: &str, kind| Node::new(id, kind).input(p(PortKind::Packets)).output(p(PortKind::Packets));
    let base = Graph {
        nodes: vec![
            node("b#0", ComponentType::Brain),
            node("t#0", ComponentType::Tentacle),
            node("t#1", ComponentType::Tentacle),
            Node::new("x#0", ComponentType::Other).input(p(PortKind::Tensor)),
        ],
        edges: vec![
            Edge::new("b#0", "p", "t#0", "p", ConnectionMode::Direct),
            Edge::new("t#0", "p", "b#0", "p", ConnectionMode::Feedback { max_iter: 1 }),
        ],
    };
    validate(&base, None).map_err(|e| format!("nmnte.integrity: lawful graph refused: {e}"))?;
    let bad_edges = [
        Edge::new("t#0", "p", "t#1", "p", ConnectionMode::Direct),
        Edge::new("b#0", "p", "x#0", "p", ConnectionMode::Direct),
        Edge::new("b#0", "p", "t#1", "p", ConnectionMode::Feedback { max_iter: 0 }),
        Edge::new("t#0", "p", "b#0", "p", ConnectionMode::Direct),
    ];
    for e in bad_edges {
        let mut g = base.clone();
        let label = format!("{} -> {} {:?}", e.from, e.to, e.mode);
        g.edges.push(e);
        if validate(&g, None).is_ok() {
            return Err(format!("nmnte.integrity: accepted {label}"));
        }
    }
    Ok(())
}
