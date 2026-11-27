//! Topology Manager queries over a graph (guide 11.1 item 3, 14.1): upstream, downstream,
//! paths, and the two anatomy relations the no-bypass law needs: which unit (tentacle or
//! drone tip) a node belongs to, and whether a node sits inside the Core Brain.
use crate::graph::Graph;
use nmtype::ComponentType;
use std::collections::BTreeSet;

/// Direct successors of `id` over every edge.
#[must_use]
pub fn downstream<'a>(g: &'a Graph, id: &str) -> Vec<&'a str> {
    g.edges.iter().filter(|e| e.from == id).map(|e| e.to.as_str()).collect()
}
/// Direct predecessors of `id` over every edge.
#[must_use]
pub fn upstream<'a>(g: &'a Graph, id: &str) -> Vec<&'a str> {
    g.edges.iter().filter(|e| e.to == id).map(|e| e.from.as_str()).collect()
}
/// Every node reachable from `id` over ordering edges (feedback and asynchronous excluded).
#[must_use]
pub fn reachable(g: &Graph, id: &str) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    let mut stack = vec![id.to_owned()];
    while let Some(n) = stack.pop() {
        for e in g.edges.iter().filter(|e| e.from == n && e.mode.orders()) {
            if seen.insert(e.to.clone()) {
                stack.push(e.to.clone());
            }
        }
    }
    seen
}
/// Is there an ordering path from `a` to `b`?
#[must_use]
pub fn path_exists(g: &Graph, a: &str, b: &str) -> bool {
    reachable(g, a).contains(b)
}
/// The ancestors of `id`, nearest first (stops on a missing parent or a parent loop).
#[must_use]
pub fn ancestors(g: &Graph, id: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = g.node(id).and_then(|n| n.parent.clone());
    while let Some(p) = cur {
        if out.contains(&p) || p == id {
            break;
        }
        cur = g.node(&p).and_then(|n| n.parent.clone());
        out.push(p);
    }
    out
}
/// The tentacle (or, when none above it, the drone tip) that `id` belongs to, itself
/// included. `None` for nodes outside the Neuropod's units.
#[must_use]
pub fn unit_of(g: &Graph, id: &str) -> Option<String> {
    let chain = std::iter::once(id.to_owned()).chain(ancestors(g, id));
    let mut drone = None;
    for n in chain {
        match g.node(&n).map(|x| x.kind) {
            Some(ComponentType::Tentacle) => return Some(n),
            Some(ComponentType::Drone) if drone.is_none() => drone = Some(n),
            _ => {}
        }
    }
    drone
}
/// Does `id` sit inside the Core Brain (itself or an ancestor of kind brain)?
#[must_use]
pub fn in_brain(g: &Graph, id: &str) -> bool {
    std::iter::once(id.to_owned()).chain(ancestors(g, id)).any(|n| g.node(&n).is_some_and(|x| x.kind == ComponentType::Brain))
}
/// Does the edge `from -> to` bypass the Core Brain? It does when either end belongs to a
/// unit, the two ends are not inside the same unit, and the other end is not in the brain.
#[must_use]
pub fn bypasses_brain(g: &Graph, from: &str, to: &str) -> bool {
    let (ua, ub) = (unit_of(g, from), unit_of(g, to));
    match (&ua, &ub) {
        (None, None) => false,
        (Some(a), Some(b)) if a == b => false,
        (Some(_), None) => !in_brain(g, to),
        (None, Some(_)) => !in_brain(g, from),
        (Some(_), Some(_)) => true,
    }
}

/// Self-check of the `topology` sub-component: reachability, unit membership and the
/// bypass relation on a four-node anatomy.
pub(crate) fn check() -> Result<(), String> {
    use crate::graph::{ConnectionMode, Edge, Node};
    let g = Graph {
        nodes: vec![
            Node::new("b#0", ComponentType::Brain),
            Node::new("t#0", ComponentType::Tentacle),
            Node::new("d#0", ComponentType::Drone).parent("t#0"),
            Node::new("t#1", ComponentType::Tentacle),
            Node::new("m#0", ComponentType::Memory),
        ],
        edges: vec![Edge::new("b#0", "o", "t#0", "i", ConnectionMode::Direct), Edge::new("t#0", "o", "d#0", "i", ConnectionMode::Direct)],
    };
    if !path_exists(&g, "b#0", "d#0") || path_exists(&g, "d#0", "b#0") {
        return Err("nmnte.topology: reachability".into());
    }
    if unit_of(&g, "d#0").as_deref() != Some("t#0") {
        return Err("nmnte.topology: unit_of".into());
    }
    let cases = [
        ("b#0", "t#0", false),
        ("t#0", "d#0", false),
        ("t#0", "t#1", true),
        ("d#0", "t#1", true),
        ("m#0", "t#0", true),
        ("b#0", "m#0", false),
    ];
    for (a, b, want) in cases {
        if bypasses_brain(&g, a, b) != want {
            return Err(format!("nmnte.topology: bypass {a} -> {b} should be {want}"));
        }
    }
    Ok(())
}
