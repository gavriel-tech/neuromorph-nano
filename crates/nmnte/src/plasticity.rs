//! Structural Plasticity Engine (guide 11, 11.1 item 6, 10.3, 14; PLAN 4c, 4d item 12):
//! anatomy changes as data. `Add` mounts nodes and their edges, `Remove` takes a node out
//! with its descendants and edges, `Swap` replaces one node's description (same id; a fresh
//! component is bound), `Rewire` changes the communication topology. A change is applied to
//! a copy of the current graph; whether it may run is the lifecycle's Validate phase.
use crate::graph::{Edge, Graph, Node};
use crate::topology;
use nmtype::{Fault, FaultKind};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Change {
    Add { nodes: Vec<Node>, edges: Vec<Edge> },
    Remove(String),
    Swap(Node),
    Rewire { add: Vec<Edge>, remove: Vec<Edge> },
}
impl Change {
    /// A short label for logs and the Parameter Map.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::Add { nodes, edges } => format!("add {} nodes, {} edges", nodes.len(), edges.len()),
            Self::Remove(id) => format!("remove {id}"),
            Self::Swap(n) => format!("swap {}", n.id),
            Self::Rewire { add, remove } => format!("rewire +{} -{}", add.len(), remove.len()),
        }
    }
    /// Node ids whose component must be (re)bound after this change, in the graph it applies
    /// to: a swapped node takes its descendants with it (a tentacle's nodes share one unit).
    #[must_use]
    pub fn touched_in(&self, g: &Graph) -> Vec<String> {
        match self {
            Self::Swap(n) => std::iter::once(n.id.clone())
                .chain(g.nodes.iter().filter(|x| topology::ancestors(g, &x.id).contains(&n.id)).map(|x| x.id.clone()))
                .collect(),
            _ => self.touched(),
        }
    }
    /// Node ids whose component must be (re)bound after this change (without the graph).
    #[must_use]
    pub fn touched(&self) -> Vec<String> {
        match self {
            Self::Add { nodes, .. } => nodes.iter().map(|n| n.id.clone()).collect(),
            Self::Swap(n) => vec![n.id.clone()],
            Self::Remove(_) | Self::Rewire { .. } => Vec::new(),
        }
    }
}

/// The graph after a change (the input graph is untouched).
/// # Errors
/// Adding an id that exists, or removing or swapping one that does not.
pub fn apply_to(g: &Graph, change: &Change) -> Result<Graph, Fault> {
    let bad = |e: String| Fault::new("nmnte.plasticity", FaultKind::Config, e);
    let mut next = g.clone();
    match change {
        Change::Add { nodes, edges } => {
            if let Some(n) = nodes.iter().find(|n| g.node(&n.id).is_some()) {
                return Err(bad(format!("{} is already mounted", n.id)));
            }
            next.nodes.extend(nodes.iter().cloned());
            next.edges.extend(edges.iter().cloned());
        }
        Change::Remove(id) => {
            if g.node(id).is_none() {
                return Err(bad(format!("no node {id}")));
            }
            let mut gone: Vec<String> = vec![id.clone()];
            gone.extend(g.nodes.iter().filter(|n| topology::ancestors(g, &n.id).contains(id)).map(|n| n.id.clone()));
            next.nodes.retain(|n| !gone.contains(&n.id));
            next.edges.retain(|e| !gone.contains(&e.from) && !gone.contains(&e.to));
        }
        Change::Swap(n) => {
            let slot = next.nodes.iter_mut().find(|x| x.id == n.id).ok_or_else(|| bad(format!("no node {}", n.id)))?;
            *slot = n.clone();
        }
        Change::Rewire { add, remove } => {
            next.edges.retain(|e| !remove.contains(e));
            next.edges.extend(add.iter().cloned());
        }
    }
    Ok(next)
}

/// Self-check of the `plasticity` sub-component: add, swap, rewire and remove (with
/// descendants) each change only what they name.
pub(crate) fn check() -> Result<(), String> {
    use crate::graph::ConnectionMode;
    use nmtype::ComponentType;
    let base = Graph {
        nodes: vec![Node::new("t#0", ComponentType::Tentacle), Node::new("d#0", ComponentType::Drone).parent("t#0")],
        edges: vec![Edge::new("t#0", "o", "d#0", "i", ConnectionMode::Direct)],
    };
    let e = |f: Fault| f.to_string();
    let add = apply_to(&base, &Change::Add { nodes: vec![Node::new("t#1", ComponentType::Tentacle)], edges: vec![] }).map_err(e)?;
    let swap = apply_to(
        &add,
        &Change::Swap(Node::new("d#0", ComponentType::Drone).parent("t#0").input(nmtype::PortDesc::new("i", nmtype::PortKind::Json, true))),
    )
    .map_err(e)?;
    let rewire = apply_to(
        &swap,
        &Change::Rewire { add: vec![Edge::new("t#0", "o", "d#0", "i", ConnectionMode::Buffered(4))], remove: base.edges.clone() },
    )
    .map_err(e)?;
    let removed = apply_to(&rewire, &Change::Remove("t#0".into())).map_err(e)?;
    let ok = add.nodes.len() == 3
        && swap.node("d#0").is_some_and(|n| n.ports_in.len() == 1)
        && rewire.edges.first().is_some_and(|x| x.mode == ConnectionMode::Buffered(4))
        && removed.nodes.iter().map(|n| n.id.as_str()).collect::<Vec<_>>() == ["t#1"]
        && removed.edges.is_empty();
    let swap_t = Change::Swap(Node::new("t#0", ComponentType::Tentacle)).touched_in(&base);
    if !ok || swap_t != ["t#0", "d#0"] || apply_to(&base, &Change::Remove("zz".into())).is_ok() {
        return Err("nmnte.plasticity: a change touched more or less than it names".into());
    }
    Ok(())
}
