//! SET ASIDE 2026-11-10 (direction of Luis Jake Gabriel III, 08:00 UTC: the weights are not placed
//! in the tentacles; one copy is held centrally and used by reference). No tentacle assembly is
//! generated and no read goes through this module; compiled and kept for reference, not deleted.
//!
//! A tentacle's own neural body (2026-11-10; direction of Luis Jake Gabriel III: the weights sit
//! inside the tentacles and drones, not in one model beside them; design
//! `docs/2026-11-10-neumor-design-tentacle-core-v1.md` sections 1.3, 1.4 and 4.1, addition A10).
//!
//! `assembly` generates, from the released layer order (the reference assembly, used only as the
//! catalogue of components and clusters) and one tentacle's drone set, an assembly that holds only
//! what that tentacle uses:
//! - its sensory segments `<t>.S<l>.*` for layers 0 to L_d - 1 (attention norm, query, key and value
//!   projections over their head clusters, head norms, rotary component, scoring, output projection,
//!   residual, feed-forward norm, gate, up, SwiGLU, down, residual) and the attention norm of layer
//!   L_d, where L_d is the deepest layer its drone set reads; the embedding `<t>.embed`;
//! - its drone tip `<t>.tip.L<l>.g<g>.*`: for every drone layer and every key-value group a drone
//!   head uses, the query rows of that group's query heads, the key and value rows of the group,
//!   the head norms, a rotary component and a scoring component over exactly that group. The tip
//!   owns the drone heads' scoring; a read records the tip's components only
//!   (`Probe::with_node_filter(TIP)`), so the sensory scoring on the same layer is not read twice.
//! - nothing else: no layer above L_d, no feed-forward of L_d, no final norm, no output head.
//!
//! Along every path the components run in the released order (no layer is reordered or dropped
//! below a component that reads it), so the drone readings equal the reference path's on the same
//! window bit for bit (section check `nmapi.tentacle_body`); the restructuring is which components
//! the tentacle holds and owns, not the arithmetic. Binding the assembly over the shared population
//! provisions every cluster by reference (`Population::prov`), and device copies are keyed by
//! cluster, so the sensory segments share the reference path's device weights; only the tip's
//! row subsets are copies of their own (small).
use crate::assembly::{Assembly, Edge, Node, nid};
use nmtype::{Fault, FaultKind};
use std::collections::{BTreeMap, BTreeSet};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmengn.tentacle", FaultKind::Execution, m)
}

/// The text that marks a drone tip component's id.
pub const TIP: &str = ".tip.";

/// Generate tentacle `t`'s assembly from `reference` (the released layer order) for its drone
/// `heads` ((layer, global head) pairs), with `q_heads` query heads and `kv_heads` key-value heads
/// per layer.
/// # Errors
/// No heads, a layer the reference lacks, or a reference component whose head clusters do not
/// match the head counts.
pub fn assembly(reference: &Assembly, t: &str, heads: &[(usize, usize)], q_heads: usize, kv_heads: usize) -> Result<Assembly, Fault> {
    let ld = heads.iter().map(|p| p.0).max().ok_or_else(|| bad(format!("tentacle {t}: no drone heads")))?;
    let group = (q_heads / kv_heads.max(1)).max(1);
    let inst = |id: &str| id.strip_prefix("nmengn.components#").unwrap_or(id).to_owned();
    // Sensory components and their new names.
    let mut rename: BTreeMap<String, String> = BTreeMap::new();
    let mut nodes: Vec<Node> = Vec::new();
    for n in &reference.nodes {
        let i = inst(&n.id);
        let new = match (n.layer, i.as_str()) {
            (None, "ids" | "pos") => n.id.clone(),
            (None, "embed") => nid(&format!("{t}.embed")),
            (None, _) => continue,
            (Some(l), _) if l < ld => nid(&format!("{t}.S{l}.{}", i.split_once('.').map_or(i.as_str(), |x| x.1))),
            (Some(l), _) if l == ld && i.ends_with(".attn_norm") => nid(&format!("{t}.S{l}.attn_norm")),
            _ => continue,
        };
        rename.insert(n.id.clone(), new.clone());
        nodes.push(Node { id: new, ..n.clone() });
    }
    if !rename.values().any(|v| v.ends_with(&format!("{t}.S{ld}.attn_norm"))) {
        return Err(bad(format!("tentacle {t}: the reference has no attention norm on layer {ld}")));
    }
    let mut edges: Vec<Edge> = reference
        .edges
        .iter()
        .filter_map(|e| {
            Some(Edge {
                from: rename.get(&e.from)?.clone(),
                from_port: e.from_port.clone(),
                to: rename.get(&e.to)?.clone(),
                to_port: e.to_port.clone(),
            })
        })
        .collect();
    // The drone tip.
    let mut by_layer: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for &(l, h) in heads {
        if h >= q_heads {
            return Err(bad(format!("tentacle {t}: head {h} past the {q_heads} query heads")));
        }
        by_layer.entry(l).or_default().insert(h / group);
    }
    let find = |l: usize, suffix: &str| {
        reference
            .nodes
            .iter()
            .find(|n| n.layer == Some(l) && inst(&n.id) == format!("L{l}.{suffix}"))
            .ok_or_else(|| bad(format!("tentacle {t}: the reference has no L{l}.{suffix}")))
    };
    for (l, groups) in by_layer {
        let (q, k, v, qn, kn) = (find(l, "q")?, find(l, "k")?, find(l, "v")?, find(l, "qnorm")?, find(l, "knorm")?);
        if q.clusters.len() != q_heads || k.clusters.len() != kv_heads || v.clusters.len() != kv_heads {
            return Err(bad(format!(
                "tentacle {t}: layer {l} projections hold {}, {} and {} clusters for {q_heads} query and {kv_heads} key-value heads",
                q.clusters.len(),
                k.clusters.len(),
                v.clusters.len()
            )));
        }
        let norm = nid(&format!("{t}.S{l}.attn_norm"));
        for g in groups {
            let p = |s: &str| nid(&format!("{t}.tip.L{l}.g{g}.{s}"));
            let rows = |src: &Node, a: usize, b: usize| src.clusters.get(a..b).map(<[String]>::to_vec).unwrap_or_default();
            nodes.push(Node { id: p("q"), clusters: rows(q, g * group, (g + 1) * group), ..q.clone() });
            nodes.push(Node { id: p("k"), clusters: rows(k, g, g + 1), ..k.clone() });
            nodes.push(Node { id: p("v"), clusters: rows(v, g, g + 1), ..v.clone() });
            nodes.push(Node { id: p("qnorm"), ..qn.clone() });
            nodes.push(Node { id: p("knorm"), ..kn.clone() });
            nodes.push(Node {
                id: p("rope"),
                op: "rope".into(),
                kind: "attention".into(),
                layer: Some(l),
                clusters: Vec::new(),
                heads: None,
                kv: None,
                stack: "rows".into(),
                precision: "f16".into(),
                labels: Vec::new(),
            });
            nodes.push(Node {
                id: p("score"),
                op: "score".into(),
                kind: "attention".into(),
                layer: Some(l),
                clusters: Vec::new(),
                heads: Some((g * group, (g + 1) * group)),
                kv: Some((g, g + 1)),
                stack: "rows".into(),
                precision: "f16".into(),
                labels: Vec::new(),
            });
            for (from, fp, to, tp) in [
                (norm.clone(), "y", p("q"), "x"),
                (norm.clone(), "y", p("k"), "x"),
                (norm.clone(), "y", p("v"), "x"),
                (p("q"), "y", p("qnorm"), "x"),
                (p("k"), "y", p("knorm"), "x"),
                (p("qnorm"), "y", p("rope"), "q"),
                (p("knorm"), "y", p("rope"), "k"),
                (nid("pos"), "y", p("rope"), "pos"),
                (p("rope"), "q", p("score"), "q"),
                (p("rope"), "k", p("score"), "k"),
                (p("v"), "y", p("score"), "v"),
            ] {
                edges.push(Edge { from, from_port: fp.into(), to, to_port: tp.into() });
            }
        }
    }
    let a = Assembly { name: format!("tentacle-{t}"), nodes, edges, names: BTreeMap::new() };
    a.check_ports()?;
    Ok(a)
}

/// The run targets of a tentacle's read: every drone tip scoring component, in assembly order.
#[must_use]
pub fn tip_targets(a: &Assembly) -> Vec<(String, String)> {
    a.nodes.iter().filter(|n| n.op == "score" && n.id.contains(TIP)).map(|n| (n.id.clone(), "y".to_owned())).collect()
}

/// Section check (processor only, no weights): the generator on the reference assembly for the
/// default drone set: no layer above L_d, no output head or final norm, the tip covers exactly the
/// drone heads' key-value groups, the port check passes, and a head past the model is refused.
pub(crate) fn check() -> Result<(), String> {
    let home = nmtype::nano_root();
    let r = Assembly::load(&home.join("designs/assemblies"), "qwen3-reference").map_err(|e| e.to_string())?;
    let set = crate::kvstore::drone_set(&home, crate::kvstore::DRONE_FALLBACK_POPULATION).map_err(|e| e.to_string())?;
    let a = assembly(&r, "meaning", &set.heads, 16, 8).map_err(|e| e.to_string())?;
    let ld = set.last_layer();
    if a.nodes.iter().any(|n| n.layer.is_some_and(|l| l > ld) || n.op == "head" || n.id.ends_with("final_norm")) {
        return Err("a tentacle assembly holds a layer above its last drone layer, a final norm or a head".into());
    }
    let tips: BTreeSet<(usize, usize)> =
        a.nodes.iter().filter(|n| n.op == "score" && n.id.contains(TIP)).filter_map(|n| Some((n.layer?, n.kv?.0))).collect();
    let want: BTreeSet<(usize, usize)> = set.heads.iter().map(|&(l, h)| (l, h / 2)).collect();
    if tips != want {
        return Err(format!("tip groups {tips:?}, want {want:?}"));
    }
    if assembly(&r, "x", &[(3, 99)], 16, 8).is_ok() {
        return Err("a head past the model was accepted".into());
    }
    Ok(())
}
