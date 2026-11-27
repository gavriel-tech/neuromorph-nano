//! Live anatomy for the organism (guide 11, 11.1, 11.2, 13, 14; PLAN 4c `nmapi::fault`):
//! every change runs the seven lifecycle phases with the organism's binder and is observed
//! by the Parameter Map; homeostasis reads each task's faults and quarantines a node that
//! faults twice in a row; `restart` binds a fresh component for one node; the composition
//! row is chosen per objective kind.
use crate::forage::Organism;
use nmbrain::homeostasis::Action;
use nmjson::Value;
use nmnte::lifecycle::{self, PhaseLog};
use nmnte::{Change, Components, Graph};
use nmtype::{Fault, FaultKind};
use std::path::Path;

fn bad(e: impl std::fmt::Display) -> Fault {
    Fault::new("nmapi.anatomy", FaultKind::Config, e.to_string())
}
fn read(path: &Path) -> Result<Value, Fault> {
    nmjson::parse(&std::fs::read(path).map_err(|e| bad(format!("{}: {e}", path.display())))?).map_err(bad)
}

/// Run one change through the lifecycle and record it in the Parameter Map.
/// # Errors
/// The failing phase's fault, with its log; the running snapshot is unchanged.
pub fn change(org: &Organism, change: &Change, run: &str) -> Result<PhaseLog, (Fault, PhaseLog)> {
    let sub = org.sub();
    let bind = |g: &Graph| -> Result<Components, Fault> { org.bind(g, &sub) };
    let result = lifecycle::replace(&org.nte, change, &bind, None, 2000);
    let (kind, detail) = match &result {
        Ok((snap, _)) => ("structure", format!("{} -> v{}", change.label(), snap.version)),
        Err((f, _)) => ("structure-refused", format!("{}: {f}", change.label())),
    };
    if let Ok(mut p) = org.parmap.lock() {
        let target = change.touched().join(",");
        let _ = p.observe("structure", if target.is_empty() { "edges" } else { &target }, kind, &detail, run);
    }
    result.map(|(_, log)| log)
}

/// Quarantine one node: from now on it is bound to an inert stand-in (a swap through the
/// lifecycle), and it stays so until `restart`.
/// # Errors
/// Unknown node or a refused change.
pub fn quarantine(org: &Organism, id: &str) -> Result<PhaseLog, Fault> {
    let node = org.nte.snapshot().graph.node(id).cloned().ok_or_else(|| bad(format!("no node {id}")))?;
    // The mark must be present while the swap binds (the binder reads it to bind the inert
    // stand-in); a refused change removes it again, so the old component is never left
    // marked (S5 review note).
    let fresh = org.quarantined.lock().is_ok_and(|mut q| q.insert(id.to_owned()));
    let out = change(org, &Change::Swap(node), "quarantine").map_err(|(f, _)| f);
    if out.is_err()
        && fresh
        && let Ok(mut q) = org.quarantined.lock()
    {
        q.remove(id);
    }
    out
}

/// Restart one node: lift its quarantine and bind a fresh component from its description,
/// on the current snapshot (a swap through the lifecycle). A fault drill marker in its
/// config is removed.
/// # Errors
/// Unknown node or a refused change.
pub fn restart(org: &Organism, id: &str) -> Result<PhaseLog, Fault> {
    let mut node = org.nte.snapshot().graph.node(id).cloned().ok_or_else(|| bad(format!("no node {id}")))?;
    if let Value::Object(cfg) = &mut node.config {
        cfg.remove("inject");
    }
    if let Ok(mut q) = org.quarantined.lock() {
        q.remove(id);
    }
    if let Ok(mut h) = org.homeostasis.lock() {
        h.reset(id);
    }
    change(org, &Change::Swap(node), "restart").map_err(|(f, _)| f)
}

/// After every task: homeostasis reads which nodes faulted and which ran, and a node at its
/// fault limit is quarantined at once.
pub fn after_task(org: &Organism, ran: &[String], faults: &[Fault], latency_ms: &std::collections::BTreeMap<String, f32>) {
    let faulted: Vec<String> = faults.iter().map(|f| f.component.clone()).collect();
    let clean: Vec<String> = ran.iter().filter(|r| !faulted.contains(r)).cloned().collect();
    let resident = org.resident_mb();
    let actions = org
        .homeostasis
        .lock()
        .map(|mut h| {
            // Roadmap Phase 4.6 (2026-11-24): the ceilings follow this machine (card total less the
            // live calls' measured room; three quarters of host memory), in place of the 3,500 MiB
            // and 24,000 MB written for one GTX 1650 and 32 GB machine.
            h.thresholds.gpu_mb_max = crate::forage::gpu_ceiling_mb();
            #[expect(clippy::cast_precision_loss, reason = "megabytes")]
            let ram = nmengn::capacity::ram_ceiling_mb() as f32;
            h.thresholds.ram_mb_max = ram;
            h.observe(&faulted, &clean, latency_ms, &resident)
        })
        .unwrap_or_default();
    for a in actions {
        match a {
            Action::Quarantine(id) => {
                let _ = quarantine(org, &id);
            }
            Action::Throttle(id) => org.throttle(&id),
            Action::Suspend(id) => org.suspend(&id),
        }
    }
}

/// The composition row for an objective (kind by rule, row from `designs/compositions.json`).
/// # Errors
/// An unreadable or malformed composition table.
pub fn composition_for(org: &Organism, objective: &str) -> Result<nmnte::compose::Row, Fault> {
    let table = read(&org.home.join("designs/compositions.json"))?;
    nmnte::compose::row(&table, nmbrain::objectives::classify(objective).name())
}

/// The nodes and edges of one more tentacle, from the template, as an `Add` change.
/// # Errors
/// An unreadable template.
pub fn tentacle_change(org: &Organism, n: usize) -> Result<Change, Fault> {
    let t = nmnte::compose::tentacle(&read(&org.home.join("designs/assemblies/tentacle.json"))?, n)?;
    Ok(Change::Add { nodes: t.nodes, edges: t.edges })
}

/// Compose the forage assembly for an objective kind and write it as a designs file.
/// # Errors
/// Unreadable designs or an unwritable output.
pub fn compose_file(home: &Path, kind: &str) -> Result<std::path::PathBuf, Fault> {
    let core = Graph::from_value(&read(&home.join("designs/assemblies/core.json"))?)?;
    let (g, row) = nmnte::compose::compose(
        kind,
        &read(&home.join("designs/compositions.json"))?,
        &core,
        &read(&home.join("designs/assemblies/tentacle.json"))?,
    )?;
    let mut v = g.to_value();
    if let Value::Object(m) = &mut v {
        m.insert("note".into(), Value::String(format!("Composed by nmnte::compose for objective kind {} ({} tentacles) from core.json, tentacle.json and compositions.json. Regenerate with `nano-lab compose {kind}`; do not edit by hand.", row.objective_kind, row.tentacles)));
        m.insert("version".into(), Value::Number("1".into()));
    }
    let path = home.join(format!("designs/assemblies/compose-{}.json", row.objective_kind));
    std::fs::write(&path, nmjson::pretty::pretty(&v)).map_err(bad)?;
    Ok(path)
}

/// Context Zoom on the anatomy axis (PLAN 4c `zoom_anatomy`, item 47; 2026-11-10): the
/// Neuromorph itself at one of five levels, finest first: `cluster` (one parameter cluster: its
/// tensor, region, bytes and the neural components that bind it), `component` (one anatomy node
/// or one neural component of a bound assembly, with its clusters), `structure` (the anatomy
/// graph by kind and each bound assembly by operation), `slot` (each Core Brain slot: control,
/// engine, qualification, residency), `organism` (counts). `id` names the subject where a level
/// needs one. Reads the running snapshot and the bound engines only.
#[must_use]
pub fn zoom(org: &Organism, level: &str, id: Option<&str>) -> Value {
    use nmjson::Value as V;
    let s = |t: &str| V::String(t.to_owned());
    let n = |x: usize| V::Number(x.to_string());
    let o = |v: Vec<(&str, V)>| V::Object(v.into_iter().map(|(k, x)| (k.to_owned(), x)).collect());
    let snap = org.nte.snapshot();
    let models = crate::neural::bound_models();
    match level {
        "organism" => {
            let mut kinds: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
            for x in snap.nodes() {
                *kinds.entry(x.kind.name().to_owned()).or_default() += 1;
            }
            o(vec![
                ("level", s("organism")),
                ("anatomy_version", n(usize::try_from(snap.version).unwrap_or(0))),
                ("nodes_by_kind", V::Object(kinds.into_iter().map(|(k, c)| (k, n(c))).collect())),
                ("edges", n(snap.edges().len())),
                ("slots", n(org.slots.by_name.len())),
                ("engines_bound", V::Array(org.slots.by_name.values().filter(|x| x.engine().is_some()).map(|x| s(&x.name)).collect())),
                (
                    "populations_bound",
                    V::Array(
                        models.iter().map(|(r, m)| s(&format!("{r}: {} ({} clusters)", m.pop.card.id, m.pop.clusters.len()))).collect(),
                    ),
                ),
                ("zoom_in", s("slot")),
            ])
        }
        "slot" => o(vec![
            ("level", s("slot")),
            (
                "slots",
                V::Array(
                    org.slots
                        .by_name
                        .values()
                        .map(|x| {
                            let rec = x.record();
                            o(vec![
                                ("slot", s(&x.name)),
                                ("control", s("exact")),
                                ("engine", x.engine().map_or(V::Null, |e| s(&e.label()))),
                                ("qualified", V::Bool(org.slots.qualified(&x.name))),
                                ("control_calls", n(usize::try_from(rec.control_calls).unwrap_or(0))),
                                ("engine_calls", n(usize::try_from(rec.engine_calls).unwrap_or(0))),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("zoom_out", s("organism")),
            ("zoom_in", s("structure")),
        ]),
        "structure" => {
            let mut kinds: std::collections::BTreeMap<String, Vec<String>> = std::collections::BTreeMap::new();
            for x in snap.nodes() {
                kinds.entry(x.kind.name().to_owned()).or_default().push(x.id.clone());
            }
            let asm = models
                .iter()
                .map(|(r, m)| {
                    let mut ops: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
                    for x in &m.asm.nodes {
                        *ops.entry(x.op.clone()).or_default() += 1;
                    }
                    o(vec![
                        ("role", s(r)),
                        ("assembly", s(&m.asm.name)),
                        ("population", s(&m.pop.card.id)),
                        ("nodes_by_operation", V::Object(ops.into_iter().map(|(k, c)| (k, n(c))).collect())),
                    ])
                })
                .collect();
            o(vec![
                ("level", s("structure")),
                ("anatomy", V::Object(kinds.into_iter().map(|(k, ids)| (k, V::Array(ids.iter().map(|i| s(i)).collect()))).collect())),
                ("assemblies", V::Array(asm)),
                ("zoom_out", s("slot")),
                ("zoom_in", s("component (give id)")),
            ])
        }
        "component" => {
            let Some(id) = id else {
                return o(vec![("level", s("component")), ("error", s("give the id of an anatomy node or of a neural component"))]);
            };
            if let Some(x) = snap.nodes().iter().find(|x| x.id == id) {
                let edges: Vec<V> =
                    snap.edges().iter().filter(|e| e.from == id || e.to == id).map(|e| s(&format!("{} -> {}", e.from, e.to))).collect();
                return o(vec![
                    ("level", s("component")),
                    ("kind", s(x.kind.name())),
                    ("id", s(id)),
                    ("parent", x.parent.as_deref().map_or(V::Null, s)),
                    ("edges", V::Array(edges)),
                    ("zoom_out", s("structure")),
                ]);
            }
            for (r, m) in &models {
                if let Some(x) = m.asm.nodes.iter().find(|x| x.id == id) {
                    return o(vec![
                        ("level", s("component")),
                        ("kind", s("neural")),
                        ("role", s(r)),
                        ("id", s(id)),
                        ("operation", s(&x.op)),
                        ("layer", x.layer.map_or(V::Null, n)),
                        ("precision", s(&x.precision)),
                        ("clusters", V::Array(x.clusters.iter().map(|c| s(c)).collect())),
                        ("zoom_out", s("structure")),
                        ("zoom_in", s("cluster (give a cluster id)")),
                    ]);
                }
            }
            o(vec![("level", s("component")), ("id", s(id)), ("error", s("no such component in the snapshot or the bound assemblies"))])
        }
        "cluster" => {
            let Some(id) = id else { return o(vec![("level", s("cluster")), ("error", s("give a cluster id"))]) };
            for (r, m) in &models {
                if let Some(c) = m.pop.clusters.iter().find(|c| c.id == id) {
                    let users: Vec<V> = m
                        .asm
                        .nodes
                        .iter()
                        .filter(|x| x.clusters.iter().any(|y| format!("{}/{y}", m.pop.card.id) == id))
                        .map(|x| s(&x.id))
                        .collect();
                    return o(vec![
                        ("level", s("cluster")),
                        ("id", s(id)),
                        ("role", s(r)),
                        ("population", s(&m.pop.card.id)),
                        ("tensor", s(&c.tensor)),
                        ("group", s(&c.group)),
                        ("dtype", s(&c.dtype)),
                        ("bytes", n(usize::try_from(c.bytes).unwrap_or(0))),
                        ("bound_by", V::Array(users)),
                        ("zoom_out", s("component")),
                    ]);
                }
            }
            o(vec![
                ("level", s("cluster")),
                ("id", s(id)),
                ("error", s("no such cluster in a bound population (bind engines with serve --neural)")),
            ])
        }
        other => o(vec![("error", s(&format!("unknown level {other}: cluster, component, structure, slot or organism")))]),
    }
}
