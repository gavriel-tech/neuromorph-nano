//! Lifecycle Manager (guide 11.1 item 5, 14; PLAN 4c, from `fwcore/lifecy`): every anatomy
//! change runs seven phases, each logged: Prepare (the next graph from the change), Validate
//! (the Integrity Validator), Initialize (components for the touched nodes; untouched nodes
//! keep their running components and state), Compare (old and new on a fixture, when one is
//! given), Commit (atomic publish), Drain (wait, bounded, for frames on the old snapshot) and
//! Retire (the old snapshot is reclaimed once no frame needs it). A fault in any phase before
//! Commit leaves the running snapshot as it was.
use crate::graph::Graph;
use crate::plasticity::{self, Change};
use crate::snapshot::{Components, Nte, Snapshot};
use nmtype::Fault;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Prepare,
    Validate,
    Initialize,
    Compare,
    Commit,
    Drain,
    Retire,
}

/// The phases a change went through, with the time each took and what it found.
#[derive(Clone, Debug, Default)]
pub struct PhaseLog {
    pub change: String,
    pub entries: Vec<(Phase, u128, String)>,
    pub committed: Option<u64>,
}
impl PhaseLog {
    fn add(&mut self, p: Phase, t: Instant, detail: impl Into<String>) {
        self.entries.push((p, t.elapsed().as_micros(), detail.into()));
    }
    /// One line per phase, for reports.
    #[must_use]
    pub fn lines(&self) -> Vec<String> {
        self.entries.iter().map(|(p, us, d)| format!("{p:?} ({us} us): {d}")).collect()
    }
}

/// Binds components for a graph (the organism's binder).
pub type Binder<'a> = &'a dyn Fn(&Graph) -> Result<Components, Fault>;
/// Compares the old snapshot with the new components on a fixture and says what it saw.
pub type Comparer<'a> = &'a dyn Fn(&Snapshot, &Components) -> Result<String, Fault>;

/// Run one change through the seven phases.
/// # Errors
/// The fault of the failing phase, with the log up to it; nothing was published.
pub fn replace(
    nte: &Nte,
    change: &Change,
    bind: Binder<'_>,
    compare: Option<Comparer<'_>>,
    drain_ms: u64,
) -> Result<(Arc<Snapshot>, PhaseLog), (Fault, PhaseLog)> {
    let mut log = PhaseLog { change: change.label(), ..PhaseLog::default() };
    let t = Instant::now();
    let old = nte.snapshot();
    let next = match plasticity::apply_to(&old.graph, change) {
        Ok(g) => g,
        Err(f) => return Err((f, log)),
    };
    log.add(Phase::Prepare, t, format!("{} on v{}: {} nodes, {} edges", change.label(), old.version, next.nodes.len(), next.edges.len()));
    let t = Instant::now();
    if let Err(f) = nte.validate(&next) {
        log.add(Phase::Validate, t, format!("refused: {f}"));
        return Err((f, log));
    }
    log.add(Phase::Validate, t, "integrity and compatibility hold");
    let t = Instant::now();
    let fresh = match bind(&next) {
        Ok(c) => c,
        Err(f) => return Err((f, log)),
    };
    let touched = change.touched_in(&old.graph);
    let mut comps: Components = Components::new();
    let mut kept = 0;
    for (id, c) in fresh {
        match old.components.get(&id) {
            Some(running) if !touched.contains(&id) => {
                comps.insert(id, Arc::clone(running));
                kept += 1;
            }
            _ => {
                comps.insert(id, c);
            }
        }
    }
    log.add(Phase::Initialize, t, format!("{} new components, {kept} running components kept", comps.len() - kept));
    let t = Instant::now();
    match compare {
        Some(cmp) => match cmp(&old, &comps) {
            Ok(seen) => log.add(Phase::Compare, t, seen),
            Err(f) => {
                log.add(Phase::Compare, t, format!("refused: {f}"));
                return Err((f, log));
            }
        },
        None => log.add(Phase::Compare, t, "no fixture listed for this change"),
    }
    let t = Instant::now();
    let snap = match nte.publish_from(old.version, next, comps) {
        Ok(s) => s,
        Err(f) => {
            log.add(Phase::Commit, t, format!("refused: {f}"));
            return Err((f, log));
        }
    };
    log.committed = Some(snap.version);
    log.add(Phase::Commit, t, format!("published v{}", snap.version));
    let t = Instant::now();
    let weak = Arc::downgrade(&old);
    let gate = Arc::clone(&old.frames);
    drop(old);
    // Drain sleeps on the snapshot's frame gate until its last frame ends or the bound
    // passes: no spin and no sleep call (S5 review note, law 10).
    let at_commit = gate.running();
    let in_use = gate.wait_empty(std::time::Duration::from_millis(drain_ms));
    let waited = if at_commit == 0 {
        "no frame was running on the old snapshot at commit".to_owned()
    } else {
        format!("{at_commit} running frames were on the old snapshot at commit")
    };
    log.add(
        Phase::Drain,
        t,
        if in_use == 0 {
            format!("{waited}; none left")
        } else {
            format!("{waited}; {in_use} still on it after the wait, they finish on it")
        },
    );
    let t = Instant::now();
    let held = weak.strong_count();
    log.add(
        Phase::Retire,
        t,
        if in_use == 0 && held == 0 {
            "old snapshot and its unshared components reclaimed".to_owned()
        } else if in_use == 0 {
            format!("no frame left; {held} other holders release it later")
        } else {
            "old snapshot retires when its last frame ends".to_owned()
        },
    );
    Ok((snap, log))
}

/// Self-check of the `lifecycle` sub-component: a valid change goes through all seven
/// phases and keeps the untouched component; an invalid one stops at Validate and leaves
/// the running snapshot.
pub(crate) fn check() -> Result<(), String> {
    use crate::graph::{ConnectionMode, Edge, Node};
    use crate::runtime::{Exec, FnExec};
    use nmtype::{ComponentType, Payload, PortDesc, PortKind};
    let p = |n: &str| PortDesc::new(n, PortKind::Text, true);
    let g = Graph {
        nodes: vec![Node::new("a#0", ComponentType::Other).output(p("o")), Node::new("b#0", ComponentType::Other).input(p("i"))],
        edges: vec![Edge::new("a#0", "o", "b#0", "i", ConnectionMode::Direct)],
    };
    let echo = || -> Arc<dyn Exec> { Arc::new(FnExec(|_: &[Payload], _: &dyn nmtype::FrameView| Ok(Payload::Text("x".into())))) };
    let bind = |g: &Graph| -> Result<Components, Fault> { Ok(g.nodes.iter().map(|n| (n.id.clone(), echo())).collect()) };
    let nte = Nte::new(None);
    nte.publish(g.clone(), bind(&g).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let a0 = nte.snapshot().components.get("a#0").cloned().ok_or("nmnte.lifecycle: no a#0")?;
    let swap = Change::Swap(Node::new("b#0", ComponentType::Other).input(p("i")));
    let (snap, log) = replace(&nte, &swap, &bind, None, 100).map_err(|(f, _)| f.to_string())?;
    let kept = snap.components.get("a#0").is_some_and(|c| Arc::ptr_eq(c, &a0));
    if log.entries.len() != 7 || !kept || snap.version != 2 {
        return Err(format!("nmnte.lifecycle: phases {} kept {kept} version {}", log.entries.len(), snap.version));
    }
    let stale = nte.publish_from(1, g.clone(), bind(&g).map_err(|e| e.to_string())?);
    if stale.is_ok() {
        return Err("nmnte.lifecycle: a change derived from an old version was published".into());
    }
    let bad = Change::Rewire { add: vec![Edge::new("b#0", "i", "a#0", "o", ConnectionMode::Direct)], remove: vec![] };
    match replace(&nte, &bad, &bind, None, 100) {
        Err((_, log)) if log.entries.last().map(|e| e.0) == Some(Phase::Validate) && nte.snapshot().version == 2 => Ok(()),
        _ => Err("nmnte.lifecycle: invalid change not stopped at Validate".into()),
    }
}
