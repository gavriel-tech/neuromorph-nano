//! Immutable topology snapshots and their atomic publication (guide 14.3; PLAN 4c, 4d items
//! 1 and 10). A frame clones the current `Arc<Snapshot>` once and keeps it to the end, so a
//! newer snapshot can be published while older frames finish. Readers hold the lock only to
//! clone the `Arc`; nothing holds it during execution, so there is no global graph lock. A
//! publish that fails validation leaves the current snapshot as it was (rollback). An old
//! snapshot, and the components only it holds, are dropped when the last frame using it ends.
use crate::frame::ExecutionFrame;
use crate::graph::{ConnectionMode, Edge, Graph, Node};
use crate::integrity;
use crate::registry::Registry;
use crate::runtime::Exec;
use crate::scheduler::{self, ExecutionPlan};
use nmtype::{Budget, ComponentId, Fault, FaultKind, Packet, Payload, Status};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::Duration;

/// Executable components by node id.
pub type Components = BTreeMap<ComponentId, Arc<dyn Exec>>;
/// A named predicate for `Conditional` edges.
pub type Predicate = Arc<dyn Fn(&Payload) -> bool + Send + Sync>;

/// One immutable, validated anatomy with the components and predicates it executes.
pub struct Snapshot {
    pub version: u64,
    pub graph: Arc<Graph>,
    pub plan: Arc<ExecutionPlan>,
    pub components: Arc<Components>,
    pub predicates: Arc<BTreeMap<String, Predicate>>,
    pub at: u64,
    /// Frames running on this snapshot; the lifecycle's Drain waits on it.
    pub frames: Arc<FrameGate>,
}

/// Count of frames running on one snapshot, with a condition variable so Drain sleeps
/// until the last frame leaves or its bound passes (no spin, no sleep call; S5 review note).
#[derive(Debug, Default)]
pub struct FrameGate {
    count: Mutex<usize>,
    empty: Condvar,
}
impl FrameGate {
    /// A frame starts on the snapshot.
    pub fn enter(&self) {
        if let Ok(mut c) = self.count.lock() {
            *c += 1;
        }
    }
    /// A frame on the snapshot ended; wakes Drain when it was the last.
    pub fn leave(&self) {
        if let Ok(mut c) = self.count.lock() {
            *c = c.saturating_sub(1);
            if *c == 0 {
                self.empty.notify_all();
            }
        }
    }
    /// Frames running now.
    #[must_use]
    pub fn running(&self) -> usize {
        self.count.lock().map_or(0, |c| *c)
    }
    /// Sleep until no frame runs or `bound` passes; returns the frames still running.
    #[must_use]
    pub fn wait_empty(&self, bound: Duration) -> usize {
        let Ok(c) = self.count.lock() else { return 0 };
        match self.empty.wait_timeout_while(c, bound, |n| *n > 0) {
            Ok((n, _)) => *n,
            Err(_) => 0,
        }
    }
}
impl std::fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Snapshot")
            .field("version", &self.version)
            .field("nodes", &self.graph.nodes.len())
            .field("edges", &self.graph.edges.len())
            .field("components", &self.components.len())
            .finish_non_exhaustive()
    }
}
impl Snapshot {
    /// Version 0, nothing in it.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            version: 0,
            graph: Arc::new(Graph::default()),
            plan: Arc::new(ExecutionPlan::default()),
            components: Arc::new(Components::new()),
            predicates: Arc::new(builtin_predicates()),
            at: nmtype::now_ms(),
            frames: Arc::new(FrameGate::default()),
        }
    }
    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.graph.nodes
    }
    #[must_use]
    pub fn edges(&self) -> &[Edge] {
        &self.graph.edges
    }
}

/// Predicates every snapshot knows: `always`, `never`, `nonempty` (a non-empty packet list
/// or text), `has_unknown` (a packet whose status is unknown).
#[must_use]
pub fn builtin_predicates() -> BTreeMap<String, Predicate> {
    let mut m: BTreeMap<String, Predicate> = BTreeMap::new();
    m.insert("always".into(), Arc::new(|_| true));
    m.insert("never".into(), Arc::new(|_| false));
    m.insert(
        "nonempty".into(),
        Arc::new(|p| match p {
            Payload::Packets(v) => !v.is_empty(),
            Payload::Text(t) => !t.is_empty(),
            _ => true,
        }),
    );
    m.insert(
        "has_unknown".into(),
        Arc::new(|p| matches!(p, Payload::Packets(v) if v.iter().any(|x: &Packet| x.get_status() == Status::Unknown))),
    );
    m
}

fn fault(kind: FaultKind, detail: impl Into<String>) -> Fault {
    Fault::new("nmnte.snapshot", kind, detail)
}

/// The executor's publication point.
pub struct Nte {
    current: RwLock<Arc<Snapshot>>,
    registry: Option<RwLock<Registry>>,
    extra_predicates: RwLock<BTreeMap<String, Predicate>>,
    next_frame: AtomicU64,
}
impl std::fmt::Debug for Nte {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Nte").field("current", &self.snapshot()).finish_non_exhaustive()
    }
}
impl Nte {
    /// An executor holding the empty snapshot. With a registry, every published node must
    /// resolve in it.
    #[must_use]
    pub fn new(registry: Option<Registry>) -> Self {
        Self {
            current: RwLock::new(Arc::new(Snapshot::empty())),
            registry: registry.map(RwLock::new),
            extra_predicates: RwLock::new(BTreeMap::new()),
            next_frame: AtomicU64::new(1),
        }
    }
    fn registry_guard(&self) -> Result<Option<std::sync::RwLockReadGuard<'_, Registry>>, Fault> {
        match &self.registry {
            Some(r) => r.read().map(Some).map_err(|_| fault(FaultKind::Resource, "registry lock")),
            None => Ok(None),
        }
    }
    /// Register a card at run time (a learning-weight unit, S8) with its lifecycle state, so
    /// `nm_anatomy` and the Integrity Validator see it. Without a registry this is refused.
    /// # Errors
    /// No registry, or a poisoned lock.
    pub fn register(&self, card: nmtype::Card, state: crate::registry::CardState) -> Result<(), Fault> {
        let r = self.registry.as_ref().ok_or_else(|| fault(FaultKind::Config, "this executor has no registry"))?;
        r.write().map_err(|_| fault(FaultKind::Resource, "registry lock"))?.register(card, state);
        Ok(())
    }
    /// Register a predicate for conditional edges in snapshots published from now on.
    pub fn add_predicate(&self, name: &str, p: Predicate) {
        if let Ok(mut m) = self.extra_predicates.write() {
            m.insert(name.to_owned(), p);
        }
    }
    /// The current snapshot (the lock is held only to clone the `Arc`).
    #[must_use]
    pub fn snapshot(&self) -> Arc<Snapshot> {
        self.current.read().map_or_else(|p| Arc::clone(&p.into_inner()), |g| Arc::clone(&g))
    }
    /// Validate and publish a new snapshot atomically. On any fault the current snapshot is
    /// unchanged.
    /// # Errors
    /// Integrity faults, a connected node with no component, a component for no node, or a
    /// conditional edge naming an unknown predicate.
    pub fn publish(&self, graph: Graph, components: Components) -> Result<Arc<Snapshot>, Fault> {
        self.publish_checked(None, graph, components)
    }
    /// Publish only if the current snapshot is still `expected` (the version the new graph
    /// was derived from), so two concurrent changes cannot lose one update.
    /// # Errors
    /// An Integrity fault when another change was published first, or as `publish`.
    pub fn publish_from(&self, expected: u64, graph: Graph, components: Components) -> Result<Arc<Snapshot>, Fault> {
        self.publish_checked(Some(expected), graph, components)
    }
    fn publish_checked(&self, expected: Option<u64>, graph: Graph, components: Components) -> Result<Arc<Snapshot>, Fault> {
        let reg = self.registry_guard()?;
        integrity::validate(&graph, reg.as_deref())?;
        let connected = |id: &str| graph.edges.iter().any(|e| e.from == id || e.to == id);
        for n in &graph.nodes {
            if connected(&n.id) && !components.contains_key(&n.id) {
                return Err(fault(FaultKind::Config, format!("{}: connected node has no component", n.id)));
            }
        }
        if let Some(extra) = components.keys().find(|k| graph.node(k).is_none()) {
            return Err(fault(FaultKind::Config, format!("{extra}: component for a node that is not in the graph")));
        }
        let mut predicates = builtin_predicates();
        if let Ok(m) = self.extra_predicates.read() {
            predicates.extend(m.iter().map(|(k, v)| (k.clone(), Arc::clone(v))));
        }
        for e in &graph.edges {
            if let ConnectionMode::Conditional(p) = &e.mode
                && !predicates.contains_key(p)
            {
                return Err(fault(FaultKind::Config, format!("edge {} -> {}: unknown predicate {p}", e.from, e.to)));
            }
        }
        let plan = scheduler::plan(&graph)?;
        let mut guard = self.current.write().map_err(|_| fault(FaultKind::Execution, "publication lock poisoned"))?;
        if let Some(want) = expected
            && guard.version != want
        {
            return Err(fault(
                FaultKind::Integrity,
                format!("change derived from v{want}, but v{} was published meanwhile; derive again", guard.version),
            ));
        }
        let next = Arc::new(Snapshot {
            version: guard.version + 1,
            graph: Arc::new(graph),
            plan: Arc::new(plan),
            components: Arc::new(components),
            predicates: Arc::new(predicates),
            at: nmtype::now_ms(),
            frames: Arc::new(FrameGate::default()),
        });
        *guard = Arc::clone(&next);
        Ok(next)
    }
    /// Run the Integrity Validator with this executor's registry (the lifecycle's Validate).
    /// # Errors
    /// The first rule the graph breaks.
    pub fn validate(&self, graph: &Graph) -> Result<(), Fault> {
        let reg = self.registry_guard()?;
        integrity::validate(graph, reg.as_deref())
    }
    /// A ready frame on the current snapshot.
    #[must_use]
    pub fn frame(&self, budget: Budget) -> ExecutionFrame {
        let id = self.next_frame.fetch_add(1, Ordering::Relaxed);
        ExecutionFrame::new(id, self.snapshot(), budget)
    }
}

/// A node that is itself a snapshot (recursive execution, guide 11.1, 14.1): each call runs
/// a nested frame on the inner executor's current snapshot, feeding `entry` and returning
/// what `exit` produced. The inner executor publishes on its own, so an inner anatomy can
/// change without touching the outer one.
pub struct SnapshotExec {
    pub inner: Arc<Nte>,
    pub runtime: crate::runtime::Runtime,
    pub entry: ComponentId,
    pub exit: ComponentId,
}
impl Exec for SnapshotExec {
    fn call(&self, inputs: &[Payload], frame: &dyn nmtype::FrameView) -> Result<Payload, Fault> {
        // Recursion stops at depth 0 (PLAN 6: depth default 1, a setting); each level spends one.
        if frame.budget().depth == 0 {
            return Err(fault(FaultKind::Resource, format!("recursion depth exhausted before inner snapshot {}", self.entry)));
        }
        let mut budget = *frame.budget();
        budget.depth -= 1;
        let inner = self.inner.frame(budget);
        let r = scheduler::run(&inner, &self.runtime, BTreeMap::from([(self.entry.clone(), inputs.to_vec())]));
        if let Some(f) = r.faults.into_iter().next() {
            return Err(f);
        }
        r.outputs.get(&self.exit).cloned().ok_or_else(|| fault(FaultKind::Execution, format!("inner exit {} produced nothing", self.exit)))
    }
}

/// Self-check of the `snapshot` sub-component: publish bumps the version, a faulty publish
/// leaves it unchanged, a frame keeps the snapshot it started on, and a snapshot node that
/// contains itself stops at the depth budget instead of recursing without end.
pub(crate) fn check() -> Result<(), String> {
    use crate::runtime::FnExec;
    use nmtype::{ComponentType, PortDesc, PortKind};
    let nte = Nte::new(None);
    let p = |n: &str| PortDesc::new(n, PortKind::Text, true);
    let g = Graph {
        nodes: vec![Node::new("a#0", ComponentType::Other).output(p("o")), Node::new("b#0", ComponentType::Other).input(p("i"))],
        edges: vec![Edge::new("a#0", "o", "b#0", "i", ConnectionMode::Direct)],
    };
    let echo: Arc<dyn Exec> = Arc::new(FnExec(|_: &[Payload], _: &dyn nmtype::FrameView| Ok(Payload::Text("x".into()))));
    let comps = |e: &Arc<dyn Exec>| Components::from([("a#0".to_owned(), Arc::clone(e)), ("b#0".to_owned(), Arc::clone(e))]);
    nte.publish(g.clone(), comps(&echo)).map_err(|e| e.to_string())?;
    let frame = nte.frame(Budget { tokens: 0, ms: 0, depth: 0, units: 0 });
    if nte.publish(g.clone(), Components::new()).is_ok() || nte.snapshot().version != 1 {
        return Err("nmnte.snapshot: faulty publish changed the snapshot".into());
    }
    nte.publish(g, comps(&echo)).map_err(|e| e.to_string())?;
    if frame.snapshot.version != 1 || nte.snapshot().version != 2 {
        return Err("nmnte.snapshot: frame did not keep its snapshot".into());
    }
    let looped = Arc::new(Nte::new(None));
    let sub: Arc<dyn Exec> = Arc::new(SnapshotExec {
        inner: Arc::clone(&looped),
        runtime: crate::runtime::Runtime::new(1),
        entry: "a#0".into(),
        exit: "b#0".into(),
    });
    let selfref = Graph {
        nodes: vec![
            Node::new("a#0", ComponentType::Other).input(p("i")).output(p("o")),
            Node::new("b#0", ComponentType::Other).input(p("i")),
        ],
        edges: vec![Edge::new("a#0", "o", "b#0", "i", ConnectionMode::Direct)],
    };
    looped
        .publish(selfref, Components::from([("a#0".to_owned(), Arc::clone(&sub)), ("b#0".to_owned(), echo)]))
        .map_err(|e| e.to_string())?;
    let f = looped.frame(Budget { tokens: 0, ms: 5_000, depth: 3, units: 0 });
    let r = scheduler::run(&f, &crate::runtime::Runtime::new(1), BTreeMap::new());
    if !r.faults.iter().any(|x| x.kind == FaultKind::Resource) {
        return Err("nmnte.snapshot: self-containing snapshot did not stop at the depth budget".into());
    }
    Ok(())
}
