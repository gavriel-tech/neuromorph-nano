//! Execution frame (guide 14.3): the snapshot a task started on, its id, budget, deadline,
//! state, trace and, when suspended, the checkpoint it resumes from. Mutable state lives
//! here, execution-local; components and snapshots stay immutable and shared.
use crate::snapshot::Snapshot;
use nmtype::{Budget, ComponentId, Fault, FrameView, Hop, Payload};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Default and ceiling for a frame's run time when its budget gives none: every run is
/// under one minute (law 7, PLAN 2c). The bin reads `units/policy.json` `max_run_seconds`
/// when it builds budgets; inside nmnte this constant applies.
pub const MAX_RUN_MS: u32 = 60_000;

/// Frame states (guide 14.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionState {
    Ready,
    Running,
    Suspended,
    Completed,
    Failed,
    Cancelled,
}

/// Where a run stands between waves: enough to resume safely (guide 14.3: suspend only
/// where the state can be checkpointed). Owned by the frame, never by a component.
#[derive(Clone, Debug, Default)]
pub struct Progress {
    pub queue: VecDeque<Vec<ComponentId>>,
    pub inputs: BTreeMap<ComponentId, Vec<Payload>>,
    pub outputs: BTreeMap<ComponentId, Payload>,
    pub buffers: BTreeMap<usize, VecDeque<Payload>>,
    pub ran: BTreeSet<ComponentId>,
    pub failed: BTreeSet<ComponentId>,
    pub skipped: BTreeSet<ComponentId>,
    pub iterations: BTreeMap<ComponentId, u8>,
    pub faults: Vec<Fault>,
    pub isolated: Vec<ComponentId>,
    pub dropped: u64,
}

/// One execution of one snapshot.
#[derive(Debug)]
pub struct ExecutionFrame {
    pub id: u64,
    pub snapshot: Arc<Snapshot>,
    pub budget: Budget,
    pub deadline: u64,
    state: Mutex<ExecutionState>,
    trace: Mutex<Vec<Hop>>,
    clock: Mutex<Vec<NodeTime>>,
    cancel: AtomicBool,
    suspend: AtomicBool,
    pub(crate) checkpoint: Mutex<Option<Progress>>,
}
impl ExecutionFrame {
    /// A ready frame on `snapshot`. `budget.ms` 0 means the policy default, [`MAX_RUN_MS`];
    /// no frame runs without a deadline.
    #[must_use]
    pub fn new(id: u64, snapshot: Arc<Snapshot>, budget: Budget) -> Self {
        let ms = if budget.ms == 0 { MAX_RUN_MS } else { budget.ms };
        let deadline = nmtype::now_ms() + u64::from(ms);
        snapshot.frames.enter();
        Self {
            id,
            snapshot,
            budget,
            deadline,
            state: Mutex::new(ExecutionState::Ready),
            trace: Mutex::new(Vec::new()),
            clock: Mutex::new(Vec::new()),
            cancel: AtomicBool::new(false),
            suspend: AtomicBool::new(false),
            checkpoint: Mutex::new(None),
        }
    }
    #[must_use]
    pub fn state(&self) -> ExecutionState {
        self.state.lock().map_or(ExecutionState::Failed, |s| *s)
    }
    pub(crate) fn set_state(&self, s: ExecutionState) {
        if let Ok(mut g) = self.state.lock() {
            *g = s;
        }
    }
    /// Ask the run to stop before its next wave.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }
    /// Ask the run to checkpoint and stop before its next wave; `scheduler::resume` goes on.
    pub fn suspend(&self) {
        self.suspend.store(true, Ordering::SeqCst);
    }
    pub(crate) fn take_cancel(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
    pub(crate) fn take_suspend(&self) -> bool {
        self.suspend.swap(false, Ordering::SeqCst)
    }
    /// The hops recorded so far, in order.
    #[must_use]
    pub fn hops(&self) -> Vec<Hop> {
        self.trace.lock().map(|t| t.clone()).unwrap_or_default()
    }
    /// Records one node call's start and length in microseconds (measuring only; the hop keeps
    /// whole milliseconds).
    pub fn time_node(&self, component: &str, start_us: u64, us: u64) {
        if let Ok(mut c) = self.clock.lock() {
            c.push(NodeTime { component: component.to_owned(), start_us, us });
        }
    }
    /// Every node call's start and length in microseconds, in the order the calls ended.
    #[must_use]
    pub fn node_times(&self) -> Vec<NodeTime> {
        self.clock.lock().map(|c| c.clone()).unwrap_or_default()
    }
}

/// One node call as the frame's clock measured it: start in microseconds since 1970 and length
/// in microseconds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeTime {
    pub component: ComponentId,
    pub start_us: u64,
    pub us: u64,
}
impl Drop for ExecutionFrame {
    fn drop(&mut self) {
        self.snapshot.frames.leave();
    }
}
impl FrameView for ExecutionFrame {
    fn version(&self) -> u64 {
        self.snapshot.version
    }
    fn deadline(&self) -> u64 {
        self.deadline
    }
    fn budget(&self) -> &Budget {
        &self.budget
    }
    fn trace(&self, hop: Hop) {
        if let Ok(mut t) = self.trace.lock() {
            t.push(hop);
        }
    }
}

/// Self-check of the `frame` sub-component: a new frame is ready on its snapshot version,
/// keeps its trace in order, and flags cancel and suspend.
pub(crate) fn check() -> Result<(), String> {
    let snap = Arc::new(Snapshot::empty());
    let f = ExecutionFrame::new(1, Arc::clone(&snap), Budget { tokens: 0, ms: 0, depth: 0, units: 0 });
    let limit = nmtype::now_ms() + u64::from(MAX_RUN_MS);
    if f.state() != ExecutionState::Ready || f.version() != snap.version || f.deadline > limit || f.deadline == u64::MAX {
        return Err("nmnte.frame: new frame not ready on its snapshot".into());
    }
    f.trace(Hop::new("a#0", 1, "call", 0));
    f.trace(Hop::new("b#0", 2, "call", 0));
    if f.hops().iter().map(|h| h.component.as_str()).collect::<Vec<_>>() != ["a#0", "b#0"] {
        return Err("nmnte.frame: trace order".into());
    }
    f.suspend();
    f.cancel();
    if !f.take_suspend() || f.take_suspend() || !f.take_cancel() {
        return Err("nmnte.frame: cancel or suspend flag".into());
    }
    Ok(())
}
