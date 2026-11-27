//! Topology scheduling and execution coordination (guide 14.1, 14.2; PLAN 4d item 1).
//! The plan is a topological order in waves over ordering edges (direct, buffered,
//! conditional); asynchronous and feedback edges never order. Feedback: after wave N, every
//! feedback edge whose source finished in wave N re-queues its destination, and the nodes
//! already run downstream of it, in the next waves, while the destination has run fewer
//! than `max_iter` extra times in this frame. Iteration state lives in the frame, so no
//! deadlock is possible: the plan has no cycle and iterations are bounded.
//! Per call, the wrappers from `fwcore/schedl`: budget (deadline) before the call, the call,
//! the output-type check after it, and a hop in the frame trace. (The input hashes before
//! and after are not taken: inputs are shared immutably, which Rust already guarantees.)
use crate::frame::{ExecutionFrame, ExecutionState, Progress};
use crate::graph::{ConnectionMode, Edge, Graph};
use crate::runtime::{Job, Runtime};
use crate::topology;
use nmtype::{ComponentId, Fault, FaultKind, FrameView, Hop, Payload, PortKind};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Waves of node ids (each wave runs in parallel) and the feedback edges.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecutionPlan {
    pub waves: Vec<Vec<ComponentId>>,
    pub feedback: Vec<Edge>,
}
impl ExecutionPlan {
    fn wave_of(&self, id: &str) -> usize {
        self.waves.iter().position(|w| w.iter().any(|x| x == id)).unwrap_or(usize::MAX)
    }
}

/// Build the plan. Only connected nodes run; structural nodes (no edges) are left out.
/// # Errors
/// An ordering cycle: a loop must be declared with a feedback edge.
pub fn plan(g: &Graph) -> Result<ExecutionPlan, Fault> {
    let connected: BTreeSet<&str> = g.edges.iter().flat_map(|e| [e.from.as_str(), e.to.as_str()]).collect();
    let mut indeg: BTreeMap<&str, usize> = connected.iter().map(|n| (*n, 0)).collect();
    for e in g.edges.iter().filter(|e| e.mode.orders()) {
        *indeg.entry(e.to.as_str()).or_default() += 1;
    }
    let mut waves = Vec::new();
    let mut ready: Vec<&str> = indeg.iter().filter(|(_, d)| **d == 0).map(|(n, _)| *n).collect();
    let mut done = 0;
    while !ready.is_empty() {
        ready.sort_unstable();
        let mut next = Vec::new();
        for n in &ready {
            for e in g.edges.iter().filter(|e| e.from == *n && e.mode.orders()) {
                if let Some(d) = indeg.get_mut(e.to.as_str()) {
                    *d -= 1;
                    if *d == 0 {
                        next.push(e.to.as_str());
                    }
                }
            }
        }
        done += ready.len();
        waves.push(ready.iter().map(|s| (*s).to_owned()).collect());
        next.dedup();
        ready = next;
    }
    if done != indeg.len() {
        let stuck: Vec<&str> = indeg.iter().filter(|(_, d)| **d > 0).map(|(n, _)| *n).collect();
        return Err(Fault::new(
            "nmnte.scheduler",
            FaultKind::Integrity,
            format!("ordering cycle through {stuck:?}; declare the loop with a feedback edge"),
        ));
    }
    let feedback = g.edges.iter().filter(|e| matches!(e.mode, ConnectionMode::Feedback { .. })).cloned().collect();
    Ok(ExecutionPlan { waves, feedback })
}

/// What a run produced.
#[derive(Clone, Debug)]
pub struct RunResult {
    pub version: u64,
    pub state: ExecutionState,
    pub outputs: BTreeMap<ComponentId, Payload>,
    pub faults: Vec<Fault>,
    pub isolated: Vec<ComponentId>,
    pub skipped: Vec<ComponentId>,
    pub iterations: BTreeMap<ComponentId, u8>,
    pub dropped: u64,
}

/// Run the frame's snapshot from the start with external inputs per node.
pub fn run(frame: &ExecutionFrame, runtime: &Runtime, inputs: BTreeMap<ComponentId, Vec<Payload>>) -> RunResult {
    let progress = Progress { queue: frame.snapshot.plan.waves.iter().cloned().collect(), inputs, ..Progress::default() };
    drive(frame, runtime, progress)
}

/// Resume a suspended frame from its checkpoint.
/// # Errors
/// The frame is not suspended.
pub fn resume(frame: &ExecutionFrame, runtime: &Runtime) -> Result<RunResult, Fault> {
    let saved = frame.checkpoint.lock().ok().and_then(|mut c| c.take());
    match (frame.state(), saved) {
        (ExecutionState::Suspended, Some(p)) => Ok(drive(frame, runtime, p)),
        (s, _) => Err(Fault::new("nmnte.scheduler", FaultKind::Execution, format!("frame {} is {s:?}, not suspended", frame.id))),
    }
}

fn finish(frame: &ExecutionFrame, p: Progress, state: ExecutionState) -> RunResult {
    frame.set_state(state);
    RunResult {
        version: frame.snapshot.version,
        state,
        outputs: p.outputs,
        faults: p.faults,
        isolated: p.isolated,
        skipped: p.skipped.into_iter().collect(),
        iterations: p.iterations,
        dropped: p.dropped,
    }
}

/// Inputs for one node, or `None` when it must not run (an ordering upstream delivered
/// nothing and there is no external input).
fn gather(frame: &ExecutionFrame, p: &mut Progress, id: &str) -> Result<Option<Vec<Payload>>, Fault> {
    let snap = &frame.snapshot;
    let mut inputs = p.inputs.get(id).cloned().unwrap_or_default();
    let mut has_ordering = false;
    let mut delivered = !inputs.is_empty();
    for (ix, e) in snap.graph.edges.iter().enumerate().filter(|(_, e)| e.to == id) {
        has_ordering |= e.mode.orders();
        let Some(buf) = p.buffers.get_mut(&ix) else { continue };
        let got: Vec<Payload> = match &e.mode {
            ConnectionMode::Buffered(_) => buf.drain(..).collect(),
            ConnectionMode::Conditional(name) => {
                let pred = snap
                    .predicates
                    .get(name)
                    .ok_or_else(|| Fault::new("nmnte.scheduler", FaultKind::Config, format!("unknown predicate {name}")))?;
                buf.back().filter(|x| pred(x)).cloned().into_iter().collect()
            }
            _ => buf.back().cloned().into_iter().collect(),
        };
        if !got.is_empty() && (e.mode.orders() || matches!(e.mode, ConnectionMode::Feedback { .. })) {
            delivered = true;
        }
        inputs.extend(got);
    }
    Ok((delivered || !has_ordering).then_some(inputs))
}

/// Microseconds since 1970 by the system clock.
fn now_us() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| u64::try_from(d.as_micros()).unwrap_or(u64::MAX))
}

/// The node's job with its wrappers: deadline, call, output-type check, trace hop.
fn job<'a>(frame: &'a ExecutionFrame, id: &'a str, inputs: Vec<Payload>) -> Job<'a> {
    Box::new(move || {
        let snap = &frame.snapshot;
        if nmtype::now_ms() > frame.deadline {
            return Err(Fault::new(id, FaultKind::Resource, "deadline passed before the call"));
        }
        let comp = snap.components.get(id).ok_or_else(|| Fault::new(id, FaultKind::Config, "no component bound"))?;
        let start_us = now_us();
        let start = std::time::Instant::now();
        let out = comp.call(&inputs, frame)?;
        let want = snap.graph.node(id).and_then(|n| n.ports_out.first()).map(|p| p.kind);
        if let Some(k) = want
            && k != PortKind::Any
            && k != out.kind()
        {
            return Err(Fault::new(id, FaultKind::Interface, format!("produced {} where its port is {}", out.kind().name(), k.name())));
        }
        let elapsed = start.elapsed();
        let cost = u32::try_from(elapsed.as_millis()).unwrap_or(u32::MAX);
        frame.trace(Hop::new(id, nmtype::now_ms(), "call", cost));
        frame.time_node(id, start_us, u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX));
        Ok(out)
    })
}

fn deliver(frame: &ExecutionFrame, p: &mut Progress, id: &str, out: &Payload) {
    for (ix, e) in frame.snapshot.graph.edges.iter().enumerate().filter(|(_, e)| e.from == id) {
        let buf = p.buffers.entry(ix).or_default();
        match e.mode {
            ConnectionMode::Buffered(bound) => {
                buf.push_back(out.clone());
                while buf.len() > bound {
                    buf.pop_front();
                    p.dropped += 1;
                }
            }
            _ => {
                buf.clear();
                buf.push_back(out.clone());
            }
        }
    }
}

/// Feedback after a wave: re-queue destinations (and the already-run nodes downstream of
/// them, in plan order) while their iteration count allows.
fn feedback(frame: &ExecutionFrame, p: &mut Progress, finished: &[ComponentId]) {
    let snap = Arc::clone(&frame.snapshot);
    let mut rerun: BTreeSet<ComponentId> = BTreeSet::new();
    for e in &snap.plan.feedback {
        let ConnectionMode::Feedback { max_iter } = e.mode else { continue };
        if !finished.contains(&e.from) || p.failed.contains(&e.to) {
            continue;
        }
        let n = p.iterations.entry(e.to.clone()).or_insert(0);
        if *n >= max_iter {
            continue;
        }
        *n += 1;
        rerun.insert(e.to.clone());
        rerun.extend(topology::reachable(&snap.graph, &e.to).into_iter().filter(|x| p.ran.contains(x)));
    }
    if rerun.is_empty() {
        return;
    }
    let mut groups: BTreeMap<usize, Vec<ComponentId>> = BTreeMap::new();
    for id in rerun {
        groups.entry(snap.plan.wave_of(&id)).or_default().push(id);
    }
    for (_, wave) in groups.into_iter().rev() {
        p.queue.push_front(wave);
    }
}

fn drive(frame: &ExecutionFrame, runtime: &Runtime, mut p: Progress) -> RunResult {
    frame.set_state(ExecutionState::Running);
    while let Some(wave) = p.queue.pop_front() {
        if frame.take_cancel() {
            return finish(frame, p, ExecutionState::Cancelled);
        }
        if frame.take_suspend() {
            p.queue.push_front(wave);
            let state = ExecutionState::Suspended;
            frame.set_state(state);
            let snapshot = p.clone();
            if let Ok(mut c) = frame.checkpoint.lock() {
                *c = Some(snapshot);
            }
            return finish(frame, p, state);
        }
        if nmtype::now_ms() > frame.deadline {
            p.faults.push(Fault::new("nmnte.scheduler", FaultKind::Resource, format!("frame {} passed its deadline", frame.id)));
            return finish(frame, p, ExecutionState::Failed);
        }
        let mut ready: Vec<(&str, Vec<Payload>)> = Vec::new();
        for id in &wave {
            let ordering_up = frame.snapshot.graph.edges.iter().filter(|e| e.to == *id && e.mode.orders());
            if ordering_up.clone().any(|e| p.failed.contains(&e.from)) {
                p.failed.insert(id.clone());
                p.isolated.push(id.clone());
                continue;
            }
            match gather(frame, &mut p, id) {
                Ok(Some(inputs)) => ready.push((id.as_str(), inputs)),
                Ok(None) => {
                    p.skipped.insert(id.clone());
                }
                Err(f) => {
                    p.failed.insert(id.clone());
                    p.faults.push(f);
                }
            }
        }
        let ids: Vec<ComponentId> = ready.iter().map(|(id, _)| (*id).to_owned()).collect();
        let jobs: Vec<Job<'_>> = ready.into_iter().map(|(id, inputs)| job(frame, id, inputs)).collect();
        let results = runtime.execute(jobs);
        let mut finished = Vec::new();
        for (id, r) in ids.into_iter().zip(results) {
            match r {
                Ok(out) => {
                    deliver(frame, &mut p, &id, &out);
                    p.skipped.remove(&id);
                    p.outputs.insert(id.clone(), out);
                    p.ran.insert(id.clone());
                    finished.push(id);
                }
                Err(mut f) => {
                    // A fault belongs to the node that raised it; the inner origin stays in the text.
                    if f.component != id {
                        f.message = format!("[{}] {}", f.component, f.message);
                        f.component.clone_from(&id);
                    }
                    p.failed.insert(id);
                    p.faults.push(f);
                }
            }
        }
        feedback(frame, &mut p, &finished);
    }
    finish(frame, p, ExecutionState::Completed)
}

/// Self-check of the `scheduler` sub-component: waves of a diamond, a refused ordering
/// cycle, and a feedback loop that runs exactly `max_iter` extra times.
pub(crate) fn check() -> Result<(), String> {
    use crate::graph::Node;
    use crate::runtime::FnExec;
    use crate::snapshot::{Components, Nte};
    use nmtype::{Budget, ComponentType, PortDesc};
    let p = |n: &str| PortDesc::new(n, PortKind::Text, true);
    let node = |id: &str| Node::new(id, ComponentType::Other).input(p("i")).output(p("o"));
    let d = |a: &str, b: &str| Edge::new(a, "o", b, "i", ConnectionMode::Direct);
    let diamond = Graph {
        nodes: ["a#0", "b#0", "c#0", "d#0"].iter().map(|i| node(i)).collect(),
        edges: vec![d("a#0", "b#0"), d("a#0", "c#0"), d("b#0", "d#0"), d("c#0", "d#0")],
    };
    let waves = plan(&diamond).map_err(|e| e.to_string())?.waves;
    if waves != [vec!["a#0".to_owned()], vec!["b#0".into(), "c#0".into()], vec!["d#0".into()]] {
        return Err(format!("nmnte.scheduler: diamond waves {waves:?}"));
    }
    let mut cyc = diamond.clone();
    cyc.edges.push(d("d#0", "a#0"));
    if plan(&cyc).is_ok() {
        return Err("nmnte.scheduler: ordering cycle accepted".into());
    }
    let mut fb = diamond;
    fb.edges.push(Edge::new("d#0", "o", "a#0", "i", ConnectionMode::Feedback { max_iter: 2 }));
    let echo: Arc<dyn crate::runtime::Exec> = Arc::new(FnExec(|_: &[Payload], _: &dyn FrameView| Ok(Payload::Text("x".into()))));
    let comps: Components = fb.nodes.iter().map(|n| (n.id.clone(), Arc::clone(&echo))).collect();
    let nte = Nte::new(None);
    nte.publish(fb, comps).map_err(|e| e.to_string())?;
    let frame = nte.frame(Budget { tokens: 0, ms: 10_000, depth: 0, units: 0 });
    let r = run(&frame, &Runtime::new(2), BTreeMap::new());
    if r.state != ExecutionState::Completed || r.iterations.get("a#0") != Some(&2) {
        return Err(format!("nmnte.scheduler: feedback ran {:?}, state {:?}", r.iterations, r.state));
    }
    Ok(())
}
