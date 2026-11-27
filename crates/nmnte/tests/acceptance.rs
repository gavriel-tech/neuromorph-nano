//! The twelve NTE acceptance tests (PLAN 4c nmnte `tests/`, guide 14, 14.1 to 14.3): the one
//! kept suite, because the executor is the foundation everything else stands on. Test 11 is
//! `lateral_via_brain`: lateral traffic between tentacles must go through the Core Brain
//! (Luis's 2026-11-07 rule replaces the NTE document's direct lateral test).
use nmnte::{Components, ConnectionMode, Edge, Exec, ExecutionState, FnExec, Graph, Node, Nte, Runtime, SnapshotExec, run};
use nmtype::{Budget, ComponentType, Fault, FaultKind, FrameView, Payload, PortDesc, PortKind};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(10);

fn budget() -> Budget {
    Budget { tokens: 0, ms: 30_000, depth: 2, units: 16 }
}
fn node(id: &str, kind: ComponentType) -> Node {
    Node::new(id, kind).input(PortDesc::new("i", PortKind::Text, false)).output(PortDesc::new("o", PortKind::Text, true))
}
fn edge(a: &str, b: &str, mode: ConnectionMode) -> Edge {
    Edge::new(a, "o", b, "i", mode)
}
fn direct(a: &str, b: &str) -> Edge {
    edge(a, b, ConnectionMode::Direct)
}
fn texts(inputs: &[Payload]) -> Vec<String> {
    inputs.iter().filter_map(|p| if let Payload::Text(t) = p { Some(t.clone()) } else { None }).collect()
}
fn exec<F>(f: F) -> Arc<dyn Exec>
where
    F: Fn(&[Payload], &dyn FrameView) -> Result<Payload, Fault> + Send + Sync + 'static,
{
    Arc::new(FnExec(f))
}
/// A component that answers a fixed text.
fn says(text: &str) -> Arc<dyn Exec> {
    let t = text.to_owned();
    exec(move |_, _| Ok(Payload::Text(t.clone())))
}
/// A component that joins its text inputs with `+`.
fn join() -> Arc<dyn Exec> {
    exec(|inputs, _| Ok(Payload::Text(texts(inputs).join("+"))))
}
fn graph(nodes: Vec<Node>, edges: Vec<Edge>) -> Graph {
    Graph { nodes, edges }
}
fn comps(list: Vec<(&str, Arc<dyn Exec>)>) -> Components {
    list.into_iter().map(|(k, v)| (k.to_owned(), v)).collect()
}
fn text_of(r: &nmnte::RunResult, id: &str) -> Option<String> {
    match r.outputs.get(id) {
        Some(Payload::Text(t)) => Some(t.clone()),
        _ => None,
    }
}
/// A one-shot gate a component can wait on and a test can open.
#[derive(Default)]
struct Gate {
    open: Mutex<bool>,
    cv: Condvar,
}
impl Gate {
    fn open(&self) {
        if let Ok(mut g) = self.open.lock() {
            *g = true;
        }
        self.cv.notify_all();
    }
    fn wait(&self) -> bool {
        let Ok(g) = self.open.lock() else { return false };
        self.cv.wait_timeout_while(g, WAIT, |o| !*o).is_ok_and(|(g, _)| *g)
    }
}

#[test]
fn t01_parallel() {
    // Four nodes in one wave only finish if all four run at the same time.
    let arrived = Arc::new((Mutex::new(0_usize), Condvar::new()));
    let meet = |name: &'static str| {
        let a = Arc::clone(&arrived);
        exec(move |_, _| {
            let (m, cv) = &*a;
            let Ok(mut n) = m.lock() else { return Err(Fault::new(name, FaultKind::Execution, "lock")) };
            *n += 1;
            cv.notify_all();
            let ok = cv.wait_timeout_while(n, WAIT, |n| *n < 4).is_ok_and(|(n, _)| *n >= 4);
            if ok { Ok(Payload::Text(name.into())) } else { Err(Fault::new(name, FaultKind::Execution, "not parallel")) }
        })
    };
    let ids = ["p0", "p1", "p2", "p3"];
    let mut nodes = vec![node("s", ComponentType::Other), node("k", ComponentType::Other)];
    let mut edges = Vec::new();
    let mut c = comps(vec![("s", says("go")), ("k", join())]);
    for id in ids {
        nodes.push(node(id, ComponentType::Other));
        edges.push(direct("s", id));
        edges.push(direct(id, "k"));
        c.insert(id.into(), meet(id));
    }
    let nte = Nte::new(None);
    nte.publish(graph(nodes, edges), c).expect("publish");
    assert_eq!(nte.snapshot().plan.waves[1].len(), 4, "the four run in one wave");
    let r = run(&nte.frame(budget()), &Runtime::new(4), BTreeMap::new());
    assert!(r.faults.is_empty(), "{:?}", r.faults);
    assert_eq!(text_of(&r, "k").as_deref(), Some("p0+p1+p2+p3"));
}

#[test]
fn t02_branching() {
    let nodes = ["s", "a", "b", "yes", "no"].iter().map(|i| node(i, ComponentType::Other)).collect();
    let edges = vec![
        direct("s", "a"),
        direct("s", "b"),
        edge("s", "yes", ConnectionMode::Conditional("always".into())),
        edge("s", "no", ConnectionMode::Conditional("never".into())),
    ];
    let c = comps(vec![("s", says("x")), ("a", join()), ("b", join()), ("yes", join()), ("no", join())]);
    let nte = Nte::new(None);
    nte.publish(graph(nodes, edges), c).expect("publish");
    let r = run(&nte.frame(budget()), &Runtime::new(2), BTreeMap::new());
    assert_eq!(text_of(&r, "a").as_deref(), Some("x"), "both direct branches receive the payload");
    assert_eq!(text_of(&r, "b").as_deref(), Some("x"));
    assert_eq!(text_of(&r, "yes").as_deref(), Some("x"), "the true conditional branch runs");
    assert!(r.skipped.contains(&"no".to_owned()), "the false conditional branch is skipped");
}

#[test]
fn t03_convergence() {
    let nodes = ["a", "b", "c"].iter().map(|i| node(i, ComponentType::Other)).collect();
    let nte = Nte::new(None);
    nte.publish(graph(nodes, vec![direct("a", "c"), direct("b", "c")]), comps(vec![("a", says("A")), ("b", says("B")), ("c", join())]))
        .expect("publish");
    let r = run(&nte.frame(budget()), &Runtime::new(2), BTreeMap::new());
    assert_eq!(text_of(&r, "c").as_deref(), Some("A+B"), "both inputs arrive, in edge order");
}

#[test]
fn t04_feedback_no_deadlock() {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (c2, s2) = (Arc::clone(&calls), Arc::clone(&seen));
    let a = exec(move |inputs, _| {
        c2.fetch_add(1, Ordering::SeqCst);
        if let Ok(mut s) = s2.lock() {
            s.push(texts(inputs).join("+"));
        }
        Ok(Payload::Text("a".into()))
    });
    let nodes = vec![node("a", ComponentType::Other), node("b", ComponentType::Other)];
    let edges = vec![direct("a", "b"), edge("b", "a", ConnectionMode::Feedback { max_iter: 3 })];
    let nte = Nte::new(None);
    nte.publish(graph(nodes, edges), comps(vec![("a", a), ("b", exec(|i, _| Ok(Payload::Text(format!("b({})", texts(i).join("+"))))))]))
        .expect("publish");
    let r = run(&nte.frame(budget()), &Runtime::new(2), BTreeMap::new());
    assert_eq!(r.state, ExecutionState::Completed);
    assert_eq!(calls.load(Ordering::SeqCst), 4, "first run plus exactly max_iter re-runs");
    assert_eq!(r.iterations.get("a"), Some(&3));
    let seen = seen.lock().expect("lock").clone();
    assert_eq!(seen.first().map(String::as_str), Some(""), "first run has no feedback yet");
    assert!(seen.iter().skip(1).all(|s| s == "b(a)"), "re-runs receive the feedback payload: {seen:?}");
}

#[test]
fn t05_hot_swap_v1_v2() {
    let gate = Arc::new(Gate::default());
    let entered = Arc::new(Gate::default());
    let (g2, e2) = (Arc::clone(&gate), Arc::clone(&entered));
    let v1 = exec(move |_, _| {
        e2.open();
        if g2.wait() { Ok(Payload::Text("v1".into())) } else { Err(Fault::new("w", FaultKind::Execution, "gate timeout")) }
    });
    let g = graph(vec![node("s", ComponentType::Other), node("w", ComponentType::Other)], vec![direct("s", "w")]);
    let nte = Arc::new(Nte::new(None));
    nte.publish(g.clone(), comps(vec![("s", says("in")), ("w", v1)])).expect("v1");
    let n1 = Arc::clone(&nte);
    let old = std::thread::spawn(move || {
        let f = n1.frame(budget());
        run(&f, &Runtime::new(2), BTreeMap::new())
    });
    assert!(entered.wait(), "v1 frame reached the component");
    nte.publish(g, comps(vec![("s", says("in")), ("w", says("v2"))])).expect("v2 while v1 runs");
    let new = run(&nte.frame(budget()), &Runtime::new(2), BTreeMap::new());
    assert_eq!((new.version, text_of(&new, "w").as_deref()), (2, Some("v2")), "new frames use v2 at once");
    gate.open();
    let old = old.join().expect("v1 thread");
    assert_eq!((old.version, text_of(&old, "w").as_deref()), (1, Some("v1")), "the running frame finished on v1");
}

#[test]
fn t06_memory_safety_during_replace() {
    let g = graph(vec![node("s", ComponentType::Other), node("w", ComponentType::Other)], vec![direct("s", "w")]);
    let nte = Arc::new(Nte::new(None));
    let publish = |v: u64| comps(vec![("s", says("in")), ("w", says(&format!("v{v}")))]);
    nte.publish(g.clone(), publish(1)).expect("v1");
    let checked = Arc::new(AtomicUsize::new(0));
    let workers: Vec<_> = (0..6)
        .map(|_| {
            let (n, c) = (Arc::clone(&nte), Arc::clone(&checked));
            std::thread::spawn(move || {
                let rt = Runtime::new(2);
                for _ in 0..40 {
                    let f = n.frame(budget());
                    let r = run(&f, &rt, BTreeMap::new());
                    assert_eq!(text_of(&r, "w"), Some(format!("v{}", r.version)), "frame output matches its snapshot");
                    c.fetch_add(1, Ordering::SeqCst);
                }
            })
        })
        .collect();
    for v in 2..=60 {
        nte.publish(g.clone(), publish(v)).expect("publish under load");
    }
    for w in workers {
        w.join().expect("no worker panicked");
    }
    assert_eq!(checked.load(Ordering::SeqCst), 240);
    assert_eq!(nte.snapshot().version, 60);
}

#[test]
fn t07_fault_isolation() {
    let nodes = ["s", "bad", "after", "good"].iter().map(|i| node(i, ComponentType::Other)).collect();
    let edges = vec![direct("s", "bad"), direct("bad", "after"), direct("s", "good")];
    let bad = exec(|_, _| -> Result<Payload, Fault> { std::panic::panic_any("seeded component panic") });
    let nte = Nte::new(None);
    nte.publish(graph(nodes, edges), comps(vec![("s", says("x")), ("bad", bad), ("after", join()), ("good", join())])).expect("publish");
    for _ in 0..2 {
        let r = run(&nte.frame(budget()), &Runtime::new(2), BTreeMap::new());
        assert_eq!(r.state, ExecutionState::Completed, "the organism keeps running");
        assert!(r.faults.iter().any(|f| f.component == "bad"), "{:?}", r.faults);
        assert_eq!(r.isolated, vec!["after".to_owned()], "only the failing node's downstream is isolated");
        assert_eq!(text_of(&r, "good").as_deref(), Some("x"), "the sibling branch completes");
    }
}

#[test]
fn t08_rollback_on_failed_publish() {
    let ok = graph(
        vec![node("b", ComponentType::Brain), node("t0", ComponentType::Tentacle), node("t1", ComponentType::Tentacle)],
        vec![direct("b", "t0"), direct("b", "t1")],
    );
    let c = || comps(vec![("b", says("plan")), ("t0", join()), ("t1", join())]);
    let nte = Nte::new(None);
    nte.publish(ok.clone(), c()).expect("v1");
    let mut bypass = ok.clone();
    bypass.edges.push(direct("t0", "t1"));
    let e = nte.publish(bypass, c()).expect_err("tentacle-to-tentacle edge refused");
    assert_eq!(e.kind, FaultKind::Integrity);
    assert!(nte.publish(ok.clone(), comps(vec![("b", says("plan"))])).is_err(), "missing components refused");
    let snap = nte.snapshot();
    assert_eq!((snap.version, snap.graph.as_ref()), (1, &ok), "the published snapshot is unchanged");
    let r = run(&nte.frame(budget()), &Runtime::new(2), BTreeMap::new());
    assert_eq!(text_of(&r, "t1").as_deref(), Some("plan"), "v1 still runs");
}

#[test]
fn t09_backpressure_bounded() {
    let rt = Runtime::new(2);
    let done = AtomicUsize::new(0);
    let jobs: Vec<nmnte::runtime::Job<'_>> = (0..1000)
        .map(|_| {
            let d = &done;
            Box::new(move || {
                d.fetch_add(1, Ordering::Relaxed);
                Ok(Payload::Text(String::new()))
            }) as nmnte::runtime::Job<'_>
        })
        .collect();
    let out = rt.execute(jobs);
    assert_eq!((out.len(), done.load(Ordering::Relaxed)), (1000, 1000), "every job ran; none dropped");
    assert!(rt.peak_queue() <= rt.bound(), "queue {} over bound {}", rt.peak_queue(), rt.bound());
    assert!(rt.peak_queue() > 0);
    // A buffered edge under a producer that feedback re-runs five times: the consumer is
    // re-run with it and drains the edge each time, so nothing queues up and nothing drops.
    let nodes = ["a", "b", "late"].iter().map(|i| node(i, ComponentType::Other)).collect();
    let edges =
        vec![direct("a", "b"), edge("b", "a", ConnectionMode::Feedback { max_iter: 5 }), edge("a", "late", ConnectionMode::Buffered(2))];
    let nte = Nte::new(None);
    nte.publish(graph(nodes, edges), comps(vec![("a", says("x")), ("b", join()), ("late", join())])).expect("publish");
    let r = run(&nte.frame(budget()), &Runtime::new(2), BTreeMap::new());
    assert_eq!(r.state, ExecutionState::Completed);
    assert_eq!((r.dropped, text_of(&r, "late").as_deref()), (0, Some("x")), "one item per run, none dropped");
}

#[test]
fn t10_recursive_execution() {
    // The inner snapshot: in -> double. The outer node "sub" is that whole snapshot.
    let inner = Arc::new(Nte::new(None));
    inner
        .publish(
            graph(vec![node("in", ComponentType::Other), node("double", ComponentType::Other)], vec![direct("in", "double")]),
            comps(vec![("in", join()), ("double", exec(|i, _| Ok(Payload::Text(texts(i).join("").repeat(2)))))]),
        )
        .expect("inner");
    let sub: Arc<dyn Exec> =
        Arc::new(SnapshotExec { inner: Arc::clone(&inner), runtime: Runtime::new(2), entry: "in".into(), exit: "double".into() });
    let outer = Nte::new(None);
    outer
        .publish(
            graph(
                vec![node("s", ComponentType::Other), node("sub", ComponentType::Other), node("k", ComponentType::Other)],
                vec![direct("s", "sub"), direct("sub", "k")],
            ),
            comps(vec![("s", says("ab")), ("sub", sub), ("k", join())]),
        )
        .expect("outer");
    let r = run(&outer.frame(budget()), &Runtime::new(2), BTreeMap::new());
    assert_eq!(text_of(&r, "k").as_deref(), Some("abab"), "{:?}", r.faults);
}

#[test]
fn t11_lateral_via_brain() {
    let nodes = || vec![node("b", ComponentType::Brain), node("t0", ComponentType::Tentacle), node("t1", ComponentType::Tentacle)];
    let nte = Nte::new(None);
    let lateral = graph(nodes(), vec![direct("t0", "t1")]);
    assert!(nte.publish(lateral, comps(vec![("t0", says("req")), ("t1", join())])).is_err(), "direct lateral link refused");
    let mediated = graph(nodes(), vec![direct("t0", "b"), direct("b", "t1")]);
    let brain = exec(|i, _| Ok(Payload::Text(format!("brain:{}", texts(i).join("+")))));
    nte.publish(mediated, comps(vec![("t0", says("request from t0")), ("b", brain), ("t1", join())])).expect("mediated");
    let r = run(&nte.frame(budget()), &Runtime::new(2), BTreeMap::new());
    assert_eq!(text_of(&r, "t1").as_deref(), Some("brain:request from t0"), "t0 reached t1 through the Core Brain");
}

#[test]
fn t12_resource_reclamation_at_zero_refs() {
    let g = graph(vec![node("s", ComponentType::Other), node("w", ComponentType::Other)], vec![direct("s", "w")]);
    let nte = Nte::new(None);
    let c1 = says("v1");
    let weak_c1 = Arc::downgrade(&c1);
    let v1 = nte.publish(g.clone(), comps(vec![("s", says("in")), ("w", c1)])).expect("v1");
    let weak_v1 = Arc::downgrade(&v1);
    drop(v1);
    let frame = nte.frame(budget());
    nte.publish(g, comps(vec![("s", says("in")), ("w", says("v2"))])).expect("v2");
    assert!(weak_v1.upgrade().is_some() && weak_c1.upgrade().is_some(), "v1 lives while a frame needs it");
    let r = run(&frame, &Runtime::new(1), BTreeMap::new());
    assert_eq!(text_of(&r, "w").as_deref(), Some("v1"));
    drop(frame);
    assert!(weak_v1.upgrade().is_none(), "v1 reclaimed at zero references");
    assert!(weak_c1.upgrade().is_none(), "its component reclaimed with it");
}
