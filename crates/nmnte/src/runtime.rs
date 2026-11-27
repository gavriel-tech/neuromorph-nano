//! Neural worker runtime (guide 14.2, 14.3): std threads over a bounded channel, so a
//! producer waits instead of flooding memory (backpressure). A job that panics becomes a
//! fault of that job only. Also the executable side of a node: `Exec`, with adapters for
//! the typed nmtype traits.
use nmtype::{ContextProcessor, Fault, FaultKind, FrameView, Packet, Payload, TensorProcessor};
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, mpsc};

/// What a node runs. Shared by every frame that executes the snapshot holding it, so it
/// takes `&self`; mutable components sit behind the adapters' lock.
pub trait Exec: Send + Sync {
    /// # Errors
    /// The component's own fault.
    fn call(&self, inputs: &[Payload], frame: &dyn FrameView) -> Result<Payload, Fault>;
}

/// A closure as a node.
pub struct FnExec<F>(pub F);
impl<F> Exec for FnExec<F>
where
    F: Fn(&[Payload], &dyn FrameView) -> Result<Payload, Fault> + Send + Sync,
{
    fn call(&self, inputs: &[Payload], frame: &dyn FrameView) -> Result<Payload, Fault> {
        (self.0)(inputs, frame)
    }
}

fn locked(id: &str) -> Fault {
    Fault::new(id, FaultKind::Execution, "component lock poisoned by an earlier panic").unrecoverable()
}

/// A `ContextProcessor` as a node: every `Packets` input is concatenated in order.
pub struct ContextExec<T>(pub Mutex<T>);
impl<T: ContextProcessor> Exec for ContextExec<T> {
    fn call(&self, inputs: &[Payload], frame: &dyn FrameView) -> Result<Payload, Fault> {
        let packets: Vec<Packet> =
            inputs.iter().filter_map(|p| if let Payload::Packets(v) = p { Some(v.clone()) } else { None }).flatten().collect();
        let mut c = self.0.lock().map_err(|_| locked("nmnte.runtime"))?;
        c.process(&packets, frame).map(Payload::Packets)
    }
}

/// A `TensorProcessor` as a node: the first `Tensor` input is the argument.
pub struct TensorExec<T>(pub Mutex<T>);
impl<T: TensorProcessor> Exec for TensorExec<T> {
    fn call(&self, inputs: &[Payload], frame: &dyn FrameView) -> Result<Payload, Fault> {
        let x = inputs
            .iter()
            .find_map(|p| if let Payload::Tensor(t) = p { Some(t) } else { None })
            .ok_or_else(|| Fault::new("nmnte.runtime", FaultKind::Interface, "tensor node got no tensor input"))?;
        let mut c = self.0.lock().map_err(|_| locked("nmnte.runtime"))?;
        c.forward(x, frame).map(Payload::Tensor)
    }
}

/// One unit of work for the pool.
pub type Job<'a> = Box<dyn FnOnce() -> Result<Payload, Fault> + Send + 'a>;

/// Worker pool settings and the queue high-water mark.
#[derive(Debug)]
pub struct Runtime {
    workers: usize,
    bound: usize,
    peak: AtomicUsize,
    /// Workers lent by the one processor budget per `execute` (made with 0 workers).
    lent: bool,
}
impl Runtime {
    /// `workers` 0 means as many as the one processor budget (`nmtype::budget`, 2026-11-18 Phase
    /// C1) lends at each `execute`: at most one per thread, at least one; a fixed count is kept
    /// as given (tests and callers that ask for it).
    #[must_use]
    pub fn new(workers: usize) -> Self {
        let lent = workers == 0;
        let workers = if lent { nmtype::budget::total() } else { workers };
        Self { workers, bound: workers * 2, peak: AtomicUsize::new(0), lent }
    }
    #[must_use]
    pub const fn workers(&self) -> usize {
        self.workers
    }
    /// Channel bound: jobs waiting at most this many deep.
    #[must_use]
    pub const fn bound(&self) -> usize {
        self.bound
    }
    /// The deepest the job queue has been since this runtime was made.
    #[must_use]
    pub fn peak_queue(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }
    /// Run jobs on the pool; results come back in job order. A panic is the job's fault.
    /// The producer waits while `bound` jobs are queued (backpressure), so the queue depth
    /// is never above the bound; `peak_queue` records the deepest it went.
    pub fn execute<'a>(&self, jobs: Vec<Job<'a>>) -> Vec<Result<Payload, Fault>> {
        let n = jobs.len();
        if n == 0 {
            return Vec::new();
        }
        let queue = Queue { state: Mutex::new((VecDeque::new(), false)), not_full: Condvar::new(), not_empty: Condvar::new() };
        let (rtx, rrx) = mpsc::channel::<(usize, Result<Payload, Fault>)>();
        // The producer (this thread) waits on the queue, so it lends its own thread to one worker.
        let grant = self.lent.then(|| nmtype::budget::Grant::take(self.workers.min(n).saturating_sub(1)));
        let spawn = grant.as_ref().map_or(self.workers.min(n), |g| 1 + g.extra());
        std::thread::scope(|sc| {
            for _ in 0..spawn {
                let (queue, rtx) = (&queue, rtx.clone());
                sc.spawn(move || {
                    while let Some((i, job)) = queue.pop() {
                        let out = catch_unwind(AssertUnwindSafe(job))
                            .unwrap_or_else(|_| Err(Fault::new("nmnte.runtime", FaultKind::Execution, "job panicked; isolated")));
                        let _ = rtx.send((i, out));
                    }
                });
            }
            for (i, job) in jobs.into_iter().enumerate() {
                let depth = queue.push((i, job), self.bound);
                self.peak.fetch_max(depth, Ordering::Relaxed);
            }
            queue.close();
        });
        drop(grant);
        drop(rtx);
        let mut out: Vec<Option<Result<Payload, Fault>>> = (0..n).map(|_| None).collect();
        for (i, r) in rrx.iter() {
            if let Some(slot) = out.get_mut(i) {
                *slot = Some(r);
            }
        }
        out.into_iter().map(|r| r.unwrap_or_else(|| Err(Fault::new("nmnte.runtime", FaultKind::Execution, "job lost")))).collect()
    }
}

/// A bounded job queue: `push` waits while full, `pop` waits while empty and open.
struct Queue<'a> {
    state: Mutex<(VecDeque<(usize, Job<'a>)>, bool)>,
    not_full: Condvar,
    not_empty: Condvar,
}
impl<'a> Queue<'a> {
    /// Enqueue, waiting for room; returns the depth right after the push.
    fn push(&self, item: (usize, Job<'a>), bound: usize) -> usize {
        let Ok(mut g) = self.state.lock() else { return 0 };
        while g.0.len() >= bound {
            match self.not_full.wait(g) {
                Ok(next) => g = next,
                Err(_) => return 0,
            }
        }
        g.0.push_back(item);
        self.not_empty.notify_one();
        g.0.len()
    }
    fn pop(&self) -> Option<(usize, Job<'a>)> {
        let mut g = self.state.lock().ok()?;
        loop {
            if let Some(item) = g.0.pop_front() {
                self.not_full.notify_one();
                return Some(item);
            }
            if g.1 {
                return None;
            }
            g = self.not_empty.wait(g).ok()?;
        }
    }
    fn close(&self) {
        if let Ok(mut g) = self.state.lock() {
            g.1 = true;
        }
        self.not_empty.notify_all();
    }
}
/// Self-check of the `runtime` sub-component: 64 jobs on 2 workers come back in order, the
/// queue never passes its bound, and a panicking job becomes one fault.
pub(crate) fn check() -> Result<(), String> {
    let rt = Runtime::new(2);
    let mut jobs: Vec<Job<'_>> = (0..64_u64).map(|i| Box::new(move || Ok(Payload::Text(i.to_string()))) as Job<'_>).collect();
    jobs.push(Box::new(|| -> Result<Payload, Fault> { std::panic::panic_any("seeded panic") }));
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = rt.execute(jobs);
    std::panic::set_hook(prev);
    for (i, r) in out.iter().take(64).enumerate() {
        if r.as_ref().ok() != Some(&Payload::Text(i.to_string())) {
            return Err(format!("nmnte.runtime: job {i} out of order or failed"));
        }
    }
    if out.last().is_none_or(Result::is_ok) {
        return Err("nmnte.runtime: panic not turned into a fault".into());
    }
    if rt.peak_queue() > rt.bound() {
        return Err(format!("nmnte.runtime: queue reached {} over bound {}", rt.peak_queue(), rt.bound()));
    }
    Ok(())
}
