//! Stage timing for part tests (2026-11-10, step P1 of
//! `docs/2026-11-10-neumor-plan-whole-architecture.md`; Gavriel Technologies). Off unless a
//! test switches it on (`set(true)`); when off, a span is one relaxed atomic load and nothing
//! else, so no result and no code path changes. When on, every span records its exclusive
//! time (its own time less the spans opened inside it) under its path (`outer > inner`), so
//! the exclusive times of every path add up exactly to the outermost spans' total. Callers
//! that time GPU work synchronise the device at span edges themselves (`nmengn::model`), so
//! queued kernels are charged to the stage that queued them. One thread's spans nest; other
//! threads keep their own stacks.
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

static ON: AtomicBool = AtomicBool::new(false);
/// Kernel timing (2026-11-13): when on together with `ON`, every kernel launch synchronises the
/// device and charges its time to `kernel <name>` in the side table (slows runs; profiling only).
static KERNELS: AtomicBool = AtomicBool::new(false);
/// Per-kernel timing on or off (only while timing is on).
pub fn set_kernels(on: bool) {
    KERNELS.store(on, Ordering::Relaxed);
}
/// Per-kernel timing is on.
#[must_use]
pub fn kernels() -> bool {
    on() && KERNELS.load(Ordering::Relaxed)
}
/// Per path: (exclusive seconds, count).
static TABLE: Mutex<BTreeMap<String, (f64, u64)>> = Mutex::new(BTreeMap::new());
/// Side totals, inclusive, not part of the exclusive sum (for example time per layer).
static SIDE: Mutex<BTreeMap<String, (f64, u64)>> = Mutex::new(BTreeMap::new());

thread_local! {
    /// Open spans of this thread: (name, seconds of the spans closed inside it).
    static STACK: RefCell<Vec<(&'static str, f64)>> = const { RefCell::new(Vec::new()) };
}

/// Timing on or off.
pub fn set(on: bool) {
    ON.store(on, Ordering::Relaxed);
}
/// Timing is on.
#[must_use]
pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}
/// Clear both tables.
pub fn reset() {
    if let Ok(mut t) = TABLE.lock() {
        t.clear();
    }
    if let Ok(mut t) = SIDE.lock() {
        t.clear();
    }
}
/// The exclusive table: (path, seconds, count), by path.
#[must_use]
pub fn table() -> Vec<(String, f64, u64)> {
    TABLE.lock().map(|t| t.iter().map(|(k, v)| (k.clone(), v.0, v.1)).collect()).unwrap_or_default()
}
/// The side table (inclusive totals under their own names).
#[must_use]
pub fn side() -> Vec<(String, f64, u64)> {
    SIDE.lock().map(|t| t.iter().map(|(k, v)| (k.clone(), v.0, v.1)).collect()).unwrap_or_default()
}
/// Add inclusive seconds to a side total (when timing is on).
pub fn add_side(name: &str, secs: f64) {
    if !on() {
        return;
    }
    if let Ok(mut t) = SIDE.lock() {
        let e = t.entry(name.to_owned()).or_insert((0.0, 0));
        e.0 += secs;
        e.1 += 1;
    }
}

/// An open span; closes when dropped. `Span::new` returns an inert span when timing is off.
#[derive(Debug)]
pub struct Span {
    start: Option<Instant>,
}
impl Span {
    /// Open a span named `name` (nested under the spans open on this thread).
    #[must_use]
    pub fn new(name: &'static str) -> Self {
        if !on() {
            return Self { start: None };
        }
        STACK.with(|s| s.borrow_mut().push((name, 0.0)));
        Self { start: Some(Instant::now()) }
    }
    /// Seconds since the span opened (0 when inert).
    #[must_use]
    pub fn elapsed(&self) -> f64 {
        self.start.map_or(0.0, |t| t.elapsed().as_secs_f64())
    }
}
impl Drop for Span {
    fn drop(&mut self) {
        let Some(t) = self.start else { return };
        let total = t.elapsed().as_secs_f64();
        let (path, inner) = STACK.with(|s| {
            let mut s = s.borrow_mut();
            let path = s.iter().map(|x| x.0).collect::<Vec<_>>().join(" > ");
            let inner = s.pop().map_or(0.0, |x| x.1);
            if let Some(parent) = s.last_mut() {
                parent.1 += total;
            }
            (path, inner)
        });
        if let Ok(mut m) = TABLE.lock() {
            let e = m.entry(path).or_insert((0.0, 0));
            e.0 += (total - inner).max(0.0);
            e.1 += 1;
        }
    }
}

/// Self-check: nested spans record exclusive times that add up to the outer span's total, and
/// nothing is recorded while timing is off.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let was = on();
    set(false);
    {
        let _s = Span::new("p1-check-off");
    }
    if table().iter().any(|r| r.0.contains("p1-check-off")) {
        set(was);
        return Err("nmffi.prof: a span recorded while timing was off".into());
    }
    set(true);
    let outer_total;
    {
        let outer = Span::new("p1-check");
        {
            let _a = Span::new("a");
            std::hint::black_box((0..20_000_u64).sum::<u64>());
        }
        {
            let _b = Span::new("b");
            std::hint::black_box((0..20_000_u64).sum::<u64>());
        }
        outer_total = outer.elapsed();
    }
    set(was);
    let rows: Vec<(String, f64, u64)> = table().into_iter().filter(|r| r.0.starts_with("p1-check")).collect();
    let sum: f64 = rows.iter().map(|r| r.1).sum();
    if rows.len() != 3 || !rows.iter().any(|r| r.0 == "p1-check > a") || sum + 1e-3 < outer_total {
        return Err(format!("nmffi.prof: nested spans {rows:?} do not add up to {outer_total}"));
    }
    if let Ok(mut t) = TABLE.lock() {
        t.retain(|k, _| !k.starts_with("p1-check"));
    }
    Ok(())
}
