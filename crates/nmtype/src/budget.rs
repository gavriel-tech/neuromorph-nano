//! One processor budget for the whole organism (tentacle master plan 3.12, Phase C1, 2026-11-18).
//! Before it, five pools each took every thread (`nmengn::cpu`, `nmpod::evidence`, three sites of
//! `nmapi::drone_windows`, `nmnte::Runtime`), so a scan running beside another stage ran on twice
//! or more the threads the processor has. Now:
//! - the thread count is read once from `available_parallelism` (`hardware`), under the user's
//!   live `limits.cpu_threads` when one is set (`total`; Roadmap Phase 4.5, 2026-11-25);
//! - a parallel stage asks for extra workers and gets what is free now (`Grant`, never blocking);
//!   the calling thread always works too, so a stage nested in another, or one that finds every
//!   thread taken, runs on its own thread instead of adding threads; the grant is returned when
//!   the stage ends (stage by stage);
//! - the work is cut into partitions whose count the caller fixes from the work alone (never from
//!   the grant), and each partition's result is computed alone and placed in partition order, so
//!   sums are the same whatever the load (`run`);
//! - two classes: a question's stages are foreground; idle work (the intake pass of master plan
//!   3.2) is background (`background`). Foreground is lent every thread foreground does not hold
//!   itself; background only what is free, and a background worker gives its thread back before
//!   its next partition when the threads are over-lent, so a question waits behind background
//!   work no longer than one partition (one kernel, about 50 ms). The class passes to nested
//!   stages and to the workers of a stage.
//!
//! Reference: the work-stealing pools of Rayon and OpenMP's `OMP_THREAD_LIMIT` keep one global
//! thread count and run nested parallel regions serially when it is spent; the fixed partition
//! count is the reproducible-reduction rule of oneTBB's `parallel_deterministic_reduce`; yielding
//! at partition boundaries is cooperative preemption as in Go's scheduler before 1.14.
use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

static TOTAL: OnceLock<usize> = OnceLock::new();
/// Extra workers lent out now to foreground and to background stages (calling threads are not
/// counted).
static FG: AtomicUsize = AtomicUsize::new(0);
static BG: AtomicUsize = AtomicUsize::new(0);
/// Foreground stages running now.
static ACTIVE: AtomicUsize = AtomicUsize::new(0);

/// A stage's class.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Class {
    /// A question's stages.
    #[default]
    Foreground,
    /// Idle work, yielding to foreground at partition boundaries.
    Background,
}

thread_local! {
    static CLASS: Cell<Class> = const { Cell::new(Class::Foreground) };
}

/// The class of stages started on this thread.
#[must_use]
pub fn class() -> Class {
    CLASS.with(Cell::get)
}

/// Run `f` with every stage it starts (on this thread and their workers) in the background class.
pub fn background<R>(f: impl FnOnce() -> R) -> R {
    let before = CLASS.with(|c| c.replace(Class::Background));
    let r = f();
    CLASS.with(|c| c.set(before));
    r
}

/// The processor's thread count, read once (the hardware, whatever the user's limit).
#[must_use]
pub fn hardware() -> usize {
    *TOTAL.get_or_init(|| std::thread::available_parallelism().map_or(4, std::num::NonZero::get))
}

/// The threads the budget works with now: the processor's count, under the user's
/// `limits.cpu_threads` when one is set. Roadmap Phase 4.5 (2026-11-25): the limit is read live
/// (`crate::limits::current`), so a changed limit applies to the next grant with no restart;
/// grants already lent finish as they are.
#[must_use]
pub fn total() -> usize {
    let hw = hardware();
    crate::limits::current().cpu_threads.map_or(hw, |c| c.clamp(1, hw))
}

fn cap() -> usize {
    total().saturating_sub(1)
}

const fn counter(class: Class) -> &'static AtomicUsize {
    match class {
        Class::Foreground => &FG,
        Class::Background => &BG,
    }
}

/// True when more extra workers are lent than the processor has threads beside the callers'
/// (a foreground stage took threads background holds): background workers give theirs back.
fn crowded() -> bool {
    FG.load(Ordering::Relaxed) + BG.load(Ordering::Relaxed) > cap()
}

/// Extra workers lent to one stage; returned on drop.
#[derive(Debug)]
pub struct Grant {
    extra: AtomicUsize,
    class: Class,
}
impl Grant {
    /// Up to `want` extra workers for a stage of this thread's class, as many as are free now.
    #[must_use]
    pub fn take(want: usize) -> Self {
        Self::take_as(class(), want)
    }
    /// Up to `want` extra workers for a stage of `class`: foreground gets what foreground does not
    /// hold; background gets only what neither holds (0 when every thread is in use).
    #[must_use]
    pub fn take_as(class: Class, want: usize) -> Self {
        let c = counter(class);
        let mut now = c.load(Ordering::Relaxed);
        let limit = cap();
        loop {
            let other = match class {
                Class::Foreground => 0,
                Class::Background => FG.load(Ordering::Relaxed),
            };
            let give = want.min(limit.saturating_sub(now + other));
            if give == 0 {
                return Self { extra: AtomicUsize::new(0), class };
            }
            match c.compare_exchange_weak(now, now + give, Ordering::AcqRel, Ordering::Relaxed) {
                Ok(_) => return Self { extra: AtomicUsize::new(give), class },
                Err(seen) => now = seen,
            }
        }
    }
    /// Extra workers held now.
    #[must_use]
    pub fn extra(&self) -> usize {
        self.extra.load(Ordering::Relaxed)
    }
    /// Take one worker back (a background worker returning after foreground ended), when one is
    /// free; false otherwise.
    fn retake_one(&self) -> bool {
        let other = FG.load(Ordering::Relaxed);
        let limit = cap();
        let ok = BG.try_update(Ordering::AcqRel, Ordering::Relaxed, |x| (x + other < limit).then_some(x + 1)).is_ok();
        if ok {
            self.extra.fetch_add(1, Ordering::AcqRel);
        }
        ok
    }
    /// Give one worker back early (a background worker leaving); false when none is held.
    fn give_back_one(&self) -> bool {
        let ok = self.extra.try_update(Ordering::AcqRel, Ordering::Relaxed, |x| x.checked_sub(1)).is_ok();
        if ok {
            counter(self.class).fetch_sub(1, Ordering::AcqRel);
        }
        ok
    }
}
impl Drop for Grant {
    fn drop(&mut self) {
        let n = self.extra.swap(0, Ordering::AcqRel);
        if n > 0 {
            counter(self.class).fetch_sub(n, Ordering::AcqRel);
        }
    }
}

/// One foreground stage running (dropped when it ends).
struct Active;
impl Active {
    fn enter() -> Self {
        ACTIVE.fetch_add(1, Ordering::AcqRel);
        Self
    }
}
impl Drop for Active {
    fn drop(&mut self) {
        ACTIVE.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Extra workers lent out now: (foreground, background).
#[must_use]
pub fn lent() -> (usize, usize) {
    (FG.load(Ordering::Relaxed), BG.load(Ordering::Relaxed))
}

/// Run `f` on partitions `0..parts` with the calling thread and as many extra workers as the
/// budget lends (at most `parts - 1`), in this thread's class; results in partition order. `parts`
/// is fixed by the caller from the work alone, so the result never depends on the load. A
/// partition that panics gives `None`; the others still run. In the background class an extra
/// worker leaves before its next partition when the threads are over-lent, and the calling thread
/// waits while foreground holds every thread.
pub fn run<R: Send>(parts: usize, f: impl Fn(usize) -> R + Sync) -> Vec<Option<R>> {
    if parts == 0 {
        return Vec::new();
    }
    let cls = class();
    let _active = (cls == Class::Foreground).then(Active::enter);
    let grant = Grant::take_as(cls, parts - 1);
    let next = AtomicUsize::new(0);
    let nap = || std::thread::park_timeout(std::time::Duration::from_millis(1));
    let work = |extra: bool| {
        CLASS.with(|c| c.set(cls));
        let mut got = Vec::new();
        loop {
            // Background yields at the partition boundary: an extra worker gives its thread back
            // while any foreground stage runs (or threads are over-lent) and returns after; the
            // calling thread waits while foreground runs.
            if cls == Class::Background {
                let busy = || ACTIVE.load(Ordering::Relaxed) > 0 || crowded();
                if extra && busy() && grant.give_back_one() {
                    while next.load(Ordering::Relaxed) < parts && !(ACTIVE.load(Ordering::Relaxed) == 0 && grant.retake_one()) {
                        nap();
                    }
                }
                while !extra && ACTIVE.load(Ordering::Relaxed) > 0 && next.load(Ordering::Relaxed) < parts {
                    nap();
                }
            }
            let p = next.fetch_add(1, Ordering::Relaxed);
            if p >= parts {
                break;
            }
            got.push((p, catch_unwind(AssertUnwindSafe(|| f(p))).ok()));
        }
        got
    };
    let mut out: Vec<Option<R>> = (0..parts).map(|_| None).collect();
    let mut place = |got: Vec<(usize, Option<R>)>| {
        for (p, r) in got {
            if let Some(slot) = out.get_mut(p) {
                *slot = r;
            }
        }
    };
    let n = grant.extra();
    if n == 0 {
        place(work(false));
    } else {
        let all: Vec<Vec<(usize, Option<R>)>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..n).map(|_| sc.spawn(|| work(true))).collect();
            let mine = work(false);
            let mut all = vec![mine];
            all.extend(hs.into_iter().map(|h| h.join().unwrap_or_default()));
            all
        });
        for got in all {
            place(got);
        }
    }
    CLASS.with(|c| c.set(cls));
    drop(grant);
    out
}

/// `run` over `n` items cut into at most `parts` runs of equal length (the last shorter): each
/// run maps its items in order with `f`; the results come back flat in item order, or `None`
/// when a run was lost.
pub fn map<R: Send>(n: usize, parts: usize, f: impl Fn(usize) -> R + Sync) -> Option<Vec<R>> {
    let size = n.div_ceil(parts.max(1)).max(1);
    let runs = n.div_ceil(size);
    let got = run(runs, |p| (p * size..((p + 1) * size).min(n)).map(&f).collect::<Vec<R>>());
    let mut out = Vec::with_capacity(n);
    for r in got {
        out.extend(r?);
    }
    Some(out)
}

/// Section check: grants never pass the total; nested stages get nothing while the outer holds
/// every worker; background gets nothing while foreground holds every thread, and foreground is
/// lent what background holds; results are in partition order and equal whatever was lent; a
/// panicking partition is isolated.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let t = total();
    let a = Grant::take_as(Class::Foreground, usize::MAX);
    if a.extra() > cap() {
        return Err(format!("nmtype.budget: lent {} of {t}", a.extra()));
    }
    let inner = Grant::take_as(Class::Foreground, 4);
    let back = Grant::take_as(Class::Background, 4);
    let nested_ok = a.extra() < cap() || (inner.extra() == 0 && back.extra() == 0);
    drop((inner, back));
    let under: Vec<Option<usize>> = run(9, |p| p * p);
    drop(a);
    let b = Grant::take_as(Class::Background, usize::MAX);
    let fg = Grant::take_as(Class::Foreground, usize::MAX);
    let lend_ok = fg.extra() == cap() && (b.extra() == 0 || crowded());
    drop((fg, b));
    let free: Vec<Option<usize>> = run(9, |p| p * p);
    let bg: Vec<Option<usize>> = background(|| run(9, |p| p * p));
    if !nested_ok || !lend_ok || under != free || bg != free || free.iter().enumerate().any(|(p, r)| *r != Some(p * p)) {
        return Err(format!("nmtype.budget: nested {nested_ok}, lending {lend_ok}, partition order"));
    }
    if class() != Class::Foreground || lent() != (0, 0) {
        return Err("nmtype.budget: class or grants not restored".into());
    }
    let hurt = run(3, |p| {
        assert!(p != 1, "isolated test panic");
        p
    });
    if hurt != vec![Some(0), None, Some(2)] {
        return Err("nmtype.budget: a panicking partition was not isolated".into());
    }
    let m = map(10, 3, |i| i + 1);
    if m != Some((1..=10).collect()) {
        return Err("nmtype.budget: map order".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn budget_check() {
        assert_eq!(super::check(), Ok(()));
    }

    // Roadmap Phase 4.5 (2026-11-25): the user's thread limit is a ceiling, never above the
    // hardware, never below one thread; no grant is taken here (the check above counts them).
    #[test]
    fn budget_total_respects_the_thread_limit() {
        let hw = super::hardware();
        let at = |c: Option<usize>| {
            let _g = crate::limits::scoped(crate::limits::Limits { cpu_threads: c, ..crate::limits::Limits::AUTO });
            (super::total(), super::cap())
        };
        assert_eq!(at(None), (hw, hw - 1));
        assert_eq!(at(Some(1)), (1, 0));
        assert_eq!(at(Some(hw + 64)), (hw, hw - 1));
        if hw > 2 {
            assert_eq!(at(Some(2)), (2, 1));
        }
    }
}
