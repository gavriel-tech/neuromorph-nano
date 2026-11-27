//! Continuous delta consolidation (guide 17.3, Comprehensive Vision 18.4): a poll thread
//! re-syncs the studied tree every N seconds. Sync compares content hashes, so exFAT's
//! whole-second file times cannot hide an edit; a change publishes a new substrate version,
//! invalidates older packets and records which files changed (stale-claim marks). The wait
//! between polls is a channel timeout, so `stop` ends the thread at once.
use crate::forage::Organism;
use nmjson::Value;
use std::sync::Mutex;
use std::sync::mpsc::{RecvTimeoutError, Sender, channel};
use std::thread::JoinHandle;
use std::time::Duration;

/// (interval ms, polls done, last substrate version seen, wall ms of the last poll).
static STATE: Mutex<(u64, u64, u64, u64)> = Mutex::new((0, 0, 0, 0));

/// A running poll thread; dropping it stops the thread.
pub struct Poller {
    stop: Option<Sender<()>>,
    handle: Option<JoinHandle<()>>,
}
impl Poller {
    /// Stop the thread and wait for it.
    pub fn stop(mut self) {
        self.halt();
    }
    fn halt(&mut self) {
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
impl Drop for Poller {
    fn drop(&mut self) {
        self.halt();
    }
}

fn ms(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

/// Run `step` every `every` until stopped; `step` returns the version it left.
pub fn spawn(every: Duration, mut step: impl FnMut() -> Option<u64> + Send + 'static) -> Poller {
    let (tx, rx) = channel::<()>();
    if let Ok(mut s) = STATE.lock() {
        *s = (ms(every), 0, s.2, s.3);
    }
    let handle = std::thread::spawn(move || {
        while let Err(RecvTimeoutError::Timeout) = rx.recv_timeout(every) {
            let t = std::time::Instant::now();
            let v = step();
            if let Ok(mut s) = STATE.lock() {
                s.1 += 1;
                if let Some(v) = v {
                    s.2 = v;
                }
                s.3 = ms(t.elapsed());
            }
        }
    });
    Poller { stop: Some(tx), handle: Some(handle) }
}

/// Start polling the organism's studied tree.
#[must_use]
pub fn start(org: &'static Organism, every: Duration) -> Poller {
    spawn(every, move || org.sync().ok().map(|(r, _)| r.version))
}

/// Poll state for `nm_status`.
#[must_use]
pub fn status() -> Value {
    let (every, polls, version, last) = STATE.lock().map_or((0, 0, 0, 0), |s| *s);
    let n = |x: u64| Value::Number(x.to_string());
    Value::Object(
        [("poll_ms", n(every)), ("polls", n(polls)), ("last_sync_version", n(version)), ("last_poll_ms", n(last))]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect(),
    )
}

/// Section check: a 20 ms poller steps at least twice in 150 ms and stops at once.
/// # Errors
/// One line naming what failed.
pub fn check() -> Result<(), String> {
    let count = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let c = std::sync::Arc::clone(&count);
    let p = spawn(Duration::from_millis(20), move || Some(c.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1));
    // Wait 150 ms on a channel nobody sends on (the sender stays alive until after the wait).
    let (tx, rx) = channel::<()>();
    let _ = rx.recv_timeout(Duration::from_millis(150));
    drop(tx);
    let t = std::time::Instant::now();
    p.stop();
    let stopped_in = t.elapsed();
    let steps = count.load(std::sync::atomic::Ordering::SeqCst);
    if steps < 2 || stopped_in > Duration::from_secs(2) {
        return Err(format!("poller stepped {steps} times, stop took {stopped_in:?}"));
    }
    Ok(())
}
