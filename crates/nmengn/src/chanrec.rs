//! Channel recording for a MicroModel's map from its real jobs (lane D, 2026-11-19).
//!
//! With `NANO_CHANNEL_RECORD=<file>` set, every run of the MicroModel named by `NANO_CHANNEL_POP`
//! (default `qwen3-0.6b`, the reader) adds, per feed-forward channel at each layer's `swigate`, the
//! sum and the sum of squares of its output over every token, under the job the run belongs to.
//! The job is the outermost tag set by `job` on the running thread (the drones' scan, the
//! tentacles' integration read, the joint read and its yes or no, the request's read); a run with
//! no tag counts as `other`. The file is rewritten at most every 20 seconds and by `flush`; it holds
//! no text, only per-channel numbers. Off (the default), nothing is read back and nothing changes.
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Per job, per layer: tokens counted, the channels' sums and sums of squares.
type Table = BTreeMap<String, BTreeMap<usize, (u64, Vec<f64>, Vec<f64>)>>;

struct Rec {
    file: std::path::PathBuf,
    pop: String,
    table: Mutex<(Table, Instant)>,
}

fn rec() -> Option<&'static Rec> {
    static R: OnceLock<Option<Rec>> = OnceLock::new();
    R.get_or_init(|| {
        let file = std::env::var("NANO_CHANNEL_RECORD").ok().filter(|f| !f.is_empty())?;
        let pop = std::env::var("NANO_CHANNEL_POP").unwrap_or_else(|_| "qwen3-0.6b".to_owned());
        Some(Rec { file: file.into(), pop, table: Mutex::new((BTreeMap::new(), Instant::now())) })
    })
    .as_ref()
}

thread_local! {
    static JOB: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
}

/// The job tag held while the guard lives (the outermost tag counts: a piece encoding inside a
/// joint read counts for the joint read).
pub struct Job(bool);
impl Drop for Job {
    fn drop(&mut self) {
        if self.0 {
            JOB.with(|j| {
                j.borrow_mut().pop();
            });
        }
    }
}

/// Tags the runs on this thread with `name` while the returned guard lives (no effect when
/// recording is off).
#[must_use]
pub fn job(name: &'static str) -> Job {
    if rec().is_none() && !crate::pmap::any_on() {
        return Job(false);
    }
    JOB.with(|j| {
        let mut j = j.borrow_mut();
        if j.is_empty() {
            j.push(name);
            Job(true)
        } else {
            Job(false)
        }
    })
}

/// Whether recording is on for the population `pop`.
#[must_use]
pub fn on(pop: &str) -> bool {
    rec().is_some_and(|r| r.pop == pop)
}

/// The active job tag on the current thread (or `"other"` when untagged).
#[must_use]
pub fn current_job() -> &'static str {
    JOB.with(|j| j.borrow().first().copied()).unwrap_or("other")
}

/// Adds one `swigate` output (`width` channels per token) at `layer`.
pub fn note(layer: usize, width: usize, y: &[f32]) {
    let Some(r) = rec() else { return };
    if width == 0 || !y.len().is_multiple_of(width) {
        return;
    }
    let job = JOB.with(|j| j.borrow().first().copied()).unwrap_or("other");
    let Ok(mut g) = r.table.lock() else { return };
    let e = g.0.entry(job.to_owned()).or_default().entry(layer).or_insert_with(|| (0, vec![0.0; width], vec![0.0; width]));
    if e.1.len() != width {
        return;
    }
    for row in y.chunks_exact(width) {
        for ((s, q), x) in e.1.iter_mut().zip(e.2.iter_mut()).zip(row) {
            let x = f64::from(*x);
            *s += x;
            *q += x * x;
        }
        e.0 += 1;
    }
    if g.1.elapsed().as_secs() >= 20 {
        g.1 = Instant::now();
        write(&r.file, &r.pop, &g.0);
    }
}

/// Writes the table now.
pub fn flush() {
    if let Some(r) = rec()
        && let Ok(g) = r.table.lock()
    {
        write(&r.file, &r.pop, &g.0);
    }
}

fn write(file: &std::path::Path, pop: &str, t: &Table) {
    use nmjson::Value;
    let num = |x: f64| Value::Number(format!("{x:e}"));
    let jobs: BTreeMap<String, Value> = t
        .iter()
        .map(|(job, layers)| {
            let l: BTreeMap<String, Value> = layers
                .iter()
                .map(|(k, (n, s, q))| {
                    (
                        format!("L{k}"),
                        Value::Object(
                            [
                                ("tokens".to_owned(), Value::Number(n.to_string())),
                                ("sum".to_owned(), Value::Array(s.iter().map(|x| num(*x)).collect())),
                                ("sq".to_owned(), Value::Array(q.iter().map(|x| num(*x)).collect())),
                            ]
                            .into_iter()
                            .collect(),
                        ),
                    )
                })
                .collect();
            (job.clone(), Value::Object(l))
        })
        .collect();
    let v = Value::Object([("model".to_owned(), Value::String(pop.to_owned())), ("jobs".to_owned(), Value::Object(jobs))].into_iter().collect());
    if let Ok(text) = v.encode() {
        let tmp = file.with_extension("tmp");
        if std::fs::write(&tmp, text).is_ok() {
            let _ = std::fs::rename(&tmp, file);
        }
    }
}
