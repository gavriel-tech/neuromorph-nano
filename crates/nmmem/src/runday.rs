//! The run-day counter (2026-11-12, memory revision 2, section 7; direction of Luis Jake Gabriel III:
//! lifetimes are counted in days on which the system ran, not calendar days). `rundays.json` holds
//! how many distinct UTC days the organism has run on and the last such day; several sessions on
//! one day count once, and days with no session do not count.
use crate::store::{fault, get, num, obj, put_safe, s, u};
use nmtype::Fault;
use std::path::{Path, PathBuf};

const MS_PER_DAY: u64 = 86_400_000;

/// The counter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunDays {
    path: PathBuf,
    /// Run-days so far (the current one included once ticked).
    pub count: u64,
    /// UTC day number (days since 1970-01-01) of the last run-day.
    pub last_day: u64,
}

impl RunDays {
    /// Open the counter of the memory folder `dir` (zero when absent).
    #[must_use]
    pub fn open(dir: &Path) -> Self {
        let path = dir.join("rundays.json");
        let v = get(&path);
        Self { count: v.as_ref().map_or(0, |v| num(v, "count")), last_day: v.as_ref().map_or(0, |v| num(v, "last_day")), path }
    }
    /// Count today if it is a new run-day; returns the current run-day number (1 on the first).
    /// # Errors
    /// The counter file cannot be written.
    pub fn tick(&mut self, now_ms: u64) -> Result<u64, Fault> {
        let day = now_ms / MS_PER_DAY;
        if self.count == 0 || day > self.last_day {
            self.count += 1;
            self.last_day = day;
            put_safe(
                &self.path,
                &obj(vec![
                    ("record", s("memory-rundays")),
                    ("count", u(self.count)),
                    ("last_day", u(self.last_day)),
                    ("note", s("days on which the system ran; idle days do not count")),
                ]),
            )?;
        } else if day < self.last_day {
            return Err(fault(format!("clock went back: day {day} before the last run-day {}", self.last_day)));
        }
        Ok(self.count)
    }
    /// Run-days elapsed since run-day `then` (0 when `then` is today or later).
    #[must_use]
    pub fn since(&self, then: u64) -> u64 {
        self.count.saturating_sub(then)
    }
}

/// Section check: several ticks on one day count once; a later day counts; idle days between
/// count nothing; the count survives reopening; a clock going back is refused.
/// # Errors
/// One fault line.
pub(crate) fn check() -> Result<(), String> {
    let dir = nmtype::check_dir("nmmem-runday").map_err(|e| e.to_string())?;
    let d0 = 20_000 * MS_PER_DAY;
    let mut r = RunDays::open(&dir);
    let a = r.tick(d0 + 1_000).map_err(|e| e.to_string())?;
    let b = r.tick(d0 + 50_000).map_err(|e| e.to_string())?;
    let c = r.tick(d0 + 9 * MS_PER_DAY).map_err(|e| e.to_string())?;
    if (a, b, c) != (1, 1, 2) {
        return Err(format!("nmmem.runday: counts {a} {b} {c} (want 1 1 2: nine idle days count once)"));
    }
    let mut again = RunDays::open(&dir);
    if again.count != 2 || again.tick(d0 + 9 * MS_PER_DAY + 5).map_err(|e| e.to_string())? != 2 || again.since(1) != 1 {
        return Err("nmmem.runday: reopen".into());
    }
    if again.tick(d0).is_ok() {
        return Err("nmmem.runday: a clock going back was accepted".into());
    }
    Ok(())
}
