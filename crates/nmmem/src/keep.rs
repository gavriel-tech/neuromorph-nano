//! Retention of what grows beside the memory: the audit log's tool rows and the training store's
//! rows. Lifetimes are counted in run-days (days on which the system ran; an idle calendar day does
//! not count), with the memory tiers' lifetimes: what is working data lives as long as a temporary
//! note (7 run-days), what is evidence lives as long as `1-core` (28 run-days), and what was
//! verified is protected and never removed by age, as a memory verified right is never removed.
//! Nothing is removed for size.
//!
//! What retention never touches: the memory journal and its views (the journal keeps every event,
//! so every change stays explainable and reversible), `episodes.jsonl`, the audit log's rows other
//! than tool calls (settings, privacy, clears, retention passes), and a verified training example.
//!
//! Run-days for the audit log and the training store are the UTC days on which the audit log has
//! rows: the system ran on them.
use crate::store::fault;
use nmtype::Fault;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const MS_PER_DAY: u64 = 86_400_000;

/// Run-days a working row lives unused: the temporary notes' lifetime.
pub const WORKING_DAYS: u64 = crate::retention::TEMP_DAYS as u64;
/// Run-days an evidence row lives: `1-core`'s lifetime, the longest written.
pub const EVIDENCE_DAYS: u64 = crate::retention::CORE_DAYS as u64;

/// One thing a pass removes (or would remove).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Removal {
    pub file: PathBuf,
    pub rows: usize,
    pub bytes: u64,
    pub why: String,
}

/// What a pass found and did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub removals: Vec<Removal>,
    /// Bytes of the audit log before and after.
    pub before: u64,
    pub after: u64,
    pub applied: bool,
}

impl Report {
    #[must_use]
    pub fn rows(&self) -> usize {
        self.removals.iter().map(|r| r.rows).sum()
    }
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.removals.iter().map(|r| r.bytes).sum()
    }
}

fn day_of(ms: u64) -> u64 {
    ms / MS_PER_DAY
}

/// The UTC days on which the audit log `audit` has rows, and today: the system's run-days.
#[must_use]
pub fn run_days(audit: &Path, now_ms: u64) -> BTreeSet<u64> {
    let mut days: BTreeSet<u64> = std::fs::read_to_string(audit).map(|t| t.lines().filter_map(time_of).map(day_of).collect()).unwrap_or_default();
    days.insert(day_of(now_ms));
    days
}

/// Run-days after the day of `ms` (0 for today).
#[must_use]
pub fn age(days: &BTreeSet<u64>, ms: u64) -> u64 {
    days.range(day_of(ms) + 1..).count() as u64
}

/// The `time_ms` of an audit line without parsing the whole row.
fn time_of(line: &str) -> Option<u64> {
    let i = line.find("\"time_ms\":")? + 10;
    let rest = line.get(i..)?;
    let digits: String = rest.chars().skip_while(|c| *c == ' ').take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

fn is_tool(line: &str) -> bool {
    line.contains("\"kind\":\"tool\"") && nmjson::parse(line.as_bytes()).ok().and_then(|v| v.get("kind").ok().and_then(|k| k.string().ok().map(str::to_owned))).as_deref() == Some("tool")
}

/// The training store's rule for one row of `file` (`nmtype::examples::JUDGED`, `EXAMPLES` or
/// `PREFERENCES`) written at `at_ms`: whether it is kept. Judged sets are working data (7 run-days);
/// examples and preference signals are evidence (28 run-days); a verified example is kept.
#[must_use]
pub fn training_keeps(days: &BTreeSet<u64>, file: &str, at_ms: u64, verdict: &str) -> bool {
    let a = age(days, at_ms);
    if file == nmtype::examples::JUDGED {
        a <= WORKING_DAYS
    } else {
        verdict == "verified" || a <= EVIDENCE_DAYS
    }
}

/// Writes `lines` to `path` through a temporary file renamed into place, keeping any tail appended
/// since the file was read (`len` bytes), so a row a serving process wrote meanwhile is not lost.
fn rewrite(path: &Path, lines: &[&str], len: u64) -> Result<(), Fault> {
    let mut text = String::new();
    for l in lines {
        text.push_str(l);
        text.push('\n');
    }
    let now = std::fs::read(path).map_err(fault)?;
    if let Some(tail) = now.get(usize::try_from(len).map_err(fault)?..) {
        text.push_str(&String::from_utf8_lossy(tail));
    }
    let tmp = path.with_extension("keep-tmp");
    std::fs::write(&tmp, text.as_bytes()).map_err(fault)?;
    std::fs::rename(&tmp, path).map_err(fault)
}

/// One retention pass over the audit log `memory/audit.jsonl` of the Nano folder `home`: tool rows
/// older than `EVIDENCE_DAYS` run-days go; every other row stays. With `apply` false nothing is
/// changed and the report is what a pass would remove.
/// # Errors
/// The log cannot be read or written while applying.
pub fn pass(home: &Path, now_ms: u64, apply: bool) -> Result<Report, Fault> {
    pass_days(home, now_ms, apply, EVIDENCE_DAYS)
}

/// `pass` with the audit log's tool rows kept `days` run-days (2026-11-26: the `audit.keep_days`
/// control; at least one run-day). Setting and other rows are never removed (they back Undo).
/// # Errors
/// The audit log cannot be rewritten.
pub fn pass_days(home: &Path, now_ms: u64, apply: bool, days_kept: u64) -> Result<Report, Fault> {
    let keep_days = days_kept.max(1);
    let path = home.join("memory").join("audit.jsonl");
    let mut rep = Report { applied: apply, ..Report::default() };
    let Ok(text) = std::fs::read_to_string(&path) else { return Ok(rep) };
    let days = run_days(&path, now_ms);
    let len = text.len() as u64;
    let lines: Vec<&str> = text.lines().collect();
    let drop: Vec<bool> = lines.iter().map(|x| is_tool(x) && time_of(x).is_some_and(|t| age(&days, t) > keep_days)).collect();
    let rows = drop.iter().filter(|d| **d).count();
    let bytes: u64 = lines.iter().zip(&drop).filter(|(_, d)| **d).map(|(x, _)| x.len() as u64 + 1).sum();
    rep.before = len;
    rep.after = len - bytes;
    if rows > 0 {
        rep.removals.push(Removal { file: path.clone(), rows, bytes, why: format!("tool rows older than {keep_days} run-days") });
        if apply {
            let keep: Vec<&str> = lines.iter().zip(&drop).filter(|(_, d)| !**d).map(|(x, _)| *x).collect();
            rewrite(&path, &keep, len)?;
            rep.after = std::fs::metadata(&path).map_or(0, |m| m.len());
        }
    }
    Ok(rep)
}

/// Section check: on a scratch folder whose audit log spans run-days 0, 1 and 30 (with idle days
/// between that do not count) and one past the evidence lifetime, a planned pass changes nothing;
/// applying it removes the old tool row only (a settings row of the same day stays); a second pass
/// removes nothing; the training rule keeps a verified example of any age, drops an unverified one
/// past 28 run-days and a judged set past 7, and counts only run-days.
/// # Errors
/// One fault line.
pub(crate) fn check() -> Result<(), String> {
    let home = nmtype::check_dir("nmmem-keep").map_err(|e| e.to_string())?;
    std::fs::create_dir_all(home.join("memory")).map_err(|e| e.to_string())?;
    let day = MS_PER_DAY;
    // Run-days: 0, then 2 to 29, then today (100); the idle days between count nothing. Day 0 is 29
    // run-days old, day 2 is 28.
    let mut audit = vec![
        format!(r#"{{"id":"a1","kind":"tool","tool":"nm_context_pack","time_ms":{}}}"#, 10),
        format!(r#"{{"id":"a2","kind":"setting","key":"mode","time_ms":{}}}"#, 20),
    ];
    for d in 2..30 {
        audit.push(format!(r#"{{"id":"d{d}","kind":"tool","tool":"nm_context_pack","time_ms":{}}}"#, d * day + 5));
    }
    std::fs::write(home.join("memory").join("audit.jsonl"), audit.join("\n") + "\n").map_err(|e| e.to_string())?;
    let now = 100 * day + 1000;
    let before = std::fs::read(home.join("memory").join("audit.jsonl")).map_err(|e| e.to_string())?;
    let planned = pass(&home, now, false).map_err(|e| e.to_string())?;
    if std::fs::read(home.join("memory").join("audit.jsonl")).map_err(|e| e.to_string())? != before || planned.rows() != 1 {
        return Err(format!("keep: a planned pass changed the log or found {} rows (want 1)", planned.rows()));
    }
    let done = pass(&home, now, true).map_err(|e| e.to_string())?;
    let after = std::fs::read_to_string(home.join("memory").join("audit.jsonl")).map_err(|e| e.to_string())?;
    if done.removals != planned.removals || after.contains("\"a1\"") || !after.contains("\"a2\"") || !after.contains("\"d2\"") {
        return Err(format!("keep: audit after the pass: {after}"));
    }
    if pass(&home, now, false).map_err(|e| e.to_string())?.rows() != 0 {
        return Err("keep: a second pass would remove more".into());
    }
    let days = run_days(&home.join("memory").join("audit.jsonl"), now);
    let ex = nmtype::examples::EXAMPLES;
    if !training_keeps(&days, ex, 10, "verified") || training_keeps(&days, ex, 10, "passed-over") || !training_keeps(&days, ex, 2 * day, "passed-over") {
        return Err("keep: the training rule on examples".into());
    }
    if training_keeps(&days, nmtype::examples::JUDGED, 20 * day, "") || !training_keeps(&days, nmtype::examples::JUDGED, 25 * day, "") {
        return Err("keep: the training rule on judged sets".into());
    }
    Ok(())
}
