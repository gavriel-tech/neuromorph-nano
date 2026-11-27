//! The idle learning orchestrator (roadmap 2026-11-23, Phase 2 item 2). One cycle runs its learners
//! in a fixed order (G3's prior replay, the preference cartridge fit, the continual learning engine),
//! each behind its own measured held-out gate, and records per cycle, under one id, which learners
//! ran, what each measured and whether its candidate was promoted.
//!
//! When it runs: once the verified examples settled into the training store since the last cycle
//! reach a threshold (`threshold`, default `DEFAULT_THRESHOLD`, `NANO_LEARN_THRESHOLD` to change it)
//! and the serving process is idle (the caller's test), or on demand (`POST /api/learning/cycle`).
//!
//! Rollback: before a learner runs, every file and folder its state lives in (`Learner::state_paths`)
//! is copied into the cycle's snapshot folder (`<memory>/learning/rollback/<cycle id>/<learner>/`,
//! or the learner's own folder when its state must stay in its own store, as muscle memory does).
//! Rolling a cycle back restores every learner that cycle touched to the state it had before, and
//! files the cycle created are removed. A folder is snapshotted by its top-level files only.
//!
//! Records: `<memory>/learning/cycles.jsonl`, one row per cycle and one per rollback; the verified
//! count at the last cycle in `<memory>/learning/last-cycle.json`. Behavioural numbers only: no
//! question, passage or code text is written here.
use crate::engine::ContinuousLearningEngine;
use crate::unit::{n, obj, s};
use nmjson::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// Folder of the orchestrator's records under a project memory folder.
pub const DIR: &str = "learning";
/// One row per cycle and per rollback.
pub const CYCLES: &str = "cycles.jsonl";
/// The verified example count when the last cycle ran.
pub const MARK: &str = "last-cycle.json";
/// Snapshot folder under `DIR`.
pub const ROLLBACK: &str = "rollback";
/// Verified examples since the last cycle that make a cycle due, by default.
pub const DEFAULT_THRESHOLD: usize = 20;
/// The variable that changes the threshold.
pub const THRESHOLD_ENV: &str = "NANO_LEARN_THRESHOLD";
/// A cycle lock older than this is taken as left by a stopped process (ms).
const STALE_LOCK_MS: u64 = 10 * 60 * 1000;
/// Verdicts of the training store that count as verified outcomes.
pub const VERIFIED: [&str; 2] = ["verified", "corrected"];

fn io(p: &Path, e: &std::io::Error) -> String {
    format!("{}: {e}", p.display())
}
fn fnum(x: f64) -> Value {
    Value::Number(format!("{x:.6}"))
}
fn text(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::string).ok().map(str::to_owned)
}

/// The threshold of verified examples since the last cycle: `NANO_LEARN_THRESHOLD` when it holds a
/// whole number above zero, else `DEFAULT_THRESHOLD`.
#[must_use]
pub fn threshold() -> usize {
    parse_threshold(std::env::var(THRESHOLD_ENV).ok().as_deref())
}

/// The threshold a setting's text gives (`DEFAULT_THRESHOLD` for none, zero or text that is not a
/// whole number).
#[must_use]
pub fn parse_threshold(v: Option<&str>) -> usize {
    v.and_then(|t| t.trim().parse::<usize>().ok()).filter(|x| *x > 0).unwrap_or(DEFAULT_THRESHOLD)
}

/// Verified outcomes in the training store (`training/examples.jsonl` rows whose verdict is one
/// of `VERIFIED`) plus explicit positive `nm_feedback` verdicts in `outcomes.jsonl` (deduplicated
/// against training-store rows so hook-settled and MCP-only clients both trigger idle learning
/// cycles without double-counting).
#[must_use]
pub fn verified_count(memory: &Path) -> usize {
    let mut store_ids = BTreeSet::new();
    let mut store_verified = 0_usize;
    for r in nmtype::examples::read_rows(memory, nmtype::examples::EXAMPLES).unwrap_or_default() {
        if text(&r, "verdict").is_some_and(|v| VERIFIED.contains(&v.as_str())) {
            store_verified += 1;
            if let Some(id) = text(&r, "id") {
                store_ids.insert(id);
            }
            if let Some(pack) = text(&r, "pack") {
                store_ids.insert(pack);
            }
        }
    }
    let mut fb_episodes = BTreeSet::new();
    if let Ok(raw) = std::fs::read_to_string(memory.join("outcomes.jsonl")) {
        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let Ok(v) = nmjson::parse(trimmed.as_bytes()) else { continue };
            if matches!(v.get("correct"), Ok(Value::Bool(true)))
                && let Some(ep) = text(&v, "episode")
                && !ep.is_empty()
                && !store_ids.contains(&ep)
            {
                fb_episodes.insert(ep);
            }
        }
    }
    store_verified + fb_episodes.len()
}

/// The verified count recorded when the last cycle ran (0 before the first).
#[must_use]
pub fn verified_at_last_cycle(memory: &Path) -> usize {
    std::fs::read(memory.join(DIR).join(MARK))
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .and_then(|v| v.get("verified").and_then(Value::integer).ok())
        .and_then(|x| usize::try_from(x).ok())
        .unwrap_or(0)
}

/// Verified examples settled since the last cycle (a cleared store counts from zero again).
#[must_use]
pub fn verified_since_last_cycle(memory: &Path) -> usize {
    let now = verified_count(memory);
    let last = verified_at_last_cycle(memory);
    if now < last { now } else { now - last }
}

/// Whether a cycle is due: at least `threshold` verified examples since the last cycle.
#[must_use]
pub fn due(memory: &Path, threshold: usize) -> bool {
    verified_since_last_cycle(memory) >= threshold.max(1)
}

/// What one learner did in a cycle.
#[derive(Clone, Debug, PartialEq)]
pub struct LearnerReport {
    pub name: String,
    /// Whether it ran (a learner that could not run here says why in `note`).
    pub ran: bool,
    /// The gate it was judged by.
    pub gate: String,
    /// Held-out items scored, and the held-out score of the current state and of the candidate.
    pub held_out: usize,
    pub current: Option<f64>,
    pub candidate: Option<f64>,
    /// Whether its candidate was promoted.
    pub promoted: bool,
    pub note: String,
}

impl LearnerReport {
    /// A learner that did not run, and why.
    #[must_use]
    pub fn not_run(name: &str, why: &str) -> Self {
        Self { name: name.to_owned(), ran: false, gate: String::new(), held_out: 0, current: None, candidate: None, promoted: false, note: why.to_owned() }
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let opt = |x: Option<f64>| x.map_or(Value::Null, fnum);
        obj(vec![
            ("name", s(&self.name)),
            ("ran", Value::Bool(self.ran)),
            ("gate", s(&self.gate)),
            ("held_out", n(self.held_out as u64)),
            ("current", opt(self.current)),
            ("candidate", opt(self.candidate)),
            ("promoted", Value::Bool(self.promoted)),
            ("note", s(&self.note)),
        ])
    }
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        let f = |k: &str| match v.get(k) {
            Ok(Value::Number(t)) => t.parse::<f64>().ok(),
            _ => None,
        };
        Self {
            name: text(v, "name").unwrap_or_default(),
            ran: matches!(v.get("ran"), Ok(Value::Bool(true))),
            gate: text(v, "gate").unwrap_or_default(),
            held_out: v.get("held_out").and_then(Value::integer).ok().and_then(|x| usize::try_from(x).ok()).unwrap_or(0),
            current: f("current"),
            candidate: f("candidate"),
            promoted: matches!(v.get("promoted"), Ok(Value::Bool(true))),
            note: text(v, "note").unwrap_or_default(),
        }
    }
}

/// One learner of the cycle.
pub trait Learner {
    /// Its name in the cycle's record (`g3-priors`, `preference-cartridge`, `engine`).
    fn name(&self) -> &str;
    /// The files and folders its state lives in, snapshotted before it runs.
    fn state_paths(&self) -> Vec<PathBuf>;
    /// Where its snapshots go, when they must stay in its own store (`None`: the project memory
    /// folder's `learning/rollback/`).
    fn snapshot_root(&self) -> Option<PathBuf> {
        None
    }
    /// Fit a candidate, measure it on held-out data against the current state, and promote it only
    /// when it scores higher.
    /// # Errors
    /// A fault that stopped it; recorded, and the cycle goes on with the next learner.
    fn run(&mut self) -> Result<LearnerReport, String>;
    /// After its files were restored by a rollback: reload any state held in RAM.
    /// # Errors
    /// The reload failed.
    fn after_restore(&mut self) -> Result<(), String> {
        Ok(())
    }
}

/// One cycle's record.
#[derive(Clone, Debug, PartialEq)]
pub struct CycleRecord {
    pub id: String,
    pub at: u64,
    pub by: String,
    pub trigger: String,
    /// Verified examples in the store when it ran.
    pub verified: usize,
    pub learners: Vec<LearnerReport>,
    /// Each learner's snapshot folder.
    pub snapshots: Vec<(String, PathBuf)>,
    pub audit_id: Option<String>,
}

impl CycleRecord {
    /// The learners the cycle ran, by name, in order.
    #[must_use]
    pub fn ran(&self) -> Vec<String> {
        self.learners.iter().filter(|l| l.ran).map(|l| l.name.clone()).collect()
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("row", s("cycle")),
            ("id", s(&self.id)),
            ("at", n(self.at)),
            ("by", s(&self.by)),
            ("trigger", s(&self.trigger)),
            ("verified", n(self.verified as u64)),
            ("ran", Value::Array(self.ran().iter().map(|x| s(x)).collect())),
            ("learners", Value::Array(self.learners.iter().map(LearnerReport::to_value).collect())),
            ("snapshots", Value::Object(self.snapshots.iter().map(|(k, p)| (k.clone(), s(&p.display().to_string()))).collect())),
            ("audit_id", self.audit_id.as_deref().map_or(Value::Null, s)),
        ])
    }
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        Self {
            id: text(v, "id").unwrap_or_default(),
            at: v.get("at").and_then(Value::integer).unwrap_or(0),
            by: text(v, "by").unwrap_or_default(),
            trigger: text(v, "trigger").unwrap_or_default(),
            verified: v.get("verified").and_then(Value::integer).ok().and_then(|x| usize::try_from(x).ok()).unwrap_or(0),
            learners: v.get("learners").and_then(Value::array).unwrap_or_default().iter().map(LearnerReport::from_value).collect(),
            snapshots: v
                .get("snapshots")
                .and_then(Value::object)
                .map(|o| o.iter().filter_map(|(k, p)| p.string().ok().map(|p| (k.clone(), PathBuf::from(p)))).collect())
                .unwrap_or_default(),
            audit_id: text(v, "audit_id"),
        }
    }
}

/// A new cycle id: `lc-<ms>-<six hex digits>` (unique within a process and across processes).
#[must_use]
pub fn new_cycle_id(now: u64) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let k = NEXT.fetch_add(1, Ordering::Relaxed);
    let h = nmhash::sha256_bytes(format!("{now}|{}|{k}", std::process::id()).as_bytes());
    format!("lc-{now}-{}", h.get(..6).unwrap_or("000000"))
}

/// Whether `id` is a cycle id this module writes (a guard before it names a folder).
#[must_use]
pub fn valid_cycle_id(id: &str) -> bool {
    id.starts_with("lc-") && id.len() <= 40 && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
}

// ---------------------------------------------------------------------------------------------
// Snapshots.

/// Copies the state at `paths` into `dir` with a manifest (`manifest.json`): each path, whether it
/// was a file, a folder (its top-level files) or absent.
/// # Errors
/// A file that cannot be read or copied.
pub fn snapshot(paths: &[PathBuf], dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| io(dir, &e))?;
    let mut rows = Vec::new();
    for (i, p) in paths.iter().enumerate() {
        let slot = dir.join(format!("p{i}"));
        let (kind, files) = if p.is_file() {
            std::fs::create_dir_all(&slot).map_err(|e| io(&slot, &e))?;
            std::fs::copy(p, slot.join("file")).map_err(|e| io(p, &e))?;
            ("file", Vec::new())
        } else if p.is_dir() {
            std::fs::create_dir_all(&slot).map_err(|e| io(&slot, &e))?;
            let mut names = Vec::new();
            for e in std::fs::read_dir(p).map_err(|e| io(p, &e))?.flatten() {
                if e.path().is_file() {
                    let name = e.file_name().to_string_lossy().into_owned();
                    std::fs::copy(e.path(), slot.join(&name)).map_err(|err| io(&e.path(), &err))?;
                    names.push(name);
                }
            }
            names.sort();
            ("dir", names)
        } else {
            ("absent", Vec::new())
        };
        rows.push(obj(vec![
            ("path", s(&p.display().to_string())),
            ("kind", s(kind)),
            ("files", Value::Array(files.iter().map(|f| s(f)).collect())),
        ]));
    }
    let m = dir.join("manifest.json");
    std::fs::write(&m, nmjson::pretty::pretty(&Value::Array(rows))).map_err(|e| io(&m, &e))
}

/// Puts back the state a snapshot folder holds: a file is copied back, a folder's top-level files
/// are copied back and files the cycle added to it are removed, a path that was absent is removed.
/// Returns the paths restored.
/// # Errors
/// A missing or unreadable manifest, or a file that cannot be written.
pub fn restore(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let m = dir.join("manifest.json");
    let bytes = std::fs::read(&m).map_err(|e| io(&m, &e))?;
    let rows = nmjson::parse(&bytes).map_err(|e| format!("{}: {e}", m.display()))?;
    let mut out = Vec::new();
    for (i, r) in rows.array().map_err(|e| format!("{}: {e}", m.display()))?.iter().enumerate() {
        let path = PathBuf::from(text(r, "path").ok_or_else(|| format!("{}: a row has no path", m.display()))?);
        let slot = dir.join(format!("p{i}"));
        match text(r, "kind").as_deref() {
            Some("file") => {
                if let Some(d) = path.parent() {
                    std::fs::create_dir_all(d).map_err(|e| io(d, &e))?;
                }
                std::fs::copy(slot.join("file"), &path).map_err(|e| io(&path, &e))?;
            }
            Some("dir") => {
                let keep: BTreeSet<String> = r.get("files").and_then(Value::array).unwrap_or_default().iter().filter_map(|f| f.string().ok().map(str::to_owned)).collect();
                std::fs::create_dir_all(&path).map_err(|e| io(&path, &e))?;
                for e in std::fs::read_dir(&path).map_err(|e| io(&path, &e))?.flatten() {
                    let name = e.file_name().to_string_lossy().into_owned();
                    if e.path().is_file() && !keep.contains(&name) {
                        std::fs::remove_file(e.path()).map_err(|err| io(&e.path(), &err))?;
                    }
                }
                for name in &keep {
                    std::fs::copy(slot.join(name), path.join(name)).map_err(|e| io(&path.join(name), &e))?;
                }
            }
            _ => {
                if path.is_file() {
                    std::fs::remove_file(&path).map_err(|e| io(&path, &e))?;
                } else if path.is_dir() {
                    for e in std::fs::read_dir(&path).map_err(|e| io(&path, &e))?.flatten() {
                        if e.path().is_file() {
                            std::fs::remove_file(e.path()).map_err(|err| io(&e.path(), &err))?;
                        }
                    }
                    // Left in place when anything other than its own files is inside.
                    let _ = std::fs::remove_dir(&path);
                }
            }
        }
        out.push(path);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// The lock, the records and the cycle.

/// The cycle lock: one cycle or rollback at a time per project.
struct Lock(PathBuf);
impl Drop for Lock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn lock(memory: &Path) -> Result<Lock, String> {
    let dir = memory.join(DIR);
    std::fs::create_dir_all(&dir).map_err(|e| io(&dir, &e))?;
    let p = dir.join("lock");
    for _ in 0..2 {
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&p) {
            Ok(mut f) => {
                let _ = write!(f, "{} {}", std::process::id(), nmtype::now_ms());
                return Ok(Lock(p));
            }
            Err(_) => {
                let age = std::fs::metadata(&p).and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));
                if age < STALE_LOCK_MS {
                    return Err("a learning cycle or rollback is already running for this project".into());
                }
                let _ = std::fs::remove_file(&p);
            }
        }
    }
    Err("the learning cycle lock could not be taken".into())
}

fn append(memory: &Path, row: &Value) -> Result<(), String> {
    let dir = memory.join(DIR);
    std::fs::create_dir_all(&dir).map_err(|e| io(&dir, &e))?;
    let p = dir.join(CYCLES);
    let line = row.encode().map_err(|e| e.to_string())?;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&p).map_err(|e| io(&p, &e))?;
    writeln!(f, "{line}").map_err(|e| io(&p, &e))
}

/// Every row of `cycles.jsonl`, oldest first.
#[must_use]
pub fn rows(memory: &Path) -> Vec<Value> {
    std::fs::read_to_string(memory.join(DIR).join(CYCLES))
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| nmjson::parse(l.as_bytes()).ok())
        .collect()
}

/// The cycles recorded, oldest first.
#[must_use]
pub fn cycles(memory: &Path) -> Vec<CycleRecord> {
    rows(memory).iter().filter(|r| text(r, "row").as_deref() == Some("cycle")).map(CycleRecord::from_value).collect()
}

/// Ids of the cycles already rolled back.
#[must_use]
pub fn rolled_back(memory: &Path) -> BTreeSet<String> {
    rows(memory).iter().filter(|r| text(r, "row").as_deref() == Some("rollback")).filter_map(|r| text(r, "rollback_of")).collect()
}

/// The audit hook: given the record (its id set), writes the caller's audit row and returns its id.
pub type Audit<'a> = dyn FnMut(&Value) -> Option<String> + 'a;

/// Runs one cycle: for each learner in order, snapshot its state, run it, record its report (a
/// learner's fault is recorded and the next one runs). The verified count is marked, the audit row
/// written through `audit`, and the record appended to `cycles.jsonl`.
/// # Errors
/// Another cycle holds the lock, or the records cannot be written.
pub fn run_cycle(memory: &Path, learners: &mut [Box<dyn Learner + '_>], by: &str, trigger: &str, audit: &mut Audit<'_>) -> Result<CycleRecord, String> {
    let _lock = lock(memory)?;
    let at = nmtype::now_ms();
    let id = new_cycle_id(at);
    let verified = verified_count(memory);
    let mut rec = CycleRecord { id: id.clone(), at, by: by.to_owned(), trigger: trigger.to_owned(), verified, learners: Vec::new(), snapshots: Vec::new(), audit_id: None };
    for l in learners.iter_mut() {
        let name = l.name().to_owned();
        let root = l.snapshot_root().unwrap_or_else(|| memory.join(DIR).join(ROLLBACK));
        let dir = root.join(&id).join(&name);
        if let Err(e) = snapshot(&l.state_paths(), &dir) {
            // Without a snapshot there is no rollback: the learner does not run.
            rec.learners.push(LearnerReport::not_run(&name, &format!("not run: its state could not be snapshotted ({e})")));
            continue;
        }
        rec.snapshots.push((name.clone(), dir));
        rec.learners.push(match l.run() {
            Ok(r) => r,
            Err(e) => LearnerReport { ran: true, note: format!("fault: {e}"), ..LearnerReport::not_run(&name, "") },
        });
    }
    let mark = memory.join(DIR).join(MARK);
    std::fs::write(&mark, obj(vec![("verified", n(verified as u64)), ("cycle", s(&id)), ("at", n(at))]).encode().unwrap_or_default()).map_err(|e| io(&mark, &e))?;
    rec.audit_id = audit(&rec.to_value());
    append(memory, &rec.to_value())?;
    Ok(rec)
}

/// Rolls back one cycle (`cycle`, or the newest not yet rolled back): every learner it ran gets
/// back the state it had before the cycle, then reloads what it holds in RAM (`after_restore`, for
/// the learners passed in by name). The rollback is recorded with its audit id. Returns the row.
/// # Errors
/// No such cycle, already rolled back, the lock is held, or a restore fails.
pub fn rollback(memory: &Path, cycle: Option<&str>, learners: &mut [Box<dyn Learner + '_>], by: &str, audit: &mut Audit<'_>) -> Result<Value, String> {
    let _lock = lock(memory)?;
    let done = rolled_back(memory);
    let all = cycles(memory);
    let rec = match cycle {
        Some(id) => all.iter().rev().find(|c| c.id == id).ok_or_else(|| format!("no learning cycle {id}"))?,
        None => all.iter().rev().find(|c| !done.contains(&c.id)).ok_or("no learning cycle to roll back")?,
    };
    if done.contains(&rec.id) {
        return Err(format!("learning cycle {} was already rolled back", rec.id));
    }
    if !valid_cycle_id(&rec.id) {
        return Err(format!("{} is not a learning cycle id", rec.id));
    }
    let touched: Vec<&LearnerReport> = rec.learners.iter().filter(|l| l.ran).collect();
    let snaps: BTreeMap<&str, &PathBuf> = rec.snapshots.iter().map(|(k, p)| (k.as_str(), p)).collect();
    let mut restored = Vec::new();
    let mut notes = Vec::new();
    for l in touched {
        let Some(dir) = snaps.get(l.name.as_str()) else {
            notes.push(format!("{}: no snapshot recorded", l.name));
            continue;
        };
        restore(dir)?;
        if let Some(live) = learners.iter_mut().find(|x| x.name() == l.name)
            && let Err(e) = live.after_restore()
        {
            notes.push(format!("{}: files restored; reload: {e}", l.name));
        }
        restored.push(l.name.clone());
    }
    let mut row = obj(vec![
        ("row", s("rollback")),
        ("rollback_of", s(&rec.id)),
        ("at", n(nmtype::now_ms())),
        ("by", s(by)),
        ("restored", Value::Array(restored.iter().map(|x| s(x)).collect())),
        ("notes", Value::Array(notes.iter().map(|x| s(x)).collect())),
    ]);
    let aid = audit(&row);
    if let Value::Object(m) = &mut row {
        m.insert("audit_id".into(), aid.as_deref().map_or(Value::Null, s));
    }
    append(memory, &row)?;
    Ok(row)
}

// ---------------------------------------------------------------------------------------------
// The engine as a learner, and its feed from the training store.

/// Feeds the training store's settled examples and preference signals into the engine as
/// experiences (each row once, by id: a row that merges into an identical experience is recorded
/// as taken too, so it never adds a second occurrence). The store's split rides in each
/// experience's provenance, so the gated cycle holds out the same items the store does. Returns
/// whether the state changed.
pub fn sync_engine(memory: &Path, eng: &mut ContinuousLearningEngine) -> bool {
    use crate::engine::{ExperienceContext, ExperienceOutcome, ExperienceRecord, ExperienceType, FeedbackSignal, SupervisionState};
    let mut known: BTreeSet<String> = eng.episodic.iter().map(|e| e.experience_id.clone()).collect();
    known.extend(eng.synced.iter().cloned());
    let mut added = false;
    for row in nmtype::examples::read_rows(memory, nmtype::examples::EXAMPLES).unwrap_or_default() {
        let Some(id) = text(&row, "id") else { continue };
        let exp_id = format!("exp_{id}");
        if known.contains(&exp_id) {
            continue;
        }
        let verdict = text(&row, "verdict").unwrap_or_else(|| "used".into());
        let ok = matches!(verdict.as_str(), "verified" | "used");
        let domains: Vec<String> = row.get("domains").and_then(Value::array).map(|a| a.iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect()).unwrap_or_default();
        let scope = domains.first().cloned().unwrap_or_else(|| "project".into());
        let cands = row.get("candidates").and_then(Value::array).map(<[Value]>::to_vec).unwrap_or_default();
        let mut places = Vec::new();
        let (mut served_n, mut used_n, mut irrel_n, mut miss_n) = (0_u64, 0_u64, 0_u64, 0_u64);
        for c in &cands {
            let served = matches!(c.get("served"), Ok(Value::Bool(true)));
            let lbl = text(c, "label").unwrap_or_default();
            if served {
                served_n += 1;
                if let Some(p) = text(c, "place") {
                    places.push(p);
                }
                if matches!(lbl.as_str(), "positive" | "used" | "verified") {
                    used_n += 1;
                } else if matches!(lbl.as_str(), "negative" | "passed-over" | "refuted") {
                    irrel_n += 1;
                }
            } else if matches!(lbl.as_str(), "positive" | "elsewhere" | "went") {
                miss_n += 1;
            }
        }
        if served_n > 0 {
            eng.record_retrieval_feedback(served_n, used_n, irrel_n, miss_n);
        }
        let split = text(&row, "split").unwrap_or_default();
        let rec = ExperienceRecord {
            experience_id: exp_id,
            timestamp_ms: row.get("at").and_then(Value::integer).unwrap_or_else(|_| nmtype::now_ms()),
            experience_type: if ok { ExperienceType::SuccessfulOutcome } else { ExperienceType::FailedOutcome },
            signal: FeedbackSignal::Implicit,
            task: text(&row, "path").unwrap_or_else(|| "context_retrieval".into()),
            scope,
            context: ExperienceContext {
                problem: text(&row, "question").unwrap_or_else(|| format!("pack {id} ({verdict})")),
                initial_approach: text(&row, "path").unwrap_or_else(|| "flow".into()),
                correction: text(&row, "decided_by").unwrap_or_else(|| verdict.clone()),
                places,
                provenance: if split.is_empty() { "evidence_ledger".into() } else { format!("evidence_ledger|split={split}") },
            },
            outcome: ExperienceOutcome { initial_success: ok, corrected_success: ok || verdict == "corrected", delayed_updated: false, tokens_used: 0, ms_used: 0 },
            learning_candidates: if domains.is_empty() { vec!["context_retrieval".into()] } else { domains },
            confidence: if verdict == "verified" { 0.95 } else { 0.78 },
            status: SupervisionState::Observed,
            occurrences: 1,
        };
        let _ = eng.observe(rec);
        eng.synced.insert(format!("exp_{id}"));
        known.insert(format!("exp_{id}"));
        added = true;
    }
    for pref in nmtype::examples::read_rows(memory, nmtype::examples::PREFERENCES).unwrap_or_default() {
        let Some(id) = text(&pref, "id") else { continue };
        let exp_id = format!("exp_{id}");
        if known.contains(&exp_id) {
            continue;
        }
        let kind = text(&pref, "kind").unwrap_or_else(|| "preference".into());
        let is_corr = kind.contains("correction") || kind.contains("revert");
        let split = text(&pref, "split").unwrap_or_default();
        let rec = ExperienceRecord {
            experience_id: exp_id,
            timestamp_ms: pref.get("at").and_then(Value::integer).unwrap_or_else(|_| nmtype::now_ms()),
            experience_type: if is_corr { ExperienceType::Correction } else { ExperienceType::Preference },
            signal: FeedbackSignal::Explicit,
            task: kind.clone(),
            scope: "project".into(),
            context: ExperienceContext {
                problem: format!("Interaction signal: {kind}"),
                initial_approach: "initial_edit".into(),
                correction: format!("Apply preference rule from {kind}"),
                places: text(&pref, "anchor").into_iter().collect(),
                provenance: if split.is_empty() { "hook_preference".into() } else { format!("hook_preference|split={split}") },
            },
            outcome: ExperienceOutcome { initial_success: !is_corr, corrected_success: true, delayed_updated: false, tokens_used: 0, ms_used: 0 },
            learning_candidates: vec![kind],
            confidence: 0.90,
            status: SupervisionState::Observed,
            occurrences: 1,
        };
        let _ = eng.observe(rec);
        eng.synced.insert(format!("exp_{id}"));
        known.insert(format!("exp_{id}"));
        added = true;
    }
    added
}

/// Rows of the training store (examples and preference signals) the engine has not taken in yet.
#[must_use]
pub fn backlog(memory: &Path, eng: &ContinuousLearningEngine) -> u64 {
    let taken = |id: &str| {
        let e = format!("exp_{id}");
        eng.synced.contains(&e) || eng.episodic.iter().any(|x| x.experience_id == e)
    };
    let mut n = 0_u64;
    for file in [nmtype::examples::EXAMPLES, nmtype::examples::PREFERENCES] {
        for row in nmtype::examples::read_rows(memory, file).unwrap_or_default() {
            if let Some(id) = text(&row, "id")
                && !taken(&id)
            {
                n += 1;
            }
        }
    }
    n
}

/// The continual learning engine as a learner: it takes the store's new experiences (when `sync`)
/// and runs its gated Cycle B (`ContinuousLearningEngine::run_gated_cycle`).
pub struct EngineLearner {
    pub memory: PathBuf,
    pub sync: bool,
}

impl Learner for EngineLearner {
    fn name(&self) -> &str {
        "engine"
    }
    fn state_paths(&self) -> Vec<PathBuf> {
        let dir = self.memory.join(crate::engine::DIR);
        vec![dir.join(crate::engine::STATE_FILE), dir.join("snapshots")]
    }
    fn run(&mut self) -> Result<LearnerReport, String> {
        let mut eng = ContinuousLearningEngine::open(&self.memory);
        let added = self.sync && sync_engine(&self.memory, &mut eng);
        let g = eng.run_gated_cycle();
        eng.save(&self.memory).map_err(|e| e.to_string())?;
        Ok(LearnerReport {
            name: self.name().to_owned(),
            ran: true,
            gate: if g.measured { "held-out outcome prediction (1 - mean squared error)".into() } else { format!("supervisor rule (held-out part too small: {} of {})", g.held_out, crate::engine::MIN_HELD_OUT) },
            held_out: g.held_out,
            current: g.current,
            candidate: g.candidate,
            promoted: g.promoted,
            note: format!("engine v{}; new experiences synced: {added}", eng.version),
        })
    }
}

/// The standard learners in their order, for `run_cycle` and `rollback`: `g3` (G3's prior replay,
/// from the serving crate, when given), the preference cartridge fit (library under `home`), then
/// the engine (fed from the training store when `sync`).
#[must_use]
pub fn standard_learners<'a>(home: &Path, memory: &Path, project: &str, g3: Option<Box<dyn Learner + 'a>>, sync: bool) -> Vec<Box<dyn Learner + 'a>> {
    let mut learners: Vec<Box<dyn Learner + 'a>> = Vec::new();
    if let Some(g) = g3 {
        learners.push(g);
    }
    learners.push(Box::new(crate::pref_fit::PreferenceLearner::new(home, memory, project)));
    learners.push(Box::new(EngineLearner { memory: memory.to_path_buf(), sync }));
    learners
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = nmtype::nano_root().join("scratch").join("unit-tests").join(format!("nmlearn-idle-{name}-{}-{}", std::process::id(), nmtype::now_ms()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("temp dir");
        d
    }

    struct Fake {
        file: PathBuf,
        promote: bool,
        reloaded: bool,
    }
    impl Learner for Fake {
        fn name(&self) -> &str {
            "fake"
        }
        fn state_paths(&self) -> Vec<PathBuf> {
            vec![self.file.clone()]
        }
        fn run(&mut self) -> Result<LearnerReport, String> {
            std::fs::write(&self.file, "after").map_err(|e| e.to_string())?;
            Ok(LearnerReport { name: "fake".into(), ran: true, gate: "test".into(), held_out: 4, current: Some(0.5), candidate: Some(0.75), promoted: self.promote, note: String::new() })
        }
        fn after_restore(&mut self) -> Result<(), String> {
            self.reloaded = true;
            Ok(())
        }
    }

    #[test]
    fn threshold_parses_and_falls_back() {
        assert_eq!(parse_threshold(None), DEFAULT_THRESHOLD);
        assert_eq!(parse_threshold(Some("0")), DEFAULT_THRESHOLD);
        assert_eq!(parse_threshold(Some("abc")), DEFAULT_THRESHOLD);
        assert_eq!(parse_threshold(Some(" 7 ")), 7);
    }

    #[test]
    fn due_counts_verified_examples_since_the_last_cycle() {
        let mem = tmp("due");
        let t = nmtype::examples::dir(&mem);
        std::fs::create_dir_all(&t).expect("dir");
        let rows: String = (0..5).map(|i| format!("{{\"id\":\"e{i}\",\"verdict\":\"{}\"}}\n", if i < 3 { "verified" } else { "refuted" })).collect();
        std::fs::write(t.join(nmtype::examples::EXAMPLES), rows).expect("write");
        assert_eq!(verified_count(&mem), 3);
        assert!(due(&mem, 3) && !due(&mem, 4));
        std::fs::create_dir_all(mem.join(DIR)).expect("dir");
        std::fs::write(mem.join(DIR).join(MARK), "{\"verified\":2}").expect("mark");
        assert_eq!(verified_since_last_cycle(&mem), 1);
        // A cleared store counts from zero again.
        std::fs::write(mem.join(DIR).join(MARK), "{\"verified\":9}").expect("mark");
        assert_eq!(verified_since_last_cycle(&mem), 3);
    }

    #[test]
    fn due_counts_direct_feedback_outcomes_without_hooks() {
        let mem = tmp("due-fb");
        // No training/examples.jsonl at all (Cursor, VS Code, Windsurf, Antigravity, or Claude without hooks).
        let outcomes = concat!(
            "{\"at\":1,\"episode\":\"ep-1\",\"correct\":true}\n",
            "{\"at\":2,\"episode\":\"ep-1\",\"correct\":true}\n",
            "{\"at\":3,\"episode\":\"ep-2\",\"correct\":false}\n",
            "{\"at\":4,\"episode\":\"ep-3\",\"correct\":true}\n",
        );
        std::fs::write(mem.join("outcomes.jsonl"), outcomes).expect("outcomes");
        assert_eq!(verified_count(&mem), 2, "distinct positive episodes count once; refuted ignored");
        assert!(due(&mem, 2) && !due(&mem, 3));
        // If ep-1 also settles into training/examples.jsonl later, it is not double-counted.
        let t = nmtype::examples::dir(&mem);
        std::fs::create_dir_all(&t).expect("dir");
        std::fs::write(t.join(nmtype::examples::EXAMPLES), "{\"id\":\"ep-1\",\"verdict\":\"verified\"}\n").expect("examples");
        assert_eq!(verified_count(&mem), 2);
    }

    #[test]
    fn snapshot_and_restore_put_files_and_folders_back() {
        let d = tmp("snap");
        let f = d.join("a.json");
        let folder = d.join("f");
        std::fs::create_dir_all(&folder).expect("dir");
        std::fs::write(&f, "one").expect("w");
        std::fs::write(folder.join("x"), "x1").expect("w");
        let absent = d.join("new.json");
        let snap = d.join("snap");
        snapshot(&[f.clone(), folder.clone(), absent.clone()], &snap).expect("snapshot");
        std::fs::write(&f, "two").expect("w");
        std::fs::write(folder.join("x"), "x2").expect("w");
        std::fs::write(folder.join("y"), "added").expect("w");
        std::fs::write(&absent, "created").expect("w");
        let back = restore(&snap).expect("restore");
        assert_eq!(back.len(), 3);
        assert_eq!(std::fs::read_to_string(&f).expect("r"), "one");
        assert_eq!(std::fs::read_to_string(folder.join("x")).expect("r"), "x1");
        assert!(!folder.join("y").exists() && !absent.exists());
    }

    #[test]
    fn a_cycle_is_recorded_and_its_rollback_restores_every_learner_it_touched() {
        let mem = tmp("cycle");
        let file = mem.join("state.txt");
        std::fs::write(&file, "before").expect("w");
        let mut learners: Vec<Box<dyn Learner>> = vec![Box::new(Fake { file: file.clone(), promote: true, reloaded: false })];
        let mut audits = 0;
        let rec = run_cycle(&mem, &mut learners, "test", "manual", &mut |_| {
            audits += 1;
            Some(format!("a{audits}"))
        })
        .expect("cycle");
        assert_eq!(rec.ran(), vec!["fake".to_owned()]);
        assert_eq!(rec.audit_id.as_deref(), Some("a1"));
        assert_eq!(std::fs::read_to_string(&file).expect("r"), "after");
        let listed = cycles(&mem);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed.first().map(|c| c.id.clone()), Some(rec.id.clone()));
        assert_eq!(listed.first().and_then(|c| c.learners.first()).and_then(|l| l.candidate), Some(0.75));
        let row = rollback(&mem, None, &mut learners, "test", &mut |_| Some("a2".into())).expect("rollback");
        assert_eq!(std::fs::read_to_string(&file).expect("r"), "before");
        assert_eq!(row.get("rollback_of").and_then(Value::string).ok(), Some(rec.id.as_str()));
        assert!(rolled_back(&mem).contains(&rec.id));
        // Nothing left to roll back; the same cycle twice is refused.
        assert!(rollback(&mem, None, &mut learners, "test", &mut |_| None).is_err());
        assert!(rollback(&mem, Some(&rec.id), &mut learners, "test", &mut |_| None).is_err());
    }

    #[test]
    fn cycle_ids_are_unique_and_valid() {
        let a = new_cycle_id(5);
        let b = new_cycle_id(5);
        assert_ne!(a, b);
        assert!(valid_cycle_id(&a) && !valid_cycle_id("../x") && !valid_cycle_id("lc-a/b"));
    }

    #[test]
    fn sync_carries_the_store_split_into_the_engine() {
        let mem = tmp("sync");
        let t = nmtype::examples::dir(&mem);
        std::fs::create_dir_all(&t).expect("dir");
        std::fs::write(t.join(nmtype::examples::EXAMPLES), "{\"id\":\"e1\",\"verdict\":\"verified\",\"split\":\"held-out\",\"domains\":[\"kind:name\"]}\n{\"id\":\"e2\",\"verdict\":\"refuted\",\"split\":\"fit\",\"domains\":[\"kind:name\"]}\n").expect("w");
        let mut eng = ContinuousLearningEngine::default();
        assert!(sync_engine(&mem, &mut eng));
        assert!(!sync_engine(&mem, &mut eng), "each row is fed once");
        let held: Vec<bool> = eng.episodic.iter().map(ContinuousLearningEngine::is_held_out).collect();
        assert_eq!(held, vec![true, false]);
    }

    #[test]
    fn identical_preference_rows_are_taken_once_and_leave_no_backlog() {
        let mem = tmp("sync-merge");
        let t = nmtype::examples::dir(&mem);
        std::fs::create_dir_all(&t).expect("dir");
        // Three identical signals: the engine merges them into one experience, three occurrences.
        let row = |id: &str| format!("{{\"id\":\"{id}\",\"kind\":\"kept-edit\",\"anchor\":\"a.rs:1-2\"}}\n");
        std::fs::write(t.join(nmtype::examples::PREFERENCES), format!("{}{}{}", row("p1"), row("p2"), row("p3"))).expect("w");
        let mut eng = ContinuousLearningEngine::default();
        assert_eq!(backlog(&mem, &eng), 3);
        assert!(sync_engine(&mem, &mut eng));
        assert_eq!(eng.episodic.len(), 1);
        let occ = eng.episodic[0].occurrences;
        assert_eq!(backlog(&mem, &eng), 0, "merged rows count as taken");
        assert!(!sync_engine(&mem, &mut eng), "nothing is taken twice");
        assert_eq!(eng.episodic[0].occurrences, occ, "no extra occurrence on a later sync");
        // The record of taken rows survives a save and an open.
        eng.save(&mem).expect("save");
        let again = ContinuousLearningEngine::open(&mem);
        assert_eq!(backlog(&mem, &again), 0);
    }

    #[test]
    fn the_engine_gate_promotes_only_on_a_held_out_gain() {
        use crate::engine::{ExperienceContext, ExperienceOutcome, ExperienceRecord, ExperienceType, FeedbackSignal, SupervisionState};
        let rec = |i: usize, held: bool, ok: bool| ExperienceRecord {
            experience_id: format!("exp_{i}"),
            timestamp_ms: 1,
            experience_type: if ok { ExperienceType::SuccessfulOutcome } else { ExperienceType::FailedOutcome },
            signal: FeedbackSignal::Implicit,
            task: format!("task {i}"),
            scope: "project".into(),
            context: ExperienceContext { problem: format!("p{i}"), provenance: format!("t|split={}", if held { "held-out" } else { "fit" }), ..ExperienceContext::default() },
            outcome: ExperienceOutcome { initial_success: ok, corrected_success: ok, ..ExperienceOutcome::default() },
            learning_candidates: vec!["kind:value".into()],
            confidence: 0.9,
            status: SupervisionState::Observed,
            occurrences: 1,
        };
        // Fit experiences on `kind:value` mostly fail, and so do the held-out ones: a pattern
        // learned from the fit part predicts the held-out outcomes better than knowing nothing.
        let mut eng = ContinuousLearningEngine::default();
        for i in 0..8 {
            eng.episodic.push(rec(i, false, i == 0));
        }
        for i in 8..14 {
            eng.episodic.push(rec(i, true, false));
        }
        let g = eng.run_gated_cycle();
        assert!(g.measured && g.held_out == 6);
        assert!(g.promoted, "{g:?}");
        assert!(g.candidate > g.current);
        let held_ids: BTreeSet<String> = (8..14).map(|i| format!("exp_{i}")).collect();
        assert!(!eng.patterns.is_empty());
        assert!(eng.patterns.iter().all(|p| p.evidence_ids.iter().all(|id| !held_ids.contains(id))), "held-out experiences are never evidence");
        assert_eq!(eng.episodic.len(), 14, "the held-out experiences stay in the state");
        // The same cycle again learns nothing new: no gain, no promotion, measured score kept.
        let again = eng.run_gated_cycle();
        assert!(!again.promoted && again.measured);
        assert_eq!(eng.held_out_score, again.current);
        assert_eq!(eng.gated_measured, 2);
        assert_eq!(eng.gated_held_back, 1);
        // Too few held-out experiences: the supervisor rule decides, unmeasured.
        let mut small = ContinuousLearningEngine::default();
        small.episodic.push(rec(1, false, true));
        let s = small.run_gated_cycle();
        assert!(!s.measured && s.current.is_none());
    }
}
