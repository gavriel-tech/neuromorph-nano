//! Temporary notes of the memory system, kept on disk in `memory/temp/notes.json`, apart from
//! saved memory. Lifetimes are counted in run-days (days on which the system ran; an idle calendar
//! day does not count): a note used on 3 run-days is saved (a use counts once per run-day; every
//! note of the same line and every recall hit is a use) and becomes an episode; a note left unused
//! for 7 run-days is cleared. Nothing is cleared for count. Settings: tier `T0` in
//! `designs/memory.json`.
use crate::episodic::{Episode, Episodic};
use crate::store::{self, obj, s, u};
use nmjson::Value;
use nmtype::{Fault, Written};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One temporary note.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TempNote {
    pub id: String,
    pub scope: String,
    pub text: String,
    /// Every use, and the run-days on which it was used (a use counts once per run-day).
    pub uses: u64,
    pub days: u64,
    /// Time (ms) of the first and last use, and the run-day of the last use.
    pub first: u64,
    pub last: u64,
    pub last_day: u64,
}

impl TempNote {
    /// One more use at `now` on run-day `day`; the run-day counts once.
    fn used(&mut self, now: u64, day: u64) {
        self.uses += 1;
        if day > self.last_day {
            self.days += 1;
            self.last_day = day;
        }
        self.last = now;
    }
}

/// When a note is saved or cleared, in run-days.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TempPolicy {
    /// Run-days of use at which a note is saved into episodic memory.
    pub promote_uses: u64,
    /// Run-days unused after which an unsaved note is cleared.
    pub clear_days: u64,
}
impl Default for TempPolicy {
    fn default() -> Self {
        Self { promote_uses: crate::pass::SAVE_DAYS, clear_days: crate::keep::WORKING_DAYS }
    }
}
impl TempPolicy {
    /// From tier `T0` of `designs/memory.json`; missing fields keep the defaults.
    #[must_use]
    pub fn from_value(design: &Value) -> Self {
        let d = Self::default();
        let t0 = design.get("tiers").and_then(|t| t.get("T0")).ok();
        let int = |k: &str| t0.and_then(|t| t.get(k).and_then(Value::integer).ok());
        Self {
            promote_uses: int("promote_uses").unwrap_or(d.promote_uses).max(1),
            clear_days: int("clear_run_days").unwrap_or(d.clear_days).max(1),
        }
    }
}

/// What one sweep did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Sweep {
    /// Episode ids the promoted notes became.
    pub promoted: Vec<String>,
    /// Notes cleared as unused.
    pub cleared: usize,
    /// Notes still held.
    pub kept: usize,
}

/// The temp memory of one memory folder.
#[derive(Debug)]
pub struct Temp {
    path: PathBuf,
    policy: TempPolicy,
    notes: BTreeMap<String, TempNote>,
    /// The current run-day (`runday::RunDays` of the folder at open; `set_day` after a tick).
    day: u64,
    /// The file held another writer's form (an older layout wrote the journal's temporary
    /// records here); `settle` writes it back in this module's form.
    foreign: bool,
}
impl Temp {
    /// Open `temp/notes.json` in a memory folder (missing means empty), on the folder's current
    /// run-day. A note written before run-days were kept counts as used on one run-day, today.
    #[must_use]
    pub fn open(dir: &Path, policy: TempPolicy) -> Self {
        let path = dir.join("temp/notes.json");
        let day = crate::runday::RunDays::open(dir).count;
        let mut notes = BTreeMap::new();
        let mut foreign = false;
        if let Some(v) = store::get(&path) {
            foreign = v.get("notes").is_err() || v.get("items").is_ok();
            for n in v.get("notes").and_then(Value::array).unwrap_or_default() {
                let note = TempNote {
                    id: store::text(n, "id"),
                    scope: store::text(n, "scope"),
                    text: store::text(n, "text"),
                    uses: store::num(n, "uses"),
                    days: if n.get("days").is_ok() { store::num(n, "days") } else { 1 },
                    first: store::num(n, "first"),
                    last: store::num(n, "last"),
                    last_day: if n.get("last_day").is_ok() { store::num(n, "last_day") } else { day },
                };
                if !note.id.is_empty() {
                    notes.insert(note.id.clone(), note);
                }
            }
        }
        Self { path, policy, notes, day, foreign }
    }
    /// The run-day uses and sweeps count on (after the folder's counter ticked).
    pub const fn set_day(&mut self, day: u64) {
        self.day = day;
    }
    /// Write the notes file back in this module's form when the open found another writer's
    /// form in it (the journal keeps those records; nothing of them is read here). Returns
    /// whether it wrote.
    /// # Errors
    /// The notes file could not be written.
    pub fn settle(&mut self) -> Result<bool, Fault> {
        if !self.foreign {
            return Ok(false);
        }
        self.save()?;
        self.foreign = false;
        Ok(true)
    }
    fn save(&self) -> Result<(), Fault> {
        let list = self
            .notes
            .values()
            .map(|n| {
                obj(vec![
                    ("id", s(&n.id)),
                    ("scope", s(&n.scope)),
                    ("text", s(&n.text)),
                    ("uses", u(n.uses)),
                    ("days", u(n.days)),
                    ("first", u(n.first)),
                    ("last", u(n.last)),
                    ("last_day", u(n.last_day)),
                ])
            })
            .collect();
        store::put(&self.path, &obj(vec![("record", s("temp")), ("notes", Value::Array(list))]))
    }
    /// Note one line; the same line in the same scope again is one more use of that note.
    /// Returns the note's id.
    /// # Errors
    /// Text that must never be saved (a credential, key, token or e-mail address; nothing is
    /// written), or the notes file could not be written.
    pub fn note(&mut self, scope: &str, text: &str) -> Result<String, Fault> {
        if let Some(why) = crate::classes::never_save(text, &[]) {
            return Err(store::fault(format!("never save: {why}")));
        }
        let id = nmhash::sha256_bytes(format!("{scope}\u{1f}{text}").as_bytes());
        let (now, day) = (nmtype::now_ms(), self.day);
        self.notes.entry(id.clone()).and_modify(|n| n.used(now, day)).or_insert_with(|| TempNote {
            id: id.clone(),
            scope: scope.into(),
            text: text.into(),
            uses: 1,
            days: 1,
            first: now,
            last: now,
            last_day: day,
        });
        self.save()?;
        Ok(id)
    }
    /// Notes sharing at least half the key terms of `q`, best first; each hit is one use.
    /// # Errors
    /// The notes file could not be written.
    pub fn recall(&mut self, q: &str, k: usize) -> Result<Vec<TempNote>, Fault> {
        let terms = nmlex::key_terms(q);
        let mut scored: Vec<(usize, String)> = self
            .notes
            .values()
            .map(|n| {
                let t = n.text.to_lowercase();
                (terms.iter().filter(|w| t.contains(w.as_str())).count(), n.id.clone())
            })
            .filter(|(c, _)| *c > 0 && *c * 2 >= terms.len().max(1))
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let (now, day) = (nmtype::now_ms(), self.day);
        let mut out = Vec::new();
        for (_, id) in scored.into_iter().take(k) {
            if let Some(n) = self.notes.get_mut(&id) {
                n.used(now, day);
                out.push(n.clone());
            }
        }
        if !out.is_empty() {
            self.save()?;
        }
        Ok(out)
    }
    /// Save every note used on at least `promote_uses` run-days into `episodic` (level `temp`), then
    /// clear every other note unused for `clear_days` run-days by the current run-day.
    /// # Errors
    /// An episode or the notes file could not be written.
    pub fn sweep(&mut self, episodic: &mut Episodic) -> Result<Sweep, Fault> {
        let day = self.day;
        let mut r = Sweep::default();
        let ids: Vec<String> = self.notes.keys().cloned().collect();
        for id in ids {
            let Some(n) = self.notes.get(&id).cloned() else { continue };
            if n.days >= self.policy.promote_uses {
                let ep = Episode::new(
                    "temp",
                    "nmmem.temp",
                    &n.text,
                    &n.text,
                    vec![format!("temp note used on {} run-days", n.days), format!("scope: {}", n.scope)],
                    0,
                );
                let ep_id = match episodic.write(ep)? {
                    Written::Stored(m) | Written::Duplicate(m) => m.id,
                    _ => continue,
                };
                r.promoted.push(ep_id);
                self.notes.remove(&id);
            } else if day.saturating_sub(n.last_day) >= self.policy.clear_days {
                r.cleared += 1;
                self.notes.remove(&id);
            }
        }
        r.kept = self.notes.len();
        self.save()?;
        Ok(r)
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.notes.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.notes.is_empty()
    }
    /// The notes held, most used first.
    #[must_use]
    pub fn list(&self) -> Vec<&TempNote> {
        let mut v: Vec<&TempNote> = self.notes.values().collect();
        v.sort_by(|a, b| b.uses.cmp(&a.uses).then(b.last.cmp(&a.last)));
        v
    }
}

/// Self-check of the `temp` sub-component on a folder inside `scratch/checks/`: the same note on
/// one run-day counts one run-day of use, on three run-days it is saved into episodic memory
/// (recallable by its text); a recall hit counts a use; a note unused 7 run-days is cleared, one
/// unused 6 stays; idle calendar days count nothing; a note holding a secret is refused.
pub(crate) fn check() -> Result<(), String> {
    let dir = crate::scratch_dir("temp")?;
    let e = |x: Fault| x.to_string();
    let mut t = Temp::open(&dir, TempPolicy::default());
    t.set_day(1);
    for _ in 0..3 {
        t.note("brain", "ParMap is defined in nmnpcs parmap.rs").map_err(e)?;
    }
    t.note("brain", "an unrelated passing thought").map_err(e)?;
    t.note("brain", "a thought of the second run-day").map_err(e)?;
    let mut ep = Episodic::open(&dir);
    if !t.sweep(&mut ep).map_err(e)?.promoted.is_empty() {
        return Err("nmmem.temp: three uses on one run-day saved a note".into());
    }
    t.set_day(2);
    if t.recall("Where is ParMap defined?", 3).map_err(e)?.len() != 1 {
        return Err("nmmem.temp: recall missed the note".into());
    }
    t.note("brain", "a thought of the second run-day").map_err(e)?;
    t.set_day(3);
    t.note("brain", "ParMap is defined in nmnpcs parmap.rs").map_err(e)?;
    let r = t.sweep(&mut ep).map_err(e)?;
    if r.promoted.len() != 1 || r.cleared != 0 || ep.len() != 1 || ep.recall("Where is ParMap defined?", 3).is_empty() {
        return Err(format!("nmmem.temp: on 3 run-days: promoted {} cleared {}", r.promoted.len(), r.cleared));
    }
    // Last used on run-day 1 and run-day 2: at run-day 8 the first is 7 run-days unused, the second 6.
    t.set_day(8);
    let r = t.sweep(&mut ep).map_err(e)?;
    let left: Vec<String> = Temp::open(&dir, TempPolicy::default()).list().iter().map(|n| n.text.clone()).collect();
    if r.cleared != 1 || left != ["a thought of the second run-day"] {
        return Err(format!("nmmem.temp: at 7 run-days unused: cleared {} left {left:?}", r.cleared));
    }
    // A note holding a secret is refused and never reaches the notes file.
    if t.note("brain", "the token is ghp_ABCdef1234567890abcdefGHIJklmn").is_ok()
        || std::fs::read_to_string(&t.path).is_ok_and(|x| x.contains("ghp_"))
    {
        return Err("nmmem.temp: a secret note was kept".into());
    }
    check_rule()
}

/// The run-day counter decides: notes on a folder whose counter says run-day 5 count that day; a
/// notes file of the older form (no run-days) reads each note as used once, today; a notes file in
/// another writer's form reads as empty and `settle` writes it back in this module's form.
fn check_rule() -> Result<(), String> {
    let dir = nmtype::check_dir("nmmem-temp-rule").map_err(|e| e.to_string())?;
    let e = |x: Fault| x.to_string();
    std::fs::write(dir.join("rundays.json"), r#"{"record":"memory-rundays","count":5,"last_day":20000}"#).map_err(|x| x.to_string())?;
    std::fs::create_dir_all(dir.join("temp")).map_err(|x| x.to_string())?;
    std::fs::write(dir.join("temp/notes.json"), r#"{"record":"temp","notes":[{"id":"n1","scope":"brain","text":"older form","uses":9,"first":1,"last":2}]}"#)
        .map_err(|x| x.to_string())?;
    let mut t = Temp::open(&dir, TempPolicy::default());
    let n = t.list().first().map(|n| (n.days, n.last_day)).ok_or("nmmem.temp: the older note was not read")?;
    if n != (1, 5) || !t.sweep(&mut Episodic::open(&dir)).map_err(e)?.promoted.is_empty() {
        return Err(format!("nmmem.temp: an older note read as {n:?} or was saved on its count of uses"));
    }
    let foreign = br#"{"record":"memory-items","tier":"temp","topic":"","items":[{"id":"ep:1","state":"active"}]}"#;
    std::fs::write(dir.join("temp/notes.json"), foreign).map_err(|x| x.to_string())?;
    let mut f = Temp::open(&dir, TempPolicy::default());
    if !f.is_empty() || !f.settle().map_err(e)? || store::get(&dir.join("temp/notes.json")).is_none_or(|v| v.get("notes").is_err()) {
        return Err("nmmem.temp: a foreign notes file was read, or not written back in this form".into());
    }
    Ok(())
}
