//! Memory hierarchy (Luis Jake Gabriel III, 2026-11-08 11:42 and 11:52 UTC): saved memory by use,
//! most needed at the top, never so deep that things get lost. Three levels, top first: core (most
//! used items, capped), active (used recently, one topic per item), archive (rarely used, kept,
//! never deleted).
//!
//! ```text
//! memory/hierarchy.json   every item with its level, topic and use counts (this module's only file)
//! ```
//!
//! The tier folders (`1-core/`, `2-active/<topic>/`, `3-archive/<topic>/`) and `index.json` are
//! views of the journal, written only by `book::Book::write_views`. This module never writes them
//! and never reads the journal's records as its own items: at open it takes only its own items
//! (by their form: use counts, no record state) from tier files an older layout left there, and
//! `settle` writes them into `hierarchy.json`.
//!
//! Each use (a recall hit, or the same item added again) counts, and the run-days on which an
//! item was used count once each (run-days: days on which the system ran). A rebalance moves the
//! most used active items used on 3 run-days up into core (up to its 32 places), moves core
//! overflow down, and drops an item one level when it is unused past its tier's lifetime: `1-core`
//! after 28 run-days, `2-active` after 14. `episodes.jsonl` stays the full record; the hierarchy
//! organizes it. Settings: `hierarchy` in `designs/memory.json`.
use crate::store::{self, obj, s, u};
use nmjson::Value;
use nmtype::Fault;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The core folder of a memory folder, where the identity record and the objectives live.
#[must_use]
pub fn core_dir(dir: &Path) -> PathBuf {
    dir.join("1-core")
}

/// Copy the identity records and objectives from their older flat place into `1-core` when
/// `1-core` does not have them yet. The older files are left where they are.
pub fn adopt_flat(dir: &Path) {
    let core = core_dir(dir);
    let _ = std::fs::create_dir_all(core.join("identity"));
    if let Ok(rd) = std::fs::read_dir(dir.join("identity")) {
        // Only identity records (.json); a folder guide such as README.md stays where it is.
        for e in rd.flatten().filter(|e| e.path().extension().is_some_and(|x| x == "json")) {
            let to = core.join("identity").join(e.file_name());
            if !to.exists() {
                let _ = std::fs::copy(e.path(), to);
            }
        }
    }
    let (from, to) = (dir.join("objectives.json"), core.join("objectives.json"));
    if from.is_file() && !to.exists() {
        let _ = std::fs::copy(from, to);
    }
}

/// The three levels, top first.
pub const LEVELS: [&str; 3] = ["1-core", "2-active", "3-archive"];

/// One saved item.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Item {
    pub id: String,
    /// 0 core, 1 active, 2 archive.
    pub level: usize,
    pub topic: String,
    pub query: String,
    pub text: String,
    pub source: String,
    pub uses: u64,
    /// Useful uses (2026-11-12, memory design "Hierarchy"): recalls whose cited line ended up in
    /// the emitted answer's evidence. Older records read as 0.
    pub useful: u64,
    /// Failed re-reads since the last demotion (2026-11-12, memory design: "demote when the
    /// cited spans fail to re-read twice"). Older records read as 0.
    pub failures: u64,
    pub first: u64,
    pub last: u64,
    /// Run-days on which the item was used, and the run-day of its last use.
    pub days: u64,
    pub last_day: u64,
}

impl Item {
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

/// Core's places and the lifetimes of the hierarchy, in run-days.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HierarchyPolicy {
    /// Most items in `1-core`.
    pub core_cap: usize,
    /// Run-days of use an active item needs before it may enter core.
    pub core_uses: u64,
    /// Run-days unused after which a core item drops to active, and an active item to archive.
    pub core_days: u64,
    pub active_days: u64,
}
impl Default for HierarchyPolicy {
    fn default() -> Self {
        Self {
            core_cap: crate::pass::CORE_CAP,
            core_uses: crate::pass::SAVE_DAYS,
            core_days: crate::keep::EVIDENCE_DAYS,
            active_days: crate::retention::ACTIVE_DAYS as u64,
        }
    }
}
impl HierarchyPolicy {
    /// From `hierarchy` in `designs/memory.json`; missing fields keep the defaults.
    #[must_use]
    pub fn from_value(design: &Value) -> Self {
        let d = Self::default();
        let h = design.get("hierarchy").ok();
        let int = |k: &str| h.and_then(|t| t.get(k).and_then(Value::integer).ok());
        let size = |k: &str, dv: usize| int(k).and_then(|c| usize::try_from(c).ok()).unwrap_or(dv).max(1);
        Self {
            core_cap: size("core_cap", d.core_cap),
            core_uses: int("core_uses").unwrap_or(d.core_uses).max(1),
            core_days: int("core_run_days").unwrap_or(d.core_days).max(1),
            active_days: int("active_run_days").unwrap_or(d.active_days).max(1),
        }
    }
}

/// What one rebalance moved.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Moves {
    pub up: usize,
    pub down: usize,
}

/// A topic folder name: lowercase letters, digits and `-`, at most 24 characters.
#[must_use]
pub fn topic_name(raw: &str) -> String {
    let t: String = raw.trim().to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
    let t = t.trim_matches('-').chars().take(24).collect::<String>();
    if t.is_empty() { "general".into() } else { t }
}

/// The hierarchy of one memory folder, held in RAM and written back per touched folder.
#[derive(Debug)]
pub struct Hierarchy {
    dir: PathBuf,
    policy: HierarchyPolicy,
    items: BTreeMap<String, Item>,
    /// The current run-day (`runday::RunDays` of the folder at open; `set_day` after a tick).
    day: u64,
    /// Items taken at open from tier files of an older layout.
    adopted: usize,
    /// `hierarchy.json` exists (read at open or written since).
    saved: bool,
}

/// The hierarchy's own file in a memory folder.
pub const FILE: &str = "hierarchy.json";

/// Every `items.json` under the three tier folders of a memory folder, with its level.
#[must_use]
pub fn tier_files(dir: &Path) -> Vec<(usize, PathBuf)> {
    let mut out = Vec::new();
    for (level, name) in LEVELS.iter().enumerate() {
        let base = dir.join(name);
        out.push((level, base.join("items.json")));
        if let Ok(rd) = std::fs::read_dir(&base) {
            let mut sub: Vec<PathBuf> = rd.flatten().filter(|e| e.path().is_dir()).map(|e| e.path().join("items.json")).collect();
            sub.sort();
            out.extend(sub.into_iter().map(|p| (level, p)));
        }
    }
    out.retain(|(_, p)| p.is_file());
    out
}

/// True for an item in this module's form (use counts, no record state or retention); false for a
/// journal record as the book writes it.
#[must_use]
pub fn is_own_item(x: &Value) -> bool {
    x.get("state").is_err() && x.get("retention").is_err() && x.get("anchors").is_err()
}
impl Hierarchy {
    /// Open `hierarchy.json` of a memory folder, then take this module's own items from any tier
    /// file an older layout left (items with use counts and no record state; ids holding `:` are
    /// journal record ids and stay the journal's). An item found in both keeps the later use.
    #[must_use]
    pub fn open(dir: &Path, policy: HierarchyPolicy) -> Self {
        let day = crate::runday::RunDays::open(dir).count;
        let mut h = Self { dir: dir.to_path_buf(), policy, items: BTreeMap::new(), day, adopted: 0, saved: false };
        let item = |x: &Value, level: usize| Item {
            id: store::text(x, "id"),
            level,
            topic: topic_name(&store::text(x, "topic")),
            query: store::text(x, "query"),
            text: store::text(x, "text"),
            source: store::text(x, "source"),
            uses: store::num(x, "uses"),
            useful: store::num(x, "useful"),
            failures: store::num(x, "failures"),
            first: store::num(x, "first"),
            last: store::num(x, "last"),
            // An item written before run-days were kept counts as used on one run-day, today.
            days: if x.get("days").is_ok() { store::num(x, "days") } else { u64::from(store::num(x, "uses") > 0) },
            last_day: if x.get("last_day").is_ok() { store::num(x, "last_day") } else { day },
        };
        if let Some(v) = store::get(&dir.join(FILE)) {
            h.saved = true;
            for x in v.get("items").and_then(Value::array).unwrap_or_default() {
                let level = LEVELS.iter().position(|l| *l == store::text(x, "level")).unwrap_or(1);
                let it = item(x, level);
                if !it.id.is_empty() {
                    h.items.insert(it.id.clone(), it);
                }
            }
        }
        for (level, f) in tier_files(dir) {
            let Some(v) = store::get(&f) else { continue };
            for x in v.get("items").and_then(Value::array).unwrap_or_default() {
                if !is_own_item(x) {
                    continue;
                }
                let it = item(x, level);
                if it.id.is_empty() || it.id.contains(':') {
                    continue;
                }
                let newer = h.items.get(&it.id).is_none_or(|held| it.last > held.last);
                if newer {
                    h.items.insert(it.id.clone(), it);
                    h.adopted += 1;
                }
            }
        }
        h
    }
    /// Items taken at open from tier files of an older layout (not yet in `hierarchy.json`).
    #[must_use]
    pub const fn adopted(&self) -> usize {
        self.adopted
    }
    /// Write `hierarchy.json` when the open took items from an older layout, or when the file
    /// does not exist yet and there are items. Returns whether it wrote.
    /// # Errors
    /// The file could not be written.
    pub fn settle(&mut self) -> Result<bool, Fault> {
        if self.adopted == 0 && (self.saved || self.items.is_empty()) {
            return Ok(false);
        }
        self.save()?;
        Ok(true)
    }
    /// Write `hierarchy.json`: every item with its level, most used first per level.
    fn save(&mut self) -> Result<(), Fault> {
        let mut list: Vec<&Item> = self.items.values().collect();
        list.sort_by(|a, b| a.level.cmp(&b.level).then(b.uses.cmp(&a.uses)).then(b.last.cmp(&a.last)).then(a.id.cmp(&b.id)));
        let rows = list
            .iter()
            .map(|i| {
                obj(vec![
                    ("id", s(&i.id)),
                    ("level", s(LEVELS.get(i.level).copied().unwrap_or("3-archive"))),
                    ("topic", s(&i.topic)),
                    ("query", s(&i.query)),
                    ("text", s(&i.text)),
                    ("source", s(&i.source)),
                    ("uses", u(i.uses)),
                    ("useful", u(i.useful)),
                    ("failures", u(i.failures)),
                    ("first", u(i.first)),
                    ("last", u(i.last)),
                    ("days", u(i.days)),
                    ("last_day", u(i.last_day)),
                ])
            })
            .collect();
        let counts = self.counts();
        let tiers: Vec<Value> = LEVELS.iter().zip(counts).map(|(l, c)| obj(vec![("level", s(l)), ("items", u(c as u64))])).collect();
        store::put(
            &self.dir.join(FILE),
            &obj(vec![
                ("record", s("memory-hierarchy")),
                ("note", s("use counts of saved items by level; the tier folders and index.json are views of the journal")),
                ("levels", Value::Array(tiers)),
                ("items", Value::Array(rows)),
            ]),
        )?;
        self.adopted = 0;
        self.saved = true;
        Ok(())
    }
    /// Save an item into `2-active/<topic>` (or count one more use of the same id wherever it
    /// sits). Returns the id.
    /// # Errors
    /// A folder file could not be written.
    pub fn add(&mut self, id: &str, topic: &str, query: &str, text: &str, source: &str) -> Result<String, Fault> {
        let (now, day) = (nmtype::now_ms(), self.day);
        if let Some(it) = self.items.get_mut(id) {
            it.used(now, day);
        } else {
            let topic = topic_name(topic);
            self.items.insert(
                id.into(),
                Item {
                    id: id.into(),
                    level: 1,
                    topic,
                    query: query.into(),
                    text: text.into(),
                    source: source.into(),
                    uses: 1,
                    useful: 0,
                    failures: 0,
                    first: now,
                    last: now,
                    days: 1,
                    last_day: day,
                },
            );
        }
        self.save()?;
        Ok(id.into())
    }
    /// A failed re-read for each of `ids` (its cited code no longer reads back): at the second
    /// failure the item drops one level (archive is the floor; nothing is deleted) and its count
    /// starts again. Returns how many items were demoted.
    /// # Errors
    /// The hierarchy cannot be saved.
    pub fn mark_failed(&mut self, ids: &[String]) -> Result<usize, Fault> {
        let mut demoted = 0;
        for id in ids {
            if let Some(it) = self.items.get_mut(id) {
                it.failures += 1;
                if it.failures >= 2 {
                    it.level = (it.level + 1).min(2);
                    it.failures = 0;
                    demoted += 1;
                }
            }
        }
        if !ids.is_empty() {
            self.save()?;
        }
        Ok(demoted)
    }
    /// Count a useful use for each of `ids` (a recall whose cited line reached the answer).
    /// # Errors
    /// The hierarchy cannot be saved.
    pub fn mark_useful(&mut self, ids: &[String]) -> Result<usize, Fault> {
        let mut n = 0;
        for id in ids {
            if let Some(it) = self.items.get_mut(id) {
                it.useful += 1;
                n += 1;
            }
        }
        if n > 0 {
            self.save()?;
        }
        Ok(n)
    }
    /// Count one use of each id that exists (for example episodes just recalled).
    /// # Errors
    /// A folder file could not be written.
    pub fn touch(&mut self, ids: &[String]) -> Result<usize, Fault> {
        let (now, day) = (nmtype::now_ms(), self.day);
        let mut n = 0;
        for id in ids {
            if let Some(it) = self.items.get_mut(id) {
                it.used(now, day);
                n += 1;
            }
        }
        if n > 0 {
            self.save()?;
        }
        Ok(n)
    }
    /// Top-down recall: items sharing at least half the key terms of `q`, core first, then
    /// active, then archive, stopping once `k` are found. Each hit is one use.
    /// # Errors
    /// A folder file could not be written.
    pub fn recall(&mut self, q: &str, k: usize) -> Result<Vec<Item>, Fault> {
        let terms = nmlex::key_terms(q);
        let mut hits: Vec<String> = Vec::new();
        for level in 0..LEVELS.len() {
            let mut scored: Vec<(usize, u64, String)> = self
                .items
                .values()
                .filter(|i| i.level == level)
                .map(|i| {
                    let t = format!("{} {}", i.query, i.text).to_lowercase();
                    (terms.iter().filter(|w| t.contains(w.as_str())).count(), i.uses, i.id.clone())
                })
                .filter(|(c, _, _)| *c > 0 && *c * 2 >= terms.len().max(1))
                .collect();
            scored.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
            hits.extend(scored.into_iter().map(|x| x.2).take(k.saturating_sub(hits.len())));
            if hits.len() >= k {
                break;
            }
        }
        self.touch(&hits)?;
        Ok(hits.iter().filter_map(|id| self.items.get(id).cloned()).collect())
    }
    /// The run-day uses and rebalances count on (after the folder's counter ticked).
    pub const fn set_day(&mut self, day: u64) {
        self.day = day;
    }
    /// Move items between levels by use on the current run-day (see the module text). Archive keeps
    /// everything.
    /// # Errors
    /// A folder file could not be written.
    pub fn rebalance(&mut self) -> Result<Moves, Fault> {
        let mut m = Moves::default();
        let (day, p) = (self.day, self.policy);
        // Down: unused past its tier's lifetime drops one level.
        for it in self.items.values_mut() {
            let life = if it.level == 0 { p.core_days } else { p.active_days };
            if it.level < 2 && day.saturating_sub(it.last_day) >= life {
                it.level += 1;
                m.down += 1;
            }
        }
        // Up: the most used, recently used active items fill core, by uses then recency
        // (an item just dropped for being unused is not lifted straight back).
        let fresh = |i: &Item| day.saturating_sub(i.last_day) < p.active_days;
        // 2026-11-12: ranked by useful uses first, then uses, then recency.
        let mut ranked: Vec<(u64, u64, u64, String, usize)> = self
            .items
            .values()
            .filter(|i| i.level == 0 || (i.level == 1 && fresh(i) && i.days >= self.policy.core_uses))
            .map(|i| (i.useful, i.uses, i.last, i.id.clone(), i.level))
            .collect();
        ranked.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)).then(b.2.cmp(&a.2)));
        for (rank, (_, _, _, id, level)) in ranked.into_iter().enumerate() {
            let want = if rank < self.policy.core_cap { 0 } else { 1 };
            if want != level {
                if let Some(it) = self.items.get_mut(&id) {
                    it.level = want;
                }
                if want < level { m.up += 1 } else { m.down += 1 }
            }
        }
        self.save()?;
        Ok(m)
    }
    /// Items held at each level, top first.
    #[must_use]
    pub fn counts(&self) -> [usize; 3] {
        let mut c = [0; 3];
        for it in self.items.values() {
            if let Some(x) = c.get_mut(it.level) {
                *x += 1;
            }
        }
        c
    }
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&Item> {
        self.items.get(id)
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// Self-check of the `hierarchy` sub-component on a folder inside `scratch/checks/`: a new
/// item lands in active under its topic, uses on enough run-days lift it into core (uses on one
/// run-day count once), recall reads core first, core unused 28 run-days drops to active and
/// active unused 14 to archive, every topic keeps its own name (no topic count limit), everything survives reopening
/// from `hierarchy.json`, no tier view is written, and the archive keeps every item.
pub(crate) fn check() -> Result<(), String> {
    // Useful uses rank above plain uses (2026-11-12).
    {
        let dir = nmtype::check_dir("nmmem-hierarchy-useful").map_err(|e| e.to_string())?;
        let mut h = Hierarchy::open(&dir, HierarchyPolicy { core_cap: 1, core_uses: 1, ..HierarchyPolicy::default() });
        h.add("busy", "context", "q1", "a1", "unit").map_err(|e| e.to_string())?;
        h.add("helpful", "context", "q2", "a2", "unit").map_err(|e| e.to_string())?;
        h.touch(&["busy".into(), "busy".into(), "busy".into()]).map_err(|e| e.to_string())?;
        h.mark_useful(&["helpful".into()]).map_err(|e| e.to_string())?;
        h.rebalance().map_err(|e| e.to_string())?;
        if h.get("helpful").map(|i| i.level) != Some(0) || h.get("busy").map(|i| i.level) != Some(1) {
            return Err("nmmem.hierarchy: useful uses did not rank first".into());
        }
        // Two failed re-reads drop an item one level; one does not.
        h.mark_failed(&["busy".into()]).map_err(|e| e.to_string())?;
        if h.get("busy").map(|i| i.level) != Some(1) {
            return Err("nmmem.hierarchy: one failed re-read demoted".into());
        }
        h.mark_failed(&["busy".into()]).map_err(|e| e.to_string())?;
        if h.get("busy").map(|i| (i.level, i.failures)) != Some((2, 0)) {
            return Err("nmmem.hierarchy: two failed re-reads did not demote".into());
        }
    }
    let dir = crate::scratch_dir("hierarchy")?;
    let policy = HierarchyPolicy { core_cap: 1, core_uses: 2, ..HierarchyPolicy::default() };
    let e = |x: Fault| x.to_string();
    let mut h = Hierarchy::open(&dir, policy);
    h.set_day(1);
    h.add("a", "Build", "Where is the Step Check target?", "scratch/stepcheck-target", "test").map_err(e)?;
    h.add("b", "context", "Where is ParMap?", "nmnpcs parmap.rs", "test").map_err(e)?;
    h.add("c", "third topic", "unrelated", "keeps its topic", "test").map_err(e)?;
    if h.get("a").map(|i| (i.level, i.topic.as_str())) != Some((1, "build")) || h.get("c").map(|i| i.topic.as_str()) != Some("third-topic") {
        return Err("nmmem.hierarchy: placement or topic name".into());
    }
    // Two uses on run-day 1 are one run-day of use: not yet core.
    h.recall("Step Check target", 1).map_err(e)?;
    h.rebalance().map_err(e)?;
    if h.get("a").map(|i| i.level) != Some(1) {
        return Err("nmmem.hierarchy: uses on one run-day lifted an item into core".into());
    }
    h.set_day(2);
    h.recall("Step Check target", 1).map_err(e)?;
    h.rebalance().map_err(e)?;
    if h.get("a").map(|i| i.level) != Some(0) || h.counts() != [1, 2, 0] {
        return Err(format!("nmmem.hierarchy: promotion on 2 run-days, counts {:?}", h.counts()));
    }
    // Run-day 15: b and c (last used on run-day 1) are 14 run-days unused and go to the archive;
    // a (core, last used on run-day 2) stays.
    h.set_day(15);
    h.rebalance().map_err(e)?;
    let back = Hierarchy::open(&dir, policy);
    if back.counts() != [1, 0, 2]
        || store::get(&dir.join(FILE)).is_none()
        || !tier_files(&dir).is_empty()
        || dir.join("index.json").exists()
    {
        return Err(format!("nmmem.hierarchy: active past 14 run-days, counts {:?}, or a tier view written", back.counts()));
    }
    // Run-day 30: core unused 28 run-days drops to active; run-day 44: on to the archive, which is
    // the floor and keeps everything.
    h.set_day(30);
    h.rebalance().map_err(e)?;
    if h.counts() != [0, 1, 2] {
        return Err(format!("nmmem.hierarchy: core past 28 run-days, counts {:?}", h.counts()));
    }
    h.set_day(44);
    h.rebalance().map_err(e)?;
    h.set_day(400);
    h.rebalance().map_err(e)?;
    if h.counts() != [0, 0, 3] || Hierarchy::open(&dir, policy).len() != 3 {
        return Err(format!("nmmem.hierarchy: archive did not keep every item, counts {:?}", h.counts()));
    }
    check_adopt()
}

/// An older layout's tier file holding both forms: this module takes its own items (with their
/// counts) into `hierarchy.json` and leaves the journal's records alone; the tier file is not
/// written.
fn check_adopt() -> Result<(), String> {
    let dir = nmtype::check_dir("nmmem-hierarchy-adopt").map_err(|e| e.to_string())?;
    let mixed = br#"{"record":"memory-items","level":"2-active","topic":"context","items":[
        {"id":"0a1b","topic":"context","query":"q","text":"t","source":"u","uses":4,"useful":1,"failures":1,"first":5,"last":9},
        {"id":"ep:0a1b2c3d4e5f6a7b","topic":"context","query":"","text":"copy","source":"","uses":0,"useful":0,"failures":0,"first":0,"last":0},
        {"id":"ep:ffff","state":"active","tier":"2-active","retention":{"uses":2},"anchors":[]}]}"#;
    std::fs::create_dir_all(dir.join("3-archive/context")).map_err(|e| e.to_string())?;
    let file = dir.join("3-archive/context/items.json");
    std::fs::write(&file, mixed).map_err(|e| e.to_string())?;
    let mut h = Hierarchy::open(&dir, HierarchyPolicy::default());
    let got = h.get("0a1b").map(|i| (i.level, i.uses, i.useful, i.failures, i.last));
    if h.len() != 1 || got != Some((2, 4, 1, 1, 9)) || h.adopted() != 1 {
        return Err(format!("nmmem.hierarchy: adoption took {} items, {got:?}", h.len()));
    }
    if !h.settle().map_err(|e| e.to_string())? || std::fs::read(&file).ok().as_deref() != Some(&mixed[..]) {
        return Err("nmmem.hierarchy: settle did not write its own file, or wrote a tier file".into());
    }
    let again = Hierarchy::open(&dir, HierarchyPolicy::default());
    if again.adopted() != 0 || again.get("0a1b").map(|i| i.uses) != Some(4) {
        return Err("nmmem.hierarchy: reopen after settle adopted again".into());
    }
    Ok(())
}
