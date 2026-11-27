//! The memory book (2026-11-12, memory revision 2, sections 4, 6 and 9): the state of every memory,
//! rebuilt by replaying the journal, and the views written from it (tier folders, index, undo
//! list) plus a snapshot for a fast start. The journal is the only truth; every file written here
//! is a view or a cache that a rebuild regenerates.
//!
//! Events (journal `kind`, then `data`):
//! - `record`: a whole record (written or replaced by a newer version of the same id);
//! - `use`: `{id, useful}` on the event's run-day (counts once per run-day toward `days_used`);
//! - `verdict`: `{id, right, by}`;
//! - `state`, `class`, `tier`: `{id, to, from, why}` (the prior value kept for undo);
//! - `relocate`: `{id, anchor, part, from}` (an anchor's part after its source changed);
//! - `link`: `{id, rel, to}`;
//! - `undo`: `{seq}` (reverses one earlier event of the kinds above; the event itself stays).
use crate::journal::Event;
use crate::record::{Class, Part, Record, State, Tier};
use crate::store::{fault, obj, put_safe, put_view, s, u};
use nmjson::Value;
use nmtype::Fault;
use std::collections::BTreeMap;
use std::path::Path;

/// The state of the memory at one journal position.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Book {
    pub records: BTreeMap<String, Record>,
    /// Journal position the state includes.
    pub seq: u64,
    pub hash: String,
    /// Reversible events by sequence number: (kind, data), for undo.
    pub reversible: BTreeMap<u64, (String, Value)>,
}

fn text(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::string).ok().map(str::to_owned)
}

impl Book {
    /// Apply one event. Unknown kinds are kept in the journal and ignored here.
    /// # Errors
    /// A record that does not parse, or a label that does not exist.
    pub fn apply(&mut self, e: &Event) -> Result<(), Fault> {
        let id = text(&e.data, "id").unwrap_or_default();
        match e.kind.as_str() {
            "record" => {
                let r = Record::from_value(&e.data)?;
                self.records.insert(r.id.clone(), r);
            }
            "use" => {
                if let Some(r) = self.records.get_mut(&id) {
                    let useful = matches!(e.data.get("useful"), Ok(Value::Bool(true)));
                    let ret = &mut r.retention;
                    ret.uses += 1;
                    if useful {
                        ret.useful += 1;
                    }
                    // Stability changes on the first use of a run-day only (spacing).
                    if ret.days_used == 0 || e.run_day > ret.last_use_day {
                        let since = e.run_day.saturating_sub(ret.last_use_day.max(r.validity.recorded_day));
                        let outcome = if useful { crate::retention::Outcome::Useful } else { crate::retention::Outcome::Plain };
                        (ret.stability, ret.difficulty) =
                            crate::retention::update(ret.stability, ret.difficulty, since, outcome, ret.salience_pending);
                        ret.days_used += 1;
                    }
                    ret.last_use_day = ret.last_use_day.max(e.run_day);
                }
            }
            "verdict" | "failed" => {
                if let Some(r) = self.records.get_mut(&id) {
                    let right = e.kind == "verdict" && matches!(e.data.get("right"), Ok(Value::Bool(true)));
                    if e.kind == "verdict" {
                        r.verdicts.push((right, text(&e.data, "by").unwrap_or_default()));
                    } else {
                        r.retention.failures += 1;
                    }
                    if !right {
                        let ret = &mut r.retention;
                        let since = e.run_day.saturating_sub(ret.last_use_day.max(r.validity.recorded_day));
                        (ret.stability, ret.difficulty) =
                            crate::retention::update(ret.stability, ret.difficulty, since, crate::retention::Outcome::Lapse, 0.0);
                    }
                }
            }
            "retention" => {
                if let Some(r) = self.records.get_mut(&id) {
                    let num = |k: &str| match e.data.get(k) {
                        Ok(Value::Number(n)) => n.parse::<f64>().ok(),
                        _ => None,
                    };
                    if let Some(x) = num("stability") {
                        r.retention.stability = x;
                    }
                    if let Some(x) = num("difficulty") {
                        r.retention.difficulty = x;
                    }
                    if let Some(x) = num("salience_pending") {
                        r.retention.salience_pending = x;
                    }
                }
            }
            "state" | "class" | "tier" => {
                let to = text(&e.data, "to").ok_or_else(|| fault(format!("seq {}: no target value", e.seq)))?;
                if let Some(r) = self.records.get_mut(&id) {
                    set_label(r, &e.kind, &to, e.run_day)?;
                }
                self.reversible.insert(e.seq, (e.kind.clone(), e.data.clone()));
            }
            "relocate" => {
                if let (Some(r), Ok(Value::Number(i))) = (self.records.get_mut(&id), e.data.get("anchor")) {
                    let i: usize = i.parse().map_err(fault)?;
                    if let (Some(a), Ok(p)) = (r.anchors.get_mut(i), e.data.get("part")) {
                        a.part = part_of(p)?;
                        if let Some(h) = text(&e.data, "part_hash") {
                            a.part_hash = h;
                        }
                    }
                }
            }
            "link" => {
                if let (Some(r), Some(rel), Some(to)) = (self.records.get_mut(&id), text(&e.data, "rel"), text(&e.data, "to"))
                    && !r.links.iter().any(|(a, b)| *a == rel && *b == to)
                {
                    r.links.push((rel, to));
                }
            }
            "undo" => {
                let n = e
                    .data
                    .get("seq")
                    .and_then(Value::integer)
                    .ok()
                    .ok_or_else(|| fault(format!("seq {}: undo without a target", e.seq)))?;
                let (kind, data) = self.reversible.remove(&n).ok_or_else(|| fault(format!("seq {}: nothing reversible at {n}", e.seq)))?;
                let from = text(&data, "from").ok_or_else(|| fault(format!("seq {n}: no prior value to restore")))?;
                if let Some(r) = self.records.get_mut(&text(&data, "id").unwrap_or_default()) {
                    set_label(r, &kind, &from, e.run_day)?;
                }
            }
            _ => {}
        }
        self.seq = e.seq;
        self.hash.clone_from(&e.h);
        Ok(())
    }
    /// The book from a list of events (a full replay).
    /// # Errors
    /// As `apply`.
    pub fn replay(events: &[Event]) -> Result<Self, Fault> {
        let mut b = Self::default();
        for e in events {
            b.apply(e)?;
        }
        Ok(b)
    }
    /// Records of a tier, most used first, grouped by topic (core has one group).
    #[must_use]
    pub fn tier(&self, t: Tier, day: u64) -> BTreeMap<String, Vec<&Record>> {
        let mut out: BTreeMap<String, Vec<&Record>> = BTreeMap::new();
        for r in self.records.values().filter(|r| r.tier == t && crate::pass::visible(r, day)) {
            let key = if t == Tier::Core || t == Tier::Temp { String::new() } else { r.topic.clone() };
            out.entry(key).or_default().push(r);
        }
        for list in out.values_mut() {
            list.sort_by(|a, b| {
                b.retention.days_used.cmp(&a.retention.days_used).then(b.retention.uses.cmp(&a.retention.uses)).then(a.id.cmp(&b.id))
            });
        }
        out
    }
    /// Write the views under memory folder `dir` for run-day `day` (only memories `pass::visible`
    /// shows): the tier folders (`1-core/items.json`, `2-active/<topic>/items.json`,
    /// `3-archive/<topic>/items.json`), the temporary tier (`temp/captured.json`), `index.json`
    /// (every tier by label with counts), `undo.json` (changes that can still be undone), and
    /// `snapshot.json`. This is the only writer of those files: a tier file on disk that holds no
    /// memory now is written empty (never deleted), so each memory is listed in one tier file only.
    /// # Errors
    /// A file cannot be written.
    pub fn write_views(&self, dir: &Path, day: u64) -> Result<(), Fault> {
        self.write_tier_views(dir, day)?;
        self.write_snapshot(dir)
    }
    /// `write_views` without the snapshot: the tier files, the temporary tier, `index.json` and
    /// `undo.json`.
    /// # Errors
    /// A file cannot be written.
    pub fn write_tier_views(&self, dir: &Path, day: u64) -> Result<(), Fault> {
        let mut tiers = Vec::new();
        let mut folders = Vec::new();
        let mut written: Vec<std::path::PathBuf> = Vec::new();
        for t in Tier::ALL {
            let groups = self.tier(*t, day);
            let count: usize = groups.values().map(Vec::len).sum();
            tiers.push(obj(vec![("tier", s(t.label())), ("items", u(count as u64))]));
            for (topic, list) in groups {
                let rel = if topic.is_empty() { t.label().to_owned() } else { format!("{}/{topic}", t.label()) };
                let file = if *t == Tier::Temp { captured_file(dir) } else { dir.join(&rel).join("items.json") };
                written.push(file.clone());
                put_view(
                    &file,
                    &obj(vec![
                        ("record", s("memory-items")),
                        ("tier", s(t.label())),
                        ("topic", s(&topic)),
                        ("items", Value::Array(list.iter().map(|r| r.to_value()).collect())),
                    ]),
                )?;
                folders.push(obj(vec![
                    ("folder", s(&rel)),
                    ("items", u(list.len() as u64)),
                    ("top", Value::Array(list.iter().take(3).map(|r| s(&r.text)).collect())),
                ]));
            }
        }
        // Tier files that hold nothing now (a memory moved tier, or an older layout left them).
        let stale = crate::hierarchy::tier_files(dir)
            .into_iter()
            .map(|(level, f)| (Tier::ALL.get(level).copied().unwrap_or(Tier::Archive), f))
            .chain(std::iter::once((Tier::Temp, captured_file(dir))).filter(|(_, f)| f.is_file()))
            .filter(|(_, f)| !written.contains(f));
        for (t, f) in stale {
            let topic = if matches!(t, Tier::Core | Tier::Temp) {
                String::new()
            } else {
                f.parent().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
            };
            put_view(
                &f,
                &obj(vec![
                    ("record", s("memory-items")),
                    ("tier", s(t.label())),
                    ("topic", s(&topic)),
                    ("items", Value::Array(Vec::new())),
                ]),
            )?;
        }
        put_view(
            &dir.join("index.json"),
            &obj(vec![
                ("record", s("memory-index")),
                (
                    "note",
                    s(
                        "read top down: 1-core, then 2-active, then 3-archive; captured temporary memories in temp/captured.json; built from the journal",
                    ),
                ),
                ("journal_seq", u(self.seq)),
                ("tiers", Value::Array(tiers)),
                ("folders", Value::Array(folders)),
            ]),
        )?;
        put_view(
            &dir.join("undo.json"),
            &obj(vec![
                ("record", s("memory-undo")),
                (
                    "changes",
                    Value::Array(
                        self.reversible
                            .iter()
                            .map(|(seq, (k, d))| obj(vec![("seq", u(*seq)), ("kind", s(k)), ("data", d.clone())]))
                            .collect(),
                    ),
                ),
            ]),
        )
    }
    /// Write `snapshot.json` (a cache for a fast start: `load` uses it only when its position is on
    /// the journal's chain, and replays the events after it, so an older snapshot is still right).
    /// # Errors
    /// The file cannot be written.
    pub fn write_snapshot(&self, dir: &Path) -> Result<(), Fault> {
        put_view(&dir.join("snapshot.json"), &self.to_snapshot())
    }
    /// Memory health for `nm_status` (2026-11-12, memory revision 2, section 14): memories per tier
    /// and per state, the journal position, and use (distinct run-days, useful share).
    #[must_use]
    pub fn health(&self, day: u64) -> Value {
        let mut tiers = Vec::new();
        for t in Tier::ALL {
            tiers.push(obj(vec![("tier", s(t.label())), ("shown", u(self.tier(*t, day).values().map(Vec::len).sum::<usize>() as u64))]));
        }
        let mut states: BTreeMap<&str, u64> = BTreeMap::new();
        for r in self.records.values() {
            *states.entry(r.state.label()).or_default() += 1;
        }
        let (uses, useful) = self.records.values().fold((0, 0), |(a, b), r| (a + r.retention.uses, b + r.retention.useful));
        #[expect(clippy::cast_precision_loss, reason = "a share")]
        let share = if uses == 0 { 0.0 } else { useful as f64 / uses as f64 };
        obj(vec![
            ("run_day", u(day)),
            ("journal_seq", u(self.seq)),
            ("memories", u(self.records.len() as u64)),
            ("tiers", Value::Array(tiers)),
            ("states", Value::Object(states.into_iter().map(|(k, v)| (k.to_owned(), u(v))).collect())),
            ("uses", u(uses)),
            ("useful_share", Value::Number(format!("{share:.3}"))),
            ("undoable_changes", u(self.reversible.len() as u64)),
        ])
    }
    fn to_snapshot(&self) -> Value {
        obj(vec![
            ("record", s("memory-snapshot")),
            ("seq", u(self.seq)),
            ("hash", s(&self.hash)),
            ("records", Value::Array(self.records.values().map(Record::to_value).collect())),
            (
                "reversible",
                Value::Array(
                    self.reversible.iter().map(|(seq, (k, d))| obj(vec![("seq", u(*seq)), ("kind", s(k)), ("data", d.clone())])).collect(),
                ),
            ),
        ])
    }
    /// Load from `snapshot.json` when its position is on the journal's chain, then replay the
    /// events after it; else replay everything. Returns the book and whether the snapshot was used.
    /// # Errors
    /// As `apply`.
    pub fn load(dir: &Path, events: &[Event]) -> Result<(Self, bool), Fault> {
        let snap = crate::store::get(&dir.join("snapshot.json"));
        let usable = snap.as_ref().and_then(|v| {
            let seq = v.get("seq").ok()?.integer().ok()?;
            let hash = text(v, "hash")?;
            events.iter().find(|e| e.seq == seq).filter(|e| e.h == hash).map(|_| (v.clone(), seq))
        });
        let Some((v, seq)) = usable else { return Ok((Self::replay(events)?, false)) };
        let mut b = Self { seq, hash: text(&v, "hash").unwrap_or_default(), ..Self::default() };
        for r in v.get("records").and_then(Value::array).map_err(fault)? {
            let rec = Record::from_value(r)?;
            b.records.insert(rec.id.clone(), rec);
        }
        for x in v.get("reversible").and_then(Value::array).map_err(fault)? {
            if let (Ok(n), Some(k), Ok(d)) = (x.get("seq").and_then(Value::integer), text(x, "kind"), x.get("data")) {
                b.reversible.insert(n, (k, d.clone()));
            }
        }
        for e in events.iter().filter(|e| e.seq > seq) {
            b.apply(e)?;
        }
        Ok((b, true))
    }
}

/// The view of the temporary tier: captured working notes kept as journal records. The older
/// temporary notes (`temp/notes.json`) belong to `temp::Temp` alone.
#[must_use]
pub fn captured_file(dir: &Path) -> std::path::PathBuf {
    dir.join("temp").join("captured.json")
}

fn set_label(r: &mut Record, kind: &str, to: &str, run_day: u64) -> Result<(), Fault> {
    match kind {
        "state" => {
            r.state = State::parse(to)?;
            if matches!(r.state, State::Superseded | State::Invalid) {
                r.validity.until_day.get_or_insert(run_day);
                r.validity.retired_day.get_or_insert(run_day);
            } else if r.state == State::Active {
                r.validity.until_day = None;
                r.validity.retired_day = None;
            }
        }
        "class" => r.class = Class::parse(to)?,
        _ => r.tier = Tier::parse(to)?,
    }
    Ok(())
}

fn part_of(v: &Value) -> Result<Part, Fault> {
    let a = v.get("lines").and_then(Value::array).map_err(fault)?;
    let n =
        |i: usize| a.get(i).and_then(|x| x.integer().ok()).and_then(|x| usize::try_from(x).ok()).ok_or_else(|| fault("relocate: lines"));
    Ok(Part::Lines { start: n(0)?, end: n(1)? })
}

/// Section check: a replay builds the records; uses count once per run-day; a state change and its
/// undo restore the record; the views written from the book match a rebuild; a snapshot plus the
/// later events equals a full replay; a snapshot off the chain is ignored.
/// # Errors
/// One fault line.
pub(crate) fn check() -> Result<(), String> {
    use crate::journal::Journal;
    use crate::record::{Anchor, Kind, Retention, Source, Validity};
    let dir = nmtype::check_dir("nmmem-book").map_err(|e| e.to_string())?;
    let rec = |id: &str, tier: Tier| Record {
        id: id.into(),
        content: id.into(),
        kind: Kind::Fact,
        class: Class::Normal,
        tier,
        state: State::Active,
        topic: "context".into(),
        text: format!("memory {id}"),
        anchors: vec![Anchor {
            source: Source::File,
            location: "a.rs".into(),
            part: Part::Lines { start: 1, end: 2 },
            part_hash: "h".into(),
            name: String::new(),
            excerpt: None,
        }],
        keys: Vec::new(),
        links: Vec::new(),
        validity: Validity::default(),
        retention: Retention { stability: 8.3, difficulty: 5.0, ..Retention::default() },
        verdicts: Vec::new(),
    };
    let (mut j, _, _) = Journal::open(&dir).map_err(|e| e.to_string())?;
    j.append(1, vec![("record", rec("m1", Tier::Active).to_value()), ("record", rec("m2", Tier::Core).to_value())])
        .map_err(|e| e.to_string())?;
    j.append(
        1,
        vec![
            ("use", obj(vec![("id", s("m1")), ("useful", Value::Bool(true))])),
            ("use", obj(vec![("id", s("m1")), ("useful", Value::Bool(false))])),
        ],
    )
    .map_err(|e| e.to_string())?;
    let st = j
        .append(
            2,
            vec![
                ("use", obj(vec![("id", s("m1")), ("useful", Value::Bool(false))])),
                ("state", obj(vec![("id", s("m1")), ("to", s("superseded")), ("from", s("active")), ("why", s("newer memory"))])),
            ],
        )
        .map_err(|e| e.to_string())?;
    let state_seq = st.last().map_or(0, |e| e.seq);
    drop(j);
    let (_, events, _) = Journal::open(&dir).map_err(|e| e.to_string())?;
    let b = Book::replay(&events).map_err(|e| e.to_string())?;
    let m1 = b.records.get("m1").ok_or("m1 missing")?;
    if (m1.retention.uses, m1.retention.useful, m1.retention.days_used, m1.state) != (3, 1, 2, State::Superseded)
        || m1.validity.until_day != Some(2)
    {
        return Err(format!("nmmem.book: replay gave {:?} {:?}", m1.retention, m1.state));
    }
    b.write_views(&dir, 2).map_err(|e| e.to_string())?;
    // Snapshot at this point, then an undo after it.
    let undo = {
        let (mut j, _, _) = Journal::open(&dir).map_err(|e| e.to_string())?;
        j.append(3, vec![("undo", obj(vec![("seq", u(state_seq))]))]).map_err(|e| e.to_string())?
    };
    let (_, events, _) = Journal::open(&dir).map_err(|e| e.to_string())?;
    let (from_snap, used) = Book::load(&dir, &events).map_err(|e| e.to_string())?;
    let full = Book::replay(&events).map_err(|e| e.to_string())?;
    if !used || from_snap != full || full.records.get("m1").map(|r| r.state) != Some(State::Active) || undo.is_empty() {
        return Err("nmmem.book: snapshot plus replay differs from a full replay, or the undo did not restore".into());
    }
    let idx = crate::store::get(&dir.join("index.json")).ok_or("no index")?;
    if idx.get("tiers").and_then(Value::array).map(<[Value]>::len).ok() != Some(Tier::ALL.len()) {
        return Err("nmmem.book: the index does not list every tier".into());
    }
    // A snapshot that is not on the chain is ignored.
    put_safe(
        &dir.join("snapshot.json"),
        &obj(vec![
            ("record", s("memory-snapshot")),
            ("seq", u(2)),
            ("hash", s("not-on-chain")),
            ("records", Value::Array(Vec::new())),
            ("reversible", Value::Array(Vec::new())),
        ]),
    )
    .map_err(|e| e.to_string())?;
    let (again, used) = Book::load(&dir, &events).map_err(|e| e.to_string())?;
    if used || again != full {
        return Err("nmmem.book: a snapshot off the chain was used".into());
    }
    Ok(())
}
