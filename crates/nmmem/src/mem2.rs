//! The live memory of revision 2 (2026-11-12, memory revision 2): the journal, the book, the
//! run-day, the nightly pass and the views, behind one small interface the organism calls. At open:
//! the journal is checked; an empty journal imports revision 1 (section 15); on the first open of a
//! new run-day the nightly pass runs; the views (tier folders, index, undo list, snapshot) are
//! written. Then each answer, use, useful use, verdict and failed re-read is one journal event, and
//! the views follow.
use crate::book::Book;
use crate::episodic::Episode;
use crate::journal::Journal;
use crate::record::{Class, Record, State, Tier};
use crate::runday::RunDays;
use crate::store::{obj, s};
use nmjson::Value;
use nmtype::Fault;
use std::path::{Path, PathBuf};

/// The live memory of revision 2.
pub struct Memory2 {
    dir: PathBuf,
    root: PathBuf,
    journal: Journal,
    pub book: Book,
    pub day: u64,
    /// What the open did (import, recovery, pass), for the session report.
    pub opened: Value,
    locked: Vec<String>,
    /// Journal position of the last snapshot written.
    snapshot_seq: u64,
    /// Views held back between `hold` and `release`, and whether a commit came in between.
    hold: bool,
    held: bool,
}

/// Whether a memory folder is a test memory (`serve --memory-dir`, under a `tests` folder) rather
/// than the live memory.
fn is_test_memory(dir: &Path) -> bool {
    dir.components().any(|c| c.as_os_str() == "tests")
}

/// A memory folder outside `tests/` (the system's temporary folder, removed after): the first open
/// of a run-day says the nightly pass ran, a reopen the same day says it did not; an ephemeral
/// record is refused with its reason logged while a normal one is kept; a folder under `tests` is a
/// test memory.
fn check_live() -> Result<(), String> {
    let dir = std::env::temp_dir().join(format!("nm-mem2-live-{}-{}", std::process::id(), nmtype::now_ms()));
    std::fs::create_dir_all(dir.join("src")).map_err(|e| e.to_string())?;
    let run = || -> Result<(), String> {
        std::fs::write(dir.join("src/x.rs"), "pub fn decide_switch(x: u8) -> bool {\n    x > 3\n}\n").map_err(|e| e.to_string())?;
        let pass = |m: &Memory2| matches!(m.opened.get("nightly_pass"), Ok(Value::Bool(true)));
        let mut m = Memory2::open(&dir, &dir).map_err(|e| e.to_string())?;
        if !pass(&m) {
            return Err("nmmem.mem2: the first open of a run-day did not say the nightly pass ran".into());
        }
        let ep = Episode::new("neuromorph", "b0", "where is the switch decided", "src/x.rs:1 decide_switch", vec!["src/x.rs:1 lines 1..3".into()], 1);
        let id = m.remember(&ep).map_err(|e| e.to_string())?;
        let mut eph = m.book.records.get(&id).cloned().ok_or("nmmem.mem2: the remembered record is missing")?;
        eph.id = format!("{id}-scratch");
        eph.class = Class::Ephemeral;
        m.commit_events(vec![("record".into(), eph.to_value())]).map_err(|e| e.to_string())?;
        let refused = crate::store::lines(&dir.join("refused.jsonl"));
        if m.book.records.contains_key(&eph.id) || !m.book.records.contains_key(&id) || refused.len() != 1 {
            return Err(format!("nmmem.mem2: ephemeral record kept or not logged ({} refusals)", refused.len()));
        }
        drop(m);
        let m = Memory2::open(&dir, &dir).map_err(|e| e.to_string())?;
        if pass(&m) {
            return Err("nmmem.mem2: a reopen on the same run-day said the nightly pass ran".into());
        }
        if is_test_memory(&dir) || !is_test_memory(Path::new("x/tests/memory-system/a")) {
            return Err("nmmem.mem2: test memory folders not told from the live memory".into());
        }
        Ok(())
    };
    let r = run();
    let _ = std::fs::remove_dir_all(&dir);
    r
}

/// Journal events between two snapshots written after commits (the snapshot is a start-up cache;
/// `Book::load` replays the events after it).
pub const SNAPSHOT_EVERY: u64 = 32;

/// The record id of a revision 1 episode id.
#[must_use]
pub fn episode_record_id(episode: &str) -> String {
    format!("ep:{}", episode.get(..16).unwrap_or(episode))
}

impl Memory2 {
    /// Open the memory folder `dir` (sources read under `root`).
    /// # Errors
    /// The journal is held or damaged, or a file cannot be written.
    pub fn open(dir: &Path, root: &Path) -> Result<Self, Fault> {
        let mut rd = RunDays::open(dir);
        let day = rd.tick(nmtype::now_ms())?;
        let (mut journal, events, rec) = Journal::open(dir)?;
        let (mut book, from_snapshot) = Book::load(dir, &events)?;
        let mut imported = 0_usize;
        if events.is_empty() {
            let batch = crate::migrate::import_events(dir, root, day);
            imported = batch.iter().filter(|(k, _)| k == "record").count();
            if !batch.is_empty() {
                for e in journal.append(day, batch.iter().map(|(k, v)| (k.as_str(), v.clone())).collect())? {
                    book.apply(&e)?;
                }
            }
        }
        // The nightly pass, once per run-day (its own event marks the run-day it ran on).
        let last_pass = events.iter().rev().find(|e| e.kind == "pass").map_or(0, |e| e.run_day);
        let mut passed = 0_usize;
        if last_pass < day {
            let mut batch = crate::pass::nightly(&book, day);
            passed = batch.len();
            batch.push(("pass".to_owned(), obj(vec![("events", Value::Number(passed.to_string()))])));
            for e in journal.append(day, batch.iter().map(|(k, v)| (k.as_str(), v.clone())).collect())? {
                book.apply(&e)?;
            }
        }
        book.write_views(dir, day)?;
        let opened = obj(vec![
            ("run_day", Value::Number(day.to_string())),
            ("journal_events", Value::Number(book.seq.to_string())),
            ("from_snapshot", Value::Bool(from_snapshot)),
            ("imported_from_revision_1", Value::Number(imported.to_string())),
            ("nightly_pass", Value::Bool(last_pass < day)),
            ("nightly_pass_events", Value::Number(passed.to_string())),
            ("torn_line_set_aside", rec.torn.map_or(Value::Null, |t| s(&t))),
        ]);
        Ok(Self {
            dir: dir.to_path_buf(),
            root: root.to_path_buf(),
            journal,
            day,
            opened,
            locked: crate::classes::locked_questions(&nmtype::nano_root()),
            snapshot_seq: book.seq,
            hold: false,
            held: false,
            book,
        })
    }
    fn commit(&mut self, mut batch: Vec<(String, Value)>) -> Result<(), Fault> {
        // An ephemeral memory (a test answer, scratch work) never reaches the live memory: its
        // record is refused here, with the reason logged; a test memory (under `tests/`) keeps it.
        if !is_test_memory(&self.dir) {
            let refused = |(k, v): &(String, Value)| k == "record" && v.get("class").and_then(Value::string).ok() == Some(Class::Ephemeral.label());
            for (_, v) in batch.iter().filter(|e| refused(e)) {
                let id = v.get("id").and_then(Value::string).unwrap_or_default();
                let _ = crate::store::append(
                    &self.dir.join("refused.jsonl"),
                    &obj(vec![("at", Value::Number(nmtype::now_ms().to_string())), ("id", s(id)), ("reason", s("ephemeral: never saved to the live memory"))]),
                );
            }
            batch.retain(|e| !refused(e));
        }
        if batch.is_empty() {
            return Ok(());
        }
        for e in self.journal.append(self.day, batch.iter().map(|(k, v)| (k.as_str(), v.clone())).collect())? {
            self.book.apply(&e)?;
        }
        if self.hold {
            self.held = true;
            return Ok(());
        }
        self.write_views()
    }
    /// The views after a commit; the snapshot every `SNAPSHOT_EVERY` events.
    fn write_views(&mut self) -> Result<(), Fault> {
        self.book.write_tier_views(&self.dir, self.day)?;
        if self.book.seq >= self.snapshot_seq + SNAPSHOT_EVERY {
            self.book.write_snapshot(&self.dir)?;
            self.snapshot_seq = self.book.seq;
        }
        Ok(())
    }
    /// Hold the views back: the commits that follow are still flushed to the journal before they
    /// count, and `release` writes the views once for all of them.
    pub const fn hold(&mut self) {
        self.hold = true;
    }
    /// Whether views are held back with a commit waiting for `release`.
    #[must_use]
    pub const fn is_held(&self) -> bool {
        self.held
    }
    /// Write the views held back since `hold`, if any commit came in between.
    /// # Errors
    /// A view cannot be written.
    pub fn release(&mut self) -> Result<(), Fault> {
        self.hold = false;
        if std::mem::take(&mut self.held) { self.write_views() } else { Ok(()) }
    }
    /// Commit a batch of journal events made elsewhere in this crate (verified outcomes from the
    /// evidence ledger, `outcomes::import_events`): flushed to the journal, applied to the book,
    /// then the views follow.
    /// # Errors
    /// A journal or view write fault.
    pub fn commit_events(&mut self, batch: Vec<(String, Value)>) -> Result<(), Fault> {
        self.commit(batch)
    }
    /// An answered question stored as a revision 1 episode becomes a memory (or, when a live memory
    /// with the same content exists, a use of it). Never-save text is refused. Returns the id.
    /// # Errors
    /// Never-save text, or a journal or view write fault.
    pub fn remember(&mut self, ep: &Episode) -> Result<String, Fault> {
        if let Some(why) = std::iter::once(ep.query.as_str())
            .chain(std::iter::once(ep.answer.as_str()))
            .chain(ep.evidence.iter().map(String::as_str))
            .find_map(|t| crate::classes::never_save(t, &self.locked))
        {
            return Err(crate::store::fault(format!("never save: {why}")));
        }
        let r = crate::migrate::record_of(ep, Tier::Active, self.day, &self.root);
        if let Some(same) = self.book.records.values().find(|x| x.content == r.content && x.state == State::Active).map(|x| x.id.clone()) {
            self.commit(vec![("use".into(), obj(vec![("id", s(&same)), ("useful", Value::Bool(false))]))])?;
            return Ok(same);
        }
        let id = r.id.clone();
        self.commit(vec![("record".into(), r.to_value())])?;
        Ok(id)
    }
    /// A working note captured at task end (tagged, reached the answer, or recurred) becomes a temp
    /// memory with its reason; it earns a lasting place only by retention. Never-save text is refused.
    /// # Errors
    /// Never-save text, or a journal or view write fault.
    pub fn capture(&mut self, note: &str, why: &str) -> Result<String, Fault> {
        if let Some(w) = crate::classes::never_save(note, &self.locked) {
            return Err(crate::store::fault(format!("never save: {w}")));
        }
        let ep = Episode::new("temp", "working", note, note, Vec::new(), 0);
        let mut r = crate::migrate::record_of(&ep, Tier::Temp, self.day, &self.root);
        r.text = format!("{note} | kept because: {why}");
        r.content = Record::content_hash(r.kind, note, &r.anchors);
        if let Some(same) = self.book.records.values().find(|x| x.content == r.content && x.state == State::Active).map(|x| x.id.clone()) {
            self.commit(vec![("use".into(), obj(vec![("id", s(&same)), ("useful", Value::Bool(false))]))])?;
            return Ok(same);
        }
        let id = r.id.clone();
        self.commit(vec![("record".into(), r.to_value())])?;
        Ok(id)
    }
    /// A note the owner asks to keep (2026-11-26, the interface's Add memory and `nm_memory`
    /// `store`, after the relevance gate admitted it) becomes an active memory under the topic
    /// `notes`, its text as written; the same live note again is one more use of it. Never-save text
    /// is refused. Returns the record id.
    /// # Errors
    /// Never-save text, or a journal or view write fault.
    pub fn keep(&mut self, note: &str, by: &str) -> Result<String, Fault> {
        if let Some(w) = crate::classes::never_save(note, &self.locked) {
            return Err(crate::store::fault(format!("never save: {w}")));
        }
        let ep = Episode::new("note", by, note, note, Vec::new(), 0);
        let mut r = crate::migrate::record_of(&ep, Tier::Active, self.day, &self.root);
        r.text = note.to_owned();
        r.topic = "notes".into();
        r.content = Record::content_hash(r.kind, note, &r.anchors);
        if let Some(same) = self.book.records.values().find(|x| x.content == r.content && x.state == State::Active).map(|x| x.id.clone()) {
            self.commit(vec![("use".into(), obj(vec![("id", s(&same)), ("useful", Value::Bool(false))]))])?;
            return Ok(same);
        }
        let id = r.id.clone();
        self.commit(vec![("record".into(), r.to_value())])?;
        Ok(id)
    }
    /// Uses of recalled memories (by revision 1 episode id), useful when their cited part reached the answer.
    /// # Errors
    /// A journal or view write fault.
    pub fn used(&mut self, uses: &[(String, bool)]) -> Result<(), Fault> {
        let batch = uses
            .iter()
            .map(|(ep, _)| episode_record_id(ep))
            .zip(uses.iter().map(|(_, u)| *u))
            .filter(|(id, _)| self.book.records.contains_key(id))
            .map(|(id, useful)| ("use".to_owned(), obj(vec![("id", s(&id)), ("useful", Value::Bool(useful))])))
            .collect();
        self.commit(batch)
    }
    /// A verdict on the memory of a revision 1 episode.
    /// # Errors
    /// A journal or view write fault.
    pub fn verdict(&mut self, episode: &str, right: bool, by: &str) -> Result<(), Fault> {
        let id = episode_record_id(episode);
        if !self.book.records.contains_key(&id) {
            return Ok(());
        }
        let mut batch = vec![("verdict".to_owned(), obj(vec![("id", s(&id)), ("right", Value::Bool(right)), ("by", s(by))]))];
        if let Some(r) = self.book.records.get(&id) {
            // Verified right: protected from removal; wrong: invalid (kept in the journal).
            if right && r.class == crate::record::Class::Normal {
                batch.push((
                    "class".into(),
                    obj(vec![("id", s(&id)), ("to", s("protected")), ("from", s("normal")), ("why", s("verified right"))]),
                ));
            } else if !right && r.state == State::Active {
                batch.push((
                    "state".into(),
                    obj(vec![("id", s(&id)), ("to", s("invalid")), ("from", s("active")), ("why", s("marked wrong"))]),
                ));
            }
        }
        self.commit(batch)
    }
    /// Reconsolidate the memory of a revision 1 episode against its sources now. Returns whether the
    /// caller's gate should re-check it.
    /// # Errors
    /// A journal or view write fault.
    pub fn reconsolidate(&mut self, episode: &str) -> Result<bool, Fault> {
        let Some(r) = self.book.records.get(&episode_record_id(episode)).cloned() else { return Ok(false) };
        let root = self.root.clone();
        let out = crate::record::reconsolidate(&r, &|a| std::fs::read_to_string(root.join(&a.location)).ok());
        self.commit(out.events)?;
        Ok(out.needs_check)
    }
    /// Pin (`true`) or unpin a memory by record id or episode id: a pinned memory never drops,
    /// never merges and never leaves its tier by itself.
    /// # Errors
    /// Unknown id, or a journal or view write fault.
    pub fn pin(&mut self, id: &str, on: bool) -> Result<String, Fault> {
        let id = self.resolve(id)?;
        let from = self.book.records.get(&id).map(|r| r.class.label().to_owned()).unwrap_or_default();
        let to = if on { "pinned" } else { "normal" };
        if from != to {
            self.commit(vec![(
                "class".into(),
                obj(vec![
                    ("id", s(&id)),
                    ("to", s(to)),
                    ("from", s(&from)),
                    ("why", s(if on { "pinned by the owner" } else { "unpinned by the owner" })),
                ]),
            )])?;
        }
        Ok(id)
    }
    /// Delete a memory at the owner's request (2026-11-26, Luis Jake Gabriel III: delete a single
    /// memory, confirmed in more than one step): its state becomes `invalid` with why "deleted by
    /// the owner", so it leaves recall and every view. The journal keeps the line that it existed
    /// (events are never erased); forgetting the project removes everything. Returns the record id.
    /// # Errors
    /// Unknown id, or a journal or view write fault.
    pub fn forget(&mut self, id: &str) -> Result<String, Fault> {
        let id = self.resolve(id)?;
        let from = self.book.records.get(&id).map(|r| r.state.label().to_owned()).unwrap_or_default();
        if from != crate::record::State::Invalid.label() {
            self.commit(vec![(
                "state".into(),
                obj(vec![("id", s(&id)), ("to", s(crate::record::State::Invalid.label())), ("from", s(&from)), ("why", s("deleted by the owner"))]),
            )])?;
        }
        Ok(id)
    }
    /// Undo one reversible change by its journal sequence number (a new event; nothing is erased).
    /// # Errors
    /// The sequence number is not reversible, or a journal or view write fault.
    pub fn undo(&mut self, seq: u64) -> Result<(), Fault> {
        if !self.book.reversible.contains_key(&seq) {
            return Err(crate::store::fault(format!("seq {seq} is not a reversible change")));
        }
        self.commit(vec![("undo".into(), obj(vec![("seq", Value::Number(seq.to_string()))]))])
    }
    /// The health view (counts by tier, class and state; leaving soon; journal head).
    #[must_use]
    pub fn health(&self) -> Value {
        let mut h = self.book.health(self.day);
        if let Value::Object(o) = &mut h {
            let (seq, hash) = self.journal.head();
            o.insert("journal_head".into(), obj(vec![("seq", Value::Number(seq.to_string())), ("hash", s(hash))]));
            o.insert("opened".into(), self.opened.clone());
        }
        h
    }
    /// A record id from a record id or a revision 1 episode id.
    /// # Errors
    /// Neither names a memory.
    pub fn resolve(&self, id: &str) -> Result<String, Fault> {
        [id.to_owned(), episode_record_id(id)]
            .into_iter()
            .find(|x| self.book.records.contains_key(x))
            .ok_or_else(|| crate::store::fault(format!("no memory {id}")))
    }
}

/// Section check: opening a folder with one revision 1 episode imports it, runs the pass, writes the
/// views; reopening on the same run-day imports nothing and runs no pass; a repeat answer becomes a
/// use; a verdict protects it; a never-save answer is refused.
/// # Errors
/// One fault line.
pub(crate) fn check() -> Result<(), String> {
    check_live()?;
    let root = nmtype::check_dir("nmmem-mem2-root").map_err(|e| e.to_string())?;
    std::fs::create_dir_all(root.join("src")).map_err(|e| e.to_string())?;
    std::fs::write(root.join("src/x.rs"), "pub fn decide_switch(x: u8) -> bool {\n    x > 3\n}\n").map_err(|e| e.to_string())?;
    let dir = nmtype::check_dir("nmmem-mem2").map_err(|e| e.to_string())?;
    let ep = Episode::new(
        "neuromorph",
        "b0",
        "where is the switch decided",
        "src/x.rs:1 decide_switch",
        vec!["src/x.rs:1 lines 1..3".into()],
        1,
    );
    crate::episodic::Episodic::open(&dir).write(ep.clone()).map_err(|e| e.to_string())?;
    let id = {
        let m = Memory2::open(&dir, &root).map_err(|e| e.to_string())?;
        if m.opened.get("imported_from_revision_1").and_then(Value::integer).ok() != Some(1)
            || !dir.join("index.json").is_file()
            || !dir.join("journal").join("000001.jsonl").is_file()
        {
            return Err(format!("nmmem.mem2: first open {}", m.opened.encode().unwrap_or_default()));
        }
        episode_record_id(&ep.id)
    };
    let mut m = Memory2::open(&dir, &root).map_err(|e| e.to_string())?;
    if m.opened.get("imported_from_revision_1").and_then(Value::integer).ok() != Some(0)
        || m.opened.get("nightly_pass_events").and_then(Value::integer).ok() != Some(0)
    {
        return Err(format!("nmmem.mem2: reopen {}", m.opened.encode().unwrap_or_default()));
    }
    let again = m.remember(&ep).map_err(|e| e.to_string())?;
    m.verdict(&ep.id, true, "check").map_err(|e| e.to_string())?;
    let r = m.book.records.get(&id).ok_or("record missing")?;
    if again != id || r.retention.uses != 1 || r.class != crate::record::Class::Protected {
        return Err(format!("nmmem.mem2: repeat and verdict gave {again} uses {} class {:?}", r.retention.uses, r.class));
    }
    let pinned = m.pin(&ep.id, true).map_err(|e| e.to_string())?;
    let seq = m.book.seq;
    m.undo(seq).map_err(|e| e.to_string())?;
    if m.book.records.get(&pinned).map(|r| r.class) != Some(crate::record::Class::Protected) || m.health().get("journal_head").is_err() {
        return Err("nmmem.mem2: pin then undo did not restore the class".into());
    }
    let a = m.capture("the switch reads x above 3", "tagged").map_err(|e| e.to_string())?;
    let b = m.capture("the switch reads x above 3", "tagged").map_err(|e| e.to_string())?;
    if a != b || m.book.records.get(&a).map(|r| (r.tier, r.retention.uses)) != Some((Tier::Temp, 1)) {
        return Err("nmmem.mem2: a captured note was not one temp memory with one repeat use".into());
    }
    let secret =
        Episode::new("neuromorph", "b0", "what is the key", "the key is sk-ABCdef1234567890abcdefGHIJ", vec!["src/x.rs:1".into()], 1);
    if m.remember(&secret).is_ok() {
        return Err("nmmem.mem2: a never-save answer was stored".into());
    }
    // Held views: two commits reach the journal at once, the index follows only at release.
    let index = || std::fs::read(dir.join("index.json")).unwrap_or_default();
    let (seq, before) = (m.book.seq, index());
    m.hold();
    m.capture("held note one", "tagged").map_err(|e| e.to_string())?;
    m.capture("held note two", "tagged").map_err(|e| e.to_string())?;
    let during = index();
    m.release().map_err(|e| e.to_string())?;
    if m.book.seq != seq + 2 || during != before || index() == before {
        return Err("nmmem.mem2: held views were written early, or not at release".into());
    }
    // The snapshot is written every SNAPSHOT_EVERY events; one older still starts the same book.
    for i in 0..SNAPSHOT_EVERY {
        m.capture(&format!("snapshot note {i}"), "tagged").map_err(|e| e.to_string())?;
    }
    let snap = crate::store::get(&dir.join("snapshot.json")).and_then(|v| v.get("seq").and_then(Value::integer).ok()).unwrap_or(0);
    let book = m.book.clone();
    drop(m);
    let (_, events, _) = crate::journal::Journal::open(&dir).map_err(|e| e.to_string())?;
    let (again, used) = Book::load(&dir, &events).map_err(|e| e.to_string())?;
    if snap == 0 || snap > book.seq || !used || again != book {
        return Err(format!("nmmem.mem2: snapshot at {snap} of {}, used {used}, same book {}", book.seq, again == book));
    }
    Ok(())
}
