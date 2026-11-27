//! The memory journal (2026-11-12, memory revision 2, section 6): the only source of truth of the
//! memory system. Every event is one line of compact JSON in a segment file under
//! `<memory>/journal/` (`000001.jsonl`, ...). Each line carries its sequence number, run-day, time,
//! kind and data, the previous line's hash (`prev`) and its own hash (`h`, SHA-256 of the line's
//! canonical encoding without `h`), so the lines form one chain and damage anywhere is found.
//!
//! Rules (research, memory Part 13, storage pass: LevelDB, RocksDB, SQLite, event sourcing):
//! - one writer: `LOCK` is held for the session (opened with no sharing on Windows);
//! - lines are appended in batches and flushed to disk before they count; a failed flush poisons
//!   the journal for the rest of the session (never retried);
//! - at open the whole chain is checked; an incomplete or broken last line is set aside in `torn/`
//!   and cut off; damage anywhere else stops the open with a named fault and nothing is repaired;
//! - segments rotate at `segment_bytes`; a closed segment never changes; `segments.json` records
//!   each closed segment's first and last hash;
//! - events are never removed: undo is a new reversing event.
use crate::store::{fault, obj, put_safe, s, u};
use nmjson::Value;
use nmtype::Fault;
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// Default segment size before rotation.
pub const SEGMENT_BYTES: u64 = 64 * 1024 * 1024;
/// The hash before the first line.
pub const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// One journal event.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub seq: u64,
    pub run_day: u64,
    pub at: u64,
    pub kind: String,
    pub data: Value,
    pub prev: String,
    pub h: String,
}

impl Event {
    fn body(&self) -> Value {
        obj(vec![
            ("seq", u(self.seq)),
            ("run_day", u(self.run_day)),
            ("at", u(self.at)),
            ("kind", s(&self.kind)),
            ("data", self.data.clone()),
            ("prev", s(&self.prev)),
        ])
    }
    fn hash_of(body: &Value) -> Result<String, Fault> {
        Ok(nmhash::sha256_bytes(body.encode().map_err(fault)?.as_bytes()))
    }
    /// The line as written (with `h`).
    /// # Errors
    /// Encoding failure.
    pub fn line(&self) -> Result<String, Fault> {
        let mut v = self.body();
        if let Value::Object(m) = &mut v {
            m.insert("h".into(), s(&self.h));
        }
        v.encode().map_err(fault)
    }
    /// Parse and verify one line against the previous hash.
    fn parse(line: &str, prev: &str) -> Result<Self, String> {
        let v = nmjson::parse(line.as_bytes()).map_err(|e| format!("not JSON: {e}"))?;
        let t = |k: &str| v.get(k).and_then(Value::string).map(str::to_owned).map_err(|e| format!("{k}: {e}"));
        let n = |k: &str| v.get(k).and_then(Value::integer).ok().ok_or_else(|| format!("{k}: not a number"));
        let e = Self {
            seq: n("seq")?,
            run_day: n("run_day")?,
            at: n("at")?,
            kind: t("kind")?,
            data: v.get("data").cloned().map_err(|e| format!("data: {e}"))?,
            prev: t("prev")?,
            h: t("h")?,
        };
        if e.prev != prev {
            return Err(format!("seq {}: chain broken (prev {} against {})", e.seq, short(&e.prev), short(prev)));
        }
        let want = Self::hash_of(&e.body()).map_err(|f| f.message)?;
        if want != e.h {
            return Err(format!("seq {}: hash does not match its content", e.seq));
        }
        Ok(e)
    }
}

fn short(h: &str) -> &str {
    h.get(..12).unwrap_or(h)
}

/// What the open found.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Recovery {
    pub segments: usize,
    pub events: u64,
    /// A torn last line that was set aside (file name under `torn/`), if any.
    pub torn: Option<String>,
}

/// The journal of one memory folder.
pub struct Journal {
    dir: PathBuf,
    segment: u64,
    seq: u64,
    prev: String,
    segment_bytes: u64,
    poisoned: bool,
    /// The current segment, open for appending between batches (dropped at rotation and after a
    /// failed write).
    file: Option<std::fs::File>,
    _lock: std::fs::File,
}

fn segment_name(n: u64) -> String {
    format!("{n:06}.jsonl")
}

fn lock(dir: &Path) -> Result<std::fs::File, Fault> {
    let mut o = std::fs::OpenOptions::new();
    o.create(true).truncate(false).write(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        o.share_mode(0);
    }
    o.open(dir.join("LOCK")).map_err(|e| fault(format!("memory journal is held by another writer ({e})")))
}

impl Journal {
    /// Open the journal of memory folder `memory`, check the whole chain, set aside a torn last
    /// line, and return every event in order.
    /// # Errors
    /// Another writer holds it, a file cannot be read, or the chain is damaged before its last line.
    pub fn open(memory: &Path) -> Result<(Self, Vec<Event>, Recovery), Fault> {
        Self::open_with(memory, SEGMENT_BYTES)
    }
    /// `open` with a segment size (checks use a small one).
    /// # Errors
    /// As `open`.
    pub fn open_with(memory: &Path, segment_bytes: u64) -> Result<(Self, Vec<Event>, Recovery), Fault> {
        let dir = memory.join("journal");
        std::fs::create_dir_all(&dir).map_err(fault)?;
        let lock = lock(&dir)?;
        let mut segs: Vec<u64> = std::fs::read_dir(&dir)
            .map_err(fault)?
            .filter_map(Result::ok)
            .filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_suffix(".jsonl")).and_then(|n| n.parse().ok()))
            .collect();
        segs.sort_unstable();
        let mut events = Vec::new();
        let mut prev = GENESIS.to_owned();
        let mut rec = Recovery { segments: segs.len(), ..Recovery::default() };
        for (i, &n) in segs.iter().enumerate() {
            let path = dir.join(segment_name(n));
            let bytes = std::fs::read(&path).map_err(fault)?;
            let text = String::from_utf8_lossy(&bytes).into_owned();
            let last_segment = i + 1 == segs.len();
            let mut good_len = 0_usize;
            let mut lines = text.split_inclusive('\n').peekable();
            while let Some(line) = lines.next() {
                let is_last = last_segment && lines.peek().is_none();
                let complete = line.ends_with('\n');
                match (complete, Event::parse(line.trim_end_matches(['\n', '\r']), &prev)) {
                    (true, Ok(e)) => {
                        prev.clone_from(&e.h);
                        good_len += line.len();
                        events.push(e);
                    }
                    (_, res) if is_last => {
                        // The last line of the journal: incomplete or broken after a crash. Set aside and cut.
                        let torn_dir = dir.join("torn");
                        std::fs::create_dir_all(&torn_dir).map_err(fault)?;
                        let name = format!("{}-{}.txt", segment_name(n), nmtype::now_ms());
                        std::fs::write(torn_dir.join(&name), line.as_bytes()).map_err(fault)?;
                        let f = std::fs::OpenOptions::new().write(true).open(&path).map_err(fault)?;
                        f.set_len(u64::try_from(good_len).map_err(fault)?).map_err(fault)?;
                        f.sync_all().map_err(fault)?;
                        rec.torn = Some(format!("{name} ({})", res.err().unwrap_or_else(|| "no line end".into())));
                    }
                    (_, Err(why)) => {
                        return Err(fault(format!("memory journal damaged in {}: {why}; nothing was changed", segment_name(n))));
                    }
                    (false, Ok(_)) => {
                        return Err(fault(format!(
                            "memory journal damaged in {}: a line without its end before the last line",
                            segment_name(n)
                        )));
                    }
                }
            }
        }
        rec.events = events.len() as u64;
        let segment = segs.last().copied().unwrap_or(1);
        let seq = events.last().map_or(0, |e| e.seq);
        Ok((Self { dir, segment, seq, prev, segment_bytes, poisoned: false, file: None, _lock: lock }, events, rec))
    }

    /// Append a batch of events (kind and data each) for run-day `run_day`; they count once flushed.
    /// # Errors
    /// The journal is poisoned by an earlier failed flush, or the write or flush fails (which poisons it).
    pub fn append(&mut self, run_day: u64, batch: Vec<(&str, Value)>) -> Result<Vec<Event>, Fault> {
        if self.poisoned {
            return Err(fault("memory journal: an earlier flush failed; reopen the memory to recover"));
        }
        let mut out = Vec::with_capacity(batch.len());
        let mut text = String::new();
        let (mut seq, mut prev) = (self.seq, self.prev.clone());
        for (kind, data) in batch {
            seq += 1;
            let mut e = Event { seq, run_day, at: nmtype::now_ms(), kind: kind.to_owned(), data, prev: prev.clone(), h: String::new() };
            e.h = Event::hash_of(&e.body())?;
            prev.clone_from(&e.h);
            text.push_str(&e.line()?);
            text.push('\n');
            out.push(e);
        }
        let path = self.dir.join(segment_name(self.segment));
        // The batch is written on the segment's open handle and flushed to disk before it counts.
        let held = self.file.take();
        let res = (|| -> std::io::Result<(std::fs::File, u64)> {
            let mut f = match held {
                Some(f) => f,
                None => std::fs::OpenOptions::new().create(true).append(true).open(&path)?,
            };
            f.write_all(text.as_bytes())?;
            f.sync_all()?;
            let len = f.metadata()?.len();
            Ok((f, len))
        })();
        let len = match res {
            Ok((f, len)) => {
                self.file = Some(f);
                len
            }
            Err(e) => {
                self.poisoned = true;
                return Err(fault(format!("memory journal: write or flush failed ({e}); the journal stops for this session")));
            }
        };
        self.seq = seq;
        self.prev = prev;
        if len >= self.segment_bytes {
            self.rotate()?;
        }
        Ok(out)
    }

    /// Close the current segment (recorded in `segments.json` with its last hash) and start the next.
    fn rotate(&mut self) -> Result<(), Fault> {
        let list_path = self.dir.join("segments.json");
        let mut list: Vec<Value> = crate::store::get(&list_path)
            .and_then(|v| v.get("closed").ok().and_then(|a| a.array().ok().map(<[Value]>::to_vec)))
            .unwrap_or_default();
        list.push(obj(vec![("segment", s(&segment_name(self.segment))), ("last_seq", u(self.seq)), ("last_hash", s(&self.prev))]));
        put_safe(&list_path, &obj(vec![("record", s("memory-journal-segments")), ("closed", Value::Array(list))]))?;
        self.file = None;
        self.segment += 1;
        Ok(())
    }

    /// The last sequence number and hash.
    #[must_use]
    pub fn head(&self) -> (u64, &str) {
        (self.seq, &self.prev)
    }
}

/// Section check: events round-trip and chain across a reopen; a second writer is refused while one
/// holds the journal; a torn last line is set aside and cut; a damaged middle line stops the open;
/// segments rotate and stay chained.
/// # Errors
/// One fault line.
pub(crate) fn check() -> Result<(), String> {
    let dir = nmtype::check_dir("nmmem-journal").map_err(|e| e.to_string())?;
    {
        let (mut j, ev, _) = Journal::open_with(&dir, 600).map_err(|e| e.to_string())?;
        if !ev.is_empty() {
            return Err("nmmem.journal: a new journal is not empty".into());
        }
        if Journal::open_with(&dir, 600).is_ok() {
            return Err("nmmem.journal: a second writer was let in".into());
        }
        for i in 0..6_u64 {
            j.append(1, vec![("record", obj(vec![("n", u(i)), ("text", s("a memory line long enough to fill segments"))]))])
                .map_err(|e| e.to_string())?;
        }
    }
    let (j, ev, rec) = Journal::open_with(&dir, 600).map_err(|e| e.to_string())?;
    if ev.len() != 6 || rec.segments < 2 || j.head().0 != 6 || rec.torn.is_some() {
        return Err(format!("nmmem.journal: reopen gave {} events in {} segments, torn {:?}", ev.len(), rec.segments, rec.torn));
    }
    drop(j);
    // A torn last line: half a line appended to the last segment.
    let mut segs: Vec<PathBuf> = std::fs::read_dir(dir.join("journal"))
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    segs.sort();
    let last = segs.last().ok_or("no segment")?.clone();
    std::fs::OpenOptions::new().append(true).open(&last).and_then(|mut f| f.write_all(b"{\"seq\":7,\"half")).map_err(|e| e.to_string())?;
    let (j, ev, rec) = Journal::open_with(&dir, 600).map_err(|e| e.to_string())?;
    if ev.len() != 6 || rec.torn.is_none() {
        return Err("nmmem.journal: a torn last line was not set aside".into());
    }
    drop(j);
    // A damaged middle line: change one character in the first segment.
    let first = segs.first().ok_or("no segment")?.clone();
    let text = std::fs::read_to_string(&first).map_err(|e| e.to_string())?;
    std::fs::write(&first, text.replacen("memory line", "memory lime", 1)).map_err(|e| e.to_string())?;
    if Journal::open_with(&dir, 600).is_ok() {
        return Err("nmmem.journal: a damaged middle line was accepted".into());
    }
    Ok(())
}
