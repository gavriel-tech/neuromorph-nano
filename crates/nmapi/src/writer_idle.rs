//! The writer at idle (master design 3.10 rules 28 and 29; register C27, C91; Luis Jake Gabriel III's
//! decision 4 of 2026-11-17: idle study allowed; the job map of 2026-11-18 07:44 UTC: the writer does
//! idle reasoning over hard cases, chain planning and notes; the orchestration study 6.2 chain C6 and
//! 6.5: senses, file notes and the outline at idle, the embedder beside it checking each note).
//!
//! The writer MicroModel (Qwen3-1.7B, its base cartridge) never loads inside a call: a load takes
//! tens of seconds (C27) and at 8 bits it does not fit on a 4 GB card beside the three finding and
//! ranking MicroModels (C91). When `serve` is idle it loads at 8 bits (the widening kernel, which
//! matches 16 bits), writes, and lets go of the card:
//! 1. Idle: no call in flight, no call for `quiet_ms` (the longer of a floor and twice the longest
//!    call seen), the meaning lane not building on the card, and work without a result.
//! 2. Memory: the writer's load holds its open weights and its cartridge in RAM while it seats, about
//!    7 GB of commit memory (lane B's trial, 2026-11-18); it loads only when that much is free
//!    (`NANO_WRITER_FREE_MB` names another floor), and unloads after every session.
//! 3. Release first: the card copies of the bound judge and reader are released (their parameters
//!    stay in RAM); the embedder stays (421 MiB beside the writer's 1,939 MiB, study 6.5) to check
//!    each note and so a call's question encoding never waits for it (`NANO_WRITER_CHECK=off`
//!    releases it too and keeps notes unchecked).
//! 4. Load the writer at 8 bits (its base cartridge seated as for every MicroModel).
//! 5. Write, in this order, each result keyed by the content hash of what it was written from, so it
//!    is written once per material version and a changed piece gets a new one:
//!    - notes (one line, the main identifiers) for the pieces a call contested or left blind
//!      (`served`, `prioritize`): the request's drone's senses;
//!    - summaries (two short sentences) of the pieces calls served (`served`): for the pack, placed
//!      beside the cited source, never in its place (`summary`, behind `NANO_WRITER_PACK=on`);
//!    - notes for every other piece, in file order;
//!    - file notes, from the notes of a file's pieces, once all of them are written;
//!    - the project outline, from the file notes, once nine in ten files have one (`outline`).
//!
//!    A note is kept only when the embedder, reading it as a request, ranks its own piece above the
//!    piece's eight nearest neighbours (study 6.5: a sense is kept only if it finds its item).
//! 6. A call that arrives preempts: the decode stops at its next token, the unfinished result is
//!    dropped, the writer is let go at once.
//! 7. Let go of the writer (its population dropped, its card memory freed), then restore the
//!    released MicroModels: only their missing pieces move from RAM to the card (`stage_parts`).
//!
//! Results are held in RAM and, when the session has a memory folder (or `NANO_WRITER_DIR` names
//! one), cached on disk in the Nano's own folders (`<memory>/world-model/notes.jsonl`, the world model; Luis Jake Gabriel
//! III, 2026-11-16 18:10 UTC, encodings may be kept on disk where they save time): each line the
//! result's key (job and content hash), the writer's values key and the writer's text; the
//! material's text is never stored. A cached result is used only while the writer's values key is
//! the same; `NANO_WRITER_CACHE=off` keeps everything in RAM.
//!
//! The same idle loop fits the project's adapters when `NANO_ADAPTER_FIT=on` (`adapter_fit`), before
//! the writer loads, with the models already on the card.
//!
//! On by default (`switched_on`; `NANO_WRITER_IDLE=off`, `serve --writer-idle off` or `writer.idle`
//! set to `off` in `settings.json` turns it off). Gavriel Technologies; architecture by Luis Jake
//! Gabriel III.
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmapi.writer_idle", FaultKind::Execution, m)
}

/// Calls in flight (`call`).
static IN_CALL: AtomicUsize = AtomicUsize::new(0);
/// Set by every call's start; the writer stops at its next token when it sees it.
static PREEMPT: AtomicBool = AtomicBool::new(false);
/// When the last call ended (ms).
static LAST_END: AtomicU64 = AtomicU64::new(0);
/// The longest call seen (ms).
static LONGEST: AtomicU64 = AtomicU64::new(0);
/// The last calls' lengths (ms), newest last, at most `RECENT_CALLS`.
static RECENT: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());
/// Calls the quiet time looks back over.
const RECENT_CALLS: usize = 32;
/// The most quiet time a session waits for (ms).
const QUIET_CAP_MS: u64 = 15_000;
/// Whether the idle loop runs.
static RUNNING: AtomicBool = AtomicBool::new(false);
/// Set on server shutdown so the idle loop stops immediately and skips GPU restores.
static SHUTDOWN: AtomicBool = AtomicBool::new(false);
/// Set while a writer session runs (2026-11-25: the idle card-memory release waits for it).
static IN_SESSION: AtomicBool = AtomicBool::new(false);

/// Whether the writer is in a session now, or has notes left to write: either way it will use the
/// card in the next idle time, so its buffers are not idle (the idle KV window release waits).
#[must_use]
pub fn busy_or_pending() -> bool {
    IN_SESSION.load(Ordering::SeqCst) || !pending(1).is_empty()
}

struct SessionMark;
impl SessionMark {
    fn on() -> Self {
        IN_SESSION.store(true, Ordering::SeqCst);
        Self
    }
}
impl Drop for SessionMark {
    fn drop(&mut self) {
        IN_SESSION.store(false, Ordering::SeqCst);
    }
}

/// Set once the writer has finished a session in this process (the idle adapter fit waits for it
/// while notes are pending).
static WROTE_ONCE: AtomicBool = AtomicBool::new(false);
/// Results one idle session writes at most (Roadmap Phase 5, 2026-11-24). Unbounded, a first
/// session on a new project wrote every pending piece (2,999 on 319 files, hours at about 2.5 s a
/// result) before it reported or let the loop do anything else; 128 is about 5 minutes, against
/// one writer load of about 17 s.
const SESSION_RESULTS: usize = 128;
/// The writer kept on the card between idle sessions while the four MicroModels fit together.
static RESIDENT_WRITER: std::sync::Mutex<Option<Arc<nmengn::Model>>> = std::sync::Mutex::new(None);
/// List lines a one-line result may carry after a line that ends on a colon.
const LIST_LINES: usize = 6;

/// A one-line result from what the writer wrote: the first line, and when it ends on a colon the
/// list lines after it joined with "; " (their markers "- ", "* ", "1." taken off).
fn one_line_of(text: &str) -> String {
    let mut lines = text.split('\n').map(str::trim);
    let first = lines.next().unwrap_or("").to_owned();
    if !first.ends_with(':') {
        return first;
    }
    let items: Vec<String> = lines
        .skip_while(|l| l.is_empty())
        .take_while(|l| !l.is_empty())
        .take(LIST_LINES)
        .map(|l| l.trim_start_matches(['-', '*', ' ']).trim_start_matches(|c: char| c.is_ascii_digit()).trim_start_matches(['.', ')', ' ']).trim().to_owned())
        .filter(|l| !l.is_empty())
        .collect();
    if items.is_empty() { first } else { format!("{first} {}", items.join("; ")) }
}

/// Results the first idle session of a process writes at most.
const FIRST_SESSION_RESULTS: usize = 16;
/// The floor of the quiet time before the writer loads (ms).
pub const QUIET_FLOOR_MS: u64 = 5_000;
/// Longest note in tokens (one line).
pub const NOTE_TOKENS: usize = 48;
/// Commit memory the writer's load needs free (MiB), unless `NANO_WRITER_FREE_MB` names another.
pub const NEED_MB: u64 = 7_000;
/// Nearest neighbours a note must rank its own piece above.
pub const CHECK_NEIGHBOURS: usize = 8;
/// Pieces remembered from served calls (summaries) and contested ones (notes first).
const QUEUE: usize = 256;
/// Questions of served calls remembered for idle reasoning (RAM only).
const QUESTIONS: usize = 64;
/// Results asked of the material per refill inside a session.
const BATCH: usize = 256;

/// What the writer writes at idle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Job {
    /// One line on a piece, with its main identifiers: the request's drone's senses.
    Note,
    /// Two short sentences on a piece that served a pack.
    Summary,
    /// One or two lines on a file, from its pieces' notes.
    FileNote,
    /// The project's outline, from the file notes.
    Outline,
    /// Stage 5 WP-C4 W1 (Doc2Query): one to three short questions a developer would ask that a piece
    /// answers, and the names they would search for, on one line; checked by the embedder like a note.
    LikelyNames,
    /// Stage 5 WP-C4 W3: one line on a module (a folder of files), from its files' notes.
    ModuleNote,
}
impl Job {
    /// The prefix of the job's keys.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Note => "note",
            Self::Summary => "sum",
            Self::FileNote => "file",
            Self::Outline => "outline",
            Self::LikelyNames => "names",
            Self::ModuleNote => "module",
        }
    }
    /// Longest result in tokens.
    #[must_use]
    pub const fn tokens(self) -> usize {
        match self {
            Self::Note => NOTE_TOKENS,
            Self::Summary => 96,
            Self::FileNote => 64,
            Self::Outline => 400,
            Self::LikelyNames => 48,
            Self::ModuleNote => 64,
        }
    }
    /// The result is one line (the decode stops at its first line break).
    const fn one_line(self) -> bool {
        !matches!(self, Self::Outline)
    }
}

/// One piece of work: the job, its content hash, where it is from, and what it is written from.
#[derive(Clone, Debug)]
struct Task {
    job: Job,
    key: String,
    path: String,
    start: usize,
    end: usize,
    text: String,
}
impl Task {
    fn store_key(&self) -> String {
        format!("{}:{}", self.job.tag(), self.key)
    }
}

/// Results by `<job>:<content hash>` (RAM; mirrored to the cache file when there is one).
static NOTES: RwLock<BTreeMap<String, String>> = RwLock::new(BTreeMap::new());
/// The writer's values key each result was written by.
static BY: RwLock<BTreeMap<String, String>> = RwLock::new(BTreeMap::new());
/// The current outline's key.
static OUTLINE: RwLock<Option<String>> = RwLock::new(None);
/// Pieces (content hashes) to note first: blind spots and contested pieces.
static FIRST: RwLock<Vec<String>> = RwLock::new(Vec::new());
/// Pieces (content hashes) calls served, newest first.
static SERVED: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());
/// Questions of served calls, newest first (RAM only).
static ASKED: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());
/// The cache file, when results are kept on disk.
static STORE: RwLock<Option<PathBuf>> = RwLock::new(None);
/// Work the writer could not do in this process (an empty result, or a note the embedder's check
/// refused): not asked again until the material changes it (a new content hash).
static REFUSED: RwLock<BTreeSet<String>> = RwLock::new(BTreeSet::new());
/// What the sessions measured.
static STATS: Mutex<Stats> = Mutex::new(Stats::new());

/// One writer session's and the running totals' measures.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub sessions: usize,
    pub preempted: usize,
    pub notes: usize,
    pub summaries: usize,
    pub file_notes: usize,
    pub outlines: usize,
    /// Stage 5 WP-C4: likely names and module notes written, and auxiliary vectors added to the
    /// meaning lane.
    pub likely_names: usize,
    pub module_notes: usize,
    pub aux_added: usize,
    /// Notes the embedder checked, and those it dropped (not finding their piece).
    pub checked: usize,
    pub dropped: usize,
    /// Results refused as empty or cut off (a one-line result ending on a colon).
    pub empty: usize,
    /// Of those, results that ended on a colon (a list announced and not written).
    pub colon: usize,
    /// Results read from the cache file at start, and those dropped for another writer.
    pub cached: usize,
    pub stale: usize,
    /// Idle moments the writer did not load for lack of free commit memory.
    pub memory_skips: usize,
    /// Stage 5 WP-C4: the last session kept the other MicroModels resident (no release, no restore).
    pub kept_resident: bool,
    /// The live calls' card room the last session kept free (MiB): the measured working peak.
    pub headroom_mb: usize,
    /// Load of the writer (open, bind at 8 bits, seat), last session (ms).
    pub load_ms: f64,
    /// Release of the finding and ranking MicroModels' card copies, last session (ms).
    pub release_ms: f64,
    /// Restore of those card copies, last session (ms).
    pub restore_ms: f64,
    /// Time from a preempting call's start to the writer let go, last preemption (ms).
    pub let_go_ms: f64,
    /// Note times, every session (ms each).
    pub note_ms: Vec<f64>,
    /// Card bytes total and the most in use at any sample (sessions so far).
    pub card_total: usize,
    pub card_peak: usize,
    /// Device bytes of the finding and ranking MicroModels before release and after restore.
    pub before: Vec<(String, usize)>,
    pub after: Vec<(String, usize)>,
    /// Faults of the last session (an allocation that failed shows here).
    pub faults: Vec<String>,
    /// The last adapter fit's report (`NANO_ADAPTER_FIT=on`).
    pub adapter_fit: String,
}
impl Stats {
    const fn new() -> Self {
        Self {
            sessions: 0,
            preempted: 0,
            notes: 0,
            summaries: 0,
            file_notes: 0,
            outlines: 0,
            likely_names: 0,
            module_notes: 0,
            aux_added: 0,
            checked: 0,
            dropped: 0,
            empty: 0,
            colon: 0,
            cached: 0,
            stale: 0,
            memory_skips: 0,
            kept_resident: false,
            headroom_mb: 0,
            load_ms: 0.0,
            release_ms: 0.0,
            restore_ms: 0.0,
            let_go_ms: 0.0,
            note_ms: Vec::new(),
            card_total: 0,
            card_peak: 0,
            before: Vec::new(),
            after: Vec::new(),
            faults: Vec::new(),
            adapter_fit: String::new(),
        }
    }
}

/// A served call's mark: held for the call's length. Its start preempts a writer session.
pub struct Call(Instant);
impl Drop for Call {
    fn drop(&mut self) {
        let ms = u64::try_from(self.0.elapsed().as_millis()).unwrap_or(u64::MAX);
        LONGEST.fetch_max(ms, Ordering::Relaxed);
        if let Ok(mut r) = RECENT.lock() {
            if r.len() >= RECENT_CALLS {
                r.remove(0);
            }
            r.push(ms);
        }
        LAST_END.store(nmtype::now_ms(), Ordering::Relaxed);
        IN_CALL.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Mark a served call (`mcp::call`); cheap, never waits.
#[must_use]
pub fn call() -> Call {
    IN_CALL.fetch_add(1, Ordering::SeqCst);
    PREEMPT.store(true, Ordering::SeqCst);
    Call(Instant::now())
}

/// Whether idle work must stop now: shutdown, or a call started since the work began, or one is in flight.
#[must_use]
pub fn preempted() -> bool {
    SHUTDOWN.load(Ordering::SeqCst) || PREEMPT.load(Ordering::SeqCst) || IN_CALL.load(Ordering::SeqCst) > 0
}

/// Signals the idle writer loop to stop immediately (called when `serve` closes stdin).
pub fn stop() {
    SHUTDOWN.store(true, Ordering::SeqCst);
    PREEMPT.store(true, Ordering::SeqCst);
    RUNNING.store(false, Ordering::SeqCst);
    if let Ok(mut g) = RESIDENT_WRITER.lock() {
        *g = None;
    }
}

/// Roadmap Phase 4.5 (2026-11-24): `f` of the writer kept on the card now, if any, read under its
/// lock so the live view never holds the writer (its card memory goes when the session lets go).
pub fn resident_writer_view<T>(f: impl FnOnce(&nmengn::Model) -> T) -> Option<T> {
    RESIDENT_WRITER.lock().ok().and_then(|g| g.as_deref().map(f))
}

/// The writer an idle session runs, held weakly so the live view never keeps it alive: its card
/// memory goes when the session lets go (2026-11-27, card bytes per MicroModel).
static SESSION_WRITER: std::sync::Mutex<Option<std::sync::Weak<nmengn::Model>>> = std::sync::Mutex::new(None);

/// `f` of the writer on the card now: the one kept between sessions, else the one a session runs.
pub fn writer_on_card_view<T>(f: impl FnOnce(&nmengn::Model) -> T) -> Option<T> {
    writer_on_card().map(|m| f(&m))
}

/// The writer on the card now (see `writer_on_card_view`); the caller drops it at once.
#[must_use]
pub fn writer_on_card() -> Option<Arc<nmengn::Model>> {
    if let Some(m) = RESIDENT_WRITER.lock().ok().and_then(|g| g.clone()) {
        return Some(m);
    }
    SESSION_WRITER.lock().ok().and_then(|g| g.as_ref().and_then(std::sync::Weak::upgrade))
}

/// Whether the idle loop runs in this process (the hooks in the served path ask this first).
#[must_use]
pub fn running() -> bool {
    RUNNING.load(Ordering::Relaxed)
}

/// The note written for a piece (its content hash), if any.
#[must_use]
pub fn note(key: &str) -> Option<String> {
    NOTES.read().ok()?.get(&format!("note:{key}")).map(|t| crate::prefs::own_text(t, None))
}

/// The summary written for a piece (its content hash), if any.
#[must_use]
pub fn summary(key: &str) -> Option<String> {
    NOTES.read().ok()?.get(&format!("sum:{key}")).map(|t| crate::prefs::own_text(t, None))
}

/// Whether the pack carries the writer's summaries beside the cited pieces (`NANO_WRITER_PACK=on`).
#[must_use]
pub fn pack_summaries() -> bool {
    std::env::var("NANO_WRITER_PACK").is_ok_and(|v| v == "on")
}

/// Roadmap Phase 1 (2026-11-24): the world model's note for a node of the shaped tree
/// (`nmsubs::tree_shape::build`'s provider): `dir:<path>` gives the folder's module note (L3), a file
/// content hash its file note (L2), a piece key its piece note (L1); `None` when none is written.
#[must_use]
pub fn tree_note(key: &str) -> Option<String> {
    let n = NOTES.read().ok()?;
    if let Some(path) = key.strip_prefix("dir:") {
        let prefix = format!("module:{path}#");
        return n.range(prefix.clone()..).take_while(|(k, _)| k.starts_with(&prefix)).map(|(_, v)| crate::prefs::own_text(v, Some(path))).next();
    }
    n.get(&format!("file:{key}")).or_else(|| n.get(&format!("note:{key}"))).map(|t| crate::prefs::own_text(t, None))
}

/// The file note of the installed material's file at `path`, if any.
#[must_use]
pub fn file_note(path: &str) -> Option<String> {
    let o = crate::tools::installed()?;
    let sub = o.sub();
    let want = path.replace('\\', "/");
    let f = sub.files.iter().find(|f| f.path.replace('\\', "/") == want)?;
    NOTES.read().ok()?.get(&format!("file:{}", f.hash)).map(|t| crate::prefs::own_text(t, Some(&want)))
}

/// The newest project outline, if one is written.
#[must_use]
pub fn outline() -> Option<String> {
    let key = OUTLINE.read().ok()?.clone()?;
    NOTES.read().ok()?.get(&format!("outline:{key}")).map(|t| crate::prefs::own_text(t, None))
}

/// Roadmap Phase 0 item 2 (W2): the project outline for `sub`: the writer's when one is written,
/// else the structural outline of the shaped tree (top levels: crates, modules and their key
/// items; at most `max_chars`), so a broad question has the project's layout before nine in ten
/// files hold a note. `None` only when the material has no files.
#[must_use]
pub fn outline_for(sub: &nmsubs::Substrate, max_chars: usize) -> Option<String> {
    if let Some(o) = outline() {
        return Some(o);
    }
    let t = crate::coordinator::tree_for(sub).structural_outline(2, max_chars);
    (!t.trim().is_empty()).then_some(t)
}

/// Results held now (every job).
#[must_use]
pub fn notes_held() -> usize {
    NOTES.read().map_or(0, |n| n.len())
}

/// Name blind-spot items (piece content hashes) to note first.
pub fn prioritize(keys: Vec<String>) {
    if let Ok(mut f) = FIRST.write() {
        *f = keys;
    }
}

/// Hook from the served path after a call (`whole.rs`, lane T1): the pieces the pack served (their
/// content hashes) are summarised at idle, the pieces in contest are noted first, and the question
/// is remembered for idle reasoning. RAM only (bounded queues), never waits. Recorded whether or
/// not the writer runs yet (roadmap Phase 0 item 2, 2026-11-23): calls served before the writer
/// started were dropped, so the served pieces never got their notes and likely names first (a
/// prefill asks its questions before the writer starts: 444 file-order notes, no likely names).
pub fn served(question: &str, pack: &[String], contested: &[String]) {
    if let Ok(mut q) = SERVED.lock() {
        for k in pack.iter().rev() {
            q.retain(|x| x != k);
            q.push_front(k.clone());
        }
        q.truncate(QUEUE);
    }
    if let Ok(mut f) = FIRST.write() {
        for k in contested {
            if !f.contains(k) {
                f.push(k.clone());
            }
        }
        let over = f.len().saturating_sub(QUEUE);
        f.drain(..over);
    }
    if let Ok(mut a) = ASKED.lock() {
        a.push_front(question.to_owned());
        a.truncate(QUESTIONS);
    }
}

/// Questions of recent served calls, newest first (for idle reasoning over hard cases).
#[must_use]
pub fn recent_questions() -> Vec<String> {
    ASKED.lock().map(|a| a.iter().cloned().collect()).unwrap_or_default()
}

/// The measures so far.
#[must_use]
pub fn stats() -> Stats {
    STATS.lock().map(|s| s.clone()).unwrap_or_default()
}

/// Whether the switch is on: on by default now that `writer-base@3` coexists alongside all three
/// finding/ranking MicroModels within ~2.17 GiB on a 4 GB card; `NANO_WRITER_IDLE=off`,
/// `serve --writer-idle off`, or `writer.idle` set to `"off"` in `settings.json` turns it off.
#[must_use]
pub fn switched_on() -> bool {
    // Only `off` (or `0`, `false`) turns it off; any other set value keeps it on (Phase 4
    // inventory 2026-11-24: any value but `on` used to turn it off).
    if let Ok(v) = std::env::var("NANO_WRITER_IDLE") {
        return !matches!(v.as_str(), "off" | "0" | "false");
    }
    crate::whole::writer_idle_control(&nmtype::nano_root())
}

/// The quiet time asked before a session (ms): twice the 90th percentile of the last calls'
/// lengths, at least `QUIET_FLOOR_MS` and at most `QUIET_CAP_MS`. 2026-11-24: it was twice the
/// longest call ever, so one slow call (about 25 s in the soak) asked 49.5 s of quiet in every 60 s
/// idle window and the writer wrote 2 results a window; a session gives way to a call within a
/// token (preemption measured at 20 ms), so the quiet time only spaces sessions from call bursts.
#[must_use]
pub fn quiet_ms() -> u64 {
    let p90 = RECENT.lock().ok().map_or(0, |r| {
        let mut v = r.clone();
        v.sort_unstable();
        v.get(v.len().saturating_mul(9) / 10).or_else(|| v.last()).copied().unwrap_or(0)
    });
    p90.saturating_mul(2).clamp(QUIET_FLOOR_MS, QUIET_CAP_MS)
}

/// Commit memory the writer's load needs free (MiB).
#[must_use]
pub fn need_mb() -> u64 {
    std::env::var("NANO_WRITER_FREE_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(NEED_MB)
}

/// Whether the writer may load now: free commit memory at least `need_mb`. Roadmap Phase 4.5
/// (2026-11-25): under the user's host memory limit (`limits.ram_mb`) the free figure is no more
/// than the limit less this process's working set (`nmengn::limits::host_room`).
#[must_use]
pub fn memory_allows() -> bool {
    let own = nmffi::dynlib::process_memory().map_or(0, |(_, now)| now as u64);
    let cap = nmengn::limits::current().ram_cap_bytes();
    nmffi::dynlib::commit_memory().is_ok_and(|(free, _)| nmengn::limits::host_room(free, cap, own) >> 20 >= need_mb())
}

/// The embedder stays on the card beside the writer and checks each note (`NANO_WRITER_CHECK=off`
/// releases it and keeps notes unchecked).
fn check_on() -> bool {
    std::env::var("NANO_WRITER_CHECK").map_or(true, |v| v != "off")
}

/// The cache file for this session: `<memory>/world-model/notes.jsonl` when memory is on, else the
/// folder `NANO_WRITER_DIR` names; none with `NANO_WRITER_CACHE=off`.
fn store_file() -> Option<PathBuf> {
    // Roadmap Phase 1 item 1 (Luis's completion roadmap of 2026-11-23, newer than decision 4 of
    // 2026-11-17 that kept the notes in RAM): the four-level world model persists in
    // `<memory>/world-model/notes.jsonl` by default (L1 piece notes and likely names, L2 file notes,
    // L3 module notes, L4 outline); `NANO_WRITER_CACHE=off` keeps everything in RAM. A note store
    // from before, `<memory>/writer/notes.jsonl`, is carried over on first use.
    if std::env::var("NANO_WRITER_CACHE").is_ok_and(|v| v == "off") {
        return None;
    }
    // With no installed organism (a bench), the folder `NANO_WRITER_DIR` names.
    let dir = match crate::tools::installed() {
        Some(o) if o.memory_on => o.memory.lock().ok().map(|m| {
            let wm = m.dir.join("world-model");
            let old = m.dir.join("writer").join("notes.jsonl");
            if !wm.join("notes.jsonl").exists() && old.exists() && std::fs::create_dir_all(&wm).is_ok() {
                let _ = std::fs::copy(&old, wm.join("notes.jsonl"));
            }
            wm
        }),
        _ => std::env::var_os("NANO_WRITER_DIR").map(PathBuf::from),
    }?;
    Some(dir.join("notes.jsonl"))
}

/// Reads the cache file into RAM; returns the results read.
fn load_store(file: &Path) -> usize {
    let Ok(text) = std::fs::read_to_string(file) else { return 0 };
    let mut n = 0;
    let (Ok(mut notes), Ok(mut by)) = (NOTES.write(), BY.write()) else { return 0 };
    for line in text.lines() {
        let Ok(v) = nmjson::parse(line.as_bytes()) else { continue };
        let get = |k: &str| v.get(k).and_then(Value::string).map(str::to_owned).ok();
        if let (Some(k), Some(b), Some(t)) = (get("k"), get("by"), get("t")) {
            if k.starts_with("outline:")
                && let Ok(mut o) = OUTLINE.write()
            {
                *o = k.strip_prefix("outline:").map(str::to_owned);
            }
            notes.insert(k.clone(), t);
            by.insert(k, b);
            n += 1;
        }
    }
    n
}

/// Appends one result to the cache file.
fn keep(key: &str, by: &str, text: &str) {
    let Some(file) = STORE.read().ok().and_then(|s| s.clone()) else { return };
    let line = Value::Object([("k", key), ("by", by), ("t", text)].into_iter().map(|(a, b)| (a.to_owned(), Value::String(b.to_owned()))).collect()).encode().unwrap_or_default();
    if let Some(dir) = file.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&file) {
        use std::io::Write as _;
        let _ = writeln!(f, "{line}");
    }
}

/// Rewrites the cache file from what RAM holds (after `shed` or results of another writer dropped).
fn rewrite_store() {
    let Some(file) = STORE.read().ok().and_then(|s| s.clone()) else { return };
    let (Ok(notes), Ok(by)) = (NOTES.read(), BY.read()) else { return };
    let mut out = String::new();
    for (k, t) in notes.iter() {
        let b = by.get(k).cloned().unwrap_or_default();
        let line = Value::Object([("k", k.as_str()), ("by", b.as_str()), ("t", t.as_str())].into_iter().map(|(a, b)| (a.to_owned(), Value::String(b.to_owned()))).collect()).encode().unwrap_or_default();
        out.push_str(&line);
        out.push('\n');
    }
    let tmp = file.with_extension("tmp");
    if std::fs::write(&tmp, out).is_ok() {
        let _ = std::fs::rename(&tmp, &file);
    }
}

/// Empties the world model this process holds (2026-11-26, the interface's Refresh world model
/// while this process serves the project): every note, its writer, the outline and the refused
/// work leave RAM and the cache file is written again empty, so the idle writer writes L1 to L4
/// again from the material. Returns the notes dropped.
pub fn clear_store() -> usize {
    let dropped = NOTES.write().map_or(0, |mut n| std::mem::take(&mut *n).len());
    if let Ok(mut b) = BY.write() {
        b.clear();
    }
    if let Ok(mut o) = OUTLINE.write() {
        *o = None;
    }
    if let Ok(mut r) = REFUSED.write() {
        r.clear();
    }
    rewrite_store();
    dropped
}

/// Drops results of another writer (its values key), keeping the cache file in step.
fn drop_stale(by_now: &str) -> usize {
    let stale: Vec<String> = BY.read().map(|b| b.iter().filter(|(_, v)| v.as_str() != by_now).map(|(k, _)| k.clone()).collect()).unwrap_or_default();
    if stale.is_empty() {
        return 0;
    }
    if let (Ok(mut n), Ok(mut b)) = (NOTES.write(), BY.write()) {
        for k in &stale {
            n.remove(k);
            b.remove(k);
        }
    }
    rewrite_store();
    stale.len()
}

/// Start the idle loop in the serving process; false when it already runs.
#[must_use]
pub fn start(home: PathBuf) -> bool {
    if RUNNING.swap(true, Ordering::SeqCst) {
        return false;
    }
    SHUTDOWN.store(false, Ordering::SeqCst);
    LAST_END.store(nmtype::now_ms(), Ordering::Relaxed);
    if let Some(file) = store_file() {
        let n = load_store(&file);
        if let Ok(mut st) = STATS.lock() {
            st.cached = n;
        }
        if let Ok(mut s) = STORE.write() {
            *s = Some(file.clone());
        }
        eprintln!("writer: results cached in {} ({n} read)", file.display());
    }
    std::thread::spawn(move || {
        let mut told_memory = 0_u64;
        let mut told_wait = 0_u64;
        let mut seen_version = u64::MAX;
        preseat(&home);
        loop {
            std::thread::park_timeout(std::time::Duration::from_secs(1));
            if SHUTDOWN.load(Ordering::SeqCst) {
                RUNNING.store(false, Ordering::SeqCst);
                break;
            }
            if !idle_now() {
                crate::live::writer_waiting(idle_gate_word());
                // 2026-11-24: once a minute, which gate holds the writer back (cold start: the
                // first session began about 400 s after the writer was seated, the card idle).
                let now = nmtype::now_ms();
                if now.saturating_sub(told_wait) > 60_000 {
                    told_wait = now;
                    eprintln!("writer: waiting: {}", idle_gate());
                }
                continue;
            }
            // A new material version: results of pieces and files gone are dropped.
            if let Some(o) = crate::tools::installed() {
                let sub = o.sub();
                if sub.version != seen_version {
                    seen_version = sub.version;
                    let live: BTreeSet<String> = sub.chunks.iter().map(|c| c.key.clone()).chain(sub.files.iter().map(|f| f.hash.clone())).collect();
                    let gone = shed(&live);
                    if gone > 0 {
                        eprintln!("writer: {gone} results of changed material dropped");
                    }
                }
            }
            // The project's adapters, with the models already on the card (`NANO_ADAPTER_FIT=on`).
            // Roadmap Phase 5 (2026-11-24): after the writer's first session, or when it has
            // nothing pending. On a project seen for the first time the fit held the card for
            // about 22 minutes before any note was written (cold start on 319 files: no notes in
            // 30 minutes); the notes are the world model's first level, so they come first.
            if crate::adapter::fit_on()
                && (WROTE_ONCE.load(Ordering::SeqCst) || pending(1).is_empty())
                && let Some(o) = crate::tools::installed()
            {
                PREEMPT.store(false, Ordering::SeqCst);
                match crate::adapter_fit::fit_at_idle(o, &preempted) {
                    Ok(Some(report)) => {
                        eprintln!("adapter: fitted at idle\n{report}");
                        if let Ok(mut st) = STATS.lock() {
                            st.adapter_fit = report;
                        }
                    }
                    Ok(None) => {}
                    Err(e) => eprintln!("adapter: idle fit stopped: {e}"),
                }
                if !idle_now() {
                    continue;
                }
            }
            if pending(1).is_empty() {
                crate::live::writer_waiting("");
                continue;
            }
            // The commit memory guards a load; a writer kept on the card needs none (soak: a kept
            // writer waited 40 minutes on this gate after its first results).
            if !RESIDENT_WRITER.lock().is_ok_and(|g| g.is_some()) && !memory_allows() {
                if let Ok(mut st) = STATS.lock() {
                    st.memory_skips += 1;
                }
                crate::live::writer_waiting("memory");
                let now = nmtype::now_ms();
                if now.saturating_sub(told_memory) > 600_000 {
                    told_memory = now;
                    eprintln!("writer: waits for {} MiB of free commit memory", need_mb());
                }
                continue;
            }
            // The first session is short, so a new project's first notes land minutes sooner
            // (cold start: 128 results took 670 s after the load).
            let limit = if WROTE_ONCE.load(Ordering::SeqCst) { SESSION_RESULTS } else { FIRST_SESSION_RESULTS };
            eprintln!("writer: session starts (up to {limit} results; {} pending in the first batch)", pending(BATCH).len());
            match session(&home, limit) {
                Ok(n) => {
                    WROTE_ONCE.store(true, Ordering::SeqCst);
                    // 2026-11-24: the session's own measures on the same line (soak 4: 49 sessions,
                    // 0 results, card idle), so a session that writes nothing says why.
                    let st = STATS.lock().map(|s| s.clone()).unwrap_or_default();
                    let last_note = st.note_ms.last().copied().unwrap_or(0.0);
                    eprintln!(
                        "writer: {n} results written at idle ({} held); session {}: load {:.0} ms, release {:.0} ms, restore {:.0} ms, last result {:.0} ms, preempted {}, kept resident {}, headroom {} MiB, refused {} (empty or cut off {} of which ending on a colon {}, checked {}, dropped by the check {}){}",
                        notes_held(),
                        st.sessions,
                        st.load_ms,
                        st.release_ms,
                        st.restore_ms,
                        last_note,
                        st.preempted,
                        st.kept_resident,
                        st.headroom_mb,
                        REFUSED.read().map_or(0, |r| r.len()),
                        st.empty,
                        st.colon,
                        st.checked,
                        st.dropped,
                        st.faults.first().map_or(String::new(), |f| format!(", fault: {f}"))
                    );
                }
                Err(e) => eprintln!("writer: {e}"),
            }
        }
    });
    true
}

/// Roadmap Phase 5 (2026-11-24, Luis: all four from the start): once the finding and ranking
/// MicroModels are bound, the writer is seated beside them in the background when the card has room
/// for it and the live calls' room (the same rule as a session's `keep_all`), and kept. On a new
/// project the writer otherwise loaded only at its first idle session (cold start: about 510 s
/// after the tools were ready), and each session paid the load. Skipped with
/// `NANO_WRITER_KEEP_RESIDENT=off`, without enough free commit memory, or without room on the card.
fn preseat(home: &Path) {
    if std::env::var("NANO_WRITER_KEEP_RESIDENT").as_deref() == Ok("off") {
        return;
    }
    // Wait (at most ten minutes) for the finders to be seated, so the room is measured beside them.
    // 2026-11-25: all three of them (reader, embedder, reranker), not the first: with only the
    // embedder and the reranker bound the reader was still loading, its transient buffers held
    // the card, and the writer read too little room and stayed off the card. Two minutes after the
    // first is bound the measure goes ahead with what is bound (a reader that loads later).
    let mut first_bound: Option<Instant> = None;
    for _ in 0..600 {
        if SHUTDOWN.load(Ordering::SeqCst) {
            return;
        }
        let bound = crate::neural::bound_models().len();
        if crate::tools::installed().is_some() && bound > 0 {
            let since = *first_bound.get_or_insert_with(Instant::now);
            if bound >= 3 || since.elapsed() >= std::time::Duration::from_secs(120) {
                break;
            }
        }
        std::thread::park_timeout(std::time::Duration::from_secs(1));
    }
    // A moment for the last load's transient buffers to be released before the room is read.
    std::thread::park_timeout(std::time::Duration::from_secs(2));
    if crate::neural::bound_models().is_empty() || !memory_allows() {
        return;
    }
    // Roadmap Phase 4.5 (2026-11-25): the effective free memory under the user's card limit
    // (`memory_now`), the same figure as a session's `keep_all`.
    let free_mb = nmengn::residency::get().memory_now().map_or(0, |(free, _)| free >> 20);
    let headroom_mb = nmengn::residency::get().reserve().0 >> 20;
    if free_mb < writer_need_mb() + headroom_mb {
        eprintln!("writer: not seated at start ({free_mb} MiB free on the card, {} MiB needed with the live calls' room); it loads per idle session", writer_need_mb() + headroom_mb);
        return;
    }
    if let Ok(mut p) = nmengn::model::PRECISION.write()
        && p.writer.is_none()
    {
        p.writer = Some(8);
    }
    let t = Instant::now();
    match crate::neural::writer_model(home, true) {
        Ok(w) => {
            // 2026-11-25: on its open weights (no base cartridge on the shelf) the writer's
            // parameters are prepared on the host and reach the card at its first generation, so
            // the card held about 1 MiB here and that figure was kept as the writer's measured size
            // (`WRITER_MB`), which the sessions plan with. Under the size of a seated writer the
            // planned size is kept instead and the line says when the parameters arrive.
            let held = w.pop.device_bytes() >> 20;
            // The resident size measured for the default writer; for a writer the user put on
            // another model (`nmtype::role_models`), half its released file size (8 bits a weight).
            let id = nmtype::role_models::model("writer");
            let planned = if nmtype::role_models::default_model("writer") == Some(id.as_str()) {
                WRITER_RESIDENT_MB
            } else {
                std::fs::read(nmtype::models_dir(home).join(&id).join("part.json"))
                    .ok()
                    .and_then(|b| nmjson::parse(&b).ok())
                    .and_then(|v| v.get("weights").and_then(|x| x.get("bytes")).and_then(nmjson::Value::integer).ok())
                    .map_or(WRITER_RESIDENT_MB, |b| usize::try_from(b >> 21).unwrap_or(WRITER_RESIDENT_MB))
            };
            let first_use = held < planned / 4;
            let mb = if first_use { planned } else { held.max(1) };
            WRITER_MB.store(mb, Ordering::Relaxed);
            if let Ok(mut g) = RESIDENT_WRITER.lock()
                && !SHUTDOWN.load(Ordering::SeqCst)
            {
                *g = Some(w);
            }
            // Roadmap Phase 4.5 (2026-11-24): the live view sees the writer seated at start.
            crate::live::writer_seated("seated", mb, u64::try_from(t.elapsed().as_millis()).unwrap_or(u64::MAX));
            let after = nmengn::residency::get().memory_now().map_or(0, |(free, _)| free >> 20);
            if first_use {
                eprintln!(
                    "writer: bound beside the other three at start in {:.1} s on its open weights; its parameters reach the card at its first generation (about {mb} MiB planned; {after} MiB free on the card now)",
                    t.elapsed().as_secs_f64()
                );
            } else {
                eprintln!("writer: seated beside the other three at start in {:.1} s ({mb} MiB; {after} MiB free on the card)", t.elapsed().as_secs_f64());
            }
        }
        Err(e) => eprintln!("writer: not seated at start: {e}"),
    }
}

/// No call in flight, quiet long enough, the meaning lane not building.
/// Which of `idle_now`'s gates is closed now, in words.
/// The gate that holds the writer back, as the fixed word the live view shows (`idle_gate` is the
/// sentence): a call in flight, the quiet time not yet passed, else the dense lane not ready.
fn idle_gate_word() -> &'static str {
    if IN_CALL.load(Ordering::SeqCst) > 0 {
        "call"
    } else if nmtype::now_ms().saturating_sub(LAST_END.load(Ordering::Relaxed)) < quiet_ms() {
        "quiet"
    } else {
        "index"
    }
}

fn idle_gate() -> String {
    if IN_CALL.load(Ordering::SeqCst) > 0 {
        return "a call is in flight".to_owned();
    }
    let quiet = nmtype::now_ms().saturating_sub(LAST_END.load(Ordering::Relaxed));
    if quiet < quiet_ms() {
        return format!("quiet {quiet} ms of the {} ms asked (twice the recent calls' 90th percentile, {QUIET_FLOOR_MS} to {QUIET_CAP_MS} ms)", quiet_ms());
    }
    let Some(o) = crate::tools::installed() else { return "no organism installed".to_owned() };
    if RESIDENT_WRITER.lock().is_ok_and(|g| g.is_some()) {
        return "open".to_owned();
    }
    let sub = o.sub();
    match sub.dense.read().ok().and_then(|d| d.clone()) {
        Some(l) if !l.ready(&sub) => {
            let (covered, total) = l.coverage(&sub);
            format!("the meaning lane is not ready ({covered} of {total} pieces)")
        }
        _ => "open".to_owned(),
    }
}

fn idle_now() -> bool {
    if IN_CALL.load(Ordering::SeqCst) > 0 || nmtype::now_ms().saturating_sub(LAST_END.load(Ordering::Relaxed)) < quiet_ms() {
        return false;
    }
    let Some(o) = crate::tools::installed() else { return false };
    // 2026-11-24: a writer already seated beside the other three needs no load, so it works while
    // the meaning lane builds (both go through the device gate, so their work interleaves). Cold
    // start on 319 files: the lane covered 2,999 pieces at about 6.5 a second and the first notes
    // waited 481 s for it with the writer seated since 87 s. A writer that would have to load still
    // waits for the lane, since its load releases the finders the lane is using.
    if RESIDENT_WRITER.lock().is_ok_and(|g| g.is_some()) {
        return true;
    }
    let sub = o.sub();
    let lane = sub.dense.read().ok().and_then(|d| d.clone());
    lane.is_none_or(|l| l.ready(&sub))
}

/// Work without a result, at most `limit`, in the order of the module's step 5.
fn pending(limit: usize) -> Vec<Task> {
    let Some(o) = crate::tools::installed() else { return Vec::new() };
    let sub = o.sub();
    let held: BTreeSet<String> = NOTES.read().map(|n| n.keys().cloned().collect()).unwrap_or_default();
    let refused: BTreeSet<String> = REFUSED.read().map(|r| r.clone()).unwrap_or_default();
    let by_key: BTreeMap<&str, usize> = sub.chunks.iter().enumerate().map(|(i, c)| (c.key.as_str(), i)).collect();
    let path_of = |c: &nmsubs::Chunk| sub.files.get(c.file).map_or_else(String::new, |f| f.path.replace('\\', "/"));
    let piece = |job: Job, i: usize| -> Option<Task> {
        let c = sub.chunks.get(i)?;
        (!c.text.trim().is_empty()).then(|| Task { job, key: c.key.clone(), path: path_of(c), start: c.start, end: c.end, text: c.text.clone() })
    };
    let mut out: Vec<Task> = Vec::new();
    let mut taken: BTreeSet<String> = BTreeSet::new();
    let mut push = |t: Task, out: &mut Vec<Task>| {
        let k = t.store_key();
        if !held.contains(&k) && !refused.contains(&k) && taken.insert(k) {
            out.push(t);
        }
    };
    // Notes for contested and blind pieces first.
    for k in FIRST.read().map(|f| f.clone()).unwrap_or_default() {
        if out.len() >= limit {
            return out;
        }
        if let Some(t) = by_key.get(k.as_str()).and_then(|i| piece(Job::Note, *i)) {
            push(t, &mut out);
        }
    }
    // Summaries of served pieces, newest first, only when packs read them (`NANO_WRITER_PACK=on`;
    // Stage 5 WP-C4: otherwise the idle time goes to the notes and likely names the lane uses).
    let summaries: Vec<String> = if pack_summaries() { SERVED.lock().map(|q| q.iter().cloned().collect::<Vec<_>>()).unwrap_or_default() } else { Vec::new() };
    for k in summaries {
        if out.len() >= limit {
            return out;
        }
        if let Some(t) = by_key.get(k.as_str()).and_then(|i| piece(Job::Summary, *i)) {
            push(t, &mut out);
        }
    }
    // Stage 5 WP-C4: the pieces calls served get their note and then their likely names before any
    // other piece is noted (the likely names follow once the note is held).
    let served_idx: Vec<usize> = SERVED.lock().map(|q| q.iter().filter_map(|k| by_key.get(k.as_str()).copied()).collect()).unwrap_or_default();
    for &i in &served_idx {
        if out.len() >= limit {
            return out;
        }
        if let Some(t) = piece(Job::Note, i) {
            push(t, &mut out);
        }
        if sub.chunks.get(i).is_some_and(|c| held.contains(&format!("note:{}", c.key)))
            && let Some(t) = piece(Job::LikelyNames, i)
        {
            push(t, &mut out);
        }
    }
    // Stage 5 WP-C4 W1: likely names for pieces that have their note, served pieces first.
    let mut names_order: Vec<usize> = SERVED.lock().map(|q| q.iter().filter_map(|k| by_key.get(k.as_str()).copied()).collect()).unwrap_or_default();
    names_order.extend(0..sub.chunks.len());
    for i in names_order {
        if out.len() >= limit {
            return out;
        }
        if sub.chunks.get(i).is_some_and(|c| held.contains(&format!("note:{}", c.key)))
            && let Some(t) = piece(Job::LikelyNames, i)
        {
            push(t, &mut out);
        }
    }
    // File notes, once each of the file's pieces has its note.
    let mut lines_of: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    let mut whole: BTreeSet<usize> = (0..sub.files.len()).collect();
    for c in &sub.chunks {
        if c.text.trim().is_empty() {
            continue;
        }
        match held.contains(&format!("note:{}", c.key)).then(|| NOTES.read().ok().and_then(|n| n.get(&format!("note:{}", c.key)).cloned())).flatten() {
            Some(nt) => lines_of.entry(c.file).or_default().push(format!("lines {}-{}: {nt}", c.start, c.end)),
            None if refused.contains(&format!("note:{}", c.key)) => {}
            None => {
                whole.remove(&c.file);
            }
        }
    }
    let mut with_note = 0;
    let mut file_lines: Vec<String> = Vec::new();
    for (fi, f) in sub.files.iter().enumerate() {
        let path = f.path.replace('\\', "/");
        if let Some(nt) = NOTES.read().ok().and_then(|n| n.get(&format!("file:{}", f.hash)).cloned()) {
            with_note += 1;
            file_lines.push(format!("{path}: {nt}"));
            continue;
        }
        if out.len() >= limit {
            return out;
        }
        if whole.contains(&fi)
            && let Some(lines) = lines_of.get(&fi)
        {
            let text: String = lines.iter().take(60).cloned().collect::<Vec<_>>().join("\n");
            push(Task { job: Job::FileNote, key: f.hash.clone(), path, start: 0, end: 0, text }, &mut out);
        }
    }
    // Stage 5 WP-C4 W3: a note per module (a folder holding at least two noted files), from its files'
    // notes; keyed by the module's path and its files' notes, so a changed file brings a new one.
    let mut by_module: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for l in &file_lines {
        let path = l.split(": ").next().unwrap_or("");
        let module = path.rsplit_once('/').map_or(".", |(d, _)| d).to_owned();
        by_module.entry(module).or_default().push(l.clone());
    }
    for (module, lines) in &by_module {
        if out.len() >= limit {
            return out;
        }
        if lines.len() < 2 {
            continue;
        }
        let text: String = lines.iter().take(40).cloned().collect::<Vec<_>>().join("\n");
        // Roadmap Phase 1 (2026-11-24): the folder is in the key (`module:<path>#<hash>`), so the
        // store says which folder a note describes (reloaded into the meaning lane after a restart)
        // and a folder's older notes can be dropped when a new one is accepted.
        let key = format!("{module}#{}", nmhash::sha256_bytes(format!("{module}\n{text}").as_bytes()));
        push(Task { job: Job::ModuleNote, key, path: module.clone(), start: 0, end: 0, text }, &mut out);
    }
    // The outline (L4). Roadmap Phase 1 (2026-11-24): from the folders' module notes (L3) once nine
    // in ten folders that need one hold one, so it is written again only when a folder's note
    // changes (a module note changes only when one of its files' notes does), not on every file
    // note; else from the file notes once nine in ten files have one, as before.
    let module_lines: Vec<String> = NOTES
        .read()
        .map(|n| {
            n.iter()
                .filter_map(|(k, v)| k.strip_prefix("module:").and_then(|r| r.split_once('#')).map(|(p, _)| format!("{p}/: {v}")))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let folders_needing = by_module.values().filter(|l| l.len() >= 2).count();
    let from_modules = folders_needing > 0 && module_lines.len() * 10 >= folders_needing * 9;
    let outline_lines: &[String] = if from_modules { &module_lines } else { &file_lines };
    if !sub.files.is_empty() && (from_modules || with_note * 10 >= sub.files.len() * 9) && out.len() < limit {
        let key = nmhash::sha256_bytes(outline_lines.join("\n").as_bytes());
        let mut text = String::new();
        for l in outline_lines {
            if text.len() + l.len() > 12_000 {
                break;
            }
            text.push_str(l);
            text.push('\n');
        }
        if let Ok(mut o) = OUTLINE.write()
            && held.contains(&format!("outline:{key}"))
        {
            *o = Some(key.clone());
        }
        push(Task { job: Job::Outline, key, path: String::new(), start: 0, end: 0, text }, &mut out);
    }
    // Notes for every other piece, in file order, last (roadmap Phase 0 item 2, 2026-11-23): the
    // derived jobs above (likely names of noted pieces, file notes of fully noted files, module
    // notes, the outline) run as soon as their inputs are held; before, this loop over every piece
    // came first, so they waited for the whole material to be noted (in practice never).
    let mut order: Vec<usize> = (0..sub.chunks.len()).collect();
    order.sort_by(|a, b| {
        let (ca, cb) = (&sub.chunks[*a], &sub.chunks[*b]);
        ca.file.cmp(&cb.file).then(ca.start.cmp(&cb.start))
    });
    // Each piece's likely names follow its note (roadmap Phase 0 item 2: names came only after every
    // piece of the material had a note, so in practice never).
    for i in order {
        if out.len() >= limit {
            return out;
        }
        if let Some(t) = piece(Job::Note, i) {
            push(t, &mut out);
        }
        if sub.chunks.get(i).is_some_and(|c| held.contains(&format!("note:{}", c.key)))
            && let Some(t) = piece(Job::LikelyNames, i)
        {
            push(t, &mut out);
        }
    }
    out
}

/// Card bytes in use now and the total, when a card is there (the driver's figures: the card's
/// real use, whatever the user's limit).
fn card_now() -> Option<(usize, usize)> {
    nmengn::residency::get().memory_driver().map(|(free, total)| (total.saturating_sub(free), total))
}

fn sample(st: &mut Stats) {
    if let Some((used, total)) = card_now() {
        st.card_total = total;
        st.card_peak = st.card_peak.max(used);
    }
}

/// One writer session: release, load at 8 bits, write up to `limit` results (stopping when a call
/// arrives), let go, restore. Returns the results written.
/// # Errors
/// The writer cannot be opened; a result's tokens cannot be made.
pub fn session(home: &Path, limit: usize) -> Result<usize, Fault> {
    if SHUTDOWN.load(Ordering::SeqCst) || pending(1).is_empty() || IN_CALL.load(Ordering::SeqCst) > 0 {
        return Ok(0);
    }
    PREEMPT.store(false, Ordering::SeqCst);
    let _mark = SessionMark::on();
    let mut st = STATS.lock().map(|s| s.clone()).unwrap_or_default();
    st.sessions += 1;
    st.faults.clear();
    sample(&mut st);
    // 3. All four MicroModels coexist on a 4 GB card; the finders are released only when the card
    // has no room for the writer beside them (`keep_all` below).
    let t = Instant::now();
    let keep_embedder = check_on();
    let finders = crate::neural::bound_models();
    st.before = finders.iter().map(|(n, m)| ((*n).to_owned(), m.pop.device_bytes())).collect();
    // Roadmap Phase 4.5 (2026-11-25): the effective free memory under the user's card limit
    // (`limits.vram_mb`; the driver's figure with `auto`), so all four seated versus evict and
    // restore follows the limit.
    let free_mb = nmengn::residency::get().memory_now().map_or(0, |(free, _)| free >> 20);
    // Stage 5 WP-C4: `NANO_WRITER_KEEP_RESIDENT=on` never releases the finding and ranking
    // MicroModels, `off` always does, `auto` (default) as below.
    let writer_held = RESIDENT_WRITER.lock().is_ok_and(|g| g.is_some());
    // Roadmap Phase 5 (2026-11-24, Luis: all four MicroModels on the card): the writer stays on
    // the card beside the other three while the card's measured free memory leaves the live calls
    // their working room (`LIVE_HEADROOM_MB`). The earlier rule capped the whole card's use at
    // 3,200 MiB (other programs' memory included) and asked 1,500 MiB free, so the writer was
    // dropped and reloaded although the four fit (1,373 MiB in use with three seated).
    // It adapts to the machine: the live calls' room is the measured working peak of this
    // process's GPU runs (an eighth of the card until one is measured), and the writer's size is
    // its measured card bytes once loaded. A large card keeps all four seated; a small one falls
    // back to releasing the finders for the writer's session and restoring them after.
    let headroom_mb = nmengn::residency::get().reserve().0 >> 20;
    let keep_all = match std::env::var("NANO_WRITER_KEEP_RESIDENT").as_deref() {
        Ok("on") => true,
        Ok("off") => false,
        _ if writer_held => free_mb >= headroom_mb,
        _ => free_mb >= writer_need_mb() + headroom_mb,
    };
    st.headroom_mb = headroom_mb;
    if !keep_all && let Ok(mut g) = RESIDENT_WRITER.lock() {
        // Card memory is tight: the kept writer goes before the finders are released.
        *g = None;
    }
    st.kept_resident = keep_all;
    // Roadmap Phase 4.5 (2026-11-24): the session's start in the live view.
    crate::live::writer_session(keep_all, headroom_mb);
    let session_t = Instant::now();
    let released: Vec<&(&'static str, Arc<nmengn::Model>)> =
        if keep_all { Vec::new() } else { finders.iter().filter(|(n, _)| !(keep_embedder && *n == "embedder")).collect() };
    for (_, m) in &released {
        if m.pop.active() == 0 {
            let _ = m.pop.try_release_device();
        }
    }
    if !released.is_empty() {
        let _ = nmengn::residency::get().release_windows();
    }
    st.release_ms = t.elapsed().as_secs_f64() * 1000.0;
    sample(&mut st);
    // 4. Load the writer at 8 bits unless the settings name its width.
    let mut written = 0;
    if !preempted() {
        if let Ok(mut p) = nmengn::model::PRECISION.write()
            && p.writer.is_none()
        {
            p.writer = Some(8);
        }
        let t = Instant::now();
        let kept = if keep_all { RESIDENT_WRITER.lock().ok().and_then(|mut g| g.take()) } else { None };
        let was_kept = kept.is_some();
        let opened = kept.map_or_else(|| crate::neural::writer_model(home, true), Ok);
        st.load_ms = t.elapsed().as_secs_f64() * 1000.0;
        sample(&mut st);
        match opened {
            Ok(w) => {
                if let Ok(mut g) = SESSION_WRITER.lock() {
                    *g = Some(Arc::downgrade(&w));
                }
                WRITER_MB.store((w.pop.device_bytes() >> 20).max(1), Ordering::Relaxed);
                if !was_kept {
                    crate::live::writer_seated("loaded", (w.pop.device_bytes() >> 20).max(1), u64::try_from(t.elapsed().as_millis()).unwrap_or(u64::MAX));
                }
                if !preempted() {
                    let by = crate::neural::parameters_key(&w);
                    st.stale += drop_stale(&by);
                    let embedder = if keep_embedder { finders.iter().find(|(n, _)| *n == "embedder").map(|(_, m)| Arc::clone(m)) } else { None };
                    written = write_all(&w, &by, embedder.as_deref(), limit, &mut st);
                }
                // 7. Let go of the writer: the population is dropped with its last holder.
                // Roadmap Phase 5 (2026-11-24): kept on the card for the next session when the
                // four MicroModels fit together (`keep_all`). Opened and dropped every session, it
                // paid about 17 s a load in every idle window; in the soak (60 s windows behind the
                // muscle and learning cycles) each load ended as the next call arrived, 0 results.
                let t = Instant::now();
                if keep_all && !SHUTDOWN.load(Ordering::SeqCst) && let Ok(mut g) = RESIDENT_WRITER.lock() {
                    // Weight placement (2026-11-27): seated on the card, its host copies of the
                    // linear maps the card holds go (read again from the files on demand).
                    w.drop_host();
                    *g = Some(w);
                } else {
                    drop(w);
                }
                if preempted() {
                    st.let_go_ms += t.elapsed().as_secs_f64() * 1000.0;
                }
            }
            Err(e) => st.faults.push(format!("writer: {e}")),
        }
    }
    // Restore the released MicroModels: only missing pieces move (skipped on server shutdown).
    let t = Instant::now();
    if !SHUTDOWN.load(Ordering::SeqCst) {
        for (name, m) in &released {
            if SHUTDOWN.load(Ordering::SeqCst) {
                break;
            }
            let Some(g) = m.pop.gpu.as_ref() else { continue };
            let full = *name == "base" && crate::joint_read::full_head();
            match g.gpu.copy_stream().map_err(bad).and_then(|s| m.stage_parts(&s, full)) {
                Ok(_) => {}
                Err(e) => st.faults.push(format!("restore {name}: {e}")),
            }
        }
    }
    st.restore_ms = t.elapsed().as_secs_f64() * 1000.0;
    // Weight placement (2026-11-27): the restore read the dropped host copies again from their
    // files and dropped them again once staged (`Model::stage_parts`); the weights area is told.
    crate::preload::note_weights();
    st.after = finders.iter().map(|(n, m)| ((*n).to_owned(), m.pop.device_bytes())).collect();
    sample(&mut st);
    crate::live::writer_session_end(written, RESIDENT_WRITER.lock().is_ok_and(|g| g.is_some()), u64::try_from(session_t.elapsed().as_millis()).unwrap_or(u64::MAX));
    if let Ok(mut g) = STATS.lock() {
        *g = st;
    }
    Ok(written)
}

/// The Writer at serve start (2026-11-27, `crate::preload`): seated on the card beside the other
/// three MicroModels when the card's effective free memory (under `limits.vram_mb`) holds it with
/// the live calls' room, and the machine has the commit memory its load needs (`memory_allows`);
/// it is then the resident writer the idle sessions use, and it yields to calls as before (a
/// session gives way to a call within a token). Otherwise nothing loads now and the reason is
/// returned: the Writer loads per idle session and stays seated once room frees (`session`).
/// Returns (seated, card MiB, note).
/// # Errors
/// The Writer could not be opened although it fit.
pub fn seat_at_start(home: &Path) -> Result<(bool, u64, String), String> {
    if RESIDENT_WRITER.lock().is_ok_and(|g| g.is_some()) {
        let mb = resident_writer_view(|w| (w.pop.device_bytes() >> 20) as u64).unwrap_or(0);
        return Ok((true, mb, "already seated".to_owned()));
    }
    let res = nmengn::residency::get();
    if res.card().is_none() && nmengn::device::open().is_err() {
        return Ok((false, 0, "no graphics card in this process: the Writer loads on the processor per idle session".to_owned()));
    }
    let free_mb = res.memory_now().map_or(0, |(free, _)| free >> 20);
    let headroom_mb = res.reserve().0 >> 20;
    let need = writer_need_mb();
    if free_mb < need + headroom_mb {
        return Ok((
            false,
            0,
            format!(
                "the card has {free_mb} MiB free (limits.vram_mb {}); the Writer needs about {need} MiB beside the live calls' {headroom_mb} MiB, so it loads per idle session (evict and restore) and stays seated once room frees",
                nmengn::limits::current().setting(nmengn::limits::VRAM_KEY)
            ),
        ));
    }
    if !memory_allows() {
        return Ok((false, 0, format!("the machine has less than {} MiB of commit memory free for the Writer's load; it loads per idle session once memory frees", need_mb())));
    }
    // Host memory (2026-11-27): while the Writer loads, its weights are held in RAM beside their
    // 8-bit copy (about one and a half times its weight files; the host copies of its linear maps
    // are dropped once staged on the card, `Model::drop_host`), so it is seated only while this
    // process stays within its budget and the machine keeps its floor available after the load
    // (`nmengn::limits::host_floor`, `process_budget`).
    let host_need = writer_host_mb(home);
    if let (Ok((available, total)), Ok((_, own))) = (nmffi::dynlib::system_memory(), nmffi::dynlib::process_memory()) {
        let (mb, own) = (|b: u64| b >> 20, own as u64);
        let floor = nmengn::limits::host_floor(total);
        let budget = nmengn::limits::process_budget(total, nmengn::limits::current().ram_cap_bytes());
        if mb(own) + host_need > mb(budget) || mb(available) < mb(floor) + host_need {
            return Ok((
                false,
                0,
                format!(
                    "host memory: the Writer holds about {host_need} MiB in RAM; this process holds {} MiB of its {} MiB budget (limits.ram_mb {}) and the machine has {} MiB available above its {} MiB floor, so it loads per idle session when memory allows and stays seated once room frees",
                    mb(own),
                    mb(budget),
                    nmengn::limits::current().setting(nmengn::limits::RAM_KEY),
                    mb(available.saturating_sub(floor)),
                    mb(floor)
                ),
            ));
        }
    }
    if let Ok(mut p) = nmengn::model::PRECISION.write()
        && p.writer.is_none()
    {
        p.writer = Some(8);
    }
    let t = Instant::now();
    let w = crate::neural::writer_model(home, true).map_err(|e| e.to_string())?;
    if let Err(e) = crate::preload::stage(&w) {
        eprintln!("writer: not staged on the card at start ({e}); it uploads at its first run");
    }
    let mb = (w.pop.device_bytes() >> 20).max(1);
    WRITER_MB.store(mb, Ordering::Relaxed);
    crate::live::writer_seated("loaded", mb, u64::try_from(t.elapsed().as_millis()).unwrap_or(u64::MAX));
    let on_card = w.pop.gpu.is_some();
    let note = if on_card { w.pop.gpu_note.clone() } else { format!("on the processor: {}", w.pop.gpu_note) };
    if let Ok(mut g) = RESIDENT_WRITER.lock() {
        *g = Some(w);
    }
    Ok((true, mb as u64, note))
}

/// The RAM the Writer's load holds on a card (MiB). Weight placement (2026-11-27): its linear maps
/// are read a tensor at a time and their 8-bit packets made in staging batches that go straight to
/// the card (`nmengn::model::STAGE_BATCH`), so the load holds its token table, head and norms
/// (about a sixth of its files) plus one batch of packets with the threads' working copies; 5,000
/// MiB when the files cannot be read.
fn writer_host_mb(home: &Path) -> u64 {
    let dir = nmtype::models_dir(home).join(nmtype::role_models::model("writer")).join("files");
    let bytes: u64 = std::fs::read_dir(&dir)
        .map(|it| it.filter_map(Result::ok).filter_map(|e| e.metadata().ok()).filter(std::fs::Metadata::is_file).map(|m| m.len()).sum())
        .unwrap_or(0);
    let batch_mb = (nmengn::model::STAGE_BATCH as u64 >> 20) * 2;
    if bytes == 0 { 5_000 } else { (bytes >> 20) / 6 + batch_mb }
}

/// Stage 5 WP-C4: the writer's card memory at 8 bits with its active head (about 1.17 GiB), used
/// until the writer has been loaded once and measured (`WRITER_MB`).
const WRITER_RESIDENT_MB: usize = 1_200;
/// The writer's card memory as measured after its last load (MiB; 0 until then).
static WRITER_MB: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// The card memory a writer load needs: its measured size, else `WRITER_RESIDENT_MB`.
fn writer_need_mb() -> usize {
    match WRITER_MB.load(Ordering::Relaxed) {
        0 => WRITER_RESIDENT_MB,
        m => m,
    }
}

/// The chat prompt for one task (the chat form without thinking).
fn prompt(t: &Task) -> String {
    let ask = match t.job {
        // Short and specific (lead, 2026-11-19 14:50 UTC): what the piece does and the names it
        // defines, one line, no lead-in such as "This part of the file".
        Job::Note => format!(
            "{} lines {}-{}:\n```\n{}\n```\nWrite one line of at most 25 words, plain text, no markdown: first what this code or text does (start with a verb; do not name the file or say \"this part\"), then \"Defines:\" and the names it defines (or \"Defines: none\").",
            t.path, t.start, t.end, t.text
        ),
        Job::Summary => format!(
            "In at most two short sentences of plain words, on one line, say what this part of {} (lines {}-{}) does and when someone working on the project would need to read it.\n\n```\n{}\n```",
            t.path, t.start, t.end, t.text
        ),
        Job::FileNote => format!("Here are notes on the parts of the file {}:\n{}\n\nIn one short line of plain words, say what this file is for.", t.path, t.text),
        Job::Outline => format!(
            "Here is one line on each file or folder of a software project:\n{}\nWrite a short outline of the project in plain words: its main parts, what each is for, and where to start reading. At most twelve lines.",
            t.text
        ),
        Job::LikelyNames => format!(
            "{} lines {}-{}:\n```\n{}\n```\nOn one line of plain text, no markdown: write one to three short questions a developer would ask that this code answers, then \"Names:\" and the names they would search for. Separate the parts with semicolons.",
            t.path, t.start, t.end, t.text
        ),
        Job::ModuleNote => format!("Here are notes on the files of the folder {}:\n{}\n\nIn one short line of plain words, say what this folder of the project is for.", t.path, t.text),
    };
    format!("<|im_start|>user\n{ask}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

/// Whether a note, read as a request by the embedder, ranks its own piece above the piece's
/// nearest neighbours in the meaning lane. None when it cannot be checked (no lane, no vector).
fn note_lands(embedder: &nmengn::Model, key: &str, note: &str) -> Option<bool> {
    let o = crate::tools::installed()?;
    let sub = o.sub();
    let lane = sub.dense.read().ok().and_then(|d| d.clone())?;
    let own = lane.vector(key)?;
    let dev = if embedder.pop.gpu.is_some() { nmengn::placement::Device::Gpu } else { nmengn::placement::Device::Cpu };
    let q = nmengn::roles::card_embed_on(embedder, dev, &format!("Instruct: {}\nQuery:{note}", nmsubs::dense::task_for(note))).ok()?;
    // 2026-11-25: each neighbour carries whether it is a sibling of the note's piece (the same file,
    // touching or overlapping its lines): a note about a function split across two pieces rightly
    // finds both.
    let me = sub.chunks.iter().find(|c| c.key == key);
    let sibling = |c: &nmsubs::Chunk| me.is_some_and(|m| m.file == c.file && c.start <= m.end + 1 && m.start <= c.end + 1);
    let mut near: Vec<(f32, Arc<Vec<f32>>, bool)> =
        sub.chunks.iter().filter(|c| c.key != key).filter_map(|c| lane.vector(&c.key).map(|v| (crate::adapter::dot(&own, &v), v, sibling(c)))).collect();
    near.sort_by(|a, b| b.0.total_cmp(&a.0));
    near.truncate(CHECK_NEIGHBOURS);
    let mine = crate::adapter::dot(&q, &own);
    let others: Vec<f32> = near.iter().map(|(_, v, _)| crate::adapter::dot(&q, v)).collect();
    let strict = others.iter().all(|s| *s < mine);
    // Every neighbour that outranks the note's piece is a sibling of it.
    let only_siblings = near.iter().zip(&others).all(|((_, _, sib), s)| *s < mine || *sib);
    // Such a note passes when it says something (six words or more). Measured 2026-11-25 (20-minute
    // soak, 80 notes checked, 19 dropped): of the 9 dropped only for siblings, 6 were specific and
    // right ("Quarantines a node by binding a swap, removes the quarantine if it failed") and 3 weak,
    // of which the floor keeps out the shortest ("defines: nmapi"). On by default;
    // `NANO_WRITER_CHECK_SIBLINGS=off` for diagnosis.
    let says_something = note.split_whitespace().count() >= 6;
    let ok = strict || (only_siblings && says_something && std::env::var("NANO_WRITER_CHECK_SIBLINGS").map_or(true, |v| v != "off"));
    // 2026-11-25 (diagnosis of the check, which dropped 55 of 179 notes in a soak): with
    // `NANO_WRITER_CHECK_LOG=<file>` each checked note is recorded with the rank its own piece took
    // among its neighbours (1 is first), the margin to the best neighbour, and the note itself.
    if let Some(path) = std::env::var_os("NANO_WRITER_CHECK_LOG") {
        let best = others.iter().copied().fold(f32::MIN, f32::max);
        let rank = 1 + others.iter().filter(|s| **s >= mine).count();
        let row = nmjson::Value::Object(
            [
                ("key", nmjson::Value::String(key.to_owned())),
                ("note", nmjson::Value::String(note.to_owned())),
                ("rank", nmjson::Value::Number(rank.to_string())),
                ("mine", nmjson::Value::Number(format!("{mine:.4}"))),
                ("best_other", nmjson::Value::Number(format!("{best:.4}"))),
                ("outranked_by_siblings_only", nmjson::Value::Bool(!strict && only_siblings)),
                ("passed", nmjson::Value::Bool(ok)),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect(),
        );
        if let (Ok(line), Ok(mut f)) = (row.encode(), std::fs::OpenOptions::new().create(true).append(true).open(path)) {
            let _ = std::io::Write::write_all(&mut f, format!("{line}\n").as_bytes());
        }
    }
    Some(ok)
}

/// Tokens of the prompt read per forward: a call that arrives waits at most one piece of the
/// prompt (lead, 2026-11-19 14:50 UTC; the mid-write call cost 1.83 s against 0.28 s when the
/// whole prompt was read in one forward).
pub const PREFILL_PIECE: usize = 64;

/// Active vocabulary subset for the writer's output head (`lm_head.weight`): keeps the first
/// 16,384 highest-frequency BPE tokens (all ASCII, punctuation, whitespace, common English & code
/// tokens), both Qwen3 EOS/turn-end tokens (`151_643`, `151_645`), and any project-specific tokens
/// from the workspace up to 24,576 rows. Cuts `lm_head` VRAM from ~593.5 MiB down to ~78-96 MiB
/// (`-500 MiB` VRAM) with ~7x faster decode logits; `NANO_WRITER_FULL_HEAD=on` restores all rows.
fn active_head_rows(w: &nmengn::Model) -> Arc<Vec<usize>> {
    if std::env::var("NANO_WRITER_FULL_HEAD").is_ok_and(|v| v == "on") {
        return Arc::new(Vec::new());
    }
    let vocab = w.pop.dims.vocab;
    if vocab == 0 {
        return Arc::new(Vec::new());
    }
    let mut keep: BTreeSet<usize> = (0..16_384.min(vocab)).collect();
    for eos in [151_643_usize, 151_645_usize] {
        if eos < vocab {
            keep.insert(eos);
        }
    }
    const CAP: usize = 24_576;
    if let Some(o) = crate::tools::installed() {
        let sub = o.sub();
        for f in sub.files.iter().take(200) {
            if keep.len() >= CAP {
                break;
            }
            if let Ok(ids) = w.pop.tokenizer.encode(&f.text) {
                for id in ids {
                    if let Ok(u) = usize::try_from(id)
                        && u < vocab
                    {
                        keep.insert(u);
                        if keep.len() >= CAP {
                            break;
                        }
                    }
                }
            }
        }
    }
    Arc::new(keep.into_iter().collect())
}

/// The token a greedy step takes: the highest logit plus its `bias` (lowest row on ties), its row
/// mapped through `head_rows` when the head is cut to those rows, never a token in `mask` (the
/// stated preferences' banned marks and words, `crate::prefs::WriterMask`). The bias (the wanted
/// terms' push, `crate::prefs::WriterBias`, keyed by token id, never negative) never reaches a
/// masked token: the mask wins. With an empty mask and no bias this is exactly the plain argmax.
fn pick(logits: &[f32], head_rows: &[usize], mask: &BTreeSet<usize>, bias: &BTreeMap<usize, f32>) -> usize {
    let id_of = |row: usize| if head_rows.is_empty() { row } else { head_rows.get(row).copied().unwrap_or(0) };
    let mut best: Option<(usize, f32)> = None;
    if mask.is_empty() {
        let row = nmengn::decode::argmax(logits);
        best = logits.get(row).map(|x| (row, *x));
    } else {
        for (row, x) in logits.iter().enumerate() {
            if mask.contains(&id_of(row)) {
                continue;
            }
            if best.is_none_or(|b| *x > b.1) {
                best = Some((row, *x));
            }
        }
    }
    // The bias only raises the tokens it names, so the biased best is the unbiased best or one of
    // them.
    for (id, b) in bias {
        if mask.contains(id) {
            continue;
        }
        let row = if head_rows.is_empty() { Some(*id) } else { head_rows.iter().position(|r| r == id) };
        let Some((row, x)) = row.and_then(|r| logits.get(r).map(|x| (r, *x + *b))) else { continue };
        if best.is_none_or(|(br, bx)| x > bx || x == bx && row < br) {
            best = Some((row, x));
        }
    }
    id_of(best.map_or_else(|| nmengn::decode::argmax(logits), |b| b.0))
}

#[expect(clippy::too_many_arguments, reason = "the model, device, prompt, budget, head rows, token mask, wanted-term bias and stop rule of one decode")]
/// Greedy decoding as `nmengn::decode::greedy_until`, with the prompt read in pieces of
/// `PREFILL_PIECE` tokens into one key and value cache and a waiting call checked between pieces:
/// a preempted read returns at once with no tokens (`stopped` false). Uses `head_rows` when
/// non-empty to avoid uploading all 151,936 `lm_head` rows (`-500 MiB` VRAM on 1.7B writer) and
/// skips `lm_head` entirely on non-final prefill chunks. Every step takes the stated preferences'
/// `mask` and `bias` (`crate::prefs::WriterMask`, `crate::prefs::WriterBias`); the mask wins.
fn decode_preemptible(
    m: &nmengn::Model,
    device: nmengn::Device,
    prompt: &[i32],
    max_new: usize,
    head_rows: &Arc<Vec<usize>>,
    mask: &BTreeSet<usize>,
    bias: &crate::prefs::WriterBias,
    stop: &dyn Fn(i32) -> bool,
) -> Result<nmengn::decode::Generation, Fault> {
    let opts = nmengn::model::RunOpts {
        cache: Some(Arc::new(nmengn::model::Cache::default())),
        head_rows: Arc::clone(head_rows),
        ..nmengn::model::RunOpts::default()
    };
    let head_target = (m.role_node("head")?, "logits".to_owned());
    let norm_target = m.role_node("final_norm").ok().map(|id| (id, "y".to_owned()));
    let logits = |ids: &[i32], at: usize| -> Result<Vec<f32>, Fault> {
        let out = m.forward_at(device, ids, at, std::slice::from_ref(&head_target), &opts)?;
        Ok(nmengn::model::values(out.get(&head_target).ok_or_else(|| bad("no logits"))?)?.to_vec())
    };
    let t0 = Instant::now();
    let mut l = Vec::new();
    let chunks: Vec<&[i32]> = prompt.chunks(PREFILL_PIECE).collect();
    for (k, piece) in chunks.iter().enumerate() {
        if PREEMPT.load(Ordering::SeqCst) {
            return Ok(nmengn::decode::Generation { tokens: Vec::new(), prefill_ms: t0.elapsed().as_secs_f64() * 1000.0, step_ms: Vec::new(), stopped: false });
        }
        let is_last = k + 1 == chunks.len();
        if !is_last
            && let Some(nt) = &norm_target
        {
            let _ = m.forward_at(device, piece, k * PREFILL_PIECE, std::slice::from_ref(nt), &opts)?;
        } else {
            l = logits(piece, k * PREFILL_PIECE)?;
        }
    }
    let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mut tokens = Vec::new();
    let mut step_ms = Vec::new();
    let mut stopped = false;
    for step in 0..max_new {
        let push = if bias.is_empty() {
            BTreeMap::new()
        } else {
            let last = tokens.last().map_or_else(String::new, |t| m.pop.tokenizer.decode(&[*t]));
            bias.at(&tokens, &last, mask)
        };
        let next = i32::try_from(pick(&l, head_rows, mask, &push)).unwrap_or(0);
        tokens.push(next);
        if stop(next) {
            stopped = true;
            break;
        }
        if step + 1 == max_new {
            break;
        }
        let t = Instant::now();
        l = logits(&[next], prompt.len() + step)?;
        step_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(nmengn::decode::Generation { tokens, prefill_ms, step_ms, stopped })
}

/// Write results until `limit`, the work ends, or a call arrives; the work is refilled from the
/// material in batches, so a long session goes on to file notes and the outline.
fn write_all(w: &Arc<nmengn::Model>, by: &str, embedder: Option<&nmengn::Model>, limit: usize, st: &mut Stats) -> usize {
    let device = if w.pop.gpu.is_some() { nmengn::Device::Gpu } else { nmengn::Device::Cpu };
    let tok = &w.pop.tokenizer;
    let head_rows = active_head_rows(w);
    // A Writer kept on the card between sessions takes up a plug change here (`distil`).
    let _ = crate::distil::seat_writer(&nmtype::nano_root(), w);
    // The stated preferences' token mask (`crate::prefs::WriterMask`), brought up to date before
    // each result, so a preference stated while the Writer works holds from its next result on.
    let vocab = w.pop.dims.vocab;
    let mut mask = crate::prefs::WriterMask::build(vocab, &|t| tok.decode(&[t]));
    // The wanted terms' bias (`crate::prefs::WriterBias`), on the same refresh as the mask.
    let encode = |t: &str| tok.encode(t).unwrap_or_default();
    let mut bias = crate::prefs::WriterBias::build(&encode);
    let mut written = 0;
    let mut tried: BTreeSet<String> = BTreeSet::new();
    'outer: while written < limit {
        let jobs: Vec<Task> = pending(BATCH).into_iter().filter(|t| !tried.contains(&t.store_key())).collect();
        if jobs.is_empty() {
            break;
        }
        // Roadmap Phase 4.5 (2026-11-24): the queue depth, each result and each preemption in the
        // live view.
        crate::live::writer_queue(jobs.len());
        for t in jobs {
            if written >= limit {
                break 'outer;
            }
            if PREEMPT.load(Ordering::SeqCst) {
                st.preempted += 1;
                crate::live::writer_preempted();
                break 'outer;
            }
            let key = t.store_key();
            tried.insert(key.clone());
            let Ok(ids) = tok.encode(&prompt(&t)) else {
                st.faults.push(format!("tokens of {} {}:{}", t.job.tag(), t.path, t.start));
                continue;
            };
            let at = Instant::now();
            let one_line = t.job.one_line();
            // Roadmap Phase 4.6 (2026-11-25): a one-line result whose line ends on a colon
            // announces a list; it was cut at the newline and refused (19 of about 100 results in
            // the soak). Now up to `LIST_LINES` list lines follow, until a blank or a plain line,
            // and the result is joined into one line (`one_line_of`).
            let seen = std::cell::RefCell::new((String::new(), false, 0_usize));
            let stop = |x: i32| {
                if x == 151_643 || x == 151_645 || PREEMPT.load(Ordering::SeqCst) {
                    return true;
                }
                if !one_line {
                    return false;
                }
                let piece = tok.decode(&[x]);
                let Ok(mut st) = seen.try_borrow_mut() else { return piece.contains('\n') };
                let (acc, in_list, lines) = &mut *st;
                if piece.contains('\n') {
                    // The line so far includes this token's text before its line break: the
                    // tokenizer often writes a colon and its line break as one token (":\n"), and
                    // looking only at what came before missed the colon (all 24 colon refusals of
                    // the kept soak).
                    let before = piece.split('\n').next().unwrap_or("");
                    let last = format!("{}{before}", acc.rsplit('\n').next().unwrap_or("")).trim().to_owned();
                    if !*in_list && last.ends_with(':') {
                        *in_list = true;
                    } else if *in_list && *lines == 0 && last.is_empty() {
                        // One blank line between the heading and the list ("include:", blank, "- a").
                    } else if *in_list && !last.is_empty() && *lines < LIST_LINES {
                        *lines += 1;
                    } else {
                        return true;
                    }
                }
                acc.push_str(&piece);
                false
            };
            mask.refresh(vocab, &|t| tok.decode(&[t]));
            bias.refresh(&encode);
            let g = decode_preemptible(w, device, &ids, t.job.tokens(), &head_rows, mask.ids(), &bias, &stop);
            sample(st);
            if PREEMPT.load(Ordering::SeqCst) {
                st.preempted += 1;
                crate::live::writer_preempted();
                break 'outer;
            }
            let g = match g {
                Ok(g) => g,
                Err(e) => {
                    st.faults.push(format!("{} {}:{}: {e}", t.job.tag(), t.path, t.start));
                    break 'outer;
                }
            };
            // The end-of-text and end-of-turn tokens close the result; they are not part of it.
            let kept: Vec<i32> = g.tokens.iter().copied().filter(|x| *x != 151_643 && *x != 151_645).collect();
            let text = tok.decode(&kept);
            let text = if one_line { one_line_of(&text) } else { text.trim().to_owned() };
            // The Writer's words follow the user's stated preferences (enforced transforms only).
            let text = crate::prefs::own_text(&text, Some(&t.path));
            // A one-line result that ends on a colon announces a list the line cut off: refused.
            if text.is_empty() || (one_line && text.ends_with(':')) {
                st.empty += 1;
                st.colon += usize::from(!text.is_empty());
                if let Ok(mut r) = REFUSED.write() {
                    r.insert(key.clone());
                }
                continue;
            }
            if matches!(t.job, Job::Note | Job::LikelyNames)
                && let Some(e) = embedder
                && let Some(ok) = note_lands(e, &t.key, &text)
            {
                st.checked += 1;
                if !ok {
                    st.dropped += 1;
                    if let Ok(mut r) = REFUSED.write() {
                        r.insert(key.clone());
                    }
                    continue;
                }
            }
            if let (Ok(mut n), Ok(mut b)) = (NOTES.write(), BY.write()) {
                n.insert(key.clone(), text.clone());
                b.insert(key.clone(), by.to_owned());
            }
            if t.job == Job::Outline
                && let Ok(mut o) = OUTLINE.write()
            {
                *o = Some(t.key.clone());
            }
            keep(&key, by, &text);
            if t.job == Job::ModuleNote {
                let prefix = format!("module:{}#", t.path);
                if let (Ok(mut n), Ok(mut b)) = (NOTES.write(), BY.write()) {
                    n.retain(|k, _| !k.starts_with(&prefix) || *k == key);
                    b.retain(|k, _| n.contains_key(k));
                }
            }
            // Stage 5 WP-C4 E1/E4: a checked note or likely names joins the meaning lane beside its
            // piece; a module note beside its structure node (`module:<path>`).
            if crate::tools::installed().is_some_and(|o| {
                let sub = o.sub();
                let lane = sub.dense.read().ok().and_then(|d| d.clone());
                lane.is_some_and(|l| match t.job {
                    Job::Note | Job::LikelyNames => l.add_aux_text(&t.key, &text),
                    Job::ModuleNote => l.add_aux_text(&format!("module:{}", t.path), &text),
                    _ => false,
                })
            }) {
                st.aux_added += 1;
            }
            written += 1;
            match t.job {
                Job::Note => st.notes += 1,
                Job::Summary => st.summaries += 1,
                Job::FileNote => st.file_notes += 1,
                Job::Outline => st.outlines += 1,
                Job::LikelyNames => st.likely_names += 1,
                Job::ModuleNote => st.module_notes += 1,
            }
            st.note_ms.push(at.elapsed().as_secs_f64() * 1000.0);
            crate::live::writer_result(t.job.tag(), u64::try_from(at.elapsed().as_millis()).unwrap_or(u64::MAX), &t.path);
        }
    }
    written
}

/// Stage 5 WP-C4: open the configured note store (`NANO_WRITER_CACHE=on`) without the idle loop:
/// read it into RAM and keep writing results to it, as `start` does. Returns the store and the
/// results read; `None` without a store.
pub fn open_store() -> Option<(PathBuf, usize)> {
    let file = store_file()?;
    let already = STORE.read().ok().is_some_and(|s| s.as_ref() == Some(&file));
    let n = if already { 0 } else { load_store(&file) };
    if let Ok(mut s) = STORE.write() {
        *s = Some(file.clone());
    }
    Some((file, n))
}

/// The material version whose held notes were last added to the meaning lane (`seed_aux`).
static SEEDED: Mutex<Option<u64>> = Mutex::new(None);

/// Stage 5 WP-C4 E1/E4: with a note store configured (`NANO_WRITER_CACHE=on`), read it once and add
/// every held note and likely names of a live piece, and every module note, to the meaning lane as
/// auxiliary vectors; once per material version (a restarted `serve` regains its note vectors).
/// Returns the vectors added (0 when already seeded, no store, or no lane).
pub fn seed_aux() -> usize {
    let Some(o) = crate::tools::installed() else { return 0 };
    let sub = o.sub();
    if SEEDED.lock().ok().is_some_and(|s| *s == Some(sub.version)) {
        return 0;
    }
    let Some(lane) = sub.dense.read().ok().and_then(|d| d.clone()) else { return 0 };
    let live: BTreeSet<String> = sub.chunks.iter().map(|c| c.key.clone()).collect();
    seed_into(&lane, &live, sub.version)
}

/// Stage 5 WP-C4: `seed_aux` for `sub` now, on the caller's thread: for a caller with no call in
/// flight (a bench before its first question), so every held note is in before anything is measured.
/// Returns the vectors added (0 without a store or lane, or when already seeded).
pub fn seed_aux_now(sub: &nmsubs::Substrate) -> usize {
    if SEEDED.lock().ok().is_some_and(|s| *s == Some(sub.version)) {
        return 0;
    }
    let Some(lane) = sub.dense.read().ok().and_then(|d| d.clone()) else { return 0 };
    let live: BTreeSet<String> = sub.chunks.iter().map(|c| c.key.clone()).collect();
    seed_into(&lane, &live, sub.version)
}

/// Add the store's held notes and likely names of the `live` pieces to `lane`; marks `version` seeded.
fn seed_into(lane: &nmsubs::dense::DenseLane, live: &BTreeSet<String>, version: u64) -> usize {
    if open_store().is_none() {
        return 0;
    }
    let held: Vec<(String, String)> = NOTES.read().map(|n| n.iter().map(|(k, v)| (k.clone(), v.clone())).collect()).unwrap_or_default();
    let mut added = 0;
    for (k, v) in held {
        let Some((tag, rest)) = k.split_once(':') else { continue };
        let ok = match tag {
            "note" | "names" if live.contains(rest) => lane.add_aux_text(rest, &v),
            // A module note beside its folder (`module:<path>#<hash>`; older keys without the folder
            // cannot be placed and are left out).
            "module" => rest.split_once('#').is_some_and(|(path, _)| lane.add_aux_text(&format!("module:{path}"), &v)),
            _ => false,
        };
        added += usize::from(ok);
    }
    if let Ok(mut s) = SEEDED.lock() {
        *s = Some(version);
    }
    added
}

/// Whether a background seeding of the meaning lane is running (`seed_aux_background`).
static SEEDING: AtomicBool = AtomicBool::new(false);

/// Stage 5 WP-C4: `seed_aux` on its own thread, once at a time. A served call holds the card gate for
/// its whole length and the embedder needs it, so seeding inside a call would wait on itself; on its
/// own thread it takes the gate between calls. Does nothing without a store.
pub fn seed_aux_background(sub: &nmsubs::Substrate) {
    let version = Some(sub.version);
    if SEEDED.lock().ok().is_some_and(|s| *s == version) || store_file().is_none() || SEEDING.swap(true, Ordering::SeqCst) {
        return;
    }
    let Some(lane) = sub.dense.read().ok().and_then(|d| d.clone()) else {
        SEEDING.store(false, Ordering::SeqCst);
        return;
    };
    let live: BTreeSet<String> = sub.chunks.iter().map(|c| c.key.clone()).collect();
    let v = sub.version;
    let ok = std::thread::Builder::new()
        .name("nano-seed-aux".into())
        .spawn(move || {
            let n = seed_into(&lane, &live, v);
            eprintln!("writer: {n} held notes and likely names added to the meaning lane");
            SEEDING.store(false, Ordering::SeqCst);
        })
        .is_ok();
    if !ok {
        SEEDING.store(false, Ordering::SeqCst);
    }
}

/// Drop results of pieces and files no longer in the material (a changed piece has a new content
/// hash): `live` holds the live piece and file content hashes. The outline is kept until a new one.
pub fn shed(live: &BTreeSet<String>) -> usize {
    let removed = {
        let (Ok(mut n), Ok(mut b)) = (NOTES.write(), BY.write()) else { return 0 };
        let before = n.len();
        n.retain(|k, _| k.starts_with("outline:") || k.starts_with("module:") || k.split_once(':').is_some_and(|(_, h)| live.contains(h)));
        b.retain(|k, _| n.contains_key(k));
        before - n.len()
    };
    if removed > 0 {
        rewrite_store();
    }
    removed
}

/// Calls in flight now.
#[must_use]
pub fn calls_in_flight() -> usize {
    IN_CALL.load(Ordering::SeqCst)
}

/// Self-check: the job keys and limits, a served call's pieces queued only while the loop runs,
/// and the prompt forms.
/// # Errors
/// The first of those that fails.
pub fn check() -> Result<(), String> {
    let t = Task { job: Job::Summary, key: "abc".into(), path: "a.rs".into(), start: 1, end: 9, text: "fn a() {}".into() };
    if t.store_key() != "sum:abc" || !prompt(&t).contains("a.rs (lines 1-9)") || !prompt(&t).ends_with("</think>\n\n") {
        return Err("nmapi.writer_idle: task key or prompt".into());
    }
    if Job::Outline.one_line() || !Job::Note.one_line() || Job::Outline.tokens() <= Job::Note.tokens() {
        return Err("nmapi.writer_idle: job limits".into());
    }
    if !running() {
        // Served pieces are recorded whether or not the writer runs (roadmap Phase 0 item 2,
        // 2026-11-23), bounded by `QUEUE`; the check's own entry is taken out again.
        served("q", &["x".into()], &[]);
        let queued = SERVED.lock().map(|q| q.iter().any(|k| k == "x")).unwrap_or(false);
        if let Ok(mut q) = SERVED.lock() {
            q.retain(|k| k != "x");
        }
        if !queued {
            return Err("nmapi.writer_idle: a served piece was not recorded".into());
        }
        stop();
        if !preempted() {
            return Err("nmapi.writer_idle: stop() did not preempt".into());
        }
        SHUTDOWN.store(false, Ordering::SeqCst);
        PREEMPT.store(false, Ordering::SeqCst);
    }
    Ok(())
}

#[cfg(test)]
mod one_line_tests {
    use super::{one_line_of, pick};
    use std::collections::{BTreeMap, BTreeSet};

    /// Roadmap Phase 4.6 (2026-11-25): a one-line result that announces a list keeps the list.
    #[test]
    fn a_list_after_a_colon_is_joined() {
        assert_eq!(one_line_of("Loads the cache.\nmore"), "Loads the cache.");
        assert_eq!(one_line_of("Defines:\n- parse_config\n- Config\n\nlater"), "Defines: parse_config; Config");
        assert_eq!(one_line_of("Steps:\n1. read\n2) check"), "Steps: read; check");
        assert_eq!(one_line_of("Ends on a colon:"), "Ends on a colon:");
        assert_eq!(one_line_of("Includes:\n\n- a\n- b"), "Includes: a; b");
    }

    #[test]
    fn the_mask_keeps_banned_tokens_out_and_changes_nothing_when_empty() {
        let logits = [0.1_f32, 3.0, 2.5, 1.0];
        let none = BTreeSet::new();
        let flat = BTreeMap::new();
        assert_eq!(pick(&logits, &[], &none, &flat), nmengn::decode::argmax(&logits), "an empty mask is the plain argmax");
        let mask: BTreeSet<usize> = [1].into();
        assert_eq!(pick(&logits, &[], &mask, &flat), 2);
        // Rows of a cut head map to token ids; the mask is by token id.
        let rows = [10, 11, 12, 13];
        assert_eq!(pick(&logits, &rows, &none, &flat), 11);
        assert_eq!(pick(&logits, &rows, &[11].into(), &flat), 12);
    }

    #[test]
    fn the_bias_pushes_within_its_bound_and_the_mask_wins() {
        let logits = [0.1_f32, 3.0, 2.5, 1.0];
        let none = BTreeSet::new();
        // 2.5 + 0.75 beats 3.0; 1.0 + 0.75 does not.
        assert_eq!(pick(&logits, &[], &none, &BTreeMap::from([(2, 0.75)])), 2);
        assert_eq!(pick(&logits, &[], &none, &BTreeMap::from([(3, 0.75)])), 1, "a bounded push never takes a token far below the best");
        assert_eq!(pick(&logits, &[], &[2].into(), &BTreeMap::from([(2, 3.0)])), 1, "a masked token is never picked, whatever its bias");
        // By token id through a cut head.
        let rows = [10, 11, 12, 13];
        assert_eq!(pick(&logits, &rows, &none, &BTreeMap::from([(12, 0.75)])), 12);
        assert_eq!(pick(&logits, &rows, &none, &BTreeMap::from([(99, 3.0)])), 11, "an id outside the cut head is skipped");
    }
}
