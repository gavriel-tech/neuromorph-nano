//! The user's hardware limits (Roadmap Phase 4.5 (2026-11-25), item 3): three controls in
//! `settings.json` at the Nano's root, each `auto` (the default) or a number:
//!
//! - `limits.vram_mb`: the Nano's own card memory cap in MiB; 0 means processor only from the
//!   next start (a running process never unbinds its MicroModels for it);
//! - `limits.ram_mb`: the host memory cap in MiB;
//! - `limits.cpu_threads`: the threads the one processor budget (`crate::budget`) lends.
//!
//! The engine reads them live, with no restart: `current` keeps the last values in atomics (a
//! read costs a clock reading and three atomic loads) and looks at the file's modification time
//! at most once a second, parsing it again only when that time changed. A test sets values for
//! its own thread with `scoped`, so tests running side by side never see each other's values.
//! The engine's rules on these values (effective free card memory, the cache ceiling) live in
//! `nmengn::limits`; this crate holds the reading because the processor budget, which every
//! crate shares, lives here too.
use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};

/// The control keys in `settings.json`.
pub const VRAM_KEY: &str = "limits.vram_mb";
pub const RAM_KEY: &str = "limits.ram_mb";
pub const CPU_KEY: &str = "limits.cpu_threads";
/// Every limit key, in display order.
pub const KEYS: [&str; 3] = [VRAM_KEY, RAM_KEY, CPU_KEY];

/// Memory areas (2026-11-27): the host memory budget of one process divided by what it holds.
/// Each area is `auto` (a fixed share of the process budget, `nmengn::limits::area_budget`) or a
/// size in MiB set by the user:
/// - `content`: the substrate's chunk text and index, the `Keep` store and the relevance keys;
/// - `caches`: the kept answer, judge, encoded-chunk and KV caches;
/// - `weights`: the host copies of the MicroModels' weights.
pub const AREAS: [&str; 3] = ["content", "caches", "weights"];
pub const CONTENT_KEY: &str = "limits.memory.content_mb";
pub const CACHES_KEY: &str = "limits.memory.caches_mb";
pub const WEIGHTS_KEY: &str = "limits.memory.weights_mb";
/// The area keys, in the order of `AREAS`.
pub const AREA_KEYS: [&str; 3] = [CONTENT_KEY, CACHES_KEY, WEIGHTS_KEY];
/// Bounds of an area size in MiB.
pub const AREA_RANGE: (u64, u64) = (128, 4_194_304);

/// Bounds of each limit when it is a number: (min, max). `limits.vram_mb` takes 0 (processor
/// only) and then any size; the host memory cap starts at 1 GiB, below which no MicroModel loads.
pub const VRAM_RANGE: (u64, u64) = (0, 262_144);
pub const RAM_RANGE: (u64, u64) = (1_024, 4_194_304);
pub const CPU_RANGE: (u64, u64) = (1, 1_024);

/// How often, at most, the file's modification time is looked at.
pub const RECHECK: Duration = Duration::from_secs(1);

/// The three limits; `None` is `auto` (the engine derives the value from the machine).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Limits {
    pub vram_mb: Option<u64>,
    pub ram_mb: Option<u64>,
    pub cpu_threads: Option<usize>,
}

impl Limits {
    /// All three at `auto`.
    pub const AUTO: Self = Self { vram_mb: None, ram_mb: None, cpu_threads: None };

    /// The card memory cap in bytes when one applies live: `None` for `auto`, and for 0 (that
    /// one applies at the next start, as processor only).
    #[must_use]
    pub fn vram_cap_bytes(&self) -> Option<u64> {
        self.vram_mb.filter(|mb| *mb > 0).map(|mb| mb.saturating_mul(1 << 20))
    }
    /// The host memory cap in bytes, `None` for `auto`.
    #[must_use]
    pub fn ram_cap_bytes(&self) -> Option<u64> {
        self.ram_mb.map(|mb| mb.saturating_mul(1 << 20))
    }
    /// The value of one key as `settings.json` holds it (`auto` or a number).
    #[must_use]
    pub fn setting(&self, key: &str) -> String {
        let n = match key {
            VRAM_KEY => self.vram_mb,
            RAM_KEY => self.ram_mb,
            CPU_KEY => self.cpu_threads.map(|c| c as u64),
            _ => None,
        };
        n.map_or_else(|| "auto".to_owned(), |x| x.to_string())
    }
}

/// The bounds of a limit key, `None` for another key.
#[must_use]
pub fn range(key: &str) -> Option<(u64, u64)> {
    match key {
        VRAM_KEY => Some(VRAM_RANGE),
        RAM_KEY => Some(RAM_RANGE),
        CPU_KEY => Some(CPU_RANGE),
        k if AREA_KEYS.contains(&k) => Some(AREA_RANGE),
        _ => None,
    }
}

/// The full key for a short name (`vram_mb`, `ram_mb`, `cpu_threads`; for an area `content`,
/// `content_mb` or `memory.content_mb`) or a full one.
#[must_use]
pub fn full_key(key: &str) -> Option<&'static str> {
    let k = key.trim();
    KEYS.iter().copied().find(|full| *full == k || full.strip_prefix("limits.") == Some(k)).or_else(|| {
        AREA_KEYS.iter().zip(AREAS).find_map(|(full, area)| {
            let short = full.strip_prefix("limits.memory.");
            (*full == k || full.strip_prefix("limits.") == Some(k) || short == Some(k) || area == k).then_some(*full)
        })
    })
}

/// The index of an area in `AREAS` by its name (`content`) or its key.
#[must_use]
pub fn area_index(area: &str) -> Option<usize> {
    let a = area.trim();
    AREAS.iter().position(|x| *x == a).or_else(|| AREA_KEYS.iter().position(|k| *k == a))
}

/// The area sizes the user set, in MiB, in the order of `AREAS`; `None` is `auto`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Areas(pub [Option<u64>; 3]);

impl Areas {
    /// Every area at `auto`.
    pub const AUTO: Self = Self([None; 3]);
    /// The size set for an area (name or key), `None` for `auto` or an unknown area.
    #[must_use]
    pub fn get(&self, area: &str) -> Option<u64> {
        area_index(area).and_then(|i| self.0[i])
    }
    /// The value of one area key as `settings.json` holds it (`auto` or a number).
    #[must_use]
    pub fn setting(&self, key: &str) -> String {
        self.get(key).map_or_else(|| "auto".to_owned(), |x| x.to_string())
    }
    /// The same areas with one set to `mb` (`None`: auto).
    #[must_use]
    pub fn with(mut self, area: &str, mb: Option<u64>) -> Self {
        if let Some(i) = area_index(area) {
            self.0[i] = mb;
        }
        self
    }
    /// The sum of the areas set to a size, in MiB.
    #[must_use]
    pub fn set_sum_mb(&self) -> u64 {
        self.0.iter().flatten().fold(0, |a, b| a.saturating_add(*b))
    }
}

/// The area sizes in the text of a `settings.json` (anything missing or not valid is `auto`).
#[must_use]
pub fn parse_areas(bytes: &[u8]) -> Areas {
    let Ok(v) = nmjson::parse(bytes) else { return Areas::AUTO };
    let mut a = Areas::AUTO;
    for (i, k) in AREA_KEYS.iter().enumerate() {
        a.0[i] = v.get(k).ok().and_then(|x| number(k, x));
    }
    a
}

/// The area sizes saved in `<home>/settings.json` now (read every time; `areas` caches).
#[must_use]
pub fn read_areas_at(home: &Path) -> Areas {
    std::fs::read(home.join("settings.json")).map_or(Areas::AUTO, |b| parse_areas(&b))
}

/// Checks and normalizes a value for a limit key: `auto` (any case) or a whole number within the
/// key's bounds. `None` when `key` is not a limit key.
#[must_use]
pub fn normalize(key: &str, raw: &str) -> Option<Result<String, String>> {
    let (min, max) = range(key)?;
    let t = raw.trim();
    if t.eq_ignore_ascii_case("auto") {
        return Some(Ok("auto".to_owned()));
    }
    Some(match t.parse::<u64>() {
        Ok(n) if (min..=max).contains(&n) => Ok(n.to_string()),
        Ok(_) => Err(format!("{key}: {raw} must be auto or between {min} and {max}")),
        Err(_) => Err(format!("{key}: {raw} must be auto or a whole number between {min} and {max}")),
    })
}

/// The number a saved value holds, `None` for `auto` or anything not valid.
fn number(key: &str, v: &nmjson::Value) -> Option<u64> {
    let text = match v {
        nmjson::Value::String(s) => s.clone(),
        nmjson::Value::Number(n) => n.clone(),
        _ => return None,
    };
    normalize(key, &text)?.ok()?.parse().ok()
}

/// The limits in the text of a `settings.json` (anything missing or not valid is `auto`).
#[must_use]
pub fn parse(bytes: &[u8]) -> Limits {
    let Ok(v) = nmjson::parse(bytes) else { return Limits::AUTO };
    let get = |k: &str| v.get(k).ok().and_then(|x| number(k, x));
    Limits { vram_mb: get(VRAM_KEY), ram_mb: get(RAM_KEY), cpu_threads: get(CPU_KEY).and_then(|c| usize::try_from(c).ok()) }
}

/// The limits saved in `<home>/settings.json` now (read every time; `current` caches).
#[must_use]
pub fn read_at(home: &Path) -> Limits {
    std::fs::read(home.join("settings.json")).map_or(Limits::AUTO, |b| parse(&b))
}

/// A cached reading of one settings file: the modification time is looked at no more often than
/// `RECHECK`, and the file is parsed again only when that time changed.
#[derive(Debug)]
pub struct Reader {
    path: PathBuf,
    checked: Option<Instant>,
    modified: Option<SystemTime>,
    limits: Limits,
    areas: Areas,
    /// Times the file was parsed (for tests and reports).
    pub parses: u64,
}

impl Reader {
    /// A reader of `path` (a `settings.json`), not read yet.
    #[must_use]
    pub fn new(path: PathBuf) -> Self {
        Self { path, checked: None, modified: None, limits: Limits::AUTO, areas: Areas::AUTO, parses: 0 }
    }
    /// The limits at `now`: the cached values within `RECHECK` of the last look, else the file
    /// again when its modification time changed (a file that went away reads as `auto`).
    pub fn get(&mut self, now: Instant) -> Limits {
        if self.checked.is_some_and(|t| now.saturating_duration_since(t) < RECHECK) {
            return self.limits;
        }
        self.checked = Some(now);
        let modified = std::fs::metadata(&self.path).and_then(|m| m.modified()).ok();
        if modified != self.modified || self.parses == 0 {
            self.modified = modified;
            let bytes = if modified.is_some() { std::fs::read(&self.path).ok() } else { None };
            self.limits = bytes.as_deref().map_or(Limits::AUTO, parse);
            self.areas = bytes.as_deref().map_or(Areas::AUTO, parse_areas);
            self.parses += 1;
        }
        self.limits
    }
    /// The area sizes of the last reading (`get` refreshes them).
    #[must_use]
    pub fn areas(&self) -> Areas {
        self.areas
    }
}

/// `u64::MAX` stands for `auto` in the atomics below.
const AUTO: u64 = u64::MAX;
static VRAM: AtomicU64 = AtomicU64::new(AUTO);
static RAM: AtomicU64 = AtomicU64::new(AUTO);
static CPU: AtomicU64 = AtomicU64::new(AUTO);
static AREA_MB: [AtomicU64; 3] = [AtomicU64::new(AUTO), AtomicU64::new(AUTO), AtomicU64::new(AUTO)];
/// Milliseconds since `EPOCH` at which the file is looked at next (0: never read yet).
static NEXT_MS: AtomicU64 = AtomicU64::new(0);
static READING: AtomicBool = AtomicBool::new(false);
static EPOCH: OnceLock<Instant> = OnceLock::new();
static READER: Mutex<Option<Reader>> = Mutex::new(None);

thread_local! {
    static SCOPED: Cell<Option<Limits>> = const { Cell::new(None) };
    static SCOPED_AREAS: Cell<Option<Areas>> = const { Cell::new(None) };
}

/// Area sizes set for this thread until the guard drops (`scoped_areas`).
#[derive(Debug)]
pub struct ScopedAreas(Option<Areas>);
impl Drop for ScopedAreas {
    fn drop(&mut self) {
        SCOPED_AREAS.with(|s| s.set(self.0));
    }
}

/// Sets `a` as this thread's area sizes until the returned guard drops (for tests).
#[must_use]
pub fn scoped_areas(a: Areas) -> ScopedAreas {
    ScopedAreas(SCOPED_AREAS.with(|s| s.replace(Some(a))))
}

/// The area sizes now, read live from `<nano root>/settings.json` with the limits (`current`).
#[must_use]
pub fn areas() -> Areas {
    if let Some(a) = SCOPED_AREAS.with(Cell::get) {
        return a;
    }
    let _ = current();
    let mut a = Areas::AUTO;
    for (i, x) in AREA_MB.iter().enumerate() {
        a.0[i] = load(x);
    }
    a
}

/// What each area holds now, in bytes, as the parts holding it report (`set_area_used`), and
/// whether a pass stopped because the area is full (`set_area_full`). Each holder reports its
/// own bytes under its name (`set_area_used`); the area's use is the sum over its holders.
type Holders = std::collections::BTreeMap<(usize, &'static str), u64>;
static AREA_HOLDERS: Mutex<Holders> = Mutex::new(std::collections::BTreeMap::new());
static AREA_FULL: [AtomicBool; 3] = [AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false)];

/// Records what one holder of an area keeps now, in bytes (an unknown area is ignored).
pub fn set_area_used(area: &str, holder: &'static str, bytes: u64) {
    if let Some(i) = area_index(area) {
        let mut g = AREA_HOLDERS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        g.insert((i, holder), bytes);
    }
}
/// What an area holds now over all its holders, in bytes (0 for an unknown area).
#[must_use]
pub fn area_used(area: &str) -> u64 {
    area_holders(area).iter().fold(0, |a, (_, b)| a.saturating_add(*b))
}
/// What each holder of an area keeps now, in bytes.
#[must_use]
pub fn area_holders(area: &str) -> Vec<(&'static str, u64)> {
    let Some(i) = area_index(area) else { return Vec::new() };
    let g = AREA_HOLDERS.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    g.iter().filter(|((a, _), _)| *a == i).map(|((_, h), b)| (*h, *b)).collect()
}
/// The last budget of each area in bytes as `nmengn::limits::area_budget` computed it (`AUTO`:
/// not computed yet), so a crate below the engine (the substrate's dense lane) can tell whether
/// an area is full.
static AREA_BUDGET: [AtomicU64; 3] = [AtomicU64::new(AUTO), AtomicU64::new(AUTO), AtomicU64::new(AUTO)];

/// Records an area's budget in bytes (`nmengn::limits::area_budget` does at each reading).
pub fn note_area_budget(area: &str, bytes: Option<u64>) {
    if let Some(i) = area_index(area) {
        AREA_BUDGET[i].store(bytes.unwrap_or(AUTO), Ordering::Relaxed);
    }
}
/// Whether an area holds its last noted budget or more (`false` while no budget is noted).
#[must_use]
pub fn area_over(area: &str) -> bool {
    area_index(area).and_then(|i| load(&AREA_BUDGET[i])).is_some_and(|b| area_used(area) >= b)
}
/// Records whether a pass stopped because the area is full.
pub fn set_area_full(area: &str, full: bool) {
    if let Some(i) = area_index(area) {
        AREA_FULL[i].store(full, Ordering::Relaxed);
    }
}
/// Whether a pass stopped because the area is full.
#[must_use]
pub fn area_full(area: &str) -> bool {
    area_index(area).is_some_and(|i| AREA_FULL[i].load(Ordering::Relaxed))
}

/// Values set for this thread until the guard drops (`scoped`).
#[derive(Debug)]
pub struct Scoped(Option<Limits>);
impl Drop for Scoped {
    fn drop(&mut self) {
        SCOPED.with(|s| s.set(self.0));
    }
}

/// Sets `l` as this thread's limits until the returned guard drops (for tests: other threads,
/// and other tests running side by side, keep reading the file).
#[must_use]
pub fn scoped(l: Limits) -> Scoped {
    Scoped(SCOPED.with(|s| s.replace(Some(l))))
}

fn load(a: &AtomicU64) -> Option<u64> {
    match a.load(Ordering::Relaxed) {
        AUTO => None,
        x => Some(x),
    }
}

/// The limits now, read live from `<nano root>/settings.json` (see the module note).
#[must_use]
pub fn current() -> Limits {
    if let Some(l) = SCOPED.with(Cell::get) {
        return l;
    }
    let epoch = *EPOCH.get_or_init(Instant::now);
    let now = Instant::now();
    let ms = u64::try_from(now.saturating_duration_since(epoch).as_millis()).unwrap_or(u64::MAX - 1).saturating_add(1);
    // One thread looks at the file; the others read the values it last stored.
    if ms >= NEXT_MS.load(Ordering::Relaxed) && !READING.swap(true, Ordering::Acquire) {
        let mut g = READER.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let r = g.get_or_insert_with(|| Reader::new(crate::nano_root().join("settings.json")));
        let l = r.get(now);
        let a = r.areas();
        drop(g);
        for (i, x) in AREA_MB.iter().enumerate() {
            x.store(a.0[i].unwrap_or(AUTO), Ordering::Relaxed);
        }
        VRAM.store(l.vram_mb.unwrap_or(AUTO), Ordering::Relaxed);
        RAM.store(l.ram_mb.unwrap_or(AUTO), Ordering::Relaxed);
        CPU.store(l.cpu_threads.map_or(AUTO, |c| c as u64), Ordering::Relaxed);
        NEXT_MS.store(ms.saturating_add(u64::try_from(RECHECK.as_millis()).unwrap_or(1000)), Ordering::Relaxed);
        READING.store(false, Ordering::Release);
        return l;
    }
    // The first reading may still be in progress on another thread: wait for it once.
    while NEXT_MS.load(Ordering::Relaxed) == 0 && READING.load(Ordering::Acquire) {
        std::hint::spin_loop();
    }
    Limits { vram_mb: load(&VRAM), ram_mb: load(&RAM), cpu_threads: load(&CPU).and_then(|c| usize::try_from(c).ok()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let d = crate::nano_root().join("scratch").join("unit-tests").join(format!("nmtype-limits-{name}-{}-{}", std::process::id(), crate::now_ms()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn limits_parse_auto_numbers_and_junk() {
        assert_eq!(parse(b"{}"), Limits::AUTO);
        assert_eq!(parse(b"not json"), Limits::AUTO);
        let l = parse(br#"{"limits.vram_mb":"2048","limits.ram_mb":"auto","limits.cpu_threads":"6","other":"x"}"#);
        assert_eq!(l, Limits { vram_mb: Some(2048), ram_mb: None, cpu_threads: Some(6) });
        // Out of bounds or not a number reads as auto; a bare JSON number is taken too.
        let l = parse(br#"{"limits.vram_mb":"lots","limits.ram_mb":"12","limits.cpu_threads":4}"#);
        assert_eq!(l, Limits { vram_mb: None, ram_mb: None, cpu_threads: Some(4) });
        let zero = parse(br#"{"limits.vram_mb":"0"}"#);
        assert_eq!(zero.vram_mb, Some(0));
        assert_eq!(zero.vram_cap_bytes(), None, "0 is processor only at the next start, not a live cap");
        assert_eq!(parse(br#"{"limits.vram_mb":"3"}"#).vram_cap_bytes(), Some(3 << 20));
        assert_eq!(zero.setting(VRAM_KEY), "0");
        assert_eq!(zero.setting(RAM_KEY), "auto");
    }

    #[test]
    fn limits_auto_validation() {
        assert_eq!(normalize(VRAM_KEY, "AUTO"), Some(Ok("auto".to_owned())));
        assert_eq!(normalize(RAM_KEY, " auto "), Some(Ok("auto".to_owned())));
        assert_eq!(normalize(CPU_KEY, "auto"), Some(Ok("auto".to_owned())));
        assert_eq!(normalize(VRAM_KEY, "0"), Some(Ok("0".to_owned())));
        assert_eq!(normalize(VRAM_KEY, " 3072 "), Some(Ok("3072".to_owned())));
        assert!(matches!(normalize(RAM_KEY, "512"), Some(Err(_))), "below 1 GiB");
        assert!(matches!(normalize(CPU_KEY, "0"), Some(Err(_))));
        assert!(matches!(normalize(CPU_KEY, "two"), Some(Err(_))));
        assert!(matches!(normalize(VRAM_KEY, "-1"), Some(Err(_))));
        assert_eq!(normalize("budget.gpu_mb_max", "auto"), None, "not a limit key");
        assert_eq!(full_key("vram_mb"), Some(VRAM_KEY));
        assert_eq!(full_key("limits.cpu_threads"), Some(CPU_KEY));
        assert_eq!(full_key("gpu_mb_max"), None);
    }

    #[test]
    fn limits_reader_caches_and_rereads_on_change() {
        let d = dir("reader");
        let p = d.join("settings.json");
        let mut r = Reader::new(p.clone());
        let t0 = Instant::now();
        // No file: auto, read once.
        assert_eq!(r.get(t0), Limits::AUTO);
        assert_eq!(r.parses, 1);
        std::fs::write(&p, br#"{"limits.vram_mb":"1024"}"#).unwrap();
        // Within a second the cached value stands, without looking at the file.
        assert_eq!(r.get(t0 + Duration::from_millis(500)), Limits::AUTO);
        assert_eq!(r.parses, 1);
        // After a second the new file is read.
        let t1 = t0 + Duration::from_millis(1_100);
        assert_eq!(r.get(t1).vram_mb, Some(1024));
        assert_eq!(r.parses, 2);
        // Unchanged modification time: no parse again.
        assert_eq!(r.get(t1 + Duration::from_millis(1_100)).vram_mb, Some(1024));
        assert_eq!(r.parses, 2);
        // A changed file (its modification time moved on) is read again.
        let later = SystemTime::now() + Duration::from_secs(5);
        std::fs::write(&p, br#"{"limits.vram_mb":"auto","limits.cpu_threads":"3"}"#).unwrap();
        std::fs::File::options().write(true).open(&p).unwrap().set_modified(later).unwrap();
        let l = r.get(t1 + Duration::from_millis(2_300));
        assert_eq!(l, Limits { vram_mb: None, ram_mb: None, cpu_threads: Some(3) });
        assert_eq!(r.parses, 3);
        // The file removed reads as auto.
        std::fs::remove_file(&p).unwrap();
        assert_eq!(r.get(t1 + Duration::from_millis(3_500)), Limits::AUTO);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn limits_scoped_override_is_per_thread() {
        let set = Limits { vram_mb: Some(512), ram_mb: Some(2048), cpu_threads: Some(2) };
        {
            let _g = scoped(set);
            assert_eq!(current(), set);
            let other = std::thread::spawn(current).join().unwrap();
            assert_ne!(other, set, "another thread reads the file, not this thread's values");
        }
        assert_ne!(current(), set);
    }

    #[test]
    fn limits_area_keys_parse_and_validate() {
        assert_eq!(full_key("content"), Some(CONTENT_KEY));
        assert_eq!(full_key("caches_mb"), Some(CACHES_KEY));
        assert_eq!(full_key("memory.weights_mb"), Some(WEIGHTS_KEY));
        assert_eq!(full_key(WEIGHTS_KEY), Some(WEIGHTS_KEY));
        assert_eq!(normalize(CONTENT_KEY, "1024"), Some(Ok("1024".to_owned())));
        assert!(matches!(normalize(CONTENT_KEY, "64"), Some(Err(_))), "below 128 MiB");
        let a = parse_areas(br#"{"limits.memory.content_mb":"1024","limits.memory.caches_mb":"auto","limits.memory.weights_mb":"x"}"#);
        assert_eq!(a, Areas([Some(1024), None, None]));
        assert_eq!(a.get("content"), Some(1024));
        assert_eq!(a.setting(CACHES_KEY), "auto");
        assert_eq!(a.with("weights", Some(2048)).set_sum_mb(), 3072);
        assert_eq!(area_index("nothing"), None);
        let _g = scoped_areas(a);
        assert_eq!(areas(), a);
    }

    #[test]
    fn limits_read_at_home() {
        let d = dir("home");
        assert_eq!(read_at(&d), Limits::AUTO);
        std::fs::write(d.join("settings.json"), br#"{"limits.ram_mb":"8192"}"#).unwrap();
        assert_eq!(read_at(&d).ram_mb, Some(8192));
        let _ = std::fs::remove_dir_all(&d);
    }
}
