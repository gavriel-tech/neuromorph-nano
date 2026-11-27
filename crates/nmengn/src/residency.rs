//! Device residency manager (step P2, 2026-11-10; `docs/2026-11-10-neumor-plan-whole-architecture.md`;
//! Gavriel Technologies, architecture by Luis Jake Gabriel III). One shared budget for what the
//! graphics card keeps between jobs: the device copies of windows and the weights of the
//! populations (base, embedder, reranker), plus the idle buffers of the `nmffi` pool. Nothing is
//! pinned and nothing is fixed: a part keeps device memory only while no other part needs it
//! (specification section 8 item 1).
//!
//! Rule, derived at run time from the device and from measurement:
//! - Reserve: the largest working set (live working buffers above the run's start) measured
//!   over the GPU runs of this process (`RunMark`); before the first run is measured, an eighth
//!   of the device's memory (stated fallback). The pool may keep that many idle bytes.
//! - A window copy is kept after its reads only while the device's free memory (read with
//!   `cuMemGetInfo`, so after the weights in use and everything else on the card, other
//!   processes included) stays at or above the reserve; older kept windows are released first,
//!   least recently used first, and a window that still does not fit is used and released.
//! - Before weights are uploaded, and whenever an allocation fails for lack of memory, room is
//!   made in this order until free memory covers the request plus the reserve: idle pool
//!   buffers, then kept windows (least recently used first), then the weights of populations
//!   that no run is using, lowest measured reload time per byte first (then least recently
//!   used). Reload times per byte were measured in step P1 (reranker 2.09 s, embedder 2.0 s,
//!   base 0.74 s for its layers); each population's own upload times are recorded here.
use crate::gpu::{DevT, Ops};
use crate::model::Population;
use nmffi::Gpu;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

/// Device keys and values per layer of one window.
pub type DevLayers = Vec<(DevT, DevT)>;

thread_local! {
    /// Set while this thread runs background device work (`background`).
    static BACKGROUND: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Marks this thread's device work as background (the dense lane's index) until the guard
/// drops: room made for it releases idle buffers and kept windows, never a model's weights, so
/// background work never makes a question upload weights again.
#[derive(Debug)]
pub struct Background(bool);
impl Drop for Background {
    fn drop(&mut self) {
        BACKGROUND.with(|b| b.set(self.0));
    }
}
/// Mark this thread's device work as background until the returned guard drops.
#[must_use]
pub fn background() -> Background {
    Background(BACKGROUND.with(|b| b.replace(true)))
}
/// Whether this thread runs background device work now.
#[must_use]
pub fn in_background() -> bool {
    BACKGROUND.with(std::cell::Cell::get)
}

static TICK: AtomicU64 = AtomicU64::new(1);
/// A use counter for least-recently-used order.
pub fn tick() -> u64 {
    TICK.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug)]
struct WinEntry {
    layers: DevLayers,
    bytes: usize,
    last: u64,
    last_ms: u64,
}

/// One resident parameter tensor (`Residency::parameters`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resident {
    pub model: String,
    pub source: String,
    pub tensor: String,
    /// RAM bytes of the tensor, and device bytes its bound views record.
    pub ram: u64,
    pub device: u64,
}

/// Counters and figures for reports.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// The reserve in bytes and whether it was measured (else the fallback).
    pub reserve: usize,
    pub measured: bool,
    /// GPU runs measured.
    pub runs: u64,
    pub windows_held: usize,
    pub window_bytes: usize,
    pub window_hits: u64,
    pub window_misses: u64,
    pub windows_kept: u64,
    pub windows_refused: u64,
    pub windows_released: u64,
    pub weights_released: u64,
    /// Times room for background work stopped short of releasing weights.
    pub weights_kept_for_background: u64,
    /// Bytes the room-making freed in all.
    pub bytes_released: u64,
    /// The last release, in words.
    pub last_release: String,
}

#[derive(Debug, Default)]
struct State {
    gpu: Option<Gpu>,
    total: usize,
    windows: BTreeMap<String, WinEntry>,
    pops: Vec<Weak<Population>>,
    work_peak: usize,
    stats: Stats,
    /// Where this machine's measures are kept (`capacity`), and the key of the card.
    home: Option<std::path::PathBuf>,
    machine: String,
}

/// The process-wide residency manager.
#[derive(Debug, Default)]
pub struct Residency {
    st: Mutex<State>,
}

static RES: OnceLock<Residency> = OnceLock::new();

/// The residency manager of this process.
pub fn get() -> &'static Residency {
    RES.get_or_init(Residency::default)
}

impl Residency {
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.st.lock().unwrap_or_else(PoisonError::into_inner)
    }
    /// Register a population that holds device copies of its weights (and its device).
    pub fn register(&self, pop: &Arc<Population>) {
        {
            let mut st = self.lock();
            st.pops.retain(|w| w.strong_count() > 0);
            st.pops.push(Arc::downgrade(pop));
        }
        let Some(o) = pop.gpu.as_ref() else { return };
        let first = {
            let mut st = self.lock();
            if st.gpu.is_none() {
                st.total = o.gpu.info().total_bytes;
                st.gpu = Some(o.gpu.clone());
                // Roadmap Phase 4.6 (2026-11-24): this machine's measured working peak, kept from
                // earlier runs, seeds the reserve so a restart does not start from an eighth.
                let key = crate::capacity::machine_key(Some((o.gpu.info().name.as_str(), st.total >> 20)));
                if let Some(peak) = crate::capacity::recall(&pop.home, &key, "work_peak_bytes").and_then(|x| usize::try_from(x).ok()) {
                    st.work_peak = st.work_peak.max(peak);
                }
                crate::autotune::seed_from_machine(&pop.home, &key);
                st.home = Some(pop.home.clone());
                st.machine = key;
                true
            } else {
                false
            }
        };
        if first {
            o.gpu.set_reclaimer(Some(Arc::new(|need| get().reclaim(need))));
        }
    }
    /// The reserve: (bytes, measured).
    #[must_use]
    pub fn reserve(&self) -> (usize, bool) {
        let st = self.lock();
        reserve_of(st.work_peak, st.total)
    }
    /// The effective free card memory now (`memory_now`), what the room-making and the window
    /// keeping measure against.
    fn free_now(&self) -> Option<usize> {
        self.memory_now().map(|m| m.0)
    }
    /// The card memory pool's counters (working peak, long-lived bytes); `None` without a card.
    #[must_use]
    pub fn pool(&self) -> Option<nmffi::PoolStats> {
        self.lock().gpu.clone().map(|g| g.pool_stats())
    }
    /// Free and total card memory now (bytes), as the driver reports it; `None` without a card.
    #[must_use]
    pub fn memory_driver(&self) -> Option<(usize, usize)> {
        let g = self.lock().gpu.clone()?;
        g.memory().ok()
    }
    /// Roadmap Phase 4.5 (2026-11-25): the Nano's own card bytes now (`limits::own_bytes`: the
    /// pool's working, idle and long-lived buffers, with the registered populations' device
    /// copies as a floor of the long-lived ones); `None` without a card. A population whose
    /// device table is locked right now (it is uploading, and may be the one asking for room) is
    /// read through the pool's count alone, so this never waits on that lock.
    #[must_use]
    pub fn own_bytes(&self) -> Option<usize> {
        let (gpu, pops) = self.gpu_and_pops()?;
        Some(own_on(&gpu, &pops))
    }
    fn gpu_and_pops(&self) -> Option<(Gpu, Vec<Arc<Population>>)> {
        let st = self.lock();
        Some((st.gpu.clone()?, st.pops.iter().filter_map(Weak::upgrade).collect()))
    }
    /// Free and total card memory now (bytes). Roadmap Phase 4.5 (2026-11-25): the free figure is
    /// the effective one, `min(driver free, limits.vram_mb - the Nano's own bytes)` under a card
    /// limit (`limits::effective_free`), so every size derived from it tiles within the limit;
    /// with `auto` it is the driver's figure. The total is the card's. `None` without a card.
    #[must_use]
    pub fn memory_now(&self) -> Option<(usize, usize)> {
        let gpu = self.lock().gpu.clone()?;
        let (free, total) = gpu.memory().ok()?;
        Some((self.effective_free(&gpu, free), total))
    }
    /// The effective free memory for a driver reading `driver_free` taken on `gpu` (the device
    /// every population shares), for `Population::gpu_free`.
    #[must_use]
    pub fn effective_free(&self, gpu: &Gpu, driver_free: usize) -> usize {
        // The Nano's own bytes are read only under a limit, so `auto` costs nothing more than
        // the driver reading.
        let Some(cap) = crate::limits::current().vram_cap_bytes() else { return driver_free };
        let pops: Vec<Arc<Population>> = self.lock().pops.iter().filter_map(Weak::upgrade).collect();
        crate::limits::effective_free(driver_free, Some(cap), own_on(gpu, &pops))
    }
    /// Record a GPU run's working peak (bytes above its start); the pool's idle limit follows
    /// the reserve.
    pub fn note_run(&self, work: usize) {
        let (gpu, limit, keep) = {
            let mut st = self.lock();
            st.stats.runs += 1;
            let grew = work > st.work_peak;
            st.work_peak = st.work_peak.max(work);
            let keep = if grew { st.home.clone().map(|h| (h, st.machine.clone(), st.work_peak)) } else { None };
            (st.gpu.clone(), reserve_of(st.work_peak, st.total).0, keep)
        };
        if let Some(g) = gpu {
            g.set_idle_limit(limit);
        }
        // A new peak is kept for this machine (rare: only when the peak grows).
        if let Some((home, key, peak)) = keep {
            crate::capacity::remember(&home, &key, "work_peak_bytes", peak as u64);
        }
    }
    /// The card in use as the driver reports it; `None` without a card.
    #[must_use]
    pub fn card(&self) -> Option<nmffi::GpuInfo> {
        self.lock().gpu.as_ref().map(|g| g.info().clone())
    }
    /// A kept window copy under `key`, marked as just used.
    #[must_use]
    pub fn window_get(&self, key: &str) -> Option<DevLayers> {
        let mut st = self.lock();
        let t = tick();
        if let Some(e) = st.windows.get_mut(key) {
            e.last = t;
            e.last_ms = nmtype::now_ms();
            let l = e.layers.clone();
            st.stats.window_hits += 1;
            return Some(l);
        }
        st.stats.window_misses += 1;
        None
    }
    /// Offer a window copy (already on the device) to be kept after its reads. Kept while the
    /// device's free memory stays at or above the reserve, releasing older windows first
    /// (least recently used); returns whether it was kept.
    pub fn window_offer(&self, key: &str, layers: DevLayers, bytes: usize) -> bool {
        loop {
            let (free, (reserve, _)) = (self.free_now(), self.reserve());
            let Some(free) = free else { return false };
            if free >= reserve {
                let mut st = self.lock();
                st.windows.insert(key.to_owned(), WinEntry { layers, bytes, last: tick(), last_ms: nmtype::now_ms() });
                st.stats.windows_kept += 1;
                return true;
            }
            if self.release_window(Some(key)).is_none() {
                let gpu = self.lock().gpu.clone();
                let trimmed = gpu.map_or(0, |g| g.trim());
                if trimmed == 0 || self.free_now().unwrap_or(0) < reserve {
                    self.lock().stats.windows_refused += 1;
                    return false;
                }
            }
        }
    }
    /// Release the least recently used kept window (not `except`); its bytes, or `None`.
    fn release_window(&self, except: Option<&str>) -> Option<usize> {
        let gone = {
            let mut st = self.lock();
            let key = st.windows.iter().filter(|(k, _)| Some(k.as_str()) != except).min_by_key(|(_, e)| e.last).map(|(k, _)| k.clone())?;
            let e = st.windows.remove(&key)?;
            st.stats.windows_released += 1;
            st.stats.bytes_released += e.bytes as u64;
            st.stats.last_release = format!("window copy of {} MB (least recently used)", e.bytes >> 20);
            e
        };
        let b = gone.bytes;
        drop(gone);
        Some(b)
    }
    /// Flexing idle KV release: release kept window copies unused for at least `max_age_ms` and
    /// trim idle scratch buffers when idle, leaving seated model weights untouched. Returns total
    /// device bytes freed.
    pub fn release_idle_windows(&self, max_age_ms: u64) -> usize {
        let now = nmtype::now_ms();
        let (gone, gpu, had_windows) = {
            let mut st = self.lock();
            let stale: Vec<String> = st
                .windows
                .iter()
                .filter(|(_, e)| now.saturating_sub(e.last_ms) >= max_age_ms)
                .map(|(k, _)| k.clone())
                .collect();
            let mut removed = Vec::with_capacity(stale.len());
            for k in stale {
                if let Some(e) = st.windows.remove(&k) {
                    st.stats.windows_released += 1;
                    st.stats.bytes_released += e.bytes as u64;
                    st.stats.last_release = format!("idle window copy of {} MB", e.bytes >> 20);
                    removed.push(e);
                }
            }
            (removed, st.gpu.clone(), !st.windows.is_empty())
        };
        let mut freed: usize = gone.iter().map(|e| e.bytes).sum();
        drop(gone);
        if (freed > 0 || !had_windows) && let Some(g) = gpu {
            freed = freed.saturating_add(g.trim());
        }
        freed
    }
    /// Make room for `need` bytes about to be allocated for `requester` (a population id, or
    /// empty): until free memory covers `need` plus the reserve, release idle pool buffers,
    /// then kept windows (least recently used), then the weights of populations no run is
    /// using (lowest measured reload time per byte, then least recently used). Returns the
    /// bytes released. Roadmap Phase 4.5 (2026-11-25): free memory here is the effective figure
    /// (`memory_now`), so under a card limit the same order evicts until the Nano's own bytes
    /// plus `need` and the reserve fit under the limit.
    pub fn make_room(&self, need: usize, requester: &str) -> usize {
        let (reserve, _) = self.reserve();
        let enough = |r: &Self| r.free_now().is_none_or(|f| f >= need.saturating_add(reserve));
        if enough(self) {
            return 0;
        }
        let mut released = 0;
        let gpu = self.lock().gpu.clone();
        if let Some(g) = &gpu {
            released += g.trim();
        }
        while !enough(self) {
            match self.release_window(None) {
                Some(b) => released += b,
                None => break,
            }
        }
        if in_background() {
            if !enough(self) {
                self.lock().stats.weights_kept_for_background += 1;
            }
            return released;
        }
        while !enough(self) {
            match self.release_weights(requester) {
                Some(b) => released += b,
                None => break,
            }
        }
        released
    }
    /// The out-of-memory hook of the pool (`nmffi::Gpu::set_reclaimer`).
    pub fn reclaim(&self, need: usize) -> usize {
        self.make_room(need, "")
    }
    /// Release the weights of one population that no run is using, other than `requester`:
    /// the lowest measured reload time per byte first, then the least recently used.
    fn release_weights(&self, requester: &str) -> Option<usize> {
        let pops: Vec<Arc<Population>> = self.lock().pops.iter().filter_map(Weak::upgrade).collect();
        let mut c: Vec<(f64, u64, Arc<Population>)> = pops
            .into_iter()
            .filter(|p| p.card.id != requester && p.active() == 0 && p.device_bytes() > 0)
            .map(|p| (p.reload_secs_per_byte(), p.last_use(), p))
            .collect();
        release_order(&mut c);
        for (_, _, p) in c {
            let b = p.device_bytes();
            if p.try_release_device() {
                let mut st = self.lock();
                st.stats.weights_released += 1;
                st.stats.bytes_released += b as u64;
                st.stats.last_release = format!("weights of {} ({} MB, idle)", p.card.id, b >> 20);
                return Some(b);
            }
        }
        None
    }
    /// Every parameter tensor resident in this process now, by MicroModel, source and name: for
    /// each live population, each tensor one of its views holds in RAM (bytes counted once per
    /// tensor) and the device bytes its views record. The proof that a source was released is
    /// that none of its tensors is listed.
    #[must_use]
    pub fn parameters(&self) -> Vec<Resident> {
        let pops: Vec<Arc<Population>> = self.lock().pops.iter().filter_map(Weak::upgrade).collect();
        let mut out: BTreeMap<(String, String, String), Resident> = BTreeMap::new();
        for p in pops {
            for (id, storage, ram_now) in p.prov.residency() {
                let Some(c) = p.clusters.iter().find(|c| c.id == id) else { continue };
                let e = out.entry((p.card.id.clone(), p.source.clone(), c.tensor.clone())).or_insert_with(|| Resident {
                    model: p.card.id.clone(),
                    source: p.source.clone(),
                    tensor: c.tensor.clone(),
                    ram: 0,
                    device: 0,
                });
                // Weight placement (2026-11-27): RAM only while the host copy is in RAM (a tensor
                // whose copy was dropped after it went to the card lists its device bytes only).
                if ram_now > 0 {
                    e.ram = c.len;
                }
                if let nmnpcs::Storage::Device { bytes } = storage {
                    e.device += bytes;
                }
            }
            // Device copies whose views were already given back still count on the card.
            let held = p.device_bytes() as u64;
            let listed: u64 = out.values().filter(|r| r.model == p.card.id && r.source == p.source).map(|r| r.device).sum();
            if held > listed {
                out.entry((p.card.id.clone(), p.source.clone(), "(device copies)".to_owned())).or_insert_with(|| Resident {
                    model: p.card.id.clone(),
                    source: p.source.clone(),
                    tensor: "(device copies)".to_owned(),
                    ram: 0,
                    device: held - listed,
                });
            }
        }
        out.into_values().collect()
    }
    /// Release every kept window (for a test, or a caller that needs the card).
    pub fn release_windows(&self) -> usize {
        let mut n = 0;
        while let Some(b) = self.release_window(None) {
            n += b;
        }
        n
    }
    /// Counters and figures now.
    #[must_use]
    pub fn stats(&self) -> Stats {
        let st = self.lock();
        let (reserve, measured) = reserve_of(st.work_peak, st.total);
        Stats {
            reserve,
            measured,
            windows_held: st.windows.len(),
            window_bytes: st.windows.values().map(|e| e.bytes).sum(),
            ..st.stats.clone()
        }
    }
    /// One line for reports: the rule's figures now.
    #[must_use]
    pub fn report(&self) -> String {
        let s = self.stats();
        let free = self.free_now().map_or_else(|| "unread".to_owned(), |f| format!("{} MB", f >> 20));
        let free = match crate::limits::current().vram_mb {
            Some(mb) if mb > 0 => format!("{free} (within the card limit of {mb} MiB)"),
            Some(_) => format!("{free} (processor only from the next start)"),
            None => free,
        };
        format!(
            "device budget: free {free}, reserve {} MB ({}), windows kept {} ({} MB; hits {}, misses {}, kept {}, refused {}, released {}), weights released {}, last release: {}",
            s.reserve >> 20,
            if s.measured { "largest working set measured" } else { "fallback: an eighth of the device" },
            s.windows_held,
            s.window_bytes >> 20,
            s.window_hits,
            s.window_misses,
            s.windows_kept,
            s.windows_refused,
            s.windows_released,
            s.weights_released,
            if s.last_release.is_empty() { "none" } else { &s.last_release }
        )
    }
}

/// The Nano's own bytes on `gpu` (`limits::own_bytes`), the populations read without waiting.
fn own_on(gpu: &Gpu, pops: &[Arc<Population>]) -> usize {
    let populations: usize = pops.iter().filter_map(|p| p.device_bytes_now()).sum();
    crate::limits::own_bytes(&gpu.pool_stats(), populations)
}

/// The release order among idle weights: (reload seconds per byte, last use, item), lowest
/// cost per byte first, then least recently used.
pub fn release_order<T>(c: &mut [(f64, u64, T)]) {
    c.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
}

/// The reserve rule on figures: the measured working peak, else an eighth of the device.
#[must_use]
pub fn reserve_of(work_peak: usize, total: usize) -> (usize, bool) {
    if work_peak > 0 { (work_peak, true) } else { (total / 8, false) }
}

/// Marks a GPU run (step P2): its population counts as in use (its weights are not released
/// meanwhile), and the run's working peak is measured when it ends.
#[derive(Debug)]
pub struct RunMark {
    pop: Arc<Population>,
    gpu: Gpu,
    start: usize,
}
impl RunMark {
    #[must_use]
    pub fn new(pop: &Arc<Population>, o: &Ops) -> Self {
        pop.mark_active(true);
        let start = o.gpu.reset_peak();
        Self { pop: Arc::clone(pop), gpu: o.gpu.clone(), start }
    }
}
impl Drop for RunMark {
    fn drop(&mut self) {
        self.pop.mark_active(false);
        let s = self.gpu.pool_stats();
        get().note_run(s.peak_work.saturating_sub(self.start));
    }
}

/// Section check of the rule on figures (no device needed): the reserve falls back to an eighth
/// of the device until a run is measured, then follows the largest working set; the release
/// order is idle pool, windows least recently used first, then idle weights by reload cost.
pub(crate) fn check() -> Result<(), String> {
    if reserve_of(0, 4 << 30) != (512 << 20, false) || reserve_of(300 << 20, 4 << 30) != (300 << 20, true) {
        return Err("nmengn.residency: reserve rule".into());
    }
    // Release order among idle weights: lowest seconds per byte first, then least recently used.
    let mut c = [(1.7e-9_f64, 5_u64, "reranker"), (1.27e-9, 9, "base"), (1.8e-9, 1, "embedder"), (1.27e-9, 2, "base-older")];
    release_order(&mut c);
    let order: Vec<&str> = c.iter().map(|x| x.2).collect();
    if order != ["base-older", "base", "reranker", "embedder"] {
        return Err(format!("nmengn.residency: weight release order {order:?}"));
    }
    Ok(())
}

/// Section check of step P2 fix 3 (2026-11-10), on the released Qwen3-0.6B on the GPU: weights
/// stay on the device after a run (no release around it); a second run uploads nothing and
/// gives the same score bit for bit; a population marked in use is not released when room is
/// asked for; an idle one is released when the budget needs the room, and the run after the
/// reload gives the same score bit for bit. Without a GPU the check reports that.
/// # Errors
/// A difference, a release that should not happen, or an engine fault.
pub(crate) fn check_weights() -> Result<(), String> {
    use std::fmt::Write as _;
    let mut rep = String::from("nmengn.residency.weights (step P2 fix 3)\n");
    let r = weights_check(&mut rep);
    let _ = writeln!(rep, "result: {}", r.as_ref().map_or_else(|e| format!("FAIL {e}"), |()| "ok".to_owned()));
    let dir = nmtype::check_dir("nmengn-residency-weights").map_err(|e| e.to_string())?;
    std::fs::write(dir.join("report.txt"), rep).map_err(|e| e.to_string())?;
    r
}
fn weights_check(rep: &mut String) -> Result<(), String> {
    use crate::placement::Device;
    use std::fmt::Write as _;
    let home = nmtype::nano_root();
    let pop = Population::open(&home, "qwen3-0.6b", true, true).map_err(|e| e.to_string())?;
    let Some(o) = pop.gpu.clone() else {
        let _ = writeln!(rep, "no GPU open: skipped");
        return Ok(());
    };
    let asm = crate::assembly::Assembly::load(&home.join("designs/assemblies"), "qwen3-reference").map_err(|e| e.to_string())?;
    let m = crate::model::Model::bind(&pop, asm).map_err(|e| e.to_string())?;
    let (q, d) = ("where is the ring buffer pushed?", "crates/x/src/ring.rs\npub fn push(&mut self, x: u32) { self.items.push(x); }\n");
    let score = || crate::roles::rerank_on(&m, Device::Gpu, q, d).map_err(|e| e.to_string());
    let s1 = score()?;
    let (b1, u1) = (pop.device_bytes(), pop.uploaded_bytes());
    let s2 = score()?;
    let (b2, u2) = (pop.device_bytes(), pop.uploaded_bytes());
    let _ = writeln!(
        rep,
        "first run: score bits {:08x}, device weight bytes {b1}, uploaded {u1}; second run: score bits {:08x}, device weight bytes {b2}, uploaded {u2} (resident: nothing uploaded again {})",
        s1.to_bits(),
        s2.to_bits(),
        u2 == u1
    );
    if b1 == 0 || b2 != b1 || u2 != u1 || s2.to_bits() != s1.to_bits() {
        return Err("weights did not stay resident, or the resident run differs".into());
    }
    let total = o.gpu.info().total_bytes;
    pop.mark_active(true);
    let r0 = get().stats().weights_released;
    get().make_room(total, "");
    let held_while_active = pop.device_bytes() == b1 && get().stats().weights_released == r0;
    pop.mark_active(false);
    let kept_for_background = {
        let _bg = background();
        get().make_room(total, "");
        pop.device_bytes() == b1 && get().stats().weights_released == r0
    };
    let _ = writeln!(rep, "room for the whole device asked by background work: idle weights kept {kept_for_background}");
    if !kept_for_background {
        return Err("background work released weights".into());
    }
    get().make_room(total, "");
    let released = pop.device_bytes() == 0 && get().stats().weights_released > r0;
    let s3 = score()?;
    let u3 = pop.uploaded_bytes();
    let _ = writeln!(
        rep,
        "room for the whole device asked: kept while in use {held_while_active}; released when idle {released}; after the reload the score bits {:08x} (equal {}), uploaded {u3}; reload cost measured {:.3} ns per byte",
        s3.to_bits(),
        s3.to_bits() == s1.to_bits(),
        pop.reload_secs_per_byte() * 1e9
    );
    let _ = writeln!(rep, "{}", get().report());
    pop.release_device();
    if !held_while_active || !released || s3.to_bits() != s1.to_bits() || u3 <= u2 {
        return Err("release rule or reload result wrong".into());
    }
    Ok(())
}
