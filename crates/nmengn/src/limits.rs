//! The engine's rules on the user's hardware limits (Roadmap Phase 4.5 (2026-11-25), item 3).
//! The three limits are read live from `settings.json` by `nmtype::limits` (cached; the file is
//! looked at no more than once a second); this module turns them into the figures the engine
//! sizes its work by, so everything that tiles from free memory tiles smaller under a cap with no
//! restart:
//!
//! - card memory: the effective free memory is `min(driver free, cap - the Nano's own card
//!   bytes)`, reported by `residency` (`memory_now`, the room-making of `make_room`, the window
//!   keeping) and by `Population::gpu_free`, which `roles::batch_tokens`, `input_cap`,
//!   `card_doc_cap`, placement and the drone and reader groups read; the idle writer's
//!   all-four-seated rule reads the same figure. A cap of 0 means processor only from the next
//!   start: it is latched when the first MicroModel opens (`cpu_only_at_start`), and a running
//!   process never unbinds its MicroModels for it;
//! - host memory: the room is `min(available, cap - this process's working set)`; the kept
//!   encoded-chunk cache takes at most half the cap (as it takes half the available memory);
//! - threads: `nmtype::budget::total` takes the cap as a ceiling at each grant.
pub use nmtype::limits::{CPU_KEY, KEYS, Limits, RAM_KEY, Reader, Scoped, VRAM_KEY, current, full_key, normalize, parse, read_at, scoped};
use nmjson::Value;
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// The effective free card memory in bytes: the driver's free memory, and under a cap no more
/// than the cap less the Nano's own card bytes. `cap` is `None` for `auto` (0 never reaches here:
/// `Limits::vram_cap_bytes` gives `None` for it, as processor only applies at the next start).
#[must_use]
pub fn effective_free(driver_free: usize, cap: Option<u64>, own: usize) -> usize {
    match cap {
        None => driver_free,
        Some(c) => driver_free.min(usize::try_from(c).unwrap_or(usize::MAX).saturating_sub(own)),
    }
}

/// The Nano's own card bytes from the pool's counters and the populations' device copies:
/// working buffers handed out, idle pooled buffers, and long-lived buffers (weights and window
/// copies). Weights are allocated as long-lived buffers, so the populations' device bytes are
/// part of `live_kept`; the larger of the two is taken, never their sum, so nothing counts twice.
#[must_use]
pub fn own_bytes(pool: &nmffi::PoolStats, populations: usize) -> usize {
    pool.live_work.saturating_add(pool.idle_bytes).saturating_add(pool.live_kept.max(populations))
}

/// The host memory room in bytes: what is available, and under a cap no more than the cap less
/// this process's own working set.
#[must_use]
pub fn host_room(available: u64, cap: Option<u64>, own: u64) -> u64 {
    cap.map_or(available, |c| available.min(c.saturating_sub(own)))
}

/// The host memory room now (`host_room` on the machine's available memory and this process's
/// working set); `None` when the machine's memory cannot be read.
#[must_use]
pub fn host_free_now() -> Option<u64> {
    let (available, _) = nmffi::dynlib::system_memory().ok()?;
    let own = nmffi::dynlib::process_memory().map_or(0, |(_, now)| now as u64);
    Some(host_room(available, current().ram_cap_bytes(), own))
}

/// The ceiling of the kept encoded-chunk cache under a host memory cap: half the cap (the cache
/// takes half the available memory without one), and never above `bytes`.
#[must_use]
pub fn cache_ceiling_for(bytes: usize, cap: Option<u64>) -> usize {
    cap.map_or(bytes, |c| bytes.min(usize::try_from(c / 2).unwrap_or(usize::MAX)))
}

/// `cache_ceiling_for` with the live limits.
#[must_use]
pub fn cache_ceiling(bytes: usize) -> usize {
    cache_ceiling_for(bytes, current().ram_cap_bytes())
}

/// Host memory homeostasis (2026-11-27, MCP stability fix). Each kept cache (the encoded-chunk
/// cache, the relevance-key store) was sized once to half of the memory free when it was made and
/// never looked at the machine again, so the caches of one process together, and of two `serve`
/// processes on one machine, could take more memory than the machine has: on a 32 GB machine one
/// process reached 18 GB before its first call and two ended in an allocation failure
/// (`std::alloc::rust_oom`, exception 0xc0000409). The ceiling of a cache is now live, read at
/// each entry kept: the cache may hold what it holds plus the room left under two bounds, and it
/// gives entries back (least recently used first) when that room is negative:
/// - the machine keeps `host_floor` available (a quarter of its memory, at least 4 GiB), for
///   every other program and for a second Nano process;
/// - this process stays within `process_budget` (`limits.ram_mb` when set, else three eighths of
///   the machine's memory), its MicroModels and everything else it holds counted.
///
/// The machine's memory as (available, total, this process's working set) is read at most every
/// `HOST_READ_MS` milliseconds (a kept entry asks often; the reading is a system call).
const HOST_READ_MS: u128 = 100;
type HostReading = Option<(std::time::Instant, (u64, u64, u64))>;
static HOST_READ: std::sync::Mutex<HostReading> = std::sync::Mutex::new(None);

fn host_reading() -> Option<(u64, u64, u64)> {
    if let Ok(g) = HOST_READ.lock()
        && let Some((at, r)) = *g
        && at.elapsed().as_millis() < HOST_READ_MS
    {
        return Some(r);
    }
    let (available, total) = nmffi::dynlib::system_memory().ok()?;
    let own = nmffi::dynlib::process_memory().map_or(0, |(_, now)| now as u64);
    let r = (available, total, own);
    if let Ok(mut g) = HOST_READ.lock() {
        *g = Some((std::time::Instant::now(), r));
    }
    Some(r)
}

/// The memory the machine keeps available for everything else: a quarter of `total`, at least
/// 4 GiB, never more than half of it.
#[must_use]
pub fn host_floor(total: u64) -> u64 {
    (total / 4).max(4 << 30).min(total / 2)
}

/// The most this process holds in RAM: `limits.ram_mb` when set (`cap`), else three eighths of
/// the machine's memory (12 GiB of 32: the four MicroModels' weights are held in RAM beside their
/// card copies, about 10 GiB on their open weights, and the caches take what is left).
#[must_use]
pub fn process_budget(total: u64, cap: Option<u64>) -> u64 {
    cap.unwrap_or(total / 8 * 3)
}

/// The rule of `host_cache_cap` on a reading: a cache holding `held` bytes under its own ceiling
/// `cap` may hold `held` plus the smaller of the room above the machine's floor and the room
/// under the process's budget (negative room shrinks it), and never more than `cap`.
#[must_use]
pub fn host_cache_cap_for(cap: usize, held: usize, available: u64, total: u64, own: u64, ram_cap: Option<u64>) -> usize {
    let held = held as u64;
    let floor = host_floor(total);
    let budget = process_budget(total, ram_cap);
    // Room may be negative: (held + available) - floor and (held + budget) - own, floored at 0.
    let by_machine = held.saturating_add(available).saturating_sub(floor);
    let by_process = held.saturating_add(budget).saturating_sub(own);
    let live = usize::try_from(by_machine.min(by_process)).unwrap_or(usize::MAX);
    cap.min(live)
}

/// The live ceiling of a kept cache that holds `held` bytes and was made with the ceiling `cap`
/// (see `host_cache_cap_for`); `cache_ceiling(cap)` when the machine's memory cannot be read.
#[must_use]
pub fn host_cache_cap(cap: usize, held: usize) -> usize {
    let cap = cache_ceiling(cap);
    match host_reading() {
        Some((available, total, own)) if total > 0 => host_cache_cap_for(cap, held, available, total, own, current().ram_cap_bytes()),
        _ => cap,
    }
}

/// The host memory rule with the figures now (for `nm_status`).
#[must_use]
pub fn host_rule_value() -> Value {
    let n = |x: u64| Value::Number((x >> 20).to_string());
    let obj = |pairs: Vec<(&str, Value)>| Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect::<BTreeMap<_, _>>());
    match host_reading() {
        Some((available, total, own)) if total > 0 => obj(vec![
            ("available_mb", n(available)),
            ("total_mb", n(total)),
            ("process_mb", n(own)),
            ("machine_floor_mb", n(host_floor(total))),
            ("process_budget_mb", n(process_budget(total, current().ram_cap_bytes()))),
            ("rule", Value::String("kept caches grow only while the machine keeps its floor available and this process stays within its budget (limits.ram_mb, or three eighths of the machine's memory on auto); past either they give entries back, least recently used first".to_owned())),
        ]),
        _ => obj(vec![("rule", Value::String("the machine's memory could not be read; caches keep their own ceilings".to_owned()))]),
    }
}

/// Memory areas (2026-11-27). The process budget (`process_budget`: `limits.ram_mb`, or three
/// eighths of the machine on `auto`) is divided among the areas the engine holds in host memory
/// (`nmtype::limits::AREAS`). An area the user set (`limits.memory.<area>_mb`) has that size; the
/// areas on `auto` share what the set areas leave, in the proportions of `AREA_SHARES`, so the
/// areas together never exceed the process budget. The sum of the set areas may not exceed the
/// budget: `check_areas` refuses such a setting.
pub use nmtype::limits::{
    AREA_KEYS, AREA_RANGE, AREAS, Areas, CACHES_KEY, CONTENT_KEY, WEIGHTS_KEY, area_full, area_holders, area_index, area_used, areas, read_areas_at, scoped_areas, set_area_full,
    set_area_used,
};

/// The auto shares of the process budget in percent, in the order of `AREAS` (content, caches,
/// weights). The rest (`ENGINE_RESERVE`, 15 percent) is the engine's own working memory, which no
/// area holds: the card runtime, staging and working buffers, the tokenizers (measured
/// 2026-11-27 on the Nano's own folder: about 2.5 to 3 GB of the process beside its areas).
pub const AREA_SHARES: [u64; 3] = [25, 15, 45];
/// The share of the process budget, in percent, no area takes on `auto`.
pub const ENGINE_RESERVE: u64 = 15;

/// The budget of each area in bytes under a process budget of `budget` bytes and the sizes `set`.
#[must_use]
pub fn area_budgets_for(budget: u64, set: &Areas) -> [u64; 3] {
    let mb = |x: u64| x.saturating_mul(1 << 20);
    // The engine's reserve stays out of the auto areas' room.
    let reserve = u64::try_from(u128::from(budget) * u128::from(ENGINE_RESERVE) / 100).unwrap_or(0);
    let rest = budget.saturating_sub(mb(set.set_sum_mb())).saturating_sub(reserve);
    let auto_share: u64 = set.0.iter().zip(AREA_SHARES).filter(|(s, _)| s.is_none()).map(|(_, w)| w).sum();
    let mut out = [0_u64; 3];
    for (i, o) in out.iter_mut().enumerate() {
        *o = match set.0[i] {
            Some(m) => mb(m),
            None if auto_share > 0 => u64::try_from(u128::from(rest) * u128::from(AREA_SHARES[i]) / u128::from(auto_share)).unwrap_or(u64::MAX),
            None => 0,
        };
    }
    out
}

/// Whether the sizes `set` fit a machine of `total` bytes under the host cap `ram_cap` (bytes,
/// `None` for auto): the set areas together may not exceed the process budget.
/// # Errors
/// The set areas exceed the budget, with the figures and what to change.
pub fn check_areas(total: u64, ram_cap: Option<u64>, set: &Areas) -> Result<(), String> {
    let budget_mb = process_budget(total, ram_cap) >> 20;
    let sum = set.set_sum_mb();
    if sum > budget_mb {
        let source = if ram_cap.is_some() { "limits.ram_mb" } else { "three eighths of this machine's memory, limits.ram_mb auto" };
        return Err(format!(
            "the memory areas set come to {sum} MiB, above the process budget of {budget_mb} MiB ({source}); lower an area, set one to auto, or raise limits.ram_mb"
        ));
    }
    Ok(())
}

/// `check_areas` on this machine's memory now; `Ok` when the machine's memory cannot be read.
/// # Errors
/// As `check_areas`.
pub fn check_areas_now(ram_mb: Option<u64>, set: &Areas) -> Result<(), String> {
    match host_reading() {
        Some((_, total, _)) if total > 0 => check_areas(total, ram_mb.map(|m| m.saturating_mul(1 << 20)), set),
        _ => Ok(()),
    }
}

/// The budget of one memory area in bytes now (`content`, `caches`, `weights`, or the full
/// `limits.memory.` key): the size the user set, or on `auto` its share of the process budget.
/// `None` for an unknown area or when the machine's memory cannot be read. Read live: a change to
/// `settings.json` applies within a second.
#[must_use]
pub fn area_budget(area: &str) -> Option<u64> {
    let i = area_index(area)?;
    let set = areas();
    let b = match set.0[i] {
        Some(mb) => Some(mb.saturating_mul(1 << 20)),
        None => host_reading().filter(|r| r.1 > 0).map(|(_, total, _)| area_budgets_for(process_budget(total, current().ram_cap_bytes()), &set)[i]),
    };
    nmtype::limits::note_area_budget(area, b);
    b
}

/// The room left in an area in bytes (its budget less what it holds); `None` when the budget is
/// not known.
#[must_use]
pub fn area_room(area: &str) -> Option<u64> {
    area_budget(area).map(|b| b.saturating_sub(area_used(area)))
}

/// The most one holder of an area may keep, in bytes, when it keeps `held` now: what it keeps
/// plus the room left in the area (negative room shrinks it), as `host_cache_cap` rules on the
/// process; `usize::MAX` when the area's budget is not known.
#[must_use]
pub fn area_holder_cap(area: &str, held: usize) -> usize {
    match area_budget(area) {
        Some(b) => usize::try_from((held as u64).saturating_add(b).saturating_sub(area_used(area))).unwrap_or(usize::MAX),
        None => usize::MAX,
    }
}

/// Each area's setting, budget, use and state now (the `memory_areas` block of `nm_status`,
/// `nm_live` and `GET /api/live`).
#[must_use]
pub fn areas_value() -> Value {
    let n = |x: u64| Value::Number((x >> 20).to_string());
    let set = areas();
    let rows = AREAS
        .iter()
        .zip(AREA_KEYS)
        .map(|(area, key)| {
            let budget = area_budget(area);
            let used = area_used(area);
            let full = area_full(area);
            let state = match (full, budget) {
                (true, _) => "full: a pass stopped at the budget; raise the area to read further",
                (false, Some(b)) if used >= b => "at budget",
                _ => "within budget",
            };
            let v = Value::Object(
                [
                    ("key", Value::String(key.to_owned())),
                    ("setting", Value::String(set.setting(key))),
                    ("budget_mb", budget.map_or(Value::Null, n)),
                    ("used_mb", n(used)),
                    ("holders_mb", Value::Object(area_holders(area).into_iter().map(|(h, b)| (h.to_owned(), n(b))).collect())),
                    ("full", Value::Bool(full)),
                    ("state", Value::String(state.to_owned())),
                ]
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v))
                .collect(),
            );
            ((*area).to_owned(), v)
        })
        .collect::<BTreeMap<_, _>>();
    Value::Object(rows)
}

static CPU_ONLY: OnceLock<bool> = OnceLock::new();

/// Whether this process runs processor only because `limits.vram_mb` was 0 when its first
/// MicroModel opened. Latched at that first call, so a change applies at the next start.
#[must_use]
pub fn cpu_only_at_start() -> bool {
    *CPU_ONLY.get_or_init(|| current().vram_mb == Some(0))
}

/// The latched value of `cpu_only_at_start`, `None` before any MicroModel asked for the card.
#[must_use]
pub fn cpu_only_latched() -> Option<bool> {
    CPU_ONLY.get().copied()
}

/// The limits in effect now, for the interface, `nm_live` and `nm_settings`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Effective {
    pub limits: Limits,
    /// Card: the driver's free memory, the Nano's own bytes, the effective free memory (MiB).
    pub card_free_mb: Option<u64>,
    pub card_own_mb: Option<u64>,
    pub card_effective_free_mb: Option<u64>,
    pub card_total_mb: Option<u64>,
    /// This process runs processor only (latched at its start), and a cap of 0 waits for the
    /// next start.
    pub cpu_only_now: bool,
    pub cpu_only_next_start: bool,
    /// Host: available, this process's working set, the room (MiB), and the cache ceiling.
    pub host_available_mb: Option<u64>,
    pub host_own_mb: Option<u64>,
    pub host_room_mb: Option<u64>,
    /// Threads: the hardware's and the budget's now.
    pub threads_hardware: usize,
    pub threads_effective: usize,
}

/// The limits in effect now in this process.
#[must_use]
pub fn effective() -> Effective {
    let limits = current();
    let res = crate::residency::get();
    let driver = res.memory_driver();
    let own = res.own_bytes();
    let host = nmffi::dynlib::system_memory().ok();
    let host_own = nmffi::dynlib::process_memory().ok().map(|(_, now)| now as u64);
    let cpu_only_now = cpu_only_latched() == Some(true);
    Effective {
        limits,
        card_free_mb: driver.map(|(f, _)| (f >> 20) as u64),
        card_own_mb: own.map(|b| (b >> 20) as u64),
        card_effective_free_mb: driver.map(|(f, _)| (effective_free(f, limits.vram_cap_bytes(), own.unwrap_or(0)) >> 20) as u64),
        card_total_mb: driver.map(|(_, t)| (t >> 20) as u64),
        cpu_only_now,
        cpu_only_next_start: limits.vram_mb == Some(0) && !cpu_only_now,
        host_available_mb: host.map(|(a, _)| a >> 20),
        host_own_mb: host_own.map(|b| b >> 20),
        host_room_mb: host.map(|(a, _)| host_room(a, limits.ram_cap_bytes(), host_own.unwrap_or(0)) >> 20),
        threads_hardware: nmtype::budget::hardware(),
        threads_effective: nmtype::budget::total(),
    }
}

/// What the card limit does now, in words.
#[must_use]
pub fn vram_note(e: &Effective) -> String {
    match (e.limits.vram_mb, e.cpu_only_now) {
        (_, true) => "processor only: limits.vram_mb was 0 when this process started; set it to auto or a size and restart to use the card".to_owned(),
        (Some(0), false) => "processor only applies at the next start (limits.vram_mb is 0); the MicroModels seated now stay seated until then".to_owned(),
        (Some(mb), false) => format!("the Nano keeps its own card memory within {mb} MiB: batches tile smaller and idle MicroModels are evicted and restored to stay under it"),
        (None, false) => "auto: the Nano uses what the driver reports free, less the live calls' room".to_owned(),
    }
}

/// The limits in effect as a value (the `limits` field of `nm_live`'s capacity block and of
/// `GET /api/limits`).
#[must_use]
pub fn to_value(e: &Effective) -> Value {
    let n = |x: u64| Value::Number(x.to_string());
    let opt = |x: Option<u64>| x.map_or(Value::Null, n);
    let obj = |pairs: Vec<(&str, Value)>| Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect::<BTreeMap<_, _>>());
    let l = e.limits;
    obj(vec![
        (
            "vram",
            obj(vec![
                ("setting", Value::String(l.setting(VRAM_KEY))),
                ("cap_mb", opt(l.vram_mb.filter(|mb| *mb > 0))),
                ("used_mb", opt(e.card_own_mb)),
                ("driver_free_mb", opt(e.card_free_mb)),
                ("effective_free_mb", opt(e.card_effective_free_mb)),
                ("total_mb", opt(e.card_total_mb)),
                ("cpu_only_now", Value::Bool(e.cpu_only_now)),
                ("cpu_only_next_start", Value::Bool(e.cpu_only_next_start)),
                ("note", Value::String(vram_note(e))),
            ]),
        ),
        (
            "ram",
            obj(vec![
                ("setting", Value::String(l.setting(RAM_KEY))),
                ("cap_mb", opt(l.ram_mb)),
                ("used_mb", opt(e.host_own_mb)),
                ("available_mb", opt(e.host_available_mb)),
                ("room_mb", opt(e.host_room_mb)),
                ("cache_ceiling_mb", opt(l.ram_mb.map(|mb| mb / 2))),
            ]),
        ),
        (
            "cpu",
            obj(vec![
                ("setting", Value::String(l.setting(CPU_KEY))),
                ("cap", opt(l.cpu_threads.map(|c| c as u64))),
                ("hardware_threads", n(e.threads_hardware as u64)),
                ("effective_threads", n(e.threads_effective as u64)),
            ]),
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: usize = 1 << 20;

    #[test]
    fn limits_host_cache_cap_keeps_floor_and_budget() {
        const GB: u64 = 1 << 30;
        let g = |x: u64| usize::try_from(x).unwrap_or(usize::MAX);
        // 32 GB machine, 20 GB available, process at 10 GB: room is min(20 - 8, 12 - 10) = 2 GB.
        assert_eq!(host_cache_cap_for(g(100 * GB), 0, 20 * GB, 32 * GB, 10 * GB, None), g(2 * GB));
        // The cache's own ceiling still bounds it.
        assert_eq!(host_cache_cap_for(g(GB), 0, 20 * GB, 32 * GB, 6 * GB, None), g(GB));
        // A second process took the memory: 6 GB available, below the 8 GB floor, so a cache
        // holding 3 GB gives 2 GB back.
        assert_eq!(host_cache_cap_for(g(100 * GB), g(3 * GB), 6 * GB, 32 * GB, 7 * GB, None), g(GB));
        // Over the process budget (the regression of 2026-11-27: 18 GB in one process): shrink.
        assert_eq!(host_cache_cap_for(g(100 * GB), g(2 * GB), 14 * GB, 32 * GB, 18 * GB, None), 0);
        // The auto budget is three eighths of the machine.
        assert_eq!(process_budget(32 * GB, None), 12 * GB);
        // limits.ram_mb sets the budget.
        assert_eq!(host_cache_cap_for(g(100 * GB), 0, 20 * GB, 32 * GB, 6 * GB, Some(12 * GB)), g(6 * GB));
        // The floor: a quarter, at least 4 GiB, at most half.
        assert_eq!(host_floor(32 * GB), 8 * GB);
        assert_eq!(host_floor(8 * GB), 4 * GB);
        assert_eq!(host_floor(4 * GB), 2 * GB);
    }

    #[test]
    fn limits_memory_areas_divide_the_budget() {
        const GB: u64 = 1 << 30;
        // All auto on a 20 GiB budget: 25, 15 and 45 percent, 15 left to the engine.
        let b = area_budgets_for(20 * GB, &Areas::AUTO);
        let sum = b.iter().sum::<u64>();
        assert!(sum <= 17 * GB && 17 * GB - sum < 3, "the areas take 85 percent, short only by rounding");
        assert_eq!(b[0], 5 * GB);
        assert_eq!(b[2], 9 * GB);
        // Content set to 1 GiB: the auto areas share 20 - 1 - 3 (the reserve) in 15:45.
        let b = area_budgets_for(20 * GB, &Areas::AUTO.with("content", Some(1_024)));
        assert_eq!(b[0], GB);
        assert!(16 * GB - (b[1] + b[2]) < 3);
        assert!(b[2] > b[1]);
        // Over the budget: refused, with the figures.
        let over = Areas([Some(8_192), Some(4_096), Some(1_024)]);
        let e = check_areas(32 * GB, None, &over).unwrap_err();
        assert!(e.contains("13312 MiB") && e.contains("12288 MiB"), "{e}");
        assert!(check_areas(32 * GB, Some(16 * GB), &over).is_ok());
        // The scoped setting reaches area_budget.
        let _g = scoped_areas(Areas::AUTO.with("weights", Some(2_048)));
        assert_eq!(area_budget("weights"), Some(2 * GB));
        assert_eq!(area_budget(WEIGHTS_KEY), Some(2 * GB));
        assert_eq!(area_budget("nothing"), None);
        assert!(areas_value().get("content").and_then(|c| c.get("key")).is_ok());
    }

    #[test]
    fn limits_effective_free_formula() {
        // Auto: the driver's figure.
        assert_eq!(effective_free(3_000 * MB, None, 1_400 * MB), 3_000 * MB);
        // A cap above what the Nano holds plus the free memory: the driver's figure still.
        assert_eq!(effective_free(1_000 * MB, Some(4_096 << 20), 1_400 * MB), 1_000 * MB);
        // A cap tighter than the card: the cap less the Nano's own bytes.
        assert_eq!(effective_free(3_000 * MB, Some(2_048 << 20), 1_400 * MB), 648 * MB);
        // Already over the cap: nothing free, so room-making evicts.
        assert_eq!(effective_free(3_000 * MB, Some(1_024 << 20), 1_400 * MB), 0);
        // 0 is not a live cap.
        assert_eq!(Limits { vram_mb: Some(0), ..Limits::AUTO }.vram_cap_bytes(), None);
    }

    #[test]
    fn limits_own_bytes_count_nothing_twice() {
        let pool = nmffi::PoolStats { live_work: 100, idle_bytes: 20, live_kept: 1_000, ..nmffi::PoolStats::default() };
        assert_eq!(own_bytes(&pool, 900), 1_120, "weights are inside live_kept");
        assert_eq!(own_bytes(&pool, 1_500), 1_620, "device copies beyond the pool's count still count");
    }

    #[test]
    fn limits_host_room_and_cache_ceiling() {
        let gib = 1_u64 << 30;
        assert_eq!(host_room(20 * gib, None, 3 * gib), 20 * gib);
        assert_eq!(host_room(20 * gib, Some(8 * gib), 3 * gib), 5 * gib);
        assert_eq!(host_room(2 * gib, Some(8 * gib), 3 * gib), 2 * gib);
        assert_eq!(host_room(20 * gib, Some(2 * gib), 3 * gib), 0);
        let half_free = 10_usize << 30;
        assert_eq!(cache_ceiling_for(half_free, None), half_free);
        assert_eq!(cache_ceiling_for(half_free, Some(8 * gib)), 4 << 30);
        assert_eq!(cache_ceiling_for(1 << 30, Some(8 * gib)), 1 << 30);
    }

    #[test]
    fn limits_scoped_values_reach_the_engine_rules() {
        let _g = scoped(Limits { vram_mb: Some(0), ram_mb: Some(4_096), cpu_threads: Some(1) });
        assert_eq!(cache_ceiling(16 << 30), 2 << 30);
        assert_eq!(nmtype::budget::total(), 1);
        let e = effective();
        let v = to_value(&e);
        assert_eq!(v.get("ram").and_then(|r| r.get("cache_ceiling_mb")).and_then(Value::integer).ok(), Some(2_048));
        assert_eq!(v.get("cpu").and_then(|r| r.get("effective_threads")).and_then(Value::integer).ok(), Some(1));
        assert!(vram_note(&e).contains("processor only"));
    }
}
