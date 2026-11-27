//! Load a system library at run time and look up its symbols (Windows: kernel32
//! `LoadLibraryW`, `GetProcAddress`; kernel32 is linked by the standard library). The handle
//! is kept for the process lifetime: the library is never unloaded, so symbol addresses
//! taken from it stay valid.
use std::ffi::c_void;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    fn GetCurrentProcess() -> *mut c_void;
    fn K32GetProcessMemoryInfo(process: *mut c_void, counters: *mut MemoryCounters, cb: u32) -> i32;
    fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
    fn GetTickCount64() -> u64;
    fn GetProcessTimes(process: *mut c_void, creation: *mut u64, exit: *mut u64, kernel: *mut u64, user: *mut u64) -> i32;
    fn SetProcessInformation(process: *mut c_void, class: u32, info: *const c_void, size: u32) -> i32;
    fn TerminateProcess(process: *mut c_void, code: u32) -> i32;
    fn GetDiskFreeSpaceExW(directory: *const u16, free_to_caller: *mut u64, total: *mut u64, total_free: *mut u64) -> i32;
}

/// `PROCESS_POWER_THROTTLING_STATE` (processthreadsapi), x64 layout.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ProcessPowerThrottlingState {
    version: u32,
    control_mask: u32,
    state_mask: u32,
}

/// `ProcessPowerThrottling` class for `SetProcessInformation`.
const PROCESS_POWER_THROTTLING: u32 = 4;
/// `PROCESS_POWER_THROTTLING_CURRENT_VERSION`.
const PROCESS_POWER_THROTTLING_VERSION: u32 = 1;
/// `PROCESS_POWER_THROTTLING_EXECUTION_SPEED`.
const PROCESS_POWER_THROTTLING_EXECUTION_SPEED: u32 = 0x1;

/// Opt this process out of Windows 11 EcoQoS efficiency-core throttling so a background MCP
/// daemon or warm bench server stays at full processor speed when another window has focus.
/// Best-effort: older Windows builds that do not support the class ignore the call cleanly.
pub fn disable_power_throttling() {
    let state = ProcessPowerThrottlingState {
        version: PROCESS_POWER_THROTTLING_VERSION,
        control_mask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        state_mask: 0,
    };
    // SAFETY: GetCurrentProcess returns the calling process's pseudo-handle, always valid.
    let process = unsafe { GetCurrentProcess() };
    // SAFETY: `state` is a valid, initialized `ProcessPowerThrottlingState` whose exact byte size
    // is passed to `SetProcessInformation` for class `ProcessPowerThrottling`.
    let _ = unsafe {
        SetProcessInformation(
            process,
            PROCESS_POWER_THROTTLING,
            (&raw const state).cast::<c_void>(),
            u32::try_from(size_of::<ProcessPowerThrottlingState>()).unwrap_or(0),
        )
    };
}

/// End this process now with `code` (kernel32 `TerminateProcess` on itself), without the
/// libraries' detach handlers (lane T1, 2026-11-19, T-086): a process that returns from `main`
/// runs `ExitProcess`, which first ends every other thread wherever it stands and then calls
/// each library's detach; a thread ended inside a card driver call leaves the driver's lock
/// held, and the driver's detach then waits on it for good (the bench stayed with one idle
/// thread and about 5 GB for an hour). The system still frees the process's memory, handles and
/// card context. Callers flush what they write before calling it.
pub fn end_process(code: u32) -> ! {
    // SAFETY: GetCurrentProcess returns the calling process's pseudo-handle, always valid.
    let process = unsafe { GetCurrentProcess() };
    // SAFETY: the handle is this process's own; TerminateProcess on it does not return when it
    // succeeds.
    unsafe {
        TerminateProcess(process, code);
    }
    // TerminateProcess on the own process does not fail in practice; the ordinary exit stands in.
    #[expect(clippy::disallowed_methods, reason = "the process ends here; no main to return to")]
    std::process::exit(i32::try_from(code).unwrap_or(1))
}

/// The free bytes available to this process and the total bytes of the volume holding `dir`
/// (kernel32 `GetDiskFreeSpaceExW`), or `None` when the volume cannot be read. 2026-11-26 (Luis
/// Jake Gabriel III: "Any dependencies should be rebuilt in rust"): the free space comes from
/// Windows itself, not from running the `fsutil` program.
#[must_use]
pub fn disk_space(dir: &std::path::Path) -> Option<(u64, u64)> {
    use std::os::windows::ffi::OsStrExt as _;
    let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    let (mut free, mut total, mut total_free) = (0_u64, 0_u64, 0_u64);
    // SAFETY: `wide` is a NUL-terminated UTF-16 path that lives across the call; the three out
    // pointers point at initialized u64 locals that GetDiskFreeSpaceExW writes.
    let ok = unsafe { GetDiskFreeSpaceExW(wide.as_ptr(), &raw mut free, &raw mut total, &raw mut total_free) };
    (ok != 0 && total > 0).then_some((free, total))
}

/// FILETIME ticks (100 ns) from 1601-01-01 to 1970-01-01.
const EPOCH_TICKS: u64 = 116_444_736_000_000_000;

/// When this process was created, in milliseconds since 1970-01-01 UTC (kernel32
/// `GetProcessTimes`), so a short-lived program can time itself from its own launch.
/// # Errors
/// The system call failed or returned a time before 1970.
pub fn process_started_ms() -> Result<u64, String> {
    let (mut creation, mut exit, mut kernel, mut user) = (0_u64, 0_u64, 0_u64, 0_u64);
    // SAFETY: GetCurrentProcess returns a pseudo-handle that needs no closing and is always
    // valid for the calling process.
    let process = unsafe { GetCurrentProcess() };
    // SAFETY: each pointer is a live, writable u64, which has the size and layout of FILETIME
    // (two u32 halves, low first, on little-endian x64) and at least its alignment.
    let ok = unsafe { GetProcessTimes(process, &raw mut creation, &raw mut exit, &raw mut kernel, &raw mut user) };
    if ok == 0 {
        return Err("nmffi.dynlib: GetProcessTimes failed".into());
    }
    creation.checked_sub(EPOCH_TICKS).map(|t| t / 10_000).ok_or_else(|| format!("nmffi.dynlib: process creation time {creation} before 1970"))
}

/// `MEMORYSTATUSEX` (sysinfoapi), x64 layout (2026-11-10).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct MemoryStatusEx {
    length: u32,
    memory_load: u32,
    total_phys: u64,
    avail_phys: u64,
    total_page_file: u64,
    avail_page_file: u64,
    total_virtual: u64,
    avail_virtual: u64,
    avail_extended_virtual: u64,
}

/// The machine's physical memory now, in bytes: (available, total) (2026-11-10; the RAM limit
/// of the kept encoded chunks is derived from it at run time).
/// # Errors
/// The system call failed.
pub fn system_memory() -> Result<(u64, u64), String> {
    let mut m = MemoryStatusEx { length: u32::try_from(size_of::<MemoryStatusEx>()).unwrap_or(0), ..MemoryStatusEx::default() };
    // SAFETY: `m` is a live, writable MEMORYSTATUSEX whose `length` field holds its size.
    let ok = unsafe { GlobalMemoryStatusEx(&raw mut m) };
    if ok == 0 {
        return Err("nmffi.dynlib: GlobalMemoryStatusEx failed".into());
    }
    Ok((m.avail_phys, m.total_phys))
}

/// The machine's commit memory now, in bytes: (available, limit) (`MEMORYSTATUSEX`
/// `ullAvailPageFile`, `ullTotalPageFile`). A load that holds open weights in RAM is charged
/// against this limit (2026-11-19, lane B: the writer at idle loads only when 7 GB are free).
/// # Errors
/// The system call failed.
pub fn commit_memory() -> Result<(u64, u64), String> {
    let mut m = MemoryStatusEx { length: u32::try_from(size_of::<MemoryStatusEx>()).unwrap_or(0), ..MemoryStatusEx::default() };
    // SAFETY: `m` is a live, writable MEMORYSTATUSEX whose `length` field holds its size.
    let ok = unsafe { GlobalMemoryStatusEx(&raw mut m) };
    if ok == 0 {
        return Err("nmffi.dynlib: GlobalMemoryStatusEx failed".into());
    }
    Ok((m.avail_page_file, m.total_page_file))
}

/// `PROCESS_MEMORY_COUNTERS` (psapi), x64 layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct MemoryCounters {
    cb: u32,
    page_fault_count: u32,
    peak_working_set: usize,
    working_set: usize,
    quota_peak_paged_pool: usize,
    quota_paged_pool: usize,
    quota_peak_non_paged_pool: usize,
    quota_non_paged_pool: usize,
    pagefile_usage: usize,
    peak_pagefile_usage: usize,
}

/// This process's peak and current working set in bytes (for run reports: peak RAM).
/// # Errors
/// The system call failed.
pub fn process_memory() -> Result<(usize, usize), String> {
    let mut c = MemoryCounters { cb: u32::try_from(size_of::<MemoryCounters>()).unwrap_or(0), ..MemoryCounters::default() };
    // SAFETY: GetCurrentProcess returns a pseudo-handle that needs no closing and is always
    // valid for the calling process.
    let process = unsafe { GetCurrentProcess() };
    // SAFETY: `c` is a live, writable PROCESS_MEMORY_COUNTERS whose `cb` field holds its size.
    let ok = unsafe { K32GetProcessMemoryInfo(process, &raw mut c, c.cb) };
    if ok == 0 {
        return Err("nmffi.dynlib: K32GetProcessMemoryInfo failed".into());
    }
    Ok((c.peak_working_set, c.working_set))
}

/// A loaded library (never unloaded).
#[derive(Debug)]
pub struct Library {
    handle: *mut c_void,
    name: String,
}
// SAFETY: the handle is an opaque module address that kernel32 allows from any thread;
// it is never freed (no FreeLibrary), so sharing it cannot dangle.
unsafe impl Send for Library {}
// SAFETY: as above; GetProcAddress is thread-safe and the handle is read-only here.
unsafe impl Sync for Library {}

impl Library {
    /// Load `name` (a DLL name resolved by the system search order, for example
    /// `nvcuda.dll`, which ships with the display driver).
    /// # Errors
    /// The library is not present.
    pub fn open(name: &str) -> Result<Self, String> {
        let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
        // SAFETY: `wide` is a NUL-terminated UTF-16 string that outlives the call.
        let handle = unsafe { LoadLibraryW(wide.as_ptr()) };
        if handle.is_null() {
            return Err(format!("nmffi.dynlib: {name} not found"));
        }
        Ok(Self { handle, name: name.to_owned() })
    }
    /// The address of an exported symbol.
    /// # Errors
    /// The symbol is not exported by the library.
    pub fn symbol(&self, symbol: &str) -> Result<*mut c_void, String> {
        let c: Vec<u8> = symbol.bytes().chain(std::iter::once(0)).collect();
        // SAFETY: the handle came from LoadLibraryW and is never freed; `c` is NUL-terminated
        // and outlives the call.
        let p = unsafe { GetProcAddress(self.handle, c.as_ptr()) };
        if p.is_null() {
            return Err(format!("nmffi.dynlib: {} has no symbol {symbol}", self.name));
        }
        Ok(p)
    }
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Self-check of the `dynlib` FFI module: kernel32 loads, a known export resolves, an
/// unknown export is refused, and the machine's physical memory reads (2026-11-10).
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    disable_power_throttling();
    let k = Library::open("kernel32.dll")?;
    k.symbol("GetTickCount64")?;
    if k.symbol("NoSuchExportForTheMini").is_ok() {
        return Err("nmffi.dynlib: an unknown symbol resolved".into());
    }
    let (peak, now) = process_memory()?;
    if peak == 0 || now > peak {
        return Err(format!("nmffi.dynlib: process memory {peak} peak, {now} now"));
    }
    let (avail, total) = system_memory()?;
    if total == 0 || avail > total {
        return Err(format!("nmffi.dynlib: system memory {avail} available of {total}"));
    }
    let started = process_started_ms()?;
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_err(|e| e.to_string())?.as_millis();
    let since = now.checked_sub(u128::from(started)).ok_or_else(|| format!("nmffi.dynlib: process created at {started} ms, after now ({now} ms)"))?;
    // SAFETY: GetTickCount64 takes no arguments and only reads the system's uptime.
    let up = u128::from(unsafe { GetTickCount64() });
    if process_started_ms()? != started || since > up + 1_000 {
        return Err(format!("nmffi.dynlib: process created at {started} ms, {since} ms before now, the system up {up} ms"));
    }
    Ok(())
}
