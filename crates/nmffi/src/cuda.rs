//! The CUDA driver API (`nvcuda.dll`, part of the display driver) and the copied cukern
//! kernels. The kernels were compiled once by nvcc 13.0 into fatbins (sm_75 SASS plus
//! compute_75 PTX; `kernels/build-fatbins.cmd`, hashed in INTEGRITY.md) and are embedded in
//! this crate, so a run reads nothing outside the folder. One primary context per device;
//! every call takes the device lock and makes the context current on the calling thread, so
//! any thread may use a `Gpu`. Buffers free themselves on drop. Launches go to the legacy
//! default stream, so operations run in issue order and a download waits for them.
//!
//! 2026-11-10 (step P2): one shared handle per device (`Gpu::open` returns the same device
//! state to every caller in the process), and a buffer pool. A working buffer that is dropped
//! goes back to the pool without a device wait; the next request of a fitting size takes it
//! (best fit, less than twice the request), zeroed as a new one is. New working buffers are
//! sized by class (a quarter of the power of two below the request, so requests of nearby sizes,
//! such as chunks of different token counts, share buffers). Reuse needs no
//! synchronisation because every caller issues to the one legacy default stream of the one
//! primary context, so the reuse is ordered after every earlier use. Long-lived buffers
//! (weights, window copies; `KeptScope`) are exact-size and never pooled. On out of memory the
//! pool frees its idle buffers, asks the registered reclaimer (the residency manager) to give
//! memory back, and retries once. Method from the caching allocators of llama.cpp
//! (`ggml_cuda_pool_leg`, MIT licence: best fit, look-ahead capacity, free the cache and retry
//! on out of memory) and PyTorch (`c10/cuda/CUDACachingAllocator.cpp`, BSD-3-Clause licence:
//! sizes rounded to classes, reuse on the same stream without synchronisation, free cached
//! blocks and retry), read
//! online and written here in the mini's own Rust. The idle limit is set at run time by the
//! residency manager (the working set measured per run); `set_pool(false)` restores the
//! allocate, zero, wait and free path exactly.
use crate::dynlib::Library;
use std::cell::Cell;
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, Weak};

// Device binary16 and host u16 share one byte order (x86-64 and the GPU are little-endian), so
// u16 rows are copied to and from the device as they are (`upload_u16_at`, `download_u16`).
const _: () = assert!(cfg!(target_endian = "little"));

/// `CUDA_ERROR_OUT_OF_MEMORY`.
const OUT_OF_MEMORY: i32 = 2;

type Ctx = *mut c_void;
type Module = *mut c_void;
type Func = *mut c_void;
type Stream = *mut c_void;

/// The kernel images of the mini (copied cukern sources, built by `kernels/build-fatbins.cmd`)
/// and the entry points each exports.
const IMAGES: &[(&str, &[u8], &[&str])] = &[
    (
        "element",
        include_bytes!("../../../kernels/build/element.fatbin"),
        &[
            "_ZN4emma12embed_kernelE12emma_op_args",
            "_ZN4emma15norm_row_kernelE12emma_op_args",
            "_ZN4emma11norm_kernelE12emma_op_args",
            "_ZN4emma13binary_kernelEi12emma_op_args",
            "_ZN4emma11rope_kernelE12emma_op_argsi",
        ],
    ),
    (
        "linear",
        include_bytes!("../../../kernels/build/linear.fatbin"),
        &[
            "_ZN4emma18vector_half_kernelE12emma_op_args",
            "_ZN4emma23vector_int8_half_kernelE12emma_op_args",
            "_ZN4emma11wide_kernelILi8ELi4ELb1EEEvi12emma_op_args",
            "_ZN4emma13linear_kernelIfEEvi12emma_op_args",
            "_ZN4emma12tiled_kernelIfEEvi12emma_op_args",
            "_ZN4emma11wide_kernelILi8ELi4ELb0EEEvi12emma_op_args",
            "_ZN4emma14wide_h2_kernelILi8ELi4EEEv12emma_op_args",
            "_ZN4emma16q8_blocks_kernelE12emma_op_args",
            "_ZN4emma16wide_dp4a_kernelILi8ELi4EEEv12emma_op_args",
            "_ZN4emma16q4_unpack_kernelE12emma_op_args",
        ],
    ),
    (
        "attention",
        include_bytes!("../../../kernels/build/attention.fatbin"),
        &[
            "_ZN4emma12score_kernelE12emma_op_argsNS_8GeometryEPf",
            "_ZN4emma17score_tile_kernelE12emma_op_argsNS_8GeometryEPf",
            "_ZN4emma18softmax_row_kernelE12emma_op_argsNS_8GeometryEPf",
            "_ZN4emma12value_kernelE12emma_op_argsNS_8GeometryEPKf",
        ],
    ),
    (
        "prefill",
        include_bytes!("../../../kernels/build/prefill.fatbin"),
        &["_ZN4emma14prefill_kernelIfEEv12emma_op_argsNS_8GeometryE", "_ZN4emma15prefill2_kernelIfEEv12emma_op_argsNS_8GeometryE"],
    ),
    ("state", include_bytes!("../../../kernels/build/state.fatbin"), &["_ZN4emma13greedy_kernelE12emma_op_args"]),
    ("segment", include_bytes!("../../../kernels/build/segment.fatbin"), &["_ZN4emma14segment_kernelE12emma_op_argsNS_8GeometryE"]),
    ("flash", include_bytes!("../../../kernels/build/flash.fatbin"), &["_ZN4emma12flash_kernelE12emma_op_argsNS_8GeometryE"]),
    ("q8half", include_bytes!("../../../kernels/build/q8half.fatbin"), &["_ZN4emma16wide_h2q8_kernelILi8ELi4EEEv12emma_op_args"]),
    ("filter", include_bytes!("../../../kernels/build/filter.fatbin"), &["_ZN4emma13filter_kernelE12emma_op_args"]),
    (
        "fast",
        include_bytes!("../../../kernels/build/fast.fatbin"),
        &[
            "_ZN4emma16dp4a_tile_kernelE12emma_op_args",
            "_ZN4emma16h2q8_tile_kernelE12emma_op_args",
            "_ZN4emma17q8_blocks2_kernelE12emma_op_args",
            "_ZN4emma17h2q8_fetch_kernelILi4EEEv12emma_op_args",
            "_ZN4emma17h2q8_fetch_kernelILi8EEEv12emma_op_args",
            "_ZN4emma14dp4a_tt_kernelILi8ELi4ELi2EEEv12emma_op_args",
            "_ZN4emma14dp4a_tt_kernelILi8ELi8ELi1EEEv12emma_op_args",
            "_ZN4emma14dp4a_tt_kernelILi4ELi8ELi2EEEv12emma_op_args",
            "_ZN4emma15wide_q8f_kernelE12emma_op_args",
            "_ZN4emma15h2_fetch_kernelE12emma_op_args",
        ],
    ),
    (
        "fastel",
        include_bytes!("../../../kernels/build/fastel.fatbin"),
        &[
            "_ZN4emma14binary8_kernelEi12emma_op_args",
            "_ZN4emma12rope2_kernelE12emma_op_args",
            "_ZN4emma16norm_warp_kernelILi4EEEv12emma_op_args",
            "_ZN4emma16norm_warp_kernelILi32EEEv12emma_op_args",
            "_ZN4emma15segments_kernelE12emma_op_argsNS_8GeometryEPKi",
            "_ZN4emma20segments_wide_kernelILi4EEEv12emma_op_argsNS_8GeometryEPKi",
            "_ZN4emma20segments_wide_kernelILi8EEEv12emma_op_argsNS_8GeometryEPKi",
            "_ZN4emma21flash_segments_kernelE12emma_op_argsNS_8GeometryEPKi",
            "_ZN4emma23flash_segments_h_kernelE12emma_op_argsNS_8GeometryEPKi",
            "_ZN4emma15part_lse_kernelE12emma_op_argsNS_8GeometryEPKiS3_",
        ],
    ),
    (
        "bmq8",
        include_bytes!("../../../kernels/build/bmq8.fatbin"),
        &[
            "_ZN4emma15bmq8_vec_kernelE12emma_op_args",
            "_ZN4emma17bmq8_small_kernelE12emma_op_args",
            "_ZN4emma18bmq8_unpack_kernelE12emma_op_args",
        ],
    ),
];

/// `cuLaunchKernel`.
type Launch = unsafe extern "system" fn(Func, u32, u32, u32, u32, u32, u32, u32, *mut c_void, *mut *mut c_void, *mut *mut c_void) -> i32;

#[derive(Clone, Copy)]
struct Api {
    init: unsafe extern "system" fn(u32) -> i32,
    device_get: unsafe extern "system" fn(*mut i32, i32) -> i32,
    device_name: unsafe extern "system" fn(*mut u8, i32, i32) -> i32,
    device_mem: unsafe extern "system" fn(*mut usize, i32) -> i32,
    device_attr: unsafe extern "system" fn(*mut i32, i32, i32) -> i32,
    ctx_retain: unsafe extern "system" fn(*mut Ctx, i32) -> i32,
    ctx_current: unsafe extern "system" fn(Ctx) -> i32,
    module_load: unsafe extern "system" fn(*mut Module, *const c_void) -> i32,
    module_func: unsafe extern "system" fn(*mut Func, Module, *const u8) -> i32,
    alloc: unsafe extern "system" fn(*mut u64, usize) -> i32,
    free: unsafe extern "system" fn(u64) -> i32,
    htod: unsafe extern "system" fn(u64, *const c_void, usize) -> i32,
    dtoh: unsafe extern "system" fn(*mut c_void, u64, usize) -> i32,
    dtod: unsafe extern "system" fn(u64, u64, usize) -> i32,
    memset: unsafe extern "system" fn(u64, u8, usize) -> i32,
    launch: Launch,
    sync: unsafe extern "system" fn() -> i32,
    mem_info: unsafe extern "system" fn(*mut usize, *mut usize) -> i32,
    // 2026-11-13, a second device stream with pinned host memory (cartridge loading step L).
    stream_create: unsafe extern "system" fn(*mut Stream, u32) -> i32,
    stream_sync: unsafe extern "system" fn(Stream) -> i32,
    stream_query: unsafe extern "system" fn(Stream) -> i32,
    stream_destroy: unsafe extern "system" fn(Stream) -> i32,
    host_alloc: unsafe extern "system" fn(*mut *mut c_void, usize) -> i32,
    host_free: unsafe extern "system" fn(*mut c_void) -> i32,
    htod_async: unsafe extern "system" fn(u64, *const c_void, usize, Stream) -> i32,
}

/// Look up one driver entry point as a typed function pointer.
macro_rules! entry {
    ($lib:expr, $name:literal, $t:ty) => {{
        let p = $lib.symbol($name)?;
        // SAFETY: the symbol is the documented CUDA driver entry point of this name, and the
        // field type it is cast to is that entry point's C signature (CUDA 13 driver API,
        // x64, where the system and C calling conventions are the same).
        unsafe { std::mem::transmute::<*mut c_void, $t>(p) }
    }};
}

impl Api {
    fn load(lib: &Library) -> Result<Self, String> {
        Ok(Self {
            init: entry!(lib, "cuInit", unsafe extern "system" fn(u32) -> i32),
            device_get: entry!(lib, "cuDeviceGet", unsafe extern "system" fn(*mut i32, i32) -> i32),
            device_name: entry!(lib, "cuDeviceGetName", unsafe extern "system" fn(*mut u8, i32, i32) -> i32),
            device_mem: entry!(lib, "cuDeviceTotalMem_v2", unsafe extern "system" fn(*mut usize, i32) -> i32),
            device_attr: entry!(lib, "cuDeviceGetAttribute", unsafe extern "system" fn(*mut i32, i32, i32) -> i32),
            ctx_retain: entry!(lib, "cuDevicePrimaryCtxRetain", unsafe extern "system" fn(*mut Ctx, i32) -> i32),
            ctx_current: entry!(lib, "cuCtxSetCurrent", unsafe extern "system" fn(Ctx) -> i32),
            module_load: entry!(lib, "cuModuleLoadData", unsafe extern "system" fn(*mut Module, *const c_void) -> i32),
            module_func: entry!(lib, "cuModuleGetFunction", unsafe extern "system" fn(*mut Func, Module, *const u8) -> i32),
            alloc: entry!(lib, "cuMemAlloc_v2", unsafe extern "system" fn(*mut u64, usize) -> i32),
            free: entry!(lib, "cuMemFree_v2", unsafe extern "system" fn(u64) -> i32),
            htod: entry!(lib, "cuMemcpyHtoD_v2", unsafe extern "system" fn(u64, *const c_void, usize) -> i32),
            dtoh: entry!(lib, "cuMemcpyDtoH_v2", unsafe extern "system" fn(*mut c_void, u64, usize) -> i32),
            dtod: entry!(lib, "cuMemcpyDtoD_v2", unsafe extern "system" fn(u64, u64, usize) -> i32),
            memset: entry!(lib, "cuMemsetD8_v2", unsafe extern "system" fn(u64, u8, usize) -> i32),
            launch: entry!(lib, "cuLaunchKernel", Launch),
            sync: entry!(lib, "cuCtxSynchronize", unsafe extern "system" fn() -> i32),
            mem_info: entry!(lib, "cuMemGetInfo_v2", unsafe extern "system" fn(*mut usize, *mut usize) -> i32),
            stream_create: entry!(lib, "cuStreamCreate", unsafe extern "system" fn(*mut Stream, u32) -> i32),
            stream_sync: entry!(lib, "cuStreamSynchronize", unsafe extern "system" fn(Stream) -> i32),
            stream_query: entry!(lib, "cuStreamQuery", unsafe extern "system" fn(Stream) -> i32),
            stream_destroy: entry!(lib, "cuStreamDestroy_v2", unsafe extern "system" fn(Stream) -> i32),
            host_alloc: entry!(lib, "cuMemAllocHost_v2", unsafe extern "system" fn(*mut *mut c_void, usize) -> i32),
            host_free: entry!(lib, "cuMemFreeHost", unsafe extern "system" fn(*mut c_void) -> i32),
            htod_async: entry!(lib, "cuMemcpyHtoDAsync_v2", unsafe extern "system" fn(u64, *const c_void, usize, Stream) -> i32),
        })
    }
}

fn ok(code: i32, what: &str) -> Result<(), String> {
    if code == 0 { Ok(()) } else { Err(format!("nmffi.cuda: {what} failed with CUDA error {code}")) }
}

/// The device as the driver reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuInfo {
    pub name: String,
    pub total_bytes: usize,
    pub major: i32,
    pub minor: i32,
}

struct Inner {
    api: Api,
    ctx: Ctx,
    functions: BTreeMap<String, Func>,
    lock: Mutex<()>,
    info: GpuInfo,
    pool: Mutex<Pool>,
    reclaimer: Mutex<Option<Reclaimer>>,
    _lib: Library,
}

/// Called on out of memory with the bytes wanted; returns the bytes it gave back (step P2).
pub type Reclaimer = Arc<dyn Fn(usize) -> usize + Send + Sync>;

/// The buffer pool of one device (step P2): idle working buffers by capacity, and counters.
#[derive(Debug, Default)]
struct Pool {
    off: bool,
    idle: BTreeMap<usize, Vec<u64>>,
    /// Idle buffers in the order they came back (oldest first), for eviction.
    returned: std::collections::VecDeque<(u64, usize)>,
    idle_bytes: usize,
    idle_limit: usize,
    live_work: usize,
    live_kept: usize,
    peak_work: usize,
    hits: u64,
    misses: u64,
    device_allocs: u64,
    device_frees: u64,
    oom_retries: u64,
}
impl Pool {
    /// The smallest idle buffer of at least `n` bytes and less than twice that, taken out.
    fn take(&mut self, n: usize) -> Option<(u64, usize)> {
        let cap = *self.idle.range(n..n.saturating_mul(2)).find(|(_, v)| !v.is_empty())?.0;
        let list = self.idle.get_mut(&cap)?;
        let ptr = list.pop()?;
        if list.is_empty() {
            self.idle.remove(&cap);
        }
        self.idle_bytes = self.idle_bytes.saturating_sub(cap);
        if let Some(i) = self.returned.iter().position(|x| x.0 == ptr) {
            self.returned.remove(i);
        }
        Some((ptr, cap))
    }
    /// Keep a returned buffer idle, first taking out the longest-idle buffers until it fits
    /// under the limit (least recently returned first). Returns the buffers to free: the taken
    /// ones, or the returned one itself when it is larger than the whole limit.
    fn give_back(&mut self, ptr: u64, cap: usize) -> Vec<u64> {
        if cap > self.idle_limit {
            return vec![ptr];
        }
        let mut out = Vec::new();
        while self.idle_bytes + cap > self.idle_limit {
            let Some((old, c)) = self.returned.pop_front() else { break };
            if let Some(list) = self.idle.get_mut(&c) {
                list.retain(|x| *x != old);
                if list.is_empty() {
                    self.idle.remove(&c);
                }
            }
            self.idle_bytes = self.idle_bytes.saturating_sub(c);
            out.push(old);
        }
        self.idle.entry(cap).or_default().push(ptr);
        self.returned.push_back((ptr, cap));
        self.idle_bytes += cap;
        out
    }
}

/// What the pool holds and did (step P2), for reports.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    pub on: bool,
    pub idle_bytes: usize,
    pub idle_buffers: usize,
    pub idle_limit: usize,
    /// Bytes of working buffers handed out and not yet returned (capacities).
    pub live_work: usize,
    /// Bytes of long-lived buffers (weights, window copies).
    pub live_kept: usize,
    /// Highest `live_work` since the last `reset_peak`.
    pub peak_work: usize,
    pub hits: u64,
    pub misses: u64,
    pub device_allocs: u64,
    pub device_frees: u64,
    pub oom_retries: u64,
}

thread_local! {
    /// Open `KeptScope`s of this thread.
    static KEPT: Cell<u32> = const { Cell::new(0) };
}

/// While alive, buffers this thread allocates are long-lived (weights, window copies): exact
/// size, never pooled, counted apart from working buffers (step P2).
#[derive(Debug)]
pub struct KeptScope(());
impl KeptScope {
    #[must_use]
    pub fn new() -> Self {
        KEPT.with(|k| k.set(k.get().saturating_add(1)));
        Self(())
    }
}
impl Default for KeptScope {
    fn default() -> Self {
        Self::new()
    }
}
impl Drop for KeptScope {
    fn drop(&mut self) {
        KEPT.with(|k| k.set(k.get().saturating_sub(1)));
    }
}

/// Capacity of a new working buffer for a request of `n` bytes: rounded up to its size class,
/// a multiple of a quarter of the largest power of two not above `n` (at least 256 bytes), so
/// at most a quarter is added and requests of nearby sizes get the same capacities.
#[must_use]
pub fn look_ahead(n: usize) -> usize {
    let n = n.max(1);
    let base = 1_usize << (usize::BITS - 1 - n.leading_zeros());
    let step = (base / 4).max(256);
    n.div_ceil(step).saturating_mul(step)
}

/// Roadmap Phase 4.6 (2026-11-25): the oldest compute capability the embedded kernels run on
/// (sm_75 SASS and compute_75 PTX are the oldest targets in the fatbins).
pub const MIN_CAPABILITY: (i32, i32) = (7, 5);
/// Roadmap Phase 4.6 (2026-11-25): the phrase every "there is no card this program can use"
/// error carries (no driver, the processor path forced, no card, a card too old), so the section
/// checks pass on such a machine as they do without the driver.
pub const NO_USABLE: &str = "no usable CUDA card";
/// `CUDA_ERROR_NO_DEVICE`.
const NO_DEVICE: i32 = 100;

/// Roadmap Phase 4.6 (2026-11-25): the switch that forces the processor path for testing.
/// `NANO_GPU=cpu` (also `off`, `none`, `processor`) makes every device open report the driver
/// missing, without loading it; `NANO_GPU=<ordinal>` picks a card (`nmengn::device`).
#[must_use]
pub fn forced_processor() -> Option<String> {
    let v = std::env::var("NANO_GPU").ok()?;
    matches!(v.trim().to_ascii_lowercase().as_str(), "cpu" | "off" | "none" | "processor")
        .then(|| format!("nmffi.dynlib: nvcuda.dll not found (treated as missing: NANO_GPU={} forces the processor path); {NO_USABLE}", v.trim()))
}

/// `cuInit`, with a driver that sees no card reported in plain words.
fn init(api: &Api) -> Result<(), String> {
    // SAFETY: cuInit with flags 0 is the documented first call; it is idempotent.
    match unsafe { (api.init)(0) } {
        NO_DEVICE => Err(format!("nmffi.cuda: the driver is installed but reports no CUDA card (CUDA error {NO_DEVICE}); {NO_USABLE}")),
        code => ok(code, "cuInit"),
    }
}

/// Roadmap Phase 4.6 (2026-11-25): the cards the driver shows and its version, read without
/// taking a context or loading a kernel, so a device can be chosen before one is opened.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Devices {
    /// `cuDriverGetVersion` (for example 13000 for CUDA 13.0); `None` when it cannot be read.
    pub driver: Option<i32>,
    /// Every card by ordinal, in the order the driver numbers them (after `CUDA_VISIBLE_DEVICES`).
    pub cards: Vec<(i32, GpuInfo)>,
}

/// Roadmap Phase 4.6 (2026-11-25): lists the cards (see `Devices`).
/// # Errors
/// The processor path is forced, the driver is missing or too old for the entry points this
/// crate calls, or it reports no card.
pub fn devices() -> Result<Devices, String> {
    if let Some(why) = forced_processor() {
        return Err(why);
    }
    let lib = Library::open("nvcuda.dll")?;
    let api = Api::load(&lib)?;
    init(&api)?;
    let driver = lib.symbol("cuDriverGetVersion").ok().and_then(|p| {
        // SAFETY: the symbol is the documented `CUresult cuDriverGetVersion(int*)` entry point.
        let f = unsafe { std::mem::transmute::<*mut c_void, unsafe extern "system" fn(*mut i32) -> i32>(p) };
        let mut v = 0_i32;
        // SAFETY: valid out pointer for the duration of the call.
        (unsafe { f(&raw mut v) } == 0).then_some(v)
    });
    let mut cards = Vec::new();
    for ordinal in 0..64 {
        let mut dev = 0_i32;
        // SAFETY: valid out pointer; an ordinal past the last card returns an error code.
        if unsafe { (api.device_get)(&raw mut dev, ordinal) } != 0 {
            break;
        }
        let mut name = [0_u8; 128];
        // SAFETY: the buffer holds 128 bytes and the length passed is 128.
        ok(unsafe { (api.device_name)(name.as_mut_ptr(), 128, dev) }, "cuDeviceGetName")?;
        let mut total = 0_usize;
        // SAFETY: valid out pointer.
        ok(unsafe { (api.device_mem)(&raw mut total, dev) }, "cuDeviceTotalMem")?;
        let (mut major, mut minor) = (0_i32, 0_i32);
        // SAFETY: valid out pointers; 75 and 76 are the compute capability attributes.
        ok(unsafe { (api.device_attr)(&raw mut major, 75, dev) }, "cuDeviceGetAttribute major")?;
        // SAFETY: as above.
        ok(unsafe { (api.device_attr)(&raw mut minor, 76, dev) }, "cuDeviceGetAttribute minor")?;
        let end = name.iter().position(|b| *b == 0).unwrap_or(name.len());
        let info = GpuInfo { name: String::from_utf8_lossy(name.get(..end).unwrap_or(&[])).into_owned(), total_bytes: total, major, minor };
        cards.push((ordinal, info));
    }
    Ok(Devices { driver, cards })
}

/// Device states by ordinal, shared by every `Gpu::open` in the process (step P2).
static DEVICES: Mutex<BTreeMap<i32, Weak<Inner>>> = Mutex::new(BTreeMap::new());
// SAFETY: the context and function handles are opaque driver objects valid in every thread
// once the context is made current, which every use does under `lock`; they are released
// only at process exit (primary context, modules never unloaded).
unsafe impl Send for Inner {}
// SAFETY: all access to the handles goes through `lock` (see `Inner::enter`).
unsafe impl Sync for Inner {}

impl Inner {
    /// Free device buffers that no one owns any more (idle pool buffers taken out), after one
    /// wait for queued work (step P2).
    fn free_now(&self, ptrs: &[u64]) {
        if let Ok(_g) = self.enter() {
            let w = crate::prof::Span::new("cuda.free.wait");
            // SAFETY: the context is current; wait so no queued kernel still reads a buffer.
            let _ = unsafe { (self.api.sync)() };
            drop(w);
            let _f = crate::prof::Span::new("cuda.free");
            for ptr in ptrs {
                // SAFETY: each pointer came from cuMemAlloc on this context, sat idle in the
                // pool (owned by nothing else) and is freed exactly once here.
                let _ = unsafe { (self.api.free)(*ptr) };
            }
        }
    }
    /// Take the device lock and make the context current on this thread.
    fn enter(&self) -> Result<std::sync::MutexGuard<'_, ()>, String> {
        let g = self.lock.lock().map_err(|_| "nmffi.cuda: device lock poisoned".to_owned())?;
        // SAFETY: `ctx` is the retained primary context of this device, valid for the
        // process lifetime.
        let code = unsafe { (self.api.ctx_current)(self.ctx) };
        ok(code, "cuCtxSetCurrent")?;
        Ok(g)
    }
}

/// One CUDA device with the mini's kernels loaded.
#[derive(Clone)]
pub struct Gpu {
    inner: Arc<Inner>,
}
impl std::fmt::Debug for Gpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gpu").field("info", &self.inner.info).finish_non_exhaustive()
    }
}

/// A device buffer; freed when dropped (a working buffer goes back to the pool, step P2).
pub struct DevBuf {
    ptr: u64,
    bytes: usize,
    /// Bytes allocated (at least `bytes`; equal for long-lived buffers and with the pool off).
    cap: usize,
    /// Long-lived (allocated inside a `KeptScope`).
    kept: bool,
    /// Allocated for the pool (returns to it when dropped, while under the idle limit).
    pooled: bool,
    inner: Arc<Inner>,
}
impl std::fmt::Debug for DevBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DevBuf").field("bytes", &self.bytes).finish_non_exhaustive()
    }
}
impl DevBuf {
    /// Device address (for kernel arguments).
    #[must_use]
    pub fn ptr(&self) -> u64 {
        self.ptr
    }
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}
impl Drop for DevBuf {
    fn drop(&mut self) {
        // Step P2: a working buffer goes back to the pool while it is under its idle limit; no
        // device call and no wait (reuse is ordered on the one stream).
        if let Ok(mut p) = self.inner.pool.lock() {
            if self.kept {
                p.live_kept = p.live_kept.saturating_sub(self.cap);
            } else {
                p.live_work = p.live_work.saturating_sub(self.cap);
            }
            if self.pooled && !p.off {
                // Kept idle; the longest-idle buffers make room under the limit (step P2).
                let evict = p.give_back(self.ptr, self.cap);
                p.device_frees += evict.len() as u64;
                drop(p);
                if !evict.is_empty() {
                    self.inner.free_now(&evict);
                }
                return;
            }
            p.device_frees += 1;
        }
        if let Ok(_g) = self.inner.enter() {
            // Stage timing (step P1, off unless a part test sets it): the wait for queued
            // work and the free itself are timed apart.
            let w = crate::prof::Span::new("cuda.free.wait");
            // SAFETY: `ptr` came from cuMemAlloc on this context and is freed exactly once
            // (here); the default stream is synchronised first so no queued kernel still
            // reads it.
            let _ = unsafe { (self.inner.api.sync)() };
            drop(w);
            let _f = crate::prof::Span::new("cuda.free");
            // SAFETY: as above; the allocation is not used after this call.
            let _ = unsafe { (self.inner.api.free)(self.ptr) };
        }
    }
}

/// `emma_tensor` (cbridg `emma_cbridg.h`): `data` is the device address of element 0.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TensorArg {
    pub data: u64,
    pub byte_offset: u64,
    pub shape: [i64; 4],
    pub rank: i32,
    pub dtype: i32,
    pub owner: i32,
    pub reserved: u32,
}
/// `emma_attrs`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Attrs {
    pub pos: u64,
    pub eps: f32,
    pub theta: f32,
    pub causal: u32,
    pub group_size: u32,
    pub scale: f32,
    pub reserved: u32,
}
/// `emma_op_args`, passed by value to every kernel.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OpArgs {
    pub inputs: [TensorArg; 4],
    pub weights: [TensorArg; 2],
    pub outputs: [TensorArg; 2],
    pub attrs: Attrs,
    pub input_count: u32,
    pub weight_count: u32,
    pub output_count: u32,
    pub reserved: u32,
}
/// cukern `Geometry` (attention.cuh), passed by value.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Geometry {
    pub tokens: u64,
    pub cached: u64,
    pub heads: u64,
    pub kv_heads: u64,
    pub width: u64,
    pub clones: u64,
}
const _: () =
    assert!(size_of::<TensorArg>() == 64 && size_of::<Attrs>() == 32 && size_of::<OpArgs>() == 560 && size_of::<Geometry>() == 48);

/// One kernel argument.
#[derive(Clone, Debug)]
pub enum Param {
    I32(i32),
    Op(Box<OpArgs>),
    Geo(Geometry),
    Ptr(u64),
}
/// Grid or block dimensions.
pub type Dims = (u32, u32, u32);

impl Gpu {
    /// Open device `ordinal` and load the mini's kernels.
    /// # Errors
    /// No driver, no device, or a kernel image the driver refuses.
    pub fn open(ordinal: i32) -> Result<Self, String> {
        if let Some(why) = forced_processor() {
            return Err(why);
        }
        // Step P2: one device state per ordinal in the process (one pool, one lock, one set of
        // loaded kernels); later opens share it.
        let mut devices = DEVICES.lock().map_err(|_| "nmffi.cuda: device table poisoned".to_owned())?;
        if let Some(inner) = devices.get(&ordinal).and_then(Weak::upgrade) {
            return Ok(Self { inner });
        }
        let g = Self::open_new(ordinal)?;
        devices.insert(ordinal, Arc::downgrade(&g.inner));
        Ok(g)
    }
    fn open_new(ordinal: i32) -> Result<Self, String> {
        let lib = Library::open("nvcuda.dll")?;
        let api = Api::load(&lib)?;
        init(&api)?;
        let mut dev = 0_i32;
        // SAFETY: `dev` is a valid out pointer for the duration of the call.
        ok(unsafe { (api.device_get)(&raw mut dev, ordinal) }, "cuDeviceGet")?;
        let mut name = [0_u8; 128];
        // SAFETY: the buffer holds 128 bytes and the length passed is 128.
        ok(unsafe { (api.device_name)(name.as_mut_ptr(), 128, dev) }, "cuDeviceGetName")?;
        let mut total = 0_usize;
        // SAFETY: valid out pointer.
        ok(unsafe { (api.device_mem)(&raw mut total, dev) }, "cuDeviceTotalMem")?;
        let (mut major, mut minor) = (0_i32, 0_i32);
        // SAFETY: valid out pointer; 75 is CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR.
        ok(unsafe { (api.device_attr)(&raw mut major, 75, dev) }, "cuDeviceGetAttribute major")?;
        // SAFETY: valid out pointer; 76 is CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR.
        ok(unsafe { (api.device_attr)(&raw mut minor, 76, dev) }, "cuDeviceGetAttribute minor")?;
        // Roadmap Phase 4.6 (2026-11-25): a card older than the kernels (compute capability under
        // 7.5, Pascal and earlier) is refused here with a plain reason, before a context is taken
        // or a kernel image offered, so the caller falls back to the processor cleanly.
        if (major, minor) < MIN_CAPABILITY {
            let end = name.iter().position(|b| *b == 0).unwrap_or(name.len());
            let shown = String::from_utf8_lossy(name.get(..end).unwrap_or(&[])).into_owned();
            return Err(format!(
                "nmffi.cuda: {shown} is compute capability {major}.{minor}, older than the kernels support ({}.{} or newer); {NO_USABLE}",
                MIN_CAPABILITY.0, MIN_CAPABILITY.1
            ));
        }
        let mut ctx: Ctx = std::ptr::null_mut();
        // SAFETY: valid out pointer; the primary context is retained for the process.
        ok(unsafe { (api.ctx_retain)(&raw mut ctx, dev) }, "cuDevicePrimaryCtxRetain")?;
        // SAFETY: `ctx` was just retained.
        ok(unsafe { (api.ctx_current)(ctx) }, "cuCtxSetCurrent")?;
        let mut functions = BTreeMap::new();
        for (label, image, names) in IMAGES {
            // The driver reads the image as 8-byte-aligned words; copy it into u64 storage.
            let mut words = vec![0_u64; image.len().div_ceil(8)];
            for (i, b) in image.iter().enumerate() {
                let w = words.get_mut(i / 8).ok_or("nmffi.cuda: image copy")?;
                *w |= u64::from(*b) << ((i % 8) * 8);
            }
            let mut module: Module = std::ptr::null_mut();
            // SAFETY: `words` holds the complete fatbin, aligned to 8 bytes, alive for the call;
            // the driver copies what it needs.
            ok(unsafe { (api.module_load)(&raw mut module, words.as_ptr().cast()) }, &format!("cuModuleLoadData {label}"))?;
            for n in *names {
                let c: Vec<u8> = n.bytes().chain(std::iter::once(0)).collect();
                let mut f: Func = std::ptr::null_mut();
                // SAFETY: `module` was loaded above; `c` is NUL-terminated and alive.
                ok(unsafe { (api.module_func)(&raw mut f, module, c.as_ptr()) }, &format!("cuModuleGetFunction {n}"))?;
                functions.insert((*n).to_owned(), f);
            }
        }
        let end = name.iter().position(|b| *b == 0).unwrap_or(name.len());
        let info = GpuInfo { name: String::from_utf8_lossy(name.get(..end).unwrap_or(&[])).into_owned(), total_bytes: total, major, minor };
        // Idle limit before the residency manager measures a run: an eighth of the device
        // (stated fallback; replaced at run time through `set_idle_limit`).
        let pool = Pool { idle_limit: total / 8, ..Pool::default() };
        Ok(Self {
            inner: Arc::new(Inner {
                api,
                ctx,
                functions,
                lock: Mutex::new(()),
                info,
                pool: Mutex::new(pool),
                reclaimer: Mutex::new(None),
                _lib: lib,
            }),
        })
    }
    #[must_use]
    pub fn info(&self) -> &GpuInfo {
        &self.inner.info
    }
    /// Free and total device memory in bytes.
    /// # Errors
    /// A driver fault.
    pub fn memory(&self) -> Result<(usize, usize), String> {
        let _g = self.inner.enter()?;
        let (mut free, mut total) = (0_usize, 0_usize);
        // SAFETY: valid out pointers; the context is current.
        ok(unsafe { (self.inner.api.mem_info)(&raw mut free, &raw mut total) }, "cuMemGetInfo")?;
        Ok((free, total))
    }
    /// Allocate `bytes` (at least 1) of device memory, zeroed.
    /// # Errors
    /// Out of device memory.
    pub fn alloc(&self, bytes: usize) -> Result<DevBuf, String> {
        let n = bytes.max(1);
        let _t = crate::prof::Span::new("cuda.alloc");
        let kept = KEPT.with(|k| k.get() > 0);
        // Step P2: a working buffer is taken from the pool when one fits (best fit, at most a
        // quarter larger than asked); else a new one with look-ahead capacity.
        let (off, reuse) = {
            let mut p = self.inner.pool.lock().map_err(|_| "nmffi.cuda: pool poisoned".to_owned())?;
            let reuse = if kept || p.off { None } else { p.take(n) };
            if !kept && !p.off {
                if reuse.is_some() {
                    p.hits += 1;
                } else {
                    p.misses += 1;
                }
            }
            (p.off, reuse)
        };
        let pooled = !kept && !off;
        let (ptr, cap) = match reuse {
            Some(x) => x,
            None => {
                let cap = if pooled { look_ahead(n) } else { n };
                (self.device_alloc(cap)?, cap)
            }
        };
        let buf = DevBuf { ptr, bytes: n, cap, kept, pooled, inner: Arc::clone(&self.inner) };
        if let Ok(mut p) = self.inner.pool.lock() {
            if kept {
                p.live_kept += cap;
            } else {
                p.live_work += cap;
                p.peak_work = p.peak_work.max(p.live_work);
            }
        }
        let _g = self.inner.enter()?;
        // SAFETY: `ptr` addresses at least `n` bytes allocated on this context (new, or a pooled
        // buffer of capacity `cap >= n` that no queued work still reads: every earlier use was
        // issued before it on the same stream).
        ok(unsafe { (self.inner.api.memset)(ptr, 0, n) }, "cuMemsetD8")?;
        Ok(buf)
    }
    /// `cuMemAlloc` of `cap` bytes; on out of memory the pool's idle buffers are freed, the
    /// reclaimer is asked for the rest, and the allocation is tried once more (step P2).
    fn device_alloc(&self, cap: usize) -> Result<u64, String> {
        let raw = || -> Result<(i32, u64), String> {
            let _g = self.inner.enter()?;
            let mut ptr = 0_u64;
            // SAFETY: valid out pointer; the context is current.
            let code = unsafe { (self.inner.api.alloc)(&raw mut ptr, cap) };
            Ok((code, ptr))
        };
        let (mut code, mut ptr) = raw()?;
        if code == OUT_OF_MEMORY {
            if let Ok(mut p) = self.inner.pool.lock() {
                p.oom_retries += 1;
            }
            self.trim();
            let hook = self.inner.reclaimer.lock().ok().and_then(|r| r.clone());
            if let Some(h) = hook {
                let _ = h(cap);
            }
            (code, ptr) = raw()?;
        }
        ok(code, &format!("cuMemAlloc {cap} bytes"))?;
        if let Ok(mut p) = self.inner.pool.lock() {
            p.device_allocs += 1;
        }
        Ok(ptr)
    }
    /// Free every idle pooled buffer (one device wait first). Returns the bytes freed.
    pub fn trim(&self) -> usize {
        let (taken, bytes) = match self.inner.pool.lock() {
            Ok(mut p) => {
                let v: Vec<u64> = std::mem::take(&mut p.idle).into_values().flatten().collect();
                p.returned.clear();
                p.device_frees += v.len() as u64;
                (v, std::mem::replace(&mut p.idle_bytes, 0))
            }
            Err(_) => return 0,
        };
        if taken.is_empty() {
            return 0;
        }
        if let Ok(_g) = self.inner.enter() {
            // SAFETY: the context is current; wait so no queued kernel still reads a buffer.
            let _ = unsafe { (self.inner.api.sync)() };
            for ptr in &taken {
                // SAFETY: each pointer came from cuMemAlloc on this context, sat idle in the
                // pool (owned by nothing else) and is freed exactly once here.
                let _ = unsafe { (self.inner.api.free)(*ptr) };
            }
        }
        bytes
    }
    /// The pool on (default) or off. Off: every buffer is allocated at its exact size, zeroed,
    /// and freed after a device wait when dropped (the path before step P2); idle buffers are
    /// freed now.
    pub fn set_pool(&self, on: bool) {
        if let Ok(mut p) = self.inner.pool.lock() {
            p.off = !on;
        }
        if !on {
            self.trim();
        }
    }
    /// Bytes of idle buffers the pool may keep (set at run time by the residency manager).
    /// Idle buffers past a lower limit are freed now.
    pub fn set_idle_limit(&self, bytes: usize) {
        let over = self.inner.pool.lock().map(|mut p| {
            p.idle_limit = bytes;
            p.idle_bytes > bytes
        });
        if over.unwrap_or(false) {
            self.trim();
        }
    }
    /// The pool's counters.
    #[must_use]
    pub fn pool_stats(&self) -> PoolStats {
        self.inner
            .pool
            .lock()
            .map(|p| PoolStats {
                on: !p.off,
                idle_bytes: p.idle_bytes,
                idle_buffers: p.idle.values().map(Vec::len).sum(),
                idle_limit: p.idle_limit,
                live_work: p.live_work,
                live_kept: p.live_kept,
                peak_work: p.peak_work,
                hits: p.hits,
                misses: p.misses,
                device_allocs: p.device_allocs,
                device_frees: p.device_frees,
                oom_retries: p.oom_retries,
            })
            .unwrap_or_default()
    }
    /// Start a new peak of working bytes at the current live bytes; returns the live bytes.
    pub fn reset_peak(&self) -> usize {
        self.inner
            .pool
            .lock()
            .map(|mut p| {
                p.peak_work = p.live_work;
                p.live_work
            })
            .unwrap_or(0)
    }
    /// Register the function asked for memory on out of memory (the residency manager).
    pub fn set_reclaimer(&self, r: Option<Reclaimer>) {
        if let Ok(mut x) = self.inner.reclaimer.lock() {
            *x = r;
        }
    }
    /// Two handles to the same device state.
    #[must_use]
    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
    /// Copy host u16 values into a buffer at byte offset `at` (binary16 rows as they are; no
    /// host repacking, step P2).
    /// # Errors
    /// A range outside the buffer, or a driver fault.
    pub fn upload_u16_at(&self, buf: &DevBuf, at: usize, data: &[u16]) -> Result<(), String> {
        let n = data.len().saturating_mul(2);
        if at.checked_add(n).is_none_or(|e| e > buf.bytes) {
            return Err(format!("nmffi.cuda: upload of {n} bytes at {at} into a {}-byte buffer", buf.bytes));
        }
        let _t = crate::prof::Span::new("cuda.htod");
        let _g = self.inner.enter()?;
        // SAFETY: the destination range lies inside the allocation (checked above); the source
        // slice holds `n` bytes and is alive for the synchronous copy.
        ok(unsafe { (self.inner.api.htod)(buf.ptr + at as u64, data.as_ptr().cast(), n) }, "cuMemcpyHtoD")
    }
    /// Copy a buffer's first `out.len()` u16 values to the host (waits for queued work; step
    /// P2: binary16 rows land in storage layout, no conversion).
    /// # Errors
    /// More values than the buffer holds, or a driver fault (including a failed kernel).
    pub fn download_u16(&self, buf: &DevBuf, out: &mut [u16]) -> Result<(), String> {
        let n = out.len().saturating_mul(2);
        if n > buf.bytes {
            return Err(format!("nmffi.cuda: download of {n} bytes from a {}-byte buffer", buf.bytes));
        }
        let _t = crate::prof::Span::new("cuda.dtoh");
        let _g = self.inner.enter()?;
        // SAFETY: the source holds at least `n` bytes (checked above); `out` is a live,
        // exclusive host slice of `n` bytes.
        ok(unsafe { (self.inner.api.dtoh)(out.as_mut_ptr().cast(), buf.ptr, n) }, "cuMemcpyDtoH")
    }
    /// Copy host bytes into a buffer at its start.
    /// # Errors
    /// More bytes than the buffer holds, or a driver fault.
    pub fn upload(&self, buf: &DevBuf, bytes: &[u8]) -> Result<(), String> {
        if bytes.len() > buf.bytes {
            return Err(format!("nmffi.cuda: upload of {} bytes into a {}-byte buffer", bytes.len(), buf.bytes));
        }
        let _t = crate::prof::Span::new("cuda.htod");
        let _g = self.inner.enter()?;
        // SAFETY: the destination holds at least `bytes.len()` bytes (checked above) and the
        // source slice is alive for the synchronous copy.
        ok(unsafe { (self.inner.api.htod)(buf.ptr, bytes.as_ptr().cast(), bytes.len()) }, "cuMemcpyHtoD")
    }
    /// Copy a buffer's first `out.len()` bytes to the host (waits for queued work).
    /// # Errors
    /// More bytes than the buffer holds, or a driver fault (including a failed kernel).
    pub fn download(&self, buf: &DevBuf, out: &mut [u8]) -> Result<(), String> {
        if out.len() > buf.bytes {
            return Err(format!("nmffi.cuda: download of {} bytes from a {}-byte buffer", out.len(), buf.bytes));
        }
        let _t = crate::prof::Span::new("cuda.dtoh");
        let _g = self.inner.enter()?;
        // SAFETY: the source holds at least `out.len()` bytes (checked above); `out` is a
        // live, exclusive host slice of that length.
        ok(unsafe { (self.inner.api.dtoh)(out.as_mut_ptr().cast(), buf.ptr, out.len()) }, "cuMemcpyDtoH")
    }
    /// Copy `bytes` from `src` (at `src_off`) to `dst` (at `dst_off`), device to device.
    /// # Errors
    /// A range outside either buffer, or a driver fault.
    pub fn copy(&self, src: &DevBuf, src_off: usize, dst: &DevBuf, dst_off: usize, bytes: usize) -> Result<(), String> {
        if src_off.checked_add(bytes).is_none_or(|e| e > src.bytes) || dst_off.checked_add(bytes).is_none_or(|e| e > dst.bytes) {
            return Err("nmffi.cuda: device copy outside a buffer".into());
        }
        let _t = crate::prof::Span::new("cuda.dtod");
        let _g = self.inner.enter()?;
        // SAFETY: both ranges lie inside their allocations (checked above).
        ok(unsafe { (self.inner.api.dtod)(dst.ptr + dst_off as u64, src.ptr + src_off as u64, bytes) }, "cuMemcpyDtoD")
    }
    /// Launch one of the loaded kernels.
    /// # Errors
    /// An unknown kernel or a launch the driver refuses.
    pub fn launch(&self, kernel: &str, grid: Dims, block: Dims, shared: u32, params: &[Param]) -> Result<(), String> {
        let f = *self.inner.functions.get(kernel).ok_or_else(|| format!("nmffi.cuda: kernel {kernel} not loaded"))?;
        let mut store: Vec<Param> = params.to_vec();
        let mut ptrs: Vec<*mut c_void> = store
            .iter_mut()
            .map(|p| match p {
                Param::I32(v) => std::ptr::from_mut(v).cast::<c_void>(),
                Param::Op(v) => std::ptr::from_mut(v.as_mut()).cast::<c_void>(),
                Param::Geo(v) => std::ptr::from_mut(v).cast::<c_void>(),
                Param::Ptr(v) => std::ptr::from_mut(v).cast::<c_void>(),
            })
            .collect();
        let _t = crate::prof::Span::new("cuda.launch");
        let _g = self.inner.enter()?;
        let t0 = std::time::Instant::now();
        // SAFETY: `f` is a function of a loaded module; `ptrs` holds one pointer per kernel
        // parameter, in order, each to a live value of the parameter's C type (`store` lives
        // past the call); the driver copies the values at launch.
        let code = unsafe {
            (self.inner.api.launch)(
                f,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared,
                std::ptr::null_mut(),
                ptrs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        ok(code, &format!("cuLaunchKernel {kernel}"))?;
        // Per-kernel timing (profiling only): wait for this kernel and charge it by name.
        if crate::prof::kernels() {
            // SAFETY: the context is current (the guard is held).
            ok(unsafe { (self.inner.api.sync)() }, "cuCtxSynchronize")?;
            let short = kernel.split("emma").nth(1).map_or(kernel, |s| s.trim_start_matches(|c: char| c.is_ascii_digit()));
            crate::prof::add_side(&format!("kernel {}", short.split('E').next().unwrap_or(short)), t0.elapsed().as_secs_f64());
        }
        Ok(())
    }
    /// Page-locked host memory of `bytes` (the source of an asynchronous copy).
    /// # Errors
    /// The driver refuses the allocation.
    pub fn pinned(&self, bytes: usize) -> Result<Pinned, String> {
        let _g = self.inner.enter()?;
        let mut p: *mut c_void = std::ptr::null_mut();
        // SAFETY: the context is current; the driver writes one host pointer.
        ok(unsafe { (self.inner.api.host_alloc)(&raw mut p, bytes.max(1)) }, "cuMemAllocHost")?;
        Ok(Pinned { ptr: p.cast::<u8>(), bytes, inner: Arc::clone(&self.inner) })
    }
    /// A second stream for copies (non-blocking: it does not wait for the default stream the
    /// kernels run on, so a copy can overlap a running job).
    /// # Errors
    /// The driver refuses the stream.
    pub fn copy_stream(&self) -> Result<CopyStream, String> {
        let _g = self.inner.enter()?;
        let mut s: Stream = std::ptr::null_mut();
        // SAFETY: the context is current; flag 1 is CU_STREAM_NON_BLOCKING.
        ok(unsafe { (self.inner.api.stream_create)(&raw mut s, 1) }, "cuStreamCreate")?;
        Ok(CopyStream { s, inner: Arc::clone(&self.inner) })
    }
    /// Queue a copy of `src[..bytes]` (pinned) into `dst` at `at` on `stream`; returns at once.
    /// The caller keeps `src` and `dst` alive and unchanged until `stream.wait()`. The copy stream
    /// does not wait for the default stream, so `dst` must have no work pending there: allocate
    /// the destinations first (a fresh allocation is zeroed on the default stream), call `sync`
    /// once, then queue the copies (2026-11-13: without that sync about 1% of bytes came back zero).
    /// # Errors
    /// A range outside either buffer, or a driver fault.
    pub fn upload_async(&self, stream: &CopyStream, dst: &DevBuf, at: usize, src: &Pinned, bytes: usize) -> Result<(), String> {
        if bytes > src.bytes || at.checked_add(bytes).is_none_or(|e| e > dst.bytes) {
            return Err("nmffi.cuda: asynchronous copy outside a buffer".into());
        }
        let _t = crate::prof::Span::new("cuda.htod.async");
        let _g = self.inner.enter()?;
        // SAFETY: both ranges lie inside their allocations (checked above); the source is
        // page-locked memory of this context; the caller keeps both alive until the stream
        // has finished (its contract).
        ok(unsafe { (self.inner.api.htod_async)(dst.ptr + at as u64, src.ptr.cast::<c_void>(), bytes, stream.s) }, "cuMemcpyHtoDAsync")
    }
    /// Wait for every queued operation.
    /// # Errors
    /// A queued kernel failed.
    pub fn sync(&self) -> Result<(), String> {
        let _g = self.inner.enter()?;
        // SAFETY: the context is current.
        ok(unsafe { (self.inner.api.sync)() }, "cuCtxSynchronize")
    }
}

/// Page-locked host memory; freed when dropped.
pub struct Pinned {
    ptr: *mut u8,
    bytes: usize,
    inner: Arc<Inner>,
}
// SAFETY: the buffer is plain host memory owned by this value; access goes through `&mut self`.
unsafe impl Send for Pinned {}
// SAFETY: shared access only reads through `as_slice`.
unsafe impl Sync for Pinned {}
impl Pinned {
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    /// The memory as a byte slice to fill.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` points to `bytes` bytes allocated by the driver and owned here.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.bytes) }
    }
}
impl Drop for Pinned {
    fn drop(&mut self) {
        if let Ok(_g) = self.inner.enter() {
            // SAFETY: the context is current; wait for queued copies that may read this memory.
            let _ = unsafe { (self.inner.api.sync)() };
            // SAFETY: `ptr` came from cuMemAllocHost on this context and is freed exactly once here.
            let _ = unsafe { (self.inner.api.host_free)(self.ptr.cast::<c_void>()) };
        }
    }
}

/// A non-blocking stream for copies; destroyed when dropped.
pub struct CopyStream {
    s: Stream,
    inner: Arc<Inner>,
}
// SAFETY: a stream handle may be used from any thread of the process with its context current
// (every use enters the context under the device lock).
unsafe impl Send for CopyStream {}
// SAFETY: as above.
unsafe impl Sync for CopyStream {}
impl CopyStream {
    /// Wait until every copy queued on this stream has finished.
    /// # Errors
    /// A failed copy.
    pub fn wait(&self) -> Result<(), String> {
        let s = self.s;
        let api = self.inner.api;
        {
            let _g = self.inner.enter()?;
            // SAFETY: the context is current; a non-blocking query.
            let q = unsafe { (api.stream_query)(s) };
            if q == 0 {
                return Ok(());
            }
        }
        let _g = self.inner.enter()?;
        // SAFETY: the context is current; waits for this stream only.
        ok(unsafe { (api.stream_sync)(s) }, "cuStreamSynchronize")
    }
    /// Whether every queued copy has finished (no wait).
    #[must_use]
    pub fn done(&self) -> bool {
        self.inner.enter().is_ok_and(|_g| {
            // SAFETY: the context is current; a non-blocking query.
            unsafe { (self.inner.api.stream_query)(self.s) == 0 }
        })
    }
}
impl Drop for CopyStream {
    fn drop(&mut self) {
        if let Ok(_g) = self.inner.enter() {
            // SAFETY: the context is current; wait for this stream's copies.
            let _ = unsafe { (self.inner.api.stream_sync)(self.s) };
            // SAFETY: the handle came from cuStreamCreate and is destroyed exactly once here.
            let _ = unsafe { (self.inner.api.stream_destroy)(self.s) };
        }
    }
}

/// Section check of the second stream (2026-11-13): 64 MB of pinned memory copied to the card on
/// the copy stream while kernels run on the default stream; every byte arrives; the copy stream
/// reports done after its wait; a copy outside a buffer is refused.
/// # Errors
/// One fault line.
pub fn check_stream() -> Result<(), String> {
    let gpu = match Gpu::open(0) {
        Ok(g) => g,
        Err(e) if e.contains("nvcuda.dll not found") || e.contains(NO_USABLE) => return Ok(()),
        Err(e) => return Err(e),
    };
    let n = 64 << 20;
    let mut src = gpu.pinned(n)?;
    for (i, b) in src.bytes_mut().iter_mut().enumerate() {
        *b = (i.wrapping_mul(31) % 251) as u8;
    }
    let dst = {
        let _k = KeptScope::new();
        gpu.alloc(n)?
    };
    let st = gpu.copy_stream()?;
    gpu.sync()?;
    gpu.upload_async(&st, &dst, 0, &src, n)?;
    // Kernels on the default stream meanwhile.
    let (x, z) = (gpu.alloc(8)?, gpu.alloc(8)?);
    let t = |d: &DevBuf| TensorArg { data: d.ptr(), shape: [4, 0, 0, 0], rank: 1, dtype: 2, owner: 1, ..TensorArg::default() };
    for _ in 0..50 {
        let mut args = OpArgs { input_count: 2, output_count: 1, ..OpArgs::default() };
        args.inputs[0] = t(&x);
        args.inputs[1] = t(&x);
        args.outputs[0] = t(&z);
        gpu.launch("_ZN4emma13binary_kernelEi12emma_op_args", (1, 1, 1), (128, 1, 1), 0, &[Param::I32(5), Param::Op(Box::new(args))])?;
    }
    st.wait()?;
    if !st.done() {
        return Err("nmffi.cuda: copy stream not done after its wait".into());
    }
    let mut back = vec![0_u8; n];
    gpu.download(&dst, &mut back)?;
    if let Some((i, b)) = back.iter().enumerate().find(|(i, b)| **b != (i.wrapping_mul(31) % 251) as u8) {
        let wrong = back.iter().enumerate().filter(|(i, b)| **b != (i.wrapping_mul(31) % 251) as u8).count();
        return Err(format!(
            "nmffi.cuda: the asynchronous copy changed bytes: first at {i} ({b}, want {}), {wrong} of {n} wrong",
            (i.wrapping_mul(31) % 251) as u8
        ));
    }
    if gpu.upload_async(&st, &dst, 1, &src, n).is_ok() {
        return Err("nmffi.cuda: a copy outside the buffer was queued".into());
    }
    Ok(())
}

/// Self-check of the `cuda` FFI module: on a PC with a CUDA device, open it, load the
/// kernels, add two float16 vectors with `binary_kernel` and read the exact sum back. On a
/// PC without the driver the check reports that and passes (the CPU path stands alone).
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let gpu = match Gpu::open(0) {
        Ok(g) => g,
        Err(e) if e.contains("nvcuda.dll not found") || e.contains(NO_USABLE) => return Ok(()),
        Err(e) => return Err(e),
    };
    let half = |v: f32| -> [u8; 2] {
        // Small integers are exact in binary16: sign 0, exponent and mantissa by hand.
        let bits: u16 = match v as i32 {
            1 => 0x3c00,
            2 => 0x4000,
            3 => 0x4200,
            _ => 0,
        };
        bits.to_le_bytes()
    };
    let a: Vec<u8> = [1.0, 2.0, 1.0, 2.0].iter().flat_map(|v| half(*v)).collect();
    let b: Vec<u8> = [2.0, 1.0, 2.0, 1.0].iter().flat_map(|v| half(*v)).collect();
    let (x, y, z) = (gpu.alloc(8)?, gpu.alloc(8)?, gpu.alloc(8)?);
    gpu.upload(&x, &a)?;
    gpu.upload(&y, &b)?;
    let t = |d: &DevBuf| TensorArg { data: d.ptr(), shape: [4, 0, 0, 0], rank: 1, dtype: 2, owner: 1, ..TensorArg::default() };
    let mut args = OpArgs { input_count: 2, output_count: 1, ..OpArgs::default() };
    args.inputs[0] = t(&x);
    args.inputs[1] = t(&y);
    args.outputs[0] = t(&z);
    gpu.launch("_ZN4emma13binary_kernelEi12emma_op_args", (1, 1, 1), (128, 1, 1), 0, &[Param::I32(5), Param::Op(Box::new(args))])?;
    let mut out = [0_u8; 8];
    gpu.download(&z, &mut out)?;
    if out.chunks(2).any(|c| c != half(3.0)) {
        return Err(format!("nmffi.cuda: 1+2 on {} gave {out:?}", gpu.info().name));
    }
    drop((x, y, z));
    pool_check(&gpu)
}

/// Pool section of the check (step P2): a second open shares the device state; a dropped
/// working buffer is reused for a fitting request and comes back zeroed; a request of less than half
/// does not take a large idle buffer; a kept buffer is exact-size and never pooled; u16 copies
/// keep every value bit for bit; with the pool off nothing is reused.
fn pool_check(gpu: &Gpu) -> Result<(), String> {
    let again = Gpu::open(0)?;
    if !gpu.same(&again) {
        return Err("nmffi.cuda: a second open did not share the device state".into());
    }
    let s0 = gpu.pool_stats();
    let a = gpu.alloc(10_000)?;
    let vals: Vec<u16> = (0..5_000_u16).map(|i| i.wrapping_mul(13).wrapping_add(7)).collect();
    gpu.upload_u16_at(&a, 0, &vals)?;
    let mut back = vec![0_u16; 5_000];
    gpu.download_u16(&a, &mut back)?;
    if back != vals {
        return Err("nmffi.cuda: u16 copies changed values".into());
    }
    let (ptr, cap) = (a.ptr(), a.cap);
    drop(a);
    let b = gpu.alloc(9_900)?;
    let mut z = vec![1_u16; 4_950];
    gpu.download_u16(&b, &mut z)?;
    let s1 = gpu.pool_stats();
    if b.ptr() != ptr || b.cap != cap || z.iter().any(|v| *v != 0) || s1.hits != s0.hits + 1 {
        return Err(format!(
            "nmffi.cuda: pool reuse (same buffer {}, zeroed {}, hits {} -> {})",
            b.ptr() == ptr,
            z.iter().all(|v| *v == 0),
            s0.hits,
            s1.hits
        ));
    }
    drop(b);
    let small = gpu.alloc(100)?;
    if small.ptr() == ptr {
        return Err("nmffi.cuda: a small request took a large idle buffer".into());
    }
    let kept = {
        let _k = KeptScope::new();
        gpu.alloc(10_000)?
    };
    if kept.cap != 10_000 || kept.pooled || !kept.kept {
        return Err("nmffi.cuda: a kept buffer is not exact-size and unpooled".into());
    }
    drop((small, kept));
    // Over the idle limit the longest-idle buffer is freed, not the one coming back.
    let limit = gpu.pool_stats().idle_limit;
    gpu.trim();
    gpu.set_idle_limit(25_000);
    let (x1, x2, x3) = (gpu.alloc(10_000)?, gpu.alloc(10_000)?, gpu.alloc(10_000)?);
    let (p1, p3) = (x1.ptr(), x3.ptr());
    let f0 = gpu.pool_stats().device_frees;
    drop(x1);
    drop(x2);
    drop(x3);
    let s = gpu.pool_stats();
    let again = gpu.alloc(10_000)?;
    let lru_ok = s.device_frees == f0 + 1 && s.idle_buffers == 2 && again.ptr() == p3 && again.ptr() != p1;
    drop(again);
    gpu.set_idle_limit(limit);
    if !lru_ok {
        return Err(format!("nmffi.cuda: idle eviction (frees {} -> {}, idle {})", f0, s.device_frees, s.idle_buffers));
    }
    gpu.set_pool(false);
    let c = gpu.alloc(10_000)?;
    let off_ok = c.cap == 10_000 && !c.pooled && gpu.pool_stats().idle_buffers == 0;
    drop(c);
    gpu.set_pool(true);
    if !off_ok {
        return Err("nmffi.cuda: pool off still pools".into());
    }
    if look_ahead(1) != 256 || look_ahead(1_000_000) != 1_048_576 || look_ahead(10_000) != 10_240 || look_ahead(1_310_720) != 1_310_720 {
        return Err("nmffi.cuda: look-ahead rule".into());
    }
    Ok(())
}
