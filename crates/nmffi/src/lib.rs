//! nmffi: the bottom crate that holds every `unsafe` line of the mini (PLAN section 3; ruling,
//! 23:40 Finnish time, 2026-11-07). Two FFI modules: `dynlib` (load a system library and
//! find a symbol, through kernel32) and `cuda` (the CUDA driver API from `nvcuda.dll`: one
//! context, fatbin modules of the copied cukern kernels, device buffers, kernel launches).
//! Everything this crate exports is safe to call; no other crate may contain `unsafe`, and
//! `nmengn::backend` reaches the GPU only through `cuda::Gpu`. Depends on nothing.
#[allow(unsafe_code, reason = "FFI module: CUDA driver API calls and kernel launches")]
pub mod cuda;
#[allow(unsafe_code, reason = "FFI module: kernel32 LoadLibraryW and GetProcAddress")]
pub mod dynlib;
#[allow(unsafe_code, reason = "FFI module: Windows' own HTTP service (winhttp.dll) for downloads and listings")]
pub mod http;
/// Stage timing for part tests (2026-11-10, step P1); off unless a test switches it on.
pub mod prof;

pub use cuda::{DevBuf, Dims, Geometry, Gpu, GpuInfo, KeptScope, OpArgs, Param, PoolStats, Reclaimer, TensorArg};

/// One section check: (sub-component id, check returning ok or one fault line).
pub type Check = (&'static str, fn() -> Result<(), String>);

/// Section checks of this crate (each FFI module has its own).
#[must_use]
pub fn checks() -> Vec<Check> {
    vec![
        ("nmffi.dynlib", dynlib::check),
        ("nmffi.cuda", cuda::check),
        ("nmffi.cuda.stream", cuda::check_stream),
        ("nmffi.prof", prof::check),
    ]
}
