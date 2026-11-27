# nmffi

The only crate with `unsafe` code. `dynlib` (loads a system library through kernel32) and `cuda`
(the CUDA driver API from `nvcuda.dll`, the embedded fatbins from `kernels/build/`, buffers, a
buffer pool, launches). Every `unsafe` block carries a SAFETY comment; everything it exports is
safe to call. Depends on nothing.

Section checks: `neuromorph-nano check nmffi`.
