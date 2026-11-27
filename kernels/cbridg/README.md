# kernels/cbridg: kernel bridge headers

`include/` and `src/internal.hpp`: the C interface the kernels are written against. Kept for kernel
rebuilds; the Rust side calls the kernels through `nmffi::cuda`.
