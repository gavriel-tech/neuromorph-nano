# kernels: GPU kernels

- `cukern/`: CUDA sources of the kernels (`*.cu`, `*.cuh`).
- `build/`: the compiled `*.fatbin` files, embedded into `nmffi` at build time (hashes in
  `INTEGRITY.md`).
- `cbridg/`: C headers of the kernel bridge.
- `build-fatbins.cmd`: rebuilds `build/` from `cukern/` (needs the CUDA toolkit and the Visual
  Studio C++ tools; only when a kernel changes).
