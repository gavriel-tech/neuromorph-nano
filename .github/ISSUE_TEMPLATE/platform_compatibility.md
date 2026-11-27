---
name: Platform & hardware compatibility
about: Propose or coordinate Linux, macOS / Apple Silicon, AMD/Vulkan, or GPU compatibility work
title: "[Compat] "
labels: compatibility
---

## Target Platform / Hardware
- **OS & Architecture:** (e.g., Linux x86_64, macOS Apple Silicon `aarch64-apple-darwin`, Windows ARM64)
- **Compute Backend:** (e.g., Linux CUDA `libcuda.so.1`, Apple Metal, Vulkan, ROCm, CPU NEON/AVX2)

## Architectural Constraints
All compatibility contributions must preserve Neuromorph Nano's core engineering invariants (see `CONTRIBUTING.md`):
- **Zero external Rust crates:** `Cargo.lock` stays `std` + workspace crates only.
- **Strict `unsafe` isolation:** All OS FFI (`dlopen`/`dlsym`, memory queries, GPU driver bindings) lives strictly inside `crates/nmffi/` with `// SAFETY:` comments.
- **Zero Windows regressions:** Existing Windows paths remain intact inside `#[cfg(windows)]`.
- **Clean verification:** `cargo clippy --workspace --all-targets -- -D warnings` passes with 0 warnings.

## Proposal & Verification Plan
Describe the proposed changes, target files, and how you plan to test parity and performance on real hardware.
