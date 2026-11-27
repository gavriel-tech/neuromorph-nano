## Summary
Briefly describe what this pull request changes and why.

## Type of Change
- [ ] Bug fix
- [ ] Platform / hardware compatibility (Linux, macOS, GPU backend)
- [ ] Performance / memory optimization
- [ ] Cartridge, parameter map, or learning improvement
- [ ] UI / MCP / documentation

## Engineering Checklist
- [ ] Read [CONTRIBUTING.md](CONTRIBUTING.md) and [CONTRIBUTOR_POOL.md](CONTRIBUTOR_POOL.md)
- [ ] Zero third-party Rust crates added (`std` + workspace crates only)
- [ ] Zero `unsafe` outside `crates/nmffi/` (all `nmffi` `unsafe` blocks carry `// SAFETY:` comments)
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes with 0 warnings
- [ ] Relevant tests, section checks (`neuromorph-nano check`), and UI/API checks pass
- [ ] Contributions are licensed under the terms in `CONTRIBUTING.md` (Contributor Licensing)
