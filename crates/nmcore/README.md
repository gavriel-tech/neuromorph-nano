# nmcore

The product binary `neuromorph-nano` (`src/bin/main.rs`) and a small library (`src/lib.rs`).

- `src/bin/main.rs`: command dispatch (`serve`, `call`, `setup`, `launch`, `ui`, `trace`,
  `check`, `seal`, `selftest engine`, `hook stop`) and the section-check registry.
- `src/bin/controls.rs`, `project.rs`, `launch.rs`, `ui.rs`, `trace.rs`: saved controls and the
  audit log, per-project memory and setup, the launcher copy, the local interface server, and the
  trace printer.
- `src/selftest.rs` (library): the engine self-test against `tests/engine-reference/`, shared with
  the lab tool.

No card (`part.json`). Research harnesses live in `crates/nmlab`.
