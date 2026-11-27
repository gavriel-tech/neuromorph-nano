# nmlab

The lab tool `nano-lab` (`src/main.rs`): research harnesses, stage runs (`run s2` to `run s11`),
generators of design data (`run s6 assemblies`, `run tentacle assemblies`, `compose`), developer
checks (`conform`, `scoreboard`, `anatomy tentacles`) and the lab integration tests in `tests/`.
It builds with the workspace so it keeps compiling with every change, but it is not part of the
released product.

Many harnesses and tests need data sets that are not in the Nano folder (the CX1 question set
under `tests/capability-gates/cx1-context/`, scan dumps under `scratch/tc1-dump/`); they name the
missing file when run, and the tests that need them are `#[ignore]` with the reason. No card
(`part.json`). `nano-lab --help` lists every command.
