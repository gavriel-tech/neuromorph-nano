# nmnpcs

The Neural Parameter Compatibility System and the Parameter Map. `parmap` (every change counted,
appended to `edits.jsonl` while recording is on; frozen clusters refused), `acquire` (weight cards
from the models folder, verified by SHA-256), `decompose` (a model cut into clusters, coverage
audit), `compat`, `provision` (one read per tensor, one shared view per cluster, residency),
`lifecycle`, `inspect` (read only, for `nm_weights`), `atlas` and `compare` (the Parameter Map
atlas), `cartridge` (the `nm-cartridge/1` manifest format; base cartridges are lab data only).

Section checks: `neuromorph-nano check nmnpcs`.
