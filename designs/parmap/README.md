# designs/parmap: the Parameter Map atlas

One file per population: `<population>.json` (schema `nm-atlas/1`, `crates/nmnpcs/src/atlas.rs`).
Each record is one piece of the population (layer, sublayer, key-value group, query head, block of
64 feed-forward channels, token table, output head) for one job, with what was measured there. It
is a record about the weights, never a copy of them, and it holds no reads of the code base.

Written by `nano-lab atlas compare [--base <id>] [--with <a,b>]` (the comparative map), report in
`runs/ATLAS-compare-<ms>.txt`. Later measurements add records per job.
