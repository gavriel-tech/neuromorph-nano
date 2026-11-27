# designs/assemblies: execution assemblies

Graphs in the NTE format that the engine or the organism binds and runs.

- `core.json`, `tentacle.json` (the one tentacle: tentacle, drone tip, core, `g0_tx`, note,
  reporting to G1) and `compose-context.json` (composed by `nano-lab compose <kind>`);
  `compose-context-territory.json` and `forage-s2.json` (earlier forms, kept).
- `qwen3-reference.json`: the one shared Qwen3 structure (cluster references name tensor and
  region; the population is supplied when it binds), used for Qwen3-0.6B and Qwen3-1.7B.
  `qwen3-split-attention.json`: the same forward with attention split in two paths. Both written by
  `nano-lab run s6 assemblies`.
- `qwen3-reranker.json`, `qwen3-embedding.json`: additions to the shared structure for the released
  reranker and embedding models, written by `nano-lab run s6 roles`.
- `tentacle-t1.json` to `tentacle-t8.json`: each tentacle's own neural body, written by
  `nano-lab run tentacle assemblies`.

Generated files are not edited by hand.
