# nmmusc

Pattern recognition of the GCX layer G3 and muscle memory. Depends on `nmjson`, `nmhash`, `nmtype`.

- `symbol`: behavioural symbols and the codebook; sensory items are refused.
- `log`: append-only symbol sequences per question in RAM. `miner`: recurring chunks ranked by
  compression gain. `gain`: the gain gate (settings in `designs/muscle.json`).
- `pack`: the pack format (magic, JSON header, quantized payload). `store`: packs under `muscle/`
  with `index.json`; refuses `memory/`, `weights/`, `units/`. `reload`, `withdraw`, `priors`.
- `g3`: G3 as its own level over the configuration records (tests G3-T and MO-T).
- `measure`: measurements (`cargo test --release -p nmmusc -- --ignored --nocapture`).

Muscle memory is its own store, separate from memory and weights. Section checks:
`neuromorph-nano check nmmusc`.
