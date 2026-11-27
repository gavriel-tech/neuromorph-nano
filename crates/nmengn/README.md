# nmengn

The neural engine over the released Qwen3 models, on the processor and the graphics card.
`half`, `cpu`, `gpu` (through `nmffi`), `placement`, `assembly`, `model`, `components`, `roles`
(embed and rerank), `reference` (the engine reference in `tests/engine-reference/` and parity
against it), `decode` (greedy stepper with key and value cache), `adapter` (learned adapters on
linear nodes), `kvstore` (encoded chunks and shared key and value windows), `drone` (drone-key
packets), `residency` (the shared device budget). No block type: assemblies in
`designs/assemblies/` connect components. Models are loaded whole at the set precision
(`--precision`, 16, 8 or 4 bits per part).

Section checks: `neuromorph-nano check nmengn`. Engine self-test: `neuromorph-nano selftest engine`.
