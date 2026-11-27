# tests/engine-reference/role-models: released reranker and embedding models

`inputs.json` (21 query and passage pairs, 5 texts) and `outputs.json` (the engine's own P(yes) per
pair and vector per text, recorded on the processor by `nano-lab run s6 roles-record`). The engine
self-test and `nano-lab run s6 roles-parity [cpu|gpu|both]` retest against it (worst P(yes)
difference at most 0.003, cosine with the recording at least 0.999). Hashes in `INTEGRITY.md`.
