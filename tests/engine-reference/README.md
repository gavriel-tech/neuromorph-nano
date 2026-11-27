# tests/engine-reference: engine reference recordings

Recordings the engine self-test compares against (`neuromorph-nano selftest engine`). Hashes in
`INTEGRITY.md`; re-record only on purpose.

- `qwen3-0.6b/`: tokenizer cases, single-layer outputs, full forwards and 64-token greedy decodes
  recorded by the engine itself (`nano-lab run s6 record`), with `MANIFEST.sha256`. Used by the
  self-test, `nano-lab run s6 parity`, `nano-lab run s7 decode` and `check nmtokz`.
- `role-models/`: inputs and recorded outputs of the released reranker and embedding models.
- `qwen3-1.7b/`: the released Qwen3-1.7B run in PyTorch, for `nano-lab run s6 writer-parity`.
- `fingerprint.txt`: the reference-path fingerprint lines (`selftest engine --record` writes it).
