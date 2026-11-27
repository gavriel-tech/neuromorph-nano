# tests/engine-reference/qwen3-1.7b: release reference for Qwen3-1.7B

The released Qwen3-1.7B run in PyTorch at float32 on the processor: `reference.json` (4 prompts
with the release tokenizer's ids and the argmax at every position) and `prompt<i>-last.f32` (the
last position's logits, little-endian float32). `nano-lab run s6 writer-parity [cpu|gpu] [f16|q8]`
binds the shared `qwen3-reference` assembly over `qwen3-1.7b` and compares (pass: last argmax equal
on every prompt, per-position agreement at least 95%, last-logits cosine at least 0.999). Hashes
in `INTEGRITY.md`.
