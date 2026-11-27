# nmtokz

Qwen3 byte-level BPE tokenizer. In-house engine code of Gavriel Technologies (see
`THIRD_PARTY_NOTICES.md`). `model`, `normalize`, `split`, `added`, `unicode_data` (with its
`.sha256`), `unicode_marks`, `json`, `error`, `check` (compares against
`tests/engine-reference/qwen3-0.6b/runs/A/tokenizer`).

Section checks: `neuromorph-nano check nmtokz`.
