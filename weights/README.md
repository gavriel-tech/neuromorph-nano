# weights: Upstream Open-Weight References

Each subfolder holds the pinned weight card (`part.json`) for one upstream open-weight model used as the read-only reference, tokenizer/configuration source, and fallback for Neuromorph Nano's four MicroModels:

| Folder | MicroModel Role | Upstream Repository | License |
|---|---|---|---|
| `qwen3-0.6b/` | Reader | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | Apache-2.0 |
| `qwen3-1.7b/` | Writer | [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) | Apache-2.0 |
| `qwen3-embedding-0.6b/` | Embedder | [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Apache-2.0 |
| `qwen3-reranker-0.6b/` | Reranker | [`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | Apache-2.0 |

Model weight binaries (`files/`) are never committed to git. They are downloaded into `weights/<id>/files/` by `tools/install.ps1` or the local interface (Models -> Downloads), verified against the SHA-256 digests in `part.json` and `INTEGRITY.md`, and kept strictly read-only on disk (never fine-tuned or altered). See [docs/cartridges-and-parameters.md](../docs/cartridges-and-parameters.md) and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
