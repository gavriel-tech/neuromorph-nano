# crates: the Rust workspace

The 21 crates of the Neuromorph Nano. Each crate has `Cargo.toml`, `src/` (one sub-component per
file) and a `README.md`; every library crate also has a sealed card `part.json`. The program crate
`nmcore` and the lab tool `nmlab` carry no card. Dependency order: `nmhash`, `nmjson`, `nmlex`,
`nmffi`, `nmsafe`, `nmtokz` at the bottom, then `nmtype`, then `nmdoc`, then the component crates; only `nmapi`
and the two binaries wire crates together.

| Crate | Part |
|---|---|
| `nmhash` | SHA-256 |
| `nmlex` | token and term helpers |
| `nmjson` | strict JSON reader and canonical encoder |
| `nmdoc` | document decoders: format table, text encodings, structure of Markdown, plain text, CSV and TSV, HTML, XML, JSON, JSON Lines, notebooks, YAML, logs |
| `nmtype` | contracts, Context Packet, `nano_root()`, `tests_dir()` |
| `nmnte` | NTE executor and RNAM (graph, snapshots, scheduler, registry, lifecycle) |
| `nmsubs` | Global Context Substrate and its views; the document model of each file, document chunks, the code lock |
| `nmpod` | Neuropod: tentacles, drone tips, territories |
| `nmbrain` | Core Brain lobes and the four slots |
| `nmmem` | memory system (hierarchy, tiers, temp notes, gate, identity) |
| `nmnpcs` | NPCS and the Parameter Map (weight cards, clusters) |
| `nmengn` | neural engine over Qwen3 (CPU and GPU, assemblies, decode, adapters, engine reference) |
| `nmffi` | the only `unsafe` crate: CUDA driver and dynamic loading |
| `nmsafe` | safetensors ranges and quantization |
| `nmtokz` | Qwen3 byte-level BPE tokenizer |
| `nmlearn` | learning-weight units, readouts, learned adapter cartridges, preferences |
| `nmtool` | build help, Step Check, improvement findings, proposal queue, conformance |
| `nmmusc` | G3 pattern recognition and muscle memory (behaviour log, chunk miner, gain gate, packs) |
| `nmapi` | MCP face and the organism (tools, forage, session, assemble, delta) |
| `nmcore` | the product binary `neuromorph-nano` and the shared engine self-test (`nmcore::selftest`) |
| `nmlab` | the lab tool `nano-lab`: research harnesses, stage runs, generators, lab tests; not released |

After a source change: `tools/gen-cards.ps1`, then `neuromorph-nano seal crates/*/part.json`, then
`neuromorph-nano check cards`.
