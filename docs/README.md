# docs: Neuromorph Nano documentation

| File | What it covers |
|---|---|
| [`user-guide.md`](user-guide.md) | Install, connect a repository to a coding agent, and use the interface. |
| [`architecture.md`](architecture.md) | How the Nano is built: the organism's parts, how a question flows, the engine crates, the interface server and the memory layout. |
| [`agents.md`](agents.md) | Using the Nano from any agent or model: MCP clients (cloud or local models), `daemon` with `ask` and `attach` without MCP, scripts and the local API, and what leaves your machine in each setup. |
| [`mcp-tools.md`](mcp-tools.md) | The 25 `nm_*` MCP tools: arguments, replies and error codes. |
| [`interface-api.md`](interface-api.md) | The interface server's local JSON API. |
| [`configuration.md`](configuration.md) | The controls in `settings.json`, the design files in `designs/`, the models folder and the command-line options. |
| [`cartridges-and-parameters.md`](cartridges-and-parameters.md) | Every cartridge type, the cartridge library and the plug and unplug rack, parameter maps and job filters, and the upstream open weights as reference and fallback. |
| [`build-flow.md`](build-flow.md) | How the weights are built, step by step: open weights, maps, surgery, base cartridges, job filters, adapters, stacks and learning, with the commands used and each map's lineage. |
| [`continuous-learning.md`](continuous-learning.md) | The continuous learning design and what of it v0.1.0 runs. |
| [`dead-skin-vectors.md`](dead-skin-vectors.md) | The dead-skin biomimetic vector store (`dense_store.rs`, `skin.json`): continuous micro-shedding, undo/branch-switch revival barrier, enzymatic age calibration, and O(1) segment drop. |
| [`privacy.md`](privacy.md) | What stays on the machine, what is stored where, Air-Gap Mode, and how to check it yourself. |

Top-level files of the repository:

| File | What it is |
|---|---|
| [`README.md`](../README.md) | What the Nano is, hardware, quickstart and the interface. |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | How to build and check a change, the code rules, how pull requests work, and contributor licensing. |
| [`SECURITY.md`](../SECURITY.md) | How to report a vulnerability, and the short security and privacy statement. |
| [`CHANGELOG.md`](../CHANGELOG.md) | Release notes. |
| [`LICENSE`](../LICENSE) | PolyForm Noncommercial License 1.0.0 (source-available, non-commercial). |
| [`COMMERCIAL.md`](../COMMERCIAL.md) | Commercial use under an individually negotiated license. |
| [`CONTRIBUTOR_POOL.md`](../CONTRIBUTOR_POOL.md) | The Neuromorph Nano-only contribution reward distribution system. |
| [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) | The upstream models, the cartridge library's notices and the interface's npm packages, with their licenses. |
| [`INTEGRITY.md`](../INTEGRITY.md) | The SHA-256 integrity manifest. |

Official repositories:
- **Source & Releases (GitHub):** [`gavriel-tech/neuromorph-nano`](https://github.com/gavriel-tech/neuromorph-nano)
- **Official Cartridge Library (Hugging Face):** [`gavriel-tech/neuromorph-nano-cartridges`](https://huggingface.co/gavriel-tech/neuromorph-nano-cartridges)
