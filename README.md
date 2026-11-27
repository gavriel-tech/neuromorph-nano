<div align="center">

<img src="docs/images/banner.jpg" alt="Neuromorph Nano" width="640" />

# Neuromorph Nano

**A local cognitive symbiont for coding agents.**

*Give Claude Code, Cursor, VS Code, Antigravity, or Windsurf a local nervous system, persistent memory, and sub-3-second cited codebase recall, running entirely on your own machine.*

[![Version](https://img.shields.io/badge/version-v0.1.0-0ea5e9?style=flat-square)](CHANGELOG.md)
[![Rust](https://img.shields.io/badge/Rust-21%20crates%20%7C%20std%20only-f97316?style=flat-square)](crates/)
[![CUDA](https://img.shields.io/badge/CUDA-12%20fatbins%20%2B%20CPU%20fallback-22c55e?style=flat-square)](kernels/)
[![Cartridges](https://img.shields.io/badge/Hugging%20Face-Cartridge%20Library-ff9d00?style=flat-square)](https://huggingface.co/gavriel-tech/neuromorph-nano-cartridges)
[![Privacy](https://img.shields.io/badge/privacy-100%25%20local%20%7C%20air--gap%20ready-6366f1?style=flat-square)](docs/privacy.md)
[![License](https://img.shields.io/badge/license-source--available%20%7C%20PolyForm%20Noncommercial-eab308?style=flat-square)](LICENSE)

[A Cognitive Symbiont](#a-local-cognitive-symbiont) · [Architecture](#anatomy-of-the-organism) · [Requirements & Hardware](#release-status-hardware-and-system-requirements) · [Quickstart](#quickstart) · [Cartridge Library](https://huggingface.co/gavriel-tech/neuromorph-nano-cartridges) · [The Interface](#the-local-interface) · [Roadmap](#frontier-roadmap--open-collaboration) · [Privacy & Air-Gap](#privacy-air-gap-and-user-control) · [License](#license-and-copyright)

</div>

---

## A Local Cognitive Symbiont

**Neuromorph Nano** is built on **mutualistic symbiosis** with your coding agent and your hardware.

It does not compete with frontier LLMs, get in the way of your workflow, or burn cloud tokens on repetitive repo searches. It yields the GPU in under 300 ms when interactive work arrives, releases idle KV buffers automatically, obeys hard VRAM, RAM and CPU caps you set, keeps every byte on your disk, and lets you plug in, unplug or remove any learned cartridge cleanly.

Three biological analogies capture how it works:

- **A Symbiotic Nervous System (how it pairs with frontier models):** Think of a frontier LLM (Claude, GPT, Gemini) as a brilliant **prefrontal cortex** in the cloud: exceptional at reasoning, planning, and writing, yet blind to your repository until you spoon-feed it files every session. Neuromorph Nano acts as its local **peripheral nervous system, hippocampus, and cerebellum**, sensing your repository in parallel, remembering verified outcomes across sessions, and turning repeated wins into fast reflexes.
- **An Octopus-Like Distributed Anatomy (how it thinks internally):** An octopus keeps roughly two-thirds of its neurons in its arms rather than in one central brain. Neuromorph Nano distributes intelligence the same way: **Neuropod tentacles and drones** forage your files in parallel, the **Globalized Context Stack (GCX)** carries distilled signals upward across a strict sensory boundary, and one **Core Brain** coordinates four small MicroModels named by their role (**Reader**, **Writer**, **Embedder**, **Reranker**) plus **muscle memory**.
- **Mycorrhiza in Your Codebase (how it lives in a repo):** In a forest, a tall tree has canopy power while a mycorrhizal network lives quietly in the soil, mapping nutrient pockets and delivering concentrated supply on demand. While you are idle, Neuromorph Nano's Writer quietly builds a **4-level world model** (`Piece -> File -> Module -> System`) of your repository so your agent receives small, exact, cited context packs instead of brute-force greps.

---

## Anatomy of the Organism

![Live Activity: the organism lit by a running context pack, with the system strip and agent calls](docs/images/01-live-activity.png)

*Live Activity captured at 1920x1080 during a real MCP context-pack call with all four MicroModels resident.*

The full flow, the engine crates and the memory layout are in [docs/architecture.md](docs/architecture.md).

### Why Four Coexisting MicroModels Beat One Monolith Here

Instead of loading one slow giant model for every operation, Neuromorph Nano runs four specialized MicroModels, each seated on its own **base cartridge**, side by side inside a compact local GPU or CPU budget that scales dynamically with your hardware. Removable **job filters** (`nmengn-sockets/1`), **project embedder lenses**, **preference cartridges** and **learned units** plug in on top of them.

| MicroModel | Job | Upstream open weights it is built from |
|---|---|---|
| **Reader** | Reads for the drones and tentacles | `Qwen/Qwen3-0.6B` |
| **Writer** | Writes the world model notes at idle | `Qwen/Qwen3-1.7B` |
| **Embedder** | The meaning index and recall by meaning | `Qwen/Qwen3-Embedding-0.6B` |
| **Reranker** | Ranks and judges close candidates | `Qwen/Qwen3-Reranker-0.6B` |

> **Base cartridges and optional upstream open weights:** A MicroModel is named by its role; the `qwen3-*` ids are the upstream open weights it was built from, not the MicroModel. **Neuromorph Nano runs directly on its four base cartridges (about 6.5 GB)** plus the small tokenizer/config JSONs (about 44 MB). The full upstream Qwen3 open weights (about 8 GB, Apache-2.0) are **optional**: kept read-only when installed, and used for fallback (switching a role to `open`), output-parity reference checks, parameter mapping, and cartridge surgery. Want to contribute base cartridges, parameter maps or mapping techniques? See the [cartridge and parameter contributor protocol](CONTRIBUTING.md#contributing-weight-cartridges-parameter-maps-and-techniques).

| Organ | Role in the Organism | Built-in Safeguard |
|---|---|---|
| **Neuropod (Tentacles + Drones)** | Parallel foraging across code, Markdown, PDFs, CSV/TSV, HTML/XML/JSON/YAML, logs, and zip/tar/Office containers | Read-only sensory access; handles mixed encodings and unseen languages without crashing |
| **GCX (Spinal Cord)** | Carries distilled features and spans upward and control signals downward, in RAM | Enforces the `nmtype::class` sensory boundary so raw code never enters learned state |
| **Core Brain (`G0 -> G1 -> G2 -> G3`)** | Routes intent (`G0`), searches dense, lexical and tree priors (`G1`), runs tentacle reads and close-duel reranking (`G2`), and packs cited spans (`G3`) | Withholds an answer instead of guessing when nothing in the workspace supports it |
| **Idle Writer & World Model** | Uses quiet moments to write `L1 Piece -> L2 File -> L3 Module -> L4 System` notes (L4 is the project outline) | Yields to every interactive query (`< 300 ms` ceiling) |
| **Memory & Muscle Memory** | Stores verified per-repo episodes, tiered memory (`core / active / archive`), and compressed behavioural reflexes | Behaviour only, never copied code; every learned cartridge, unit or filter can be plugged in, unplugged or removed |
| **Local Interface** | 11 pages at `http://127.0.0.1:4870/`: live activity, topology, anatomy, MicroModels, cartridges, parameter maps, memory, learning, storage, audit and settings | No telemetry, no outside requests of its own, and one-switch Air-Gap Mode |

---

## Release Status, Hardware, and System Requirements

**`v0.1.0` is the first public prototype of Neuromorph Nano** (Windows 10/11 x64), built on a 4 GB GPU. It ships the complete end-to-end local organism: zero-dependency Rust runtime, 12 custom multi-arch CUDA fatbins, four coexisting MicroModels seated on base cartridges with dynamic VRAM/RAM scaling, 25 MCP tools, parameter mapping and cartridge surgery, continuous learning primitives, and the 11-page local observability interface. Formal cross-hardware benchmark reports will be published as the platform matures beyond this initial prototype.

### System Requirements (`v0.1.0`)

| Component | Requirement |
|---|---|
| **Operating System** | Windows 10 / 11 (x64). Native Linux x64 and macOS (Apple Silicon) support are next on the roadmap. |
| **GPU (Recommended)** | NVIDIA Turing or newer (`sm_75`+: GTX 16xx, RTX 20xx-50xx; adapts automatically to your GPU's available VRAM) with a current NVIDIA display driver. No CUDA Toolkit install required. |
| **CPU Fallback** | Any x86-64 processor with AVX2, FMA, and F16C when no usable NVIDIA GPU is present. |
| **Host RAM** | 16 GB+ recommended (user-configurable hard caps via `limits.ram_mb` and per-area memory budgets). |
| **Disk Space** | About 7 GB free for the four SHA-256-verified base cartridges (about 6.5 GB) and tokenizer/config JSONs (about 44 MB). Add about 8 GB if you also install the optional upstream open weights for fallback, reference checks, parameter mapping, or surgery. |

### Hardware Compatibility & Adaptation

- **Zero-config multi-architecture CUDA:** All 12 custom CUDA fatbins ship prebuilt across **7 targets** (`sm_75` Turing, `sm_80`/`sm_86` Ampere, `sm_89` Ada Lovelace, `sm_90` Hopper SASS, plus `compute_75` and `compute_90` PTX for forward JIT on newer GPUs).
- **Per-chip auto-tuning:** On first run, `nmengn::autotune` benchmarks the four bit-identical `dp4a` tile kernels on your GPU in under 100 ms and caches the fastest pair in `memory/machine.json`.
- **Dynamic residency & memory areas:** VRAM ceilings, KV-window reserves, host-weight offloading, and per-area memory budgets (`content`, `caches`, `weights`) adapt automatically to your hardware and obey hard user limits.
- **Automatic CPU fallback:** Machines without an NVIDIA GPU or with an older driver automatically fall back to the pure-Rust processor path (AVX2, FMA, F16C) and report the active path in the CLI, UI, and MCP status.

---

## Quickstart

### Option 1: Windows Installer (x64)

```powershell
irm https://raw.githubusercontent.com/gavriel-tech/neuromorph-nano/main/tools/install.ps1 | iex
```

The installer downloads the release zip and `SHA256SUMS.txt` from GitHub Releases, checks the zip's SHA-256, installs into `%LOCALAPPDATA%\NeuromorphNano` and adds `bin\` to your user `PATH`. It then fetches the four base cartridges (about 6.5 GB, from the official cartridge library) plus the small tokenizer/config JSONs (about 44 MB) so Neuromorph Nano runs standalone immediately. Pass `-WithOpenWeights` (or use **Models -> Downloads** in the interface) if you also want the optional upstream open weights (about 8 GB) for fallback, reference parity checks, parameter mapping, or surgery. Every file is checked against its SHA-256 before it is placed, and a partial download resumes.

Installing on an offline or air-gapped machine? Copy the release zip, `SHA256SUMS.txt`, the models and the cartridges over once, then run:

```powershell
.\tools\install.ps1 -LocalZip .\neuromorph-nano-v0.1.0-win-x64.zip -ModelsFrom E:\nano\weights -CartridgesFrom E:\nano\library
```

### Option 2: Build From Source (Windows 11 x64)

Requires Rust `1.99.1` (MSVC, selected by `rust-toolchain.toml`) and Node.js `20.9+` (only to build the static interface). The CUDA toolkit is needed only if you edit the `.cu` kernels.

```powershell
cargo build --release
cd interface; npm ci; npm run build; cd ..
```

The program is `target\release\neuromorph-nano.exe`. A source build does not download models or cartridges; get them from the interface (Models and Cartridges pages) or run the installer. See [docs/user-guide.md](docs/user-guide.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

### Connect Your Coding Agent (One Command)

`neuromorph-nano connect` writes the MCP entry for one client and keeps every other server's entry. Add `--dry-run` first to see the exact difference it would make:

```powershell
# Pick your client: claude | cursor | vscode | antigravity | windsurf | generic
neuromorph-nano connect claude --project C:\path\to\your\repo

# Open the local interface at http://127.0.0.1:4870/
neuromorph-nano launch ui
```

`generic` prints the command line and JSON for any other stdio MCP client. Each call's audit row records the name the client gives when it connects, and Live Activity lists every coding agent's calls. The 25 tools are listed in [docs/mcp-tools.md](docs/mcp-tools.md).

---

## The Local Interface

`neuromorph-nano launch ui` serves the interface at `http://127.0.0.1:4870/`. Every page has the same frame: the page list on the left, the working system in the middle (live visuals and their controls), and a right side panel with the page's collections, references and an About tab, which also shows the detail of whatever you pick in the middle.

| Group | Pages |
|---|---|
| **Observe** | **Live Activity** (`/`): every coding agent's calls, the organism at work (zoom and pan), stage times. **Topology**: ganglion stages, tentacles, base and learned cartridges. **Anatomy**: every part, its card and what it depends on. |
| **Engine** | **Models**: the four MicroModels, what each runs on, base cartridge choice, routing and weights downloads. **Cartridges**: one deck per MicroModel with its base cartridge seated and its cartridges docked to plug, unplug or change version; the cartridge library, base versions, learned units and cartridge sources in the side panel. **Parameters**: the map atlas, with each map's lineage (what it was made on, by which command, for what), archive and restore. **Memory**: memory tiers, the world model and recall. **Learning**: learning cycles and rollback, the evidence ledger and the training data. |
| **System** | **Storage**: disk use by category, with safe pruning. **Audit**: every call and change, with undo. **Settings**: every control, hardware limits and Air-Gap Mode. |

The interface shows measured data only; a figure that was not measured is named as not measured. Every feature is also reachable through MCP and the local JSON API ([docs/interface-api.md](docs/interface-api.md)), and deleting anything goes through a preview and more than one confirmation step.

### Screenshots

Captured at 1920x1080 during live MCP calls against a real repository, started the way Claude Code starts it. All four MicroModels are held on the card, a context pack is running and the Embedder is executing; every lit part and every figure is a live reading of that run.

| | |
|---|---|
| ![Topology](docs/images/02-topology.png) **Topology**: the organism's systems, each with its state now. | ![Anatomy](docs/images/03-anatomy.png) **Anatomy**: every part and the crate that holds it. |
| ![Models](docs/images/04-models.png) **Models**: the four MicroModels, each on the card with its layers lit by real calls. | ![Cartridges](docs/images/05-cartridges.png) **Cartridges**: one deck per MicroModel, base seated, cartridges docked. |
| ![Parameters](docs/images/06-parameters.png) **Parameters**: the map atlas and the cartridge parameters. | ![Memory](docs/images/07-memory.png) **Memory**: tiers as strata, one column per project. |
| ![World model](docs/images/08-world-model.png) **World model**: the four levels the idle Writer builds. | ![Loops and graph](docs/images/09-loops-graph.png) **Loops and graph**: the guidance circuit for your coding agent. |
| ![Learning](docs/images/10-learning.png) **Learning**: the learning loop, distillation, learned units and preferences. | ![Storage](docs/images/11-storage.png) **Storage**: disk use by category and your backup repositories. |
| ![Audit](docs/images/12-audit.png) **Audit**: every call and change along time, with undo. | ![Settings](docs/images/13-settings.png) **Settings**: every control, hardware limits and memory areas. |

### Cartridge Libraries

The official cartridge library is on Hugging Face at [`gavriel-tech/neuromorph-nano-cartridges`](https://huggingface.co/gavriel-tech/neuromorph-nano-cartridges). You can also add third-party Hugging Face libraries as cartridge sources. Nothing is fetched until you ask; everything fetched goes through the same checks as the cartridge library on this machine (stacks arrive on trial, an existing map is never replaced, base cartridges come from the official library only). See [docs/cartridges-and-parameters.md](docs/cartridges-and-parameters.md) and [docs/build-flow.md](docs/build-flow.md).

---

## Field Guide: Explore the Organism

Every folder in this repository corresponds to a layer of the organism. Because the Rust workspace uses **only the Rust standard library (`std`)**, you can read any crate end to end without chasing dependency trees.

<details>
<summary><strong>The 21 Rust Crates (Grouped by Organ System)</strong></summary>

| Organ System | Crates | What to Explore |
|---|---|---|
| **Sensory & Neuropod** | [`nmdoc`](crates/nmdoc/), [`nmlex`](crates/nmlex/), [`nmsubs`](crates/nmsubs/), [`nmpod`](crates/nmpod/) | Code lexers for many languages, PDF, Office and archive readers, the Global Context Substrate, parallel tentacles, and `RowCache` drones |
| **Core Brain & Nervous System** | [`nmbrain`](crates/nmbrain/), [`nmnte`](crates/nmnte/), [`nmmem`](crates/nmmem/), [`nmmusc`](crates/nmmusc/) | Ganglion stages (`G0-G3`), the executor, tiered memory (`core/active/archive`), and muscle memory |
| **Neural Engine & Custom CUDA** | [`nmengn`](crates/nmengn/), [`nmffi`](crates/nmffi/), [`nmsafe`](crates/nmsafe/), [`nmtokz`](crates/nmtokz/) | Safetensors reading, the Qwen3 BPE tokenizer, the 4-model residency manager, the per-chip `dp4a` auto-tuner, idle KV release, and zero-VRAM speculative decoding |
| **Cartridges & Continuous Learning** | [`nmnpcs`](crates/nmnpcs/), [`nmlearn`](crates/nmlearn/) | Model cards and parameter maps, `nmengn-sockets/1` job filters, lenses, preference cartridges, learned units, and split-gate verification |
| **MCP, Build Help & Foundations** | [`nmapi`](crates/nmapi/), [`nmcore`](crates/nmcore/), [`nmtool`](crates/nmtool/), [`nmtype`](crates/nmtype/), [`nmjson`](crates/nmjson/), [`nmhash`](crates/nmhash/), [`nmlab`](crates/nmlab/) | The 25 MCP tools and the interface parity layer, the `neuromorph-nano` program and interface server, the `nmtype::class` sensory boundary, Cargo build help and Step Check, and the `nano-lab` measurement tool |

</details>

<details>
<summary><strong>The 25 MCP Tools & Documentation (<code>docs/</code>)</strong></summary>

Every user-facing capability works the same in the interface and over MCP: `nm_context_pack`, `nm_repo_map`, `nm_symbol_find`, `nm_deps`, `nm_view`, `nm_sync`, `nm_memory`, `nm_feedback`, `nm_goal`, `nm_unit`, `nm_build_run`, `nm_build_explain`, `nm_improve_scan`, `nm_patch_propose`, `nm_verify`, `nm_stepcheck`, `nm_status`, `nm_anatomy`, `nm_weights`, `nm_live`, `nm_storage`, `nm_cartridges`, `nm_learning`, `nm_comprehension` and `nm_settings`. All 25 tools are advertised by default over `tools/list` and can be filtered through the `mcp.*` settings.

- [Architecture](docs/architecture.md) · [User Guide](docs/user-guide.md) · [MCP Tools Reference](docs/mcp-tools.md)
- [Cartridges & Parameters](docs/cartridges-and-parameters.md) · [Build Flow](docs/build-flow.md) · [Continuous Learning](docs/continuous-learning.md)
- [Agents and models](docs/agents.md) · [Configuration](docs/configuration.md) · [Interface API](docs/interface-api.md) · [Privacy](docs/privacy.md)
- The full index is [docs/README.md](docs/README.md).

</details>

---

## Privacy, Air-Gap, and User Control

- **The Nano sends nothing off your machine on its own:** The engine sends no telemetry and makes no network request of its own. Your coding agent is separate: the context the Nano hands it goes to the agent's model like any file the agent reads (with a cloud model, to its provider); run a local model behind your agent for nothing to leave at all (see [Agents and models](docs/agents.md)). The MCP server talks to your agent over standard input and output; the interface binds `127.0.0.1`. The only outbound requests are downloads and library listings you explicitly trigger (models, cartridges, cartridge sources), fetched from Hugging Face over HTTPS (`curl.exe`) with every file verified against its SHA-256.
- **Verifiable Air-Gap Mode:** Turn on **Air-Gap Mode** in Settings, over MCP (`nm_settings op=set key=air_gap value=on`), or lock it for the process with `$env:NANO_AIR_GAP = "on"`. Air-Gap Mode keeps the interface on `127.0.0.1`, turns off local network access and tablet pairing, and refuses every download and remote listing. `GET /api/airgap` and `nm_settings op=airgap` show its state live.
- **Hardware limiters:** Cap GPU VRAM (`limits.vram_mb`), host RAM (`limits.ram_mb`) or background threads (`limits.cpu_threads`) in Settings or over MCP, or apply the recommended values detected for your hardware in one click.
- **Transparent storage:** See the exact paths and sizes of memory, dense vectors, quantized copies and cartridges in Storage or `nm_storage`, and prune rebuildable caches whenever you want.
- **Your code stays your code:** What the Nano reads from your repository is held in RAM and never written to disk; the audit log never holds file contents.

See [docs/privacy.md](docs/privacy.md) and [SECURITY.md](SECURITY.md) for the full statement.

---

## Repository Layout

| Path | Contents |
|---|---|
| [`crates/`](crates/) | The 21 Rust crates (standard library only; `unsafe` forbidden everywhere except the two FFI modules of `nmffi`) |
| [`kernels/`](kernels/) | CUDA kernel sources and the 12 prebuilt multi-arch `.fatbin` files |
| [`designs/`](designs/) | The design files the engine reads: anatomy, memory policy, models, chains, the serving base cartridges |
| [`interface/`](interface/) | The local interface (Next.js, TypeScript, static export) |
| [`docs/`](docs/) | Architecture, user guide, MCP and API reference, privacy, cartridges and learning |
| [`tools/`](tools/) | Installer (`install.ps1`), release packager, card and integrity sealing, and end-to-end verification |
| [`tests/`](tests/) | Engine reference recordings, development question sets and section-check fixtures |
| [`weights/`](weights/) | The cards of the upstream open weights (SHA-256 of each file); the weights themselves are downloaded, never committed |
| `rules.md`, `INTEGRITY.md` | The Nano's own rules and the SHA-256 integrity manifest |

*Runtime state (`memory/`, `runs/`, `library/`, `muscle/`, `units/`, `settings.json`, `cartridge-sources.json`, the downloaded model files, `bin/`, `scratch/`, `target/`) stays on your machine and out of git.*

---

## Frontier Roadmap & Open Collaboration

`v0.1.0` proves the core architecture end to end on Windows x64 (NVIDIA CUDA and CPU fallback). From here, we are pushing the frontier of local, modular cognitive systems and actively welcome collaborators who want to expand platform reach, hardware efficiency, and intelligence:

- **Cross-Platform Runtime & Hardware Backends:** Native Linux x64 (`libcuda.so.1` + POSIX `nmffi`), Apple Silicon ARM64 (NEON + native Metal GPU kernels), and AMD/Vulkan GPU acceleration, while preserving the zero-dependency `std`-only Rust core with all `unsafe` isolated in `nmffi`.
- **Dynamic Memory & Larger Local Models on Consumer Hardware:** Finer-grained dynamic weight placement, on-demand disk-mapped token tables, and running larger frontier local models comfortably inside modest VRAM/RAM budgets.
- **Parameter Surgery, Cartridges & Continual Learning:** Higher-throughput quantization, richer teacher distillation, automated parameter mapping, and community-contributed base cartridges, job filters, and lenses on Hugging Face.
- **Multi-Repository & Enterprise Workflows:** Shared local knowledge graphs, multi-repository orchestration, and commercial support.

If you build Rust systems, CUDA/Metal/Vulkan kernels, quantization passes, or parameter maps, read [CONTRIBUTING.md](CONTRIBUTING.md) and [CONTRIBUTOR_POOL.md](CONTRIBUTOR_POOL.md) to join the build.

---

## License and Copyright

Copyright (c) 2026 Luis Jake Gabriel III (Gavriel Technologies). All rights reserved.  
Designed by Luis Jake Gabriel III.

Neuromorph Nano is **source-available and non-commercial**. It is not open source. Personal, academic and other non-commercial use is free under the **PolyForm Noncommercial License 1.0.0** ([LICENSE](LICENSE)). Commercial use of any kind requires an individually negotiated commercial license from Luis Jake Gabriel III (Gavriel Technologies); only the copyright holder may offer Neuromorph Nano as a commercial product. See [COMMERCIAL.md](COMMERCIAL.md).

The official cartridge library ([`gavriel-tech/neuromorph-nano-cartridges`](https://huggingface.co/gavriel-tech/neuromorph-nano-cartridges)) is under the same PolyForm Noncommercial license. The upstream Qwen3 models it is built from are open weights by the Qwen team under the Apache License 2.0; their notices ship beside the library and are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Contributions are licensed as [CONTRIBUTING.md](CONTRIBUTING.md#contributor-licensing) says, and merged contributions are credited under [CONTRIBUTOR_POOL.md](CONTRIBUTOR_POOL.md) (the Neuromorph Nano-only 55% / 30% / 15% contribution reward distribution system). Questions: `gavriel.techno@gmail.com`.
