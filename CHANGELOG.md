# Changelog

All notable changes to Neuromorph Nano are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Nothing yet.

## [0.1.0] - 2026-11-27

The first public prototype release of **Neuromorph Nano** by Luis Jake Gabriel III (Gavriel Technologies),
for Windows x64. Neuromorph Nano is source-available and non-commercial: it is licensed under the
PolyForm Noncommercial License 1.0.0 (`LICENSE`), with commercial use only under an individually
negotiated license (`COMMERCIAL.md`). Cross-platform Linux/macOS support and formal cross-hardware
benchmark reports follow in subsequent releases.

### Added

#### Engine

- A Rust workspace of 21 crates using only the Rust standard library and its own crates; `unsafe`
  is forbidden everywhere except the two FFI modules of `nmffi`, the bridge to the NVIDIA driver.
- 12 prebuilt CUDA fatbins (`sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90` SASS, plus `compute_75`
  and `compute_90` PTX), a per-card `dp4a` kernel auto-tuner cached in `memory/machine.json`, and
  a processor-only fallback (AVX2, FMA, F16C) when no usable NVIDIA card is present.
- Four MicroModels named by their role, each served from its own base cartridge: the **Reader**,
  **Writer**, **Embedder** and **Reranker**, built from the upstream open weights
  `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-Embedding-0.6B` and
  `Qwen/Qwen3-Reranker-0.6B`. The upstream weights are read only, never altered, and are the
  reference and the fallback.
- All four MicroModels resident on a 4 GB card, with residency homeostasis, idle KV window release
  (on by default, after 15 minutes idle), and hard limits on VRAM, RAM and background threads
  (`limits.vram_mb`, `limits.ram_mb`, `limits.cpu_threads`) with recommended values detected for
  the machine.
- The organism's path for a question: the Global Context Substrate (files, symbols and the
  reference graph, held in RAM), the Neuropod (tentacles and drones), the Globalized Context Stack
  (GCX: g0, G1, G2, G3) and the Core Brain, returning a cited context pack within the token budget
  and withholding the answer when nothing in the material supports it.
- Readers for code in many languages (a language table with a language-neutral fallback), PDF,
  Markdown, tables, logs, structured text, and zip, gzip, tar and Office containers, all read in
  memory.
- The idle Writer: in quiet time it builds a 4-level world model (`L1 Piece -> L2 File ->
  L3 Module -> L4 System`, the project outline), and yields to an interactive call at once.
- Memory per repository: verified episodes, tiers (core, active, archive), temporary notes and
  structural summaries, with lifetimes in run-days and a write gate that refuses anything that
  looks like a secret. Muscle memory packs hold behaviour, never code.
- The dense lane follows its Embedder: when the served Embedder's values change (a plug, an overlay,
  a new base version), a serving process answers on the lexical and structural paths, rebuilds the
  dense index at idle with the new Embedder and switches to it, with no restart.
- Continuous learning: gated learning cycles with held-out checks and atomic rollback, for clients
  with hooks and for MCP-only clients; preference cartridges and G3 muscle priors.

#### Cartridges and parameter maps

- Base cartridges with versions under your control: choose the version each MicroModel serves, and
  on an upgrade keep both versions (the default) or replace the old one.
- Job filters (`nmengn-sockets/1`), project embedder lenses, adapters and stacks that plug in on
  top of a base cartridge. Each job filter row in `/api/surgery` (and `nm_cartridges` filters)
  carries `built_on`, `serving_base` and `base_match`, so a filter built on another base version is
  shown as such; it still loads, with one warning line.
- Learned units: plug each one in or out, or switch a whole compartment, in the interface, the API
  and MCP.
- Map lineage: each parameter map records what it was made on, by which command and for what,
  shown both ways (what a map was made from and what was made from it).
- Map cleanup: archive old maps (reversible, with undo in Audit), restore them, or delete one map or
  a whole group in two steps. What serves, and what the serving base cartridge is built from, is
  protected.
- Cartridge sources: the official library on Hugging Face (`gavriel-tech/neuromorph-nano-cartridges`)
  and any third-party Hugging Face library you add, browsed and fetched only when you ask, through
  the library's own checks: stacks arrive on trial, an existing map is never replaced, and base
  cartridges come from the official library only.
- Every map, cartridge, adapter, lens, unit, memory, goal and connection can be inspected and
  controlled from the interface, the HTTP API and MCP; deletes are gated and changes are audited
  with undo.

- Parameter editing for every object that holds parameters: filter cartridges and job filters,
  adapters and lenses, learning units, preference values, base cartridges and the upstream open
  weights (read in place). Selections by path, tensor, layer, head, row, column, channel or range;
  operations set, add, multiply, strength, clamp, zero, negate, copy, interpolate, paste, import,
  export and rank truncation, each with a preview. Every edit is a new version beside the source;
  a base cartridge edit is a new base version on trial. A held-out check of an edited job filter
  against the open weights.
- A live parameter overlay: edit, mask (served as zero) and freeze (read-only) any parameter,
  tensor, layer, head, range, set of parameters or whole cartridge, from the interface, the API and
  MCP. The overlay is applied when a MicroModel is seated and takes effect at the next call; the
  released files are never written.
- A whole-set parameter viewer: every tensor of a cartridge or of the open weights as a value
  heatmap with rulers and a numeric scale, zoom down to single parameters, and an inspector with
  each parameter's value, coordinates, layer and statistics. Overlay state is drawn on the heatmap
  for reference; all changes are made in the controls. Weights are read only when opened in the
  editor. `neuromorph-nano library parameters <key> --overview` prints the same view as JSON.
- Distillation: teacher data (imported, or answered live by your agent over MCP or by a local
  open-weights model) trained into learned cartridges through one held-out gate: Writer adapters
  and steers, Reader and Reranker read-outs, Embedder request lenses, learning units, job filters,
  and preference cartridges from teacher verdicts.
- Stated preference cartridges: preferences written in plain words (English, Spanish, French,
  German or Portuguese, with file-kind and folder scopes) become rules of what to avoid and what to
  want. Avoided terms are masked in the Nano's own text; wanted terms and styles are favoured with
  a strength from 0.1 to 2.0; every rule is checked on a draft or passed to the agent as guidance,
  and each rule and value can be edited. Quoted code is never changed.
- Agent guidance: reminders, loops and a step graph for the coding agent, set by the user or the
  agent, delivered through the hooks. A loop ends when its condition holds (one condition, or all
  of or any of up to four), at its cap, when the same unmet reason repeats past its stall
  threshold, or when its time or tool-call budget is used; each ended loop leaves an outcome
  record. Steps depend on all of their steps or on any one of a list; a probe step is done when its
  check passes, with no edit. A reminder can fire when the same error repeats in a session.
- Your own repositories (any git remote, a Hugging Face repository or a folder) for backing up and
  rolling back memory, parameter maps, cartridges, learned units and settings; nothing is sent
  until you back up.
- `library/` holds only the cartridge library and the parameter map library.

#### MCP server and coding agents

- An MCP server over standard input and output with 25 `nm_*` tools: 19 for context, memory and
  build, and 6 parity tools (`nm_live`, `nm_storage`, `nm_cartridges`, `nm_learning`,
  `nm_comprehension`, `nm_settings`) that match the interface. `tools/list` advertises 19 by
  default; every tool is callable by name.
- `neuromorph-nano connect <claude|cursor|vscode|antigravity|windsurf|generic>`: writes the MCP
  entry for one client, keeps every other server's entry, and shows the exact difference first with
  `--dry-run`. Claude Code hooks are installed with `--hooks`.
- The agent calls list: every audit row records the MCP client's name given at connection, and Live
  Activity lists every coding agent's calls.
- Build help and Step Check (`nm_build_run`, `nm_build_explain`, `nm_stepcheck`, `nm_verify`), run
  with cargo `--offline`.

#### Interface

- The local interface rebuilt as 11 pages at `http://127.0.0.1:4870/` (`neuromorph-nano launch
  ui`), in three groups: Observe (Live Activity, Topology, Anatomy), Engine (Models, Cartridges,
  Parameters, Memory, Learning) and System (Storage, Audit, Settings).
- Every page has the working system in the middle and a right side panel with the page's
  libraries, references and an About tab, which also shows the detail of whatever is picked.
- Models shows the four MicroModels by role, what each runs on, base cartridge choice, routing and
  the downloads of models and base cartridges.
- Cartridges is built around decks: one deck per MicroModel with its base cartridge seated and its
  cartridges docked to plug, unplug or change version; the library, base versions, learned units,
  cartridge sources and a Components view are in the side panel.
- Parameters is a map atlas: lanes per MicroModel, layer heat, tiles by size, kind and state,
  lineage both ways, archive and restore.
- Memory has three tabs in the middle: Memory (tiers, strata, recent calls), World model (the four
  levels and the outline) and Loops and graph (the agent guidance and its verdicts).
- Live Activity and Topology show the organism at work with zoom and pan.
- Measured data only: a figure that was not measured is named as not measured.
- Cartridges holds the four decks in one rack; its side panel has a Sources tab for the cartridge
  libraries, and the glossary is part of the Components tab.
- Parameters has a `cartridges` view, and the cartridge parameters lane shows each MicroModel as a
  chain: open weights, base cartridge, filters, adapters and units.
- Parameters: a map's lineage (made on, made by, purpose, outcome and its reason, note) is recorded
  or corrected from the map's row or its detail; logged in Audit, with undo.
- Parameters: the focus controls export the selection's values to `exports/parameters/` and name
  the file written.
- Memory: the Loops and graph tab opens on the guidance circuit figure; the World model tab is one
  figure of the four levels with the levels, the outline and the idle Writer below it.
- Learning has four tabs (Loop, Distillation, Learned units, Preferences); Distillation shows its
  path in five steps with the engine's counts (teacher data, fill, train, held-out gate,
  cartridges).
- Storage has Disk and Backups tabs. Audit has an undoable filter, and an undoable row carries its
  Undo.
- Settings has a Settings file panel showing `settings.json` key by key, with the base cartridge,
  the `base_compensation` entry and the role model of each MicroModel as controls.
- Every page has a system strip under the page list (engine state, card memory, RAM, MicroModels,
  calls, idle Writer, disk); Live Activity, Topology, Anatomy and Models have a System panel in the
  side panel.
- The live view lights only the parts a call names, marks a replayed call as a replay, and draws
  the organism at rest when no engine is serving.
- Figure labels fit narrow windows: the Memory strata folder labels and the Anatomy legend no
  longer clip or overlap at 1366x768.

#### Install and release

- `tools/install.ps1`: one-line online install, or offline from local files (`-LocalZip`,
  `-ModelsFrom`, `-CartridgesFrom`). It checks the release zip, every model file and every base
  cartridge against its SHA-256, resumes partial downloads, and keeps your models, library and
  learned cartridges on an upgrade.
- `tools/package-release.ps1`: builds the Windows x64 release zip and `SHA256SUMS.txt`.
- End-to-end checks: `tools/e2e-verify.ps1` (HTTP API and MCP) and
  `interface/scripts/e2e-ui-verify.mjs` (every interface page and control).

### Security

- The interface server binds `127.0.0.1`; a change is accepted only with its local token, as JSON,
  from its own pages or a local program without a browser origin.
- Air-Gap Mode (`air_gap`, or `NANO_AIR_GAP=on`) keeps the interface local, turns off local
  network access and pairing, and refuses every download and remote listing; `GET /api/airgap` and
  `nm_settings op=airgap` report it.
- Reads of your code stay in RAM; the audit log never holds file contents.
- Optional tablet pairing on the local network (off by default) has no encryption on the wire; see
  `SECURITY.md`.

[Unreleased]: https://github.com/gavriel-tech/neuromorph-nano/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gavriel-tech/neuromorph-nano/releases/tag/v0.1.0
