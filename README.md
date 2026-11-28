<div align="center">

<img src="docs/images/banner.jpg" alt="Neuromorph Nano" width="640" />

# Neuromorph Nano

**A local cognitive symbiont for coding agents.**

*Give Claude Code, Cursor, VS Code, Antigravity, or Windsurf a local nervous system, persistent memory, and sub-3-second cited codebase recall, running entirely on your own machine.*

[![Version](https://img.shields.io/badge/version-v0.1.01-0ea5e9?style=flat-square)](CHANGELOG.md)
[![Rust](https://img.shields.io/badge/Rust-21%20crates%20%7C%20std%20only-f97316?style=flat-square)](crates/)
[![CUDA](https://img.shields.io/badge/CUDA-12%20fatbins%20%2B%20CPU%20fallback-22c55e?style=flat-square)](kernels/)
[![Cartridges](https://img.shields.io/badge/Hugging%20Face-Cartridge%20Library-ff9d00?style=flat-square)](https://huggingface.co/gavriel-tech/neuromorph-nano-cartridges)
[![Privacy](https://img.shields.io/badge/privacy-100%25%20local%20%7C%20air--gap%20ready-6366f1?style=flat-square)](docs/privacy.md)
[![License](https://img.shields.io/badge/license-source--available%20%7C%20PolyForm%20Noncommercial-eab308?style=flat-square)](LICENSE)

[Digital Organism](#a-local-digital-organism--cognitive-symbiont) · [Architecture](#anatomy-of-the-organism) · [Modular Cartridges](#modular-parameter-cartridges--preventing-catastrophic-forgetting) · [Biomimetic Memory & Learning](#biomimetic-memory-dead-skin-vectors-continuous-learning--muscle-memory) · [Parameter Mapping & Surgery](#precision-parameter-mapping-surgery--live-editing) · [Priming & Primes Studio](#proactive-priming-primes-studio--agent-guidance) · [Features](#complete-feature-set) · [Quickstart](#quickstart) · [The Interface](#the-local-interface) · [Privacy](#privacy-air-gap-and-user-control) · [License](#license-and-copyright)

</div>

---

## A Local Digital Organism & Cognitive Symbiont

**Neuromorph Nano** is a **digital organism** built on **mutualistic symbiosis** with your coding agent and your hardware.

It does not compete with frontier LLMs, get in the way of your workflow, or burn cloud tokens on repetitive repo searches. It yields the GPU in under 300 ms when interactive work arrives, releases idle KV buffers automatically, obeys hard VRAM, RAM and CPU caps you set, keeps every byte on your disk, and lets you plug in, unplug or remove any learned cartridge cleanly.

Three biological principles define its architecture:

- **A Symbiotic Nervous System (how it pairs with frontier models):** Think of a frontier LLM (Claude, GPT, Gemini) as a **prefrontal cortex** in the cloud: strong at reasoning, planning, and writing, yet blind to your repository until you feed it files every session. Neuromorph Nano serves as its local **peripheral nervous system, spinal cord, hippocampus, and cerebellum**, sensing your repository in parallel, remembering verified outcomes across sessions, nudging the agent with verified primes and guidance, and compressing repeated wins into fast reflexes.
- **An Octopus-Like Distributed Digital Organism (how it thinks internally):** An octopus keeps roughly two-thirds of its neurons in its arms rather than in one central brain. Neuromorph Nano distributes cognition the same way: **drones act as dendrites** at the sensory tips of **eight role-agnostic tentacles** (each tentacle is a small brain that can take any lens dynamically per question across its own disjoint 12,000-token window, reading 96,000 tokens in parallel at default over the **Global Context Substrate (GCS)**). The **Globalized Context Stack (GCX)** acts as the **spinal cord and brain stem** (`g0 -> G1 -> G2 -> G3`), carrying afferent sensory **Context Packets** upward and efferent motor control downward across a strict sensory boundary, into one singular **Core Brain** whose regions are four small **MicroModels** (**Reader**, **Writer**, **Embedder**, **Reranker**) and whose **parameter sets are its neurons**.
- **Mycorrhiza in Your Codebase (how it lives in a repo):** While a frontier model holds canopy reasoning power, a mycorrhizal network lives quietly in the soil, mapping nutrient pockets and delivering concentrated supply on demand. While you are idle, Neuromorph Nano's Writer builds a **4-level world model** (`L1 Piece -> L2 File -> L3 Module -> L4 System`) of your repository so your coding agent receives compact, cited Context Packets instead of brute-force greps.

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

| Organ | Role in the Digital Organism | Built-in Safeguard |
|---|---|---|
| **Global Context Substrate (GCS)** | Holds the studied workspace in RAM as a unified document model, symbol table, and reference graph across code, Markdown, pure-Rust PDFs, CSV/TSV, HTML/XML/JSON/YAML, logs, and in-memory zip/tar/Office/EPUB containers | Read-only sensory intake; `.nanoignore` exclusions (root and nested); language-neutral fallback lexer ensures no readable format is skipped |
| **Neuropod (8 Tentacles + Dendrite Drones)** | Eight role-agnostic tentacles (`job: "any"`, taking any lens dynamically per question across disjoint 12,000-token windows = 96k parallel tokens) with **drones as dendrites** running truncated early-layer cuts (`0..k`) by reference with ColBERT `MaxSim` late interaction and sensory adaptation | No tentacle owns weights and drones of different tentacles never exchange data directly |
| **Globalized Context Stack (GCX: `g0 -> G1 -> G2 -> G3`)** | The **spinal cord and brain stem** in RAM: `g0` per-tentacle encoder (electric-fish negative image + spinal reflexes `R1-R5`), `G1` coordinating communicator (sole bridge between tentacles, honeybee quorum & fading accumulators), `G2` bidirectional sensory-up / motor-down gate, `G3` pattern recognition & muscle memory | Enforces the `nmtype::class` sensory boundary: afferent sensory Context Packets travel up, efferent motor orders travel down; raw code never enters learned state |
| **Core Brain (Singular)** | One central brain whose regions are the four MicroModels (**Reader**, **Writer**, **Embedder**, **Reranker**) and whose parameter sets are its neurons, coordinating `Objectives`, `Rivers`, `Arbiter`, `Contradiction`, `Homeostasis` (interactive latency budget), and `Synthesis` | Re-reads cited lines fresh from disk and withholds answers instead of guessing when evidence is absent |
| **Idle Writer & 4-Level World Model** | Uses quiet moments to write `L1 Piece -> L2 File -> L3 Module -> L4 System` notes (L4 is the project outline) using a job-conditioned output head | Yields to every interactive query (`< 300 ms` ceiling) |
| **Memory, Dead-Skin Vectors & Muscle Memory** | Three distinct stores: 3-strata tiered Memory (`1-core / 2-active / 3-archive` + `T0-T3`), **Dead-Skin Biomimetic Vector Store** (`skin.json`), and cerebellar **Muscle Memory** (`.nmmusc` packs at `G3`) | Strict separation of memory, continuous learning cartridges, and behavioral-only muscle memory; nothing deleted for size |
| **Local Interface** | 11 pages at `http://127.0.0.1:4870/`: live activity, topology, anatomy, MicroModels, cartridges, parameter maps, memory, learning, storage, audit and settings | No telemetry, no outside requests of its own, and one-switch Air-Gap Mode |

---

## Modular Parameter Cartridges & Preventing Catastrophic Forgetting

Traditional continuous learning fine-tunes one monolithic weight file in place: teaching a model a new codebase, task, or preference overwrites the parameters that carried earlier skills (**catastrophic forgetting**).

**Neuromorph Nano uses a modular weight-separation and cartridge-stacking architecture developed at Gavriel Technologies.** A MicroModel's fixed neural assembly (attention, feed-forward, norm, and readout structure) is cleanly separated from its weights and specializations. Instead of mutating one giant checkpoint, weights and learned parameters are stored as **modular, hash-verified cartridges** that stream in a deterministic authority stack on top of a frozen base cartridge:

```text
  Fixed MicroModel Assembly (Reader / Writer / Embedder / Reranker)
    |
    +-- 1. Base Cartridge             [tensors]   Frozen foundation cut & restored for the role's jobs
    +-- 2. Hardware Settings          [settings]  Per-GPU precision, batching, and residency budgets
    +-- 3. Adapters & Project Lenses  [low-rank]  Unmerged y += scale * up(down(x)) per repository
    +-- 4. Job Filters & Compensation [affine]    f64-folded y = x * gain + bias at named sockets
    +-- 5. Learned Units              [units]     Compartmentalized continual learning (nano / shared / project)
    +-- 6. Preference Cartridges      [prefs]     Always plugged last: user rules, wants, and style
```

![Modular Cartridge Rack: four MicroModel decks with seated base cartridges and stacked job filters, project lenses, and adapters](docs/images/17-cartridge-rack.png)

*The 4-deck Cartridge Rack: each MicroModel's base cartridge seats in its base slot with modular job filters, project lenses, and adapters stacked unmerged on top, plus whole-engine hardware cartridges below.*

### How Cartridge Stacking Prevents Catastrophic Forgetting

- **Base parameters stay frozen during continual learning:** Base cartridge creation is strictly operational and surgical, never mutated by daily learning. All continual learning lives outside base cartridges in isolated **project adapters/lenses**, **learned units**, **job filters**, and six versioned continual records (`clc-knowledge`, `clc-procedural`, `clc-behavioral`, `clc-retrieval`, `clc-interaction`, `clc-evaluation`).
- **Unmerged streaming in stacks:** Stacked affine job filters (`nmengn-sockets/1`) at the same socket are folded on the fly in float64 into a single pass, and low-rank adapters apply unmerged after linear nodes. Swapping a job filter or stack takes effect at the next call with **no restart**; unplugging any cartridge or switching off a project compartment returns **bit-for-bit identical base outputs**.
- **Strict compartment isolation:** Learned units and project lenses belong to their own compartment (`nano`, `shared`, or a specific repository). Learning patterns in Repo A never leaks into or degrades Repo B.
- **Evidence-gated promotion with atomic rollback:** Every continual learning cycle verifies candidate updates against held-out outcomes from real coding sessions before promotion and keeps rollback snapshots (`continual/snapshots/`), so a regressing update is rejected or rolled back cleanly.
- **Untouched upstream fallback:** Optional upstream Qwen3 open weights (`weights/`) are kept strictly read-only as a reference oracle, fallback (`open` mode), and donor source for surgical add-back.

See [docs/cartridges-and-parameters.md](docs/cartridges-and-parameters.md) and [docs/continuous-learning.md](docs/continuous-learning.md) for the full design.

---

## Biomimetic Memory, Dead-Skin Vectors, Continuous Learning & Muscle Memory

Neuromorph Nano strictly separates **Memory** (what the organism retains), **Continuous Learning** (how accumulated experience improves future performance), and **Muscle Memory** (compressed behavioral reflexes), applying biological engineering principles implemented directly in the `v0.1.01` Rust engine:

### 1. Dead-Skin Biomimetic Vector Store (`dense_store.rs`, `skin.json`)
Stale embeddings from edited files, undos, and Git branch switches are managed through **epidermal desquamation and barrier homeostasis** ([docs/dead-skin-vectors.md](docs/dead-skin-vectors.md)):
- **Continuous micro-desquamation (`SHED_BATCH = 64`):** writes and syncs shed at most 64 expired dead rows at a time rather than triggering disruptive bulk compaction across the full vector store.
- **Stratum corneum barrier / undo layer (`skin.json`):** recently non-viable vectors remain structurally intact on disk (tracked by opaque 64-hex chunk hashes and 16-hex file digests only; zero code text and zero file paths). An undo, revert, or branch switch revives earlier vectors instantly with **zero GPU re-embedding**.
- **Enzymatic age-calibrated turnover kinetics (`p90 * 2`):** retention TTL (`10 min` to `30 days`) auto-calibrates from each repository's empirical revival rate and revival-age distribution (analogous to proteolytic cleavage loosening only the oldest outer layer), shrinking when revivals are rare and expanding when branch switching pays for the barrier bytes.
- **Basal-to-apical stratified segments & O(1) detachment (`NMDENSE1`):** new rows append at the newest basal 512-row segment (`vectors.bin`, `seg-000001.bin`, ...); once every row in an oldest outer segment expires, the segment file detaches whole via `remove_file` with **zero bytes rewritten**.
- **Accelerated clearance of detached tissue (`1/4` TTL):** rows belonging to deleted files shed four times faster than edited-file rows.

### 2. Tiered Memory, Synaptic Tag-and-Capture & Fresh Reconsolidation (`nmmem`)
- **Working Memory & Synaptic Tag-and-Capture (`T1` / `T0`):** `T1` working memory lives in RAM scoped per drone, tentacle, task, and brain and clears cleanly at task end. `T0` temporary notes (`temp/notes.json`) implement **synaptic tag-and-capture**: transient tags require multi-day reinforcement across **3 run-days** for late-phase consolidation into permanent episodic memory, decaying naturally after **7 unreinforced run-days**.
- **Use-Dependent Potentiation & Graded Attenuation (`1-core / 2-active / 3-archive`):** journal-sourced persistent memory (`T2` append-only episodes and `T3` per-version structural summaries) undergoes use-dependent potentiation into `1-core` on repeated recall and graded attenuation when unused (`28` unused run-days from `1-core`, `14` from `2-active`) into non-destructive `3-archive` retention. **Nothing is ever deleted for size.**
- **Write Gate & Fresh Re-Read Reconsolidation:** `nmmem::gate` blocks unverified or low-confidence writes (`require_evidence: true`, logged to `refused.jsonl`). At recall, past questions are matched only against new questions, and cited lines are **always re-read fresh from disk** so stale code is never parroted.

### 3. Online Plasticity, Offline Consolidation & Learning Supervisor (`nmlearn`)
- **Two biological learning cycles across three levels (Experiential, Knowledge, Behavioral):** **Cycle A (online short-term adaptation)** captures explicit corrections, implicit build/test outcomes, internal feedback, and delayed session verdicts through the **Evidence Ledger** without interrupting active work; **Cycle B (offline systems consolidation at idle)** extracts generalized patterns, procedures, and scoped behavioral policies into **6 versioned continual learning cartridges** (`clc-knowledge`, `clc-procedural`, `clc-behavioral`, `clc-retrieval`, `clc-interaction`, `clc-evaluation`), alongside idle project adapter/lens fitting and stated preference cartridges.
- **Learning Supervisor:** every candidate moves through `observed -> candidate -> under_evaluation -> validated | rejected` behind a held-out outcome prediction gate with atomic rollback snapshots (`continual/snapshots/` and `learning/rollback/`).

### 4. Immune Negative Selection & Cerebellar Muscle Memory (`nmmusc` at `G3`)
- **Efferent behavioral-only reflex store:** lives in `muscle/` (`<fingerprint>.nmmusc`), strictly separate from memory, model weights, and cartridges, enforcing the afferent/efferent boundary (`nmmusc-sensory-refused`).
- **Negative selection, Welch promotion & behavioral extinction:** `G3` applies **immune-style negative selection** to reject self-confounded loops (outcomes triggered by the organism's own primes or edits), compresses recurring procedural gains with a Stitch-style motif miner gated by **Welch's `z = 1.96`**, and monitors every promoted reflex pack with **Page's CUSUM** to drive automatic **behavioral extinction** whenever real-world gain fades.

### 5. Biomimetic Signal Processing in the Nervous System
- **Dendritic Early-Layer Sensing (`drone_dendrite.rs`, on by default):** runs early-layer (`0..k`) receptive-field cuts of the Reader MicroModel shared by reference, scoring pieces in RAM via ColBERT-style `MaxSim` late interaction and request-to-piece layer-`k` attention with position-0 attention-sink exclusion and posed-minus-content-free baseline normalization (**sensory adaptation**).
- **Afferent/Efferent Spinal Cord & Brain Stem (`GCX`):** `g0` cancels predictable background fields via Huber residualisation (**mormyrid electric-fish negative image**) and executes fast local **spinal reflex arcs (`R1-R5`)** installed by `G2`; `G1` integrates independent evidence across all 8 role-agnostic tentacles via fading accumulators and **quorum sensing** (**honeybee quorum**); **Homeostasis** regulates interactive latency bounds (`TIME_BUDGET`).

---

## Precision Parameter Mapping, Surgery & Live Editing

Neuromorph Nano approaches model specialization with the engineering discipline of **functional brain mapping and stereotactic neurosurgery**: map the exact circuits responsible for each capability before intervening, protect critical pathways at exact coordinates, verify every change against held-out checks, and preserve full reversibility.

### 1. Functional Parameter Mapping & Bidirectional Cartridge Surgery
- **Coordinate-level functional mapping (Wanda + OWL):** Before modifying any parameter, Neuromorph Nano runs calibrated task prompts through the untouched reference open weights to measure per-weight importance (`s_t(w_ij) = |w_ij| * ||x_j||_t`) across every attention and feed-forward linear projection (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). Outlier-Weighted Layerwise sparsity (`OWL`) allocates per-layer retention budgets inversely to activation outlier ratios so sensitive layers remain denser while meeting the global parameter budget.
- **Eloquent-pathway protection (`super-weights.json` & pinned heads):** Just as neurosurgical mapping marks critical regions that must remain intact, Neuromorph Nano records exact `[row, col]` high-leverage super-weight coordinates and pins tentacle relevance heads so pruning and cascade surgery never sever essential pathways.
- **Bidirectional surgical intervention (pruning + surgical add-back):** Unlike destructive one-way pruning, surgery is bidirectional. Unused parameter coordinates are removed (`M_next = M_current AND NOT M_unused`), and if a held-out evaluation gate detects regression on any job, Neuromorph Nano computes the exact missing coordinate set (`Delta = M_failing AND NOT M_current`) and performs **surgical add-back** from the untouched reference open weights into cartridge version `N+1`, logging parent provenance and SHA-256 checksums.
- **Hardware-aware sparse execution:** On consumer GPUs without hardware sparse tensor cores, `kernels/cukern/bmq8.cu` compresses 8-bit linear tensors with >= 20% exact zeros into packed bitmask blocks decompressed inline with `__popc()`, reducing VRAM footprint and memory bandwidth while keeping disk cartridges at full shapes.

![Precision Parameter Mapping: BF16 heatmap of model.layers.0.mlp.down_proj.weight with 128/128 pinned super-weight coordinates, floating block Inspector, and per-layer L0-L27 switches](docs/images/14-superweights-map.png)

*Coordinate-level functional mapping: whole-tensor BF16 heatmap of `model.layers.0.mlp.down_proj.weight` (`3,145,728` weights, `15%` zeros) with `128/128` pinned `super-weights.json` coordinates overlaid in gold, the floating Parameter Inspector open on a selected 49-weight block (`rows 457:464, cols 1197:1204`) with inline block editing, and per-layer `L0..L27` switches on the right.*

### 2. Whole-Set Parameter Viewer, Reversible Live Overlays & Direct Editing
Every parameter-holding object (**base cartridges**, **job filters**, **compensation/socket cartridges**, **adapters and project lenses**, **learning units**, **preference cartridges**, and **optional upstream open weights**) is governed by one unified inspection and editing engine (`nm-parameters/1`, `crates/nmcore/src/bin/pedit.rs`):

- **Chunk-streamed whole-set imaging:** Inspect every tensor grouped by layer and block (`attention`, `mlp`, `norm`, token table) via interactive heatmaps with signed (`diverging`) or magnitude (`sequential`) scales, index rulers, distributions, and `mean / min / max / rms` summaries. Regions stream in `<= 4 MiB` chunks so even the `151,669 x 1,024` token embedding matrix is never loaded whole into host RAM.
- **Map overlays on live tensors:** Project `super-weights.json` directly onto heatmaps as gold `[row, col]` coordinate pins, or highlight layer coverage from job filters and OWL maps, with `on / emphasize / hide` visual controls.
- **Floating, movable, resizable Parameter Inspector:** Click any individual weight or multi-weight block to inspect exact stored values, coordinates, tensor percentile, and paired parameter-map readings; double-click any cell or block to open and focus manual inline editing directly.
- **Addressable coordinate operations with norm-diff previews:** Target `*`, whole layers (`L<n>`), individual tensors, rows, columns, channels, attention heads, or `[r0:r1, c0:c1]` blocks with `set`, `add`, `mul`, `scale`, `clamp`, `zero`, `identity`, `negate`, `paste`, `import`, `export`, `copy`, `interpolate`, `blend`, `drop`, `add_point`, and truncated-SVD `rank` reduction. Every operation computes a dry-run diff (`changed` count, L2 norm delta, max delta, sampled old vs new values) before writing.
- **Reversible live overlays & per-layer controls:** Like reversible functional silencing during surgical mapping, every **edit**, **mask** (exact residual identity pass), or **freeze** (read-only protection lock) can be applied in a live overlay (`memory/overlays/`) without modifying the underlying cartridge or open-weight files, or written as a new trial version beside the source. Engineers can toggle individual MicroModel layers on or off directly from the **Models** page or inside the Parameter Viewer. Every change is audited with one-click Undo and shares identical parity across the UI, MCP (`nm_cartridges`), HTTP API, and CLI.

![1:1 Exact Square Parameter Grid with Pinned Super-Weights and Floating Inspector](docs/images/15-parameter-inspector-1to1.png)

*Zoomed-in `1:1 EXACT` (`70x`) view of `model.layers.0.mlp.down_proj.weight`: individual BF16 parameters rendered as square cells with numeric values, three gold-outlined pinned super-weights visible across row 712 (`3/128 pts`), and the floating Parameter Inspector locked on `row 712, col 677` showing `★ super-weights · PINNED`, exact value (`0.010925293`), tensor percentile (`66.8%`), and inline reversible overlay editing (`set`, `mask`, `freeze`, `unmask`).*

![MicroModels Per-Layer On/Off Controls](docs/images/16-layer-controls.png)

*Per-layer controls (`L0..L27`): toggle any MicroModel layer on or off through the reversible parameter overlay directly beside live layer activity and base-cartridge controls.*

---

## Proactive Priming, Primes Studio & Agent Guidance

Beyond answering explicit MCP queries, Neuromorph Nano can **proactively nudge** your coding agent (Claude Code via `additionalContext`, Antigravity via `injectSteps`) through lifecycle hooks, **Primes Studio**, and `nm_goal`, keeping the agent aligned with verified repository history, open objectives, dependency graphs, timed check-ins, and passing checks.

### How Priming Works

1. **Opt-in silence by default (`priming = off`):** Both the master priming switch (`off | quiet | on`) and all six built-in verified rules start `off` so nothing injects into your agent until you explicitly enable it.
2. **Fresh-on-disk verification & latency guard:** Before any built-in prime fires, Neuromorph Nano re-reads its cited lines fresh on disk (dropping stale primes automatically), guarantees the same verified `(kind, source)` never repeats twice in a session, and enforces a **self-calibrating latency deadline** (`2x` the host's median hook execution time, `25 ms` fallback) so hooks never stall your agent.
3. **One live-synced control across Web UI, HTTP API, and MCP:** Every switch, custom prime, preset, and undo shares the exact same atomic per-project files (`memory/projects/<key>/primes-config.json`, `memory/prime-presets.json`, `settings.json`, and `memory/audit.jsonl`) across **Audit -> Primes Studio**, the right-side **Quick prime control** rail, `/api/primes`, and MCP `nm_settings` (`primes`, `prime_builtin`, `prime_add`, `prime_edit`, `prime_toggle`, `prime_remove`, `prime_all_off`, `prime_preset_*`). Changing a prime from MCP, the API, or the browser updates every surface immediately with zero conflicts and full one-click Undo.

### 1. Six Verified Built-In Prime Rules (`prime.rs`, `primes_ctl.rs`)

| Verified Prime | Lifecycle Trigger | What It Nudges the Agent With |
|---|---|---|
| **`prompt-recall`** | `UserPromptSubmit` / `PreInvocation` | Surfaces verified past answers matching the prompt, with cited lines re-read fresh. |
| **`failure-fix`** | `PostToolUseFailure` / failed build or test | Surfaces the verified fix lines when the same error code and file signature recur. |
| **`edit-impact`** | `PreToolUse` (`Edit`, `MultiEdit`, `Write`, `NotebookEdit`) | Warns before modifying lines that hold a verified fix, sit under a refuted outcome, or change a function signature with live callers elsewhere in the package. |
| **`resume`** | `SessionStart` (after compaction, or startup with `priming = on`) | Restores active files, failing checks, and open work from the session's situation record so context compaction never causes amnesia. |
| **`core`** | `SessionStart` (when `resume` has nothing to say) | Surfaces the repository's most-used `1-core` memories at session open (holding back any whose lines changed). |
| **`open-loop`** | `Stop` (`priming = on`) | Flags files edited during the turn that were not built or tested since, or are still failing. |

### 2. Primes Studio: Custom Triggers, Timers, Schedules & Presets

Open **Audit -> Primes Studio** (`/audit/#primes`) or the right-side **Quick prime control** rail on Audit and Learning:

- **Alarm-style switches (ON/OFF separate from Add/Remove):** Every prime starts `off`; turning a switch off keeps the prime in your list until you explicitly remove or delete it, and **All off** (`all_off`) disarms everything in one click.
- **Tabbed `Choose a preset` pop-up inside `+ Add Prime`:** Pick from **Verified presets** or **Prompt & schedule** starter and user-saved presets (`quiet-verified`, `active-guardrails`, `build-discipline`, `compaction-shield`, `workday-checkin`, `pre-edit-guard`) to prefill clean, ready-to-edit prompts, triggers, and schedules without cluttering the form.
- **Custom event, interval, and UTC schedule primes:** Trigger on lifecycle events (`session-start`, `prompt`, `compact`, `stop`, `edit`, `failure`, `tools`, `error-repeats`), minute-based **interval** timers (`interval_min`), or UTC day/clock **schedule** windows (`days`, `after`, `before`), with optional file-path globs, session repetition caps (`always`, `once`, `max`), and links to Guidance loops.

![Primes Studio: + Add Prime menu open above active verified primes, with the right-side Quick prime control rail](docs/images/19-primes-studio.png)

*Primes Studio (`/audit/#primes`): `+ Add Prime` sheet open with `Browse presets` (`Verified presets` and `Prompt & schedule` pop-up), alarm-style ON/OFF switches below, and the right-side Quick prime control rail.*

### 3. Directed Agent Guidance: Loops, Step Graphs, Objectives & Reminders (`guidance.rs`, `nm_goal`)
Configured from **Memory -> Loops and graph**, the CLI (`neuromorph-nano guidance`), HTTP API (`/api/guidance`), or by the agent itself over MCP (`nm_goal`), riding on the same hook channel and scored by the **Evidence Ledger**:
- **Step Graphs:** directed task graphs (standalone or attached to an objective) with AND-joins (`depends_on`), OR-joins (`depends_any`), cycle rejection, probe steps (`probe: true`), automatic completion when a step's build/test check passes after edits under its path globs, and automatic next-ready-step nudges at prompts and stops.
- **Completion Loops (Claude Code):** keep the agent continuing at `Stop` until a verified condition holds (`check-passes`, `goal-done`, `steps-done`, `manual`, `all-of`, or `any-of`), guarded by iteration caps (default 3, max 20), consecutive stall detection (`stalled`), and optional wall-time (`budget_ms`) or tool-call (`budget_tools`) limits.
- **Trigger Reminders:** scoped notes delivered whole at `session-start`, `prompt`, `compact`, `stop`, `edit` (glob-matched), `failure`, `tools` (every Nth tool call), or `error-repeats` (when the same error code or message recurs 2 to 5 times in a session).

Every prime and guidance delivery is tracked without storing file contents (with secret-dropping on hook logs) and judged by the **Evidence Ledger** so the interface shows what your agent actually followed.

---

## Complete Feature Set

| Subsystem | Engineering Capabilities | Control Surface |
|---|---|---|
| **Symbiotic Codebase Recall, GCS & 4-Level World Model** | Pure-Rust **Global Context Substrate (GCS)** across code (with language-neutral fallback), Markdown, in-memory PDFs, CSV/TSV, JSON/YAML/HTML/XML, logs, and zero-extraction zip/tar/Office/EPUB containers; 8 role-agnostic tentacles (96k parallel tokens) + **dendrite drones** (`0..k`, `MaxSim`, sensory adaptation); **GCX** (`g0 -> G1 -> G2 -> G3` with electric-fish negative image, spinal reflexes `R1-R5`, honeybee quorum); singular **Core Brain** returning cited Context Packets with fresh re-read & decoy withholding; idle Writer (`L1-L4` world model, job-conditioned output head, `< 300 ms` yield). | MCP (`nm_context_pack`, `nm_repo_map`, `nm_symbol_find`, `nm_deps`, `nm_view`, `nm_sync`), UI |
| **Modular Parameter Cartridges & Anti-Forgetting Stacking** | Gavriel Technologies architecture separating fixed MicroModel assemblies from modular weight cartridges; deterministic runtime stacking (`base -> hardware -> adapters/lenses -> job filters -> learned units -> preferences`) with unmerged application, `f64`-folded affine sockets, compartment isolation, instant swap without restart, and bit-identical rollback when unplugged. | UI (**Cartridges**), MCP (`nm_cartridges`), API, CLI |
| **Biomimetic Memory, Dead-Skin Vectors & Muscle Memory** | **Dead-Skin Biomimetic Vector Store** (`skin.json`: continuous micro-shedding, zero-GPU undo/branch-switch revival barrier, enzymatic `p90 * 2` age timing, O(1) segment drop, 4x faster deleted-file shedding); synaptic tag-and-capture `T0` + `T1` working RAM + `1-core / 2-active / 3-archive` strata (never deleted for size) + Write Gate; immune negative-selection & cerebellar **Muscle Memory** (`G3`, `.nmmusc`, Welch `z = 1.96`, Page's CUSUM). | UI (**Memory**, **Storage**), MCP (`nm_memory`, `nm_comprehension`), API |
| **Precision Parameter Mapping, Surgery & Built-In Research Lab** | Wanda + OWL per-weight functional mapping, pinned `super-weights.json` coordinates, bidirectional pruning and surgical add-back, `bmq8.cu` bitmask-sparse GPU execution, chunk-streamed whole-set heatmaps, floating movable/resizable Inspector, per-layer on/off switches, reversible `edit / mask / freeze` overlays, and the built-in `nmlab` (`neuromorph-nano lab`) research and surgery harness. | UI (**Parameters**, **Models**), MCP (`nm_cartridges`), API, CLI (`lab`) |
| **Continuous Learning, Evidence Ledger & Distillation** | Awake (`Cycle A`) and idle (`Cycle B`) continuous learning with **Learning Supervisor**, 6 versioned continual learning cartridges (`clc-*`), closed-loop Evidence Ledger, split-gate validation with atomic rollback snapshots, curated and live teacher distillation (`adapter`, `filter`, `readout`, `unit`), and dynamic dense-index following that automatically rebuilds project vectors at idle whenever Embedder weights or overlays change. | UI (**Learning**, **Audit**), MCP (`nm_learning`, `nm_feedback`, `nm_unit`), API, CLI |
| **Stated Preference Cartridges** | Natural-language preference parser supporting 5 languages (English, Spanish, French, German, Portuguese): enforced text transformations, checked code/style rules, bounded logit-bias wanted terms with per-rule strength (`0.1-2.0`), and scoped style guidance (global, project, file extension, or folder), always plugged last in the stack. | UI (**Learning -> Preferences**), MCP (`nm_learning`), API, CLI |
| **Proactive Priming, Primes Studio & Build Verification** | Default-off opt-in priming across 6 verified lifecycle rules (`prompt-recall`, `failure-fix`, `edit-impact`, `resume`, `core`, `open-loop`) with fresh-on-disk verification and self-calibrating latency deadlines; **Primes Studio** alarm-style switches (ON/OFF independent of Add/Remove), tabbed **Choose a preset** pop-up (Verified + 6 Prompt & schedule starter presets + user-saved presets), custom event/interval/UTC-schedule triggers, right-rail quick control, live synced state across UI/API/MCP with Audit Undo; directed **Step Graphs**, **Completion Loops**, **Trigger Reminders**, and offline Cargo build help, Step Check, and improvement scanning (`nm_build_run`, `nm_build_explain`, `nm_stepcheck`, `nm_verify`, `nm_improve_scan`, `nm_patch_propose`). | UI (**Audit -> Primes Studio**, right-side **Quick prime control**, **Memory -> Loops & graph**), API (`/api/primes`), MCP (`nm_settings`, `nm_goal`, `nm_build_*`, `nm_verify`, `nm_stepcheck`), Hooks, CLI |
| **Global Project Management & `.nanoignore` Privacy** | Globally synced active project menu across every page (switch, add custom/folder-backed workspaces, edit display name/description, two-step double-confirmed delete without touching source code) plus centered **Project Settings** modal with live per-project prime switches and `.nanoignore` quick-exclude; root and nested `.nanoignore` exclusions (gitignore syntax) across GCS, substrate cache, dense vectors, idle Writer, recall, build help, hooks, and primes with instant cache invalidation and Audit Undo. | UI (**Settings -> Projects**, **Settings -> Privacy & network**, global project dropdown), API (`/api/projects`, `/api/nanoignore`), MCP (`nm_settings`) |
| **Hugging Face Cartridge Libraries & Download Manager** | Built-in public official library (`gavriel-tech/neuromorph-nano-cartridges`, no token required) + third-party Hugging Face sources; floating movable/resizable **HF downloads** window covering base cartridges, job filters, parameter maps, shared stacks, and optional open weights with category/role/search filters, explicit what-it-maps descriptions, `archived/` hiding, SHA-256 verification, two-step delete, and confirmation-gated **Publish to Hugging Face** (`source_publish`). | UI (**Cartridges**, **Models**, **Parameters**), MCP (`nm_settings`), API |
| **Zero-Dependency Runtime, Hardware Scaling & Privacy** | 21 `std`-only Rust crates (zero external crates, `unwrap` denied, `unsafe` confined to `nmffi`), 12 prebuilt multi-arch CUDA fatbins (`sm_75`-`sm_90` + PTX) with AVX2/FMA/F16C CPU fallback, zero-VRAM speculative decoding, per-card `dp4a` auto-tuner, multi-session loopback engine sharing, idle KV-cache release, hard VRAM/RAM/CPU/memory-area caps (`content`, `caches`, `weights`), user-owned backup/restore repositories, opt-in paired LAN tablet access, and verifiable one-switch Air-Gap Mode. | Engine, UI (**Settings**, **Storage**), MCP (`nm_settings`, `nm_storage`, `nm_status`, `nm_live`) |

---

## Release Status, Hardware, and System Requirements

**`v0.1.01` is the current public prototype release of Neuromorph Nano** (Windows 10/11 x64), built on a 4 GB GPU. It ships the complete end-to-end local organism: zero-dependency Rust runtime, 12 custom multi-arch CUDA fatbins, four coexisting MicroModels seated on base cartridges with dynamic VRAM/RAM scaling, 25 MCP tools, parameter mapping and cartridge surgery, public Hugging Face parameter-map/filter downloads and opt-in publishing, continuous learning primitives, and the 11-page local observability interface. Formal cross-hardware benchmark reports will be published as the platform matures beyond this initial prototype.

### System Requirements (`v0.1.01`)

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
.\tools\install.ps1 -LocalZip .\neuromorph-nano-v0.1.01-win-x64.zip -ModelsFrom E:\nano\weights -CartridgesFrom E:\nano\library
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
| **Engine** | **Models**: the four MicroModels, what each runs on, per-layer on/off switches, base cartridge choice, routing, the floating **HF downloads** window, and Hugging Face library browsing, `fetch all`, and opt-in publishing. **Cartridges**: one deck per MicroModel with its base cartridge seated and its cartridges docked to plug, unplug, inspect parameters or change version; the floating **HF downloads** manager (browse everything Hugging Face offers, check items to download, or safely delete downloaded items), plus the cartridge library, base versions, Downloads, Sources, and learned units in the side panel. **Parameters**: the map atlas and universal Parameter Viewer (clickable layer bars, base cartridges and filters, paired parameter-map overlays, floating movable/resizable Inspector with inline manual overlay editing, floating **HF downloads**, and the 4-step Build panel), with each map's lineage, archive and restore. **Memory**: memory tiers, the world model and recall. **Learning**: learning cycles and rollback, the evidence ledger and the training data. |
| **System** | **Storage**: disk use by category, with safe pruning. **Audit**: Audit Log timeline, **Primes Studio** (`/audit/#primes`), Episodes, right-side **Quick prime control** rail, inline question show/hide, and one-click undo. **Settings**: Hardware limits, **Projects** (globally synced workspaces & centered **Project Settings** modal), Agents & tools, Controls, **Privacy & network** (Air-Gap Mode + **`.nanoignore`** editor & path tester), and Stage times. |

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
| ![Primes Studio](docs/images/19-primes-studio.png) **Primes Studio**: `+ Add Prime`, tabbed preset picker, alarm-style switches, and Quick prime control rail. | ![Privacy & .nanoignore](docs/images/22-nanoignore.png) **Privacy & `.nanoignore`**: one-switch Air-Gap Mode and interactive `.nanoignore` editor with live path tester. |

### Cartridge Libraries

The official cartridge library is on Hugging Face at [`gavriel-tech/neuromorph-nano-cartridges`](https://huggingface.co/gavriel-tech/neuromorph-nano-cartridges) (public, no token required). You can also add third-party Hugging Face libraries as cartridge sources, or publish your own job filters, parameter maps, and non-base cartridge stacks directly to your own Hugging Face repository (or as a Pull Request to the official library) via **Publish to Hugging Face** (`source_publish`). Nothing is fetched or uploaded until you ask; everything fetched goes through the same checks as the cartridge library on this machine (stacks arrive on trial, an existing map is never replaced, base cartridges come from the official library only, and decommissioned files under `archived/` are hidden from the downloader). See [docs/cartridges-and-parameters.md](docs/cartridges-and-parameters.md) and [docs/build-flow.md](docs/build-flow.md).

![Floating Hugging Face Download Manager over the Cartridge Rack](docs/images/18-hf-download-manager.png)

*The floating Hugging Face Download Manager: inspect everything in `gavriel-tech/neuromorph-nano-cartridges` (base cartridges, job filters, parameter maps, shared stacks, and optional open weights) with explicit descriptions, category/role filters, multi-select downloads, and safe two-step deletion.*

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
- [Cartridges & Parameters](docs/cartridges-and-parameters.md) · [Build Flow](docs/build-flow.md) · [Continuous Learning](docs/continuous-learning.md) · [Dead-Skin Vector Store](docs/dead-skin-vectors.md)
- [Agents and models](docs/agents.md) · [Configuration](docs/configuration.md) · [Interface API](docs/interface-api.md) · [Privacy](docs/privacy.md)
- The full index is [docs/README.md](docs/README.md).

</details>

---

## Privacy, Air-Gap, and User Control

- **The Nano sends nothing off your machine on its own (and what your coding agent sends):** The engine sends no telemetry and makes no network request of its own. **Your coding agent is separate:** any Context Packet, MCP tool reply, or lifecycle prime the Nano hands to your agent goes to whichever model backs your agent (with a cloud-backed agent like Claude Code, Antigravity, Cursor, VS Code, or Windsurf, that is its cloud provider). Run a local model behind your agent with Air-Gap Mode on for nothing to leave at all (see [Agents and models](docs/agents.md)). The MCP server talks to your agent over standard input and output; the interface binds `127.0.0.1`. The only outbound requests the Nano itself makes are downloads, library listings, and confirmation-gated **Publish to Hugging Face** uploads you explicitly trigger, made over Windows' native HTTPS (WinHTTP, with `curl.exe` fallback) with every downloaded file verified against its SHA-256.
- **Explicit file exclusions (`.nanoignore`):** Manage exclusions manually via `<repo>/.nanoignore` (and nested subdirectory `.nanoignore` files), in the local Interface (**Settings -> Privacy and network -> File exclusions**), over the local JSON API (`GET/POST /api/nanoignore`), or via MCP (`nm_settings` `op=nanoignore | nanoignore_check | nanoignore_set | nanoignore_add | nanoignore_remove`, audited with one-click Undo). Using gitignore-style patterns (`*.pem`, `secrets/`, `/root_only.txt`, `!reinclude.rs`), `.nanoignore` applies in every repository, invalidates `substrate-cache-*.bin` immediately when changed, and stops Neuromorph Nano from reading, embedding, writing world-model notes for, priming from, returning build-help source snippets for, or including matching files in Context Packets handed to a cloud coding agent like Claude Code. **Transparent boundary:** `.nanoignore` governs what Neuromorph Nano reads and sends; your coding agent's own direct file/shell tools remain controlled by your agent's own ignore and permission settings.
- **Verifiable Air-Gap Mode:** Turn on **Air-Gap Mode** in Settings, over MCP (`nm_settings op=set key=air_gap value=on`), or lock it for the process with `$env:NANO_AIR_GAP = "on"`. Air-Gap Mode keeps the interface on `127.0.0.1`, turns off local network access and tablet pairing, and refuses every download and remote listing. `GET /api/airgap` and `nm_settings op=airgap` show its state live.
- **Hardware limiters:** Cap GPU VRAM (`limits.vram_mb`), host RAM (`limits.ram_mb`) or background threads (`limits.cpu_threads`) in Settings or over MCP, or apply the recommended values detected for your hardware in one click.
- **Transparent storage:** See the exact paths and sizes of memory, dense vectors, quantized copies and cartridges in Storage or `nm_storage`, and prune rebuildable caches whenever you want.
- **Honest local disk footprint:** With `memory = off`, workspace reads stay strictly in RAM. With `memory = on`, your local `memory/projects/<repo>/` folder holds rebuildable restart caches (`substrate-cache-*.bin`, which stores derived text for fast restarts and can be pruned in **Storage** or disabled with `NANO_SUBSTRATE_CACHE=off`) and idle Writer summaries (`world-model/notes.jsonl` when `writer.idle = on`), while the dense vector index (`dense/`, `skin.json`) stores only vectors and opaque hashes, episodes store citations and line hashes rather than copied files, and `memory/audit.jsonl` never holds file contents. Files matched by `.nanoignore` never enter any of these stores.

![Privacy & Network: Air-Gap isolation and interactive .nanoignore editor with live path tester](docs/images/22-nanoignore.png)

*Settings -> Privacy & network (`/settings/#privacy`): verifiable one-switch Air-Gap Mode and the interactive `.nanoignore` editor with live path testing, quick-add exclusion patterns, instant substrate cache invalidation, and Audit Undo.*

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

`v0.1.01` proves the core architecture end to end on Windows x64 (NVIDIA CUDA and CPU fallback). From here, we are pushing the frontier of local, modular cognitive systems and actively welcome collaborators who want to expand platform reach, hardware efficiency, and intelligence:

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
