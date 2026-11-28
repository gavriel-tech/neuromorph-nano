# Configuration

Neuromorph Nano is configured in three places: the controls in `settings.json`, the design files in
`designs/`, and command-line flags. A command-line flag wins over a saved control.

## Controls (`settings.json`)

`settings.json` sits in the Nano folder and is written by the interface (Settings page, and the
controls shown on other pages), by MCP (`nm_settings` `op: set`) and by the API (`POST
/api/controls`). Every change is logged in `memory/audit.jsonl` and can be undone from the Audit
page. A missing or unknown value falls back to the default. The full list, with each control's
help text, is `crates/nmcore/src/bin/controls.rs`.

From the command line, the same MCP op runs once with `call`:

```
neuromorph-nano call nm_settings "{\"op\":\"set\",\"key\":\"<key>\",\"value\":\"<value>\"}" --root <repo>
```

`{"op":"view"}` lists every control and `{"op":"saved"}` the whole file, key by key. In the
interface, the Settings page's side panel (Guide) carries the same reference: every control with
its key, the values it takes, its default and when a change applies, and for each section the
API route, MCP op and command that read or set the same thing.

The four precision controls name a MicroModel by its role; in brackets, the upstream open weights
the MicroModel is built from.

### Precision

| Key | Values | Default | Applies |
|---|---|---|---|
| `precision.base` | `default`, `16`, `8`, `4` | `default` | Next start. The Reader (Qwen3 0.6B), with the drones and the judge that run on it |
| `precision.writer` | same | `default` | Next start. The Writer (Qwen3 1.7B) |
| `precision.reranker` | same | `default` | Next start. The Reranker (Qwen3 Reranker 0.6B) |
| `precision.embedder` | same | `default` | Next start. The Embedder (Qwen3 Embedding 0.6B) |

`default` keeps the engine's own width for that model. Precision is the bits per weight the model
is loaded at; the files on disk never change.

### Budgets and allowances

| Key | Values | Default | Applies |
|---|---|---|---|
| `budget.pack_tokens` | integer `1000` to `16000` (step `250`; presets `1000`, `2000`, `4000`, `6000`, `8000`, `12000`, `16000`) | `4000` | Next call. Default token budget of `nm_context_pack` when a call does not pass `tokens` |
| `budget.tentacle_tokens` | integer `2000` to `32000` (step `500`; presets `2000`, `6000`, `12000`, `18000`, `24000`, `32000`) | `12000` | Next call. Chunk tokens each tentacle window holds when composing its lens territory |
| `budget.judge_window` | integer `128` to `2048` (step `64`; presets `128`, `256`, `384`, `512`, `768`, `1024`, `2048`) | `384` | Next call. Tokens of each candidate piece the judge reads around the question's evidence in full mode |
| `budget.fast_k` | integer `1` to `16` (step `1`; presets `1`, `2`, `4`, `6`, `8`, `12`, `16`) | `4` | Next call. Top candidates from the stack that the Reranker judges in fast mode |
| `budget.max_run_s` | integer `5` to `180` (step `5`; presets `15`, `30`, `60`, `90`, `120`, `180`) | `60` | Next call. Most wall-clock seconds for one context pack run |
| `budget.gpu_mb_max` | `auto`, or integer `1024` to `81920` MiB (step `128`; presets `2048`, `3072`, `4096`, `8192`, `16384`, `24576`) | `auto` | Next call. Card memory above which homeostasis narrows neural ordering and suspends idle slots. `auto` follows this machine's card: its total memory less the live calls' measured room |
| `budget.adaptive_pack` | `on`, `off` | `on` | Next call. On: when a call omits `tokens` and the top reranked piece dominates with high confidence, the pack is capped at `budget.focused_tokens` instead of filling `budget.pack_tokens` with lower-ranked pieces |
| `budget.focused_tokens` | integer `500` to `8000` (step `250`; presets `1000`, `1500`, `1800`, `2500`, `4000`) | `1800` | Next call. Token ceiling when `budget.adaptive_pack` finds a focused answer |
| `rerank.early_drop` | `off`, `balanced`, `aggressive` | `balanced` | Next call. Mid-layer candidate pruning inside the batched Reranker pass (including the close-call judge) when `--rerank-drop` is not given. `balanced`: at layer 18, drops candidates more than 4.0 log-odds behind the leader, keeping at least the top 5. `aggressive`: layer 16, margin 3.0, keep 3. `off`: every candidate runs all layers |
| `writer.idle` | `on`, `off` | `on` | Next start. On: when the server is quiet and memory allows, the Writer summarizes served and contested pieces at idle (the world-model notes), yielding at once when a request arrives or the server exits. Off: no background notes |
| `quota.session_tokens` | integer `0` to `500000` (step `5000`; presets `0`, `25000`, `50000`, `100000`, `250000`) | `0` | Next call. Token quota for context packs in one session (`0`: unlimited). At 80% packs scale down to the focused allowance; at 100% `quota.action` applies |
| `quota.daily_tokens` | integer `0` to `2000000` (step `25000`; presets `0`, `100000`, `250000`, `500000`, `1000000`) | `0` | Next call. The same over 24 hours |
| `quota.session_packs` | integer `0` to `500` (step `5`; presets `0`, `20`, `50`, `100`, `250`) | `0` | Next call. Most context pack calls in one session (`0`: unlimited) before `quota.action` applies |
| `quota.action` | `compact`, `fast`, `withhold` | `compact` | Next call. At a reached quota: `compact` scales each pack to a compact focused budget; `fast` also switches to fast mode; `withhold` withholds uncached packs with a quota notice |

### Memory and privacy

| Key | Values | Default | Applies |
|---|---|---|---|
| `memory` | `on`, `off` | `on` | Next start. Off: nothing recalled and nothing written to memory |
| `dense_cache` | `on`, `off` | `on` | Next start. On, with memory on: the meaning index's vectors (a content hash and 1,024 numbers per piece, never text or file names) are kept in the project's memory folder, `dense/vectors.bin`, so a restart embeds only changed pieces. Off: neither read nor written, nothing deleted. See `privacy.md`, Meaning index on disk |
| `recall_cache` | `on`, `off` | `on` | Next call. On: recall-by-meaning episode query vectors, decoy calibration floors and Reranker pair scores (hashes, vectors and numbers only, never text) are kept in the project's memory folder under `meaning/`. Off: kept in memory for the current process only |
| `prompts` | `keep`, `off` | `keep` | Next prompt. Keep: the text of your prompts, seen through the hooks, is kept in the project's memory folder (`prompts.jsonl`) after anything that looks like a secret is dropped. Off: only each prompt's length and a hash. See `privacy.md`, Hooks |
| `prompts.days` | integer `1` to `3650` (presets `1`, `7`, `30`, `90`, `365`) | `30` | Next prompt. Kept prompt text older than this is removed at the next `situation` read, and at a prompt when the last removal was more than an hour ago |
| `audit.keep_days` | `7`, `28`, `90`, `365` | `28` | Next trim. Days of use (days the Nano ran) the audit log keeps its tool-call rows when you trim it (Audit page, or `keep --apply`). Setting changes and every other row are always kept, so undo keeps working |

### Learning

| Key | Values | Default | Applies |
|---|---|---|---|
| `learning` | `on`, `off` | `on` | Next call. On: the continuous learning engine records outcomes, corrections and preferences, extracts patterns and procedures, and adapts retrieval under its supervisor's checks (`continuous-learning.md`). Off: no learning observations or adaptations |
| `learning.cycle` | `auto`, `immediate`, `manual` | `auto` | Next call. Auto: each judged outcome is recorded as an experience as it comes, and a learning cycle (patterns, procedures, policies and the preference cartridge, each kept only if it beats the current one on held-out outcomes) runs once enough verified examples have settled and the server has been quiet a while. Immediate: experiences recorded as they come; a cycle only when you ask. Manual: a cycle only when you ask from the Learning page |
| `examples` | `off`, `features`, `full` | `features` | Next call. The training store (`training/` in the project's memory folder), kept for fitting cartridges later. Features: places, content hashes, labels, the Reranker's scores and provenance, no question or passage text. Full: also the question and each judged passage's text, secrets dropped. Off: nothing. Nothing with memory off. See `privacy.md`, Training store |
| `muscle` | `on`, `off` | `on` | Next start. On: G3 captures repeating positive decision patterns into the separate muscle memory store (never touching base cartridges) and runs bounded idle replays. Off: G3 and muscle memory off |

### Hardware limits

| Key | Values | Default | Applies |
|---|---|---|---|
| `limits.vram_mb` | `auto`, `0`, or a number of MiB (step `128`; presets `2048`, `3072`, `4096`, `8192`, `16384`) | `auto` | Next call (`0`: next start). The most card memory the Nano keeps (its MicroModels, window copies and buffers); under a limit it tiles batches smaller and evicts idle MicroModels. A serving process loads all four MicroModels at start, the Writer only when this memory holds it beside the other three (else it loads per idle session and stays seated once room frees); `nm_status` `micromodels` says which are held and why. `auto` uses what the driver reports free. `0` runs every MicroModel on the processor |
| `limits.ram_mb` | `auto`, or a number of MiB (step `512`; presets `4096`, `8192`, `12288`, `16384`, `24576`, `32768`) | `auto` | Next call. The most host memory a serving process holds: its kept caches (encoded chunks, relevance keys) grow only within it and only while the machine keeps a quarter of its memory available (at least 4 GiB), and give entries back past either; the encoded-chunk cache takes at most half, and the idle writer loads only when there is room. `auto` is three eighths of the machine's memory (12 GB of 32; the four MicroModels' weights are held in RAM beside their card copies, about 10 GB on their open weights) |
| `limits.memory.content_mb` | `auto`, or a number of MiB (at least `128`, step `128`; presets `1024`, `2048`, `4096`, `8192`) | `auto` | Next call. The memory area for the project's content: the files and pieces read, the drones' intake of them, the relevance keys and the dense lane's vectors. The intake, the relevance-key intake and the dense lane work in bounded steps and stop when the area is full; the pieces not reached are read as their paths only until the area is raised. `auto` is a quarter of the process budget |
| `limits.memory.caches_mb` | `auto`, or a number of MiB (at least `128`, step `128`; presets `512`, `1024`, `2048`, `4096`) | `auto` | When the cache is next made. The memory area for the kept encoded-chunk cache; lower keeps fewer encodings, so repeated questions encode again. `auto` is 15 percent of the process budget |
| `limits.memory.weights_mb` | `auto`, or a number of MiB (at least `128`, step `128`; presets `2048`, `4096`, `6144`, `8192`) | `auto` | Next call. The memory area for the host copies of the MicroModels' weights, which weight placement reads (`nmengn::limits::area_budget("weights")`); lower leaves fewer weights in RAM. `auto` is 45 percent of the process budget |
| `limits.cpu_threads` | `auto`, or a number (presets `1`, `2`, `4`, `6`, `8`, `12`, `16`) | `auto` | Next call. The most processor threads the Nano's work runs on at once; `auto` uses every thread, background work still yields to a question |
| `residency.idle_release` | `off`, `5`, `15`, `30` | `15` | Live. After the server has been idle that many minutes, window copies unused that long and idle scratch buffers are released from the card; the models stay seated. Off: released only when something needs the memory |

The three `limits.memory.*` keys divide the process budget (`limits.ram_mb`, or three eighths of
the machine's memory on `auto`) by memory area. An area on `auto` takes its share of what the
areas set to a size leave, less 15 percent of the budget kept for the engine's own working
memory (the card runtime, staging and working buffers), so the areas together never exceed the budget; a size that would
take the set areas past the budget is refused with the figures, and so is a `limits.ram_mb`
below the sum of the set areas. `nm_status` and `nm_live` (`memory_areas`) give each area's
budget, use, holders and whether a pass stopped at it.

The Settings page also shows this machine's recommended values; `nm_settings` `op: recommend`
reads them and `op: apply_recommended` sets them.

### Agents (MCP)

| Key | Values | Default | Applies |
|---|---|---|---|
| `mode` | `full`, `fast` | `full` | Next call. Default answer mode of `nm_context_pack` on the whole path (`serve --whole`); a call's own `mode` wins |
| `mcp.tools` | `full`, `lean` | `full` | Next call. Full: `tools/list` holds up to `mcp.tool_count` tools. Lean: `tools/list` holds the 5 core coding tools (`nm_context_pack`, `nm_symbol_find`, `nm_view`, `nm_deps`, `nm_feedback`). All 25 stay callable by name |
| `mcp.tool_count` | integer `1` to `25` (presets `1`, `5`, `8`, `12`, `19`, `25`) | `25` | Next call. How many tools `tools/list` holds, in coding priority order; the last 6 are the interface tools. A client such as Claude Code calls only listed tools, so every tool is listed by default; with a lower count the server's instructions name only the listed tools, and dynamic tool activation brings others into the list when a call needs them |
| `mcp.tool_control` | `off`, `on` | `off` | Next call. On: you pick the tools one by one (`mcp.enabled_tools`); with dynamic activation off, the others are hidden and refused |
| `mcp.enabled_tools` | `auto`, or a comma-separated list of tool names | `auto` | Next call. The tools you picked, used when `mcp.tool_control` is on |
| `mcp.dynamic_tools` | `on`, `off` | `on` | Next call. On: further tools are brought into `tools/list` (with `notifications/tools/list_changed`) when a question or a direct call needs them. Off: the list stays fixed |
| `host.claude_lean` | `off`, `on` | `off` | Next `setup --write`. On (or `setup --lean`): setup also writes `"enableArtifact": false` into `.claude/settings.local.json`, leaving out a large built-in Claude Code tool description |
| `priming` | `off`, `quiet`, `on` | `off` | Next hook. Master switch for proactive lifecycle primes (`off` by default; all 6 built-in prime rules also start `off`, opt-in per project). Quiet: strong verified evidence only. On: also weaker evidence, session-start resume, and stop open-loop nudges. Off: none. Every cited line is re-read fresh on disk first and never repeats twice in a session. A value stored under the earlier name `nudges` is still read. See `user-guide.md`, Priming |
| `reminders` | `on`, `off` | `on` | Next hook. On: the reminders you or your coding agent set for a project are delivered through the hooks at their triggers, and the next ready step of the step graph is given at prompts and at stops. Off: none are delivered; nothing set is removed. See `user-guide.md`, Reminders, loops and steps |
| `loops` | `on`, `off` | `on` | Next stop. On: a loop you or your coding agent set continues Claude Code at a stop (`decision: block`) while its condition is unmet, at most its maximum of iterations (3 by default, never more than 20); the one case in which the Nano's hooks block a stop, and only to continue the agent. Off: no loop acts, and every loop stays as it is. A loop does nothing unless one is set and on. See `user-guide.md`, Reminders, loops and steps |

### Routing

| Key | Values | Default | Applies |
|---|---|---|---|
| `routing.code` | `measured`, `embedder`, `judge` | `measured` | Next call. Who decides close calls on code: the MicroModel the measured routing record names strongest (the published strengths table without one), or always the Embedder, or always the judge |
| `routing.prose` | same | `measured` | Next call. The same for prose and documents |
| `routing.record` | `on`, `off` | `on` | Next start. On: the routing record measured on the serving base cartridges replaces the published strengths where it measured enough duels; it is set aside by itself when the base cartridges or the lens change |
| `routing.ask` | `judge`, `always`, `off` | `judge` | Next start. When the Core Brain asks for more context: when the judge finds the pack below its bar, on every question, or never. `--ask` wins |
| `routing.tentacle_judge` | `default`, `heads`, `reranker`, `rrf`, `sum`, `place` | `default` | Next start. How the tentacles judge what they read. `--tentacle-judge` wins |

### Network, storage and cartridges

| Key | Values | Default | Applies |
|---|---|---|---|
| `air_gap` | `off`, `on` | `off` | At once (also locked on by the environment variable `NANO_AIR_GAP=on`). The interface stays on `127.0.0.1`, local network access and tablet pairing are off, and every remote URL or download is refused |
| `network` | `off`, `view`, `control` | `off` | Next start of the interface server; switching off applies at once. Off: this computer only. View: paired tablets on the local network read everything and change nothing. Control: they can also change settings. No encryption on the wire (see `privacy.md`) |
| `storage.disk_floor_gb` | integer `1` to `1000` (presets `2`, `4`, `8`, `16`, `32`, `64`) | `4` | Next build. Free space the Nano keeps on its drive: below it a job-filter build refuses to start and the Storage page warns |
| `library.upgrade` | `keep`, `replace` | `keep` | Next upgrade. Keep: a new base cartridge version is added beside the older ones and serves by default; you can choose or remove any version on the Cartridges page. Replace: once the new version is in the cartridge library and usable, older versions are removed (a version you chose for a MicroModel is never removed) |

### Other choices kept in `settings.json`

- **Cartridges plugged in or out.** A cartridge switched from the Cartridges page (`POST
  /api/cartridges`, MCP `nm_cartridges` `op: on|off`) is logged as a setting with the key
  `cartridge:<id>` and undone like any other.
- **The base cartridge each MicroModel serves**, under `base_cartridges`: a version in the cartridge library,
  the newest usable (`auto`, also what serves when nothing is chosen) or `open` for the upstream
  open weights. Set it on the
  Cartridges page, with `library base <model> <key|auto|open>`, or MCP `nm_cartridges`
  `op: base_serve`. From the next start.
- **The model each role runs on**, under `role_models` (Models page, `library role <role> <model
  id|default>`, MCP `nm_settings` `op: role_set`). From the next start.
- **Base cartridge compensation**, under `base_compensation`: per MicroModel, `on` or `off` (the
  default). On: a base cartridge serves with the compensation cartridge its label names (a bias per
  channel that restores the mean contribution of what the base cartridge left out); a base
  cartridge that names none is not changed. Set it on the Settings page (Controls, Settings file),
  with `POST /api/controls` and the key `base_compensation:<model id>`, or MCP `nm_settings`
  `op: set` with the same key. From the next start; logged as a setting and undone like any other.
  The environment variable `NANO_BASE_COMP` (`on` or `off`), when set, decides it for every
  MicroModel instead.
- **`note`**: one line the program writes with every save, saying what the file is. It holds no
  setting and is shown read only.

Every key of these three maps is also set one entry at a time through the controls route, so one
call sets any key of the file: `POST /api/controls` (or MCP `nm_settings` `op: set`) with the key
`base_cartridges:<model id>` (a base cartridge key, `auto` or `open`), `role_models:<role>` (a
model id or `default`) or `base_compensation:<model id>` (`on` or `off`).

The Settings page shows the whole file: the controls by group, then under **Settings file** every
other key with its entries, each set there. The same view is `GET /api/controls/saved` and MCP
`nm_settings` `op: saved`; a key in the file that no part of the build reads is listed under
`unknown` and kept as it is. `neuromorph-nano check nmcore.parity` fails when a control or a key
of the file has no row in these views, or when a route and an MCP op fall out of step.

## Models folder

The engine reads the upstream open weights from the first of:

1. the `NANO_MODELS` environment variable;
2. `dir` in `designs/models.json` (a relative path is taken from the Nano folder; today
   `"weights"`);
3. `weights/` in the Nano folder.

Each model sits in `<models folder>/<id>/` with its card `part.json` and its files in `files/`.
The ids are `qwen3-0.6b` (the Reader), `qwen3-1.7b` (the Writer), `qwen3-reranker-0.6b` (the
Reranker) and `qwen3-embedding-0.6b` (the Embedder). The engine only reads this folder. The
MicroModels normally serve their Neuromorph base cartridges (`docs/cartridges-and-parameters.md`),
but the open weights must still be installed: each MicroModel reads its tokenizer and configuration
from them, and a base cartridge is seated only after a check run against them.

## Cartridge libraries

The list of cartridge sources is `cartridge-sources.json` in the Nano folder: the official library
on Hugging Face, `gavriel-tech/neuromorph-nano-cartridges` (built in), and any third-party Hugging
Face library you add (Models page, side panel, Libraries; MCP `nm_settings` `op: source_add`). Every change is
logged in the audit. See `docs/cartridges-and-parameters.md`, Cartridge libraries.

## Environment variables

| Variable | Effect |
|---|---|
| `NANO_HOME` | The Nano folder, when the program is not run from inside it |
| `NANO_MODELS` | The models folder (above) |
| `NANO_AIR_GAP=on` | Locks air-gap mode on |
| `NANO_GPU=cpu` | Runs the MicroModels on the processor (also `off`, `none`, `processor`); `NANO_GPU=<n>` names a graphics card by its number |
| `HF_TOKEN` | Your Hugging Face token, for a download that needs one (also read from `HUGGING_FACE_HUB_TOKEN`, or from the token file the `hf` command line's login leaves in your user folder); sent only to huggingface.co |

## Memory

- Each repository served keeps its memory in `memory/projects/<name>-<8 hex>/`; the Nano's own
  folder uses `memory/`. Layout: `docs/architecture.md`, Memory layout.
- Memory policy (tier limits, temporary note rules, the write gate) is in `designs/memory.json`.
  Text that looks like a secret is never saved; the gate's refusal log keeps only the time, the
  reason and a hash of the refused item's key.
- Muscle memory settings are in `designs/muscle.json`; packs are stored in `muscle/`.
- Switch memory off with the `memory` control or `serve --memory off`.
- The meaning index (`--neural`, `--dense`) is kept in `dense/vectors.bin` in the project's memory
  folder while memory and the `dense_cache` control are on. A file written for another embedding
  model, precision or arithmetic setting is not used and is replaced as the index is built again.
  When the Embedder's served values change while `serve` runs (a parameter overlay or plug change
  that binds it again), the index is marked stale, the meaning ranking is withheld, and a new index
  is built at idle with the new Embedder and switched in when complete; no restart is needed.
  Clear it with `neuromorph-nano dense clear --project <repo>` (or a POST to `/api/dense/clear`
  with `{"project": "<key>"}`, the key as `/api/projects` lists it, `nano` for the Nano's own
  folder); the removal is logged in `memory/audit.jsonl` with the rows removed.

## Command line

### `serve`

`neuromorph-nano serve --root <repo> [flags]` serves one repository over MCP on standard input and
output. `setup --write` writes an entry that runs it as `launch serve --root <repo> --whole
--neural` (the whole system; `setup --plain` writes `launch serve --root <repo>`, the plain path);
add or remove flags in the `args` list in `.mcp.json` to change how it runs.

| Flag | Effect |
|---|---|
| `--root <repo>` | The repository to study (required) |
| `--memory off` | No recall and nothing written to memory |
| `--memory-dir <category>/<name>` | Keep a test session's memory apart, under `tests/<category>/<name>` |
| `--skip <folder>` | Leave folders of this name out of what is read (repeatable) |
| `--poll <seconds>` | Keep the read of the repository in step with changes |
| `--neural` | Bind the Embedder and the Reranker; starts the dense lane |
| `--dense [on\|off]` | Bind the Embedder alone and start the dense lane; `off` with `--neural` stops it |
| `--whole` | Answer through the whole path: drones, tentacles, the GCX, the Core Brain's Reranker order and second look |
| `--precision <w>` | Every model at 16, 8 or 4 bits, or per model: `base=8,writer=16,reranker=16,embedder=8` |
| `--precision-map <file>` | Per-piece widths per model from a file |
| `--refine off\|uniform\|nbv` | The second coordination round (default `nbv`) |
| `--record` | Record parameter map activity for this session |
| `--unit <id>` | Load a registered learning-weight unit |
| `--learn on\|off`, `--muscle on\|off`, `--muscle-store <dir>` | Continuous learning and muscle memory for this run, over the saved controls |
| `--rack off` | Leave the cartridge library's stacks out for this run: no stack's filters applied |
| `--base-cartridges off` | Every MicroModel serves its upstream open weights for this run (a comparison) |
| `--try-cartridge <id>[,<id>]` | Plug the named cartridges into their MicroModels' sockets for this run, even on trial (a measurement; the start records it) |

Advanced options, for measurement and research of single parts:
`--tentacles`, `--tentacle-judge heads|reranker|rrf|sum|place`, `--tentacle-pose plain|focus`,
`--tentacle-score mass|density|evidence|peak`, `--tentacle-shortlist <n>`, `--tentacle-core`,
`--brain-chains`, `--rerank-neural`, `--rerank-withhold <p>`, `--recall-meaning off`,
`--recall-rank r2`, `--embedder-q8`, `--reranker-q8`, `--dp4a`. Speed settings, on by default for
the served models: `--half-math on|off` (half2 products in the 16-bit linear layers) and
`--segment-attention on|off` (the attention kernel for packed sequences); both off give the
reference arithmetic. With `--whole`:
`--ask judge|always|off`, `--reflexes <list>`, `--withhold off|<p>`, `--up-next <k>`,
`--reach on|off`, `--reach-k <n>`, `--uses on|off`, `--look off`, `--anatomy`, `--packs off`,
`--rerank-cap <n>`, `--look-gate <log-odds>|off` (second look only when the Reranker's top two are
closer), `--rerank-drop <layer>:<margin>:<keep>|off` (early drop in the batched Reranker; without
it the `rerank.early_drop` control applies), `--rerank-single` (one document at a time, for
comparison), `--rerank-compare` (score every set both ways and record the difference), `--fast-k
<n>` (pieces fast mode reads, default 4), `--judge-window <tokens>|off` (the Reranker reads the
path line and at most this many tokens of each piece, whole lines around the question's terms;
default 384; the pack still carries whole pieces), `--fast-judge-window <tokens>|off` (the same in
fast mode; default off, whole pieces), `--look-window on|off` (the second look reads the same
windows; default on), `--full-head` (the second look reads the whole output head; comparison only),
and `--inject-fault <part>` (reliability tests only).

### `ui`

`neuromorph-nano launch ui [--port <n>] [--dev]` serves the interface and its API on 127.0.0.1,
default port 4870. Start it through `launch` so the server runs from a copy of the newest build and
never blocks a rebuild. Settings changes need the server's local token, which the server places in
the pages it serves. `--dev` also accepts the `npm run dev` server on port 3000 and lets it fetch
the token from `/api/token`; use it only while developing the interface.

With the control `network` at `view` or `control` when the server starts, it also listens on this
machine's private addresses (10/8, 172.16/12, 192.168/16, 169.254/16), on the same port, and
prints them. A tablet opens `http://<address>:<port>/`, gets the pairing page, and enters the
six-digit code from `neuromorph-nano pair` (or from the interface). A code lasts five minutes and
works once; five wrong codes lock pairing for ten minutes. A paired tablet gets its own token
(kept only as a SHA-256 hash in `memory/network/devices.json`, with its name and when it was last
seen) and can be revoked from the Settings page. Requests are served only from private or
link-local addresses and only when the Host header names the address and port they reached.
There is no encryption on the wire: anyone who can listen on the local network can read the
traffic, including a tablet's token. The operating system may ask once whether to let the program
accept connections; that question is yours to answer, and the Nano changes no firewall rule. The
air gap turns all of this off. The API is described in `interface-api.md`.

### Other commands

| Command | Effect |
|---|---|
| `setup --project <repo> [--plain] [--lean] [--write]` | Print, or write, the repository's `.mcp.json` entry for Claude Code (the whole path; `--plain` for the plain path; `--lean` also sets `"enableArtifact": false` in `.claude/settings.local.json`) |
| `setup --project <repo> --hooks [--write]` | Print, or write, the Claude Code hook entries in the repository's personal `.claude/settings.local.json` (other entries kept byte for byte; a backup of the file as it was in the project's memory folder, `setup/backups/`) |
| `setup --project <repo> --antigravity\|--agy [--plain] [--hooks] [--write]` | Print, or write, the Antigravity project plugin in `.agents/plugins/neuromorph-nano/` |
| `setup --project <repo> [--antigravity] --remove` | Take out the Nano's MCP entry and hook entries, and nothing else; say what was removed |
| `setup --project <repo> --status` | Show what is installed: the MCP entry, the hook entries and the program they run, the prompt setting, the situation record |
| `connect <claude\|antigravity\|cursor\|vscode\|windsurf\|generic> [--project <repo>] [--plain] [--hooks] [--dry-run] [--remove]` | Connect a repository to one MCP client: Claude Code (`.mcp.json` and its approval), Antigravity (the project plugin), Cursor (`.cursor/mcp.json`), VS Code (`.vscode/mcp.json`), Windsurf (its user file `~/.codeium/windsurf/mcp_config.json`) or any other client over standard input and output (the command line and JSON, printed). Other servers' entries keep their bytes; a backup of the file as it was goes to the project's memory folder; the write is audited. `--dry-run` prints the line difference and writes nothing; `--remove` takes out exactly the entry written |
| `pair` | Make a six-digit code to pair a tablet (five minutes, once; see `ui`) |
| `guidance --project <repo or key> [view \| add \| edit \| enable \| disable \| done \| reset \| remove] ...`, `guidance undo <audit id>` | The reminders, loops and step graph for the coding agent, through the interface's own route: add, change, switch, end, mark done, reset, remove (a preview; with `--confirm <id>` the id typed again) and undo. See `user-guide.md`, Reminders, loops and steps |
| `situation --project <repo> [--session <id>] [--last <n>] [--json] [--clear-prompts]` | Print the hook records of a repository (newest 50 by default), with kept prompt text; `--clear-prompts` removes the kept prompt text |
| `session --project <repo> [--session <id>] [--list] [--json]` | The state of one Claude Code session (the newest by default): the task, files read and edited with their lines, commands and checks, what is open, what the Nano served and what followed each pack; `--list` lists the sessions. Read only |
| `ledger --project <repo> [--pack <id>] [--file <path>] [--json] [--as-of <ms>] [--clear]` | Bring the evidence ledger up to date and print it: counts by rule and strength, the rows (all, one pack's, or one file's), and packs whose window is still open. `--as-of` judges windows as of that time (to replay records); `--clear` removes the ledger. Nothing is written with the `memory` control off |
| `outcomes --project <repo> [--recall "<question>"] [--import] [--json]` | What memory knows of verified and refuted outcomes (places that answered a question, changes that fixed a failure), each place read fresh; `--recall` lists what a question recalls; `--import` takes the ledger's outcomes into the memory journal now (refused while a server holds it) |
| `examples --project <repo> [--show <id>] [--json] [--export <file>] [--import <file>] [--clear] [--as-of <ms>]` | The training store: brings the ledger and the store up to date, then counts by family, verdict, strength, split, domain, candidate label and engine, and shows the newest example; `--show` one example (by id or pack); `--export` the whole store to one file for moving it to another of your machines, `--import` such a file (its hash checked first); `--clear` removes the store (audited). `--capture <file>` records one judged set given in a file, in the shape the serving path hands over (tests and replays) |
| `examples --references`, `--reference-add <path> --licence <id> [--source <name>] [--domain <tag>]`, `--reference-fitted <id> --cartridge <id>` | The registry of reference material for fundamentals cartridges (`memory/training/references.json`): source, where it is, licence, SHA-256, what was fitted from it. The material is never copied. Audited |
| `keep [--apply] [--json]` | Retention of the audit log: tool rows older than `audit.keep_days` days of use go, every other row stays. Without `--apply` it only shows what would go |
| `backup [list\|add\|remove\|now\|snapshots\|preview\|restore] ...` | Your own repositories (any git remote, a Hugging Face repository or a folder) for backing up and rolling back memory, parameter maps, cartridges, learned units and settings: `add <url> [--parts ...] [--token]`, `now <id>`, `snapshots <id>`, `preview <id> <commit>`, `restore <id> <commit> --confirm`. Nothing is sent until you back up; a restore copies each changed file aside first and Audit undoes it |
| `library <list\|show\|stack\|switch\|export\|import\|rollback\|bases\|base\|base-remove\|compat\|roles\|role\|merge> ...` | The cartridge library on this machine (`library/` in the Nano folder). `list` shows cartridges (domain, MicroModel, layers) and stacks; `show <id>` one stack or cartridge; `stack [<use>] [--machine <file>]` the stack in use for each use on this machine and why; `switch <use> <stack>\|off` picks the stack for a use (a running server plugs its socket cartridges at its next call, its settings at the next start), `switch <stack> trial\|serve` puts a stack on trial or serves it (serving needs evidence measured on its own hardware piece); `export <stack> --out <file>` and `import <file>` carry a stack as one file (an import is hash-checked and arrives on trial); `bases` lists every base cartridge version per MicroModel and which serves; `base <model> <key\|auto\|open>` chooses what serves from the next start; `base-remove <key> --confirm` deletes one version (not the serving or chosen one); `rollback <model>` goes back to the version the serving one was made over; `compat` shows which MicroModels each cartridge and job filter fits; `roles` and `role <role> <model id\|default>` show and set the model each role runs on; `merge <release-catalogue.json>` adds a release's new cartridges and stacks (the installer runs it). Every change is audited and undone with `/api/undo`. Nothing is sent anywhere |
| `dense clear --project <repo>` | Remove the meaning index kept on disk for a repository; the next start embeds it again |
| `daemon --root <repo> [--port <n>] [--port-file <file>] [serve options]` | Serve warm on 127.0.0.1 (default port 7421; the port goes to `scratch/daemon.port` or `--port-file`) for `attach` and `ask` |
| `attach [--port <n>] <tool> [json]`, `attach [--port <n>] --stop` | Run one tool on the warm daemon, or stop it |
| `ask [--port <n>] <question>...` | `nm_context_pack` on the warm daemon, printed |
| `launch <command> [args]` | Run any command from a fresh copy of the newest build in `scratch/live/`, so a running server never locks `target/release`; `setup` writes `.mcp.json` entries that use it |
| `call <tool> [json] [--root <repo>]` | One tool call, memory off |
| `trace --root <repo> [--memory-dir <c>/<n>] [--neural] [--tentacles] <question>...` | Print what each tentacle and drone did and how the context was chosen |
| `bench [--mode plain\|fast\|full] [--n <count>] [--root <repo>] [--repeat] [--wait-dense] [--profile] [--memory-map] [-- <serve options>]` | Time the context path per stage over `tests/dev/questions.json`, with card memory and whether the right place is first, in the top three and in the pack; written to `runs/bench-<mode>-<ms>.txt` |
| `soak ...`, `coldstart --repo <path> ...` | Long-running and first-run reliability harnesses for developers (`soak --help`); reports go to `runs/` |
| `check [all \| <crate\|crate.part\|cards\|designs\|anatomy\|integrity\|lock\|documents\|languages>...]` | Section checks; with no name (or `all`) every section and every crate runs, each even after a failure, and the pass count is printed |
| `lock [--root <dir>]...`, `seal <part.json>...`, `selftest engine [cpu\|gpu\|both] [--record]` | Developer checks: the code lock of a tree, write a card's hash, the engine against its recorded reference |
| `hook <Event>` | Claude Code hook intake (written by `setup --hooks`): reads the hook's JSON on standard input, appends one record to `situation.jsonl` in the project's memory folder, always exits 0; prints nothing but a prime (when the `priming` setting allows one) and the reminders and steps set for the project (setting `reminders`), as one JSON object with `additionalContext`, except at a stop while a loop is unmet and under its cap (setting `loops`), when it prints `decision: block` with the loop's prompt. The project is `CLAUDE_PROJECT_DIR`, else the hook's `cwd`. With the `memory` control off nothing is recorded |
| `hook stop [--root <repo>]` | Claude Code Stop hook: blocks once when Rust files changed and `cargo check` fails |

`--precision`, `--precision-map`, `--half-math` and `--dp4a` are read by every command.
