# Interface API

The interface server (`neuromorph-nano launch ui`, port 4870 by default) serves the interface's
pages and a JSON API. Every page of the interface reads and writes through this API, and the MCP
tools `nm_storage`, `nm_cartridges`, `nm_learning`, `nm_comprehension`, `nm_settings` and
`nm_live` run the same routes inside the serving process, so a page and a tool always run the same
function (see `mcp-tools.md`).

This page describes the API as of API version 7 (`/api/status` field `api`). The routes marked
v0.1.0 below were added for the first public release without raising `api`; a client that needs
one should treat a 404 as "not served by this engine build".

Version history (each version only adds; every earlier field is still there):

- Version 7 adds `/api/examples`, `POST /api/examples/clear` and the setting `examples`, and
  renames the nudges of version 6 to priming: one note is a prime, the setting is `priming`, the
  ledger row kind is `primed`, the lists and counts are `primes` (the earlier names `nudges` are
  still given beside them, and a stored `nudges` setting and earlier `nudged` rows are still read).
- Version 6 adds `/api/outcomes`, `nudges` in `/api/session`, `nudges` and `outcomes` counts and
  the row kinds `nudged` and `outcome` in `/api/ledger`, and the setting `nudges` in
  `/api/controls`.
- Version 5 adds the cartridge library: its cartridges as further rows of `/api/cartridges`,
  `/api/cartridges/library`, and switching the stack for a use on `POST /api/cartridges`.
- Version 4 adds `/api/session`, `/api/ledger` and `POST /api/ledger/clear`.
- Version 3 adds a clock on every step of a tool row (`timed`, `us`, `inside`, `cells`, `counts`),
  `/api/activity`, `/api/memory/tiers` and each project's `tiers`.

Reads are `GET` (or `HEAD`). Writes are `POST` with a JSON body (`Content-Type: application/json`),
the header `X-Nano-Token` carrying the local token (placed by the server in every page it serves,
`<meta name="nano-token">`), and no browser origin other than the interface's own. The server
answers only requests addressed to `127.0.0.1` or `localhost` on its port. A body is limited to
64 KiB (8 MiB for `/api/parameter-maps/save`). A paired tablet on the local network uses its own
token instead (see `configuration.md`, `ui`). Errors come back as `{"error": "..."}`.

A project is named by its key as `/api/projects` lists it (`nano` for the Nano's own folder).
The MicroModels are named by role: `reader`, `writer`, `embedder`, `reranker`. Fields called
`model` (and the `id` of a models folder entry) hold the id of the upstream open weights a
MicroModel is built from, for example `qwen3-0.6b`; the role each one serves is in
`/api/models/roles`.

## Reads

| Path | What |
|---|---|
| `/api/status` | Name, version, `api`, folders, models on disk, models held now, precision |
| `/api/controls` | Every control with its `key`, `group`, `kind` (`"enum"` or `"integer"`), `value`, `values` (allowed choices or presets), `default`, `help` (`desc`), `wired`, `applies`, and for integer allowances `min`, `max`, `step` and `unit` |
| `/api/controls/saved` | `settings.json` key by key (v0.1.0): `file`, `present`, `controls` (how many controls, listed by `/api/controls`), `controls_saved` (how many differ from their default), `keys` and `unknown`. `keys` holds every key of the file beside the controls (`base_cartridges`, `base_compensation`, `role_models`, `note`), each with `label`, `help`, `applies`, `undo`, `in_file`, `read_only` (null, or the reason nothing sets it), `set` (`{route, key, mcp}`: the controls route and the key form `<key>:<entry>`), `also` (`{route, mcp, interface}`: the other route, MCP op and interface page that set it) and `entries`, one per MicroModel or role: `name`, `role`, `value`, `default`, `saved`, `in_effect`, `choices` (`[{value, ok, why}]`), `locked` (null or the reason) and `set_key` (the key `POST /api/controls` takes for it). `unknown` names keys in the file that no part of this build reads; they are kept |
| `/api/parts` | Every crate's card (`crates/*/part.json`) by crate name |
| `/api/settings` | Every design file (`designs/*.json`) by name, the anatomy among them |
| `/api/projects` | Project memory folders and the projects served now |
| `/api/memory` | `memory/index.json` |
| `/api/memory/tiers` | Records per memory tier, for the Nano's own memory and every project (version 3) |
| `/api/memory/items?project=<key>[&limit=<n>][&q=<text>]` | A project's memories as the tiers hold them, newest first (`limit` 200 by default, at most 2,000; `q` filters by text) (v0.1.0) |
| `/api/goals?project=<key>` | The session objectives (`nm_goal`) kept for a project (v0.1.0) |
| `/api/guidance?project=<key>` | The agent's guidance: `reminders` (each with `delivered`), `loops` (`state`, `iterations`, `max`, `blocks`, `not_acted_antigravity`, and for an active loop `now`: whether its condition holds in the newest session and why), `steps` (each with `status`: `done`, `ready` or `blocked`, and `depends_on`, `depends_any`, `probe`), `counts`, `next_step`, the `settings` in force (`reminders`, `loops`, `memory`), `limits`, `hosts` (what Claude Code and Antigravity can take), the newest 40 `deliveries` (time, session, event, kind, id, channel, host, text, and `verdict`: the evidence ledger's judgment of that delivery, `{delivery, kind, item, event, session, at, window, closed, verdict: {rule, state, strength, event, steps_after, why, detail}}`, or null for a delivery no rule judges: a loop's final note, a step the Nano marked done, a stop not acted on), `delivered_total`, and `verdicts` (`rules`, `judged`, `by_kind`: counts by delivery kind and state); a loop also carries `parts`, `stall`, `budget_ms`, `budget_tools`, `repeats`, `unmet_seq`, and its `state` may be `stalled` or `budget-spent`; `outcomes`: the newest ended loops, newest first (20, or `&outcomes=<1..200>`), each `{id, by, prompt, condition: {until, parts, goal}, iterations, max, stall, budget: {ms, tools, used_ms, used_tools}, end: met \| cap \| stalled \| budget-spent \| ended \| removed, end_note, unmet, verdicts: {rule: "loop-met", state, continuations}, started_ms, ended_ms}`, from `guidance-outcomes.jsonl`; `outcomes_total`, `outcomes_file` (v0.1.0) |
| `/api/activity[?window=][&project=]` | Per-component activity over a time window, from the audit rows (version 3) |
| `/api/audit[?kind=][&project=][&limit=]` | Audit rows and episodes, newest `limit` (200 by default, at most 10,000) of each |
| `/api/session?project=<key>[&session=<id>]` | A Claude Code session's state: task, files, commands, what is open, what the Nano served and what followed (version 4) |
| `/api/ledger?project=<key>[&pack=<id>][&file=<path>][&limit=<n>]` | The evidence ledger: chain state, counts by rule and strength, rows (all, one pack's, or one file's), packs whose window is still open (version 4) |
| `/api/outcomes?project=<key>` | What memory knows of verified and refuted outcomes in a project, each place read fresh (version 6) |
| `/api/examples?project=<key>[&id=<example id>][&limit=<n>]` | The training store: counts by family, verdict, strength, split, domain, candidate label and engine, and the newest examples (50) or one (version 7) |
| `/api/training/references` | Reference material listed on this machine for fundamentals cartridges, each with whether its path is still there (nothing is copied) (v0.1.0) |
| `/api/training/distil[?set=<name>][&limit=<n>]` | Distillation's teacher data (`memory/training/distil.jsonl`): each set with its rows, healthy rows, splits, outcomes, teachers, licences and pairs with teacher log-probabilities; a page of rows with short previews; the kinds (`targets`), techniques, which technique fills which kind (`fills`), what is available (`built`, by role, kind and technique), what is not and why (`unavailable`: role, target, technique, why), the gate in words (`gate`; the Reranker's, from teacher scores, `gate_scores`), each set's scored pairs (`with_score`), and the distilled cartridges with their kind (`lowrank`, or `readout` for a judge read-out), gate figures and plug state |
| `/api/learning?project=<key>` | The project's continuous learning state and the learning cycle's standing (verified examples since the last cycle, the threshold, the cycles recorded); read only (v0.1.0) |
| `/api/comprehension[?project=<key>]` | The project's world model: notes per level (L1 piece, L2 file, L3 module, L4 system), the outline, the measured routing record, the drone head sets per lens and the number of serving base cartridges; the Nano's own memory without `project` (v0.1.0) |
| `/api/cartridges` | Learned cartridges, then the cartridges in the cartridge library (version 5) |
| `/api/cartridges/library` | The cartridge library: cartridges by domain, stacks for each use, the stacks in use on this machine and why (version 5) |
| `/api/cartridges/bases` | Every base cartridge version in the cartridge library, by MicroModel, and what serves (v0.1.0; below) |
| `/api/cartridges/compat` | Which MicroModels each cartridge and job filter fits, and why not the others (v0.1.0; below) |
| `/api/preferences` | The stated preference cartridges (rules with what each does and how it is held, scope, plug state with its use and stack, versions, whether it also holds learned values), the rules in force, how each kind of rule is held and the forms the parser reads (v0.1.0) |
| `/api/cartridges/inspect?key=<id@version or id>` | One cartridge's label and record, the stacks holding it and the uses they run, its values as far as they are settings, and its `parameters` section (below); a job filter (`filter:<role>/<job>`) and an adapter library version (`<id>@<n>`) are read too (v0.1.0) |
| `/api/cartridges/parameters[?key=<key>[&tensor=<name>&offset=<n>&limit=<n>]]` | Cartridge parameters (below): every object that holds learned parameters; one object's parameters section; one page of a tensor's values (v0.1.0) |
| `/api/parameters/overview?key=<key>[&tensor=<name>&w=<cols>&h=<rows>&r0=&r1=&c0=&c1=]` | The whole-set overview of an opened object (`nm-parameter-overview/1`, MCP `nm_cartridges` `parameters_overview`). Without `tensor`: `{key, model, kind, name, count, weights, tensors}`, every tensor in layout order (outside layers first, such as the token table, then each layer, then the rest) with `name`, `layer`, `part`, `shape`, `dtype`, its two-dimensional view `rows` and `cols` (a vector is one row; three or more dimensions are the first by the product of the rest), `size`, `readable` and `unread`; no values are read (the open weights: the safetensors headers only). With `tensor`: rows `r0..r1` and columns `c0..c1` (default the whole tensor) as a `w x h` grid (default 128 a side, at most 1024 a side and 65,536 cells, at most one cell per weight); cell `j` covers columns `c0 + floor(j * W / w)` to `c0 + floor((j + 1) * W / w)`, rows likewise. Answers `{tensor, layer, part, shape, dtype, rows, cols, region, path, w, h, cell: {rows, cols, weights}, exact, summary: {min, max, mean, rms, count, nonfinite}, read: {bytes, chunks, largest_chunk, chunk_cap, ms}}` with `min`, `max`, `mean` and `rms` per cell in row order, or, when every cell is one weight (`exact`), `values` (the exact stored values). `path` is the region in the editor's path syntax. The region is streamed from the file in chunks of at most 4 MiB and folded into the cells as it is read; a tensor is never loaded whole (v0.1.0). The command line `neuromorph-nano library parameters <key> --overview [--tensor <name>] [--w N --h N] [--region r0:r1,c0:c1] [--hist]` prints the same JSON |
| `/api/backups` | Your own repositories for backup and rollback: each with its URL, the parts it holds, whether your token is sent and its last snapshot; the parts there are, whether git runs here, and whether the air gap is on (v0.1.0) |
| `/api/cartridge-sources` | The cartridge sources: the official library and the third-party Hugging Face libraries added (v0.1.0; below) |
| `/api/downloads` | The open weights and base cartridges on disk, and the download in progress (v0.1.0; below) |
| `/api/models/roles` | Each role, the model it runs on from the next start, and which installed models may run as it (v0.1.0; below) |
| `/api/models/weights?model=<id>[&limit=<n>]` | What `nm_weights` reads for an installed model: its weight card, tensor and cluster counts, element types, the first `limit` (40) tensors and clusters, and what is resident when this process runs it (v0.1.0) |
| `/api/parameter-maps` | Every parameter map of every MicroModel, with its lineage and state (v0.1.0; below) |
| `/api/parameter-maps/read?role=<r>&file=<f>[&section=<a/b>][&layer=<n>][&offset=<n>][&limit=<n>]` | A map's contents, whole when small, else by keys or by a page of elements (v0.1.0) |
| `/api/surgery` | The job filters and surgery records of each MicroModel, and the job filter build in progress (v0.1.0; below) |
| `/api/learned-units` | Every learned unit by compartment, each plugged in or out (v0.1.0; below) |
| `/api/units` | The registered learning units (`units/registry.json`): target, rank, slot, status, whether loaded, the model it was made over and whether its files match their hashes (v0.1.0) |
| `/api/lenses[?project=<key>]` | The project lenses (the Embedder's per-project adapter): compartment, plug state, whether it acts and why not, versions and files (v0.1.0) |
| `/api/storage` | Disk use by category with exact paths, what a prune would free, and the free space against the disk floor (v0.1.0) |
| `/api/storage/settings` | The storage paths and limits (v0.1.0) |
| `/api/limits` | The hardware limits: saved (with the three `limits.memory.*` areas), in effect in the serving process, each memory area's budget and use (`memory_areas`), recommended for this machine, and its profile (v0.1.0) |
| `/api/airgap` | The air gap and what leaves the machine: `air_gap`, `source`, `locked_by_env`, `network`, `loopback_only`, `lan_listeners` (v0.1.0) |
| `/api/stage-times[?file=STAGE-BENCH-<stamp>.txt]` | Per-stage median and p95 times of a warm question from the newest stage-bench report in `runs/`, and recent live calls when a serving process publishes them (v0.1.0) |
| `/api/live` | What the serving processes do now: state, active stages, MicroModels executing, card memory, each memory area's use against its budget (`memory_areas`), the idle writer and the recent timeline (v0.1.0; below) |
| `/api/live/stream` | The same as Server-Sent Events (v0.1.0; below) |
| `/api/connect?project=<folder or key>` | For each MCP client `connect` writes for, whether the Nano's entry is in that client's settings for the project, and the command lines to connect or preview (read only) (v0.1.0) |
| `/api/devices` | The `network` setting, the addresses listened on, paired devices, pairing state |
| `/api/token` | The local token, only to the development server under `--dev`; otherwise 404 |

### `/api/status`

Added in version 2. Also given: `root`, `models_dir`, `memory_dir`, `parts` (the crate names),
`precision` (the saved widths per part: `base`, `writer`, `reranker`, `embedder`), `time_ms`, and on
each `models[]` row `precision_source` (`serving process`, or `saved setting (applies at the next
start)`).

Taken from a run on 2026-11-15 (one `serve --whole` process over `crates/nmjson`; lists shortened):

```json
{
  "api": 2,
  "models": [
    {"id": "qwen3-0.6b", "bytes": 1519210332, "card": true, "main_file": "files/model.safetensors",
     "precision": "default", "loaded": true, "layers": 28, "heads": 16, "head_groups": 8,
     "kv_heads": 8, "hidden": 1024}
  ],
  "loaded": [
    {"id": "qwen3-reranker-0.6b", "card": "qwen3-reranker-0.6b", "roles": ["reranker"],
     "ram_bytes": 1191553024, "gpu_bytes": 880939008, "bytes": 2072492032, "device": "gpu",
     "precision": "default", "effective_bits": 16, "mixed_precision": false,
     "layers": 28, "heads": 16, "head_groups": 8, "kv_heads": 8,
     "layers_on_card": [0, 1, 2, 27], "on_card": [0, 1, 2, 27],
     "pid": 49092, "project": "nmjson-c46abbda"}
  ],
  "resident_bytes": 2383652864,
  "gpu_bytes": 1321476096,
  "serving": 1,
  "substrate": {"chunks": 22, "files": 9}
}
```

- `models[]`: every folder of the models folder (one per set of upstream open weights, named by
  its id), with `layers`, `heads`, `head_groups` (key-value heads) and `hidden` read from its
  `files/config.json` (`kv_heads` is the same as `head_groups`), and `loaded` when a serving
  process holds it now. `precision` is the width a serving process reports for it, else the saved
  setting.
- `loaded[]`: the models the serving processes hold now, one row per process and model, read from
  each server's live file (`memory/live/`, rewritten every 10 s; older than 30 s is not shown).
  `ram_bytes` are the bytes of the model's parameter views in RAM, `gpu_bytes` the bytes of its
  weight copies on the card, `bytes` their sum. `precision` is the width asked (`default` or bits);
  `effective_bits` is the width most of its linear layers run at now, `mixed_precision` true when
  they differ. `layers_on_card` (also as `on_card`) are the layers with weights on the card now. The Writer
  (built from the Qwen3 1.7B open weights) is not reported while the serving path has not bound
  it.
- `resident_bytes` and `gpu_bytes`: the sums of `ram_bytes` and `gpu_bytes` over `loaded`. They
  count model weights only, not the rest of each process's memory.
- `serving`: how many serving processes have a fresh live file.
- `substrate`: chunks and files the most recently active server holds (null when none serves).

### `/api/audit` tool rows

A tool call is written when it starts, with `"state": "running"`, and written again under the same
`id` and `time_ms` when it ends. The API returns one row per `id`, in the place of its first
writing, with the content of its last. Each row also carries `project`, `mode` (the serving path),
`via` (`serve` for an MCP call, `call` for the command line), `session`, and `client` (v0.1.0): the
name the connecting MCP client gave itself in `initialize` (`clientInfo.name`, for example
`claude-code`, `cursor` or `Visual Studio Code`; empty before its `initialize`, absent on older
rows). The Live Activity page lists every coding agent's calls by this name
(`/api/audit?kind=tool`). `state`:

- `running`: the call is in progress. `parts` lists the parts the tool goes through; `steps` is
  empty.
- `done`: the call ended (`ok` says how).
- `stale`: the call never ended (the process crashed or was killed). A server marks its own open
  row stale at its next call and every other server's open rows older than ten minutes when it
  starts; the API also shows any running row older than ten minutes as stale.

`steps` lists every part the call used, in run order:

```json
{"part": "nmbrain.g1#g1", "start_ms": 212, "ms": 37}
```

- `part`: an anatomy id (`nmpod.unit#t3`, `nmbrain.g2#g2`, `nmbrain.slot#reranker`), a
  sub-component id (`nmapi.assemble`) or a crate id.
- `start_ms`: milliseconds from the call's start; `ms`: whole milliseconds spent (rounded down).
  Both come from the part's own clock. A part whose start was not measured has no `start_ms`.
  Nothing is estimated.
- `timed` (version 3): true when the part's clock measured it. A part that was not timed has
  `"ms": null` and `"timed": false`; it is never written as 0 ms. A timed step with `"ms": 0` ran
  for less than a millisecond, and its `us` says how long.
- `us` (version 3): the same length in microseconds, when the part's clock measured it that
  finely (the frame's nodes, every part timed by the tools' clock, the MicroModels' engine time).
  Steps timed from a stage clock in seconds (the stack's drones, tentacles, G1, G2, the second
  look, the assembly) have `ms` only.
- `inside` (version 3): the step ran inside the step named here and its time is part of that
  step's (the memory recall, the Core Brain's emit and the memory write inside
  `nmapi.whole#assembly`; a tool's parts inside `nmapi.tools`). Leave such steps out when summing.
- `cells` (version 3): the cells of the part the step touched, by name. Memory steps
  (`nmmem.gate#m0`): `read:episodes` / `write:episodes` (the episodic store), `read:core`,
  `read:active`, `read:archive` and the `write:` forms (the tier holding each item read or
  written), `read:journal` / `write:journal` (memory revision 2), `cocitations`, `write:none`
  (nothing stored: withheld, refused). The Core Brain's reduce step (`nmbrain.synthesis`): the
  components it runs through (`nmbrain.rivers#b0`, `nmbrain.arbiter#b0`,
  `nmbrain.contradiction#b0`, `nmbrain.synthesis#b0`). Its emit (`nmbrain.stack#emit`):
  `nmbrain.slot#writer`. Others: `territories`, `plan`, `tuner`, `card gate`, `open`, `running row`.
- `counts` (version 3): what the step reported, by name. Drones on the whole path: `chunks` read
  and `flagged`; tentacles and their `g0_tx`: `window` (chunks in the tentacle's window this call)
  and `reported` (its gate depth); G1: `handover`, `queue`, `rounds`; G2: `up`, `instructions`;
  memory: `episodes`, `reads`, `chunks`; the reduce step: `packets_in`, `packets_out`; the plan:
  `jobs`; the Neuropod: `tentacles`.
- `layers`: on a MicroModel step, the first and last model layer it ran (the reranker runs all 28;
  the second look reads up to the drones' last layer).
- `shared`: present when several parts ran as one pass and each carries the pass's time
  (`"drones' intake"`: the eight drones read every chunk in one pass; `"windows and g0_tx"`: the
  tentacles' windows and their `g0_tx` are made in one pass). Count such a group once when
  summing.
- `job`: on a tentacle step (`nmpod.unit#t0` to `nmpod.unit#t7`, and `nmapi.tentacle_core#tN`
  where that path runs), the job the Core Brain's plan gave that tentacle for this call:
  `structural`, `functional`, `dependency`, `execution`, `semantic`, `historical`, `constraint`,
  `predictive`, `verification`, `reconciliation`, `retrieval` or `reasoning`. Present only when
  the path assigns jobs: the plain and neural path (the forage frame) does; the whole path does
  not (its tentacles take windows, not jobs), so there the field is absent. A tentacle that got no
  assignment on a call (no territory) has no `job` either.
- `lens`: on a tentacle step with a `job`, the lens its window was filled with for this call, when
  the tentacle read a window. The territory path gives no lens, so it is absent there.

```json
{"part": "nmpod.unit#t4", "start_ms": 1, "ms": 0, "job": "dependency"}
```

What each path reports (version 3):

- Plain and neural path (the forage frame): `nmapi.mcp` (the start hook writing the running row,
  from the call's start), `nmpod.pod#p0` (the Neuropod cutting the territories),
  `nmbrain.slot#router` (the Core Brain's plan), every node the frame ran (`nmbrain.stack#plan`,
  `nmpod.unit#tN`, `nmpod.unit#dN`, `nmpod.pipeline#tN`, `nmmem.working#tN`, `nmbrain.rivers#b0`
  and so on) with the frame's own start and time in microseconds, `nmapi.tentacle_pass` (when a
  pass is bound), `nmapi.coordinator` (the Map Coordinator), `nmmem.gate#m0` (the recall, memory
  on only), `nmbrain.synthesis` (the reduce step and its withholding check, timed together),
  `nmbrain.stack#emit`, `nmapi.brain_side` (when the Core Brain side is on), `nmapi.anatomy`,
  `nmmem.gate#m0` (the write, memory on only), the MicroModels' engine time per slot
  (`nmbrain.slot#...`, length only, with `layers`; it overlaps the steps that called the engine)
  and `nmapi.assemble`. Every step is timed.
- Whole path (`serve --whole`): `nmapi.mcp`; the work before the stack runs, each timed:
  `nmmem.gate#m0` (the chunks memory recalls, `read:episodes`; recall by meaning runs the embedder
  and the reranker here), `nmlearn.tuner` (the second look's threshold from the ledger),
  `nmmem.gate#m0` (co-citations), `nmapi.ask` (the judge, when used), `nmengn.placement` (waiting
  for the card gate), `nmengn.model` (models opened), `nmapi.drone_windows` (the intake opened,
  then its index when the code changed); then the drones `nmpod.unit#d0..7`, the tentacles
  `nmpod.unit#t0..7` with `nmbrain.g0#t0..7`, `nmbrain.g1#g1` and `nmbrain.g2#g2` (from the
  stack's own stage clocks, which run back to back; G2 is the stack's time after G1),
  `nmbrain.slot#embedder` (dense lane), `nmengn.model` (weights staged and models opened), each
  `nmbrain.slot#reranker` call, `nmapi.decide` (the second look on the base model) with
  `nmapi.joint_read` inside it (length only), `nmapi.whole#assembly` (the pack, memory recall and
  the memory write, timed together) with `nmmem.gate#m0` (recall), `nmbrain.stack#emit` and
  `nmmem.gate#m0` (write) inside it, the MicroModels' engine time per slot (length only) and
  `nmapi.assemble`.
- Other tools: `nmapi.tools` with the call's whole time from `start_ms` 0, then the parts the tool
  goes through, `"timed": false` and `"inside": "nmapi.tools"` (they run inside the handler and
  have no clock of their own). A context pack answered from the reply store is reported the same
  way.

`parts` (version 1) stays as it was.

A real row (whole path, 2026-11-15, `crates/nmjson`; the eight drone and tentacle steps shortened
to one each):

```json
{"id": "a1794760479265-9092", "kind": "tool", "tool": "nm_context_pack", "mode": "whole",
 "time_ms": 1794760479265, "ms": 12852, "ok": true, "state": "done",
 "steps": [
  {"part": "nmapi.drone_windows", "start_ms": 3505, "ms": 21},
  {"part": "nmpod.unit#d0", "start_ms": 3526, "ms": 4, "shared": "drones' intake"},
  {"part": "nmpod.unit#t0", "start_ms": 3530, "ms": 1, "shared": "windows and g0_tx"},
  {"part": "nmbrain.g0#t0", "start_ms": 3530, "ms": 1, "shared": "windows and g0_tx"},
  {"part": "nmbrain.g1#g1", "start_ms": 3532, "ms": 4},
  {"part": "nmbrain.g2#g2", "start_ms": 3536, "ms": 0},
  {"part": "nmengn.model", "start_ms": 3536, "ms": 1696},
  {"part": "nmengn.model", "start_ms": 5232, "ms": 0},
  {"part": "nmbrain.slot#reranker", "start_ms": 5243, "ms": 3619, "layers": [0, 27]},
  {"part": "nmbrain.slot#reranker", "start_ms": 8863, "ms": 206, "layers": [0, 27]},
  {"part": "nmapi.decide", "start_ms": 9069, "ms": 3256, "layers": [0, 17]},
  {"part": "nmapi.joint_read", "ms": 3234},
  {"part": "nmapi.whole#assembly", "start_ms": 12326, "ms": 556},
  {"part": "nmmem.gate#m0"},
  {"part": "nmapi.assemble", "start_ms": 12882, "ms": 0}
 ]}
```

That row is of version 2: the 3.5 s before the first step had no clock then. In version 3 that
work is timed step by step (above). A whole-path row of version 3 on the fixture repository
(2026-11-15, scratch home, memory on; drone and tentacle steps shortened to one each):

```json
{"part": "nmapi.mcp", "start_ms": 0, "ms": 17, "us": 17538, "timed": true, "cells": ["running row"]},
{"part": "nmmem.gate#m0", "start_ms": 18, "ms": 9010, "us": 9010931, "timed": true,
 "cells": ["read:episodes"], "counts": {"chunks": 3}},
{"part": "nmlearn.tuner", "start_ms": 9029, "ms": 9, "us": 9718, "timed": true, "cells": ["tuner"]},
{"part": "nmengn.placement", "start_ms": 9038, "ms": 0, "us": 5, "timed": true, "cells": ["card gate"]},
{"part": "nmapi.drone_windows", "start_ms": 9038, "ms": 604, "us": 604523, "timed": true},
{"part": "nmpod.unit#d0", "start_ms": 9646, "ms": 3, "timed": true, "shared": "drones' intake",
 "counts": {"chunks": 7, "flagged": 4}},
{"part": "nmpod.unit#t0", "start_ms": 9649, "ms": 0, "timed": true, "shared": "windows and g0_tx",
 "counts": {"window": 7, "reported": 4}},
{"part": "nmbrain.g1#g1", "start_ms": 9650, "ms": 1, "timed": true,
 "counts": {"handover": 4, "queue": 3, "rounds": 1}},
{"part": "nmbrain.slot#reranker", "start_ms": 9798, "ms": 990, "us": 990847, "timed": true, "layers": [0, 27]},
{"part": "nmapi.whole#assembly", "start_ms": 10802, "ms": 1948, "timed": true},
{"part": "nmmem.gate#m0", "start_ms": 10803, "ms": 1736, "us": 1736093, "timed": true,
 "inside": "nmapi.whole#assembly", "cells": ["read:episodes", "read:active", "read:journal"],
 "counts": {"episodes": 5, "reads": 7}},
{"part": "nmmem.gate#m0", "start_ms": 12620, "ms": 120, "us": 120793, "timed": true,
 "inside": "nmapi.whole#assembly", "cells": ["write:episodes", "write:active", "write:journal"]},
{"part": "nmbrain.slot#embedder", "ms": 4085, "us": 4085460, "timed": true, "layers": [0, 27]},
{"part": "nmapi.assemble", "start_ms": 12751, "ms": 0, "us": 93, "timed": true}
```

The 9 s at the start were memory's recall by meaning on the first question of the process (the
embedder and the reranker judging stored questions); the slot steps without a start carry that
engine time.

A plain-path row of version 3 (same repository, memory on): `nmpod.pod#p0` 19 us,
`nmbrain.slot#router` 4 us, the frame's nodes 2 to 252 us each (`nmpod.unit#d0` 166 us),
`nmapi.coordinator` 221 us, `nmmem.gate#m0` recall 205 us, `nmbrain.synthesis` 99 us
(`packets_in` 5, `packets_out` 5), `nmbrain.stack#emit` 20 us, `nmmem.gate#m0` write 97.6 ms
(`write:episodes`, `write:active`, `write:journal`), `nmapi.assemble` 24 us; the call took 100 ms.

### `/api/activity` (version 3)

Per-component activity over a time window, built only from the tool rows already in
`memory/audit.jsonl` (no other state). Query: `window` = `<n>m`, `<n>h` or `<n>d` up to `7d`
(`1h` when absent or unreadable; the interface's two are `1h` and `24h`), `project` = a project
id to count only its calls. The log is read back from its end only as far as the window needs,
and a result is reused for 3 s while the log is unchanged, so it can be polled every few seconds.

```json
{"window": "24h", "window_ms": 86400000, "from_ms": 1794697576000, "to_ms": 1794783976000,
 "project": null, "calls": 10, "done": 10, "stale": 0, "running": 0,
 "timed_steps": 243, "untimed_steps": 12, "tools": {"nm_context_pack": 6, "nm_status": 4},
 "components": {
  "nmpod.unit#t0": {"calls": 6, "steps": 19, "timed_steps": 19, "untimed_steps": 0,
   "total_ms": 5.425, "median_ms": 0.368, "last_ms": 1794783719729,
   "jobs": {"structural": 4},
   "sub": {"nmpod.unit#d0": {"calls": 6, "steps": 6, "timed_steps": 6, "untimed_steps": 0,
            "total_ms": 6.379, "median_ms": 0.513, "last_ms": 1794783719726},
           "nmpod.pipeline#t0": {"calls": 4, "total_ms": 0.796, "median_ms": 0.189, "...": "..."},
           "nmbrain.g0#t0": {"calls": 2, "total_ms": 4.0, "...": "..."}}},
  "nmbrain.slot#reranker": {"calls": 2, "steps": 4, "total_ms": 42899.374, "median_ms": 21449.687,
   "layers": {"0": {"calls": 2, "steps": 4, "total_ms": 42899.374, "...": "..."}, "...": {}}},
  "nmmem.gate#m0": {"calls": 6, "steps": 16, "total_ms": 34370.633, "median_ms": 197.216,
   "sub": {"read:active": {"calls": 3, "...": "..."}, "write:episodes": {"calls": 6, "...": "..."}}},
  "nmpod.unit#t5": {"calls": 0, "steps": 0, "timed_steps": 0, "untimed_steps": 0,
   "total_ms": null, "median_ms": null, "last_ms": null}
 },
 "parts": {"nmapi.assemble": {"calls": 6, "...": "..."}, "nmapi.decide": {"layers": {"0": {}}, "...": "..."}}}
```

- Top level: `calls` counts ended calls in the window (`done` plus `stale`); `running` calls have
  no steps yet and are counted apart; `timed_steps` and `untimed_steps` over every step;
  `tools`: ended calls per tool.
- `components`: every anatomy id (`designs/anatomy.json`), also the ones not used (calls 0, times
  null). A step counts for a component when its part is the component; when it is a part of a
  tentacle or drone instance (`nmpod.pipeline#t0`, `nmpod.pipeline#t0-emit`,
  `nmmem.working#t0`), under that tentacle or drone, also listed in its `sub`; when it has no
  instance (`nmbrain.synthesis`), under the one component of that name.
- Per entry: `calls` (distinct calls that used it), `steps`, `timed_steps`, `untimed_steps`,
  `total_ms` (sum of its timed steps, from `us` when present, else `ms`; null when no step of it
  was timed), `median_ms` (median over the calls of each call's timed total; null when none),
  `last_ms` (when it last ran: the call's start plus the step's `start_ms`, or the call's start
  when the step has none; null when unused). Steps of a shared pass each carry the pass's time.
- `sub` (where it applies): the anatomy children that ran (the Core Brain's components under
  `nmbrain.stack#b0`, a tentacle's drone, `g0_tx` and tentacle core, the tentacles under
  `nmpod.pod#p0`), the instance parts above, and each named `cells` value of its steps (memory
  tiers read or written; the reduce step's components). A sub-entry carries the time of the steps
  that touched it; the time is not split between sub-parts.
- `layers` (MicroModels and other steps with `layers`): per model layer, the steps and calls that
  ran it, with the time of those steps (each step's whole time; not split by layer).
- `jobs` (tentacles on the plain and neural path): calls per job the plan gave it.
- `parts`: every raw step part seen in the window (also the ones that are no anatomy id, such as
  `nmapi.assemble`, `nmapi.decide`, `nmengn.model`), with the same counts and `layers`.

### `/api/memory/tiers` and project `tiers` (version 3)

Records held per tier, read from the tier files of each memory folder: `core` (items in
`1-core/items.json`), `active` and `archive` (items in every topic folder's `items.json` under
`2-active/` and `3-archive/`), `temporary` (notes in `temp/notes.json`).

```json
{"nano": {"core": 0, "active": 0, "archive": 0, "temporary": 0}, "nano_overlap": 0,
 "nano_topics": {"active": 0, "archive": 0},
 "projects": {"repo-c28e80ed": {"core": 0, "active": 5, "archive": 5, "temporary": 0}},
 "projects_overlap": {"repo-c28e80ed": 5},
 "total": {"core": 0, "active": 5, "archive": 5, "temporary": 0}}
```

Each project of `/api/projects` carries `dense`: `{"rows": n, "bytes": n}` of the meaning index kept
on disk (`dense/vectors.bin`; null when there is none); `nano_dense` gives the same for the Nano's
own folder. Each project of `/api/projects` carries the same `tiers`, plus `tier_topics` (topic folders in
the active and archive tiers) and `tiers_overlap`: item ids listed in more than one tier file.
The tier files are written from the memory journal only, which places each memory in one tier,
so the overlap is 0. A folder written by an older version, where the use counts shared the tier
files with the journal (the example above: five answers listed in `2-active/context` and again
in `3-archive/context`), is brought over when it is next opened, and its overlap goes to 0.
`temporary` counts the temporary notes of `temp/notes.json`; working notes captured at task end
are in `temp/captured.json`.

### `/api/cartridges`

Each learned cartridge (a learning unit seen as a cartridge) keeps its label fields and adds:

```json
{"id": "u-rerank-1", "bytes": 3145728, "loaded_bytes": 3145728, "model": "qwen3-0.6b",
 "class": "learning", "layers": [27, 27], "verified": true, "switch_key": "cartridge:u-rerank-1"}
```

- `bytes`: size of the unit's folder. `loaded_bytes`: `bytes` when its card says it is loaded,
  else 0 (the serving process loads it; this server does not see a load in progress).
- `model`: the open weights id of the MicroModel it fits. `class`: `learning` (learning units; the preference layer keeps
  records, not cartridges, so no `preference` cartridge exists yet). `layers`: the first and last
  layer its target touches, or null when its target is not a layer. `verified`: every file of the
  unit matches the hash on its card.

Version 5 appends the cartridge library's cartridges to the same list. Each keeps the fields above
(`class` is `self`; `kind` is the filter kind, `settings` or `affine`; `model` is the MicroModel it
filters, or `settings`; `layers` the layers it filters, or null; `scopes` holds its domain;
`check` is `trial` or `accept`; `loaded` says whether the last start put it in use; `family` is
`fundamentals`, `learning` or `preference`; `quantization` is `q8`, `f32` or `none`) and adds:

```json
{"library": {"key": "op-gtx-1650-4gb@1", "label": {"name": "GTX 1650 operating numbers", "does": "...",
 "model": "settings", "loads_at": "settings", "layers": null, "made_from": "...", "made": "2026-11-16",
 "bytes": 4766, "effect": "..."}, "family": "fundamentals", "domain": "hardware:gpu", "target": "settings",
 "trial": false, "in_use": "served", "stacks": ["gpu-nvidia-geforce-gtx-1650-4096mib@1"],
 "hash": "<sha256 of its file list>", "evidence_rows": 9}}
```

### `/api/cartridges/library` (version 5)

The cartridge library on this machine. A cartridge holds learned parameters on
one knowledge domain, used as a filter over one MicroModel's released weights at named layers, or over the
values its settings take. A stack is the set of cartridges plugged in for one use, applied in order.
Nothing slow is probed on a request: the machine is the one cached in `memory/stacks/machine.json`.

```json
{"exists": true,
 "catalogue": {"schema": "nm-library/1", "cartridges": [ ... ], "stacks": [ ... ],
               "active": {"hardware:gpu": "gpu-nvidia-geforce-gtx-1650-4096mib@1"}},
 "in_use": {"machine": {"pieces": [{"kind": "gpu", "identity": "gpu:nvidia:geforce-gtx-1650:4096mib:driver-581",
                                    "family_key": "gpu:nvidia:turing", "name": "NVIDIA GeForce GTX 1650", ...}, ...]},
            "uses": [{"use": "hardware:gpu", "piece": "gpu:nvidia:geforce-gtx-1650:4096mib:driver-581",
                      "stack": "gpu-nvidia-geforce-gtx-1650-4096mib@1", "trial": false,
                      "why": "stack for this exact piece, gpu:nvidia:geforce-gtx-1650:4096mib:driver-581"}],
            "loaded": [{"key": "op-gtx-1650-4gb@1", "trial": false, "hardware": "gpu:...", "refused": null}],
            "set": {"judge_window": {"value": "384", "from": "op-gtx-1650-4gb@1"}, ...},
            "trial": [],
            "options": ["--fast-judge-window", "off", "--fast-k", "4", "--half-math", "on", ...],
            "refused": []},
 "last_start": { the same "in_use" as written at the last server start }}
```

- `catalogue.cartridges`: each cartridge as its file's label gives it: `id`, `version`, `name`,
  `description` (what it does), `made_from`, `made`, `effect`, `family`, `domain`, `scope`, `model`,
  `weights` (the bare weights' SHA-256), `layers`, `kind`, `target` (the sockets it loads into, comma
  separated, or `settings`), `files` (its one file and SHA-256), `hash`, `notes`, `evidence` (rows
  each tagged with the exact identity of the hardware piece they were measured on), `trial`.
  `catalogue.stacks`: `id`, `version`, `use`, `piece` (for a hardware use), `cartridges` in
  order, `hash`, `trial`. `catalogue.active`: the stack switched on for each use by hand.
- `in_use.uses`: the stack for each use, in the order they apply (general, hardware from RAM to
  graphics card, project, job), and why: switched by hand, the stack for this exact piece, or
  the nearest stack of the same family on trial (`trial: true`) for a piece the cartridge library has no stack
  for. `loaded`: the cartridges loaded; `refused` names why one was not (for example a file that
  fails its hash).
- `set`: the settings the served filters set; `trial`: what trial cartridges would change, never
  applied; `options`: the `serve` options the set values become when not given on the command
  line.
- `last_start` also holds `sockets` (each cartridge plugged into a MicroModel's sockets: `key`,
  `family`, `model`, `points`, `values`, `trial` when tried with `--try-cartridge`),
  `sockets_refused` and `tried`. Only a serving process writes it; bench runs write nothing.

### `/api/projects`

Adds `serving` (one row per serving process: `project`, `project_key`, `root`, `mode`, `pid`,
`started_ms`, `beat_ms`, and `call` with `tool` and `since_ms` while a call runs), `current` (the
project of the most recently active server, or null) and, on each project, `serving: true|false`.

### `/api/session` (version 4)

`project` is a key as `/api/projects` lists it (`nano` for the Nano's own folder); without
`session` the newest session is shown. Computed from the situation record on each request; read
only. Taken from a scripted session on the fixture repository (lists shortened):

```json
{"project": "repo-a8deec05",
 "sessions": [{"session": "demo-1", "first_ms": 1794791508936, "last_ms": 1794791509703, "records": 10}],
 "session": {
  "session": "demo-1", "first_ms": 1794791508936, "last_ms": 1794791509703, "records": 10,
  "model": null, "starts": [{"t": 1794791508936, "source": "startup"}], "compactions": 0,
  "task": {"t": 1794791509639, "sha": "69eb2f454ae1e8e7", "chars": 18, "text": "thanks, that works"},
  "prompts": 2,
  "files": [{"path": "crates/shop/src/lib.rs", "reads": 1, "read_lines": [[13, 19]], "edits": 1,
             "writes": 0, "edit_lines": [[18, 18]], "last": 1794791509375, "sensitive": false}],
  "commands": [{"t": 1794791509484, "kind": "test", "class": "cargo-test", "cmd": "cargo test -p shop",
                "pkg": "shop", "ok": true, "exit": 0, "tests": [3, 0]}],
  "open": {"failing_checks": [], "tool_failures": [], "unchecked_changes": []},
  "served": [{"t": 1794791509142, "tool": "nm_context_pack", "ask": "where is the checkout total computed",
              "episode": "ep-demo",
              "pack": {"id": "a1794791509142-1", "at": 1794791509142, "joined_by": "episode",
                       "cited": ["crates/shop/src/lib.rs:13-19", "crates/inventory/src/lib.rs:1-21"],
                       "window": "closed", "closed": "next turn ended", "in_ledger": true,
                       "joins": [{"rule": "read", "state": "used", "strength": "direct", "steps_after": 0,
                                  "event": {"at": 1794791509241, "event": "PostToolUse", "kind": "read",
                                            "tool": "Read", "place": "crates/shop/src/lib.rs:13-19", "ok": true}}]}}]}}
```

- `task`: the newest prompt; `text` is null when the prompt's text is not kept (`prompts` off, or
  expired after `prompts.days`).
- `files`: each file read or edited, with merged line ranges (`edit_lines` are where the new text
  sits after each edit).
- `open`: `failing_checks` (the newest run of each build or test, by class and package, that
  failed, with located errors), `tool_failures` (a failed tool call not followed by a success of
  the same tool on the same paths), `unchecked_changes` (files edited after the newest passing
  build or test).
- `served[].pack`: the ledger's view of each context pack (see below); `in_ledger` says whether its
  evidence rows are written yet.
- `primes` (version 7; the same list is also given as `nudges`, its version 6 name): each prime
  given in the session: `t`, `event` (the hook), `kind` (`prompt-recall`, `failure-fix`,
  `edit-impact`, `resume`, `open-loop`), `id`, `places`, `from` (memory record ids, `situation` or
  `source`), and `ledger`, the ledger's view of it in the shape of `served[].pack` (its joins end
  with a `followed` row: `used` or `passed-over`).

### `/api/examples` (version 7)

Read only. The training store of a project (see `privacy.md`, Training store): what the Nano keeps
for fitting cartridges later.

```json
{"project": "repo-a8deec05", "folder": "...\\memory\\projects\\repo-a8deec05\\training",
 "setting": "features", "memory": "on",
 "families": {"learning": 12, "preference": 30, "fundamentals": {"hardware_rows": 41, "references": 0}},
 "judged_waiting": 2, "verdicts": {"verified": 7, "refuted": 2, "corrected": 1, "passed-over": 2},
 "strengths": {"direct": 2, "indirect": 8, "weak": 2}, "splits": {"fit": 10, "tune": 1, "held-out": 1},
 "domains": {"code:rust": 12, "project:repo-a8deec05": 12},
 "labels": {"verified": 7, "unused": 9, "judged": 140, "passed-over": 4, "went": 1, "refuted": 2},
 "engines": {"e2b96ccb229b333d": 12}, "preference_kinds": {"edit-kept": 20, "correction": 3},
 "bytes": 48213, "examples": [...], "engines_known": {"e2b96ccb229b333d": {"weights": "...", ...}}}
```

`examples` are rows of `examples.jsonl` (newest `limit`, 50; or the one named by `id`): `id`
(`x-<pack>`), `family` (`learning`), `verdict` (`verified`, `refuted`, `corrected`: the Nano was
wrong and the place Claude Code went to was then verified, `passed-over`), `strength`,
`decided_by`, `split` (`fit`, `tune`, `held-out`, fixed by the hash of the question and the anchor
place), `domains`, `ask_sha`, `anchor`, `engine` (the reranker's identity, `engines_known` describes
it), `path` (`plain`, `fast`, `full`), `settings`, `cartridges`, `stacks`, `loaded`, `hardware`, and
`candidates`: `place`, `hash` (16 hex digits of the SHA-256 of its lines), `judged`, `served`,
`rank`, `score` (the reranker's P(yes)), `bits` (its exact binary32 bits), `index` (in the judged
set), `label` (`verified`, `refuted`, `used`, `unused`, `passed-over`, `went`, `judged`) and
`strength`. With the setting at `full` a row also has `question`; passage texts are not served
here. `POST /api/examples/clear` removes the store (see Writes).

### `/api/outcomes` (version 6)

`{"project", "root_known", "outcomes": [{"record", "kind", "question", "failure", "places":
[{"place", "fresh"}], "verified", "refuted", "strength", "standing", "in_memory"}]}`. `kind` is
`answered` (with `question`) or `fixed` (with `failure`: `check`, `code`, `file`, `line`, `msg`);
`standing` is the newest verdict (`verified` or `refuted`); `in_memory` is false while the outcome
waits in the ledger for the memory journal to take it in; `fresh` says whether the place's lines
still hash as when verified (false when `root_known` is false: the project's root is read from its
`project.json`, written by a serving process).

### `/api/ledger` (version 4)

Read only: the ledger is written by the hook program and the `ledger` command, never by a read.
`pack` selects one pack's rows; `file` selects the rows of packs that cited that file and evidence
rows whose event touched it; `limit` keeps the newest (500).

```json
{"project": "repo-a8deec05", "file": "...\\memory\\projects\\repo-a8deec05\\evidence.jsonl",
 "chain": {"rows": 8, "intact": true, "fault": null, "torn": null},
 "packs": {"served": 1, "window_open": 0},
 "guidance": {"given": 2, "window_open": 1, "rules": ["step-followed", "loop-met", "glob-check"],
              "by_kind": {"step": {"used": 1}, "reminder": {"none": 1}}, "deliveries": [...]},
 "states": {"none": 3, "used": 2, "verified": 2},
 "counts": {"read": {"used/direct": 1}, "edit": {"used/direct": 1}, "elsewhere": {"none": 1},
            "check": {"verified/indirect": 1}, "revert": {"none": 1},
            "next-prompt": {"verified/weak": 1}, "asked-again": {"none": 1}},
 "rows": [...], "pending": [], "written": 0}
```

A `served` row (one per pack, written when the pack is first seen):

```json
{"v": 1, "seq": 1, "row": "served", "pack": "a1794791509142-1", "session": "demo-1", "at": 1794791509142,
 "tool": "nm_context_pack", "joined_by": "episode", "episode": "ep-demo", "ask_sha": "8332dc9df8206c94",
 "cited": ["crates/shop/src/lib.rs:13-19", "crates/inventory/src/lib.rs:1-21"],
 "steps": ["nmapi.mcp", "nmapi.whole", "nmbrain.slot", "nmapi.assemble"],
 "settings": {"path": "whole", "via": "serve", "mode": "full", "memory": "on", "dense_cache": "on",
              "precision.base": "default", "precision.writer": "default",
              "precision.reranker": "default", "precision.embedder": "default", "cartridges": []},
 "withheld": false,
 "hardware": {"id": "9cf7d12253d4b1f3", "cpu": "Intel64 Family 6 Model 154 Stepping 3, GenuineIntel",
              "threads": 12, "gpu": "NVIDIA GeForce GTX 1650", "gpu_mib": 4096, "ram_mib": 32436,
              "used": ["cpu", "gpu"], "used_from": "serving mode"},
 "t": 1794791509720, "prev": "000...000", "h": "..."}
```

`hardware` is this machine's description from `memory/hardware.json` (read again after a day).
`ram_mib` is null on Windows in rows a hook wrote before the `ledger` command first ran (reading
the RAM size there takes half a second, which a hook does not spend); `id` names the processor,
thread count and graphics card, not the RAM, so it is the same either way. `used` is inferred from
the serving path: the plain path runs on the processor, the neural and whole paths also on the
graphics card.

An `evidence` row (one per rule, all written when the pack's window closes):

```json
{"v": 1, "seq": 3, "row": "evidence", "pack": "a1794791509142-1", "session": "demo-1",
 "rule": "edit", "state": "used", "strength": "direct", "steps_after": 1,
 "event": {"at": 1794791509375, "event": "PostToolUse", "kind": "edit", "tool": "Edit",
           "place": "crates/shop/src/lib.rs:18", "ok": true},
 "closed": "next turn ended", "hardware": "9cf7d12253d4b1f3", "t": 1794791509720, "prev": "...", "h": "..."}
```

- `settings`: replayed from the audit log's setting changes: controls that apply at the next start
  as of the serving process's start, the answer `mode` as of the call; `path` is the serving path
  (plain, neural, whole); `cartridges` those switched on at the server's start.
- `steps`: the parts that ran for the pack, each once, in run order.
- `joined_by`: how the hook record of Claude Code's call was matched to the Nano's audit row:
  `episode`, `question and time`, `time`, or `none` (then `cited` is empty and the place rules
  record `unknown`).
- `state`: `used`, `passed-over`, `verified`, `refuted`, or `none` (the rule found nothing) or
  `unknown` (it could not be judged; `why` says why). `strength` is `direct`, `indirect` or
  `weak`, and null for `none` and `unknown`.
- `steps_after`: completed tool calls between the pack and the event.
- `closed`: why the window closed (`next turn ended`, `session ended`, `conversation cleared or
  started again`, `an hour with no activity in the session`, `a day after the pack`).
- `detail`: `went` (passed over: where Claude Code went instead), `runs` (checks passed and failed
  in the turn), `after_cited_edit`, `marker` (the fixed word or phrase matched in the next prompt),
  `by` (asked again: by the same question to the Nano, or the same prompt from the user).
- `prev` and `h`: the previous line's hash and this line's (SHA-256 of the line without `h`).
- `pending`: packs whose window is still open, with their joins so far (not in the file).

The rules, their states and strengths:

| Rule | Finds | State and strength |
|---|---|---|
| `read` | A cited place read in the rest of the pack's turn (Read, or a reading shell command) | `used`: `direct` when the lines read overlap the cited lines (a cited whole file overlaps any), `indirect` when only the file matches |
| `edit` | A cited place edited or written in the turn (lines before or after the edit) | `used`: `direct` or `indirect` as for `read` |
| `elsewhere` | No cited place read or edited, other files were | `passed-over` `indirect`, with `went` |
| `check` | The last build or test in the turn | `verified` (passed) or `refuted` (failed): `indirect` after an edit to a cited file, else `weak` |
| `revert` | A `git revert`, `restore`, `checkout`, `reset` or `stash` after an edit to a cited file | `refuted`: `direct` when it names that file, `weak` when it names no file or the whole tree |
| `next-prompt` | Your next prompt (its kept text) | `refuted` `weak` on a correction marker, `verified` `weak` on an acceptance marker |
| `asked-again` | The same question asked again, in the turn or the next | `refuted`: `indirect` when asked of the Nano again, `weak` when your prompt is repeated word for word |

Guidance deliveries (v0.1.0). Each reminder delivered, step nudged and loop continuation (a stop a
loop blocked) named on a hook record's `guidance` field gets a `guided` row when first seen and one
evidence row by its kind's rule when its window closes. Its window is the rest of its session: it
closes when the session ends or is cleared, after an hour with no record in the session, or a day
after the delivery (`closed` says which). Its pack id is `guide:<kind>:<item>:<session, 8
characters>:<time>`.

```json
{"v": 1, "seq": 9, "row": "guided", "pack": "guide:step:parse:demo-1:1794791509142", "session": "demo-1",
 "at": 1794791509142, "kind": "step", "item": "parse", "hook": "UserPromptSubmit", "channel": "context",
 "hardware": "9cf7d12253d4b1f3", "t": 1794791509720, "prev": "...", "h": "..."}
```

A loop continuation's row also carries `iteration` (`[k, max]`).

| Rule | Delivery | State and strength |
|---|---|---|
| `step-followed` | A step nudge | `used`: `direct` when the step is marked done later in the session (by the Nano after its check passed, or by the user or the agent), `indirect` when its check passes later in the session after an edit under its paths, `weak` when its check passes with no such edit; `passed-over` once the window closed without that (`indirect` when its check ran and did not pass, `weak` otherwise); `unknown` when the step is no longer in the graph |
| `loop-met` | A loop continuation | `verified` `direct` when a later stop of the session ended the loop as done (its condition met before the cap), `refuted` `direct` when it ended exhausted (the cap reached first); `detail` holds `iteration`, `ended_at` and `max`; `unknown` when the loop was ended by hand or the session ended before it was judged again |
| `glob-check` | A reminder tied to an edit glob (its `edit` trigger's glob, else its file scope) | `verified` or `refuted` `indirect`: the last build or test after the last edit under the glob, later in the session, passed or failed; `detail` holds `glob` and `edits`; `unknown` for a reminder with no glob, one no longer set, or no edit under the glob or no check after it before the window closed |

The answer's `guidance` block: `given`, `window_open`, `rules`, `by_kind` (counts by delivery kind
and state, as judged now) and `deliveries`, the newest judged deliveries (at most 200, and at most
`limit`) in the shape of `/api/guidance`'s `verdict`. Open guidance deliveries are also in
`pending`, with `joined_by: "guidance"`.

`POST /api/ledger/clear` (see Writes) removes the ledger; packs served before it are not written
again.

### `/api/cartridges/bases` (v0.1.0)

`{"policy", "applies": "next start", "models": [...]}`. `policy` is the `library.upgrade` setting
(`keep` or `replace`). One row per MicroModel (`model`, its open weights id): `serving` (the base
cartridge key that serves, or `open weights`), `choice` (the user's choice: a key, `open`, or
`auto` for the newest usable version) and `versions`, newest first, each with `key`
(`<id>@<version>`), `version`, `made`, `bytes`, `present` (its file is on disk), `refused` (why it
cannot serve, or null) and `serving`. Choosing and removing a version are writes (below).

### `/api/cartridges/parameters` and the parameters section (v0.1.0)

One read format, `nm-parameters/1`, for every object that holds learned parameters, whether or not
it is a cartridge in the catalogue. Keys:

| Key | Object | Tensors |
|---|---|---|
| `<id>@<version>` (library) | A socket cartridge (`nmengn-sockets/1`, any family, adapters included) | `<node>.gain` and `<node>.bias` per point; layer from the node id (`L27.swigate`) |
| `<id>@<version>` (library) | A preference cartridge | One per `pref.*` key: `[good, signals]` (shape `[2]`, type `count`) |
| `<id>@<version>` (library) | A base cartridge | Every parameter set by name, layer, block, shape and type; values read for parts of at most 256 KiB, larger parts paged from the file; sets taken from the open weights are not held |
| `<id>@<version>` (library) | An operating settings cartridge | None: `holds` is false and `note` is `this cartridge holds no parameters` |
| `filter:<role>/<job>` (or `filter:<cartridge>` as `compat` names it) | A job filter, `library/parameter-maps/<role>/job-filter-<job>.json` | Gain and bias per point, as a socket cartridge |
| `<id>@<n>` or `<id>` (newest) | A project adapter in the adapter library: the embedder's lens, the judge's code read-out | Lens: `flat [k, d]`, `beta [k]`, `down [r, d]`, `up [d, r]`; read-out: `w [d]`; no layers |
| `<unit id>` | A learning unit of the registry | `down [rank, k]`, `up [n, rank]`; the layer of its target node |

Without `key`, the answer is `{format, objects, roles, read, edit, overlay}`, read from labels and
the catalogue alone: no tensor, value or norm is read for the list. One row per object (`key`,
`source` `open-weights` \| `library` \| `job-filter` \| `adapter` \| `unit`, `kind` `open` \|
`filter` \| `adapter` \| `lens` \| `readout` \| `unit` \| `preference` \| `base`, `model`,
`role`, `name`, `state`, `editable` (as a new version), `layer_range` where the label records one,
`bytes` on disk, `read: false`, and `overlay`: `{edits, masks, freezes}` or null).

With `key` and `head=1`: `{key, tensors, root}`, the count of tensors opening the object reads, from
the open weights' safetensors header or a base cartridge's label alone (null where only opening the
object tells), so a view can say what it is reading.

With `key`, the parameters section (also `parameters` in `/api/cartridges/inspect`):

| Field | Meaning |
|---|---|
| `format`, `key`, `source`, `kind`, `model`, `role`, `name`, `state`, `path` | What the object is and where it is; `state` is `trial` or `served` (library), `plugged in` or `plugged out` (job filter), the newest or an earlier version (adapter), or a unit's status |
| `holds`, `note` | Whether it holds parameters; a plain statement when it holds none (not an error) |
| `editable` | `scale`, `layers`, `values` (booleans), `why` (the reason when it is read only), `layer_off` (whether switching a layer off drops its points or zeroes its change), `scale_max` (4), `small` (65,536: the most elements a tensor may have for direct value edits) |
| `totals` | `tensors`, `read`, `values`, `l2` |
| `layers` | Per layer: `tensors`, `read`, `effect_l2` (the L2 norm of what the parameters change: the difference from 1 for a gain, from 0 otherwise), `l2` |
| `tensors` | Per tensor or point: `name`, `target`, `layer`, `part`, `shape`, `dtype`, `size`, `bytes`, `identity`, `scales`, `read`, `paged`, `values_editable`, `l2`, `max_abs`, `effect_l2`, `summary` (`min`, `max`, `mean`, `histogram` of 8 bins with `lo`, `hi`, `counts`), and `unread` (why the values were not read) |
| `root` | The Nano's folder, so a view can show `path` relative to it |
| `overlay` | The object's parameter overlay (below): `entries`, `counts`, per-tensor `tensors` (`edited`, `masked`, `frozen` value counts), per-layer `layers` (`edited`, `masked`, `frozen`), `whole_masked`, `takes_effect`, `file` |
| `held_out` | What a held-out check of the object needs (a job filter's split gate runs in its build on the graphics card; an adapter's control in its idle fit); none runs from this route |

With `key` and `tensor`: `{key, tensor, shape, dtype, size, offset, limit, values}`, at most 4,096
values per page. Figures are JSON numbers at the precision the parameters are held in (float32).

With `key` and `path` (and `offset` within the selection): a region, `{key, path, tensor, target,
layer, shape, dtype, identity, selected, offset, indices, values, summary, distribution, maps}`: at
most 4,096 of the selected values of the first tensor the path names, read from memory or by ranged
reads of a base cartridge's file; `summary` of the selection (`l2`, `max_abs`, `effect_l2`, `min`,
`max`, `mean`); `distribution`, the whole tensor's 8-bin histogram (`lo`, `hi`, `mean`, `counts`);
`maps`, the values parameter maps hold at the same place (`{map, at: "same point", values}` for a
job filter map, `{map, at: "layer <n>", value}` for a per-layer map). Paths: `*`, `L3`,
`L3.o.gain`, `L3.o.gain[4]`, `L3.o.gain[0:16]`, `down[2:4, 0:8]`, `down[2:3, :]`, `down[:, 5:6]`.
`overlay` is `[value, masked, frozen]` per index with the object's overlay applied, or null when the
overlay changes none of them.

### `/api/parameters/overlay` (v0.1.0)

The parameter overlay of one object (`?key=`, the keys above, `open:<model>` included): format
`nm-overlay/1`, kept in `memory/overlays/<key with unsafe characters as _>-<8 hex of its
SHA-256>.json`, an ordered list of entries. The object's files are never written; the engine applies
the overlay when it seats the object. `GET` answers `{format, key, editable, file, entries, counts,
tensors, layers, whole_masked, takes_effect, route}`; each entry is `{id, action (edit, mask,
freeze), op, selection, whole, regions: [{path, tensor, values, identity, captured}], values, by,
at_ms, note}`. `takes_effect` is `{view, when, reason}`, `when` one of `applied now`, `applied at
the next call`, `applied at the next start`.

Replay: edits in entry order, each result rounded to the type the engine holds the values in (BF16
for the open weights and base cartridges, f32 otherwise); masks then set their regions to the
identity (0; 1 for a gain and a norm weight of the open weights or a base cartridge); freezes change
nothing. A paste, copy, interpolation, blend (`source` and `source_b`) or import is kept as its
captured values (at most 16 Mi per region; more than 4,096 are kept in
`memory/overlays/values/<sha256>.f32` beside the file and checked by hash when read); structural
operations (drop of a point, add_point, rank) are refused here and made as a new version.

A serving process applies a change at its next call: job filters, plugged cartridges, preference
values, the lens and read-out, and loaded learning units in place; the open weights and base
cartridges by binding that MicroModel again where the call starts, its time recorded in
`memory/overlays/state/reloads.json` and stated in `takes_effect.reason`.

With `key` and `maps=1`: `{key, role_dir, base, maps}`, the parameter maps of the object's
MicroModel that relate to it, each `{file, kind, bytes, made_on, used_by, purpose, why, targets,
layers: [{layer, value}], read}`.

### `/api/cartridges/compat` (v0.1.0)

`{"models": [installed open weights ids], "cartridges": [...]}`. One row per cartridge in the
catalogue and per job filter: `key` (job filters as `filter:<cartridge>`), `name`, `kind` (`base`,
`learned`, `preference`, `lens`, `settings`, `filter`), `made_for` (an open weights id, or `engine`
for cartridges that set the engine's own values), `role`, `fits` (the models it fits) and `not`
(each other model with why: made for another model, made for other open weights of the same model,
or, for a compensation cartridge, made for another base cartridge version than the one serving).
Cartridges stay with the model they were made for: when a role is moved to another model, it runs
with that model's cartridges and finds its own again when it goes back.

### `/api/models/roles` (v0.1.0)

`{"applies": "next start", "roles": [...], "models": [...]}`. Each role (`reader`, `writer`,
`embedder`, `reranker`) with `model` (the model it runs on from the next start), `default`,
`needs` (the kind of model the role needs), `locked` (why the role cannot be moved, or null) and
`choices` (each installed model with `ok` and `why` not). `models`: each installed model with its
`kind` and the `role` it serves (or null).

### `/api/downloads` (v0.1.0)

`models`: each installed weight card's open weights with `id`, `repo` and `revision` (the Hugging
Face repository and revision the card pins), `bytes`, `files`, `present`, `partial`, `on_disk` and
`state` (`installed`, `partial`, `missing` or `unknown`). `cartridges`: each serving base cartridge
the serving list (`designs/base-cartridges.json`) names, with `key`, `present`, `on_disk` and its
`path` in the Nano folder. Also `cartridge_repo` (the Hugging Face repository the base cartridges
come from), `token` (whether a Hugging Face token is on this machine; never its value), `air_gap`
and `job` (the download in progress or the last one). The official base cartridges come from the
official library on Hugging Face, `gavriel-tech/neuromorph-nano-cartridges`; the Qwen3 open weights
(Apache-2.0) are needed beside them (each base cartridge is checked against them when it is
seated) and are the fallback in open mode. A file is placed only once its SHA-256 matches; a partial
file resumes; one download runs at a time; nothing is fetched while the air gap is on.

### `/api/cartridge-sources` (v0.1.0)

`{"sources": [...], "air_gap", "token_found", "fallback", "file"}`. A source is a Hugging Face
repository that holds cartridges: the official Neuromorph library (`id` `official`, built in) and
every third-party library the user added by its repository name (`id` is the repository with `/`
written as `--`). Each source has `id`, `repo`, `kind` (`model` or `dataset`), `revision`,
`official`, `token` (whether the user's Hugging Face token is sent to huggingface.co for it),
`added_ms`, `url` and `listing` (the last listing, or null). `fallback` names the Qwen3 open
weights. Nothing is fetched until the user asks. A listed file is classified as a stack file
(imported on trial through the cartridge library's checks), a parameter map under `parameter-maps/<role>/`
(imported through the map checks, never over an existing map), or a base cartridge (listed, but in
v0.1.0 installed only from the official library through Downloads); anything else is listed with
why it is not installed.

### `/api/parameter-maps` (v0.1.0)

`{"roles": [...]}`, one row per role (`reader`, `writer`, `embedder`, `reranker`) with `role`,
`model`, `serving_base` (the base cartridge that MicroModel serves now, `id@version`, or null),
`maps`, `archived` and `cleanup` (the count and bytes of maps in the states `experiment`, `history`
and `stale`, the ones a cleanup would look at). The maps live in
`library/parameter-maps/<role>/`. Each map has:

- `file`, `kind` (`parameter map`, `activation map`, `separation map`, `tolerance`, `block map`,
  `role weights`, `energy record`, `super-weights`, `surgery probe`, `surgery ledger`, `job
  filter` or `other`), `about`, `bytes`, `modified_ms` and `summary` (the jobs, layers and sockets
  it covers, as far as its shape says).
- `purpose`: `serving`, `base-build`, `job-filter`, `probe`, `measurement` or `experiment`, or
  empty when not recorded.
- `lineage`: where it came from (null when there is no record): `made_on` (a base cartridge
  `id@version`, or `open:<model>` for the open weights), `made_by` (the command that made it),
  `made_ms`, `purpose`, `used_by` (what was built from it), `recorded` (`tool` when the tool that
  made it wrote it, `backfill` when it was proven afterwards from the map and the cartridges,
  `user` when stated in the interface or over MCP) and `note`.
- `state` and `state_why`: `current` (made on, or building, what serves now; or measured on the
  open weights this MicroModel runs), `stale` (made on another base cartridge version than the one
  serving, or on other open weights), `history` (it built an earlier version only), `experiment`
  (nothing that serves or builds reads it), `log` (a log, such as the surgery ledger) or
  `origin not recorded`.

So the lineage reads both ways: from a map to what was built from it (`used_by`), and from a
cartridge back to the map it was cut from. Archived maps (`<role>/archive/`) are listed with their
lineage; nothing serves or builds from there. Archive, restore, delete, edit, import and stating a
map's lineage are writes (below).

### `/api/surgery` (v0.1.0)

`{"roles": {...}, "ledger": [...], "job_filter_build": {...}}`. Per role that has any: `probe` (the
surgery probe, layer by layer drift of the base cartridge against the open weights),
`super_weights` (how many are pinned), `ledger` (the surgery ledger rows) and `job_filters`. Each
job filter row has `cartridge`, `job`, `base`, `model`, `base_cos`, `filtered_cos`,
`base_max_diff`, `filtered_max_diff`, `enabled` (plugged in), `layers` (how many it covers), and:

- `built_on`: the base cartridge version the filter was built on (its lineage `made_on`), or null
  when not recorded;
- `serving_base`: the base cartridge serving its MicroModel now, or `open weights`;
- `base_match`: whether the two are the same, or null when either is unknown.

These three are reported only: the filter loads either way. `nm_cartridges` `op: filters` returns
the same rows. `job_filter_build` is the build in progress or the last one (`status` `idle`,
`running`, `done`, `error` or `stopped`, with `role`, `job`, `by`, `started_ms`, `finished_ms`,
`updated_ms` and `message`); it is shared through `memory/job-filter-build.json`, so a build started
over MCP shows in the interface and the other way round, and a build whose process ended reads as
`stopped`.

### `/api/learned-units` (v0.1.0)

Every learned unit by compartment, each plugged in or out. `{"file", "total", "acting", "applies",
"compartments": [...]}`; `file` is the plug file (`units/plugs.json`), and a change applies at the
next call of the serving process. A compartment is `nano` (the Nano's own folder), `shared` (the
registered learning units) or a project key; each has `key`, `label`, `root`, `holds`, `on`,
`switch_key` (`compartment:<key>`) and `units`. Each unit has `key` (`<compartment>/<id>`), `id`,
`kind` (`adapter`: a project adapter such as the Embedder's lens; `piece`: a learned piece of
`nm_unit`; `unit`: a registered learning unit), `compartment`, `plugged`, `acts`, `why_not` (for
example `its compartment is switched off` or `unplugged`) and `switch_key`
(`learned:<compartment>/<id>`, or `cartridge:<id>` for a registered unit), with its own fields.
Switching a compartment off takes every unit in it out, each keeping its own plug, so switching it
on again brings back exactly what was plugged in. An unplugged unit's outputs are the base's, bit
for bit.

### `/api/live` and `/api/live/stream` (v0.1.0)

`/api/live` is the live state of the busiest (else the freshest) serving process, with `state`
(`busy` or `idle`), `active` (the stages running, each tagged with its process), `timeline` (the
recent events of every serving process, merged in time order), `serving` (how many) and
`processes` (`pid`, `project`, `state`). With no serving process it is this server's own state and
`backend` is null. `/api/live/stream` sends Server-Sent Events: first `state` with the whole
`/api/live` reply, then `live` for each new timeline event and `now` whenever the active stages,
the MicroModels executing or the idle writer change.

### Anatomy

`/api/settings` `anatomy` has the nodes `nmbrain.g2#g2` (G2) and `nmmusc#g3` (G3, pattern
recognition and muscle memory), with the edges G1 to G2, G2 to `nmbrain.rivers#b0`, G2 down to
each tentacle (feedback) and G2 to G3 and back. Per-model layer and head-group counts and the
layers on the card are in `/api/status`.

## Writes

Every write is logged in `memory/audit.jsonl`. Most answer with `audit_id` and the view they
changed; `/api/undo` with that id undoes it where the table says so.

### Deletes are confirmed in two steps (v0.1.0)

A write that deletes something first answers with a preview and changes nothing:
`{"step": "preview", "deleted": false, "preview": {"title", "items", "bytes", "files", "undo",
"note"}, "confirm_token", "expires_ms", "confirm_word", "next"}`. Only the same request sent again
with that `confirm_token`, within two minutes, deletes. A token works once and only for the request
it was made for. This applies to `/api/storage/delete`, `/api/storage/prune`, `/api/dense/clear`,
`/api/ledger/clear`, `/api/examples/clear`, `/api/cartridges/bases/remove`,
`/api/parameter-maps/remove`, `/api/projects/forget`, `/api/memory/forget`,
`/api/audit/retention`, `/api/comprehension/refresh`, `/api/surgery/job-filter` with `action:
delete`, `/api/units` with `action: unregister`, `/api/lenses` with `action: remove` and
`/api/cartridge-sources` with `action: remove`. The MCP tools borrow these routes, so their deletes
pass the same gate.

### Settings and undo

| Path | Body | Effect |
|---|---|---|
| `/api/controls` | `{"key": "...", "value": "..."}` | Sets a control; answers `{audit_id, controls}`. The key is a control's key, a switch (`cartridge:<id>`, `learned:<compartment>/<id>`, `compartment:<key>`; `on` or `off`), or one entry of a map `settings.json` holds, as `/api/controls/saved` lists them (`set_key`): `base_cartridges:<model id>` (a base cartridge key, `auto` or `open`), `role_models:<role>` (a model id or `default`), `base_compensation:<model id>` (`on` or `off`) |
| `/api/undo` | `{"id": "<audit id>"}` | Undoes a setting change (cartridge, learned unit and compartment switches too), a cartridge library change (stack switch, trial, stack, values, import), a job filter change, a parameter map edit, import, lineage, archive or restore, a learning unit unregistered, a memory pin, an objective change, or a cartridge source added or removed. The undo is itself logged |
| `/api/limits` | `{"key": "vram_mb", "value": "auto"}` (`vram_mb`, `ram_mb`, `cpu_threads`, a memory area `content_mb`, `caches_mb`, `weights_mb`, or the full `limits.` key) | Sets one hardware limit or memory area (v0.1.0; a set of areas above the process budget is refused) |
| `/api/limits/apply-recommended` | `{"confirm": true}` | Writes this machine's recommended limits (v0.1.0) |
| `/api/storage/settings` | `{"key": "disk_floor_gb", "value": "8"}` | Sets the disk floor, the one storage setting (v0.1.0) |
| `/api/audit/retention` | `{}` | Trims the audit log's tool rows to `audit.keep_days` days of use; every other row stays (two-step delete) (v0.1.0) |

### Cartridges, MicroModels and downloads

| Path | Body | Effect |
|---|---|---|
| `/api/cartridges` | `{"id": "...", "on": true}` or `{"id": "...", "value": "off"}` | Switches a cartridge on (slot check: its base hash equals the model's, files hash-checked) or off; logged as setting `cartridge:<id>`; answers `{audit_id, cartridges}` |
| `/api/cartridges` | `{"library": "<use>", "stack": "<stack id>"}`, or `"stack": null` (version 5) | Switches the stack of cartridges for a use (`hardware:gpu`, `job:coding`, ...), or the use off; audited as a `library` row; a running server plugs the new stack's socket cartridges where its next call starts, and its settings at the next start; answers `{audit_id, cartridges}` |
| `/api/cartridges/bases` | `{"model": "<open weights id>", "choice": "<key>" \| "auto" \| "open"}` | Chooses what serves for one MicroModel from the next start: a base cartridge version, the newest usable one, or its open weights (open mode) (v0.1.0) |
| `/api/cartridges/bases/remove` | `{"key": "<id@version>"}` | Removes one base cartridge version and its file; the version serving now and one chosen in settings are refused (two-step delete) (v0.1.0) |
| `/api/cartridges/values` | `{"key": "...", "values": {"<name>": "<value>" \| null}}` | Sets values of a settings or preference cartridge; the edit becomes a new version of the cartridge and of every stack holding it (v0.1.0) |
| `/api/cartridges/parameters` (operations) | `{"key", "op", "path" or "selection": {tensor, layer, target, part, range, index, rows, cols, head, head_dim, channel}, "value", "lo", "hi", "values", "file", "source": {key} or {map}, "t", "target", "width", "rank", "name", "description", "preview"}`, or `"ops": [...]` of such operations in order | The same edit through the selection engine: `set`, `add`, `mul`, `scale`, `clamp`, `zero`, `identity`, `negate`, `paste`, `import`, `copy`, `interpolate`, `drop`, `add_point`, `rank` (see Cartridges and parameters). `preview: true` answers `{preview, key, edit, diff, sockets, writes}` and writes nothing; `diff` is `{tensors: [{name, old_shape, shape, changed, old_l2, new_l2, old_effect_l2, new_effect_l2, max_change, sample: [[index, old, new]]}], changed, added, dropped}`. A point at a node the MicroModel lacks is refused; preference counts are written through `/api/cartridges/values` (v0.1.0) |
| `/api/parameters/overlay` | `{"key", "action", ...}` | One change of an object's parameter overlay. `action`: `edit` (an operation as `/api/cartridges/parameters` reads it: `op` with `path` or `selection`, `ops`, `scale`, `layers`, `values`), `mask` and `freeze` (`path` or `selection`; none, `*` or `whole: true` for the whole object), `unmask` and `unfreeze` (`id`, `ids`, or a `path` meeting the entries), `clear` (every entry; through the delete gate: a preview with a `confirm_token`, then the same request with it; refused while a freeze is held), `undo` (the newest change of the key). `preview: true` answers `{preview, change, added, removed, diff, writes, takes_effect}` and writes nothing. A change touching a frozen region is refused (400) with the freeze named. Answers `{changed, audit_id, key, action, change, added, removed, takes_effect, overlay}`; one `parameter-overlay` audit row, undone by `/api/undo` (what it added out, what it removed back at its place) (v0.1.0) |
| `/api/cartridges/parameters/export` | `{"key", "path"}` | Writes the selected values (at most 65,536) to `exports/parameters/` as `nm-parameter-values/1` (`key`, `path`, `tensor`, `shape`, `indices`, `values`); nothing of the object is written; answers `{file, tensor, values, format}` (v0.1.0) |
| `/api/cartridges/parameters` | `{"key": "<key>", "scale"?: <0 to 4>, "layers"?: {"<n>": "on" \| "off" \| <scale>}, "values"?: {"<tensor>": [every value] \| {"offset": <n>, "values": [...]}}}` | Edits an object's parameters into a new version beside it; the source is never written. A strength moves every scalable tensor toward its identity (`identity + s * (x - identity)`: a gain toward 1, a bias, a lens's `beta` and `up`, a read-out's `w` and a unit's `up` toward 0); `off` drops a layer's points (socket cartridges, job filters) or zeroes its change (units); values are shape checked and edited directly up to 65,536 elements. The new version: a library cartridge version on trial (no stack changes), a job filter file `job-filter-<job>-p<n>.json` plugged out, the next adapter library version, or a candidate unit `<id>-p<n>` (the registry refuses one while the source unit is open on the same node). Base cartridges, preference counts (set as values) and settings are refused. Answers `{changed, audit_id, from, key, file, edit, state, compat, parameters}`; `compat` is the new version's `/api/cartridges/compat` row (library, job filter), its shape and values-key check (adapter) or its base hash check (unit). Audited as a `cartridge-parameters` `edit` row; undo sets the new version aside (`library/cartridges/set-aside/`, or the role's map archive for a job filter) or retires the new unit, deleting nothing (v0.1.0) |
| `/api/cartridges/trial` | `{"name": "<stack>", "trial": true \| false}` | Puts a stack on trial, or serves it (serving needs its evidence) (v0.1.0) |
| `/api/cartridges/stack` | `{"id", "use", "cartridges": [keys in order], "description"?, "switch"?}` | Makes a stack, or a new version of one, through the cartridge library's stacking order; a new stack starts on trial (v0.1.0) |
| `/api/cartridges/import` | `{"path": "<stack file on this machine>"}` | Imports a stack file exported by `library export`; every file hash-checked; arrives on trial; a base cartridge is refused (v0.1.0) |
| `/api/models/roles` | `{"role": "reader" \| "writer" \| "embedder" \| "reranker", "model": "<id>" \| "default"}` | Chooses the model a role runs on from the next start; answers `{audit_id, line, roles}` (v0.1.0) |
| `/api/downloads` | `{"what": "models:<id>" \| "cartridges", "only"?: "<name suffix>"}` | Starts a download in the background: one model's open weights, or the serving base cartridges from the official library; refused while the air gap is on or a download runs (v0.1.0) |
| `/api/training/distil/teach` | `{"action": "start", "set", "prompts", "instruction"?, "licence"?}`, `{"action": "next", "set", "count"?}` or `{"action": "answer", "set", "answers": [{"prompt_id" \| "prompt", "completion", "verdict"?, "paths"?, "tool"?, "score"?: 0..1}], "teacher"?, "licence"?}`; `GET` lists the sets | A live teacher: opens a teaching set, gives its open prompts, takes the teacher's answers into the teacher data (audited; undone in Audit). A local model answers a set through `/api/lab/build` with `{"kind": "teach", "set", "teacher"}` (v0.1.0) |
| `/api/backups` | `{"action": "add", "url", "parts"?, "token"?}`, `{"action": "remove" \| "backup" \| "snapshots", "id"}`, `{"action": "preview", "id", "commit"}` or `{"action": "restore", "id", "commit", "confirm": true}` | Adds one of your repositories, takes it off the list (its contents stay), backs up the chosen parts as a new snapshot and sends it, lists the snapshots, shows what a restore would write, or restores one (each changed file copied aside first; Audit undoes it) (v0.1.0) |
| `/api/cartridge-sources` | `{"action": "add", "repo": "<owner/name>", "kind"?, "revision"?, "token"?}`, or `{"action": "remove" \| "browse", "id"}`, or `{"action": "fetch", "id", "path"}` | Adds a third-party Hugging Face library, removes one (two-step delete; what was imported from it stays), lists a source's files (one request to huggingface.co), or fetches one listed file and imports it through its checks (v0.1.0) |

### Stated preferences (v0.1.0)

| Path | Body | What |
|---|---|---|
| `/api/preferences/say` | `{text, name?, scope?, cartridge?, project?, plug?, trial?, confirm?}` | Without `confirm`: `{step: "preview", rules, cartridge, kept}` and nothing is written. With `confirm: true`: a new version of the cartridge (the one `cartridge` names, or the one the name gives, made when new), plugged in unless `plug: false`; `{step: "written", key, audit_id, rules, cartridge}` |
| `/api/preferences/edit` | `{id, add?, rule?, text?, strength?, drop?, name?, scope?, confirm?}` | Adds rules from new words, rewrites rule `rule` from `text`, sets want `rule`'s `strength` (0.1 to 2.0, a number or its text), drops rule ids, renames, rescopes; a preview first, then `confirm: true` writes a new version that stays plugged in (a strength edit is audited as `preference-edit` with `strength: {rule, to}`) |
| `/api/preferences/remove` | `{id, confirm_token?}` | A delete in two steps (above): every version, the stacks that hold only it and its switched use leave the library; the files stay, so Audit undo puts it back |
| `/api/preferences/plug` | `{id, on}` | Switches the cartridge's stack use on or off (the library switch, audited as a `use` row) |
| `/api/preferences/apply` | `{text, path?, project?}` | The rules in force over a draft: `{text, changed, mode, changes, violations, wants_missed, guidance, rules_in_force}`; prose is transformed outside code and evidence lines, a code path is only checked; `wants_missed` lists each wanted term in scope the draft does not hold (`{cartridge, rule, term, strength, what}`, information, not a violation) |

Each rule row carries `id`, `kind`, `label`, `said`, `does`, `enforcement` (`enforced`, `checked`,
`biased` or `guidance`), `scope`, `scope_text` and `scope_words`, and the fields of its kind. The
stored kind of an avoided word stays `ban` (`phrase`), shown with `label: "avoid"`. A want is
`kind: "want"` with `term` or `style` (one of them empty) and `strength` (a number, 0.1 to 2.0,
0.5 by default); a want with a term is `biased`, one with a style is `guidance`.

A scope is `global`, or parts joined by `;`: `project:<key>`, `kinds:<ext>,<ext>`,
`folders:<path>,<path>`. Writes are audited as `library` rows (`preference-say`,
`preference-edit`, `preference-remove`) and undone with `POST /api/undo`. A stated rule cannot be
changed through `/api/cartridges/values`; that route refuses `pref.stated.*` keys.

### Parameter maps and job filters (v0.1.0)

| Path | Body | Effect |
|---|---|---|
| `/api/parameter-maps/save` | `{"role", "file", "content", "section"?, "layer"?}` | Replaces an editable map, or one value in it, after checking it keeps the map's shape; the file as it was is kept under `<role>/backups/` |
| `/api/parameter-maps/import` | `{"role", "path", "name"?, "replace"?}` | Copies a map file from this machine into the role's folder after its checks; an existing map is replaced only with `replace: true` |
| `/api/parameter-maps/lineage` | `{"role", "file", "made_on"?, "made_by"?, "purpose"?, "note"?}` | States where a map came from or what it is for (`recorded: user`) |
| `/api/parameter-maps/archive` | `{"role", "files": [...]}` | Moves maps into `<role>/archive/`, all or none; a map that serves, or that the serving base is built from, is refused |
| `/api/parameter-maps/restore` | `{"role", "files": [...]}` | Brings archived maps back, all or none |
| `/api/parameter-maps/remove` | `{"role", "file"}`, or `{"role", "files": [...], "archived"?}` | Deletes maps (from the folder, or from the archive with `archived: true`) and their lineage; job filters are removed through the surgery route; two-step delete, cannot be undone |
| `/api/surgery/job-filter` | `{"role", "job", "action": "enable" \| "disable" \| "delete"}` | Plugs a job filter in or out, or removes it (kept under `<role>/removed/` so the undo can put it back) |
| `/api/surgery/job-filter/build` | `{"role", "job", "layers"?: "all" \| "12-27"}` | Builds a job filter through the held-out split gate in the background; refused while a build runs or the drive is under the disk floor |
| `/api/lab/build` | `{"kind": "filter" \| "map" \| "base" \| "distil", "role", "job"?, "layers"?, "share"?: 0.05..0.60, "device"?: "gpu" \| "cpu", "target"?: "adapter" \| "filter" \| "preference" \| "readout", "technique"?: "lens" \| "teacher-loop" \| "channel" \| "unit" \| "readout" \| "steer", "set"?, "rounds"?: 1..8, "budget_s"?: 1..120}` | `distil` trains the healthy teacher pairs of `set` (all by default) into a learned cartridge of kind `target` for the role's MicroModel by `technique`, through the held-out gate (`lab run distil build`); built: the Writer and the Reader (`adapter` by `lens`, `teacher-loop` or `steer`, `filter`, from teacher completions; the Reader also `adapter` by `unit`, a learning unit, each forward-only fit within `budget_s` seconds), the Reranker (`adapter`, `filter`, `readout`, from teacher scores) and the Embedder (`adapter` by `lens` or `teacher-loop`, a request lens, from teacher similarity scores); a pair of kind and technique not available yet is refused with its reason. The lab's builds from one route, run inside the program in the background, one at a time, shown in the live view and audited when they end: a job filter (as `/api/surgery/job-filter/build`), a parameter map with the base cartridge version cut from it (`map`), or a base cartridge version by restoration (`base`). A new base cartridge version is kept beside the others when `library.upgrade` is keep (the default); the user chooses what serves. The state is `lab_build` (also `job_filter_build`) in `/api/surgery`, with its `kind` |

### Learning, memory and storage

| Path | Body | Effect |
|---|---|---|
| `/api/learned-units` | `{"unit": "<compartment>/<id>", "on": true}` or `{"compartment": "<key>", "on": false}` (`value: "on" \| "off"` also works) | Plugs one learned unit in or out, or switches a whole compartment; logged as setting `learned:<compartment>/<id>` or `compartment:<key>` (a registered unit, `shared/<id>`, as `cartridge:<id>`); answers `{audit_id, key, learned_units}` (v0.1.0) |
| `/api/units` | `{"id", "action": "load" \| "unload" \| "register" \| "unregister"}` | Loads or unloads a registered learning unit (through the cartridge switch), registers one (only inside a serving process with the engine bound, so over MCP), or takes an unfitted one out of the registry (v0.1.0) |
| `/api/lenses` | `{"id", "action": "enable" \| "disable" \| "remove", "version"?}` | Plugs a project lens in or out, or deletes one version or all (cannot be undone; the idle fit makes it again) (v0.1.0) |
| `/api/learning/cycle` | `{"project", "action"?: "cycle" \| "sync" \| "rollback", "cycle"?, "version"?}` | Runs one learning cycle (each learner promoted only when it beats the current one on held-out outcomes), syncs, or rolls back one cycle (v0.1.0) |
| `/api/training/export` | `{"project", "name"?}` | Writes the project's training store to `exports/` in the Nano folder, to carry to another of your machines (v0.1.0) |
| `/api/training/import` | `{"project", "file"}` | Imports such a file (hash checked; nothing is added twice) (v0.1.0) |
| `/api/training/references` | `{"path", "licence", "source"?, "domain"?}` | Lists reference material for fundamentals cartridges with its licence and hash; nothing is copied (v0.1.0) |
| `/api/training/distil/import` | `{"file", "set"?, "licence"?, "teacher"?}` | Imports curated teacher pairs from a JSON lines file on this machine (`prompt`, `completion`, `teacher`, `licence`, optional `topk` and `template`; a scored pair for the Reranker also `score` from 0 to 1, the prompt a query and the completion a document): the file listed as reference material with its licence and hash, the split fixed by each prompt's hash, pairs already there passed over, the new ones labelled `used`; a row without a licence (here or on the row) is refused. Audit undoes it (the rows come out again) |
| `/api/training/distil/label` | `{"outcome", "ids"? \| "set"?, "why"?}` | Labels teacher pairs with a map outcome (used, superseded, failed, duplicate, control, unused); only `used` and `superseded` pairs are trained on. Audit undoes it |
| `/api/memory/add` | `{"project", "kind": "store" \| "note" \| "temp", "text", "scope"?}` | A note through the same write path as `nm_memory`; text that looks like a secret is refused (v0.1.0) |
| `/api/memory/pin` | `{"project", "id", "pin": true \| false}` | Pins or unpins a memory (v0.1.0) |
| `/api/memory/sweep` | `{"project"}` | One sweep of the temporary notes, as `nm_memory` `sweep` (v0.1.0) |
| `/api/memory/forget` | `{"project", "id"}` | Deletes one memory; when a server holds the project's memory it applies the deletion within seconds (two-step delete) (v0.1.0) |
| `/api/projects/forget` | `{"project"}` | Removes everything the Nano keeps for a project (its whole memory folder); refused while a server serves it (two-step delete) (v0.1.0) |
| `/api/goals` | `{"project", "objective"}`, or `{"project", "id", "status": "done" \| "dropped" \| "open"}` | Adds an objective, or closes or reopens one, as `nm_goal` (v0.1.0) |
| `/api/guidance` | `{"project", "action": "add" \| "edit" \| "remove" \| "enable" \| "disable" \| "done" \| "reset", "kind": "reminder" \| "loop" \| "step", "id"?, ...}`; a reminder takes `text`, `trigger` (`session-start`, `prompt`, `compact`, `stop`, `edit`, `failure`, `tools`, `error-repeats`), `every`, `glob`, `repeat` (`once`, `always`, `max`), `max`, `files`, `error` and `count` (with `error-repeats`: the error code or a piece of its message, 1 to 80 characters, and 2 to 5 repeats, default 2), `enabled`; a loop `prompt`, `until` (`check-passes`, `goal-done`, `steps-done`, `manual`, `all-of`, `any-of`), `parts` (with `all-of` or `any-of`: 2 to 4 distinct conditions; `all_of` or `any_of` with the list also sets `until`), `goal`, `max` (1 to 20, default 3), `stall` (1 to 10, default 2), `budget_ms` (1000 to 86400000), `budget_tools` (1 to 10000; either budget 0 or null clears it), `enabled`; a step `title`, `id` (on add), `depends_on` (an AND-join; `depends` is taken as the same), `depends_any` (an OR-join: at least one of these done as well; an id in both lists is refused, and the cycle check covers both), `goal`, `check` (`build`, `test`, `none`), `paths`, `probe` (with a check: done when that check passes while its dependencies are met, no edit needed) | One change to the agent's guidance; `done` ends a loop or marks a step done, `reset` counts a reminder again, restarts a loop or opens a step again. A cycle in the step graph, an unknown goal or a value out of bounds is refused (400). `remove` is a two-step delete. Every change is an audit row (`kind: "guidance"`, ids only) undone by `/api/undo`; the caller names who set it (MCP is the agent, anything else you). Answers `{project, action, kind, id, version, audit_id, note, guidance}` (v0.1.0) |
| `/api/comprehension/refresh` | `{"project"}` | Removes the project's world-model notes so the idle writer writes L1 to L4 again (two-step delete; cannot be undone) (v0.1.0) |
| `/api/storage/prune` | `{"category"}` (`memory`, `vectors`, `compression`, `cartridges`, `junk`) | Removes a category's rebuildable items; items changed in the last ten minutes are skipped (two-step delete) (v0.1.0) |
| `/api/storage/delete` | `{"path"}` | Deletes one removable item (two-step delete) (v0.1.0) |
| `/api/dense/clear` | `{"project": "<key>"}` | Removes that project's meaning index kept on disk (`dense/vectors.bin`); logged as a `dense-clear` audit row; answers `{audit_id, project, rows, bytes}` |
| `/api/ledger/clear` | `{"project": "<key>"}` | Removes that project's evidence ledger (`evidence.jsonl`); logged as a `privacy` audit row; answers `{audit_id, project, rows}` (version 4) |
| `/api/examples/clear` | `{"project": "<key>"}` | Removes that project's training store (`training/`); logged as a `privacy` audit row; answers `{audit_id, project, rows}` (version 7) |

### Devices

| Path | Body | Effect |
|---|---|---|
| `/api/pair/code` | `{}` | This computer only: a new six-digit pairing code, `{code, expires_ms}`; refused while the air gap is on |
| `/api/devices/revoke` | `{"id": "<device id>"}` | This computer only: revokes a paired device |

A paired device in `control` mode may use every write but these two; its changes are audited with
its name in `by`. In `view` mode every write from a device is refused.

### Pairing (from a tablet)

`POST /api/pair` with `{"code": "123456", "name": "kitchen tablet"}` from the pairing page. On the
right code: `{"device_id", "name"}` and a `nano_device` cookie holding the device's token. Wrong,
expired or used code: 403; locked: 429.
