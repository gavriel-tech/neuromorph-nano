# MCP tools

`neuromorph-nano serve --root <repo>` offers 25 tools over MCP (standard input and output). They
are defined in `crates/nmapi/src/tools.rs` and `crates/nmapi/src/parity.rs`. By default the first
19 (in the order of `crates/nmapi/src/mcp.rs`, coding tools first) are listed in `tools/list`; the
other 6 are the interface tools (`nm_storage`, `nm_cartridges`, `nm_learning`, `nm_comprehension`,
`nm_settings`, `nm_live`), callable by name at any time and brought into `tools/list` on demand.
Which tools are listed is set by the controls `mcp.tools`, `mcp.tool_count`, `mcp.tool_control`,
`mcp.enabled_tools` and `mcp.dynamic_tools` (`docs/configuration.md`).

Errors come back with a code: `nm-400` bad input, `nm-404` not found, `nm-409` refused in this
state (for example memory off), `nm-500` engine fault, `nm-503` no repository served (or, for the
interface tools, the interface's routes not available in this process).

Every call is written to the audit log (`memory/audit.jsonl`) with the name the connecting client
gave itself in `initialize` (`client`, for example `claude-code`), so the Live Activity page and the
Audit page can show each coding agent's calls by name.

To try a tool without an agent, `call` runs one tool with memory off:

```
target\release\neuromorph-nano.exe call <tool> "<json arguments>" --root <repo>
```

## What the agent sees

The agent reads the tool list on every turn, so each tool's description is one line: what it
returns and when to call it. The tables below give the fuller detail. The server's instructions
(sent with `initialize`) say: the server has already read the project's folder; for every question
or task about the project, call `nm_context_pack` with the question as the first step, before
Grep, Glob or Read (one call searches the whole folder), then read only the ranges it cites, and
search further only if they are not enough; `nm_symbol_find` for an exact name; `nm_deps` or
`nm_repo_map` for package relations; `nm_build_run` (check, then test) before finishing a Rust
change. The instructions also name the tools callable on demand.

## Context and comprehension

| Tool | What it does | Main arguments |
|---|---|---|
| `nm_context_pack` | A cited context pack for a question about any material in the studied folder (code, documents, notes, tables, logs) within a token budget; withholds when nothing is observed. Document spans name their section path, table slices their header line | `question` (required), `tokens` (default from the `budget.pack_tokens` control, 4000), `form` (adaptive, answer, cited, full), `format` (text or json), `since`, `debug`, `mode` (fast or full, on the whole path) |
| `nm_repo_map` | Packages of the repository (Cargo crates, npm packages, Python projects) with their dependencies on each other, ecosystem and manifest | `crate` |
| `nm_symbol_find` | Where a code symbol is defined: kind, file, first and last line, enclosing class, defining text | `name` (required; `Class.method` for a member) |
| `nm_deps` | Direct dependencies of a crate or package (out) or the packages that depend on it (in); for a file, the files it imports or that import it, and the outside packages it names | `crate` or `package`, or `file`; `direction` (out or in) |
| `nm_view` | One of eight views (global, local, structural, semantic, temporal, operational, epistemic, objective), or Context Zoom | `view`, `query`, `zoom` (symbol, file, crate, workspace) |
| `nm_sync` | Re-reads the repository incrementally; changed files invalidate older packets | none |

## Memory and learning

| Tool | What it does | Main arguments |
|---|---|---|
| `nm_memory` | Recall saved memory, episodes and temporary notes; add a note or temporary note; sweep temporary notes; offer a claim to the write gate; read the hierarchy, identity or tier use; memory revision 2 views and controls | `op` (recall, note, temp, temp_list, sweep, hierarchy, store, identity, tiers, health, record, recall2, pin, unpin, undo), `text`, `scope`, `source` |
| `nm_feedback` | Marks an answer right or wrong; the verdict joins its episode and is logged as an outcome | `episode` (required), `correct` (required), `taste` (accept or reject), `note` (a correction or observation for continuous learning) |
| `nm_goal` | Hands over an objective, closes or reopens one, or reads the objectives held; with `op`, reads and sets the reminders, loops and step graph that guide the agent (through `/api/guidance`, on the project this server serves unless `project` names another; what the agent sets is kept as set by the agent) | `objective`, or `id` with `status` (done, dropped, open); `op` (goal, guidance, reminder_add, reminder_list, reminder_edit, loop_add, loop_edit, loop_end, loop_outcomes, step_add, step_done, step_edit, remove, enable, disable, reset), `item`, `kind`, `text`, `trigger`, `every`, `glob`, `repeat`, `max`, `files`, `error`, `count`, `prompt`, `until`, `parts`, `all_of`, `any_of`, `stall`, `budget_ms`, `budget_tools`, `limit`, `goal`, `title`, `depends_on`, `depends_any`, `probe`, `check`, `paths`, `enabled`, `project`, `confirm_token` |
| `nm_unit` | Learning-weight units: list, register, unregister, load or unload (models must be bound) | `op`, `id`, `target`, `slot` (default reranker), `rank` (default 8), `seed` (default 8) |

## Build help

All cargo commands run with `--offline` and never write into the studied tree.

| Tool | What it does | Main arguments |
|---|---|---|
| `nm_build_run` | Runs one cargo command for one package (the whole workspace when none is given); errors come back as compact cited lines, warnings counted | `package`, `kind` (check, clippy, test, build), `warnings`, `budget` (default 12), `form` (lines or json) |
| `nm_build_explain` | Explains the first error of a build: file, line, the lines around it, cause and fix | `run` (from `nm_build_run`), or `package` and `kind` for a fresh run |
| `nm_improve_scan` | Finds concrete defects: rule words, files over the line limit, clippy machine-applicable suggestions. Applies nothing | `filter`, `package`, `budget` (default 20) |
| `nm_patch_propose` | The diff for one finding of the last scan. Applies nothing | `index` (required) |
| `nm_verify` | Checks a diff before anything runs: interface and manifest changes, tests that name the changed items, rule words, `unsafe` and `unwrap` in added lines; lists runtime, performance, security and regression as open questions | `diff` (required), `episode` |
| `nm_stepcheck` | Step Check: applies a diff to a new scratch copy and runs check, clippy with warnings denied, tests when asked, size and rules. Never touches the source | `diff` (required), `package`, `tests` |

## The Nano itself

These read and control the Nano, not the studied material. The last five go through the same
routes as the web interface (`docs/interface-api.md`), so a change made here is logged in the audit
and can be undone the same way. An action that changes data needs `confirm: true` where noted;
every delete takes two calls: the first returns a preview and a `confirm_token`, the same call again
with that token carries it out.

| Tool | What it does | Main arguments |
|---|---|---|
| `nm_status` | Status: snapshot and substrate version and digest, file, symbol and crate counts, files by reader, files read generally, files skipped by reason, models loaded, the run ceiling, where the MicroModels run (`backend`), each memory area's setting, budget, use and holders with where the drones' intake stands (`memory_areas`) | none |
| `nm_anatomy` | The running anatomy: nodes, edges, waves; zoom by level | `node`, `level` (organism, slot, structure, component, cluster), `id` |
| `nm_weights` | The model parameters, read only: tensors, shapes, precision, clusters, residency; with `model`, the weight view of any installed model (the interface's `GET /api/models/weights`) | `limit` (default 40), `model` (an installed model id; default the served base model) |
| `nm_live` | What the Nano is doing now: active stages, MicroModels executing, card memory, the machine's capacity, each memory area's use against its budget (`memory_areas`), the idle writer and the recent timeline | none |
| `nm_storage` | Disk use by category with exact paths; prune rebuildable caches; delete one removable item; the disk floor; clear a project's meaning index; forget a project or one memory; trim the audit log | `op` (view, settings, prune, delete, set, dense_clear, forget_project, memories, forget_memory, audit_retention), `category`, `path`, `id`, `project`, `key`, `value`, `confirm_token` |
| `nm_cartridges` | Cartridges, stacks, base cartridge versions, job filters and parameter maps: list them, plug a cartridge or a use's stack in or out, choose the base cartridge a MicroModel serves (a version, `auto` or `open` for its open weights), see which MicroModels each cartridge fits, read, edit, import, archive, restore and remove parameter maps and record their lineage, build, plug or remove a job filter, map a MicroModel's parameters and cut a base cartridge version from the map (`map_build`, with `share` and `device`), build a base cartridge version by restoration (`base_build`; new versions are kept beside the others by default), train teacher pairs into a learned cartridge through the held-out gate (`distil_build`, with `role`, `target`, `technique`, `set`, `rounds`) and list the distilled cartridges with their gate figures (`distils`), inspect a cartridge (a distilled one with its `parameters`) and set its values (a distilled one: `scale`, `rank`, written as a new version), read any object's parameters (`parameters`: the list, one object's tensors with shapes and norms, a page of values) and edit them into a new version (`parameters_edit`: `edit` names the operation, with `path` or `selection`, `value`, `lo`, `hi`, `values`, `source`, `t`, `target`, `width`, `rank`, `file`, `ops`, `name`, `description`, `preview: true` for the difference only; or the short forms `scale`, `layers`, `values`; the source is never written; `parameters` with `path` reads a region and the map values at the same place, with `maps: true` the maps beside the object; `parameters_export` writes a selection to a file), hold edits, masks and freezes of any object's parameters in its overlay, the open weights included (`overlay` reads it; `overlay_edit` with `edit` and `path` or `selection`; `mask` and `freeze` with `path`, or the whole object with `whole: true` or no path; `unmask` and `unfreeze` with `id`, `ids` or `path`; `overlay_undo`; `overlay_clear` in two calls; each with `preview: true`; the object's files are never written, a change touching a frozen region is refused, and every reply says when the change takes effect), check an existing job filter against its held-out split without writing anything (`filter_check`), read the whole-set overview of an opened object (`parameters_overview`), put a stack on trial or save one, import a stack file, and manage project lenses. `filters` returns each job filter with the base it was built on (`built_on`), the base serving now (`serving_base`) and whether they match (`base_match`, reported only: the filter loads either way) | `op` (list, library, filters, on, off, use, bases, compat, maps, map_read, map_save, map_import, map_lineage, map_archive, map_restore, map_remove, map_build, base_serve, base_build, base_remove, distil_build, distils, filter_on, filter_off, filter_build, filter_check, filter_remove, inspect, set_values, parameters, parameters_edit, parameters_export, parameters_overview, overlay, overlay_edit, mask, unmask, freeze, unfreeze, overlay_undo, overlay_clear, trial, stack_save, cartridge_import, lenses, lens_switch, lens_remove), `role` (reader, writer, embedder, reranker), `target` (adapter, filter, preference, readout), `technique` (lens, teacher-loop, channel, unit, readout, steer), `set`, `rounds`, `budget_s`, `job`, `file`, `files`, `id`, `key`, `model`, `choice`, `use`, `stack`, `tensor`, `path`, `maps`, `offset`, `limit`, `edit`, `selection`, `value`, `lo`, `hi`, `source`, `source_b`, `t`, `target`, `width`, `rank`, `ops`, `name`, `description`, `preview`, `scale`, `layers`, `values`, `whole`, `ids`, `confirm`, `confirm_token` |
| `nm_learning` | A project's learning state, training examples, evidence ledger and outcomes; run one learning cycle or bring stored examples into the engine; roll a cycle back; export or import the training store; list or add reference material; clear the examples or the ledger; list, import and label the teacher data for distillation (`distil`, `distil_import`, `distil_label`; only `used` and `superseded` pairs are trained on); list learned units by compartment and plug one unit, or a whole compartment, in or out; turn the user's stated preferences into a preference cartridge of rules (a preview, then confirm), list, edit, plug, unplug or remove those cartridges, and check a draft against the rules in force | `op` (view, cycle, sync, rollback, examples, ledger, outcomes, export, import, references, reference_add, distil, distil_import, distil_label, clear_examples, clear_ledger, units, unit_on, unit_off, compartment_on, compartment_off, preference_list, preference_say, preference_edit, preference_remove, preference_plug, preference_unplug, preference_apply), `project`, `unit` (`<compartment>/<id>`), `compartment` (nano, shared or a project key), `text`, `name`, `scope`, `cartridge`, `plug`, `trial`, `id`, `add`, `rule`, `strength`, `drop`, `path`,`file`, `set`, `licence`, `teacher`, `ids`, `outcome`, `why`, `limit`, `confirm`, `confirm_token`; a live teacher (`teach_list`; `teach_start` with `set`, `prompts`, `instruction`, `licence`; `teach_next` with `set`, `count`; `teach_answer` with `set`, `answers`, `teacher`; `teach_local` with `set`, `teacher`, an installed model) |
| `nm_comprehension` | A project's 4-level world model (L1 piece, L2 file, L3 module, L4 system notes), routing record, lens heads and memory tiers; `refresh` removes the world-model notes so the idle writer writes them again | `op` (view, tiers, memory, refresh), `project`, `confirm_token` |
| `nm_settings` | The controls (read, set, undo), status, projects, design files, the audit log, activity, stage times, the session record, the parts, the air-gap report, a repository's connection status, paired devices, the model each role runs on, downloads, the cartridge libraries (official and third party: list, add, remove, browse, fetch, fetch all, publish), hardware limits and this machine's recommended values | `op` (view, saved, set, undo, status, projects, designs, audit, activity, stage_times, session, parts, airgap, connect, devices, device_revoke, roles, role_set, downloads, download, sources, source_add, source_remove, source_browse, source_fetch, source_fetch_all, source_publish, limits, recommend, set_limit, apply_recommended), `key` (`set`: a control key, or one entry of a `settings.json` map as `saved` lists it: `base_cartridges:<model id>`, `base_compensation:<model id>`, `role_models:<role>`; `set_limit`: `vram_mb`, `ram_mb`, `cpu_threads`, or a memory area `content_mb`, `caches_mb`, `weights_mb`), `value`, `id`, `role`, `model`, `what`, `only`, `repo`, `kind`, `revision`, `token`, `path`, `item_kind`, `item`, `create_pr`, `summary`, `confirm`, `confirm_token`; your own repositories for backup and rollback (`backups`, `backup_add` with `url`, `parts`, `token`; `backup_remove`, `backup_now`, `backup_snapshots` with `id`; `backup_preview`, `backup_restore` with `id`, `commit`, the restore with `confirm: true`) |

### Reminders, loops and steps: one feature, four surfaces

| What | Interface (Memory, Loops and graph) | API | MCP (`nm_goal`) | Command line |
|---|---|---|---|---|
| Read all, with recent deliveries and their verdicts | the Guidance section | `GET /api/guidance` | `guidance` (`reminder_list` for the reminders only) | `guidance --project <repo>` |
| Add a reminder | add reminder | `POST` `action: add`, `kind: reminder` | `reminder_add` | `guidance ... add reminder --text ... --trigger ...` |
| Edit | edit | `action: edit` | `reminder_edit`, `step_edit` | `guidance ... edit <kind> <id> ...` |
| Switch on or off | the switch | `action: enable`, `disable` | `enable`, `disable` (`kind`, `item`) | `guidance ... enable\|disable <kind> <id>` |
| Add a loop | add loop | `action: add`, `kind: loop` | `loop_add` | `guidance ... add loop --prompt ... --until ...` |
| End a loop | end | `action: done`, `kind: loop` | `loop_end` | `guidance ... done loop <id>` |
| Combined condition, stall, budgets | until all of or any of, parts, stall, budget s, budget tools | `until: all-of\|any-of`, `parts`, `stall`, `budget_ms`, `budget_tools` | `loop_add`, `loop_edit` (same fields; `all_of`, `any_of`) | `add loop --until all-of --parts a,b --stall N --budget-ms MS --budget-tools N` |
| Ended loops (outcome rows) | Ended loops | `GET /api/guidance` `outcomes` (`&outcomes=N`) | `loop_outcomes` (`limit`) | `guidance --project <repo>` (`outcomes`) |
| OR-join and probe steps | after one of, probe (no edit needed); OR edges dashed, probes with round ends | `depends_any`, `probe` | `step_add`, `step_edit` (`depends_any`, `probe`) | `add step --after-any a,b --check test --probe true` |
| Recurring error reminder | when the same error comes back, error code, repeats | `trigger: error-repeats`, `error`, `count` | `reminder_add` (`error`, `count`) | `add reminder --trigger error-repeats --error E0308 --count 2` |
| Add a step | add step | `action: add`, `kind: step` | `step_add` | `guidance ... add step --title ... [--after ...]` |
| Mark a step done | mark done | `action: done`, `kind: step` | `step_done` | `guidance ... done step <id>` |
| Reset | reset, restart, open again | `action: reset` | `reset` | `guidance ... reset <kind> <id>` |
| Remove (two steps) | remove, typed confirmation | `action: remove` | `remove` (`confirm_token`) | `guidance ... remove <kind> <id> [--confirm <id>]` |
| Undo | Audit page | `POST /api/undo` | `nm_settings` `undo` | `guidance undo <audit id>` |

### Stated preferences: one feature, four surfaces

| What | Interface | API | MCP (`nm_learning`) | Command line |
|---|---|---|---|---|
| List cartridges and rules | Learning, Preferences | `GET /api/preferences` | `preference_list` | `preferences list` |
| Words to rules (preview, then write) | chat box, preview rules, write | `POST /api/preferences/say` | `preference_say` (`confirm`) | `preferences say "<words>" [--yes]` |
| Edit (a new version) | add, drop | `POST /api/preferences/edit` | `preference_edit` | `preferences edit <id> ... --yes` |
| Set a want's strength (a new version) | strength, set | `POST /api/preferences/edit` (`rule`, `strength`) | `preference_edit` (`rule`, `strength`) | `preferences edit <id> --rule rN --strength 1.5 --yes` |
| Plug in or out | plug in, unplug (also the Cartridges rack) | `POST /api/preferences/plug` | `preference_plug`, `preference_unplug` | `preferences plug <id>`, `preferences unplug <id>` |
| Remove (two steps) | remove cartridge, typed confirmation | `POST /api/preferences/remove` | `preference_remove` (`confirm_token`) | `preferences remove <id> [--confirm <id>]` |
| Check a draft (violations, and wants a draft misses) | check | `POST /api/preferences/apply` | `preference_apply` | `preferences apply` |
| Undo | Audit page | `POST /api/undo` | `nm_settings` `undo` | `preferences undo <audit id>` |

### Whole-set parameter overview: one feature, three surfaces

Read only, and only for an object opened in the parameter editor; nothing is read beforehand and
nothing polls.

| What | Interface (Parameters, the opened object) | API | MCP (`nm_cartridges`) |
|---|---|---|---|
| The set's tensors in layout order (shape, type, layer, block; no values) | the overview | `GET /api/parameters/overview?key=<key>` | `parameters_overview` (`key`) |
| One tensor as a grid of cell min, max, mean and RMS, streamed from the file | a heatmap, read when it scrolls into view | `...&tensor=<name>&w=<cols>&h=<rows>` | `parameters_overview` (`key`, `tensor`, `w`, `h`) |
| A region, down to one weight per cell (exact values) | drag to zoom; the zoom trail and reset | `...&r0=&r1=&c0=&c1=` | `parameters_overview` (`r0`, `r1`, `c0`, `c1`) |
| Every value of a region, paged | values | the same route at one weight per cell | the same op at one weight per cell |

### Parameter overlay: one feature, four surfaces

Edits, masks and freezes of any object's parameters (the open weights, base cartridges, socket
cartridges, job filters, adapters, learning units, preference counts), held in
`memory/overlays/` beside the object and applied when the engine seats it; the object's files are
never written. Every reply says when the change takes effect (`takes_effect.when`).

| What | Interface (Parameters, the opened object) | API | MCP (`nm_cartridges`) | Command line |
|---|---|---|---|---|
| Read the overlay | the overlay list; marks on the grid, layer strip and tensor chips | `GET /api/parameters/overlay?key=<key>` | `overlay` | `library overlay <key>` |
| Edit a selection | apply in: the overlay, then preview and apply | `POST {key, action: edit, op, path}` | `overlay_edit` (`edit`, `path`) | `library overlay <key> edit --op <op> --path <p>` |
| Mask, freeze (a selection, a tensor, a layer, the whole object) | mask, freeze | `action: mask`, `freeze` | `mask`, `freeze` (`path`, `whole`) | `... mask`, `... freeze` |
| Unmask, unfreeze | unmask, unfreeze (each entry, or the selection) | `action: unmask`, `unfreeze` | `unmask`, `unfreeze` (`id`, `ids`, `path`) | `... unmask --id <e>` |
| Undo | undo last change; Audit page | `action: undo`; `POST /api/undo` | `overlay_undo`; `nm_settings` `undo` | `... undo` |
| Clear (two steps) | clear, confirm clear | `action: clear` with `confirm_token` | `overlay_clear` (`confirm_token`) | `... clear [--confirm-token <t>]` |

## Examples

### nm_context_pack

```json
{ "question": "Where is the struct Coordinator defined?", "tokens": 2000 }
```

The default text reply starts with a summary line and an answer, then each cited span as an
`@@ path:first-last` header followed by its lines. With memory on, the reply's episode can later
be marked with `nm_feedback`. The episode keeps the question and each citation as its place, line
range and a hash of the cited lines, never the code; a recalled episode whose cited lines changed
carries an `obsolete` note naming the stale citation.

On the whole path (`serve --whole`), `mode` picks how much runs: `full` (the default, from the
`mode` control) runs every part; `fast` judges the stack's first pieces and the dense lane's with
the Reranker and nothing more, and its reply says so on the line after the summary. An identical
call on an unchanged code version is answered from the stored reply.

### nm_memory

```json
{ "op": "store", "text": "the checkout total is in cents" }
```

`store` offers a claim to the memory gate and answers `admitted` with a reason when refused. Text
that looks like a secret is refused by `store` and `temp` alike, and nothing of it is written: the
refusal log keeps only the time, the reason and a hash of the key. `health`, `record` and
`recall2` read memory revision 2 (`record` and `recall2` take an id or a query in `text`); `pin`
and `unpin` take a record id in `text`; `undo` takes a journal sequence number in `text`.

### nm_symbol_find

```json
{ "name": "Coordinator" }
```

Returns each definition with its kind, file, first and last line, enclosing class or namespace
(empty at top level) and defining text. `{ "name": "TaskStore.add" }` finds the `add` method of
`TaskStore`. A Rust item spans its first line only.

### nm_deps

```json
{ "file": "src/app/page.tsx", "direction": "out" }
```

Returns the files of the tree the file imports, the outside packages it names, and each import's
line. With `"direction": "in"`, the files that import it. With `crate` or `package`, the package's
dependencies inside the tree.

### nm_build_run

```json
{ "package": "my-crate", "kind": "clippy", "budget": 8 }
```

Returns one line per error with its source line and help, and a count of warnings. Pass the run id
to `nm_build_explain` for the first error explained.

### nm_cartridges

```json
{ "op": "filters" }
```

Returns every job filter by role and job, plugged in or not, with `built_on`, `serving_base` and
`base_match`. `{ "op": "base_serve", "model": "qwen3-1.7b", "choice": "open" }` makes the Writer
serve its upstream open weights from the next start (the `model` field takes the open weights id
the MicroModel is built from).

```json
{ "op": "distil_build", "role": "writer", "target": "adapter", "technique": "lens", "set": "docs-notes" }
```

Starts a distillation build in the one lab build slot (`/api/lab/build` kind `distil`); its state
and result figures are in `filters` (`lab_build`) while it runs and after. A pair of kind and
technique that is not available yet is refused with its reason. `{ "op": "distils" }` lists the
distilled cartridges with their gate figures; `nm_learning` `{ "op": "distil_import", "file":
"<pairs.jsonl>", "licence": "Apache-2.0" }` imports teacher pairs first.

Built: the Writer and the Reader from teacher completions (`adapter` by `lens`, `teacher-loop` or
`steer`, `filter` by `channel` or `teacher-loop`; the Reader also `adapter` by `unit`, a learning
unit fitted forward-only within `budget_s` seconds per fit, 20 by default), the Reranker from teacher scores (the same
adapter and filter techniques, and `readout` by `readout` or `teacher-loop`, a judge read-out), and
the Embedder from teacher similarity scores (`adapter` by `lens` or `teacher-loop`: a request lens
on its request vectors only). A scored pair: `prompt` the query, `completion` the document,
`score` from 0 to 1; a live teacher sends it with `teach_answer` as `{"prompt_id", "completion",
"score"}`. What is not built (an embedder filter, a steer for the Embedder or the Reranker, a unit
for any role but the Reader, a read-out for any role but the Reranker) answers with its reason
(`nm_learning` `distil` lists every such pair under `unavailable`).

```json
{ "op": "distil_build", "role": "reranker", "target": "readout", "technique": "readout", "set": "judged-docs" }
```
