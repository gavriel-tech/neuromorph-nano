# How the Nano's weights are built, step by step

This guide follows the chain the interface shows on Cartridges, side panel tab "Components" (`/cartridges/#flow`): from the upstream open weights to the base cartridges the MicroModels serve, then the job filters, adapters, stacks and what the Nano learns. Every command below is the one actually used for v0.1.0 (traced from the cartridge labels, the run reports in `runs/` and the lab's code on 2026-11-26). Lab commands run from the repository root with `cargo run --release -p nmlab --bin nano-lab --` (or the built `nano-lab.exe`).

You do not need any of these steps to use the Nano: the base cartridges, job filters and hardware cartridge it serves come ready from the official cartridge library on Hugging Face, `gavriel-tech/neuromorph-nano-cartridges` (Models, Downloads). The steps are for building new cartridges and maps.

## 0. Words

| Word | What it is |
|---|---|
| MicroModel | one of the four models the Nano runs, named by its role: Embedder, Reranker, Reader, Writer |
| Upstream open weights | the Qwen3 open weights (Apache-2.0) each MicroModel was built from and everything is measured against: `qwen3-embedding-0.6b` (Embedder), `qwen3-reranker-0.6b` (Reranker), `qwen3-0.6b` (Reader), `qwen3-1.7b` (Writer); installed with the Nano, because each base cartridge is checked against them when it is seated |
| Parameter map | a measurement of a MicroModel's weights: what to cut, what to restore, what a job needs |
| Surgery | the lab step that makes a new base cartridge from the one before it and the open weights |
| Base cartridge | a MicroModel's serving weights (`embedder-base`, `reranker-base`, `reader-base`, `writer-base`), one version serving at a time |
| Job filter | a gain and bias per channel at a MicroModel's sockets for one job, read while serving |
| Lens, judge read-out | project adapters fitted at idle from one project's own material |
| Learned unit | anything learned that is plugged in or out, alone or by compartment (the project it was fitted in): lenses, read-outs, learned pieces, learning units |
| Stack | the cartridges plugged in together for one use: base, hardware, adapter, learned, then preference |

## 1. The upstream open weights

Interface: Models, side panel, Downloads, "Upstream open weights". Each file is checked against the SHA-256 its weight card pins. The installer fetches them, and the Nano needs them to run: each MicroModel reads its tokenizer and configuration from them, and a base cartridge is seated only after a check run against them (`nmengn::base::seat`), after which they are released from memory. They are also the source for building new cartridges and maps, for open mode (serving the open weights instead of a base cartridge), and for parity checks.

## 2. The first cut: `pmap build-base`

```
neuromorph-nano lab run pmap build-base <role> <share> gpu [v=<n>]
```

It records how each weight is used on the role's jobs (drone, integration, judge; the writer its own jobs), cuts the given share, and writes both a base cartridge version and the map of the cut, `library/parameter-maps/<role>/pmap-owl-<share*100>.json`. Used for v0.1.0:

| Role | Command | Map | Cartridge it made |
|---|---|---|---|
| Reader | `pmap build-base reader 0.40 gpu` | `pmap-owl-40.json` | reader-base@2 |
| Writer | `pmap build-base writer 0.45 gpu v=3` | `pmap-owl-45.json` | writer-base@3 |
| Embedder | `pmap build-base embedder 0.35 gpu` | `pmap-owl-35.json` | embedder-base@3 |
| Reranker | `pmap build-base reranker 0.35 gpu` | `pmap-owl-35.json` | reranker-base@4 |

(For the embedder and the reranker the share is read from the map's file name.)

## 3. Surgery: the serving base cartridges

```
neuromorph-nano lab run pmap surgery cascade <role> --anchor-budget 8 --budget 5 --write
```

It starts from the newest version of the role's base cartridge in the cartridge library, reads what was cut from that cartridge's own zero weights, measures against the open weights, and restores the weights that matter most (anchor layers first). It reads one map, `super-weights.json`, which only blocks removals, so in a cascade it changes nothing that was added back. On `--write` it writes, in order: `super-weights.json`, `surgery-probe.json`, the new cartridge, one line in `surgery-ledger.jsonl`.

| Serving now | Made from |
|---|---|
| embedder-base@4 | embedder-base@3 + open weights (anchors L00, L02, L27) |
| reader-base@3 | reader-base@2 + open weights (anchors L00, L02, L27) |
| reranker-base@5 | reranker-base@4 + open weights (anchors L00, L02, L27) |
| writer-base@4 | writer-base@3 + open weights (anchors L00, L01, L02, L27) |

Each cartridge's label says this in `made_from`. The cut made in step 2 carries into these versions: surgery since then only added weights back.

Interface: Cartridges, side panel tab Base versions, chooses which version serves, removes old ones, and sets whether an upgrade keeps or replaces older versions (the `library.upgrade` control, `keep` by default); Models, Downloads fetches the serving versions; the surgery itself is a lab step.

## 4. Job filters

```
neuromorph-nano lab run pmap surgery job-filter <role> --job <job> --layers <all|12-27> --split-gate --write
```

Built over the serving base, kept only if the held-out split gate passes, written as `library/parameter-maps/<role>/job-filter-<job>.json` with the base it was built on. Interface: Parameters, side panel tab Add, runs exactly this; each filter is plugged in or out on its lane, rebuilt on other layers, inspected socket by socket, or removed (two steps; Audit undoes). A filter is plugged in unless switched off, and the serving engine reads the plugged-in filters of each MicroModel while it serves, picking up a change without a restart.

Each job filter row of `/api/surgery` (and of MCP `nm_cartridges op=filters`) reports `built_on` (the base the filter was built on, from the map lineage), `serving_base` (the base its MicroModel serves now) and `base_match`. This is only reported: a filter built on another base still loads, with one warning line.

v0.1.0 filters: Embedder `code-slice`, Reranker `code-slice` and `judge-code`, Reader `read`, Writer `code-slice` and `write`, all built on the base serving now.

## 5. Adapters and lenses

Fitted at idle from a project's own material (commit messages to the functions they changed, doc comments hidden from their definitions), kept only if they beat a shuffled-label control on held-out files: the Embedder's lens and the Reranker's code read-out (the judge read-out), with the adapters in the cartridge library, `library/cartridges/adapter/<id>/`. Each loads only for its project and only on the base it was fitted on. Interface: Cartridges, side panel tab Learned (plug in or out, alone or by compartment, remove a version); the Learning page shows them in brief.

## 6. Stacks

The cartridges plugged in together for each use. Interface: Cartridges, the stacks and layers views (put a stack on trial or let it serve, make your own, switch a use). New stacks start on trial: what they would change is recorded and never applied until their evidence allows.

## 7. Learning

Learning cycles, learned pieces, learning units and preference cartridges, all kept beside the base and never merged into it. Interface: Learning; every learned unit can be unplugged on its own or with its whole compartment (Cartridges, Learned). See `docs/continuous-learning.md`, "What runs in v0.1.0".

## 8. Libraries

The official Neuromorph cartridge library on Hugging Face, `gavriel-tech/neuromorph-nano-cartridges` (built in), or third-party Hugging Face libraries you add by repository name; a file you import from this machine goes straight into the cartridge library or the parameter map library. Interface: Models, Downloads, "Cartridge libraries · official and third party", and Parameters, side panel tab Import. See `docs/cartridges-and-parameters.md`, "Cartridge Libraries".

## Map lineage

Each role folder keeps `lineage.json` beside its maps: what each map was made on (a base cartridge or the open weights), the command, its purpose and what was built from it. The tools record it as they write a map (`pmap build-base`, surgery, job filters, imports); the maps made before 2026-11-26 were filled in from proof. The Parameters page groups the maps by it:

- Serving now: the job filters.
- What the serving base cartridge is built from: the `pmap-owl` cut and `super-weights.json`.
- Probes and logs: `surgery-probe.json` (it describes the base before the serving one: it was not rewritten after the last write) and `surgery-ledger.jsonl`.
- History: maps that built earlier versions (`act-20*` for embedder-base@2 and writer-base@2, `map.json`, `role.json`, `tol-*`, `record.json` for reranker-base@3, and the compensation cartridges' bias maps).
- Experiments: sweeps and comparisons nothing reads (`pmap-v1`, byte-identical to the role's `pmap-owl` map; `act-20-next*`; `zsep-*`; `scores.json`; `union*`).

A map's origin can be stated or corrected on Parameters (or `nm_cartridges op=map_lineage`); Audit undoes it.

## Known gaps in v0.1.0

- Serving checks a job filter's model, not the base version it was built on: a filter built on an older base would still load. The Parameters page marks such a filter stale.
- `surgery-probe.json` was not rewritten after the last surgery write, so each describes the previous base.
- The compensation cartridges (`*-base-comp`) were made for earlier base versions; they are on trial (never applied) and the compatibility view marks them as not fitting.
- `pmap` run reports in `runs/` do not record their command line; the commands above come from the cartridge labels and the lab's code.
