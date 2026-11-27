# designs: data the engine runs on

JSON the code reads as data; every `*.json` must parse strictly (`neuromorph-nano check designs`).

- Organism and policy: `anatomy.json` (initial snapshot), `reflexes.json`, `homeostasis.json`,
  `memory.json` (memory tiers and the writer gate), `compositions.json` (task kinds),
  `placement.json` (processor or graphics card per component, measured by `nano-lab run s6 place`),
  `models.json` (the models folder).
- `drones.json`: the drone reader sets (attention heads per population, optional per tentacle),
  read by `nmengn::kvstore` and validated by `check designs`. `tentacle-core.json`: the tentacle
  core module specification, read by `nmapi::tentacle_core`, validated by `check designs`.
- `chains.json`: MicroModel chains of the Core Brain by task kind (`nmbrain::chain`).
  `muscle.json`: muscle memory and G3 settings (`nmmusc`). `cartridges.json`: the label schema and
  stacking policy of learned adapter cartridges (`nmlearn::cartridge`).
- `conformance.json` (conformance rules, `nmtool::conformance`) and `scoreboard.json` (one row per
  part, `nmtool::scoreboard`).
- Test inputs read by the lab tool: `questions-context.json`, `questions-codebase.json`,
  `breakages.json`, `defects.json`, `writer-errors.json`, `compare.json`, `train-rerank.json`,
  `precision-judge-mix.json`.
- `assemblies/`: execution assemblies. `parmap/`: the Parameter Map atlas.
