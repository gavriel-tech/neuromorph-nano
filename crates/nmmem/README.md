# nmmem

The memory system and identity. Depends on `nmjson`, `nmhash`, `nmlex`, `nmtype`.

- `working`: bounded per scope, in RAM, dropped at task end. `episodic`: append-only
  `episodes.jsonl`; recall by question and citations. `structural`: one summary per substrate
  version. `identity`: per component and for the organism, surviving restart.
- `gate`: writer policy from `designs/memory.json`, refusals logged to `refused.jsonl`.
  `consolidation`: a checked outcome plus the task's notes become one episode.
- Lifetimes are counted in run-days (days on which the system ran, `runday`): `temp`, temporary
  notes (used on 3 run-days: saved as an episode; unused 7 run-days: cleared; a use counts once
  per run-day); `hierarchy`, use counts of saved items by level in `hierarchy.json` only (core
  unused 28 run-days drops to active, active unused 14 to archive).
- `evidence`: memory's evidence for a request (verified outcomes of like requests and the places
  used with them, read fresh; refuted ones shown, never served). `session_open`: the `1-core`
  memories handed to the agent at session open, each place read fresh. `keep`: retention of the
  audit log's tool rows and the training store's rows by run-days; nothing for size, and the
  journal is never pruned.
- `journal`, `book`, `mem2`: the memory's record (the journal) and its views, the only writer of
  `1-core/`, `2-active/<topic>/`, `3-archive/<topic>/`, `index.json`, `temp/captured.json`,
  `undo.json` and `snapshot.json`; each memory sits in one tier. `onewriter`: brings a folder of
  an older layout (use counts and journal views in the same tier files) to one writer per file,
  keeping a copy of the old files in `migrated/`. `tiers`, `classes`, `recall2` and the rest.
- `outcomes`: verified and refuted outcomes from the evidence ledger (`evidence.jsonl`, only read
  here) become journal records under the topic `outcomes` (each outcome once, the newest verdict
  deciding active or invalid, refuted ones kept); `known` reads them back from the views and the
  ledger, `recall` matches a question, `fresh` re-reads a place against its hash.
- State lives in each project's memory folder under `memory/` (see `nmtype::project`). A test
  memory goes in `tests/<category>/<name>` (`serve --memory-dir`). `serve --memory off`: no recall
  and nothing written to memory.

Section checks: `neuromorph-nano check nmmem`.
