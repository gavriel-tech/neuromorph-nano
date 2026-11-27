# tests: test data and test records

The engine never studies this folder.

- `engine-reference/`: recordings the engine self-test compares against (tracked; hashes in
  `INTEGRITY.md`).
- `fixtures/small-repo/`: a small two-crate Rust workspace served by the MCP smoke test
  (`tools/mcp-smoke.ps1`) (tracked).
- `documents/`: the any-data test material (manual, textbook chapters, notes, tables and logs, a
  mixed repository), the decoder fixtures, the document development set and the locked held-out
  set; written in-house for the test; every file hashed in `INTEGRITY.md` (tracked). Checked by
  `neuromorph-nano check documents`; the development set runs with `bench --root
  tests/documents/corpus --questions tests/documents/questions/dev.json`.
- `languages/`: the language fixtures (lexer snippets, one small repository per language family
  with hand-written definitions, imports and packages, ten questions each), written in-house;
  every file hashed in `INTEGRITY.md` (tracked). Checked by `neuromorph-nano check languages`;
  see `languages/README.md`.
- `dev/code-lock.txt`: the code lock of `scratch/frozen-p2` and `fixtures/small-repo`, recorded
  before the any-data work; `neuromorph-nano check lock` compares it (tracked).
- `section-checks/<crate>-<part>/`: written by `neuromorph-nano check`, cleared at the start of
  each check's run (not tracked).
- Test memories: a test session uses `serve --memory-dir <category>/<name>`, which keeps its memory
  in `tests/<category>/<name>` and never in `memory/`. Categories: `memory-system`,
  `claude-code-sessions`, `capability-gates/<gate>`, `mcp-launcher`, `stage-gates/s<n>` (not
  tracked).
- `capability-gates/cx1-context/`: the CX1 question set used by the lab harnesses, when present.
- `real-use/`: the standing real-use set (measured, never inspected by builders); `real-use-2/`:
  the sealed real-use set, run only by the main session at milestones (tracked).
- `real-use-harness/`: prompts, pack check and scorer for the with-and-without runs (tracked).
- `replay/`: the replay bench and loss origins (`record.py`, `replay.py`, `origins.py`,
  `losses.py`): selection rules replayed on recorded candidates in seconds, and where each loss
  starts; recordings under `scratch/replay/rec/` (see `replay/README.md`).
