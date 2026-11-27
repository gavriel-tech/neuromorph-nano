# Replay bench and loss origins

Tools for testing a line selection idea in seconds on recorded candidates, and for finding where each
failure starts (guide `internal/design/2026-11-16-nano-guide-implementation.md` sections 12 and 13). Python
standard library only; run with `python -I` from the Nano folder. Nothing here changes how the Nano
chooses. Written by the replay builder, 2026-11-17; records in `internal/plans/2026-11-15-nano-build-plan.md`
under "Replay bench and loss origins".

- `record.py`: a recording of what reaches G2's selection, per question, from the bench's diagnosis
  (`bench --mode board --paths stack ... --why-r`, partial: the chosen set, the answer pieces and the
  four strongest per part) or from the bench's `--dump-candidates` (full; the bench change is in
  `scratch/replay/PATCH.md`). Recordings go under `scratch/replay/rec/`.
- `replay.py`: a selection rule (a Python function) replayed on a recording; sent up, in pack, right
  first, top 3 and tokens per question, paired against the recorded served choice with
  `tests/dev-large/compare.py`'s McNemar test. `--repro` shows how closely the served rule reproduces
  the recording.
- `origins.py`: the answer forced in by stage (G2's candidates, G2's set) and the stage input checks.
- `losses.py`: per lost question where the loss shows, where it starts, the answer piece against the
  piece that beat it, and a cause label; tables by starting stage and cause.
- `drone-check.json`: the hand reading of the request's drone on 20 questions of the large set's
  tuning fold.

Example (program c1 recordings):

    python -I tests/replay/replay.py --rec scratch/replay/rec/large-code.json scratch/replay/rec/large-docs.json --rule served defs_first my_rules.py:rule --repro
    python -I tests/replay/losses.py --rec scratch/replay/rec/large-code.json scratch/replay/rec/large-docs.json --target first pack --list

Use the large set's tuning fold only (`questions-*-tune.json`), never the held-out fold.
