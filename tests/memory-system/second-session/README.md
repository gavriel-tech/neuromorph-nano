# Second-session test of memory

Does memory earn its time? Session one works through questions with Claude Code's outcome for
each scripted into the evidence ledger; session two, a new process, asks again with memory on and
with memory off, the same program, back to back. Written for work package WP13; a development test,
never a release measure.

Run (about 30 seconds, processor only, no graphics card):

```
python tests/memory-system/second-session/run.py --exe <neuromorph-nano.exe> --label <name>
```

`--exe` defaults to `scratch/lane-c/target/release/neuromorph-nano.exe`. Everything it writes goes
to `scratch/lane-c/wp13/<label>/` (a scratch Nano folder per arm through `NANO_HOME`, the replies,
the ledger and outcome listings); the report goes to `runs/memory-second-session-<label>-<ms>.json`.
The real `memory/` is never opened. The tree is `scratch/frozen-p2`, only read.

Options: `--work <folder>` in place of `scratch/lane-c/wp13`; `--lane off` keeps the meaning lane
out of both arms (as served it runs only with memory on, because its vectors are kept in the memory
folder, so without this option the arms differ in the lane as well as in memory); `--on-args
"<serve arguments>"` adds serve arguments to the memory-on arm only. Environment switches for the
memory-on arm: `NANO_MEMORY_FAMILY=off` (memory's family of evidence out of the drones'
judgment), `NANO_MEMORY_SEEDS=off` (no recalled places read in the second round).

What it does:

1. Session one, on a fresh scratch Nano folder: one `serve` with memory on answers the answerable
   questions of fold 0 of `tests/dev/folds.json` (from `tests/dev/questions.json` and
   `tests/dev/closure.json`). Around each call a scripted Claude Code session goes through the real
   `hook` command: SessionStart, the prompt, the pack as received, a Read of each answer range and
   an Edit of its anchor line (the tree is not changed), a passing `cargo test` (question `d05`: a
   failing one, so its outcome is refuted), Stop and SessionEnd. The real `ledger` command settles
   the windows and writes the outcome rows; `outcomes` lists them.
2. Session two, a new `serve` on a copy of that folder, asks `session-two.json`: every fold 0
   question again (group `repeated`, the decoy `c29` reported apart), a rewording of each with the
   same meaning in other words (`reworded`), and a question about the same places from another
   angle (`angle`). Once with memory on, once with `--memory off` on another copy.
3. Each reply is scored against the answers' anchor lines as `bench` scores them: right first, in
   the top 3, in the pack, every answer range in the pack, reply tokens (characters / 4), seconds
   per call (client side) and the server's own milliseconds and memory step times from the audit
   rows (recall: the memory read steps added; the outcome evidence step runs inside them and is
   not added again; the write before the reply and its second half after the reply,
   `after-reply`, apart). With memory on: replies holding a span tagged `recalled:`, whether each recalled span's
   lines are the tree's lines now, a refuted outcome shown (`refuted earlier`) and never served as a
   recalled span. With memory off: whether any file of the project memory folder changed.

Pass rules (set before the first run): with memory on, session two is not lower than memory off on
any count, and higher on right first or on every answer range for the repeated and the reworded
questions; the refuted outcome is never served as an answer; recall adds at most 50 ms median to a
plain call; with memory off nothing is written to memory. (The rule's earlier clause "never more
than its deadline" lapsed on 2026-11-16, when recall's deadline was removed: recall always
completes, and its time is held to the 50 ms median.) Run `--work` takes any folder; relative
paths are made absolute.

Files: `run.py` (the harness), `session-two.json` (rewordings and angle questions, written by hand
for this test, answers by the source question's anchors).
