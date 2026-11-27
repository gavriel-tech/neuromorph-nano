# Development questions

`questions.json`: 40 questions about the Nano's own code (kinds: name, value, relation, behaviour,
explain, file), each with the file and line range that answers it. An answer is located at run time
by its `anchor` text (the line it names), so the range follows the code as it moves; `lines` is the
range when the set was written and checked by hand. `match: "lines"` counts a cited chunk whose line
range overlaps the answer; `match: "file"` counts any cited chunk of that file.

Used by `neuromorph-nano bench` (right first, in the top 3, in the pack). For development only, never
a release measure: the set is small and was written by the same hands that tune the parts. `tests/`
is never studied by the Nano, so the questions do not reach the material they are asked about.

`closure.json`: 30 tuning questions on the frozen tree `scratch/frozen-p2` (6 locate, 4 value, 8
explain, 6 trace, 4 on files that are not Rust, 2 decoys with no answer). Each lists every deciding
range as its own answer (a definition and what it depends on: callee, caller, constant, default,
type), a one-sentence `statement`, the `facts` a right reply states, its `kind` and its `group`.
Run with `bench --mode board --root scratch/frozen-p2 --questions tests/dev/closure.json`.

`folds.json`: the fixed split of the 70 tuning questions (`questions.json` and `closure.json`) by
id into folds 0 to 4, used for every fitted number; never changed once used.

`code-lock.txt`: the code lock (`neuromorph-nano lock`) of `scratch/frozen-p2` and
`tests/fixtures/small-repo`, recorded twice before the any-data work (2026-11-15) with equal
results. `neuromorph-nano check lock` recomputes it; any change to how code is read breaks it,
and documents never enter it.
