# questions

- `dev.json`: development set, 50 questions. Use it to tune.
- `general.json`: development set for general reading, 10 questions over `general/` (not
  `corpus/`): answers inside zip, Office, tar and gzip containers and binary files. `file` is
  `container!member` or the file itself; lines are lines of the derived text. Run with
  `bench --mode plain --root tests/documents/general --questions tests/documents/questions/general.json`.
- `pdf.json`: development set for the PDF reader, 12 questions over `pdf/` (not `corpus/`):
  answers in the printed manual; every copy holding the passage (one column, two columns, Microsoft
  Print to PDF, the encrypted copies) is listed and any one counts; lines are lines of the derived
  text. Run with
  `bench --mode plain --root tests/documents/pdf --questions tests/documents/questions/pdf.json`.
- `held-out.json`: held-out set, 80 questions. Locked: score on it only, never tune on it, and never
  copy or paraphrase its questions anywhere else.

Format: `{"note": ..., "questions": [{"id", "kind", "question", "answers": [{"file", "anchor",
"lines", "match"}]}]}`.

- `kind`: define, lookup, procedure, explain, compare, summarise, locate, table, temporal,
  multihop or decoy.
- `file`: path under `corpus/`, forward slashes.
- `anchor`: exact substring of the first line of the answer span, found on exactly one line.
- `lines`: [first, last], 1-based; first is the anchor's line. Procedures span the whole numbered
  procedure; summarise and locate span the section (heading to the line before the next heading of
  the same or higher level); table answers are the matching data rows (one answer per row).
- `match`: `lines` (a citation overlapping the span counts) or `file` (any citation of the file
  counts; used only where a whole short note is the answer).
- compare and multihop questions have two answers; finding either counts. Several answers on other
  kinds mean any one of them counts. Decoys have no answers: the words appear in the material, the
  answer does not.

Checked by `gen/check_questions.py`.
