# formats

Fixtures for the decoders, one folder per test, each with the hand-written expected result.

- `encodings/`: the same short note in every text encoding the reader decodes (UTF-8 with and
  without a byte order mark, UTF-16 LE and BE, UTF-16 LE without a mark, Windows-1252) and files
  the format table must skip (binary, image, PDF, Office, archive, lock, minified, secrets, too
  large). Written by `../gen/make_formats.py`; the expected outcome of each file is in
  `encodings/expected.json`, the decoded text in `encodings/text.txt`.
- `html/`, `xml/`, `notebook/`, `json/`, `jsonl/`, `yaml/`, `log/`: one input each, written by
  hand, with its expected result: `expected.txt` (the derived text, compared after white space is
  normalised), `expected-outline.tsv` (line, level, heading), `expected-blocks.tsv` (first line,
  kind, label of every block) or `expected-events.tsv` (first line, last line, label and time in
  milliseconds of every log event).

Checked by `neuromorph-nano check documents`. Never studied as material: no question points here.
