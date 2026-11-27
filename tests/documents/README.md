# tests/documents: retrieval test material beyond code

Made-up material for testing whether the Neuromorph Nano finds the right passage in any kind of
material: a maintenance manual, textbook chapters, dated notes, data files and a small code
workspace with its documents. Every text, product, company and person here is invented for the
test.

## Folders

- `corpus/`: the material the engine studies. Every question's file path is relative to it.
- `gen/`: the fixed-seed data generator and the two check scripts.
- `expected/outline/`: the expected heading outline of eight corpus files.
- `questions/`: `dev.json` (development set) and `held-out.json` (locked held-out set).
- `inflate/`: inflate fixtures from `gen/make_inflate.ps1`: `original/` files, `compressed/`
  (Windows' own Deflate at each level, gzip, zlib), `corrupt/` and `bomb/`; expected faults in
  `expected/inflate.tsv`.
- `general/`: containers and binary files for general reading, from `gen/make_general.ps1` (zip,
  .docx, .xlsx, .pptx, tar, `.tgz`, gzip, a .NET resources file; `legacy.doc` and `locked.zip` are
  crafted byte by byte, as the script says); what the reader must read or skip in
  `expected/general.tsv`. Questions over it: `questions/general.json`.
- `pdf/`: PDF fixtures from `gen/make_pdf.ps1`: the manual printed by Microsoft Edge headless in
  one and two columns (`manual-1col.pdf`, `manual-2col.pdf`, with a running header and footer) and
  by Microsoft Print to PDF (`manual-msprint.pdf`), a three-page record whose middle page is a
  picture only (`inspection-scan.pdf`), and files crafted byte by byte by `gen/make_pdf.py`, as their
  names say: four encrypted copies of `manual-1col.pdf` (`manual-1col-locked-*.pdf`: RC4 40 and
  128 bit, AES-128, AES-256; empty user password, printing and copying not permitted; no program
  shipped with Windows encrypts a PDF), `crafted-features.pdf` (older filters, an object stream, a
  cross-reference stream, a `/Differences` encoding, a composite font with a ToUnicode map) and
  `crafted-nomap.pdf` (a composite font with no way to map glyphs to characters). Expected text
  and gold passages in `expected/pdf/`; questions in `questions/pdf.json`.
- `formats/`: decoder fixtures with hand-written expected results (`encodings/` from
  `gen/make_formats.py`; one folder per structured format, written by hand).

## Encodings

All files are UTF-8 without a byte order mark, with LF line ends, except:

- `corpus/notes/2026-10-09-site-visit-report.txt`: UTF-16 LE with a byte order mark, CRLF.
- `corpus/notes/2026-10-22-supplier-visit.txt`: Windows-1252 (five non-ASCII bytes), CRLF.
- `corpus/data/parts.csv`: UTF-8, CRLF line ends as RFC 4180 asks.

## How the material was made

- `corpus/manual/`, `corpus/textbook/`, `corpus/notes/` (apart from the two files above) and
  `corpus/mixed/docs/` were written by hand for this test.
- `corpus/data/` and the two special-encoding notes are written by `gen/make_data.py` (Python
  standard library, fixed seeds). Running it again gives byte-identical files.
- `corpus/mixed/` is a byte copy of `tests/fixtures/small-repo`, plus `mixed/docs/`.

## Checks

    python gen/make_data.py         regenerate data/ and the two special notes
    python gen/check_outlines.py    check expected/outline against the corpus
    python gen/check_questions.py   check both question files against the corpus
    python gen/make_formats.py      regenerate formats/encodings/
    powershell -File gen/make_inflate.ps1   regenerate inflate/ (zip and gzip bytes differ per run)
    powershell -File gen/make_general.ps1   regenerate general/ (timestamps differ per run)
    powershell -File gen/make_pdf.ps1       regenerate pdf/ and expected/pdf/ (printer bytes differ per run)
    neuromorph-nano check documents the reader, document model and formats against the fixtures

Line numbers everywhere are 1-based lines of the decoded file.
