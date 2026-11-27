# gen

Scripts for this test material. Python 3.11, standard library only. Run from any folder.

- `make_data.py`: writes `corpus/data/*` and the two special-encoding notes
  (`2026-10-09-site-visit-report.txt` in UTF-16 LE with BOM, `2026-10-22-supplier-visit.txt` in
  Windows-1252). Fixed seeds; a second run gives byte-identical files.
- `make_formats.py`: writes `formats/encodings/` (one note in every text encoding the reader
  decodes, files it must skip, and `expected.json`); a second run gives byte-identical files.
- `make_pdf.py`: the PDF material. `html` writes the manual as HTML pages (one and two columns, a
  page holding only a picture), the plain lines for Microsoft Print to PDF, the expected text of
  every page and the gold passages (`expected/pdf/`); pages are made by construction (each HTML
  page a section with a forced page break), so each page's text is known before printing. `craft`
  writes the files crafted byte by byte: four encrypted copies of `pdf/manual-1col.pdf` (its own
  RC4, AES and the revision 6 hash, checked against FIPS-197 and RFC 6229 vectors first),
  `crafted-features.pdf` and `crafted-nomap.pdf`.
- `make_pdf.ps1` (Windows PowerShell 5.1): runs `make_pdf.py html`, prints the HTML with Microsoft
  Edge headless (`--print-to-pdf`, its own profile and temporary folder under `scratch/`, every
  process of that profile ended after each print) and the lines with Microsoft Print to PDF (.NET
  `PrintDocument` with `PrintToFile`, no dialog), checks each printed file's page count from its
  bytes, then runs `make_pdf.py craft`.
- `check_outlines.py`: checks every file in `expected/outline/` line by line against the corpus and
  against its own heading parser; `--write` rewrites the outline files.
- `check_questions.py`: checks the shape of both question files, that every answer file exists,
  that every anchor is on exactly one line, that the anchor line is `lines[0]` and that the span
  lies inside the file; `--show` prints each span's first and last line.
