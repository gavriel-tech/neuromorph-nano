# expected

`outline/`: the expected heading outline of eight corpus files: the manual, both textbook chapters
and five notes (three Markdown notes and the two special-encoding notes). File name: the path under
`corpus/` with `/` replaced by `__`, plus `.tsv`.

One line per heading, in order: `line_number<TAB>level<TAB>heading text`, 1-based lines of the
decoded file.

- Markdown: ATX headings without the leading and closing `#`s, trimmed; setext headings count;
  nothing inside fenced code blocks or YAML front matter is a heading.
- Plain text: a title line underlined with `=` (level 1) or `-` (level 2), at least 3 characters,
  with the title line's number; or a numbered line like `2.3 Measures of spread` after a blank
  line, under 80 characters, no final period, level = count of numbers.

Checked by `gen/check_outlines.py`.

`inflate.tsv`: each corrupt or bomb file of `inflate/` and the named fault it must give
(`file<TAB>fault`). `general.tsv`: every document and skip the reader must give for `general/`
(`path<TAB>decoder<TAB>general mark`, or `path<TAB>skip<TAB>reason`). Both checked by
`neuromorph-nano check documents`.

`pdf/`: what the PDF reader must give for `pdf/`, written by `gen/make_pdf.py`:

- `<name>.txt`: the expected text of every page of `pdf/<name>.pdf`, in `--- page N ---`
  sections: headings, paragraphs, list items with their bullet or number, table rows with their
  cells in order, code lines; no running header or footer. The encrypted copies are expected to
  read exactly as `manual-1col.pdf`.
- `gold.tsv`: gold passages (`file`, `page`, `passage`): each must be found on that page only.
- `pages.tsv`: the number of pages each printed file was planned to have; `gen/make_pdf.ps1`
  checks it against the printed bytes.
- `src/`: the HTML pages and plain lines the printers were given, and the picture of the scanned
  page.

The pass rules (`neuromorph-nano check documents`): per page, character similarity of the derived
text to the expected text at least 0.98 (one minus the edit distance over the longer text, white
space collapsed); every gold passage on its page; each encrypted copy equal to the plain file; the
picture page of `inspection-scan.pdf` and the page of `crafted-nomap.pdf` reported, not indexed.
