# nmdoc

Document decoders of the Neuromorph Nano: bytes in, the one document model of `nmtype::doc` out
(derived text plus blocks), or a named reason the file is not read. Standard library only; depends
on `nmtype`, `nmjson` and `nmlex`.

- `format`: the format table. Which files are read and by which decoder, by name and then by their
  first bytes (a PDF named `.txt` is still a PDF, `Doc("pdf")`); which are read generally (`Doc("general")`);
  why the few others are skipped (model weights, secrets, lock, minified, generated and hidden
  files, too large). Build output folders never read (`SKIP_DIRS`).
- `general`: general reading. Containers opened in memory, each member read by the reader that
  fits it and located as `container!member`; anything else read raw. Size, nesting and inflate
  limits; encrypted members listed by name only.
- `inflate`: DEFLATE (RFC 1951), zlib (RFC 1950), gzip (RFC 1952), CRC-32 and Adler-32, with named
  faults and a decompression bomb guard.
- `zip`, `tar`: the two container formats, read in memory (zip: central directory, ZIP64, stored
  and deflate members, CRC check; tar: ustar, GNU long names, pax paths).
- `raw`: runs of readable text in any bytes (UTF-8 and Latin-1, UTF-16 LE and BE), each with its
  byte offset, and what a file is by its first bytes.
- `text`: bytes to text. UTF-8 with and without a byte order mark, UTF-16 LE and BE (by the mark,
  and by the pattern of zero bytes without one), Windows-1252; binary content refused.
- `md`: Markdown and plain-text structure over the file's own lines (headings, lists, tables, code,
  quotes and note boxes, captions, definitions, paragraphs).
- `csv`: CSV and TSV (RFC 4180), delimiter sniffing, header detection, the schema of a table, and
  the rows a question about a column's extreme or empty values points at.
- `markup`: the tag tokenizer HTML and XML share (no DTD, no external entities).
- `html`, `xml`: derived text in Markdown form.
- `json`: JSON and JSON Lines records by key path (tolerant of comments and trailing commas), and
  notebooks as Markdown and fenced code cells.
- `yaml`: a YAML subset read for structure (top-level keys, sequence items, documents).
- `log`: log events with their time stamps (ISO 8601, bracketed, syslog).
- `pdf`: the PDF reader. `lex` (object syntax), `file` (cross-reference tables and streams,
  incremental updates, object streams, repair by scanning), `crypt` (the standard security handler
  with an empty user password: MD5, RC4, AES-128 and AES-256, SHA-256/384/512, revisions 2 to 6),
  `filters` (Flate with predictors, ASCIIHex, ASCII85, LZW, RunLength; image filters named and not
  decoded), `font` (ToUnicode maps, simple encodings with `/Differences` and glyph names, Type0
  fonts, widths), `content` (the text and graphics state of content streams, form XObjects, marked
  content), `layout` (words, lines and paragraphs in reading order by tags or by geometry with
  columns), `structure` (tagged structure, outline, page labels), `fixtures` (the PDF fixtures of
  `tests/documents/` against their pass rules). Pages with no text layer and pages whose font has no
  character map are reported, not indexed.

Section checks: `neuromorph-nano check nmdoc`. Fixtures: `neuromorph-nano check documents`
(`tests/documents/`, PDF part included).
