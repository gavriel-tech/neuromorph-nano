# nmsubs

The Global Context Substrate (GCS) and its views. Depends on `nmjson`, `nmhash`, `nmlex`, `nmtype`, `nmdoc`.

- `repo`: files read through the format table of `nmdoc` (any material: code, documents, tables,
  data, logs; every file not read is kept with its reason), the item scanner, crates from cargo
  metadata, history from git; skips build output and installed packages, dot folders, and in a
  Neuromorph folder its own `scratch`, `tests` and runtime state.
- `doc`: the document model inside the substrate: code files mapped in (an item block per symbol),
  heading paths and outlines, and the structure-aware chunks of documents (whole blocks up to a
  token budget, never across a heading, a table's header named in every slice, a section link).
  Code keeps its 30-line windows.
- `lock`: the code lock (code symbols, code chunks, crates and reference graph, hashed); read by
  `neuromorph-nano lock` and `check lock`.
- `views`: the views over one store (lexical, symbols, semantic, operational, verification,
  dependency, constraint, interface, uncertainty); `zoom` (symbol, file, crate, workspace); `sync`
  (content hashes, one version counter, invalidation packets).
- `refgraph` (reference graph over items), `lens` (the tentacle lenses), `dense` (the dense lane,
  vectors in RAM only).
- Reads of the user's material live in RAM while it is unchanged; they are never written to disk.

Section checks: `neuromorph-nano check nmsubs`.
