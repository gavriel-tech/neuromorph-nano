# nmapi

The MCP face of the Neuromorph Nano and the organism that answers it. Only this crate and the
two binaries wire crates together; the contract table lives in `lib.rs`.

- `mcp`: the MCP server over standard input and output. `tools`: the `nm_*` tools.
- `forage`: the organism's ask path on the NTE, the coordination round and the second round
  (`serve --refine off|uniform|nbv`, default `nbv`). `coordinator`: the Map Coordinator (context
  summary from the GCX context map, model summary from the slots; never touches the Parameter Map).
- `session`: open and close, memory and recall (memory packets plus fresh reads of recalled
  citations). `assemble`: pack forms (adaptive, answer, cited, full; plain-text replies).
- `anatomy`: snapshot, zoom, quarantine, restart, composition. `neural`: binding of the released
  embedder and reranker, the dense lane. `learn`: training items and the loss for `nmlearn`.
- `build`: build, explain, Step Check, improve and propose tools. `delta`: the poll loop that keeps
  the substrate in step with the studied tree.
- `tentacle_pass`, `tentacle_core`, `tentacle_body`, `tentacle_unit`, `drone_windows`,
  `drone_path`, `key_scan`, `scan_dump`, `whole`, `ask`, `brain_side`: the tentacle pass and the
  whole path (drones, tentacles, GCX, Core Brain); off unless their `serve` options are given.

Section checks: `neuromorph-nano check nmapi`.
