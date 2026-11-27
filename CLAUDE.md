# Neuromorph Nano: notes for Claude Code

Neuromorph Nano is a local, source-available digital organism that works beside a coding agent, by
Luis Jake Gabriel III (Gavriel Technologies). On the user's own machine (a 4 GB GPU is enough), with
four small MicroModels, it reads and remembers the project, serves the context the agent needs,
builds and checks with it, maps parameters and makes its own cartridges, and learns from what
follows; it is reached over MCP, its command line and its local interface. Read `README.md` first, then
`CONTRIBUTING.md` and `docs/README.md`.

## Layout
- `crates/`: the Rust workspace (21 crates). `nmcore` builds the product binary `neuromorph-nano`
  (engine, MCP server, interface server, command line, and the lab as `neuromorph-nano lab`);
  `nmlab` is the lab (parameter maps, cartridge building, benches), also built as its own program
  `nano-lab` for development.
- `interface/`: the local interface (Next.js, static export) served by `neuromorph-nano ui`.
- `designs/`: the design files the engine reads (anatomy, models, settings defaults).
- `kernels/`: the GPU kernels. `docs/`: user and developer documentation. `tools/`: install,
  packaging, sealing and end-to-end checks. `weights/`: the cards of the upstream open weights
  (the weights themselves are downloaded, never committed).

## Build and check
- Rust: `cargo build --release`, `cargo clippy --workspace --all-targets -- -D warnings` (0
  warnings), `cargo test --workspace`. The toolchain is pinned in `rust-toolchain.toml`.
- Interface: `cd interface`, then `npm ci`, `npm run lint`, `npm test`, `npm run build`.
- End to end, against a built binary: `tools/e2e-verify.ps1 -Exe <path to neuromorph-nano.exe>`
  and `node interface/scripts/e2e-ui-verify.mjs --exe <path to neuromorph-nano.exe>`.
- `neuromorph-nano check` runs the engine's own checks.

## Code rules
- Rust uses the standard library and this workspace's own crates only. `unsafe` is forbidden and
  `unwrap` is denied (see `Cargo.toml`, `[workspace.lints]`).
- No em dashes in any text, code comments included.
- Every feature is reachable from the interface, MCP and the API, and they say the same thing.
- The interface shows real data only. A figure that was not measured is named as not measured,
  never given a number.
- Every interface page uses `interface/src/components/Workspace.tsx`: the working system in the
  middle, libraries, references and descriptions in the right rail. Keep the `data-e2e`
  attributes the end-to-end script uses.
- Deleting anything goes through a preview and more than one confirmation step.

## Product rules
- The upstream open weights are never altered; they are only read. Base cartridges, job filters,
  learned units and quantized copies are separate files.
- Nothing leaves the user's machine. Downloads happen only when the user asks.
- Context is never cut to improve a number; it is selected and routed.
- Reads of the user's code live in RAM while the code is unchanged.

## Names
Neuromorph is a digital organism. Say MicroModel (Reader, Writer, Embedder, Reranker, named by the
role; the upstream weights they are built from are not the MicroModel). The Core Brain is singular.
Write the Globalized Context Stack in full or as GCX. GCS is the Global Context Substrate.

## License
PolyForm Noncommercial 1.0.0 (`LICENSE`); commercial use under `COMMERCIAL.md`. Contributions are
licensed as `CONTRIBUTING.md` says. Say "source-available", never "open source", for the Nano.
