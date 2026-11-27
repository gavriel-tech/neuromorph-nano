# Contributing to Neuromorph Nano

How to build Neuromorph Nano, check a change, write code that fits the project, and how
contributions are taken in and licensed.

Neuromorph Nano is source-available, not open source: it is licensed under the PolyForm
Noncommercial License 1.0.0 (`LICENSE`), with commercial use only under an individually negotiated
license (`COMMERCIAL.md`). Read the [contributor licensing](#contributor-licensing) section before
you send a contribution.

## How this repository works

- The maintainer's work arrives in this repository as one commit per release (for example
  "Release v0.1.0"), tagged with the version.
- Contributors fork this repository and open pull requests against `main` here. Accepted pull
  requests are merged here, keeping their commits and authorship, and are carried into every later
  release unchanged, so the merged lines keep their author in `git blame`. That is how
  contribution credit is measured (see [`CONTRIBUTOR_POOL.md`](CONTRIBUTOR_POOL.md)).
- A later release may change or replace merged code like any other code; credit follows the lines
  that survive, as `CONTRIBUTOR_POOL.md` says.

## Build

The toolchain is pinned in `rust-toolchain.toml` (Rust 1.99.1, MSVC, Windows x64). Build output
stays in `target/`; `.cargo/config.toml` sets the processor baseline (AVX2, FMA, F16C).

```
cargo build --release
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

The build makes `target\release\neuromorph-nano.exe`, the product: engine, MCP server, interface
server, command line, and the lab (`neuromorph-nano lab ...`: measurement, parameter maps and
cartridge building). The same lab commands are also built as their own program,
`target\release\nano-lab.exe`, for development. Clippy must report 0 warnings.

Interface (Node.js 20.9 or later):

```
cd interface
npm ci
npm run lint
npm test
npm run build
```

`npm run build` writes the static export to `interface/out/`, which `neuromorph-nano launch ui`
serves. `npm run dev` runs the development server on port 3000; it reaches the engine's API on
4870, so start the engine with `neuromorph-nano launch ui --dev` (writes need the engine's local
token, which it gives the development server only under `--dev`).

GPU kernels: `kernels/build/*.fatbin` are prebuilt and embedded at build time. Rebuild them with
`kernels/build-fatbins.cmd` (CUDA 13.0 and the Visual Studio C++ tools) only when a file in
`kernels/cukern/` changes, then run `tools/gen-integrity.ps1` and
`neuromorph-nano selftest engine`.

## Checks before a change is accepted

1. `cargo build --release` with no warnings, and
   `cargo clippy --workspace --all-targets -- -D warnings` with 0 warnings.
2. Section checks of every crate you touched, then designs and anatomy:
   `target\release\neuromorph-nano.exe check <crate>... designs anatomy`.
   `check <crate>.<part>` runs one check; `check` with no name runs every section.
3. Cards. Every library crate has a `part.json` card holding the SHA-256 of each source file
   (`nmcore` and `nmlab`, the two programs, have none), so any edit (even a comment) makes the card
   stale. Regenerate and seal, then verify:
   ```
   powershell -File tools\gen-cards.ps1
   target\release\neuromorph-nano.exe seal crates\<crate>\part.json
   target\release\neuromorph-nano.exe check cards
   ```
4. Engine fingerprint, when the engine or a design it reads changed:
   `target\release\nano-lab.exe run s6 fingerprint`; the output must not change.
5. `target\release\nano-lab.exe conform`: no new violation.
6. `cargo test --workspace` and `cargo test --release -p nmnte --test acceptance`.
7. MCP smoke test:
   `target\release\neuromorph-nano.exe call nm_context_pack "{\"question\":\"Where is the struct Coordinator defined?\"}" --root .`
   answers with citations (`tools/mcp-smoke.ps1` checks `initialize`, `tools/list` and one cited
   answer over standard input and output).
8. Interface changes: `npm run lint`, `npm test` and `npm run build` in `interface/`.
9. End to end, against the built program, each on an isolated home and a free local port:
   ```
   powershell -File tools\e2e-verify.ps1 -Exe target\release\neuromorph-nano.exe
   node interface\scripts\e2e-ui-verify.mjs --exe target\release\neuromorph-nano.exe
   ```
   The first checks the HTTP API and the MCP server; the second checks every interface page and
   control in headless Microsoft Edge and needs Node.js 22 or later. Both fail on any failed or
   skipped check. Keep the `data-e2e` attributes the interface script uses.

`tools/verify.ps1` runs the release checks in order and stops at the first failure (it builds
into `scratch/gate/target`).

## Rules for code

- **No outside crates.** The engine uses the Rust standard library and this workspace's own crates
  only. `Cargo.lock` must list nothing else.
- **No `unsafe`.** `unsafe_code` is forbidden in the workspace (`Cargo.toml`,
  `[workspace.lints]`). The one exception is `nmffi`, the bridge to the NVIDIA driver, where
  `unsafe` is allowed only in its two FFI modules and every block carries a `SAFETY` comment
  (enforced by clippy).
- **No `unwrap`, `dbg!` or `todo!`** outside tests (clippy denies them). Return an error from
  `main` instead of `std::process::exit`.
- **No em dashes** in any text: code, comments, documentation and the interface.
- **The upstream open weights are never written.** Base cartridges, job filters, learned units and
  quantized copies are separate files.
- **Reads of the studied code stay in RAM.** Never write a copy of the user's code to disk.
- **The Nano sends nothing off the user's machine.** No telemetry, no network request of its own;
  downloads happen only when the user asks, and never while the air gap is on. (What the user's
  coding agent sends to its own model is the agent's, and the docs say so.)
- **Context is selected and routed, never cut** to improve a number.
- **Real data only.** The interface shows measured figures; a figure that was not measured is named
  as not measured, never given a number.
- **Parity.** Every feature is reachable from the interface, MCP and the API, and they say the same
  thing.
- **Deletes are gated.** Deleting anything goes through a preview and more than one confirmation
  step.
- **Every interface page uses `interface/src/components/Workspace.tsx`:** the working system in
  the middle, collections, references and descriptions in the right side panel.
- **Fixed model input text stays byte-identical.** Some constants are tokenized and fed to the
  model in self-tests; changing their text changes the fingerprint.

## Comments

- Module docs (`//!`) say what the module does, its inputs and outputs, and its invariants.
- Item docs (`///`) say what the item does and its contract, with `# Errors` and `# Panics` where
  they apply, and units.
- Inline comments (`//`) say why the code does something that is not obvious: a constraint, a
  measured reason, a hazard. Never what the next line does.
- No dates, people's names, plan or step names, or history in new code. History belongs in
  `CHANGELOG.md` and git.
- Keep the `reason = "..."` on every `#[allow]` and `#[expect]`.

## How to change a part

One change, then one short isolated test: about one minute, three at most, with a smoke run
first. Judge it by two questions: did the primary goals improve (context, codebase understanding,
code improvement, build), and is the part closer to its design. A part with a design reason behind
it that shows no gain is not removed; find out why it does not perform.

## Naming

- Neuromorph is a **digital organism**. The Nano is its first commercial release.
- **Neuropod**: tentacles and drones. **Core Brain**: singular; MicroModels are its regions.
- **MicroModels** are named by their role: **Reader**, **Writer**, **Embedder**, **Reranker**. The
  `qwen3-*` ids are the upstream open weights a MicroModel is built from, not the MicroModel.
- **Globalized Context Stack**: written in full, as **GCX**, or called the spinal cord or brain
  stem. No other short form or name.
- **GCS** is the Global Context Substrate, a different part.
- Write **MicroModel**, **muscle memory**, **cartridge**, **Context Packet**.
- Say **source-available**, never "open source", for the Nano. The upstream Qwen3 models are
  **open weights**.

## Contributing weight cartridges, parameter maps and techniques

Neuromorph Nano is built around a modular cartridge and parameter-mapping ecosystem, and
contributions are welcome in three areas.

- **Base cartridges are what the MicroModels serve.** The upstream Qwen3 open weights are kept as
  released, read only: they supply each model's tokenizer and configuration, they are the
  reference every base cartridge is checked against, and they are the fallback a MicroModel serves
  when its base cartridge is missing or switched off (a role switched to `open`). They are also the
  starting point for mapping and building new base cartridges and filters from scratch.
- **What you can contribute:**
  1. **New base cartridges** for the four MicroModel roles (Reader, Writer, Embedder, Reranker), or
     for new compatible open-weight foundations, for the official cartridge library on Hugging Face
     (`gavriel-tech/neuromorph-nano-cartridges`).
  2. **Parameter maps, job filters (`nmengn-sockets/1`) and lenses** (`pmap-*`, `act-*`,
     `zsep-map`, `tol-*`, `super-weights.json`, `job-filter-*.json`).
  3. **New parameter-mapping, surgery, pruning and quantization techniques** (new mapping passes,
     calibration algorithms, CUDA or CPU kernels, or hardware-tier profiles).

The official library is under the same PolyForm Noncommercial License 1.0.0 as the Nano, with the
upstream Apache-2.0 notices beside it. A cartridge or map you contribute to it is licensed as the
[contributor licensing](#contributor-licensing) section says.

### Submission protocol and acceptance gate

Every cartridge, parameter map or parameter technique submission must include:

1. **Provenance and upstream licensing:** the exact upstream open-weight model repository, the
   pinned commit revision, and the upstream license (Apache-2.0, MIT or compatible), for
   attribution in `THIRD_PARTY_NOTICES.md`.
2. **SHA-256 integrity manifest:** the SHA-256 of every uploaded `.cartridge`, `.json` or binary
   file, plus its compatibility card (target role, base model id, layer and head dimensions, base
   SHA-256, precision and VRAM footprint).
3. **Held-out split-gate evidence:** reproducible held-out benchmark output (split gate or
   `bench --mode board`) showing decoy withholding does not regress and measuring the accuracy,
   latency or VRAM improvement, naming the hardware it was tested on.
4. **Strict sensory boundary (`nmtype::class`):** only behavioural weights, masks, scales and
   parameter metadata. Never embed proprietary code, private prompts, personal paths or unlicensed
   training text.
5. **Reproducible build steps:** the exact `nano-lab` or mapping command used to produce the
   cartridge or map, so maintainers can verify and rebuild it independently
   (`docs/build-flow.md` lists the commands used for v0.1.0).

Open a pull request on this repository (or a cartridge submission issue referencing your verified
upload) with the checklist above. Once it passes the held-out gate and the SHA-256 check,
maintainers publish accepted cartridges to `gavriel-tech/neuromorph-nano-cartridges`.

Substantive merged code, kernel, cartridge and interface contributions by verified human
controllers are credited in [`contributors-ledger.json`](contributors-ledger.json) under the
**Neuromorph Nano-only** contribution reward distribution system and anti-cheat policy in
[`CONTRIBUTOR_POOL.md`](CONTRIBUTOR_POOL.md).

## Contributor licensing

Neuromorph Nano is source-available under the PolyForm Noncommercial License 1.0.0 (`LICENSE`) and
is also offered under separate, individually negotiated commercial licenses (`COMMERCIAL.md`).

By submitting a pull request, patch, issue attachment, or other contribution to this repository,
you confirm that you have the legal right to submit it and you grant **Luis Jake Gabriel III
(Gavriel Technologies)** a perpetual, worldwide, non-exclusive, irrevocable, royalty-free,
transferable copyright and patent license to use, reproduce, modify, display, perform, distribute,
sublicense, and re-license your contribution under both `LICENSE` and any commercial or other
license terms offered by Luis Jake Gabriel III (Gavriel Technologies).
