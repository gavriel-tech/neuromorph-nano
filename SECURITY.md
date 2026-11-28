# Security and privacy

## Reporting a vulnerability

Report security issues privately by e-mail to **`gavriel.techno@gmail.com`** (Gavriel Technologies).
Please do not open a public issue or pull request for a security problem.

Include the version (`neuromorph-nano --version`), what you found, the steps to reproduce it, and
what it exposes. Fixes are released as a new version and recorded in `CHANGELOG.md`.

Supported version: v0.1.01 (Windows x64), the current release.

## Statement

Neuromorph Nano runs entirely on your machine. Your code, your questions, its memory and its
audit log stay on your disk. The Nano sends no telemetry and makes no network request of its own.
The cargo commands it runs for build help and Step Check are passed `--offline`.

- **Local only (the Nano itself).** The MCP server talks to the agent over standard input and
  output. The interface server binds 127.0.0.1 and is not reachable from other machines, unless the
  control `network` is set to `view` or `control` (off by default; see below).
- **What your coding agent sends (cloud vs. local agents).** Neuromorph Nano sends no telemetry and
  makes no outbound network requests of its own, **but your coding agent is a separate process**:
  any Context Packet, MCP tool reply, or lifecycle prime Neuromorph Nano hands to your agent enters
  that agent's conversation and is sent to whichever model backs it. With a cloud-backed agent
  (such as Claude Code, Antigravity, Cursor, VS Code, or Windsurf), that means the cloud model
  provider. Use `.nanoignore` (below) to stop Neuromorph Nano from reading or putting sensitive
  files into Context Packets or primes, or run a local model behind your agent with Air-Gap Mode on
  for zero external transmission (see `docs/agents.md` and `docs/privacy.md`).
- **Downloads and publishing only when you ask.** The only outbound requests are the ones you start: downloading
  the base cartridges, parameter maps and job filters, optional upstream open weights, browsing or fetching from a cartridge source
  (the official library `gavriel-tech/neuromorph-nano-cartridges` or a third-party Hugging Face
  library you added), or explicitly publishing a local parameter map or non-base stack to Hugging Face (`source_publish`, confirmation-gated and audited). They go to huggingface.co through Windows' own HTTPS service (WinHTTP, part of every Windows; TLS and certificates stay with Windows); a downloaded file is kept only
  once its SHA-256 matches. Public reads from the official library send no token by default. A Hugging Face token, when needed (explicit publishing, or a third-party source you marked as needing it), is read from `HF_TOKEN` or the
  `hf` login file, sent only to huggingface.co (never to the file servers a download is redirected to); for the fallback path, from a temporary header file (never on a command
  line), and that file is removed afterwards. Files from any source go through the checks of the cartridge library on this machine: stacks arrive on trial, an
  existing map is never replaced, and base cartridges come from the official library only.
- **Air-Gap Mode.** The control `air_gap` (or `NANO_AIR_GAP=on` in the environment) keeps the
  interface on 127.0.0.1, turns off local network access and tablet pairing, and refuses every
  download, remote listing and publish upload. `GET /api/airgap` and `nm_settings op=airgap` report its state.
- **Writes only from the interface.** The interface server accepts a change (a control or an
  undo) only with its local token, only as JSON, and only from its own pages or a local program
  that sends no browser origin. The token is made when the server starts, kept in memory only,
  placed in the pages the server sends, and compared in constant time. The interface development
  server (`http://localhost:3000`) is accepted only when the server is started with `--dev`. A
  request from any other site, without the token, or addressed to a host name other than
  127.0.0.1 or localhost is refused (403). Every accepted change is logged.
- **Deletes are gated.** Deleting anything (a memory, a project's memory, a downloaded base cartridge, job filter or parameter map, a cartridge
  source, audit rows) is previewed first and confirmed in more than one step, and is audited.
- **Tablets on the local network (opt in).** With `network` at `view` or `control`, the interface
  server also listens on this machine's private addresses for paired tablets: private and
  link-local sources only, Host header checked, a six-digit pairing code (five minutes, once, five
  wrong codes lock pairing for ten minutes), a token per device kept only as a SHA-256 hash and
  revocable, read only in `view`, every pairing, revocation and change audited. **There is no
  encryption on the wire:** on the local network the traffic, including a device's token, can be
  read by anyone who can listen there. The Nano changes no firewall rule.
- **Models read only.** The upstream model files are never written. Each model's main weights
  file is checked against the SHA-256 on its card when the engine opens it. Base cartridges, job
  filters, learned units and quantized copies are separate files.
- **Local code handling & rebuildable caches.** With `memory = off`, reads of your repository stay
  strictly in RAM and nothing about your code is written to disk. With `memory = on` (default), two
  rebuildable local caches live in `memory/projects/<name>-<8 hex>/` on your disk:
  `substrate-cache-<12 hex>.bin` (which holds derived text, symbols, and chunks for fast restarts;
  prunable on **Storage** or via `NANO_SUBSTRATE_CACHE=off`) and `world-model/notes.jsonl` (idle
  Writer summaries when `writer.idle = on`). The dense meaning index (`dense/`) keeps only binary16
  vectors and opaque hashes (`skin.json`), never file text or paths; episodes keep citations and
  line hashes, never copied code; the audit log never holds file contents; and muscle memory
  (`muscle/`) keeps behavioral packs only (`nmmusc-sensory-refused`). Nothing matched by
  `.nanoignore` enters any of these stores.
- **Explicit file exclusions (`.nanoignore`).** Manage exclusions manually via `<repo>/.nanoignore`
  (and nested subdirectory `.nanoignore` files), through the local Interface (**Settings -> Privacy
  and network -> File exclusions (.nanoignore)**), through the local JSON API
  (`GET/POST /api/nanoignore`), or via MCP (`nm_settings` `op=nanoignore | nanoignore_check |
  nanoignore_set | nanoignore_add | nanoignore_remove`), all using gitignore-style patterns (`#`
  comments, `*`, `?`, `**`, trailing `/` for folders, leading or inner `/` anchored to that folder,
  and `!` negation). API, UI, and MCP writes are atomic, logged in Audit (`kind: "nanoignore"`), and
  undoable. Unlike `.gitignore`, `.nanoignore` applies in every repository, immediately invalidates
  the substrate cache when changed, and stops Neuromorph Nano from reading, indexing, embedding,
  writing world-model notes for, priming from, returning build-help source snippets for, or
  including matching files in Context Packets handed to a cloud coding agent (such as Claude Code).
  **Scope boundary:** `.nanoignore` controls what Neuromorph Nano reads and sends; your coding
  agent's own direct tools (such as Claude Code's `Read` or shell commands) remain governed by your
  agent's own ignore and permission settings. For zero external transmission, pair `.nanoignore`
  with your agent's own rules or run a local model behind your agent with Air-Gap Mode on.

## What is stored on disk, and where

All paths are inside the Nano folder, except the MCP entries in the last row.

| Path | What | Written by |
|---|---|---|
| `settings.json` | Your controls (precision per MicroModel, memory, learning, limits, routing, answer mode, local network, air gap) | The interface, `nm_settings` |
| `memory/audit.jsonl` | One row per tool call (written when it starts and again when it ends, with the MCP client's name, the parts it used and their times), setting change, outcome, MCP entry written, pairing and revocation | `serve`, `call`, `setup`, `connect`, `pair`, the interface |
| `memory/live/<project>.json` | While a server runs: the project it serves, the call in progress (tool name only) and the models it holds with their bytes; no file contents, no questions | `serve` |
| `memory/network/` | Pairing state (the code as a salted hash only) and paired devices (token hashes, names, times) | `pair`, the interface server |
| `memory/machine.json` | This machine's measured hardware figures and the GPU kernel choice | The engine |
| `memory/projects/<name>-<8 hex>/` | Memory of one repository: `project.json` (its path, first and last use), episodes, memory hierarchy, temporary notes, structural summaries, world model notes, the meaning index (vectors and hashes), the gate's refusal log | `serve` |
| `memory/` (top level) | The same, for the Nano's own folder | `serve` |
| `muscle/` | Muscle memory packs | The engine |
| `units/` | Learned units, their registry and which are plugged in | The engine, the interface |
| `library/` | The cartridge library and the parameter map library on this machine: base cartridges, stacks, parameter maps, learned cartridges | The installer, the engine, the interface |
| `incoming/` | Files fetched from an online cartridge library until they pass the checks | The engine |
| `cartridge-sources.json` | The cartridge sources you added | The interface, MCP |
| `weights/<model>/files/` | The upstream model files, once downloaded | The installer, the downloads |
| `scratch/` | Disposable work: launcher copies, Step Check copies, hash stamps | The engine |
| `<repository>/.mcp.json`, `.cursor/mcp.json`, `.vscode/mcp.json`, Windsurf's `~/.codeium/windsurf/mcp_config.json`, the Antigravity project plugin | The MCP entry for one client (other servers' entries are kept), only with `setup --write` or `connect` | `setup`, `connect` |

An audit row for a tool call keeps the tool, the project, the MCP client's name, the time taken,
the outcome, the first 200 characters of the question or main argument, the files cited and the
reply size. An argument that looks like a secret is not copied: the row names only why it was left
out.

## What is never stored

- File contents in the audit log. A diff is recorded only as its line count and the files it names.
- Your code base as a copy. Episodes keep the question and each citation as its file, line range
  and a hash of the cited lines, never the lines themselves; recall re-reads the cited lines fresh
  and marks an episode stale when the lines no longer match the hash.
- Anything at all in memory when memory is off: the audit row then keeps only the tool, project,
  time and outcome.

Text that looks like a secret (keys, tokens, private keys, e-mail addresses) is refused by the
memory gate and not saved: not as a note, a temporary note, an episode's question or answer, nor in
the audit log. The gate's refusal log (`refused.jsonl` in the project's memory folder) keeps only
the time, the reason and a SHA-256 of the refused item's key, never its text.

The full privacy statement, with how to check each point yourself, is in `docs/privacy.md`.
