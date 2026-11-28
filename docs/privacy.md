# Privacy

Neuromorph Nano runs entirely on your machine. Your code, your questions, its memory and its audit
log stay on your disk. This page says what that means in practice and how you can check it.
The short form is in `SECURITY.md`.

## What leaves your machine

The Nano itself sends nothing off your machine, unless you ask for a download: no telemetry and no
network request of its own.

Your coding agent is separate: the context the Nano gives it (context packs, primes) goes to the agent's model the same way as any file the agent reads. With a cloud model, such as Claude in Claude Code, that is its provider. For nothing to leave at all, run a local model behind your agent (see [Agents and models](agents.md)) and turn on Air-Gap Mode.

- Serving, answering, memory, learning and the interface make no network connection.
- The MCP server talks to your agent over standard input and output, not over the network. The
  warm daemon (`neuromorph-nano daemon`) listens on 127.0.0.1 only (default port 7421).
- The cargo commands it runs for build help, Step Check and the Stop hook are passed `--offline`.
- The models are open weights on your disk (the Qwen3 open weights, Apache-2.0) and the Nano's
  base cartridges built from them. No outside model or service is called.
- Building the interface uses npm on the development machine; the built interface is served by
  the engine and loads nothing from outside.

### Downloads and publishing, only when you ask

The Nano has one piece of outbound network code: requests to Hugging Face (`huggingface.co`),
started only by you, from the interface (Models, Downloads / Libraries; Cartridges, Get more) or through MCP
(`nm_settings`). It connects through Windows' own HTTPS service (WinHTTP, part of Windows; `curl.exe` only as the
fallback), one download at a time, and sends a token only to huggingface.co. What it
fetches or publishes:

- the four base cartridges, active parameter maps, job filters and shared stacks from the official Neuromorph cartridge library on Hugging Face,
  `gavriel-tech/neuromorph-nano-cartridges` (public; fetched without sending a token by default);
- the optional released open weights, each from the Hugging Face repository and revision its card pins;
- files from cartridge libraries you add yourself by their Hugging Face repository name
  (third-party sources). Browsing a source lists its files (one request); fetching downloads one
  file (or `fetch all` downloads every active parameter map and non-base stack listed). A third party's file is held to the same checks as one of ours: a library stack arrives on
  trial, a parameter map goes through the map checks and never over an existing map, and a base
  cartridge from a third party is listed but not installed. The list of sources is
  `cartridge-sources.json` in the Nano folder; every change to it is an audit row (kind
  `cartridge-source`);
- explicit, confirmation-gated publishing (**Publish to Hugging Face**, `source_publish`): uploads only the single local parameter map or non-base stack you pick to a Hugging Face repository you specify (or as a Pull Request to the official library), via native WinHTTP `POST`, audited and refused while Air-Gap Mode is on.

A downloaded file is placed only once its SHA-256 matches (where the source publishes one); a partial file
resumes. Your Hugging Face token (`HF_TOKEN`, else the `hf` login file) is sent only to
`huggingface.co`, and only when you explicitly publish or mark a private/gated third-party source as needing it; on the fallback path it is passed in a header file
that is removed afterwards. Nothing about your code, questions or memory is sent with a download or publish.
The download state is kept in `memory/downloads.json`. With the air gap on, browsing,
downloading and publishing are refused.

### Your own repositories, only when you back up

When you add one of your own repositories (Storage, Your repositories) and press back up, the parts
you chose (memory, parameter maps, cartridges, learned units, settings) are sent to that repository
and nowhere else; your token goes to that repository's host only, and only if you ticked it. Nothing
is sent until you back up, and the air gap refuses it.

### Air-gap mode

The control `air_gap` (off by default; also locked on by the environment variable
`NANO_AIR_GAP=on`) applies at once: the interface stays bound to 127.0.0.1, local network access
and tablet pairing are switched off, and every remote listing and download is refused.
`GET /api/airgap` (and `nm_settings` with `op: airgap`) reports the state as it is now: whether
the air gap is on and from where, the network setting in force, whether only the loopback address
is served and how many local network listeners there are, whether outbound network code and
remote imports are allowed, and whether a download is running. The interface shows it under the
page list.

### Your agent

What your agent does with the context the Nano gives it is up to the agent. Claude Code, for
example, sends its conversation, including the tool replies and any prime, to its own model
service.

### Excluding files with `.nanoignore`

If your repository contains tracked files you want Neuromorph Nano to avoid completely (for example
keys, internal notes, legal or customer data, or proprietary files you do not want placed into
Context Packets sent to a cloud agent like Claude Code), manage exclusions across any of four
equivalent surfaces:

| Control Surface | How to Use |
|---|---|
| **Manual / agent file edit** | Create or edit `<repo>/.nanoignore` (and optionally nested `<repo>/<subdir>/.nanoignore`) directly in your editor or repository. |
| **Local Interface** | Open **Settings -> Privacy and network -> File exclusions (`.nanoignore`)** to pick a repository, edit rules or quick-add patterns, test whether a relative path is excluded, and save with one-click Undo on Audit. |
| **Local JSON API** | `GET /api/nanoignore?project=<key_or_folder>[&test=<rel_path>]` reads rules, nested `.nanoignore` files, and path match status; `POST /api/nanoignore` (`action: "set" | "add" | "remove"`) writes atomically with an undoable `nanoignore` audit row. |
| **MCP (`nm_settings`)** | `op=nanoignore`, `nanoignore_check` (`test`), `nanoignore_set` (`content`), `nanoignore_add` (`pattern`), `nanoignore_remove` (`pattern`), and `op=undo` (`id`) so coding agents can manage exclusions with full Audit accountability. |

`.nanoignore` uses gitignore-style syntax:

```gitignore
# Exclude sensitive files and directories from Neuromorph Nano
*.pem
.env*
secrets/
internal/
/top_only.txt
!secrets/public-template.json
```

- **Enforced everywhere in Neuromorph Nano:** matching files and directories are never walked into
  the Global Context Substrate, never embedded into the dense meaning index, never summarized by
  the idle Writer, never re-read by fresh memory recall (`fresh_text`), never cited in primes
  (`prompt-recall`, `failure-fix`, `edit-impact`, `core`, `resume`), never surfaced as build-help
  source snippets, and marked sensitive in hook logs.
- **Immediate cache invalidation:** adding, editing, or removing a `.nanoignore` file changes the
  substrate manifest fingerprint so `substrate-cache-*.bin` rebuilds immediately without serving
  previously cached content.
- **Applies in every repository:** unlike `.gitignore`, `.nanoignore` is never bypassed, even in a
  Neuromorph folder.
- **Honest scope boundary:** `.nanoignore` stops **Neuromorph Nano** from reading or sending
  matching files to your agent. Your coding agent itself (such as Claude Code) still has its own
  direct file and shell tools; pair `.nanoignore` with your agent's own ignore/permission settings
  (or use a local model with Air-Gap Mode on) if you also want to block direct reads by the agent.

## The interface server

- It binds 127.0.0.1, so it cannot be reached from another machine, unless you set the control
  `network` to `view` or `control` (see Tablets on the local network below).
- Reads (GET, HEAD) are open to programs on this machine.
- A change (any POST) needs the server's local token. The token is
  made when the server starts, kept in memory only, and placed in the pages the server sends; a
  page from another site cannot read those pages, so it cannot learn the token. The change must
  also be JSON and come from the interface's own pages or a local program that sends no browser
  origin. The interface development server on `http://localhost:3000` is accepted only when the
  server is started with `--dev`. Anything else, and any request addressed to a host name other
  than 127.0.0.1 or localhost, is refused with 403.
- Every accepted change is written to the audit log and can be undone.

## Tablets on the local network

Off by default. With the control `network` at `view` or `control`, the interface server also
listens on this machine's private network addresses so a paired tablet can open it.

- **No encryption on the wire.** The engine has no outside libraries, so there is no TLS. On your
  local network the traffic, including a tablet's token and everything the interface shows (audit
  rows, questions, settings), can be read by anyone who can listen on that network. Use it only on
  a network you trust. This is why the default is off and why `view` exists apart from `control`.
- Only private and link-local addresses are served, and only requests whose Host header names the
  address and port they reached.
- A tablet pairs with a six-digit code (five minutes, once; five wrong codes lock pairing for ten
  minutes) and then gets its own token. Only a SHA-256 hash of the token is kept, in
  `memory/network/devices.json`, with the tablet's name and when it was paired and last seen. A
  tablet never receives the computer's own local token. Revoke a tablet from the Settings page.
- `view`: a paired tablet reads everything and changes nothing. `control`: it can also change
  settings; each change is audited with the tablet's name and can be undone.
- Pairing, wrong codes, lockouts, revocations and each tablet's first request after the server
  starts are written to the audit log.
- The operating system may ask once whether to let the program accept connections. The Nano never
  changes a firewall rule itself.

## Your code

- What the Nano reads from your repository is held in RAM while the material is unchanged, and
  read again when it changes. With memory on, two caches keep it on disk in the repository's
  memory folder so that a restart is fast:
  - the substrate cache, `substrate-cache-<12 hex>.bin`: the read of the folder as it was (the
    derived text of each file, its pieces, symbols and imports), keyed by a fingerprint of every
    file's path, size and time. **It holds the text of your material.** Any changed file makes it
    unused, and it is written again. It is rebuildable: prune it on the Storage page or delete the
    file. With memory off it is not written.
  - the meaning index (see Meaning index on disk below): one vector per piece, never its text.
- The idle writer (the Writer MicroModel, while the `writer.idle` control is `on`, the default)
  writes short notes about your material when the Nano is idle: one-line notes per piece, short
  summaries of served pieces, file notes and a project outline. They are model-written text about
  your code, kept with memory on in `world-model/notes.jsonl` in the memory folder, each keyed by
  the content hash of what it was written from. Set `writer.idle` to `off` to stop them; they are
  rebuildable and go with the memory folder.
- Memory keeps questions, the citations of each answer (file, line range and a hash of the cited
  lines, never the lines themselves) and outcomes, not copies of your files. Recall re-reads the
  cited lines fresh from your files, finds them again if they moved, and marks the episode stale
  when they changed. A structural summary keeps only counts (crates, files, symbols) and a digest
  per code version.
- Muscle memory keeps behaviour only (which steps helped), never code that was read.
- The training store keeps places, hashes and labels, never code, unless you set the `examples`
  control to `full`; then it also keeps the text of the passages the Nano judged, secrets dropped
  (see Training store below).
- Build help and Step Check work on a scratch copy and never write into your repository. The Nano
  writes in your repository only when you connect it, and only its own entries: `.mcp.json` and
  `.claude/settings.local.json` (`setup --project <repo> --write`, `--hooks --write`), the
  Antigravity plugin folder `.agents/plugins/neuromorph-nano/` (`setup --antigravity`), and
  `.cursor/mcp.json` or `.vscode/mcp.json` (`connect cursor`, `connect vscode`). `connect windsurf`
  writes Windsurf's own file in your user folder (`~/.codeium/windsurf/mcp_config.json`). Each
  write keeps every other entry, is backed up first and is audited.

## The models

- The released open weights are only read, never written or altered.
- Each model's main weights file is checked against the SHA-256 on its card when the engine opens
  it. A base cartridge is used only over the open weights it was made for.
- What the Nano learns is kept apart from the weights: learned units and cartridges, each of which
  you can plug out (Cartridges page) or remove. Nothing learned changes the open weights or a base
  cartridge.

## What is kept on disk

| Path | What |
|---|---|
| `settings.json` | Your controls |
| `memory/audit.jsonl` | One row per tool call, setting change, outcome, download, source change and setup write. A tool call row keeps the tool, project, mode, the name the connecting MCP client gave (`client`), the session, time, outcome, the stages it went through with their timings, the first 200 characters of the question or main argument (left out, with the reason, when it looks like a secret), the files cited and the reply size. A diff is kept only as its line count and the files it names. File contents are never logged. Tool call rows are trimmed to the `audit.keep_days` control (28 days of use by default) when you trim the log (Audit page, or `neuromorph-nano keep --apply`); every other row is kept, so undo keeps working |
| `memory/projects/<name>-<8 hex>/` | One repository's memory: its path, episodes, memory hierarchy, temporary notes, structural summaries, and the memory system's bookkeeping |
| `memory/projects/<name>-<8 hex>/substrate-cache-<12 hex>.bin` | The substrate cache, with the text of your material (see Your code above) |
| `memory/projects/<name>-<8 hex>/world-model/` | The idle writer's notes about your material (see Your code above) |
| `memory/projects/<name>-<8 hex>/meaning/` | The recall meaning cache, while the `recall_cache` control is on: query vectors of earlier questions, calibration floors and reranker pair scores; hashes, vectors and numbers only, never text |
| `memory/projects/<name>-<8 hex>/continual/` | The continuous learning engine's state: experiences recorded from judged outcomes, with their places, and the patterns, procedures and policies kept from them, with snapshots for rollback |
| `memory/projects/<name>-<8 hex>/learned/` | That project's learned units (the Nano's own folder keeps them in `units/learned/`) |
| `memory/projects/<name>-<8 hex>/situation.jsonl` | What Claude Code did in that repository, as its hooks reported it (see Hooks below); no file contents |
| `memory/projects/<name>-<8 hex>/prompts.jsonl` | The text of your prompts, after secret-dropping, when the `prompts` control is `keep` (see Hooks below) |
| `memory/projects/<name>-<8 hex>/evidence.jsonl` | The evidence ledger: what followed each context pack the Nano served (see Evidence ledger below); paths, line ranges, outcomes and hashes, no contents |
| `memory/projects/<name>-<8 hex>/guidance.json` | The reminders, loops and steps you or your coding agent set for that repository, with their text (see `user-guide.md`, Reminders, loops and steps) |
| `memory/projects/<name>-<8 hex>/guidance-deliveries.jsonl`, `guidance-history.jsonl`, `guidance/<session>.json` | What was sent to the agent and when, with its text (the newest 1,000 kept once the file passes 512 KiB); each guidance change before and after, for undo; per-session counters (prompts, tool calls, deliveries, steps nudged, files touched by path) |
| `memory/projects/<name>-<8 hex>/primes/<session>.jsonl` | Which primes were given in one Claude Code session (time, id, kind, hook, size), for the never-twice rule; no text (see Priming below) |
| `memory/hardware.json` | This machine's processor, thread count, graphics card name and memory, and RAM size, which the evidence ledger tags its rows with; read again after a day |
| `memory/machine.json` | The engine's measures of this machine's capacity |
| `memory/projects/<name>-<8 hex>/training/` | The training store: what is kept for fitting cartridges later, as the `examples` control says (see Training store below) |
| `memory/projects/<name>-<8 hex>/dense/vectors.bin` | The meaning index, when memory and the `dense_cache` control are on (see Meaning index on disk below); the Nano's own folder keeps it in `memory/dense/` |
| `memory/projects/<name>-<8 hex>/setup/` | What `setup` created in the repository's Claude Code files, and a backup copy of each file as it was before every write and removal |
| `memory/network/` | Paired tablets (see Tablets on the local network above) |
| `memory/downloads.json` | The state of the downloads you started |
| `cartridge-sources.json` | The cartridge libraries you added |
| `library/` | The cartridge library and the parameter map library: base cartridges, parameter maps and job filters, learned and preference cartridges, stacks |
| `incoming/` | Files fetched from an online cartridge library, held until they pass the checks, then moved into the cartridge library |
| `muscle/` | Muscle memory packs |
| `scratch/` | Disposable work; safe to delete when the Nano is not running |

All of it is inside the Nano folder. The Storage page shows every item with its exact path and
size, and whether it is rebuildable, removable or read only. Forgetting a project there
(`POST /api/projects/forget`, confirmed in two steps, audited), or deleting its folder under
`memory/projects/` while no server serves it, removes everything the Nano kept about that
repository.

## Hooks: what Claude Code tells the Nano

With `setup --project <repo> --hooks --write`, Claude Code runs the Nano's hook program at eight
moments: a session starts, you send a prompt, it is about to edit a file, a tool has run, a tool
has failed, it stops, it is about to shorten its conversation, and a session ends. (The
Antigravity plugin's hooks, from `setup --antigravity --hooks`, run at a session start, before
each invocation, after a tool and at a stop, and are recorded the same way.) Each time it sends the program
the full event as JSON on standard input: for a prompt, your prompt's text; for a tool, the tool's
input and output (for a file write, the whole file; for a command, its whole output).

The program runs on your machine, sends nothing anywhere, and keeps only a short record per event
in `situation.jsonl` in the repository's memory folder:

- the time, the session and prompt ids, the event, the tool's name;
- the files touched, relative to the repository (a file outside it is written `<outside>`), and
  the line ranges when known (a read's range, or where an edit's text sits, found by reading the
  file once in memory; the text itself is not kept); a file whose name says it holds secrets
  (`.env`, `*.pem`, `*.key`, `id_rsa`, ...) is kept as a path and the record is marked
  `sensitive`;
- for a command: the command after secret-dropping, cut to 200 characters; whether it passed; its
  exit code; for a build or test, the test counts and up to five located errors (code, file, line,
  a message of at most 120 characters);
- for long output, a digest only: its size, a 16-hex hash, and its first and last non-empty lines
  after secret-dropping, each at most 120 characters;
- for a call to the Nano's own tools, the question (after secret-dropping) and the episode id;
- for a prompt: its length, a hash, how many secrets were dropped from it.

Never kept: file contents, an edit's old or new text, full command output, Claude Code's own replies,
compaction summaries and instructions (only their length), transcript paths.

### Your prompts

The text of a prompt is kept only while the control `prompts` is `keep` (the default), on this
machine only, in `prompts.jsonl` in the repository's memory folder, after secret-dropping. It is
shown by `neuromorph-nano situation --project <repo>`. Text older than the control `prompts.days`
(default 30 days) is removed at the next `situation` read, and at a prompt when the last removal was
more than an hour ago. Clear it at any time:

```
neuromorph-nano situation --project <repo> --clear-prompts
```

The clearing is logged in the audit (the count only). Set `prompts` to `off` to keep only each
prompt's length and hash from then on; text already kept stays until it expires or is cleared.

### Secret-dropping

One routine is applied to every piece of prompt or tool text before it is stored (prompts, commands,
error messages, digest lines, questions to the Nano). It replaces anything that looks like a secret
with a marker naming its kind, for example `[secret dropped: password]`: private key blocks; keys and
tokens with known prefixes (`sk-`, `sk_live_`, `ghp_`, `github_pat_`, `glpat-`, `xoxb-`, `AIza`,
`AKIA`, `hf_`, `npm_` and others); JSON web tokens; `Authorization:` header values and `Bearer`
tokens; the password in `scheme://user:password@host`; values given to names such as `password`,
`PGPASSWORD`, `secret`, `token`, `api_key`, `AccountKey`, `client_secret`, `sig`, `cookie`, as
`name=value`, `name: value` or `--name value`; `curl -u user:pass`, `mysql -pPASS`, `docker login
-p PASS`, `sshpass -p PASS`; prose such as "the password is hunter2"; and long random-looking
strings. Its test list holds 40 cases that must be dropped and 32 that must be left alone, run by
the program's section checks.

Known misses, kept: a key written as plain hexadecimal with no name before it (commit ids and hashes
look the same, so they are left alone); tokens made of short parts joined by dots that do not start
with `eyJ`; a password of several words gives up only its first word; short PINs. When in doubt,
clear the kept prompt text or switch `prompts` off.

### Switching it off and removing everything

- Set the control `memory` to `off`: the hooks still run but record nothing.
- Set `prompts` to `off`: no prompt text is kept.
- `neuromorph-nano setup --project <repo> --remove` takes the hook entries (and the MCP entry) out of
  the repository's Claude Code files, leaving every other entry as it was.
- Delete the repository's folder under `memory/projects/` to remove everything the Nano kept about
  it, the situation record, prompt text and evidence ledger included.

The situation record itself has no time limit yet; it holds no contents and grows by a few hundred
bytes per event.

### Session state

`neuromorph-nano session --project <repo>` and `/api/session` show one Claude Code session as the
situation record describes it: the task (your newest prompt, its text only while `prompts` is
`keep` and within `prompts.days`), the files read and edited with their lines, the commands and
checks, what is still open, and what the Nano served. It is computed when you ask for it; nothing
of it is stored apart from the situation record. Like every read of the interface, a tablet paired
in `view` or `control` mode can read it, prompt text included.

## Evidence ledger

The evidence ledger is how the Nano finds out whether a context pack it served helped. For each
pack it records what followed in the same Claude Code session: whether the places it cited were
then read or edited, whether a following build or test passed or failed, whether an edit was
undone, whether your next prompt corrected course or accepted the result, and whether the same
question was asked again. Each finding names the rule that made it and how strong it is.

- Where: `evidence.jsonl` in the repository's memory folder, on this machine only. It is written
  when a Claude Code session stops, starts or ends (by the hook program) and by
  `neuromorph-nano ledger --project <repo>`.
- What a row keeps: the pack's audit id, the session id, the places the pack cited (paths and line
  ranges), the parts that ran, the settings in effect, a 16-hex hash of the question, and for each
  finding the event's time, kind, tool, path and lines, check class (for example `cargo-test`),
  pass or fail, exit code and test counts, and this machine's hardware.
- Never kept: file contents, command text or output, prompt text, the question's text (only its
  hash). The rule that reads your next prompt looks at its kept text in memory, and only when `prompts` is `keep`;
  the ledger keeps only which fixed marker matched (for example `thanks` or `still failing`). With
  `prompts` off that rule records `unknown`.
- Memory off: nothing is written to the ledger (the hooks record nothing either); the command and
  the interface still read what is there.
- Each line holds the hash of the line before it, so a changed or removed line is found; the
  command then refuses to write and says where.
- To remove it: `neuromorph-nano ledger --project <repo> --clear` (or POST `/api/ledger/clear`
  from the interface). The removal is logged in `memory/audit.jsonl` with the row count, and packs
  served before it are not written to the ledger again from the records that remain. Deleting the
  repository's folder under `memory/projects/` removes it too.
- Primes (below) are judged the same way: a `primed` row (kind, hook, the places it cited, the
  memory it came from, the `priming` setting) and, when its window closes, one row per rule plus a
  `followed` row saying whether a primed place was then read or edited, or the prime was passed
  over.
- Outcomes: when a pack's findings verify or refute it, or a failed build or test is fixed by an
  edit and then passes, one `outcome` row keeps what memory will learn: the kind (`answered` or
  `fixed`), verified or refuted, the strength and the rule that decided it, the places (paths, line
  ranges and a hash of those lines, read once in memory), and the hash of the question asked of
  the Nano (`answered`; the memory takes the question's text from the situation record) or the
  failing check with its first error's code, file, line and message (`fixed`, the message after
  secret-dropping, as the situation record keeps it).

## Priming

A prime is a short note the Nano adds to Claude Code's context, unasked, through a hook: at a
prompt, the places that answered a similar verified question before; after a failed build or test,
the change that fixed the same failure last time; before an edit, a verified fix in those lines, or
the callers of a function whose signature the edit changes; after the conversation is compacted,
what was in play. Each names where it came from and is read fresh first; a note whose lines changed
is not given. The control `priming` (`off` the default, `quiet`, `on`) sets which: all priming and all 6 built-in prime rules start `off` (opt-in); `quiet` gives notes only from strong evidence, and `on` also gives notes from weaker evidence, at session start, and at stop. Custom primes (event, interval, and UTC schedule triggers) configured in Primes Studio, `/api/primes`, or MCP `nm_settings` also start `off` until explicitly switched on. There is no fixed count per session; built-in notes fire only when their verified rule matches, and never twice.

- The Nano sends nothing itself: a prime goes to Claude Code on standard output of the hook program,
  as context Claude Code then sends to its model with your conversation, like the files it reads.
- Kept: on the hook's own record in `situation.jsonl`, the prime's kind, id, places and sources
  (never its text); in `primes/<session>.jsonl`, the id, kind and size of each prime given in that
  session; in the ledger, as above.
- With memory off nothing is recorded and no prime is given. With `priming` off none is given;
  memory still keeps verified outcomes.
- The prompt is read in memory to match it against earlier questions; it is kept only as the
  `prompts` control says.
- To remove: the ledger's `--clear` removes its rows; delete `primes/` in the repository's memory
  folder for the per-session lists; deleting the repository's folder under `memory/projects/`
  removes everything.

## Outcomes in memory

The memory system takes the ledger's `outcome` rows into its journal, as memories of what really
happened: which places answered which question, and which change fixed which failure. A memory
keeps its evidence (each verification and refutation, with the rule and strength) and the places as
paths, line ranges and hashes; never the lines themselves. A refuted one is kept and shown as
refuted. When it is recalled, the lines are read fresh and compared with the hash. Memory changes no
model weights and no settings. An answered memory keeps the question asked of the Nano (as the
Nano's memory keeps the question of each answer it gives), never your prompt's text; a question no
longer in the situation record is not taken in.

- When: a serving process takes them in when it opens the repository's memory; `neuromorph-nano
  outcomes --project <repo> --import` does it at once (refused while a server holds the memory).
  Until then primes read them from the ledger directly.
- See them: `neuromorph-nano outcomes --project <repo>` (`--recall "<question>"` lists what a
  question recalls), and `/api/outcomes`.
- With memory off nothing is taken in. Like every memory, they are removed by deleting the
  repository's folder under `memory/projects/`; the memory's own undo and pin apply to them.

## Training store

The training store keeps what future cartridges can be fitted from: for each context pack that
something then verified or refuted, every candidate the Nano judged, what it served, and what
happened next. It belongs to learning, not to memory: memory recalls what happened, the training
store is data for fitting. Fitting itself is a separate step; keeping examples changes nothing the
Nano does.

The control `examples` sets what is kept:

- `off`: nothing. The serving process and the hooks write nothing to the store.
- `features` (the default): no text of your questions or your files. Per served pack: a 16-hex hash
  of the question, every candidate the judge read (its path and line range, a 16-hex hash of those
  lines, whether it was served and in which place, the reranker's score), the label the evidence gave
  each (`verified`, `refuted`, `used`, `unused`, `passed-over`, `went` for the place Claude Code
  went to instead, `judged`), the verdict and its strength, the settings, the cartridges and
  stacks in use from the cartridge library, the path taken (plain, fast, full), this machine's hardware, the
  time and the project. Preference signals: an edit kept or reverted (path), a correcting or
  accepting next prompt and a stated rule (which fixed marker matched and the prompt's hash, read
  from kept prompt text only when `prompts` is `keep`), a refused tool call (tool and path). Timings
  per hardware piece and setting, aggregated from the audit rows.
- `full`: as `features`, plus the question's text and each judged passage's text, after the same
  secret-dropping as everywhere else (see Secret-dropping), so an example stays usable after the
  code changes. Passages are kept once per content hash; a stated rule or correction also keeps the
  prompt's kept text.

Where: `training/` in the repository's memory folder, on this machine only. Files:
`judged.jsonl` (judged sets waiting for their pack's window to close; removed once used or after
7 run-days unused), `examples.jsonl` (learning examples), `preferences.jsonl` (preference signals, kept
apart from learning examples), `hardware.json` (timings), `engines.json` (which model weights,
precision and arithmetic switches the scores came from, so scores from two engines are never
mixed), `passages.jsonl` (only at `full`), `views/` (reserved for the reranker's last-layer values,
one file per engine; nothing writes it yet), `seen.txt` (which packs and signals were handled).
The registry of reference material for fundamentals cartridges is `memory/training/references.json`
in the Nano's own folder: where the material is on this machine, its licence, its hash and what was
fitted from it, never a copy of the material.

- With memory off nothing is kept.
- Kept by lifetimes counted in days the Nano ran (run-days), never removed for size: judged sets
  (working data) go after 7 run-days, examples and preference signals (evidence) after 28, and a
  verified example is kept at any age. The same example (same question, places, content hashes and
  labels) is kept once. `neuromorph-nano keep --apply` applies the lifetimes.
- See it: `neuromorph-nano examples --project <repo>` (`--show <id>` one example), and
  `/api/examples`. A paired tablet can read it like every other read.
- Move it between your own machines: `--export <file>` writes one file (its hash in its first
  line), `--import <file>` reads it on the other machine; nothing is sent anywhere by the Nano.
- To remove it: `neuromorph-nano examples --project <repo> --clear` (or POST
  `/api/examples/clear`). The removal is logged in `memory/audit.jsonl` with the row count, and packs
  served before it are not written again. Switching `examples` to `off` stops new writes and
  deletes nothing. Deleting the repository's folder under `memory/projects/` removes it too.

## Meaning index on disk

With `--neural` (or `--dense`) the Nano embeds every piece of your material with the Embedder
MicroModel to find code by meaning. Embedding a whole repository takes minutes on the graphics
card, so the vectors are kept on disk and loaded at the next start; only pieces that changed are
embedded again.

- Where: `dense/` in the repository's memory folder (`memory/projects/<name>-<8 hex>/dense/`, or
  `memory/dense/` for the Nano's own folder): `vectors.bin`, further segment files
  (`seg-000001.bin`, ...) as it grows, and `skin.json`, which keeps recently removed rows for a
  while (by content hash and a 16-hex digest of the file path) so that an undo or a branch switch
  does not embed them again.
- What: per piece, a SHA-256 hash of the text that was embedded and its vector (1,024 numbers at
  16 bits). Each file's header names the embedding model's weight hash, its precision, the
  arithmetic switches in force and the recipe of the embedded text. No text of your material and
  no file or folder names are kept. A vector is derived from the text, so it says something about
  it (that is what it is for), but the text cannot be read back from the file.
- When the header does not match the running model, or the file is damaged or cut off, it is not
  used (the server's start line says why) and is replaced as the index is built again.
- Not kept with memory off (the `memory` control or `serve --memory off`): the index then lives in
  RAM only, as before.
- Set the control `dense_cache` to `off` to stop reading and writing it; nothing is deleted.
  To remove it, run `neuromorph-nano dense clear --project <repo>` (or POST `/api/dense/clear`
  from the interface); it removes every file of the index in `dense/`, says how much went and logs the removal
  in `memory/audit.jsonl`. The next start embeds everything again.

## Secrets

The memory write gate refuses text that looks like a credential, key, token or e-mail address, so
it is not saved: not as a note or temporary note, not in an episode's question or answer, and not
in the audit log. The gate's refusal log (`refused.jsonl` in the project's memory folder) keeps only
the time, the reason and a SHA-256 of the refused item's key; the refused text is never written to
disk.

## Memory off

With the `memory` control off, or `serve --memory off`, nothing is recalled and nothing is written
to memory: no episodes, no substrate cache, no meaning index, no idle writer notes, no training
examples, no ledger. Each tool call still gets an audit row, holding only the tool, project, mode,
the client's name, the session, the time, the stages and the outcome, never the question.

## Checking it yourself

- Open the Audit page, or read `memory/audit.jsonl`, to see every call and change, each download
  and each change of a cartridge source included.
- Open the Storage page to see every file the Nano keeps, with its path and size.
- Read `GET /api/airgap` (or the line under the interface's page list) for the network state as it
  is now; switch the air gap on to refuse every download.
- Watch the Nano's network activity with your operating system's tools (for example Resource
  Monitor on Windows) while it runs.
