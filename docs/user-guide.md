# User guide

How to install Neuromorph Nano v0.1.01, connect it to a repository and a coding agent, and use the
interface. v0.1.01 ships for Windows x64; the MicroModels run on an NVIDIA graphics card when one
the kernels support is present, and on the processor otherwise.

## Install

### With the installer

```powershell
irm https://raw.githubusercontent.com/gavriel-tech/neuromorph-nano/main/tools/install.ps1 | iex
```

The installer downloads the release zip and `SHA256SUMS.txt` from GitHub Releases, checks the
zip's SHA-256, installs into `%LOCALAPPDATA%\NeuromorphNano` and adds `bin\` to your user `PATH`.
It then fetches the four serving base cartridges from the official public cartridge library
(`gavriel-tech/neuromorph-nano-cartridges`) plus the small tokenizer and config JSONs (about 44 MB) so Neuromorph Nano runs standalone immediately. Pass `-WithOpenWeights` (or use **Models -> Downloads** in the interface) if you also want the optional upstream Qwen3 open weights (about 8 GB) for fallback, reference checks, parameter mapping, or surgery. Every file is checked against its SHA-256 before it
is placed, a file already present with the right hash is kept, and a partial download resumes.

On a machine without network access, copy the release zip, `SHA256SUMS.txt`, the models and the
cartridges over once, then run:

```powershell
.\tools\install.ps1 -LocalZip .\neuromorph-nano-v0.1.01-win-x64.zip -ModelsFrom E:\nano\weights -CartridgesFrom E:\nano\library
```

An upgrade keeps `weights\` and `library\` (your models, cartridges and learned cartridges). New
base cartridge versions are added to the cartridge library; the control `library.upgrade` decides whether
the older versions are kept beside them (the default) or removed.

### From source

1. Install the requirements listed in `README.md` (Rust through rustup, the pinned version
   installs itself from `rust-toolchain.toml`; the MSVC build tools; Node.js 20.9 or later for the
   interface).
2. Build the engine and the interface:
   ```
   cargo build --release
   cd interface
   npm ci
   npm run build
   cd ..
   ```
3. A source build does not download models or cartridges. Get them from the interface (Models
   page, side panel, Downloads: the released models and the base cartridges, each on its own), or
   run the installer. The models go in `weights\`, one folder per model, each with its card
   `part.json` and its files in `files\`:
   ```
   weights\qwen3-0.6b\
   weights\qwen3-1.7b\
   weights\qwen3-reranker-0.6b\
   weights\qwen3-embedding-0.6b\
   ```
   The files must be exactly as released; the engine checks each model's main file against the
   SHA-256 on its card. To keep the models elsewhere, see the models folder in
   `docs/configuration.md`.

The commands below are written for a source build (`target\release\neuromorph-nano.exe`). After
the installer, run `neuromorph-nano` from any terminal instead.

## The MicroModels

The Nano runs four MicroModels, each named by its role:

| MicroModel | What it does | Upstream open weights it is built from |
|---|---|---|
| Reader | Reads the material for the drones and tentacles, judges and asks | Qwen3 0.6B (`qwen3-0.6b`) |
| Writer | Writes notes about the material at idle (the world model) | Qwen3 1.7B (`qwen3-1.7b`) |
| Embedder | The meaning index and recall by meaning | Qwen3 Embedding 0.6B (`qwen3-embedding-0.6b`) |
| Reranker | Orders the candidates, the deciding piece first | Qwen3 Reranker 0.6B (`qwen3-reranker-0.6b`) |

Each MicroModel runs directly on its Neuromorph base cartridge (about 6.5 GB total across all four
MicroModels) plus the small tokenizer and configuration JSONs (about 44 MB total). The full upstream
Qwen3 open weights (about 8 GB, Apache-2.0, `THIRD_PARTY_NOTICES.md`) are **optional** and never altered:
install them only when you want **fallback** (open mode, serving a MicroModel on its open weights
instead of a base cartridge), **output-parity reference checks** when seating base cartridges,
**parameter mapping**, or **cartridge surgery**. When open weights are installed, seating verifies
each base cartridge against them and releases the open weights from memory.

**Loading at start.** A serving process (`serve`, the MCP server a coding agent starts) loads all
four MicroModels in the background at start: the Reader, the Embedder and the Reranker, then the
Writer beside them when the card's memory holds it (`limits.vram_mb`; `auto` uses the driver's
figure). The tool list and the handshake answer at once; a call that needs a MicroModel still
loading waits for that MicroModel only. When the card cannot hold all four, the Writer loads per
idle session (the others are released for its session and restored after) and stays seated once
room frees. `nm_status` (`micromodels`) and the Live view (`/api/live`, `micromodels`) give each
MicroModel's state (`loading`, `held`, `per idle session`, `not loaded`), its device and card
memory, and the reason, with `held` as "N of 4". The Writer still gives way to any call within a
token.

**Two sessions on one project.** Two coding-agent sessions on the same project each start a
`serve`. The first holds the engine and shares it on a loopback port (`127.0.0.1`, registered in
`scratch/engines/` in the Nano folder); the second attaches to it and forwards its client's calls,
so both are served by one engine with one set of MicroModels on the card, one call at a time. The
engine stays up while any session is attached. `nm_status` (`engine`) says whether this client
holds the engine or is attached, and how many clients are attached. A `serve` with other options
(`--whole`, `--memory-dir`, a different folder) holds its own engine.

**Memory.** The caches a serving process keeps (the encoded chunks, the relevance keys) grow only
while the machine keeps a quarter of its memory available (at least 4 GiB) and the process stays
within its budget (`limits.ram_mb`, or three eighths of the machine's memory on `auto`); past either
they give entries back, least recently used first. `nm_status` (`host_memory`) gives the figures.

**Memory by area.** The process budget is divided into three areas, each `auto` or a size you set
on the Settings page (Machine, under the system memory limit), through `nm_settings`
(`op: set_limit`, `key: content_mb`) or `POST /api/limits`: `limits.memory.content_mb` (the
project's content: files and pieces read, the drones' intake, the relevance keys, the dense
lane's vectors; a quarter on `auto`), `limits.memory.caches_mb` (the kept encoded-chunk
cache; 15 percent) and `limits.memory.weights_mb` (the MicroModels' host weight copies; 45
percent); the last 15 percent stays with the engine's own working memory.
Nothing is left out by folder name. The drones' intake, the relevance-key intake and the dense
lane run in bounded steps within the content area: a call does not wait for a whole-folder
pass, it takes in the pieces its question likely needs and reads the rest as their paths, and
the passes go on between calls. When the content area is full the passes stop and `nm_status`
(`memory_areas`) says so; raising the area lets them continue.

## Connect a repository

One command connects a repository to one coding agent. Any other MCP client (including agents
that run a local model) and models without MCP are covered in [Agents and models](agents.md).

```
target\release\neuromorph-nano.exe connect claude --project C:\path\to\your\repo --dry-run
target\release\neuromorph-nano.exe connect claude --project C:\path\to\your\repo
```

The client is one of `claude` (Claude Code), `antigravity`, `cursor`, `vscode`, `windsurf` or
`generic` (any other client over standard input and output: the command line and JSON are printed
for you to paste). `--dry-run` prints the exact difference the write would make and writes nothing.
Every write merges the Nano's one server entry into the client's file and keeps every other server
and setting as it was; a copy of the file as it was is saved in the project's memory folder; the
write is logged in the audit. `--hooks` also writes the Claude Code hooks (below), `--plain` the
plain path, and `--remove` takes out exactly the entry that was written.

For Claude Code, `setup` does the same step by step:

```
target\release\neuromorph-nano.exe setup --project C:\path\to\your\repo
```

This prints the MCP entry and the `.mcp.json` it would write in the repository, and the memory
folder the repository will use. Nothing is written. To write it:

```
target\release\neuromorph-nano.exe setup --project C:\path\to\your\repo --write
```

`--write` merges a `neuromorph-nano` server into the repository's `.mcp.json` (other servers are
kept), approves that server in the repository's personal settings file (`enabledMcpjsonServers` in
`.claude\settings.local.json`, other entries kept), copies the launcher to
`bin\neuromorph-nano.exe`, and logs the write in the audit log. The entry runs
`bin\neuromorph-nano.exe launch serve --root <repo> --whole --neural`: the whole system (drones,
tentacles, the GCX, the Core Brain's MicroModels and the dense lane; it needs the models and base
cartridges in place). The launcher starts a copy of the newest build, so you can rebuild while a
session is open and the next session uses the new build. `setup --project <repo> --plain` writes
the plain path instead (no models, no graphics card needed). `setup --project <repo> --antigravity`
writes the Antigravity project plugin in `.agents\plugins\neuromorph-nano\`.

Then open Claude Code in that repository, either `claude` in a terminal in the repository folder
or the repository in the desktop app's Code tab. On the first open of a folder Claude Code asks
whether you trust it; after that the approved `neuromorph-nano` server starts by itself, also
under `claude -p`. The server answers at once and reads the material in the background; the first
`nm_context_pack` waits for that reading. The agent now has the `nm_*` tools: 19 listed by
default, all 25 callable (`docs/mcp-tools.md`).

`.mcp.json` holds absolute paths of this machine (the launcher and the repository). It shows up as
an untracked file in git; leave it out of commits (add it to `.gitignore`) unless everyone who uses
the repository has the same paths.

Each repository gets its own memory folder, `memory\projects\<name>-<8 hex>\`.

`--write` keeps every other entry of `.mcp.json` exactly as it was (comments included), and first
saves a copy of the file as it was in the project's memory folder, under `setup\backups\`.

## Let the Nano see Claude Code's work (hooks)

Over MCP the Nano only sees the questions the agent asks it. With hooks it also sees what Claude
Code then does: each prompt, each file read or edited, each command and whether it passed, each
stop and each shortening of the conversation. That record is what later learning is built on.

```
target\release\neuromorph-nano.exe setup --project C:\path\to\your\repo --hooks
```

This prints the hook entries and the file as it would be. Nothing is written. To write them:

```
target\release\neuromorph-nano.exe setup --project C:\path\to\your\repo --hooks --write
```

- Only the repository's personal settings file, `.claude\settings.local.json`, is written; never
  the shared `.claude\settings.json` and never your user settings. Setup refuses to write when the
  repository is your home folder.
- Every other entry in the file stays byte for byte as it was, comments included. A copy of the
  file as it was is saved first in the project's memory folder (`setup\backups\`), and the write
  is logged in the audit.
- Running it again changes nothing.
- A file that is not valid JSON (comments and trailing commas are fine) is refused with the reason
  and left as it is.

Each entry runs `bin\neuromorph-nano.exe hook <Event>` for one of: `SessionStart`,
`UserPromptSubmit`, `PreToolUse` (edits), `PostToolUse` (shell commands and the Nano's own tools;
reads, searches and edits in the background), `PostToolUseFailure`, `Stop`, `PreCompact` and
`SessionEnd` (so a session that ends before any stop still closes in the record). The hook reads
the event from Claude Code, appends one short record to the project's situation record and exits.
It always exits with success and never blocks Claude Code, with one deliberate exception: a loop
you or the agent set (below), which continues the agent at a stop. A fault is logged in the Nano's
audit, not shown to Claude Code. It prints nothing, except a prime and the reminders and steps
you or the agent set (below). What is kept:
`docs\privacy.md`, Hooks.

Claude Code asks you to trust a folder before its hooks run, and `/hooks` in Claude Code lists them.

See what was recorded:

```
target\release\neuromorph-nano.exe situation --project C:\path\to\your\repo [--session <id>] [--last <n>] [--json]
```

See where a session stands (task, files, checks, what is open, what the Nano served):

```
target\release\neuromorph-nano.exe session --project C:\path\to\your\repo [--session <id>]
```

See whether the context the Nano served helped: for each context pack, the evidence ledger records
what followed in the same session (the cited places read or edited, a following build or test
passing or failing, an edit undone, your next prompt correcting course or accepting, the question
asked again), each finding with its rule and strength. The hooks keep it up to date at every stop;
this command does too and prints it:

```
target\release\neuromorph-nano.exe ledger --project C:\path\to\your\repo [--pack <id>] [--file <path>]
```

What it keeps and how to clear it: `docs\privacy.md`, Evidence ledger.

### Priming

With hooks on, the Nano can add a short note, a prime, to Claude Code's context at the moment it
is likely to help, without being asked. Each note states located facts and where they came from:

- at a prompt: the places that answered a similar question before, when that answer was verified;
- after a failed build or test: the change that fixed the same failure last time;
- before an edit: that the lines hold a verified fix, or that the edit changes a function's
  signature and where the function is called;
- after Claude Code compacts its conversation: what was in play (the task, files edited, failing
  checks, changes not yet verified);
- with the setting at `on`, also at a session start (what the last session left open) and at a stop
  (files changed and not built or tested since).

The source of each note is read fresh before it is given; if the cited lines changed, nothing is
said. The same note is never given twice in a session. The setting `priming` sets which: `quiet`
(the default, strong evidence only), `on` (weaker evidence too, and at a session start and a stop)
or `off`. There is no count per session. Change it in the interface's settings, where it is
audited and can be undone. Each note given is judged in the evidence ledger (whether the place was
then read or edited, whether the next check passed), so the ledger shows which kinds help.

### Reminders, loops and steps

Beside the primes, which the Nano chooses, you and your coding agent can set guidance of your own
for a project. It is delivered through the same hooks, so it needs `setup --hooks` and memory on.
Set it on the Memory page (the Loops and graph tab, `/memory/#loops`; the side panel's Guidance tab
explains each part), from the command line (`neuromorph-nano guidance`), over MCP (`nm_goal` with an `op`,
which is how the agent sets its own) or through the API (`/api/guidance`). All four run the same
route and keep the same file, `guidance.json` in the project's memory folder.

**Reminders.** A note added to Claude Code's context at its trigger, whole and prefixed
`Neuromorph Nano reminder (set by the user): ` or `(set by the agent): `.

| Setting | Values |
|---|---|
| Trigger | a session start; every prompt or every Nth prompt; after a compaction; at a stop; before an edit whose path matches a glob; after a failed build or test; after every Nth tool call; `error-repeats`: when the same recorded error comes back in 2 to 5 of the session's tool results |
| Repetition | every time, once per session, or at most N times per session (a reset counts again from then) |
| File scope | optional glob: the reminder applies only when the event's files, or a file the session already touched, match |
| Switch | on or off, without removing it |

A reminder may share one `additionalContext` with a prime. It does not count against the prime's
never-twice rule and is never shortened.

An `error-repeats` reminder names an error as the hook records it: a located error's code (for
example `E0308`, `panic`) or a piece of its message, or a piece of a failed tool's message (case is
ignored). Each tool result of the session that carries it counts once; when the count reaches the
reminder's `count` (2 to 5, default 2) and the current result carries the error, the reminder is
delivered and the count starts again. Its repetition setting applies as for any reminder.

**Loops.** A loop keeps the agent working at a stop until its condition holds:

| Condition | Met when |
|---|---|
| `check-passes` | in that session, the last build or test after the last edit passed (records from before the loop was set do not count) |
| `goal-done` | the named goal is closed (done or dropped) |
| `steps-done` | every step (of the named goal, when one is named) is done |
| `manual` | never by itself: it runs until you or the agent end it |
| `all-of` | every one of its `parts` (2 to 4 distinct conditions above) is met |
| `any-of` | at least one of its `parts` is met |

For `all-of` the unmet reason names each part not yet met with its own reason; for `any-of` it names
every part. A `goal-done` part needs `goal`, as the condition alone does.

Each loop also ends early when it makes no progress or uses its budget:

| Field | Values | Ends the loop as |
|---|---|---|
| `stall` | 1 to 10, default 2 | `stalled`: the same unmet reason came back this many times in a row (with 2, at three stops running). A `manual` loop has no progress reason and does not stall |
| `budget_ms` | 1000 to 86400000, optional | `budget-spent`: that much wall time passed since the loop was set or restarted |
| `budget_tools` | 1 to 10000, optional | `budget-spent`: that many tool calls completed in the project since the loop was set or restarted |

These are checked at each stop beside `max`, after the condition (a met condition always ends the
loop as done). A budget given as 0 or null is cleared.

At each stop while the condition is unmet and the loop is under its maximum (3 by default, at
most 20), the Stop hook answers `{"decision":"block","reason":"..."}` with the loop's prompt and
what is unmet. That is Claude Code's documented way to make the agent continue, and the loop's
iteration count rises by one. When the condition holds the loop is marked done; at its maximum it
is marked exhausted. Whatever the end, a final note, which does not block, says so once and names
the reason (also shown to you as a system message). End a loop at any time, switch it off, or
restart it (a restart clears its unmet reasons and starts its budgets again).

**Ended loops.** Every end of a loop (`met`, `cap`, `stalled`, `budget-spent`, `ended` by hand, or
`removed` while active) appends one row to `guidance-outcomes.jsonl` beside `guidance.json`: the
loop's id, who set it, its prompt and condition (`until`, `parts`, `goal`), iterations and maximum,
its stall threshold, its budgets and the budget used (`used_ms`, `used_tools`), the end and its
note, the distinct unmet reasons in the order they came, the verdict the evidence ledger's
`loop-met` rule gives its continuations (`verified` when met, `refuted` at the cap, a stall or a
spent budget, `unknown` when ended by hand or removed), and its start and end times. The Guidance
section lists the newest under Ended loops. The rows are kept as evidence for Learning, which is to
read them to judge which loop settings worked for a project; in v0.1.0 Learning does not read them
yet.

> **The one exception to "hooks never block".** Everywhere else the Nano's hooks only record and
> add context; they never stop, deny or hold up Claude Code. A loop is the single deliberate
> exception, and it only ever continues the agent, never stops it. It acts only when you or the
> agent set a loop and left it on, it is capped (at most 20 continuations, 3 by default), and the
> `loops` setting switches every loop off at once.

Antigravity: reminders reach it at a session start and at prompts, the two events that accept
injected steps. It has no documented stop continuation the Nano relies on, so loops do not act
there; each stop at which a loop would have acted is recorded, and the Guidance section counts
them.

**Steps.** A step graph of steps with dependencies, standalone or attached to a goal. A step is
ready when every step it waits for is done, blocked otherwise; a dependency that would make a
cycle is refused. With reminders on, the next ready step is given at prompts and at stops, once per
step per session unless the step changes. A step is marked done by you or the agent, or by the
Nano: a step with a check (`build` or `test`) and optional path globs is marked done when that
check passes after an edit under its globs, and the evidence (the check, the command, the edited
files, the session) is kept on the step.

A step waits on two lists. `depends_on` is an AND-join: every step in it must be done.
`depends_any` is an optional OR-join: at least one step in it must be done as well. A step id may
be in one list only, and the cycle check covers both lists. A probe step (`probe: true`) must have a
check. It is marked done when a check of that kind passes while its dependencies are met, with no
edit needed, for work that may already hold (for example, "the tests already pass on this branch").
Its dependencies must be met before that check: a check that marks a dependency done does not
also mark the probe. In the Guidance section an OR edge is dashed with a diamond head, and a probe
step has round ends.

Every change, including the Nano's own (a step it marks done, a loop it ends), is an audit row the
Audit page undoes. Removing an item shows a preview and asks for its id to be typed. Everything
sent to the agent is named on the hook's record in `situation.jsonl`, where the evidence ledger
reads it, and kept with its text in `guidance-deliveries.jsonl`, which the Guidance section lists.
The settings `reminders` and `loops` (both `on` by default) switch delivery off without removing
anything.

**Verdicts.** The evidence ledger judges each delivery the way it judges a prime: a `guided` row is
written when the delivery is seen, and one evidence row by the rule of its kind once its window,
the rest of its session, closes (the session ends or is cleared, an hour passes with no activity,
or a day after the delivery). Until then the verdict is shown as open.

| Delivery | Rule | Verdict |
|---|---|---|
| Step nudge | `step-followed` | `used` when the step is marked done later in the session (by the Nano after its check passed, or by you or the agent: `direct`) or when its check passes later in the session (`indirect` after an edit under the step's paths, `weak` otherwise); `passed-over` when the window closes without that |
| Loop continuation | `loop-met` | `verified` when the loop's condition was met before its cap, `refuted` when it reached its cap first (both `direct`); `unknown` when it was ended by hand or the session ended before it was judged again |
| Reminder tied to an edit glob | `glob-check` | `verified` when the last build or test after the last edit under the glob (the `edit` trigger's glob, else the file scope) passed, `refuted` when it failed (both `indirect`); a reminder with no glob is recorded as not judged |

The Guidance section shows each delivery with its verdict and counts them (held, missed, open, not
judged); `GET /api/guidance` gives the same per delivery, `GET /api/ledger` gives the guided rows,
their evidence rows and a `guidance` block, and over MCP `nm_goal` with `op: "guidance"` and
`nm_learning` with `op: "ledger"` return those two answers.

```
neuromorph-nano guidance --project C:\path\to\repo add reminder --text "run the full tests before you finish" --trigger stop --repeat once
neuromorph-nano guidance --project C:\path\to\repo add loop --prompt "Fix what fails and run the tests again." --until check-passes --max 3
neuromorph-nano guidance --project C:\path\to\repo add step --title "write the parser" --id parse
neuromorph-nano guidance --project C:\path\to\repo add step --title "test the parser" --after parse --check test --paths "src/**"
neuromorph-nano guidance --project C:\path\to\repo
```

### What memory learns from the ledger

Verified outcomes become memories: which places answered which question, and which change fixed
which failure, each with its evidence (verified or refuted, the rule and how strong). A refuted one
is kept and shown as refuted. The server takes them into memory when it opens a repository's
memory; to see them, or take them in now:

```
target\release\neuromorph-nano.exe outcomes --project C:\path\to\your\repo [--recall "<question>"] [--import]
```

### What is kept for fitting cartridges

As the ledger settles, the Nano also keeps training examples: for each context pack that was then
verified or refuted, every candidate it judged, what it served, and what happened next (including
where Claude Code went instead when the Nano was wrong), with the settings, cartridges and hardware
in use. Preference signals (edits kept or reverted, corrections, stated rules, refusals) and timings
per hardware piece are kept in their own files. The learning cycle fits from this
(`docs/continuous-learning.md`). The setting `examples` chooses: `features` (the default, no text
of your questions or code), `full` (also the text, secrets dropped) or `off`.

```
target\release\neuromorph-nano.exe examples --project C:\path\to\your\repo [--show <id>] [--export <file>] [--clear]
```

What each position keeps: `docs\privacy.md`, Training store.

See what is installed: `setup --project <repo> --status`.

## Disconnect a repository

```
target\release\neuromorph-nano.exe setup --project C:\path\to\your\repo --remove
```

This takes out the Nano's MCP entry from `.mcp.json`, its approval and hook entries from
`.claude\settings.local.json`, and nothing else, and says what it removed. A file that setup
created and that holds nothing else is removed again (and the `.claude` folder, when setup created
it and it is empty). A file you had before is left exactly as it was before setup wrote to it. A
copy of each file as it was before the removal is kept in `setup\backups\`. Hand-written hooks
that run the Nano (for example your own `hook stop` gate) are not setup's and are left in place.
For another client, `connect <client> --project <repo> --remove` takes out the entry `connect`
wrote.

To remove everything the Nano kept about a repository as well, forget the project on the Storage
page (a preview first, then a confirmation), or delete its folder under `memory\projects\` (the
`--status` output names it).

## The interface

```
target\release\neuromorph-nano.exe launch ui
```

Open http://127.0.0.1:4870/. Use `--port <n>` for another port. If the page says the interface is
not built, run `npm run build` in `interface\`. `launch` runs the server from a copy of the newest
build in `scratch\live\`, so the server never holds `target\release` open and a rebuild is never
blocked; `neuromorph-nano ui` starts it directly.

Changes are sent with the server's local token, which the server places in each page it serves; a
page from anywhere else cannot change anything.

Every page has the same frame: the page list on the left; the working system in the middle (its
live visuals and controls); and a side panel on the right with the page's collections and
references in tabs, the detail of whatever you pick in the middle, and an About tab saying what
the page is and how it works. The side panel folds away (remembered per page); on a narrow window
it slides over from the right. Figures come from the engine; a figure the engine does not report is
shown as not reported.

Under the page list, on every page, the system strip gives the engine's state (serving, or no
engine serving) and its readings: the graphics card's memory, RAM free, the MicroModels loaded, the
calls in the last hour, the idle Writer and the disk free against the disk floor. A reading the
engine does not report says not measured. Live Activity (its System tab), Topology, Anatomy and
Models (their Live tab) carry the System panel in the side panel: the card's memory by MicroModel,
each MicroModel's forward passes and time since the engine started, the idle Writer in full,
background work with a measurable end, the tokens the recent calls sent back against their budget
and the tokens of each request (the material read to answer is not measured), this machine's
capacity and the engine's recent events. A guidance delivery to your agent (a reminder, a loop
continuation, a step nudge) shows as "guidance delivered" in the strip, the System panel and the
timeline; the organism has no part for it, so nothing lights. A backup shows the same way.

The live view (Live Activity and Topology) lights only the parts a call names: its stages, the
MicroModels it ran and the parts its events report; a call that names no component lights no
part and says so. A past call played again from the timeline is marked replay. History is not
replayed on arrival, and with no engine serving the organism is drawn at rest, lit only by the
interface's own work (a lab build, a backup).

| Group | Page | In the middle | Side panel tabs |
|---|---|---|---|
| Observe | Live Activity (`/`) | The organism drawn from its anatomy, lit by each call as it runs; every coding agent's calls by client name with their real figures; the engine now; stage timings; the path of the last call; the session the hooks record | Activity, System, MicroModels, Cartridges, Set up |
| Observe | Topology | The organism full screen: every system, its components and the connections between them; click a part to open it, zoom and pan | Systems, Last call, Live |
| Observe | Anatomy | One floorplan of the systems and their components, and every engine crate with the system that holds it | Crates, Systems, Live |
| Engine | Models | Each MicroModel (Reader, Writer, Embedder, Reranker) drawn as its layers, as bright as the real calls that ran it over the last 24 hours, with what it runs on: its base cartridge, the upstream open weights it was built from, and their download state; cartridges, job filters and preference cartridges dock beside the layers they touch; precision and base cartridge choice | Live, Downloads, Libraries, Routing, Lenses |
| Engine | Cartridges | One rack holding a deck per MicroModel: its base cartridge seated, its cartridges docked in the order they apply (compensation, job filters, adapters and lenses, preference); plug, unplug and choose a version; the whole-engine cartridges and stacks | Library, Bases, Build, Sources (the cartridge libraries: the official one and third-party ones on Hugging Face), Learned, Specs, Components (how the parts fit, the kinds of cartridges and the glossary) |
| Engine | Parameters | The map atlas: a lane per MicroModel with its job filters as switches, the heat of its layers, its parameter maps as tiles by kind and state, and the cartridges they feed; lineage lit both ways; pick several for one archive, restore or delete; a list view. Below it, the cartridge parameters: every object holding learned parameters, as a chain per MicroModel (open weights, base cartridge, filters, adapters and units) against the layer strip; the `cartridges` view shows that chain alone | Build, Import, Archive, Kinds; Detail (a map with its lineage editor, or an object's parameters with strength, layer switches and Apply) |
| Engine | Memory | Three tabs: Memory (what is held per tier, the tiers as strata with one column per project, recent calls), World model (one figure of the four levels the idle Writer writes, with its queue; below it the levels, the project outline and the idle Writer's state), Loops and graph (opens on the guidance circuit figure: reminders, loops, the step graph and what was sent to your coding agent; then loops, the step graph, notes at their triggers, what was sent and how it was judged) | Memories, Goals, Guidance, Outcomes, Routing, Files |
| Engine | Learning | Four tabs: Loop (the evidence gathering into a cartridge, the learning switches per project, then the six steps from verified work through the gate to units acting), Distillation (the path in five steps with the engine's counts: teacher data, fill, train, the held-out gate, cartridges; each step opens its view), Learned units, Preferences | Evidence, Primes, Training data, Cartridges, Preferences, Distillation |
| System | Storage | Two tabs: Disk (use by category with exact paths; prune rebuildable items; delete removable ones; forget a project) and Backups (your own repositories: add, back up, snapshots, preview and restore) | Settings, Kinds, Backups |
| System | Audit | A timeline of everything logged, filtered by all, ok, failed or undoable; an undoable row carries its Undo, including a base cartridge choice and a role model change (rows logged before v0.1.0 without their earlier state are not offered); trim the audit log | Stage times, Retention, Kinds |
| System | Settings | Every control, hardware limits and this machine's recommended values, agents and tools, privacy and network (air gap, tablets), stage times; the Settings file panel shows `settings.json` key by key, with each MicroModel's base cartridge (`base_cartridges`), each base cartridge's compensation (`base_compensation`) and each role's model (`role_models`) as a control | Guide, Design files |

The parameter map states on the Parameters page: `current` (serves, or builds the base cartridge
serving now), `stale` (made on an older base cartridge), `history` (built earlier versions; kept
for the record and for rebuilding), `experiment` (sweeps and comparisons nothing reads), `log`
(the build's log) and `origin not recorded` (maps that do not say what they were measured on). An
archived map can be restored; deleting goes through a preview and a confirmation.

A map's lineage is recorded or corrected with **lineage**, on the map's row or in its detail in the
side panel: made on (the cartridge or map it was measured on), made by, purpose (serving,
base-build, job-filter, probe, measurement, experiment), outcome (used, superseded, failed,
duplicate, control, unused), the reason for the outcome (required for failed, duplicate and
control) and a note. The route has no preview; the form says what the write does. A map labelled
with an outcome other than used or superseded moves into its archive in the same step, and a map in
use is refused. The record is marked as recorded by you, logged in Audit and undone there (API
`POST /api/parameter-maps/lineage`; MCP `nm_cartridges` `op: map_lineage`).

## Controls

The controls are on the Settings page (and beside what they change on other pages). The most
used:

| Control | Values | Effect |
|---|---|---|
| Base model precision | default, 16, 8, 4 | The Reader, with the drones and the judge |
| Writer precision | default, 16, 8, 4 | The Writer |
| Reranker precision | default, 16, 8, 4 | The Reranker |
| Embedder precision | default, 16, 8, 4 | The Embedder |
| Memory | on, off | Off: nothing recalled and nothing written to memory |
| Continuous learning | on, off | Off: no learning observations or adaptations |
| Answer mode | full, fast | Full: the whole path. Fast: seconds, the Reranker on the first few. Applies on the whole path, from the next call |
| Idle writer notes | on, off | On: the Writer writes notes about the material while the server is quiet |
| Air-gap mode | off, on | On: the Nano makes no network request at all; downloads and tablets are refused |

Every control, its values, its default and when it applies: `docs/configuration.md`. A
command-line flag wins over a saved control. Controls are saved in `settings.json`.

## Audit and undo

Everything the Nano does is written to `memory\audit.jsonl` and shown on the Audit page:

- every tool call: tool, project, the client that made it (the name the coding agent gave when it
  connected), question (first 200 characters), time, outcome, the files cited and the reply size;
- every setting change, with its old and new value;
- every cartridge plugged in or out, every base cartridge version chosen, every cartridge library, job
  filter and parameter map change, and every cartridge source added or removed;
- outcomes reported for an earlier answer (`nm_feedback`, or `nm_verify` given an episode);
- every MCP entry written by `setup --write` or `connect`, every hook entry written by
  `setup --hooks --write`, and every removal, each with the path of the backup copy;
- a fault of a hook (`kind: "hook-fault"`), which is never shown to Claude Code;
- every clearing of kept prompt text (the count only).

Undo sets a change back: a setting, a cartridge or learned unit switch, a cartridge library change, a job
filter, a parameter map edit, import, lineage, archive or restore, a memory pin, an objective, or a
cartridge source. The undo is itself logged. The full list is in `docs/interface-api.md`,
`/api/undo`.

The audit log's tool-call rows are kept for `audit.keep_days` days of use (28 by default) when you
trim it (Audit page, or `neuromorph-nano keep --apply`); every other row stays.

## Cartridges

A cartridge is a set of parameters or values that loads into a MicroModel, or into the engine, and
works there, like a cartridge plugged into a console: the engine has sockets (named places in a
model's layers, for example the Reranker's final norm) and does not change when a cartridge goes in
or out. The kinds:

- **Base cartridge**: the MicroModel's own parameters, mapped and compressed for its job from the
  upstream open weights. One serves per MicroModel; other versions wait in the cartridge library.
- **Compensation**: small gains and biases that add back what one base cartridge version leaves
  out; it fits only that version.
- **Job filter**: a filter over some of a MicroModel's layers for one job; plugs in or out from the
  next call.
- **Adapter and lens**: learned from one project's verified places; loads only for that project,
  on the base it was fitted on.
- **Preference**: your way of working, from edits you kept or reverted and rules you stated; it
  sits on top, so your way wins.
- **Engine cartridge**: operating numbers for a piece of hardware, or values for the whole engine;
  it sets values, not parameters.

Each cartridge is a file in `library/cartridges/` with a plain label: its name, what it does, which
MicroModel and where it loads, what it was made from and over which open weights, when, its size
and its measured effect. A cartridge without a label, or made for another model or other open
weights, is refused. Unplugging gives the base back exactly. A stack is the set of cartridges
plugged in for one use, applied in order; preference cartridges go last. Cartridges are never your
code: that changes and is read live.

On the Cartridges page each MicroModel has a deck: plug, unplug, or choose which base cartridge
version serves (or its open weights). Every change goes through the engine and is logged, with
undo. A job filter or a learned unit applies from the next call; a base version from the next
start.

Learned units (what the learning cycle keeps) are grouped by the compartment they were fitted in
(the Nano itself, shared, or one project) and load only there. Each unit has its own plug and each
compartment a switch for all its units (Cartridges page, Learned; MCP `nm_learning`
`op: unit_on|unit_off|compartment_on|compartment_off`).

### Stated preferences

A stated preference cartridge holds rules you state in plain words: what you want as well as what
to avoid. For example "no em dashes", "use 'use' instead of 'utilize'", "prefer 'colour'",
"strongly prefer concise replies", "indent with 4 spaces", "max line length 100", "snake_case for
functions", or "for .kicad_pcb files, keep power traces at least 0.5 mm wide". It is a `preference`
cartridge of the library like any other: versioned, plugged in or out, applied after every other
stack, and audited with undo. It holds rules, not weights; it may also carry learned preference
values, and the interface says so when it does.

The Nano reads each statement with a fixed parser, so the same words always give the same rules.
Each rule is held in one of four ways:

| Held as | What happens |
|---|---|
| enforced | Text replacements and vocabulary. The Nano applies them to the text it writes itself (the Writer's notes, summaries and outlines; the Writer is also kept from choosing a token that spells an avoided mark such as the em dash) and to the prose of a draft checked with `preference_apply`. |
| checked | Avoided words ("never say X", "avoid X"), indentation, line length and naming. Reported with the line on a draft; the text is left as it is. |
| biased | Wanted terms ("prefer 'colour'", "always use 'Result'"). The Writer's decoding is pushed toward the term by a bounded logit bias; your agent sees the term as `prefer: "<term>"`; a draft without the term is noted by `preference_apply` as information, never as a violation. |
| guidance | Wanted styles ("I like concise replies", listed to your agent as `prefer: concise`), and anything the parser does not read as one of the forms above, kept word for word. Passed to your agent in each `nm_context_pack` reply whose project and cited files fit its scope. A statement is never guessed into an enforced rule. |

#### Wants and their strength

A want names a term (a quoted word or phrase, or one plain word) or a style the parser knows:
concise, terse, formal, plain language, numbered lists, bulleted lists, active voice and short
sentences. A style it does not know, or several unquoted words ("prefer elegant code"), stays
guidance word for word.

Each want holds a strength from 0.1 to 2.0. The default is 0.5; "strongly", "really", "always" and
their equivalents give 1.0, and "slightly", "a bit" and their equivalents give 0.25. The strength
is editable per rule on the Learning page, Preferences (each edit is a new version, audited), with
`preference_edit` (`rule`, `strength`), or with `preferences edit <id> --rule rN --strength 1.5
--yes`.

In the Writer's decoding, a wanted term's next token gets an additive bias of the strength times
1.5 logits, capped at 3.0 per token (summed over every want). The first token of the term is
pushed only where the term can follow (its spelling with a leading space after a word, its bare
spelling at the start or after whitespace), and not again while the term was written within the
last 48 tokens; once the Writer has written the term's first tokens, the rest are pushed so the
term is finished. A token the model holds far below its best is never taken by the push. The mask
of an avoided word always wins: a masked token is never biased and never chosen. The push applies
to wants whose scope names no file kind or folder, since the Writer's own text has no path; a
scoped want reaches your agent and the draft check where its scope fits.

The parser reads these forms in English, Spanish, French, German and Portuguese:

| Form | English | Spanish | French | German | Portuguese |
|---|---|---|---|---|---|
| No X, remove X | no em dashes; remove smart quotes | sin rayas; elimina las comillas curvas | pas de tirets cadratins; supprime les guillemets typographiques | keine Geviertstriche; entferne alle Gedankenstriche | sem travessões; remova as aspas curvas |
| Use A instead of B | use 'use' instead of 'utilize'; replace em dashes with ' - ' | usa 'usar' en lugar de 'utilizar'; reemplaza las rayas por comas | utilise "utiliser" au lieu de "employer"; remplace les tirets cadratins par des virgules | verwende 'nutzen' statt 'verwenden'; ersetze Geviertstriche durch Kommas | use 'usar' em vez de 'utilizar'; substitua travessões por vírgulas |
| Never say X (avoid) | never say 'leverage' | nunca digas 'básicamente' | ne dis jamais "en fait" | sag niemals 'eigentlich' | nunca diga 'basicamente' |
| Want X | prefer 'colour'; always use 'Result'; I like concise replies; more numbered lists; be terse | prefiero respuestas concisas; usa siempre 'Resultado'; me gustan las listas numeradas | je préfère des réponses concises; utilise toujours 'courriel'; privilégie la voix active | ich bevorzuge kurze Sätze; verwende immer 'Ergebnis'; sei knapp | prefiro respostas concisas; use sempre 'resultado'; seja formal |
| Want strength | strongly prefer (1.0); slightly prefer (0.25) | prefiero realmente; prefiero ligeramente | préfère vraiment; préfère un peu | bevorzuge unbedingt; bevorzuge etwas | prefiro muito; prefira um pouco |
| Scope clause | for .rs files, ...; under docs/, ... | para archivos .rs, ...; en docs/ | pour les fichiers .md, ...; dans docs/ | für .kicad_pcb-Dateien ...; im Ordner docs/ | em arquivos .py, ...; na pasta docs/ |
| Indentation | indent with 4 spaces; use tabs | sangría de 4 espacios; usa tabuladores | indentation de 4 espaces | Einrückung mit 4 Leerzeichen; Tabs statt Leerzeichen | recuo de 2 espaços; use tabulações |
| Line length | max line length 100 | longitud máxima de línea 100 | longueur de ligne maximale de 100 caractères | maximale Zeilenlänge 100 Zeichen | comprimento máximo de linha 100 |
| Naming case | snake_case for functions | snake_case para las funciones | snake_case pour les fonctions | snake_case für Funktionen | snake_case para funções |

The other languages are read only when no English form reads a statement, so English statements
give exactly the rules they gave before. Accents are optional ("linea" reads as "línea"); the
characters these languages name (raya, tiret cadratin, Geviertstrich, travessão and the others)
map to the same rules as their English names, and German "Gedankenstrich" stands for both long
dashes. Statements joined by y, et, und or e are read part by part when every part is a form.
Anything else stays guidance, word for word, in whatever language it was said.

Want forms are read only for a statement no other form reads, and a message gives wants only when
it gave no rule without them, so every statement that gave rules before wants existed gives the
same rules. Scope clauses in the other languages (a file kind such as `.rs` or `.kicad_pcb-Dateien`,
a folder such as `docs/` or "la carpeta docs", or "este proyecto") are read only when no English
scope clause scopes the statement; a leading clause may end at a comma or at its file kind or
folder ("para archivos .rs indenta con 4 espacios").

The Writer's token mask and its want bias hold for every decode of the Writer that writes the
Nano's own text: notes, summaries, file and module notes, the outline and likely names all go
through the one decoder, which takes both. Both are brought up to date before each result, so a
preference you state, or a strength you edit, while the Writer is working holds from its next
result on.

Quoted code, inline code, quoted lines (`>`) and evidence lines (`path:line`) are never changed,
so citations stay byte for byte; a draft for a code file is checked and never rewritten. A rule's
scope is global, a project (`project:<key>`), file kinds (`kinds:kicad_pcb,kicad_sch`) or folders
(`folders:docs`); a statement such as "for PCB designs, ..." or "under docs/, ..." sets it.

* Interface: Learning page, Preferences. Type your words, choose **preview rules** to see each rule
  and how it is held, then **write** to make the cartridge. Each cartridge lists its rules (each
  labelled want, avoid or its kind, a want with its strength and a **set** control for a new
  strength), scope, versions and plug state, with add, drop, plug in or unplug, and remove (a
  preview and a typed confirmation). The draft check lists the wants a draft misses. The side panel's Preferences tab describes the forms and how rules are held. The
  cartridge also sits on the Cartridges page's whole-engine rack with its plug.
* Command line: `neuromorph-nano preferences say "<your words>" [--name N] [--scope S]` previews;
  add `--yes` to write it. Also `list`, `edit <id> [--add WORDS] [--rule rN --text WORDS] [--rule
  rN --strength 0.1..2.0] [--drop r1,r2] --yes`, `plug <id>`, `unplug <id>`, `remove <id>` (a preview) then `remove <id> --confirm
  <id>`, `undo <audit id>`, and `apply [--path P] --text "<draft>"`.
* MCP: `nm_learning` with `op` `preference_say`, `preference_list`, `preference_edit`,
  `preference_remove`, `preference_plug`, `preference_unplug` and `preference_apply`
  (`docs/agents.md`).

Every edit writes a new version; the earlier version stays in the library and undo returns to it.

### Cartridge parameters

Parameters are their own domain, on the Parameters page. Some cartridges hold parameters and some do
not; the parameters of objects that are not cartridges in the library (job filters, the project
adapters in the adapter library, learning units) are read and edited the same way.

Below the map atlas, the cartridge parameters list every object that holds parameters, per
MicroModel, with each MicroModel's upstream open weights first (`open:<model>`: read in place, never
written; edited, masked and frozen in their overlay, below): socket cartridges and compensation, job
filters, the embedder's lens and the judge's code read-out, learning units, preference counts and
base cartridges. The list is read from labels and the catalogue alone: key, kind, state, the layer
range a label records (shaded on the strip), the size on disk and the overlay's entry counts. No
tensor, value or norm is read for the list, and it is not polled (refresh reads it again). Pick one
and its parameters are read and open in the side panel:

- every tensor or point with its shape, element type, L2 norm, largest magnitude and a histogram;
  pick a tensor to read its values page by page;
- the norm per layer, with a switch per layer;
- a strength from 0 to 4 (0 makes the object change nothing, 1 leaves it as it is);
- for tensors of at most 65,536 values, the values themselves, checked against the tensor's shape.

**Apply** makes a new version beside the source, which is never written: a library cartridge's new
version arrives on trial (put it in a stack, or try it for one run with `--try-cartridge`); a job
filter's new file arrives plugged out (plug it in, and the source out, to use it); a project
adapter's new version is the newest in the adapter library and loads at a serving process's next
look when it fits the base it was fitted on; a learning unit's new version is a candidate (the
registry holds one open unit per node, so retire the source first). The new version's fit to the
installed MicroModels is checked and shown, and Audit undoes the edit by setting the new version
aside (`library/cartridges/set-aside/`, or the role's map archive), deleting nothing. A base
cartridge's edit is written as a new base version beside the others, on trial: the version serving
now and the upstream open weights are never written, the new version is checked against the open
weights when it is seated, and you choose it to serve on Cartridges, Base versions. An edit reads at
most 32 Mi values at once (select a tensor, a layer or a range), a part the cartridge takes from the
open weights has no values of its own to edit, and the edit is refused when the drive lacks room for
the new version above the disk floor. Preference counts are set as values (Cartridges page,
inspector). A cartridge that holds no parameters, such as an engine cartridge, says so.

#### Focusing on one cartridge

On the Cartridges page, a cartridge's drawer and its inspector offer **view parameters**; a
cartridge that holds no parameters says so instead. It opens the Parameters page focused on that
cartridge (`/parameters/?cartridge=<key>`; picking a row of the cartridge parameters does the same).
The middle of the page is then the cartridge's working view, with **back to Cartridges** and **all
maps** to return:

- its tensors and points on its MicroModel's layer strip, with the norm of what it changes per
  layer; click a layer to narrow to it;
- its mapping beside it: the parameter maps measured on it or on the base it was built on, the maps
  that used it, job filters at the same points and maps holding values on the same layers, each
  drawn on the same layers;
- pick a tensor to drill down into a grid of its values (64 by 64 at a time, paged, never loaded
  whole); click a value, or shift and click to draw a range or a block, or take its row, its channel or its attention head (rows of the head size you give, 128 for the Qwen3 MicroModels);
  the selection shows its values beside the map values at the same place and its place in the
  tensor's distribution;
- edit the selection, or a layer, a tensor or the whole cartridge (or type a path such as
  `L3.o.gain[0:16]`): set, add, multiply, a strength, clamp, zero, identity, negate, paste, copy or
  interpolate from another version or a job filter map, drop, or truncate the rank. **Preview**
  shows old against new with the change in norm; **apply as a new version** makes it, as above;
- **export values**, in the same controls block, writes the values of the first tensor the selection
  names (at most 65,536) to `exports\parameters\` as `nm-parameter-values/1` and names the file it
  wrote; nothing of the cartridge is written, and the file can be read back with `import`.

MCP and the API take the same operations on a path or a selection, with a preview mode
(`nm_cartridges` `op: parameters_edit` with `edit`, `path`, `preview`; `POST
/api/cartridges/parameters`), read a region with the map values at the same place (`op:
parameters` with `path`) and the maps beside a cartridge (`maps: true`), and export a selection to
a file that `import` reads back (`op: parameters_export`).

The inspector on the Cartridges page links to the same view (**view parameters**), as does
`/parameters/?cartridge=<key>`. MCP: `nm_cartridges` `op: parameters` (without `key`, the list; with
`key`, the section; with `tensor`, a page of values) and `op: parameters_edit` (`key`, `scale`,
`layers`, `values`). The command line:

```powershell
neuromorph-nano library parameters
neuromorph-nano library parameters filter:reranker/code-slice
neuromorph-nano library parameters-edit filter:reranker/code-slice --scale 0.5 --layer 27=off
neuromorph-nano library parameters open:qwen3-0.6b --overview
neuromorph-nano library parameters open:qwen3-0.6b --overview --tensor model.norm.weight --w 64 --h 1
neuromorph-nano library parameters open:qwen3-0.6b --overview --tensor model.embed_tokens.weight --w 128 --h 128 --region 0:4096,0:1024
```

A held-out check of an edited job filter runs in a job filter build, on the graphics card; the
Parameters page states this and does not run it.

#### The parameter overlay: live edits, masks and freezes

Every object that holds parameters (the open weights, base cartridges, socket cartridges, job
filters, adapters, learning units and preference counts) has a parameter overlay: an ordered list of
edits, masks and freezes held beside it in `memory/overlays/<key>-<hash>.json`. The overlay is the
default for every edit in the focus view, and the only way to change the open weights: their files
(`weights/<model>/files/*.safetensors`) and every cartridge file stay exactly as released or written,
and the engine applies the overlay when it seats the object. **Apply as a new version** stays
available beside it (the **apply in** choice) for every object that makes new versions.

- **Edit**: set, add, multiply, strength, clamp, zero, identity, negate, paste, copy, interpolate
  toward one version, or blend two (`source`, `source_b`, `t`) on any selection, a whole tensor
  included (up to 16 Mi values). A paste, copy, interpolation or blend keeps the values it gives
  (beside the overlay file once large), so it replays the same whatever happens to its source
  later. Structural changes (drop, add a point, rank by truncated singular value decomposition) are
  made as a new version.
- **Mask** the selection, a tensor, a layer, or the whole object (`*`): a weight goes to 0, a gain
  to 1 and a bias to 0, a norm weight of the open weights or a base cartridge to 1. A whole mask
  leaves a cartridge's effect out while it stays plugged; a base cartridge masked whole is left out
  where it is offered, so the next one or the open weights serve; the open weights masked whole have
  every layer an exact identity on the residual stream, and the token table, final norm and output
  head stay, because masking them leaves no output (the entry says so). Edits apply in order and a
  mask then wins over them; unmasking returns the edited values exactly.
- **Freeze** makes a region read-only: an edit, mask or unmask touching it is refused with the freeze
  named, and so is clearing the overlay. Only **unfreeze** lifts it.
- The value grid draws masked values dimmed and frozen ones hatched, with the overlaid value in the
  title; the layer strip and the tensor chips carry the same marks; the selection lists the overlaid
  value beside the released one. The overlay's entries are listed with unmask and unfreeze each,
  **undo last change**, and **clear** (a preview, then a confirmation). Every change is one audit row,
  undone from Audit.

When a change takes effect is stated in the reply and on the page:

| Object | Takes effect |
| --- | --- |
| Job filter | applied at the next call (a serving process loads job filters at each forward pass, checked once a second) |
| Socket cartridge, adapter-family cartridge | applied at the next call where it is plugged (the serving process re-reads overlays within a second) |
| Preference counts | applied at the next call (the next reply) |
| Lens, judge read-out | applied at the next call (loaded again at the next look) |
| Learning unit | applied at the next call (a loaded unit is set on its node again; masked whole, taken off it) |
| Base cartridge | applied at the next call (the server binds that MicroModel again); the seat check compares the cartridge as written, then the overlay applies |
| Open weights | applied at the next call when they serve the MicroModel (no base cartridge seated; the server binds it again); the Writer at its next load |

For the open weights and base cartridges the serving process binds the whole MicroModel again with
its overlays where the next call starts, rather than rewriting touched tensors in place (a bound
model holds packed and card copies of every tensor, made for its precision). The reply states how
long the last one took on this machine, or that it is not measured yet.

When the Embedder is bound again with other served values (an overlay or a plug change on its
open weights or base cartridge), the dense lane follows without a restart. Its vectors carry a
header naming the values that made them; when the Embedder's header differs, the lane's vectors are
marked stale. From then on the meaning ranking is withheld, and questions are answered by the
lexical and structural paths alone, because a question encoded by the new Embedder does not lie in
the old vectors' space (only a change of the piece-side text recipe would leave the old vectors
valid, and then they keep serving). A new index is built in the background with the new Embedder
when the process is idle: no call in flight and none for the idle Writer's quiet time. A call that
starts stops it at its next batch, as it stops the idle Writer, and the work done so far is kept.
The new index is written to the project's `dense/` folder under the new header and replaces the
stale one in one switch once every current piece has a vector. The Writer's notes and likely names
are encoded again with it. A restart during the rebuild resumes from the rows already written. A
lens or distilled request lens change acts on the question side only, so it does not mark the
vectors stale. The rebuild shows in the serving log and the live view (`dense`), and a context
pack asked with `debug` names it in its diagnostics: `dense_lane_stale` (why the vectors are stale,
empty while current), `dense_lane_withheld` (questions answered without them) and
`dense_lane_follow` (the last step: the change seen, or the rebuild switched in).

Lab measurements and builds apply overlays by default and say so in their first line
(`neuromorph-nano lab ... --no-overlay` reads the released files). Every report they write under
`runs/` (distillation, parameter maps, surgery, and the lab's own runs) starts with one line
recording it: `overlay: applied (<n> objects)`, `overlay: not applied (--no-overlay)` or
`overlay: none present`; a surgery probe file holds the same in its `overlay` field. Values of the open weights and base cartridges are held as BF16, so an edit's result is
rounded to BF16 as the engine holds it.

MCP: `nm_cartridges` `op: overlay` (`key`), `overlay_edit` (`edit`, `path` or `selection`, `value`,
...), `mask`, `freeze` (`path`, or `whole: true`), `unmask`, `unfreeze` (`id`, `ids` or `path`),
`overlay_undo`, `overlay_clear` (two calls), each with `preview: true`. The API: `GET` and `POST
/api/parameters/overlay`. The command line:

```powershell
neuromorph-nano library overlay open:qwen3-reranker-0.6b
neuromorph-nano library overlay open:qwen3-reranker-0.6b mask --path L3 --preview
neuromorph-nano library overlay filter:reranker/code-slice edit --op mul --value 0.5 --path L2.o.gain
neuromorph-nano library overlay filter:reranker/code-slice freeze --path L2
neuromorph-nano library overlay filter:reranker/code-slice undo
```

#### The whole-set overview and floating Parameter Inspector

An object opened in the parameter editor can be seen whole: every tensor of the set, grouped by
layer and then by block (attention, mlp, norm, the token table), each drawn as a heatmap. You can
switch directly to any parameter object (job filters, compensation/socket cartridges, adapters/lenses/units,
preference cartridges, base cartridges, and optional open weights) from the grouped selector on the
Parameters page or inside the viewer header, or click any `base cartridge`, `open weights`, job filter,
`feeds` badge, or per-layer heat bar on `/parameters/` to jump straight into the editor (with an
in-viewer `layer: all / L0..Ln` filter bar). On the **Models** page (`/models/`), each MicroModel
also includes **Layers on/off** switches to mask or unmask individual layers through the reversible
overlay (audited with Undo). Nothing is read until an object is opened, a tensor's heatmap is read
only when it scrolls into view, and nothing polls. The weights are only read.

- Each cell of a heatmap summarises the weights it covers; choose the statistic drawn: mean, min,
  max or RMS, and signed (`diverging`) or magnitude (`sequential`) scale. Paired parameter maps can
  be overlaid directly on the heatmap (coordinate maps such as `super-weights.json` draw gold weight
  pins; layer/filter maps highlight covered layers and tensors), with `on / emphasize / hide` toggles
  for both `masks` and `map`.
- Use the mouse wheel to zoom at the cursor, drag to pan, Shift+drag (or switch to **box zoom**) to
  zoom into a box, or click `-`, `+`, **1:1 exact**, and **fit** in the toolbar. Zooming in to one
  weight per cell shows exact values printed in cells and under the cursor.
- **Floating Parameter Inspector:** Clicking any weight or multi-weight block opens a movable,
  resizable, scrollable floating **Inspector** popup (drag its title bar to move, drag edges or the
  bottom-right corner to resize, double-click the title bar to collapse, or click **inspector** in
  the toolbar to toggle). Single-clicking inspects values, coordinates, tensor percentile, and
  parameter maps at that place; double-clicking any weight or block opens the popup and focuses the
  inline **edit manually (overlay)** input (`set`, `add`, `mul`, `scale`, `mask`, `freeze`, `unmask`),
  where every edit is reversible and audited.
- **select view** (and the Inspector's weight/row/column/head/block buttons) hands the region to the
  edit, freeze and mask controls below as a path (`name[r0:r1, c0:c1]`, or `name[c0:c1]` for a vector).
- **values** lists every value of the zoomed region with its index, row and column, page by page.
- **Parameters -> Build** walks through creating a new job filter, parameter map, or pruned base
  cartridge copy in four steps, with a collapsible `how it works` guide, layer range presets
  (`all`, `first half`, `second half`, `last 8`), and a `share mapped out` slider for parameter maps.

A tensor is read from its file in chunks of at most 4 MiB and folded into the cells as it is read,
so even the 151,669 by 1,024 token table is never loaded whole. MCP: `nm_cartridges` `op:
parameters_overview` (`key`; with `tensor`, `w`, `h` and the region `r0`, `r1`, `c0`, `c1`); the
API: `GET /api/parameters/overview`; the command line: `neuromorph-nano library parameters <key>
--overview [--tensor <name>] [--w N --h N] [--region r0:r1,c0:c1] [--hist]`, which prints the same
JSON (an `error` the route answers with is the command's error).

### Cartridge libraries

Base cartridges, active parameter maps, job filters and shared hardware stacks come from the
official public library on Hugging Face,
[`gavriel-tech/neuromorph-nano-cartridges`](https://huggingface.co/gavriel-tech/neuromorph-nano-cartridges)
(no token required), and from any third-party Hugging Face library you add (**Cartridges -> Sources / Downloads**,
**Models -> Libraries / Downloads**, or the floating movable/resizable **HF downloads** window on
**Cartridges**, **Models**, and **Parameters**). In the floating **HF downloads** window (or the side
panel) you can view everything the repository offers across `base cartridges`, `job filters`, `maps`,
`stacks`, and optional `open weights`, filter by category, MicroModel role, or search text, expand
`info` on any file to see what it does, check the exact files you want and click **download selected**
(with confirmation and live queue progress), pull everything with **fetch all**, or share your own
local parameter maps and non-base stacks via **Publish to Hugging Face** (`source_publish`,
confirmation-gated and audited). Nothing is fetched or uploaded until you ask, everything fetched
goes through the checks of the cartridge library on this machine, and the air gap blocks it all.
Details: `docs/cartridges-and-parameters.md`, Cartridge libraries.

### From the command line

```
target\release\neuromorph-nano.exe library list
target\release\neuromorph-nano.exe library stack
target\release\neuromorph-nano.exe library bases
```

`stack` reads this machine's pieces (graphics card, processor, RAM, the disk the Nano folder is
on) and shows the stack in use for each use and why. Hardware is one family of domains: each
graphics card model, processor and RAM size has its own stack. A piece the cartridge library has no stack for
gets the nearest stack of the same family on trial: loaded and recorded, never changing what is
served. `library switch <use> <stack>` picks a stack for a use (`off` to unplug): a running server
plugs its socket cartridges where its next call starts, with no restart, and its settings at the
next start of a server; `/api/undo` undoes a switch. `library base <model> <key|auto|open>`
chooses what a MicroModel serves from the next start. `library export <stack> --out <file>` and
`library import <file>` carry a stack to another machine as one file; an import is checked file by
file and arrives on trial. Every subcommand: `docs/configuration.md`, Other commands.

The cartridge library starts with one hardware stack, for the GeForce GTX 1650 with 4 GB: the operating
numbers the speed passes chose on that card. `serve ... --rack off` leaves the cartridge library's stacks out for one
run. `serve ... --try-cartridge <id>` plugs a named cartridge for one run even while it is on
trial, to measure it.

## Make your own maps and cartridges

The lab is part of the program (`neuromorph-nano lab ...`). It needs the upstream open weights on
disk (Models, Downloads) and runs on the graphics card, or more slowly on the processor.

All three builders are on Parameters, side panel tab Build. The two whose result is a cartridge
(job filters and base cartridge versions) are also on Cartridges, side panel tab Build: the same
builder, the same routes. They are in MCP (`nm_cartridges` `filter_build`, `map_build`,
`base_build`) and the API (`/api/lab/build`). One build runs at a time,
it shows in Live Activity while it runs, and Audit records it. A new base cartridge version is kept
beside the ones you have (nothing is removed); you choose what serves on Cartridges, Base versions.

- **A job filter:** pick the MicroModel, name the job and choose the layers, then Build. The Nano runs the lab in the background and keeps the
  filter only if it passes the held-out split gate; it then appears in the atlas and on that
  MicroModel's deck on Cartridges, where you plug it in or out. In this version a filter is fitted
  on the lab's built-in prompts for that MicroModel's role; the job name labels it. The same
  build from the command line:

  ```powershell
  neuromorph-nano lab run pmap surgery job-filter <role> --job <job> --layers all --split-gate --write
  ```

- **A parameter map:** measures how much each part of a MicroModel's weights matters, then cuts a
  new base cartridge version from it. In the Build tab: the share mapped out (0.05 to 0.60) and
  the graphics card or the processor.

  ```powershell
  neuromorph-nano lab run pmap build-base <role> <share> gpu --keep
  ```

- **A base cartridge:** a new version of the one serving now, with the weights that matter most
  restored from the upstream open weights.

  ```powershell
  neuromorph-nano lab run pmap surgery cascade <role> --anchor-budget 8 --budget 5 --write --keep
  ```

  On the command line, `--keep` adds the new version beside the others; without it the lab replaces
  the base cartridge in place.

Every map the lab writes is recorded with where it came from and the command that made it. On
Parameters you label how each map turned out (used, superseded, failed, duplicate, control or
unused, with the reason): a map labelled anything but used or superseded moves out of the parameter map library
into its archive at once, and only healthy maps are learned from. `neuromorph-nano lab --help`
lists every lab command; [How the weights are built](build-flow.md) walks through the whole chain.

## Distillation: teacher data into learned cartridges

Distillation is a learning function. It takes a teacher's outputs (or curated open-model data) as
one more learning source, beside usage signals and stated preferences, and fills learned
parameters from them into a learned cartridge of the kind you choose, written in that kind's own
format, so it plugs into the decks and stacks exactly as cartridges of that kind do. Nothing
leaves the machine: the teacher data is a file you bring, or the output of a local open-weights
teacher (`lab run distil teach`).

Its home is the Learning page, section Distillation (the same builder is shared with the
Cartridges build tab), with the same actions in MCP (`nm_learning` `distil`, `distil_import`,
`distil_label`; `nm_cartridges` `distil_build`, `distils`) and the API (`/api/training/distil`,
`/api/training/distil/import`, `/api/training/distil/label`, `/api/lab/build` with kind `distil`).

1. **Teacher data.** Import a JSON lines file of teacher pairs on this machine (Learning,
   Distillation, teacher data). Each line holds `prompt`, `completion`, `teacher`, `licence`, and
   optionally `topk` (the teacher's top log-probabilities per completion token, Qwen3 token ids) and
   `template` (`chat`, the default, or `raw`). For a preference cartridge, a line also carries the
   teacher's `verdict` on the completion (`accept` or `reject`) and its circumstance: `paths` (the
   files it concerns; their file kinds and top folders are the circumstances) and `tool`. For the
   Reranker, a line is a scored pair: `prompt` is the query, `completion` the document, and `score`
   the teacher's judgment from 0 to 1 that the document answers the query (a hard 0 or 1 is read as
   0.01 or 0.99). A scored pair or a rejected completion is never read as a next-token target for
   the Writer or the Reader. The import lists the file as reference material with
   its licence and hash, fixes each pair's split by its prompt's hash (80 fit, 10 tune, 10 held
   out), passes over pairs already there, and labels the pairs `used`. Label a set `unused` or
   `failed` to keep it apart: only `used` and `superseded` pairs are ever trained on. Audit undoes
   an import or a label.
2. **What to fill.** Choose the MicroModel, the kind of learned cartridge and the technique. The
   kinds and the techniques that can fill them:

   | Kind | Techniques | Available in v0.1.x |
   |---|---|---|
   | Adapter cartridge (a learned unit on the MicroModel's deck) | `lens`, `teacher-loop`, `unit`, `steer` | Writer and Reader: `lens`, `teacher-loop`, `steer` (teacher completions); Reader: `unit` (a learning unit); Reranker: `lens`, `teacher-loop` (teacher scores); Embedder: `lens`, `teacher-loop` (a request lens, from teacher similarity scores) |
   | Filter cartridge (a job filter, `nmengn-sockets/1`) | `channel`, `teacher-loop` | Writer and Reader (teacher completions); Reranker (teacher scores); not the Embedder (a filter would change the stored vectors) |
   | Preference cartridge (`pref.*` values) | `readout` | every role (preference values are engine-wide), from judged pairs |
   | Read-out cartridge (a judge read-out) | `readout`, `teacher-loop` | Reranker, from teacher scores (a read-out adds to one score; the other roles do not give one) |

   The Embedder is distilled into a request lens only: it maps the request's vector (the dense
   lane's question), never the stored piece vectors, so the dense store stays valid with it plugged
   in or out. It is applied before the project's embedder lens, and its key joins the lens key and
   the lens epoch, so cached request vectors and the routing records that hold for one lens are
   made again. `steer` is built for the Writer and the Reader; `unit` for the Reader, whose model
   the learning units serve on (`serve --unit`). A pair that is not available yet is shown as such with its reason
   (the builder shows the engine's reason under the train button), and a build of it is refused
   with that reason; nothing stands in for it.
3. **Train.** One lab build runs at a time and shows in Live Activity. `lens` fits a low-rank map
   at the MicroModel's final norm in closed form: under teacher forcing, each completion position's
   target is the smallest shift of the final-norm state that moves the head's logits toward the
   teacher's token (or its log-probabilities), then reduced-rank ridge fits the map, its strength,
   rank and scale chosen on the tune split. `channel` fits a gain and bias per channel toward the
   same targets. For a preference cartridge, `readout` counts the teacher's verdicts per
   circumstance into the preference values the serving path reads, smoothed the same way as the
   values learned from your own work, on the processor. `teacher-loop` refits over rounds, each round weighing up the fit pairs the student
   still follows the teacher worst on, and stops when the tune split stops gaining.

   For the Reranker the targets are scores, not tokens. Each scored pair is read as the serving
   Reranker reads it (its prompt with the code instruction); its state at the last token gives the
   judge's log-odds (its yes logit minus its no logit). `lens` and `channel` fit toward the
   smallest shift of that state that moves the log-odds to the teacher's. `readout` fits a ridge
   read-out of the difference between the teacher's log-odds and the judge's, and serves it along
   the one direction that raises the yes logit over the no logit, so the judge's score moves by the
   read-out and nothing else it reads changes.

   `steer` fits one vector added to the residual after the last layer, at every position: the mean
   shift that moves the final norm's output toward the teacher's targets in first order (through
   the norm's own gain), its scale chosen with the vector in place in the engine. `unit` fits a
   learning unit (rank 8) at the Reader's learning target forward-only (MeZO) against the
   teacher-token loss, each fit within a wall budget (`--budget-s`, 20 seconds by default, at
   most 120; a build runs five fits: the fit, the control and three trials), and gates it as its
   quantized files load. For the Embedder, the teacher's similarity scores are put on its own
   cosine scale (their mean and spread matched on the fit split); per query, the smallest shift of
   its vector that moves its cosines to those targets is fitted by reduced-rank ridge.
4. **The held-out gate.** A cartridge is written only when, on the held-out split: the
   teacher-token loss falls and top-1 agreement rises; a shuffled-target control gains at most a
   quarter as much; the drift on neutral text (the Nano's own source) stays within 0.10 nats per
   token; the Writer's note check (its answer must find its own passage above its eight nearest
   neighbours) is not worse; and three repeated trials, each without a third of the fit pairs, all
   gain. A steer or a unit is measured by the engine with it in place (the Writer's note check with
   the steer seated on it). The Embedder's gate, on the held-out scored pairs: the error of its
   cosines from the teacher's scores falls and same-query order agreement does not fall; a
   shuffled-target control gains at most a quarter as much; neutral requests (from the Nano's own
   source) move by at most 0.02 in cosine distance; three repeated trials all gain; and the lens
   read back from its file is the one gated. The Reader has the same gate without the note check (its answers are not notes). The
   Reranker's gate, on the held-out scored pairs: the relative entropy of its verdicts from the
   teacher's scores falls and its verdict agreement with the teacher (above or below one half) does
   not fall; a shuffled-target control gains at most a quarter as much; the drift of its verdicts on
   neutral pairs made from the Nano's own source stays within 0.05 nats per pair; three repeated
   trials all gain; and an adapter or a read-out, seated on the Reranker and read through its
   serving forward, gives the scores the fit measured (within 0.05 in log-odds). A preference
   cartridge is written only when its held-out verdicts are predicted better
   (Brier score) than with no cartridge. Otherwise the build reports `REFUSED:` with the figures and
   writes nothing. Every build
   writes a report under `runs/` and an audit row.
5. **Plug it in.** A new distilled adapter starts unplugged on the MicroModel's deck on Cartridges;
   a new distilled job filter starts switched off on its lane; a new distilled preference cartridge
   starts unplugged on the whole-engine rack, beside any stated preference cartridge. Its label records the model it was
   trained for, the open weights' SHA-256, the base cartridge version it was fitted on, its socket
   (node, rank, shape), the teacher data, its licences and its gate figures; the compatibility
   table shows it, and it plugs in only where it fits. A distilled adapter or read-out serves on its
   MicroModel as the MicroModel opens (the Reader and the Reranker in the serving process, the
   Writer also at each idle writing session), and a plug change reaches a MicroModel already open.
   A distilled Reader or Reranker job filter serves through the job filter path, which loads the
   Reader's and the Reranker's job filters by default. A distilled steering vector serves through
   the MicroModel's steer seat, used by every run that gives none of its own (a MicroModel without
   one pays nothing for it); a distilled request lens serves on the embedder's request vectors as
   the embedder opens. A distilled learning unit is registered with the learning units (Learning,
   learned units; `/api/units`) and loads as they do, with `serve --unit <id>`. Inspect it with `nm_cartridges` `inspect`
   (or `/api/cartridges/inspect`); `set_values` (`scale`, `rank`; a read-out or a steer has `scale` only)
   writes a new version, never over the old one, and Audit undoes it; `lens_remove` removes it in
   two steps.

### A live teacher

The teacher can also answer live, with no file in between:

- **Your agent** (Claude Code, Antigravity or any MCP client, with a cloud model or a local model
  behind it). Open a teaching set (Learning, Distillation, live teacher: a name, the prompts, the
  instruction the teacher follows, the licence of its answers), then ask your agent to teach it.
  It calls `nm_learning` `teach_next` for the open prompts, answers each as the teacher, and sends
  the answers with `teach_answer`, naming itself as the teacher (for the Reranker, each answer
  carries the document as `completion` and the agent's `score` from 0 to 1). The answers join the teacher data
  like an import: split by the prompt, labelled `used`, listed as reference material with the
  teacher and the licence, and undone in Audit.
- **A local open-weights model.** Choose an installed model and **teach with a local model**: it
  answers the set's open prompts on the graphics card (with its top log-probabilities), and the
  answers join the teacher data when it ends.

The instruction is shown to the teacher only, so the student learns to do it without being told.
A set's answers then train through the same held-out gate as any teacher data.

Expected effect: a closed-form map at the last layer changes the teacher's style, format and
wording preferences in the MicroModel's output, not its knowledge or reasoning. For the Reranker,
its yes or no score moves toward the teacher's scores.

From the command line:

```powershell
neuromorph-nano lab run distil teach --prompts <prompts.jsonl> --out <pairs.jsonl> [--teacher qwen3-1.7b] [--instruction "<what the student should learn to do>"]
neuromorph-nano lab run distil build writer --target adapter --set <set> --technique lens
neuromorph-nano lab run distil build writer --target preference --set <set> --technique readout
neuromorph-nano lab run distil build reader --target filter --set <set> --technique channel
neuromorph-nano lab run distil build reranker --target readout --set <set> --technique readout
neuromorph-nano lab run distil build embedder --target adapter --set <set> --technique lens
neuromorph-nano lab run distil build writer --target adapter --set <set> --technique steer
neuromorph-nano lab run distil build reader --target adapter --set <set> --technique unit --budget-s 20
```

## Your own repositories: backup and rollback

You can keep your Nano data in repositories you own, to back it up, store it elsewhere and roll it
back: any git remote (your own GitHub repository or another git host), a Hugging Face repository
(also a git repository), or a local folder. It needs git on the machine (Git for Windows).

For each repository you choose what it holds: **memory**, **parameter maps**, **cartridges**,
**learned units** and **settings** (all but cartridges by default). Nothing is sent until you back
up, and the air gap refuses it.

- **Back up now** copies the chosen parts into a new snapshot (one commit, with a `MANIFEST.json`
  naming what is in it and what was left out) and sends it to your repository. Stores the Nano
  rebuilds (dense vectors, meaning caches, live snapshots, temporary files) are left out, and so are
  files over 50 MiB: git hosts refuse them without large-file support, and base cartridges come
  back from the cartridge library.
- **Snapshots** lists every backup, newest first.
- **Preview restore** shows what restoring a snapshot would write (new and changed files); it
  writes nothing.
- **Restore** writes those files back. Every file it changes is copied aside first
  (`backups/<id>-before-<time>/`), files only on this machine are kept, and Audit undoes the
  restore. Restore while no serving process is running, or restart it after.
- **Send my token** sends your token for that host only: `HF_TOKEN` (or the `hf` login) for Hugging
  Face, `GITHUB_TOKEN` or `GH_TOKEN` for other hosts. Without it, git's own sign-in answers.

Interface: Storage page, Your repositories. MCP: `nm_settings` with `op` `backups`, `backup_add`,
`backup_remove`, `backup_now`, `backup_snapshots`, `backup_preview`, `backup_restore` (needs
`confirm: true`). API: `GET /api/backups`, `POST /api/backups`. Command line:

```powershell
neuromorph-nano backup add https://github.com/you/nano-backup.git --parts memory,maps,units,settings
neuromorph-nano backup now <id>
neuromorph-nano backup snapshots <id>
neuromorph-nano backup preview <id> <commit>
neuromorph-nano backup restore <id> <commit> --confirm
```

## Reading any file

The Nano reads every file of a connected folder, by one of two means.

Read specifically, with structure (headings, sections, tables, records, definitions): code in any
language, Markdown and plain text in any common encoding, CSV and TSV, HTML, XML, JSON, JSON Lines,
notebooks, YAML, logs and PDF.

PDF: the text of every page is taken from the file's content streams and put back in reading
order, by the file's tagged structure when it has one and otherwise by the layout (columns are
read one after the other; titles and tables that run across both columns stay whole). Running
headers, footers and page numbers are left out, words hyphenated at line ends are joined, and
headings come from the tagged structure, the bookmarks, or the font size. A file that opens with an
empty password (protected only against printing or copying) is read. Each paragraph or table row
is one line of the derived text, and every pack span names its page, for example
`@@ manual.pdf:123-136 | manual.pdf:p7 | 4 Routine maintenance > 4.2 Descaling` (lines of the
derived text, then the page and the section).

Read generally, with no reader made for the format:

- Containers: zip, gzip and tar (also `.tgz`) are opened in memory; nothing is extracted to disk
  and member names are never used as paths. Each member is read by whichever reader fits it, by
  the same rules as a file on disk, and is cited as `container!member:lines`, for example
  `handbook.docx!word/document.xml:7`. Word, Excel, PowerPoint, EPUB and OpenDocument files are zips
  of XML, so their text arrives through the XML reader. Members that hold no text (images, fonts,
  binary parts) are skipped. A zip inside a zip is opened up to three levels deep.
- Anything else (old binary Office files, programs, databases, images, unknown data):
  the runs of readable text recovered from the bytes, in UTF-8, UTF-16 and Latin-1, each line
  starting with its byte offset (`byte 1508: ...`), under a short header giving the file's name,
  its size and what it is by its first bytes.

Every pack line of a file read generally says so (`| read generally: container member`, `| read
generally: text runs recovered from the bytes, no headings or pages`), and `nm_status` counts such
files under `read_generally`.

Limits, plainly:

- Compressed or encrypted content gives little or nothing by raw reading. Formats that compress
  with something other than deflate (7-Zip, RAR, bzip2, xz) are not opened.
- Encrypted or password-protected containers: only the member names are read, and the listing says
  so. Save a copy without a password to have it read.
- Text read generally has no headings or pages, and Word headings, Excel cells and sheets and slide
  order are not rebuilt: a spreadsheet's numbers and its labels arrive as separate lists.
- Runs shorter than eight characters are not kept, so single short words inside binary files are
  lost; non-Latin scripts other than Greek and Cyrillic are not recovered from UTF-16.
- Images and scans hold no text; the Nano does not run OCR. A PDF page that is only a picture (a
  scanned page) is reported in the document's first line (`not read: page 2 (no text layer: a
  picture only, such as a scan)`) and nothing from it is indexed. Run OCR (for example in the
  scanner's software) and add the text version.
- A PDF font that gives no way to map its glyphs to characters (no ToUnicode map and no usable
  encoding) makes its text unreadable; such a page is detected (more than three in ten glyphs
  without a character) and reported the same way, not indexed as noise.
- A PDF that needs a real password is skipped with that reason; the Nano never asks for or keeps
  passwords. A PDF the reader cannot open at all is read raw, with a note.
- PDF layout: tables come out row by row with cells separated by tabs, and a cell that wraps onto a
  second line can mix with its neighbours; list bullets drawn as shapes are not text; footnotes,
  sidebars, equations, right-to-left and vertical scripts are not put in order. Composite fonts of
  Chinese, Japanese and Korean without a ToUnicode map are not read.
- Size limits: 512 KB for a text file, 64 MB for a PDF or a file read generally, 8 MB for one member and
  64 MB inflated in all from one file; a deflate stream that expands more than 200 times past its
  first megabyte is stopped as a decompression bomb.

Skipped, with the reason counted in `nm_status`: model weights, files over the size limit, secrets
(`.env` files, private keys and credential stores), the Nano's own runtime state, and generated,
minified, lock, vendored, hidden and `.gitignore`d files.

With memory on, the text derived from your files is kept in a cache in the project's memory folder
so a restart does not read everything again (`docs\privacy.md`).

## Switching memory off

Set Memory to off in the interface, or start the server with `serve --memory off`. Nothing is then
recalled or written to memory; the audit keeps only the tool, project, time and outcome of each
call.
