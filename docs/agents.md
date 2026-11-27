# Agents and models

The Neuromorph Nano works beside the coding agent you already use, whether that agent runs a cloud
model or a local one: it reads and remembers your project, gives the agent the context a question
needs, builds and checks with it, and learns from what follows. There
are three ways in.

| You have | Use | What you get |
|---|---|---|
| An agent that speaks MCP (Claude Code, Cursor, VS Code, Antigravity, Windsurf, or any other MCP client, cloud or local model) | `neuromorph-nano connect <client>` | The Nano's tools inside the agent |
| A model or tool without MCP (a local model runner, a script, your own harness) | `neuromorph-nano daemon`, then `ask` or `attach` | Context packs and tool results as text or JSON on the command line |
| A script that reads the Nano's state | The local interface API | JSON from `http://127.0.0.1:4870/api/...` |

## 1. Any MCP agent

```powershell
neuromorph-nano connect claude --project C:\path\to\your\repo
```

The named clients are `claude`, `cursor`, `vscode`, `antigravity` and `windsurf`; `connect` writes
their configuration file and keeps the other servers already in it. Add `--dry-run` first to see
the exact change.

For any other MCP client, including agents that run a local model:

```powershell
neuromorph-nano connect generic --project C:\path\to\your\repo
```

prints the command line and the JSON entry to put in that client's MCP configuration. The server
runs over standard input and output (`neuromorph-nano serve --root <repo>`), so it needs no network
port.

When the agent connects, the Nano sends it standing instructions (MCP `instructions`): call
`nm_context_pack` with the question as the first step, before searching files, then read only the
ranges it cites (`path:first-last`). Every tool also describes itself and its arguments, so an
agent needs nothing more to use them. All tools, their arguments and their errors are in
[MCP tools](mcp-tools.md).

## 2. A model or tool without MCP

Start the Nano warm on this machine (it listens on `127.0.0.1` only, port 7421 by default):

```powershell
neuromorph-nano daemon --root C:\path\to\your\repo
```

Then, from anything that can run a command:

```powershell
neuromorph-nano ask "where is the request vector normalized"
neuromorph-nano attach nm_status
neuromorph-nano attach --stop
```

`ask` runs `nm_context_pack` and prints the context pack; `attach <tool> [json]` runs any tool with
its arguments as one JSON object and prints its JSON reply (from a script, pass the JSON as one
argument; Windows PowerShell 5.1 strips inner quotes, so a script language or `cmd` is simpler). Put the context pack into your model's prompt the way you would put a file.

## 3. Scripts and the local API

The interface server (`neuromorph-nano ui`) answers JSON on `http://127.0.0.1:4870/api/...`, for
example `/api/status`, `/api/live`, `/api/audit` and `/api/parameter-maps`. Reads are open on this
machine. Writes are accepted only from this machine's own interface, carrying its local token, so
no other program changes the Nano by accident; to change settings from a script or an agent, use
the MCP tools (`nm_settings`, `nm_cartridges`, `nm_learning` and the others), which are audited
the same way. Every route is in [Interface API](interface-api.md).

## Stated preferences from the user's words

When the user tells their agent how they want things done ("no em dashes", "use 'use' instead of
'utilize'", "for .kicad_pcb files, keep power traces at least 0.5 mm wide"), the agent passes the
words, as the user said them, to `nm_learning`:

1. `{"op": "preference_say", "text": "<the user's words>", "name": "<a short name>"}` returns a
   preview: each rule with what it does and whether it is enforced, checked or guidance. Nothing is
   written. Show it to the user.
2. On the user's confirmation, the same call with `"confirm": true` writes the cartridge, plugs it
   in and returns its key and audit id (undone with `nm_settings` `op: undo`).
3. Before writing text or code, `{"op": "preference_apply", "text": "<draft>", "path": "<file>"}`
   returns the draft with the enforced transforms applied to its prose and every violation with
   its line; a code path is checked, never rewritten.

Every `nm_context_pack` reply then carries a "preferences in force" block with the rules whose
scope fits the project and the cited files (`preferences_in_force` in the JSON form); follow them.
`preference_list`, `preference_edit` (`add`, `rule` with `text`, `drop`), `preference_plug`,
`preference_unplug` and `preference_remove` (two calls: a preview with a `confirm_token`, then the
same call with it) manage the cartridges. The interface and the `neuromorph-nano preferences`
command run the same routes.

## Reminders, loops and steps the agent sets for itself

An agent connected over MCP sets its own guidance with `nm_goal` and an `op`. It lands on the
project the server serves (unless `project` names another), is kept as set by the agent, and is
shown on the Memory page beside what you set; the hooks then deliver it back to the agent. This
needs the hooks (`setup --hooks`) for delivery.

- A reminder for later: `{"op": "reminder_add", "text": "update the changelog before finishing", "trigger": "stop", "repeat": "once"}`.
  Triggers: `session-start`, `prompt` (with `every`), `compact`, `stop`, `edit` (with `glob`),
  `failure`, `tools` (with `every`), `error-repeats` (with `error`, a recorded error code such as
  `E0308` or a piece of its message, and `count`, 2 to 5): a hint for a recurring error,
  `{"op": "reminder_add", "text": "read the type signatures before editing again", "trigger": "error-repeats", "error": "E0308", "count": 2}`.
- A loop that continues it at a stop until a check passes:
  `{"op": "loop_add", "prompt": "Fix what fails and run the tests again.", "until": "check-passes", "max": 3}`.
  Other conditions: `goal-done` (with `goal`), `steps-done`, `manual`. The hook continues the agent
  with `decision: block` at most `max` times (never more than 20); `{"op": "loop_end", "item": "l2"}`
  ends it. Combine conditions with `all_of` or `any_of` (2 to 4 of them):
  `{"op": "loop_add", "prompt": "Finish the steps and keep the tests passing.", "all_of": ["check-passes", "steps-done"]}`;
  the unmet reason then names each part not met. A loop ends early as `stalled` when the same unmet
  reason comes back `stall` times in a row (default 2, 1 to 10; not for `manual`), and as
  `budget-spent` when `budget_ms` (wall time) or `budget_tools` (tool calls) since it was set is
  used. Each end reaches the agent once, in a final note that names the reason; after a stalled
  note, change the approach or ask the user rather than set the same loop again.
  `{"op": "loop_outcomes"}` lists the ended loops with their unmet reasons, budget used and verdict.
- A plan as a step graph: `{"op": "step_add", "item": "parse", "title": "write the parser"}`, then
  `{"op": "step_add", "title": "test the parser", "depends_on": ["parse"], "check": "test", "paths": ["src/**"]}`.
  The next ready step comes back at prompts and stops; `{"op": "step_done", "item": "parse"}` marks
  one done, and a step with a check is marked done by the Nano when that check passes after an edit
  under its paths. `depends_any` adds an OR-join (at least one of those steps done), and
  `"probe": true` on a step with a check makes it done when that check passes once its
  dependencies are met, with no edit:
  `{"op": "step_add", "title": "confirm the tests already pass", "depends_any": ["parse", "lexer"], "check": "test", "probe": true}`.
- `{"op": "guidance"}` reads it all back with what was delivered and when; `remove` takes two calls
  (a preview with a `confirm_token`, then the same call with it); every change is undone on the
  Audit page.

Antigravity takes reminders at a session start and at prompts; loops do not act there (see
`user-guide.md`, Reminders, loops and steps).

## Beyond context: what an agent can use

Context packs are the first step, not the whole of it. Through the same connection an agent can:

- **Remember:** `nm_memory` (notes, episodes, temporary notes per project), `nm_goal` (objectives
  handed over, closed or reopened) and `nm_feedback` (marks an answer right or wrong, which joins
  the outcome record).
- **Build and check:** `nm_build_run` (one cargo command, errors as cited lines),
  `nm_build_explain`, `nm_verify` (a diff checked before anything runs) and `nm_stepcheck` (a diff
  tried on a scratch copy: check, clippy, tests).
- **Improve:** `nm_improve_scan` finds concrete defects and `nm_patch_propose` gives the diff for
  one; neither applies anything.
- **Run the Nano itself:** `nm_cartridges` (cartridges, stacks, base versions, job filters and
  parameter maps), `nm_learning` (learning state, training data, a learning cycle or a rollback),
  `nm_comprehension` (the 4-level world model), `nm_settings`, `nm_storage` and `nm_live`, the same
  controls as the interface, each change in the audit with undo.

**Memory and continuous learning work with every agent.** Memory is kept per project on your
disk. Continuous learning learns from what followed each answer (verified, refuted, kept or
reverted): with any MCP agent from its tool calls and `nm_feedback`, and with Claude Code's hooks
also from the session (files edited, checks run). A learning cycle runs once enough verified
examples have settled and is kept only if it passes a held-out gate; every cycle can be rolled
back. Nothing of memory or learning is sent anywhere; what memory adds to a context pack goes to
your agent's model with the rest of the pack.

## What leaves your machine, by setup

| Setup | The Nano | Your agent |
|---|---|---|
| Cloud model (for example Claude in Claude Code) | Sends nothing of its own | Sends what the Nano gives it, like any file it reads, to its model provider |
| Local model behind your agent | Sends nothing of its own | Stays on your machine |
| Local model, Air-Gap Mode on | Makes no network request at all, downloads refused | Stays on your machine |

See [Privacy](privacy.md) for what the Nano keeps on disk and for how long.

## Which agents are tested

`connect` is tested for the five named clients and for the generic entry. Other MCP clients and
local model runners work through the generic entry or `ask` and `attach`; they are not each tested
by name.

## Teach the Nano as a live teacher

When the user asks you to teach a teaching set, call `nm_learning` with `op: teach_next` and the
set's name, answer each prompt as the teacher (following the set's instruction), and send the
answers with `op: teach_answer`: `answers` as a list of `{prompt_id, completion}` (and, for a
preference cartridge, `verdict` of `accept` or `reject` with `paths` and `tool`), and `teacher` set
to your name and model. Repeat until `open` is 0. The user builds a distilled cartridge from the set
afterwards; the held-out gate decides whether it is kept.
