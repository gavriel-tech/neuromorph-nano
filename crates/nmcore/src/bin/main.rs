//! `neuromorph-nano mcp` serves the tools to Claude Code over MCP on stdio;
//! `neuromorph-nano call <tool> [json]` runs one tool and prints its result.
use nmjson::Value;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

mod activity;
mod bench;
mod coldstart;
mod connect;
mod controls;
mod cparams;
mod pedit;
mod distil;
mod documents;
mod examples;
mod fetch;
mod forget;
mod guidance;
mod harness;
mod languages;
mod learned_units;
mod hook;
use nmcore::jsonc;
mod keep;
mod launch;
mod ledger;
mod library;
mod live;
mod memory_api;
mod network;
mod onboard;
mod overlay;
mod pmaps;
mod poverview;
mod preferences;
mod backups;
mod teach;
mod prime;
mod primes_ctl;
mod outcomes;
mod project;
mod secrets;
mod sessions;
mod situation;
mod soak;
mod sources;
mod trace;
mod ui;
mod writer_trial;

const USAGE: &str = "neuromorph-nano: the Neuromorph Nano, a local context and memory engine for Claude Code over MCP

usage: neuromorph-nano <command> [args]

commands:
  serve --root <repo> [options]   serve the nm_* tools over MCP on standard input and output
  call <tool> [json] [--root <repo>]
                                  run one tool once and print its result
  setup --project <repo> [--plain] [--lean] [--write]
                                  print (or write) the repository's .mcp.json entry
  setup --project <repo> --hooks [--write]
                                  print (or write) the Claude Code hook entries in the
                                  repository's personal .claude/settings.local.json
  setup --project <repo> --antigravity|--agy [--plain] [--hooks] [--write]
                                  print (or write) the Antigravity project plugin in
                                  .agents/plugins/neuromorph-nano/
  setup --project <repo> [--antigravity|--agy] --remove
                                  take out the Nano's MCP entry and hook entries again
  setup --project <repo> --status show what is installed
  connect <claude|antigravity|cursor|vscode|windsurf|generic> [--project <repo>] [--plain] [--hooks] [--dry-run] [--remove]
                                  connect a repository to one MCP client: Claude Code (.mcp.json
                                  and its approval), Antigravity (the project plugin), Cursor
                                  (.cursor/mcp.json), VS Code (.vscode/mcp.json), Windsurf
                                  (~/.codeium/windsurf/mcp_config.json) or any other stdio client
                                  (the command line and JSON, printed); other servers' entries are
                                  kept; --dry-run prints the line difference a write would make
  lab <command> ...               the lab inside the product: parameter maps (lab run pmap ...), base
                                  cartridges (lab run pmap surgery ...), job filters (lab run job-filter ...),
                                  benches and checks; lab --help lists every command
  daemon --root <repo> [--port <n>] [--port-file <file>] [serve options]
                                  serve warm on 127.0.0.1 (default port 7421; the port goes to
                                  scratch/daemon.port or --port-file) for attach and ask
  attach [--port <n>] <tool> [json] | attach [--port <n>] --stop
                                  run one tool on the warm daemon, or stop it
  ask [--port <n>] <question>...  nm_context_pack on the warm daemon, printed
  soak [--duration 4h] [--project <repo>] [--idle 60s] [--sample 30s] [--keep] [-- <serve options>]
                                  long-running daemon soak on a scratch copy: questions, idle
                                  writer time, file edits and call preemption; memory and card
                                  samples, writes outside D: checked, clean shutdown; report in
                                  runs/soak-<ms>/ (soak --help for every option; --self-test)
  coldstart --repo <path> [--question <text>] [--timeout 30m] [-- <serve options>]
                                  first run on an unseen repository: indexing, first
                                  nm_context_pack, dense index built, idle writer's first notes;
                                  report in runs/coldstart-<ms>/
  situation --project <repo> [--session <id>] [--last <n>] [--json] [--clear-prompts]
                                  print what the hooks recorded for a repository; clear the
                                  kept prompt text
  guidance --project <repo> [view|add|edit|enable|disable|done|reset|remove|undo] ...
                                  reminders, loops and steps for the coding agent: delivered
                                  through the hooks, a loop continues the agent at a stop until
                                  its condition holds (capped; setting loops); audited, undone
                                  in Audit
  session --project <repo> [--session <id>] [--list] [--json]
                                  the state of a Claude Code session (the newest by default):
                                  task, files read and edited, commands and checks, what is
                                  open, what the Nano served and what followed it
  ledger --project <repo> [--pack <id>] [--file <path>] [--json] [--as-of <ms>] [--clear]
                                  bring the evidence ledger up to date and print it: what
                                  followed each context pack, by rule and strength; --clear
                                  removes it
  outcomes --project <repo> [--recall \"<question>\"] [--import] [--json]
                                  what memory knows of verified and refuted outcomes (places
                                  that answered a question, changes that fixed a failure), each
                                  place read fresh; --import takes the ledger's outcomes into
                                  the memory journal now
  keep [--apply] [--json]         retention of the audit log: tool rows older than 28 run-days
                                  (days the system ran) go, every other row stays; shown only
                                  without --apply
  examples --project <repo> [--show <id>] [--json] [--export <file>] [--import <file>] [--clear]
                                  the training store: examples for fitting cartridges later,
                                  counted by family, domain, verdict, label, strength and split;
                                  one shown; moved between your own machines; cleared (setting
                                  examples: off, features, full)
  examples --references | --reference-add <path> --licence <id> | --reference-fitted <id> --cartridge <id>
                                  reference material for fundamentals cartridges (listed, never copied)
  library [list|show|stack|switch|export|import|bases|base|base-remove|rollback|compat|roles|role|merge] ...
                                  the cartridge library on this machine: stacks and cartridges
                                  with scope, proof and state, this machine's pieces and the
                                  rack assembled for them;
                                  changes are audited and undone by their audit id
  backup [list|add|remove|now|snapshots|preview|restore] ...  your own repositories (git, Hugging Face or a
                         folder) for backing up and rolling back memory, maps, cartridges, units and settings
  preferences [list|say|edit|remove|plug|unplug|apply] ...
                                  stated preference cartridges: your words turned into rules
                                  (enforced, checked or guidance), previewed, written as
                                  versions, plugged in or out, applied to a draft
  launch <command> [args]         run any command above from a fresh copy of the newest build in
                                  scratch/live/, so a running server never locks target/release
                                  and a rebuild is never blocked (setup's .mcp.json uses it)
  ui [--port <n>] [--dev]         the local interface on 127.0.0.1 (default port 4870); start it
                                  as `launch ui`; --dev also accepts the npm run dev server; with
                                  the setting network at view or control it also listens on this
                                  machine's private addresses for paired tablets (no encryption
                                  on the wire: anyone on the local network can read the traffic)
  pair                            make a six-digit code to pair a tablet (five minutes, once)
  dense clear --project <repo>    remove the meaning index kept on disk for a repository (its
                                  dense/vectors.bin); the next start embeds it again
  trace --root <repo> [--memory-dir <category>/<name>] [--neural] [--tentacles] <question>...
                                  print what each tentacle and drone tip did for a question
  bench [--mode plain|fast|full] [--n <count>] [--root <repo>] [-- <serve options>]
                                  time the context path per stage over tests/dev/questions.json
                                  (card memory, right place first, in the top three, in the pack)
  check [all | <crate|crate.part|cards|designs|anatomy|integrity|lock|documents|languages>...]
                                  section checks; lock compares the code lock with
                                  tests/dev/code-lock.txt; documents runs the document fixtures;
                                  with no name (or all) every section and every crate runs, each
                                  even after a failure, and the pass count is printed
  lock [--root <dir>]...          the code lock of a tree: code symbols, code chunks, crates and
                                  the reference graph, hashed (documents never enter it)
  seal <part.json>...             write a card's hash
  selftest engine [cpu|gpu|both] [--record]
                                  the engine against tests/engine-reference
  hook <Event>                    Claude Code hook intake: reads the hook's JSON on standard input,
                                  appends one record to the project's situation record, always
                                  exits 0; prints nothing but a prime when one is due (setting
                                  priming: off, quiet, on) (setup --hooks writes these entries)
  hook stop [--root <repo>]       Claude Code Stop hook: blocks while changed Rust fails cargo check
  --help, --version

serve options:
  --memory off                    no recall and nothing written to memory
  --memory-dir <category>/<name>  a separate test memory under tests/<category>/<name>
  --skip <folder>                 keep folders of that name out of the substrate (repeatable)
  --neural                        bind the embedder and reranker (dense lane on)
  --whole                         answer through the whole path (drones, tentacles, GCX, Core Brain)
  --poll <seconds>                keep the substrate in step with the studied tree
  --precision 8|16|<part>=<bits>,...   model precision (base, writer, reranker, embedder)

advanced options (research parts, off unless given):
  --precision-map <file> --dense [on|off] --tentacles --tentacle-judge heads|reranker|rrf|sum|place
  --tentacle-pose plain|focus --tentacle-score mass|density|evidence|peak --tentacle-shortlist <n>
  --tentacle-core --brain-chains --refine off|uniform|nbv --rerank-neural --rerank-withhold <p>
  --recall-meaning off --recall-rank r2 --embedder-q8 --reranker-q8 --dp4a
  --ask judge|always|off --reflexes <list> --withhold off|<p> --up-next <k> --reach on|off
  --reach-k <n> --uses on|off --look off --rerank-cap <n> --rerank-single --rerank-compare --rerank-drop <l>:<m>:<k>|off --look-gate <x>|off --fast-k <n> --segment-attention on|off --anatomy --packs off
  --half-math on|off --judge-window <tokens>|off --fast-judge-window <tokens>|off
  --learn on|off --muscle on|off --muscle-store <dir>
  --look-window on|off --full-head
  --inject-fault <part> --record --unit <id> --rack off

research harnesses and generators live in the lab tool: nano-lab --help";

type Check = (&'static str, fn() -> Result<(), String>);

/// Section checks per crate (PLAN 4b rule 10). Temporary home in this bin until the bin
/// moves to the nmapi plus bin layout (S0 decision, 2026-11-07).
fn crate_checks(name: &str) -> Option<Vec<Check>> {
    match name {
        "nmhash" => Some(nmhash::checks()),
        "nmjson" => Some(nmjson::check::checks()),
        "nmtype" => Some(nmtype::checks()),
        "nmnte" => Some(nmnte::checks()),
        "nmlex" => Some(nmlex::checks()),
        "nmdoc" => Some(nmdoc::checks()),
        "nmsubs" => Some(nmsubs::checks()),
        "nmpod" => Some(nmpod::checks()),
        "nmbrain" => Some(nmbrain::checks()),
        "nmmem" => Some(nmmem::checks()),
        "nmnpcs" => Some(nmnpcs::checks()),
        "nmffi" => Some(nmffi::checks()),
        "nmsafe" => Some(nmsafe::checks()),
        "nmtokz" => Some(nmtokz::checks()),
        "nmengn" => Some(nmengn::checks()),
        "nmlearn" => Some(nmlearn::checks()),
        "nmtool" => Some(nmtool::checks()),
        "nmmusc" => Some(nmmusc::checks()),
        "nmapi" => Some(nmapi::checks()),
        "nmcore" => Some(vec![
            ("nmcore.bench", bench::check),
            ("nmcore.launch", launch::check),
            ("nmcore.project.audit", project::check),
            ("nmcore.activity", activity::check),
            ("nmcore.ui", ui::check),
            ("nmcore.controls", controls::check),
            ("nmcore.parity", controls::check_parity),
            ("nmcore.network", network::check),
            ("nmcore.secrets", secrets::check),
            ("nmcore.connect", connect::check),
            ("nmcore.hook", hook::check),
            ("nmcore.sessions", sessions::check),
            ("nmcore.ledger", ledger::check),
            ("nmcore.outcomes", outcomes::check),
            ("nmcore.examples", examples::check),
            ("nmcore.prime", prime::check),
            ("nmcore.guidance", guidance::check),
            ("nmcore.library", library::check),
            ("nmcore.library.base", library::check_base),
            ("nmcore.library.served", library::check_served),
        ]),
        _ => None,
    }
}

/// `dense clear --project <repo>`: removes the meaning index kept on disk for that repository
/// (`dense/vectors.bin` in its memory folder; the Nano's own folder when `--project` names it),
/// logs it in the audit log and prints the rows and bytes removed.
fn dense_command(args: &[String]) -> ExitCode {
    let home = nano_root();
    let project = args.windows(2).find(|w| w.first().map(String::as_str) == Some("--project")).and_then(|w| w.get(1));
    let (Some("clear"), Some(repo)) = (args.get(1).map(String::as_str), project) else {
        eprintln!("usage: neuromorph-nano dense clear --project <repo>");
        return ExitCode::FAILURE;
    };
    if !Path::new(repo).is_dir() {
        eprintln!("dense clear: --project {repo} is not a folder");
        return ExitCode::FAILURE;
    }
    let root = nmsubs::repo::canon(Path::new(repo));
    let memory = nmtype::project::memory_dir(&home, &root);
    match controls::dense_clear(&home, &memory, &nmtype::project::label(&home, &root), "command line") {
        Ok((rows, bytes, id)) => {
            println!("dense clear: {rows} rows removed ({bytes} bytes) from {}; audit {id}", memory.join("dense").display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("dense clear: {e}");
            ExitCode::FAILURE
        }
    }
}

/// The mini's own folder, found from the running program (`nmtype::nano_root`), canonical;
/// nothing outside it.
fn nano_root() -> PathBuf {
    nmsubs::repo::canon(&nmtype::nano_root())
}

/// Verify every `crates/*/part.json`: parses, card fields present, stored hash equals the
/// recomputed one. Prints the recomputed hash on a mismatch.
fn check_cards() -> Result<usize, String> {
    let dir = nano_root().join("crates");
    let entries = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path().join("part.json")).filter(|p| p.is_file()).collect();
    paths.sort();
    for p in &paths {
        let card = nmtype::Card::load(p).map_err(|e| format!("cards: {}: {e}", p.display()))?;
        // A built sub-component's hash must equal the SHA-256 of its source file.
        for sub in card.subcomponents.iter().filter(|s| !s.hash.is_empty()) {
            let (krate, module) = sub.id.split_once('.').unwrap_or((&sub.id, ""));
            let file = dir.join(krate).join("src").join(format!("{module}.rs"));
            let got = std::fs::File::open(&file).and_then(nmhash::sha256).map_err(|e| format!("cards: {}: {e}", file.display()))?;
            if got != sub.hash {
                return Err(format!("cards: {} is stale: source hash is now {got}", sub.id));
            }
        }
    }
    Ok(paths.len())
}

/// The code lock of every tree recorded in `tests/dev/code-lock.txt` equals the lock computed
/// now. A recorded tree whose folder is missing (a disposable frozen copy) is reported and not
/// counted; a file with no tree present fails.
fn check_lock() -> Result<usize, String> {
    let home = nano_root();
    let file = home.join("tests/dev/code-lock.txt");
    let text = std::fs::read_to_string(&file).map_err(|e| format!("lock: {}: {e}", file.display()))?;
    let mut checked = 0;
    let mut records: Vec<(String, Vec<&str>)> = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        match (line.strip_prefix("root "), records.last_mut()) {
            (Some(r), _) => records.push((r.trim().to_owned(), Vec::new())),
            (None, Some(last)) => last.1.push(line.trim()),
            (None, None) => return Err(format!("lock: {}: a line before the first root", file.display())),
        }
    }
    for (root, want) in records {
        let dir = home.join(&root);
        if !dir.is_dir() {
            println!("     lock: {root} is missing, not checked");
            continue;
        }
        let got = nmsubs::lock::of_root(&dir).lines;
        if got != want {
            return Err(format!("lock: {root} differs; recorded {} now {}", want.join(" | "), got.join(" | ")));
        }
        checked += 1;
    }
    if checked == 0 {
        return Err("lock: no recorded tree is present".into());
    }
    Ok(checked)
}

/// `INTEGRITY.md`: every listed file, and every file a listed `MANIFEST.sha256` names, still has
/// its recorded SHA-256. Rows read are table rows whose first two cells are a back-quoted path and
/// a back-quoted 64-digit hash. A path is relative to the Nano folder, except one starting with
/// `models/`, which is relative to the models folder (`nmtype::models_dir`: `NANO_MODELS`, else
/// `designs/models.json`), so the check follows the models wherever they are kept. Every mismatch
/// is reported, not only the first.
fn check_integrity() -> Result<usize, String> {
    let root = nano_root();
    let text = std::fs::read_to_string(root.join("INTEGRITY.md")).map_err(|e| format!("integrity: INTEGRITY.md: {e}"))?;
    let hash = |p: &Path| std::fs::File::open(p).and_then(nmhash::sha256).map_err(|e| format!("{}: {e}", p.display()));
    let quoted = |c: &str| c.strip_prefix('`').and_then(|s| s.strip_suffix('`')).map(str::to_owned);
    let is_hash = |h: &str| h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit());
    let (mut n, mut bad, mut unread) = (0_usize, Vec::new(), Vec::new());
    let mut verify = |shown: String, file: &Path, want: &str, n: &mut usize| {
        *n += 1;
        match hash(file) {
            Ok(got) if got == want => {}
            Ok(got) => bad.push(format!("{shown}: now {got}")),
            Err(e) => bad.push(e),
        }
    };
    for line in text.lines() {
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        // A row whose first cell is a quoted path must have a clean quoted SHA-256 as its second
        // cell (an optional note column may follow); anything else is a fault, never a silently
        // skipped row (2026-11-09).
        let Some(p) = cells.get(1).and_then(|c| quoted(c)) else {
            continue;
        };
        let h = match cells.get(2).and_then(|c| quoted(c)) {
            Some(h) if is_hash(&h) => h,
            _ => {
                unread.push(format!(
                    "unreadable row (hash cell must be exactly `sha256`; put notes in a third column or under the table): {}",
                    line.trim()
                ));
                continue;
            }
        };
        let file = integrity_path(&root, &p);
        verify(p.clone(), &file, &h, &mut n);
        if p.ends_with("MANIFEST.sha256") {
            let dir = file.parent().map_or_else(|| root.clone(), Path::to_path_buf);
            for l in std::fs::read_to_string(&file).unwrap_or_default().lines() {
                match l.split_once("  ") {
                    Some((want, name)) if is_hash(want.trim()) => {
                        verify(format!("{p} -> {name}"), &dir.join(name.trim()), want.trim(), &mut n)
                    }
                    _ if l.trim().is_empty() => {}
                    _ => unread.push(format!("{p}: unreadable manifest line: {l}")),
                }
            }
        }
    }
    if n == 0 {
        return Err("integrity: no rows found in INTEGRITY.md".into());
    }
    bad.extend(unread);
    if bad.is_empty() { Ok(n) } else { Err(format!("integrity: {} of {n} files differ:\n  {}", bad.len(), bad.join("\n  "))) }
}

/// Where an `INTEGRITY.md` row's file is: `models/<rest>` under the models folder, any other
/// path under the Nano folder `home`.
fn integrity_path(home: &Path, row: &str) -> PathBuf {
    match row.strip_prefix(nmnpcs::acquire::MODELS_PREFIX) {
        Some(rest) => nmtype::models_dir(home).join(rest),
        None => home.join(row),
    }
}

/// `designs/anatomy.json` passes the Integrity Validator with every node resolved in the
/// registry loaded from `crates/*/part.json`.
fn check_anatomy() -> Result<usize, String> {
    let root = nano_root();
    let bytes = std::fs::read(root.join("designs/anatomy.json")).map_err(|e| format!("anatomy: {e}"))?;
    let v = nmjson::parse(&bytes).map_err(|e| format!("anatomy: {e}"))?;
    let g = nmnte::Graph::from_value(&v).map_err(|e| format!("anatomy: {e}"))?;
    let reg = nmnte::Registry::load(&root.join("crates")).map_err(|e| format!("anatomy: {e}"))?;
    nmnte::integrity::validate(&g, Some(&reg)).map_err(|e| format!("anatomy: {e}"))?;
    let (path, run) = nmapi::forage::forage_assembly(&root).map_err(|e| format!("anatomy: {e}"))?;
    nmnte::integrity::validate(&run, Some(&reg)).map_err(|e| format!("anatomy: {}: {e}", path.display()))?;
    Ok(g.nodes.len() + run.nodes.len())
}

/// The path that answers, for the audit log: flow, whole, neural, dense or plain.
fn serve_mode(args: &[String]) -> &'static str {
    let has = |f: &str| args.iter().any(|a| a == f);
    if flow_on(args) {
        "flow"
    } else if has("--whole") {
        "whole"
    } else if has("--neural") {
        "neural"
    } else if has("--dense") || has("--tentacles") {
        "dense"
    } else {
        "plain"
    }
}

/// Since 2026-11-17 the served plain path runs the meaning lane by default when its vectors can be
/// kept on disk (memory on, `dense_cache` not off) and no `--dense`, `--neural` or `--whole` says
/// otherwise: the material is embedded once in the background and kept, and each call encodes only
/// the question (tuning fold of the large set: top 3 27 to 45 of 74, 0.01 s less per call with the
/// question on the card; base cartridge builder's record of that date). It starts after the tools
/// answer, so the first call does not wait for the embedder to bind.
fn dense_by_default(org: &nmapi::forage::Organism, args: &[String]) -> bool {
    !args.iter().any(|a| a == "--whole" || a == "--neural" || a == "--dense") && org.memory_on && nmsubs::dense::store::cache_on(&org.home)
}

/// Whether `serve` should pre-warm `WholePath`'s `Keep` intake in the background after tools install.
fn should_warm_keep(args: &[String]) -> bool {
    flow_on(args) || args.iter().any(|a| a == "--whole")
}

/// Whether the one flow answers (`--flow`, not followed by `off`).
fn flow_on(args: &[String]) -> bool {
    args.windows(2).any(|w| w.first().map(String::as_str) == Some("--flow") && w.get(1).map(String::as_str) != Some("off"))
        || args.last().map(String::as_str) == Some("--flow")
}

/// The served default is the one flow (2026-11-18, Luis Jake Gabriel III through the lead: level with
/// the plain path on accuracy over eight tuning sets and far better on decoys): `serve` answers
/// through it unless `--flow off`, `--whole` or `--neural` says otherwise; `--flow off` serves the
/// plain path.
fn with_flow_default(args: &[String]) -> Vec<String> {
    let mut out = args.to_vec();
    if !args.iter().any(|a| a == "--flow" || a == "--whole" || a == "--neural") {
        out.push("--flow".to_owned());
    }
    out
}

/// `neuromorph-nano serve --root <repo>`: the Neuromorph over MCP stdio for Claude Code. The
/// studied repository is an argument; without it the server refuses to start.
fn serve(args: &[String]) -> ExitCode {
    let args_flow = with_flow_default(args);
    let args = args_flow.as_slice();
    let Some(root) = args.iter().position(|a| a == "--root").and_then(|i| args.get(i + 1)) else {
        eprintln!("usage: neuromorph-nano serve --root <repository to study>");
        return ExitCode::FAILURE;
    };
    // Any repository (Nano build plan E1, 2026-11-15): the root is taken as its absolute,
    // canonical path, so its project memory and audit rows always name the same folder.
    if !Path::new(root).is_dir() {
        eprintln!("serve: --root {root} is not a folder");
        return ExitCode::FAILURE;
    }
    let root_abs = nmsubs::repo::canon(Path::new(root));
    // Memory off by the flag, or by the saved control (Neuromorph Nano, 2026-11-15).
    let memory_on =
        !args.windows(2).any(|w| w.first().map(String::as_str) == Some("--memory") && w.get(1).map(String::as_str) == Some("off"))
            && !controls::memory_off(&nano_root());
    // `--memory-dir NAME` keeps a separate, named memory under `tests/NAME`, where NAME starts
    // with its category (`memory-system/<name>` for tests of the memory system,
    // `claude-code-sessions/<name>`, `capability-gates/<cN-...>/<name>`, `mcp-launcher/<name>`), so
    // test sessions never write into the organism's own memory. `memory/` is the hierarchical
    // memory system; `tests/` holds tests by category; `scratch/` only disposable work (Luis Jake Gabriel III,
    // 2026-11-08).
    // A name without a category: the locked stage harnesses pass `s<n>-<label>-<ms>`, which
    // goes to `stage-gates/s<n>/`; any other bare name is refused (2026-11-09).
    // Without `--memory-dir`, the Nano's own folder keeps `memory/` and any other repository its
    // own folder `memory/projects/<name>-<8 hex>/` (E1, 2026-11-15; `nmtype::project`).
    let memory_dir = match args.iter().position(|a| a == "--memory-dir").and_then(|i| args.get(i + 1)) {
        None => nmtype::project::memory_dir(&nano_root(), &root_abs),
        Some(n) if n.contains('/') => nmtype::tests_dir(n),
        Some(n) => match stage_of(n) {
            Some(stage) => nmtype::tests_dir(&format!("stage-gates/{stage}/{n}")),
            None => {
                eprintln!("serve: --memory-dir needs <category>/<name> (categories in tests/README.md), got {n}");
                return ExitCode::FAILURE;
            }
        },
    };
    // `--skip <folder>` (repeatable) keeps folders of that name out of the substrate, for
    // example `runs` while a test's answer material sits there (S12 part C).
    let mut skip = vec!["neuromorph-nano".to_owned()];
    skip.extend(args.windows(2).filter(|w| w.first().map(String::as_str) == Some("--skip")).filter_map(|w| w.get(1).cloned()));
    // 2026-11-27 (MCP stability fix): one engine per folder and options. A second `serve` (a
    // second coding-agent session on the same project) attaches to the engine the first holds
    // and forwards its client's lines, instead of opening a second engine with its own four
    // MicroModels on the same card (`nmapi::engine`). The daemon mode has its own listener.
    let holder = if args.iter().any(|a| a == "--daemon") {
        None
    } else {
        match nmapi::engine::claim(&nano_root(), &nmapi::engine::key(&root_abs, args)) {
            Ok(nmapi::engine::Part::Attach { stream, port, pid }) => {
                eprintln!("engine: attached to the shared engine of process {pid} on 127.0.0.1:{port} (same folder and options); this process forwards its client's calls and opens no engine of its own");
                return match nmapi::engine::attach(stream, std::io::stdin().lock(), std::io::stdout()) {
                    Ok(()) => ExitCode::SUCCESS,
                    Err(e) => {
                        eprintln!("engine: {e}; this client's session ended (a new session starts or attaches to an engine)");
                        ExitCode::FAILURE
                    }
                };
            }
            Ok(nmapi::engine::Part::Own(h)) => Some(h),
            Err(e) => {
                eprintln!("engine: not shared ({e}); this process holds its own engine");
                None
            }
        }
    };
    // Audit of real work (E2, 2026-11-15): every tool call appends one row to memory/audit.jsonl.
    project::start_audit(&nano_root(), &root_abs, memory_on, serve_mode(args), "serve");
    examples::start(&nano_root(), &root_abs, &memory_dir, memory_on);
    // The heavy start (reading the material, the library and hardware stacks, the models the flags
    // bind, session open) runs in the background, so the MCP handshake and the tool list answer at
    // once; a tool call that needs the organism waits for it. Each step's time goes to stderr
    // (stdout carries only JSON-RPC).
    nmapi::tools::starting();
    let start = {
        let (args, root_abs, memory_dir) = (args.to_vec(), root_abs.clone(), memory_dir.clone());
        std::thread::spawn(move || {
            let clock = std::time::Instant::now();
            let mut steps: Vec<(&str, f64)> = Vec::new();
            let lap = |name: &'static str, steps: &mut Vec<(&str, f64)>| steps.push((name, clock.elapsed().as_secs_f64()));
            if memory_on {
                project::note_project(&nano_root(), &root_abs, &memory_dir);
            }
            let org = match nmapi::forage::Organism::open_with(&nano_root(), &root_abs, &skip, memory_on, &memory_dir) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("open: {e}");
                    nmapi::tools::start_failed(&format!("open: {e}"));
                    return None;
                }
            };
            lap("open (material read, memory opened)", &mut steps);
            // The plain path's drones take the material in beside the rest of the start, so the
            // first question does not pay for it (guide WP8); the whole path has its own readers.
            let drones = (!args.iter().any(|a| a == "--whole") && !flow_on(&args)).then(|| org.warm_drones());
            // This process serves: its library start is recorded and it follows stack switches.
            library::SERVING.store(true, std::sync::atomic::Ordering::Relaxed);
            if let Err(e) = configure(&org, &args) {
                eprintln!("{e}");
                nmapi::tools::start_failed(&e);
                return None;
            }
            lap("configure (cartridge library, hardware stacks, flags)", &mut steps);
            eprintln!("session: {}", nmapi::session::open(&org).encode().unwrap_or_default());
            eprintln!("memory: {}", if memory_on { memory_dir.display().to_string() } else { "off".to_owned() });
            lap("session open", &mut steps);
            if let Err(e) = nmapi::tools::install(org) {
                eprintln!("{}: {}", e.code, e.message);
                return None;
            }
            // `--muscle on`: G3's cycle runs at idle on the evidence ledger's verdicts.
            if nmapi::g3::serve_idle() {
                eprintln!("muscle: idle cycle on");
            }
            // Roadmap Phase 2: the unified learning cycle at idle (G3 priors, the preference
            // cartridge, the engine), when enough verified examples have come in since the last.
            if crate::examples::start_idle_learning(nano_root()) {
                eprintln!("learning: idle cycle on");
            }
            // The writer at idle (on by default; `--writer-idle off` or `NANO_WRITER_IDLE=off` turns
            // it off): it loads only at idle, writes notes, and lets go of the card
            // (`nmapi::writer_idle`).
            let writer_on = args.windows(2).any(|w| w.first().map(String::as_str) == Some("--writer-idle") && w.get(1).map(String::as_str) == Some("on"));
            let writer_off = args.windows(2).any(|w| w.first().map(String::as_str) == Some("--writer-idle") && w.get(1).map(String::as_str) == Some("off"));
            if (writer_on || nmapi::writer_idle::switched_on()) && !writer_off && nmapi::writer_idle::start(nano_root()) {

                eprintln!("writer: idle notes on");
            }
            // 2026-11-27 (Luis Jake Gabriel III: all four MicroModels load at start by default, as
            // many as the card holds): the Reader, Embedder, Reranker and Writer load in the
            // background (`nmapi::preload`); a call that needs one still loading waits for it only.
            nmapi::preload::start(nano_root());
            let mut prev = 0.0;
            let line: Vec<String> = steps
                .iter()
                .map(|(n, at)| {
                    let s = format!("{n} {:.2} s", at - prev);
                    prev = *at;
                    s
                })
                .collect();
            eprintln!("start in the background: {}; ready at {:.2} s", line.join(", "), prev);
            // The plain path's meaning lane binds and starts after the tools answer.
            if let Some(o) = nmapi::tools::installed().filter(|o| dense_by_default(o, &args)) {
                std::thread::spawn(move || {
                    let line = nmapi::neural::bind_embedder(o).and_then(|_| nmapi::neural::start_dense(o));
                    match line {
                        Ok(m) => eprintln!("dense (plain path default): {m}; at {:.2} s", clock.elapsed().as_secs_f64()),
                        Err(e) => eprintln!("dense (plain path default): not started ({}); the path runs without it", e.message),
                    }
                });
            }
            // The one flow's or whole path's drones take the material in after the tools answer, so no call pays for it.
            if should_warm_keep(&args)
                && let Some(o) = nmapi::tools::installed()
                && let Some(w) = o.whole.read().ok().and_then(|w| w.clone())
            {
                let label = if flow_on(&args) { "flow" } else { "whole" };
                std::thread::spawn(move || match w.warm_keep(&o.sub()) {
                    Ok(n) => eprintln!("{label}: drones took in {n} pieces; done at {:.2} s", clock.elapsed().as_secs_f64()),
                    Err(e) => eprintln!("{label}: drones' intake at start failed ({}); the first call takes it in", e.message),
                });
            }
            // The drones' intake goes on after the tools answer; its line follows when it ends.
            if let Some(h) = drones {
                std::thread::spawn(move || {
                    if let Ok(line) = h.join() {
                        eprintln!("{line}; done at {:.2} s", clock.elapsed().as_secs_f64());
                    }
                });
            }
            // The substrate is kept in step with the studied tree (S10, guide 17.3). Roadmap Phase 1
            // item 2 (2026-11-24): on by default every 5 s (a metadata fingerprint makes an
            // unchanged tree cheap); `--poll <seconds>` sets the period, `--poll off` turns it off.
            let poll = args.iter().position(|a| a == "--poll").and_then(|i| args.get(i + 1));
            let secs = match poll.map(String::as_str) {
                Some("off" | "0") => None,
                Some(v) => v.parse::<f64>().ok().filter(|s| *s > 0.0),
                None => Some(5.0),
            };
            secs.and_then(|s| nmapi::tools::installed().map(|o| nmapi::delta::start(o, std::time::Duration::from_secs_f64(s))))
        })
    };
    let api = nmapi::tools::api().observed(examples::observe);
    nmapi::mcp::on_start(project::begin);
    // Roadmap Phase 4.5 (2026-11-24): the interface's routes back the parity tools (`nm_cartridges`,
    // `nm_learning`, `nm_comprehension`, `nm_settings`), so a tool runs what a page runs.
    nmapi::parity::lend_routes(ui::lent_route);
    let daemon_port: Option<u16> = args.iter().position(|a| a == "--daemon").map(|i| {
        args.iter()
            .position(|a| a == "--port")
            .and_then(|pi| args.get(pi + 1))
            .and_then(|v| v.parse::<u16>().ok())
            .or_else(|| args.get(i + 1).and_then(|v| v.parse::<u16>().ok()))
            .unwrap_or(7421)
    });
    let served = if let Some(port) = daemon_port {
        // `--port-file <path>`: where the daemon writes its port (a harness's own file, so a
        // daemon under test never touches the everyday `scratch/daemon.port`).
        let port_file = args
            .windows(2)
            .find(|w| w.first().map(String::as_str) == Some("--port-file"))
            .and_then(|w| w.get(1))
            .map_or_else(|| nano_root().join("scratch/daemon.port"), PathBuf::from);
        run_warm_daemon(&api, port, &port_file)
    } else {
        let stdin = std::io::stdin();
        let stop = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            if let Some(h) = holder.as_ref() {
                nmapi::mcp::set_engine_port(h.port);
                let (api, stop) = (&api, &stop);
                scope.spawn(move || nmapi::engine::listen(api, h, stop));
            }
            let served = nmapi::mcp::run(&api, stdin.lock(), std::io::stdout());
            if nmapi::mcp::attached() > 0 {
                eprintln!("engine: this process's own client left; serving the {} attached client(s) until they leave", nmapi::mcp::attached());
            }
            stop.store(true, std::sync::atomic::Ordering::SeqCst);
            served
        })
    };
    drop(holder);
    // Stop any background idle writer immediately before waiting for start or closing the GPU gate.
    nmapi::writer_idle::stop();
    // The input ended: the start finishes before the session closes.
    let poller = start.join().ok().flatten();
    live::end();
    if let Some(p) = poller {
        p.stop();
    }
    if let Some(org) = nmapi::tools::installed() {
        nmapi::session::close(org);
    }
    let code = match served {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("{e}");
            1
        }
    };
    bench::shutdown(code)
}

/// Persistent warm local daemon over loopback TCP (`127.0.0.1:<port>`): keeps the Organism,
/// substrate, dense lane, and all 4 base cartridges resident in RAM/VRAM so CLI commands
/// (`neuromorph-nano ask` / `neuromorph-nano attach`) connect in `< 5 ms` with zero cold start.
fn run_warm_daemon(api: &nmapi::Api, port: u16, port_file: &Path) -> std::io::Result<()> {
    use std::io::{BufRead as _, Read as _, Write as _};
    let listener = std::net::TcpListener::bind(("127.0.0.1", port))?;
    let actual_port = listener.local_addr().map_or(port, |a| a.port());
    if let Some(p) = port_file.parent() {
        let _ = std::fs::create_dir_all(p);
    }
    let _ = std::fs::write(port_file, format!("{actual_port}\n"));
    eprintln!("daemon: listening on 127.0.0.1:{actual_port} (attach with `neuromorph-nano ask` or `attach`)");
    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        let Ok(writer) = stream.try_clone() else { continue };
        let mut reader = std::io::BufReader::new(stream);
        // The first line as bytes (a line that is not UTF-8 was dropped silently with `read_line`).
        let mut first_raw = Vec::new();
        match reader.read_until(b'\n', &mut first_raw) {
            Ok(0) => continue,
            Ok(_) => {}
            Err(e) => {
                eprintln!("daemon: a connection's first line could not be read: {e}");
                continue;
            }
        }
        let first = String::from_utf8_lossy(&first_raw).into_owned();
        if first.trim() == "STOP" {
            let mut w = writer;
            let _ = writeln!(w, "{{\"ok\":true,\"stopped\":true}}");
            let _ = std::fs::remove_file(port_file);
            eprintln!("daemon: stop requested");
            break;
        }
        let chained = std::io::Cursor::new(first.into_bytes()).chain(reader);
        // 2026-11-24: a connection that ends with an error is logged (it was dropped silently;
        // the 2-hour soak saw one call reset with no trace in the daemon's log).
        if let Err(e) = nmapi::mcp::run(api, std::io::BufReader::new(chained), &writer) {
            eprintln!("daemon: a connection ended with an error: {e}");
        }
        // 2026-11-25: the sending half is closed explicitly once the input has ended, so the
        // client sees an orderly end of stream before the socket handles are released.
        if let Err(e) = writer.shutdown(std::net::Shutdown::Write) {
            eprintln!("daemon: a connection's sending half could not be closed: {e}");
        }
    }
    let _ = std::fs::remove_file(port_file);
    Ok(())
}

/// Fast CLI attach client (`neuromorph-nano attach [--port P] <tool> [json]` or
/// `neuromorph-nano ask [--port P] "<question>"`, or `attach --stop`).
fn daemon_attach(args: &[String], is_ask: bool) -> ExitCode {
    use std::io::{BufRead as _, Write as _};
    let t0 = std::time::Instant::now();
    let mut port: Option<u16> = None;
    let mut rest: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--port"
            && let Some(p) = args.get(i + 1).and_then(|v| v.parse::<u16>().ok())
        {
            port = Some(p);
            i += 2;
            continue;
        }
        rest.push(args[i].as_str());
        i += 1;
    }
    let port = port.unwrap_or_else(|| {
        std::fs::read_to_string(nano_root().join("scratch/daemon.port"))
            .ok()
            .and_then(|s| s.trim().parse::<u16>().ok())
            .unwrap_or(7421)
    });
    let Ok(mut stream) = std::net::TcpStream::connect(("127.0.0.1", port)) else {
        eprintln!("daemon attach: no warm daemon listening on 127.0.0.1:{port} (start with `neuromorph-nano daemon --root <repo>`)");
        return ExitCode::FAILURE;
    };
    let connect_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if rest.first().copied() == Some("--stop") {
        let _ = writeln!(stream, "STOP");
        let _ = stream.flush();
        let mut resp = String::new();
        let _ = std::io::BufReader::new(stream).read_line(&mut resp);
        println!("{}", resp.trim());
        return ExitCode::SUCCESS;
    }
    let (tool, arguments_json) = if is_ask {
        if rest.is_empty() {
            eprintln!("usage: neuromorph-nano ask [--port P] <question>");
            return ExitCode::FAILURE;
        }
        let q = rest.join(" ");
        let mut m = std::collections::BTreeMap::new();
        m.insert("question".to_owned(), nmjson::Value::String(q));
        ("nm_context_pack", nmjson::Value::Object(m).encode().unwrap_or_else(|_| "{}".to_owned()))
    } else {
        let Some(&t) = rest.first() else {
            eprintln!("usage: neuromorph-nano attach [--port P] <tool_name> [json_args] | --stop");
            return ExitCode::FAILURE;
        };
        let j = rest.get(1).copied().unwrap_or("{}");
        (t, j.to_owned())
    };
    let init = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\"clientInfo\":{\"name\":\"nano-attach\",\"version\":\"1.0\"}}}";
    let call = format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{{\"name\":\"{tool}\",\"arguments\":{arguments_json}}}}}"
    );
    if writeln!(stream, "{init}\n{call}").is_err() || stream.flush().is_err() {
        eprintln!("daemon attach: write failed");
        return ExitCode::FAILURE;
    }
    let _ = stream.shutdown(std::net::Shutdown::Write);
    let reader = std::io::BufReader::new(stream);
    let mut last_line = String::new();
    for line in reader.lines().map_while(Result::ok) {
        if line.contains("\"id\":2") {
            last_line = line;
        }
    }
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if last_line.is_empty() {
        eprintln!("daemon attach: empty response from 127.0.0.1:{port}");
        return ExitCode::FAILURE;
    }
    if let Ok(v) = nmjson::parse(last_line.as_bytes())
        && let Ok(arr) = v.get("result").and_then(|r| r.get("content")).and_then(nmjson::Value::array)
        && let Some(text) = arr.first().and_then(|x| x.get("text").ok()).and_then(|t| t.string().ok())
    {
        println!("{text}");
    } else {
        println!("{last_line}");
    }
    eprintln!("daemon attach: connect {connect_ms:.1} ms, total {total_ms:.1} ms (127.0.0.1:{port})");
    ExitCode::SUCCESS
}

/// An on or off switch: `Some(true)` for `<flag>` alone or `<flag> on`, `Some(false)` for
/// `<flag> off`, `None` when the flag is absent.
fn switch(args: &[String], flag: &str) -> Option<bool> {
    let i = args.iter().position(|a| a == flag)?;
    Some(args.get(i + 1).map(String::as_str) != Some("off"))
}

/// Applies the `serve` options that configure the organism (models, paths, switches) to `org`,
/// so `serve` and `bench` run the same served configuration.
/// # Errors
/// The first option that is malformed or fails to bind, as one line.
fn configure(org: &nmapi::forage::Organism, args: &[String]) -> Result<(), String> {
    // The cartridge rack for this machine: its served operating numbers become options not given
    // on the command line (`--rack off` leaves it out); shadow cartridges change nothing.
    let with_rack = library::apply(&org.home, args);
    let args = with_rack.as_slice();
    // Reranker qualification arms (2026-11-10): `--rerank-neural` lets the bound neural
    // reranker order (an experiment arm until a qualification run sets it by default);
    // `--rerank-withhold <p>` withholds a pack whose best neural score is below p.
    let after_bind = |org: &nmapi::forage::Organism| {
        // 2026-11-12: recall by meaning is on by default (it acts only with both engines bound);
        // `--recall-meaning off` switches it off.
        if args.windows(2).any(|w| w.first().map(String::as_str) == Some("--recall-meaning") && w.get(1).map(String::as_str) == Some("off"))
        {
            org.recall_meaning.store(false, std::sync::atomic::Ordering::SeqCst);
        }
        // 2026-11-13: `--recall-rank r2` orders word recall by memory revision 2 (floor kept).
        if args.windows(2).any(|w| w.first().map(String::as_str) == Some("--recall-rank") && w.get(1).map(String::as_str) == Some("r2")) {
            org.recall_r2.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        if args.iter().any(|a| a == "--rerank-neural") {
            org.slots.set_qualified("reranker", true);
        }
        // Withholding is on by default once the reranker is bound (R1); an explicit value
        // overrides it, and 0 switches it off.
        if let Some(p) =
            args.iter().position(|a| a == "--rerank-withhold").and_then(|i| args.get(i + 1)).and_then(|x| x.parse::<f32>().ok())
        {
            org.slots.set_withhold(p);
        }
    };
    // `--refine off|uniform|nbv` sets the second coordination round (CM1, 2026-11-09).
    if let Some(m) = args.iter().position(|a| a == "--refine").and_then(|i| args.get(i + 1)) {
        if !["off", "uniform", "nbv"].contains(&m.as_str()) {
            return Err(format!("serve: --refine takes off, uniform or nbv, got {m}"));
        }
        if let Ok(mut r) = org.refine.write() {
            r.clone_from(m);
        }
    }
    // `--learn on`: the 3-level Continuous Learning Engine & Supervisor (`nmlearn::engine`).
    if (switch(args, "--learn") == Some(true) || (switch(args, "--learn").is_none() && nmapi::whole::learning_control(&org.home)))
        && let Ok(m) = org.memory.lock()
    {
        let eng = nmlearn::engine::ContinuousLearningEngine::open(&m.dir);
        eprintln!(
            "learn: continuous learning engine on (version {}, {} experiences, {} patterns, {} policies)",
            eng.version,
            eng.episodic.len(),
            eng.patterns.len(),
            eng.policies.len()
        );
    }
    // G3 and muscle memory: on by default (controlled by `muscle` in settings.json; `--muscle off`
    // or `--muscle on` overrides it). Separate from `--learn` (2026-11-18).
    let muscle_on = match switch(args, "--muscle") {
        Some(v) => v,
        None => nmapi::whole::muscle_control(&org.home),
    };
    if muscle_on {
        // `--muscle-store DIR`: a separate store (tests of real service); the live store otherwise.
        let store = args.iter().position(|a| a == "--muscle-store").and_then(|i| args.get(i + 1)).map(PathBuf::from);
        let n = nmapi::g3::enable(&org.home, store.as_deref()).map_err(|e| format!("muscle: {e}"))?;
        eprintln!("muscle: on, {n} muscle-memory packs loaded");
    }
    // `--neural` binds the embedder and reranker slots to the base population (S6).
    if args.iter().any(|a| a == "--neural") {
        match nmapi::neural::bind(org) {
            Ok(m) => eprintln!("neural: {m}"),
            Err(e) => {
                return Err(format!("neural: {e}"));
            }
        }
        after_bind(org);
    }
    // `--tentacles` (2026-11-10, GCX build step 1): the tentacle pass, eight lens windows read by
    // the drone heads of the base model; off unless the flag is given.
    match nmapi::tentacle_pass::from_args(args) {
        Ok(Some(s)) => eprintln!("{}", nmapi::tentacle_pass::enable(org, s)),
        Ok(None) => {}
        Err(e) => {
            return Err(format!("serve: {e}"));
        }
    }
    // `--embedder-q8` (2026-11-13): the embedder bound at 8 bits (its closeness check passed).
    // `--reranker-q8` (2026-11-13): the reranker bound at 8 bits (cartridge step F closeness passed).
    // `--half-math` (2026-11-13, speed): dense binary16 linears with half2 products.
    // `--dp4a` (2026-11-13, speed): weight-only int8 linears with 8-bit activations and dot products.
    if args.iter().any(|a| a == "--dp4a") {
        nmengn::gpu::DP4A.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    // Half math is on for the served models (speed pass 2, P2.2); `--half-math off` keeps the
    // binary32 products. The reference path (`selftest engine`) never passes here.
    nmengn::gpu::HALF_MATH.store(switch(args, "--half-math").unwrap_or(true), std::sync::atomic::Ordering::Relaxed);
    // `--full-head`: the second look's tails read the base model's whole output head and the
    // whole path stages it (comparison only; the default reads the yes and no rows).
    if args.iter().any(|a| a == "--full-head") {
        nmapi::joint_read::FULL_HEAD.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    // `--rerank-drop <layer>:<margin>:<keep>|off`: early drop inside the batched reranker.
    if let Some(v) = args.windows(2).find(|w| w.first().map(String::as_str) == Some("--rerank-drop")).and_then(|w| w.get(1)) {
        let d = if v == "off" {
            None
        } else {
            let p: Vec<&str> = v.split(':').collect();
            match (p.first().and_then(|x| x.parse().ok()), p.get(1).and_then(|x| x.parse().ok()), p.get(2).and_then(|x| x.parse().ok())) {
                (Some(layer), Some(margin), Some(keep)) if p.len() == 3 => Some(nmengn::roles::EarlyDrop { layer, margin, keep }),
                _ => return Err(format!("serve: --rerank-drop takes off or <layer>:<margin log-odds>:<keep>, got {v}")),
            }
        };
        if let Ok(mut g) = nmengn::roles::RERANK_DROP.lock() {
            *g = d;
        }
        nmengn::roles::RERANK_DROP_CLI.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    // `--segment-attention on|off`: packed sequences use the one-key-per-lane attention kernel; on
    // for the served models (speed pass 2, P2.3, with half math), off keeps the prefill kernels.
    nmengn::gpu::SEGMENT_ATTENTION.store(switch(args, "--segment-attention").unwrap_or(true), std::sync::atomic::Ordering::Relaxed);
    // `--rerank-single`: the reranker scores one document at a time (the earlier path, for comparison).
    if args.iter().any(|a| a == "--rerank-single") {
        nmengn::roles::RERANK_BATCH.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    if args.iter().any(|a| a == "--reranker-q8") {
        nmapi::neural::RERANK_Q8.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if args.iter().any(|a| a == "--embedder-q8") {
        nmapi::neural::EMBED_Q8.store(true, std::sync::atomic::Ordering::Relaxed);
        eprintln!("embedder: 8 bits");
    }
    // `--whole` (2026-11-12): questions answered through the whole path (drones, tentacles, the
    // Globalized Context Stack, the Core Brain's reranker order and second look); off unless given.
    // `--flow`: every call through the one flow (`whole::Mode::Flow`: the tentacles as a team, G1,
    // G2 choosing by coverage on the settle chance, what G2 sends up as the pack); off unless given.
    let flow = flow_on(args);
    if flow {
        nmapi::drone_path::set_g2(true);
        nmapi::drone_path::set_settle(true);
    }
    if (flow || args.iter().any(|a| a == "--whole"))
        && let Ok(mut w) = org.whole.write()
    {
        let mut path = nmapi::whole::WholePath::new(&org.home);
        if org.max_run_ms > 0 {
            path.budget_s = f64::from(org.max_run_ms) / 1000.0;
        }
        if flow {
            path.default_mode = Some(nmapi::whole::Mode::Flow);
        }
        // `--ask judge|always|off` (2026-11-12): when the Core Brain asks for more context.
        match nmapi::ask::Mode::from_args(args) {
            Ok(m) => path.ask = m,
            Err(e) => {
                return Err(format!("serve: {e}"));
            }
        }
        // `--reflexes R2,R3,R4,...` (2026-11-13): reflexes G2 installs below the Core Brain
        // (`nmbrain::reflex` short forms); none unless given.
        if let Some(list) = args.windows(2).find(|w| w.first().map(String::as_str) == Some("--reflexes")).and_then(|w| w.get(1)) {
            for r in list.split(',').filter(|x| !x.is_empty()) {
                match nmbrain::reflex::Installed::parse(r) {
                    Ok(x) => {
                        if let Ok(mut set) = path.reflexes.lock() {
                            set.install(x);
                        }
                    }
                    Err(e) => {
                        return Err(format!("serve: {e}"));
                    }
                }
            }
        }
        // `--withhold off|<p>` (2026-11-13): the decoy cut on the reranker's best score;
        // default `nmapi::whole::WITHHOLD_BELOW`.
        if let Some(v) = args.windows(2).find(|w| w.first().map(String::as_str) == Some("--withhold")).and_then(|w| w.get(1)) {
            path.withhold_below = if v == "off" {
                None
            } else {
                match v.parse::<f32>() {
                    Ok(p) if (0.0..=1.0).contains(&p) => Some(p),
                    _ => {
                        return Err(format!("serve: --withhold takes off or a probability, got {v}"));
                    }
                }
            };
        }
        // `--up-next <k>` (2026-11-13): G2's standing I11 order, k more pieces up per question.
        if let Some(v) = args.windows(2).find(|w| w.first().map(String::as_str) == Some("--up-next")).and_then(|w| w.get(1)) {
            match v.parse::<usize>() {
                Ok(k) => path.up_next = k,
                Err(_) => {
                    return Err(format!("serve: --up-next takes a count, got {v}"));
                }
            }
        }
        // `--reach off` (2026-11-13): no within-file reach around the first place (on by default).
        if let Some(v) = args.windows(2).find(|w| w.first().map(String::as_str) == Some("--reach")).and_then(|w| w.get(1)) {
            match v.as_str() {
                "off" => path.reach = false,
                "on" => path.reach = true,
                _ => {
                    return Err(format!("serve: --reach takes on or off, got {v}"));
                }
            }
        }
        // `--uses off` (2026-11-13): no use sites for questions asking for callers (on by default).
        if let Some(v) = args.windows(2).find(|w| w.first().map(String::as_str) == Some("--uses")).and_then(|w| w.get(1)) {
            match v.as_str() {
                "off" => path.uses = false,
                "on" => path.uses = true,
                _ => {
                    return Err(format!("serve: --uses takes on or off, got {v}"));
                }
            }
        }
        // `--inject-fault <part>` (2026-11-13, reliability run only): that part fails every question.
        if let Some(v) = args.windows(2).find(|w| w.first().map(String::as_str) == Some("--inject-fault")).and_then(|w| w.get(1)) {
            if !["reranker", "look", "reach", "uses", "ask"].contains(&v.as_str()) {
                return Err(format!("serve: --inject-fault takes reranker, look, reach, uses or ask, got {v}"));
            }
            path.inject = Some(v.clone());
        }
        // `--rerank-cap <n>` (2026-11-13, speed): tokens of each piece the reranker reads.
        if let Some(v) = args.windows(2).find(|w| w.first().map(String::as_str) == Some("--rerank-cap")).and_then(|w| w.get(1)) {
            match v.parse::<usize>() {
                Ok(n) if n >= 16 => path.rerank_cap = Some(n),
                _ => {
                    return Err(format!("serve: --rerank-cap takes a token count of at least 16, got {v}"));
                }
            }
        }
        // `--look off` (2026-11-13): no second look; `--anatomy` (2026-11-13, live anatomy): the
        // arrangement per question kind from `designs/compositions.json` (`whole_path`).
        if args.windows(2).any(|w| w.first().map(String::as_str) == Some("--look") && w.get(1).map(String::as_str) == Some("off")) {
            path.look = false;
        }
        // `--lenses on|off`: each tentacle's window composed by its lens, or from the BM25F flags.
        if let Some(v) = args.windows(2).find(|w| w.first().map(String::as_str) == Some("--lenses")).and_then(|w| w.get(1)) {
            path.lenses = match v.as_str() {
                "on" => true,
                "off" => false,
                _ => return Err(format!("serve: --lenses takes on or off, got {v}")),
            };
        }
        if let Some(v) = args.windows(2).find(|w| w.first().map(String::as_str) == Some("--reach-k")).and_then(|w| w.get(1)) {
            match v.parse::<usize>() {
                Ok(n) if n >= 1 => path.reach_k = n,
                _ => {
                    return Err(format!("serve: --reach-k takes a count of at least 1, got {v}"));
                }
            }
        }
        // `--rerank-compare`: every reranker set scored batched and one at a time (measurement).
        path.rerank_compare = args.iter().any(|a| a == "--rerank-compare");
        // `--fast-k <n>`: pieces of the stack's order the reranker reads in fast mode.
        if let Some(v) = args.windows(2).find(|w| w.first().map(String::as_str) == Some("--fast-k")).and_then(|w| w.get(1)) {
            path.fast_k = v
                .parse::<usize>()
                .ok()
                .filter(|k| *k >= 1)
                .ok_or_else(|| format!("serve: --fast-k takes a count of at least 1, got {v}"))?;
        }
        // `--look-gate <x>|off`: the second look runs only when the reranker's top two are closer
        // than x log-odds.
        if let Some(v) = args.windows(2).find(|w| w.first().map(String::as_str) == Some("--look-gate")).and_then(|w| w.get(1)) {
            path.look_gate = match v.as_str() {
                "off" => None,
                x => Some(
                    x.parse::<f32>()
                        .ok()
                        .filter(|g| *g >= 0.0)
                        .ok_or_else(|| format!("serve: --look-gate takes off or a log-odds gap, got {x}"))?,
                ),
            };
        }
        // `--judge-window <tokens>|off`: the judge reads at most that many tokens of each piece,
        // whole lines around the question's evidence (`nmapi::whole::judge_view`); off reads
        // pieces whole. `--look-window on|off`: the second look reads the same windows.
        if let Some(v) = args.windows(2).find(|w| w.first().map(String::as_str) == Some("--judge-window")).and_then(|w| w.get(1)) {
            path.judge_window = match v.as_str() {
                "off" => None,
                x => Some(
                    x.parse::<usize>()
                        .ok()
                        .filter(|n| *n >= 32)
                        .ok_or_else(|| format!("serve: --judge-window takes off or a token count of at least 32, got {x}"))?,
                ),
            };
        }
        if let Some(v) = args.windows(2).find(|w| w.first().map(String::as_str) == Some("--fast-judge-window")).and_then(|w| w.get(1)) {
            path.fast_judge_window = match v.as_str() {
                "off" => None,
                x => Some(
                    x.parse::<usize>()
                        .ok()
                        .filter(|n| *n >= 32)
                        .ok_or_else(|| format!("serve: --fast-judge-window takes off or a token count of at least 32, got {x}"))?,
                ),
            };
        }
        if let Some(on) = switch(args, "--look-window") {
            path.look_window = on;
        }
        if args.iter().any(|a| a == "--anatomy") {
            match std::fs::read(org.home.join("designs/compositions.json"))
                .map_err(|e| e.to_string())
                .and_then(|b| nmjson::parse(&b).map_err(|e| e.to_string()))
                .and_then(|t| nmapi::whole::arrangements(&t).map_err(|e| e.message))
            {
                Ok(a) => path.arrangements = a,
                Err(e) => {
                    return Err(format!("serve: --anatomy: {e}"));
                }
            }
        }
        // `--packs off` (2026-11-13, MO8): G2 vetoes every muscle-memory pack step.
        if args.windows(2).any(|w| w.first().map(String::as_str) == Some("--packs") && w.get(1).map(String::as_str) == Some("off"))
            && let Ok(mut d) = path.desk.lock()
        {
            d.veto_packs = true;
        }
        eprintln!("whole path: on (ask {:?}; reflexes {})", path.ask, path.reflexes.lock().map(|s| s.installed.len()).unwrap_or(0));
        // The Core Brain's models bind and stage at start, beside answering, so no question pays
        // for loading them (the first question waits only if it arrives before they are ready).
        let path = std::sync::Arc::new(path);
        let warm = std::sync::Arc::clone(&path);
        std::thread::spawn(move || match warm.warm() {
            Ok((secs, n)) => eprintln!("whole path: models warm in {secs:.1} s ({n} tensors staged)"),
            Err(e) => eprintln!("whole path: warm start failed ({e}); models load on the first question"),
        });
        *w = Some(path);
    }
    // `--brain-chains` (2026-11-10): the Core Brain side (G2 selection, a MicroModel chain, the G3
    // behaviour log, cartridges, preferences); off unless the flag is given.
    if nmapi::brain_side::from_args(args) {
        match nmapi::brain_side::enable(org) {
            Ok(m) => eprintln!("{m}"),
            Err(e) => {
                return Err(format!("serve: {e}"));
            }
        }
    }
    // `--dense` (2026-11-10) binds the embedder alone and starts the dense lane; with `--neural`
    // the lane starts too, unless `--dense off`.
    let dense_arg = args.iter().position(|a| a == "--dense").map(|i| args.get(i + 1).map_or("on", String::as_str));
    // The plain path's default lane (`dense_by_default`) starts after the tools answer, in `serve`.
    let dense_on = matches!(dense_arg, Some(v) if v != "off") || (dense_arg.is_none() && args.iter().any(|a| a == "--neural"));
    if dense_on {
        if !args.iter().any(|a| a == "--neural") {
            match nmapi::neural::bind_embedder(org) {
                Ok(m) => eprintln!("dense: {m}"),
                Err(e) => {
                    return Err(format!("dense: {e}"));
                }
            }
        }
        match nmapi::neural::start_dense(org) {
            Ok(m) => eprintln!("dense: {m}"),
            Err(e) => {
                return Err(format!("dense: {e}"));
            }
        }
    }
    // `--record` switches the Parameter Map recording on for this session (off by default).
    if args.iter().any(|a| a == "--record")
        && let Ok(mut p) = org.parmap.lock()
    {
        p.recording = true;
    }
    // `--unit <id>` loads a registered learning-weight unit into its component (S8).
    if let Some(id) = args.iter().position(|a| a == "--unit").and_then(|i| args.get(i + 1)) {
        match nmapi::learn::load_for_serve(org, id) {
            Ok(m) => eprintln!("unit: {m}"),
            Err(e) => {
                return Err(format!("unit: {e}"));
            }
        }
    }
    Ok(())
}

/// Every `designs/*.json` parses with the strict nmjson reader (duplicate keys, depth).
fn check_designs() -> Result<usize, String> {
    let dir = nano_root().join("designs");
    let entries = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut n = 0;
    for p in entries.flatten().map(|e| e.path()) {
        if p.extension().is_some_and(|x| x == "json") {
            let bytes = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            let v = nmjson::parse(&bytes).map_err(|e| format!("designs: {}: {e}", p.display()))?;
            // 2026-11-10 (step P0): the drone reader sets are validated, not only parsed: pairs
            // inside each population's layers and heads, no pair twice, a non-empty default,
            // tentacle sets named by lens.
            if p.file_name().is_some_and(|f| f == "drones.json") {
                let sets = nmengn::kvstore::drone_sets(&nano_root(), &v).map_err(|e| format!("designs: {e}"))?;
                for s in sets.values() {
                    if let Some(t) = s.tentacles.keys().find(|t| !nmsubs::lens::NAMES.contains(&t.as_str())) {
                        return Err(format!(
                            "designs: {}: {}: tentacle set {t} is not a lens name ({})",
                            nmengn::kvstore::DRONE_FILE,
                            s.population,
                            nmsubs::lens::NAMES.join(", ")
                        ));
                    }
                }
            }
            // 2026-11-10 (tentacle core A8): the one module specification of the tentacle cores is
            // validated and every tentacle's clusters resolve in the population.
            if p.file_name().is_some_and(|f| f == "tentacle-core.json") {
                nmapi::tentacle_core::check_spec_file(&nano_root(), &v).map_err(|e| format!("designs: {e}"))?;
            }
            n += 1;
        }
    }
    Ok(n)
}

/// Write the recomputed card hash into the last `"hash": "..."` of a `part.json`, leaving
/// the rest of the file text as written.
fn seal(path: &Path) -> Result<String, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v = nmjson::parse(text.as_bytes()).map_err(|e| e.to_string())?;
    let hash = nmtype::card_hash(&v);
    let key = "\"hash\": \"";
    let start = text.rfind(key).ok_or("no \"hash\" field")? + key.len();
    let end = start + text[start..].find('"').ok_or("unterminated hash")?;
    let sealed = format!("{}{hash}{}", &text[..start], &text[end..]);
    std::fs::write(path, sealed).map_err(|e| format!("{}: {e}", path.display()))?;
    nmtype::Card::load(path).map_err(|e| e.to_string())?;
    Ok(hash)
}

/// The sections `check` knows beside the crates, in the order the whole run takes them.
const SECTIONS: &[&str] = &["cards", "designs", "anatomy", "integrity", "lock", "documents", "languages"];

/// Every crate with registered section checks, in the order the whole run takes them (the
/// foundations first, the program last).
const CHECKED_CRATES: &[&str] = &[
    "nmhash", "nmjson", "nmtype", "nmnte", "nmlex", "nmdoc", "nmsubs", "nmpod", "nmbrain", "nmmem", "nmnpcs", "nmffi", "nmsafe", "nmtokz",
    "nmengn", "nmlearn", "nmtool", "nmmusc", "nmapi", "nmcore",
];

/// One named section (`SECTIONS`): `None` when the name is not a section, else what it checked
/// (a count and the line printed on a pass) or its fault line.
fn run_section(name: &str) -> Option<Result<(usize, String), String>> {
    Some(match name {
        "designs" => check_designs().map(|n| (n, format!("designs ({n} files parse as strict JSON)"))),
        "anatomy" => check_anatomy().map(|n| {
            (1, format!("anatomy (anatomy.json and the forage assembly it names: {n} nodes pass the Integrity Validator against the registry)"))
        }),
        "integrity" => check_integrity().map(|n| (n, format!("integrity ({n} files match INTEGRITY.md)"))),
        "documents" => documents::check(&nano_root())
            .map(|n| (n, format!("documents ({n} fixture cases match tests/documents)")))
            .map_err(|e| format!("documents: {e}")),
        "languages" => languages::check(&nano_root())
            .map(|n| (n, format!("languages ({n} lexer snippets and fixtures match tests/languages)")))
            .map_err(|e| format!("languages: {e}")),
        "lock" => check_lock().map(|n| (n, format!("lock ({n} trees match tests/dev/code-lock.txt)"))),
        "cards" => check_cards().map(|n| (n, format!("cards ({n} part.json verified)"))),
        _ => return None,
    })
}

/// What a whole run (`check` with no name, or `check all`) found: one entry per section and per
/// crate check, passed or with its fault line.
#[derive(Debug, Default)]
struct Tally {
    passed: usize,
    failed: Vec<String>,
}

impl Tally {
    fn pass(&mut self, line: &str) {
        self.passed += 1;
        println!("ok   {line}");
    }
    fn fail(&mut self, id: &str, e: &str) {
        println!("FAIL {e}");
        self.failed.push(format!("{id}: {}", e.lines().next().unwrap_or_default()));
    }
    fn total(&self) -> usize {
        self.passed + self.failed.len()
    }
    /// The closing line: passes out of the total, the failures named.
    fn summary(&self, ms: u128) -> String {
        if self.failed.is_empty() {
            format!("check all ok: {} of {} passed in {ms} ms", self.passed, self.total())
        } else {
            format!(
                "check all FAILED: {} of {} passed, {} failed in {ms} ms:\n  {}",
                self.passed,
                self.total(),
                self.failed.len(),
                self.failed.join("\n  ")
            )
        }
    }
}

/// Every name the whole run takes: the sections, then each crate.
fn all_check_names() -> Vec<&'static str> {
    SECTIONS.iter().chain(CHECKED_CRATES).copied().collect()
}

/// `check` with no name or `check all`: every section and every crate's checks, each run even
/// when an earlier one failed, then the count passed; fails when any failed.
fn check_all() -> ExitCode {
    let start = std::time::Instant::now();
    let mut t = Tally::default();
    for name in all_check_names() {
        if let Some(r) = run_section(name) {
            match r {
                Ok((_, line)) => t.pass(&line),
                Err(e) => t.fail(name, &e),
            }
            continue;
        }
        let Some(list) = crate_checks(name) else {
            t.fail(name, &format!("{name}: no checks registered for this crate"));
            continue;
        };
        for (id, check) in list {
            match check() {
                Ok(()) => t.pass(id),
                Err(e) => t.fail(id, &format!("{id}: {e}")),
            }
        }
    }
    println!("{}", t.summary(start.elapsed().as_millis()));
    if t.failed.is_empty() { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}

/// `neuromorph-nano check <crate|cards|designs|anatomy|integrity>...`: runs each crate's checks top-down and stops at
/// the first failing sub-component, which is the pinpoint. With no name (or `all`) it runs
/// everything and counts (`check_all`).
fn section_check(names: &[String]) -> ExitCode {
    if names.is_empty() || (names.len() == 1 && names.first().is_some_and(|n| n == "all")) {
        return check_all();
    }
    let start = std::time::Instant::now();
    let mut ran = 0;
    for name in names {
        if let Some(r) = run_section(name) {
            match r {
                Ok((n, line)) => {
                    println!("ok   {line}");
                    ran += n;
                }
                Err(e) => {
                    println!("FAIL {e}");
                    return ExitCode::FAILURE;
                }
            }
            continue;
        }
        // 2026-11-10: `check <crate>.<part>` runs that one section check (for example a
        // processor-only check while a graphics-card job runs).
        let krate = name.split('.').next().unwrap_or(name);
        let Some(mut list) = crate_checks(krate) else {
            println!("FAIL {name}: no checks registered for this crate");
            return ExitCode::FAILURE;
        };
        if name.contains('.') {
            list.retain(|(id, _)| id == name);
            if list.is_empty() {
                println!("FAIL {name}: no such section check");
                return ExitCode::FAILURE;
            }
        }
        for (id, check) in list {
            ran += 1;
            if let Err(e) = check() {
                println!("FAIL {id}: {e}");
                return ExitCode::FAILURE;
            }
            println!("ok   {id}");
        }
    }
    println!("check ok: {ran} checks in {} ms", start.elapsed().as_millis());
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // Hook intake runs first and alone: it must stay fast and always exit 0. `hook --probe` exits
    // at once, so setup can time the program's launch and exit (`connect::launch_ms`).
    if args.first().map(String::as_str) == Some("hook") && args.get(1).map(String::as_str) == Some("--probe") {
        return ExitCode::SUCCESS;
    }
    if args.first().map(String::as_str) == Some("hook") && args.get(1).is_some_and(|e| e != "stop") {
        return hook::run(args.get(1..).unwrap_or_default());
    }
    nmffi::dynlib::disable_power_throttling();
    // The Nano folder must be found before any command reads from it.
    let info = matches!(args.first().map(String::as_str), Some("--help" | "-h" | "help" | "--version" | "-V"));
    if !info && let Err(e) = nmtype::find_root() {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }
    // The live launcher runs before anything else reads the arguments: they all go to the copy.
    if args.first().map(String::as_str) == Some("launch") {
        return launch::run(&nano_root(), args.get(1..).unwrap_or_default());
    }
    // `--precision 8|16` or `--precision base=8,writer=16,reranker=16,embedder=8` (2026-11-13,
    // directions of Luis Jake Gabriel III): every part, or part by part, before anything binds.
    // Saved controls (Neuromorph Nano, 2026-11-15): the precision set in the interface applies
    // at start; a `--precision` flag on the command line wins over it.
    let saved_precision = controls::precision_arg(&nano_root());
    if let Some(v) =
        args.windows(2).find(|w| w.first().map(String::as_str) == Some("--precision")).and_then(|w| w.get(1)).or(saved_precision.as_ref())
    {
        match nmengn::model::Precision::parse(v) {
            Ok(p) => {
                if let Ok(mut g) = nmengn::model::PRECISION.write() {
                    *g = p;
                }
            }
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        }
    }
    // `--precision-map <file>` (2026-11-14): per-piece widths per part.
    if let Some(v) = args.windows(2).find(|w| w.first().map(String::as_str) == Some("--precision-map")).and_then(|w| w.get(1)) {
        match nmengn::model::read_precision_map(std::path::Path::new(v)) {
            Ok(m) => {
                if let Ok(mut g) = nmengn::model::PRECISION_MAP.write() {
                    *g = Some(m);
                }
            }
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        }
    }
    // Engine kernel switches for any command (2026-11-13): `--half-math`, `--dp4a`.
    if switch(&args, "--half-math") == Some(true) {
        nmengn::gpu::HALF_MATH.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if args.iter().any(|a| a == "--dp4a") {
        nmengn::gpu::DP4A.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    match args.first().map(String::as_str) {
        Some("--help" | "-h" | "help") => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some("--version" | "-V") => {
            println!("neuromorph-nano {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        // `selftest engine [cpu|gpu|both] [--record]`: the engine against its recorded reference.
        Some("selftest") if args.get(1).map(String::as_str) == Some("engine") => {
            let want = args.get(2).filter(|a| !a.starts_with("--")).map_or("both", String::as_str);
            let record = args.iter().any(|a| a == "--record");
            match nmcore::selftest::engine(&nano_root(), want, record) {
                Ok((text, pass)) => {
                    print!("{text}");
                    if pass { ExitCode::SUCCESS } else { ExitCode::FAILURE }
                }
                Err(e) => {
                    eprintln!("selftest engine: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        // The earlier `mcp` face (12 tools over the nmcore copies) was retired 2026-11-10;
        // `serve` is the MCP face.
        // The interface server (Neuromorph Nano, 2026-11-15): `ui [--port <n>]`, local only.
        Some("ui") => match ui::run(&nano_root(), &args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("ui: {e}");
                ExitCode::FAILURE
            }
        },
        Some("mcp") => {
            eprintln!("neuromorph-nano mcp was retired 2026-11-10; use neuromorph-nano serve --root .");
            ExitCode::FAILURE
        }
        // One live tool call on this folder (or `--root <repo>`), memory off (nothing written to
        // the memory folder); the call is still audited (E2, 2026-11-15), without its argument
        // when the memory control is off.
        Some("call") => {
            let home = nano_root();
            let root = match args.windows(2).find(|w| w.first().map(String::as_str) == Some("--root")).and_then(|w| w.get(1)) {
                Some(r) if Path::new(r).is_dir() => nmsubs::repo::canon(Path::new(r)),
                Some(r) => {
                    eprintln!("call: --root {r} is not a folder");
                    return ExitCode::FAILURE;
                }
                None => home.clone(),
            };
            let org = match nmapi::forage::Organism::open_with(
                &home,
                &root,
                &["neuromorph-nano".to_owned()],
                false,
                &nmtype::project::memory_dir(&home, &root),
            ) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("organism: {e}");
                    return ExitCode::FAILURE;
                }
            };
            if let Err(e) = nmapi::tools::install(org) {
                eprintln!("install: {}: {}", e.code, e.message);
                return ExitCode::FAILURE;
            }
            project::start_audit(&home, &root, !controls::memory_off(&home), "plain", "call");
            nmapi::parity::lend_routes(ui::lent_route);
            let api = nmapi::tools::api().observed(project::observe);
            let tool = args.get(1).cloned().unwrap_or_default();
            let input = args
                .get(2)
                .filter(|a| a.as_str() != "--root")
                .map_or_else(|| Ok(Value::Object(std::collections::BTreeMap::new())), |j| nmjson::parse(j.as_bytes()));
            let Ok(input) = input else {
                eprintln!("input is not JSON");
                return ExitCode::FAILURE;
            };
            match api.dispatch(&tool, &input) {
                Ok(Value::String(t)) => {
                    print!("{t}");
                    ExitCode::SUCCESS
                }
                Ok(v) => {
                    println!("{}", v.encode().unwrap_or_default());
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("{}: {}", e.code, e.message);
                    ExitCode::FAILURE
                }
            }
        }
        Some("check") => section_check(args.get(1..).unwrap_or_default()),
        Some("serve") => serve(&args),
        // `setup --project <repo> [--write]` (E1, 2026-11-15): the repository's `.mcp.json` entry.
        Some("setup") => project::setup(&nano_root(), &args),
        Some("situation") => situation::run(&nano_root(), &args),
        // Reminders, loops and the step graph for the coding agent (`guidance.rs`).
        Some("guidance") => guidance::run(&nano_root(), &args),
        Some("session") => sessions::run(&nano_root(), &args),
        Some("ledger") => ledger::run(&nano_root(), &args),
        Some("outcomes") => outcomes::run(&nano_root(), &args),
        Some("examples") => examples::run(&nano_root(), &args),
        Some("keep") => keep::run(&nano_root(), &args),
        Some("library") => library::run(&nano_root(), &args),
        // Stated preference cartridges: the user's words turned into rules (nmlearn::stated).
        Some("preferences") => preferences::run(&nano_root(), &args),
        // The user's own repositories for backing up and rolling back their data (`backups.rs`).
        Some("backup") => backups::cli(&nano_root(), &args),
        // 2026-11-26: the lab, inside the one program (the same commands as the nano-lab program).
        Some("lab") => nmlab::run(args.get(1..).unwrap_or_default().to_vec()),
        Some("pair") => ui::pair_command(&nano_root()),
        // `dense clear --project <repo>`: removes the project's dense index kept on disk.
        Some("dense") => dense_command(&args),
        Some("trace") => trace::run(&args, &nano_root()),
        Some("bench") => bench::run(&nano_root(), args.get(1..).unwrap_or_default()),
        Some("writer-idle") => writer_trial::run(&nano_root(), args.get(1..).unwrap_or_default()),
        Some("adapter-fit") => writer_trial::fit_run(&nano_root(), args.get(1..).unwrap_or_default()),
        // 2026-11-14 (guiding Claude Code): `hook stop [--root <repo>]`, a Claude Code Stop hook. It
        // reads the hook's JSON on standard input; when Rust files in the repository have changed
        // (git) and `cargo check` fails, it returns `decision: block` with the first errors, so the
        // agent goes back to close the loop; otherwise it lets the agent stop. Never blocks twice in
        // a row (`stop_hook_active`).
        Some("hook") if args.get(1).map(String::as_str) == Some("stop") => {
            let mut input = String::new();
            let _ = std::io::Read::read_to_string(&mut std::io::stdin(), &mut input);
            let parsed = nmjson::parse(input.as_bytes()).ok();
            let agy = parsed.as_ref().is_some_and(|v| v.get("hookEventName").is_ok() || v.get("continuationActive").is_ok());
            let again = parsed.as_ref().is_some_and(|v| {
                matches!(v.get("stop_hook_active"), Ok(nmjson::Value::Bool(true)))
                    || matches!(v.get("continuationActive"), Ok(nmjson::Value::Bool(true)))
            });
            let root = args
                .windows(2)
                .find(|w| w.first().map(String::as_str) == Some("--root"))
                .and_then(|w| w.get(1))
                .map_or_else(|| std::path::PathBuf::from("."), std::path::PathBuf::from);
            if again {
                if agy {
                    println!("{{}}");
                }
                return ExitCode::SUCCESS;
            }
            let changed = std::process::Command::new("git")
                .args(["-c", "safe.directory=*", "status", "--porcelain"])
                .current_dir(&root)
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).lines().any(|l| l.trim_end().ends_with(".rs")))
                .unwrap_or(false);
            if !changed {
                if agy {
                    println!("{{}}");
                }
                return ExitCode::SUCCESS;
            }
            let out = std::process::Command::new("cargo")
                .args(["check", "--offline", "--message-format", "short", "--quiet"])
                .current_dir(&root)
                .output();
            if let Ok(o) = out
                && !o.status.success()
            {
                let errs: Vec<String> =
                    String::from_utf8_lossy(&o.stderr).lines().filter(|l| l.contains("error")).take(3).map(str::to_owned).collect();
                let reason = format!("Rust files changed and `cargo check` fails; fix these before finishing: {}", errs.join(" | "));
                let decision = if agy { "continue" } else { "block" };
                let v = nmjson::Value::Object(
                    [("decision".to_owned(), nmjson::Value::String(decision.into())), ("reason".to_owned(), nmjson::Value::String(reason))]
                        .into_iter()
                        .collect(),
                );
                println!("{}", v.encode().unwrap_or_default());
            } else if agy {
                println!("{{}}");
            }
            ExitCode::SUCCESS
        }
        Some("lock") => {
            // The code lock of each `--root` (default: the Nano folder), as `check lock` reads it.
            let roots: Vec<&String> = args.windows(2).filter(|w| w.first().map(String::as_str) == Some("--root")).filter_map(|w| w.get(1)).collect();
            let home = nano_root();
            let roots: Vec<String> = if roots.is_empty() { vec![".".to_owned()] } else { roots.into_iter().cloned().collect() };
            for r in roots {
                let dir = if Path::new(&r).is_absolute() { PathBuf::from(&r) } else { home.join(&r) };
                if !dir.is_dir() {
                    eprintln!("lock: {r} is not a folder");
                    return ExitCode::FAILURE;
                }
                print!("root {}\n{}", r.replace('\\', "/"), nmsubs::lock::of_root(&dir).text());
            }
            ExitCode::SUCCESS
        }
        Some("daemon") => {
            let mut forwarded = vec!["--daemon".to_owned()];
            forwarded.extend_from_slice(args.get(1..).unwrap_or_default());
            serve(&forwarded)
        }
        Some("connect") => onboard::run(&nano_root(), args.get(1..).unwrap_or_default()),
        Some("soak") => soak::run(&nano_root(), args.get(1..).unwrap_or_default()),
        Some("coldstart") => coldstart::run(&nano_root(), args.get(1..).unwrap_or_default()),
        Some("attach") => daemon_attach(args.get(1..).unwrap_or_default(), false),
        Some("ask") => daemon_attach(args.get(1..).unwrap_or_default(), true),
        Some("seal") => {
            for p in args.get(1..).unwrap_or_default() {
                match seal(Path::new(p)) {
                    Ok(h) => println!("sealed {p} {h}"),
                    Err(e) => {
                        eprintln!("seal {p}: {e}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("{USAGE}");
            ExitCode::FAILURE
        }
    }
}

/// The stage of a bare test memory name such as `s6-ctrl-1794...` (`s6`), if it has one.
fn stage_of(name: &str) -> Option<&str> {
    let (stage, _) = name.split_once('-')?;
    let digits = stage.strip_prefix('s')?;
    (!digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())).then_some(stage)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vecs(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| (*x).to_owned()).collect()
    }

    #[test]
    fn check_all_covers_every_section_and_crate() {
        let names = all_check_names();
        assert_eq!(names.len(), SECTIONS.len() + CHECKED_CRATES.len());
        for c in CHECKED_CRATES {
            assert!(crate_checks(c).is_some_and(|l| !l.is_empty()), "{c} has no checks");
        }
        for sct in SECTIONS {
            assert!(crate_checks(sct).is_none(), "{sct} is a section, not a crate");
        }
        assert!(run_section("nmapi").is_none());
        let mut t = Tally { passed: 3, failed: Vec::new() };
        assert!(t.summary(5).starts_with("check all ok: 3 of 3"));
        t.failed.push("nmx.a: broke".to_owned());
        let line = t.summary(5);
        assert!(line.contains("3 of 4 passed, 1 failed") && line.contains("nmx.a: broke"));
        for cmd in ["daemon", "attach", "ask", "connect", "soak", "coldstart"] {
            assert!(USAGE.lines().any(|l| l.starts_with(&format!("  {cmd} "))), "{cmd} missing from USAGE");
        }
    }

    #[test]
    fn flow_default_and_warm_keep_predicates() {
        assert_eq!(with_flow_default(&[]), vecs(&["--flow"]));
        assert_eq!(with_flow_default(&vecs(&["--flow"])), vecs(&["--flow"]));
        assert_eq!(with_flow_default(&vecs(&["--flow", "off"])), vecs(&["--flow", "off"]));
        assert_eq!(with_flow_default(&vecs(&["--whole"])), vecs(&["--whole"]));
        assert_eq!(with_flow_default(&vecs(&["--neural"])), vecs(&["--neural"]));

        assert!(!flow_on(&[]));
        assert!(flow_on(&vecs(&["--flow"])));
        assert!(!flow_on(&vecs(&["--flow", "off"])));
        assert!(flow_on(&vecs(&["--flow", "--root", "."])));

        assert_eq!(serve_mode(&vecs(&["--flow"])), "flow");
        assert_eq!(serve_mode(&vecs(&["--flow", "off"])), "plain");
        assert_eq!(serve_mode(&vecs(&["--whole"])), "whole");
        assert_eq!(serve_mode(&vecs(&["--whole", "--neural"])), "whole");
        assert_eq!(serve_mode(&vecs(&["--neural"])), "neural");
        assert_eq!(serve_mode(&vecs(&["--dense"])), "dense");

        assert!(!should_warm_keep(&[]));
        assert!(!should_warm_keep(&vecs(&["--flow", "off"])));
        assert!(should_warm_keep(&vecs(&["--flow"])));
        assert!(should_warm_keep(&vecs(&["--whole"])));
        assert!(should_warm_keep(&vecs(&["--whole", "--neural"])));

        let dir = nmtype::check_dir("nmcore-main-dense-pred").unwrap();
        let mem = dir.join("mem");
        let home = nmtype::nano_root();
        let org = nmapi::forage::Organism::open_with(&home, &dir, &[], true, &mem).unwrap();
        assert_eq!(dense_by_default(&org, &vecs(&["--flow", "off"])), nmsubs::dense::store::cache_on(&home));
        assert!(!dense_by_default(&org, &vecs(&["--whole"])));
        assert!(!dense_by_default(&org, &vecs(&["--neural"])));
        assert!(!dense_by_default(&org, &vecs(&["--dense"])));
    }
}
