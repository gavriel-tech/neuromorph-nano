//! Connecting a repository to Claude Code and disconnecting it again: the Nano's hook entries in
//! the repository's personal settings file (`.claude/settings.local.json`) and its MCP server
//! entry in `.mcp.json`. Every write is shown first, merged as one contiguous insertion (other
//! entries keep their bytes), preceded by a backup copy in the project's memory folder, and logged
//! in the audit. Removing takes out exactly the Nano's entries and gives back the file as it was.
use crate::jsonc::{self, Node, Out, Style};
// Roadmap Phase 4.6 (2026-11-25): the server-entry merge, its install record and the entry itself
// live in the library (`nmcore::mcp_clients`) so their tests run with it; they keep their names here.
pub use nmcore::mcp_clients::{Created, Plan, is_empty_object, mcp_entry, plan_servers, unplan_servers};
use nmcore::mcp_clients::root_of;
use nmjson::Value;
use std::path::{Path, PathBuf};

/// The MCP server name the entries use.
pub const SERVER: &str = crate::project::SERVER;

/// The hook groups setup installs: (event, matcher, run in the background). Each handler that
/// Claude Code waits for carries the timeout `timeout_s` derives; a background handler none.
pub const GROUPS: &[(&str, Option<&str>, bool)] = &[
    ("SessionStart", None, false),
    ("UserPromptSubmit", None, false),
    ("PreToolUse", Some("Edit|MultiEdit|Write|NotebookEdit"), false),
    ("PostToolUse", Some("Bash|PowerShell|mcp__neuromorph-nano__.*"), false),
    ("PostToolUse", Some("Read|Edit|MultiEdit|Write|NotebookEdit|Grep|Glob"), true),
    ("PostToolUseFailure", None, false),
    ("Stop", None, false),
    ("PreCompact", None, false),
    // A session that ends before any Stop (a turn that failed before the model answered) still
    // closes in the situation record.
    ("SessionEnd", None, false),
];

/// The events setup installs, each once, in order.
#[must_use]
pub fn events() -> Vec<&'static str> {
    let mut e: Vec<&str> = Vec::new();
    for g in GROUPS {
        if !e.contains(&g.0) {
            e.push(g.0);
        }
    }
    e
}

/// How many times the slowest whole hook measured on this machine a hook's timeout allows: the
/// one stated multiple, generous, since the timeout is only a backstop against a stuck handler.
pub const TIMEOUT_MULTIPLE: u64 = 20;
/// The stated fallback timeout in seconds while the measurements are not settled (`settled`).
pub const FALLBACK_TIMEOUT_S: i64 = 10;

/// Whether a series of times, oldest first, is large enough to judge its slowest: its slowest was
/// already reached in the older half, so the newer half did not raise it. Derived from the series
/// itself: a slowest that is still climbing is not judged.
#[must_use]
pub fn settled(ms: &[u64]) -> bool {
    if ms.len() < 2 {
        return false;
    }
    let (older, newer) = ms.split_at(ms.len() - ms.len() / 2);
    older.iter().max() >= newer.iter().max()
}

/// The time to launch the program and see it exit on this machine, in milliseconds, measured from
/// outside (process creation to exit) with `hook --probe`, which exits as soon as it starts:
/// probes are run until their slowest is settled, within the fallback timeout; `None` when the
/// program cannot be run or the probes never settle.
#[must_use]
pub fn launch_ms(exe: &str) -> Option<u64> {
    let until = std::time::Instant::now() + std::time::Duration::from_secs(FALLBACK_TIMEOUT_S.unsigned_abs());
    let mut ms: Vec<u64> = Vec::new();
    while std::time::Instant::now() < until {
        let mut c = std::process::Command::new(exe);
        c.args(["hook", "--probe"]).stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt as _;
            c.creation_flags(0x0800_0000);
        }
        let t = std::time::Instant::now();
        if !c.status().ok()?.success() {
            return None;
        }
        ms.push(u64::try_from(t.elapsed().as_millis()).unwrap_or(u64::MAX));
        if settled(&ms) {
            return ms.iter().max().copied();
        }
    }
    None
}

/// The timeout from the hooks' own whole times: settled, 20 times the slowest (with the launch of
/// a missing program, which cannot be measured from outside, the recorded time alone); climbing,
/// the fallback; none recorded, the outside probe's path (the fallback with its own reason). A
/// scratch memory folder, removed after.
fn whole_check() -> Result<(), String> {
    let dir = std::env::temp_dir().join(format!("nm-connect-whole-{}-{}", std::process::id(), nmtype::now_ms()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let exe = dir.join("missing.exe").display().to_string();
    let write = |rows: &[&str]| std::fs::write(crate::hook::situation_file(&dir), rows.iter().map(|r| format!("{r}\n")).collect::<String>());
    let run = || -> Result<(), String> {
        write(&[r#"{"hook_ms":1}"#, r#"{"hook_ms":2}"#]).map_err(|e| e.to_string())?;
        let (s, why) = timeout_s(&dir, &exe);
        if s != FALLBACK_TIMEOUT_S || why.contains("whole hooks recorded") {
            return Err(format!("connect: without whole times, {s} s ({why})"));
        }
        write(&[r#"{"hook_ms":1,"whole_ms":90}"#, r#"{"hook_ms":1,"whole_ms":40}"#, r#"{"hook_ms":1,"whole_ms":60}"#, r#"{"hook_ms":1,"whole_ms":50}"#])
            .map_err(|e| e.to_string())?;
        let (s, why) = timeout_s(&dir, &exe);
        if s != 2 || !why.contains("90 ms from process creation") || !why.contains("could not be measured") {
            return Err(format!("connect: settled whole times gave {s} s ({why})"));
        }
        write(&[r#"{"whole_ms":40}"#, r#"{"whole_ms":50}"#, r#"{"whole_ms":60}"#, r#"{"whole_ms":900}"#]).map_err(|e| e.to_string())?;
        let (s, why) = timeout_s(&dir, &exe);
        if s != FALLBACK_TIMEOUT_S || !why.contains("do not yet settle") {
            return Err(format!("connect: climbing whole times gave {s} s ({why})"));
        }
        Ok(())
    };
    let r = run();
    let _ = std::fs::remove_dir_all(&dir);
    r
}

/// The hook timeout in seconds for a project: `TIMEOUT_MULTIPLE` times the slowest whole hook on
/// this machine, rounded up to whole seconds. A whole hook is the larger of what the hooks record
/// themselves (`whole_ms` in the project's situation record: from their process's creation to
/// their record) and the program's launch measured from outside now (`launch_ms`), which also
/// counts the time before a process exists. Before any hook has recorded that, it is the launch (`launch_ms`,
/// measured from outside) plus the hook's own work (`hook_ms`, from its start to its record). The
/// stated fallback while the recorded hooks are not `settled` or the launch cannot be measured.
/// Returns the timeout and how it was found. Setup writes it, and writes it again when it is run
/// again.
#[must_use]
pub fn timeout_s(memory: &Path, exe: &str) -> (i64, String) {
    let recs = crate::situation::read(memory, None, None);
    let whole: Vec<u64> = recs.iter().filter_map(|v| v.get("whole_ms").and_then(Value::integer).ok()).collect();
    if !whole.is_empty() {
        if !settled(&whole) {
            return (FALLBACK_TIMEOUT_S, format!("the stated fallback: the {} whole hooks recorded here do not yet settle their slowest", whole.len()));
        }
        // The time before a process exists (the system's look at a new program) is not in the
        // recorded times; the launch measured from outside, once per setup, counts it.
        let recorded = whole.iter().max().copied().unwrap_or(0);
        let launch = launch_ms(exe);
        let slowest = launch.map_or(recorded, |l| l.max(recorded));
        let s = i64::try_from((slowest * TIMEOUT_MULTIPLE).div_ceil(1000).max(1)).unwrap_or(FALLBACK_TIMEOUT_S);
        let probe = launch.map_or_else(|| "the launch could not be measured from outside".to_owned(), |l| format!("launch measured from outside now {l} ms"));
        return (
            s,
            format!(
                "{TIMEOUT_MULTIPLE} times the slowest whole hook here ({slowest} ms: the larger of {recorded} ms from process creation to record, over {} hooks, and {probe})",
                whole.len()
            ),
        );
    }
    let ms: Vec<u64> = recs.iter().filter_map(|v| v.get("hook_ms").and_then(Value::integer).ok()).collect();
    if !settled(&ms) {
        return (FALLBACK_TIMEOUT_S, format!("the stated fallback: the {} hooks measured here do not yet settle their slowest", ms.len()));
    }
    let Some(launch) = launch_ms(exe) else {
        return (FALLBACK_TIMEOUT_S, "the stated fallback: the program's launch could not be measured".to_owned());
    };
    let slowest = ms.iter().max().copied().unwrap_or(0) + launch;
    let s = i64::try_from((slowest * TIMEOUT_MULTIPLE).div_ceil(1000).max(1)).unwrap_or(FALLBACK_TIMEOUT_S);
    (s, format!("{TIMEOUT_MULTIPLE} times the slowest whole hook here ({slowest} ms: launch {launch} ms measured now, work at most {} ms over {} hooks)", slowest - launch, ms.len()))
}

fn handler(exe: &str, event: &str, background: bool, timeout: i64) -> Out {
    let mut h = vec![
        ("type".to_owned(), Out::Str("command".into())),
        ("command".to_owned(), Out::Str(exe.to_owned())),
        ("args".to_owned(), Out::Arr(vec![Out::Str("hook".into()), Out::Str(event.to_owned())])),
    ];
    if background {
        h.push(("async".to_owned(), Out::Bool(true)));
    } else {
        h.push(("timeout".to_owned(), Out::Num(timeout)));
    }
    Out::Obj(h)
}

/// The matcher groups of one event, for the program at `exe`, with the hook timeout in seconds.
#[must_use]
pub fn groups_of(exe: &str, event: &str, timeout: i64) -> Vec<Out> {
    GROUPS
        .iter()
        .filter(|g| g.0 == event)
        .map(|(e, matcher, background)| {
            let mut o = Vec::new();
            if let Some(m) = matcher {
                o.push(("matcher".to_owned(), Out::Str((*m).to_owned())));
            }
            o.push(("hooks".to_owned(), Out::Arr(vec![handler(exe, e, *background, timeout)])));
            Out::Obj(o)
        })
        .collect()
}

/// Every hook entry setup writes, as the `hooks` object.
#[must_use]
pub fn hooks_object(exe: &str, timeout: i64) -> Out {
    Out::Obj(events().into_iter().map(|e| (e.to_owned(), Out::Arr(groups_of(exe, e, timeout)))).collect())
}

/// True when a hook handler is one setup writes: a command handler running a `neuromorph-nano`
/// program with `hook <event>` for an event setup installs. Hand-written Nano hooks (such as a
/// `hook stop` gate) are not setup's and are left alone.
#[must_use]
pub fn is_ours(h: &Node) -> bool {
    let name = h
        .get("command")
        .and_then(Node::str)
        .map(|c| c.rsplit(['/', '\\']).next().unwrap_or_default().to_ascii_lowercase())
        .unwrap_or_default();
    let args: Vec<&str> = h.get("args").and_then(Node::items).unwrap_or_default().iter().filter_map(Node::str).collect();
    h.get("type").and_then(Node::str) == Some("command")
        && (name == "neuromorph-nano" || name == "neuromorph-nano.exe")
        && args.len() == 2
        && args.first() == Some(&"hook")
        && args.get(1).is_some_and(|e| events().contains(e))
}

/// A Nano hook handler that setup did not write (shown by status, never touched).
fn is_other_nano(h: &Node) -> bool {
    let cmd = h.get("command").and_then(Node::str).unwrap_or_default().to_ascii_lowercase();
    !is_ours(h) && cmd.contains("neuromorph-nano")
}

/// The `hooks` object, checked: an object whose events are arrays of objects.
fn hooks_checked<'a>(root: &'a Node, file: &str) -> Result<Option<&'a Node>, String> {
    let Some(h) = root.get("hooks") else { return Ok(None) };
    let Some(events) = h.members() else { return Err(format!("{file}: \"hooks\" is not an object")) };
    for e in events {
        if e.value.items().is_none() {
            return Err(format!("{file}: hooks.{} is not an array", e.key));
        }
    }
    Ok(Some(h))
}

/// Takes out every handler setup writes. A matcher group left with no handler goes with it.
/// With `drop`, an event array left empty goes when the install created it (or, with no record,
/// when this removal emptied it), and `hooks` likewise. Returns the text and the handlers removed
/// by event.
fn strip_hooks(text: &str, file: &str, drop: Option<Option<&Created>>) -> Result<(String, Vec<String>), String> {
    let mut t = text.to_owned();
    let mut removed: Vec<String> = Vec::new();
    loop {
        let root = root_of(&t, file)?;
        let Some(hooks) = hooks_checked(&root, file)? else { break };
        let mut edit: Option<String> = None;
        'find: for ev in hooks.members().unwrap_or_default() {
            for (gi, g) in ev.value.items().unwrap_or_default().iter().enumerate() {
                let Some(hs) = g.get("hooks") else { continue };
                let list = hs.items().unwrap_or_default();
                let ours: Vec<usize> = list.iter().enumerate().filter(|(_, h)| is_ours(h)).map(|(i, _)| i).collect();
                if ours.is_empty() {
                    continue;
                }
                removed.push(ev.key.clone());
                edit = Some(if ours.len() == list.len() {
                    jsonc::remove(&t, &ev.value, gi)
                } else {
                    jsonc::remove(&t, hs, ours.first().copied().unwrap_or(0))
                });
                break 'find;
            }
        }
        match edit {
            Some(n) => t = n,
            None => break,
        }
    }
    let Some(record) = drop else { return Ok((t, removed)) };
    let no_record = record.is_none();
    let created = record.cloned().unwrap_or_default();
    loop {
        let root = root_of(&t, file)?;
        let Some(hooks) = hooks_checked(&root, file)? else { break };
        let members = hooks.members().unwrap_or_default();
        let gone = members.iter().position(|e| {
            e.value.items().is_some_and(<[Node]>::is_empty)
                && (created.events.contains(&e.key) || (no_record && removed.contains(&e.key)))
        });
        if let Some(i) = gone {
            t = jsonc::remove(&t, hooks, i);
            continue;
        }
        if members.is_empty() && (created.container || (no_record && !removed.is_empty()))
            && let Some(i) = root.members().unwrap_or_default().iter().position(|m| m.key == "hooks")
        {
            t = jsonc::remove(&t, &root, i);
        }
        break;
    }
    Ok((t, removed))
}

/// Plans the hook entries for `exe` in a settings text (`None`: no file yet). Entries setup wrote
/// before are replaced in place of a second copy; running it again with the same program changes
/// nothing.
/// # Errors
/// The text is not JSON, or its `hooks` has the wrong shape.
pub fn plan_hooks(text: Option<&str>, exe: &str, file: &str, timeout: i64) -> Result<Plan, String> {
    let Some(text) = text else {
        let st = Style { nl: "\n", unit: "  ".into() };
        let whole = Out::Obj(vec![("hooks".to_owned(), hooks_object(exe, timeout))]);
        return Ok(Plan {
            text: Some(format!("{}\n", jsonc::pretty(&whole, "", &st))),
            created: Created { file: true, dir: false, container: true, events: events().iter().map(|e| (*e).to_owned()).collect() },
        });
    };
    let st = Style::of(text);
    let (mut t, _) = strip_hooks(text, file, None)?;
    let mut created = Created::default();
    let root = root_of(&t, file)?;
    if hooks_checked(&root, file)?.is_none() {
        t = jsonc::append(&t, &root, Some("hooks"), &hooks_object(exe, timeout), &st);
        created.container = true;
        created.events = events().iter().map(|e| (*e).to_owned()).collect();
    } else {
        for ev in events() {
            for g in groups_of(exe, ev, timeout) {
                let root = root_of(&t, file)?;
                let hooks = hooks_checked(&root, file)?.ok_or_else(|| format!("{file}: hooks vanished while merging"))?;
                match hooks.get(ev) {
                    Some(arr) => t = jsonc::append(&t, arr, None, &g, &st),
                    None => {
                        t = jsonc::append(&t, hooks, Some(ev), &Out::Arr(vec![g]), &st);
                        created.events.push(ev.to_owned());
                    }
                }
            }
        }
    }
    Ok(Plan { text: (t != text).then_some(t), created })
}

/// Plans removing the hook entries: the new text (`None` when the file should be removed, as it
/// was created by setup and holds nothing else) and the events they were removed from.
/// # Errors
/// The text is not JSON, or its `hooks` has the wrong shape.
pub fn unplan_hooks(text: &str, created: Option<&Created>, file: &str) -> Result<(Option<String>, Vec<String>), String> {
    let (t, removed) = strip_hooks(text, file, Some(created))?;
    Ok((if created.is_some_and(|c| c.file) && is_empty_object(&t) { None } else { Some(t) }, removed))
}

/// Plans the MCP server entry in a `.mcp.json` text (`None`: no file yet).
/// # Errors
/// The text is not JSON, or `mcpServers` is not an object.
pub fn plan_mcp(text: Option<&str>, entry: &Out) -> Result<Plan, String> {
    plan_servers(text, ".mcp.json", "mcpServers", entry)
}

/// The settings key that approves servers of a project's `.mcp.json` (Claude Code in the terminal
/// and in the desktop app read it from the project's personal settings file).
pub const APPROVAL: &str = "enabledMcpjsonServers";
/// The Claude Code personal setting that disables the built-in interactive Artifact tool in lean
/// host mode (`"enableArtifact": false`).
pub const ARTIFACT_KEY: &str = "enableArtifact";

/// Plans the approval of the Nano's server in `.claude/settings.local.json` (`None`: no file yet):
/// the server's name added to `enabledMcpjsonServers`, other entries kept.
/// # Errors
/// The text is not JSON, or `enabledMcpjsonServers` is not a list.
pub fn plan_approval(text: Option<&str>) -> Result<Plan, String> {
    plan_approval_with(text, false)
}

/// `plan_approval`, also merging `"enableArtifact": false` into `.claude/settings.local.json` when
/// `lean` is true (`host.claude_lean` is `"on"` or `setup --lean`).
/// # Errors
/// As `plan_approval`.
pub fn plan_approval_with(text: Option<&str>, lean: bool) -> Result<Plan, String> {
    let file = ".claude/settings.local.json";
    let ours = || Out::Arr(vec![Out::Str(SERVER.to_owned())]);
    let Some(text) = text else {
        let st = Style { nl: "\n", unit: "  ".into() };
        let mut fields = vec![(APPROVAL.to_owned(), ours())];
        if lean {
            fields.push((ARTIFACT_KEY.to_owned(), Out::Bool(false)));
        }
        let whole = Out::Obj(fields);
        return Ok(Plan { text: Some(format!("{}\n", jsonc::pretty(&whole, "", &st))), created: Created { file: true, container: true, ..Created::default() } });
    };
    let st = Style::of(text);
    let root = root_of(text, file)?;
    let mut created = Created::default();
    let mut t = match root.get(APPROVAL) {
        None => {
            created.container = true;
            jsonc::append(text, &root, Some(APPROVAL), &ours(), &st)
        }
        Some(a) => match a.items() {
            None => return Err(format!("{file}: {APPROVAL} is not a list")),
            Some(items) if items.iter().any(|n| n.str() == Some(SERVER)) => text.to_owned(),
            Some(_) => jsonc::append(text, a, None, &Out::Str(SERVER.to_owned()), &st),
        },
    };
    if lean {
        let root2 = root_of(&t, file)?;
        t = match root2.member(ARTIFACT_KEY) {
            None => jsonc::append(&t, &root2, Some(ARTIFACT_KEY), &Out::Bool(false), &st),
            Some(m) if matches!(m.value.v, crate::jsonc::V::Bool(false)) => t,
            Some(m) => jsonc::replace_value(&t, m, &Out::Bool(false), &st),
        };
    }
    Ok(Plan { text: (t != text).then_some(t), created })
}

/// Plans removing the approval: (new text, or `None` to remove a file setup created and that holds
/// nothing else; whether the approval was there).
/// # Errors
/// The text is not JSON.
pub fn unplan_approval(text: &str, created: Option<&Created>) -> Result<(Option<String>, bool), String> {
    let file = ".claude/settings.local.json";
    let root = root_of(text, file)?;
    let Some(a) = root.get(APPROVAL) else { return Ok((Some(text.to_owned()), false)) };
    let Some(i) = a.items().and_then(|it| it.iter().position(|n| n.str() == Some(SERVER))) else { return Ok((Some(text.to_owned()), false)) };
    let mut t = jsonc::remove(text, a, i);
    let root = root_of(&t, file)?;
    if let Some(a) = root.get(APPROVAL)
        && a.items().is_some_and(<[Node]>::is_empty)
        && created.is_none_or(|c| c.container)
        && let Some(j) = root.members().unwrap_or_default().iter().position(|m| m.key == APPROVAL)
    {
        t = jsonc::remove(&t, &root, j);
    }
    let root2 = root_of(&t, file)?;
    if created.is_none_or(|c| c.container)
        && let Some(j) = root2
            .members()
            .unwrap_or_default()
            .iter()
            .position(|m| m.key == ARTIFACT_KEY && matches!(m.value.v, crate::jsonc::V::Bool(false)))
    {
        t = jsonc::remove(&t, &root2, j);
    }
    Ok((if created.is_some_and(|c| c.file) && is_empty_object(&t) { None } else { Some(t) }, true))
}

/// Plans removing the MCP entry: (new text, or `None` to remove a file setup created and that
/// holds nothing else; whether an entry was there).
/// # Errors
/// The text is not JSON, or `mcpServers` is not an object.
pub fn unplan_mcp(text: &str, created: Option<&Created>) -> Result<(Option<String>, bool), String> {
    unplan_servers(text, ".mcp.json", "mcpServers", created)
}

/// Where setup keeps its records for a project: install records and backups.
#[must_use]
pub fn records_dir(home: &Path, repo: &Path) -> PathBuf {
    nmtype::project::memory_dir(home, repo).join("setup")
}

/// The install record of one file (`settings-local` or `mcp`), if setup kept one.
#[must_use]
pub fn record(home: &Path, repo: &Path, which: &str) -> Option<Created> {
    std::fs::read(records_dir(home, repo).join("installed.json"))
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .and_then(|v| v.get(which).ok().map(Created::from_value))
}

/// Saves (or, with `None`, forgets) the install record of one file.
/// # Errors
/// The records folder cannot be written.
pub fn save_record(home: &Path, repo: &Path, which: &str, c: Option<&Created>) -> Result<(), String> {
    let dir = records_dir(home, repo);
    let p = dir.join("installed.json");
    let mut o = std::fs::read(&p).ok().and_then(|b| nmjson::parse(&b).ok()).and_then(|v| v.object().ok().cloned()).unwrap_or_default();
    match c {
        Some(c) => {
            o.insert(which.to_owned(), c.to_value());
        }
        None => {
            o.remove(which);
        }
    }
    o.insert("note".to_owned(), Value::String("What setup created in this project's Claude Code files, so that setup --remove takes out exactly that.".into()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    std::fs::write(&p, nmjson::pretty::pretty(&Value::Object(o))).map_err(|e| format!("{}: {e}", p.display()))
}

/// Copies a file as it is now (or notes that it is absent) into the project's backups; returns
/// the backup's path.
/// # Errors
/// The backup cannot be written.
pub fn backup(home: &Path, repo: &Path, file: &Path, now: Option<&[u8]>) -> Result<PathBuf, String> {
    let dir = records_dir(home, repo).join("backups");
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let name = file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let stamp = nmtype::now_ms();
    let (p, bytes): (PathBuf, &[u8]) = match now {
        Some(b) => (dir.join(format!("{stamp}-{name}")), b),
        None => (dir.join(format!("{stamp}-{name}.absent")), b"The file did not exist before setup wrote it.\n"),
    };
    std::fs::write(&p, bytes).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok(p)
}

/// Writes a file through a temporary file and a rename, creating its folder when needed.
/// Returns whether the folder was created.
/// # Errors
/// The folder or the file cannot be written.
pub fn write_file(file: &Path, text: &str) -> Result<bool, String> {
    let mut made = false;
    if let Some(d) = file.parent()
        && !d.is_dir()
    {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        made = true;
    }
    let tmp = file.with_extension("nano-tmp");
    std::fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, file).map_err(|e| format!("{}: {e}", file.display()))?;
    Ok(made)
}

/// What is installed in a settings text: (handlers setup wrote, by event; other Nano handlers;
/// commands the setup handlers run).
#[must_use]
pub fn installed(text: &str) -> (Vec<String>, usize, Vec<String>) {
    let Ok(root) = jsonc::parse(text) else { return (Vec::new(), 0, Vec::new()) };
    let mut ours = Vec::new();
    let mut other = 0;
    let mut cmds: Vec<String> = Vec::new();
    for ev in root.get("hooks").and_then(Node::members).unwrap_or_default() {
        for g in ev.value.items().unwrap_or_default() {
            for h in g.get("hooks").and_then(Node::items).unwrap_or_default() {
                if is_ours(h) {
                    ours.push(ev.key.clone());
                    if let Some(c) = h.get("command").and_then(Node::str)
                        && !cmds.iter().any(|x| x == c)
                    {
                        cmds.push(c.to_owned());
                    }
                } else if is_other_nano(h) {
                    other += 1;
                }
            }
        }
    }
    (ours, other, cmds)
}

/// The command of the Nano's MCP entry in a `.mcp.json` text, if there is one.
#[must_use]
pub fn mcp_installed(text: &str) -> Option<String> {
    let root = jsonc::parse(text).ok()?;
    let e = root.get("mcpServers")?.get(SERVER)?;
    let args: Vec<&str> = e.get("args").and_then(Node::items).unwrap_or_default().iter().filter_map(Node::str).collect();
    Some(format!("{} {}", e.get("command").and_then(Node::str).unwrap_or("?"), args.join(" ")))
}

/// Section check: the hook entries merge into a settings file with other people's entries and
/// comments and come out again byte for byte; writing twice changes nothing; a missing file is
/// created and removed again; bad JSON is refused; the MCP entry does the same.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    whole_check()?;
    let exe = "C:\\Nano\\bin\\neuromorph-nano.exe";
    let file = ".claude/settings.local.json";
    for original in FIXTURES {
        let p = plan_hooks(Some(original), exe, file, FALLBACK_TIMEOUT_S)?;
        let written = p.text.ok_or("connect: write changed nothing")?;
        if plan_hooks(Some(&written), exe, file, FALLBACK_TIMEOUT_S)?.text.is_some() {
            return Err("connect: a second write changed the file".into());
        }
        // The sample rule: a slowest still climbing is not judged; one reached in the older half is.
        if settled(&[1, 2]) || settled(&[5]) || !settled(&[9, 3, 4, 2]) || settled(&[3, 1, 2, 8]) {
            return Err("connect: the settled rule".into());
        }
        // Setup run again with a new derived timeout rewrites it, and only it.
        let again = plan_hooks(Some(&written), exe, file, 3)?.text.ok_or("connect: a new timeout was not written")?;
        if !again.contains("\"timeout\": 3") || again.contains(&format!("\"timeout\": {FALLBACK_TIMEOUT_S}")) {
            return Err("connect: the rewritten timeout".into());
        }
        let (ours, _, _) = installed(&written);
        if ours.len() != GROUPS.len() {
            return Err(format!("connect: {} handlers installed, {} expected", ours.len(), GROUPS.len()));
        }
        let (back, removed) = unplan_hooks(&written, Some(&p.created), file)?;
        if back.as_deref() != Some(*original) || removed.len() != GROUPS.len() {
            return Err(format!("connect: removal gave {back:?} for {original:?}"));
        }
        // With the install record lost, removal still gives back the original, except where the
        // file held an empty hooks container of its own (then the emptied container goes too).
        let (back2, _) = unplan_hooks(&written, None, file)?;
        let own_empty = original.contains("[]") || original.contains("\"hooks\": {}");
        if back2.as_deref() != Some(*original) && !own_empty {
            return Err(format!("connect: removal without a record gave {back2:?}"));
        }
    }
    let p = plan_hooks(None, exe, file, FALLBACK_TIMEOUT_S)?;
    let (gone, _) = unplan_hooks(p.text.as_deref().unwrap_or_default(), Some(&p.created), file)?;
    if gone.is_some() {
        return Err("connect: a file setup created was not removed".into());
    }
    for bad in ["{\"hooks\": [}", "", "[1]", "{\"hooks\": []}", "{\"hooks\": {\"Stop\": {}}}"] {
        if plan_hooks(Some(bad), exe, file, FALLBACK_TIMEOUT_S).is_ok() {
            return Err(format!("connect: {bad:?} was not refused"));
        }
    }
    // The approval: added once, kept beside other entries, removed exactly.
    for original in ["{\n  \"enabledMcpjsonServers\": [\"other\"]\n}\n", "{ \"keep\": 1 }"] {
        let p = plan_approval(Some(original))?;
        let w = p.text.ok_or("connect: approval write changed nothing")?;
        if plan_approval(Some(&w))?.text.is_some() || !w.contains(SERVER) {
            return Err("connect: second approval write changed the file".into());
        }
        let (back, was) = unplan_approval(&w, Some(&p.created))?;
        if !was || back.as_deref() != Some(original) {
            return Err(format!("connect: approval removal gave {back:?}"));
        }
    }
    let p = plan_approval(None)?;
    if unplan_approval(p.text.as_deref().unwrap_or_default(), Some(&p.created))?.0.is_some() {
        return Err("connect: an approval file setup created was not removed".into());
    }
    let p_lean = plan_approval_with(None, true)?;
    let w_lean = p_lean.text.as_deref().unwrap_or_default();
    if !w_lean.contains("\"enableArtifact\": false")
        || plan_approval_with(Some(w_lean), true)?.text.is_some()
        || unplan_approval(w_lean, Some(&p_lean.created))?.0.is_some()
    {
        return Err("connect: lean approval with enableArtifact=false did not round-trip".into());
    }
    let entry = mcp_entry(Path::new("C:\\Nano\\bin\\neuromorph-nano.exe"), Path::new("C:\\w\\shop"), false);
    for original in ["{\n  \"mcpServers\": {\n    \"other\": { \"command\": \"x\" } // theirs\n  }\n}\n", "{ \"keep\": 1 }"] {
        let p = plan_mcp(Some(original), &entry)?;
        let w = p.text.ok_or("connect: mcp write changed nothing")?;
        if plan_mcp(Some(&w), &entry)?.text.is_some() || mcp_installed(&w).is_none() {
            return Err("connect: second mcp write changed the file".into());
        }
        let (back, had) = unplan_mcp(&w, Some(&p.created))?;
        if back.as_deref() != Some(original) || !had {
            return Err(format!("connect: mcp removal gave {back:?}"));
        }
    }
    Ok(())
}

/// Settings files of varied shape with other people's entries and comments (used by the check
/// and by the product tests).
pub const FIXTURES: &[&str] = &[
    "{\n  \"permissions\": {\n    \"allow\": [\"Bash(git status)\"]\n  }\n}\n",
    "{\n  // personal settings\n  \"permissions\": { \"allow\": [\"Read\"] },\n  \"hooks\": {\n    \"Stop\": [\n      { \"hooks\": [ { \"type\": \"command\", \"command\": \"say done\" } ] }\n    ],\n    /* theirs */\n    \"PostToolUse\": [\n      { \"matcher\": \"Bash\", \"hooks\": [ { \"type\": \"command\", \"command\": \"log.sh\" } ] },\n    ],\n  },\n}\n",
    "{\r\n\t\"env\": {\"A\": \"1\"},\r\n\t\"hooks\": {}\r\n}",
    "{}",
    "{\"hooks\":{\"Stop\":[]},\"model\":\"x\"}",
    "{\n    \"hooks\": {\n        \"SessionStart\": [ { \"hooks\": [ { \"type\": \"command\", \"command\": \"C:\\\\Tools\\\\neuromorph-nano.exe hook stop\" } ] } ]\n    }\n}\n",
];

#[cfg(test)]
mod tests {
    #[test]
    fn round_trips() {
        super::check().unwrap();
    }
}
