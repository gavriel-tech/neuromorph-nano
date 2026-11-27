//! `neuromorph-nano connect <client>`: connects a repository to one MCP client in one step.
//! Claude Code and Antigravity go through the paths `setup` already has (`.mcp.json` with its
//! approval, and the Antigravity project plugin); Cursor gets `.cursor/mcp.json` and VS Code
//! `.vscode/mcp.json` in the project, Windsurf its user file `~/.codeium/windsurf/mcp_config.json`
//! (Roadmap Phase 4.6 (2026-11-25)), and any other stdio client the command line and JSON,
//! printed. Every write merges the Nano's one server entry into the file (other servers and
//! settings keep their bytes; an existing file is never replaced), checks the result against the
//! client's documented shape (`nmcore::mcp_clients`, each shape with its official page), writes a
//! backup of the file as it was into the project's memory folder, keeps an install record so
//! `--remove` takes out exactly that entry, and logs an audit row. `--dry-run` prints the line
//! difference the write would make and writes nothing.
use crate::connect::{self, SERVER};
use crate::jsonc::{Out, Style};
pub use nmcore::mcp_clients::Client;
use nmcore::mcp_clients::{self as clients, Scope};
use nmjson::Value;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

pub const USAGE: &str = "usage: neuromorph-nano connect <claude|antigravity|cursor|vscode|windsurf|generic> [--project <repo>] [--plain] [--hooks] [--dry-run] [--remove]
  claude       .mcp.json entry and its approval in .claude/settings.local.json (--hooks adds the hook entries)
  antigravity  the project plugin in .agents/plugins/neuromorph-nano/ (also: agy; --hooks adds hooks.json)
  cursor       .cursor/mcp.json in the project
  vscode       .vscode/mcp.json in the project (also: code)
  windsurf     ~/.codeium/windsurf/mcp_config.json (one file for every project: the entry serves this one)
  generic      print the command line and the mcpServers JSON for any other stdio client (also: stdio)
  --project    the repository (default: the current folder)
  --plain      serve the plain path (--flow off) instead of the one flow
  --dry-run    print what would change, line by line; write nothing
  --remove     take the Nano's entry out again, other entries kept";

/// Where a file-based client keeps its MCP settings: (path relative to the project, or to the
/// home folder for Windsurf; the key of the servers object; the install record's name). Claude
/// Code and Antigravity are written by `setup`'s own paths and the generic client is printed only.
pub fn client_file(c: Client) -> Option<(&'static str, &'static str, &'static str)> {
    match c {
        Client::Cursor | Client::VsCode | Client::Windsurf => clients::spec(c).map(|s| (s.file, s.key, s.record)),
        Client::Claude | Client::Antigravity | Client::Generic => None,
    }
}

/// The server entry for a client, in its documented shape (`nmcore::mcp_clients::entry`).
pub fn client_entry(c: Client, launcher: &Path, repo: &Path, plain: bool) -> Out {
    clients::entry(c, launcher, repo, plain)
}

/// The settings file of a file-based client for a project.
fn client_path(c: Client, repo: &Path) -> Result<PathBuf, String> {
    clients::file_for(c, repo, clients::user_home().as_deref()).ok_or_else(|| format!("{}: no home folder is known (USERPROFILE or HOME)", c.name()))
}

/// The parsed command line of `connect`.
#[derive(Debug, PartialEq, Eq)]
pub struct Args {
    pub client: Client,
    pub project: Option<String>,
    pub plain: bool,
    pub hooks: bool,
    pub dry_run: bool,
    pub remove: bool,
}

/// Reads `connect`'s arguments (without the word `connect`).
/// # Errors
/// No client, an unknown client or an unknown flag, as one line.
pub fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut client = None;
    let mut out = Args { client: Client::Claude, project: None, plain: false, hooks: false, dry_run: false, remove: false };
    let mut i = 0;
    while let Some(a) = args.get(i) {
        match a.as_str() {
            "--project" | "--root" => {
                out.project = Some(args.get(i + 1).ok_or_else(|| format!("{a} needs a folder"))?.clone());
                i += 1;
            }
            "--plain" => out.plain = true,
            "--hooks" => out.hooks = true,
            "--dry-run" | "--dry" => out.dry_run = true,
            "--remove" => out.remove = true,
            x if x.starts_with("--") => return Err(format!("unknown option {x}")),
            x if client.is_none() => client = Some(Client::parse(x).ok_or_else(|| format!("unknown client {x}"))?),
            x => return Err(format!("one client at a time; {x} is extra")),
        }
        i += 1;
    }
    out.client = client.ok_or("name the client: claude, antigravity, cursor, vscode, windsurf or generic")?;
    if out.client == Client::Generic && out.remove {
        return Err("the generic client is only printed; there is nothing to remove".to_owned());
    }
    if out.hooks && matches!(out.client, Client::Cursor | Client::VsCode | Client::Windsurf | Client::Generic) {
        return Err(format!("--hooks is for Claude Code and Antigravity; {} takes the MCP entry only", out.client.name()));
    }
    Ok(out)
}

/// `neuromorph-nano connect <client> ...`.
pub fn run(home: &Path, args: &[String]) -> ExitCode {
    let a = match parse_args(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("connect: {e}\n{USAGE}");
            return ExitCode::FAILURE;
        }
    };
    let repo = a.project.clone().map_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")), PathBuf::from);
    if !repo.is_dir() {
        eprintln!("connect: {} is not a folder", repo.display());
        return ExitCode::FAILURE;
    }
    let repo = nmsubs::repo::canon(&repo);
    println!("connect {} to {}{}", repo.display(), a.client.name(), if a.dry_run { " (dry run: nothing is written)" } else { "" });
    let r = match (a.client, a.remove) {
        (Client::Claude, true) if a.dry_run => Err("--remove with --dry-run: see `setup --project <repo> --status` for what is installed".to_owned()),
        (Client::Claude, true) => crate::project::remove(home, &repo),
        (Client::Claude, false) => {
            crate::project::setup_mcp(home, &repo, a.plain, false, !a.dry_run).and_then(|()| {
                if a.hooks { crate::project::setup_hooks(home, &repo, !a.dry_run) } else { Ok(()) }
            })
        }
        (Client::Antigravity, true) if a.dry_run => {
            println!("would remove {}", repo.join(".agents").join("plugins").join(SERVER).display());
            Ok(())
        }
        (Client::Antigravity, true) => crate::project::remove_agy(home, &repo),
        (Client::Antigravity, false) => crate::project::setup_agy(home, &repo, a.plain, a.hooks, !a.dry_run),
        (Client::Generic, _) => {
            println!("{}", clients::generic_text(&crate::launch::launcher_path(home), &repo, a.plain));
            println!("nothing written: add this to your client's MCP settings (the launcher at that path is copied from the newest build by any connect or setup --write)");
            Ok(())
        }
        (c, true) => remove_client(home, &repo, c, a.dry_run),
        (c, false) => write_client(home, &repo, c, a.plain, a.dry_run),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("connect: {e}");
            ExitCode::FAILURE
        }
    }
}

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}

/// Writes (or with `dry_run` shows) the Nano's entry in a file-based client's settings file.
fn write_client(home: &Path, repo: &Path, c: Client, plain: bool, dry_run: bool) -> Result<(), String> {
    let Some((rel, key, record)) = client_file(c) else { return Err(format!("{} has no settings file here", c.name())) };
    let file = client_path(c, repo)?;
    let existing = crate::project::read_text(&file)?;
    let launcher = crate::launch::launcher_path(home);
    let entry = client_entry(c, &launcher, repo, plain);
    let plan = connect::plan_servers(existing.as_deref(), rel, key, &entry).map_err(|x| format!("refused, nothing changed: {x}"))?;
    let st = Style { nl: "\n", unit: "  ".into() };
    println!("entry for {} (server \"{SERVER}\" in \"{key}\"):\n{}", c.name(), crate::jsonc::pretty(&entry, "", &st));
    if let Some(s) = clients::spec(c) {
        println!("format: {} (as documented at {})", s.file, s.source);
        if s.scope == Scope::User {
            println!("{} reads one file for every project: this entry serves {}; connecting another project replaces the Nano's entry, other servers kept", c.name(), repo.display());
        }
    }
    println!("memory folder: {}", nmtype::project::memory_dir(home, repo).display());
    let Some(text) = plan.text else {
        println!("{} already holds this entry; nothing to change", file.display());
        return Ok(());
    };
    // Roadmap Phase 4.6 (2026-11-25): what is written is checked against the client's documented
    // shape first; a file that would not fit is refused, nothing changed.
    clients::check_shape(c, &text).map_err(|x| format!("refused, nothing changed: the merged {} would not fit {}'s format ({x})", file.display(), c.name()))?;
    if dry_run {
        println!(
            "{} {} (line difference; - taken out, + put in):\n{}",
            if existing.is_some() { "would merge into" } else { "would write" },
            file.display(),
            clients::diff(existing.as_deref(), &text)
        );
        println!("launcher: {} (copied from the newest build when written)", launcher.display());
        println!("nothing written (dry run)");
        return Ok(());
    }
    if crate::project::is_user_home(repo) {
        return Err(format!("refused: {} is your home folder; nothing changed", repo.display()));
    }
    println!("{}", crate::launch::write_launcher(home)?);
    let fresh = existing.as_deref().and_then(|t| entry_present(t, key)).is_none();
    let bak = connect::backup(home, repo, &file, existing.as_deref().map(str::as_bytes))?;
    let made_dir = connect::write_file(&file, &text)?;
    if fresh {
        let mut created = plan.created.clone();
        created.dir = made_dir;
        connect::save_record(home, repo, record, Some(&created))?;
    }
    let _ = crate::controls::audit(
        home,
        "setup",
        vec![
            ("label", s(&format!("{} MCP entry written", c.name()))),
            ("project", s(&nmtype::project::label(home, repo))),
            ("file", s(&file.display().to_string())),
            ("merged", Value::Bool(existing.is_some())),
            ("backup", s(&bak.display().to_string())),
        ],
    );
    println!(
        "wrote {}: {} the server entry \"{SERVER}\"; backup of the file as it was: {}",
        file.display(),
        if existing.is_some() { "merged in, other entries kept," } else { "a new file holding" },
        bak.display()
    );
    Ok(())
}

/// Takes the Nano's entry out of a file-based client's settings file, other entries kept; the file
/// (and its folder) go only when the Nano created them and nothing else is left.
fn remove_client(home: &Path, repo: &Path, c: Client, dry_run: bool) -> Result<(), String> {
    let Some((rel, key, record)) = client_file(c) else { return Err(format!("{} has no settings file here", c.name())) };
    let file = client_path(c, repo)?;
    let Some(text) = crate::project::read_text(&file)? else {
        println!("{}: not present", file.display());
        return Ok(());
    };
    let created = connect::record(home, repo, record);
    let (after, had) = connect::unplan_servers(&text, rel, key, created.as_ref())?;
    if !had {
        println!("{} holds no \"{SERVER}\" entry; nothing to change", file.display());
        return Ok(());
    }
    if dry_run {
        match &after {
            Some(t) => println!("would write {} (line difference):\n{}", file.display(), clients::diff(Some(&text), t)),
            None => println!("would remove {} (the Nano created it and nothing else is left)", file.display()),
        }
        return Ok(());
    }
    let bak = connect::backup(home, repo, &file, Some(text.as_bytes()))?;
    match &after {
        Some(t) => {
            connect::write_file(&file, t)?;
        }
        None => {
            std::fs::remove_file(&file).map_err(|e| format!("{}: {e}", file.display()))?;
            if created.as_ref().is_some_and(|c| c.dir)
                && let Some(d) = file.parent()
            {
                let _ = std::fs::remove_dir(d);
            }
        }
    }
    connect::save_record(home, repo, record, None)?;
    let _ = crate::controls::audit(
        home,
        "setup",
        vec![
            ("label", s(&format!("{} MCP entry removed", c.name()))),
            ("project", s(&nmtype::project::label(home, repo))),
            ("file", s(&file.display().to_string())),
            ("backup", s(&bak.display().to_string())),
        ],
    );
    println!("removed the \"{SERVER}\" entry from {}; backup: {}", file.display(), bak.display());
    Ok(())
}

/// One command-line word for a copyable command, quoted when it holds a space or a quote.
fn shell_word(w: &str) -> String {
    if w.is_empty() || w.contains([' ', '"', '\'']) { format!("\"{}\"", w.replace('"', "\\\"")) } else { w.to_owned() }
}

/// `GET /api/connect?project=<folder or project key>` (2026-11-26, Phase B item B4): for each MCP
/// client `connect` writes for, whether the Nano's entry is in that client's settings file for this
/// project (read the way `connect` writes it), the entry's command, and the exact copyable command
/// lines to connect (and to preview with `--dry-run`). Read only: nothing is written.
#[must_use]
pub fn status_view(home: &Path, query: &str) -> Value {
    let Some(given) = crate::project::query_arg(query, "project").filter(|p| !p.is_empty()) else {
        return Value::Object([("error".to_owned(), s("project: give the project's folder (or its key as /api/projects lists it)"))].into_iter().collect());
    };
    // A key of a known project names its folder; else the text is the folder.
    let from_key = crate::ledger::project_memory(home, &format!("project={given}"))
        .and_then(|(_, m)| std::fs::read(m.join("project.json")).ok())
        .and_then(|b| nmjson::parse(&b).ok())
        .and_then(|v| v.get("root").and_then(Value::string).ok().map(PathBuf::from));
    let folder = from_key.unwrap_or_else(|| PathBuf::from(&given));
    if !folder.is_dir() {
        return Value::Object([("error".to_owned(), s(&format!("{}: not a folder on this machine", folder.display())))].into_iter().collect());
    }
    let repo = nmsubs::repo::canon(&folder);
    let shown = repo.display().to_string();
    let launcher = crate::launch::launcher_path(home);
    let user = clients::user_home();
    let rows: Vec<Value> = clients::CLIENTS
        .iter()
        .map(|c| {
            let cli = match c {
                Client::Claude => "claude",
                Client::Antigravity => "antigravity",
                Client::Cursor => "cursor",
                Client::VsCode => "vscode",
                Client::Windsurf => "windsurf",
                Client::Generic => "generic",
            };
            let command = format!("neuromorph-nano connect {cli} --project {}", shell_word(&shown));
            let mut f: Vec<(String, Value)> = vec![
                ("client".into(), s(cli)),
                ("name".into(), s(c.name())),
                ("command".into(), s(&command)),
                ("dry_run_command".into(), s(&format!("{command} --dry-run"))),
            ];
            match clients::spec(*c) {
                None => {
                    f.extend([
                        ("scope".into(), s("printed")),
                        ("file".into(), Value::Null),
                        ("connected".into(), Value::Null),
                        ("preview".into(), s(&clients::generic_text(&launcher, &repo, false))),
                        ("note".into(), s("a generic stdio client is given the command and JSON to add to its own settings; nothing is written, so there is nothing to read back")),
                    ]);
                }
                Some(sp) => {
                    let file = clients::file_for(*c, &repo, user.as_deref());
                    let text = file.as_ref().and_then(|p| std::fs::read_to_string(p).ok());
                    let entry = text.as_deref().and_then(|t| crate::jsonc::parse(t).ok()).and_then(|root| {
                        let e = root.get(sp.key)?.get(SERVER)?;
                        let cmd = e.get("command").and_then(crate::jsonc::Node::str).map(str::to_owned);
                        let args: Vec<String> = e.get("args").and_then(crate::jsonc::Node::items).map(|i| i.iter().filter_map(|x| x.str().map(str::to_owned)).collect()).unwrap_or_default();
                        Some((cmd, args))
                    });
                    let serves = entry.as_ref().is_some_and(|(_, args)| args.windows(2).any(|w| w.first().is_some_and(|a| a == "--root") && w.get(1).is_some_and(|r| nmsubs::repo::canon(Path::new(r)) == repo)));
                    let user_scope = sp.scope == Scope::User;
                    f.extend([
                        ("scope".into(), s(if user_scope { "user" } else { "project" })),
                        ("file".into(), file.as_ref().map_or(Value::Null, |p| s(&p.display().to_string()))),
                        ("file_exists".into(), Value::Bool(text.is_some())),
                        ("entry".into(), Value::Bool(entry.is_some())),
                        ("serves_project".into(), Value::Bool(serves)),
                        ("connected".into(), Value::Bool(entry.is_some() && serves)),
                        ("entry_command".into(), entry.as_ref().and_then(|(c0, _)| c0.as_deref()).map_or(Value::Null, s)),
                        ("launcher_matches".into(), Value::Bool(entry.as_ref().and_then(|(c0, _)| c0.as_deref()).is_some_and(|c0| Path::new(c0) == launcher))),
                        ("source".into(), s(sp.source)),
                        (
                            "note".into(),
                            s(if user_scope { "one file for every project: connected means its entry serves this folder" } else { "the entry in this project's own settings file" }),
                        ),
                    ]);
                }
            }
            Value::Object(f.into_iter().collect())
        })
        .collect();
    Value::Object(
        [
            ("project".to_owned(), s(&shown)),
            ("launcher".to_owned(), s(&launcher.display().to_string())),
            ("launcher_present".to_owned(), Value::Bool(launcher.is_file())),
            ("clients".to_owned(), Value::Array(rows)),
        ]
        .into_iter()
        .collect(),
    )
}

/// The command of the Nano's entry under `key` in a settings text, if there is one.
fn entry_present(text: &str, key: &str) -> Option<String> {
    let root = crate::jsonc::parse(text).ok()?;
    let e = root.get(key)?.get(SERVER)?;
    e.get("command").and_then(crate::jsonc::Node::str).map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| (*x).to_owned()).collect()
    }

    #[test]
    fn parses_clients_and_flags() {
        let a = parse_args(&v(&["cursor", "--project", "D:\\w\\shop", "--dry-run"])).unwrap();
        assert_eq!(a.client, Client::Cursor);
        assert_eq!(a.project.as_deref(), Some("D:\\w\\shop"));
        assert!(a.dry_run && !a.remove && !a.plain);
        assert_eq!(parse_args(&v(&["agy", "--hooks"])).unwrap().client, Client::Antigravity);
        assert_eq!(parse_args(&v(&["code", "--remove"])).unwrap().client, Client::VsCode);
        assert_eq!(parse_args(&v(&["Claude", "--plain"])).unwrap().client, Client::Claude);
        assert!(parse_args(&v(&[])).is_err());
        assert!(parse_args(&v(&["emacs"])).is_err());
        assert!(parse_args(&v(&["cursor", "vscode"])).is_err());
        assert!(parse_args(&v(&["cursor", "--hooks"])).is_err());
        assert!(parse_args(&v(&["cursor", "--project"])).is_err());
        assert!(parse_args(&v(&["cursor", "--write"])).is_err());
        assert_eq!(parse_args(&v(&["windsurf", "--dry-run"])).unwrap().client, Client::Windsurf);
        assert_eq!(parse_args(&v(&["stdio"])).unwrap().client, Client::Generic);
        assert!(parse_args(&v(&["generic", "--remove"])).is_err());
        assert!(parse_args(&v(&["windsurf", "--hooks"])).is_err());
    }

    #[test]
    fn entries_per_client() {
        let l = Path::new("D:\\Nano\\bin\\neuromorph-nano.exe");
        let r = Path::new("D:\\w\\shop");
        let cur = client_entry(Client::Cursor, l, r, false);
        let code = client_entry(Client::VsCode, l, r, true);
        let st = Style { nl: "\n", unit: "  ".into() };
        let ct = crate::jsonc::pretty(&cur, "", &st);
        let vt = crate::jsonc::pretty(&code, "", &st);
        // Roadmap Phase 4.6 (2026-11-25): Cursor's field table lists "type": "stdio" as required.
        assert!(ct.contains("\"type\": \"stdio\"") && ct.contains("\"--flow\""));
        assert!(vt.contains("\"type\": \"stdio\"") && vt.contains("\"off\""));
        let wt = crate::jsonc::pretty(&client_entry(Client::Windsurf, l, r, false), "", &st);
        assert!(!wt.contains("\"type\"") && wt.contains("\"command\""));
        assert_eq!(client_file(Client::VsCode).map(|f| f.1), Some("servers"));
        assert_eq!(client_file(Client::Cursor).map(|f| f.0), Some(".cursor/mcp.json"));
        assert_eq!(client_file(Client::Windsurf).map(|f| f.0), Some(".codeium/windsurf/mcp_config.json"));
        assert!(client_file(Client::Claude).is_none() && client_file(Client::Generic).is_none());
    }

    /// Other servers survive a write byte for byte, a second write changes nothing, and removal
    /// gives back the file as it was, for both file-based clients.
    #[test]
    fn merges_and_removes_beside_other_servers() {
        let l = Path::new("D:\\Nano\\bin\\neuromorph-nano.exe");
        let r = Path::new("D:\\w\\shop");
        let cases = [
            (Client::Cursor, "{\n  \"mcpServers\": {\n    \"github\": { \"command\": \"gh-mcp\" } // theirs\n  }\n}\n"),
            (Client::VsCode, "{\n  // workspace servers\n  \"inputs\": [],\n  \"servers\": {\n    \"db\": { \"type\": \"stdio\", \"command\": \"db-mcp\" },\n  },\n}\n"),
            (Client::VsCode, "{ \"inputs\": [] }"),
        ];
        for (c, original) in cases {
            let (rel, key, _) = client_file(c).unwrap();
            let e = client_entry(c, l, r, false);
            let p = connect::plan_servers(Some(original), rel, key, &e).unwrap();
            let w = p.text.clone().unwrap();
            assert!(w.contains("gh-mcp") || w.contains("db-mcp") || !original.contains("mcp\""));
            assert!(entry_present(&w, key).is_some());
            assert!(connect::plan_servers(Some(&w), rel, key, &e).unwrap().text.is_none());
            let (back, had) = connect::unplan_servers(&w, rel, key, Some(&p.created)).unwrap();
            assert!(had);
            assert_eq!(back.as_deref(), Some(original));
        }
        // A file the Nano created goes again whole.
        let e = client_entry(Client::Cursor, l, r, false);
        let p = connect::plan_servers(None, ".cursor/mcp.json", "mcpServers", &e).unwrap();
        let (gone, _) = connect::unplan_servers(p.text.as_deref().unwrap(), ".cursor/mcp.json", "mcpServers", Some(&p.created)).unwrap();
        assert!(gone.is_none());
        // A servers value that is not an object is refused.
        assert!(connect::plan_servers(Some("{\"servers\": []}"), ".vscode/mcp.json", "servers", &e).is_err());
        assert!(connect::plan_servers(Some("{bad"), ".vscode/mcp.json", "servers", &e).is_err());
    }
}
