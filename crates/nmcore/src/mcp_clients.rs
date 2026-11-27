//! The MCP clients `neuromorph-nano setup` and `connect` write for, and how each one's settings
//! file is shaped (Roadmap Phase 4.6 (2026-11-25)). Each client's documented format was read from
//! its official page on 2026-11-25; the page is named beside it. The Nano's one server entry is
//! merged into a client's file (every other server and setting keeps its bytes, `jsonc`), checked
//! against the client's shape, and shown as a line difference before anything is written.
//!
//! - Claude Code: `.mcp.json` in the project, `mcpServers`, `"type": "stdio"` with `command` and
//!   `args` (the `claude mcp add-json` example shows `type` stdio; an entry without one is read as
//!   stdio too). <https://code.claude.com/docs/en/mcp>
//! - Cursor: `.cursor/mcp.json` in the project, `mcpServers`; its field table lists `type`
//!   (`"stdio"`) as required with `command`, `args` optional. <https://cursor.com/docs/context/mcp>
//! - VS Code: `.vscode/mcp.json` in the project, `servers` (beside an optional `inputs` array);
//!   `type` (`"stdio"`) and `command` required, `args` optional.
//!   <https://code.visualstudio.com/docs/copilot/reference/mcp-configuration>
//! - Antigravity: a workspace plugin, `.agents/plugins/<name>/` with `plugin.json` (only `name`
//!   required, lowercase kebab case) and `mcp_config.json` at the plugin root holding
//!   `mcpServers` with `command` and `args` (no `type`).
//!   <https://antigravity.google/docs/build-plugins>, <https://antigravity.google/docs/mcp>
//! - Windsurf: the user file `~/.codeium/windsurf/mcp_config.json`, `mcpServers` with `command`,
//!   `args` and `env` (no `type`; the Claude Desktop schema). The page is now served at
//!   <https://docs.devin.ai/windsurf/plugins/cascade/mcp> (from
//!   <https://docs.windsurf.com/windsurf/cascade/mcp>).
//! - Any other stdio client: the command line and the `mcpServers` JSON every client that follows
//!   the Claude Desktop form reads, printed, nothing written.
//!   <https://modelcontextprotocol.io/docs/develop/connect-local-servers>
use crate::jsonc::{self, Node, Out, Style};
use nmjson::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The MCP server name every entry uses.
pub const SERVER: &str = "neuromorph-nano";

/// The MCP clients `connect` writes for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Client {
    Claude,
    Antigravity,
    Cursor,
    VsCode,
    Windsurf,
    Generic,
}

/// Every client, in the order `connect` lists them.
pub const CLIENTS: [Client; 6] = [Client::Claude, Client::Antigravity, Client::Cursor, Client::VsCode, Client::Windsurf, Client::Generic];

impl Client {
    /// The client a command-line name means.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "claude" | "claude-code" | "claudecode" => Some(Self::Claude),
            "antigravity" | "agy" => Some(Self::Antigravity),
            "cursor" => Some(Self::Cursor),
            "vscode" | "vs-code" | "code" => Some(Self::VsCode),
            "windsurf" | "codeium" => Some(Self::Windsurf),
            "generic" | "stdio" | "other" => Some(Self::Generic),
            _ => None,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Claude => "Claude Code",
            Self::Antigravity => "Antigravity",
            Self::Cursor => "Cursor",
            Self::VsCode => "VS Code",
            Self::Windsurf => "Windsurf",
            Self::Generic => "a generic stdio MCP client",
        }
    }
}

/// Where a client's settings file lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// Under the project folder.
    Project,
    /// Under the user's home folder (one file for every project).
    User,
}

/// One client's settings file and entry shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    /// The file, relative to the project (or to the home folder for `Scope::User`).
    pub file: &'static str,
    pub scope: Scope,
    /// The key of the servers object.
    pub key: &'static str,
    /// Whether the entry carries `"type": "stdio"` (the documented shape says so).
    pub typed: bool,
    /// The install record's name (what `--remove` takes out again).
    pub record: &'static str,
    /// The official page the shape was read from (2026-11-25).
    pub source: &'static str,
}

/// The settings file and entry shape of a client; `None` for the generic client (printed only).
#[must_use]
pub const fn spec(c: Client) -> Option<Spec> {
    match c {
        Client::Claude => Some(Spec { file: ".mcp.json", scope: Scope::Project, key: "mcpServers", typed: true, record: "mcp", source: "https://code.claude.com/docs/en/mcp" }),
        Client::Antigravity => Some(Spec {
            file: ".agents/plugins/neuromorph-nano/mcp_config.json",
            scope: Scope::Project,
            key: "mcpServers",
            typed: false,
            record: "agy-mcp",
            source: "https://antigravity.google/docs/build-plugins",
        }),
        Client::Cursor => Some(Spec { file: ".cursor/mcp.json", scope: Scope::Project, key: "mcpServers", typed: true, record: "cursor-mcp", source: "https://cursor.com/docs/context/mcp" }),
        Client::VsCode => Some(Spec {
            file: ".vscode/mcp.json",
            scope: Scope::Project,
            key: "servers",
            typed: true,
            record: "vscode-mcp",
            source: "https://code.visualstudio.com/docs/copilot/reference/mcp-configuration",
        }),
        Client::Windsurf => Some(Spec {
            file: ".codeium/windsurf/mcp_config.json",
            scope: Scope::User,
            key: "mcpServers",
            typed: false,
            record: "windsurf-mcp",
            source: "https://docs.devin.ai/windsurf/plugins/cascade/mcp",
        }),
        Client::Generic => None,
    }
}

/// The page the generic stdio form was read from.
pub const GENERIC_SOURCE: &str = "https://modelcontextprotocol.io/docs/develop/connect-local-servers";

/// The arguments the server runs with: `launch serve --root <repo>`, plus `--flow` (or
/// `--flow off` for the plain path).
#[must_use]
pub fn serve_args(repo: &Path, plain: bool) -> Vec<String> {
    let mut args = vec!["launch".to_owned(), "serve".to_owned(), "--root".to_owned(), repo.display().to_string()];
    if plain {
        args.extend(["--flow".to_owned(), "off".to_owned()]);
    } else {
        args.push("--flow".to_owned());
    }
    args
}

/// The `.mcp.json` entry for a repository: the launcher with `serve_args`.
#[must_use]
pub fn mcp_entry(launcher: &Path, repo: &Path, plain: bool) -> Out {
    entry(Client::Claude, launcher, repo, plain)
}

/// A client's server entry: `"type": "stdio"` first where the client's documented shape has it,
/// then the launcher and its arguments.
#[must_use]
pub fn entry(c: Client, launcher: &Path, repo: &Path, plain: bool) -> Out {
    let mut fields = Vec::new();
    if spec(c).is_some_and(|s| s.typed) {
        fields.push(("type".to_owned(), Out::Str("stdio".into())));
    }
    fields.push(("command".to_owned(), Out::Str(launcher.display().to_string())));
    fields.push(("args".to_owned(), Out::Arr(serve_args(repo, plain).into_iter().map(Out::Str).collect())));
    Out::Obj(fields)
}

/// A whole new settings file holding only the Nano's entry, as written when none exists.
#[must_use]
pub fn new_file_text(c: Client, launcher: &Path, repo: &Path, plain: bool) -> String {
    let key = spec(c).map_or("mcpServers", |s| s.key);
    let root = Out::Obj(vec![(key.to_owned(), Out::Obj(vec![(SERVER.to_owned(), entry(c, launcher, repo, plain))]))]);
    format!("{}\n", jsonc::pretty(&root, "", &Style { nl: "\n", unit: "  ".into() }))
}

/// One command-line word, quoted when it holds a space.
fn word(w: &str) -> String {
    if w.contains(' ') || w.is_empty() { format!("\"{w}\"") } else { w.to_owned() }
}

/// What a generic stdio client needs: the command line to start the server and the
/// `mcpServers` JSON.
#[must_use]
pub fn generic_text(launcher: &Path, repo: &Path, plain: bool) -> String {
    let line: Vec<String> = std::iter::once(launcher.display().to_string()).chain(serve_args(repo, plain)).map(|w| word(&w)).collect();
    format!(
        "command (the client starts it and speaks MCP over its standard input and output):\n  {}\nJSON for a client that reads the mcpServers form ({GENERIC_SOURCE}):\n{}",
        line.join(" "),
        new_file_text(Client::Generic, launcher, repo, plain)
    )
}

/// The `claude mcp add` line that writes the same project entry through Claude Code itself.
#[must_use]
pub fn claude_add_line(launcher: &Path, repo: &Path, plain: bool) -> String {
    let args: Vec<String> = serve_args(repo, plain).iter().map(|w| word(w)).collect();
    format!("claude mcp add --transport stdio --scope project {SERVER} -- {} {}", word(&launcher.display().to_string()), args.join(" "))
}

/// Checks a settings text against the client's documented shape: an object, its servers key an
/// object, the Nano's entry an object with a non-empty `command` string, `args` a list of strings,
/// `"type": "stdio"` present exactly where the client's shape has it, and no field outside
/// `type`, `command`, `args`, `env` and `cwd`.
/// # Errors
/// What does not fit, as one line.
pub fn check_shape(c: Client, text: &str) -> Result<(), String> {
    let s = spec(c).map_or(("mcpServers", false), |s| (s.key, s.typed));
    let root = jsonc::parse(text).map_err(|e| format!("not JSON: {e}"))?;
    let servers = root.get(s.0).ok_or_else(|| format!("no \"{}\" object", s.0))?;
    if servers.members().is_none() {
        return Err(format!("\"{}\" is not an object", s.0));
    }
    let e = servers.get(SERVER).ok_or_else(|| format!("no \"{SERVER}\" entry"))?;
    let fields = e.members().ok_or("the entry is not an object")?;
    if let Some(f) = fields.iter().find(|m| !["type", "command", "args", "env", "cwd"].contains(&m.key.as_str())) {
        return Err(format!("unexpected field \"{}\"", f.key));
    }
    if e.get("command").and_then(Node::str).is_none_or(str::is_empty) {
        return Err("no command".into());
    }
    if let Some(a) = e.get("args")
        && a.items().is_none_or(|i| i.iter().any(|x| x.str().is_none()))
    {
        return Err("args is not a list of strings".into());
    }
    match (s.1, e.get("type").map(Node::str)) {
        (true, Some(Some("stdio"))) | (false, None) => Ok(()),
        (true, _) => Err(format!("{} needs \"type\": \"stdio\"", c.name())),
        (false, Some(_)) => Err(format!("{}'s documented shape has no \"type\"", c.name())),
    }
}

/// What an install created, so removal takes out exactly that and leaves what was there.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Created {
    pub file: bool,
    pub dir: bool,
    pub container: bool,
    pub events: Vec<String>,
}

impl Created {
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut o = BTreeMap::new();
        o.insert("file".to_owned(), Value::Bool(self.file));
        o.insert("dir".to_owned(), Value::Bool(self.dir));
        o.insert("container".to_owned(), Value::Bool(self.container));
        o.insert("events".to_owned(), Value::Array(self.events.iter().map(|e| Value::String(e.clone())).collect()));
        Value::Object(o)
    }
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        let flag = |k: &str| matches!(v.get(k), Ok(Value::Bool(true)));
        Self {
            file: flag("file"),
            dir: flag("dir"),
            container: flag("container"),
            events: v.get("events").and_then(Value::array).unwrap_or_default().iter().filter_map(|e| e.string().ok().map(str::to_owned)).collect(),
        }
    }
}

/// The result of planning a write: the new text (or `None` when nothing changes) and what it
/// creates.
#[derive(Debug)]
pub struct Plan {
    pub text: Option<String>,
    pub created: Created,
}

/// The root object of a settings text, or why it cannot be edited.
/// # Errors
/// Not JSON, or the top level is not an object.
pub fn root_of(text: &str, file: &str) -> Result<Node, String> {
    let root = jsonc::parse(text).map_err(|e| format!("{file} is not valid JSON: {e}"))?;
    if root.members().is_none() {
        return Err(format!("{file}: the top level is not a JSON object"));
    }
    Ok(root)
}

/// True for a text holding only an empty JSON object.
#[must_use]
pub fn is_empty_object(t: &str) -> bool {
    jsonc::parse(t).ok().is_some_and(|r| r.members().is_some_and(<[jsonc::Member]>::is_empty))
        && t.trim_start_matches('\u{feff}').chars().filter(|c| !c.is_whitespace()).collect::<String>() == "{}"
}

/// Structural equality with object keys in any order.
#[must_use]
pub fn same(a: &Out, b: &Out) -> bool {
    match (a, b) {
        (Out::Obj(x), Out::Obj(y)) => x.len() == y.len() && x.iter().all(|(k, v)| y.iter().any(|(k2, v2)| k == k2 && same(v, v2))),
        (Out::Arr(x), Out::Arr(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q)),
        _ => a == b,
    }
}

/// Plans the Nano's server entry in any client's MCP settings text (`None`: no file yet): `key`
/// names the object of servers (`mcpServers`, or `servers` for VS Code); every other server's
/// entry and every other setting keeps its bytes, and an existing file is merged, never replaced.
/// # Errors
/// The text is not JSON, or the servers object is not an object.
pub fn plan_servers(text: Option<&str>, file: &str, key: &str, entry: &Out) -> Result<Plan, String> {
    let servers = |e: &Out| Out::Obj(vec![(SERVER.to_owned(), e.clone())]);
    let Some(text) = text else {
        let st = Style { nl: "\n", unit: "  ".into() };
        let whole = Out::Obj(vec![(key.to_owned(), servers(entry))]);
        return Ok(Plan { text: Some(format!("{}\n", jsonc::pretty(&whole, "", &st))), created: Created { file: true, container: true, ..Created::default() } });
    };
    let st = Style::of(text);
    let root = root_of(text, file)?;
    let mut created = Created::default();
    let t = match root.get(key) {
        None => {
            created.container = true;
            jsonc::append(text, &root, Some(key), &servers(entry), &st)
        }
        Some(s) if s.members().is_none() => return Err(format!("{file}: {key} is not an object")),
        Some(s) => match s.member(SERVER) {
            Some(m) if same(&jsonc::to_out(&m.value), entry) => text.to_owned(),
            Some(m) => jsonc::replace_value(text, m, entry, &st),
            None => jsonc::append(text, s, Some(SERVER), entry, &st),
        },
    };
    Ok(Plan { text: (t != text).then_some(t), created })
}

/// Plans removing the Nano's server entry from any client's MCP settings text (`key` as in
/// `plan_servers`): (new text, or `None` to remove a file the Nano created and that holds nothing
/// else; whether an entry was there).
/// # Errors
/// The text is not JSON, or the servers object is not an object.
pub fn unplan_servers(text: &str, file: &str, key: &str, created: Option<&Created>) -> Result<(Option<String>, bool), String> {
    let root = root_of(text, file)?;
    let Some(s) = root.get(key) else { return Ok((Some(text.to_owned()), false)) };
    let Some(i) = s.members().ok_or_else(|| format!("{file}: {key} is not an object"))?.iter().position(|m| m.key == SERVER) else {
        return Ok((Some(text.to_owned()), false));
    };
    let mut t = jsonc::remove(text, s, i);
    let root = root_of(&t, file)?;
    if let Some(s) = root.get(key)
        && s.members().is_some_and(<[jsonc::Member]>::is_empty)
        && created.is_none_or(|c| c.container)
        && let Some(j) = root.members().unwrap_or_default().iter().position(|m| m.key == key)
    {
        t = jsonc::remove(&t, &root, j);
    }
    Ok((if created.is_some_and(|c| c.file) && is_empty_object(&t) { None } else { Some(t) }, true))
}

/// The largest product of line counts the difference is worked out for (a settings file is a
/// few dozen lines; past this the whole new text is shown instead).
const DIFF_CELLS: usize = 4_000_000;

/// What a write would change, line by line: `- ` for a line taken out, `+ ` for a line put in,
/// one unchanged line of context around each change (`  `), `...` between distant changes. A
/// file that does not exist yet is shown whole, every line `+ `.
#[must_use]
pub fn diff(old: Option<&str>, new: &str) -> String {
    let b: Vec<&str> = new.lines().collect();
    let Some(old) = old else { return b.iter().map(|l| format!("+ {l}\n")).collect() };
    let a: Vec<&str> = old.lines().collect();
    if a.len().saturating_mul(b.len()) > DIFF_CELLS {
        return format!("(too long to compare line by line; the new text)\n{new}");
    }
    // Longest common subsequence table, from the ends.
    let w = b.len() + 1;
    let mut lcs = vec![0_u32; (a.len() + 1) * w];
    for i in (0..a.len()).rev() {
        for j in (0..b.len()).rev() {
            let v = if a.get(i) == b.get(j) {
                lcs.get((i + 1) * w + j + 1).copied().unwrap_or(0) + 1
            } else {
                lcs.get((i + 1) * w + j).copied().unwrap_or(0).max(lcs.get(i * w + j + 1).copied().unwrap_or(0))
            };
            if let Some(c) = lcs.get_mut(i * w + j) {
                *c = v;
            }
        }
    }
    let at = |i: usize, j: usize| lcs.get(i * w + j).copied().unwrap_or(0);
    let mut rows: Vec<(char, &str)> = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        if i < a.len() && j < b.len() && a.get(i) == b.get(j) {
            rows.push((' ', a.get(i).copied().unwrap_or_default()));
            i += 1;
            j += 1;
        } else if i < a.len() && (j == b.len() || at(i + 1, j) >= at(i, j + 1)) {
            rows.push(('-', a.get(i).copied().unwrap_or_default()));
            i += 1;
        } else {
            rows.push(('+', b.get(j).copied().unwrap_or_default()));
            j += 1;
        }
    }
    let near = |k: usize| rows.get(k.saturating_sub(1)..(k + 2).min(rows.len())).is_some_and(|r| r.iter().any(|(c, _)| *c != ' '));
    let mut out = String::new();
    let mut skipped = false;
    for (k, (c, l)) in rows.iter().enumerate() {
        if *c == ' ' && !near(k) {
            skipped = true;
            continue;
        }
        if skipped && !out.is_empty() {
            out.push_str("...\n");
        }
        skipped = false;
        out.push_str(&format!("{c} {l}\n"));
    }
    if out.is_empty() { "(no change)\n".to_owned() } else { out }
}

/// The user's home folder (`USERPROFILE`, else `HOME`).
#[must_use]
pub fn user_home() -> Option<PathBuf> {
    ["USERPROFILE", "HOME"].iter().filter_map(std::env::var_os).find(|h| !h.is_empty()).map(PathBuf::from)
}

/// Where a client's settings file is for a project (`None` for the generic client, or a user
/// file when no home folder is known).
#[must_use]
pub fn file_for(c: Client, repo: &Path, home: Option<&Path>) -> Option<PathBuf> {
    let s = spec(c)?;
    match s.scope {
        Scope::Project => Some(repo.join(s.file)),
        Scope::User => home.map(|h| h.join(s.file)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths() -> (&'static Path, &'static Path) {
        (Path::new("D:\\Nano Home\\bin\\neuromorph-nano.exe"), Path::new("D:\\w\\shop"))
    }

    /// Each client's generated file matches its documented shape, as a new file and merged into a
    /// file holding other servers and settings; other servers keep their bytes, a second write
    /// changes nothing, and removal gives the file back as it was.
    #[test]
    fn mcp_clients_shapes_and_merges() {
        let (l, r) = paths();
        let others: &[(Client, &str)] = &[
            (Client::Claude, "{\n  \"mcpServers\": {\n    \"github\": { \"command\": \"gh-mcp\" } // theirs\n  }\n}\n"),
            (Client::Cursor, "{\n  \"mcpServers\": {\n    \"github\": { \"type\": \"stdio\", \"command\": \"gh-mcp\", \"args\": [] }\n  }\n}\n"),
            (Client::VsCode, "{\n  // workspace servers\n  \"inputs\": [ { \"type\": \"promptString\", \"id\": \"key\" } ],\n  \"servers\": {\n    \"db\": { \"type\": \"stdio\", \"command\": \"db-mcp\" },\n  },\n}\n"),
            (Client::Windsurf, "{\r\n\t\"mcpServers\": {\r\n\t\t\"github\": {\"command\": \"npx\", \"args\": [\"-y\", \"@modelcontextprotocol/server-github\"], \"env\": {\"TOKEN\": \"x\"}}\r\n\t}\r\n}\r\n"),
            (Client::Antigravity, "{ \"mcpServers\": {} }"),
            (Client::VsCode, "{ \"inputs\": [] }"),
        ];
        for c in CLIENTS {
            let Some(s) = spec(c) else { continue };
            let e = entry(c, l, r, false);
            let fresh = plan_servers(None, s.file, s.key, &e).unwrap();
            let text = fresh.text.clone().unwrap();
            check_shape(c, &text).unwrap_or_else(|x| panic!("{}: {x}\n{text}", c.name()));
            assert_eq!(text, new_file_text(c, l, r, false));
            assert!(fresh.created.file && fresh.created.container);
            assert_eq!(unplan_servers(&text, s.file, s.key, Some(&fresh.created)).unwrap().0, None);
            assert!(s.source.starts_with("https://"));
        }
        for (c, original) in others {
            let s = spec(*c).unwrap();
            let e = entry(*c, l, r, true);
            let p = plan_servers(Some(original), s.file, s.key, &e).unwrap();
            let w = p.text.clone().unwrap();
            check_shape(*c, &w).unwrap_or_else(|x| panic!("{}: {x}\n{w}", c.name()));
            for keep in ["gh-mcp", "db-mcp", "promptString", "TOKEN", "// theirs", "// workspace servers"] {
                assert_eq!(original.contains(keep), w.contains(keep), "{keep} in {}", c.name());
            }
            assert!(!original.contains("\r\n") || !w.replace("\r\n", "").contains('\n'), "line breaks kept");
            assert!(plan_servers(Some(&w), s.file, s.key, &e).unwrap().text.is_none());
            let (back, had) = unplan_servers(&w, s.file, s.key, Some(&p.created)).unwrap();
            assert!(had);
            assert_eq!(back.as_deref(), Some(*original));
            // The dry run shows only additions for a merge.
            let d = diff(Some(original), &w);
            assert!(d.contains("+ ") && d.contains(SERVER), "{d}");
        }
        // A changed entry (the plain path) replaces the Nano's entry only.
        let s = spec(Client::Cursor).unwrap();
        let w = plan_servers(None, s.file, s.key, &entry(Client::Cursor, l, r, false)).unwrap().text.unwrap();
        let w2 = plan_servers(Some(&w), s.file, s.key, &entry(Client::Cursor, l, r, true)).unwrap().text.unwrap();
        let d = diff(Some(&w), &w2);
        assert!(w2.contains("\"off\"") && d.lines().any(|x| x.starts_with("+ ") && x.contains("\"off\"")) && !d.contains("- {"), "{d}");
    }

    #[test]
    fn mcp_clients_shape_check_refuses_wrong_shapes() {
        let ok = "{\"mcpServers\": {\"neuromorph-nano\": {\"type\": \"stdio\", \"command\": \"x\", \"args\": [\"a\"]}}}";
        check_shape(Client::Cursor, ok).unwrap();
        // VS Code reads `servers`, not `mcpServers`.
        assert!(check_shape(Client::VsCode, ok).is_err());
        // Windsurf and Antigravity have no `type`; Claude Code, Cursor and VS Code have it.
        assert!(check_shape(Client::Windsurf, ok).is_err());
        assert!(check_shape(Client::Claude, "{\"mcpServers\": {\"neuromorph-nano\": {\"command\": \"x\"}}}").is_err());
        assert!(check_shape(Client::Windsurf, "{\"mcpServers\": {\"neuromorph-nano\": {\"command\": \"x\", \"args\": [1]}}}").is_err());
        assert!(check_shape(Client::Windsurf, "{\"mcpServers\": {\"neuromorph-nano\": {\"command\": \"\"}}}").is_err());
        assert!(check_shape(Client::Windsurf, "{\"mcpServers\": {\"neuromorph-nano\": {\"command\": \"x\", \"url\": \"y\"}}}").is_err());
        assert!(check_shape(Client::Windsurf, "{\"mcpServers\": []}").is_err());
        // Bad files are refused by the merge, nothing changed.
        let e = entry(Client::VsCode, Path::new("x"), Path::new("y"), false);
        assert!(plan_servers(Some("{\"servers\": []}"), ".vscode/mcp.json", "servers", &e).is_err());
        assert!(plan_servers(Some("{bad"), ".vscode/mcp.json", "servers", &e).is_err());
        assert!(plan_servers(Some("[1]"), ".vscode/mcp.json", "servers", &e).is_err());
    }

    #[test]
    fn mcp_clients_parse_paths_and_generic() {
        assert_eq!(Client::parse("Windsurf"), Some(Client::Windsurf));
        assert_eq!(Client::parse("stdio"), Some(Client::Generic));
        assert_eq!(Client::parse("agy"), Some(Client::Antigravity));
        assert_eq!(Client::parse("emacs"), None);
        let (l, r) = paths();
        let home = Path::new("C:\\Users\\someone");
        assert_eq!(file_for(Client::Windsurf, r, Some(home)), Some(home.join(".codeium/windsurf/mcp_config.json")));
        assert_eq!(file_for(Client::VsCode, r, Some(home)), Some(r.join(".vscode/mcp.json")));
        assert_eq!(file_for(Client::Windsurf, r, None), None);
        assert_eq!(file_for(Client::Generic, r, Some(home)), None);
        let g = generic_text(l, r, false);
        assert!(g.contains("\"D:\\Nano Home\\bin\\neuromorph-nano.exe\" launch serve --root D:\\w\\shop --flow"), "{g}");
        let json = g.rsplit_once("):\n").map(|x| x.1).unwrap();
        check_shape(Client::Generic, json).unwrap();
        let add = claude_add_line(l, r, true);
        assert!(add.starts_with("claude mcp add --transport stdio --scope project neuromorph-nano -- \"D:\\Nano Home") && add.ends_with("--flow off"), "{add}");
    }

    #[test]
    fn mcp_clients_diff_lines() {
        assert_eq!(diff(None, "a\nb\n"), "+ a\n+ b\n");
        assert_eq!(diff(Some("a\nb\n"), "a\nb\n"), "(no change)\n");
        let d = diff(Some("1\n2\n3\n4\n5\n6\n7\n"), "1\n2\nX\n4\n5\n6\n7\nY\n");
        assert_eq!(d, "  2\n- 3\n+ X\n  4\n...\n  7\n+ Y\n");
    }
}
