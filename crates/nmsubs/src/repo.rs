//! Repository graph (guide 6.2, 17.1): files, symbols with their defining line, crates and
//! their dependencies, file history. Our own item scanner over the `nmlex` tokens (comment
//! and string aware); crates from `cargo metadata`, history from `git log`, both run on the
//! repository under study (input, not a dependency). Body first written in the earlier
//! `nmcore::substrate` pass, rebuilt over nmtype.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Folders never read in any repository: build output, installed packages and caches (dot
/// folders, version control among them, are skipped too). The list is the format table's.
pub const SKIP: &[&str] = nmdoc::format::SKIP_DIRS;
/// Folders skipped only when the studied root is a Neuromorph folder itself (its weights, test
/// memories, disposable work, runtime state and retired code are never studied). Another
/// repository's `tests`, `scratch` or `home` folders are its own code and are read (Neuromorph
/// Nano, 2026-11-15, build plan E1: serve any repository).
pub const SKIP_OWN: &[&str] = &["9-weight", "golden", "golden-data", "home", "scratch", "tests", "private", "retired"];
/// Within a Neuromorph folder's `memory/` (runtime state: the audit log, episodes, project
/// records), only Markdown is read, as before the format table.
const OWN_MEMORY: &str = "memory";

/// True when `root` is a Neuromorph folder (the Nano or the Mini): it holds `rules.md` and
/// `crates/nmcore/Cargo.toml`, the marks `nmtype::nano_root` finds it by.
#[must_use]
pub fn is_own(root: &Path) -> bool {
    root.join("rules.md").is_file() && root.join("crates/nmcore/Cargo.toml").is_file()
}

/// One file read: its path, its derived text (for a text file the file itself), the SHA-256 of
/// that text, the decoder that read it and its blocks (the document model; empty for code until
/// the substrate maps its items in).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileInfo {
    pub path: String,
    pub text: String,
    pub hash: String,
    /// `rust`, `toml`, a language of the language table (`typescript`, `python`, `c`, ...),
    /// `markdown`, `text`, `csv`, `tsv`, `html`, `xml`, `json`, `jsonl`, `notebook`, `yaml`, `log`.
    pub decoder: &'static str,
    pub blocks: Vec<nmtype::doc::Block>,
    /// How the file was read when no specific reader fits it (`nmtype::doc::Doc::general`):
    /// `None`, `container`, `member` (a path `container!member`), `gzip` or `raw`.
    pub general: Option<&'static str>,
}

impl FileInfo {
    /// A file from text already in memory, read by the decoder its name gives (a skipped name is
    /// read as plain text).
    #[must_use]
    pub fn new(path: &str, text: &str) -> Self {
        let name = path.rsplit('/').next().unwrap_or(path);
        let decoder = match nmdoc::format::by_name(name, 0) {
            nmdoc::Class::Code(d) => d,
            nmdoc::Class::Doc("text?" | nmdoc::general::GENERAL) | nmdoc::Class::Skip(_) => "text",
            nmdoc::Class::Doc(d) => d,
        };
        let doc = nmdoc::from_text(path, decoder, "utf-8", text.to_owned());
        Self { path: path.to_owned(), hash: nmhash::sha256_bytes(text.as_bytes()), text: doc.text, decoder, blocks: doc.blocks, general: None }
    }
    /// True for source code (Rust, TOML and the languages of the language table); false for
    /// documents and data.
    #[must_use]
    pub fn is_code(&self) -> bool {
        matches!(self.decoder, "rust" | "toml" | "code") || nmlex::lang::by_id(self.decoder).is_some()
    }
}

/// One file not read and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skipped {
    pub path: String,
    pub reason: &'static str,
}
/// One definition: its name and kind, the file and the lines it spans, the enclosing class or
/// namespace (empty at top level), and the trimmed text of its first line. A Rust item spans its
/// first line only (`end == line`) and has no parent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Symbol {
    pub name: String,
    pub kind: String,
    pub file: String,
    pub line: usize,
    pub end: usize,
    pub parent: String,
    pub text: String,
}
/// One package: a Cargo crate, an npm package (`package.json`) or a Python project
/// (`pyproject.toml`), with its dependencies on the other packages of the tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Crate {
    pub name: String,
    pub dir: String,
    pub deps: Vec<String>,
    /// The manifest file, root-relative.
    pub manifest: String,
    /// `cargo`, `npm` or `python`.
    pub ecosystem: &'static str,
}

/// Canonical path without the Windows verbatim prefix.
#[must_use]
pub fn canon(p: &Path) -> PathBuf {
    p.canonicalize().map_or_else(|_| p.to_path_buf(), |d| PathBuf::from(d.to_string_lossy().trim_start_matches(r"\\?\")))
}

/// Run a program in a folder: (success, stdout).
pub(crate) fn run(dir: &Path, prog: &str, args: &[&str]) -> (bool, String) {
    let mut cmd = Command::new(prog);
    cmd.args(args).current_dir(dir);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt as _;
        // No console window for a program started without one.
        cmd.creation_flags(0x0800_0000);
    }
    match cmd.output() {
        Ok(o) => (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned()),
        Err(_) => (false, String::new()),
    }
}

/// Every file the format table reads, sorted, skipping `SKIP`, dot folders and, when `root` is a
/// Neuromorph folder, `SKIP_OWN`. In any other root, files its `.gitignore` rules leave out are
/// skipped too (`Ignore`). Files not read are dropped here; `walk_counted` keeps them with their
/// reasons.
pub fn walk(root: &Path, dir: &Path, skip: &[String], out: &mut Vec<FileInfo>) {
    let mut skipped = Vec::new();
    walk_counted(root, dir, skip, out, &mut skipped);
}
/// `walk`, also listing every file not read with its reason.
pub fn walk_counted(root: &Path, dir: &Path, skip: &[String], out: &mut Vec<FileInfo>, skipped: &mut Vec<Skipped>) {
    let own = is_own(root);
    let ignore = if own { Ignore::default() } else { Ignore::of(root) };
    let mut entries = Vec::new();
    collect_walk(root, dir, &Rules { skip, own, ignore: &ignore }, &mut entries);
    let decoded = nmtype::budget::run(entries.len(), |i| match entries.get(i) {
        Some(WalkEntry::Skip(s)) => (Vec::new(), vec![s.clone()]),
        Some(WalkEntry::Read { abs, rel, class }) => {
            let Ok(bytes) = std::fs::read(abs) else {
                return (Vec::new(), vec![Skipped { path: rel.clone(), reason: "unreadable" }]);
            };
            let mut files = Vec::new();
            let mut skips = Vec::new();
            for item in nmdoc::read(rel, *class, &bytes) {
                match item {
                    Ok(doc) => files.push(FileInfo::of(doc)),
                    Err((path, reason)) => skips.push(Skipped { path, reason }),
                }
            }
            (files, skips)
        }
        None => (Vec::new(), Vec::new()),
    });
    for (files, skips) in decoded.into_iter().flatten() {
        out.extend(files);
        skipped.extend(skips);
    }
}

/// Fast metadata-only fingerprint of every walked entry under `root` (path, size, mtime) plus
/// `.git/HEAD` and `config_tag` (zero file reads). Used by `crate::cache` to validate the binary
/// substrate cache in milliseconds.
#[must_use]
pub(crate) fn manifest_fingerprint(root: &Path, skip: &[String], config_tag: &str) -> String {
    use std::fmt::Write as _;
    let own = is_own(root);
    let ignore = if own { Ignore::default() } else { Ignore::of(root) };
    let mut entries = Vec::new();
    collect_walk(root, root, &Rules { skip, own, ignore: &ignore }, &mut entries);
    let mut buf = String::with_capacity(entries.len().saturating_mul(64) + 256);
    let _ = writeln!(buf, "cfg:{config_tag}|own:{own}|skip:{}", skip.join(","));
    if let Ok(m) = std::fs::metadata(root.join(".git").join("HEAD")) {
        let (secs, nanos) = m
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or((0, 0), |d| (d.as_secs(), d.subsec_nanos()));
        let _ = writeln!(buf, "git:{}:{secs}:{nanos}", m.len());
    }
    for e in &entries {
        match e {
            WalkEntry::Skip(s) => {
                let _ = writeln!(buf, "S:{}:{}", s.path, s.reason);
            }
            WalkEntry::Read { abs, rel, .. } => {
                let (len, secs, nanos) = std::fs::metadata(abs).map_or((0, 0, 0), |m| {
                    let (s, n) = m
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map_or((0, 0), |d| (d.as_secs(), d.subsec_nanos()));
                    (m.len(), s, n)
                });
                let _ = writeln!(buf, "R:{rel}:{len}:{secs}:{nanos}");
            }
        }
    }
    nmhash::sha256_bytes(buf.as_bytes())
}

enum WalkEntry {
    Skip(Skipped),
    Read { abs: PathBuf, rel: String, class: nmdoc::Class },
}

/// What one walk leaves out: extra folder names, whether the root is a Neuromorph folder, and the
/// root's ignore rules.
struct Rules<'a> {
    skip: &'a [String],
    own: bool,
    ignore: &'a Ignore,
}

/// Why a file is skipped when the ignore rules leave it out.
pub const IGNORED: &str = "ignored by .gitignore";

/// The ignore rules of a studied root that is not a Neuromorph folder. When the root is the top of
/// a git work tree, git itself lists the ignored paths (`git ls-files --others --ignored
/// --exclude-standard --directory`), so every `.gitignore` level and the user's excludes count.
/// Otherwise the root's own `.gitignore` is read by a small matcher: plain names, `*`, `?` and
/// `**`, a leading `/` or an inner `/` anchoring a rule to the root, a trailing `/` for folders,
/// and `!` to take a path back; the last matching rule wins.
#[derive(Clone, Debug, Default)]
pub struct Ignore {
    /// Paths listed by git (folders end with `/`).
    git: Option<std::collections::BTreeSet<String>>,
    /// Rules of the root `.gitignore`: (pattern, negated, folders only, anchored).
    rules: Vec<(String, bool, bool, bool)>,
}

impl Ignore {
    /// The rules for `root`.
    #[must_use]
    pub fn of(root: &Path) -> Self {
        let (ok, top) = run(root, "git", &["-c", "safe.directory=*", "rev-parse", "--show-toplevel"]);
        if ok && canon(Path::new(top.trim())) == canon(root) {
            let (ok, list) = run(root, "git", &["-c", "safe.directory=*", "ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--directory"]);
            if ok {
                return Self { git: Some(list.split('\0').filter(|x| !x.is_empty()).map(str::to_owned).collect()), rules: Vec::new() };
            }
        }
        let text = std::fs::read_to_string(root.join(".gitignore")).unwrap_or_default();
        Self { git: None, rules: Self::parse(&text) }
    }
    fn parse(text: &str) -> Vec<(String, bool, bool, bool)> {
        text.lines()
            .map(str::trim_end)
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
            .map(|l| {
                let (neg, l) = l.strip_prefix('!').map_or((false, l), |r| (true, r));
                let (dir, l) = l.strip_suffix('/').map_or((false, l), |r| (true, r));
                let anchored = l.starts_with('/') || l.contains('/');
                (l.trim_start_matches('/').to_owned(), neg, dir, anchored)
            })
            .collect()
    }
    /// True when the root-relative `path` is left out (`dir`: it is a folder).
    #[must_use]
    pub fn ignored(&self, path: &str, dir: bool) -> bool {
        if let Some(set) = &self.git {
            return set.contains(path) || (dir && set.contains(&format!("{path}/")));
        }
        let name = path.rsplit('/').next().unwrap_or(path);
        let mut out = false;
        for (pat, neg, only_dir, anchored) in &self.rules {
            if *only_dir && !dir {
                continue;
            }
            let hit = if *anchored { glob(pat, path) } else { glob(pat, name) };
            if hit {
                out = !neg;
            }
        }
        out
    }
}

/// Glob match: `*` any run without `/`, `**` any run, `?` one character other than `/`.
fn glob(pat: &str, text: &str) -> bool {
    let (p, t): (Vec<char>, Vec<char>) = (pat.chars().collect(), text.chars().collect());
    fn go(p: &[char], t: &[char]) -> bool {
        match p.first() {
            None => t.is_empty(),
            Some('*') if p.get(1) == Some(&'*') => {
                let rest = p.get(2..).unwrap_or(&[]);
                let rest = rest.strip_prefix(&['/']).unwrap_or(rest);
                (0..=t.len()).any(|k| go(rest, t.get(k..).unwrap_or(&[])))
            }
            Some('*') => {
                let rest = p.get(1..).unwrap_or(&[]);
                for k in 0..=t.len() {
                    if go(rest, t.get(k..).unwrap_or(&[])) {
                        return true;
                    }
                    if t.get(k) == Some(&'/') {
                        break;
                    }
                }
                false
            }
            Some('?') => t.first().is_some_and(|c| *c != '/') && go(p.get(1..).unwrap_or(&[]), t.get(1..).unwrap_or(&[])),
            Some(c) => t.first() == Some(c) && go(p.get(1..).unwrap_or(&[]), t.get(1..).unwrap_or(&[])),
        }
    }
    go(&p, &t)
}

/// True when a hidden folder (its name starts with a dot) is the project's own configuration,
/// decided by what it holds, not by its name: it holds no folder, every file in it is one the
/// format table reads (no hidden file, no file skipped, nothing read only as binary or a
/// container), and at least one is a configuration format the readers know (TOML, JSON, YAML,
/// XML). A tool's store is not: version control's and a virtual environment's hold folders, a
/// cache holds folders or marks itself with a hidden ignore file, and the format table skips
/// their data. A hidden folder the root's ignore rules leave out stays out (`walk_with`). Files
/// read from such a folder are kept out of the material's rarities (`nmtype::in_hidden_folder`).
fn own_configuration(dir: &Path) -> bool {
    let Ok(rd) = std::fs::read_dir(dir) else { return false };
    let mut config = false;
    for e in rd.flatten() {
        if e.path().is_dir() {
            return false;
        }
        let name = e.file_name().to_string_lossy().into_owned();
        let size = e.metadata().map_or(u64::MAX, |m| m.len());
        match nmdoc::format::by_name(&name, size) {
            nmdoc::Class::Skip(_) => return false,
            nmdoc::Class::Doc(d) if d == nmdoc::general::GENERAL => return false,
            nmdoc::Class::Code("toml") | nmdoc::Class::Doc("json" | "yaml" | "xml") => config = true,
            _ => {}
        }
    }
    config
}

/// Whether the walk reads a project's own hidden configuration (`own_configuration`): only with
/// `NANO_HIDDEN=on` (2026-11-17: built and measured, not kept on; build plan "Request frame parts
/// and hidden configuration").
#[must_use]
pub fn hidden_on() -> bool {
    std::env::var("NANO_HIDDEN").is_ok_and(|v| v == "on")
}

/// Roadmap Phase 4.6 (2026-11-25): why a linked folder (a symbolic link or a Windows junction) is
/// left out: following one can loop back into the tree (a link to a parent folder) or reach a
/// tree outside the project, so the walk never follows it.
pub const LINKED_FOLDER: &str = "linked folder (symbolic link or junction): not followed";
/// Roadmap Phase 4.6 (2026-11-25): a link whose target is gone.
pub const BROKEN_LINK: &str = "broken link: not read";

fn collect_walk(root: &Path, dir: &Path, rules: &Rules<'_>, out: &mut Vec<WalkEntry>) {
    let (skip, own, ignore) = (rules.skip, rules.own, rules.ignore);
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = rd.flatten().collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for e in entries {
        let p = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        let path = p.strip_prefix(root).unwrap_or(&p).to_string_lossy().replace('\\', "/");
        // Roadmap Phase 4.6 (2026-11-25): the entry's own type, not its target's, so a link is seen
        // as a link. A linked folder is never followed (a loop or a walk out of the project); a
        // linked file is read by its target's size, so a link to a huge file cannot pass the size
        // limits as a tiny one.
        let link = e.file_type().is_ok_and(|t| t.is_symlink());
        if link {
            match std::fs::metadata(&p) {
                Ok(m) if m.is_dir() => out.push(WalkEntry::Skip(Skipped { path, reason: LINKED_FOLDER })),
                Ok(m) => {
                    let class = nmdoc::format::by_name(&name, m.len());
                    match class {
                        nmdoc::Class::Skip(reason) => out.push(WalkEntry::Skip(Skipped { path, reason })),
                        _ if ignore.ignored(&path, false) => out.push(WalkEntry::Skip(Skipped { path, reason: IGNORED })),
                        _ => out.push(WalkEntry::Read { abs: p, rel: path, class }),
                    }
                }
                Err(_) => out.push(WalkEntry::Skip(Skipped { path, reason: BROKEN_LINK })),
            }
            continue;
        }
        if p.is_dir() {
            if (!name.starts_with('.') || (hidden_on() && own_configuration(&p)))
                && !SKIP.contains(&name.as_str())
                && !(own && SKIP_OWN.contains(&name.as_str()))
                && !skip.contains(&name)
                && !ignore.ignored(&path, true)
            {
                collect_walk(root, &p, rules, out);
            }
            continue;
        }
        let size = e.metadata().map_or(u64::MAX, |m| m.len());
        let mut class = nmdoc::format::by_name(&name, size);
        if own && path.starts_with(&format!("{OWN_MEMORY}/")) && !matches!(class, nmdoc::Class::Doc("markdown")) {
            class = nmdoc::Class::Skip("Neuromorph runtime state: not read");
        }
        if !matches!(class, nmdoc::Class::Skip(_)) && ignore.ignored(&path, false) {
            class = nmdoc::Class::Skip(IGNORED);
        }
        if let nmdoc::Class::Skip(reason) = class {
            out.push(WalkEntry::Skip(Skipped { path, reason }));
            continue;
        }
        out.push(WalkEntry::Read { abs: p, rel: path, class });
    }
}

impl FileInfo {
    /// A file from a document the reader made.
    #[must_use]
    pub fn of(doc: nmtype::doc::Doc) -> Self {
        let hash = nmhash::sha256_bytes(doc.text.as_bytes());
        Self { path: doc.path, text: doc.text, hash, decoder: doc.decoder, blocks: doc.blocks, general: doc.general }
    }
}

/// The text of one read path as it is now: the file on disk when it is plain UTF-8 text, else
/// what the reader derives from it (a container member `file!member`, a file read raw). `None`
/// when the file or member is gone.
#[must_use]
pub fn fresh_text(root: &Path, path: &str) -> Option<String> {
    let disk = path.split('!').next().unwrap_or(path);
    // Roadmap Phase 4.6 (2026-11-25): a file that has grown past the largest size the walk reads
    // is not read whole into memory here either.
    if std::fs::metadata(root.join(disk)).ok()?.len() >= nmdoc::format::MAX_GENERAL {
        return None;
    }
    if disk.len() == path.len()
        && let Ok(t) = std::fs::read_to_string(root.join(path))
    {
        return Some(t);
    }
    let bytes = std::fs::read(root.join(disk)).ok()?;
    let name = disk.rsplit('/').next().unwrap_or(disk);
    let class = nmdoc::format::by_name(name, bytes.len() as u64);
    nmdoc::read(disk, class, &bytes).into_iter().flatten().find(|d| d.path == path).map(|d| d.text)
}

/// Skip counts by reason, largest first.
#[must_use]
pub fn skip_counts(skipped: &[Skipped]) -> Vec<(&'static str, usize)> {
    let mut m: BTreeMap<&'static str, usize> = BTreeMap::new();
    for s in skipped {
        *m.entry(s.reason).or_default() += 1;
    }
    let mut v: Vec<(&'static str, usize)> = m.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    v
}

/// Items defined in one Rust file: fn, struct, enum, trait, mod, type, const, static, macro.
#[must_use]
pub fn parse_items(file: &str, text: &str) -> Vec<Symbol> {
    parse_items_with(file, text, &nmlex::tokens(text, true)).0
}

/// An `impl` or `trait` block: first line, last line and the type it implements for (or the
/// trait's name).
pub type ImplBlock = (usize, usize, String);

/// `parse_items` over tokens already read (`nmlex::tokens(text, true)`), also returning the
/// `impl` and `trait` blocks the items sit in.
#[must_use]
pub fn parse_items_with(file: &str, text: &str, toks: &[(String, usize)]) -> (Vec<Symbol>, Vec<ImplBlock>) {
    const KW: [&str; 9] = ["fn", "struct", "enum", "trait", "mod", "type", "const", "static", "macro_rules"];
    let lines: Vec<&str> = text.lines().collect();
    // The line of each opening brace's match (comments and strings are not tokens).
    let mut close = vec![0_usize; toks.len()];
    let mut open: Vec<usize> = Vec::new();
    for (i, (t, line)) in toks.iter().enumerate() {
        match t.as_str() {
            "{" => open.push(i),
            "}" => {
                if let Some(o) = open.pop()
                    && let Some(c) = close.get_mut(o)
                {
                    *c = *line;
                }
            }
            _ => {}
        }
    }
    // An item ends at its body's closing brace, or at the `;` that ends it first.
    let end_of = |i: usize, line: usize| -> usize {
        for (j, (t, l)) in toks.iter().enumerate().skip(i + 1) {
            match t.as_str() {
                ";" => return *l,
                "{" => return close.get(j).copied().filter(|c| *c >= line).unwrap_or(line),
                _ => {}
            }
        }
        line
    };
    // The blocks items can sit in: `impl [<..>] [Trait for] Type {` (the type it implements for)
    // and `trait Name {`, each with its first and last line. Items inside are its members.
    let mut blocks: Vec<(usize, usize, String)> = Vec::new();
    for (i, (_, line)) in toks.iter().enumerate().filter(|(_, (t, _))| t == "impl" || t == "trait") {
        let Some(open_at) = toks.iter().skip(i + 1).position(|(x, _)| x == "{" || x == ";").map(|p| p + i + 1) else { continue };
        if toks.get(open_at).is_none_or(|(x, _)| x != "{") {
            continue;
        }
        let head = toks.get(i + 1..open_at).unwrap_or_default();
        let after_for = head.iter().position(|(x, _)| x == "for").map_or(0, |p| p + 1);
        // The first name at angle depth 0 (after `for` when there is one), past `where`.
        let mut depth = 0_i32;
        let mut name = None;
        for (x, _) in head.iter().skip(after_for) {
            match x.as_str() {
                "<" => depth += 1,
                ">" => depth -= 1,
                "where" => break,
                _ if depth == 0 && x.chars().next().is_some_and(|c| c.is_alphabetic() || c == '_') && x != "dyn" && x != "unsafe" => {
                    name = Some(x.clone());
                    break;
                }
                _ => {}
            }
        }
        if let (Some(n), Some(end)) = (name, close.get(open_at).copied().filter(|c| *c >= *line)) {
            blocks.push((*line, end, n));
        }
    }
    let parent_of = |line: usize| -> String {
        blocks.iter().filter(|(a, b, _)| line > *a && line <= *b).min_by_key(|(a, b, _)| b - a).map(|x| x.2.clone()).unwrap_or_default()
    };
    let mut out = Vec::new();
    for (i, (t, line)) in toks.iter().enumerate() {
        if !KW.contains(&t.as_str()) {
            continue;
        }
        let at = if t == "macro_rules" { i + 2 } else { i + 1 };
        if let Some((n, _)) = toks.get(at)
            && n.chars().next().is_some_and(|c| c.is_alphabetic() || c == '_')
            && !(t == "const" && (n == "fn" || n == "_"))
        {
            let kind = if t == "macro_rules" { "macro" } else { t.as_str() };
            let text = lines.get(line.saturating_sub(1)).map_or("", |l| l.trim()).to_owned();
            let end = if kind == "macro" { *line } else { end_of(at, *line) };
            out.push(Symbol { name: n.clone(), kind: kind.to_owned(), file: file.to_owned(), line: *line, end, parent: parent_of(*line), text });
        }
    }
    (out, blocks)
}

/// Workspace crates from `cargo metadata --no-deps --offline` (normal, dev and build deps
/// between workspace members). Empty when the root has no manifest.
#[must_use]
pub fn crates_of(root: &Path) -> Vec<Crate> {
    let manifest = root.join("Cargo.toml");
    if !manifest.is_file() {
        return Vec::new();
    }
    let m = manifest.to_string_lossy().into_owned();
    let (_, out) = run(root, "cargo", &["metadata", "--format-version", "1", "--no-deps", "--offline", "--manifest-path", &m]);
    let Ok(v) = nmjson::parse(out.as_bytes()) else { return Vec::new() };
    let Ok(pkgs) = v.get("packages").and_then(nmjson::Value::array) else { return Vec::new() };
    let field = |p: &nmjson::Value, k: &str| p.get(k).and_then(nmjson::Value::string).ok().map(str::to_owned);
    let names: Vec<String> = pkgs.iter().filter_map(|p| field(p, "name")).collect();
    pkgs.iter()
        .filter_map(|p| {
            let name = field(p, "name")?;
            let mp = field(p, "manifest_path")?;
            let dir = canon(Path::new(&mp).parent()?).strip_prefix(root).ok()?.to_string_lossy().replace('\\', "/");
            let mut deps: Vec<String> = p
                .get("dependencies")
                .and_then(nmjson::Value::array)
                .ok()?
                .iter()
                // Every kind of dependency (normal, dev, build): a test-only edge is still a code
                // relation the crates have (2026-11-13; live gate C2 counts all three).
                .filter_map(|d| field(d, "name"))
                .filter(|d| names.contains(d) && *d != name)
                .collect();
            deps.sort();
            deps.dedup();
            let manifest = format!("{dir}/Cargo.toml");
            Some(Crate { name, dir, deps, manifest, ecosystem: "cargo" })
        })
        .collect()
}

/// File to commits (`sha date subject`), newest first, from the last 300 commits; a file
/// changed since the last commit, or never committed, gets a first entry saying so (2026-11-09:
/// before, work not yet committed had no history at all).
#[must_use]
pub fn history_of(root: &Path) -> BTreeMap<String, Vec<String>> {
    let (_, out) = run(
        root,
        "git",
        &["-c", "safe.directory=*", "log", "-n", "300", "--relative", "--date=short", "--format=@%h %ad %s", "--name-only", "--", "."],
    );
    let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut head = String::new();
    for line in out.lines() {
        if let Some(h) = line.strip_prefix('@') {
            h.clone_into(&mut head);
        } else if !line.trim().is_empty() {
            map.entry(line.trim().to_owned()).or_default().push(head.clone());
        }
    }
    let (_, status) = run(root, "git", &["-c", "safe.directory=*", "status", "--porcelain", "--untracked-files=all", "--", "."]);
    for line in status.lines() {
        let (code, path) = (line.get(..2).unwrap_or(""), line.get(3..).unwrap_or("").trim().trim_matches('"'));
        if path.is_empty() {
            continue;
        }
        let path = path.rsplit(" -> ").next().unwrap_or(path).to_owned();
        let note = if code == "??" {
            "working tree: new file, never committed"
        } else {
            "working tree: changed since the last commit, not committed"
        };
        map.entry(path).or_default().insert(0, format!("- {note}"));
    }
    map
}

/// Self-check of the `repo` sub-component: the scanner finds the item kinds on a built-in
/// fixture and ignores names inside comments and strings.
pub(crate) fn check() -> Result<(), String> {
    let src = "// fn hidden() {}\npub struct A;\nenum B { X }\ntrait C {}\nmod d {}\ntype E = u8;\nconst F: u8 = 1;\nstatic G: u8 = 2;\nmacro_rules! h { () => {} }\nfn i() { let s = \"fn nope\"; }\nimpl A {}\n";
    let got: Vec<(String, String)> = parse_items("x.rs", src).into_iter().map(|s| (s.kind, s.name)).collect();
    let want = [
        ("struct", "A"),
        ("enum", "B"),
        ("trait", "C"),
        ("mod", "d"),
        ("type", "E"),
        ("const", "F"),
        ("static", "G"),
        ("macro", "h"),
        ("fn", "i"),
    ];
    let want: Vec<(String, String)> = want.iter().map(|(k, n)| ((*k).to_owned(), (*n).to_owned())).collect();
    if got != want {
        return Err(format!("nmsubs.repo: scanner found {got:?}"));
    }
    // Members of an `impl` or `trait` block name the type they belong to, with their own lines.
    let src = "struct L { a: u8 }\nimpl Default for L {\n    fn default() -> Self {\n        Self { a: 1 }\n    }\n}\nimpl<T: Copy> std::fmt::Debug for W<T> {\n    fn fmt(&self) {}\n}\nfn free() {}\n";
    let got: Vec<(String, String, usize, usize)> = parse_items("x.rs", src).into_iter().map(|s| (s.parent, s.name, s.line, s.end)).collect();
    let want = [("", "L", 1, 1), ("L", "default", 3, 5), ("W", "fmt", 8, 8), ("", "free", 10, 10)];
    if got.iter().map(|(p, n, a, b)| (p.as_str(), n.as_str(), *a, *b)).collect::<Vec<_>>() != want {
        return Err(format!("nmsubs.repo: members {got:?}"));
    }
    // 2026-11-15 (E1): another repository's `tests` folder is read and `target` is not; a
    // Neuromorph folder's own `tests` stays out.
    let dir = nmtype::check_dir("nmsubs-repo-walk").map_err(|e| format!("nmsubs.repo: {e}"))?;
    let (plain, own) = (dir.join("plain"), dir.join("own"));
    let files = [
        (&plain, "tests/a.rs"),
        (&plain, "target/b.rs"),
        (&plain, "src/c.rs"),
        (&own, "tests/a.rs"),
        (&own, "src/c.rs"),
        (&own, "rules.md"),
        (&own, "crates/nmcore/Cargo.toml"),
    ];
    for (root, f) in files {
        let p = root.join(f);
        p.parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| {
                std::fs::write(
                    &p,
                    "fn x() {}
",
                )
            })
            .map_err(|e| format!("nmsubs.repo: {e}"))?;
    }
    let read = |root: &Path| {
        let mut out = Vec::new();
        walk(root, root, &[], &mut out);
        out.into_iter().map(|f| f.path).collect::<Vec<_>>()
    };
    let (p, o) = (read(&plain), read(&own));
    if p != ["src/c.rs", "tests/a.rs"] || o.iter().any(|f| f.starts_with("tests/")) || !o.contains(&"src/c.rs".to_owned()) {
        return Err(format!("nmsubs.repo: walk read {p:?} in a plain repository and {o:?} in a Neuromorph folder"));
    }
    // 2026-11-17 (T-009): a hidden folder of the project's own configuration is read; version
    // control's store (it holds folders) and a cache that marks itself with a hidden ignore file
    // are not.
    let hid = nmtype::check_dir("nmsubs-repo-hidden").map_err(|e| format!("nmsubs.repo: {e}"))?;
    for (f, text) in [
        (".cargo/config.toml", "[build]\nincremental = false\n"),
        (".cargo/README.md", "# Cargo settings\n"),
        (".git/HEAD", "ref: refs/heads/master\n"),
        (".git/refs/heads/master", "0000\n"),
        (".tool_cache/.gitignore", "*\n"),
        (".tool_cache/state.json", "{}\n"),
        ("src/c.rs", "fn x() {}\n"),
    ] {
        let p = hid.join(f);
        p.parent().map_or(Ok(()), std::fs::create_dir_all).and_then(|()| std::fs::write(&p, text)).map_err(|e| format!("nmsubs.repo: {e}"))?;
    }
    let rule = [".cargo", ".git", ".tool_cache"].map(|d| own_configuration(&hid.join(d)));
    let h = read(&hid);
    let want: &[&str] = if hidden_on() { &[".cargo/README.md", ".cargo/config.toml", "src/c.rs"] } else { &["src/c.rs"] };
    if rule != [true, false, false] || h != want {
        return Err(format!("nmsubs.repo: hidden folders judged {rule:?}, read as {h:?}"));
    }
    Ok(())
}

/// Roadmap Phase 4.6 (2026-11-25): a synthetic project of awkward material for the resilience
/// tests: non-UTF-8 and UTF-16 text, binary files under text names, archives (a truncated zip, a
/// gzip that inflates far past the member limit, a broken tar), ignored build folders, empty
/// files, files over the size limits, a path over 260 characters, and links (a folder link back
/// to the root, a file link, a broken link) where this machine lets them be made. Kept under
/// `scratch/unit-tests/` of the checkout. Returns the root and whether links could be made.
#[cfg(test)]
pub(crate) fn resilience_fixture(name: &str) -> std::io::Result<(PathBuf, bool)> {
    use std::io::Write as _;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("scratch").join("unit-tests").join(name);
    if root.exists() {
        std::fs::remove_dir_all(&root)?;
    }
    std::fs::create_dir_all(&root)?;
    let root = canon(&root);
    let put = |rel: &str, bytes: &[u8]| -> std::io::Result<()> {
        let p = root.join(rel);
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(p, bytes)
    };
    put(".gitignore", b"target/\nnode_modules/\ndist/\n.venv/\n__pycache__/\n*.tmp\n")?;
    put("src/lib.rs", b"pub fn plain() -> u32 { 1 }\n")?;
    put("src/latin1.rs", b"// caf\xe9 au lait \xff\xfe\nfn caf\xe9() {}\n")?;
    put("docs/utf16.txt", &[0xFF, 0xFE, b'h', 0, b'i', 0, b'\n', 0])?;
    put("docs/broken-utf8.md", b"# Title\n\xc3\x28 invalid \xa0\xa1 sequence\n")?;
    put("data/blob.dat", &(0..4096_u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect::<Vec<u8>>())?;
    put("docs/nul.txt", b"text\0with\0zero\0bytes\0\0\0\0\0\0\0\0\0\0\0\0")?;
    put("assets/fake.png", b"\x89PNG\r\n\x1a\nnot really an image")?;
    put("src/empty.rs", b"")?;
    put("docs/empty.txt", b"")?;
    put("archives/truncated.zip", b"PK\x03\x04\x14\x00\x00\x00\x08\x00garbage")?;
    put("archives/broken.tar", &[b'x'; 700])?;
    put("archives/bomb.gz", &gzip_bomb(80 << 20))?;
    put("scratch.tmp", b"ignored by a pattern\n")?;
    for (d, f) in [("target", "debug/build.rs"), ("node_modules", "pkg/index.js"), ("dist", "bundle.js"), (".venv", "lib/site.py"), ("__pycache__", "mod.cpython-312.pyc"), ("sub/target", "x.rs")] {
        put(&format!("{d}/{f}"), b"fn should_not_be_read() {}\n")?;
    }
    // Over the text limit (512 KB) and over the general limit (64 MB, written sparse-free but
    // removed with the fixture).
    put("logs/big.log", &b"a log line that repeats\n".repeat(30_000))?;
    {
        let f = std::fs::File::create(root.join("data").join("huge.bin"))?;
        f.set_len(65 << 20)?;
    }
    let mut big = std::fs::File::create(root.join("data").join("large-zeros.bin"))?;
    big.write_all(&vec![0_u8; 12 << 20])?;
    // A path over 260 characters.
    let mut deep = root.join("deep");
    while deep.as_os_str().len() < 300 {
        deep = deep.join("a_rather_long_folder_name_for_the_windows_path_limit");
    }
    std::fs::create_dir_all(&deep)?;
    std::fs::write(deep.join("leaf.rs"), b"pub fn deep_leaf() {}\n")?;
    let links = make_links(&root);
    Ok((root, links))
}

/// A gzip member whose deflate stream (fixed Huffman codes: one zero byte, then copies of 258
/// bytes at distance 1) inflates to about `out` bytes from a few hundred kilobytes.
#[cfg(test)]
fn gzip_bomb(out: usize) -> Vec<u8> {
    struct Bits {
        v: Vec<u8>,
        acc: u32,
        n: u32,
    }
    impl Bits {
        fn put(&mut self, value: u32, len: u32) {
            self.acc |= value << self.n;
            self.n += len;
            while self.n >= 8 {
                self.v.push((self.acc & 0xFF) as u8);
                self.acc >>= 8;
                self.n -= 8;
            }
        }
        /// A Huffman code, written most significant bit first.
        fn code(&mut self, code: u32, len: u32) {
            let rev = (0..len).fold(0, |r, i| r | (((code >> i) & 1) << (len - 1 - i)));
            self.put(rev, len);
        }
    }
    let mut b = Bits { v: vec![0x1F, 0x8B, 8, 0, 0, 0, 0, 0, 0, 0xFF], acc: 0, n: 0 };
    b.put(1, 1); // final block
    b.put(1, 2); // fixed Huffman codes
    b.code(0x30, 8); // literal 0
    for _ in 0..out / 258 {
        b.code(0xC5, 8); // length 258 (code 285)
        b.code(0, 5); // distance 1
    }
    b.code(0, 7); // end of block
    if b.n > 0 {
        b.put(0, 8 - b.n);
    }
    b.v.extend_from_slice(&[0, 0, 0, 0]);
    b.v.extend_from_slice(&u32::try_from(out % (1 << 32)).unwrap_or(0).to_le_bytes());
    b.v
}

/// Links in the fixture: a folder link back to the root (a loop if followed), a file link and a
/// broken link. Symbolic links need a privilege on Windows unless developer mode is on; a
/// junction (`mklink /J`) does not, so it stands in for the folder link. True when the folder
/// link exists.
#[cfg(test)]
fn make_links(root: &Path) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::{symlink_dir, symlink_file};
        let folder = symlink_dir(root, root.join("loop")).is_ok() || {
            let mut c = Command::new("cmd");
            c.args(["/C", "mklink", "/J"]).arg(root.join("loop")).arg(root);
            c.output().is_ok_and(|o| o.status.success())
        };
        let _ = symlink_file(root.join("src").join("lib.rs"), root.join("src").join("lib-link.rs"));
        let _ = symlink_file(root.join("gone.rs"), root.join("src").join("dangling.rs"));
        folder
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        let folder = symlink(root, root.join("loop")).is_ok();
        let _ = symlink(root.join("src").join("lib.rs"), root.join("src").join("lib-link.rs"));
        let _ = symlink(root.join("gone.rs"), root.join("src").join("dangling.rs"));
        folder
    }
}

#[cfg(test)]
mod resilience_tests {
    use super::*;

    /// Roadmap Phase 4.6 (2026-11-25): the walk reads awkward material without a crash, a loop
    /// or a runaway in memory, and says why it left each skipped file out.
    #[test]
    fn repo_resilience_on_awkward_material() {
        let (root, links) = resilience_fixture("nmsubs-repo-resilience").unwrap();
        let mut files = Vec::new();
        let mut skipped = Vec::new();
        let t = std::time::Instant::now();
        walk_counted(&root, &root, &[], &mut files, &mut skipped);
        let secs = t.elapsed().as_secs_f64();
        let read: Vec<&str> = files.iter().map(|f| f.path.as_str()).collect();
        let why = |p: &str| skipped.iter().find(|s| s.path == p).map(|s| s.reason);
        // Ordinary code and non-UTF-8 text are read; the Latin-1 bytes come out as text.
        assert!(read.contains(&"src/lib.rs"), "{read:?}");
        let latin = files.iter().find(|f| f.path == "src/latin1.rs").expect("latin1.rs read");
        assert!(latin.text.contains("café"), "{}", latin.text);
        assert!(read.contains(&"docs/utf16.txt") && read.contains(&"docs/broken-utf8.md"));
        // The path over 260 characters is read.
        assert!(read.iter().any(|p| p.ends_with("leaf.rs") && p.len() > 200), "{read:?}");
        // Ignored build folders are not walked (by the folder list or the .gitignore).
        for d in ["target/", "node_modules/", "dist/", ".venv/", "__pycache__/", "sub/target/"] {
            assert!(!read.iter().any(|p| p.starts_with(d)), "{d} was read: {read:?}");
        }
        assert_eq!(why("scratch.tmp"), Some(IGNORED));
        // Over the limits: skipped with the reason.
        assert_eq!(why("logs/big.log"), Some("too large: over 512 KB"));
        assert_eq!(why("data/huge.bin"), Some("too large: over 64 MB"));
        // Empty files neither crash nor vanish without a word.
        for p in ["src/empty.rs", "docs/empty.txt"] {
            assert!(read.contains(&p) || why(p).is_some(), "{p} neither read nor skipped");
        }
        // Binary under a text name, a fake image, broken archives and a gzip bomb: each read
        // generally or skipped, and the bomb stops at the member limit.
        for p in ["data/blob.dat", "docs/nul.txt", "assets/fake.png", "archives/truncated.zip", "archives/broken.tar", "archives/bomb.gz", "data/large-zeros.bin"] {
            assert!(read.iter().any(|r| r.starts_with(p)) || skipped.iter().any(|s| s.path.starts_with(p)), "{p} neither read nor skipped");
        }
        let total: usize = files.iter().map(|f| f.text.len()).sum();
        assert!(total < 24 << 20, "{total} bytes of text from the fixture");
        if links {
            assert_eq!(why("loop"), Some(LINKED_FOLDER), "{skipped:?}");
            assert!(!read.iter().any(|p| p.starts_with("loop/")));
        }
        if root.join("src").join("dangling.rs").symlink_metadata().is_ok() {
            assert_eq!(why("src/dangling.rs"), Some(BROKEN_LINK));
            assert!(read.contains(&"src/lib-link.rs"));
        }
        // The metadata fingerprint walks the same tree without following the loop.
        let a = manifest_fingerprint(&root, &[], "t");
        assert_eq!(a, manifest_fingerprint(&root, &[], "t"));
        // A file grown past the limit is not read whole by `fresh_text`.
        assert!(fresh_text(&root, "data/huge.bin").is_none());
        assert!(secs < 120.0, "walk took {secs:.1} s");
        let _ = std::fs::remove_dir_all(&root);
    }
}
