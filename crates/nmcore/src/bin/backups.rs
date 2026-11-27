//! The user's own repositories for backing up, storing and rolling back their Nano data.
//!
//! A repository is any git remote the user names: their own GitHub (or other git host) repository,
//! or a Hugging Face repository (which is a git repository too), or a local folder. For each one the
//! user chooses what it holds: memory, parameter maps, cartridges, learned units, settings. Nothing
//! is sent unless the user asks for a backup; the air gap refuses it. A token is sent only where the
//! user said so, and only to that repository's host.
//!
//! - Back up: the chosen parts are copied into this repository's local mirror (`backups/<id>/`),
//!   with a `MANIFEST.json` naming what is in it and what was left out, committed, and pushed.
//!   Rebuildable stores (dense vectors, meaning caches, live snapshots, temporary files) and files
//!   over 50 MiB (base cartridges: git hosts refuse them without large-file support, and they come
//!   back from the cartridge library) are left out and listed.
//! - Snapshots: every backup is one commit; the list is read from the repository.
//! - Preview: what restoring a snapshot would write (new files, changed files); nothing is written.
//! - Restore: those files are written back. Each file it changes is copied aside first
//!   (`backups/<id>-before-<time>/`); nothing local is deleted. Audit undoes a restore by putting the
//!   copies back and taking out the files the restore added.
//!
//! Git is run as the system's `git` program, as the rest of the Nano reads git history.
use nmjson::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The list of the user's repositories, at the Nano's root (user state, never shipped).
const FILE: &str = "backup-repos.json";
/// Files above this are left out of a backup and listed (git hosts refuse them without large-file
/// support).
const MAX_FILE: u64 = 50 << 20;
/// What a repository can hold, each with the paths under the Nano's folder it takes.
pub const PARTS: [(&str, &[&str]); 5] = [
    ("memory", &["memory"]),
    ("maps", &["library/parameter-maps"]),
    ("cartridges", &["library/cartridges"]),
    ("units", &["units"]),
    ("settings", &["settings.json", "cartridge-sources.json", "designs/models.json"]),
];
/// Folders under the parts that are rebuilt by the Nano and never backed up.
const REBUILDABLE: [&str; 5] = ["dense", "meaning", "live", "set-aside", "incoming"];
const AUDIT_KIND: &str = "backup-repo";

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}
fn obj(f: Vec<(&str, Value)>) -> Value {
    Value::Object(f.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn text(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default()
}

/// One of the user's repositories.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Repo {
    pub id: String,
    /// A git URL (`https://...`, `git@host:owner/name.git`) or a local folder.
    pub url: String,
    /// The parts it holds (names from `PARTS`).
    pub parts: Vec<String>,
    /// Whether the user's token for the host is sent (Hugging Face: `HF_TOKEN` or the hf login;
    /// other hosts: `GITHUB_TOKEN` or `GH_TOKEN`); without it, git's own credential helper answers.
    pub token: bool,
    pub added_ms: u64,
}

impl Repo {
    fn to_value(&self) -> Value {
        obj(vec![("id", s(&self.id)), ("url", s(&self.url)), ("parts", Value::Array(self.parts.iter().map(|p| s(p)).collect())), ("token", Value::Bool(self.token)), ("added_ms", n(self.added_ms))])
    }
    fn from_value(v: &Value) -> Option<Self> {
        let id = text(v, "id");
        let url = text(v, "url");
        (!id.is_empty() && !url.is_empty()).then(|| Self {
            id,
            url,
            parts: v.get("parts").and_then(Value::array).unwrap_or_default().iter().filter_map(|p| p.string().ok().map(str::to_owned)).collect(),
            token: matches!(v.get("token"), Ok(Value::Bool(true))),
            added_ms: v.get("added_ms").and_then(Value::integer).unwrap_or(0),
        })
    }
}

/// The repositories the user added.
#[must_use]
pub fn list(home: &Path) -> Vec<Repo> {
    std::fs::read(home.join(FILE))
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .and_then(|v| v.get("repos").and_then(Value::array).ok().map(|a| a.iter().filter_map(Repo::from_value).collect()))
        .unwrap_or_default()
}

fn write(home: &Path, repos: &[Repo]) -> Result<(), String> {
    let v = obj(vec![
        ("schema", s("nm-backup-repos/1")),
        ("note", s("The user's own repositories for backing up and rolling back their Nano data. Written by the Neuromorph Nano.")),
        ("repos", Value::Array(repos.iter().map(Repo::to_value).collect())),
    ]);
    std::fs::write(home.join(FILE), nmjson::pretty::pretty(&v)).map_err(|e| format!("{FILE}: {e}"))
}

/// A repository's id from its URL: its last path part, letters, digits and `-`.
#[must_use]
pub fn id_of(url: &str) -> String {
    let last = url.trim_end_matches(['/', '\\']).rsplit(['/', '\\', ':']).next().unwrap_or("repo").trim_end_matches(".git");
    let id: String = last.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect();
    let id = id.trim_matches('-').chars().take(48).collect::<String>();
    if id.is_empty() { "repo".to_owned() } else { id }
}

fn url_ok(url: &str) -> bool {
    let u = url.trim();
    !u.is_empty() && u.len() <= 400 && !u.chars().any(|c| c.is_control() || c == ' ' && !Path::new(u).is_dir()) && (u.starts_with("https://") || u.starts_with("git@") || u.starts_with("ssh://") || Path::new(u).is_absolute())
}

fn mirror(home: &Path, id: &str) -> PathBuf {
    home.join("backups").join(id)
}

/// The host a URL names (`huggingface.co`, `github.com`), empty for a local folder.
fn host(url: &str) -> String {
    if let Some(r) = url.strip_prefix("https://").or_else(|| url.strip_prefix("ssh://")) {
        return r.split(['/', ':']).next().unwrap_or("").rsplit('@').next().unwrap_or("").to_ascii_lowercase();
    }
    if let Some(r) = url.strip_prefix("git@") {
        return r.split(':').next().unwrap_or("").to_ascii_lowercase();
    }
    String::new()
}

fn base64(b: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for c in b.chunks(3) {
        let x = (u32::from(c[0]) << 16) | (u32::from(*c.get(1).unwrap_or(&0)) << 8) | u32::from(*c.get(2).unwrap_or(&0));
        for i in 0..4 {
            if i <= c.len() {
                out.push(char::from(T[((x >> (18 - 6 * i)) & 63) as usize]));
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// The header that carries the user's token to this repository's host, when the user said to send
/// it; never logged.
fn auth_header(r: &Repo) -> Option<String> {
    if !r.token || !r.url.starts_with("https://") {
        return None;
    }
    let h = host(&r.url);
    if h == "huggingface.co" {
        return crate::fetch::token().map(|t| format!("Authorization: Bearer {t}"));
    }
    std::env::var("GITHUB_TOKEN").or_else(|_| std::env::var("GH_TOKEN")).ok().filter(|t| !t.trim().is_empty()).map(|t| format!("Authorization: Basic {}", base64(format!("x-access-token:{}", t.trim()).as_bytes())))
}

/// Runs git in `dir` (or with `-C dir`), never prompting; the token header is passed for this call
/// only and kept out of every message.
fn git(dir: &Path, args: &[&str], auth: Option<&str>) -> Result<String, String> {
    let mut c = Command::new("git");
    // The mirror and a local repository may sit on a drive whose owner git cannot read (exFAT):
    // the user chose them, so ownership is not checked.
    c.arg("-C").arg(dir).env("GIT_TERMINAL_PROMPT", "0").args(["-c", "safe.directory=*", "-c", "core.autocrlf=false", "-c", "core.safecrlf=false"]);
    if let Some(h) = auth {
        c.args(["-c", &format!("http.extraHeader={h}")]);
    }
    c.args(args);
    let out = c.output().map_err(|e| format!("git is not available on this machine ({e}); install Git for Windows to use your own repositories"))?;
    let msg = |b: &[u8]| String::from_utf8_lossy(b).trim().to_owned();
    if out.status.success() {
        Ok(msg(&out.stdout))
    } else {
        let mut e = msg(&out.stderr);
        if let Some(h) = auth {
            e = e.replace(h, "[token]");
        }
        Err(format!("git {}: {}", args.first().unwrap_or(&""), e.lines().last().unwrap_or("failed")))
    }
}

/// Whether git can be run.
#[must_use]
pub fn git_available() -> bool {
    Command::new("git").arg("--version").output().is_ok_and(|o| o.status.success())
}

/// `GET /api/backups`: the repositories, what each holds, its last snapshot when the mirror has one,
/// the parts there are, whether git runs here and whether the air gap is on.
#[must_use]
pub fn view(home: &Path) -> Value {
    let repos: Vec<Value> = list(home)
        .iter()
        .map(|r| {
            let mut v = r.to_value();
            let m = mirror(home, &r.id);
            let last = if m.join(".git").is_dir() { git(&m, &["log", "-1", "--format=%H%x09%ct%x09%s"], None).ok() } else { None };
            if let (Value::Object(o), Some(l)) = (&mut v, last.filter(|l| !l.is_empty())) {
                let mut it = l.splitn(3, '\t');
                o.insert("last".into(), obj(vec![("commit", s(it.next().unwrap_or(""))), ("at_s", n(it.next().and_then(|x| x.parse().ok()).unwrap_or(0))), ("message", s(it.next().unwrap_or("")))]));
            }
            v
        })
        .collect();
    obj(vec![
        ("repos", Value::Array(repos)),
        ("parts", Value::Array(PARTS.iter().map(|(p, paths)| obj(vec![("part", s(p)), ("paths", Value::Array(paths.iter().map(|x| s(x)).collect()))])).collect())),
        ("git", Value::Bool(git_available())),
        ("air_gap", Value::Bool(crate::controls::air_gap_on(home))),
        ("max_file_bytes", n(MAX_FILE)),
    ])
}

/// Adds a repository; audited, undone by taking it out again.
/// # Errors
/// A URL that does not read, an unknown part, or one already added.
pub fn add(home: &Path, url: &str, parts: &[String], token: bool, by: &str) -> Result<String, String> {
    let url = url.trim().to_owned();
    if !url_ok(&url) {
        return Err(format!("{url}: give a git URL (https://..., git@host:owner/name.git) or a local folder"));
    }
    let parts: Vec<String> = if parts.is_empty() { PARTS.iter().map(|(p, _)| (*p).to_owned()).filter(|p| p != "cartridges").collect() } else { parts.to_vec() };
    if let Some(bad) = parts.iter().find(|p| !PARTS.iter().any(|(k, _)| k == p)) {
        return Err(format!("part {bad}: use {}", PARTS.iter().map(|(k, _)| *k).collect::<Vec<_>>().join(", ")));
    }
    let mut repos = list(home);
    let id = id_of(&url);
    if repos.iter().any(|r| r.id == id || r.url == url) {
        return Err(format!("{url} is already one of your repositories"));
    }
    let r = Repo { id: id.clone(), url: url.clone(), parts: parts.clone(), token, added_ms: nmtype::now_ms() };
    repos.push(r);
    write(home, &repos)?;
    crate::controls::audit(
        home,
        AUDIT_KIND,
        vec![
            ("action", s("add")),
            ("repo", s(&id)),
            ("url", s(&url)),
            ("parts", Value::Array(parts.iter().map(|p| s(p)).collect())),
            ("token", Value::Bool(token)),
            ("by", s(by)),
            ("label", s(&format!("Backup repository {url} added"))),
        ],
    )
}

/// Takes a repository off the list (what it holds stays where it is); audited, undone by adding it
/// again.
/// # Errors
/// No such repository.
pub fn remove(home: &Path, id: &str, by: &str) -> Result<String, String> {
    let mut repos = list(home);
    let r = repos.iter().find(|r| r.id == id).cloned().ok_or_else(|| format!("no backup repository {id}"))?;
    repos.retain(|x| x.id != id);
    write(home, &repos)?;
    crate::controls::audit(
        home,
        AUDIT_KIND,
        vec![
            ("action", s("remove")),
            ("repo", s(id)),
            ("url", s(&r.url)),
            ("parts", Value::Array(r.parts.iter().map(|p| s(p)).collect())),
            ("token", Value::Bool(r.token)),
            ("by", s(by)),
            ("label", s(&format!("Backup repository {} taken off the list (its contents stay)", r.url))),
        ],
    )
}

fn find(home: &Path, id: &str) -> Result<Repo, String> {
    list(home).into_iter().find(|r| r.id == id).ok_or_else(|| format!("no backup repository {id}: add it first"))
}

/// The files the repository's parts take: (path relative to the Nano's folder, full path, bytes),
/// and the files left out with why.
/// A file a backup takes: (path relative to the Nano's folder, full path, bytes).
type Taken = (String, PathBuf, u64);
/// A file a backup leaves out: (relative path, bytes, why).
type LeftOut = (String, u64, &'static str);

fn collect(home: &Path, parts: &[String]) -> (Vec<Taken>, Vec<LeftOut>) {
    let mut take = Vec::new();
    let mut skipped = Vec::new();
    let mut stack: Vec<(String, PathBuf)> = Vec::new();
    for (part, paths) in PARTS {
        if !parts.iter().any(|p| p == part) {
            continue;
        }
        for rel in paths {
            stack.push(((*rel).to_owned(), home.join(rel)));
        }
    }
    while let Some((rel, p)) = stack.pop() {
        let Ok(md) = std::fs::metadata(&p) else { continue };
        if md.is_dir() {
            let name = p.file_name().and_then(|x| x.to_str()).unwrap_or("");
            if REBUILDABLE.contains(&name) || name.starts_with("substrate-cache") {
                skipped.push((rel, 0, "rebuilt by the Nano"));
                continue;
            }
            if let Ok(rd) = std::fs::read_dir(&p) {
                for e in rd.flatten() {
                    let n = e.file_name().to_string_lossy().into_owned();
                    stack.push((format!("{rel}/{n}"), e.path()));
                }
            }
        } else if rel.ends_with(".tmp") || rel.ends_with(".lock") {
            skipped.push((rel, md.len(), "temporary"));
        } else if md.len() > MAX_FILE {
            skipped.push((rel, md.len(), "over 50 MiB (git hosts refuse it without large-file support; base cartridges come back from the cartridge library)"));
        } else {
            take.push((rel, p, md.len()));
        }
    }
    take.sort();
    skipped.sort();
    (take, skipped)
}

fn clear_except_git(dir: &Path) -> Result<(), String> {
    for e in std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?.flatten() {
        if e.file_name() == ".git" {
            continue;
        }
        let p = e.path();
        let r = if p.is_dir() { std::fs::remove_dir_all(&p) } else { std::fs::remove_file(&p) };
        r.map_err(|x| format!("{}: {x}", p.display()))?;
    }
    Ok(())
}

/// The local mirror of a repository, cloned (or made, for an empty repository) on first use and
/// brought to the repository's newest snapshot.
fn ready_mirror(home: &Path, r: &Repo, auth: Option<&str>) -> Result<PathBuf, String> {
    let m = mirror(home, &r.id);
    if !m.join(".git").is_dir() {
        std::fs::create_dir_all(&m).map_err(|e| format!("{}: {e}", m.display()))?;
        git(&m, &["init", "-q"], None)?;
        git(&m, &["remote", "add", "origin", &r.url], None)?;
    }
    // A repository in a local folder is read and written by git's own programs in that folder,
    // which check its owner again; the user chose the folder, so they are told not to.
    if Path::new(&r.url).is_absolute() {
        git(&m, &["config", "remote.origin.receivepack", "git -c safe.directory=* receive-pack"], None)?;
        git(&m, &["config", "remote.origin.uploadpack", "git -c safe.directory=* upload-pack"], None)?;
    }
    // A repository with snapshots: the mirror is brought to its newest; an empty one has none yet.
    if git(&m, &["fetch", "-q", "origin"], auth).is_ok() && git(&m, &["rev-parse", "--verify", "-q", "origin/main"], None).is_ok() {
        git(&m, &["checkout", "-q", "-B", "main", "origin/main"], None)?;
    } else if git(&m, &["rev-parse", "--verify", "-q", "HEAD"], None).is_err() {
        git(&m, &["symbolic-ref", "HEAD", "refs/heads/main"], None)?;
    }
    Ok(m)
}

/// Backs up the repository's parts: copied into its mirror with a manifest, committed and pushed.
/// # Errors
/// The air gap on, git missing, or git refusing (the message says why).
pub fn backup(home: &Path, id: &str, by: &str) -> Result<Value, String> {
    if crate::controls::air_gap_on(home) {
        return Err("the air gap is on: nothing is sent; switch it off on Settings to back up".to_owned());
    }
    let r = find(home, id)?;
    // The live view shows the backup while it copies, commits and pushes.
    let mut live = nmapi::live::stage_as("backup", "backup", "", "failed");
    let auth = auth_header(&r);
    let m = ready_mirror(home, &r, auth.as_deref())?;
    clear_except_git(&m)?;
    let (take, skipped) = collect(home, &r.parts);
    let mut bytes = 0_u64;
    for (rel, src, len) in &take {
        let dst = m.join(rel);
        if let Some(d) = dst.parent() {
            std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        std::fs::copy(src, &dst).map_err(|e| format!("{}: {e}", src.display()))?;
        bytes += len;
    }
    let manifest = obj(vec![
        ("schema", s("nm-backup/1")),
        ("made_ms", n(nmtype::now_ms())),
        ("nano", s(env!("CARGO_PKG_VERSION"))),
        ("parts", Value::Array(r.parts.iter().map(|p| s(p)).collect())),
        ("files", n(take.len() as u64)),
        ("bytes", n(bytes)),
        ("left_out", Value::Array(skipped.iter().map(|(p, b, why)| obj(vec![("path", s(p)), ("bytes", n(*b)), ("why", s(why))])).collect())),
    ]);
    std::fs::write(m.join("MANIFEST.json"), nmjson::pretty::pretty(&manifest)).map_err(|e| format!("MANIFEST.json: {e}"))?;
    git(&m, &["add", "-A"], None)?;
    let when = nmlearn::cartridge::date_of(nmtype::now_ms());
    let msg = format!("Neuromorph Nano backup {when}: {} ({} files)", r.parts.join(", "), take.len());
    let changed = git(&m, &["status", "--porcelain"], None)?;
    if !changed.is_empty() {
        git(&m, &["-c", "user.name=Neuromorph Nano", "-c", "user.email=nano@localhost", "commit", "-q", "-m", &msg], None)?;
    }
    git(&m, &["push", "-q", "origin", "HEAD:refs/heads/main"], auth.as_deref())?;
    let commit = git(&m, &["rev-parse", "HEAD"], None)?;
    live.count("files", take.len());
    live.label("pushed");
    drop(live);
    let audit = crate::controls::audit(
        home,
        AUDIT_KIND,
        vec![
            ("action", s("backup")),
            ("repo", s(id)),
            ("commit", s(&commit)),
            ("files", n(take.len() as u64)),
            ("bytes", n(bytes)),
            ("left_out", n(skipped.len() as u64)),
            ("by", s(by)),
            ("label", s(&format!("Backed up {} to {} ({} files)", r.parts.join(", "), r.url, take.len()))),
        ],
    )?;
    Ok(obj(vec![
        ("audit_id", s(&audit)),
        ("repo", s(id)),
        ("commit", s(&commit)),
        ("changed", Value::Bool(!changed.is_empty())),
        ("files", n(take.len() as u64)),
        ("bytes", n(bytes)),
        ("left_out", Value::Array(skipped.iter().take(200).map(|(p, b, why)| obj(vec![("path", s(p)), ("bytes", n(*b)), ("why", s(why))])).collect())),
    ]))
}

/// The repository's snapshots, newest first (one per backup).
/// # Errors
/// No such repository, or git refusing.
pub fn snapshots(home: &Path, id: &str) -> Result<Value, String> {
    let r = find(home, id)?;
    let m = mirror(home, &r.id);
    if !m.join(".git").is_dir() {
        if crate::controls::air_gap_on(home) {
            return Ok(obj(vec![("repo", s(id)), ("snapshots", Value::Array(Vec::new())), ("note", s("no local mirror yet, and the air gap is on"))]));
        }
        ready_mirror(home, &r, auth_header(&r).as_deref())?;
    } else if !crate::controls::air_gap_on(home) {
        let _ = git(&m, &["fetch", "-q", "origin"], auth_header(&r).as_deref());
    }
    let reference = if git(&m, &["rev-parse", "--verify", "-q", "origin/main"], None).is_ok() { "origin/main" } else { "HEAD" };
    let log = git(&m, &["log", "-n", "100", "--format=%H%x09%ct%x09%s", reference], None).unwrap_or_default();
    let rows: Vec<Value> = log
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| {
            let mut it = l.splitn(3, '\t');
            obj(vec![("commit", s(it.next().unwrap_or(""))), ("at_s", n(it.next().and_then(|x| x.parse().ok()).unwrap_or(0))), ("message", s(it.next().unwrap_or("")))])
        })
        .collect();
    Ok(obj(vec![("repo", s(id)), ("snapshots", Value::Array(rows))]))
}

fn commit_ok(c: &str) -> bool {
    (7..=40).contains(&c.len()) && c.chars().all(|x| x.is_ascii_hexdigit())
}

/// A snapshot's files written into a folder of their own: (folder, its files relative).
fn unpack(home: &Path, r: &Repo, commit: &str) -> Result<(PathBuf, Vec<String>), String> {
    if !commit_ok(commit) {
        return Err(format!("{commit}: give a snapshot's commit id as the snapshots list it"));
    }
    let m = mirror(home, &r.id);
    if !m.join(".git").is_dir() {
        return Err(format!("{}: no local mirror yet; list its snapshots first", r.id));
    }
    let names = git(&m, &["ls-tree", "-r", "--name-only", commit], None)?;
    let out = home.join("backups").join(format!("{}-snapshot-{}", r.id, &commit[..7.min(commit.len())]));
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let mut files = Vec::new();
    for rel in names.lines().filter(|l| !l.is_empty() && *l != "MANIFEST.json") {
        if rel.split('/').any(|p| p == ".." || p.is_empty()) {
            continue;
        }
        let blob = Command::new("git").arg("-C").arg(&m).args(["-c", "safe.directory=*", "show", &format!("{commit}:{rel}")]).output().map_err(|e| format!("git show: {e}"))?;
        if !blob.status.success() {
            return Err(format!("{rel}: not in snapshot {commit}"));
        }
        let dst = out.join(rel);
        if let Some(d) = dst.parent() {
            std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        std::fs::write(&dst, &blob.stdout).map_err(|e| format!("{}: {e}", dst.display()))?;
        files.push(rel.to_owned());
    }
    Ok((out, files))
}

/// What restoring a snapshot would write: new files and changed files (files only on this machine
/// are kept, never deleted).
fn plan(home: &Path, unpacked: &Path, files: &[String]) -> (Vec<String>, Vec<String>, usize) {
    let (mut added, mut changed, mut same) = (Vec::new(), Vec::new(), 0);
    for rel in files {
        let local = home.join(rel);
        match std::fs::read(&local) {
            Err(_) => added.push(rel.clone()),
            Ok(b) if std::fs::read(unpacked.join(rel)).ok().as_deref() != Some(b.as_slice()) => changed.push(rel.clone()),
            Ok(_) => same += 1,
        }
    }
    (added, changed, same)
}

/// `preview`: what restoring `commit` would write; nothing is written to the Nano's data.
/// # Errors
/// No such repository or snapshot.
pub fn preview(home: &Path, id: &str, commit: &str) -> Result<Value, String> {
    let r = find(home, id)?;
    let (out, files) = unpack(home, &r, commit)?;
    let (added, changed, same) = plan(home, &out, &files);
    let _ = std::fs::remove_dir_all(&out);
    Ok(obj(vec![
        ("repo", s(id)),
        ("commit", s(commit)),
        ("new", n(added.len() as u64)),
        ("changed", n(changed.len() as u64)),
        ("same", n(same as u64)),
        ("new_files", Value::Array(added.iter().take(200).map(|x| s(x)).collect())),
        ("changed_files", Value::Array(changed.iter().take(200).map(|x| s(x)).collect())),
        ("note", s("restoring writes the new and changed files; each changed file is copied aside first and Audit undoes the restore; files only on this machine are kept")),
    ]))
}

/// Restores a snapshot: every changed file copied aside first, then the snapshot's new and changed
/// files written; audited, undone by putting the copies back and taking out the added files.
/// # Errors
/// Not confirmed, or no such repository or snapshot.
pub fn restore(home: &Path, id: &str, commit: &str, confirm: bool, by: &str) -> Result<Value, String> {
    if !confirm {
        return Err("restore writes files back: preview it first, then send confirm: true".to_owned());
    }
    let r = find(home, id)?;
    let (out, files) = unpack(home, &r, commit)?;
    let (added, changed, _) = plan(home, &out, &files);
    let aside = home.join("backups").join(format!("{}-before-{}", r.id, nmtype::now_ms()));
    for rel in &changed {
        let dst = aside.join(rel);
        if let Some(d) = dst.parent() {
            std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        std::fs::copy(home.join(rel), &dst).map_err(|e| format!("{rel}: {e}"))?;
    }
    for rel in added.iter().chain(&changed) {
        let dst = home.join(rel);
        if let Some(d) = dst.parent() {
            std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        std::fs::copy(out.join(rel), &dst).map_err(|e| format!("{rel}: {e}"))?;
    }
    std::fs::create_dir_all(&aside).map_err(|e| format!("{}: {e}", aside.display()))?;
    let record = obj(vec![("added", Value::Array(added.iter().map(|x| s(x)).collect())), ("changed", Value::Array(changed.iter().map(|x| s(x)).collect()))]);
    std::fs::write(aside.join("RESTORE.json"), nmjson::pretty::pretty(&record)).map_err(|e| format!("RESTORE.json: {e}"))?;
    let _ = std::fs::remove_dir_all(&out);
    let audit = crate::controls::audit(
        home,
        AUDIT_KIND,
        vec![
            ("action", s("restore")),
            ("repo", s(id)),
            ("commit", s(commit)),
            ("aside", s(&aside.display().to_string())),
            ("new", n(added.len() as u64)),
            ("changed", n(changed.len() as u64)),
            ("by", s(by)),
            ("label", s(&format!("Restored snapshot {} of {} ({} new, {} changed files)", &commit[..7.min(commit.len())], r.url, added.len(), changed.len()))),
        ],
    )?;
    Ok(obj(vec![("audit_id", s(&audit)), ("repo", s(id)), ("commit", s(commit)), ("new", n(added.len() as u64)), ("changed", n(changed.len() as u64)), ("aside", s(&aside.display().to_string()))]))
}

/// Undoes an audited change: an add (the repository taken off again), a remove (added again), a
/// restore (the copies put back, the added files taken out). A backup is not undone here: its
/// snapshot stays in the user's repository, which is theirs.
/// # Errors
/// An action this does not undo, or a file that cannot be put back.
pub fn undo(home: &Path, row: &Value, id: &str, by: &str) -> Result<Option<String>, String> {
    let parts: Vec<String> = row.get("parts").and_then(Value::array).unwrap_or_default().iter().filter_map(|p| p.string().ok().map(str::to_owned)).collect();
    let audit = match text(row, "action").as_str() {
        "add" => remove(home, &text(row, "repo"), by)?,
        "remove" => add(home, &text(row, "url"), &parts, matches!(row.get("token"), Ok(Value::Bool(true))), by)?,
        "restore" => {
            let aside = PathBuf::from(text(row, "aside"));
            let rec = std::fs::read(aside.join("RESTORE.json")).ok().and_then(|b| nmjson::parse(&b).ok()).ok_or_else(|| format!("{id}: the copies of this restore are not on disk any more"))?;
            let strs = |k: &str| -> Vec<String> { rec.get(k).and_then(Value::array).unwrap_or_default().iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect() };
            for rel in strs("changed") {
                std::fs::copy(aside.join(&rel), home.join(&rel)).map_err(|e| format!("{rel}: {e}"))?;
            }
            let mut removed = 0_u64;
            for rel in strs("added") {
                if std::fs::remove_file(home.join(&rel)).is_ok() {
                    removed += 1;
                }
            }
            crate::controls::audit(
                home,
                AUDIT_KIND,
                vec![("action", s("undo")), ("undoes", s(id)), ("repo", s(&text(row, "repo"))), ("removed", n(removed)), ("by", s(by)), ("label", s(&format!("Restore {id} undone: the files it changed put back, the {removed} it added taken out")))],
            )?
        }
        a => return Err(format!("{id}: a backup {a} is not undone (a snapshot stays in your repository)")),
    };
    Ok(Some(audit))
}

/// `POST /api/backups`.
/// # Errors
/// As each action.
pub fn api(home: &Path, v: &Value, by: &str) -> Result<Value, String> {
    let action = v.get("action").and_then(Value::string).map_err(|_| "action: add, remove, backup, snapshots, preview or restore".to_owned())?;
    let id = || v.get("id").and_then(Value::string).map_err(|_| "id: the repository's id as /api/backups lists it".to_owned());
    let commit = || v.get("commit").and_then(Value::string).map_err(|_| "commit: a snapshot's commit id".to_owned());
    match action {
        "add" => {
            let url = v.get("url").and_then(Value::string).map_err(|_| "url: the repository's git URL or a local folder".to_owned())?;
            let parts: Vec<String> = v.get("parts").and_then(Value::array).unwrap_or_default().iter().filter_map(|p| p.string().ok().map(str::to_owned)).collect();
            let audit = add(home, url, &parts, matches!(v.get("token"), Ok(Value::Bool(true))), by)?;
            Ok(obj(vec![("audit_id", s(&audit)), ("backups", view(home))]))
        }
        "remove" => {
            let audit = remove(home, id()?, by)?;
            Ok(obj(vec![("audit_id", s(&audit)), ("backups", view(home))]))
        }
        "backup" => backup(home, id()?, by),
        "snapshots" => snapshots(home, id()?),
        "preview" => preview(home, id()?, commit()?),
        "restore" => restore(home, id()?, commit()?, matches!(v.get("confirm"), Ok(Value::Bool(true))), by),
        other => Err(format!("action {other}: use add, remove, backup, snapshots, preview or restore")),
    }
}

/// `neuromorph-nano backup ...`.
pub fn cli(home: &Path, args: &[String]) -> std::process::ExitCode {
    let a = |i: usize| args.get(i).map_or("", String::as_str);
    let flag = |k: &str| args.iter().any(|x| x == k);
    let opt = |k: &str| args.iter().position(|x| x == k).and_then(|i| args.get(i + 1)).map(String::as_str);
    let res = match a(1) {
        "" | "list" => Ok(view(home)),
        "add" => {
            let parts: Vec<String> = opt("--parts").map(|p| p.split(',').map(|x| x.trim().to_owned()).filter(|x| !x.is_empty()).collect()).unwrap_or_default();
            add(home, a(2), &parts, flag("--token"), "cli").map(|id| obj(vec![("audit_id", s(&id))]))
        }
        "remove" => remove(home, a(2), "cli").map(|id| obj(vec![("audit_id", s(&id))])),
        "now" => backup(home, a(2), "cli"),
        "snapshots" => snapshots(home, a(2)),
        "preview" => preview(home, a(2), a(3)),
        "restore" => restore(home, a(2), a(3), flag("--confirm"), "cli"),
        _ => Err("usage: backup [list] | add <git url or folder> [--parts memory,maps,cartridges,units,settings] [--token] | remove <id> | now <id> | snapshots <id> | preview <id> <commit> | restore <id> <commit> --confirm".to_owned()),
    };
    match res {
        Ok(v) => {
            println!("{}", nmjson::pretty::pretty(&v));
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("backup: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Home(PathBuf);
    impl Drop for Home {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn home(name: &str) -> Home {
        let d = std::env::temp_dir().join(format!("nano-backups-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("memory/dense")).unwrap();
        std::fs::create_dir_all(d.join("library/parameter-maps/reranker")).unwrap();
        std::fs::write(d.join("memory/episodes.jsonl"), b"one\n").unwrap();
        std::fs::write(d.join("memory/dense/vectors.bin"), b"rebuilt").unwrap();
        std::fs::write(d.join("library/parameter-maps/reranker/pmap-a.json"), b"{\"a\":1}").unwrap();
        Home(d)
    }

    #[test]
    fn ids_hosts_and_base64_read() {
        assert_eq!(id_of("https://github.com/me/nano-backups.git"), "nano-backups");
        assert_eq!(id_of("git@github.com:me/My_Backups.git"), "my-backups");
        assert_eq!(host("https://huggingface.co/datasets/me/x"), "huggingface.co");
        assert_eq!(host("git@github.com:me/x.git"), "github.com");
        assert_eq!(base64(b"x-access-token:abc"), "eC1hY2Nlc3MtdG9rZW46YWJj");
        assert!(!url_ok("ftp://x"));
    }

    /// A local bare repository stands in for the user's own: back up, change a file, preview,
    /// restore, undo; rebuildable stores are left out.
    #[test]
    fn a_backup_is_restored_and_the_restore_undone() {
        if !git_available() {
            return;
        }
        let h = home("round");
        let remote = h.0.join("remote.git");
        std::fs::create_dir_all(&remote).unwrap();
        assert!(Command::new("git").arg("-C").arg(&remote).args(["-c", "safe.directory=*", "init", "-q", "--bare"]).status().unwrap().success());
        add(&h.0, &remote.display().to_string(), &["memory".into(), "maps".into()], false, "test").unwrap();
        let id = list(&h.0)[0].id.clone();
        let b = backup(&h.0, &id, "test").unwrap();
        let int = |v: &Value, k: &str| v.get(k).and_then(Value::integer).unwrap_or(0);
        assert!(int(&b, "files") >= 2, "episodes and the map (and the audit log)");
        let left: Vec<String> = b.get("left_out").and_then(Value::array).unwrap_or_default().iter().map(|x| text(x, "path")).collect();
        assert!(left.iter().any(|p| p == "memory/dense"), "the dense store is rebuilt, not backed up: {left:?}");
        let snaps = snapshots(&h.0, &id).unwrap();
        let commit = snaps.get("snapshots").and_then(Value::array).unwrap()[0].get("commit").and_then(Value::string).unwrap().to_owned();
        std::fs::write(h.0.join("memory/episodes.jsonl"), b"two\n").unwrap();
        std::fs::remove_file(h.0.join("library/parameter-maps/reranker/pmap-a.json")).unwrap();
        let p = preview(&h.0, &id, &commit).unwrap();
        assert_eq!((int(&p, "changed") >= 1, int(&p, "new")), (true, 1));
        assert_eq!(std::fs::read(h.0.join("memory/episodes.jsonl")).unwrap(), b"two\n", "a preview writes nothing");
        assert!(restore(&h.0, &id, &commit, false, "test").is_err(), "a restore needs confirm");
        let r = restore(&h.0, &id, &commit, true, "test").unwrap();
        assert_eq!(std::fs::read(h.0.join("memory/episodes.jsonl")).unwrap(), b"one\n");
        assert!(h.0.join("library/parameter-maps/reranker/pmap-a.json").is_file());
        let audit_id = text(&r, "audit_id");
        let row = obj(vec![("action", s("restore")), ("repo", s(&id)), ("aside", r.get("aside").unwrap().clone())]);
        undo(&h.0, &row, &audit_id, "test").unwrap();
        assert_eq!(std::fs::read(h.0.join("memory/episodes.jsonl")).unwrap(), b"two\n", "the undo puts the changed file back");
        assert!(!h.0.join("library/parameter-maps/reranker/pmap-a.json").exists(), "and takes the added one out");
    }
}
