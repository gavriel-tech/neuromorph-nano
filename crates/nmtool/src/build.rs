//! Build runs: one package, one cargo command, rustc JSON diagnostics into cited packets
//! (guide 17.3; vision 18.1 build output as a foraging source). Body first written in the
//! earlier `nmcore::build` pass (kept there); extended here with error codes, suggestion
//! applicability, panic locations, a deadline and the `--offline --locked` rule so a run
//! never writes a lock file into the studied tree.
use crate::run::{self, Output};
use nmjson::Value;
use nmtype::packet::{Category, Certainty, Provenance, Stance, Status};
use nmtype::{Fault, FaultKind, Packet};
use std::path::{Path, PathBuf};
use std::time::Duration;

const ME: &str = "nmtool.build";

/// One child note or help line of a diagnostic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Child {
    /// note, help.
    pub level: String,
    pub message: String,
    /// The replacement text rustc suggests, when it suggests one.
    pub replacement: Option<String>,
    /// `MachineApplicable`, `MaybeIncorrect`, `HasPlaceholders`, `Unspecified` or empty.
    pub applicability: String,
    /// Where the suggestion applies: file, line, first column, last column.
    pub at: Option<(String, usize, usize, usize)>,
}

/// One compiler message, located at its primary span.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    /// error, warning.
    pub level: String,
    /// `E0425`, a lint name, or empty.
    pub code: String,
    pub message: String,
    /// Relative to the root, `/` separators; empty when rustc gives no span.
    pub file: String,
    pub line: usize,
    pub col: usize,
    /// The primary span's label.
    pub label: String,
    pub children: Vec<Child>,
}

/// A failed test, located at its panic when the output gives one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TestFail {
    pub name: String,
    pub at: Option<(String, usize)>,
    pub message: String,
}

/// The cargo commands a build run may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Check,
    Clippy,
    Test,
    Build,
}
impl Kind {
    /// From a tool argument; `check` when empty.
    /// # Errors
    /// Any other word.
    pub fn parse(s: &str) -> Result<Self, Fault> {
        match s {
            "" | "check" => Ok(Self::Check),
            "clippy" => Ok(Self::Clippy),
            "test" => Ok(Self::Test),
            "build" => Ok(Self::Build),
            o => Err(Fault::new(ME, FaultKind::Config, format!("kind {o:?}: check, clippy, test or build"))),
        }
    }
    #[must_use]
    pub const fn word(self) -> &'static str {
        match self {
            Self::Check => "check",
            Self::Clippy => "clippy",
            Self::Test => "test",
            Self::Build => "build",
        }
    }
}

/// What to run.
#[derive(Clone, Debug)]
pub struct Request {
    /// The workspace root (the studied tree or a scratch copy).
    pub root: PathBuf,
    /// Cargo target folder, always inside the Nano folder's `scratch/` or `target/`.
    pub target: PathBuf,
    pub kind: Kind,
    /// One package (`-p`); the whole workspace when `None`.
    pub package: Option<String>,
    pub deadline: Duration,
}

/// One build run.
#[derive(Clone, Debug, Default)]
pub struct BuildRun {
    /// `b<ms>`: the id `nm_build_explain` takes.
    pub id: String,
    pub kind: String,
    pub package: Option<String>,
    /// Cargo exited with success.
    pub ok: bool,
    pub timed_out: bool,
    pub diags: Vec<Diagnostic>,
    pub tests: Vec<TestFail>,
    /// Cargo-level error lines when rustc gave no diagnostic (unknown package, lock file).
    pub cargo_errors: Vec<String>,
    /// Bytes of raw cargo output (JSON stdout plus stderr).
    pub raw_bytes: usize,
    /// Wall milliseconds of the cargo process.
    pub cargo_ms: u64,
}
impl BuildRun {
    /// Errors first, then warnings.
    #[must_use]
    pub fn errors(&self) -> Vec<&Diagnostic> {
        self.diags.iter().filter(|d| d.level == "error").collect()
    }
    #[must_use]
    pub fn warnings(&self) -> usize {
        self.diags.iter().filter(|d| d.level == "warning").count()
    }
}

fn st(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default()
}
fn int(v: &Value, k: &str) -> usize {
    v.get(k).and_then(Value::integer).ok().and_then(|x| usize::try_from(x).ok()).unwrap_or(0)
}
fn arr<'a>(v: &'a Value, k: &str) -> &'a [Value] {
    v.get(k).and_then(Value::array).unwrap_or(&[])
}

/// `path` relative to `root` with `/` separators (rustc gives workspace-relative paths for
/// workspace members and absolute ones for some others).
#[must_use]
pub fn relative(root: &Path, path: &str) -> String {
    let p = path.replace('\\', "/");
    let r = root.to_string_lossy().replace('\\', "/");
    let r = r.trim_end_matches('/');
    p.strip_prefix(r).map_or(p.clone(), |x| x.trim_start_matches('/').to_owned())
}

fn child(root: &Path, c: &Value) -> Child {
    let sp = arr(c, "spans").first();
    let replacement = sp.and_then(|s| s.get("suggested_replacement").and_then(Value::string).ok().map(str::to_owned));
    Child {
        level: st(c, "level"),
        message: st(c, "message"),
        replacement,
        applicability: sp.map(|s| st(s, "suggestion_applicability")).unwrap_or_default(),
        at: sp.map(|s| (relative(root, &st(s, "file_name")), int(s, "line_start"), int(s, "column_start"), int(s, "column_end"))),
    }
}

fn diagnostic(root: &Path, m: &Value) -> Option<Diagnostic> {
    let level = st(m, "level");
    if level != "error" && level != "warning" {
        return None;
    }
    let message = st(m, "message");
    let spans = arr(m, "spans");
    if spans.is_empty()
        && (message.starts_with("aborting due to") || message.contains("warning emitted") || message.contains("warnings emitted"))
    {
        return None;
    }
    let primary = spans.iter().find(|s| matches!(s.get("is_primary"), Ok(Value::Bool(true)))).or_else(|| spans.first());
    let code = m.get("code").ok().map(|c| st(c, "code")).unwrap_or_default();
    Some(Diagnostic {
        level,
        code,
        message,
        file: primary.map(|s| relative(root, &st(s, "file_name"))).unwrap_or_default(),
        line: primary.map_or(0, |s| int(s, "line_start")),
        col: primary.map_or(0, |s| int(s, "column_start")),
        label: primary.map(|s| st(s, "label")).unwrap_or_default(),
        children: arr(m, "children").iter().map(|c| child(root, c)).filter(|c| !c.message.is_empty()).collect(),
    })
}

/// `thread 'name' panicked at src/x.rs:12:5:` gives (name, file, line).
fn panic_at(root: &Path, line: &str) -> Option<(String, String, usize)> {
    let rest = line.strip_prefix("thread '")?;
    let (name, rest) = rest.split_once("' panicked at ")?;
    let loc = rest.trim_end_matches(':');
    let mut it = loc.rsplitn(3, ':');
    let _col = it.next()?;
    let ln: usize = it.next()?.parse().ok()?;
    Some((name.to_owned(), relative(root, it.next()?), ln))
}

/// Parse cargo's JSON stdout and plain stderr into diagnostics, failed tests and cargo-level
/// errors. Duplicates (the same message for two targets) are kept once.
#[must_use]
pub fn parse(root: &Path, stdout: &str, stderr: &str) -> (Vec<Diagnostic>, Vec<TestFail>, Vec<String>) {
    let mut diags: Vec<Diagnostic> = Vec::new();
    let mut tests: Vec<TestFail> = Vec::new();
    let mut panics: Vec<(String, String, usize, String)> = Vec::new();
    let mut lines = stdout.lines().chain(stderr.lines()).peekable();
    while let Some(line) = lines.next() {
        if line.starts_with('{') {
            let Ok(v) = nmjson::parse(line.as_bytes()) else { continue };
            if st(&v, "reason") != "compiler-message" {
                continue;
            }
            if let Some(d) = v.get("message").ok().and_then(|m| diagnostic(root, m))
                && !diags.contains(&d)
            {
                diags.push(d);
            }
        } else if let Some(name) = line.strip_prefix("test ").and_then(|l| l.strip_suffix(" ... FAILED")) {
            if !tests.iter().any(|t| t.name == name) {
                tests.push(TestFail { name: name.to_owned(), at: None, message: String::new() });
            }
        } else if let Some((name, file, ln)) = panic_at(root, line) {
            let msg = lines.peek().map(|m| m.trim().to_owned()).unwrap_or_default();
            panics.push((name, file, ln, msg));
        }
    }
    for t in &mut tests {
        if let Some((_, f, l, m)) = panics.iter().find(|p| p.0 == t.name || t.name.ends_with(&p.0)) {
            t.at = Some((f.clone(), *l));
            t.message.clone_from(m);
        }
    }
    let cargo_errors = if diags.iter().any(|d| d.level == "error") {
        Vec::new()
    } else {
        stderr.lines().filter(|l| l.starts_with("error")).map(|l| l.chars().take(300).collect()).collect()
    };
    diags.sort_by_key(|d| usize::from(d.level != "error"));
    (diags, tests, cargo_errors)
}

/// Run one cargo command for one package under the deadline.
#[must_use]
pub fn run(req: &Request) -> BuildRun {
    let mut args = vec![req.kind.word(), "--message-format=json", "--offline", "--locked", "--color", "never"];
    if let Some(p) = &req.package {
        args.extend(["-p", p.as_str()]);
    }
    if req.kind == Kind::Test {
        args.push("--no-fail-fast");
    }
    let td = req.target.to_string_lossy().into_owned();
    let env = [("CARGO_TARGET_DIR", td.as_str()), ("CARGO_INCREMENTAL", "0"), ("CARGO_TERM_COLOR", "never")];
    let o: Output = run::run(&req.root, "cargo", &args, &env, req.deadline);
    let (diags, tests, mut cargo_errors) = parse(&req.root, &o.stdout, &o.stderr);
    if o.timed_out {
        cargo_errors.push(format!("killed at the {} s deadline", req.deadline.as_secs()));
    }
    BuildRun {
        id: format!("b{}", nmtype::now_ms()),
        kind: req.kind.word().to_owned(),
        package: req.package.clone(),
        ok: o.ok,
        timed_out: o.timed_out,
        diags,
        tests,
        cargo_errors,
        raw_bytes: o.stdout.len() + o.stderr.len(),
        cargo_ms: o.ms,
    }
}

fn observed(kind: &str, key: &str, text: &str, source: &str) -> Packet {
    Packet::new(kind, key, text, Provenance::new(source, ME, 0, nmtype::now_ms()))
        .status(Status::Observed)
        .stance(Stance::Objective)
        .certainty(Certainty::Absolute)
        .category(Category::Fact)
        .confidence(1.0)
}

/// One `build-error` (or `build-warning`) packet per diagnostic, errors first, at most
/// `budget` of them; warnings only when asked. Failed tests become `test-result` packets.
#[must_use]
pub fn packets(run: &BuildRun, budget: usize, warnings: bool) -> Vec<Packet> {
    let mut out = Vec::new();
    for d in run.diags.iter().filter(|d| warnings || d.level == "error").take(budget) {
        let kind = if d.level == "error" { "build-error" } else { "build-warning" };
        let head = if d.code.is_empty() { d.level.clone() } else { format!("{}[{}]", d.level, d.code) };
        let source = format!("{}:{}", d.file, d.line);
        let mut ev = Vec::new();
        if !d.label.is_empty() {
            ev.push(d.label.clone());
        }
        for c in &d.children {
            ev.push(match &c.replacement {
                Some(r) if !r.is_empty() => format!("{}: {} -> `{r}`", c.level, c.message),
                _ => format!("{}: {}", c.level, c.message),
            });
        }
        out.push(observed(kind, &format!("{}:{source}", d.code), &format!("{head}: {}", d.message), &source).evidence(ev));
    }
    for t in run.tests.iter().take(budget.saturating_sub(out.len()).max(1)) {
        let source = t.at.as_ref().map_or_else(|| format!("test:{}", t.name), |(f, l)| format!("{f}:{l}"));
        out.push(observed("test-result", &format!("failed:{}", t.name), &format!("test {} failed: {}", t.name, t.message), &source));
    }
    out
}

/// Section check: the parser on a recorded rustc message, a test panic and a cargo error.
/// # Errors
/// One line naming what failed.
pub fn check() -> Result<(), String> {
    let root = Path::new("D:/w/tree");
    let msg = r#"{"reason":"compiler-message","message":{"level":"error","message":"cannot find value `x` in this scope","code":{"code":"E0425"},"spans":[{"file_name":"crates/a/src/lib.rs","line_start":7,"column_start":13,"column_end":14,"is_primary":true,"label":"not found in this scope","suggested_replacement":null,"suggestion_applicability":null}],"children":[{"level":"help","message":"a local variable with a similar name exists","spans":[{"file_name":"crates/a/src/lib.rs","line_start":7,"column_start":13,"column_end":14,"is_primary":true,"label":null,"suggested_replacement":"y","suggestion_applicability":"MaybeIncorrect"}],"children":[]}]}}"#;
    let summary = r#"{"reason":"compiler-message","message":{"level":"error","message":"aborting due to 1 previous error","code":null,"spans":[],"children":[]}}"#;
    let stdout = format!("{msg}\n{msg}\n{summary}\ntest t::one ... FAILED\n");
    let stderr = "thread 't::one' panicked at D:/w/tree/crates/a/src/lib.rs:20:9:\nassertion failed\n";
    let (d, t, c) = parse(root, &stdout, stderr);
    let first = d.first().ok_or("no diagnostic")?;
    if d.len() != 1 || first.code != "E0425" || first.file != "crates/a/src/lib.rs" || first.line != 7 || first.col != 13 {
        return Err(format!("diagnostics {d:?}"));
    }
    if first.children.first().and_then(|c| c.replacement.as_deref()) != Some("y") {
        return Err("child replacement lost".into());
    }
    if t.len() != 1 || t.first().and_then(|x| x.at.clone()) != Some(("crates/a/src/lib.rs".into(), 20)) {
        return Err(format!("tests {t:?}"));
    }
    if !c.is_empty() {
        return Err(format!("cargo errors with a diagnostic present: {c:?}"));
    }
    let (_, _, c2) = parse(root, "", "error: package ID specification `zz` did not match any packages\n");
    if c2.len() != 1 {
        return Err(format!("cargo-level error not kept: {c2:?}"));
    }
    let r = BuildRun { diags: d, tests: t, ..BuildRun::default() };
    let p = packets(&r, 12, false);
    if p.len() != 2 || p.first().map(Packet::kind) != Some("build-error") || p.first().map(|x| x.get_status()) != Some(Status::Observed) {
        return Err(format!("packets {}", p.len()));
    }
    Kind::parse("bench").err().ok_or("unknown kind accepted")?;
    Ok(())
}
