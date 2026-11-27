//! The S9 build tools Claude Code calls (PLAN section 7; guide 17.3, 17.5): `nm_build_run`,
//! `nm_build_explain`, `nm_stepcheck`, over `nmtool`. Replies are compact: errors as cited
//! lines, warnings counted, so Claude reads a fraction of cargo's output (capability 4).
//! Every run is also logged as a build outcome, and an error that a later clean run of the
//! same package no longer shows is logged with the lines before and after: the first
//! learn-from-mistakes data (S9 reorder, Luis Jake Gabriel III 08:24 UTC).
use crate::ApiError;
use nmjson::Value;
use nmtool::build::{self, BuildRun, Kind, Request};
use nmtool::{explain, run, stepcheck};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// Build runs kept for `nm_build_explain` (newest last).
const KEEP: usize = 8;
static RUNS: Mutex<Vec<BuildRun>> = Mutex::new(Vec::new());
static TARGET: OnceLock<PathBuf> = OnceLock::new();

/// Set the cargo target folder for build runs (first call wins); by default
/// `<home>/scratch/build-target`. Always a folder inside the mini.
pub fn set_target(dir: PathBuf) {
    let _ = TARGET.set(dir);
}
fn target(home: &Path) -> PathBuf {
    TARGET.get().cloned().unwrap_or_else(|| home.join("scratch/build-target"))
}

fn s(t: &str) -> Value {
    Value::String(t.to_owned())
}
fn n(x: impl ToString) -> Value {
    Value::Number(x.to_string())
}
fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn arg<'a>(v: &'a Value, k: &str) -> Option<&'a str> {
    v.get(k).and_then(Value::string).ok().filter(|x| !x.is_empty())
}
fn int(v: &Value, k: &str, default: usize) -> usize {
    v.get(k).and_then(Value::integer).ok().and_then(|x| usize::try_from(x).ok()).unwrap_or(default)
}
fn flag(v: &Value, k: &str) -> bool {
    matches!(v.get(k), Ok(Value::Bool(true)))
}

fn lines_of(root: &Path, file: &str, line: usize) -> Vec<String> {
    std::fs::read_to_string(root.join(file))
        .map(|t| t.lines().enumerate().skip(line.saturating_sub(3)).take(5).map(|(i, l)| format!("{}: {}", i + 1, l.trim_end())).collect())
        .unwrap_or_default()
}

fn append(path: &Path, v: &Value) {
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let line = v.encode().unwrap_or_default();
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| std::io::Write::write_all(&mut f, format!("{line}\n").as_bytes()));
}

/// Log the run, and the errors of the last failing run of the same package that this
/// clean run no longer shows (error with its lines then and now).
fn log_outcome(root: &Path, memory: &Path, r: &BuildRun, before: Option<&BuildRun>) {
    let path = memory.join("build-outcomes.jsonl");
    let errs: Vec<Value> = r
        .errors()
        .iter()
        .map(|d| {
            obj(vec![
                ("code", s(&d.code)),
                ("at", s(&format!("{}:{}", d.file, d.line))),
                ("message", s(&d.message)),
                ("lines", Value::Array(lines_of(root, &d.file, d.line).iter().map(|l| s(l)).collect())),
            ])
        })
        .collect();
    append(
        &path,
        &obj(vec![
            ("at", n(nmtype::now_ms())),
            ("event", s("run")),
            ("run", s(&r.id)),
            ("package", s(r.package.as_deref().unwrap_or("*"))),
            ("kind", s(&r.kind)),
            ("ok", Value::Bool(r.ok)),
            ("errors", Value::Array(errs)),
            ("verified", Value::Bool(true)),
        ]),
    );
    let Some(prev) = before.filter(|p| !p.errors().is_empty() && r.errors().is_empty() && r.ok) else { return };
    for d in prev.errors() {
        append(
            &path,
            &obj(vec![
                ("at", n(nmtype::now_ms())),
                ("event", s("fixed")),
                ("package", s(prev.package.as_deref().unwrap_or("*"))),
                ("from_run", s(&prev.id)),
                ("by_run", s(&r.id)),
                ("code", s(&d.code)),
                ("error_at", s(&format!("{}:{}", d.file, d.line))),
                ("message", s(&d.message)),
                ("lines_now", Value::Array(lines_of(root, &d.file, d.line).iter().map(|l| s(l)).collect())),
            ]),
        );
    }
}

fn keep(r: BuildRun) {
    if let Ok(mut runs) = RUNS.lock() {
        runs.push(r);
        let over = runs.len().saturating_sub(KEEP);
        runs.drain(..over);
    }
}
fn last_of(package: Option<&str>) -> Option<BuildRun> {
    RUNS.lock().ok().and_then(|runs| runs.iter().rev().find(|r| r.package.as_deref() == package).cloned())
}
/// A kept build run by id (the runner reads its raw byte count).
#[must_use]
pub fn by_id(id: &str) -> Option<BuildRun> {
    RUNS.lock().ok().and_then(|runs| runs.iter().find(|r| r.id == id).cloned())
}

fn compact(r: &BuildRun, budget: usize, warnings: bool) -> Value {
    let rows: Vec<Value> = r
        .diags
        .iter()
        .filter(|d| warnings || d.level == "error")
        .take(budget)
        .map(|d| {
            let mut f = vec![("at", s(&format!("{}:{}:{}", d.file, d.line, d.col))), ("code", s(&d.code)), ("message", s(&d.message))];
            if !d.label.is_empty() {
                f.push(("label", s(&d.label)));
            }
            let help: Vec<Value> = d
                .children
                .iter()
                .filter(|c| c.level == "help")
                .take(2)
                .map(|c| {
                    s(&c.replacement
                        .as_ref()
                        .filter(|x| !x.is_empty())
                        .map_or_else(|| c.message.clone(), |x| format!("{} -> `{x}`", c.message)))
                })
                .collect();
            if !help.is_empty() {
                f.push(("help", Value::Array(help)));
            }
            if d.level != "error" {
                f.push(("level", s(&d.level)));
            }
            obj(f)
        })
        .collect();
    let tests: Vec<Value> = r
        .tests
        .iter()
        .take(budget)
        .map(|t| {
            obj(vec![
                ("test", s(&t.name)),
                ("at", s(&t.at.as_ref().map_or_else(String::new, |(f, l)| format!("{f}:{l}")))),
                ("message", s(&t.message)),
            ])
        })
        .collect();
    let mut f = vec![
        ("run", s(&r.id)),
        ("ok", Value::Bool(r.ok)),
        ("errors", n(r.errors().len())),
        ("warnings", n(r.warnings())),
        ("status", s("observed")),
        ("items", Value::Array(rows)),
    ];
    if !tests.is_empty() {
        f.push(("failed_tests", Value::Array(tests)));
    }
    if !r.cargo_errors.is_empty() {
        f.push(("cargo", Value::Array(r.cargo_errors.iter().map(|e| s(e)).collect())));
    }
    if r.timed_out {
        f.push(("timed_out", Value::Bool(true)));
    }
    obj(f)
}

/// Run one build and keep it; logs the outcome under `memory`.
/// # Errors
/// A bad `kind`.
pub fn run_build(home: &Path, root: &Path, memory: &Path, v: &Value) -> Result<BuildRun, ApiError> {
    let kind = Kind::parse(arg(v, "kind").unwrap_or("check")).map_err(|e| ApiError::new("nm-400", e.to_string()))?;
    let package = arg(v, "package").map(str::to_owned);
    let before = last_of(package.as_deref());
    let r = build::run(&Request { root: root.to_path_buf(), target: target(home), kind, package, deadline: run::DEADLINE });
    log_outcome(root, memory, &r, before.as_ref());
    keep(r.clone());
    Ok(r)
}

/// `nm_build_run`: one cargo command for one package; compact cited errors.
/// # Errors
/// A bad `kind`.
pub fn tool_run(home: &Path, root: &Path, memory: &Path, v: &Value) -> Result<Value, ApiError> {
    let r = run_build(home, root, memory, v)?;
    let (budget, warnings) = (int(v, "budget", 12).max(1), flag(v, "warnings"));
    match arg(v, "form").unwrap_or("lines") {
        "json" => Ok(compact(&r, budget, warnings)),
        "lines" => Ok(lines(root, &r, budget, warnings)),
        f => Err(ApiError::new("nm-400", format!("form {f:?}: lines (default) or json"))),
    }
}

/// The `lines` form of a build run (2026-11-13, build output compression; S9 found the two JSON
/// replies 0.999 of plain cargo's text): the counts, then `lines`: one line per error (code, file, line
/// and column, message), the source line itself read fresh from the tree, and rustc's help; a
/// failed test as its name, place and message. Localisation is rustc's primary span, unchanged.
fn lines(root: &Path, r: &BuildRun, budget: usize, warnings: bool) -> Value {
    let mut text = String::new();
    for d in r.diags.iter().filter(|d| warnings || d.level == "error").take(budget) {
        let lvl = if d.level == "error" { String::new() } else { format!("{} ", d.level) };
        let code = if d.code.is_empty() { String::new() } else { format!("{} ", d.code) };
        text.push_str(&format!("{lvl}{code}{}:{}:{} {}\n", d.file, d.line, d.col, d.message));
        if let Some(src) = std::fs::read_to_string(root.join(&d.file))
            .ok()
            .and_then(|t| t.lines().nth(d.line.saturating_sub(1)).map(|l| l.trim().to_owned()))
        {
            text.push_str(&format!("  {}| {src}\n", d.line));
        }
        for c in d.children.iter().filter(|c| c.level == "help").take(1) {
            match c.replacement.as_ref().filter(|x| !x.is_empty()) {
                Some(x) => text.push_str(&format!("  help: {} -> `{x}`\n", c.message)),
                None => text.push_str(&format!("  help: {}\n", c.message)),
            }
        }
    }
    for t in r.tests.iter().take(budget) {
        let at = t.at.as_ref().map_or_else(String::new, |(f, l)| format!(" {f}:{l}"));
        text.push_str(&format!("test {} failed{at}: {}\n", t.name, t.message));
    }
    for e in &r.cargo_errors {
        text.push_str(&format!("cargo: {e}\n"));
    }
    let mut f = vec![("run", s(&r.id)), ("ok", Value::Bool(r.ok)), ("errors", n(r.errors().len())), ("warnings", n(r.warnings()))];
    if r.timed_out {
        f.push(("timed_out", Value::Bool(true)));
    }
    // The next step of the loop (2026-11-14, guiding Claude Code): fix the first error and run
    // again; after a clean check, the tests.
    let next = match (r.ok, r.diags.iter().find(|d| d.level == "error")) {
        (false, Some(d)) => format!("fix the first error at {}:{} and run nm_build_run again", d.file, d.line),
        (false, None) if !r.tests.is_empty() => "fix the first failed test and run nm_build_run with kind test again".to_owned(),
        (false, None) => "read the cargo lines above, fix, and run nm_build_run again".to_owned(),
        (true, _) => "clean: run nm_build_run with kind test before finishing".to_owned(),
    };
    f.push(("next", s(&next)));
    // One string per line: the encoder writes a newline inside a string as six characters.
    if !text.is_empty() {
        f.push(("lines", Value::Array(text.lines().map(s).collect())));
    }
    obj(f)
}

/// `nm_build_explain`: the first error of a kept run (or of a fresh run of `package`),
/// located, with the lines around it, the cause and the fix from exact sources; withholds
/// on a clean build.
/// # Errors
/// An unknown run id, or no run to explain.
pub fn tool_explain(home: &Path, root: &Path, memory: &Path, v: &Value) -> Result<Value, ApiError> {
    let r = match arg(v, "run") {
        Some(id) => {
            by_id(id).ok_or_else(|| ApiError::new("nm-404", format!("no kept run {id}; runs are kept for the last {KEEP} builds")))?
        }
        None if arg(v, "package").is_some() => run_build(home, root, memory, v)?,
        None => last_of(None)
            .or_else(|| RUNS.lock().ok().and_then(|x| x.last().cloned()))
            .ok_or_else(|| ApiError::new("nm-404", "no build run yet: pass `run` or `package`"))?,
    };
    let Some(e) = explain::explain(root, &r, None) else {
        return Ok(obj(vec![("run", s(&r.id)), ("withheld", s("clean build: no error or failed test to explain"))]));
    };
    let around: Vec<Value> = e.around.iter().map(|(i, l)| s(&format!("{}{i}: {l}", if *i == e.line { ">" } else { " " }))).collect();
    Ok(obj(vec![
        ("run", s(&r.id)),
        ("at", s(&format!("{}:{}:{}", e.file, e.line, e.col))),
        ("file", s(&e.file)),
        ("line", n(e.line)),
        ("code", s(&e.code)),
        ("message", s(&e.message)),
        ("cause", s(&e.cause)),
        ("fix", s(&e.fix)),
        ("fix_source", s(e.fix_source.word())),
        ("around", Value::Array(around)),
        ("other_errors", n(e.others)),
    ]))
}

/// Step Check scratch copies kept (2026-11-09): each check copies the whole tree into
/// `scratch/stepcheck-<ms>/`, and 90 left behind by one day of tests filled the D: drive. The
/// newest copies stay for inspection; the shared build folder `scratch/stepcheck-target/` stays.
pub const KEEP_COPIES: usize = 3;

/// Remove all but the `keep` newest `stepcheck-<ms>` copies under `scratch`.
fn prune_copies(scratch: &Path, keep: usize) {
    let Ok(rd) = std::fs::read_dir(scratch) else { return };
    let mut copies: Vec<(u64, PathBuf)> = rd
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let ms = name.strip_prefix("stepcheck-")?.parse::<u64>().ok()?;
            e.path().is_dir().then(|| (ms, e.path()))
        })
        .collect();
    copies.sort_by_key(|c| std::cmp::Reverse(c.0));
    for (_, dir) in copies.into_iter().skip(keep) {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// `nm_stepcheck`: a diff Step Checked on a new scratch copy of the studied tree (copy,
/// apply, check, clippy with warnings denied, tests when asked, size, rules); never applied
/// to the source.
/// # Errors
/// No diff given.
pub fn tool_stepcheck(home: &Path, root: &Path, memory: &Path, v: &Value) -> Result<Value, ApiError> {
    let diff = arg(v, "diff").ok_or_else(|| ApiError::new("nm-400", "`diff` (unified diff against the studied tree) is required"))?;
    let plan = stepcheck::Plan {
        root: root.to_path_buf(),
        scratch: home.join("scratch").join(format!("stepcheck-{}", nmtype::now_ms())),
        target: home.join("scratch/stepcheck-target"),
        diff: diff.to_owned(),
        tests: flag(v, "tests"),
        package: arg(v, "package").map(str::to_owned),
        deadline: run::DEADLINE,
    };
    let st = stepcheck::run(&plan);
    prune_copies(&home.join("scratch"), KEEP_COPIES);
    let pass = st.iter().all(|x| x.ok) && st.iter().any(|x| x.name == "rules");
    // The verdict on Claude's diff is a verified outcome (from the tool's own stages).
    let failed = st.iter().find(|x| !x.ok);
    append(
        &memory.join("build-outcomes.jsonl"),
        &obj(vec![
            ("at", n(nmtype::now_ms())),
            ("event", s("stepcheck")),
            ("diff_sha256", s(&nmhash::sha256_bytes(diff.as_bytes()))),
            (
                "files",
                Value::Array(
                    stepcheck::added_lines(diff)
                        .iter()
                        .map(|(f, _)| f.clone())
                        .collect::<std::collections::BTreeSet<_>>()
                        .iter()
                        .map(|f| s(f))
                        .collect(),
                ),
            ),
            ("verdict", s(if pass { "pass" } else { "fail" })),
            ("failed_stage", s(failed.map_or("", |x| x.name))),
            ("detail", s(failed.map_or("", |x| x.detail.as_str()))),
            // M2 (2026-11-09): the lines the diff added, so a failed check and the later passing one
            // on the same files become a fix record (`nmsubs::Substrate::load_operational`).
            (
                "added",
                Value::Array(
                    stepcheck::added_lines(diff)
                        .iter()
                        .take(6)
                        .map(|(f, l)| s(&format!("{f}: {}", l.trim().chars().take(160).collect::<String>())))
                        .collect(),
                ),
            ),
            ("verified", Value::Bool(true)),
        ]),
    );
    // 2026-11-12 (motor, own changes by content): on a pass, each file as it will be once the diff
    // is applied (the scratch copy), by SHA-256; the caller records them as predictions.
    let predicted: Vec<Value> = if pass {
        stepcheck::added_lines(diff)
            .iter()
            .map(|(f, _)| f.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .iter()
            .filter_map(|f| {
                std::fs::read(plan.scratch.join(f)).ok().map(|b| obj(vec![("file", s(f)), ("sha256", s(&nmhash::sha256_bytes(&b)))]))
            })
            .collect()
    } else {
        Vec::new()
    };
    Ok(obj(vec![
        ("verdict", s(if pass { "pass" } else { "fail" })),
        ("predicted", Value::Array(predicted)),
        ("scratch", s(&plan.scratch.strip_prefix(home).unwrap_or(&plan.scratch).to_string_lossy().replace('\\', "/"))),
        ("applied_to_source", Value::Bool(false)),
        (
            "stages",
            Value::Array(
                st.iter()
                    .map(|x| obj(vec![("name", s(x.name)), ("ok", Value::Bool(x.ok)), ("ms", n(x.ms)), ("detail", s(&x.detail))]))
                    .collect(),
            ),
        ),
    ]))
}

/// Findings of the last `nm_improve_scan`, for `nm_patch_propose` by index.
static FINDINGS: Mutex<Vec<nmtool::improve::Finding>> = Mutex::new(Vec::new());

/// `nm_improve_scan`: rule words, oversize files and, for a named package, clippy's
/// machine-applicable suggestions, from exact sources; nothing is applied.
/// # Errors
/// None today (the shape keeps room for refusals).
pub fn tool_improve_scan(home: &Path, root: &Path, v: &Value) -> Result<Value, ApiError> {
    let mut found = nmtool::improve::scan(root, arg(v, "filter").unwrap_or(""));
    if let Some(p) = arg(v, "package") {
        found.extend(nmtool::improve::clippy(root, &target(home), p));
    }
    let budget = int(v, "budget", 20).max(1);
    let rows: Vec<Value> = found
        .iter()
        .take(budget)
        .enumerate()
        .map(|(i, f)| {
            obj(vec![
                ("index", n(i)),
                ("kind", s(&f.kind)),
                ("at", s(&format!("{}:{}", f.file, f.line))),
                ("message", s(&f.message)),
                ("fix", Value::Bool(f.fix.is_some())),
            ])
        })
        .collect();
    let total = found.len();
    if let Ok(mut k) = FINDINGS.lock() {
        *k = found;
    }
    Ok(obj(vec![
        ("findings", n(total)),
        ("items", Value::Array(rows)),
        ("next", s("nm_patch_propose with an index, then nm_stepcheck; Claude Code applies")),
    ]))
}

/// `nm_patch_propose`: the unified diff for one finding of the last scan, recorded in the
/// approval queue; never applied by the mini.
/// # Errors
/// No scan yet, an index out of range, or a finding without a one-line fix.
pub fn tool_patch_propose(memory: &Path, v: &Value) -> Result<Value, ApiError> {
    let i = int(v, "index", usize::MAX);
    let f = FINDINGS
        .lock()
        .ok()
        .and_then(|k| k.get(i).cloned())
        .ok_or_else(|| ApiError::new("nm-404", "no such finding; run nm_improve_scan first"))?;
    let diff = nmtool::improve::propose(&f).ok_or_else(|| ApiError::new("nm-409", format!("{}:{} has no one-line fix", f.file, f.line)))?;
    let id = nmtool::approval::record(memory, &format!("{} {}:{} {}", f.kind, f.file, f.line, f.message), &diff)
        .map_err(|e| ApiError::new("nm-500", e))?;
    Ok(obj(vec![
        ("id", s(&id)),
        ("diff", s(&diff)),
        ("applied", Value::Bool(false)),
        ("next", s("nm_stepcheck with this diff; Claude Code applies it after a pass")),
    ]))
}

/// Self-check of the `build` sub-component's copy pruning: five copies become the three newest,
/// and a folder that is not a copy (the shared build folder) stays.
pub fn check() -> Result<(), String> {
    let dir = nmtype::check_dir("nmapi-build").map_err(|e| e.to_string())?;
    for ms in [10, 20, 30, 40, 50] {
        std::fs::create_dir_all(dir.join(format!("stepcheck-{ms}"))).map_err(|e| e.to_string())?;
    }
    std::fs::create_dir_all(dir.join("stepcheck-target")).map_err(|e| e.to_string())?;
    prune_copies(&dir, 3);
    let left: Vec<String> =
        std::fs::read_dir(&dir).map_err(|e| e.to_string())?.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    let mut left = left;
    left.sort();
    if left != ["stepcheck-30", "stepcheck-40", "stepcheck-50", "stepcheck-target"] {
        return Err(format!("nmapi.build: prune left {left:?}"));
    }
    Ok(())
}
