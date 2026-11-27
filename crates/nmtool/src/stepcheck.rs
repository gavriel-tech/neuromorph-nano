//! Step Check for code changes (guide 17.3, 17.5): a proposed unified diff is applied to a
//! NEW scratch copy (never the source), then check, clippy with warnings denied, tests when
//! asked, size and rules. Body first written in the earlier `nmcore::stepcheck` pass (kept
//! there). Changes here: the scratch folder is new per check and never deleted by the check
//! (law 13); every cargo call has a deadline; line endings are kept; size fails only when
//! the diff takes a file past the line limit; rules look at added lines only.
use crate::{lex, run};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Lines past which a Rust file is flagged (PLAN 4c `improve`).
pub const MAX_LINES: usize = 400;
/// Folders never copied into a scratch tree (build output, state, weights, VCS).
const SKIP: &[&str] =
    &["target", "scratch", "runs", "weights", "golden", "node_modules", "9-weight", "memory", "units", "tests", "private", "retired"];

/// One stage result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage {
    /// copy, apply, check, clippy, test, size, rules.
    pub name: &'static str,
    pub ok: bool,
    pub ms: u64,
    pub detail: String,
}

/// What to check.
#[derive(Clone, Debug)]
pub struct Plan {
    /// The tree the diff is written against (read only).
    pub root: PathBuf,
    /// A new folder for the copy; refused when it exists.
    pub scratch: PathBuf,
    /// Cargo target folder for the copy.
    pub target: PathBuf,
    pub diff: String,
    pub tests: bool,
    pub package: Option<String>,
    pub deadline: Duration,
}

/// Copy a source tree into `dst`, skipping build output, state, weights and dot folders
/// (except `.cargo`).
/// # Errors
/// Disk failure.
pub fn copy_tree(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for e in std::fs::read_dir(src)?.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let p = e.path();
        if p.is_dir() {
            if (name.starts_with('.') && name != ".cargo") || SKIP.contains(&name.as_str()) {
                continue;
            }
            copy_tree(&p, &dst.join(&name))?;
        } else {
            std::fs::copy(&p, dst.join(&name))?;
        }
    }
    Ok(())
}

fn rel_path(header: &str) -> String {
    let p = header.trim().split('\t').next().unwrap_or_default();
    p.strip_prefix("b/").or_else(|| p.strip_prefix("a/")).unwrap_or(p).to_owned()
}

/// Files a diff changes and the lines it adds: (file, added line).
#[must_use]
pub fn added_lines(diff: &str) -> Vec<(String, String)> {
    let mut file = String::new();
    let mut out = Vec::new();
    for l in diff.lines() {
        if let Some(h) = l.strip_prefix("+++ ") {
            file = rel_path(h);
        } else if let Some(a) = l.strip_prefix('+') {
            out.push((file.clone(), a.to_owned()));
        }
    }
    out
}

/// Apply a unified diff under `root`. Hunks must match their context exactly; files keep
/// their line endings.
/// # Errors
/// The first hunk or file that does not apply.
pub fn apply(root: &Path, diff: &str) -> Result<Vec<String>, String> {
    struct Open {
        path: PathBuf,
        body: Vec<String>,
        crlf: bool,
        offset: isize,
    }
    fn flush(f: &mut Option<Open>) -> Result<(), String> {
        if let Some(o) = f.take() {
            let nl = if o.crlf { "\r\n" } else { "\n" };
            std::fs::write(&o.path, o.body.join(nl) + nl).map_err(|e| e.to_string())?;
            // exFAT keeps whole seconds: a file written in the same second as cargo's last build
            // of this package (the scratch copies share one target) would look unchanged, and the
            // Step Check would pass a diff it never compiled. The time is set 2 s ahead.
            std::fs::File::options()
                .write(true)
                .open(&o.path)
                .and_then(|h| h.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(2)))
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    let mut changed = Vec::new();
    let mut file: Option<Open> = None;
    let lines: Vec<&str> = diff.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let l = lines.get(i).copied().unwrap_or_default();
        if let Some(h) = l.strip_prefix("+++ ") {
            flush(&mut file)?;
            let rel = rel_path(h);
            if rel == "/dev/null" || rel.contains("..") {
                return Err(format!("{rel}: only edits to existing files inside the tree are applied"));
            }
            let path = root.join(&rel);
            let text = std::fs::read_to_string(&path).map_err(|e| format!("{rel}: {e}"))?;
            file = Some(Open { path, crlf: text.contains("\r\n"), body: text.lines().map(str::to_owned).collect(), offset: 0 });
            changed.push(rel);
        } else if let Some(h) = l.strip_prefix("@@ -") {
            let start: usize = h.split([',', ' ']).next().and_then(|x| x.parse().ok()).ok_or("bad hunk header")?;
            let (mut old, mut new) = (Vec::new(), Vec::new());
            i += 1;
            while let Some(t) = lines.get(i).copied() {
                if t.starts_with("@@") || t.starts_with("--- ") {
                    break;
                }
                let rest = t.get(1..).unwrap_or_default().to_owned();
                match t.chars().next() {
                    Some('-') => old.push(rest),
                    Some('+') => new.push(rest),
                    Some(' ') | None => {
                        old.push(rest.clone());
                        new.push(rest);
                    }
                    _ => {}
                }
                i += 1;
            }
            let o = file.as_mut().ok_or("hunk before file header")?;
            let at = usize::try_from(isize::try_from(start.saturating_sub(1)).unwrap_or(0) + o.offset).unwrap_or(0);
            if o.body.get(at..at + old.len()) != Some(&old[..]) {
                return Err(format!("hunk at line {start} does not match the file"));
            }
            o.offset += isize::try_from(new.len()).unwrap_or(0) - isize::try_from(old.len()).unwrap_or(0);
            o.body.splice(at..at + old.len(), new);
            continue;
        }
        i += 1;
    }
    flush(&mut file)?;
    Ok(changed)
}

/// The rules stage on its own: added lines with a rule word or a new `unsafe` outside nmffi.
#[must_use]
pub fn rules_of(diff: &str) -> Vec<String> {
    let mut bad = Vec::new();
    for (f, l) in added_lines(diff) {
        if let Some((_, w)) = lex::rule_hits(&l).first() {
            bad.push(format!("{f}: rule word {w:?}"));
        }
        if !lex::unsafe_outside_ffi(&f, &l).is_empty() {
            bad.push(format!("{f}: unsafe outside nmffi"));
        }
    }
    bad
}

fn ms_since(t: std::time::Instant) -> u64 {
    u64::try_from(t.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// Run the Step Check of a diff on a new copy of `plan.root` in `plan.scratch`; stops at
/// the first failing stage.
#[must_use]
pub fn run(plan: &Plan) -> Vec<Stage> {
    let mut st = Vec::new();
    let t = std::time::Instant::now();
    if plan.scratch.exists() {
        st.push(Stage {
            name: "copy",
            ok: false,
            ms: 0,
            detail: format!("{} exists; a Step Check never reuses a scratch folder", plan.scratch.display()),
        });
        return st;
    }
    if let Err(e) = copy_tree(&plan.root, &plan.scratch) {
        st.push(Stage { name: "copy", ok: false, ms: ms_since(t), detail: format!("copy failed: {e}") });
        return st;
    }
    st.push(Stage { name: "copy", ok: true, ms: ms_since(t), detail: plan.scratch.display().to_string() });
    let t = std::time::Instant::now();
    let changed = match apply(&plan.scratch, &plan.diff) {
        Ok(c) => {
            st.push(Stage { name: "apply", ok: true, ms: ms_since(t), detail: c.join(", ") });
            c
        }
        Err(e) => {
            st.push(Stage { name: "apply", ok: false, ms: ms_since(t), detail: e });
            return st;
        }
    };
    let td = plan.target.to_string_lossy().into_owned();
    let env = [("CARGO_TARGET_DIR", td.as_str()), ("CARGO_INCREMENTAL", "0"), ("CARGO_TERM_COLOR", "never")];
    let mut pkg: Vec<&str> = Vec::new();
    if let Some(p) = &plan.package {
        pkg.extend(["-p", p.as_str()]);
    }
    let mut cmds: Vec<(&'static str, Vec<&str>)> = vec![
        ("check", [vec!["check", "--offline", "--all-targets"], pkg.clone()].concat()),
        ("clippy", [vec!["clippy", "--offline", "--all-targets"], pkg.clone(), vec!["--", "-D", "warnings"]].concat()),
    ];
    if plan.tests {
        cmds.push(("test", [vec!["test", "--offline"], pkg].concat()));
    }
    for (name, args) in cmds {
        let o = run::run(&plan.scratch, "cargo", &args, &env, plan.deadline);
        let mut detail = o.stderr.lines().filter(|l| l.starts_with("error") || l.contains("-->")).take(4).collect::<Vec<_>>().join(" | ");
        if o.timed_out {
            detail = format!("killed at the {} s deadline; {detail}", plan.deadline.as_secs());
        }
        st.push(Stage { name, ok: o.ok, ms: o.ms, detail });
        if !o.ok {
            return st;
        }
    }
    let t = std::time::Instant::now();
    let big: Vec<String> = changed
        .iter()
        .filter(|c| c.ends_with(".rs"))
        .filter(|c| {
            let lines = |p: &Path| std::fs::read_to_string(p).map(|s| s.lines().count()).unwrap_or(0);
            lines(&plan.root.join(c)) <= MAX_LINES && lines(&plan.scratch.join(c)) > MAX_LINES
        })
        .cloned()
        .collect();
    st.push(Stage { name: "size", ok: big.is_empty(), ms: ms_since(t), detail: big.join(", ") });
    let bad = rules_of(&plan.diff);
    st.push(Stage { name: "rules", ok: bad.is_empty(), ms: 0, detail: bad.join("; ") });
    st
}

/// Section check: a two-line package in its own workspace under `scratch/checks/`; a good
/// diff passes copy, apply, check, clippy, size and rules; a stale diff fails at apply; a
/// diff adding a rule word is caught by the rules stage.
/// # Errors
/// One line naming what failed.
pub fn check() -> Result<(), String> {
    let base = crate::scratch_dir("stepcheck")?;
    let src = base.join("src-tree");
    std::fs::create_dir_all(src.join("src")).map_err(|e| e.to_string())?;
    std::fs::write(src.join("Cargo.toml"), "[package]\nname = \"nmtool-probe\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[workspace]\n")
        .map_err(|e| e.to_string())?;
    std::fs::write(src.join("src/lib.rs"), "//! Probe.\n/// One.\n#[must_use]\npub const fn one() -> u32 {\n    1\n}\n")
        .map_err(|e| e.to_string())?;
    let good = "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -4,3 +4,3 @@\n pub const fn one() -> u32 {\n-    1\n+    2 - 1\n }\n";
    let target = base.join("target");
    let plan = |name: &str, diff: &str| Plan {
        root: src.clone(),
        scratch: base.join(name),
        target: target.clone(),
        diff: diff.into(),
        tests: false,
        package: None,
        deadline: run::DEADLINE,
    };
    let st = run(&plan("good", good));
    let names: Vec<&str> = st.iter().map(|s| s.name).collect();
    if !st.iter().all(|s| s.ok) || names != ["copy", "apply", "check", "clippy", "size", "rules"] {
        return Err(format!("good diff stages {st:?}"));
    }
    let after = std::fs::read_to_string(base.join("good/src/lib.rs")).map_err(|e| e.to_string())?;
    let source = std::fs::read_to_string(src.join("src/lib.rs")).map_err(|e| e.to_string())?;
    if !after.contains("2 - 1") || source.contains("2 - 1") {
        return Err("the diff must change the copy and never the source".into());
    }
    let broken = run(&plan("broken", &good.replace("+    2 - 1", "+    \"one\"")));
    if broken.last().map(|s| (s.name, s.ok)) != Some(("check", false)) {
        return Err(format!("a diff that cannot compile must fail at check right after a good one: {broken:?}"));
    }
    let stale = run(&plan("stale", &good.replace("    1\n+", "    7\n+")));
    if stale.last().map(|s| (s.name, s.ok)) != Some(("apply", false)) {
        return Err(format!("stale diff stages {stale:?}"));
    }
    if run(&plan("good", good)).first().map(|s| s.ok) != Some(false) {
        return Err("a reused scratch folder must be refused".into());
    }
    let worded = format!("+++ b/src/lib.rs\n+// the {} way\n", ["spec", "ialist"].concat());
    if rules_of(&worded).is_empty() {
        return Err("rules stage missed a rule word".into());
    }
    Ok(())
}
