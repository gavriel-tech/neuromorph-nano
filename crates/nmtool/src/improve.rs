//! Code improvement findings and proposals (guide 17.5; PLAN 4c `improve`). Exact sources only:
//! rule words (with their fixed line), Rust files over the line limit (no fix), and clippy's
//! machine-applicable suggestions (one-line spans). A proposal is a unified diff for one finding;
//! nothing is ever applied here: Claude Code applies a diff after its Step Check (law: the
//! Neuromorph never writes its own source). Body first written in the earlier `nmcore::improve`
//! pass (kept there).
use crate::build::{self, BuildRun, Kind, Request};
use crate::{lex, stepcheck};
use std::path::{Path, PathBuf};

/// One finding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Finding {
    /// rule, oversize, clippy.
    pub kind: String,
    pub file: String,
    pub line: usize,
    pub message: String,
    /// (old line, new line) when a one-line fix is known.
    pub fix: Option<(String, String)>,
}

fn rust_files(root: &Path, dir: &Path, filter: &str, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut es: Vec<_> = rd.flatten().collect();
    es.sort_by_key(std::fs::DirEntry::file_name);
    for e in es {
        let p = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        if p.is_dir() {
            if !name.starts_with('.')
                && !["target", "scratch", "runs", "weights", "golden", "node_modules", "tests"].contains(&name.as_str())
            {
                rust_files(root, &p, filter, out);
            }
        } else if name.ends_with(".rs") && p.strip_prefix(root).is_ok_and(|r| r.to_string_lossy().replace('\\', "/").contains(filter)) {
            out.push(p);
        }
    }
}

/// Scan the Rust files under `root` whose relative path contains `filter`.
#[must_use]
pub fn scan(root: &Path, filter: &str) -> Vec<Finding> {
    let mut files = Vec::new();
    rust_files(root, root, filter, &mut files);
    let mut out = Vec::new();
    for p in files {
        let rel = build::relative(root, &p.to_string_lossy());
        let Ok(text) = std::fs::read_to_string(&p) else { continue };
        for (line, word) in lex::rule_hits(&text) {
            let old = text.lines().nth(line - 1).unwrap_or_default().to_owned();
            out.push(Finding {
                kind: "rule".into(),
                file: rel.clone(),
                line,
                message: format!("rule word {word:?}"),
                fix: Some((old.clone(), lex::fix_line(&old))),
            });
        }
        let n = text.lines().count();
        if n > stepcheck::MAX_LINES {
            out.push(Finding {
                kind: "oversize".into(),
                file: rel,
                line: 1,
                message: format!("{n} lines (limit {})", stepcheck::MAX_LINES),
                fix: None,
            });
        }
    }
    out
}

/// Machine-applicable clippy suggestions of a build run, as findings (one-line spans only;
/// files read under `root`).
#[must_use]
pub fn from_clippy(root: &Path, run: &BuildRun) -> Vec<Finding> {
    let mut out = Vec::new();
    for d in &run.diags {
        for c in d.children.iter().filter(|c| c.applicability == "MachineApplicable") {
            let (Some(rep), Some((file, line, a, b))) = (&c.replacement, &c.at) else { continue };
            let Ok(text) = std::fs::read_to_string(root.join(file)) else { continue };
            let Some(old) = text.lines().nth(line.saturating_sub(1)) else { continue };
            let chars: Vec<char> = old.chars().collect();
            if *a == 0 || *b < *a || *b - 1 > chars.len() {
                continue;
            }
            let new: String =
                chars.iter().take(a - 1).chain(rep.chars().collect::<Vec<_>>().iter()).chain(chars.iter().skip(b - 1)).collect();
            out.push(Finding {
                kind: "clippy".into(),
                file: file.clone(),
                line: *line,
                message: format!("{}: {}", d.code, d.message),
                fix: Some((old.to_owned(), new)),
            });
        }
    }
    out
}

/// Run clippy for one package and collect its machine-applicable suggestions.
#[must_use]
pub fn clippy(root: &Path, target: &Path, package: &str) -> Vec<Finding> {
    let r = build::run(&Request {
        root: root.to_path_buf(),
        target: target.to_path_buf(),
        kind: Kind::Clippy,
        package: Some(package.to_owned()),
        deadline: crate::run::DEADLINE,
    });
    from_clippy(root, &r)
}

/// A unified diff replacing one line, for `nm_stepcheck` and then Claude Code.
#[must_use]
pub fn propose(f: &Finding) -> Option<String> {
    let (old, new) = f.fix.as_ref()?;
    if old == new {
        return None;
    }
    Some(format!("--- a/{0}\n+++ b/{0}\n@@ -{1},1 +{1},1 @@\n-{old}\n+{new}\n", f.file, f.line))
}

/// Section check: a rule word is found with its fixed line, the proposal applies on a scratch
/// copy, and a clippy suggestion span becomes a one-line fix.
/// # Errors
/// One line naming what failed.
pub fn check() -> Result<(), String> {
    use crate::build::{Child, Diagnostic};
    let dir = crate::scratch_dir("improve")?;
    std::fs::create_dir_all(dir.join("src")).map_err(|e| e.to_string())?;
    let dash = '\u{2014}';
    std::fs::write(dir.join("src/lib.rs"), format!("//! Probe {dash} line.\npub fn f(v: &[u8]) -> bool {{\n    v.len() == 0\n}}\n"))
        .map_err(|e| e.to_string())?;
    let found = scan(&dir, "src");
    let rule = found.iter().find(|f| f.kind == "rule").ok_or("rule word not found")?;
    let diff = propose(rule).ok_or("no proposal")?;
    let copy = dir.join("copy");
    stepcheck::copy_tree(&dir.join("src"), &copy.join("src")).map_err(|e| e.to_string())?;
    stepcheck::apply(&copy, &diff)?;
    let fixed = std::fs::read_to_string(copy.join("src/lib.rs")).map_err(|e| e.to_string())?;
    if fixed.contains(dash)
        || std::fs::read_to_string(dir.join("src/lib.rs")).map_err(|e| e.to_string())?.lines().next() == fixed.lines().next()
    {
        return Err("the proposal did not fix the copy, or touched the source".into());
    }
    let child = Child {
        level: "help".into(),
        message: "use is_empty".into(),
        replacement: Some("v.is_empty()".into()),
        applicability: "MachineApplicable".into(),
        at: Some(("src/lib.rs".into(), 3, 5, 17)),
    };
    let run = BuildRun {
        diags: vec![Diagnostic {
            level: "warning".into(),
            code: "clippy::len_zero".into(),
            message: "length comparison to zero".into(),
            file: "src/lib.rs".into(),
            line: 3,
            col: 5,
            label: String::new(),
            children: vec![child],
        }],
        ..BuildRun::default()
    };
    let c = from_clippy(&dir, &run);
    if c.first().and_then(|f| f.fix.as_ref()).map(|x| x.1.as_str()) != Some("    v.is_empty()") {
        return Err(format!("clippy span to fix: {c:?}"));
    }
    Ok(())
}
