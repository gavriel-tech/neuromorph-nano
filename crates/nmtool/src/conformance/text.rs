//! Text helpers shared by the conformance detectors: directory walks inside the mini, relative
//! paths, comment lines, whole-word search, string literals, and parameter reads from the rule
//! data. Exact text only; nothing here interprets prose.
use nmjson::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Folder names never walked: build output, disposable work, version control, retired code.
const SKIP: [&str; 6] = ["target", "scratch", ".git", "node_modules", "retired", "backup"];

/// Every file under `dir` whose extension is one of `exts` (all files when `exts` is empty),
/// sorted, skipping the folders in `SKIP`.
pub(super) fn walk(dir: &Path, exts: &[String], out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut es: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    es.sort();
    for p in es {
        let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if p.is_dir() {
            if !SKIP.contains(&name.as_str()) {
                walk(&p, exts, out);
            }
        } else if exts.is_empty() || p.extension().is_some_and(|x| exts.iter().any(|e| x.eq_ignore_ascii_case(e.as_str()))) {
            out.push(p);
        }
    }
}

/// Every Rust file under `crates/*/src` of `root`.
pub(super) fn crate_sources(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(root.join("crates")) else { return out };
    let mut crates: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    crates.sort();
    for c in crates {
        walk(&c.join("src"), &["rs".to_owned()], &mut out);
    }
    out
}

/// `p` relative to `root` with forward slashes (the full path when it is outside).
pub(super) fn rel(root: &Path, p: &Path) -> String {
    p.strip_prefix(root).map_or_else(|_| p.display().to_string(), |r| r.to_string_lossy().replace('\\', "/"))
}

/// The file's text, empty when unreadable.
pub(super) fn read(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap_or_default()
}

/// A line that is only a comment (line, doc or block comment continuation).
pub(super) fn is_comment(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("//") || t.starts_with("/*") || t.starts_with("* ") || t == "*"
}

fn ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Does `hay` contain `w` as a whole identifier?
pub(super) fn has_word(hay: &str, w: &str) -> bool {
    if w.is_empty() {
        return false;
    }
    let mut from = 0;
    while let Some(i) = hay[from..].find(w) {
        let a = from + i;
        let b = a + w.len();
        let before = hay[..a].chars().next_back().is_none_or(|c| !ident(c));
        let after = hay[b..].chars().next().is_none_or(|c| !ident(c));
        if before && after {
            return true;
        }
        from = b;
    }
    false
}

/// The double-quoted literals of one line (simple split; escaped quotes are rare in the
/// lines this reads).
pub(super) fn literals(line: &str) -> Vec<&str> {
    line.split('"').skip(1).step_by(2).collect()
}

/// A required string parameter.
pub(super) fn s(v: &Value, k: &str) -> Result<String, String> {
    v.get(k).and_then(Value::string).map(str::to_owned).map_err(|e| format!("parameter {k}: {e}"))
}

/// An optional string parameter (empty when missing).
pub(super) fn so(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default()
}

/// An array parameter (empty when missing).
pub(super) fn list<'a>(v: &'a Value, k: &str) -> &'a [Value] {
    v.get(k).and_then(Value::array).unwrap_or_default()
}

/// A list of strings parameter (non-strings skipped).
pub(super) fn strs(v: &Value, k: &str) -> Vec<String> {
    list(v, k).iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect()
}

/// Engine assemblies (files under `dir` that bind engine components or extend another
/// assembly) named by a string literal in the code under `paths`: name to the first file.
pub(super) fn served_assemblies(root: &Path, dir: &Path, paths: &[String]) -> BTreeMap<String, String> {
    let mut bound = BTreeMap::new();
    for sp in paths {
        let mut files = Vec::new();
        walk(&root.join(sp), &["rs".to_owned()], &mut files);
        for f in files {
            let text = read(&f);
            for line in text.lines().filter(|l| !is_comment(l)) {
                for lit in literals(line) {
                    if lit.is_empty() || lit.len() > 80 || lit.contains(['/', '\\', ' ', '{']) || bound.contains_key(lit) {
                        continue;
                    }
                    let file = dir.join(format!("{lit}.json"));
                    let t = read(&file);
                    if t.contains("nmengn.components#") || t.contains("\"base\"") {
                        bound.insert(lit.to_owned(), rel(root, &f));
                    }
                }
            }
        }
    }
    bound
}
