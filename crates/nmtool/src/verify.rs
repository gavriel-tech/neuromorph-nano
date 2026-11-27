//! Verification dimensions of a proposed diff (PLAN 4c `verify`, 2026-11-10): what a change
//! touches, read from the diff and the tree, before anything runs. Functional and coverage:
//! the tests that name a changed item (or none); interface: public signatures added or removed;
//! dependency: manifest dependency lines changed; architecture: rule words, `unsafe` and
//! `unwrap` in added lines. Runtime, performance, security and regression cannot be read from
//! the text and come back as open questions, never as a verdict. Nothing is applied.
use nmjson::Value;

/// One finding: its dimension, whether it is a fact read from the text or an open question,
/// the file and line it bears on, and what it says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Finding {
    pub dimension: &'static str,
    pub open: bool,
    pub at: String,
    pub text: String,
}

/// Changed lines of a unified diff: (file, added or removed, text).
fn changes(diff: &str) -> Vec<(String, bool, String)> {
    let mut file = String::new();
    let mut out = Vec::new();
    for l in diff.lines() {
        if let Some(h) = l.strip_prefix("+++ ") {
            file = h.trim().trim_start_matches("b/").to_owned();
        } else if l.starts_with("--- ") {
            continue;
        } else if let Some(a) = l.strip_prefix('+') {
            out.push((file.clone(), true, a.to_owned()));
        } else if let Some(r) = l.strip_prefix('-') {
            out.push((file.clone(), false, r.to_owned()));
        }
    }
    out
}

/// The item name a line defines (`fn x`, `struct X`, `enum X`, `trait X`, `const X`), if any.
fn item_name(line: &str) -> Option<String> {
    let t = line.trim_start().trim_start_matches("pub(crate) ").trim_start_matches("pub ");
    for kw in ["fn ", "struct ", "enum ", "trait ", "const ", "static ", "type "] {
        if let Some(rest) = t.strip_prefix(kw) {
            let name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
            return (!name.is_empty()).then_some(name);
        }
    }
    None
}

/// The verification dimensions of `diff` over a tree given as (path, text) pairs.
#[must_use]
pub fn verify(diff: &str, files: &[(String, String)]) -> Vec<Finding> {
    let ch = changes(diff);
    let mut out = Vec::new();
    let mut names: Vec<String> = Vec::new();
    for (file, added, text) in &ch {
        if let Some(n) = item_name(text)
            && !names.contains(&n)
        {
            names.push(n);
        }
        let t = text.trim_start();
        if t.starts_with("pub ") && item_name(text).is_some() {
            out.push(Finding {
                dimension: "interface",
                open: false,
                at: file.clone(),
                text: format!("public item {}: {}", if *added { "added or changed" } else { "removed or changed" }, t.trim()),
            });
        }
        if file.ends_with("Cargo.toml") && (t.contains("path =") || t.contains("version =") || t.contains("workspace = true")) {
            out.push(Finding {
                dimension: "dependency",
                open: false,
                at: file.clone(),
                text: format!("manifest line {}: {}", if *added { "added" } else { "removed" }, t.trim()),
            });
        }
        if *added {
            if t.contains('\u{2014}') {
                out.push(Finding {
                    dimension: "architecture",
                    open: false,
                    at: file.clone(),
                    text: "an em dash in an added line (rules.md)".into(),
                });
            }
            if t.contains("unsafe ") || t.starts_with("unsafe") {
                out.push(Finding {
                    dimension: "architecture",
                    open: false,
                    at: file.clone(),
                    text: format!("unsafe in an added line (allowed only in nmffi): {}", t.trim()),
                });
            }
            if t.contains(".unwrap()") && !file.contains("/tests/") {
                out.push(Finding {
                    dimension: "architecture",
                    open: false,
                    at: file.clone(),
                    text: format!("unwrap in an added line outside tests: {}", t.trim()),
                });
            }
        }
    }
    for n in &names {
        let tests: Vec<String> = files
            .iter()
            .filter(|(p, t)| {
                let lines: Vec<&str> = t.lines().collect();
                let start = if p.contains("/tests/") {
                    0
                } else {
                    lines.iter().position(|l| l.contains("#[cfg(test)]") || l.contains("fn check(")).unwrap_or(lines.len())
                };
                lines.iter().skip(start).any(|l| l.contains(n.as_str()))
            })
            .map(|(p, _)| p.clone())
            .collect();
        if tests.is_empty() {
            out.push(Finding {
                dimension: "coverage",
                open: true,
                at: n.clone(),
                text: format!("no test or section check names `{n}`: what proves the change works?"),
            });
        } else {
            out.push(Finding {
                dimension: "functional",
                open: false,
                at: n.clone(),
                text: format!("`{n}` is named by tests in {}", tests.join(", ")),
            });
        }
    }
    for (d, q) in [
        ("runtime", "does the change behave the same at run time (inputs, errors, panics)? run the section checks and tests"),
        ("performance", "is the change as fast as before? time the affected path before and after"),
        ("security", "does the change read or write anything new outside the folder, or take untrusted input?"),
        ("regression", "do the capability gates and suites still pass after the change?"),
    ] {
        out.push(Finding { dimension: d, open: true, at: String::new(), text: q.into() });
    }
    out
}

/// The findings as JSON for the MCP tool.
#[must_use]
pub fn to_value(f: &[Finding]) -> Value {
    Value::Array(
        f.iter()
            .map(|x| {
                Value::Object(
                    [
                        ("dimension", Value::String(x.dimension.into())),
                        ("open", Value::Bool(x.open)),
                        ("at", Value::String(x.at.clone())),
                        ("text", Value::String(x.text.clone())),
                    ]
                    .into_iter()
                    .map(|(k, v)| (k.to_owned(), v))
                    .collect(),
                )
            })
            .collect(),
    )
}

/// Self-check of the `verify` sub-component: a changed public function with a test that names
/// it, an added unwrap, a manifest edge, and the open dimensions.
pub fn check() -> Result<(), String> {
    let diff = "--- a/a/src/x.rs\n+++ b/a/src/x.rs\n@@ -1,1 +1,2 @@\n-pub fn go(a: u8) {}\n+pub fn go(a: u16) { let _v = Some(1).unwrap(); }\n--- a/a/Cargo.toml\n+++ b/a/Cargo.toml\n@@ -1 +1,2 @@\n+nmjson = { path = \"../nmjson\" }\n";
    let files = vec![("a/src/x.rs".to_owned(), "pub fn go(a: u8) {}\nfn check() {\n    go(1);\n}\n".to_owned())];
    let f = verify(diff, &files);
    let has = |d: &str, open: bool| f.iter().any(|x| x.dimension == d && x.open == open);
    if !(has("interface", false)
        && has("dependency", false)
        && has("architecture", false)
        && has("functional", false)
        && has("security", true))
    {
        return Err(format!("nmtool.verify: {f:?}"));
    }
    Ok(())
}
