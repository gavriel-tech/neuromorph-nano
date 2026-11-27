//! Definitions as served units: what belongs to a definition above its first line (doc
//! comments, attributes, decorators), and the size a served definition is kept within. The
//! structural view serves a definition from its doc comments to its last line (`views`), cut at
//! `WHOLE` budgets; the budget is derived at run time from the tree (`budget`), so a served
//! definition carries about as much text as a line window.

/// A definition up to this many budgets is served whole; a larger one is served from its doc
/// comments and signature into its body up to this size.
pub const WHOLE: usize = 2;

/// True when a line belongs to the definition below it: a doc or plain comment, an attribute, a
/// decorator, or a block comment's lines (`#` comments where the language uses them).
#[must_use]
pub fn leads(line: &str, hash_comments: bool) -> bool {
    let t = line.trim_start();
    if t.is_empty() {
        return false;
    }
    t.starts_with("///")
        || (t.starts_with("//") && !t.starts_with("//!"))
        || t.starts_with("#[")
        || t.starts_with('@')
        || t.starts_with("/*")
        || t.starts_with('*')
        || (hash_comments && t.starts_with('#') && !t.starts_with("#!"))
}

/// The size budget for a tree: the median non-whitespace size of its code line windows
/// (`windows`), so a served definition and a window carry about the same text. `fallback` when
/// there are none.
#[must_use]
pub fn budget(windows: &[usize], fallback: usize) -> usize {
    let mut w: Vec<usize> = windows.iter().copied().filter(|x| *x > 0).collect();
    if w.is_empty() {
        return fallback;
    }
    w.sort_unstable();
    w.get(w.len() / 2).copied().unwrap_or(fallback).max(1)
}

/// Section check: what leads a definition, and the budget as the median window size.
pub(crate) fn check() -> Result<(), String> {
    for (line, hash, want) in [
        ("/// doc", false, true),
        ("    #[must_use]", false, true),
        ("@decorator", false, true),
        (" * block", false, true),
        ("//! module note", false, false),
        ("# comment", true, true),
        ("# comment", false, false),
        ("fn x() {", false, false),
        ("", false, false),
    ] {
        if leads(line, hash) != want {
            return Err(format!("units: leads({line:?}, {hash}) is not {want}"));
        }
    }
    if budget(&[10, 30, 20], 5) != 20 || budget(&[], 5) != 5 || budget(&[0, 0], 7) != 7 {
        return Err("units: budget is not the median window size".into());
    }
    Ok(())
}
