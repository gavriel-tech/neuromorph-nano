//! The rule layer over `nmlex` (the byte copy of `tools/stdchk/src/lex.rs`; no second copy
//! of the lexer): the folder's rule words and the `unsafe` boundary (only `nmffi` may hold
//! `unsafe`, PLAN section 3). Rule words are built at run time so this file does not flag
//! itself.

/// Rule words and their replacements: the long dash, the word the EMMA terms guide avoids
/// for MicroModels, and the older word for the Neuropod.
#[must_use]
pub fn rule_words() -> Vec<(String, String)> {
    let dash = '\u{2014}'.to_string();
    vec![
        (format!(" {dash} "), ", ".into()),
        (dash, ", ".into()),
        (["spec", "ialists"].concat(), "MicroModels".into()),
        (["spec", "ialist"].concat(), "MicroModel".into()),
        (["Octo", "pus"].concat(), "Neuropod".into()),
        (["octo", "pus"].concat(), "Neuropod".into()),
    ]
}

/// The line with every rule word replaced.
#[must_use]
pub fn fix_line(line: &str) -> String {
    rule_words().iter().fold(line.to_owned(), |l, (a, b)| l.replace(a.as_str(), b))
}

/// 1-based lines of `text` that hold a rule word, with the first word found.
#[must_use]
pub fn rule_hits(text: &str) -> Vec<(usize, String)> {
    let words = rule_words();
    text.lines()
        .enumerate()
        .filter_map(|(i, l)| words.iter().find(|(w, _)| l.contains(w.as_str())).map(|(w, _)| (i + 1, w.clone())))
        .collect()
}

/// 1-based lines of Rust `text` with an `unsafe` token in code, unless `path` is inside
/// the `nmffi` crate (the one crate allowed to hold it).
#[must_use]
pub fn unsafe_outside_ffi(path: &str, text: &str) -> Vec<usize> {
    let p = path.replace('\\', "/");
    if !p.ends_with(".rs") || p.contains("crates/nmffi/") {
        return Vec::new();
    }
    nmlex::unsafe_lines(text)
}

/// Section check on a built-in fixture.
/// # Errors
/// One line naming what failed.
pub fn check() -> Result<(), String> {
    let dash = '\u{2014}';
    let text = format!("ok line\nbad {dash} here\nthe {}s list\n", ["spec", "ialist"].concat());
    let hits = rule_hits(&text);
    if hits.iter().map(|h| h.0).collect::<Vec<_>>() != vec![2, 3] {
        return Err(format!("rule hits {hits:?}"));
    }
    if fix_line(&format!("a {dash} b")) != "a, b" {
        return Err("fix_line did not replace the dash".into());
    }
    let code = "fn f() { unsafe { g() } }\n// unsafe in a comment\n";
    if unsafe_outside_ffi("crates/nmapi/src/x.rs", code) != vec![1] {
        return Err(format!("unsafe lines {:?}", unsafe_outside_ffi("crates/nmapi/src/x.rs", code)));
    }
    if !unsafe_outside_ffi("crates/nmffi/src/cuda.rs", code).is_empty() {
        return Err("nmffi must be allowed unsafe".into());
    }
    Ok(())
}
