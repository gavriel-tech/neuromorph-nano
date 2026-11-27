//! Classes and the never-save scanner (2026-11-12, memory revision 2, section 8; direction of Luis Jake Gabriel III:
//! the "never delete" and "never save" controls are to be more nuanced).
//!
//! Classes: pinned (identity, objectives, anything Luis marks: never drops, never removed, never
//! merged); protected (memories verified right, corrections: never removed, may be merged);
//! normal (kept by retention and the archive states); ephemeral (test answers, scratch: never
//! written to the live memory).
//!
//! Never save: text that looks like a credential, key or token, an e-mail address, or a locked test
//! question's wording is refused at the writer gate with the reason logged. Matching is by hand
//! (no outside crates): known key prefixes, private-key blocks, `name = value` secrets, long mixed
//! high-entropy strings, e-mail shapes, and locked questions by their exact normalised wording.
use crate::record::Class;

/// Whether a memory of class `c` may leave its tier by retention.
#[must_use]
pub const fn may_drop(c: Class) -> bool {
    !matches!(c, Class::Pinned)
}
/// Whether it may be removed from the views (archive cleanup).
#[must_use]
pub const fn may_remove(c: Class) -> bool {
    matches!(c, Class::Normal)
}
/// Whether it may be merged with a duplicate.
#[must_use]
pub const fn may_merge(c: Class) -> bool {
    matches!(c, Class::Normal | Class::Protected)
}
/// Whether it may be written to the live memory at all.
#[must_use]
pub const fn may_save_live(c: Class) -> bool {
    !matches!(c, Class::Ephemeral)
}

const PREFIXES: [&str; 9] = ["sk-", "ghp_", "gho_", "ghs_", "github_pat_", "xoxb-", "xoxp-", "hf_", "glpat-"];
const SECRET_NAMES: [&str; 7] = ["password", "passwd", "secret", "api_key", "apikey", "access_token", "private_key"];

fn token_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

/// Why `text` must never be saved, if it must not.
#[must_use]
pub fn never_save(text: &str, locked: &[String]) -> Option<String> {
    if text.contains("-----BEGIN") && text.contains("PRIVATE KEY") {
        return Some("a private key block".into());
    }
    for p in PREFIXES {
        for (i, _) in text.match_indices(p) {
            let rest: String = text.get(i + p.len()..).unwrap_or("").chars().take_while(|c| token_char(*c)).collect();
            let starts_word = i == 0 || text.get(..i).and_then(|t| t.chars().last()).is_none_or(|c| !token_char(c));
            if starts_word && rest.len() >= 20 {
                return Some(format!("a token with the prefix {p}"));
            }
        }
    }
    for (i, _) in text.match_indices("AKIA") {
        let rest: String = text.get(i + 4..).unwrap_or("").chars().take_while(char::is_ascii_alphanumeric).collect();
        if rest.len() >= 16 && rest.chars().take(16).all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()) {
            return Some("a cloud access key".into());
        }
    }
    let low = text.to_ascii_lowercase();
    for name in SECRET_NAMES {
        for (i, _) in low.match_indices(name) {
            let after = low.get(i + name.len()..).unwrap_or("").trim_start_matches(['"', '\'']);
            let after = after.trim_start();
            if let Some(v) = after.strip_prefix(['=', ':']) {
                let value: String = v
                    .trim_start()
                    .trim_start_matches(['"', '\''])
                    .chars()
                    .take_while(|c| !c.is_whitespace() && !matches!(c, '"' | '\'' | ',' | ';'))
                    .collect();
                let placeholder = value.is_empty()
                    || value.starts_with('<')
                    || value.chars().all(|c| c == '*' || c == 'x')
                    || value == "none"
                    || value == "null";
                if value.len() >= 6 && !placeholder {
                    return Some(format!("a {name} value"));
                }
            }
        }
    }
    for word in text.split(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | ',' | ';' | '(' | ')' | '<' | '>')) {
        let w = word.trim_matches(['.', ':']);
        if w.len() >= 32
            && w.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '=' | '_' | '-'))
            && w.chars().any(|c| c.is_ascii_digit())
            && w.chars().any(|c| c.is_ascii_uppercase())
            && w.chars().any(|c| c.is_ascii_lowercase())
        {
            return Some("a long high-entropy string".into());
        }
        if let Some((user, host)) = w.split_once('@')
            && !user.is_empty()
            && user.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '+'))
            && host.contains('.')
            && host.rsplit('.').next().is_some_and(|t| t.len() >= 2 && t.chars().all(|c| c.is_ascii_alphabetic()))
        {
            return Some("an e-mail address".into());
        }
    }
    let norm = |t: &str| t.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
    let n = norm(text);
    if locked.iter().map(|q| norm(q)).any(|q| q.len() >= 20 && n.contains(&q)) {
        return Some("a locked test question's wording".into());
    }
    None
}

/// The wording of every locked test question under `<home>/tests/capability-gates/` (files named
/// `questions*.json`, field `q`), for the never-save check.
#[must_use]
pub fn locked_questions(home: &std::path::Path) -> Vec<String> {
    fn walk(dir: &std::path::Path, out: &mut Vec<String>, depth: usize) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() && depth < 3 {
                walk(&p, out, depth + 1);
            } else if p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("questions") && n.ends_with(".json"))
                && let Some(v) = std::fs::read(&p).ok().and_then(|b| nmjson::parse(&b).ok())
                && let Ok(qs) = v.get("questions").and_then(nmjson::Value::array)
            {
                out.extend(qs.iter().filter_map(|q| q.get("q").and_then(nmjson::Value::string).ok().map(str::to_owned)));
            }
        }
    }
    let mut out = Vec::new();
    walk(&home.join("tests").join("capability-gates"), &mut out, 0);
    out
}

/// Section check: each secret shape is refused and ordinary text passes, placeholders included; a
/// locked question's wording is refused; the class rules hold.
/// # Errors
/// One fault line.
pub(crate) fn check() -> Result<(), String> {
    let locked = vec!["Which function decides how many pieces the gate sends up?".to_owned()];
    let bad = [
        "the key is sk-ABCdef1234567890abcdefGHIJ",
        "export GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz0123",
        "AKIAIOSFODNN7EXAMPLE was in the config",
        "password = hunter2secret",
        "api_key: \"9f8e7d6c5b4a\"",
        "-----BEGIN RSA PRIVATE KEY-----",
        "token Zm9vYmFyQmF6UXV4MTIzNDU2Nzg5MGFiY2RlZg== here",
        "write to someone@example.com",
        "they asked: which function decides how many pieces the gate sends up?",
    ];
    for b in bad {
        if never_save(b, &locked).is_none() {
            return Err(format!("nmmem.classes: not refused: {b}"));
        }
    }
    let good = [
        "fn decide_switch(x: u8) -> bool { x > 3 }",
        "password = <set in the environment>",
        "the api_key field is read from the config",
        "sk-learn is a library",
        "decide.rs:241 lines 241..270",
        "Luis Jake Gabriel III set the windows to 28, 14 and 7 run-days",
    ];
    for g in good {
        if let Some(why) = never_save(g, &locked) {
            return Err(format!("nmmem.classes: refused ordinary text ({why}): {g}"));
        }
    }
    if may_remove(Class::Pinned)
        || may_remove(Class::Protected)
        || may_merge(Class::Pinned)
        || !may_merge(Class::Protected)
        || may_drop(Class::Pinned)
        || may_save_live(Class::Ephemeral)
    {
        return Err("nmmem.classes: class rules".into());
    }
    Ok(())
}
