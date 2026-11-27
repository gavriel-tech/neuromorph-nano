//! Comment and literal aware tokenization for Rust and C family checks.
#[derive(Debug, Clone)]
pub(crate) struct Token {
    pub(crate) text: String,
    pub(crate) line: usize,
}

pub(crate) fn tokens(source: &str, rust: bool) -> Vec<Token> {
    let chars: Vec<char> = source.chars().collect();
    let mut out = Vec::new();
    let mut at = 0;
    let mut line = 1;
    while at < chars.len() {
        let start = at;
        if chars[at].is_whitespace() {
            at += 1;
        } else if chars.get(at..at + 2) == Some(&['/', '/']) {
            while at < chars.len() && chars[at] != '\n' {
                at += 1;
            }
        } else if chars.get(at..at + 2) == Some(&['/', '*']) {
            at = comment_end(&chars, at, rust);
        } else if let Some(end) = raw_end(&chars, at, rust) {
            at = end;
        } else if chars[at] == '"' || (chars[at] == '\'' && (!rust || char_end(&chars, at).is_some())) {
            at = quoted_end(&chars, at);
        } else if rust && chars.get(at..at + 2) == Some(&['r', '#']) && chars.get(at + 2).is_some_and(|c| c.is_alphabetic() || *c == '_') {
            at += 2;
            while chars.get(at).is_some_and(|c| c.is_alphanumeric() || *c == '_') {
                at += 1;
            }
            out.push(Token { text: chars[start..at].iter().collect(), line });
        } else {
            at += 1;
            if chars[start].is_alphanumeric() || chars[start] == '_' {
                while chars.get(at).is_some_and(|c| c.is_alphanumeric() || *c == '_') {
                    at += 1;
                }
            }
            out.push(Token { text: chars[start..at].iter().collect(), line });
        }
        line += chars[start..at].iter().filter(|c| **c == '\n').count();
    }
    out
}

fn comment_end(chars: &[char], start: usize, nested: bool) -> usize {
    let mut at = start + 2;
    let mut depth = 1;
    while at < chars.len() {
        if nested && chars.get(at..at + 2) == Some(&['/', '*']) {
            depth += 1;
            at += 2;
        } else if chars.get(at..at + 2) == Some(&['*', '/']) {
            depth -= 1;
            at += 2;
            if depth == 0 {
                break;
            }
        } else {
            at += 1;
        }
    }
    at
}

fn quoted_end(chars: &[char], start: usize) -> usize {
    let mut at = start + 1;
    while at < chars.len() {
        if chars[at] == '\\' {
            at = (at + 2).min(chars.len());
        } else if chars[at] == chars[start] {
            return at + 1;
        } else {
            at += 1;
        }
    }
    at
}

fn char_end(chars: &[char], start: usize) -> Option<usize> {
    let mut at = start + 1;
    if chars.get(at) == Some(&'\\') {
        at += 2;
        if chars.get(at - 1) == Some(&'u') && chars.get(at) == Some(&'{') {
            while chars.get(at).is_some_and(|c| *c != '}') {
                at += 1;
            }
            at += 1;
        } else if chars.get(at - 1) == Some(&'x') {
            at += 2;
        }
    } else {
        at += 1;
    }
    (chars.get(at) == Some(&'\'')).then_some(at + 1)
}

fn raw_end(chars: &[char], start: usize, rust: bool) -> Option<usize> {
    if !rust {
        return cpp_raw_end(chars, start);
    }
    let mut at = start;
    if matches!(chars.get(at), Some('b' | 'c')) {
        at += 1;
    }
    if chars.get(at) != Some(&'r') {
        return None;
    }
    at += 1;
    let hash_start = at;
    while chars.get(at) == Some(&'#') {
        at += 1;
    }
    let hashes = at - hash_start;
    if chars.get(at) != Some(&'"') {
        return None;
    }
    at += 1;
    while at < chars.len() {
        if chars[at] == '"' && (0..hashes).all(|i| chars.get(at + 1 + i) == Some(&'#')) {
            return Some(at + 1 + hashes);
        }
        at += 1;
    }
    Some(chars.len())
}

fn cpp_raw_end(chars: &[char], start: usize) -> Option<usize> {
    if chars.get(start..start + 2) != Some(&['R', '"']) {
        return None;
    }
    let begin = start + 2;
    let mut at = begin;
    while chars.get(at).is_some_and(|c| *c != '(') {
        at += 1;
    }
    let delimiter = chars.get(begin..at)?;
    at += 1;
    while at < chars.len() {
        if chars[at] == ')'
            && chars.get(at + 1..at + 1 + delimiter.len()) == Some(delimiter)
            && chars.get(at + 1 + delimiter.len()) == Some(&'"')
        {
            return Some(at + 2 + delimiter.len());
        }
        at += 1;
    }
    Some(chars.len())
}

pub(crate) fn unsafe_lines(source: &str) -> Vec<usize> {
    let ts = tokens(source, true);
    let mut lines = Vec::new();
    let mut attribute = false;
    let mut depth = 0;
    for (i, t) in ts.iter().enumerate() {
        if t.text == "#"
            && (ts.get(i + 1).is_some_and(|t| t.text == "[")
                || (ts.get(i + 1).is_some_and(|t| t.text == "!") && ts.get(i + 2).is_some_and(|t| t.text == "[")))
        {
            attribute = true;
        }
        if attribute && t.text == "[" {
            depth += 1;
        }
        if t.text == "unsafe" || (attribute && matches!(t.text.as_str(), "unsafe_code" | "r#unsafe_code")) {
            lines.push(t.line);
        }
        if attribute && t.text == "]" {
            depth -= 1;
            if depth == 0 {
                attribute = false;
            }
        }
    }
    lines.sort_unstable();
    lines.dedup();
    lines
}
