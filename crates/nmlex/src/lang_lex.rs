//! The generic lexer: one pass over a source file driven by its `lang::Lang` entry. Comments are
//! dropped; strings (with their prefixes, triple quotes, template literals and the expressions
//! inside `${ }`, raw and verbatim forms, here-strings) become one `Str` token each; a JavaScript
//! `/` becomes a `Regex` token where a regular expression can start. Every token carries its line,
//! its column and its place in the source, so the tagger can find line starts, indentation and
//! the exact text between two tokens. It never fails: an unclosed string or comment runs to the
//! end of the file.
use crate::lang::Lang;

/// What a token is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Ident,
    Num,
    Punct,
    /// A string literal; its text is the literal's content between its delimiters.
    Str,
    /// A regular expression literal (JavaScript).
    Regex,
}

/// One token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tok {
    pub text: String,
    pub kind: Kind,
    /// 1-based line of the token's first character.
    pub line: usize,
    /// 1-based line of the token's last character (a string may span lines).
    pub last_line: usize,
    /// 0-based column (in characters) of the token's first character.
    pub col: usize,
    /// Index (in characters) of the first character in the source, and one past the last.
    pub start: usize,
    pub end: usize,
}

/// Words after which a `/` opens a regular expression rather than a division.
const REGEX_AFTER: &[&str] =
    &["return", "typeof", "case", "do", "else", "in", "of", "new", "delete", "void", "throw", "instanceof", "yield", "await"];

fn starts(c: &[char], i: usize, s: &str) -> bool {
    s.chars().enumerate().all(|(k, ch)| c.get(i + k) == Some(&ch))
}

/// Index just past the first `s` found at or after `i` (the end of input when absent).
fn past(c: &[char], i: usize, s: &str) -> usize {
    let n = s.chars().count();
    (i..c.len()).find(|&j| starts(c, j, s)).map_or(c.len(), |j| j + n)
}

/// End of an ordinary quoted string opened at `i`.
fn quoted(c: &[char], i: usize, escape: char, doubled: bool) -> usize {
    let Some(&q) = c.get(i) else { return i };
    let mut j = i + 1;
    while let Some(&ch) = c.get(j) {
        if ch == escape {
            j += 2;
        } else if ch == q {
            if doubled && c.get(j + 1) == Some(&q) {
                j += 2;
            } else {
                return j + 1;
            }
        } else {
            j += 1;
        }
    }
    c.len()
}

/// A Python string at `i` (with its prefix): (content start, content end, end).
fn python_string(c: &[char], i: usize) -> Option<(usize, usize, usize)> {
    let mut j = i;
    let mut format = false;
    while j < i + 2 && c.get(j).is_some_and(|ch| matches!(ch.to_ascii_lowercase(), 'r' | 'b' | 'f' | 'u')) {
        format |= c.get(j).is_some_and(|ch| ch.eq_ignore_ascii_case(&'f'));
        j += 1;
    }
    let q = *c.get(j)?;
    if q != '"' && q != '\'' {
        return None;
    }
    let triple = c.get(j + 1) == Some(&q) && c.get(j + 2) == Some(&q);
    let open = if triple { 3 } else { 1 };
    let body = j + open;
    let mut k = body;
    let mut depth = 0_usize;
    while let Some(&ch) = c.get(k) {
        if ch == '\\' {
            k += 2;
            continue;
        }
        if format && ch == '{' {
            if depth == 0 && c.get(k + 1) == Some(&'{') {
                k += 2;
                continue;
            }
            depth += 1;
        } else if format && ch == '}' && depth > 0 {
            depth -= 1;
        } else if depth > 0 && (ch == '"' || ch == '\'') {
            // A string inside a replacement field (allowed with any quote since Python 3.12).
            k = quoted(c, k, '\\', false);
            continue;
        } else if depth == 0 && ch == q && (!triple || (c.get(k + 1) == Some(&q) && c.get(k + 2) == Some(&q))) {
            return Some((body, k, k + open));
        } else if depth == 0 && ch == '\n' && !triple {
            return Some((body, k, k));
        }
        k += 1;
    }
    Some((body, c.len(), c.len()))
}

/// The template literal body from `i` (just past the backtick or a closing `}`): the index past
/// the closing backtick and false, or past a `${` and true.
fn template(c: &[char], mut i: usize) -> (usize, bool) {
    while let Some(&ch) = c.get(i) {
        match ch {
            '\\' => i += 2,
            '`' => return (i + 1, false),
            '$' if c.get(i + 1) == Some(&'{') => return (i + 2, true),
            _ => i += 1,
        }
    }
    (c.len(), false)
}

/// End of a regular expression literal opened by `/` at `i`, if one is there.
fn regex(c: &[char], i: usize) -> Option<usize> {
    let mut j = i + 1;
    let mut class = false;
    if matches!(c.get(j), Some('/' | '*')) {
        return None;
    }
    while let Some(&ch) = c.get(j) {
        match ch {
            '\\' => j += 2,
            '\n' => return None,
            '[' => {
                class = true;
                j += 1;
            }
            ']' => {
                class = false;
                j += 1;
            }
            '/' if !class => {
                j += 1;
                while c.get(j).is_some_and(|x| x.is_ascii_alphabetic()) {
                    j += 1;
                }
                return Some(j);
            }
            _ => j += 1,
        }
    }
    None
}

fn regex_allowed(out: &[Tok]) -> bool {
    match out.last() {
        None => true,
        Some(t) => match t.kind {
            Kind::Punct => !matches!(t.text.as_str(), ")" | "]" | "}"),
            Kind::Ident => REGEX_AFTER.contains(&t.text.as_str()),
            _ => false,
        },
    }
}

struct Lexer<'a> {
    c: &'a [char],
    lang: &'a Lang,
    line_starts: Vec<usize>,
    out: Vec<Tok>,
}

impl Lexer<'_> {
    fn push(&mut self, start: usize, end: usize, kind: Kind, text: String) {
        let li = self.line_starts.partition_point(|&s| s <= start).saturating_sub(1);
        let col = start - self.line_starts.get(li).copied().unwrap_or(0);
        let last = self.line_starts.partition_point(|&s| s < end.max(start + 1)).saturating_sub(1).max(li);
        self.out.push(Tok { text, kind, line: li + 1, last_line: last + 1, col, start, end });
    }
    fn slice(&self, a: usize, b: usize) -> String {
        self.c.get(a..b.min(self.c.len())).map(|s| s.iter().collect()).unwrap_or_default()
    }
    fn ident_char(&self, j: usize) -> bool {
        let Some(&ch) = self.c.get(j) else { return false };
        if ch.is_alphanumeric() || ch == '_' {
            return true;
        }
        if ch == '-' && self.lang.ident_extra.contains('-') {
            let prev = j.checked_sub(1).and_then(|p| self.c.get(p)).is_some_and(|p| p.is_alphanumeric() || *p == '_' || *p == '-');
            let next = self.c.get(j + 1).is_some_and(|n| n.is_alphabetic() || *n == '-');
            return prev && next;
        }
        ch != '-' && self.lang.ident_extra.contains(ch)
    }
    fn ident_start(&self, i: usize) -> bool {
        let Some(&ch) = self.c.get(i) else { return false };
        if ch.is_alphabetic() || ch == '_' {
            return true;
        }
        if !self.lang.ident_start.contains(ch) {
            return false;
        }
        let next = self.c.get(i + 1).copied();
        match ch {
            // CSS custom properties and vendor prefixes: `--x`, `-webkit-x`.
            '-' => next.is_some_and(|n| n == '-' || n.is_alphabetic()),
            // PowerShell variables: `$name`, `$_`, `${name}` is read as `$` then a brace.
            '$' if self.lang.id == "powershell" => next.is_some_and(|n| n.is_alphabetic() || n == '_'),
            _ => true,
        }
    }
    /// A string form special to the language at `i`: (content start, content end, end).
    fn special_string(&self, i: usize) -> Option<(usize, usize, usize)> {
        let (c, lang) = (self.c, self.lang);
        let ch = *c.get(i)?;
        if lang.here_strings && ch == '@' && matches!(c.get(i + 1), Some('"' | '\'')) {
            let q = *c.get(i + 1)?;
            let close: String = ['\n', q, '@'].iter().collect();
            let end = past(c, i + 2, &close);
            return Some((i + 2, end.saturating_sub(3).max(i + 2), end));
        }
        if lang.python_strings {
            let prefixed = matches!(ch.to_ascii_lowercase(), 'r' | 'b' | 'f' | 'u');
            if ch == '"' || ch == '\'' || prefixed {
                return python_string(c, i);
            }
        }
        if lang.csharp_strings {
            if starts(c, i, "\"\"\"") {
                let end = past(c, i + 3, "\"\"\"");
                return Some((i + 3, end.saturating_sub(3).max(i + 3), end));
            }
            let mut j = i;
            let mut verbatim = false;
            while j < i + 2 && matches!(c.get(j), Some('@' | '$')) {
                verbatim |= c.get(j) == Some(&'@');
                j += 1;
            }
            if j > i && c.get(j) == Some(&'"') {
                let end = if verbatim { quoted(c, j, '\u{0}', true) } else { quoted(c, j, '\\', false) };
                return Some((j + 1, end.saturating_sub(1).max(j + 1), end));
            }
        }
        if lang.cpp_raw && starts(c, i, "R\"") {
            let open = (i + 2..c.len()).find(|&j| c.get(j) == Some(&'(')).unwrap_or(c.len());
            let delim: String = self.slice(i + 2, open);
            let close = format!("){delim}\"");
            let end = past(c, open + 1, &close);
            return Some((open + 1, end.saturating_sub(close.chars().count()).max(open + 1), end));
        }
        if lang.backtick_raw && ch == '`' {
            let end = past(c, i + 1, "`");
            return Some((i + 1, end.saturating_sub(1).max(i + 1), end));
        }
        None
    }
    fn run(mut self) -> Vec<Tok> {
        let c = self.c;
        let lang = self.lang;
        // A `#!` line at the very start (scripts run by an interpreter) is not code.
        let mut i =
            if lang.regex_literal && starts(c, 0, "#!") { (0..c.len()).find(|&j| c.get(j) == Some(&'\n')).unwrap_or(c.len()) } else { 0 };
        // Open template interpolations: the brace depth inside each `${ }`.
        let mut interp: Vec<usize> = Vec::new();
        while let Some(&ch) = c.get(i) {
            if ch.is_whitespace() {
                i += 1;
                continue;
            }
            if let Some((_, close)) = lang.block_comments.iter().find(|(o, _)| starts(c, i, o)) {
                i = past(c, i + 2, close);
                continue;
            }
            if lang.line_comments.iter().any(|o| starts(c, i, o)) {
                i = (i..c.len()).find(|&j| c.get(j) == Some(&'\n')).unwrap_or(c.len());
                continue;
            }
            if let Some((a, b, end)) = self.special_string(i) {
                let text = self.slice(a, b);
                self.push(i, end, Kind::Str, text);
                i = end.max(i + 1);
                continue;
            }
            if lang.template && ch == '`' {
                let (next, open) = template(c, i + 1);
                self.push(i, next, Kind::Str, String::new());
                if open {
                    interp.push(0);
                }
                i = next;
                continue;
            }
            if lang.quotes.contains(&ch) {
                let end = quoted(c, i, lang.escape, lang.doubled_quote);
                let text = self.slice(i + 1, end.saturating_sub(1).max(i + 1));
                self.push(i, end, Kind::Str, text);
                i = end;
                continue;
            }
            if let Some(depth) = interp.last_mut() {
                if ch == '{' {
                    *depth += 1;
                } else if ch == '}' {
                    if *depth == 0 {
                        interp.pop();
                        let (next, open) = template(c, i + 1);
                        if open {
                            interp.push(0);
                        }
                        i = next;
                        continue;
                    }
                    *depth -= 1;
                }
            }
            if lang.regex_literal
                && ch == '/'
                && regex_allowed(&self.out)
                && let Some(end) = regex(c, i)
            {
                let text = self.slice(i, end);
                self.push(i, end, Kind::Regex, text);
                i = end;
                continue;
            }
            let start = i;
            let kind = if self.ident_start(i) {
                i += 1;
                while self.ident_char(i) {
                    i += 1;
                }
                Kind::Ident
            } else if ch.is_ascii_digit() {
                i += 1;
                while c.get(i).is_some_and(|x| x.is_alphanumeric() || *x == '_')
                    || (c.get(i) == Some(&'.') && c.get(i + 1).is_some_and(char::is_ascii_digit))
                {
                    i += 1;
                }
                Kind::Num
            } else {
                i += if ["=>", "->", "::"].iter().any(|p| starts(c, start, p)) { 2 } else { 1 };
                Kind::Punct
            };
            let text = self.slice(start, i);
            self.push(start, i, kind, text);
        }
        self.out
    }
}

/// The tokens of `source` in language `lang`.
#[must_use]
pub fn lex(source: &str, lang: &Lang) -> Vec<Tok> {
    let c: Vec<char> = source.chars().collect();
    let mut line_starts = vec![0];
    line_starts.extend(c.iter().enumerate().filter(|(_, ch)| **ch == '\n').map(|(i, _)| i + 1));
    Lexer { c: &c, lang, line_starts, out: Vec::new() }.run()
}

/// Tokens rendered for comparison with a hand list: identifiers, numbers and punctuation as
/// written, every string as `$S`, every regular expression as `$R`, joined by single spaces.
#[must_use]
pub fn render(toks: &[Tok]) -> String {
    toks.iter()
        .map(|t| match t.kind {
            Kind::Str => "$S",
            Kind::Regex => "$R",
            _ => t.text.as_str(),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn check() -> Result<(), String> {
    let lang = |id: &str| crate::lang::by_id(id).ok_or_else(|| format!("nmlex.lang_lex: no language {id}"));
    let cases = [
        ("typescript", "const a = `x${b + `y${c}`}z`; // c", "const a = $S b + $S c ;"),
        ("typescript", "let r = /a\\/b[/]/g.test(s) / 2;", "let r = $R . test ( s ) / 2 ;"),
        ("python", "x = a // 2  # half\ny = f\"{d[\"k\"]}\" + '''q\n'''", "x = a / / 2 y = $S + $S"),
        ("powershell", "<# note #>\n$x = 'it''s' # c\nfunction Get-Item { }", "$x = $S function Get-Item { }"),
        ("css", ":root { --accent-color: #fff; } /* c */", ": root { --accent-color : # fff ; }"),
        ("c", "#include <a.h>\nint f(void) { return 'x'; }", "# include < a . h > int f ( void ) { return $S ; }"),
    ];
    for (id, src, want) in cases {
        let got = render(&lex(src, lang(id)?));
        if got != want {
            return Err(format!("nmlex.lang_lex: {id} {src:?} gave {got:?}, want {want:?}"));
        }
    }
    let t = lex("a\n  bb", lang("python")?);
    if t.get(1).map(|x| (x.line, x.col)) != Some((2, 2)) {
        return Err("nmlex.lang_lex: line and column".into());
    }
    Ok(())
}
