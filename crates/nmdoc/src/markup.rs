//! Tag tokenizer shared by the HTML and XML readers: text, start and end tags with attributes,
//! comments, CDATA, processing instructions and declarations, and character references. No DTD
//! is read and no external entity is ever resolved. Tolerant: a stray `<` is text, an unclosed
//! tag ends at the end of the input.

/// One token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tok {
    Text(String),
    /// Name (lower case when `fold`), attributes, self-closing.
    Start(String, Vec<(String, String)>, bool),
    End(String),
}

/// Named character references: the common ones (the full HTML list is long; these cover text).
const NAMED: &[(&str, char)] = &[
    ("amp", '&'),
    ("lt", '<'),
    ("gt", '>'),
    ("quot", '"'),
    ("apos", '\''),
    ("nbsp", ' '),
    ("copy", '\u{a9}'),
    ("reg", '\u{ae}'),
    ("trade", '\u{2122}'),
    ("hellip", '\u{2026}'),
    ("ndash", '\u{2013}'),
    ("mdash", '\u{2014}'),
    ("lsquo", '\u{2018}'),
    ("rsquo", '\u{2019}'),
    ("ldquo", '\u{201c}'),
    ("rdquo", '\u{201d}'),
    ("bull", '\u{2022}'),
    ("middot", '\u{b7}'),
    ("deg", '\u{b0}'),
    ("plusmn", '\u{b1}'),
    ("times", '\u{d7}'),
    ("divide", '\u{f7}'),
    ("micro", '\u{b5}'),
    ("para", '\u{b6}'),
    ("sect", '\u{a7}'),
    ("frac12", '\u{bd}'),
    ("frac14", '\u{bc}'),
    ("frac34", '\u{be}'),
    ("sup2", '\u{b2}'),
    ("sup3", '\u{b3}'),
    ("euro", '\u{20ac}'),
    ("pound", '\u{a3}'),
    ("yen", '\u{a5}'),
    ("cent", '\u{a2}'),
    ("laquo", '\u{ab}'),
    ("raquo", '\u{bb}'),
    ("eacute", '\u{e9}'),
    ("egrave", '\u{e8}'),
    ("ecirc", '\u{ea}'),
    ("aacute", '\u{e1}'),
    ("agrave", '\u{e0}'),
    ("acirc", '\u{e2}'),
    ("auml", '\u{e4}'),
    ("ouml", '\u{f6}'),
    ("uuml", '\u{fc}'),
    ("oacute", '\u{f3}'),
    ("iacute", '\u{ed}'),
    ("uacute", '\u{fa}'),
    ("ccedil", '\u{e7}'),
    ("ntilde", '\u{f1}'),
    ("szlig", '\u{df}'),
    ("alpha", '\u{3b1}'),
    ("beta", '\u{3b2}'),
    ("gamma", '\u{3b3}'),
    ("delta", '\u{3b4}'),
    ("mu", '\u{3bc}'),
    ("pi", '\u{3c0}'),
    ("sigma", '\u{3c3}'),
    ("omega", '\u{3c9}'),
    ("larr", '\u{2190}'),
    ("rarr", '\u{2192}'),
    ("le", '\u{2264}'),
    ("ge", '\u{2265}'),
    ("ne", '\u{2260}'),
    ("asymp", '\u{2248}'),
    ("infin", '\u{221e}'),
];

/// Replace character references (`&amp;`, `&#38;`, `&#x26;`); an unknown one stays as written.
#[must_use]
pub fn unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_owned();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(rest.get(..i).unwrap_or(""));
        let after = rest.get(i + 1..).unwrap_or("");
        let end = after.find(';').filter(|e| *e <= 10);
        let decoded = end.and_then(|e| {
            let name = after.get(..e)?;
            if let Some(num) = name.strip_prefix('#') {
                let v = if let Some(h) = num.strip_prefix(['x', 'X']) { u32::from_str_radix(h, 16).ok() } else { num.parse().ok() };
                v.and_then(char::from_u32).map(|c| (c, e))
            } else {
                NAMED.iter().find(|(n, _)| *n == name).map(|(_, c)| (*c, e))
            }
        });
        match decoded {
            Some((c, e)) => {
                out.push(c);
                rest = after.get(e + 1..).unwrap_or("");
            }
            None => {
                out.push('&');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

fn attrs(s: &str, fold: bool) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut it = s.trim().chars().peekable();
    loop {
        while it.peek().is_some_and(|c| c.is_whitespace()) {
            it.next();
        }
        let mut name = String::new();
        while let Some(&c) = it.peek() {
            if c.is_whitespace() || c == '=' {
                break;
            }
            name.push(c);
            it.next();
        }
        if name.is_empty() {
            break;
        }
        while it.peek().is_some_and(|c| c.is_whitespace()) {
            it.next();
        }
        let mut value = String::new();
        if it.peek() == Some(&'=') {
            it.next();
            while it.peek().is_some_and(|c| c.is_whitespace()) {
                it.next();
            }
            match it.peek().copied() {
                Some(q @ ('"' | '\'')) => {
                    it.next();
                    for c in it.by_ref() {
                        if c == q {
                            break;
                        }
                        value.push(c);
                    }
                }
                _ => {
                    while let Some(&c) = it.peek() {
                        if c.is_whitespace() {
                            break;
                        }
                        value.push(c);
                        it.next();
                    }
                }
            }
        }
        out.push((if fold { name.to_lowercase() } else { name }, unescape(&value)));
    }
    out
}

/// Tokens of a markup text. `fold` lower-cases names (HTML); `raw` names elements whose content
/// is text up to their end tag (HTML `script`, `style`).
#[must_use]
pub fn tokens(s: &str, fold: bool, raw: &[&str]) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut rest = s;
    let mut text = String::new();
    let flush = |text: &mut String, out: &mut Vec<Tok>| {
        if !text.is_empty() {
            out.push(Tok::Text(unescape(text)));
            text.clear();
        }
    };
    while !rest.is_empty() {
        let Some(i) = rest.find('<') else {
            text.push_str(rest);
            break;
        };
        text.push_str(rest.get(..i).unwrap_or(""));
        let tail = rest.get(i..).unwrap_or("");
        if let Some(r) = tail.strip_prefix("<!--") {
            flush(&mut text, &mut out);
            rest = r.find("-->").and_then(|e| r.get(e + 3..)).unwrap_or("");
            continue;
        }
        if let Some(r) = tail.strip_prefix("<![CDATA[") {
            let e = r.find("]]>").unwrap_or(r.len());
            flush(&mut text, &mut out);
            out.push(Tok::Text(r.get(..e).unwrap_or("").to_owned()));
            rest = r.get(e + 3..).unwrap_or("");
            continue;
        }
        if tail.starts_with("<!") || tail.starts_with("<?") {
            flush(&mut text, &mut out);
            rest = tail.find('>').and_then(|e| tail.get(e + 1..)).unwrap_or("");
            continue;
        }
        let closing = tail.starts_with("</");
        let body_start = if closing { 2 } else { 1 };
        let name_ok = tail.get(body_start..).and_then(|x| x.chars().next()).is_some_and(|c| c.is_alphabetic() || c == '_' || c == ':');
        if !name_ok {
            text.push('<');
            rest = tail.get(1..).unwrap_or("");
            continue;
        }
        let Some(end) = tag_end(tail) else {
            text.push_str(tail);
            break;
        };
        flush(&mut text, &mut out);
        let inner = tail.get(body_start..end).unwrap_or("");
        let selfc = inner.ends_with('/');
        let inner = inner.trim_end_matches('/');
        let (name, rest_attrs) = inner.split_once(|c: char| c.is_whitespace()).unwrap_or((inner, ""));
        let name = if fold { name.to_lowercase() } else { name.to_owned() };
        rest = tail.get(end + 1..).unwrap_or("");
        if closing {
            out.push(Tok::End(name));
        } else {
            let is_raw = raw.contains(&name.as_str()) && !selfc;
            out.push(Tok::Start(name.clone(), attrs(rest_attrs, fold), selfc));
            if is_raw {
                let close = format!("</{name}");
                let lower = rest.to_lowercase();
                let e = lower.find(&close).unwrap_or(rest.len());
                out.push(Tok::Text(rest.get(..e).unwrap_or("").to_owned()));
                rest = rest.get(e..).unwrap_or("");
            }
        }
    }
    flush(&mut text, &mut out);
    out
}

/// The `>` that ends the tag starting `s`, outside quoted attribute values.
fn tag_end(s: &str) -> Option<usize> {
    let mut quote = None;
    for (i, c) in s.char_indices() {
        match (quote, c) {
            (Some(q), _) if c == q => quote = None,
            (Some(_), _) => {}
            (None, '"' | '\'') => quote = Some(c),
            (None, '>') => return Some(i),
            _ => {}
        }
    }
    None
}

/// Collapse runs of white space to one space.
#[must_use]
pub fn collapse(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(crate) fn check() -> Result<(), String> {
    let t = tokens("<p class=\"a b\">Fish &amp; chips &#176;C<br/></p><!-- x --><script>if (a < b) {}</script>", true, &["script"]);
    let want = vec![
        Tok::Start("p".into(), vec![("class".into(), "a b".into())], false),
        Tok::Text("Fish & chips \u{b0}C".into()),
        Tok::Start("br".into(), Vec::new(), true),
        Tok::End("p".into()),
        Tok::Start("script".into(), Vec::new(), false),
        Tok::Text("if (a < b) {}".into()),
        Tok::End("script".into()),
    ];
    if t != want {
        return Err(format!("nmdoc.markup: tokens {t:?}"));
    }
    if unescape("a &unknown; b &lt;") != "a &unknown; b <" || tokens("1 < 2", true, &[]) != vec![Tok::Text("1 < 2".into())] {
        return Err("nmdoc.markup: references or stray <".into());
    }
    Ok(())
}
