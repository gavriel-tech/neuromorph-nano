//! Reads and edits JSON settings files in place. Comments (`//`, `/* */`) and trailing commas are
//! accepted, as Claude Code's settings files may hold them. Every edit is one contiguous insertion
//! or deletion, so the bytes of every other entry stay exactly as they were, and removing what was
//! inserted gives back the original file byte for byte.

/// A parsed value with its byte span in the text.
#[derive(Debug)]
pub struct Node {
    pub start: usize,
    pub end: usize,
    pub v: V,
}

/// One member of an object: key text, where its key starts, its value.
#[derive(Debug)]
pub struct Member {
    pub key: String,
    pub key_start: usize,
    pub value: Node,
}

/// The value kinds.
#[derive(Debug)]
pub enum V {
    Obj(Vec<Member>),
    Arr(Vec<Node>),
    Str(String),
    Num(String),
    Bool(bool),
    Null,
}

impl Node {
    /// The member named `key` of an object.
    #[must_use]
    pub fn member(&self, key: &str) -> Option<&Member> {
        match &self.v {
            V::Obj(m) => m.iter().find(|x| x.key == key),
            _ => None,
        }
    }
    /// The value of member `key`.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Node> {
        self.member(key).map(|m| &m.value)
    }
    /// The text of a string value.
    #[must_use]
    pub fn str(&self) -> Option<&str> {
        match &self.v {
            V::Str(s) => Some(s),
            _ => None,
        }
    }
    /// The elements of an array.
    #[must_use]
    pub fn items(&self) -> Option<&[Node]> {
        match &self.v {
            V::Arr(a) => Some(a),
            _ => None,
        }
    }
    /// The members of an object.
    #[must_use]
    pub fn members(&self) -> Option<&[Member]> {
        match &self.v {
            V::Obj(m) => Some(m),
            _ => None,
        }
    }
}

struct P<'a> {
    b: &'a [u8],
    i: usize,
    depth: usize,
}

/// Line and column (1-based) of a byte offset, for error messages.
fn place(b: &[u8], at: usize) -> String {
    let before = b.get(..at.min(b.len())).unwrap_or_default();
    let line = before.iter().filter(|c| **c == b'\n').count() + 1;
    let col = before.iter().rev().take_while(|c| **c != b'\n').count() + 1;
    format!("line {line}, column {col}")
}

impl P<'_> {
    fn err<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("{what} at {}", place(self.b, self.i)))
    }
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }
    /// Skips white space and comments.
    fn ws(&mut self) -> Result<(), String> {
        loop {
            match self.peek() {
                Some(b' ' | b'\t' | b'\r' | b'\n') => self.i += 1,
                Some(b'/') if self.b.get(self.i + 1) == Some(&b'/') => {
                    while self.peek().is_some_and(|c| c != b'\n') {
                        self.i += 1;
                    }
                }
                Some(b'/') if self.b.get(self.i + 1) == Some(&b'*') => {
                    let from = self.i;
                    self.i += 2;
                    loop {
                        match self.peek() {
                            None => {
                                self.i = from;
                                return self.err("unclosed comment");
                            }
                            Some(b'*') if self.b.get(self.i + 1) == Some(&b'/') => {
                                self.i += 2;
                                break;
                            }
                            Some(_) => self.i += 1,
                        }
                    }
                }
                _ => return Ok(()),
            }
        }
    }
    fn value(&mut self) -> Result<Node, String> {
        self.ws()?;
        let start = self.i;
        let v = match self.peek() {
            Some(b'{') => self.object()?,
            Some(b'[') => self.array()?,
            Some(b'"') => V::Str(self.string()?),
            Some(b't') => self.word("true", V::Bool(true))?,
            Some(b'f') => self.word("false", V::Bool(false))?,
            Some(b'n') => self.word("null", V::Null)?,
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number()?,
            Some(_) => return self.err("unexpected character"),
            None => return self.err("unexpected end of file"),
        };
        Ok(Node { start, end: self.i, v })
    }
    fn word(&mut self, w: &str, v: V) -> Result<V, String> {
        if self.b.get(self.i..self.i + w.len()) == Some(w.as_bytes()) {
            self.i += w.len();
            Ok(v)
        } else {
            self.err("unexpected word")
        }
    }
    fn number(&mut self) -> Result<V, String> {
        let s = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_digit() || matches!(c, b'-' | b'+' | b'.' | b'e' | b'E')) {
            self.i += 1;
        }
        let t = std::str::from_utf8(self.b.get(s..self.i).unwrap_or_default()).unwrap_or_default();
        if t.parse::<f64>().is_err() {
            self.i = s;
            return self.err("bad number");
        }
        Ok(V::Num(t.to_owned()))
    }
    fn string(&mut self) -> Result<String, String> {
        let s = self.i;
        self.i += 1;
        let mut out: Vec<u16> = Vec::new();
        let mut raw: Vec<u8> = Vec::new();
        let flush = |raw: &mut Vec<u8>, out: &mut Vec<u16>| {
            out.extend(String::from_utf8_lossy(raw).encode_utf16());
            raw.clear();
        };
        loop {
            match self.peek() {
                None | Some(b'\n') => {
                    self.i = s;
                    return self.err("unclosed string");
                }
                Some(b'"') => {
                    self.i += 1;
                    flush(&mut raw, &mut out);
                    return Ok(String::from_utf16_lossy(&out));
                }
                Some(b'\\') => {
                    flush(&mut raw, &mut out);
                    let e = self.b.get(self.i + 1).copied();
                    self.i += 2;
                    let c = match e {
                        Some(b'"') => u16::from(b'"'),
                        Some(b'\\') => u16::from(b'\\'),
                        Some(b'/') => u16::from(b'/'),
                        Some(b'b') => 8,
                        Some(b'f') => 12,
                        Some(b'n') => 10,
                        Some(b'r') => 13,
                        Some(b't') => 9,
                        Some(b'u') => {
                            let h = self.b.get(self.i..self.i + 4).and_then(|h| std::str::from_utf8(h).ok());
                            let Some(n) = h.and_then(|h| u16::from_str_radix(h, 16).ok()) else { return self.err("bad escape") };
                            self.i += 4;
                            n
                        }
                        _ => return self.err("bad escape"),
                    };
                    out.push(c);
                }
                Some(c) => {
                    raw.push(c);
                    self.i += 1;
                }
            }
        }
    }
    fn enter(&mut self) -> Result<(), String> {
        self.depth += 1;
        if self.depth > 64 { self.err("nested too deeply") } else { Ok(()) }
    }
    fn object(&mut self) -> Result<V, String> {
        self.enter()?;
        self.i += 1;
        let mut members: Vec<Member> = Vec::new();
        loop {
            self.ws()?;
            match self.peek() {
                Some(b'}') => {
                    self.i += 1;
                    break;
                }
                Some(b'"') => {}
                _ => return self.err("expected a key or }"),
            }
            let key_start = self.i;
            let key = self.string()?;
            if members.iter().any(|m| m.key == key) {
                self.i = key_start;
                return self.err(&format!("duplicate key {key:?}"));
            }
            self.ws()?;
            if self.peek() != Some(b':') {
                return self.err("expected :");
            }
            self.i += 1;
            let value = self.value()?;
            members.push(Member { key, key_start, value });
            self.ws()?;
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    break;
                }
                _ => return self.err("expected , or }"),
            }
        }
        self.depth -= 1;
        Ok(V::Obj(members))
    }
    fn array(&mut self) -> Result<V, String> {
        self.enter()?;
        self.i += 1;
        let mut items = Vec::new();
        loop {
            self.ws()?;
            if self.peek() == Some(b']') {
                self.i += 1;
                break;
            }
            items.push(self.value()?);
            self.ws()?;
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    break;
                }
                _ => return self.err("expected , or ]"),
            }
        }
        self.depth -= 1;
        Ok(V::Arr(items))
    }
}

/// Parses a whole settings file (a byte order mark at the start is allowed).
/// # Errors
/// Why the text is not JSON (comments and trailing commas allowed), with line and column.
pub fn parse(text: &str) -> Result<Node, String> {
    let b = text.as_bytes();
    let start = if text.starts_with('\u{feff}') { 3 } else { 0 };
    let mut p = P { b, i: start, depth: 0 };
    p.ws()?;
    if p.peek().is_none() {
        return Err("the file is empty".into());
    }
    let root = p.value()?;
    p.ws()?;
    if p.i < b.len() {
        return p.err("text after the end of the JSON value");
    }
    Ok(root)
}

/// A value to write, with its keys in the order given.
#[derive(Clone, Debug, PartialEq)]
pub enum Out {
    Obj(Vec<(String, Out)>),
    Arr(Vec<Out>),
    Str(String),
    Num(i64),
    Bool(bool),
}

/// The JSON text of a string.
#[must_use]
pub fn quote(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// The layout of a file: its line break, its indent unit.
#[derive(Clone, Debug)]
pub struct Style {
    pub nl: &'static str,
    pub unit: String,
}

impl Style {
    /// The style a file already uses: `\r\n` when it holds one, the smallest indent seen (a tab
    /// when lines are indented with tabs), two spaces otherwise.
    #[must_use]
    pub fn of(text: &str) -> Self {
        let nl = if text.contains("\r\n") { "\r\n" } else { "\n" };
        let mut unit = String::from("  ");
        let mut best = usize::MAX;
        for l in text.lines() {
            if l.starts_with('\t') {
                unit = "\t".into();
                break;
            }
            let n = l.len() - l.trim_start_matches(' ').len();
            if n > 0 && n < best && !l.trim().is_empty() {
                best = n;
                unit = " ".repeat(n);
            }
        }
        Self { nl, unit }
    }
}

/// Pretty text of a value whose first line sits at `indent`: objects and arrays one entry per
/// line, entries at `indent` plus one unit, the closing bracket at `indent`.
#[must_use]
pub fn pretty(v: &Out, indent: &str, st: &Style) -> String {
    let inner = format!("{indent}{}", st.unit);
    match v {
        Out::Obj(m) if m.is_empty() => "{}".into(),
        Out::Arr(a) if a.is_empty() => "[]".into(),
        Out::Obj(m) => {
            let rows: Vec<String> = m.iter().map(|(k, x)| format!("{inner}{}: {}", quote(k), pretty(x, &inner, st))).collect();
            format!("{{{}{}{}{indent}}}", st.nl, rows.join(&format!(",{}", st.nl)), st.nl)
        }
        // A short array of plain values stays on one line: `["hook", "Stop"]`.
        Out::Arr(a) if a.iter().all(|x| !matches!(x, Out::Obj(_) | Out::Arr(_))) && a.iter().map(|x| pretty(x, "", st).len() + 2).sum::<usize>() <= 60 => {
            format!("[{}]", a.iter().map(|x| pretty(x, "", st)).collect::<Vec<_>>().join(", "))
        }
        Out::Arr(a) => {
            let rows: Vec<String> = a.iter().map(|x| format!("{inner}{}", pretty(x, &inner, st))).collect();
            format!("[{}{}{}{indent}]", st.nl, rows.join(&format!(",{}", st.nl)), st.nl)
        }
        Out::Str(s) => quote(s),
        Out::Num(n) => n.to_string(),
        Out::Bool(b) => b.to_string(),
    }
}

/// The leading white space of the line holding byte `at`.
fn line_indent(text: &str, at: usize) -> String {
    let line_start = text.get(..at).and_then(|t| t.rfind('\n')).map_or(0, |n| n + 1);
    text.get(line_start..).unwrap_or_default().chars().take_while(|c| *c == ' ' || *c == '\t').collect()
}

/// True when only white space lies between `a` and `b`.
fn blank(text: &str, a: usize, b: usize) -> bool {
    text.get(a..b).is_some_and(|t| t.chars().all(char::is_whitespace))
}

/// Applies one edit: replace bytes `a..b` with `with`.
fn splice(text: &str, a: usize, b: usize, with: &str) -> String {
    format!("{}{with}{}", text.get(..a).unwrap_or_default(), text.get(b..).unwrap_or_default())
}

/// Inserts an entry at the end of a container (object member `key: v`, or array element `v`),
/// as one contiguous insertion after the last entry's value.
#[must_use]
pub fn append(text: &str, container: &Node, key: Option<&str>, v: &Out, st: &Style) -> String {
    let (last_start, last_end) = match &container.v {
        V::Obj(m) => (m.last().map(|x| x.key_start), m.last().map(|x| x.value.end)),
        V::Arr(a) => (a.last().map(|x| x.start), a.last().map(|x| x.end)),
        _ => (None, None),
    };
    let entry = |indent: &str| match key {
        Some(k) => format!("{}: {}", quote(k), pretty(v, indent, st)),
        None => pretty(v, indent, st),
    };
    match (last_start, last_end) {
        (Some(ls), Some(le)) => {
            // Entries on their own lines take the last entry's indent; a one-line container
            // gets one unit more than its own line.
            let own_line = blank(text, text.get(..ls).and_then(|t| t.rfind('\n')).map_or(0, |n| n + 1), ls);
            let indent = if own_line { line_indent(text, ls) } else { format!("{}{}", line_indent(text, container.start), st.unit) };
            splice(text, le, le, &format!(",{}{indent}{}", st.nl, entry(&indent)))
        }
        _ => {
            let open = container.start + 1;
            let base = line_indent(text, container.start);
            if text.get(open..container.end.saturating_sub(1)).is_some_and(str::is_empty) {
                splice(text, open, open, &entry(&base))
            } else {
                let indent = format!("{base}{}", st.unit);
                splice(text, open, open, &format!("{}{indent}{}", st.nl, entry(&indent)))
            }
        }
    }
}

/// Removes entry `index` of a container (member or element) as one contiguous deletion: from the
/// comma after the entry before it to the end of the entry; for a first entry with others after
/// it, through to the next entry; for an only entry, with the white space before it.
#[must_use]
pub fn remove(text: &str, container: &Node, index: usize) -> String {
    let spans: Vec<(usize, usize)> = match &container.v {
        V::Obj(m) => m.iter().map(|x| (x.key_start, x.value.end)).collect(),
        V::Arr(a) => a.iter().map(|x| (x.start, x.end)).collect(),
        _ => return text.to_owned(),
    };
    let Some(&(s, e)) = spans.get(index) else { return text.to_owned() };
    if index > 0 {
        let prev_end = spans.get(index - 1).map_or(s, |p| p.1);
        let comma = text.get(prev_end..s).and_then(comma_at).map_or(prev_end, |n| prev_end + n);
        return splice(text, comma, e, "");
    }
    if let Some(&(next, _)) = spans.get(1) {
        return splice(text, s, next, "");
    }
    let open = container.start + 1;
    // An only entry with a trailing comma takes the comma with it.
    let close = container.end.saturating_sub(1);
    let tail = text.get(e..close).and_then(comma_at).map_or(e, |n| e + n + 1);
    if blank(text, open, s) { splice(text, open, tail, "") } else { splice(text, s, tail, "") }
}

/// The offset of the separating comma in the text between two entries (outside comments).
fn comma_at(t: &str) -> Option<usize> {
    let b = t.as_bytes();
    let mut i = 0;
    while let Some(&c) = b.get(i) {
        match c {
            b',' => return Some(i),
            b'/' if b.get(i + 1) == Some(&b'/') => {
                while b.get(i).is_some_and(|c| *c != b'\n') {
                    i += 1;
                }
            }
            b'/' if b.get(i + 1) == Some(&b'*') => {
                i += 2;
                while b.get(i).is_some() && !(b.get(i) == Some(&b'*') && b.get(i + 1) == Some(&b'/')) {
                    i += 1;
                }
                i += 2;
            }
            _ => i += 1,
        }
    }
    None
}

/// Replaces the value of an object member with `v`, keeping the member's key and place.
#[must_use]
pub fn replace_value(text: &str, member: &Member, v: &Out, st: &Style) -> String {
    let indent = line_indent(text, member.key_start);
    splice(text, member.value.start, member.value.end, &pretty(v, &indent, st))
}

/// The structure of a parsed value as an `Out`, for comparisons (numbers that are not integers
/// are kept as strings).
#[must_use]
pub fn to_out(n: &Node) -> Out {
    match &n.v {
        V::Obj(m) => Out::Obj(m.iter().map(|x| (x.key.clone(), to_out(&x.value))).collect()),
        V::Arr(a) => Out::Arr(a.iter().map(to_out).collect()),
        V::Str(s) => Out::Str(s.clone()),
        V::Num(s) => s.parse::<i64>().map_or_else(|_| Out::Str(s.clone()), Out::Num),
        V::Bool(b) => Out::Bool(*b),
        V::Null => Out::Str("null".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn out() -> Out {
        Out::Obj(vec![("a".into(), Out::Num(1)), ("b".into(), Out::Arr(vec![Out::Str("x".into())]))])
    }

    #[test]
    fn append_then_remove_is_exact() {
        let cases = [
            "{}",
            "{\n}\n",
            "{ // only a comment\n}",
            "{\n  \"k\": 1\n}\n",
            "{\n  \"k\": 1, // trailing comment\n}\n",
            "{\r\n\t\"k\": [1, 2],\r\n\t/* block */ \"z\": {}\r\n}\r\n",
            "\u{feff}{\"k\":1}",
        ];
        for c in cases {
            let root = parse(c).unwrap();
            let st = Style::of(c);
            let added = append(c, &root, Some("hooks"), &out(), &st);
            let r2 = parse(&added).unwrap();
            let idx = r2.members().unwrap().iter().position(|m| m.key == "hooks").unwrap();
            assert_eq!(remove(&added, &r2, idx), c, "case {c:?} gave {added:?}");
        }
    }

    #[test]
    fn arrays_append_then_remove() {
        for c in ["[]", "[\n]", "[1]", "[\n  1,\n  2\n]", "[1, /* c */ 2,]"] {
            let root = parse(c).unwrap();
            let st = Style::of(c);
            let added = append(c, &root, None, &out(), &st);
            let r2 = parse(&added).unwrap();
            let n = r2.items().unwrap().len();
            assert_eq!(remove(&added, &r2, n - 1), c, "case {c:?} gave {added:?}");
        }
    }

    #[test]
    fn refuses_bad_json() {
        for c in ["", "{", "{\"a\":}", "{\"a\":1}}", "{'a':1}", "{\"a\":1,\"a\":2}", "/* x"] {
            assert!(parse(c).is_err(), "{c:?} parsed");
        }
    }
}
