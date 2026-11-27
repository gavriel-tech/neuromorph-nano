//! JSON, JSON Lines and notebooks. JSON and JSON Lines keep the file as their derived text; their
//! blocks follow the data: each member of the top object (or element of the top array) is a
//! `Record` labelled by its key path, and a member larger than `LARGE` lines becomes a heading
//! with its own members as records below it. The scanner tracks lines and is tolerant: comments
//! and trailing commas are read past (a strict reader, `nmjson`, is not needed for structure). A
//! JSON Lines record is one line, labelled by its id field. A notebook's derived text is its cells
//! in Markdown form: Markdown cells as written, code cells and their text output fenced.
use nmtype::doc::{Block, BlockKind};

/// A member at least this many lines long is entered: it becomes a heading over its own members.
pub const LARGE: usize = 40;

/// One value: its first and last line, and for an object or array its members (key or index).
#[derive(Debug, Default)]
struct Val {
    first: usize,
    last: usize,
    members: Vec<(String, Val)>,
    /// For a string or number: its text; used for id fields.
    scalar: String,
}

struct Scan<'a> {
    b: &'a [u8],
    i: usize,
    line: usize,
}

impl Scan<'_> {
    fn skip(&mut self) {
        while let Some(&c) = self.b.get(self.i) {
            match c {
                b'\n' => {
                    self.line += 1;
                    self.i += 1;
                }
                b' ' | b'\t' | b'\r' | b',' => self.i += 1,
                b'/' if self.b.get(self.i + 1) == Some(&b'/') => {
                    while self.b.get(self.i).is_some_and(|c| *c != b'\n') {
                        self.i += 1;
                    }
                }
                b'/' if self.b.get(self.i + 1) == Some(&b'*') => {
                    self.i += 2;
                    while self.i < self.b.len() && !(self.b.get(self.i) == Some(&b'*') && self.b.get(self.i + 1) == Some(&b'/')) {
                        if self.b.get(self.i) == Some(&b'\n') {
                            self.line += 1;
                        }
                        self.i += 1;
                    }
                    self.i += 2;
                }
                _ => return,
            }
        }
    }
    fn string(&mut self) -> String {
        let mut out = Vec::new();
        self.i += 1;
        while let Some(&c) = self.b.get(self.i) {
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    if let Some(&e) = self.b.get(self.i) {
                        out.push(e);
                        self.i += 1;
                    }
                }
                b'\n' => {
                    self.line += 1;
                    out.push(c);
                }
                _ => out.push(c),
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }
    fn value(&mut self, depth: usize) -> Val {
        self.skip();
        let first = self.line;
        let mut v = Val { first, last: first, ..Val::default() };
        match self.b.get(self.i).copied() {
            Some(b'{') if depth < 64 => {
                self.i += 1;
                loop {
                    self.skip();
                    match self.b.get(self.i).copied() {
                        None => break,
                        Some(b'}') => {
                            self.i += 1;
                            break;
                        }
                        Some(b'"') => {
                            let key_line = self.line;
                            let key = self.string();
                            self.skip();
                            if self.b.get(self.i) == Some(&b':') {
                                self.i += 1;
                            }
                            let mut m = self.value(depth + 1);
                            m.first = key_line;
                            v.members.push((key, m));
                        }
                        Some(_) => {
                            // A bare key (tolerant): read to the colon.
                            let key_line = self.line;
                            let start = self.i;
                            while self.b.get(self.i).is_some_and(|c| *c != b':' && *c != b'}' && *c != b'\n') {
                                self.i += 1;
                            }
                            let key = String::from_utf8_lossy(self.b.get(start..self.i).unwrap_or(&[])).trim().to_owned();
                            if self.b.get(self.i) == Some(&b':') {
                                self.i += 1;
                                let mut m = self.value(depth + 1);
                                m.first = key_line;
                                v.members.push((key, m));
                            }
                        }
                    }
                }
            }
            Some(b'[') if depth < 64 => {
                self.i += 1;
                let mut k = 0;
                loop {
                    self.skip();
                    match self.b.get(self.i).copied() {
                        None => break,
                        Some(b']') => {
                            self.i += 1;
                            break;
                        }
                        Some(_) => {
                            let before = self.i;
                            let m = self.value(depth + 1);
                            v.members.push((format!("[{k}]"), m));
                            k += 1;
                            if self.i == before {
                                self.i += 1;
                            }
                        }
                    }
                }
            }
            Some(b'"') => v.scalar = self.string(),
            Some(_) => {
                let start = self.i;
                while self.b.get(self.i).is_some_and(|c| !matches!(c, b',' | b'}' | b']' | b'\n' | b' ' | b'\t' | b'\r')) {
                    self.i += 1;
                }
                v.scalar = String::from_utf8_lossy(self.b.get(start..self.i).unwrap_or(&[])).into_owned();
            }
            None => {}
        }
        v.last = self.line.max(first);
        v
    }
}

/// The id of a record: the value of its `id`, first `*_id`, `key`, `name` or `title` member.
fn id_of(v: &Val) -> Option<String> {
    let find = |p: &dyn Fn(&str) -> bool| v.members.iter().find(|(k, m)| p(k) && !m.scalar.is_empty()).map(|(_, m)| m.scalar.clone());
    find(&|k| k == "id").or_else(|| find(&|k| k.ends_with("_id") || k.ends_with("Id"))).or_else(|| find(&|k| matches!(k, "key" | "name" | "title")))
}

fn records(path: &str, v: &Val, level: u8, out: &mut Vec<Block>) {
    for (key, m) in &v.members {
        let label = if key.starts_with('[') {
            let at = if path.is_empty() { key.clone() } else { format!("{path}{key}") };
            id_of(m).map_or(at.clone(), |id| format!("{at} {id}"))
        } else if path.is_empty() {
            key.clone()
        } else {
            format!("{path}.{key}")
        };
        if m.last + 1 - m.first >= LARGE && !m.members.is_empty() && level < 3 {
            out.push(Block::new(BlockKind::Heading(level), m.first, m.first, &label));
            records(&label, m, level + 1, out);
        } else {
            out.push(Block::new(BlockKind::Record, m.first, m.last, &label));
        }
    }
}

/// Blocks of a JSON file over its own lines.
#[must_use]
pub fn blocks(text: &str) -> Vec<Block> {
    let n = text.lines().count();
    let mut s = Scan { b: text.as_bytes(), i: 0, line: 1 };
    let top = s.value(0);
    let mut out = Vec::new();
    records("", &top, 1, &mut out);
    // Keep blocks in order and apart (a member starting on the line another ends is left out).
    out.sort_by_key(|b| b.first);
    let mut kept: Vec<Block> = Vec::new();
    for b in out {
        if kept.last().is_none_or(|l: &Block| b.first > l.last || (l.kind.level().is_some() && b.first > l.first)) {
            kept.push(b);
        }
    }
    crate::md::close_gaps(&mut kept, n);
    nmtype::doc::link_parents(&mut kept);
    kept
}

/// Blocks of a JSON Lines file: one `Record` per line, labelled by its id field.
#[must_use]
pub fn lines_blocks(text: &str) -> Vec<Block> {
    let n = text.lines().count();
    let mut out = Vec::new();
    for (i, l) in text.lines().enumerate() {
        if l.trim().is_empty() {
            continue;
        }
        let mut s = Scan { b: l.as_bytes(), i: 0, line: 1 };
        let v = s.value(0);
        let label = id_of(&v).unwrap_or_else(|| format!("line {}", i + 1));
        out.push(Block::new(BlockKind::Record, i + 1, i + 1, &label));
    }
    crate::md::close_gaps(&mut out, n);
    out
}

fn cell_source(c: &nmjson::Value) -> String {
    match c.get("source") {
        Ok(nmjson::Value::Array(parts)) => parts.iter().filter_map(|p| p.string().ok()).collect::<String>(),
        Ok(nmjson::Value::String(s)) => s.clone(),
        _ => String::new(),
    }
}

/// A notebook's derived text: Markdown cells as written, code cells fenced with the notebook's
/// language, and each code cell's text output (stream or `text/plain`, at most 20 lines) fenced as
/// `output`.
/// # Errors
/// The file is not a JSON notebook.
pub fn notebook(text: &str) -> Result<String, &'static str> {
    let v = nmjson::parse(text.as_bytes()).map_err(|_| "notebook: not valid JSON")?;
    let cells = v.get("cells").and_then(nmjson::Value::array).map_err(|_| "notebook: no cells")?;
    let lang = v
        .get("metadata")
        .and_then(|m| m.get("language_info"))
        .and_then(|l| l.get("name"))
        .and_then(nmjson::Value::string)
        .unwrap_or("python")
        .to_owned();
    let mut parts: Vec<String> = Vec::new();
    for c in cells {
        let kind = c.get("cell_type").and_then(nmjson::Value::string).unwrap_or("");
        let src = cell_source(c);
        let src = src.trim_end_matches('\n');
        match kind {
            "markdown" | "raw" if !src.trim().is_empty() => parts.push(src.to_owned()),
            "code" => {
                parts.push(format!("```{lang}\n{src}\n```"));
                let mut shown: Vec<String> = Vec::new();
                for o in c.get("outputs").and_then(nmjson::Value::array).unwrap_or(&[]) {
                    let t = match o.get("text") {
                        Ok(nmjson::Value::Array(p)) => p.iter().filter_map(|x| x.string().ok()).collect::<String>(),
                        Ok(nmjson::Value::String(s)) => s.clone(),
                        _ => match o.get("data").and_then(|d| d.get("text/plain")) {
                            Ok(nmjson::Value::Array(p)) => p.iter().filter_map(|x| x.string().ok()).collect::<String>(),
                            Ok(nmjson::Value::String(s)) => s.clone(),
                            _ => String::new(),
                        },
                    };
                    shown.extend(t.lines().map(str::to_owned));
                }
                if !shown.is_empty() {
                    shown.truncate(20);
                    parts.push(format!("```output\n{}\n```", shown.join("\n")));
                }
            }
            _ => {}
        }
    }
    let mut out = parts.join("\n\n");
    if !out.is_empty() {
        out.push('\n');
    }
    Ok(out)
}

pub(crate) fn check() -> Result<(), String> {
    let mut big = String::from("{\n  \"name\": \"pump\", // tolerant\n  \"parts\": [\n");
    for i in 0..20 {
        big.push_str(&format!("    {{\"id\": \"P-{i}\",\n     \"qty\": {i}}},\n"));
    }
    big.push_str("  ],\n}\n");
    let b = blocks(&big);
    nmtype::doc::check_cover(&b, big.lines().count()).map_err(|e| format!("nmdoc.json: {e}"))?;
    let labels: Vec<&str> = b.iter().map(|x| x.label.as_str()).collect();
    if labels.first() != Some(&"name") || labels.get(1) != Some(&"parts") || labels.get(2) != Some(&"parts[0] P-0") || b.len() != 22 {
        return Err(format!("nmdoc.json: labels {labels:?}"));
    }
    let jl = lines_blocks("{\"order_id\": \"O-1\", \"qty\": 2}\n{\"qty\": 3}\n");
    if jl.iter().map(|x| x.label.as_str()).collect::<Vec<_>>() != ["O-1", "line 2"] {
        return Err("nmdoc.json: json lines labels".into());
    }
    let nb = "{\"cells\":[{\"cell_type\":\"markdown\",\"source\":[\"# Fit\\n\",\"Text.\"]},{\"cell_type\":\"code\",\"source\":\"x = 1\",\"outputs\":[{\"name\":\"stdout\",\"text\":[\"1\\n\"]}]}],\"metadata\":{}}";
    if notebook(nb).as_deref() != Ok("# Fit\nText.\n\n```python\nx = 1\n```\n\n```output\n1\n```\n") {
        return Err(format!("nmdoc.json: notebook {:?}", notebook(nb)));
    }
    Ok(())
}
