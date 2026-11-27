//! PDF objects and their syntax: numbers, literal and hexadecimal strings, names with `#xx`
//! escapes, arrays, dictionaries, indirect references `n g R`, and the keywords around them. The
//! same tokenizer reads content streams, where keywords are operators.

/// One PDF object.
#[derive(Clone, Debug, PartialEq)]
pub enum Obj {
    Null,
    Bool(bool),
    Int(i64),
    Real(f64),
    Str(Vec<u8>),
    Name(String),
    Array(Vec<Obj>),
    Dict(Dict),
    /// A stream: its dictionary and its bytes as stored (still filtered, still encrypted).
    Stream(Box<Stream>),
    Ref(u32, u16),
}

/// A dictionary, in file order (dictionaries are small; a linear search is enough).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Dict(pub Vec<(String, Obj)>);

/// A stream object.
#[derive(Clone, Debug, PartialEq)]
pub struct Stream {
    pub dict: Dict,
    pub data: Vec<u8>,
}

impl Dict {
    /// The value under `key`.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Obj> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    /// The name under `key`.
    #[must_use]
    pub fn name(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Obj::as_name)
    }
    /// The number under `key`.
    #[must_use]
    pub fn num(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(Obj::as_num)
    }
    /// The integer under `key`.
    #[must_use]
    pub fn int(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(Obj::as_int)
    }
    pub fn set(&mut self, key: &str, v: Obj) {
        if let Some(slot) = self.0.iter_mut().find(|(k, _)| k == key) {
            slot.1 = v;
        } else {
            self.0.push((key.to_owned(), v));
        }
    }
}

impl Obj {
    #[must_use]
    pub fn as_name(&self) -> Option<&str> {
        if let Self::Name(n) = self { Some(n) } else { None }
    }
    #[must_use]
    #[allow(clippy::cast_precision_loss, reason = "PDF numbers are small")]
    pub fn as_num(&self) -> Option<f64> {
        match self {
            Self::Int(i) => Some(*i as f64),
            Self::Real(r) => Some(*r),
            _ => None,
        }
    }
    #[must_use]
    #[allow(clippy::cast_possible_truncation, reason = "a real used where an integer is meant")]
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(*i),
            Self::Real(r) if r.is_finite() => Some(r.round() as i64),
            _ => None,
        }
    }
    #[must_use]
    pub fn as_dict(&self) -> Option<&Dict> {
        match self {
            Self::Dict(d) => Some(d),
            Self::Stream(s) => Some(&s.dict),
            _ => None,
        }
    }
    #[must_use]
    pub fn as_array(&self) -> Option<&[Obj]> {
        if let Self::Array(a) = self { Some(a) } else { None }
    }
    #[must_use]
    pub fn as_str(&self) -> Option<&[u8]> {
        if let Self::Str(s) = self { Some(s) } else { None }
    }
}

/// True for the PDF white-space bytes.
#[must_use]
pub fn is_white(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0C | 0)
}

/// True for the PDF delimiter bytes.
#[must_use]
pub fn is_delim(b: u8) -> bool {
    matches!(b, b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%')
}

/// One token.
#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Obj(Obj),
    /// A bare word: `obj`, `R`, `stream`, or a content stream operator.
    Word(Vec<u8>),
    ArrayOpen,
    ArrayClose,
    DictOpen,
    DictClose,
}

/// A reader of tokens and objects over a byte slice.
pub struct Lexer<'a> {
    pub data: &'a [u8],
    pub pos: usize,
}

/// Deepest nesting of arrays and dictionaries read.
const MAX_NEST: usize = 64;

impl<'a> Lexer<'a> {
    #[must_use]
    pub fn new(data: &'a [u8], pos: usize) -> Self {
        Self { data, pos }
    }

    fn peek(&self) -> Option<u8> {
        self.data.get(self.pos).copied()
    }

    /// Skip white space and comments.
    pub fn skip_white(&mut self) {
        while let Some(b) = self.peek() {
            if is_white(b) {
                self.pos += 1;
            } else if b == b'%' {
                while let Some(c) = self.peek() {
                    if c == b'\n' || c == b'\r' {
                        break;
                    }
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    /// The next token, or `None` at the end of the data.
    pub fn token(&mut self) -> Option<Tok> {
        self.skip_white();
        let b = self.peek()?;
        match b {
            b'[' => {
                self.pos += 1;
                Some(Tok::ArrayOpen)
            }
            b']' => {
                self.pos += 1;
                Some(Tok::ArrayClose)
            }
            b'<' if self.data.get(self.pos + 1) == Some(&b'<') => {
                self.pos += 2;
                Some(Tok::DictOpen)
            }
            b'>' if self.data.get(self.pos + 1) == Some(&b'>') => {
                self.pos += 2;
                Some(Tok::DictClose)
            }
            b'<' => Some(Tok::Obj(Obj::Str(self.hex_string()))),
            b'(' => Some(Tok::Obj(Obj::Str(self.literal_string()))),
            b'/' => Some(Tok::Obj(Obj::Name(self.name()))),
            b'+' | b'-' | b'.' | b'0'..=b'9' => Some(self.number()),
            b')' | b'>' | b'{' | b'}' => {
                // A stray delimiter: taken as a word of its own so reading moves on.
                self.pos += 1;
                Some(Tok::Word(vec![b]))
            }
            _ => {
                let start = self.pos;
                while let Some(c) = self.peek() {
                    if is_white(c) || is_delim(c) {
                        break;
                    }
                    self.pos += 1;
                }
                let w = self.data.get(start..self.pos).unwrap_or(&[]).to_vec();
                Some(match w.as_slice() {
                    b"true" => Tok::Obj(Obj::Bool(true)),
                    b"false" => Tok::Obj(Obj::Bool(false)),
                    b"null" => Tok::Obj(Obj::Null),
                    _ => Tok::Word(w),
                })
            }
        }
    }

    fn number(&mut self) -> Tok {
        let start = self.pos;
        self.pos += 1;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == b'.' || c == b'-' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let s = std::str::from_utf8(self.data.get(start..self.pos).unwrap_or(&[])).unwrap_or("0");
        // Some writers put a second sign inside a number (`--5`, `4-2`); read what parses.
        let s = s.trim_start_matches('+');
        if !s.contains('.')
            && let Ok(i) = s.parse::<i64>()
        {
            return Tok::Obj(Obj::Int(i));
        }
        if let Ok(r) = s.parse::<f64>() {
            return Tok::Obj(Obj::Real(r));
        }
        let clean: String = s.chars().enumerate().filter(|(i, c)| c.is_ascii_digit() || *c == '.' || (*i == 0 && *c == '-')).map(|x| x.1).collect();
        Tok::Obj(Obj::Real(clean.parse::<f64>().unwrap_or(0.0)))
    }

    fn name(&mut self) -> String {
        self.pos += 1;
        let mut out = Vec::new();
        while let Some(c) = self.peek() {
            if is_white(c) || is_delim(c) {
                break;
            }
            if c == b'#'
                && let Some(h) = self.data.get(self.pos + 1..self.pos + 3).and_then(|h| std::str::from_utf8(h).ok()).and_then(|h| u8::from_str_radix(h, 16).ok())
            {
                out.push(h);
                self.pos += 3;
                continue;
            }
            out.push(c);
            self.pos += 1;
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    fn hex_string(&mut self) -> Vec<u8> {
        self.pos += 1;
        let mut out = Vec::new();
        let mut half: Option<u8> = None;
        while let Some(c) = self.peek() {
            self.pos += 1;
            if c == b'>' {
                break;
            }
            let v = match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                b'A'..=b'F' => c - b'A' + 10,
                _ => continue,
            };
            match half.take() {
                Some(h) => out.push(h << 4 | v),
                None => half = Some(v),
            }
        }
        if let Some(h) = half {
            out.push(h << 4);
        }
        out
    }

    fn literal_string(&mut self) -> Vec<u8> {
        self.pos += 1;
        let mut out = Vec::new();
        let mut depth = 1;
        while let Some(c) = self.peek() {
            self.pos += 1;
            match c {
                b'(' => {
                    depth += 1;
                    out.push(c);
                }
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    out.push(c);
                }
                b'\\' => {
                    let Some(n) = self.peek() else { break };
                    self.pos += 1;
                    match n {
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'0'..=b'7' => {
                            let mut v = u32::from(n - b'0');
                            for _ in 0..2 {
                                match self.peek() {
                                    Some(d @ b'0'..=b'7') => {
                                        v = v * 8 + u32::from(d - b'0');
                                        self.pos += 1;
                                    }
                                    _ => break,
                                }
                            }
                            out.push((v & 0xFF) as u8);
                        }
                        b'\r' => {
                            if self.peek() == Some(b'\n') {
                                self.pos += 1;
                            }
                        }
                        b'\n' => {}
                        _ => out.push(n),
                    }
                }
                _ => out.push(c),
            }
        }
        out
    }

    /// The next object: a direct object, or `n g R` as a reference. Bare words come back as
    /// `Err(word)` so callers can see `obj`, `endobj`, `stream` and operators.
    pub fn object(&mut self) -> Option<Result<Obj, Vec<u8>>> {
        self.object_at(0)
    }

    fn object_at(&mut self, depth: usize) -> Option<Result<Obj, Vec<u8>>> {
        let t = self.token()?;
        Some(match t {
            Tok::Obj(Obj::Int(n)) => {
                // `n g R`: look two tokens ahead and come back when it is not a reference.
                let save = self.pos;
                if let (Some(Tok::Obj(Obj::Int(g))), Some(Tok::Word(w))) = (self.token(), self.token())
                    && w == b"R"
                    && n >= 0
                {
                    return Some(Ok(Obj::Ref(u32::try_from(n).unwrap_or(0), u16::try_from(g).unwrap_or(0))));
                }
                self.pos = save;
                Ok(Obj::Int(n))
            }
            Tok::Obj(o) => Ok(o),
            Tok::ArrayOpen => {
                let mut v = Vec::new();
                if depth > MAX_NEST {
                    return Some(Ok(Obj::Array(v)));
                }
                loop {
                    self.skip_white();
                    match self.peek() {
                        None => break,
                        Some(b']') => {
                            self.pos += 1;
                            break;
                        }
                        _ => {}
                    }
                    match self.object_at(depth + 1) {
                        Some(Ok(o)) => v.push(o),
                        Some(Err(w)) if w.as_slice() == b"endobj" || w.as_slice() == b"stream" => break,
                        Some(Err(_)) => {}
                        None => break,
                    }
                }
                Ok(Obj::Array(v))
            }
            Tok::DictOpen => Ok(Obj::Dict(self.dict_body(depth))),
            Tok::Word(w) => Err(w),
            Tok::ArrayClose | Tok::DictClose => Err(Vec::new()),
        })
    }

    /// The entries of a dictionary after its `<<`, up to `>>`.
    fn dict_body(&mut self, depth: usize) -> Dict {
        let mut d = Dict::default();
        if depth > MAX_NEST {
            return d;
        }
        loop {
            self.skip_white();
            match self.peek() {
                None => break,
                Some(b'>') if self.data.get(self.pos + 1) == Some(&b'>') => {
                    self.pos += 2;
                    break;
                }
                Some(b'/') => {}
                Some(_) => {
                    // Not a key: skip one token (a broken entry) and go on.
                    match self.token() {
                        Some(Tok::Word(w)) if w.as_slice() == b"endobj" || w.as_slice() == b"stream" => {
                            self.pos -= w.len();
                            break;
                        }
                        None => break,
                        _ => continue,
                    }
                }
            }
            let Some(Tok::Obj(Obj::Name(key))) = self.token() else { break };
            self.skip_white();
            if self.peek() == Some(b'>') && self.data.get(self.pos + 1) == Some(&b'>') {
                // A key with no value before `>>`.
                d.0.push((key, Obj::Null));
                continue;
            }
            match self.object_at(depth + 1) {
                Some(Ok(v)) => d.0.push((key, v)),
                Some(Err(w)) => {
                    if w.as_slice() == b"endobj" || w.as_slice() == b"stream" {
                        self.pos -= w.len();
                        break;
                    }
                }
                None => break,
            }
        }
        d
    }
}

/// Parse one direct object from bytes (used for object stream members and tests).
#[must_use]
pub fn parse(data: &[u8]) -> Option<Obj> {
    Lexer::new(data, 0).object().and_then(Result::ok)
}

pub(crate) fn check() -> Result<(), String> {
    let o = parse(b"<< /Type /Page /Kids [1 0 R 2 0 R] /N -3.5 /S (a\\(b\\)c\\101) /H <48 65 6c6c 6f> /K#20x 7 /E [] >>")
        .ok_or("nmdoc.pdf.lex: no object")?;
    let d = o.as_dict().ok_or("nmdoc.pdf.lex: not a dictionary")?;
    let kids = d.get("Kids").and_then(Obj::as_array).unwrap_or(&[]);
    if d.name("Type") != Some("Page")
        || kids != [Obj::Ref(1, 0), Obj::Ref(2, 0)]
        || d.num("N") != Some(-3.5)
        || d.get("S").and_then(Obj::as_str) != Some(b"a(b)cA")
        || d.get("H").and_then(Obj::as_str) != Some(b"Hello")
        || d.int("K x") != Some(7)
        || d.get("E").and_then(Obj::as_array).is_none_or(|a| !a.is_empty())
    {
        return Err(format!("nmdoc.pdf.lex: parsed {o:?}"));
    }
    // Broken input never panics and still yields what it can.
    for bad in [&b"<< /A [1 2 (unclosed"[..], b"[[[[", b"<< /A >>", b"(\\", b"<4", b"/", b"<< 5 /B 2 >>"] {
        let _ = parse(bad);
    }
    if parse(b"<< 5 /B 2 >>").and_then(|o| o.as_dict().and_then(|d| d.int("B"))) != Some(2) {
        return Err("nmdoc.pdf.lex: a broken entry stopped the dictionary".into());
    }
    Ok(())
}
