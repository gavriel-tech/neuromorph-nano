//! One PDF file opened in memory: its cross-reference (classic tables, cross-reference streams
//! with their predictors, hybrid files, incremental updates chained by `/Prev` where the newest
//! entry wins), objects inside object streams, a repair pass that scans for `n g obj` when the
//! table is missing or wrong, and decryption of strings and streams as objects are loaded.
use super::crypt::Crypt;
use super::filters;
use super::lex::{Dict, Lexer, Obj, Stream};
use std::cell::RefCell;
use std::collections::HashMap;

/// Where an object is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Entry {
    /// At a byte offset, with its generation.
    At(usize, u16),
    /// Member `index` of object stream `stream`.
    InStream(u32, usize),
}

/// A decoded object stream: its bytes, its members (number, offset) and the offset of the first.
type ObjStream = (Vec<u8>, Vec<(u32, usize)>, usize);

/// An opened file.
pub struct File<'a> {
    pub data: &'a [u8],
    xref: HashMap<u32, Entry>,
    pub trailer: Dict,
    pub crypt: Option<Crypt>,
    /// Number of the encryption dictionary (never decrypted itself).
    enc_num: Option<u32>,
    cache: RefCell<HashMap<u32, Obj>>,
    streams: RefCell<HashMap<u32, ObjStream>>,
    /// True when the cross-reference had to be rebuilt by scanning the file.
    pub repaired: bool,
}

/// Why a file was not opened.
pub const NOT_PDF: &str = "pdf: no PDF header";
pub const NO_ROOT: &str = "pdf: no document catalog found";

/// Deepest chain of references followed while resolving one value.
const MAX_RESOLVE: usize = 32;

fn find_last(data: &[u8], pat: &[u8]) -> Option<usize> {
    if data.len() < pat.len() {
        return None;
    }
    (0..=data.len() - pat.len()).rev().find(|&i| data.get(i..i + pat.len()) == Some(pat))
}

fn find_from(data: &[u8], from: usize, pat: &[u8]) -> Option<usize> {
    if pat.is_empty() || data.len() < pat.len() {
        return None;
    }
    (from..=data.len() - pat.len()).find(|&i| data.get(i..i + pat.len()) == Some(pat))
}

impl<'a> File<'a> {
    /// Open a file: cross-reference, trailer, encryption.
    /// # Errors
    /// `NOT_PDF`, `NO_ROOT`, or the encryption reason (`crypt::NEEDS_PASSWORD`, `crypt::UNSUPPORTED`).
    pub fn open(data: &'a [u8]) -> Result<Self, &'static str> {
        let head = data.get(..data.len().min(1024)).unwrap_or(&[]);
        let shift = find_from(head, 0, b"%PDF-").ok_or(NOT_PDF)?;
        let mut f = Self {
            data,
            xref: HashMap::new(),
            trailer: Dict::default(),
            crypt: None,
            enc_num: None,
            cache: RefCell::new(HashMap::new()),
            streams: RefCell::new(HashMap::new()),
            repaired: false,
        };
        let ok = f.read_xref_chain(shift);
        if !ok || f.root().is_none() || !f.offsets_ok() {
            f.repair();
        }
        if f.root().is_none() {
            return Err(NO_ROOT);
        }
        if let Some(enc) = f.trailer.get("Encrypt").cloned() {
            if let Obj::Ref(n, _) = enc {
                f.enc_num = Some(n);
            }
            let enc = f.resolve(&enc);
            let id = f.trailer.get("ID").and_then(Obj::as_array).and_then(|a| a.first()).and_then(Obj::as_str).map(<[u8]>::to_vec).unwrap_or_default();
            if let Some(d) = enc.as_dict() {
                f.crypt = Some(Crypt::open(d, &id)?);
                f.cache.borrow_mut().clear();
                f.streams.borrow_mut().clear();
            }
        }
        Ok(f)
    }

    /// The document catalog.
    #[must_use]
    pub fn root(&self) -> Option<Dict> {
        let r = self.trailer.get("Root")?;
        let d = self.resolve(r);
        d.as_dict().filter(|d| d.name("Type") != Some("XRef")).cloned()
    }

    /// Read the cross-reference sections from `startxref` back through `/Prev`.
    fn read_xref_chain(&mut self, shift: usize) -> bool {
        let tail_from = self.data.len().saturating_sub(2048);
        let Some(at) = find_last(self.data.get(tail_from..).unwrap_or(&[]), b"startxref").map(|i| i + tail_from) else { return false };
        let mut lx = Lexer::new(self.data, at + 9);
        let Some(Ok(Obj::Int(off))) = lx.object() else { return false };
        let mut next = usize::try_from(off).ok();
        let mut seen = Vec::new();
        let mut first = true;
        while let Some(off) = next {
            if seen.contains(&off) || seen.len() > 64 {
                break;
            }
            seen.push(off);
            // A file with bytes before its header has every offset moved by that much.
            let Some(trailer) = self.read_section(off).or_else(|| self.read_section(off + shift)) else { return !first };
            if let Some(x) = trailer.int("XRefStm").and_then(|x| usize::try_from(x).ok()) {
                let _ = self.read_section(x);
            }
            for (k, v) in &trailer.0 {
                if self.trailer.get(k).is_none() && k != "Prev" && k != "XRefStm" {
                    self.trailer.0.push((k.clone(), v.clone()));
                }
            }
            next = trailer.int("Prev").and_then(|p| usize::try_from(p).ok());
            first = false;
        }
        true
    }

    /// One cross-reference section at `off` (a table or a stream); returns its trailer.
    fn read_section(&mut self, off: usize) -> Option<Dict> {
        let mut lx = Lexer::new(self.data, off);
        lx.skip_white();
        if self.data.get(lx.pos..lx.pos + 4) == Some(b"xref") {
            lx.pos += 4;
            return self.read_table(&mut lx);
        }
        // A cross-reference stream: `n g obj << /Type /XRef ... >> stream`.
        let (_, obj) = self.parse_at(off)?;
        let Obj::Stream(s) = obj else { return None };
        if s.dict.name("Type") != Some("XRef") {
            return None;
        }
        let data = filters::decode(&s.dict, &s.data).ok()?;
        let w: Vec<usize> = s.dict.get("W")?.as_array()?.iter().map(|x| usize::try_from(x.as_int().unwrap_or(0)).unwrap_or(0).min(8)).collect();
        let (w0, w1, w2) = (*w.first()?, *w.get(1)?, *w.get(2)?);
        let size = s.dict.int("Size").unwrap_or(0);
        let index: Vec<i64> = match s.dict.get("Index").and_then(Obj::as_array) {
            Some(a) => a.iter().filter_map(Obj::as_int).collect(),
            None => vec![0, size],
        };
        let row = w0 + w1 + w2;
        if row == 0 {
            return None;
        }
        let field = |b: &[u8]| b.iter().fold(0u64, |acc, x| acc << 8 | u64::from(*x));
        let mut at = 0;
        for pair in index.chunks(2) {
            let (Some(&start), Some(&count)) = (pair.first(), pair.get(1)) else { break };
            for i in 0..count.clamp(0, 1 << 22) {
                let Some(r) = data.get(at..at + row) else { break };
                at += row;
                let kind = if w0 == 0 { 1 } else { field(&r[..w0]) };
                let a = field(&r[w0..w0 + w1]);
                let b = field(&r[w0 + w1..]);
                let num = u32::try_from(start + i).unwrap_or(0);
                if self.xref.contains_key(&num) {
                    continue;
                }
                match kind {
                    1 => {
                        self.xref.insert(num, Entry::At(usize::try_from(a).unwrap_or(0), u16::try_from(b).unwrap_or(0)));
                    }
                    2 => {
                        self.xref.insert(num, Entry::InStream(u32::try_from(a).unwrap_or(0), usize::try_from(b).unwrap_or(0)));
                    }
                    _ => {
                        // A free entry hides older entries for the same number.
                        self.xref.insert(num, Entry::At(0, u16::MAX));
                    }
                }
            }
        }
        Some(s.dict)
    }

    fn read_table(&mut self, lx: &mut Lexer<'_>) -> Option<Dict> {
        loop {
            let save = lx.pos;
            match lx.object() {
                Some(Ok(Obj::Int(start))) => {
                    let Some(Ok(Obj::Int(count))) = lx.object() else { return None };
                    lx.skip_white();
                    for i in 0..count.clamp(0, 1 << 22) {
                        // Each entry: offset, generation, `n` or `f`; read as tokens, so odd line
                        // ends and short entries still parse.
                        let (Some(Ok(Obj::Int(off))), Some(Ok(Obj::Int(gen_)))) = (lx.object(), lx.object()) else { return None };
                        let kind = match lx.token() {
                            Some(super::lex::Tok::Word(w)) => w,
                            _ => return None,
                        };
                        let num = u32::try_from(start + i).unwrap_or(0);
                        if self.xref.contains_key(&num) {
                            continue;
                        }
                        let e = if kind.as_slice() == b"n" && off > 0 {
                            Entry::At(usize::try_from(off).unwrap_or(0), u16::try_from(gen_).unwrap_or(0))
                        } else {
                            Entry::At(0, u16::MAX)
                        };
                        self.xref.insert(num, e);
                    }
                }
                Some(Err(w)) if w.as_slice() == b"trailer" => {
                    return match lx.object() {
                        Some(Ok(Obj::Dict(d))) => Some(d),
                        _ => None,
                    };
                }
                _ => {
                    lx.pos = save;
                    return None;
                }
            }
        }
    }

    /// True when a sample of the table's offsets point at `n g obj` headers.
    fn offsets_ok(&self) -> bool {
        let mut checked = 0;
        for (num, e) in &self.xref {
            if let Entry::At(off, g) = e
                && *g != u16::MAX
            {
                match self.header_at(*off) {
                    Some((n, _, _)) if n == *num => {}
                    _ => return false,
                }
                checked += 1;
                if checked >= 16 {
                    break;
                }
            }
        }
        true
    }

    /// `n g obj` at `off`: (num, gen, position after the keyword).
    fn header_at(&self, off: usize) -> Option<(u32, u16, usize)> {
        let mut lx = Lexer::new(self.data, off);
        let (Some(super::lex::Tok::Obj(Obj::Int(n))), Some(super::lex::Tok::Obj(Obj::Int(g))), Some(super::lex::Tok::Word(w))) =
            (lx.token(), lx.token(), lx.token())
        else {
            return None;
        };
        (w.as_slice() == b"obj").then(|| (u32::try_from(n).unwrap_or(0), u16::try_from(g).unwrap_or(0), lx.pos))
    }

    /// Rebuild the cross-reference by scanning for `n g obj` (the last one wins) and the trailer
    /// from the last `trailer` dictionary or cross-reference stream, else the catalog found.
    fn repair(&mut self) {
        self.repaired = true;
        self.xref.clear();
        self.cache.borrow_mut().clear();
        let d = self.data;
        let mut i = 0;
        let mut found: Vec<(u32, usize, u16)> = Vec::new();
        while let Some(at) = find_from(d, i, b"obj") {
            i = at + 3;
            // Walk back over `n g ` before the keyword.
            let mut j = at;
            while j > 0 && d.get(j - 1).is_some_and(|b| super::lex::is_white(*b)) {
                j -= 1;
            }
            let gen_end = j;
            while j > 0 && d.get(j - 1).is_some_and(u8::is_ascii_digit) {
                j -= 1;
            }
            let gen_start = j;
            while j > 0 && d.get(j - 1).is_some_and(|b| super::lex::is_white(*b)) {
                j -= 1;
            }
            let num_end = j;
            while j > 0 && d.get(j - 1).is_some_and(u8::is_ascii_digit) {
                j -= 1;
            }
            if j == num_end || gen_start == gen_end || (j > 0 && d.get(j - 1).is_some_and(u8::is_ascii_alphanumeric)) {
                continue;
            }
            let num = std::str::from_utf8(d.get(j..num_end).unwrap_or(&[])).ok().and_then(|s| s.parse::<u32>().ok());
            let g = std::str::from_utf8(d.get(gen_start..gen_end).unwrap_or(&[])).ok().and_then(|s| s.parse::<u16>().ok());
            if let (Some(n), Some(g)) = (num, g) {
                found.push((n, j, g));
            }
        }
        for (n, off, g) in found {
            self.xref.insert(n, Entry::At(off, g));
        }
        if let Some(t) = find_last(d, b"trailer") {
            let mut lx = Lexer::new(d, t + 7);
            if let Some(Ok(Obj::Dict(td))) = lx.object() {
                self.trailer = td;
            }
        }
        if self.root().is_none() {
            // Object streams and the catalog from what the scan found.
            let nums: Vec<(u32, usize)> = self.xref.iter().filter_map(|(n, e)| if let Entry::At(o, _) = e { Some((*n, *o)) } else { None }).collect();
            let mut catalog = None;
            for (n, off) in nums {
                let Some((_, obj)) = self.parse_at(off) else { continue };
                if let Some(dd) = obj.as_dict() {
                    if dd.name("Type") == Some("XRef") && dd.get("Root").is_some() {
                        self.trailer = dd.clone();
                    }
                    if dd.name("Type") == Some("Catalog") {
                        catalog = Some(n);
                    }
                    if dd.name("Type") == Some("ObjStm")
                        && let Some((_, list, _)) = self.object_stream(n)
                    {
                        for (idx, (m, _)) in list.iter().enumerate() {
                            self.xref.entry(*m).or_insert(Entry::InStream(n, idx));
                        }
                    }
                }
            }
            if self.root().is_none()
                && let Some(c) = catalog
            {
                self.trailer.set("Root", Obj::Ref(c, 0));
            }
        }
    }

    /// Parse the indirect object at `off`: (number, object), with its stream bytes when it is a
    /// stream (the length from `/Length`, checked against `endstream`).
    fn parse_at(&self, off: usize) -> Option<(u32, Obj)> {
        let (num, _, after) = self.header_at(off)?;
        let mut lx = Lexer::new(self.data, after);
        let obj = match lx.object()? {
            Ok(o) => o,
            Err(_) => Obj::Null,
        };
        let Obj::Dict(dict) = obj else { return Some((num, obj)) };
        match lx.token() {
            Some(super::lex::Tok::Word(w)) if w.as_slice() == b"stream" => {}
            _ => return Some((num, Obj::Dict(dict))),
        }
        let mut start = lx.pos;
        if self.data.get(start) == Some(&b'\r') {
            start += 1;
        }
        if self.data.get(start) == Some(&b'\n') {
            start += 1;
        }
        let declared = match dict.get("Length") {
            Some(Obj::Int(n)) => usize::try_from(*n).ok(),
            Some(r @ Obj::Ref(rn, _)) if *rn != num => self.resolve(r).as_int().and_then(|n| usize::try_from(n).ok()),
            _ => None,
        };
        let ends_well = |end: usize| {
            let mut p = end;
            while self.data.get(p).is_some_and(|b| super::lex::is_white(*b)) {
                p += 1;
            }
            self.data.get(p..p + 9) == Some(b"endstream")
        };
        let end = match declared {
            Some(n) if start + n <= self.data.len() && ends_well(start + n) => start + n,
            _ => {
                let e = find_from(self.data, start, b"endstream").unwrap_or(self.data.len());
                let mut e2 = e;
                if e2 > start && self.data.get(e2 - 1) == Some(&b'\n') {
                    e2 -= 1;
                }
                if e2 > start && self.data.get(e2 - 1) == Some(&b'\r') {
                    e2 -= 1;
                }
                e2
            }
        };
        let data = self.data.get(start..end).unwrap_or(&[]).to_vec();
        Some((num, Obj::Stream(Box::new(Stream { dict, data }))))
    }

    /// The decoded data of object stream `n`: (bytes, member numbers and offsets, first offset).
    fn object_stream(&self, n: u32) -> Option<ObjStream> {
        if let Some(s) = self.streams.borrow().get(&n) {
            return Some(s.clone());
        }
        let obj = self.get(n)?;
        let Obj::Stream(s) = obj else { return None };
        let data = filters::decode(&s.dict, &s.data).ok()?;
        let count = usize::try_from(s.dict.int("N").unwrap_or(0)).unwrap_or(0).min(1 << 20);
        let first = usize::try_from(s.dict.int("First").unwrap_or(0)).unwrap_or(0);
        let mut lx = Lexer::new(&data, 0);
        let mut list = Vec::with_capacity(count);
        for _ in 0..count {
            let (Some(Ok(Obj::Int(m))), Some(Ok(Obj::Int(o)))) = (lx.object(), lx.object()) else { break };
            list.push((u32::try_from(m).unwrap_or(0), usize::try_from(o).unwrap_or(0)));
        }
        let v = (data, list, first);
        self.streams.borrow_mut().insert(n, v.clone());
        Some(v)
    }

    /// Object `n`, loaded (and decrypted) once.
    #[must_use]
    pub fn get(&self, n: u32) -> Option<Obj> {
        if let Some(o) = self.cache.borrow().get(&n) {
            return Some(o.clone());
        }
        let obj = match *self.xref.get(&n)? {
            Entry::At(_, u16::MAX) => return None,
            Entry::At(off, g) => {
                let (num, mut o) = self.parse_at(off)?;
                if num != n {
                    return None;
                }
                if let Some(c) = &self.crypt
                    && self.enc_num != Some(n)
                {
                    c.strings_in(n, g, &mut o);
                    if let Obj::Stream(s) = &mut o
                        && s.dict.name("Type") != Some("XRef")
                    {
                        s.data = c.stream(n, g, &s.data);
                    }
                }
                o
            }
            Entry::InStream(sn, idx) => {
                if sn == n {
                    return None;
                }
                let (data, list, first) = self.object_stream(sn)?;
                let (m, o) = *list.get(idx)?;
                if m != n {
                    return None;
                }
                let mut lx = Lexer::new(&data, first + o);
                lx.object()?.ok()?
            }
        };
        self.cache.borrow_mut().insert(n, obj.clone());
        Some(obj)
    }

    /// A value with references followed (a missing object is null).
    #[must_use]
    pub fn resolve(&self, o: &Obj) -> Obj {
        let mut cur = o.clone();
        for _ in 0..MAX_RESOLVE {
            match cur {
                Obj::Ref(n, _) => cur = self.get(n).unwrap_or(Obj::Null),
                _ => return cur,
            }
        }
        Obj::Null
    }

    /// The value under `key` of `d`, resolved.
    #[must_use]
    pub fn at(&self, d: &Dict, key: &str) -> Obj {
        d.get(key).map_or(Obj::Null, |v| self.resolve(v))
    }

    /// The decoded bytes of a stream object.
    /// # Errors
    /// Why it was not decoded.
    pub fn stream_data(&self, s: &Stream) -> Result<Vec<u8>, filters::Skip> {
        filters::decode(&s.dict, &s.data)
    }

    /// Every page dictionary in order, with inherited `Resources`, `MediaBox`, `CropBox` and
    /// `Rotate` filled in, and its object number.
    #[must_use]
    pub fn pages(&self) -> Vec<(Option<u32>, Dict)> {
        let mut out = Vec::new();
        let Some(root) = self.root() else { return out };
        let mut seen = Vec::new();
        if let Some(p) = root.get("Pages") {
            self.walk_pages(p, &Dict::default(), &mut out, &mut seen, 0);
        }
        out
    }

    fn walk_pages(&self, node: &Obj, inherit: &Dict, out: &mut Vec<(Option<u32>, Dict)>, seen: &mut Vec<u32>, depth: usize) {
        if depth > 64 || out.len() > 100_000 {
            return;
        }
        let num = if let Obj::Ref(n, _) = node {
            if seen.contains(n) {
                return;
            }
            seen.push(*n);
            Some(*n)
        } else {
            None
        };
        let o = self.resolve(node);
        let Some(d) = o.as_dict() else { return };
        let mut inh = inherit.clone();
        for k in ["Resources", "MediaBox", "CropBox", "Rotate"] {
            if let Some(v) = d.get(k) {
                inh.set(k, v.clone());
            }
        }
        let kids = self.at(d, "Kids");
        let is_page = d.name("Type") == Some("Page") || (d.name("Type").is_none() && kids.as_array().is_none());
        if is_page {
            let mut page = d.clone();
            for (k, v) in &inh.0 {
                if page.get(k).is_none() {
                    page.0.push((k.clone(), v.clone()));
                }
            }
            out.push((num, page));
            return;
        }
        if let Some(kids) = kids.as_array() {
            for k in kids {
                self.walk_pages(k, &inh, out, seen, depth + 1);
            }
        }
    }
}

/// A text string (outline titles, document info): UTF-16BE or UTF-8 with their marks, else
/// PDFDocEncoding.
#[must_use]
pub fn text_string(b: &[u8]) -> String {
    if let Some(rest) = b.strip_prefix(&[0xFE, 0xFF]) {
        let units: Vec<u16> = rest.as_chunks::<2>().0.iter().map(|c| u16::from_be_bytes(*c)).collect();
        return String::from_utf16_lossy(&units).replace('\0', "");
    }
    if let Some(rest) = b.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        return String::from_utf8_lossy(rest).into_owned();
    }
    b.iter().map(|&c| super::font::pdfdoc(c)).collect()
}

pub(crate) fn check() -> Result<(), String> {
    // A two-object file with a correct table, then the same file with a wrong offset (repaired).
    let body = b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 100 100] >> endobj\n3 0 obj << /Type /Page /Parent 2 0 R >> endobj\n";
    for wrong in [false, true] {
        let mut f = body.to_vec();
        let x = f.len();
        let off = |pat: &[u8]| find_from(body, 0, pat).unwrap_or(0) + usize::from(wrong) * 3;
        f.extend_from_slice(b"xref\n0 4\n0000000000 65535 f \n");
        for p in [&b"1 0 obj"[..], b"2 0 obj", b"3 0 obj"] {
            f.extend_from_slice(format!("{:010} 00000 n \n", off(p)).as_bytes());
        }
        f.extend_from_slice(format!("trailer << /Size 4 /Root 1 0 R >>\nstartxref\n{x}\n%%EOF\n").as_bytes());
        let file = File::open(&f).map_err(|e| format!("nmdoc.pdf.file: {e}"))?;
        let pages = file.pages();
        if pages.len() != 1 || pages.first().and_then(|p| p.1.get("MediaBox")).is_none() || file.repaired != wrong {
            return Err(format!("nmdoc.pdf.file: pages {} repaired {} (wrong offsets {wrong})", pages.len(), file.repaired));
        }
    }
    if File::open(b"not a pdf").is_ok() || File::open(b"%PDF-1.4\n garbage").is_ok() {
        return Err("nmdoc.pdf.file: opened what is no PDF".into());
    }
    if text_string(&[0xFE, 0xFF, 0, b'A', 0x20, 0x19]) != "A\u{2019}" || text_string(b"Pump") != "Pump" {
        return Err("nmdoc.pdf.file: text strings".into());
    }
    Ok(())
}
