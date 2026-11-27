//! Fonts as text: how a font splits a string into character codes, the character each code
//! stands for, and each code's width. A `/ToUnicode` map is used first when present (`bfchar`,
//! `bfrange` with a start value or an array); else a simple font's encoding (Standard, WinAnsi,
//! MacRoman, plus `/Differences` glyph names through the glyph list and `uniXXXX` names); a
//! composite (Type0) font without a map reads only through a predefined UCS-2 or UTF-16 CMap.
//! Codes with no character are reported as unmapped, never guessed.
use super::file::File;
use super::lex::{Dict, Lexer, Obj, Tok};
use super::tables::{GLYPHS, MAC_ROMAN, PDF_DOC, STANDARD, WIN_ANSI};
use std::collections::HashMap;

/// A character of PDFDocEncoding (text strings).
#[must_use]
pub fn pdfdoc(c: u8) -> char {
    match c {
        0x18 => '\u{2d8}',
        0x19 => '\u{2c7}',
        0x1A => '\u{2c6}',
        0x1B => '\u{2d9}',
        0x1C => '\u{2dd}',
        0x1D => '\u{2db}',
        0x1E => '\u{2da}',
        0x1F => '\u{2dc}',
        0..=0x7F => char::from(c),
        _ => PDF_DOC.get(usize::from(c - 0x80)).copied().unwrap_or('\u{fffd}'),
    }
}

/// The character of a glyph name: the glyph list, `uniXXXX`, `uXXXX` to `uXXXXXX`, a name with a
/// suffix after `.` (`a.sc`), ligatures of names joined by `_`.
#[must_use]
pub fn glyph_char(name: &str) -> Option<String> {
    let base = name.split('.').next().unwrap_or(name);
    if base.is_empty() {
        return None;
    }
    if let Ok(i) = GLYPHS.binary_search_by(|(n, _)| (*n).cmp(base)) {
        return GLYPHS.get(i).map(|(_, c)| c.to_string());
    }
    if base.contains('_') {
        let parts: Option<String> = base.split('_').map(glyph_char).collect();
        return parts;
    }
    if let Some(hex) = base.strip_prefix("uni")
        && hex.len() >= 4
        && hex.len() % 4 == 0
        && hex.bytes().all(|b| b.is_ascii_hexdigit())
    {
        let units: Vec<u16> = hex.as_bytes().chunks(4).filter_map(|c| std::str::from_utf8(c).ok()).filter_map(|c| u16::from_str_radix(c, 16).ok()).collect();
        return Some(String::from_utf16_lossy(&units));
    }
    if let Some(hex) = base.strip_prefix('u')
        && (4..=6).contains(&hex.len())
        && hex.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return u32::from_str_radix(hex, 16).ok().and_then(char::from_u32).map(String::from);
    }
    None
}

/// Codes 128 to 255 of a named base encoding, and whether codes 39 and 96 are curly quotes.
fn base_encoding(name: &str) -> Option<(&'static [char; 128], bool)> {
    match name {
        "WinAnsiEncoding" => Some((&WIN_ANSI, false)),
        "MacRomanEncoding" => Some((&MAC_ROMAN, false)),
        "StandardEncoding" => Some((&STANDARD, true)),
        "PDFDocEncoding" => Some((&PDF_DOC, false)),
        _ => None,
    }
}

fn utf16_be(b: &[u8]) -> String {
    let units: Vec<u16> = b.chunks(2).map(|c| u16::from_be_bytes([c.first().copied().unwrap_or(0), c.get(1).copied().unwrap_or(0)])).collect();
    String::from_utf16_lossy(&units)
}

fn code_of(b: &[u8]) -> u32 {
    b.iter().take(4).fold(0u32, |acc, x| acc << 8 | u32::from(*x))
}

/// What a CMap stream says: code space ranges (length, low, high), code to text (`bfchar`,
/// `bfrange`) and code to CID (`cidchar`, `cidrange`).
#[derive(Clone, Debug, Default)]
pub struct CMap {
    pub space: Vec<(usize, u32, u32)>,
    pub text: HashMap<u32, String>,
    pub cid: HashMap<u32, u32>,
    pub cid_ranges: Vec<(u32, u32, u32)>,
}

/// Most entries one range may add.
const MAX_RANGE: u32 = 1 << 16;

impl CMap {
    /// Parse a CMap stream's decoded bytes.
    #[must_use]
    pub fn parse(data: &[u8]) -> Self {
        let mut m = Self::default();
        let mut lx = Lexer::new(data, 0);
        let mut ops: Vec<Obj> = Vec::new();
        let mut array: Option<Vec<Obj>> = None;
        while let Some(t) = lx.token() {
            match t {
                Tok::ArrayOpen => array = Some(Vec::new()),
                Tok::ArrayClose => {
                    if let Some(a) = array.take() {
                        ops.push(Obj::Array(a));
                    }
                }
                Tok::Obj(o) => match array.as_mut() {
                    Some(a) => a.push(o),
                    None => ops.push(o),
                },
                Tok::Word(w) => {
                    match w.as_slice() {
                        b"endcodespacerange" => {
                            for p in ops.chunks(2) {
                                if let [Obj::Str(lo), Obj::Str(hi)] = p {
                                    m.space.push((lo.len().clamp(1, 4), code_of(lo), code_of(hi)));
                                }
                            }
                        }
                        b"endbfchar" => {
                            for p in ops.chunks(2) {
                                match p {
                                    [Obj::Str(src), Obj::Str(dst)] => {
                                        m.text.insert(code_of(src), utf16_be(dst));
                                    }
                                    [Obj::Str(src), Obj::Name(n)] => {
                                        if let Some(s) = glyph_char(n) {
                                            m.text.insert(code_of(src), s);
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        b"endbfrange" => {
                            for p in ops.chunks(3) {
                                let [Obj::Str(lo), Obj::Str(hi), dst] = p else { continue };
                                let (lo, hi) = (code_of(lo), code_of(hi));
                                if hi < lo || hi - lo > MAX_RANGE {
                                    continue;
                                }
                                match dst {
                                    Obj::Str(d) => {
                                        let mut units: Vec<u16> =
                                            d.chunks(2).map(|c| u16::from_be_bytes([c.first().copied().unwrap_or(0), c.get(1).copied().unwrap_or(0)])).collect();
                                        for c in lo..=hi {
                                            m.text.insert(c, String::from_utf16_lossy(&units));
                                            if let Some(last) = units.last_mut() {
                                                *last = last.wrapping_add(1);
                                            }
                                        }
                                    }
                                    Obj::Array(a) => {
                                        for (c, d) in (lo..=hi).zip(a.iter()) {
                                            if let Obj::Str(d) = d {
                                                m.text.insert(c, utf16_be(d));
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        b"endcidchar" => {
                            for p in ops.chunks(2) {
                                if let [Obj::Str(src), Obj::Int(cid)] = p {
                                    m.cid.insert(code_of(src), u32::try_from(*cid).unwrap_or(0));
                                }
                            }
                        }
                        b"endcidrange" => {
                            for p in ops.chunks(3) {
                                if let [Obj::Str(lo), Obj::Str(hi), Obj::Int(cid)] = p {
                                    m.cid_ranges.push((code_of(lo), code_of(hi), u32::try_from(*cid).unwrap_or(0)));
                                }
                            }
                        }
                        _ => {}
                    }
                    ops.clear();
                }
                Tok::DictOpen | Tok::DictClose => {}
            }
        }
        m
    }
}

/// One font, ready to turn strings into characters and widths.
#[derive(Clone, Debug)]
pub struct Font {
    /// Code space ranges of a composite font; empty for a simple font (one byte per code).
    space: Vec<(usize, u32, u32)>,
    /// Bytes per code when no range fits (2 for composite fonts).
    default_len: usize,
    to_unicode: HashMap<u32, String>,
    /// Simple font: the character of each code by its encoding.
    simple: Option<Vec<Option<String>>>,
    /// Composite font: codes are UCS-2 or UTF-16 values (a predefined Uni CMap).
    codes_are_unicode: bool,
    /// Composite font: code to CID.
    cmap_cid: Option<CMap>,
    widths: HashMap<u32, f64>,
    default_width: f64,
    /// Scale from glyph space to text space (1/1000, or a Type 3 font's matrix).
    scale: f64,
    pub bold: bool,
    pub mono: bool,
    pub name: String,
}

impl Font {
    /// A font with no information: every code is one byte, unmapped, 500 units wide.
    #[must_use]
    pub fn unknown() -> Self {
        Self {
            space: Vec::new(),
            default_len: 1,
            to_unicode: HashMap::new(),
            simple: None,
            codes_are_unicode: false,
            cmap_cid: None,
            widths: HashMap::new(),
            default_width: 500.0,
            scale: 0.001,
            bold: false,
            mono: false,
            name: String::new(),
        }
    }

    /// Read a font dictionary.
    #[must_use]
    pub fn load(file: &File<'_>, d: &Dict) -> Self {
        let mut f = Self::unknown();
        let subtype = d.name("Subtype").unwrap_or("Type1");
        f.name = d.name("BaseFont").unwrap_or("").to_owned();
        let tu = file.resolve(d.get("ToUnicode").unwrap_or(&Obj::Null));
        if let Obj::Stream(s) = &tu
            && let Ok(data) = file.stream_data(s)
        {
            f.to_unicode = CMap::parse(&data).text;
        }
        let desc_font = if subtype == "Type0" {
            file.at(d, "DescendantFonts").as_array().and_then(|a| a.first()).map(|x| file.resolve(x)).and_then(|x| x.as_dict().cloned())
        } else {
            None
        };
        let fd_owner = desc_font.as_ref().unwrap_or(d);
        let fd = file.at(fd_owner, "FontDescriptor");
        let fd = fd.as_dict();
        let flags = fd.and_then(|x| x.int("Flags")).unwrap_or(0);
        let weight = fd.and_then(|x| x.num("FontWeight")).unwrap_or(400.0);
        let low = f.name.to_lowercase();
        f.bold = low.contains("bold") || low.contains("black") || low.contains("heavy") || flags & (1 << 18) != 0 || weight >= 600.0;
        f.mono = low.contains("mono") || low.contains("courier") || low.contains("consola") || low.contains("menlo") || flags & 1 != 0;
        if subtype == "Type0" {
            f.default_len = 2;
            match file.at(d, "Encoding") {
                Obj::Name(n) => {
                    f.codes_are_unicode = n.contains("UCS2") || n.contains("UTF16");
                    if n.starts_with("Identity") {
                        f.space = vec![(2, 0, 0xFFFF)];
                    }
                }
                Obj::Stream(s) => {
                    if let Ok(data) = file.stream_data(&s) {
                        let cm = CMap::parse(&data);
                        f.space.clone_from(&cm.space);
                        f.cmap_cid = Some(cm);
                    }
                }
                _ => {}
            }
            if let Some(df) = &desc_font {
                f.default_width = df.num("DW").unwrap_or(1000.0);
                if let Obj::Array(w) = file.at(df, "W") {
                    f.cid_widths(file, &w);
                }
            }
        } else {
            f.simple_encoding(file, d, subtype);
            let first = d.int("FirstChar").unwrap_or(0);
            if let Obj::Array(w) = file.at(d, "Widths") {
                for (i, x) in w.iter().enumerate() {
                    if let (Some(v), Ok(c)) = (file.resolve(x).as_num(), u32::try_from(first + i64::try_from(i).unwrap_or(0))) {
                        f.widths.insert(c, v);
                    }
                }
            }
            f.default_width = fd.and_then(|x| x.num("MissingWidth")).filter(|w| *w > 0.0).unwrap_or(500.0);
            if subtype == "Type3"
                && let Some(m) = file.at(d, "FontMatrix").as_array().and_then(|a| a.first()).and_then(Obj::as_num)
            {
                f.scale = m.abs();
            }
        }
        f
    }

    fn cid_widths(&mut self, file: &File<'_>, w: &[Obj]) {
        let mut i = 0;
        while i < w.len() {
            let Some(start) = w.get(i).and_then(|x| file.resolve(x).as_int()) else { break };
            match w.get(i + 1).map(|x| file.resolve(x)) {
                Some(Obj::Array(list)) => {
                    for (k, x) in list.iter().enumerate() {
                        if let (Some(v), Ok(c)) = (x.as_num(), u32::try_from(start + i64::try_from(k).unwrap_or(0))) {
                            self.widths.insert(c, v);
                        }
                    }
                    i += 2;
                }
                Some(o) => {
                    let end = o.as_int().unwrap_or(start);
                    let v = w.get(i + 2).and_then(|x| file.resolve(x).as_num()).unwrap_or(self.default_width);
                    for c in start..=end.min(start + i64::from(MAX_RANGE)) {
                        if let Ok(c) = u32::try_from(c) {
                            self.widths.insert(c, v);
                        }
                    }
                    i += 3;
                }
                None => break,
            }
        }
    }

    fn simple_encoding(&mut self, file: &File<'_>, d: &Dict, subtype: &str) {
        let symbolic = matches!(self.name.trim_start_matches(|c: char| c.is_ascii_uppercase() && c != '+').trim_start_matches('+'), "Symbol" | "ZapfDingbats");
        let (base_name, diffs) = match file.at(d, "Encoding") {
            Obj::Name(n) => (Some(n), None),
            Obj::Dict(e) => (e.name("BaseEncoding").map(str::to_owned), file.at(&e, "Differences").as_array().map(<[Obj]>::to_vec)),
            _ => (None, None),
        };
        let default = if subtype == "TrueType" { "WinAnsiEncoding" } else { "StandardEncoding" };
        let (high, curly) = base_encoding(base_name.as_deref().unwrap_or(default)).unwrap_or((&STANDARD, true));
        let mut table: Vec<Option<String>> = (0..256u32)
            .map(|c| {
                if symbolic && diffs.is_none() {
                    return None;
                }
                match c {
                    39 if curly => Some("\u{2019}".to_owned()),
                    96 if curly => Some("\u{2018}".to_owned()),
                    32..=126 => char::from_u32(c).map(String::from),
                    128..=255 => high.get(usize::try_from(c - 128).unwrap_or(0)).filter(|x| **x != '\u{fffd}').map(char::to_string),
                    _ => None,
                }
            })
            .collect();
        if let Some(diffs) = diffs {
            let mut code: usize = 0;
            for x in diffs {
                match x {
                    Obj::Int(n) => code = usize::try_from(n).unwrap_or(0),
                    Obj::Name(n) => {
                        if let Some(slot) = table.get_mut(code) {
                            *slot = glyph_char(&n);
                        }
                        code += 1;
                    }
                    _ => {}
                }
            }
        }
        self.simple = Some(table);
    }

    /// The codes of a string: (code, byte length).
    #[must_use]
    pub fn codes(&self, s: &[u8]) -> Vec<(u32, usize)> {
        if self.simple.is_some() || (self.space.is_empty() && self.default_len == 1) {
            return s.iter().map(|b| (u32::from(*b), 1)).collect();
        }
        let mut out = Vec::new();
        let mut i = 0;
        while i < s.len() {
            let mut len = 0;
            for n in 1..=4 {
                let Some(b) = s.get(i..i + n) else { break };
                let c = code_of(b);
                if self.space.iter().any(|(l, lo, hi)| *l == n && c >= *lo && c <= *hi) {
                    len = n;
                    break;
                }
            }
            if len == 0 {
                len = self.default_len.min(s.len() - i).max(1);
            }
            out.push((code_of(s.get(i..i + len).unwrap_or(&[])), len));
            i += len;
        }
        out
    }

    /// The text of a code, or `None` when nothing maps it to a character.
    #[must_use]
    pub fn text(&self, code: u32) -> Option<String> {
        if let Some(t) = self.to_unicode.get(&code) {
            return Some(t.clone());
        }
        if let Some(table) = &self.simple {
            return table.get(usize::try_from(code).ok()?).cloned().flatten();
        }
        if self.codes_are_unicode {
            return char::from_u32(code).map(String::from);
        }
        None
    }

    /// True when a font has some way to map codes to characters.
    #[must_use]
    pub fn has_map(&self) -> bool {
        !self.to_unicode.is_empty() || self.simple.is_some() || self.codes_are_unicode
    }

    fn cid(&self, code: u32) -> u32 {
        let Some(cm) = &self.cmap_cid else { return code };
        if let Some(c) = cm.cid.get(&code) {
            return *c;
        }
        cm.cid_ranges.iter().find(|(lo, hi, _)| code >= *lo && code <= *hi).map_or(code, |(lo, _, base)| base + (code - lo))
    }

    /// The width of a code in text space units (glyph space times the font's scale).
    #[must_use]
    pub fn width(&self, code: u32) -> f64 {
        let key = if self.simple.is_some() { code } else { self.cid(code) };
        self.widths.get(&key).copied().unwrap_or(self.default_width) * self.scale
    }
}

pub(crate) fn check() -> Result<(), String> {
    for (name, want) in [("a", "a"), ("Aacute", "\u{c1}"), ("quoteright", "\u{2019}"), ("uni2022", "\u{2022}"), ("u1F600", "\u{1f600}"), ("f_i", "fi"), ("a.sc", "a")]
    {
        if glyph_char(name).as_deref() != Some(want) {
            return Err(format!("nmdoc.pdf.font: glyph {name} is {:?}", glyph_char(name)));
        }
    }
    if glyph_char("g123").is_some() || glyph_char("cid45").is_some() {
        return Err("nmdoc.pdf.font: a glyph index name was given a character".into());
    }
    let cm = CMap::parse(
        b"1 begincodespacerange <0000> <FFFF> endcodespacerange 2 beginbfchar <0003> <0020> <0010> <00660069> endbfchar \
          2 beginbfrange <0024> <0026> <0041> <0030> <0031> [<0078> <0079>] endbfrange",
    );
    let want: [(u32, &str); 6] = [(3, " "), (0x10, "fi"), (0x24, "A"), (0x26, "C"), (0x30, "x"), (0x31, "y")];
    for (c, w) in want {
        if cm.text.get(&c).map(String::as_str) != Some(w) {
            return Err(format!("nmdoc.pdf.font: cmap code {c:#x} is {:?}", cm.text.get(&c)));
        }
    }
    if cm.space != [(2, 0, 0xFFFF)] {
        return Err("nmdoc.pdf.font: code space".into());
    }
    if pdfdoc(0x90) != '\u{2019}' || pdfdoc(b'A') != 'A' {
        return Err("nmdoc.pdf.font: PDFDocEncoding".into());
    }
    Ok(())
}
