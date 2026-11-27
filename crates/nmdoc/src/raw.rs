//! Raw reading: for a file no reader fits, the runs of readable text recovered from its bytes
//! (UTF-8 and Latin-1 in one pass, UTF-16 little and big endian at both byte alignments), each
//! located by its byte offset, under a short header saying what the file is by its first bytes,
//! its size and its name. Text found this way has no headings and no pages; compressed or
//! encrypted content yields little or nothing.

/// Shortest run kept, in characters. Chosen by measurement: the shortest length at which a
/// megabyte of random bytes yields almost no runs (16 at 8 characters, 59 at 6) while the test
/// binaries keep every sentence and give no stray runs; words shorter than this alone are lost.
pub const MIN_RUN: usize = 8;
/// Longest line written for one run; a longer run is split at a space.
const LINE: usize = 240;
/// Derived text stops after this many bytes (a note says so).
pub const MAX_DERIVED: usize = 2 << 20;

/// One run of text: its byte offset, its text and the encoding it was read in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Run {
    pub at: usize,
    pub end: usize,
    pub text: String,
    pub enc: &'static str,
}

/// True when a recovered span reads as text: enough characters, at least three letters, mostly
/// letters, digits and spaces, and either mostly ASCII with a word of three ASCII letters, or
/// mostly letters of one script beyond Latin-1 (one block of 128 code points), which random bytes
/// rarely give.
fn readable(chars: &[char], min: usize) -> bool {
    let n = chars.len();
    if n < min {
        return false;
    }
    let letters = chars.iter().filter(|c| c.is_alphabetic()).count();
    let words = chars.iter().filter(|c| c.is_alphanumeric() || **c == ' ').count();
    if letters < 3 || words * 20 < n * 17 {
        return false;
    }
    let ascii = chars.iter().filter(|c| c.is_ascii()).count();
    if ascii * 5 >= n * 4 {
        // Words have vowels (about two letters in five; random letters one in five) and keep one
        // case inside a word (random letters change case about every second letter).
        let latin: Vec<char> = chars.iter().copied().filter(char::is_ascii_alphabetic).collect();
        let vowels = latin.iter().filter(|c| "aeiouyAEIOUY".contains(**c)).count();
        let pairs = chars.windows(2).filter(|w| w.iter().all(char::is_ascii_alphabetic)).count();
        let humps = chars.windows(2).filter(|w| w.first().is_some_and(char::is_ascii_lowercase) && w.get(1).is_some_and(char::is_ascii_uppercase)).count();
        let flips = chars.windows(3).filter(|w| w.iter().all(char::is_ascii_alphabetic) && w.first().map(char::is_ascii_uppercase) == w.get(2).map(char::is_ascii_uppercase) && w.first().map(char::is_ascii_uppercase) != w.get(1).map(char::is_ascii_uppercase)).count();
        return chars.windows(3).any(|w| w.iter().all(char::is_ascii_alphabetic))
            && vowels * 4 >= latin.len()
            && humps * 5 <= pairs
            && flips * 6 <= pairs;
    }
    // Latin-1 letters count only beside ASCII (single high bytes are common in binary data).
    let blocks: Vec<u32> = chars.iter().filter(|c| !c.is_ascii() && c.is_alphabetic()).map(|c| u32::from(*c) >> 7).collect();
    blocks.len() * 5 >= n * 3 && blocks.first().is_some_and(|b| *b >= 2) && blocks.windows(2).all(|w| w.first() == w.get(1))
}

/// Keep the readable spans of one run of printable characters (each with its byte offset): the
/// run is split where a character is neither ASCII nor a letter, and each span is trimmed.
fn push(out: &mut Vec<Run>, chars: &[(usize, char)], end: usize, enc: &'static str, min: usize) {
    let mut i = 0;
    while i < chars.len() {
        let mut j = i;
        while chars.get(j).is_some_and(|(_, c)| c.is_ascii() || c.is_alphabetic()) {
            j += 1;
        }
        let mut span = chars.get(i..j).unwrap_or(&[]);
        while span.first().is_some_and(|(_, c)| c.is_whitespace()) {
            span = span.get(1..).unwrap_or(&[]);
        }
        while span.last().is_some_and(|(_, c)| c.is_whitespace()) {
            span = span.get(..span.len() - 1).unwrap_or(&[]);
        }
        let text: Vec<char> = span.iter().map(|(_, c)| *c).collect();
        if readable(&text, min)
            && let Some((at, _)) = span.first()
        {
            let stop = chars.get(j).map_or(end, |(o, _)| *o);
            out.push(Run { at: *at, end: stop, text: text.into_iter().collect(), enc });
        }
        i = j + 1;
    }
}

/// Runs in UTF-8, with bytes that are not valid UTF-8 read as Windows-1252 (Latin-1).
fn single(b: &[u8], min: usize, out: &mut Vec<Run>) {
    let mut chars: Vec<(usize, char)> = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let x = b.get(i).copied().unwrap_or(0);
        let mut step = 1;
        let ch: Option<char> = match x {
            0x20..=0x7E | 0x09 => Some(char::from(x)),
            0xC2..=0xF4 => {
                let len = if x < 0xE0 { 2 } else if x < 0xF0 { 3 } else { 4 };
                match b.get(i..i + len).and_then(|s| std::str::from_utf8(s).ok()).and_then(|s| s.chars().next()) {
                    Some(c) if !c.is_control() => {
                        step = len;
                        Some(c)
                    }
                    _ => crate::text::windows_1252(&[x]).chars().next(),
                }
            }
            0x80..=0xFF => crate::text::windows_1252(&[x]).chars().next().filter(|c| !c.is_control()),
            _ => None,
        };
        match ch {
            Some(c) => chars.push((i, c)),
            None => {
                push(out, &chars, i, "utf-8", min);
                chars.clear();
            }
        }
        i += step;
    }
    push(out, &chars, b.len(), "utf-8", min);
}

/// A UTF-16 code unit that is part of a run: printable Latin, Greek, Cyrillic or common
/// punctuation. Other scripts are not taken, since random bytes look like them.
fn unit_ok(u: u16) -> Option<char> {
    let c = char::from_u32(u32::from(u))?;
    let ok = matches!(u, 0x20..=0x7E | 0x09 | 0xA0..=0x04FF | 0x2010..=0x2027) && !c.is_control();
    ok.then_some(c)
}

fn wide(b: &[u8], min: usize, little: bool, out: &mut Vec<Run>) {
    let enc = if little { "utf-16le" } else { "utf-16be" };
    for align in 0..2 {
        let mut chars: Vec<(usize, char)> = Vec::new();
        let mut i = align;
        while i + 1 < b.len() {
            let pair = [b.get(i).copied().unwrap_or(0), b.get(i + 1).copied().unwrap_or(0)];
            let u = if little { u16::from_le_bytes(pair) } else { u16::from_be_bytes(pair) };
            match unit_ok(u) {
                Some(c) => chars.push((i, c)),
                None => {
                    push(out, &chars, i, enc, min);
                    chars.clear();
                }
            }
            i += 2;
        }
        push(out, &chars, i, enc, min);
    }
}

/// Every run of readable text of at least `min` characters, in byte order.
#[must_use]
pub fn runs(b: &[u8], min: usize) -> Vec<Run> {
    let mut out = Vec::new();
    single(b, min, &mut out);
    wide(b, min, true, &mut out);
    wide(b, min, false, &mut out);
    out.sort_by_key(|r| (r.at, r.end));
    // Runs that overlap are the same bytes read two ways (UTF-16 little endian text is also big
    // endian text one byte later): the longer reading is kept; on a tie UTF-8, then UTF-16 little
    // endian (the common order in files).
    let rank = |e: &str| match e {
        "utf-8" => 0,
        "utf-16le" => 1,
        _ => 2,
    };
    let mut kept: Vec<Run> = Vec::new();
    for r in out {
        if let Some(k) = kept.last_mut()
            && r.at < k.end
        {
            let (rn, kn) = (r.text.chars().count(), k.text.chars().count());
            if rn > kn || (rn == kn && rank(r.enc) < rank(k.enc)) {
                *k = r;
            }
            continue;
        }
        kept.push(r);
    }
    kept
}

/// What a file is by its first bytes, and a limit worth stating for that kind.
#[must_use]
pub fn kind(b: &[u8]) -> (&'static str, Option<&'static str>) {
    const COMPRESSED: Option<&str> = Some("compressed content: raw reading recovers little or nothing of it");
    const IMAGE: Option<&str> = Some("image: holds no text to recover (a scan needs OCR); only text stored beside the picture is found");
    const MEDIA: Option<&str> = Some("audio or video: holds no text; only tags stored beside the media are found");
    let at = |o: usize, m: &[u8]| b.get(o..o + m.len()) == Some(m);
    if at(0, b"%PDF-") {
        return ("PDF document", Some("PDF: most of its text sits in compressed streams that raw reading cannot reach; words may be out of order"));
    }
    if at(0, b"PK\x03\x04") || at(0, b"PK\x05\x06") {
        return ("zip container that could not be opened", COMPRESSED);
    }
    if at(0, &[0x1F, 0x8B]) {
        return ("gzip data that could not be opened", COMPRESSED);
    }
    if at(0, &[0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1]) {
        return ("Compound File Binary (old binary Office file or similar)", Some("old binary Office files: text is recovered without layout; saving as .docx, .xlsx or .pptx reads better"));
    }
    if at(0, b"{\\rtf") {
        return ("RTF document", Some("RTF: the text is read with its control words in place"));
    }
    let table: &[(usize, &[u8], &str, Option<&str>)] = &[
        (0, b"\x89PNG\r\n\x1a\n", "PNG image", IMAGE),
        (0, &[0xFF, 0xD8, 0xFF], "JPEG image", IMAGE),
        (0, b"GIF8", "GIF image", IMAGE),
        (0, b"BM", "BMP image", IMAGE),
        (0, b"II*\0", "TIFF image", IMAGE),
        (0, b"MM\0*", "TIFF image", IMAGE),
        (0, b"8BPS", "Photoshop image", IMAGE),
        (0, &[0, 0, 1, 0], "icon", IMAGE),
        (8, b"WEBP", "WebP image", IMAGE),
        (8, b"WAVE", "WAV audio", MEDIA),
        (8, b"AVI ", "AVI video", MEDIA),
        (4, b"ftyp", "MP4 or QuickTime media", MEDIA),
        (0, b"ID3", "MP3 audio", MEDIA),
        (0, b"OggS", "Ogg media", MEDIA),
        (0, b"fLaC", "FLAC audio", MEDIA),
        (0, &[0x1A, 0x45, 0xDF, 0xA3], "Matroska or WebM media", MEDIA),
        (0, b"MZ", "Windows program or library", None),
        (0, b"\x7fELF", "ELF program or library", None),
        (0, &[0xCF, 0xFA, 0xED, 0xFE], "Mach-O program", None),
        (0, &[0xCA, 0xFE, 0xBA, 0xBE], "Java class or Mach-O program", None),
        (0, b"\0asm", "WebAssembly module", None),
        (0, b"SQLite format 3\0", "SQLite database", None),
        (0, b"7z\xBC\xAF\x27\x1C", "7-Zip archive", COMPRESSED),
        (0, b"Rar!", "RAR archive", COMPRESSED),
        (0, b"BZh", "bzip2 data", COMPRESSED),
        (0, &[0xFD, b'7', b'z', b'X', b'Z', 0], "xz data", COMPRESSED),
        (0, &[0x28, 0xB5, 0x2F, 0xFD], "Zstandard data", COMPRESSED),
        (0, b"MSCF", "cabinet archive", COMPRESSED),
        (0, b"wOFF", "web font", None),
        (0, b"wOF2", "web font", COMPRESSED),
        (0, b"OTTO", "OpenType font", None),
        (0, &[0, 1, 0, 0], "TrueType font", None),
    ];
    for (o, m, name, note) in table {
        if at(*o, m) {
            return (name, *note);
        }
    }
    ("binary data of no known kind", None)
}

/// True when a file read generally carries text worth taking from its bytes: its head is text
/// (no NUL bytes, few control bytes; `text::is_binary` over the first 8 KB), or it is a document
/// whose prose sits between binary structures (a PDF the PDF reader could not open, an old binary
/// Office file). A program, library, image, media, font, database, archive, cartridge or weights
/// file, or any other binary data, is not: runs recovered from it are fragments of its bytes, not
/// material, so it is kept by its metadata only (`describe`).
#[must_use]
pub fn holds_text(b: &[u8]) -> bool {
    !crate::text::is_binary(b) || b.starts_with(b"%PDF-") || b.starts_with(&[0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1])
}

/// The metadata of a binary file (`holds_text` false): its name, its kind by its first bytes and
/// its size, with any note, and no text from its bytes. A question about the file itself still
/// finds it by these lines; its bytes never become text to cite.
#[must_use]
pub fn describe(name: &str, b: &[u8], extra: Option<&str>) -> String {
    let (what, _) = kind(b);
    let mut out = format!("file: {name}\nkind: {what} (by its first bytes)\nsize: {} bytes\nbinary content: kept by its metadata only\n", b.len());
    if let Some(note) = extra {
        out.push_str(&format!("note: {note}\n"));
    }
    out
}

/// Group runs into paragraphs: a new paragraph after a gap of more than this many bytes, or after
/// this many lines.
const GAP: usize = 256;
const PARA: usize = 16;

/// Derived text of a file read raw: the header, then one line per run (`byte <offset>: <text>`),
/// paragraphs separated by blank lines.
#[must_use]
pub fn derive(name: &str, b: &[u8], extra: Option<&str>) -> String {
    let (what, limit) = kind(b);
    let mut out = format!(
        "file: {name}\nkind: {what} (by its first bytes)\nsize: {} bytes\nread generally: runs of readable text recovered from the bytes, each with its byte offset; no headings or pages\n",
        b.len()
    );
    for note in [extra, limit].into_iter().flatten() {
        out.push_str(&format!("note: {note}\n"));
    }
    let list = runs(b, MIN_RUN);
    if list.is_empty() {
        out.push_str("note: no readable text was found in the bytes\n");
        return out;
    }
    let (mut last_end, mut in_para) = (usize::MAX, PARA);
    for r in &list {
        if out.len() > MAX_DERIVED {
            out.push_str(&format!("\nnote: text stops here, at the {} MB limit of a file read generally\n", MAX_DERIVED >> 20));
            break;
        }
        if in_para >= PARA || last_end == usize::MAX || r.at > last_end + GAP {
            out.push('\n');
            in_para = 0;
        }
        // A long run is written as several lines, each with its own offset.
        let mut rest: &str = &r.text;
        let mut at = r.at;
        let width = if r.enc == "utf-8" { 1 } else { 2 };
        while !rest.is_empty() {
            let cut = if rest.chars().count() <= LINE {
                rest.len()
            } else {
                let limit = rest.char_indices().nth(LINE).map_or(rest.len(), |(i, _)| i);
                rest.get(..limit).and_then(|s| s.rfind(' ')).filter(|i| *i > 0).unwrap_or(limit)
            };
            let (line, tail) = rest.split_at(cut);
            out.push_str(&format!("byte {at}: {}\n", line.trim()));
            in_para += 1;
            at += if width == 1 { line.len() } else { line.chars().count() * 2 };
            rest = tail.trim_start();
        }
        last_end = r.end;
    }
    out
}

pub(crate) fn check() -> Result<(), String> {
    let mut b: Vec<u8> = vec![0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1, 0, 1, 2, 3, 0, 0];
    let a = b.len();
    b.extend(b"The pump seal is rated for 40 bar.");
    b.extend([0, 7, 0x90, 0]);
    let w = b.len();
    for u in "Caf\u{e9} order 7 shipped".encode_utf16() {
        b.extend(u.to_le_bytes());
    }
    b.extend([0, 0, 3, 0xFF, 0xFE]);
    let got: Vec<(usize, String)> = runs(&b, MIN_RUN).into_iter().map(|r| (r.at, r.text)).collect();
    let want = vec![(a, "The pump seal is rated for 40 bar.".to_owned()), (w, "Caf\u{e9} order 7 shipped".to_owned())];
    if got != want {
        return Err(format!("nmdoc.raw: runs {got:?}"));
    }
    let exe = [b"MZ\x90\0\x03\0\0\0".as_slice(), b"This program cannot be run in DOS mode.".as_slice()].concat();
    let meta = describe("tool.exe", &exe, None);
    if holds_text(&exe) || !holds_text(&b) || !meta.contains("kind: Windows program") || meta.contains("DOS mode") {
        return Err(format!("nmdoc.raw: binary kept as text {meta:?}"));
    }
    let text = derive("old.doc", &b, None);
    if !text.contains("kind: Compound File Binary") || !text.contains(&format!("byte {a}: The pump seal")) {
        return Err(format!("nmdoc.raw: derived {text:?}"));
    }
    Ok(())
}
