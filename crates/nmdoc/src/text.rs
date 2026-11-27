//! Text decoder: bytes to text. UTF-8 with or without a byte order mark; UTF-16 little and big
//! endian by their byte order mark, and without one by the pattern of zero bytes; Windows-1252 for
//! bytes that are not valid UTF-8 (the WHATWG mapping, so the label `latin1` reads the same way).
//! Binary content is refused. Line ends are kept as they are.

/// Bytes looked at for the binary and UTF-16 tests.
const SAMPLE: usize = 8192;

/// Windows-1252 code points for bytes 0x80 to 0x9F (WHATWG; the five unassigned bytes map to the
/// C1 control of the same value).
const CP1252: [u16; 32] = [
    0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0x008D, 0x017D, 0x008F, 0x0090,
    0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014, 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x009D, 0x017E, 0x0178,
];

/// UTF-16 without a byte order mark: `Some(true)` little endian, `Some(false)` big endian, when
/// zero bytes fall on one side of almost every pair (text in a Latin script).
fn utf16_guess(sample: &[u8]) -> Option<bool> {
    let pairs = sample.len() / 2;
    if pairs < 4 {
        return None;
    }
    let (mut even0, mut odd0) = (0, 0);
    for [a, b] in sample.as_chunks::<2>().0 {
        if *a == 0 {
            even0 += 1;
        }
        if *b == 0 {
            odd0 += 1;
        }
    }
    let share = |n: usize| n as f64 / pairs as f64;
    if share(odd0) > 0.6 && share(even0) < 0.05 {
        Some(true)
    } else if share(even0) > 0.6 && share(odd0) < 0.05 {
        Some(false)
    } else {
        None
    }
}

/// True when the bytes are not text: a zero byte in the first 8 KB that UTF-16 does not explain,
/// or many control characters.
#[must_use]
pub fn is_binary(bytes: &[u8]) -> bool {
    let sample = bytes.get(..bytes.len().min(SAMPLE)).unwrap_or(&[]);
    if sample.starts_with(&[0xFF, 0xFE]) || sample.starts_with(&[0xFE, 0xFF]) {
        return false;
    }
    if sample.contains(&0) {
        return utf16_guess(sample).is_none();
    }
    let control = sample.iter().filter(|b| **b < 0x20 && !matches!(**b, b'\t' | b'\n' | b'\r' | 0x0C | 0x1B)).count();
    control * 200 > sample.len().max(1)
}

fn utf16(bytes: &[u8], little: bool) -> String {
    let units = bytes.as_chunks::<2>().0.iter().map(|p| if little { u16::from_le_bytes(*p) } else { u16::from_be_bytes(*p) });
    char::decode_utf16(units).map(|c| c.unwrap_or('\u{FFFD}')).collect()
}

/// Windows-1252 bytes to text.
#[must_use]
pub fn windows_1252(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|&b| match b {
            0x80..=0x9F => char::from_u32(u32::from(CP1252.get(usize::from(b - 0x80)).copied().unwrap_or(0xFFFD))).unwrap_or('\u{FFFD}'),
            _ => char::from(b),
        })
        .collect()
}

/// Decode bytes to text and name the encoding.
/// # Errors
/// `binary content` when the bytes are not text.
pub fn decode(bytes: &[u8]) -> Result<(String, &'static str), &'static str> {
    if let Some(rest) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        return Ok(match std::str::from_utf8(rest) {
            Ok(s) => (s.to_owned(), "utf-8-bom"),
            Err(_) => (windows_1252(rest), "windows-1252"),
        });
    }
    if let Some(rest) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        return Ok((utf16(rest, true), "utf-16le"));
    }
    if let Some(rest) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        return Ok((utf16(rest, false), "utf-16be"));
    }
    let sample = bytes.get(..bytes.len().min(SAMPLE)).unwrap_or(&[]);
    if sample.contains(&0) {
        return match utf16_guess(sample) {
            Some(little) => Ok((utf16(bytes, little), if little { "utf-16le" } else { "utf-16be" })),
            None => Err("binary content"),
        };
    }
    if is_binary(bytes) {
        return Err("binary content");
    }
    Ok(match std::str::from_utf8(bytes) {
        Ok(s) => (s.to_owned(), "utf-8"),
        Err(_) => (windows_1252(bytes), "windows-1252"),
    })
}

pub(crate) fn check() -> Result<(), String> {
    let word = "Caf\u{e9} 25\u{b0}C \u{a3}4 \u{fc}ber\r\nline two\n";
    let mut le = vec![0xFF, 0xFE];
    let mut be = vec![0xFE, 0xFF];
    let mut le_bare = Vec::new();
    for u in word.encode_utf16() {
        le.extend(u.to_le_bytes());
        be.extend(u.to_be_bytes());
        le_bare.extend(u.to_le_bytes());
    }
    let mut bom = vec![0xEF, 0xBB, 0xBF];
    bom.extend(word.as_bytes());
    let cp: Vec<u8> = b"Caf\xe9 25\xb0C \xa34 \xfcber\r\nline two\n".to_vec();
    let cases: [(&[u8], &str); 5] =
        [(word.as_bytes(), "utf-8"), (&bom, "utf-8-bom"), (&le, "utf-16le"), (&be, "utf-16be"), (&le_bare, "utf-16le")];
    for (bytes, enc) in cases {
        match decode(bytes) {
            Ok((t, e)) if t == word && e == enc => {}
            other => return Err(format!("nmdoc.text: {enc} decoded as {other:?}")),
        }
    }
    match decode(&cp) {
        Ok((t, "windows-1252")) if t == word => {}
        other => return Err(format!("nmdoc.text: windows-1252 decoded as {other:?}")),
    }
    if decode(b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR\0\0\x01").is_ok() || decode(b"\x80\x93 quoted \x94").map(|x| x.0) != Ok("\u{20ac}\u{201c} quoted \u{201d}".into())
    {
        return Err("nmdoc.text: binary or windows-1252 punctuation".into());
    }
    Ok(())
}
