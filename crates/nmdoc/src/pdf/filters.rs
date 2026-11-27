//! Stream filters: FlateDecode (through `crate::inflate`, with PNG and TIFF predictors),
//! ASCIIHexDecode, ASCII85Decode, LZWDecode (with its early change and predictors) and
//! RunLengthDecode. Image filters (DCT, JPX, JBIG2, CCITT fax) hold no text and are named and not
//! decoded. A damaged Flate stream gives what was inflated before the damage, as viewers do.
use super::lex::{Dict, Obj};
use crate::inflate::{self, Limits};

/// Largest decoded stream, in bytes.
pub const MAX_STREAM: usize = 64 << 20;

/// Why a stream was not decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Skip {
    /// An image filter: the stream is a picture.
    Image(&'static str),
    /// A filter this reader does not know.
    Unknown,
    /// The data could not be decoded at all.
    Broken,
}

/// The filter names and their parameters of a stream dictionary, in order.
fn chain(d: &Dict) -> Vec<(String, Option<Dict>)> {
    let parms: Vec<Option<Dict>> = match d.get("DecodeParms").or_else(|| d.get("DP")) {
        Some(Obj::Array(a)) => a.iter().map(|x| x.as_dict().cloned()).collect(),
        Some(Obj::Dict(p)) => vec![Some(p.clone())],
        _ => Vec::new(),
    };
    let names: Vec<String> = match d.get("Filter").or_else(|| d.get("F")) {
        Some(Obj::Name(n)) => vec![n.clone()],
        Some(Obj::Array(a)) => a.iter().filter_map(|x| x.as_name().map(str::to_owned)).collect(),
        _ => Vec::new(),
    };
    names.into_iter().enumerate().map(|(i, n)| (n, parms.get(i).cloned().flatten())).collect()
}

/// True when the stream's last filter is an image filter (a picture).
#[must_use]
pub fn is_image_data(d: &Dict) -> bool {
    chain(d).iter().any(|(n, _)| image_filter(n).is_some())
}

fn image_filter(n: &str) -> Option<&'static str> {
    match n {
        "DCTDecode" | "DCT" => Some("DCTDecode"),
        "JPXDecode" => Some("JPXDecode"),
        "JBIG2Decode" => Some("JBIG2Decode"),
        "CCITTFaxDecode" | "CCF" => Some("CCITTFaxDecode"),
        _ => None,
    }
}

/// Decode a stream's bytes through every filter of its dictionary.
/// # Errors
/// Why it was not decoded.
pub fn decode(d: &Dict, data: &[u8]) -> Result<Vec<u8>, Skip> {
    let mut cur = data.to_vec();
    for (name, parms) in chain(d) {
        if let Some(img) = image_filter(&name) {
            return Err(Skip::Image(img));
        }
        cur = match name.as_str() {
            "FlateDecode" | "Fl" => predict(flate(&cur)?, parms.as_ref()),
            "LZWDecode" | "LZW" => {
                let early = parms.as_ref().and_then(|p| p.int("EarlyChange")).unwrap_or(1) != 0;
                predict(lzw(&cur, early), parms.as_ref())
            }
            "ASCIIHexDecode" | "AHx" => ascii_hex(&cur),
            "ASCII85Decode" | "A85" => ascii85(&cur),
            "RunLengthDecode" | "RL" => run_length(&cur),
            "Crypt" => cur,
            _ => return Err(Skip::Unknown),
        };
        if cur.len() > MAX_STREAM {
            return Err(Skip::Broken);
        }
    }
    Ok(cur)
}

/// Inflate a zlib stream; a bad checksum or a damaged end keeps what was inflated.
fn flate(data: &[u8]) -> Result<Vec<u8>, Skip> {
    let lim = Limits { max_out: MAX_STREAM, max_ratio: 1100 };
    if let Ok(v) = inflate::zlib(data, &lim) {
        return Ok(v);
    }
    // Skip the zlib header when it is there, else read the bytes as raw DEFLATE.
    let body = match (data.first(), data.get(1)) {
        (Some(&c), Some(&f)) if c & 15 == 8 && (u16::from(c) << 8 | u16::from(f)) % 31 == 0 => data.get(2..).unwrap_or(&[]),
        _ => data,
    };
    let mut out = Vec::new();
    match inflate::inflate_into(body, &mut out, &lim) {
        Ok(_) => Ok(out),
        Err(inflate::Fault::TooLarge | inflate::Fault::Ratio) => Err(Skip::Broken),
        Err(_) if !out.is_empty() => Ok(out),
        Err(_) => Err(Skip::Broken),
    }
}

/// Undo a PNG (10 to 15) or TIFF (2) predictor.
fn predict(data: Vec<u8>, parms: Option<&Dict>) -> Vec<u8> {
    let Some(p) = parms else { return data };
    let pred = p.int("Predictor").unwrap_or(1);
    if pred < 2 {
        return data;
    }
    let colors = usize::try_from(p.int("Colors").unwrap_or(1).clamp(1, 32)).unwrap_or(1);
    let bpc = usize::try_from(p.int("BitsPerComponent").unwrap_or(8).clamp(1, 16)).unwrap_or(8);
    let cols = usize::try_from(p.int("Columns").unwrap_or(1).clamp(1, 1 << 20)).unwrap_or(1);
    let bpp = (colors * bpc).div_ceil(8).max(1);
    let row = (colors * bpc * cols).div_ceil(8);
    if pred == 2 {
        // TIFF predictor 2, for 8-bit components only (the common case).
        if bpc != 8 {
            return data;
        }
        let mut out = data;
        for r in out.chunks_mut(row) {
            for i in bpp..r.len() {
                r[i] = r[i].wrapping_add(r[i - bpp]);
            }
        }
        return out;
    }
    let mut out = Vec::with_capacity(data.len());
    let mut prev = vec![0u8; row];
    for chunk in data.chunks(row + 1) {
        let Some((&kind, line)) = chunk.split_first() else { break };
        let mut cur = line.to_vec();
        cur.resize(row, 0);
        for i in 0..row {
            let a = if i >= bpp { cur[i - bpp] } else { 0 };
            let b = prev[i];
            let c = if i >= bpp { prev[i - bpp] } else { 0 };
            cur[i] = match kind {
                1 => cur[i].wrapping_add(a),
                2 => cur[i].wrapping_add(b),
                3 => cur[i].wrapping_add(((u16::from(a) + u16::from(b)) / 2) as u8),
                4 => {
                    let p = i16::from(a) + i16::from(b) - i16::from(c);
                    let (pa, pb, pc) = ((p - i16::from(a)).abs(), (p - i16::from(b)).abs(), (p - i16::from(c)).abs());
                    cur[i].wrapping_add(if pa <= pb && pa <= pc { a } else if pb <= pc { b } else { c })
                }
                _ => cur[i],
            };
        }
        out.extend_from_slice(&cur[..line.len().min(row)]);
        prev = cur;
    }
    out
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

fn ascii_hex(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 2);
    let mut half: Option<u8> = None;
    for &c in data {
        if c == b'>' {
            break;
        }
        let Some(v) = hex_val(c) else { continue };
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

fn ascii85(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4 / 5);
    let mut group = [0u32; 5];
    let mut n = 0;
    let body = data.strip_prefix(b"<~").unwrap_or(data);
    for &c in body {
        match c {
            b'~' => break,
            b'z' if n == 0 => out.extend_from_slice(&[0; 4]),
            b'!'..=b'u' => {
                group[n] = u32::from(c - b'!');
                n += 1;
                if n == 5 {
                    let v = group.iter().fold(0u32, |acc, d| acc.wrapping_mul(85).wrapping_add(*d));
                    out.extend_from_slice(&v.to_be_bytes());
                    n = 0;
                }
            }
            _ => {}
        }
    }
    if n > 1 {
        for g in group.iter_mut().skip(n) {
            *g = 84;
        }
        let v = group.iter().fold(0u32, |acc, d| acc.wrapping_mul(85).wrapping_add(*d));
        out.extend_from_slice(&v.to_be_bytes()[..n - 1]);
    }
    out
}

fn run_length(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(&len) = data.get(i) {
        i += 1;
        match len {
            128 => break,
            0..=127 => {
                let n = usize::from(len) + 1;
                out.extend_from_slice(data.get(i..(i + n).min(data.len())).unwrap_or(&[]));
                i += n;
            }
            _ => {
                if let Some(&b) = data.get(i) {
                    out.extend(std::iter::repeat_n(b, 257 - usize::from(len)));
                }
                i += 1;
            }
        }
        if out.len() > MAX_STREAM {
            break;
        }
    }
    out
}

/// LZW with codes of 9 to 12 bits; `early`: the code width grows one code early (the default).
fn lzw(data: &[u8], early: bool) -> Vec<u8> {
    let mut out = Vec::new();
    let mut table: Vec<Vec<u8>> = Vec::new();
    let reset = |t: &mut Vec<Vec<u8>>| {
        t.clear();
        t.extend((0..=255u8).map(|b| vec![b]));
        t.push(Vec::new());
        t.push(Vec::new());
    };
    reset(&mut table);
    let (mut buf, mut nbits, mut pos) = (0u32, 0u32, 0usize);
    let mut width = 9u32;
    let mut prev: Option<Vec<u8>> = None;
    loop {
        while nbits < width {
            let Some(&b) = data.get(pos) else { return out };
            buf = (buf << 8) | u32::from(b);
            nbits += 8;
            pos += 1;
        }
        let code = ((buf >> (nbits - width)) & ((1 << width) - 1)) as usize;
        nbits -= width;
        buf &= (1u32 << nbits).wrapping_sub(1);
        match code {
            256 => {
                reset(&mut table);
                width = 9;
                prev = None;
                continue;
            }
            257 => return out,
            _ => {}
        }
        let entry = if let Some(e) = table.get(code).filter(|e| !e.is_empty()) {
            e.clone()
        } else if code == table.len()
            && let Some(p) = &prev
        {
            let mut e = p.clone();
            e.push(p.first().copied().unwrap_or(0));
            e
        } else {
            return out;
        };
        out.extend_from_slice(&entry);
        if let Some(p) = prev.take()
            && table.len() < 4096
        {
            let mut e = p;
            e.push(entry.first().copied().unwrap_or(0));
            table.push(e);
        }
        prev = Some(entry);
        let next = table.len() + usize::from(early);
        width = if next >= 2048 { 12 } else if next >= 1024 { 11 } else if next >= 512 { 10 } else { 9 };
        if out.len() > MAX_STREAM {
            return out;
        }
    }
}

pub(crate) fn check() -> Result<(), String> {
    if ascii_hex(b"48 65 6C6c 6f>") != b"Hello" || ascii_hex(b"414") != b"A@" {
        return Err("nmdoc.pdf.filters: ascii hex".into());
    }
    // The start of the classic ASCII85 example ("Man sure"), with a partial last group, and a z group.
    if ascii85(b"<~9jqo^F*2M~>") != b"Man sur" || ascii85(b"z~>") != [0, 0, 0, 0] {
        return Err(format!("nmdoc.pdf.filters: ascii85 {:?}", ascii85(b"<~9jqo^F*2M~>")));
    }
    if run_length(&[2, b'a', b'b', b'c', 254, b'x', 128]) != b"abcxxx" {
        return Err("nmdoc.pdf.filters: run length".into());
    }
    // The LZW example of the PDF reference: 45 45 45 45 45 65 45 45 45 66.
    let lzw_in = [0x80, 0x0B, 0x60, 0x50, 0x22, 0x0C, 0x0C, 0x85, 0x01];
    if lzw(&lzw_in, true) != [45, 45, 45, 45, 45, 65, 45, 45, 45, 66] {
        return Err(format!("nmdoc.pdf.filters: lzw {:?}", lzw(&lzw_in, true)));
    }
    // PNG Up predictor over two rows of three bytes.
    let mut p = Dict::default();
    p.set("Predictor", Obj::Int(12));
    p.set("Columns", Obj::Int(3));
    if predict(vec![2, 1, 2, 3, 2, 1, 1, 1], Some(&p)) != [1, 2, 3, 2, 3, 4] {
        return Err("nmdoc.pdf.filters: png predictor".into());
    }
    let mut d = Dict::default();
    d.set("Filter", Obj::Name("DCTDecode".into()));
    if decode(&d, b"\xFF\xD8") != Err(Skip::Image("DCTDecode")) {
        return Err("nmdoc.pdf.filters: image filter not named".into());
    }
    // Damaged and random input never panics.
    for bad in [&b"\x78\x9c\xff\xff"[..], b"", b"\x80\x80\x80", b"~~~"] {
        let _ = flate(bad);
        let _ = lzw(bad, true);
        let _ = run_length(bad);
        let _ = ascii85(bad);
    }
    Ok(())
}
