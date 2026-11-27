//! Zip containers, read in memory: the end of central directory record (and its ZIP64 form),
//! the central directory, local headers, members stored (method 0) or deflated (method 8), each
//! checked by its CRC-32 and size. Nothing is extracted to disk and member names are never used
//! as disk paths. Encrypted members are listed by name and not read.
use crate::inflate::{self, Limits};

/// Most members read from one container.
pub const MAX_MEMBERS: usize = 20_000;

/// One member as the central directory lists it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub method: u16,
    pub flags: u16,
    pub crc: u32,
    pub csize: u64,
    pub size: u64,
    /// Offset of the local header (prepended data already accounted for).
    pub offset: u64,
}

impl Entry {
    /// True when the member is encrypted (traditional or strong encryption, or AES).
    #[must_use]
    pub fn encrypted(&self) -> bool {
        self.flags & 1 != 0 || self.method == 99
    }
    /// True for a folder entry.
    #[must_use]
    pub fn is_dir(&self) -> bool {
        self.name.ends_with('/')
    }
}

fn u16_at(b: &[u8], at: usize) -> Option<u16> {
    b.get(at..at + 2).map(|x| u16::from_le_bytes([x[0], x[1]]))
}
fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    b.get(at..at + 4).map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]))
}
fn u64_at(b: &[u8], at: usize) -> Option<u64> {
    b.get(at..at + 8).map(|x| u64::from_le_bytes([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]]))
}

/// True when the bytes start like a zip file (a local header, or the record of an empty zip).
#[must_use]
pub fn is_zip(b: &[u8]) -> bool {
    b.starts_with(b"PK\x03\x04") || b.starts_with(b"PK\x05\x06")
}

/// The members listed in the central directory, folders included.
/// # Errors
/// A named fault: no end record, a directory outside the file, a malformed entry, too many members.
pub fn entries(b: &[u8]) -> Result<Vec<Entry>, &'static str> {
    const BAD: &str = "zip: malformed central directory";
    let lo = b.len().saturating_sub(22 + 65_535);
    let eocd = (lo..=b.len().saturating_sub(22)).rev().find(|&i| b.get(i..i + 4) == Some(b"PK\x05\x06")).ok_or("zip: no end of central directory record")?;
    let mut count = u64::from(u16_at(b, eocd + 10).ok_or(BAD)?);
    let mut cd_size = u64::from(u32_at(b, eocd + 12).ok_or(BAD)?);
    let mut cd_off = u64::from(u32_at(b, eocd + 16).ok_or(BAD)?);
    let mut end = eocd as u64;
    if (count == 0xFFFF || cd_size == 0xFFFF_FFFF || cd_off == 0xFFFF_FFFF) && eocd >= 20 && b.get(eocd - 20..eocd - 16) == Some(b"PK\x06\x07") {
        let at = usize::try_from(u64_at(b, eocd - 12).ok_or(BAD)?).map_err(|_| BAD)?;
        if b.get(at..at + 4) == Some(b"PK\x06\x06") {
            count = u64_at(b, at + 32).ok_or(BAD)?;
            cd_size = u64_at(b, at + 40).ok_or(BAD)?;
            cd_off = u64_at(b, at + 48).ok_or(BAD)?;
            end = at as u64;
        }
    }
    if count > MAX_MEMBERS as u64 {
        return Err("zip: too many members");
    }
    // Data prepended to the archive (a self-extracting stub) moves every offset by the same amount.
    let base = end.checked_sub(cd_size).and_then(|s| s.checked_sub(cd_off)).ok_or("zip: central directory outside the file")?;
    let mut at = usize::try_from(base + cd_off).map_err(|_| BAD)?;
    let mut out = Vec::new();
    for _ in 0..count {
        if b.get(at..at + 4) != Some(b"PK\x01\x02") {
            return Err(BAD);
        }
        let flags = u16_at(b, at + 8).ok_or(BAD)?;
        let method = u16_at(b, at + 10).ok_or(BAD)?;
        let crc = u32_at(b, at + 16).ok_or(BAD)?;
        let mut csize = u64::from(u32_at(b, at + 20).ok_or(BAD)?);
        let mut size = u64::from(u32_at(b, at + 24).ok_or(BAD)?);
        let nlen = usize::from(u16_at(b, at + 28).ok_or(BAD)?);
        let xlen = usize::from(u16_at(b, at + 30).ok_or(BAD)?);
        let clen = usize::from(u16_at(b, at + 32).ok_or(BAD)?);
        let mut offset = u64::from(u32_at(b, at + 42).ok_or(BAD)?);
        let raw = b.get(at + 46..at + 46 + nlen).ok_or(BAD)?;
        let name = if flags & 0x800 != 0 {
            String::from_utf8_lossy(raw).into_owned()
        } else {
            std::str::from_utf8(raw).map_or_else(|_| crate::text::windows_1252(raw), str::to_owned)
        };
        // ZIP64 extra field: the 64-bit values of the fields that hold 0xFFFFFFFF, in order.
        let extra = b.get(at + 46 + nlen..at + 46 + nlen + xlen).ok_or(BAD)?;
        let mut x = 0;
        while x + 4 <= extra.len() {
            let id = u16_at(extra, x).ok_or(BAD)?;
            let len = usize::from(u16_at(extra, x + 2).ok_or(BAD)?);
            if id == 1 {
                let mut p = x + 4;
                for field in [&mut size, &mut csize, &mut offset] {
                    if *field == 0xFFFF_FFFF {
                        *field = u64_at(extra, p).ok_or(BAD)?;
                        p += 8;
                    }
                }
            }
            x += 4 + len;
        }
        out.push(Entry { name: name.replace('\\', "/"), method, flags, crc, csize, size, offset: offset + base });
        at += 46 + nlen + xlen + clen;
    }
    Ok(out)
}

/// The bytes of one member, inflated when deflated, checked by size and CRC-32.
/// # Errors
/// A named fault: encrypted, an unsupported method, a bad local header, an inflate fault, or a
/// size or checksum that does not match.
pub fn data(b: &[u8], e: &Entry, lim: &Limits) -> Result<Vec<u8>, &'static str> {
    if e.encrypted() {
        return Err("encrypted member: name only");
    }
    let at = usize::try_from(e.offset).map_err(|_| "zip: member outside the file")?;
    if b.get(at..at + 4) != Some(b"PK\x03\x04") {
        return Err("zip: bad local header");
    }
    let nlen = usize::from(u16_at(b, at + 26).ok_or("zip: bad local header")?);
    let xlen = usize::from(u16_at(b, at + 28).ok_or("zip: bad local header")?);
    let start = at + 30 + nlen + xlen;
    let len = usize::try_from(e.csize).map_err(|_| "zip: member outside the file")?;
    let raw = b.get(start..start.saturating_add(len)).ok_or("zip: member outside the file")?;
    if e.size > lim.max_out as u64 {
        return Err(inflate::Fault::TooLarge.name());
    }
    let out = match e.method {
        0 => raw.to_vec(),
        8 => inflate::inflate(raw, lim).map_err(inflate::Fault::name)?,
        _ => return Err("zip: compression method not supported (stored and deflate are read)"),
    };
    if out.len() as u64 != e.size {
        return Err(inflate::Fault::Length.name());
    }
    if inflate::crc32(&out) != e.crc {
        return Err(inflate::Fault::Checksum.name());
    }
    Ok(out)
}

/// Builds a zip of stored members in memory (for the checks).
#[must_use]
pub fn stored_zip(members: &[(&str, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut cd = Vec::new();
    for (name, body) in members {
        let off = out.len() as u32;
        let crc = inflate::crc32(body);
        let n = name.len() as u16;
        let sz = body.len() as u32;
        out.extend(b"PK\x03\x04");
        out.extend([20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        out.extend(crc.to_le_bytes());
        out.extend(sz.to_le_bytes());
        out.extend(sz.to_le_bytes());
        out.extend(n.to_le_bytes());
        out.extend([0, 0]);
        out.extend(name.as_bytes());
        out.extend(*body);
        cd.extend(b"PK\x01\x02");
        cd.extend([20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        cd.extend(crc.to_le_bytes());
        cd.extend(sz.to_le_bytes());
        cd.extend(sz.to_le_bytes());
        cd.extend(n.to_le_bytes());
        cd.extend([0; 12]);
        cd.extend(off.to_le_bytes());
        cd.extend(name.as_bytes());
    }
    let (cd_off, cd_len, count) = (out.len() as u32, cd.len() as u32, members.len() as u16);
    out.extend(cd);
    out.extend(b"PK\x05\x06");
    out.extend([0, 0, 0, 0]);
    out.extend(count.to_le_bytes());
    out.extend(count.to_le_bytes());
    out.extend(cd_len.to_le_bytes());
    out.extend(cd_off.to_le_bytes());
    out.extend([0, 0]);
    out
}

pub(crate) fn check() -> Result<(), String> {
    let z = stored_zip(&[("docs/a.txt", b"pump seal"), ("b.md", b"# B\n")]);
    let list = entries(&z).map_err(|e| format!("nmdoc.zip: {e}"))?;
    let names: Vec<&str> = list.iter().map(|e| e.name.as_str()).collect();
    if names != ["docs/a.txt", "b.md"] {
        return Err(format!("nmdoc.zip: names {names:?}"));
    }
    let lim = Limits::default();
    let first = list.first().ok_or("nmdoc.zip: no member")?;
    if data(&z, first, &lim).as_deref() != Ok(b"pump seal") {
        return Err("nmdoc.zip: member bytes".into());
    }
    let mut bad = first.clone();
    bad.crc ^= 1;
    if data(&z, &bad, &lim) != Err(inflate::Fault::Checksum.name()) {
        return Err("nmdoc.zip: checksum not checked".into());
    }
    bad.crc ^= 1;
    bad.flags |= 1;
    if data(&z, &bad, &lim) != Err("encrypted member: name only") {
        return Err("nmdoc.zip: encryption not refused".into());
    }
    let mut prefixed = b"stub bytes before the archive".to_vec();
    prefixed.extend(&z);
    let moved = entries(&prefixed).map_err(|e| format!("nmdoc.zip: prefixed: {e}"))?;
    if moved.first().map(|e| data(&prefixed, e, &lim)) != Some(Ok(b"pump seal".to_vec())) {
        return Err("nmdoc.zip: prepended data".into());
    }
    if entries(b"PK\x03\x04 broken").is_ok() || entries(z.get(..z.len() - 4).unwrap_or(&[])).is_ok() {
        return Err("nmdoc.zip: a broken zip was accepted".into());
    }
    Ok(())
}
