//! Tar archives, read in memory: 512-byte headers (ustar, with the name prefix), checksums,
//! sizes in octal or base-256, GNU long names and pax `path` records. Regular files are listed;
//! folders, links and devices are passed over.

/// One regular file: its name and the byte range of its data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Member {
    pub name: String,
    pub start: usize,
    pub len: usize,
}

fn field(h: &[u8], a: usize, b: usize) -> &[u8] {
    let f = h.get(a..b).unwrap_or(&[]);
    let end = f.iter().position(|x| *x == 0).unwrap_or(f.len());
    f.get(..end).unwrap_or(&[])
}

fn number(f: &[u8]) -> Option<u64> {
    if f.first().is_some_and(|b| b & 0x80 != 0) {
        // Base-256: the remaining bits, big endian.
        let mut v: u64 = u64::from(f.first().copied().unwrap_or(0) & 0x7F);
        for &b in f.get(1..).unwrap_or(&[]) {
            v = v.checked_mul(256)?.checked_add(u64::from(b))?;
        }
        return Some(v);
    }
    let s = std::str::from_utf8(f).ok()?.trim_matches(|c: char| c == ' ' || c == '\0');
    if s.is_empty() {
        return Some(0);
    }
    u64::from_str_radix(s, 8).ok()
}

fn checksum_ok(h: &[u8]) -> bool {
    let Some(want) = number(h.get(148..156).unwrap_or(&[])) else { return false };
    let sum: u64 = h.iter().enumerate().map(|(i, b)| if (148..156).contains(&i) { 32 } else { u64::from(*b) }).sum();
    sum == want
}

/// True when the bytes start with a valid tar header.
#[must_use]
pub fn is_tar(b: &[u8]) -> bool {
    b.get(..512).is_some_and(|h| h.iter().any(|x| *x != 0) && checksum_ok(h))
}

/// The regular files of a tar archive.
/// # Errors
/// A named fault: a header whose checksum fails, a size past the end, too many members.
pub fn members(b: &[u8]) -> Result<Vec<Member>, &'static str> {
    let mut out = Vec::new();
    let mut at = 0_usize;
    let mut long: Option<String> = None;
    while let Some(h) = b.get(at..at + 512) {
        if h.iter().all(|x| *x == 0) {
            break;
        }
        if !checksum_ok(h) {
            return Err("tar: header checksum failed");
        }
        let size = usize::try_from(number(field(h, 124, 136)).ok_or("tar: bad size")?).map_err(|_| "tar: bad size")?;
        let start = at + 512;
        let data = b.get(start..start.checked_add(size).ok_or("tar: bad size")?).ok_or("tar: member past the end of the archive")?;
        let kind = h.get(156).copied().unwrap_or(0);
        match kind {
            b'L' => long = Some(String::from_utf8_lossy(field(data, 0, data.len())).into_owned()),
            b'x' => {
                // pax records: "<len> <key>=<value>\n"
                for rec in String::from_utf8_lossy(data).split('\n') {
                    if let Some((_, kv)) = rec.split_once(' ')
                        && let Some(p) = kv.strip_prefix("path=")
                    {
                        long = Some(p.to_owned());
                    }
                }
            }
            b'0' | 0 | b'7' => {
                let name = long.take().unwrap_or_else(|| {
                    let (prefix, name) = (field(h, 345, 500), field(h, 0, 100));
                    let ustar = h.get(257..262) == Some(b"ustar");
                    let name = String::from_utf8_lossy(name).into_owned();
                    if ustar && !prefix.is_empty() { format!("{}/{name}", String::from_utf8_lossy(prefix)) } else { name }
                });
                if out.len() >= crate::zip::MAX_MEMBERS {
                    return Err("tar: too many members");
                }
                out.push(Member { name: name.replace('\\', "/"), start, len: size });
            }
            _ => long = None,
        }
        at = start + size.div_ceil(512) * 512;
    }
    Ok(out)
}

/// One tar header block for a regular file (for the checks).
#[must_use]
pub fn header(name: &str, size: usize) -> Vec<u8> {
    let mut h = vec![0_u8; 512];
    for (d, s) in h.iter_mut().zip(name.bytes().take(100)) {
        *d = s;
    }
    let mut put = |at: usize, s: &str| {
        for (i, c) in s.bytes().enumerate() {
            if let Some(d) = h.get_mut(at + i) {
                *d = c;
            }
        }
    };
    put(100, "0000644");
    put(124, &format!("{size:011o}"));
    put(156, "0");
    put(257, "ustar");
    put(263, "00");
    put(148, "        ");
    let sum: u32 = h.iter().map(|b| u32::from(*b)).sum();
    let s = format!("{sum:06o}\0 ");
    for (i, c) in s.bytes().enumerate() {
        if let Some(d) = h.get_mut(148 + i) {
            *d = c;
        }
    }
    h
}

pub(crate) fn check() -> Result<(), String> {
    let body = b"valve torque 12 Nm\n";
    let mut t = header("notes/valve.txt", body.len());
    t.extend(body);
    t.resize(1024, 0);
    t.extend([0_u8; 1024]);
    if !is_tar(&t) {
        return Err("nmdoc.tar: header not recognised".into());
    }
    let m = members(&t).map_err(|e| format!("nmdoc.tar: {e}"))?;
    let got: Vec<(&str, &[u8])> = m.iter().map(|x| (x.name.as_str(), t.get(x.start..x.start + x.len).unwrap_or(&[]))).collect();
    if got != [("notes/valve.txt", &body[..])] {
        return Err(format!("nmdoc.tar: members {got:?}"));
    }
    let mut bad = t.clone();
    if let Some(b) = bad.get_mut(3) {
        *b ^= 1;
    }
    if members(&bad).is_ok() || is_tar(b"plain text that is no archive") {
        return Err("nmdoc.tar: a bad header was accepted".into());
    }
    Ok(())
}
