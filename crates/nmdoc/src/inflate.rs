//! Inflate (DEFLATE, RFC 1951), the zlib wrapper (RFC 1950), the gzip wrapper (RFC 1952),
//! CRC-32 and Adler-32, written from the specifications. Every fault is named and nothing panics
//! on bad input. Limits guard against decompression bombs: a cap on the output, and on the ratio
//! of output to input once the output passes `RATIO_FROM` bytes.

/// Why a stream could not be inflated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    /// The data ended inside a block or a trailer.
    Truncated,
    /// Block type 3 (reserved).
    BlockType,
    /// A stored block whose length and its complement disagree.
    StoredLength,
    /// A code length table that is over-subscribed, overruns its count, or has no end code.
    CodeLengths,
    /// A bit pattern that is no code of the table in use, or a reserved symbol.
    Code,
    /// A distance reaching back before the start of the output.
    Distance,
    /// Output over the limit.
    TooLarge,
    /// Output over the allowed multiple of the input (a decompression bomb).
    Ratio,
    /// A zlib or gzip header that is not valid.
    Header,
    /// A zlib stream that needs a preset dictionary.
    Dictionary,
    /// The checksum in the trailer does not match the output.
    Checksum,
    /// The length in the trailer does not match the output.
    Length,
}

impl Fault {
    /// The fault's name, as a skip reason.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Truncated => "compressed data: truncated",
            Self::BlockType => "compressed data: reserved block type",
            Self::StoredLength => "compressed data: stored block length check failed",
            Self::CodeLengths => "compressed data: invalid code lengths",
            Self::Code => "compressed data: invalid code",
            Self::Distance => "compressed data: distance before the start",
            Self::TooLarge => "compressed data: output over the size limit",
            Self::Ratio => "compressed data: expands too far (decompression bomb guard)",
            Self::Header => "compressed data: invalid header",
            Self::Dictionary => "compressed data: needs a preset dictionary",
            Self::Checksum => "compressed data: checksum mismatch",
            Self::Length => "compressed data: length mismatch",
        }
    }
}

/// Output at which the ratio guard starts to apply.
pub const RATIO_FROM: usize = 1 << 20;

/// Limits on one inflate: the largest output, and the largest ratio of output to input bytes
/// read (checked once the output passes `RATIO_FROM`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_out: usize,
    pub max_ratio: usize,
}

impl Default for Limits {
    /// 64 MB of output and 200 times the input. Text compresses about 3 to 10 times and markup
    /// up to about 30 times; DEFLATE itself cannot pass about 1,032 times.
    fn default() -> Self {
        Self { max_out: 64 << 20, max_ratio: 200 }
    }
}

/// Bit reader, least significant bit first.
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    n: u32,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0, buf: 0, n: 0 }
    }
    /// Fill the buffer with as many whole bytes as fit.
    fn fill(&mut self) {
        while self.n <= 56 {
            let Some(b) = self.data.get(self.pos) else { break };
            self.buf |= u64::from(*b) << self.n;
            self.n += 8;
            self.pos += 1;
        }
    }
    fn bits(&mut self, k: u32) -> Result<u32, Fault> {
        if self.n < k {
            self.fill();
            if self.n < k {
                return Err(Fault::Truncated);
            }
        }
        let v = (self.buf & ((1_u64 << k) - 1)) as u32;
        self.buf >>= k;
        self.n -= k;
        Ok(v)
    }
    /// Drop the bits up to the next byte boundary.
    fn align(&mut self) {
        let r = self.n % 8;
        self.buf >>= r;
        self.n -= r;
    }
    fn byte(&mut self) -> Result<u8, Fault> {
        if self.n >= 8 {
            let b = (self.buf & 0xFF) as u8;
            self.buf >>= 8;
            self.n -= 8;
            return Ok(b);
        }
        let b = *self.data.get(self.pos).ok_or(Fault::Truncated)?;
        self.pos += 1;
        Ok(b)
    }
    /// Input bytes consumed so far (whole bytes still in the buffer are given back).
    fn used(&self) -> usize {
        self.pos - (self.n / 8) as usize
    }
}

/// A canonical Huffman code as a lookup table indexed by the next `bits` input bits (the code
/// bit-reversed, since codes are packed from their most significant bit). Each entry holds
/// `symbol << 4 | length`; length 0 marks a pattern that is no code.
struct Huff {
    table: Vec<u32>,
    bits: u32,
}

impl Huff {
    fn new(lengths: &[u8]) -> Result<Self, Fault> {
        let mut count = [0_u32; 16];
        for &l in lengths {
            count[usize::from(l & 15)] += 1;
        }
        count[0] = 0;
        let mut left: i64 = 1;
        for c in count.iter().skip(1) {
            left = (left << 1) - i64::from(*c);
            if left < 0 {
                return Err(Fault::CodeLengths);
            }
        }
        let bits = (1..16_u32).rev().find(|l| count[*l as usize] > 0).unwrap_or(1);
        let mut next = [0_u32; 16];
        let mut code = 0_u32;
        for l in 1..16 {
            code = (code + count[l - 1]) << 1;
            next[l] = code;
        }
        let size = 1_usize << bits;
        let mut table = vec![0_u32; size];
        for (sym, &l) in lengths.iter().enumerate() {
            if l == 0 {
                continue;
            }
            let l = u32::from(l);
            let c = next[l as usize];
            next[l as usize] += 1;
            let rev = c.reverse_bits() >> (32 - l);
            let entry = ((sym as u32) << 4) | l;
            let mut k = rev as usize;
            while k < size {
                if let Some(t) = table.get_mut(k) {
                    *t = entry;
                }
                k += 1 << l;
            }
        }
        Ok(Self { table, bits })
    }
    fn decode(&self, r: &mut Bits<'_>) -> Result<usize, Fault> {
        if r.n < self.bits {
            r.fill();
        }
        let avail = r.n.min(self.bits);
        let idx = (r.buf & ((1_u64 << self.bits) - 1)) as usize;
        let e = self.table.get(idx).copied().unwrap_or(0);
        let len = e & 15;
        if len == 0 {
            return Err(if avail < self.bits { Fault::Truncated } else { Fault::Code });
        }
        if len > avail {
            return Err(Fault::Truncated);
        }
        r.buf >>= len;
        r.n -= len;
        Ok((e >> 4) as usize)
    }
}

const LBASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEXT: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DBASE: [u16; 30] =
    [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DEXT: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
/// Order in which the code length code lengths are sent.
const CL_ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

fn guard(out: &[u8], used: usize, lim: &Limits) -> Result<(), Fault> {
    if out.len() > lim.max_out {
        return Err(Fault::TooLarge);
    }
    if out.len() > RATIO_FROM && out.len() / used.max(1) > lim.max_ratio {
        return Err(Fault::Ratio);
    }
    Ok(())
}

fn codes(r: &mut Bits<'_>, out: &mut Vec<u8>, lit: &Huff, dist: &Huff, lim: &Limits) -> Result<(), Fault> {
    let mut since = 0_usize;
    loop {
        let sym = lit.decode(r)?;
        if sym < 256 {
            out.push(sym as u8);
            since += 1;
        } else if sym == 256 {
            return guard(out, r.used(), lim);
        } else {
            let i = sym - 257;
            let (Some(base), Some(ext)) = (LBASE.get(i), LEXT.get(i)) else { return Err(Fault::Code) };
            let len = usize::from(*base) + r.bits(u32::from(*ext))? as usize;
            let d = dist.decode(r)?;
            let (Some(base), Some(ext)) = (DBASE.get(d), DEXT.get(d)) else { return Err(Fault::Code) };
            let back = usize::from(*base) + r.bits(u32::from(*ext))? as usize;
            if back > out.len() {
                return Err(Fault::Distance);
            }
            let from = out.len() - back;
            for k in 0..len {
                let b = out.get(from + k).copied().ok_or(Fault::Distance)?;
                out.push(b);
            }
            since += len;
        }
        if since >= 1 << 16 {
            since = 0;
            guard(out, r.used(), lim)?;
        }
    }
}

fn stored(r: &mut Bits<'_>, out: &mut Vec<u8>, lim: &Limits) -> Result<(), Fault> {
    r.align();
    let len = r.bits(16)?;
    let nlen = r.bits(16)?;
    if len != !nlen & 0xFFFF {
        return Err(Fault::StoredLength);
    }
    for _ in 0..len {
        out.push(r.byte()?);
    }
    guard(out, r.used(), lim)
}

fn fixed() -> Result<(Huff, Huff), Fault> {
    let mut l = [0_u8; 288];
    for (i, x) in l.iter_mut().enumerate() {
        *x = match i {
            0..=143 => 8,
            144..=255 => 9,
            256..=279 => 7,
            _ => 8,
        };
    }
    Ok((Huff::new(&l)?, Huff::new(&[5_u8; 30])?))
}

fn dynamic(r: &mut Bits<'_>) -> Result<(Huff, Huff), Fault> {
    let hlit = r.bits(5)? as usize + 257;
    let hdist = r.bits(5)? as usize + 1;
    let hclen = r.bits(4)? as usize + 4;
    if hlit > 286 || hdist > 30 {
        return Err(Fault::CodeLengths);
    }
    let mut cl = [0_u8; 19];
    for &i in CL_ORDER.iter().take(hclen) {
        if let Some(x) = cl.get_mut(i) {
            *x = r.bits(3)? as u8;
        }
    }
    let clh = Huff::new(&cl)?;
    let mut lengths = vec![0_u8; hlit + hdist];
    let mut i = 0;
    while i < hlit + hdist {
        let sym = clh.decode(r)?;
        let (value, times) = match sym {
            0..=15 => (sym as u8, 1),
            16 => {
                let prev = *lengths.get(i.wrapping_sub(1)).ok_or(Fault::CodeLengths)?;
                (prev, 3 + r.bits(2)? as usize)
            }
            17 => (0, 3 + r.bits(3)? as usize),
            18 => (0, 11 + r.bits(7)? as usize),
            _ => return Err(Fault::CodeLengths),
        };
        if i + times > hlit + hdist {
            return Err(Fault::CodeLengths);
        }
        for x in lengths.iter_mut().skip(i).take(times) {
            *x = value;
        }
        i += times;
    }
    if lengths.get(256).copied().unwrap_or(0) == 0 {
        return Err(Fault::CodeLengths);
    }
    let (l, d) = lengths.split_at(hlit);
    Ok((Huff::new(l)?, Huff::new(d)?))
}

/// Inflate a raw DEFLATE stream onto `out`. Returns the input bytes the stream used.
/// # Errors
/// The named fault; `out` then holds what was inflated before it.
pub fn inflate_into(data: &[u8], out: &mut Vec<u8>, lim: &Limits) -> Result<usize, Fault> {
    let mut r = Bits::new(data);
    loop {
        let last = r.bits(1)? == 1;
        match r.bits(2)? {
            0 => stored(&mut r, out, lim)?,
            1 => {
                let (l, d) = fixed()?;
                codes(&mut r, out, &l, &d, lim)?;
            }
            2 => {
                let (l, d) = dynamic(&mut r)?;
                codes(&mut r, out, &l, &d, lim)?;
            }
            _ => return Err(Fault::BlockType),
        }
        if last {
            return Ok(r.used());
        }
    }
}

/// Inflate a raw DEFLATE stream.
/// # Errors
/// The named fault.
pub fn inflate(data: &[u8], lim: &Limits) -> Result<Vec<u8>, Fault> {
    let mut out = Vec::new();
    inflate_into(data, &mut out, lim)?;
    Ok(out)
}

/// CRC-32 table (polynomial 0xEDB88320, reflected), built at compile time.
const CRC: [u32; 256] = {
    let mut t = [0_u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 == 1 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
};

/// CRC-32 of bytes (as zip and gzip use it).
#[must_use]
pub fn crc32(data: &[u8]) -> u32 {
    let mut c = !0_u32;
    for &b in data {
        c = CRC[((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8);
    }
    !c
}

/// Adler-32 of bytes (as zlib uses it).
#[must_use]
pub fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1_u32, 0_u32);
    for chunk in data.chunks(5552) {
        for &x in chunk {
            a += u32::from(x);
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

/// Inflate a zlib stream (RFC 1950) and check its Adler-32.
/// # Errors
/// The named fault.
pub fn zlib(data: &[u8], lim: &Limits) -> Result<Vec<u8>, Fault> {
    let (Some(&cmf), Some(&flg)) = (data.first(), data.get(1)) else { return Err(Fault::Truncated) };
    if cmf & 15 != 8 || cmf >> 4 > 7 || (u16::from(cmf) << 8 | u16::from(flg)) % 31 != 0 {
        return Err(Fault::Header);
    }
    if flg & 0x20 != 0 {
        return Err(Fault::Dictionary);
    }
    let body = data.get(2..).unwrap_or(&[]);
    let mut out = Vec::new();
    let used = inflate_into(body, &mut out, lim)?;
    let tail = body.get(used..used + 4).ok_or(Fault::Truncated)?;
    let want = u32::from_be_bytes([tail[0], tail[1], tail[2], tail[3]]);
    if want != adler32(&out) {
        return Err(Fault::Checksum);
    }
    Ok(out)
}

/// What a gzip file held: the original name from its header (empty when not given) and the
/// inflated bytes of all its members, one after the other.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gzip {
    pub name: String,
    pub data: Vec<u8>,
}

/// True when the bytes start like a gzip member.
#[must_use]
pub fn is_gzip(data: &[u8]) -> bool {
    data.starts_with(&[0x1F, 0x8B, 0x08])
}

/// Inflate a gzip file (RFC 1952): every member, each checked by its CRC-32 and length.
/// # Errors
/// The named fault.
pub fn gzip(data: &[u8], lim: &Limits) -> Result<Gzip, Fault> {
    let mut out = Vec::new();
    let mut name = String::new();
    let mut at = 0_usize;
    let mut first = true;
    loop {
        let rest = data.get(at..).unwrap_or(&[]);
        // After the first member, anything that is not another member (padding, trailing data)
        // ends the file.
        if !first && !is_gzip(rest) {
            return Ok(Gzip { name, data: out });
        }
        if !is_gzip(rest) {
            return Err(Fault::Header);
        }
        let flg = *rest.get(3).ok_or(Fault::Truncated)?;
        if flg & 0xE0 != 0 {
            return Err(Fault::Header);
        }
        let mut p = 10_usize;
        if rest.len() < p {
            return Err(Fault::Truncated);
        }
        if flg & 4 != 0 {
            let x = rest.get(p..p + 2).ok_or(Fault::Truncated)?;
            p += 2 + usize::from(u16::from_le_bytes([x[0], x[1]]));
        }
        for bit in [8_u8, 16] {
            if flg & bit != 0 {
                let end = rest.get(p..).and_then(|s| s.iter().position(|b| *b == 0)).ok_or(Fault::Truncated)?;
                if bit == 8 && first {
                    name = crate::text::windows_1252(rest.get(p..p + end).unwrap_or(&[]));
                }
                p += end + 1;
            }
        }
        if flg & 2 != 0 {
            p += 2;
        }
        let body = rest.get(p..).ok_or(Fault::Truncated)?;
        let start = out.len();
        let used = inflate_into(body, &mut out, lim)?;
        let t = body.get(used..used + 8).ok_or(Fault::Truncated)?;
        let crc = u32::from_le_bytes([t[0], t[1], t[2], t[3]]);
        let size = u32::from_le_bytes([t[4], t[5], t[6], t[7]]);
        let member = out.get(start..).unwrap_or(&[]);
        if crc != crc32(member) {
            return Err(Fault::Checksum);
        }
        if size != member.len() as u32 {
            return Err(Fault::Length);
        }
        at += p + used + 8;
        first = false;
    }
}

pub(crate) fn check() -> Result<(), String> {
    // A stored block, a fixed block with a back reference, and a dynamic block made by hand are
    // covered by the fixtures (`check documents`); here: the checksums, a stored block, a fixed
    // block and the named faults.
    if crc32(b"123456789") != 0xCBF4_3926 || adler32(b"Wikipedia") != 0x11E6_0398 {
        return Err("nmdoc.inflate: checksums".into());
    }
    let lim = Limits::default();
    // Stored, final: header bits 1 (final) 00 (stored), then LEN 5, NLEN.
    let stored = [0x01, 0x05, 0x00, 0xFA, 0xFF, b'h', b'e', b'l', b'l', b'o'];
    // Fixed Huffman "a" then a length 3, distance 1 back reference and the end code
    // (the zlib output for "aaaa" at level 9 without its wrapper).
    let fixed = [0x4B, 0x4C, 0x4C, 0x4C, 0x04, 0x00];
    if inflate(&stored, &lim).as_deref() != Ok(b"hello") || inflate(&fixed, &lim).as_deref() != Ok(b"aaaa") {
        return Err(format!("nmdoc.inflate: stored {:?}, fixed {:?}", inflate(&stored, &lim), inflate(&fixed, &lim)));
    }
    let faults: [(&[u8], Fault); 4] = [
        (&[0x07], Fault::BlockType),
        (&[0x01, 0x05, 0x00, 0x00, 0x00], Fault::StoredLength),
        (&stored[..7], Fault::Truncated),
        // Fixed block whose first code is a back reference: length 3, distance 1, nothing before it.
        (&[0x03, 0x02], Fault::Distance),
    ];
    for (data, want) in faults {
        let got = inflate(data, &lim);
        if got != Err(want) {
            return Err(format!("nmdoc.inflate: {data:?} gave {got:?}, want {want:?}"));
        }
    }
    let z = [0x78, 0x9C, 0x4B, 0x4C, 0x4C, 0x4C, 0x04, 0x00, 0x03, 0xCE, 0x01, 0x85];
    if zlib(&z, &lim).as_deref() != Ok(b"aaaa") {
        return Err(format!("nmdoc.inflate: zlib {:?}", zlib(&z, &lim)));
    }
    let mut bad = z;
    bad[11] ^= 1;
    if zlib(&bad, &lim) != Err(Fault::Checksum) {
        return Err("nmdoc.inflate: zlib checksum not checked".into());
    }
    Ok(())
}
