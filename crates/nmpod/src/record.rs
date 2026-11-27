//! The drone's record of one piece (2026-11-13; research file Solution 3, Sensory, "THE SOLUTION":
//! a drone emits per piece about 120 bytes: content hash, path, line span, token count, kind, the
//! raw signals as separate numbers on whole-tree statistics, a standout mask, up to 8 rare-name
//! ids). Test SN3: the stack replayed from records equals the full scan.
//!
//! Layout (little-endian, fixed 76 bytes plus 4 per rare-name id, at most 108):
//! content hash 32 bytes (the chunk key's SHA-256 as bytes); path id u32 (index into the tree's
//! file list); first and last line u32 each; tokens u32; kind u8; standout mask u8 (bit i: signal i
//! stands out in the tree for this request); rare-name count u8; one byte spare; the six signals as
//! f32 each (exact, ident, structural, affinity, BM25, BM25F: full precision, so a replay ranks
//! exactly as the scan); then the rare-name ids, each the FNV-1a hash of a request name found in the
//! piece. The embedder vector lives apart, in RAM only.
use crate::intake::Signals;

/// Bytes of a record without rare-name ids.
pub const FIXED: usize = 32 + 4 * 4 + 4 + 24;
/// Most rare-name ids a record carries.
pub const MAX_RARE: usize = 8;

/// One drone record.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub hash: [u8; 32],
    pub path: u32,
    pub start: u32,
    pub end: u32,
    pub tokens: u32,
    pub kind: u8,
    pub standout: u8,
    pub signals: Signals,
    pub rare: Vec<u32>,
}

/// FNV-1a over a name's bytes.
#[must_use]
pub fn name_id(name: &str) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in name.bytes() {
        h ^= u32::from(b);
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// The six signals in their record order.
#[must_use]
pub fn values(s: &Signals) -> [f32; 6] {
    [s.exact, s.ident, s.structural, s.affinity, s.bm25, s.bm25f]
}

/// Standout masks for every piece: bit i set when signal i of the piece is above the tree's mean
/// plus three standard deviations for that signal on this request (and above zero).
#[must_use]
pub fn standouts(all: &[Signals]) -> Vec<u8> {
    let n = all.len().max(1) as f32;
    let mut bars = [0.0_f32; 6];
    for (i, bar) in bars.iter_mut().enumerate() {
        let v: Vec<f32> = all.iter().map(|s| values(s)[i]).collect();
        let mean = v.iter().sum::<f32>() / n;
        let sd = (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n).sqrt();
        *bar = mean + 3.0 * sd;
    }
    all.iter()
        .map(|s| {
            let v = values(s);
            (0..6).fold(0_u8, |m, i| if v[i] > 0.0 && v[i] > bars[i] { m | (1 << i) } else { m })
        })
        .collect()
}

impl Record {
    /// Encode (little-endian).
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(FIXED + 4 * self.rare.len());
        out.extend_from_slice(&self.hash);
        for x in [self.path, self.start, self.end, self.tokens] {
            out.extend_from_slice(&x.to_le_bytes());
        }
        let rare = self.rare.len().min(MAX_RARE);
        out.extend_from_slice(&[self.kind, self.standout, u8::try_from(rare).unwrap_or(0), 0]);
        for v in values(&self.signals) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for r in self.rare.iter().take(MAX_RARE) {
            out.extend_from_slice(&r.to_le_bytes());
        }
        out
    }
    /// Decode one record from the start of `b`; returns it and its length.
    /// # Errors
    /// Too few bytes.
    pub fn decode(b: &[u8]) -> Result<(Self, usize), String> {
        let short = || "nmpod.record: too few bytes".to_owned();
        let u = |at: usize| -> Result<u32, String> {
            Ok(u32::from_le_bytes(b.get(at..at + 4).ok_or_else(short)?.try_into().map_err(|_| short())?))
        };
        let f = |at: usize| -> Result<f32, String> { Ok(f32::from_bits(u(at)?)) };
        let mut hash = [0_u8; 32];
        hash.copy_from_slice(b.get(..32).ok_or_else(short)?);
        let (path, start, end, tokens) = (u(32)?, u(36)?, u(40)?, u(44)?);
        let (kind, standout, rare_n) =
            (*b.get(48).ok_or_else(short)?, *b.get(49).ok_or_else(short)?, usize::from(*b.get(50).ok_or_else(short)?));
        let signals = Signals { exact: f(52)?, ident: f(56)?, structural: f(60)?, affinity: f(64)?, bm25: f(68)?, bm25f: f(72)? };
        let rare = (0..rare_n.min(MAX_RARE)).map(|i| u(FIXED + 4 * i)).collect::<Result<Vec<u32>, String>>()?;
        Ok((Self { hash, path, start, end, tokens, kind, standout, signals, rare }, FIXED + 4 * rare_n.min(MAX_RARE)))
    }
}

/// Encode many records back to back.
#[must_use]
pub fn encode_all(rs: &[Record]) -> Vec<u8> {
    rs.iter().flat_map(Record::encode).collect()
}

/// Decode records back to back.
/// # Errors
/// A truncated record.
pub fn decode_all(mut b: &[u8]) -> Result<Vec<Record>, String> {
    let mut out = Vec::new();
    while !b.is_empty() {
        let (r, n) = Record::decode(b)?;
        out.push(r);
        b = b.get(n..).unwrap_or(&[]);
    }
    Ok(out)
}

/// Section check: a record round-trips exactly (signals bit for bit, rare ids capped at 8); the
/// fixed length; standout masks mark only the outlier; a truncated buffer is refused.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let s = Signals { exact: 0.5, ident: 0.25, structural: 1.0, affinity: 0.125, bm25: 3.7, bm25f: 9.1 };
    let r = Record {
        hash: [7; 32],
        path: 12,
        start: 31,
        end: 60,
        tokens: 410,
        kind: 1,
        standout: 0b10_0000,
        signals: s,
        rare: (0..10).map(|i| name_id(&format!("n{i}"))).collect(),
    };
    let b = r.encode();
    let (back, n) = Record::decode(&b)?;
    if n != FIXED + 4 * MAX_RARE
        || b.len() != n
        || back.signals != s
        || back.rare.len() != MAX_RARE
        || back.path != 12
        || back.hash != [7; 32]
    {
        return Err(format!("nmpod.record: round trip {back:?}"));
    }
    let all: Vec<Record> = vec![r.clone(), Record { rare: Vec::new(), ..r.clone() }];
    if decode_all(&encode_all(&all))?.len() != 2 || Record::decode(&b[..FIXED - 1]).is_ok() {
        return Err("nmpod.record: many records or truncation".into());
    }
    let mut sig = vec![Signals::default(); 50];
    if let Some(x) = sig.get_mut(7) {
        x.bm25f = 10.0;
    }
    let m = standouts(&sig);
    if m.get(7) != Some(&0b10_0000) || m.iter().enumerate().any(|(i, v)| i != 7 && *v != 0) {
        return Err(format!("nmpod.record: standouts {m:?}"));
    }
    Ok(())
}
