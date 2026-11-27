//! Muscle-memory pack format, version 1.
//!
//! File layout (all integers little-endian):
//! 1. magic, 8 bytes: `NMMUSC` 0x00 0x01;
//! 2. header length `u32`, then the header as canonical JSON (self-describing: kind,
//!    data class, description, content fingerprint, base-weights hash, quantization, format
//!    and pack versions, promotion gain record, running gain, withdrawn flag, UTC date, the
//!    pack's alphabet with each symbol's kind and descriptor);
//! 3. payload section: width `u8` (1, 2 or 4 bytes), count `u32`, then `count` indices into
//!    the alphabet, each `width` bytes: the symbol sequence quantized to the smallest integer
//!    width that holds the pack's alphabet;
//! 4. vector section (reserved for encoded behavioural vectors): length `u32`, then that
//!    many bytes; empty in the first form.
//!
//! Fingerprint: SHA-256 of the payload section followed by the alphabet lines
//! (`kind TAB descriptor LF`). It names the content only, so updating the running gain or the
//! withdrawn flag rewrites the header without changing the fingerprint.
use crate::gain::GainRecord;
use crate::symbol::{Codebook, MessageKind};
use crate::withdraw::{Recent, RunningGain};
use crate::{Fault, Result};
use nmjson::Value;
use std::collections::BTreeMap;

pub const MAGIC: [u8; 8] = *b"NMMUSC\x00\x01";
pub const FORMAT_VERSION: u64 = 1;
pub const KIND: &str = "muscle-memory";
pub const CLASS: &str = "behavioural";

/// One symbol of a pack's alphabet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlphabetEntry {
    /// Code in the log the pack was mined from (for tracing back only).
    pub code: u32,
    pub kind: MessageKind,
    pub descriptor: String,
}

/// Pack header (the JSON part).
#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub description: String,
    pub fingerprint: String,
    pub base_hash: String,
    /// Payload width in bytes (1, 2 or 4).
    pub width: u8,
    pub pack_version: u64,
    pub date: String,
    pub promotion: GainRecord,
    pub running: RunningGain,
    /// Helpful and harmful counts and the recent estimator (2026-11-13; empty in older packs).
    pub recent: Recent,
    pub withdrawn: bool,
    pub alphabet: Vec<AlphabetEntry>,
}

/// A pack in RAM.
#[derive(Clone, Debug, PartialEq)]
pub struct Pack {
    pub header: Header,
    /// Sequence as alphabet indices.
    pub indices: Vec<u32>,
    /// Reserved encoded behavioural vectors (empty in the first form).
    pub vectors: Vec<u8>,
}

/// Smallest width in bytes that holds `alphabet` distinct indices.
#[must_use]
pub const fn width_for(alphabet: usize) -> u8 {
    if alphabet <= 1 << 8 {
        1
    } else if alphabet <= 1 << 16 {
        2
    } else {
        4
    }
}

fn payload_bytes(width: u8, indices: &[u32]) -> Result<Vec<u8>> {
    let count = u32::try_from(indices.len()).map_err(|_| Fault::new("nmmusc-too-long", "sequence over u32"))?;
    let mut out = Vec::with_capacity(5 + indices.len() * usize::from(width));
    out.push(width);
    out.extend_from_slice(&count.to_le_bytes());
    for &i in indices {
        let b = i.to_le_bytes();
        out.extend_from_slice(b.get(..usize::from(width)).unwrap_or(&b));
    }
    Ok(out)
}

fn fingerprint(payload: &[u8], alphabet: &[AlphabetEntry]) -> String {
    let mut buf = payload.to_vec();
    buf.push(b'\n');
    for a in alphabet {
        buf.extend_from_slice(a.kind.as_str().as_bytes());
        buf.push(b'\t');
        buf.extend_from_slice(a.descriptor.as_bytes());
        buf.push(b'\n');
    }
    nmhash::sha256_bytes(&buf)
}

/// UTC date and time `YYYY-MM-DDTHH:MM:SSZ` from milliseconds since the epoch (civil date
/// from day count, the standard proleptic Gregorian conversion).
#[must_use]
pub fn utc_stamp(ms: u64) -> String {
    let secs = ms / 1000;
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

impl Pack {
    /// Build a pack from a mined chunk. The description is written from the symbols when
    /// `description` is empty.
    /// # Errors
    /// A code outside the codebook, an empty chunk or a non-finite gain.
    pub fn build(chunk: &[u32], codebook: &Codebook, description: &str, base_hash: &str, promotion: GainRecord) -> Result<Self> {
        if chunk.is_empty() {
            return Err(Fault::new("nmmusc-empty", "empty chunk"));
        }
        for x in [promotion.diff, promotion.se, promotion.with_mean, promotion.without_mean] {
            if !x.is_finite() {
                return Err(Fault::new("nmmusc-bad-gain", "gain record not finite"));
            }
        }
        let mut alphabet: Vec<AlphabetEntry> = Vec::new();
        let mut indices = Vec::with_capacity(chunk.len());
        for &c in chunk {
            let e = codebook.entry(c).ok_or_else(|| Fault::new("nmmusc-unknown-code", format!("code {c} not in the codebook")))?;
            let i = if let Some(p) = alphabet.iter().position(|a| a.code == c) {
                p
            } else {
                alphabet.push(AlphabetEntry { code: c, kind: e.kind, descriptor: e.descriptor.clone() });
                alphabet.len() - 1
            };
            indices.push(u32::try_from(i).map_err(|_| Fault::new("nmmusc-alphabet-full", "alphabet over u32"))?);
        }
        let width = width_for(alphabet.len());
        let fp = fingerprint(&payload_bytes(width, &indices)?, &alphabet);
        let description = if description.is_empty() {
            let steps: Vec<String> = indices.iter().filter_map(|&i| alphabet.get(i as usize).map(|a| a.descriptor.clone())).collect();
            format!("{} behavioural messages in order: {}", steps.len(), steps.join("; "))
        } else {
            description.to_owned()
        };
        Ok(Self {
            header: Header {
                description,
                fingerprint: fp,
                base_hash: base_hash.to_owned(),
                width,
                pack_version: 1,
                date: utc_stamp(nmtype::hop::now_ms()),
                promotion,
                running: RunningGain::from_promotion(&promotion),
                recent: Recent::default(),
                withdrawn: false,
                alphabet,
            },
            indices,
            vectors: Vec::new(),
        })
    }

    /// The sequence as behavioural messages: (kind, descriptor) in order.
    #[must_use]
    pub fn steps(&self) -> Vec<(MessageKind, String)> {
        self.indices.iter().filter_map(|&i| self.header.alphabet.get(i as usize).map(|a| (a.kind, a.descriptor.clone()))).collect()
    }

    fn header_value(&self) -> Value {
        let h = &self.header;
        let g = &h.promotion;
        obj(vec![
            ("kind", s(KIND)),
            ("data_class", s(CLASS)),
            ("format_version", Value::Number(FORMAT_VERSION.to_string())),
            ("pack_version", Value::Number(h.pack_version.to_string())),
            ("description", s(&h.description)),
            ("fingerprint", s(&h.fingerprint)),
            ("base_hash", s(&h.base_hash)),
            ("date", s(&h.date)),
            ("withdrawn", Value::Bool(h.withdrawn)),
            (
                "quantization",
                obj(vec![
                    ("payload", s("alphabet index, smallest unsigned width holding the alphabet")),
                    ("width_bytes", Value::Number(h.width.to_string())),
                ]),
            ),
            ("length", Value::Number(self.indices.len().to_string())),
            ("vectors", obj(vec![("bytes", Value::Number(self.vectors.len().to_string())), ("status", s("reserved"))])),
            (
                "promotion",
                obj(vec![
                    ("with_n", Value::Number(g.with_n.to_string())),
                    ("with_mean", f(g.with_mean)),
                    ("without_n", Value::Number(g.without_n.to_string())),
                    ("without_mean", f(g.without_mean)),
                    ("diff", f(g.diff)),
                    ("se", f(g.se)),
                ]),
            ),
            ("running", obj(vec![("n", Value::Number(h.running.n.to_string())), ("mean", f(h.running.mean)), ("m2", f(h.running.m2))])),
            (
                "recent",
                obj(vec![
                    ("helpful", Value::Number(h.recent.helpful.to_string())),
                    ("harmful", Value::Number(h.recent.harmful.to_string())),
                    ("n", Value::Number(h.recent.n.to_string())),
                    ("mean", f(h.recent.mean)),
                    ("var", f(h.recent.var)),
                    ("cusum", f(h.recent.cusum)),
                ]),
            ),
            (
                "alphabet",
                Value::Array(
                    h.alphabet
                        .iter()
                        .map(|a| {
                            obj(vec![
                                ("code", Value::Number(a.code.to_string())),
                                ("kind", s(a.kind.as_str())),
                                ("descriptor", s(&a.descriptor)),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }

    /// Encode to file bytes.
    /// # Errors
    /// JSON encoding failure or sizes over `u32`.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let header = self.header_value().encode().map_err(|e| Fault::new("nmmusc-encode", e.to_string()))?;
        let hl = u32::try_from(header.len()).map_err(|_| Fault::new("nmmusc-too-long", "header over u32"))?;
        let vl = u32::try_from(self.vectors.len()).map_err(|_| Fault::new("nmmusc-too-long", "vectors over u32"))?;
        let mut out = MAGIC.to_vec();
        out.extend_from_slice(&hl.to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&payload_bytes(self.header.width, &self.indices)?);
        out.extend_from_slice(&vl.to_le_bytes());
        out.extend_from_slice(&self.vectors);
        Ok(out)
    }

    /// Decode and verify file bytes: magic, kind, data class, format version, and the
    /// fingerprint recomputed from the payload and alphabet.
    /// # Errors
    /// `nmmusc-bad-pack` naming the first failed field; `nmmusc-sensory-refused` when the
    /// header names any data class other than behavioural.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let bad = |d: &str| Fault::new("nmmusc-bad-pack", d.to_owned());
        let mut r = Cursor { b: bytes, at: 0 };
        if r.take(8)? != MAGIC {
            return Err(bad("magic"));
        }
        let hl = r.u32()? as usize;
        let hv = nmjson::parse(r.take(hl)?).map_err(|e| bad(&format!("header JSON: {e}")))?;
        let width = r.take(1)?.first().copied().unwrap_or(0);
        if !matches!(width, 1 | 2 | 4) {
            return Err(bad("payload width"));
        }
        let count = r.u32()? as usize;
        let raw = r.take(count * usize::from(width))?;
        let indices: Vec<u32> = raw
            .chunks_exact(usize::from(width))
            .map(|c| {
                let mut b = [0u8; 4];
                if let Some(d) = b.get_mut(..c.len()) {
                    d.copy_from_slice(c);
                }
                u32::from_le_bytes(b)
            })
            .collect();
        let vl = r.u32()? as usize;
        let vectors = r.take(vl)?.to_vec();
        if r.at != bytes.len() {
            return Err(bad("trailing bytes"));
        }
        let txt = |k: &str| hv.get(k).and_then(Value::string).map(str::to_owned).map_err(|_| bad(k));
        let int = |v: &Value, k: &str| v.get(k).and_then(Value::integer).map_err(|_| bad(k));
        if txt("kind")? != KIND {
            return Err(bad("kind is not muscle-memory"));
        }
        if txt("data_class")? != CLASS {
            return Err(Fault::new("nmmusc-sensory-refused", "pack header names a non-behavioural data class"));
        }
        if int(&hv, "format_version")? != FORMAT_VERSION {
            return Err(bad("format version"));
        }
        let q = hv.get("quantization").map_err(|_| bad("quantization"))?;
        if int(q, "width_bytes")? != u64::from(width) {
            return Err(bad("quantization width differs from payload"));
        }
        let mut alphabet = Vec::new();
        for a in hv.get("alphabet").and_then(Value::array).map_err(|_| bad("alphabet"))? {
            let kind = MessageKind::parse(a.get("kind").and_then(Value::string).map_err(|_| bad("alphabet kind"))?)?;
            let descriptor = a.get("descriptor").and_then(Value::string).map_err(|_| bad("descriptor"))?.to_owned();
            let code = u32::try_from(int(a, "code")?).map_err(|_| bad("code"))?;
            alphabet.push(AlphabetEntry { code, kind, descriptor });
        }
        if indices.iter().any(|&i| i as usize >= alphabet.len()) {
            return Err(bad("index outside alphabet"));
        }
        let fp = txt("fingerprint")?;
        let payload = payload_bytes(width, &indices)?;
        if fingerprint(&payload, &alphabet) != fp {
            return Err(bad("fingerprint does not match content"));
        }
        let p = hv.get("promotion").map_err(|_| bad("promotion"))?;
        let rn = hv.get("running").map_err(|_| bad("running"))?;
        let header = Header {
            description: txt("description")?,
            fingerprint: fp,
            base_hash: txt("base_hash")?,
            width,
            pack_version: int(&hv, "pack_version")?,
            date: txt("date")?,
            promotion: GainRecord {
                with_n: int(p, "with_n")?,
                with_mean: num(p, "with_mean")?,
                without_n: int(p, "without_n")?,
                without_mean: num(p, "without_mean")?,
                diff: num(p, "diff")?,
                se: num(p, "se")?,
            },
            running: RunningGain { n: int(rn, "n")?, mean: num(rn, "mean")?, m2: num(rn, "m2")? },
            recent: hv.get("recent").ok().map_or_else(Recent::default, |r| Recent {
                helpful: int(r, "helpful").unwrap_or(0),
                harmful: int(r, "harmful").unwrap_or(0),
                n: int(r, "n").unwrap_or(0),
                mean: num(r, "mean").unwrap_or(0.0),
                var: num(r, "var").unwrap_or(0.0),
                cusum: num(r, "cusum").unwrap_or(0.0),
            }),
            withdrawn: matches!(hv.get("withdrawn"), Ok(Value::Bool(true))),
            alphabet,
        };
        Ok(Self { header, indices, vectors })
    }
}

struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}
impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let s = self.b.get(self.at..self.at.saturating_add(n)).ok_or_else(|| Fault::new("nmmusc-bad-pack", "truncated"))?;
        self.at += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        let s = self.take(4)?;
        let mut b = [0u8; 4];
        b.copy_from_slice(s);
        Ok(u32::from_le_bytes(b))
    }
}

fn obj(fields: Vec<(&str, Value)>) -> Value {
    Value::Object(fields.into_iter().map(|(k, v)| (k.to_owned(), v)).collect::<BTreeMap<_, _>>())
}
fn s(t: &str) -> Value {
    Value::String(t.to_owned())
}
fn f(x: f64) -> Value {
    Value::Number(if x.is_finite() { format!("{x}") } else { "0".to_owned() })
}
/// Read a JSON number field as `f64`.
/// # Errors
/// Missing field or not a number.
pub fn num(v: &Value, k: &str) -> Result<f64> {
    match v.get(k) {
        Ok(Value::Number(t)) => t.parse().map_err(|_| Fault::new("nmmusc-bad-pack", format!("{k} not a number"))),
        _ => Err(Fault::new("nmmusc-bad-pack", format!("{k} missing"))),
    }
}

/// A small pack for checks: four behavioural messages, one repeated.
pub(crate) fn sample(base: &str) -> Result<Pack> {
    use crate::symbol::Message;
    let mut cb = Codebook::default();
    let msgs = [
        Message::behavioural(MessageKind::Instruction, "g0_t2", "focus:tests"),
        Message::behavioural(MessageKind::Route, "g1", "forward:g2"),
        Message::behavioural(MessageKind::Reflex, "tentacle:2", "re-read"),
        Message::behavioural(MessageKind::Instruction, "g0_t2", "focus:tests"),
    ];
    let mut chunk = Vec::new();
    for m in &msgs {
        chunk.push(cb.code(m)?);
    }
    let g = GainRecord { with_n: 12, with_mean: 0.7, without_n: 40, without_mean: 0.55, diff: 0.15, se: 0.04 };
    Pack::build(&chunk, &cb, "", base, g)
}

/// Section check: encode then decode round-trips exactly; a flipped payload byte fails the
/// fingerprint.
/// # Errors
/// One fault line.
pub fn check() -> std::result::Result<(), String> {
    let p = sample("base-hash-a").map_err(|e| e.to_string())?;
    let bytes = p.encode().map_err(|e| e.to_string())?;
    let q = Pack::decode(&bytes).map_err(|e| e.to_string())?;
    if q != p {
        return Err("decoded pack differs".to_owned());
    }
    if q.encode().map_err(|e| e.to_string())? != bytes {
        return Err("re-encoded bytes differ".to_owned());
    }
    if p.header.width != 1 || p.indices != [0, 1, 2, 0] {
        return Err(format!("quantization wrong: width {} indices {:?}", p.header.width, p.indices));
    }
    let mut bad = bytes.clone();
    let at = bad.len() - 6; // inside the payload indices
    if let Some(b) = bad.get_mut(at) {
        *b ^= 1;
    }
    match Pack::decode(&bad) {
        Err(f) if f.code == "nmmusc-bad-pack" => {}
        other => return Err(format!("corrupt payload accepted: {other:?}")),
    }
    if utc_stamp(0) != "1970-01-01T00:00:00Z" || utc_stamp(1_794_281_978_000) != "2026-11-10T03:39:38Z" {
        return Err(format!("utc stamp wrong: {}", utc_stamp(1_794_281_978_000)));
    }
    Ok(())
}
