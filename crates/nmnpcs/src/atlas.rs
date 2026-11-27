//! The Parameter Map atlas (research pass, solution 5b and the cartridge section of
//! `docs/2026-11-10-neumor-research-protocol-pass-all-parts.md`): one coordinate for every
//! structural piece of a population, and one record per (coordinate, job) of what was
//! measured there: recording, lesion, set lesion, stimulation, tolerated bits, divergence from
//! sibling populations. It is a record about the weights, never a copy of them; stored as a
//! design file `designs/parmap/<population>.json`. This module holds the format only:
//! coordinates, records, encoding and decoding, with an exact round trip.
//!
//! Coordinate grammar: `<model>@<hash>` then either `/table`, `/head`, or `/L<n>` followed by
//! nothing (the layer), `/attn`, `/attn/kv<g>`, `/attn/kv<g>/h<h>`, `/ffn`, `/ffn/blk<b>`
//! (channels `b*64 .. b*64+64`) or `/ffn/ch<c>`.
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::collections::BTreeMap;

/// Schema name written into every atlas file.
pub const SCHEMA: &str = "nm-atlas/1";
/// Feed-forward channels per block (kernel-friendly unit of selection).
pub const BLOCK: u32 = 64;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmnpcs.atlas", FaultKind::Config, m)
}

/// The structural piece a coordinate names, inside a layer or outside all layers.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Piece {
    /// The token table (outside the layers).
    Table,
    /// The output head (outside the layers).
    Head,
    /// A whole layer.
    Layer(u16),
    /// The attention sublayer of a layer.
    Attn(u16),
    /// One key-value group (its key and value rows and the query heads that read it).
    Kv(u16, u16),
    /// One query head, inside its key-value group.
    QHead(u16, u16, u16),
    /// The feed-forward sublayer of a layer.
    Ffn(u16),
    /// One block of `BLOCK` feed-forward channels.
    Block(u16, u32),
    /// One feed-forward channel.
    Channel(u16, u32),
}
impl Piece {
    /// Level of the piece, coarse to fine (`table` and `head` count as layer level).
    #[must_use]
    pub fn level(&self) -> &'static str {
        match self {
            Self::Table | Self::Head | Self::Layer(_) => "layer",
            Self::Attn(_) | Self::Ffn(_) => "sublayer",
            Self::Kv(..) => "kv-group",
            Self::QHead(..) => "head",
            Self::Block(..) => "block",
            Self::Channel(..) => "channel",
        }
    }
    fn path(&self) -> String {
        match self {
            Self::Table => "table".into(),
            Self::Head => "head".into(),
            Self::Layer(l) => format!("L{l}"),
            Self::Attn(l) => format!("L{l}/attn"),
            Self::Kv(l, g) => format!("L{l}/attn/kv{g}"),
            Self::QHead(l, g, h) => format!("L{l}/attn/kv{g}/h{h}"),
            Self::Ffn(l) => format!("L{l}/ffn"),
            Self::Block(l, b) => format!("L{l}/ffn/blk{b}"),
            Self::Channel(l, c) => format!("L{l}/ffn/ch{c}"),
        }
    }
}

fn num<T: std::str::FromStr>(s: &str, prefix: &str) -> Option<T> {
    s.strip_prefix(prefix).filter(|r| !r.is_empty() && r.bytes().all(|b| b.is_ascii_digit())).and_then(|r| r.parse().ok())
}

/// A coordinate: population, its base hash prefix and the piece.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Coordinate {
    pub model: String,
    pub hash: String,
    pub piece: Piece,
}
impl Coordinate {
    /// Parse `<model>@<hash>/<piece path>`.
    /// # Errors
    /// Anything outside the grammar in the module notes.
    pub fn parse(s: &str) -> Result<Self, Fault> {
        let err = || bad(format!("not an atlas coordinate: {s}"));
        let (model, rest) = s.split_once('@').ok_or_else(err)?;
        let (hash, path) = rest.split_once('/').ok_or_else(err)?;
        if model.is_empty() || hash.is_empty() || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(err());
        }
        let parts: Vec<&str> = path.split('/').collect();
        let piece = match parts.as_slice() {
            ["table"] => Piece::Table,
            ["head"] => Piece::Head,
            [l, more @ ..] => {
                let l: u16 = num(l, "L").ok_or_else(err)?;
                match more {
                    [] => Piece::Layer(l),
                    ["attn"] => Piece::Attn(l),
                    ["attn", g] => Piece::Kv(l, num(g, "kv").ok_or_else(err)?),
                    ["attn", g, h] => Piece::QHead(l, num(g, "kv").ok_or_else(err)?, num(h, "h").ok_or_else(err)?),
                    ["ffn"] => Piece::Ffn(l),
                    ["ffn", b] if b.starts_with("blk") => Piece::Block(l, num(b, "blk").ok_or_else(err)?),
                    ["ffn", c] => Piece::Channel(l, num(c, "ch").ok_or_else(err)?),
                    _ => return Err(err()),
                }
            }
            [] => return Err(err()),
        };
        Ok(Self { model: model.to_owned(), hash: hash.to_owned(), piece })
    }
}
impl std::fmt::Display for Coordinate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}/{}", self.model, self.hash, self.piece.path())
    }
}

/// What a lesion (switching the piece off) did to the job's own measure.
#[derive(Clone, Debug, PartialEq)]
pub struct Lesion {
    /// Mean change of the job's measure over the calibration set.
    pub effect: f32,
    /// Spread (standard deviation) of that change.
    pub spread: f32,
    /// Calibration items measured.
    pub count: u32,
}

/// One measurement record: a piece, for one job.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub at: Coordinate,
    /// The job the measurement was made for (drone, tentacle, judge, embed, write).
    pub job: String,
    /// Recording: mean activity and selectivity for the job.
    pub recording: Option<(f32, f32)>,
    pub lesion: Option<Lesion>,
    /// Effect of switching off the set this piece was ranked into (against compensation).
    pub set_lesion: Option<f32>,
    /// Effect of amplifying the piece.
    pub stimulation: Option<f32>,
    /// Input features that drive the piece on the calibration set.
    pub receptive: Vec<String>,
    /// Fewest bits per weight the job tolerated for this piece.
    pub bits: Option<u8>,
    /// Weight difference from each sibling population (name to relative difference).
    pub divergence: BTreeMap<String, f32>,
    /// Run record the numbers came from.
    pub run: String,
    /// UTC date measured, `YYYY-MM-DD`.
    pub date: String,
}
impl Record {
    /// An empty record for a piece and job.
    #[must_use]
    pub fn new(at: Coordinate, job: &str, run: &str, date: &str) -> Self {
        Self {
            at,
            job: job.to_owned(),
            recording: None,
            lesion: None,
            set_lesion: None,
            stimulation: None,
            receptive: Vec::new(),
            bits: None,
            divergence: BTreeMap::new(),
            run: run.to_owned(),
            date: date.to_owned(),
        }
    }
}

/// The atlas of one population.
#[derive(Clone, Debug, PartialEq)]
pub struct Atlas {
    pub population: String,
    /// SHA-256 of the released weights the atlas was measured on.
    pub base_sha256: String,
    /// SHA-256 of the fixed calibration set.
    pub calibration_sha256: String,
    /// Atlas version: raised on every re-measurement; cartridges name the version they used.
    pub version: u32,
    pub records: Vec<Record>,
}

fn s(t: &str) -> Value {
    Value::String(t.to_owned())
}
fn f(v: f32) -> Value {
    // Rust prints the shortest spelling that parses back to the same f32.
    Value::Number(format!("{v}"))
}
fn opt(v: Option<f32>) -> Value {
    v.map_or(Value::Null, f)
}
fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn rf(v: &Value, k: &str) -> Result<f32, Fault> {
    match v.get(k).map_err(|e| bad(e.to_string()))? {
        Value::Number(n) => n.parse().map_err(|e| bad(format!("{k}: {e}"))),
        _ => Err(bad(format!("{k}: expected a number"))),
    }
}
fn ropt(v: &Value, k: &str) -> Result<Option<f32>, Fault> {
    match v.get(k).map_err(|e| bad(e.to_string()))? {
        Value::Null => Ok(None),
        _ => rf(v, k).map(Some),
    }
}
fn rs(v: &Value, k: &str) -> Result<String, Fault> {
    v.get(k).and_then(Value::string).map(str::to_owned).map_err(|e| bad(format!("{k}: {e}")))
}
fn ri(v: &Value, k: &str) -> Result<u64, Fault> {
    v.get(k).and_then(Value::integer).map_err(|e| bad(format!("{k}: {e}")))
}

impl Record {
    fn to_value(&self) -> Value {
        obj(vec![
            ("at", s(&self.at.to_string())),
            ("level", s(self.at.piece.level())),
            ("job", s(&self.job)),
            ("recording", self.recording.map_or(Value::Null, |(m, sel)| obj(vec![("mean", f(m)), ("selectivity", f(sel))]))),
            (
                "lesion",
                self.lesion.as_ref().map_or(Value::Null, |l| {
                    obj(vec![("effect", f(l.effect)), ("spread", f(l.spread)), ("count", Value::Number(l.count.to_string()))])
                }),
            ),
            ("set_lesion", opt(self.set_lesion)),
            ("stimulation", opt(self.stimulation)),
            ("receptive", Value::Array(self.receptive.iter().map(|r| s(r)).collect())),
            ("bits", self.bits.map_or(Value::Null, |b| Value::Number(b.to_string()))),
            ("divergence", Value::Object(self.divergence.iter().map(|(k, v)| (k.clone(), f(*v))).collect())),
            ("run", s(&self.run)),
            ("date", s(&self.date)),
        ])
    }
    fn from_value(v: &Value) -> Result<Self, Fault> {
        let at = Coordinate::parse(&rs(v, "at")?)?;
        if rs(v, "level")? != at.piece.level() {
            return Err(bad(format!("{at}: level does not match the coordinate")));
        }
        let recording = match v.get("recording").map_err(|e| bad(e.to_string()))? {
            Value::Null => None,
            r => Some((rf(r, "mean")?, rf(r, "selectivity")?)),
        };
        let lesion = match v.get("lesion").map_err(|e| bad(e.to_string()))? {
            Value::Null => None,
            l => Some(Lesion {
                effect: rf(l, "effect")?,
                spread: rf(l, "spread")?,
                count: u32::try_from(ri(l, "count")?).map_err(|e| bad(e.to_string()))?,
            }),
        };
        let bits = match v.get("bits").map_err(|e| bad(e.to_string()))? {
            Value::Null => None,
            _ => Some(u8::try_from(ri(v, "bits")?).map_err(|e| bad(e.to_string()))?),
        };
        let receptive = v
            .get("receptive")
            .and_then(Value::array)
            .map_err(|e| bad(e.to_string()))?
            .iter()
            .map(|r| r.string().map(str::to_owned).map_err(|e| bad(e.to_string())))
            .collect::<Result<_, _>>()?;
        let divergence = v
            .get("divergence")
            .and_then(Value::object)
            .map_err(|e| bad(e.to_string()))?
            .iter()
            .map(|(k, x)| match x {
                Value::Number(n) => n.parse().map(|p| (k.clone(), p)).map_err(|e| bad(format!("divergence {k}: {e}"))),
                _ => Err(bad(format!("divergence {k}: expected a number"))),
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            at,
            job: rs(v, "job")?,
            recording,
            lesion,
            set_lesion: ropt(v, "set_lesion")?,
            stimulation: ropt(v, "stimulation")?,
            receptive,
            bits,
            divergence,
            run: rs(v, "run")?,
            date: rs(v, "date")?,
        })
    }
}

impl Atlas {
    /// The atlas as a JSON value (records in stored order).
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("schema", s(SCHEMA)),
            ("population", s(&self.population)),
            ("base_sha256", s(&self.base_sha256)),
            ("calibration_sha256", s(&self.calibration_sha256)),
            ("version", Value::Number(self.version.to_string())),
            ("block", Value::Number(BLOCK.to_string())),
            ("records", Value::Array(self.records.iter().map(Record::to_value).collect())),
        ])
    }
    /// Read an atlas value.
    /// # Errors
    /// Wrong schema or block size, a malformed field or coordinate.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        if rs(v, "schema")? != SCHEMA {
            return Err(bad(format!("schema is not {SCHEMA}")));
        }
        if ri(v, "block")? != u64::from(BLOCK) {
            return Err(bad(format!("block size is not {BLOCK}")));
        }
        let records = v
            .get("records")
            .and_then(Value::array)
            .map_err(|e| bad(e.to_string()))?
            .iter()
            .map(Record::from_value)
            .collect::<Result<_, _>>()?;
        Ok(Self {
            population: rs(v, "population")?,
            base_sha256: rs(v, "base_sha256")?,
            calibration_sha256: rs(v, "calibration_sha256")?,
            version: u32::try_from(ri(v, "version")?).map_err(|e| bad(e.to_string()))?,
            records,
        })
    }
    /// Readable text for the design file.
    #[must_use]
    pub fn to_text(&self) -> String {
        nmjson::pretty::pretty(&self.to_value())
    }
    /// Parse design-file text.
    /// # Errors
    /// Invalid JSON or any `from_value` fault.
    pub fn from_text(text: &str) -> Result<Self, Fault> {
        Self::from_value(&nmjson::parse(text.as_bytes()).map_err(|e| bad(e.to_string()))?)
    }
}

/// Self-check of the atlas format: every piece kind and every field round-trips exactly
/// through text, and malformed coordinates, a wrong level and a wrong schema are refused.
pub(crate) fn check() -> Result<(), String> {
    let h = "a1b2c3d4e5f6";
    let pieces = [
        Piece::Table,
        Piece::Head,
        Piece::Layer(0),
        Piece::Attn(3),
        Piece::Kv(14, 5),
        Piece::QHead(14, 5, 10),
        Piece::Ffn(27),
        Piece::Block(17, 23),
        Piece::Channel(17, 1500),
    ];
    let mut records = Vec::new();
    for (i, p) in pieces.iter().enumerate() {
        let at = Coordinate { model: "qwen3-0.6b".into(), hash: h.into(), piece: p.clone() };
        let text = at.to_string();
        if Coordinate::parse(&text).map_err(|e| e.to_string())? != at {
            return Err(format!("nmnpcs.atlas: coordinate {text} does not round-trip"));
        }
        let mut r = Record::new(at, "judge", "runs/check", "2026-11-11");
        if i % 2 == 0 {
            r.recording = Some((0.1 + 0.37 * i as f32, 1.0 / 3.0));
            r.lesion = Some(Lesion { effect: -0.003_14, spread: 1e-7, count: 72 });
            r.set_lesion = Some(-0.25);
            r.stimulation = Some(0.012_5);
            r.receptive = vec!["identifier".into(), "call site".into()];
            r.bits = Some(4);
            r.divergence.insert("qwen3-reranker-0.6b".into(), 0.071_828);
        }
        records.push(r);
    }
    let atlas =
        Atlas { population: "qwen3-0.6b".into(), base_sha256: "00".repeat(32), calibration_sha256: "11".repeat(32), version: 1, records };
    let back = Atlas::from_text(&atlas.to_text()).map_err(|e| e.to_string())?;
    if back != atlas {
        return Err("nmnpcs.atlas: the atlas does not round-trip through text".into());
    }
    for wrong in ["qwen3@zz/L1", "qwen3@ab/L1/ffn/blk", "qwen3@ab/L1/attn/kv1/q2", "qwen3@ab/Lx", "@ab/L1", "qwen3@ab/L1/ffn/blk2/x"] {
        if Coordinate::parse(wrong).is_ok() {
            return Err(format!("nmnpcs.atlas: malformed coordinate {wrong} was accepted"));
        }
    }
    let swapped = atlas.to_text().replacen("\"level\": \"layer\"", "\"level\": \"block\"", 1);
    if Atlas::from_text(&swapped).is_ok() {
        return Err("nmnpcs.atlas: a wrong level was accepted".into());
    }
    if Atlas::from_text(&atlas.to_text().replace(SCHEMA, "nm-atlas/0")).is_ok() {
        return Err("nmnpcs.atlas: a wrong schema was accepted".into());
    }
    Ok(())
}
