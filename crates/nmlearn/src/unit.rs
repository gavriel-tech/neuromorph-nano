//! Learning-weight unit: its card, its weights and its quantized files (PLAN 4c `unit`;
//! guide 10.1). A unit is labelled by id, bound to one base population by hash and to one
//! component node by id; `down [rank, k]` is drawn once from the seed and stays fixed, `up
//! [n, rank]` starts at zero (the unit is the identity until fitted) and is the only part a
//! fit moves. Files are weight-only int8 with one binary16 scale per row, read back exactly as
//! the engine dequantizes its own int8 rows.
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::collections::BTreeMap;
use std::path::Path;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmlearn.unit", FaultKind::Config, m)
}
pub(crate) fn s(t: &str) -> Value {
    Value::String(t.to_owned())
}
pub(crate) fn n(x: impl ToString) -> Value {
    Value::Number(x.to_string())
}
pub(crate) fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn gs(v: &Value, k: &str) -> Result<String, Fault> {
    v.get(k).and_then(Value::string).map(str::to_owned).map_err(|e| bad(format!("{k}: {e}")))
}
fn gu(v: &Value, k: &str) -> Result<u64, Fault> {
    v.get(k).and_then(Value::integer).map_err(|e| bad(format!("{k}: {e}")))
}
/// A JSON number as f32 (nmjson keeps numbers as text).
#[must_use]
pub fn float_of(v: &Value) -> Option<f32> {
    match v {
        Value::Number(t) => t.parse::<f32>().ok(),
        _ => None,
    }
}
fn gf(v: &Value, k: &str) -> Result<f32, Fault> {
    v.get(k).ok().and_then(float_of).ok_or_else(|| bad(format!("{k}: not a number")))
}

/// One learning-weight unit card (`units/<id>/unit.json`, also a row of the registry).
#[derive(Clone, Debug, PartialEq)]
pub struct UnitCard {
    pub id: String,
    pub base: String,
    pub base_sha256: String,
    pub slot: String,
    /// The component node the unit stacks on (`nmengn.components#L27.down`).
    pub target: String,
    pub rank: usize,
    pub scale: f32,
    pub seed: u64,
    pub lineage: Vec<String>,
    /// unfitted, candidate, fitted, rejected, rolled-back, retired.
    pub status: String,
    /// Weight files relative to `units/<id>/` with their SHA-256 (empty while unfitted).
    pub files: Vec<(String, String)>,
    pub loaded: bool,
    pub retire_after: Option<u64>,
    pub registered_at: u64,
}
impl UnitCard {
    /// A new unfitted card.
    #[must_use]
    #[expect(clippy::too_many_arguments, reason = "a card names its id, base and hash, slot, target, rank and seed")]
    pub fn new(id: &str, base: &str, base_sha256: &str, slot: &str, target: &str, rank: usize, seed: u64) -> Self {
        Self {
            id: id.into(),
            base: base.into(),
            base_sha256: base_sha256.into(),
            slot: slot.into(),
            target: target.into(),
            rank,
            scale: 1.0,
            seed,
            lineage: Vec::new(),
            status: "unfitted".into(),
            files: Vec::new(),
            loaded: false,
            retire_after: None,
            registered_at: nmtype::now_ms(),
        }
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("id", s(&self.id)),
            ("base", s(&self.base)),
            ("base_sha256", s(&self.base_sha256)),
            ("slot", s(&self.slot)),
            ("target", s(&self.target)),
            ("rank", n(self.rank)),
            ("scale", n(self.scale)),
            ("seed", n(self.seed)),
            ("lineage", Value::Array(self.lineage.iter().map(|l| s(l)).collect())),
            ("status", s(&self.status)),
            ("files", Value::Array(self.files.iter().map(|(p, h)| obj(vec![("path", s(p)), ("sha256", s(h))])).collect())),
            ("loaded", Value::Bool(self.loaded)),
            ("retire_after", self.retire_after.map_or(Value::Null, n)),
            ("registered_at", n(self.registered_at)),
        ])
    }
    /// # Errors
    /// A field missing or of the wrong type.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let list = |k: &str| {
            v.get(k)
                .and_then(Value::array)
                .map(|a| a.iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect::<Vec<_>>())
                .unwrap_or_default()
        };
        let files = v
            .get("files")
            .and_then(Value::array)
            .map(|a| {
                a.iter()
                    .filter_map(|f| Some((f.get("path").ok()?.string().ok()?.to_owned(), f.get("sha256").ok()?.string().ok()?.to_owned())))
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self {
            id: gs(v, "id")?,
            base: gs(v, "base")?,
            base_sha256: gs(v, "base_sha256")?,
            slot: gs(v, "slot")?,
            target: gs(v, "target")?,
            rank: usize::try_from(gu(v, "rank")?).map_err(|e| bad(e.to_string()))?,
            scale: gf(v, "scale").unwrap_or(1.0),
            seed: gu(v, "seed").unwrap_or(0),
            lineage: list("lineage"),
            status: gs(v, "status").unwrap_or_else(|_| "unfitted".into()),
            files,
            loaded: matches!(v.get("loaded"), Ok(Value::Bool(true))),
            retire_after: gu(v, "retire_after").ok(),
            registered_at: gu(v, "registered_at").unwrap_or(0),
        })
    }
    /// The unit as a component card for the Component Registry (kind adapter, parent and
    /// dependency the target node), hashed like every card.
    #[must_use]
    pub fn to_component_card(&self) -> Value {
        let mut m: BTreeMap<String, Value> = BTreeMap::new();
        m.insert("id".into(), s(&self.id));
        m.insert("kind".into(), s("adapter"));
        m.insert("version".into(), n(1));
        m.insert("parent".into(), s(&self.target));
        m.insert(
            "ports_in".into(),
            Value::Array(vec![obj(vec![("name", s("x")), ("kind", s("tensor")), ("required", Value::Bool(true))])]),
        );
        m.insert(
            "ports_out".into(),
            Value::Array(vec![obj(vec![("name", s("y")), ("kind", s("tensor")), ("required", Value::Bool(true))])]),
        );
        m.insert("deps".into(), Value::Array(vec![s(&self.target)]));
        m.insert("config_schema".into(), Value::Object(BTreeMap::new()));
        m.insert("config_default".into(), obj(vec![("rank", n(self.rank)), ("scale", n(self.scale)), ("seed", n(self.seed))]));
        m.insert("subcomponents".into(), Value::Array(Vec::new()));
        m.insert(
            "lineage".into(),
            Value::Array(self.lineage.iter().map(|l| obj(vec![("id", s(l)), ("version", n(1)), ("hash", s(""))])).collect()),
        );
        m.insert(
            "source".into(),
            s(&format!("learning-weight unit over {} ({}), slot {}", self.base, self.base_sha256.get(..12).unwrap_or(""), self.slot)),
        );
        m.insert("hash".into(), s(""));
        let v = Value::Object(m.clone());
        m.insert("hash".into(), s(&nmtype::card_hash(&v)));
        Value::Object(m)
    }
}

/// Seeded standard normals (xorshift64* and Box-Muller), from `nrntrn/src/mezo.rs`: the same
/// seed and step give the same draws, so `z` is never stored.
#[derive(Debug, Clone)]
pub struct Normals(u64);
impl Normals {
    #[must_use]
    pub fn new(seed: u64, step: u64) -> Self {
        Self((seed ^ step.wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1)
    }
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let top = u32::try_from(self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32).unwrap_or(u32::MAX);
        (f64::from(top) + 0.5) / 4_294_967_296.0
    }
    pub fn next_normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// f64 to f32 (buffers are f32).
#[must_use]
pub fn narrow(x: f64) -> f32 {
    #[expect(clippy::cast_possible_truncation, reason = "unit buffers are f32")]
    let y = x as f32;
    y
}

/// A unit's weights in memory: `down [rank, k]`, `up [n, rank]`.
#[derive(Clone, Debug, PartialEq)]
pub struct UnitWeights {
    pub rank: usize,
    pub k: usize,
    pub n: usize,
    pub down: Vec<f32>,
    pub up: Vec<f32>,
    pub scale: f32,
}
impl UnitWeights {
    /// The unfitted weights of `card` on a component of input width `k` and output width
    /// `n`: `down` from the card's seed scaled by `1/sqrt(k)`, `up` all zero.
    #[must_use]
    pub fn unfitted(card: &UnitCard, k: usize, n: usize) -> Self {
        let mut z = Normals::new(card.seed, 0);
        #[expect(clippy::cast_precision_loss, reason = "a width is small")]
        let sd = 1.0 / (k as f64).sqrt();
        let down = (0..card.rank * k).map(|_| narrow(z.next_normal() * sd)).collect();
        Self { rank: card.rank, k, n, down, up: vec![0.0; n * card.rank], scale: card.scale }
    }
    /// Quantize one matrix (`rows` x `cols`) to the engine's int8 row format: `"NMQ8"`,
    /// rows, cols (u32 LE), then the binary16 row scales, then the int8 values.
    /// # Errors
    /// A non-finite value.
    pub fn encode_q8(m: &[f32], rows: usize, cols: usize) -> Result<Vec<u8>, Fault> {
        if m.len() != rows * cols {
            return Err(bad(format!("matrix {} values, want {rows}x{cols}", m.len())));
        }
        let q = nmsafe::quantize(m, cols).map_err(|e| bad(e.to_string()))?;
        let mut out = Vec::with_capacity(12 + rows * 2 + m.len());
        out.extend(b"NMQ8");
        out.extend(u32::try_from(rows).unwrap_or(u32::MAX).to_le_bytes());
        out.extend(u32::try_from(cols).unwrap_or(u32::MAX).to_le_bytes());
        out.extend(q.scales.iter().flat_map(|x| x.to_le_bytes()));
        out.extend(q.values);
        Ok(out)
    }
    /// Dequantize a `encode_q8` matrix exactly as the engine does (`r16(value * scale)`).
    /// # Errors
    /// A malformed file.
    pub fn decode_q8(bytes: &[u8]) -> Result<(Vec<f32>, usize, usize), Fault> {
        let hdr = bytes.get(..12).filter(|h| h.starts_with(b"NMQ8")).ok_or_else(|| bad("not an NMQ8 file"))?;
        let u = |a: usize| usize::try_from(u32::from_le_bytes([hdr[a], hdr[a + 1], hdr[a + 2], hdr[a + 3]])).unwrap_or(0);
        let (rows, cols) = (u(4), u(8));
        let scales = bytes.get(12..12 + rows * 2).ok_or_else(|| bad("short scales"))?;
        let vals = bytes.get(12 + rows * 2..12 + rows * 2 + rows * cols).ok_or_else(|| bad("short values"))?;
        let mut out = Vec::with_capacity(rows * cols);
        for (r, row) in vals.chunks_exact(cols).enumerate() {
            let sb = scales.get(r * 2..r * 2 + 2).ok_or_else(|| bad("scale"))?;
            let scale = nmsafe::f16_to_f32(u16::from_le_bytes([sb[0], sb[1]]));
            out.extend(row.iter().map(|b| r16(f32::from(i8::from_ne_bytes([*b])) * scale)));
        }
        Ok((out, rows, cols))
    }
    /// The quantized bytes of `down` and `up` (what the files hold).
    /// # Errors
    /// As `encode_q8`.
    pub fn files(&self) -> Result<Vec<(String, Vec<u8>)>, Fault> {
        Ok(vec![
            ("down.q8".into(), Self::encode_q8(&self.down, self.rank, self.k)?),
            ("up.q8".into(), Self::encode_q8(&self.up, self.n, self.rank)?),
        ])
    }
    /// Read the unit back from its quantized files under `dir`, checking the hash of every
    /// listed file (the fit record too) and decoding the `.q8` matrices.
    /// # Errors
    /// A missing file, a hash mismatch, or widths that disagree.
    pub fn read(card: &UnitCard, dir: &Path) -> Result<Self, Fault> {
        let mut got: BTreeMap<String, (Vec<f32>, usize, usize)> = BTreeMap::new();
        for (p, h) in &card.files {
            let bytes = std::fs::read(dir.join(p)).map_err(|e| bad(format!("{p}: {e}")))?;
            if nmhash::sha256_bytes(&bytes) != *h {
                return Err(Fault::new("nmlearn.unit", FaultKind::Integrity, format!("{}: {p} does not match its hash", card.id)));
            }
            if p.ends_with(".q8") {
                got.insert(p.clone(), Self::decode_q8(&bytes)?);
            }
        }
        let (down, rank, k) = got.remove("down.q8").ok_or_else(|| bad("no down.q8"))?;
        let (up, n, r2) = got.remove("up.q8").ok_or_else(|| bad("no up.q8"))?;
        if rank != card.rank || r2 != rank {
            return Err(bad(format!("{}: rank {rank} and {r2}, card says {}", card.id, card.rank)));
        }
        Ok(Self { rank, k, n, down, up, scale: card.scale })
    }
    /// The weights as they will be after a quantized round trip (what a Step Check scores).
    /// # Errors
    /// As `encode_q8`.
    pub fn quantized(&self) -> Result<Self, Fault> {
        let (down, _, _) = Self::decode_q8(&Self::encode_q8(&self.down, self.rank, self.k)?)?;
        let (up, _, _) = Self::decode_q8(&Self::encode_q8(&self.up, self.n, self.rank)?)?;
        Ok(Self { down, up, ..self.clone() })
    }
}

/// Round to the nearest binary16 value (the engine's rounding point), as f32.
#[must_use]
pub fn r16(x: f32) -> f32 {
    nmsafe::f16_to_f32(nmsafe::f32_to_f16(x))
}

/// Self-check of `unit`: the card round-trips through JSON; an unfitted unit has a zero `up`
/// and a seeded `down` that repeats; a matrix survives the int8 round trip inside one scale
/// step per value; the component card hashes.
pub(crate) fn check() -> Result<(), String> {
    let c = UnitCard::new("lwu-t", "pop", "aa", "reranker", "nmengn.components#L0.q", 4, 7);
    let back = UnitCard::from_value(&c.to_value()).map_err(|e| e.to_string())?;
    if back != c {
        return Err("nmlearn.unit: card round trip".into());
    }
    let w = UnitWeights::unfitted(&c, 6, 3);
    if w.up.iter().any(|v| *v != 0.0) || w.down != UnitWeights::unfitted(&c, 6, 3).down || w.down.len() != 24 {
        return Err("nmlearn.unit: unfitted weights".into());
    }
    let m: Vec<f32> = (0_u32..12).map(|i| narrow(f64::from(i) / 7.0 - 0.8)).collect();
    let (d, r, k) = UnitWeights::decode_q8(&UnitWeights::encode_q8(&m, 3, 4).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let worst = d.iter().zip(&m).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    if r != 3 || k != 4 || worst > 0.02 {
        return Err(format!("nmlearn.unit: int8 round trip worst {worst}"));
    }
    if c.to_component_card().get("hash").and_then(Value::string).map_or(true, str::is_empty) {
        return Err("nmlearn.unit: component card hash".into());
    }
    Ok(())
}
