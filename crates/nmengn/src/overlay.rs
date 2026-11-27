//! Parameter overlays: edits, masks and freezes of one object's parameters, held beside the object
//! and never written into its files.
//!
//! An object is anything that holds parameters, by the key the parameter editor uses: a MicroModel's
//! upstream open weights (`open:<model>`), a library cartridge (`id@version`: a base cartridge, a
//! socket cartridge, a preference cartridge), a job filter (`filter:<role>/<job>`), an adapter version
//! or a learning unit. Its overlay is one file, `memory/overlays/<slug>.json` (`FORMAT`), an ordered
//! list of entries:
//! - `edit`: one operation (set, add, multiply, scale toward the identity, clamp, zero, identity,
//!   negate, or paste of captured values) on resolved regions of named tensors. Values that come from
//!   outside the object (a paste, a copy, an interpolation) are captured in the entry, so a replay
//!   gives the same values whatever happens to their source later; the other operations are pure
//!   functions of the released values, which never change.
//! - `mask`: regions whose contribution is switched off: each value set to the value that changes
//!   nothing (its identity: 1 for a gain or a norm weight, 0 otherwise). `whole` masks the whole
//!   object: a loader leaves its effect out while it stays plugged.
//! - `freeze`: regions made read-only; an edit, mask or unmask touching one is refused by the editor.
//!
//! Replay: edits apply in entry order, each result rounded to the stored element type as the engine
//! holds it; masks then set their regions to the identity whatever the edits did, so a masked region
//! contributes nothing and unmasking returns the edited values exactly. Freezes change no value.
//!
//! A region is a flat element range `[a, b)` of a tensor, or a block of rows and columns of a
//! two-dimensional tensor (`width` columns), in row-major order. The released files stay as they
//! are: the engine reads a touched tensor's bytes from its file, applies the overlay to a copy in RAM
//! and offers that copy to the provisioner under its own content hash (`patch`).
use crate::filter::Part;
use nmjson::Value;
use nmnpcs::TensorDesc;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// The overlay file format.
pub const FORMAT: &str = "nm-overlay/1";
/// Where overlays are kept, under the Nano's folder.
pub const DIR: &str = "memory/overlays";

/// Where a region lies inside its tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Span {
    /// Flat elements `[a, b)`.
    Flat(usize, usize),
    /// Rows `[r0, r1)` and columns `[c0, c1)` of a matrix.
    Block { rows: (usize, usize), cols: (usize, usize) },
}

/// One region of one tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    pub tensor: String,
    pub span: Span,
    /// Columns of the tensor (its size for a vector), so flat ranges and blocks compare.
    pub width: usize,
    /// The value at which an element changes nothing.
    pub identity: f32,
    /// The stored element type the engine holds the values in (`BF16`, `F16`, `F32`).
    pub dtype: String,
    /// Captured values, in region order, for a paste.
    pub values: Option<Vec<f32>>,
}
impl Region {
    /// Elements in the region.
    #[must_use]
    pub fn len(&self) -> usize {
        match self.span {
            Span::Flat(a, b) => b.saturating_sub(a),
            Span::Block { rows, cols } => rows.1.saturating_sub(rows.0) * cols.1.saturating_sub(cols.0),
        }
    }
    /// Whether the region holds no element.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The flat indices, in region order, given to `f` with their place in the region.
    pub fn each(&self, mut f: impl FnMut(usize, usize)) {
        match self.span {
            Span::Flat(a, b) => (a..b).enumerate().for_each(|(k, i)| f(k, i)),
            Span::Block { rows, cols } => {
                let mut k = 0;
                for r in rows.0..rows.1 {
                    for c in cols.0..cols.1 {
                        f(k, r * self.width + c);
                        k += 1;
                    }
                }
            }
        }
    }
    /// The place of flat index `i` in the region, if it lies there.
    #[must_use]
    pub fn place(&self, i: usize) -> Option<usize> {
        match self.span {
            Span::Flat(a, b) => (a..b).contains(&i).then(|| i - a),
            Span::Block { rows, cols } => {
                let w = self.width.max(1);
                let (r, c) = (i / w, i % w);
                ((rows.0..rows.1).contains(&r) && (cols.0..cols.1).contains(&c)).then(|| (r - rows.0) * (cols.1 - cols.0) + (c - cols.0))
            }
        }
    }
    /// The region as rectangles of (rows, columns) over its tensor's width.
    fn rects(&self) -> Vec<((usize, usize), (usize, usize))> {
        let w = self.width.max(1);
        match self.span {
            Span::Block { rows, cols } => vec![(rows, cols)],
            Span::Flat(a, b) if a >= b => Vec::new(),
            Span::Flat(a, b) => {
                let (ra, rb) = (a / w, (b - 1) / w);
                if ra == rb {
                    return vec![((ra, ra + 1), (a % w, (b - 1) % w + 1))];
                }
                let mut v = vec![((ra, ra + 1), (a % w, w))];
                if rb > ra + 1 {
                    v.push(((ra + 1, rb), (0, w)));
                }
                v.push(((rb, rb + 1), (0, (b - 1) % w + 1)));
                v
            }
        }
    }
    /// Whether two regions share an element.
    #[must_use]
    pub fn meets(&self, o: &Self) -> bool {
        if self.tensor != o.tensor {
            return false;
        }
        let x = |a: (usize, usize), b: (usize, usize)| a.0 < b.1 && b.0 < a.1;
        self.rects().iter().any(|(r, c)| o.rects().iter().any(|(r2, c2)| x(*r, *r2) && x(*c, *c2)))
    }
    /// The region as a path (`name[a:b]`, `name[r0:r1, c0:c1]`).
    #[must_use]
    pub fn path(&self) -> String {
        match self.span {
            Span::Flat(a, b) => format!("{}[{a}:{b}]", self.tensor),
            Span::Block { rows, cols } => format!("{}[{}:{}, {}:{}]", self.tensor, rows.0, rows.1, cols.0, cols.1),
        }
    }
}

/// An edit's operation, replayed on the released values.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Set(f32),
    Add(f32),
    Mul(f32),
    /// Toward the identity: `identity + s * (x - identity)`.
    Scale(f32),
    Clamp(f32, f32),
    Zero,
    Identity,
    Negate,
    /// The region's captured values.
    Paste,
}
impl Op {
    /// The operation's name and arguments.
    #[must_use]
    pub fn phrase(&self) -> String {
        match self {
            Self::Set(x) => format!("set {x}"),
            Self::Add(x) => format!("add {x}"),
            Self::Mul(x) => format!("multiply by {x}"),
            Self::Scale(x) => format!("strength {x}"),
            Self::Clamp(a, b) => format!("clamp to {a}..{b}"),
            Self::Zero => "zero".to_owned(),
            Self::Identity => "identity".to_owned(),
            Self::Negate => "negate".to_owned(),
            Self::Paste => "paste captured values".to_owned(),
        }
    }
    fn step(&self, x: f32, identity: f32, pasted: Option<f32>) -> f32 {
        match self {
            Self::Set(v) => *v,
            Self::Add(v) => x + v,
            Self::Mul(v) => x * v,
            Self::Scale(v) => identity + v * (x - identity),
            Self::Clamp(a, b) => x.clamp(*a, *b),
            Self::Zero => 0.0,
            Self::Identity => identity,
            Self::Negate => -x,
            Self::Paste => pasted.unwrap_or(x),
        }
    }
}

/// What an entry does.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    Edit(Op),
    Mask,
    Freeze,
}
impl Action {
    /// `edit`, `mask` or `freeze`.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Edit(_) => "edit",
            Self::Mask => "mask",
            Self::Freeze => "freeze",
        }
    }
}

/// One entry of an overlay.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    /// Unique within the overlay (`e1`, `m2`, `f3`).
    pub id: String,
    pub action: Action,
    /// The selection as it was given (a path, `*` for the whole object).
    pub selection: String,
    /// The whole object (a mask: its effect left out while it stays plugged).
    pub whole: bool,
    pub regions: Vec<Region>,
    pub by: String,
    pub at_ms: u64,
    /// Why a part of the selection was left out (a tensor the form cannot mask), if any.
    pub note: String,
}

/// Keeps a region's captured values elsewhere: the name they are kept under, or `None` to write
/// them inline.
pub type Aside<'a> = &'a mut dyn FnMut(&[f32]) -> Result<Option<String>, String>;
/// The values kept under a name, with their count.
pub type Kept<'a> = &'a dyn Fn(&str, usize) -> Result<Vec<f32>, String>;

/// One object's overlay.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Overlay {
    pub key: String,
    /// The object's kind as the parameter editor names it (`open`, `base`, `filter`, ...).
    pub kind: String,
    pub model: String,
    pub entries: Vec<Entry>,
    /// The next entry number.
    pub next: u64,
}

/// The value that changes nothing for a tensor of the open weights or a base cartridge: 1 for a norm
/// weight (a gain on the normalized stream), 0 for every other set (a projection zeroed adds nothing
/// to the residual stream).
#[must_use]
pub fn weights_identity(name: &str) -> f32 {
    if name.ends_with("norm.weight") { 1.0 } else { 0.0 }
}

/// A value rounded to a stored element type, as the engine holds it.
#[must_use]
pub fn round_to(dtype: &str, x: f32) -> f32 {
    match dtype {
        "BF16" => f32::from_bits(u32::from(bf16(x)) << 16),
        "F16" => crate::half::from16(crate::half::bits16(x)),
        _ => x,
    }
}

/// f32 to bfloat16 bits, rounded to nearest even.
fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    let r = if x.is_nan() { (b >> 16) | 0x40 } else { (b + 0x7fff + ((b >> 16) & 1)) >> 16 };
    #[expect(clippy::cast_possible_truncation, reason = "the upper 16 bits of an f32")]
    let h = r as u16;
    h
}

impl Overlay {
    /// An empty overlay for an object.
    #[must_use]
    pub fn new(key: &str, kind: &str, model: &str) -> Self {
        Self { key: key.to_owned(), kind: kind.to_owned(), model: model.to_owned(), entries: Vec::new(), next: 1 }
    }
    /// Whether any entry changes a value (an edit or a mask).
    #[must_use]
    pub fn has_effect(&self) -> bool {
        self.entries.iter().any(|e| !matches!(e.action, Action::Freeze))
    }
    /// Whether the whole object is masked.
    #[must_use]
    pub fn whole_masked(&self) -> bool {
        self.entries.iter().any(|e| e.whole && matches!(e.action, Action::Mask))
    }
    /// Whether an edit or mask touches `tensor`.
    #[must_use]
    pub fn touches(&self, tensor: &str) -> bool {
        self.entries.iter().any(|e| !matches!(e.action, Action::Freeze) && e.regions.iter().any(|r| r.tensor == tensor))
    }
    /// The freeze entry that meets any of `regions`, if one does.
    #[must_use]
    pub fn frozen_by(&self, regions: &[Region]) -> Option<&Entry> {
        self.entries.iter().filter(|e| matches!(e.action, Action::Freeze)).find(|e| e.regions.iter().any(|f| regions.iter().any(|r| r.meets(f))))
    }
    /// Counts of edits, masks and freezes.
    #[must_use]
    pub fn counts(&self) -> (usize, usize, usize) {
        let c = |n: &str| self.entries.iter().filter(|e| e.action.name() == n).count();
        (c("edit"), c("mask"), c("freeze"))
    }

    /// Applies the overlay to one tensor's values in place (`vals` in row-major order, `dtype` the
    /// type the engine holds them in). Returns whether any value was touched.
    pub fn apply(&self, tensor: &str, dtype: &str, vals: &mut [f32]) -> bool {
        let mut touched = false;
        for e in &self.entries {
            let Action::Edit(op) = &e.action else { continue };
            for r in e.regions.iter().filter(|r| r.tensor == tensor) {
                r.each(|k, i| {
                    if let Some(x) = vals.get_mut(i) {
                        *x = round_to(dtype, op.step(*x, r.identity, r.values.as_ref().and_then(|v| v.get(k).copied())));
                        touched = true;
                    }
                });
            }
        }
        for e in self.entries.iter().filter(|e| matches!(e.action, Action::Mask)) {
            for r in e.regions.iter().filter(|r| r.tensor == tensor) {
                r.each(|_, i| {
                    if let Some(x) = vals.get_mut(i) {
                        *x = r.identity;
                        touched = true;
                    }
                });
            }
        }
        touched
    }

    /// The value at flat index `i` of `tensor` with the overlay applied to the released value `x`,
    /// whether it is masked, and whether it is frozen.
    #[must_use]
    pub fn at(&self, tensor: &str, dtype: &str, i: usize, x: f32) -> (f32, bool, bool) {
        let mut v = x;
        let (mut masked, mut frozen) = (false, false);
        for e in &self.entries {
            for r in e.regions.iter().filter(|r| r.tensor == tensor) {
                let Some(k) = r.place(i) else { continue };
                match &e.action {
                    Action::Edit(op) => v = round_to(dtype, op.step(v, r.identity, r.values.as_ref().and_then(|p| p.get(k).copied()))),
                    Action::Mask => masked = true,
                    Action::Freeze => frozen = true,
                }
            }
        }
        if masked {
            v = self.entries.iter().filter(|e| matches!(e.action, Action::Mask)).flat_map(|e| e.regions.iter()).find(|r| r.tensor == tensor && r.place(i).is_some()).map_or(v, |r| r.identity);
        }
        (v, masked, frozen)
    }

    /// Applies the overlay to point parts (`<target>.gain`, `<target>.bias`) in place. Returns whether
    /// any value was touched.
    pub fn apply_parts(&self, parts: &mut [Part]) -> bool {
        let mut touched = false;
        for p in parts.iter_mut() {
            touched |= self.apply(&format!("{}.gain", p.target), "F32", &mut p.gain);
            touched |= self.apply(&format!("{}.bias", p.target), "F32", &mut p.bias);
        }
        touched
    }

    /// Applies the overlay to stored bytes of one tensor in place (`BF16`, `F16` or `F32`, little
    /// endian). Returns whether any value was touched.
    /// # Errors
    /// A stored type that is not read here.
    pub fn apply_bytes(&self, tensor: &str, dtype: &str, bytes: &mut [u8]) -> Result<bool, String> {
        let w = match dtype {
            "BF16" | "F16" => 2,
            "F32" => 4,
            other => return Err(format!("{tensor}: {other} is not a stored type the overlay writes")),
        };
        let get = |b: &[u8], i: usize| -> Option<f32> {
            let s = b.get(i * w..i * w + w)?;
            Some(match dtype {
                "BF16" => f32::from_bits(u32::from(u16::from_le_bytes([*s.first()?, *s.get(1)?])) << 16),
                "F16" => crate::half::from16(u16::from_le_bytes([*s.first()?, *s.get(1)?])),
                _ => f32::from_le_bytes([*s.first()?, *s.get(1)?, *s.get(2)?, *s.get(3)?]),
            })
        };
        let put = |b: &mut [u8], i: usize, x: f32| {
            let raw: Vec<u8> = match dtype {
                "BF16" => bf16(x).to_le_bytes().to_vec(),
                "F16" => crate::half::bits16(x).to_le_bytes().to_vec(),
                _ => x.to_le_bytes().to_vec(),
            };
            if let Some(s) = b.get_mut(i * w..i * w + w) {
                s.copy_from_slice(&raw);
            }
        };
        let mut touched = false;
        for e in &self.entries {
            let Action::Edit(op) = &e.action else { continue };
            for r in e.regions.iter().filter(|r| r.tensor == tensor) {
                r.each(|k, i| {
                    if let Some(x) = get(bytes, i) {
                        put(bytes, i, op.step(x, r.identity, r.values.as_ref().and_then(|v| v.get(k).copied())));
                        touched = true;
                    }
                });
            }
        }
        for e in self.entries.iter().filter(|e| matches!(e.action, Action::Mask)) {
            for r in e.regions.iter().filter(|r| r.tensor == tensor) {
                r.each(|_, i| {
                    if i * w + w <= bytes.len() {
                        put(bytes, i, r.identity);
                        touched = true;
                    }
                });
            }
        }
        Ok(touched)
    }

    /// The overlay as its file's JSON, every captured value inline.
    #[must_use]
    pub fn to_value(&self) -> Value {
        self.to_value_with(&mut |_| Ok(None)).unwrap_or(Value::Null)
    }

    /// The overlay as its file's JSON; `aside` may keep a region's captured values elsewhere and
    /// return the name it keeps them under (written as `values_sha256` with `values_count`), or
    /// `None` to write them inline.
    /// # Errors
    /// As `aside`.
    pub fn to_value_with(&self, aside: Aside<'_>) -> Result<Value, String> {
        let mut kept: Vec<Vec<Option<String>>> = Vec::with_capacity(self.entries.len());
        for e in &self.entries {
            let mut row = Vec::with_capacity(e.regions.len());
            for r in &e.regions {
                row.push(match &r.values {
                    Some(v) => aside(v)?,
                    None => None,
                });
            }
            kept.push(row);
        }
        let s = |t: &str| Value::String(t.to_owned());
        let n = |x: u64| Value::Number(x.to_string());
        let f = |x: f32| Value::Number(format!("{x}"));
        let pair = |a: usize, b: usize| Value::Array(vec![n(a as u64), n(b as u64)]);
        let entries = self
            .entries
            .iter()
            .zip(&kept)
            .map(|(e, aside_of)| {
                let mut o = BTreeMap::new();
                o.insert("id".to_owned(), s(&e.id));
                o.insert("action".to_owned(), s(e.action.name()));
                if let Action::Edit(op) = &e.action {
                    let (name, args): (&str, Vec<(&str, f32)>) = match op {
                        Op::Set(x) => ("set", vec![("value", *x)]),
                        Op::Add(x) => ("add", vec![("value", *x)]),
                        Op::Mul(x) => ("mul", vec![("value", *x)]),
                        Op::Scale(x) => ("scale", vec![("value", *x)]),
                        Op::Clamp(a, b) => ("clamp", vec![("lo", *a), ("hi", *b)]),
                        Op::Zero => ("zero", vec![]),
                        Op::Identity => ("identity", vec![]),
                        Op::Negate => ("negate", vec![]),
                        Op::Paste => ("paste", vec![]),
                    };
                    let mut m: BTreeMap<String, Value> = args.into_iter().map(|(k, v)| (k.to_owned(), f(v))).collect();
                    m.insert("op".to_owned(), s(name));
                    o.insert("op".to_owned(), Value::Object(m));
                }
                o.insert("selection".to_owned(), s(&e.selection));
                o.insert("whole".to_owned(), Value::Bool(e.whole));
                o.insert("by".to_owned(), s(&e.by));
                o.insert("at_ms".to_owned(), n(e.at_ms));
                o.insert("note".to_owned(), s(&e.note));
                let regions = e
                    .regions
                    .iter()
                    .zip(aside_of)
                    .map(|(r, kept_as)| {
                        let mut m = BTreeMap::new();
                        m.insert("tensor".to_owned(), s(&r.tensor));
                        match r.span {
                            Span::Flat(a, b) => {
                                m.insert("flat".to_owned(), pair(a, b));
                            }
                            Span::Block { rows, cols } => {
                                m.insert("rows".to_owned(), pair(rows.0, rows.1));
                                m.insert("cols".to_owned(), pair(cols.0, cols.1));
                            }
                        }
                        m.insert("width".to_owned(), n(r.width as u64));
                        m.insert("identity".to_owned(), f(r.identity));
                        m.insert("dtype".to_owned(), s(&r.dtype));
                        match (&r.values, kept_as) {
                            (Some(v), Some(sha)) => {
                                m.insert("values_sha256".to_owned(), s(sha));
                                m.insert("values_count".to_owned(), n(v.len() as u64));
                            }
                            (Some(v), None) => {
                                m.insert("values".to_owned(), Value::Array(v.iter().map(|x| f(*x)).collect()));
                            }
                            _ => {}
                        }
                        Value::Object(m)
                    })
                    .collect();
                o.insert("regions".to_owned(), Value::Array(regions));
                Value::Object(o)
            })
            .collect();
        Ok(Value::Object(
            [
                ("format", s(FORMAT)),
                ("key", s(&self.key)),
                ("kind", s(&self.kind)),
                ("model", s(&self.model)),
                ("next", n(self.next)),
                ("entries", Value::Array(entries)),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect(),
        ))
    }

    /// Reads an overlay from its file's JSON, every captured value inline.
    /// # Errors
    /// Another format, or a field that does not read.
    pub fn from_value(v: &Value) -> Result<Self, String> {
        Self::from_value_with(v, &|_, _| Err("values kept beside the overlay file: read it with overlay::load".to_owned()))
    }

    /// `from_value`, with `kept` giving the values a region keeps elsewhere (by `values_sha256` and
    /// `values_count`).
    /// # Errors
    /// As `from_value`, or as `kept`.
    pub fn from_value_with(v: &Value, kept: Kept<'_>) -> Result<Self, String> {
        let text = |v: &Value, k: &str| v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default();
        if text(v, "format") != FORMAT {
            return Err(format!("not an overlay ({FORMAT})"));
        }
        let fl = |v: &Value, k: &str| -> Result<f32, String> {
            match v.get(k) {
                Ok(Value::Number(t) | Value::String(t)) => t.trim().parse::<f32>().ok().filter(|x| x.is_finite()).ok_or_else(|| format!("{k}: a finite number")),
                _ => Err(format!("{k}: a number")),
            }
        };
        let int = |v: &Value| v.integer().ok().and_then(|x| usize::try_from(x).ok());
        let pair = |v: &Value, k: &str| -> Option<(usize, usize)> {
            match v.get(k).and_then(Value::array).ok()? {
                [a, b] => Some((int(a)?, int(b)?)),
                _ => None,
            }
        };
        let mut entries = Vec::new();
        for e in v.get("entries").and_then(Value::array).unwrap_or_default() {
            let action = match text(e, "action").as_str() {
                "mask" => Action::Mask,
                "freeze" => Action::Freeze,
                "edit" => {
                    let o = e.get("op").map_err(|_| "edit: no op".to_owned())?;
                    Action::Edit(match text(o, "op").as_str() {
                        "set" => Op::Set(fl(o, "value")?),
                        "add" => Op::Add(fl(o, "value")?),
                        "mul" => Op::Mul(fl(o, "value")?),
                        "scale" => Op::Scale(fl(o, "value")?),
                        "clamp" => Op::Clamp(fl(o, "lo")?, fl(o, "hi")?),
                        "zero" => Op::Zero,
                        "identity" => Op::Identity,
                        "negate" => Op::Negate,
                        "paste" => Op::Paste,
                        x => return Err(format!("edit: op {x} is not replayed")),
                    })
                }
                x => return Err(format!("entry action {x}: edit, mask or freeze")),
            };
            let mut regions = Vec::new();
            for r in e.get("regions").and_then(Value::array).unwrap_or_default() {
                let span = match (pair(r, "flat"), pair(r, "rows"), pair(r, "cols")) {
                    (Some((a, b)), _, _) => Span::Flat(a, b),
                    (None, Some(rows), Some(cols)) => Span::Block { rows, cols },
                    _ => return Err(format!("{}: a region of flat, or rows and cols", text(r, "tensor"))),
                };
                let values = match r.get("values") {
                    Ok(Value::Array(a)) => Some(a.iter().map(|x| match x {
                        Value::Number(t) => t.parse::<f32>().map_err(|_| "values: numbers".to_owned()),
                        _ => Err("values: numbers".to_owned()),
                    }).collect::<Result<Vec<f32>, String>>()?),
                    _ => match r.get("values_sha256").and_then(Value::string) {
                        Ok(sha) => Some(kept(sha, r.get("values_count").ok().and_then(int).unwrap_or(0))?),
                        Err(_) => None,
                    },
                };
                regions.push(Region { tensor: text(r, "tensor"), span, width: r.get("width").ok().and_then(int).unwrap_or(1), identity: fl(r, "identity").unwrap_or(0.0), dtype: text(r, "dtype"), values });
            }
            entries.push(Entry {
                id: text(e, "id"),
                action,
                selection: text(e, "selection"),
                whole: matches!(e.get("whole"), Ok(Value::Bool(true))),
                regions,
                by: text(e, "by"),
                at_ms: e.get("at_ms").and_then(Value::integer).unwrap_or(0),
                note: text(e, "note"),
            });
        }
        Ok(Self { key: text(v, "key"), kind: text(v, "kind"), model: text(v, "model"), entries, next: v.get("next").and_then(Value::integer).unwrap_or(1) })
    }
}

/// The file name of an object's overlay: the key with every character outside `[A-Za-z0-9._-]`
/// written as `_`, and the first 8 hex digits of the key's SHA-256 so two keys never share a file.
#[must_use]
pub fn slug(key: &str) -> String {
    let safe: String = key.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') { c } else { '_' }).collect();
    let h = nmhash::sha256_bytes(key.as_bytes());
    format!("{safe}-{}", h.get(..8).unwrap_or(""))
}

/// The overlay file of an object.
#[must_use]
pub fn path(home: &Path, key: &str) -> PathBuf {
    home.join(DIR).join(format!("{}.json", slug(key)))
}

/// An object's overlay, if it has one.
/// # Errors
/// A file that does not read, is not an overlay, or belongs to another key.
pub fn load(home: &Path, key: &str) -> Result<Option<Overlay>, String> {
    let p = path(home, key);
    let b = match std::fs::read(&p) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("{}: {e}", p.display())),
    };
    let o = decode(home, &b).map_err(|e| format!("{}: {e}", p.display()))?;
    if o.key != key {
        return Err(format!("{}: holds the overlay of {}, not {key}", p.display(), o.key));
    }
    Ok(Some(o))
}

/// The overlay that has an effect, if any (a file that does not read is reported once on standard
/// error and taken as no overlay, so a damaged file never changes what serves).
#[must_use]
pub fn effective(home: &Path, key: &str) -> Option<Overlay> {
    match load(home, key) {
        Ok(o) => o.filter(Overlay::has_effect),
        Err(e) => {
            static SAID: Mutex<Vec<String>> = Mutex::new(Vec::new());
            if let Ok(mut s) = SAID.lock()
                && !s.contains(&e)
            {
                eprintln!("overlay: {e}; served without it");
                s.push(e);
            }
            None
        }
    }
}

/// Writes an object's overlay (through a temporary file, read back, renamed over the last one).
/// # Errors
/// The folder or the file cannot be written.
pub fn save(home: &Path, o: &Overlay) -> Result<PathBuf, String> {
    let p = path(home, &o.key);
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let bytes = encode(home, o)?;
    let tmp = p.with_extension("tmp");
    std::fs::write(&tmp, bytes.as_bytes()).map_err(|e| format!("{}: {e}", tmp.display()))?;
    if std::fs::read(&tmp).map_err(|e| e.to_string())? != bytes.as_bytes() {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("{}: read-back differs", tmp.display()));
    }
    std::fs::rename(&tmp, &p).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok(p)
}

/// Captured values of more than this many are kept in `memory/overlays/values/<sha256>.f32`
/// (little-endian f32, named by their content's hash, never written over) rather than inline.
const INLINE_MAX: usize = 4096;
const VALUES: &str = "memory/overlays/values";

fn values_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// An overlay as the text of its file, large captured values kept beside it.
/// # Errors
/// A values file that cannot be written.
pub fn encode(home: &Path, o: &Overlay) -> Result<String, String> {
    let dir = home.join(VALUES);
    let v = o.to_value_with(&mut |vals: &[f32]| {
        if vals.len() <= INLINE_MAX {
            return Ok(None);
        }
        let b = values_bytes(vals);
        let sha = nmhash::sha256_bytes(&b);
        let p = dir.join(format!("{sha}.f32"));
        if !p.is_file() {
            std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            let tmp = p.with_extension("tmp");
            std::fs::write(&tmp, &b).map_err(|e| format!("{}: {e}", tmp.display()))?;
            std::fs::rename(&tmp, &p).map_err(|e| format!("{}: {e}", p.display()))?;
        }
        Ok(Some(sha))
    })?;
    Ok(nmjson::pretty::pretty(&v))
}

/// An overlay from the bytes of its file, values kept beside it read back and checked by hash.
/// # Errors
/// Text that does not read as an overlay, or a values file missing or not matching its hash.
pub fn decode(home: &Path, bytes: &[u8]) -> Result<Overlay, String> {
    let v = nmjson::parse(bytes).map_err(|e| e.to_string())?;
    let dir = home.join(VALUES);
    Overlay::from_value_with(&v, &|sha: &str, count: usize| {
        let p = dir.join(format!("{sha}.f32"));
        let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        if nmhash::sha256_bytes(&b) != sha || b.len() != count * 4 {
            return Err(format!("{}: does not match its hash or count", p.display()));
        }
        Ok(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
    })
}

/// A stamp of every overlay file (name, time and length): it changes whenever an overlay is written,
/// so a serving process sees a change at its next look.
#[must_use]
pub fn stamp(home: &Path) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let Ok(rd) = std::fs::read_dir(home.join(DIR)) else { return 0 };
    let mut rows: Vec<(String, u128, u64)> = rd
        .flatten()
        .filter_map(|e| {
            let m = e.metadata().ok().filter(std::fs::Metadata::is_file)?;
            let t = m.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos();
            Some((e.file_name().to_string_lossy().into_owned(), t, m.len()))
        })
        .filter(|(n, _, _)| n.ends_with(".json"))
        .collect();
    rows.sort();
    for (n, t, l) in rows {
        for b in n.bytes().chain(t.to_le_bytes()).chain(l.to_le_bytes()) {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// `stamp`, looked at most once a second per folder, for paths a call runs on.
#[must_use]
pub fn stamp_cached(home: &Path) -> u64 {
    static LAST: Mutex<Vec<(PathBuf, std::time::Instant, u64)>> = Mutex::new(Vec::new());
    let now = std::time::Instant::now();
    if let Ok(g) = LAST.lock()
        && let Some((_, at, s)) = g.iter().find(|(h, _, _)| h == home)
        && now.duration_since(*at) < std::time::Duration::from_secs(1)
    {
        return *s;
    }
    let s = stamp(home);
    if let Ok(mut g) = LAST.lock() {
        g.retain(|(h, _, _)| h != home);
        g.push((home.to_path_buf(), now, s));
    }
    s
}

/// Where the serving process records how long binding a MicroModel again with its overlays took.
const RELOADS: &str = "memory/overlays/state/reloads.json";

/// Records the seconds binding `model` again with its overlays took on this machine (the newest).
pub fn record_reload(home: &Path, model: &str, secs: f64) {
    let p = home.join(RELOADS);
    let mut m = std::fs::read(&p).ok().and_then(|b| nmjson::parse(&b).ok()).and_then(|v| if let Value::Object(o) = v { Some(o) } else { None }).unwrap_or_default();
    let row: BTreeMap<String, Value> = [("secs".to_owned(), Value::Number(format!("{secs:.3}"))), ("at_ms".to_owned(), Value::Number(nmtype::now_ms().to_string()))].into_iter().collect();
    m.insert(model.to_owned(), Value::Object(row));
    if let Some(d) = p.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let _ = std::fs::write(&p, nmjson::pretty::pretty(&Value::Object(m)));
}

/// The seconds the newest recorded reload of `model` took on this machine, if one was measured.
#[must_use]
pub fn reload_secs(home: &Path, model: &str) -> Option<f64> {
    let v = nmjson::parse(&std::fs::read(home.join(RELOADS)).ok()?).ok()?;
    match v.get(model).ok()?.get("secs").ok()? {
        Value::Number(t) => t.parse().ok(),
        _ => None,
    }
}

/// A short identity of an overlay's effect (the first 12 hex digits of the SHA-256 of its file's
/// JSON), for keys of outputs kept on disk.
#[must_use]
pub fn identity(o: &Overlay) -> String {
    // Large captured values enter by their hash, so the identity is cheap however much is captured.
    let v = o.to_value_with(&mut |vals: &[f32]| Ok((vals.len() > INLINE_MAX).then(|| nmhash::sha256_bytes(&values_bytes(vals))))).unwrap_or(Value::Null);
    let h = nmhash::sha256_bytes(v.encode().unwrap_or_default().as_bytes());
    h.get(..12).unwrap_or("").to_owned()
}

/// The descriptors of a population with an overlay applied: every tensor it touches read from its
/// file (or taken from RAM by its content hash), patched in a copy, and offered under the copy's own
/// content hash; the released files are never written.
#[derive(Debug)]
pub struct Patched {
    pub descs: Vec<TensorDesc>,
    /// The patched copies; held while the population serves (`hold`).
    pub held: Vec<Arc<[u8]>>,
    /// The tensors patched.
    pub tensors: Vec<String>,
    /// `identity` of the overlay.
    pub identity: String,
}

/// A tensor name as the engine's population names it (`layers.3...` of a headless release is
/// `model.layers.3...`).
fn canonical(n: &str) -> String {
    if n.starts_with("model.") || n == "lm_head.weight" { n.to_owned() } else { format!("model.{n}") }
}

/// Patches `descs` (a population's tensors; relative file names are read under `dir`) with the
/// overlay of `key`. `None` when the object has no overlay with an effect.
/// # Errors
/// A touched tensor that cannot be read, or of a stored type the overlay does not write.
pub fn patch(home: &Path, key: &str, descs: &[TensorDesc], dir: &Path) -> Result<Option<Patched>, String> {
    let Some(o) = effective(home, key) else { return Ok(None) };
    patch_with(&o, descs, dir).map(Some)
}

/// `patch` with an overlay in hand.
/// # Errors
/// As `patch`.
pub fn patch_with(o: &Overlay, descs: &[TensorDesc], dir: &Path) -> Result<Patched, String> {
    // Region tensors by the population's names.
    let mut by: BTreeMap<String, String> = BTreeMap::new();
    for e in &o.entries {
        for r in &e.regions {
            by.insert(canonical(&r.tensor), r.tensor.clone());
        }
    }
    let mut out = Vec::with_capacity(descs.len());
    let (mut held, mut tensors) = (Vec::new(), Vec::new());
    for d in descs {
        let Some(own) = by.get(&canonical(&d.name)).filter(|n| o.touches(n)) else {
            out.push(d.clone());
            continue;
        };
        let mut bytes = match (!d.sha256.is_empty()).then(|| nmnpcs::provision::held(&d.sha256)).flatten() {
            Some(b) => b.to_vec(),
            None => {
                use std::io::{Read as _, Seek as _};
                let file = dir.join(&d.file);
                let mut f = std::fs::File::open(&file).map_err(|e| format!("{}: {e}", file.display()))?;
                f.seek(std::io::SeekFrom::Start(d.start)).map_err(|e| format!("{}: {e}", file.display()))?;
                let mut b = vec![0_u8; usize::try_from(d.end.saturating_sub(d.start)).map_err(|e| e.to_string())?];
                f.read_exact(&mut b).map_err(|e| format!("{}: {}: {e}", file.display(), d.name))?;
                b
            }
        };
        o.apply_bytes(own, &d.dtype, &mut bytes)?;
        let sha = nmhash::sha256_bytes(&bytes);
        let arc: Arc<[u8]> = Arc::from(bytes);
        nmnpcs::provision::hold(&sha, &arc);
        held.push(arc);
        tensors.push(d.name.clone());
        out.push(TensorDesc { sha256: sha, ..d.clone() });
    }
    Ok(Patched { descs: out, held, tensors, identity: identity(o) })
}

/// Whether `Population::open` and `Population::open_cartridge` apply the object's overlay as they
/// open (`open:<model>`, the cartridge's key): set by the lab (on by default, off with
/// `--no-overlay`). A serving process leaves it off and applies overlays itself, so each base
/// cartridge's seat check compares it with the open weights as released.
pub static AT_OPEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The keys of the objects whose overlay in `home` holds an effect.
#[must_use]
pub fn effective_keys(home: &Path) -> Vec<String> {
    std::fs::read_dir(home.join(DIR))
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
                .filter_map(|e| std::fs::read(e.path()).ok().and_then(|b| decode(home, &b).ok()))
                .filter(Overlay::has_effect)
                .map(|o| o.key)
                .collect()
        })
        .unwrap_or_default()
}

/// The line a lab report records (`lab_line` in a fixed form): `overlay: applied (<n> objects)`,
/// `overlay: not applied (--no-overlay)` or `overlay: none present` (no overlay with an effect).
#[must_use]
pub fn report_line(home: &Path, on: bool) -> String {
    if !on {
        return "overlay: not applied (--no-overlay)".to_owned();
    }
    match effective_keys(home).len() {
        0 => "overlay: none present".to_owned(),
        n => format!("overlay: applied ({n} objects)"),
    }
}

/// The line a lab run states: whether overlays are applied, and which objects hold one with an
/// effect.
#[must_use]
pub fn lab_line(home: &Path, on: bool) -> String {
    if !on {
        return "parameter overlays not applied (--no-overlay): every population is read from its released files".to_owned();
    }
    let keys = effective_keys(home);
    if keys.is_empty() {
        "parameter overlays applied: none holds an effect, so the released files are read as they are".to_owned()
    } else {
        format!("parameter overlays applied where a population opens ({}); --no-overlay reads the released files", keys.join(", "))
    }
}

/// Patched copies held per MicroModel while it serves; the provisioner finds them by content hash.
static HELD: Mutex<BTreeMap<String, Vec<Arc<[u8]>>>> = Mutex::new(BTreeMap::new());

/// Holds a MicroModel's patched copies (replacing what was held for it).
pub fn hold(model: &str, held: Vec<Arc<[u8]>>) {
    if let Ok(mut g) = HELD.lock() {
        g.insert(model.to_owned(), held);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(t: &str, span: Span, width: usize, identity: f32) -> Region {
        Region { tensor: t.into(), span, width, identity, dtype: "F32".into(), values: None }
    }
    fn entry(id: &str, action: Action, regions: Vec<Region>) -> Entry {
        Entry { id: id.into(), action, selection: String::new(), whole: false, regions, by: "test".into(), at_ms: 0, note: String::new() }
    }

    #[test]
    fn edits_replay_in_order_and_masks_win_then_restore_exactly() {
        let mut o = Overlay::new("k", "filter", "m");
        let r = region("w", Span::Flat(1, 3), 4, 0.0);
        o.entries.push(entry("e1", Action::Edit(Op::Add(1.0)), vec![r.clone()]));
        o.entries.push(entry("e2", Action::Edit(Op::Mul(2.0)), vec![r.clone()]));
        let base = vec![1.0, 2.0, 3.0, 4.0];
        let mut v = base.clone();
        assert!(o.apply("w", "F32", &mut v));
        assert_eq!(v, vec![1.0, 6.0, 8.0, 4.0], "add then multiply, in entry order");
        o.entries.swap(0, 1);
        let mut v2 = base.clone();
        o.apply("w", "F32", &mut v2);
        assert_eq!(v2, vec![1.0, 5.0, 7.0, 4.0], "the order is kept");
        o.entries.push(entry("m3", Action::Mask, vec![region("w", Span::Flat(2, 4), 4, 0.0)]));
        let mut m = base.clone();
        o.apply("w", "F32", &mut m);
        assert_eq!(m, vec![1.0, 5.0, 0.0, 0.0], "a mask wins over the edits");
        assert_eq!(o.at("w", "F32", 2, 3.0), (0.0, true, false));
        o.entries.retain(|e| e.id != "m3");
        let mut back = base.clone();
        o.apply("w", "F32", &mut back);
        assert_eq!(back, v2, "unmasking returns the edited values bit for bit");
        assert_eq!(o.at("w", "F32", 1, 2.0).0, v2[1]);
    }

    #[test]
    fn regions_meet_across_flat_ranges_and_blocks() {
        let blk = region("m", Span::Block { rows: (1, 3), cols: (2, 4) }, 5, 0.0);
        assert!(blk.meets(&region("m", Span::Flat(7, 8), 5, 0.0)), "row 1, column 2");
        assert!(!blk.meets(&region("m", Span::Flat(5, 7), 5, 0.0)), "row 1, columns 0 and 1");
        assert!(blk.meets(&region("m", Span::Flat(0, 25), 5, 0.0)));
        assert!(!blk.meets(&region("other", Span::Flat(0, 25), 5, 0.0)));
        assert_eq!(blk.place(13), Some(3));
        let mut seen = Vec::new();
        blk.each(|k, i| seen.push((k, i)));
        assert_eq!(seen, vec![(0, 7), (1, 8), (2, 12), (3, 13)]);
    }

    #[test]
    fn bytes_round_to_the_stored_type_and_the_file_form_reads_back() {
        let mut o = Overlay::new("open:x", "open", "x");
        let mut r = region("t", Span::Flat(0, 2), 4, 0.0);
        r.dtype = "BF16".into();
        o.entries.push(entry("e1", Action::Edit(Op::Add(0.001)), vec![r.clone()]));
        o.entries.push(entry("m2", Action::Mask, vec![Region { identity: 1.0, ..region("t", Span::Flat(3, 4), 4, 1.0) }]));
        let vals = [1.0_f32, -2.0, 3.0, 4.0];
        let mut bytes: Vec<u8> = vals.iter().flat_map(|x| bf16(*x).to_le_bytes()).collect();
        assert!(o.apply_bytes("t", "BF16", &mut bytes).unwrap());
        let back: Vec<f32> = bytes.as_chunks::<2>().0.iter().map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16)).collect();
        let mut want = vals.to_vec();
        o.apply("t", "BF16", &mut want);
        assert_eq!(back, want, "bytes and values agree after rounding to bfloat16");
        assert_eq!(back[3], 1.0);
        let again = Overlay::from_value(&nmjson::parse(nmjson::pretty::pretty(&o.to_value()).as_bytes()).unwrap()).unwrap();
        assert_eq!(again, o);
        let mut parts = vec![Part { target: "L1.o".into(), gain: vec![2.0, 2.0], bias: vec![1.0, 1.0] }];
        let mut f = Overlay::new("filter:r/j", "filter", "m");
        f.entries.push(entry("m1", Action::Mask, vec![region("L1.o.gain", Span::Flat(0, 2), 2, 1.0), region("L1.o.bias", Span::Flat(0, 2), 2, 0.0)]));
        assert!(f.apply_parts(&mut parts));
        assert_eq!((parts[0].gain.clone(), parts[0].bias.clone()), (vec![1.0, 1.0], vec![0.0, 0.0]));
    }

    #[test]
    fn large_captures_are_kept_beside_the_file_and_read_back_by_hash() {
        let home = std::env::temp_dir().join(format!("nano-overlay-aside-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        let mut o = Overlay::new("open:x", "open", "x");
        #[expect(clippy::cast_precision_loss, reason = "test values")]
        let vals: Vec<f32> = (0..10_000).map(|i| i as f32 * 0.5).collect();
        let mut r = region("t", Span::Flat(0, 10_000), 10_000, 0.0);
        r.values = Some(vals);
        o.entries.push(entry("e1", Action::Edit(Op::Paste), vec![r]));
        save(&home, &o).unwrap();
        let text = std::fs::read_to_string(path(&home, "open:x")).unwrap();
        assert!(text.contains("values_sha256") && text.len() < 4000, "the overlay file stays small");
        let back = load(&home, "open:x").unwrap().unwrap();
        assert_eq!(back, o, "read back whole");
        assert_eq!(identity(&back), identity(&o));
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn a_patch_reads_the_file_copies_and_never_writes_it() {
        let dir = std::env::temp_dir().join(format!("nano-overlay-patch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let vals = [0.5_f32, 1.5, -1.0, 2.0];
        let raw: Vec<u8> = vals.iter().flat_map(|x| x.to_le_bytes()).collect();
        let file = dir.join("w.bin");
        std::fs::write(&file, [vec![9_u8; 8], raw.clone()].concat()).unwrap();
        let before = nmhash::sha256_bytes(&std::fs::read(&file).unwrap());
        let desc = TensorDesc { name: "model.layers.0.mlp.down_proj.weight".into(), dtype: "F32".into(), shape: vec![2, 2], file: "w.bin".into(), start: 8, end: 24, sha256: String::new() };
        let other = TensorDesc { name: "model.norm.weight".into(), ..desc.clone() };
        let mut o = Overlay::new("open:x", "open", "x");
        o.entries.push(entry("m1", Action::Mask, vec![region("layers.0.mlp.down_proj.weight", Span::Block { rows: (0, 1), cols: (0, 2) }, 2, 0.0)]));
        let p = patch_with(&o, &[desc, other], &dir).unwrap();
        assert_eq!(p.tensors, vec!["model.layers.0.mlp.down_proj.weight".to_owned()], "a headless name meets its canonical name");
        assert!(p.descs[1].sha256.is_empty(), "an untouched tensor is offered as it was");
        let held = nmnpcs::provision::held(&p.descs[0].sha256).unwrap();
        let got: Vec<f32> = held.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        assert_eq!(got, vec![0.0, 0.0, -1.0, 2.0]);
        assert_eq!(nmhash::sha256_bytes(&std::fs::read(&file).unwrap()), before, "the file is never written");
        drop(p);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
