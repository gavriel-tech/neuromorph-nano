//! The whole-set parameter overview: every tensor of one opened object laid out, and any tensor or
//! region of one summarised as a grid of cells, each cell the minimum, maximum, mean and RMS of the
//! weights it covers.
//!
//! Owner's direction (2026-11-26, Luis Jake Gabriel III): "What about seeing the whole one parameter
//! set with each parameter and weight?" and "The weights shouldn't be opened up or shown unless they
//! are opened up in the editor." Restated: the overview is read only for an object the user has
//! opened in the editor, on request; nothing is read beforehand and nothing polls.
//!
//! Two reads, `GET /api/parameters/overview` and `nm_cartridges` `parameters_overview`:
//! - the set (`key` alone): the object's tensors in layout order (tensors outside a layer that come
//!   first in the forward pass, such as the embedding table, then each layer in order, then the rest),
//!   each with its shape, element type, layer, block and its two-dimensional view (`rows`, `cols`).
//!   For the open weights (`open:<model>`) only the safetensors headers are read; other objects are
//!   read as `cparams::held` reads them;
//! - one tensor (`tensor`), or a region of it (rows `r0..r1`, columns `c0..c1`), as a `w x h` grid.
//!   A tensor is viewed as two-dimensional: a vector is one row, a tensor of three or more dimensions
//!   is its first dimension by the product of the rest. Cell `j` of `w` covers columns
//!   `c0 + floor(j * W / w) .. c0 + floor((j + 1) * W / w)` (rows likewise), so every cell covers at
//!   least one weight. The region is streamed from the file in chunks of at most `CHUNK` bytes and
//!   folded into the cells as it is read, so a tensor is never loaded whole: the memory held is the
//!   cells and one chunk. When every cell covers one weight (`exact`), the exact values are returned
//!   in place of the four summaries.
//! - with `hist=1`, the region's distribution, folded in the same streamed pass: the weights are
//!   counted by their top sixteen bits (the bf16 pattern, exact for BF16 and to bf16 precision for
//!   F32 and F16), and from those counts the reply gives `HIST_BINS` linear bins from the region's
//!   minimum to its maximum, the value at each whole percentile 0..=100, the count of exact zeros,
//!   the standard deviation and the L2 norm. The memory held for it is one table of 65,536 counts.
//!
//! The open weights and every object are only read here.
use nmjson::Value;
use std::io::{Read as _, Seek as _};
use std::path::{Path, PathBuf};

/// The overview's format name.
pub const FORMAT: &str = "nm-parameter-overview/1";
/// Bytes read from a file at one time.
pub const CHUNK: usize = 4 << 20;
/// Cells in one grid at most.
pub const CELLS_MAX: usize = 65_536;
/// Cells on one side of a grid at most.
pub const SIDE_MAX: usize = 1024;
/// Cells on a side when the request does not say.
const SIDE_DEFAULT: usize = 128;
/// Linear bins of the distribution (`hist=1`).
pub const HIST_BINS: usize = 64;
/// Buckets of the distribution table: one per bf16 bit pattern.
const BUCKETS: usize = 1 << 16;

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}
fn nu(v: usize) -> Value {
    n(v as u64)
}
/// A figure at f32 precision (the precision parameters are held in); non-finite is `null`.
fn f32v(y: f32) -> Value {
    if y.is_finite() { Value::Number(format!("{y:e}")) } else { Value::Null }
}
#[expect(clippy::cast_possible_truncation, reason = "figures are reported at the f32 precision parameters are held in")]
fn f64v(x: f64) -> Value {
    f32v(x as f32)
}
fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn fail(e: &str) -> Value {
    obj(vec![("error", s(e))])
}

// ---------------------------------------------------------------------------------------------
// The tensors of one object.

/// One tensor of an object, as the overview reads it.
#[derive(Clone, Debug)]
struct Src {
    name: String,
    layer: Option<usize>,
    part: String,
    shape: Vec<usize>,
    dtype: String,
    /// Values held in memory (small objects).
    values: Option<Vec<f32>>,
    /// The file and byte offset of the values, for a streamed read.
    at: Option<(PathBuf, u64)>,
    /// Why the values cannot be read here.
    unread: Option<String>,
}
impl Src {
    fn size(&self) -> usize {
        self.shape.iter().product()
    }
    /// The two-dimensional view: (rows, columns).
    fn dims(&self) -> (usize, usize) {
        match self.shape.as_slice() {
            [] => (1, 1),
            [c] => (1, *c),
            [r, rest @ ..] => (*r, rest.iter().product()),
        }
    }
    fn readable(&self) -> bool {
        self.values.is_some() || (self.at.is_some() && width(&self.dtype).is_some())
    }
}

/// One object's tensors in layout order.
struct Set {
    key: String,
    model: String,
    kind: String,
    name: String,
    tensors: Vec<Src>,
}

/// Element bytes of a stored type read here.
fn width(dtype: &str) -> Option<usize> {
    match dtype {
        "BF16" | "F16" => Some(2),
        "F32" => Some(4),
        _ => None,
    }
}

/// The block a tensor of the open weights belongs to, from its name (as `cparams` names it).
fn open_block(name: &str) -> &'static str {
    if name.contains("embed_tokens") {
        "table"
    } else if name.contains("self_attn") {
        "attention"
    } else if name.contains(".mlp.") {
        "mlp"
    } else if name.contains("norm") {
        "norm"
    } else if name.contains("lm_head") {
        "head"
    } else {
        "other"
    }
}

/// Layout order: tensors outside a layer that come first in the forward pass (the token table),
/// then the layers in order (each layer's tensors as the object lists them), then the rest.
fn order(ts: &mut [Src]) {
    ts.sort_by_key(|t| match t.layer {
        None if matches!(t.part.as_str(), "table" | "embed") => (0, 0),
        Some(l) => (1, l),
        None => (2, 0),
    });
}

/// The tensors of a safetensors folder, from its headers alone (no value is read).
fn open_index(key: &str, model: &str, folder: &Path) -> Result<Set, String> {
    let idx = nmsafe::TensorIndex::open(folder).map_err(|e| format!("{model}: the open weights are not on this machine or do not read ({e})"))?;
    let mut tensors: Vec<Src> = idx
        .tensors
        .values()
        .map(|t| {
            let shape: Vec<usize> = t.shape.iter().filter_map(|x| usize::try_from(*x).ok()).collect();
            let layer = t.name.strip_prefix("model.layers.").and_then(|r| r.split('.').next()).and_then(|x| x.parse::<usize>().ok());
            let ok = width(&t.dtype).is_some();
            Src {
                name: t.name.clone(),
                layer,
                part: open_block(&t.name).to_owned(),
                shape,
                dtype: t.dtype.clone(),
                values: None,
                at: ok.then(|| (t.file.clone(), t.start)),
                unread: (!ok).then(|| format!("{}: a stored type not read here", t.dtype)),
            }
        })
        .collect();
    order(&mut tensors);
    Ok(Set { key: key.to_owned(), model: model.to_owned(), kind: "open".to_owned(), name: format!("{model} open weights"), tensors })
}

/// The object a key names: the open weights by their headers, anything else as `cparams` reads it.
fn set_of(home: &Path, key: &str) -> Result<Set, String> {
    if let Some(model) = key.strip_prefix(crate::cparams::OPEN_PREFIX) {
        if model.is_empty() || model.contains(['/', '\\']) || model.contains("..") {
            return Err(format!("{key}: not a MicroModel id"));
        }
        return open_index(key, model, &nmtype::models_dir(home).join(model).join("files"));
    }
    let h = crate::cparams::held(home, key)?;
    let mut tensors: Vec<Src> = h
        .tensors
        .into_iter()
        .map(|t| {
            let at = t.at.filter(|_| width(&t.dtype).is_some());
            let unread = if t.values.is_none() && at.is_none() { Some(t.unread.unwrap_or_else(|| format!("{}: its values are not held here", t.dtype))) } else { None };
            Src { name: t.name, layer: t.layer, part: t.part, shape: t.shape, dtype: t.dtype, values: t.values, at, unread }
        })
        .collect();
    order(&mut tensors);
    Ok(Set { key: h.key, model: h.model, kind: h.kind.to_owned(), name: h.name, tensors })
}

fn shape_v(shape: &[usize]) -> Value {
    Value::Array(shape.iter().map(|x| nu(*x)).collect())
}

/// The set-level reply: every tensor in layout order, no values.
fn set_view(set: &Set) -> Value {
    let rows: Vec<Value> = set
        .tensors
        .iter()
        .map(|t| {
            let (r, c) = t.dims();
            obj(vec![
                ("name", s(&t.name)),
                ("layer", t.layer.map_or(Value::Null, nu)),
                ("part", s(&t.part)),
                ("shape", shape_v(&t.shape)),
                ("dtype", s(&t.dtype)),
                ("rows", nu(r)),
                ("cols", nu(c)),
                ("size", nu(t.size())),
                ("readable", Value::Bool(t.readable())),
                ("unread", t.unread.as_deref().map_or(Value::Null, s)),
            ])
        })
        .collect();
    obj(vec![
        ("format", s(FORMAT)),
        ("key", s(&set.key)),
        ("model", s(&set.model)),
        ("kind", s(&set.kind)),
        ("name", s(&set.name)),
        ("count", nu(rows.len())),
        ("weights", nu(set.tensors.iter().map(Src::size).sum())),
        ("tensors", Value::Array(rows)),
    ])
}

// ---------------------------------------------------------------------------------------------
// The streamed read.

/// Where a tensor's values are read from, one chunk at a time.
enum Store<'a> {
    Mem(&'a [f32]),
    File { fh: std::fs::File, base: u64, dtype: &'a str, width: usize, raw: Vec<u8> },
}
impl Store<'_> {
    fn width(&self) -> usize {
        match self {
            Store::Mem(_) => 4,
            Store::File { width, .. } => *width,
        }
    }
    /// `count` values from flat element `off` into `out` (cleared first).
    fn read(&mut self, off: usize, count: usize, out: &mut Vec<f32>) -> Result<(), String> {
        out.clear();
        match self {
            Store::Mem(v) => out.extend_from_slice(v.get(off..off + count).ok_or_else(|| format!("values {off}..{} outside the {} held", off + count, v.len()))?),
            Store::File { fh, base, dtype, width, raw } => {
                let io = |e: std::io::Error| format!("read at element {off}: {e}");
                fh.seek(std::io::SeekFrom::Start(*base + (off as u64) * (*width as u64))).map_err(io)?;
                raw.resize(count * *width, 0);
                fh.read_exact(raw).map_err(io)?;
                match *dtype {
                    "BF16" => out.extend(raw.as_chunks::<2>().0.iter().map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16))),
                    "F16" => out.extend(raw.as_chunks::<2>().0.iter().map(|c| nmengn::half::from16(u16::from_le_bytes(*c)))),
                    "F32" => out.extend(raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c))),
                    other => return Err(format!("{other}: a stored type not read here")),
                }
            }
        }
        Ok(())
    }
}

/// Rows `r0..r1` and columns `c0..c1` of a tensor's two-dimensional view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Region {
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
}

/// The cells of one grid and what was read to fill them.
#[derive(Debug)]
struct Grid {
    min: Vec<f32>,
    max: Vec<f32>,
    sum: Vec<f64>,
    sq: Vec<f64>,
    count: Vec<u64>,
    nonfinite: u64,
    bytes: u64,
    chunks: u64,
    largest: usize,
    /// The distribution table (`hist=1`): counts by the top sixteen bits of each finite weight.
    hist: Option<Vec<u64>>,
    /// Weights exactly zero (either sign).
    zeros: u64,
}
impl Grid {
    fn new(w: usize, h: usize, hist: bool) -> Self {
        let k = w * h;
        Self { min: vec![f32::INFINITY; k], max: vec![f32::NEG_INFINITY; k], sum: vec![0.0; k], sq: vec![0.0; k], count: vec![0; k], nonfinite: 0, bytes: 0, chunks: 0, largest: 0, hist: hist.then(|| vec![0; BUCKETS]), zeros: 0 }
    }
    fn add(&mut self, cell: usize, x: f32) {
        if !x.is_finite() {
            self.nonfinite += 1;
            return;
        }
        if x == 0.0 {
            self.zeros += 1;
        }
        if let Some(b) = self.hist.as_mut().and_then(|h| h.get_mut((x.to_bits() >> 16) as usize)) {
            *b += 1;
        }
        if let (Some(lo), Some(hi), Some(sm), Some(sq), Some(c)) = (self.min.get_mut(cell), self.max.get_mut(cell), self.sum.get_mut(cell), self.sq.get_mut(cell), self.count.get_mut(cell)) {
            *lo = lo.min(x);
            *hi = hi.max(x);
            let y = f64::from(x);
            *sm += y;
            *sq += y * y;
            *c += 1;
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "a mean over a count of weights")]
    fn mean(&self, k: usize) -> Option<f64> {
        let c = *self.count.get(k)?;
        (c > 0).then(|| self.sum.get(k).copied().unwrap_or(0.0) / c as f64)
    }
    #[expect(clippy::cast_precision_loss, reason = "a mean over a count of weights")]
    fn rms(&self, k: usize) -> Option<f64> {
        let c = *self.count.get(k)?;
        (c > 0).then(|| (self.sq.get(k).copied().unwrap_or(0.0) / c as f64).sqrt())
    }
}

/// The cell index along one side: position `x` of `len` in `cells` cells (the largest `j` with
/// `floor(j * len / cells) <= x`). The scan below steps it without division; this is its definition.
#[cfg(test)]
fn cell_of(x: usize, len: usize, cells: usize) -> usize {
    ((x + 1) * cells - 1) / len
}
/// The first position of cell `j + 1` of `cells` over `len`.
fn edge(j: usize, len: usize, cells: usize) -> usize {
    let e = (j as u128 + 1) * len as u128 / cells as u128;
    usize::try_from(e).unwrap_or(usize::MAX)
}

/// Folds a region of a tensor into a `w x h` grid, reading at most `chunk` bytes at a time.
fn fold(t: &Src, r: Region, w: usize, h: usize, chunk: usize, hist: bool) -> Result<Grid, String> {
    let (_, cols) = t.dims();
    let mut store = match (&t.values, &t.at) {
        (Some(v), _) => Store::Mem(v),
        (None, Some((file, at))) => {
            let wd = width(&t.dtype).ok_or_else(|| format!("{}: {}: a stored type not read here", t.name, t.dtype))?;
            let fh = std::fs::File::open(file).map_err(|e| format!("{}: {e}", file.display()))?;
            Store::File { fh, base: *at, dtype: &t.dtype, width: wd, raw: Vec::new() }
        }
        (None, None) => return Err(format!("{}: {}", t.name, t.unread.as_deref().unwrap_or("its values are not held here"))),
    };
    let (rw, rh) = (r.c1 - r.c0, r.r1 - r.r0);
    let per = (chunk / store.width()).max(1);
    let mut g = Grid::new(w, h, hist);
    let mut buf = Vec::new();
    // A full-width region is one contiguous span; otherwise each row's columns are one span.
    let spans: Box<dyn Iterator<Item = (usize, usize)>> = if r.c0 == 0 && r.c1 == cols { Box::new(std::iter::once((r.r0 * cols, rh * cols))) } else { Box::new((r.r0..r.r1).map(|row| (row * cols + r.c0, rw))) };
    // The scan position: row and column inside the region, the cell row and column, and the
    // positions where the next cell row and column begin.
    let (mut row, mut col, mut cr, mut cc) = (0_usize, 0_usize, 0_usize, 0_usize);
    let (mut next_r, mut next_c) = (edge(0, rh, h), edge(0, rw, w));
    for (start, len) in spans {
        let mut k = 0;
        while k < len {
            let count = per.min(len - k);
            store.read(start + k, count, &mut buf)?;
            g.chunks += 1;
            let b = count * store.width();
            g.bytes += b as u64;
            g.largest = g.largest.max(b);
            for x in &buf {
                g.add(cr * w + cc, *x);
                col += 1;
                if col == next_c && col < rw {
                    cc += 1;
                    next_c = edge(cc, rw, w);
                }
                if col == rw {
                    col = 0;
                    cc = 0;
                    next_c = edge(0, rw, w);
                    row += 1;
                    if row == next_r && row < rh {
                        cr += 1;
                        next_r = edge(cr, rh, h);
                    }
                }
            }
            k += count;
        }
    }
    Ok(g)
}

// ---------------------------------------------------------------------------------------------
// The request.

/// A grid request: the region (each bound optional) and the grid's size.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ask {
    pub w: Option<usize>,
    pub h: Option<usize>,
    pub r0: Option<usize>,
    pub r1: Option<usize>,
    pub c0: Option<usize>,
    pub c1: Option<usize>,
    /// The region's distribution as well (`hist=1`).
    pub hist: bool,
}

/// The region and grid an ask names on a tensor, checked against its view.
fn plan(t: &Src, a: &Ask) -> Result<(Region, usize, usize), String> {
    let (rows, cols) = t.dims();
    let r = Region { r0: a.r0.unwrap_or(0), r1: a.r1.unwrap_or(rows), c0: a.c0.unwrap_or(0), c1: a.c1.unwrap_or(cols) };
    if r.r0 >= r.r1 || r.r1 > rows {
        return Err(format!("{} is {rows} x {cols}: rows {}..{} do not lie inside it", t.name, r.r0, r.r1));
    }
    if r.c0 >= r.c1 || r.c1 > cols {
        return Err(format!("{} is {rows} x {cols}: columns {}..{} do not lie inside it", t.name, r.c0, r.c1));
    }
    let w = a.w.unwrap_or(SIDE_DEFAULT).clamp(1, SIDE_MAX).min(r.c1 - r.c0);
    let h = a.h.unwrap_or(SIDE_DEFAULT).clamp(1, SIDE_MAX).min(r.r1 - r.r0);
    if w * h > CELLS_MAX {
        return Err(format!("a grid of {w} x {h} cells: at most {CELLS_MAX} cells"));
    }
    Ok((r, w, h))
}

/// The editor's path for a region (`pedit::parse_path`): `name[r0:r1, c0:c1]` for a matrix,
/// `name[c0:c1]` for a vector, and the flat range of the whole rows for three or more dimensions.
fn path_of(t: &Src, r: Region) -> String {
    let (_, cols) = t.dims();
    match t.shape.len() {
        0 | 1 => format!("{}[{}:{}]", t.name, r.c0, r.c1),
        2 => format!("{}[{}:{}, {}:{}]", t.name, r.r0, r.r1, r.c0, r.c1),
        _ => format!("{}[{}:{}]", t.name, r.r0 * cols, r.r1 * cols),
    }
}

/// The grid reply of one tensor.
fn grid_view(set: &Set, t: &Src, a: &Ask, chunk: usize) -> Result<Value, String> {
    let began = std::time::Instant::now();
    let (r, w, h) = plan(t, a)?;
    let g = fold(t, r, w, h, chunk, a.hist)?;
    let (rows, cols) = t.dims();
    let (rw, rh) = (r.c1 - r.c0, r.r1 - r.r0);
    let exact = w == rw && h == rh;
    let k = w * h;
    let (mut lo, mut hi, mut sm, mut sq, mut cnt) = (f32::INFINITY, f32::NEG_INFINITY, 0.0_f64, 0.0_f64, 0_u64);
    for i in 0..k {
        lo = lo.min(g.min.get(i).copied().unwrap_or(f32::INFINITY));
        hi = hi.max(g.max.get(i).copied().unwrap_or(f32::NEG_INFINITY));
        sm += g.sum.get(i).copied().unwrap_or(0.0);
        sq += g.sq.get(i).copied().unwrap_or(0.0);
        cnt += g.count.get(i).copied().unwrap_or(0);
    }
    #[expect(clippy::cast_precision_loss, reason = "a mean over a count of weights")]
    let (mean, rms) = if cnt > 0 { (sm / cnt as f64, (sq / cnt as f64).sqrt()) } else { (0.0, 0.0) };
    let opt = |x: Option<f64>| x.map_or(Value::Null, f64v);
    let ext = |x: f32, c: u64| if c > 0 { f32v(x) } else { Value::Null };
    let count = |i: usize| g.count.get(i).copied().unwrap_or(0);
    #[expect(clippy::cast_precision_loss, reason = "weights per cell, reported as a ratio")]
    let cell = obj(vec![("rows", f64v(rh as f64 / h as f64)), ("cols", f64v(rw as f64 / w as f64)), ("weights", f64v((rw * rh) as f64 / k as f64))]);
    let mut out = vec![
        ("format", s(FORMAT)),
        ("key", s(&set.key)),
        ("tensor", s(&t.name)),
        ("layer", t.layer.map_or(Value::Null, nu)),
        ("part", s(&t.part)),
        ("shape", shape_v(&t.shape)),
        ("dtype", s(&t.dtype)),
        ("rows", nu(rows)),
        ("cols", nu(cols)),
        ("region", obj(vec![("r0", nu(r.r0)), ("r1", nu(r.r1)), ("c0", nu(r.c0)), ("c1", nu(r.c1))])),
        ("path", s(&path_of(t, r))),
        ("w", nu(w)),
        ("h", nu(h)),
        ("cell", cell),
        ("exact", Value::Bool(exact)),
        ("summary", obj(vec![("min", ext(lo, cnt)), ("max", ext(hi, cnt)), ("mean", f64v(mean)), ("rms", f64v(rms)), ("count", n(cnt)), ("nonfinite", n(g.nonfinite))])),
        ("read", obj(vec![("bytes", n(g.bytes)), ("chunks", n(g.chunks)), ("largest_chunk", nu(g.largest)), ("chunk_cap", nu(chunk)), ("ms", n(u64::try_from(began.elapsed().as_millis()).unwrap_or(u64::MAX)))])),
    ];
    if let Some(table) = &g.hist {
        #[expect(clippy::cast_precision_loss, reason = "a variance over a count of weights")]
        let std = if cnt > 0 { (sq / cnt as f64 - mean * mean).max(0.0).sqrt() } else { 0.0 };
        out.push(("hist", hist_view(table, lo, hi, cnt, g.zeros, (std, sq.sqrt()))));
    }
    if exact {
        out.push(("values", Value::Array((0..k).map(|i| ext(g.min.get(i).copied().unwrap_or(f32::NAN), count(i))).collect())));
    } else {
        out.push(("min", Value::Array((0..k).map(|i| ext(g.min.get(i).copied().unwrap_or(f32::NAN), count(i))).collect())));
        out.push(("max", Value::Array((0..k).map(|i| ext(g.max.get(i).copied().unwrap_or(f32::NAN), count(i))).collect())));
        out.push(("mean", Value::Array((0..k).map(|i| opt(g.mean(i))).collect())));
        out.push(("rms", Value::Array((0..k).map(|i| opt(g.rms(i))).collect())));
    }
    Ok(obj(out))
}

/// The value a bucket of the distribution table stands for (its bf16 pattern).
fn bucket_value(k: usize) -> f32 {
    f32::from_bits(u32::try_from(k).unwrap_or(0) << 16)
}

/// The buckets in ascending value: the negative patterns from the largest magnitude down, then the
/// positive ones up.
fn ascending() -> impl Iterator<Item = usize> {
    (BUCKETS / 2..BUCKETS).rev().chain(0..BUCKETS / 2)
}

/// The distribution of a region from its table: linear bins over `lo..=hi`, the value at each whole
/// percentile (percentile `p` is the weight of rank `floor(p * (count - 1) / 100)`, from 0), the
/// exact zeros, the standard deviation and the L2 norm.
fn hist_view(table: &[u64], lo: f32, hi: f32, cnt: u64, zeros: u64, (std, l2): (f64, f64)) -> Value {
    let mut bins = vec![0_u64; HIST_BINS];
    let finite = cnt > 0 && lo.is_finite() && hi.is_finite();
    let span = if finite { f64::from(hi) - f64::from(lo) } else { 0.0 };
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss, reason = "a bin index from a position in 0..1")]
    let bin_of = |x: f32| -> usize {
        if span <= 0.0 {
            return 0;
        }
        let at = ((f64::from(x) - f64::from(lo)) / span * HIST_BINS as f64).floor();
        (at.max(0.0) as usize).min(HIST_BINS - 1)
    };
    let mut q = Vec::with_capacity(101);
    let (mut seen, mut p) = (0_u64, 0_u64);
    for k in ascending() {
        let c = table.get(k).copied().unwrap_or(0);
        if c == 0 {
            continue;
        }
        let x = if finite { bucket_value(k).clamp(lo, hi) } else { bucket_value(k) };
        if let Some(b) = bins.get_mut(bin_of(x)) {
            *b += c;
        }
        seen += c;
        while p <= 100 && p * (cnt - 1) / 100 < seen {
            q.push(f32v(x));
            p += 1;
        }
    }
    obj(vec![
        ("bins", Value::Array(bins.into_iter().map(n).collect())),
        ("lo", if finite { f32v(lo) } else { Value::Null }),
        ("hi", if finite { f32v(hi) } else { Value::Null }),
        ("q", Value::Array(q)),
        ("zeros", n(zeros)),
        ("count", n(cnt)),
        ("std", f64v(std)),
        ("l2", f64v(l2)),
        ("precision", s("bf16")),
    ])
}

/// The overview of one opened object: the set without `tensor`, a tensor's grid with it.
/// # Errors
/// An unknown object or tensor, a region outside the tensor, a grid above `CELLS_MAX`, values not
/// held here, or a read failure.
pub fn overview(home: &Path, key: &str, tensor: Option<&str>, a: &Ask) -> Result<Value, String> {
    let set = set_of(home, key)?;
    match tensor {
        None => Ok(set_view(&set)),
        Some(name) => {
            let t = set.tensors.iter().find(|t| t.name == name).ok_or_else(|| format!("{key} has no tensor {name}"))?;
            grid_view(&set, t, a, CHUNK)
        }
    }
}

/// `GET /api/parameters/overview?key=<key>[&tensor=<name>&w=&h=&r0=&r1=&c0=&c1=&hist=1]`: the set,
/// or one tensor's grid (with the region's distribution for `hist=1`); a failure is `{"error": ...}`.
pub fn view(home: &Path, query: &str) -> Value {
    let arg = |k: &str| crate::project::query_arg(query, k).filter(|x| !x.is_empty());
    let Some(key) = arg("key") else { return fail("key: the opened object (a cartridge id@version, filter:<role>/<job>, an adapter, a unit or open:<model>)") };
    match ask_of(query) {
        Ok(a) => overview(home, &key, arg("tensor").as_deref(), &a).unwrap_or_else(|e| fail(&e)),
        Err(e) => fail(&e),
    }
}

/// The grid fields of a query, each a whole number when given.
fn ask_of(query: &str) -> Result<Ask, String> {
    let int = |k: &str| -> Result<Option<usize>, String> { crate::project::query_arg(query, k).filter(|x| !x.is_empty()).map(|x| x.trim().parse::<usize>().map_err(|_| format!("{k}: a whole number"))).transpose() };
    let hist = crate::project::query_arg(query, "hist").is_some_and(|x| x == "1" || x == "true");
    Ok(Ask { w: int("w")?, h: int("h")?, r0: int("r0")?, r1: int("r1")?, c0: int("c0")?, c1: int("c1")?, hist })
}

/// `library parameters <key> --overview [--tensor <name>] [--w N --h N] [--region r0:r1,c0:c1]
/// [--hist]`: the same JSON as `GET /api/parameters/overview` for the same fields (`view`). `args`
/// are the words after `library`. A failure the route answers with `{"error": ...}` is the error.
/// # Errors
/// No key, a `--region` that does not read as `r0:r1,c0:c1`, or the route's own error.
pub fn cli(home: &Path, args: &[String]) -> Result<Value, String> {
    const USE: &str = "usage: neuromorph-nano library parameters <key> --overview [--tensor <name>] [--w N --h N] [--region r0:r1,c0:c1] [--hist]";
    let flag = |name: &str| args.windows(2).find(|w| w.first().is_some_and(|a| a == name)).and_then(|w| w.get(1)).map(String::as_str);
    // Everything but the unreserved characters is percent-encoded, as a browser sends it.
    let enc = |x: &str| {
        x.bytes()
            .map(|b| if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) { char::from(b).to_string() } else { format!("%{b:02X}") })
            .collect::<String>()
    };
    let key = args.get(2).filter(|a| !a.starts_with("--")).ok_or(USE)?;
    let mut q = vec![format!("key={}", enc(key))];
    if let Some(t) = flag("--tensor") {
        q.push(format!("tensor={}", enc(t)));
    }
    for (f, k) in [("--w", "w"), ("--h", "h")] {
        if let Some(v) = flag(f) {
            q.push(format!("{k}={}", enc(v)));
        }
    }
    if let Some(r) = flag("--region") {
        let bad = || format!("--region: rows and columns as r0:r1,c0:c1, not {r}");
        let (rows, cols) = r.split_once(',').ok_or_else(bad)?;
        let ((r0, r1), (c0, c1)) = (rows.split_once(':').ok_or_else(bad)?, cols.split_once(':').ok_or_else(bad)?);
        for (k, v) in [("r0", r0), ("r1", r1), ("c0", c0), ("c1", c1)] {
            q.push(format!("{k}={}", enc(v.trim())));
        }
    }
    if args.iter().any(|a| a == "--hist") {
        q.push("hist=1".to_owned());
    }
    let v = view(home, &q.join("&"));
    match v.get("error").and_then(Value::string) {
        Ok(e) => Err(e.to_owned()),
        Err(_) => Ok(v),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Dir(PathBuf);
    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn dir(name: &str) -> Dir {
        let d = std::env::temp_dir().join(format!("nano-poverview-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Dir(d)
    }
    fn bf16(x: f32) -> [u8; 2] {
        let b = (x.to_bits() >> 16) as u16;
        b.to_le_bytes()
    }
    /// A safetensors file of the named tensors (name, dtype, shape, values).
    fn write_safetensors(path: &Path, ts: &[(&str, &str, Vec<usize>, Vec<f32>)]) {
        let mut head = String::from("{");
        let mut body = Vec::new();
        for (i, (name, dtype, shape, vals)) in ts.iter().enumerate() {
            let a = body.len();
            for v in vals {
                match *dtype {
                    "F32" => body.extend_from_slice(&v.to_le_bytes()),
                    "BF16" => body.extend_from_slice(&bf16(*v)),
                    _ => panic!("dtype"),
                }
            }
            let sh: Vec<String> = shape.iter().map(ToString::to_string).collect();
            if i > 0 {
                head.push(',');
            }
            head.push_str(&format!("\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{a},{}]}}", sh.join(","), body.len()));
        }
        head.push('}');
        let mut out = (head.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(head.as_bytes());
        out.extend_from_slice(&body);
        std::fs::write(path, out).unwrap();
    }
    /// A deterministic matrix with signed values of varied size.
    fn matrix(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols).map(|i| ((i * 7919 % 1013) as f32 - 506.0) / 64.0).collect()
    }
    fn fixture(d: &Path) -> Set {
        let files = d.join("files");
        std::fs::create_dir_all(&files).unwrap();
        write_safetensors(
            &files.join("model.safetensors"),
            &[
                ("model.layers.1.mlp.up_proj.weight", "F32", vec![37, 23], matrix(37, 23)),
                ("model.embed_tokens.weight", "BF16", vec![50, 8], matrix(50, 8)),
                ("model.norm.weight", "F32", vec![23], matrix(1, 23)),
                ("model.layers.0.input_layernorm.weight", "F32", vec![23], matrix(1, 23)),
            ],
        );
        open_index("open:test", "test", &files).unwrap()
    }
    fn get<'a>(v: &'a Value, k: &str) -> &'a Value {
        v.get(k).unwrap()
    }
    fn fl(v: &Value) -> f64 {
        match v {
            Value::Number(t) => t.parse().unwrap(),
            other => panic!("not a number: {other:?}"),
        }
    }
    fn arr(v: &Value, k: &str) -> Vec<f64> {
        get(v, k).array().unwrap().iter().map(fl).collect()
    }
    fn ask(w: usize, h: usize) -> Ask {
        Ask { w: Some(w), h: Some(h), ..Ask::default() }
    }
    /// Brute force: the cells by their definition, from the whole matrix.
    fn brute(vals: &[f32], cols: usize, r: Region, w: usize, h: usize) -> Vec<(f32, f32, f64, f64)> {
        let (rw, rh) = (r.c1 - r.c0, r.r1 - r.r0);
        let mut out = Vec::new();
        for i in 0..h {
            for j in 0..w {
                let (ra, rb) = (r.r0 + i * rh / h, r.r0 + (i + 1) * rh / h);
                let (ca, cb) = (r.c0 + j * rw / w, r.c0 + (j + 1) * rw / w);
                let xs: Vec<f32> = (ra..rb).flat_map(|y| (ca..cb).map(move |x| (y, x))).map(|(y, x)| vals[y * cols + x]).collect();
                let lo = xs.iter().copied().fold(f32::INFINITY, f32::min);
                let hi = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mean = xs.iter().map(|x| f64::from(*x)).sum::<f64>() / xs.len() as f64;
                let rms = (xs.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>() / xs.len() as f64).sqrt();
                out.push((lo, hi, mean, rms));
            }
        }
        out
    }
    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-5 * (1.0 + b.abs())
    }

    #[test]
    fn cell_edges_match_their_definition() {
        for (len, cells) in [(23, 7), (37, 5), (10, 10), (151_669, 256), (1, 1)] {
            for x in 0..len.min(5000) {
                let j = cell_of(x, len, cells);
                assert!(j * len / cells <= x && x < edge(j, len, cells), "x {x} len {len} cells {cells}");
            }
        }
    }

    #[test]
    fn the_set_is_laid_out_without_reading_values() {
        let d = dir("set");
        let set = fixture(&d.0);
        let names: Vec<&str> = set.tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["model.embed_tokens.weight", "model.layers.0.input_layernorm.weight", "model.layers.1.mlp.up_proj.weight", "model.norm.weight"]);
        assert!(set.tensors.iter().all(|t| t.values.is_none() && t.readable()));
        let v = set_view(&set);
        assert_eq!(fl(get(&v, "count")), 4.0);
        let first = &get(&v, "tensors").array().unwrap()[0];
        assert_eq!(get(first, "part").string().unwrap(), "table");
        assert_eq!((fl(get(first, "rows")), fl(get(first, "cols"))), (50.0, 8.0));
        let norm = &get(&v, "tensors").array().unwrap()[1];
        assert_eq!((fl(get(norm, "rows")), fl(get(norm, "cols")), fl(get(norm, "layer"))), (1.0, 23.0, 0.0));
    }

    #[test]
    fn cells_equal_brute_force_min_max_mean_rms() {
        let d = dir("brute");
        let set = fixture(&d.0);
        let t = set.tensors.iter().find(|t| t.name.contains("up_proj")).unwrap();
        let vals = matrix(37, 23);
        for (w, h) in [(7, 5), (1, 1), (23, 36), (4, 37)] {
            let v = grid_view(&set, t, &ask(w, h), CHUNK).unwrap();
            assert_eq!(get(&v, "exact"), &Value::Bool(false));
            let want = brute(&vals, 23, Region { r0: 0, r1: 37, c0: 0, c1: 23 }, w, h);
            let (lo, hi, mean, rms) = (arr(&v, "min"), arr(&v, "max"), arr(&v, "mean"), arr(&v, "rms"));
            assert_eq!(lo.len(), w * h);
            for (k, (a, b, m, r)) in want.iter().enumerate() {
                assert!(close(lo[k], f64::from(*a)) && close(hi[k], f64::from(*b)), "cell {k} of {w}x{h}");
                assert!(close(mean[k], *m) && close(rms[k], *r), "cell {k} of {w}x{h}: {} vs {m}", mean[k]);
            }
        }
    }

    #[test]
    fn one_weight_per_cell_gives_exact_values() {
        let d = dir("exact");
        let set = fixture(&d.0);
        let t = set.tensors.iter().find(|t| t.name.contains("up_proj")).unwrap();
        let vals = matrix(37, 23);
        let v = grid_view(&set, t, &Ask { w: Some(5), h: Some(4), r0: Some(10), r1: Some(14), c0: Some(3), c1: Some(8), hist: false }, CHUNK).unwrap();
        assert_eq!(get(&v, "exact"), &Value::Bool(true));
        assert!(v.get("mean").is_err(), "exact replies carry values only");
        let got = arr(&v, "values");
        let want: Vec<f64> = (10..14).flat_map(|r| (3..8).map(move |c| (r, c))).map(|(r, c)| f64::from(vals[r * 23 + c])).collect();
        assert_eq!(got, want);
        assert_eq!(get(&v, "path").string().unwrap(), "model.layers.1.mlp.up_proj.weight[10:14, 3:8]");
        // BF16: the exact stored values, rounded as stored.
        let e = set.tensors.iter().find(|t| t.name.contains("embed")).unwrap();
        let v = grid_view(&set, e, &ask(8, 50), CHUNK).unwrap();
        let want: Vec<f64> = matrix(50, 8).iter().map(|x| f64::from(f32::from_bits(u32::from(u16::from_le_bytes(bf16(*x))) << 16))).collect();
        assert_eq!(arr(&v, "values"), want);
    }

    #[test]
    fn a_region_request_summarises_only_the_region() {
        let d = dir("region");
        let set = fixture(&d.0);
        let t = set.tensors.iter().find(|t| t.name.contains("up_proj")).unwrap();
        let vals = matrix(37, 23);
        let r = Region { r0: 5, r1: 30, c0: 2, c1: 19 };
        let v = grid_view(&set, t, &Ask { w: Some(4), h: Some(6), r0: Some(5), r1: Some(30), c0: Some(2), c1: Some(19), hist: false }, 64).unwrap();
        let want = brute(&vals, 23, r, 4, 6);
        let (lo, mean) = (arr(&v, "min"), arr(&v, "mean"));
        for (k, (a, _, m, _)) in want.iter().enumerate() {
            assert!(close(lo[k], f64::from(*a)) && close(mean[k], *m), "cell {k}");
        }
        assert_eq!(fl(get(get(&v, "read"), "bytes")), (25 * 17 * 4) as f64, "only the region's weights are read");
        // A vector is one row; its path is a range.
        let norm = set.tensors.iter().find(|t| t.name == "model.norm.weight").unwrap();
        let v = grid_view(&set, norm, &Ask { w: Some(3), c0: Some(4), c1: Some(10), ..Ask::default() }, CHUNK).unwrap();
        assert_eq!((fl(get(&v, "w")), fl(get(&v, "h"))), (3.0, 1.0));
        assert_eq!(get(&v, "path").string().unwrap(), "model.norm.weight[4:10]");
        // Regions outside the tensor and grids above the cap are refused.
        assert!(plan(t, &Ask { r1: Some(38), ..Ask::default() }).is_err());
        assert!(plan(t, &Ask { c0: Some(9), c1: Some(9), ..Ask::default() }).is_err());
        let big = Src { shape: vec![4096, 4096], ..t.clone() };
        assert!(plan(&big, &ask(1024, 1024)).is_err());
    }

    #[test]
    fn reads_are_chunked_and_bounded() {
        let d = dir("chunks");
        let set = fixture(&d.0);
        let t = set.tensors.iter().find(|t| t.name.contains("up_proj")).unwrap();
        let chunk = 40; // ten F32 weights
        let whole = grid_view(&set, t, &ask(7, 5), CHUNK).unwrap();
        let v = grid_view(&set, t, &ask(7, 5), chunk).unwrap();
        let read = get(&v, "read");
        assert!(fl(get(read, "largest_chunk")) <= chunk as f64);
        assert_eq!(fl(get(read, "chunks")), (37.0_f64 * 23.0 / 10.0).ceil());
        assert_eq!(fl(get(read, "bytes")), 37.0 * 23.0 * 4.0);
        assert_eq!(get(&v, "mean"), get(&whole, "mean"), "the chunk size does not change a cell");
        assert_eq!(get(&v, "min"), get(&whole, "min"));
        // A part-width region: one span per row, each chunked.
        let v = grid_view(&set, t, &Ask { w: Some(2), h: Some(2), c0: Some(1), c1: Some(22), ..Ask::default() }, chunk).unwrap();
        assert_eq!(fl(get(get(&v, "read"), "chunks")), 37.0 * 3.0);
    }

    #[test]
    fn values_held_in_memory_are_read_the_same_way() {
        let t = Src { name: "L3.o.gain".into(), layer: Some(3), part: "gain".into(), shape: vec![6], dtype: "F32".into(), values: Some(vec![1.0, -2.0, 3.0, 0.5, 0.25, -4.0]), at: None, unread: None };
        let set = Set { key: "k@1".into(), model: "m".into(), kind: "filter".into(), name: "n".into(), tensors: vec![t.clone()] };
        let v = grid_view(&set, &t, &ask(2, 1), 8).unwrap();
        assert_eq!(arr(&v, "min"), [-2.0, -4.0]);
        assert_eq!(arr(&v, "max"), [3.0, 0.5]);
        let none = Src { values: None, unread: Some("taken from the open weights".into()), ..t };
        assert!(grid_view(&set, &none, &ask(2, 1), 8).unwrap_err().contains("open weights"));
    }

    #[test]
    fn the_distribution_matches_a_sort_of_the_region() {
        let d = dir("hist");
        let set = fixture(&d.0);
        // BF16: the table is exact, so the percentiles are the sorted stored values.
        let e = set.tensors.iter().find(|t| t.name.contains("embed")).unwrap();
        let v = grid_view(&set, e, &Ask { w: Some(2), h: Some(2), hist: true, ..Ask::default() }, 64).unwrap();
        let hst = get(&v, "hist");
        let mut stored: Vec<f32> = matrix(50, 8).iter().map(|x| f32::from_bits(u32::from(u16::from_le_bytes(bf16(*x))) << 16)).collect();
        stored.sort_by(f32::total_cmp);
        let q = arr(hst, "q");
        assert_eq!(q.len(), 101);
        for (p, got) in q.iter().enumerate() {
            assert_eq!(*got, f64::from(stored[p * (stored.len() - 1) / 100]), "percentile {p}");
        }
        let bins = arr(hst, "bins");
        assert_eq!(bins.len(), HIST_BINS);
        assert_eq!(bins.iter().sum::<f64>(), 400.0);
        assert_eq!(fl(get(hst, "lo")), f64::from(stored[0]));
        assert_eq!(fl(get(hst, "hi")), f64::from(stored[399]));
        let zeros = stored.iter().filter(|x| **x == 0.0).count();
        assert_eq!(fl(get(hst, "zeros")), zeros as f64);
        let l2 = stored.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
        assert!(close(fl(get(hst, "l2")), l2));
        let mean = stored.iter().map(|x| f64::from(*x)).sum::<f64>() / 400.0;
        let std = (stored.iter().map(|x| (f64::from(*x) - mean).powi(2)).sum::<f64>() / 400.0).sqrt();
        assert!(close(fl(get(hst, "std")), std));
        // A region's distribution covers only the region; without hist=1 there is none.
        let v = grid_view(&set, e, &Ask { w: Some(1), h: Some(1), r0: Some(3), r1: Some(4), c0: Some(2), c1: Some(3), hist: true }, CHUNK).unwrap();
        assert_eq!(fl(get(get(&v, "hist"), "count")), 1.0);
        assert!(grid_view(&set, e, &ask(2, 2), CHUNK).unwrap().get("hist").is_err());
        assert!(ask_of("hist=1").unwrap().hist && !ask_of("w=3").unwrap().hist);
    }

    #[test]
    fn the_route_needs_a_key_and_whole_numbers() {
        let home = dir("route");
        assert!(view(&home.0, "").get("error").is_ok());
        assert!(view(&home.0, "key=open:x&tensor=a&w=-3").get("error").is_ok());
        assert!(view(&home.0, "key=open:..%2Fx").get("error").is_ok());
    }
}

