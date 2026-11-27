//! The selection and operation engine of parameter editing: every kind of object holding parameters
//! (`cparams`) is edited through it, so a selection and an operation mean the same thing for a job
//! filter, a socket cartridge, an adapter, a learning unit or preference counts.
//!
//! A selection names tensors (all of them, one by name, the tensors at a layer, at a target node,
//! or of one part such as `gain`) and, inside each, elements: a flat range, one index, a block of
//! rows and columns of a two-dimensional tensor, one attention head (`head` with `head_dim`: rows
//! of a matrix, elements of a vector), or one channel (a column of a matrix, an element of a
//! vector). A path gives the same in one string: `L3` (a layer), `L3.o.gain` (a tensor),
//! `L3.o.gain[4]`, `L3.o.gain[0:16]`, `down[2:4, 0:8]` (rows, columns), `*` (everything).
//!
//! Operations on the selected elements: set, add, multiply, scale toward the identity (`identity +
//! s * (x - identity)`: a gain toward 1, everything else toward 0), clamp, zero, identity (the value
//! that changes nothing), negate, paste (values in selection order), copy from another object or a
//! parameter map at the same place, interpolate toward one (`x + t * (source - x)`), and blend two
//! sources (`a + t * (b - a)`: any two versions, `source` and `source_b`). Structure, where the form
//! allows: drop the selected tensors (a list of points drops whole points), add a point at a node
//! (gain 1, bias 0), and reduce the rank of a low-rank pair to its best approximation of that rank
//! (the truncated singular value decomposition of `up * down`, through the small `r x r` problem:
//! the two factors' QR, then a one-sided Jacobi decomposition of `R_up * R_down^T`, in float64).
//! Every operation checks the selection against the tensor's shape,
//! every value is finite, and `diff` reports what would change before anything is written.
use crate::cparams::Tensor;
use nmjson::Value;

/// A selection inside an object.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Sel {
    pub tensor: Option<String>,
    pub layer: Option<usize>,
    pub target: Option<String>,
    pub part: Option<String>,
    /// Flat element range `[a, b)`.
    pub range: Option<(usize, usize)>,
    /// Rows `[a, b)` of a two-dimensional tensor (a range of a vector).
    pub rows: Option<(usize, usize)>,
    /// Columns `[a, b)` of a two-dimensional tensor.
    pub cols: Option<(usize, usize)>,
    /// One attention head: (head, head dimension).
    pub head: Option<(usize, usize)>,
    /// One channel: a column of a matrix, an element of a vector.
    pub channel: Option<usize>,
}

/// Where copied or interpolated values come from: another object by key (another version, another
/// cartridge), or a parameter map (`<role>/<file>`) that holds values at the same place.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    Object(String),
    Map(String),
}

/// One operation.
#[derive(Clone, Debug, PartialEq)]
pub enum OpKind {
    Set(f32),
    Add(f32),
    Mul(f32),
    Scale(f32),
    Clamp(f32, f32),
    Zero,
    Identity,
    Negate,
    Paste(Vec<f32>),
    Copy(Source),
    Interp(Source, f32),
    /// `a + t * (b - a)` with `a` from the first source and `b` from the second.
    Blend(Source, Source, f32),
    Drop,
    AddPoint { target: String, width: usize },
    Rank(usize),
}

/// One operation on one selection.
#[derive(Clone, Debug, PartialEq)]
pub struct Op {
    pub sel: Sel,
    pub kind: OpKind,
}
impl Op {
    /// The operation in a short phrase.
    #[must_use]
    pub fn phrase(&self) -> String {
        let k = match &self.kind {
            OpKind::Set(x) => format!("set {x}"),
            OpKind::Add(x) => format!("add {x}"),
            OpKind::Mul(x) => format!("multiply by {x}"),
            OpKind::Scale(x) => format!("strength {x}"),
            OpKind::Clamp(a, b) => format!("clamp to {a}..{b}"),
            OpKind::Zero => "zero".to_owned(),
            OpKind::Identity => "identity".to_owned(),
            OpKind::Negate => "negate".to_owned(),
            OpKind::Paste(v) => format!("paste {} values", v.len()),
            OpKind::Copy(s) => format!("copy from {}", source_name(s)),
            OpKind::Interp(s, t) => format!("interpolate {t} toward {}", source_name(s)),
            OpKind::Blend(a, b, t) => format!("blend {t} from {} to {}", source_name(a), source_name(b)),
            OpKind::Drop => "drop".to_owned(),
            OpKind::AddPoint { target, width } => format!("add point {target} ({width})"),
            OpKind::Rank(r) => format!("rank {r}"),
        };
        format!("{k} on {}", sel_name(&self.sel))
    }
}

fn source_name(s: &Source) -> String {
    match s {
        Source::Object(k) => k.clone(),
        Source::Map(m) => format!("map {m}"),
    }
}

/// A selection as a path-like phrase.
#[must_use]
pub fn sel_name(s: &Sel) -> String {
    let mut out = s.tensor.clone().or_else(|| s.target.clone()).or_else(|| s.layer.map(|l| format!("L{l}"))).unwrap_or_else(|| "*".to_owned());
    if let Some(p) = &s.part {
        out.push_str(&format!(" ({p})"));
    }
    let r = |x: (usize, usize)| format!("{}:{}", x.0, x.1);
    match (s.range, s.rows, s.cols, s.head, s.channel) {
        (Some(x), ..) => out.push_str(&format!("[{}]", r(x))),
        (_, Some(a), Some(b), ..) => out.push_str(&format!("[{}, {}]", r(a), r(b))),
        (_, Some(a), None, ..) => out.push_str(&format!("[{}, :]", r(a))),
        (_, None, Some(b), ..) => out.push_str(&format!("[:, {}]", r(b))),
        (.., Some((h, d)), _) => out.push_str(&format!(" head {h} of {d}")),
        (.., Some(c)) => out.push_str(&format!(" channel {c}")),
        _ => {}
    }
    out
}

fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Number(t) => t.parse().ok(),
        Value::String(t) => t.trim().parse().ok(),
        _ => None,
    }
}
fn finite(x: f64, what: &str) -> Result<f32, String> {
    #[expect(clippy::cast_possible_truncation, reason = "parameters are held in f32")]
    let y = x as f32;
    if y.is_finite() { Ok(y) } else { Err(format!("{what}: a finite number")) }
}
fn whole(v: &Value, what: &str) -> Result<usize, String> {
    num(v).filter(|x| *x >= 0.0 && x.fract() == 0.0 && *x < 1e15).map(|x| {
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a whole non-negative number below 1e15")]
        let u = x as usize;
        u
    }).ok_or_else(|| format!("{what}: a whole number"))
}
fn span(v: &Value, what: &str) -> Result<(usize, usize), String> {
    let a = v.array().map_err(|_| format!("{what}: [from, to)"))?;
    match a {
        [x, y] => {
            let (x, y) = (whole(x, what)?, whole(y, what)?);
            if x < y { Ok((x, y)) } else { Err(format!("{what}: from before to")) }
        }
        _ => Err(format!("{what}: [from, to)")),
    }
}

fn slice_of(t: &str, what: &str) -> Result<(usize, usize), String> {
    let t = t.trim();
    match t.split_once(':') {
        Some((a, b)) => {
            let a = if a.trim().is_empty() { 0 } else { a.trim().parse().map_err(|_| format!("{what}: {t}"))? };
            let b = b.trim().parse::<usize>().map_err(|_| format!("{what}: {t} (give the end)"))?;
            if a < b { Ok((a, b)) } else { Err(format!("{what}: {t}")) }
        }
        None => {
            let i = t.parse::<usize>().map_err(|_| format!("{what}: {t}"))?;
            Ok((i, i + 1))
        }
    }
}

/// Reads a path (`*`, `L3`, `L3.o.gain`, `name[i]`, `name[a:b]`, `name[r0:r1, c0:c1]`).
/// # Errors
/// A path that does not read.
pub fn parse_path(p: &str) -> Result<Sel, String> {
    let p = p.trim();
    let (name, idx) = match p.split_once('[') {
        Some((n, rest)) => (n.trim(), Some(rest.strip_suffix(']').ok_or_else(|| format!("path {p}: close the brackets"))?)),
        None => (p, None),
    };
    let mut sel = Sel::default();
    if name != "*" && !name.is_empty() {
        match name.strip_prefix('L').and_then(|d| d.parse::<usize>().ok()) {
            Some(l) => sel.layer = Some(l),
            None => sel.tensor = Some(name.to_owned()),
        }
    }
    if let Some(idx) = idx {
        match idx.split_once(',') {
            Some((r, c)) => {
                if r.trim() != ":" {
                    sel.rows = Some(slice_of(r, "rows")?);
                }
                if c.trim() != ":" {
                    sel.cols = Some(slice_of(c, "columns")?);
                }
            }
            None => sel.range = Some(slice_of(idx, "range")?),
        }
    }
    Ok(sel)
}

/// Reads a selection from `selection` (an object of `tensor`, `layer`, `target`, `part`, `range`,
/// `rows`, `cols`, `head` with `head_dim`, `channel`) or `path`; none selects everything.
/// # Errors
/// Fields that do not read.
pub fn parse_sel(v: &Value) -> Result<Sel, String> {
    let mut sel = match v.get("path").and_then(Value::string) {
        Ok(p) if !p.is_empty() => parse_path(p)?,
        _ => Sel::default(),
    };
    let Ok(Value::Object(_)) = v.get("selection") else { return Ok(sel) };
    let o = v.get("selection").map_err(|e| e.to_string())?;
    let st = |k: &str| o.get(k).and_then(Value::string).ok().filter(|x| !x.is_empty()).map(str::to_owned);
    if let Some(p) = st("path") {
        sel = parse_path(&p)?;
    }
    sel.tensor = st("tensor").or(sel.tensor);
    sel.target = st("target").or(sel.target);
    sel.part = st("part").or(sel.part);
    if let Ok(x) = o.get("layer") {
        sel.layer = Some(whole(x, "layer").or_else(|_| x.string().ok().and_then(|t| t.trim_start_matches('L').parse().ok()).ok_or("layer: a number"))?);
    }
    if let Ok(x) = o.get("range") {
        sel.range = Some(span(x, "range")?);
    }
    if let Ok(x) = o.get("index") {
        let i = whole(x, "index")?;
        sel.range = Some((i, i + 1));
    }
    if let Ok(x) = o.get("rows") {
        sel.rows = Some(span(x, "rows")?);
    }
    if let Ok(x) = o.get("cols") {
        sel.cols = Some(span(x, "cols")?);
    }
    if let Ok(x) = o.get("head") {
        let d = whole(o.get("head_dim").map_err(|_| "head: give head_dim too")?, "head_dim")?;
        if d == 0 {
            return Err("head_dim: more than zero".to_owned());
        }
        sel.head = Some((whole(x, "head")?, d));
    }
    if let Ok(x) = o.get("channel") {
        sel.channel = Some(whole(x, "channel")?);
    }
    Ok(sel)
}

fn source_of(v: &Value) -> Result<Source, String> {
    source_at(v, "source")
}

fn source_at(v: &Value, field: &str) -> Result<Source, String> {
    let src = v.get(field).map_err(|_| format!("{field}: {{key}} of another object, or {{map: <role>/<file>}}"))?;
    if let Ok(k) = src.string() {
        return Ok(Source::Object(k.to_owned()));
    }
    if let Ok(m) = src.get("map").and_then(Value::string) {
        return Ok(Source::Map(m.to_owned()));
    }
    src.get("key").and_then(Value::string).map(|k| Source::Object(k.to_owned())).map_err(|_| format!("{field}: {{key}} of another object, or {{map: <role>/<file>}}"))
}

/// Reads one operation (`op` and its fields: `value`, `lo` and `hi`, `values`, `source`, `t`,
/// `target` and `width`, `rank`) with its selection.
/// # Errors
/// An unknown operation or fields that do not read.
pub fn parse_op(v: &Value) -> Result<Op, String> {
    let op = v.get("op").and_then(Value::string).map_err(|_| "op: set, add, mul, scale, clamp, zero, identity, negate, paste, copy, interpolate, drop, add_point or rank".to_owned())?;
    let val = |k: &str| v.get(k).ok().and_then(num).ok_or_else(|| format!("{op}: give {k}")).and_then(|x| finite(x, k));
    let kind = match op {
        "set" => OpKind::Set(val("value")?),
        "add" => OpKind::Add(val("value")?),
        "mul" | "multiply" => OpKind::Mul(val("value")?),
        "scale" | "strength" => {
            let s = val("value")?;
            if !(0.0..=crate::cparams::SCALE_MAX).contains(&s) {
                return Err(format!("scale: from 0 to {}", crate::cparams::SCALE_MAX));
            }
            OpKind::Scale(s)
        }
        "clamp" => {
            let (lo, hi) = (val("lo")?, val("hi")?);
            if lo > hi {
                return Err("clamp: lo at most hi".to_owned());
            }
            OpKind::Clamp(lo, hi)
        }
        "zero" => OpKind::Zero,
        "identity" => OpKind::Identity,
        "negate" => OpKind::Negate,
        "paste" => {
            let a = v.get("values").and_then(Value::array).map_err(|_| "paste: give values (a list)".to_owned())?;
            OpKind::Paste(a.iter().map(|x| num(x).ok_or_else(|| "paste: numbers".to_owned()).and_then(|y| finite(y, "values"))).collect::<Result<_, _>>()?)
        }
        "copy" => OpKind::Copy(source_of(v)?),
        "interpolate" | "interp" => {
            let t = val("t")?;
            if !(0.0..=1.0).contains(&t) {
                return Err("interpolate: t from 0 to 1".to_owned());
            }
            // Between any two versions when `source_b` is given; else toward one from the values here.
            if v.get("source_b").is_ok() { OpKind::Blend(source_of(v)?, source_at(v, "source_b")?, t) } else { OpKind::Interp(source_of(v)?, t) }
        }
        "drop" => OpKind::Drop,
        "add_point" => OpKind::AddPoint {
            target: v.get("target").and_then(Value::string).map_err(|_| "add_point: give target (a node id such as L12.o)".to_owned())?.to_owned(),
            width: whole(v.get("width").map_err(|_| "add_point: give width".to_owned())?, "width")?,
        },
        "rank" => OpKind::Rank(whole(v.get("rank").or_else(|_| v.get("value")).map_err(|_| "rank: give rank".to_owned())?, "rank")?),
        o => return Err(format!("op {o}: set, add, mul, scale, clamp, zero, identity, negate, paste, copy, interpolate, drop, add_point or rank")),
    };
    Ok(Op { sel: parse_sel(v)?, kind })
}

/// Whether a tensor is in a selection's tensors.
pub fn picks(s: &Sel, t: &Tensor) -> bool {
    s.tensor.as_ref().is_none_or(|n| *n == t.name) && s.layer.is_none_or(|l| t.layer == Some(l)) && s.target.as_ref().is_none_or(|x| *x == t.target) && s.part.as_ref().is_none_or(|p| *p == t.part)
}

fn within(r: (usize, usize), n: usize, what: &str, t: &Tensor) -> Result<(usize, usize), String> {
    if r.1 > n {
        return Err(format!("shape check: {} is {:?}; {what} {}:{} goes past {n}", t.name, t.shape, r.0, r.1));
    }
    Ok(r)
}

/// The flat indices a selection names inside one tensor, in order.
/// # Errors
/// A selection outside the tensor's shape, or one that needs two dimensions on a vector.
pub fn indices(s: &Sel, t: &Tensor) -> Result<Vec<usize>, String> {
    let size: usize = t.shape.iter().product();
    let (r, c) = match t.shape.as_slice() {
        [r, c] => (*r, *c),
        _ => (size, 1),
    };
    let two = t.shape.len() == 2;
    if let Some(x) = s.range {
        let (a, b) = within(x, size, "range", t)?;
        return Ok((a..b).collect());
    }
    if !two && s.cols.is_some() {
        return Err(format!("{} is a vector {:?}: select by range, rows or channel", t.name, t.shape));
    }
    let mut rows = s.rows.map_or(Ok((0, r)), |x| within(x, r, "rows", t))?;
    let mut cols = s.cols.map_or(Ok((0, c)), |x| within(x, c, "columns", t))?;
    if let Some((h, d)) = s.head {
        rows = within((h * d, (h + 1) * d), r, "head", t)?;
    }
    if let Some(ch) = s.channel {
        if two {
            cols = within((ch, ch + 1), c, "channel", t)?;
        } else {
            rows = within((ch, ch + 1), r, "channel", t)?;
        }
    }
    Ok((rows.0..rows.1).flat_map(|i| (cols.0..cols.1).map(move |j| i * c + j)).collect())
}

/// Values of a source at a tensor's place: (tensor name, its shape) to values.
pub type SourceFn<'a> = &'a dyn Fn(&Source, &str, &[usize]) -> Result<Vec<f32>, String>;

/// Applies the operations in order to a copy of `ts` (the source tensors are untouched). `drops`:
/// whether the form is a list of points (dropping a tensor drops its whole point).
/// # Errors
/// A selection that picks nothing or lies outside a shape, values that are not read here or too many
/// to edit, a paste of the wrong length, a source without values at the same place, or a structural
/// change the form does not allow.
pub fn run(ts: &[Tensor], ops: &[Op], drops: bool, small: usize, source: SourceFn) -> Result<Vec<Tensor>, String> {
    let mut ts = ts.to_vec();
    for op in ops {
        match &op.kind {
            OpKind::Drop => {
                if !drops {
                    return Err("this form holds a fixed set of tensors: drop is for a list of points (zero or identity them instead)".to_owned());
                }
                let targets: Vec<String> = ts.iter().filter(|t| picks(&op.sel, t)).map(|t| t.target.clone()).collect();
                if targets.is_empty() {
                    return Err(format!("{} selects nothing", sel_name(&op.sel)));
                }
                ts.retain(|t| !targets.contains(&t.target));
                if ts.is_empty() {
                    return Err("everything dropped: the new version would hold nothing; take it out of its stack or plug it out instead".to_owned());
                }
            }
            OpKind::AddPoint { target, width } => {
                if !drops {
                    return Err("points are added to a list of points (a socket cartridge or a job filter)".to_owned());
                }
                if ts.iter().any(|t| t.target == *target) {
                    return Err(format!("{target} is already a point"));
                }
                if *width == 0 || *width > small {
                    return Err(format!("width: 1 to {small}"));
                }
                let dtype = ts.first().map_or_else(|| "F32".to_owned(), |t| t.dtype.clone());
                let [g, b] = crate::cparams::point_tensors(target, vec![1.0; *width], vec![0.0; *width], &dtype);
                ts.push(g);
                ts.push(b);
            }
            OpKind::Rank(r) => {
                let pos = |n: &str| ts.iter().position(|t| t.name == n);
                let (Some(d), Some(u)) = (pos("down"), pos("up")) else { return Err("rank: this object has no low-rank pair (down, up)".to_owned()) };
                let (Some(&[dr, dc]), Some(&[ur, uc])) = (ts.get(d).map(|t| t.shape.as_slice()), ts.get(u).map(|t| t.shape.as_slice())) else { return Err("rank: down and up are matrices".to_owned()) };
                if dr != uc {
                    return Err(format!("rank: down has {dr} rows, up {uc} columns"));
                }
                if *r == 0 || *r > dr {
                    return Err(format!("rank: 1 to {dr} (the best approximation of that rank is kept)"));
                }
                let (Some(dv), Some(uv)) = (ts.get(d).and_then(|t| t.values.clone()), ts.get(u).and_then(|t| t.values.clone())) else { return Err("rank: the values of down and up are not held here".to_owned()) };
                let (nd, nu) = svd_rank(&dv, &uv, dr, dc, ur, *r);
                if let Some(t) = ts.get_mut(d) {
                    t.values = Some(nd);
                    t.shape = vec![*r, dc];
                    t.bytes = t.bytes * *r as u64 / dr.max(1) as u64;
                }
                if let Some(t) = ts.get_mut(u) {
                    t.values = Some(nu);
                    t.shape = vec![ur, *r];
                    t.bytes = t.bytes * *r as u64 / uc.max(1) as u64;
                }
            }
            kind => {
                let mut hit = false;
                for t in ts.iter_mut().filter(|t| picks(&op.sel, t)) {
                    // A strength or a layer scale moves only the tensors that carry the change.
                    if matches!(kind, OpKind::Scale(_)) && !t.scales && op.sel.tensor.is_none() {
                        continue;
                    }
                    let idx = indices(&op.sel, t)?;
                    if idx.is_empty() {
                        continue;
                    }
                    let size: usize = t.shape.iter().product();
                    if size > small {
                        return Err(format!("{} holds {size} values; values are edited here up to {small}", t.name));
                    }
                    let name = t.name.clone();
                    let shape = t.shape.clone();
                    let id = t.identity;
                    let vals = t.values.as_mut().ok_or_else(|| format!("{name}: its values are not held here"))?;
                    let src = match kind {
                        OpKind::Copy(s) | OpKind::Interp(s, _) => {
                            let sv = source(s, &name, &shape)?;
                            if sv.len() != vals.len() {
                                return Err(format!("shape check: {} at {name} holds {} values, {name} {}", source_name(s), sv.len(), vals.len()));
                            }
                            Some(sv)
                        }
                        OpKind::Blend(a, b, tt) => {
                            let (sa, sb) = (source(a, &name, &shape)?, source(b, &name, &shape)?);
                            if sa.len() != vals.len() || sb.len() != vals.len() {
                                return Err(format!("shape check: {} and {} at {name} hold {} and {} values, {name} {}", source_name(a), source_name(b), sa.len(), sb.len(), vals.len()));
                            }
                            Some(sa.iter().zip(&sb).map(|(x, y)| x + tt * (y - x)).collect())
                        }
                        _ => None,
                    };
                    if let OpKind::Paste(p) = kind
                        && p.len() != idx.len()
                    {
                        return Err(format!("shape check: {} selects {} values of {name}; {} pasted", sel_name(&op.sel), idx.len(), p.len()));
                    }
                    for (k, i) in idx.iter().enumerate() {
                        let Some(x) = vals.get_mut(*i) else { continue };
                        let from = src.as_ref().and_then(|v| v.get(*i).copied()).unwrap_or(*x);
                        *x = match kind {
                            OpKind::Set(v) => *v,
                            OpKind::Add(v) => *x + v,
                            OpKind::Mul(v) => *x * v,
                            OpKind::Scale(v) => id + v * (*x - id),
                            OpKind::Clamp(a, b) => x.clamp(*a, *b),
                            OpKind::Zero => 0.0,
                            OpKind::Identity => id,
                            OpKind::Negate => -*x,
                            OpKind::Paste(p) => p.get(k).copied().unwrap_or(*x),
                            OpKind::Copy(_) | OpKind::Blend(..) => from,
                            OpKind::Interp(_, tt) => *x + tt * (from - *x),
                            _ => *x,
                        };
                        if !x.is_finite() {
                            return Err(format!("{name}[{i}]: the result is not a finite number"));
                        }
                    }
                    hit = true;
                }
                if !hit {
                    return Err(format!("{} selects no editable values", sel_name(&op.sel)));
                }
            }
        }
    }
    Ok(ts)
}

/// Gram-Schmidt QR of a `rows x cols` matrix (row-major), columns kept in order: `Q` (`rows x
/// cols`, orthonormal columns; a column with no independent part is zero) and `R` (`cols x cols`,
/// upper triangular), so `A = Q R`.
fn qr(a: &[f64], rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
    let mut q = a.to_vec();
    let mut r = vec![0.0_f64; cols * cols];
    let at = |m: &[f64], i: usize, j: usize| m.get(i * cols + j).copied().unwrap_or(0.0);
    for j in 0..cols {
        // Twice for orthogonality in float64 (re-orthogonalization).
        for _ in 0..2 {
            for p in 0..j {
                let dot: f64 = (0..rows).map(|i| at(&q, i, p) * at(&q, i, j)).sum();
                if let Some(x) = r.get_mut(p * cols + j) {
                    *x += dot;
                }
                for i in 0..rows {
                    let v = at(&q, i, j) - dot * at(&q, i, p);
                    if let Some(x) = q.get_mut(i * cols + j) {
                        *x = v;
                    }
                }
            }
        }
        let norm = (0..rows).map(|i| at(&q, i, j).powi(2)).sum::<f64>().sqrt();
        if let Some(x) = r.get_mut(j * cols + j) {
            *x = norm;
        }
        for i in 0..rows {
            let v = if norm > 1e-300 { at(&q, i, j) / norm } else { 0.0 };
            if let Some(x) = q.get_mut(i * cols + j) {
                *x = v;
            }
        }
    }
    (q, r)
}

/// One-sided Jacobi decomposition of a square `n x n` matrix (row-major): `(U, s, V)` with
/// `A = U diag(s) V^T`, singular values in decreasing order.
fn jacobi_svd(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut u = a.to_vec();
    let mut v: Vec<f64> = (0..n * n).map(|i| if i / n == i % n { 1.0 } else { 0.0 }).collect();
    let col = |m: &[f64], j: usize| -> Vec<f64> { (0..n).map(|i| m.get(i * n + j).copied().unwrap_or(0.0)).collect() };
    for _sweep in 0..60 {
        let mut off = 0.0_f64;
        for p in 0..n {
            for q in p + 1..n {
                let (cp, cq) = (col(&u, p), col(&u, q));
                let alpha: f64 = cp.iter().map(|x| x * x).sum();
                let beta: f64 = cq.iter().map(|x| x * x).sum();
                let gamma: f64 = cp.iter().zip(&cq).map(|(x, y)| x * y).sum();
                if gamma.abs() <= 1e-15 * (alpha * beta).sqrt() {
                    continue;
                }
                off = off.max(gamma.abs() / (alpha * beta).sqrt().max(1e-300));
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let sn = c * t;
                for m in [&mut u, &mut v] {
                    for i in 0..n {
                        let (x, y) = (m.get(i * n + p).copied().unwrap_or(0.0), m.get(i * n + q).copied().unwrap_or(0.0));
                        if let Some(z) = m.get_mut(i * n + p) {
                            *z = c * x - sn * y;
                        }
                        if let Some(z) = m.get_mut(i * n + q) {
                            *z = sn * x + c * y;
                        }
                    }
                }
            }
        }
        if off < 1e-14 {
            break;
        }
    }
    let mut s: Vec<(f64, usize)> = (0..n).map(|j| (col(&u, j).iter().map(|x| x * x).sum::<f64>().sqrt(), j)).collect();
    s.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut uo = vec![0.0_f64; n * n];
    let mut vo = vec![0.0_f64; n * n];
    for (k, (sv, j)) in s.iter().enumerate() {
        for i in 0..n {
            if let Some(z) = uo.get_mut(i * n + k) {
                *z = if *sv > 1e-300 { u.get(i * n + j).copied().unwrap_or(0.0) / sv } else { 0.0 };
            }
            if let Some(z) = vo.get_mut(i * n + k) {
                *z = v.get(i * n + j).copied().unwrap_or(0.0);
            }
        }
    }
    (uo, s.iter().map(|x| x.0).collect(), vo)
}

/// The best rank-`r` approximation of `up * down` (`down` `r0 x k`, `up` `n x r0`, row-major) as a
/// new pair (`down` `r x k`, `up` `n x r`), each factor carrying the square root of the singular
/// values: QR of `up` and of `down^T`, the singular value decomposition of the small `r0 x r0`
/// product of their `R` factors, then the first `r` components.
#[must_use]
pub fn svd_rank(down: &[f32], up: &[f32], r0: usize, k: usize, n: usize, r: usize) -> (Vec<f32>, Vec<f32>) {
    let up64: Vec<f64> = up.iter().map(|x| f64::from(*x)).collect();
    // down^T as k x r0.
    let dt: Vec<f64> = (0..k).flat_map(|i| (0..r0).map(move |j| (i, j))).map(|(i, j)| f64::from(down.get(j * k + i).copied().unwrap_or(0.0))).collect();
    let (q1, r1) = qr(&up64, n, r0);
    let (q2, r2) = qr(&dt, k, r0);
    // M = R1 * R2^T.
    let m: Vec<f64> = (0..r0 * r0).map(|x| (0..r0).map(|t| r1.get((x / r0) * r0 + t).copied().unwrap_or(0.0) * r2.get((x % r0) * r0 + t).copied().unwrap_or(0.0)).sum()).collect();
    let (um, sv, vm) = jacobi_svd(&m, r0);
    #[expect(clippy::cast_possible_truncation, reason = "parameters are held in f32")]
    let narrow = |x: f64| x as f32;
    // up' = Q1 * Um[:, :r] * sqrt(s), n x r.
    let nu: Vec<f32> = (0..n * r)
        .map(|x| {
            let (i, c) = (x / r, x % r);
            let root = sv.get(c).copied().unwrap_or(0.0).sqrt();
            narrow(root * (0..r0).map(|t| q1.get(i * r0 + t).copied().unwrap_or(0.0) * um.get(t * r0 + c).copied().unwrap_or(0.0)).sum::<f64>())
        })
        .collect();
    // down' = sqrt(s) * Vm[:, :r]^T * Q2^T, r x k.
    let nd: Vec<f32> = (0..r * k)
        .map(|x| {
            let (c, i) = (x / k, x % k);
            let root = sv.get(c).copied().unwrap_or(0.0).sqrt();
            narrow(root * (0..r0).map(|t| vm.get(t * r0 + c).copied().unwrap_or(0.0) * q2.get(i * r0 + t).copied().unwrap_or(0.0)).sum::<f64>())
        })
        .collect();
    (nd, nu)
}

/// What an edit changes, tensor by tensor, before anything is written: changed values, old and new
/// norms, the largest change and a sample of changed elements; tensors added and dropped.
#[must_use]
pub fn diff(old: &[Tensor], new: &[Tensor]) -> Value {
    let s = |t: &str| Value::String(t.to_owned());
    let f = |x: f64| {
        #[expect(clippy::cast_possible_truncation, reason = "figures at the precision parameters are held in")]
        let y = x as f32;
        if y.is_finite() { Value::Number(format!("{y:e}")) } else { Value::Null }
    };
    let n = |x: usize| Value::Number(x.to_string());
    let mut rows = Vec::new();
    let mut changed_total = 0;
    for t in new {
        let Some(o) = old.iter().find(|o| o.name == t.name) else { continue };
        let (Some(a), Some(b)) = (&o.values, &t.values) else { continue };
        let same_shape = o.shape == t.shape;
        let mut changed = 0;
        let mut max = 0.0_f64;
        let mut sample = Vec::new();
        if same_shape {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                if x.to_bits() != y.to_bits() {
                    changed += 1;
                    max = max.max((f64::from(*y) - f64::from(*x)).abs());
                    if sample.len() < 32 {
                        sample.push(Value::Array(vec![n(i), f(f64::from(*x)), f(f64::from(*y))]));
                    }
                }
            }
        }
        if changed == 0 && same_shape {
            continue;
        }
        changed_total += changed;
        let (so, sn) = (crate::cparams::stats(a, o.identity), crate::cparams::stats(b, t.identity));
        rows.push(nmjson_obj(vec![
            ("name", s(&t.name)),
            ("old_shape", Value::Array(o.shape.iter().map(|x| n(*x)).collect())),
            ("shape", Value::Array(t.shape.iter().map(|x| n(*x)).collect())),
            ("changed", n(changed)),
            ("old_l2", f(so.l2)),
            ("new_l2", f(sn.l2)),
            ("old_effect_l2", f(so.effect_l2)),
            ("new_effect_l2", f(sn.effect_l2)),
            ("max_change", f(max)),
            ("sample", Value::Array(sample)),
        ]));
    }
    let names = |a: &[Tensor], b: &[Tensor]| Value::Array(a.iter().filter(|t| !b.iter().any(|x| x.name == t.name)).map(|t| s(&t.name)).collect());
    nmjson_obj(vec![("tensors", Value::Array(rows)), ("changed", n(changed_total)), ("added", names(new, old)), ("dropped", names(old, new))])
}

fn nmjson_obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(name: &str, shape: Vec<usize>, identity: f32) -> Tensor {
        let size: usize = shape.iter().product();
        #[expect(clippy::cast_precision_loss, reason = "test values")]
        let values = (0..size).map(|i| i as f32).collect();
        Tensor { name: name.into(), target: name.split('.').take(2).collect::<Vec<_>>().join("."), layer: crate::cparams::layer_of(name), part: name.rsplit('.').next().unwrap_or("").into(), shape, dtype: "F32".into(), identity, scales: true, values: Some(values), bytes: 0, unread: None, at: None }
    }
    fn none(_: &Source, _: &str, _: &[usize]) -> Result<Vec<f32>, String> {
        Err("no source".into())
    }
    fn op(json: &str) -> Op {
        parse_op(&nmjson::parse(json.as_bytes()).unwrap()).unwrap()
    }

    #[test]
    fn paths_read_into_selections() {
        assert_eq!(parse_path("L3").unwrap().layer, Some(3));
        let s = parse_path("down[2:4, 0:8]").unwrap();
        assert_eq!((s.tensor.as_deref(), s.rows, s.cols), (Some("down"), Some((2, 4)), Some((0, 8))));
        assert_eq!(parse_path("L3.o.gain[5]").unwrap().range, Some((5, 6)));
        assert!(parse_path("x[4:2]").is_err());
        assert_eq!(parse_path("*").unwrap(), Sel::default());
    }

    #[test]
    fn indices_follow_rows_columns_heads_and_channels() {
        let m = t("m", vec![4, 3], 0.0);
        let sel = |j: &str| parse_sel(&nmjson::parse(j.as_bytes()).unwrap()).unwrap();
        assert_eq!(indices(&sel(r#"{"selection":{"rows":[1,2]}}"#), &m).unwrap(), vec![3, 4, 5]);
        assert_eq!(indices(&sel(r#"{"selection":{"channel":2}}"#), &m).unwrap(), vec![2, 5, 8, 11]);
        assert_eq!(indices(&sel(r#"{"selection":{"head":1,"head_dim":2}}"#), &m).unwrap(), vec![6, 7, 8, 9, 10, 11]);
        assert!(indices(&sel(r#"{"selection":{"rows":[3,5]}}"#), &m).unwrap_err().contains("shape check"));
        let v = t("v", vec![5], 0.0);
        assert_eq!(indices(&sel(r#"{"selection":{"channel":4}}"#), &v).unwrap(), vec![4]);
        assert!(indices(&sel(r#"{"selection":{"cols":[0,1]}}"#), &v).is_err());
    }

    #[test]
    fn operations_change_only_the_selection() {
        let ts = vec![t("L1.o.gain", vec![4], 1.0), t("L1.o.bias", vec![4], 0.0), t("L2.o.gain", vec![2], 1.0), t("L2.o.bias", vec![2], 0.0)];
        let out = run(&ts, &[op(r#"{"op":"set","value":9,"path":"L1.o.gain[1:3]"}"#)], true, 100, &none).unwrap();
        assert_eq!(out[0].values.as_deref(), Some([0.0, 9.0, 9.0, 3.0].as_slice()));
        assert_eq!(ts[0].values.as_deref(), Some([0.0, 1.0, 2.0, 3.0].as_slice()), "the source is untouched");
        let out = run(&ts, &[op(r#"{"op":"scale","value":0,"path":"L1"}"#)], true, 100, &none).unwrap();
        assert!(out[0].values.as_ref().unwrap().iter().all(|x| *x == 1.0) && out[1].values.as_ref().unwrap().iter().all(|x| *x == 0.0), "strength 0 is the identity");
        let out = run(&ts, &[op(r#"{"op":"clamp","lo":0.5,"hi":2,"path":"L1.o.bias"}"#), op(r#"{"op":"negate","path":"L2.o.bias"}"#)], true, 100, &none).unwrap();
        assert_eq!(out[1].values.as_deref(), Some([0.5, 1.0, 2.0, 2.0].as_slice()));
        assert_eq!(out[3].values.as_deref(), Some([-0.0, -1.0].as_slice()));
        assert!(run(&ts, &[op(r#"{"op":"paste","values":[1,2],"path":"L1.o.gain"}"#)], true, 100, &none).unwrap_err().contains("shape check"));
        let out = run(&ts, &[op(r#"{"op":"drop","path":"L2"}"#)], true, 100, &none).unwrap();
        assert_eq!(out.len(), 2);
        assert!(run(&ts, &[op(r#"{"op":"drop","path":"L2"}"#)], false, 100, &none).is_err());
        let out = run(&ts, &[op(r#"{"op":"add_point","target":"L9.o","width":3}"#)], true, 100, &none).unwrap();
        assert_eq!(out.len(), 6);
        assert!(run(&ts, &[op(r#"{"op":"set","value":1,"path":"L7"}"#)], true, 100, &none).is_err(), "a selection that picks nothing");
        assert!(run(&ts, &[op(r#"{"op":"set","value":1,"path":"L1.o.gain"}"#)], true, 2, &none).unwrap_err().contains("up to 2"));
    }

    #[test]
    fn copy_and_interpolate_read_the_source_at_the_same_place() {
        let ts = vec![t("w", vec![3], 0.0)];
        let src = |_: &Source, name: &str, shape: &[usize]| -> Result<Vec<f32>, String> {
            assert_eq!((name, shape), ("w", [3].as_slice()));
            Ok(vec![10.0, 10.0, 10.0])
        };
        let out = run(&ts, &[op(r#"{"op":"interpolate","t":0.5,"source":"other@1","path":"w[1:3]"}"#)], false, 100, &src).unwrap();
        assert_eq!(out[0].values.as_deref(), Some([0.0, 5.5, 6.0].as_slice()));
        let out = run(&ts, &[op(r#"{"op":"copy","source":{"map":"reranker/job-filter-x.json"},"path":"w[0]"}"#)], false, 100, &src).unwrap();
        assert_eq!(out[0].values.as_deref(), Some([10.0, 1.0, 2.0].as_slice()));
        // Between two versions: a + t * (b - a), whatever the values here.
        let two = |s: &Source, _: &str, _: &[usize]| -> Result<Vec<f32>, String> { Ok(if *s == Source::Object("a@1".into()) { vec![0.0, 2.0, 4.0] } else { vec![4.0, 2.0, 0.0] }) };
        let out = run(&ts, &[op(r#"{"op":"interpolate","t":0.25,"source":"a@1","source_b":"b@2","path":"w"}"#)], false, 100, &two).unwrap();
        assert_eq!(out[0].values.as_deref(), Some([1.0, 2.0, 3.0].as_slice()));
    }

    fn product(down: &[f32], up: &[f32], r: usize, k: usize, n: usize) -> Vec<f64> {
        (0..n * k).map(|x| (0..r).map(|t| f64::from(up[(x / k) * r + t]) * f64::from(down[t * k + x % k])).sum()).collect()
    }

    #[test]
    fn rank_reduction_keeps_the_best_approximation_and_diff_reports() {
        // up * down = diag(5, 3, 1) in a rotated basis: rank 2 keeps 5 and 3 and drops 1, whatever the
        // order of the components in the factors.
        let (c, s) = (0.6_f32, 0.8_f32);
        let down = vec![c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0];
        let up = vec![1.0 * c, 3.0 * -s, 5.0 * 0.0, 1.0 * s, 3.0 * c, 0.0, 0.0, 0.0, 5.0];
        let full = product(&down, &up, 3, 3, 3);
        let (nd, nu) = svd_rank(&down, &up, 3, 3, 3, 3);
        let same = product(&nd, &nu, 3, 3, 3);
        assert!(full.iter().zip(&same).all(|(a, b)| (a - b).abs() < 1e-5), "full rank reproduces the product");
        let (nd, nu) = svd_rank(&down, &up, 3, 3, 3, 2);
        let two = product(&nd, &nu, 2, 3, 3);
        let err: f64 = full.iter().zip(&two).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        assert!((err - 1.0).abs() < 1e-5, "the smallest singular value (1) is the error, got {err}");
        let naive: f64 = full.iter().zip(&product(&down[..6], &up.iter().enumerate().filter(|(i, _)| i % 3 != 2).map(|(_, x)| *x).collect::<Vec<_>>(), 2, 3, 3)).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        assert!(naive > err + 1.0, "truncation would drop the largest component here ({naive})");
        let ts = vec![t("down", vec![3, 2], 0.0), t("up", vec![2, 3], 0.0)];
        let out = run(&ts, &[op(r#"{"op":"rank","rank":2}"#)], false, 100, &none).unwrap();
        assert_eq!((out[0].shape.clone(), out[1].shape.clone()), (vec![2, 2], vec![2, 2]));
        assert!(run(&ts, &[op(r#"{"op":"rank","rank":4}"#)], false, 100, &none).is_err());
        let d = diff(&ts, &run(&ts, &[op(r#"{"op":"add","value":1,"path":"down[0:2]"}"#)], false, 100, &none).unwrap());
        assert_eq!(d.get("changed").unwrap().integer().unwrap(), 2);
        assert_eq!(d.get("tensors").unwrap().array().unwrap()[0].get("sample").unwrap().array().unwrap().len(), 2);
    }
}
