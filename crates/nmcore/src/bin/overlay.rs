//! The parameter overlay of every object that holds parameters, as the interface, MCP and the API
//! reach it: read an object's overlay, and change it by edit, mask, unmask, freeze, unfreeze, clear
//! and undo, each with a preview.
//!
//! Objects are read by `cparams::held` (the same keys: `open:<model>`, `id@version`,
//! `filter:<role>/<job>`, `<adapter>@<n>`, a unit's id). A selection (`path` or `selection`, as the
//! parameter editor reads them; none or `*` is the whole object) is resolved here into regions of
//! named tensors (`nmengn::overlay::Region`), which is what the overlay file keeps and the engine
//! replays when it seats the object. The released open weights and every cartridge file stay as they
//! are; only `memory/overlays/<slug>.json` is written, through a temporary file.
//!
//! Every change is one audit row of kind `parameter-overlay` naming the entries it added and removed
//! and a snapshot of the overlay before it (`memory/overlays/history/`); undo (the row's id through
//! the audit log's undo, or `action: undo` for the newest change of a key) removes what was added and
//! puts back what was removed at its place, refused where a freeze covers it. Freezes are removed only
//! by `unfreeze`: an edit, mask or unmask that touches a frozen region is refused with the freeze
//! named, and so is `clear` while the overlay holds a freeze.
use crate::cparams::{Held, Tensor};
use crate::pedit::{OpKind, Sel};
use nmengn::overlay::{Action, Entry, Op, Overlay, Region, Span};
use nmjson::Value;
use std::path::{Path, PathBuf};

/// The audit row kind of an overlay change.
pub const AUDIT_KIND: &str = "parameter-overlay";
/// Where the overlay before each change is kept, under the Nano's folder.
pub const HISTORY: &str = "memory/overlays/history";
/// Most values an edit may capture per region (a paste, a copy, an interpolation or a blend): a whole
/// tensor of the MicroModels up to 16 Mi values (64 MiB as f32), kept in a file beside the overlay
/// (`nmengn::overlay`, past a few thousand values) so the overlay itself stays small.
pub const CAPTURE_MAX: usize = 16 << 20;

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}
fn f(x: f32) -> Value {
    if x.is_finite() { Value::Number(format!("{x:e}")) } else { Value::Null }
}
fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn text(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default()
}
fn strs(v: &[String]) -> Value {
    Value::Array(v.iter().map(|x| s(x)).collect())
}

/// Counts of an object's overlay entries for the list (`{edits, masks, freezes}`), or null when it
/// has none. Reads the overlay file alone.
#[must_use]
pub fn counts(home: &Path, key: &str) -> Value {
    match nmengn::overlay::load(home, key) {
        Ok(Some(o)) if !o.entries.is_empty() => {
            let (e, m, fz) = o.counts();
            obj(vec![("edits", n(e as u64)), ("masks", n(m as u64)), ("freezes", n(fz as u64))])
        }
        _ => Value::Null,
    }
}

/// The value at which a tensor of `h` changes nothing: a norm weight's 1 for the open weights and a
/// base cartridge, else the tensor's own (1 for a gain, 0 otherwise).
fn identity_of(h: &Held, t: &Tensor) -> f32 {
    if matches!(h.kind, "open" | "base") { nmengn::overlay::weights_identity(&t.name) } else { t.identity }
}

/// The element type the engine holds a tensor's values in: the stored type for the open weights and
/// a base cartridge (bytes are patched in place), f32 for every other object (read into f32).
fn dtype_of(h: &Held, t: &Tensor) -> String {
    if matches!(h.kind, "open" | "base") { t.dtype.clone() } else { "F32".to_owned() }
}

/// Where a selection lies inside one tensor, as a flat range or a block of a matrix.
/// # Errors
/// A selection outside the tensor's shape, or one that needs two dimensions on a vector.
fn span_of(sel: &Sel, t: &Tensor) -> Result<Span, String> {
    let size: usize = t.shape.iter().product();
    let two = t.shape.len() == 2;
    let (r, c) = match t.shape.as_slice() {
        [r, c] => (*r, *c),
        _ => (size, 1),
    };
    let within = |x: (usize, usize), lim: usize, what: &str| if x.1 > lim { Err(format!("shape check: {} is {:?}; {what} {}:{} goes past {lim}", t.name, t.shape, x.0, x.1)) } else { Ok(x) };
    if let Some(x) = sel.range {
        let (a, b) = within(x, size, "range")?;
        return Ok(Span::Flat(a, b));
    }
    if !two && sel.cols.is_some() {
        return Err(format!("{} is a vector {:?}: select by range, rows or channel", t.name, t.shape));
    }
    let mut rows = sel.rows.map_or(Ok((0, r)), |x| within(x, r, "rows"))?;
    let mut cols = sel.cols.map_or(Ok((0, c)), |x| within(x, c, "columns"))?;
    if let Some((h, d)) = sel.head {
        rows = within((h * d, (h + 1) * d), r, "head")?;
    }
    if let Some(ch) = sel.channel {
        if two {
            cols = within((ch, ch + 1), c, "channel")?;
        } else {
            rows = within((ch, ch + 1), r, "channel")?;
        }
    }
    Ok(if two { Span::Block { rows, cols } } else { Span::Flat(rows.0, rows.1) })
}

/// The regions a selection names in an object. `scaled` keeps only tensors that carry a change
/// (a strength's tensors) where a tensor is not named and the object has any.
/// # Errors
/// A selection that picks nothing or lies outside a shape, or a layer on an object without layers.
fn regions(h: &Held, sel: &Sel, scaled: bool) -> Result<Vec<Region>, String> {
    if sel.layer.is_some() && !h.tensors.iter().any(|t| t.layer.is_some()) {
        return Err(format!("{} has no layers: it acts on {}", h.key, h.tensors.first().map_or("one node", |t| t.target.as_str())));
    }
    let only_scaled = scaled && sel.tensor.is_none() && h.tensors.iter().any(|t| t.scales);
    let mut out = Vec::new();
    for t in h.tensors.iter().filter(|t| crate::pedit::picks(sel, t)) {
        if only_scaled && !t.scales {
            continue;
        }
        let span = span_of(sel, t)?;
        let width = if t.shape.len() == 2 { t.shape.get(1).copied().unwrap_or(1) } else { t.shape.iter().product() };
        let r = Region { tensor: t.name.clone(), span, width, identity: identity_of(h, t), dtype: dtype_of(h, t), values: None };
        if !r.is_empty() {
            out.push(r);
        }
    }
    if out.is_empty() {
        return Err(format!("{} selects nothing in {}", crate::pedit::sel_name(sel), h.key));
    }
    Ok(out)
}

/// The regions a whole-object mask switches off, and what it leaves out with why: every tensor of a
/// filter and of preference counts; the tensors that carry the change of an adapter or a unit; every
/// layer's sets of the open weights and of a base cartridge (each layer then an exact identity on the
/// residual stream), the token table, final norm and output head left as they are.
fn whole_regions(h: &Held) -> (Vec<Region>, String) {
    let all = Sel::default();
    let keep: Box<dyn Fn(&Tensor) -> bool> = match h.kind {
        "open" | "base" => Box::new(|t: &Tensor| t.layer.is_some()),
        "adapter" | "lens" | "readout" | "unit" => Box::new(|t: &Tensor| t.scales),
        _ => Box::new(|_: &Tensor| true),
    };
    let mut out = Vec::new();
    let mut left = Vec::new();
    for t in &h.tensors {
        if !keep(t) {
            left.push(t.name.clone());
            continue;
        }
        if let Ok(span) = span_of(&all, t) {
            let width = if t.shape.len() == 2 { t.shape.get(1).copied().unwrap_or(1) } else { t.shape.iter().product() };
            out.push(Region { tensor: t.name.clone(), span, width, identity: identity_of(h, t), dtype: dtype_of(h, t), values: None });
        }
    }
    let note = match h.kind {
        "open" => format!(
            "every layer's sets masked (projections to 0, norm weights to 1: each layer an exact identity on the residual stream); left as they are, because masking them leaves no output: {}",
            left.join(", ")
        ),
        "base" => "masked whole, the base cartridge is left out where it is offered and the next one or the open weights serve".to_owned(),
        _ if left.is_empty() => String::new(),
        _ => format!("the tensors that carry the change are masked; left as they are (they carry no change by themselves): {}", left.join(", ")),
    };
    (out, note)
}

/// The released values of one region, read from memory or by ranged reads of the file.
fn region_released(t: &Tensor, r: &Region) -> Result<Vec<f32>, String> {
    let mut idx = Vec::with_capacity(r.len());
    r.each(|_, i| idx.push(i));
    crate::cparams::region_values(t, &idx)
}

/// The entry an edit makes: its operation replayed on the released values, or for a paste, a copy, an
/// interpolation or an import, the values it gives captured per region.
fn edit_entry(home: &Path, h: &Held, ov: &Overlay, op: &crate::pedit::Op) -> Result<(Action, Vec<Region>), String> {
    let scale = matches!(op.kind, OpKind::Scale(_) | OpKind::Drop);
    let mut regs = regions(h, &op.sel, scale)?;
    let action = match &op.kind {
        OpKind::Set(x) => Action::Edit(Op::Set(*x)),
        OpKind::Add(x) => Action::Edit(Op::Add(*x)),
        OpKind::Mul(x) => Action::Edit(Op::Mul(*x)),
        OpKind::Scale(x) => Action::Edit(Op::Scale(*x)),
        // A layer switched off in an overlay zeroes its change (the points stay; mask removes it).
        OpKind::Drop if op.sel.layer.is_some() && op.sel.tensor.is_none() => Action::Edit(Op::Scale(0.0)),
        OpKind::Clamp(a, b) => Action::Edit(Op::Clamp(*a, *b)),
        OpKind::Zero => Action::Edit(Op::Zero),
        OpKind::Identity => Action::Edit(Op::Identity),
        OpKind::Negate => Action::Edit(Op::Negate),
        OpKind::Paste(p) => {
            for r in &mut regs {
                if p.len() != r.len() {
                    return Err(format!("shape check: {} selects {} values; {} pasted", r.path(), r.len(), p.len()));
                }
                r.values = Some(p.clone());
            }
            Action::Edit(Op::Paste)
        }
        OpKind::Copy(_) | OpKind::Interp(..) | OpKind::Blend(..) => {
            for r in &mut regs {
                if r.len() > CAPTURE_MAX {
                    return Err(format!("{} selects {} values; a copy or interpolation captures at most {CAPTURE_MAX}: select a smaller region", r.path(), r.len()));
                }
                let t = h.tensors.iter().find(|t| t.name == r.tensor).ok_or("no tensor")?;
                let get = |src: &crate::pedit::Source| crate::cparams::source_values(home, src, &t.name, &t.shape);
                // (first source, second source for a blend, weight): a copy is weight 1 toward one.
                let (sa, sb, tt) = match &op.kind {
                    OpKind::Copy(a) => (get(a)?, None, 1.0),
                    OpKind::Interp(a, x) => (get(a)?, None, *x),
                    OpKind::Blend(a, b, x) => (get(a)?, Some(get(b)?), *x),
                    _ => return Err("not a copy".to_owned()),
                };
                // Toward one source, from the values here with the overlay so far; a blend needs none.
                let base = if sb.is_none() { region_released(t, r)? } else { Vec::new() };
                let mut got = Vec::with_capacity(r.len());
                let mut k = 0;
                let mut fail = None;
                r.each(|_, i| {
                    match (sa.get(i), sb.as_ref().map(|b| b.get(i))) {
                        (Some(a), Some(Some(b))) => got.push(a + tt * (b - a)),
                        (Some(a), None) => {
                            let x = ov.at(&r.tensor, &r.dtype, i, base.get(k).copied().unwrap_or(0.0)).0;
                            got.push(x + tt * (a - x));
                        }
                        _ => fail = Some(i),
                    }
                    k += 1;
                });
                if let Some(i) = fail {
                    return Err(format!("the source holds no value at {}[{i}]", r.tensor));
                }
                r.values = Some(got);
            }
            Action::Edit(Op::Paste)
        }
        OpKind::Drop | OpKind::AddPoint { .. } | OpKind::Rank(_) => {
            return Err("a structural change (drop, add_point, rank) is made as a new version: apply it as a new version (POST /api/cartridges/parameters)".to_owned());
        }
    };
    for r in &regs {
        if let Some(v) = &r.values
            && v.len() > CAPTURE_MAX
        {
            return Err(format!("{}: {} values to capture; at most {CAPTURE_MAX}", r.path(), v.len()));
        }
    }
    Ok((action, regs))
}

/// The freeze a change would touch, as a refusal.
fn refuse_frozen(ov: &Overlay, regs: &[Region], what: &str) -> Result<(), String> {
    if let Some(fz) = ov.frozen_by(regs) {
        let at = regs.iter().find(|r| fz.regions.iter().any(|x| x.meets(r))).map_or_else(String::new, Region::path);
        return Err(format!("{what} refused: {at} is frozen by {} ({}); unfreeze it first (action: unfreeze, id: {})", fz.id, fz.selection, fz.id));
    }
    Ok(())
}

/// When a change of an object's overlay takes effect, by its kind: `when` is one of `applied now`,
/// `applied at the next call`, `applied at the next start`, with the reason.
#[must_use]
pub fn takes_effect(home: &Path, h: &Held) -> Value {
    // A MicroModel bound again at a call boundary: how long it took the last time on this machine.
    let reload = |model: &str| match nmengn::overlay::reload_secs(home, model) {
        Some(x) => format!("the last time on this machine it took {x:.1} s"),
        None => "how long it takes on this machine is not measured yet (it is recorded at the first one)".to_owned(),
    };
    let (when, why): (&str, String) = match (h.source, h.kind) {
        ("job-filter", _) => ("applied at the next call", "a serving process loads its job filters at each forward pass (checked once a second) and their overlays with them".to_owned()),
        ("library", "filter" | "adapter") => (
            "applied at the next call",
            "a serving process re-reads the overlays of its plugged socket cartridges within a second and replugs them where its next call starts; a cartridge not plugged carries its overlay when it is plugged".to_owned(),
        ),
        ("library", "preference") => ("applied at the next call", "a serving process re-reads the preference values with their overlays within a second; the next reply uses them".to_owned()),
        ("adapter", _) => ("applied at the next call", "the lens and the judge's read-out are loaded again at the next look after an overlay changes".to_owned()),
        ("unit", _) => ("applied at the next call", "a unit loaded on a serving process is set on its node again with its overlay where the next call starts (masked whole: taken off its node); a unit not loaded carries it when it loads".to_owned()),
        ("library", "base") | ("open-weights", _) => (
            "applied at the next call",
            format!(
                "a serving process that holds {} binds that MicroModel again with its overlays where its next call starts (its packed and card copies of every tensor are made at bind, so a touched tensor is not rewritten in place); {}. The base cartridge's seat check compares it as written, then its overlay applies; the open weights' overlay applies when they serve it. The Writer loads per writing run. When the Embedder is bound again with other values, the dense lane follows without a restart: it is marked stale (questions run on the lexical and structural paths meanwhile), a lane for the new values is built at idle, and it is put in place once it covers every piece",
                h.model,
                reload(&h.model)
            ),
        ),
        _ => ("applied at the next start", "the object is read again when it is next loaded".to_owned()),
    };
    obj(vec![("view", s("applied now: the parameters view and region reads show the overlaid values")), ("when", s(when)), ("reason", s(&why))])
}

/// One entry as the API shows it.
fn entry_value(e: &Entry) -> Value {
    let op = match &e.action {
        Action::Edit(o) => s(&o.phrase()),
        _ => Value::Null,
    };
    let values: usize = e.regions.iter().map(Region::len).sum();
    obj(vec![
        ("id", s(&e.id)),
        ("action", s(e.action.name())),
        ("op", op),
        ("selection", s(&e.selection)),
        ("whole", Value::Bool(e.whole)),
        ("regions", Value::Array(e.regions.iter().map(|r| obj(vec![("path", s(&r.path())), ("tensor", s(&r.tensor)), ("values", n(r.len() as u64)), ("identity", f(r.identity)), ("captured", Value::Bool(r.values.is_some()))])).collect())),
        ("values", n(values as u64)),
        ("by", s(&e.by)),
        ("at_ms", n(e.at_ms)),
        ("note", if e.note.is_empty() { Value::Null } else { s(&e.note) }),
    ])
}

/// Per tensor and per layer, how much of it the overlay edits, masks and freezes (for drawing the
/// layer strip and the tensor chips).
fn marks(h: &Held, ov: &Overlay) -> (Value, Value) {
    let mut by_t: std::collections::BTreeMap<String, [u64; 3]> = std::collections::BTreeMap::new();
    let mut by_l: std::collections::BTreeMap<usize, [bool; 3]> = std::collections::BTreeMap::new();
    for e in &ov.entries {
        let slot = match e.action {
            Action::Edit(_) => 0,
            Action::Mask => 1,
            Action::Freeze => 2,
        };
        for r in &e.regions {
            if let Some(c) = by_t.entry(r.tensor.clone()).or_default().get_mut(slot) {
                *c += r.len() as u64;
            }
            if let Some(l) = h.tensors.iter().find(|t| t.name == r.tensor).and_then(|t| t.layer)
                && let Some(b) = by_l.entry(l).or_default().get_mut(slot)
            {
                *b = true;
            }
        }
    }
    let t = Value::Object(by_t.into_iter().map(|(k, [a, b, c])| (k, obj(vec![("edited", n(a)), ("masked", n(b)), ("frozen", n(c))]))).collect());
    let l = Value::Array(by_l.into_iter().map(|(k, [a, b, c])| obj(vec![("layer", n(k as u64)), ("edited", Value::Bool(a)), ("masked", Value::Bool(b)), ("frozen", Value::Bool(c))])).collect());
    (t, l)
}

/// The overlay section of an object that is open (for `cparams::section_of`): its entries, the
/// marks per tensor and layer, the file, and when a change takes effect.
#[must_use]
pub fn section(home: &Path, h: &Held) -> Value {
    let ov = match nmengn::overlay::load(home, &h.key) {
        Ok(o) => o.unwrap_or_else(|| Overlay::new(&h.key, h.kind, &h.model)),
        Err(e) => return obj(vec![("error", s(&e))]),
    };
    let (tensors, layers) = marks(h, &ov);
    let (e, m, fz) = ov.counts();
    obj(vec![
        ("format", s(nmengn::overlay::FORMAT)),
        ("key", s(&h.key)),
        ("editable", Value::Bool(!h.tensors.is_empty())),
        ("why", if h.tensors.is_empty() { s("this object holds no parameters") } else { Value::Null }),
        ("file", s(&nmengn::overlay::path(home, &h.key).display().to_string())),
        ("entries", Value::Array(ov.entries.iter().map(entry_value).collect())),
        ("counts", obj(vec![("edits", n(e as u64)), ("masks", n(m as u64)), ("freezes", n(fz as u64))])),
        ("tensors", tensors),
        ("layers", layers),
        ("whole_masked", Value::Bool(ov.whole_masked())),
        ("takes_effect", takes_effect(home, h)),
        ("route", s("POST /api/parameters/overlay {key, action: edit|mask|unmask|freeze|unfreeze|clear|undo, path or selection, op..., preview}")),
    ])
}

/// `GET /api/parameters/overlay?key=<key>`: an object's overlay.
#[must_use]
pub fn view(home: &Path, key: &str) -> Value {
    match crate::cparams::held(home, key) {
        Ok(h) => section(home, &h),
        Err(e) => obj(vec![("key", s(key)), ("error", s(&e))]),
    }
}

/// The overlay's values against the released values for the read tensors a change touches: the
/// difference before and after (`pedit::diff`), and for tensors whose values are read by page, the
/// count of touched values with a sample from the first region.
fn preview_diff(h: &Held, before: &Overlay, after: &Overlay, regs: &[Region]) -> Value {
    let touched: std::collections::BTreeSet<&str> = regs.iter().map(|r| r.tensor.as_str()).collect();
    let (mut old, mut new) = (Vec::new(), Vec::new());
    let mut paged = Vec::new();
    for t in h.tensors.iter().filter(|t| touched.contains(t.name.as_str())) {
        let dt = dtype_of(h, t);
        if let Some(v) = &t.values {
            let (mut a, mut b) = (v.clone(), v.clone());
            before.apply(&t.name, &dt, &mut a);
            after.apply(&t.name, &dt, &mut b);
            old.push(Tensor { values: Some(a), ..t.clone() });
            new.push(Tensor { values: Some(b), ..t.clone() });
            continue;
        }
        let Some(r) = regs.iter().find(|r| r.tensor == t.name) else { continue };
        let mut idx = Vec::new();
        r.each(|_, i| {
            if idx.len() < 32 {
                idx.push(i);
            }
        });
        let sample = crate::cparams::region_values(t, &idx).map(|xs| {
            Value::Array(idx.iter().zip(xs).map(|(i, x)| Value::Array(vec![n(*i as u64), f(before.at(&t.name, &dt, *i, x).0), f(after.at(&t.name, &dt, *i, x).0)])).collect())
        });
        let values: usize = regs.iter().filter(|x| x.tensor == t.name).map(Region::len).sum();
        paged.push(obj(vec![("name", s(&t.name)), ("touched", n(values as u64)), ("sample", sample.unwrap_or_else(|e| s(&e)))]));
    }
    let mut d = crate::pedit::diff(&old, &new);
    if let Value::Object(o) = &mut d {
        o.insert("paged".into(), Value::Array(paged));
        o.insert("touched".into(), n(regs.iter().map(|r| r.len() as u64).sum()));
    }
    d
}

/// A change of an overlay before it is written: the overlay after it, the entries added and removed,
/// and a short label.
struct Change {
    after: Overlay,
    added: Vec<String>,
    removed: Vec<String>,
    label: String,
    regions: Vec<Region>,
}

fn next_id(ov: &mut Overlay, prefix: char) -> String {
    let id = format!("{prefix}{}", ov.next.max(1));
    ov.next = ov.next.max(1) + 1;
    id
}

/// The selection a request names (`path`, `selection`), and whether it is the whole object.
fn selection_of(v: &Value) -> Result<(Sel, String, bool), String> {
    let sel = crate::pedit::parse_sel(v)?;
    let whole = sel == Sel::default() || matches!(v.get("whole"), Ok(Value::Bool(true)));
    let name = if whole { "*".to_owned() } else { text(v, "path").trim().to_owned() };
    let name = if name.is_empty() { crate::pedit::sel_name(&sel) } else { name };
    Ok((sel, name, whole))
}

/// The entries of `kind` an unmask or unfreeze takes out: by `id`, or every one meeting the selection.
fn picked(ov: &Overlay, h: &Held, v: &Value, kind: &str) -> Result<Vec<String>, String> {
    let ids: Vec<String> = match v.get("id") {
        Ok(Value::String(x)) if !x.is_empty() => vec![x.clone()],
        _ => match v.get("ids") {
            Ok(Value::Array(a)) => a.iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect(),
            _ => Vec::new(),
        },
    };
    if !ids.is_empty() {
        for id in &ids {
            match ov.entries.iter().find(|e| e.id == *id) {
                Some(e) if e.action.name() == kind => {}
                Some(e) => return Err(format!("{id} is a {} entry, not a {kind}", e.action.name())),
                None => return Err(format!("{} has no overlay entry {id}", h.key)),
            }
        }
        return Ok(ids);
    }
    let (sel, _, whole) = selection_of(v)?;
    let regs = if whole { Vec::new() } else { regions(h, &sel, false)? };
    let out: Vec<String> = ov.entries.iter().filter(|e| e.action.name() == kind && (whole || e.regions.iter().any(|r| regs.iter().any(|x| x.meets(r))))).map(|e| e.id.clone()).collect();
    if out.is_empty() {
        return Err(format!("no {kind} entry of {} meets the selection", h.key));
    }
    Ok(out)
}

fn change_of(home: &Path, h: &Held, ov: &Overlay, action: &str, v: &Value, by: &str) -> Result<Change, String> {
    let mut after = ov.clone();
    let now = nmtype::now_ms();
    match action {
        "edit" => {
            let e = crate::cparams::parse_edit(v)?;
            if e.ops.is_empty() {
                return Err("edit: give op (with path or selection), ops, scale, layers or values".to_owned());
            }
            let (mut added, mut regs_all, mut phrases) = (Vec::new(), Vec::new(), Vec::new());
            for op in &e.ops {
                let (act, regs) = edit_entry(home, h, &after, op)?;
                refuse_frozen(&after, &regs, "edit")?;
                let id = next_id(&mut after, 'e');
                phrases.push(op.phrase());
                after.entries.push(Entry { id: id.clone(), action: act, selection: crate::pedit::sel_name(&op.sel), whole: false, regions: regs.clone(), by: by.to_owned(), at_ms: now, note: String::new() });
                regs_all.extend(regs);
                added.push(id);
            }
            Ok(Change { after, label: format!("edit {}", phrases.join(", ")), added, removed: Vec::new(), regions: regs_all })
        }
        "mask" | "freeze" => {
            let (sel, name, whole) = selection_of(v)?;
            let (regs, note) = if whole { whole_regions(h) } else { (regions(h, &sel, false)?, String::new()) };
            if action == "mask" {
                refuse_frozen(&after, &regs, "mask")?;
            }
            let id = next_id(&mut after, if action == "mask" { 'm' } else { 'f' });
            let act = if action == "mask" { Action::Mask } else { Action::Freeze };
            let label = format!("{action} {name}");
            after.entries.push(Entry { id: id.clone(), action: act, selection: name, whole, regions: regs.clone(), by: by.to_owned(), at_ms: now, note });
            Ok(Change { after, label, added: vec![id], removed: Vec::new(), regions: regs })
        }
        "unmask" | "unfreeze" => {
            let kind = if action == "unmask" { "mask" } else { "freeze" };
            let ids = picked(ov, h, v, kind)?;
            let regs: Vec<Region> = ov.entries.iter().filter(|e| ids.contains(&e.id)).flat_map(|e| e.regions.clone()).collect();
            if action == "unmask" {
                refuse_frozen(ov, &regs, "unmask")?;
            }
            after.entries.retain(|e| !ids.contains(&e.id));
            Ok(Change { after, label: format!("{action} {}", ids.join(", ")), added: Vec::new(), removed: ids, regions: regs })
        }
        "clear" => {
            if let Some(fz) = ov.entries.iter().find(|e| matches!(e.action, Action::Freeze)) {
                return Err(format!("clear refused: the overlay holds freeze {} ({}); unfreezing is explicit, unfreeze first", fz.id, fz.selection));
            }
            let ids: Vec<String> = ov.entries.iter().map(|e| e.id.clone()).collect();
            if ids.is_empty() {
                return Err(format!("{} has no overlay entries to clear", h.key));
            }
            let regs = ov.entries.iter().flat_map(|e| e.regions.clone()).collect();
            after.entries.clear();
            Ok(Change { after, label: format!("clear {} entries", ids.len()), added: Vec::new(), removed: ids, regions: regs })
        }
        o => Err(format!("action {o}: edit, mask, unmask, freeze, unfreeze, clear or undo")),
    }
}

/// Keeps the overlay before a change, for undo. Returns the snapshot's path.
fn snapshot(home: &Path, ov: &Overlay) -> Result<PathBuf, String> {
    let dir = home.join(HISTORY);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let p = dir.join(format!("{}-{}-{}.json", nmengn::overlay::slug(&ov.key), nmtype::now_ms(), ov.next));
    std::fs::write(&p, nmengn::overlay::encode(home, ov)?.as_bytes()).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok(p)
}

/// Writes a change: the snapshot, the overlay, one audit row. Returns the audit id.
#[expect(clippy::too_many_arguments, reason = "the change, the object and the audit row's own fields")]
fn commit(home: &Path, h: &Held, before: &Overlay, c: &Change, action: &str, by: &str, undo_of: Option<&str>) -> Result<String, String> {
    let snap = snapshot(home, before)?;
    let file = nmengn::overlay::save(home, &c.after)?;
    let mut fields = vec![
        ("action", s(action)),
        ("key", s(&h.key)),
        ("object", s(h.kind)),
        ("added", strs(&c.added)),
        ("removed", strs(&c.removed)),
        ("snapshot", s(&snap.display().to_string())),
        ("file", s(&file.display().to_string())),
        ("applies", s(&text(&takes_effect(home, h), "when"))),
        ("by", s(by)),
        ("label", s(&format!("Parameter overlay of {}: {}", h.key, c.label))),
    ];
    if let Some(u) = undo_of {
        fields.push(("undo_of", s(u)));
    }
    crate::controls::audit(home, AUDIT_KIND, fields)
}

fn held_ok(home: &Path, key: &str) -> Result<Held, String> {
    let h = crate::cparams::held(home, key)?;
    if h.tensors.is_empty() {
        return Err(format!("{key} holds no parameters: {}", h.editable.clone().err().unwrap_or_else(|| "nothing to overlay".to_owned())));
    }
    Ok(h)
}

/// `POST /api/parameters/overlay {key, action, ...}`: one change of an object's overlay, previewed
/// (`preview: true`, nothing written) or written with its audit row. `clear` comes here only after
/// the delete gate's second step (`confirm: true`).
/// # Errors
/// An unknown object or one without parameters, a selection or operation that does not read, a
/// change touching a frozen region, or a write failure.
pub fn post(home: &Path, v: &Value, by: &str) -> Result<Value, String> {
    let key = v.get("key").and_then(Value::string).map_err(|_| "key: the object's key (open:<model>, id@version, filter:<role>/<job>, <adapter>@<n> or a unit's id)".to_owned())?.to_owned();
    let action = v.get("action").and_then(Value::string).map_err(|_| "action: edit, mask, unmask, freeze, unfreeze, clear or undo".to_owned())?.to_owned();
    if action == "undo" {
        return undo_newest(home, &key, by);
    }
    let h = held_ok(home, &key)?;
    let before = nmengn::overlay::load(home, &key)?.unwrap_or_else(|| Overlay::new(&key, h.kind, &h.model));
    let c = change_of(home, &h, &before, &action, v, by)?;
    if matches!(v.get("preview"), Ok(Value::Bool(true))) {
        return Ok(obj(vec![
            ("preview", Value::Bool(true)),
            ("key", s(&key)),
            ("action", s(&action)),
            ("change", s(&c.label)),
            ("added", Value::Array(c.after.entries.iter().filter(|e| c.added.contains(&e.id)).map(entry_value).collect())),
            ("removed", strs(&c.removed)),
            ("diff", preview_diff(&h, &before, &c.after, &c.regions)),
            ("writes", s(&format!("{} only; the object's files are never written", nmengn::overlay::path(home, &key).display()))),
            ("takes_effect", takes_effect(home, &h)),
        ]));
    }
    if action == "clear" && !matches!(v.get("confirm"), Ok(Value::Bool(true))) {
        return Err("clear removes every entry of the overlay: send it through the delete gate (a preview with a confirm_token, then the same request with it)".to_owned());
    }
    let id = commit(home, &h, &before, &c, &action, by, None)?;
    Ok(obj(vec![
        ("changed", Value::Bool(true)),
        ("audit_id", s(&id)),
        ("key", s(&key)),
        ("action", s(&action)),
        ("change", s(&c.label)),
        ("added", strs(&c.added)),
        ("removed", strs(&c.removed)),
        ("takes_effect", takes_effect(home, &h)),
        ("overlay", section(home, &h)),
    ]))
}

/// The preview `clear` shows at the delete gate's first step: the entries it would remove.
/// # Errors
/// An unknown object, or an overlay that cannot be cleared (empty, or holding a freeze).
pub fn clear_preview(home: &Path, key: &str) -> Result<(Value, String), String> {
    let h = held_ok(home, key)?;
    let ov = nmengn::overlay::load(home, key)?.unwrap_or_else(|| Overlay::new(key, h.kind, &h.model));
    let c = change_of(home, &h, &ov, "clear", &Value::Null, "preview")?;
    let items: Vec<Value> = ov.entries.iter().map(|e| obj(vec![("label", s(&format!("{} {} on {}", e.id, e.action.name(), e.selection))), ("path", Value::Null), ("bytes", Value::Null), ("files", Value::Null)])).collect();
    Ok((
        obj(vec![
            ("title", s(&format!("Clear the parameter overlay of {key}"))),
            ("items", Value::Array(items)),
            ("bytes", Value::Null),
            ("files", Value::Null),
            ("undo", s("can be undone: Audit, Undo puts every entry back")),
            ("note", s(&format!("{}; the object's files were never written", c.label))),
        ]),
        "clear".to_owned(),
    ))
}

/// The audit rows of the overlay, oldest first.
fn rows(home: &Path) -> Vec<Value> {
    std::fs::read_to_string(home.join("memory/audit.jsonl")).map(|t| t.lines().filter_map(|l| nmjson::parse(l.as_bytes()).ok()).filter(|r| text(r, "kind") == AUDIT_KIND).collect()).unwrap_or_default()
}

/// `action: undo`: undoes the newest change of a key that is not undone yet.
fn undo_newest(home: &Path, key: &str, by: &str) -> Result<Value, String> {
    let all = rows(home);
    let undone: std::collections::BTreeSet<String> = all.iter().map(|r| text(r, "undo_of")).filter(|x| !x.is_empty()).collect();
    let row = all.iter().rev().find(|r| text(r, "key") == key && text(r, "action") != "undo" && !undone.contains(&text(r, "id"))).ok_or_else(|| format!("{key}: no overlay change left to undo"))?;
    let id = text(row, "id");
    let a = undo(home, row, &id, by)?;
    let h = held_ok(home, key)?;
    Ok(obj(vec![("changed", Value::Bool(true)), ("audit_id", a.map_or(Value::Null, |x| s(&x))), ("undo_of", s(&id)), ("key", s(key)), ("takes_effect", takes_effect(home, &h)), ("overlay", section(home, &h))]))
}

/// Undoes one overlay change by its audit row: the entries it added are taken out, those it removed
/// are put back at their places (from the row's snapshot). Refused where a freeze made since covers
/// what would change. The undo is itself a change with its own row (undone the same way).
/// # Errors
/// A row without its snapshot, or a freeze covering the change.
pub fn undo(home: &Path, row: &Value, id: &str, by: &str) -> Result<Option<String>, String> {
    let key = text(row, "key");
    let h = held_ok(home, &key)?;
    let now = nmengn::overlay::load(home, &key)?.unwrap_or_else(|| Overlay::new(&key, h.kind, &h.model));
    let list = |k: &str| row.get(k).and_then(Value::array).map(|a| a.iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect::<Vec<_>>()).unwrap_or_default();
    let (added, removed) = (list("added"), list("removed"));
    let snap_path = PathBuf::from(text(row, "snapshot"));
    let snap = std::fs::read(&snap_path).map_err(|e| format!("{}: {e}", snap_path.display())).and_then(|b| nmengn::overlay::decode(home, &b))?;
    let mut after = now.clone();
    let mut touched: Vec<Region> = Vec::new();
    // What was added comes out; a freeze it added comes out without a check (that is the undo).
    for e in now.entries.iter().filter(|e| added.contains(&e.id)) {
        if !matches!(e.action, Action::Freeze) {
            touched.extend(e.regions.clone());
        }
    }
    after.entries.retain(|e| !added.contains(&e.id));
    // What was removed goes back at its place in the snapshot.
    for (pos, e) in snap.entries.iter().enumerate().filter(|(_, e)| removed.contains(&e.id)) {
        if after.entries.iter().any(|x| x.id == e.id) {
            continue;
        }
        if !matches!(e.action, Action::Freeze) {
            touched.extend(e.regions.clone());
        }
        let before_ids: Vec<&str> = snap.entries.iter().take(pos).map(|x| x.id.as_str()).collect();
        let at = after.entries.iter().rposition(|x| before_ids.contains(&x.id.as_str())).map_or(0, |i| i + 1);
        after.entries.insert(at, e.clone());
    }
    let mut check = after.clone();
    check.entries.retain(|e| !added.contains(&e.id));
    refuse_frozen(&check, &touched, "undo")?;
    after.next = after.next.max(now.next).max(snap.next);
    let c = Change { after, added: removed.clone(), removed: added.clone(), label: format!("undo of {id}"), regions: touched };
    commit(home, &h, &now, &c, "undo", by, Some(id)).map(Some)
}

/// The overlaid value and marks of each value a region read shows (for `cparams::region`):
/// `[value, masked, frozen]` per index, and whether the overlay changes any of them.
#[must_use]
pub fn region_marks(home: &Path, h: &Held, t: &Tensor, idx: &[usize], vals: &[f32]) -> Value {
    let Ok(Some(ov)) = nmengn::overlay::load(home, &h.key) else { return Value::Null };
    let dt = dtype_of(h, t);
    let mut out = Vec::with_capacity(idx.len());
    let mut any = false;
    for (i, x) in idx.iter().zip(vals) {
        let (v, m, fz) = ov.at(&t.name, &dt, *i, *x);
        any |= m || fz || v.to_bits() != x.to_bits();
        out.push(Value::Array(vec![f(v), Value::Bool(m), Value::Bool(fz)]));
    }
    if any { Value::Array(out) } else { Value::Null }
}

const USAGE: &str = "usage: neuromorph-nano library overlay <key>
       neuromorph-nano library overlay <key> edit --op <op> [--path <path>] [--value <x>] [--lo <x> --hi <x>] [--preview]
       neuromorph-nano library overlay <key> mask|freeze [--path <path>] [--preview]   (no path: the whole object)
       neuromorph-nano library overlay <key> unmask|unfreeze [--id <entry> | --path <path>] [--preview]
       neuromorph-nano library overlay <key> undo
       neuromorph-nano library overlay <key> clear [--confirm-token <token>]   (first call: a preview with a token)";

/// `library overlay ...`: the same reads and changes as the API, through the same route (so a clear
/// passes the delete gate in two calls).
/// # Errors
/// Arguments that do not read, or the route's own refusal.
pub fn cli(home: &Path, args: &[String]) -> Result<Value, String> {
    let key = args.get(2).filter(|a| !a.starts_with("--")).ok_or(USAGE)?;
    let flag = |name: &str| args.windows(2).find(|w| w.first().is_some_and(|a| a == name)).and_then(|w| w.get(1)).cloned();
    let Some(action) = args.get(3).filter(|a| !a.starts_with("--")) else {
        return crate::ui::lent_route(home, "/api/parameters/overlay", &format!("key={}", enc(key)), None, "command");
    };
    let mut b = vec![("key", s(key)), ("action", s(action))];
    for (k, name) in [("path", "--path"), ("id", "--id"), ("op", "--op"), ("value", "--value"), ("lo", "--lo"), ("hi", "--hi"), ("confirm_token", "--confirm-token")] {
        if let Some(x) = flag(name) {
            b.push((k, s(&x)));
        }
    }
    if args.iter().any(|a| a == "--preview") {
        b.push(("preview", Value::Bool(true)));
    }
    let body = obj(b).encode().map_err(|e| e.to_string())?;
    crate::ui::lent_route(home, "/api/parameters/overlay", "", Some(body.as_bytes()), "command")
}

fn enc(x: &str) -> String {
    x.bytes().map(|b| if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') { char::from(b).to_string() } else { format!("%{b:02X}") }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nmlearn::library::{CartridgeSpec, Filter, Library, Point};

    struct Home(PathBuf);
    impl Drop for Home {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn home(name: &str) -> Home {
        let d = std::env::temp_dir().join(format!("nano-overlay-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("memory")).unwrap();
        Home(d)
    }
    fn req(json: &str) -> Value {
        nmjson::parse(json.as_bytes()).unwrap()
    }
    fn job(h: &Path) {
        let dir = h.join("library/parameter-maps/reranker");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("job-filter-ov.json"), r#"{"cartridge":"reranker-ov@1","contract":"nmengn-sockets/1","enabled":true,"job":"ov","model":"qwen3-reranker-0.6b","parts":[{"target":"L2.o","gain":[1.5,0.5,2.0],"bias":[0.1,0.2,0.3]},{"target":"L4.o","gain":[2.0],"bias":[1.0]}]}"#).unwrap();
    }
    const K: &str = "filter:reranker/ov";
    fn effective(h: &Path, name: &str) -> Vec<f32> {
        let held = crate::cparams::held(h, K).unwrap();
        let t = held.tensors.iter().find(|t| t.name == name).unwrap();
        let mut v = t.values.clone().unwrap();
        if let Some(o) = nmengn::overlay::load(h, K).unwrap() {
            o.apply(name, "F32", &mut v);
        }
        v
    }

    #[test]
    fn overlay_apply_gives_the_expected_values_and_the_file_stays() {
        let h = home("apply");
        job(&h.0);
        let file = h.0.join("library/parameter-maps/reranker/job-filter-ov.json");
        let before = std::fs::read(&file).unwrap();
        let p = post(&h.0, &req(r#"{"key":"filter:reranker/ov","action":"edit","op":"set","value":3,"path":"L2.o.gain[0:2]","preview":true}"#), "test").unwrap();
        assert!(matches!(p.get("preview"), Ok(Value::Bool(true))));
        assert_eq!(p.get("diff").unwrap().get("changed").unwrap().integer().unwrap(), 2);
        assert!(nmengn::overlay::load(&h.0, K).unwrap().is_none(), "a preview writes nothing");
        let r = post(&h.0, &req(r#"{"key":"filter:reranker/ov","action":"edit","op":"set","value":3,"path":"L2.o.gain[0:2]"}"#), "test").unwrap();
        assert!(!text(&r, "audit_id").is_empty());
        assert_eq!(text(r.get("takes_effect").unwrap(), "when"), "applied at the next call");
        assert_eq!(effective(&h.0, "L2.o.gain"), vec![3.0, 3.0, 2.0]);
        assert_eq!(std::fs::read(&file).unwrap(), before, "the job filter's file is never written");
        // The engine's loader reads the same overlay into the filter's parts.
        let ov = nmengn::overlay::effective(&h.0, K).unwrap();
        let mut parts = vec![nmengn::filter::Part { target: "L2.o".into(), gain: vec![1.5, 0.5, 2.0], bias: vec![0.1, 0.2, 0.3] }];
        ov.apply_parts(&mut parts);
        assert_eq!(parts[0].gain, vec![3.0, 3.0, 2.0]);
        // A copy captures its values, so it replays the same whatever happens to the source later.
        let r = post(&h.0, &req(r#"{"key":"filter:reranker/ov","action":"edit","op":"interpolate","t":0.5,"source":{"map":"reranker/job-filter-ov.json"},"path":"L4.o.bias"}"#), "test").unwrap();
        assert!(r.get("audit_id").is_ok());
        let o = nmengn::overlay::load(&h.0, K).unwrap().unwrap();
        assert_eq!(o.entries.last().unwrap().regions[0].values.as_deref(), Some([1.0_f32].as_slice()));
        // A structural change goes to a new version.
        assert!(post(&h.0, &req(r#"{"key":"filter:reranker/ov","action":"edit","op":"add_point","target":"L9.o","width":2}"#), "test").unwrap_err().contains("new version"));
    }

    #[test]
    fn mask_and_unmask_restore_exactly_and_a_freeze_refuses() {
        let h = home("mask");
        job(&h.0);
        post(&h.0, &req(r#"{"key":"filter:reranker/ov","action":"edit","op":"mul","value":3,"path":"L2.o.bias"}"#), "test").unwrap();
        let edited = effective(&h.0, "L2.o.bias");
        post(&h.0, &req(r#"{"key":"filter:reranker/ov","action":"mask","path":"L2"}"#), "test").unwrap();
        assert_eq!((effective(&h.0, "L2.o.gain"), effective(&h.0, "L2.o.bias")), (vec![1.0; 3], vec![0.0; 3]), "masked: gain 1, bias 0");
        post(&h.0, &req(r#"{"key":"filter:reranker/ov","action":"unmask","path":"L2.o.bias[1]"}"#), "test").unwrap();
        assert_eq!(effective(&h.0, "L2.o.bias"), edited, "unmasked: the edited values bit for bit");
        // A freeze refuses an edit and a mask touching it, names itself, and only unfreeze lifts it.
        let fz = post(&h.0, &req(r#"{"key":"filter:reranker/ov","action":"freeze","path":"L4.o.gain"}"#), "test").unwrap();
        let fid = text(&fz, "added").trim().to_owned();
        let fid = if fid.is_empty() { fz.get("added").unwrap().array().unwrap()[0].string().unwrap().to_owned() } else { fid };
        let e = post(&h.0, &req(r#"{"key":"filter:reranker/ov","action":"edit","op":"set","value":9,"path":"L4"}"#), "test").unwrap_err();
        assert!(e.contains("frozen by") && e.contains(&fid), "{e}");
        assert!(post(&h.0, &req(r#"{"key":"filter:reranker/ov","action":"mask"}"#), "test").unwrap_err().contains("frozen"));
        assert!(post(&h.0, &req(r#"{"key":"filter:reranker/ov","action":"clear","confirm":true}"#), "test").unwrap_err().contains("unfreeze"));
        assert!(post(&h.0, &req(r#"{"key":"filter:reranker/ov","action":"edit","op":"set","value":9,"path":"L4.o.bias"}"#), "test").is_ok(), "the bias is not frozen");
        post(&h.0, &req(&format!(r#"{{"key":"filter:reranker/ov","action":"unfreeze","id":"{fid}"}}"#)), "test").unwrap();
        assert!(post(&h.0, &req(r#"{"key":"filter:reranker/ov","action":"edit","op":"set","value":9,"path":"L4.o.gain"}"#), "test").is_ok());
        // A whole mask marks the object: the loader leaves it out while it stays plugged.
        post(&h.0, &req(r#"{"key":"filter:reranker/ov","action":"mask","whole":true}"#), "test").unwrap();
        assert!(nmengn::overlay::effective(&h.0, K).unwrap().whole_masked());
    }

    #[test]
    fn ordering_persistence_and_undo() {
        let h = home("undo");
        job(&h.0);
        post(&h.0, &req(r#"{"key":"filter:reranker/ov","action":"edit","op":"add","value":1,"path":"L4.o.gain"}"#), "test").unwrap();
        post(&h.0, &req(r#"{"key":"filter:reranker/ov","action":"edit","op":"mul","value":2,"path":"L4.o.gain"}"#), "test").unwrap();
        assert_eq!(effective(&h.0, "L4.o.gain"), vec![6.0], "(2 + 1) * 2: entries replay in order");
        // Persistence: the file reads back to the same overlay.
        let o = nmengn::overlay::load(&h.0, K).unwrap().unwrap();
        assert_eq!(o.entries.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), vec!["e1", "e2"]);
        // Undo the newest: the multiply comes out.
        post(&h.0, &req(r#"{"key":"filter:reranker/ov","action":"undo"}"#), "test").unwrap();
        assert_eq!(effective(&h.0, "L4.o.gain"), vec![3.0]);
        // Clear, then undo it through the audit log: both entries back in place.
        post(&h.0, &req(r#"{"key":"filter:reranker/ov","action":"edit","op":"mul","value":2,"path":"L4.o.gain"}"#), "test").unwrap();
        let c = post(&h.0, &req(r#"{"key":"filter:reranker/ov","action":"clear","confirm":true}"#), "test").unwrap();
        assert_eq!(effective(&h.0, "L4.o.gain"), vec![2.0]);
        crate::controls::undo_by(&h.0, &text(&c, "audit_id"), "test").unwrap();
        assert_eq!(effective(&h.0, "L4.o.gain"), vec![6.0], "the cleared entries are back in order");
        assert!(post(&h.0, &req(r#"{"key":"filter:reranker/ov","action":"clear"}"#), "test").unwrap_err().contains("delete gate"));
    }

    #[test]
    fn library_preference_and_adapter_kinds_take_overlays() {
        let h = home("kinds");
        let mut lib = Library::open(&h.0).unwrap();
        let spec = CartridgeSpec { id: "ov-socket".into(), version: 1, family: "learning".into(), domain: "t".into(), scope: "general".into(), model: "qwen3-reranker-0.6b".into(), weights: String::new(), layers: Some((3, 3)), name: "t".into(), description: "t".into(), made_from: "t".into(), effect: "not measured".into() };
        let key = lib.add_cartridge(&spec, &Filter::Points { points: vec![Point { target: "L3.o".into(), gain: vec![2.0, 2.0], bias: vec![0.5, 0.5] }], q8: None }, Value::Null, Vec::new(), false).unwrap();
        let mut p = std::collections::BTreeMap::new();
        p.insert("pref.global".to_owned(), "3/4".to_owned());
        let pk = lib.add_cartridge(&CartridgeSpec { id: "ov-pref".into(), model: "settings".into(), layers: None, family: nmlearn::library::PREFERENCE.into(), ..spec.clone() }, &Filter::Settings(p), Value::Null, Vec::new(), false).unwrap();
        lib.save().unwrap();
        let file = { let c = lib.cartridge(&key).unwrap(); lib.cartridge_dir(c).join(&c.files[0].name) };
        let before = std::fs::read(&file).unwrap();
        post(&h.0, &req(&format!(r#"{{"key":"{key}","action":"mask"}}"#)), "test").unwrap();
        assert!(nmengn::overlay::effective(&h.0, &key).unwrap().whole_masked());
        assert_eq!(std::fs::read(&file).unwrap(), before);
        post(&h.0, &req(&format!(r#"{{"key":"{pk}","action":"edit","op":"set","value":1,"path":"pref.global[0]"}}"#)), "test").unwrap();
        let mut x = [3.0_f32, 4.0];
        nmengn::overlay::effective(&h.0, &pk).unwrap().apply("pref.global", "F32", &mut x);
        assert_eq!(x, [1.0, 4.0]);
        let sec = crate::cparams::section(&h.0, &pk);
        assert_eq!(text(sec.get("overlay").unwrap().get("takes_effect").unwrap(), "when"), "applied at the next call");
    }

    /// The real-model check: Qwen3-Reranker-0.6B's open weights (read from this checkout's
    /// `weights/`, never written) served in this process on an isolated home, processor only. A
    /// mask of layer 5 through the API's own route reaches the served model at the next call
    /// boundary (`settle_overlays`), changes its score, and unmasking returns the score bit for bit.
    #[test]
    #[ignore = "reads the Qwen3-Reranker-0.6B open weights (about 1.2 GB) and runs it on the processor; run with --ignored"]
    fn a_served_reranker_takes_a_layer_mask_live_and_returns_exactly() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
        let weights = repo.join("weights");
        let h = home("live");
        let copy = |from: &Path, to: &Path| {
            std::fs::create_dir_all(to).unwrap();
            for e in std::fs::read_dir(from).unwrap().flatten() {
                if e.path().is_file() {
                    std::fs::copy(e.path(), to.join(e.file_name())).unwrap();
                }
            }
        };
        copy(&repo.join("designs/assemblies"), &h.0.join("designs/assemblies"));
        std::fs::copy(repo.join("designs/placement.json"), h.0.join("designs/placement.json")).unwrap();
        std::fs::write(h.0.join("designs/models.json"), format!("{{\"dir\": {}}}", Value::String(weights.display().to_string()).encode().unwrap())).unwrap();
        std::fs::write(h.0.join("nano.json"), "{}").unwrap();
        let model = "qwen3-reranker-0.6b";
        let file = weights.join(model).join("files/model.safetensors");
        let sha = nmhash::sha256(std::io::BufReader::new(std::fs::File::open(&file).unwrap())).unwrap();
        let _cpu = nmengn::limits::scoped(nmengn::limits::Limits { vram_mb: Some(0), ..nmengn::limits::Limits::AUTO });
        let (q, doc) = ("Where is the ring's head advanced?", "pub struct Ring { items: Vec<u32>, head: usize }\nimpl Ring { pub fn push(&mut self, x: u32) { self.head += 1; } }");
        let score = || {
            let m = nmapi::neural::reranker_model(&h.0).unwrap();
            (nmengn::roles::card_rerank(&m, nmengn::roles::CODE_INSTRUCTION, q, doc).unwrap().0, m.pop.source.clone())
        };
        let (s0, src0) = score();
        assert_eq!(src0, nmengn::model::OPEN_WEIGHTS);
        let post = |b: &str| crate::ui::lent_route(&h.0, "/api/parameters/overlay", "", Some(b.as_bytes()), "test").unwrap();
        let r = post(&format!(r#"{{"key":"open:{model}","action":"mask","path":"L5"}}"#));
        assert_eq!(text(r.get("takes_effect").unwrap(), "when"), "applied at the next call");
        let t0 = std::time::Instant::now();
        while t0.elapsed() < std::time::Duration::from_millis(1100) {
            std::thread::park_timeout(std::time::Duration::from_millis(1100).saturating_sub(t0.elapsed()));
        }
        let lines = nmapi::neural::settle_overlays(&h.0, None);
        assert!(lines.iter().any(|l| l.contains("bound again")), "{lines:?}");
        let (s1, src1) = score();
        assert!(src1.starts_with("open weights with overlay"), "{src1}");
        assert_ne!(s1.to_bits(), s0.to_bits(), "layer 5 masked changes the score");
        let secs = nmengn::overlay::reload_secs(&h.0, model).unwrap();
        post(&format!(r#"{{"key":"open:{model}","action":"unmask","path":"L5"}}"#));
        let t0 = std::time::Instant::now();
        while t0.elapsed() < std::time::Duration::from_millis(1100) {
            std::thread::park_timeout(std::time::Duration::from_millis(1100).saturating_sub(t0.elapsed()));
        }
        let lines = nmapi::neural::settle_overlays(&h.0, None);
        assert!(lines.iter().any(|l| l.contains("bound again")), "{lines:?}");
        let (s2, src2) = score();
        assert_eq!(src2, nmengn::model::OPEN_WEIGHTS);
        assert_eq!(s2.to_bits(), s0.to_bits(), "unmasked, the score is back bit for bit");
        assert_eq!(nmhash::sha256(std::io::BufReader::new(std::fs::File::open(&file).unwrap())).unwrap(), sha, "the open weights' file is unchanged");
        eprintln!("live overlay check: score {s0} -> masked L5 {s1} -> unmasked {s2}; bound again in {secs:.2} s (processor only)");
    }

    /// A tiny safetensors folder as the open weights of `model` under `home/weights`.
    fn open_weights(h: &Path, model: &str) -> PathBuf {
        let dir = h.join("weights").join(model).join("files");
        std::fs::create_dir_all(&dir).unwrap();
        let vals: Vec<u16> = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0].iter().map(|x| (x.to_bits() >> 16) as u16).collect();
        let norm: Vec<u16> = [0.5_f32, 0.25].iter().map(|x| (x.to_bits() >> 16) as u16).collect();
        let header = r#"{"model.layers.0.mlp.down_proj.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]},"model.layers.0.input_layernorm.weight":{"dtype":"BF16","shape":[2],"data_offsets":[12,16]}}"#;
        let mut b = (header.len() as u64).to_le_bytes().to_vec();
        b.extend(header.as_bytes());
        b.extend(vals.iter().chain(&norm).flat_map(|x| x.to_le_bytes()));
        let f = dir.join("model.safetensors");
        std::fs::write(&f, b).unwrap();
        f
    }

    #[test]
    fn open_weights_take_edits_through_the_overlay_and_their_files_stay() {
        let h = home("open");
        let f = open_weights(&h.0, "tiny-open");
        let sha = nmhash::sha256_bytes(&std::fs::read(&f).unwrap());
        let key = "open:tiny-open";
        let pv = post(&h.0, &req(r#"{"key":"open:tiny-open","action":"edit","op":"mul","value":2,"path":"model.layers.0.mlp.down_proj.weight[0:1, 0:3]","preview":true}"#), "test").unwrap();
        assert_eq!(pv.get("diff").unwrap().get("changed").unwrap().integer().unwrap(), 3);
        post(&h.0, &req(r#"{"key":"open:tiny-open","action":"edit","op":"mul","value":2,"path":"model.layers.0.mlp.down_proj.weight[0:1, 0:3]"}"#), "test").unwrap();
        post(&h.0, &req(r#"{"key":"open:tiny-open","action":"mask","path":"L0"}"#), "test").unwrap();
        let o = nmengn::overlay::effective(&h.0, key).unwrap();
        // The engine's patch over the population's descriptors (as `Population::open` lists them).
        let idx = nmsafe::TensorIndex::open(f.parent().unwrap()).unwrap();
        let descs: Vec<nmnpcs::TensorDesc> = idx.tensors.values().map(|t| nmnpcs::TensorDesc { name: t.name.clone(), dtype: t.dtype.clone(), shape: t.shape.iter().map(|x| *x as usize).collect(), file: "model.safetensors".into(), start: t.start, end: t.end, sha256: String::new() }).collect();
        let p = nmengn::overlay::patch_with(&o, &descs, f.parent().unwrap()).unwrap();
        let d = p.descs.iter().find(|d| d.name.ends_with("down_proj.weight")).unwrap();
        let bytes = nmnpcs::provision::held(&d.sha256).unwrap();
        assert!(bytes.iter().all(|b| *b == 0), "a masked projection is zero");
        let n = p.descs.iter().find(|d| d.name.ends_with("input_layernorm.weight")).unwrap();
        let nb = nmnpcs::provision::held(&n.sha256).unwrap();
        assert_eq!(f32::from_bits(u32::from(u16::from_le_bytes([nb[0], nb[1]])) << 16), 1.0, "a masked norm weight is 1");
        // Unmasked, the edit is back: row 0 doubled in bfloat16.
        post(&h.0, &req(r#"{"key":"open:tiny-open","action":"undo"}"#), "test").unwrap();
        let o = nmengn::overlay::effective(&h.0, key).unwrap();
        let p = nmengn::overlay::patch_with(&o, &descs, f.parent().unwrap()).unwrap();
        let d = p.descs.iter().find(|d| d.name.ends_with("down_proj.weight")).unwrap();
        let b = nmnpcs::provision::held(&d.sha256).unwrap();
        let got: Vec<f32> = b.as_chunks::<2>().0.iter().map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16)).collect();
        assert_eq!(got, vec![2.0, 4.0, 6.0, 4.0, 5.0, 6.0]);
        assert_eq!(nmhash::sha256_bytes(&std::fs::read(&f).unwrap()), sha, "the open weights' file is unchanged");
        let sec = crate::cparams::section(&h.0, key);
        assert_eq!(text(sec.get("overlay").unwrap().get("takes_effect").unwrap(), "when"), "applied at the next call");
        let w = post(&h.0, &req(r#"{"key":"open:tiny-open","action":"mask","preview":true}"#), "test").unwrap();
        assert!(w.get("added").unwrap().array().unwrap()[0].get("whole").is_ok());
    }
}
