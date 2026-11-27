//! Learned pieces saved as adapter cartridges (2026-11-12; research file, cartridge section:
//! "adapter cartridge = learned, attached to one base, removable"; disk ruling of Luis Jake
//! Gabriel III, 2026-11-10 19:55 UTC: written quantized parameters are their own store on disk).
//!
//! A piece of any learning layer that passes its own gate (the L4 calibrator, the L1 settings,
//! later L5 vectors and adapters) is written once to `units/learned/<id>/`:
//! - `cartridge.json`: its label (`nm-cartridge/1`, kind learning-unit, class behavioural, the
//!   base it fits by hash, the component it attaches to, check `accept`);
//! - `numbers.q8`: its numbers at 8 bits, one f16 scale per row of 64 (`nmq8-row-f16`);
//! - `recipe.json`: how to fit it again (signal, method, settings, the episodes or decisions it
//!   was fitted on), its gate report, the shape, the SHA-256 of `numbers.q8` and the largest
//!   rounding error.
//!   A piece that did not pass its gate is refused; a folder is never overwritten. Loading checks
//!   the label, the base hash against what is loaded, and the file hash; a refit from the recipe
//!   must land within the rounding error of the saved numbers.
use crate::calibrate::Calibrator;
use crate::cartridge::{CartFault, DataClass, Kind, Label, Loaded, Policy, date_of, slot_check};
use crate::unit::{n, obj, s};
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::path::{Path, PathBuf};

/// Numbers per scaled row.
pub const ROW: usize = 64;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmlearn.learned", FaultKind::Policy, m)
}
fn io(m: impl Into<String>) -> Fault {
    Fault::new("nmlearn.learned", FaultKind::Resource, m)
}

/// The store folder of learned pieces under the mini's home.
#[must_use]
pub fn store(home: &Path) -> PathBuf {
    home.join("units").join("learned")
}

/// IEEE binary16 from f32 (round to nearest even; overflow to infinity, tiny to zero).
#[must_use]
pub fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man == 0 { 0 } else { 0x200 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = 1_u32 << (shift - 1);
        let r = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let r = if rem > half || (rem == half && r & 1 == 1) { r + 1 } else { r };
        return sign | r as u16;
    }
    let r = man >> 13;
    let rem = man & 0x1fff;
    let mut v = ((e as u32) << 10) | r;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        v += 1;
    }
    sign | v as u16
}

/// f32 from IEEE binary16.
#[must_use]
pub fn f16_value(h: u16) -> f32 {
    let sign = if h & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let man = f32::from(h & 0x3ff);
    match exp {
        0 => sign * man * 2f32.powi(-24),
        0x1f => {
            if man == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// Numbers at 8 bits: per row of `ROW`, an f16 scale (little-endian) then the row's signed bytes.
#[must_use]
pub fn q8_encode(xs: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() + xs.len().div_ceil(ROW) * 2);
    for row in xs.chunks(ROW) {
        let m = row.iter().fold(0.0_f32, |a, x| a.max(x.abs()));
        let h = f16_bits(m / 127.0);
        let sc = f16_value(h);
        out.extend_from_slice(&h.to_le_bytes());
        for x in row {
            #[expect(clippy::cast_possible_truncation, reason = "clamped to the i8 range")]
            let q = if sc > 0.0 { (x / sc).round().clamp(-127.0, 127.0) as i8 } else { 0 };
            out.push(q.to_le_bytes()[0]);
        }
    }
    out
}

/// Decode `count` numbers of `q8_encode`.
/// # Errors
/// The bytes are shorter than `count` numbers need.
pub fn q8_decode(b: &[u8], count: usize) -> Result<Vec<f32>, Fault> {
    let mut out = Vec::with_capacity(count);
    let mut at = 0;
    while out.len() < count {
        let k = (count - out.len()).min(ROW);
        let (Some(&lo), Some(&hi)) = (b.get(at), b.get(at + 1)) else { return Err(io("numbers.q8 is short")) };
        let sc = f16_value(u16::from_le_bytes([lo, hi]));
        let row = b.get(at + 2..at + 2 + k).ok_or_else(|| io("numbers.q8 is short"))?;
        out.extend(row.iter().map(|q| f32::from(i8::from_le_bytes([*q])) * sc));
        at += 2 + k;
    }
    if at != b.len() {
        return Err(io("numbers.q8 has trailing bytes"));
    }
    Ok(out)
}

/// One learned piece.
#[derive(Clone, Debug, PartialEq)]
pub struct Piece {
    pub label: Label,
    pub shape: Vec<usize>,
    pub numbers: Vec<f32>,
    /// How it was fitted (signal, method, settings, data ids) and its gate report.
    pub recipe: Value,
}

/// A label for a learned piece fitted against `loaded`, attached to `component`.
#[must_use]
pub fn label(id: &str, loaded: &Loaded, component: &str, description: &str) -> Label {
    Label {
        id: id.to_owned(),
        kind: Kind::LearningUnit,
        data_class: DataClass::Behavioural,
        base: loaded.base.clone(),
        base_sha256: loaded.base_sha256.clone(),
        components: vec![component.to_owned()],
        scopes: vec!["*".into()],
        quantization: "nmq8-row-f16".into(),
        description: description.to_owned(),
        version: 1,
        check: "accept".into(),
        date: date_of(nmtype::now_ms()),
        loaded: false,
    }
}

/// The next free id `<prefix>-<nnn>` in the store.
#[must_use]
pub fn next_id(dir: &Path, prefix: &str) -> String {
    let taken = std::fs::read_dir(dir)
        .map(|r| {
            r.flatten()
                .filter_map(|e| {
                    e.file_name().to_str().and_then(|n| n.strip_prefix(&format!("{prefix}-"))).and_then(|x| x.parse::<u32>().ok())
                })
                .max()
                .unwrap_or(0)
        })
        .unwrap_or(0);
    format!("{prefix}-{:03}", taken + 1)
}

/// Save a piece that passed its gate, once. Returns its folder.
/// # Errors
/// The gate did not pass, the label is invalid, the folder exists, or a write fails.
pub fn save(dir: &Path, piece: &Piece, gate_passed: bool, policy: &Policy) -> Result<PathBuf, Fault> {
    if !gate_passed {
        return Err(bad(format!("{}: its gate did not pass; nothing is written", piece.label.id)));
    }
    piece.label.validate(policy).map_err(|e| e.to_fault())?;
    if piece.label.check != "accept" || piece.shape.iter().product::<usize>() != piece.numbers.len() {
        return Err(bad(format!("{}: check must be accept and the shape must hold the numbers", piece.label.id)));
    }
    let out = dir.join(&piece.label.id);
    if out.exists() {
        return Err(bad(format!("{} exists; learned pieces are written once", out.display())));
    }
    let q = q8_encode(&piece.numbers);
    let back = q8_decode(&q, piece.numbers.len())?;
    let err = piece.numbers.iter().zip(&back).fold(0.0_f32, |a, (x, y)| a.max((x - y).abs()));
    let mut recipe = piece.recipe.clone();
    if let Value::Object(m) = &mut recipe {
        m.insert("shape".into(), Value::Array(piece.shape.iter().map(|x| n(*x as u64)).collect()));
        m.insert("numbers_sha256".into(), s(&nmhash::sha256_bytes(&q)));
        m.insert("rounding_error_max".into(), Value::Number(format!("{err:e}")));
    } else {
        return Err(bad("the recipe must be an object"));
    }
    let tmp = dir.join(format!(".{}.writing", piece.label.id));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).map_err(|e| io(e.to_string()))?;
    let w = |name: &str, bytes: &[u8]| std::fs::write(tmp.join(name), bytes).map_err(|e| io(format!("{name}: {e}")));
    w("numbers.q8", &q)?;
    w("cartridge.json", nmjson::pretty::pretty(&piece.label.to_value()).as_bytes())?;
    w("recipe.json", nmjson::pretty::pretty(&recipe).as_bytes())?;
    std::fs::rename(&tmp, &out).map_err(|e| io(e.to_string()))?;
    Ok(out)
}

/// Load a saved piece for what is loaded now.
/// # Errors
/// A label, base, component or file-hash fault, by name.
pub fn load(dir: &Path, id: &str, policy: &Policy, loaded: &Loaded) -> Result<Piece, Fault> {
    let at = dir.join(id);
    let label = Label::load(&at.join("cartridge.json"), policy).map_err(|e| e.to_fault())?;
    slot_check(&label, loaded).map_err(|e| e.to_fault())?;
    let recipe = std::fs::read(at.join("recipe.json"))
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .ok_or_else(|| CartFault::Unreadable(format!("{id}/recipe.json")).to_fault())?;
    let q = std::fs::read(at.join("numbers.q8")).map_err(|e| io(format!("{id}/numbers.q8: {e}")))?;
    let want = recipe.get("numbers_sha256").and_then(Value::string).unwrap_or_default();
    if nmhash::sha256_bytes(&q) != want {
        return Err(Fault::new("nmlearn.learned", FaultKind::Integrity, format!("{id}: numbers.q8 does not match its recorded hash")));
    }
    let shape: Vec<usize> = recipe
        .get("shape")
        .and_then(Value::array)
        .map(|a| a.iter().filter_map(|x| x.integer().ok().and_then(|i| usize::try_from(i).ok())).collect())
        .unwrap_or_default();
    let numbers = q8_decode(&q, shape.iter().product())?;
    Ok(Piece { label, shape, numbers, recipe })
}

/// Every saved piece's id, sorted.
#[must_use]
pub fn list(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .map(|r| {
            r.flatten()
                .filter(|e| e.path().join("cartridge.json").is_file())
                .filter_map(|e| e.file_name().to_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

/// A calibrator as numbers: (weights with the bias last, means, spreads), shape [3, d + 1]
/// (means and spreads padded with 0 and 1 in the bias column).
#[must_use]
pub fn calibrator_numbers(c: &Calibrator) -> (Vec<usize>, Vec<f32>) {
    let k = c.w.len();
    #[expect(clippy::cast_possible_truncation, reason = "stored at 8 bits")]
    let cast = |v: &[f64], pad: f64| (0..k).map(|i| v.get(i).copied().unwrap_or(pad) as f32).collect::<Vec<f32>>();
    let mut out = cast(&c.w, 0.0);
    out.extend(cast(&c.mean, 0.0));
    out.extend(cast(&c.sd, 1.0));
    (vec![3, k], out)
}

/// The calibrator of `calibrator_numbers`.
#[must_use]
pub fn calibrator_of(shape: &[usize], xs: &[f32]) -> Calibrator {
    let k = shape.get(1).copied().unwrap_or(0);
    let row = |r: usize, len: usize| xs.iter().skip(r * k).take(len).map(|x| f64::from(*x)).collect::<Vec<f64>>();
    Calibrator {
        w: row(0, k),
        mean: row(1, k.saturating_sub(1)),
        sd: row(2, k.saturating_sub(1)).into_iter().map(|x| x.max(1e-9)).collect(),
    }
}

/// Section check: f16 and 8-bit round trips; a fitted calibrator saved and loaded predicts within
/// 0.02 of the original; a refit from the same data lands within the rounding error; a piece
/// whose gate failed is refused; a second save under the same id is refused; a changed byte is
/// refused by its hash; another base is refused by the slot check.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    for x in [0.0_f32, 1.0, -2.5, 0.000_123, 65_504.0, 1e-8] {
        let y = f16_value(f16_bits(x));
        if (x - y).abs() > x.abs() * 1e-3 + 6e-8 {
            return Err(format!("nmlearn.learned: f16 {x} -> {y}"));
        }
    }
    let xs: Vec<f32> = (0..150).map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0).collect();
    let back = q8_decode(&q8_encode(&xs), xs.len()).map_err(|e| e.to_string())?;
    if xs.iter().zip(&back).any(|(a, b)| (a - b).abs() > 50.0 / 7.0 / 127.0 * 0.51 + 1e-3) {
        return Err("nmlearn.learned: 8-bit round trip error above half a step".into());
    }
    // A calibrator on separable data.
    let feats: Vec<Vec<f64>> = (0..40).map(|i| vec![f64::from(i) / 10.0 - 2.0, f64::from(i % 7) / 7.0]).collect();
    let ys: Vec<bool> = (0..40).map(|i| i >= 20).collect();
    let c = Calibrator::fit(&feats, &ys, 1.0);
    let (shape, nums) = calibrator_numbers(&c);
    let dir = nmtype::check_dir("nmlearn-learned").map_err(|e| e.to_string())?;
    let policy = Policy::fallback("check");
    let loaded =
        Loaded { base: "qwen3-reranker-0.6b".into(), base_sha256: "a".repeat(64), components: vec!["nmbrain.core-brain#0".into()] };
    let id = next_id(&dir, "l4-calibrator");
    let recipe = obj(vec![
        ("signal", s("verdict")),
        ("method", s("newton ridge 1.0")),
        ("data", Value::Array((0..40).map(|i| s(&format!("ep{i}"))).collect())),
    ]);
    let piece =
        Piece { label: label(&id, &loaded, "nmbrain.core-brain#0", "L4 read-out calibrator (check)"), shape, numbers: nums, recipe };
    if save(&dir, &piece, false, &policy).is_ok() {
        return Err("nmlearn.learned: a piece whose gate failed was saved".into());
    }
    save(&dir, &piece, true, &policy).map_err(|e| e.to_string())?;
    if save(&dir, &piece, true, &policy).is_ok() {
        return Err("nmlearn.learned: a piece was written twice".into());
    }
    let got = load(&dir, &id, &policy, &loaded).map_err(|e| e.to_string())?;
    let c2 = calibrator_of(&got.shape, &got.numbers);
    let worst = feats.iter().map(|x| (c.predict(x) - c2.predict(x)).abs()).fold(0.0, f64::max);
    if worst > 0.02 {
        return Err(format!("nmlearn.learned: saved calibrator drifts {worst}"));
    }
    let refit = calibrator_numbers(&Calibrator::fit(&feats, &ys, 1.0)).1;
    let bound = got
        .recipe
        .get("rounding_error_max")
        .ok()
        .and_then(|v| if let Value::Number(t) = v { t.parse::<f32>().ok() } else { None })
        .unwrap_or(0.0);
    if refit.iter().zip(&got.numbers).any(|(a, b)| (a - b).abs() > bound + 1e-6) {
        return Err("nmlearn.learned: the refit from the recipe left the rounding error".into());
    }
    let other = Loaded { base_sha256: "b".repeat(64), ..loaded.clone() };
    if load(&dir, &id, &policy, &other).is_ok() {
        return Err("nmlearn.learned: loaded against another base".into());
    }
    let p = dir.join(&id).join("numbers.q8");
    let mut b = std::fs::read(&p).map_err(|e| e.to_string())?;
    if let Some(x) = b.get_mut(3) {
        *x ^= 1;
    }
    std::fs::write(&p, b).map_err(|e| e.to_string())?;
    if load(&dir, &id, &policy, &loaded).is_ok() {
        return Err("nmlearn.learned: a changed numbers file loaded".into());
    }
    if list(&dir) != vec![id.clone()] || next_id(&dir, "l4-calibrator") != "l4-calibrator-002" {
        return Err(format!("nmlearn.learned: list {:?}", list(&dir)));
    }
    Ok(())
}
