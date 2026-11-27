//! The comparative map (atlas step 0, cartridge build step B): sibling populations released
//! from one base share its architecture; where a sibling's weights moved furthest from the
//! base is where its own training specialised it. For every atlas piece the relative
//! difference `||W_sibling - W_base|| / ||W_base||` (Frobenius, over the piece's rows and
//! columns) is recorded as `divergence[sibling]`, job `comparative`. Read-only, processor only,
//! no forward pass; tensors are read from the released files in place, eight at a time.
use crate::atlas::{Atlas, BLOCK, Coordinate, Piece, Record};
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::collections::BTreeMap;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmnpcs.compare", FaultKind::Config, m)
}

/// Attention shape of the populations compared.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub heads: usize,
    pub kv: usize,
    pub head_dim: usize,
}

#[derive(Clone, Debug)]
struct Tensor {
    dtype: String,
    shape: Vec<usize>,
    start: u64,
    end: u64,
}

/// Tensor names without the `model.` prefix some releases add.
fn canon(name: &str) -> String {
    name.strip_prefix("model.").unwrap_or(name).to_owned()
}

/// The safetensors header of one file: canonical name to tensor (absolute byte offsets).
fn header(path: &Path) -> Result<BTreeMap<String, Tensor>, Fault> {
    let mut f = std::fs::File::open(path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
    let mut n = [0_u8; 8];
    f.read_exact(&mut n).map_err(|e| bad(e.to_string()))?;
    let n = u64::from_le_bytes(n);
    if n > 64 << 20 {
        return Err(bad(format!("{}: header of {n} bytes", path.display())));
    }
    let mut h = vec![0_u8; usize::try_from(n).map_err(|e| bad(e.to_string()))?];
    f.read_exact(&mut h).map_err(|e| bad(e.to_string()))?;
    let v = nmjson::parse(&h).map_err(|e| bad(e.to_string()))?;
    let mut out = BTreeMap::new();
    for (k, t) in v.object().map_err(|e| bad(e.to_string()))? {
        if k == "__metadata__" {
            continue;
        }
        let dtype = t.get("dtype").and_then(Value::string).map_err(|e| bad(format!("{k}: {e}")))?.to_owned();
        let shape = t
            .get("shape")
            .and_then(Value::array)
            .map_err(|e| bad(format!("{k}: {e}")))?
            .iter()
            .map(|x| x.integer().map(|u| usize::try_from(u).unwrap_or(0)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| bad(e.to_string()))?;
        let off = t.get("data_offsets").and_then(Value::array).map_err(|e| bad(format!("{k}: {e}")))?;
        let (Some(a), Some(b)) = (off.first().and_then(|x| x.integer().ok()), off.get(1).and_then(|x| x.integer().ok())) else {
            return Err(bad(format!("{k}: data_offsets")));
        };
        out.insert(canon(k), Tensor { dtype, shape, start: 8 + n + a, end: 8 + n + b });
    }
    Ok(out)
}

fn read(path: &Path, t: &Tensor) -> Result<Vec<u8>, Fault> {
    let mut f = std::fs::File::open(path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
    f.seek(SeekFrom::Start(t.start)).map_err(|e| bad(e.to_string()))?;
    let mut b = vec![0_u8; usize::try_from(t.end - t.start).map_err(|e| bad(e.to_string()))?];
    f.read_exact(&mut b).map_err(|e| bad(e.to_string()))?;
    Ok(b)
}

fn bf16(b: &[u8], i: usize) -> f64 {
    let lo = b.get(2 * i).copied().unwrap_or(0);
    let hi = b.get(2 * i + 1).copied().unwrap_or(0);
    f64::from(f32::from_bits(u32::from(u16::from_le_bytes([lo, hi])) << 16))
}

/// How a tensor's rows or columns fall into atlas pieces.
enum Cut {
    /// The whole tensor counts for these pieces.
    Whole(Vec<Piece>),
    /// Each row goes to the pieces the function names (plus `also`).
    Rows(Box<dyn Fn(usize) -> Vec<Piece> + Send + Sync>, Vec<Piece>),
    /// Each column goes to the pieces the function names (plus `also`).
    Cols(Box<dyn Fn(usize) -> Vec<Piece> + Send + Sync>, Vec<Piece>),
}

fn cut(name: &str, s: Shape) -> Option<Cut> {
    match name {
        "embed_tokens.weight" => return Some(Cut::Whole(vec![Piece::Table])),
        "lm_head.weight" | "norm.weight" => return Some(Cut::Whole(vec![Piece::Head])),
        _ => {}
    }
    let rest = name.strip_prefix("layers.")?;
    let (l, part) = rest.split_once('.')?;
    let l: u16 = l.parse().ok()?;
    let (hd, per) = (s.head_dim.max(1), (s.heads / s.kv.max(1)).max(1));
    let q = move |i: usize| {
        let h = i / hd;
        let g = h / per;
        let (g, h) = (u16::try_from(g).unwrap_or(u16::MAX), u16::try_from(h).unwrap_or(u16::MAX));
        vec![Piece::QHead(l, g, h), Piece::Kv(l, g)]
    };
    let kv = move |i: usize| vec![Piece::Kv(l, u16::try_from(i / hd).unwrap_or(u16::MAX))];
    let blk = move |i: usize| vec![Piece::Block(l, u32::try_from(i).unwrap_or(u32::MAX) / BLOCK)];
    let attn = vec![Piece::Attn(l), Piece::Layer(l)];
    let ffn = vec![Piece::Ffn(l), Piece::Layer(l)];
    Some(match part {
        "self_attn.q_proj.weight" => Cut::Rows(Box::new(q), attn),
        "self_attn.o_proj.weight" => Cut::Cols(Box::new(q), attn),
        "self_attn.k_proj.weight" | "self_attn.v_proj.weight" => Cut::Rows(Box::new(kv), attn),
        "self_attn.q_norm.weight" | "self_attn.k_norm.weight" | "input_layernorm.weight" => Cut::Whole(attn),
        "mlp.gate_proj.weight" | "mlp.up_proj.weight" => Cut::Rows(Box::new(blk), ffn),
        "mlp.down_proj.weight" => Cut::Cols(Box::new(blk), ffn),
        "post_attention_layernorm.weight" => Cut::Whole(ffn),
        _ => return None,
    })
}

/// Sums per piece: `[base squared, then difference squared per sibling]`.
type Sums = BTreeMap<Piece, Vec<f64>>;

fn add(m: &mut Sums, p: &Piece, v: &[f64]) {
    let e = m.entry(p.clone()).or_insert_with(|| vec![0.0; v.len()]);
    for (a, b) in e.iter_mut().zip(v) {
        *a += *b;
    }
}

/// One tensor's contribution. `sibs` holds each sibling's bytes of the same tensor.
fn one(name: &str, t: &Tensor, base: &[u8], sibs: &[Vec<u8>], s: Shape, out: &mut Sums) {
    let Some(c) = cut(name, s) else { return };
    let (rows, cols) = match t.shape.as_slice() {
        [n] => (1, *n),
        [r, c] => (*r, *c),
        _ => (1, t.shape.iter().product()),
    };
    let k = 1 + sibs.len();
    let by_col = matches!(c, Cut::Cols(..));
    let lines = if by_col { cols } else { rows };
    let mut acc = vec![0.0_f64; lines * k];
    for r in 0..rows {
        for col in 0..cols {
            let i = r * cols + col;
            let b = bf16(base, i);
            let line = if by_col { col } else { r };
            let slot = line * k;
            if let Some(x) = acc.get_mut(slot) {
                *x += b * b;
            }
            for (j, sb) in sibs.iter().enumerate() {
                let d = bf16(sb, i) - b;
                if let Some(x) = acc.get_mut(slot + 1 + j) {
                    *x += d * d;
                }
            }
        }
    }
    let mut total = vec![0.0_f64; k];
    for line in acc.chunks(k) {
        for (a, b) in total.iter_mut().zip(line) {
            *a += *b;
        }
    }
    match c {
        Cut::Whole(ps) => ps.iter().for_each(|p| add(out, p, &total)),
        Cut::Rows(f, also) | Cut::Cols(f, also) => {
            also.iter().for_each(|p| add(out, p, &total));
            for (line, v) in acc.chunks(k).enumerate() {
                for p in f(line) {
                    add(out, &p, v);
                }
            }
        }
    }
}

/// Compare `base` (id, main file, SHA-256) with each sibling (id, main file) piece by piece.
/// A sibling without an output head of its own (tied head) is compared on its token table.
/// # Errors
/// An unreadable file, or a tensor whose type or shape differs between populations.
pub fn compare(base: (&str, &Path, &str), sibs: &[(String, PathBuf)], s: Shape, run: &str, date: &str) -> Result<Atlas, Fault> {
    let (bid, bpath, bhash) = base;
    let bh = header(bpath)?;
    let shs = sibs.iter().map(|(_, p)| header(p)).collect::<Result<Vec<_>, _>>()?;
    let names: Vec<&String> = bh.keys().collect();
    let next = AtomicUsize::new(0);
    let total: Mutex<Sums> = Mutex::new(BTreeMap::new());
    let fault: Mutex<Option<Fault>> = Mutex::new(None);
    let threads = std::thread::available_parallelism().map_or(8, std::num::NonZero::get).min(8);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                let mut mine: Sums = BTreeMap::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(name) = names.get(i) else { break };
                    let Some(t) = bh.get(*name) else { continue };
                    let mut run_one = || -> Result<(), Fault> {
                        if t.dtype != "BF16" {
                            return Err(bad(format!("{name}: {} (BF16 expected)", t.dtype)));
                        }
                        let mut b = read(bpath, t)?;
                        let mut sb = Vec::new();
                        // The token table and output head may carry padding rows in one release
                        // and not another: those two are compared over the rows all share.
                        let vocab = matches!(name.as_str(), "embed_tokens.weight" | "lm_head.weight");
                        let mut shape = t.shape.clone();
                        for ((sid, sp), sh) in sibs.iter().zip(&shs) {
                            let st = sh
                                .get(*name)
                                .or_else(|| if name.as_str() == "lm_head.weight" { sh.get("embed_tokens.weight") } else { None })
                                .ok_or_else(|| bad(format!("{sid}: no tensor {name}")))?;
                            let rows_only = vocab && st.shape.len() == 2 && st.shape.get(1) == t.shape.get(1);
                            if (st.shape != t.shape && !rows_only) || st.dtype != t.dtype {
                                return Err(bad(format!("{sid}: {name}: shape or type differs from the base")));
                            }
                            if let (Some(r), Some(sr)) = (shape.first_mut(), st.shape.first()) {
                                *r = (*r).min(*sr);
                            }
                            sb.push(read(sp, st)?);
                        }
                        let keep = shape.iter().product::<usize>() * 2;
                        b.truncate(keep);
                        sb.iter_mut().for_each(|x| x.truncate(keep));
                        let t = Tensor { shape, ..t.clone() };
                        one(name, &t, &b, &sb, s, &mut mine);
                        Ok(())
                    };
                    if let Err(e) = run_one() {
                        if let Ok(mut f) = fault.lock() {
                            f.get_or_insert(e);
                        }
                        break;
                    }
                }
                if let Ok(mut t) = total.lock() {
                    for (p, v) in &mine {
                        add(&mut t, p, v);
                    }
                }
            });
        }
    });
    if let Some(e) = fault.into_inner().map_err(|_| bad("a worker panicked"))? {
        return Err(e);
    }
    let total = total.into_inner().map_err(|_| bad("a worker panicked"))?;
    let hash: String = bhash.chars().take(12).collect();
    let records = total
        .into_iter()
        .filter(|(_, v)| v.first().is_some_and(|b| *b > 0.0))
        .map(|(p, v)| {
            let mut r = Record::new(Coordinate { model: bid.to_owned(), hash: hash.clone(), piece: p }, "comparative", run, date);
            let b = v.first().copied().unwrap_or(1.0);
            for ((sid, _), d) in sibs.iter().zip(v.iter().skip(1)) {
                // Rounded to 6 significant digits on purpose: a record, not a computation input.
                #[expect(clippy::cast_possible_truncation, reason = "a ratio near 0 to 2 fits f32")]
                r.divergence.insert(sid.clone(), (d / b).sqrt() as f32);
            }
            r
        })
        .collect();
    Ok(Atlas {
        population: bid.to_owned(),
        base_sha256: bhash.to_owned(),
        calibration_sha256: "none (no forward pass)".into(),
        version: 1,
        records,
    })
}

fn tensor_bytes(name: &str, shape: &[usize], vals: &[f32], at: &mut Vec<u8>, head: &mut Vec<(String, String)>) {
    let start = at.len();
    for v in vals {
        at.extend_from_slice(&u16::try_from(v.to_bits() >> 16).unwrap_or(0).to_le_bytes());
    }
    let dims = shape.iter().map(ToString::to_string).collect::<Vec<_>>().join(",");
    head.push((name.to_owned(), format!("{{\"dtype\":\"BF16\",\"shape\":[{dims}],\"data_offsets\":[{start},{}]}}", at.len())));
}

fn toy(path: &Path, q_head1: f32, k_scale: f32, prefix: &str) -> Result<(), String> {
    // heads 2, kv 1, head_dim 2, hidden 2: q [4,2], k [2,2].
    let mut data = Vec::new();
    let mut head = Vec::new();
    tensor_bytes(
        &format!("{prefix}layers.0.self_attn.q_proj.weight"),
        &[4, 2],
        &[1.0, 1.0, 1.0, 1.0, q_head1, q_head1, q_head1, q_head1],
        &mut data,
        &mut head,
    );
    tensor_bytes(&format!("{prefix}layers.0.self_attn.k_proj.weight"), &[2, 2], &[k_scale; 4], &mut data, &mut head);
    let json = format!("{{{}}}", head.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect::<Vec<_>>().join(","));
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(json.as_bytes());
    bytes.extend_from_slice(&data);
    std::fs::write(path, bytes).map_err(|e| e.to_string())
}

/// Self-check on two toy files: a sibling equal to the base except query head 1 (doubled)
/// gives divergence 0 on head 0, exactly 1 on head 1, 0 on the key rows; the `model.` prefix
/// difference is matched; a shape mismatch is refused.
pub(crate) fn check() -> Result<(), String> {
    let dir = nmtype::check_dir("nmnpcs-compare").map_err(|e| e.to_string())?;
    let (a, b) = (dir.join("base.safetensors"), dir.join("sib.safetensors"));
    toy(&a, 1.0, 0.5, "model.")?;
    toy(&b, 2.0, 0.5, "")?;
    let s = Shape { heads: 2, kv: 1, head_dim: 2 };
    let atlas =
        compare(("toy", &a, "abcdef0123456789"), &[("sib".into(), b.clone())], s, "check", "2026-11-11").map_err(|e| e.to_string())?;
    let d = |p: Piece| atlas.records.iter().find(|r| r.at.piece == p).and_then(|r| r.divergence.get("sib").copied());
    if d(Piece::QHead(0, 0, 0)) != Some(0.0) || d(Piece::QHead(0, 0, 1)) != Some(1.0) {
        return Err(format!("nmnpcs.compare: head divergences {:?} {:?}", d(Piece::QHead(0, 0, 0)), d(Piece::QHead(0, 0, 1))));
    }
    // Kv(0,0) holds q (4 rows, base energy 8 with 4 rows doubled giving diff 4) and k (energy 1, diff 0): sqrt(4/9).
    let want = (4.0_f32 / 9.0).sqrt();
    if d(Piece::Kv(0, 0)).is_none_or(|x| (x - want).abs() > 1e-6) {
        return Err(format!("nmnpcs.compare: kv divergence {:?}, expected {want}", d(Piece::Kv(0, 0))));
    }
    let c = dir.join("bad.safetensors");
    let mut data = Vec::new();
    let mut head = Vec::new();
    tensor_bytes("layers.0.self_attn.q_proj.weight", &[2, 2], &[1.0; 4], &mut data, &mut head);
    let json = format!("{{{}}}", head.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect::<Vec<_>>().join(","));
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(json.as_bytes());
    bytes.extend_from_slice(&data);
    std::fs::write(&c, bytes).map_err(|e| e.to_string())?;
    if compare(("toy", &a, "ab"), &[("bad".into(), c)], s, "check", "2026-11-11").is_ok() {
        return Err("nmnpcs.compare: a shape mismatch was accepted".into());
    }
    Ok(())
}

/// Run the comparative map for `base` against `sibs` (population ids under `weights/`): write
/// the atlas to `designs/parmap/<base>.json` and a report to `runs/ATLAS-compare-<stamp>.txt`;
/// return both paths and the report text.
/// # Errors
/// A missing card or file, or any `compare` fault.
pub fn run(home: &Path, base: &str, sibs: &[String], stamp: u64, date: &str) -> Result<(PathBuf, PathBuf, String), Fault> {
    let card = crate::acquire::WeightCard::load(home, base)?;
    let s = Shape { heads: card.int("num_attention_heads")?, kv: card.int("num_key_value_heads")?, head_dim: card.int("head_dim")? };
    let mut with = Vec::new();
    for id in sibs {
        with.push((id.clone(), crate::acquire::WeightCard::load(home, id)?.main_path()));
    }
    let report = home.join("runs").join(format!("ATLAS-compare-{stamp}.txt"));
    let rel = format!("runs/ATLAS-compare-{stamp}.txt");
    let t0 = std::time::Instant::now();
    let atlas = compare((base, &card.main_path(), &card.sha_main), &with, s, &rel, date)?;
    let secs = t0.elapsed().as_secs_f64();
    let dir = home.join("designs").join("parmap");
    std::fs::create_dir_all(&dir).map_err(|e| bad(e.to_string()))?;
    let out = dir.join(format!("{base}.json"));
    std::fs::write(&out, atlas.to_text()).map_err(|e| bad(e.to_string()))?;
    let mut txt = format!(
        "Atlas step 0, comparative map ({date}). Base {base} ({}), siblings {}. Relative weight difference per piece, ||W_sibling - W_base|| / ||W_base||. {} records in {secs:.1} s; atlas {}.\n",
        card.sha_main,
        sibs.join(", "),
        atlas.records.len(),
        out.display()
    );
    for (sid, _) in &with {
        let div =
            |p: &Piece| atlas.records.iter().find(|r| &r.at.piece == p).and_then(|r| r.divergence.get(sid).copied()).unwrap_or(f32::NAN);
        txt.push_str(&format!(
            "\n== {sid}\ntable {:.4}  head {:.4}\nper layer (layer / attention / feed-forward):\n",
            div(&Piece::Table),
            div(&Piece::Head)
        ));
        let layers = atlas.records.iter().filter_map(|r| if let Piece::Layer(l) = r.at.piece { Some(l) } else { None }).collect::<Vec<_>>();
        for l in layers {
            txt.push_str(&format!("  L{l:<2} {:.4}  {:.4}  {:.4}\n", div(&Piece::Layer(l)), div(&Piece::Attn(l)), div(&Piece::Ffn(l))));
        }
        for level in ["kv-group", "head", "block"] {
            let mut v: Vec<(f32, String)> = atlas
                .records
                .iter()
                .filter(|r| r.at.piece.level() == level)
                .filter_map(|r| r.divergence.get(sid).map(|d| (*d, r.at.to_string())))
                .collect();
            v.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mean = v.iter().map(|x| f64::from(x.0)).sum::<f64>() / v.len().max(1) as f64;
            txt.push_str(&format!("top 12 {level} pieces (of {}, mean {mean:.4}):\n", v.len()));
            for (d, c) in v.iter().take(12) {
                txt.push_str(&format!("  {d:.4}  {c}\n"));
            }
        }
    }
    std::fs::write(&report, &txt).map_err(|e| bad(e.to_string()))?;
    Ok((out, report, txt))
}
