//! The engine reference (S6 runbook gates 3 and 4, S7 gate 1): tokenizer cases, single-layer
//! outputs and full forwards of Qwen3-0.6B, recorded by the mini's own engine on CPU into
//! `tests/engine-reference/qwen3-0.6b/` (`record`). It replaced the earlier external recording on
//! 2026-11-08 (Luis), after the engine had matched that recording on CPU and GPU; the file layout
//! is the same, so `tokenizer`, `block` and `prompt` compare against it unchanged. Tolerances are
//! read from its `tolerance.json` and never changed here. Hashes: `INTEGRITY.md`.
use crate::assembly::nid;
use crate::model::{Act, Model, RunOpts, values};
use crate::placement::Device;
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::path::{Path, PathBuf};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmengn.reference", FaultKind::Integrity, m)
}

/// The engine reference folder inside the mini.
#[must_use]
pub fn root(home: &Path) -> PathBuf {
    home.join("tests/engine-reference/qwen3-0.6b")
}
fn bytes(p: &Path) -> Result<Vec<u8>, Fault> {
    std::fs::read(p).map_err(|e| bad(format!("{}: {e}", p.display())))
}
fn f16s(p: &Path) -> Result<Vec<f32>, Fault> {
    Ok(bytes(p)?.as_chunks::<2>().0.iter().map(|c| crate::half::from16(u16::from_le_bytes(*c))).collect())
}
fn f32s(p: &Path) -> Result<Vec<f32>, Fault> {
    Ok(bytes(p)?.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}
fn i32s(p: &Path) -> Result<Vec<i32>, Fault> {
    Ok(bytes(p)?.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect())
}
/// The recorded tolerance for a reference file key (for example `tblock-forward/block0-on.f16`).
/// # Errors
/// The key is not in the tolerance file.
pub fn atol(home: &Path, key: &str) -> Result<f32, Fault> {
    atol_in(&root(home), key)
}
fn atol_in(dir: &Path, key: &str) -> Result<f32, Fault> {
    let v = nmjson::parse(&bytes(&dir.join("tolerance.json"))?).map_err(|e| bad(e.to_string()))?;
    match v.get("files").and_then(|f| f.get(key)).and_then(|f| f.get("atol")) {
        Ok(Value::Number(n)) => n.parse().map_err(|_| bad(format!("tolerance {key}"))),
        _ => Err(bad(format!("no tolerance for {key}"))),
    }
}
fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v.get(best).copied().unwrap_or(f32::NEG_INFINITY) {
            best = i;
        }
    }
    best
}

/// One comparison: what, the measured difference, the tolerance, and pass.
#[derive(Clone, Debug, PartialEq)]
pub struct Case {
    pub name: String,
    pub max_abs: f32,
    pub atol: f32,
    pub extra: String,
    pub pass: bool,
    pub ms: f64,
}

/// The single-layer cases the S6 parity run compares: (layer, query heads off, file).
pub const BLOCKS: [(usize, &[usize], &str); 4] =
    [(0, &[], "block0-on.f16"), (13, &[], "block13-on.f16"), (13, &[0, 5], "block13-off.f16"), (27, &[], "block27-on.f16")];
/// The recorded prompts (0 and 1) and the greedy length of each.
pub const PROMPTS: usize = 2;
const GREEDY: usize = 64;

/// Tokenizer parity on every recorded text: (exact, total, first failing case).
/// # Errors
/// Unreadable reference files.
pub fn tokenizer(home: &Path, tok: &nmtokz::Tokenizer) -> Result<(usize, usize, Option<String>), Fault> {
    let g = root(home).join("runs/A/tokenizer");
    let inputs = nmjson::parse(&bytes(&g.join("inputs.u8"))?).map_err(|e| bad(e.to_string()))?;
    let texts = inputs.array().map_err(|e| bad(e.to_string()))?;
    let (mut ok, mut first) = (0, None);
    for (i, t) in texts.iter().enumerate() {
        let s = t.string().map_err(|e| bad(e.to_string()))?;
        let ids = tok.encode(s).map_err(|e| bad(e.to_string()))?;
        let want = i32s(&g.join(format!("text{i}-ids.i32")))?;
        let dec = bytes(&g.join(format!("text{i}-decode.u8")))?;
        if ids == want && tok.decode(&ids).as_bytes() == dec.as_slice() {
            ok += 1;
        } else if first.is_none() {
            first = Some(format!("text {i}"));
        }
    }
    Ok((ok, texts.len(), first))
}

/// One layer of the assembly on input `x` (positions 0 up): its output and the time taken.
fn block_out(m: &Model, device: Device, layer: usize, off: &[usize], x: Vec<f32>) -> Result<(Vec<f32>, f64), Fault> {
    let tokens = x.len() / m.pop.dims.hidden;
    let pos: Vec<i32> = (0..i32::try_from(tokens).unwrap_or(0)).collect();
    // The layer input is seeded at its producer's port: the producer of L{n}.attn_norm.x.
    let (pf, pp) = m.asm.producer(&nid(&format!("L{layer}.attn_norm")), "x").ok_or_else(|| bad("layer input unfed"))?;
    let seeds = vec![(pf.to_owned(), pp.to_owned(), Act::Cpu(x)), (nid("pos"), "y".to_owned(), Act::Ids(pos))];
    let mut opts = RunOpts::default();
    if !off.is_empty() {
        opts.heads_off.insert(layer, off.to_vec());
    }
    let target = (nid(&format!("L{layer}.res2")), "y".to_owned());
    let t0 = std::time::Instant::now();
    let out = m.run(device, seeds, std::slice::from_ref(&target), tokens, &opts)?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok((values(out.get(&target).ok_or_else(|| bad("no block output"))?)?.to_vec(), ms))
}

/// One layer of the assembly on the fixed block input, compared with its reference file. `off`
/// lists query heads switched off in that layer.
/// # Errors
/// Engine or file failure.
pub fn block(home: &Path, m: &Model, device: Device, layer: usize, off: &[usize], file: &str) -> Result<Case, Fault> {
    let g = root(home).join("runs/A/tblock-forward");
    let x = f16s(&g.join("input.f16"))?;
    let want = f16s(&g.join(file))?;
    let (got, ms) = block_out(m, device, layer, off, x)?;
    let tol = atol(home, &format!("tblock-forward/{file}"))?;
    let d = max_abs(&got, &want);
    Ok(Case { name: format!("{} {file}", device.name()), max_abs: d, atol: tol, extra: String::new(), pass: d <= tol, ms })
}

fn prompt_text(dir: &Path, i: usize) -> Result<String, Fault> {
    let inputs = nmjson::parse(&bytes(&dir.join("inputs.json"))?).map_err(|e| bad(e.to_string()))?;
    inputs
        .get("prompts")
        .and_then(Value::array)
        .ok()
        .and_then(|a| a.get(i))
        .and_then(|v| v.string().ok())
        .map(str::to_owned)
        .ok_or_else(|| bad(format!("prompt {i}")))
}
fn logits(m: &Model, device: Device, ids: &[i32]) -> Result<(Vec<f32>, f64), Fault> {
    let target = (nid("head"), "logits".to_owned());
    let t0 = std::time::Instant::now();
    let out = m.forward(device, ids, std::slice::from_ref(&target), &RunOpts::default())?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok((values(out.get(&target).ok_or_else(|| bad("no logits"))?)?.to_vec(), ms))
}

/// A full forward of recorded prompt `i`: last-position logits within tolerance and the argmax
/// equal to the recorded first greedy token.
/// # Errors
/// Engine or file failure.
pub fn prompt(home: &Path, m: &Model, device: Device, i: usize) -> Result<Case, Fault> {
    let ids = m.pop.tokenizer.encode(&prompt_text(&root(home), i)?).map_err(|e| bad(e.to_string()))?;
    let g = root(home).join("runs/A/full-forward");
    let want = f32s(&g.join(format!("prompt{i}-logits.f32")))?;
    let greedy = i32s(&g.join(format!("prompt{i}-greedy.i32")))?;
    let (got, ms) = logits(m, device, &ids)?;
    let tol = atol(home, &format!("full-forward/prompt{i}-logits.f32"))?;
    let d = max_abs(&got, &want);
    let top = argmax(&got);
    let first = greedy.first().and_then(|x| usize::try_from(*x).ok()).unwrap_or(usize::MAX);
    Ok(Case {
        name: format!("{} prompt{i} ({} tokens)", device.name(), ids.len()),
        max_abs: d,
        atol: tol,
        extra: format!("argmax {top}, recorded first token {first}"),
        pass: d <= tol && top == first,
        ms,
    })
}

fn put(path: &Path, data: &[u8], written: &mut Vec<PathBuf>) -> Result<(), Fault> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| bad(format!("{}: {e}", d.display())))?;
    }
    std::fs::write(path, data).map_err(|e| bad(format!("{}: {e}", path.display())))?;
    written.push(path.to_path_buf());
    Ok(())
}
fn le_f16(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| crate::half::bits16(*x).to_le_bytes()).collect()
}
fn le_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn le_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn text(v: &Value) -> Result<Vec<u8>, Fault> {
    Ok(v.encode().map_err(|e| bad(e.to_string()))?.into_bytes())
}

/// Record the reference with this engine on CPU into `root(home)`, reading the test texts,
/// prompts and tolerances from `source` (a folder with the same layout: the earlier recording
/// once, the reference itself on any later re-record). The block input is the engine's own
/// embedding of the first 16 tokens of the test texts. Returns every file written.
/// # Errors
/// Engine or file failure.
pub fn record(home: &Path, m: &Model, source: &Path) -> Result<Vec<PathBuf>, Fault> {
    let dir = root(home);
    let mut w = Vec::new();
    // Tokenizer cases.
    let inputs = nmjson::parse(&bytes(&source.join("runs/A/tokenizer/inputs.u8"))?).map_err(|e| bad(e.to_string()))?;
    let texts: Vec<String> = inputs
        .array()
        .map_err(|e| bad(e.to_string()))?
        .iter()
        .map(|t| t.string().map(str::to_owned).map_err(|e| bad(e.to_string())))
        .collect::<Result<_, _>>()?;
    let tk = dir.join("runs/A/tokenizer");
    put(&tk.join("inputs.u8"), &text(&inputs)?, &mut w)?;
    for (i, t) in texts.iter().enumerate() {
        let ids = m.pop.tokenizer.encode(t).map_err(|e| bad(e.to_string()))?;
        put(&tk.join(format!("text{i}-ids.i32")), &le_i32(&ids), &mut w)?;
        put(&tk.join(format!("text{i}-decode.u8")), m.pop.tokenizer.decode(&ids).as_bytes(), &mut w)?;
    }
    // Block input: the engine's embedding of the first 16 tokens, stored as binary16 and read
    // back, so the recorded outputs come from exactly the stored input.
    let mut ids = Vec::new();
    for t in &texts {
        ids.extend(m.pop.tokenizer.encode(t).map_err(|e| bad(e.to_string()))?);
        if ids.len() >= 16 {
            break;
        }
    }
    ids.truncate(16);
    let (pf, pp) = m.asm.producer(&nid("L0.attn_norm"), "x").ok_or_else(|| bad("layer 0 input unfed"))?;
    let embed = (pf.to_owned(), pp.to_owned());
    let out = m.forward(Device::Cpu, &ids, std::slice::from_ref(&embed), &RunOpts::default())?;
    let x = values(out.get(&embed).ok_or_else(|| bad("no embedding"))?)?.to_vec();
    let tb = dir.join("runs/A/tblock-forward");
    put(&tb.join("input.f16"), &le_f16(&x), &mut w)?;
    let x = f16s(&tb.join("input.f16"))?;
    let mut tol = Vec::new();
    for (layer, off, file) in BLOCKS {
        let (y, _) = block_out(m, Device::Cpu, layer, off, x.clone())?;
        put(&tb.join(file), &le_f16(&y), &mut w)?;
        let key = format!("tblock-forward/{file}");
        tol.push((key.clone(), atol_in(source, &key)?));
    }
    // Full forwards and greedy decodes.
    let ff = dir.join("runs/A/full-forward");
    let mut prompts = Vec::new();
    for i in 0..PROMPTS {
        let p = prompt_text(source, i)?;
        let ids = m.pop.tokenizer.encode(&p).map_err(|e| bad(e.to_string()))?;
        let (l, _) = logits(m, Device::Cpu, &ids)?;
        put(&ff.join(format!("prompt{i}-logits.f32")), &le_f32(&l), &mut w)?;
        let g = crate::decode::greedy(m, Device::Cpu, &ids, GREEDY, &[])?;
        put(&ff.join(format!("prompt{i}-greedy.i32")), &le_i32(&g.tokens), &mut w)?;
        let key = format!("full-forward/prompt{i}-logits.f32");
        tol.push((key.clone(), atol_in(source, &key)?));
        prompts.push(Value::String(p));
    }
    let s = |x: &str| Value::String(x.to_owned());
    let obj = |kv: Vec<(&str, Value)>| Value::Object(kv.into_iter().map(|(k, v)| (k.to_owned(), v)).collect());
    put(
        &dir.join("inputs.json"),
        &text(&obj(vec![("prompts", Value::Array(prompts)), ("note", s("prompts recorded by nmengn::reference::record on CPU"))]))?,
        &mut w,
    )?;
    let files = tol
        .iter()
        .map(|(k, a)| (k.as_str(), obj(vec![("atol", Value::Number(a.to_string())), ("rtol", Value::Number("0".into()))])))
        .collect();
    put(
        &dir.join("tolerance.json"),
        &text(&obj(vec![("files", obj(files)), ("note", s("tolerances carried unchanged from the earlier recording's tolerance.json"))]))?,
        &mut w,
    )?;
    put(
        &dir.join("index.json"),
        &text(&obj(vec![
            ("record", s("engine-reference")),
            ("population", s("qwen3-0.6b")),
            ("recorded_by", s("nmengn::reference::record, CPU")),
            ("at_ms", Value::Number(nmtype::now_ms().to_string())),
        ]))?,
        &mut w,
    )?;
    Ok(w)
}
