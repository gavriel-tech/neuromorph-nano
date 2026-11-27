//! Greedy decoding (PLAN 4c `decode`, from emma-labs `native/rscore/decode` and the recording
//! oracle's `generate`): prefill the prompt into a fresh key and value cache, then one token
//! per step at the next position; the highest logit wins and ties go to the lowest id (the
//! recording's rule). Stops at a stop id or the step budget.
use crate::model::{Cache, Model, RunOpts, values};
use crate::placement::Device;
use nmtype::{Fault, FaultKind};
use std::sync::Arc;
use std::time::Instant;

/// The result of one generation.
#[derive(Clone, Debug, PartialEq)]
pub struct Generation {
    pub tokens: Vec<i32>,
    pub prefill_ms: f64,
    pub step_ms: Vec<f64>,
    pub stopped: bool,
}

/// The highest logit, lowest id on ties.
#[must_use]
pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v.get(best).copied().unwrap_or(f32::NEG_INFINITY) {
            best = i;
        }
    }
    best
}

/// Generate up to `max_new` tokens after `prompt` on `device`.
/// # Errors
/// An engine fault.
pub fn greedy(m: &Model, device: Device, prompt: &[i32], max_new: usize, stop: &[i32]) -> Result<Generation, Fault> {
    greedy_until(m, device, prompt, max_new, &|t| stop.contains(&t))
}

/// Greedy decoding that stops after the first token `stop` accepts (that token is kept).
/// # Errors
/// An engine fault.
pub fn greedy_until(m: &Model, device: Device, prompt: &[i32], max_new: usize, stop: &dyn Fn(i32) -> bool) -> Result<Generation, Fault> {
    let opts = RunOpts { cache: Some(Arc::new(Cache::default())), ..RunOpts::default() };
    let target = (m.role_node("head")?, "logits".to_owned());
    let logits = |ids: &[i32], at: usize| -> Result<Vec<f32>, Fault> {
        let out = m.forward_at(device, ids, at, std::slice::from_ref(&target), &opts)?;
        Ok(values(out.get(&target).ok_or_else(|| Fault::new("nmengn.decode", FaultKind::Execution, "no logits"))?)?.to_vec())
    };
    let t0 = Instant::now();
    let mut l = logits(prompt, 0)?;
    let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mut tokens = Vec::new();
    let mut step_ms = Vec::new();
    let mut stopped = false;
    for step in 0..max_new {
        let next = i32::try_from(argmax(&l)).unwrap_or(0);
        tokens.push(next);
        if stop(next) {
            stopped = true;
            break;
        }
        if step + 1 == max_new {
            break;
        }
        let t = Instant::now();
        l = logits(&[next], prompt.len() + step)?;
        step_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(Generation { tokens, prefill_ms, step_ms, stopped })
}

/// Sampling settings (2026-11-12; research file Part 10 (d): the Qwen3 card says not to decode
/// greedily and gives its settings). Read from the card at run time by `Sampling::from_card`.
#[derive(Clone, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub seed: u64,
    /// Where the values came from.
    pub source: String,
}

impl Sampling {
    /// The card's settings for `thinking` mode or not, from the weights folder `dir`: thinking from
    /// `generation_config.json` (the card's default setting), non-thinking from the README's
    /// suggestion line (`Temperature=`, `TopP=`, `TopK=`, `MinP=` after "non-thinking mode").
    /// Stated fallback when neither reads: the README's values as published (0.7, 0.8, 20, 0;
    /// thinking 0.6, 0.95, 20, 0).
    #[must_use]
    pub fn from_card(dir: &std::path::Path, thinking: bool, seed: u64) -> Self {
        let fallback = if thinking { (0.6, 0.95, 20, 0.0) } else { (0.7, 0.8, 20, 0.0) };
        let num = |text: &str, key: &str| -> Option<f32> {
            let at = text.find(key)? + key.len();
            let rest = text.get(at..)?.trim_start_matches(['`', ' ']);
            let end = rest.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(rest.len());
            rest.get(..end)?.parse().ok()
        };
        let from_readme = || -> Option<(f32, f32, usize, f32)> {
            let text = std::fs::read_to_string(dir.join("README.md")).ok()?;
            let line = text
                .lines()
                .find(|l| l.contains(if thinking { "For thinking mode" } else { "For non-thinking mode" }) && l.contains("Temperature="))?;
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a small count")]
            Some((num(line, "Temperature=")?, num(line, "TopP=")?, num(line, "TopK=")? as usize, num(line, "MinP=")?))
        };
        let from_config = || -> Option<(f32, f32, usize, f32)> {
            let v = nmjson::parse(&std::fs::read(dir.join("generation_config.json")).ok()?).ok()?;
            let f = |k: &str| match v.get(k) {
                Ok(nmjson::Value::Number(n)) => n.parse::<f32>().ok(),
                _ => None,
            };
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a small count")]
            Some((f("temperature")?, f("top_p")?, f("top_k")? as usize, f("min_p").unwrap_or(0.0)))
        };
        let (got, source) = if thinking {
            match from_config() {
                Some(x) => (x, "card generation_config.json (thinking)"),
                None => match from_readme() {
                    Some(x) => (x, "card README (thinking)"),
                    None => (fallback, "stated fallback (thinking)"),
                },
            }
        } else {
            match from_readme() {
                Some(x) => (x, "card README (non-thinking)"),
                None => (fallback, "stated fallback (non-thinking)"),
            }
        };
        Self { temperature: got.0, top_p: got.1, top_k: got.2, min_p: got.3, seed, source: source.to_owned() }
    }
}

/// A uniform draw in [0, 1) (xorshift64*), advancing `state`.
fn uniform(state: &mut u64) -> f64 {
    let mut x = *state | 1;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    #[expect(clippy::cast_precision_loss, reason = "a uniform draw")]
    let u = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1_u64 << 53) as f64;
    u
}

/// One token by the card's rule: temperature, then top-k, then top-p (nucleus, at least one kept),
/// then min-p relative to the most likely; drawn with `state`. Temperature 0 is greedy.
#[must_use]
pub fn sample_index(logits: &[f32], s: &Sampling, state: &mut u64) -> usize {
    if s.temperature <= 0.0 || logits.is_empty() {
        return argmax(logits);
    }
    let mut ids: Vec<usize> = (0..logits.len()).filter(|i| logits.get(*i).is_some_and(|x| x.is_finite())).collect();
    ids.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]).then(a.cmp(b)));
    if s.top_k > 0 {
        ids.truncate(s.top_k);
    }
    let top = ids.first().map_or(0.0, |i| f64::from(logits[*i]));
    let w: Vec<f64> = ids.iter().map(|i| ((f64::from(logits[*i]) - top) / f64::from(s.temperature)).exp()).collect();
    let total: f64 = w.iter().sum();
    let mut keep = 0;
    let mut cum = 0.0;
    for x in &w {
        keep += 1;
        cum += x / total;
        if cum >= f64::from(s.top_p) {
            break;
        }
    }
    let first = w.first().copied().unwrap_or(1.0);
    let kept: Vec<(usize, f64)> =
        ids.iter().zip(&w).take(keep.max(1)).filter(|(_, x)| **x >= f64::from(s.min_p) * first).map(|(i, x)| (*i, *x)).collect();
    let sum: f64 = kept.iter().map(|k| k.1).sum();
    let mut u = uniform(state) * sum;
    for (i, x) in &kept {
        if u < *x {
            return *i;
        }
        u -= x;
    }
    kept.last().map_or_else(|| argmax(logits), |k| k.0)
}

/// Decoding by `s` that stops after the first token `stop` accepts (that token is kept). The same
/// seed gives the same tokens.
/// # Errors
/// An engine fault.
pub fn sample_until(
    m: &Model,
    device: Device,
    prompt: &[i32],
    max_new: usize,
    stop: &dyn Fn(i32) -> bool,
    s: &Sampling,
) -> Result<Generation, Fault> {
    let opts = RunOpts { cache: Some(Arc::new(Cache::default())), ..RunOpts::default() };
    let target = (m.role_node("head")?, "logits".to_owned());
    let logits = |ids: &[i32], at: usize| -> Result<Vec<f32>, Fault> {
        let out = m.forward_at(device, ids, at, std::slice::from_ref(&target), &opts)?;
        Ok(values(out.get(&target).ok_or_else(|| Fault::new("nmengn.decode", FaultKind::Execution, "no logits"))?)?.to_vec())
    };
    let mut state = s.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let t0 = Instant::now();
    let mut l = logits(prompt, 0)?;
    let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (mut tokens, mut step_ms, mut stopped) = (Vec::new(), Vec::new(), false);
    for step in 0..max_new {
        let next = i32::try_from(sample_index(&l, s, &mut state)).unwrap_or(0);
        tokens.push(next);
        if stop(next) {
            stopped = true;
            break;
        }
        if step + 1 == max_new {
            break;
        }
        let t = Instant::now();
        l = logits(&[next], prompt.len() + step)?;
        step_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(Generation { tokens, prefill_ms, step_ms, stopped })
}

/// Section check: the card's settings read from a README and a generation config; sampling at
/// temperature 0 is greedy; top-k 1 is greedy at any temperature; a token outside top-k or the
/// nucleus is never drawn; frequencies follow the softmax within 3 points; the same seed repeats.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let dir = nmtype::check_dir("nmengn-decode-sampling").map_err(|e| e.to_string())?;
    std::fs::write(dir.join("README.md"), "> For thinking mode, use `Temperature=0.6`, `TopP=0.95`, `TopK=20`, and `MinP=0` (default).\n> For non-thinking mode, we suggest using `Temperature=0.7`, `TopP=0.8`, `TopK=20`, and `MinP=0`.\n").map_err(|e| e.to_string())?;
    std::fs::write(dir.join("generation_config.json"), r#"{"temperature": 0.6, "top_k": 20, "top_p": 0.95}"#).map_err(|e| e.to_string())?;
    let nt = Sampling::from_card(&dir, false, 1);
    let th = Sampling::from_card(&dir, true, 1);
    if (nt.temperature, nt.top_p, nt.top_k, nt.min_p) != (0.7, 0.8, 20, 0.0)
        || (th.temperature, th.top_p, th.top_k) != (0.6, 0.95, 20)
        || !nt.source.contains("README")
        || !th.source.contains("generation_config")
    {
        return Err(format!("nmengn.decode: card settings {nt:?} {th:?}"));
    }
    let logits = [2.0_f32, 1.0, 0.5, -1.0, 3.0];
    let mut st = 7_u64;
    let greedy0 = Sampling { temperature: 0.0, ..nt.clone() };
    let k1 = Sampling { top_k: 1, temperature: 5.0, ..nt.clone() };
    if sample_index(&logits, &greedy0, &mut st) != 4 || (0..50).any(|_| sample_index(&logits, &k1, &mut st) != 4) {
        return Err("nmengn.decode: greedy limits".into());
    }
    let s = Sampling { temperature: 1.0, top_p: 1.0, top_k: 3, min_p: 0.0, seed: 3, source: String::new() };
    let mut counts = [0_usize; 5];
    for _ in 0..20_000 {
        if let Some(c) = counts.get_mut(sample_index(&logits, &s, &mut st)) {
            *c += 1;
        }
    }
    let z: f64 = [3.0_f64, 2.0, 1.0].iter().map(|x| x.exp()).sum();
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let f = |i: usize| counts[i] as f64 / 20_000.0;
    if counts[2] + counts[3] != 0 || (f(4) - 3.0_f64.exp() / z).abs() > 0.03 || (f(0) - 2.0_f64.exp() / z).abs() > 0.03 {
        return Err(format!("nmengn.decode: frequencies {counts:?}"));
    }
    let nucleus = Sampling { top_p: 0.5, ..s.clone() };
    if (0..200).any(|_| sample_index(&logits, &nucleus, &mut st) != 4) {
        return Err("nmengn.decode: a token outside the nucleus was drawn".into());
    }
    let (mut a, mut b) = (11_u64, 11_u64);
    if (0..20).map(|_| sample_index(&logits, &s, &mut a)).collect::<Vec<_>>()
        != (0..20).map(|_| sample_index(&logits, &s, &mut b)).collect::<Vec<_>>()
    {
        return Err("nmengn.decode: the same seed did not repeat".into());
    }
    Ok(())
}
