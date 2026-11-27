//! Part test X1, encodings across model sizes (2026-11-10; design
//! `docs/2026-11-10-neumor-design-g-muscle-memory-cartridges.md`, sections 4 and 8; direction
//! of Luis Jake Gabriel III: each part gets a short isolated test before any long run).
//!
//! Question: can Qwen3-1.7B use the stored keys and values Qwen3-0.6B made of a code chunk?
//! Both have 28 layers, 8 key-value heads of width 128 and the same rope base, so the tensors
//! have one shape; each model makes them with its own weights.
//!
//! Chunks (rule fixed before the first run): the `.rs` files under `crates/*/src/` in sorted
//! path order; the first `N` whose plain tokenization holds at least `P + C` tokens; prefix =
//! tokens `0..P`, continuation = tokens `P..P + C`. Three arms, all on 1.7B with the same
//! linear precision and device:
//! - A (own): 1.7B encodes the prefix itself (`kvstore::encode`, keys before rope), the window
//!   is built from it, and 1.7B reads the continuation over it.
//! - B (foreign): 0.6B encodes the prefix (all 28 layers, keys before rope), the window is built
//!   from those encodings, and 1.7B reads the continuation over it. Keys are stored before rope
//!   and both models use rope base 1,000,000, so 1.7B's own rope places them at positions
//!   `0..P` exactly as it places its own (a fair handover; the only difference is which
//!   weights made the keys and values).
//! - C (none): 1.7B reads the continuation alone at positions `0..C` (lower bound).
//!
//! Teacher forcing: the continuation's tokens are the input; logits at input position `i`
//! score true token `i + 1`, so `C - 1` tokens are scored per chunk. Measures per chunk and
//! pooled: mean loss per token (natural log), top-1 on the true tokens; for B, top-1 agreement
//! with A's choices (A agrees with itself by definition).
//!
//! Reading rule (fixed before the first run): B is usable when its mean loss is within 10% of
//! A's and its agreement with A is at least 90%; not usable when B's loss is nearer C's than
//! A's; otherwise partly.
//!
//! Models are loaded one at a time (0.6B first, its encodings kept in RAM, then released) so
//! the 4 GB GPU never holds both. `run x1`: report `runs/2026-11-10-X1-encodings-across-sizes-<ms>.txt`,
//! a new file per run.
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use nmengn::assembly::{Assembly, nid};
use nmengn::kvstore::{ChunkKv, Window, encode};
use nmengn::model::{Model, Population, RunOpts, values};
use nmengn::placement::Device;
use nmtype::{Fault, FaultKind};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("x1", FaultKind::Execution, m)
}

/// Chunks read. Reason: eight files give about a thousand scored tokens per arm, enough to
/// resolve a 10% loss gap, and keep the whole run under the five-minute part-test limit.
const N: usize = 8;
/// Prefix tokens. Reason: long enough that a prefix changes the continuation's loss clearly
/// (arm C against arm A), short enough that 1.7B encodes eight of them in seconds and the GPU
/// attention over prefix plus continuation stays small.
const P: usize = 1024;
/// Continuation tokens (C - 1 scored per chunk). Reason: 127 scored tokens per chunk, about a
/// thousand pooled.
const C: usize = 128;
/// Linear weight precision, the same for both models and every arm: the 8-bit form the
/// engine offers (`q8`, the setting the design's part tests name).
const PRECISION: &str = "q8";
/// Reading rule thresholds (test gates, fixed before the first run).
const LOSS_WITHIN: f64 = 0.10;
const AGREE_MIN: f64 = 0.90;

/// The `.rs` files under `crates/*/src/`, sorted by path relative to the mini (`/` separators).
fn source_files(home: &Path) -> Vec<(String, PathBuf)> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }
    let mut files = Vec::new();
    if let Ok(rd) = std::fs::read_dir(home.join("crates")) {
        for e in rd.flatten() {
            walk(&e.path().join("src"), &mut files);
        }
    }
    let mut v: Vec<(String, PathBuf)> =
        files.into_iter().map(|p| (p.strip_prefix(home).unwrap_or(&p).to_string_lossy().replace('\\', "/"), p)).collect();
    v.sort();
    v
}

/// One chunk: its file, prefix ids and continuation ids.
struct Chunk {
    file: String,
    prefix: Vec<i32>,
    cont: Vec<i32>,
}

fn pick(home: &Path, pop: &Population) -> Result<(Vec<Chunk>, usize), Fault> {
    let mut out = Vec::new();
    let mut seen = 0;
    for (rel, p) in source_files(home) {
        if out.len() == N {
            break;
        }
        seen += 1;
        let text = std::fs::read_to_string(&p).map_err(|e| bad(format!("{rel}: {e}")))?;
        let ids = pop.tokenizer.encode_plain(&text).map_err(|e| bad(format!("{rel}: {e}")))?;
        if ids.len() >= P + C {
            out.push(Chunk { file: rel, prefix: ids[..P].to_vec(), cont: ids[P..P + C].to_vec() });
        }
    }
    if out.len() < N {
        return Err(bad(format!("only {} files hold {} tokens", out.len(), P + C)));
    }
    Ok((out, seen))
}

/// Device bytes a model's weights need at their precision (a q8 linear holds one byte per
/// weight plus a row scale; other weights two bytes), each binding once.
fn weight_need(m: &Model) -> u64 {
    let mut keys = std::collections::BTreeSet::new();
    m.bound
        .iter()
        .filter(|b| keys.insert(format!("{}:{}:{}", b.node.precision, b.node.stack, b.node.clusters.join(","))))
        .map(|b| {
            let bytes: u64 = b.views.iter().map(|v| v.cluster.bytes).sum();
            if b.node.op == "linear" && b.node.precision == "q8" { bytes / 2 + b.lin.as_ref().map_or(0, |l| l.n as u64 * 2) } else { bytes }
        })
        .sum()
}

/// Bind `qwen3-reference` over `id` with linear weights at `PRECISION`, and choose its device
/// by the placement rule from the GPU memory free now and what this test puts there: weights,
/// one window of `P` tokens over every layer, and one pass's logits (`C` rows, f32).
fn bind(home: &Path, id: &str, out: &mut String) -> Result<(Model, Device), Fault> {
    let t0 = Instant::now();
    let pop = Population::open(home, id, true, true)?;
    let mut a = Assembly::load(&home.join("designs/assemblies"), "qwen3-reference")?;
    for n in a.nodes.iter_mut().filter(|n| n.op == "linear") {
        n.precision = PRECISION.to_owned();
    }
    let m = Model::bind(&pop, a)?;
    let d = m.pop.dims;
    let window = (P * 2 * d.layers * d.kv_heads * d.head_dim * 2) as u64;
    let logits = (C * d.vocab * 4) as u64;
    let need = weight_need(&m) + window + logits;
    let free = m.pop.gpu_free();
    let choice = m.pop.placement.choose(&nmengn::placement::shape("x1", P + C), need, free);
    let _ = writeln!(
        out,
        "{id}: {} layers, hidden {}, {} kv heads x {}, rope base {}; linear {PRECISION}; GPU {}; free {} bytes, need {need} bytes (weights {} + window {window} + logits {logits}); device {} ({}); open and bind {} ms",
        d.layers,
        d.hidden,
        d.kv_heads,
        d.head_dim,
        d.theta,
        m.pop.gpu_note,
        free.map_or_else(|| "n/a".to_owned(), |f| f.to_string()),
        weight_need(&m),
        choice.device.name(),
        choice.why,
        t0.elapsed().as_millis()
    );
    Ok((m, choice.device))
}

/// Per-chunk measures of one arm.
#[derive(Clone, Copy, Default)]
struct Score {
    loss: f64,
    top1: usize,
    agree: usize,
    n: usize,
}

/// Logits rows (`[C, vocab]`) scored against the continuation shifted by one; `agree_with`
/// gives the argmax per row of a reference arm. Returns the score and this arm's argmax.
fn score(lg: &[f32], vocab: usize, cont: &[i32], agree_with: Option<&[usize]>) -> (Score, Vec<usize>) {
    let mut s = Score::default();
    let mut arg = Vec::new();
    for (i, row) in lg.chunks(vocab).enumerate().take(cont.len().saturating_sub(1)) {
        let target = usize::try_from(cont[i + 1]).unwrap_or(0);
        let (a, mx) = row.iter().enumerate().fold((0, f32::MIN), |b, (j, x)| if *x > b.1 { (j, *x) } else { b });
        let lse = f64::from(mx) + row.iter().map(|x| (f64::from(*x) - f64::from(mx)).exp()).sum::<f64>().ln();
        s.loss += lse - f64::from(row.get(target).copied().unwrap_or(f32::MIN));
        s.top1 += usize::from(a == target);
        if let Some(r) = agree_with {
            s.agree += usize::from(r.get(i) == Some(&a));
        }
        s.n += 1;
        arg.push(a);
    }
    (s, arg)
}

fn sha_file(p: &Path) -> String {
    std::fs::read(p).map(|b| nmhash::sha256_bytes(&b)).unwrap_or_else(|e| format!("unreadable: {e}"))
}

fn test(home: &Path) -> Result<String, Fault> {
    let t0 = Instant::now();
    let mut out = String::new();
    let _ = writeln!(out, "X1 encodings across sizes (part test, 2026-11-10): does Qwen3-1.7B read Qwen3-0.6B's stored keys and values?");
    let tk = |id: &str| sha_file(&nmtype::models_dir(home).join(format!("{id}/files/tokenizer.json")));
    let (t06, t17) = (tk("qwen3-0.6b"), tk("qwen3-1.7b"));
    let _ = writeln!(out, "tokenizer.json sha256: 0.6B {t06}, 1.7B {t17} ({})", if t06 == t17 { "byte-identical" } else { "DIFFERENT" });
    if t06 != t17 {
        return Err(bad("the tokenizers differ; token ids would not carry across"));
    }
    let _ = writeln!(
        out,
        "values: N {N} chunks, P {P} prefix tokens, C {C} continuation tokens ({} scored per chunk), linear {PRECISION} in every arm, loss in nats per token",
        C - 1
    );
    let _ = writeln!(
        out,
        "chunk rule: .rs files under crates/*/src/ in sorted path order, the first {N} with at least {} plain tokens; prefix tokens 0..{P}, continuation {P}..{}",
        P + C,
        P + C
    );

    // 0.6B first: encode every prefix (28 layers, keys before rope), then release it.
    let (m06, dev06) = bind(home, "qwen3-0.6b", &mut out)?;
    let (chunks, scanned) = pick(home, &m06.pop)?;
    let _ = writeln!(out, "files scanned {scanned}; chunks: {}", chunks.iter().map(|c| c.file.as_str()).collect::<Vec<_>>().join(", "));
    let t = Instant::now();
    let mut foreign: Vec<Arc<ChunkKv>> = Vec::new();
    for c in &chunks {
        foreign.push(Arc::new(encode(&m06, dev06, None, &c.prefix)?));
    }
    let enc06 = t.elapsed().as_secs_f64();
    let held: usize = foreign.iter().map(|c| c.bytes()).sum();
    let layers06 = foreign.first().map_or(0, |c| c.layers());
    let _ = writeln!(
        out,
        "0.6B encoded {} prefixes on {} in {enc06:.1} s ({} layers each, {held} bytes held in RAM)",
        foreign.len(),
        dev06.name(),
        layers06
    );
    drop(m06);

    let (m17, dev) = bind(home, "qwen3-1.7b", &mut out)?;
    let d = m17.pop.dims;
    let head = (nid("head"), "logits".to_owned());
    let logits = |ids: &[i32], start: usize, window: Option<Arc<Window>>| -> Result<Vec<f32>, Fault> {
        let opts = RunOpts { all_positions: true, window, ..RunOpts::default() };
        Ok(values(m17.forward_at(dev, ids, start, std::slice::from_ref(&head), &opts)?.get(&head).ok_or_else(|| bad("no logits"))?)?
            .to_vec())
    };
    let _ = writeln!(
        out,
        "per chunk (loss = mean nats per scored token; top1 = share of scored tokens whose argmax is the true token; agree = share of positions where B's argmax equals A's)"
    );
    let _ = writeln!(out, "chunk  file  A_loss  A_top1  B_loss  B_top1  B_agree_A  C_loss  C_top1  A_s  B_s  C_s");
    let (mut sa, mut sb, mut sc) = (Score::default(), Score::default(), Score::default());
    let (mut ta, mut tb, mut tc) = (0.0_f64, 0.0_f64, 0.0_f64);
    let add = |t: &mut Score, s: Score| {
        t.loss += s.loss;
        t.top1 += s.top1;
        t.agree += s.agree;
        t.n += s.n;
    };
    let pct = |a: usize, n: usize| 100.0 * a as f64 / n.max(1) as f64;
    for (i, c) in chunks.iter().enumerate() {
        let t = Instant::now();
        let own = Arc::new(encode(&m17, dev, None, &c.prefix)?);
        let wa = Arc::new(Window::build(&m17, &[own])?);
        let (a, arg_a) = score(&logits(&c.cont, P, Some(wa))?, d.vocab, &c.cont, None);
        let a_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let fk = foreign.get(i).cloned().ok_or_else(|| bad("missing foreign chunk"))?;
        let wb = Arc::new(Window::build(&m17, &[fk])?);
        let (b, _) = score(&logits(&c.cont, P, Some(wb))?, d.vocab, &c.cont, Some(&arg_a));
        let b_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (cc, _) = score(&logits(&c.cont, 0, None)?, d.vocab, &c.cont, None);
        let c_s = t.elapsed().as_secs_f64();
        let _ = writeln!(
            out,
            "{}  {}  {:.4}  {:.1}%  {:.4}  {:.1}%  {:.1}%  {:.4}  {:.1}%  {a_s:.1}  {b_s:.1}  {c_s:.1}",
            i + 1,
            c.file,
            a.loss / a.n.max(1) as f64,
            pct(a.top1, a.n),
            b.loss / b.n.max(1) as f64,
            pct(b.top1, b.n),
            pct(b.agree, b.n),
            cc.loss / cc.n.max(1) as f64,
            pct(cc.top1, cc.n)
        );
        add(&mut sa, a);
        add(&mut sb, b);
        add(&mut sc, cc);
        ta += a_s;
        tb += b_s;
        tc += c_s;
    }
    let (la, lb, lc) = (sa.loss / sa.n.max(1) as f64, sb.loss / sb.n.max(1) as f64, sc.loss / sc.n.max(1) as f64);
    let agree = sb.agree as f64 / sb.n.max(1) as f64;
    let _ = writeln!(out, "mean over {} scored tokens per arm:", sa.n);
    let _ = writeln!(
        out,
        "  A own:     loss {la:.4}, top1 {:.1}%, agreement with A 100% by definition, {ta:.1} s (1.7B encodes the prefix, then reads)",
        pct(sa.top1, sa.n)
    );
    let _ = writeln!(
        out,
        "  B foreign: loss {lb:.4}, top1 {:.1}%, agreement with A {:.1}%, {tb:.1} s (1.7B reads over 0.6B's encodings)",
        pct(sb.top1, sb.n),
        agree * 100.0
    );
    let _ = writeln!(out, "  C none:    loss {lc:.4}, top1 {:.1}%, {tc:.1} s (no prefix)", pct(sc.top1, sc.n));
    let gap = (lb - la) / la.max(1e-12);
    let _ = writeln!(out, "B loss against A: {:+.1}%; |B - A| {:.4}, |B - C| {:.4}", gap * 100.0, (lb - la).abs(), (lb - lc).abs());
    let _ = writeln!(
        out,
        "rule (fixed before the run): usable if B's mean loss is within {:.0}% of A's and its agreement with A is at least {:.0}%; not usable if B's loss is nearer C's than A's; otherwise partly",
        LOSS_WITHIN * 100.0,
        AGREE_MIN * 100.0
    );
    let verdict = if gap.abs() <= LOSS_WITHIN && agree >= AGREE_MIN {
        "usable"
    } else if (lb - lc).abs() < (lb - la).abs() {
        "not usable"
    } else {
        "partly"
    };
    let _ = writeln!(out, "reading: {verdict}");
    let _ = writeln!(out, "wall {:.1} s", t0.elapsed().as_secs_f64());
    Ok(out)
}

/// `run x1`.
pub fn run(_args: &[String], home: &Path) -> ExitCode {
    let mut stamp = nmtype::now_ms();
    let mut txt = home.join(format!("runs/2026-11-10-X1-encodings-across-sizes-{stamp}.txt"));
    while txt.exists() {
        stamp += 1;
        txt = home.join(format!("runs/2026-11-10-X1-encodings-across-sizes-{stamp}.txt"));
    }
    match test(home) {
        Ok(text) => {
            let _ = std::fs::write(&txt, &text);
            print!("{text}");
            println!("report: {}", txt.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            let _ = std::fs::write(&txt, format!("FAULT {e}\n"));
            eprintln!("x1: {e} (report {})", txt.display());
            ExitCode::FAILURE
        }
    }
}
