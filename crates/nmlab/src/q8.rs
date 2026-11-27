//! Card residency and seconds per pass of one served MicroModel at one precision (`run q8`).
//!
//! `nano-lab run q8 <base|embedder|reranker|writer> [--bits=default|8|16] [--dp4a] [--reps=<n>]
//! [--kernels]`. The model is opened as `serve` opens it: the served arithmetic switches on (half
//! math, packed attention), the library's base cartridges for it offered in the serving order, the
//! selector's width when `--bits` names one (`default` leaves the selector empty: the product's 16
//! bits with half math; an explicit 16 is binary16 weights with binary32 math). Reported: the seat
//! line, the model's bytes on the card and the driver's view before and after the bind, then the
//! median seconds of each kind of pass after one warm-up: one at a time and batched for the
//! embedder and the reranker; one token at a time (a decode step) and prefill of 512 and 2048
//! tokens for the base model and the writer. `--kernels` adds the per-kernel table of one more
//! repetition (synchronised launches, so its seconds are only shares). `--save=<file>` writes the
//! role's outputs (the embedder's vectors, the judge's P(yes) over its pairs, the reader's or the
//! writer's check outputs and 64 greedy tokens) and `--against=<file>` compares them with a saved
//! reference. Text is the Nano's own source, read in place.
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use nmengn::model::Model;
use nmengn::placement::Device;
use nmtype::{Fault, FaultKind};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("q8", FaultKind::Execution, m)
}
fn opt<'a>(args: &'a [String], k: &str) -> Option<&'a str> {
    args.iter().find_map(|a| a.strip_prefix(k).and_then(|r| r.strip_prefix('=')))
}

/// Served parts and their populations.
const PARTS: [(&str, &str); 4] = [("base", "qwen3-0.6b"), ("embedder", "qwen3-embedding-0.6b"), ("reranker", "qwen3-reranker-0.6b"), ("writer", "qwen3-1.7b")];
/// Source file the texts are cut from.
const SOURCE: &str = "crates/nmengn/src/model.rs";

/// The library's base cartridges for `pop`, newest first, as the serving program offers them.
fn offer(home: &Path, pop: &str) -> usize {
    let Ok(lib) = nmlearn::library::Library::open(home) else { return 0 };
    let mut cs: Vec<&nmlearn::library::Cartridge> =
        lib.cartridges.iter().filter(|c| c.kind == "tensors" && c.family != nmlearn::library::ADAPTER && c.model == pop).collect();
    let file = |c: &nmlearn::library::Cartridge| lib.cartridge_dir(c).join(c.files.first().map_or("", |f| f.name.as_str()));
    let stamp = |c: &nmlearn::library::Cartridge| std::fs::metadata(file(c)).and_then(|m| m.modified()).ok();
    cs.sort_by(|a, b| (b.made.as_str(), stamp(b), b.version).cmp(&(a.made.as_str(), stamp(a), a.version)));
    let files: Vec<PathBuf> = cs.iter().map(|c| file(c)).collect();
    let n = files.len();
    if let Ok(mut g) = nmapi::neural::BASE_CARTRIDGES.write() {
        g.insert(pop.to_owned(), files);
    }
    n
}

/// Texts of `lines` source lines each.
fn texts(home: &Path, n: usize, lines: usize) -> Result<Vec<String>, Fault> {
    let t = std::fs::read_to_string(home.join(SOURCE)).map_err(|e| bad(format!("{SOURCE}: {e}")))?;
    let ls: Vec<&str> = t.lines().collect();
    Ok((0..n).map(|i| ls.iter().skip((i * lines) % ls.len().max(1)).take(lines).copied().collect::<Vec<_>>().join("\n")).collect())
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

/// Median seconds of `reps` runs of `f` after one warm-up.
fn time(reps: usize, mut f: impl FnMut() -> Result<(), Fault>) -> Result<f64, Fault> {
    f()?;
    let mut v = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f()?;
        v.push(t.elapsed().as_secs_f64());
    }
    Ok(median(v))
}

fn ids(m: &Model, text: &str, len: usize) -> Result<Vec<i32>, Fault> {
    let mut v = m.pop.tokenizer.encode(text).map_err(|e| bad(e.to_string()))?;
    while v.len() < len {
        let more = v.clone();
        v.extend(more);
    }
    v.truncate(len);
    Ok(v)
}

#[expect(clippy::cast_precision_loss, reason = "megabytes for a report")]
fn mib(b: usize) -> f64 {
    b as f64 / 1_048_576.0
}

fn measure(home: &Path, args: &[String]) -> Result<String, Fault> {
    let part = args.get(2).map_or("", String::as_str);
    let (_, pop) = PARTS.iter().find(|(p, _)| *p == part).ok_or_else(|| bad("part: base, embedder, reranker or writer"))?;
    let bits = opt(args, "--bits").unwrap_or("default");
    let reps: usize = opt(args, "--reps").and_then(|x| x.parse().ok()).unwrap_or(5);
    nmengn::gpu::HALF_MATH.store(true, Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
    nmengn::gpu::DP4A.store(args.iter().any(|a| a == "--dp4a"), Ordering::Relaxed);
    if bits != "default" {
        let p = nmengn::model::Precision::parse(&format!("{part}={bits}")).map_err(bad)?;
        if let Ok(mut g) = nmengn::model::PRECISION.write() {
            *g = p;
        }
    }
    let offered = offer(home, pop);
    let before = nmengn::residency::get().memory_now();
    let t = Instant::now();
    let m: Arc<Model> = match part {
        "base" => nmapi::neural::model(home)?,
        "embedder" => nmapi::neural::embedder_model(home)?,
        "reranker" => nmapi::neural::reranker_model(home)?,
        _ => nmapi::neural::writer_model(home, true)?,
    };
    let open_s = t.elapsed().as_secs_f64();
    if m.pop.gpu.is_none() {
        return Err(bad("no graphics card"));
    }
    let after = m.pop.gpu.as_ref().and_then(|g| g.gpu.memory().ok());
    let used = |x: Option<(usize, usize)>| x.map_or(0.0, |(f, t)| mib(t - f));
    let mut out = String::new();
    let mut line = |s: String| {
        out.push_str(&s);
        out.push('\n');
    };
    let widths: Vec<&str> = m.asm.nodes.iter().filter(|n| n.op == "linear").map(|n| n.precision.as_str()).collect();
    let mut kinds: Vec<&str> = widths.clone();
    kinds.sort_unstable();
    kinds.dedup();
    line(format!(
        "q8 {part} ({pop}) bits {bits}{}: linears {} ({}), open {open_s:.1} s, base cartridges offered {offered}",
        if nmengn::gpu::DP4A.load(Ordering::Relaxed) { " dp4a" } else { "" },
        widths.len(),
        kinds.join(",")
    ));
    for s in nmapi::neural::seat_notes().iter().filter(|l| l.contains(pop)) {
        line(format!("  seat: {}", s.chars().take(220).collect::<String>()));
    }
    line(format!(
        "  card: model {:.0} MiB on the card; driver view used {:.0} MiB before the bind, {:.0} MiB after",
        mib(m.pop.device_bytes()),
        used(before),
        used(after)
    ));
    let dev = Device::Gpu;
    let mut passes: Vec<(String, f64, usize)> = Vec::new();
    match part {
        "embedder" => {
            let one = texts(home, 1, 30)?;
            let n1 = m.pop.tokenizer.encode(&one[0]).map_or(0, |v| v.len());
            let s = time(reps, || nmengn::roles::card_embed_on(&m, dev, &one[0]).map(|_| ()))?;
            passes.push((format!("one text ({n1} tokens)"), s, 1));
            let many = texts(home, nmapi::neural::DENSE_BATCH, 30)?;
            let cap = 512;
            let bt = nmengn::roles::batch_tokens(&m, cap + 1);
            let s = time(reps, || nmengn::roles::card_embed_batch(&m, &many, cap, bt).map(|_| ()))?;
            passes.push((format!("batch of {} texts (cap {cap}, {bt} tokens per forward)", many.len()), s, many.len()));
        }
        "reranker" => {
            let q = "Where does the engine choose the kernel for an 8-bit linear?";
            let ins = nmengn::roles::CODE_INSTRUCTION;
            let docs = texts(home, 12, 40)?;
            let n1 = m.pop.tokenizer.encode(&nmengn::roles::card_prompt(ins, q, &docs[0])).map_or(0, |v| v.len());
            let s = time(reps, || nmengn::roles::card_rerank_on(&m, dev, ins, q, &docs[0]).map(|_| ()))?;
            passes.push((format!("one pair ({n1} tokens)"), s, 1));
            let s = time(reps, || nmengn::roles::card_rerank_batch(&m, dev, ins, q, &docs, None).map(|_| ()))?;
            passes.push((format!("batch of {} pairs", docs.len()), s, docs.len()));
        }
        _ => {
            let text = texts(home, 1, 400)?;
            let short = ids(&m, &text[0], 256)?;
            let steps = 32;
            let mut step = Vec::new();
            nmengn::decode::greedy(&m, dev, &short, 2, &[])?;
            for _ in 0..reps.max(1) {
                let g = nmengn::decode::greedy(&m, dev, &short, steps + 1, &[])?;
                step.extend(g.step_ms.iter().skip(1).map(|x| x / 1000.0));
            }
            passes.push((format!("one token at a time (decode step after 256 tokens, {steps} steps x {reps})"), median(step), 1));
            for len in [512, 2048] {
                let p = ids(&m, &text[0], len)?;
                let s = time(reps, || nmengn::decode::greedy(&m, dev, &p, 1, &[]).map(|_| ()))?;
                passes.push((format!("prefill of {len} tokens"), s, len));
            }
        }
    }
    for (name, s, n) in &passes {
        #[expect(clippy::cast_precision_loss, reason = "a rate for a report")]
        let per = s / (*n).max(1) as f64;
        line(format!("  pass {name}: {:.2} ms (per item {:.3} ms)", s * 1000.0, per * 1000.0));
    }
    // Role outputs, saved and compared across precisions (`--save=<file>`, `--against=<file>`).
    let (role, tokens): (Vec<f32>, Vec<i32>) = match part {
        "embedder" => {
            let many = texts(home, nmapi::neural::DENSE_BATCH, 30)?;
            (nmengn::roles::card_embed_batch(&m, &many, 512, nmengn::roles::batch_tokens(&m, 513))?.concat(), Vec::new())
        }
        "reranker" => {
            let docs = texts(home, 12, 40)?;
            (nmengn::roles::card_rerank_batch(&m, dev, nmengn::roles::CODE_INSTRUCTION, "Where does the engine choose the kernel for an 8-bit linear?", &docs, None)?, Vec::new())
        }
        _ => {
            let text = texts(home, 1, 400)?;
            let g = nmengn::decode::greedy(&m, dev, &ids(&m, &text[0], 256)?, 64, &[])?;
            (nmengn::base::check_outputs(&m)?.into_iter().map(f32::from_bits).collect(), g.tokens)
        }
    };
    if let Some(f) = opt(args, "--save") {
        let mut b: Vec<u8> = (u32::try_from(tokens.len()).unwrap_or(0)).to_le_bytes().to_vec();
        b.extend(tokens.iter().flat_map(|x| x.to_le_bytes()));
        b.extend(role.iter().flat_map(|x| x.to_le_bytes()));
        std::fs::write(f, b).map_err(|e| bad(format!("{f}: {e}")))?;
    }
    if let Some(f) = opt(args, "--against") {
        let b = std::fs::read(f).map_err(|e| bad(format!("{f}: {e}")))?;
        let words: Vec<[u8; 4]> = b.as_chunks::<4>().0.to_vec();
        let nt = words.first().map_or(0, |w| u32::from_le_bytes(*w) as usize);
        let rt: Vec<i32> = words.iter().skip(1).take(nt).map(|w| i32::from_le_bytes(*w)).collect();
        let rv: Vec<f32> = words.iter().skip(1 + nt).map(|w| f32::from_le_bytes(*w)).collect();
        if rv.len() != role.len() {
            line(format!("  against {f}: {} values there, {} here", rv.len(), role.len()));
        } else {
            match part {
                "embedder" => {
                    let h = m.pop.dims.hidden;
                    let cos: Vec<f64> = role.chunks(h).zip(rv.chunks(h)).map(|(a, b)| {
                        let d: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
                        let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
                        let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
                        d / (na * nb).max(1e-30)
                    }).collect();
                    #[expect(clippy::cast_precision_loss, reason = "a mean for a report")]
                    let mean = cos.iter().sum::<f64>() / cos.len().max(1) as f64;
                    line(format!("  role against {f}: cosine to the reference vectors, least {:.5}, mean {mean:.5}, over {} texts", cos.iter().copied().fold(1.0, f64::min), cos.len()));
                }
                "reranker" => {
                    let z = |p: f32| { let p = p.clamp(1e-6, 1.0 - 1e-6); (p / (1.0 - p)).ln() };
                    let dz = role.iter().zip(&rv).map(|(a, b)| (z(*a) - z(*b)).abs()).fold(0.0_f32, f32::max);
                    let rank = |v: &[f32]| { let mut o: Vec<usize> = (0..v.len()).collect(); o.sort_by(|a, b| v[*b].total_cmp(&v[*a])); o };
                    let (ra, rb) = (rank(&role), rank(&rv));
                    line(format!("  role against {f}: largest change of the yes-or-no log-odds {dz:.4} over {} pairs; order identical {}; first place same {}", role.len(), ra == rb, ra.first() == rb.first()));
                }
                _ => {
                    let num: f64 = role.iter().zip(&rv).map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2)).sum();
                    let den: f64 = rv.iter().map(|b| f64::from(*b).powi(2)).sum::<f64>().max(1e-30);
                    let same = tokens.iter().zip(&rt).take_while(|(a, b)| a == b).count();
                    line(format!("  role against {f}: check outputs relative RMS change {:.5}; greedy tokens identical for the first {same} of {}", (num / den).sqrt(), rt.len().min(tokens.len())));
                }
            }
        }
    }
    if let Some(p) = nmengn::residency::get().pool() {
        line(format!("  pool: long-lived {:.0} MiB, working peak {:.0} MiB", mib(p.live_kept), mib(p.peak_work)));
    }
    if args.iter().any(|a| a == "--kernels") {
        nmffi::prof::reset();
        nmffi::prof::set(true);
        nmffi::prof::set_kernels(true);
        match part {
            "embedder" => {
                let many = texts(home, nmapi::neural::DENSE_BATCH, 30)?;
                let _ = nmengn::roles::card_embed_batch(&m, &many, 512, nmengn::roles::batch_tokens(&m, 513))?;
            }
            "reranker" => {
                let docs = texts(home, 12, 40)?;
                let _ = nmengn::roles::card_rerank_batch(&m, dev, nmengn::roles::CODE_INSTRUCTION, "Where is the kernel chosen?", &docs, None)?;
            }
            _ => {
                let text = texts(home, 1, 400)?;
                let _ = nmengn::decode::greedy(&m, dev, &ids(&m, &text[0], 256)?, 9, &[])?;
            }
        }
        nmffi::prof::set(false);
        let mut k: Vec<(String, f64, u64)> = nmffi::prof::side().into_iter().filter(|r| r.0.starts_with("kernel ")).collect();
        k.sort_by(|a, b| b.1.total_cmp(&a.1));
        let tot: f64 = k.iter().map(|r| r.1).sum();
        line(format!("  kernels (one {} pass, synchronised): {:.1} ms in all", if matches!(part, "base" | "writer") { "prefill plus 8 decode steps" } else { "batched" }, tot * 1000.0));
        for (name, s, c) in k.iter().take(10) {
            line(format!("    {:7.2} ms {c:6}  {name}", s * 1000.0));
        }
    }
    Ok(out)
}

/// `nano-lab run q8 ...`.
pub fn run(args: &[String], home: &Path) -> ExitCode {
    match measure(home, args) {
        Ok(t) => {
            print!("{t}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("q8: {e}");
            ExitCode::FAILURE
        }
    }
}
