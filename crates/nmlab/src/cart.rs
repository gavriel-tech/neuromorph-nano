//! `nano-lab run cart <step> ...` (2026-11-11): cartridge build steps D onward for the
//! judge job (research file `docs/2026-11-10-neumor-research-protocol-pass-all-parts.md`,
//! sections "Solutions 5 and 6 joined" and "Cartridges"). The judge job is the released
//! Qwen3-Reranker-0.6B with the `qwen3-reranker` assembly, read as P(yes) of a code document for
//! a question. Each step writes `runs/CART-<step>-<ms>.txt`; records hold indices and atlas
//! coordinates only, never question text.
//!
//! Judge pairs: the CX1 development half (`e2::suite`), each question with its first reviewed
//! line range (the right document) and the first range of the next question in another file (a
//! wrong document), both cut to `DOC_TOKENS` tokens. Even positions map (calibration), odd
//! positions are held out.
//!
//! Steps: `d` recording pass (activity per query head and per feed-forward block of 64 channels,
//! all 28 layers, written as `judge` records to `designs/parmap/qwen3-reranker-0.6b.json`).
use std::fmt::Write as _;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use nmengn::Assembly;
use nmengn::assembly::nid;
use nmengn::model::{Act, Model, Population, RunOpts, values};
use nmengn::placement::Device;
use nmnpcs::atlas::{Atlas, Coordinate, Piece, Record};
use nmtype::{Fault, FaultKind};

/// The base cartridge manifests: lab data, kept with the lab tool. The product loads the released
/// models whole (quantization only) and uses cartridges only for learned additions.
const BASE_CARTRIDGES: &str = "crates/nmlab/data/cartridges";

const POP: &str = "qwen3-reranker-0.6b";
const ASM: &str = "qwen3-reranker";
const JOB: &str = "judge";
/// Document tokens per pair (kept short so a lesion sweep fits its time limit).
const DOC_TOKENS: usize = 256;
const BLOCK: usize = 64;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("cart", FaultKind::Execution, m)
}

/// One judge pair: question index in the development half, whether the document is the right
/// one, and the prompt's token ids.
pub(crate) struct Pair {
    pub q: usize,
    pub right: bool,
    pub ids: Vec<i32>,
}

fn lines(home: &Path, file: &str, a: usize, b: usize) -> String {
    let text = std::fs::read_to_string(home.join(file)).unwrap_or_default();
    let body: Vec<&str> = text.lines().skip(a.saturating_sub(1)).take(b + 1 - a.max(1)).collect();
    nmsubs::dense::doc_text(file, &body.join("\n"))
}

/// The judge pairs of the development half: (calibration, held out).
pub(crate) fn pairs(home: &Path, m: &Model) -> Result<(Vec<Pair>, Vec<Pair>), Fault> {
    let items = crate::e2::suite(home)?;
    let tok = |s: &str| m.pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
    let doc = |k: &(String, usize, usize)| -> Result<String, Fault> {
        let mut d = tok(&lines(home, &k.0, k.1, k.2))?;
        d.truncate(DOC_TOKENS);
        Ok(m.pop.tokenizer.decode(&d))
    };
    let (mut cal, mut held) = (Vec::new(), Vec::new());
    let n = items.len();
    for (i, it) in items.iter().enumerate() {
        let Some(k) = it.keys.first() else { continue };
        let other = (1..n).map(|s| (i + s) % n).find_map(|j| items.get(j).and_then(|x| x.keys.first()).filter(|o| o.0 != k.0));
        let Some(o) = other else { continue };
        let into = if i % 2 == 0 { &mut cal } else { &mut held };
        for (right, d) in [(true, doc(k)?), (false, doc(o)?)] {
            let ids = tok(&nmengn::roles::card_prompt(nmengn::roles::CODE_INSTRUCTION, &it.q, &d))?;
            into.push(Pair { q: i, right, ids });
        }
    }
    Ok((cal, held))
}

/// SHA-256 of a pair set's token ids (the calibration hash written into the atlas).
fn set_hash(p: &[Pair]) -> String {
    nmhash::sha256_bytes(&p.iter().flat_map(|x| x.ids.iter().flat_map(|i| i.to_le_bytes()).chain([u8::from(x.right)])).collect::<Vec<u8>>())
}

/// Open the judge on the GPU (the processor when no GPU opens).
pub(crate) fn open(home: &Path) -> Result<(Model, Device), Fault> {
    let pop = Population::open(home, POP, true, true)?;
    let dev = if pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    Ok((Model::bind(&pop, Assembly::load(&home.join("designs/assemblies"), ASM)?)?, dev))
}

/// P(yes) of one pair under `opts`.
pub(crate) fn p_yes(m: &Model, dev: Device, p: &Pair, opts: &RunOpts) -> Result<f32, Fault> {
    let label = (m.role_node("label")?, "y".to_owned());
    let out = m.forward(dev, &p.ids, std::slice::from_ref(&label), opts)?;
    values(out.get(&label).ok_or_else(|| bad("no label output"))?)?.first().copied().ok_or_else(|| bad("empty label"))
}

/// Share of questions whose right document scores above its wrong one.
pub(crate) fn pair_accuracy(set: &[Pair], p: &[f32]) -> f64 {
    let (mut ok, mut n) = (0_u32, 0_u32);
    for (i, x) in set.iter().enumerate() {
        if x.right
            && let (Some(a), Some(b)) = (p.get(i), p.get(i + 1))
        {
            n += 1;
            ok += u32::from(a > b);
        }
    }
    f64::from(ok) / f64::from(n.max(1))
}

fn rms(v: &[f32]) -> f32 {
    #[expect(clippy::cast_precision_loss, reason = "lengths")]
    let n = v.len().max(1) as f32;
    (v.iter().map(|x| x * x).sum::<f32>() / n).sqrt()
}

/// Per piece: activity over all tokens and at the last token, for one pair.
fn activity(y: &[f32], tokens: usize, unit: usize) -> (Vec<f32>, Vec<f32>) {
    let width = y.len() / tokens.max(1);
    let units = width / unit;
    let mut all = vec![0.0_f32; units];
    let mut last = vec![0.0_f32; units];
    for (t, row) in y.chunks(width.max(1)).enumerate() {
        for (u, s) in row.chunks(unit).enumerate().take(units) {
            let r = rms(s);
            if let Some(a) = all.get_mut(u) {
                *a += r;
            }
            if t + 1 == tokens
                && let Some(l) = last.get_mut(u)
            {
                *l = r;
            }
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    all.iter_mut().for_each(|a| *a /= tokens.max(1) as f32);
    (all, last)
}

/// The judge atlas, or a new one on the population's weights.
fn load_atlas(home: &Path, card_hash: &str) -> Result<Atlas, Fault> {
    let path = home.join(format!("designs/parmap/{POP}.json"));
    let atlas = match std::fs::read_to_string(&path) {
        Ok(t) => Atlas::from_text(&t)?,
        Err(_) => Atlas {
            population: POP.to_owned(),
            base_sha256: card_hash.to_owned(),
            calibration_sha256: String::new(),
            version: 0,
            records: Vec::new(),
        },
    };
    if atlas.base_sha256 != card_hash {
        return Err(bad(format!("{}: atlas base hash differs from the weights", path.display())));
    }
    Ok(atlas)
}
/// Write the judge atlas with its version raised.
fn save_atlas(home: &Path, atlas: &mut Atlas) -> Result<(), Fault> {
    let path = home.join(format!("designs/parmap/{POP}.json"));
    atlas.version += 1;
    std::fs::write(&path, atlas.to_text()).map_err(|e| bad(format!("{}: {e}", path.display())))
}
/// Update the judge record at `at` in place (made when missing); run and date are refreshed.
fn put(atlas: &mut Atlas, at: &Coordinate, run: &str, date: &str, f: impl FnOnce(&mut Record)) {
    let i = match atlas.records.iter().position(|r| r.job == JOB && &r.at == at) {
        Some(i) => i,
        None => {
            atlas.records.push(Record::new(at.clone(), JOB, run, date));
            atlas.records.len() - 1
        }
    };
    if let Some(r) = atlas.records.get_mut(i) {
        run.clone_into(&mut r.run);
        date.clone_into(&mut r.date);
        f(r);
    }
}

/// Step D: the recording pass.
fn record(home: &Path, stamp: u64) -> Result<String, Fault> {
    let t0 = Instant::now();
    let (m, dev) = open(home)?;
    let d = m.pop.dims;
    let (cal, held) = pairs(home, &m)?;
    let mut out = String::new();
    let label = (m.role_node("label")?, "y".to_owned());
    let mut targets = vec![label.clone()];
    for l in 0..d.layers {
        targets.push((nid(&format!("L{l}.score")), "y".to_owned()));
        targets.push((nid(&format!("L{l}.swigate")), "y".to_owned()));
    }
    // [layer][unit] sums: all-token activity, last-token activity on right and wrong documents.
    let z = |u: usize| vec![vec![0.0_f64; u]; d.layers];
    let (mut h_all, mut h_r, mut h_w) = (z(d.heads), z(d.heads), z(d.heads));
    let blocks = d.inter / BLOCK;
    let (mut b_all, mut b_r, mut b_w) = (z(blocks), z(blocks), z(blocks));
    let (mut pc, mut tokens, mut fwd_ms) = (Vec::new(), 0_usize, 0.0_f64);
    for p in &cal {
        let t = Instant::now();
        let o = m.forward(dev, &p.ids, &targets, &RunOpts::default())?;
        fwd_ms += t.elapsed().as_secs_f64() * 1000.0;
        tokens += p.ids.len();
        pc.push(values(o.get(&label).ok_or_else(|| bad("no label"))?)?.first().copied().unwrap_or(0.0));
        for l in 0..d.layers {
            for (key, unit, all, r, w) in [
                (format!("L{l}.score"), d.head_dim, &mut h_all, &mut h_r, &mut h_w),
                (format!("L{l}.swigate"), BLOCK, &mut b_all, &mut b_r, &mut b_w),
            ] {
                let y = values(o.get(&(nid(&key), "y".to_owned())).ok_or_else(|| bad(format!("no {key}")))?)?;
                let (a, last) = activity(y, p.ids.len(), unit);
                let (Some(all), Some(side)) = (all.get_mut(l), if p.right { r.get_mut(l) } else { w.get_mut(l) }) else { continue };
                all.iter_mut().zip(&a).for_each(|(s, x)| *s += f64::from(*x));
                side.iter_mut().zip(&last).for_each(|(s, x)| *s += f64::from(*x));
            }
        }
    }
    let mut ph = Vec::new();
    for p in &held {
        ph.push(p_yes(&m, dev, p, &RunOpts::default())?);
    }
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let (n, half) = (cal.len().max(1) as f64, (cal.len() / 2).max(1) as f64);
    let mean = |v: &[f32], right: bool, set: &[Pair]| -> f64 {
        let x: Vec<f64> = v.iter().zip(set).filter(|(_, p)| p.right == right).map(|(x, _)| f64::from(*x)).collect();
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let k = x.len().max(1) as f64;
        x.iter().sum::<f64>() / k
    };
    let _ = writeln!(
        out,
        "Cartridge step D, recording pass for the judge job ({POP}, assembly {ASM}, {}). Pairs: {} calibration ({} questions), {} held out; documents cut to {DOC_TOKENS} tokens; mean prompt {:.0} tokens; forward with {} recorded outputs {:.0} ms mean.",
        dev.name(),
        cal.len(),
        cal.len() / 2,
        held.len(),
        tokens as f64 / n,
        targets.len() - 1,
        fwd_ms / n
    );
    let _ = writeln!(
        out,
        "Judge baseline (full model): pair accuracy (right document above wrong) calibration {:.3}, held out {:.3}; mean P(yes) right {:.3} wrong {:.3} (calibration), right {:.3} wrong {:.3} (held out).",
        pair_accuracy(&cal, &pc),
        pair_accuracy(&held, &ph),
        mean(&pc, true, &cal),
        mean(&pc, false, &cal),
        mean(&ph, true, &held),
        mean(&ph, false, &held)
    );
    // Atlas records and the summary.
    let card_hash = m.pop.card.sha_main.clone();
    let short: String = card_hash.chars().take(12).collect();
    let date = nmlearn::cartridge::date_of(stamp);
    let run = format!("runs/CART-d-{stamp}.txt");
    let mut recs = Vec::new();
    let per = d.heads / d.kv_heads.max(1);
    let mut lines_out = Vec::new();
    let mut sel_all: Vec<(f64, String)> = Vec::new();
    for l in 0..d.layers {
        let lu = u16::try_from(l).unwrap_or(u16::MAX);
        let mut row = format!("L{l:>2}");
        for (kind, all, r, w) in [("heads", &h_all, &h_r, &h_w), ("blocks", &b_all, &b_r, &b_w)] {
            let (Some(a), Some(r), Some(w)) = (all.get(l), r.get(l), w.get(l)) else { continue };
            let act: Vec<f64> = a.iter().map(|x| x / n).collect();
            let total: f64 = act.iter().sum();
            let mut sorted = act.clone();
            sorted.sort_by(|x, y| y.total_cmp(x));
            let top: f64 = sorted.iter().take(act.len() / 4).sum();
            let _ = write!(row, "  {kind}: top quarter holds {:.0}% of activity", 100.0 * top / total.max(1e-12));
            for (u, x) in act.iter().enumerate() {
                let (rr, ww) = (r.get(u).copied().unwrap_or(0.0) / half, w.get(u).copied().unwrap_or(0.0) / half);
                let sel = (rr - ww) / ((rr + ww) / 2.0).max(1e-9);
                let uu = u16::try_from(u).unwrap_or(u16::MAX);
                let piece = if kind == "heads" {
                    Piece::QHead(lu, u16::try_from(u / per.max(1)).unwrap_or(u16::MAX), uu)
                } else {
                    Piece::Block(lu, u32::try_from(u).unwrap_or(u32::MAX))
                };
                let at = Coordinate { model: POP.to_owned(), hash: short.clone(), piece };
                sel_all.push((sel, at.to_string().split_once('/').map(|(_, p)| p.to_owned()).unwrap_or_default()));
                let mut rec = Record::new(at, JOB, &run, &date);
                #[expect(clippy::cast_possible_truncation, reason = "f32 records")]
                {
                    rec.recording = Some((*x as f32, sel as f32));
                }
                recs.push(rec);
            }
        }
        lines_out.push(row);
    }
    sel_all.sort_by(|a, b| b.0.abs().total_cmp(&a.0.abs()));
    let _ = writeln!(
        out,
        "Most selective pieces (last-token activity, right minus wrong over their mean): {}.",
        sel_all.iter().take(12).map(|(s, p)| format!("{p} {s:+.3}")).collect::<Vec<_>>().join(", ")
    );
    let _ = writeln!(out, "Concentration per layer (activity over all tokens, mean over calibration pairs):");
    for r in lines_out {
        let _ = writeln!(out, "{r}");
    }
    // Merge into the judge atlas: each record's recording is set in place (other fields and
    // other jobs' records kept).
    let path = home.join(format!("designs/parmap/{POP}.json"));
    let mut atlas = load_atlas(home, &card_hash)?;
    for r in recs {
        put(&mut atlas, &r.at, &run, &date, |x| x.recording = r.recording);
    }
    atlas.calibration_sha256 = set_hash(&cal);
    save_atlas(home, &mut atlas)?;
    let _ = writeln!(
        out,
        "Atlas {} version {}: {} records ({} judge recordings), calibration hash {}. Total {:.1} s.",
        path.display(),
        atlas.version,
        atlas.records.len(),
        d.layers * (d.heads + blocks),
        atlas.calibration_sha256,
        t0.elapsed().as_secs_f64()
    );
    Ok(out)
}

/// One lesion: its atlas piece, the layer it starts at, and the run options that make it.
struct Lesion {
    piece: Piece,
    layer: usize,
    opts: RunOpts,
}

/// P(yes) of every pair under every lesion: `[lesion][pair]`, plus the full model's `[pair]`.
/// Pairs run one at a time: the full run keeps the input of every layer, and each lesion runs
/// only from its own layer up, seeded with that input (an exact copy on either device).
fn sweep(m: &Model, dev: Device, set: &[Pair], lesions: &[Lesion]) -> Result<(Vec<Vec<f32>>, Vec<f32>), Fault> {
    let d = m.pop.dims;
    let label = (m.role_node("label")?, "y".to_owned());
    let inputs: Vec<(String, String)> = (0..d.layers)
        .map(|l| {
            m.asm
                .producer(&nid(&format!("L{l}.attn_norm")), "x")
                .map(|(a, b)| (a.to_owned(), b.to_owned()))
                .ok_or_else(|| bad(format!("L{l} input unfed")))
        })
        .collect::<Result<_, _>>()?;
    let mut targets = vec![label.clone()];
    targets.extend(inputs.iter().cloned());
    let mut out = vec![Vec::with_capacity(set.len()); lesions.len()];
    let mut base = Vec::with_capacity(set.len());
    for p in set {
        let o = m.forward(dev, &p.ids, &targets, &RunOpts::default())?;
        base.push(values(o.get(&label).ok_or_else(|| bad("no label"))?)?.first().copied().unwrap_or(0.0));
        let pos: Vec<i32> = (0..i32::try_from(p.ids.len()).unwrap_or(0)).collect();
        for (k, les) in lesions.iter().enumerate() {
            let at = inputs.get(les.layer).ok_or_else(|| bad("layer"))?;
            let x = values(o.get(at).ok_or_else(|| bad("no layer input"))?)?.to_vec();
            let seeds = vec![(at.0.clone(), at.1.clone(), Act::Cpu(x)), (nid("pos"), "y".to_owned(), Act::Ids(pos.clone()))];
            let r = m.run(dev, seeds, std::slice::from_ref(&label), p.ids.len(), &les.opts)?;
            let v = values(r.get(&label).ok_or_else(|| bad("no label"))?)?.first().copied().unwrap_or(0.0);
            if let Some(o) = out.get_mut(k) {
                o.push(v);
            }
        }
    }
    Ok((out, base))
}

/// Mean and spread of |change| of P(yes), and the largest.
fn change(p: &[f32], base: &[f32]) -> (f32, f32, f32) {
    let d: Vec<f32> = p.iter().zip(base).map(|(a, b)| (a - b).abs()).collect();
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let n = d.len().max(1) as f32;
    let mean = d.iter().sum::<f32>() / n;
    let sd = (d.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n).sqrt();
    (mean, sd, d.iter().copied().fold(0.0, f32::max))
}

/// Step E: lesions. `e1 <from> <to> [questions]`: each layer's attention and feed-forward off in turn.
/// `e2 <from> <to> [questions]`: each key-value head group and each 64-channel block off in turn,
/// layers `from..=to`.
fn lesions(home: &Path, stamp: u64, step: &str, args: &[String]) -> Result<String, Fault> {
    let t0 = Instant::now();
    let (m, dev) = open(home)?;
    let d = m.pop.dims;
    let num = |i: usize, dflt: usize| args.get(i).and_then(|x| x.parse::<usize>().ok()).unwrap_or(dflt);
    let (from, to, q) = (num(3, 0), num(4, d.layers - 1).min(d.layers - 1), num(5, 3));
    let (cal, _) = pairs(home, &m)?;
    let set: Vec<Pair> = cal.into_iter().take(q * 2).collect();
    let per = d.heads / d.kv_heads.max(1);
    let blocks = d.inter / BLOCK;
    let mut ls = Vec::new();
    for l in from..=to {
        let lu = u16::try_from(l).unwrap_or(u16::MAX);
        let mut o = |piece: Piece, heads: Vec<usize>, blk: Vec<usize>| {
            let mut opts = RunOpts::default();
            if !heads.is_empty() {
                opts.heads_off.insert(l, heads);
            }
            if !blk.is_empty() {
                opts.blocks_off.insert(l, blk);
            }
            ls.push(Lesion { piece, layer: l, opts });
        };
        if step == "e1" {
            o(Piece::Attn(lu), (0..d.heads).collect(), Vec::new());
            o(Piece::Ffn(lu), Vec::new(), (0..blocks).collect());
        } else {
            for g in 0..d.kv_heads {
                o(Piece::Kv(lu, u16::try_from(g).unwrap_or(u16::MAX)), (g * per..(g + 1) * per).collect(), Vec::new());
            }
            for b in 0..blocks {
                o(Piece::Block(lu, u32::try_from(b).unwrap_or(u32::MAX)), Vec::new(), vec![b]);
            }
        }
    }
    // Seeded continuation check: with nothing switched off, a run from layer `to` equals the full run.
    ls.push(Lesion { piece: Piece::Layer(u16::try_from(to).unwrap_or(0)), layer: to, opts: RunOpts::default() });
    let (res, base) = sweep(&m, dev, &set, &ls)?;
    let check = res.last().is_some_and(|r| r.iter().zip(&base).all(|(a, b)| a.to_bits() == b.to_bits()));
    let mut out = String::new();
    let _ = writeln!(
        out,
        "Cartridge step E ({step}), lesions for the judge job ({POP}, {}): layers {from} to {to}, {} lesions over {} pairs ({q} calibration questions); full-model pair accuracy {:.3}. Seeded continuation identical to the full run: {check}.",
        dev.name(),
        ls.len() - 1,
        set.len(),
        pair_accuracy(&set, &base)
    );
    let card_hash = m.pop.card.sha_main.clone();
    let short: String = card_hash.chars().take(12).collect();
    let date = nmlearn::cartridge::date_of(stamp);
    let run = format!("runs/CART-{step}-{stamp}.txt");
    let mut atlas = load_atlas(home, &card_hash)?;
    let mut rows: Vec<(f32, f32, f64, String)> = Vec::new();
    for (les, p) in ls.iter().zip(&res).take(ls.len() - 1) {
        let (mean, sd, max) = change(p, &base);
        let at = Coordinate { model: POP.to_owned(), hash: short.clone(), piece: les.piece.clone() };
        let path = at.to_string().split_once('/').map(|(_, x)| x.to_owned()).unwrap_or_default();
        rows.push((mean, max, pair_accuracy(&set, p), path));
        put(&mut atlas, &at, &run, &date, |r| {
            r.lesion = Some(nmnpcs::atlas::Lesion { effect: mean, spread: sd, count: u32::try_from(set.len()).unwrap_or(0) })
        });
    }
    save_atlas(home, &mut atlas)?;
    if step == "e1" {
        let _ = writeln!(out, "Per layer: mean |change of P(yes)| (largest) and pair accuracy, attention off | feed-forward off:");
        for pair in rows.chunks(2) {
            if let [a, f] = pair {
                let _ = writeln!(
                    out,
                    "{:<4} attn {:.4} ({:.3}) acc {:.2} | ffn {:.4} ({:.3}) acc {:.2}",
                    a.3.split('/').next().unwrap_or(""),
                    a.0,
                    a.1,
                    a.2,
                    f.0,
                    f.1,
                    f.2
                );
            }
        }
    } else {
        let close = 0.003_f32;
        for l in from..=to {
            let tag = format!("L{l}/");
            let mine: Vec<&(f32, f32, f64, String)> = rows.iter().filter(|r| r.3.starts_with(&tag)).collect();
            let (mut gs, mut bs): (Vec<_>, Vec<_>) = mine.iter().copied().partition(|r| r.3.contains("/attn/"));
            bs.sort_by(|x, y| y.0.total_cmp(&x.0));
            gs.sort_by(|x, y| y.0.total_cmp(&x.0));
            let sum: f32 = bs.iter().map(|r| r.0).sum();
            let top8: f32 = bs.iter().take(8).map(|r| r.0).sum();
            let name = |r: &&(f32, f32, f64, String)| format!("{} {:.4}", r.3.rsplit('/').next().unwrap_or(""), r.0);
            let _ = writeln!(
                out,
                "L{l:>2}: groups {} | blocks above {close}: {} of {}; top 8 blocks hold {:.0}% of the summed block effect {sum:.4}; top blocks {}",
                gs.iter().map(name).collect::<Vec<_>>().join(", "),
                bs.iter().filter(|r| r.0 > close).count(),
                bs.len(),
                100.0 * top8 / sum.max(1e-12),
                bs.iter().take(4).map(name).collect::<Vec<_>>().join(", ")
            );
        }
    }
    let _ = writeln!(out, "Atlas version {}. Total {:.1} s.", atlas.version, t0.elapsed().as_secs_f64());
    Ok(out)
}

/// Timing probe: the full forward, a seeded run from layer 14, and both with a lesion.
fn timing(home: &Path) -> Result<String, Fault> {
    let t = Instant::now();
    let (m, dev) = open(home)?;
    let mut out = format!("open {:.0} ms on {}\n", t.elapsed().as_secs_f64() * 1000.0, dev.name());
    let (cal, _) = pairs(home, &m)?;
    let set: Vec<Pair> = cal.into_iter().take(2).collect();
    let mut off = RunOpts::default();
    off.blocks_off.insert(14, vec![3]);
    let mut hoff = RunOpts::default();
    hoff.heads_off.insert(14, vec![0, 1]);
    for (name, o) in
        [("plain", RunOpts::default()), ("plain", RunOpts::default()), ("blocks_off", off.clone()), ("heads_off", hoff.clone())]
    {
        let t = Instant::now();
        for p in &set {
            p_yes(&m, dev, p, &o)?;
        }
        let _ = writeln!(out, "forward {name}: {:.0} ms per pair", t.elapsed().as_secs_f64() * 500.0);
    }
    nmffi::prof::set(true);
    nmffi::prof::reset();
    if let Some(p) = set.first() {
        p_yes(&m, dev, p, &RunOpts::default())?;
    }
    nmffi::prof::set(false);
    let mut tab = nmffi::prof::table();
    tab.sort_by(|a, b| b.1.total_cmp(&a.1));
    let _ = writeln!(
        out,
        "stages of one forward: {}",
        tab.iter().take(10).map(|(n, s, c)| format!("{n} {:.0} ms x{c}", s * 1000.0)).collect::<Vec<_>>().join(", ")
    );
    let ls = vec![
        Lesion { piece: Piece::Layer(14), layer: 14, opts: RunOpts::default() },
        Lesion { piece: Piece::Layer(14), layer: 14, opts: off },
        Lesion { piece: Piece::Layer(14), layer: 14, opts: hoff },
    ];
    let t = Instant::now();
    sweep(&m, dev, &set, &ls)?;
    let _ = writeln!(out, "sweep of 3 seeded lesions from L14 plus full run: {:.0} ms per pair", t.elapsed().as_secs_f64() * 500.0);
    Ok(out)
}

/// Forward-only importance of every feed-forward block, `[layer][block]` (the Wanda score of
/// Sun et al. 2023, summed per block): for each channel, the norm of its column of the released
/// down projection times the norm of its SwiGLU output over the set's tokens, summed over the
/// block's 64 channels. No gradients, no lesions.
fn block_scores(m: &Model, dev: Device, set: &[Pair]) -> Result<Vec<Vec<f64>>, Fault> {
    let d = m.pop.dims;
    let targets: Vec<(String, String)> = (0..d.layers).map(|l| (nid(&format!("L{l}.swigate")), "y".to_owned())).collect();
    let mut act = vec![vec![0.0_f64; d.inter]; d.layers];
    for p in set {
        let o = m.forward(dev, &p.ids, &targets, &RunOpts::default())?;
        for (l, t) in targets.iter().enumerate() {
            let y = values(o.get(t).ok_or_else(|| bad("no swigate"))?)?;
            let Some(a) = act.get_mut(l) else { continue };
            for row in y.chunks(d.inter) {
                a.iter_mut().zip(row).for_each(|(s, x)| *s += f64::from(x * x));
            }
        }
    }
    let mut out = Vec::with_capacity(d.layers);
    for (l, a) in act.iter().enumerate() {
        let c = m.pop.cluster(&format!("{}/model.layers.{l}.mlp.down_proj.weight/whole", m.pop.card.id))?;
        let (rows, cols) = (c.shape.first().copied().unwrap_or(0), c.shape.get(1).copied().unwrap_or(0));
        if c.dtype != "BF16" || cols != d.inter {
            return Err(bad(format!("{}: expected BF16 [{}, {}]", c.tensor, d.hidden, d.inter)));
        }
        let v = m.pop.prov.acquire(&c)?;
        let w = v.tensor_bytes();
        let mut col = vec![0.0_f64; cols];
        for r in 0..rows {
            for (j, s) in col.iter_mut().enumerate() {
                let i = 2 * (r * cols + j);
                let x = f32::from_bits(
                    u32::from(u16::from_le_bytes([w.get(i).copied().unwrap_or(0), w.get(i + 1).copied().unwrap_or(0)])) << 16,
                );
                *s += f64::from(x * x);
            }
        }
        out.push(
            col.iter().zip(a).map(|(w2, x2)| w2.sqrt() * x2.sqrt()).collect::<Vec<f64>>().chunks(BLOCK).map(|b| b.iter().sum()).collect(),
        );
    }
    Ok(out)
}

/// Feed-forward sublayer lesion effects from the atlas (step E1), `[layer]`.
fn ffn_effects(home: &Path, m: &Model) -> Result<Vec<f32>, Fault> {
    let atlas = load_atlas(home, &m.pop.card.sha_main)?;
    (0..m.pop.dims.layers)
        .map(|l| {
            let lu = u16::try_from(l).unwrap_or(u16::MAX);
            atlas
                .records
                .iter()
                .find(|r| r.job == JOB && r.at.piece == Piece::Ffn(lu))
                .and_then(|r| r.lesion.as_ref())
                .map(|x| x.effect)
                .ok_or_else(|| bad(format!("no step E1 lesion for L{l} feed-forward: run `run cart e1` first")))
        })
        .collect()
}

/// A fixed pseudo-random order of `0..n` (xorshift, seeded).
fn shuffle(n: usize, seed: u64) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    for i in (1..n).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v.swap(i, usize::try_from(s % (i as u64 + 1)).unwrap_or(0));
    }
    v
}

/// Blocks to switch off per quiet layer: the `frac` lowest-scored (`low`), the highest (`high`)
/// or a random set of the same size (`rand<seed>`).
fn block_set(scores: &[Vec<f64>], quiet: &[usize], frac: f64, how: &str) -> RunOpts {
    let mut o = RunOpts::default();
    for &l in quiet {
        let Some(s) = scores.get(l) else { continue };
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss, reason = "block counts")]
        let k = (s.len() as f64 * frac).round() as usize;
        let mut idx: Vec<usize> = (0..s.len()).collect();
        idx.sort_by(|a, b| s.get(*a).unwrap_or(&0.0).total_cmp(s.get(*b).unwrap_or(&0.0)));
        let pick: Vec<usize> = match how {
            "low" => idx.into_iter().take(k).collect(),
            "high" => idx.into_iter().rev().take(k).collect(),
            r => shuffle(s.len(), r.trim_start_matches("rand").parse().unwrap_or(1) + l as u64).into_iter().take(k).collect(),
        };
        o.blocks_off.insert(l, pick);
    }
    o
}

/// Step E3: set lesion with random control. In every feed-forward sublayer whose own lesion moved
/// P(yes) by less than `QUIET` (step E1), switch off a share of blocks at once: the lowest-scored
/// by `block_scores`, the highest, and three random sets of the same size.
fn set_lesion(home: &Path, stamp: u64, args: &[String]) -> Result<String, Fault> {
    const QUIET: f32 = 0.01;
    let t0 = Instant::now();
    let (m, dev) = open(home)?;
    let q = args.get(3).and_then(|x| x.parse::<usize>().ok()).unwrap_or(3);
    let (cal, _) = pairs(home, &m)?;
    let set: Vec<Pair> = cal.into_iter().take(q * 2).collect();
    let eff = ffn_effects(home, &m)?;
    let quiet: Vec<usize> = eff.iter().enumerate().filter(|(_, e)| **e < QUIET).map(|(l, _)| l).collect();
    let scores = block_scores(&m, dev, &set)?;
    let base: Vec<f32> = set.iter().map(|p| p_yes(&m, dev, p, &RunOpts::default())).collect::<Result<_, _>>()?;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "Cartridge step E3, set lesion with random control (judge job, {}): {} pairs ({q} calibration questions); quiet feed-forward sublayers (step E1 effect under {QUIET}): {} of {} ({}). Units: 64-channel blocks ranked by block_scores (down-projection column norm times SwiGLU output norm). Mean |change of P(yes)| (largest), pair accuracy:",
        dev.name(),
        set.len(),
        quiet.len(),
        eff.len(),
        quiet.iter().map(|l| format!("L{l}")).collect::<Vec<_>>().join(" ")
    );
    let mut low_effect = Vec::new();
    for frac in [0.25, 0.5] {
        for how in ["low", "high", "rand1", "rand2", "rand3"] {
            let o = block_set(&scores, &quiet, frac, how);
            let p: Vec<f32> = set.iter().map(|x| p_yes(&m, dev, x, &o)).collect::<Result<_, _>>()?;
            let (mean, _, max) = change(&p, &base);
            if how == "low" {
                low_effect.push((frac, mean, o));
            }
            let _ = writeln!(
                out,
                "  {:>3.0}% of blocks off, {how:<5}: {mean:.4} ({max:.4}), accuracy {:.2}",
                frac * 100.0,
                pair_accuracy(&set, &p)
            );
        }
    }
    // The lowest-half set's effect goes into the atlas on each of its blocks.
    let card_hash = m.pop.card.sha_main.clone();
    let short: String = card_hash.chars().take(12).collect();
    let (date, run) = (nmlearn::cartridge::date_of(stamp), format!("runs/CART-e3-{stamp}.txt"));
    let mut atlas = load_atlas(home, &card_hash)?;
    if let Some((_, eff50, o)) = low_effect.last() {
        for (l, bs) in &o.blocks_off {
            for b in bs {
                let at = Coordinate {
                    model: POP.to_owned(),
                    hash: short.clone(),
                    piece: Piece::Block(u16::try_from(*l).unwrap_or(0), u32::try_from(*b).unwrap_or(0)),
                };
                put(&mut atlas, &at, &run, &date, |r| r.set_lesion = Some(*eff50));
            }
        }
    }
    save_atlas(home, &mut atlas)?;
    let _ = writeln!(out, "Atlas version {}. Total {:.1} s.", atlas.version, t0.elapsed().as_secs_f64());
    Ok(out)
}

/// Bytes of the distinct parameter clusters an assembly binds.
fn bytes_bound(m: &Model, a: &Assembly) -> Result<u64, Fault> {
    let ids: std::collections::BTreeSet<&String> = a.nodes.iter().flat_map(|n| n.clusters.iter()).collect();
    ids.iter().map(|r| m.pop.cluster(&format!("{}/{r}", m.pop.card.id)).map(|c| c.bytes)).sum()
}

/// Step F: the judge's base cartridge. On the calibration pairs, the largest share of
/// lowest-scored blocks (in the quiet feed-forward sublayers of step E1) whose removal keeps
/// every pair's P(yes) within `CLOSE` of the full model; that mask is built as an assembly of row
/// and column sub-clusters (the dropped blocks are never read), then checked on held-out pairs,
/// with its bytes; the same mask at 8 bits is checked too.
fn base_cartridge(home: &Path, stamp: u64, args: &[String]) -> Result<String, Fault> {
    const CLOSE: f32 = 0.003;
    let t0 = Instant::now();
    let (m, dev) = open(home)?;
    let d = m.pop.dims;
    let num = |i: usize, dflt: usize| args.get(i).and_then(|x| x.parse::<usize>().ok()).unwrap_or(dflt);
    let (cal, held) = pairs(home, &m)?;
    let cal: Vec<Pair> = cal.into_iter().take(num(3, 3) * 2).collect();
    let held: Vec<Pair> = held.into_iter().take(num(4, 6) * 2).collect();
    let eff = ffn_effects(home, &m)?;
    let quiet: Vec<usize> = eff.iter().enumerate().filter(|(_, e)| **e < 0.01).map(|(l, _)| l).collect();
    let scores = block_scores(&m, dev, &cal)?;
    let pc: Vec<f32> = cal.iter().map(|p| p_yes(&m, dev, p, &RunOpts::default())).collect::<Result<_, _>>()?;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "Cartridge step F, judge base cartridge ({POP}, {}): closeness bound |change of P(yes)| <= {CLOSE} on every pair; search on {} calibration pairs, check on {} held-out pairs; quiet feed-forward sublayers {}.",
        dev.name(),
        cal.len(),
        held.len(),
        quiet.len()
    );
    let mut chosen = (0.0, RunOpts::default());
    for frac in [0.25, 0.2, 0.15, 0.1, 0.05, 0.025] {
        let o = block_set(&scores, &quiet, frac, "low");
        let p: Vec<f32> = cal.iter().map(|x| p_yes(&m, dev, x, &o)).collect::<Result<_, _>>()?;
        let (mean, _, max) = change(&p, &pc);
        let pass = max <= CLOSE;
        let _ = writeln!(
            out,
            "  search {:>4.1}% of quiet blocks off: mean {mean:.4} largest {max:.4} -> {}",
            frac * 100.0,
            if pass { "within" } else { "outside" }
        );
        if pass {
            chosen = (frac, o);
            break;
        }
    }
    let (frac, mask) = chosen;
    let off_total: usize = mask.blocks_off.values().map(Vec::len).sum();
    let blocks = d.inter / BLOCK;
    let mut asm = Assembly::load(&home.join("designs/assemblies"), ASM)?;
    for (l, off) in &mask.blocks_off {
        let kept: Vec<usize> = (0..blocks).filter(|b| !off.contains(b)).collect();
        asm.keep_ffn_blocks(*l, &kept, BLOCK)?;
    }
    let full_asm = Assembly::load(&home.join("designs/assemblies"), ASM)?;
    let (full_bytes, cart_bytes) = (bytes_bound(&m, &full_asm)?, bytes_bound(&m, &asm)?);
    let mut q8 = asm.clone();
    q8.nodes.iter_mut().filter(|n| n.op == "linear").for_each(|n| "q8".clone_into(&mut n.precision));
    let cart = Model::bind(&m.pop, asm)?;
    let cart8 = Model::bind(&m.pop, q8)?;
    // The sub-cluster assembly against the run-time block mask on the calibration pairs.
    let pa: Vec<f32> = cal.iter().map(|x| p_yes(&cart, dev, x, &RunOpts::default())).collect::<Result<_, _>>()?;
    let pm: Vec<f32> = cal.iter().map(|x| p_yes(&m, dev, x, &mask)).collect::<Result<_, _>>()?;
    let (_, _, asm_vs_mask) = change(&pa, &pm);
    let ph: Vec<f32> = held.iter().map(|x| p_yes(&m, dev, x, &RunOpts::default())).collect::<Result<_, _>>()?;
    let pch: Vec<f32> = held.iter().map(|x| p_yes(&cart, dev, x, &RunOpts::default())).collect::<Result<_, _>>()?;
    let p8: Vec<f32> = held.iter().map(|x| p_yes(&cart8, dev, x, &RunOpts::default())).collect::<Result<_, _>>()?;
    let (hm, _, hx) = change(&pch, &ph);
    let (h8m, _, h8x) = change(&p8, &ph);
    #[expect(clippy::cast_precision_loss, reason = "bytes for a report")]
    let mb = |b: u64| b as f64 / 1_048_576.0;
    let _ = writeln!(
        out,
        "Chosen: {:.1}% of quiet blocks off ({off_total} of {} blocks in all layers). Sub-cluster assembly against the run-time mask on calibration: largest difference {asm_vs_mask:.5}.",
        frac * 100.0,
        d.layers * blocks
    );
    let _ = writeln!(
        out,
        "Bytes bound (16-bit, clusters as read): full {:.1} MB, cartridge {:.1} MB ({:.1}% of full); at 8 bits the linear weights take half.",
        mb(full_bytes),
        mb(cart_bytes),
        100.0 * mb(cart_bytes) / mb(full_bytes).max(1e-9)
    );
    let _ = writeln!(
        out,
        "Held out, 16-bit cartridge: mean {hm:.4} largest {hx:.4} -> {}; pair accuracy full {:.3}, cartridge {:.3}.",
        if hx <= CLOSE { "within" } else { "outside" },
        pair_accuracy(&held, &ph),
        pair_accuracy(&held, &pch)
    );
    let _ = writeln!(
        out,
        "Held out, 8-bit cartridge (every linear at q8): mean {h8m:.4} largest {h8x:.4} -> {}; pair accuracy {:.3}.",
        if h8x <= CLOSE { "within" } else { "outside" },
        pair_accuracy(&held, &p8)
    );
    let _ = writeln!(
        out,
        "Mask (blocks off per layer): {}",
        mask.blocks_off
            .iter()
            .map(|(l, b)| format!("L{l}:{}", b.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let _ = writeln!(out, "Total {:.1} s. Run {stamp}.", t0.elapsed().as_secs_f64());
    Ok(out)
}

/// The judge's log-odds z = logit(yes) - logit(no) of one pair (P(yes) is the logistic of z),
/// read from the head's last-position logits.
fn judge_z(m: &Model, dev: Device, p: &Pair, opts: &RunOpts) -> Result<f32, Fault> {
    let one = |s: &str| -> Result<usize, Fault> {
        match m.pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()))?.as_slice() {
            [id] => usize::try_from(*id).map_err(|e| bad(e.to_string())),
            _ => Err(bad(format!("label {s:?} is not one token"))),
        }
    };
    let head = (m.role_node("head")?, "logits".to_owned());
    let o = m.forward(dev, &p.ids, std::slice::from_ref(&head), opts)?;
    let l = values(o.get(&head).ok_or_else(|| bad("no logits"))?)?;
    let v = m.pop.dims.vocab;
    let last = l.get(l.len().saturating_sub(v)..).unwrap_or(&[]);
    Ok(last.get(one("yes")?).copied().unwrap_or(0.0) - last.get(one("no")?).copied().unwrap_or(0.0))
}

/// The judge's closeness check (2026-11-12, rework of step F): every pair's z within `tau` of
/// the full model, and every question's right document still above its wrong one. Returns
/// (largest |change of z|, mean, questions whose order holds, questions).
fn closeness(set: &[Pair], z: &[f32], base: &[f32]) -> (f32, f32, usize, usize) {
    let (mean, _, max) = change(z, base);
    let (mut held, mut n) = (0, 0);
    for (i, x) in set.iter().enumerate() {
        if x.right
            && let (Some(a), Some(b)) = (z.get(i), z.get(i + 1))
        {
            n += 1;
            held += usize::from(a > b);
        }
    }
    (max, mean, held, n)
}

/// Disposable state of the step F rework (`scratch/cart/judge-f1.txt`): calibration z of the
/// full model, the bound, and the block scores.
struct F1 {
    z: Vec<f32>,
    tau: f32,
    scores: Vec<Vec<f64>>,
}
fn f1_path(home: &Path) -> std::path::PathBuf {
    home.join("scratch/cart/judge-f1.txt")
}
fn f1_load(home: &Path) -> Result<F1, Fault> {
    let t = std::fs::read_to_string(f1_path(home)).map_err(|e| bad(format!("run `run cart f1` first: {e}")))?;
    let mut f = F1 { z: Vec::new(), tau: 0.0, scores: Vec::new() };
    for line in t.lines() {
        let mut w = line.split_whitespace();
        match w.next() {
            Some("z") => f.z = w.filter_map(|x| x.parse().ok()).collect(),
            Some("tau") => f.tau = w.next().and_then(|x| x.parse().ok()).unwrap_or(0.0),
            Some("score") => f.scores.push(w.filter_map(|x| x.parse().ok()).collect()),
            _ => {}
        }
    }
    Ok(f)
}

/// Rework of step F, part 1: on all calibration pairs, the full model's z and the 8-bit full
/// model's z; the bound `tau` is the largest |change of z| the 8-bit copy shows (the precision
/// the project already serves), so a cartridge must stay as close as that. Block scores on the
/// first 10 pairs.
fn f1(home: &Path) -> Result<String, Fault> {
    let t0 = Instant::now();
    let (m, dev) = open(home)?;
    let (cal, _) = pairs(home, &m)?;
    let mut q8 = Assembly::load(&home.join("designs/assemblies"), ASM)?;
    q8.nodes.iter_mut().filter(|n| n.op == "linear").for_each(|n| "q8".clone_into(&mut n.precision));
    let m8 = Model::bind(&m.pop, q8)?;
    let z: Vec<f32> = cal.iter().map(|p| judge_z(&m, dev, p, &RunOpts::default())).collect::<Result<_, _>>()?;
    let z8: Vec<f32> = cal.iter().map(|p| judge_z(&m8, dev, p, &RunOpts::default())).collect::<Result<_, _>>()?;
    let (tau, mean8, held8, n) = closeness(&cal, &z8, &z);
    let (_, _, heldf, _) = closeness(&cal, &z, &z);
    let scores = block_scores(&m, dev, cal.get(..10.min(cal.len())).unwrap_or(&[]))?;
    let mut txt = format!("z {}\ntau {tau}\n", z.iter().map(ToString::to_string).collect::<Vec<_>>().join(" "));
    for s in &scores {
        let _ = writeln!(txt, "score {}", s.iter().map(ToString::to_string).collect::<Vec<_>>().join(" "));
    }
    let p = f1_path(home);
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir).map_err(|e| bad(e.to_string()))?;
    }
    std::fs::write(&p, txt).map_err(|e| bad(e.to_string()))?;
    let zs: Vec<f32> = z.iter().map(|x| x.abs()).collect();
    let low4 = scores
        .iter()
        .filter(|s| {
            let mut i: Vec<usize> = (0..s.len()).collect();
            i.sort_by(|a, b| s.get(*a).unwrap_or(&0.0).total_cmp(s.get(*b).unwrap_or(&0.0)));
            i.iter().take(4).filter(|b| **b >= s.len().saturating_sub(4)).count() >= 2
        })
        .count();
    Ok(format!(
        "Step F rework, part 1 ({}): {} calibration pairs ({n} questions). Full model: right above wrong on {heldf} of {n}; |z| from {:.2} to {:.2}. 8-bit full model against it: largest |change of z| {tau:.4}, mean {mean8:.4}, order held on {held8} of {n}. Bound for a cartridge: tau = {tau:.4} on every pair and every order held. Block scores on 10 pairs; layers where at least 2 of the 4 lowest-scored blocks are among the last 4 blocks: {low4} of {}. State: {}. Total {:.1} s.\n",
        dev.name(),
        cal.len(),
        zs.iter().copied().fold(f32::MAX, f32::min),
        zs.iter().copied().fold(0.0, f32::max),
        scores.len(),
        p.display(),
        t0.elapsed().as_secs_f64()
    ))
}

/// Rework of step F, part 2: for each share given, the lowest-scored blocks off in the quiet
/// feed-forward sublayers (run-time mask), on all calibration pairs, against the bound.
fn f2(home: &Path, args: &[String]) -> Result<String, Fault> {
    let t0 = Instant::now();
    let (m, dev) = open(home)?;
    let st = f1_load(home)?;
    let (cal, _) = pairs(home, &m)?;
    let eff = ffn_effects(home, &m)?;
    let quiet: Vec<usize> = eff.iter().enumerate().filter(|(_, e)| **e < 0.01).map(|(l, _)| l).collect();
    let mut out = format!(
        "Step F rework, part 2 ({}): {} calibration pairs, bound tau {:.4} on |change of z|, orders held; {} quiet feed-forward sublayers.\n",
        dev.name(),
        cal.len(),
        st.tau,
        quiet.len()
    );
    for a in args.iter().skip(3) {
        let (f, how) = a.split_once(':').unwrap_or((a.as_str(), "low"));
        let Ok(frac) = f.parse::<f64>() else { continue };
        let o = block_set(&st.scores, &quiet, frac, how);
        let z: Vec<f32> = cal.iter().map(|p| judge_z(&m, dev, p, &o)).collect::<Result<_, _>>()?;
        let (max, mean, held, n) = closeness(&cal, &z, &st.z);
        let pass = max <= st.tau && held == n;
        let _ = writeln!(
            out,
            "  {:>5.1}% off ({how}): largest |change of z| {max:.4}, mean {mean:.4}, order held {held} of {n} -> {}",
            frac * 100.0,
            if pass { "within" } else { "outside" }
        );
    }
    let _ = writeln!(out, "Total {:.1} s.", t0.elapsed().as_secs_f64());
    Ok(out)
}

/// Rework of step F, part 3: the chosen share built as a sub-cluster assembly (dropped blocks
/// never read), checked on all held-out pairs against the full model, with its bytes.
fn f3(home: &Path, args: &[String]) -> Result<String, Fault> {
    let t0 = Instant::now();
    let (m, dev) = open(home)?;
    let st = f1_load(home)?;
    let frac: f64 = args.get(3).and_then(|x| x.parse().ok()).ok_or_else(|| bad("give the share, e.g. 0.25"))?;
    let (_, held) = pairs(home, &m)?;
    let eff = ffn_effects(home, &m)?;
    let quiet: Vec<usize> = eff.iter().enumerate().filter(|(_, e)| **e < 0.01).map(|(l, _)| l).collect();
    let mask = block_set(&st.scores, &quiet, frac, "low");
    let d = m.pop.dims;
    let blocks = d.inter / BLOCK;
    let mut asm = Assembly::load(&home.join("designs/assemblies"), ASM)?;
    for (l, off) in &mask.blocks_off {
        asm.keep_ffn_blocks(*l, &(0..blocks).filter(|b| !off.contains(b)).collect::<Vec<_>>(), BLOCK)?;
    }
    let full_bytes = bytes_bound(&m, &Assembly::load(&home.join("designs/assemblies"), ASM)?)?;
    let mut cart_bytes = bytes_bound(&m, &asm)?;
    // Precision of the cartridge's linear weights (`f16`, the release's 16 bits, or `q8`).
    let precision = args.get(4).map_or("f16", String::as_str);
    if precision == "q8" {
        let lin: std::collections::BTreeSet<&String> =
            asm.nodes.iter().filter(|n| n.op == "linear").flat_map(|n| n.clusters.iter()).collect();
        let lin_bytes: u64 =
            lin.iter().map(|r| m.pop.cluster(&format!("{}/{r}", m.pop.card.id)).map(|c| c.bytes)).sum::<Result<u64, Fault>>()?;
        cart_bytes -= lin_bytes / 2;
        asm.nodes.iter_mut().filter(|n| n.op == "linear").for_each(|n| "q8".clone_into(&mut n.precision));
    }
    let cart = Model::bind(&m.pop, asm)?;
    let z: Vec<f32> = held.iter().map(|p| judge_z(&m, dev, p, &RunOpts::default())).collect::<Result<_, _>>()?;
    let zc: Vec<f32> = held.iter().map(|p| judge_z(&cart, dev, p, &RunOpts::default())).collect::<Result<_, _>>()?;
    let (max, mean, held_n, n) = closeness(&held, &zc, &z);
    let (_, _, full_held, _) = closeness(&held, &z, &z);
    let pass = max <= st.tau && held_n == full_held;
    let off: usize = mask.blocks_off.values().map(Vec::len).sum();
    #[expect(clippy::cast_precision_loss, reason = "bytes for a report")]
    let mb = |b: u64| b as f64 / 1_048_576.0;
    Ok(format!(
        "Step F rework, part 3 ({}): {:.1}% of quiet blocks off ({off} of {} blocks), built as sub-clusters, linear weights at {precision}. Held out, {} pairs ({n} questions): largest |change of z| {max:.4} (bound {:.4}), mean {mean:.4}; right above wrong: full {full_held} of {n}, cartridge {held_n} of {n} -> {}. Bytes bound: full {:.1} MB, cartridge {:.1} MB ({:.1}%), {:.1} MB less ({} quiet feed-forward sublayers masked). Total {:.1} s.\nMask: {}\n",
        dev.name(),
        frac * 100.0,
        d.layers * blocks,
        held.len(),
        st.tau,
        if pass { "PASS" } else { "FAIL" },
        mb(full_bytes),
        mb(cart_bytes),
        100.0 * mb(cart_bytes) / mb(full_bytes).max(1e-9),
        mb(full_bytes.saturating_sub(cart_bytes)),
        quiet.len(),
        t0.elapsed().as_secs_f64(),
        mask.blocks_off
            .iter()
            .map(|(l, b)| format!("L{l}:{}", b.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")))
            .collect::<Vec<_>>()
            .join(" ")
    ))
}

/// The manifest of an assembly bound on a population (step I): one packet per distinct cluster,
/// at the node's precision, content-hashed from the weight file's tensor bytes, its region and
/// its precision (so equal content in two populations is one packet).
#[expect(clippy::too_many_arguments, reason = "the population, assembly, id, job, precision, atlas version and date of one manifest")]
fn manifest(
    pop: &std::sync::Arc<Population>,
    asm: &Assembly,
    id: &str,
    job: &str,
    precision: Option<&str>,
    atlas_version: u32,
    date: &str,
) -> Result<nmnpcs::cartridge::Manifest, Fault> {
    let mut tensor_hash: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    let mut seen = std::collections::BTreeSet::new();
    let mut packets = Vec::new();
    for n in &asm.nodes {
        let prec = precision.filter(|_| n.op == "linear").unwrap_or(&n.precision).to_owned();
        for r in &n.clusters {
            if !seen.insert((r.clone(), prec.clone())) {
                continue;
            }
            let c = pop.cluster(&format!("{}/{r}", pop.card.id))?;
            // Large tables are split into blocks of rows, each hashed on its own bytes, so rows
            // two populations share are one packet even when the tables differ in length.
            if let ([rows, k], true) = (c.shape.as_slice(), r.ends_with("/whole") && c.shape.first().is_some_and(|x| *x > 65_536)) {
                let v = pop.prov.acquire(&c)?;
                let w = usize::try_from(c.bytes).unwrap_or(0) / (rows * k).max(1);
                let stem = r.trim_end_matches("/whole");
                let mut a = 0;
                while a < *rows {
                    let b = (a + 8_192).min(*rows);
                    let all = v.tensor_bytes();
                    let bytes = all.get(a * k * w..b * k * w).unwrap_or(&[]);
                    let h = nmhash::sha256_bytes(bytes);
                    packets.push(nmnpcs::cartridge::PacketRef {
                        hash: nmhash::sha256_bytes(format!("{h}/{prec}").as_bytes()),
                        cluster: format!("{stem}/rows{a}-{b}"),
                        precision: prec.clone(),
                        bytes: u64::try_from(bytes.len()).unwrap_or(0),
                    });
                    a = b;
                }
                continue;
            }
            let th = match tensor_hash.get(&c.tensor) {
                Some(h) => h.clone(),
                None => {
                    let v = pop.prov.acquire(&c)?;
                    let h = nmhash::sha256_bytes(&v.tensor_bytes());
                    tensor_hash.insert(c.tensor.clone(), h.clone());
                    h
                }
            };
            let region = r.rsplit_once('/').map_or("whole", |(_, x)| x);
            let bytes = if prec == "q8" { c.bytes / 2 } else { c.bytes };
            packets.push(nmnpcs::cartridge::PacketRef {
                hash: nmhash::sha256_bytes(format!("{th}/{region}/{prec}").as_bytes()),
                cluster: r.clone(),
                precision: prec.clone(),
                bytes,
            });
        }
    }
    let layers = (0, u16::try_from(pop.dims.layers.saturating_sub(1)).unwrap_or(0));
    Ok(nmnpcs::cartridge::Manifest {
        id: id.into(),
        population: pop.card.id.clone(),
        base_sha256: pop.card.sha_main.clone(),
        atlas_version,
        job: job.into(),
        layers,
        mask_sha256: nmhash::sha256_bytes(b"empty mask"),
        closeness: String::new(),
        date: date.into(),
        packets,
    })
}

/// Steps I, J, K and M on the real weights: manifests of the base model, the reranker and the
/// judge's base cartridge (all units, 8 bits); the resident ledger holds shared packets once;
/// switch costs; a refused label falls back to the full model.
fn loading(home: &Path, stamp: u64) -> Result<String, Fault> {
    let t0 = Instant::now();
    let date = nmlearn::cartridge::date_of(stamp);
    let base = Population::open(home, "qwen3-0.6b", true, false)?;
    let rr = Population::open(home, POP, true, false)?;
    let dir = home.join("designs/assemblies");
    let atlas_v = load_atlas(home, &rr.card.sha_main).map(|a| a.version).unwrap_or(0);
    let mb = manifest(&base, &Assembly::load(&dir, "qwen3-reference")?, "base-full", "full", None, 0, &date)?;
    let mr = manifest(&rr, &Assembly::load(&dir, ASM)?, "judge-full", "judge", None, atlas_v, &date)?;
    let mut mj = manifest(&rr, &Assembly::load(&dir, ASM)?, "judge-base-q8", "judge", Some("q8"), atlas_v, &date)?;
    "within (step F rework: held out largest |change of z| 0.45, bound 0.78)".clone_into(&mut mj.closeness);
    let hash_s = t0.elapsed().as_secs_f64();
    let mut out = String::new();
    #[expect(clippy::cast_precision_loss, reason = "bytes for a report")]
    let m = |b: u64| b as f64 / 1_048_576.0;
    // I: round trip.
    let rt = [&mb, &mr, &mj].iter().all(|x| nmnpcs::cartridge::Manifest::from_text(&x.to_text()).is_ok_and(|y| y == **x));
    let _ = writeln!(
        out,
        "Cartridge loading steps I, J, K, M on the released weights ({date}); manifests hashed in {hash_s:.1} s on the processor."
    );
    let _ = writeln!(
        out,
        "I manifests: base-full {} packets {:.1} MB; judge-full {} packets {:.1} MB; judge-base-q8 {} packets {:.1} MB; text round trip exact: {rt}",
        mb.packets.len(),
        m(mb.bytes()),
        mr.packets.len(),
        m(mr.bytes()),
        mj.packets.len(),
        m(mj.bytes())
    );
    // J: shared packets.
    let shared: Vec<&nmnpcs::cartridge::PacketRef> = mr.packets.iter().filter(|p| mb.packets.iter().any(|q| q.hash == p.hash)).collect();
    let mut res = nmnpcs::cartridge::Resident::default();
    let a = res.load(&mb);
    let k = res.switch_cost(&mr);
    let b = res.load(&mr);
    let union = res.bytes();
    let j_ok = union == mb.bytes() + mr.bytes() - shared.iter().map(|p| p.bytes).sum::<u64>() && b == k;
    let _ = writeln!(
        out,
        "J shared packets: {} packets of the reranker are byte-identical to the base ({:.1} MB: {}); resident after both {:.1} MB against {:.1} MB loaded separately -> {}",
        shared.len(),
        m(shared.iter().map(|p| p.bytes).sum()),
        shared.iter().map(|p| p.cluster.as_str()).take(4).collect::<Vec<_>>().join(", "),
        m(union),
        m(mb.bytes() + mr.bytes()),
        if j_ok { "PASS" } else { "FAIL" }
    );
    // K: switch costs.
    let freed = res.release("judge-full");
    let to_q8 = res.switch_cost(&mj);
    let k_ok = to_q8 == mj.packets.iter().filter(|p| !mb.packets.iter().any(|q| q.hash == p.hash)).map(|p| p.bytes).sum::<u64>();
    let _ = writeln!(
        out,
        "K switch cost: base loaded {:.1} MB; base to judge-full moves {:.1} MB; judge released frees {:.1} MB; base to judge-base-q8 moves {:.1} MB (only packets the base does not hold) -> {}",
        m(a),
        m(k),
        m(freed),
        m(to_q8),
        if k_ok { "PASS" } else { "FAIL" }
    );
    // M: label and fallback.
    let mut r2 = nmnpcs::cartridge::Resident::default();
    let (got, _, why) = r2.load_or_fallback(&mj, &mr, &base.card.id, &base.card.sha_main, atlas_v);
    let (ok_got, _, ok_why) = nmnpcs::cartridge::Resident::default().load_or_fallback(&mj, &mr, &rr.card.id, &rr.card.sha_main, atlas_v);
    let m_ok = got.id == "judge-full" && why.is_some() && ok_got.id == "judge-base-q8" && ok_why.is_none();
    let _ = writeln!(
        out,
        "M fallback: the judge cartridge offered to the base weights is refused ({}) and the full judge loads; on its own weights it loads -> {}",
        why.unwrap_or_default(),
        if m_ok { "PASS" } else { "FAIL" }
    );
    std::fs::create_dir_all(home.join(BASE_CARTRIDGES)).map_err(|e| bad(e.to_string()))?;
    std::fs::write(home.join(BASE_CARTRIDGES).join("judge-base-q8.json"), mj.to_text()).map_err(|e| bad(e.to_string()))?;
    let _ = writeln!(
        out,
        "Manifest written: crates/nmlab/data/cartridges/judge-base-q8.json (a record about the weights: hashes, names and sizes, no parameter values). Total {:.1} s.",
        t0.elapsed().as_secs_f64()
    );
    Ok(out)
}

/// `run cart p` (2026-11-13, cartridges for every part; research file, cartridge section: the
/// drones' shared early layers one shared base cartridge, overlapping cartridges share packets by
/// hash): manifests of the drones' shared early components (the base model's layers through the
/// last layer any drone head reads, with the token table), each tentacle's own assembly, the
/// embedder (full and 8 bits) and the writer (the base model at 8 bits); then the resident ledger
/// over the organism's set (drones, eight tentacles, the judge's 8-bit cartridge, the embedder at 8
/// bits, the writer): what is held once against loaded separately, and the switch cost from the
/// drones to each part. Manifests written under `crates/nmlab/data/cartridges/`. Processor only.
fn parts(home: &Path, stamp: u64) -> Result<String, Fault> {
    let t0 = Instant::now();
    let date = nmlearn::cartridge::date_of(stamp);
    let dir = home.join("designs/assemblies");
    let base = Population::open(home, "qwen3-0.6b", true, false)?;
    let emb = Population::open(home, "qwen3-embedding-0.6b", true, false)?;
    let rr = Population::open(home, POP, true, false)?;
    let last = nmapi::tentacle_pass::drone_last_layer();
    let mut drone_asm = Assembly::load(&dir, "qwen3-reference")?;
    drone_asm.nodes.retain(|n| n.layer.is_none_or(|l| l <= last) && !n.id.ends_with("#head") && !n.id.ends_with("#final_norm"));
    let mut drones = manifest(&base, &drone_asm, "drones-shared", "drones", None, 0, &date)?;
    drones.layers = (0, u16::try_from(last).unwrap_or(0));
    let mut tentacles = Vec::new();
    for t in 1..=8 {
        let a = Assembly::load(&dir, &format!("tentacle-t{t}"))?;
        let mut m = manifest(&base, &a, &format!("tentacle-t{t}"), "tentacle", None, 0, &date)?;
        m.layers = (0, u16::try_from(last).unwrap_or(0));
        tentacles.push(m);
    }
    let emb_asm = Assembly::load(&dir, "qwen3-embedding")?;
    let emb_full = manifest(&emb, &emb_asm, "embedder-full", "embed", None, 0, &date)?;
    let emb_q8 = manifest(&emb, &emb_asm, "embedder-q8", "embed", Some("q8"), 0, &date)?;
    let writer_q8 = manifest(&base, &Assembly::load(&dir, "qwen3-reference")?, "writer-q8", "write", Some("q8"), 0, &date)?;
    let atlas_v = load_atlas(home, &rr.card.sha_main).map(|a| a.version).unwrap_or(0);
    let judge_q8 = manifest(&rr, &Assembly::load(&dir, ASM)?, "judge-base-q8", "judge", Some("q8"), atlas_v, &date)?;
    #[expect(clippy::cast_precision_loss, reason = "bytes for a report")]
    let mb = |b: u64| b as f64 / 1_048_576.0;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "Cartridges for every part ({date}); drones read through layer {last}; manifests hashed in {:.1} s on the processor.",
        t0.elapsed().as_secs_f64()
    );
    let all: Vec<&nmnpcs::cartridge::Manifest> =
        std::iter::once(&drones).chain(tentacles.iter()).chain([&emb_full, &emb_q8, &writer_q8, &judge_q8]).collect();
    for m in &all {
        let rt = nmnpcs::cartridge::Manifest::from_text(&m.to_text()).is_ok_and(|y| y == **m);
        let shared = m.packets.iter().filter(|p| drones.packets.iter().any(|q| q.hash == p.hash)).map(|p| p.bytes).sum::<u64>();
        let _ = writeln!(
            out,
            "{}: {} packets, {:.1} MB, layers {}..{}, shares {:.1} MB with the drones' cartridge; text round trip {rt}",
            m.id,
            m.packets.len(),
            mb(m.bytes()),
            m.layers.0,
            m.layers.1,
            mb(shared)
        );
    }
    // The organism's set resident at once.
    let set: Vec<&nmnpcs::cartridge::Manifest> =
        std::iter::once(&drones).chain(tentacles.iter()).chain([&judge_q8, &emb_q8, &writer_q8]).collect();
    let mut res = nmnpcs::cartridge::Resident::default();
    let separate: u64 = set.iter().map(|m| m.bytes()).sum();
    let _ = res.load(&drones);
    let mut costs = Vec::new();
    for m in set.iter().skip(1) {
        costs.push(format!("{} {:.1} MB", m.id, mb(res.switch_cost(m))));
        res.load(m);
    }
    let union = res.bytes();
    let _ = writeln!(
        out,
        "Resident set (drones, 8 tentacles, judge 8-bit, embedder 8-bit, writer 8-bit): held once {:.1} MB against {:.1} MB loaded separately ({:.0}% saved by shared packets)",
        mb(union),
        mb(separate),
        100.0 * (1.0 - union as f64 / separate.max(1) as f64)
    );
    let _ = writeln!(out, "Bytes moved per part after the drones (in load order): {}", costs.join("; "));
    std::fs::create_dir_all(home.join(BASE_CARTRIDGES)).map_err(|e| bad(e.to_string()))?;
    for m in all.iter().filter(|m| m.id != "judge-base-q8") {
        std::fs::write(home.join(BASE_CARTRIDGES).join(format!("{}.json", m.id)), m.to_text()).map_err(|e| bad(e.to_string()))?;
    }
    let _ = writeln!(
        out,
        "Manifests written under crates/nmlab/data/cartridges/ (names, sizes, hashes; no parameter values). Total {:.1} s.",
        t0.elapsed().as_secs_f64()
    );
    Ok(out)
}

/// `run cart pc <spec>` (2026-11-14, precision down to 4 bits): the judge's closeness at a
/// precision. The full judge is bound with no precision setting; then `spec` is applied (a
/// `--precision` value such as `reranker=4` or `4`, or `map:<file>` for a per-piece map) and a
/// second copy is bound; on all calibration pairs, the largest and mean change of the yes-or-no
/// log-odds, the pairs still ordered right, and the seconds per pair of each copy.
fn precision_closeness(home: &Path, args: &[String]) -> Result<String, Fault> {
    let spec = args.get(3).cloned().unwrap_or_else(|| "reranker=8".to_owned());
    let (m, dev) = open(home)?;
    let (cal, _) = pairs(home, &m)?;
    if let Some(f) = spec.strip_prefix("map:") {
        let mp = nmengn::model::read_precision_map(std::path::Path::new(f)).map_err(bad)?;
        if let Ok(mut g) = nmengn::model::PRECISION_MAP.write() {
            *g = Some(mp);
        }
    } else if let Ok(mut g) = nmengn::model::PRECISION.write() {
        *g = nmengn::model::Precision::parse(&spec).map_err(bad)?;
    }
    let m2 = Model::bind(&m.pop, Assembly::load(&home.join("designs/assemblies"), ASM)?)?;
    let t0 = Instant::now();
    let z: Vec<f32> = cal.iter().map(|p| judge_z(&m, dev, p, &RunOpts::default())).collect::<Result<_, _>>()?;
    let s_full = t0.elapsed().as_secs_f64() / cal.len().max(1) as f64;
    let t1 = Instant::now();
    let z2: Vec<f32> = cal.iter().map(|p| judge_z(&m2, dev, p, &RunOpts::default())).collect::<Result<_, _>>()?;
    let s_two = t1.elapsed().as_secs_f64() / cal.len().max(1) as f64;
    let (tau, mean, held, n) = closeness(&cal, &z2, &z);
    let (_, _, heldf, _) = closeness(&cal, &z, &z);
    let precisions: std::collections::BTreeMap<String, usize> =
        m2.asm.nodes.iter().filter(|x| x.op == "linear").fold(std::collections::BTreeMap::new(), |mut acc, x| {
            *acc.entry(x.precision.clone()).or_insert(0) += 1;
            acc
        });
    Ok(format!(
        "Judge closeness at precision {spec} (2026-11-14): {} calibration pairs; linears by precision {precisions:?}
full judge: right above wrong on {heldf} of {n}; {s_full:.3} s per pair
second copy: largest |change of z| {tau:.4}, mean {mean:.4}; right above wrong on {held} of {n}; {s_two:.3} s per pair
",
        cal.len()
    ))
}

/// `run cart q4err` (2026-11-14, precision down to 4 bits, the first short measure): for linear
/// weights of layers 0, 9, 18 and 27 of each released population, the relative root-mean-square
/// error against the released bfloat16 values of: binary16 (the engine's 16-bit path, converted
/// once), 8 bits per output row (today's), and 4 bits round-to-nearest with one binary16 scale per
/// group of 32, 64 or 128 along the input (values -8 to 7, as llama.cpp's Q4 blocks). Processor,
/// nothing written but the report.
fn q4_error(home: &Path) -> Result<String, Fault> {
    let mut out = String::from(
        "Weight error by precision (2026-11-14), relative RMS against the released bfloat16, linear weights of layers 0, 9, 18, 27\npopulation | tensors | binary16 | 8-bit per row | 4-bit g32 | 4-bit g64 | 4-bit g128 | binary16 values changed\n",
    );
    let f16 = |x: f32| nmengn::half::r16(x);
    for pop in ["qwen3-0.6b", "qwen3-reranker-0.6b", "qwen3-embedding-0.6b", "qwen3-1.7b"] {
        let idx = nmsafe::TensorIndex::open(&nmtype::models_dir(home).join(pop).join("files")).map_err(|e| bad(e.to_string()))?;
        let (mut n, mut e) = (0_usize, [0.0_f64; 5]);
        let mut changed = (0_u64, 0_u64);
        for (name, t) in &idx.tensors {
            let layer_ok = ["layers.0.", "layers.9.", "layers.18.", "layers.27."].iter().any(|l| name.contains(l));
            if !layer_ok || !name.ends_with("proj.weight") || t.shape.len() != 2 || t.dtype != "BF16" {
                continue;
            }
            let k = usize::try_from(t.shape[1]).unwrap_or(1);
            let bytes = t.read_range(0, usize::try_from(t.end - t.start).unwrap_or(0)).map_err(|e| bad(e.to_string()))?;
            let w: Vec<f32> = bytes.as_chunks::<2>().0.iter().map(|s| nmsafe::bfloat16_to_f32(u16::from_le_bytes(*s))).collect();
            let norm: f64 = w.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().max(1e-30);
            let rel = |d: &[f32]| {
                (w.iter()
                    .zip(d)
                    .map(|(a, b)| {
                        let x = f64::from(*a - *b);
                        x * x
                    })
                    .sum::<f64>()
                    / norm)
                    .sqrt()
            };
            let h: Vec<f32> = w.iter().map(|x| f16(*x)).collect();
            changed.0 += w.iter().zip(&h).filter(|(a, b)| a.to_bits() != b.to_bits()).count() as u64;
            changed.1 += w.len() as u64;
            let mut q8 = vec![0.0_f32; w.len()];
            for (row, dst) in w.chunks_exact(k).zip(q8.chunks_exact_mut(k)) {
                let s = f16(row.iter().fold(0.0_f32, |m, x| m.max(x.abs())).max(1e-8) / 127.0);
                for (a, d) in row.iter().zip(dst.iter_mut()) {
                    *d = f16((a / s).round().clamp(-127.0, 127.0) * s);
                }
            }
            let q4 = |g: usize| -> Vec<f32> {
                let mut d = vec![0.0_f32; w.len()];
                for (blk, dst) in w.chunks(g).zip(d.chunks_mut(g)) {
                    let s = f16(blk.iter().fold(0.0_f32, |m, x| m.max(x.abs())).max(1e-8) / 7.0);
                    for (a, o) in blk.iter().zip(dst.iter_mut()) {
                        *o = f16((a / s).round().clamp(-8.0, 7.0) * s);
                    }
                }
                d
            };
            let r = [rel(&h), rel(&q8), rel(&q4(32)), rel(&q4(64)), rel(&q4(128))];
            for (acc, x) in e.iter_mut().zip(r) {
                *acc += x;
            }
            n += 1;
        }
        let m = n.max(1) as f64;
        let _ = writeln!(
            out,
            "{pop} | {n} | {:.5} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4}",
            e[0] / m,
            e[1] / m,
            e[2] / m,
            e[3] / m,
            e[4] / m,
            changed.0 as f64 / changed.1.max(1) as f64
        );
    }
    Ok(out)
}

/// `run cart e8 [n]` (2026-11-13): the embedder's 8-bit closeness check. The released
/// Qwen3-Embedding-0.6B bound as released and with its linear weights at 8 bits, on the card;
/// `n` chunks of this tree's own Rust sources (default 48, read from `crates/`, in path order) are
/// embedded as documents, and one query per chunk is formed from the first name the chunk defines
/// ("where is <name> defined", card query format). Measured: the cosine of each 8-bit vector with
/// its full-precision vector (documents and queries), and retrieval: for each query, the full
/// model's first document against the 8-bit model's, and the overlap of their first five.
/// Bound fixed before the run (the judge's step F rule in kind): within when the lowest cosine is
/// at least 0.99 and the first document agrees on at least 95% of the queries.
fn embed_q8(home: &Path, args: &[String]) -> Result<String, Fault> {
    let n: usize = args.get(3).and_then(|x| x.parse().ok()).unwrap_or(48);
    let t0 = Instant::now();
    let mut docs: Vec<(String, String, String)> = Vec::new();
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    let mut stack = vec![home.join("crates")];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                files.push(p);
            }
        }
    }
    files.sort();
    'outer: for f in &files {
        let Ok(text) = std::fs::read_to_string(f) else { continue };
        let rel = f.strip_prefix(home).map_or_else(|_| f.display().to_string(), |r| r.display().to_string().replace('\\', "/"));
        let lines: Vec<&str> = text.lines().collect();
        for (i, l) in lines.iter().enumerate() {
            if let Some(name) = nmmem::migrate::defined_name(l).filter(|n| n.len() > 3 && l.trim_start().starts_with("pub fn")) {
                let body = lines.get(i..(i + 30).min(lines.len())).unwrap_or(&[]).join("\n");
                docs.push((rel.clone(), body, name));
                if docs.len() >= n {
                    break 'outer;
                }
                break;
            }
        }
    }
    let dir = home.join("designs/assemblies");
    let bind = |q8: bool| -> Result<std::sync::Arc<Model>, Fault> {
        let pop = Population::open(home, "qwen3-embedding-0.6b", true, true)?;
        let mut a = Assembly::load(&dir, "qwen3-embedding")?;
        if q8 {
            for nd in a.nodes.iter_mut().filter(|nd| nd.op == "linear") {
                // `run cart e8 [n] --bits 4` (2026-11-14): the same check at 4 bits.
                if args.iter().any(|x| x == "--bits") && args.windows(2).any(|w| w[0] == "--bits" && w[1] == "4") {
                    "q4".clone_into(&mut nd.precision)
                } else {
                    "q8".clone_into(&mut nd.precision)
                }
            }
        }
        Ok(std::sync::Arc::new(Model::bind(&pop, a)?))
    };
    // Document vectors and query vectors.
    type Vectors = (Vec<Vec<f32>>, Vec<Vec<f32>>);
    let embed_all = |m: &Model| -> Result<Vectors, Fault> {
        let dev = if m.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
        let d: Vec<Vec<f32>> =
            docs.iter().map(|(p, t, _)| nmengn::roles::card_embed_on(m, dev, &nmsubs::dense::doc_text(p, t))).collect::<Result<_, _>>()?;
        let q: Vec<Vec<f32>> = docs
            .iter()
            .map(|(_, _, name)| {
                nmengn::roles::card_embed_on(
                    m,
                    dev,
                    &nmengn::roles::card_query(nmsubs::dense::QUERY_TASK, &format!("where is {name} defined")),
                )
            })
            .collect::<Result<_, _>>()?;
        Ok((d, q))
    };
    let full = bind(false)?;
    let (df, qf) = embed_all(&full)?;
    drop(full);
    let q8 = bind(true)?;
    let (d8, q8v) = embed_all(&q8)?;
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let cos = |a: &[f32], b: &[f32]| dot(a, b) / (dot(a, a).sqrt() * dot(b, b).sqrt()).max(1e-12);
    let cs: Vec<f32> = df.iter().zip(&d8).chain(qf.iter().zip(&q8v)).map(|(a, b)| cos(a, b)).collect();
    let low = cs.iter().copied().fold(1.0_f32, f32::min);
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let mean = cs.iter().sum::<f32>() / cs.len().max(1) as f32;
    let rank = |q: &[f32], ds: &[Vec<f32>]| -> Vec<usize> {
        let mut v: Vec<usize> = (0..ds.len()).collect();
        v.sort_by(|a, b| dot(q, &ds[*b]).total_cmp(&dot(q, &ds[*a])).then(a.cmp(b)));
        v
    };
    let (mut agree, mut overlap, mut right_f, mut right_8) = (0_usize, 0_usize, 0_usize, 0_usize);
    for (i, (a, b)) in qf.iter().zip(&q8v).enumerate() {
        let (rf, r8) = (rank(a, &df), rank(b, &d8));
        agree += usize::from(rf.first() == r8.first());
        overlap += rf.iter().take(5).filter(|x| r8.iter().take(5).any(|y| y == *x)).count();
        right_f += usize::from(rf.first() == Some(&i));
        right_8 += usize::from(r8.first() == Some(&i));
    }
    let q = docs.len().max(1);
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let within = low >= 0.99 && agree as f64 >= 0.95 * q as f64;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "Embedder 8-bit closeness (2026-11-13): Qwen3-Embedding-0.6B as released against linear weights at 8 bits, on the card; {} chunks of this tree's Rust sources and one query each (\"where is <name> defined\", card query format).",
        docs.len()
    );
    let _ = writeln!(out, "cosine of 8-bit with full vectors over {} vectors: lowest {low:.4}, mean {mean:.4}", cs.len());
    let _ = writeln!(
        out,
        "retrieval: first document agrees on {agree} of {q} queries; first-five overlap {:.2} of 5; the defining chunk first: full {right_f} of {q}, 8-bit {right_8} of {q}",
        overlap as f64 / q as f64
    );
    let _ = writeln!(
        out,
        "bound fixed before the run: lowest cosine at least 0.99 and first document agreeing on at least 95% -> {}",
        if within { "WITHIN: the 8-bit embedder cartridge may serve" } else { "NOT WITHIN: the embedder stays as released" }
    );
    let _ = writeln!(out, "total {:.1} s", t0.elapsed().as_secs_f64());
    Ok(out)
}

/// `run cart l2` (2026-11-13, loading step L, second part: the second device stream with pinned
/// memory). On the card: (1) exactness: the base model's logits for a fixed prompt after its
/// weights were staged on the copy stream equal, bit for bit, the logits after the ordinary
/// upload; (2) overlap: eight reranker jobs on the default stream alone, the base model's staging
/// alone, then both at once (staging on a second thread): wall time against the sum.
fn stream_stage(home: &Path) -> Result<String, Fault> {
    use nmengn::model::RunOpts;
    let dir = home.join("designs/assemblies");
    let base_pop = Population::open(home, "qwen3-0.6b", true, true)?;
    let gpu = base_pop.gpu.as_ref().ok_or_else(|| bad("no GPU"))?.gpu.clone();
    let base = std::sync::Arc::new(Model::bind(&base_pop, Assembly::load(&dir, "qwen3-reference")?)?);
    let ids = base.pop.tokenizer.encode("fn main() { let total = items.iter().map(|x| x.cost).sum(); }").map_err(|e| bad(e.to_string()))?;
    let target = (base.role_node("head")?, "logits".to_owned());
    let logits = |m: &Model| -> Result<Vec<f32>, Fault> {
        let out = m.forward(Device::Gpu, &ids, std::slice::from_ref(&target), &RunOpts::default())?;
        Ok(nmengn::model::values(out.get(&target).ok_or_else(|| bad("no logits"))?)?.to_vec())
    };
    let plain = logits(&base)?;
    base.pop.release_device();
    let st = gpu.copy_stream().map_err(bad)?;
    let s = base.stage(&st)?;
    let staged = logits(&base)?;
    let exact = plain.len() == staged.len() && plain.iter().zip(&staged).all(|(a, b)| a.to_bits() == b.to_bits());
    let mut out = String::new();
    let _ = writeln!(out, "Second device stream with pinned memory (2026-11-13), on the {}.", gpu.info().name);
    let _ = writeln!(
        out,
        "(1) staged {} tensors, {:.1} MB (host preparation {:.2} s, copies {:.2} s, {:.0} MB/s); logits after staging bit-identical to the ordinary upload: {exact}",
        s.tensors,
        s.bytes as f64 / 1_048_576.0,
        s.host_s,
        s.copy_s,
        s.bytes as f64 / 1_048_576.0 / s.copy_s.max(1e-9)
    );
    // (2) overlap.
    let rr = nmapi::neural::reranker_model(home)?;
    let docs: Vec<String> = (0..8).map(|i| format!("<doc{i}> fn step_{i}(x: u32) -> u32 {{ x * {i} + 1 }}")).collect();
    let jobs = |n: usize| -> Result<f64, Fault> {
        let t = Instant::now();
        for _ in 0..n {
            let _ = nmengn::roles::card_rerank_set(&rr, Device::Gpu, nmengn::roles::CODE_INSTRUCTION, "which function adds one", &docs)?;
        }
        Ok(t.elapsed().as_secs_f64())
    };
    let _ = jobs(1)?;
    let j = jobs(8)?;
    base.pop.release_device();
    let t = Instant::now();
    let _ = base.stage(&st)?;
    let alone = t.elapsed().as_secs_f64();
    base.pop.release_device();
    let t = Instant::now();
    let b2 = std::sync::Arc::clone(&base);
    let st2 = gpu.copy_stream().map_err(bad)?;
    let h = std::thread::spawn(move || b2.stage(&st2).map(|x| x.tensors));
    let j2 = jobs(8)?;
    let staged_n = h.join().map_err(|_| bad("staging thread"))??;
    let both = t.elapsed().as_secs_f64();
    let _ = writeln!(
        out,
        "(2) eight reranker jobs alone {j:.2} s; base staging alone {alone:.2} s; sum {:.2} s; both at once {both:.2} s (jobs {j2:.2} s, {staged_n} tensors staged): {:.0}% of the sum",
        j + alone,
        100.0 * both / (j + alone).max(1e-9)
    );
    Ok(out)
}

/// `run cart eb [n] [max_tokens]` (2026-11-13, batched embedding): `n` chunks of this tree's Rust
/// sources (default 48) embedded one at a time (`card_embed_on_capped`, 128 tokens) and batched
/// (`card_embed_batch`, packed up to `max_tokens`, default 2048) by the released embedder at 8 bits
/// on the card: lowest cosine between the two vectors of each chunk, and seconds per chunk each way.
fn embed_batch(home: &Path, args: &[String]) -> Result<String, Fault> {
    let n: usize = args.get(3).and_then(|x| x.parse().ok()).unwrap_or(48);
    let max_tokens: usize = args.get(4).and_then(|x| x.parse().ok()).unwrap_or(2048);
    nmapi::neural::EMBED_Q8.store(true, std::sync::atomic::Ordering::Relaxed);
    let m = nmapi::neural::embedder_model(home)?;
    let mut texts: Vec<String> = Vec::new();
    let mut stack = vec![home.join("crates")];
    let mut files = Vec::new();
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p)
            } else if p.extension().is_some_and(|x| x == "rs") {
                files.push(p)
            }
        }
    }
    files.sort();
    'outer: for f in &files {
        let Ok(text) = std::fs::read_to_string(f) else { continue };
        let lines: Vec<&str> = text.lines().collect();
        for k in 0..(lines.len() / 30).min(3) {
            texts.push(nmsubs::dense::doc_text(&f.display().to_string(), &lines[k * 30..(k + 1) * 30].join("\n")));
            if texts.len() >= n {
                break 'outer;
            }
        }
    }
    let _ = nmengn::roles::card_embed_on_capped(&m, Device::Gpu, &texts[0], Some(128))?;
    let t = Instant::now();
    let one: Vec<Vec<f32>> =
        texts.iter().map(|x| nmengn::roles::card_embed_on_capped(&m, Device::Gpu, x, Some(128))).collect::<Result<_, _>>()?;
    let t_one = t.elapsed().as_secs_f64();
    let _ = nmengn::roles::card_embed_batch(&m, &texts[..2], 128, max_tokens)?;
    let t = Instant::now();
    let many = nmengn::roles::card_embed_batch(&m, &texts, 128, max_tokens)?;
    let t_many = t.elapsed().as_secs_f64();
    // `prof` as the fifth argument: one more batched pass with per-kernel timing, table to stderr.
    if args.get(5).map(String::as_str) == Some("prof") {
        nmffi::prof::reset();
        nmffi::prof::set(true);
        nmffi::prof::set_kernels(true);
        let _ = nmengn::roles::card_embed_batch(&m, &texts, 128, max_tokens)?;
        nmffi::prof::set_kernels(false);
        nmffi::prof::set(false);
        let mut rows: Vec<(String, f64, u64)> = nmffi::prof::side().into_iter().chain(nmffi::prof::table()).collect();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (k, s, n) in rows.iter().take(14) {
            eprintln!("prof {k}: {s:.3} s over {n}");
        }
    }
    let cos = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let low = one.iter().zip(&many).map(|(a, b)| cos(a, b)).fold(1.0_f32, f32::min);
    let norms: Vec<String> =
        many.iter().take(4).map(|v| format!("{:.3}/{}", v.iter().map(|x| x * x).sum::<f32>().sqrt(), v.len())).collect();
    eprintln!(
        "batched norms {norms:?}; single norms {:?}",
        one.iter().take(4).map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt()).collect::<Vec<_>>()
    );
    let k = texts.len() as f64;
    Ok(format!(
        "Batched embedding (2026-11-13): {} chunks, 128 tokens each, embedder at 8 bits on the card; packed up to {max_tokens} tokens per forward.\none at a time {t_one:.2} s ({:.3} s per chunk); batched {t_many:.2} s ({:.3} s per chunk, {:.1} times faster); lowest cosine between the two vectors of a chunk {low:.5}; vectors {} and {}\n",
        texts.len(),
        t_one / k,
        t_many / k,
        t_one / t_many.max(1e-9),
        one.len(),
        many.len()
    ))
}

/// `run cart kp` (2026-11-13, faster kernels, step 1: where the card's time goes): the reranker
/// scores twelve 30-line pieces of this tree's sources for one question, with per-kernel timing
/// on (`nmffi::prof::set_kernels`: a sync after every launch, time charged by kernel name); the
/// table of kernels by total time, with counts, and the run's wall time without timing.
fn kernel_profile(home: &Path) -> Result<String, Fault> {
    let rr = nmapi::neural::reranker_model(home)?;
    let mut docs: Vec<String> = Vec::new();
    for f in
        ["crates/nmmem/src/journal.rs", "crates/nmnpcs/src/cartridge.rs", "crates/nmlearn/src/tuner.rs", "crates/nmbrain/src/instruct.rs"]
    {
        let text = std::fs::read_to_string(home.join(f)).map_err(|e| bad(e.to_string()))?;
        let lines: Vec<&str> = text.lines().collect();
        for k in 0..3 {
            docs.push(nmsubs::dense::doc_text(f, &lines.iter().skip(40 + 30 * k).take(30).copied().collect::<Vec<_>>().join("\n")));
        }
    }
    let q = "how does the journal recover a torn last line";
    let ins = nmengn::roles::CODE_INSTRUCTION;
    let _ = nmengn::roles::card_rerank_set(&rr, Device::Gpu, ins, q, &docs[..1])?;
    let t = Instant::now();
    let _ = nmengn::roles::card_rerank_set(&rr, Device::Gpu, ins, q, &docs)?;
    let wall = t.elapsed().as_secs_f64();
    nmffi::prof::reset();
    nmffi::prof::set(true);
    nmffi::prof::set_kernels(true);
    let t = Instant::now();
    let _ = nmengn::roles::card_rerank_set(&rr, Device::Gpu, ins, q, &docs)?;
    let timed = t.elapsed().as_secs_f64();
    nmffi::prof::set_kernels(false);
    nmffi::prof::set(false);
    let mut rows: Vec<(String, f64, u64)> = nmffi::prof::side().into_iter().filter(|r| r.0.starts_with("kernel ")).collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    let total: f64 = rows.iter().map(|r| r.1).sum();
    let mut out = format!(
        "Kernel profile (2026-11-13): reranker over 12 pieces on the card; wall {wall:.2} s untimed, {timed:.2} s with a sync after every launch; kernel time {total:.2} s{}",
        "\n"
    );
    for (name, secs, n) in &rows {
        out.push_str(&format!("{name}: {secs:.3} s over {n} launches ({:.0}%){}", 100.0 * secs / total.max(1e-9), "\n"));
    }
    Ok(out)
}

/// `run cart sn6` (2026-11-13, sensory SN6: serial reads against the card and the processor at
/// once): the reranker scores twelve pieces of this tree's sources on the card alone, then with
/// the split derived from measured per-piece times (the processor takes the number of pieces it
/// finishes in the card's time for the rest); pass when at least 30% faster with the same order
/// and every P(yes) within 0.01 (the processor computes in f32, the card in binary16).
fn sn6(home: &Path) -> Result<String, Fault> {
    let rr = nmapi::neural::reranker_model(home)?;
    let mut docs: Vec<String> = Vec::new();
    for f in
        ["crates/nmmem/src/journal.rs", "crates/nmnpcs/src/cartridge.rs", "crates/nmlearn/src/tuner.rs", "crates/nmbrain/src/instruct.rs"]
    {
        let t = std::fs::read_to_string(home.join(f)).map_err(|e| bad(e.to_string()))?;
        let lines: Vec<&str> = t.lines().collect();
        for k in 0..3 {
            docs.push(nmsubs::dense::doc_text(f, &lines.iter().skip(40 + 30 * k).take(30).copied().collect::<Vec<_>>().join("\n")));
        }
    }
    let q = "how does the journal recover a torn last line";
    let ins = nmengn::roles::CODE_INSTRUCTION;
    let _ = nmengn::roles::card_rerank_set(&rr, Device::Gpu, ins, q, &docs[..1])?;
    let t = Instant::now();
    let gpu = nmengn::roles::card_rerank_set(&rr, Device::Gpu, ins, q, &docs)?;
    let tg_all = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let _ = nmengn::roles::card_rerank_set(&rr, Device::Cpu, ins, q, &docs[..2])?;
    let tc = t.elapsed().as_secs_f64() / 2.0;
    let tg = tg_all / docs.len() as f64;
    // Split so both sides finish together: n_c * tc = (N - n_c) * tg.
    let n = docs.len() as f64;
    let n_c = ((n * tg / (tg + tc)).floor() as usize).min(docs.len() - 1);
    let t = Instant::now();
    let split = nmengn::roles::card_rerank_split(&rr, ins, q, &docs, n_c)?;
    let ts = t.elapsed().as_secs_f64();
    let order = |v: &[f32]| -> Vec<usize> {
        let mut o: Vec<usize> = (0..v.len()).collect();
        o.sort_by(|a, b| v[*b].total_cmp(&v[*a]).then(a.cmp(b)));
        o
    };
    let worst = gpu.iter().zip(&split).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    let same = order(&gpu) == order(&split);
    let gain = 1.0 - ts / tg_all;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "SN6 (2026-11-13): reranker over {} pieces; card {tg:.2} s per piece, processor {tc:.2} s per piece; split {} on the processor, {} on the card.",
        docs.len(),
        n_c,
        docs.len() - n_c
    );
    let _ = writeln!(
        out,
        "card alone {tg_all:.2} s; card and processor at once {ts:.2} s ({:.0}% faster); order equal {same}; largest P(yes) difference {worst:.4}",
        100.0 * gain
    );
    let _ = writeln!(out, "pass (at least 30% faster, same order, within 0.01): {}", gain >= 0.30 && same && worst <= 0.01);
    Ok(out)
}

/// `run cart sn4 [windows]` (2026-11-13, sensory SN4: joining separately read pieces; research
/// file Solution 3, Sensory: CacheBlend and EPIC as the open counterparts; build order "plain join
/// against 32-token recompute against 15% recompute on 5 windows"). Each window: 8 consecutive
/// 30-line chunks of one of this tree's source files, a question naming an item of one chunk. Arms:
/// truth (the whole window encoded as one sequence), plain join (chunk 0 alone, every later chunk
/// encoded after chunk 0 as its anchor, placed side by side by reference, as kept chunks are),
/// recompute of the first 32 tokens and of the first 15% of each later chunk (re-encoded with the
/// real preceding chunks of the window as context, keys and values spliced into a copy of the
/// chunk). Measured per arm against the truth: the drone heads' attention mass per chunk (top
/// chunk agreement, rank agreement of the eight) and the relative error of keys and values at the
/// drones' last layer. On the card; the encodings stay in RAM.
fn sn4(home: &Path, args: &[String]) -> Result<String, Fault> {
    use nmengn::kvstore::{ChunkKv, Probe, Window, encode_through, score_targets};
    use nmengn::model::RunOpts;
    use std::sync::Arc;
    let nwin: usize = args.get(3).and_then(|x| x.parse().ok()).unwrap_or(5);
    let t0 = Instant::now();
    let base = nmapi::neural::model(home)?;
    let set = nmapi::tentacle_pass::drones(home);
    let last = set.last_layer();
    let d = base.pop.dims;
    let width = d.kv_heads * d.head_dim;
    let files = [
        "crates/nmmem/src/journal.rs",
        "crates/nmapi/src/forage.rs",
        "crates/nmbrain/src/g1.rs",
        "crates/nmbrain/src/instruct.rs",
        "crates/nmmem/src/book.rs",
        "crates/nmapi/src/ask.rs",
        "crates/nmpod/src/record.rs",
    ];
    let tok = |s: &str| base.pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
    let targets = score_targets(&base, last)?;
    // Per-chunk drone mass of a question read after window `w`.
    let read = |w: &Arc<Window>, q: &[i32], parts: &[(usize, usize)]| -> Result<Vec<f32>, Fault> {
        let p = Arc::new(Probe::of_heads(&set.heads));
        let opts = RunOpts { window: Some(Arc::clone(w)), probe: Some(Arc::clone(&p)), ..RunOpts::default() };
        base.forward_at(Device::Gpu, q, w.len, &targets, &opts)?;
        let mut pos = vec![0.0_f32; w.len];
        for &l in &p.layers {
            let Some(m) = p.get(l) else { continue };
            let span = m.len() / d.heads.max(1);
            for h in 0..d.heads {
                for (i, x) in pos.iter_mut().enumerate() {
                    *x += m.get(h * span + i).copied().unwrap_or(0.0);
                }
            }
        }
        Ok(parts.iter().map(|(a, n)| pos.iter().skip(*a).take(*n).sum()).collect())
    };
    let ranks = |v: &[f32]| -> Vec<usize> {
        let mut o: Vec<usize> = (0..v.len()).collect();
        o.sort_by(|a, b| v[*b].total_cmp(&v[*a]).then(a.cmp(b)));
        let mut r = vec![0; v.len()];
        for (k, i) in o.iter().enumerate() {
            r[*i] = k;
        }
        r
    };
    let spearman = |a: &[f32], b: &[f32]| -> f64 {
        let (ra, rb) = (ranks(a), ranks(b));
        let n = a.len() as f64;
        let s: f64 = ra.iter().zip(&rb).map(|(x, y)| ((*x as f64) - (*y as f64)).powi(2)).sum();
        1.0 - 6.0 * s / (n * (n * n - 1.0)).max(1.0)
    };
    let rel = |parts: &[Arc<ChunkKv>], truth: &ChunkKv| -> f64 {
        let (mut num, mut den) = (0.0_f64, 0.0_f64);
        for (k, kv) in [(true, false), (false, true)] {
            let _ = kv;
            let joined: Vec<u16> = parts.iter().flat_map(|c| if k { c.k_pre[last].clone() } else { c.v[last].clone() }).collect();
            let t = if k { &truth.k_pre[last] } else { &truth.v[last] };
            for (a, b) in joined.iter().zip(t) {
                let (a, b) = (f64::from(nmengn::half::from16(*a)), f64::from(nmengn::half::from16(*b)));
                num += (a - b) * (a - b);
                den += b * b;
            }
        }
        (num / den.max(1e-12)).sqrt()
    };
    // arm totals: top agree, spearman sum, kv error sum.
    let mut tot = [(0_usize, 0.0_f64, 0.0_f64); 4];
    let mut lines = Vec::new();
    for wi in 0..nwin {
        let f = files[wi % files.len()];
        let text = std::fs::read_to_string(home.join(f)).map_err(|e| bad(e.to_string()))?;
        let all: Vec<&str> = text.lines().collect();
        let start = 30 + 240 * (wi / files.len());
        let chunks: Vec<String> =
            (0..8).map(|k| all.iter().skip(start + 30 * k).take(30).copied().collect::<Vec<_>>().join("\n")).collect();
        if chunks.iter().any(String::is_empty) {
            continue;
        }
        let ids: Vec<Vec<i32>> = chunks.iter().map(|c| tok(c)).collect::<Result<_, _>>()?;
        let name = chunks.iter().skip(3).find_map(|c| c.lines().find_map(nmmem::migrate::defined_name)).unwrap_or_else(|| "it".into());
        let q = tok(&format!("\nQuestion: where is {name} defined and what does it do?"))?;
        // Truth.
        let cat: Vec<i32> = ids.iter().flatten().copied().collect();
        let truth = Arc::new(encode_through(&base, Device::Gpu, None, &cat, last)?);
        let tw = Arc::new(Window::build(&base, &[Arc::clone(&truth)])?);
        let mut parts_pos = Vec::new();
        let mut at = 0;
        for i in &ids {
            parts_pos.push((at, i.len()));
            at += i.len();
        }
        let truth_mass = read(&tw, &q, &parts_pos)?;
        // Plain join.
        let c0 = Arc::new(encode_through(&base, Device::Gpu, None, &ids[0], last)?);
        let anchor = Arc::new(Window::build(&base, &[Arc::clone(&c0)])?);
        let mut plain: Vec<Arc<ChunkKv>> = vec![Arc::clone(&c0)];
        for i in ids.iter().skip(1) {
            plain.push(Arc::new(encode_through(&base, Device::Gpu, Some(&anchor), i, last)?));
        }
        let mut arms: Vec<(String, Vec<Arc<ChunkKv>>)> = vec![("plain join".into(), plain.clone())];
        for (label, r_of) in [("recompute 32 tokens", 0_usize), ("recompute 15%", 1)] {
            let mut rec: Vec<Arc<ChunkKv>> = vec![Arc::clone(&c0)];
            for (k, i) in ids.iter().enumerate().skip(1) {
                let r = if r_of == 0 { 32.min(i.len()) } else { (i.len() * 15).div_ceil(100).min(i.len()) };
                let prefix = Arc::new(Window::build(&base, &rec)?);
                let fresh = encode_through(&base, Device::Gpu, Some(&prefix), &i[..r], last)?;
                let mut c = (*plain[k]).clone();
                for l in 0..=last {
                    c.k_pre[l][..r * width].copy_from_slice(&fresh.k_pre[l]);
                    c.v[l][..r * width].copy_from_slice(&fresh.v[l]);
                }
                rec.push(Arc::new(c));
            }
            arms.push((label.into(), rec));
        }
        // The engine function, 15%: must equal the inline arm above.
        let given: Vec<(Arc<ChunkKv>, Vec<i32>)> = plain.iter().cloned().zip(ids.iter().cloned()).collect();
        arms.push(("engine recompute 15%".into(), nmengn::kvstore::recompute(&base, Device::Gpu, &given, 0.15, last)?));
        let mut row = format!("window {} ({f}, {} tokens, question on {name}):", wi + 1, cat.len());
        for (a, (label, parts)) in arms.iter().enumerate() {
            let w = Arc::new(Window::build(&base, parts)?);
            let m = read(&w, &q, &parts_pos)?;
            let top = ranks(&m).iter().position(|r| *r == 0) == ranks(&truth_mass).iter().position(|r| *r == 0);
            let sp = spearman(&m, &truth_mass);
            let e = rel(parts, &truth);
            tot[a].0 += usize::from(top);
            tot[a].1 += sp;
            tot[a].2 += e;
            row.push_str(&format!(" {label}: top {top}, rank agreement {sp:.2}, key-value error {e:.3};"));
        }
        lines.push(row);
    }
    let n = lines.len().max(1) as f64;
    let mut out = format!(
        "SN4 (2026-11-13): joining separately read pieces, {} windows of 8 chunks, drone heads' mass per chunk against the whole window encoded at once, on the card.\n",
        lines.len()
    );
    for (a, label) in ["plain join", "recompute 32 tokens", "recompute 15%", "engine recompute 15%"].iter().enumerate() {
        out.push_str(&format!(
            "{label}: top chunk equal to the truth on {} of {}; mean rank agreement {:.2}; mean key-value error at layer {last} {:.3}\n",
            tot[a].0,
            lines.len(),
            tot[a].1 / n,
            tot[a].2 / n
        ));
    }
    for l in &lines {
        out.push_str(l);
        out.push('\n');
    }
    out.push_str(&format!("total {:.1} s\n", t0.elapsed().as_secs_f64()));
    Ok(out)
}

/// Step L: prefetch. The judge's 8-bit base cartridge bound cold (quantized inside the bind),
/// against prepared ahead on 8 processor threads and bound while the prepared packets are held;
/// the same pair's P(yes) must be identical. Processor only.
fn prefetch(home: &Path) -> Result<String, Fault> {
    let pop = Population::open(home, POP, true, false)?;
    let mut asm = Assembly::load(&home.join("designs/assemblies"), ASM)?;
    asm.nodes.iter_mut().filter(|n| n.op == "linear").for_each(|n| "q8".clone_into(&mut n.precision));
    let probe = Model::bind(&pop, Assembly::load(&home.join("designs/assemblies"), ASM)?)?;
    let (cal, _) = pairs(home, &probe)?;
    drop(probe);
    let pair = cal.first().ok_or_else(|| bad("no pair"))?;
    let t = Instant::now();
    let cold = Model::bind(&pop, asm.clone())?;
    let cold_s = t.elapsed().as_secs_f64();
    let p_cold = p_yes(&cold, Device::Cpu, pair, &RunOpts::default())?;
    drop(cold);
    let t = Instant::now();
    let held = pop.prefetch_q8(&asm, 8)?;
    let pre_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let warm = Model::bind(&pop, asm)?;
    let warm_s = t.elapsed().as_secs_f64();
    let p_warm = p_yes(&warm, Device::Cpu, pair, &RunOpts::default())?;
    let same = p_cold.to_bits() == p_warm.to_bits();
    Ok(format!(
        "Cartridge loading step L, prefetch (processor, 2026-11-12): judge base cartridge at 8 bits ({} packets prepared). Bind cold (quantizing inside, one thread) {cold_s:.2} s; prefetch on 8 threads {pre_s:.2} s; bind while prefetched packets are held {warm_s:.2} s. P(yes) on one pair cold {p_cold:.6}, prefetched {p_warm:.6}: identical {same} -> {}
",
        held.len(),
        if same && warm_s < cold_s && pre_s < cold_s { "PASS" } else { "FAIL" }
    ))
}

/// L5, a steering vector in closed form (2026-11-12; research file Solution 6 L5: "steering in
/// closed form (mean difference of states for right against wrong pieces)"). On the judge: the
/// last position's output of layer `l` over the calibration pairs; the vector is the mean of the
/// right documents' states minus the wrong ones'; it is added (times a scale) to that layer's
/// output on the held-out pairs. Measures: right-over-wrong order and the mean margin of z (right
/// minus wrong), against no steering. Kept only if the margin rises and no order is lost. Held in
/// RAM (the disk rule for L5 awaits the word of Luis Jake Gabriel III).
fn steering(home: &Path, args: &[String]) -> Result<String, Fault> {
    let t0 = Instant::now();
    let (m, dev) = open(home)?;
    let l: usize = args.get(3).and_then(|x| x.parse().ok()).unwrap_or(18);
    let (cal, held) = pairs(home, &m)?;
    let cal: Vec<Pair> = cal.into_iter().take(12).collect();
    let held: Vec<Pair> = held.into_iter().take(12).collect();
    let target = (nid(&format!("L{l}.res2")), "y".to_owned());
    let h = m.pop.dims.hidden;
    let (mut right, mut wrong, mut nr, mut nw) = (vec![0.0_f64; h], vec![0.0_f64; h], 0.0_f64, 0.0_f64);
    for p in &cal {
        let o = m.forward(dev, &p.ids, std::slice::from_ref(&target), &RunOpts::default())?;
        let y = values(o.get(&target).ok_or_else(|| bad("no layer output"))?)?;
        let last = y.get(y.len().saturating_sub(h)..).unwrap_or(&[]);
        let (acc, n) = if p.right { (&mut right, &mut nr) } else { (&mut wrong, &mut nw) };
        acc.iter_mut().zip(last).for_each(|(a, x)| *a += f64::from(*x));
        *n += 1.0;
    }
    #[expect(clippy::cast_possible_truncation, reason = "f32 vector")]
    let v: Vec<f32> = right.iter().zip(&wrong).map(|(r, w)| (r / nr.max(1.0) - w / nw.max(1.0)) as f32).collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let v = std::sync::Arc::new(v);
    let score = |opts: &RunOpts| -> Result<(f64, usize, usize), Fault> {
        let z: Vec<f32> = held.iter().map(|p| judge_z(&m, dev, p, opts)).collect::<Result<_, _>>()?;
        let (_, _, ok, n) = closeness(&held, &z, &z);
        let (mut sr, mut sw, mut kr, mut kw) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        for (p, x) in held.iter().zip(&z) {
            if p.right {
                sr += f64::from(*x);
                kr += 1.0
            } else {
                sw += f64::from(*x);
                kw += 1.0
            }
        }
        Ok((sr / kr.max(1.0) - sw / kw.max(1.0), ok, n))
    };
    let (m0, ok0, n) = score(&RunOpts::default())?;
    let mut out = format!("L5 steering vector, closed form, on the judge ({}): layer {l}, fitted on {} calibration pairs (vector norm {norm:.2}); held out {} pairs. No steering: mean margin of z (right minus wrong) {m0:.3}, order held {ok0} of {n}.
", dev.name(), cal.len(), held.len());
    let mut kept = None;
    for s in [0.5_f32, 1.0] {
        let mut opts = RunOpts::default();
        opts.steer.insert(l, (std::sync::Arc::clone(&v), s));
        let (ms, oks, _) = score(&opts)?;
        let keep = ms > m0 && oks >= ok0;
        let _ = writeln!(
            out,
            "  scale {s}: margin {ms:.3} ({:+.3}), order held {oks} of {n} -> {}",
            ms - m0,
            if keep { "would keep" } else { "not kept" }
        );
        if keep && kept.is_none() {
            kept = Some(s);
        }
    }
    let _ = writeln!(
        out,
        "Verdict: {}. Total {:.1} s.",
        kept.map_or_else(|| "not kept (no gain on the held-out pairs)".to_owned(), |s| format!("kept at scale {s} (RAM only)")),
        t0.elapsed().as_secs_f64()
    );
    Ok(out)
}

/// One candidate of a Core Brain question, as cached for step H: whether it is right, and the last
/// position's SwiGLU output and residual of layer 27 (the last layer).
struct Cand {
    right: bool,
    s: Vec<f32>,
    r: Vec<f32>,
}

/// Step H, the first fitted adapter on the judge (research file, "Solutions 5 and 6 joined" and
/// Solution 6 L5; the EMMA method: frozen weights, a small scoped unit, a check before keeping,
/// exact rollback). Site: the last layer's down projection, as a rank-`R` low-rank path
/// `delta = U (D s)` added to the layer's output (D fixed and drawn from a seed, U fitted from
/// zero, so the unit starts as the identity). The last layer works position by position, so its
/// effect on the judge's yes-or-no score depends only on the last position: those values are
/// cached once per candidate (RAM only) and the fit runs on the processor. Task: the Core Brain's
/// real one, ranking the pieces sent up for a question (the whole-path records' candidates and
/// gold, documents cut to 256 tokens). Fit: forward-only (MeZO, `nmlearn::forward::perturb_rows`)
/// on the listwise loss over the even-numbered development questions; scored on the odd ones;
/// a shuffled-gold control is fitted the same way. Kept only if it raises right-first on the
/// held-out questions and the control does not.
fn fitted_adapter(home: &Path, args: &[String]) -> Result<String, Fault> {
    const RANK: usize = 4;
    let t0 = Instant::now();
    let num = |i: usize, d: usize| args.get(i).and_then(|x| x.parse::<usize>().ok()).unwrap_or(d);
    let (nq, steps) = (num(3, 30), num(4, 3000));
    let lr_arg: f64 = args.get(5).and_then(|x| x.parse().ok()).unwrap_or(2e-4);
    let (m, dev) = open(home)?;
    let d = m.pop.dims;
    let dir = {
        let mut v: Vec<std::path::PathBuf> = std::fs::read_dir(home.join("scratch/tc1-dump"))
            .map(|rd| rd.filter_map(Result::ok).map(|e| e.path()).filter(|p| p.join("substrate.json").is_file()).collect())
            .unwrap_or_default();
        v.sort();
        v.pop().ok_or_else(|| bad("no scan dump under scratch/tc1-dump"))?
    };
    let sub = nmapi::scan_dump::load_substrate(&dir)?;
    let items = crate::e2::suite(home)?;
    // Candidates and gold from the development whole-path records.
    let mut qs: std::collections::BTreeMap<usize, (Vec<usize>, Vec<usize>)> = std::collections::BTreeMap::new();
    for f in [
        "DX-2026-11-10-whole-1794487111506.txt",
        "DX-2026-11-10-whole-1794487327555.txt",
        "DX-2026-11-10-whole-1794487529060.txt",
        "DX-2026-11-10-whole-1794487678351.txt",
        "DX-2026-11-10-whole-1794487838873.txt",
    ] {
        let text = std::fs::read_to_string(home.join("runs").join(f)).map_err(|e| bad(format!("{f}: {e}")))?;
        for line in text.lines() {
            let Some((n, rest)) = line.strip_prefix("judges q").and_then(|r| r.split_once(": gold ")) else { continue };
            let Some((gold, pieces)) = rest.split_once("; pieces ") else { continue };
            let Ok(n) = n.parse::<usize>() else { continue };
            let chunks = pieces.split_whitespace().filter_map(|p| p.split(':').next()?.parse().ok()).collect();
            qs.entry(n).or_insert((gold.split_whitespace().filter_map(|x| x.parse().ok()).collect(), chunks));
        }
    }
    let tok = |s: &str| m.pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
    let targets = [(nid("L27.swigate"), "y".to_owned()), (nid("L27.res2"), "y".to_owned())];
    let mut data: Vec<(usize, Vec<Cand>)> = Vec::new();
    let mut check_z: Option<(Pair, f32)> = None;
    for (&n, (gold, chunks)) in qs.iter().take(nq) {
        let Some(it) = items.get(n - 1) else { continue };
        let mut cands = Vec::new();
        for &c in chunks {
            let Some(ch) = sub.chunks.get(c) else { continue };
            let path = sub.files.get(ch.file).map(|f| f.path.clone()).unwrap_or_default();
            let mut dt = tok(&nmsubs::dense::doc_text(&path, &ch.text))?;
            dt.truncate(DOC_TOKENS);
            let ids = tok(&nmengn::roles::card_prompt(nmengn::roles::CODE_INSTRUCTION, &it.q, &m.pop.tokenizer.decode(&dt)))?;
            let o = m.forward(dev, &ids, &targets, &RunOpts::default())?;
            let last = |k: usize, w: usize| -> Result<Vec<f32>, Fault> {
                let t = targets.get(k).ok_or_else(|| bad("target"))?;
                let y = values(o.get(t).ok_or_else(|| bad("no target"))?)?;
                Ok(y.get(y.len().saturating_sub(w)..).unwrap_or(&[]).to_vec())
            };
            let cand = Cand { right: gold.contains(&c), s: last(0, d.inter)?, r: last(1, d.hidden)? };
            if check_z.is_none() {
                let p = Pair { q: n, right: cand.right, ids: ids.clone() };
                let z = judge_z(&m, dev, &p, &RunOpts::default())?;
                check_z = Some((p, z));
            }
            cands.push(cand);
        }
        if cands.iter().any(|c| c.right) {
            data.push((n, cands));
        }
    }
    let collect_s = t0.elapsed().as_secs_f64();
    // The final norm and the two label rows of the head (tied to the token table).
    let bf = |bytes: &[u8]| -> Vec<f32> {
        bytes.as_chunks::<2>().0.iter().map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)).collect()
    };
    let norm_c = m.pop.cluster(&format!("{}/model.norm.weight/whole", m.pop.card.id))?;
    let g = bf(&m.pop.prov.acquire(&norm_c)?.tensor_bytes());
    let table_c = m.pop.cluster(&format!("{}/model.embed_tokens.weight/whole", m.pop.card.id))?;
    let table = m.pop.prov.acquire(&table_c)?;
    let row = |id: usize| bf(table.tensor_bytes().get(id * d.hidden * 2..(id + 1) * d.hidden * 2).unwrap_or(&[]));
    let one = |s: &str| -> Result<usize, Fault> { tok(s)?.first().and_then(|x| usize::try_from(*x).ok()).ok_or_else(|| bad("label")) };
    let wd: Vec<f32> = row(one("yes")?).iter().zip(row(one("no")?)).map(|(a, b)| a - b).collect();
    let eps = d.eps;
    let z_of = |r: &[f32]| -> f32 {
        #[expect(clippy::cast_precision_loss, reason = "width")]
        let ms = r.iter().map(|x| x * x).sum::<f32>() / r.len().max(1) as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        r.iter().zip(&g).zip(&wd).map(|((x, gi), w)| x * inv * gi * w).sum()
    };
    // The fixed down factor D [RANK x inter], seeded normals scaled by 1/sqrt(inter).
    let mut dfac = vec![0.0_f32; RANK * d.inter];
    nmlearn::forward::perturb_rows(&mut dfac, d.inter, &(0..RANK).collect::<Vec<_>>(), 77, 0, 1.0 / (d.inter as f64).sqrt())?;
    let proj: Vec<Vec<Vec<f32>>> = data
        .iter()
        .map(|(_, cs)| {
            cs.iter()
                .map(|c| {
                    (0..RANK)
                        .map(|k| dfac.get(k * d.inter..(k + 1) * d.inter).unwrap_or(&[]).iter().zip(&c.s).map(|(a, b)| a * b).sum())
                        .collect()
                })
                .collect()
        })
        .collect();
    let score = |up: &[f32], qi: usize, ci: usize| -> f32 {
        let (Some((_, cs)), Some(p)) = (data.get(qi), proj.get(qi).and_then(|x| x.get(ci))) else { return 0.0 };
        let Some(c) = cs.get(ci) else { return 0.0 };
        let r: Vec<f32> =
            c.r.iter()
                .enumerate()
                .map(|(i, x)| {
                    x + (0..RANK).map(|k| up.get(i * RANK + k).copied().unwrap_or(0.0) * p.get(k).copied().unwrap_or(0.0)).sum::<f32>()
                })
                .collect();
        z_of(&r)
    };
    let lse = |v: &[f32]| -> f64 {
        let mx = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        f64::from(mx) + v.iter().map(|x| f64::from((x - mx).exp())).sum::<f64>().ln()
    };
    let loss_on = |up: &[f32], idx: &[usize], gold: &dyn Fn(usize, usize) -> bool| -> f64 {
        let mut tot = 0.0;
        for &qi in idx {
            let n = data.get(qi).map_or(0, |x| x.1.len());
            let zs: Vec<f32> = (0..n).map(|ci| score(up, qi, ci)).collect();
            let gz: Vec<f32> = (0..n).filter(|ci| gold(qi, *ci)).filter_map(|ci| zs.get(ci).copied()).collect();
            if !gz.is_empty() {
                tot += lse(&zs) - lse(&gz);
            }
        }
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let k = idx.len().max(1) as f64;
        tot / k
    };
    let first_right = |up: &[f32], idx: &[usize]| -> usize {
        idx.iter()
            .filter(|&&qi| {
                let n = data.get(qi).map_or(0, |x| x.1.len());
                let best = (0..n).max_by(|a, b| score(up, qi, *a).total_cmp(&score(up, qi, *b)).then(b.cmp(a)));
                best.and_then(|ci| data.get(qi).and_then(|x| x.1.get(ci))).is_some_and(|c| c.right)
            })
            .count()
    };
    let cal: Vec<usize> = (0..data.len()).filter(|i| data.get(*i).is_some_and(|x| x.0 % 2 == 0)).collect();
    let held: Vec<usize> = (0..data.len()).filter(|i| data.get(*i).is_some_and(|x| x.0 % 2 == 1)).collect();
    let rows: Vec<usize> = (0..d.hidden).collect();
    let fit = |gold: &dyn Fn(usize, usize) -> bool| -> Result<(Vec<f32>, f64, f64), Fault> {
        let mut up = vec![0.0_f32; d.hidden * RANK];
        let before = loss_on(&up, &cal, gold);
        let (e, lr) = (1e-3_f64, lr_arg);
        for step in 0..steps {
            let st = u64::try_from(step).unwrap_or(0);
            nmlearn::forward::perturb_rows(&mut up, RANK, &rows, 5, st, e)?;
            let plus = loss_on(&up, &cal, gold);
            nmlearn::forward::perturb_rows(&mut up, RANK, &rows, 5, st, -2.0 * e)?;
            let minus = loss_on(&up, &cal, gold);
            nmlearn::forward::perturb_rows(&mut up, RANK, &rows, 5, st, e)?;
            let projected = (plus - minus) / (2.0 * e);
            nmlearn::forward::perturb_rows(&mut up, RANK, &rows, 5, st, -lr * projected)?;
        }
        let after = loss_on(&up, &cal, gold);
        Ok((up, before, after))
    };
    let real = |qi: usize, ci: usize| data.get(qi).and_then(|x| x.1.get(ci)).is_some_and(|c| c.right);
    // Control: the gold moved to the next piece of the same question.
    let shuffled = |qi: usize, ci: usize| {
        let n = data.get(qi).map_or(1, |x| x.1.len()).max(1);
        real(qi, (ci + n - 1) % n)
    };
    let zero = vec![0.0_f32; d.hidden * RANK];
    let (base_cal, base_held) = (first_right(&zero, &cal), first_right(&zero, &held));
    let t1 = Instant::now();
    let (up, lb, la) = fit(&real)?;
    let (upc, _, _) = fit(&shuffled)?;
    let fit_s = t1.elapsed().as_secs_f64();
    let (a_cal, a_held, c_held) = (first_right(&up, &cal), first_right(&up, &held), first_right(&upc, &held));
    let keep = a_held > base_held && c_held <= base_held;
    let zc = check_z.as_ref().map_or(String::new(), |(p, ze)| {
        let qi = data.iter().position(|x| x.0 == p.q).unwrap_or(0);
        format!("z rebuilt from the cache {:.3} against the engine's {ze:.3}", score(&zero, qi, 0))
    });
    Ok(format!(
        "Step H, first fitted adapter on the judge ({}): rank-{RANK} path on the last layer's down projection ({} numbers fitted, D fixed by seed). {} questions with the right piece among the candidates ({} calibration, {} held out), {} candidates cached in {collect_s:.1} s; {zc}.\nForward-only fit (MeZO, {steps} steps, learning rate {lr_arg}, {fit_s:.1} s for both fits): calibration loss {lb:.3} -> {la:.3}.\nRight piece first: calibration {base_cal} -> {a_cal} of {}; held out {base_held} -> {a_held} of {}; shuffled-gold control on held out {c_held}.\nVerdict: {}. The released weights are unchanged; removing the unit gives the base back exactly (step G). Total {:.1} s.\n",
        dev.name(),
        d.hidden * RANK,
        data.len(),
        cal.len(),
        held.len(),
        data.iter().map(|x| x.1.len()).sum::<usize>(),
        cal.len(),
        held.len(),
        if keep { "KEEP (RAM only until the disk rule is settled)" } else { "not kept" },
        t0.elapsed().as_secs_f64()
    ))
}

/// Step G: the identity adapter on the judge base cartridge (the step F mask: 5% of the lowest-
/// scored quiet blocks off, as row and column sub-clusters). An adapter is the engine's unmerged
/// low-rank unit (`nmengn::adapter`, `y += scale * up(down(x))`). Sites: nodes with a lesion
/// effect (L11 and L15 attention output, step E1) and the masked L13 down projection. T1 zero
/// `up` on all sites: P(yes) bit-identical to the cartridge alone. T2 a small nonzero `up`:
/// output changes. T3 unloaded: bit-identical to the cartridge alone again.
fn identity_adapter(home: &Path) -> Result<String, Fault> {
    let t0 = Instant::now();
    let (m, dev) = open(home)?;
    let d = m.pop.dims;
    let (cal, _) = pairs(home, &m)?;
    let set: Vec<Pair> = cal.into_iter().take(6).collect();
    let eff = ffn_effects(home, &m)?;
    let quiet: Vec<usize> = eff.iter().enumerate().filter(|(_, e)| **e < 0.01).map(|(l, _)| l).collect();
    let mask = block_set(&block_scores(&m, dev, &set)?, &quiet, 0.05, "low");
    let blocks = d.inter / BLOCK;
    let mut asm = Assembly::load(&home.join("designs/assemblies"), ASM)?;
    for (l, off) in &mask.blocks_off {
        asm.keep_ffn_blocks(*l, &(0..blocks).filter(|b| !off.contains(b)).collect::<Vec<_>>(), BLOCK)?;
    }
    let cart = Model::bind(&m.pop, asm)?;
    let kept13 = blocks - mask.blocks_off.get(&13).map_or(0, Vec::len);
    let sites = [
        (nid("L11.o"), d.heads * d.head_dim, d.hidden),
        (nid("L15.o"), d.heads * d.head_dim, d.hidden),
        (nid("L13.down"), kept13 * BLOCK, d.hidden),
    ];
    let run_all = |model: &Model| -> Result<Vec<f32>, Fault> { set.iter().map(|p| p_yes(model, dev, p, &RunOpts::default())).collect() };
    let same = |a: &[f32], b: &[f32]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    let rank = 4;
    let unit = |site: &(String, usize, usize), up: f32, i: usize| -> Result<std::sync::Arc<nmengn::adapter::Adapter>, Fault> {
        let r16 = |x: f32| nmengn::half::r16(x);
        let down: Vec<f32> = shuffle(rank * site.1, 7 + i as u64).iter().map(|v| r16(((*v % 17) as f32 - 8.0) / 256.0)).collect();
        let upv: Vec<f32> = shuffle(site.2 * rank, 11 + i as u64).iter().map(|v| r16(up * ((*v % 5) as f32 - 2.0))).collect();
        Ok(std::sync::Arc::new(nmengn::adapter::Adapter::new(&format!("judge-g-{i}"), &site.0, rank, site.1, site.2, down, upv, 1.0)?))
    };
    let base = run_all(&cart)?;
    for (i, s) in sites.iter().enumerate() {
        cart.set_unit(&s.0, Some(unit(s, 0.0, i)?))?;
    }
    let t1 = run_all(&cart)?;
    for (i, s) in sites.iter().enumerate() {
        cart.set_unit(&s.0, Some(unit(s, 0.01, i)?))?;
    }
    let t2 = run_all(&cart)?;
    for s in &sites {
        cart.set_unit(&s.0, None)?;
    }
    let t3 = run_all(&cart)?;
    let (c2, _, x2) = change(&t2, &base);
    let (p1, p2, p3) = (same(&t1, &base), !same(&t2, &base), same(&t3, &base));
    let mut out = String::new();
    let _ = writeln!(
        out,
        "Cartridge step G, identity adapter on the judge base cartridge ({}, {} pairs; rank {rank} units on {}):",
        dev.name(),
        set.len(),
        sites.iter().map(|s| s.0.rsplit('#').next().unwrap_or("").to_owned()).collect::<Vec<_>>().join(", ")
    );
    let _ =
        writeln!(out, "T1 zero up, all sites loaded: bit-identical to the cartridge alone: {p1} -> {}", if p1 { "PASS" } else { "FAIL" });
    let _ = writeln!(
        out,
        "T2 nonzero up (0.01 scale): output changes: {p2}, mean |change of P(yes)| {c2:.4}, largest {x2:.4} -> {}",
        if p2 { "PASS" } else { "FAIL" }
    );
    let _ = writeln!(out, "T3 unloaded: bit-identical to the cartridge alone: {p3} -> {}", if p3 { "PASS" } else { "FAIL" });
    let _ =
        writeln!(out, "G identity adapter: {}. Total {:.1} s.", if p1 && p2 && p3 { "PASS" } else { "FAIL" }, t0.elapsed().as_secs_f64());
    Ok(out)
}

/// `run cart <d | e1 <from> <to> [questions] | e2 <from> <to> [questions] | e3 [questions] | f [cal q] [held q] | g>`.
pub fn run(args: &[String], home: &Path) -> ExitCode {
    let step = args.get(2).map_or("d", String::as_str);
    let stamp = nmtype::now_ms();
    let result = match step {
        "d" => record(home, stamp),
        "e1" | "e2" => lesions(home, stamp, step, args),
        "e3" => set_lesion(home, stamp, args),
        "f" => base_cartridge(home, stamp, args),
        "g" => identity_adapter(home),
        "i" => loading(home, stamp),
        "p" => parts(home, stamp),
        "e8" => embed_q8(home, args),
        "q4err" => q4_error(home),
        "pc" => precision_closeness(home, args),
        "l2" => stream_stage(home),
        "sn6" => sn6(home),
        "eb" => embed_batch(home, args),
        "kp" => kernel_profile(home),
        "sn4" => sn4(home, args),
        "l" => prefetch(home),
        "s" => steering(home, args),
        "h" => fitted_adapter(home, args),
        "f1" => f1(home),
        "f2" => f2(home, args),
        "f3" => f3(home, args),
        "t" => timing(home),
        other => Err(bad(format!("unknown cartridge step {other}"))),
    };
    match result {
        Ok(text) => {
            let p = home.join(format!("runs/CART-{step}-{stamp}.txt"));
            let _ = std::fs::write(&p, &text);
            print!("{text}");
            println!("report: {}", p.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("cart {step}: {e}");
            ExitCode::FAILURE
        }
    }
}
