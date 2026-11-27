//! `nano-lab run emap <step> ...`: channel maps for the judge's and the embedder's base cartridges
//! on their jobs (research note 2026-11-18, item 5), each checked against 20 random masks with the
//! same count of channels left out in every layer, on questions the map never saw.
//!
//! The judged set is a capture of the served path (`nano-lab run bcart capture`) on a tuning
//! question file, named on the command line (a held-out file is refused). Every arm zeroes its
//! left-out feed-forward channels at each layer's `swigate` socket (the terms the narrowed tensors
//! drop; a measurement, no cartridge written). A compensation arm (`+bias`) puts each left-out
//! channel's mean output on the job's reads in its place (FLAP's mean term); that compensation is
//! written as its own file beside the map, never into a cartridge or the released weights.
//!
//! The judge: its maps (`map.json`, energy; `role.json`, direct share of z) were made on the
//! development questions, so every question of the tuning set is held out for it. Measure: z (yes
//! minus no) per candidate, packed as the served judge reads them.
//! The embedder: its map is made here, from activations counted on the job's own reads (COMPACT's
//! rule: the tokens the job sees), on the even places of the set, and measured on the odd places.
//! Measure: cosine of the posed question with each candidate.
//! Per question: the pair margin (mean over right and wrong pairs of the score difference), best
//! right minus best wrong, and whether a right candidate is first. A map passes when its pair
//! margin is above the 95th percentile of the random masks' (above at least 19 of 20).
//!
//! Steps:
//! - `judge-mean`: the judge's channel means over the map part's reads (`set-p2-map.json`).
//! - `judge <set> <arm>[,<arm>...]`: arms `open`, `map:20`, `role:20`, `role+bias:20`, `rand:20:<seed>`.
//! - `embed-record <set>`: the embedder's channel statistics on the fit half's reads.
//! - `embed <set> <arm>[,<arm>...]`: arms `open`, `act:20`, `act+bias:20`, `rand:20:<seed>`.
//! - `report <set>`.
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use nmengn::Assembly;
use nmengn::assembly::nid;
use nmengn::model::{Model, Population, RunOpts, values};
use nmengn::placement::Device;
use nmjson::Value;
use nmtype::{Fault, FaultKind};

const JUDGE_POP: &str = "qwen3-reranker-0.6b";
const JUDGE_ASM: &str = "qwen3-reranker";
const EMBED_POP: &str = "qwen3-embedding-0.6b";
const EMBED_ASM: &str = "qwen3-embedding";
const JUDGE_MAPS: &str = "library/parameter-maps/reranker";
const EMBED_MAPS: &str = "library/parameter-maps/embedder";
const OUT: &str = "scratch/lane-d/emap";
const INTER: usize = 3072;
const LAYERS: usize = 28;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("emap", FaultKind::Execution, m)
}
fn text(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default()
}
fn int(v: &Value, k: &str) -> usize {
    v.get(k).and_then(Value::integer).ok().and_then(|x| usize::try_from(x).ok()).unwrap_or(0)
}
fn read_json(p: &Path) -> Result<Value, Fault> {
    nmjson::parse(&std::fs::read(p).map_err(|e| bad(format!("{}: {e}", p.display())))?).map_err(|e| bad(format!("{}: {e}", p.display())))
}
fn nums(v: &Value) -> Vec<f64> {
    v.array().unwrap_or_default().iter().map(|x| if let Value::Number(t) = x { t.parse().unwrap_or(0.0) } else { 0.0 }).collect()
}
fn num_arr(x: &[f64]) -> Value {
    Value::Array(x.iter().map(|y| Value::Number(format!("{y:e}"))).collect())
}
fn write_json(p: &Path, v: &Value) -> Result<(), Fault> {
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).map_err(|e| bad(e.to_string()))?;
    }
    std::fs::write(p, v.encode().map_err(|e| bad(e.to_string()))?).map_err(|e| bad(format!("{}: {e}", p.display())))
}
fn bf16(b: &[u8], i: usize) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([b.get(2 * i).copied().unwrap_or(0), b.get(2 * i + 1).copied().unwrap_or(0)])) << 16)
}
/// Column norms of a released `[rows, cols]` BF16 tensor.
fn col_norms(pop: &Population, name: &str) -> Result<Vec<f64>, Fault> {
    let c = pop.clusters.iter().find(|c| c.tensor == name).ok_or_else(|| bad(format!("no tensor {name}")))?;
    let v = pop.prov.acquire(c)?;
    let (r, cols) = (c.shape.first().copied().unwrap_or(0), c.shape.get(1).copied().unwrap_or(0));
    let b = v.tensor_bytes();
    let mut out = vec![0.0_f64; cols];
    for i in 0..r {
        for (j, o) in out.iter_mut().enumerate() {
            let x = f64::from(bf16(&b, i * cols + j));
            *o += x * x;
        }
    }
    Ok(out.into_iter().map(f64::sqrt).collect())
}
/// A fixed pseudo-random order of `0..len` (xorshift, seeded); the same draws as `bcart`'s.
fn shuffle(len: usize, seed: u64) -> Vec<usize> {
    let mut v: Vec<usize> = (0..len).collect();
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    for i in (1..len).rev() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        v.swap(i, usize::try_from(x % (i as u64 + 1)).unwrap_or(0));
    }
    v
}
/// Indices of `score` from lowest to highest.
fn rising(score: &[f64]) -> Vec<usize> {
    let mut i: Vec<usize> = (0..score.len()).collect();
    i.sort_by(|a, b| score.get(*a).copied().unwrap_or(0.0).total_cmp(&score.get(*b).copied().unwrap_or(0.0)));
    i
}

#[derive(Clone, Debug)]
pub(crate) struct Cand {
    pub(crate) path: String,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) doc: String,
}
#[derive(Clone, Debug)]
pub(crate) struct Q {
    pub(crate) id: String,
    pub(crate) question: String,
    pub(crate) answers: Vec<(String, usize, usize)>,
    pub(crate) cands: Vec<Cand>,
}
impl Q {
    pub(crate) fn right(&self, c: &Cand) -> bool {
        self.answers.iter().any(|(f, a, b)| *f == c.path && c.start <= *b && c.end >= *a)
    }
    fn judged(&self) -> bool {
        self.cands.iter().any(|c| self.right(c)) && self.cands.iter().any(|c| !self.right(c))
    }
}

/// A judged set file (the form `bcart capture` writes); a held-out file is refused unopened.
pub(crate) fn load(home: &Path, name: &str) -> Result<Vec<Q>, Fault> {
    if name.contains("-held") || name.contains("real-use") {
        return Err(bad(format!("{name}: held-out files are not opened")));
    }
    let v = read_json(&home.join("scratch/base-cartridge").join(name))?;
    let mut out = Vec::new();
    for q in v.get("questions").and_then(Value::array).map_err(|e| bad(e.to_string()))? {
        let answers = q
            .get("answers")
            .and_then(Value::array)
            .unwrap_or_default()
            .iter()
            .map(|a| (text(a, "file"), int(a, "first"), a.get("last").and_then(Value::integer).ok().and_then(|x| usize::try_from(x).ok()).unwrap_or(usize::MAX)))
            .collect();
        let cands = q
            .get("cands")
            .and_then(Value::array)
            .unwrap_or_default()
            .iter()
            .map(|c| Cand { path: text(c, "path"), start: int(c, "start"), end: int(c, "end"), doc: text(c, "doc") })
            .collect();
        out.push(Q { id: text(q, "id"), question: text(q, "question"), answers, cands });
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}
/// The questions with a right and a wrong candidate: `fit` the even places, `test` the odd, `all`.
pub(crate) fn part<'a>(set: &'a [Q], which: &str) -> Vec<&'a Q> {
    set.iter().filter(|q| q.judged()).enumerate().filter(|(i, _)| which == "all" || (which == "fit") == (i % 2 == 0)).map(|(_, q)| q).collect()
}

fn half_math() {
    nmengn::gpu::HALF_MATH.store(true, Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
}
fn bind(home: &Path, pop: &str, asm: &str) -> Result<(Arc<Population>, Model), Fault> {
    let p = Population::open(home, pop, true, true)?;
    let a = Assembly::load(&home.join("designs/assemblies"), asm).map_err(|e| bad(e.to_string()))?;
    let m = Model::bind(&p, a)?;
    let m = if pop == JUDGE_POP { m.label_rows() } else { m };
    Ok((p, m))
}
fn swigates() -> Vec<(String, String)> {
    (0..LAYERS).map(|l| (nid(&format!("L{l}.swigate")), "y".to_owned())).collect()
}

/// Left-out channels per layer and the bias put in their place.
type Arm = (BTreeMap<usize, BTreeSet<usize>>, BTreeMap<usize, Vec<f32>>);

/// Plugs an arm's lesion on `m` (none for `open`). Returns the channels left out.
fn plug(m: &Model, pop: &str, arm: &Arm) -> Result<usize, Fault> {
    let (off, bias) = arm;
    if off.is_empty() {
        m.set_filters(None);
        return Ok(0);
    }
    let parts = off
        .iter()
        .map(|(l, o)| nmengn::filter::Part {
            target: format!("L{l}.swigate"),
            gain: (0..INTER).map(|c| if o.contains(&c) { 0.0 } else { 1.0 }).collect(),
            bias: bias.get(l).cloned().unwrap_or_else(|| vec![0.0; INTER]),
        })
        .collect();
    let f = nmengn::filter::build(pop, &[nmengn::filter::CartridgeFilter { cartridge: "lesion".into(), parts }]).map_err(bad)?;
    m.set_filters(Some(Arc::new(f)));
    Ok(off.values().map(BTreeSet::len).sum())
}

/// Channel means per layer from a statistics file (`mean` rows).
fn means(p: &Path) -> Result<Vec<Vec<f64>>, Fault> {
    let v = read_json(p)?;
    Ok(v.get("mean").and_then(Value::array).unwrap_or_default().iter().map(nums).collect())
}
/// The compensation for `off`: each left-out channel's mean, zero elsewhere, per layer.
fn compensation(off: &BTreeMap<usize, BTreeSet<usize>>, mean: &[Vec<f64>]) -> BTreeMap<usize, Vec<f32>> {
    off.iter()
        .map(|(l, o)| {
            let row = mean.get(*l).map_or(&[][..], Vec::as_slice);
            #[expect(clippy::cast_possible_truncation, reason = "a bias at binary32")]
            let b = (0..INTER).map(|c| if o.contains(&c) { row.get(c).copied().unwrap_or(0.0) as f32 } else { 0.0 }).collect();
            (*l, b)
        })
        .collect()
}
/// Writes a compensation as its own file (socket form: a bias at each layer's `swigate`).
fn write_bias(p: &Path, what: &str, bias: &BTreeMap<usize, Vec<f32>>) -> Result<(), Fault> {
    let layers: BTreeMap<String, Value> = bias.iter().map(|(l, b)| (format!("L{l}.swigate"), num_arr(&b.iter().map(|x| f64::from(*x)).collect::<Vec<_>>()))).collect();
    let v = Value::Object(
        [
            ("what".to_owned(), Value::String(what.to_owned())),
            ("socket".to_owned(), Value::String("gain-bias at each layer's swigate y (gain 0 for a left-out channel, 1 elsewhere)".to_owned())),
            ("bias".to_owned(), Value::Object(layers)),
        ]
        .into_iter()
        .collect(),
    );
    write_json(p, &v)
}

// ---------------------------------------------------------------------------------------------
// The judge.

fn judge_arm(home: &Path, arm: &str) -> Result<Arm, Fault> {
    let p: Vec<&str> = arm.split(':').collect();
    let map = || read_json(&home.join(JUDGE_MAPS).join("map.json"));
    let kept_lists = |share: &str| -> Result<Vec<Vec<usize>>, Fault> {
        let m = map()?;
        let at = m.get(share).map_err(|_| bad(format!("map.json has no share {share}")))?;
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "channel indices")]
        let rows = at.get("ch").and_then(Value::array).unwrap_or_default().iter().map(|r| nums(r).iter().map(|x| *x as usize).collect()).collect();
        Ok(rows)
    };
    let off_of_kept = |rows: Vec<Vec<usize>>| -> BTreeMap<usize, BTreeSet<usize>> {
        rows.into_iter()
            .enumerate()
            .filter(|(_, r)| r.len() < INTER)
            .map(|(l, r)| {
                let k: BTreeSet<usize> = r.into_iter().collect();
                (l, (0..INTER).filter(|c| !k.contains(c)).collect())
            })
            .collect()
    };
    let role = |share: &str| -> Result<BTreeMap<usize, BTreeSet<usize>>, Fault> {
        let m = map()?;
        let budget = m.get(share).and_then(|x| x.get("budget")).map(nums).map_err(|_| bad(format!("map.json has no budget at {share}")))?;
        let sc = read_json(&home.join(JUDGE_MAPS).join("role.json"))?;
        let rows: Vec<Vec<f64>> = sc.array().unwrap_or_default().iter().map(nums).collect();
        let mut out = BTreeMap::new();
        for (l, r) in budget.iter().enumerate() {
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "channel counts")]
            let r = *r as usize;
            if r > 0 {
                out.insert(l, rising(rows.get(l).map_or(&[][..], Vec::as_slice)).into_iter().take(r).collect());
            }
        }
        Ok(out)
    };
    match p.as_slice() {
        ["open"] => Ok((BTreeMap::new(), BTreeMap::new())),
        ["file", name] => file_arm(home, JUDGE_MAPS, name),
        ["map", share] => Ok((off_of_kept(kept_lists(share)?), BTreeMap::new())),
        ["role", share] => Ok((role(share)?, BTreeMap::new())),
        ["role+bias", share] => {
            let off = role(share)?;
            let b = compensation(&off, &means(&home.join(JUDGE_MAPS).join("swigate-mean.json"))?);
            write_bias(&home.join(JUDGE_MAPS).join(format!("role-{share}-bias.json")), &format!("compensation for the judge's role map at {share} percent: each left-out channel's mean output over the map part's reads"), &b)?;
            Ok((off, b))
        }
        ["rand", share, seed] => {
            let seed: u64 = seed.parse().map_err(|_| bad(format!("seed {seed}")))?;
            let rows = kept_lists(share)?
                .iter()
                .enumerate()
                .map(|(l, r)| {
                    let mut k: Vec<usize> = shuffle(INTER, seed * 1000 + l as u64).into_iter().take(r.len()).collect();
                    k.sort_unstable();
                    k
                })
                .collect();
            Ok((off_of_kept(rows), BTreeMap::new()))
        }
        _ => Err(bad(format!("judge arm {arm}: open | map:<share> | role:<share> | role+bias:<share> | rand:<share>:<seed>"))),
    }
}

/// `judge-mean`: the judge's channel means at every layer's `swigate` over all tokens of the map
/// part's reads (`set-p2-map.json`, the development questions its maps were made on).
fn judge_mean(home: &Path) -> Result<String, Fault> {
    let t0 = Instant::now();
    let set = load(home, "set-p2-map.json")?;
    half_math();
    let (pop, m) = bind(home, JUDGE_POP, JUDGE_ASM)?;
    let targets = swigates();
    let mut sum = vec![vec![0.0_f64; INTER]; LAYERS];
    let mut tokens = 0_usize;
    for q in &set {
        for c in &q.cands {
            let ids = pop.tokenizer.encode(&nmengn::roles::card_prompt(nmengn::roles::CODE_INSTRUCTION, &q.question, &c.doc)).map_err(|e| bad(e.to_string()))?;
            let o = m.forward(Device::Gpu, &ids, &targets, &RunOpts::default())?;
            for (t, sl) in targets.iter().zip(sum.iter_mut()) {
                let y = values(o.get(t).ok_or_else(|| bad("no swigate"))?)?;
                for row in y.as_chunks::<INTER>().0 {
                    for (s, x) in sl.iter_mut().zip(row) {
                        *s += f64::from(*x);
                    }
                }
            }
            tokens += ids.len();
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "a mean over tokens")]
    let n = tokens.max(1) as f64;
    let mean: Vec<Value> = sum.iter().map(|r| num_arr(&r.iter().map(|x| x / n).collect::<Vec<_>>())).collect();
    let v = Value::Object([("mean".to_owned(), Value::Array(mean)), ("tokens".to_owned(), Value::Number(tokens.to_string()))].into_iter().collect());
    write_json(&home.join(JUDGE_MAPS).join("swigate-mean.json"), &v)?;
    Ok(format!("judge-mean: {} questions, {tokens} tokens, {:.1} s\n", set.len(), t0.elapsed().as_secs_f64()))
}

fn out_file(home: &Path, model: &str, set: &str, arm: &str) -> PathBuf {
    home.join(OUT).join(format!("{model}-{}-{}.json", set.trim_end_matches(".json"), arm.replace([':', '+'], "-")))
}

/// The judge on a library base cartridge (`<id@v>`, `+comp` for its compensation), seated over
/// the open weights' model `open` by the engine's own seat. Returns the model and its seat line.
fn seat_judge(home: &Path, open: &Model, spec: &str) -> Result<(Model, String), Fault> {
    let (key, comp) = spec.strip_suffix("+comp").map_or((spec, false), |k| (k, true));
    let lib = nmlearn::library::Library::open(home).map_err(|e| bad(e.to_string()))?;
    let c = lib.cartridge(key).ok_or_else(|| bad(format!("{key} is not in the cartridge library")))?;
    let file = lib.cartridge_dir(c).join(c.files.first().map_or("", |f| f.name.as_str()));
    let offer = nmapi::neural::offer_of(&file)?;
    let (cart, st) = nmengn::base::seat(open, &offer).map_err(|e| bad(e.to_string()))?;
    let mut line = format!("seated {key}: largest change {:e}, bound {:?}", st.largest, offer.bound);
    if comp {
        match nmapi::neural::compensation_of(home, &file, &offer.key, &cart) {
            Some(Ok((k, _, p))) => {
                cart.set_filters(Some(Arc::new(p)));
                let _ = write!(line, "; {k} plugged");
            }
            Some(Err(e)) => return Err(bad(format!("compensation refused: {e}"))),
            None => return Err(bad(format!("{key} names no compensation"))),
        }
    }
    Ok((cart, line))
}

/// `judge <set> <arms>`: z per candidate for every question with a right and a wrong candidate.
fn judge(home: &Path, args: &[String]) -> Result<String, Fault> {
    let name = args.get(3).ok_or_else(|| bad("judge <set> <arm>[,<arm>...]"))?;
    let set = load(home, name)?;
    let qs = part(&set, "all");
    half_math();
    let (_pop, m) = bind(home, JUDGE_POP, JUDGE_ASM)?;
    let mut out = String::new();
    for arm in args.get(4).map_or("open", String::as_str).split(',') {
        let t = Instant::now();
        // `cart:<id@v>[+comp]`: the judge seated on a base cartridge from the library (the engine's
        // seat over the open weights), with the compensation its label names when `+comp`.
        let seated = match arm.strip_prefix("cart:") {
            Some(c) => Some(seat_judge(home, &m, c)?),
            None => None,
        };
        let n_off = if seated.is_some() { 0 } else { plug(&m, JUDGE_POP, &judge_arm(home, arm)?)? };
        let mm = seated.as_ref().map_or(&m, |(s, _)| s);
        let z: Result<Vec<Vec<f32>>, Fault> = qs
            .iter()
            .map(|q| {
                let docs: Vec<String> = q.cands.iter().map(|c| c.doc.clone()).collect();
                nmengn::roles::card_rerank_batch_z(mm, nmengn::roles::CODE_INSTRUCTION, &q.question, &docs, None)
            })
            .collect();
        m.set_filters(None);
        if let Some((s, line)) = seated {
            let _ = writeln!(out, "{line}");
            drop(s);
            nmengn::base::pull_base(JUDGE_POP);
        }
        let z = z?;
        let v: BTreeMap<String, Value> = qs.iter().zip(&z).map(|(q, r)| (q.id.clone(), num_arr(&r.iter().map(|x| f64::from(*x)).collect::<Vec<_>>()))).collect();
        write_json(&out_file(home, "judge", name, arm), &Value::Object(v))?;
        let _ = writeln!(out, "judge {arm}: {n_off} channels out, {} questions, {:.1} s", qs.len(), t.elapsed().as_secs_f64());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// The embedder.

/// The posed question as the meaning lane encodes it (the lab binds no request kind, so the
/// lane's answer instruction).
fn posed(question: &str) -> String {
    format!("Instruct: {}\nQuery:{question}", nmsubs::dense::task_for(question))
}

/// `embed-record <set>`: per channel at every layer's `swigate`, over all tokens of the fit half's
/// reads (each posed question and each candidate as the lane encodes them): the mean and the
/// energy (mean square times the down column's squared norm). Written to `record.json`.
fn embed_record(home: &Path, args: &[String]) -> Result<String, Fault> {
    let t0 = Instant::now();
    let name = args.get(3).ok_or_else(|| bad("embed-record <set>"))?;
    let set = load(home, name)?;
    let qs = part(&set, "fit");
    half_math();
    let (pop, m) = bind(home, EMBED_POP, EMBED_ASM)?;
    let targets = swigates();
    let mut sum = vec![vec![0.0_f64; INTER]; LAYERS];
    let mut sq = vec![vec![0.0_f64; INTER]; LAYERS];
    let mut tokens = 0_usize;
    let mut reads = 0_usize;
    for q in &qs {
        let texts: Vec<String> = std::iter::once(posed(&q.question)).chain(q.cands.iter().map(|c| c.doc.clone())).collect();
        for t in &texts {
            let ids = nmengn::roles::card_embed_ids(&m, t)?;
            let o = m.forward(Device::Gpu, &ids, &targets, &RunOpts::default())?;
            for ((tg, sl), ql) in targets.iter().zip(sum.iter_mut()).zip(sq.iter_mut()) {
                let y = values(o.get(tg).ok_or_else(|| bad("no swigate"))?)?;
                for row in y.as_chunks::<INTER>().0 {
                    for ((s, s2), x) in sl.iter_mut().zip(ql.iter_mut()).zip(row) {
                        let x = f64::from(*x);
                        *s += x;
                        *s2 += x * x;
                    }
                }
            }
            tokens += ids.len();
            reads += 1;
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "a mean over tokens")]
    let n = tokens.max(1) as f64;
    let (mut mean, mut energy) = (Vec::new(), Vec::new());
    for l in 0..LAYERS {
        let w = col_norms(&pop, &format!("model.layers.{l}.mlp.down_proj.weight"))?;
        let (sl, ql) = (sum.get(l).map_or(&[][..], Vec::as_slice), sq.get(l).map_or(&[][..], Vec::as_slice));
        mean.push(num_arr(&sl.iter().map(|x| x / n).collect::<Vec<_>>()));
        energy.push(num_arr(&ql.iter().zip(&w).map(|(q2, wc)| q2 / n * wc * wc).collect::<Vec<_>>()));
    }
    let v = Value::Object(
        [
            ("set".to_owned(), Value::String(name.clone())),
            ("questions".to_owned(), Value::Number(qs.len().to_string())),
            ("reads".to_owned(), Value::Number(reads.to_string())),
            ("tokens".to_owned(), Value::Number(tokens.to_string())),
            ("mean".to_owned(), Value::Array(mean)),
            ("energy".to_owned(), Value::Array(energy)),
        ]
        .into_iter()
        .collect(),
    );
    write_json(&home.join(EMBED_MAPS).join("record.json"), &v)?;
    Ok(format!("embed-record: {} fit questions, {reads} reads, {tokens} tokens, {:.1} s\n", qs.len(), t0.elapsed().as_secs_f64()))
}

/// An arm from a map file in the off form (`<maps>/<name>.json`, `off` per layer) with its
/// compensation (`<name>-bias.json`, the bias at each layer's `swigate`): the left-out channels
/// zeroed and their means put in their place, the same terms a cartridge cut from that map with its
/// compensation gives (used to measure the add-back steps without writing a cartridge).
fn file_arm(home: &Path, maps: &str, name: &str) -> Result<Arm, Fault> {
    let v = read_json(&home.join(maps).join(format!("{name}.json")))?;
    let mut off = BTreeMap::new();
    for (k, row) in v.get("off").and_then(Value::object).map_err(|e| bad(format!("{name}.json: {e}")))? {
        let Some(l) = k.strip_prefix('L').and_then(|x| x.parse::<usize>().ok()) else { continue };
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "channel indices")]
        let set: BTreeSet<usize> = nums(row).iter().map(|x| *x as usize).collect();
        if !set.is_empty() {
            off.insert(l, set);
        }
    }
    let mut bias = BTreeMap::new();
    if let Ok(b) = read_json(&home.join(maps).join(format!("{name}-bias.json"))) {
        for (k, row) in b.get("bias").and_then(Value::object).map_err(|e| bad(format!("{name}-bias.json: {e}")))? {
            let Some(l) = k.strip_prefix('L').and_then(|x| x.split('.').next()).and_then(|x| x.parse::<usize>().ok()) else { continue };
            #[expect(clippy::cast_possible_truncation, reason = "a bias at binary32")]
            bias.insert(l, nums(row).iter().map(|x| *x as f32).collect());
        }
    }
    Ok((off, bias))
}

fn embed_arm(home: &Path, arm: &str) -> Result<Arm, Fault> {
    let p: Vec<&str> = arm.split(':').collect();
    let count = |share: &str| -> Result<usize, Fault> {
        let s: f64 = share.parse().map_err(|_| bad(format!("share {share}")))?;
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss, reason = "a channel count")]
        let k = (s / 100.0 * INTER as f64).round() as usize;
        Ok(k)
    };
    let act = |share: &str| -> Result<BTreeMap<usize, BTreeSet<usize>>, Fault> {
        let k = count(share)?;
        let v = read_json(&home.join(EMBED_MAPS).join("record.json"))?;
        let rows: Vec<Vec<f64>> = v.get("energy").and_then(Value::array).unwrap_or_default().iter().map(nums).collect();
        if rows.len() != LAYERS {
            return Err(bad("record.json: run embed-record first"));
        }
        Ok(rows.iter().enumerate().map(|(l, r)| (l, rising(r).into_iter().take(k).collect())).collect())
    };
    match p.as_slice() {
        ["open"] => Ok((BTreeMap::new(), BTreeMap::new())),
        ["file", name] => file_arm(home, EMBED_MAPS, name),
        // The same map without its compensation (as the embedder serves by default).
        ["bare", name] => Ok((file_arm(home, EMBED_MAPS, name)?.0, BTreeMap::new())),
        ["act", share] => Ok((act(share)?, BTreeMap::new())),
        ["act+bias", share] => {
            let off = act(share)?;
            let b = compensation(&off, &means(&home.join(EMBED_MAPS).join("record.json"))?);
            write_bias(&home.join(EMBED_MAPS).join(format!("act-{share}-bias.json")), &format!("compensation for the embedder's activation map at {share} percent: each left-out channel's mean output over the fit half's reads"), &b)?;
            Ok((off, b))
        }
        ["rand", share, seed] => {
            let k = count(share)?;
            let seed: u64 = seed.parse().map_err(|_| bad(format!("seed {seed}")))?;
            Ok(((0..LAYERS).map(|l| (l, shuffle(INTER, seed * 1000 + l as u64).into_iter().take(k).collect())).collect(), BTreeMap::new()))
        }
        _ => Err(bad(format!("embed arm {arm}: open | act:<share> | act+bias:<share> | rand:<share>:<seed>"))),
    }
}

/// Writes the embedder's map at `share` percent (channels left out per layer) as its own file.
fn write_embed_map(home: &Path, share: &str, off: &BTreeMap<usize, BTreeSet<usize>>) -> Result<(), Fault> {
    let rows: BTreeMap<String, Value> = off.iter().map(|(l, o)| (format!("L{l}"), Value::Array(o.iter().map(|c| Value::Number(c.to_string())).collect()))).collect();
    let v = Value::Object(
        [
            ("what".to_owned(), Value::String(format!("the embedder's feed-forward channels left out at {share} percent per layer, lowest energy on the job's reads (record.json)"))),
            ("off".to_owned(), Value::Object(rows)),
        ]
        .into_iter()
        .collect(),
    );
    write_json(&home.join(EMBED_MAPS).join(format!("act-{share}.json")), &v)
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum()
}

/// `embed <set> <arms>`: per test question, the cosine of the posed question with each candidate.
fn embed(home: &Path, args: &[String]) -> Result<String, Fault> {
    let name = args.get(3).ok_or_else(|| bad("embed <set> <arm>[,<arm>...]"))?;
    let set = load(home, name)?;
    // `part=<fit|test|all>` after the arms (default `test`, the half the map was not made on).
    let which = args.iter().find_map(|a| a.strip_prefix("part=")).unwrap_or("test");
    let qs = part(&set, which);
    half_math();
    let (_pop, m) = bind(home, EMBED_POP, EMBED_ASM)?;
    let mut out = String::new();
    for arm in args.get(4).map_or("open", String::as_str).split(',') {
        let t = Instant::now();
        let a = embed_arm(home, arm)?;
        if let Some(share) = arm.strip_prefix("act:") {
            write_embed_map(home, share, &a.0)?;
        }
        let n_off = plug(&m, EMBED_POP, &a)?;
        let run = || -> Result<BTreeMap<String, Value>, Fault> {
            let mut v = BTreeMap::new();
            for q in &qs {
                let qv = nmengn::roles::card_embed_on(&m, Device::Gpu, &posed(&q.question))?;
                let mut row = Vec::with_capacity(q.cands.len());
                for c in &q.cands {
                    row.push(cos(&qv, &nmengn::roles::card_embed_on(&m, Device::Gpu, &c.doc)?));
                }
                v.insert(q.id.clone(), num_arr(&row));
            }
            Ok(v)
        };
        let v = run();
        m.set_filters(None);
        write_json(&out_file(home, "embed", name, arm), &Value::Object(v?))?;
        let _ = writeln!(out, "embed {arm}: {n_off} channels out, {} questions, {:.1} s", qs.len(), t.elapsed().as_secs_f64());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// The report.

/// Per question: pair margin, best right minus best wrong, right first (1 or 0).
fn measures(q: &Q, z: &[f64]) -> Option<[f64; 3]> {
    let (mut r, mut w) = (Vec::new(), Vec::new());
    for (c, x) in q.cands.iter().zip(z) {
        if q.right(c) { r.push(*x) } else { w.push(*x) }
    }
    if r.is_empty() || w.is_empty() || z.len() != q.cands.len() {
        return None;
    }
    let best = |v: &[f64]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut d = 0.0;
    for a in &r {
        for b in &w {
            d += a - b;
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "pair counts")]
    let pairs = (r.len() * w.len()) as f64;
    Some([d / pairs, best(&r) - best(&w), if best(&r) > best(&w) { 1.0 } else { 0.0 }])
}

#[expect(clippy::cast_precision_loss, reason = "statistics for a report")]
fn report_model(home: &Path, model: &str, name: &str, qs: &[&Q]) -> Result<String, Fault> {
    let prefix = format!("{model}-{}-", name.trim_end_matches(".json"));
    let mut arms: BTreeMap<String, BTreeMap<String, Vec<f64>>> = BTreeMap::new();
    let dir = home.join(OUT);
    for e in std::fs::read_dir(&dir).map_err(|e| bad(format!("{}: {e}", dir.display())))?.flatten() {
        let f = e.file_name().to_string_lossy().into_owned();
        if let Some(a) = f.strip_prefix(&prefix).and_then(|x| x.strip_suffix(".json")) {
            let Value::Object(o) = read_json(&e.path())? else { continue };
            arms.insert(a.to_owned(), o.iter().map(|(k, x)| (k.clone(), nums(x))).collect());
        }
    }
    let mut out = String::new();
    let Some(open) = arms.get("open") else { return Ok(format!("{model}: no open arm yet\n")) };
    let qs: Vec<&Q> = qs.iter().copied().filter(|q| open.get(&q.id).and_then(|z| measures(q, z)).is_some()).collect();
    let nq = qs.len().max(1) as f64;
    let meas = |z: &BTreeMap<String, Vec<f64>>| -> Vec<[f64; 3]> { qs.iter().map(|q| z.get(&q.id).and_then(|v| measures(q, v)).unwrap_or([f64::NAN; 3])).collect() };
    let mean_of = |m: &[[f64; 3]], k: usize| m.iter().map(|x| x.get(k).copied().unwrap_or(f64::NAN)).sum::<f64>() / nq;
    let _ = writeln!(out, "{model}: {} questions with a right and a wrong candidate", qs.len());
    let _ = writeln!(out, "  arm              | pair margin | best right minus best wrong | right first | mean |change| against open");
    let m0 = meas(open);
    let drift = |z: &BTreeMap<String, Vec<f64>>| -> f64 {
        let (mut s, mut c) = (0.0, 0_usize);
        for q in &qs {
            if let (Some(a), Some(b)) = (open.get(&q.id), z.get(&q.id)) {
                for (x, y) in a.iter().zip(b) {
                    s += (x - y).abs();
                    c += 1;
                }
            }
        }
        s / c.max(1) as f64
    };
    for (a, z) in arms.iter().filter(|(a, _)| !a.starts_with("rand-")) {
        let m = meas(z);
        let _ = writeln!(out, "  {a:<16} | {:+.4} | {:+.4} | {:.0} | {:.4}", mean_of(&m, 0), mean_of(&m, 1), mean_of(&m, 2) * nq, drift(z));
    }
    let shares: BTreeSet<String> = arms.keys().filter_map(|a| a.strip_prefix("rand-").and_then(|x| x.split('-').next()).map(str::to_owned)).collect();
    for sh in shares {
        let draws: Vec<(&String, &BTreeMap<String, Vec<f64>>)> = arms.iter().filter(|(a, _)| a.starts_with(&format!("rand-{sh}-"))).collect();
        let dm: Vec<Vec<[f64; 3]>> = draws.iter().map(|(_, z)| meas(z)).collect();
        let mut pm: Vec<f64> = dm.iter().map(|m| mean_of(m, 0)).collect();
        pm.sort_by(f64::total_cmp);
        let dd: Vec<f64> = draws.iter().map(|(_, z)| drift(z)).collect();
        let _ = writeln!(
            out,
            "  {} random masks at {sh} percent (same count per layer): pair margin lowest {:+.4}, median {:+.4}, highest {:+.4}; mean |change| {:.4} to {:.4}",
            draws.len(),
            pm.first().copied().unwrap_or(f64::NAN),
            pm.get(pm.len() / 2).copied().unwrap_or(f64::NAN),
            pm.last().copied().unwrap_or(f64::NAN),
            dd.iter().copied().fold(f64::INFINITY, f64::min),
            dd.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        );
        for (a, z) in arms.iter().filter(|(a, _)| !a.starts_with("rand-") && a.ends_with(&format!("-{sh}"))) {
            let m = meas(z);
            let p = mean_of(&m, 0);
            let above = pm.iter().filter(|x| p > **x).count();
            // Paired over questions: the arm's pair margin minus the random masks' mean.
            let d: Vec<f64> = (0..qs.len())
                .map(|i| m.get(i).map_or(f64::NAN, |x| x[0]) - dm.iter().map(|r| r.get(i).map_or(f64::NAN, |x| x[0])).sum::<f64>() / dm.len().max(1) as f64)
                .collect();
            let n = d.len().max(1) as f64;
            let mu = d.iter().sum::<f64>() / n;
            let se = (d.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (n - 1.0).max(1.0) / n).sqrt();
            let up = d.iter().filter(|x| **x > 0.0).count();
            let lost = m0.iter().zip(&m).map(|(o, x)| o[0] - x[0]).sum::<f64>() / nq;
            let verdict = if above * 20 >= draws.len() * 19 { "passes (above the 95th percentile)" } else { "does not pass" };
            let _ = writeln!(
                out,
                "    {a}: pair margin {p:+.4}, above {above} of {} masks: {verdict}; minus the masks' mean {mu:+.4} (standard error {se:.4}, t {:.2}, higher on {up} of {}); lost against open {lost:+.4}; mean |change| {:.4}, below {} of {} masks",
                draws.len(),
                mu / se.max(1e-12),
                qs.len(),
                drift(z),
                dd.iter().filter(|x| drift(z) < **x).count(),
                draws.len()
            );
        }
    }
    Ok(out)
}

fn report(home: &Path, args: &[String]) -> Result<String, Fault> {
    let name = args.get(3).ok_or_else(|| bad("report <set>"))?;
    let set = load(home, name)?;
    let mut out = report_model(home, "judge", name, &part(&set, "all"))?;
    out.push_str(&report_model(home, "embed", name, &part(&set, args.iter().find_map(|a| a.strip_prefix("part=")).unwrap_or("test")))?);
    Ok(out)
}

pub fn run(args: &[String], home: &Path) -> ExitCode {
    let step = args.get(2).map_or("report", String::as_str);
    let stamp = nmtype::now_ms();
    let result = match step {
        "judge-mean" => judge_mean(home),
        "judge" => judge(home, args),
        "embed-record" => embed_record(home, args),
        "embed" => embed(home, args),
        "report" => report(home, args),
        other => Err(bad(format!("unknown step {other} (judge-mean, judge, embed-record, embed, report)"))),
    };
    match result {
        Ok(t) => {
            let p = home.join(format!("runs/EMAP-{step}-{stamp}.txt"));
            let _ = std::fs::write(&p, format!("nano-lab {}\n{t}", args.get(1..).unwrap_or_default().join(" ")));
            print!("{t}");
            println!("report: {}", p.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("emap {step}: {e}");
            ExitCode::FAILURE
        }
    }
}
