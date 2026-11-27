//! `nano-lab run dx <name>...` (2026-11-10): root-cause diagnostics of the drone read
//! (direction of Luis Jake Gabriel III: find out by isolated pipeline tests why comprehension is not
//! improving: the MicroModel, the weights used, or the arrangement of the modules). Each test changes
//! one thing against one baseline and writes its own record `runs/DX-<name>-<ms>.txt`. Measure only;
//! nothing is fixed here.
//!
//! Inputs: the newest scan dump (`run tc1 dump`, `scratch/tc1-dump/<ms>/`): its tree (chunk texts),
//! and per question the window that holds the right chunk (the gold window) and the exact scan of
//! every other window. Fixed subset: the first ten development questions (by index) whose right
//! chunk is in some window. Even-index questions select heads, odd-index questions score them.
//!
//! Baseline: the released Qwen3-0.6B at q8, keys and values through layer 17 behind the anchor, the
//! exact read (question posed after the window), the default drone set of `designs/drones.json`,
//! pose "Question: ...", content-free pose "Question: N/A". Measures, per question on its gold
//! window: right chunk first inside its window, in the best 6, and first across all windows (the
//! other windows' scores from the dump; only where the other windows are unchanged by the test),
//! for the calibrated score and for the mass.
//!
//! Tests: `base`, `d1` window length, `d2` position, `d3` populations (reranker and embedding weights
//! as readers), `d4` model size (Qwen3-1.7B), `d5` head selection on full windows, `d6` depth, `d7`
//! precision, `d8` pose, `d9` chunk size. Added 2026-11-10 from the research note
//! `docs/2026-11-10-neumor-research-why-reads-degrade.md` section 8, every window of each question
//! read: `t1a` shared question, `t1` final short read (`t1b` re-encoded union), `t4` boundary and
//! temperature, `t5` order averaging. Records hold indices and ids only.
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use crate::MINI_ANCHOR;
use nmapi::scan_dump;
use nmapi::tentacle_pass::{self, calibrated};
use nmengn::Assembly;
use nmengn::kvstore::{self, ChunkKv, Window};
use nmengn::model::{Model, Population};
use nmengn::placement::Device;
use nmtype::{Fault, FaultKind};

use crate::e2::Item;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("dx", FaultKind::Execution, m)
}

/// Questions in the fixed subset.
const SUBSET: usize = 10;
/// Heads a selected set holds (the size of the default set, for a like-for-like comparison).
const PICK: usize = 16;
/// Stated poses.
const NULL: &str = "\nQuestion: N/A";

/// One question of the subset.
struct Case {
    idx: usize,
    id: String,
    text: String,
    keys: Vec<(String, usize, usize)>,
    /// The gold window: substrate chunk indices in window order and their gold flags.
    place: Vec<usize>,
    gold: Vec<bool>,
    /// Every chunk of the other windows from the dump: (calibrated above anchor, mass, gold).
    others: Vec<(f32, f32, bool)>,
}

/// A reader: a bound model, its device, the stop layer and the anchor window.
struct Reader {
    m: Arc<Model>,
    dev: Device,
    last: usize,
    stop: Vec<(String, String)>,
    aw: Arc<Window>,
    name: String,
}

/// A read's table: heads, parts (anchor first) and per head and part the posed and content-free
/// log-sum-exp rows and mass sums.
struct Table {
    heads: Vec<(usize, usize)>,
    parts: usize,
    cp: Vec<f32>,
    cn: Vec<f32>,
    mp: Vec<f32>,
    mn: Vec<f32>,
}
impl Table {
    /// Chunk scores (anchor excluded) for heads `set` (indices into `heads`): (calibrated above the
    /// anchor, mass).
    fn scores(&self, set: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let n = self.parts;
        let pick = |v: &[f32]| -> Vec<f32> { set.iter().flat_map(|&h| v.get(h * n..(h + 1) * n).unwrap_or(&[]).iter().copied()).collect() };
        let cal = calibrated(&pick(&self.cp), &pick(&self.cn), n);
        let a = cal.first().copied().unwrap_or(0.0);
        let cal: Vec<f32> = cal.into_iter().skip(1).map(|x| x - a).collect();
        let mass = (1..n)
            .map(|k| {
                set.iter().map(|&h| self.mp.get(h * n + k).copied().unwrap_or(0.0) - self.mn.get(h * n + k).copied().unwrap_or(0.0)).sum()
            })
            .collect();
        (cal, mass)
    }
    /// One head's reading per chunk above the anchor's.
    fn head(&self, h: usize) -> Vec<f32> {
        let n = self.parts;
        let at = |k: usize| self.cp.get(h * n + k).copied().unwrap_or(0.0) - self.cn.get(h * n + k).copied().unwrap_or(0.0);
        (1..n).map(|k| at(k) - at(0)).collect()
    }
    /// Indices of `pairs` in this table's heads.
    fn index(&self, pairs: &[(usize, usize)]) -> Vec<usize> {
        pairs.iter().filter_map(|p| self.heads.iter().position(|x| x == p)).collect()
    }
}

/// Place of the best gold chunk under `s` (ties: window order).
fn gold_place(s: &[f32], gold: &[bool]) -> Option<usize> {
    let mut idx: Vec<usize> = (0..s.len()).collect();
    idx.sort_by(|a, b| {
        s.get(*b).copied().unwrap_or(f32::NEG_INFINITY).total_cmp(&s.get(*a).copied().unwrap_or(f32::NEG_INFINITY)).then(a.cmp(b))
    });
    idx.iter().position(|i| gold.get(*i).copied().unwrap_or(false))
}

/// Place of the best gold chunk across this window and the others (`cal` true: calibrated).
fn across(s: &[f32], gold: &[bool], others: &[(f32, f32, bool)], cal: bool) -> Option<usize> {
    let mut v: Vec<(f32, bool)> = s.iter().zip(gold).map(|(x, g)| (*x, *g)).collect();
    v.extend(others.iter().map(|o| (if cal { o.0 } else { o.1 }, o.2)));
    v.sort_by(|a, b| b.0.total_cmp(&a.0));
    v.iter().position(|x| x.1)
}

/// Counts over cases: (first inside, best 6 inside, first across or `None` when not measured).
#[derive(Clone, Copy, Default)]
struct Tally {
    n: usize,
    first: usize,
    six: usize,
    across: Option<usize>,
}
impl Tally {
    fn add(&mut self, place: Option<usize>, acr: Option<Option<usize>>) {
        self.n += 1;
        self.first += usize::from(place == Some(0));
        self.six += usize::from(place.is_some_and(|p| p < 6));
        if let Some(a) = acr {
            self.across = Some(self.across.unwrap_or(0) + usize::from(a == Some(0)));
        }
    }
    fn show(&self) -> String {
        format!(
            "first {}/{}, best 6 {}/{}, across {}",
            self.first,
            self.n,
            self.six,
            self.n,
            self.across.map_or_else(|| "not measured".to_owned(), |a| format!("{a}/{}", self.n))
        )
    }
}

/// Calibrated and mass tallies of one variant.
#[derive(Clone, Copy, Default)]
struct Pair {
    cal: Tally,
    mass: Tally,
}
impl Pair {
    fn add(&mut self, t: &Table, set: &[usize], gold: &[bool], others: Option<&[(f32, f32, bool)]>) {
        let (c, m) = t.scores(set);
        self.cal.add(gold_place(&c, gold), others.map(|o| across(&c, gold, o, true)));
        self.mass.add(gold_place(&m, gold), others.map(|o| across(&m, gold, o, false)));
    }
    fn line(&self, name: &str) -> String {
        format!("| {name} | {} | {} |", self.cal.show(), self.mass.show())
    }
}

/// xorshift64* order of `0..n` for a seed (a fixed, stated shuffle).
fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut v: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        let r = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
        let j = usize::try_from(r % (i as u64 + 1)).unwrap_or(0);
        v.swap(i, j);
    }
    v
}

impl Reader {
    fn open(home: &Path, pop: &str, last: usize, f16: bool) -> Result<Self, Fault> {
        let m = if pop == nmapi::neural::POPULATION && !f16 {
            nmapi::neural::model(home)?
        } else {
            let p = Population::open(home, pop, true, true)?;
            let mut a = Assembly::load(&home.join("designs/assemblies"), "qwen3-reference")?;
            if f16 {
                for n in a.nodes.iter_mut().filter(|n| n.op == "linear") {
                    "f16".clone_into(&mut n.precision);
                }
            }
            Arc::new(Model::bind(&p, a)?)
        };
        let dev = if m.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
        let stop = kvstore::score_targets(&m, last)?;
        let ids = m.pop.tokenizer.encode(MINI_ANCHOR).map_err(|e| bad(e.to_string()))?;
        let ak = Arc::new(kvstore::encode_through(&m, dev, None, &ids, last)?);
        let aw = Arc::new(Window::build(&m, &[ak])?);
        Ok(Self { m, dev, last, stop, aw, name: format!("{pop}{}", if f16 { " (f16 linears)" } else { "" }) })
    }
    /// Encodings of `docs` behind the anchor through the stop layer, held in RAM only (2026-11-10:
    /// encodings are never written to disk).
    fn encode(&self, docs: &[String]) -> Result<Vec<Arc<ChunkKv>>, Fault> {
        docs.iter()
            .map(|d| {
                Ok(Arc::new(kvstore::encode_through(
                    &self.m,
                    self.dev,
                    Some(&self.aw),
                    &self.m.pop.tokenizer.encode(d).map_err(|e| bad(e.to_string()))?,
                    self.last,
                )?))
            })
            .collect()
    }
    fn read(&self, chunks: &[Arc<ChunkKv>], heads: &[(usize, usize)], posed: &str) -> Result<Table, Fault> {
        let mut parts = vec![Arc::clone(self.aw.chunks.first().ok_or_else(|| bad("no anchor"))?)];
        parts.extend(chunks.iter().cloned());
        let w = Arc::new(Window::build(&self.m, &parts)?);
        let (mp, cp) = tentacle_pass::drone_read_logits(&self.m, self.dev, &w, &self.stop, heads, posed)?;
        let (mn, cn) = tentacle_pass::drone_read_logits(&self.m, self.dev, &w, &self.stop, heads, NULL)?;
        w.release_device();
        let (len, wp) = (w.len, w.parts.clone());
        let sums = |v: &[f32]| -> Vec<f32> {
            let mut out = Vec::with_capacity(heads.len() * wp.len());
            for h in 0..heads.len() {
                for &(at, n) in &wp {
                    out.push(v.get(h * len + at..h * len + at + n).map_or(0.0, |s| s.iter().sum::<f32>()));
                }
            }
            out
        };
        Ok(Table { heads: heads.to_vec(), parts: wp.len(), cp, cn, mp: sums(&mp), mn: sums(&mn) })
    }
    /// Every head of layers `0..=last`.
    fn all_heads(&self) -> Vec<(usize, usize)> {
        (0..=self.last).flat_map(|l| (0..self.m.pop.dims.heads).map(move |h| (l, h))).collect()
    }
}

/// One line appended to `runs/DX-progress-<stamp>.txt` after a question (2026-11-10: a stopped run
/// keeps what it measured).
fn progress(home: &Path, stamp: u64, line: &str) {
    use std::io::Write as _;
    if let Ok(mut f) =
        std::fs::OpenOptions::new().create(true).append(true).open(home.join("runs").join(format!("DX-progress-{stamp}.txt")))
    {
        let _ = writeln!(f, "{line}");
    }
}

/// The best `PICK` heads (indices into the tables' heads) on the given cases: per head, the mean
/// over cases of the log-softmax over the window's chunks of the gold chunks (log-sum-exp over the
/// gold minus over all), CoRe style (design note 2.3); heads limited to `allowed`.
fn select(tables: &[(&Table, &[bool])], allowed: &dyn Fn(usize, usize) -> bool) -> Vec<usize> {
    let Some((t0, _)) = tables.first() else { return Vec::new() };
    let lse = |v: &[f32]| {
        let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if m.is_finite() { m + v.iter().map(|x| (x - m).exp()).sum::<f32>().ln() } else { m }
    };
    let mut score: Vec<(f32, usize)> = (0..t0.heads.len())
        .filter(|&h| t0.heads.get(h).is_some_and(|&(l, x)| allowed(l, x)))
        .map(|h| {
            let mut s = 0.0;
            for (t, gold) in tables {
                let r = t.head(h);
                let g: Vec<f32> = r.iter().zip(gold.iter()).filter(|(_, g)| **g).map(|(x, _)| *x).collect();
                s += lse(&g) - lse(&r);
            }
            (s, h)
        })
        .collect();
    score.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    score.into_iter().take(PICK).map(|x| x.1).collect()
}

fn doc(sub: &nmsubs::Substrate, c: usize) -> String {
    sub.chunks.get(c).and_then(|ch| sub.files.get(ch.file).map(|f| nmsubs::dense::doc_text(&f.path, &ch.text))).unwrap_or_default()
}

/// The subset from the newest dump.
fn cases(home: &Path, dir: &Path) -> Result<(nmsubs::Substrate, Vec<Case>), Fault> {
    let items: Vec<Item> = crate::e2::suite(home)?;
    let sub = scan_dump::load_substrate(dir)?;
    let mut out = Vec::new();
    for idx in scan_dump::indices(dir) {
        if out.len() >= SUBSET {
            break;
        }
        let Some(x) = items.get(idx) else { continue };
        let (q, scan) = scan_dump::load_question(dir, idx, &x.q)?;
        let Some(gw) = scan.windows.iter().position(|w| w.place.iter().any(|c| q.gold.contains(c))) else { continue };
        let mut others = Vec::new();
        for (i, w) in scan.windows.iter().enumerate() {
            if i == gw {
                continue;
            }
            let c = calibrated(&w.cal_posed, &w.cal_null, w.parts.len());
            let a = c.first().copied().unwrap_or(0.0);
            let mass = w.scores(tentacle_pass::Score::Mass);
            for (k, ch) in w.place.iter().enumerate() {
                others.push((
                    c.get(k + 1).copied().unwrap_or(f32::NEG_INFINITY) - a,
                    mass.get(k).copied().unwrap_or(f32::NEG_INFINITY),
                    q.gold.contains(ch),
                ));
            }
        }
        let place = scan.windows.get(gw).map(|w| w.place.clone()).unwrap_or_default();
        let gold = place.iter().map(|c| q.gold.contains(c)).collect();
        out.push(Case { idx, id: q.id, text: x.q.clone(), keys: x.keys.clone(), place, gold, others });
    }
    Ok((sub, out))
}

/// Re-chunked gold windows (`d9`): `half` splits every chunk in two by lines, otherwise neighbours of
/// the same file are merged in pairs. Returns per new chunk its document and gold flag.
fn rechunk(sub: &nmsubs::Substrate, c: &Case, half: bool) -> Vec<(String, bool)> {
    let span = |file: &str, a: usize, b: usize| c.keys.iter().any(|(f, x, y)| f == file && a <= *y && b >= *x);
    let mut base: Vec<(String, usize, usize, String)> = Vec::new();
    for &ci in &c.place {
        let Some(ch) = sub.chunks.get(ci) else { continue };
        let path = sub.files.get(ch.file).map(|f| f.path.clone()).unwrap_or_default();
        if half {
            let lines: Vec<&str> = ch.text.split_inclusive('\n').collect();
            let h = lines.len() / 2;
            if h == 0 {
                base.push((path, ch.start, ch.end, ch.text.clone()));
            } else {
                base.push((path.clone(), ch.start, ch.start + h - 1, lines.get(..h).unwrap_or(&[]).concat()));
                base.push((path, ch.start + h, ch.end, lines.get(h..).unwrap_or(&[]).concat()));
            }
        } else {
            base.push((path, ch.start, ch.end, ch.text.clone()));
        }
    }
    let mut pieces: Vec<(String, usize, usize, String)> = Vec::new();
    let mut i = 0;
    while let Some(a) = base.get(i) {
        match base.get(i + 1).filter(|b| !half && b.0 == a.0) {
            Some(b) => {
                let text = if a.3.ends_with('\n') { format!("{}{}", a.3, b.3) } else { format!("{}\n{}", a.3, b.3) };
                pieces.push((a.0.clone(), a.1.min(b.1), a.2.max(b.2), text));
                i += 2;
            }
            None => {
                pieces.push(a.clone());
                i += 1;
            }
        }
    }
    pieces.into_iter().map(|(p, a, b, t)| (nmsubs::dense::doc_text(&p, &t), span(&p, a, b))).collect()
}

/// `run dx <name>...`.
#[expect(clippy::too_many_lines, reason = "one battery, in the order it runs")]
fn battery(home: &Path, names: &[String], dir: &Path, stamp: u64) -> Result<Vec<(String, String)>, Fault> {
    let (sub, cs) = cases(home, dir)?;
    let set = tentacle_pass::drones(home);
    let head = format!(
        "dump {}; subset: the first {} development questions with the right chunk in some window: {}; even-index questions select, odd-index questions score",
        dir.display(),
        cs.len(),
        cs.iter().map(|c| format!("{} ({})", c.idx, c.id)).collect::<Vec<_>>().join(", ")
    );
    let want = |n: &str| names.iter().any(|x| x == n);
    let mut recs: Vec<(String, String)> = Vec::new();
    let even = |c: &Case| c.idx.is_multiple_of(2);
    let started = Instant::now();

    // Base-encoding group: base, d1, d2, d5, d8, d9.
    if ["base", "d1", "d2", "d5", "d8", "d9"].iter().any(|n| want(n)) {
        let r = Reader::open(home, nmapi::neural::POPULATION, set.last_layer(), false)?;
        let mut base = Pair::default();
        let mut equal_dump = 0;
        let mut d1: BTreeMap<usize, (Pair, f64)> = BTreeMap::new();
        let mut d2: BTreeMap<String, Pair> = BTreeMap::new();
        let mut d8: BTreeMap<&str, Pair> = BTreeMap::new();
        let mut d9: BTreeMap<&str, (Pair, usize, usize)> = BTreeMap::new();
        let mut all_tables: Vec<(Table, Vec<bool>, bool)> = Vec::new();
        let mut t_enc = 0.0;
        let frames: [(&str, &str); 3] = [
            ("current", "\nQuestion: {q}"),
            ("instruction", "\nInstruct: Given a question about this repository, retrieve the code that answers it\nQuery: {q}"),
            ("code comment", "\n// Question about the code above: {q}\n// The code that answers it:"),
        ];
        for c in &cs {
            let t = Instant::now();
            let docs: Vec<String> = c.place.iter().map(|x| doc(&sub, *x)).collect();
            let kv = r.encode(&docs)?;
            t_enc += t.elapsed().as_secs_f64();
            let posed = format!("\nQuestion: {}", c.text);
            let def: Vec<(usize, usize)> = set.heads.clone();
            let tb = r.read(&kv, &def, &posed)?;
            let all: Vec<usize> = (0..def.len()).collect();
            base.add(&tb, &all, &c.gold, Some(&c.others));
            {
                let (cs1, ms1) = tb.scores(&all);
                progress(
                    home,
                    stamp,
                    &format!(
                        "base q{} ({}): gold place inside calibrated {:?} mass {:?}",
                        c.idx,
                        c.id,
                        gold_place(&cs1, &c.gold),
                        gold_place(&ms1, &c.gold)
                    ),
                );
            }
            // The re-read equals the dump's own exact read of this window?
            if let Ok((_, scan)) = scan_dump::load_question(dir, c.idx, &c.text)
                && let Some(w) = scan.windows.iter().find(|w| w.place == c.place)
                && w.cal_posed.iter().map(|x| x.to_bits()).eq(tb.cp.iter().map(|x| x.to_bits()))
            {
                equal_dump += 1;
            }
            if want("d1") {
                for cut in [1_500_usize, 3_000, 6_000, 12_000] {
                    for seed in 1..=3_u64 {
                        let order = shuffled(kv.len(), seed);
                        let mut keep = vec![false; kv.len()];
                        let mut used: usize = c.gold.iter().zip(&kv).filter(|(g, _)| **g).map(|(_, k)| k.tokens).sum();
                        for (i, g) in c.gold.iter().enumerate() {
                            if *g && let Some(x) = keep.get_mut(i) {
                                *x = true;
                            }
                        }
                        for i in order {
                            let tk = kv.get(i).map_or(0, |k| k.tokens);
                            if !keep.get(i).copied().unwrap_or(true) && used + tk <= cut {
                                if let Some(x) = keep.get_mut(i) {
                                    *x = true;
                                }
                                used += tk;
                            }
                        }
                        let sub_kv: Vec<Arc<ChunkKv>> = kv.iter().zip(&keep).filter(|(_, k)| **k).map(|(x, _)| Arc::clone(x)).collect();
                        let gold: Vec<bool> = c.gold.iter().zip(&keep).filter(|(_, k)| **k).map(|(g, _)| *g).collect();
                        let t2 = r.read(&sub_kv, &def, &posed)?;
                        let e = d1.entry(cut).or_default();
                        e.0.add(&t2, &all, &gold, Some(&c.others));
                        #[expect(clippy::cast_precision_loss, reason = "counts")]
                        {
                            e.1 += gold.len() as f64;
                        }
                    }
                }
            }
            if want("d2") {
                let gi = c.gold.iter().position(|g| *g).unwrap_or(0);
                let rest: Vec<usize> = (0..kv.len()).filter(|i| *i != gi).collect();
                let mut orders: Vec<(String, Vec<usize>)> =
                    (1..=3_u64).map(|s| (format!("shuffle seed {s}"), shuffled(kv.len(), s))).collect();
                let mut first = vec![gi];
                first.extend(&rest);
                let mut last = rest.clone();
                last.push(gi);
                let mut mid = rest.clone();
                mid.insert(rest.len() / 2, gi);
                orders.push(("right chunk first".into(), first));
                orders.push(("right chunk middle".into(), mid));
                orders.push(("right chunk last".into(), last));
                for (name, o) in orders {
                    let okv: Vec<Arc<ChunkKv>> = o.iter().filter_map(|i| kv.get(*i).cloned()).collect();
                    let og: Vec<bool> = o.iter().map(|i| c.gold.get(*i).copied().unwrap_or(false)).collect();
                    let t2 = r.read(&okv, &def, &posed)?;
                    d2.entry(name).or_default().add(&t2, &all, &og, Some(&c.others));
                }
            }
            if want("d8") {
                for (name, f) in frames {
                    let t2 = r.read(&kv, &def, &f.replace("{q}", &c.text))?;
                    d8.entry(name).or_default().add(&t2, &all, &c.gold, Some(&c.others));
                }
            }
            if want("d5") {
                let heads = r.all_heads();
                all_tables.push((r.read(&kv, &heads, &posed)?, c.gold.clone(), even(c)));
            }
            drop(kv);
            if want("d9") {
                for (name, half) in [("half size", true), ("double size", false)] {
                    let pieces = rechunk(&sub, c, half);
                    let docs: Vec<String> = pieces.iter().map(|p| p.0.clone()).collect();
                    let gold: Vec<bool> = pieces.iter().map(|p| p.1).collect();
                    let kv2 = r.encode(&docs)?;
                    let t2 = r.read(&kv2, &def, &posed)?;
                    let e = d9.entry(name).or_insert((Pair::default(), 0, 0));
                    e.0.add(&t2, &all, &gold, Some(&c.others));
                    e.1 += gold.len();
                    e.2 += kv2.iter().map(|k| k.tokens).sum::<usize>();
                }
            }
            nmengn::residency::get().release_windows();
        }
        let base_line = base.line("baseline (0.6B q8, layer 17, default set, full window)");
        let hdr = "| variant | calibrated | mass |\n|---|---|---|";
        if want("base") {
            recs.push(("base".into(), format!("DX base: baseline of the root-cause battery\n{head}\nreader {}; encode {t_enc:.0} s for {} windows\nre-read equals the dump's exact read of the same window bit for bit: {equal_dump} of {}\n{hdr}\n{base_line}\n", r.name, cs.len(), cs.len())));
        }
        if want("d1") {
            let mut t = format!(
                "DX d1: window length (the gold window cut to about 1,500, 3,000, 6,000 and 12,000 tokens keeping the right chunk; the other chunks kept in a seeded shuffled order (seeds 1 to 3) until the cut, in window order; the dump does not hold the lens rank, so a fixed shuffle stands in for it)\n{head}\n{hdr}\n{base_line}\n"
            );
            for (cut, (p, n)) in &d1 {
                let _ = writeln!(
                    t,
                    "{} (mean {:.1} chunks in the window; counts over 3 seeds x {} questions)",
                    p.line(&format!("cut {cut} tokens")),
                    n / (3.0 * cs.len().max(1) as f64),
                    cs.len()
                );
            }
            recs.push(("d1".into(), t));
        }
        if want("d2") {
            let mut t = format!(
                "DX d2: position (same chunks, order shuffled with 3 seeds, and the right chunk moved first, middle, last with the others in window order)\n{head}\n{hdr}\n{base_line}\n"
            );
            for (name, p) in &d2 {
                let _ = writeln!(t, "{}", p.line(name));
            }
            recs.push(("d2".into(), t));
        }
        if want("d8") {
            let mut t = format!("DX d8: pose (question frame changed; content-free pose unchanged)\n{head}\n{hdr}\n{base_line}\n");
            for (name, f) in frames {
                let _ =
                    writeln!(t, "{} frame {:?}", d8.get(name).map_or_else(String::new, |p| p.line(name)), f.replace("{q}", "<question>"));
            }
            recs.push(("d8".into(), t));
        }
        if want("d9") {
            let mut t = format!(
                "DX d9: chunk size (gold window re-chunked: every chunk split in two by lines, or same-file neighbours merged in pairs; gold by the same rule on the new line spans)\n{head}\n{hdr}\n{base_line}\n"
            );
            for (name, (p, n, tok)) in &d9 {
                let _ = writeln!(t, "{} ({n} chunks, {tok} tokens over {} windows)", p.line(name), cs.len());
            }
            recs.push(("d9".into(), t));
        }
        if want("d5") {
            let evens: Vec<(&Table, &[bool])> = all_tables.iter().filter(|x| x.2).map(|x| (&x.0, x.1.as_slice())).collect();
            let pick = select(&evens, &|_, _| true);
            let mut sel = Pair::default();
            let mut def_odd = Pair::default();
            for (tb, g, ev) in &all_tables {
                if !*ev {
                    sel.add(tb, &pick, g, None);
                    def_odd.add(tb, &tb.index(&set.heads), g, None);
                }
            }
            let names: Vec<String> =
                pick.iter().filter_map(|h| all_tables.first().and_then(|t| t.0.heads.get(*h)).map(|(l, x)| format!("L{l}.h{x}"))).collect();
            recs.push(("d5".into(), format!("DX d5: head selection on full 12,000-token windows (16 heads of layers 0..=17 selected on the even-index questions by the gold log-softmax of each head's calibrated reading; scored on the odd-index questions; across windows not measured: the other windows hold no readings of the new heads)\n{head}\nselected: {}\n{hdr}\n{}\n{}\n", names.join(" "), def_odd.line("current set (selected on short windows), odd questions"), sel.line("set selected on full windows, odd questions"))));
        }
    }

    // Reader group: d3, d4, d6, d7.
    for (name, pops) in [
        ("d3", vec![("qwen3-reranker-0.6b", false), ("qwen3-embedding-0.6b", false), (nmapi::neural::POPULATION, false)]),
        ("d4", vec![("qwen3-1.7b", false), (nmapi::neural::POPULATION, false)]),
        ("d7", vec![(nmapi::neural::POPULATION, true), (nmapi::neural::POPULATION, false)]),
    ] {
        if !want(name) {
            continue;
        }
        let mut t = format!(
            "DX {name}: {}\n{head}\n| reader | heads | calibrated | mass |\n|---|---|---|---|\n",
            match name {
                "d3" =>
                    "which weights read (drone heads from the reranker and the embedding weights, keys and values encoded by the same population; default indices, and 16 heads selected on even questions scored on odd)",
                "d4" =>
                    "model size (Qwen3-1.7B as the reader, encoded by itself through layer 17; default indices, and 16 heads selected on even questions scored on odd)",
                _ =>
                    "precision (linear weights at 16-bit floats against the served 8-bit weights, same windows; keys and values are stored at the engine's activation precision, binary16, in both)",
            }
        );
        let mut base_tables: Option<Vec<Table>> = None;
        for (pop, f16) in pops {
            let r = match Reader::open(home, pop, set.last_layer(), f16) {
                Ok(r) => r,
                Err(e) => {
                    let _ = writeln!(t, "| {pop} | - | not bound: {} | |", e.message);
                    continue;
                }
            };
            let heads = if name == "d7" { set.heads.clone() } else { r.all_heads() };
            let te = Instant::now();
            let mut tabs: Vec<(Table, Vec<bool>, bool)> = Vec::new();
            for c in &cs {
                let docs: Vec<String> = c.place.iter().map(|x| doc(&sub, *x)).collect();
                let kv = r.encode(&docs)?;
                tabs.push((r.read(&kv, &heads, &format!("\nQuestion: {}", c.text))?, c.gold.clone(), even(c)));
                if let Some((tb, g, _)) = tabs.last() {
                    let ix = tb.index(&set.heads);
                    let (cs1, ms1) = tb.scores(&ix);
                    progress(
                        home,
                        stamp,
                        &format!(
                            "{} {} q{} ({}): default set, gold place inside calibrated {:?} mass {:?}; {} heads recorded",
                            name,
                            r.name,
                            c.idx,
                            c.id,
                            gold_place(&cs1, g),
                            gold_place(&ms1, g),
                            tb.heads.len()
                        ),
                    );
                }
                drop(kv);
                nmengn::residency::get().release_windows();
            }
            let secs = te.elapsed().as_secs_f64();
            let mut def_all = Pair::default();
            let mut def_odd = Pair::default();
            for (tb, g, ev) in &tabs {
                let ix = tb.index(&set.heads);
                def_all.add(tb, &ix, g, None);
                if !*ev {
                    def_odd.add(tb, &ix, g, None);
                }
            }
            let _ = writeln!(t, "| {} | default indices, all 10 | {} | {} |", r.name, def_all.cal.show(), def_all.mass.show());
            let _ = writeln!(t, "| {} | default indices, odd | {} | {} |", r.name, def_odd.cal.show(), def_odd.mass.show());
            if name != "d7" {
                let evens: Vec<(&Table, &[bool])> = tabs.iter().filter(|x| x.2).map(|x| (&x.0, x.1.as_slice())).collect();
                let pick = select(&evens, &|_, _| true);
                let mut sel = Pair::default();
                for (tb, g, ev) in &tabs {
                    if !*ev {
                        sel.add(tb, &pick, g, None);
                    }
                }
                let hn: Vec<String> =
                    pick.iter().filter_map(|h| tabs.first().and_then(|x| x.0.heads.get(*h)).map(|(l, x)| format!("L{l}.h{x}"))).collect();
                let _ = writeln!(
                    t,
                    "| {} | 16 selected on even, scored on odd ({}) | {} | {} |",
                    r.name,
                    hn.join(" "),
                    sel.cal.show(),
                    sel.mass.show()
                );
            }
            let _ = writeln!(t, "({}: {secs:.0} s for {} windows)", r.name, cs.len());
            if name == "d7" {
                let mine: Vec<Table> = tabs.into_iter().map(|x| x.0).collect();
                if let Some(other) = base_tables.as_ref() {
                    let mut worst = 0.0_f32;
                    let mut rho = Vec::new();
                    for (a, b) in mine.iter().zip(other) {
                        let all: Vec<usize> = (0..a.heads.len()).collect();
                        let (ca, _) = a.scores(&all);
                        let (cb, _) = b.scores(&all);
                        for (x, y) in ca.iter().zip(&cb) {
                            worst = worst.max((x - y).abs());
                        }
                        if let Some(r) = nmapi::key_scan::spearman(&ca, &cb) {
                            rho.push(r);
                        }
                    }
                    #[expect(clippy::cast_precision_loss, reason = "counts")]
                    let m = rho.iter().sum::<f64>() / rho.len().max(1) as f64;
                    let _ = writeln!(
                        t,
                        "f16 against q8 linears: largest calibrated score difference {worst:.3}; rank correlation of the chunk scores per window, mean {m:.4}"
                    );
                } else {
                    base_tables = Some(mine);
                }
            }
            drop(r);
        }
        recs.push((name.into(), t));
    }

    if want("d6") {
        let depths = [11_usize, 14, 17, 20, 23];
        let deep = depths.iter().copied().max().unwrap_or(17);
        let r = Reader::open(home, nmapi::neural::POPULATION, deep, false)?;
        let heads = r.all_heads();
        let mut tabs: Vec<(Table, Vec<bool>, bool)> = Vec::new();
        let te = Instant::now();
        for c in &cs {
            let docs: Vec<String> = c.place.iter().map(|x| doc(&sub, *x)).collect();
            let kv = r.encode(&docs)?;
            tabs.push((r.read(&kv, &heads, &format!("\nQuestion: {}", c.text))?, c.gold.clone(), even(c)));
            if let Some((tb, g, _)) = tabs.last() {
                let ix = tb.index(&set.heads);
                let (cs1, ms1) = tb.scores(&ix);
                progress(
                    home,
                    stamp,
                    &format!(
                        "{} {} q{} ({}): default set, gold place inside calibrated {:?} mass {:?}; {} heads recorded",
                        "d6",
                        "qwen3-0.6b through layer 23",
                        c.idx,
                        c.id,
                        gold_place(&cs1, g),
                        gold_place(&ms1, g),
                        tb.heads.len()
                    ),
                );
            }
            drop(kv);
            nmengn::residency::get().release_windows();
        }
        let mut t = format!(
            "DX d6: depth (keys and values through layer {deep}; one read stopped at layer {deep} records every head of layers 0..={deep}, which equals a read stopped at any lower layer for those layers; per stop layer, 16 heads of layers 0..=stop selected on even questions, scored on odd)\n{head}\n({:.0} s for {} windows)\n| stop layer | selected heads | calibrated | mass |\n|---|---|---|---|\n",
            te.elapsed().as_secs_f64(),
            cs.len()
        );
        let evens: Vec<(&Table, &[bool])> = tabs.iter().filter(|x| x.2).map(|x| (&x.0, x.1.as_slice())).collect();
        let mut def_odd = Pair::default();
        for (tb, g, ev) in &tabs {
            if !*ev {
                def_odd.add(tb, &tb.index(&set.heads), g, None);
            }
        }
        let _ = writeln!(t, "| 17 (current set) | default | {} | {} |", def_odd.cal.show(), def_odd.mass.show());
        for d in depths {
            let pick = select(&evens, &|l, _| l <= d);
            let mut sel = Pair::default();
            for (tb, g, ev) in &tabs {
                if !*ev {
                    sel.add(tb, &pick, g, None);
                }
            }
            let hn: Vec<String> = pick.iter().filter_map(|h| heads.get(*h).map(|(l, x)| format!("L{l}.h{x}"))).collect();
            let _ = writeln!(t, "| {d} | {} | {} | {} |", hn.join(" "), sel.cal.show(), sel.mass.show());
        }
        recs.push(("d6".into(), t));
    }
    for (_, t) in &mut recs {
        let ram =
            nmffi::dynlib::process_memory().map_or_else(|e| e, |(peak, now)| format!("RAM peak {} MB, now {} MB", peak >> 20, now >> 20));
        let _ = writeln!(t, "\nrun {stamp}: {:.0} s so far; {ram}", started.elapsed().as_secs_f64());
    }
    Ok(recs)
}

// ---- T battery (2026-11-10, research note `docs/2026-11-10-neumor-research-why-reads-degrade.md`
// section 8): tests of the loss between windows, every window of each question read. ----

/// One window of a question: chunks, their encodings, gold flags, documents.
#[expect(dead_code, reason = "kept from the DX diagnostics for the next read step; no current subcommand calls it")]
type WinData = (Vec<usize>, Vec<Arc<ChunkKv>>, Vec<bool>, Vec<String>);
/// Chunk scores of one window with gold flags.
#[expect(dead_code, reason = "kept from the DX diagnostics for the next read step; no current subcommand calls it")]
type Scored = (Vec<f32>, Vec<bool>);
/// A window's raw reads: posed and content-free scores, tokens, length, parts.
#[expect(dead_code, reason = "kept from the DX diagnostics for the next read step; no current subcommand calls it")]
type RawRead = (Vec<Vec<f32>>, Vec<Vec<f32>>, usize, usize, Vec<(usize, usize)>);

/// Raw scores of an exact read: per head of `heads`, `[tokens, len]`, and the token count.
fn read_raw(r: &Reader, w: &Arc<Window>, heads: &[(usize, usize)], text: &str) -> Result<(Vec<Vec<f32>>, usize), Fault> {
    let ids = r.m.pop.tokenizer.encode(text).map_err(|e| bad(e.to_string()))?;
    let probe = Arc::new(nmengn::kvstore::Probe::of_heads(heads).with_logits());
    r.m.forward_at(
        r.dev,
        &ids,
        w.len,
        &r.stop,
        &nmengn::model::RunOpts { window: Some(Arc::clone(w)), probe: Some(Arc::clone(&probe)), ..nmengn::model::RunOpts::default() },
    )?;
    let mut out = Vec::with_capacity(heads.len());
    for &(l, h) in heads {
        out.push(probe.logits(l, h).ok_or_else(|| bad(format!("no logits for layer {l} head {h}")))?.logits);
    }
    Ok((out, ids.len()))
}

/// Per head and part, `T * lse(s / T)` over the part's positions not masked, averaged over the
/// question tokens; `heads x parts` (the calibrated table's form; `T` 1 and no mask give
/// `tentacle_pass::part_lse` up to rounding).
fn table_of(
    logits: &[Vec<f32>],
    tokens: usize,
    len: usize,
    parts: &[(usize, usize)],
    keep: &dyn Fn(usize, usize) -> bool,
    temp: f32,
) -> Vec<f32> {
    #[expect(clippy::cast_precision_loss, reason = "token counts")]
    let per = 1.0 / tokens.max(1) as f32;
    let mut out = Vec::with_capacity(logits.len() * parts.len());
    for lg in logits {
        for (k, &(at, n)) in parts.iter().enumerate() {
            let mut acc = 0.0_f32;
            for t in 0..tokens {
                let row: Vec<f32> =
                    (at..at + n).filter(|p| keep(k, p - at)).filter_map(|p| lg.get(t * len + p).map(|x| x / temp)).collect();
                let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if m.is_finite() {
                    acc += temp * (m + row.iter().map(|x| (x - m).exp()).sum::<f32>().ln());
                }
            }
            out.push(acc * per);
        }
    }
    out
}

/// Chunk scores above the anchor from two `heads x parts` tables.
fn above_anchor(p: &[f32], n: &[f32], parts: usize) -> Vec<f32> {
    let c = calibrated(p, n, parts);
    let a = c.first().copied().unwrap_or(0.0);
    c.into_iter().skip(1).map(|x| x - a).collect()
}

/// Mean entropy (nats) of the attention over the window positions in a content-free read: per head
/// and token, the softmax of the scores over the window only.
fn entropy(logits: &[Vec<f32>], tokens: usize, len: usize) -> f64 {
    let mut s = 0.0_f64;
    let mut k = 0_usize;
    for lg in logits {
        for t in 0..tokens {
            let row = lg.get(t * len..(t + 1) * len).unwrap_or(&[]);
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if !m.is_finite() {
                continue;
            }
            let z: f64 = row.iter().map(|x| f64::from((x - m).exp())).sum();
            let h: f64 = row.iter().map(|x| f64::from((x - m).exp()) / z).filter(|p| *p > 0.0).map(|p| -p * p.ln()).sum();
            s += h;
            k += 1;
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let n = k.max(1) as f64;
    s / n
}

/// Cosine of two equally long vectors.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (x, y) in a.iter().zip(b) {
        ab += f64::from(*x) * f64::from(*y);
        aa += f64::from(*x) * f64::from(*x);
        bb += f64::from(*y) * f64::from(*y);
    }
    if aa > 0.0 && bb > 0.0 { ab / (aa * bb).sqrt() } else { 0.0 }
}

/// Place of the best gold chunk across windows from per-window (scores, gold) lists.
#[expect(dead_code, reason = "kept from the DX diagnostics for the next read step; no current subcommand calls it")]
fn across_all(ws: &[(Vec<f32>, Vec<bool>)]) -> Option<usize> {
    let mut v: Vec<(f32, bool)> = ws.iter().flat_map(|(s, g)| s.iter().zip(g).map(|(x, y)| (*x, *y))).collect();
    v.sort_by(|a, b| b.0.total_cmp(&a.0));
    v.iter().position(|x| x.1)
}

/// Best gold place inside the windows (minimum over windows that hold gold).
#[expect(dead_code, reason = "kept from the DX diagnostics for the next read step; no current subcommand calls it")]
fn inside_all(ws: &[(Vec<f32>, Vec<bool>)]) -> Option<usize> {
    ws.iter().filter_map(|(s, g)| gold_place(s, g)).min()
}

/// Counts of one variant over questions: first inside, best 6 inside, first across.
#[derive(Clone, Copy, Default)]
#[expect(dead_code, reason = "kept from the DX diagnostics for the next read step; no current subcommand calls it")]
struct Wide {
    n: usize,
    first: usize,
    six: usize,
    across: usize,
    secs: f64,
}
#[expect(dead_code, reason = "kept from the DX diagnostics for the next read step; no current subcommand calls it")]
impl Wide {
    fn add(&mut self, ws: &[(Vec<f32>, Vec<bool>)], secs: f64) {
        self.n += 1;
        let p = inside_all(ws);
        self.first += usize::from(p == Some(0));
        self.six += usize::from(p.is_some_and(|x| x < 6));
        self.across += usize::from(across_all(ws) == Some(0));
        self.secs += secs;
    }
    fn line(&self, name: &str) -> String {
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let s = self.secs / self.n.max(1) as f64;
        format!("| {name} | {}/{} | {}/{} | {}/{} | {s:.2} |", self.first, self.n, self.six, self.n, self.across, self.n)
    }
}

/// The T tests (2026-11-10, redone short on the direction of Luis Jake Gabriel III that an isolated
/// test finishes in 2 to 3 minutes, 5 at most): one test per command, the window holding the right
/// chunk of each question of the subset, encodings held in RAM only (never written to disk), the
/// record written after every question with the elapsed time. `t1a` shared question, `t1` final
/// short read over the gold window's best 2 and every other window's best 2 by the dump's exact
/// scores (`t1b` the same union re-encoded together), `t4` boundary positions and temperature, `t5`
/// order averaging, `t6` span reading (without and with global context per span). First across
/// windows compares the gold window's scores with the dump's scores of the other windows.
#[expect(clippy::too_many_lines, reason = "one test, in the order it runs")]
fn tbattery(home: &Path, names: &[String], dir: &Path, stamp: u64, limit: usize) -> Result<Vec<(String, String)>, Fault> {
    let name = names.first().cloned().unwrap_or_default();
    let items: Vec<Item> = crate::e2::suite(home)?;
    let (sub, mut cs) = cases(home, dir)?;
    cs.truncate(limit);
    let set = tentacle_pass::drones(home);
    let started = Instant::now();
    let r = Reader::open(home, nmapi::neural::POPULATION, set.last_layer(), false)?;
    let d = r.m.pop.dims;
    let group = d.heads / d.kv_heads.max(1);
    let anchor = Arc::clone(r.aw.chunks.first().ok_or_else(|| bad("no anchor"))?);
    let head = format!(
        "dump {}; {} development questions ({}); the window holding the right chunk; encodings held in RAM for the run (never written to disk); baseline: exact read of the whole window, default drone set, calibrated score; first across windows against the dump's exact scores of the other windows",
        dir.display(),
        cs.len(),
        cs.iter().map(|c| format!("{} ({})", c.idx, c.id)).collect::<Vec<_>>().join(", ")
    );
    let (span_derived, span_why) = span_from_d1(home);
    let mut spans_used: Vec<usize> = vec![1_500, 3_000, span_derived];
    spans_used.sort_unstable();
    spans_used.dedup();
    let mut base = Pair::default();
    let mut vars: BTreeMap<String, Pair> = BTreeMap::new();
    let mut secs: BTreeMap<String, f64> = BTreeMap::new();
    let mut lines: Vec<String> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    let mut tokens_read = (0_usize, 0_usize);
    let all: Vec<usize> = (0..set.heads.len()).collect();
    let mut recs = Vec::new();
    for c in &cs {
        let tq = Instant::now();
        let x = items.get(c.idx).ok_or_else(|| bad("question index"))?;
        let posed = format!("\nQuestion: {}", c.text);
        let docs: Vec<String> = c.place.iter().map(|p| doc(&sub, *p)).collect();
        let kv = r.encode(&docs)?;
        let mut parts = vec![Arc::clone(&anchor)];
        parts.extend(kv.iter().cloned());
        let w = Arc::new(Window::build(&r.m, &parts)?);
        let heads = set.heads.clone();
        let tb0 = Instant::now();
        let (pl, pt) = read_raw(&r, &w, &heads, &posed)?;
        let (nl, nt) = read_raw(&r, &w, &heads, NULL)?;
        *secs.entry("baseline".into()).or_default() += tb0.elapsed().as_secs_f64();
        let tp = table_of(&pl, pt, w.len, &w.parts, &|_, _| true, 1.0);
        let tn = table_of(&nl, nt, w.len, &w.parts, &|_, _| true, 1.0);
        let tab =
            |p: Vec<f32>, n: Vec<f32>| Table { heads: heads.clone(), parts: w.parts.len(), cp: p, cn: n, mp: Vec::new(), mn: Vec::new() };
        let base_t = tab(tp.clone(), tn.clone());
        base.add(&base_t, &all, &c.gold, Some(&c.others));
        let base_s = base_t.scores(&all).0;
        let add = |k: String,
                   s: &[f32],
                   gold: &[bool],
                   t: f64,
                   across: bool,
                   vars: &mut BTreeMap<String, Pair>,
                   secs: &mut BTreeMap<String, f64>| {
            let e = vars.entry(k.clone()).or_default();
            e.cal.add(gold_place(s, gold), across.then(|| self::across(s, gold, &c.others, true)));
            *secs.entry(k).or_default() += t;
        };
        match name.as_str() {
            "t1a" => {
                let ids = r.m.pop.tokenizer.encode(&posed).map_err(|e| bad(e.to_string()))?;
                let nids = r.m.pop.tokenizer.encode(NULL).map_err(|e| bad(e.to_string()))?;
                let t = Instant::now();
                let eq = nmengn::drone::question_path(&r.m, r.dev, &w, &ids, w.len, &heads)?;
                let aq = nmengn::drone::question_path(&r.m, r.dev, &r.aw, &ids, w.len, &heads)?;
                let an = nmengn::drone::question_path(&r.m, r.dev, &r.aw, &nids, w.len, &heads)?;
                let cos: Vec<f64> = eq.q.iter().zip(&aq.q).map(|(a, b)| cosine(a, b)).collect();
                let packets: Vec<Arc<nmengn::drone::DroneKeys>> = parts
                    .iter()
                    .map(|p| {
                        nmengn::drone::DroneKeys::from_chunk(p, &nmengn::drone::pairs(&heads, group), d.kv_heads, d.head_dim).map(Arc::new)
                    })
                    .collect::<Result<_, _>>()?;
                let job = nmapi::key_scan::Job { packets, parts: w.parts.clone(), len: w.len, posed: aq, null: an };
                let (rows, _) = nmapi::key_scan::score_windows(std::slice::from_ref(&job), d.head_dim, group, d.theta);
                let (kp, kn) = rows.into_iter().next().ok_or_else(|| bad("no rows"))??;
                let s = tab(kp, kn).scores(&all).0;
                add(
                    "shared queries (question over the anchor only)".into(),
                    &s,
                    &c.gold,
                    t.elapsed().as_secs_f64(),
                    true,
                    &mut vars,
                    &mut secs,
                );
                #[expect(clippy::cast_precision_loss, reason = "counts")]
                let m = cos.iter().sum::<f64>() / cos.len().max(1) as f64;
                notes.push(format!("{}: query cosine exact against anchor-only, mean {m:.3}", c.id));
            }
            "t4" => {
                let toks: Vec<Vec<bool>> = docs
                    .iter()
                    .map(|dt| {
                        r.m.pop
                            .tokenizer
                            .encode(dt)
                            .unwrap_or_default()
                            .iter()
                            .map(|id| r.m.pop.tokenizer.decode(&[*id]).contains('\n'))
                            .collect()
                    })
                    .collect();
                for k0 in [1_usize, 2, 4, 8] {
                    let keep = |part: usize, off: usize| part == 0 || off >= k0;
                    let s =
                        tab(table_of(&pl, pt, w.len, &w.parts, &keep, 1.0), table_of(&nl, nt, w.len, &w.parts, &keep, 1.0)).scores(&all).0;
                    add(format!("first {k0} tokens of each chunk left out"), &s, &c.gold, 0.0, true, &mut vars, &mut secs);
                }
                let keep = |part: usize, off: usize| part == 0 || !toks.get(part - 1).and_then(|v| v.get(off)).copied().unwrap_or(false);
                let s = tab(table_of(&pl, pt, w.len, &w.parts, &keep, 1.0), table_of(&nl, nt, w.len, &w.parts, &keep, 1.0)).scores(&all).0;
                add("newline tokens left out".into(), &s, &c.gold, 0.0, true, &mut vars, &mut secs);
                for temp in [0.5_f32, 0.75] {
                    let s =
                        tab(table_of(&pl, pt, w.len, &w.parts, &|_, _| true, temp), table_of(&nl, nt, w.len, &w.parts, &|_, _| true, temp))
                            .scores(&all)
                            .0;
                    add(format!("temperature {temp} on the drone heads' scores"), &s, &c.gold, 0.0, true, &mut vars, &mut secs);
                }
                notes.push(format!("{}: content-free entropy over the window {:.2} nats", c.id, entropy(&nl, nt, w.len)));
            }
            "t5" => {
                let t = Instant::now();
                let mut sum = base_s.clone();
                for seed in 1..=2_u64 {
                    let order = shuffled(kv.len(), seed);
                    let mut p2 = vec![Arc::clone(&anchor)];
                    p2.extend(order.iter().filter_map(|i| kv.get(*i).cloned()));
                    let w2 = Arc::new(Window::build(&r.m, &p2)?);
                    let (a2, at2) = read_raw(&r, &w2, &heads, &posed)?;
                    let (b2, bt2) = read_raw(&r, &w2, &heads, NULL)?;
                    let s2 = tab(
                        table_of(&a2, at2, w2.len, &w2.parts, &|_, _| true, 1.0),
                        table_of(&b2, bt2, w2.len, &w2.parts, &|_, _| true, 1.0),
                    )
                    .scores(&all)
                    .0;
                    w2.release_device();
                    for (j, i) in order.iter().enumerate() {
                        if let (Some(x), Some(y)) = (sum.get_mut(*i), s2.get(j)) {
                            *x += y;
                        }
                    }
                    if seed == 1 {
                        add("2 orders averaged".into(), &sum, &c.gold, t.elapsed().as_secs_f64(), true, &mut vars, &mut secs);
                    }
                }
                add("3 orders averaged".into(), &sum, &c.gold, t.elapsed().as_secs_f64(), true, &mut vars, &mut secs);
            }
            "t6" => {
                // The span's place in the whole: crates of the tree, the window's files, and for each
                // span its files, items and neighbours (encoded once per text, cached).
                let crates: Vec<String> = sub.crates.iter().map(|k| k.name.clone()).collect();
                let crates = if crates.is_empty() {
                    sub.files
                        .iter()
                        .filter_map(|f| f.path.split('/').nth(1).map(str::to_owned))
                        .collect::<std::collections::BTreeSet<_>>()
                        .into_iter()
                        .collect()
                } else {
                    crates
                };
                let label = |i: usize| {
                    c.place.get(i).and_then(|p| sub.chunks.get(*p)).map_or_else(String::new, |ch| {
                        format!("{}:{}-{}", sub.files.get(ch.file).map_or("", |f| f.path.as_str()), ch.start, ch.end)
                    })
                };
                let items_of = |i: usize| -> String {
                    let Some(ch) = c.place.get(i).and_then(|p| sub.chunks.get(*p)) else { return String::new() };
                    let path = sub.files.get(ch.file).map_or("", |f| f.path.as_str());
                    sub.symbols
                        .iter()
                        .filter(|s| s.file.replace('\\', "/") == path && s.line >= ch.start && s.line <= ch.end)
                        .map(|s| s.name.clone())
                        .take(6)
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                let ctx = |g: &[usize]| -> Result<Arc<ChunkKv>, Fault> {
                    let first = g.first().copied().unwrap_or(0);
                    let last = g.last().copied().unwrap_or(0);
                    let text = format!(
                        "Code base crates: {}. This part of the window: {}; items {}; before it {}; after it {}.\n",
                        crates.join(", "),
                        g.iter().map(|i| label(*i)).collect::<Vec<_>>().join(", "),
                        g.iter().map(|i| items_of(*i)).filter(|s| !s.is_empty()).collect::<Vec<_>>().join("; "),
                        if first == 0 { "nothing (window start)".to_owned() } else { label(first - 1) },
                        if last + 1 >= kv.len() { "nothing (window end)".to_owned() } else { label(last + 1) },
                    );
                    Ok(r.encode(&[text])?.into_iter().next().unwrap_or_default())
                };
                for &span in &spans_used {
                    for with in [false, true] {
                        let t = Instant::now();
                        let (s, n_tok) = span_read(
                            &r,
                            &anchor,
                            &kv,
                            &heads,
                            &posed,
                            span,
                            with.then_some(&ctx as &dyn Fn(&[usize]) -> Result<Arc<ChunkKv>, Fault>),
                        )?;
                        let k = format!("spans of about {span} tokens{}", if with { ", global context per span" } else { "" });
                        add(k, &s, &c.gold, t.elapsed().as_secs_f64(), false, &mut vars, &mut secs);
                        if span == span_derived && !with {
                            tokens_read.0 += n_tok;
                            tokens_read.1 += kv.iter().map(|x| x.tokens).sum::<usize>();
                        }
                    }
                }
            }
            "t1" | "t1b" => {
                // Union: the gold window's best 2 by the baseline read and every other window's best 2
                // by the dump's exact scores.
                let (_, scan) = scan_dump::load_question(dir, c.idx, &x.q)?;
                let mut union: Vec<(usize, String, bool)> = Vec::new();
                let mut idx: Vec<usize> = (0..base_s.len()).collect();
                idx.sort_by(|a, b| {
                    base_s.get(*b).copied().unwrap_or(f32::NEG_INFINITY).total_cmp(&base_s.get(*a).copied().unwrap_or(f32::NEG_INFINITY))
                });
                for i in idx.into_iter().take(2) {
                    if let (Some(p), Some(g)) = (c.place.get(i), c.gold.get(i)) {
                        union.push((*p, doc(&sub, *p), *g));
                    }
                }
                for ow in scan.windows.iter().filter(|ow| ow.place != c.place) {
                    let cal = calibrated(&ow.cal_posed, &ow.cal_null, ow.parts.len());
                    let mut o: Vec<usize> = (0..ow.place.len()).collect();
                    o.sort_by(|a, b| {
                        cal.get(b + 1)
                            .copied()
                            .unwrap_or(f32::NEG_INFINITY)
                            .total_cmp(&cal.get(a + 1).copied().unwrap_or(f32::NEG_INFINITY))
                    });
                    for i in o.into_iter().take(2) {
                        if let Some(p) = ow.place.get(i) {
                            union.push((
                                *p,
                                doc(&sub, *p),
                                c.place.contains(p)
                                    && c.gold.get(c.place.iter().position(|q| q == p).unwrap_or(0)).copied().unwrap_or(false)
                                    || scan_dump_gold(dir, c.idx, *p),
                            ));
                        }
                    }
                }
                union.sort_by_key(|u| path_key(&sub, u.0));
                let gold: Vec<bool> = union.iter().map(|u| u.2).collect();
                let udocs: Vec<String> = union.iter().map(|u| u.1.clone()).collect();
                let t = Instant::now();
                if name == "t1" {
                    let ukv = r.encode(&udocs)?;
                    let mut p = vec![Arc::clone(&anchor)];
                    p.extend(ukv);
                    let wu = Arc::new(Window::build(&r.m, &p)?);
                    let (a, at) = read_raw(&r, &wu, set.verification_set(), &posed)?;
                    let (b, bt) = read_raw(&r, &wu, set.verification_set(), NULL)?;
                    let s = above_anchor(
                        &table_of(&a, at, wu.len, &wu.parts, &|_, _| true, 1.0),
                        &table_of(&b, bt, wu.len, &wu.parts, &|_, _| true, 1.0),
                        wu.parts.len(),
                    );
                    wu.release_device();
                    add(
                        format!("union of the best 2 per window ({} chunks), one short read, kept encodings", union.len()),
                        &s,
                        &gold,
                        t.elapsed().as_secs_f64(),
                        false,
                        &mut vars,
                        &mut secs,
                    );
                } else {
                    let mut all_ids: Vec<i32> = Vec::new();
                    let mut spans = vec![(0_usize, anchor.tokens)];
                    for u in &udocs {
                        let ids = r.m.pop.tokenizer.encode(u).map_err(|e| bad(e.to_string()))?;
                        spans.push((anchor.tokens + all_ids.len(), ids.len()));
                        all_ids.extend(ids);
                    }
                    let joint = Arc::new(kvstore::encode_through(&r.m, r.dev, Some(&r.aw), &all_ids, r.last)?);
                    let wj = Arc::new(Window::build(&r.m, &[Arc::clone(&anchor), joint])?);
                    let (a, at) = read_raw(&r, &wj, set.verification_set(), &posed)?;
                    let (b, bt) = read_raw(&r, &wj, set.verification_set(), NULL)?;
                    let s = above_anchor(
                        &table_of(&a, at, wj.len, &spans, &|_, _| true, 1.0),
                        &table_of(&b, bt, wj.len, &spans, &|_, _| true, 1.0),
                        spans.len(),
                    );
                    wj.release_device();
                    add(
                        format!("union of the best 2 per window ({} chunks), re-encoded together", union.len()),
                        &s,
                        &gold,
                        t.elapsed().as_secs_f64(),
                        false,
                        &mut vars,
                        &mut secs,
                    );
                }
            }
            other => return Err(bad(format!("unknown test {other}"))),
        }
        w.release_device();
        nmengn::residency::get().release_windows();
        lines.push(format!(
            "q{} ({}): baseline place {:?}; {:.1} s",
            c.idx,
            c.id,
            gold_place(&base_s, &c.gold),
            tq.elapsed().as_secs_f64()
        ));
        // The record so far.
        let n = base.cal.n;
        let mut t = format!(
            "DX {name}: {}\n{head}\n| variant | right chunk first inside its window | in the best 6 inside | first across windows | seconds per window |\n|---|---|---|---|---|\n",
            match name.as_str() {
                "t1a" =>
                    "shared question (the window scored with the queries of the question posed over the anchor only, keys from the kept encodings)",
                "t4" =>
                    "boundary positions left out of the score, and temperature on the drone heads' scores (posed and content-free alike)",
                "t5" =>
                    "order averaging (the window read again with its chunks in seeded shuffled orders, keys roped on read, no re-encoding)",
                "t6" =>
                    "span reading: the whole window is read, as consecutive spans of kept encodings, each span its own short read behind the anchor (optionally with a global context part: the code base's crates, the span's files, items and neighbours), the best 2 of each span in one final short read; nothing dropped",
                _ =>
                    "final short read over the union of the best 2 chunks of every window (the gold window by the baseline read, the others by the dump's exact scores)",
            }
        );
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let per = |k: &str| secs.get(k).copied().unwrap_or(0.0) / n.max(1) as f64;
        let _ = writeln!(
            t,
            "| baseline (whole window, one exact read) | {}/{n} | {}/{n} | {} | {:.2} |",
            base.cal.first,
            base.cal.six,
            base.cal.across.map_or_else(|| "-".to_owned(), |a| format!("{a}/{n}")),
            per("baseline")
        );
        for (k, v) in &vars {
            let _ = writeln!(
                t,
                "| {k} | {}/{n} | {}/{n} | {} | {:.2} |",
                v.cal.first,
                v.cal.six,
                v.cal.across.map_or_else(|| "not comparable".to_owned(), |a| format!("{a}/{n}")),
                per(k)
            );
        }
        if name == "t6" {
            let _ = writeln!(
                t,
                "span size derived from the D1 record: {span_why}; sizes used {spans_used:?}; tokens read by the span reads at the derived size {} of {} tokens held in the windows (every token read)",
                tokens_read.0, tokens_read.1
            );
        }
        for l in lines.iter().chain(&notes) {
            let _ = writeln!(t, "{l}");
        }
        let ram =
            nmffi::dynlib::process_memory().map_or_else(|e| e, |(peak, now)| format!("RAM peak {} MB, now {} MB", peak >> 20, now >> 20));
        let _ = writeln!(t, "\nrun {stamp}: {} of {} questions, elapsed {:.0} s; {ram}", n, cs.len(), started.elapsed().as_secs_f64());
        let _ = std::fs::write(home.join("runs").join(format!("DX-{name}-{stamp}.txt")), &t);
        recs = vec![(name.clone(), t)];
    }
    Ok(recs)
}

/// True when chunk `p` is a gold chunk of question `idx` in the dump.
fn scan_dump_gold(dir: &Path, idx: usize, p: usize) -> bool {
    let jp = dir.join(format!("{}.json", scan_dump::stem(idx)));
    std::fs::read(&jp)
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .and_then(|v| v.get("gold").ok().and_then(|g| g.array().ok()).map(|a| a.iter().any(|x| x.integer().ok() == u64::try_from(p).ok())))
        .unwrap_or(false)
}

/// Chunks of each span that go to the final short read (T6).
const SPAN_KEEP: usize = 2;

/// T6 span size from the newest D1 record: the largest cut whose calibrated first-inside count is at
/// least two thirds of the best cut's (the length up to which the read holds most of its short-window
/// accuracy); fallback 1,500 tokens (the shortest cut measured) when no D1 record is found.
fn span_from_d1(home: &Path) -> (usize, String) {
    let mut files: Vec<PathBuf> = std::fs::read_dir(home.join("runs"))
        .map(|rd| {
            rd.filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("DX-d1-")))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    let Some(f) = files.pop() else { return (1_500, "no D1 record: fallback 1,500 tokens".into()) };
    let text = std::fs::read_to_string(&f).unwrap_or_default();
    let mut cuts: Vec<(usize, usize, usize)> = Vec::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("| cut ") else { continue };
        let cut = rest.split_whitespace().next().and_then(|x| x.parse::<usize>().ok());
        let first = rest.split("first ").nth(1).and_then(|x| x.split('/').next()).and_then(|x| x.parse::<usize>().ok());
        let of = rest
            .split("first ")
            .nth(1)
            .and_then(|x| x.split('/').nth(1))
            .and_then(|x| x.split(',').next())
            .and_then(|x| x.parse::<usize>().ok());
        if let (Some(c), Some(a), Some(b)) = (cut, first, of) {
            cuts.push((c, a, b));
        }
    }
    let best = cuts.iter().map(|x| x.1).max().unwrap_or(0);
    let pick = cuts.iter().filter(|x| 3 * x.1 >= 2 * best).map(|x| x.0).max();
    match pick {
        Some(c) => (
            c,
            format!(
                "{}: calibrated first inside by cut {}; best {best}; largest cut with at least two thirds of it: {c} tokens",
                f.file_name().and_then(|n| n.to_str()).unwrap_or(""),
                cuts.iter().map(|x| format!("{} {}/{}", x.0, x.1, x.2)).collect::<Vec<_>>().join(", ")
            ),
        ),
        None => (1_500, format!("{}: no cut lines parsed: fallback 1,500 tokens", f.display())),
    }
}

/// T6 span read of one window: consecutive spans of its chunks of at most `span` tokens (a longer
/// chunk alone), each read as its own short window behind the anchor (posed and content-free); the
/// best `SPAN_KEEP` of each span read again together in window order. Returns a score per chunk
/// whose order is: winners by the final read, then the others by their span score; and the reads.
#[expect(clippy::too_many_arguments, clippy::type_complexity, reason = "one DX read step; the inputs are the read's separate parts")]
fn span_read(
    r: &Reader,
    anchor: &Arc<ChunkKv>,
    kv: &[Arc<ChunkKv>],
    heads: &[(usize, usize)],
    posed: &str,
    span: usize,
    ctx: Option<&dyn Fn(&[usize]) -> Result<Arc<ChunkKv>, Fault>>,
) -> Result<(Vec<f32>, usize), Fault> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut used = 0;
    for (i, k) in kv.iter().enumerate() {
        if !cur.is_empty() && used + k.tokens > span {
            groups.push(std::mem::take(&mut cur));
            used = 0;
        }
        cur.push(i);
        used += k.tokens;
    }
    if !cur.is_empty() {
        groups.push(cur);
    }
    let tokens_read = std::cell::Cell::new(0_usize);
    let read = |idx: &[usize]| -> Result<Vec<f32>, Fault> {
        let mut p = vec![Arc::clone(anchor)];
        let extra = match ctx {
            Some(f) => {
                p.push(f(idx)?);
                1
            }
            None => 0,
        };
        p.extend(idx.iter().filter_map(|i| kv.get(*i).cloned()));
        let w = Arc::new(Window::build(&r.m, &p)?);
        tokens_read.set(tokens_read.get() + idx.iter().filter_map(|i| kv.get(*i)).map(|k| k.tokens).sum::<usize>());
        let (a, at) = read_raw(r, &w, heads, posed)?;
        let (b, bt) = read_raw(r, &w, heads, NULL)?;
        w.release_device();
        let s = above_anchor(
            &table_of(&a, at, w.len, &w.parts, &|_, _| true, 1.0),
            &table_of(&b, bt, w.len, &w.parts, &|_, _| true, 1.0),
            w.parts.len(),
        );
        Ok(s.into_iter().skip(extra).collect())
    };
    let mut winners: Vec<usize> = Vec::new();
    let mut losers: Vec<(usize, f32)> = Vec::new();
    for g in &groups {
        let s = read(g)?;
        let mut idx: Vec<usize> = (0..g.len()).collect();
        idx.sort_by(|a, b| {
            s.get(*b).copied().unwrap_or(f32::NEG_INFINITY).total_cmp(&s.get(*a).copied().unwrap_or(f32::NEG_INFINITY)).then(a.cmp(b))
        });
        for (rank, j) in idx.into_iter().enumerate() {
            if let Some(&c) = g.get(j) {
                if rank < SPAN_KEEP {
                    winners.push(c);
                } else {
                    losers.push((c, s.get(j).copied().unwrap_or(f32::NEG_INFINITY)));
                }
            }
        }
    }
    winners.sort_unstable();
    let before = tokens_read.get();
    let fs = read(&winners)?;
    let fs_tokens = tokens_read.get() - before;
    // Order: winners by the final read, then the others by their span score.
    let mut order: Vec<usize> = (0..winners.len()).collect();
    order.sort_by(|a, b| {
        fs.get(*b).copied().unwrap_or(f32::NEG_INFINITY).total_cmp(&fs.get(*a).copied().unwrap_or(f32::NEG_INFINITY)).then(a.cmp(b))
    });
    losers.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let ranked: Vec<usize> = order.into_iter().filter_map(|k| winners.get(k).copied()).chain(losers.into_iter().map(|x| x.0)).collect();
    let mut scores = vec![f32::NEG_INFINITY; kv.len()];
    #[expect(clippy::cast_precision_loss, reason = "ranks are small")]
    for (rank, c) in ranked.iter().enumerate() {
        if let Some(x) = scores.get_mut(*c) {
            *x = -(rank as f32);
        }
    }
    // Tokens of the window read in the span reads (the final read re-reads winners, not counted).
    let _ = winners.len();
    Ok((scores, tokens_read.get().saturating_sub(fs_tokens)))
}

/// Window order key of a chunk (file, then first line), as windows are laid out.
/// Place of the best gold chunk in an explicit order of the window's chunks.
fn place_in(order: &[usize], gold: &[bool]) -> Option<usize> {
    order.iter().position(|i| gold.get(*i).copied().unwrap_or(false))
}

/// One arm of `r2a`: its tally and the seconds of its reads and scoring, summed over windows.
#[derive(Clone, Default)]
struct Arm {
    name: String,
    t: Tally,
    secs: f64,
}

/// `run dx r2a [--n k]` (2026-11-10, refinement step 2a; components `nmapi::core_read`): the read of
/// the tentacle core, five arms in one process on the window holding the right chunk of each of the
/// fixed development questions. Per question the window's chunks are encoded once and held in RAM
/// (nothing encoded is written to disk), every arm is read and scored, the question's line is
/// written and the encodings are released. Arms: (a) today's core read, the calibrated score of
/// the full read; (b) the mass score of the full read; (c) b plus the second look (shortlist by
/// the full read's mass, size from `core_read::look_size`, one short read from the held encodings,
/// first place by the short read's mass, the rest of the window in the full read's order); (d) c
/// with the short read scored by attention-weighted votes, low-signal positions of the
/// content-free read left out; (e) d with the length-matched content-free pose on the full read
/// and the short read. Reference rows (not arms): the same with the stated fallback shortlist of
/// 6, the short read scored by the calibrated score, and the full read under each component
/// alone. Measures: right chunk first inside its window, in the best 6, seconds per window of the
/// arm's reads and scoring. Only the window holding the right chunk is read (the measures are
/// in-window), which keeps the run to minutes.
#[expect(clippy::too_many_lines, reason = "one test, in the order it runs")]
fn r2a(home: &Path, dir: &Path, stamp: u64, limit: usize) -> Result<Vec<(String, String)>, Fault> {
    use nmapi::core_read::{self, Kind, RawRead};
    let (sub, mut cs) = cases(home, dir)?;
    cs.truncate(limit);
    let set = tentacle_pass::drones(home);
    let started = Instant::now();
    let r = Reader::open(home, nmapi::neural::POPULATION, set.last_layer(), false)?;
    let open_s = started.elapsed().as_secs_f64();
    let anchor = Arc::clone(r.aw.chunks.first().ok_or_else(|| bad("no anchor"))?);
    let heads = set.heads.clone();
    let verify = set.verification_set().to_vec();
    let count = |s: &str| r.m.pop.tokenizer.encode(s).ok().map(|v| v.len());
    let names = [
        "a: full read, calibrated score (today's core read)",
        "b: full read, mass score",
        "c: b + second look (derived shortlist, mass in the look)",
        "d: c + attention-weighted votes, low-signal positions left out",
        "e: d + length-matched content-free pose",
        "ref: c with the calibrated score in the look",
        "ref: c with shortlist 6 (stated fallback)",
        "ref: d with shortlist 6",
        "ref: e with shortlist 6",
        "ref: full read, attention-weighted votes alone",
        "ref: full read, weighted votes + low-signal positions left out",
        "ref: full read, mass + low-signal positions left out",
        "ref: full read, calibrated with the length-matched pose",
        "ref: full read, mass with the length-matched pose",
    ];
    let mut arms: Vec<Arm> = names.iter().map(|n| Arm { name: (*n).to_owned(), ..Arm::default() }).collect();
    let mut lines: Vec<String> = Vec::new();
    let (mut enc_s, mut enc_tokens, mut windows) = (0.0_f64, 0_usize, 0_usize);
    let (mut k_sum, mut k_inside, mut six_inside, mut tok_look, mut tok_six) = (0_usize, 0_usize, 0_usize, 0_usize, 0_usize);
    let (mut mask_out, mut mask_all, mut nl_masked, mut nl_all, mut masked_any, mut masked_nl) =
        (0_usize, 0_usize, 0_usize, 0_usize, 0_usize, 0_usize);
    let mut poses: Vec<String> = Vec::new();
    for c in &cs {
        let tq = Instant::now();
        let posed = format!("\nQuestion: {}", c.text);
        let plain = core_read::plain_null("");
        let (matched, mn, mwhy) = core_read::matched_null(&count, &posed, "");
        poses.push(format!("{}: {mwhy}", c.id));
        let docs: Vec<String> = c.place.iter().map(|p| doc(&sub, *p)).collect();
        let te = Instant::now();
        let kv = r.encode(&docs)?;
        enc_s += te.elapsed().as_secs_f64();
        enc_tokens += kv.iter().map(|k| k.tokens).sum::<usize>();
        windows += 1;
        let mut parts = vec![Arc::clone(&anchor)];
        parts.extend(kv.iter().cloned());
        let w = Arc::new(Window::build(&r.m, &parts)?);
        let timed = |win: &Arc<Window>, hs: &[(usize, usize)], text: &str| -> Result<(RawRead, f64), Fault> {
            let t = Instant::now();
            let x = core_read::raw_read(&r.m, r.dev, win, &r.stop, hs, text)?;
            Ok((x, t.elapsed().as_secs_f64()))
        };
        let scored = |f: &dyn Fn() -> Vec<f32>| -> (Vec<f32>, f64) {
            let t = Instant::now();
            let v = f();
            (v, t.elapsed().as_secs_f64())
        };
        // The full window, read first by every arm.
        let (fp, t_fp) = timed(&w, &heads, &posed)?;
        let (fnl, t_fn) = timed(&w, &heads, &plain)?;
        let (fm, t_fm) = timed(&w, &heads, &matched)?;
        w.release_device();
        let wp = w.parts.clone();
        let (cal, s_cal) = scored(&|| core_read::chunk_scores(&fp, &fnl, &wp, Kind::Calibrated, None));
        let (mass, s_mass) = scored(&|| core_read::chunk_scores(&fp, &fnl, &wp, Kind::Mass, None));
        let (mass_m, s_mass_m) = scored(&|| core_read::chunk_scores(&fp, &fm, &wp, Kind::Mass, None));
        let mut places: Vec<Option<usize>> = Vec::new();
        let add = |i: usize, place: Option<usize>, secs: f64, arms: &mut Vec<Arm>, places: &mut Vec<Option<usize>>| {
            if let Some(a) = arms.get_mut(i) {
                a.t.add(place, None);
                a.secs += secs;
            }
            places.push(place);
        };
        add(0, gold_place(&cal, &c.gold), t_fp + t_fn + s_cal, &mut arms, &mut places);
        add(1, gold_place(&mass, &c.gold), t_fp + t_fn + s_mass, &mut arms, &mut places);
        // Low-signal positions of the full window (content-free read), and what they are.
        let tm = Instant::now();
        let mask = core_read::low_signal(&fnl, &wp);
        let s_mask = tm.elapsed().as_secs_f64();
        mask_out += mask.left_out;
        mask_all += mask.considered;
        for (k, dt) in docs.iter().enumerate() {
            let Some(&(at, n)) = wp.get(k + 1) else { continue };
            let ids = r.m.pop.tokenizer.encode(dt).unwrap_or_default();
            for (j, id) in ids.iter().enumerate().take(n) {
                let nl = r.m.pop.tokenizer.decode(&[*id]).trim().is_empty();
                let out = mask.any_head(at + j);
                nl_all += usize::from(nl);
                nl_masked += usize::from(nl && out);
                masked_any += usize::from(out);
                masked_nl += usize::from(nl && out);
            }
        }
        // The second look, for one shortlist size and one content-free pose.
        type LookOut = (Option<usize>, Option<usize>, Option<usize>, f64, f64, bool, usize);
        let look = |k: usize, full_mass: &[f32], null: &str| -> Result<LookOut, Fault> {
            let pick = core_read::look_pick(full_mass, k);
            let mut p = vec![Arc::clone(&anchor)];
            p.extend(pick.iter().filter_map(|i| kv.get(*i).cloned()));
            let sw = Arc::new(Window::build(&r.m, &p)?);
            let (sp, t1) = timed(&sw, &verify, &posed)?;
            let (sn, t2) = timed(&sw, &verify, null)?;
            sw.release_device();
            let t = Instant::now();
            let by_mass = core_read::chunk_scores(&sp, &sn, &sw.parts, Kind::Mass, None);
            let t_mass = t.elapsed().as_secs_f64();
            let by_cal = core_read::chunk_scores(&sp, &sn, &sw.parts, Kind::Calibrated, None);
            let t = Instant::now();
            let m = core_read::low_signal(&sn, &sw.parts);
            let by_votes = core_read::chunk_scores(&sp, &sn, &sw.parts, Kind::Weighted, Some(&m));
            let t_votes = t.elapsed().as_secs_f64();
            let place = |s: &[f32]| place_in(&core_read::look_order(full_mass, &pick, s), &c.gold);
            let inside = pick.iter().any(|i| c.gold.get(*i).copied().unwrap_or(false));
            Ok((place(&by_mass), place(&by_cal), place(&by_votes), t1 + t2 + t_mass, t1 + t2 + t_votes, inside, sw.len))
        };
        let (k, kwhy) = core_read::look_size(&mass);
        let (km, _) = core_read::look_size(&mass_m);
        let six = core_read::LOOK_FALLBACK.min(mass.len());
        let d = look(k, &mass, &plain)?;
        let e = look(km, &mass_m, &matched)?;
        let d6 = look(six, &mass, &plain)?;
        let e6 = look(six, &mass_m, &matched)?;
        k_sum += k;
        k_inside += usize::from(d.5);
        six_inside += usize::from(d6.5);
        tok_look += d.6;
        tok_six += d6.6;
        let full_b = t_fp + t_fn + s_mass;
        let full_e = t_fp + t_fm + s_mass_m;
        add(2, d.0, full_b + d.3, &mut arms, &mut places);
        add(3, d.2, full_b + d.4, &mut arms, &mut places);
        add(4, e.2, full_e + e.4, &mut arms, &mut places);
        add(5, d.1, full_b + d.3, &mut arms, &mut places);
        add(6, d6.0, full_b + d6.3, &mut arms, &mut places);
        add(7, d6.2, full_b + d6.4, &mut arms, &mut places);
        add(8, e6.2, full_e + e6.4, &mut arms, &mut places);
        // Each component alone on the full read (processor only, from the kept raw scores).
        let (wv, s_wv) = scored(&|| core_read::chunk_scores(&fp, &fnl, &wp, Kind::Weighted, None));
        let (wvm, s_wvm) = scored(&|| core_read::chunk_scores(&fp, &fnl, &wp, Kind::Weighted, Some(&mask)));
        let (mm, s_mm) = scored(&|| core_read::chunk_scores(&fp, &fnl, &wp, Kind::Mass, Some(&mask)));
        let (calm, s_calm) = scored(&|| core_read::chunk_scores(&fp, &fm, &wp, Kind::Calibrated, None));
        add(9, gold_place(&wv, &c.gold), t_fp + t_fn + s_wv, &mut arms, &mut places);
        add(10, gold_place(&wvm, &c.gold), t_fp + t_fn + s_mask + s_wvm, &mut arms, &mut places);
        add(11, gold_place(&mm, &c.gold), t_fp + t_fn + s_mask + s_mm, &mut arms, &mut places);
        add(12, gold_place(&calm, &c.gold), t_fp + t_fm + s_calm, &mut arms, &mut places);
        add(13, gold_place(&mass_m, &c.gold), full_e, &mut arms, &mut places);
        let show = |p: &Option<usize>| p.map_or_else(|| "-".to_owned(), |x| (x + 1).to_string());
        let line = format!(
            "q{} ({}): {} chunks, {} tokens, question {} tokens (content-free {} matched {mn}); shortlist {k} ({kwhy}), right chunk inside {}, short window {} tokens; with 6: inside {}, {} tokens; place of the right chunk per row: {}; raw scores held {} MB; {:.1} s",
            c.idx + 1,
            c.id,
            c.place.len(),
            w.len,
            fp.tokens,
            fnl.tokens,
            d.5,
            d.6,
            d6.5,
            d6.6,
            places.iter().map(show).collect::<Vec<_>>().join(" "),
            (fp.bytes() + fnl.bytes() + fm.bytes()) >> 20,
            tq.elapsed().as_secs_f64()
        );
        progress(home, stamp, &line);
        lines.push(line);
        drop((fp, fnl, fm, kv, parts));
        nmengn::residency::get().release_windows();
    }
    let n = cs.len().max(1);
    let mut out = String::new();
    let _ = writeln!(
        out,
        "DX r2a (refinement step 2a: second look, token votes, length-matched content-free pose; components nmapi::core_read)"
    );
    let _ = writeln!(
        out,
        "dump {}; {} development questions by index ({}); only the window holding the right chunk of each is read (in-window measures); the released base on {}, default drone set, verification set for the short reads; encodings held in RAM for one question at a time and released, nothing encoded written to disk",
        dir.display(),
        cs.len(),
        cs.iter().map(|c| format!("{} ({})", c.idx + 1, c.id)).collect::<Vec<_>>().join(", "),
        r.dev.name()
    );
    let _ = writeln!(
        out,
        "\n| row | right chunk first in its window | in the best 6 | seconds per window (the row's reads and scoring) |\n|---|---|---|---|"
    );
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    for a in &arms {
        let _ = writeln!(out, "| {} | {}/{} | {}/{} | {:.2} |", a.name, a.t.first, a.t.n, a.t.six, a.t.n, a.secs / n as f64);
    }
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    {
        let _ = writeln!(
            out,
            "\nshortlist (derived from the full read's mass): mean {:.1} chunks, short window mean {} tokens, right chunk inside on {k_inside} of {}; stated fallback 6: short window mean {} tokens, inside on {six_inside}",
            k_sum as f64 / n as f64,
            tok_look / n,
            cs.len(),
            tok_six / n
        );
        let _ = writeln!(
            out,
            "low-signal positions of the full windows (rule: `core_read::low_signal`, from the content-free read, per head): {mask_out} of {mask_all} head positions left out ({:.1}%); positions left out by at least one head {masked_any}, of which whitespace-only tokens {masked_nl} ({:.1}%); whitespace-only tokens in the windows {nl_all}, left out by at least one head {nl_masked} ({:.1}%)",
            100.0 * mask_out as f64 / mask_all.max(1) as f64,
            100.0 * masked_nl as f64 / masked_any.max(1) as f64,
            100.0 * nl_masked as f64 / nl_all.max(1) as f64
        );
        let _ = writeln!(
            out,
            "time: model open and anchor {open_s:.1} s; encoding {enc_tokens} tokens of {windows} windows {enc_s:.1} s ({:.1} s per window, unavoidable); everything else {:.1} s; total {:.1} s",
            enc_s / windows.max(1) as f64,
            started.elapsed().as_secs_f64() - open_s - enc_s,
            started.elapsed().as_secs_f64()
        );
    }
    let _ = writeln!(out, "\ncontent-free poses:");
    for p in &poses {
        let _ = writeln!(out, "{p}");
    }
    let _ = writeln!(out, "\nper question (rows in the table's order; place 1 is first; - not found):");
    for l in &lines {
        let _ = writeln!(out, "{l}");
    }
    Ok(vec![("r2a".to_owned(), out)])
}

/// What one population measured in `d3r`.
struct PopRead {
    name: String,
    dev: String,
    arms: Vec<Arm>,
    lines: Vec<String>,
    tabs: Vec<(Table, Vec<bool>, bool)>,
    open_s: f64,
    enc_s: f64,
    all_s: f64,
    total_s: f64,
}

/// One population of `d3r`: the windows encoded by `pop` and read by `pop` (keys, values and
/// queries of the same weights), arm b (full read, mass) and arm c (b plus the second look) as in
/// `r2a`, and one read of every head of layers `0..=last` for the head re-selection.
#[expect(clippy::too_many_arguments, reason = "one test's inputs, passed once")]
fn d3r_pop(
    home: &Path,
    pop: &str,
    sub: &nmsubs::Substrate,
    cs: &[Case],
    heads: &[(usize, usize)],
    verify: &[(usize, usize)],
    last: usize,
    stamp: u64,
) -> Result<PopRead, Fault> {
    use nmapi::core_read::{self, Kind};
    let started = Instant::now();
    let r = Reader::open(home, pop, last, false)?;
    let open_s = started.elapsed().as_secs_f64();
    let anchor = Arc::clone(r.aw.chunks.first().ok_or_else(|| bad("no anchor"))?);
    let all = r.all_heads();
    let mut arms: Vec<Arm> = ["b: full read, mass score", "c: b + second look (derived shortlist, mass in the look)"]
        .iter()
        .map(|n| Arm { name: (*n).to_owned(), ..Arm::default() })
        .collect();
    let mut lines = Vec::new();
    let mut tabs = Vec::new();
    let (mut enc_s, mut all_s) = (0.0_f64, 0.0_f64);
    let plain = core_read::plain_null("");
    for c in cs {
        let tq = Instant::now();
        let posed = format!("\nQuestion: {}", c.text);
        let docs: Vec<String> = c.place.iter().map(|p| doc(sub, *p)).collect();
        let te = Instant::now();
        let kv = r.encode(&docs)?;
        enc_s += te.elapsed().as_secs_f64();
        let mut parts = vec![Arc::clone(&anchor)];
        parts.extend(kv.iter().cloned());
        let w = Arc::new(Window::build(&r.m, &parts)?);
        let t = Instant::now();
        let fp = core_read::raw_read(&r.m, r.dev, &w, &r.stop, heads, &posed)?;
        let fnl = core_read::raw_read(&r.m, r.dev, &w, &r.stop, heads, &plain)?;
        w.release_device();
        let mass = core_read::chunk_scores(&fp, &fnl, &w.parts, Kind::Mass, None);
        let pb = gold_place(&mass, &c.gold);
        let s_b = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (k, _) = core_read::look_size(&mass);
        let pick = core_read::look_pick(&mass, k);
        let mut p = vec![Arc::clone(&anchor)];
        p.extend(pick.iter().filter_map(|i| kv.get(*i).cloned()));
        let sw = Arc::new(Window::build(&r.m, &p)?);
        let sp = core_read::raw_read(&r.m, r.dev, &sw, &r.stop, verify, &posed)?;
        let sn = core_read::raw_read(&r.m, r.dev, &sw, &r.stop, verify, &plain)?;
        sw.release_device();
        let by = core_read::chunk_scores(&sp, &sn, &sw.parts, Kind::Mass, None);
        let pc = place_in(&core_read::look_order(&mass, &pick, &by), &c.gold);
        let s_c = s_b + t.elapsed().as_secs_f64();
        let inside = pick.iter().any(|i| c.gold.get(*i).copied().unwrap_or(false));
        drop((fp, fnl, sp, sn, sw, w, parts, p));
        if let Some(a) = arms.get_mut(0) {
            a.t.add(pb, None);
            a.secs += s_b;
        }
        if let Some(a) = arms.get_mut(1) {
            a.t.add(pc, None);
            a.secs += s_c;
        }
        let t = Instant::now();
        let tb = r.read(&kv, &all, &posed)?;
        all_s += t.elapsed().as_secs_f64();
        let show = |x: Option<usize>| x.map_or_else(|| "-".to_owned(), |v| (v + 1).to_string());
        let line = format!(
            "{} q{} ({}): {} chunks; place of the right chunk: b {} c {}; shortlist {k}, right chunk inside {inside}; {} heads recorded for the selection; {:.1} s",
            r.name,
            c.idx + 1,
            c.id,
            c.place.len(),
            show(pb),
            show(pc),
            tb.heads.len(),
            tq.elapsed().as_secs_f64()
        );
        progress(home, stamp, &line);
        lines.push(line);
        tabs.push((tb, c.gold.clone(), c.idx.is_multiple_of(2)));
        drop(kv);
        nmengn::residency::get().release_windows();
    }
    let (name, dev) = (r.name.clone(), r.dev.name().to_owned());
    drop((anchor, r));
    nmengn::residency::get().release_windows();
    Ok(PopRead { name, dev, arms, lines, tabs, open_s, enc_s, all_s, total_s: started.elapsed().as_secs_f64() })
}

/// `run dx d3r [--n k]` (2026-11-10): do the same drone heads read better with the released
/// reranker weights than with the released base weights? One variable: the population that encodes
/// the window and reads it (keys, values and queries always from the same population). Same fixed
/// questions and windows as `r2a` (by index), same heads, poses and scores as its arms b and c.
/// Added arm: 16 heads re-selected per population on the even-index questions, scored on the
/// odd-index ones, beside the default heads on the same odd-index questions. One process, the
/// populations one after the other; encodings in RAM for one question at a time, nothing encoded
/// written to disk; a line per question to `runs/DX-progress-<stamp>.txt`.
fn d3r(home: &Path, dir: &Path, stamp: u64, limit: usize) -> Result<Vec<(String, String)>, Fault> {
    let (sub, mut cs) = cases(home, dir)?;
    cs.truncate(limit);
    let set = tentacle_pass::drones(home);
    let heads = set.heads.clone();
    let verify = set.verification_set().to_vec();
    let started = Instant::now();
    let mut pops = Vec::new();
    for pop in ["qwen3-reranker-0.6b", nmapi::neural::POPULATION] {
        pops.push(d3r_pop(home, pop, &sub, &cs, &heads, &verify, set.last_layer(), stamp)?);
    }
    pops.reverse();
    let n = cs.len().max(1);
    let mut out = String::new();
    let _ = writeln!(
        out,
        "DX d3r (which released weights the drone heads read with: base against reranker; the window encoded and read by the same population)"
    );
    let _ = writeln!(
        out,
        "dump {}; {} development questions by index ({}); only the window holding the right chunk of each is read; default drone set and verification set, poses and scores of r2a arms b and c; released weights unaltered; encodings held in RAM for one question at a time and released, nothing encoded written to disk",
        dir.display(),
        cs.len(),
        cs.iter().map(|c| format!("{} ({})", c.idx + 1, c.id)).collect::<Vec<_>>().join(", ")
    );
    let _ = writeln!(
        out,
        "\n| population | row | right chunk first in its window | in the best 6 | seconds per window (reads and scoring) |\n|---|---|---|---|---|"
    );
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    for p in &pops {
        for a in &p.arms {
            let _ =
                writeln!(out, "| {} | {} | {}/{} | {}/{} | {:.2} |", p.name, a.name, a.t.first, a.t.n, a.t.six, a.t.n, a.secs / n as f64);
        }
    }
    let _ = writeln!(
        out,
        "\nhead re-selection (16 heads of layers 0..={} selected on the even-index questions by the gold log-softmax of each head's calibrated reading, scored on the odd-index questions; table scores, in-window):\n| population | heads | calibrated | mass |\n|---|---|---|---|",
        set.last_layer()
    );
    for p in &pops {
        let evens: Vec<(&Table, &[bool])> = p.tabs.iter().filter(|x| x.2).map(|x| (&x.0, x.1.as_slice())).collect();
        let pick = select(&evens, &|_, _| true);
        let (mut def_all, mut def_odd, mut sel) = (Pair::default(), Pair::default(), Pair::default());
        for (tb, g, ev) in &p.tabs {
            let ix = tb.index(&heads);
            def_all.add(tb, &ix, g, None);
            if !*ev {
                def_odd.add(tb, &ix, g, None);
                sel.add(tb, &pick, g, None);
            }
        }
        let hn: Vec<String> =
            pick.iter().filter_map(|h| p.tabs.first().and_then(|x| x.0.heads.get(*h)).map(|(l, x)| format!("L{l}.h{x}"))).collect();
        let _ = writeln!(out, "{}", def_all.line(&format!("{} | default heads, all questions", p.name)));
        let _ = writeln!(out, "{}", def_odd.line(&format!("{} | default heads, odd-index questions", p.name)));
        let _ = writeln!(out, "{}", sel.line(&format!("{} | re-selected on even, scored on odd ({})", p.name, hn.join(" "))));
    }
    let _ = writeln!(out, "\ntime:");
    for p in &pops {
        let _ = writeln!(
            out,
            "{} on {}: open and anchor {:.1} s; encoding {:.1} s; reads of every head for the selection {:.1} s; total {:.1} s",
            p.name, p.dev, p.open_s, p.enc_s, p.all_s, p.total_s
        );
    }
    let ram = nmffi::dynlib::process_memory().map_or_else(|e| e, |(peak, now)| format!("RAM peak {} MB, now {} MB", peak >> 20, now >> 20));
    let _ = writeln!(out, "whole run {:.1} s; {ram}", started.elapsed().as_secs_f64());
    let _ = writeln!(out, "\nper question (place 1 is first; - not found):");
    for p in &pops {
        for l in &p.lines {
            let _ = writeln!(out, "{l}");
        }
    }
    Ok(vec![("d3r".to_owned(), out)])
}

fn path_key(sub: &nmsubs::Substrate, c: usize) -> (String, usize) {
    sub.chunks
        .get(c)
        .map_or_else(|| (String::new(), 0), |ch| (sub.files.get(ch.file).map(|f| f.path.clone()).unwrap_or_default(), ch.start))
}

/// `run dx <name>...` (names: base d1 d2 d3 d4 d5 d6 d7 d8 d9; t1a t1 t1b t4 t5; r2a).
pub fn run(args: &[String], home: &Path) -> ExitCode {
    let stamp = nmtype::now_ms();
    let names: Vec<String> = args.iter().skip(2).cloned().collect();
    if names.is_empty() {
        eprintln!("usage: nano-lab run dx <base|d1|d2|d3|d4|d5|d6|d7|d8|d9|t1a|t1|t1b|t4|t5>...");
        return ExitCode::FAILURE;
    }
    let dir: Option<PathBuf> = {
        let mut v: Vec<PathBuf> = std::fs::read_dir(home.join("scratch/tc1-dump"))
            .map(|rd| rd.filter_map(Result::ok).map(|e| e.path()).filter(|p| p.join("substrate.json").is_file()).collect())
            .unwrap_or_default();
        v.sort();
        v.pop()
    };
    let Some(dir) = dir else {
        eprintln!("dx: no dump under scratch/tc1-dump");
        return ExitCode::FAILURE;
    };
    // 2026-11-10: `--n <k>` limits the questions (smoke runs); a T test is one test per run.
    let limit = args
        .windows(2)
        .find(|w| w.first().map(String::as_str) == Some("--n"))
        .and_then(|w| w.get(1))
        .and_then(|x| x.parse::<usize>().ok())
        .unwrap_or(SUBSET);
    let names: Vec<String> = names.into_iter().filter(|n| n != "--n" && n.parse::<usize>().is_err()).collect();
    // 2026-11-10 (refinement step 2a): `r2a`, the core read's components, one process.
    let r2a_wanted = names.iter().any(|n| n == "r2a");
    let names: Vec<String> = names.into_iter().filter(|n| n != "r2a").collect();
    // 2026-11-10: `d3r`, base against reranker weights as the reading population, one process.
    let d3r_wanted = names.iter().any(|n| n == "d3r");
    let names: Vec<String> = names.into_iter().filter(|n| n != "d3r").collect();
    // 2026-11-10 (step 3a): `j3a`, the Core Brain's joint read, one process, its own options.
    // 2026-11-10: `j3d`, the Core Brain's decision step (`nmapi::decide`), one process.
    // 2026-11-10: `j3e`, the Core Brain's decision fed by the drones, one process.
    // 2026-11-10: `whole`, the whole path through the drones in first form, stage by stage.
    if names.iter().any(|n| n == "cb1") {
        return finish(cb1(home, &dir, stamp, args), home, stamp, &names);
    }
    // 2026-11-12: `crag`, the Core Brain asking for more context, a processor replay of recorded judges.
    if names.iter().any(|n| n == "sn1") {
        return finish(sn1(home, &dir, args), home, stamp, &names);
    }
    if names.iter().any(|n| n == "sn5") {
        return finish(sn5(home, &dir), home, stamp, &names);
    }
    if names.iter().any(|n| n == "crag") {
        return finish(crag(home, &dir, stamp, args), home, stamp, &names);
    }
    if names.iter().any(|n| n == "whole") {
        return finish(whole(home, &dir, stamp, args), home, stamp, &names);
    }
    if names.iter().any(|n| n == "j3e") {
        return finish(j3e(home, &dir, stamp, args), home, stamp, &names);
    }
    if names.iter().any(|n| n == "j3d") {
        return finish(j3d(home, &dir, stamp, args), home, stamp, &names);
    }
    if names.iter().any(|n| n == "j3a") {
        return finish(j3a(home, &dir, stamp, args), home, stamp, &names);
    }
    let t_names: Vec<String> = names.iter().filter(|n| n.starts_with('t')).cloned().collect();
    let d_names: Vec<String> = names.iter().filter(|n| !n.starts_with('t')).cloned().collect();
    let result = (|| -> Result<Vec<(String, String)>, Fault> {
        let mut v = Vec::new();
        if r2a_wanted {
            v.extend(r2a(home, &dir, stamp, limit)?);
        }
        if d3r_wanted {
            v.extend(d3r(home, &dir, stamp, limit)?);
        }
        for t in &t_names {
            v.extend(tbattery(home, std::slice::from_ref(t), &dir, stamp, limit)?);
        }
        if !d_names.is_empty() {
            v.extend(battery(home, &d_names, &dir, stamp)?);
        }
        Ok(v)
    })();
    finish(result, home, stamp, &names)
}

/// Write the records of a dx run (or its fault).
fn finish(result: Result<Vec<(String, String)>, Fault>, home: &Path, stamp: u64, names: &[String]) -> ExitCode {
    match result {
        Ok(recs) => {
            for (n, t) in recs {
                let p = home.join("runs").join(format!("DX-{n}-{stamp}.txt"));
                if let Err(e) = std::fs::write(&p, &t) {
                    eprintln!("dx: {}: {e}", p.display());
                    return ExitCode::FAILURE;
                }
                println!("{t}\nreport: {}\n", p.display());
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            let p = home.join("runs").join(format!("DX-fault-{stamp}.txt"));
            let _ = std::fs::write(&p, format!("DX {names:?}: FAULT {}: {}\n", e.component, e.message));
            eprintln!("dx: {}: {}", e.component, e.message);
            ExitCode::FAILURE
        }
    }
}

/// One question of `j3a`: its index, id, text, gold chunks, the scan and the served orders.
struct JCase {
    idx: usize,
    id: String,
    text: String,
    gold: Vec<usize>,
    scan: tentacle_pass::Scan,
    served: Vec<usize>,
    core: Vec<usize>,
}

/// The questions of `j3a` from the dump: the first `limit` development questions by index (all of
/// them when `all`; otherwise the first ten whose right chunk is in some window, as `cases`).
fn jcases(home: &Path, dir: &Path, limit: usize, all: bool) -> Result<Vec<JCase>, Fault> {
    let items: Vec<Item> = crate::e2::suite(home)?;
    let mut out = Vec::new();
    for idx in scan_dump::indices(dir) {
        if out.len() >= limit {
            break;
        }
        let Some(x) = items.get(idx) else { continue };
        let (q, scan) = scan_dump::load_question(dir, idx, &x.q)?;
        let held = scan.windows.iter().any(|w| w.place.iter().any(|c| q.gold.contains(c)));
        if !all && !held {
            continue;
        }
        out.push(JCase { idx, id: q.id, text: x.q.clone(), gold: q.gold, scan, served: q.control.order, core: q.core.order });
    }
    Ok(out)
}

/// `run dx j3a [--n k] [--all] [--plan] [--group t] [--yesno off] [--budget s]` (2026-11-10, step
/// 3a: the Core Brain's joint read, `nmapi::joint_read`). Per question: every tentacle window of
/// the dump gives its shortlist exactly as `r2a` arm c (full read's mass from the dump,
/// `core_read::look_size` and `look_pick`, one second look over the shortlist's chunks encoded
/// behind the anchor, verification set, mass in the look, `look_order`); the union across
/// tentacles ordered by evidence is read jointly with the base weights in two orders (groups of at
/// most the group length, a final read over the group winners). Arms: today's served first (the
/// dump's control order), G1's hand-over first, drone-only first across tentacles (second-look
/// winner with the largest look mass; and the full read's mass across windows), joint read first
/// by R1 attention mass, R1 calibrated and R2 yes or no label logits. Also whether the right chunk
/// is in the union at all. One process; encodings in RAM for one question at a time; nothing
/// encoded written to disk; a line per question to `runs/DX-progress-<stamp>.txt`; stops before a
/// question that would pass `budget` seconds (default 600). `--plan` prints the union sizes only
/// (processor, no model).
#[expect(clippy::too_many_lines, reason = "one test, in the order it runs")]
fn j3a(home: &Path, dir: &Path, stamp: u64, args: &[String]) -> Result<Vec<(String, String)>, Fault> {
    use nmapi::core_read::{self, Kind};
    use nmapi::joint_read::{self, Evidence, Piece};
    let opt = |k: &str| args.windows(2).find(|w| w.first().map(String::as_str) == Some(k)).and_then(|w| w.get(1)).cloned();
    let flag = |k: &str| args.iter().any(|a| a == k);
    let all = flag("--all");
    let limit = opt("--n").and_then(|x| x.parse().ok()).unwrap_or(if all { 30 } else { SUBSET });
    let budget = opt("--budget").and_then(|x| x.parse::<f64>().ok()).unwrap_or(600.0);
    let group = opt("--group").and_then(|x| x.parse::<usize>().ok()).unwrap_or(joint_read::GROUP_FALLBACK);
    let yes_no = opt("--yesno").is_none_or(|x| x != "off");
    // 2026-11-10 (planning session): `--orders 1` reads each group once in evidence order;
    // `--look off` takes the shortlist order from the dump's full-read mass (no window or chunk is
    // encoded again; the drone-only arm is then the full read's mass across windows).
    let two = opt("--orders").is_none_or(|x| x != "1");
    let look = opt("--look").is_none_or(|x| x != "off");
    let started = Instant::now();
    let sub = scan_dump::load_substrate(dir)?;
    let cs = jcases(home, dir, limit, all)?;
    let set = tentacle_pass::drones(home);
    let heads = set.heads.clone();
    let verify = set.verification_set().to_vec();
    let piece_of = |c: usize, ev: Evidence| -> Piece {
        let ch = sub.chunks.get(c);
        Piece {
            chunk: c,
            path: ch.and_then(|x| sub.files.get(x.file)).map(|f| f.path.clone()).unwrap_or_default(),
            start: ch.map_or(0, |x| x.start),
            end: ch.map_or(0, |x| x.end),
            text: ch.map(|x| x.text.clone()).unwrap_or_default(),
            evidence: ev,
        }
    };
    let is_gold = |c: &JCase, x: Option<&usize>| x.is_some_and(|v| c.gold.contains(v));
    // Plan: shortlist sizes from the dump's full-read mass (processor only).
    let mut plan = String::new();
    for c in &cs {
        let mut chunks = 0;
        let mut tokens = 0;
        let mut inside = false;
        let mut gold_place = String::from("-");
        for w in &c.scan.windows {
            let mass = w.scores(tentacle_pass::Score::Mass);
            let (k, _) = core_read::look_size(&mass);
            let pick = core_read::look_pick(&mass, k);
            chunks += pick.len();
            tokens += pick.iter().filter_map(|i| w.parts.get(i + 1)).map(|p| p.1).sum::<usize>();
            inside |= pick.iter().any(|i| w.place.get(*i).is_some_and(|x| c.gold.contains(x)));
            if gold_place == "-"
                && let Some(g) = w.place.iter().position(|x| c.gold.contains(x))
            {
                let gm = mass.get(g).copied().unwrap_or(f32::NEG_INFINITY);
                gold_place = (1 + mass.iter().filter(|m| **m > gm).count()).to_string();
            }
        }
        let _ = writeln!(
            plan,
            "q{} ({}): {} windows, union {chunks} chunks, {tokens} tokens (as encoded), right chunk in the union {inside}; place of the right chunk in its window by the dump's full-read mass {gold_place}",
            c.idx + 1,
            c.id,
            c.scan.windows.len()
        );
    }
    if flag("--plan") {
        return Ok(vec![("j3a-plan".to_owned(), format!("DX j3a plan (processor only; group {group})\n{plan}"))]);
    }
    let r = Reader::open(home, nmapi::neural::POPULATION, set.last_layer(), false)?;
    let open_s = started.elapsed().as_secs_f64();
    if flag("--kcurve") {
        // Isolated component test (planning session, direction of Luis Jake Gabriel III): per
        // question the right chunk plus the k hardest wrong chunks of its own shortlist union
        // (dump full-read mass order, no second look), texts from the dump; the joint read alone
        // (one order, R1 attention mass, through layer 17) against the drone mass score of one
        // short window of the same chunks (each chunk encoded once behind the anchor, reused
        // across k). Right chunk at a position fixed by the question index. Stops before a
        // question that would pass the budget.
        let ks: Vec<usize> = opt("--ks").map_or_else(|| vec![1, 3, 7, 15], |x| x.split(',').filter_map(|v| v.parse().ok()).collect());
        let kmax = ks.iter().copied().max().unwrap_or(1);
        let anchor = Arc::clone(r.aw.chunks.first().ok_or_else(|| bad("no anchor"))?);
        let plain = core_read::plain_null("");
        let full = flag("--full");
        let mut hit = vec![(0_usize, 0_usize); ks.len()];
        let mut fhit = vec![(0_usize, 0.0_f64); ks.len()];
        let mut tok = vec![0_usize; ks.len()];
        let mut secs = vec![(0.0_f64, 0.0_f64); ks.len()];
        let (mut lines, mut done, mut enc_s, mut stop_note) = (Vec::new(), 0_usize, 0.0_f64, String::new());
        for c in &cs {
            let el = started.elapsed().as_secs_f64();
            #[expect(clippy::cast_precision_loss, reason = "counts")]
            let per = if done > 0 { (el - open_s) / done as f64 } else { 0.0 };
            if done > 0 && el + per > budget {
                stop_note =
                    format!("stopped before q{} ({}): {el:.0} s elapsed, {per:.0} s per question, cap {budget:.0} s", c.idx + 1, c.id);
                break;
            }
            let Some(g) = c.scan.windows.iter().flat_map(|w| w.place.iter()).find(|x| c.gold.contains(x)).copied() else { continue };
            let mut lists = Vec::new();
            for w in &c.scan.windows {
                let mass = w.scores(tentacle_pass::Score::Mass);
                let (k, _) = core_read::look_size(&mass);
                let mut by = core_read::look_pick(&mass, k);
                by.sort_by(|&a, &b| {
                    mass.get(b)
                        .copied()
                        .unwrap_or(f32::NEG_INFINITY)
                        .total_cmp(&mass.get(a).copied().unwrap_or(f32::NEG_INFINITY))
                        .then(a.cmp(&b))
                });
                lists.push(
                    by.iter()
                        .enumerate()
                        .filter_map(|(rank, &i)| {
                            w.place.get(i).map(|&ch| {
                                piece_of(
                                    ch,
                                    Evidence { tentacle: w.tentacle, rank, mass: mass.get(i).copied().unwrap_or(0.0), look: f32::NAN },
                                )
                            })
                        })
                        .collect::<Vec<_>>(),
                );
            }
            let ds: Vec<usize> =
                joint_read::union_order(&lists).into_iter().map(|p| p.chunk).filter(|x| !c.gold.contains(x)).take(kmax).collect();
            let all: Vec<usize> = std::iter::once(g).chain(ds.iter().copied()).collect();
            let te = Instant::now();
            let kv = r.encode(&all.iter().map(|x| doc(&sub, *x)).collect::<Vec<_>>())?;
            enc_s += te.elapsed().as_secs_f64();
            let mut row = Vec::new();
            for (ki, &k) in ks.iter().enumerate() {
                let n = (k + 1).min(all.len());
                let at = (c.idx * 7919) % n;
                let mut idx: Vec<usize> = (1..n).collect();
                idx.insert(at, 0);
                let pieces: Vec<Piece> = idx.iter().filter_map(|i| all.get(*i)).map(|x| piece_of(*x, Evidence::default())).collect();
                let jo = joint_read::read(&r.m, r.dev, &c.text, &pieces, &heads, set.last_layer(), usize::MAX, false, false)?;
                let jf = jo.mass.order.first() == Some(&at);
                // Full depth (all 28 layers), yes or no label per piece, same inputs and order.
                let (ff, fs) = if full {
                    let fo = joint_read::read(&r.m, r.dev, &c.text, &pieces, &heads, set.last_layer(), usize::MAX, true, false)?;
                    nmengn::residency::get().release_windows();
                    (Some(fo.yes_no.order.first() == Some(&at)), fo.seconds)
                } else {
                    (None, 0.0)
                };
                if let Some(f) = fhit.get_mut(ki) {
                    f.0 += usize::from(ff == Some(true));
                    f.1 += fs;
                }
                let t = Instant::now();
                let mut p = vec![Arc::clone(&anchor)];
                p.extend(idx.iter().filter_map(|i| kv.get(*i).cloned()));
                let sw = Arc::new(Window::build(&r.m, &p)?);
                let sp = core_read::raw_read(
                    &r.m,
                    r.dev,
                    &sw,
                    &r.stop,
                    &heads,
                    &format!(
                        "
Question: {}",
                        c.text
                    ),
                )?;
                let sn = core_read::raw_read(&r.m, r.dev, &sw, &r.stop, &heads, &plain)?;
                sw.release_device();
                let ms = core_read::chunk_scores(&sp, &sn, &sw.parts, Kind::Mass, None);
                let dg = gold_place(&ms, &idx.iter().map(|i| *i == 0).collect::<Vec<_>>());
                let ds_s = t.elapsed().as_secs_f64();
                drop((sp, sn, sw, p));
                nmengn::residency::get().release_windows();
                if let (Some(h), Some(tk), Some(sc)) = (hit.get_mut(ki), tok.get_mut(ki), secs.get_mut(ki)) {
                    h.0 += usize::from(jf);
                    h.1 += usize::from(dg == Some(0));
                    *tk += jo.tokens;
                    sc.0 += jo.seconds;
                    sc.1 += ds_s;
                }
                row.push(format!("k {k}: {n} chunks, joint {} tokens, right chunk at {} joint first {jf} ({:.1} s) full-depth yes or no first {} ({fs:.1} s) drone place {} ({ds_s:.1} s)", jo.tokens, at + 1, jo.seconds, ff.map_or_else(|| "not run".to_owned(), |x| x.to_string()), dg.map_or_else(|| "-".to_owned(), |x| (x + 1).to_string())));
            }
            drop(kv);
            nmengn::residency::get().release_windows();
            done += 1;
            let line = format!("q{} ({}): {}", c.idx + 1, c.id, row.join("; "));
            progress(home, stamp, &line);
            lines.push(line);
        }
        let mut out = format!("DX j3a kcurve (isolated component test: joint read alone against the drone mass score of one short window, identical chunks; base 0.6B through layer 17; one order; right chunk plus the k hardest wrong chunks of the question's own shortlist union by the dump's full-read mass)
{done} questions; cap {budget:.0} s

| k | chunks | joint read first | drone mass first | joint tokens per read | joint s per read | drone s per read (window of held chunks) | full-depth yes or no first | full-depth s per read |
|---|---|---|---|---|---|---|---|---|
");
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        for (ki, k) in ks.iter().enumerate() {
            let d = done.max(1) as f64;
            let (h, t, sc) =
                (hit.get(ki).copied().unwrap_or_default(), tok.get(ki).copied().unwrap_or(0), secs.get(ki).copied().unwrap_or_default());
            let f = fhit.get(ki).copied().unwrap_or_default();
            let fcol = if full { format!("{}/{done} | {:.1}", f.0, f.1 / d) } else { "not run | -".to_owned() };
            let _ = writeln!(
                out,
                "| {k} | {} | {}/{done} | {}/{done} | {:.0} | {:.1} | {:.1} | {fcol} |",
                k + 1,
                h.0,
                h.1,
                t as f64 / d,
                sc.0 / d,
                sc.1 / d
            );
        }
        let _ = writeln!(
            out,
            "
chunk encoding for the drone reads (once per question, reused across k) {enc_s:.1} s; model open {open_s:.1} s; total {:.1} s",
            started.elapsed().as_secs_f64()
        );
        if !stop_note.is_empty() {
            let _ = writeln!(out, "{stop_note}");
        }
        for l in &lines {
            let _ = writeln!(out, "{l}");
        }
        return Ok(vec![("j3a-kcurve".to_owned(), out)]);
    }
    if flag("--curve") {
        // Reliable joint length (analysis D gap 10.1 h2): per question the right chunk plus the
        // hardest distractors (best full-read mass across every window, right chunks left out)
        // up to each length, the right chunk at a position fixed by the question index, one
        // joint read in two orders, R1 only; the reliable length keeps two thirds of the best.
        let lens = [2_000_usize, 4_000, 8_000];
        let mut hit = [0_usize; 3];
        let mut secs = [0.0_f64; 3];
        let mut lines = Vec::new();
        for c in &cs {
            let Some(g) = c.scan.windows.iter().flat_map(|w| w.place.iter()).find(|x| c.gold.contains(x)).copied() else { continue };
            let mut ds: Vec<(f32, usize)> = c
                .scan
                .windows
                .iter()
                .flat_map(|w| w.scores(tentacle_pass::Score::Mass).into_iter().zip(w.place.iter().copied()))
                .filter(|x| !c.gold.contains(&x.1))
                .collect();
            ds.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            let mut row = Vec::new();
            for (li, &len) in lens.iter().enumerate() {
                let mut pieces = vec![piece_of(g, Evidence::default())];
                let mut tok = joint_read::piece_tokens(&r.m, 1, &pieces[0])?;
                for &(_, d) in &ds {
                    if tok >= len {
                        break;
                    }
                    let p = piece_of(d, Evidence::default());
                    tok += joint_read::piece_tokens(&r.m, pieces.len() + 1, &p)?;
                    pieces.push(p);
                }
                let at = (c.idx * 7919) % pieces.len();
                let gp = pieces.remove(0);
                pieces.insert(at, gp);
                let jo = joint_read::read(&r.m, r.dev, &c.text, &pieces, &heads, set.last_layer(), usize::MAX, false, true)?;
                nmengn::residency::get().release_windows();
                let first = jo.mass.order.first() == Some(&at);
                hit[li] += usize::from(first);
                secs[li] += jo.seconds;
                row.push(format!(
                    "{len}: {} pieces, {tok} tokens, right chunk at {} first {first} {:.1} s",
                    pieces.len(),
                    at + 1,
                    jo.seconds
                ));
            }
            let line = format!("q{} ({}): {}", c.idx + 1, c.id, row.join("; "));
            progress(home, stamp, &line);
            lines.push(line);
        }
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let curve: Vec<(usize, f64)> = lens.iter().zip(hit).map(|(l, h)| (*l, h as f64 / lines.len().max(1) as f64)).collect();
        let rel = joint_read::reliable_length(&curve);
        let mut out = format!("DX j3a curve (reliable joint length, analysis D gap 10.1 h2; R1 attention mass, two orders; base 0.6B through the drone layers)
{} questions

| joint length | right chunk first | seconds per read (two orders) |
|---|---|---|
", lines.len());
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        for (i, l) in lens.iter().enumerate() {
            let _ = writeln!(out, "| {l} | {}/{} | {:.1} |", hit[i], lines.len(), secs[i] / lines.len().max(1) as f64);
        }
        let _ = writeln!(
            out,
            "
reliable length (largest keeping two thirds of the best): {rel} tokens; total {:.1} s
",
            started.elapsed().as_secs_f64()
        );
        for l in &lines {
            let _ = writeln!(out, "{l}");
        }
        return Ok(vec![("j3a-curve".to_owned(), out)]);
    }
    let anchor = Arc::clone(r.aw.chunks.first().ok_or_else(|| bad("no anchor"))?);
    let plain = core_read::plain_null("");
    let names = [
        "served first today (the dump's control order: mass, reranker judge)",
        "G1 hand-over first (the dump's core arm)",
        "drone-only first across tentacles (second-look winner with the largest look mass)",
        "drone-only first across tentacles (full read's mass, every chunk)",
        "joint read first, R1 attention mass (two orders)",
        "joint read first, R1 calibrated (two orders)",
        "joint read first, R2 yes or no label logits (two orders)",
    ];
    let mut tal: Vec<(usize, usize)> = vec![(0, 0); names.len()];
    let (mut held_n, mut in_union, mut n_done) = (0_usize, 0_usize, 0_usize);
    let (mut look_s, mut joint_s, mut enc_s, mut tails_s, mut union_tok, mut passes, mut groups) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0_usize, 0_usize, 0_usize);
    let mut lines = Vec::new();
    let mut stopped = String::new();
    for c in &cs {
        let elapsed = started.elapsed().as_secs_f64();
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let per = if n_done > 0 { (elapsed - open_s) / n_done as f64 } else { 0.0 };
        if n_done > 0 && elapsed + per > budget {
            stopped =
                format!("stopped before q{} ({}): {elapsed:.0} s elapsed, {per:.0} s per question, budget {budget:.0} s", c.idx + 1, c.id);
            break;
        }
        let tq = Instant::now();
        let held = c.scan.windows.iter().any(|w| w.place.iter().any(|x| c.gold.contains(x)));
        let posed = format!("\nQuestion: {}", c.text);
        // Shortlists as r2a arm c, per window.
        let tl = Instant::now();
        let mut lists: Vec<Vec<Piece>> = Vec::new();
        let mut look_best: Vec<(f32, usize)> = Vec::new();
        let mut full_all: Vec<(f32, usize)> = Vec::new();
        for w in &c.scan.windows {
            let mass = w.scores(tentacle_pass::Score::Mass);
            for (i, m) in mass.iter().enumerate() {
                if let Some(&ch) = w.place.get(i) {
                    full_all.push((*m, ch));
                }
            }
            let (k, _) = core_read::look_size(&mass);
            let pick = core_read::look_pick(&mass, k);
            if !look {
                let mut by: Vec<usize> = pick.clone();
                by.sort_by(|&a, &b| {
                    mass.get(b)
                        .copied()
                        .unwrap_or(f32::NEG_INFINITY)
                        .total_cmp(&mass.get(a).copied().unwrap_or(f32::NEG_INFINITY))
                        .then(a.cmp(&b))
                });
                let mut list = Vec::new();
                for (rank, &i) in by.iter().enumerate() {
                    let Some(&ch) = w.place.get(i) else { continue };
                    let ms = mass.get(i).copied().unwrap_or(f32::NEG_INFINITY);
                    if rank == 0 {
                        look_best.push((ms, ch));
                    }
                    list.push(piece_of(ch, Evidence { tentacle: w.tentacle, rank, mass: ms, look: f32::NAN }));
                }
                lists.push(list);
                continue;
            }
            let docs: Vec<String> = pick.iter().filter_map(|i| w.place.get(*i)).map(|ch| doc(&sub, *ch)).collect();
            let te = Instant::now();
            let kv = r.encode(&docs)?;
            enc_s += te.elapsed().as_secs_f64();
            let mut p = vec![Arc::clone(&anchor)];
            p.extend(kv.iter().cloned());
            let sw = Arc::new(Window::build(&r.m, &p)?);
            let sp = core_read::raw_read(&r.m, r.dev, &sw, &r.stop, &verify, &posed)?;
            let sn = core_read::raw_read(&r.m, r.dev, &sw, &r.stop, &verify, &plain)?;
            sw.release_device();
            let by = core_read::chunk_scores(&sp, &sn, &sw.parts, Kind::Mass, None);
            let order = core_read::look_order(&mass, &pick, &by);
            let mut list = Vec::new();
            for (rank, &i) in order.iter().enumerate().filter(|(_, i)| pick.contains(i)) {
                let Some(&ch) = w.place.get(i) else { continue };
                let j = pick.iter().position(|x| *x == i).unwrap_or(0);
                let lk = by.get(j).copied().unwrap_or(f32::NEG_INFINITY);
                if rank == 0 {
                    look_best.push((lk, ch));
                }
                list.push(piece_of(
                    ch,
                    Evidence { tentacle: w.tentacle, rank, mass: mass.get(i).copied().unwrap_or(f32::NEG_INFINITY), look: lk },
                ));
            }
            lists.push(list);
            drop((sp, sn, sw, p, kv));
            nmengn::residency::get().release_windows();
        }
        let t_look = tl.elapsed().as_secs_f64();
        look_s += t_look;
        eprintln!("j3a q{}: shortlists {t_look:.1} s", c.idx + 1);
        let union = joint_read::union_order(&lists);
        let inside = union.iter().any(|p| c.gold.contains(&p.chunk));
        let utok: usize = union.iter().enumerate().map(|(k, p)| joint_read::piece_tokens(&r.m, k + 1, p).unwrap_or(0)).sum();
        // The joint read.
        let jo = joint_read::read(&r.m, r.dev, &c.text, &union, &heads, set.last_layer(), group, yes_no, two)?;
        nmengn::residency::get().release_windows();
        joint_s += jo.seconds;
        enc_s += jo.encode_s;
        tails_s += jo.tails_s;
        union_tok += utok;
        passes += jo.passes;
        groups += jo.groups;
        let top = |v: &[(f32, usize)]| v.iter().copied().max_by(|a, b| a.0.total_cmp(&b.0)).map(|x| x.1);
        let first_of = |o: &[usize]| o.first().and_then(|i| union.get(*i)).map(|p| p.chunk);
        let firsts = [
            is_gold(c, c.served.first()),
            is_gold(c, c.core.first()),
            is_gold(c, top(&look_best).as_ref()),
            is_gold(c, top(&full_all).as_ref()),
            is_gold(c, first_of(&jo.mass.order).as_ref()),
            is_gold(c, first_of(&jo.calibrated.order).as_ref()),
            yes_no && is_gold(c, first_of(&jo.yes_no.order).as_ref()),
        ];
        for (t, f) in tal.iter_mut().zip(firsts) {
            t.0 += usize::from(f);
            t.1 += usize::from(held);
        }
        held_n += usize::from(held);
        in_union += usize::from(inside);
        n_done += 1;
        let place = |o: &[usize]| {
            o.iter()
                .position(|i| union.get(*i).is_some_and(|p| c.gold.contains(&p.chunk)))
                .map_or_else(|| "-".to_owned(), |x| (x + 1).to_string())
        };
        let line = format!(
            "q{} ({}): held {held}; union {} chunks of {} windows, {utok} tokens as placed, right chunk in the union {inside}; groups {}, passes {}; first per arm (table order): {}; place of the right chunk in the joint order: R1 mass {} R1 calibrated {} R2 {}; seconds: shortlists {t_look:.1} joint {:.1} (encode {:.1}, tails {:.1}) question {:.1}",
            c.idx + 1,
            c.id,
            union.len(),
            lists.len(),
            jo.groups,
            jo.passes,
            firsts.iter().map(|f| if *f { "1" } else { "0" }).collect::<Vec<_>>().join(" "),
            place(&jo.mass.order),
            place(&jo.calibrated.order),
            if yes_no { place(&jo.yes_no.order) } else { "off".to_owned() },
            jo.seconds,
            jo.encode_s,
            jo.tails_s,
            tq.elapsed().as_secs_f64()
        );
        progress(home, stamp, &line);
        lines.push(line);
    }
    let mut out = String::new();
    let _ = writeln!(out, "DX j3a (step 3a: the Core Brain's joint read, nmapi::joint_read; base 0.6B released weights, unaltered)");
    let _ = writeln!(
        out,
        "dump {}; {} development questions by index ({}); shortlists per tentacle window as r2a arm c (derived size by the full read's mass); union across tentacles ordered by evidence (place in the tentacle after the second look, then full-read mass); joint read: question, then every piece labelled with path and lines and whole, encoded once (through all 28 layers with R2, through the drone layers without), readouts from that encoding (R1 at the repeated question over the default drone set, posed minus content-free; R2 one yes or no tail per piece), {} (second look {}); group length {group} tokens, final joint read over the group winners; encodings in RAM for one question at a time, nothing encoded written to disk",
        dir.display(),
        n_done,
        cs.iter().take(n_done).map(|c| format!("{} ({})", c.idx + 1, c.id)).collect::<Vec<_>>().join(", "),
        if two { "two orders averaged" } else { "one order (evidence order)" },
        if look { "on, as r2a arm c" } else { "off: shortlist order from the dump's full-read mass" }
    );
    let _ = writeln!(out, "\n| arm | right chunk first overall (questions whose right chunk is in some window) |\n|---|---|");
    for (nm, t) in names.iter().zip(&tal) {
        let _ = writeln!(out, "| {nm} | {}/{} |", t.0, t.1);
    }
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    {
        let n = n_done.max(1) as f64;
        let _ = writeln!(
            out,
            "\nright chunk in the union: {in_union} of {held_n} held ({n_done} questions); mean union {:.0} tokens as placed; groups {groups}, joint passes {passes}",
            union_tok as f64 / n
        );
        let _ = writeln!(
            out,
            "time: model open and anchor {open_s:.1} s; shortlists (encode and second look) {look_s:.1} s ({:.1} per question); joint reads {joint_s:.1} s ({:.1} per question; their tails {tails_s:.1} s); all encoding {enc_s:.1} s; total {:.1} s",
            look_s / n,
            joint_s / n,
            started.elapsed().as_secs_f64()
        );
    }
    if !stopped.is_empty() {
        let _ = writeln!(out, "{stopped}");
    }
    let _ = writeln!(out, "\nplan (shortlists from the dump's full-read mass):\n{plan}");
    let _ = writeln!(out, "per question (first per arm: 1 yes, 0 no; place 1 is first; - not in the union):");
    for l in &lines {
        let _ = writeln!(out, "{l}");
    }
    Ok(vec![(if all { "j3a-all".to_owned() } else { "j3a".to_owned() }, out)])
}

/// `run dx j3d [--n k] [--cap s] [--budget s]` (2026-11-10): the Core Brain's decision step,
/// `nmapi::decide`. Per question (the first `k` of the fixed development subset by index): the
/// shortlist of every tentacle window as `r2a` arm c with texts and order from the dump (derived
/// size by the full read's mass, `look_size` and `look_pick`; no window or chunk encoded), the
/// union across tentacles ordered by evidence; the released reranker scores every candidate,
/// then the tournament of full-depth joint reads with the base weights (groups of
/// `decide::GROUP`, joint rounds bounded by `--budget`, fallback `decide::BUDGET_FALLBACK_S`).
/// Arms: right chunk first by the reranker alone, by the tournament, and today's served overall
/// first (the dump's control order). One process; encodings in RAM only; stops before a question
/// that would pass `--cap` seconds (default 170).
fn j3d(home: &Path, dir: &Path, stamp: u64, args: &[String]) -> Result<Vec<(String, String)>, Fault> {
    use nmapi::core_read;
    use nmapi::decide;
    use nmapi::joint_read::{self, Evidence, Piece};
    let opt = |k: &str| args.windows(2).find(|w| w.first().map(String::as_str) == Some(k)).and_then(|w| w.get(1)).cloned();
    let limit = opt("--n").and_then(|x| x.parse().ok()).unwrap_or(3);
    let cap = opt("--cap").and_then(|x| x.parse::<f64>().ok()).unwrap_or(170.0);
    let budget = opt("--budget").and_then(|x| x.parse::<f64>().ok());
    let skip = opt("--skip").and_then(|x| x.parse::<usize>().ok()).unwrap_or(0);
    let started = Instant::now();
    let sub = scan_dump::load_substrate(dir)?;
    let cs: Vec<JCase> = jcases(home, dir, limit, false)?.into_iter().skip(skip).collect();
    let set = tentacle_pass::drones(home);
    let base = nmapi::neural::model(home)?;
    let rr = nmapi::neural::reranker_model(home)?;
    let open_s = started.elapsed().as_secs_f64();
    let piece_of = |c: usize, ev: Evidence| -> Piece {
        let ch = sub.chunks.get(c);
        Piece {
            chunk: c,
            path: ch.and_then(|x| sub.files.get(x.file)).map(|f| f.path.clone()).unwrap_or_default(),
            start: ch.map_or(0, |x| x.start),
            end: ch.map_or(0, |x| x.end),
            text: ch.map(|x| x.text.clone()).unwrap_or_default(),
            evidence: ev,
        }
    };
    let (mut hit, mut done, mut secs, mut seen) = ([0_usize; 3], 0_usize, 0.0_f64, 0_usize);
    let (mut lines, mut stopped) = (Vec::new(), String::new());
    for c in &cs {
        let tq = Instant::now();
        let mut lists: Vec<Vec<Piece>> = Vec::new();
        for w in &c.scan.windows {
            let mass = w.scores(tentacle_pass::Score::Mass);
            let (k, _) = core_read::look_size(&mass);
            let mut by = core_read::look_pick(&mass, k);
            by.sort_by(|&a, &b| {
                mass.get(b)
                    .copied()
                    .unwrap_or(f32::NEG_INFINITY)
                    .total_cmp(&mass.get(a).copied().unwrap_or(f32::NEG_INFINITY))
                    .then(a.cmp(&b))
            });
            lists.push(
                by.iter()
                    .enumerate()
                    .filter_map(|(rank, &i)| {
                        w.place.get(i).map(|&ch| {
                            piece_of(ch, Evidence { tentacle: w.tentacle, rank, mass: mass.get(i).copied().unwrap_or(0.0), look: f32::NAN })
                        })
                    })
                    .collect(),
            );
        }
        let union = joint_read::union_order(&lists);
        // Stop before a question whose estimate (measured seconds per candidate so far) passes the cap.
        let el = started.elapsed().as_secs_f64();
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let est = if seen > 0 { secs / seen as f64 * union.len() as f64 } else { 0.0 };
        if done > 0 && el + est > cap {
            stopped = format!(
                "stopped before q{} ({}): {el:.0} s elapsed, estimate {est:.0} s for its {} candidates, cap {cap:.0} s",
                c.idx + 1,
                c.id,
                union.len()
            );
            break;
        }
        seen += union.len();
        let d = decide::decide(&base, &rr, &c.text, &union, &set.heads, set.last_layer(), budget)?;
        nmengn::residency::get().release_windows();
        let gold = |i: Option<usize>| i.and_then(|i| union.get(i)).is_some_and(|p| c.gold.contains(&p.chunk));
        let f = [gold(d.by_score.first().copied()), gold(d.winner()), c.served.first().is_some_and(|x| c.gold.contains(x))];
        for (h, x) in hit.iter_mut().zip(f) {
            *h += usize::from(x);
        }
        let qs = tq.elapsed().as_secs_f64();
        secs += qs;
        done += 1;
        let place = |o: &[usize]| {
            o.iter()
                .position(|i| union.get(*i).is_some_and(|p| c.gold.contains(&p.chunk)))
                .map_or_else(|| "-".to_owned(), |x| (x + 1).to_string())
        };
        let reads: Vec<String> = d
            .reads
            .iter()
            .map(|r| {
                format!(
                    "round {} [{}] yes-no [{}] winner {} ({} tokens, {:.1} s)",
                    r.round,
                    r.group.iter().map(|i| union.get(*i).map_or(0, |p| p.chunk).to_string()).collect::<Vec<_>>().join(" "),
                    r.yes_no.iter().map(|y| format!("{y:.2}")).collect::<Vec<_>>().join(" "),
                    union.get(r.winner).map_or(0, |p| p.chunk),
                    r.tokens,
                    r.seconds
                )
            })
            .collect();
        let line = format!(
            "q{} ({}): union {} candidates, right chunk in the union {}; first (reranker, tournament, served): {}; place of the right chunk: reranker {} tournament {}; reads {}, queued {}; seconds: reranker {:.1} joint {:.1} question {qs:.1}; reads: {}",
            c.idx + 1,
            c.id,
            union.len(),
            union.iter().any(|p| c.gold.contains(&p.chunk)),
            f.iter().map(|x| if *x { "1" } else { "0" }).collect::<Vec<_>>().join(" "),
            place(&d.by_score),
            place(&d.order),
            d.reads.len(),
            d.queued.len(),
            d.rerank_s,
            d.joint_s,
            reads.join("; ")
        );
        progress(home, stamp, &line);
        eprintln!("{line}");
        lines.push(line);
    }
    let mut out = format!(
        "DX j3d (the Core Brain's decision step, first form, nmapi::decide; released Qwen3-Reranker-0.6B and base Qwen3-0.6B, unaltered, full depth)\ndump {}; {done} of the first {limit} development questions by index (first {skip} skipped); shortlists per tentacle window as r2a arm c with texts and order from the dump (no window or chunk encoded); union ordered by evidence; reranker scores every candidate; tournament of joint reads, groups of {} (from the measured curve runs/DX-j3a-kcurve-1794320274135.txt: full-depth yes or no right on every read of 2 and 4 chunks, wrong at 8), one order (score order), yes or no label per piece, group best to the next round; joint rounds bounded by {} s; unread candidates queued in score order; encodings in RAM only\n\n| arm | right chunk first |\n|---|---|\n",
        dir.display(),
        decide::GROUP,
        budget.map_or_else(|| format!("{} (fallback)", decide::BUDGET_FALLBACK_S), |b| b.to_string())
    );
    for (nm, h) in ["(a) reranker alone", "(b) tournament", "(c) served overall first today"].iter().zip(hit) {
        let _ = writeln!(out, "| {nm} | {h}/{done} |");
    }
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let _ = writeln!(
        out,
        "\nseconds per question {:.1}; model open {open_s:.1} s; total {:.1} s",
        secs / done.max(1) as f64,
        started.elapsed().as_secs_f64()
    );
    if !stopped.is_empty() {
        let _ = writeln!(out, "{stopped}");
    }
    for l in &lines {
        let _ = writeln!(out, "{l}");
    }
    Ok(vec![("j3d".to_owned(), out)])
}

/// The CX1 claim half (2026-11-12): the locked suite reader of `e2::suite` (runbook
/// `runs/2026-11-10-CX1-runbook.md`; `e2.rs` is hash-locked, so the reader is repeated here),
/// taking the odd positions within each file by type then id.
fn claim_suite(home: &Path) -> Result<Vec<Item>, Fault> {
    const TYPES: [&str; 5] = ["behaviour", "name", "value", "explain", "relation"];
    let mut out = Vec::new();
    for area in ["context", "engine", "tools"] {
        let p = home.join(format!("tests/capability-gates/cx1-context/questions-{area}.json"));
        let v = nmjson::parse(&std::fs::read(&p).map_err(|e| bad(format!("{}: {e}", p.display())))?).map_err(|e| bad(e.to_string()))?;
        let s = |x: &nmjson::Value, k: &str| -> Result<String, Fault> {
            Ok(x.get(k).and_then(|y| y.string()).map_err(|e| bad(e.to_string()))?.to_owned())
        };
        let range = |x: &nmjson::Value| -> Result<(String, usize, usize), Fault> {
            let l = x.get("lines").and_then(|y| y.array()).map_err(|e| bad(e.to_string()))?;
            let n = |y: Option<&nmjson::Value>| -> Result<usize, Fault> {
                usize::try_from(y.ok_or_else(|| bad("empty lines"))?.integer().map_err(|e| bad(e.to_string()))?)
                    .map_err(|e| bad(e.to_string()))
            };
            Ok((s(x, "file")?, n(l.first())?, n(l.last())?))
        };
        let mut qs = Vec::new();
        for x in v.get("questions").and_then(|y| y.array()).map_err(|e| bad(e.to_string()))? {
            let t = s(x, "type")?;
            let ti = TYPES.iter().position(|y| *y == t).ok_or_else(|| bad(format!("unknown type {t}")))?;
            let mut keys = vec![range(x)?];
            if let Ok(also) = x.get("also").and_then(|y| y.array()) {
                for a in also {
                    keys.push(range(a)?);
                }
            }
            qs.push((ti, s(x, "id")?, s(x, "q")?, keys));
        }
        qs.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
        out.extend(qs.into_iter().enumerate().filter(|(pos, _)| pos % 2 == 1).map(|(_, (_, id, q, keys))| Item { id, q, keys }));
    }
    Ok(out)
}

/// `run dx cb1 --records a.txt,b.txt [--k 3] [--skip s] [--n k] [--cap s]` (2026-11-12; research
/// file Part 10, smallest test CB-1, the within-file second look). From the `judges q<n>:` lines
/// of `whole` records: the Core Brain's agreement order (`decide::agreement_from`); the winner's
/// file; that file's first `k` pieces in agreement order (the winner first among them). When the
/// file has more than one piece sent up, those pieces are read together by the base Qwen3-0.6B
/// (`joint_read::read`, both orders, drone heads for the attention readouts, full-depth yes or no
/// per piece), and first place is decided again among them; every other place is unchanged.
/// Reported per readout (yes or no, attention mass, calibrated mass): right chunk first, fixed
/// and broken against the agreement order. Encodings in RAM only.
fn cb1(home: &Path, dir: &Path, stamp: u64, args: &[String]) -> Result<Vec<(String, String)>, Fault> {
    use nmapi::joint_read::{self, Evidence, Piece};
    let opt = |k: &str| args.windows(2).find(|w| w.first().map(String::as_str) == Some(k)).and_then(|w| w.get(1)).cloned();
    let k = opt("--k").and_then(|x| x.parse::<usize>().ok()).unwrap_or(3);
    let cap = opt("--cap").and_then(|x| x.parse::<f64>().ok()).unwrap_or(170.0);
    let skip = opt("--skip").and_then(|x| x.parse::<usize>().ok()).unwrap_or(0);
    let limit = opt("--n").and_then(|x| x.parse::<usize>().ok()).unwrap_or(30);
    let started = Instant::now();
    // Judge lines: question number -> (gold, pieces (chunk, reranker, embedder) in arrival order).
    // Per question: gold chunks, and each piece's chunk with its two judges' scores.
    type Judged = (Vec<usize>, Vec<(usize, f32, f32)>);
    let mut judged: BTreeMap<usize, Judged> = BTreeMap::new();
    for f in opt("--records").unwrap_or_default().split(',').filter(|x| !x.is_empty()) {
        let text = std::fs::read_to_string(home.join("runs").join(f)).map_err(|e| bad(format!("{f}: {e}")))?;
        for line in text.lines() {
            let Some((n, rest)) = line.strip_prefix("judges q").and_then(|r| r.split_once(": gold ")) else { continue };
            let Some((gold, pieces)) = rest.split_once("; pieces ") else { continue };
            let Ok(n) = n.parse::<usize>() else { continue };
            let ps = pieces
                .split_whitespace()
                .filter_map(|p| {
                    let v: Vec<&str> = p.split(':').collect();
                    Some((v.first()?.parse().ok()?, v.get(1)?.parse().ok()?, v.get(2)?.parse().ok()?))
                })
                .collect();
            judged.entry(n).or_insert((gold.split_whitespace().filter_map(|x| x.parse().ok()).collect(), ps));
        }
    }
    if judged.is_empty() {
        return Err(bad("cb1: no `judges q` lines in --records"));
    }
    let sub = scan_dump::load_substrate(dir)?;
    // 2026-11-12: `--half claim` as in `whole` (questions q101 to q130); `--lead rerank` starts
    // from the reranker's order instead of the three-judge agreement.
    let claim = opt("--half").as_deref() == Some("claim");
    let lead_rerank = opt("--lead").as_deref() == Some("rerank");
    let cs: Vec<JCase> = if claim {
        claim_suite(home)?
            .into_iter()
            .take(limit)
            .enumerate()
            .map(|(i, x)| JCase {
                idx: 100 + i,
                id: x.id.clone(),
                text: x.q.clone(),
                gold: crate::tc1::gold(&sub, &x),
                scan: tentacle_pass::Scan::default(),
                served: Vec::new(),
                core: Vec::new(),
            })
            .skip(skip)
            .collect()
    } else {
        jcases(home, dir, limit, true)?.into_iter().skip(skip).collect()
    };
    let set = tentacle_pass::drones(home);
    let base = nmapi::neural::model(home)?;
    let dev = if base.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let open_s = started.elapsed().as_secs_f64();
    let file_of = |c: usize| sub.chunks.get(c).map(|x| x.file);
    let piece_of = |c: usize| -> Piece {
        let ch = sub.chunks.get(c);
        Piece {
            chunk: c,
            path: ch.and_then(|x| sub.files.get(x.file)).map(|f| f.path.clone()).unwrap_or_default(),
            start: ch.map_or(0, |x| x.start),
            end: ch.map_or(0, |x| x.end),
            text: ch.map(|x| x.text.clone()).unwrap_or_default(),
            evidence: Evidence::default(),
        }
    };
    // Counts: agreement first; per readout (yes or no, mass, calibrated): first, fixed, broken.
    let (mut base_first, mut first, mut fixed, mut broken) = (0_usize, [0_usize; 3], [0_usize; 3], [0_usize; 3]);
    let (mut done, mut looked, mut lines, mut stopped, mut read_s) = (0_usize, 0_usize, Vec::new(), String::new(), 0.0_f64);
    let mut switches: Vec<(usize, bool, f32, bool)> = Vec::new();
    for c in &cs {
        let el = started.elapsed().as_secs_f64();
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let per = if looked > 0 { read_s / looked as f64 } else { 0.0 };
        if done > 0 && el + per > cap {
            stopped = format!("stopped before q{} ({}): {el:.0} s elapsed, cap {cap:.0} s", c.idx + 1, c.id);
            break;
        }
        let Some((gold, ps)) = judged.get(&(c.idx + 1)) else { continue };
        done += 1;
        let d = nmapi::decide::agreement_from(ps.iter().map(|p| p.1).collect(), ps.iter().map(|p| p.2).collect());
        let lead = if lead_rerank { &d.by_score } else { &d.order };
        let order: Vec<usize> = lead.iter().filter_map(|i| ps.get(*i).map(|p| p.0)).collect();
        let right = |x: Option<&usize>| x.is_some_and(|v| gold.contains(v));
        let was = right(order.first());
        base_first += usize::from(was);
        let wf = order.first().and_then(|w| file_of(*w));
        let look: Vec<usize> = order.iter().copied().filter(|x| file_of(*x) == wf).take(k).collect();
        let mut row = format!(
            "q{} ({}): agreement first {}; file look {} piece(s)",
            c.idx + 1,
            c.id,
            if was { "right" } else { "wrong" },
            look.len()
        );
        if look.len() < 2 {
            for f in &mut first {
                *f += usize::from(was);
            }
            row.push_str("; nothing to re-decide");
        } else {
            let pieces: Vec<Piece> = look.iter().map(|x| piece_of(*x)).collect();
            let t = Instant::now();
            let jo = joint_read::read(&base, dev, &c.text, &pieces, &set.heads, set.last_layer(), usize::MAX, true, true)?;
            let s = t.elapsed().as_secs_f64();
            read_s += s;
            looked += 1;
            // Margin of the yes or no readout: the new first's score minus the agreement winner's.
            let ys = &jo.yes_no.group_score;
            let margin = jo.yes_no.order.first().and_then(|i| ys.get(*i)).copied().unwrap_or(0.0) - ys.first().copied().unwrap_or(0.0);
            let _ = write!(
                row,
                ", right chunk among them {}; read {s:.1} s; yes-no margin over the agreement winner {margin:.3}; scores {}; new first:",
                look.iter().any(|x| gold.contains(x)),
                ys.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>().join(" ")
            );
            // For the switch threshold: (question, agreement right, margin, joint read's first right).
            let jr = right(jo.yes_no.order.first().and_then(|i| look.get(*i)));
            switches.push((c.idx + 1, was, margin, jr));
            for (r, ranked) in [&jo.yes_no, &jo.mass, &jo.calibrated].into_iter().enumerate() {
                let now = right(ranked.order.first().and_then(|i| look.get(*i)));
                first[r] += usize::from(now);
                fixed[r] += usize::from(now && !was);
                broken[r] += usize::from(was && !now);
                let _ = write!(row, " {} {}", ["yes-no", "mass", "calibrated"][r], if now { "right" } else { "wrong" });
            }
        }
        eprintln!("{row}");
        lines.push(row);
    }
    let mut out = format!(
        "DX cb1 (2026-11-12; the Core Brain's within-file second look, research file Part 10 CB-1; base Qwen3-0.6B unaltered, joint read of the winner file's first {k} pieces in agreement order, both orders, full-depth yes or no per piece and drone-head attention readouts)\ndump {}; {done} development questions (first {skip} skipped); {looked} re-decided by a joint read; open {open_s:.1} s, joint reads {read_s:.1} s; records {}\n\nright chunk first: agreement {base_first} of {done}\n",
        dir.display(),
        opt("--records").unwrap_or_default()
    );
    let _ = writeln!(
        out,
        "questions: {} half; first-stage order: {} (the line \"agreement\" above counts that order)",
        if claim { "claim" } else { "development" },
        if lead_rerank { "the reranker alone" } else { "the three-judge agreement" }
    );
    for r in 0..3 {
        let _ = writeln!(
            out,
            "  after the second look ({}): {} of {done} (fixed {}, broken {})",
            ["yes-no", "attention mass", "calibrated mass"][r],
            first[r],
            fixed[r],
            broken[r]
        );
    }
    // Switch only when the yes or no margin over the agreement winner reaches a threshold. The
    // threshold is chosen on one half (odd or even question numbers) and scored on the other:
    // candidates are the midpoints between observed margins; best gain (fixed minus broken), ties
    // to the larger threshold (fewer switches).
    let gain = |set: &[&(usize, bool, f32, bool)], t: f32| -> (usize, usize) {
        let mut f = 0;
        let mut b = 0;
        for (_, was, m, now) in set {
            if *m >= t && *m > 0.0 {
                f += usize::from(*now && !*was);
                b += usize::from(*was && !*now);
            }
        }
        (f, b)
    };
    let mut ms: Vec<f32> = switches.iter().map(|x| x.2).filter(|m| *m > 0.0).collect();
    ms.sort_by(f32::total_cmp);
    let mut cands: Vec<f32> = vec![0.0];
    cands.extend(ms.windows(2).filter_map(|w| match w {
        [a, b] => Some((a + b) / 2.0),
        _ => None,
    }));
    cands.extend(ms.last().map(|m| m + 1.0));
    let mut oof = (0, 0);
    for half in [1, 0] {
        let train: Vec<&(usize, bool, f32, bool)> = switches.iter().filter(|x| x.0 % 2 == half).collect();
        let test: Vec<&(usize, bool, f32, bool)> = switches.iter().filter(|x| x.0 % 2 != half).collect();
        #[expect(clippy::cast_possible_wrap, reason = "small counts")]
        let net = |g: (usize, usize)| g.0 as i64 - g.1 as i64;
        let t = cands.iter().copied().max_by(|a, b| net(gain(&train, *a)).cmp(&net(gain(&train, *b))).then(a.total_cmp(b))).unwrap_or(0.0);
        let (tf, tb) = gain(&test, t);
        oof = (oof.0 + tf, oof.1 + tb);
        let _ = writeln!(
            out,
            "  switch threshold chosen on the {} questions: {t:.3} -> on the other half fixed {tf}, broken {tb}",
            if half == 1 { "odd" } else { "even" }
        );
    }
    let _ = writeln!(
        out,
        "  out of fold with the chosen thresholds: {} of {done} (fixed {}, broken {}); pass rule of CB-1: at least 20 with at most 1 broken -> {}",
        base_first + oof.0 - oof.1,
        oof.0,
        oof.1,
        if base_first + oof.0 - oof.1 >= 20 && oof.1 <= 1 { "PASS" } else { "FAIL" }
    );
    if let Some(t) = opt("--min-margin").and_then(|x| x.parse::<f32>().ok()) {
        let all: Vec<&(usize, bool, f32, bool)> = switches.iter().collect();
        let (f, b) = gain(&all, t);
        let _ = writeln!(out, "  fixed threshold {t:.3} on all questions: {} of {done} (fixed {f}, broken {b})", base_first + f - b);
    }
    if !stopped.is_empty() {
        let _ = writeln!(out, "{stopped}");
    }
    for l in &lines {
        let _ = writeln!(out, "{l}");
    }
    let _ = stamp;
    Ok(vec![("2026-11-12-cb1".to_owned(), out)])
}

/// Judge scores recorded in earlier `j3e` reports (files under `runs/`, comma separated), keyed
/// by (question index, chunk); single chunks only.
fn known_scores(home: &Path, list: Option<String>, which: usize) -> Result<BTreeMap<(usize, usize), f32>, Fault> {
    let mut known = BTreeMap::new();
    for f in list.unwrap_or_default().split(',').filter(|x| !x.is_empty()) {
        let text =
            std::fs::read_to_string(home.join("runs").join(f)).map_err(|e| Fault::new("dx", FaultKind::Execution, format!("{f}: {e}")))?;
        for line in text.lines() {
            // 2026-11-12: `judges q<n>: gold ...; pieces chunk:reranker:embedder ...` lines of `whole`
            // records (`which` 1 takes the reranker's score, 2 the embedder's).
            if let Some((qn, pieces)) = line
                .strip_prefix("judges q")
                .and_then(|r| r.split_once(": gold "))
                .and_then(|(n, r)| Some((n.parse::<usize>().ok()?, r.split_once("; pieces ")?.1)))
            {
                for p in pieces.split_whitespace() {
                    let v: Vec<&str> = p.split(':').collect();
                    if let (Some(Ok(ch)), Some(Ok(x))) = (v.first().map(|c| c.parse::<usize>()), v.get(which).map(|x| x.parse::<f32>())) {
                        known.entry((qn.saturating_sub(1), ch)).or_insert(x);
                    }
                }
                continue;
            }
            let Some(rest) = line.strip_prefix('q') else { continue };
            let Some((num, _)) = rest.split_once(' ') else { continue };
            let (Ok(qn), Some((_, order))) = (num.parse::<usize>(), line.split_once("judge order: ")) else { continue };
            for item in order.split(", ") {
                let Some((ids, sc)) = item.strip_prefix('[').and_then(|x| x.split_once("] ")) else { continue };
                if let (Ok(ch), Ok(v)) = (ids.parse::<usize>(), sc.trim().parse::<f32>()) {
                    known.entry((qn.saturating_sub(1), ch)).or_insert(v);
                }
            }
        }
    }
    Ok(known)
}

/// `run dx sn1 [cap]` (2026-11-13, research file Solution 3, Sensory, SN1: the model sense, the
/// token-embedding affinity (layer-0 model sense) against the released embedder's cosine, each
/// fused with BM25F; pass when the embedder's fusion is above BM25F). Every chunk of the dump's tree
/// embedded once (document format, first `cap` tokens, default 128) on the card at 8 bits (its
/// closeness check passed), held in RAM only; the 30 development questions in the card's query
/// format. Per question: the right chunk's place by BM25F alone, the embedder alone, BM25F fused
/// with affinity, BM25F fused with the embedder (reciprocal rank, k 60). A smoke of 64 chunks first
/// projects the whole run and stops it when the projection passes 170 s.
fn sn1(home: &Path, dir: &Path, args: &[String]) -> Result<Vec<(String, String)>, Fault> {
    use nmapi::drone_windows::Keep;
    let cap: usize = args.iter().skip(3).find_map(|a| a.parse().ok()).unwrap_or(128);
    let t0 = Instant::now();
    let sub = scan_dump::load_substrate(dir)?;
    let mut keep = Keep::open(&nmtype::models_dir(home).join("qwen3-0.6b/files"))?;
    keep.sync(&sub)?;
    nmapi::neural::EMBED_Q8.store(true, std::sync::atomic::Ordering::Relaxed);
    let m = nmapi::neural::embedder_model(home)?;
    let dev = if m.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let path = |c: &nmsubs::Chunk| sub.files.get(c.file).map_or(String::new(), |f| f.path.clone());
    let embed = |c: &nmsubs::Chunk| nmengn::roles::card_embed_on_capped(&m, dev, &nmsubs::dense::doc_text(&path(c), &c.text), Some(cap));
    let cs = jcases(home, dir, 30, true)?;
    // `--pool n`: embed only the union of every question's first `n` chunks by BM25F (the
    // candidate pool), so the run fits the test ceiling on this card; 0 embeds the whole tree.
    let pool: usize = args
        .windows(2)
        .find(|w| w.first().map(String::as_str) == Some("--pool"))
        .and_then(|w| w.get(1))
        .and_then(|x| x.parse().ok())
        .unwrap_or(0);
    let mut wanted: Vec<usize> = (0..sub.chunks.len()).collect();
    if pool > 0 {
        let mut set = std::collections::BTreeSet::new();
        for c in &cs {
            let sig = keep.signals(&c.text)?;
            let mut o: Vec<usize> = (0..sig.len()).collect();
            o.sort_by(|a, b| sig[*b].bm25f.total_cmp(&sig[*a].bm25f).then(a.cmp(b)));
            set.extend(o.into_iter().take(pool));
        }
        wanted = set.into_iter().collect();
    }
    // `--batch n`: embed through the batched path (`card_embed_batch`, up to n tokens per forward).
    let batch: usize = args
        .windows(2)
        .find(|w| w.first().map(String::as_str) == Some("--batch"))
        .and_then(|w| w.get(1))
        .and_then(|x| x.parse().ok())
        .unwrap_or(0);
    let ts = Instant::now();
    let mut vecs: Vec<Vec<f32>> = vec![Vec::new(); sub.chunks.len()];
    if batch > 0 {
        let texts: Vec<String> =
            wanted.iter().filter_map(|&i| sub.chunks.get(i)).map(|c| nmsubs::dense::doc_text(&path(c), &c.text)).collect();
        let got = nmengn::roles::card_embed_batch(&m, &texts, cap, batch)?;
        for (&i, v) in wanted.iter().zip(got) {
            vecs[i] = v;
        }
    }
    for &i in wanted.iter().take(64) {
        if let Some(c) = sub.chunks.get(i).filter(|_| vecs[i].is_empty()) {
            vecs[i] = embed(c)?;
        }
    }
    let per = ts.elapsed().as_secs_f64() / 64.0_f64.min(wanted.len().max(1) as f64);
    let projected = per * wanted.len() as f64 + t0.elapsed().as_secs_f64() + 15.0;
    // `--ceiling s` (2026-11-13, direction to finish everything): a longer limit for this run, which
    // cannot be split (its encodings stay in RAM); default the 170 s test ceiling.
    let ceiling: f64 = args
        .windows(2)
        .find(|w| w.first().map(String::as_str) == Some("--ceiling"))
        .and_then(|w| w.get(1))
        .and_then(|x| x.parse().ok())
        .unwrap_or(170.0);
    if projected > ceiling {
        return Ok(vec![(
            "2026-11-13-sn1".to_owned(),
            format!(
                "DX sn1 stopped by its own projection: {per:.3} s per chunk at cap {cap}, {} chunks to embed, projected {projected:.0} s against the {ceiling:.0} s limit\n",
                wanted.len()
            ),
        )]);
    }
    for &i in wanted.iter().skip(64) {
        if let Some(c) = sub.chunks.get(i).filter(|_| vecs[i].is_empty()) {
            vecs[i] = embed(c)?;
        }
    }
    let embed_s = ts.elapsed().as_secs_f64();
    let place = |order: &[usize], gold: &[usize]| order.iter().position(|c| gold.contains(c)).map(|p| p + 1);
    let rank_by = |v: &[f32]| -> Vec<usize> {
        let mut o: Vec<usize> = (0..v.len()).collect();
        o.sort_by(|a, b| v[*b].total_cmp(&v[*a]).then(a.cmp(b)));
        o
    };
    let rrf = |a: &[usize], b: &[usize]| -> Vec<usize> {
        let mut s = vec![0.0_f64; a.len()];
        for list in [a, b] {
            for (r, &i) in list.iter().enumerate() {
                if let Some(x) = s.get_mut(i) {
                    *x += 1.0 / (60.0 + r as f64 + 1.0);
                }
            }
        }
        let sv: Vec<f32> = s.iter().map(|x| *x as f32).collect();
        rank_by(&sv)
    };
    let mut tally = [[0_usize; 2]; 4];
    let mut lines = Vec::new();
    for c in &cs {
        let sig = keep.signals(&c.text)?;
        let bm = rank_by(&sig.iter().map(|s| s.bm25f).collect::<Vec<_>>());
        let aff = rank_by(&sig.iter().map(|s| s.affinity).collect::<Vec<_>>());
        let q = nmengn::roles::card_embed_on(&m, dev, &nmengn::roles::card_query(nmsubs::dense::QUERY_TASK, &c.text))?;
        // Chunks outside the pool have no vector and rank last for the embedder.
        let cos: Vec<f32> =
            vecs.iter().map(|v| if v.is_empty() { f32::NEG_INFINITY } else { v.iter().zip(&q).map(|(a, b)| a * b).sum() }).collect();
        let em = rank_by(&cos);
        let arms = [bm.clone(), em.clone(), rrf(&bm, &aff), rrf(&bm, &em)];
        let ps: Vec<Option<usize>> = arms.iter().map(|o| place(o, &c.gold)).collect();
        for (k, p) in ps.iter().enumerate() {
            tally[k][0] += usize::from(*p == Some(1));
            tally[k][1] += usize::from(p.is_some_and(|x| x <= 12));
        }
        let sh = |p: Option<usize>| p.map_or_else(|| "-".to_owned(), |x| x.to_string());
        lines.push(format!(
            "q{} ({}): right chunk place: BM25F {}, embedder {}, BM25F+affinity {}, BM25F+embedder {}",
            c.idx + 1,
            c.id,
            sh(ps[0]),
            sh(ps[1]),
            sh(ps[2]),
            sh(ps[3])
        ));
    }
    let scope = if pool > 0 {
        format!("the candidate pool (union of every question's first {pool} by BM25F: {} of {} chunks)", wanted.len(), sub.chunks.len())
    } else {
        format!("the whole tree ({} chunks)", sub.chunks.len())
    };
    let mut out = format!(
        "DX sn1 (2026-11-13): model sense over {scope}, embedded at 8 bits, first {cap} tokens each, in {embed_s:.0} s ({per:.3} s per chunk); 30 development questions; fusion by reciprocal rank (k 60)\n\n| arm | right first | within the first 12 |\n|---|---|---|\n"
    );
    for (k, name) in ["BM25F alone", "embedder alone", "BM25F + affinity (layer-0 sense)", "BM25F + embedder"].iter().enumerate() {
        out.push_str(&format!("| {name} | {}/30 | {}/30 |\n", tally[k][0], tally[k][1]));
    }
    out.push_str(&format!("\npass (BM25F + embedder above BM25F alone on right first): {}\n", tally[3][0] > tally[0][0]));
    for l in &lines {
        out.push_str(l);
        out.push('\n');
    }
    out.push_str(&format!("total {:.0} s\n", t0.elapsed().as_secs_f64()));
    Ok(vec![("2026-11-13-sn1".to_owned(), out)])
}

/// `run dx sn5` (2026-11-13, research file Solution 3, Sensory, SN5: the local rules on decoys):
/// over the dump's tree, the 6 CX1 decoys (2 per area; ids only in the record) against the first 6
/// development questions: per question the tentacles reporting no information (R1 installed and
/// logged), and the rarity-weighted share of the question's key terms absent from the tree (the
/// withhold rule, `nmbrain::synthesis::ABSENT_SHARE`). Processor only.
fn sn5(home: &Path, dir: &Path) -> Result<Vec<(String, String)>, Fault> {
    use nmapi::drone_windows::Keep;
    let sub = scan_dump::load_substrate(dir)?;
    let mut keep = Keep::open(&nmtype::models_dir(home).join("qwen3-0.6b/files"))?;
    keep.sync(&sub)?;
    let mut decoys: Vec<(String, String)> = Vec::new();
    for area in ["context", "engine", "tools"] {
        let p = home.join(format!("tests/capability-gates/cx1-context/questions-{area}.json"));
        let v = nmjson::parse(&std::fs::read(&p).map_err(|e| bad(format!("{}: {e}", p.display())))?).map_err(|e| bad(e.to_string()))?;
        for d in v.get("decoys").and_then(|x| x.array()).map_err(|e| bad(e.to_string()))? {
            decoys.push((
                d.get("id").and_then(|x| x.string()).unwrap_or("?").to_owned(),
                d.get("q").and_then(|x| x.string()).unwrap_or("").to_owned(),
            ));
        }
    }
    let real: Vec<(String, String)> = jcases(home, dir, 6, true)?.into_iter().map(|c| (c.id, c.text)).collect();
    let r1 = vec![nmbrain::reflex::Installed::parse("R1").map_err(bad)?];
    let mut out = String::from("DX sn5 (2026-11-13): local rules on decoys; per question: tentacles reporting no information (R1), share of key terms absent from the tree, withheld by the rule (share at least ABSENT_SHARE)
");
    let mut tally = [(0_usize, 0_usize, 0_usize); 2];
    let mut spread = [(0_usize, 0_usize, 0.0_f32); 2];
    for (k, set) in [&decoys, &real].into_iter().enumerate() {
        for (id, q) in set {
            let g2in = nmapi::drone_path::G2In { reflexes: r1.clone(), ..nmapi::drone_path::G2In::default() };
            let t = nmapi::drone_path::stack_run_g2(&sub, &keep, q, None, true, true, &[], &BTreeMap::new(), &g2in)?;
            let none = t.g2.iter().filter(|l| l.contains("R1 no information")).count();
            // The tree's own spread for this question: windows whose best flag is not above the
            // tree mean plus k standard deviations (k 2 and 3).
            let n = t.flags.len().max(1) as f32;
            let mean = t.flags.iter().sum::<f32>() / n;
            let sd = (t.flags.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n).sqrt();
            let quiet = |k: f32| {
                t.windows
                    .iter()
                    .filter(|w| w.iter().map(|c| t.flags.get(*c).copied().unwrap_or(0.0)).fold(0.0_f32, f32::max) <= mean + k * sd)
                    .count()
            };
            let (q2, q3) = (quiet(2.0), quiet(3.0));
            let top_z = (t.flags.iter().copied().fold(0.0_f32, f32::max) - mean) / sd.max(1e-9);
            if let Some(x) = spread.get_mut(k) {
                x.0 += q2;
                x.1 += q3;
                x.2 += top_z;
            }
            let (absent, share) = sub.absent_terms(q);
            let withheld = !absent.is_empty() && share >= nmbrain::synthesis::ABSENT_SHARE;
            if let Some(x) = tally.get_mut(k) {
                x.0 += none;
                x.1 += usize::from(withheld);
                x.2 += 1;
            }
            let _ = writeln!(
                out,
                "{} {id}: no information from {none} of {} tentacles; windows with nothing above the tree mean + 2 sd {q2}, + 3 sd {q3}; tree top at {top_z:.1} sd; absent share {share:.2}; withheld {withheld}",
                if k == 0 { "decoy" } else { "real" },
                t.windows.len()
            );
        }
    }
    for (k, name) in [(0, "decoys"), (1, "real questions")] {
        let (none, w, n) = tally[k];
        let (q2, q3, z) = spread[k];
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let _ = writeln!(
            out,
            "{name}: {n} questions; tentacles reporting no information {none} in all; quiet windows (2 sd) {q2}, (3 sd) {q3}; mean tree top {:.1} sd; withheld {w} of {n}",
            z / n.max(1) as f32
        );
    }
    Ok(vec![("2026-11-13-sn5".to_owned(), out)])
}

/// `run dx crag [--known-rr files] [--k n]` (2026-11-12; research file Part 10 (c), analysis D
/// gap 10.4): processor replay over the development questions of the dump with the reranker's
/// recorded scores. Per candidate judge of the Core Brain's first place (the reranker's top P(yes),
/// its log-odds gap to the second, agreement with the embedder's top, re-read verification of the
/// first piece, the share of pieces verified), the area under the curve for right-first, and the
/// CRAG thresholds `nmapi::ask::derive` gives (none when the judge does not separate). Then one
/// request cycle (`nmapi::ask::request`, Q3 then Q1) per question: whether it brings the right
/// chunk in where it was missing, and how many chunks it adds. Ids and numbers only.
/// One question of `crag`: top P(yes), log-odds gap, embedder agreement, first piece verified,
/// share verified, right first, right sent up, reached by the request, chunks added.
type CragRow = (f64, f64, f64, f64, f64, bool, bool, bool, usize);

fn crag(home: &Path, dir: &Path, stamp: u64, args: &[String]) -> Result<Vec<(String, String)>, Fault> {
    use nmapi::drone_windows::Keep;
    let opt = |k: &str| args.windows(2).find(|w| w.first().map(String::as_str) == Some(k)).and_then(|w| w.get(1)).cloned();
    let k = opt("--k").and_then(|x| x.parse().ok()).unwrap_or(nmapi::ask::Q3_K);
    let files = opt("--known-rr").unwrap_or_else(|| "DX-2026-11-10-whole-1794504146384.txt".into());
    let known_rr = known_scores(home, Some(files.clone()), 1)?;
    let known_em = known_scores(home, Some(files.clone()), 2)?;
    let sub = scan_dump::load_substrate(dir)?;
    let cs = jcases(home, dir, 30, true)?;
    let mut keep = Keep::open(&nmtype::models_dir(home).join("qwen3-0.6b/files"))?;
    keep.sync(&sub)?;
    let lo = |p: f32| {
        let p = f64::from(p).clamp(5e-4, 1.0 - 5e-4);
        (p / (1.0 - p)).ln()
    };
    // Per question: (top, gap, agree, verified first, verified share, right first, right sent, request reached, added).
    let mut rows: Vec<CragRow> = Vec::new();
    let mut lines = Vec::new();
    for c in &cs {
        let t = nmapi::drone_path::stack_run(&sub, &keep, &c.text, None, true, true)?;
        // The pieces the recorded run's judges scored (today's stack may send up others); the
        // queue for the request comes from today's stack run.
        let scored: Vec<(usize, f32, f32)> = known_rr
            .range((c.idx, 0)..(c.idx + 1, 0))
            .map(|(&(_, ch), &r)| (ch, r, known_em.get(&(c.idx, ch)).copied().unwrap_or(0.0)))
            .collect();
        if scored.len() < 2 {
            lines.push(format!("q{} ({}): no recorded pieces; left out", c.idx + 1, c.id));
            continue;
        }
        let mut by_rr = scored.clone();
        by_rr.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let top = by_rr[0];
        let em_top = scored.iter().max_by(|a, b| a.2.total_cmp(&b.2)).map_or(0, |x| x.0);
        let terms = nmapi::whole::verify_terms(&sub, &c.text);
        let ver = |ch: usize| sub.chunks.get(ch).is_some_and(|x| nmapi::whole::verified_text(&terms, &x.low));
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let share = scored.iter().filter(|x| ver(x.0)).count() as f64 / scored.len() as f64;
        let held: Vec<usize> = by_rr.iter().map(|x| x.0).collect();
        let queued: Vec<usize> = t.handover.iter().chain(&t.queue).copied().filter(|x| !held.contains(x)).collect();
        let req = nmapi::ask::request(&sub, &held, &queued, k);
        let right_first = c.gold.contains(&top.0);
        let sent = held.iter().any(|x| c.gold.contains(x));
        let reached = !sent && req.iter().any(|(_, x)| c.gold.contains(x));
        rows.push((
            f64::from(top.1),
            lo(top.1) - lo(by_rr[1].1),
            f64::from(u8::from(em_top == top.0)),
            f64::from(u8::from(ver(top.0))),
            share,
            right_first,
            sent,
            reached,
            req.len(),
        ));
        lines.push(format!(
            "q{} ({}): right first {}; right sent up {}; first piece verified {} ({} terms); pieces verified {:.2}; request {} ({}); right chunk reached by the request {}",
            c.idx + 1, c.id, right_first, sent, ver(top.0), terms.len(), share, req.len(),
            req.iter().map(|(q, x)| format!("{q}:{x}")).collect::<Vec<_>>().join(" "), reached
        ));
    }
    let mut out = format!(
        "DX crag (2026-11-12; the Core Brain asking for more context, CRAG rule over development outcomes; processor replay of the reranker's recorded scores from {files}; Gavriel Technologies, architecture by Luis Jake Gabriel III)\ndump {}; {} questions replayed; request cycle Q3 {k} then Q1\n\n| judge of the first place | AUC right-first | thresholds (lower, upper) |\n|---|---|---|\n",
        dir.display(),
        rows.len()
    );
    type Judge = (&'static str, fn(&CragRow) -> f64);
    let judges: [Judge; 5] = [
        ("reranker top P(yes)", |r| r.0),
        ("reranker log-odds gap to the second", |r| r.1),
        ("embedder agrees on the first", |r| r.2),
        ("re-read verification of the first piece", |r| r.3),
        ("share of pieces verified", |r| r.4),
    ];
    for (name, f) in judges {
        let cases: Vec<(f64, bool, bool)> = rows.iter().map(|r| (f(r), r.5, r.6)).collect();
        let a = nmapi::ask::auc(&cases.iter().map(|c| (c.0, c.1)).collect::<Vec<_>>());
        let d = nmapi::ask::derive(&cases);
        let _ = writeln!(
            out,
            "| {name} | {} | {} |",
            a.map_or_else(|| "-".into(), |x| format!("{x:.3}")),
            d.map_or_else(|| "none (does not separate)".into(), |c| format!("{:.3}, {:.3}", c.lower, c.upper))
        );
    }
    let missing = rows.iter().filter(|r| !r.6).count();
    let reached = rows.iter().filter(|r| r.7).count();
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let added = rows.iter().map(|r| r.8).sum::<usize>() as f64 / rows.len().max(1) as f64;
    let _ = writeln!(
        out,
        "\nright first {} of {}; right chunk not sent up on {missing}; one request cycle reaches it on {reached} of those; chunks added per cycle {added:.1}",
        rows.iter().filter(|r| r.5).count(),
        rows.len()
    );
    for l in &lines {
        let _ = writeln!(out, "{l}");
    }
    progress(home, stamp, "crag done");
    Ok(vec![("2026-11-12-crag".to_owned(), out)])
}

/// `run dx whole [--n k] [--skip s] [--up n] [--plan] [--cap s] [--g1 first|stack]
/// [--known-rr files] [--known-emb files]` (2026-11-10): the whole path through the drones in
/// first form (`nmapi::drone_path`): drones sense in parallel, eight tentacles compose their
/// windows, one `g0_tx` each from its own drones' readings, G1's loop, G2's capacity gate, the
/// Core Brain's agreement decision (reranker, embedder and the Globalized Context Stack's order). Per question the
/// place of the right chunk after every stage, so the stage that loses it is named. `--plan`
/// stops after G2 (processor only). Judge scores recorded by earlier `j3e` runs can be reused
/// (`--known-rr`, `--known-emb`; three decimals), and only pieces not yet judged are read.
#[expect(clippy::too_many_lines, reason = "one study with its report")]
fn whole(home: &Path, dir: &Path, stamp: u64, args: &[String]) -> Result<Vec<(String, String)>, Fault> {
    use nmapi::drone_windows::Keep;
    let opt = |k: &str| args.windows(2).find(|w| w.first().map(String::as_str) == Some(k)).and_then(|w| w.get(1)).cloned();
    let limit = opt("--n").and_then(|x| x.parse().ok()).unwrap_or(30);
    let skip = opt("--skip").and_then(|x| x.parse::<usize>().ok()).unwrap_or(0);
    let up = opt("--up").and_then(|x| x.parse::<usize>().ok());
    let cap = opt("--cap").and_then(|x| x.parse::<f64>().ok()).unwrap_or(170.0);
    let plan = args.iter().any(|a| a == "--plan");
    // `--scale window`: each g0 reports its standing inside its own window (the first form);
    // default: the common scale.
    let common = opt("--scale").as_deref() != Some("window");
    // Strong files are dealt across the eight tentacles (default since 2026-11-10: G1's hand-over
    // holds the right chunk on 29 of 30 against 27 first-fit); `--deal first` gives first-fit.
    let deal = opt("--deal").as_deref() != Some("first");
    let first_form = opt("--g1").as_deref() == Some("first");
    let mut rules = if first_form { nmbrain::g1::Rules::first_form() } else { nmbrain::g1::Rules::stack() };
    // `--combine linear:c,s,a,r`: G1 combines by a weighted sum of confidence, whole-code-base
    // support, agreement share and the tentacle's own rank, in place of the product.
    let combine_text = opt("--combine").unwrap_or_else(|| "product".to_owned());
    if let Some(ws) = combine_text.strip_prefix("linear:") {
        let v: Vec<f32> = ws.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        if let [c, s, a, r] = v.as_slice() {
            rules.combine = nmbrain::g1::Combine::Linear(nmbrain::g1::Weights { confidence: *c, support: *s, agreement: *a, rank: *r });
        } else {
            return Err(Fault::new("dx", FaultKind::Config, "--combine linear:c,s,a,r needs four numbers"));
        }
    }
    nmbrain::g1::set_rules(rules);
    let known_rr = known_scores(home, opt("--known-rr"), 1)?;
    let known_emb = known_scores(home, opt("--known-emb"), 2)?;
    let started = Instant::now();
    let sub = scan_dump::load_substrate(dir)?;
    // 2026-11-12: `--half claim`, the CX1 claim half (questions never used in design), gold by the
    // TC1 gold rule over the saved tree, numbered from q101 so its records never mix with the
    // development half's. Ids only in every record.
    let claim = opt("--half").as_deref() == Some("claim");
    let cs: Vec<JCase> = if claim {
        claim_suite(home)?
            .into_iter()
            .take(limit)
            .enumerate()
            .map(|(i, x)| JCase {
                idx: 100 + i,
                id: x.id.clone(),
                text: x.q.clone(),
                gold: crate::tc1::gold(&sub, &x),
                scan: tentacle_pass::Scan::default(),
                served: Vec::new(),
                core: Vec::new(),
            })
            .skip(skip)
            .collect()
    } else {
        jcases(home, dir, limit, true)?.into_iter().skip(skip).collect()
    };
    let mut keep = Keep::open(&nmtype::models_dir(home).join("qwen3-0.6b/files"))?;
    keep.sync(&sub)?;
    let intake_s = started.elapsed().as_secs_f64();
    let (mut rr_m, mut emb_m): (Option<Arc<Model>>, Option<Arc<Model>>) = (None, None);
    let place = |order: &[usize], gold: &[usize]| order.iter().position(|c| gold.contains(c)).map(|p| p + 1);
    let show = |p: Option<usize>| p.map_or_else(|| "-".to_owned(), |x| x.to_string());
    // Counts: drones first, in some window, in the hand-over, hand-over first, sent up, sent-up first, Core Brain first, served first.
    let mut n = [0_usize; 8];
    let mut n2 = [0_usize; 5];
    let (mut replay_equal, mut replay_bytes) = (0_usize, 0_usize);
    let (mut done, mut lines, mut stopped, mut stack_s, mut brain_s, mut read_now) =
        (0_usize, Vec::new(), String::new(), 0.0_f64, 0.0_f64, 0_usize);
    let mut judge_lines: Vec<String> = Vec::new();
    let cb1_on = opt("--cb1").as_deref() != Some("off");
    let lead_rerank = opt("--lead").as_deref() != Some("agree");
    let drone_set = tentacle_pass::drones(home);
    let (mut base_m, mut looks, mut switched): (Option<Arc<Model>>, usize, usize) = (None, 0, 0);
    for c in &cs {
        let el = started.elapsed().as_secs_f64();
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let per = if done > 0 { (el - intake_s) / done as f64 } else { 0.0 };
        if done > 0 && el + per > cap {
            stopped = format!("stopped before q{} ({}): {el:.0} s elapsed, {per:.0} s per question, cap {cap:.0} s", c.idx + 1, c.id);
            break;
        }
        let mut t = nmapi::drone_path::stack_run(&sub, &keep, &c.text, up, common, deal)?;
        // 2026-11-12: `--g2 "<instr>;<instr>"` (G2's instructions, `nmbrain::instruct` short forms);
        // `first` stands for G1's first handed-over chunk on this question; the stack runs again
        // with them.
        if opt("--g2").is_some() || opt("--reflexes").is_some() {
            let spec = opt("--g2").unwrap_or_default();
            let first = t.handover.first().copied().unwrap_or(0).to_string();
            let orders: Vec<nmbrain::instruct::Instr> = spec
                .split(';')
                .filter(|x| !x.is_empty())
                .map(|x| nmbrain::instruct::Instr::parse(&x.replace("first", &first)))
                .collect::<Result<_, String>>()
                .map_err(|e| Fault::new("dx", FaultKind::Config, e))?;
            let reflexes: Vec<nmbrain::reflex::Installed> = opt("--reflexes")
                .unwrap_or_default()
                .split(',')
                .filter(|x| !x.is_empty())
                .map(nmbrain::reflex::Installed::parse)
                .collect::<Result<_, String>>()
                .map_err(|e| Fault::new("dx", FaultKind::Config, e))?;
            let g2in = nmapi::drone_path::G2In { orders, reflexes, ..nmapi::drone_path::G2In::default() };
            t = nmapi::drone_path::stack_run_g2(&sub, &keep, &c.text, up, common, deal, &[], &BTreeMap::new(), &g2in)?;
        }
        // 2026-11-13, SN3: `--replay records`: the drones' signals as records (encoded, decoded),
        // the stack run again from them; windows, hand-over and what goes up must be equal.
        if opt("--replay").as_deref() == Some("records") {
            let sig = keep.signals(&c.text)?;
            let marks = nmpod::record::standouts(&sig);
            let terms = nmlex::key_terms(&c.text);
            let recs: Vec<nmpod::record::Record> = sig
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let ch = sub.chunks.get(i);
                    let mut hash = [0_u8; 32];
                    if let Some(h) = ch.and_then(|x| {
                        (0..32).map(|k| u8::from_str_radix(x.key.get(2 * k..2 * k + 2)?, 16).ok()).collect::<Option<Vec<u8>>>()
                    }) {
                        hash.copy_from_slice(&h);
                    }
                    let rare: Vec<u32> = ch
                        .map(|x| {
                            terms
                                .iter()
                                .filter(|w| x.low.contains(w.as_str()))
                                .take(nmpod::record::MAX_RARE)
                                .map(|w| nmpod::record::name_id(w))
                                .collect()
                        })
                        .unwrap_or_default();
                    nmpod::record::Record {
                        hash,
                        path: ch.map_or(0, |x| u32::try_from(x.file).unwrap_or(0)),
                        start: ch.map_or(0, |x| u32::try_from(x.start).unwrap_or(0)),
                        end: ch.map_or(0, |x| u32::try_from(x.end).unwrap_or(0)),
                        tokens: u32::try_from(keep.tokens(i)).unwrap_or(0),
                        kind: 0,
                        standout: marks.get(i).copied().unwrap_or(0),
                        signals: *s,
                        rare,
                    }
                })
                .collect();
            let bytes = nmpod::record::encode_all(&recs);
            let back = nmpod::record::decode_all(&bytes).map_err(|e| Fault::new("dx", FaultKind::Execution, e))?;
            let g2in =
                nmapi::drone_path::G2In { signals: Some(back.iter().map(|r| r.signals).collect()), ..nmapi::drone_path::G2In::default() };
            let r = nmapi::drone_path::stack_run_g2(&sub, &keep, &c.text, up, common, deal, &[], &BTreeMap::new(), &g2in)?;
            let same = r.windows == t.windows && r.handover == t.handover && r.up == t.up;
            replay_equal += usize::from(same);
            replay_bytes = bytes.len();
        }
        stack_s += t.total_s;
        let mut by_flag: Vec<usize> = (0..t.flags.len()).collect();
        by_flag.sort_by(|a, b| t.flags.get(*b).copied().unwrap_or(0.0).total_cmp(&t.flags.get(*a).copied().unwrap_or(0.0)).then(a.cmp(b)));
        let p_drone = place(&by_flag, &c.gold);
        // 2026-11-13, SN2: two-stage order (file first, then piece; Zoekt): file score its best
        // piece's flag (`max`) or the sum of its best three (`top3`); pieces by flag inside a file.
        let two_stage = |top3: bool| -> Vec<usize> {
            let mut by_file: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for &c in &by_flag {
                if let Some(ch) = sub.chunks.get(c) {
                    by_file.entry(ch.file).or_default().push(c);
                }
            }
            let fscore = |cs: &Vec<usize>| -> f32 {
                cs.iter().take(if top3 { 3 } else { 1 }).map(|c| t.flags.get(*c).copied().unwrap_or(0.0)).sum()
            };
            let mut files: Vec<(&usize, &Vec<usize>)> = by_file.iter().collect();
            files.sort_by(|a, b| fscore(b.1).total_cmp(&fscore(a.1)).then(a.0.cmp(b.0)));
            files.into_iter().flat_map(|(_, cs)| cs.iter().copied()).collect()
        };
        let p_two_max = place(&two_stage(false), &c.gold);
        let p_two_top3 = place(&two_stage(true), &c.gold);
        n2[0] += usize::from(p_two_max == Some(1));
        n2[1] += usize::from(p_two_top3 == Some(1));
        n2[2] += usize::from(p_two_max.is_some_and(|p| p <= 12));
        n2[3] += usize::from(p_two_top3.is_some_and(|p| p <= 12));
        n2[4] += usize::from(p_drone.is_some_and(|p| p <= 12));
        let win = t.windows.iter().enumerate().find_map(|(w, v)| place(v, &c.gold).map(|p| (w + 1, p)));
        let p_hand = place(&t.handover, &c.gold);
        let all: Vec<usize> = t.handover.iter().chain(&t.queue).copied().collect();
        let p_all = place(&all, &c.gold);
        let p_up = place(&t.up, &c.gold);
        let mut brain = String::from("not run");
        let mut p_brain = None;
        if !plan {
            let tb = Instant::now();
            let pieces = nmapi::drone_path::pieces(&sub, &t.up, &t.windows);
            let docs: Vec<String> = pieces.iter().map(|p| nmsubs::dense::doc_text(&p.path, &p.text)).collect();
            let mut rr: Vec<f32> = pieces.iter().map(|p| known_rr.get(&(c.idx, p.chunk)).copied().unwrap_or(f32::NAN)).collect();
            let mut em: Vec<f32> = pieces.iter().map(|p| known_emb.get(&(c.idx, p.chunk)).copied().unwrap_or(f32::NAN)).collect();
            let need_rr: Vec<usize> = (0..pieces.len()).filter(|i| rr.get(*i).is_some_and(|x| x.is_nan())).collect();
            let need_em: Vec<usize> = (0..pieces.len()).filter(|i| em.get(*i).is_some_and(|x| x.is_nan())).collect();
            if !need_rr.is_empty() {
                let m = match rr_m.as_ref() {
                    Some(m) => Arc::clone(m),
                    None => {
                        let m = nmapi::neural::reranker_model(home)?;
                        rr_m = Some(Arc::clone(&m));
                        m
                    }
                };
                let dev = if m.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
                let fresh: Vec<String> = need_rr.iter().filter_map(|i| docs.get(*i).cloned()).collect();
                let got = nmengn::roles::card_rerank_set(&m, dev, nmengn::roles::CODE_INSTRUCTION, &c.text, &fresh)?;
                for (i, v) in need_rr.iter().zip(got) {
                    if let Some(x) = rr.get_mut(*i) {
                        *x = v;
                    }
                }
            }
            if !need_em.is_empty() {
                let m = match emb_m.as_ref() {
                    Some(m) => Arc::clone(m),
                    None => {
                        let m = nmapi::neural::embedder_model(home)?;
                        emb_m = Some(Arc::clone(&m));
                        m
                    }
                };
                let dev = if m.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
                let q = nmengn::roles::card_embed_on(&m, dev, &nmengn::roles::card_query(nmsubs::dense::QUERY_TASK, &c.text))?;
                for i in &need_em {
                    let e = nmengn::roles::card_embed_on(&m, dev, docs.get(*i).map_or("", String::as_str))?;
                    if let Some(x) = em.get_mut(*i) {
                        *x = q.iter().zip(&e).map(|(a, b)| a * b).sum::<f32>();
                    }
                }
            }
            read_now += need_rr.len() + need_em.len();
            // 2026-11-12: the judges' scores per piece, in arrival order, for processor replays of
            // the Core Brain's merge (`run dx arb`). Ids and numbers only.
            let judges = format!(
                "judges q{}: gold {}; pieces {}",
                c.idx + 1,
                c.gold.iter().map(ToString::to_string).collect::<Vec<_>>().join(" "),
                pieces.iter().zip(rr.iter().zip(&em)).map(|(p, (r, e))| format!("{}:{r}:{e}", p.chunk)).collect::<Vec<_>>().join(" ")
            );
            judge_lines.push(judges);
            let mut d = nmapi::decide::agreement_from(rr, em);
            // 2026-11-12: the reranker's order leads (`--lead agree` for the three-judge agreement):
            // over all 60 CX1 questions with the second look, both give 35, and the reranker's order,
            // with nothing tuned, gives 17 of 30 on the unseen half against 13.
            if lead_rerank {
                d.order.clone_from(&d.by_score);
            }
            // 2026-11-12: the Core Brain's second look inside the winning file (CB-1), unless `--cb1 off`.
            let mut look_note = String::new();
            if cb1_on {
                let bm = match base_m.as_ref() {
                    Some(m) => Arc::clone(m),
                    None => {
                        let m = nmapi::neural::model(home)?;
                        base_m = Some(Arc::clone(&m));
                        m
                    }
                };
                let lk = nmapi::decide::second_look(
                    &bm,
                    &drone_set.heads,
                    drone_set.last_layer(),
                    &c.text,
                    &pieces,
                    &mut d.order,
                    nmapi::decide::LOOK_MARGIN,
                )?;
                looks += usize::from(lk.read.len() >= 2);
                switched += usize::from(lk.switched);
                look_note = format!(
                    "; second look {} piece(s), margin {:.3}, {}, {:.1} s",
                    lk.read.len(),
                    lk.margin,
                    if lk.switched { "switched" } else { "kept" },
                    lk.seconds
                );
            }
            let order: Vec<usize> = d.order.iter().filter_map(|i| pieces.get(*i).map(|p| p.chunk)).collect();
            p_brain = place(&order, &c.gold);
            let bs = tb.elapsed().as_secs_f64();
            brain_s += bs;
            brain = format!(
                "place {} ({} reranker and {} embedder reads now{look_note}, {bs:.1} s); order {}",
                show(p_brain),
                need_rr.len(),
                need_em.len(),
                order.iter().map(ToString::to_string).collect::<Vec<_>>().join(" ")
            );
        }
        let served = c.served.first().is_some_and(|x| c.gold.contains(x));
        for (k, hit) in [
            p_drone == Some(1),
            win.is_some(),
            p_hand.is_some(),
            p_hand == Some(1),
            p_up.is_some(),
            p_up == Some(1),
            p_brain == Some(1),
            served,
        ]
        .into_iter()
        .enumerate()
        {
            if let Some(x) = n.get_mut(k) {
                *x += usize::from(hit);
            }
        }
        done += 1;
        let line = format!(
            "q{} ({}): right chunk place: drones {}; window {} (place {} in it); G1 hand-over {} of {} (with queue {} of {}); sent up {} of {}; Core Brain {}; windows {} with depths {:?} and pieces {:?}; G1 rounds {} stop \"{}\" gate {}; seconds: drones {:.2} tentacles {:.2} G1 {:.2}; sent up: {}; G2 and reflexes: {}",
            c.idx + 1,
            c.id,
            show(p_drone),
            win.map_or_else(|| "-".to_owned(), |w| w.0.to_string()),
            win.map_or_else(|| "-".to_owned(), |w| w.1.to_string()),
            show(p_hand),
            t.handover.len(),
            show(p_all),
            all.len(),
            show(p_up),
            t.up.len(),
            brain,
            t.windows.len(),
            t.depths,
            t.windows.iter().map(Vec::len).collect::<Vec<_>>(),
            t.rounds,
            t.stop,
            t.gate,
            t.sense_s,
            t.compose_s,
            t.g1_s,
            t.up.iter().map(ToString::to_string).collect::<Vec<_>>().join(" "),
            if t.g2.is_empty() { "none".to_owned() } else { t.g2.join(" | ") }
        );
        progress(home, stamp, &line);
        eprintln!("{line}");
        lines.push(line);
    }
    let mut out = format!(
        "DX whole (2026-11-10; the whole path through the drones, first form: drones sense in parallel (BM25F flags, every sense as a drone row), eight tentacle windows composed from flags and specific relations ({}), one g0 per tentacle from its own drones' readings, G1 loop ({} rules, combine {}; g0 reports on {}), G2 capacity gate, Core Brain agreement of reranker, embedder and the context stack's order; Gavriel Technologies, architecture by Luis Jake Gabriel III)\ndump {}; {done} development questions (first {skip} skipped); {} chunks, intake once {intake_s:.1} s; pieces the Core Brain reads {}{}{}\n\n| stage | right chunk |\n|---|---|\n",
        if deal { "strong files dealt in serpentine order" } else { "first-fit" },
        if first_form { "first-form" } else { "context stack (queue)" },
        combine_text,
        if common { "the common scale" } else { "each window's own scale" },
        dir.display(),
        sub.chunks.len(),
        up.unwrap_or(nmapi::drone_path::UP_FALLBACK),
        if plan { "; plan: stopped after G2, processor only" } else { "" },
        if known_rr.is_empty() && known_emb.is_empty() {
            String::new()
        } else {
            format!(
                "; recorded judge scores reused ({} reranker, {} embedder, three decimals); {read_now} reads made now",
                known_rr.len(),
                known_emb.len()
            )
        }
    );
    for (nm, k) in [
        ("drones: first by flag strength", 0),
        ("tentacles: in some window", 1),
        ("G1: in the hand-over", 2),
        ("G1: first in the hand-over", 3),
        ("G2: sent up to the Core Brain", 4),
        ("G2: first of what is sent up", 5),
        ("Core Brain: first after the decision", 6),
        ("served overall first today (reference)", 7),
    ] {
        let _ = writeln!(out, "| {nm} | {}/{done} |", n.get(k).copied().unwrap_or(0));
    }
    let _ = writeln!(out, "| drones, two-stage (file by best piece, then piece): first | {}/{done} |", n2[0]);
    let _ = writeln!(out, "| drones, two-stage (file by its best three, then piece): first | {}/{done} |", n2[1]);
    let _ = writeln!(out, "| drones by flag strength: within the first 12 | {}/{done} |", n2[4]);
    let _ = writeln!(out, "| drones, two-stage best piece: within the first 12 | {}/{done} |", n2[2]);
    let _ = writeln!(out, "| drones, two-stage best three: within the first 12 | {}/{done} |", n2[3]);
    if opt("--replay").as_deref() == Some("records") {
        let _ = writeln!(out, "| SN3 replay from drone records (windows, hand-over and what goes up equal) | {replay_equal}/{done} |");
        let _ = writeln!(out, "drone records of the whole tree: {replay_bytes} bytes ({} chunks)", sub.chunks.len());
    }
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let _ = writeln!(
        out,
        "\nseconds per question: context stack (drones to G2) {:.2}, Core Brain {:.1}; total {:.1} s",
        stack_s / done.max(1) as f64,
        brain_s / done.max(1) as f64,
        started.elapsed().as_secs_f64()
    );
    let _ = writeln!(
        out,
        "Core Brain first stage: {}",
        if lead_rerank { "the reranker's order" } else { "agreement of reranker, embedder and arrival order" }
    );
    let _ = writeln!(
        out,
        "Core Brain second look inside the winning file (CB-1): {}",
        if cb1_on {
            format!("on, threshold {}; joint reads {looks}, first place moved {switched}", nmapi::decide::LOOK_MARGIN)
        } else {
            "off".to_owned()
        }
    );
    if !stopped.is_empty() {
        let _ = writeln!(out, "{stopped}");
    }
    for l in lines.iter().chain(&judge_lines) {
        let _ = writeln!(out, "{l}");
    }
    Ok(vec![(if plan { "2026-11-10-whole-plan" } else { "2026-11-10-whole" }.to_owned(), out)])
}

/// The judge's reading count from the drones' flag strengths over the whole tree (2026-11-10):
/// the chunks that stand out of the tree's distribution, fused score at or above mean plus `z`
/// standard deviations (`z` 5 by default, the five-sigma bound: the fused scores are heavy-tailed,
/// so 3 sigma admits 59 to 79 chunks of 2,490 on the ten development questions and 5 sigma 11 to
/// 26, `runs/DX-2026-11-10-j3e-plan-*.txt`), at least 2 so the judge has a choice. Fallback `J3E_N_FALLBACK` when the
/// spread is zero or nothing stands out.
fn j3e_n(fused: &[f32], z: f32) -> (usize, f32, f32) {
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let len = fused.len().max(1) as f32;
    let mean = fused.iter().sum::<f32>() / len;
    let sd = (fused.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / len).sqrt();
    let k = fused.iter().filter(|x| **x >= mean + z * sd).count();
    if sd <= 0.0 || k == 0 { (J3E_N_FALLBACK, mean, sd) } else { (k.max(2), mean, sd) }
}

/// The stated fallback reading count of `j3e`.
const J3E_N_FALLBACK: usize = 12;

/// The reading count that fits the 3 minute test on this card (2026-11-10: about 12 chunks per
/// question read separately at about 3.4 ms per token); `--ncap` overrides.
const J3E_N_FIT: usize = 12;

/// `run dx j3e [--n k] [--cap s] [--z z] [--fixed n]` (2026-11-10): the Core Brain's decision fed
/// by the drones, one process, off by default in the served path. Per question (the first `k`
/// development questions by index, all of them): (1) the drones sense the whole tree against the
/// request (the `window_pieces` hand-off, with the tree taken in once and kept in RAM,
/// `drone_windows::Keep`), giving the drone order; (2) neighbour merge: walking the drone order,
/// each chunk joins with its adjacent chunks of the same file while those are also near the top
/// (within the top 2N of the drone order), into one candidate span with the full text of every
/// chunk joined, nothing cut; (3) the judge (released Qwen3-Reranker-0.6B, unaltered, full depth,
/// the model card's prompt) reads the first N spans in drone order, N from `j3e_n`; every span
/// below stays queued by reference; (4) winner = the judge's first. Arms: right chunk first by
/// (a) the drones alone, (b) drones then judge, (c) today's served overall first (from the dump);
/// right chunk inside the top N.
fn j3e(home: &Path, dir: &Path, stamp: u64, args: &[String]) -> Result<Vec<(String, String)>, Fault> {
    use nmapi::drone_windows::Keep;
    let opt = |k: &str| args.windows(2).find(|w| w.first().map(String::as_str) == Some(k)).and_then(|w| w.get(1)).cloned();
    let limit = opt("--n").and_then(|x| x.parse().ok()).unwrap_or(10);
    let cap = opt("--cap").and_then(|x| x.parse::<f64>().ok()).unwrap_or(170.0);
    let z = opt("--z").and_then(|x| x.parse::<f32>().ok()).unwrap_or(5.0);
    let fixed = opt("--fixed").and_then(|x| x.parse::<usize>().ok());
    let skip = opt("--skip").and_then(|x| x.parse::<usize>().ok()).unwrap_or(0);
    let plan = args.iter().any(|a| a == "--plan");
    let merge = args.iter().any(|a| a == "--merge");
    // 2026-11-10: `--order bm25` gives the drones' BM25 order alone (D1 rerun: right chunk in the
    // top 12 on 28 of 30 against 22 for the first-form fusion).
    let bm25 = opt("--order").as_deref() == Some("bm25");
    // 2026-11-10: `--order bm25f`, Sourcegraph's BM25F (definition-line and path hits boosted).
    let bm25f = opt("--order").as_deref() == Some("bm25f");
    // 2026-11-10: `--judge embed`: the judge is the released Qwen3-Embedding-0.6B (card query
    // format with the search lane's code instruction, document as the lane embeds it), scoring
    // each piece by the cosine of the two unit vectors; vectors in RAM only. Recorded reranker
    // scores (`--known`) are not used in this mode.
    let judge_embed = opt("--judge").as_deref() == Some("embed");
    // 2026-11-10: `--known a.txt,b.txt`: judge scores recorded in earlier j3e reports (same model,
    // prompt and full-length documents; the judge scores each piece alone) are reused for single
    // chunks, so only pieces not yet judged are read. Keyed by (question index, chunk).
    let mut known: BTreeMap<(usize, usize), f32> = BTreeMap::new();
    if let Some(list) = opt("--known") {
        for f in list.split(',').filter(|x| !x.is_empty()) {
            let text = std::fs::read_to_string(home.join("runs").join(f))
                .map_err(|e| Fault::new("dx", FaultKind::Execution, format!("{f}: {e}")))?;
            for line in text.lines() {
                let Some(rest) = line.strip_prefix('q') else { continue };
                let Some((num, _)) = rest.split_once(' ') else { continue };
                let (Ok(qn), Some((_, order))) = (num.parse::<usize>(), line.split_once("judge order: ")) else { continue };
                for item in order.split(", ") {
                    let Some((ids, sc)) = item.strip_prefix('[').and_then(|x| x.split_once("] ")) else { continue };
                    if let (Ok(ch), Ok(v)) = (ids.parse::<usize>(), sc.trim().parse::<f32>()) {
                        known.entry((qn.saturating_sub(1), ch)).or_insert(v);
                    }
                }
            }
        }
    }
    // 2026-11-10: `--prof` times the judge by stage (`nmffi::prof`; the engine synchronises the
    // device at component edges while it is on, so results are unchanged, time a little longer).
    let profile = args.iter().any(|a| a == "--prof");
    let ncap = opt("--ncap").and_then(|x| x.parse::<usize>().ok()).unwrap_or(J3E_N_FIT);
    let started = Instant::now();
    let sub = scan_dump::load_substrate(dir)?;
    let cs: Vec<JCase> = jcases(home, dir, limit, true)?.into_iter().skip(skip).collect();
    let mut keep = Keep::open(&nmtype::models_dir(home).join("qwen3-0.6b/files"))?;
    keep.sync(&sub)?;
    let intake_s = started.elapsed().as_secs_f64();
    let rr_opt = if plan {
        None
    } else if judge_embed {
        Some(nmapi::neural::embedder_model(home)?)
    } else {
        Some(nmapi::neural::reranker_model(home)?)
    };
    let dev = if rr_opt.as_ref().is_some_and(|m| m.pop.gpu.is_some()) { Device::Gpu } else { Device::Cpu };
    let open_s = started.elapsed().as_secs_f64();
    let path_of = |c: usize| sub.chunks.get(c).and_then(|x| sub.files.get(x.file)).map(|f| f.path.clone()).unwrap_or_default();
    let same_file = |a: usize, b: usize| matches!((sub.chunks.get(a), sub.chunks.get(b)), (Some(x), Some(y)) if x.file == y.file);
    let (mut hit, mut inn, mut hitn, mut done, mut secs, mut seen) = ([0_usize; 5], 0_usize, 0_usize, 0_usize, 0.0_f64, 0_usize);
    let (mut lines, mut stopped, mut ns) = (Vec::new(), String::new(), Vec::new());
    for c in &cs {
        let tq = Instant::now();
        let fused = if bm25f {
            keep.sense_with(&c.text, &[|s| s.bm25f])?
        } else if bm25 {
            keep.sense_with(&c.text, &[|s| s.bm25])?
        } else {
            keep.sense(&c.text)?
        };
        let total = fused.len();
        let mut order: Vec<usize> = (0..total).collect();
        order.sort_by(|a, b| fused.get(*b).copied().unwrap_or(0.0).total_cmp(&fused.get(*a).copied().unwrap_or(0.0)).then(a.cmp(b)));
        let sense_s = tq.elapsed().as_secs_f64();
        let (n_rule, mean, sd) = j3e_n(&fused, z);
        let n = fixed.unwrap_or_else(|| n_rule.min(ncap));
        let mut place = vec![usize::MAX; total];
        for (p, &i) in order.iter().enumerate() {
            if let Some(x) = place.get_mut(i) {
                *x = p;
            }
        }
        let near = 2 * n;
        let is_near = |i: usize| place.get(i).is_some_and(|p| *p < near);
        // Neighbour merge over the drone order; a span keeps the drone place of its best chunk.
        let mut used = vec![false; total];
        let mut spans: Vec<Vec<usize>> = Vec::new();
        for &i in &order {
            if spans.len() >= n {
                break;
            }
            if used.get(i).copied().unwrap_or(true) {
                continue;
            }
            used[i] = true;
            let mut s = vec![i];
            if merge && is_near(i) {
                let mut lo = i;
                while lo > 0 && !used[lo - 1] && same_file(lo - 1, i) && is_near(lo - 1) {
                    lo -= 1;
                    used[lo] = true;
                    s.push(lo);
                }
                let mut hi = i;
                while hi + 1 < total && !used[hi + 1] && same_file(hi + 1, i) && is_near(hi + 1) {
                    hi += 1;
                    used[hi] = true;
                    s.push(hi);
                }
                s.sort_unstable();
            }
            spans.push(s);
        }
        let el = started.elapsed().as_secs_f64();
        #[expect(clippy::cast_precision_loss, reason = "counts")]
        let est = if seen > 0 { secs / seen as f64 * spans.len() as f64 } else { 0.0 };
        if done > 0 && el + est > cap {
            stopped = format!(
                "stopped before q{} ({}): {el:.0} s elapsed, estimate {est:.0} s for its {} spans, cap {cap:.0} s",
                c.idx + 1,
                c.id,
                spans.len()
            );
            break;
        }
        if plan {
            let gp = c.gold.iter().filter_map(|g| place.get(*g)).min().map_or(0, |p| p + 1);
            let in_n = spans.iter().any(|s| s.iter().any(|x| c.gold.contains(x)));
            let f = [order.first().is_some_and(|x| c.gold.contains(x)), c.served.first().is_some_and(|x| c.gold.contains(x))];
            let line = format!(
                "plan q{} ({}): N {n} (z {z}; z 3 would give {}, z 4 {}, z 6 {}); mean {mean:.5} sd {sd:.5}; spans {} merged {}; right chunk drone place {gp}; inside top N {}; first (drones chunk, served): {} {}",
                c.idx + 1,
                c.id,
                j3e_n(&fused, 3.0).0,
                j3e_n(&fused, 4.0).0,
                j3e_n(&fused, 6.0).0,
                spans.len(),
                spans.iter().filter(|s| s.len() > 1).count(),
                u8::from(in_n),
                u8::from(f[0]),
                u8::from(f[1])
            );
            eprintln!("{line}");
            lines.push(line);
            continue;
        }
        let Some(rr) = rr_opt.as_ref() else { continue };
        eprintln!("q{} ({}): judging {} spans", c.idx + 1, c.id, spans.len());
        seen += spans.len();
        let docs: Vec<String> = spans
            .iter()
            .map(|s| {
                nmsubs::dense::doc_text(
                    &path_of(s.first().copied().unwrap_or(0)),
                    &s.iter().filter_map(|i| sub.chunks.get(*i)).map(|x| x.text.as_str()).collect::<Vec<_>>().join("\n"),
                )
            })
            .collect();
        // Nothing cut: the document cap is the longest span, bounded only by the model's own
        // reference ceiling (8,192 tokens less the prompt); a span above it is named in the record.
        let lens: Vec<usize> = docs.iter().map(|d| rr.pop.tokenizer.encode(d).map_or(0, |v| v.len())).collect();
        let fixed_toks =
            rr.pop.tokenizer.encode(&nmengn::roles::card_prompt(nmengn::roles::CODE_INSTRUCTION, &c.text, "")).map_or(0, |v| v.len());
        let ceiling = nmengn::roles::REFERENCE_MAX_TOKENS.saturating_sub(fixed_toks);
        let longest = lens.iter().copied().max().unwrap_or(0);
        let derived = nmengn::roles::input_cap(rr, dev, fixed_toks);
        let doc_cap = longest.min(ceiling);
        let over = lens.iter().filter(|l| **l > ceiling).count();
        if profile {
            nmffi::prof::reset();
            nmffi::prof::set(true);
        }
        let tj = Instant::now();
        let known_of =
            |s: &Vec<usize>| if s.len() == 1 && !judge_embed { s.first().and_then(|ch| known.get(&(c.idx, *ch)).copied()) } else { None };
        let fresh: Vec<usize> = (0..spans.len()).filter(|j| spans.get(*j).and_then(known_of).is_none()).collect();
        let fresh_docs: Vec<String> = fresh.iter().filter_map(|j| docs.get(*j).cloned()).collect();
        let fresh_scores = if fresh_docs.is_empty() {
            Vec::new()
        } else if judge_embed {
            let q = nmengn::roles::card_embed_on_capped(rr, dev, &nmengn::roles::card_query(nmsubs::dense::QUERY_TASK, &c.text), None)?;
            let mut v = Vec::with_capacity(fresh_docs.len());
            for d in &fresh_docs {
                let e = nmengn::roles::card_embed_on_capped(rr, dev, d, Some(doc_cap + 1))?;
                v.push(q.iter().zip(&e).map(|(a, b)| a * b).sum::<f32>());
            }
            v
        } else {
            nmengn::roles::card_rerank_set_capped(rr, dev, nmengn::roles::CODE_INSTRUCTION, &c.text, &fresh_docs, Some(doc_cap))?
        };
        let mut scores: Vec<f32> = spans.iter().map(|s| known_of(s).unwrap_or(f32::NAN)).collect();
        for (j, v) in fresh.iter().zip(fresh_scores) {
            if let Some(x) = scores.get_mut(*j) {
                *x = v;
            }
        }
        if !known.is_empty() {
            eprintln!("q{} ({}): {} of {} pieces read now, the rest from recorded scores", c.idx + 1, c.id, fresh.len(), spans.len());
        }
        let judge_s = tj.elapsed().as_secs_f64();
        if profile {
            nmffi::prof::set(false);
            let mut tb = nmffi::prof::table();
            tb.sort_by(|a, b| b.1.total_cmp(&a.1));
            let tot: f64 = tb.iter().map(|r| r.1).sum();
            let mut pl = format!(
                "prof q{} ({}): judge {judge_s:.1} s over {} tokens in {} pieces (longest {longest}); stages:",
                c.idx + 1,
                c.id,
                lens.iter().sum::<usize>(),
                lens.len()
            );
            for (k, secs, n) in tb.iter().take(8) {
                let _ = write!(pl, " {k} {secs:.2} s ({:.0}%, x{n});", 100.0 * secs / tot.max(1e-9));
            }
            eprintln!("{pl}");
            lines.push(pl);
        }
        let mut jo: Vec<usize> = (0..spans.len()).collect();
        jo.sort_by(|a, b| {
            scores
                .get(*b)
                .copied()
                .unwrap_or(f32::NEG_INFINITY)
                .total_cmp(&scores.get(*a).copied().unwrap_or(f32::NEG_INFINITY))
                .then(a.cmp(b))
        });
        let gold_span = |s: &[usize]| s.iter().any(|x| c.gold.contains(x));
        // 2026-11-10 arm (d): reciprocal-rank fusion (k 60) of the judge's order and the drone
        // order over the N spans read (spans are in drone order, so span j has drone rank j).
        let mut jrank = vec![0_usize; spans.len()];
        for (r, j) in jo.iter().enumerate() {
            if let Some(x) = jrank.get_mut(*j) {
                *x = r;
            }
        }
        #[expect(clippy::cast_precision_loss, reason = "ranks")]
        let rrf = |j: usize| 1.0 / (60.0 + (j + 1) as f64) + 1.0 / (60.0 + (jrank.get(j).copied().unwrap_or(0) + 1) as f64);
        let mut fo: Vec<usize> = (0..spans.len()).collect();
        fo.sort_by(|a, b| rrf(*b).total_cmp(&rrf(*a)).then(a.cmp(b)));
        let f = [
            order.first().is_some_and(|x| c.gold.contains(x)),
            spans.first().is_some_and(|s| gold_span(s)),
            jo.first().and_then(|j| spans.get(*j)).is_some_and(|s| gold_span(s)),
            c.served.first().is_some_and(|x| c.gold.contains(x)),
            fo.first().and_then(|j| spans.get(*j)).is_some_and(|s| gold_span(s)),
        ];
        for (h, x) in hit.iter_mut().zip(f) {
            *h += usize::from(x);
        }
        let in_n = spans.iter().any(|s| gold_span(s));
        inn += usize::from(in_n);
        let qs = tq.elapsed().as_secs_f64();
        secs += qs;
        done += 1;
        ns.push(n);
        let gold_drone = c.gold.iter().filter_map(|g| place.get(*g)).min().map_or_else(|| "-".to_owned(), |p| (p + 1).to_string());
        let gold_judge =
            jo.iter().position(|j| spans.get(*j).is_some_and(|s| gold_span(s))).map_or_else(|| "-".to_owned(), |p| (p + 1).to_string());
        let shown: Vec<String> = jo
            .iter()
            .filter_map(|j| {
                spans.get(*j).map(|s| {
                    format!(
                        "[{}] {:.3}",
                        s.iter().map(ToString::to_string).collect::<Vec<_>>().join("+"),
                        scores.get(*j).copied().unwrap_or(f32::NAN)
                    )
                })
            })
            .collect();
        let merged = spans.iter().filter(|s| s.len() > 1).count();
        let attached: Vec<usize> = jo
            .first()
            .and_then(|j| spans.get(*j))
            .map(|s| {
                let (lo, hi) = (s.first().copied().unwrap_or(0), s.last().copied().unwrap_or(0));
                let mut v: Vec<usize> = Vec::new();
                if lo > 0 && same_file(lo - 1, lo) {
                    v.push(lo - 1);
                }
                v.extend(s.iter().copied());
                if hi + 1 < total && same_file(hi + 1, hi) {
                    v.push(hi + 1);
                }
                v
            })
            .unwrap_or_default();
        hitn += usize::from(attached.iter().any(|x| c.gold.contains(x)));
        let line = format!(
            "q{} ({}): N {n} (rule {n_rule}; fused mean {mean:.5} sd {sd:.5}); spans read {} of which merged {merged}, chunks queued by reference {}; winner with neighbours attached [{}]; right chunk place: drones {gold_drone} judge {gold_judge}; inside top N {}; first (drones chunk, drones span, judge, served): {}; span tokens total {} longest {longest}, document cap {doc_cap} (memory-derived cap would be {derived}), spans above the model ceiling {over}; seconds: sense {sense_s:.2} judge {judge_s:.1} question {qs:.1}; drone order of the spans read: {}; judge order: {}",
            c.idx + 1,
            c.id,
            spans.len(),
            total - spans.iter().map(Vec::len).sum::<usize>(),
            attached.iter().map(ToString::to_string).collect::<Vec<_>>().join("+"),
            u8::from(in_n),
            f.iter().map(|x| if *x { "1" } else { "0" }).collect::<Vec<_>>().join(" "),
            lens.iter().sum::<usize>(),
            spans.iter().map(|s| s.iter().map(ToString::to_string).collect::<Vec<_>>().join("+")).collect::<Vec<_>>().join(" "),
            shown.join(", ")
        );
        progress(home, stamp, &line);
        eprintln!("{line}");
        lines.push(line);
    }
    let setup = format!(
        "judge {}; drone order {}{}; {}",
        if judge_embed { "Qwen3-Embedding-0.6B cosine (card query format, lane instruction)" } else { "Qwen3-Reranker-0.6B" },
        if bm25f {
            "BM25F (Sourcegraph, boost 5)"
        } else if bm25 {
            "BM25 alone"
        } else {
            "first-form fusion"
        },
        if known.is_empty() {
            String::new()
        } else {
            format!("; {} recorded judge scores reused (pieces already judged are not read again)", known.len())
        },
        if merge {
            "neighbour merge of adjacent same-file chunks both within the top 2N before judging"
        } else {
            "no merge before judging: every chunk read separately"
        }
    );
    let mut out = format!(
        "DX j3e (2026-11-10; the Core Brain's decision fed by the drones; released Qwen3-Reranker-0.6B unaltered, full depth, model card prompt; Gavriel Technologies, architecture by Luis Jake Gabriel III)\ndump {}; {done} of the first {limit} development questions by index (first {skip} skipped); drones sense the whole tree ({} chunks, taken in once in RAM, {intake_s:.1} s); {}; judge reads the first N in drone order, each at full length (document cap = longest, nothing cut); N = chunks with fused score at or above mean + {z} sd over the tree, at least 2, fallback {J3E_N_FALLBACK}, capped at {ncap} (what fits the 3 minute test){}; the judge winner is reported with its adjacent same-file chunks attached; the rest queued by reference; encodings in RAM only\n\n| arm | right chunk first |\n|---|---|\n",
        dir.display(),
        sub.chunks.len(),
        setup,
        fixed.map_or_else(String::new, |f| format!("; this run N fixed at {f}"))
    );
    for (nm, h) in [
        "(a) drones alone (first chunk)",
        "(a2) drones alone (first span)",
        "(b) drones then judge on top N",
        "(c) served overall first today",
        "(d) judge and drone order fused (RRF k 60)",
    ]
    .iter()
    .zip(hit)
    {
        let _ = writeln!(out, "| {nm} | {h}/{done} |");
    }
    let _ = writeln!(out, "| right chunk inside the top N | {inn}/{done} |");
    let _ = writeln!(out, "| judge winner with its neighbours attached holds the right chunk | {hitn}/{done} |");
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let _ = writeln!(
        out,
        "\nN per question {ns:?}; seconds per question {:.1}; intake and model open {open_s:.1} s; total {:.1} s",
        secs / done.max(1) as f64,
        started.elapsed().as_secs_f64()
    );
    if !stopped.is_empty() {
        let _ = writeln!(out, "{stopped}");
    }
    for l in &lines {
        let _ = writeln!(out, "{l}");
    }
    Ok(vec![(if plan { "2026-11-10-j3e-plan" } else { "2026-11-10-j3e" }.to_owned(), out)])
}
