//! `nano-lab run bcart <step> ...`: the reranker's base cartridge for one job, judging which
//! passages of code answer a question, measured against the bare weights on held-out questions.
//!
//! The judged set is what the served whole path really hands the reranker: each question is asked
//! in full mode with the dense lane off, the training store's tap gives the candidates, and each is
//! rebuilt exactly as the judge read it (its window and path line). The split is fixed by question
//! id (`part_of`): odd development questions map, even ones and the real-use questions are held out.
//!
//! A spec names what one arm runs: `base` (the open weights), `ffn=<way>:<share>` (that share of
//! feed-forward blocks of 64 channels left out in every layer, lowest first by the mapping way
//! `wanda`, `act` or `rand<seed>`), `bits=<n>` (the MicroModel's precision selector), joined by
//! commas; `cart=<file>` seats a base cartridge (the open weights released) and `addon=<file>`
//! plugs an additional cartridge on top. Measurements over the open weights only, never written
//! into a cartridge: `depth=<k>`, `drop=<a+b>`, `heads=<way>:<n>`, `fbits`, `abits`.
//!
//! Steps: `capture <root> <questions> <out> [ids]`, `eval <spec> <part> [cpu=<n>]`, `record <part>`
//! (mapping: Wanda and activity scores, the released head read at every layer), `build <spec>
//! <id>` (writes the base cartridge), `release <cartridge>` (seat, release, pull, with residency),
//! `whole <spec> <root> <questions> [ids]` (the served path with the cartridge seated as the
//! reranker). Reports go to `runs/BCART-<step>-<ms>.txt`.
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use nmengn::Assembly;
use nmengn::assembly::{Edge, nid};
use nmengn::model::{Model, Population, RunOpts, values};
use nmengn::placement::Device;
use nmjson::Value;
use nmtype::{Fault, FaultKind};

const POP: &str = "qwen3-reranker-0.6b";
const ASM: &str = "qwen3-reranker";
const BLOCK: usize = 64;
const DIR: &str = "scratch/base-cartridge";

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("bcart", FaultKind::Execution, m)
}
fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}
fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn text(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default()
}
fn int(v: &Value, k: &str) -> usize {
    v.get(k).and_then(Value::integer).ok().and_then(|x| usize::try_from(x).ok()).unwrap_or(0)
}
fn float(v: &Value, k: &str) -> f32 {
    match v.get(k) {
        Ok(Value::Number(t)) => t.parse().unwrap_or(0.0),
        _ => 0.0,
    }
}
fn opt<'a>(args: &'a [String], k: &str) -> Option<&'a str> {
    args.iter().find_map(|a| a.strip_prefix(k).and_then(|r| r.strip_prefix('=')))
}

/// The split, fixed by id before any run: development questions with an odd number map; even
/// ones and every real-use question are held out.
fn part_of(id: &str) -> &'static str {
    match id.strip_prefix('d').and_then(|x| x.parse::<u32>().ok()) {
        Some(k) if k % 2 == 1 => "map",
        _ => "held",
    }
}

/// One candidate as the judge read it.
#[derive(Clone, Debug)]
struct Cand {
    path: String,
    start: usize,
    end: usize,
    doc: String,
}
/// One question of the judged set.
#[derive(Clone, Debug)]
struct Q {
    id: String,
    question: String,
    answers: Vec<(String, usize, usize)>,
    cands: Vec<Cand>,
}
impl Q {
    fn right(&self, c: &Cand) -> bool {
        self.answers.iter().any(|(f, a, b)| *f == c.path && c.start <= *b && c.end >= *a)
    }
}

/// A question: id, text, and its answers as (file, first line, last line).
type Asked = (String, String, Vec<(String, usize, usize)>);

/// Questions of a question file, each answer's lines moved to where its anchor is in `root`.
fn questions(root: &Path, file: &Path) -> Result<Vec<Asked>, Fault> {
    let v = nmjson::parse(&std::fs::read(file).map_err(|e| bad(format!("{}: {e}", file.display())))?).map_err(|e| bad(e.to_string()))?;
    let mut out = Vec::new();
    for q in v.get("questions").and_then(Value::array).map_err(|e| bad(e.to_string()))? {
        let id = text(q, "id");
        let mut answers = Vec::new();
        for a in q.get("answers").and_then(Value::array).unwrap_or_default() {
            let path = text(a, "file");
            let lines: Vec<usize> = a.get("lines").and_then(Value::array).unwrap_or_default().iter().filter_map(|x| x.integer().ok().and_then(|k| usize::try_from(k).ok())).collect();
            if text(a, "match") == "file" {
                answers.push((path, 1, usize::MAX));
                continue;
            }
            let body = std::fs::read(root.join(&path)).ok().and_then(|b| nmdoc_text(&b)).ok_or_else(|| bad(format!("{id}: {path} not read")))?;
            let anchor = text(a, "anchor");
            let hits: Vec<usize> = body.lines().enumerate().filter(|(_, l)| l.contains(anchor.as_str())).map(|(i, _)| i + 1).collect();
            let ([first], [a0, b0]) = (hits.as_slice(), lines.as_slice()) else {
                return Err(bad(format!("{id}: anchor in {path} found {} times", hits.len())));
            };
            answers.push((path, *first, first + (b0 - a0)));
        }
        out.push((id, text(q, "question"), answers));
    }
    Ok(out)
}
fn nmdoc_text(b: &[u8]) -> Option<String> {
    String::from_utf8(b.to_vec()).ok()
}

/// Every judged set under `scratch/base-cartridge/set-*.json`, in id order.
fn load_set(home: &Path) -> Result<Vec<Q>, Fault> {
    let mut out = Vec::new();
    let dir = home.join(DIR);
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| bad(format!("{}: {e}", dir.display())))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|f| f.to_str()).is_some_and(|f| f.starts_with("set-") && f.ends_with(".json")))
        .collect();
    files.sort();
    for f in files {
        let v = nmjson::parse(&std::fs::read(&f).map_err(|e| bad(e.to_string()))?).map_err(|e| bad(e.to_string()))?;
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
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

fn in_part<'a>(set: &'a [Q], part: &str) -> Vec<&'a Q> {
    set.iter().filter(|q| part == "all" || part_of(&q.id) == part).collect()
}

// ---------------------------------------------------------------------------------------------
// The served whole path, opened as `bench` opens it (memory off, full mode settings).

fn open_served(home: &Path, root: &Path, seat: Option<Arc<Model>>) -> Result<(), Fault> {
    let memory = home.join(DIR).join("memory");
    let org = nmapi::forage::Organism::open_with(home, root, &["neuromorph-nano".to_owned()], false, &memory)?;
    nmengn::gpu::HALF_MATH.store(true, Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
    if let Some(m) = seat
        && !nmapi::neural::seat_reranker(m)
    {
        return Err(bad("a reranker was already bound"));
    }
    nmapi::neural::bind(&org)?;
    let mut path = nmapi::whole::WholePath::new(&org.home);
    if org.max_run_ms > 0 {
        path.budget_s = f64::from(org.max_run_ms) / 1000.0;
    }
    let path = Arc::new(path);
    path.warm()?;
    if let Ok(mut w) = org.whole.write() {
        *w = Some(path);
    }
    nmapi::tools::install(org).map_err(|e| bad(format!("{}: {}", e.code, e.message)))
}

fn ask(question: &str, mode: &str) -> String {
    let input = obj(vec![("question", s(question)), ("form", s("full")), ("mode", s(mode))]);
    match nmapi::tools::api().dispatch("nm_context_pack", &input) {
        Ok(Value::String(t)) => t,
        Ok(v) => v.encode().unwrap_or_default(),
        Err(e) => format!("error {}: {}", e.code, e.message),
    }
}

/// `capture <root> <questions> <out> [ids]`: the judged sets of the served path.
fn capture(home: &Path, args: &[String]) -> Result<String, Fault> {
    let root = nmsubs::repo::canon(Path::new(args.get(3).ok_or_else(|| bad("capture <root> <questions> <out> [ids]"))?));
    let qs = questions(&root, Path::new(args.get(4).ok_or_else(|| bad("no question file"))?))?;
    let out_file = home.join(DIR).join(args.get(5).ok_or_else(|| bad("no output name"))?);
    let only: Option<BTreeSet<&str>> = args.get(6).map(|x| x.split(',').collect());
    let t0 = Instant::now();
    open_served(home, &root, None)?;
    nmtype::examples::set_level(nmtype::examples::FULL);
    let o = nmapi::tools::installed().ok_or_else(|| bad("not installed"))?;
    let reranker = nmapi::neural::reranker_model(home)?;
    let count = |t: &str| reranker.pop.tokenizer.encode(t).map_or(0, |v| v.len());
    let w = nmapi::whole::JUDGE_WINDOW.unwrap_or(usize::MAX);
    let mut rows = Vec::new();
    let mut rep = String::new();
    for (id, question, answers) in qs.iter().filter(|q| only.as_ref().is_none_or(|o| o.contains(q.0.as_str()))) {
        let t = Instant::now();
        let _ = ask(question, "full");
        let _ = nmapi::tools::take_diagnostics();
        let Some(tap) = nmtype::examples::take_tap() else {
            let _ = writeln!(rep, "{id}: no judged set (the whole path did not judge it), left out");
            continue;
        };
        let sub = o.sub();
        let terms = nmapi::whole::term_weights(&sub, question);
        let mut cands = Vec::new();
        let mut docs = Vec::new();
        for c in &tap.cands {
            let piece = nmapi::joint_read::Piece { chunk: 0, path: c.path.clone(), start: c.start, end: c.end, text: c.text.clone().unwrap_or_default(), evidence: nmapi::joint_read::Evidence::default() };
            let (v, _) = nmapi::whole::judge_view(&piece, &terms, w, &count);
            let heading = sub
                .chunks
                .iter()
                .find(|ch| ch.start == c.start && ch.end == c.end && sub.files.get(ch.file).is_some_and(|f| f.path.replace('\\', "/") == c.path))
                .map_or("", |ch| ch.heading.as_str());
            let doc = nmsubs::dense::doc_text_with(&v.path, heading, &v.text);
            docs.push(doc.clone());
            cands.push(obj(vec![
                ("path", s(&c.path)),
                ("start", n(c.start as u64)),
                ("end", n(c.end as u64)),
                ("served", Value::Number(format!("{:e}", c.score))),
                ("doc", s(&doc)),
            ]));
        }
        // The rebuilt documents scored again in one batch: how close they come to what was served.
        let again = nmengn::roles::card_rerank_batch(&reranker, Device::Gpu, nmengn::roles::CODE_INSTRUCTION, question, &docs, None)?;
        let diff = again.iter().zip(&tap.cands).map(|(a, c)| (a - c.score).abs()).fold(0.0_f32, f32::max);
        let _ = writeln!(rep, "{id} ({}): {} candidates, rebuilt documents scored again within {diff:.5} of the served scores, {:.1} s", part_of(id), cands.len(), t.elapsed().as_secs_f64());
        let ans = answers.iter().map(|(f, a, b)| obj(vec![("file", s(f)), ("first", n(*a as u64)), ("last", n((*b).min(u32::MAX as usize) as u64))])).collect();
        rows.push(obj(vec![("id", s(id)), ("part", s(part_of(id))), ("question", s(question)), ("answers", Value::Array(ans)), ("cands", Value::Array(cands))]));
    }
    let v = obj(vec![("root", s(&root.display().to_string())), ("questions", Value::Array(rows))]);
    std::fs::write(&out_file, v.encode().map_err(|e| bad(e.to_string()))?).map_err(|e| bad(e.to_string()))?;
    let _ = writeln!(rep, "written {} in {:.1} s", out_file.display(), t0.elapsed().as_secs_f64());
    Ok(rep)
}

// ---------------------------------------------------------------------------------------------
// Pruning specs and the models they make.

#[derive(Clone, Debug, Default)]
struct Spec {
    layers: Vec<usize>,
    /// Feed-forward channels kept per layer, ascending (a layer absent keeps all).
    ffn: BTreeMap<usize, Vec<usize>>,
    /// Key-value groups kept per layer (absent keeps all).
    groups: BTreeMap<usize, Vec<usize>>,
    cart: Option<PathBuf>,
    addon: Option<PathBuf>,
    /// Linear weights at 8 or 4 bits (`bits=<n>`), quantized as the engine binds them.
    bits: Option<u8>,
    /// The same for the feed-forward linears only (`fbits=<n>`) and the attention linears only
    /// (`abits=<n>`), over `bits`.
    fbits: Option<u8>,
    abits: Option<u8>,
}

/// A fixed pseudo-random order of `0..len` (xorshift, seeded).
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

/// The judge's mapping scores (Wanda and activity, `record`), with its parameter map.
fn scores_file(home: &Path) -> PathBuf {
    home.join(MAP_DIR).join("scores.json")
}
fn read_scores(home: &Path, key: &str) -> Result<Vec<Vec<f64>>, Fault> {
    let v = nmjson::parse(&std::fs::read(scores_file(home)).map_err(|e| bad(format!("scores: {e}; run `record map` first")))?).map_err(|e| bad(e.to_string()))?;
    Ok(v.get(key)
        .and_then(Value::array)
        .map_err(|e| bad(format!("scores {key}: {e}")))?
        .iter()
        .map(|row| row.array().unwrap_or_default().iter().map(|x| if let Value::Number(t) = x { t.parse().unwrap_or(0.0) } else { 0.0 }).collect())
        .collect())
}

/// The units left out of one layer: the `k` lowest by `scores[layer]`, or a seeded random `k`.
fn lowest(home: &Path, way: &str, kind: &str, layer: usize, units: usize, k: usize) -> Result<Vec<usize>, Fault> {
    if let Some(seed) = way.strip_prefix("rand") {
        return Ok(shuffle(units, seed.parse::<u64>().unwrap_or(1) * 1000 + layer as u64).into_iter().take(k).collect());
    }
    let sc = read_scores(home, &format!("{kind}_{way}"))?;
    let row = sc.get(layer).ok_or_else(|| bad(format!("no {kind}_{way} scores for layer {layer}")))?;
    let mut idx: Vec<usize> = (0..units).collect();
    idx.sort_by(|a, b| row.get(*a).unwrap_or(&0.0).total_cmp(row.get(*b).unwrap_or(&0.0)).then(a.cmp(b)));
    Ok(idx.into_iter().take(k).collect())
}

fn parse_spec(home: &Path, t: &str, layers: usize, blocks: usize, groups: usize) -> Result<Spec, Fault> {
    let mut sp = Spec { layers: (0..layers).collect(), ..Spec::default() };
    let mut ffn: Option<(String, f64)> = None;
    let mut heads: Option<(String, usize)> = None;
    for part in t.split(',').filter(|x| !x.is_empty() && *x != "base") {
        let (k, v) = part.split_once('=').ok_or_else(|| bad(format!("spec part {part}")))?;
        match k {
            "depth" => {
                let d: usize = v.parse().map_err(|_| bad(format!("depth {v}")))?;
                sp.layers.retain(|l| *l <= d);
            }
            "drop" => {
                let out: BTreeSet<usize> = v.split('+').filter_map(|x| x.parse().ok()).collect();
                sp.layers.retain(|l| !out.contains(l));
            }
            "ffn" => {
                let (w, share) = v.split_once(':').ok_or_else(|| bad(format!("ffn {v}")))?;
                ffn = Some((w.to_owned(), share.parse().map_err(|_| bad(format!("ffn share {share}")))?));
            }
            "fmap" => {
                // `fmap=<share>[:<unit>]`: the Parameter Map's kept channels at that share of all
                // feed-forward channels left out (`unit` `ch`, the default, or `blk`), or a random
                // set of the same size in every layer (`rand<seed>`).
                let mut it = v.split(':');
                let share = it.next().unwrap_or("");
                let unit = it.next().unwrap_or("ch");
                sp.ffn = map_kept(home, share, unit)?;
            }
            // `rmap=<share>`: the role map's kept channels (`role.json`) on the map's budget;
            // `flap=<share>`: FLAP's kept channels (`flap.json`), without its compensation.
            "rmap" => sp.ffn = role_kept(home, v)?,
            "flap" => sp.ffn = flap_kept(home, v)?.0,
            "heads" => {
                let (w, k) = v.split_once(':').ok_or_else(|| bad(format!("heads {v}")))?;
                heads = Some((w.to_owned(), k.parse().map_err(|_| bad(format!("heads count {k}")))?));
            }
            "bits" | "fbits" | "abits" => {
                let b = Some(v.parse().ok().filter(|b| matches!(b, 4 | 8 | 16)).ok_or_else(|| bad(format!("{k} {v}: 4, 8 or 16")))?);
                match k {
                    "bits" => sp.bits = b,
                    "fbits" => sp.fbits = b,
                    _ => sp.abits = b,
                }
            }
            "cart" => sp.cart = Some(home.join(v)),
            "addon" => sp.addon = Some(home.join(v)),
            o => return Err(bad(format!("unknown spec part {o}"))),
        }
    }
    for &l in &sp.layers {
        if let Some((w, share)) = &ffn {
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss, reason = "block counts")]
            let k = ((blocks as f64) * share).round() as usize;
            let off: BTreeSet<usize> = lowest(home, w, "ffn", l, blocks, k.min(blocks - 1))?.into_iter().collect();
            sp.ffn.insert(l, (0..blocks).filter(|b| !off.contains(b)).flat_map(|b| b * BLOCK..(b + 1) * BLOCK).collect());
        }
        if let Some((w, k)) = &heads {
            let off: BTreeSet<usize> = lowest(home, w, "kv", l, groups, (*k).min(groups - 1))?.into_iter().collect();
            sp.groups.insert(l, (0..groups).filter(|g| !off.contains(g)).collect());
        }
    }
    Ok(sp)
}

fn ranges(units: &[usize], width: usize) -> Vec<(usize, usize)> {
    let mut r: Vec<(usize, usize)> = Vec::new();
    for &u in units {
        match r.last_mut() {
            Some(x) if x.1 == u * width => x.1 = (u + 1) * width,
            _ => r.push((u * width, (u + 1) * width)),
        }
    }
    r
}

/// The assembly a spec makes over the open weights, for measurement: kept rows and columns are
/// named as regions of the released tensors. A base cartridge is never laid out by this: it is
/// seated on the MicroModel's own assembly.
fn pruned_assembly(home: &Path, asm_name: &str, sp: &Spec, layers: usize, head_dim: usize) -> Result<Assembly, Fault> {
    let mut a = Assembly::load(&home.join("designs/assemblies"), asm_name)?;
    let name = |b: u8| match b {
        4 => "q4",
        8 => "q8",
        _ => "f16",
    };
    for nd in a.nodes.iter_mut().filter(|x| x.op == "linear") {
        let ffn = [".gate", ".up", ".down"].iter().any(|s| nd.id.ends_with(s));
        if let Some(b) = if ffn { sp.fbits.or(sp.bits) } else { sp.abits.or(sp.bits) } {
            name(b).clone_into(&mut nd.precision);
        }
    }
    let per = 2; // query heads per key-value group in Qwen3-0.6B (16 over 8)
    for (l, kept) in &sp.ffn {
        a.keep_ffn_blocks(*l, kept, 1)?;
    }
    for (l, g) in &sp.groups {
        let hs: Vec<usize> = g.iter().flat_map(|x| [x * per, x * per + 1]).collect();
        let pick = |units: &[usize]| -> Vec<(usize, usize)> { units.iter().map(|u| (u * head_dim, (u + 1) * head_dim)).collect() };
        for (part, t, units, axis) in [("q", "q_proj", &hs, "rows"), ("k", "k_proj", g, "rows"), ("v", "v_proj", g, "rows"), ("o", "o_proj", &hs, "cols")] {
            let id = nid(&format!("L{l}.{part}"));
            if let Some(nd) = a.nodes.iter_mut().find(|x| x.id == id) {
                nd.clusters = pick(units).iter().map(|(x, y)| format!("model.layers.{l}.self_attn.{t}.weight/{axis}{x}-{y}")).collect();
            }
        }
        if let Some(nd) = a.nodes.iter_mut().find(|x| x.id == nid(&format!("L{l}.score"))) {
            nd.heads = Some((0, hs.len()));
            nd.kv = Some((0, g.len()));
        }
    }
    let keep: BTreeSet<usize> = sp.layers.iter().copied().collect();
    if keep.len() < layers {
        let out_of = |l: usize| -> String { (0..=l).rev().find(|k| keep.contains(k)).map_or_else(|| nid("embed"), |k| nid(&format!("L{k}.res2"))) };
        let dropped_node = |id: &str| a.nodes.iter().any(|x| x.id == id && x.layer.is_some_and(|l| !keep.contains(&l)));
        let mut edges: Vec<Edge> = Vec::new();
        for e in &a.edges {
            if dropped_node(&e.to) {
                continue;
            }
            let mut e2 = e.clone();
            if dropped_node(&e.from) {
                let layer = a.node(&e.from).and_then(|x| x.layer).unwrap_or(0);
                if !e.from.ends_with(".res2") {
                    continue;
                }
                e2.from = if layer == 0 { nid("embed") } else { out_of(layer - 1) };
            }
            edges.push(e2);
        }
        a.edges = edges;
        a.nodes.retain(|x| x.layer.is_none_or(|l| keep.contains(&l)));
    }
    a.check_ports()?;
    Ok(a)
}

/// A judge model for a spec: over the bare weights, or loaded from a cartridge file.
fn model_for(home: &Path, sp: &Spec, gpu: bool) -> Result<Model, Fault> {
    if let Some(file) = &sp.cart {
        return Ok(load_cartridge(home, file, gpu, sp.bits)?.0);
    }
    let pop = Population::open(home, POP, true, gpu)?;
    let d = pop.dims;
    let a = pruned_assembly(home, ASM, sp, d.layers, d.head_dim)?;
    Ok(Model::bind(&pop, a)?.label_rows())
}

fn spec_for(home: &Path, t: &str) -> Result<Spec, Fault> {
    // Qwen3-Reranker-0.6B: 28 layers, 3,072 feed-forward channels (48 blocks), 8 key-value groups.
    parse_spec(home, t, 28, 3072 / BLOCK, 8)
}

// ---------------------------------------------------------------------------------------------
// Add-on cartridges (gain and bias at a socket), plugged on the reranker for an arm.

fn plug_addon(file: &Path) -> Result<String, Fault> {
    let b = std::fs::read(file).map_err(|e| bad(format!("{}: {e}", file.display())))?;
    let (label, body) = nmlearn::library::read_cartridge(&b).map_err(|e| bad(e.to_string()))?;
    let mut parts = Vec::new();
    for p in label.get("parts").and_then(Value::array).unwrap_or_default() {
        if text(p, "form") != "gain-bias" {
            continue;
        }
        let w = p.get("shape").and_then(Value::array).ok().and_then(|a| a.first()?.integer().ok()).and_then(|x| usize::try_from(x).ok()).unwrap_or(0);
        let raw = body.get(int(p, "at")..int(p, "at") + int(p, "bytes")).ok_or_else(|| bad("part outside the parameters"))?;
        let (gain, bias): (Vec<f32>, Vec<f32>) = match text(p, "encoding").as_str() {
            "q8" => {
                let (gs, bs) = (float(p, "gain_scale"), float(p, "bias_scale"));
                let q: Vec<i8> = raw.iter().map(|x| i8::from_le_bytes([*x])).collect();
                let (g, bb) = q.split_at(w.min(q.len()));
                (g.iter().map(|v| 1.0 + gs * f32::from(*v)).collect(), bb.iter().map(|v| bs * f32::from(*v)).collect())
            }
            _ => {
                let xs: Vec<f32> = raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
                let (g, bb) = xs.split_at(w.min(xs.len()));
                (g.to_vec(), bb.to_vec())
            }
        };
        parts.push(nmengn::filter::Part { target: text(p, "socket"), gain, bias });
    }
    let key = format!("{}@{}", text(&label, "id"), int(&label, "version"));
    nmengn::filter::plug(POP, &[nmengn::filter::CartridgeFilter { cartridge: key.clone(), parts }]).map_err(bad)?;
    Ok(format!("{key} ({}; made for bare weights {:.12})", text(&label, "name"), text(&label, "weights")))
}

// ---------------------------------------------------------------------------------------------
// The measure.

#[derive(Clone, Debug, Default)]
struct Tally {
    judged: usize,
    first: usize,
    top3: usize,
    rr: f64,
    margin: f64,
    secs: f64,
    questions: usize,
    /// Candidates judged in all (pieces).
    pieces: usize,
    rows: Vec<String>,
}

/// Rank of the first right candidate by score (higher first, ties by order), if any is right.
fn first_right(q: &Q, sc: &[f32]) -> Option<usize> {
    let mut idx: Vec<usize> = (0..sc.len()).collect();
    idx.sort_by(|a, b| sc.get(*b).unwrap_or(&f32::MIN).total_cmp(sc.get(*a).unwrap_or(&f32::MIN)).then(a.cmp(b)));
    idx.iter().position(|i| q.cands.get(*i).is_some_and(|c| q.right(c)))
}

fn log_odds(p: f32) -> f64 {
    let p = f64::from(p).clamp(1e-7, 1.0 - 1e-7);
    (p / (1.0 - p)).ln()
}

/// One question into the tally: `sc` are P(yes) (`prob`) or log-odds, ranked as given.
fn tally_one(t: &mut Tally, q: &Q, sc: &[f32], prob: bool, secs: f64) {
    t.questions += 1;
    t.secs += secs;
    t.pieces += sc.len();
    let r = first_right(q, sc);
    let lo = |x: f32| if prob { log_odds(x) } else { f64::from(x) };
    let best = |right: bool| sc.iter().zip(&q.cands).filter(|(_, c)| q.right(c) == right).map(|(x, _)| lo(*x)).fold(f64::NEG_INFINITY, f64::max);
    if let Some(k) = r {
        t.judged += 1;
        t.first += usize::from(k == 0);
        t.top3 += usize::from(k < 3);
        #[expect(clippy::cast_precision_loss, reason = "ranks")]
        let rr = 1.0 / (k as f64 + 1.0);
        t.rr += rr;
        let m = best(true) - best(false);
        if m.is_finite() {
            t.margin += m.clamp(-20.0, 20.0);
        }
    }
    t.rows.push(format!("{} rank {} of {} {:.2}s", q.id, r.map_or("-".to_owned(), |k| (k + 1).to_string()), sc.len(), secs));
}

impl Tally {
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    fn line(&self) -> String {
        let j = self.judged.max(1) as f64;
        format!(
            "right first {} of {}, top 3 {}, mean reciprocal rank {:.3}, mean margin {:+.2} log-odds; {:.3} s per question, {:.4} s per piece judged",
            self.first,
            self.judged,
            self.top3,
            self.rr / j,
            self.margin / j,
            self.secs / self.questions.max(1) as f64,
            self.secs / self.pieces.max(1) as f64
        )
    }
}

/// Lowest free card memory while a job runs, sampled every 20 ms.
struct Sampler {
    stop: Arc<AtomicBool>,
    min_free: Arc<AtomicUsize>,
    total: Arc<AtomicUsize>,
    h: Option<std::thread::JoinHandle<()>>,
}
impl Sampler {
    fn start() -> Self {
        let (stop, min_free, total) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicUsize::new(usize::MAX)), Arc::new(AtomicUsize::new(0)));
        let (s2, m2, t2) = (Arc::clone(&stop), Arc::clone(&min_free), Arc::clone(&total));
        let h = std::thread::spawn(move || {
            while !s2.load(Ordering::Relaxed) {
                if let Some((f, t)) = nmengn::residency::get().memory_now() {
                    m2.fetch_min(f, Ordering::Relaxed);
                    t2.store(t, Ordering::Relaxed);
                }
                std::thread::park_timeout(std::time::Duration::from_millis(20));
            }
        });
        Self { stop, min_free, total, h: Some(h) }
    }
    /// Peak MiB in use on the card (driver view).
    fn peak(mut self) -> f64 {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.h.take() {
            let _ = h.join();
        }
        let (t, f) = (self.total.load(Ordering::Relaxed), self.min_free.load(Ordering::Relaxed));
        mib(t.saturating_sub(f.min(t)))
    }
}
#[expect(clippy::cast_precision_loss, reason = "bytes for a report")]
fn mib(b: usize) -> f64 {
    b as f64 / 1_048_576.0
}

fn score(m: &Model, dev: Device, q: &Q) -> Result<Vec<f32>, Fault> {
    let docs: Vec<String> = q.cands.iter().map(|c| c.doc.clone()).collect();
    nmengn::roles::card_rerank_batch(m, dev, nmengn::roles::CODE_INSTRUCTION, &q.question, &docs, None)
}

/// The head's yes-minus-no log-odds per candidate, one forward each: the same reading as P(yes)
/// without its saturation at 1.0 in binary32 (a measurement of ties, not the served path).
fn score_z(m: &Model, dev: Device, q: &Q) -> Result<Vec<f32>, Fault> {
    let head = (m.role_node("head")?, "logits".to_owned());
    q.cands
        .iter()
        .map(|c| {
            let ids = m.pop.tokenizer.encode(&nmengn::roles::card_prompt(nmengn::roles::CODE_INSTRUCTION, &q.question, &c.doc)).map_err(|e| bad(e.to_string()))?;
            let o = m.forward(dev, &ids, std::slice::from_ref(&head), &RunOpts::default())?;
            let l = values(o.get(&head).ok_or_else(|| bad("no logits"))?)?;
            Ok(l.first().copied().unwrap_or(0.0) - l.get(1).copied().unwrap_or(0.0))
        })
        .collect()
}

/// `eval <spec> <part> [cpu=<n>]`: the job's measure for one model.
fn eval(home: &Path, args: &[String]) -> Result<String, Fault> {
    let t0 = Instant::now();
    let spec_t = args.get(3).ok_or_else(|| bad("eval <spec> <part>"))?;
    let part = args.get(4).map_or("held", String::as_str);
    let set = load_set(home)?;
    let qs = in_part(&set, part);
    nmengn::gpu::HALF_MATH.store(true, Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
    let sp = spec_for(home, spec_t)?;
    let mut out = String::new();
    if let Some(a) = &sp.addon {
        let _ = writeln!(out, "add-on plugged: {}", plug_addon(a)?);
    }
    if opt(args, "q8half") == Some("off") {
        nmengn::roles::JUDGE_Q8_HALF.store(false, Ordering::Relaxed);
    }
    let cpu_n: usize = opt(args, "cpu").and_then(|x| x.parse().ok()).unwrap_or(0);
    let gpu_on = opt(args, "gpu") != Some("off");
    let z_read = args.iter().any(|x| x == "z");
    let mut tally = Tally::default();
    if gpu_on {
        let m = match &sp.cart {
            Some(file) => {
                let t = Instant::now();
                let (m, s2) = load_cartridge(home, file, true, sp.bits)?;
                let _ = writeln!(
                    out,
                    "seated: {:.2} s in all (open weights opened and bound at the selection, then {:.2} s to check, bind the cartridge, run the check: {} of {} outputs identical, largest change {:e}; open weights released)",
                    t.elapsed().as_secs_f64(),
                    s2.secs,
                    s2.identical,
                    s2.outputs,
                    s2.largest
                );
                m
            }
            None => model_for(home, &sp, true)?,
        };
        let (host_tensors, host_bytes, _) = m.holdings();
        // One untimed call first: weights go up to the card.
        if let Some(q) = qs.first() {
            score(&m, Device::Gpu, q)?;
        }
        let sampler = Sampler::start();
        for q in &qs {
            let t = Instant::now();
            let sc = if z_read { score_z(&m, Device::Gpu, q)? } else { score(&m, Device::Gpu, q)? };
            tally_one(&mut tally, q, &sc, !z_read, t.elapsed().as_secs_f64());
        }
        let peak = sampler.peak();
        let channels: usize = (0..28).filter(|l| sp.layers.contains(l)).map(|l| sp.ffn.get(&l).map_or(3072, Vec::len)).sum();
        let groups: usize = (0..28).filter(|l| sp.layers.contains(l)).map(|l| sp.groups.get(&l).map_or(8, Vec::len)).sum();
        let _ = writeln!(
            out,
            "eval {spec_t} on {part} ({} questions), card: {}\n  layers {}, feed-forward channels {channels} of 86016, key-value groups {groups} of 224; weights on the card {:.1} MiB, card peak {peak:.0} MiB (driver view), host {host_tensors} clusters {:.1} MiB",
            qs.len(),
            tally.line(),
            sp.layers.len(),
            mib(m.pop.device_bytes()),
            mib(usize::try_from(host_bytes).unwrap_or(0))
        );
    }
    if cpu_n > 0 {
        let m = model_for(home, &sp, false)?;
        let mut tc = Tally::default();
        for q in qs.iter().take(cpu_n) {
            let t = Instant::now();
            let sc = score(&m, Device::Cpu, q)?;
            tally_one(&mut tc, q, &sc, true, t.elapsed().as_secs_f64());
        }
        let _ = writeln!(out, "  processor, first {cpu_n} questions: {}", tc.line());
        if !gpu_on {
            tally = tc;
        }
    }
    let _ = writeln!(out, "  per question: {}", tally.rows.join("; "));
    let _ = writeln!(out, "total {:.1} s", t0.elapsed().as_secs_f64());
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Mapping.

fn bf16(b: &[u8], i: usize) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([b.get(2 * i).copied().unwrap_or(0), b.get(2 * i + 1).copied().unwrap_or(0)])) << 16)
}

/// The released tensor's bytes (whole tensor, BF16) from the population.
fn tensor(pop: &Population, name: &str) -> Result<(Arc<nmnpcs::View>, Vec<usize>), Fault> {
    let c = pop.clusters.iter().find(|c| c.tensor == name).ok_or_else(|| bad(format!("no tensor {name}")))?;
    Ok((pop.prov.acquire(c)?, c.shape.clone()))
}

/// Column norms of a `[rows, cols]` BF16 tensor.
fn col_norms(pop: &Population, name: &str) -> Result<Vec<f64>, Fault> {
    let (v, sh) = tensor(pop, name)?;
    let (r, c) = (sh.first().copied().unwrap_or(0), sh.get(1).copied().unwrap_or(0));
    let b = v.tensor_bytes();
    let mut out = vec![0.0_f64; c];
    for i in 0..r {
        for (j, o) in out.iter_mut().enumerate() {
            let x = f64::from(bf16(&b, i * c + j));
            *o += x * x;
        }
    }
    Ok(out.into_iter().map(f64::sqrt).collect())
}

/// Ridge readout `w` on rows `x` (with a constant) for targets `y`, by Cholesky.
fn ridge(x: &[Vec<f64>], y: &[f64], lambda: f64) -> Vec<f64> {
    let d = x.first().map_or(0, Vec::len) + 1;
    let mut a = vec![0.0_f64; d * d];
    let mut b = vec![0.0_f64; d];
    for (row, t) in x.iter().zip(y) {
        let r: Vec<f64> = row.iter().copied().chain(std::iter::once(1.0)).collect();
        for i in 0..d {
            let ri = r.get(i).copied().unwrap_or(0.0);
            if let Some(bi) = b.get_mut(i) {
                *bi += ri * t;
            }
            for j in 0..=i {
                if let Some(aij) = a.get_mut(i * d + j) {
                    *aij += ri * r.get(j).copied().unwrap_or(0.0);
                }
            }
        }
    }
    for i in 0..d.saturating_sub(1) {
        if let Some(aii) = a.get_mut(i * d + i) {
            *aii += lambda;
        }
    }
    // Cholesky, lower triangle in place.
    for j in 0..d {
        let mut sum = a.get(j * d + j).copied().unwrap_or(0.0);
        for k in 0..j {
            sum -= a.get(j * d + k).copied().unwrap_or(0.0).powi(2);
        }
        let l = sum.max(1e-12).sqrt();
        if let Some(x) = a.get_mut(j * d + j) {
            *x = l;
        }
        for i in j + 1..d {
            let mut v = a.get(i * d + j).copied().unwrap_or(0.0);
            for k in 0..j {
                v -= a.get(i * d + k).copied().unwrap_or(0.0) * a.get(j * d + k).copied().unwrap_or(0.0);
            }
            if let Some(x) = a.get_mut(i * d + j) {
                *x = v / l;
            }
        }
    }
    let mut z = b.clone();
    for i in 0..d {
        let mut v = z.get(i).copied().unwrap_or(0.0);
        for k in 0..i {
            v -= a.get(i * d + k).copied().unwrap_or(0.0) * z.get(k).copied().unwrap_or(0.0);
        }
        if let Some(x) = z.get_mut(i) {
            *x = v / a.get(i * d + i).copied().unwrap_or(1.0);
        }
    }
    for i in (0..d).rev() {
        let mut v = z.get(i).copied().unwrap_or(0.0);
        for k in i + 1..d {
            v -= a.get(k * d + i).copied().unwrap_or(0.0) * z.get(k).copied().unwrap_or(0.0);
        }
        if let Some(x) = z.get_mut(i) {
            *x = v / a.get(i * d + i).copied().unwrap_or(1.0);
        }
    }
    z
}

/// Counts of a per-candidate score over questions.
fn tally_scores(qs: &[&Q], sc: &[Vec<f64>]) -> Tally {
    let mut t = Tally::default();
    for (q, v) in qs.iter().zip(sc) {
        #[expect(clippy::cast_possible_truncation, reason = "log-odds for ranking")]
        let z: Vec<f32> = v.iter().map(|z| *z as f32).collect();
        tally_one(&mut t, q, &z, false, 0.0);
    }
    t
}

/// `record <part>`: one recording pass over the part's candidates (one forward each, every
/// layer's feed-forward and attention output and residual), giving Wanda and activity scores per
/// feed-forward block and key-value group, and the job's counts with the released head read at
/// every layer and with a ridge readout per layer fitted on half the questions, scored on the other.
#[expect(clippy::too_many_lines, reason = "one recording pass and its three readings")]
fn record(home: &Path, args: &[String]) -> Result<String, Fault> {
    let t0 = Instant::now();
    let part = args.get(3).map_or("map", String::as_str);
    if part != "map" {
        return Err(bad("mapping reads the map part only"));
    }
    let set = load_set(home)?;
    let qs = in_part(&set, part);
    nmengn::gpu::HALF_MATH.store(true, Ordering::Relaxed);
    let pop = Population::open(home, POP, true, true)?;
    let m = Model::bind(&pop, Assembly::load(&home.join("designs/assemblies"), ASM)?)?.label_rows();
    let d = pop.dims;
    let (layers, hidden, inter, hdim, kvh) = (d.layers, d.hidden, d.inter, d.head_dim, d.kv_heads);
    let label = (m.role_node("label")?, "y".to_owned());
    let mut targets = vec![label.clone()];
    for l in 0..layers {
        for p in ["swigate", "score", "res2"] {
            targets.push((nid(&format!("L{l}.{p}")), "y".to_owned()));
        }
    }
    let mut ffn_sq = vec![vec![0.0_f64; inter]; layers];
    let mut att_sq = vec![vec![0.0_f64; d.heads * hdim]; layers];
    // Last-token residual per layer per candidate, and the candidate's question index.
    let mut states: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut model_p: Vec<Vec<f32>> = Vec::new();
    let mut tokens = 0_usize;
    let fwd = Instant::now();
    for q in &qs {
        let mut ps = Vec::new();
        for c in &q.cands {
            let ids = pop.tokenizer.encode(&nmengn::roles::card_prompt(nmengn::roles::CODE_INSTRUCTION, &q.question, &c.doc)).map_err(|e| bad(e.to_string()))?;
            let o = m.forward(Device::Gpu, &ids, &targets, &RunOpts::default())?;
            tokens += ids.len();
            ps.push(values(o.get(&label).ok_or_else(|| bad("no label"))?)?.first().copied().unwrap_or(0.0));
            let mut st = Vec::with_capacity(layers);
            for l in 0..layers {
                let get = |p: &str| -> Result<&[f32], Fault> { values(o.get(&(nid(&format!("L{l}.{p}")), "y".to_owned())).ok_or_else(|| bad(format!("no L{l}.{p}")))?) };
                for (y, acc, w) in [(get("swigate")?, &mut ffn_sq, inter), (get("score")?, &mut att_sq, d.heads * hdim)] {
                    let Some(a) = acc.get_mut(l) else { continue };
                    for row in y.chunks_exact(w) {
                        a.iter_mut().zip(row).for_each(|(s2, x)| *s2 += f64::from(x * x));
                    }
                }
                let r = get("res2")?;
                st.push(r.get(r.len().saturating_sub(hidden)..).unwrap_or(&[]).to_vec());
            }
            states.push(st);
        }
        model_p.push(ps);
    }
    let fwd_s = fwd.elapsed().as_secs_f64();
    // Wanda (weight column norm times input norm) and activity (input norm) per unit.
    let mut v = BTreeMap::new();
    let (mut fw, mut fa, mut kw, mut ka) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for l in 0..layers {
        let dn = col_norms(&pop, &format!("model.layers.{l}.mlp.down_proj.weight"))?;
        let on = col_norms(&pop, &format!("model.layers.{l}.self_attn.o_proj.weight"))?;
        let xs = |a: &[f64]| a.iter().map(|x| x.sqrt()).collect::<Vec<f64>>();
        let (fx, ax) = (xs(ffn_sq.get(l).map_or(&[][..], Vec::as_slice)), xs(att_sq.get(l).map_or(&[][..], Vec::as_slice)));
        let blocks = |w: Option<&[f64]>, x: &[f64], width: usize| -> Vec<f64> {
            x.iter().enumerate().map(|(i, xi)| xi * w.and_then(|w| w.get(i)).copied().unwrap_or(1.0)).collect::<Vec<f64>>().chunks(width).map(|b| b.iter().sum()).collect()
        };
        fw.push(blocks(Some(&dn), &fx, BLOCK));
        fa.push(blocks(None, &fx, BLOCK));
        kw.push(blocks(Some(&on), &ax, hdim * d.heads / kvh));
        ka.push(blocks(None, &ax, hdim * d.heads / kvh));
    }
    let arr = |x: &[Vec<f64>]| Value::Array(x.iter().map(|r| Value::Array(r.iter().map(|y| Value::Number(format!("{y:e}"))).collect())).collect());
    v.insert("ffn_wanda".to_owned(), arr(&fw));
    v.insert("ffn_act".to_owned(), arr(&fa));
    v.insert("kv_wanda".to_owned(), arr(&kw));
    v.insert("kv_act".to_owned(), arr(&ka));
    v.insert("questions".to_owned(), Value::Array(qs.iter().map(|q| s(&q.id)).collect()));
    std::fs::write(scores_file(home), Value::Object(v).encode().map_err(|e| bad(e.to_string()))?).map_err(|e| bad(e.to_string()))?;
    let map_s = t0.elapsed().as_secs_f64() - fwd_s;
    // Depth: the released head read at each layer (final norm, then yes minus no rows), on the host.
    let (nv, _) = tensor(&pop, "model.norm.weight")?;
    let g: Vec<f32> = (0..hidden).map(|i| bf16(&nv.tensor_bytes(), i)).collect();
    let lab = m.bound.iter().find(|b| b.node.op == "labelsoftmax").map(|b| b.label_ids.clone()).unwrap_or_default();
    let (ev, _) = tensor(&pop, "model.embed_tokens.weight")?;
    let row = |r: usize| -> Vec<f32> { (0..hidden).map(|i| bf16(&ev.tensor_bytes(), r * hidden + i)).collect() };
    let (yes, no) = (row(lab.first().copied().unwrap_or(0)), row(lab.get(1).copied().unwrap_or(0)));
    let dir: Vec<f64> = yes.iter().zip(&no).zip(&g).map(|((a, b), gi)| f64::from((a - b) * gi)).collect();
    let normed = |x: &[f32]| -> Vec<f64> {
        let ms = x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / x.len().max(1) as f64;
        let r = 1.0 / (ms + f64::from(d.eps)).sqrt();
        x.iter().map(|v| f64::from(*v) * r).collect()
    };
    let mut out = String::new();
    let base = {
        let mut t = Tally::default();
        for (q, p) in qs.iter().zip(&model_p) {
            tally_one(&mut t, q, p, true, 0.0);
        }
        t
    };
    let _ = writeln!(
        out,
        "record on {part}: {} questions, {} candidates, {tokens} tokens; forward with {} recorded outputs {fwd_s:.1} s ({:.0} ms per candidate); scores {map_s:.1} s\nmodel (one at a time, full arithmetic of the recording run): {}",
        qs.len(),
        states.len(),
        targets.len() - 1,
        fwd_s * 1000.0 / states.len().max(1) as f64,
        base.line()
    );
    let half: Vec<bool> = (0..qs.len()).map(|i| i % 2 == 0).collect();
    let _ = writeln!(out, "depth (layer: released head read there | ridge readout fitted on alternate questions, scored on the others):");
    for l in 0..layers {
        let mut k = 0;
        let mut z_head = Vec::new();
        let mut feats: Vec<Vec<Vec<f64>>> = Vec::new();
        for q in &qs {
            let mut zq = Vec::new();
            let mut fq = Vec::new();
            for _ in &q.cands {
                let x = normed(states.get(k).and_then(|st| st.get(l)).map_or(&[][..], Vec::as_slice));
                zq.push(x.iter().zip(&dir).map(|(a, b)| a * b).sum::<f64>());
                fq.push(x.iter().zip(&g).map(|(a, gi)| a * f64::from(*gi)).collect());
                k += 1;
            }
            z_head.push(zq);
            feats.push(fq);
        }
        let head = tally_scores(&qs, &z_head);
        let mut z_fit = z_head.clone();
        for fold in [true, false] {
            let (mut x, mut y) = (Vec::new(), Vec::new());
            for (qi, q) in qs.iter().enumerate().filter(|(i, _)| half.get(*i) == Some(&fold)) {
                for (c, f) in q.cands.iter().zip(feats.get(qi).map_or(&[][..], Vec::as_slice)) {
                    x.push(f.clone());
                    y.push(if q.right(c) { 1.0 } else { 0.0 });
                }
            }
            let tr: f64 = x.iter().map(|r| r.iter().map(|v| v * v).sum::<f64>()).sum::<f64>() / (hidden as f64 * x.len().max(1) as f64);
            let w = ridge(&x, &y, tr * 10.0 * x.len() as f64 / 100.0);
            for (qi, zq) in z_fit.iter_mut().enumerate().filter(|(i, _)| half.get(*i) != Some(&fold)) {
                for (z, f) in zq.iter_mut().zip(feats.get(qi).map_or(&[][..], Vec::as_slice)) {
                    *z = 8.0 * (f.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>() + w.last().copied().unwrap_or(0.0));
                }
            }
        }
        let fit = tally_scores(&qs, &z_fit);
        let _ = writeln!(out, "  L{l:>2}: first {:>2} top3 {:>2} rr {:.3} | first {:>2} top3 {:>2} rr {:.3}", head.first, head.top3, head.rr / head.judged.max(1) as f64, fit.first, fit.top3, fit.rr / fit.judged.max(1) as f64);
    }
    let concentration = |x: &[Vec<f64>]| -> String {
        x.iter()
            .enumerate()
            .map(|(l, r)| {
                let mut srt = r.clone();
                srt.sort_by(|a, b| b.total_cmp(a));
                let tot: f64 = srt.iter().sum::<f64>().max(1e-12);
                format!("L{l} {:.0}%", 100.0 * srt.iter().take(srt.len() / 4).sum::<f64>() / tot)
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    let _ = writeln!(out, "feed-forward Wanda, share held by the top quarter of blocks: {}", concentration(&fw));
    let _ = writeln!(out, "key-value groups Wanda, share held by the top quarter: {}", concentration(&kw));
    let _ = writeln!(out, "scores written to {}; total {:.1} s", scores_file(home).display(), t0.elapsed().as_secs_f64());
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// The Parameter Map of the judge's role: feed-forward channels, every layer.
//
// `map record`: one pass over the map part on the open weights; per channel, its energy into the
// residual (sum over tokens of its gated activation squared, times its down column's squared norm:
// what zeroing it removes), for each half of the map questions (`a`, `b`), and z (yes minus no
// logit) per candidate. `map tol <half> <first> <last>`: tolerance per layer, measured on the
// role's own output: that layer alone with its lowest 25 and 50 percent of channels left out, and a
// random 25 percent as the control, mean |dz| over the half's candidates. `map solve`: one budget
// across layers (removals in tiles of 64 channels to the layer with the least measured drift per
// channel, interpolated between the measured points), kept channels per layer for shares of all
// channels by channel rank and by block rank, and how far the two halves agree. Files in
// `library/parameter-maps/reranker/` (owner direction 2026-11-17: parameter maps live in their proper
// folder, `library/parameter-maps/<reranker|reader|embedder|writer>/`, never in scratch).

const MAP_DIR: &str = "library/parameter-maps/reranker";
const TILE: usize = 64;
const SHARES: [u32; 8] = [5, 10, 15, 20, 25, 30, 35, 40];

fn map_file(home: &Path, name: &str) -> PathBuf {
    home.join(MAP_DIR).join(name)
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
fn idx_arr(x: &[usize]) -> Value {
    Value::Array(x.iter().map(|y| n(*y as u64)).collect())
}
/// The map questions of one half: `a` the even places of the map part, `b` the odd, `all` both.
fn half_of<'a>(qs: &[&'a Q], half: &str) -> Vec<&'a Q> {
    qs.iter().enumerate().filter(|(i, _)| half == "all" || (half == "a") == (i % 2 == 0)).map(|(_, q)| *q).collect()
}

/// The judge's z per candidate for `qs`, packed on the card as the served judge reads them.
fn zs(m: &Model, qs: &[&Q]) -> Result<Vec<Vec<f32>>, Fault> {
    qs.iter()
        .map(|q| {
            let docs: Vec<String> = q.cands.iter().map(|c| c.doc.clone()).collect();
            nmengn::roles::card_rerank_batch_z(m, nmengn::roles::CODE_INSTRUCTION, &q.question, &docs, None)
        })
        .collect()
}

/// Mean |dz| of `z` against `z0`, and the job's counts read from `z`.
fn drift(qs: &[&Q], z0: &[Vec<f32>], z: &[Vec<f32>]) -> (f64, Tally) {
    let (mut s, mut k) = (0.0, 0_usize);
    for (a, b) in z0.iter().zip(z) {
        for (x, y) in a.iter().zip(b) {
            s += f64::from((x - y).abs());
            k += 1;
        }
    }
    let mut t = Tally::default();
    for (q, v) in qs.iter().zip(z) {
        tally_one(&mut t, q, v, false, 0.0);
    }
    #[expect(clippy::cast_precision_loss, reason = "a mean")]
    let mean = s / k.max(1) as f64;
    (mean, t)
}

/// The channels of one layer in rising order of a score.
fn rising(score: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..score.len()).collect();
    idx.sort_by(|a, b| score.get(*a).unwrap_or(&0.0).total_cmp(score.get(*b).unwrap_or(&0.0)).then(a.cmp(b)));
    idx
}

/// `map record`.
fn map_record(home: &Path) -> Result<String, Fault> {
    let t0 = Instant::now();
    let set = load_set(home)?;
    let qs = in_part(&set, "map");
    nmengn::gpu::HALF_MATH.store(true, Ordering::Relaxed);
    let pop = Population::open(home, POP, true, true)?;
    let m = Model::bind(&pop, judge_assembly(home, None)?)?.label_rows();
    let d = pop.dims;
    let head = (m.role_node("head")?, "logits".to_owned());
    let mut targets = vec![head.clone()];
    targets.extend((0..d.layers).map(|l| (nid(&format!("L{l}.swigate")), "y".to_owned())));
    let mut sq = [vec![vec![0.0_f64; d.inter]; d.layers], vec![vec![0.0_f64; d.inter]; d.layers]];
    let mut z0 = Vec::new();
    let mut tokens = 0;
    for (qi, q) in qs.iter().enumerate() {
        let mut zq = Vec::new();
        for c in &q.cands {
            let ids = pop.tokenizer.encode(&nmengn::roles::card_prompt(nmengn::roles::CODE_INSTRUCTION, &q.question, &c.doc)).map_err(|e| bad(e.to_string()))?;
            tokens += ids.len();
            let o = m.forward(Device::Gpu, &ids, &targets, &RunOpts::default())?;
            let l = values(o.get(&head).ok_or_else(|| bad("no logits"))?)?;
            zq.push(l.first().copied().unwrap_or(0.0) - l.get(1).copied().unwrap_or(0.0));
            let acc = sq.get_mut(qi % 2).ok_or_else(|| bad("half"))?;
            for (l, a) in acc.iter_mut().enumerate() {
                let y = values(o.get(&(nid(&format!("L{l}.swigate")), "y".to_owned())).ok_or_else(|| bad("no swigate"))?)?;
                for row in y.chunks_exact(d.inter) {
                    a.iter_mut().zip(row).for_each(|(s2, x)| *s2 += f64::from(x * x));
                }
            }
        }
        z0.push(zq);
    }
    let mut v = BTreeMap::new();
    for (h, acc) in ["a", "b"].iter().zip(&sq) {
        let mut rows = Vec::new();
        for (l, a) in acc.iter().enumerate() {
            let dn = col_norms(&pop, &format!("model.layers.{l}.mlp.down_proj.weight"))?;
            rows.push(num_arr(&a.iter().zip(&dn).map(|(x, w)| x * w * w).collect::<Vec<_>>()));
        }
        v.insert(format!("energy_{h}"), Value::Array(rows));
    }
    v.insert("z0".to_owned(), Value::Array(z0.iter().map(|r| num_arr(&r.iter().map(|x| f64::from(*x)).collect::<Vec<_>>())).collect()));
    v.insert("questions".to_owned(), Value::Array(qs.iter().map(|q| s(&q.id)).collect()));
    std::fs::create_dir_all(home.join(MAP_DIR)).map_err(|e| bad(e.to_string()))?;
    std::fs::write(map_file(home, "record.json"), Value::Object(v).encode().map_err(|e| bad(e.to_string()))?).map_err(|e| bad(e.to_string()))?;
    let (_, t) = drift(&qs, &z0, &z0);
    Ok(format!(
        "map record: {} questions, {} candidates, {tokens} tokens, {:.1} s; open weights on the map part by z: {}\n",
        qs.len(),
        z0.iter().map(Vec::len).sum::<usize>(),
        t0.elapsed().as_secs_f64(),
        t.line()
    ))
}

/// The recorded energy per layer and channel for a half (`all` sums both).
fn energy(rec: &Value, half: &str) -> Vec<Vec<f64>> {
    let get = |h: &str| rec.get(&format!("energy_{h}")).and_then(Value::array).unwrap_or_default().iter().map(nums).collect::<Vec<_>>();
    match half {
        "all" => get("a").iter().zip(get("b")).map(|(x, y)| x.iter().zip(&y).map(|(p, q)| p + q).collect()).collect(),
        h => get(h),
    }
}

/// `map tol <half> <first> <last>`: per-layer tolerance, appended to `tol-<half>.json`.
fn map_tol(home: &Path, args: &[String]) -> Result<String, Fault> {
    let t0 = Instant::now();
    let half = args.get(4).map_or("a", String::as_str);
    let first: usize = args.get(5).and_then(|x| x.parse().ok()).unwrap_or(0);
    let last: usize = args.get(6).and_then(|x| x.parse().ok()).unwrap_or(27);
    let rec = read_json(&map_file(home, "record.json"))?;
    let set = load_set(home)?;
    let all = in_part(&set, "map");
    let qs = half_of(&all, half);
    let en = energy(&rec, half);
    nmengn::gpu::HALF_MATH.store(true, Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
    let pop = Population::open(home, POP, true, true)?;
    // The lesion, a measurement only: the left-out channels' gated activations times zero at the
    // layer's `swigate` socket, which removes exactly their terms from the down projection.
    let m = Model::bind(&pop, judge_assembly(home, None)?)?.label_rows();
    let z0 = zs(&m, &qs)?;
    let file = map_file(home, &format!("tol-{half}.json"));
    let mut tol = read_json(&file).unwrap_or(Value::Object(BTreeMap::new()));
    let mut out = String::new();
    for l in first..=last {
        let order = rising(en.get(l).map_or(&[][..], Vec::as_slice));
        let inter = order.len();
        let mut row = BTreeMap::new();
        for (key, off) in [
            ("low25", order.iter().take(inter / 4).copied().collect::<BTreeSet<usize>>()),
            ("low50", order.iter().take(inter / 2).copied().collect()),
            ("rand25", shuffle(inter, 7000 + l as u64).into_iter().take(inter / 4).collect()),
        ] {
            let gain: Vec<f32> = (0..inter).map(|c| if off.contains(&c) { 0.0 } else { 1.0 }).collect();
            let part = nmengn::filter::Part { target: format!("L{l}.swigate"), gain, bias: vec![0.0; inter] };
            let lesion = nmengn::filter::build(POP, &[nmengn::filter::CartridgeFilter { cartridge: "lesion".into(), parts: vec![part] }]).map_err(bad)?;
            m.set_filters(Some(Arc::new(lesion)));
            let z = zs(&m, &qs);
            m.set_filters(None);
            let (dz, t) = drift(&qs, &z0, &z?);
            row.insert(key.to_owned(), Value::Number(format!("{dz:e}")));
            let _ = writeln!(out, "L{l} {key}: mean |dz| {dz:.4}; {}", t.line());
        }
        if let Value::Object(o) = &mut tol {
            o.insert(format!("L{l}"), Value::Object(row));
        }
        std::fs::write(&file, tol.encode().map_err(|e| bad(e.to_string()))?).map_err(|e| bad(e.to_string()))?;
    }
    let _ = writeln!(out, "map tol {half} L{first} to L{last}: {:.1} s", t0.elapsed().as_secs_f64());
    Ok(out)
}

/// Removals per layer for a total of `remove` channels: tiles of 64 go one at a time to the layer
/// whose measured drift rises least for the tile (drift per layer interpolated through 0, 25 and
/// 50 percent, never falling as more goes: a lower reading at 50 is noise). No layer loses more
/// than the measured range (half its channels); past it nothing is known.
fn budget(tol: &[(f64, f64)], inter: usize, remove: usize) -> Vec<usize> {
    #[expect(clippy::cast_precision_loss, reason = "shares")]
    let curve = |l: usize, r: usize| -> f64 {
        let (d25, d50) = tol.get(l).copied().unwrap_or((f64::MAX, f64::MAX));
        let d50 = d50.max(d25);
        let x = r as f64 / inter as f64;
        if x <= 0.25 { d25 * x / 0.25 } else { d25 + (d50 - d25) * (x - 0.25) / 0.25 }
    };
    let measured = inter / 2;
    let mut out = vec![0_usize; tol.len()];
    let mut left = remove;
    while left >= TILE {
        let best = (0..tol.len())
            .filter(|l| out.get(*l).is_some_and(|r| r + TILE <= measured))
            .min_by(|a, b| {
                let ra = out.get(*a).copied().unwrap_or(0);
                let rb = out.get(*b).copied().unwrap_or(0);
                (curve(*a, ra + TILE) - curve(*a, ra)).total_cmp(&(curve(*b, rb + TILE) - curve(*b, rb)))
            });
        let Some(l) = best else { break };
        if let Some(r) = out.get_mut(l) {
            *r += TILE;
        }
        left -= TILE;
    }
    out
}

fn tol_of(home: &Path, half: &str, layers: usize) -> Result<Vec<(f64, f64)>, Fault> {
    let t = read_json(&map_file(home, &format!("tol-{half}.json")))?;
    (0..layers)
        .map(|l| {
            let r = t.get(&format!("L{l}")).map_err(|_| bad(format!("tol-{half}: no L{l}; run map tol {half} first")))?;
            Ok((f64::from(float(r, "low25")), f64::from(float(r, "low50"))))
        })
        .collect()
}

/// `map solve`: budgets, kept channels by channel and by block, and agreement of the halves.
#[expect(clippy::cast_precision_loss, reason = "shares and overlaps for a report")]
fn map_solve(home: &Path) -> Result<String, Fault> {
    let rec = read_json(&map_file(home, "record.json"))?;
    let (layers, inter) = (28_usize, 3072_usize);
    let total = layers * inter;
    let mut out = String::new();
    let removed_set = |en: &[Vec<f64>], b: &[usize], blocks: bool| -> Vec<BTreeSet<usize>> {
        (0..layers)
            .map(|l| {
                let e = en.get(l).map_or(&[][..], Vec::as_slice);
                let r = b.get(l).copied().unwrap_or(0);
                if blocks {
                    let bs: Vec<f64> = e.chunks(TILE).map(|c| c.iter().sum()).collect();
                    rising(&bs).into_iter().take(r / TILE).flat_map(|k| k * TILE..(k + 1) * TILE).collect()
                } else {
                    rising(e).into_iter().take(r).collect()
                }
            })
            .collect()
    };
    let mut halves = BTreeMap::new();
    for h in ["a", "b"] {
        halves.insert(h, (energy(&rec, h), tol_of(home, h, layers)?));
    }
    let tol_all: Vec<(f64, f64)> = {
        let (ta, tb) = (&halves["a"].1, &halves["b"].1);
        ta.iter().zip(tb).map(|(x, y)| (f64::midpoint(x.0, y.0), f64::midpoint(x.1, y.1))).collect()
    };
    let en_all = energy(&rec, "all");
    let _ = writeln!(out, "tolerance per layer (mean |dz| at 25 / 50 percent lowest out, random 25 percent; halves a, b):");
    let (ra, rb) = (read_json(&map_file(home, "tol-a.json"))?, read_json(&map_file(home, "tol-b.json"))?);
    let mut beat = 0;
    for l in 0..layers {
        let g = |t: &Value, k: &str| t.get(&format!("L{l}")).map(|r| float(r, k)).unwrap_or(0.0);
        beat += usize::from(g(&ra, "low25") < g(&ra, "rand25")) + usize::from(g(&rb, "low25") < g(&rb, "rand25"));
        let _ = writeln!(out, "  L{l:>2}: {:.3} / {:.3}, rand {:.3} | {:.3} / {:.3}, rand {:.3}", g(&ra, "low25"), g(&ra, "low50"), g(&ra, "rand25"), g(&rb, "low25"), g(&rb, "low50"), g(&rb, "rand25"));
    }
    let _ = writeln!(out, "the lowest-energy quarter moved z less than a random quarter in {beat} of {} layer tests", 2 * layers);
    let mut v = BTreeMap::new();
    for share in SHARES {
        let remove = total * share as usize / 100;
        let b_all = budget(&tol_all, inter, remove);
        let (ba, bb) = (budget(&halves["a"].1, inter, remove), budget(&halves["b"].1, inter, remove));
        let (sa, sb) = (removed_set(&halves["a"].0, &ba, false), removed_set(&halves["b"].0, &bb, false));
        let (mut inter_n, mut union_n) = (0_usize, 0_usize);
        for (x, y) in sa.iter().zip(&sb) {
            inter_n += x.intersection(y).count();
            union_n += x.union(y).count();
        }
        let diff: usize = ba.iter().zip(&bb).map(|(x, y)| x.abs_diff(*y)).sum();
        let kept = |rm: &[BTreeSet<usize>]| Value::Array(rm.iter().map(|r| idx_arr(&(0..inter).filter(|c| !r.contains(c)).collect::<Vec<_>>())).collect());
        let mut o = BTreeMap::new();
        o.insert("budget".to_owned(), idx_arr(&b_all));
        o.insert("ch".to_owned(), kept(&removed_set(&en_all, &b_all, false)));
        o.insert("blk".to_owned(), kept(&removed_set(&en_all, &b_all, true)));
        v.insert(share.to_string(), Value::Object(o));
        let _ = writeln!(
            out,
            "share {share}%: {remove} channels out; per layer {:?}; halves agree on {:.0}% of the channels left out (overlap over union), budgets differ by {diff} channels in all",
            b_all,
            100.0 * inter_n as f64 / union_n.max(1) as f64
        );
    }
    std::fs::write(map_file(home, "map.json"), Value::Object(v).encode().map_err(|e| bad(e.to_string()))?).map_err(|e| bad(e.to_string()))?;
    Ok(out)
}

/// `map role`: each feed-forward channel's effect on the judge's own read. A channel's term in the
/// residual stream is its gated activation times its column of the down projection; read through
/// the final norm and the head's yes-minus-no row, its direct share of z at the last token is
/// `a_c * d_c / rms`, with `d_c = sum_i down[i, c] * norm[i] * (head[yes, i] - head[no, i])` and
/// `rms` the final residual's root mean square. What matters for the role is how much that share
/// moves between the candidates of one question (a share equal on all candidates ranks nothing):
/// the score is its variance within each question, summed over the map part's questions.
/// Written to `role.json` (one row of scores per layer).
fn map_role(home: &Path) -> Result<String, Fault> {
    let t0 = Instant::now();
    let set = load_set(home)?;
    let qs = in_part(&set, "map");
    nmengn::gpu::HALF_MATH.store(true, Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
    let pop = Population::open(home, POP, true, true)?;
    let m = Model::bind(&pop, judge_assembly(home, None)?)?.label_rows();
    let d = pop.dims;
    let head = m.bound.iter().find(|b| b.node.op == "head").ok_or_else(|| bad("no head"))?;
    let head_t = head.node.clusters.first().and_then(|c| c.split('/').next()).ok_or_else(|| bad("head tensor"))?.to_owned();
    let labels = m.bound.iter().find(|b| b.node.op == "labelsoftmax").map(|b| b.label_ids.clone()).ok_or_else(|| bad("no label read"))?;
    let (&yes, &no) = (labels.first().ok_or_else(|| bad("labels"))?, labels.get(1).ok_or_else(|| bad("labels"))?);
    let (hv, _) = tensor(&pop, &head_t)?;
    let (nv, _) = tensor(&pop, "model.norm.weight")?;
    let h = d.hidden;
    let gu: Vec<f64> = (0..h).map(|i| f64::from(bf16(&nv.tensor_bytes(), i)) * (f64::from(bf16(&hv.tensor_bytes(), yes * h + i)) - f64::from(bf16(&hv.tensor_bytes(), no * h + i)))).collect();
    let mut dir = vec![vec![0.0_f64; d.inter]; d.layers];
    for (l, row) in dir.iter_mut().enumerate() {
        let (v, _) = tensor(&pop, &format!("model.layers.{l}.mlp.down_proj.weight"))?;
        let b = v.tensor_bytes();
        for (i, g) in gu.iter().enumerate() {
            for (c, o) in row.iter_mut().enumerate() {
                *o += f64::from(bf16(&b, i * d.inter + c)) * g;
            }
        }
    }
    let res = (nid(&format!("L{}.res2", d.layers - 1)), "y".to_owned());
    let mut targets = vec![res.clone()];
    targets.extend((0..d.layers).map(|l| (nid(&format!("L{l}.swigate")), "y".to_owned())));
    let mut score = vec![vec![0.0_f64; d.inter]; d.layers];
    let mut cands = 0;
    for q in &qs {
        // Per layer and channel: sum and sum of squares of the share over this question's candidates.
        let mut acc = vec![vec![(0.0_f64, 0.0_f64); d.inter]; d.layers];
        for c in &q.cands {
            let ids = pop.tokenizer.encode(&nmengn::roles::card_prompt(nmengn::roles::CODE_INSTRUCTION, &q.question, &c.doc)).map_err(|e| bad(e.to_string()))?;
            let o = m.forward(Device::Gpu, &ids, &targets, &RunOpts::default())?;
            let x = values(o.get(&res).ok_or_else(|| bad("no residual"))?)?;
            let last = x.get(x.len().saturating_sub(h)..).unwrap_or_default();
            #[expect(clippy::cast_precision_loss, reason = "a mean")]
            let rms = (last.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / h as f64 + f64::from(d.eps)).sqrt();
            for (l, a) in acc.iter_mut().enumerate() {
                let y = values(o.get(&(nid(&format!("L{l}.swigate")), "y".to_owned())).ok_or_else(|| bad("no swigate"))?)?;
                let row = y.get(y.len().saturating_sub(d.inter)..).unwrap_or_default();
                for ((s2, act), dc) in a.iter_mut().zip(row).zip(dir.get(l).map_or(&[][..], Vec::as_slice)) {
                    let share = f64::from(*act) * dc / rms;
                    s2.0 += share;
                    s2.1 += share * share;
                }
            }
            cands += 1;
        }
        #[expect(clippy::cast_precision_loss, reason = "a variance")]
        let n = q.cands.len().max(1) as f64;
        for (sl, al) in score.iter_mut().zip(&acc) {
            for (s2, (sum, sq)) in sl.iter_mut().zip(al) {
                *s2 += (sq / n - (sum / n).powi(2)).max(0.0);
            }
        }
    }
    let v = Value::Array(score.iter().map(|r| num_arr(r)).collect());
    std::fs::write(map_file(home, "role.json"), v.encode().map_err(|e| bad(e.to_string()))?).map_err(|e| bad(e.to_string()))?;
    // How far the role ranking and the energy ranking agree on each layer's lowest quarter.
    let rec = read_json(&map_file(home, "record.json"))?;
    let en = energy(&rec, "all");
    let mut agree = Vec::new();
    for l in 0..d.layers {
        let q4 = d.inter / 4;
        let a: BTreeSet<usize> = rising(score.get(l).map_or(&[][..], Vec::as_slice)).into_iter().take(q4).collect();
        let b: BTreeSet<usize> = rising(en.get(l).map_or(&[][..], Vec::as_slice)).into_iter().take(q4).collect();
        agree.push(a.intersection(&b).count() * 100 / q4.max(1));
    }
    Ok(format!(
        "map role: {} questions, {cands} candidates, {:.1} s; lowest quarter shared with the energy ranking, percent per layer: {agree:?}\n",
        qs.len(),
        t0.elapsed().as_secs_f64()
    ))
}

/// The kept channels per layer at `share` percent out, the map's budget per layer filled by the
/// role scores (`role.json`) in place of the recorded energy.
fn role_kept(home: &Path, share: &str) -> Result<BTreeMap<usize, Vec<usize>>, Fault> {
    let m = read_json(&map_file(home, "map.json"))?;
    let budget = m.get(share).and_then(|x| x.get("budget")).map(nums).map_err(|_| bad(format!("map.json has no share {share}")))?;
    let sc = read_json(&map_file(home, "role.json"))?;
    let rows: Vec<Vec<f64>> = sc.array().unwrap_or_default().iter().map(nums).collect();
    let mut out = BTreeMap::new();
    for (l, r) in budget.iter().enumerate() {
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "channel counts")]
        let r = *r as usize;
        if r == 0 {
            continue;
        }
        let off: BTreeSet<usize> = rising(rows.get(l).map_or(&[][..], Vec::as_slice)).into_iter().take(r).collect();
        out.insert(l, (0..3072).filter(|c| !off.contains(c)).collect());
    }
    Ok(out)
}

/// The kept channels per layer the map gives at `share` percent out: by channel (`ch`), by block of
/// 64 (`blk`), or a seeded random set of the same size in every layer (`rand<seed>`).
fn map_kept(home: &Path, share: &str, unit: &str) -> Result<BTreeMap<usize, Vec<usize>>, Fault> {
    let m = read_json(&map_file(home, "map.json"))?;
    let at = m.get(share).map_err(|_| bad(format!("map.json has no share {share}; run map solve")))?;
    let lists = |k: &str| -> Vec<Vec<usize>> { at.get(k).and_then(Value::array).unwrap_or_default().iter().map(|r| nums(r).iter().map(|x| *x as usize).collect()).collect() };
    let rows = if let Some(seed) = unit.strip_prefix("rand") {
        let seed: u64 = seed.parse().unwrap_or(1);
        lists("ch").iter().enumerate().map(|(l, r)| { let mut k: Vec<usize> = shuffle(3072, seed * 1000 + l as u64).into_iter().take(r.len()).collect(); k.sort_unstable(); k }).collect()
    } else {
        lists(unit)
    };
    Ok(rows.into_iter().enumerate().filter(|(_, r)| r.len() < 3072).collect())
}

// ---------------------------------------------------------------------------------------------
// Separating a channel set from a random one: z per candidate on every judged question, per arm.

/// The kept channels an arm names: `open` (none left out), `map:<share>` (the map's channels),
/// `rand:<share>:<seed>` (a random set of the same size in every layer).
fn arm_kept(home: &Path, arm: &str) -> Result<BTreeMap<usize, Vec<usize>>, Fault> {
    let p: Vec<&str> = arm.split(':').collect();
    match p.as_slice() {
        ["open"] => Ok(BTreeMap::new()),
        ["map", share] => map_kept(home, share, "ch"),
        ["role", share] => role_kept(home, share),
        ["rand", share, seed] => map_kept(home, share, &format!("rand{seed}")),
        ["flap" | "flapnc", share] => Ok(flap_kept(home, share)?.0),
        _ => Err(bad(format!("arm {arm}: open | map:<share> | role:<share> | flap:<share> | flapnc:<share> | rand:<share>:<seed>"))),
    }
}
/// The bias an arm puts in place of each left-out channel at its layer's `swigate` socket: FLAP's
/// compensation (`flap:`) is the channel's mean output, so the down projection keeps its mean term;
/// every other arm puts nothing.
fn arm_bias(home: &Path, arm: &str) -> Result<BTreeMap<usize, Vec<f32>>, Fault> {
    match arm.split(':').collect::<Vec<_>>().as_slice() {
        ["flap", share] => Ok(flap_kept(home, share)?.1),
        _ => Ok(BTreeMap::new()),
    }
}

/// `map flap`: FLAP's fluctuation score of every feed-forward channel over the map part, all tokens:
/// the variance of its gated output times the squared norm of its down column (how much its term
/// moves around its mean), and its mean output (the compensation). Written to `flap.json`.
fn map_flap(home: &Path) -> Result<String, Fault> {
    let t0 = Instant::now();
    let set = load_set(home)?;
    let qs = in_part(&set, "map");
    nmengn::gpu::HALF_MATH.store(true, Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
    let pop = Population::open(home, POP, true, true)?;
    let m = Model::bind(&pop, judge_assembly(home, None)?)?.label_rows();
    let d = pop.dims;
    let targets: Vec<(String, String)> = (0..d.layers).map(|l| (nid(&format!("L{l}.swigate")), "y".to_owned())).collect();
    let mut sum = vec![vec![0.0_f64; d.inter]; d.layers];
    let mut sq = vec![vec![0.0_f64; d.inter]; d.layers];
    let mut tokens = 0_usize;
    for q in &qs {
        for c in &q.cands {
            let ids = pop.tokenizer.encode(&nmengn::roles::card_prompt(nmengn::roles::CODE_INSTRUCTION, &q.question, &c.doc)).map_err(|e| bad(e.to_string()))?;
            let o = m.forward(Device::Gpu, &ids, &targets, &RunOpts::default())?;
            for (l, t) in targets.iter().enumerate() {
                let y = values(o.get(t).ok_or_else(|| bad("no swigate"))?)?;
                let (sl, ql) = (sum.get_mut(l).ok_or_else(|| bad("layer"))?, sq.get_mut(l).ok_or_else(|| bad("layer"))?);
                for row in y.chunks_exact(d.inter) {
                    for ((s2, q2), x) in sl.iter_mut().zip(ql.iter_mut()).zip(row) {
                        let x = f64::from(*x);
                        *s2 += x;
                        *q2 += x * x;
                    }
                }
            }
            tokens += ids.len();
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "a mean over tokens")]
    let n = tokens.max(1) as f64;
    let (mut means, mut scores) = (Vec::new(), Vec::new());
    for l in 0..d.layers {
        let w = col_norms(&pop, &format!("model.layers.{l}.mlp.down_proj.weight"))?;
        let (sl, ql) = (sum.get(l).map_or(&[][..], Vec::as_slice), sq.get(l).map_or(&[][..], Vec::as_slice));
        means.push(num_arr(&sl.iter().map(|x| x / n).collect::<Vec<_>>()));
        scores.push(num_arr(&sl.iter().zip(ql).zip(&w).map(|((s2, q2), wc)| ((q2 / n) - (s2 / n).powi(2)).max(0.0) * wc * wc).collect::<Vec<_>>()));
    }
    let v = obj(vec![("mean", Value::Array(means)), ("score", Value::Array(scores))]);
    std::fs::write(map_file(home, "flap.json"), v.encode().map_err(|e| bad(e.to_string()))?).map_err(|e| bad(e.to_string()))?;
    Ok(format!("map flap: {} questions, {tokens} tokens, {:.1} s\n", qs.len(), t0.elapsed().as_secs_f64()))
}

/// FLAP's removal at `share` percent of all feed-forward channels: each layer's scores standardised
/// (minus the layer's mean, over its standard deviation) so layers compare, the lowest across the
/// whole model taken out, no layer losing more than half its channels (the range the per-layer
/// tolerance measured; a layer is never emptied). Returns the kept channels and the compensation
/// (each left-out channel's mean output) per layer.
#[expect(clippy::type_complexity, reason = "kept channels and compensation per layer")]
fn flap_kept(home: &Path, share: &str) -> Result<(BTreeMap<usize, Vec<usize>>, BTreeMap<usize, Vec<f32>>), Fault> {
    let v = read_json(&map_file(home, "flap.json"))?;
    let rows = |k: &str| -> Vec<Vec<f64>> { v.get(k).and_then(Value::array).unwrap_or_default().iter().map(nums).collect() };
    let (means, scores) = (rows("mean"), rows("score"));
    let inter = 3072_usize;
    let pct: f64 = share.parse().map_err(|_| bad(format!("share {share}")))?;
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss, reason = "a channel count")]
    let remove = (pct / 100.0 * (scores.len() * inter) as f64).round() as usize;
    let mut pool: Vec<(f64, usize, usize)> = Vec::new();
    for (l, r) in scores.iter().enumerate() {
        #[expect(clippy::cast_precision_loss, reason = "a mean")]
        let k = r.len().max(1) as f64;
        let mu = r.iter().sum::<f64>() / k;
        let sd = (r.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / k).sqrt().max(1e-30);
        pool.extend(r.iter().enumerate().map(|(c, x)| ((x - mu) / sd, l, c)));
    }
    pool.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut off: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    let mut taken = 0;
    for (_, l, c) in pool {
        if taken >= remove {
            break;
        }
        let e = off.entry(l).or_default();
        if e.len() < inter / 2 {
            e.insert(c);
            taken += 1;
        }
    }
    let mut kept = BTreeMap::new();
    let mut bias = BTreeMap::new();
    for (l, o) in &off {
        kept.insert(*l, (0..inter).filter(|c| !o.contains(c)).collect());
        let mean = means.get(*l).map_or(&[][..], Vec::as_slice);
        #[expect(clippy::cast_possible_truncation, reason = "a bias at binary32")]
        bias.insert(*l, (0..inter).map(|c| if o.contains(&c) { mean.get(c).copied().unwrap_or(0.0) as f32 } else { 0.0 }).collect());
    }
    Ok((kept, bias))
}
fn zsep_file(home: &Path, arm: &str) -> PathBuf {
    map_file(home, &format!("zsep-{}.json", arm.replace(':', "-")))
}

/// `zsep run <arm>[,<arm>...]`: z per candidate on every judged question (both parts) with the
/// arm's left-out channels zeroed at each layer's `swigate` socket (the same terms the narrowed
/// tensors drop), packed on the card as the served judge reads them. One file per arm.
fn zsep_run(home: &Path, args: &[String]) -> Result<String, Fault> {
    let t0 = Instant::now();
    let arms: Vec<&str> = args.get(4).map_or("open", String::as_str).split(',').collect();
    let set = load_set(home)?;
    let qs = in_part(&set, "all");
    nmengn::gpu::HALF_MATH.store(true, Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
    let pop = Population::open(home, POP, true, true)?;
    let m = Model::bind(&pop, judge_assembly(home, None)?)?.label_rows();
    let mut out = String::new();
    for arm in arms {
        let t = Instant::now();
        let kept = arm_kept(home, arm)?;
        let biases = arm_bias(home, arm)?;
        let parts: Vec<nmengn::filter::Part> = kept
            .iter()
            .map(|(l, k)| {
                let keep: BTreeSet<usize> = k.iter().copied().collect();
                let gain = (0..3072).map(|c| if keep.contains(&c) { 1.0 } else { 0.0 }).collect();
                let bias = biases.get(l).cloned().unwrap_or_else(|| vec![0.0; 3072]);
                nmengn::filter::Part { target: format!("L{l}.swigate"), gain, bias }
            })
            .collect();
        let out_n: usize = kept.values().map(|k| 3072 - k.len()).sum();
        if parts.is_empty() {
            m.set_filters(None);
        } else {
            let f = nmengn::filter::build(POP, &[nmengn::filter::CartridgeFilter { cartridge: "lesion".into(), parts }]).map_err(bad)?;
            m.set_filters(Some(Arc::new(f)));
        }
        let z = zs(&m, &qs);
        m.set_filters(None);
        let z = z?;
        let v: BTreeMap<String, Value> = qs.iter().zip(&z).map(|(q, r)| (q.id.clone(), num_arr(&r.iter().map(|x| f64::from(*x)).collect::<Vec<_>>()))).collect();
        std::fs::write(zsep_file(home, arm), Value::Object(v).encode().map_err(|e| bad(e.to_string()))?).map_err(|e| bad(e.to_string()))?;
        let _ = writeln!(out, "zsep {arm}: {out_n} channels out, {} questions, {:.1} s", qs.len(), t.elapsed().as_secs_f64());
    }
    let _ = writeln!(out, "total {:.1} s", t0.elapsed().as_secs_f64());
    Ok(out)
}

/// Per-question measures of one arm's z: best right minus best wrong (z-margin), the mean over
/// every right and wrong pair of the z difference and of the P(yes) difference (what the greedy
/// at G2's inclusion edge compares once the judge's chance replaces the drones'), and the rank of
/// the first right piece. `None` when the question has no right or no wrong candidate.
fn q_measures(q: &Q, z: &[f64]) -> Option<[f64; 4]> {
    let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
    let (mut r, mut w) = (Vec::new(), Vec::new());
    for (c, x) in q.cands.iter().zip(z) {
        if q.right(c) { r.push(*x) } else { w.push(*x) }
    }
    if r.is_empty() || w.is_empty() {
        return None;
    }
    let best = |v: &[f64]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let (mut dz, mut dp) = (0.0, 0.0);
    for a in &r {
        for b in &w {
            dz += a - b;
            dp += sig(*a) - sig(*b);
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "pair counts")]
    let pairs = (r.len() * w.len()) as f64;
    let br = best(&r);
    #[expect(clippy::cast_precision_loss, reason = "a rank")]
    let rank = w.iter().filter(|x| **x > br).count() as f64;
    Some([br - best(&w), dz / pairs, dp / pairs, rank])
}

/// Mean, standard error and sign count of paired differences.
#[expect(clippy::cast_precision_loss, reason = "statistics for a report")]
fn paired(d: &[f64]) -> (f64, f64, usize) {
    let n = d.len().max(1) as f64;
    let m = d.iter().sum::<f64>() / n;
    let var = d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (m, (var / n).sqrt(), d.iter().filter(|x| **x > 0.0).count())
}

/// `zsep report`: every arm against the open weights and the map against the random draws of the
/// same share, per part (`held` never tuned on; `map` is where the map was made).
#[expect(clippy::cast_precision_loss, reason = "statistics for a report")]
fn zsep_report(home: &Path) -> Result<String, Fault> {
    let set = load_set(home)?;
    let mut arms: BTreeMap<String, BTreeMap<String, Vec<f64>>> = BTreeMap::new();
    for e in std::fs::read_dir(home.join(MAP_DIR)).map_err(|e| bad(e.to_string()))?.flatten() {
        let f = e.file_name().to_string_lossy().into_owned();
        if let Some(a) = f.strip_prefix("zsep-").and_then(|x| x.strip_suffix(".json")) {
            let v = read_json(&e.path())?;
            let Value::Object(o) = v else { continue };
            arms.insert(a.to_owned(), o.iter().map(|(k, x)| (k.clone(), nums(x))).collect());
        }
    }
    let open = arms.get("open").ok_or_else(|| bad("no zsep-open.json; run zsep run open"))?.clone();
    let names = ["z-margin (best right minus best wrong)", "pair z margin (right minus wrong, all pairs)", "pair P(yes) margin", "wrong pieces above the best right"];
    let mut out = String::new();
    for part in ["held", "map"] {
        let qs: Vec<&Q> = in_part(&set, part).into_iter().filter(|q| open.get(&q.id).and_then(|z| q_measures(q, z)).is_some()).collect();
        let _ = writeln!(out, "part {part}: {} questions with a right and a wrong candidate", qs.len());
        let meas = |arm: &BTreeMap<String, Vec<f64>>| -> Vec<[f64; 4]> { qs.iter().map(|q| arm.get(&q.id).and_then(|z| q_measures(q, z)).unwrap_or([f64::NAN; 4])).collect() };
        let m0 = meas(&open);
        for (a, z) in &arms {
            let m = meas(z);
            let mut row = format!("  {a:<12}");
            for (k, _) in names.iter().enumerate() {
                let mean = m.iter().map(|x| x[k]).sum::<f64>() / qs.len().max(1) as f64;
                let _ = write!(row, " | {mean:+.3}");
            }
            let (mut s, mut c) = (0.0, 0_usize);
            for q in &qs {
                if let (Some(x), Some(y)) = (open.get(&q.id), z.get(&q.id)) {
                    for (p, r) in x.iter().zip(y) {
                        s += (p - r).abs();
                        c += 1;
                    }
                }
            }
            let first = m.iter().filter(|x| x[3] == 0.0).count();
            let _ = writeln!(row, " | mean |dz| {:.3} | right first {first}", s / c.max(1) as f64);
            out.push_str(&row);
        }
        // The map against random draws of the same share: per question, the map's loss against the
        // open weights minus the mean random loss; positive means the map kept more of the measure.
        let shares: BTreeSet<String> = arms.keys().filter_map(|a| a.strip_prefix("map-").map(str::to_owned)).collect();
        for sh in shares {
            let draws: Vec<&BTreeMap<String, Vec<f64>>> = arms.iter().filter(|(a, _)| a.starts_with(&format!("rand-{sh}-"))).map(|(_, z)| z).collect();
            if draws.is_empty() {
                continue;
            }
            let md: Vec<Vec<[f64; 4]>> = draws.iter().map(|z| meas(z)).collect();
            let _ = writeln!(out, "  {sh}% against {} random draws of the same size per layer:", draws.len());
            // Mean |dz| per question against the open weights; positive means the map moved z less.
            let qdz = |z: &BTreeMap<String, Vec<f64>>, q: &Q| -> f64 {
                let (a, b) = (open.get(&q.id).cloned().unwrap_or_default(), z.get(&q.id).cloned().unwrap_or_default());
                a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum::<f64>() / a.len().max(1) as f64
            };
            let others: Vec<(String, &BTreeMap<String, Vec<f64>>)> =
                arms.iter().filter(|(a, _)| a.ends_with(&format!("-{sh}")) && !a.starts_with("rand-")).map(|(a, z)| (a.clone(), z)).collect();
            for (name, z) in others {
                let mm = meas(z);
                let d: Vec<f64> = qs.iter().map(|q| draws.iter().map(|r| qdz(r, q)).sum::<f64>() / draws.len() as f64 - qdz(z, q)).collect();
                let (mean, se, pos) = paired(&d);
                let below = draws.iter().filter(|r| qs.iter().map(|q| qdz(z, q)).sum::<f64>() < qs.iter().map(|q| qdz(r, q)).sum::<f64>()).count();
                let _ = writeln!(out, "    {name} mean |dz|: random minus it {mean:+.3} (standard error {se:.3}, t {:.2}), it moved z less on {pos} of {} questions, below {below} of {} draws", mean / se.max(1e-9), qs.len(), draws.len());
                for (k, nm) in names.iter().enumerate().take(3) {
                    let d: Vec<f64> = (0..qs.len()).map(|i| mm[i][k] - md.iter().map(|r| r[i][k]).sum::<f64>() / md.len() as f64).collect();
                    let (mean, se, pos) = paired(&d);
                    let draws_beaten = md.iter().filter(|r| mm.iter().map(|x| x[k]).sum::<f64>() > r.iter().map(|x| x[k]).sum::<f64>()).count();
                    let lost_map = m0.iter().zip(&mm).map(|(o, x)| o[k] - x[k]).sum::<f64>() / qs.len().max(1) as f64;
                    let lost_rand = md.iter().map(|r| m0.iter().zip(r).map(|(o, x)| o[k] - x[k]).sum::<f64>()).sum::<f64>() / (qs.len().max(1) * md.len()) as f64;
                    let _ = writeln!(
                        out,
                        "      {nm}: it minus random {mean:+.3} (standard error {se:.3}, t {:.2}), higher on {pos} of {} questions, above {draws_beaten} of {} draws; lost against the open weights: it {lost_map:.3}, random {lost_rand:.3}",
                        mean / se.max(1e-9),
                        qs.len(),
                        md.len()
                    );
                }
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// The cartridge file.

/// The kept rows of a `[rows, cols]` BF16 tensor (`rows` as ranges), or its kept columns.
fn cut(b: &[u8], cols: usize, keep: &[(usize, usize)], by_cols: bool, rows: usize) -> Vec<u8> {
    let mut out = Vec::new();
    if by_cols {
        for r in 0..rows {
            for (a, e) in keep {
                out.extend_from_slice(b.get(2 * (r * cols + a)..2 * (r * cols + e)).unwrap_or(&[]));
            }
        }
    } else {
        for (a, e) in keep {
            out.extend_from_slice(b.get(2 * a * cols..2 * e * cols).unwrap_or(&[]));
        }
    }
    out
}

/// `len` bytes of a file from `offset` (a parameter set's place in a cartridge file).
fn read_at(file: &Path, offset: u64, len: u64) -> Result<Vec<u8>, Fault> {
    use std::io::{Read as _, Seek as _};
    let mut f = std::fs::File::open(file).map_err(|e| bad(format!("{}: {e}", file.display())))?;
    f.seek(std::io::SeekFrom::Start(offset)).map_err(|e| bad(e.to_string()))?;
    let mut b = vec![0_u8; usize::try_from(len).map_err(|e| bad(e.to_string()))?];
    f.read_exact(&mut b).map_err(|e| bad(format!("{}: {e}", file.display())))?;
    Ok(b)
}

/// `siblings`: the base model, the judge and the embedder compared tensor by tensor (processor
/// only): identical bytes, or the relative difference of the values, with the bytes involved.
fn siblings(home: &Path) -> Result<String, Fault> {
    let t0 = Instant::now();
    let base = Population::open(home, "qwen3-0.6b", true, false)?;
    let mut out = String::new();
    for other_id in [POP, "qwen3-embedding-0.6b"] {
        let other = Population::open(home, other_id, true, false)?;
        let (mut same, mut same_b, mut close, mut close_b, mut differ, mut absent) = (0, 0_u64, 0, 0_u64, 0, 0);
        let mut same_names: Vec<String> = Vec::new();
        let mut rel_by_kind: BTreeMap<String, (f64, usize)> = BTreeMap::new();
        for td in &base.tensors {
            let Some(od) = other.tensors.iter().find(|x| x.name == td.name) else {
                absent += 1;
                continue;
            };
            let (a, _) = tensor(&base, &td.name)?;
            let (b, _) = tensor(&other, &od.name)?;
            let (x, y) = (a.tensor_bytes(), b.tensor_bytes());
            let bytes = x.len() as u64;
            if x == y {
                same += 1;
                same_b += bytes;
                same_names.push(td.name.clone());
                continue;
            }
            if x.len() != y.len() {
                differ += 1;
                continue;
            }
            let (mut num, mut den) = (0.0_f64, 0.0_f64);
            for i in 0..x.len() / 2 {
                let (p, q) = (f64::from(bf16(&x, i)), f64::from(bf16(&y, i)));
                num += (p - q) * (p - q);
                den += p * p;
            }
            let rel = (num / den.max(1e-30)).sqrt();
            let kind = td.name.rsplit('.').nth(1).unwrap_or("").to_owned();
            let e = rel_by_kind.entry(kind).or_insert((0.0, 0));
            e.0 += rel;
            e.1 += 1;
            if rel < 0.01 {
                close += 1;
                close_b += bytes;
            } else {
                differ += 1;
            }
        }
        let kinds = rel_by_kind.iter().map(|(k, (s2, c))| format!("{k} {:.4}", s2 / *c as f64)).collect::<Vec<_>>().join(", ");
        let names = |pre: &str| same_names.iter().filter(|x| x.contains(pre)).count();
        let _ = writeln!(
            out,
            "base model against {other_id}: {same} tensors identical ({:.2} MiB: token table {}, norms {}, others {}), {close} within 1% relative ({:.1} MiB), {differ} further apart, {absent} absent there; mean relative difference by kind: {kinds}",
            mib(usize::try_from(same_b).unwrap_or(0)),
            names("embed_tokens"),
            same_names.iter().filter(|x| x.contains("norm")).count(),
            same - names("embed_tokens") - same_names.iter().filter(|x| x.contains("norm")).count(),
            mib(usize::try_from(close_b).unwrap_or(0))
        );
    }
    let _ = writeln!(out, "total {:.1} s", t0.elapsed().as_secs_f64());
    Ok(out)
}

/// Parameter tensors resident in this process for the judge, by source (from the engine's
/// residency accounting): (source, tensors, RAM bytes, card bytes).
fn resident_by_source(pop: &str) -> Vec<(String, usize, u64, u64)> {
    let mut by: BTreeMap<String, (usize, u64, u64)> = BTreeMap::new();
    for r in nmengn::residency::get().parameters().into_iter().filter(|r| r.model.ends_with(pop)) {
        let e = by.entry(r.source).or_default();
        e.0 += usize::from(!r.tensor.starts_with('('));
        e.1 += r.ram;
        e.2 += r.device;
    }
    by.into_iter().map(|(k, (a, b, c))| (k, a, b, c)).collect()
}
fn resident_text(v: &[(String, usize, u64, u64)]) -> String {
    if v.is_empty() {
        return "none".to_owned();
    }
    v.iter()
        .map(|(src, k, ram, dev)| format!("{src}: {k} tensors, RAM {:.1} MiB, card {:.1} MiB", mib(usize::try_from(*ram).unwrap_or(0)), mib(usize::try_from(*dev).unwrap_or(0))))
        .collect::<Vec<_>>()
        .join("; ")
}

/// `release <cartridge> [bits=<n>]`: the judge seated on a base cartridge against the open weights,
/// on the processor and the card: the parameter tensors resident by source (counted by name from
/// the engine's residency accounting) with the open weights in service, with the cartridge seated
/// and the open weights released, and after pulling it; RAM and card bytes; scores bit for bit
/// against the open weights; the time to seat (check, bind, check run, release) and to pull (drop
/// the cartridge, open and bind the open weights).
fn release(home: &Path, args: &[String]) -> Result<String, Fault> {
    let file = home.join(args.get(3).ok_or_else(|| bad("release <cartridge> [bits=<n>] [model=base|embed|writer] [on=cpu|gpu]"))?);
    let bits: Option<u8> = opt(args, "bits").and_then(|x| x.parse().ok());
    let mo = model_of(args);
    nmengn::gpu::HALF_MATH.store(true, Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
    let set = load_set(home)?;
    let qs: Vec<&Q> = in_part(&set, "map").into_iter().take(2).collect();
    let offer = nmapi::neural::offer_of(&file)?;
    let mut out = format!(
        "cartridge {} ({} parameter sets) for {} ({}), precision {}
",
        offer.key,
        offer.parts.len(),
        mo.name,
        mo.job,
        bits.map_or_else(|| "as the assembly writes it (16)".to_owned(), |b| format!("{b} bits"))
    );
    let bind = |pop: &Arc<Population>| -> Result<Model, Fault> {
        let m = Model::bind(pop, pruned_assembly(home, mo.asm, &Spec { layers: (0..28).collect(), bits, ..Spec::default() }, 28, 128)?)?;
        Ok(if mo.judge { m.label_rows() } else { m })
    };
    // `on=cpu|gpu`: one device only (the writer's two copies on the processor need much RAM).
    let devices: Vec<(Device, bool)> = match opt(args, "on") {
        Some("cpu") => vec![(Device::Cpu, false)],
        Some("gpu") => vec![(Device::Gpu, true)],
        _ => vec![(Device::Cpu, false), (Device::Gpu, true)],
    };
    for (dev, gpu) in devices {
        // The role's own outputs: the judge's P(yes); the embedder's vector of each passage; the
        // base model's final norm over each passage (what the second look and the drones read).
        let run = |m: &Model| -> Result<Vec<u32>, Fault> {
            let mut v = Vec::new();
            for q in &qs {
                let few = Q { cands: q.cands.iter().take(if gpu { q.cands.len() } else { 3 }).cloned().collect(), ..(*q).clone() };
                if mo.judge {
                    v.extend(score(m, dev, &few)?.iter().map(|x| x.to_bits()));
                    continue;
                }
                for c in &few.cands {
                    if mo.pop == "qwen3-embedding-0.6b" {
                        v.extend(nmengn::roles::card_embed_on(m, dev, &c.doc)?.iter().map(|x| x.to_bits()));
                    } else {
                        let ids = m.pop.tokenizer.encode(&c.doc).map_err(|e| bad(e.to_string()))?;
                        let t = (m.role_node("final_norm")?, "y".to_owned());
                        let o = m.forward(dev, &ids, std::slice::from_ref(&t), &RunOpts::default())?;
                        v.extend(values(o.get(&t).ok_or_else(|| bad("no final norm"))?)?.iter().map(|x| x.to_bits()));
                    }
                }
            }
            Ok(v)
        };
        let t = Instant::now();
        let open_pop = Population::open(home, mo.pop, true, gpu)?;
        let open = bind(&open_pop)?;
        let a = run(&open)?;
        let t_open = t.elapsed().as_secs_f64();
        let r_open = resident_by_source(mo.pop);
        let t = Instant::now();
        let (cart, seated) = nmengn::base::seat(&open, &offer).map_err(|e| bad(e.to_string()))?;
        drop(open);
        drop(open_pop);
        let t_seat = t.elapsed().as_secs_f64();
        let r_seat = resident_by_source(mo.pop);
        let open_left: usize = r_seat.iter().filter(|r| r.0 == nmengn::model::OPEN_WEIGHTS).map(|r| r.1).sum();
        let b = run(&cart)?;
        let t = Instant::now();
        drop(cart);
        nmengn::base::pull_base(mo.pop);
        let back_pop = Population::open(home, mo.pop, true, gpu)?;
        let back = bind(&back_pop)?;
        let t_pull = t.elapsed().as_secs_f64();
        let r_pull = resident_by_source(mo.pop);
        let c = run(&back)?;
        drop(back);
        drop(back_pop);
        let same = |x: &[u32], y: &[u32]| x.iter().zip(y).filter(|(p, r)| p == r).count();
        let _ = writeln!(
            out,
            "{}: {} outputs; seated against the open weights {} identical, pulled against the open weights {} identical; seat check {} of {} outputs identical\n  open weights in service: {}\n  cartridge seated, open weights released: {}; open-weights tensors resident {open_left}\n  pulled: {}\n  open and bind the open weights {t_open:.2} s (with first scores); seat {t_seat:.2} s; pull {t_pull:.2} s",
            dev.name(),
            a.len(),
            same(&a, &b),
            same(&a, &c),
            seated.identical,
            seated.outputs,
            resident_text(&r_open),
            resident_text(&r_seat),
            resident_text(&r_pull),
        );
        if open_left > 0 || a != b || a != c {
            let _ = writeln!(out, "  FAIL on {}", dev.name());
        }
    }
    Ok(out)
}

/// `stack <use> <id> <cartridge,...> [v=<n>] [about=<text, underscores for spaces>]`: puts a
/// stack on the library's shelf for a use (on trial), its cartridges in order.
fn stack(home: &Path, args: &[String]) -> Result<String, Fault> {
    let usage = || bad("stack <use> <id> <cartridge,...>");
    let use_for = args.get(3).ok_or_else(usage)?;
    let id = args.get(4).ok_or_else(usage)?;
    let cartridges: Vec<String> = args.get(5).ok_or_else(usage)?.split(',').map(str::to_owned).collect();
    let version: u64 = opt(args, "v").and_then(|x| x.parse().ok()).unwrap_or(1);
    let description = opt(args, "about").unwrap_or("").replace('_', " ");
    let mut lib = nmlearn::library::Library::open(home).map_err(|e| bad(e.to_string()))?;
    let key = lib
        .add_stack(nmlearn::library::Stack { id: id.clone(), version, use_for: use_for.clone(), piece: None, cartridges, description, hash: String::new(), trial: true })
        .map_err(|e| bad(e.to_string()))?;
    lib.save().map_err(|e| bad(e.to_string()))?;
    Ok(format!("stack {key} for {use_for} in the cartridge library (trial)"))
}

/// `unlist <id@version>`: takes a cartridge off the library's shelf (its file stays on disk).
fn unlist(home: &Path, args: &[String]) -> Result<String, Fault> {
    let key = args.get(3).ok_or_else(|| bad("unlist <id@version>"))?;
    let mut lib = nmlearn::library::Library::open(home).map_err(|e| bad(e.to_string()))?;
    let was = lib.unlist(key);
    lib.save().map_err(|e| bad(e.to_string()))?;
    Ok(format!("{key}: {}", if was { "taken out of the cartridge library; its file stays on disk" } else { "was not in the cartridge library" }))
}

/// The MicroModel a step works on: the reranker (default, the judge), the base 0.6B (`model=base`,
/// the second look and the drones' reading), the embedder (`model=embed`, the dense lane) or the
/// writer 1.7B (`model=writer`, short notes and summaries above cited source, on demand).
struct Mo {
    pop: &'static str,
    asm: &'static str,
    judge: bool,
    name: &'static str,
    job: &'static str,
}
fn model_of(args: &[String]) -> Mo {
    match opt(args, "model") {
        Some("base") => Mo { pop: "qwen3-0.6b", asm: "qwen3-reference", judge: false, name: "Reader", job: "the reader's roles (the drones' and tentacles' reading, the second look)" },
        Some("embed") => Mo { pop: "qwen3-embedding-0.6b", asm: "qwen3-embedding", judge: false, name: "Embedder", job: "the embedder's role (the dense lane and recall by meaning)" },
        Some("writer") => Mo { pop: "qwen3-1.7b", asm: "qwen3-reference", judge: false, name: "Writer", job: "the writer's role (short notes and summaries above cited source, loaded on demand)" },
        _ => Mo { pop: POP, asm: ASM, judge: true, name: "Reranker", job: "judging which passages of code answer a question (the reranker's role)" },
    }
}

/// The bare weights' SHA-256 as the weight card states it (what a cartridge label names).
fn bare_sha(home: &Path, pop_id: &str) -> Result<String, Fault> {
    let card = nmtype::models_dir(home).join(pop_id).join("part.json");
    let v = nmjson::parse(&std::fs::read(&card).map_err(|e| bad(format!("{}: {e}", card.display())))?).map_err(|e| bad(e.to_string()))?;
    v.get("weights").and_then(|w| w.get("sha256")).and_then(Value::string).map(str::to_owned).map_err(|e| bad(format!("{}: {e}", card.display())))
}

/// `build <spec> <id> [v=<n>] [model=base] [effect=<text, underscores for spaces>]`: the base
/// cartridge of a job, written by the library as one labelled file on its shelf (on trial, so it
/// serves nothing until switched on): every parameter set every block of the job's assembly reads,
/// each with the layer and block it loads onto, as BF16 exactly as released; less only inside the
/// feed-forward blocks (`ffn=<way>:<share>`: the channel blocks the map keeps). No structure is
/// written: the blocks are the MicroModel's. A spec that leaves out layers or attention, or sets a
/// precision for some pieces, is refused: precision is the MicroModel's one selector.
fn build(home: &Path, args: &[String]) -> Result<String, Fault> {
    let t0 = Instant::now();
    let spec_t = args.get(3).ok_or_else(|| bad("build <spec> <id>"))?;
    let id = args.get(4).ok_or_else(|| bad("build <spec> <id>"))?;
    let version: u64 = opt(args, "v").and_then(|x| x.parse().ok()).unwrap_or(1);
    let effect = opt(args, "effect").unwrap_or("").replace('_', " ");
    let sp = spec_for(home, spec_t)?;
    if sp.layers.len() < 28 || !sp.groups.is_empty() {
        return Err(bad("a base cartridge keeps every layer and every attention block whole: depth, drop and heads are measurements only"));
    }
    if sp.bits.is_some() || sp.fbits.is_some() || sp.abits.is_some() || sp.cart.is_some() || sp.addon.is_some() {
        return Err(bad("a base cartridge holds parameters only: precision is the MicroModel's selector, and cart or addon is no part of one"));
    }
    let mo = model_of(args);
    let pop = Population::open(home, mo.pop, true, false)?;
    let d = pop.dims;
    let asm = Assembly::load(&home.join("designs/assemblies"), mo.asm)?;
    let needed = nmengn::base::read_by(&asm);
    let mut tensors: Vec<nmlearn::library::Tensor> = Vec::new();
    let mut ffn_bytes = 0_usize;
    for name in &needed {
        let (layer, block) = nmengn::base::address(name).ok_or_else(|| bad(format!("{name}: no layer and block for it")))?;
        let (v, sh) = tensor(&pop, name)?;
        let (r, c) = (sh.first().copied().unwrap_or(0), sh.get(1).copied().unwrap_or(0));
        let ffn = [".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight"].iter().find(|x| name.ends_with(*x));
        let (shape, bytes) = match (ffn, layer.and_then(|l| sp.ffn.get(&l))) {
            (Some(part), Some(channels)) => {
                let keep = ranges(channels, 1);
                let kept: usize = keep.iter().map(|(a, e)| e - a).sum();
                let by_cols = *part == ".mlp.down_proj.weight";
                let bytes = cut(&v.tensor_bytes(), c, &keep, by_cols, r);
                (if by_cols { vec![r, kept] } else { vec![kept, c] }, bytes)
            }
            _ => (sh.clone(), v.tensor_bytes().to_vec()),
        };
        if ffn.is_some() {
            ffn_bytes += bytes.len();
        }
        tensors.push(nmlearn::library::Tensor { name: name.clone(), layer, block: block.to_owned(), dtype: "BF16".to_owned(), shape, bytes });
    }
    let all: usize = tensors.iter().map(|t| t.bytes.len()).sum();
    let table: usize = tensors.iter().filter(|t| t.layer.is_none()).map(|t| t.bytes.len()).sum();
    let open_bytes: usize = needed.iter().filter_map(|nm| pop.tensors.iter().find(|t| &t.name == nm)).map(|t| usize::try_from(t.end - t.start).unwrap_or(0)).sum();
    let exact = sp.ffn.is_empty();
    let job = mo.job;
    let spec = nmlearn::library::CartridgeSpec {
        id: id.clone(),
        version,
        family: String::new(),
        domain: (if mo.judge {
            "job:judge-code"
        } else if mo.pop == "qwen3-1.7b" {
            "job:write"
        } else {
            "job:base"
        })
        .to_owned(),
        scope: "general".to_owned(),
        model: mo.pop.to_owned(),
        weights: bare_sha(home, mo.pop)?,
        layers: Some((0, d.layers as u64 - 1)),
        name: format!("{}, base cartridge{}", mo.name, if exact { ", exact" } else { "" }),
        description: format!(
            "All the parameters the blocks of every one of the {} layers read for {job}, attention parameters complete at full shape, each set addressed to its layer and block, at the open weights' own values (BF16){}. {:.0} MiB ({:.0} MiB of feed-forward blocks).",
            d.layers,
            if exact { "" } else { "; inside the feed-forward blocks only the channel blocks the map keeps" },
            mib(all),
            mib(ffn_bytes)
        ),
        made_from: format!(
            "{} open weights, copied unchanged{} into a full working copy at 16 bits (BF16), `nano-lab run bcart build {spec_t} {id}`",
            mo.pop,
            if exact { "" } else { " by the Parameter Map of the job" }
        ),
        effect,
    };
    drop(pop);
    let mut notes = vec![
        ("content", s("base")),
        ("exact", Value::Bool(exact)),
        ("spec", s(spec_t)),
        ("token_table_bytes", n(table as u64)),
        ("layer_bytes", n((all - table) as u64)),
        ("open_weights_bytes", n(open_bytes as u64)),
    ];
    // The version it was made over and what changed, with its measured result on the role, for
    // the manifest and for roll back (`library rollback`).
    if let Some(p) = opt(args, "parent") {
        notes.push(("parent", s(p)));
    }
    if let Some(r) = opt(args, "result") {
        notes.push(("measured", s(&r.replace('_', " "))));
    }
    if let Some(c) = opt(args, "changed") {
        notes.push(("changed", s(&c.replace('_', " "))));
    }
    // The version's record (owner direction 2026-11-17): mapped and pruned, by which map and how much.
    let ffn_kept: usize = sp.ffn.values().map(Vec::len).sum();
    notes.push(("mapped", s(&if exact { "no".to_owned() } else { format!("yes: {spec_t}") })));
    notes.push((
        "pruned",
        s(&if exact { "no".to_owned() } else { format!("yes: feed-forward channels kept by the map in {} layers ({ffn_kept} channels kept), {:.0} MiB of feed-forward blocks", sp.ffn.len(), mib(ffn_bytes)) }),
    ));
    notes.push(("edits", s(&opt(args, "edits").map_or_else(|| "none".to_owned(), |e| e.replace('_', " ")))));
    let mut bound_line = String::new();
    if !exact {
        let (b, line) = check_bound(home, &mo, &tensors)?;
        notes.push(("check_bound", Value::Number(format!("{b:e}"))));
        bound_line = line;
    }
    let lib_home = if opt(args, "shelf") == Some("scratch") { home.join(DIR).join("curve") } else { home.to_path_buf() };
    let mut lib = nmlearn::library::Library::open(&lib_home).map_err(|e| bad(e.to_string()))?;
    // A full working copy of every set the blocks read (owner direction 2026-11-17 00:21: the base
    // cartridge is kept whole and edited over time; storing it as differences is cancelled).
    let key = lib.add_cartridge(&spec, &nmlearn::library::Filter::Tensors(tensors), obj(notes), Vec::new(), true).map_err(|e| bad(e.to_string()))?;
    lib.save().map_err(|e| bad(e.to_string()))?;
    let file = lib.cartridge(&key).map(|c| lib.cartridge_dir(c).join(c.files.first().map_or("", |f| f.name.as_str()))).unwrap_or_default();
    let bytes = std::fs::metadata(&file).map(|m| usize::try_from(m.len()).unwrap_or(0)).unwrap_or(0);
    Ok(format!(
        "{bound_line}built {key} in the cartridge library (trial): {} ({bytes} bytes on disk; {} parameter sets taken from the open weights, {:.1} MiB once built, of which feed-forward {:.1} MiB; the open weights' sets for this job {:.1} MiB) from spec {spec_t} in {:.1} s",
        file.display(),
        needed.len(),
        mib(all),
        mib(ffn_bytes),
        mib(open_bytes),
        t0.elapsed().as_secs_f64()
    ))
}

/// `restore <cartridge> <address>[,<address>...] [keep=yes]`: parts of a base cartridge put back to
/// the open weights' own values by layer and block address (`L<n>:<block>`, `L<n>` for every block of
/// a layer, `table`, `final norm`; `all:<block>` for that block in every layer). Each set at an
/// address becomes the whole open-weights set (an edited value goes too). The result is seated over
/// the open weights in this run and checked; with `keep=yes` it is written as the cartridge's next
/// version (the earlier version stays the roll-back point), otherwise nothing is written.
fn restore(home: &Path, args: &[String]) -> Result<String, Fault> {
    let t0 = Instant::now();
    let file = home.join(args.get(3).ok_or_else(|| bad("restore <cartridge> <address>[,...] [keep=yes]"))?);
    let want: Vec<String> = args.get(4).ok_or_else(|| bad("restore <cartridge> <address>[,...]"))?.split(',').map(str::to_owned).collect();
    let label = nmlearn::library::label_of(&file).map_err(|e| bad(e.to_string()))?;
    let mut parts = nmlearn::library::tensor_parts_of(&file).map_err(|e| bad(e.to_string()))?;
    let model = text(&label, "model");
    let mo = match model.as_str() {
        "qwen3-0.6b" => model_of(&["".into(), "".into(), "".into(), "model=base".into()]),
        "qwen3-embedding-0.6b" => model_of(&["".into(), "".into(), "".into(), "model=embed".into()]),
        "qwen3-1.7b" => model_of(&["".into(), "".into(), "".into(), "model=writer".into()]),
        _ => model_of(&[]),
    };
    let at = |layer: Option<usize>, block: &str| {
        want.iter().any(|w| match (w.split_once(':'), layer) {
            (Some(("all", b)), Some(_)) => b == block,
            (Some((l, b)), Some(k)) => l == format!("L{k}") && b == block,
            (None, Some(k)) => *w == format!("L{k}"),
            (None, None) => w == block,
            _ => false,
        })
    };
    nmengn::gpu::HALF_MATH.store(true, Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
    let pop = Population::open(home, mo.pop, true, true)?;
    let mut restored = 0;
    for p in parts.iter_mut().filter(|p| at(p.layer, &p.block)) {
        let od = pop.tensors.iter().find(|t| t.name == p.name).ok_or_else(|| bad(format!("{}: not in the open weights", p.name)))?;
        let (v, sh) = tensor(&pop, &p.name)?;
        p.shape = sh;
        p.dtype.clone_from(&od.dtype);
        p.sha256 = nmhash::sha256_bytes(&v.tensor_bytes());
        p.open = true;
        p.keep = None;
        p.len = od.end - od.start;
        p.file = PathBuf::new();
        p.offset = 0;
        restored += 1;
    }
    if restored == 0 {
        return Err(bad(format!("no parameter set at {}", want.join(","))));
    }
    let exact = parts.iter().all(|p| p.open && p.keep.is_none());
    let mut offer = nmapi::neural::offer_from(&label, parts.clone());
    offer.exact = exact;
    let open = Model::bind(&pop, pruned_assembly(home, mo.asm, &Spec { layers: (0..28).collect(), ..Spec::default() }, 28, 128)?)?;
    let open = if mo.judge { open.label_rows() } else { open };
    let (cart, seated) = nmengn::base::seat(&open, &offer).map_err(|e| bad(e.to_string()))?;
    drop(cart);
    nmengn::base::pull_base(mo.pop);
    let mut out = format!(
        "restore {} at {}: {restored} parameter sets put back to the open weights' values; seated over the open weights: check {} of {} outputs bit-identical, largest change {:e}{}; {:.1} s\n",
        offer.key,
        want.join(","),
        seated.identical,
        seated.outputs,
        seated.largest,
        if exact { " (every set now the open weights' own)" } else { "" },
        t0.elapsed().as_secs_f64()
    );
    if opt(args, "keep") == Some("yes") {
        let mut lib = nmlearn::library::Library::open(home).map_err(|e| bad(e.to_string()))?;
        let id = text(&label, "id");
        let version = lib.cartridges.iter().filter(|c| c.id == id).map(|c| c.version).max().unwrap_or(0) + 1;
        // The next version is a full working copy: restored sets from the open weights (whole, or
        // the rows or columns a set taken from them keeps), every other set as the file holds it.
        let mut tensors: Vec<nmlearn::library::Tensor> = Vec::with_capacity(parts.len());
        for p in &parts {
            let bytes = if p.open {
                let (v, sh) = tensor(&pop, &p.name)?;
                match &p.keep {
                    None => v.tensor_bytes().to_vec(),
                    Some(k) => cut(&v.tensor_bytes(), sh.get(1).copied().unwrap_or(1), &k.ranges, k.cols, sh.first().copied().unwrap_or(0)),
                }
            } else {
                read_at(&p.file, p.offset, p.len)?
            };
            if nmhash::sha256_bytes(&bytes) != p.sha256 {
                return Err(bad(format!("{}: its bytes do not match the content hash its label states", p.name)));
            }
            tensors.push(nmlearn::library::Tensor { name: p.name.clone(), layer: p.layer, block: p.block.clone(), dtype: p.dtype.clone(), shape: p.shape.clone(), bytes });
        }
        let mut notes = label.get("notes").cloned().unwrap_or(Value::Null);
        if let Value::Object(o) = &mut notes {
            o.insert("exact".into(), Value::Bool(exact));
            o.insert("restored".into(), s(&format!("from {} at {}", offer.key, want.join(","))));
        }
        let spec = nmlearn::library::CartridgeSpec {
            id: id.clone(),
            version,
            family: text(&label, "family"),
            domain: text(&label, "domain"),
            scope: text(&label, "scope"),
            model: model.clone(),
            weights: text(&label, "weights"),
            layers: Some((0, 27)),
            name: text(&label, "name"),
            description: text(&label, "does"),
            made_from: format!("{}; {} put back to the open weights' values (`nano-lab run bcart restore`)", text(&label, "made_from"), want.join(",")),
            effect: text(&label, "effect"),
        };
        let key = lib.add_cartridge(&spec, &nmlearn::library::Filter::Tensors(tensors), notes, Vec::new(), true).map_err(|e| bad(e.to_string()))?;
        lib.save().map_err(|e| bad(e.to_string()))?;
        let _ = writeln!(out, "kept as {key}; {} stays the roll-back point", offer.key);
    }
    Ok(out)
}

/// `manifest`: one `MANIFEST.md` in each cartridge folder of the layout by kind
/// (`library/cartridges/<kind>/<MicroModel or hardware>/`), as the backup repository keeps them: the
/// version the folder holds, its file, the version it was made over, its file's SHA-256 (read again
/// from the file, streamed) and bytes, what it is, what changed and how it measured. One line per
/// file is printed, for the version backup.
fn manifest(home: &Path) -> Result<String, Fault> {
    let lib = nmlearn::library::Library::open(home).map_err(|e| bad(e.to_string()))?;
    let mut out = String::new();
    for c in lib.cartridges.iter().filter(|c| c.files.first().is_some_and(|f| f.name.contains('/'))) {
        let rel = c.files.first().map_or("", |f| f.name.as_str());
        let file = lib.cartridge_dir(c).join(rel);
        let size = std::fs::metadata(&file).map_err(|e| bad(format!("{}: {e}", file.display())))?.len();
        let sha = nmhash::sha256(std::io::BufReader::new(std::fs::File::open(&file).map_err(|e| bad(e.to_string()))?)).map_err(|e| bad(e.to_string()))?;
        let settings = c.kind == "settings";
        let parent = match text(&c.notes, "parent") {
            p if !p.is_empty() => p,
            _ if settings => "none (settings, measured on this hardware)".to_owned(),
            _ => "none (made from the open weights)".to_owned(),
        };
        let changed = if text(&c.notes, "changed").is_empty() { c.made_from.clone() } else { text(&c.notes, "changed") };
        let field = |k: &str| match text(&c.notes, k) {
            v if !v.is_empty() => v,
            _ if settings && k != "edits" => "not applicable (no parameters)".to_owned(),
            _ => "not recorded".to_owned(),
        };
        let measured = [text(&c.notes, "measured"), text(&c.notes, "role_result"), c.effect.clone()].into_iter().find(|x| !x.is_empty()).unwrap_or_default();
        let name = file.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
        let same = sha == c.files.first().map_or("", |f| f.sha256.as_str());
        let _ = writeln!(out, "{rel}: {}, parent {parent}, sha256 {sha}, {size} bytes, {}{}", c.key(), c.name, if same { "; matches the catalogue" } else { "; DIFFERS from the catalogue" });
        let label_key = text(&c.notes, "label_key");
        // The columns of the backup repository's manifests (`cartridges/base/reranker/MANIFEST.md`
        // there): version, made, parent, mapped, pruned, other edits, sha256, measured.
        let md = format!(
            "# {}\n\nThe folder holds the current version only, named with its version: `{name}` ({size} bytes). Earlier versions are in the backup repository's history.{}\n\n## Versions\n\n| version | made (UTC) | parent | mapped | pruned | other edits | sha256 | measured on its role |\n|---|---|---|---|---|---|---|---|\n| {} (current) | {} | {parent} | {} | {} | {} | {sha} | {} |\n\nMicroModel or use: {}. What it is: {} What changed: {}\n",
            c.name,
            if label_key.is_empty() { String::new() } else { format!(" Its label names it `{label_key}` (written before the catalogue name).") },
            c.version,
            c.made,
            field("mapped"),
            field("pruned"),
            field("edits"),
            if measured.is_empty() { "not recorded".to_owned() } else { measured.replace('|', "/") },
            if c.model.is_empty() { c.domain.clone() } else { c.model.clone() },
            c.description,
            if changed.is_empty() { "not recorded".to_owned() } else { changed },
        );
        if let Some(dir) = file.parent() {
            std::fs::write(dir.join("MANIFEST.md"), md).map_err(|e| bad(e.to_string()))?;
        }
    }
    Ok(out)
}

/// `rename <id@version> <new id@version> [parent=<id@version>]`: names a cartridge as the catalogue
/// should and moves its file (a rename on the same drive, never a copy) into the layout by kind
/// (`nmlearn::library::shelf_path`); stacks holding it follow.
fn rename(home: &Path, args: &[String]) -> Result<String, Fault> {
    let usage = || bad("rename <id@version> <new id@version> [parent=<id@version>]");
    let from = args.get(3).ok_or_else(usage)?;
    let (id, v) = args.get(4).and_then(|x| x.split_once('@')).ok_or_else(usage)?;
    let version: u64 = v.parse().map_err(|_| usage())?;
    let mut lib = nmlearn::library::Library::open(home).map_err(|e| bad(e.to_string()))?;
    let c = lib.cartridge(from).cloned().ok_or_else(|| bad(format!("no cartridge {from}")))?;
    let path = nmlearn::library::shelf_path(&c.family, &c.kind, &c.domain, &c.model, id, version);
    let key = lib.rename(from, id, version, &path, opt(args, "parent")).map_err(|e| bad(e.to_string()))?;
    // The version's record given here (underscores for spaces): mapped, pruned, edits, measured.
    if let Some(c) = lib.cartridges.iter_mut().find(|c| c.key() == key) {
        if !matches!(c.notes, Value::Object(_)) {
            c.notes = Value::Object(BTreeMap::new());
        }
        if let Value::Object(o) = &mut c.notes {
            for k in ["mapped", "pruned", "edits", "measured"] {
                if let Some(v) = opt(args, k) {
                    o.insert(k.to_owned(), s(&v.replace('_', " ")));
                }
            }
        }
    }
    lib.save().map_err(|e| bad(e.to_string()))?;
    Ok(format!("{from}: now {key}, file library/cartridges/{path}"))
}

/// The largest change of the seat's check-run outputs (`nmengn::base::check_outputs`) that a
/// cartridge's parameter sets give against the open weights, measured from the sets themselves on
/// the processor and the card at each running selection (16, 8 and 4 bits), on the card also with
/// the served arithmetic (half math, the segment kernel), for its label: the check passes within it.
fn check_bound(home: &Path, mo: &Mo, tensors: &[nmlearn::library::Tensor]) -> Result<(f32, String), Fault> {
    let dir = home.join(DIR).join("tmp");
    std::fs::create_dir_all(&dir).map_err(|e| bad(e.to_string()))?;
    let file = dir.join("check-sets.bin");
    let mut body = Vec::new();
    let mut descs = Vec::new();
    for t in tensors {
        let start = body.len() as u64;
        body.extend_from_slice(&t.bytes);
        descs.push(nmnpcs::TensorDesc { name: t.name.clone(), dtype: t.dtype.clone(), shape: t.shape.clone(), file: file.display().to_string(), start, end: body.len() as u64, sha256: String::new() });
    }
    std::fs::write(&file, &body).map_err(|e| bad(e.to_string()))?;
    drop(body);
    let mut largest = 0.0_f32;
    let mut line = String::new();
    let switches = (nmengn::gpu::HALF_MATH.load(Ordering::Relaxed), nmengn::gpu::SEGMENT_ATTENTION.load(Ordering::Relaxed));
    for (gpu, served) in [(false, false), (true, false), (true, true)] {
        // The served arithmetic (half math, the segment kernel) as `serve` seats it, and without.
        nmengn::gpu::HALF_MATH.store(served, Ordering::Relaxed);
        nmengn::gpu::SEGMENT_ATTENTION.store(served, Ordering::Relaxed);
        for bits in [None, Some(8_u8), Some(4_u8)] {
            let sp = Spec { layers: (0..28).collect(), bits, ..Spec::default() };
            let asm = pruned_assembly(home, mo.asm, &sp, 28, 128)?;
            let rows = |m: Model| if mo.judge { m.label_rows() } else { m };
            let open_pop = Population::open(home, mo.pop, true, gpu)?;
            let a = nmengn::base::check_outputs(&rows(Model::bind(&open_pop, asm.clone())?))?;
            drop(open_pop);
            let cart_pop = Population::open_cartridge(home, mo.pop, gpu, "check", descs.clone())?;
            let b = nmengn::base::check_outputs(&rows(Model::bind(&cart_pop, asm)?))?;
            let d = a.iter().zip(&b).map(|(x, y)| (f32::from_bits(*x) - f32::from_bits(*y)).abs()).fold(0.0_f32, f32::max);
            let _ = write!(line, "check change {}{} at {}: {d:e}; ", if gpu { "card" } else { "processor" }, if served { " (served arithmetic)" } else { "" }, bits.map_or_else(|| "16".to_owned(), |b| b.to_string()));
            largest = largest.max(d);
        }
    }
    nmengn::gpu::HALF_MATH.store(switches.0, Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(switches.1, Ordering::Relaxed);
    let _ = std::fs::remove_file(&file);
    let _ = writeln!(line, "label bound {largest:e}");
    Ok((largest, line))
}

/// The judge's assembly at the precision selected for the whole MicroModel (`None`: as written).
fn judge_assembly(home: &Path, bits: Option<u8>) -> Result<Assembly, Fault> {
    pruned_assembly(home, ASM, &Spec { layers: (0..28).collect(), bits, ..Spec::default() }, 28, 128)
}

/// A judge model run from a base cartridge through the product's seating path: the open weights'
/// model is bound at the selected precision, the cartridge is checked complete and connected and
/// run against it (`nmengn::base::seat`), and the open weights are released.
fn load_cartridge(home: &Path, file: &Path, gpu: bool, bits: Option<u8>) -> Result<(Model, nmengn::base::Seated), Fault> {
    let offer = nmapi::neural::offer_of(file)?;
    let open_pop = Population::open(home, POP, true, gpu)?;
    let open = Model::bind(&open_pop, judge_assembly(home, bits)?)?.label_rows();
    let seated = nmengn::base::seat(&open, &offer).map_err(|e| bad(e.to_string()))?;
    drop(open);
    drop(open_pop);
    Ok(seated)
}

/// `check <spec> <cartridge>`: the cartridge's scores against the same spec over the bare weights
/// on a few map questions (they must be bit-identical).
fn check(home: &Path, args: &[String]) -> Result<String, Fault> {
    let sp = spec_for(home, args.get(3).ok_or_else(|| bad("check <spec> <cartridge>"))?)?;
    let cart = spec_for(home, &format!("cart={}", args.get(4).ok_or_else(|| bad("check <spec> <cartridge>"))?))?;
    nmengn::gpu::HALF_MATH.store(true, Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
    let set = load_set(home)?;
    let qs: Vec<&Q> = in_part(&set, "map").into_iter().take(3).collect();
    let (a, b) = (model_for(home, &sp, true)?, model_for(home, &cart, true)?);
    let (mut same, mut all, mut worst) = (0, 0, 0.0_f32);
    for q in &qs {
        let (x, y) = (score(&a, Device::Gpu, q)?, score(&b, Device::Gpu, q)?);
        for (p, r) in x.iter().zip(&y) {
            all += 1;
            same += usize::from(p.to_bits() == r.to_bits());
            worst = worst.max((p - r).abs());
        }
    }
    Ok(format!("cartridge against the same cut over the bare weights: {same} of {all} scores bit-identical, largest difference {worst:e}; card weights {:.1} against {:.1} MiB", mib(b.pop.device_bytes()), mib(a.pop.device_bytes())))
}

// ---------------------------------------------------------------------------------------------
// The served whole path with a cartridge seated.

/// Places a reply cites, in order (`@@ path:a-b` and `~~ path:line`).
fn cited(reply: &str) -> Vec<(String, usize, usize)> {
    reply
        .lines()
        .filter_map(|l| {
            let (head, span) = l.strip_prefix("@@ ").map(|r| (r, true)).or_else(|| l.strip_prefix("~~ ").map(|r| (r, false)))?;
            let (path, at) = head.split_whitespace().next()?.rsplit_once(':')?;
            let (a, b) = if span { at.split_once('-').unwrap_or((at, at)) } else { (at, at) };
            Some((path.replace('\\', "/"), a.parse().ok()?, b.parse().ok()?))
        })
        .collect()
}

/// `write <cartridge> [k=<n>] [bits=<n>]`: the writer on its job, a short note above cited source:
/// for `k` questions of the judged set (default 8) with a right passage, the writer (Qwen3-1.7B,
/// `neural::writer_model`, on demand) writes one line on what that passage does for the question,
/// greedy, served arithmetic. Arm one on the open weights; arm two with `<cartridge>` offered as its
/// base cartridge (seated by `on_base`, the open weights released). Reported: the seat line, notes
/// identical between the arms, time per note, the device each ran on, the card in use and the
/// parameter tensors resident by source with the writer bound.
fn write(home: &Path, args: &[String]) -> Result<String, Fault> {
    let t0 = Instant::now();
    let file = home.join(args.get(3).ok_or_else(|| bad("write <cartridge> [k=<n>] [bits=<n>]"))?);
    let k: usize = opt(args, "k").and_then(|x| x.parse().ok()).unwrap_or(8);
    let bits: Option<u8> = opt(args, "bits").and_then(|x| x.parse().ok());
    if let Ok(mut g) = nmengn::model::PRECISION.write() {
        g.writer = bits;
    }
    nmengn::gpu::HALF_MATH.store(true, Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
    let set = load_set(home)?;
    let jobs: Vec<(String, String)> = set
        .iter()
        .filter_map(|q| q.cands.iter().find(|c| q.right(c)).map(|c| (q.id.clone(), format!("{}

Question: {}
A one-line note on what the code above does for this question:", c.doc, q.question))))
        .take(k)
        .collect();
    let card_used = |m: &Model| m.pop.gpu.as_ref().and_then(|g| g.gpu.memory().ok()).map_or(0, |(f, t)| (t - f) >> 20);
    let mut out = format!(
        "writer ({}) on its job, {} notes, precision {}
",
        nmapi::neural::WRITER,
        jobs.len(),
        bits.map_or_else(|| "16 (the default)".to_owned(), |b| format!("{b} bits"))
    );
    let mut arms: Vec<Vec<String>> = Vec::new();
    for (arm, offered) in [("open weights", None), ("base cartridge", Some(file.clone()))] {
        if let Ok(mut g) = nmapi::neural::BASE_CARTRIDGES.write() {
            g.remove(nmapi::neural::WRITER);
            if let Some(f) = &offered {
                g.insert(nmapi::neural::WRITER.to_owned(), vec![f.clone()]);
            }
        }
        let t = Instant::now();
        let m = nmapi::neural::writer_model(home, true)?;
        let t_open = t.elapsed().as_secs_f64();
        let bound = card_used(&m);
        let resident = resident_text(&resident_by_source(nmapi::neural::WRITER));
        let mut notes = Vec::new();
        let mut secs = 0.0;
        let mut devices = BTreeSet::new();
        for (_, prompt) in &jobs {
            let t = Instant::now();
            let (note, choice, _) = nmengn::roles::write(&m, prompt, 48)?;
            secs += t.elapsed().as_secs_f64();
            devices.insert(format!("{} ({})", choice.device.name(), choice.why));
            notes.push(note);
        }
        let after = card_used(&m);
        drop(m);
        if offered.is_some() {
            nmengn::base::pull_base(nmapi::neural::WRITER);
        }
        let seat = nmapi::neural::seat_notes().into_iter().rev().find(|l| l.starts_with(nmapi::neural::WRITER)).filter(|_| offered.is_some()).unwrap_or_default();
        let kf = f64::from(u32::try_from(jobs.len().max(1)).unwrap_or(1));
        let _ = writeln!(
            out,
            "{arm}: open and bind {t_open:.2} s{}; {:.2} s per note on {}; card in use (device total minus free) {bound} MiB bound, {after} MiB after the notes; resident: {resident}",
            if seat.is_empty() { String::new() } else { format!(" ({seat})") },
            secs / kf,
            devices.into_iter().collect::<Vec<_>>().join(", ")
        );
        arms.push(notes);
    }
    let (a, b) = (arms.first().cloned().unwrap_or_default(), arms.get(1).cloned().unwrap_or_default());
    let same = a.iter().zip(&b).filter(|(x, y)| x == y).count();
    let _ = writeln!(out, "notes identical between the arms: {same} of {}", a.len());
    for (((id, _), note), other) in jobs.iter().zip(&a).zip(&b) {
        let _ = writeln!(out, "  {id}: {}", note.chars().take(160).collect::<String>());
        // A cut cartridge is not exact: its own note is shown beside the open weights' to be read.
        if other != note {
            let _ = writeln!(out, "  {id} (base cartridge): {}", other.chars().take(160).collect::<String>());
        }
    }
    let _ = writeln!(out, "total {:.1} s", t0.elapsed().as_secs_f64());
    if same != a.len() {
        let _ = writeln!(out, "FAIL: the base cartridge's notes differ from the open weights'");
    }
    Ok(out)
}

/// `served <id@version>[,...]`: takes a base cartridge off trial once it passes its check now: it
/// seats over its open weights on the card as the product seats it (complete, connected, the check
/// run within its bound), the open weights released; only then is its catalogue row's trial flag
/// cleared. A cartridge that does not seat keeps its flag, with the reason printed.
fn served(home: &Path, args: &[String]) -> Result<String, Fault> {
    let keys: Vec<String> = args.get(3).ok_or_else(|| bad("served <id@version>[,...]"))?.split(',').map(str::to_owned).collect();
    nmengn::gpu::HALF_MATH.store(true, Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
    let mut lib = nmlearn::library::Library::open(home).map_err(|e| bad(e.to_string()))?;
    let mut out = String::new();
    let mut passed = Vec::new();
    for key in &keys {
        let c = lib.cartridge(key).cloned().ok_or_else(|| bad(format!("no cartridge {key}")))?;
        let file = lib.cartridge_dir(&c).join(c.files.first().map_or("", |f| f.name.as_str()));
        let mo = match c.model.as_str() {
            "qwen3-0.6b" => model_of(&["".into(), "".into(), "".into(), "model=base".into()]),
            "qwen3-embedding-0.6b" => model_of(&["".into(), "".into(), "".into(), "model=embed".into()]),
            "qwen3-1.7b" => model_of(&["".into(), "".into(), "".into(), "model=writer".into()]),
            _ => model_of(&[]),
        };
        let t = Instant::now();
        let result = nmapi::neural::offer_of(&file).and_then(|offer| {
            let pop = Population::open(home, mo.pop, true, true)?;
            let m = Model::bind(&pop, pruned_assembly(home, mo.asm, &Spec { layers: (0..28).collect(), ..Spec::default() }, 28, 128)?)?;
            let m = if mo.judge { m.label_rows() } else { m };
            let (cart, seated) = nmengn::base::seat(&m, &offer).map_err(|e| bad(e.to_string()))?;
            drop(m);
            drop(pop);
            let left: usize = resident_by_source(mo.pop).iter().filter(|r| r.0 == nmengn::model::OPEN_WEIGHTS).map(|r| r.1).sum();
            drop(cart);
            nmengn::base::pull_base(mo.pop);
            Ok((seated, left))
        });
        match result {
            Ok((s, 0)) => {
                let _ = writeln!(out, "{key}: seated, check {} of {} outputs identical, largest change {:e}, open weights released (0 of their tensors resident), {:.1} s; trial flag cleared", s.identical, s.outputs, s.largest, t.elapsed().as_secs_f64());
                passed.push(key.clone());
            }
            Ok((_, left)) => {
                let _ = writeln!(out, "{key}: seated but {left} open-weights tensors still resident; stays on trial");
            }
            Err(e) => {
                let _ = writeln!(out, "{key}: refused ({}); stays on trial", e.message);
            }
        }
    }
    for c in lib.cartridges.iter_mut().filter(|c| passed.contains(&c.key())) {
        c.trial = false;
    }
    if !passed.is_empty() {
        lib.save().map_err(|e| bad(e.to_string()))?;
    }
    Ok(out)
}

/// `judgecost [n=<pieces per call>] [k=<questions>]`: seconds per piece judged at G2's edge, the
/// judge's reads as the served round makes them (`drone_path` WP6 step 1: one packed batch for the
/// posed question, then one per content-free form for each piece not read before, held by the
/// text read) against `neural::judge_edge` (posed and content-free pairs of every piece in one
/// packed read, content-free labels held per piece). The judge is the served reranker with its base
/// cartridge seated as at start (newest on the shelf). `n` pieces per call (the first candidates of
/// each question of the judged set), over `k` questions, twice: pass 1 sees each piece first, pass
/// 2 is the same material again (content-free labels held). Reported: seconds per piece judged per
/// pass and form, pairs read, and the largest difference of the calibrated values between the forms.
fn judgecost(home: &Path, args: &[String]) -> Result<String, Fault> {
    let n: usize = opt(args, "n").and_then(|x| x.parse().ok()).unwrap_or(3);
    let k: usize = opt(args, "k").and_then(|x| x.parse().ok()).unwrap_or(16);
    nmengn::gpu::HALF_MATH.store(true, Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
    let lib = nmlearn::library::Library::open(home).map_err(|e| bad(e.to_string()))?;
    let shelf: Vec<PathBuf> = lib
        .cartridges
        .iter()
        .filter(|c| c.model == POP && c.files.first().is_some_and(|f| f.name.contains('/')))
        .max_by_key(|c| c.version)
        .map(|c| vec![lib.cartridge_dir(c).join(c.files.first().map_or("", |f| f.name.as_str()))])
        .unwrap_or_default();
    if let Ok(mut g) = nmapi::neural::BASE_CARTRIDGES.write() {
        g.insert(POP.to_owned(), shelf.clone());
    }
    let m = nmapi::neural::reranker_model(home)?;
    let seat = nmapi::neural::seat_notes().into_iter().find(|l| l.starts_with(POP)).unwrap_or_default();
    let set = load_set(home)?;
    let qs: Vec<&Q> = set.iter().filter(|q| q.cands.len() >= n).take(k).collect();
    let dev = if m.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let logit = |p: f32| {
        let p = p.clamp(1e-6, 1.0 - 1e-6);
        (p / (1.0 - p)).ln()
    };
    // Warm the card's routines once (neither form pays the process's first read).
    let warm: Vec<String> = qs.first().map(|q| q.cands.iter().take(2).map(|c| c.doc.clone()).collect()).unwrap_or_default();
    nmengn::roles::card_rerank_batch(&m, dev, nmengn::roles::CODE_INSTRUCTION, "warm", &warm, None)?;
    let mut out = format!("judge {} ({seat}); {n} pieces per call, {} questions of the judged set, device {}\n", m.pop.card.id, qs.len(), dev.name());
    let mut held_old: BTreeMap<String, f32> = BTreeMap::new();
    let mut ys_old: Vec<Vec<f32>> = Vec::new();
    let mut ys_new: Vec<Vec<f32>> = Vec::new();
    for pass in 1..=2 {
        // The served round's reads (drone_path, WP6 step 1).
        let t = Instant::now();
        let mut reads_old = 0;
        for q in &qs {
            let docs: Vec<String> = q.cands.iter().take(n).map(|c| c.doc.clone()).collect();
            let posed = nmengn::roles::card_rerank_batch(&m, dev, nmengn::roles::CODE_INSTRUCTION, &q.question, &docs, None)?;
            reads_old += docs.len();
            let fresh: Vec<String> = docs.iter().filter(|d| !held_old.contains_key(*d)).cloned().collect();
            let mut null = vec![0.0_f32; fresh.len()];
            for f in nmapi::neural::CONTENT_FREE {
                if fresh.is_empty() {
                    break;
                }
                let v = nmengn::roles::card_rerank_batch(&m, dev, nmengn::roles::CODE_INSTRUCTION, f, &fresh, None)?;
                reads_old += fresh.len();
                #[expect(clippy::cast_precision_loss, reason = "a count of forms")]
                null.iter_mut().zip(v).for_each(|(a, p)| *a += logit(p) / nmapi::neural::CONTENT_FREE.len() as f32);
            }
            for (d, v) in fresh.iter().zip(null) {
                held_old.insert(d.clone(), v);
            }
            let y: Vec<f32> = docs.iter().zip(&posed).map(|(d, p)| logit(*p) - held_old.get(d).copied().unwrap_or(0.0)).collect();
            if pass == 1 {
                ys_old.push(y);
            }
        }
        let s_old = t.elapsed().as_secs_f64();
        // One packed read per call.
        let t = Instant::now();
        let (mut pairs, mut held) = (0, 0);
        for q in &qs {
            let pieces: Vec<nmapi::neural::EdgePiece> = q
                .cands
                .iter()
                .take(n)
                .map(|c| nmapi::neural::EdgePiece { key: format!("{}:{}-{}", c.path, c.start, c.end), posed: c.doc.clone(), plain: c.doc.clone() })
                .collect();
            let (y, r) = nmapi::neural::judge_edge(&m, &q.question, &pieces)?;
            pairs += r.pairs;
            held += r.held;
            if pass == 1 {
                ys_new.push(y);
            }
        }
        let s_new = t.elapsed().as_secs_f64();
        #[expect(clippy::cast_precision_loss, reason = "piece counts")]
        let pieces = (qs.len() * n) as f64;
        let _ = writeln!(
            out,
            "pass {pass}: served round {:.3} s per piece judged ({reads_old} reads, one card call per form); one packed read {:.3} s per piece judged ({pairs} pairs, {held} content-free labels held); {:.2} times faster",
            s_old / pieces,
            s_new / pieces,
            s_old / s_new.max(1e-9)
        );
    }
    let worst = ys_old.iter().flatten().zip(ys_new.iter().flatten()).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    let _ = writeln!(out, "calibrated label log-odds, served round against one packed read: largest difference {worst:.4} (binary16 rounding of the packed read)");
    Ok(out)
}

/// `whole <spec> <root> <questions> [ids]`: the served path in full mode, dense lane off, with the
/// spec's judge seated as the reranker (`base` seats nothing: the product's own binding).
fn whole(home: &Path, args: &[String]) -> Result<String, Fault> {
    let t0 = Instant::now();
    let spec_t = args.get(3).ok_or_else(|| bad("whole <spec> <root> <questions> [ids]"))?;
    let root = nmsubs::repo::canon(Path::new(args.get(4).ok_or_else(|| bad("no root"))?));
    let qs = questions(&root, Path::new(args.get(5).ok_or_else(|| bad("no question file"))?))?;
    let only: Option<BTreeSet<&str>> = args.get(6).filter(|x| !x.contains('=')).map(|x| x.split(',').collect());
    let mode = opt(args, "mode").unwrap_or("full");
    let sp = spec_for(home, spec_t)?;
    let mut out = String::new();
    let structural = sp.cart.is_some() || sp.bits.is_some() || sp.fbits.is_some() || sp.abits.is_some() || sp.layers.len() < 28 || !sp.ffn.is_empty() || !sp.groups.is_empty();
    let seat = if structural { Some(Arc::new(model_for(home, &sp, true)?)) } else { None };
    open_served(home, &root, seat)?;
    if let Some(a) = &sp.addon {
        let _ = writeln!(out, "add-on plugged: {}", plug_addon(a)?);
    }
    let (mut first, mut top3, mut pack, mut k, mut secs, mut rr_s) = (0, 0, 0, 0, 0.0, 0.0);
    let mut rows = Vec::new();
    for (id, question, answers) in qs.iter().filter(|q| only.as_ref().is_none_or(|o| o.contains(q.0.as_str()))) {
        let t = Instant::now();
        let reply = ask(question, mode);
        let dt = t.elapsed().as_secs_f64();
        let diag = nmapi::tools::take_diagnostics();
        let rer = diag
            .as_ref()
            .and_then(|d| d.get("context_map").ok())
            .and_then(|c| c.get("stages").ok())
            .and_then(|s2| s2.get("reranker").ok())
            .and_then(|x| if let Value::Number(t2) = x { t2.parse::<f64>().ok() } else { None })
            .unwrap_or(0.0);
        let c = cited(&reply);
        let hit = |p: &(String, usize, usize)| answers.iter().any(|(f, a, b)| *f == p.0 && p.1 <= *b && p.2 >= *a);
        let pos = c.iter().position(hit);
        k += 1;
        first += usize::from(pos == Some(0));
        top3 += usize::from(pos.is_some_and(|x| x < 3));
        pack += usize::from(pos.is_some());
        secs += dt;
        rr_s += rer;
        rows.push(format!("{id} {} {dt:.2}s", pos.map_or("-".to_owned(), |x| (x + 1).to_string())));
    }
    let kf = f64::from(u32::try_from(k).unwrap_or(1).max(1));
    let _ = writeln!(
        out,
        "whole path, {mode} mode, dense lane off, judge {spec_t}: {k} questions; right first {first}, top 3 {top3}, in pack {pack}; {:.3} s per question, reranker {:.3} s\n  per question (place of the first right citation): {}\ntotal {:.1} s",
        secs / kf,
        rr_s / kf,
        rows.join("; "),
        t0.elapsed().as_secs_f64()
    );
    Ok(out)
}

pub fn run(args: &[String], home: &Path) -> ExitCode {
    let step = args.get(2).map_or("eval", String::as_str);
    let stamp = nmtype::now_ms();
    let result = match step {
        "capture" => capture(home, args),
        "eval" => eval(home, args),
        "record" => record(home, args),
        "build" => build(home, args),
        "check" => check(home, args),
        "siblings" => siblings(home),
        "unlist" => unlist(home, args),
        "stack" => stack(home, args),
        "map" => match args.get(3).map(String::as_str) {
            Some("record") => map_record(home),
            Some("tol") => map_tol(home, args),
            Some("solve") => map_solve(home),
            Some("role") => map_role(home),
            Some("flap") => map_flap(home),
            _ => Err(bad("map record | tol <half> <first> <last> | solve | role | flap")),
        },
        "zsep" => match args.get(3).map(String::as_str) {
            Some("run") => zsep_run(home, args),
            Some("report") => zsep_report(home),
            _ => Err(bad("zsep run <arm>[,<arm>...] | report")),
        },
        "release" => release(home, args),
        "restore" => restore(home, args),
        "manifest" => manifest(home),
        "rename" => rename(home, args),
        "whole" => whole(home, args),
        "write" => write(home, args),
        "judgecost" => judgecost(home, args),
        "served" => served(home, args),
        other => Err(bad(format!("unknown step {other} (capture, eval, record, build, check, siblings, unlist, release, whole)"))),
    };
    match result {
        Ok(t) => {
            let p = home.join(format!("runs/BCART-{step}-{stamp}.txt"));
            let _ = std::fs::write(&p, format!("nano-lab {}\n{t}", args.get(1..).unwrap_or_default().join(" ")));
            print!("{t}");
            println!("report: {}", p.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("bcart {step}: {e}");
            ExitCode::FAILURE
        }
    }
}
