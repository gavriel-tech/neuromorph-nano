//! `nano-lab run ecart <step> ... [model=embed|judge|writer]`: a MicroModel's base cartridge cut from
//! its map, with its compensation as a file of its own.
//!
//! - `embed` (default): the embedder, from its activation map (C113,
//!   `library/parameter-maps/embedder/act-20.json`), `embedder-base@2` over `@1`.
//! - `judge`: the reranker, from its role map (`library/parameter-maps/reranker/role.json` on the
//!   per-layer budget of `map.json` at the share named, default 20; C113: above 18 of 20 random
//!   masks, beats their mean on 38 of 38), compensation `role-<share>-bias.json`; `reranker-base@3`
//!   over `@2`.
//! - `writer`: the writer (Qwen3-1.7B), from an activation map made on its own job (`record`, then
//!   `map`: the notes it writes over cited code, the same prompt `bcart write` serves), into
//!   `library/parameter-maps/writer/`; `writer-base@2` over `@1`.
//!
//! The cartridge is a full working copy (owner direction 2026-11-17 00:21) of every parameter set
//! the blocks of the MicroModel's assembly read, at the open weights' own values (BF16), each
//! addressed to its layer and block; less only inside the feed-forward blocks: the channels the map
//! keeps (gate and up rows, down columns). Every layer and every attention block stays (owner's
//! rule); a layer the map takes nothing from is copied whole. The open weights are only read.
//!
//! The compensation is each left-out channel's mean output over the map's reads (the bias at each
//! layer's `swigate`) carried through the down projection the cartridge no longer holds: per layer
//! `W_down[:, off] * mean_off` of the stream's width, stored as a gain-bias cartridge (gain 1) at
//! that layer's `down` node. It is a separate file named in the base cartridge's label; the serving
//! path plugs it on the seated model alone with `NANO_BASE_COMP=on` (`nmapi::neural`; off by default).
//!
//! Steps:
//! - `plan [map=...]`: what the cartridge would hold and its size; nothing run or written.
//! - `build [map=...] [id=...] [v=<n>] [parent=<id@v>] [margin=1.25] [big=yes]`: the processor check
//!   through the engine's own seat (the cut built from the open weights in RAM, compared with the
//!   open weights on the fixed check input; its largest change times `margin` is the label's
//!   bound), then the two files on the library's shelf. Refuses a cartridge above 1 GB without
//!   `big=yes`, and any write that would leave the drive under 4 GB free. Pins the MicroModel's
//!   current default in `settings.json` first when nothing is named there, so the new cartridge
//!   serves only when tried (`--try-cartridge <id@v>`) or named.
//! - `record model=writer [set=lane-d-tune-code.json] [gpu=no]`: per channel at every layer's
//!   `swigate`, over all tokens of the writer's job prompts on the set's fit half (each question's
//!   right candidate, the note prompt): the mean and the energy (mean square times the down
//!   column's squared norm, COMPACT's rule, as the embedder's map). `record.json`.
//! - `map model=writer [share=20]`: the lowest-energy channels, the same count in every layer, as
//!   `act-<share>.json` (`off` per layer) and their means as `act-<share>-bias.json`.
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use nmengn::Assembly;
use nmengn::assembly::nid;
use nmengn::model::{Model, Population, RunOpts, values};
use nmengn::placement::Device;
use nmjson::Value;
use nmtype::{Fault, FaultKind};

const GATE: &str = ".mlp.gate_proj.weight";
const UP: &str = ".mlp.up_proj.weight";
const DOWN: &str = ".mlp.down_proj.weight";
/// The size above which a cartridge is reported before it is written (lane rule).
const BIG: usize = 1_000_000_000;
/// Free space the drive keeps after a write (lane rule: stop under 4 GB).
const KEEP_FREE: u64 = 4_000_000_000;

/// One MicroModel's cut: where its weights, assembly and maps are, and how its cartridge is named.
struct Mo {
    key: &'static str,
    pop: &'static str,
    asm: &'static str,
    maps: &'static str,
    name: &'static str,
    job: &'static str,
    domain: &'static str,
    id: &'static str,
    version: u64,
    parent: &'static str,
    map: &'static str,
    /// The map is the role scores on `map.json`'s budget (the judge), not an `off` file.
    role: bool,
}
fn model_of(args: &[String]) -> Result<Mo, Fault> {
    Ok(match opt(args, "model").unwrap_or("embed") {
        "embed" => Mo {
            key: "embed",
            pop: "qwen3-embedding-0.6b",
            asm: "qwen3-embedding",
            maps: "library/parameter-maps/embedder",
            name: "Embedder",
            job: "the embedder's role (the dense lane and recall by meaning)",
            domain: "job:base",
            id: "embedder-base",
            version: 2,
            parent: "embedder-base@1",
            map: "act-20",
            role: false,
        },
        "judge" => Mo {
            key: "judge",
            pop: "qwen3-reranker-0.6b",
            asm: "qwen3-reranker",
            maps: "library/parameter-maps/reranker",
            name: "Reranker",
            job: "judging which passages of code answer a question (the judge's role)",
            domain: "job:judge-code",
            id: "reranker-base",
            version: 3,
            parent: "reranker-base@2",
            map: "role-20",
            role: true,
        },
        "writer" => Mo {
            key: "writer",
            pop: "qwen3-1.7b",
            asm: "qwen3-reference",
            maps: "library/parameter-maps/writer",
            name: "Writer",
            job: "the writer's role (short notes and summaries above cited source, loaded on demand)",
            domain: "job:write",
            id: "writer-base",
            version: 2,
            parent: "writer-base@1",
            map: "act-20",
            role: false,
        },
        "reader" => Mo {
            key: "reader",
            pop: "qwen3-0.6b",
            asm: "qwen3-reference",
            maps: "library/parameter-maps/reader",
            name: "Reader",
            job: "the reader's roles (the drones' layers on raw pieces, the tentacles' integration reads, the joint read and its yes or no, the request's reads)",
            domain: "job:base",
            id: "reader-base",
            version: 2,
            parent: "reader-base@1",
            map: "union",
            role: false,
        },
        o => return Err(bad(format!("model={o}: embed, judge, writer or reader"))),
    })
}

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmlab.ecart", FaultKind::Config, m)
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
fn opt<'a>(args: &'a [String], k: &str) -> Option<&'a str> {
    args.iter().find_map(|a| a.strip_prefix(k).and_then(|r| r.strip_prefix('=')))
}
fn read_json(p: &Path) -> Result<Value, Fault> {
    nmjson::parse(&std::fs::read(p).map_err(|e| bad(format!("{}: {e}", p.display())))?).map_err(|e| bad(format!("{}: {e}", p.display())))
}
fn write_json(p: &Path, v: &Value) -> Result<(), Fault> {
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).map_err(|e| bad(format!("{}: {e}", d.display())))?;
    }
    std::fs::write(p, v.encode().map_err(|e| bad(e.to_string()))?).map_err(|e| bad(format!("{}: {e}", p.display())))
}
fn nums(v: &Value) -> Vec<f64> {
    v.array().unwrap_or_default().iter().map(|x| if let Value::Number(t) = x { t.parse().unwrap_or(0.0) } else { 0.0 }).collect()
}
fn num_arr(x: &[f64]) -> Value {
    Value::Array(x.iter().map(|v| Value::Number(format!("{v:e}"))).collect())
}
fn mib(b: usize) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "a size in MiB")]
    let x = b as f64 / 1_048_576.0;
    x
}
fn bf16(b: &[u8], i: usize) -> f64 {
    f64::from(f32::from_bits(u32::from(u16::from_le_bytes([b.get(2 * i).copied().unwrap_or(0), b.get(2 * i + 1).copied().unwrap_or(0)])) << 16))
}
/// Indices in rising order of score (ties by index).
fn rising(score: &[f64]) -> Vec<usize> {
    let mut i: Vec<usize> = (0..score.len()).collect();
    i.sort_by(|a, b| score.get(*a).copied().unwrap_or(0.0).total_cmp(&score.get(*b).copied().unwrap_or(0.0)).then(a.cmp(b)));
    i
}

/// Free bytes on the drive holding `home` (`None` when it cannot be read).
fn free_bytes(home: &Path) -> Option<u64> {
    let out = std::process::Command::new("df").arg("-B1").arg("--output=avail").arg(home).output().ok()?;
    String::from_utf8_lossy(&out.stdout).lines().nth(1)?.trim().parse().ok()
}

/// The channels the map leaves out, per layer, every layer present (empty: copied whole), each
/// keeping at least one.
fn left_out(home: &Path, mo: &Mo, map: &str, layers: usize, inter: usize) -> Result<BTreeMap<usize, BTreeSet<usize>>, Fault> {
    let mut out: BTreeMap<usize, BTreeSet<usize>> = (0..layers).map(|l| (l, BTreeSet::new())).collect();
    if mo.role && map.starts_with("role-") {
        // `role-<share>`: the role scores, lowest first, on the energy map's per-layer budget.
        let share = map.strip_prefix("role-").ok_or_else(|| bad(format!("map={map}: the judge's map is role-<share>")))?;
        let m = read_json(&home.join(mo.maps).join("map.json"))?;
        let budget = m.get(share).and_then(|x| x.get("budget")).map(nums).map_err(|_| bad(format!("map.json has no budget at {share}")))?;
        let sc = read_json(&home.join(mo.maps).join("role.json"))?;
        let rows: Vec<Vec<f64>> = sc.array().unwrap_or_default().iter().map(nums).collect();
        for (l, r) in budget.iter().enumerate() {
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "channel counts")]
            let r = *r as usize;
            if r > 0 {
                out.insert(l, rising(rows.get(l).map_or(&[][..], Vec::as_slice)).into_iter().take(r).collect());
            }
        }
    } else {
        let v = read_json(&home.join(mo.maps).join(format!("{map}.json")))?;
        let off = v.get("off").and_then(Value::object).map_err(|e| bad(format!("{map}.json: {e}")))?;
        for l in 0..layers {
            let row = off.get(&format!("L{l}")).ok_or_else(|| bad(format!("{map}.json names no layer {l}: a base cartridge keeps every layer")))?;
            out.insert(l, row.array().unwrap_or_default().iter().filter_map(|x| x.integer().ok()).filter_map(|x| usize::try_from(x).ok()).collect());
        }
    }
    for (l, o) in &out {
        if *l >= layers || o.len() >= inter || o.iter().any(|c| *c >= inter) {
            return Err(bad(format!("{map}: layer {l}: {} channels out of {inter}", o.len())));
        }
    }
    Ok(out)
}

/// Kept channels as ascending index ranges.
fn ranges(off: &BTreeSet<usize>, inter: usize) -> Vec<(usize, usize)> {
    let mut r: Vec<(usize, usize)> = Vec::new();
    for c in (0..inter).filter(|c| !off.contains(c)) {
        match r.last_mut() {
            Some(x) if x.1 == c => x.1 = c + 1,
            _ => r.push((c, c + 1)),
        }
    }
    r
}

/// The kept rows (`cols` false) or columns of a `[rows, cols]` BF16 tensor, laid out exactly as
/// the engine's seat builds a narrowed set (`nmengn::base`).
fn cut(b: &[u8], rows: usize, cols: usize, keep: &[(usize, usize)], by_cols: bool) -> Vec<u8> {
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

/// The released tensor's bytes and shape.
fn tensor(pop: &Population, name: &str) -> Result<(std::sync::Arc<nmnpcs::View>, Vec<usize>), Fault> {
    let c = pop.clusters.iter().find(|c| c.tensor == name).ok_or_else(|| bad(format!("no tensor {name}")))?;
    Ok((pop.prov.acquire(c)?, c.shape.clone()))
}

/// The assembly at the served default precision (16 bits).
fn assembly(home: &Path, mo: &Mo) -> Result<Assembly, Fault> {
    let mut a = Assembly::load(&home.join("designs/assemblies"), mo.asm)?;
    for nd in a.nodes.iter_mut().filter(|x| x.op == "linear") {
        "f16".clone_into(&mut nd.precision);
    }
    Ok(a)
}

/// One set of the cartridge: its name, place, shape, and the kept ranges and axis when narrowed.
struct Set {
    name: String,
    layer: Option<usize>,
    block: &'static str,
    shape: Vec<usize>,
    keep: Option<(bool, Vec<(usize, usize)>)>,
}

/// Every set the MicroModel's blocks read, narrowed where the map leaves channels out.
fn sets(pop: &Population, asm: &Assembly, off: &BTreeMap<usize, BTreeSet<usize>>) -> Result<Vec<Set>, Fault> {
    let inter = pop.dims.inter;
    let mut out = Vec::new();
    for name in &nmengn::base::read_by(asm) {
        let (layer, block) = nmengn::base::address(name).ok_or_else(|| bad(format!("{name}: no layer and block for it")))?;
        let t = pop.tensors.iter().find(|t| &t.name == name).ok_or_else(|| bad(format!("{name}: not in the open weights")))?;
        if t.dtype != "BF16" {
            return Err(bad(format!("{name}: {} (the cut reads BF16)", t.dtype)));
        }
        let (r, c) = (t.shape.first().copied().unwrap_or(0), t.shape.get(1).copied().unwrap_or(0));
        let ffn = [GATE, UP, DOWN].iter().any(|x| name.ends_with(x));
        let (shape, keep) = match (ffn, layer.and_then(|l| off.get(&l)).filter(|o| !o.is_empty())) {
            (true, Some(o)) => {
                let keep = ranges(o, inter);
                let kept: usize = keep.iter().map(|(a, e)| e - a).sum();
                let by_cols = name.ends_with(DOWN);
                (if by_cols { vec![r, kept] } else { vec![kept, c] }, Some((by_cols, keep)))
            }
            _ => (t.shape.clone(), None),
        };
        out.push(Set { name: name.clone(), layer, block, shape, keep });
    }
    Ok(out)
}

fn bytes_of(shape: &[usize]) -> usize {
    shape.iter().product::<usize>() * 2
}

/// `plan`: what the cartridge holds and its size, from the map and the open weights' header.
fn plan(home: &Path, args: &[String]) -> Result<String, Fault> {
    let mo = model_of(args)?;
    let map = opt(args, "map").unwrap_or(mo.map);
    let pop = Population::open(home, mo.pop, false, false)?;
    let d = pop.dims;
    let off = left_out(home, &mo, map, d.layers, d.inter)?;
    let asm = assembly(home, &mo)?;
    let all = sets(&pop, &asm, &off)?;
    let total: usize = all.iter().map(|x| bytes_of(&x.shape)).sum();
    let open: usize = all.iter().filter_map(|x| pop.tensors.iter().find(|t| t.name == x.name)).map(|t| usize::try_from(t.end - t.start).unwrap_or(0)).sum();
    let ffn: usize = all.iter().filter(|x| x.keep.is_some()).map(|x| bytes_of(&x.shape)).sum();
    let table: usize = all.iter().filter(|x| x.layer.is_none()).map(|x| bytes_of(&x.shape)).sum();
    let out_n: usize = off.values().map(BTreeSet::len).sum();
    let mut t = String::new();
    let _ = writeln!(
        t,
        "{} base cartridge from {}/{map}: {} parameter sets, {out_n} of {} feed-forward channels out ({} per layer at most, {} layers untouched), every layer and attention block kept",
        mo.name,
        mo.maps,
        all.len(),
        d.layers * d.inter,
        off.values().map(BTreeSet::len).max().unwrap_or(0),
        off.values().filter(|o| o.is_empty()).count()
    );
    let _ = writeln!(
        t,
        "parameters {total} bytes ({:.1} MiB; open weights' sets for the job {:.1} MiB, saved {:.1} MiB); narrowed feed-forward {:.1} MiB; table and final norm {:.1} MiB; layers {:.1} MiB on the card at 16 bits (open {:.1})",
        mib(total),
        mib(open),
        mib(open.saturating_sub(total)),
        mib(ffn),
        mib(table),
        mib(total - table),
        mib(open - table)
    );
    let free = free_bytes(home);
    let _ = writeln!(
        t,
        "{}; drive free {}",
        if total > BIG { "above 1 GB: reported before writing (build needs big=yes)" } else { "within 1 GB" },
        free.map_or_else(|| "unknown".to_owned(), |f| format!("{:.2} GB, {:.2} GB after", f as f64 / 1e9, (f.saturating_sub(total as u64)) as f64 / 1e9))
    );
    Ok(t)
}

/// Each layer's compensation at its `down` node: `W_down[:, off] * mean_off` over the stream's
/// width, from the map's bias file (the mean at each left-out channel, zero elsewhere). A layer
/// with nothing left out has none.
fn compensation(home: &Path, mo: &Mo, pop: &Population, map: &str, off: &BTreeMap<usize, BTreeSet<usize>>) -> Result<Vec<nmlearn::library::Point>, Fault> {
    let inter = pop.dims.inter;
    let v = read_json(&home.join(mo.maps).join(format!("{map}-bias.json")))?;
    let b = v.get("bias").and_then(Value::object).map_err(|e| bad(format!("{map}-bias.json: {e}")))?;
    let mut points = Vec::new();
    for (l, o) in off.iter().filter(|(_, o)| !o.is_empty()) {
        let row: Vec<f64> = nums(b.get(&format!("L{l}.swigate")).ok_or_else(|| bad(format!("{map}-bias.json: no L{l}.swigate")))?);
        if row.len() != inter {
            return Err(bad(format!("{map}-bias.json L{l}: {} values, {inter} expected", row.len())));
        }
        if let Some(c) = (0..inter).find(|c| !o.contains(c) && row.get(*c).is_some_and(|x| *x != 0.0)) {
            return Err(bad(format!("{map}-bias.json L{l}: a bias at kept channel {c}: the file is not this map's")));
        }
        let (w, sh) = tensor(pop, &format!("model.layers.{l}{DOWN}"))?;
        let (rows, cols) = (sh.first().copied().unwrap_or(0), sh.get(1).copied().unwrap_or(0));
        if cols != inter {
            return Err(bad(format!("layer {l} down projection {sh:?}")));
        }
        let wb = w.tensor_bytes();
        #[expect(clippy::cast_possible_truncation, reason = "a bias at binary32")]
        let bias: Vec<f32> = (0..rows).map(|r| o.iter().map(|c| bf16(&wb, r * cols + c) * row.get(*c).copied().unwrap_or(0.0)).sum::<f64>() as f32).collect();
        points.push(nmlearn::library::Point { target: format!("L{l}.down"), gain: vec![1.0; rows], bias });
    }
    Ok(points)
}

/// The processor check through the engine's own seat: the cut is offered as sets built from the
/// open weights in RAM (nothing written), seated over the open weights' model and compared on the
/// fixed check input. Returns the largest change and the line.
fn check(mo: &Mo, pop: &std::sync::Arc<Population>, asm: &Assembly, all: &[Set], shas: &BTreeMap<String, String>) -> Result<(f32, String), Fault> {
    let open = Model::bind(pop, asm.clone())?;
    let mut parts = Vec::new();
    let mut recipes = BTreeMap::new();
    for x in all {
        let t = pop.tensors.iter().find(|t| t.name == x.name).ok_or_else(|| bad(format!("{}: not in the open weights", x.name)))?;
        match &x.keep {
            None => {
                parts.push(t.clone());
                recipes.insert(x.name.clone(), None);
            }
            Some((cols, r)) => {
                parts.push(nmnpcs::TensorDesc {
                    name: x.name.clone(),
                    dtype: "BF16".to_owned(),
                    shape: x.shape.clone(),
                    file: String::new(),
                    start: 0,
                    end: bytes_of(&x.shape) as u64,
                    sha256: shas.get(&x.name).cloned().unwrap_or_default(),
                });
                recipes.insert(x.name.clone(), Some(nmengn::base::Recipe { cols: *cols, ranges: r.clone() }));
            }
        }
    }
    let offer = nmengn::base::Offer {
        key: format!("{} cut (check)", mo.name),
        weights: pop.card.sha_main.clone(),
        parameters: String::new(),
        stated: all.iter().map(|x| (x.name.clone(), (x.layer, x.block.to_owned()))).collect(),
        parts,
        exact: false,
        bound: Some(f32::MAX),
        open: recipes,
    };
    let r = nmengn::base::seat(&open, &offer);
    nmengn::base::pull_base(mo.pop);
    let (_cart, st) = r.map_err(|e| bad(e.to_string()))?;
    Ok((st.largest, format!("processor check at 16 bits: {} outputs, {} bit-identical, largest change {:e}, in {:.1} s", st.outputs, st.identical, st.largest, st.secs)))
}

/// Names the MicroModel's current default base cartridge in `settings.json` when nothing is named
/// for it, so a newer cartridge on the shelf serves only when tried or named. Returns the line.
fn pin_default(home: &Path, mo: &Mo, lib: &nmlearn::library::Library) -> Result<String, Fault> {
    let path = home.join("settings.json");
    let mut v = read_json(&path)?;
    if let Ok(k) = v.get("base_cartridges").and_then(|b| b.get(mo.pop)) {
        return Ok(format!("settings.json already names the {} base cartridge ({})", mo.name, k.string().unwrap_or("?")));
    }
    let mut cs: Vec<&nmlearn::library::Cartridge> = lib.cartridges.iter().filter(|c| c.kind == "tensors" && c.model == mo.pop).collect();
    cs.sort_by(|a, b| (b.made.as_str(), b.version).cmp(&(a.made.as_str(), a.version)));
    let now = cs.first().map_or_else(|| "open".to_owned(), |c| c.key());
    if let Value::Object(o) = &mut v
        && let Value::Object(m) = o.entry("base_cartridges".to_owned()).or_insert_with(|| Value::Object(BTreeMap::new()))
    {
        m.insert(mo.pop.to_owned(), s(&now));
    }
    std::fs::write(&path, nmjson::pretty::pretty(&v)).map_err(|e| bad(format!("{}: {e}", path.display())))?;
    Ok(format!("settings.json: the {} base cartridge pinned to {now} (the new one serves only when tried or named)", mo.name))
}

/// `build`: the processor check, then the base cartridge and its compensation on the shelf.
#[expect(clippy::too_many_lines, reason = "one build, read top to bottom")]
fn build(home: &Path, args: &[String]) -> Result<String, Fault> {
    let t0 = Instant::now();
    let mo = model_of(args)?;
    let map = opt(args, "map").unwrap_or(mo.map);
    let id = opt(args, "id").unwrap_or(mo.id);
    let version: u64 = opt(args, "v").and_then(|x| x.parse().ok()).unwrap_or(mo.version);
    let parent = opt(args, "parent").unwrap_or(mo.parent);
    let margin: f32 = opt(args, "margin").and_then(|x| x.parse().ok()).unwrap_or(1.25);
    let pop = Population::open(home, mo.pop, true, false)?;
    let d = pop.dims;
    let off = left_out(home, &mo, map, d.layers, d.inter)?;
    let asm = assembly(home, &mo)?;
    let all = sets(&pop, &asm, &off)?;
    let total: usize = all.iter().map(|x| bytes_of(&x.shape)).sum();
    if total > BIG && opt(args, "big") != Some("yes") {
        return Err(bad(format!("the cartridge would hold {total} bytes ({:.1} MiB), above 1 GB: reported before writing; run again with big=yes once agreed", mib(total))));
    }
    if let Some(f) = free_bytes(home)
        && f < KEEP_FREE + total as u64
    {
        return Err(bad(format!("the drive has {:.2} GB free; writing {:.2} GB would leave it under 4 GB", f as f64 / 1e9, total as f64 / 1e9)));
    }
    // The narrowed sets' content hashes (checked by the seat against what it builds).
    let mut shas = BTreeMap::new();
    for x in all.iter().filter(|x| x.keep.is_some()) {
        let (v, sh) = tensor(&pop, &x.name)?;
        let (r, c) = (sh.first().copied().unwrap_or(0), sh.get(1).copied().unwrap_or(0));
        let (by_cols, keep) = x.keep.as_ref().map_or((false, &[][..]), |(b, k)| (*b, k.as_slice()));
        shas.insert(x.name.clone(), nmhash::sha256_bytes(&cut(&v.tensor_bytes(), r, c, keep, by_cols)));
    }
    let (largest, check_line) = check(&mo, &pop, &asm, &all, &shas)?;
    if !largest.is_finite() {
        return Err(bad(format!("{check_line}: not finite")));
    }
    let bound = largest * margin;
    let points = compensation(home, &mo, &pop, map, &off)?;
    let weights = pop.card.sha_main.clone();
    let mut lib = nmlearn::library::Library::open(home).map_err(|e| bad(e.to_string()))?;
    let key = format!("{id}@{version}");
    let comp_id = format!("{id}-comp");
    // The compensation's own versions count on from the last one on the shelf.
    let comp_v = 1 + lib.cartridges.iter().filter(|c| c.id == comp_id).map(|c| c.version).max().unwrap_or(0);
    let comp_key = format!("{comp_id}@{comp_v}");
    if lib.cartridge(&key).is_some() || lib.cartridge(&comp_key).is_some() {
        return Err(bad(format!("{key} or {comp_key} is already in the cartridge library")));
    }
    let pin = pin_default(home, &mo, &lib)?;
    // The cartridge's parameter sets, each a full copy of the kept values.
    let mut tensors = Vec::with_capacity(all.len());
    for x in &all {
        let (v, sh) = tensor(&pop, &x.name)?;
        let bytes = match &x.keep {
            Some((by_cols, keep)) => cut(&v.tensor_bytes(), sh.first().copied().unwrap_or(0), sh.get(1).copied().unwrap_or(0), keep, *by_cols),
            None => v.tensor_bytes().to_vec(),
        };
        tensors.push(nmlearn::library::Tensor { name: x.name.clone(), layer: x.layer, block: x.block.to_owned(), dtype: "BF16".to_owned(), shape: x.shape.clone(), bytes });
    }
    drop(pop);
    let out_n: usize = off.values().map(BTreeSet::len).sum();
    let ffn_bytes: usize = tensors.iter().filter(|t| [GATE, UP, DOWN].iter().any(|x| t.name.ends_with(x))).map(|t| t.bytes.len()).sum();
    let table: usize = tensors.iter().filter(|t| t.layer.is_none()).map(|t| t.bytes.len()).sum();
    let map_file = if mo.role && map.starts_with("role-") { format!("{}/role.json on map.json's budget at {}", mo.maps, map.trim_start_matches("role-")) } else { format!("{}/{map}.json", mo.maps) };
    let spec = nmlearn::library::CartridgeSpec {
        id: id.to_owned(),
        version,
        family: String::new(),
        domain: mo.domain.to_owned(),
        scope: "general".to_owned(),
        model: mo.pop.to_owned(),
        weights: weights.clone(),
        layers: Some((0, d.layers as u64 - 1)),
        name: format!("{}, base cartridge", mo.name),
        description: format!(
            "All the parameters the blocks of every one of the {} layers read for {}, attention complete at full shape, each set addressed to its layer and block, at the open weights' own values (BF16); inside the feed-forward blocks only the channels the map keeps. {:.0} MiB ({:.0} MiB of feed-forward blocks).",
            d.layers,
            mo.job,
            mib(total),
            mib(ffn_bytes)
        ),
        made_from: format!("{} open weights, copied unchanged by the map ({map_file}) into a full working copy at 16 bits (BF16), `nano-lab run ecart build model={} map={map}`", mo.pop, mo.key),
        effect: String::new(),
    };
    let mapped = match mo.key {
        "embed" => format!("yes: activation energy on the job's reads ({map_file}; C113: above 20 of 20 random masks, 82 percent of the open weights' pair margin, 85 with compensation)"),
        "judge" => format!("yes: role map ({map_file}; C113: above 18 of 20 random masks, beats their mean on 38 of 38, keeps and raises the open weights' pair margin)"),
        _ => format!("yes: activation energy on the writer's job prompts ({map_file})"),
    };
    let notes = obj(vec![
        ("content", s("base")),
        ("exact", Value::Bool(false)),
        ("spec", s(&format!("map={map}"))),
        ("token_table_bytes", n(table as u64)),
        ("layer_bytes", n((total - table) as u64)),
        ("parent", s(parent)),
        ("mapped", s(&mapped)),
        ("pruned", s(&format!("yes: {out_n} of {} feed-forward channels left out ({} per layer at most), attention and every layer kept", d.layers * d.inter, off.values().map(BTreeSet::len).max().unwrap_or(0)))),
        ("edits", s(&format!("compensation as its own file, {comp_key}"))),
        ("compensation", s(&comp_key)),
        ("measured", s("not yet measured on its role")),
        ("check_bound", Value::Number(format!("{bound:e}"))),
        ("check_line", s(&format!("{check_line}; bound = {margin} times the processor's largest change (the card is checked at its seat)"))),
    ]);
    let key_got = lib.add_cartridge(&spec, &nmlearn::library::Filter::Tensors(tensors), notes, Vec::new(), true).map_err(|e| bad(e.to_string()))?;
    let comp_spec = nmlearn::library::CartridgeSpec {
        id: comp_id.clone(),
        version: comp_v,
        family: String::new(),
        domain: mo.domain.to_owned(),
        scope: "general".to_owned(),
        model: mo.pop.to_owned(),
        weights,
        layers: Some((0, d.layers as u64 - 1)),
        name: format!("{}, compensation for {key_got}", mo.name),
        description: format!("At each layer's feed-forward output, the mean contribution of the channels {key_got} leaves out (their mean output over the map's reads through the open weights' down projection), added back as a bias; plugged on that base cartridge only."),
        made_from: format!("{}/{map}-bias.json and the open weights' down projections, `nano-lab run ecart build model={} map={map}`", mo.maps, mo.key),
        effect: String::new(),
    };
    let comp_notes = obj(vec![("content", s("compensation")), ("for", s(&key_got)), ("socket", s("gain 1 and a bias at each layer's down node"))]);
    let comp_got = lib.add_cartridge(&comp_spec, &nmlearn::library::Filter::Points { points, q8: None }, comp_notes, Vec::new(), true).map_err(|e| bad(e.to_string()))?;
    lib.save().map_err(|e| bad(e.to_string()))?;
    let file_of = |k: &str| lib.cartridge(k).map(|c| lib.cartridge_dir(c).join(c.files.first().map_or("", |f| f.name.as_str()))).unwrap_or_default();
    let size = |k: &str| std::fs::metadata(file_of(k)).map(|m| m.len()).unwrap_or(0);
    Ok(format!(
        "{check_line}; label bound {bound:e}\n{pin}\nbuilt {key_got}: {} ({} bytes)\nbuilt {comp_got}: {} ({} bytes)\nin {:.1} s\n",
        file_of(&key_got).display(),
        size(&key_got),
        file_of(&comp_got).display(),
        size(&comp_got),
        t0.elapsed().as_secs_f64()
    ))
}

/// `record model=writer`: channel means and energy at every layer's `swigate` over the writer's
/// job prompts on the fit half of a judged set.
fn record(home: &Path, args: &[String]) -> Result<String, Fault> {
    let t0 = Instant::now();
    let mo = model_of(args)?;
    if mo.key != "writer" {
        return Err(bad("record: model=writer (the embedder's and the judge's maps are made by `emap` and `bcart map`)"));
    }
    let set_name = opt(args, "set").unwrap_or("lane-d-tune-code.json");
    let set = crate::emap::load(home, set_name)?;
    let qs = crate::emap::part(&set, "fit");
    let gpu = opt(args, "gpu") != Some("no");
    if gpu {
        nmengn::gpu::HALF_MATH.store(true, std::sync::atomic::Ordering::Relaxed);
        nmengn::gpu::SEGMENT_ATTENTION.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let pop = Population::open(home, mo.pop, true, gpu)?;
    let d = pop.dims;
    let m = Model::bind(&pop, assembly(home, &mo)?)?;
    let targets: Vec<(String, String)> = (0..d.layers).map(|l| (nid(&format!("L{l}.swigate")), "y".to_owned())).collect();
    let mut sum = vec![vec![0.0_f64; d.inter]; d.layers];
    let mut sq = vec![vec![0.0_f64; d.inter]; d.layers];
    let (mut tokens, mut reads) = (0_usize, 0_usize);
    let dev = if gpu { Device::Gpu } else { Device::Cpu };
    for q in &qs {
        // The writer's job as served (`bcart write`): a one-line note on the right candidate's code.
        for c in q.cands.iter().filter(|c| q.right(c)) {
            let prompt = format!("{}\n\nQuestion: {}\nA one-line note on what the code above does for this question:", c.doc, q.question);
            let ids = pop.tokenizer.encode(&prompt).map_err(|e| bad(e.to_string()))?;
            let o = m.forward(dev, &ids, &targets, &RunOpts::default())?;
            for ((tg, sl), ql) in targets.iter().zip(sum.iter_mut()).zip(sq.iter_mut()) {
                let y = values(o.get(tg).ok_or_else(|| bad("no swigate"))?)?;
                for row in y.chunks_exact(d.inter) {
                    for ((s1, s2), x) in sl.iter_mut().zip(ql.iter_mut()).zip(row) {
                        let x = f64::from(*x);
                        *s1 += x;
                        *s2 += x * x;
                    }
                }
            }
            tokens += ids.len();
            reads += 1;
        }
    }
    if reads == 0 {
        return Err(bad(format!("{set_name}: no right candidate on the fit half")));
    }
    #[expect(clippy::cast_precision_loss, reason = "a mean over tokens")]
    let nt = tokens as f64;
    let (mut mean, mut energy) = (Vec::new(), Vec::new());
    for l in 0..d.layers {
        let (w, sh) = tensor(&pop, &format!("model.layers.{l}{DOWN}"))?;
        let (rows, cols) = (sh.first().copied().unwrap_or(0), sh.get(1).copied().unwrap_or(0));
        let wb = w.tensor_bytes();
        let mut norm2 = vec![0.0_f64; cols];
        for r in 0..rows {
            for (c, acc) in norm2.iter_mut().enumerate() {
                let x = bf16(&wb, r * cols + c);
                *acc += x * x;
            }
        }
        let (sl, ql) = (sum.get(l).map_or(&[][..], Vec::as_slice), sq.get(l).map_or(&[][..], Vec::as_slice));
        mean.push(num_arr(&sl.iter().map(|x| x / nt).collect::<Vec<_>>()));
        energy.push(num_arr(&ql.iter().zip(&norm2).map(|(q2, w2)| q2 / nt * w2).collect::<Vec<_>>()));
    }
    let v = obj(vec![
        ("model", s(mo.pop)),
        ("set", s(set_name)),
        ("questions", n(qs.len() as u64)),
        ("reads", n(reads as u64)),
        ("tokens", n(tokens as u64)),
        ("job", s("a one-line note on the right candidate's code for the question (bcart write's prompt), fit half")),
        ("mean", Value::Array(mean)),
        ("energy", Value::Array(energy)),
    ]);
    write_json(&home.join(mo.maps).join("record.json"), &v)?;
    Ok(format!("record: {} on {} fit questions, {reads} reads, {tokens} tokens, {} in {:.1} s\n", mo.name, qs.len(), dev.name(), t0.elapsed().as_secs_f64()))
}

/// `map model=writer [share=20]`: the activation map at `share` percent out in every layer and its
/// compensation (the left-out channels' means), each its own file.
fn map(home: &Path, args: &[String]) -> Result<String, Fault> {
    let mo = model_of(args)?;
    if mo.key == "reader" {
        return union_map(home, &mo, args);
    }
    if mo.key != "writer" {
        return Err(bad("map: model=writer or model=reader"));
    }
    let share: f64 = opt(args, "share").and_then(|x| x.parse().ok()).unwrap_or(20.0);
    let rec = read_json(&home.join(mo.maps).join("record.json"))?;
    let energy: Vec<Vec<f64>> = rec.get("energy").and_then(Value::array).unwrap_or_default().iter().map(nums).collect();
    let mean: Vec<Vec<f64>> = rec.get("mean").and_then(Value::array).unwrap_or_default().iter().map(nums).collect();
    let inter = energy.first().map_or(0, Vec::len);
    if energy.is_empty() || inter == 0 || energy.len() != mean.len() {
        return Err(bad("record.json: run `ecart record model=writer` first"));
    }
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss, reason = "a channel count")]
    let k = ((share / 100.0 * inter as f64).round() as usize).min(inter - 1);
    let tag = format!("{share}");
    let mut off = BTreeMap::new();
    let mut bias = BTreeMap::new();
    for (l, (e, mu)) in energy.iter().zip(&mean).enumerate() {
        let o: BTreeSet<usize> = rising(e).into_iter().take(k).collect();
        let b: Vec<f64> = (0..inter).map(|c| if o.contains(&c) { mu.get(c).copied().unwrap_or(0.0) } else { 0.0 }).collect();
        off.insert(format!("L{l}"), Value::Array(o.iter().map(|c| n(*c as u64)).collect()));
        bias.insert(format!("L{l}.swigate"), num_arr(&b));
    }
    write_json(
        &home.join(mo.maps).join(format!("act-{tag}.json")),
        &obj(vec![("what", s(&format!("the writer's feed-forward channels left out at {tag} percent per layer, lowest energy on its job prompts (record.json)"))), ("off", Value::Object(off))]),
    )?;
    write_json(
        &home.join(mo.maps).join(format!("act-{tag}-bias.json")),
        &obj(vec![
            ("what", s(&format!("compensation for the writer's activation map at {tag} percent: each left-out channel's mean output over its job prompts"))),
            ("socket", s("gain-bias at each layer's swigate y (gain 0 for a left-out channel, 1 elsewhere)")),
            ("bias", Value::Object(bias)),
        ]),
    )?;
    Ok(format!("map: {} channels out of {inter} in each of {} layers; {}/act-{tag}.json and act-{tag}-bias.json\n", k, energy.len(), mo.maps))
}

/// `probe [bits=8] [n=256] [file=<cartridge>]`: the embedder on the card, open weights against the
/// cut cartridge without and with its compensation: weights held on the card, seconds and tokens a
/// second over `n` pieces of the frozen tree (packed batches as the lane embeds them, a warm pass
/// first), and each arm's vectors against the open weights' (mean and least cosine).
#[expect(clippy::too_many_lines, reason = "one probe, read top to bottom")]
fn probe(home: &Path, args: &[String]) -> Result<String, Fault> {
    let bits: u8 = opt(args, "bits").and_then(|x| x.parse().ok()).unwrap_or(8);
    let count: usize = opt(args, "n").and_then(|x| x.parse().ok()).unwrap_or(256);
    let file = home.join(opt(args, "file").unwrap_or("library/cartridges/base/embedder/embedder-base-v2.cartridge"));
    let mo = model_of(&["model=embed".to_owned()])?;
    nmengn::gpu::HALF_MATH.store(true, std::sync::atomic::Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, std::sync::atomic::Ordering::Relaxed);
    // The pieces: the frozen tree's Rust files cut at about 1,200 bytes on line ends, in path order.
    let mut paths = Vec::new();
    let mut stack = vec![home.join("scratch/frozen-p2/crates")];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).map_err(|e| bad(format!("{}: {e}", d.display())))?.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                paths.push(p);
            }
        }
    }
    paths.sort();
    let mut texts: Vec<String> = Vec::new();
    'files: for p in &paths {
        let body = std::fs::read_to_string(p).unwrap_or_default();
        let mut cur = String::new();
        for line in body.lines() {
            cur.push_str(line);
            cur.push('\n');
            if cur.len() >= 1200 {
                texts.push(std::mem::take(&mut cur));
                if texts.len() >= count {
                    break 'files;
                }
            }
        }
    }
    let form = match bits {
        4 => "q4",
        8 => "q8",
        _ => "f16",
    };
    let mut out = format!("embedder probe: {} pieces of scratch/frozen-p2, linears at {bits} bits, half math, packed batches\n", texts.len());
    let mut first: Option<Vec<Vec<f32>>> = None;
    for arm in ["open weights", "cut", "cut with compensation"] {
        let pop = Population::open(home, mo.pop, true, true)?;
        let mut asm = assembly(home, &mo)?;
        for nd in asm.nodes.iter_mut().filter(|x| x.op == "linear") {
            form.clone_into(&mut nd.precision);
        }
        let open = Model::bind(&pop, asm)?;
        drop(pop);
        let (m, seat_line) = if arm == "open weights" {
            (open, String::new())
        } else {
            let offer = nmapi::neural::offer_of(&file)?;
            let (cart, st) = nmengn::base::seat(&open, &offer).map_err(|e| bad(e.to_string()))?;
            drop(open);
            let mut line = format!("; seat largest change {:e}", st.largest);
            if arm == "cut with compensation" {
                match nmapi::neural::compensation_of(home, &file, &offer.key, &cart) {
                    Some(Ok((k, _, p))) => {
                        cart.set_filters(Some(std::sync::Arc::new(p)));
                        let _ = write!(line, "; {k} plugged");
                    }
                    Some(Err(e)) => return Err(bad(format!("compensation refused: {e}"))),
                    None => return Err(bad("the cartridge names no compensation")),
                }
            }
            (cart, line)
        };
        let cap = 512;
        let tokens: usize = texts.iter().map(|t| nmengn::roles::card_embed_ids_capped(&m, t, Some(cap)).map_or(0, |v| v.len())).sum();
        let budget = nmengn::roles::batch_tokens(&m, cap + 1);
        let _ = nmengn::roles::card_embed_batch(&m, &texts[..texts.len().min(16)], cap, budget)?;
        let t = Instant::now();
        let v = nmengn::roles::card_embed_batch(&m, &texts, cap, budget)?;
        let secs = t.elapsed().as_secs_f64();
        let held = m.pop.device_bytes();
        let cos = first.as_ref().map(|f| {
            let c: Vec<f64> = f
                .iter()
                .zip(&v)
                .map(|(a, b)| {
                    let (d, na, nb) = a.iter().zip(b).fold((0.0_f64, 0.0_f64, 0.0_f64), |(d, x, y), (p, q)| (d + f64::from(*p) * f64::from(*q), x + f64::from(*p).powi(2), y + f64::from(*q).powi(2)));
                    d / (na.sqrt() * nb.sqrt()).max(1e-12)
                })
                .collect();
            #[expect(clippy::cast_precision_loss, reason = "a mean")]
            let mean = c.iter().sum::<f64>() / c.len().max(1) as f64;
            (mean, c.iter().copied().fold(1.0_f64, f64::min))
        });
        #[expect(clippy::cast_precision_loss, reason = "a rate")]
        let rate = tokens as f64 / secs.max(1e-9);
        let _ = writeln!(
            out,
            "{arm}: weights on the card {:.1} MiB; {tokens} tokens in {secs:.2} s ({rate:.0} tokens a second){}{seat_line}",
            mib(held),
            cos.map_or_else(String::new, |(a, b)| format!("; cosine with the open weights' vectors mean {a:.4}, least {b:.4}"))
        );
        if first.is_none() {
            first = Some(v);
        }
        nmengn::base::pull_base(mo.pop);
        drop(m);
    }
    Ok(out)
}

/// `map model=reader [rec=<file>[,<file>...]] [mass=0.99] [cap=30]`: the reader's map from every
/// job it does, each recorded on its own prompts in the served flow (`NANO_CHANNEL_RECORD`,
/// `nmengn::chanrec`: per job and layer the channels' sums and squares at `swigate`). Per job and
/// layer a channel's energy is its mean square times its down column's squared norm; the job needs
/// the fewest highest-energy channels that carry `mass` of its energy there. A channel is kept when
/// any job needs it (the union); the rest are left out, at most `cap` percent of a layer, never all;
/// a layer no job ran is copied whole. Written as `union.json` (`off` per layer, with each job's
/// needed counts) and `union-bias.json` (each left-out channel's mean over every recorded token).
#[expect(clippy::too_many_lines, reason = "one map, read top to bottom")]
fn union_map(home: &Path, mo: &Mo, args: &[String]) -> Result<String, Fault> {
    type Acc = BTreeMap<String, BTreeMap<usize, (f64, Vec<f64>, Vec<f64>)>>;
    let files: Vec<String> = opt(args, "rec").unwrap_or("scratch/lane-d/reader/channels.json").split(',').map(str::to_owned).collect();
    let mass: f64 = opt(args, "mass").and_then(|x| x.parse().ok()).unwrap_or(0.99);
    let cap: f64 = opt(args, "cap").and_then(|x| x.parse().ok()).unwrap_or(30.0);
    // `out=<name>`: the map file name (default `union`), so candidates at several masses sit side by side.
    let out_name = opt(args, "out").unwrap_or("union");
    let pop = Population::open(home, mo.pop, false, false)?;
    let d = pop.dims;
    let mut jobs: Acc = BTreeMap::new();
    for f in &files {
        let v = read_json(&home.join(f))?;
        if v.get("model").and_then(Value::string).ok() != Some(mo.pop) {
            return Err(bad(format!("{f}: not a record of {}", mo.pop)));
        }
        for (job, layers) in v.get("jobs").and_then(Value::object).map_err(|e| bad(format!("{f}: {e}")))? {
            for (lk, row) in layers.object().map_err(|e| bad(format!("{f}: {e}")))? {
                let Some(l) = lk.strip_prefix('L').and_then(|x| x.parse::<usize>().ok()) else { continue };
                #[expect(clippy::cast_precision_loss, reason = "a token count")]
                let toks = row.get("tokens").and_then(Value::integer).unwrap_or(0) as f64;
                let s = row.get("sum").map(nums).unwrap_or_default();
                let q = row.get("sq").map(nums).unwrap_or_default();
                if s.len() != d.inter || q.len() != d.inter {
                    return Err(bad(format!("{f}: {job} {lk}: {} channels, {} expected", s.len(), d.inter)));
                }
                let e = jobs.entry(job.clone()).or_default().entry(l).or_insert_with(|| (0.0, vec![0.0; d.inter], vec![0.0; d.inter]));
                e.0 += toks;
                for (a, b) in e.1.iter_mut().zip(&s) {
                    *a += b;
                }
                for (a, b) in e.2.iter_mut().zip(&q) {
                    *a += b;
                }
            }
        }
    }
    if jobs.is_empty() {
        return Err(bad("no job recorded: run the served flow with NANO_CHANNEL_RECORD first"));
    }
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss, reason = "a channel count")]
    let most = ((cap / 100.0 * d.inter as f64).floor() as usize).min(d.inter - 1);
    let mut off_v = BTreeMap::new();
    let mut bias_v = BTreeMap::new();
    let mut need_v = BTreeMap::new();
    let mut lines = String::new();
    let mut total_off = 0_usize;
    for l in 0..d.layers {
        let (w, sh) = tensor(&pop, &format!("model.layers.{l}{DOWN}"))?;
        let (rows, cols) = (sh.first().copied().unwrap_or(0), sh.get(1).copied().unwrap_or(0));
        let wb = w.tensor_bytes();
        let mut norm2 = vec![0.0_f64; cols];
        for r in 0..rows {
            for (c, acc) in norm2.iter_mut().enumerate() {
                let x = bf16(&wb, r * cols + c);
                *acc += x * x;
            }
        }
        let energy_of = |n: f64, q: &[f64]| -> Vec<f64> { q.iter().zip(&norm2).map(|(x, w2)| x / n * w2).collect() };
        let mut kept: BTreeSet<usize> = BTreeSet::new();
        let mut needs: BTreeMap<String, usize> = BTreeMap::new();
        let mut pooled_n = 0.0_f64;
        let mut pooled_s = vec![0.0_f64; d.inter];
        let mut best = vec![0.0_f64; d.inter];
        for (job, layers) in &jobs {
            let Some((n, s, q)) = layers.get(&l) else { continue };
            if *n <= 0.0 {
                continue;
            }
            pooled_n += n;
            for (a, b) in pooled_s.iter_mut().zip(s) {
                *a += b;
            }
            let energy = energy_of(*n, q);
            let total: f64 = energy.iter().sum();
            for (b, e) in best.iter_mut().zip(&energy) {
                *b = b.max(*e / total.max(1e-30));
            }
            let mut order = rising(&energy);
            order.reverse();
            let (mut acc, mut k) = (0.0, 0_usize);
            for c in &order {
                if acc >= mass * total {
                    break;
                }
                acc += energy.get(*c).copied().unwrap_or(0.0);
                kept.insert(*c);
                k += 1;
            }
            needs.insert(job.clone(), k);
        }
        let mut off: BTreeSet<usize> = if pooled_n > 0.0 { (0..d.inter).filter(|c| !kept.contains(c)).collect() } else { BTreeSet::new() };
        if off.len() > most {
            // Over the cap: of the channels no job needs, those with the smallest share of any job's
            // energy go first.
            let mut by: Vec<usize> = off.iter().copied().collect();
            by.sort_by(|a, b| best.get(*a).copied().unwrap_or(0.0).total_cmp(&best.get(*b).copied().unwrap_or(0.0)).then(a.cmp(b)));
            off = by.into_iter().take(most).collect();
        }
        total_off += off.len();
        let b: Vec<f64> = (0..d.inter).map(|c| if off.contains(&c) && pooled_n > 0.0 { pooled_s.get(c).copied().unwrap_or(0.0) / pooled_n } else { 0.0 }).collect();
        let _ = writeln!(lines, "L{l}: {} out of {}; needed by job: {}", off.len(), d.inter, needs.iter().map(|(k, v)| format!("{k} {v}")).collect::<Vec<_>>().join(", "));
        off_v.insert(format!("L{l}"), Value::Array(off.iter().map(|c| n(*c as u64)).collect()));
        bias_v.insert(format!("L{l}.swigate"), num_arr(&b));
        need_v.insert(format!("L{l}"), Value::Object(needs.into_iter().map(|(k, v)| (k, n(v as u64))).collect()));
    }
    let job_tokens: BTreeMap<String, Value> = jobs.iter().map(|(j, ls)| (j.clone(), Value::Number(format!("{}", ls.values().map(|x| x.0).fold(0.0, f64::max))))).collect();
    write_json(
        &home.join(mo.maps).join(format!("{out_name}.json")),
        &obj(vec![
            ("what", s(&format!("the reader's feed-forward channels left out: those no job needs (each job needs the fewest highest-energy channels carrying {mass} of its energy at a layer), at most {cap} percent of a layer"))),
            ("records", Value::Array(files.iter().map(|f| s(f)).collect())),
            ("job_tokens", Value::Object(job_tokens)),
            ("needed", Value::Object(need_v)),
            ("off", Value::Object(off_v)),
        ]),
    )?;
    write_json(
        &home.join(mo.maps).join(format!("{out_name}-bias.json")),
        &obj(vec![
            ("what", s("compensation for the reader's union map: each left-out channel's mean output over every recorded token of every job")),
            ("socket", s("gain-bias at each layer's swigate y (gain 0 for a left-out channel, 1 elsewhere)")),
            ("bias", Value::Object(bias_v)),
        ]),
    )?;
    #[expect(clippy::cast_precision_loss, reason = "a share")]
    let share = 100.0 * total_off as f64 / (d.layers * d.inter) as f64;
    Ok(format!(
        "map (union of {} jobs: {}): {total_off} channels out ({share:.1} percent); {}/{out_name}.json and {out_name}-bias.json\n{lines}",
        jobs.len(),
        jobs.keys().cloned().collect::<Vec<_>>().join(", "),
        mo.maps
    ))
}

/// `record-joint [set=lane-d-tune-code.json] [part=fit] [gpu=no]`: the reader's joint read and its
/// yes or no deep check on their own prompts (`nmapi::joint_read::read_with`, the label tails on,
/// one order, the stated group length), over each judged question's candidates, while
/// `NANO_CHANNEL_RECORD` records the reader's channels (tag `joint`). The served flow records the
/// drones', the tentacles' and the request's reads; the joint read has no served caller yet, so it
/// is driven here.
fn record_joint(home: &Path, args: &[String]) -> Result<String, Fault> {
    let t0 = Instant::now();
    let out_file = opt(args, "out");
    if out_file.is_none() && std::env::var("NANO_CHANNEL_RECORD").map_or(true, |v| v.is_empty()) {
        return Err(bad("record-joint: set NANO_CHANNEL_RECORD=<file> (the reader's channel record), or name out=<file> for a comparison arm"));
    }
    let set_name = opt(args, "set").unwrap_or("lane-d-tune-code.json");
    let set = crate::emap::load(home, set_name)?;
    let qs = crate::emap::part(&set, opt(args, "part").unwrap_or("fit"));
    let gpu = opt(args, "gpu") != Some("no");
    if gpu {
        nmengn::gpu::HALF_MATH.store(true, std::sync::atomic::Ordering::Relaxed);
        nmengn::gpu::SEGMENT_ATTENTION.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let pop = Population::open(home, "qwen3-0.6b", true, gpu)?;
    let open = Model::bind(&pop, Assembly::load(&home.join("designs/assemblies"), "qwen3-reference")?)?;
    drop(pop);
    // `arm=cart:<id@v>[+comp]`: the reader seated on a base cartridge (the engine's own seat), for
    // the jobs' comparison against the full copy; default the open weights.
    let arm = opt(args, "arm").unwrap_or("open");
    let m = match arm.strip_prefix("cart:") {
        None => open,
        Some(spec) => {
            let (key, comp) = spec.strip_suffix("+comp").map_or((spec, false), |k| (k, true));
            let lib = nmlearn::library::Library::open(home).map_err(|e| bad(e.to_string()))?;
            let c = lib.cartridge(key).ok_or_else(|| bad(format!("{key} is not in the cartridge library")))?;
            let file = lib.cartridge_dir(c).join(c.files.first().map_or("", |f| f.name.as_str()));
            let offer = nmapi::neural::offer_of(&file)?;
            let (cart, _) = nmengn::base::seat(&open, &offer).map_err(|e| bad(e.to_string()))?;
            if comp {
                match nmapi::neural::compensation_of(home, &file, &offer.key, &cart) {
                    Some(Ok((_, _, p))) => cart.set_filters(Some(std::sync::Arc::new(p))),
                    Some(Err(e)) => return Err(bad(format!("compensation refused: {e}"))),
                    None => return Err(bad(format!("{key} names no compensation"))),
                }
            }
            cart
        }
    };
    let dev = if gpu { Device::Gpu } else { Device::Cpu };
    let mut verdicts = BTreeMap::new();
    let mut pieces_read = 0_usize;
    for q in &qs {
        let pieces: Vec<nmapi::joint_read::Piece> = q
            .cands
            .iter()
            .enumerate()
            .map(|(i, c)| nmapi::joint_read::Piece { chunk: i, path: c.path.clone(), start: c.start, end: c.end, text: c.doc.clone(), ..Default::default() })
            .collect();
        pieces_read += pieces.len();
        let r = nmapi::joint_read::read_with(&m, dev, &q.question, &pieces, &[], 17, nmapi::joint_read::GROUP_FALLBACK, true, false, false)?;
        // The deep check's verdicts: each piece's yes-or-no score and the order they give.
        verdicts.insert(
            q.id.clone(),
            obj(vec![
                ("yes_no", num_arr(&r.yes_no.group_score.iter().map(|x| f64::from(*x)).collect::<Vec<_>>())),
                ("order", Value::Array(r.yes_no.order.iter().map(|i| n(*i as u64)).collect())),
                ("right", Value::Array(q.cands.iter().map(|c| Value::Bool(q.right(c))).collect())),
            ]),
        );
    }
    nmengn::chanrec::flush();
    if let Some(f) = out_file {
        write_json(&home.join(f), &obj(vec![("arm", s(arm)), ("questions", Value::Object(verdicts))]))?;
    }
    Ok(format!("record-joint: {} questions, {pieces_read} pieces read jointly with the yes or no tails, {} in {:.1} s\n", qs.len(), dev.name(), t0.elapsed().as_secs_f64()))
}

/// `record-items model=embed|judge items=<file> [gpu=no]`: the open weights (the full copy) over the
/// served reads where a cut cartridge lost (`items`: `[{"question": ..., "texts": [...]}]`, made
/// from the stage ladder's rows: each lost question with its right pieces and the piece served
/// first in their place), while `NANO_CHANNEL_RECORD` (with `NANO_CHANNEL_POP` naming the
/// MicroModel) records its channels. The embedder reads the posed question and each text as the
/// meaning lane does; the judge reads each text against the question.
fn record_items(home: &Path, args: &[String]) -> Result<String, Fault> {
    let t0 = Instant::now();
    let mo = model_of(args)?;
    if !matches!(mo.key, "embed" | "judge") {
        return Err(bad("record-items: model=embed or model=judge"));
    }
    if std::env::var("NANO_CHANNEL_POP").ok().as_deref() != Some(mo.pop) || std::env::var("NANO_CHANNEL_RECORD").map_or(true, |v| v.is_empty()) {
        return Err(bad(format!("record-items: set NANO_CHANNEL_RECORD=<file> and NANO_CHANNEL_POP={}", mo.pop)));
    }
    let items = read_json(&home.join(opt(args, "items").ok_or_else(|| bad("record-items: items=<file>"))?))?;
    let gpu = opt(args, "gpu") != Some("no");
    if gpu {
        nmengn::gpu::HALF_MATH.store(true, std::sync::atomic::Ordering::Relaxed);
        nmengn::gpu::SEGMENT_ATTENTION.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let dev = if gpu { Device::Gpu } else { Device::Cpu };
    let pop = Population::open(home, mo.pop, true, gpu)?;
    let mut m = Model::bind(&pop, Assembly::load(&home.join("designs/assemblies"), mo.asm)?)?;
    if mo.key == "judge" {
        m = m.label_rows();
    }
    let (mut q_n, mut reads) = (0_usize, 0_usize);
    for it in items.array().map_err(|e| bad(format!("items: {e}")))? {
        let question = it.get("question").and_then(Value::string).map_err(|e| bad(format!("items: {e}")))?;
        let texts: Vec<String> = it.get("texts").and_then(Value::array).unwrap_or_default().iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect();
        if mo.key == "embed" {
            let posed = format!("Instruct: {}\nQuery:{question}", nmsubs::dense::task_for(question));
            nmengn::roles::card_embed_on(&m, dev, &posed)?;
            for t in &texts {
                nmengn::roles::card_embed_on(&m, dev, t)?;
            }
            reads += texts.len() + 1;
        } else {
            nmengn::roles::card_rerank_batch(&m, dev, nmengn::roles::CODE_INSTRUCTION, question, &texts, None)?;
            reads += texts.len();
        }
        q_n += 1;
    }
    nmengn::chanrec::flush();
    Ok(format!("record-items: {} on {q_n} lost questions, {reads} reads, {} in {:.1} s\n", mo.name, dev.name(), t0.elapsed().as_secs_f64()))
}

/// `addback model=embed|judge base=<map> rec=<file>[,<file>...] [base_rec=<file>]
/// [steps=64,128,256,512,1024] [out=<name>]`: the next version's map over the current one, in steps
/// (Luis's method, refined by the lead 2026-11-19). The full copy (`record-items`, the open weights)
/// was recorded on the served reads where the trimmed cartridge lost. Each channel the current map
/// leaves out is scored by the output difference its removal makes on those reads beyond what the
/// compensation puts back: its variance there times its down column's squared norm. For each step
/// K the K highest-scoring left-out channels (over all layers) come back on top of the current map,
/// every other channel as it was. The compensation of what stays out is refitted: each channel's
/// mean over the base map's reads (`base_rec`, its `mean` and `tokens`: the embedder's
/// `record.json`, the judge's `swigate-mean.json`) and the lost reads together. Written as
/// `<out>-<K>.json` and `<out>-<K>-bias.json`; each step is measured (`emap ... file:<out>-<K>`, the
/// lost items and the other sets) and the smallest that recovers the lost items without hurting the
/// others is built (`ecart build map=<out>-<K> v=<next> parent=<current>`). The current version is
/// not touched.
#[expect(clippy::too_many_lines, reason = "one step, read top to bottom")]
fn addback(home: &Path, args: &[String]) -> Result<String, Fault> {
    let mo = model_of(args)?;
    let base = opt(args, "base").unwrap_or(mo.map);
    let out = opt(args, "out").map_or_else(|| format!("{base}-plus"), str::to_owned);
    let steps: Vec<usize> = opt(args, "steps").unwrap_or("64,128,256,512,1024").split(',').filter_map(|x| x.parse().ok()).collect();
    let base_rec = opt(args, "base_rec").map_or_else(
        || home.join(mo.maps).join(if mo.key == "judge" { "swigate-mean.json" } else { "record.json" }),
        |f| home.join(f),
    );
    let pop = Population::open(home, mo.pop, false, false)?;
    let d = pop.dims;
    let off = left_out(home, &mo, base, d.layers, d.inter)?;
    let br = read_json(&base_rec)?;
    #[expect(clippy::cast_precision_loss, reason = "a token count")]
    let n0 = br.get("tokens").and_then(Value::integer).unwrap_or(0) as f64;
    let mean0: Vec<Vec<f64>> = br.get("mean").and_then(Value::array).unwrap_or_default().iter().map(nums).collect();
    if n0 <= 0.0 || mean0.len() != d.layers {
        return Err(bad(format!("{}: needs `mean` per layer and `tokens`", base_rec.display())));
    }
    // The lost reads: tokens, sums and squares per layer, over every job and file.
    let mut rec: BTreeMap<usize, (f64, Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for f in opt(args, "rec").ok_or_else(|| bad("addback: rec=<file>"))?.split(',') {
        let v = read_json(&home.join(f))?;
        if v.get("model").and_then(Value::string).ok() != Some(mo.pop) {
            return Err(bad(format!("{f}: not a record of {}", mo.pop)));
        }
        for layers in v.get("jobs").and_then(Value::object).map_err(|e| bad(format!("{f}: {e}")))?.values() {
            for (lk, row) in layers.object().map_err(|e| bad(format!("{f}: {e}")))? {
                let Some(l) = lk.strip_prefix('L').and_then(|x| x.parse::<usize>().ok()) else { continue };
                #[expect(clippy::cast_precision_loss, reason = "a token count")]
                let toks = row.get("tokens").and_then(Value::integer).unwrap_or(0) as f64;
                let (s, q) = (row.get("sum").map(nums).unwrap_or_default(), row.get("sq").map(nums).unwrap_or_default());
                if s.len() != d.inter || q.len() != d.inter {
                    continue;
                }
                let e = rec.entry(l).or_insert_with(|| (0.0, vec![0.0; d.inter], vec![0.0; d.inter]));
                e.0 += toks;
                for (a, b) in e.1.iter_mut().zip(&s) {
                    *a += b;
                }
                for (a, b) in e.2.iter_mut().zip(&q) {
                    *a += b;
                }
            }
        }
    }
    // Score every left-out channel: variance on the lost reads times its down column's squared norm.
    let mut scored: Vec<(f64, usize, usize)> = Vec::new();
    for (l, o) in &off {
        let Some((n, s, q)) = rec.get(l).filter(|r| r.0 > 0.0) else { continue };
        let (w, sh) = tensor(&pop, &format!("model.layers.{l}{DOWN}"))?;
        let (rows, cols) = (sh.first().copied().unwrap_or(0), sh.get(1).copied().unwrap_or(0));
        let wb = w.tensor_bytes();
        let mut layer: Vec<(f64, usize, usize)> = Vec::with_capacity(o.len());
        for &c in o {
            let w2: f64 = (0..rows).map(|r| bf16(&wb, r * cols + c).powi(2)).sum();
            let m = s.get(c).copied().unwrap_or(0.0) / n;
            let var = (q.get(c).copied().unwrap_or(0.0) / n - m * m).max(0.0);
            layer.push((var * w2, *l, c));
        }
        // `norm=layer`: each score as a share of its layer's left-out total, so every layer competes
        // on what its removal does there (the late layers' larger outputs do not take every step).
        if opt(args, "norm") == Some("layer") {
            let total: f64 = layer.iter().map(|x| x.0).sum::<f64>().max(1e-30);
            for x in &mut layer {
                x.0 /= total;
            }
        }
        scored.extend(layer);
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut lines = format!("addback over {base}: {} left-out channels scored on the lost reads\n", scored.len());
    for &k in &steps {
        let back: BTreeSet<(usize, usize)> = scored.iter().take(k).map(|x| (x.1, x.2)).collect();
        let mut off_v = BTreeMap::new();
        let mut bias_v = BTreeMap::new();
        let mut per = Vec::new();
        for l in 0..d.layers {
            let o = off.get(&l).cloned().unwrap_or_default();
            let still: BTreeSet<usize> = o.iter().copied().filter(|c| !back.contains(&(l, *c))).collect();
            if o.len() != still.len() {
                per.push(format!("L{l} {}", o.len() - still.len()));
            }
            let (nr, sr) = rec.get(&l).map_or((0.0, None), |r| (r.0, Some(&r.1)));
            let m0 = mean0.get(l).map_or(&[][..], Vec::as_slice);
            let b: Vec<f64> = (0..d.inter)
                .map(|c| {
                    if still.contains(&c) {
                        (n0 * m0.get(c).copied().unwrap_or(0.0) + sr.and_then(|s| s.get(c)).copied().unwrap_or(0.0)) / (n0 + nr)
                    } else {
                        0.0
                    }
                })
                .collect();
            off_v.insert(format!("L{l}"), Value::Array(still.iter().map(|c| n(*c as u64)).collect()));
            bias_v.insert(format!("L{l}.swigate"), num_arr(&b));
        }
        let name = format!("{out}-{k}");
        write_json(
            &home.join(mo.maps).join(format!("{name}.json")),
            &obj(vec![
                ("what", s(&format!("{base} with the {k} left-out channels whose removal moves the lost reads most (variance there times the down column's squared norm) brought back"))),
                ("base", s(base)),
                ("off", Value::Object(off_v)),
            ]),
        )?;
        write_json(
            &home.join(mo.maps).join(format!("{name}-bias.json")),
            &obj(vec![("what", s(&format!("compensation for {name}, refitted on the base map's reads and the lost reads"))), ("bias", Value::Object(bias_v))]),
        )?;
        let _ = writeln!(lines, "{name}: {} channels back ({})", back.len(), per.join(", "));
    }
    Ok(lines)
}

pub fn run(args: &[String], home: &Path) -> ExitCode {
    let step = args.get(2).map_or("plan", String::as_str);
    let stamp = nmtype::now_ms();
    let result = match step {
        "plan" => plan(home, args),
        "build" => build(home, args),
        "record" => record(home, args),
        "map" => map(home, args),
        "probe" => probe(home, args),
        "record-joint" => record_joint(home, args),
        "record-items" => record_items(home, args),
        "addback" => addback(home, args),
        other => Err(bad(format!("unknown step {other} (plan, build, record, map, probe, record-joint, record-items, addback)"))),
    };
    match result {
        Ok(t) => {
            let p = home.join(format!("runs/ECART-{step}-{stamp}.txt"));
            let _ = std::fs::write(&p, format!("nano-lab {}\n{t}", args.get(1..).unwrap_or_default().join(" ")));
            print!("{t}");
            println!("report: {}", p.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("ecart {step}: {e}");
            ExitCode::FAILURE
        }
    }
}
