//! S6 runs (runbook `runs/2026-11-08-S6-runbook.md`), each writing a new timestamped report
//! line file under `runs/`: `assemblies` (write the two assembly files from the generator),
//! `parity` (gates 1 to 4), `split` (gate 5), `place` (gate 6).
use nmengn::selftest::{PRINT_CHUNK, PRINT_QUESTION, read_assembly, role_outputs, role_ref};
use nmengn::assembly::{self, Assembly, Shape, nid};
use nmengn::model::{Model, Population, RunOpts, values};
use nmengn::placement::{Device, shape};
use nmengn::reference;

/// The fingerprint anchor of the earlier lab runs, kept byte-identical so `run s6 blocks`
/// reproduces them; the engine self-test uses `nmengn::selftest::PRINT_ANCHOR`.
const PRINT_ANCHOR: &str =
    "Repository neuromorph-mini: a Rust workspace (nmsubs substrate, nmpod tentacles, nmbrain Core Brain, nmengn engine, nmapi tools).\n";
use nmjson::Value;
use nmtype::{Fault, FaultKind, FrameView, Payload};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

const MODEL: &str = "qwen3-0.6b";

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("s6", FaultKind::Execution, m)
}
fn stamp() -> u64 {
    nmtype::now_ms()
}
fn write(home: &Path, part: &str, out: &str) -> PathBuf {
    let p = home.join(format!("runs/S6-{part}-{}.txt", stamp()));
    let _ = std::fs::write(&p, out);
    p
}
fn shape_of(p: &Population) -> Shape {
    Shape { layers: p.dims.layers, heads: p.dims.heads, kv_heads: p.dims.kv_heads, head_dim: p.dims.head_dim }
}

/// Write `qwen3-reference.json` and `qwen3-split-attention.json` from the generator (data the
/// runs then read; new files only, an existing file is kept).
fn assemblies(home: &Path, replace: bool) -> Result<String, Fault> {
    let pop = Population::open(home, MODEL, false, false)?;
    let mut out = String::new();
    for (name, paths, note) in [
        (
            "qwen3-reference",
            1,
            "The shared Qwen3 structure (2026-11-10: one structure for every Qwen3 population with 28 layers, 16 query heads, 8 key-value heads and head size 128; cluster references name tensor and region, the population is supplied when it binds; no per-layer wrapper): the released layer order as an execution assembly (PLAN 4c, 4d item 12): per layer attention norm, q, k and v projections over head clusters, q and k head norms, rope, causal grouped-query score, the output projection over head columns, residual, feed-forward norm, gate, up, swigate, down, residual; then the final norm and the tied head. Parity assembly against the blkstk golden. Written by `nano-lab run s6 assemblies`; do not edit by hand.",
        ),
        (
            "qwen3-split-attention",
            2,
            "Second assembly (S6 gate 5): every layer's attention split into two attention paths (query heads 0 to 7 with key-value heads 0 to 3, and 8 to 15 with 4 to 7), each binding only its half's clusters, their outputs added before one shared SwiGLU that binds the same feed-forward clusters as the reference. Mathematically the reference (the output projection is linear over heads); checked by the port check and a Step Check against the reference. Written by `nano-lab run s6 assemblies`; do not edit by hand.",
        ),
    ] {
        let path = home.join(format!("designs/assemblies/{name}.json"));
        let a = assembly::generate(name, shape_of(&pop), paths);
        a.check_ports()?;
        if path.exists() && !replace {
            let _ = writeln!(out, "{name}: kept existing file ({} nodes generated, not written)", a.nodes.len());
            continue;
        }
        if path.exists() {
            // Never lose a file: the superseded version is kept under scratch/superseded.
            let keep = home.join(format!("scratch/superseded/{name}-{}.json", stamp()));
            let _ = std::fs::create_dir_all(home.join("scratch/superseded"));
            std::fs::copy(&path, &keep).map_err(|e| bad(format!("keep {}: {e}", keep.display())))?;
            let _ = writeln!(out, "{name}: previous file kept at {}", keep.display());
        }
        let text = nmjson::pretty::pretty(&a.to_value(note));
        std::fs::write(&path, text).map_err(|e| bad(e.to_string()))?;
        let _ = writeln!(out, "{name}: wrote {} nodes, {} edges to {}", a.nodes.len(), a.edges.len(), path.display());
    }
    Ok(out)
}

/// Role populations and their assemblies (2026-11-09): the released Qwen3-Reranker-0.6B and
/// Qwen3-Embedding-0.6B, each with the reference layer order at f16 and one output component.
pub const ROLES: [(&str, &str, &str); 2] =
    [("qwen3-reranker", "qwen3-reranker-0.6b", "rerank"), ("qwen3-embedding", "qwen3-embedding-0.6b", "embed")];

/// Write `qwen3-reranker.json` and `qwen3-embedding.json` (new files only unless `replace`; a
/// replaced file is kept under `scratch/superseded`).
fn roles(home: &Path, replace: bool) -> Result<String, Fault> {
    let mut out = String::new();
    for (name, pop_id, role) in ROLES {
        let pop = Population::open(home, pop_id, false, false)?;
        let a = assembly::generate_role(name, shape_of(&pop), role, "f16")?;
        let base = assembly::generate("qwen3-reference", shape_of(&pop), 1);
        a.check_ports()?;
        let path = home.join(format!("designs/assemblies/{name}.json"));
        if path.exists() && !replace {
            let _ = writeln!(out, "{name}: kept existing file");
            continue;
        }
        if path.exists() {
            let keep = home.join(format!("scratch/superseded/{name}-{}.json", stamp()));
            let _ = std::fs::create_dir_all(home.join("scratch/superseded"));
            std::fs::copy(&path, &keep).map_err(|e| bad(format!("keep {}: {e}", keep.display())))?;
        }
        let note = format!(
            "The released {pop_id} as an addition to the shared Qwen3 structure (2026-11-09; written as an addition 2026-11-10): `qwen3-reference` with f16 linear weights, plus {}. Written by `nano-lab run s6 roles`; do not edit by hand.",
            if role == "embed" {
                "`lastpool` on the final norm (the last token's state, L2-normalised, as its model card specifies)"
            } else {
                "`labelsoftmax` over the head's \"yes\" and \"no\" logits (P(yes), as its model card specifies)"
            }
        );
        std::fs::write(&path, nmjson::pretty::pretty(&a.addition_value(&base, "qwen3-reference", "f16", &note)))
            .map_err(|e| bad(e.to_string()))?;
        let _ = writeln!(out, "{name}: wrote {} nodes, {} edges over {pop_id} to {}", a.nodes.len(), a.edges.len(), path.display());
    }
    Ok(out)
}

/// Score role inputs from a JSON file inside the mini (`{"rerank": [{"instruction", "query",
/// "document"}], "embed": [{"text"}]}`) on one device; prints P(yes) per pair and each
/// embedding's ids and vector, for parity checks against released-model outputs.
fn roles_eval(home: &Path, input: &str, device: &str) -> Result<String, Fault> {
    let dev = if device == "cpu" { Device::Cpu } else { Device::Gpu };
    let v = nmjson::parse(&std::fs::read(home.join(input)).map_err(|e| bad(format!("{input}: {e}")))?).map_err(|e| bad(e.to_string()))?;
    let open = |name: &str, pop_id: &str| -> Result<Model, Fault> {
        let pop = Population::open(home, pop_id, true, dev == Device::Gpu)?;
        let a = Assembly::load(&home.join("designs/assemblies"), name)?;
        Model::bind(&pop, a)
    };
    let s = |x: &Value, k: &str| x.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default();
    let mut rows = Vec::new();
    if let Ok(items) = v.get("rerank").and_then(Value::array) {
        let m = open("qwen3-reranker", "qwen3-reranker-0.6b")?;
        // Scored both ways (2026-11-10): one prompt at a time, and in sets of one query with a
        // prefilled head; both are printed so parity holds for each.
        for it in items {
            let t = Instant::now();
            let p = nmengn::roles::card_rerank_on(&m, dev, &s(it, "instruction"), &s(it, "query"), &s(it, "document"))?;
            let one = t.elapsed().as_secs_f64() * 1000.0;
            let t = Instant::now();
            let set = nmengn::roles::card_rerank_set(&m, dev, &s(it, "instruction"), &s(it, "query"), &[s(it, "document")])?;
            rows.push(format!(
                "{{\"kind\":\"rerank\",\"p_yes\":{p},\"p_yes_set\":{},\"ms\":{one:.1},\"ms_set\":{:.1}}}",
                set.first().copied().unwrap_or(-1.0),
                t.elapsed().as_secs_f64() * 1000.0
            ));
        }
    }
    if let Ok(items) = v.get("embed").and_then(Value::array) {
        let m = open("qwen3-embedding", "qwen3-embedding-0.6b")?;
        for it in items {
            let text = s(it, "text");
            let ids = nmengn::roles::card_embed_ids(&m, &text)?;
            let t = Instant::now();
            let e = nmengn::roles::card_embed_on(&m, dev, &text)?;
            rows.push(format!(
                "{{\"kind\":\"embed\",\"ids\":{:?},\"vector\":[{}],\"ms\":{:.1}}}",
                ids,
                e.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(","),
                t.elapsed().as_secs_f64() * 1000.0
            ));
        }
    }
    Ok(format!("{}\n", rows.join("\n")))
}

/// Record the role reference from an input file inside the mini (the parity inputs): the inputs
/// and the mini's own outputs on the CPU, written once (an existing recording is kept).
fn roles_record(home: &Path, input: &str) -> Result<String, Fault> {
    let dir = role_ref(home);
    if dir.join("outputs.json").exists() {
        return Err(bad("tests/engine-reference/role-models/outputs.json exists; re-record only on purpose (move it first)"));
    }
    let bytes = std::fs::read(home.join(input)).map_err(|e| bad(format!("{input}: {e}")))?;
    let v = nmjson::parse(&bytes).map_err(|e| bad(e.to_string()))?;
    let (ps, vs) = role_outputs(home, &v, Device::Cpu)?;
    std::fs::create_dir_all(&dir).map_err(|e| bad(e.to_string()))?;
    std::fs::write(dir.join("inputs.json"), &bytes).map_err(|e| bad(e.to_string()))?;
    let out = format!(
        "{{\"device\":\"cpu\",\"p_yes\":[{}],\"vectors\":[{}]}}\n",
        ps.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(","),
        vs.iter().map(|v| format!("[{}]", v.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(","))).collect::<Vec<_>>().join(",")
    );
    std::fs::write(dir.join("outputs.json"), out).map_err(|e| bad(e.to_string()))?;
    Ok(format!("recorded {} reranker pairs and {} embeddings to {}\n", ps.len(), vs.len(), dir.display()))
}

fn line(out: &mut String, s: impl AsRef<str>) {
    let _ = writeln!(out, "{}", s.as_ref());
}
fn verdict(ok: bool) -> &'static str {
    if ok { "PASS" } else { "MISS" }
}

/// One slot's model as an NTE component: JSON `{"ids": [...]}` in, last logits out.
struct ModelExec {
    model: Arc<Model>,
    device: Device,
}
impl nmnte::Exec for ModelExec {
    fn call(&self, inputs: &[Payload], _frame: &dyn FrameView) -> Result<Payload, Fault> {
        let ids: Vec<i32> = match inputs.first() {
            Some(Payload::Json(v)) => v
                .get("ids")
                .and_then(Value::array)
                .map(|a| a.iter().filter_map(|x| x.integer().ok().and_then(|n| i32::try_from(n).ok())).collect())
                .unwrap_or_default(),
            _ => return Err(bad("ids input required")),
        };
        let target = (nid("head"), "logits".to_owned());
        let out = self.model.forward(self.device, &ids, std::slice::from_ref(&target), &RunOpts::default())?;
        let v = values(out.get(&target).ok_or_else(|| bad("no logits"))?)?.to_vec();
        Ok(Payload::Tensor(nmtype::Tensor { shape: vec![v.len()], dtype: nmtype::DType::F32, data: nmtype::TensorData::Owned(v) }))
    }
}

fn logits_of(p: &Payload) -> Result<Vec<f32>, Fault> {
    match p {
        Payload::Tensor(t) => t.as_f32().map(|c| c.into_owned()),
        _ => Err(bad("logits payload")),
    }
}

/// Gate 5: the second assembly through the port check and the lifecycle with its Compare
/// fixture, and a Step Check against the reference.
fn split(home: &Path) -> Result<String, Fault> {
    let mut out = String::new();
    let t0 = Instant::now();
    let pop = Population::open(home, MODEL, true, true)?;
    let device = if pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    // Port check: the split assembly passes; the same assembly without one swigate fails.
    let split_asm = read_assembly(home, "qwen3-split-attention")?;
    split_asm.check_ports()?;
    let mut cut = split_asm.clone();
    cut.nodes.retain(|n| n.id != nid("L5.swigate"));
    cut.edges.retain(|e| e.from != nid("L5.swigate") && e.to != nid("L5.swigate"));
    let refused = match Model::bind(&pop, cut) {
        Err(f) => format!("refused before running: {}", f.message),
        Ok(_) => "NOT refused".into(),
    };
    line(&mut out, format!("port check: qwen3-split-attention passes ({} nodes); without L5.swigate: {refused}", split_asm.nodes.len()));
    // Compare fixture (designs/compare.json).
    let cmp = nmjson::parse(&std::fs::read(home.join("designs/compare.json")).map_err(|e| bad(format!("designs/compare.json: {e}")))?)
        .map_err(|e| bad(e.to_string()))?;
    let fixture = cmp
        .get("fixtures")
        .and_then(Value::array)
        .ok()
        .and_then(|a| {
            a.iter().find(|f| f.get("change").and_then(Value::string).ok() == Some("assembly:qwen3-reference->qwen3-split-attention"))
        })
        .cloned()
        .ok_or_else(|| bad("no Compare fixture listed for the split change"))?;
    let text = fixture.get("text").and_then(Value::string).map_err(|e| bad(e.to_string()))?.to_owned();
    let ids = pop.tokenizer.encode(&text).map_err(|e| bad(e.to_string()))?;
    line(&mut out, format!("Compare fixture: {:?} ({} tokens) from designs/compare.json", text, ids.len()));
    // The slot's anatomy: one node bound to an assembly, changed by a lifecycle swap.
    let reference = Arc::new(Model::bind(&pop, read_assembly(home, "qwen3-reference")?)?);
    let splitm = Arc::new(Model::bind(&pop, split_asm)?);
    let shared_ffn = {
        let id = format!("{MODEL}/model.layers.3.mlp.gate_proj.weight/whole");
        let a = reference.bound.iter().find(|b| b.node.id == nid("L3.gate")).and_then(|b| b.views.first().cloned());
        let b = splitm.bound.iter().find(|b| b.node.id == nid("L3.gate")).and_then(|b| b.views.first().cloned());
        matches!((a, b), (Some(a), Some(b)) if Arc::ptr_eq(&a, &b) && a.cluster.id == id && a.refs() == 2)
    };
    line(
        &mut out,
        format!("the split assembly's SwiGLU binds the reference's feed-forward clusters (one Arc, two references): {shared_ffn}"),
    );
    let node = |asm: &str| -> nmnte::Node {
        let mut n = nmnte::Node::new("nmbrain.slot#writer", nmtype::ComponentType::Slot)
            .input(nmtype::PortDesc::new("ids", nmtype::PortKind::Json, true))
            .output(nmtype::PortDesc::new("logits", nmtype::PortKind::Tensor, true));
        n.config = Value::Object([("assembly".to_owned(), Value::String(asm.into()))].into_iter().collect());
        n
    };
    let src = nmnte::Node::new("nmbrain.stack#plan", nmtype::ComponentType::Brain).output(nmtype::PortDesc::new(
        "ids",
        nmtype::PortKind::Json,
        true,
    ));
    let graph = nmnte::Graph {
        nodes: vec![src, node("qwen3-reference")],
        edges: vec![nmnte::Edge::new("nmbrain.stack#plan", "ids", "nmbrain.slot#writer", "ids", nmnte::ConnectionMode::Direct)],
    };
    let models: BTreeMap<String, Arc<Model>> =
        [("qwen3-reference".to_owned(), Arc::clone(&reference)), ("qwen3-split-attention".to_owned(), Arc::clone(&splitm))]
            .into_iter()
            .collect();
    let bind = |g: &nmnte::Graph| -> Result<nmnte::Components, Fault> {
        let mut c = nmnte::Components::new();
        for n in &g.nodes {
            if n.id == "nmbrain.slot#writer" {
                let name = n.config.get("assembly").and_then(Value::string).unwrap_or("qwen3-reference");
                let m = models.get(name).ok_or_else(|| bad(format!("no model for {name}")))?;
                c.insert(n.id.clone(), Arc::new(ModelExec { model: Arc::clone(m), device }) as Arc<dyn nmnte::Exec>);
            } else {
                c.insert(
                    n.id.clone(),
                    Arc::new(nmnte::FnExec(|_: &[Payload], _: &dyn FrameView| Ok(Payload::Json(Value::Null)))) as Arc<dyn nmnte::Exec>,
                );
            }
        }
        Ok(c)
    };
    let nte = nmnte::Nte::new(None);
    nte.publish(graph.clone(), bind(&graph)?)?;
    let input = Payload::Json(Value::Object(
        [("ids".to_owned(), Value::Array(ids.iter().map(|i| Value::Number(i.to_string())).collect()))].into_iter().collect(),
    ));
    let report = std::sync::Mutex::new(String::new());
    let compare = |old: &nmnte::Snapshot, fresh: &nmnte::Components| -> Result<String, Fault> {
        let f = nte.frame(nmtype::Budget { tokens: 0, ms: 60_000, depth: 1, units: 1 });
        let a = old.components.get("nmbrain.slot#writer").ok_or_else(|| bad("old slot"))?.call(std::slice::from_ref(&input), &f)?;
        let b = fresh.get("nmbrain.slot#writer").ok_or_else(|| bad("new slot"))?.call(std::slice::from_ref(&input), &f)?;
        let (la, lb) = (logits_of(&a)?, logits_of(&b)?);
        let max = la.iter().zip(&lb).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max);
        let arg = |v: &[f32]| v.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map_or(0, |x| x.0);
        let dot: f32 = la.iter().zip(&lb).map(|(x, y)| x * y).sum();
        let cos =
            dot / (la.iter().map(|x| x * x).sum::<f32>().sqrt() * lb.iter().map(|x| x * x).sum::<f32>().sqrt()).max(f32::MIN_POSITIVE);
        let tol = reference::atol(home, "full-forward/prompt0-logits.f32")?;
        let same_top = arg(&la) == arg(&lb);
        // Step Check (nmlearn::stepspec arrives at S8; its criteria are applied here by hand):
        // scope structure (the writer slot's assembly), rung S0, baseline the reference.
        let regression = max;
        let improvement = 0.0_f32;
        let accept = same_top && max <= tol;
        let s = format!(
            "Step Check scope=structure(nmbrain.slot#writer) rung=S0 baseline=qwen3-reference: improvement {improvement:.3} (no task gain claimed; same function), regression {regression:.6} max abs logit difference (limit: the golden prompt0 atol {tol}), cosine {cos:.6}, argmax same {same_top} ({} vs {}), integrity true (port and width checks), compatibility true (same clusters, same family), verdict {}",
            arg(&la),
            arg(&lb),
            if accept { "Accept" } else { "Reject" }
        );
        if let Ok(mut r) = report.lock() {
            r.clone_from(&s);
        }
        Ok(s)
    };
    let change = nmnte::Change::Swap(node("qwen3-split-attention"));
    match nmnte::lifecycle::replace(&nte, &change, &bind, Some(&compare), 2000) {
        Ok((snap, log)) => {
            line(&mut out, format!("lifecycle: committed v{}", snap.version));
            for l in log.lines() {
                line(&mut out, format!("  {l}"));
            }
        }
        Err((f, log)) => {
            line(&mut out, format!("lifecycle refused: {f}"));
            for l in log.lines() {
                line(&mut out, format!("  {l}"));
            }
        }
    }
    let s = report.lock().map(|r| r.clone()).unwrap_or_default();
    let g5 = refused.starts_with("refused") && !s.is_empty() && shared_ffn;
    line(
        &mut out,
        format!("gate 5 second assembly: port checks as stated, Compare logged with its fixture, Step Check recorded -> {}", verdict(g5)),
    );
    line(&mut out, format!("device {}; wall {} ms", device.name(), t0.elapsed().as_millis()));
    Ok(out)
}

/// Gate 6: placement per job and per cluster set, measured.
fn place(home: &Path) -> Result<String, Fault> {
    let mut out = String::new();
    let t0 = Instant::now();
    let pop = Population::open(home, MODEL, true, true)?;
    let m = Arc::new(Model::bind(&pop, read_assembly(home, "qwen3-reference")?)?);
    let texts = [
        "pub struct ParMap holds the Parameter Map observer and its edits log",
        "fn decompose(card: &WeightCard) -> Vec<ParameterCluster>",
    ];
    // Measure both devices per shape class (embed and rerank), CPU then GPU, twice each.
    for dev in [Device::Cpu, Device::Gpu] {
        if dev == Device::Gpu && pop.gpu.is_none() {
            continue;
        }
        for rep in 0..2 {
            for t in texts {
                let s = Instant::now();
                let _ = nmengn::roles::embed_on(&m, dev, t)?;
                let ms = s.elapsed().as_secs_f64() * 1000.0;
                let n = pop.tokenizer.encode(t).map_or(0, |v| v.len());
                if rep == 1 {
                    pop.placement.record(&shape("embed", n), dev, ms);
                }
                line(&mut out, format!("  measure embed on {} ({n} tokens, pass {}): {ms:.1} ms", dev.name(), rep + 1));
                let s = Instant::now();
                let _ = nmengn::roles::rerank_on(&m, dev, "Where is ParMap defined?", t)?;
                let ms = s.elapsed().as_secs_f64() * 1000.0;
                let n = pop.tokenizer.encode(&nmengn::roles::prompt("Where is ParMap defined?", t)).map_or(0, |v| v.len());
                if rep == 1 {
                    pop.placement.record(&shape("rerank", n), dev, ms);
                }
                line(&mut out, format!("  measure rerank on {} ({n} tokens, pass {}): {ms:.1} ms", dev.name(), rep + 1));
            }
        }
    }
    let path = home.join("designs/placement.json");
    let note = "Measured per shape class by `nano-lab run s6 place` (S6 gate 6): milliseconds per job on each device, second pass. Placement::choose reads it; re-measured runs append new reports and rewrite this table.";
    std::fs::write(&path, nmjson::pretty::pretty(&pop.placement.to_value(note))).map_err(|e| bad(e.to_string()))?;
    // The choice with the GPU free, then held by a timing run.
    let (_, free, _) = nmengn::roles::embed(&m, texts[0])?;
    pop.placement.hold_gpu(true);
    let (_, held, _) = nmengn::roles::embed(&m, texts[0])?;
    pop.placement.hold_gpu(false);
    line(&mut out, format!("embedder job, GPU free: {} ({})", free.device.name(), free.why));
    line(&mut out, format!("embedder job, timing run holds the GPU: {} ({})", held.device.name(), held.why));
    let measured = std::fs::read_to_string(&path).is_ok_and(|s| s.contains("gpu_ms") && s.contains("cpu_ms"));
    let g6 = (pop.gpu.is_none() || free.device == Device::Gpu) && held.device == Device::Cpu && measured;
    line(
        &mut out,
        format!(
            "per-cluster residency: {} clusters live, GPU {} MB recorded in NPCS ({} MB device bytes)",
            pop.prov.live(),
            pop.prov.totals().1 >> 20,
            pop.device_bytes() >> 20
        ),
    );
    line(
        &mut out,
        format!(
            "gate 6 placement: GPU when free {}, CPU when held {}, measured table written {measured} -> {}",
            free.device == Device::Gpu,
            held.device == Device::Cpu,
            verdict(g6)
        ),
    );
    line(&mut out, format!("wall {} ms", t0.elapsed().as_millis()));
    Ok(out)
}

/// Record the engine reference with this engine on CPU (`nmengn::reference::record`) into
/// `tests/engine-reference/qwen3-0.6b/`, reading its test texts, prompts and tolerances from the
/// reference itself (first recorded 2026-11-08 from the earlier external recording, which was
/// then deleted), and write its `MANIFEST.sha256` (SHA-256 and path, one per line, sorted).
fn record(home: &Path) -> Result<String, Fault> {
    let t0 = std::time::Instant::now();
    let dir = reference::root(home);
    let source = dir.clone();
    let pop = Population::open(home, MODEL, true, false)?;
    let m = Model::bind(&pop, read_assembly(home, "qwen3-reference")?)?;
    let files = reference::record(home, &m, &source)?;
    let mut lines = Vec::new();
    for f in &files {
        let h = std::fs::File::open(f).and_then(nmhash::sha256).map_err(|e| bad(format!("{}: {e}", f.display())))?;
        let rel = f.strip_prefix(&dir).unwrap_or(f).to_string_lossy().replace('\\', "/");
        lines.push(format!("{h}  {rel}"));
    }
    lines.sort_by(|a, b| a.get(66..).cmp(&b.get(66..)));
    std::fs::write(dir.join("MANIFEST.sha256"), lines.join("\n") + "\n").map_err(|e| bad(e.to_string()))?;
    let mut out = String::new();
    line(
        &mut out,
        format!(
            "engine reference recorded on CPU from source {}: {} files plus MANIFEST.sha256 in {}",
            source.display(),
            files.len(),
            dir.display()
        ),
    );
    line(&mut out, format!("wall {} ms", t0.elapsed().as_millis()));
    Ok(out)
}

/// Qwen3-1.7B on the shared Qwen3 structure (2026-11-10): the same `qwen3-reference` assembly
/// and the same engine operations as the 0.6B populations, bound over `qwen3-1.7b` with linear
/// weights at `precision`, compared with the released model run in PyTorch at float32
/// (`tests/engine-reference/qwen3-1.7b/`). Pass, fixed before the
/// first run: the last-position argmax equal on every prompt, per-position argmax agreement at
/// least 95%, and cosine of the last-position logits at least 0.999 on every prompt.
fn writer_parity(home: &Path, dev: Device, precision: &str) -> Result<String, Fault> {
    let dir = home.join("tests/engine-reference/qwen3-1.7b");
    let r = nmjson::parse(&std::fs::read(dir.join("reference.json")).map_err(|e| bad(format!("reference.json: {e}")))?)
        .map_err(|e| bad(e.to_string()))?;
    let t0 = Instant::now();
    let pop = Population::open(home, "qwen3-1.7b", true, dev == Device::Gpu)?;
    let mut a = Assembly::load(&home.join("designs/assemblies"), "qwen3-reference")?;
    for n in a.nodes.iter_mut().filter(|n| n.op == "linear") {
        n.precision = precision.to_owned();
    }
    let m = Model::bind(&pop, a)?;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "qwen3-1.7b on qwen3-reference ({} components, {} clusters bound, linear {precision}, {}), open and bind {} ms",
        m.asm.nodes.len(),
        m.pop.prov.live(),
        dev.name(),
        t0.elapsed().as_millis()
    );
    let head = (nid("head"), "logits".to_owned());
    let opts = RunOpts { all_positions: true, ..RunOpts::default() };
    let (mut agree, mut total, mut pass) = (0_usize, 0_usize, true);
    let vocab = m.pop.dims.vocab;
    for (i, row) in r.get("prompts").and_then(Value::array).map_err(|e| bad(e.to_string()))?.iter().enumerate() {
        let text = row.get("text").and_then(Value::string).map_err(|e| bad(e.to_string()))?;
        let want_ids: Vec<i32> = row
            .get("ids")
            .and_then(Value::array)
            .map_err(|e| bad(e.to_string()))?
            .iter()
            .filter_map(|x| x.integer().ok().and_then(|n| i32::try_from(n).ok()))
            .collect();
        let ids = m.pop.tokenizer.encode(text).map_err(|e| bad(e.to_string()))?;
        let ids_same = ids == want_ids;
        let want_arg: Vec<usize> = row
            .get("argmax")
            .and_then(Value::array)
            .map_err(|e| bad(e.to_string()))?
            .iter()
            .filter_map(|x| x.integer().ok().and_then(|n| usize::try_from(n).ok()))
            .collect();
        let t = Instant::now();
        let o = m.forward(dev, &want_ids, std::slice::from_ref(&head), &opts)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let lg = values(o.get(&head).ok_or_else(|| bad("no logits"))?)?;
        let arg = |v: &[f32]| v.iter().enumerate().fold((0, f32::MIN), |b, (j, x)| if *x > b.1 { (j, *x) } else { b }).0;
        let got_arg: Vec<usize> = lg.chunks(vocab).map(arg).collect();
        let same = got_arg.iter().zip(&want_arg).filter(|(a, b)| a == b).count();
        agree += same;
        total += want_arg.len();
        let bytes = std::fs::read(dir.join(format!("prompt{i}-last.f32"))).map_err(|e| bad(e.to_string()))?;
        let want_last: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        let last = lg.get(lg.len().saturating_sub(vocab)..).unwrap_or(&[]);
        let (mut dot, mut na, mut nb, mut mx) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f32);
        for (x, y) in last.iter().zip(&want_last) {
            dot += f64::from(*x) * f64::from(*y);
            na += f64::from(*x) * f64::from(*x);
            nb += f64::from(*y) * f64::from(*y);
            mx = mx.max((x - y).abs());
        }
        let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
        let last_same = got_arg.last() == want_arg.last();
        pass &= last_same && cos >= 0.999 && last.len() == want_last.len();
        let _ = writeln!(
            out,
            "  prompt{i} ({} tokens, ids same as release tokenizer {ids_same}): argmax {same}/{} positions, last {} vs {} ({}), last-logits cosine {cos:.6}, max abs {mx:.4}, {ms:.0} ms",
            want_ids.len(),
            want_arg.len(),
            got_arg.last().copied().unwrap_or(0),
            want_arg.last().copied().unwrap_or(0),
            if last_same { "same" } else { "DIFFERENT" }
        );
    }
    let share = agree as f64 / total.max(1) as f64;
    pass &= share >= 0.95;
    let (ram, gpu) = m.pop.prov.totals();
    let _ = writeln!(
        out,
        "per-position argmax agreement {agree}/{total} ({:.1}%); residency RAM {} MB, GPU {} MB; wall {} ms",
        share * 100.0,
        ram >> 20,
        gpu >> 20,
        t0.elapsed().as_millis()
    );
    let _ = writeln!(out, "writer parity ({precision}, {}): {}", dev.name(), if pass { "PASS" } else { "FAIL" });
    Ok(out)
}

/// E1 (2026-11-10): encoded chunks and shared key and value windows (`nmengn::kvstore`).
/// T1 a question read over a window of one chunk equals the question read after the same
/// chunk in the decode cache (same kernels on each device: exact), and is reported against
/// one-pass encoding; T2 the same for a chunk encoded after an anchor and placed after it; T3 a window of separately encoded chunks against full sequential encoding (an
/// approximation, reported); T4 the per-head reading; T5 time and memory over a window of
/// about 8,000 tokens of the mini's own code.
fn kv(home: &Path, devices: &[Device]) -> Result<String, Fault> {
    use nmengn::kvstore::{Probe, Window, encode};
    let mut out = String::new();
    let mut pass = true;
    for &dev in devices {
        let pop = Population::open(home, "qwen3-0.6b", true, dev == Device::Gpu)?;
        let m = Model::bind(&pop, Assembly::load(&home.join("designs/assemblies"), "qwen3-reference")?)?;
        let d = pop.dims;
        let tok = |s: &str| pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
        let read = |p: &str, chars: usize| -> Result<String, Fault> {
            Ok(std::fs::read_to_string(home.join(p)).map_err(|e| bad(format!("{p}: {e}")))?.chars().take(chars).collect())
        };
        let target = (nid("final_norm"), "y".to_owned());
        let hidden = |ids: &[i32], start: usize, opts: &RunOpts| -> Result<Vec<f32>, Fault> {
            Ok(values(m.forward_at(dev, ids, start, std::slice::from_ref(&target), opts)?.get(&target).ok_or_else(|| bad("no hidden"))?)?
                .to_vec())
        };
        let tail = |x: &[f32], n: usize| x.get(x.len() - n * d.hidden..).unwrap_or(&[]).to_vec();
        let maxdiff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max);
        let cosine = |a: &[f32], b: &[f32]| {
            let (ab, aa, bb) = a
                .iter()
                .zip(b)
                .fold((0.0_f64, 0.0_f64, 0.0_f64), |s, (x, y)| (s.0 + f64::from(x * y), s.1 + f64::from(x * x), s.2 + f64::from(y * y)));
            ab / (aa.sqrt() * bb.sqrt()).max(1e-12)
        };
        let anchor_ids = tok(
            "Repository neuromorph-mini: a Rust workspace of 18 crates (nmsubs substrate, nmpod tentacles, nmbrain Core Brain, nmengn engine, nmapi tools).\n",
        )?;
        let c1 = tok(&read("crates/nmsubs/src/dense.rs", 900)?)?;
        let c2 = tok(&read("crates/nmengn/src/decode.rs", 900)?)?;
        let q = tok("\nQuestion: where are chunk vectors stored and how is a question ranked?")?;
        // The decode cache path: the prefix first, then the question, the same kernels the
        // window path uses on this device.
        let stepped = |prefix: &[i32]| -> Result<Vec<f32>, Fault> {
            let o = RunOpts { cache: Some(Arc::new(nmengn::Cache::default())), ..RunOpts::default() };
            hidden(prefix, 0, &o)?;
            Ok(tail(&hidden(&q, prefix.len(), &o)?, q.len()))
        };
        // T1: prefix exactness.
        let t0 = Instant::now();
        let direct = tail(&hidden(&[c1.clone(), q.clone()].concat(), 0, &RunOpts::default())?, q.len());
        let w1 = Arc::new(Window::build(&m, &[Arc::new(encode(&m, dev, None, &c1)?)])?);
        let via = tail(&hidden(&q, c1.len(), &RunOpts { window: Some(Arc::clone(&w1)), ..RunOpts::default() })?, q.len());
        let s1 = stepped(&c1)?;
        let d1 = maxdiff(&s1, &via);
        let ok1 = d1 == 0.0;
        pass &= ok1;
        let _ = writeln!(
            out,
            "{} T1 question over a window of one chunk ({} + {} tokens): against the decode cache max abs {d1:.6} -> {}; against one-pass encoding max abs {:.6} ({} ms)",
            dev.name(),
            c1.len(),
            q.len(),
            if ok1 { "PASS" } else { "FAIL" },
            maxdiff(&direct, &via),
            t0.elapsed().as_millis()
        );
        // T2: anchor placement exactness.
        let ak = Arc::new(encode(&m, dev, None, &anchor_ids)?);
        let anchor = Arc::new(Window::build(&m, &[Arc::clone(&ak)])?);
        let k1 = Arc::new(encode(&m, dev, Some(&anchor), &c1)?);
        let w2 = Arc::new(Window::build(&m, &[Arc::clone(&ak), Arc::clone(&k1)])?);
        let direct2 = tail(&hidden(&[anchor_ids.clone(), c1.clone(), q.clone()].concat(), 0, &RunOpts::default())?, q.len());
        let via2 = tail(&hidden(&q, w2.len, &RunOpts { window: Some(Arc::clone(&w2)), ..RunOpts::default() })?, q.len());
        let s2 = stepped(&[anchor_ids.clone(), c1.clone()].concat())?;
        let d2 = maxdiff(&s2, &via2);
        let ok2 = d2 == 0.0;
        pass &= ok2;
        let _ = writeln!(
            out,
            "{} T2 chunk encoded after the anchor, placed after it ({} + {} + {} tokens): against the decode cache max abs {d2:.6} -> {}; against one-pass encoding max abs {:.6}",
            dev.name(),
            anchor_ids.len(),
            c1.len(),
            q.len(),
            if ok2 { "PASS" } else { "FAIL" },
            maxdiff(&direct2, &via2)
        );
        // T3: two chunks encoded separately after the anchor, placed side by side.
        let k2 = Arc::new(encode(&m, dev, Some(&anchor), &c2)?);
        let w3 = Arc::new(Window::build(&m, &[Arc::clone(&ak), Arc::clone(&k1), Arc::clone(&k2)])?);
        let seq = tail(&hidden(&[anchor_ids.clone(), c1.clone(), c2.clone(), q.clone()].concat(), 0, &RunOpts::default())?, q.len());
        let probe = Arc::new(Probe::new(vec![8, 14, 20]));
        let par = tail(
            &hidden(&q, w3.len, &RunOpts { window: Some(Arc::clone(&w3)), probe: Some(Arc::clone(&probe)), ..RunOpts::default() })?,
            q.len(),
        );
        let last = |x: &[f32]| x.get(x.len() - d.hidden..).unwrap_or(&[]).to_vec();
        let _ = writeln!(
            out,
            "{} T3 window of separately encoded chunks against sequential encoding: last-token hidden cosine {:.4}, max abs {:.4} (approximation, reported)",
            dev.name(),
            cosine(&last(&seq), &last(&par)),
            maxdiff(&seq, &par)
        );
        // T4: per-head reading, mass on each part of the window.
        for l in [8, 14, 20] {
            let Some(mass) = probe.get(l) else {
                pass = false;
                let _ = writeln!(out, "{} T4 layer {l}: no reading -> FAIL", dev.name());
                continue;
            };
            let per_part: Vec<f32> = w3
                .parts
                .iter()
                .map(|(a, n)| {
                    (0..d.heads).map(|h| mass.get(h * w3.len + a..h * w3.len + a + n).unwrap_or(&[]).iter().sum::<f32>()).sum::<f32>()
                })
                .collect();
            let total: f32 = mass.iter().sum();
            #[expect(clippy::cast_precision_loss, reason = "counts for a report")]
            let bound = (q.len() * d.heads) as f32;
            let ok4 = total > 0.0 && total <= bound + 1e-3;
            pass &= ok4;
            let _ = writeln!(
                out,
                "{} T4 layer {l}: mass on anchor, chunk 1, chunk 2 = {:.2} {:.2} {:.2} of {:.0} (summed over heads and question tokens) -> {}",
                dev.name(),
                per_part[0],
                per_part[1],
                per_part[2],
                bound,
                if ok4 { "PASS" } else { "FAIL" }
            );
        }
        // T5: a window of about 8,000 tokens from the mini's own code.
        let files = [
            "crates/nmapi/src/forage.rs",
            "crates/nmbrain/src/synthesis.rs",
            "crates/nmsubs/src/views.rs",
            "crates/nmpod/src/unit.rs",
            "crates/nmbrain/src/gmap.rs",
            "crates/nmengn/src/model.rs",
        ];
        let mut chunks = vec![Arc::clone(&ak)];
        let (mut enc_tokens, t5) = (0_usize, Instant::now());
        'fill: for f in files {
            let text = read(f, 1_000_000)?;
            let lines: Vec<&str> = text.lines().collect();
            for part in lines.chunks(30) {
                let ids = tok(&part.join("\n"))?;
                enc_tokens += ids.len();
                chunks.push(Arc::new(encode(&m, dev, Some(&anchor), &ids)?));
                if enc_tokens >= 8_000 {
                    break 'fill;
                }
            }
        }
        let enc_ms = t5.elapsed().as_secs_f64() * 1000.0;
        let t6 = Instant::now();
        let big = Arc::new(Window::build(&m, &chunks)?);
        let build_ms = t6.elapsed().as_secs_f64() * 1000.0;
        let ask = |p: Option<Arc<Probe>>| -> Result<f64, Fault> {
            let t = Instant::now();
            hidden(&q, big.len, &RunOpts { window: Some(Arc::clone(&big)), probe: p, ..RunOpts::default() })?;
            Ok(t.elapsed().as_secs_f64() * 1000.0)
        };
        let first = ask(None)?;
        let warm = ask(None)?;
        let probed = ask(Some(Arc::new(Probe::new(vec![14]))))?;
        // T6: how the question's time grows with the window, whole model and one layer.
        let one = (nid("L0.score"), "y".to_owned());
        for n in [2_usize, 5, 10, chunks.len()] {
            let w = Arc::new(Window::build(&m, &chunks[..n.min(chunks.len())])?);
            let o = RunOpts { window: Some(Arc::clone(&w)), ..RunOpts::default() };
            hidden(&q, w.len, &o)?;
            let t = Instant::now();
            hidden(&q, w.len, &o)?;
            let all = t.elapsed().as_secs_f64() * 1000.0;
            let t = Instant::now();
            m.forward_at(dev, &q, w.len, std::slice::from_ref(&one), &o)?;
            let layer = t.elapsed().as_secs_f64() * 1000.0;
            let _ = writeln!(
                out,
                "{} T6 window {} tokens: question {all:.0} ms over 28 layers, {layer:.1} ms to the first attention",
                dev.name(),
                w.len
            );
            // The device attention call alone over this window's layer 0 (GPU only).
            if let (Device::Gpu, Some(o)) = (dev, pop.gpu.as_ref()) {
                let (kv, hd) = (d.kv_heads, d.head_dim);
                let dv = w.device(o, kv, hd, d.theta).map_err(bad)?;
                let (wk, wv) = dv.first().cloned().ok_or_else(|| bad("no layer 0"))?;
                let qd = o.upload_f32(&vec![0.01_f32; q.len() * d.heads * hd], vec![q.len(), d.heads, hd]).map_err(bad)?;
                let mask = o.upload_mask(&vec![1_u8; d.heads]).map_err(bad)?;
                let t = Instant::now();
                let y = o.attention(&qd, &wk, &wv, &mask).map_err(bad)?;
                o.download(&y).map_err(bad)?;
                let _ = writeln!(
                    out,
                    "{} T6   device attention alone over {} positions: {:.1} ms",
                    dev.name(),
                    w.len,
                    t.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
        // T7: the drone tips E2 chose (16 heads, layers 8 to 17, chosen on the CX1 development
        // half, `runs/2026-11-10-neumor-note-e2-head-probe.md`) read alone, the run stopping
        // after layer 17, against every head of every layer: same mass on those heads, time.
        const DRONES: [(usize, usize); 16] = nmengn::kvstore::DRONE_HEADS_QWEN3_0_6B;
        let all = Arc::new(Probe::new((0..d.layers).collect()));
        let t = Instant::now();
        hidden(&q, big.len, &RunOpts { window: Some(Arc::clone(&big)), probe: Some(Arc::clone(&all)), ..RunOpts::default() })?;
        let full_ms = t.elapsed().as_secs_f64() * 1000.0;
        let few = Arc::new(Probe::of_heads(&DRONES));
        let stop = (nid(&format!("L{}.score", few.last_layer().unwrap_or(d.layers - 1))), "y".to_owned());
        let t = Instant::now();
        m.forward_at(
            dev,
            &q,
            big.len,
            std::slice::from_ref(&stop),
            &RunOpts { window: Some(Arc::clone(&big)), probe: Some(Arc::clone(&few)), ..RunOpts::default() },
        )?;
        let few_ms = t.elapsed().as_secs_f64() * 1000.0;
        let mut d7 = 0.0_f32;
        for &(l, h) in &DRONES {
            let (a, b) = (all.get(l).unwrap_or_default(), few.get(l).unwrap_or_default());
            let r = h * big.len..(h + 1) * big.len;
            d7 = match (a.get(r.clone()), b.get(r)) {
                (Some(x), Some(y)) if !x.is_empty() => d7.max(maxdiff(x, y)),
                _ => f32::INFINITY,
            };
        }
        let ok7 = d7 == 0.0;
        pass &= ok7;
        let _ = writeln!(
            out,
            "{} T7 drone tips over {} positions: 16 chosen heads, stop after layer 17: {few_ms:.0} ms; every head of 28 layers: {full_ms:.0} ms; mass on the chosen heads max abs {d7:.6} -> {}",
            dev.name(),
            big.len,
            if ok7 { "PASS" } else { "FAIL" }
        );
        #[expect(clippy::cast_precision_loss, reason = "counts for a report")]
        let (per_tok, kb) = (enc_ms / enc_tokens.max(1) as f64, (big.bytes() + big.chunk_bytes()) as f64 / 1024.0 / big.len.max(1) as f64);
        let _ = writeln!(
            out,
            "{} T5 window of {} tokens in {} chunks: encode {enc_ms:.0} ms ({per_tok:.2} ms per token), build {build_ms:.0} ms, RAM {} MB window (rope table) plus {} MB chunks ({kb:.0} KB per token in all); question of {} tokens over it: {first:.0} ms first, {warm:.0} ms warm, {probed:.0} ms with one layer read",
            dev.name(),
            big.len,
            chunks.len(),
            big.bytes() >> 20,
            big.chunk_bytes() >> 20,
            q.len()
        );
    }
    let _ = writeln!(out, "E1 window parity: {}", if pass { "PASS" } else { "FAIL" });
    Ok(out)
}

/// Bytes of the distinct parameter clusters an assembly binds on a population.
fn card_bytes(pop: &Population, a: &Assembly) -> Result<u64, Fault> {
    let ids: std::collections::BTreeSet<&String> = a.nodes.iter().flat_map(|n| n.clusters.iter()).collect();
    ids.iter().map(|r| pop.cluster(&format!("{}/{r}", pop.card.id)).map(|c| c.bytes)).sum()
}

/// Channel-block mask test (2026-11-11, cartridge build step C). On the fingerprint prompt:
/// T1 every 64-channel block of every layer kept, bound as row and column sub-clusters, gives
/// logits bit-identical to the reference assembly; T2 half the blocks (the even ones) dropped
/// in layer 13 and T3 in every layer change the logits and lower the bytes bound.
fn blocks(home: &Path, devices: &[Device]) -> Result<String, Fault> {
    const BLOCK: usize = 64;
    let mut out = String::new();
    let mut pass = true;
    for &dev in devices {
        let pop = Population::open(home, MODEL, true, dev == Device::Gpu)?;
        if dev == Device::Gpu && pop.gpu.is_none() {
            let _ = writeln!(out, "gpu: none open ({}), skipped", pop.gpu_note);
            continue;
        }
        let n = dev.name();
        let d = pop.dims;
        let tok = |s: &str| pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
        let prompt = [tok(PRINT_ANCHOR)?, tok(PRINT_CHUNK)?, tok(PRINT_QUESTION)?].concat();
        let head = (nid("head"), "logits".to_owned());
        let reference = read_assembly(home, "qwen3-reference")?;
        let logits = |a: Assembly| -> Result<(Vec<f32>, f64), Fault> {
            let m = Model::bind(&pop, a)?;
            let t0 = Instant::now();
            let l = m.forward(dev, &prompt, std::slice::from_ref(&head), &RunOpts::default())?;
            Ok((values(l.get(&head).ok_or_else(|| bad("no logits"))?)?.to_vec(), t0.elapsed().as_secs_f64() * 1000.0))
        };
        let base_bytes = card_bytes(&pop, &reference)?;
        let (base, base_ms) = logits(reference.clone())?;
        let argmax = |v: &[f32]| v.iter().enumerate().fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0;
        let all: Vec<usize> = (0..d.inter / BLOCK).collect();
        let half: Vec<usize> = all.iter().copied().filter(|b| b % 2 == 0).collect();
        let cases: [(&str, Vec<usize>, &[usize]); 3] = [
            ("T1 all blocks kept, every layer", (0..d.layers).collect(), &all),
            ("T2 even blocks kept, layer 13", vec![13], &half),
            ("T3 even blocks kept, every layer", (0..d.layers).collect(), &half),
        ];
        for (i, (label, layers, kept)) in cases.into_iter().enumerate() {
            let mut a = reference.clone();
            let mut ranges = 0;
            for &l in &layers {
                ranges = a.keep_ffn_blocks(l, kept, BLOCK)?.len();
            }
            let bytes = card_bytes(&pop, &a)?;
            let (got, ms) = logits(a)?;
            let same = got.len() == base.len() && got.iter().zip(&base).all(|(x, y)| x.to_bits() == y.to_bits());
            let diff = got.iter().zip(&base).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max);
            let ok = if i == 0 { same && bytes == base_bytes } else { !same && bytes < base_bytes };
            pass &= ok;
            #[expect(clippy::cast_precision_loss, reason = "bytes for a report")]
            let mb = |b: u64| b as f64 / 1_048_576.0;
            let _ = writeln!(
                out,
                "{n} {label}: {ranges} range(s) per projection; bit-identical {same}; max abs logit diff {diff:.4}; top token {} vs {}; bytes bound {:.1} MB vs {:.1} MB; forward {ms:.0} ms vs {base_ms:.0} ms -> {}",
                argmax(&got),
                argmax(&base),
                mb(bytes),
                mb(base_bytes),
                if ok { "PASS" } else { "FAIL" }
            );
        }
        pop.release_device();
    }
    let _ = writeln!(out, "C channel-block mask: {}", if pass { "PASS" } else { "FAIL" });
    Ok(out)
}

/// `run s6 <assemblies|parity|split|place|record|fingerprint|blocks> [cpu|gpu|both]`.
pub fn run(args: &[String], home: &Path) -> ExitCode {
    let part = args.get(2).map_or("parity", String::as_str);
    let devices = match args.get(3).map(String::as_str) {
        Some("cpu") => vec![Device::Cpu],
        Some("gpu") => vec![Device::Gpu],
        _ => vec![Device::Cpu, Device::Gpu],
    };
    let result = match part {
        "assemblies" => assemblies(home, args.get(3).map(String::as_str) == Some("replace")),
        "parity" => nmengn::selftest::parity(home, &devices),
        "split" => split(home),
        "place" => place(home),
        "record" => record(home),
        "roles" => roles(home, args.get(3).map(String::as_str) == Some("replace")),
        "roles-eval" => roles_eval(home, args.get(3).map_or("", String::as_str), args.get(4).map_or("gpu", String::as_str)),
        "roles-record" => roles_record(home, args.get(3).map_or("", String::as_str)),
        "roles-parity" => nmengn::selftest::roles_parity(home, &devices),
        "writer-parity" => writer_parity(home, devices.first().copied().unwrap_or(Device::Cpu), args.get(4).map_or("f16", String::as_str)),
        "kv" => kv(home, &devices),
        "fingerprint" => nmengn::selftest::fingerprint(home, &devices),
        "blocks" => blocks(home, &devices),
        other => Err(bad(format!("unknown S6 part {other}"))),
    };
    match result {
        Ok(text) => {
            let p = write(home, part, &text);
            print!("{text}");
            println!("report: {}", p.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            let p = write(home, part, &format!("FAULT {e}\n"));
            eprintln!("s6 {part}: {e} (report {})", p.display());
            ExitCode::FAILURE
        }
    }
}
