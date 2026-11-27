//! Engine self-test against the recorded engine reference in `tests/engine-reference/`:
//! the reference-path fingerprint (compared with `tests/engine-reference/fingerprint.txt`),
//! tokenizer and reference-assembly parity, role-model parity (reranker and embedder) and greedy
//! decode parity. Used by `neuromorph-nano selftest engine` and by the lab tool.
use crate::assembly::{Assembly, nid};
use crate::model::{Model, Population, RunOpts, values};
use crate::placement::Device;
use crate::reference;
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// The population every engine self-test runs on.
pub const MODEL: &str = "qwen3-0.6b";

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("selftest", FaultKind::Execution, m)
}
fn line(out: &mut String, s: impl AsRef<str>) {
    let _ = writeln!(out, "{}", s.as_ref());
}
fn verdict(ok: bool) -> &'static str {
    if ok { "PASS" } else { "MISS" }
}

/// Load an assembly from `designs/assemblies` and pass it through the NTE Integrity Validator
/// (ports and edges) before it is used.
///
/// # Errors
/// A fault when the file is missing, does not parse, or fails validation.
pub fn read_assembly(home: &Path, name: &str) -> Result<Assembly, Fault> {
    let a = Assembly::load(&home.join("designs/assemblies"), name)?;
    let g = nmnte::Graph::from_value(&a.to_value(""))?;
    nmnte::integrity::validate(&g, None)?;
    Ok(a)
}

/// The role engine reference: `tests/engine-reference/role-models/` (2026-11-10).
pub fn role_ref(home: &Path) -> PathBuf {
    home.join("tests/engine-reference/role-models")
}
/// P(yes) per pair (set path) and vectors per text, for one device.
pub fn role_outputs(home: &Path, v: &Value, dev: Device) -> Result<(Vec<f32>, Vec<Vec<f32>>), Fault> {
    let open = |name: &str, pop_id: &str| -> Result<Model, Fault> {
        let pop = Population::open(home, pop_id, true, dev == Device::Gpu)?;
        let a = Assembly::load(&home.join("designs/assemblies"), name)?;
        Model::bind(&pop, a)
    };
    let s = |x: &Value, k: &str| x.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default();
    let mut ps = Vec::new();
    let r = open("qwen3-reranker", "qwen3-reranker-0.6b")?;
    for it in v.get("rerank").and_then(Value::array).map_err(|e| bad(e.to_string()))? {
        ps.extend(crate::roles::card_rerank_set(&r, dev, &s(it, "instruction"), &s(it, "query"), &[s(it, "document")])?);
    }
    drop(r);
    let e = open("qwen3-embedding", "qwen3-embedding-0.6b")?;
    let mut vs = Vec::new();
    for it in v.get("embed").and_then(Value::array).map_err(|e| bad(e.to_string()))? {
        vs.push(crate::roles::card_embed_on(&e, dev, &s(it, "text"))?);
    }
    Ok((ps, vs))
}
/// Role parity against the recording: worst P(yes) difference at most 0.003 and worst cosine
/// of each vector with its recording at least 0.999, per device.
pub fn roles_parity(home: &Path, devices: &[Device]) -> Result<String, Fault> {
    let dir = role_ref(home);
    let v = nmjson::parse(&std::fs::read(dir.join("inputs.json")).map_err(|e| bad(format!("role inputs: {e}")))?)
        .map_err(|e| bad(e.to_string()))?;
    let rec = nmjson::parse(&std::fs::read(dir.join("outputs.json")).map_err(|e| bad(format!("role outputs: {e}")))?)
        .map_err(|e| bad(e.to_string()))?;
    let num = |x: &Value| if let Value::Number(n) = x { n.parse::<f32>().unwrap_or(f32::NAN) } else { f32::NAN };
    let rp: Vec<f32> = rec.get("p_yes").and_then(Value::array).map_err(|e| bad(e.to_string()))?.iter().map(num).collect();
    let rv: Vec<Vec<f32>> = rec
        .get("vectors")
        .and_then(Value::array)
        .map_err(|e| bad(e.to_string()))?
        .iter()
        .map(|a| a.array().map(|xs| xs.iter().map(num).collect()).unwrap_or_default())
        .collect();
    let mut out = String::new();
    let mut ok = true;
    for &dev in devices {
        let (ps, vs) = role_outputs(home, &v, dev)?;
        let dp = ps.iter().zip(&rp).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        let cs = vs.iter().zip(&rv).map(|(a, b)| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()).fold(1.0_f32, f32::min);
        let pass = dp <= 0.003 && cs >= 0.999 && ps.len() == rp.len() && vs.len() == rv.len();
        ok &= pass;
        let _ = writeln!(
            out,
            "{}: reranker {} pairs, worst P(yes) difference {dp:.4}; embedding {} texts, worst cosine with the recording {cs:.5} -> {}",
            dev.name(),
            ps.len(),
            vs.len(),
            if pass { "PASS" } else { "FAIL" }
        );
    }
    if !ok {
        return Err(bad(format!("role parity failed:\n{out}")));
    }
    Ok(out)
}

/// Gates 1 to 4.
pub fn parity(home: &Path, devices: &[Device]) -> Result<String, Fault> {
    let mut out = String::new();
    let t0 = Instant::now();
    let pop = Population::open(home, MODEL, true, devices.contains(&Device::Gpu))?;
    line(
        &mut out,
        format!(
            "population {} sha256 {} verified and opened in {} ms; GPU: {}",
            pop.card.id,
            pop.card.sha_main,
            t0.elapsed().as_millis(),
            pop.gpu_note
        ),
    );
    // Gate 1: decomposition.
    let groups: std::collections::BTreeSet<&str> = pop.clusters.iter().map(|c| c.group.as_str()).collect();
    let heads = groups.iter().filter(|g| g.contains(".head")).count();
    let ffn = groups.iter().filter(|g| g.ends_with(".ffn")).count();
    let kv = groups.iter().filter(|g| g.contains(".kv")).count();
    let embed = pop.clusters.iter().any(|c| c.group == "embed");
    let fin = pop.clusters.iter().any(|c| c.group == "final");
    let cov = nmnpcs::decompose::coverage(&pop.tensors, &pop.clusters);
    let g1 = heads + ffn == 476 && embed && fin && cov.is_ok();
    line(
        &mut out,
        format!(
            "gate 1 decomposition: {} tensors, {} clusters; head groups {heads} + ffn groups {ffn} = {} (want 476); kv groups {kv}; embed {embed}, final_norm {fin}; coverage {:?} -> {}",
            pop.tensors.len(),
            pop.clusters.len(),
            heads + ffn,
            cov.as_ref().map(|c| format!("{} of {} bytes", c.1, c.2)).map_err(ToString::to_string),
            verdict(g1)
        ),
    );
    // Gate 2: one Arc per cluster.
    let c = pop.cluster(&format!("{MODEL}/model.norm.weight/whole"))?;
    let reads0 = pop.prov.reads();
    let (a, b) = (pop.prov.acquire(&c)?, pop.prov.acquire(&c)?);
    let same = Arc::ptr_eq(&a, &b) && a.refs() == 2 && pop.prov.reads() == reads0 + 1;
    let counted = pop.prov.residency().iter().filter(|r| r.0 == c.id).count() == 1;
    drop((a, b));
    pop.prov.release(&c.id)?;
    pop.prov.release(&c.id)?;
    let refused_with_refs = {
        let x = pop.prov.acquire(&c)?;
        let r = pop.prov.retire(&c.id).is_err();
        drop(x);
        pop.prov.release(&c.id)?;
        r
    };
    pop.prov.retire(&c.id)?;
    let again = pop.prov.acquire(&c)?;
    let reread = pop.prov.reads() == reads0 + 2;
    drop(again);
    pop.prov.release(&c.id)?;
    pop.prov.retire(&c.id)?;
    let g2 = same && counted && refused_with_refs && reread;
    line(
        &mut out,
        format!(
            "gate 2 one Arc per cluster: two acquires share one Arc and one read {same}; residency counts it once {counted}; retire refused with references {refused_with_refs}; after retire the next acquire reads again {reread} -> {}",
            verdict(g2)
        ),
    );
    // Gate 3: tokenizer.
    let (ok, total, first) = reference::tokenizer(home, &pop.tokenizer)?;
    line(
        &mut out,
        format!(
            "gate 3 tokenizer parity: {ok} of {total} texts exact (ids and decode){} -> {}",
            first.map(|f| format!("; first miss {f}")).unwrap_or_default(),
            verdict(ok == total)
        ),
    );
    // Gate 4: reference assembly parity.
    let reference = Arc::new(Model::bind(&pop, read_assembly(home, "qwen3-reference")?)?);
    line(
        &mut out,
        format!(
            "reference assembly bound: {} nodes, {} clusters live, RAM {} MB",
            reference.bound.len(),
            pop.prov.live(),
            pop.prov.totals().0 >> 20
        ),
    );
    let mut g4 = BTreeMap::new();
    for &dev in devices {
        let mut all = true;
        for (layer, off, file) in [
            (0, vec![], "block0-on.f16"),
            (13, vec![], "block13-on.f16"),
            (13, vec![0, 5], "block13-off.f16"),
            (27, vec![], "block27-on.f16"),
        ] {
            let c = reference::block(home, &reference, dev, layer, &off, file)?;
            all &= c.pass;
            line(&mut out, format!("  {}: max abs {:.6} (atol {}) {:.0} ms -> {}", c.name, c.max_abs, c.atol, c.ms, verdict(c.pass)));
        }
        for i in 0..2 {
            let c = reference::prompt(home, &reference, dev, i)?;
            all &= c.pass;
            line(
                &mut out,
                format!("  {}: max abs {:.6} (atol {}), {} {:.0} ms -> {}", c.name, c.max_abs, c.atol, c.extra, c.ms, verdict(c.pass)),
            );
        }
        g4.insert(dev.name(), all);
        line(&mut out, format!("gate 4 reference parity on {}: {}", dev.name(), verdict(all)));
    }
    let (ram, gpu) = pop.prov.totals();
    line(
        &mut out,
        format!(
            "residency after the run: RAM {} MB in {} tensor reads, GPU {} MB recorded, {} device bytes",
            ram >> 20,
            pop.prov.reads(),
            gpu >> 20,
            pop.device_bytes()
        ),
    );
    line(&mut out, format!("wall {} ms", t0.elapsed().as_millis()));
    Ok(out)
}

/// Fixed input text of the fingerprint (written for it; not from any test); keep byte-identical,
/// or re-record `tests/engine-reference/fingerprint.txt` on purpose.
pub const PRINT_ANCHOR: &str =
    "Repository neuromorph-nano: a Rust workspace (nmsubs substrate, nmpod tentacles, nmbrain Core Brain, nmengn engine, nmapi tools).\n";
pub const PRINT_CHUNK: &str = "crates/x/src/ring.rs\npub struct Ring { items: Vec<u32>, head: usize }\nimpl Ring {\n    pub fn push(&mut self, x: u32) { let n = self.items.len(); self.items[self.head % n] = x; self.head += 1; }\n}\n";
pub const PRINT_QUESTION: &str = "\nQuestion: where does the ring write a new item";

/// Reference-path fingerprint (2026-11-10, step P0): SHA-256 of the bits of every output the
/// reference assembly gives on fixed inputs (encoded keys and values, full and shortened; drone
/// and all-head attention mass over a window; hidden state over a window; logits; heads-off
/// logits; decode-cache hidden state; greedy tokens; the base-model embedding and rerank
/// score). Run before and after an engine change: equal lines mean bit-identical results.
pub fn fingerprint(home: &Path, devices: &[Device]) -> Result<String, Fault> {
    use crate::kvstore::{DRONE_HEADS_QWEN3_0_6B as DRONES, Probe, Window, encode, encode_through};
    // 2026-11-25: the fingerprint is the bare reference path's; the served per-role job filters
    // (on by default since 2026-11-24) changed 18 of its 20 lines and are held off here, as in the
    // socket check (with them held off it equals the recording on the processor and the card).
    let _bare = crate::model::JobFiltersHeldOff::new();
    let mut out = String::new();
    let f32s = |v: &[f32]| nmhash::sha256_bytes(&v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect::<Vec<u8>>());
    let u16s = |v: &[Vec<u16>]| nmhash::sha256_bytes(&v.iter().flatten().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    for &dev in devices {
        let pop = Population::open(home, MODEL, true, dev == Device::Gpu)?;
        if dev == Device::Gpu && pop.gpu.is_none() {
            let _ = writeln!(out, "gpu: none open ({}), skipped", pop.gpu_note);
            continue;
        }
        let m = Model::bind(&pop, Assembly::load(&home.join("designs/assemblies"), "qwen3-reference")?)?;
        let d = pop.dims;
        let tok = |s: &str| pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
        let (a_ids, c_ids, q) = (tok(PRINT_ANCHOR)?, tok(PRINT_CHUNK)?, tok(PRINT_QUESTION)?);
        let n = dev.name();
        // Encoded keys and values: full and through the drone heads' last layer.
        let last = Probe::of_heads(&DRONES).last_layer().unwrap_or(0);
        let ak = Arc::new(encode(&m, dev, None, &a_ids)?);
        let aw = Arc::new(Window::build(&m, &[Arc::clone(&ak)])?);
        let ck = Arc::new(encode(&m, dev, Some(&aw), &c_ids)?);
        let _ = writeln!(out, "{n} encode anchor k {} v {}", u16s(&ak.k_pre), u16s(&ak.v));
        let _ = writeln!(out, "{n} encode chunk k {} v {}", u16s(&ck.k_pre), u16s(&ck.v));
        let sk = Arc::new(encode_through(&m, dev, None, &a_ids, last)?);
        let sw = Arc::new(Window::build(&m, &[Arc::clone(&sk)])?);
        let sc = Arc::new(encode_through(&m, dev, Some(&sw), &c_ids, last)?);
        let _ = writeln!(
            out,
            "{n} encode_through {last} anchor k {} v {}; chunk k {} v {}",
            u16s(&sk.k_pre),
            u16s(&sk.v),
            u16s(&sc.k_pre),
            u16s(&sc.v)
        );
        // Drone mass over the short window, stopping after the last drone layer.
        let w = Arc::new(Window::build(&m, &[Arc::clone(&sk), Arc::clone(&sc)])?);
        let probe = Arc::new(Probe::of_heads(&DRONES));
        let stop = (nid(&format!("L{last}.score")), "y".to_owned());
        m.forward_at(
            dev,
            &q,
            w.len,
            std::slice::from_ref(&stop),
            &RunOpts { window: Some(Arc::clone(&w)), probe: Some(Arc::clone(&probe)), ..RunOpts::default() },
        )?;
        let mut mass = Vec::new();
        for l in probe.layers.clone() {
            mass.extend(probe.get(l).ok_or_else(|| bad(format!("layer {l} not read")))?);
        }
        let _ = writeln!(out, "{n} drone mass ({} values) {}", mass.len(), f32s(&mass));
        // Every head of every layer over the full window, and the hidden state.
        let wf = Arc::new(Window::build(&m, &[Arc::clone(&ak), Arc::clone(&ck)])?);
        let all = Arc::new(Probe::new((0..d.layers).collect()));
        let fin = (nid("final_norm"), "y".to_owned());
        let h = m.forward_at(
            dev,
            &q,
            wf.len,
            std::slice::from_ref(&fin),
            &RunOpts { window: Some(Arc::clone(&wf)), probe: Some(Arc::clone(&all)), ..RunOpts::default() },
        )?;
        let mut mass = Vec::new();
        for l in 0..d.layers {
            mass.extend(all.get(l).ok_or_else(|| bad(format!("layer {l} not read")))?);
        }
        let _ = writeln!(
            out,
            "{n} all-head mass ({} values) {}; hidden over the window {}",
            mass.len(),
            f32s(&mass),
            f32s(values(h.get(&fin).ok_or_else(|| bad("no hidden"))?)?)
        );
        // Logits, heads off, decode cache, greedy.
        let prompt = [a_ids.clone(), c_ids.clone(), q.clone()].concat();
        let head = (nid("head"), "logits".to_owned());
        let l = m.forward(dev, &prompt, std::slice::from_ref(&head), &RunOpts::default())?;
        let _ = writeln!(out, "{n} logits {}", f32s(values(l.get(&head).ok_or_else(|| bad("no logits"))?)?));
        let mut off = RunOpts::default();
        off.heads_off.insert(13, vec![0, 5, 9]);
        let l = m.forward(dev, &prompt, std::slice::from_ref(&head), &off)?;
        let _ = writeln!(out, "{n} logits heads off {}", f32s(values(l.get(&head).ok_or_else(|| bad("no logits"))?)?));
        let o = RunOpts { cache: Some(Arc::new(crate::Cache::default())), ..RunOpts::default() };
        m.forward_at(dev, &[a_ids.clone(), c_ids.clone()].concat(), 0, std::slice::from_ref(&fin), &o)?;
        let h = m.forward_at(dev, &q, a_ids.len() + c_ids.len(), std::slice::from_ref(&fin), &o)?;
        let _ = writeln!(out, "{n} decode-cache hidden {}", f32s(values(h.get(&fin).ok_or_else(|| bad("no hidden"))?)?));
        let g = crate::decode::greedy(&m, dev, &prompt, 8, &[])?;
        let _ = writeln!(out, "{n} greedy {:?}", g.tokens);
        let e = crate::roles::embed_on(&m, dev, PRINT_CHUNK)?;
        let r = crate::roles::rerank_on(&m, dev, "where does the ring write", PRINT_CHUNK)?;
        let _ = writeln!(out, "{n} embed {}; rerank {:08x}", f32s(&e), r.to_bits());
        pop.release_device();
    }
    Ok(out)
}

pub fn resources(m: &Model) -> String {
    let peak = nmffi::dynlib::process_memory().map_or(0, |(p, _)| p >> 20);
    let gpu = m.pop.gpu.as_ref().and_then(|g| g.gpu.memory().ok()).map_or(0, |(f, t)| (t - f) >> 20);
    format!("peak RAM {peak} MB (process working set), GPU in use {gpu} MB (device total minus free)")
}

/// Gate 1: greedy decode of golden prompts 0 and 1 on CPU and GPU.
pub fn decode(home: &Path, devices: &[Device]) -> Result<String, Fault> {
    let mut out = String::new();
    let pop = Population::open(home, MODEL, true, devices.contains(&Device::Gpu))?;
    let m = Model::bind(&pop, read_assembly(home, "qwen3-reference")?)?;
    let inputs = nmjson::parse(&std::fs::read(crate::reference::root(home).join("inputs.json")).map_err(|e| bad(e.to_string()))?)
        .map_err(|e| bad(e.to_string()))?;
    let mut all = true;
    for &dev in devices {
        for i in 0..2 {
            let text = inputs
                .get("prompts")
                .and_then(Value::array)
                .ok()
                .and_then(|a| a.get(i))
                .and_then(|v| v.string().ok())
                .ok_or_else(|| bad("prompt"))?
                .to_owned();
            let ids = m.pop.tokenizer.encode(&text).map_err(|e| bad(e.to_string()))?;
            let want: Vec<i32> = std::fs::read(crate::reference::root(home).join(format!("runs/A/full-forward/prompt{i}-greedy.i32")))
                .map_err(|e| bad(e.to_string()))?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c))
                .collect();
            let g = crate::decode::greedy(&m, dev, &ids, want.len(), &[])?;
            let first_diff = g.tokens.iter().zip(&want).position(|(a, b)| a != b);
            let ok = g.tokens == want;
            all &= ok;
            #[expect(clippy::cast_precision_loss, reason = "step counts are small")]
            let step = if g.step_ms.is_empty() { 0.0 } else { g.step_ms.iter().sum::<f64>() / g.step_ms.len() as f64 };
            line(
                &mut out,
                format!(
                    "  {} prompt{i}: {} of {} tokens equal{}; prefill {:.0} ms, {step:.1} ms per step -> {}",
                    dev.name(),
                    g.tokens.iter().zip(&want).filter(|(a, b)| a == b).count(),
                    want.len(),
                    first_diff.map(|d| format!(", first difference at step {d}")).unwrap_or_default(),
                    g.prefill_ms,
                    if ok { "PASS" } else { "MISS" }
                ),
            );
        }
    }
    line(&mut out, format!("gate 1 decode parity: {}", if all { "PASS" } else { "MISS" }));
    line(&mut out, resources(&m));
    Ok(out)
}

/// The recorded fingerprint lines: `tests/engine-reference/fingerprint.txt`.
#[must_use]
pub fn fingerprint_file(home: &Path) -> PathBuf {
    home.join("tests/engine-reference/fingerprint.txt")
}

/// Compare fingerprint lines with the recording, device by device: every line of a device that
/// ran must equal the recorded line in the same place. A device with no recorded lines fails.
fn compare_fingerprint(home: &Path, got: &str, devices: &[Device]) -> Result<String, String> {
    let path = fingerprint_file(home);
    let want = std::fs::read_to_string(&path)
        .map_err(|e| format!("{}: {e} (record it with `neuromorph-nano selftest engine --record`)", path.display()))?;
    let mut out = String::new();
    let mut bad_lines = Vec::new();
    for dev in devices {
        let prefix = format!("{} ", dev.name());
        let w: Vec<&str> = want.lines().filter(|l| l.starts_with(&prefix)).collect();
        let g: Vec<&str> = got.lines().filter(|l| l.starts_with(&prefix)).collect();
        if w.is_empty() {
            bad_lines.push(format!("{}: no recorded lines", dev.name()));
            continue;
        }
        let same = w.iter().zip(&g).filter(|(a, b)| a == b).count();
        if same == w.len() && g.len() == w.len() {
            line(&mut out, format!("fingerprint {}: {same} of {} lines equal the recording -> PASS", dev.name(), w.len()));
        } else {
            for (i, (a, b)) in w.iter().zip(&g).enumerate().filter(|(_, (a, b))| a != b) {
                bad_lines.push(format!("{} line {}: recorded `{a}`, now `{b}`", dev.name(), i + 1));
            }
            if g.len() != w.len() {
                bad_lines.push(format!("{}: {} lines recorded, {} now", dev.name(), w.len(), g.len()));
            }
        }
    }
    if bad_lines.is_empty() {
        Ok(out)
    } else {
        Err(format!(
            "{out}fingerprint differs from {}:
  {}",
            path.display(),
            bad_lines.join(
                "
  "
            )
        ))
    }
}

/// The devices to test: the processor always; the graphics card when asked and one opens.
fn devices_for(home: &Path, want: &str) -> Result<(Vec<Device>, String), Fault> {
    let mut devices = match want {
        "cpu" => vec![Device::Cpu],
        "gpu" => vec![Device::Gpu],
        _ => vec![Device::Cpu, Device::Gpu],
    };
    let mut note = String::new();
    if devices.contains(&Device::Gpu) {
        let pop = Population::open(home, MODEL, false, true)?;
        if pop.gpu.is_none() {
            devices.retain(|d| *d != Device::Gpu);
            note = format!(
                "graphics card: none open ({}); processor only
",
                pop.gpu_note
            );
        } else {
            pop.release_device();
        }
    }
    if devices.is_empty() {
        return Err(bad("no device to test: no graphics card opened and the processor was not asked for"));
    }
    Ok((devices, note))
}

/// `selftest engine [cpu|gpu|both] [--record]`: the fingerprint against its recording, then
/// tokenizer and reference parity, role-model parity and greedy decode parity. With `record`
/// the fingerprint lines are written to `tests/engine-reference/fingerprint.txt` instead of
/// compared. Returns the report and whether every part passed.
///
/// # Errors
/// A fault when a model, assembly or reference file cannot be read.
pub fn engine(home: &Path, want: &str, record: bool) -> Result<(String, bool), Fault> {
    let t0 = Instant::now();
    let (devices, note) = devices_for(home, want)?;
    let mut out = note;
    let mut pass = true;
    let fp = fingerprint(home, &devices)?;
    out.push_str(&fp);
    if record {
        let path = fingerprint_file(home);
        let lines: String = fp
            .lines()
            .filter(|l| devices.iter().any(|d| l.starts_with(&format!("{} ", d.name()))))
            .map(|l| {
                format!(
                    "{l}
"
                )
            })
            .collect();
        std::fs::write(&path, lines).map_err(|e| bad(format!("{}: {e}", path.display())))?;
        line(&mut out, format!("fingerprint recorded to {}", path.display()));
    } else {
        match compare_fingerprint(home, &fp, &devices) {
            Ok(t) => out.push_str(&t),
            Err(t) => {
                pass = false;
                line(&mut out, t);
            }
        }
    }
    let mut part = |name: &str, r: Result<String, Fault>, out: &mut String| match r {
        Ok(t) => {
            let ok = !t.contains("MISS") && !t.contains("FAIL");
            pass &= ok;
            out.push_str(&t);
            line(out, format!("{name}: {}", if ok { "PASS" } else { "FAIL" }));
        }
        Err(e) => {
            pass = false;
            line(out, format!("{name}: FAIL {e}"));
        }
    };
    part("tokenizer and reference parity", parity(home, &devices), &mut out);
    part("role-model parity", roles_parity(home, &devices), &mut out);
    part("decode parity", decode(home, &devices), &mut out);
    line(&mut out, format!("selftest engine: {} in {} ms", if pass { "PASS" } else { "FAIL" }, t0.elapsed().as_millis()));
    Ok((out, pass))
}
