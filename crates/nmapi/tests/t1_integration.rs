//! Phase D4 card job (lane T1, 2026-11-18; tentacle master plan 3.3 step 5 as decided by Phase A):
//! the tentacle's integration read on the Mini's TC1 dump (development half, the 20 questions lane
//! A3 read). Per question and window, the D3 shortlist (`scratch/lane-t1/d3/shortlists.json`, from
//! the wording and meaning evidence) is read on the card through the relevance layers behind the
//! anchor, pieces in relation order (the leader first, then its related pieces breadth first over
//! the specific reference graph, then the rest by path and line), scored at the window tentacle's
//! drone heads by the per-piece log-sum-exp of the raw scores (posed and content-free), two forms:
//! - `alone`: each piece encoded alone behind the anchor, then joined (the exact read's form);
//! - `joint`: the joined text encoded from layer 0, each piece behind the anchor and the pieces
//!   before it (the plain joint read, in-context read form).
//!
//! Writes per window the rows of both forms and the times (piece indices only, no text) to
//! `scratch/lane-t1/d4/q*.json`. Env: `D4_FROM`, `D4_N`. Run under the card lock:
//! `cargo test --release -p nmapi --test t1_integration -- --ignored --nocapture --test-threads 1`.
//! `d4_timing` reads 1 to 4 groups of 200 to 1,500 tokens and reports card time per call.
use nmapi::{scan_dump, tentacle_pass};
use nmengn::kvstore::{self, ChunkKv, Window};
use nmengn::placement::Device;
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

type R<T> = Result<T, String>;

const MINI_ANCHOR: &str = "Repository neuromorph-mini: a Rust workspace of 18 crates (nmsubs substrate, nmpod tentacles, nmbrain Core Brain, nmengn engine, nmapi tools).\n";
const DUMP: &str = "1794288995065";

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// The development half of the Mini's CX1 questions (as lane A3's test reads it).
fn dev_half(mini: &Path) -> R<Vec<(String, String)>> {
    const TYPES: [&str; 5] = ["behaviour", "name", "value", "explain", "relation"];
    let mut out = Vec::new();
    for area in ["context", "engine", "tools"] {
        let p = mini.join(format!("tests/capability-gates/cx1-context/questions-{area}.json"));
        let v = nmjson::parse(&std::fs::read(&p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let mut qs: Vec<(usize, String, String)> = Vec::new();
        for x in v.get("questions").and_then(|y| y.array()).map_err(|e| e.to_string())? {
            let s = |k: &str| x.get(k).and_then(|y| y.string()).map(str::to_owned).map_err(|e| e.to_string());
            let t = s("type")?;
            qs.push((TYPES.iter().position(|y| *y == t).ok_or("type")?, s("id")?, s("q")?));
        }
        qs.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
        out.extend(qs.into_iter().enumerate().filter(|(pos, _)| pos % 2 == 0).map(|(_, (_, id, q))| (id, q)));
    }
    Ok(out)
}

fn floats(v: &[f32]) -> String {
    let parts: Vec<String> = v.iter().map(|x| if x.is_finite() { format!("{x}") } else { "null".into() }).collect();
    format!("[{}]", parts.join(","))
}

/// Relation order: the leader first, then pieces related to those already placed (breadth first
/// over the specific graph), then the rest by path and first line.
fn relation_order(sub: &nmsubs::Substrate, list: &[usize]) -> Vec<usize> {
    let g = sub.refs();
    let want: BTreeSet<usize> = list.iter().copied().collect();
    let mut out: Vec<usize> = list.first().copied().into_iter().collect();
    let mut i = 0;
    while i < out.len() {
        let c = out.get(i).copied().unwrap_or(0);
        for e in g.specific.edges_of(c) {
            let t = e.to as usize;
            if want.contains(&t) && !out.contains(&t) {
                out.push(t);
            }
        }
        i += 1;
    }
    let mut rest: Vec<usize> = list.iter().copied().filter(|c| !out.contains(c)).collect();
    let key = |c: &usize| sub.chunks.get(*c).map(|ch| (sub.files.get(ch.file).map(|f| f.path.clone()).unwrap_or_default(), ch.start)).unwrap_or_default();
    rest.sort_by_key(key);
    out.extend(rest);
    out
}

struct Reader {
    model: Arc<nmengn::Model>,
    dev: Device,
    last: usize,
    stop: Vec<(String, String)>,
    ak: Arc<ChunkKv>,
    aw: Arc<Window>,
}
impl Reader {
    fn open(home: &Path) -> R<Self> {
        let model = nmapi::neural::model(home).map_err(|e| e.message)?;
        let dev = if model.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
        let last = tentacle_pass::drones(home).last_layer();
        let stop = kvstore::score_targets(&model, last).map_err(|e| e.message)?;
        let ids = model.pop.tokenizer.encode(MINI_ANCHOR).map_err(|e| e.to_string())?;
        let ak = Arc::new(kvstore::encode_through(&model, dev, None, &ids, last).map_err(|e| e.message)?);
        let aw = Arc::new(Window::build(&model, &[Arc::clone(&ak)]).map_err(|e| e.message)?);
        Ok(Self { model, dev, last, stop, ak, aw })
    }
    /// The window of `texts` behind the anchor: each alone (`joint` false) or each behind the
    /// pieces before it (`joint` true). Returns the window and the encode milliseconds.
    fn window(&self, texts: &[String], joint: bool) -> R<(Arc<Window>, f64, usize)> {
        let t = Instant::now();
        let mut parts = vec![Arc::clone(&self.ak)];
        let mut toks = 0;
        for s in texts {
            let ids = self.model.pop.tokenizer.encode(s).map_err(|e| e.to_string())?;
            toks += ids.len();
            let prefix = if joint { Arc::new(Window::build(&self.model, &parts).map_err(|e| e.message)?) } else { Arc::clone(&self.aw) };
            let gate = (self.dev == Device::Gpu).then(|| nmengn::placement::gpu_gate().enter("t1 d4 encode", true));
            let kv = kvstore::encode_through(&self.model, self.dev, Some(&prefix), &ids, self.last).map_err(|e| e.message)?;
            drop(gate);
            parts.push(Arc::new(kv));
        }
        let w = Arc::new(Window::build(&self.model, &parts).map_err(|e| e.message)?);
        Ok((w, t.elapsed().as_secs_f64() * 1000.0, toks))
    }
    /// Posed and content-free rows (heads x parts) of `w`, and the read milliseconds.
    fn read(&self, w: &Arc<Window>, heads: &[(usize, usize)], question: &str) -> R<(Vec<f32>, Vec<f32>, f64)> {
        let t = Instant::now();
        let gate = (self.dev == Device::Gpu).then(|| nmengn::placement::gpu_gate().enter("t1 d4 read", true));
        let (_, posed) = tentacle_pass::drone_read_logits(&self.model, self.dev, w, &self.stop, heads, &format!("\nQuestion: {question}")).map_err(|e| e.message)?;
        let (_, null) = tentacle_pass::drone_read_logits(&self.model, self.dev, w, &self.stop, heads, "\nQuestion: N/A").map_err(|e| e.message)?;
        drop(gate);
        Ok((posed, null, t.elapsed().as_secs_f64() * 1000.0))
    }
}

fn doc(sub: &nmsubs::Substrate, c: usize) -> String {
    sub.chunks.get(c).and_then(|ch| sub.files.get(ch.file).map(|f| nmsubs::dense::doc_text(&f.path, &ch.text))).unwrap_or_default()
}

#[test]
#[ignore = "card job: needs the reader and the Mini's TC1 dump; run under the card lock"]
fn d4_integration() {
    if let Err(e) = d4() {
        panic!("d4: {e}");
    }
}

fn d4() -> R<()> {
    let home = nmtype::nano_root();
    let mini = home.join("..").join("neuromorph-mini");
    let dir = mini.join("scratch").join("tc1-dump").join(DUMP);
    let out = home.join("scratch/lane-t1/d4");
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let items = dev_half(&mini)?;
    let sub = scan_dump::load_substrate(&dir).map_err(|e| e.message)?;
    let sl = nmjson::parse(&std::fs::read(home.join("scratch/lane-t1/d3/shortlists.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let r = Reader::open(&home)?;
    let sets = tentacle_pass::drones(&home);
    let (from, n) = (env_usize("D4_FROM", 0), env_usize("D4_N", 2));
    let num = |v: &nmjson::Value| v.integer().map_err(|e| e.to_string()).and_then(|x| usize::try_from(x).map_err(|e| e.to_string()));
    for q in sl.array().map_err(|e| e.to_string())?.iter().skip(from).take(n) {
        let idx = num(q.get("index").map_err(|e| e.to_string())?)?;
        let (_, text) = items.get(idx).ok_or("question")?;
        let mut js = format!("{{\"index\":{idx},\"windows\":[");
        let mut line = format!("q{idx:02}:");
        for (wi, w) in q.get("windows").and_then(|x| x.array()).map_err(|e| e.to_string())?.iter().enumerate() {
            let tentacle = num(w.get("tentacle").map_err(|e| e.to_string())?)?;
            let list: Vec<usize> = w.get("shortlist").and_then(|x| x.array()).map_err(|e| e.to_string())?.iter().filter_map(|x| num(x).ok()).collect();
            if wi > 0 {
                js.push(',');
            }
            if list.is_empty() {
                let _ = write!(js, "{{\"tentacle\":{tentacle},\"order\":[]}}");
                continue;
            }
            let order = relation_order(&sub, &list);
            let texts: Vec<String> = order.iter().map(|c| doc(&sub, *c)).collect();
            let heads = sets.for_tentacle(nmsubs::lens::NAMES.get(tentacle).copied().unwrap_or("lens"));
            let _ = write!(js, "{{\"tentacle\":{tentacle},\"order\":{order:?}");
            for (name, joint) in [("alone", false), ("joint", true)] {
                let (win, enc_ms, toks) = r.window(&texts, joint)?;
                let (posed, null, read_ms) = r.read(&win, heads, text)?;
                let _ = write!(js, ",\"{name}_posed\":{},\"{name}_null\":{},\"{name}_enc_ms\":{enc_ms:.1},\"{name}_read_ms\":{read_ms:.1},\"tokens\":{toks}", floats(&posed), floats(&null));
                let _ = write!(line, " w{wi} {name} {toks} tok {enc_ms:.0}+{read_ms:.0} ms;");
            }
            js.push('}');
        }
        js.push_str("]}\n");
        std::fs::write(out.join(format!("q{idx:02}.json")), js).map_err(|e| e.to_string())?;
        eprintln!("{line}");
    }
    Ok(())
}

#[test]
#[ignore = "card timing: run under the card lock"]
fn d4_timing() {
    let home = nmtype::nano_root();
    let r = Reader::open(&home).expect("reader");
    let heads = tentacle_pass::drones(&home).heads;
    let src = std::fs::read_to_string(home.join("crates/nmapi/src/tentacle_pass.rs")).expect("text");
    let ids = r.model.pop.tokenizer.encode(&src).expect("tokens");
    for size in [200_usize, 500, 1000, 1500] {
        for groups in 1..=4_usize {
            let t = Instant::now();
            for g in 0..groups {
                let piece: Vec<i32> = ids.iter().copied().skip(g * 1500).take(size).collect();
                let text = r.model.pop.tokenizer.decode(&piece);
                let (w, _, _) = r.window(&[text], true).expect("window");
                let _ = r.read(&w, &heads, "where is the window composed").expect("read");
            }
            println!("D4 timing: {groups} groups of {size} tokens, joint read: {:.0} ms on {}", t.elapsed().as_secs_f64() * 1000.0, r.dev.name());
        }
    }
}

/// D4 with lane S's packed reads (`Integrator::read_packed`): per question the windows' shortlists
/// read in packed runs of up to `D4P_PER` tentacles (posed and content-free groups each), scores per
/// piece in the order read, to `scratch/lane-t1/d4p/q*.json`; and card time for 1 to 4 groups.
#[test]
#[ignore = "card job: run under the card lock"]
fn d4_packed() {
    if let Err(e) = d4p() {
        panic!("d4p: {e}");
    }
}

fn d4p() -> R<()> {
    use nmapi::tentacle_read::{Integrator, relation_order as order_of};
    // As `serve` sets it for the served models (the packed path's one-launch attention).
    nmengn::gpu::SEGMENT_ATTENTION.store(true, std::sync::atomic::Ordering::Relaxed);
    let home = nmtype::nano_root();
    let mini = home.join("..").join("neuromorph-mini");
    let dir = mini.join("scratch").join("tc1-dump").join(DUMP);
    let out = home.join("scratch/lane-t1/d4p");
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let items = dev_half(&mini)?;
    let sub = scan_dump::load_substrate(&dir).map_err(|e| e.message)?;
    let sl = nmjson::parse(&std::fs::read(home.join("scratch/lane-t1/d3/shortlists.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let model = nmapi::neural::model(&home).map_err(|e| e.message)?;
    let dev = if model.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let sets = tentacle_pass::drones(&home);
    let r = Integrator::new(Arc::clone(&model), dev, MINI_ANCHOR, sets.last_layer()).map_err(|e| e.message)?;
    let per = env_usize("D4P_PER", 3);
    let (from, n) = (env_usize("D4_FROM", 0), env_usize("D4_N", 20));
    let num = |v: &nmjson::Value| v.integer().map_err(|e| e.to_string()).and_then(|x| usize::try_from(x).map_err(|e| e.to_string()));
    let mut total_ms = 0.0;
    for q in sl.array().map_err(|e| e.to_string())?.iter().skip(from).take(n) {
        let idx = num(q.get("index").map_err(|e| e.to_string())?)?;
        let (_, text) = items.get(idx).ok_or("question")?;
        let mut wins = Vec::new();
        for w in q.get("windows").and_then(|x| x.array()).map_err(|e| e.to_string())? {
            let tentacle = num(w.get("tentacle").map_err(|e| e.to_string())?)?;
            let list: Vec<usize> = w.get("shortlist").and_then(|x| x.array()).map_err(|e| e.to_string())?.iter().filter_map(|x| num(x).ok()).collect();
            let order = order_of(&sub, &list);
            let heads = sets.for_tentacle(nmsubs::lens::NAMES.get(tentacle).copied().unwrap_or("lens")).to_vec();
            wins.push((tentacle, order, heads));
        }
        let t = Instant::now();
        let mut scores: Vec<Vec<f32>> = Vec::new();
        let live: Vec<_> = wins.iter().filter(|w| !w.1.is_empty()).collect();
        for chunk in live.chunks(per.max(1)) {
            let batch: Vec<nmapi::tentacle_read::Item> =
                chunk.iter().map(|w| (w.1.iter().map(|c| doc(&sub, *c)).collect(), w.2.clone(), text.clone())).collect();
            for rd in r.read_packed(&batch).map_err(|e| e.message)? {
                scores.push(rd.scores);
            }
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        total_ms += ms;
        let mut js = format!("{{\"index\":{idx},\"ms\":{ms:.0},\"windows\":[");
        let mut it = scores.iter();
        for (wi, (tentacle, order, _)) in wins.iter().enumerate() {
            if wi > 0 {
                js.push(',');
            }
            let sc = if order.is_empty() { Vec::new() } else { it.next().cloned().unwrap_or_default() };
            let _ = write!(js, "{{\"tentacle\":{tentacle},\"order\":{order:?},\"packed\":{}}}", floats(&sc));
        }
        js.push_str("]}
");
        std::fs::write(out.join(format!("q{idx:02}.json")), js).map_err(|e| e.to_string())?;
        eprintln!("q{idx:02}: {} windows packed {per} a run in {ms:.0} ms", live.len());
    }
    eprintln!("d4p: {total_ms:.0} ms over {n} questions");
    // Card time for 1 to 4 groups of 200 to 1,500 tokens (posed and content-free each).
    let src = std::fs::read_to_string(home.join("crates/nmapi/src/tentacle_pass.rs")).map_err(|e| e.to_string())?;
    let ids = model.pop.tokenizer.encode(&src).map_err(|e| e.to_string())?;
    for size in [200_usize, 500, 1000, 1500] {
        for groups in 1..=4_usize {
            let batch: Vec<nmapi::tentacle_read::Item> = (0..groups)
                .map(|g| (vec![model.pop.tokenizer.decode(&ids.iter().copied().skip(g * 1500).take(size).collect::<Vec<i32>>())], sets.heads.clone(), "where is the window composed".to_owned()))
                .collect();
            let t = Instant::now();
            r.read_packed(&batch).map_err(|e| e.message)?;
            eprintln!("D4 packed timing: {groups} groups of {size} tokens: {:.0} ms on {}", t.elapsed().as_secs_f64() * 1000.0, dev.name());
        }
    }
    Ok(())
}

/// D7 card time (lane T1, 2026-11-18): the served flow (`nm_context_pack`, serve's kernel
/// switches, memory off, the dense store from `NANO_DENSE_DIR`) on the large code tuning fold,
/// every question asked with the tentacles' read off, then on (`D7_ARM` picks one arm a process); per arm the median and 95th
/// percentile seconds a call after the first, and the trace's read lines. Env: `D7_ROOT`,
/// `D7_QUESTIONS`, `D7_N`. Run under the card lock:
/// `cargo test --release -p nmapi --test t1_integration d7_served_time -- --ignored --nocapture`.
#[test]
#[ignore = "card job: run under the card lock"]
fn d7_served_time() {
    if let Err(e) = d7t() {
        panic!("d7t: {e}");
    }
}

fn d7t() -> R<()> {
    use nmjson::Value;
    let home = nmtype::nano_root();
    let root = nmsubs::repo::canon(&home.join(std::env::var("D7_ROOT").unwrap_or_else(|_| "tests/dev-large/material-code".into())));
    let qf = home.join(std::env::var("D7_QUESTIONS").unwrap_or_else(|_| "tests/dev-large/questions-code-tune.json".into()));
    let n = env_usize("D7_N", 1000);
    let qs = nmjson::parse(&std::fs::read(&qf).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let questions: Vec<String> = qs
        .get("questions")
        .and_then(|q| q.array())
        .map_err(|e| e.to_string())?
        .iter()
        .filter_map(|q| q.get("question").and_then(Value::string).ok().map(str::to_owned))
        .take(n)
        .collect();
    let memory = home.join("scratch/lane-t1/d7mem");
    let org = nmapi::forage::Organism::open_with(&home, &root, &["neuromorph-nano".to_owned()], false, &memory).map_err(|e| e.message)?;
    nmengn::gpu::HALF_MATH.store(true, std::sync::atomic::Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, std::sync::atomic::Ordering::Relaxed);
    nmapi::neural::bind(&org).map_err(|e| e.message)?;
    let mut path = nmapi::whole::WholePath::new(&org.home);
    if org.max_run_ms > 0 {
        path.budget_s = f64::from(org.max_run_ms) / 1000.0;
    }
    // The one flow, as `serve` sets it.
    path.default_mode = Some(nmapi::whole::Mode::Flow);
    let path = Arc::new(path);
    path.warm().map_err(|e| e.message)?;
    if let Ok(mut w) = org.whole.write() {
        *w = Some(path);
    }
    nmapi::tools::install(org).map_err(|e| format!("{}: {}", e.code, e.message))?;
    let ask = |q: &str| -> String {
        let input = Value::Object([("question".to_owned(), Value::String(q.to_owned()))].into_iter().collect());
        match nmapi::tools::api().dispatch("nm_context_pack", &input) {
            Ok(Value::String(t)) => t,
            Ok(v) => v.encode().unwrap_or_default(),
            Err(e) => format!("error {}: {}", e.code, e.message),
        }
    };
    // One arm a process (`D7_ARM=off|on`; both when unset): the served path keeps answered
    // questions, so a second arm in the same process would be served from that store.
    let arms: Vec<bool> = match std::env::var("D7_ARM").as_deref() {
        Ok("on") => vec![true],
        Ok("off") => vec![false],
        _ => vec![false, true],
    };
    for arm in arms {
        nmapi::tentacle_read::set_on(Some(arm));
        let mut secs = Vec::new();
        let mut fails = 0;
        for (i, q) in questions.iter().enumerate() {
            let t = Instant::now();
            let out = ask(q);
            let _ = nmapi::tools::take_diagnostics();
            let s = t.elapsed().as_secs_f64();
            if out.starts_with("error") {
                fails += 1;
            }
            if i > 0 {
                secs.push(s);
            }
            eprintln!("d7 {} q{i:02} {s:.3} s{}", if arm { "on" } else { "off" }, if out.contains("the read failed") { " (read failed)" } else { "" });
        }
        secs.sort_by(f64::total_cmp);
        let at = |p: f64| {
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss, reason = "an index into a short list")]
            let k = ((secs.len().max(1) - 1) as f64 * p).round() as usize;
            secs.get(k).copied().unwrap_or(0.0)
        };
        eprintln!("D7 served time, read {}: {} calls after the first, median {:.3} s, p95 {:.3} s, worst {:.3} s, errors {fails}", if arm { "on" } else { "off" }, secs.len(), at(0.5), at(0.95), at(1.0));
    }
    nmapi::tentacle_read::set_on(None);
    Ok(())
}
