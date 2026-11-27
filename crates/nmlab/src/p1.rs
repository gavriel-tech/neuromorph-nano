//! Step P1 timing part tests (2026-11-10, `docs/2026-11-10-neumor-plan-whole-architecture.md`;
//! Gavriel Technologies, architecture by Luis Jake Gabriel III). Measurement only: nothing here
//! changes a result. Stage times come from `nmffi::prof` spans (off in every other command);
//! with timing on, the engine synchronises the device at the edges of every component so
//! queued GPU work is charged to the component that queued it.
//!
//! - `run p1 s1`: where encoding time goes. Chunks: window 1 (the first lens window) of
//!   development question index 0, in the order the pass places them (path, first line), taken
//!   from the start until at least `S1_TOKENS` tokens. Each chunk is encoded after the anchor
//!   through layer 17 (as the pass keeps them) three times: timing off, timing on, and through
//!   all 28 layers with timing off. Report `runs/P1-S1-encode-<ms>.txt`.
//! - `run p1 s2`: where read time goes. Window 1 of question index 0 is encoded whole and held;
//!   questions index 0 and 1 are each read as the pass reads them (a new `Window` per question,
//!   the posed read, then the no-question read), timing off and on; then the role-model release
//!   and reload and the reranker judge (load apart from scoring). Report
//!   `runs/P1-S2-read-<ms>.txt`.
//! - `run p1 s3`: token counts (base tokenizer, the text the pass encodes per chunk) of what the
//!   mini studies, by kind, with the bytes each kind needs at the bytes per token kept. Report
//!   `runs/P1-S3-size-<ms>.txt`.
//!
//! Each report is a new file per run.
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use crate::MINI_ANCHOR;
use nmapi::forage::Organism;
use nmapi::tentacle_pass::{Settings, TentaclePass, drone_read, drones};
use nmengn::kvstore::{self, ChunkCache, ChunkKv, Window, encode, encode_through};
use nmengn::placement::Device;
use nmffi::prof;
use nmsubs::Substrate;
use nmtype::{Fault, FaultKind};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("p1", FaultKind::Execution, m)
}

/// Tokens S1 encodes (about half a window; the brief's "about 6,000").
const S1_TOKENS: usize = 6_000;
/// Shortlist per window the judge timing uses (the served shortlist).
const JUDGE_PER_WINDOW: usize = nmapi::tentacle_pass::SERVED_SHORTLIST;

fn open(home: &Path, out: &mut String) -> Result<Organism, Fault> {
    let org = Organism::open_with(home, home, &["neuromorph-nano".to_owned()], false, &home.join("memory"))?;
    let _ = writeln!(out, "neural: {}", nmapi::neural::bind(&org)?);
    let _ = writeln!(out, "dense: {}", nmapi::neural::start_dense(&org)?);
    // The dense lane indexes missing chunks on the GPU in a background thread; timing waits
    // until every chunk has its vector so that work does not share the device with the test
    // (seen in the first S1 run: chunks timed during indexing took about twice as long).
    let sub = org.sub();
    let t = Instant::now();
    let lane = sub.dense.read().ok().and_then(|g| g.clone());
    let missing = || lane.as_ref().map_or(0, |l| sub.chunks.iter().filter(|c| l.vector(&c.key).is_none()).count());
    let first = missing();
    while missing() > 0 && t.elapsed().as_secs() < 240 {
        #[expect(clippy::disallowed_methods, reason = "synchronous part test polling a background indexing thread; no async runtime here")]
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    let _ = writeln!(out, "dense lane: {first} chunks lacked a vector at start; waited {:.1} s; still missing {}", secs(t), missing());
    Ok(org)
}

fn path_of(sub: &Substrate, c: usize) -> String {
    sub.chunks.get(c).and_then(|ch| sub.files.get(ch.file)).map_or_else(String::new, |f| f.path.replace('\\', "/"))
}
fn doc_of(sub: &Substrate, c: usize) -> String {
    sub.chunks.get(c).and_then(|ch| sub.files.get(ch.file).map(|f| nmsubs::dense::doc_text(&f.path, &ch.text))).unwrap_or_default()
}
/// A window's chunks in the order the pass places them (path, then first line).
fn placed(sub: &Substrate, win: &[usize]) -> Vec<usize> {
    let mut place: Vec<usize> = win.iter().copied().filter(|c| sub.chunks.get(*c).is_some()).collect();
    place.sort_by(|a, b| {
        (path_of(sub, *a), sub.chunks.get(*a).map_or(0, |c| c.start)).cmp(&(path_of(sub, *b), sub.chunks.get(*b).map_or(0, |c| c.start)))
    });
    place
}

fn secs(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}

/// Stages named by a span of their own (a path through one of them belongs to it).
const NAMED: [&str; 19] = [
    "capture.download",
    "capture.store",
    "window.device",
    "probe.download",
    "probe.mass",
    "score.mask",
    "score.concat",
    "weights.upload",
    "weights.release",
    "window.build",
    "encode.collect",
    "encode.setup",
    "tokenize",
    "cache.put",
    "run.targets",
    "run.drop",
    "run.seeds",
    "run.plan",
    "result.convert",
];

/// The stage group of a span path: the named stage when the path passes one; inside an
/// operation, what the time was (kernels waited for at the component's end sync or inside a
/// buffer free, buffer allocation, buffer free, launch calls, host work); else the wrapper.
fn group(path: &str) -> String {
    let parts: Vec<&str> = path.split(" > ").collect();
    if let Some(n) = NAMED.iter().find(|n| parts.contains(n)) {
        return (*n).to_owned();
    }
    let leaf = parts.last().copied().unwrap_or("");
    if let Some(op) = parts.iter().find(|p| p.starts_with("op.")) {
        let what = match leaf {
            "sync.edge" => "kernels (waited at the end sync)",
            "cuda.free.wait" => "kernels (waited inside a buffer free)",
            "cuda.alloc" => "buffer alloc and zeroing",
            "cuda.free" => "buffer free",
            "cuda.launch" => "launch calls",
            x if x.starts_with("cuda.") => "copies",
            _ => "host work",
        };
        return format!("{op}: {what}");
    }
    if leaf == "sync.edge" {
        return "sync.edge before a component (queued work left over)".to_owned();
    }
    format!("{leaf} (own)")
}

/// The coarse category of a span path (one table that adds up to the total).
fn category(path: &str) -> &'static str {
    let parts: Vec<&str> = path.split(" > ").collect();
    let leaf = parts.last().copied().unwrap_or("");
    let has = |x: &str| parts.contains(&x);
    if has("weights.upload") {
        "weights upload to the device"
    } else if has("weights.release") {
        "weights release"
    } else if has("window.device") {
        "window upload to the device (keys, values, rope)"
    } else if has("capture.download") {
        "key and value capture: download to host"
    } else if has("capture.store") {
        "key and value capture: u16 packing and storing"
    } else if has("probe.download") {
        "probe: download of query and key rows"
    } else if has("probe.mass") {
        "probe: attention mass on the host"
    } else if has("tokenize") {
        "tokenizing"
    } else if leaf == "sync.edge" || leaf == "cuda.free.wait" {
        if parts.iter().any(|p| p.starts_with("op.")) { "GPU kernels (forward pass)" } else { "waits outside a component" }
    } else if leaf == "cuda.alloc" || leaf == "cuda.free" {
        "device buffer alloc, zeroing and free (per operation)"
    } else if leaf == "cuda.launch" {
        "kernel launch calls (host)"
    } else if leaf.starts_with("cuda.") {
        "other copies (ids, mask, concat, targets)"
    } else {
        "host work in the engine and test (planning, glue)"
    }
}

/// The current timing tables as text: grouped stages, totals per leaf, per-layer side totals,
/// and the full rows. `wall` is the wall time of what was timed.
fn tables(out: &mut String, title: &str, wall: f64, layers: bool) -> f64 {
    let rows = prof::table();
    let sum: f64 = rows.iter().map(|r| r.1).sum();
    let pct = |x: f64| if sum > 0.0 { 100.0 * x / sum } else { 0.0 };
    let _ = writeln!(
        out,
        "\n{title}: wall {wall:.3} s; sum of the exclusive stage times {sum:.3} s ({:.1}% of the wall; the rest is untimed glue in the test itself)",
        if wall > 0.0 { 100.0 * sum / wall } else { 0.0 }
    );
    let mut g: BTreeMap<String, (f64, u64)> = BTreeMap::new();
    let mut leaf: BTreeMap<String, (f64, u64)> = BTreeMap::new();
    for (p, s, n) in &rows {
        let e = g.entry(group(p)).or_default();
        e.0 += s;
        e.1 += n;
        let l = p.rsplit(" > ").next().unwrap_or(p).to_owned();
        let e = leaf.entry(l).or_default();
        e.0 += s;
        e.1 += n;
    }
    let mut cat: BTreeMap<&str, f64> = BTreeMap::new();
    for (p, sec, _) in &rows {
        *cat.entry(category(p)).or_default() += sec;
    }
    let mut cv: Vec<_> = cat.into_iter().collect();
    cv.sort_by(|a, b| b.1.total_cmp(&a.1));
    let _ = writeln!(out, "category  seconds  share");
    for (k, sec) in &cv {
        let _ = writeln!(out, "  {k}  {sec:.3}  {:.1}%", pct(*sec));
    }
    let mut gv: Vec<_> = g.into_iter().collect();
    gv.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
    let _ = writeln!(
        out,
        "stage group  seconds  share  spans  (exclusive time; children such as cuda.alloc inside an operation are counted in that operation's group)"
    );
    for (k, (s, n)) in &gv {
        let _ = writeln!(out, "  {k}  {s:.3}  {:.1}%  {n}", pct(*s));
    }
    let mut lv: Vec<_> = leaf.into_iter().collect();
    lv.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
    let _ = writeln!(out, "by innermost span (all groups together)  seconds  share  count");
    for (k, (s, n)) in &lv {
        let _ = writeln!(out, "  {k}  {s:.3}  {:.1}%  {n}", pct(*s));
    }
    if layers {
        let _ = writeln!(out, "per layer (inclusive time of the components on that layer, with sync edges)  seconds  components");
        for (k, s, n) in prof::side() {
            let _ = writeln!(out, "  {k}  {s:.3}  {n}");
        }
    }
    let mut rv = rows.clone();
    rv.sort_by(|a, b| b.1.total_cmp(&a.1));
    let _ = writeln!(out, "full span paths (top 40 by time)  seconds  share  count");
    for (p, s, n) in rv.iter().take(40) {
        let _ = writeln!(out, "  {p}  {s:.4}  {:.1}%  {n}", pct(*s));
    }
    sum
}

/// Least squares of y on x: (intercept, slope).
fn fit(pts: &[(f64, f64)]) -> (f64, f64) {
    #[expect(clippy::cast_precision_loss, reason = "a few dozen points")]
    let n = pts.len() as f64;
    let (sx, sy) = (pts.iter().map(|p| p.0).sum::<f64>(), pts.iter().map(|p| p.1).sum::<f64>());
    let (sxx, sxy) = (pts.iter().map(|p| p.0 * p.0).sum::<f64>(), pts.iter().map(|p| p.0 * p.1).sum::<f64>());
    let den = n * sxx - sx * sx;
    if den.abs() < 1e-12 {
        return (sy / n.max(1.0), 0.0);
    }
    let b = (n * sxy - sx * sy) / den;
    ((sy - b * sx) / n, b)
}

fn s1(home: &Path) -> Result<String, Fault> {
    let mut out = String::new();
    let _ = writeln!(out, "P1 S1: encode time by stage (GPU), 2026-11-10");
    let org = open(home, &mut out)?;
    let sub = org.sub();
    let items = crate::e2::suite(home)?;
    let q0 = &items.first().ok_or_else(|| bad("no development question"))?.q;
    let pass = TentaclePass::new(home, Settings::default());
    let (tokens, windows) = pass.layout(&sub, q0)?;
    let model = nmapi::neural::model(home)?;
    let pop = Arc::clone(&model.pop);
    let ops = pop.gpu.as_ref().ok_or_else(|| bad("S1 measures the GPU path; no GPU is open"))?;
    let _ = writeln!(out, "device: {}", pop.gpu_note);
    // The role models give the device up as the pass does.
    for (_, m) in nmapi::neural::bound_models() {
        if m.pop.card.id != pop.card.id {
            m.pop.release_device();
        }
    }
    let dev = Device::Gpu;
    let tok = |s: &str| pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
    let last = drones(home).last_layer();
    let win = windows.first().ok_or_else(|| bad("no window"))?;
    let place = placed(&sub, win);
    let mut chosen = Vec::new();
    let mut acc = 0;
    for c in place {
        if acc >= S1_TOKENS {
            break;
        }
        acc += tokens.get(c).copied().unwrap_or(0);
        chosen.push(c);
    }
    let _ = writeln!(
        out,
        "rule: window 1 (first lens window) of development question index 0, chunks in the pass's placing order (path, first line), from the start until at least {S1_TOKENS} tokens: {} chunks, {acc} tokens (layout count); encoded after the anchor through layer {last} as the pass keeps them",
        chosen.len()
    );

    // Anchor, and a warm-up encode of the first chosen chunk (weights uploaded, kernels warm).
    let t = Instant::now();
    let ak = Arc::new(encode_through(&model, dev, None, &tok(MINI_ANCHOR)?, last)?);
    let anchor = Arc::new(Window::build(&model, &[Arc::clone(&ak)])?);
    let warm_ids = tok(&doc_of(&sub, *chosen.first().ok_or_else(|| bad("no chunk"))?))?;
    let _ = encode_through(&model, dev, Some(&anchor), &warm_ids, last)?;
    let _ = writeln!(
        out,
        "warm-up (anchor {} tokens, weights upload, first chunk once): {:.3} s, device weight bytes {}",
        ak.tokens,
        secs(t),
        pop.device_bytes()
    );

    let docs: Vec<String> = chosen.iter().map(|c| doc_of(&sub, *c)).collect();
    // Pass A: timing off.
    let mut per = Vec::new();
    let mut tok_s = 0.0;
    let mut enc_a = 0.0;
    let mut cache = ChunkCache::new(usize::MAX);
    let ta = Instant::now();
    for (i, d) in docs.iter().enumerate() {
        let t = Instant::now();
        let ids = tok(d)?;
        let tk = secs(t);
        let t = Instant::now();
        let kv = Arc::new(encode_through(&model, dev, Some(&anchor), &ids, last)?);
        let e = secs(t);
        cache.put(&format!("{i}"), kv);
        tok_s += tk;
        enc_a += e;
        per.push((ids.len(), tk, e));
    }
    let wall_a = secs(ta);
    let n_tok: usize = per.iter().map(|p| p.0).sum();
    #[expect(clippy::cast_precision_loss, reason = "token counts")]
    let ntf = n_tok as f64;
    let _ = writeln!(
        out,
        "\nA. timing off (normal run): {} chunks, {n_tok} tokens; wall {wall_a:.3} s; tokenizing {tok_s:.3} s; encode_through {enc_a:.3} s = {:.3} ms per token",
        per.len(),
        enc_a * 1000.0 / ntf.max(1.0)
    );
    let _ = writeln!(out, "chunk  tokens  tokenize_ms  encode_ms  encode_ms_per_token");
    for (i, (n, tk, e)) in per.iter().enumerate() {
        #[expect(clippy::cast_precision_loss, reason = "token counts")]
        let pt = e * 1000.0 / (*n).max(1) as f64;
        let _ = writeln!(out, "  {}  {n}  {:.2}  {:.1}  {pt:.3}", i + 1, tk * 1000.0, e * 1000.0);
    }
    #[expect(clippy::cast_precision_loss, reason = "token counts")]
    let pts: Vec<(f64, f64)> = per.iter().map(|p| (p.0 as f64, p.2 * 1000.0)).collect();
    let (a0, b0) = fit(&pts);
    let _ = writeln!(
        out,
        "fit of encode ms on chunk tokens (least squares, {} chunks): fixed {a0:.1} ms per call + {b0:.3} ms per token; fixed share of this set {:.1}%",
        pts.len(),
        if enc_a > 0.0 { 100.0 * a0 * pts.len() as f64 / (enc_a * 1000.0) } else { 0.0 }
    );

    // Pass B: timing on, same chunks.
    prof::reset();
    prof::set(true);
    let tb = Instant::now();
    let mut cache_b = ChunkCache::new(usize::MAX);
    for (i, d) in docs.iter().enumerate() {
        let ids = {
            let _s = prof::Span::new("tokenize");
            tok(d)?
        };
        let kv = Arc::new(encode_through(&model, dev, Some(&anchor), &ids, last)?);
        let _s = prof::Span::new("cache.put");
        cache_b.put(&format!("{i}"), kv);
    }
    let _ = ops.gpu.sync();
    let wall_b = secs(tb);
    prof::set(false);
    let sum_b = tables(&mut out, "B. timing on (device synchronised at every component edge), same chunks", wall_b, true);
    let _ = writeln!(
        out,
        "B against A: timed wall {wall_b:.3} s against {wall_a:.3} s untimed ({:+.1}%: the cost of the sync edges and spans)",
        100.0 * (wall_b - wall_a) / wall_a.max(1e-9)
    );
    let _ = sum_b;
    // Same bits with timing on (the timing code changes no result).
    let same = (0..docs.len()).all(|i| {
        let k = format!("{i}");
        match (cache.get(&k), cache_b.get(&k)) {
            (Some(x), Some(y)) => x.k_pre == y.k_pre && x.v == y.v,
            _ => false,
        }
    });
    let _ = writeln!(out, "encoded keys and values with timing on equal those with timing off, bit for bit: {same}");

    // Pass A2: timing off again after B (order control: device clocks, background work).
    let mut enc_a2 = 0.0;
    let mut per2 = Vec::new();
    for d in &docs {
        let ids = tok(d)?;
        let t = Instant::now();
        let _kv = encode_through(&model, dev, Some(&anchor), &ids, last)?;
        let e = secs(t);
        enc_a2 += e;
        per2.push(format!("{:.0}", e * 1000.0));
    }
    let _ = writeln!(
        out,
        "A2. timing off again after B (order control): encode_through {enc_a2:.3} s = {:.3} ms per token (A was {enc_a:.3} s); per chunk ms: {}",
        enc_a2 * 1000.0 / ntf.max(1.0),
        per2.join(" ")
    );

    // Pass C: all 28 layers, timing off (the anchor encoded through every layer as well, since
    // a full encoding reads the anchor on every layer).
    let full_ak = Arc::new(encode(&model, dev, None, &tok(MINI_ANCHOR)?)?);
    let full_anchor = Arc::new(Window::build(&model, &[full_ak])?);
    let tc = Instant::now();
    let mut enc_c = 0.0;
    for d in &docs {
        let ids = tok(d)?;
        let t = Instant::now();
        let _kv: ChunkKv = encode(&model, dev, Some(&full_anchor), &ids)?;
        enc_c += secs(t);
    }
    let wall_c = secs(tc);
    let _ = writeln!(
        out,
        "\nC. timing off, all {} layers (anchor also encoded through every layer): encode {enc_c:.3} s ({:.3} ms per token), wall {wall_c:.3} s; through layer {last}: {enc_a:.3} s; saving of stopping at layer {last}: {:.1}%",
        pop.dims.layers,
        enc_c * 1000.0 / ntf.max(1.0),
        100.0 * (enc_c - enc_a) / enc_c.max(1e-9)
    );
    pop.release_device();
    Ok(out)
}

fn s2(home: &Path) -> Result<String, Fault> {
    let mut out = String::new();
    let _ = writeln!(out, "P1 S2: read time by stage (GPU), 2026-11-10");
    let org = open(home, &mut out)?;
    let sub = org.sub();
    let items = crate::e2::suite(home)?;
    let (q0, q1) = (&items.first().ok_or_else(|| bad("no question"))?.q, &items.get(1).ok_or_else(|| bad("no second question"))?.q);
    let pass = TentaclePass::new(home, Settings::default());
    let model = nmapi::neural::model(home)?;
    let pop = Arc::clone(&model.pop);
    let ops = pop.gpu.as_ref().ok_or_else(|| bad("S2 measures the GPU path; no GPU is open"))?;
    let _ = writeln!(out, "device: {}", pop.gpu_note);

    // Layout of question index 0 (the embedder orders the lenses), timed with spans.
    prof::reset();
    prof::set(true);
    let t = Instant::now();
    let (_tokens, windows) = pass.layout(&sub, q0)?;
    let lay0 = secs(t);
    prof::set(false);
    tables(
        &mut out,
        "layout of question index 0 (first in the process: chunk tokenizing of the whole tree, embedder load and query embedding)",
        lay0,
        false,
    );

    let release_roles = || -> f64 {
        let t = Instant::now();
        for (_, m) in nmapi::neural::bound_models() {
            if m.pop.card.id != pop.card.id {
                m.pop.release_device();
            }
        }
        secs(t)
    };
    let r0 = release_roles();
    let dev = Device::Gpu;
    let tok = |s: &str| pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
    let ds = drones(home);
    let last = ds.last_layer();
    let heads = ds.for_tentacle(nmsubs::lens::NAMES.first().copied().unwrap_or("lens")).to_vec();
    let stop = kvstore::score_targets(&model, last)?;
    let win = windows.first().ok_or_else(|| bad("no window"))?;
    let place = placed(&sub, win);
    let t = Instant::now();
    let ak = Arc::new(encode_through(&model, dev, None, &tok(MINI_ANCHOR)?, last)?);
    let anchor = Arc::new(Window::build(&model, &[Arc::clone(&ak)])?);
    let mut parts = vec![Arc::clone(&ak)];
    let mut enc_tokens = 0;
    for &c in &place {
        let ids = tok(&doc_of(&sub, c))?;
        enc_tokens += ids.len();
        parts.push(Arc::new(encode_through(&model, dev, Some(&anchor), &ids, last)?));
    }
    let _ = writeln!(
        out,
        "\nrole models released before encoding: {r0:.3} s. Window 1 of question index 0 encoded and held: {} chunks, {enc_tokens} tokens plus the anchor's {}, {:.1} s (setup, not part of a read)",
        place.len(),
        ak.tokens,
        secs(t)
    );
    let _ = writeln!(out, "drone set: {} pairs ({}), stop at layer {last}", heads.len(), ds.source);

    let posed = |q: &str| format!("\nQuestion: {q}");
    let null = "\nQuestion: N/A".to_owned();
    // Untimed reads (timing off) for each question, as the pass does them: a new Window, then
    // the posed read, then the no-question read; then a second posed read on the same Window
    // (device copies already made) to isolate the window upload.
    let mut nulls = Vec::new();
    let mut poseds = Vec::new();
    let _ = writeln!(
        out,
        "\nA. timing off (normal run), one window of {} tokens (anchor included)",
        parts.iter().map(|p| p.tokens).sum::<usize>()
    );
    let _ = writeln!(
        out,
        "question  build_ms  posed_read_ms  null_read_ms  read_total_ms  posed_again_same_window_ms  null_again_same_window_ms  question_tokens"
    );
    let mut base_read = 0.0;
    for (i, q) in [q0, q1].iter().enumerate() {
        let t = Instant::now();
        let w = Arc::new(Window::build(&model, &parts)?);
        let b = secs(t);
        let t = Instant::now();
        let a = drone_read(&model, dev, &w, &stop, &heads, &posed(q))?;
        let pa = secs(t);
        let t = Instant::now();
        let n = drone_read(&model, dev, &w, &stop, &heads, &null)?;
        let na = secs(t);
        let t = Instant::now();
        let a2 = drone_read(&model, dev, &w, &stop, &heads, &posed(q))?;
        let pa2 = secs(t);
        let t = Instant::now();
        let n2 = drone_read(&model, dev, &w, &stop, &heads, &null)?;
        let na2 = secs(t);
        let _ = writeln!(
            out,
            "  index {i}  {:.1}  {:.1}  {:.1}  {:.1}  {:.1}  {:.1}  {}",
            b * 1000.0,
            pa * 1000.0,
            na * 1000.0,
            (b + pa + na) * 1000.0,
            pa2 * 1000.0,
            na2 * 1000.0,
            tok(&posed(q))?.len()
        );
        if i == 0 {
            base_read = b + pa + na;
        }
        let _ = writeln!(out, "    repeat reads on the same window equal the first, bit for bit: posed {}, null {}", a == a2, n == n2);
        poseds.push(a);
        nulls.push(n);
    }
    let null_same = nulls.first() == nulls.get(1);
    let null_maxdiff =
        nulls.first().zip(nulls.get(1)).map_or(f32::NAN, |(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max));
    let posed_same = poseds.first() == poseds.get(1);
    let _ = writeln!(
        out,
        "(b) no-question read of the same window for question index 0 and index 1 (new Window each time): identical bit for bit {null_same} ({} values, max abs difference {null_maxdiff}); posed reads identical {posed_same} (expected false)",
        nulls.first().map_or(0, Vec::len)
    );

    // Timed reads.
    for (i, q) in [q0, q1].iter().enumerate() {
        prof::reset();
        prof::set(true);
        let t = Instant::now();
        let w = Arc::new(Window::build(&model, &parts)?);
        {
            let _s = prof::Span::new("read.posed");
            let _ = drone_read(&model, dev, &w, &stop, &heads, &posed(q))?;
        }
        {
            let _s = prof::Span::new("read.null");
            let _ = drone_read(&model, dev, &w, &stop, &heads, &null)?;
        }
        let _ = ops.gpu.sync();
        let wall = secs(t);
        prof::set(false);
        tables(
            &mut out,
            &format!(
                "B{}. timing on, question index {i}: Window::build, posed read, no-question read (stage paths start with read.posed or read.null)",
                i + 1
            ),
            wall,
            true,
        );
        // Split per read.
        let rows = prof::table();
        let part = |pre: &str, what: &str| rows.iter().filter(|r| r.0.starts_with(pre) && r.0.contains(what)).map(|r| r.1).sum::<f64>();
        let (pt, nt) = (part("read.posed", ""), part("read.null", ""));
        let _ = writeln!(
            out,
            "split: posed read {pt:.3} s (window upload {:.3} s), no-question read {nt:.3} s (window upload {:.3} s)",
            part("read.posed", "window.device"),
            part("read.null", "window.device")
        );
    }

    // (c) role models and judge.
    let _ = writeln!(out, "\n(c) role-model release and reload, and the reranker judge");
    let t = Instant::now();
    pop.release_device();
    let rb = secs(t);
    let mut docs = Vec::new();
    for w in &windows {
        for c in placed(&sub, w).into_iter().take(JUDGE_PER_WINDOW) {
            docs.push(doc_of(&sub, c));
        }
    }
    let doc_tokens: usize = docs.iter().map(|d| tok(d).map_or(0, |v| v.len())).sum();
    let _ = writeln!(out, "base model released after the reads (tentacle_pass.rs line 830): {rb:.3} s");
    let _ = writeln!(
        out,
        "judge docs: the first {JUDGE_PER_WINDOW} chunks (placing order) of each of the {} windows of question index 0 = {} documents, {doc_tokens} tokens before the reranker's cap (stands in for the shortlisted chunks: same count and kind)",
        windows.len(),
        docs.len()
    );
    for (i, q) in [q0, q1].iter().enumerate() {
        prof::reset();
        prof::set(true);
        let t = Instant::now();
        let v = org.slots.rerank_set_neural(q, &docs);
        let wall = secs(t);
        prof::set(false);
        let rows = prof::table();
        let up: f64 = rows.iter().filter(|r| r.0.contains("weights.upload")).map(|r| r.1).sum();
        let _ = writeln!(
            out,
            "judge call {} (question index {i}, {}): wall {wall:.3} s; weights upload {up:.3} s; scoring and the rest {:.3} s; scores returned {}",
            i + 1,
            if i == 0 { "reranker not resident, as after every scan" } else { "reranker resident" },
            wall - up,
            v.as_ref().map_or(0, Vec::len)
        );
        tables(&mut out, &format!("judge call {} stages", i + 1), wall, false);
    }
    // Next question: the pass releases the role models, the layout reloads the embedder, the
    // base model's weights go up again with the first read.
    let r1 = release_roles();
    let _ = writeln!(
        out,
        "\nrole models released at the next scan (tentacle_pass.rs lines 733 to 737), reranker and embedder resident: {r1:.3} s"
    );
    prof::reset();
    prof::set(true);
    let t = Instant::now();
    let _ = pass.layout(&sub, q1)?;
    let lay1 = secs(t);
    prof::set(false);
    let up_lay: f64 = prof::table().iter().filter(|r| r.0.contains("weights.upload")).map(|r| r.1).sum();
    let _ = writeln!(out, "layout of question index 1 (chunk tokens kept): {lay1:.3} s, of which embedder weights upload {up_lay:.3} s");
    tables(&mut out, "layout of question index 1 stages", lay1, false);
    let r2 = release_roles();
    prof::reset();
    prof::set(true);
    let t = Instant::now();
    let w = Arc::new(Window::build(&model, &parts)?);
    let _ = drone_read(&model, dev, &w, &stop, &heads, &posed(q1))?;
    let _ = drone_read(&model, dev, &w, &stop, &heads, &null)?;
    let _ = ops.gpu.sync();
    let wall = secs(t);
    prof::set(false);
    let up_base: f64 = prof::table().iter().filter(|r| r.0.contains("weights.upload")).map(|r| r.1).sum();
    let _ = writeln!(
        out,
        "embedder released again: {r2:.3} s. First window read after the base model was released: {wall:.3} s timed, of which base weights upload {up_base:.3} s (charged once per question, to the first read or encode)"
    );
    let _ =
        writeln!(out, "\nreference: one untimed read of this window (build, posed, no-question) for question index 0: {:.3} s", base_read);
    pop.release_device();
    Ok(out)
}

fn kind_of(path: &str) -> &'static str {
    let p = path.replace('\\', "/");
    if p.starts_with("crates/") && p.ends_with(".rs") {
        "Rust source under crates/"
    } else if p.starts_with("crates/") {
        "other files under crates/ (cards, Cargo.toml, READMEs)"
    } else if p.starts_with("designs/") {
        "design data under designs/"
    } else if p.starts_with("docs/") {
        "documents under docs/"
    } else if p.starts_with("runs/") {
        "run records under runs/"
    } else {
        "everything else studied"
    }
}

fn s3(home: &Path) -> Result<String, Fault> {
    let mut out = String::new();
    let _ = writeln!(out, "P1 S3: size of the studied code base in base-tokenizer tokens, 2026-11-10");
    let org = open(home, &mut out)?;
    let sub = org.sub();
    let items = crate::e2::suite(home)?;
    let q0 = &items.first().ok_or_else(|| bad("no question"))?.q;
    let pass = TentaclePass::new(home, Settings::default());
    let (tokens, _) = pass.layout(&sub, q0)?;
    let model = nmapi::neural::model(home)?;
    let per = kvstore::bytes_per_token(&model, drones(home).last_layer() + 1);
    let mut k: BTreeMap<&str, (usize, usize, usize, usize, std::collections::BTreeSet<usize>)> = BTreeMap::new();
    let mut other: BTreeMap<String, usize> = BTreeMap::new();
    let mut failed = 0;
    let mut total = 0;
    for (c, ch) in sub.chunks.iter().enumerate() {
        let n = tokens.get(c).copied().unwrap_or(usize::MAX);
        if n == usize::MAX {
            failed += 1;
            continue;
        }
        total += n;
        let path = path_of(&sub, c);
        let kind = kind_of(&path);
        let e = k.entry(kind).or_default();
        e.0 += n;
        e.1 += 1;
        e.2 += ch.text.len();
        e.4.insert(ch.file);
        if kind == "everything else studied" {
            *other.entry(path.split('/').next().unwrap_or("").to_owned()).or_default() += n;
        }
    }
    for e in k.values_mut() {
        e.3 = e.4.len();
    }
    let _ = writeln!(
        out,
        "studied: {} files, {} chunks; token count per chunk = tokens of the text the pass encodes for it (path line plus chunk lines, `nmsubs::dense::doc_text`), the same count the layout sums",
        sub.files.len(),
        sub.chunks.len()
    );
    let _ = writeln!(out, "bytes per token kept (layers 0..={}): {per}", drones(home).last_layer());
    let _ = writeln!(out, "kind  tokens  share  chunks  files  text_bytes  bytes_kept_at_{per}  GiB");
    let mut kv: Vec<_> = k.iter().collect();
    kv.sort_by_key(|e| std::cmp::Reverse(e.1.0));
    for (name, (n, ch, by, files, _)) in kv {
        #[expect(clippy::cast_precision_loss, reason = "a report")]
        let (share, gib) = (100.0 * *n as f64 / total.max(1) as f64, (*n * per) as f64 / f64::from(1_u32 << 30));
        let _ = writeln!(out, "  {name}  {n}  {share:.1}%  {ch}  {files}  {by}  {}  {gib:.2}", n * per);
    }
    #[expect(clippy::cast_precision_loss, reason = "a report")]
    let gib = (total * per) as f64 / f64::from(1_u32 << 30);
    let _ = writeln!(out, "  total  {total}  100%  {}  {}  -  {}  {gib:.2}", sub.chunks.len() - failed, sub.files.len(), total * per);
    let _ = writeln!(out, "chunks whose tokenizing failed (not counted): {failed}");
    let _ = writeln!(out, "everything else studied, by top-level folder or file:");
    let mut ov: Vec<_> = other.into_iter().collect();
    ov.sort_by_key(|e| std::cmp::Reverse(e.1));
    for (f, n) in ov {
        let _ = writeln!(out, "  {f}  {n}");
    }
    let limit = kvstore::derive_ram_limit();
    let _ = writeln!(out, "RAM limit for kept chunks now: {} bytes (half of {} free of {} total)", limit.bytes, limit.free, limit.total);
    Ok(out)
}

/// `run p1 s1|s2|s3`.
pub fn run(args: &[String], home: &Path) -> ExitCode {
    let stamp = nmtype::now_ms();
    let which = args.get(2).map_or("", String::as_str);
    type Step = fn(&Path) -> Result<String, Fault>;
    let (name, f): (&str, Step) = match which {
        "s1" => ("P1-S1-encode", s1),
        "s2" => ("P1-S2-read", s2),
        "s3" => ("P1-S3-size", s3),
        _ => {
            eprintln!("usage: nano-lab run p1 s1|s2|s3");
            return ExitCode::FAILURE;
        }
    };
    let txt = home.join(format!("runs/{name}-{stamp}.txt"));
    match f(home) {
        Ok(text) => {
            let _ = std::fs::write(&txt, &text);
            print!("{text}");
            println!("report: {}", txt.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            let _ = std::fs::write(&txt, format!("FAULT {e}\n"));
            eprintln!("p1 {which}: {e} (report {})", txt.display());
            ExitCode::FAILURE
        }
    }
}
