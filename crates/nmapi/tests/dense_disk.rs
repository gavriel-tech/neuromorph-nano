//! The dense index kept on disk (`nmsubs::dense::store`): short tests on the embedder.
//!
//! `kept_fixture` runs the served lane (`nmapi::neural::start_dense`) on a copy of the small
//! fixture repository `scratch/dense-test/repo`, with test memory folders under
//! `scratch/dense-test/` (never the Nano's own `memory/`): (1) a restart embeds 0 chunks and gives
//! the same nearest chunks for 10 questions; (2) one changed file embeds only its new chunks;
//! (3) a header mismatch, a cut-off file and a file of another kind are each not used, with a
//! named reason; (4) memory off writes nothing.
//!
//! `kept_frozen` pairs stored against fresh vectors on the frozen tree `scratch/frozen-p2` (read
//! only): one full index on the graphics card, kept on disk; a lane over the same embedder outputs
//! held at full binary32; a restarted lane that loads the store. For the 40 development questions
//! the lane's nearest chunks and fused order are compared, and whether the right lines are among
//! the lane's fused places is counted for both.
//!
//! Run: `cargo test --release -p nmapi --test dense_disk -- --ignored --nocapture --test-threads 1`
//! (`kept_fixture` a few minutes, `kept_frozen` about 20 minutes). Each writes
//! `runs/dense-disk-<name>-<ms>.txt`.
use nmapi::forage::Organism;
use nmsubs::dense::{DenseLane, store};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Questions about the fixture's code, written for this test.
const QUESTIONS: [&str; 10] = [
    "where is a JSON text parsed into a value",
    "how is a string escaped when a value is encoded",
    "which function prints a value with indentation",
    "where is the SHA-256 compression of one block",
    "how are the bytes of a file hashed while they are read",
    "what error type carries a code and a repair hint",
    "where is an object member looked up by its key",
    "which checks run for the json crate",
    "how is an integer read from a value",
    "where are the known hash test vectors",
];

fn home() -> PathBuf {
    nmtype::nano_root()
}

/// The served defaults of the arithmetic switches (`serve` sets both on).
fn serve_switches() {
    nmengn::gpu::HALF_MATH.store(true, std::sync::atomic::Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, std::sync::atomic::Ordering::Relaxed);
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("folder");
    for e in std::fs::read_dir(from).expect("read fixture").filter_map(Result::ok) {
        let p = e.path();
        if p.is_dir() {
            copy_dir(&p, &to.join(e.file_name()));
        } else {
            std::fs::copy(&p, to.join(e.file_name())).expect("copy");
        }
    }
}

/// Every file under `dir`, with its size.
fn files_under(dir: &Path) -> Vec<(String, u64)> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else { return out };
    for e in rd.filter_map(Result::ok) {
        let p = e.path();
        if p.is_dir() {
            out.extend(files_under(&p));
        } else {
            out.push((p.display().to_string(), e.metadata().map_or(0, |m| m.len())));
        }
    }
    out
}

/// A served session over `root`: the embedder slot bound and the dense lane started.
fn session(root: &Path, mem: &Path, memory_on: bool) -> (Organism, String) {
    let org = Organism::open_with(&home(), root, &["neuromorph-nano".to_owned()], memory_on, mem).expect("organism");
    nmapi::neural::bind_embedder(&org).expect("embedder");
    let note = nmapi::neural::start_dense(&org).expect("dense lane");
    (org, note)
}

/// Waits until the lane covers every chunk; returns seconds waited and the lane.
fn ready(org: &Organism) -> (f64, Arc<DenseLane>) {
    let sub = org.sub();
    let lane = sub.dense.read().ok().and_then(|d| d.clone()).expect("lane");
    let t = Instant::now();
    while !lane.ready(&sub) {
        assert!(t.elapsed() < Duration::from_secs(1800), "lane not ready after 30 minutes");
        std::thread::park_timeout(Duration::from_millis(50));
    }
    (t.elapsed().as_secs_f64(), lane)
}

/// The keys of the `k` chunks nearest `q` by meaning.
fn nearest(lane: &DenseLane, sub: &nmsubs::Substrate, q: &str, k: usize) -> Vec<String> {
    lane.embedding_order(sub, q).unwrap_or_default().into_iter().take(k).filter_map(|i| sub.chunks.get(i).map(|c| c.key.clone())).collect()
}

#[test]
#[ignore = "embeds on the graphics card (run on demand, a few minutes)"]
#[expect(clippy::too_many_lines, reason = "one short test with its record")]
fn kept_fixture() {
    serve_switches();
    let base = home().join("scratch").join("dense-test");
    let (work, mem, mem_off) = (base.join("work"), base.join("mem"), base.join("mem-off"));
    for d in [&work, &mem, &mem_off] {
        let _ = std::fs::remove_dir_all(d);
    }
    copy_dir(&base.join("repo"), &work);
    let rec = home().join("runs").join(format!("dense-disk-fixture-{}.txt", nmtype::now_ms()));
    let mut rep = String::from("dense index kept on disk: fixture tests (scratch/dense-test)\n");
    let save = |r: &str| std::fs::write(&rec, r).expect("record");
    let store_file = mem.join(store::DIR).join(store::FILE);

    // (1) first start, then a restart.
    let t = Instant::now();
    let (org, note) = session(&work, &mem, true);
    let bind_s = t.elapsed().as_secs_f64();
    let (wait_a, lane) = ready(&org);
    let sub = org.sub();
    let st_a = lane.stats.lock().expect("stats").clone();
    let first: Vec<Vec<String>> = QUESTIONS.iter().map(|q| nearest(&lane, &sub, q, 5)).collect();
    let size = std::fs::metadata(&store_file).map_or(0, |m| m.len());
    let _ = writeln!(
        rep,
        "start 1: {} chunks; embedder bound and lane started in {bind_s:.1} s; ready {wait_a:.1} s later; embedded {}; store {} bytes\n  {note}",
        sub.chunks.len(),
        st_a.embedded_now,
        size
    );
    let keys_a: BTreeSet<String> = sub.chunks.iter().map(|c| c.key.clone()).collect();
    drop((lane, sub, org));
    save(&rep);

    let t = Instant::now();
    let (org, note) = session(&work, &mem, true);
    let (wait_b, lane) = ready(&org);
    let restart_s = t.elapsed().as_secs_f64();
    let sub = org.sub();
    let st_b = lane.stats.lock().expect("stats").clone();
    let second: Vec<Vec<String>> = QUESTIONS.iter().map(|q| nearest(&lane, &sub, q, 5)).collect();
    let same = first.iter().zip(&second).filter(|(a, b)| a == b && !a.is_empty()).count();
    let _ = writeln!(
        rep,
        "(1) restart: ready {restart_s:.2} s after start (wait {wait_b:.2} s); embedded {}; loaded {}; same 5 nearest chunks on {same} of {} questions\n  {note}",
        st_b.embedded_now,
        st_b.loaded,
        QUESTIONS.len()
    );
    let pass1 = st_b.embedded_now == 0 && st_b.loaded == sub.chunks.len() && same == QUESTIONS.len();
    let _ = writeln!(rep, "  test 1: {}", if pass1 { "PASS" } else { "FAIL" });
    drop((lane, sub, org));
    save(&rep);

    // (2) one file changed.
    let changed = work.join("nmjson").join("src").join("pretty.rs");
    let mut text = std::fs::read_to_string(&changed).expect("fixture file");
    text.push_str("\n/// Counts the values in a tree, for the test of the kept index.\npub fn count(v: &Value) -> usize {\n    match v {\n        Value::Array(a) => 1 + a.iter().map(count).sum::<usize>(),\n        Value::Object(o) => 1 + o.values().map(count).sum::<usize>(),\n        _ => 1,\n    }\n}\n");
    std::fs::write(&changed, text).expect("change fixture file");
    let (org, note) = session(&work, &mem, true);
    let (_, lane) = ready(&org);
    let sub = org.sub();
    let st_c = lane.stats.lock().expect("stats").clone();
    let new_keys: Vec<&nmsubs::Chunk> = sub.chunks.iter().filter(|c| !keys_a.contains(&c.key)).collect();
    let new_files: BTreeSet<String> = new_keys.iter().filter_map(|c| sub.files.get(c.file).map(|f| f.path.replace('\\', "/"))).collect();
    let _ = writeln!(
        rep,
        "(2) one file changed (nmjson/src/pretty.rs): embedded {} at start; chunks with a new hash {} (in files {new_files:?}); loaded {}\n  {note}",
        st_c.embedded_now,
        new_keys.len(),
        st_c.loaded
    );
    let pass2 = !new_keys.is_empty()
        && st_c.embedded_now == new_keys.len()
        && new_files.iter().all(|f| f.ends_with("nmjson/src/pretty.rs"))
        && st_c.loaded + new_keys.len() == sub.chunks.len();
    let _ = writeln!(rep, "  test 2: {}", if pass2 { "PASS" } else { "FAIL" });
    drop(lane);
    save(&rep);

    // (3) a file that cannot be used: each is named, nothing is loaded, nothing panics.
    let model = nmapi::neural::embedder_model(&home()).expect("embedder");
    let header = nmapi::neural::dense_header(&model);
    let good = std::fs::read(&store_file).expect("store file");
    let probe = |bytes: &[u8]| {
        std::fs::write(&store_file, bytes).expect("write probe");
        let l = DenseLane::in_ram(Arc::new(|_: &str| None), Arc::new(|_: &str| None));
        let line = l.keep_on_disk(&mem.join(store::DIR), header.clone(), &sub);
        (line, l.len())
    };
    let w = header.get("weights").cloned().unwrap_or_default();
    let flipped = format!("weights={}{}", if w.starts_with('0') { '1' } else { '0' }, w.get(1..).unwrap_or(""));
    let text = String::from_utf8_lossy(&good).into_owned();
    let pos = text.find(&format!("weights={w}")).expect("weights in the header");
    let mut mismatch = good.clone();
    mismatch.splice(pos..pos + flipped.len(), flipped.bytes());
    let cases = [
        ("header mismatch", mismatch, "header mismatch: weights"),
        ("cut-off file", good.get(..good.len() - 100).unwrap_or_default().to_vec(), "cut off"),
        ("another kind of file", b"not a store at all".to_vec(), "not a dense store file"),
    ];
    let mut pass3 = true;
    for (name, bytes, want) in cases {
        let (line, n) = probe(&bytes);
        let ok = line.contains(want) && n == 0;
        pass3 &= ok;
        let _ = writeln!(rep, "(3) {name}: {n} vectors loaded; {line}");
    }
    std::fs::write(&store_file, &good).expect("restore store");
    let (line, n) = probe(&good);
    pass3 &= n == sub.chunks.len();
    let _ = writeln!(rep, "(3) restored file: {n} vectors loaded; {line}\n  test 3: {}", if pass3 { "PASS" } else { "FAIL" });
    drop((sub, org));
    save(&rep);

    // (4) memory off.
    let (org, note) = session(&work, &mem_off, false);
    let (_, lane) = ready(&org);
    let st_d = lane.stats.lock().expect("stats").clone();
    let written = files_under(&mem_off);
    let pass4 = !mem_off.join(store::DIR).exists() && !lane.keeps_on_disk() && st_d.embedded_now > 0;
    let _ = writeln!(
        rep,
        "(4) memory off: embedded {} in RAM; files under the memory folder afterwards {written:?}\n  {note}\n  test 4: {}",
        st_d.embedded_now,
        if pass4 { "PASS" } else { "FAIL" }
    );
    drop((lane, org));
    let _ = writeln!(rep, "store folder after the tests: {:?}", files_under(&mem.join(store::DIR)));
    save(&rep);
    eprintln!("{rep}\nrecord {}", rec.display());
    assert!(pass1 && pass2 && pass3 && pass4, "a kept-index test failed");
}

/// One development question: its text and the line ranges that answer it.
struct Dev {
    id: String,
    question: String,
    answers: Vec<(String, usize, usize)>,
}

fn dev_questions(root: &Path) -> Vec<Dev> {
    let v = nmjson::parse(&std::fs::read(home().join("tests/dev/questions.json")).expect("questions")).expect("json");
    let mut out = Vec::new();
    for q in v.get("questions").and_then(nmjson::Value::array).expect("list") {
        let s = |k: &str| q.get(k).and_then(nmjson::Value::string).unwrap_or_default().to_owned();
        let mut answers = Vec::new();
        for a in q.get("answers").and_then(nmjson::Value::array).unwrap_or_default() {
            let file = a.get("file").and_then(nmjson::Value::string).unwrap_or_default().to_owned();
            let anchor = a.get("anchor").and_then(nmjson::Value::string).unwrap_or_default();
            let lines: Vec<usize> = a
                .get("lines")
                .and_then(nmjson::Value::array)
                .map(|l| l.iter().filter_map(|x| x.integer().ok().and_then(|n| usize::try_from(n).ok())).collect())
                .unwrap_or_default();
            let span = match lines.as_slice() {
                [x, y] if y >= x => y - x,
                _ => 0,
            };
            let text = std::fs::read_to_string(root.join(&file)).unwrap_or_default();
            if let Some(i) = text.lines().position(|l| l.contains(anchor)) {
                answers.push((file, i + 1, i + 1 + span));
            }
        }
        out.push(Dev { id: s("id"), question: s("question"), answers });
    }
    out
}

/// The chunk answers the question: it lies in an answer's file and overlaps its lines.
fn answers(sub: &nmsubs::Substrate, i: usize, d: &Dev) -> bool {
    let Some(c) = sub.chunks.get(i) else { return false };
    let Some(f) = sub.files.get(c.file) else { return false };
    let path = f.path.replace('\\', "/");
    d.answers.iter().any(|(file, a, b)| path == *file && c.start <= *b && c.end >= *a)
}

#[test]
#[ignore = "a full index of the frozen tree on the graphics card (about 20 minutes)"]
#[expect(clippy::too_many_lines, reason = "one paired run with its record")]
fn kept_frozen() {
    serve_switches();
    let root = nmsubs::repo::canon(&home().join("scratch").join("frozen-p2"));
    let dir = home().join("scratch").join("dense-test").join("frozen-mem").join(store::DIR);
    let _ = store::clear(&dir);
    let rec = home().join("runs").join(format!("dense-disk-frozen-{}.txt", nmtype::now_ms()));
    let mut rep = String::from("dense index kept on disk: stored against fresh vectors on scratch/frozen-p2\n");
    let save = |r: &str| std::fs::write(&rec, r).expect("record");
    let sub = nmsubs::Substrate::scan(&root, &["neuromorph-nano".to_owned()], Arc::new(|_: &str| Vec::new()));
    let t = Instant::now();
    let m = nmapi::neural::embedder_model(&home()).expect("embedder");
    let dev = if m.pop.gpu.is_some() { nmengn::placement::Device::Gpu } else { nmengn::placement::Device::Cpu };
    let header = nmapi::neural::dense_header(&m);
    let _ = writeln!(rep, "{} chunks; embedder bound on {} in {:.1} s; header {header:?}", sub.chunks.len(), dev.name(), t.elapsed().as_secs_f64());
    save(&rep);
    // The embedder outputs at binary32, kept by text, so the fresh lane needs no second index.
    let raw: Arc<Mutex<BTreeMap<String, Vec<f32>>>> = Arc::new(Mutex::new(BTreeMap::new()));
    let (m1, r1) = (Arc::clone(&m), Arc::clone(&raw));
    let embed = move |texts: &[String]| -> Option<Vec<Vec<f32>>> {
        let cap = nmengn::roles::input_cap(&m1, nmengn::placement::Device::Gpu, 1);
        let limit = nmengn::roles::batch_tokens(&m1, cap + 1);
        let mut out = Vec::with_capacity(texts.len());
        for part in texts.chunks(nmapi::neural::DENSE_PART) {
            out.extend(nmengn::roles::card_embed_batch(&m1, part, cap, limit).ok()?);
        }
        if let Ok(mut r) = r1.lock() {
            for (t, v) in texts.iter().zip(&out) {
                r.insert(t.clone(), v.clone());
            }
        }
        Some(out)
    };
    let lane = |m: &Arc<nmengn::model::Model>| {
        let (a, b) = (Arc::clone(m), Arc::clone(m));
        DenseLane::in_ram(
            Arc::new(move |s: &str| nmengn::roles::card_embed_on(&a, dev, s).ok()),
            Arc::new(move |s: &str| nmengn::roles::card_embed_on(&b, dev, s).ok()),
        )
    };
    // Before: an empty store, the whole tree embedded.
    let stored = lane(&m);
    let t = Instant::now();
    let note = stored.keep_on_disk(&dir, header.clone(), &sub);
    let n_full = stored.fill_batched(&sub, &embed, nmapi::neural::DENSE_BATCH, &|| true) + stored.fill(&sub, &|| true);
    let before_s = t.elapsed().as_secs_f64();
    let size = std::fs::metadata(dir.join(store::FILE)).map_or(0, |x| x.len());
    let _ = writeln!(rep, "before (no stored vectors): {n_full} chunks embedded, ready {before_s:.1} s after start; {note}\nstore file {size} bytes ({:.2} MiB)", size as f64 / 1_048_576.0);
    save(&rep);
    // Fresh: the same embedder outputs at binary32, no store.
    let fresh = lane(&m);
    let r2 = Arc::clone(&raw);
    let replay = move |texts: &[String]| -> Option<Vec<Vec<f32>>> { r2.lock().ok().and_then(|r| texts.iter().map(|t| r.get(t).cloned()).collect()) };
    fresh.fill_batched(&sub, &replay, nmapi::neural::DENSE_BATCH, &|| true);
    // After: a restart that loads the store.
    let restarted = lane(&m);
    let t = Instant::now();
    let note = restarted.keep_on_disk(&dir, header, &sub);
    let n_after = restarted.fill_batched(&sub, &embed, nmapi::neural::DENSE_BATCH, &|| true);
    let after_s = t.elapsed().as_secs_f64();
    let _ = writeln!(
        rep,
        "after (restart with the store): {n_after} chunks embedded, ready {} in {after_s:.3} s; {note}\nfresh lane ready {}",
        restarted.ready(&sub),
        fresh.ready(&sub)
    );
    save(&rep);
    // The 40 development questions: nearest chunks, fused order, right lines among the fused places.
    let qs = dev_questions(&root);
    let (mut same6, mut same20, mut same_fused, mut hit_f, mut hit_r, mut first_f, mut first_r) = (0, 0, 0, 0, 0, 0, 0);
    let mut lines = String::new();
    for d in &qs {
        let ef = fresh.embedding_order(&sub, &d.question).unwrap_or_default();
        let er = restarted.embedding_order(&sub, &d.question).unwrap_or_default();
        let of = fresh.order(&sub, &d.question, nmsubs::dense::TOP).unwrap_or_default();
        let or = restarted.order(&sub, &d.question, nmsubs::dense::TOP).unwrap_or_default();
        same6 += usize::from(ef.get(..6) == er.get(..6));
        same20 += usize::from(ef.get(..20) == er.get(..20));
        same_fused += usize::from(of == or);
        let (hf, hr) = (of.iter().any(|&i| answers(&sub, i, d)), or.iter().any(|&i| answers(&sub, i, d)));
        hit_f += usize::from(hf);
        hit_r += usize::from(hr);
        first_f += usize::from(of.first().is_some_and(|&i| answers(&sub, i, d)));
        first_r += usize::from(or.first().is_some_and(|&i| answers(&sub, i, d)));
        if of != or || hf != hr {
            let _ = writeln!(lines, "  {}: fused order differs (fresh {of:?}, stored {or:?}); right lines among them {hf} and {hr}", d.id);
        }
    }
    let n = qs.len();
    let _ = writeln!(
        rep,
        "{n} development questions, fresh (binary32) against stored (binary16):\n  same 6 nearest chunks {same6} of {n}; same 20 nearest {same20} of {n}; same fused places {same_fused} of {n}\n  right lines first among the fused places: fresh {first_f}, stored {first_r}\n  right lines among the {} fused places: fresh {hit_f}, stored {hit_r}\n{lines}",
        nmsubs::dense::TOP
    );
    save(&rep);
    eprintln!("{rep}\nrecord {}", rec.display());
    assert_eq!((n_after, hit_f, first_f), (0, hit_r, first_r), "stored vectors changed the counts or a restart embedded chunks");
}
