//! The search lane's vectors held in RAM only (2026-11-10, refinement step 1, item 7).
//! Gavriel Technologies; architecture by Luis Jake Gabriel III.
//!
//! Short isolated test of `docs/2026-11-10-neumor-analysis-g-vectors-and-g1-placement.md`
//! question 1, section h: one variable (where the vectors are held). A smoke run of 5 chunks, then
//! about 200 chunks of the mini's own tree (chunks whose content key the old store file holds) are
//! embedded twice into RAM-only lanes in this one process: the two builds must be bit-identical,
//! their vectors are compared with the old file's records for the same keys (the file is read,
//! never written), and the meaning ranking from RAM must equal the ranking from the file on a few
//! queries written here (no locked test question is used). Nothing encoded is written to disk;
//! the record holds numbers only.
//!
//! Run: `cargo test --release -p nmapi --test dense_ram -- --ignored --nocapture`.
//! Writes `runs/D7-dense-ram-<ms>.txt`.
use nmengn::model::{Model, Population};
use nmsubs::dense::DenseLane;
use std::fmt::Write as _;
use std::sync::Arc;
use std::time::Instant;

/// Queries for the ranking comparison (written for this test).
const QUERIES: [&str; 4] = [
    "where are the tentacle windows laid out for a question",
    "how does the device gate let foreground jobs go first",
    "where is the scan dump written for replay",
    "which function reduces packets to a cited pack",
];
/// Chunks of the main build (about 200, as the research note's test asks) and of the smoke run.
const CHUNKS: usize = 200;
const SMOKE: usize = 5;

#[test]
#[ignore = "embeds on the graphics card (run on demand, a few minutes)"]
#[expect(clippy::too_many_lines, reason = "one short test with its record")]
fn dense_lane_in_ram() {
    let home = nmtype::nano_root();
    let started = Instant::now();
    let mut rep = String::from("D7 dense lane held in RAM only (refinement step 1, item 7)\n");
    let rec_path = home.join("runs").join(format!("D7-dense-ram-{}.txt", nmtype::now_ms()));
    let write = |r: &str| std::fs::write(&rec_path, r).expect("record");
    let sub = nmsubs::Substrate::scan(&home, &[], Arc::new(|_: &str| Vec::new()));
    // The old store file: read only.
    let dir = home.join("scratch").join("dense");
    let file = std::fs::read_dir(&dir)
        .ok()
        .and_then(|rd| rd.filter_map(Result::ok).map(|e| e.path()).find(|p| p.extension().is_some_and(|x| x == "vec")));
    // Read only: the document function returns nothing, so no record is ever embedded or appended;
    // the question side is set after the embedder opens (`file_query`).
    let file_query: Arc<std::sync::OnceLock<Arc<Model>>> = Arc::new(std::sync::OnceLock::new());
    let fq = Arc::clone(&file_query);
    let file_lane = file.as_ref().map(|f| {
        DenseLane::open(
            Some(f.clone()),
            Arc::new(|_: &str| None),
            Arc::new(move |s: &str| {
                fq.get().and_then(|m| {
                    nmengn::roles::card_embed_on(
                        m,
                        if m.pop.gpu.is_some() { nmengn::placement::Device::Gpu } else { nmengn::placement::Device::Cpu },
                        s,
                    )
                    .ok()
                })
            }),
        )
    });
    let _ = writeln!(
        rep,
        "old store file (read only, never written): {} ({} bytes, {} records)",
        file.as_ref().map_or_else(|| "none".to_owned(), |f| f.display().to_string()),
        file.as_ref().and_then(|f| std::fs::metadata(f).ok()).map_or(0, |m| m.len()),
        file_lane.as_ref().map_or(0, DenseLane::len)
    );
    // The test tree: the first chunks whose content key the file holds (so every vector compares).
    let keep: Vec<usize> = sub
        .chunks
        .iter()
        .enumerate()
        .filter(|(_, c)| file_lane.as_ref().is_none_or(|l| l.vector(&c.key).is_some()))
        .map(|(i, _)| i)
        .take(CHUNKS)
        .collect();
    let small = |n: usize| {
        let mut s = sub.clone();
        s.chunks = keep.iter().take(n).filter_map(|i| sub.chunks.get(*i).cloned()).collect();
        s
    };
    let (smoke, test) = (small(SMOKE), small(CHUNKS));
    // One embedder, one process, on the graphics card when one opens.
    let t = Instant::now();
    let pop = Population::open(&home, "qwen3-embedding-0.6b", true, true).expect("embedder population");
    let dev = if pop.gpu.is_some() { nmengn::placement::Device::Gpu } else { nmengn::placement::Device::Cpu };
    let m = Arc::new(
        Model::bind(&pop, nmengn::Assembly::load(&home.join("designs/assemblies"), "qwen3-embedding").expect("assembly")).expect("bind"),
    );
    let _ = file_query.set(Arc::clone(&m));
    let _ = writeln!(rep, "embedder opened on {} in {:.1} s", dev.name(), t.elapsed().as_secs_f64());
    let lane = || {
        let (m1, m2) = (Arc::clone(&m), Arc::clone(&m));
        DenseLane::in_ram(
            Arc::new(move |s: &str| nmengn::roles::card_embed_on(&m1, dev, s).ok()),
            Arc::new(move |s: &str| nmengn::roles::card_embed_on(&m2, dev, s).ok()),
        )
    };
    // Smoke run.
    let s0 = lane();
    let t = Instant::now();
    let n0 = s0.fill(&smoke, &|| true);
    let _ = writeln!(rep, "smoke: {n0} chunks in {:.1} s, ready {}", t.elapsed().as_secs_f64(), s0.ready(&smoke));
    write(&rep);
    assert_eq!(n0, smoke.chunks.len(), "smoke run embedded every chunk");
    // Two builds of the test set.
    let a = lane();
    let not_ready = a.embedding_order(&test, QUERIES[0]).is_none();
    let t = Instant::now();
    let na = a.fill(&test, &|| true);
    let sa = t.elapsed().as_secs_f64();
    let _ = writeln!(
        rep,
        "build 1: {na} chunks in {sa:.1} s ({:.3} s per chunk); {} bytes of vectors in RAM; ranking withheld before full coverage: {not_ready}",
        sa / na.max(1) as f64,
        a.bytes()
    );
    write(&rep);
    let b = lane();
    let t = Instant::now();
    let nb = b.fill(&test, &|| true);
    let sb = t.elapsed().as_secs_f64();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    let same_ab = test.chunks.iter().all(|c| a.vector(&c.key).zip(b.vector(&c.key)).is_some_and(|(x, y)| bits(&x) == bits(&y)));
    let _ = writeln!(
        rep,
        "build 2: {nb} chunks in {sb:.1} s ({:.3} s per chunk); bit-identical to build 1 on every chunk: {same_ab}",
        sb / nb.max(1) as f64
    );
    write(&rep);
    // Against the old file.
    let (mut same_file, mut max_diff) = (0, 0.0_f32);
    if let Some(fl) = file_lane.as_ref() {
        for c in &test.chunks {
            if let (Some(x), Some(y)) = (a.vector(&c.key), fl.vector(&c.key)) {
                same_file += usize::from(bits(&x) == bits(&y));
                max_diff = x.iter().zip(y.iter()).map(|(p, q)| (p - q).abs()).fold(max_diff, f32::max);
            }
        }
    }
    let _ = writeln!(
        rep,
        "vectors equal bit for bit to the old file's records: {same_file} of {}; largest absolute difference {max_diff:e}",
        test.chunks.len()
    );
    let mut rank_equal = 0;
    for (i, q) in QUERIES.iter().enumerate() {
        let ra = a.embedding_order(&test, q).unwrap_or_default();
        let rf = file_lane.as_ref().and_then(|l| l.embedding_order(&test, q)).unwrap_or_default();
        let eq = !ra.is_empty() && ra == rf;
        rank_equal += usize::from(eq);
        let first_diff = ra.iter().zip(&rf).position(|(x, y)| x != y);
        let _ = writeln!(
            rep,
            "query {} (written for this test): ranking from RAM equals ranking from the file: {eq} (first difference at place {first_diff:?}; {} chunks ranked)",
            i + 1,
            ra.len()
        );
    }
    let st = a.stats.lock().map(|s| s.clone()).unwrap_or_default();
    let _ = writeln!(
        rep,
        "lane statistics: ready {}, covered {} of {}, not-ready queries {}, embedded {} in {:.1} s of embedding, full coverage after {:.1} s",
        st.ready,
        st.covered,
        st.total,
        st.not_ready,
        st.embedded_now,
        st.index_ms / 1000.0,
        st.full_ms.unwrap_or(0.0) / 1000.0
    );
    let _ = writeln!(
        rep,
        "files under scratch/dense after the test: {:?} (nothing written by the test)",
        std::fs::read_dir(&dir)
            .map(|rd| rd
                .filter_map(Result::ok)
                .map(|e| (e.file_name().to_string_lossy().into_owned(), e.metadata().map_or(0, |m| m.len())))
                .collect::<Vec<_>>())
            .unwrap_or_default()
    );
    let _ = writeln!(rep, "total {:.1} s", started.elapsed().as_secs_f64());
    write(&rep);
    eprintln!("{rep}\nrecord {}", rec_path.display());
    assert!(same_ab, "two RAM builds differ");
    assert!(not_ready, "the lane ranked before full coverage");
    assert_eq!(rank_equal, QUERIES.len(), "a ranking from RAM differs from the file's");
}
