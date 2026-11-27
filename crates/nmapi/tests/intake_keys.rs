//! Phase C2 tests of the tentacle master plan (2026-11-18): the relevance-key intake pass
//! (`nmapi::key_scan::Keys::intake`, pieces encoded alone in segments through the deepest relevance
//! layer at the lowest class). Needs the release weights; uses the card when present, so it runs by
//! hand under `scratch/LOCKS/card.lock`:
//! `cargo test --release -p nmapi --test intake_keys -- --ignored --nocapture --test-threads 1`.
//! - segments (T-069): keys encoded in segments equal the whole-piece encode within tolerance;
//! - store bytes per token (equals the drone set's packet width: pairs x head width x 2);
//! - PA-T's resident part: process working set up, beyond the store's own bytes, under 5 percent;
//! - the gate: a question entering the device gate while the pass runs waits no longer than the
//!   longest single kernel (one segment);
//! - order: the touched pieces are encoded first; signals and local memory kept per piece; a
//!   changed piece counts one change and a later version re-encodes only it.
use nmapi::key_scan::{Keys, intake_order, intake_pieces};
use std::sync::Arc;
use std::time::Instant;

fn material(changed: bool) -> Vec<(String, String)> {
    let root = nmtype::nano_root().join("crates/nmengn/src");
    let mut files = Vec::new();
    for name in ["drone.rs", "half.rs", "residency.rs"] {
        if let Ok(mut t) = std::fs::read_to_string(root.join(name)) {
            if changed && name == "half.rs" {
                t.insert_str(0, "// changed for the C2 test\n");
            }
            files.push((format!("crates/nmengn/src/{name}"), t));
        }
    }
    files
}

fn substrate(files: &[(String, String)]) -> nmsubs::Substrate {
    let borrowed: Vec<(&str, &str)> = files.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let embed: nmsubs::Embed = Arc::new(|_: &str| vec![0.0]);
    nmsubs::from_files(&borrowed, embed)
}

#[test]
#[ignore = "Phase C2: needs the release weights and the card; run by hand under the card lock"]
#[expect(clippy::too_many_lines, reason = "one test, in the order it runs")]
fn c2_intake() {
    let home = nmtype::nano_root();
    let sub = substrate(&material(false));
    let model = nmapi::neural::model(&home).expect("model");
    let dev = if model.pop.gpu.is_some() { nmengn::Device::Gpu } else { nmengn::Device::Cpu };
    let sets = nmapi::tentacle_pass::drones(&home);
    let d = model.pop.dims;
    let group = d.heads / d.kv_heads.max(1);
    let mut keep = nmapi::drone_windows::Keep::open(&nmtype::models_dir(&home).join("qwen3-0.6b/files")).expect("keep");
    keep.sync(&sub).expect("sync");
    let before = nmffi::dynlib::process_memory().map(|m| m.1).unwrap_or(0);
    let keys = Arc::new(Keys::new_alone(&sets.heads, group, d.head_dim, 1 << 30));

    // (a) Segments: three pieces, keys encoded in segments of 64 against the whole piece.
    let mut worst = 0.0_f32;
    for c in [0, 3, 7] {
        let Some(ch) = sub.chunks.get(c) else { continue };
        let f = sub.files.get(ch.file).expect("file");
        let ids = model.pop.tokenizer.encode(&nmsubs::dense::doc_text(&f.path, &ch.text)).expect("tokens");
        let whole = keys.encode_alone(&model, dev, &ids).expect("whole");
        let (seg, times) = keys.encode_segmented(&model, dev, &ids, 64, &mut || None).expect("segments");
        assert_eq!(whole.rows.len(), seg.rows.len());
        for (a, b) in whole.rows.iter().zip(&seg.rows) {
            for (x, y) in a.iter().zip(b) {
                let (x, y) = (nmengn::cpu::wide(*x), nmengn::cpu::wide(*y));
                worst = worst.max((x - y).abs() / x.abs().max(1.0));
            }
        }
        println!("piece {c}: {} tokens, {} segments, {:.1} ms each at most", ids.len(), times.len(), times.iter().copied().fold(0.0_f64, f64::max));
    }

    // (b) The pass, with questions at the gate meanwhile.
    let touched = [sub.chunks.len().saturating_sub(1), 0];
    let order = intake_order(&sub, &keys, &touched);
    let pieces = intake_pieces(&sub, Some(&keep), &order);
    let order_ok = order.first() == touched.first() && order.get(1) == touched.get(1) && order.len() == sub.chunks.len();
    let t = Instant::now();
    assert!(keys.start_intake(Arc::clone(&model), dev, pieces.clone(), 1), "intake did not start");
    let mut waits: Vec<f64> = Vec::new();
    while keys.status().running {
        if dev == nmengn::Device::Gpu {
            let g = nmengn::placement::gpu_gate().enter("C2 test question", true);
            waits.push(g.waited * 1000.0);
            drop(g);
        }
        std::thread::park_timeout(std::time::Duration::from_millis(30));
    }
    let wall = t.elapsed().as_secs_f64();
    let st = keys.status();
    let (bytes, packets, _) = keys.held();
    let after = nmffi::dynlib::process_memory().map(|m| m.1).unwrap_or(0);
    let bpt = keys.bytes_per_token_held();
    let grown = after.saturating_sub(before).saturating_sub(bytes);
    #[expect(clippy::cast_precision_loss, reason = "a ratio for the record")]
    let grown_pct = grown as f64 / before.max(1) as f64 * 100.0;
    let wait_max = waits.iter().copied().fold(0.0_f64, f64::max);
    let first = pieces.first().map(|p| p.identity.clone()).unwrap_or_default();
    let mem = keys.piece(&first);

    // (c) A changed version: one piece's content changes; only it is re-encoded; it counts a change.
    let sub2 = substrate(&material(true));
    keep.sync(&sub2).expect("sync 2");
    let order2 = intake_order(&sub2, &keys, &[]);
    let pieces2 = intake_pieces(&sub2, Some(&keep), &order2);
    keys.intake(&model, dev, &pieces2, 2);
    let st2 = keys.status();
    let changed: Vec<_> = pieces2.iter().filter_map(|p| keys.piece(&p.identity)).filter(|m| m.changes > 0).collect();
    println!(
        "C2 intake on {}: segments of a piece equal the whole-piece keys within {worst:.2e} relative; {} pieces, {} encoded ({} tokens) in {wall:.2} s as {} segments, {} skipped; longest kernel {:.1} ms; store {packets} packets, {bytes} bytes, {bpt:.0} bytes per token (packet width {}); working set beyond the store {grown_pct:.2} percent; {} questions at the gate, longest wait {wait_max:.1} ms; touched first: {order_ok}; first piece's memory {mem:?}; after a change: {} encoded, {} held, {} pieces with a change; faults {:?}",
        dev.name(),
        st.chunks,
        st.encoded,
        st.tokens,
        st.segments,
        st.skipped,
        st.kernel_max_ms,
        keys.bytes_per_token,
        waits.len(),
        st2.encoded,
        st2.held,
        changed.len(),
        st.faults
    );
    assert!(worst < 2e-2, "segmented keys differ from the whole piece by {worst}");
    assert!(order_ok, "touched pieces not first");
    assert_eq!(st.encoded + st.skipped + st.held, st.chunks, "every piece encoded, skipped or held");
    #[expect(clippy::cast_precision_loss, reason = "bytes per token")]
    let width = keys.bytes_per_token as f64;
    assert!((bpt - width).abs() < 1e-6, "store bytes per token {bpt} against the packet width {width}");
    assert!(grown_pct < 5.0, "PA-T resident part: working set up {grown_pct:.2} percent beyond the store");
    assert!(wait_max <= st.kernel_max_ms + 5.0, "a question waited {wait_max:.1} ms, longer than one kernel ({:.1} ms)", st.kernel_max_ms);
    assert!(mem.as_ref().is_some_and(|m| m.version == 1 && m.signals.tokens > 0 && !m.signals.kind.is_empty()), "signals and memory not kept");
    assert!(st2.encoded >= 1 && st2.encoded < st.encoded && !changed.is_empty(), "a later version re-encoded held pieces or missed the change");
    assert!(!keys.start_intake(Arc::clone(&model), dev, pieces2, 2), "a second intake of the same version started");
}

/// DR-1 (master plan section 8): the drones' layers 0 to k (`drone_dendrite::Dendrite`, the
/// reference assembly cut after layer k on the shared population) equal the reference forward's
/// hidden state after layer k on the same piece alone, bit for bit: k 0 and 2 on the card (three
/// pieces), k 2 on the processor (one piece).
#[test]
#[ignore = "DR-1: needs the release weights and the card; run by hand under the card lock"]
fn dr1_drone_layers() {
    let home = nmtype::nano_root();
    let sub = substrate(&material(false));
    let reference = nmengn::Assembly::load(&home.join("designs/assemblies"), "qwen3-reference").expect("assembly");
    let mut lines = Vec::new();
    let mut all = true;
    for dev in [nmengn::Device::Gpu, nmengn::Device::Cpu] {
        let pop = nmengn::Population::open(&home, "qwen3-0.6b", true, dev == nmengn::Device::Gpu).expect("population");
        if dev == nmengn::Device::Gpu && pop.gpu.is_none() {
            lines.push("gpu: none, skipped".to_owned());
            continue;
        }
        let refm = nmengn::Model::bind(&pop, reference.clone()).expect("reference");
        let (ks, picks): (Vec<usize>, Vec<usize>) = if dev == nmengn::Device::Gpu { (vec![0, 2], vec![0, 3, 7]) } else { (vec![2], vec![3]) };
        for k in ks {
            let dendrite = nmapi::drone_dendrite::Dendrite::new(&pop, &reference, k, dev).expect("dendrite");
            for &p in &picks {
                let ch = sub.chunks.get(p).expect("chunk");
                let f = sub.files.get(ch.file).expect("file");
                let ids = pop.tokenizer.encode(&nmsubs::dense::doc_text(&f.path, &ch.text)).expect("tokens");
                let target = nmapi::drone_dendrite::output_node(k);
                let r = refm.forward(dev, &ids, std::slice::from_ref(&target), &nmengn::RunOpts::default()).expect("reference forward");
                let rv = nmengn::model::values(r.get(&target).expect("output")).expect("values");
                let d = dendrite.sense(&ids).expect("drone");
                let same = d.states.len() == rv.len() && d.states.iter().zip(rv).all(|(a, b)| a.to_bits() == b.to_bits());
                all &= same;
                lines.push(format!("{} k {k} piece {p} ({} tokens): bit-identical {same}", dev.name(), ids.len()));
            }
        }
        pop.release_device();
    }
    println!("DR-1: {}; {}", if all { "pass" } else { "FAIL" }, lines.join("; "));
    assert!(all, "DR-1 failed");
}
