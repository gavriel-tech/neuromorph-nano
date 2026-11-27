//! Step P6 first form (2026-11-10): the G0 and G1 section checks, the first measurement on the
//! constructed cases (loop against one-shot forwarding), and the size of the whole-code-base
//! layer for this repository (file tree and item signatures from `nmsubs`, tokens under the
//! base tokenizer). Processor only; no weights are loaded.
use nmbrain::g1::{self, ItemSig, WholeCodebase};
use std::sync::Arc;
use std::time::Instant;

#[test]
fn g0_g1_checks_pass() {
    for (id, f) in nmbrain::checks().into_iter().filter(|(id, _)| *id == "nmbrain.g0" || *id == "nmbrain.g1") {
        assert_eq!(f(), Ok(()), "{id}");
    }
}

#[test]
fn g1_measure_constructed_cases() {
    let m = g1::measure();
    println!("{}", nmjson::pretty::pretty(&m));
}

#[test]
fn whole_codebase_layer_size_on_this_repository() {
    let root = nmtype::nano_root();
    let t0 = Instant::now();
    let sub = nmsubs::Substrate::scan(&root, &[], Arc::new(|_: &str| Vec::new()));
    let scan_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t1 = Instant::now();
    let files: Vec<String> = sub.files.iter().map(|f| f.path.clone()).collect();
    let items: Vec<ItemSig> = sub
        .symbols
        .iter()
        .map(|s| ItemSig { name: s.name.clone(), kind: s.kind.clone(), file: s.file.clone(), line: s.line, signature: s.text.clone() })
        .collect();
    let layer = WholeCodebase::build(files, items);
    let build_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let t2 = Instant::now();
    let rel = layer.relevance(g1::CASE_QUESTION);
    let rel_ms = t2.elapsed().as_secs_f64() * 1000.0;
    let text = layer.render();
    let tok = nmtokz::Tokenizer::open(&nmtype::models_dir(&root).join("qwen3-0.6b/files/tokenizer.json"));
    let tokens = tok.as_ref().ok().and_then(|t| t.encode(&text).ok()).map(|v| v.len());
    let rs_files = layer.files.iter().filter(|f| f.ends_with(".rs")).count();
    println!(
        "whole-code-base layer: {} files ({} .rs), {} items, {} bytes in RAM, rendered {} bytes, {} tokens (base tokenizer){}; scan {scan_ms:.0} ms, build {build_ms:.1} ms, relevance {rel_ms:.1} ms, relevant files {}",
        layer.files.len(),
        rs_files,
        layer.items.len(),
        layer.bytes(),
        text.len(),
        tokens.map_or_else(|| "n/a".to_owned(), |t| t.to_string()),
        if tokens.is_none() { " (tokenizer not readable)" } else { "" },
        layer.ranked_files(&rel).len(),
    );
    assert!(!layer.files.is_empty() && !layer.items.is_empty());
}
