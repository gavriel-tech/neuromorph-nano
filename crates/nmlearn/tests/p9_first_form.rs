//! Step P9 first form: runs the `cartridge` and `preference` section checks and prints the
//! first measurements (leave-one-out accuracy pair, time per selection, time per recall).
//! `cargo test --release -p nmlearn --test p9_first_form -- --nocapture`
use nmlearn::cartridge::{DataClass, Kind, Label, Loaded, Policy, Request, select, units_as_cartridges};
use nmlearn::preference::{ExactEmbedder, Index, PrefRecord, Prefs, leave_one_out, recall, working_rule_cases};
use std::collections::BTreeMap;
use std::time::Instant;

#[test]
fn p9_checks_and_measurements() {
    for (name, check) in nmlearn::checks().into_iter().filter(|(n, _)| *n == "nmlearn.cartridge" || *n == "nmlearn.preference") {
        assert_eq!(check(), Ok(()), "{name}");
        println!("check {name}: ok");
    }
    let root = nmtype::nano_root();
    let policy = Policy::load(&root);
    let units = nmlearn::Units::open(&root);
    let mut labels = units_as_cartridges(&units, &policy);
    let live = labels.len();
    let loaded = Loaded::population(&root, "qwen3-0.6b", vec!["nmengn.components#*".into()]).expect("population");
    // The selection is measured over a live unit's label when one exists; a folder with no
    // learned unit yet (a fresh install) measures over a synthetic label of the same shape.
    let proto = labels.first().cloned().unwrap_or_else(|| Label {
        id: "synthetic-proto".into(),
        kind: Kind::LearningUnit,
        data_class: DataClass::Behavioural,
        base: loaded.base.clone(),
        base_sha256: loaded.base_sha256.clone(),
        components: vec!["nmengn.components#linear".into()],
        scopes: vec!["*".into()],
        quantization: "nmq8-row-f16".into(),
        description: "synthetic label for the selection measurement".into(),
        version: 1,
        check: "accept".into(),
        date: "2026-11-15".into(),
        loaded: true,
    });
    if live == 0 {
        println!("no learned unit in units/: selection measured over synthetic labels only");
    }
    for i in 0..64 {
        let mut l = proto.clone();
        l.id = format!("synthetic-{i:02}");
        l.scopes = vec![["reranker:where-fn", "reranker:value", "router", "*"][i % 4].to_owned()];
        labels.push(l);
    }
    let req = Request { scope: "reranker:where-fn:neuromorph-nano".into(), data_class: DataClass::Behavioural };
    let tracks = BTreeMap::new();
    let reps = 2000_u32;
    let t = Instant::now();
    let mut active = 0;
    for _ in 0..reps {
        active = select(&req, &labels, &loaded, &tracks, &policy).active.len();
    }
    let per_sel = t.elapsed().as_secs_f64() * 1e6 / f64::from(reps);
    println!(
        "selection: {} cartridges ({live} live units + 64 synthetic), {active} active, {per_sel:.2} us per selection (mean of {reps}); policy from {}",
        labels.len(),
        policy.source
    );

    let cases = working_rule_cases();
    let t = Instant::now();
    let loo = leave_one_out(&cases, &ExactEmbedder);
    let loo_ms = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "leave-one-out on {} constructed cases: recall {} of {} ({:.1}%), always the more common answer {} of {} ({:.1}%), ties {}; whole run {loo_ms:.2} ms",
        loo.n,
        loo.recall_right,
        loo.n,
        100.0 * loo.recall_right as f64 / loo.n as f64,
        loo.majority_right,
        loo.n,
        100.0 * loo.majority_right as f64 / loo.n as f64,
        loo.ties
    );
    let prefs = Prefs::in_memory(cases.iter().filter_map(|c| PrefRecord::from_feedback(c, 0).ok()).collect());
    let t = Instant::now();
    let idx = Index::build(&prefs, &ExactEmbedder);
    let build_us = t.elapsed().as_secs_f64() * 1e6;
    let q = &cases[5].circumstance;
    let t = Instant::now();
    let mut got = 0;
    for _ in 0..reps {
        got = recall(&idx, q, &ExactEmbedder, None, None).len();
    }
    let per_recall = t.elapsed().as_secs_f64() * 1e6 / f64::from(reps);
    println!(
        "recall: {} records, k = {got}, index build {build_us:.1} us, {per_recall:.2} us per recall (mean of {reps})",
        prefs.records.len()
    );
}
