//! G3 and muscle memory on the served call (`nmapi::g3`, `serve --muscle on`):
//! a scripted replay of 20 verified calls promotes one muscle-memory pack, and the next call of
//! that kind uses it.
//!
//! The calls are real `nm_context_pack` calls on the small fixture repository, processor only.
//! Verdicts are scripted (no ledger in a test): name requests verify better with the `uniform`
//! allocation, explain requests worse. The cycle replays each verified call
//! through the served call with the other modes; the script scores each replay.
use nmapi::forage::Organism;
use nmjson::Value;
use std::collections::BTreeMap;

const NAME: [&str; 10] = [
    "Where is take_from_shelf defined?",
    "Where is total_units defined?",
    "Where is unit_price_cents defined?",
    "Where is checkout defined?",
    "Where is StockItem defined?",
    "Which function takes items from the shelf?",
    "Which function counts the total units?",
    "Which function gives the unit price in cents?",
    "Which function runs the checkout?",
    "Which struct holds a stock item?",
];
const EXPLAIN: [&str; 10] = [
    "How does checkout compute the price?",
    "How does take_from_shelf decide it can take items?",
    "How does total_units add the stock?",
    "How does unit_price_cents pick a price?",
    "Why does take_from_shelf return false?",
    "Why does checkout return None?",
    "How does the shop crate use the inventory?",
    "Why does total_units return a u64?",
    "How does checkout change the stock item?",
    "Why does unit_price_cents take a code?",
];

/// The scripted verifier: the outcome of a call of `kind` run with `mode`, with a small spread.
/// `uniform` gains on name requests and loses on explain requests, so over all calls no mode leads
/// while G3 finds the gain of one kind; `off` loses on explain requests.
fn script(kind: &str, mode: &str, i: usize) -> f64 {
    let spread = |salt: usize| ((i * 37 + salt * 11) % 13) as f64 / 13.0 * 0.06 - 0.03;
    match (kind, mode) {
        ("name", "uniform") => 0.9 + spread(1),
        ("explain", "uniform") => 0.1 + spread(1),
        // Without the second round explain requests lose places (so `off`, the fastest, does not
        // win on time alone).
        ("explain", "off") => 0.3 + spread(2),
        _ => 0.5 + spread(0),
    }
}

fn input(q: &str) -> Value {
    Value::Object(BTreeMap::from([("question".to_owned(), Value::String(q.to_owned()))]))
}

#[test]
fn twenty_verified_calls_promote_one_pack_and_the_next_call_uses_it() {
    let home = nmtype::nano_root();
    let dir = nmtype::check_dir("nmapi-g3-service").expect("check folder");
    let root = home.join("tests").join("fixtures").join("small-repo");
    let org = Organism::open_with(&home, &root, &[], false, &dir.join("memory")).expect("organism opens");
    nmapi::tools::install(org).expect("install");
    let api = nmapi::tools::api();
    assert_eq!(nmapi::g3::enable(&home, Some(&dir.join("muscle"))).expect("muscle on"), 0, "an empty store loads no pack");

    // 20 served calls, each verified by the script as served (the configured mode, `nbv`).
    let mut calls: BTreeMap<String, (String, usize)> = BTreeMap::new();
    for (i, q) in NAME.iter().chain(EXPLAIN.iter()).enumerate() {
        api.dispatch("nm_context_pack", &input(q)).expect("served call");
        let d = nmapi::tools::take_diagnostics().expect("diagnostics");
        let cfg = d.get("config").unwrap_or_else(|_| panic!("no configuration record in the diagnostics: {}", d.encode().unwrap_or_default()));
        assert_eq!(cfg.get("allocation").and_then(Value::string).ok(), Some("nbv"));
        assert_eq!(cfg.get("by").and_then(Value::string).ok(), Some("rule"));
        let id = nmapi::g3::last_call().expect("recorded");
        let kind = nmapi::whole::kind_of(q).to_owned();
        nmapi::g3::verdict(&id, script(&kind, "nbv", i)).expect("verdict");
        calls.insert(id, (kind, i));
    }
    assert_eq!(calls.values().filter(|(k, _)| k == "name").count(), 10, "ten name requests by wording");
    assert_eq!(calls.values().filter(|(k, _)| k == "explain").count(), 10, "ten explain requests by wording");

    // The idle cycle: paired replays through the served call, scored by the script (time enters
    // as a cost below the script's spread).
    let mut ran = 0usize;
    let mut score = |id: &str, choices: &[(nmmusc::priors::Site, String)], _: &str| {
        let (kind, i) = calls.get(id)?.clone();
        let mode = choices.iter().find(|(s, _)| *s == nmmusc::priors::Site::Allocation).map(|(_, m)| m.clone())?;
        ran += 1;
        Some((script(&kind, &mode, i), 1))
    };
    let cy = nmapi::g3::cycle(&mut score).expect("cycle");
    eprintln!("cycle: {cy:?}");
    assert!(cy.replays > 0 && cy.replays <= 40, "G3's halving replays calls through the served call ({})", cy.replays);
    assert_eq!(cy.replays, ran, "the cycle reports the replays it scored");
    assert_eq!(cy.promoted.len(), 1, "one pack promoted: {cy:?}");
    let (fp, desc) = cy.promoted.first().cloned().expect("pack");
    assert!(desc.starts_with("name requests in area ") && desc.contains("g1.allocation uniform in place of nbv"), "{desc}");

    // The next call of that kind uses the pack; a call of the other kind does not.
    api.dispatch("nm_context_pack", &input("Where is the StockItem struct defined?")).expect("next call");
    let d = nmapi::tools::take_diagnostics().expect("diagnostics");
    let cfg = d.get("config").expect("configuration record");
    assert_eq!(cfg.get("allocation").and_then(Value::string).ok(), Some("uniform"));
    assert_eq!(cfg.get("by").and_then(Value::string).ok(), Some(format!("pack:{fp}").as_str()));
    api.dispatch("nm_context_pack", &input("How does checkout use the stock?")).expect("other call");
    let d = nmapi::tools::take_diagnostics().expect("diagnostics");
    assert_eq!(d.get("config").and_then(|c| c.get("by")).and_then(Value::string).ok(), Some("rule"));

    // The pack is in muscle memory's store, behaviour only, and a new session reloads it.
    let store = nmmusc::MuscleStore::open(&dir.join("muscle")).expect("store");
    assert_eq!(store.index().expect("index").len(), 1);
    let _ = nmapi::g3::disable();
    assert_eq!(nmapi::g3::enable(&home, Some(&dir.join("muscle"))).expect("muscle on again"), 1, "the pack reloads");
    let _ = nmapi::g3::disable();

    // A sensory item is refused by the store: the same pack file tagged sensory does not load.
    let file = std::fs::read_dir(dir.join("muscle"))
        .expect("store folder")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with(fp.as_str())))
        .expect("the pack's file");
    let bytes = std::fs::read(&file).expect("pack file");
    let (from, to) = (b"\"data_class\":\"behavioural\"", b"\"data_class\":\"sensory\"    ");
    let at = bytes.windows(from.len()).position(|w| w == from).expect("data class in the header");
    let mut tagged = bytes.clone();
    tagged.splice(at..at + from.len(), to.iter().copied());
    std::fs::write(&file, &tagged).expect("write tagged");
    let base = nmapi::g3::base_hash(&home);
    let refused = store.load(&fp, &base).err().map(|e| e.code);
    assert_eq!(refused, Some("nmmusc-sensory-refused"), "the store refuses a sensory-tagged pack");
    assert_eq!(nmapi::g3::enable(&home, Some(&dir.join("muscle"))).expect("muscle on"), 0, "G3 loads no sensory pack");
    let _ = nmapi::g3::disable();
    std::fs::write(&file, &bytes).expect("restore");
}
