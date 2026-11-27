//! Co-citation prior at G1 (2026-11-12, memory design): a chunk co-cited with a seed receives
//! structure value even where the reference graph gives it none; with no links the value is the
//! reference graph's own.
use nmbrain::evidence::Structure;
use std::collections::BTreeMap;

#[test]
fn cocited_partner_receives_structure() {
    let root = nmtype::check_dir("nmapi-cocitation").expect("folder");
    std::fs::create_dir_all(root.join("src")).expect("src");
    std::fs::write(root.join("src/a.rs"), "pub fn alpha_one() -> u8 { 1 }\n").expect("a");
    std::fs::write(root.join("src/b.rs"), "pub fn beta_two() -> u8 { 2 }\n").expect("b");
    let mem = nmtype::check_dir("nmapi-cocitation-memory").expect("memory");
    let org = nmapi::forage::Organism::open_with(&nmtype::nano_root(), &root, &[], false, &mem).expect("open");
    let sub = org.sub();
    let key = |p: &str| {
        sub.chunks
            .iter()
            .find(|c| sub.files.get(c.file).is_some_and(|f| f.path.replace('\\', "/") == p))
            .map(|c| c.key.clone())
            .expect("chunk")
    };
    let (a, b) = (key("src/a.rs"), key("src/b.rs"));
    let base = nmapi::tentacle_core::SubGraph(sub.refs());
    let plain = base.rank(&[(a.clone(), 1.0)], std::slice::from_ref(&b));
    let mut links = BTreeMap::new();
    links.insert(a.clone(), vec![(b.clone(), 2.0)]);
    links.insert(b.clone(), vec![(a.clone(), 2.0)]);
    let with = nmapi::tentacle_core::WithCoCitation { base: nmapi::tentacle_core::SubGraph(sub.refs()), links };
    let got = with.rank(&[(a.clone(), 1.0)], std::slice::from_ref(&b));
    assert!(got[0] > plain[0], "co-cited partner gained nothing: {got:?} against {plain:?}");
    assert!(with.related(&a).iter().any(|(k, r, _)| *k == b && r == "co-cited"));
    let none = nmapi::tentacle_core::WithCoCitation { base: nmapi::tentacle_core::SubGraph(sub.refs()), links: BTreeMap::new() };
    assert_eq!(none.rank(&[(a, 1.0)], std::slice::from_ref(&b)), plain, "no links must equal the reference graph");
}
