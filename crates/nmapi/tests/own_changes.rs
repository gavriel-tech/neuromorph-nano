//! Own changes by content (2026-11-12, motor division): a change the organism predicted (a passing
//! Step Check records each file's post-edit hash) is recognised as its own when it arrives, and the
//! prediction is used up; any other change is outside input.
use std::path::PathBuf;

fn home() -> PathBuf {
    nmtype::nano_root()
}

#[test]
fn predicted_change_is_own_and_others_are_outside() {
    let root = nmtype::check_dir("nmapi-own-changes").expect("folder");
    std::fs::create_dir_all(root.join("src")).expect("src");
    std::fs::write(root.join("src/a.rs"), "pub fn a() -> u8 { 1 }\n").expect("a");
    std::fs::write(root.join("src/b.rs"), "pub fn b() -> u8 { 2 }\n").expect("b");
    let mem = nmtype::check_dir("nmapi-own-changes-memory").expect("memory");
    let org = nmapi::forage::Organism::open_with(&home(), &root, &[], false, &mem).expect("open");
    // The organism predicts the content a.rs will have once its own change is applied.
    let next_a = "pub fn a() -> u8 { 3 }\n";
    org.own.lock().expect("ledger").predict("src/a.rs", &nmhash::sha256_bytes(next_a.as_bytes()));
    std::fs::write(root.join("src/a.rs"), next_a).expect("write a");
    std::fs::write(root.join("src/b.rs"), "pub fn b() -> u8 { 9 }\n").expect("write b");
    let (report, _) = org.sync().expect("sync");
    assert_eq!(report.changed.len(), 2, "{report:?}");
    let (own, outside) = org.last_sync_origin.lock().expect("origin").clone();
    assert_eq!(own.iter().map(|f| f.replace('\\', "/")).collect::<Vec<_>>(), vec!["src/a.rs".to_owned()], "own {own:?}");
    assert_eq!(outside.iter().map(|f| f.replace('\\', "/")).collect::<Vec<_>>(), vec!["src/b.rs".to_owned()], "outside {outside:?}");
    // The prediction was used up: the same content again is no longer own.
    assert_eq!(org.own.lock().expect("ledger").pending(), 0);
}
