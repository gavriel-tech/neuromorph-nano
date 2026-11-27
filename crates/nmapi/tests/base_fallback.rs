//! A base cartridge that is damaged or missing is refused with its reason when the MicroModel opens,
//! and the open weights serve.
//!
//! The embedder is offered two cartridges in order: a cut-off copy of the head of its base cartridge
//! (the label reads, the parameters do not match it) and a file that does not exist. Both are
//! refused by name and the embedder runs on its open weights. Writes only `scratch/lane-b/fallback/`.
//! Run: `cargo test --release -p nmapi --test base_fallback -- --ignored --nocapture`.
use std::io::Read as _;

#[test]
#[ignore = "binds the embedder on the graphics card (about a minute)"]
fn damaged_and_missing_fall_back_to_the_open_weights() {
    let home = nmtype::nano_root();
    let dir = home.join("scratch/lane-b/fallback");
    std::fs::create_dir_all(&dir).expect("folder");
    let damaged = dir.join("embedder-base-cut.cartridge");
    let mut head = vec![0_u8; 12];
    let mut f = std::fs::File::open(home.join("library/cartridges/base/embedder/embedder-base-v1.cartridge")).expect("embedder base cartridge");
    f.read_exact(&mut head).expect("head");
    let len = u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize;
    let mut rest = vec![0_u8; len + 4096];
    f.read_exact(&mut rest).expect("label");
    head.extend(rest);
    std::fs::write(&damaged, &head).expect("cut-off copy");
    let missing = dir.join("no-such.cartridge");
    nmapi::neural::BASE_CARTRIDGES.write().expect("lock").insert("qwen3-embedding-0.6b".into(), vec![damaged.clone(), missing]);
    let m = nmapi::neural::embedder_model(&home).expect("embedder opens");
    let _ = std::fs::remove_file(&damaged);
    let notes = nmapi::neural::seat_notes();
    let line = notes.iter().find(|l| l.starts_with("qwen3-embedding-0.6b")).cloned().unwrap_or_default();
    println!("{line}");
    assert_eq!(m.pop.source, nmengn::model::OPEN_WEIGHTS, "the open weights must serve");
    assert!(line.contains("no base cartridge seated"), "{line}");
    assert!(line.contains("embedder-base-cut.cartridge:") && line.contains("no-such.cartridge:"), "each refusal is named: {line}");
}
