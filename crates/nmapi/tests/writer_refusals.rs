//! The writer's base cartridge offers that must not seat are refused by name when the writer opens,
//! and the open weights serve.
//!
//! The writer (Qwen3-1.7B, `neural::writer_model`) is offered in order: a cut-off copy of the head of
//! its base cartridge (the label reads, the parameters do not match it), a cartridge made for other
//! open weights (the judge's base cartridge), and a file that does not exist. Each is refused with its
//! reason and the writer runs on its open weights. Writes only `scratch/lane-b/fallback/`.
//! Run: `cargo test --release -p nmapi --test writer_refusals -- --ignored --nocapture`.
use std::io::Read as _;

#[test]
#[ignore = "binds the writer (1.7B) on the graphics card"]
fn wrong_damaged_and_missing_leave_the_writer_on_its_open_weights() {
    let home = nmtype::nano_root();
    let shelf = home.join("library/cartridges/base");
    let dir = home.join("scratch/lane-b/fallback");
    std::fs::create_dir_all(&dir).expect("folder");
    let damaged = dir.join("writer-base-cut.cartridge");
    let mut head = vec![0_u8; 12];
    let mut f = std::fs::File::open(shelf.join("writer/writer-base-v1.cartridge")).expect("writer base cartridge");
    f.read_exact(&mut head).expect("head");
    let len = u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize;
    let mut rest = vec![0_u8; len + 4096];
    f.read_exact(&mut rest).expect("label");
    head.extend(rest);
    std::fs::write(&damaged, &head).expect("cut-off copy");
    let other = shelf.join("reranker/reranker-base-v2.cartridge");
    let missing = dir.join("no-such.cartridge");
    nmapi::neural::BASE_CARTRIDGES.write().expect("lock").insert(nmapi::neural::WRITER.into(), vec![damaged.clone(), other, missing]);
    let m = nmapi::neural::writer_model(&home, true).expect("writer opens");
    let _ = std::fs::remove_file(&damaged);
    let line = nmapi::neural::seat_notes().into_iter().find(|l| l.starts_with(nmapi::neural::WRITER)).unwrap_or_default();
    println!("{line}");
    assert_eq!(m.pop.source, nmengn::model::OPEN_WEIGHTS, "the open weights must serve");
    assert!(line.contains("no base cartridge seated"), "{line}");
    assert!(line.contains("writer-base-cut.cartridge:"), "the damaged copy is named: {line}");
    assert!(line.contains("reranker-base-v2.cartridge:") && line.contains("made for open weights"), "the other weights are named: {line}");
    assert!(line.contains("no-such.cartridge:"), "the missing file is named: {line}");
}
