//! `run s6 fingerprint both` as a test (lane T1, 2026-11-18): the reference path's fingerprint on
//! the processor and the card equals the recording `tests/engine-reference/fingerprint.txt` line
//! for line (PA-T). Run by hand under the card lock, with the switch under test set, e.g.
//! `NANO_INTAKE_KEYS=on cargo test --release -p nmcore --test fingerprint_t1 -- --ignored --nocapture`.
use nmengn::Device;

#[test]
#[ignore = "needs the release weights and the card; run by hand under the card lock"]
fn fingerprint_both_equals_record() {
    let home = nmtype::nano_root();
    let got = nmcore::selftest::fingerprint(&home, &[Device::Cpu, Device::Gpu]).expect("fingerprint");
    let want = std::fs::read_to_string(nmcore::selftest::fingerprint_file(&home)).expect("recording");
    let mut diff = Vec::new();
    let mut compared = 0;
    for dev in ["cpu ", "gpu "] {
        let g: Vec<&str> = got.lines().filter(|l| l.starts_with(dev)).collect();
        let w: Vec<&str> = want.lines().filter(|l| l.starts_with(dev)).collect();
        if g.is_empty() {
            continue;
        }
        if g.len() != w.len() {
            diff.push(format!("{dev}: {} lines against {} recorded", g.len(), w.len()));
        }
        for (a, b) in g.iter().zip(&w) {
            compared += 1;
            if a != b {
                diff.push(format!("{a} | recorded {b}"));
            }
        }
    }
    println!("fingerprint: {compared} lines compared, {} differ; NANO_INTAKE_KEYS={:?}", diff.len(), std::env::var("NANO_INTAKE_KEYS").ok());
    for d in &diff {
        println!("  {d}");
    }
    assert!(compared > 0 && diff.is_empty(), "fingerprint differs from the recording");
}
