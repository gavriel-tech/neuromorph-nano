//! The process's creation time: before now, after the system started, steady across calls, and
//! not later than a moment taken at the start of this test.
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn process_started_ms_reads_this_process() {
    let first = nmffi::dynlib::process_started_ms().expect("creation time");
    let now = u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH).expect("clock").as_millis()).expect("ms");
    assert!(first <= now, "created at {first} ms, after now ({now} ms)");
    assert_eq!(nmffi::dynlib::process_started_ms().expect("creation time"), first, "the creation time moved");
    nmffi::dynlib::check().expect("dynlib check");
}
