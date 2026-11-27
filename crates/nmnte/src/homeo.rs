//! Homeostatic Integration (guide 11.1 item 8, 13): the hook RNAM and the executor call after
//! every publish and every frame, so the Core Brain's homeostasis sees each new anatomy and
//! each run's faults, skips and timing. Implemented by the organism (nmapi), never here.
use crate::scheduler::RunResult;
use crate::snapshot::Snapshot;

pub trait Homeostatic: Send {
    fn after_publish(&mut self, snap: &Snapshot);
    fn after_frame(&mut self, run: &RunResult);
}

/// A hook that counts what it was told (used by the section check and as a default).
#[derive(Debug, Default)]
pub struct Counter {
    pub publishes: u64,
    pub frames: u64,
    pub faults: u64,
}
impl Homeostatic for Counter {
    fn after_publish(&mut self, _: &Snapshot) {
        self.publishes += 1;
    }
    fn after_frame(&mut self, run: &RunResult) {
        self.frames += 1;
        self.faults += run.faults.len() as u64;
    }
}

/// Self-check of the `homeo` sub-component: the hook sees a publish and a frame's faults.
pub(crate) fn check() -> Result<(), String> {
    let mut c = Counter::default();
    c.after_publish(&Snapshot::empty());
    let run = RunResult {
        version: 1,
        state: crate::frame::ExecutionState::Completed,
        outputs: std::collections::BTreeMap::new(),
        faults: vec![nmtype::Fault::new("x", nmtype::FaultKind::Execution, "seeded")],
        isolated: Vec::new(),
        skipped: Vec::new(),
        iterations: std::collections::BTreeMap::new(),
        dropped: 0,
    };
    c.after_frame(&run);
    if c.publishes != 1 || c.frames != 1 || c.faults != 1 {
        return Err("nmnte.homeo: hook counts".into());
    }
    Ok(())
}
