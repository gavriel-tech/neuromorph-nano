//! nmtool: build help and Step Check on scratch copies (guide 17.3, 17.5; PLAN 4c; vision
//! 18.1 build output as a foraging source). Sub-components, one per file: `run` (child
//! process with a deadline), `lex` (rule layer over `nmlex`), `build` (one package, one
//! command, rustc diagnostics into `build-error` packets), `explain` (first error located
//! and explained from exact sources), `stepcheck` (diff on a new scratch copy). `improve` (findings
//! and one-line proposals from exact sources) and `approval` (the proposal queue; nothing is
//! applied here). `verify` stays planned. Depends on nmjson, nmlex, nmtype only.
pub mod approval;
pub mod build;
pub mod conformance;
pub mod explain;
pub mod improve;
pub mod lex;
pub mod run;
pub mod scoreboard;
pub mod stepcheck;
pub mod verify;

use nmjson::Value;
use nmtype::{
    Component, ComponentDescriptor, ComponentType, ContextProcessor, Fault, FaultKind, FrameView, Health, Packet, PortDesc, PortKind,
    Telemetry,
};
use std::path::PathBuf;

/// The working folder of one section check: `tests/section-checks/nmtool-<name>/`, reused
/// and cleared at each run (`nmtype::check_dir`).
pub(crate) fn scratch_dir(name: &str) -> Result<PathBuf, String> {
    nmtype::check_dir(&format!("nmtool-{name}")).map_err(|e| format!("scratch: {e}"))
}

/// One section check: (sub-component id, check returning ok or one fault line).
pub type Check = (&'static str, fn() -> Result<(), String>);

/// Section checks of this crate, one per sub-component, cheapest first.
#[must_use]
pub fn checks() -> Vec<Check> {
    vec![
        ("nmtool.lex", lex::check),
        ("nmtool.run", run::check),
        ("nmtool.build", build::check),
        ("nmtool.explain", explain::check),
        ("nmtool.stepcheck", stepcheck::check),
        ("nmtool.improve", improve::check),
        ("nmtool.verify", verify::check),
        ("nmtool.approval", approval::check),
        ("nmtool.conformance", conformance::check),
        ("nmtool.scoreboard", scoreboard::check),
    ]
}

/// The build tool as a graph component: a `build-request` packet (key: package or `*`,
/// text: check, clippy, test or build) in, the run's `build-error` packets out.
pub struct BuildTool {
    desc: ComponentDescriptor,
    root: PathBuf,
    target: PathBuf,
    budget: usize,
    calls: u64,
    ms: f64,
    suspended: bool,
}
impl BuildTool {
    /// A build tool over `root`, with cargo output in `target`.
    #[must_use]
    pub fn new(root: PathBuf, target: PathBuf) -> Self {
        Self {
            desc: ComponentDescriptor {
                id: "nmtool.build#n0".into(),
                kind: ComponentType::Tool,
                version: 1,
                parent: Some("nmapi#n0".into()),
                ports_in: vec![PortDesc::new("request", PortKind::Packets, true)],
                ports_out: vec![PortDesc::new("packets", PortKind::Packets, true)],
                deps: vec!["nmjson".into(), "nmlex".into(), "nmtype".into()],
                card_hash: String::new(),
            },
            root,
            target,
            budget: 12,
            calls: 0,
            ms: 0.0,
            suspended: false,
        }
    }
}
impl Component for BuildTool {
    fn descriptor(&self) -> &ComponentDescriptor {
        &self.desc
    }
    fn initialize(&mut self, cfg: &Value) -> Result<(), Fault> {
        if let Some(b) = cfg.get("budget").and_then(Value::integer).ok().and_then(|b| usize::try_from(b).ok()) {
            self.budget = b.max(1);
        }
        Ok(())
    }
    fn suspend(&mut self) -> Result<(), Fault> {
        self.suspended = true;
        Ok(())
    }
    fn resume(&mut self) -> Result<(), Fault> {
        self.suspended = false;
        Ok(())
    }
    fn shutdown(&mut self) {
        self.suspended = true;
    }
    fn inspect_telemetry(&self) -> Telemetry {
        let mut t = Telemetry::idle(&self.desc.id);
        t.executions = self.calls;
        #[expect(clippy::cast_possible_truncation, reason = "milliseconds per call fit f32")]
        let avg = if self.calls == 0 { 0.0 } else { (self.ms / self.calls as f64) as f32 };
        t.latency_ms_avg = avg;
        t.health = if self.suspended { Health::Unavailable } else { Health::Healthy };
        t
    }
    fn check(&mut self) -> Result<(), Fault> {
        if !self.root.join("Cargo.toml").is_file() {
            return Err(Fault::new(&self.desc.id, FaultKind::Config, format!("no Cargo.toml under {}", self.root.display())));
        }
        Ok(())
    }
}
impl ContextProcessor for BuildTool {
    fn process(&mut self, input: &[Packet], _frame: &dyn FrameView) -> Result<Vec<Packet>, Fault> {
        if self.suspended {
            return Err(Fault::new(&self.desc.id, FaultKind::Policy, "suspended"));
        }
        let mut out = Vec::new();
        for p in input.iter().filter(|p| p.kind() == "build-request") {
            let kind = build::Kind::parse(p.text())?;
            let package = (p.key() != "*" && !p.key().is_empty()).then(|| p.key().to_owned());
            let req = build::Request { root: self.root.clone(), target: self.target.clone(), kind, package, deadline: run::DEADLINE };
            let r = build::run(&req);
            self.calls += 1;
            #[expect(clippy::cast_precision_loss, reason = "milliseconds")]
            let ms = r.cargo_ms as f64;
            self.ms += ms;
            out.extend(build::packets(&r, self.budget, false));
        }
        Ok(out)
    }
}
