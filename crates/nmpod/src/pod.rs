//! The Neuropod (guide 4, 4.1 to 4.3): eight tentacles and eight drone tips by default, any
//! count allowed, built from config; with eight of each it is an Octopod, the general term
//! is Neuropod and the count can extend or contract (Luis Jake Gabriel III, 2026-11-08). Its Neuropod interface steps: plan (territories and jobs),
//! forage (every tentacle runs its tip and pipeline), reduce (hand everything to the brain
//! unmerged), emit (one summary packet). Foraging here runs the units in turn; the
//! organism runs them on the NTE through nmapi, with the same phases.
use crate::pipeline::ReflexTable;
use crate::territory;
use crate::unit::{Job, Level, Outcome, Unit};
use nmjson::Value;
use nmtype::{
    Budget, Component, ComponentDescriptor, ComponentType, Fault, FaultKind, FrameView, Neuropod as NeuropodInterface, Pack, Packet, Plan,
    Provenance, SlotView, Status, SubstrateView, Task, Telemetry,
};
use std::sync::Arc;

/// The eight initial cognitive dimensions, carried by the tentacles as default jobs.
pub const DEFAULT_JOBS: [Job; 8] =
    [Job::Retrieval, Job::Structural, Job::Dependency, Job::Semantic, Job::Historical, Job::Execution, Job::Verification, Job::Reasoning];

pub struct Neuropod {
    descriptor: ComponentDescriptor,
    pub tentacles: Vec<Unit>,
    pub tips: Vec<Unit>,
    pub max_depth: u8,
    sub: Arc<dyn SubstrateView>,
    slots: Arc<dyn SlotView>,
    last: Vec<Outcome>,
}
impl std::fmt::Debug for Neuropod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Neuropod").field("tentacles", &self.tentacles.len()).field("tips", &self.tips.len()).finish_non_exhaustive()
    }
}

impl Neuropod {
    /// The eight-tentacle, eight-drone-tip configuration is an Octopod (Luis Jake Gabriel III, 2026-11-08);
    /// any other count is a Neuropod of that size.
    #[must_use]
    pub fn is_octopod(&self) -> bool {
        self.tentacles.len() == 8 && self.tips.len() == 8
    }
    /// Build from `{"tentacles": n, "tips": n, "max_depth": d}`.
    /// # Errors
    /// Counts out of range.
    pub fn from_config(
        cfg: &Value,
        reflexes: &Arc<ReflexTable>,
        sub: Arc<dyn SubstrateView>,
        slots: Arc<dyn SlotView>,
    ) -> Result<Self, Fault> {
        let get = |k: &str, d: u64| cfg.get(k).and_then(Value::integer).unwrap_or(d);
        let (t, d, depth) = (get("tentacles", 8), get("tips", 8), get("max_depth", 1));
        if t == 0 || t > 64 || d > 64 {
            return Err(Fault::new("nmpod.pod", FaultKind::Config, format!("{t} tentacles and {d} tips is out of range")));
        }
        let depth = u8::try_from(depth).unwrap_or(1);
        let job = |i: u64| DEFAULT_JOBS.get(usize::try_from(i).unwrap_or(0) % 8).copied().unwrap_or(Job::Retrieval);
        let tentacles =
            (0..t).map(|i| Unit::new(&format!("nmpod.unit#t{i}"), Level::Tentacle, job(i), Arc::clone(reflexes), depth)).collect();
        let tips = (0..d).map(|i| Unit::new(&format!("nmpod.unit#d{i}"), Level::Drone, job(i), Arc::clone(reflexes), depth)).collect();
        Ok(Self {
            descriptor: ComponentDescriptor {
                id: "nmpod.pod#p0".into(),
                kind: ComponentType::Neuropod,
                version: 1,
                parent: None,
                ports_in: Vec::new(),
                ports_out: Vec::new(),
                deps: Vec::new(),
                card_hash: String::new(),
            },
            tentacles,
            tips,
            max_depth: depth,
            sub,
            slots,
            last: Vec::new(),
        })
    }
    /// Territories for a task, one per tentacle.
    #[must_use]
    pub fn territories(&self) -> Vec<Vec<nmtype::Region>> {
        territory::partition(&self.sub.files(), self.tentacles.len())
    }
}

impl Component for Neuropod {
    fn descriptor(&self) -> &ComponentDescriptor {
        &self.descriptor
    }
    fn initialize(&mut self, _: &Value) -> Result<(), Fault> {
        Ok(())
    }
    fn suspend(&mut self) -> Result<(), Fault> {
        self.tentacles.iter_mut().chain(self.tips.iter_mut()).try_for_each(Component::suspend)
    }
    fn resume(&mut self) -> Result<(), Fault> {
        self.tentacles.iter_mut().chain(self.tips.iter_mut()).try_for_each(Component::resume)
    }
    fn shutdown(&mut self) {
        self.tentacles.iter_mut().chain(self.tips.iter_mut()).for_each(Component::shutdown);
    }
    fn inspect_telemetry(&self) -> Telemetry {
        let mut t = Telemetry::idle(&self.descriptor.id);
        t.executions = self.tentacles.iter().map(|u| u.inspect_telemetry().executions).sum();
        t.anomalies = self.tentacles.iter().map(|u| u.inspect_telemetry().anomalies).sum();
        t
    }
    fn check(&mut self) -> Result<(), Fault> {
        check().map_err(|e| Fault::new("nmpod.pod", FaultKind::Execution, e))
    }
}

impl NeuropodInterface for Neuropod {
    fn plan(&mut self, task: &Task) -> Plan {
        let regions = self.territories().into_iter().flatten().collect();
        Plan { task: task.id, regions, jobs: DEFAULT_JOBS.iter().map(|j| j.name().to_owned()).collect(), views: Vec::new() }
    }
    fn forage(&mut self, plan: &Plan, frame: &dyn FrameView) -> Vec<Packet> {
        let n = self.tentacles.len();
        let mut territories: Vec<Vec<nmtype::Region>> = vec![Vec::new(); n];
        for r in &plan.regions {
            if let Some(t) = territories.get_mut(usize::from(r.territory)) {
                t.push(r.clone());
            }
        }
        let jobs: Vec<Job> = plan.jobs.iter().filter_map(|j| Job::from_name(j)).collect();
        let task = Task {
            id: plan.task,
            objective: plan.views.first().cloned().unwrap_or_default(),
            budget: *frame.budget(),
            scope: String::new(),
            parent: None,
            kind: nmtype::TaskKind::Context,
        };
        self.last.clear();
        for i in 0..n {
            let focus: Vec<Job> = jobs.iter().copied().filter(|j| Job::FOCUS.contains(j)).collect();
            let Some(a) = self.tentacles.get(i).and_then(|t| t.assign(&task, &territories, i, &jobs, &focus)) else { continue };
            let (found, readers) = match self.tips.get(i) {
                Some(tip) => tip.sense(&a, self.sub.as_ref(), self.slots.as_ref()),
                None => (Vec::new(), 0),
            };
            if let Some(t) = self.tentacles.get_mut(i) {
                let out = t.process(&a, found, readers, self.sub.as_ref(), self.slots.as_ref());
                self.last.push(out);
            }
        }
        self.last.iter().flat_map(|o| o.packets.iter().chain(&o.questions).cloned()).collect()
    }
    fn reduce(&mut self, packets: Vec<Packet>, budget: &Budget) -> Pack {
        Pack {
            task: 0,
            tokens: budget.tokens,
            withheld: packets.is_empty(),
            reason: "handed to the Core Brain unmerged".into(),
            packets,
            floor_met: true,
        }
    }
    fn emit(&mut self, pack: &Pack) -> Packet {
        Packet::new(
            "pack",
            "neuropod",
            &format!("{} packets from {} tentacles", pack.packets.len(), self.tentacles.len()),
            Provenance::new("unit:nmpod.pod#p0", "nmpod.pod#p0", self.sub.version(), nmtype::now_ms()),
        )
        .status(Status::Inferred)
    }
}

/// Self-check of the `pod` sub-component: eight tentacles and eight tips forage the fixture
/// and both values of the constant reach the reduce step, each from its own territory.
pub(crate) fn check() -> Result<(), String> {
    let (sub, slots) = crate::fixture::views();
    let reflexes = Arc::new(ReflexTable::default());
    let mut pod = Neuropod::from_config(&Value::Null, &reflexes, Arc::new(sub), Arc::new(slots)).map_err(|e| e.to_string())?;
    if pod.tentacles.len() != 8 || pod.tips.len() != 8 {
        return Err("nmpod.pod: default anatomy is not 8 and 8".into());
    }
    let task = Task {
        id: 1,
        objective: "What is LIMIT?".into(),
        budget: Budget { tokens: 400, ms: 5000, depth: 1, units: 16 },
        scope: String::new(),
        parent: None,
        kind: nmtype::TaskKind::Context,
    };
    let mut plan = pod.plan(&task);
    plan.views = vec![task.objective.clone()];
    struct F(Budget);
    impl FrameView for F {
        fn version(&self) -> u64 {
            1
        }
        fn deadline(&self) -> u64 {
            u64::MAX
        }
        fn budget(&self) -> &Budget {
            &self.0
        }
        fn trace(&self, _: nmtype::Hop) {}
    }
    let packets = pod.forage(&plan, &F(task.budget));
    let values: Vec<&str> = packets.iter().filter(|p| p.key() == "value:LIMIT").map(Packet::text).collect();
    if !values.contains(&"LIMIT = 4") || !values.contains(&"LIMIT = 9") {
        return Err(format!("nmpod.pod: values reaching reduce {values:?}"));
    }
    Ok(())
}
