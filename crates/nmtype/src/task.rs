//! Task, budget, region, plan, pack and the frame view (guide 3, 14.3, 17.4). `FrameView`
//! is the trait object an NTE `ExecutionFrame` implements, so nmtype never depends on nmnte.
use crate::component::Fault;
use crate::hop::Hop;
use crate::packet::Packet;
use crate::wire::{self, named_enum};
use nmjson::Value;

/// Demand-based budget for one task or unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    pub tokens: u32,
    pub ms: u32,
    pub depth: u8,
    pub units: u8,
}
impl Budget {
    #[must_use]
    pub fn to_value(self) -> Value {
        wire::obj(vec![
            ("tokens", wire::u(u64::from(self.tokens))),
            ("ms", wire::u(u64::from(self.ms))),
            ("depth", wire::u(u64::from(self.depth))),
            ("units", wire::u(u64::from(self.units))),
        ])
    }
    /// # Errors
    /// Rejects a value missing any field or out of range.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        Ok(Self {
            tokens: wire::get_n(v, "task", "tokens")?,
            ms: wire::get_n(v, "task", "ms")?,
            depth: wire::get_n(v, "task", "depth")?,
            units: wire::get_n(v, "task", "units")?,
        })
    }
}

named_enum! {
    /// Objective kind; picks the row of `designs/compositions.json` (PLAN 4d item 4).
    pub enum TaskKind in "task" {
        Context => "context", Understand => "understand", Improve => "improve",
        Build => "build", Verify => "verify", Anatomy => "anatomy",
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Task {
    pub id: u64,
    pub objective: String,
    pub budget: Budget,
    pub scope: String,
    pub parent: Option<u64>,
    pub kind: TaskKind,
}
impl Task {
    #[must_use]
    pub fn to_value(&self) -> Value {
        wire::obj(vec![
            ("id", wire::u(self.id)),
            ("objective", wire::s(&self.objective)),
            ("budget", self.budget.to_value()),
            ("scope", wire::s(&self.scope)),
            ("parent", wire::opt_u(self.parent)),
            ("kind", self.kind.to_value()),
        ])
    }
    /// # Errors
    /// Rejects a value missing any field.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let m = "task";
        Ok(Self {
            id: wire::get_u(v, m, "id")?,
            objective: wire::get_s(v, m, "objective")?,
            budget: Budget::from_value(wire::get_v(v, m, "budget")?)?,
            scope: wire::get_s(v, m, "scope")?,
            parent: wire::get_opt_u(v, m, "parent")?,
            kind: wire::get_e(v, m, "kind", TaskKind::from_value)?,
        })
    }
}

/// A territory slice for one task: file range plus priority. Never changes the anatomy.
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    pub territory: u8,
    pub path: String,
    pub start: usize,
    pub end: usize,
    pub priority: f32,
}
impl Region {
    #[must_use]
    pub fn to_value(&self) -> Value {
        wire::obj(vec![
            ("territory", wire::u(u64::from(self.territory))),
            ("path", wire::s(&self.path)),
            ("start", wire::u(self.start as u64)),
            ("end", wire::u(self.end as u64)),
            ("priority", wire::f(self.priority)),
        ])
    }
    /// # Errors
    /// Rejects a value missing any field.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let m = "task";
        Ok(Self {
            territory: wire::get_n(v, m, "territory")?,
            path: wire::get_s(v, m, "path")?,
            start: wire::get_n(v, m, "start")?,
            end: wire::get_n(v, m, "end")?,
            priority: wire::get_f(v, m, "priority")?,
        })
    }
}

/// Step 1 output: regions, jobs and views for one task.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub task: u64,
    pub regions: Vec<Region>,
    pub jobs: Vec<String>,
    pub views: Vec<String>,
}
impl Plan {
    #[must_use]
    pub fn to_value(&self) -> Value {
        wire::obj(vec![
            ("task", wire::u(self.task)),
            ("regions", Value::Array(self.regions.iter().map(Region::to_value).collect())),
            ("jobs", wire::strings(&self.jobs)),
            ("views", wire::strings(&self.views)),
        ])
    }
    /// # Errors
    /// Rejects a value missing any field.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let m = "task";
        Ok(Self {
            task: wire::get_u(v, m, "task")?,
            regions: wire::get_a(v, m, "regions")?.iter().map(Region::from_value).collect::<Result<_, _>>()?,
            jobs: wire::get_strings(v, m, "jobs")?,
            views: wire::get_strings(v, m, "views")?,
        })
    }
}

/// Step 3 output: the reduced, cited pack; `withheld` when nothing is observed or verified.
#[derive(Clone, Debug, PartialEq)]
pub struct Pack {
    pub task: u64,
    pub packets: Vec<Packet>,
    pub withheld: bool,
    pub reason: String,
    pub tokens: u32,
    /// False when the pack is served although no packet reached the relevance floor: low
    /// confidence, kept apart from withheld so decoy behaviour stays measurable.
    pub floor_met: bool,
}
impl Pack {
    #[must_use]
    pub fn to_value(&self) -> Value {
        wire::obj(vec![
            ("task", wire::u(self.task)),
            ("packets", Value::Array(self.packets.iter().map(Packet::to_value).collect())),
            ("withheld", Value::Bool(self.withheld)),
            ("reason", wire::s(&self.reason)),
            ("tokens", wire::u(u64::from(self.tokens))),
            ("floor_met", Value::Bool(self.floor_met)),
        ])
    }
    /// # Errors
    /// Rejects a value missing any field.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let m = "task";
        Ok(Self {
            task: wire::get_u(v, m, "task")?,
            packets: wire::get_a(v, m, "packets")?.iter().map(Packet::from_value).collect::<Result<_, _>>()?,
            withheld: wire::get_b(v, m, "withheld")?,
            reason: wire::get_s(v, m, "reason")?,
            tokens: wire::get_n(v, m, "tokens")?,
            // Absent in packs written before the field existed: the floor was met then.
            floor_met: wire::get_b(v, m, "floor_met").unwrap_or(true),
        })
    }
}

/// What a component may see of its execution frame: snapshot version, deadline, budget,
/// and an append-only trace.
pub trait FrameView: Send + Sync {
    fn version(&self) -> u64;
    fn deadline(&self) -> u64;
    fn budget(&self) -> &Budget;
    fn trace(&self, hop: Hop);
}

/// Self-check of the `task` sub-component: task, plan and pack round-trip.
pub(crate) fn check() -> Result<(), String> {
    let budget = Budget { tokens: 4000, ms: 50_000, depth: 1, units: 16 };
    let task = Task {
        id: 1,
        objective: "where is Packet defined".into(),
        budget,
        scope: "workspace".into(),
        parent: None,
        kind: TaskKind::Context,
    };
    if Task::from_value(&task.to_value()).map_err(|e| e.to_string())? != task {
        return Err("nmtype.task: task round trip".into());
    }
    let plan = Plan {
        task: 1,
        regions: vec![Region { territory: 0, path: "crates".into(), start: 0, end: 10, priority: 0.5 }],
        jobs: vec!["structural".into()],
        views: vec!["structural".into(), "objective".into()],
    };
    if Plan::from_value(&plan.to_value()).map_err(|e| e.to_string())? != plan {
        return Err("nmtype.task: plan round trip".into());
    }
    let pack = Pack { task: 1, packets: vec![], withheld: true, reason: "nothing observed".into(), tokens: 0, floor_met: true };
    if Pack::from_value(&pack.to_value()).map_err(|e| e.to_string())? != pack {
        return Err("nmtype.task: pack round trip".into());
    }
    Ok(())
}
