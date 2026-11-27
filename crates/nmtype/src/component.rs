//! Component identity, telemetry, faults and the trait family (guide 8, 8.1; Luis's lifecycle
//! contract). Identity (`Component`) is kept apart from the typed execution interfaces
//! (`ContextProcessor`, `TensorProcessor`, `MemoryComponent`); there is no universal execute.
use crate::card::{Card, PartRef};
use crate::memory::{Query, Written};
use crate::packet::Packet;
use crate::port::PortDesc;
use crate::task::{Budget, FrameView, Pack, Plan, Task};
use crate::tensor::Tensor;
use crate::wire::{self, named_enum};
use nmjson::Value;
use std::fmt;

/// `crate.module#instance`, for example `nmpod.unit#t3`.
pub type ComponentId = String;

named_enum! {
    /// What a component is, from the organism down to one parameter cluster.
    pub enum ComponentType in "component" {
        Organism => "organism", Brain => "brain", Neuropod => "neuropod", Tentacle => "tentacle",
        Drone => "drone", Slot => "slot", Block => "block", Attention => "attention",
        FeedForward => "feedforward", Norm => "norm", Head => "head", Adapter => "adapter",
        Memory => "memory", Gating => "gating", Routing => "routing",
        ParameterCluster => "parametercluster", Tool => "tool", Other => "other",
    }
}
named_enum! {
    /// Health as homeostasis and fault isolation see it.
    pub enum Health in "component" {
        Healthy => "healthy", Degraded => "degraded", Isolated => "isolated", Unavailable => "unavailable",
    }
}
named_enum! {
    /// Fault class: which contract was broken.
    pub enum FaultKind in "component" {
        Config => "config", Resource => "resource", Interface => "interface",
        Execution => "execution", Integrity => "integrity", Policy => "policy",
    }
}

/// A failure of one component. Recoverable by default; quarantine decides what to do.
#[derive(Clone, Debug, PartialEq)]
pub struct Fault {
    pub component: ComponentId,
    pub kind: FaultKind,
    pub message: String,
    pub recoverable: bool,
}
impl Fault {
    #[must_use]
    pub fn new(component: &str, kind: FaultKind, message: impl Into<String>) -> Self {
        Self { component: component.to_owned(), kind, message: message.into(), recoverable: true }
    }
    /// Mark the fault as not recoverable in place (the unit must be rebuilt from its card).
    #[must_use]
    pub fn unrecoverable(mut self) -> Self {
        self.recoverable = false;
        self
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        wire::obj(vec![
            ("component", wire::s(&self.component)),
            ("kind", self.kind.to_value()),
            ("message", wire::s(&self.message)),
            ("recoverable", Value::Bool(self.recoverable)),
        ])
    }
    /// # Errors
    /// Rejects a value missing any field.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let m = "component";
        Ok(Self {
            component: wire::get_s(v, m, "component")?,
            kind: wire::get_e(v, m, "kind", FaultKind::from_value)?,
            message: wire::get_s(v, m, "message")?,
            recoverable: wire::get_b(v, m, "recoverable")?,
        })
    }
}
impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{}] {}", self.component, self.kind.name(), self.message)
    }
}
impl std::error::Error for Fault {}

/// Identity, ports, dependencies and the card hash of one component instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentDescriptor {
    pub id: ComponentId,
    pub kind: ComponentType,
    pub version: u64,
    pub parent: Option<ComponentId>,
    pub ports_in: Vec<PortDesc>,
    pub ports_out: Vec<PortDesc>,
    pub deps: Vec<ComponentId>,
    pub card_hash: String,
}
impl ComponentDescriptor {
    /// Fields only, without `card_hash` (the card writes it as `hash`).
    pub(crate) fn fields(&self) -> Vec<(&'static str, Value)> {
        vec![
            ("id", wire::s(&self.id)),
            ("kind", self.kind.to_value()),
            ("version", wire::u(self.version)),
            ("parent", wire::opt_s(self.parent.as_deref())),
            ("ports_in", Value::Array(self.ports_in.iter().map(PortDesc::to_value).collect())),
            ("ports_out", Value::Array(self.ports_out.iter().map(PortDesc::to_value).collect())),
            ("deps", wire::strings(&self.deps)),
        ]
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut f = self.fields();
        f.push(("card_hash", wire::s(&self.card_hash)));
        wire::obj(f)
    }
    /// Read the fields; `hash_key` names the hash field (`card_hash` here, `hash` in a card).
    pub(crate) fn read(v: &Value, hash_key: &str) -> Result<Self, Fault> {
        let m = "component";
        let ports = |key| -> Result<Vec<PortDesc>, Fault> { wire::get_a(v, m, key)?.iter().map(PortDesc::from_value).collect() };
        Ok(Self {
            id: wire::get_s(v, m, "id")?,
            kind: wire::get_e(v, m, "kind", ComponentType::from_value)?,
            version: wire::get_u(v, m, "version")?,
            parent: wire::get_opt_s(v, m, "parent")?,
            ports_in: ports("ports_in")?,
            ports_out: ports("ports_out")?,
            deps: wire::get_strings(v, m, "deps")?,
            card_hash: wire::get_s(v, m, hash_key)?,
        })
    }
    /// # Errors
    /// Rejects a value missing any field.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        Self::read(v, "card_hash")
    }
}

/// Read-only telemetry of one component (load, latency, confidence, anomalies).
#[derive(Clone, Debug, PartialEq)]
pub struct Telemetry {
    pub id: ComponentId,
    pub load: f32,
    pub memory_bytes: usize,
    pub executions: u64,
    pub latency_ms_avg: f32,
    pub confidence_avg: f32,
    pub anomalies: u64,
    pub health: Health,
}
impl Telemetry {
    /// Idle, healthy telemetry for a new component.
    #[must_use]
    pub fn idle(id: &str) -> Self {
        Self {
            id: id.to_owned(),
            load: 0.0,
            memory_bytes: 0,
            executions: 0,
            latency_ms_avg: 0.0,
            confidence_avg: 0.0,
            anomalies: 0,
            health: Health::Healthy,
        }
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        wire::obj(vec![
            ("id", wire::s(&self.id)),
            ("load", wire::f(self.load)),
            ("memory_bytes", wire::u(self.memory_bytes as u64)),
            ("executions", wire::u(self.executions)),
            ("latency_ms_avg", wire::f(self.latency_ms_avg)),
            ("confidence_avg", wire::f(self.confidence_avg)),
            ("anomalies", wire::u(self.anomalies)),
            ("health", self.health.to_value()),
        ])
    }
    /// # Errors
    /// Rejects a value missing any field.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let m = "component";
        Ok(Self {
            id: wire::get_s(v, m, "id")?,
            load: wire::get_f(v, m, "load")?,
            memory_bytes: wire::get_n(v, m, "memory_bytes")?,
            executions: wire::get_u(v, m, "executions")?,
            latency_ms_avg: wire::get_f(v, m, "latency_ms_avg")?,
            confidence_avg: wire::get_f(v, m, "confidence_avg")?,
            anomalies: wire::get_u(v, m, "anomalies")?,
            health: wire::get_e(v, m, "health", Health::from_value)?,
        })
    }
}

/// The lifecycle contract every component and sub-component implements (guide 8.1).
/// `check` is the one self-check per sub-component (PLAN 4b rule 10): it runs the
/// component alone on a tiny built-in fixture and answers ok or one fault.
pub trait Component: Send {
    fn descriptor(&self) -> &ComponentDescriptor;
    /// # Errors
    /// A config the component cannot accept.
    fn initialize(&mut self, cfg: &Value) -> Result<(), Fault>;
    /// # Errors
    /// The component cannot release its resources now.
    fn suspend(&mut self) -> Result<(), Fault>;
    /// # Errors
    /// The component cannot reacquire its resources.
    fn resume(&mut self) -> Result<(), Fault>;
    fn shutdown(&mut self);
    fn inspect_telemetry(&self) -> Telemetry;
    /// # Errors
    /// One fault line naming what failed on the built-in fixture.
    fn check(&mut self) -> Result<(), Fault>;
}

/// Packets in, packets out.
pub trait ContextProcessor: Component {
    /// # Errors
    /// The input violates the component's contract or execution failed.
    fn process(&mut self, input: &[Packet], frame: &dyn FrameView) -> Result<Vec<Packet>, Fault>;
}
/// A tensor in, a tensor out (neural components: Norm, `AttnProj`, Gate, and so on).
pub trait TensorProcessor: Component {
    /// # Errors
    /// Shape or dtype mismatch, or execution failed.
    fn forward(&mut self, x: &Tensor, frame: &dyn FrameView) -> Result<Tensor, Fault>;
}
/// Memory: write through a gate, recall by query.
pub trait MemoryComponent: Component {
    fn write(&mut self, p: &Packet) -> Written;
    fn recall(&self, q: &Query) -> Vec<Packet>;
}
/// The one interface reused at every level: drone, tentacle, Neuropod, Core Brain,
/// Neuromorph (decision 1). The four steps of the Context Stack. Neuropod is the general
/// term (Luis Jake Gabriel III, 2026-11-08 07:42 UTC); Octopod names the eight-tentacle, eight-drone-tip
/// configuration of it, and the count can extend or contract.
pub trait Neuropod: Component {
    /// Step 1, reasoning and mapping.
    fn plan(&mut self, task: &Task) -> Plan;
    /// Step 2, drones.
    fn forage(&mut self, plan: &Plan, frame: &dyn FrameView) -> Vec<Packet>;
    /// Step 3, the reduced pack.
    fn reduce(&mut self, packets: Vec<Packet>, budget: &Budget) -> Pack;
    /// Step 4, one packet for the level above.
    fn emit(&mut self, pack: &Pack) -> Packet;
}
/// Resolve a card by reference (the Component Registry implements it).
pub trait Resolve: Send + Sync {
    /// # Errors
    /// Unknown reference or hash mismatch.
    fn resolve(&self, r: &PartRef) -> Result<Card, Fault>;
}
/// Construction from data: every component is built from its card and a config value.
pub trait FromCard: Sized {
    /// # Errors
    /// The card or config does not fit this component.
    fn from_card(card: &Card, cfg: &Value, reg: &dyn Resolve) -> Result<Self, Fault>;
}

/// Self-check of the `component` sub-component: names, descriptor, telemetry and fault
/// round-trip through JSON.
pub(crate) fn check() -> Result<(), String> {
    for t in ComponentType::ALL {
        if ComponentType::from_name(t.name()) != Some(*t) {
            return Err(format!("nmtype.component: name {} does not round-trip", t.name()));
        }
    }
    let d = ComponentDescriptor {
        id: "nmpod.unit#t3".into(),
        kind: ComponentType::Tentacle,
        version: 1,
        parent: Some("nmpod.pod#p0".into()),
        ports_in: vec![PortDesc::new("in", crate::port::PortKind::Packets, true)],
        ports_out: vec![],
        deps: vec!["nmbrain.rivers#b0".into()],
        card_hash: "00".into(),
    };
    if ComponentDescriptor::from_value(&d.to_value()).map_err(|e| e.to_string())? != d {
        return Err("nmtype.component: descriptor round trip".into());
    }
    let mut t = Telemetry::idle("nmpod.unit#t3");
    t.load = 0.25;
    t.latency_ms_avg = 1.5;
    if Telemetry::from_value(&t.to_value()).map_err(|e| e.to_string())? != t {
        return Err("nmtype.component: telemetry round trip".into());
    }
    let fault = Fault::new("x.y#0", FaultKind::Policy, "refused").unrecoverable();
    if Fault::from_value(&fault.to_value()).map_err(|e| e.to_string())? != fault {
        return Err("nmtype.component: fault round trip".into());
    }
    Ok(())
}
