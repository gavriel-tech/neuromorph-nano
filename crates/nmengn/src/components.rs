//! Neural components (PLAN 4c `components`, 4d item 12): every node of a bound assembly is a
//! `Component` with its own descriptor, its own clusters and its own telemetry; there is no
//! Block type. Single-input components (norm, linear, head) are also `TensorProcessor`s, so
//! one can be mounted wherever a tensor port fits (inside a tentacle, for example); the
//! multi-port ones (rope, score, add, swigate) run inside an assembly, which feeds their ports.
use crate::model::{Act, Model, RunOpts};
use crate::placement::Device;
use nmjson::Value;
use nmtype::{
    Component, ComponentDescriptor, ComponentType, Fault, FaultKind, FrameView, Health, PortDesc, PortKind, Telemetry, Tensor, TensorData,
    TensorProcessor,
};
use std::sync::Arc;
use std::time::Instant;

/// One node of a bound assembly as a component.
pub struct Neural {
    model: Arc<Model>,
    index: usize,
    desc: ComponentDescriptor,
    suspended: bool,
    calls: u64,
    ms: f64,
}
impl Neural {
    /// The component for node `index` of `model`.
    /// # Errors
    /// No such node.
    pub fn new(model: &Arc<Model>, index: usize) -> Result<Self, Fault> {
        let b = model.bound.get(index).ok_or_else(|| Fault::new("nmengn.components", FaultKind::Config, format!("no node {index}")))?;
        let kind = match b.node.op.as_str() {
            "norm" | "headnorm" => ComponentType::Norm,
            "linear" | "rope" | "score" => ComponentType::Attention,
            "swigate" => ComponentType::FeedForward,
            "head" => ComponentType::Head,
            _ => ComponentType::Other,
        };
        let desc = ComponentDescriptor {
            id: b.node.id.clone(),
            kind,
            version: 1,
            parent: Some(format!("nmbrain.slot#{}", model.asm.name)),
            ports_in: vec![PortDesc::new("x", PortKind::Tensor, true)],
            ports_out: vec![PortDesc::new("y", PortKind::Tensor, true)],
            deps: b.views.iter().map(|v| v.cluster.id.clone()).collect(),
            card_hash: String::new(),
        };
        Ok(Self { model: Arc::clone(model), index, desc, suspended: false, calls: 0, ms: 0.0 })
    }
}
impl Component for Neural {
    fn descriptor(&self) -> &ComponentDescriptor {
        &self.desc
    }
    fn initialize(&mut self, _cfg: &Value) -> Result<(), Fault> {
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
        t.memory_bytes =
            self.model.bound.get(self.index).map_or(0, |b| b.views.iter().map(|v| usize::try_from(v.cluster.bytes).unwrap_or(0)).sum());
        t.health = if self.suspended { Health::Unavailable } else { Health::Healthy };
        t
    }
    fn check(&mut self) -> Result<(), Fault> {
        if self.model.bound.get(self.index).is_none() {
            return Err(Fault::new(&self.desc.id, FaultKind::Integrity, "node left its assembly"));
        }
        Ok(())
    }
}
impl TensorProcessor for Neural {
    /// Run this one component on the CPU over `x` (`[tokens, width]`, f32 values).
    fn forward(&mut self, x: &Tensor, _frame: &dyn FrameView) -> Result<Tensor, Fault> {
        if self.suspended {
            return Err(Fault::new(&self.desc.id, FaultKind::Policy, "suspended"));
        }
        let b = self.model.bound.get(self.index).ok_or_else(|| Fault::new(&self.desc.id, FaultKind::Integrity, "no node"))?;
        if !matches!(b.node.op.as_str(), "norm" | "headnorm" | "linear" | "head") {
            return Err(Fault::new(
                &self.desc.id,
                FaultKind::Interface,
                format!("{} has several ports; it runs inside an assembly", b.node.op),
            ));
        }
        let TensorData::Owned(v) = &x.data else {
            return Err(Fault::new(&self.desc.id, FaultKind::Interface, "owned f32 input required"));
        };
        let tokens = x.shape.first().copied().unwrap_or(1);
        let (f, fp) =
            self.model.asm.producer(&b.node.id, "x").ok_or_else(|| Fault::new(&self.desc.id, FaultKind::Interface, "input port unfed"))?;
        let port = if b.node.op == "head" { "logits" } else { "y" };
        let t0 = Instant::now();
        let opts = RunOpts { all_positions: true, ..RunOpts::default() };
        let out = self.model.run(
            Device::Cpu,
            vec![(f.to_owned(), fp.to_owned(), Act::Cpu(v.clone()))],
            &[(b.node.id.clone(), port.to_owned())],
            tokens,
            &opts,
        )?;
        self.calls += 1;
        self.ms += t0.elapsed().as_secs_f64() * 1000.0;
        let y = out.into_values().next().ok_or_else(|| Fault::new(&self.desc.id, FaultKind::Execution, "no output"))?;
        let y = crate::model::values(&y)?.to_vec();
        let width = y.len() / tokens.max(1);
        Ok(Tensor { shape: vec![tokens, width], dtype: nmtype::DType::F32, data: TensorData::Owned(y) })
    }
}
