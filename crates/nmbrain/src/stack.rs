//! The Core Brain's Context Stack as a Neuropod interface (guide 3, 6, 17.4; PLAN 4c, section 5):
//! plan (router slot picks jobs; the Neuropod cuts territories), forage (the NTE runs the
//! Neuropod: injected by the organism, since this crate does not depend on the executor),
//! reduce (packets through the river mailbox, arbitration, Tension Buffer, synthesis) and
//! emit (one cited packet for the level above, its text taken from the evidence by the
//! extractive writer).
use crate::arbiter;
use crate::contradiction::TensionBuffer;
use crate::rivers::Mailbox;
use crate::slot::Slots;
use crate::synthesis;
use nmjson::Value;
use nmtype::{
    Budget, Category, Component, ComponentDescriptor, ComponentType, Fault, FaultKind, FrameView, Neuropod, Pack, Packet, Plan, Provenance,
    Status, Task, Telemetry,
};
use std::collections::BTreeSet;
use std::sync::Arc;

/// How the organism runs the Neuropod for a plan (the NTE, wired in nmapi).
pub type Forager = Box<dyn FnMut(&Plan, &dyn FrameView) -> Vec<Packet> + Send>;

pub struct CoreBrain {
    descriptor: ComponentDescriptor,
    pub slots: Arc<Slots>,
    pub mailbox: Mailbox,
    pub buffer: TensionBuffer,
    pub verified: BTreeSet<String>,
    pub questions: Vec<Packet>,
    objective: String,
    version: u64,
    forager: Option<Forager>,
    telemetry: Telemetry,
}
impl std::fmt::Debug for CoreBrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoreBrain").field("objective", &self.objective).field("mailbox", &self.mailbox.len()).finish_non_exhaustive()
    }
}

impl CoreBrain {
    #[must_use]
    pub fn new(slots: Arc<Slots>) -> Self {
        let mut mailbox = Mailbox::default();
        for kind in ["symbol", "deps", "grep", "recall", "chunk", "history", "finding", "question", "invalidation"] {
            mailbox.subscribe(kind, "nmbrain.stack#b0");
        }
        Self {
            descriptor: ComponentDescriptor {
                id: "nmbrain.stack#b0".into(),
                kind: ComponentType::Brain,
                version: 1,
                parent: None,
                ports_in: Vec::new(),
                ports_out: Vec::new(),
                deps: Vec::new(),
                card_hash: String::new(),
            },
            slots,
            mailbox,
            buffer: TensionBuffer::default(),
            verified: BTreeSet::new(),
            questions: Vec::new(),
            objective: String::new(),
            version: 0,
            forager: None,
            telemetry: Telemetry::idle("nmbrain.stack#b0"),
        }
    }
    /// Give the brain the organism's way of running the Neuropod.
    pub fn set_forager(&mut self, f: Forager) {
        self.forager = Some(f);
    }
    /// The substrate version packets are read at (for invalidation and emit provenance).
    pub fn set_version(&mut self, v: u64) {
        if v > self.version {
            self.mailbox.invalidate(v);
        }
        self.version = v;
    }
}

impl Component for CoreBrain {
    fn descriptor(&self) -> &ComponentDescriptor {
        &self.descriptor
    }
    fn initialize(&mut self, _: &Value) -> Result<(), Fault> {
        Ok(())
    }
    fn suspend(&mut self) -> Result<(), Fault> {
        Ok(())
    }
    fn resume(&mut self) -> Result<(), Fault> {
        Ok(())
    }
    fn shutdown(&mut self) {
        let _ = self.mailbox.take_all();
    }
    fn inspect_telemetry(&self) -> Telemetry {
        self.telemetry.clone()
    }
    fn check(&mut self) -> Result<(), Fault> {
        check().map_err(|e| Fault::new("nmbrain.stack", FaultKind::Execution, e))
    }
}

impl Neuropod for CoreBrain {
    fn plan(&mut self, task: &Task) -> Plan {
        self.objective.clone_from(&task.objective);
        Plan { task: task.id, regions: Vec::new(), jobs: self.slots.route(&task.objective), views: vec![task.objective.clone()] }
    }
    fn forage(&mut self, plan: &Plan, frame: &dyn FrameView) -> Vec<Packet> {
        match self.forager.as_mut() {
            Some(f) => f(plan, frame),
            None => Vec::new(),
        }
    }
    fn reduce(&mut self, packets: Vec<Packet>, budget: &Budget) -> Pack {
        for p in packets {
            self.mailbox.publish(p);
        }
        let all = self.mailbox.take_all();
        let (questions, claims): (Vec<Packet>, Vec<Packet>) = all.into_iter().partition(|p| p.kind() == "question");
        let labelled = arbiter::arbitrate(&claims, &self.verified);
        let mut asked = self.buffer.hold(&labelled, self.version);
        asked.extend(questions);
        self.questions = asked;
        self.telemetry.executions += 1;
        synthesis::reduce(0, &self.objective, &labelled, budget.tokens, &self.slots)
    }
    fn emit(&mut self, pack: &Pack) -> Packet {
        self.emit_inner(pack)
    }
}

impl CoreBrain {
    /// `reduce` on the new path (2026-11-10, refinement step 1): the same mailbox, arbiter and
    /// Tension Buffer, then `synthesis::reduce_evidence` with the question's evidence set (tentacle
    /// chunks ordered by their records; what the floor or the token room would drop is held low
    /// in the set's queue).
    pub fn reduce_evidence(&mut self, packets: Vec<Packet>, budget: &Budget, set: &mut crate::evidence::EvidenceSet) -> Pack {
        for p in packets {
            self.mailbox.publish(p);
        }
        let all = self.mailbox.take_all();
        let (questions, claims): (Vec<Packet>, Vec<Packet>) = all.into_iter().partition(|p| p.kind() == "question");
        let labelled = arbiter::arbitrate(&claims, &self.verified);
        let mut asked = self.buffer.hold(&labelled, self.version);
        asked.extend(questions);
        self.questions = asked;
        self.telemetry.executions += 1;
        synthesis::reduce_evidence(0, &self.objective, &labelled, budget.tokens, &self.slots, set)
    }
    fn emit_inner(&mut self, pack: &Pack) -> Packet {
        // What a packet carries for the judge only (a piece's fields) is neither emitted nor
        // written.
        let shown = |e: &&String| !e.starts_with(synthesis::PIECE);
        let mut evidence: Vec<String> = Vec::new();
        for e in pack.packets.iter().flat_map(|p| p.get_evidence().iter()).filter(shown) {
            if !evidence.contains(e) {
                evidence.push(e.clone());
            }
        }
        let text = if pack.withheld {
            pack.reason.clone()
        } else {
            // The writer reads the evidence of the three best packets, in the pack's order.
            let lead: Vec<String> = pack.packets.iter().take(3).flat_map(|p| p.get_evidence().iter().filter(shown).cloned()).collect();
            let written = self.slots.by_name.get("writer").map(|w| w.control.write(&self.objective, &lead)).unwrap_or_default();
            // M1 revision 2: the writer keeps only lines sharing the question's words, which a
            // reworded question's recalled line may not; a recalled line among the three best is
            // also allowed. Lines follow the pack's order, at most three.
            let recalled: Vec<&String> =
                pack.packets.iter().take(3).filter(|p| p.kind() == "recall").filter_map(|p| p.get_evidence().first()).collect();
            // 2026-11-13: the first line is the Core Brain's first place (the first packet's first
            // evidence line); the writer's word-sharing lines follow. Its word filter had put a
            // later packet's line first on 2 of 24 development behaviour questions, against the
            // reranker's and the second look's judgment.
            let mut out: Vec<&String> = pack.packets.first().and_then(|p| p.get_evidence().first()).into_iter().collect();
            for l in &lead {
                if (written.lines().any(|w| w == l) || recalled.contains(&l)) && !out.contains(&l) {
                    out.push(l);
                }
            }
            // CM1 revision 2 (2026-11-09): an answer always names what its best packets cite.
            // Lines sharing no word with the question (a source chunk's `file:1 lines 1..30`)
            // were dropped, so many answers listed nothing; the writer's lines follow the first line.
            for l in &lead {
                if out.len() >= 3 {
                    break;
                }
                if !out.contains(&l) {
                    out.push(l);
                }
            }
            out.truncate(3);
            let lines = out.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(
                "
",
            );
            format!("{}\n{lines}", pack.reason)
        };
        let status = if pack.withheld {
            Status::Unknown
        } else if !pack.floor_met {
            Status::Inferred
        } else if pack.packets.iter().any(|p| p.get_status() == Status::Contradicted) {
            Status::Contradicted
        } else {
            Status::Observed
        };
        let mut out = Packet::new(
            "pack",
            &format!("answer:{}", self.objective),
            &text,
            Provenance::new("unit:nmbrain.stack#b0", "nmbrain.stack#b0", self.version, nmtype::now_ms()),
        )
        .status(status)
        .category(Category::Fact)
        .confidence(grounded_share(pack))
        .evidence(evidence);
        for p in &pack.packets {
            out = out.relate("cites", p.id());
        }
        out
    }
}

/// Share of a pack's packets that are observed, verified or observed in conflict.
fn grounded_share(pack: &Pack) -> f32 {
    if pack.packets.is_empty() {
        return 0.0;
    }
    let g = pack.packets.iter().filter(|p| matches!(p.get_status(), Status::Observed | Status::Verified | Status::Contradicted)).count();
    #[expect(clippy::cast_precision_loss, reason = "pack sizes are small")]
    let share = g as f32 / pack.packets.len() as f32;
    share
}

/// Self-check of the `stack` sub-component: with a fixture forager, a question with two
/// observed values comes out answered with both kept and a conflict question asked; a
/// decoy comes out withheld.
pub(crate) fn check() -> Result<(), String> {
    let mut brain = CoreBrain::new(Arc::new(Slots::default()));
    brain.set_forager(Box::new(|_: &Plan, _: &dyn FrameView| {
        let mk = |t: &str, s: &str| {
            Packet::new("symbol", "value:LIMIT", t, Provenance::new(s, "fixture", 1, 0))
                .status(Status::Observed)
                .evidence(vec![format!("{s} const LIMIT")])
        };
        vec![mk("LIMIT = 4", "a.rs:1"), mk("LIMIT = 9", "b.rs:1")]
    }));
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
    let b = Budget { tokens: 4000, ms: 5000, depth: 1, units: 16 };
    for (q, want_withheld) in [("What is LIMIT set to?", false), ("What is FANOUT_CEILING set to?", true)] {
        let task = Task { id: 1, objective: q.into(), budget: b, scope: String::new(), parent: None, kind: nmtype::TaskKind::Context };
        let plan = brain.plan(&task);
        let packets = brain.forage(&plan, &F(b));
        let pack = brain.reduce(packets, &b);
        let out = brain.emit(&pack);
        if pack.withheld != want_withheld {
            return Err(format!("nmbrain.stack: {q} withheld={} ({})", pack.withheld, pack.reason));
        }
        if !want_withheld && (pack.packets.len() != 2 || out.get_status() != Status::Contradicted || brain.questions.is_empty()) {
            return Err("nmbrain.stack: conflict not kept or not asked about".into());
        }
    }
    Ok(())
}
