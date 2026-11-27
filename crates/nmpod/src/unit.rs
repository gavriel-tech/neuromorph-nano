//! The one unit type for tentacles and drone tips (guide 4.1, 4.2, 12; PLAN 4c): same code,
//! any job. A run has three phases, which the NTE can mount as separate nodes: `assign`
//! (tentacle: take the plan, pick the job and territory), `sense` (drone tip: run readers),
//! `process` (pipeline: reflexes, confidence, anomaly, working note, distil, recursion). A
//! unit holds only its own state and the shared reflex table; no handle to another unit
//! exists, which `to_value` shows field by field.
//!
//! 2026-11-10 (step P4 first form, one tentacle system; specification 2.2 and 2.3): an
//! assignment may carry the tentacle's own `Window` (laid out over the whole code base by a lens
//! for this question) in place of a territory. Then the drone tip runs the window-bound readers
//! inside the window and the whole-code-base readers dealt to this tentacle for the question
//! (`whole`), over the whole tree; further readers (the attention-head drone readers of a bound
//! neural core) join through `sense_readers`. An assignment without a window behaves exactly as
//! before (the served control path while `--one-tentacle` is off).
use crate::pipeline::{self, Reflex, ReflexTable, Signals};
use crate::recursion;
use crate::tip::{self, Finding, Reach, ReadCtx, Reader as _};
use crate::window::Window;
use nmjson::Value;
use nmtype::{
    Component, ComponentDescriptor, ComponentType, Fault, FaultKind, Health, Packet, PortDesc, PortKind, Region, SlotView, SubstrateView,
    Task, Telemetry,
};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Tentacle or drone tip: a placement, not a capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Tentacle,
    Drone,
}

/// The twelve supported jobs; any unit can take any job (no permanent assignment).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Job {
    Structural,
    Functional,
    Dependency,
    Execution,
    Semantic,
    Historical,
    Constraint,
    Predictive,
    Verification,
    Reconciliation,
    Retrieval,
    Reasoning,
}
impl Job {
    pub const ALL: [Self; 12] = [
        Self::Structural,
        Self::Functional,
        Self::Dependency,
        Self::Execution,
        Self::Semantic,
        Self::Historical,
        Self::Constraint,
        Self::Predictive,
        Self::Verification,
        Self::Reconciliation,
        Self::Retrieval,
        Self::Reasoning,
    ];
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Structural => "structural",
            Self::Functional => "functional",
            Self::Dependency => "dependency",
            Self::Execution => "execution",
            Self::Semantic => "semantic",
            Self::Historical => "historical",
            Self::Constraint => "constraint",
            Self::Predictive => "predictive",
            Self::Verification => "verification",
            Self::Reconciliation => "reconciliation",
            Self::Retrieval => "retrieval",
            Self::Reasoning => "reasoning",
        }
    }
    #[must_use]
    pub fn from_name(t: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|j| j.name() == t)
    }
    /// The readers a job adds to the base readers every unit always runs (symbols, lexical,
    /// semantic since 2026-11-08), so a territory is never left without definitions or its
    /// nearest code whatever job it drew; a reader named twice runs once. Escalation may add
    /// the slot reader. Since 2026-11-09 Dependency, Verification and Reasoning have readers
    /// of their own (before, six of the eight default jobs ran exactly the base readers).
    #[must_use]
    pub const fn readers(self) -> &'static [&'static str] {
        match self {
            Self::Structural | Self::Retrieval | Self::Semantic | Self::Predictive | Self::Reconciliation => &[],
            Self::Constraint => &["rules"],
            Self::Functional => &["interface"],
            Self::Dependency => &["uses"],
            Self::Verification => &["tests"],
            Self::Reasoning => &["deep"],
            Self::Historical => &["git"],
            Self::Execution => &["build"],
        }
    }
    /// Jobs that answer a kind of question over the whole tree: when the router picks one
    /// for a question, every tentacle carries it in its own territory (2026-11-09; before,
    /// each job read one territory only, so a history question read the history of one
    /// eighth of the files).
    pub const FOCUS: [Self; 6] =
        [Self::Dependency, Self::Historical, Self::Execution, Self::Verification, Self::Constraint, Self::Functional];
}

/// One unit's share of a task: job, territory, the task itself, and the focus jobs the
/// router picked for the question, carried by every unit in its own territory. In the one
/// tentacle system (2026-11-10) the unit's own window replaces the territory (`regions` is then
/// empty) and `whole` names the whole-code-base readers this tentacle carries for the question.
#[derive(Clone, Debug, PartialEq)]
pub struct Assignment {
    pub job: Job,
    pub regions: Vec<Region>,
    pub task: Task,
    pub focus: Vec<Job>,
    pub window: Option<Window>,
    pub whole: Vec<String>,
}
impl Assignment {
    /// A territory assignment (the control path; no window).
    #[must_use]
    pub fn territory(job: Job, regions: Vec<Region>, task: Task, focus: Vec<Job>) -> Self {
        Self { job, regions, task, focus, window: None, whole: Vec::new() }
    }
    /// The window the exact readers read in: the window when the assignment gives no territory.
    /// With both (the exact readers keep their territory reach, `nmapi::tentacle_unit`), the
    /// window is what the tentacle's drones and core read and the exact readers read as before.
    #[must_use]
    pub fn window_reach(&self) -> Option<&Window> {
        self.window.as_ref().filter(|_| self.regions.is_empty())
    }
    /// The scope a unit's packets carry: its territory paths, or its window's lens.
    #[must_use]
    pub fn scope(&self) -> String {
        match self.window_reach() {
            Some(w) => format!("window:t{}:{}", w.tentacle, w.lens),
            None => format!("territory:{}", self.paths().join(",")),
        }
    }
    #[must_use]
    pub fn paths(&self) -> Vec<String> {
        self.regions.iter().map(|r| r.path.clone()).collect()
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        Value::Object(
            [
                ("job".to_owned(), Value::String(self.job.name().into())),
                ("regions".to_owned(), Value::Array(self.regions.iter().map(Region::to_value).collect())),
                ("task".to_owned(), self.task.to_value()),
                ("focus".to_owned(), Value::Array(self.focus.iter().map(|j| Value::String(j.name().into())).collect())),
            ]
            .into_iter()
            .chain(self.window.iter().flat_map(|w| {
                [
                    ("window".to_owned(), w.to_value()),
                    ("whole".to_owned(), Value::Array(self.whole.iter().map(|r| Value::String(r.clone())).collect())),
                ]
            }))
            .collect(),
        )
    }
    /// # Errors
    /// Missing field or unknown job.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let bad = |e: String| Fault::new("nmpod.unit", FaultKind::Interface, e);
        let job = v.get("job").and_then(Value::string).map_err(|e| bad(e.to_string()))?;
        Ok(Self {
            job: Job::from_name(job).ok_or_else(|| bad(format!("unknown job {job}")))?,
            regions: v
                .get("regions")
                .and_then(Value::array)
                .map_err(|e| bad(e.to_string()))?
                .iter()
                .map(Region::from_value)
                .collect::<Result<_, _>>()?,
            task: Task::from_value(v.get("task").map_err(|e| bad(e.to_string()))?)?,
            focus: v
                .get("focus")
                .and_then(Value::array)
                .map(|a| a.iter().filter_map(|j| j.string().ok().and_then(Job::from_name)).collect())
                .unwrap_or_default(),
            window: match v.get("window") {
                Ok(w) => Some(Window::from_value(w)?),
                Err(_) => None,
            },
            whole: v
                .get("whole")
                .and_then(Value::array)
                .map(|a| a.iter().filter_map(|r| r.string().ok().map(str::to_owned)).collect())
                .unwrap_or_default(),
        })
    }
}

/// The whole-code-base readers of one question dealt to the tentacles (one tentacle system,
/// 2026-11-10): a tentacle carrying a job runs that job's whole-code-base readers; the base
/// whole-code-base readers and the focus jobs' readers not carried that way are dealt in turn
/// (`window::deal`, from `offset`). Each reader is carried once, so the whole tree is read once by
/// it and nothing is doubled across tentacles.
#[must_use]
pub fn plan_whole(jobs: &[Job], focus: &[Job], tentacles: usize, offset: usize) -> Vec<Vec<String>> {
    let n = tentacles.max(1);
    let mut out: Vec<Vec<String>> = vec![Vec::new(); n];
    let mut carried: Vec<&str> = Vec::new();
    for (slot, t) in out.iter_mut().enumerate() {
        let job = jobs.get(slot % jobs.len().max(1)).copied().unwrap_or(Job::Retrieval);
        for r in job.readers().iter().filter(|r| tip::reach_of(r) == Reach::Whole) {
            if !carried.contains(r) {
                carried.push(r);
                t.push((*r).to_owned());
            }
        }
    }
    let rest: Vec<&str> = BASE_READERS
        .iter()
        .copied()
        .chain(focus.iter().filter(|j| Job::FOCUS.contains(j)).flat_map(|j| j.readers().iter().copied()))
        .filter(|r| tip::reach_of(r) == Reach::Whole && !carried.contains(r))
        .collect();
    for (t, extra) in out.iter_mut().zip(crate::window::deal(&rest, n, offset)) {
        t.extend(extra);
    }
    out
}

/// What a unit hands up: cited packets, barrier questions, an optional subordinate request.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub packets: Vec<Packet>,
    pub questions: Vec<Packet>,
    pub subordinate: Option<Task>,
    pub telemetry: Telemetry,
}

/// How many different readers found something.
fn distinct_readers(found: &[Finding]) -> usize {
    let mut r: Vec<&str> = found.iter().map(|f| f.reader.as_str()).collect();
    r.sort_unstable();
    r.dedup();
    r.len()
}

/// Readers every unit runs whatever its job. The semantic reader joined the base set on
/// 2026-11-08 (context pack fix): with it only in three of the eight jobs, five territories
/// were never read for their nearest code and packs on them came back empty.
pub const BASE_READERS: [&str; 3] = ["symbols", "lexical", "semantic"];

/// Working-memory cap per unit (PLAN 4c nmmem working cap 32).
const WORKING_CAP: usize = 32;

/// A tentacle or a drone tip.
#[derive(Debug)]
pub struct Unit {
    descriptor: ComponentDescriptor,
    pub level: Level,
    pub job: Job,
    pub depth: u8,
    pub max_depth: u8,
    working: Vec<String>,
    reflexes: Arc<ReflexTable>,
    telemetry: Telemetry,
}

impl Unit {
    #[must_use]
    pub fn new(id: &str, level: Level, job: Job, reflexes: Arc<ReflexTable>, max_depth: u8) -> Self {
        let kind = if level == Level::Tentacle { ComponentType::Tentacle } else { ComponentType::Drone };
        Self {
            descriptor: ComponentDescriptor {
                id: id.to_owned(),
                kind,
                version: 1,
                parent: None,
                ports_in: vec![PortDesc::new("plan", PortKind::Json, true)],
                ports_out: vec![PortDesc::new("packets", PortKind::Packets, true)],
                deps: Vec::new(),
                card_hash: String::new(),
            },
            level,
            job,
            depth: 0,
            max_depth,
            working: Vec::new(),
            reflexes,
            telemetry: Telemetry::idle(id),
        }
    }
    #[must_use]
    pub fn id(&self) -> &str {
        &self.descriptor.id
    }
    /// Tentacle phase: this unit's assignment from the plan's territories (`territory`
    /// index = this unit's slot), or `None` when the plan gives it no region.
    #[must_use]
    pub fn assign(&self, task: &Task, territories: &[Vec<Region>], slot: usize, jobs: &[Job], focus: &[Job]) -> Option<Assignment> {
        let regions = territories.get(slot)?.clone();
        if regions.is_empty() {
            return None;
        }
        let job = jobs.get(slot % jobs.len().max(1)).copied().unwrap_or(self.job);
        let focus = focus.iter().copied().filter(|j| Job::FOCUS.contains(j)).collect();
        Some(Assignment::territory(job, regions, task.clone(), focus))
    }
    /// Tentacle phase in the one tentacle system: this unit's own window for the question (its
    /// lens is a temporary specialisation), its job, the focus jobs and the whole-code-base
    /// readers it carries. `None` when the slot has no window.
    #[must_use]
    #[expect(clippy::too_many_arguments, reason = "one assignment's parts, passed once")]
    pub fn assign_window(
        &self,
        task: &Task,
        windows: &[Window],
        slot: usize,
        jobs: &[Job],
        focus: &[Job],
        whole: &[Vec<String>],
    ) -> Option<Assignment> {
        self.assign_window_in(task, windows, Vec::new(), slot, jobs, focus, whole)
    }
    /// `assign_window` with a territory kept as the exact readers' reach (`regions`, empty for the
    /// window reach): the tentacle still owns its window (drones, core, report), and its exact
    /// readers read exactly as in the territory path.
    #[must_use]
    #[expect(clippy::too_many_arguments, reason = "one assignment's parts, passed once")]
    pub fn assign_window_in(
        &self,
        task: &Task,
        windows: &[Window],
        regions: Vec<Region>,
        slot: usize,
        jobs: &[Job],
        focus: &[Job],
        whole: &[Vec<String>],
    ) -> Option<Assignment> {
        let window = windows.get(slot)?.clone();
        let job = jobs.get(slot % jobs.len().max(1)).copied().unwrap_or(self.job);
        let focus = focus.iter().copied().filter(|j| Job::FOCUS.contains(j)).collect();
        Some(Assignment {
            job,
            regions,
            task: task.clone(),
            focus,
            window: Some(window),
            whole: whole.get(slot).cloned().unwrap_or_default(),
        })
    }
    /// The exact readers a drone tip runs for an assignment, by name. Territory: the base
    /// readers, the job's readers and the focus jobs' readers over the territory. Window: the
    /// window-bound ones of those inside the window, plus the whole-code-base readers dealt to
    /// this tentacle. A reader named twice runs once.
    #[must_use]
    pub fn reader_names(a: &Assignment) -> Vec<&str> {
        let mut names: Vec<&str> = BASE_READERS.to_vec();
        for r in a.job.readers().iter().chain(a.focus.iter().flat_map(|j| j.readers().iter())) {
            if !names.contains(r) {
                names.push(r);
            }
        }
        if a.window_reach().is_some() {
            names.retain(|r| tip::reach_of(r) == Reach::Window);
            for r in &a.whole {
                if !names.contains(&r.as_str()) {
                    names.push(r.as_str());
                }
            }
        }
        names
    }
    /// Drone tip phase: the base readers, the job's readers and the focus jobs' readers over
    /// the territory (or the window and the whole code base, `reader_names`); a reader named
    /// twice runs once.
    #[must_use]
    pub fn sense(&self, a: &Assignment, sub: &dyn SubstrateView, slots: &dyn SlotView) -> (Vec<Finding>, usize) {
        let names = Self::reader_names(a);
        self.sense_with(a, &names, sub, slots)
    }
    /// Drone tip phase with named readers (a subordinate drone's deeper reading).
    #[must_use]
    pub fn sense_with(&self, a: &Assignment, names: &[&str], sub: &dyn SubstrateView, slots: &dyn SlotView) -> (Vec<Finding>, usize) {
        let readers: Vec<Box<dyn tip::Reader>> = names.iter().filter_map(|r| tip::reader(r)).collect();
        self.sense_readers(a, &readers, sub, slots)
    }
    /// Drone tip phase with given readers behind the one reader interface (exact readers and,
    /// when a neural core is bound, its attention-head drone readers).
    #[must_use]
    pub fn sense_readers(
        &self,
        a: &Assignment,
        readers: &[Box<dyn tip::Reader>],
        sub: &dyn SubstrateView,
        slots: &dyn SlotView,
    ) -> (Vec<Finding>, usize) {
        let paths = a.paths();
        let ctx = ReadCtx { sub, slots, query: &a.task.objective, paths: &paths, window: a.window_reach() };
        let found = readers.iter().flat_map(|r| r.scan(&ctx)).collect();
        (found, readers.len())
    }
    /// Pipeline phase: reflexes (escalation runs the slot reader once), confidence,
    /// anomaly, a working note, distillation, and a subordinate request when warranted.
    pub fn process(
        &mut self,
        a: &Assignment,
        findings: Vec<Finding>,
        readers_run: usize,
        sub: &dyn SubstrateView,
        slots: &dyn SlotView,
    ) -> Outcome {
        let started = std::time::Instant::now();
        let level = if self.level == Level::Tentacle { "tentacle" } else { "drone" };
        let signals = Signals { territory_empty: findings.is_empty(), readers_disagree: pipeline::disagree(&findings) };
        let mut kept = Vec::new();
        let mut questions = Vec::new();
        let mut escalate = false;
        let probe = Finding {
            key: String::new(),
            kind: "finding".into(),
            text: String::new(),
            source: String::new(),
            evidence: Vec::new(),
            reader: String::new(),
            score: 1.0,
            status: nmtype::Status::Unknown,
        };
        let list = if findings.is_empty() { vec![probe] } else { findings };
        for f in list {
            match self.reflexes.decide_now(level, &f, signals) {
                Reflex::Keep if !f.key.is_empty() => kept.push(f),
                Reflex::Keep | Reflex::Drop => {}
                Reflex::Escalate(_) => {
                    escalate = true;
                    if !f.key.is_empty() {
                        kept.push(f);
                    }
                }
                Reflex::Hold(why) => questions.push(nmtype::Packet::new(
                    "question",
                    &format!("hold:{}", self.id()),
                    &format!("{}: {why}", self.id()),
                    nmtype::Provenance::new(&format!("unit:{}", self.id()), self.id(), sub.version(), nmtype::now_ms()),
                )),
            }
        }
        if escalate {
            let paths = a.paths();
            let ctx = ReadCtx { sub, slots, query: &a.task.objective, paths: &paths, window: a.window_reach() };
            kept.extend(tip::Slot.scan(&ctx));
        }
        let conf = pipeline::confidence(&kept, readers_run);
        let anomaly = signals.readers_disagree || signals.territory_empty;
        let packets = pipeline::distill(&kept, &self.descriptor.id, sub.version(), conf, anomaly, &a.scope());
        self.note(&format!("task {}: {} packets, confidence {conf:.2}", a.task.id, packets.len()));
        // A weak territory (findings from only one of three or more readers, or readers that
        // disagree) asks the brain for one subordinate drone with a deeper reading of the same
        // territory (2026-11-09: before, only disagreement under one key asked, which almost
        // never happens, and the live path never served the request).
        let weak = signals.readers_disagree || (readers_run >= 3 && distinct_readers(&kept) == 1);
        let mut scoped = a.task.clone();
        scoped.scope = if a.window_reach().is_some() { a.scope() } else { a.paths().join(",") };
        let subordinate =
            if packets.is_empty() { None } else { recursion::should_spawn(&scoped, self.depth, self.max_depth, weak, &[], 0) };
        if let Some(s) = &subordinate {
            questions.push(recursion::request(self.id(), s, sub.version()));
        }
        let ms = started.elapsed().as_secs_f32() * 1000.0;
        let t = &mut self.telemetry;
        #[expect(clippy::cast_precision_loss, reason = "execution counts stay small")]
        let n = t.executions as f32;
        t.latency_ms_avg = (t.latency_ms_avg * n + ms) / (n + 1.0);
        t.confidence_avg = (t.confidence_avg * n + conf) / (n + 1.0);
        t.executions += 1;
        t.anomalies += u64::from(anomaly);
        t.memory_bytes = self.working.iter().map(String::len).sum();
        Outcome { packets, questions, subordinate, telemetry: self.telemetry.clone() }
    }
    /// All three phases in one call (drone-level recursion, checks).
    pub fn run(&mut self, a: &Assignment, sub: &dyn SubstrateView, slots: &dyn SlotView) -> Outcome {
        let (found, n) = self.sense(a, sub, slots);
        self.process(a, found, n, sub, slots)
    }
    fn note(&mut self, text: &str) {
        self.working.push(text.to_owned());
        if self.working.len() > WORKING_CAP {
            self.working.remove(0);
        }
    }
    /// Every field of the unit, by name (the no-handle check reads this).
    #[must_use]
    pub fn to_value(&self) -> Value {
        let s = |t: &str| Value::String(t.to_owned());
        let fields: BTreeMap<String, Value> = [
            ("descriptor", self.descriptor.to_value()),
            ("level", s(if self.level == Level::Tentacle { "tentacle" } else { "drone" })),
            ("job", s(self.job.name())),
            ("depth", Value::Number(self.depth.to_string())),
            ("max_depth", Value::Number(self.max_depth.to_string())),
            ("working", Value::Array(self.working.iter().map(|w| s(w)).collect())),
            ("reflexes", Value::Number(self.reflexes.rows.len().to_string())),
            ("telemetry", self.telemetry.to_value()),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect();
        Value::Object(fields)
    }
}

impl Component for Unit {
    fn descriptor(&self) -> &ComponentDescriptor {
        &self.descriptor
    }
    fn initialize(&mut self, cfg: &Value) -> Result<(), Fault> {
        if let Ok(j) = cfg.get("job").and_then(Value::string) {
            self.job = Job::from_name(j).ok_or_else(|| Fault::new(self.id(), FaultKind::Config, format!("unknown job {j}")))?;
        }
        Ok(())
    }
    fn suspend(&mut self) -> Result<(), Fault> {
        self.telemetry.health = Health::Degraded;
        Ok(())
    }
    fn resume(&mut self) -> Result<(), Fault> {
        self.telemetry.health = Health::Healthy;
        Ok(())
    }
    fn shutdown(&mut self) {
        self.working.clear();
        self.telemetry.health = Health::Unavailable;
    }
    fn inspect_telemetry(&self) -> Telemetry {
        self.telemetry.clone()
    }
    fn check(&mut self) -> Result<(), Fault> {
        check().map_err(|e| Fault::new(self.id(), FaultKind::Execution, e))
    }
}

/// Self-check of the `unit` sub-component: a drone finds the fixture constant in its
/// territory, the same type runs every job, and its fields hold no handle to another unit.
pub(crate) fn check() -> Result<(), String> {
    let (sub, slots) = crate::fixture::views();
    let table = Arc::new(ReflexTable::default());
    let task = Task {
        id: 1,
        objective: "What is LIMIT?".into(),
        budget: nmtype::Budget { tokens: 100, ms: 1000, depth: 1, units: 2 },
        scope: String::new(),
        parent: None,
        kind: nmtype::TaskKind::Context,
    };
    let region = Region { territory: 0, path: "a/".into(), start: 0, end: 1, priority: 1.0 };
    for job in Job::ALL {
        let mut u = Unit::new("nmpod.unit#d0", Level::Drone, job, Arc::clone(&table), 1);
        let a = Assignment::territory(job, vec![region.clone()], task.clone(), Vec::new());
        let _ = u.run(&a, &sub, &slots);
    }
    let mut u = Unit::new("nmpod.unit#d0", Level::Drone, Job::Structural, Arc::clone(&table), 1);
    let out = u.run(&Assignment::territory(Job::Structural, vec![region], task.clone(), Vec::new()), &sub, &slots);
    if !out.packets.iter().any(|p| p.key() == "value:LIMIT" && p.text() == "LIMIT = 4") {
        return Err(format!("nmpod.unit: drone missed LIMIT: {:?}", out.packets.iter().map(Packet::key).collect::<Vec<_>>()));
    }
    let fields = u.to_value();
    let names: Vec<&String> = fields.object().map_err(|e| e.to_string())?.keys().collect();
    let allowed = ["depth", "descriptor", "job", "level", "max_depth", "reflexes", "telemetry", "working"];
    if names.iter().any(|n| !allowed.contains(&n.as_str())) {
        return Err(format!("nmpod.unit: unexpected field in a unit: {names:?}"));
    }
    // One tentacle system: a window assignment round-trips through JSON, its tip runs only
    // window-bound readers plus the whole-code-base readers dealt to it, and it finds the
    // constant inside its window while the dealt symbol reader reads the whole tree.
    let span = crate::window::Span { file: "b/y.rs".into(), start: 1, end: 1, key: "k".into(), tokens: 5 };
    let w = Window { tentacle: 0, lens: "words".into(), budget: 50, spans: vec![span] };
    let jobs = [Job::Structural, Job::Historical];
    let whole = plan_whole(&jobs, &[Job::Historical], 2, 0);
    if whole != vec![vec!["symbols".to_owned()], vec!["git".to_owned()]] {
        return Err(format!("nmpod.unit: whole-code-base readers dealt {whole:?}"));
    }
    let a = u.assign_window(&task, std::slice::from_ref(&w), 0, &jobs, &[], &whole).ok_or("nmpod.unit: no window assignment")?;
    if Assignment::from_value(&a.to_value()).map_err(|e| e.to_string())? != a
        || Unit::reader_names(&a) != ["lexical", "semantic", "symbols"]
    {
        return Err(format!("nmpod.unit: window assignment {:?}", Unit::reader_names(&a)));
    }
    let out = u.run(&a, &sub, &slots);
    let sources: Vec<String> = out.packets.iter().map(|p| p.provenance().source.clone()).collect();
    if !sources.iter().any(|s| s == "a/x.rs:1")
        || !sources.iter().any(|s| s == "b/y.rs:1")
        || out.packets.iter().any(|p| p.get_scope() != "window:t0:words")
    {
        return Err(format!("nmpod.unit: window run {sources:?}"));
    }
    Ok(())
}
