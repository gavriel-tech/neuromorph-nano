//! MicroModel chains of the Core Brain (guide 4.1; plan step P7, first form, 2026-11-10).
//! Designs: Luis Jake Gabriel III, Gavriel Technologies.
//!
//! A request is processed by a chain of MicroModel nodes, never one monolithic pass. A chain
//! is data (`designs/chains.json`): an ordered list of nodes, each naming its role, the
//! fields of its controlled input and the one field it emits. The runner gives every hop
//! the overall call context, a controlled input holding exactly the declared fields, and the
//! accumulated chain (the records of the hops before it), and appends the node's output to
//! the chain. The chain's output field is the refined final output, kept apart from the
//! internal log. Nodes sit behind one trait (`Node`), so exact, stub and neural nodes are
//! interchangeable per role; decision and reasoning nodes are ordinary nodes.
//!
//! Three formation modes behind one interface (`Former`): fixed (a chain from the data
//! file), decision (a decision node picks among the listed chains) and planner (a rule-based
//! composer in this first form). `compare` runs all three for one task.
//!
//! The Core Brain's intake (Phase I1, lane M 2026-11-19): `Routing` (the job map), `IntakeIn` (G2's
//! gate and per-piece evidence), `run_evidence` (hops adding calibrated log-odds, stopping at an
//! evidence bound or the deadline), `duel` and `Labels` (judge duels, both orders, content-free
//! labels held at idle).
//!
//! Later replacement: `CallContext`, `HopIn` and `HopRecord` carry what the in-house message
//! envelope of `nmtype` (step P5, written in parallel) will carry; once it lands, a hop's
//! input and record become envelopes tagged behavioural (the chain log) and sensory (the
//! parts handed up), and `nmtype::Hop` stays the cost trace it is now.
use crate::slot::Slots;
use nmjson::Value;
use nmtype::{Fault, FaultKind, Hop, HopLog, now_ms};
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Component id prefix of this sub-component.
pub const ID: &str = "nmbrain.chain";
/// Task kinds the data file must define (brief of step P7).
pub const REQUIRED: [&str; 4] = ["locate-code", "explain-code", "propose-edit", "verify-answer"];

/// Named faults of the chain loader, validator and runner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChainFault {
    Parse(String),
    NoChains,
    MissingRequired(String),
    DuplicateKind(String),
    EmptyChain(String),
    UnknownRole { chain: String, role: String },
    MissingInput { chain: String, node: usize, field: String },
    DuplicateEmit { chain: String, field: String },
    NoOutput { chain: String, field: String },
    ReasoningIsOutput { chain: String },
    UnknownChain(String),
    MissingNode(String),
    NodeFailed { role: String, message: String },
}
impl ChainFault {
    /// Stable short code of the fault.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::Parse(_) => "chain-parse",
            Self::NoChains => "chain-none",
            Self::MissingRequired(_) => "chain-missing-required",
            Self::DuplicateKind(_) => "chain-duplicate-kind",
            Self::EmptyChain(_) => "chain-empty",
            Self::UnknownRole { .. } => "chain-unknown-role",
            Self::MissingInput { .. } => "chain-missing-input",
            Self::DuplicateEmit { .. } => "chain-duplicate-emit",
            Self::NoOutput { .. } => "chain-no-output",
            Self::ReasoningIsOutput { .. } => "chain-reasoning-is-output",
            Self::UnknownChain(_) => "chain-unknown",
            Self::MissingNode(_) => "chain-missing-node",
            Self::NodeFailed { .. } => "chain-node-failed",
        }
    }
}
impl fmt::Display for ChainFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let detail = match self {
            Self::Parse(m) => m.clone(),
            Self::NoChains => "the file lists no chains".into(),
            Self::MissingRequired(k) => format!("required chain {k} is not defined"),
            Self::DuplicateKind(k) => format!("chain {k} defined twice"),
            Self::EmptyChain(k) => format!("chain {k} has no nodes"),
            Self::UnknownRole { chain, role } => format!("chain {chain}: role {role} is not listed"),
            Self::MissingInput { chain, node, field } => {
                format!("chain {chain}: node {node} takes {field}, which no seed or earlier node provides")
            }
            Self::DuplicateEmit { chain, field } => format!("chain {chain}: field {field} emitted twice"),
            Self::NoOutput { chain, field } => format!("chain {chain}: output {field} is emitted by no node"),
            Self::ReasoningIsOutput { chain } => {
                format!("chain {chain}: a reasoning node emits the final output (it must collaborate, not answer)")
            }
            Self::UnknownChain(k) => format!("no chain {k}"),
            Self::MissingNode(r) => format!("no node implementation for role {r}"),
            Self::NodeFailed { role, message } => format!("node {role} failed: {message}"),
        };
        write!(f, "{}: {detail}", self.code())
    }
}
impl From<ChainFault> for Fault {
    fn from(e: ChainFault) -> Self {
        let kind = match e {
            ChainFault::MissingNode(_) | ChainFault::NodeFailed { .. } => FaultKind::Execution,
            _ => FaultKind::Config,
        };
        Fault::new(ID, kind, e.to_string())
    }
}

/// One node of a chain definition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeDef {
    pub role: String,
    pub takes: Vec<String>,
    pub emits: String,
}
/// One chain: its task kind, a description (what the decision node reads), its output field
/// and its nodes in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainDef {
    pub kind: String,
    pub description: String,
    pub output: String,
    pub nodes: Vec<NodeDef>,
}
impl ChainDef {
    /// The roles of the chain's nodes, in order.
    #[must_use]
    pub fn roles(&self) -> Vec<String> {
        self.nodes.iter().map(|n| n.role.clone()).collect()
    }
}

/// The validated chain definitions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainBook {
    pub seeds: Vec<String>,
    pub roles: Vec<String>,
    pub chains: Vec<ChainDef>,
}
fn strings(v: &Value, key: &str, at: &str) -> Result<Vec<String>, ChainFault> {
    let a = v.get(key).and_then(Value::array).map_err(|e| ChainFault::Parse(format!("{at}.{key}: {e}")))?;
    a.iter().map(|x| x.string().map(str::to_owned).map_err(|e| ChainFault::Parse(format!("{at}.{key}: {e}")))).collect()
}
fn string(v: &Value, key: &str, at: &str) -> Result<String, ChainFault> {
    v.get(key).and_then(Value::string).map(str::to_owned).map_err(|e| ChainFault::Parse(format!("{at}.{key}: {e}")))
}
impl ChainBook {
    /// Parse and validate the chain file's JSON.
    /// # Errors
    /// A named fault for malformed JSON or any rule a chain breaks.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ChainFault> {
        let v = nmjson::parse(bytes).map_err(|e| ChainFault::Parse(e.to_string()))?;
        Self::from_value(&v)
    }
    /// # Errors
    /// As `from_bytes`.
    pub fn from_value(v: &Value) -> Result<Self, ChainFault> {
        let seeds = strings(v, "seeds", "chains.json")?;
        let roles = strings(v, "roles", "chains.json")?;
        let list = v.get("chains").and_then(Value::array).map_err(|e| ChainFault::Parse(format!("chains.json.chains: {e}")))?;
        let mut chains: Vec<ChainDef> = Vec::new();
        for (i, c) in list.iter().enumerate() {
            let at = format!("chains[{i}]");
            let nodes = c.get("nodes").and_then(Value::array).map_err(|e| ChainFault::Parse(format!("{at}.nodes: {e}")))?;
            let nodes = nodes
                .iter()
                .enumerate()
                .map(|(j, n)| {
                    let at = format!("{at}.nodes[{j}]");
                    Ok(NodeDef { role: string(n, "role", &at)?, takes: strings(n, "takes", &at)?, emits: string(n, "emits", &at)? })
                })
                .collect::<Result<Vec<_>, ChainFault>>()?;
            chains.push(ChainDef {
                kind: string(c, "kind", &at)?,
                description: string(c, "description", &at)?,
                output: string(c, "output", &at)?,
                nodes,
            });
        }
        let book = Self { seeds, roles, chains };
        if book.chains.is_empty() {
            return Err(ChainFault::NoChains);
        }
        for (i, c) in book.chains.iter().enumerate() {
            if book.chains.iter().take(i).any(|o| o.kind == c.kind) {
                return Err(ChainFault::DuplicateKind(c.kind.clone()));
            }
            book.validate(c)?;
        }
        for k in REQUIRED {
            if book.get(k).is_none() {
                return Err(ChainFault::MissingRequired(k.into()));
            }
        }
        Ok(book)
    }
    /// # Errors
    /// Unreadable file, or as `from_bytes`.
    pub fn load(path: &Path) -> Result<Self, ChainFault> {
        let b = std::fs::read(path).map_err(|e| ChainFault::Parse(format!("{}: {e}", path.display())))?;
        Self::from_bytes(&b)
    }
    /// `designs/chains.json` of this mini.
    /// # Errors
    /// As `load`.
    pub fn from_mini() -> Result<Self, ChainFault> {
        Self::load(&nmtype::nano_root().join("designs").join("chains.json"))
    }
    #[must_use]
    pub fn get(&self, kind: &str) -> Option<&ChainDef> {
        self.chains.iter().find(|c| c.kind == kind)
    }
    #[must_use]
    pub fn kinds(&self) -> Vec<String> {
        self.chains.iter().map(|c| c.kind.clone()).collect()
    }
    /// Validate one chain against this book's seeds and roles (used for listed and composed
    /// chains alike): known roles, every input provided earlier, one emitter per field, the
    /// output emitted, the output not emitted by a reasoning node.
    /// # Errors
    /// The first broken rule, as a named fault.
    pub fn validate(&self, c: &ChainDef) -> Result<(), ChainFault> {
        if c.nodes.is_empty() {
            return Err(ChainFault::EmptyChain(c.kind.clone()));
        }
        let mut have: Vec<&str> = self.seeds.iter().map(String::as_str).collect();
        for (i, n) in c.nodes.iter().enumerate() {
            if !self.roles.contains(&n.role) {
                return Err(ChainFault::UnknownRole { chain: c.kind.clone(), role: n.role.clone() });
            }
            if let Some(f) = n.takes.iter().find(|f| !have.contains(&f.as_str())) {
                return Err(ChainFault::MissingInput { chain: c.kind.clone(), node: i, field: f.clone() });
            }
            if have.contains(&n.emits.as_str()) {
                return Err(ChainFault::DuplicateEmit { chain: c.kind.clone(), field: n.emits.clone() });
            }
            have.push(&n.emits);
        }
        match c.nodes.iter().find(|n| n.emits == c.output) {
            None => Err(ChainFault::NoOutput { chain: c.kind.clone(), field: c.output.clone() }),
            Some(n) if n.role == "reasoning" => Err(ChainFault::ReasoningIsOutput { chain: c.kind.clone() }),
            Some(_) => Ok(()),
        }
    }
    /// Validate and run one chain over the seeds `task` and `parts`.
    /// # Errors
    /// A validation fault, a missing node or a node failure.
    pub fn run(&self, def: &ChainDef, task: &str, parts: &[String], nodes: &dyn NodeSet, mode: Mode) -> Result<Run, ChainFault> {
        self.validate(def)?;
        run_with(def, task, vec![("task".into(), vec![task.to_owned()]), ("parts".into(), parts.to_vec())], nodes, mode)
    }
}

/// How the chain was formed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Fixed,
    Decided,
    Composed,
}
impl Mode {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Decided => "decision",
            Self::Composed => "planner",
        }
    }
}

/// One field's value: lines of text (envelope payloads later).
pub type Field = Vec<String>;

/// The overall call context every hop receives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallContext {
    pub task: String,
    pub chain: String,
    pub mode: Mode,
    pub hop: usize,
    pub of: usize,
}
/// What one node sees: the call context, its controlled input (exactly the declared fields,
/// in the declared order) and the accumulated chain.
#[derive(Debug)]
pub struct HopIn<'a> {
    pub context: &'a CallContext,
    pub input: &'a [(String, Field)],
    pub chain: &'a [HopRecord],
}
impl HopIn<'_> {
    /// A field of the controlled input.
    #[must_use]
    pub fn field(&self, name: &str) -> Option<&Field> {
        self.input.iter().find(|(n, _)| n == name).map(|(_, f)| f)
    }
    /// The first line of the first input field (the question for most nodes).
    #[must_use]
    pub fn lead(&self) -> &str {
        self.input.first().and_then(|(_, f)| f.first()).map_or(self.context.task.as_str(), String::as_str)
    }
}
/// One MicroModel node. Exact, stub and neural implementations are interchangeable.
pub trait Node: Send + Sync {
    fn role(&self) -> &str;
    /// What implements the node, for the log (exact control, stub, engine label).
    fn label(&self) -> String;
    /// # Errors
    /// The node failed; the runner names it and stops the chain.
    fn run(&self, hop: &HopIn<'_>) -> Result<Field, Fault>;
    /// True when the node orders or selects lines of one of its input fields (embedder,
    /// reranker, extractive writer). The order contract of the runner then applies to its
    /// output: the incoming order is kept unless the node earns the right to change it.
    fn ranks(&self) -> bool {
        false
    }
    /// The node's own confidence, on the scale of its track record, that its proposed order
    /// is better than the incoming one. `None`: the node has no calibrated confidence (every
    /// exact stand-in), so it never reorders what an earlier part ranked.
    fn order_confidence(&self, hop: &HopIn<'_>, proposed: &Field) -> Option<f32> {
        let _ = (hop, proposed);
        None
    }
}
/// The nodes available to the runner, by role.
pub trait NodeSet {
    fn node(&self, role: &str) -> Option<&dyn Node>;
    /// The measured track record of a node implementation (by its label), if any.
    fn track(&self, label: &str) -> Option<&Track> {
        let _ = label;
        None
    }
    /// False only for the first-form behaviour (ranking nodes reorder freely), kept as an
    /// option so the comparison with the order contract stays repeatable.
    fn order_contract(&self) -> bool {
        true
    }
}

/// Measured track record of a ranking node: per past call where it proposed an order other
/// than the incoming one, its confidence then and whether its order turned out better on the
/// checked outcome (right line first). Filled from checked outcomes; empty for exact nodes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Track {
    pub calls: Vec<(f32, bool)>,
}
impl Track {
    pub fn record(&mut self, confidence: f32, improved: bool) {
        self.calls.push((confidence, improved));
    }
    /// The confidence bar derived from the record: the lowest confidence `c` such that, over
    /// the recorded calls at or above `c`, reorders that improved outnumber those that did
    /// not (the point where reordering beats keeping on the record). `None` when no such
    /// point exists, including an empty record: the node then never reorders.
    #[must_use]
    pub fn bar(&self) -> Option<f32> {
        let mut c: Vec<(f32, bool)> = self.calls.clone();
        c.sort_by(|a, b| b.0.total_cmp(&a.0));
        let (mut wins, mut losses, mut best) = (0usize, 0usize, None);
        for (conf, ok) in c {
            if ok {
                wins += 1;
            } else {
                losses += 1;
            }
            if wins > losses {
                best = Some(conf);
            }
        }
        best
    }
}

/// What the order contract did on one hop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderOutcome {
    /// Not a ranking node, or its output is not drawn from one input field (new text).
    Free,
    /// The node's output already kept the incoming order.
    Kept,
    /// The node proposed another order without the evidence to change it; the incoming
    /// order was kept over the lines it selected.
    Held,
    /// The node's confidence cleared the bar of its track record; its order stands.
    Reordered,
}
impl OrderOutcome {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Free => "free",
            Self::Kept => "kept",
            Self::Held => "held",
            Self::Reordered => "reordered",
        }
    }
}

/// The order contract for ranking nodes (2026-11-10, Phase B): `proposed` holds lines drawn
/// from `incoming`; the lines it selected are emitted in the incoming order unless the node's
/// own confidence clears the bar derived from its measured track record. Used by the chain
/// runner and by G2 when it hands parts up.
#[must_use]
pub fn hold_order(incoming: &[String], proposed: &[String], confidence: Option<f32>, track: Option<&Track>) -> (Field, OrderOutcome) {
    let mut kept: Field = Vec::new();
    for l in incoming {
        if proposed.contains(l) && !kept.contains(l) {
            kept.push(l.clone());
        }
    }
    let mut dedup: Field = Vec::new();
    for l in proposed {
        if !dedup.contains(l) {
            dedup.push(l.clone());
        }
    }
    if kept == dedup {
        return (dedup, OrderOutcome::Kept);
    }
    match (confidence, track.and_then(Track::bar)) {
        (Some(c), Some(b)) if c >= b => (dedup, OrderOutcome::Reordered),
        _ => (kept, OrderOutcome::Held),
    }
}
/// The input field a ranking node's output is drawn from: the largest field that holds every
/// output line (first such field on a tie); `None` when the output holds a line of no field.
fn source_field<'a>(input: &'a [(String, Field)], out: &Field) -> Option<&'a Field> {
    let mut best: Option<&Field> = None;
    for (_, f) in input {
        if !f.is_empty() && out.iter().all(|l| f.contains(l)) && best.is_none_or(|b| f.len() > b.len()) {
            best = Some(f);
        }
    }
    best
}
/// One record of the accumulated chain (the internal log).
#[derive(Clone, Debug, PartialEq)]
pub struct HopRecord {
    pub index: usize,
    pub role: String,
    pub label: String,
    pub takes: Vec<String>,
    pub emits: String,
    pub output: Field,
    pub ms: f64,
    pub bytes_in: usize,
    pub bytes_out: usize,
    /// What the order contract did on this hop.
    pub order: OrderOutcome,
}
/// A finished chain run: the refined final output, separate from the internal log.
#[derive(Clone, Debug)]
pub struct Run {
    pub kind: String,
    pub mode: Mode,
    pub final_output: Field,
    pub log: Vec<HopRecord>,
    pub hops: HopLog,
    pub total_ms: f64,
    /// The parts (seed `parts` lines) the final output rests on: its own part lines, or, when
    /// it holds none (a disposition), the nearest part-bearing field upstream of its emitter.
    pub citations: Field,
}
impl Run {
    /// Bytes moved through all hops (inputs, outputs).
    #[must_use]
    pub fn bytes(&self) -> (usize, usize) {
        (self.log.iter().map(|h| h.bytes_in).sum(), self.log.iter().map(|h| h.bytes_out).sum())
    }
}
fn size(f: &Field) -> usize {
    f.iter().map(String::len).sum()
}

/// Run a chain over seed fields (no validation: `ChainBook::run` validates first; internal
/// one-node chains such as the decision step use this directly).
/// # Errors
/// A field the chain needs is missing, a role has no node, or a node failed.
pub fn run_with(def: &ChainDef, task: &str, seeds: Vec<(String, Field)>, nodes: &dyn NodeSet, mode: Mode) -> Result<Run, ChainFault> {
    let start = Instant::now();
    let mut board: BTreeMap<String, Field> = seeds.into_iter().collect();
    let mut log: Vec<HopRecord> = Vec::new();
    let mut hops = HopLog::default();
    for (i, n) in def.nodes.iter().enumerate() {
        let node = nodes.node(&n.role).ok_or_else(|| ChainFault::MissingNode(n.role.clone()))?;
        let input: Vec<(String, Field)> = n
            .takes
            .iter()
            .map(|f| {
                board.get(f).cloned().map(|v| (f.clone(), v)).ok_or_else(|| ChainFault::MissingInput {
                    chain: def.kind.clone(),
                    node: i,
                    field: f.clone(),
                })
            })
            .collect::<Result<_, _>>()?;
        let context = CallContext { task: task.to_owned(), chain: def.kind.clone(), mode, hop: i, of: def.nodes.len() };
        let bytes_in = input.iter().map(|(_, f)| size(f)).sum::<usize>() + task.len();
        let t = Instant::now();
        let hop = HopIn { context: &context, input: &input, chain: &log };
        let proposed = node.run(&hop).map_err(|e| ChainFault::NodeFailed { role: n.role.clone(), message: e.message })?;
        let (out, order) = match (nodes.order_contract() && node.ranks(), source_field(&input, &proposed)) {
            (true, Some(src)) => hold_order(src, &proposed, node.order_confidence(&hop, &proposed), nodes.track(&node.label())),
            _ => (proposed, OrderOutcome::Free),
        };
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        hops.append(Hop::new(&format!("{ID}#{}", n.role), now_ms(), &format!("{}:{i}:{}", def.kind, n.emits), ms.ceil() as u32));
        log.push(HopRecord {
            index: i,
            role: n.role.clone(),
            label: node.label(),
            takes: n.takes.clone(),
            emits: n.emits.clone(),
            bytes_out: size(&out),
            output: out.clone(),
            ms,
            bytes_in,
            order,
        });
        board.insert(n.emits.clone(), out);
    }
    let citations = citations(def, &board);
    let final_output =
        board.remove(&def.output).ok_or_else(|| ChainFault::NoOutput { chain: def.kind.clone(), field: def.output.clone() })?;
    Ok(Run { kind: def.kind.clone(), mode, final_output, log, hops, total_ms: start.elapsed().as_secs_f64() * 1000.0, citations })
}

/// The parts the chain's output rests on (see `Run::citations`): breadth-first from the
/// output field through the fields each emitter took, the first field whose lines are all
/// seed parts (the output's own part lines first). 2026-11-12: when no field is made only of
/// parts, the part lines of the nearest field that holds any; failing that, the first three seed
/// parts, since every chain starts from them: an output is never left without citations.
fn citations(def: &ChainDef, board: &BTreeMap<String, Field>) -> Field {
    let parts = board.get("parts").cloned().unwrap_or_default();
    let own: Field = board.get(&def.output).map(|o| o.iter().filter(|l| parts.contains(l)).cloned().collect()).unwrap_or_default();
    if !own.is_empty() {
        return own;
    }
    let mut queue: std::collections::VecDeque<String> = std::collections::VecDeque::from([def.output.clone()]);
    let mut seen: Vec<String> = Vec::new();
    let mut nearest: Field = Vec::new();
    while let Some(f) = queue.pop_front() {
        if seen.contains(&f) {
            continue;
        }
        seen.push(f.clone());
        if let Some(v) = board.get(&f).filter(|v| !v.is_empty() && v.iter().all(|l| parts.contains(l))) {
            return v.clone();
        }
        if nearest.is_empty() && f != "parts" {
            nearest = board.get(&f).map(|v| v.iter().filter(|l| parts.contains(l)).cloned().collect()).unwrap_or_default();
        }
        if let Some(n) = def.nodes.iter().find(|n| n.emits == f) {
            queue.extend(n.takes.iter().cloned());
        }
    }
    if !nearest.is_empty() {
        return nearest;
    }
    parts.into_iter().take(3).collect()
}

/// Nodes by role; any mix of exact, stub and neural nodes.
#[derive(Default)]
pub struct NodeMap {
    nodes: BTreeMap<String, Box<dyn Node>>,
    tracks: BTreeMap<String, Track>,
    /// First-form behaviour: no order contract (ranking nodes reorder freely).
    pub first_form_order: bool,
}
impl NodeMap {
    /// Add or replace the node for its role.
    pub fn insert(&mut self, n: Box<dyn Node>) {
        self.nodes.insert(n.role().to_owned(), n);
    }
    /// Exact nodes for every role over the Core Brain's slots.
    #[must_use]
    pub fn exact(slots: &Arc<Slots>, roles: &[String]) -> Self {
        let mut m = Self::default();
        for r in roles {
            m.insert(Box::new(ExactNode { role: r.clone(), slots: Arc::clone(slots) }));
        }
        m
    }
}
impl NodeSet for NodeMap {
    fn node(&self, role: &str) -> Option<&dyn Node> {
        self.nodes.get(role).map(|b| b.as_ref())
    }
    fn track(&self, label: &str) -> Option<&Track> {
        self.tracks.get(label)
    }
    fn order_contract(&self) -> bool {
        !self.first_form_order
    }
}

/// The exact node for one role, over the slots' exact controls (law 8: every exact part
/// works with no model loaded). Router: the keyword router. Embedder and reranker: order the
/// parts (they order only, law 6). Writer: extractive. Reasoning: which question terms the
/// evidence covers and which it lacks (a collaborating note, never the answer). Verifier:
/// grounding of a draft in its evidence, or term coverage of the task. Decision: picks among
/// listed options, or turns a verdict into accept or escalate.
pub struct ExactNode {
    pub role: String,
    pub slots: Arc<Slots>,
}
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn order(docs: &[String], scores: &[f32]) -> Field {
    let mut ix: Vec<usize> = (0..docs.len()).collect();
    ix.sort_by(|&a, &b| scores.get(b).unwrap_or(&0.0).total_cmp(scores.get(a).unwrap_or(&0.0)));
    ix.into_iter().filter_map(|i| docs.get(i).cloned()).collect()
}
impl Node for ExactNode {
    fn role(&self) -> &str {
        &self.role
    }
    fn label(&self) -> String {
        format!("exact:{}", self.role)
    }
    fn ranks(&self) -> bool {
        matches!(self.role.as_str(), "embedder" | "reranker" | "writer")
    }
    fn run(&self, hop: &HopIn<'_>) -> Result<Field, Fault> {
        let q = hop.lead().to_owned();
        let last: Field = hop.input.last().map(|(_, f)| f.clone()).unwrap_or_default();
        Ok(match self.role.as_str() {
            "router" => self.slots.route(&q),
            "embedder" => {
                let qv = self.slots.embed(&q);
                let scores: Vec<f32> = last.iter().map(|d| dot(&qv, &self.slots.embed(d))).collect();
                order(&last, &scores)
            }
            "reranker" => order(&last, &self.slots.rerank_all(&q, &last)),
            "writer" => {
                let ev = hop.input.get(1).map(|(_, f)| f.clone()).unwrap_or_default();
                let w = self.slots.by_name.get("writer").map(|s| s.control.write(&q, &ev)).unwrap_or_default();
                let lines: Field = w.lines().map(str::to_owned).collect();
                if lines.is_empty() { ev.into_iter().take(1).collect() } else { lines }
            }
            "reasoning" => {
                let ev = hop.input.get(1).map(|(_, f)| f.join("\n").to_lowercase()).unwrap_or_default();
                let (covered, missing): (Vec<String>, Vec<String>) =
                    nmlex::key_terms(&q).into_iter().partition(|t| ev.contains(t.as_str()));
                let lead = hop.input.get(1).and_then(|(_, f)| f.first()).cloned().unwrap_or_default();
                vec![format!("covered: {}", covered.join(", ")), format!("missing: {}", missing.join(", ")), format!("lead: {lead}")]
            }
            "verifier" => {
                let (target_name, target) = hop.input.first().map(|(n, f)| (n.as_str(), f.clone())).unwrap_or(("task", Vec::new()));
                if target_name == "task" {
                    let ev = last.join("\n").to_lowercase();
                    let terms = nmlex::key_terms(&q);
                    let hit = terms.iter().filter(|t| ev.contains(t.as_str())).count();
                    let share = if terms.is_empty() { 0.0 } else { hit as f32 / terms.len() as f32 };
                    let word = if share >= crate::synthesis::FLOOR { "supported" } else { "unsupported" };
                    vec![format!("{word} {share:.2} of task terms in evidence")]
                } else {
                    let k = target.iter().filter(|l| last.contains(l)).count();
                    let word = if k == target.len() && k > 0 { "supported" } else { "unsupported" };
                    vec![format!("{word} {k}/{} lines grounded", target.len())]
                }
            }
            "decision" => {
                if let Some(opts) = hop.field("options") {
                    let scores: Vec<f32> = opts.iter().map(|o| self.slots.rerank(&q, o)).collect();
                    order(opts, &scores).into_iter().take(1).map(|o| o.split(':').next().unwrap_or("").trim().to_owned()).collect()
                } else if let Some(v) = hop.field("verdict") {
                    vec![if v.first().is_some_and(|l| l.starts_with("supported")) { "accept".into() } else { "escalate".into() }]
                } else {
                    vec!["abstain".into()]
                }
            }
            other => return Err(Fault::new(ID, FaultKind::Config, format!("no exact node for role {other}"))),
        })
    }
}

/// What a stub node saw on one hop (for the hop-contract check).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Seen {
    pub role: String,
    pub task: String,
    pub hop: usize,
    pub input: Vec<String>,
    pub chain_len: usize,
}
/// Stub node: records what it saw and emits one line naming its inputs. As a decision node
/// over options it picks the first option.
pub struct StubNode {
    pub role: String,
    pub seen: Arc<Mutex<Vec<Seen>>>,
}
impl Node for StubNode {
    fn role(&self) -> &str {
        &self.role
    }
    fn label(&self) -> String {
        format!("stub:{}", self.role)
    }
    fn run(&self, hop: &HopIn<'_>) -> Result<Field, Fault> {
        let names: Vec<String> = hop.input.iter().map(|(n, _)| n.clone()).collect();
        if let Ok(mut s) = self.seen.lock() {
            s.push(Seen {
                role: self.role.clone(),
                task: hop.context.task.clone(),
                hop: hop.context.hop,
                input: names.clone(),
                chain_len: hop.chain.len(),
            });
        }
        if let Some(o) = hop.field("options").and_then(|o| o.first()) {
            return Ok(vec![o.split(':').next().unwrap_or("").trim().to_owned()]);
        }
        Ok(vec![format!("{}<-{}", self.role, names.join("+"))])
    }
}

/// A formed chain and how it was formed (the decision step's own hops, if any).
#[derive(Clone, Debug)]
pub struct Formed {
    pub chain: ChainDef,
    pub mode: Mode,
    pub why: String,
    pub log: Vec<HopRecord>,
}
/// One interface for the three formation modes.
pub trait Former {
    fn mode(&self) -> Mode;
    /// # Errors
    /// The mode could not produce a valid chain.
    fn form(&self, task: &str, parts: &[String], book: &ChainBook, nodes: &dyn NodeSet) -> Result<Formed, ChainFault>;
}
/// Fixed: the chain listed for a task kind (G2 or the caller names the kind).
pub struct FixedFormer {
    pub kind: String,
}
impl Former for FixedFormer {
    fn mode(&self) -> Mode {
        Mode::Fixed
    }
    fn form(&self, _: &str, _: &[String], book: &ChainBook, _: &dyn NodeSet) -> Result<Formed, ChainFault> {
        let c = book.get(&self.kind).ok_or_else(|| ChainFault::UnknownChain(self.kind.clone()))?;
        Ok(Formed { chain: c.clone(), mode: Mode::Fixed, why: format!("listed chain {}", self.kind), log: Vec::new() })
    }
}
/// Decision: the decision node picks among the listed chains; it sees the task and one
/// `kind: description` line per chain, nothing else.
pub struct DecisionFormer;
impl Former for DecisionFormer {
    fn mode(&self) -> Mode {
        Mode::Decided
    }
    fn form(&self, task: &str, _: &[String], book: &ChainBook, nodes: &dyn NodeSet) -> Result<Formed, ChainFault> {
        let options: Field = book.chains.iter().map(|c| format!("{}: {}", c.kind, c.description)).collect();
        let select = ChainDef {
            kind: "select-chain".into(),
            description: String::new(),
            output: "choice".into(),
            nodes: vec![NodeDef { role: "decision".into(), takes: vec!["task".into(), "options".into()], emits: "choice".into() }],
        };
        let r = run_with(&select, task, vec![("task".into(), vec![task.to_owned()]), ("options".into(), options)], nodes, Mode::Decided)?;
        let pick = r.final_output.first().cloned().unwrap_or_default();
        let c = book.get(&pick).ok_or(ChainFault::UnknownChain(pick.clone()))?;
        Ok(Formed { chain: c.clone(), mode: Mode::Decided, why: format!("decision node picked {pick}"), log: r.log })
    }
}
/// Planner: composes a chain from the task's wording, rule-based in this first form. A
/// learned planner replaces the word rules with a policy fitted on recorded chain outcomes
/// (which composition answered which kind of task, from G2's labelled cases and memory),
/// choosing nodes by their measured strength.
pub struct PlannerFormer;
impl PlannerFormer {
    fn compose(task: &str) -> ChainDef {
        let l = task.to_lowercase();
        let has = |xs: &[&str]| xs.iter().any(|x| l.contains(x));
        let edit = has(&["change", "fix", "add ", "refactor", "rename", "remove", "replace", "implement", "edit", "patch", "modify"]);
        let check = has(&["verify", "check", "confirm", "is it true", "correct", "supported", "pass"]);
        let why = has(&["why", "how ", "explain", "what does", "purpose", "relate"]);
        let n = |role: &str, takes: &[&str], emits: &str| NodeDef {
            role: role.into(),
            takes: takes.iter().map(|s| (*s).to_owned()).collect(),
            emits: emits.into(),
        };
        let mut nodes = vec![n("router", &["task"], "jobs"), n("reranker", &["task", "parts"], "ranked")];
        let reasoned = why || edit;
        if reasoned {
            nodes.push(n("reasoning", &["task", "ranked", "jobs"], "reasoning"));
        }
        let output = if check && !edit {
            nodes.push(n("verifier", &["task", "ranked"], "verdict"));
            nodes.push(n("decision", &["task", "verdict"], "disposition"));
            "disposition"
        } else {
            let out = if edit { "draft" } else { "answer" };
            nodes.push(n("writer", if reasoned { &["task", "ranked", "reasoning"] } else { &["task", "ranked"] }, out));
            nodes.push(n("verifier", &[out, "ranked"], "verdict"));
            out
        };
        let flags: Vec<&str> = [(edit, "edit"), (check, "check"), (why, "why")].into_iter().filter(|(b, _)| *b).map(|(_, s)| s).collect();
        ChainDef {
            kind: format!("composed:{}", if flags.is_empty() { "plain".into() } else { flags.join("+") }),
            description: "rule-based composition".into(),
            output: output.into(),
            nodes,
        }
    }
}
impl Former for PlannerFormer {
    fn mode(&self) -> Mode {
        Mode::Composed
    }
    fn form(&self, task: &str, _: &[String], book: &ChainBook, _: &dyn NodeSet) -> Result<Formed, ChainFault> {
        let c = Self::compose(task);
        book.validate(&c)?;
        Ok(Formed { why: format!("planner composed {}", c.kind), chain: c, mode: Mode::Composed, log: Vec::new() })
    }
}

/// Form and run one chain by the given mode.
/// # Errors
/// As `Former::form` and `ChainBook::run`.
pub fn form_and_run(
    former: &dyn Former,
    task: &str,
    parts: &[String],
    book: &ChainBook,
    nodes: &dyn NodeSet,
) -> Result<(Formed, Run), ChainFault> {
    let f = former.form(task, parts, book, nodes)?;
    let r = book.run(&f.chain, task, parts, nodes, f.mode)?;
    Ok((f, r))
}

/// One mode's result in `compare`.
pub type ModeResult = (Mode, Result<(Formed, Run), ChainFault>);

/// Run all three formation modes for one task (guide 4.1 point 6: compared per task). The
/// fixed mode uses `fixed_kind` (G2's choice in the served path).
#[must_use]
pub fn compare(task: &str, parts: &[String], fixed_kind: &str, book: &ChainBook, nodes: &dyn NodeSet) -> Vec<ModeResult> {
    let formers: [&dyn Former; 3] = [&FixedFormer { kind: fixed_kind.into() }, &DecisionFormer, &PlannerFormer];
    formers.iter().map(|f| (f.mode(), form_and_run(*f, task, parts, book, nodes))).collect()
}

fn stub_nodes(roles: &[String]) -> (NodeMap, Arc<Mutex<Vec<Seen>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut m = NodeMap::default();
    for r in roles {
        m.insert(Box::new(StubNode { role: r.clone(), seen: Arc::clone(&seen) }));
    }
    (m, seen)
}

/// Self-check of the `chain` sub-component: the data file loads with the four required
/// chains; broken definitions give their named faults; each chain runs end to end on stub
/// nodes with the hop contract held (each node saw exactly its declared fields, the call
/// context and the chain so far; the final output is the output field, apart from the log);
/// each chain runs on exact nodes; all three formation modes give a valid chain.
pub(crate) fn check() -> Result<(), String> {
    let e = |m: String| format!("{ID}: {m}");
    let book = ChainBook::from_mini().map_err(|f| e(f.to_string()))?;
    // Named faults.
    let base = r#"{"seeds":["task","parts"],"roles":["writer","reasoning"],"chains":[CH]}"#;
    let mk = |chains: &str| base.replace("CH", chains);
    let ok4 = REQUIRED
        .iter()
        .map(|k| format!(r#"{{"kind":"{k}","description":"d","output":"a","nodes":[{{"role":"writer","takes":["task"],"emits":"a"}}]}}"#))
        .collect::<Vec<_>>()
        .join(",");
    let cases = [
        (
            mk(&format!(
                r#"{ok4},{{"kind":"x","description":"d","output":"a","nodes":[{{"role":"oracle","takes":["task"],"emits":"a"}}]}}"#
            )),
            "chain-unknown-role",
        ),
        (
            mk(&format!(
                r#"{ok4},{{"kind":"x","description":"d","output":"a","nodes":[{{"role":"writer","takes":["draft"],"emits":"a"}}]}}"#
            )),
            "chain-missing-input",
        ),
        (
            mk(&format!(
                r#"{ok4},{{"kind":"x","description":"d","output":"a","nodes":[{{"role":"reasoning","takes":["task"],"emits":"a"}}]}}"#
            )),
            "chain-reasoning-is-output",
        ),
        (
            mk(&format!(
                r#"{ok4},{{"kind":"x","description":"d","output":"b","nodes":[{{"role":"writer","takes":["task"],"emits":"a"}}]}}"#
            )),
            "chain-no-output",
        ),
        (
            mk(&format!(
                r#"{ok4},{{"kind":"x","description":"d","output":"a","nodes":[{{"role":"writer","takes":["task"],"emits":"task"}}]}}"#
            )),
            "chain-duplicate-emit",
        ),
        (
            mk(&format!(
                r#"{ok4},{{"kind":"locate-code","description":"d","output":"a","nodes":[{{"role":"writer","takes":["task"],"emits":"a"}}]}}"#
            )),
            "chain-duplicate-kind",
        ),
        (
            mk(r#"{"kind":"locate-code","description":"d","output":"a","nodes":[{"role":"writer","takes":["task"],"emits":"a"}]}"#),
            "chain-missing-required",
        ),
        (mk(""), "chain-none"),
        ("{".to_owned(), "chain-parse"),
    ];
    for (text, want) in cases {
        match ChainBook::from_bytes(text.as_bytes()) {
            Err(f) if f.code() == want => {}
            other => return Err(e(format!("expected {want}, got {other:?}"))),
        }
    }
    // Hop contract on stub nodes.
    let parts: Field =
        vec!["src/a.rs:3 pub struct ParMap {".into(), "src/b.rs:9 fn observe(edit: &Edit) {".into(), "README.md:1 notes".into()];
    let task = "Where is ParMap defined and who calls observe?";
    for c in &book.chains {
        let (nodes, seen) = stub_nodes(&book.roles);
        let r = book.run(c, task, &parts, &nodes, Mode::Fixed).map_err(|f| e(f.to_string()))?;
        let seen = seen.lock().map(|s| s.clone()).map_err(|_| e("stub log poisoned".into()))?;
        if seen.len() != c.nodes.len() || r.log.len() != c.nodes.len() || r.hops.hops().len() != c.nodes.len() {
            return Err(e(format!("{}: {} hops seen for {} nodes", c.kind, seen.len(), c.nodes.len())));
        }
        for (i, (s, n)) in seen.iter().zip(&c.nodes).enumerate() {
            if s.input != n.takes || s.chain_len != i || s.hop != i || s.task != task || s.role != n.role {
                return Err(e(format!("{}: hop {i} saw {s:?}, declared {n:?}", c.kind)));
            }
        }
        let emitter = r.log.iter().find(|h| h.emits == c.output).map(|h| h.output.clone());
        if emitter.as_ref() != Some(&r.final_output) || r.final_output.is_empty() {
            return Err(e(format!("{}: final output is not the output field", c.kind)));
        }
        if r.log.iter().any(|h| h.bytes_in == 0 || h.ms < 0.0) {
            return Err(e(format!("{}: hop cost not recorded", c.kind)));
        }
    }
    // Exact nodes end to end.
    let slots = Arc::new(Slots::default());
    let exact = NodeMap::exact(&slots, &book.roles);
    for c in &book.chains {
        let r = book.run(c, task, &parts, &exact, Mode::Fixed).map_err(|f| e(f.to_string()))?;
        if c.kind == "locate-code" && (r.final_output.is_empty() || !r.final_output.iter().all(|l| parts.contains(l))) {
            return Err(e(format!("locate-code on exact nodes is not extractive: {:?}", r.final_output)));
        }
        // Order contract: exact ranking nodes have no track record, so every ranking hop
        // keeps the incoming order over the lines it selected.
        for h in r.log.iter().filter(|h| matches!(h.role.as_str(), "embedder" | "reranker" | "writer")) {
            let incoming: Vec<&String> = parts.iter().filter(|p| h.output.contains(p)).collect();
            if h.order == OrderOutcome::Reordered || h.output.iter().collect::<Vec<_>>() != incoming {
                return Err(e(format!("{}: exact {} changed the incoming order ({})", c.kind, h.role, h.order.as_str())));
            }
        }
        if r.citations.is_empty() || !r.citations.iter().all(|l| parts.contains(l)) {
            return Err(e(format!("{}: final output carries no citations of the parts", c.kind)));
        }
    }
    // The bar comes from the track record: none without a record or with more losses than
    // wins; a confident node with a winning record may reorder, an unconfident one may not.
    let (inc, prop) = (vec!["a".to_owned(), "b".to_owned(), "c".to_owned()], vec!["c".to_owned(), "a".to_owned()]);
    let mut losing = Track::default();
    losing.record(0.9, false);
    losing.record(0.8, true);
    let mut winning = Track::default();
    for (c, ok) in [(0.9, true), (0.7, true), (0.6, false), (0.4, false), (0.3, false)] {
        winning.record(c, ok);
    }
    let cases = [
        (None, None, OrderOutcome::Held),
        (Some(0.99), Some(&losing), OrderOutcome::Held),
        (Some(0.8), Some(&winning), OrderOutcome::Reordered),
        (Some(0.5), Some(&winning), OrderOutcome::Held),
    ];
    for (conf, track, want) in cases {
        let (out, got) = hold_order(&inc, &prop, conf, track);
        let expect: Vec<String> = if want == OrderOutcome::Reordered { prop.clone() } else { vec!["a".into(), "c".into()] };
        if got != want || out != expect {
            return Err(e(format!("hold_order with confidence {conf:?}: {} {out:?}, expected {} {expect:?}", got.as_str(), want.as_str())));
        }
    }
    if Track::default().bar().is_some() || winning.bar() != Some(0.6) {
        return Err(e(format!("track bar: empty {:?}, winning {:?}", Track::default().bar(), winning.bar())));
    }
    // Three formation modes, on stub and exact nodes.
    for t in [
        task,
        "Explain how the river mailbox relates to synthesis",
        "Fix the reranker so it skips empty lines",
        "Verify that the writer is extractive",
    ] {
        for nodes in [&exact as &dyn NodeSet, &stub_nodes(&book.roles).0] {
            for (mode, out) in compare(t, &parts, "locate-code", &book, nodes) {
                let (f, r) = out.map_err(|f| e(format!("{} on {t}: {f}", mode.as_str())))?;
                book.validate(&f.chain).map_err(|f| e(f.to_string()))?;
                if r.mode != mode || r.log.len() != f.chain.nodes.len() {
                    return Err(e(format!("{} on {t}: run does not match its chain", mode.as_str())));
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Phase I: the Core Brain's intake as an evidence chain (tentacle master plan 3.9 and Phase I1;
// master design 3.7; lane M, 2026-11-19). The chain reads G2's gate and per-piece evidence; G2
// chooses the chain from the routing table by the job map (embedder for code close calls, judge
// for prose close calls, reader for joint reads and one-word judgements, writer at idle), with a
// prior from past configurations or muscle memory deciding only past the rule's margin; each hop
// adds calibrated evidence on the log-odds scale; the chain stops at an evidence bound or the
// deadline. Judge duels read both sides (both orders), each order less its content-free label,
// labels held at idle so a duel costs two reads (C77, C79). Readers sit behind `HopReader` and
// `DuelReader`, so the card's MicroModels plug in from `nmapi` without this crate loading any.
// ---------------------------------------------------------------------------------------------

/// A job of the Core Brain's intake (the job map).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Job {
    /// A close call between code pieces.
    CodeClose,
    /// A close call involving prose.
    ProseClose,
    /// A joint read of tied pieces.
    Joint,
    /// A one-word judgement (sufficiency: does the set answer the request).
    OneWord,
    /// Work at idle (labels, the writer's background work).
    Idle,
}
impl Job {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::CodeClose => "code-close",
            Self::ProseClose => "prose-close",
            Self::Joint => "joint",
            Self::OneWord => "one-word",
            Self::Idle => "idle",
        }
    }
}

/// The MicroModels a chain's hop can be.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MicroModel {
    Embedder,
    Judge,
    Reader,
    Writer,
}
impl MicroModel {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Embedder => "embedder",
            Self::Judge => "judge",
            Self::Reader => "reader",
            Self::Writer => "writer",
        }
    }
}

/// The routing table: per job, the MicroModels that can do it with their measured strength
/// (higher first wins among the available).
#[derive(Clone, Debug, PartialEq)]
pub struct Routing {
    pub rows: Vec<(Job, MicroModel, f32)>,
}
impl Routing {
    /// The job map approved 2026-11-18 08:38 with the measured strengths: on code the embedder
    /// (75.41) above the judge (73.42); on prose the judge well above (C102), the embedder the
    /// fallback; the reader for joint reads and one-word judgements; the writer at idle.
    #[must_use]
    pub fn job_map() -> Self {
        Self {
            rows: vec![
                (Job::CodeClose, MicroModel::Embedder, 75.41),
                (Job::CodeClose, MicroModel::Judge, 73.42),
                (Job::ProseClose, MicroModel::Judge, 1.0),
                (Job::ProseClose, MicroModel::Embedder, 0.5),
                (Job::Joint, MicroModel::Reader, 1.0),
                (Job::OneWord, MicroModel::Reader, 1.0),
                (Job::Idle, MicroModel::Writer, 1.0),
            ],
        }
    }
    /// The MicroModel for `job`: the strongest available; a prior (past configurations or muscle
    /// memory: a model and its evidence in standard errors) decides instead only when its model can
    /// do the job, is available, and its evidence exceeds `margin` (priors never a hard override).
    #[must_use]
    pub fn choose(&self, job: Job, available: &dyn Fn(MicroModel) -> bool, prior: Option<(MicroModel, f64)>, margin: f64) -> Option<MicroModel> {
        let can = |m: MicroModel| self.rows.iter().any(|r| r.0 == job && r.1 == m) && available(m);
        if let Some((m, ev)) = prior
            && can(m)
            && ev > margin
        {
            return Some(m);
        }
        self.rows.iter().filter(|r| r.0 == job && available(r.1)).max_by(|a, b| a.2.total_cmp(&b.2)).map(|r| r.1)
    }
    /// The job map with the measured routing record laid over it (Stage 5 WP-C2): each measured
    /// row's strength replaces the job map's for its job and MicroModel; rows the record does not
    /// measure keep the job map's value.
    #[must_use]
    pub fn from_record(record: &RoutingRecord) -> Self {
        let mut r = Self::job_map();
        for m in &record.rows {
            if let Some(row) = r.rows.iter_mut().find(|x| x.0 == m.job && x.1 == m.model) {
                row.2 = m.strength;
            }
        }
        r
    }
}

/// One measured row of the routing record: a MicroModel's accuracy on one job (Stage 5 WP-C2).
#[derive(Clone, Debug, PartialEq)]
pub struct RecordRow {
    pub job: Job,
    pub model: MicroModel,
    /// Accuracy on the fit folds in percent.
    pub strength: f32,
    /// Fit duels and the logistic slope turning the MicroModel's difference into log-odds.
    pub n: u64,
    pub slope: f32,
}

/// The measured routing record (`designs/routing-record.json`, written by
/// `nano-lab stage-bench routing-record`): measured rows, the jobs whose two strong MicroModels
/// combine their evidence in log-odds, and the serving base cartridges it was measured on.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct RoutingRecord {
    pub rows: Vec<RecordRow>,
    pub combine: Vec<Job>,
    /// Per combined job, the |embedder calibrated log-odds| from which the judge's read is skipped
    /// (derived on the fit folds by the record's builder; absent: the serving default).
    pub decisive: Vec<(Job, f32)>,
    /// (MicroModel family, sha256) of the base cartridges the record was measured on.
    pub cartridges: Vec<(String, String)>,
    /// The project lenses the duels were read with (`id@version` or `none`, comma separated; a
    /// record without the field was measured with none).
    pub lens: String,
}

impl RoutingRecord {
    /// Read a routing record. `None` when the file is missing or malformed (the caller falls back).
    #[must_use]
    pub fn load(path: &Path) -> Option<Self> {
        let v = nmjson::parse(&std::fs::read(path).ok()?).ok()?;
        let job_of = |s: &str| match s {
            "code-close" => Some(Job::CodeClose),
            "prose-close" => Some(Job::ProseClose),
            "joint" => Some(Job::Joint),
            "one-word" => Some(Job::OneWord),
            "idle" => Some(Job::Idle),
            _ => None,
        };
        let model_of = |s: &str| match s {
            "embedder" => Some(MicroModel::Embedder),
            "judge" => Some(MicroModel::Judge),
            "reader" => Some(MicroModel::Reader),
            "writer" => Some(MicroModel::Writer),
            _ => None,
        };
        let num = |x: &nmjson::Value, k: &str| match x.get(k).ok() {
            Some(nmjson::Value::Number(n)) => n.parse::<f64>().ok(),
            _ => None,
        };
        let mut rec = Self::default();
        for r in v.get("rows").ok()?.array().ok()? {
            let Some(job) = r.get("job").ok().and_then(|x| x.string().ok()).and_then(job_of) else { continue };
            if let Ok(nmjson::Value::Bool(c)) = r.get("combine") {
                if *c {
                    rec.combine.push(job);
                }
                if let Some(d) = num(r, "decisive") {
                    #[expect(clippy::cast_possible_truncation, reason = "a log-odds threshold")]
                    rec.decisive.push((job, d as f32));
                }
                continue;
            }
            let Some(model) = r.get("model").ok().and_then(|x| x.string().ok()).and_then(model_of) else { continue };
            let n = r.get("n").ok().and_then(|x| x.integer().ok()).unwrap_or(0);
            let Some(strength) = num(r, "strength") else { continue };
            #[expect(clippy::cast_possible_truncation, reason = "a percentage and a slope")]
            rec.rows.push(RecordRow { job, model, strength: strength as f32, n, slope: num(r, "slope").unwrap_or(0.0) as f32 });
        }
        if let Ok(c) = v.get("base_cartridges").and_then(nmjson::Value::object) {
            rec.cartridges = c.iter().filter_map(|(k, x)| x.string().ok().map(|s| (k.clone(), s.to_owned()))).collect();
        }
        rec.lens = v.get("lens").ok().and_then(|x| x.string().ok()).unwrap_or("none").to_owned();
        Some(rec)
    }
    /// Whether the record was measured on these serving cartridges (family, sha256): every family
    /// it names must serve the same file.
    #[must_use]
    pub fn valid_for(&self, serving: &[(String, String)]) -> bool {
        !self.cartridges.is_empty() && self.cartridges.iter().all(|(k, sha)| serving.iter().any(|(sk, ss)| sk == k && ss == sha && !sha.is_empty()))
    }
    /// The measured row for a job and MicroModel.
    #[must_use]
    pub fn row(&self, job: Job, model: MicroModel) -> Option<&RecordRow> {
        self.rows.iter().find(|r| r.job == job && r.model == model)
    }
}

/// G2's gate as the chain reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateStatus {
    Settled,
    Low,
    Withheld,
    Tie,
}
impl GateStatus {
    /// From G2's gate line (its first word: `settled`, `low`, `withheld`, `tie`; anything else,
    /// `absent` included, is low: an aspect absent from G2's set calls for reads, not for silence;
    /// T-083).
    #[must_use]
    pub fn of(gate: &str) -> Self {
        match gate.split([' ', ':', ',']).next().unwrap_or("") {
            "settled" => Self::Settled,
            "withheld" => Self::Withheld,
            "tie" => Self::Tie,
            _ => Self::Low,
        }
    }
}

/// One piece G2 hands up, with its evidence on the log-odds scale.
#[derive(Clone, Debug, PartialEq)]
pub struct PieceEvidence {
    /// The piece's content key (a reference; the text travels apart, sensory).
    pub key: String,
    pub prose: bool,
    pub logodds: f32,
}

/// What the chain starts from: G2's gate and its pieces in G2's order.
#[derive(Clone, Debug, PartialEq)]
pub struct IntakeIn {
    pub gate: GateStatus,
    pub pieces: Vec<PieceEvidence>,
}

/// One hop's reader: calibrated log-odds per piece named (content-free label already subtracted).
pub trait HopReader {
    fn model(&self) -> MicroModel;
    /// # Errors
    /// The reader's own failure (the hop is skipped and logged).
    fn read(&mut self, job: Job, pieces: &[&PieceEvidence]) -> Result<Vec<f32>, String>;
}

/// A judge's duel reads: the posed read (A against B, log-odds that A answers better) and the
/// content-free read of the same order (the label).
pub trait DuelReader {
    /// # Errors
    /// The reader's failure.
    fn posed(&mut self, a: &str, b: &str) -> Result<f32, String>;
    /// # Errors
    /// The reader's failure.
    fn content_free(&mut self, a: &str, b: &str) -> Result<f32, String>;
    /// Reads one content-free label costs (its mean over the content-free forms; `nmapi`'s
    /// `neural::CONTENT_FREE` has three, so an unheld duel costs eight reads, C79).
    fn forms(&self) -> usize {
        3
    }
}

/// Content-free labels per ordered pair, held at idle (RAM, for the process).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Labels {
    pub held: BTreeMap<(String, String), f32>,
}
impl Labels {
    /// At idle: read and hold the labels of both orders of each pair not held yet; labels read.
    pub fn prepare_idle(&mut self, reader: &mut dyn DuelReader, pairs: &[(String, String)]) -> usize {
        let mut n = 0;
        for (a, b) in pairs {
            for (x, y) in [(a, b), (b, a)] {
                let k = (x.clone(), y.clone());
                if !self.held.contains_key(&k)
                    && let Ok(v) = reader.content_free(x, y)
                {
                    self.held.insert(k, v);
                    n += 1;
                }
            }
        }
        n
    }
    fn get_or_read(&mut self, reader: &mut dyn DuelReader, a: &str, b: &str, reads: &mut usize) -> Result<f32, String> {
        let k = (a.to_owned(), b.to_owned());
        if let Some(v) = self.held.get(&k) {
            return Ok(*v);
        }
        let v = reader.content_free(a, b)?;
        *reads += reader.forms();
        self.held.insert(k, v);
        Ok(v)
    }
}

/// A duel's verdict: A's evidence over B (log-odds, mean of the two orders), whether the two
/// orders agree beyond the band, and the reads it cost.
#[derive(Clone, Debug, PartialEq)]
pub struct Duel {
    pub a_over_b: f32,
    pub tie: bool,
    pub reads: usize,
}

/// A judge duel: both orders read, each less its content-free label; orders that disagree, or
/// agree within `band`, are a tie.
/// # Errors
/// The reader's failure.
pub fn duel(reader: &mut dyn DuelReader, labels: &mut Labels, a: &str, b: &str, band: f32) -> Result<Duel, String> {
    let mut reads = 2;
    let ab = reader.posed(a, b)? - labels.get_or_read(reader, a, b, &mut reads)?;
    let ba = reader.posed(b, a)? - labels.get_or_read(reader, b, a, &mut reads)?;
    let e = (ab - ba) / 2.0;
    let agree = (ab > band && ba < -band) || (ab < -band && ba > band);
    Ok(Duel { a_over_b: e, tie: !agree, reads })
}

/// A judge as a hop: duels the first two pieces named; the verdict's evidence goes half to each
/// side (plus to the winner, minus to the loser); a tie moves nothing.
pub struct DuelHop<'a> {
    pub reader: &'a mut dyn DuelReader,
    pub labels: &'a mut Labels,
    pub band: f32,
    /// Reads spent so far.
    pub reads: usize,
}
impl HopReader for DuelHop<'_> {
    fn model(&self) -> MicroModel {
        MicroModel::Judge
    }
    fn read(&mut self, _job: Job, pieces: &[&PieceEvidence]) -> Result<Vec<f32>, String> {
        let mut out = vec![0.0; pieces.len()];
        if let (Some(a), Some(b)) = (pieces.first(), pieces.get(1)) {
            let d = duel(self.reader, self.labels, &a.key, &b.key, self.band)?;
            self.reads += d.reads;
            if !d.tie {
                out[0] = d.a_over_b / 2.0;
                out[1] = -d.a_over_b / 2.0;
            }
        }
        Ok(out)
    }
}

/// Why the chain stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// The leader's evidence over the runner-up reached the bound.
    Bound,
    /// The deadline came.
    Deadline,
    /// Every planned hop ran.
    Planned,
    /// G2 withheld: nothing goes up to be read.
    Withheld,
}

/// One hop of the evidence chain, for the trace and the decision-site record.
#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceHop {
    pub job: Job,
    pub model: Option<MicroModel>,
    /// Pieces read, the leader's margin after the hop, and a note (skips and failures).
    pub read: usize,
    pub margin: f32,
    pub note: String,
}

/// The chain's result: the pieces' order by accumulated evidence, the totals in G2's order, the
/// hops and why it stopped.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainOut {
    pub order: Vec<usize>,
    pub totals: Vec<f32>,
    pub hops: Vec<EvidenceHop>,
    pub stop: Stop,
}

/// The jobs a gate calls for: settled, none beyond the reads already made; tie and low, the close
/// call of the leading pieces' kind, then a joint read; low also a one-word sufficiency
/// judgement; withheld, none.
#[must_use]
pub fn plan_of(i: &IntakeIn) -> Vec<Job> {
    let close = if i.pieces.iter().take(2).any(|p| p.prose) { Job::ProseClose } else { Job::CodeClose };
    match i.gate {
        GateStatus::Settled | GateStatus::Withheld => Vec::new(),
        GateStatus::Tie => vec![close, Job::Joint],
        GateStatus::Low => vec![close, Job::Joint, Job::OneWord],
    }
}

/// A hop's evidence is clipped to this many log-odds units (one hop never decides alone past it).
pub const HOP_CLIP: f32 = 8.0;

fn ordered(totals: &[f32]) -> Vec<usize> {
    let mut o: Vec<usize> = (0..totals.len()).collect();
    o.sort_by(|a, b| totals[*b].total_cmp(&totals[*a]).then(a.cmp(b)));
    o
}
fn margin_of(totals: &[f32], order: &[usize]) -> f32 {
    match (order.first(), order.get(1)) {
        (Some(a), Some(b)) => totals[*a] - totals[*b],
        (Some(_), None) => f32::INFINITY,
        _ => 0.0,
    }
}

/// Run the evidence chain: start from G2's per-piece evidence; for each job of the plan, the
/// routing table names the MicroModel (prior per job as `priors` gives it); the hop reads the
/// pieces within `bound` of the leader (at least the top two) and adds its calibrated evidence;
/// stop when the leader's margin reaches `bound`, at `deadline`, or when the plan is done. A job
/// with no reader available or a reader that fails is logged and skipped.
#[expect(clippy::too_many_arguments, reason = "intake, plan, routing, readers, priors, bound and deadline are independent inputs")]
pub fn run_evidence(
    i: &IntakeIn,
    plan: &[Job],
    routing: &Routing,
    readers: &mut [&mut dyn HopReader],
    priors: &dyn Fn(Job) -> Option<(MicroModel, f64)>,
    bound: f32,
    deadline: Instant,
) -> ChainOut {
    let mut totals: Vec<f32> = i.pieces.iter().map(|p| if p.logodds.is_finite() { p.logodds } else { 0.0 }).collect();
    let mut hops = Vec::new();
    if i.gate == GateStatus::Withheld {
        return ChainOut { order: ordered(&totals), totals, hops, stop: Stop::Withheld };
    }
    let mut stop = Stop::Planned;
    for &job in plan {
        let order = ordered(&totals);
        if margin_of(&totals, &order) >= bound {
            stop = Stop::Bound;
            break;
        }
        if Instant::now() >= deadline {
            stop = Stop::Deadline;
            break;
        }
        let have: Vec<MicroModel> = readers.iter().map(|r| r.model()).collect();
        let Some(m) = routing.choose(job, &|m| have.contains(&m), priors(job), 0.0) else {
            hops.push(EvidenceHop { job, model: None, read: 0, margin: margin_of(&totals, &order), note: "no reader available".into() });
            continue;
        };
        let lead = order.first().map_or(0.0, |a| totals[*a]);
        let named: Vec<usize> = order.iter().enumerate().filter(|(k, c)| *k < 2 || lead - totals[**c] < bound).map(|(_, c)| *c).collect();
        let refs: Vec<&PieceEvidence> = named.iter().filter_map(|c| i.pieces.get(*c)).collect();
        let Some(r) = readers.iter_mut().find(|r| r.model() == m) else { continue };
        let note = match r.read(job, &refs) {
            Ok(ev) => {
                for (c, e) in named.iter().zip(ev) {
                    if e.is_finite()
                        && let Some(t) = totals.get_mut(*c)
                    {
                        *t += e.clamp(-HOP_CLIP, HOP_CLIP);
                    }
                }
                String::new()
            }
            Err(e) => format!("failed: {e}"),
        };
        let order = ordered(&totals);
        hops.push(EvidenceHop { job, model: Some(m), read: refs.len(), margin: margin_of(&totals, &order), note });
    }
    if stop == Stop::Planned && margin_of(&totals, &ordered(&totals)) >= bound {
        stop = Stop::Bound;
    }
    ChainOut { order: ordered(&totals), totals, hops, stop }
}

/// Section check of the intake: the job map routes code close calls to the embedder, prose to the
/// judge, joint and one-word to the reader, idle to the writer; a prior decides only past the
/// margin; a duel reads both orders less their labels, held labels make it two reads, disagreeing
/// orders tie; the chain adds hop evidence, stops at the bound, at the deadline, and withholds.
/// # Errors
/// One fault line.
pub fn check_intake() -> Result<(), String> {
    let r = Routing::job_map();
    let all = |_m: MicroModel| true;
    let no_judge = |m: MicroModel| m != MicroModel::Judge;
    let want = [
        (Job::CodeClose, Some(MicroModel::Embedder)),
        (Job::ProseClose, Some(MicroModel::Judge)),
        (Job::Joint, Some(MicroModel::Reader)),
        (Job::OneWord, Some(MicroModel::Reader)),
        (Job::Idle, Some(MicroModel::Writer)),
    ];
    for (j, m) in want {
        if r.choose(j, &all, None, 0.0) != m {
            return Err(format!("nmbrain.chain.intake: {} routed to {:?}", j.name(), r.choose(j, &all, None, 0.0)));
        }
    }
    if r.choose(Job::ProseClose, &no_judge, None, 0.0) != Some(MicroModel::Embedder)
        || r.choose(Job::CodeClose, &all, Some((MicroModel::Judge, 1.0)), 2.0) != Some(MicroModel::Embedder)
        || r.choose(Job::CodeClose, &all, Some((MicroModel::Judge, 3.0)), 2.0) != Some(MicroModel::Judge)
        || r.choose(Job::CodeClose, &all, Some((MicroModel::Writer, 9.0)), 0.0) != Some(MicroModel::Embedder)
    {
        return Err("nmbrain.chain.intake: fallback or prior arbitration".into());
    }
    // A judge that favours whatever comes first by 1.0 (position bias) and A over B by 2.0.
    struct J(usize);
    impl DuelReader for J {
        fn posed(&mut self, a: &str, b: &str) -> Result<f32, String> {
            self.0 += 1;
            Ok(1.0 + if a == "A" && b == "B" { 2.0 } else if a == "B" && b == "A" { -2.0 } else { 0.0 })
        }
        fn content_free(&mut self, _a: &str, _b: &str) -> Result<f32, String> {
            self.0 += 1;
            Ok(1.0)
        }
    }
    let mut j = J(0);
    let mut labels = Labels::default();
    if labels.prepare_idle(&mut j, &[("A".into(), "B".into())]) != 2 {
        return Err("nmbrain.chain.intake: labels at idle".into());
    }
    let d = duel(&mut j, &mut labels, "A", "B", 0.5)?;
    if d.tie || (d.a_over_b - 2.0).abs() > 1e-6 || d.reads != 2 || j.0 != 4 {
        return Err(format!("nmbrain.chain.intake: duel {d:?} after {} reads", j.0));
    }
    let d = duel(&mut j, &mut labels, "C", "D", 0.5)?;
    if !d.tie || d.reads != 8 {
        return Err(format!("nmbrain.chain.intake: unheld or position-only duel {d:?}"));
    }
    struct E(MicroModel, f32);
    impl HopReader for E {
        fn model(&self) -> MicroModel {
            self.0
        }
        fn read(&mut self, _job: Job, pieces: &[&PieceEvidence]) -> Result<Vec<f32>, String> {
            Ok(pieces.iter().map(|p| if p.key == "right" { self.1 } else { 0.0 }).collect())
        }
    }
    let piece = |k: &str, l: f32| PieceEvidence { key: k.into(), prose: false, logodds: l };
    let tie = IntakeIn { gate: GateStatus::Tie, pieces: vec![piece("wrong", 1.0), piece("right", 0.9), piece("far", -5.0)] };
    let far = Instant::now() + std::time::Duration::from_secs(60);
    let none = |_j: Job| None;
    let mut emb = E(MicroModel::Embedder, 1.5);
    let mut rd = E(MicroModel::Reader, 1.5);
    let out = run_evidence(&tie, &plan_of(&tie), &r, &mut [&mut emb, &mut rd], &none, 1.0, far);
    if out.order.first() != Some(&1) || out.stop != Stop::Bound || out.hops.len() != 1 || out.hops[0].model != Some(MicroModel::Embedder) || out.hops[0].read != 2 {
        return Err(format!("nmbrain.chain.intake: tie chain {out:?}"));
    }
    let mut weak = E(MicroModel::Embedder, 0.2);
    let out = run_evidence(&tie, &plan_of(&tie), &r, &mut [&mut weak], &none, 1.0, far);
    if out.stop != Stop::Planned || out.hops.len() != 2 || out.hops[1].model.is_some() {
        return Err(format!("nmbrain.chain.intake: missing reader {out:?}"));
    }
    let out = run_evidence(&tie, &plan_of(&tie), &r, &mut [&mut emb], &none, 1.0, Instant::now());
    if out.stop != Stop::Deadline || !out.hops.is_empty() || out.order.first() != Some(&0) {
        return Err(format!("nmbrain.chain.intake: deadline {out:?}"));
    }
    let held = IntakeIn { gate: GateStatus::of("withheld: no aspect covered"), pieces: tie.pieces.clone() };
    let out = run_evidence(&held, &plan_of(&held), &r, &mut [&mut emb], &none, 1.0, far);
    if out.stop != Stop::Withheld || GateStatus::of("settled") != GateStatus::Settled || !plan_of(&IntakeIn { gate: GateStatus::Settled, pieces: Vec::new() }).is_empty() {
        return Err("nmbrain.chain.intake: withheld or settled".into());
    }
    let prose = IntakeIn { gate: GateStatus::Low, pieces: vec![PieceEvidence { prose: true, ..piece("wrong", 1.0) }, piece("right", 0.9)] };
    let mut j = J(0);
    let mut labels = Labels::default();
    let mut judge = DuelHop { reader: &mut j, labels: &mut labels, band: 0.5, reads: 0 };
    let out = run_evidence(&prose, &plan_of(&prose), &r, &mut [&mut judge], &none, 100.0, far);
    if plan_of(&prose) != vec![Job::ProseClose, Job::Joint, Job::OneWord] || out.hops.first().and_then(|h| h.model) != Some(MicroModel::Judge) {
        return Err(format!("nmbrain.chain.intake: prose low {out:?}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ChainBook, NodeMap, compare};
    use crate::slot::Slots;
    use std::sync::Arc;
    #[test]
    fn intake() {
        assert_eq!(super::check_intake(), Ok(()));
    }
    /// Prints the three formation modes per task on exact nodes
    /// (`cargo test --release -p nmbrain chain_modes -- --nocapture`).
    #[test]
    fn chain_modes() {
        let book = ChainBook::from_mini().expect("chains.json");
        let exact = NodeMap::exact(&Arc::new(Slots::default()), &book.roles);
        let parts: Vec<String> =
            vec!["src/a.rs:3 pub struct ParMap {".into(), "src/b.rs:9 fn observe(edit: &Edit) {".into(), "README.md:1 notes".into()];
        for (t, fixed) in [
            ("Where is ParMap defined?", "locate-code"),
            ("Explain how observe uses ParMap", "explain-code"),
            ("Fix observe so it skips empty edits", "propose-edit"),
            ("Verify that ParMap is a struct", "verify-answer"),
        ] {
            for (mode, r) in compare(t, &parts, fixed, &book, &exact) {
                let (f, r) = r.expect("mode runs");
                let (bi, bo) = r.bytes();
                println!(
                    "{t} | {} | {} | hops {} | {:.3} ms | bytes in {bi} out {bo} | final {:?}",
                    mode.as_str(),
                    f.chain.kind,
                    r.log.len(),
                    r.total_ms,
                    r.final_output
                );
            }
        }
    }
}
