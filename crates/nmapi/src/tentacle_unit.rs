//! The one tentacle system, first form (2026-11-10, step P4; specification
//! `docs/2026-11-10-neumor-guide-how-it-works.md` sections 2.1 to 2.4 and 3, and its dated
//! addition of 2026-11-10 08:00 UTC; architecture by Luis Jake Gabriel III). This adapter joins an
//! `nmpod` tentacle unit to `crate::tentacle_core`, so `nmpod` itself does not depend on the
//! engine. Behind `--one-tentacle` (off by default; with it off the territory graph and the side
//! pass run exactly as before) each of the graph's tentacles is one autonomous processing pathway,
//! running in parallel with the others, and owns:
//! - its window: laid out for the question over the whole code base by a lens (a temporary
//!   specialisation), cut into chunk spans, no chunk in two windows; laid out with no weights
//!   (token counts from the base tokenizer file alone);
//! - its drone tip: every reader behind the one `nmpod::Reader` interface: the attention-head
//!   drone readers (`DroneReader`) when a model is loaded, and the exact readers as further readers
//!   (`ExactReach` says where they read);
//! - its core: the neural tentacle core (`tentacle_core::TentacleCore`, through its public
//!   interface) when `--tentacle-core` is on, else the stated exact fallback core (`nmpod::core`),
//!   which also runs when the neural core faults;
//! - its `g0_tx`: the core's state as a `nmbrain::g0::TentacleSide`, reporting only to G1.
//!
//! Parameters (addition of 2026-11-10 08:00 UTC): no tentacle owns, stores or loads weights. One
//! copy of the weights is held centrally (`crate::neural::model`, one population) and every
//! tentacle's drones and core use it by reference: the drone reading of all windows is one shared
//! encoding kept by content key (`tentacle_pass::TentaclePass::scan`, run once per question by the
//! first drone tip that needs it), and each core holds an `Arc` to the shared model
//! (`tentacle_core::CoreShared`).
//!
//! G1 (`nmbrain::g1::G1::run`) is the convergence point and the only path between tentacles. Its
//! hand-over goes on to the reduce step: with the neural core, G1's forwarded spans as `chunk`
//! packets through the pass's bridge (as `--tentacle-core` does today); with a model but no core,
//! the pass's own judge over the kept scan; with no model, the hand-over is recorded in the
//! context map and adds no packet (first form: the exact answers are kept). Every tentacle's
//! cited packets reach the reduce step through its `g0_tx` and G1 in all modes. G2 is not yet a
//! graph node on this path (its first form selects chains after the reduce step behind
//! `--brain-chains`).
//!
//! Parallel rule: the executor runs one thread per tentacle node of a wave up to the processor
//! threads available at run time (`nmnte::Runtime`, `available_parallelism`, 12 on this machine,
//! so all eight tentacles run at once; more tentacles than threads queue); reads with the shared
//! weights go one at a time through the device gate (one card, one stream).
use crate::forage::Organism;
use crate::tentacle_core::{self, CoreShared, TentacleCore};
use crate::tentacle_pass::{self, Cand, Scan, Score, TentaclePass};
use nmbrain::Slots;
use nmbrain::g0::{G0, HeldChunk, SpanRef, TentacleSide};
use nmbrain::g1::{Budget as G1Budget, G1};
use nmbrain::gmap::WindowRecord;
use nmengn::placement::Device;
use nmjson::Value;
use nmnte::{Exec, FnExec, Node};
use nmpod::{Assignment, CoreKind, Finding, Job, Level, Reach, ReadCtx, Reader, Span, Unit, Window};
use nmsubs::Substrate;
use nmtype::{ComponentType, Fault, FaultKind, FrameView, Packet, Payload, SlotView, SubstrateView, Task};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// The served switch.
pub const FLAG: &str = "--one-tentacle";
/// Bytes per token when the tokenizer file cannot be opened (stated fallback; recorded).
pub const BYTES_PER_TOKEN_FALLBACK: usize = 4;
/// Node id prefix of G1 in the one-tentacle graph; a graph holding it is the unified graph.
pub const G1_NODE: &str = "nmbrain.g1#";

/// Where the exact readers read in the one tentacle system.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactReach {
    /// As in the territory path (the partition kept as the exact readers' reach, dealt by
    /// tentacle slot), so the exact answers are unchanged; the window is what the tentacle's drones
    /// and core read and report on. The default of this first form (nothing is lost).
    Territory,
    /// The specification's form: window-bound exact readers inside the tentacle's window,
    /// whole-code-base readers over the whole tree, each dealt to one tentacle. Measured first
    /// form: the exact answers change (note `runs/2026-11-10-neumor-note-p4-one-tentacle-first-form.md`).
    Window,
}
impl ExactReach {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Territory => "territory",
            Self::Window => "window",
        }
    }
}

fn bad(e: impl std::fmt::Display) -> Fault {
    Fault::new("nmapi.tentacle_unit", FaultKind::Execution, e.to_string())
}
fn json(v: &Payload) -> Option<&Value> {
    if let Payload::Json(j) = v { Some(j) } else { None }
}
fn num(x: usize) -> Value {
    Value::Number(x.to_string())
}
fn fnum(x: f64) -> Value {
    Value::Number(if x.is_finite() { format!("{x:.3}") } else { "0".into() })
}
fn text(s: &str) -> Value {
    Value::String(s.to_owned())
}
fn obj(m: BTreeMap<String, Value>) -> Value {
    Value::Object(m)
}

/// Is `--one-tentacle` given?
#[must_use]
pub fn from_args(args: &[String]) -> bool {
    args.iter().any(|a| a == FLAG)
}

/// Is `g` the one-tentacle graph (it holds the G1 node)?
#[must_use]
pub fn is_unified(g: &nmnte::Graph) -> bool {
    g.nodes.iter().any(|n| n.id.starts_with(G1_NODE))
}

/// One question's windows, as the tentacles laid them out.
struct Layout {
    windows: Vec<Window>,
    ms: f64,
    how: String,
}

/// One question's state, shared by its tentacles' nodes and G1 by task id.
pub struct Question {
    pub task: u64,
    pub question: String,
    sub: Arc<Substrate>,
    pass: Option<Arc<TentaclePass>>,
    tentacles: usize,
    territories: Vec<Vec<nmtype::Region>>,
    reach: ExactReach,
    layout: OnceLock<Result<Layout, String>>,
    /// The shared drone reading of every window with the central weights (with a pass bound).
    scan: OnceLock<Result<Arc<Scan>, String>>,
    shared: OnceLock<Result<Arc<CoreShared>, String>>,
    /// A core's state, handed to its `g0_tx`.
    pending: Mutex<BTreeMap<usize, Box<dyn TentacleSide>>>,
    /// The `g0_tx` states, held until G1 takes them.
    held: Mutex<BTreeMap<usize, Box<dyn TentacleSide>>>,
    facts: Mutex<BTreeMap<usize, BTreeMap<String, Value>>>,
    g1: Mutex<BTreeMap<String, Value>>,
    records: Mutex<Vec<WindowRecord>>,
    started: Instant,
}

impl Question {
    fn fact(&self, slot: usize, key: &str, v: Value) {
        if let Ok(mut f) = self.facts.lock() {
            f.entry(slot).or_default().insert(key.to_owned(), v);
        }
    }
    fn g1_fact(&self, key: &str, v: Value) {
        if let Ok(mut g) = self.g1.lock() {
            g.insert(key.to_owned(), v);
        }
    }
    fn layout(&self, ot: &OneTentacle) -> Result<&Layout, Fault> {
        self.layout
            .get_or_init(|| {
                let t = Instant::now();
                // With a pass bound the meaning lens follows the pass's own rule (the dense lane's
                // order, or empty), so the shared drone reading covers the very windows laid out.
                let mut l = ot.exact_layout(&self.sub, &self.question, self.tentacles, self.pass.is_none());
                l.ms = t.elapsed().as_secs_f64() * 1000.0;
                Ok(l)
            })
            .as_ref()
            .map_err(bad)
    }
    /// The neural mode in force: `Some(core)` with a pass bound, `None` with no model.
    fn neural(&self) -> Option<bool> {
        self.pass.as_ref().map(|p| p.settings.core)
    }
    /// The shared drone reading of every window (encoded once with the central weights, used by
    /// reference, kept by content key), and how many of its windows match the tentacles' windows.
    fn drone_scan(&self, ot: &OneTentacle) -> Result<Arc<Scan>, Fault> {
        let pass = self.pass.as_ref().ok_or_else(|| bad("no model loaded (no tentacle pass)"))?;
        self.scan
            .get_or_init(|| {
                let (scan, reused) = pass.scan(&self.sub, &self.question, &pass.settings).map_err(|e| e.message)?;
                let mine = self.layout(ot).map_err(|e| e.message)?;
                let keys = |w: &Window| w.spans.iter().map(|s| s.key.clone()).collect::<Vec<_>>();
                let same = mine.windows.iter().filter(|w| scan.windows.iter().find(|s| s.tentacle == w.tentacle).is_some_and(|s| s.place.iter().filter_map(|c| self.sub.chunks.get(*c).map(|x| x.key.clone())).collect::<Vec<_>>() == keys(w))).count();
                self.g1_fact(
                    "drone_reading",
                    text(&format!("shared drone reading with the central weights on {} ({}; encoded {} new tokens in {:.0} ms, reads {:.0} ms); windows matching the tentacles' {same} of {}", scan.device, if reused { "kept scan" } else { "new scan" }, scan.enc_tokens, scan.enc_ms, scan.read_ms, mine.windows.len())),
                );
                Ok(scan)
            })
            .clone()
            .map_err(bad)
    }
}

/// The adapter's state, kept by the organism while `--one-tentacle` is on.
pub struct OneTentacle {
    home: PathBuf,
    /// Token count per chunk content key (exact layout).
    tokens: Mutex<BTreeMap<String, usize>>,
    tokenizer: OnceLock<Option<nmengn::tokenizer::Tokenizer>>,
    questions: Mutex<BTreeMap<u64, Arc<Question>>>,
    /// Where the exact readers read (default `ExactReach::Territory`).
    pub reach: ExactReach,
}

impl std::fmt::Debug for OneTentacle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OneTentacle").field("home", &self.home).finish_non_exhaustive()
    }
}

/// A window of slot `k` from substrate chunk indices.
fn window_of(sub: &Substrate, k: usize, chunks: &[usize], tokens: &dyn Fn(usize) -> usize) -> Window {
    let spans = chunks
        .iter()
        .filter_map(|&c| {
            let ch = sub.chunks.get(c)?;
            let f = sub.files.get(ch.file)?;
            Some(Span { file: f.path.clone(), start: ch.start, end: ch.end, key: ch.key.clone(), tokens: tokens(c) })
        })
        .collect();
    Window { tentacle: k, lens: nmsubs::lens::NAMES.get(k).copied().unwrap_or("none").to_owned(), budget: nmsubs::lens::BUDGET, spans }
}

impl OneTentacle {
    #[must_use]
    pub fn new(home: &Path) -> Self {
        Self {
            home: home.to_path_buf(),
            tokens: Mutex::new(BTreeMap::new()),
            tokenizer: OnceLock::new(),
            questions: Mutex::new(BTreeMap::new()),
            reach: ExactReach::Territory,
        }
    }
    /// The same adapter with another exact reach.
    #[must_use]
    pub fn with_reach(mut self, reach: ExactReach) -> Self {
        self.reach = reach;
        self
    }
    /// Open (or reuse) the state of one question.
    #[expect(clippy::too_many_arguments, reason = "one question's inputs, passed once")]
    pub fn begin(
        &self,
        task: u64,
        question: &str,
        sub: Arc<Substrate>,
        pass: Option<Arc<TentaclePass>>,
        tentacles: usize,
        territories: Vec<Vec<nmtype::Region>>,
    ) -> Arc<Question> {
        let mut qs = match self.questions.lock() {
            Ok(q) => q,
            Err(p) => p.into_inner(),
        };
        Arc::clone(qs.entry(task).or_insert_with(|| {
            Arc::new(Question {
                task,
                question: question.to_owned(),
                sub,
                pass,
                tentacles,
                territories,
                reach: self.reach,
                layout: OnceLock::new(),
                scan: OnceLock::new(),
                shared: OnceLock::new(),
                pending: Mutex::new(BTreeMap::new()),
                held: Mutex::new(BTreeMap::new()),
                facts: Mutex::new(BTreeMap::new()),
                g1: Mutex::new(BTreeMap::new()),
                records: Mutex::new(Vec::new()),
                started: Instant::now(),
            })
        }))
    }
    fn question(&self, task: u64) -> Result<Arc<Question>, Fault> {
        self.questions
            .lock()
            .map_err(|_| bad("question state poisoned"))?
            .get(&task)
            .cloned()
            .ok_or_else(|| bad(format!("no one-tentacle state for task {task}")))
    }
    /// The window records of a question, for the GCX context map (exact windows at level 0, read windows as the
    /// reader recorded them).
    #[must_use]
    pub fn records(&self, task: u64) -> Vec<WindowRecord> {
        self.question(task).ok().and_then(|q| q.records.lock().ok().map(|r| r.clone())).unwrap_or_default()
    }
    /// Close a question: its record for the context map.
    #[must_use]
    pub fn finish(&self, task: u64) -> Option<Value> {
        let q = self.questions.lock().ok()?.remove(&task)?;
        let mut m = BTreeMap::new();
        let mode = match q.neural() {
            Some(true) => {
                "neural core in every tentacle (drone readers and exact readers; central weights by reference; exact fallback per tentacle on a fault)"
            }
            Some(false) => "drone readers and exact readers, exact cores (no --tentacle-core)",
            None => "exact: no model loaded, exact readers and the exact fallback core",
        };
        m.insert("mode".into(), text(mode));
        m.insert("exact_reach".into(), text(q.reach.name()));
        m.insert("threads".into(), num(std::thread::available_parallelism().map_or(1, usize::from)));
        let facts = q.facts.lock().map(|f| f.clone()).unwrap_or_default();
        if let Some(Ok(l)) = q.layout.get() {
            m.insert("layout".into(), text(&l.how));
            m.insert("layout_ms".into(), fnum(l.ms));
            m.insert("crossover".into(), num(nmpod::window::crossover(&l.windows)));
            let rows = l
                .windows
                .iter()
                .map(|w| {
                    let mut r = facts.get(&w.tentacle).cloned().unwrap_or_default();
                    r.insert("tentacle".into(), num(w.tentacle));
                    r.insert("lens".into(), text(&w.lens));
                    r.insert("spans".into(), num(w.spans.len()));
                    r.insert("tokens".into(), num(w.tokens()));
                    r.insert("files".into(), num(w.files().len()));
                    obj(r)
                })
                .collect();
            m.insert("windows".into(), Value::Array(rows));
        }
        m.insert("g1".into(), obj(q.g1.lock().map(|g| g.clone()).unwrap_or_default()));
        m.insert(
            "territories".into(),
            text("exact readers' reach under territory reach, G's coverage segments and the second-round refinement drones"),
        );
        m.insert("ms".into(), fnum(q.started.elapsed().as_secs_f64() * 1000.0));
        Some(obj(m))
    }
    fn token_counter(&self) -> Option<&nmengn::tokenizer::Tokenizer> {
        self.tokenizer
            .get_or_init(|| {
                nmengn::tokenizer::Tokenizer::open(
                    &nmtype::models_dir(&self.home).join(crate::neural::POPULATION).join("files").join("tokenizer.json"),
                )
                .ok()
            })
            .as_ref()
    }
    /// The windows of one question, laid out with no weights (module notes). `exact_meaning`
    /// fills the meaning lens from the exact embedder order when the dense lane does not run.
    fn exact_layout(&self, sub: &Substrate, question: &str, n: usize, exact_meaning: bool) -> Layout {
        let tok = self.token_counter();
        let tokens: Vec<usize> = {
            let mut known = match self.tokens.lock() {
                Ok(k) => k,
                Err(p) => p.into_inner(),
            };
            sub.chunks
                .iter()
                .map(|ch| {
                    *known.entry(ch.key.clone()).or_insert_with(|| {
                        let path = sub.files.get(ch.file).map_or("", |f| f.path.as_str());
                        let doc = nmsubs::dense::doc_text(path, &ch.text);
                        match tok {
                            Some(t) => t.encode(&doc).map_or(usize::MAX, |v| v.len()),
                            None => doc.len().div_ceil(BYTES_PER_TOKEN_FALLBACK),
                        }
                    })
                })
                .collect()
        };
        let lane = sub.dense.read().ok().and_then(|g| g.clone()).and_then(|l| l.embedding_order(sub, question));
        let (order, by) = match lane {
            Some(o) => (Some(o), "dense lane embedding order"),
            None if exact_meaning => (Some(sub.lexical_order(question)), "exact embedder order"),
            None => (None, "no order (empty, as the pass lays it out)"),
        };
        let lists = nmsubs::lens::orders(sub, question, order.as_deref(), &[]);
        let chosen = nmsubs::lens::assign(&lists, &|c| tokens.get(c).copied().unwrap_or(usize::MAX), nmsubs::lens::BUDGET);
        let windows = (0..n)
            .map(|k| window_of(sub, k, chosen.get(k).map_or(&[][..], Vec::as_slice), &|c| tokens.get(c).copied().unwrap_or(0)))
            .collect();
        let how = format!(
            "windows laid out with no weights: tokens by {}, meaning lens from the {by}",
            if tok.is_some() { "the base tokenizer file" } else { "the bytes fallback" }
        );
        Layout { windows, ms: 0.0, how }
    }
}

/// The attention-head drone readers of one tentacle, behind the one reader interface: its
/// window's chunks ranked by the shared drone reading's statistic, best first, `depth` of them;
/// the finding score is `1 / (1 + place)` so it sits on the exact readers' 0 to 1 scale. It holds
/// a reference to the shared reading, no weights.
pub struct DroneReader {
    scan: Arc<Scan>,
    sub: Arc<Substrate>,
    tentacle: usize,
    score: Score,
    depth: usize,
}
impl Reader for DroneReader {
    fn name(&self) -> &'static str {
        "drones"
    }
    fn reach(&self) -> Reach {
        Reach::Window
    }
    fn scan(&self, _: &ReadCtx<'_>) -> Vec<Finding> {
        let Some(w) = self.scan.windows.iter().find(|w| w.tentacle == self.tentacle) else { return Vec::new() };
        w.ranked(self.score)
            .into_iter()
            .take(self.depth)
            .enumerate()
            .filter_map(|(place, (k, _))| {
                let ch = self.sub.chunks.get(*w.place.get(k)?)?;
                let f = self.sub.files.get(ch.file)?;
                let src = format!("{}:{}", f.path, ch.start);
                #[expect(clippy::cast_precision_loss, reason = "places are small")]
                let score = 1.0 / (1.0 + place as f32);
                Some(Finding {
                    key: format!("drone:{src}"),
                    kind: "chunk".into(),
                    text: ch.text.clone(),
                    evidence: vec![format!("{src} lines {}..{}", ch.start, ch.end)],
                    source: src,
                    reader: "drones".into(),
                    score,
                    status: nmtype::Status::Inferred,
                })
            })
            .collect()
    }
}

/// The exact fallback core's `g0_tx` state over one window.
fn exact_g0(sub: &Substrate, w: &Window, r: &nmpod::ExactReport) -> (G0, WindowRecord) {
    let by_key: BTreeMap<&str, usize> = sub.chunks.iter().enumerate().map(|(i, c)| (c.key.as_str(), i)).collect();
    let held: Vec<HeldChunk> = w
        .spans
        .iter()
        .map(|s| HeldChunk {
            span: SpanRef { file: s.file.clone(), start: s.start, end: s.end, key: s.key.clone() },
            chunk: by_key.get(s.key.as_str()).copied().unwrap_or(usize::MAX),
            tokens: s.tokens,
        })
        .collect();
    let positive = r.scores.iter().filter(|s| **s > 0.0).count();
    let best = r.scores.iter().copied().fold(0.0_f32, f32::max);
    let rec = WindowRecord {
        tentacle: u8::try_from(w.tentacle).unwrap_or(u8::MAX),
        lens: w.lens.clone(),
        pose: "exact".into(),
        chunks: held.iter().map(|h| h.chunk).collect(),
        tokens: w.tokens(),
        level: 0,
        findings: positive,
        best,
        baseline: 0.0,
        no_information: r.no_information,
        files: w.files(),
    };
    // Report depth: the spans holding a finding, at least one, at most the served shortlist.
    let depth = positive.clamp(1, tentacle_pass::SERVED_SHORTLIST);
    let scores = vec![(nmpod::core::STATISTICS[0].to_owned(), r.scores.clone()), (nmpod::core::STATISTICS[1].to_owned(), r.agree.clone())];
    (G0::from_window(&rec, held, scores, std::slice::from_ref(&r.scores), depth), rec)
}

/// What every neural core of one question shares by reference (the central model, device, stop
/// targets, anchor, question, verification set), built as `tentacle_core::run_taped` builds it,
/// from public items only.
fn neural_shared(home: &Path, scan: &Scan) -> Result<Arc<CoreShared>, Fault> {
    let model = crate::neural::model(home)?;
    let dev = if model.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let sets = tentacle_pass::drones(home);
    let stop = nmengn::kvstore::score_targets(&model, sets.last_layer())?;
    let anchor = scan
        .windows
        .iter()
        .find_map(|w| w.window.as_ref().and_then(|x| x.chunks.first().cloned()))
        .ok_or_else(|| bad("no window with an anchor in the scan"))?;
    Ok(Arc::new(CoreShared {
        model: Some(model),
        dev,
        stop,
        anchor,
        question: scan.question.clone(),
        verify: sets.verification_set().to_vec(),
        reads: Mutex::new((0, 0.0, Vec::new())),
        names: BTreeMap::new(),
        tape: None,
        fill: scan.fill.clone(),
        read: crate::core_read::ReadOptions::default(),
    }))
}

/// The neural core of tentacle `slot` over its window of the shared drone reading.
fn neural_core(home: &Path, q: &Question, scan: &Arc<Scan>, slot: usize) -> Result<TentacleCore, Fault> {
    let w = scan
        .windows
        .iter()
        .find(|w| w.tentacle == slot)
        .ok_or_else(|| bad(format!("tentacle {slot}: the drone reading holds no window")))?;
    let shared = q.shared.get_or_init(|| neural_shared(home, scan).map_err(|e| e.message)).clone().map_err(bad)?;
    let lens = nmsubs::lens::NAMES.get(slot).copied().unwrap_or("lens");
    TentacleCore::from_scan(&q.sub, scan, w, tentacle_pass::drones(home).for_tentacle(lens).to_vec(), shared)
}

type Units = BTreeMap<String, (usize, Job, Arc<Mutex<Unit>>)>;

/// The component for one node of the one-tentacle graph: per tentacle the tentacle (window and
/// temporary specialisation), its drone tip, its core and its `g0_tx`; then G1.
/// # Errors
/// A node this adapter does not bind, or a part with no tentacle above it.
pub fn exec(ot: &Arc<OneTentacle>, org: &Organism, n: &Node, sub: &Arc<Substrate>, units: &Units) -> Result<Arc<dyn Exec>, Fault> {
    let module = n.id.split('#').next().unwrap_or("");
    let parent = n.parent.clone().unwrap_or_default();
    let tentacles = units.len();
    let ot = Arc::clone(ot);
    let unit_of = |id: &str| units.get(id).cloned().ok_or_else(|| bad(format!("{}: {id} is not a tentacle", n.id)));
    Ok(match (module, n.kind) {
        ("nmpod.unit", ComponentType::Tentacle) => {
            let (slot, _, unit) = unit_of(&n.id)?;
            Arc::new(FnExec(move |inp: &[Payload], _: &dyn FrameView| {
                let plan = inp.first().and_then(json).ok_or_else(|| bad("tentacle got no plan"))?;
                let task = Task::from_value(plan.get("task").map_err(bad)?)?;
                let names = |k: &str| -> Vec<Job> {
                    plan.get(k)
                        .and_then(Value::array)
                        .map(|a| a.iter().filter_map(|j| j.string().ok().and_then(Job::from_name)).collect())
                        .unwrap_or_default()
                };
                let (jobs, focus) = (names("jobs"), names("focus"));
                let q = ot.question(task.id)?;
                let layout = q.layout(&ot)?;
                let whole = nmpod::unit::plan_whole(&jobs, &focus, tentacles, 0);
                let u = unit.lock().map_err(|_| bad("tentacle lock"))?;
                let a = match q.reach {
                    ExactReach::Window => u.assign_window(&task, &layout.windows, slot, &jobs, &focus, &whole),
                    // Territory reach: an empty territory gives no assignment, as in the territory path.
                    ExactReach::Territory => q
                        .territories
                        .get(slot)
                        .filter(|r| !r.is_empty())
                        .and_then(|r| u.assign_window_in(&task, &layout.windows, r.clone(), slot, &jobs, &focus, &whole)),
                };
                Ok(Payload::Json(a.map_or(Value::Null, |a| a.to_value())))
            }))
        }
        ("nmpod.unit", ComponentType::Drone) => {
            let (slot, job, _) = unit_of(&parent)?;
            let tip = Mutex::new(Unit::new(&n.id, Level::Drone, job, Arc::clone(&org.reflexes), 1));
            let (sub, slots, home) = (Arc::clone(sub), Arc::clone(&org.slots), org.home.clone());
            Arc::new(FnExec(move |inp: &[Payload], _: &dyn FrameView| {
                let a = Assignment::from_value(inp.first().and_then(json).ok_or_else(|| bad("tip got no assignment"))?)?;
                let q = ot.question(a.task.id)?;
                let names = Unit::reader_names(&a);
                let exact: Vec<Box<dyn Reader>> = names.iter().filter_map(|r| nmpod::tip::reader(r)).collect();
                let t = tip.lock().map_err(|_| bad("tip lock"))?;
                let (found, readers) = t.sense_readers(&a, &exact, sub.as_ref() as &dyn SubstrateView, slots.as_ref() as &dyn SlotView);
                let mut reader_names: Vec<Value> = names.iter().map(|r| text(r)).collect();
                // Attention-head drone readers when a model is loaded (the central weights by
                // reference, through the shared drone reading of every window).
                let drones = match q.pass.as_ref() {
                    Some(p) => match q.drone_scan(&ot) {
                        Ok(scan) => {
                            let score = if p.settings.core { Score::Calibrated } else { p.settings.score };
                            let d: Vec<Box<dyn Reader>> = vec![Box::new(DroneReader {
                                scan,
                                sub: Arc::clone(&sub),
                                tentacle: slot,
                                score,
                                depth: p.settings.shortlist.max(1),
                            })];
                            let lens = nmsubs::lens::NAMES.get(slot).copied().unwrap_or("lens");
                            q.fact(slot, "drone_heads", num(tentacle_pass::drones(&home).for_tentacle(lens).len()));
                            reader_names.insert(0, text("drones"));
                            t.sense_readers(&a, &d, sub.as_ref() as &dyn SubstrateView, slots.as_ref() as &dyn SlotView).0
                        }
                        Err(e) => {
                            q.fact(slot, "drone_fault", text(&e.message));
                            Vec::new()
                        }
                    },
                    None => Vec::new(),
                };
                q.fact(slot, "readers", Value::Array(reader_names));
                q.fact(slot, "exact_findings", num(found.len()));
                q.fact(slot, "drone_findings", num(drones.len()));
                let mut m = BTreeMap::new();
                m.insert("assignment".to_owned(), a.to_value());
                m.insert("readers".to_owned(), num(readers));
                m.insert("findings".to_owned(), Value::Array(found.iter().map(Finding::to_value).collect()));
                m.insert("drones".to_owned(), Value::Array(drones.iter().map(Finding::to_value).collect()));
                Ok(Payload::Json(obj(m)))
            }))
        }
        ("nmapi.tentacle_core", _) => {
            let (slot, _, unit) = unit_of(&parent)?;
            let (sub, slots, home) = (Arc::clone(sub), Arc::clone(&org.slots), org.home.clone());
            Arc::new(FnExec(move |inp: &[Payload], _: &dyn FrameView| {
                let v = inp.first().and_then(json).ok_or_else(|| bad("core got no findings"))?;
                if matches!(v, Value::Null) {
                    return Ok(Payload::Json(Value::Null));
                }
                let a = Assignment::from_value(v.get("assignment").map_err(bad)?)?;
                let readers = usize::try_from(v.get("readers").and_then(Value::integer).unwrap_or(0)).unwrap_or(0);
                let parse = |k: &str| -> Result<Vec<Finding>, Fault> {
                    v.get(k).and_then(Value::array).map(|x| x.iter().map(Finding::from_value).collect()).unwrap_or_else(|_| Ok(Vec::new()))
                };
                let (found, drones) = (parse("findings")?, parse("drones")?);
                let q = ot.question(a.task.id)?;
                let window = a.window.clone().unwrap_or_default();
                let all: Vec<Finding> = found.iter().chain(&drones).cloned().collect();
                let report = nmpod::core::exact_report(&window, &all);
                let out = unit.lock().map_err(|_| bad("tentacle lock"))?.process(
                    &a,
                    found,
                    readers,
                    sub.as_ref() as &dyn SubstrateView,
                    slots.as_ref() as &dyn SlotView,
                );
                let (g0, rec) = exact_g0(&sub, &window, &report);
                if q.pass.is_none()
                    && let Ok(mut r) = q.records.lock()
                {
                    r.push(rec);
                }
                let mut kind = CoreKind::Exact;
                let mut why = if q.pass.is_none() {
                    "no model loaded: exact fallback core".to_owned()
                } else {
                    "no --tentacle-core: exact core over drone and exact readings".to_owned()
                };
                let side: Box<dyn TentacleSide> = match (q.neural(), q.pass.is_some()) {
                    (Some(true), true) => match q.drone_scan(&ot).and_then(|scan| neural_core(&home, &q, &scan, slot)) {
                        Ok(c) => {
                            kind = CoreKind::Neural;
                            why = format!("neural tentacle core, central weights by reference ({})", c.depth_why);
                            Box::new(c)
                        }
                        Err(e) => {
                            why = format!("neural core fault ({}): exact fallback core", e.message);
                            Box::new(g0)
                        }
                    },
                    _ => Box::new(g0),
                };
                q.fact(slot, "core", text(kind.name()));
                q.fact(slot, "core_why", text(&why));
                q.fact(
                    slot,
                    "exact_report",
                    obj([
                        ("inside".to_owned(), num(report.inside)),
                        ("outside".to_owned(), num(report.outside)),
                        ("no_information".to_owned(), Value::Bool(report.no_information)),
                        ("spans_with_findings".to_owned(), num(report.scores.iter().filter(|s| **s > 0.0).count())),
                    ]
                    .into_iter()
                    .collect()),
                );
                q.fact(slot, "packets", num(out.packets.len() + out.questions.len()));
                if let Ok(mut p) = q.pending.lock() {
                    p.insert(slot, side);
                }
                let packets: Vec<Value> = out.packets.iter().chain(&out.questions).map(Packet::to_value).collect();
                Ok(Payload::Json(obj([
                    ("task".to_owned(), Value::Number(a.task.id.to_string())),
                    ("tentacle".to_owned(), num(slot)),
                    ("packets".to_owned(), Value::Array(packets)),
                ]
                .into_iter()
                .collect())))
            }))
        }
        ("nmbrain.g0", _) => {
            let (slot, _, _) = unit_of(&parent)?;
            Arc::new(FnExec(move |inp: &[Payload], _: &dyn FrameView| {
                let Some(v) = inp.first().and_then(json).filter(|v| !matches!(v, Value::Null)) else {
                    return Ok(Payload::Packets(Vec::new()));
                };
                let task = v.get("task").and_then(Value::integer).map_err(bad)?;
                let packets: Vec<Packet> =
                    v.get("packets").and_then(Value::array).map_err(bad)?.iter().map(Packet::from_value).collect::<Result<_, _>>()?;
                let q = ot.question(task)?;
                let side = q.pending.lock().ok().and_then(|mut p| p.remove(&slot));
                if let Some(side) = side {
                    q.fact(slot, "g0", text(side.id()));
                    q.fact(slot, "held", num(side.held().len()));
                    if let Ok(mut h) = q.held.lock() {
                        h.insert(slot, side);
                    }
                }
                Ok(Payload::Packets(packets))
            }))
        }
        ("nmbrain.g1", _) => {
            let (sub, slots): (Arc<Substrate>, Arc<Slots>) = (Arc::clone(sub), Arc::clone(&org.slots));
            Arc::new(FnExec(move |inp: &[Payload], _: &dyn FrameView| {
                let plan = inp.iter().find_map(json).ok_or_else(|| bad("G1 got no plan"))?;
                let task = Task::from_value(plan.get("task").map_err(bad)?)?;
                let mut packets: Vec<Packet> =
                    inp.iter().filter_map(|p| if let Payload::Packets(v) = p { Some(v.clone()) } else { None }).flatten().collect();
                let q = ot.question(task.id)?;
                packets.extend(g1_round(&ot, &q, &sub, &slots, &task)?);
                Ok(Payload::Packets(packets))
            }))
        }
        (m, k) => return Err(bad(format!("{}: the one-tentacle adapter binds no {m} of kind {}", n.id, k.name()))),
    })
}

/// G1 over the question's `g0_tx` states; returns the hand-over packets for the reduce step.
fn g1_round(ot: &OneTentacle, q: &Question, sub: &Substrate, slots: &Slots, task: &Task) -> Result<Vec<Packet>, Fault> {
    let t = Instant::now();
    let sides = q.held.lock().map(|mut h| std::mem::take(&mut *h)).unwrap_or_default();
    q.g1_fact("tentacles", num(sides.len()));
    if sides.is_empty() {
        return Ok(Vec::new());
    }
    let neural_core = q.neural() == Some(true);
    let layout = q.layout(ot)?;
    let toks: Vec<usize> = layout.windows.iter().flat_map(|w| w.spans.iter().map(|s| s.tokens)).collect();
    let mean = toks.iter().sum::<usize>().checked_div(toks.len()).unwrap_or(0).max(1);
    // Forward room: what the pass's cut forwards when the neural core runs (as `tentacle_core::run`
    // sets it), else the question's pack tokens.
    let room = match (neural_core, q.pass.as_ref()) {
        (true, Some(p)) => p.settings.per_tentacle.max(1) * layout.windows.len() * mean,
        _ => usize::try_from(task.budget.tokens).unwrap_or(usize::MAX),
    };
    let mut boxes: Vec<Box<dyn TentacleSide>> = sides.into_values().collect();
    let budget = G1Budget::derive(boxes.len(), Some(room), Some(mean), None);
    let outcome = G1::new(tentacle_core::codebase(sub)).run(&q.question, &mut boxes, &budget);
    let h = &outcome.handover;
    let refused_msg = h.messages().iter().filter(|e| nmtype::class::boundaries::MESSAGE_CHANNEL.admit(*e).is_err()).count();
    let (out, how) = match (q.pass.as_ref(), neural_core) {
        (Some(p), true) => {
            let by_key: BTreeMap<&str, usize> = sub.chunks.iter().enumerate().map(|(i, c)| (c.key.as_str(), i)).collect();
            let cands: Vec<Cand> = h
                .items
                .iter()
                .enumerate()
                .filter_map(|(place, f)| {
                    let chunk = *by_key.get(f.span.key.as_str())?;
                    let tentacle = f.from.strip_prefix("g0_t").and_then(|x| x.parse::<usize>().ok()).map_or(0, |x| x.saturating_sub(1));
                    Some(Cand { tentacle, chunk, score: f.combined, place, lines: Vec::new() })
                })
                .collect();
            let order: Vec<usize> = (0..cands.len()).collect();
            if let Some(Ok(scan)) = q.scan.get() {
                let mut r = scan.records.clone();
                for w in &mut r {
                    let id = nmbrain::g0::g0_id(w.tentacle);
                    w.findings = h.items.iter().filter(|f| f.from == id).count();
                    w.best = h.items.iter().filter(|f| f.from == id).map(|f| f.confidence).fold(0.0_f32, f32::max);
                }
                if let Ok(mut slot) = q.records.lock() {
                    *slot = r;
                }
            }
            (tentacle_pass::bridge(sub, &cands, &order, &p.settings), "G1's forwarded spans as chunk packets (bridge)")
        }
        (Some(p), false) => {
            // No core: the pass's own judge over the kept drone reading (no read is repeated).
            let pass = p.read_as(sub, slots, &q.question, &p.settings)?;
            if let Ok(mut r) = q.records.lock() {
                *r = pass.windows;
            }
            (pass.packets, "the pass judge over the kept drone reading (no --tentacle-core)")
        }
        (None, _) => (Vec::new(), "recorded only; the tentacles' cited packets go on through G1 (exact first form)"),
    };
    let refused_pk = out.iter().filter(|p| nmtype::class::boundaries::CHAIN_INPUT.admit(*p).is_err()).count();
    q.g1_fact("rounds", num(usize::try_from(h.rounds).unwrap_or(0)));
    q.g1_fact("ceiling", num(usize::try_from(outcome.ceiling).unwrap_or(0)));
    q.g1_fact("stop", text(&h.stop));
    q.g1_fact("forwarded", Value::Array(h.items.iter().map(|f| text(&format!("{} {}", f.from, f.span.label()))).collect()));
    q.g1_fact("isolation_gate", num(h.gate_direct));
    q.g1_fact("budget", text(&budget.why));
    q.g1_fact("refused_wrong_class", num(refused_msg + refused_pk));
    q.g1_fact("handover_to_reduce", text(how));
    q.g1_fact("handover_packets", num(out.len()));
    q.g1_fact("ms", fnum(t.elapsed().as_secs_f64() * 1000.0));
    Ok(out)
}

/// Switch the one tentacle system on for an organism: the adapter's state, then the one-tentacle
/// graph (`designs/anatomy.json` `assemblies.forage`) bound and published. Returns what was set.
/// # Errors
/// The graph does not load, bind or pass the Integrity Validator; the organism keeps the graph it
/// had.
pub fn enable(org: &Organism) -> Result<String, Fault> {
    enable_with(org, ExactReach::Territory)
}

/// `enable` with a chosen exact reach.
/// # Errors
/// As `enable`.
pub fn enable_with(org: &Organism, reach: ExactReach) -> Result<String, Fault> {
    let (path, g) = crate::forage::forage_assembly(&org.home)?;
    if !is_unified(&g) {
        return Err(bad(format!("{} is not the one-tentacle graph (no {G1_NODE} node)", path.display())));
    }
    let before = org.one_tentacle.read().ok().and_then(|o| o.clone());
    if let Ok(mut o) = org.one_tentacle.write() {
        *o = Some(Arc::new(OneTentacle::new(&org.home).with_reach(reach)));
    }
    let published = org.bind(&g, &org.sub()).and_then(|comps| org.nte.publish(g.clone(), comps).map(|_| ()));
    if let Err(e) = published {
        if let Ok(mut o) = org.one_tentacle.write() {
            *o = before;
        }
        return Err(e);
    }
    let tentacles = g.nodes.iter().filter(|n| n.kind == ComponentType::Tentacle).count();
    let neural = org.tentacle_pass.read().ok().and_then(|p| p.as_ref().map(|p| p.settings.core));
    Ok(format!(
        "one tentacle system on: {tentacles} tentacles, autonomous pathways in parallel, each with its own window (lens per question), drone tip, core and g0_tx (no weights owned; one central copy used by reference), reporting only to G1 ({}); exact readers' reach {}; graph {}",
        match neural {
            Some(true) => "neural core in every tentacle (central weights by reference), exact fallback on a fault",
            Some(false) => "drone readers (central weights by reference), exact cores",
            None => "no model loaded: exact readers and the exact fallback core",
        },
        reach.name(),
        path.strip_prefix(&org.home).unwrap_or(&path).display()
    ))
}

/// Section check of the one tentacle system (2026-11-10, processor only, no weights): on a
/// three-file tree, with `--one-tentacle` on and no model, (a) the one-tentacle graph binds and
/// passes the Integrity Validator, with exactly one tentacle type and no edge between tentacles;
/// (b) a question is answered from the constant's definition; (c) every tentacle reports through
/// its `g0_tx` to G1, the isolation gate reads 0 and no chunk is in two windows; (d) the switch
/// off (a fresh organism) gives the answer from the same file. Values:
/// `tests/section-checks/nmapi-tentacle-unit/report.txt`.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    use std::fmt::Write as _;
    let dir = nmtype::check_dir("nmapi-tentacle-unit").map_err(|e| e.to_string())?;
    let tree = dir.join("tree");
    let files = [
        (
            "alpha/src/lib.rs",
            "//! Alpha crate.\npub const GATE_LIMIT: u32 = 42;\npub fn open_gate(x: u32) -> bool {\n    x < GATE_LIMIT\n}\n",
        ),
        ("beta/src/lib.rs", "//! Beta crate.\npub fn close_gate() -> bool {\n    false\n}\n"),
        ("README.md", "# Fixture\nTwo crates for the one tentacle check.\n"),
    ];
    for (p, t) in files {
        let path = tree.join(p);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(&path, t).map_err(|e| e.to_string())?;
    }
    let home = nmtype::nano_root();
    let mem = dir.join("memory");
    let question = "What is the value of GATE_LIMIT?";
    let mut rep = String::from("nmapi.tentacle_unit (one tentacle system, first form)\n");
    let off = Organism::open_with(&home, &tree, &[], false, &mem).map_err(|e| e.to_string())?;
    let a_off = off.ask(question, 2000).map_err(|e| e.to_string())?;
    let org = Organism::open_with(&home, &tree, &[], false, &mem).map_err(|e| e.to_string())?;
    let note = enable(&org).map_err(|e| e.to_string())?;
    let _ = writeln!(rep, "{note}");
    let snap = org.nte.snapshot();
    let tentacle_nodes = snap.nodes().iter().filter(|n| n.kind == ComponentType::Tentacle).count();
    let node_count = snap.nodes().len();
    let cross = snap
        .edges()
        .iter()
        .filter(|e| {
            nmnte::topology::unit_of(&snap.graph, &e.from).zip(nmnte::topology::unit_of(&snap.graph, &e.to)).is_some_and(|(a, b)| a != b)
        })
        .count();
    drop(snap);
    let _ = writeln!(rep, "graph: {tentacle_nodes} tentacles, {node_count} nodes, edges between tentacles {cross}");
    if tentacle_nodes != 8 || cross != 0 {
        return Err(format!("nmapi.tentacle_unit: graph with {tentacle_nodes} tentacles and {cross} edges between tentacles"));
    }
    let a = org.ask(question, 2000).map_err(|e| e.to_string())?;
    let cited = |p: &nmtype::Pack| -> Vec<String> {
        let mut f: Vec<String> = p.packets.iter().map(|x| x.provenance().source.split(':').next().unwrap_or("").to_owned()).collect();
        f.sort();
        f.dedup();
        f
    };
    let rec = a.context_map.get("one_tentacle").cloned().unwrap_or(Value::Null);
    let g1 = rec.get("g1").cloned().unwrap_or(Value::Null);
    let gate = g1.get("isolation_gate").and_then(Value::integer).unwrap_or(u64::MAX);
    let crossover = rec.get("crossover").and_then(Value::integer).unwrap_or(u64::MAX);
    let reported = g1.get("tentacles").and_then(Value::integer).unwrap_or(0);
    // Territory reach: a tentacle whose territory is empty gets no assignment, as in the territory
    // path, so the reports are the non-empty territories (3 on this tree).
    let expected = nmpod::territory::partition(&org.sub().files.iter().map(|f| f.path.clone()).collect::<Vec<_>>(), 8)
        .iter()
        .filter(|t| !t.is_empty())
        .count();
    // Window reach: every tentacle with a window reports.
    let win = Organism::open_with(&home, &tree, &[], false, &mem).map_err(|e| e.to_string())?;
    enable_with(&win, ExactReach::Window).map_err(|e| e.to_string())?;
    let w = win.ask(question, 2000).map_err(|e| e.to_string())?;
    let wrec = w.context_map.get("one_tentacle").cloned().unwrap_or(Value::Null);
    let w_reported = wrec.get("g1").and_then(|g| g.get("tentacles")).and_then(Value::integer).unwrap_or(0);
    let _ = writeln!(rep, "window reach: cited {:?}, faults {}, g0_tx reports {w_reported}", cited(&w.pack), w.faults.len());
    let _ = writeln!(
        rep,
        "on: cited {:?}, faults {}, isolation gate {gate}, crossover {crossover}, g0_tx reports {reported}",
        cited(&a.pack),
        a.faults.len()
    );
    let _ = writeln!(rep, "off: cited {:?}", cited(&a_off.pack));
    let _ = writeln!(rep, "record: {}", rec.encode().unwrap_or_default());
    std::fs::write(dir.join("report.txt"), &rep).map_err(|e| e.to_string())?;
    if !a.pack.packets.iter().any(|p| p.text().contains("42")) || !a.faults.is_empty() {
        return Err(format!("nmapi.tentacle_unit: the answer missed the constant or faulted ({} faults)", a.faults.len()));
    }
    if gate != 0 || crossover != 0 || usize::try_from(reported).ok() != Some(expected) || w_reported != 8 || !w.faults.is_empty() {
        return Err(format!(
            "nmapi.tentacle_unit: gate {gate}, crossover {crossover}, reports {reported} of {expected} (territory reach), {w_reported} of 8 (window reach)"
        ));
    }
    if !cited(&a.pack).contains(&"alpha/src/lib.rs".to_owned()) || !cited(&a_off.pack).contains(&"alpha/src/lib.rs".to_owned()) {
        return Err("nmapi.tentacle_unit: on and off do not both cite the defining file".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_check() {
        assert_eq!(check(), Ok(()));
    }

    /// Switch off against switch on (no model) over the questions in
    /// `scratch/one-tentacle/questions.txt` on the mini's own tree, interleaved per question:
    /// cited files of each pack, time per question, windows and readers per tentacle. Writes
    /// `scratch/one-tentacle/compare.txt`. Processor only, no weights.
    #[test]
    #[ignore = "measurement on the Nano's own tree; run by hand"]
    fn compare_off_on() {
        use std::fmt::Write as _;
        let home = nmtype::nano_root();
        let questions = std::fs::read_to_string(home.join("scratch/one-tentacle/questions.txt")).expect("questions");
        let mem = home.join("memory");
        let skip = ["neuromorph-nano".to_owned()];
        let off = Organism::open_with(&home, &home, &skip, false, &mem).expect("open off");
        let on = Organism::open_with(&home, &home, &skip, false, &mem).expect("open on");
        let note = enable(&on).expect("enable");
        let win = Organism::open_with(&home, &home, &skip, false, &mem).expect("open window arm");
        let note_w = enable_with(&win, ExactReach::Window).expect("enable window arm");
        let mut out = format!("{note}\nwindow arm: {note_w}\n");
        let (mut same_w, mut t_w, mut same_packets) = (0, 0.0, 0);
        let files = |p: &nmtype::Pack| -> Vec<String> {
            let mut f: Vec<String> = p.packets.iter().map(|x| x.provenance().source.split(':').next().unwrap_or("").to_owned()).collect();
            f.sort();
            f.dedup();
            f
        };
        let (mut same, mut n, mut t_off, mut t_on) = (0, 0, 0.0, 0.0);
        for q in questions.lines().filter(|l| !l.trim().is_empty()) {
            n += 1;
            let t = Instant::now();
            let a = off.ask(q, 4000).expect("ask off");
            let ms_off = t.elapsed().as_secs_f64() * 1000.0;
            let t = Instant::now();
            let b = on.ask(q, 4000).expect("ask on");
            let ms_on = t.elapsed().as_secs_f64() * 1000.0;
            let t = Instant::now();
            let c = win.ask(q, 4000).expect("ask window arm");
            let ms_w = t.elapsed().as_secs_f64() * 1000.0;
            t_w += ms_w;
            t_off += ms_off;
            t_on += ms_on;
            let fc = files(&c.pack);
            let eq_w = files(&a.pack) == fc;
            same_w += usize::from(eq_w);
            let key_text = |p: &nmtype::Pack| {
                p.packets.iter().map(|x| (x.key().to_owned(), x.text().to_owned(), x.provenance().source.clone())).collect::<Vec<_>>()
            };
            let packets_same = key_text(&a.pack) == key_text(&b.pack);
            same_packets += usize::from(packets_same);
            let _ = writeln!(
                out,
                "\nq{n} pack packets (key, text, source, order) identical off and on: {packets_same}; window arm {ms_w:.0} ms, cited files {}",
                if eq_w { "same" } else { "different" }
            );
            if !eq_w {
                let fa = files(&a.pack);
                let _ = writeln!(
                    out,
                    "  window arm only off: {:?}\n  window arm only on: {:?}",
                    fa.iter().filter(|x| !fc.contains(x)).collect::<Vec<_>>(),
                    fc.iter().filter(|x| !fa.contains(x)).collect::<Vec<_>>()
                );
            }
            let (fa, fb) = (files(&a.pack), files(&b.pack));
            let eq = fa == fb;
            same += usize::from(eq);
            let _ = writeln!(
                out,
                "\nq{n} {}: off {ms_off:.0} ms, on {ms_on:.0} ms, withheld off {} on {}, faults on {}",
                if eq { "same cited files" } else { "DIFFERENT cited files" },
                a.pack.withheld,
                b.pack.withheld,
                b.faults.len()
            );
            if !eq {
                let _ = writeln!(
                    out,
                    "  only off: {:?}\n  only on: {:?}",
                    fa.iter().filter(|x| !fb.contains(x)).collect::<Vec<_>>(),
                    fb.iter().filter(|x| !fa.contains(x)).collect::<Vec<_>>()
                );
            }
            let rec = b.context_map.get("one_tentacle").cloned().unwrap_or(Value::Null);
            let _ = writeln!(out, "  record: {}", rec.encode().unwrap_or_default());
        }
        let d = f64::from(u32::try_from(n).unwrap_or(1));
        let _ = writeln!(
            out,
            "\nsame cited files {same} of {n} (territory reach; packets identical {same_packets} of {n}), {same_w} of {n} (window reach); mean ms off {:.0}, on {:.0}, window arm {:.0}",
            t_off / d,
            t_on / d,
            t_w / d
        );
        std::fs::write(home.join("scratch/one-tentacle/compare.txt"), &out).expect("write");
        println!("{out}");
    }
}
