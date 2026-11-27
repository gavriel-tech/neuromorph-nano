//! Learning-weight units over the shared model (S8; guide 10, 10.1 to 10.6): the engine
//! side of `nmlearn`. Loads and unloads a unit's quantized files into its component node,
//! scores the reranker loss over checked outcomes for the forward-only fitter, scores a
//! held-out shard for the Step Check, runs the isolation probes, registers the unit's card in
//! the executor's Component Registry, and serves `nm_unit`. The sealed capability fixtures
//! are never read here.
use crate::forage::Organism;
use crate::neural::{POPULATION, model};
use nmengn::adapter::Adapter;
use nmengn::model::{Model, RunOpts, values};
use nmengn::placement::Device;
use nmjson::Value;
use nmlearn::isolation::{Probe, bytes_of};
use nmlearn::{Loss, UnitCard, UnitWeights, Units};
use nmnte::CardState;
use nmtype::{Fault, FaultKind};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// The S8 target node: the last feed-forward's down projection of the reference assembly.
/// Since 2026-11-10 (step P0) the target of a bound model is `target(m)`: the node its assembly
/// names as `learn_target`, with this id as the stated default.
pub const TARGET: &str = "nmengn.components#L27.down";

/// The learning target of `m`: the `learn_target` its assembly names (default `TARGET`'s
/// instance), which must be a `linear` node.
/// # Errors
/// The assembly has no such node, or it is not a linear component.
pub fn target(m: &Model) -> Result<String, Fault> {
    let id = m.role_node("learn_target")?;
    widths(m, &id)?;
    Ok(id)
}

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmapi.learn", FaultKind::Config, m)
}
fn s(t: &str) -> Value {
    Value::String(t.to_owned())
}
fn n(x: impl ToString) -> Value {
    Value::Number(x.to_string())
}
fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

/// Input and output widths of a linear node.
/// # Errors
/// No such node or not a linear.
pub fn widths(m: &Model, node: &str) -> Result<(usize, usize), Fault> {
    let b = m.bound.iter().find(|b| b.node.id == node).ok_or_else(|| bad(format!("no node {node}")))?;
    let l = b.lin.as_ref().filter(|_| b.node.op == "linear").ok_or_else(|| bad(format!("{node} is not a linear component")))?;
    Ok((l.k, l.n))
}
/// The engine adapter of a unit's weights.
/// # Errors
/// Shapes that do not fit.
pub fn adapter(card: &UnitCard, w: &UnitWeights) -> Result<Arc<Adapter>, Fault> {
    Adapter::new(&card.id, &card.target, w.rank, w.k, w.n, w.down.clone(), w.up.clone(), w.scale).map(Arc::new)
}
/// Compatibility of a unit with the model: base hash, target present, widths.
/// # Errors
/// A mismatch, named.
pub fn compat(m: &Model, card: &UnitCard, w: Option<&UnitWeights>) -> Result<(usize, usize), Fault> {
    nmnpcs::compat::check_unit_base(&card.id, &card.base_sha256, &m.pop.card.sha_main)?;
    let (k, nn) = widths(m, &card.target)?;
    if let Some(w) = w
        && (w.k != k || w.n != nn || w.rank != card.rank)
    {
        return Err(bad(format!(
            "{}: files are {}x{} rank {}, the node takes {k} in and {nn} out, the card says rank {}",
            card.id, w.n, w.k, w.rank, card.rank
        )));
    }
    Ok((k, nn))
}
/// Load a unit's quantized files into its node (a rejected unit is refused).
/// # Errors
/// Unknown, rejected, incompatible, or files that fail their hash.
pub fn load(m: &Model, units: &mut Units, id: &str) -> Result<UnitWeights, Fault> {
    let card = units.get(id).cloned().ok_or_else(|| bad(format!("no unit {id}")))?;
    if card.status == "rejected" {
        return Err(Fault::new("nmapi.learn", FaultKind::Policy, format!("unit {id} is rejected and does not load")));
    }
    if card.files.is_empty() {
        return Err(bad(format!("unit {id} has no files yet (unfitted); fit it first")));
    }
    let mut w = UnitWeights::read(&card, &units.unit_dir(id))?;
    // The unit's parameter overlay (`memory/overlays/`): masked whole it does not load; otherwise its
    // edits and masks apply to `down` and `up` as they load (the unit's files are never written).
    if let Some(ov) = nmengn::overlay::effective(&m.pop.home, id) {
        if ov.whole_masked() {
            return Err(Fault::new("nmapi.learn", FaultKind::Policy, format!("unit {id} is masked whole in its parameter overlay and does not load")));
        }
        ov.apply("down", "F32", &mut w.down);
        ov.apply("up", "F32", &mut w.up);
    }
    if let Ok(mut g) = UNIT_OVERLAYS.lock() {
        g.insert(id.to_owned(), unit_overlay(&m.pop.home, id));
    }
    compat(m, &card, Some(&w))?;
    m.set_unit(&card.target, Some(adapter(&card, &w)?))?;
    units.mark_loaded(id)?;
    Ok(w)
}
/// The overlay identity each loaded unit runs with in this process (empty: none).
static UNIT_OVERLAYS: std::sync::Mutex<std::collections::BTreeMap<String, String>> = std::sync::Mutex::new(std::collections::BTreeMap::new());

fn unit_overlay(home: &std::path::Path, id: &str) -> String {
    nmengn::overlay::effective(home, id).map(|o| nmengn::overlay::identity(&o)).unwrap_or_default()
}

/// Applies the parameter overlays of the units loaded on `m` live: a unit whose overlay changed
/// since it was set on its node is set again from its files with the overlay applied (masked
/// whole: taken off its node, the base's output bit for bit). `force` sets every loaded unit again
/// (after the model was bound again). Returns one line per unit set.
pub fn apply_units(m: &Model, home: &std::path::Path, force: bool) -> Vec<String> {
    let units = Units::open(home);
    let mut lines = Vec::new();
    for card in units.cards.iter().filter(|c| c.loaded && !c.files.is_empty() && c.status != "rejected") {
        let now = unit_overlay(home, &card.id);
        let seen = UNIT_OVERLAYS.lock().ok().and_then(|g| g.get(&card.id).cloned());
        // Only units this process set on a node (`load`); a unit another server loaded is not here.
        let Some(seen) = seen else { continue };
        if !force && seen == now {
            continue;
        }
        let ov = nmengn::overlay::effective(home, &card.id);
        let r = if ov.as_ref().is_some_and(nmengn::overlay::Overlay::whole_masked) {
            m.set_unit(&card.target, None).map(|()| "taken off its node (masked whole)".to_owned())
        } else {
            (|| {
                let mut w = UnitWeights::read(card, &units.unit_dir(&card.id))?;
                if let Some(o) = &ov {
                    o.apply("down", "F32", &mut w.down);
                    o.apply("up", "F32", &mut w.up);
                }
                compat(m, card, Some(&w))?;
                m.set_unit(&card.target, Some(adapter(card, &w)?))?;
                Ok::<String, Fault>(if ov.is_some() { "set again with its overlay".to_owned() } else { "set again from its files".to_owned() })
            })()
        };
        match r {
            Ok(what) => lines.push(format!("unit {}: {what}", card.id)),
            Err(e) => lines.push(format!("unit {}: not set again ({e})", card.id)),
        }
        if let Ok(mut g) = UNIT_OVERLAYS.lock() {
            g.insert(card.id.clone(), now);
        }
    }
    lines
}

/// Unload a unit from its node (the exact rollback of its effect).
/// # Errors
/// Unknown unit.
pub fn unload(m: &Model, units: &mut Units, id: &str) -> Result<(), Fault> {
    let card = units.get(id).cloned().ok_or_else(|| bad(format!("no unit {id}")))?;
    m.set_unit(&card.target, None)?;
    units.mark_unloaded(id)
}
/// `serve --unit <id>`: the engine must be bound (`--neural`); loads the unit and registers
/// its card in the Component Registry as a candidate (promoted when its status is fitted).
/// # Errors
/// As `load`.
pub fn load_for_serve(org: &Organism, id: &str) -> Result<String, Fault> {
    // 2026-11-26 (plugs): the registered units' compartment switched off keeps every one of them out.
    if !crate::plugs::read(&org.home).compartment_on(crate::plugs::SHARED) {
        return Err(Fault::new("nmapi.learn", FaultKind::Policy, format!("unit {id}: the shared learning units are switched off (Learning, Learned units)")));
    }
    let m = model(&org.home)?;
    let mut units = Units::open(&org.home);
    let w = load(&m, &mut units, id)?;
    let card = units.get(id).cloned().ok_or_else(|| bad(format!("no unit {id}")))?;
    let state = if card.status == "fitted" { CardState::Promoted } else { CardState::Candidate };
    register_component(org, &card, state)?;
    observe(
        org,
        &card.target,
        "load",
        &format!(
            "unit {id} ({}) status {}",
            card.files.iter().map(|f| f.1.get(..8).unwrap_or("")).collect::<Vec<_>>().join(","),
            card.status
        ),
        "serve",
    )?;
    Ok(format!("{id} loaded on {} (rank {}, {}x{}, status {}); registered {state:?}", card.target, w.rank, w.n, w.k, card.status))
}
/// Register the unit's component card in the executor's registry.
/// # Errors
/// A malformed card or no registry.
pub fn register_component(org: &Organism, card: &UnitCard, state: CardState) -> Result<(), Fault> {
    let v = card.to_component_card();
    let mut c = nmtype::Card::from_value(&v)?;
    c.descriptor.card_hash = nmtype::card_hash(&v);
    org.nte.register(c, state)
}
/// A Parameter Map note for a unit event.
/// # Errors
/// The map's log could not be written while recording.
pub fn observe(org: &Organism, target: &str, kind: &str, detail: &str, run: &str) -> Result<(), Fault> {
    org.parmap.lock().map_err(|_| bad("parmap lock"))?.observe("parameter", target, kind, detail, run)
}

/// One reranker outcome: the question, the positive document, the negatives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Item {
    pub id: String,
    pub name: String,
    pub question: String,
    pub positive: String,
    pub negatives: Vec<String>,
}
/// Items of `designs/train-rerank.json` under `key` (`train` or `heldout`); a document is
/// `source text`, the form the reranker sees in the pack.
/// # Errors
/// A malformed file.
pub fn items(home: &Path, key: &str) -> Result<Vec<Item>, Fault> {
    let v = nmjson::parse(&std::fs::read(home.join("designs/train-rerank.json")).map_err(|e| bad(e.to_string()))?)
        .map_err(|e| bad(e.to_string()))?;
    let doc = |d: &Value| -> Result<String, Fault> {
        Ok(format!(
            "{} {}",
            d.get("source").and_then(Value::string).map_err(|e| bad(e.to_string()))?,
            d.get("text").and_then(Value::string).map_err(|e| bad(e.to_string()))?
        ))
    };
    let mut out = Vec::new();
    for it in v.get(key).and_then(Value::array).map_err(|e| bad(format!("{key}: {e}")))? {
        let g = |k: &str| it.get(k).and_then(Value::string).map(str::to_owned).map_err(|e| bad(format!("{k}: {e}")));
        let negatives =
            it.get("negatives").and_then(Value::array).map_err(|e| bad(e.to_string()))?.iter().map(doc).collect::<Result<Vec<_>, _>>()?;
        out.push(Item {
            id: g("id")?,
            name: g("name")?,
            question: g("question")?,
            positive: doc(it.get("positive").map_err(|e| bad(e.to_string()))?)?,
            negatives,
        });
    }
    Ok(out)
}
/// Softmax cross-entropy of the first score (the positive) against the rest, in f64.
#[must_use]
pub fn cross_entropy(scores: &[f32]) -> f64 {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = f64::from(max) + scores.iter().map(|x| (f64::from(*x) - f64::from(max)).exp()).sum::<f64>().ln();
    lse - f64::from(scores.first().copied().unwrap_or(0.0))
}
fn score(m: &Model, device: Device, q: &str, d: &str) -> Result<f32, Fault> {
    nmengn::roles::rerank_on(m, device, q, d)
}

/// The reranker loss for a unit's `up` over training items (the engine runs the forwards;
/// the unit is set on the model before every loss read).
pub struct Scorer {
    pub model: Arc<Model>,
    pub card: UnitCard,
    pub weights: UnitWeights,
    pub items: Vec<Item>,
    pub device: Device,
    /// Negatives per item during the fit (the research note: one).
    pub negatives: usize,
    pub forwards: u64,
}
impl Loss for Scorer {
    fn loss(&mut self, up: &[f32], batch: &[usize]) -> Result<f64, Fault> {
        let mut w = self.weights.clone();
        w.up = up.to_vec();
        self.model.set_unit(&self.card.target, Some(adapter(&self.card, &w)?))?;
        let mut total = 0.0;
        for &i in batch {
            let it = self.items.get(i).ok_or_else(|| bad(format!("no item {i}")))?;
            let mut sc = vec![score(&self.model, self.device, &it.question, &it.positive)?];
            for d in it.negatives.iter().take(self.negatives) {
                sc.push(score(&self.model, self.device, &it.question, d)?);
            }
            self.forwards += u64::try_from(sc.len()).unwrap_or(0);
            total += cross_entropy(&sc);
        }
        #[expect(clippy::cast_precision_loss, reason = "batch sizes are small")]
        let mean = total / batch.len().max(1) as f64;
        Ok(mean)
    }
}

/// A held-out shard scored with whatever unit is loaded now.
#[derive(Clone, Debug, PartialEq)]
pub struct Shard {
    /// Share of items whose positive scored first.
    pub score: f32,
    /// Mean four-way cross-entropy.
    pub loss: f64,
    /// Position of the positive per item (0 is first).
    pub ranks: Vec<usize>,
    pub forwards: u64,
    pub ms: f64,
}
/// Score `items` (positive against every negative).
/// # Errors
/// An engine fault.
pub fn shard(m: &Model, device: Device, items: &[Item]) -> Result<Shard, Fault> {
    let t0 = Instant::now();
    let (mut ranks, mut loss, mut forwards) = (Vec::new(), 0.0, 0_u64);
    for it in items {
        let mut sc = vec![score(m, device, &it.question, &it.positive)?];
        for d in &it.negatives {
            sc.push(score(m, device, &it.question, d)?);
        }
        forwards += u64::try_from(sc.len()).unwrap_or(0);
        let pos = sc.first().copied().unwrap_or(0.0);
        ranks.push(sc.iter().skip(1).filter(|x| **x >= pos).count());
        loss += cross_entropy(&sc);
    }
    #[expect(clippy::cast_precision_loss, reason = "item counts are small")]
    let score = ranks.iter().filter(|r| **r == 0).count() as f32 / items.len().max(1) as f32;
    #[expect(clippy::cast_precision_loss, reason = "item counts are small")]
    let mean = loss / items.len().max(1) as f64;
    Ok(Shard { score, loss: mean, ranks, forwards, ms: t0.elapsed().as_secs_f64() * 1000.0 })
}
/// A shard as the Step Check's `before` or `after` value, with extra fields.
#[must_use]
pub fn shard_value(sh: &Shard, extra: Vec<(&str, Value)>) -> Value {
    let mut pairs = vec![
        ("score", n(sh.score)),
        ("loss", n(sh.loss)),
        ("ranks", Value::Array(sh.ranks.iter().map(n).collect())),
        ("forwards", n(sh.forwards)),
        ("ms", n(format!("{:.0}", sh.ms))),
    ];
    pairs.extend(extra);
    obj(pairs)
}

/// The three fixed isolation probes: the reranker prompt's final hidden state (every
/// position) and the last-position logits, as bytes.
/// # Errors
/// An engine fault.
pub fn probes(m: &Model, device: Device) -> Result<Vec<Probe>, Fault> {
    let pairs = [
        ("Where is the struct Coordinator defined?", "framework/contxt/mapcor/src/lib.rs:69 pub struct Coordinator {"),
        ("What value is the constant SCOPE_WIDE set to?", "framework/contxt/ctxstk/src/plan.rs:64 map.query(&name, want * SCOPE_WIDE)"),
        ("Which crates depend on nmtype?", "architecture/params/README.md:3 The Parameter Map and the Map Coordinator."),
    ];
    let targets = [(m.role_node("final_norm")?, "y".to_owned()), (m.role_node("head")?, "logits".to_owned())];
    let mut out = Vec::new();
    for (i, (q, d)) in pairs.iter().enumerate() {
        let ids = m.pop.tokenizer.encode(&nmengn::roles::prompt(q, d)).map_err(|e| bad(e.to_string()))?;
        let r = m.forward(device, &ids, &targets, &RunOpts::default())?;
        let mut bytes = Vec::new();
        for t in &targets {
            bytes.extend(bytes_of(values(r.get(t).ok_or_else(|| bad("probe output"))?)?));
        }
        out.push(Probe { label: format!("probe{i}"), bytes });
    }
    Ok(out)
}

/// Where learned pieces are saved for this session: `units/learned/` for the live memory folder;
/// beside the memory folder of a test session otherwise (as the decision ledger).
#[must_use]
pub fn learned_dir(home: &std::path::Path, memory_dir: &std::path::Path) -> std::path::PathBuf {
    if memory_dir == home.join("memory") {
        nmlearn::learned::store(home)
    } else if memory_dir.parent() == Some(nmtype::project::projects_dir(home).as_path()) {
        // A project memory (Nano, 2026-11-15, E1) keeps its learned pieces inside its own folder.
        memory_dir.join("learned")
    } else {
        let name = memory_dir.file_name().and_then(|n| n.to_str()).unwrap_or("session");
        memory_dir.with_file_name(format!("{name}-learned"))
    }
}

/// The component the Core Brain's learned pieces attach to.
pub const CORE_BRAIN: &str = "nmbrain.core-brain#0";

/// `nm_unit` ops `calibrate`, `tune`, `learned` (2026-11-12, learned pieces as adapter cartridges):
/// each learning layer runs its own gate on the session's decision ledger and its piece is saved
/// only when the gate passes; `learned` lists what is saved.
fn learned_op(org: &Organism, op: &str) -> Result<Value, Fault> {
    use nmlearn::route::{Layer, SignalKind, admit};
    if !org.memory_on {
        return Err(bad("memory is off for this session: no ledger is read and nothing is saved"));
    }
    let dir = org.memory.lock().map(|m| m.dir.clone()).map_err(|_| bad("memory lock"))?;
    let ledger = crate::whole::ledger_path(&org.home, &dir);
    let store = learned_dir(&org.home, &dir);
    if op == "learned" {
        return Ok(obj(vec![
            ("store", s(&store.display().to_string())),
            ("pieces", Value::Array(nmlearn::learned::list(&store).iter().map(|x| s(x)).collect())),
        ]));
    }
    let ds = nmlearn::ledger::load(&ledger)?;
    let policy = nmlearn::cartridge::Policy::load(&org.home);
    let loaded =
        nmlearn::cartridge::Loaded::population(&org.home, "qwen3-reranker-0.6b", vec![CORE_BRAIN.into()]).map_err(|e| e.to_fault())?;
    let num = |x: f64| Value::Number(format!("{x:.4}"));
    let un = |x: usize| Value::Number(x.to_string());
    if op == "calibrate" {
        // L4 read-out calibration: verdicts train it (router), its own gate decides.
        admit(SignalKind::Verdict, Layer::L4Readout)?;
        let with: Vec<&nmlearn::ledger::Decision> = ds.iter().filter(|d| d.correct.is_some() && !d.rerank.is_empty()).collect();
        if with.len() < nmlearn::tuner::MIN_N {
            return Ok(obj(vec![
                ("saved", Value::Bool(false)),
                ("reason", s(&format!("{} decisions with a verdict; the gate needs {}", with.len(), nmlearn::tuner::MIN_N))),
            ]));
        }
        let xs: Vec<Vec<f64>> = with.iter().map(|d| nmlearn::calibrate::features(&d.rerank)).collect();
        let ys: Vec<bool> = with.iter().map(|d| d.correct == Some(true)).collect();
        let base: Vec<f64> = with.iter().map(|d| f64::from(d.rerank.iter().fold(0.0_f32, |a, x| a.max(*x)))).collect();
        let g = nmlearn::calibrate::gate(&xs, &ys, &base, 1.0);
        let report = obj(vec![
            ("base_log_loss", num(g.base)),
            ("calibrated_log_loss", num(g.calibrated)),
            ("shuffled_log_loss", num(g.shuffled)),
            ("better", un(g.better)),
            ("worse", un(g.worse)),
            ("sign_p", num(g.p)),
            ("keep", Value::Bool(g.keep)),
        ]);
        if !g.keep {
            return Ok(obj(vec![("saved", Value::Bool(false)), ("gate", report)]));
        }
        let c = nmlearn::calibrate::Calibrator::fit(&xs, &ys, 1.0);
        let (shape, numbers) = nmlearn::learned::calibrator_numbers(&c);
        let id = nmlearn::learned::next_id(&store, "l4-calibrator");
        let recipe = obj(vec![
            ("layer", s("L4 read-out calibration")),
            ("signal", s("verdict")),
            ("method", s("logistic map by Newton's method, ridge 1.0, standardised features (nmlearn::calibrate)")),
            ("features", s("log-odds of the reranker's best, gap to the second, share above one half")),
            ("data", Value::Array(with.iter().filter_map(|d| d.episode.as_deref()).map(s).collect())),
            ("ledger", s(&ledger.display().to_string())),
            ("gate", report.clone()),
        ]);
        let piece = nmlearn::learned::Piece {
            label: nmlearn::learned::label(&id, &loaded, CORE_BRAIN, "L4 read-out calibrator of the Core Brain's reranker confidence"),
            shape,
            numbers,
            recipe,
        };
        let at = nmlearn::learned::save(&store, &piece, g.keep, &policy)?;
        return Ok(obj(vec![("saved", Value::Bool(true)), ("id", s(&id)), ("at", s(&at.display().to_string())), ("gate", report)]));
    }
    // L1 decision tuning: verification and cost train it (router); verdicts only gate it.
    admit(SignalKind::Verification, Layer::L1Decision)?;
    admit(SignalKind::Cost, Layer::L1Decision)?;
    let t = nmlearn::tuner::Tuner::from_ledger(&ds);
    let best = t.best();
    let arms = Value::Array(
        (0..nmlearn::tuner::ARMS.len())
            .map(|a| {
                let (worse, better) = t.paired.get(a).copied().unwrap_or((0, 0));
                obj(vec![
                    ("threshold", num(f64::from(nmlearn::tuner::ARMS[a].min(1e9)))),
                    ("verified_share", t.estimate(a).map_or(Value::Null, num)),
                    ("replays", un(t.value.get(a).map_or(0, |v| v.1))),
                    ("worse", un(worse)),
                    ("better", un(better)),
                    ("gate", Value::Bool(t.gate(a))),
                ])
            })
            .collect(),
    );
    let report = obj(vec![
        ("decisions", un(ds.len())),
        ("verdicts", un(t.verdicts)),
        ("best", un(best)),
        ("default", un(nmlearn::tuner::DEFAULT)),
        ("arms", arms),
    ]);
    if best == nmlearn::tuner::DEFAULT {
        return Ok(obj(vec![("saved", Value::Bool(false)), ("reason", s("the default leads; nothing to save")), ("tuner", report)]));
    }
    let id = nmlearn::learned::next_id(&store, "l1-settings");
    let recipe = obj(vec![
        ("layer", s("L1 decision tuning")),
        ("signal", s("re-read verification and cost; verdicts as the gate only")),
        ("method", s("exact replay of the logged second-look scores (nmlearn::tuner)")),
        ("setting", s("second-look threshold")),
        ("ledger", s(&ledger.display().to_string())),
        ("gate", report.clone()),
    ]);
    let piece = nmlearn::learned::Piece {
        label: nmlearn::learned::label(&id, &loaded, CORE_BRAIN, "L1 settings: the Core Brain's second-look threshold"),
        shape: vec![1],
        numbers: vec![nmlearn::tuner::ARMS[best].min(1e9)],
        recipe,
    };
    let at = nmlearn::learned::save(&store, &piece, true, &policy)?;
    Ok(obj(vec![("saved", Value::Bool(true)), ("id", s(&id)), ("at", s(&at.display().to_string())), ("tuner", report)]))
}

/// `nm_unit`: list, register, load, unload; calibrate, tune, learned.
/// # Errors
/// A refused or failed operation, named.
pub fn tool(org: &Organism, v: &Value) -> Result<Value, Fault> {
    let arg = |k: &str| v.get(k).and_then(Value::string).ok();
    let mut units = Units::open(&org.home);
    match arg("op").unwrap_or("list") {
        op @ ("calibrate" | "tune" | "learned") => learned_op(org, op),
        "register" => {
            let id = arg("id").ok_or_else(|| bad("register needs an id"))?.to_owned();
            let m = model(&org.home)?;
            let target = match arg("target") {
                Some(t) => t.to_owned(),
                None => target(&m)?,
            };
            let rank = usize::try_from(v.get("rank").and_then(Value::integer).unwrap_or(8)).unwrap_or(8);
            let card = UnitCard::new(
                &id,
                POPULATION,
                &m.pop.card.sha_main,
                arg("slot").unwrap_or("reranker"),
                &target,
                rank,
                v.get("seed").and_then(Value::integer).unwrap_or(8),
            );
            compat(&m, &card, None)?;
            units.register(card.clone())?;
            register_component(org, &card, CardState::Candidate)?;
            observe(org, &target, "register", &format!("unit {id} rank {rank}"), "nm_unit")?;
            Ok(card.to_value())
        }
        "load" => {
            let id = arg("id").ok_or_else(|| bad("load needs an id"))?;
            Ok(obj(vec![("unit", s(id)), ("loaded", s(&load_for_serve(org, id)?))]))
        }
        "unregister" => {
            let id = arg("id").ok_or_else(|| bad("unregister needs an id"))?;
            units.unregister(id)?;
            observe(org, TARGET, "unregister", &format!("unit {id} (unfitted, no files)"), "nm_unit")?;
            Ok(obj(vec![("unit", s(id)), ("registered", Value::Bool(false))]))
        }
        "unload" => {
            let id = arg("id").ok_or_else(|| bad("unload needs an id"))?;
            let m = model(&org.home)?;
            unload(&m, &mut units, id)?;
            observe(org, TARGET, "unload", &format!("unit {id}"), "nm_unit")?;
            Ok(obj(vec![("unit", s(id)), ("loaded", Value::Bool(false))]))
        }
        _ => {
            let engine: Vec<Value> = crate::neural::bound_model()
                .map(|m| m.units_loaded().into_iter().map(|(node, id)| obj(vec![("node", s(&node)), ("unit", s(&id))])).collect())
                .unwrap_or_default();
            Ok(obj(vec![
                ("units", units.to_value()),
                ("loaded", Value::Array(units.loaded().iter().map(|u| s(u)).collect())),
                ("engine_units", Value::Array(engine)),
                ("outcomes_logged", n(nmlearn::outcomes::since(&org.home, 0).len())),
            ]))
        }
    }
}

/// Section check of step P0 item 3 (2026-11-10), on the released Qwen3-0.6B (CPU): every
/// fixed node-name lookup fails with a named fault, not a wrong result or a panic, when the
/// bound assembly lacks the node: the role paths (`final_norm` for the base embedding, `head`
/// for the base rerank and greedy decoding, `label` for the card reranker and its prefilled
/// set, `pool` for the card embedding), the learning target, encoding through a layer with no
/// attention component, encoding with attention components that cover only some key-value
/// heads, and a window over a chunk that holds only some key-value heads. An assembly that
/// names its final norm under another id in its own `names` data gives the same embedding, bit
/// for bit, as the default name. Values: `tests/section-checks/nmapi-names/report.txt`.
///
/// # Errors
/// One fault line.
pub fn check_names() -> Result<(), String> {
    use std::fmt::Write as _;
    let mut rep = String::from("nmapi.learn.names (step P0 item 3)\n");
    let r = names_check(&mut rep);
    let _ = writeln!(rep, "result: {}", r.as_ref().map_or_else(|e| format!("FAIL {e}"), |()| "ok".to_owned()));
    let dir = nmtype::check_dir("nmapi-names").map_err(|e| e.to_string())?;
    std::fs::write(dir.join("report.txt"), rep).map_err(|e| e.to_string())?;
    r
}
fn names_check(rep: &mut String) -> Result<(), String> {
    use nmengn::assembly::{Assembly, Shape, generate, nid};
    use nmengn::kvstore::{ChunkKv, Window, encode_through};
    use std::fmt::Write as _;
    let home = nmtype::nano_root();
    let pop = nmengn::Population::open(&home, POPULATION, true, false).map_err(|e| e.to_string())?;
    let d = pop.dims;
    let one = Shape { layers: 1, heads: d.heads, kv_heads: d.kv_heads, head_dim: d.head_dim };
    let cut = |a: &mut Assembly, insts: &[&str]| {
        let ids: Vec<String> = insts.iter().map(|i| nid(i)).collect();
        a.nodes.retain(|n| !ids.contains(&n.id));
        a.edges.retain(|e| !ids.contains(&e.from) && !ids.contains(&e.to));
    };
    // A one-layer assembly with no final norm and no head (so no pool, label or L27.down).
    let mut small = generate("p0-no-roles", one, 1);
    cut(&mut small, &["final_norm", "head"]);
    let m = Model::bind(&pop, small).map_err(|e| format!("the small assembly did not bind: {e}"))?;
    let dev = Device::Cpu;
    let text = "pub fn room(&self) -> usize { self.cap - self.bytes }";
    let ids = pop.tokenizer.encode(text).map_err(|e| e.to_string())?;
    let mut expect = |what: &str, r: Result<String, Fault>, needle: &str| -> Result<(), String> {
        match r {
            Ok(v) => Err(format!("{what}: answered ({v}) on an assembly without the node")),
            Err(e) if e.to_string().contains(needle) => {
                let _ = writeln!(rep, "{what}: {e}");
                Ok(())
            }
            Err(e) => Err(format!("{what}: refused, but not by name: {e}")),
        }
    };
    expect(
        "base embedding (final_norm)",
        nmengn::roles::embed_on(&m, dev, text).map(|v| format!("{} values", v.len())),
        "has no final_norm node",
    )?;
    expect("base rerank (head)", nmengn::roles::rerank_on(&m, dev, "q", text).map(|v| v.to_string()), "has no head node")?;
    expect("greedy decoding (head)", nmengn::decode::greedy(&m, dev, &ids, 2, &[]).map(|g| format!("{:?}", g.tokens)), "has no head node")?;
    expect(
        "card reranker (label)",
        nmengn::roles::card_rerank_on(&m, dev, nmengn::roles::CARD_INSTRUCTION, "q", text).map(|v| v.to_string()),
        "has no label node",
    )?;
    expect(
        "card reranker set (label)",
        nmengn::roles::card_rerank_set(&m, dev, nmengn::roles::CARD_INSTRUCTION, "q", &[text.to_owned()]).map(|v| format!("{v:?}")),
        "has no label node",
    )?;
    expect(
        "card embedding (pool)",
        nmengn::roles::card_embed_on(&m, dev, text).map(|v| format!("{} values", v.len())),
        "has no pool node",
    )?;
    expect("learning target", target(&m), "has no learn_target node")?;
    expect(
        "encoding through layer 5",
        encode_through(&m, dev, None, &ids, 5).map(|c| format!("{} layers", c.layers())),
        "no score component on layer 5",
    )?;
    // Attention components over half the key-value heads only: path b removed, path a feeds
    // the residual directly.
    let mut half = generate("p0-half-heads", one, 2);
    cut(&mut half, &["L0.qb", "L0.kb", "L0.vb", "L0.qnormb", "L0.knormb", "L0.ropeb", "L0.scoreb", "L0.ob", "L0.attn_sum1"]);
    half.edges.push(nmengn::assembly::Edge { from: nid("L0.oa"), from_port: "y".into(), to: nid("L0.res1"), to_port: "b".into() });
    let hm = Model::bind(&pop, half).map_err(|e| format!("the half-heads assembly did not bind: {e}"))?;
    expect(
        "encoding with half the key-value heads",
        encode_through(&hm, dev, None, &ids, 0).map(|c| format!("{} layers", c.layers())),
        "not recorded exactly once",
    )?;
    let width = (d.kv_heads / 2) * d.head_dim;
    let part = Arc::new(ChunkKv { tokens: 2, anchor: 0, k_pre: vec![vec![0; 2 * width]], v: vec![vec![0; 2 * width]] });
    expect(
        "window over a chunk of half the key-value heads",
        Window::build(&m, &[part]).map(|w| format!("{} positions", w.len)),
        "every key-value head",
    )?;
    // The final norm named in the assembly's own data, under another id.
    let plain = Model::bind(&pop, generate("p0-plain", one, 1)).map_err(|e| e.to_string())?;
    let mut renamed = generate("p0-renamed", one, 1);
    let (old, new) = (nid("final_norm"), nid("out_norm"));
    for n in &mut renamed.nodes {
        if n.id == old {
            n.id.clone_from(&new);
        }
    }
    for e in &mut renamed.edges {
        if e.from == old {
            e.from.clone_from(&new);
        }
        if e.to == old {
            e.to.clone_from(&new);
        }
    }
    renamed.names.insert("final_norm".into(), "out_norm".into());
    let back = Assembly::from_value(&renamed.to_value("p0")).map_err(|e| e.to_string())?;
    if back != renamed {
        return Err("an assembly with names does not round-trip through its file form".into());
    }
    let rm = Model::bind(&pop, back).map_err(|e| e.to_string())?;
    let (a, b) = (
        nmengn::roles::embed_on(&plain, dev, text).map_err(|e| e.to_string())?,
        nmengn::roles::embed_on(&rm, dev, text).map_err(|e| e.to_string())?,
    );
    let same = !a.is_empty() && a.iter().map(|x| x.to_bits()).eq(b.iter().map(|x| x.to_bits()));
    let _ = writeln!(
        rep,
        "final_norm named out_norm in the assembly data: embedding of {} values, bit for bit equal to the default name: {same}",
        b.len()
    );
    if !same {
        return Err("the embedding through a named final norm differs from the default".into());
    }
    if plain.role_node("learn_target").is_ok() {
        return Err("a one-layer assembly found a learning target".into());
    }
    Ok(())
}
