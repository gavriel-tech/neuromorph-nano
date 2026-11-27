//! Parallel Context Foraging, the organism loop end to end (guide 17.1, 17.4, 2, 3, 4, 6;
//! PLAN section 5, 15.1). The executed graph is data: `designs/anatomy.json` names the
//! forage assembly (`designs/assemblies/forage-s2.json`), and this module only binds a
//! component to each connected node by its kind and id. One Core Brain lives as long as the
//! organism, so held conflicts, the mailbox and the verified set carry across questions
//! (guide 6.4). A substrate sync publishes a new snapshot bound to the new substrate.
use crate::coordinator::{Budgets, ContextSummary, ModelSummary};
use nmbrain::gmap::GMap;
use nmbrain::{CoreBrain, Slots};
use nmjson::Value;
use nmnte::{Components, Exec, FnExec, Graph, Nte, Registry, Runtime};
use nmpod::{Assignment, Finding, Job, Level, Neuropod, ReflexTable, Unit};
use nmsubs::Substrate;
use nmsubs::sync::SyncReport;
use nmtype::{
    Budget, ComponentType, Fault, FaultKind, FrameView, Hop, Neuropod as NeuropodInterface, Pack, Packet, Payload, Plan, SlotView,
    SubstrateView, Task, TaskKind,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

/// What the forager saw of its frame: nodes that ran, nodes skipped, faults, packet count.
/// What one forage frame did: nodes that ran, skipped nodes, faults, raw packet count, snapshot
/// version, and milliseconds per node from the frame's hops (the homeostasis latency map).
type FrameRecord = (Vec<String>, Vec<String>, Vec<Fault>, usize, u64, BTreeMap<String, f32>);

/// What one question produced.
#[derive(Debug)]
pub struct Answer {
    pub pack: Pack,
    pub emitted: Packet,
    pub questions: Vec<Packet>,
    pub version: u64,
    pub ran: Vec<String>,
    pub skipped: Vec<String>,
    pub faults: Vec<Fault>,
    pub raw_packets: usize,
    /// The episode this answer became, and how the memory gate took it.
    pub episode: Option<String>,
    pub written: String,
    pub recalled: usize,
    /// The snapshot version the task's frame ran on (`version` is the one current at its end).
    pub ran_version: u64,
    /// The GCX context map of this question (CM1), as JSON.
    pub context_map: Value,
    /// What the Core Brain side did (`serve --brain-chains`, 2026-11-10); `None` when off.
    pub brain: Option<crate::brain_side::BrainRecord>,
}

/// The organism's memory: episodic, working, temp notes, the gate, its own identity, the tier design.
pub struct MemoryState {
    pub dir: PathBuf,
    pub episodic: nmmem::Episodic,
    pub working: nmmem::Working,
    /// Temporary notes: promoted into `episodic` and `hierarchy` when used often, else cleared.
    pub temp: nmmem::Temp,
    /// Saved memory as folders, most used at the top (1-core, 2-active, 3-archive).
    pub hierarchy: nmmem::Hierarchy,
    pub gate: nmmem::Gate,
    pub identity: nmmem::Identity,
    pub design: Value,
    /// The run-day counter (memory revision 2): days on which the system ran.
    pub rundays: nmmem::runday::RunDays,
    /// Memory revision 2 (2026-11-12): journal, book and views, written beside revision 1 while
    /// both run; `None` with writes off or when the journal is held by another session.
    pub mem2: Option<nmmem::mem2::Memory2>,
    /// Why revision 2 did not open, when it did not.
    pub mem2_fault: Option<String>,
    /// Embeddings of stored questions, by episode id (recall by meaning, in RAM, 2026-11-10).
    pub vectors: BTreeMap<String, Vec<f32>>,
    /// Hierarchy items of answers stored before the reply, written after it (`session::settle`).
    pub pending: Vec<crate::session::Pending>,
}

impl Answer {
    /// A bare answer around a pack, for section checks of the delivery code.
    #[must_use]
    pub fn for_check(pack: Pack, emitted: Packet) -> Self {
        Self {
            pack,
            emitted,
            questions: Vec::new(),
            version: 1,
            ran: Vec::new(),
            skipped: Vec::new(),
            faults: Vec::new(),
            raw_packets: 0,
            episode: None,
            written: String::new(),
            recalled: 0,
            ran_version: 1,
            context_map: Value::Null,
            brain: None,
        }
    }
}

/// The assembled organism: substrate, slots, reflexes, executor, one Core Brain.
pub struct Organism {
    pub home: PathBuf,
    pub assembly: Graph,
    sub: RwLock<Arc<Substrate>>,
    pub slots: Arc<Slots>,
    pub reflexes: Arc<ReflexTable>,
    pub nte: Arc<Nte>,
    pub brain: Mutex<CoreBrain>,
    pub tentacles: usize,
    pub memory: Mutex<MemoryState>,
    pub memory_on: bool,
    pub objectives: Mutex<Vec<(u64, String, String)>>,
    /// Nodes under quarantine: bound to an inert stand-in until restarted.
    pub quarantined: Mutex<std::collections::BTreeSet<String>>,
    pub homeostasis: Mutex<nmbrain::homeostasis::Homeostasis>,
    pub parmap: Mutex<nmnpcs::ParMap>,
    /// Tentacle working memory (T1), written by each tentacle's note node.
    pub memory_handle: Arc<Mutex<nmmem::Working>>,
    pub workers: usize,
    pub max_run_ms: u32,
    /// Substrate version at which each file last changed (filled by `sync`; stale-claim marks).
    pub changed_at: Mutex<BTreeMap<String, u64>>,
    /// Own changes predicted by content (2026-11-12, motor division): a Step Check that passes
    /// records the files' post-edit hashes; each sync tells own changes from outside ones.
    pub own: Mutex<nmtype::ActionLedger>,
    /// The last sync's changed files: (own, from outside).
    pub last_sync_origin: Mutex<(Vec<String>, Vec<String>)>,
    /// Every node's output of the last forage frame (`trace` command, 2026-11-09): what each
    /// tentacle, drone tip and pipeline produced, read without Claude Code.
    pub last_frame: Arc<Mutex<BTreeMap<String, Payload>>>,
    /// Second-round refinement: `nbv` (next-best-view over the GCX context map, the default since 2026-11-10: it
    /// tied on the small sealed CM1 set and won on the large-context benchmark, 20 of 20 against
    /// 19 with no refinement, `runs/2026-11-10-LC1-large-context.md`), `uniform` (the spread-out
    /// baseline) or `off`; `serve --refine`.
    pub refine: RwLock<String>,
    /// Recall by meaning (2026-11-10): off by default (it gained answers but lost a decoy on the
    /// development set, `runs/2026-11-10-R2-memory-by-meaning.md`); `serve --recall-meaning`.
    pub recall_meaning: std::sync::atomic::AtomicBool,
    /// Word recall ranked by memory revision 2 (`nmmem::recall2`) instead of revision 1, the floor
    /// kept (`serve --recall-rank r2`, 2026-11-13; off until measured).
    pub recall_r2: std::sync::atomic::AtomicBool,
    /// Feeds each new substrate version to the dense lane's indexer (2026-11-10), when running.
    pub dense_feed: Mutex<Option<std::sync::mpsc::Sender<Arc<Substrate>>>>,
    /// The tentacle pass (2026-11-10, GCX build step 1): eight lens windows read by the drone
    /// heads, joined to the question's packets and recorded on G's window axis. Off (`None`)
    /// until it is qualified; `serve --tentacles`, `trace --tentacles`.
    pub tentacle_pass: RwLock<Option<Arc<crate::tentacle_pass::TentaclePass>>>,
    /// The whole path on the served path (`serve --whole`, 2026-11-12); `None` when off.
    pub whole: RwLock<Option<Arc<crate::whole::WholePath>>>,
    /// The last pass's one-line note (device, tokens, times), for the trace.
    pub last_pass: Arc<Mutex<String>>,
    /// The Core Brain side (2026-11-10, first form): G2 selection, chains in the Core Brain,
    /// behaviour log, packs, cartridges, preferences, settings. Off (`None`) unless
    /// `--brain-chains`; with it off the path below is unchanged (control and fallback).
    pub brain_side: RwLock<Option<Arc<crate::brain_side::BrainSide>>>,
    /// The one tentacle system (2026-11-10, step P4 first form, `crate::tentacle_unit`): each
    /// tentacle owns its window, drone tip, core and `g0_tx` and reports only to G1, inside the
    /// graph. Off (`None`) unless `--one-tentacle`; with it off the territory graph and the side
    /// pass below run exactly as before.
    pub one_tentacle: RwLock<Option<Arc<crate::tentacle_unit::OneTentacle>>>,
    /// Refinement step 1 (2026-11-10): the last question's evidence set from G1 (question,
    /// set), on the tentacle core path under the queue rule; read by the reduce step and G2.
    pub stack: Mutex<Option<(String, Arc<Mutex<nmbrain::evidence::EvidenceSet>>)>>,
    /// The drones at intake on the plain path: every piece taken in once per material version and
    /// held in RAM, read against each request for its calibrated evidence, which judges the
    /// question's packets in the reduce step (`drone_judgments`). Opened at the first question,
    /// or before it by `warm_drones` (a served process starts it at start, guide WP8).
    drones: Arc<Mutex<Option<crate::drone_windows::Keep>>>,
}

/// The weight folder the drones at intake read (release tokenizer, central embedding table).
fn drone_weights(home: &Path) -> PathBuf {
    nmtype::models_dir(home).join("qwen3-0.6b/files")
}

fn bad(e: impl std::fmt::Display) -> Fault {
    Fault::new("nmapi.forage", FaultKind::Config, e.to_string())
}
/// The job and lens a tentacle carried in one run, read from its assignment (the output of its
/// `nmpod.unit#tN` node); `nmapi.tentacle_core#tN` carries its tentacle's. Both `None` for any
/// other part, or when the tentacle had no assignment.
fn tentacle_job(outputs: &BTreeMap<String, Payload>, part: &str) -> (Option<&'static str>, Option<String>) {
    let Some(n) = part.strip_prefix("nmpod.unit#t").or_else(|| part.strip_prefix("nmapi.tentacle_core#t")) else { return (None, None) };
    if n.is_empty() || !n.bytes().all(|b| b.is_ascii_digit()) {
        return (None, None);
    }
    let Some(a) = outputs.get(&format!("nmpod.unit#t{n}")).and_then(json) else { return (None, None) };
    let job = a.get("job").and_then(Value::string).ok().and_then(Job::from_name).map(Job::name);
    let lens = a.get("window").and_then(|w| w.get("lens")).and_then(Value::string).ok().map(str::to_owned);
    (job, lens)
}

fn json(v: &Payload) -> Option<&Value> {
    if let Payload::Json(j) = v { Some(j) } else { None }
}
fn read_json(path: &Path) -> Result<Value, Fault> {
    let bytes = std::fs::read(path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
    nmjson::parse(&bytes).map_err(|e| bad(format!("{}: {e}", path.display())))
}

/// The executed assembly named by `designs/anatomy.json` (`assemblies.forage`).
/// # Errors
/// Missing or malformed design files.
pub fn forage_assembly(home: &Path) -> Result<(PathBuf, Graph), Fault> {
    let anatomy = read_json(&home.join("designs/anatomy.json"))?;
    let rel = anatomy
        .get("assemblies")
        .and_then(|a| a.get("forage"))
        .and_then(Value::string)
        .map_err(|e| bad(format!("anatomy.json assemblies.forage: {e}")))?;
    let path = home.join(rel);
    let g = Graph::from_value(&read_json(&path)?)?;
    Ok((path, g))
}

/// The graph served while `--one-tentacle` is off (2026-11-10): `assemblies.forage-territory`
/// of `designs/anatomy.json` (the territory form, kept byte-identical as the control), or the
/// forage assembly when the anatomy names none.
/// # Errors
/// Missing or malformed design files.
pub fn territory_assembly(home: &Path) -> Result<(PathBuf, Graph), Fault> {
    let anatomy = read_json(&home.join("designs/anatomy.json"))?;
    match anatomy.get("assemblies").and_then(|a| a.get("forage-territory")).and_then(Value::string) {
        Ok(rel) => {
            let path = home.join(rel);
            let g = Graph::from_value(&read_json(&path)?)?;
            Ok((path, g))
        }
        Err(_) => forage_assembly(home),
    }
}

/// Run budget ceiling from `budget.max_run_s` in `settings.json`, else `units/policy.json`
/// (`max_run_seconds`), 60 s when absent.
#[must_use]
pub fn max_run_ms(home: &Path) -> u32 {
    if let Some(s) = crate::whole::control_int(home, "budget.max_run_s", 5, 180)
        && let Ok(ms) = u32::try_from(s.saturating_mul(1000))
    {
        return ms;
    }
    read_json(&home.join("units/policy.json"))
        .ok()
        .and_then(|v| v.get("max_run_seconds").and_then(Value::integer).ok())
        .and_then(|s| u32::try_from(s.saturating_mul(1000)).ok())
        .unwrap_or(nmnte::MAX_RUN_MS)
}

/// Readers of a subordinate drone: a deeper reading of its tentacle's territory.
pub const SUBORDINATE_READERS: [&str; 2] = ["deep", "symbols"];
/// Candidates the bound reranker judges for withholding when the dense lane runs (2026-11-10).
pub const DENSE_JUDGE: usize = 1;

/// GPU megabytes at which homeostasis suspends slots: `budget.gpu_mb_max` in `settings.json` when
/// pinned to a number; else (`auto`, the default) this machine's card total less the live calls'
/// measured room (Roadmap Phase 4.6, 2026-11-24: it was 3,500 MiB for every card, and a pinned
/// value above 4,096 was dropped); without a card, `designs/homeostasis.json`.
/// The same ceiling for the homeostasis thresholds (`anatomy::after_task`).
#[must_use]
pub fn gpu_ceiling_mb() -> f32 {
    gpu_mb_max()
}

fn gpu_mb_max() -> f32 {
    let home = nmtype::nano_root();
    if let Some(mb) = crate::whole::control_int(&home, "budget.gpu_mb_max", 1024, 81_920) {
        #[expect(clippy::cast_precision_loss, reason = "megabytes")]
        return mb as f32;
    }
    if let Some(mb) = nmengn::capacity::gpu_ceiling_mb() {
        #[expect(clippy::cast_precision_loss, reason = "megabytes")]
        return mb as f32;
    }
    static MAX: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *MAX.get_or_init(|| {
        let home = nmtype::nano_root();
        read_json(&home.join("designs/homeostasis.json"))
            .ok()
            .and_then(|v| {
                v.get("thresholds")
                    .ok()
                    .and_then(|t| t.get("gpu_mb_max").ok())
                    .or_else(|| v.get("gpu_mb_max").ok())
                    .and_then(|x| x.integer().ok())
            })
            .map_or(3500.0, |x| {
                #[expect(clippy::cast_precision_loss, reason = "megabytes")]
                let f = x as f32;
                f
            })
    })
}

/// The request enters through its own drone first: its kind (by the wording rules and the router
/// slot's jobs) and its aspects, recorded as the call's first audit step.
pub fn request_step(slots: &Slots, question: &str) -> nmpod::intake::Request {
    let (r, c) = crate::tools::clocked(|| nmpod::intake::Request::read(question, &slots.route(question)));
    // The request's drone takes the request only; its product never enters a material path.
    let mut cells = r.cells();
    if let Err(f) = nmtype::class::inlets::REQUEST_DRONE.admit(&r) {
        cells.push(format!("refused: {}", f.message));
    }
    crate::tools::note(crate::tools::Step {
        cells,
        counts: vec![("aspects", r.aspects.len() as u64)],
        ..crate::tools::step_at("nmpod.intake#request", c)
    });
    r
}

/// Levels of a finding in the GCX context map by its kind: focus and job readers (build, tests, uses, history)
/// are level 2, the base readers level 1.
fn level_of(kind: &str) -> u8 {
    if matches!(kind, "build" | "test" | "uses" | "deps" | "history") { 2 } else { 1 }
}

/// Coordination of one question (CM1, 2026-11-09): the GCX context map is built over the territories and fed the
/// first round; the Map Coordinator reads G's summary and the model side's and sets the
/// budgets; the second round refines the segments chosen by next-best-view (or the uniform
/// baseline, or none) with one subordinate drone each, a deeper reading of that segment only.
/// The tentacles' requests for subordinate drones count as signals for their territories. The
/// segments holding the places memory recalled (`pinned`, files) are always read in the second round,
/// whatever the refinement mode chose (master design 3.11 rule 14: recalled places are guaranteed a
/// reading).
#[expect(clippy::too_many_arguments, reason = "one round's inputs, passed once")]
fn coordinate_round(
    packets: Vec<Packet>,
    territories: &[Vec<nmtype::Region>],
    focus: &[String],
    plan: &Plan,
    budget: Budget,
    base: Budgets,
    mode: &str,
    sub: &Arc<Substrate>,
    slots: &Arc<Slots>,
    reflexes: &Arc<ReflexTable>,
    windows: Vec<nmbrain::gmap::WindowRecord>,
    pinned: &[String],
) -> (Vec<Packet>, GMap, Budgets) {
    let objective = plan.views.first().cloned().unwrap_or_default();
    let (requests, mut out): (Vec<Packet>, Vec<Packet>) =
        packets.into_iter().partition(|p| p.kind() == "question" && p.key().starts_with("subordinate:"));
    let asked: Vec<u8> = requests
        .iter()
        .filter_map(|r| r.key().rsplit('#').next().and_then(|t| t.strip_prefix('t')).and_then(|i| i.parse::<u8>().ok()))
        .collect();
    let files = sub.files.iter().map(|f| f.path.clone()).collect::<Vec<_>>();
    let mut map = GMap::new(&objective, focus, territories, &files);
    // Stage 5 WP-C4 W2: the project outline in G1's context map (the writer's, else the structural
    // outline of the shaped tree).
    map.outline = crate::writer_idle::outline_for(sub, 4_000);
    let (one, two): (Vec<Packet>, Vec<Packet>) = out.iter().cloned().partition(|p| level_of(p.kind()) == 1);
    map.record(1, &one);
    map.record(2, &two);
    map.record_windows(windows);
    let family = focus.first().cloned().unwrap_or_else(|| "context".into());
    let ctx = ContextSummary {
        task_family: family.clone(),
        coverage: map.coverage(),
        scope_files: map.scope_files(),
        salient: map.salient(),
        conflicts: map.conflicts.len(),
    };
    let res = slots.resident();
    let ram: f32 = res.iter().map(|(_, r, _)| *r).sum::<f32>().max(0.0) + 0.0;
    let gpu: f32 = res.iter().map(|(_, _, g)| *g).sum::<f32>().max(0.0) + 0.0;
    let rr = slots.by_name.get("reranker");
    let rec = rr.map(nmbrain::Slot::record).unwrap_or_default();
    #[expect(clippy::cast_precision_loss, reason = "shadow counts are small")]
    let agree = if rec.shadow == 0 { 1.0 } else { rec.shadow_agree as f32 / rec.shadow as f32 };
    let model = ModelSummary {
        task_family: family,
        reranker_engine: rr.is_some_and(|s| s.engine().is_some()),
        reranker_qualified: slots.qualified("reranker"),
        resident_mb: ram + gpu,
        ram_mb: ram,
        gpu_mb: gpu,
        gpu_mb_max: gpu_mb_max(),
        engines: slots.by_name.values().filter(|s| s.engine().is_some()).map(|s| s.name.clone()).collect(),
        shadow: rec.shadow,
        shadow_agree: agree,
    };
    let (budgets, fired) = crate::coordinator::coordinate(&ctx, &model, base);
    map.coordinator = crate::coordinator::to_value(&ctx, &model, budgets, &fired);
    let mut chosen = match mode {
        "off" => Vec::new(),
        "uniform" => map.uniform(budgets.refine),
        _ => map.next_best_view(budgets.refine, &asked),
    };
    for i in pinned.iter().filter_map(|f| map.segment_of(f)) {
        if !chosen.contains(&i) {
            chosen.push(i);
        }
    }
    map.refine_mode = mode.to_owned();
    for i in chosen {
        let Some(region) = map.region(i) else { continue };
        let id = format!("nmpod.unit#r{i}");
        let task = Task {
            id: plan.task * 100 + 1,
            objective: objective.clone(),
            budget: Budget { depth: 0, units: 1, ..budget },
            scope: region.path.clone(),
            parent: Some(plan.task),
            kind: TaskKind::Verify,
        };
        let a = Assignment::territory(Job::Reasoning, vec![region], task, Vec::new());
        let mut drone = Unit::new(&id, Level::Drone, Job::Reasoning, Arc::clone(reflexes), 1);
        drone.depth = 1;
        let (found, readers) =
            drone.sense_with(&a, &SUBORDINATE_READERS, sub.as_ref() as &dyn SubstrateView, slots.as_ref() as &dyn SlotView);
        let o = drone.process(&a, found, readers, sub.as_ref() as &dyn SubstrateView, slots.as_ref() as &dyn SlotView);
        let fresh: Vec<Packet> = o.packets.into_iter().filter(|p| p.kind() != "question").collect();
        map.record(3, &fresh);
        map.refined.push(i);
        out.extend(fresh);
    }
    (out, map, budgets)
}

/// A frame view for the brain's own steps (the NTE frame is made inside foraging).
struct BrainView(Budget);
impl FrameView for BrainView {
    fn version(&self) -> u64 {
        0
    }
    fn deadline(&self) -> u64 {
        u64::MAX
    }
    fn budget(&self) -> &Budget {
        &self.0
    }
    fn trace(&self, _: Hop) {}
}

impl Organism {
    /// Assemble from the mini's own folder (`home`: cards, designs, policy) and a repository
    /// to study, given at run time.
    /// # Errors
    /// Unreadable designs or cards, or an assembly the Integrity Validator refuses.
    pub fn open(home: &Path, root: &Path, skip: &[String]) -> Result<Self, Fault> {
        Self::open_with(home, root, skip, true, &home.join("memory"))
    }
    /// As `open`, with memory switched on or off. Off: no recall and nothing written to the
    /// memory folder (identity, structural summary, objectives, episodes, hierarchy, temp
    /// notes). The build tools still log their outcomes there (S9 learning data).
    /// # Errors
    /// As `open`.
    pub fn open_with(home: &Path, root: &Path, skip: &[String], memory_on: bool, memory_dir: &Path) -> Result<Self, Fault> {
        // The part cards (`crates/<part>/part.json`) and `designs/` are read before anything runs;
        // a Nano folder without them is named here instead of failing on the first missing file.
        for need in ["crates", "designs"] {
            if !home.join(need).is_dir() {
                return Err(bad(format!(
                    "nm-home-incomplete: the Nano folder {} has no `{need}` folder; the server reads its part cards and designs from there. Use a complete Nano folder (a source checkout has both) or set NANO_HOME to one",
                    home.display()
                )));
            }
        }
        let slots = Arc::new(Slots::default());
        let s2 = Arc::clone(&slots);
        let cache_file = memory_on.then(|| {
            let root_key = nmhash::sha256_bytes(nmsubs::repo::canon(root).to_string_lossy().as_bytes());
            memory_dir.join(format!("substrate-cache-{}.bin", &root_key[..12]))
        });
        let sub = Arc::new(Substrate::scan_cached(
            root,
            skip,
            Arc::new(move |t: &str| s2.embed(t)),
            cache_file.as_deref(),
        ));
        let reflexes = Arc::new(ReflexTable::from_value(&read_json(&home.join("designs/reflexes.json"))?)?);
        let (_, assembly) = territory_assembly(home)?;
        let tentacles = assembly.nodes.iter().filter(|n| n.kind == ComponentType::Tentacle).count();
        let nte = Arc::new(Nte::new(Some(Registry::load(&home.join("crates"))?)));
        nte.add_predicate("assigned", Arc::new(|p: &Payload| json(p).is_some_and(|j| !matches!(j, Value::Null))));
        let mut brain = CoreBrain::new(Arc::clone(&slots));
        brain.set_version(sub.version);
        // Memory areas (2026-11-27): the substrate's content counts in the `content` area.
        nmengn::limits::set_area_used("content", "substrate", sub.content_bytes());
        let org = Self {
            home: home.to_path_buf(),
            assembly,
            sub: RwLock::new(Arc::clone(&sub)),
            slots,
            reflexes,
            nte,
            brain: Mutex::new(brain),
            tentacles,
            workers: 0,
            max_run_ms: max_run_ms(home),
            changed_at: Mutex::new(BTreeMap::new()),
            own: Mutex::new(nmtype::ActionLedger::default()),
            last_sync_origin: Mutex::new((Vec::new(), Vec::new())),
            memory: Mutex::new(crate::session::memory_state(home, &sub.root, memory_dir, memory_on)),
            memory_on,
            objectives: Mutex::new(nmmem::structural::load_objectives(&nmmem::hierarchy::core_dir(memory_dir))),
            quarantined: Mutex::new(std::collections::BTreeSet::new()),
            homeostasis: Mutex::new(nmbrain::homeostasis::Homeostasis::new(nmbrain::homeostasis::Thresholds::from_value(
                &read_json(&home.join("designs/homeostasis.json")).unwrap_or(Value::Null),
            ))),
            parmap: Mutex::new(nmnpcs::ParMap::new(&memory_dir.join("parmap"))),
            memory_handle: Arc::new(Mutex::new(nmmem::Working::new())),
            last_frame: Arc::new(Mutex::new(BTreeMap::new())),
            refine: RwLock::new("nbv".to_owned()),
            // 2026-11-12: on by default; it acts only when the embedder and reranker are bound.
            recall_meaning: std::sync::atomic::AtomicBool::new(true),
            recall_r2: std::sync::atomic::AtomicBool::new(false),
            dense_feed: Mutex::new(None),
            tentacle_pass: RwLock::new(None),
            whole: RwLock::new(None),
            last_pass: Arc::new(Mutex::new(String::new())),
            brain_side: RwLock::new(None),
            one_tentacle: RwLock::new(None),
            stack: Mutex::new(None),
            drones: Arc::new(Mutex::new(None)),
        };
        let comps = org.bind(&org.assembly, &sub)?;
        org.nte.publish(org.assembly.clone(), comps)?;
        org.reload_operational();
        Ok(org)
    }
    /// Resident memory of every suspendable holder (the slot engines' parameter views).
    #[must_use]
    pub fn resident_mb(&self) -> Vec<(String, f32, f32)> {
        self.slots.resident()
    }
    /// Homeostasis Throttle: a slow slot's engine orders half as many candidates; recorded in
    /// the Parameter Map. A node that is not a slot is recorded only (tentacle budgets are
    /// thin until S7).
    pub fn throttle(&self, id: &str) {
        let applied = self.slots.throttle(id);
        if let Ok(mut p) = self.parmap.lock() {
            let _ = p.observe(
                "homeostasis",
                id,
                "throttle",
                if applied { "engine candidates halved" } else { "recorded; no slot engine at this node" },
                "homeostasis",
            );
        }
    }
    /// Homeostasis Suspend: memory pressure suspends a slot's engine and releases its device
    /// views (the exact control keeps answering); recorded in the Parameter Map.
    pub fn suspend(&self, id: &str) {
        let applied = self.slots.suspend(id);
        if let Ok(mut p) = self.parmap.lock() {
            let _ = p.observe(
                "homeostasis",
                id,
                "suspend",
                if applied { "engine suspended, device views released" } else { "recorded; no slot engine at this node" },
                "homeostasis",
            );
        }
    }
    /// Reload the verified build and Step Check records (`build-outcomes.jsonl` in the memory
    /// folder) into the substrate's operational view; read even with memory off (the build tools
    /// log there in both modes). Returns how many records were loaded.
    pub fn reload_operational(&self) -> usize {
        let dir = self.memory.lock().map(|m| m.dir.clone()).unwrap_or_default();
        self.sub().load_operational(&dir.join("build-outcomes.jsonl"))
    }
    /// The current substrate.
    #[must_use]
    pub fn sub(&self) -> Arc<Substrate> {
        self.sub.read().map_or_else(|p| Arc::clone(&p.into_inner()), |g| Arc::clone(&g))
    }

    /// Opens the drones at intake and takes the material in on a thread of its own, so the first
    /// question on the plain path finds every piece taken in (guide WP8). The thread holds the
    /// drones' lock while it works: a question that arrives first waits for that one intake
    /// instead of starting a second. Pieces that change later are taken in at the question, as
    /// before. The thread's result: pieces taken in, or why it did nothing, and its time.
    #[must_use]
    pub fn warm_drones(&self) -> std::thread::JoinHandle<String> {
        let (cell, sub, weights) = (Arc::clone(&self.drones), self.sub(), drone_weights(&self.home));
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let handle = std::thread::spawn(move || {
            let clock = std::time::Instant::now();
            let Ok(mut guard) = cell.lock() else {
                let _ = tx.send(());
                return "drones: lock poisoned".to_owned();
            };
            let _ = tx.send(());
            if guard.is_none() {
                match crate::drone_windows::Keep::open(&weights) {
                    Ok(k) => *guard = Some(k),
                    Err(e) => return format!("drones: not opened ({}); the question's packets keep the exact control", e.message),
                }
            }
            let opened = clock.elapsed().as_secs_f64();
            let taken = guard.as_mut().map(|k| k.sync(&sub));
            let all = clock.elapsed().as_secs_f64();
            match taken {
                Some(Ok(n)) => format!("drones: opened {opened:.2} s, {n} of {} pieces taken in {:.2} s", sub.chunks.len(), all - opened),
                Some(Err(e)) => format!("drones: intake refused ({}) after {all:.2} s", e.message),
                None => "drones: not opened".to_owned(),
            }
        });
        // The lock is taken before this returns, so no question can reach the drones first.
        let _ = rx.recv();
        handle
    }

    /// Bind a component to every connected node of an assembly, by kind and id: the plan
    /// node, the river mailbox, each tentacle (one unit per tentacle node, in file order),
    /// its drone tip and its pipeline (which shares its tentacle's unit).
    /// # Errors
    /// A connected node of a kind this organism has no component for.
    pub fn bind(&self, g: &Graph, sub: &Arc<Substrate>) -> Result<Components, Fault> {
        let mut comps: Components = BTreeMap::new();
        let mut tentacle_units: BTreeMap<String, (usize, Job, Arc<Mutex<Unit>>)> = BTreeMap::new();
        for (slot, n) in g.nodes.iter().filter(|n| n.kind == ComponentType::Tentacle).enumerate() {
            let job = nmpod::DEFAULT_JOBS.get(slot % 8).copied().unwrap_or(Job::Retrieval);
            tentacle_units.insert(
                n.id.clone(),
                (slot, job, Arc::new(Mutex::new(Unit::new(&n.id, Level::Tentacle, job, Arc::clone(&self.reflexes), 1)))),
            );
        }
        let connected = |id: &str| g.edges.iter().any(|e| e.from == id || e.to == id);
        // The one-tentacle graph (2026-11-10): its tentacle, drone tip, core, g0_tx and G1 nodes
        // are bound by `crate::tentacle_unit`; the territory graph never takes this branch.
        let unified = crate::tentacle_unit::is_unified(g);
        let one = self.one_tentacle.read().ok().and_then(|o| o.clone());
        for n in g.nodes.iter().filter(|n| connected(&n.id)) {
            let module = n.id.split('#').next().unwrap_or("");
            let parent = n.parent.clone().unwrap_or_default();
            let quarantined = self.quarantined.lock().is_ok_and(|q| q.contains(&n.id));
            let injected = n.config.get("inject").and_then(Value::string).ok() == Some("fault");
            if quarantined || injected {
                // Quarantine binds an inert stand-in of the right output kind; a fault drill
                // (`"inject": "fault"` in the node's config) binds a component that faults.
                let out = n.ports_out.first().map_or(nmtype::PortKind::Any, |p| p.kind);
                let id = n.id.clone();
                let exec: Arc<dyn Exec> = Arc::new(FnExec(move |_: &[Payload], _: &dyn FrameView| {
                    if injected && !quarantined {
                        return Err(Fault::new(&id, FaultKind::Execution, "fault drill: injected fault"));
                    }
                    Ok(match out {
                        nmtype::PortKind::Packets => Payload::Packets(Vec::new()),
                        nmtype::PortKind::Refs => Payload::Refs(Vec::new()),
                        _ => Payload::Json(Value::Null),
                    })
                }));
                comps.insert(n.id.clone(), exec);
                continue;
            }
            let exec: Arc<dyn Exec> = match (module, n.kind) {
                ("nmpod.unit" | "nmapi.tentacle_core" | "nmbrain.g0" | "nmbrain.g1", _) if unified => {
                    let ot = one.as_ref().ok_or_else(|| bad(format!("{}: the one-tentacle graph needs --one-tentacle", n.id)))?;
                    crate::tentacle_unit::exec(ot, self, n, sub, &tentacle_units)?
                }
                ("nmbrain.stack", _) => {
                    Arc::new(FnExec(|i: &[Payload], _: &dyn FrameView| i.first().cloned().ok_or_else(|| bad("plan node got no task"))))
                }
                ("nmbrain.rivers", _) => Arc::new(FnExec(|i: &[Payload], _: &dyn FrameView| {
                    Ok(Payload::Packets(
                        i.iter().filter_map(|p| if let Payload::Packets(v) = p { Some(v.clone()) } else { None }).flatten().collect(),
                    ))
                })),
                ("nmpod.unit", ComponentType::Tentacle) => {
                    let (slot, _, unit) = tentacle_units.get(&n.id).cloned().ok_or_else(|| bad(format!("{}: no tentacle unit", n.id)))?;
                    Arc::new(FnExec(move |inp: &[Payload], _: &dyn FrameView| {
                        let plan = inp.first().and_then(json).ok_or_else(|| bad("tentacle got no plan"))?;
                        let task = Task::from_value(plan.get("task").map_err(bad)?)?;
                        let territories: Vec<Vec<nmtype::Region>> = plan
                            .get("territories")
                            .and_then(Value::array)
                            .map_err(bad)?
                            .iter()
                            .map(|t| t.array().map_err(bad).and_then(|rs| rs.iter().map(nmtype::Region::from_value).collect()))
                            .collect::<Result<_, _>>()?;
                        let jobs: Vec<Job> = plan
                            .get("jobs")
                            .and_then(Value::array)
                            .map_err(bad)?
                            .iter()
                            .filter_map(|j| j.string().ok().and_then(Job::from_name))
                            .collect();
                        let focus: Vec<Job> = plan
                            .get("focus")
                            .and_then(Value::array)
                            .map(|a| a.iter().filter_map(|j| j.string().ok().and_then(Job::from_name)).collect())
                            .unwrap_or_default();
                        let u = unit.lock().map_err(|_| bad("tentacle lock"))?;
                        Ok(Payload::Json(u.assign(&task, &territories, slot, &jobs, &focus).map_or(Value::Null, |a| a.to_value())))
                    }))
                }
                ("nmpod.unit", ComponentType::Drone) => {
                    let job = tentacle_units.get(&parent).map_or(Job::Retrieval, |t| t.1);
                    let tip = Mutex::new(Unit::new(&n.id, Level::Drone, job, Arc::clone(&self.reflexes), 1));
                    let (sub, slots) = (Arc::clone(sub), Arc::clone(&self.slots));
                    Arc::new(FnExec(move |inp: &[Payload], _: &dyn FrameView| {
                        let a = Assignment::from_value(inp.first().and_then(json).ok_or_else(|| bad("tip got no assignment"))?)?;
                        let t = tip.lock().map_err(|_| bad("tip lock"))?;
                        let (found, readers) = t.sense(&a, sub.as_ref() as &dyn SubstrateView, slots.as_ref() as &dyn SlotView);
                        let mut m = BTreeMap::new();
                        m.insert("assignment".to_owned(), a.to_value());
                        m.insert("readers".to_owned(), Value::Number(readers.to_string()));
                        m.insert("findings".to_owned(), Value::Array(found.iter().map(Finding::to_value).collect()));
                        Ok(Payload::Json(Value::Object(m)))
                    }))
                }
                ("nmpod.pipeline", _) if n.id.ends_with("-emit") => Arc::new(FnExec(|i: &[Payload], _: &dyn FrameView| {
                    // Emit: the tentacle's claims go on to the Core Brain; requests go by recurse.
                    Ok(Payload::Packets(
                        i.iter()
                            .filter_map(|p| if let Payload::Packets(v) = p { Some(v.clone()) } else { None })
                            .flatten()
                            .filter(|p| p.kind() != "question")
                            .collect(),
                    ))
                })),
                ("nmpod.recursion", _) => Arc::new(FnExec(|i: &[Payload], _: &dyn FrameView| {
                    // Recurse: only subordinate requests and barrier questions travel on.
                    Ok(Payload::Packets(
                        i.iter()
                            .filter_map(|p| if let Payload::Packets(v) = p { Some(v.clone()) } else { None })
                            .flatten()
                            .filter(|p| p.kind() == "question")
                            .collect(),
                    ))
                })),
                ("nmmem.working", _) => {
                    let scope = parent.trim_start_matches("nmpod.unit#").to_owned();
                    let mem = Arc::clone(&self.memory_handle);
                    Arc::new(FnExec(move |i: &[Payload], _: &dyn FrameView| {
                        // Note: one working-memory line per frame for this tentacle's scope.
                        let n = i.iter().map(|p| if let Payload::Packets(v) = p { v.len() } else { 0 }).sum::<usize>();
                        if let Ok(mut m) = mem.lock() {
                            m.note(&scope, &format!("{n} packets at {}", nmtype::now_ms()));
                        }
                        Ok(Payload::Refs(Vec::new()))
                    }))
                }
                ("nmpod.pipeline", _) => {
                    let unit = tentacle_units
                        .get(&parent)
                        .map(|t| Arc::clone(&t.2))
                        .ok_or_else(|| bad(format!("{}: parent {parent} is not a tentacle", n.id)))?;
                    let (sub, slots) = (Arc::clone(sub), Arc::clone(&self.slots));
                    Arc::new(FnExec(move |inp: &[Payload], _: &dyn FrameView| {
                        let v = inp.first().and_then(json).ok_or_else(|| bad("pipeline got no findings"))?;
                        if matches!(v, Value::Null) {
                            // A quarantined tip hands on nothing; the pipeline passes nothing on.
                            return Ok(Payload::Packets(Vec::new()));
                        }
                        let a = Assignment::from_value(v.get("assignment").map_err(bad)?)?;
                        let readers = usize::try_from(v.get("readers").and_then(Value::integer).unwrap_or(0)).unwrap_or(0);
                        let found: Vec<Finding> = v
                            .get("findings")
                            .and_then(Value::array)
                            .map_err(bad)?
                            .iter()
                            .map(Finding::from_value)
                            .collect::<Result<_, _>>()?;
                        let mut u = unit.lock().map_err(|_| bad("tentacle lock"))?;
                        let out = u.process(&a, found, readers, sub.as_ref() as &dyn SubstrateView, slots.as_ref() as &dyn SlotView);
                        Ok(Payload::Packets(out.packets.into_iter().chain(out.questions).collect()))
                    }))
                }
                (m, k) => return Err(bad(format!("{}: no component for {m} of kind {}", n.id, k.name()))),
            };
            comps.insert(n.id.clone(), exec);
        }
        Ok(comps)
    }

    /// One question through the organism, with the one Core Brain.
    /// # Errors
    /// The brain's lock was poisoned by an earlier panic.
    pub fn ask(&self, question: &str, tokens: u32) -> Result<Answer, Fault> {
        self.ask_in(question, tokens, None)
    }

    /// `ask` in a mode (`whole::Mode`; `None`: the saved `mode` control). The mode applies on the
    /// whole path; the forage path has one way to answer.
    /// # Errors
    /// As `ask`.
    pub fn ask_in(&self, question: &str, tokens: u32, mode: Option<crate::whole::Mode>) -> Result<Answer, Fault> {
        // A caller that does not settle after its reply (a bench, a check) settles here, first.
        crate::session::settle(self);
        if let Some(w) = self.whole.read().ok().and_then(|x| x.clone()) {
            let mode = mode.unwrap_or_else(|| crate::whole::mode_control(&self.home));
            return self.ask_whole(&w, question, tokens, mode);
        }
        let sub = self.sub();
        request_step(&self.slots, question);
        let budget = Budget { tokens, ms: self.max_run_ms, depth: 1, units: 16 };
        let task = Task {
            id: nmtype::now_ms(),
            objective: question.to_owned(),
            budget,
            scope: "workspace".into(),
            parent: None,
            kind: TaskKind::Context,
        };
        let tentacles = self.nte.snapshot().nodes().iter().filter(|n| n.kind == ComponentType::Tentacle).count();
        let pod_cfg = Value::Object([("tentacles".to_owned(), Value::Number(tentacles.to_string()))].into_iter().collect());
        // The Neuropod cuts the tentacles' territories (timed for the audit row).
        let (territories, c) = crate::tools::clocked(|| {
            Neuropod::from_config(
                &pod_cfg,
                &self.reflexes,
                Arc::clone(&sub) as Arc<dyn SubstrateView>,
                Arc::clone(&self.slots) as Arc<dyn SlotView>,
            )
            .map(|p| p.territories())
        });
        let territories = territories?;
        crate::tools::note(crate::tools::Step {
            cells: vec!["territories".into()],
            counts: vec![("tentacles", territories.len() as u64)],
            ..crate::tools::step_at("nmpod.pod#p0", c)
        });
        let record: Arc<Mutex<FrameRecord>> = Arc::new(Mutex::new((Vec::new(), Vec::new(), Vec::new(), 0, 0, BTreeMap::new())));
        let (nte, rec, workers) = (Arc::clone(&self.nte), Arc::clone(&record), self.workers);
        let frame_out = Arc::clone(&self.last_frame);
        let mut brain = self.brain.lock().map_err(|_| bad("brain lock poisoned"))?;
        brain.set_version(sub.version);
        let (sub_f, slots_f, reflexes_f) = (Arc::clone(&sub), Arc::clone(&self.slots), Arc::clone(&self.reflexes));
        let mode = self.refine.read().map(|m| m.clone()).unwrap_or_else(|_| "nbv".into());
        let map_cell: Arc<Mutex<Option<(GMap, Budgets)>>> = Arc::new(Mutex::new(None));
        let map_f = Arc::clone(&map_cell);
        let base = Budgets { refine: crate::coordinator::REFINE_BASE, pack_tokens: tokens, rerank_cap: nmbrain::synthesis::NEURAL_CAP };
        let pass_f = self.tentacle_pass.read().ok().and_then(|p| p.clone());
        // The one tentacle system runs only on its own graph (published by `tentacle_unit::enable`).
        let one =
            self.one_tentacle.read().ok().and_then(|o| o.clone()).filter(|_| crate::tentacle_unit::is_unified(&self.nte.snapshot().graph));
        let one_f = one.clone();
        let (question_f, pass_note) = (question.to_owned(), Arc::clone(&self.last_pass));
        // Memory's recalled places, as files: their segments are read in the second round.
        let pinned: Vec<String> = if self.memory_on {
            let (files, c) = crate::tools::clocked(|| crate::session::recalled_files(self, question));
            crate::tools::note(crate::tools::Step {
                cells: vec!["read:episodes".into(), "pinned".into()],
                counts: vec![("files", files.len() as u64)],
                ..crate::tools::step_at("nmmem.gate#m0", c)
            });
            files
        } else {
            Vec::new()
        };
        // One pass per question, whatever the number of forage rounds.
        let pass_done: Mutex<Option<crate::tentacle_pass::Pass>> = Mutex::new(None);
        // Refinement step 1 (item 5): on the tentacle core path the memory lens reads the chunks
        // memory recalls for this question (read only; the recall that feeds the reduce step
        // runs later as before).
        if let Some(p) = pass_f.as_ref().filter(|p| p.settings.core) {
            let chunks = if self.memory_on {
                let (chunks, c) = crate::tools::clocked(|| crate::session::recalled_chunks(self, question));
                crate::tools::note(crate::tools::Step {
                    cells: vec!["read:episodes".into()],
                    counts: vec![("chunks", chunks.len() as u64)],
                    ..crate::tools::step_at("nmmem.gate#m0", c)
                });
                chunks
            } else {
                Vec::new()
            };
            p.set_recalled(question, chunks);
        }
        // Refinement step 1 (items 1 and 2): G1's evidence set leaves the pass by this cell.
        let stack_cell: Arc<Mutex<Option<Arc<Mutex<nmbrain::evidence::EvidenceSet>>>>> = Arc::new(Mutex::new(None));
        let stack_f = Arc::clone(&stack_cell);
        brain.set_forager(Box::new(move |plan: &Plan, _: &dyn FrameView| {
            let focus: Vec<String> =
                plan.jobs.iter().filter(|j| Job::from_name(j).is_some_and(|x| Job::FOCUS.contains(&x))).cloned().collect();
            let mut jobs: Vec<String> = plan.jobs.clone();
            for j in nmpod::DEFAULT_JOBS {
                if !jobs.iter().any(|x| x == j.name()) {
                    jobs.push(j.name().to_owned());
                }
            }
            let mut m = BTreeMap::new();
            m.insert(
                "task".to_owned(),
                Task {
                    id: plan.task,
                    objective: plan.views.first().cloned().unwrap_or_default(),
                    budget,
                    scope: "workspace".into(),
                    parent: None,
                    kind: TaskKind::Context,
                }
                .to_value(),
            );
            m.insert(
                "territories".to_owned(),
                Value::Array(territories.iter().map(|t| Value::Array(t.iter().map(nmtype::Region::to_value).collect())).collect()),
            );
            m.insert("jobs".to_owned(), Value::Array(jobs.into_iter().map(Value::String).collect()));
            m.insert("focus".to_owned(), Value::Array(focus.iter().cloned().map(Value::String).collect()));
            // One tentacle system: the question's state opens before the frame; with a pass bound the
            // shared sensory path (encode and drone reads) runs inside the frame, so its deadline
            // is lifted as the side pass had none (the G1 loop keeps its own derived ceiling).
            let frame = match one_f.as_ref() {
                Some(o) => {
                    o.begin(
                        plan.task,
                        &plan.views.first().cloned().unwrap_or_default(),
                        Arc::clone(&sub_f),
                        pass_f.clone(),
                        tentacles,
                        territories.clone(),
                    );
                    nte.frame(if pass_f.is_some() { Budget { ms: u32::MAX / 2, ..budget } } else { budget })
                }
                None => nte.frame(budget),
            };
            let r = nmnte::run(
                &frame,
                &Runtime::new(workers),
                BTreeMap::from([("nmbrain.stack#plan".to_owned(), vec![Payload::Json(Value::Object(m))])]),
            );
            let packets = match r.outputs.get("nmbrain.rivers#b0") {
                Some(Payload::Packets(v)) => v.clone(),
                _ => Vec::new(),
            };
            let mut packets = packets;
            let mut windows = Vec::new();
            if let Some(o) = one_f.as_ref() {
                windows = o.records(plan.task);
            } else if let (Some(pass), Ok(mut done)) = (pass_f.as_ref(), pass_done.lock()) {
                if done.is_none() {
                    match crate::tools::timed_step("nmapi.tentacle_pass", || pass.read(&sub_f, &slots_f, &question_f)) {
                        Ok(p) => *done = Some(p),
                        Err(e) => {
                            eprintln!("tentacle pass skipped: {}", e.message);
                            *done = Some(crate::tentacle_pass::Pass {
                                note: format!("tentacle pass skipped: {}", e.message),
                                ..Default::default()
                            });
                        }
                    }
                }
                if let Some(p) = done.as_ref() {
                    packets.extend(p.packets.iter().cloned());
                    windows.clone_from(&p.windows);
                    if let (Some(e), Ok(mut t)) = (p.evidence.as_ref(), stack_f.lock()) {
                        *t = Some(Arc::clone(e));
                    }
                    if let Ok(mut n) = pass_note.lock() {
                        n.clone_from(&p.note);
                    }
                }
            }
            let (packets, map, budgets) = crate::tools::timed_step("nmapi.coordinator", || {
                coordinate_round(packets, &territories, &focus, plan, budget, base, &mode, &sub_f, &slots_f, &reflexes_f, windows, &pinned)
            });
            if let Ok(mut c) = map_f.lock() {
                *c = Some((map, budgets));
            }
            if let Ok(mut f) = frame_out.lock() {
                f.clone_from(&r.outputs);
            }
            if let Ok(mut g) = rec.lock() {
                let mut latency: BTreeMap<String, f32> = BTreeMap::new();
                for h in frame.hops() {
                    #[expect(clippy::cast_precision_loss, reason = "per-node milliseconds are small")]
                    let ms = h.cost_ms as f32;
                    *latency.entry(h.component).or_default() += ms;
                }
                // The audit row's steps: each node's call as the frame's clock measured it (start
                // and length in microseconds); a tentacle's step also names the job its
                // assignment gave it and its lens.
                for nt in frame.node_times() {
                    let (job, lens) = tentacle_job(&r.outputs, &nt.component);
                    crate::tools::note(crate::tools::Step {
                        part: nt.component,
                        start_ms: Some(nt.start_us / 1000),
                        ms: Some(nt.us / 1000),
                        us: Some(nt.us),
                        job,
                        lens,
                        ..crate::tools::Step::default()
                    });
                }
                *g = (r.outputs.keys().cloned().collect(), r.skipped.clone(), r.faults.clone(), packets.len(), r.version, latency);
            }
            packets
        }));
        let engine_before = self.slots.engine_ms();
        // The Core Brain's plan: the router slot picks the jobs.
        let (plan, c) = crate::tools::clocked(|| brain.plan(&task));
        crate::tools::note(crate::tools::Step {
            cells: vec!["plan".into()],
            counts: vec![("jobs", plan.jobs.len() as u64)],
            ..crate::tools::step_at("nmbrain.slot#router", c)
        });
        let packets = brain.forage(&plan, &BrainView(budget));
        let (memory_packets, fresh, recalled) = if self.memory_on { self.recall_timed(question, None) } else { (Vec::new(), Vec::new(), 0) };
        let (mut map, budgets) = map_cell.lock().ok().and_then(|mut c| c.take()).unwrap_or_else(|| (GMap::default(), base));
        let reduce_budget = Budget { tokens: budgets.pack_tokens, ..budget };
        // With the dense lane running, the fused ranking already orders the head; the reranker
        // only judges whether its best candidates answer at all (R1 withholding), so it scores
        // the first `DENSE_JUDGE` (2026-11-10: 12 cost about 4 s per question).
        let lane_on = sub.dense.read().is_ok_and(|d| d.is_some());
        self.slots.set_rerank_cap(if lane_on { budgets.rerank_cap.min(DENSE_JUDGE) } else { budgets.rerank_cap });
        // Refinement step 1: with G1's evidence set (tentacle core, queue rule) the reduce step
        // orders tentacle chunks by their records and holds what it passes over in the queue.
        let stack = stack_cell.lock().ok().and_then(|mut t| t.take());
        if let Ok(mut t) = self.stack.lock() {
            *t = stack.as_ref().map(|e| (question.to_owned(), Arc::clone(e)));
        }
        let mut all: Vec<Packet> = packets.into_iter().chain(fresh).chain(memory_packets).collect();
        // The drones' findings join the readers': every piece they accept for an aspect of the
        // request, and they judge every packet (`drone_judgments`).
        let judged = self.drone_judgments(&sub, question, &mut all, usize::try_from(reduce_budget.tokens).unwrap_or(usize::MAX));
        let packets_in = all.len() as u64;
        // The Core Brain's reduce step (river mailbox, arbiter, Tension Buffer, synthesis) and its
        // withholding check, timed together for the audit row.
        let (pack, c) = crate::tools::clocked(|| {
            // The reducer's coverage is measured against the material: question terms it holds in no
            // form are left out (no packet can hold them), and a term held only by its stem counts
            // as far as the exact form would not be expected (`Substrate::expected_presence`).
            let presence = sub.expected_presence(general_use(&self.home));
            self.slots.set_material(nmbrain::slot::Material { unmatched: sub.unmatched_terms(question), form_weight: 1.0 - presence });
            self.slots.set_judged(judged);
            let pack = match stack.as_ref().map(|e| e.lock()) {
                Some(Ok(mut set)) => brain.reduce_evidence(all, &reduce_budget, &mut set),
                _ => brain.reduce(all, &reduce_budget),
            };
            self.slots.set_material(nmbrain::slot::Material::default());
            self.slots.set_judged(BTreeMap::new());
            self.slots.set_rerank_cap(usize::MAX);
            let (absent, share) = sub.absent_terms_with(question, general_use(&self.home));
            let mut pack = nmbrain::synthesis::withhold_uncovered(pack, &absent, share);
            if pack.withheld {
                pack.tokens = 0;
                pack.floor_met = false;
            }
            pack
        });
        crate::tools::note(crate::tools::Step {
            cells: ["nmbrain.rivers#b0", "nmbrain.arbiter#b0", "nmbrain.contradiction#b0", "nmbrain.synthesis#b0"].map(str::to_owned).to_vec(),
            counts: vec![("packets_in", packets_in), ("packets_out", pack.packets.len() as u64)],
            ..crate::tools::step_at("nmbrain.synthesis", c)
        });
        map.mark_kept(&pack);
        let mut context_map = map.to_value();
        if let (Some(rec), Value::Object(m)) = (one.as_ref().and_then(|o| o.finish(task.id)), &mut context_map) {
            m.insert("one_tentacle".to_owned(), rec);
        }
        let (emitted, c) = crate::tools::clocked(|| brain.emit(&pack));
        crate::tools::note(crate::tools::Step { cells: vec!["nmbrain.slot#writer".into()], ..crate::tools::step_at("nmbrain.stack#emit", c) });
        let questions = brain.questions.clone();
        drop(brain);
        let side = self.brain_side.read().ok().and_then(|s| s.clone());
        let (emitted, brain_record) = match side.as_ref() {
            Some(s) => crate::tools::timed_step("nmapi.brain_side", || s.after_reduce(self, question, &pack, emitted)),
            None => (emitted, None),
        };
        let (ran, skipped, faults, raw, ran_version, mut latency) = record.lock().map(|g| g.clone()).unwrap_or_default();
        for (id, ms) in self.slots.engine_ms() {
            let spent = ms - engine_before.get(&id).copied().unwrap_or(0.0);
            if spent > 0.0 {
                #[expect(clippy::cast_possible_truncation, reason = "milliseconds per task fit f32")]
                latency.insert(id, spent as f32);
            }
        }
        crate::tools::timed_step("nmapi.anatomy", || crate::anatomy::after_task(self, &ran, &faults, &latency));
        let (episode, written) = self.remember_timed(question, &pack, &emitted, sub.version, None);
        if let (Some(s), Some(r)) = (side.as_ref(), brain_record.as_ref()) {
            crate::tools::timed_step("nmapi.brain_side", || s.after_remember(self, r, episode.as_deref()));
        }
        Ok(Answer {
            pack,
            emitted,
            questions,
            version: self.nte.snapshot().version,
            ran,
            skipped,
            faults,
            raw_packets: raw,
            episode,
            written,
            recalled,
            ran_version,
            context_map,
            brain: brain_record,
        })
    }

    /// The drones' judgment of the question's packets: each packet's relevance from the
    /// calibrated evidence of the piece it cites (`nmpod::evidence`, through the drones' intake): a
    /// piece accepted for an aspect of the request stands at or above the reducer's relevance
    /// floor, in proportion to its evidence against the strongest piece's; one accepted for none
    /// stands below it. A packet citing no piece of the material (a recall, a history line, a
    /// unit) is left to the exact control, as is every packet when the drones cannot open (the
    /// release tokenizer or embedding table missing): the stated fallback.
    fn drone_judgments(&self, sub: &Substrate, question: &str, packets: &mut Vec<Packet>, room: usize) -> BTreeMap<u64, f32> {
        let mut out = BTreeMap::new();
        let Ok(mut guard) = self.drones.lock() else { return out };
        if guard.is_none() {
            match crate::drone_windows::Keep::open(&drone_weights(&self.home)) {
                Ok(k) => *guard = Some(k),
                Err(_) => return out,
            }
        }
        let Some(keep) = guard.as_mut() else { return out };
        let mode = crate::drone_windows::items_mode();
        if mode != crate::drone_windows::ItemsMode::Off {
            return item_judgments(keep, sub, question, packets, room, mode);
        }
        let ((ev, accepted), c) = crate::tools::clocked(|| {
            if keep.sync(sub).is_err() {
                return (None, 0);
            }
            let meaning = sub.dense.read().ok().and_then(|d| d.clone()).and_then(|l| l.cosines(sub, question));
            let ev = keep.evidence_with(question, meaning.as_deref()).ok();
            let accepted = ev.as_ref().map_or(0, |e| (0..e.len()).filter(|i| (0..e.aspects.len()).any(|a| e.accepted(a, *i))).count());
            (ev, accepted)
        });
        crate::tools::note(crate::tools::Step {
            cells: vec!["evidence".into()],
            counts: vec![("pieces", sub.chunks.len() as u64), ("accepted", accepted as u64)],
            ..crate::tools::step_at("nmpod.evidence", c)
        });
        let Some(ev) = ev else { return out };
        let e = ev.pieces();
        let top = e.iter().copied().fold(0.0_f32, f32::max);
        if top <= 0.0 {
            return out;
        }
        let mut by_file: BTreeMap<String, Vec<(usize, usize, usize)>> = BTreeMap::new();
        for (i, ch) in sub.chunks.iter().enumerate() {
            if let Some(f) = sub.files.get(ch.file) {
                by_file.entry(f.path.replace('\\', "/")).or_default().push((ch.start, ch.end, i));
            }
        }
        let floor = nmbrain::synthesis::FLOOR;
        // Every accepted piece no packet cites yet goes in as the drones' own finding.
        let cited: std::collections::BTreeSet<String> = packets.iter().map(|p| p.provenance().source.replace('\\', "/")).collect();
        for (i, ch) in sub.chunks.iter().enumerate() {
            if !(0..ev.aspects.len()).any(|a| ev.accepted(a, i)) {
                continue;
            }
            let Some(f) = sub.files.get(ch.file) else { continue };
            let src = format!("{}:{}", f.path.replace('\\', "/"), ch.start);
            if cited.contains(&src) {
                continue;
            }
            let from = nmtype::Provenance::new(&src, "nmpod.evidence", sub.version, nmtype::now_ms());
            packets.push(
                Packet::new("chunk", &format!("chunk:{src}"), &ch.text, from)
                    .status(nmtype::Status::Inferred)
                    .confidence(e.get(i).copied().unwrap_or(0.0) / top)
                    .evidence(nmsubs::views::piece_evidence(ch, format!("{src} lines {}..{}", ch.start, ch.end))),
            );
        }
        let mut held: Vec<(u64, (f32, bool, usize))> = Vec::new();
        // Memory's family of evidence (`session::memory_family`; empty with memory off).
        let family = crate::session::memory_family(self, question);
        let mut recalled_at: BTreeMap<u64, f32> = BTreeMap::new();
        // A refuted outcome's memory note (`session::recall`; memory on only) is shown, never cut:
        // it serves no line and warns against the places it names.
        for p in packets.iter().filter(|p| crate::session::is_refuted_note(p)) {
            out.insert(p.id(), 1.0);
        }
        for p in packets.iter() {
            let Some((path, line)) = p.provenance().source.rsplit_once(':') else { continue };
            let Ok(at) = line.parse::<usize>() else { continue };
            let (a, b) = p
                .get_evidence()
                .iter()
                .find_map(|x| {
                    let (a, b) = x.split_once(" lines ")?.1.split_whitespace().next()?.split_once("..")?;
                    Some((a.parse().ok()?, b.parse().ok()?))
                })
                .unwrap_or((at, at));
            if let Some(m) = crate::session::family_strength(&family, path, a, b) {
                recalled_at.insert(p.id(), m);
            }
            let Some(pieces) = by_file.get(&path.replace('\\', "/")) else { continue };
            let best = pieces
                .iter()
                .filter(|(s, t, _)| *s <= b && *t >= a)
                .map(|(_, _, i)| (e.get(*i).copied().unwrap_or(0.0), (0..ev.aspects.len()).any(|x| ev.accepted(x, *i)), *i))
                .fold(None, |m: Option<(f32, bool, usize)>, x| match m {
                    Some(y) if (y.1, y.0) >= (x.1, x.0) => Some(y),
                    _ => Some(x),
                });
            if let Some(x) = best {
                held.push((p.id(), x));
            }
        }
        // With the joint order on, a packet's share is its piece's value in the joint order over
        // the pieces the packets cite (`Evidence::joint_order`), from the last's (0) to the
        // first's (1); else its piece's evidence as a share of the strongest piece's.
        let joint: Option<BTreeMap<usize, f32>> = nmpod::evidence::joint_on().then(|| {
            let mut cands: Vec<usize> = held.iter().map(|h| h.1.2).collect();
            cands.sort_unstable();
            cands.dedup();
            let file_of = |c: usize| sub.chunks.get(c).map_or(usize::MAX, |x| x.file);
            // The fusion constant follows how many pieces are plausible for this request
            // (`Evidence::plausible`).
            #[expect(clippy::cast_precision_loss, reason = "a count of pieces")]
            let k = ev.plausible(&cands) as f32;
            let order = ev.joint_order(&cands, &file_of, k);
            let first = order.first().map_or(1.0, |x| x.1);
            let last = order.last().map_or(0.0, |x| x.1);
            let span = (first - last).max(f32::MIN_POSITIVE);
            order.into_iter().map(|(c, v)| (c, (v - last) / span)).collect()
        });
        for (id, (v, ok, i)) in held {
            let share = match &joint {
                Some(j) => j.get(&i).copied().unwrap_or(0.0),
                None => v / top,
            }
            .clamp(0.0, 1.0);
            let (ok, share) = crate::session::with_family(ok, v > 0.0, share, recalled_at.get(&id).copied());
            out.insert(id, if ok { floor + (1.0 - floor) * share } else { floor * share });
        }
        out
    }

    /// `session::recall`, timed for the audit row with the memory cells it read (memory on only).
    fn recall_timed(&self, question: &str, inside: Option<&'static str>) -> (Vec<Packet>, Vec<Packet>, usize) {
        let (r, c) = crate::tools::clocked(|| crate::session::recall(self, question));
        let ids: Vec<String> = r.0.iter().filter_map(|p| p.key().strip_prefix("memory:").map(str::to_owned)).collect();
        crate::tools::note(crate::tools::Step {
            cells: crate::session::memory_cells(self, "read", &ids),
            counts: vec![("episodes", r.2 as u64), ("reads", r.1.len() as u64)],
            inside,
            ..crate::tools::step_at("nmmem.gate#m0", c)
        });
        r
    }

    /// `session::remember`, timed for the audit row with the memory cells it wrote (none when
    /// memory is off: nothing is written and no step is noted).
    fn remember_timed(&self, question: &str, pack: &Pack, emitted: &Packet, version: u64, inside: Option<&'static str>) -> (Option<String>, String) {
        // Task end: the tentacles' working notes of this task are dropped (working memory lasts a
        // task; none of them is tagged).
        if let Ok(mut w) = self.memory_handle.lock() {
            w.end_all();
        }
        if !self.memory_on {
            return crate::session::remember(self, question, pack, emitted, version);
        }
        let ((episode, written), c) = crate::tools::clocked(|| crate::session::remember(self, question, pack, emitted, version));
        let cells = match episode.as_ref() {
            Some(id) => crate::session::memory_cells(self, "write", std::slice::from_ref(id)),
            None => vec!["write:none".to_owned()],
        };
        crate::tools::note(crate::tools::Step { cells, inside, ..crate::tools::step_at("nmmem.gate#m0", c) });
        (episode, written)
    }

    /// One question through the whole path (`serve --whole`, 2026-11-12; `crate::whole`): the
    /// Globalized Context Stack, then the Core Brain (reranker order, second look), then memory's
    /// recalled packets after the Core Brain's, one pack within the budget; emitted and remembered
    /// as on the forage path.
    /// # Errors
    /// Engine or weight faults of the whole path.
    pub fn ask_whole(&self, w: &crate::whole::WholePath, question: &str, tokens: u32, mode: crate::whole::Mode) -> Result<Answer, Fault> {
        crate::session::settle(self);
        let sub = self.sub();
        let task = nmtype::now_ms();
        request_step(&self.slots, question);
        let recalled_chunks = if self.memory_on {
            let (chunks, c) = crate::tools::clocked(|| crate::session::recalled_chunks(self, question));
            crate::tools::note(crate::tools::Step {
                cells: vec!["read:episodes".into()],
                counts: vec![("chunks", chunks.len() as u64)],
                ..crate::tools::step_at("nmmem.gate#m0", c)
            });
            chunks
        } else {
            Vec::new()
        };
        let (prep, prep_clock) = crate::tools::clocked(|| {
        // L1 decision tuning (continuous learning step 5): with memory on, the second look's
        // threshold is picked by the tuner from the ledger's verdicts (exploring a little, with the
        // exact propensity logged); with memory off, the stated default.
        let ledger = if self.memory_on { self.memory.lock().ok().map(|m| crate::whole::ledger_path(&self.home, &m.dir)) } else { None };
        // MO8: the strongest live whole-path pack acts as priors at decision sites this question.
        let mut pack_ctl = crate::brain_side::side(self).and_then(|s| s.whole_path_priors()).map(nmmusc::priors::Control::new);
        let (threshold, propensity) = match ledger.as_ref() {
            Some(p) => {
                let tuner = nmlearn::tuner::Tuner::from_ledger(&nmlearn::ledger::load(p).unwrap_or_default());
                let seed = task ^ question.bytes().fold(0_u64, |h, b| h.wrapping_mul(1_099_511_628_211) ^ u64::from(b));
                let u = nmlearn::tuner::draw(seed);
                let (_, t, pr) = tuner.pick(u);
                // MO8: when the tuner exploits (does not explore), a whole-path pack may set the
                // arm as a prior: it decides only when its evidence exceeds the tuner's margin.
                match (u >= nmlearn::tuner::EPS, pack_ctl.as_mut()) {
                    (true, Some(ctl)) => {
                        let label = |a: f32| if a.is_finite() { format!("{a}") } else { "never".to_owned() };
                        let labels: Vec<String> = nmlearn::tuner::ARMS.iter().map(|a| label(*a)).collect();
                        let allowed: Vec<&str> = labels.iter().map(String::as_str).collect();
                        let rule = label(t);
                        let best = tuner.best();
                        let margin = |alt: &str| -> f64 {
                            let Some(a) = labels.iter().position(|x| x == alt) else { return f64::MAX };
                            match (tuner.estimate(best), tuner.estimate(a), tuner.value.get(best), tuner.value.get(a)) {
                                (Some(pb), Some(pa), Some(nb), Some(na)) if nb.1 > 0 && na.1 > 0 => {
                                    #[expect(clippy::cast_precision_loss, reason = "counts")]
                                    let se = (pb * (1.0 - pb) / nb.1 as f64 + pa * (1.0 - pa) / na.1 as f64).sqrt().max(1e-6);
                                    (pb - pa) / se
                                }
                                _ => 0.0,
                            }
                        };
                        let veto = |_: &nmmusc::priors::Prior| w.desk.lock().is_ok_and(|d| d.veto_packs);
                        let chosen = ctl.decide(nmmusc::priors::Site::LookArm, &rule, &allowed, &margin, &veto, &|_| false);
                        let arm = labels.iter().position(|x| *x == chosen).and_then(|i| nmlearn::tuner::ARMS.get(i).copied()).unwrap_or(t);
                        (arm, if (arm - t).abs() < f32::EPSILON || (arm.is_infinite() && t.is_infinite()) { pr } else { 1.0 })
                    }
                    _ => (t, pr),
                }
            }
            None => (crate::decide::LOOK_MARGIN, 1.0),
        };
        // Roadmap Phase 2 (2026-11-24): G3 can force the look arm for a replay, and every call
        // records the arm it used, so LookArm is replayed with the other sites.
        let (threshold, propensity) = crate::g3::forced_look_arm().map_or((threshold, propensity), |a| (a, 1.0));
        crate::g3::note_look_arm(threshold);
        (ledger, pack_ctl, threshold, propensity)
        });
        let (ledger, mut pack_ctl, threshold, propensity) = prep;
        // The decision tuner's pick of the second look's threshold (ledger read) and the whole-path
        // pack's priors.
        crate::tools::note(crate::tools::Step { cells: vec!["tuner".into()], ..crate::tools::step_at("nmlearn.tuner", prep_clock) });
        let cocited = if self.memory_on {
            let (links, c) = crate::tools::clocked(|| crate::session::cocitations(self));
            crate::tools::note(crate::tools::Step {
                cells: vec!["read:episodes".into(), "cocitations".into()],
                counts: vec![("chunks", links.len() as u64)],
                ..crate::tools::step_at("nmmem.gate#m0", c)
            });
            links
        } else {
            BTreeMap::new()
        };
        // The judge for asking more (`nmapi::ask`): the saved L4 calibrator with thresholds from
        // the ledger's verdicts; none with memory off or before one is saved.
        let judge = if self.memory_on && w.ask == crate::ask::Mode::Judge {
            crate::tools::timed_step("nmapi.ask", || {
                self.memory.lock().ok().map(|m| m.dir.clone()).and_then(|dir| {
                    crate::ask::judge(&self.home, &crate::learn::learned_dir(&self.home, &dir), &crate::whole::ledger_path(&self.home, &dir))
                })
            })
        } else {
            None
        };
        // G2's instructions for this question: the organism's own last edits (action ledger) are
        // held back from going up (I10; analysis C: "self-caused edit"), still held by the tentacles.
        let own: Vec<String> = self.last_sync_origin.lock().map(|o| o.0.clone()).unwrap_or_default();
        if let Ok(mut o) = w.own_files.lock() {
            o.clone_from(&own);
        }
        let mut orders: Vec<nmbrain::instruct::Instr> = own
            .iter()
            .map(|f| nmbrain::instruct::Instr::Inhibit {
                target: nmbrain::instruct::Target::File(f.replace('\\', "/")),
                reason: "own edit (action ledger)".into(),
                ttl_ms: 0,
            })
            .collect();
        // MO8: G2's up-count as a site: a pack naming `many` sends up as many more of G1's queue as the
        // call's token budget holds at the material's mean piece size (I11; the budget is the caller's,
        // `drone_path::PACK_TOKENS` when it names none); the rule's choice is `some` (the capacity gate),
        // with no evidence of its own.
        if let Some(ctl) = pack_ctl.as_mut() {
            let veto = |_: &nmmusc::priors::Prior| w.desk.lock().is_ok_and(|d| d.veto_packs);
            if ctl.decide(nmmusc::priors::Site::UpCount, "some", &["some", "many"], &|_| 0.0, &veto, &|_| false) == "many" {
                let mean = sub.chunks.iter().map(|c| c.tokens).sum::<usize>() / sub.chunks.len().max(1);
                let budget = usize::try_from(tokens).unwrap_or(crate::drone_path::PACK_TOKENS);
                orders.push(nmbrain::instruct::Instr::Next { k: (budget / mean.max(1)).max(1) });
            }
        }
        if w.up_next > 0 && !orders.iter().any(|o| matches!(o, nmbrain::instruct::Instr::Next { .. })) {
            orders.push(nmbrain::instruct::Instr::Next { k: w.up_next });
        }
        self.slots.note_best(-1.0);
        let call_t0 = std::time::Instant::now();
        let learned_rw = if self.memory_on && crate::whole::learning_control(&self.home) {
            self.memory.lock().ok().map(|m| {
                let eng = nmlearn::engine::ContinuousLearningEngine::open(&m.dir);
                crate::drone_path::RetrievalWeights::from_adaptation(&eng.retrieval)
            }).unwrap_or_default()
        } else {
            crate::drone_path::RetrievalWeights::default()
        };
        // Stage 5 WP-C1: G2's room follows this call's token budget (Luis approved this hook, 2026-11-22).
        w.room_tokens.store(usize::try_from(tokens).unwrap_or(crate::drone_path::PACK_TOKENS), std::sync::atomic::Ordering::Relaxed);
        // Stage 5 WP-C2: the call's refinement allocation (G3's allocation site writes it for the call).
        if let (Ok(mut r), Ok(m)) = (w.refine.lock(), self.refine.read()) {
            r.clone_from(&m);
        }
        let (run, ordered) = crate::drone_path::with_retrieval_weights(learned_rw, || {
            w.run(&sub, question, &recalled_chunks, &cocited, threshold, judge.as_ref(), &orders, mode)
        })?;
        // Memory areas (2026-11-27): the call read the material as taken in so far; when pieces
        // still wait and no pass runs, the background pass takes them in between calls.
        if w.keep_waiting()
            && let Some(arc) = self.whole.read().ok().and_then(|x| x.clone())
        {
            let sub = Arc::clone(&sub);
            std::thread::spawn(move || nmtype::budget::background(|| arc.warm_keep(&sub)));
        }
        if run.rerank_ok {
            let best = run.rerank.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if best.is_finite() {
                self.slots.note_best(best);
            }
        }
        let assembly_at = std::time::Instant::now();
        let core = crate::whole::packets(&sub, &ordered, &run.recalled);
        let raw = core.len();
        let (memory_packets, fresh, recalled) =
            if self.memory_on { self.recall_timed(question, Some("nmapi.whole#assembly")) } else { (Vec::new(), Vec::new(), 0) };
        // Useful uses (memory hierarchy, 2026-11-12): an episode whose cited line is among the
        // evidence of the Core Brain's packets in the pack counts one useful use.
        let episodes: Vec<(String, Vec<String>)> = memory_packets
            .iter()
            .filter_map(|p| p.key().strip_prefix("memory:").map(|id| (id.to_owned(), p.get_evidence().to_vec())))
            .collect();
        let default_pack = crate::whole::pack_tokens_control(&self.home);
        let focused_high_conf = (run.rerank_ok && run.top_gap.is_finite() && run.top_gap >= 1.25)
            || (mode == crate::whole::Mode::Flow
                && run.flow_status.as_ref().is_some_and(|(st, miss)| st == "answered" && miss.is_empty())
                && run.g1_rounds <= 1
                && run.order.len() <= 6);
        let effective_tokens = if tokens == default_pack && crate::whole::adaptive_pack_control(&self.home) && focused_high_conf {
            crate::whole::focused_tokens_control(&self.home).min(tokens)
        } else {
            tokens
        };
        // Stage 5 WP-C4 W2: for a broad question (an explain or overview kind) the project outline
        // (the writer's, else the shaped tree's structural outline) goes in after the answer pieces
        // (never before them), within a tenth of the pack's tokens. On by default (roadmap Phase 0
        // item 2); `NANO_OUTLINE_PACK=off` only for diagnosis.
        let cap = usize::try_from(effective_tokens / 10).unwrap_or(400).max(1) * 4;
        // Architectural packs (roadmap Phase 1 item 1): explain questions and edit requests (the
        // Core Brain's propose-edit task, `edit_request`).
        let outline_packet: Option<Packet> = (!std::env::var("NANO_OUTLINE_PACK").is_ok_and(|v| v == "off") && (crate::whole::kind_of(question) == "explain" || edit_request(question)))
            .then(|| crate::writer_idle::outline_for(&sub, cap))
            .flatten()
            .map(|o| {
                let text: String = o.chars().take(cap).collect();
                Packet::new("note", "outline:project", &text, nmtype::Provenance::new("writer:outline", "nmapi.writer_idle", sub.version, nmtype::now_ms())).status(nmtype::Status::Inferred)
            });
        let pack = crate::whole::pack(task, core.into_iter().chain(fresh).chain(memory_packets).chain(outline_packet).collect(), effective_tokens);
        // The decoy rule: marked low when no candidate's reranker score reaches the cut.
        // A failed reranker gives no scores to judge by: the decoy cut is not applied then.
        let pack = if run.rerank_ok { crate::whole::withhold_weak(pack, &run.rerank, w.withhold_below) } else { pack };
        let pack = crate::whole::flow_status(pack, &run);
        let mut pack = if pack.withheld {
            pack
        } else {
            let (absent, share) = sub.absent_terms_with(question, general_use(&self.home));
            nmbrain::synthesis::withhold_uncovered(pack, &absent, share)
        };
        if pack.withheld {
            pack.packets.clear();
            pack.tokens = 0;
            pack.floor_met = false;
        }
        let used: std::collections::BTreeSet<&String> =
            pack.packets.iter().filter(|p| p.kind() != "memory").flat_map(|p| p.get_evidence().iter()).collect();
        let useful: Vec<String> = episodes.iter().filter(|(_, ev)| ev.iter().any(|e| used.contains(e))).map(|(id, _)| id.clone()).collect();
        if !useful.is_empty()
            && let Ok(mut m) = self.memory.lock()
        {
            let _ = m.hierarchy.mark_useful(&useful);
        }
        // Revision 2: each recalled memory counts one use, useful when its cited part reached the pack.
        // A withheld pack carries nothing, so it says nothing about the recalled memories'
        // usefulness (2026-11-13 review: it had counted every one as not useful).
        if !episodes.is_empty()
            && !pack.withheld
            && let Ok(mut m) = self.memory.lock()
            && let Some(m2) = m.mem2.as_mut()
        {
            let uses: Vec<(String, bool)> = episodes.iter().map(|(id, _)| (id.clone(), useful.contains(id))).collect();
            let _ = m2.used(&uses);
        }
        // Record compact parameter/retrieval outcome telemetry for long-term parameter-system learning.
        if self.memory_on
            && !crate::g3::replaying()
            && crate::whole::learning_control(&self.home)
            && let Ok(m) = self.memory.lock()
        {
            let eng = nmlearn::engine::ContinuousLearningEngine::open(&m.dir);
            let ms_used = u64::try_from(call_t0.elapsed().as_millis()).unwrap_or(u64::MAX);
            let active_carts = vec![format!("filter_gen:{}", nmengn::filter::generation()), format!("mode:{}", mode.name())];
            eng.append_parameter_telemetry(
                &m.dir,
                question,
                "project",
                (!pack.withheld && pack.floor_met, u64::from(pack.tokens), ms_used),
                &active_carts,
            );
        }
        let mut brain = self.brain.lock().map_err(|_| bad("brain lock poisoned"))?;
        brain.set_version(sub.version);
        // The Core Brain's objective is the question (set by its plan, as on the forage path).
        let budget = Budget { tokens, ms: self.max_run_ms, depth: 1, units: 16 };
        let _ = brain.plan(&Task {
            id: task,
            objective: question.to_owned(),
            budget,
            scope: "workspace".into(),
            parent: None,
            kind: TaskKind::Context,
        });
        let (emitted, c) = crate::tools::clocked(|| brain.emit(&pack));
        crate::tools::note(crate::tools::Step {
            cells: vec!["nmbrain.slot#writer".into()],
            inside: Some("nmapi.whole#assembly"),
            ..crate::tools::step_at("nmbrain.stack#emit", c)
        });
        let questions = brain.questions.clone();
        drop(brain);
        let (episode, written) = self.remember_timed(question, &pack, &emitted, sub.version, Some("nmapi.whole#assembly"));
        // The decision ledger (continuous learning step 2): the Core Brain's choice with what it
        // chose from; written only when memory is on (nothing is written with memory off).
        let mut ledger_note = Value::Null;
        // A question on which a part failed is not a decision record (its scores are not the reranker's).
        if let Some(path) = ledger.as_ref().filter(|_| run.faults.is_empty()) {
            let d = nmlearn::ledger::Decision {
                at: nmtype::now_ms(),
                episode: episode.clone(),
                question_sha: nmhash::sha256_bytes(question.as_bytes()),
                version: sub.version,
                candidates: run.candidates.clone(),
                rerank: run.rerank.clone(),
                look: run.look.clone(),
                look_scores: run.look_scores.clone(),
                threshold,
                chosen: run.order.first().copied().unwrap_or(0),
                propensity,
                // Standalone L1 signals: fresh re-read verification of each look chunk, and cost.
                look_verified: crate::whole::verify_chunks(&sub, question, &run.look),
                cost_ms: nmtype::now_ms().saturating_sub(task),
                agent_used: None,
                correct: None,
            };
            ledger_note = Value::Bool(nmlearn::ledger::append(path, &d).is_ok());
        }
        // MO5: the decisions of G1, G2 and the Core Brain into the behaviour log (with the Core
        // Brain side on), bound to the episode so a verdict becomes their outcome.
        let behaviour = match self.brain_side.read().ok().and_then(|s| s.clone()) {
            Some(side) => {
                let bucket = |k: usize| {
                    if k <= 4 {
                        "few"
                    } else if k <= 12 {
                        "some"
                    } else {
                        "many"
                    }
                };
                let events = [
                    (crate::brain_side::G1, format!("stop:{}:rounds:{}", run.g1_stop.replace(' ', "-"), run.g1_rounds)),
                    (crate::brain_side::G2, format!("send-up:{}", bucket(run.order.len()))),
                    (
                        "nmbrain.core-brain#0",
                        format!(
                            "look:{}:arm:{}",
                            if run.moved_by_look { "move" } else { "keep" },
                            if threshold.is_finite() { format!("{threshold}") } else { "never".into() }
                        ),
                    ),
                ];
                side.record_whole(&events, episode.as_deref()).map_or(Value::Null, Value::String)
            }
            None => Value::Null,
        };
        let n = |x: usize| Value::Number(x.to_string());
        let f = |x: f64| Value::Number(format!("{x:.3}"));
        let assembly_s = assembly_at.elapsed().as_secs_f64();
        let stages = Value::Object(
            run.stages.iter().map(|(k, v)| ((*k).to_owned(), f(if *k == "assembly" { *v + assembly_s } else { *v }))).collect(),
        );
        let context_map = Value::Object(
            [
                ("stages".to_owned(), stages),
                ("top_gap".to_owned(), Value::Number(if run.top_gap.is_finite() { format!("{:.3}", run.top_gap) } else { "1e9".into() })),
                ("look_ran".to_owned(), Value::Bool(run.look_ran)),
                ("rerank_docs".to_owned(), n(run.rerank_docs)),
                ("rerank_cap".to_owned(), n(run.rerank_cap)),
                ("rerank_dropped".to_owned(), n(run.rerank_dropped)),
                ("dense_added".to_owned(), Value::Array(run.dense_added.iter().map(|c| n(*c)).collect())),
                ("pre_look_first".to_owned(), Value::String(run.pre_look_first.clone())),
                ("judge_pieces".to_owned(), n(run.judge_pieces)),
                ("judge_windowed".to_owned(), n(run.judge_windowed)),
                ("judge_tokens".to_owned(), n(run.judge_tokens)),
                ("look_pieces".to_owned(), n(run.look_pieces)),
                ("look_tokens".to_owned(), n(run.look_tokens)),
                ("look_passes".to_owned(), n(run.look_passes)),
                (
                    // Each candidate's place and reranker P(yes), and the second look's scores, for
                    // paired comparison of runs (`bench`).
                    "scores".to_owned(),
                    Value::Array(
                        run.candidates
                            .iter()
                            .zip(&run.rerank)
                            .map(|(c, p)| (c, *p, "rr"))
                            .chain(run.look.iter().zip(&run.look_scores).map(|(c, p)| (c, *p, "look")))
                            .map(|(c, p, k)| {
                                let at = sub.chunks.get(*c).map_or(String::new(), |ch| {
                                    format!("{}:{}-{}", sub.files.get(ch.file).map_or("", |f| f.path.as_str()).replace('\\', "/"), ch.start, ch.end)
                                });
                                Value::String(format!("{k} {at} {p:.6}"))
                            })
                            .collect(),
                    ),
                ),
                (
                    "rerank_check".to_owned(),
                    run.rerank_check.map_or(Value::Null, |(d, same)| {
                        Value::Object(
                            [("max_diff".to_owned(), Value::Number(format!("{d:.6}"))), ("same_order".to_owned(), Value::Bool(same))]
                                .into_iter()
                                .collect(),
                        )
                    }),
                ),
                ("path".to_owned(), Value::String("whole".into())),
                ("mode".to_owned(), Value::String(mode.name().into())),
                ("sent_up".to_owned(), n(run.order.len())),
                ("queued".to_owned(), Value::Array(run.queued.iter().map(|c| n(*c)).collect())),
                ("second_look_moved".to_owned(), Value::Bool(run.moved_by_look)),
                ("g2".to_owned(), Value::Array(run.g2.iter().map(|x| Value::String(x.clone())).collect())),
                (
                    "packs".to_owned(),
                    Value::Array(
                        pack_ctl
                            .as_ref()
                            .map(|c| {
                                c.log
                                    .iter()
                                    .map(|d| {
                                        Value::String(format!(
                                            "{}: {} ({}{})",
                                            d.site.name(),
                                            d.chosen,
                                            d.by_pack.as_ref().map_or_else(|| "rule".to_owned(), |(f, s)| format!("pack {f} step {s}; ")),
                                            d.why
                                        ))
                                    })
                                    .collect()
                            })
                            .unwrap_or_default(),
                    ),
                ),
                (
                    "ask".to_owned(),
                    Value::Object(
                        [
                            ("mode".to_owned(), Value::String(format!("{:?}", w.ask).to_lowercase())),
                            ("judge".to_owned(), judge.as_ref().map_or(Value::Null, |j| Value::String(j.id.clone()))),
                            ("class".to_owned(), run.ask_class.map_or(Value::Null, |c| Value::String(format!("{c:?}").to_lowercase()))),
                            ("confidence".to_owned(), run.ask_conf.map_or(Value::Null, |c| Value::Number(format!("{c:.3}")))),
                            ("requested".to_owned(), Value::Array(run.asked.iter().map(|x| Value::String(x.clone())).collect())),
                        ]
                        .into_iter()
                        .collect(),
                    ),
                ),
                ("recalled_chunks".to_owned(), Value::Array(recalled_chunks.iter().map(|c| n(*c)).collect())),
                ("recall_sent_up".to_owned(), Value::Array(run.recalled.iter().map(|c| n(*c)).collect())),
                ("recalled_in_windows".to_owned(), n(run.recalled_in_windows)),
                ("memory_useful".to_owned(), n(useful.len())),
                ("cocited_chunks".to_owned(), n(cocited.len())),
                (
                    "meaning_floor".to_owned(),
                    self.memory
                        .lock()
                        .ok()
                        .filter(|_| self.memory_on && self.slots.by_name.get("reranker").is_some_and(|s| s.engine().is_some()))
                        .map_or(Value::Null, |m| {
                            let fl = crate::session::meaning_floor(self, &m);
                            Value::Object(
                                [("request".to_owned(), f(f64::from(fl.request))), ("material".to_owned(), f(f64::from(fl.material)))].into_iter().collect(),
                            )
                        }),
                ),
                ("second_look_margin".to_owned(), f(f64::from(run.look_margin))),
                ("ledger_written".to_owned(), ledger_note),
                ("behaviour_trace".to_owned(), behaviour),
                ("order".to_owned(), Value::String(run.order_line.clone())),
                ("faults".to_owned(), Value::Array(run.faults.iter().map(|x| Value::String(x.clone())).collect())),
                ("look_threshold".to_owned(), Value::Number(if threshold.is_finite() { threshold.to_string() } else { "1e9".into() })),
                ("look_threshold_propensity".to_owned(), Value::Number(propensity.to_string())),
                ("stack_s".to_owned(), f(run.stack_s)),
                ("core_brain_s".to_owned(), f(run.brain_s)),
            ]
            .into_iter()
            .collect(),
        );
        let version = self.nte.snapshot().version;
        let faults: Vec<Fault> = run.faults.iter().map(|x| bad(x.clone())).collect();
        Ok(Answer {
            pack,
            emitted,
            questions,
            version,
            ran: vec!["nmapi.whole".into()],
            skipped: Vec::new(),
            faults,
            raw_packets: raw,
            episode,
            written,
            recalled,
            ran_version: sub.version,
            context_map,
            brain: None,
        })
    }

    /// Re-read the studied repository; on any change, publish a new snapshot bound to the new
    /// substrate and move the brain to the new version (older packets are invalidated).
    /// # Errors
    /// The new binding or publication failed; the old snapshot keeps running.
    pub fn sync(&self) -> Result<(SyncReport, Vec<Packet>), Fault> {
        let mut next = (*self.sub()).clone();
        let (report, packets) = next.sync();
        if report.changed.is_empty() && report.added.is_empty() && report.removed.is_empty() {
            return Ok((report, packets));
        }
        let next = Arc::new(next);
        let graph = (*self.nte.snapshot().graph).clone();
        let comps = self.bind(&graph, &next)?;
        self.nte.publish(graph, comps)?;
        if let Ok(mut g) = self.sub.write() {
            *g = Arc::clone(&next);
        }
        nmengn::limits::set_area_used("content", "substrate", next.content_bytes());
        if let Some(tx) = self.dense_feed.lock().ok().and_then(|f| f.clone()) {
            let _ = tx.send(Arc::clone(&next));
        }
        if let Ok(mut b) = self.brain.lock() {
            b.set_version(next.version);
        }
        if let Ok(mut c) = self.changed_at.lock() {
            for f in report.changed.iter().chain(&report.added).chain(&report.removed) {
                c.insert(f.clone(), report.version);
            }
        }
        // Own changes against changes from outside, by content.
        if let (Ok(mut led), Ok(mut origin)) = (self.own.lock(), self.last_sync_origin.lock()) {
            let (mut own, mut outside) = (Vec::new(), Vec::new());
            for f in report.changed.iter().chain(&report.added) {
                let sha = std::fs::read(next.root.join(f)).map(|b| nmhash::sha256_bytes(&b)).unwrap_or_default();
                if led.observe(&f.replace('\\', "/"), &sha) { own.push(f.clone()) } else { outside.push(f.clone()) }
            }
            outside.extend(report.removed.iter().cloned());
            *origin = (own, outside);
        }
        Ok((report, packets))
    }
}

/// Section check (2026-11-13, review: the outward interface had no check of its own): an organism
/// opened on a small tree of its own, memory on in the check's folder. A value question is answered
/// with the defining file cited and no faults; a question about a term the tree does not hold is
/// withheld; the answered question becomes an episode, and once verified, asked again it is
/// recalled.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let dir = nmtype::check_dir("nmapi-forage").map_err(|e| e.to_string())?;
    let tree = dir.join("tree");
    for (p, t) in [
        (
            "src/limits.rs",
            "/// The most packets one tide may carry.\npub const TIDE_LIMIT: usize = 37;\n\n/// True when `n` packets fit one tide.\npub fn fits_tide(n: usize) -> bool {\n    n <= TIDE_LIMIT\n}\n",
        ),
        (
            "src/report.rs",
            "/// A plain report line for a count of shells.\npub fn shell_line(n: usize) -> String {\n    format!(\"{n} shells\")\n}\n",
        ),
        ("Cargo.toml", "[package]\nname = \"tidecheck\"\nversion = \"0.1.0\"\nedition = \"2024\"\n"),
    ] {
        let path = tree.join(p);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(&path, t).map_err(|e| e.to_string())?;
    }
    let home = nmtype::nano_root();
    let org = Organism::open_with(&home, &tree, &[], true, &dir.join("memory")).map_err(|e| e.to_string())?;
    let q = "What is the value of TIDE_LIMIT?";
    let a = org.ask(q, 2000).map_err(|e| e.to_string())?;
    let cites = |a: &Answer, f: &str| a.pack.packets.iter().any(|p| p.provenance().source.replace('\\', "/").contains(f));
    if a.pack.withheld || !cites(&a, "src/limits.rs") || !a.faults.is_empty() {
        return Err(format!(
            "nmapi.forage: value question withheld {}, cites limits.rs {}, faults {}",
            a.pack.withheld,
            cites(&a, "src/limits.rs"),
            a.faults.len()
        ));
    }
    let decoy = org.ask("Which function encrypts the harbour ledger with zebrafish keys?", 2000).map_err(|e| e.to_string())?;
    if !decoy.pack.withheld {
        return Err("nmapi.forage: a question about terms the tree does not hold must be withheld".into());
    }
    let Some(id) = a.episode.clone() else {
        return Err(format!("nmapi.forage: the answered question was not kept ({})", a.written));
    };
    // Only verified episodes are recall sources: the answer is verified first, as a passing check
    // would verify it.
    {
        let mut m = org.memory.lock().map_err(|_| "nmapi.forage: memory lock".to_owned())?;
        m.episodic.verdict(&id, true).map_err(|e| e.to_string())?;
        if let Some(m2) = m.mem2.as_mut() {
            m2.verdict(&id, true, "check").map_err(|e| e.to_string())?;
        }
    }
    let again = org.ask(q, 2000).map_err(|e| e.to_string())?;
    if again.recalled == 0 || !cites(&again, "src/limits.rs") {
        return Err(format!("nmapi.forage: asked again, recalled {} (cites limits.rs {})", again.recalled, cites(&again, "src/limits.rs")));
    }
    Ok(())
}

/// The base model's vocabulary as a measure of general word use, read once per process from the
/// models folder (`<models>/<base>/files/vocab.json`); `None` when it cannot be read, and the
/// withholding rule then excludes only function words and irregular verb forms.
pub fn general_use(home: &std::path::Path) -> Option<&'static nmsubs::general::GeneralUse> {
    static GENERAL: std::sync::OnceLock<Option<nmsubs::general::GeneralUse>> = std::sync::OnceLock::new();
    GENERAL
        .get_or_init(|| {
            let path = nmtype::models_dir(home).join(crate::neural::POPULATION).join("files").join("vocab.json");
            std::fs::read(&path).ok().and_then(|b| nmsubs::general::GeneralUse::from_vocab(&b).ok()).filter(|g| !g.is_empty())
        })
        .as_ref()
}

/// `Forager::drone_judgments` with the item as the unit of belief (`nmsubs::items`; lead's design
/// 1.1 to 1.3, 1.9): every item is scored by the drones from its own text (`Keep::item_evidence`);
/// each packet stands for the item its lines meet with the strongest evidence (accepted first),
/// and its share is that item's value in the joint order over the items the packets stand for
/// (fusion constant from `Evidence::plausible`). The drones' own findings are the accepted items
/// no packet cites yet: with `On`, each written as the item (whole when it fits `room`, else its
/// attached comment and definition line and its lines holding the request's parts, citing the
/// item's whole range in its evidence); with `Order`, as the item's pieces.
fn item_judgments(
    keep: &mut crate::drone_windows::Keep,
    sub: &Substrate,
    question: &str,
    packets: &mut Vec<Packet>,
    room: usize,
    mode: crate::drone_windows::ItemsMode,
) -> BTreeMap<u64, f32> {
    let mut out = BTreeMap::new();
    let ((ev, accepted, pev), c) = crate::tools::clocked(|| {
        if keep.sync(sub).is_err() || keep.sync_items(sub).is_err() {
            return (None, 0, None);
        }
        let meaning = sub.dense.read().ok().and_then(|d| d.clone()).and_then(|l| l.cosines(sub, question));
        let ev = keep.item_evidence(question, meaning.as_deref()).ok();
        let accepted = ev.as_ref().map_or(0, |e| (0..e.len()).filter(|i| (0..e.aspects.len()).any(|a| e.accepted(a, *i))).count());
        // With `Order`, the drones' own findings stay the pieces they accept.
        let pev = (mode == crate::drone_windows::ItemsMode::Order).then(|| keep.evidence_with(question, meaning.as_deref()).ok()).flatten();
        (ev, accepted, pev)
    });
    let items = keep.items();
    crate::tools::note(crate::tools::Step {
        cells: vec!["evidence".into()],
        counts: vec![("items", items.items.len() as u64), ("accepted", accepted as u64)],
        ..crate::tools::step_at("nmpod.evidence", c)
    });
    let Some(ev) = ev else { return out };
    let e = ev.pieces();
    if e.iter().copied().fold(0.0_f32, f32::max) <= 0.0 {
        return out;
    }
    let path_of = |fi: usize| sub.files.get(fi).map(|f| f.path.replace('\\', "/")).unwrap_or_default();
    let mut by_file: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, it) in items.items.iter().enumerate() {
        by_file.entry(path_of(it.file)).or_default().push(i);
    }
    let is_accepted = |i: usize| (0..ev.aspects.len()).any(|a| ev.accepted(a, i));
    let floor = nmbrain::synthesis::FLOOR;
    // The lines of an item a pack shows when the whole does not fit: its attached comment and
    // definition line, and its own lines holding a stem of a part it holds (of any part when it
    // holds none), as runs of lines.
    let request = nmpod::intake::Request::read(question, &[]);
    let part_stems: Vec<Vec<String>> = request.aspects.iter().flat_map(|a| a.parts.iter()).map(|p| p.stems.clone()).collect();
    let shown_runs = |i: usize| -> Vec<(usize, usize)> {
        let Some(it) = items.items.get(i) else { return Vec::new() };
        let Some(f) = sub.files.get(it.file) else { return Vec::new() };
        let held: Vec<usize> = (0..ev.parts.len()).filter(|k| ev.part_holds(*k, i)).collect();
        let stems: Vec<&str> = if held.is_empty() {
            part_stems.iter().flatten().map(String::as_str).collect()
        } else {
            held.iter().filter_map(|k| part_stems.get(*k)).flatten().map(String::as_str).collect()
        };
        let lines: Vec<&str> = f.text.lines().collect();
        let holds = |l: &str| {
            l.split(|x: char| !(x.is_alphanumeric() || x == '_')).filter(|w| !w.is_empty()).flat_map(nmlex::name_parts).any(|p| stems.contains(&nmlex::stem(&p).as_str()))
        };
        let mut keep_lines: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        if let Some((a, b)) = it.described {
            keep_lines.extend(a..=b);
        }
        if it.def > 0 {
            keep_lines.insert(it.def);
        }
        for (a, b) in &it.runs {
            for n in *a..=*b {
                if lines.get(n.saturating_sub(1)).is_some_and(|l| holds(l)) {
                    keep_lines.insert(n);
                }
            }
        }
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for n in keep_lines {
            match runs.last_mut() {
                Some(r) if r.1 + 1 == n => r.1 = n,
                _ => runs.push((n, n)),
            }
        }
        runs
    };
    // Every accepted item no packet cites yet goes in as the drones' own finding.
    let cited: std::collections::BTreeSet<String> = packets.iter().map(|p| p.provenance().source.replace('\\', "/")).collect();
    let top = e.iter().copied().fold(0.0_f32, f32::max);
    if let Some(pev) = &pev {
        let pe = pev.pieces();
        let ptop = pe.iter().copied().fold(0.0_f32, f32::max).max(f32::MIN_POSITIVE);
        for (i, ch) in sub.chunks.iter().enumerate() {
            if !(0..pev.aspects.len()).any(|a| pev.accepted(a, i)) {
                continue;
            }
            let Some(f) = sub.files.get(ch.file) else { continue };
            let src = format!("{}:{}", f.path.replace('\\', "/"), ch.start);
            if cited.contains(&src) {
                continue;
            }
            let from = nmtype::Provenance::new(&src, "nmpod.evidence", sub.version, nmtype::now_ms());
            packets.push(
                Packet::new("chunk", &format!("chunk:{src}"), &ch.text, from)
                    .status(nmtype::Status::Inferred)
                    .confidence(pe.get(i).copied().unwrap_or(0.0) / ptop)
                    .evidence(nmsubs::views::piece_evidence(ch, format!("{src} lines {}..{}", ch.start, ch.end))),
            );
        }
    }
    for (i, it) in items.items.iter().enumerate() {
        if pev.is_some() || !is_accepted(i) {
            continue;
        }
        let Some(f) = sub.files.get(it.file) else { continue };
        let path = path_of(it.file);
        let lines: Vec<&str> = f.text.lines().collect();
        let whole = format!("{path}:{}-{}", it.start(), it.end());
        let pieces: Vec<usize> = items.pieces_of.get(i).cloned().unwrap_or_default();
        let first_piece = pieces.first().and_then(|p| sub.chunks.get(*p));
        // What is written: the pieces (`Order`), the item whole, or its shown lines.
        let runs: Vec<(usize, usize)> = match mode {
            crate::drone_windows::ItemsMode::Order => pieces.iter().filter_map(|p| sub.chunks.get(*p)).map(|ch| (ch.start, ch.end)).collect(),
            _ => {
                let tokens: usize = lines.get(it.start().saturating_sub(1)..it.end().min(lines.len())).map_or(0, |v| v.iter().map(|l| l.chars().count() + 1).sum::<usize>()) / 4;
                if tokens <= room { vec![(it.start(), it.end())] } else { shown_runs(i) }
            }
        };
        for (a, b) in runs {
            let src = format!("{path}:{a}");
            if cited.contains(&src) {
                continue;
            }
            let text = lines.get(a.saturating_sub(1)..b.min(lines.len())).map(|v| v.join("\n")).unwrap_or_default();
            let from = nmtype::Provenance::new(&src, "nmpod.evidence", sub.version, nmtype::now_ms());
            let mut evidence = first_piece.map_or_else(|| vec![format!("{src} lines {a}..{b}")], |ch| nmsubs::views::piece_evidence(ch, format!("{src} lines {a}..{b}")));
            if (a, b) != (it.start(), it.end()) && mode == crate::drone_windows::ItemsMode::On {
                evidence.push(format!("item {whole}"));
            }
            packets.push(
                Packet::new("chunk", &format!("chunk:{src}"), &text, from)
                    .status(nmtype::Status::Inferred)
                    .confidence(e.get(i).copied().unwrap_or(0.0) / top)
                    .evidence(evidence),
            );
        }
    }
    // Each packet stands for the item its lines meet with the strongest evidence, accepted first.
    let mut held: Vec<(u64, (f32, bool, usize))> = Vec::new();
    for p in packets.iter() {
        let Some((path, line)) = p.provenance().source.rsplit_once(':') else { continue };
        let Ok(at) = line.parse::<usize>() else { continue };
        let (a, b) = p
            .get_evidence()
            .iter()
            .find_map(|x| {
                let (a, b) = x.split_once(" lines ")?.1.split_whitespace().next()?.split_once("..")?;
                Some((a.parse().ok()?, b.parse().ok()?))
            })
            .unwrap_or((at, at));
        let Some(list) = by_file.get(&path.replace('\\', "/")) else { continue };
        let best = list
            .iter()
            .filter(|i| items.items.get(**i).is_some_and(|it| it.runs.iter().any(|(s, t)| *s <= b && *t >= a)))
            .map(|i| (e.get(*i).copied().unwrap_or(0.0), is_accepted(*i), *i))
            .fold(None, |m: Option<(f32, bool, usize)>, x| match m {
                Some(y) if (y.1, y.0) >= (x.1, x.0) => Some(y),
                _ => Some(x),
            });
        if let Some(x) = best {
            held.push((p.id(), x));
        }
    }
    let mut cands: Vec<usize> = held.iter().map(|h| h.1.2).collect();
    cands.sort_unstable();
    cands.dedup();
    let file_of = |i: usize| items.items.get(i).map_or(usize::MAX, |x| x.file);
    #[expect(clippy::cast_precision_loss, reason = "a count of items")]
    let k = ev.plausible(&cands) as f32;
    let order = ev.joint_order(&cands, &file_of, k);
    // For a measurement: the first items of the order on standard error (`NANO_ITEMS_DEBUG`).
    if std::env::var("NANO_ITEMS_DEBUG").is_ok() {
        eprintln!("items: {question} | k {k} | {} candidates of {} items, {} packets", cands.len(), items.items.len(), held.len());
        for (i, v) in order.iter().take(12) {
            let it = items.items.get(*i);
            eprintln!(
                "  {:.4} {}:{}-{} {} acc {} e {:.2} settle {:.3}",
                v,
                it.map_or(String::new(), |x| path_of(x.file)),
                it.map_or(0, nmsubs::items::Item::start),
                it.map_or(0, nmsubs::items::Item::end),
                it.map_or(String::new(), |x| x.label.clone()),
                is_accepted(*i),
                e.get(*i).copied().unwrap_or(0.0),
                ev.settle_sum(*i)
            );
        }
    }
    let first = order.first().map_or(1.0, |x| x.1);
    let last = order.last().map_or(0.0, |x| x.1);
    let span = (first - last).max(f32::MIN_POSITIVE);
    let joint: BTreeMap<usize, f32> = order.into_iter().map(|(c, v)| (c, (v - last) / span)).collect();
    for (id, (_, ok, i)) in held {
        let share = joint.get(&i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        out.insert(id, if ok { floor + (1.0 - floor) * share } else { floor * share });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountEngine {
        embed_calls: Arc<AtomicUsize>,
    }

    impl nmbrain::slot::Engine for CountEngine {
        fn embed(&self, text: &str) -> Result<Vec<f32>, Fault> {
            self.embed_calls.fetch_add(1, Ordering::SeqCst);
            let h = nmhash::sha256_bytes(text.as_bytes());
            let mut v = vec![0.0_f32; 8];
            for (i, b) in h.bytes().take(8).enumerate() {
                v[i] = f32::from(b) / 255.0;
            }
            Ok(v)
        }
        fn rerank(&self, _q: &str, _d: &str) -> Result<f32, Fault> {
            Ok(0.8)
        }
        fn resident_mb(&self) -> (f32, f32) {
            (1.0, 0.0)
        }
        fn release_device(&self) {}
        fn label(&self) -> String {
            "count-mock".to_owned()
        }
    }

    #[test]
    fn ask_whole_flow_withholds_dec08_and_preserves_in_repo_loc08() {
        let dir = nmtype::check_dir("nmapi-forage-dec08").unwrap();
        let tree = dir.join("tree");
        for (rel, body) in [
            (
                "crates/nmhash/src/lib.rs",
                "/// Compute the SHA-256 content key.\npub fn sha256_bytes(bytes: &[u8]) -> String {\n    format!(\"{:x}\", bytes.len())\n}\n",
            ),
            (
                "crates/nmsubs/src/general.rs",
                "/// Check that a file name inside an archive would not lead outside the archive.\npub fn escapes(name: &str) -> bool {\n    name.starts_with('/') || name.contains(\"..\")\n}\n\n/// Read an archive member after checking that its file name does not lead outside the archive.\npub fn read_archive_entry(file_name: &str) -> bool {\n    !escapes(file_name)\n}\n",
            ),
            ("settings.json", "{\"hash\": \"sha256\"}\n"),
            ("Cargo.toml", "[package]\nname = \"sample\"\nversion = \"0.1.0\"\nedition = \"2024\"\n"),
        ] {
            let p = tree.join(rel);
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(p, body).unwrap();
        }
        let home = nmtype::nano_root();
        let org = Organism::open_with(&home, &tree, &[], false, &dir.join("mem")).unwrap();
        *org.whole.write().unwrap() = Some(Arc::new(crate::whole::WholePath::new(&home)));

        let decoy = org
            .ask_in(
                "Which setting picks the hash algorithm, for example BLAKE3 or SHA-1, used for content keys?",
                4000,
                Some(crate::whole::Mode::Flow),
            )
            .unwrap();
        assert!(decoy.pack.withheld, "dec08 must be withheld: {:?}", decoy.pack.reason);
        assert!(decoy.pack.packets.is_empty());
        assert_eq!(decoy.pack.tokens, 0);
        assert!(!decoy.pack.floor_met);
        assert!(
            decoy.pack.reason.to_lowercase().contains("blake3")
                || decoy.pack.reason.to_lowercase().contains("withheld"),
            "unexpected decoy withhold reason: {}",
            decoy.pack.reason
        );
        assert!((org.slots.last_best() - (-1.0)).abs() < 1e-6);

        let real = org
            .ask_in(
                "Where is escapes defined to check that a file name inside an archive would not lead outside the archive?",
                4000,
                Some(crate::whole::Mode::Flow),
            )
            .unwrap();
        assert!(!real.pack.withheld, "loc08 must not be withheld: {:?}", real.pack.reason);
        assert!(!real.pack.packets.is_empty());
        assert!(real.pack.tokens > 0);
        assert!((org.slots.last_best() - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn ask_recalled_chunks_and_recall_timed_reuse_meaning() {
        let dir = nmtype::check_dir("nmapi-forage-last-reuse").unwrap();
        let tree = dir.join("tree");
        std::fs::create_dir_all(tree.join("src")).unwrap();
        std::fs::write(tree.join("src/lib.rs"), "pub fn parse_tide(n: u32) -> u32 {\n    n + 1\n}\n").unwrap();
        let home = nmtype::nano_root();
        let org = Organism::open_with(&home, &tree, &[], true, &dir.join("mem")).unwrap();
        let embed_calls = Arc::new(AtomicUsize::new(0));
        let eng = Arc::new(CountEngine { embed_calls: Arc::clone(&embed_calls) });
        org.slots.bind("embedder", Arc::clone(&eng) as Arc<dyn nmbrain::slot::Engine>).unwrap();
        org.slots.bind("reranker", eng as Arc<dyn nmbrain::slot::Engine>).unwrap();

        let sub = org.sub();
        let cite = sub.cite_ref("src/lib.rs:1 lines 1..3");
        {
            let mut m = org.memory.lock().unwrap();
            let mut ep = nmmem::episodic::Episode::new("neuromorph", "test", "where is parse_tide defined?", "src/lib.rs:1", vec![cite], sub.version);
            ep.id = "ep-reuse".to_owned();
            let _ = m.episodic.write(ep);
            let _ = m.episodic.verdict("ep-reuse", true);
        }
        let _ = crate::session::recalled_chunks(&org, "warmup question");
        embed_calls.store(0, Ordering::SeqCst);

        let q = "how does parse_tide work?";
        let _ = crate::session::recalled_chunks(&org, q);
        let _ = org.recall_timed(q, None);
        assert_eq!(embed_calls.load(Ordering::SeqCst), 1, "LAST cache should reuse meaning on recall_timed");
    }
}

/// Whether `question` asks for a change to the code (the Core Brain's `propose-edit` task, which is
/// classified only inside the chain book): its first word is an edit verb. Roadmap Phase 1 item 1
/// (2026-11-24): such requests are architectural packs and carry the project outline.
#[must_use]
pub fn edit_request(question: &str) -> bool {
    const VERBS: [&str; 16] = ["fix", "change", "add", "implement", "refactor", "rename", "remove", "delete", "update", "make", "replace", "move", "rewrite", "extend", "support", "split"];
    let first = question.trim_start().split(|c: char| !c.is_alphanumeric()).next().unwrap_or("").to_ascii_lowercase();
    VERBS.contains(&first.as_str())
}

#[cfg(test)]
mod edit_request_tests {
    use super::edit_request;

    #[test]
    fn edit_verbs_lead_edit_requests() {
        assert!(edit_request("Fix observe so it skips empty edits"));
        assert!(edit_request("  add a flag for the cache"));
        assert!(!edit_request("Where is ParMap defined?"));
        assert!(!edit_request("How does the reader decide how many spans to read?"));
    }
}
