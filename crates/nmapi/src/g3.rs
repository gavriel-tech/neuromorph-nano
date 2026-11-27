//! G3 and muscle memory on the served path (`serve --muscle on`, off by default). Muscle memory is
//! not continuous learning: no learning layer runs here, `--learn` never starts this, and nothing
//! here reads or writes a learning layer's state (2026-11-18, Luis Jake Gabriel III through the
//! lead).
//!
//! Decision sites (each a choice that changes what is served):
//! - `g1.allocation`: the Map Coordinator's allocation of the second round (`nbv`, `uniform`,
//!   `off`), on every call.
//! - `g2.chain`: which MicroModel chain the stack hands the request to (`fast`: the reranker over
//!   the first pieces; `full`: every part the arrangement allows; `flow`: G2's set in its order with
//!   no MicroModel round). Only where the stack serves (`serve --whole` or `--flow`, the served
//!   default since 2026-11-18) and the call names no mode; a mode the request names is never
//!   overridden.
//!
//! With it on, each `nm_context_pack` call:
//! 1. `begin`: at each site the configured choice is the rule; G3's priors for the request's kind
//!    and area (packs reloaded from muscle memory, and biases whose gain held on held-out records)
//!    arbitrate against it (`nmmusc::priors::Control`); the chosen choices run for this call only.
//! 2. `end`: the choices are put back; the call's configuration record goes to G3 in RAM and to the
//!    call's diagnostics (so the audit row carries it): kind, area, choices, who chose them, and what
//!    the flow did (the parts that ran, the pack's status). The places the reply cited and its time
//!    are kept in RAM.
//! 3. Positive by the organism's own measure (master design 3.5, G3 rule 3: performance rises): each
//!    call is its own reference at once; a replay with another choice is judged by how much of the
//!    served reply's places it keeps (reciprocal-place agreement, 1 for the same order) with time as
//!    a cost below one place, so a choice that reaches the same places in less time or fewer rounds
//!    gains. Claude Code's verdicts are one more signal when present: the evidence ledger's verified
//!    `answered` outcome (joined through the audit row, `answered:<audit id>` whose `config.call` is
//!    the call) replaces the reference with the verified places, so a choice that recovers an answer
//!    the served reply missed also gains.
//! 4. The cycle (`serve`: at idle, `serve_idle`): G3 replays what it judges through the served call
//!    with one site changed (paired replay, same request, nothing written: no memory, no answer
//!    cache, no audit), mines the records (choices and the flow's behaviour together), selects,
//!    promotes and withdraws (`nmmusc::g3`).
//!
//! The request's words and the reply's places stay in this module's RAM for replays only; G3 sees
//! behaviour only.
use crate::forage::Organism;
use nmjson::Value;
use nmmusc::MuscleStore;
use nmmusc::g3::{Cycle, G3, Origin, Record};
use nmmusc::priors::{Prior, Site};
use nmtype::{Fault, FaultKind};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// The modes the allocation site offers.
pub const ALLOCATIONS: [&str; 3] = ["nbv", "uniform", "off"];
/// The chains the stack offers (`crate::whole::Mode`).
pub const CHAINS: [&str; 3] = ["fast", "full", "flow"];

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmapi.g3", FaultKind::Config, m)
}
fn musc(e: &nmmusc::Fault) -> Fault {
    let kind = if e.code == "nmmusc-sensory-refused" { FaultKind::Policy } else { FaultKind::Config };
    Fault::new("nmapi.g3", kind, e.to_string())
}

/// Places as (path, first line, last line).
pub type Places = Vec<(String, usize, usize)>;

/// What a served reply left for its verdict: the places it cited, in order, and its time.
#[derive(Clone, Debug, Default)]
pub struct Served {
    pub cited: Vec<(String, usize, usize)>,
    pub ms: u64,
}

/// G3's state in the serving process.
pub struct MuscleState {
    pub g3: G3,
    pub store: Option<MuscleStore>,
    /// Each call's tool input, in RAM, for its replays.
    inputs: BTreeMap<String, Value>,
    /// Each call's cited places and time, in RAM, for its verdict.
    served: BTreeMap<String, Served>,
    /// The places verified for a call (from the ledger), in RAM, to score its replays.
    verified: BTreeMap<String, Vec<(String, usize, usize)>>,
    next: u64,
    /// The last call recorded.
    pub last: Option<String>,
    /// The configured allocation (`serve --refine`, default `nbv`), as the last call found it.
    pub configured: String,
    /// Outcomes joined since the last cycle (the calls' own, and the ledger's).
    pub fresh: usize,
    /// The last cycle run, and how many have run.
    pub last_cycle: Option<Cycle>,
    pub cycles: usize,
}

static STATE: Mutex<Option<MuscleState>> = Mutex::new(None);
thread_local! {
    /// The choices a replay runs with, on the replaying thread only.
    static FORCED: RefCell<Option<Vec<(Site, String)>>> = const { RefCell::new(None) };
    /// Phase J1: the decision sites logged on this thread for the call running on it.
    static SITES: RefCell<nmmusc::sites::SiteLog> = RefCell::new(nmmusc::sites::SiteLog::default());
    /// WP6: the configuration record of the call running on this thread, when the flow wrote one.
    static CONFIG: RefCell<Option<nmmusc::config::CallConfig>> = const { RefCell::new(None) };
    /// The kind and area of the call running on this thread (for priors at the flow's own sites).
    static CURRENT: RefCell<Option<(String, String)>> = const { RefCell::new(None) };
    /// Choices the flow made on this thread at sites decided outside G3's own controls (the L1
    /// tuner's look arm), noted for the call's record (`note_choice`).
    static NOTED: RefCell<Vec<(Site, String)>> = const { RefCell::new(Vec::new()) };
}

/// WP6: write the configuration record of the call running on this thread (kind, aspects, lenses,
/// orders, chain, cartridges, budgets; master design 3.7 rule 12). It joins the call's G3 record at
/// `end` as behaviour steps G3 mines and goes into the call's diagnostics as `config`. Does nothing
/// while G3 is off and no replay runs.
/// # Errors
/// A field that is not a label (material is refused).
pub fn note_config(cfg: nmmusc::config::CallConfig) -> Result<(), Fault> {
    if !is_on() && !replaying() {
        return Ok(());
    }
    cfg.validate().map_err(|e| musc(&e))?;
    CONFIG.with(|c| *c.borrow_mut() = Some(cfg));
    Ok(())
}

/// The choice at one of the flow's learnable sites (`team.mode`, `core-brain.intake`,
/// `tentacle.integrations`): a replay's forced choice; else G3's prior for the call's kind and area
/// when its evidence beats the rule's margin (0) and G2 does not veto (`veto`, e.g. the desk's
/// `veto_packs`); else the rule's own choice. With G3 off: the rule's choice, always.
#[must_use]
pub fn decide_site(site: Site, rule: &str, veto: bool) -> String {
    if let Some(c) = FORCED.with(|f| f.borrow().as_ref().and_then(|v| v.iter().find(|(s, _)| *s == site).map(|(_, c)| c.clone()))) {
        return c;
    }
    let allowed = nmmusc::sites::alternatives(site);
    let Some((kind, area)) = CURRENT.with(|c| c.borrow().clone()) else { return rule.to_owned() };
    with(|st| {
        let mut ctl = st.g3.control(&kind, &area);
        ctl.decide(site, rule, allowed, &|_| 0.0, &|_| veto, &|_| false)
    })
    .unwrap_or_else(|| rule.to_owned())
}

/// Phase J1 (tentacle master plan 3.10; lane M 2026-11-19): log one decision of the call running
/// on this thread at a decision site, as a behavioural envelope to G3 (`nmmusc::sites`). Does
/// nothing while G3 is off and no replay runs, so the served path is unchanged with muscle memory
/// off. The steps join the call's configuration record at `end` (`Record::observed`).
/// # Errors
/// A choice or subject that is not a short label (material is refused).
pub fn note_site(site: nmmusc::sites::SiteKind, unit: &str, subject: &str, choice: &str, landed: Option<bool>) -> Result<(), Fault> {
    if !is_on() && !replaying() {
        return Ok(());
    }
    SITES.with(|s| s.borrow_mut().note(site, unit, subject, choice, landed)).map_err(|e| musc(&e))
}

/// The decision sites logged on this thread since the call began, taken (the log is emptied).
#[must_use]
pub fn take_sites() -> nmmusc::sites::SiteLog {
    SITES.with(|s| std::mem::take(&mut *s.borrow_mut()))
}

/// The base population's hash packs must match.
#[must_use]
pub fn base_hash(home: &Path) -> String {
    nmlearn::cartridge::Loaded::population(home, crate::brain_side::BASE, Vec::new()).map(|l| l.base_sha256).unwrap_or_default()
}

/// Switch G3 and muscle memory on (`serve --muscle on`): G3 for the base in use, muscle memory from
/// `store_dir` (the live store, `designs/muscle.json`, when `None`), its packs reloaded. Returns the
/// packs loaded.
/// # Errors
/// The store cannot be opened or read.
pub fn enable(home: &Path, store_dir: Option<&Path>) -> Result<usize, Fault> {
    let store = match store_dir {
        Some(d) => MuscleStore::open(d),
        None => MuscleStore::live(),
    }
    .map_err(|e| musc(&e))?;
    let mut g3 = G3::new(&base_hash(home));
    let n = g3.reload(&store).map_err(|e| musc(&e))?;
    let mut s = STATE.lock().map_err(|_| bad("muscle lock poisoned"))?;
    *s = Some(MuscleState {
        g3,
        store: Some(store),
        inputs: BTreeMap::new(),
        served: BTreeMap::new(),
        verified: BTreeMap::new(),
        next: 0,
        last: None,
        configured: "nbv".into(),
        fresh: 0,
        last_cycle: None,
        cycles: 0,
    });
    Ok(n)
}

/// Switch G3 and muscle memory off; the state they held.
pub fn disable() -> Option<MuscleState> {
    STATE.lock().ok().and_then(|mut s| s.take())
}

#[must_use]
pub fn is_on() -> bool {
    STATE.lock().is_ok_and(|s| s.is_some())
}

/// Run `f` with the state, if G3 is on.
pub fn with<T>(f: impl FnOnce(&mut MuscleState) -> T) -> Option<T> {
    STATE.lock().ok()?.as_mut().map(f)
}

/// Whether this thread is running a replay.
#[must_use]
pub fn replaying() -> bool {
    FORCED.with(|f| f.borrow().is_some())
}

/// The area a call works in: the project.
fn area(o: &Organism) -> String {
    nmtype::project::label(&o.home, &o.sub().root)
}

/// The chain the stack would run for this call, when the stack serves and the call names no mode.
fn chain_rule(o: &Organism, input: &Value) -> Option<String> {
    if input.get("mode").is_ok() {
        return None;
    }
    let default = o.whole.read().ok().and_then(|w| w.as_ref().map(|p| p.default_mode))?;
    Some(default.unwrap_or_else(|| crate::whole::mode_control(&o.home)).name().to_owned())
}

/// One call's decisions.
#[derive(Clone, Debug)]
pub struct Call {
    pub id: String,
    pub kind: String,
    pub area: String,
    pub allocation: String,
    /// The chain chosen, when the chain site is open for this call.
    pub chain: Option<String>,
    /// The rule's choice at the allocation site (the configured mode).
    pub rule: String,
    /// What decided, per site: `rule`, `pack:<fingerprint>`, `bias:<context>` or `replay`.
    pub by: String,
    /// The input the call runs with, when a chain choice changed it.
    pub input: Option<Value>,
    restore: String,
    record: bool,
    started: Instant,
}

/// Decide the call's choices and set them; `None` with G3 off and no replay running.
pub fn begin(o: &Organism, input: &Value) -> Option<Call> {
    let configured = o.refine.read().map(|r| r.clone()).unwrap_or_else(|_| "nbv".into());
    let forced = FORCED.with(|f| f.borrow().clone());
    let question = input.get("question").and_then(Value::string).unwrap_or_default();
    let kind = crate::whole::kind_of(question).to_owned();
    let chain_rule = chain_rule(o, input);
    let _ = take_sites();
    CONFIG.with(|c| *c.borrow_mut() = None);
    CURRENT.with(|c| *c.borrow_mut() = None);
    NOTED.with(|c| c.borrow_mut().clear());
    let mut call = if let Some(choices) = forced {
        let pick = |s: Site| choices.iter().find(|(x, _)| *x == s).map(|(_, c)| c.clone());
        Call {
            id: String::new(),
            kind,
            area: String::new(),
            allocation: pick(Site::Allocation).unwrap_or_else(|| configured.clone()),
            chain: chain_rule.as_ref().map(|r| pick(Site::Chain).unwrap_or_else(|| r.clone())),
            rule: configured.clone(),
            by: "replay".into(),
            input: None,
            restore: configured,
            record: false,
            started: Instant::now(),
        }
    } else {
        let mut g = STATE.lock().ok()?;
        let st = g.as_mut()?;
        let area = area(o);
        st.configured.clone_from(&configured);
        let rule = configured.clone();
        let none = |_: &Prior| false;
        let mut by: Vec<String> = Vec::new();
        let (allocation, chain) = {
            let mut ctl = st.g3.control(&kind, &area);
            let a = ctl.decide(Site::Allocation, &rule, &ALLOCATIONS, &|_| 0.0, &none, &none);
            let decided = |ctl: &nmmusc::priors::Control| {
                ctl.log.last().and_then(|d| d.by_pack.as_ref().map(|(p, _)| if p.starts_with("bias:") { p.clone() } else { format!("pack:{p}") }))
            };
            if let Some(b) = decided(&ctl) {
                by.push(b);
            }
            let c = chain_rule.as_ref().map(|r| {
                let before = ctl.log.len();
                let c = ctl.decide(Site::Chain, r, &CHAINS, &|_| 0.0, &none, &none);
                if ctl.log.len() > before
                    && let Some(b) = decided(&ctl)
                    && !by.contains(&b)
                {
                    by.push(b);
                }
                c
            });
            (a, c)
        };
        st.next += 1;
        let id = format!("{}-{}", nmtype::now_ms(), st.next);
        st.inputs.insert(id.clone(), input.clone());
        Call {
            id,
            kind,
            area,
            allocation,
            chain,
            rule,
            by: if by.is_empty() { "rule".into() } else { by.join(",") },
            input: None,
            restore: configured,
            record: true,
            started: Instant::now(),
        }
    };
    if let (Some(c), Some(r)) = (&call.chain, &chain_rule)
        && c != r
        && let Value::Object(mut m) = input.clone()
    {
        m.insert("mode".into(), Value::String(c.clone()));
        call.input = Some(Value::Object(m));
    }
    if call.allocation != call.restore
        && let Ok(mut r) = o.refine.write()
    {
        r.clone_from(&call.allocation);
    }
    CURRENT.with(|c| *c.borrow_mut() = Some((call.kind.clone(), call.area.clone())));
    Some(call)
}

/// The places a text reply cites, in order (`@@ path:a-b` and `~~ path:a` headers).
#[must_use]
pub fn cited(reply: &str) -> Vec<(String, usize, usize)> {
    let mut out = Vec::new();
    for l in reply.lines() {
        let (head, span) = if let Some(r) = l.strip_prefix("@@ ") {
            (r, true)
        } else if let Some(r) = l.strip_prefix("~~ ") {
            (r, false)
        } else {
            continue;
        };
        let Some((path, at)) = head.split_whitespace().next().map(|h| h.trim_end_matches(';')).and_then(|h| h.rsplit_once(':')) else { continue };
        let (a, b) = if span { at.split_once('-').unwrap_or((at, at)) } else { (at, at) };
        if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
            out.push((path.replace('\\', "/"), a, b));
        }
    }
    out
}

/// Mean reciprocal place of `answers` in `cited` (0 for an answer not cited).
#[must_use]
#[expect(clippy::cast_precision_loss, reason = "places and counts")]
pub fn reciprocal_place(cited: &[(String, usize, usize)], answers: &[(String, usize, usize)]) -> f64 {
    if answers.is_empty() {
        return 0.0;
    }
    let hit = |a: &(String, usize, usize), x: &(String, usize, usize)| a.0 == x.0 && x.1 <= a.2 && x.2 >= a.1;
    answers.iter().map(|a| cited.iter().position(|x| hit(a, x)).map_or(0.0, |p| 1.0 / (p + 1) as f64)).sum::<f64>() / answers.len() as f64
}

/// A replay's outcome with time as a cost below one place: `places + eps * (1 - ms / ref_ms)`
/// (the time term held in [-1, 1]), where `eps` is half the smallest step one answer's place can
/// make among `cited` places (`1 / (k (k + 1))` over `answers` answers). Equal places: the faster
/// choice gains; a place moved: the places decide, whatever the time.
#[must_use]
#[expect(clippy::cast_precision_loss, reason = "places, counts and milliseconds")]
pub fn outcome(places: f64, answers: usize, cited: usize, ms: u64, ref_ms: u64) -> f64 {
    let k = cited.max(1) as f64;
    let eps = 0.5 / (answers.max(1) as f64 * k * (k + 1.0));
    let t = if ref_ms == 0 { 0.0 } else { (1.0 - ms as f64 / ref_ms as f64).clamp(-1.0, 1.0) };
    places + eps * t
}

/// What the flow did on a call, as behaviour, from its diagnostics: the parts that ran
/// (`ran:<part>`, the stage of a part instance only) and the pack's status.
#[must_use]
pub fn behaviour(diag: Option<&Value>) -> Vec<String> {
    let Some(d) = diag else { return Vec::new() };
    let mut out: Vec<String> = Vec::new();
    for p in d.get("parts").and_then(Value::array).unwrap_or_default() {
        if let Ok(t) = p.string() {
            let base = t.split('#').next().unwrap_or(t);
            let step = format!("ran:{base}");
            if !out.contains(&step) {
                out.push(step);
            }
        }
    }
    if let Ok(Value::Bool(w)) = d.get("withheld") {
        out.push(if *w { "status:withheld".into() } else { "status:served".into() });
    }
    if let Ok(Value::Bool(f)) = d.get("floor_met") {
        out.push(if *f { "floor:met".into() } else { "floor:unmet".into() });
    }
    // Phase J1: decision-site steps the flow put in its diagnostics (`sites`, each a step of
    // `nmmusc::sites`); a step that does not read back as labels is dropped.
    for p in d.get("sites").and_then(Value::array).unwrap_or_default() {
        if let Ok(t) = p.string()
            && nmmusc::sites::SiteEvent::parse_step(t, "flow").is_ok()
            && !out.iter().any(|x| x == t)
        {
            out.push(t.to_owned());
        }
    }
    out
}

/// Put the choices back and record the call; the configuration record as a value for the call's
/// diagnostics. `reply` is the text reply, whose cited places are kept; `diag` the call's
/// diagnostics, from which the flow's behaviour is read. The call is its own reference at once
/// (outcome 1, module note step 3).
pub fn end(o: &Organism, call: Call, reply: Option<&str>, diag: Option<&Value>) -> Value {
    if call.allocation != call.restore
        && let Ok(mut r) = o.refine.write()
    {
        r.clone_from(&call.restore);
    }
    let ms = u64::try_from(call.started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let sites = take_sites();
    let config = CONFIG.with(|c| c.borrow_mut().take());
    let noted = NOTED.with(|c| std::mem::take(&mut *c.borrow_mut()));
    CURRENT.with(|c| *c.borrow_mut() = None);
    if call.record {
        let _ = with(|st| {
            let mut choices = vec![(Site::Allocation, call.allocation.clone())];
            if let Some(c) = &call.chain {
                choices.push((Site::Chain, c.clone()));
            }
            choices.extend(sites.choices());
            for (site, choice) in noted {
                if !choices.iter().any(|(s, _)| *s == site) {
                    choices.push((site, choice));
                }
            }
            let rec = Record {
                id: call.id.clone(),
                kind: call.kind.clone(),
                area: call.area.clone(),
                choices,
                by_pack: call.by.split(',').find_map(|b| b.strip_prefix("pack:")).map(str::to_owned),
                origin: Origin::Service,
                outcome: None,
                observed: {
                    let mut b = behaviour(diag);
                    b.extend(sites.steps());
                    if let Some(cfg) = &config {
                        b.extend(cfg.steps().unwrap_or_default());
                    }
                    b
                },
            };
            st.last = Some(call.id.clone());
            let served = Served { cited: reply.map(cited).unwrap_or_default(), ms };
            let own = !served.cited.is_empty();
            st.served.insert(call.id.clone(), served);
            st.g3.observe(rec)?;
            if own {
                st.fresh += 1;
                st.g3.verdict(&call.id, 1.0)?;
            }
            Ok::<(), nmmusc::Fault>(())
        });
        LONGEST.fetch_max(ms, Ordering::Relaxed);
    }
    let s = |t: &str| Value::String(t.to_owned());
    let mut f: Vec<(&str, Value)> =
        vec![("call", s(&call.id)), ("kind", s(&call.kind)), ("area", s(&call.area)), ("allocation", s(&call.allocation)), ("rule", s(&call.rule)), ("by", s(&call.by))];
    if let Some(c) = &call.chain {
        f.push(("chain", s(c)));
    }
    if let Some(v) = config.as_ref().and_then(|c| c.to_value().ok()) {
        f.push(("config", v));
    }
    Value::Object(f.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

/// Join an outcome (in [0, 1]) to a recorded call, in place of its own (tests and the bench, where
/// an answer key stands for the ledger).
/// # Errors
/// G3 is off, or the call is unknown.
pub fn verdict(id: &str, outcome: f64) -> Result<(), Fault> {
    with(|st| {
        st.fresh += 1;
        st.g3.verdict(id, outcome).map_err(|e| musc(&e))
    })
    .ok_or_else(|| bad("G3 is off"))?
}

/// The id of the last call recorded.
#[must_use]
pub fn last_call() -> Option<String> {
    with(|st| st.last.clone()).flatten()
}

/// Run `f` with the allocation forced to `mode` (a replay: nothing is recorded).
pub fn forced<T>(mode: &str, f: impl FnOnce() -> T) -> T {
    forced_with(&[(Site::Allocation, mode.to_owned())], f)
}

/// Run `f` on this thread with `choices` forced (a replay: nothing is recorded or written).
pub fn forced_with<T>(choices: &[(Site, String)], f: impl FnOnce() -> T) -> T {
    FORCED.with(|x| *x.borrow_mut() = Some(choices.to_vec()));
    let out = f();
    FORCED.with(|x| *x.borrow_mut() = None);
    out
}

// ---------------------------------------------------------------------------------------------
// The look arm (the L1 tuner's second-look threshold) as a replayed site.

/// The look arm's choices as G3 labels them, one per arm of the L1 tuner (`nmlearn::tuner::ARMS`),
/// written as the forage loop writes them (`0.205`, `never` for no second look).
#[must_use]
pub fn look_arm_labels() -> Vec<String> {
    nmlearn::tuner::ARMS.iter().map(|a| look_arm_label(*a)).collect()
}

/// One arm's label.
#[must_use]
pub fn look_arm_label(a: f32) -> String {
    if a.is_finite() { format!("{a}") } else { "never".to_owned() }
}

/// The choice a replay forces at `site` on this thread, if any (for a site the flow decides
/// outside `decide_site`, such as the look arm).
#[must_use]
pub fn forced_choice(site: Site) -> Option<String> {
    FORCED.with(|f| f.borrow().as_ref().and_then(|v| v.iter().find(|(s, _)| *s == site).map(|(_, c)| c.clone())))
}

/// Note the choice the flow made at `site` for the call running on this thread; it joins the
/// call's record at `end`, so G3 can replay the call with another choice there. Does nothing while
/// G3 is off and no replay runs; a choice that is not a short label is not noted.
pub fn note_choice(site: Site, choice: &str) {
    if (!is_on() && !replaying()) || choice.is_empty() || choice.len() > 16 || !choice.chars().all(|c| c.is_ascii_alphanumeric() || c == '.') {
        return;
    }
    NOTED.with(|n| {
        let mut n = n.borrow_mut();
        n.retain(|(s, _)| *s != site);
        n.push((site, choice.to_owned()));
    });
}

/// The look arm a replay forces on this thread, as the tuner's threshold.
#[must_use]
pub fn forced_look_arm() -> Option<f32> {
    let c = forced_choice(Site::LookArm)?;
    nmlearn::tuner::ARMS.iter().copied().find(|a| look_arm_label(*a) == c)
}

/// Note the look arm the call ran with (`note_choice` at `Site::LookArm`).
pub fn note_look_arm(arm: f32) {
    note_choice(Site::LookArm, &look_arm_label(arm));
}

/// The muscle store's folder while G3 is on.
#[must_use]
pub fn store_dir() -> Option<PathBuf> {
    with(|st| st.store.as_ref().map(|s| s.dir().to_path_buf())).flatten()
}

/// Reload the packs acting as priors from the muscle store (after its files were put back by a
/// learning rollback). Returns the packs loaded.
/// # Errors
/// G3 is off, or the store's index cannot be read.
pub fn reload_packs() -> Result<usize, Fault> {
    with(|st| match st.store.clone() {
        Some(store) => st.g3.reload(&store).map_err(|e| musc(&e)),
        None => Ok(0),
    })
    .ok_or_else(|| bad("G3 is off"))?
}

// ---------------------------------------------------------------------------------------------
// Serving beside the idle cycle: the gate.

/// Held by a served call for its whole length (start hook, dispatch, audit row) and by one replay
/// at a time, so the two never share the call's diagnostics and steps.
static GATE: Mutex<()> = Mutex::new(());
/// Served calls waiting at the gate; a replay is not started while one waits.
static WAITING: AtomicUsize = AtomicUsize::new(0);
/// When the last served call ended (ms).
static LAST_ACTIVE: AtomicU64 = AtomicU64::new(0);
/// The longest served call seen (ms).
static LONGEST: AtomicU64 = AtomicU64::new(0);
/// Whether the idle cycle runs (`serve_idle`); the gate is taken only then.
static IDLE: AtomicBool = AtomicBool::new(false);
/// Served calls in flight, and when the last one ended (ms), counted whether or not G3 is on, for
/// the idle learning cycle's test of quiet (`quiet_ms`).
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
static SERVED_END: AtomicU64 = AtomicU64::new(0);

/// A served call's hold on the gate (no gate when the idle cycle is not running).
pub struct Busy(Option<std::sync::MutexGuard<'static, ()>>);
impl Drop for Busy {
    fn drop(&mut self) {
        let now = nmtype::now_ms();
        if self.0.is_some() {
            LAST_ACTIVE.store(now, Ordering::Relaxed);
        }
        SERVED_END.store(now, Ordering::Relaxed);
        IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
    }
}

/// How long the serving process has been quiet (ms since the last served call ended, or since the
/// first time this is asked when no call has run); `None` while a call is in flight. Two atomics
/// per call, kept whether or not G3 is on.
#[must_use]
pub fn quiet_ms() -> Option<u64> {
    if IN_FLIGHT.load(Ordering::SeqCst) > 0 {
        return None;
    }
    let now = nmtype::now_ms();
    let end = match SERVED_END.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => now,
        Err(e) => e,
    };
    Some(now.saturating_sub(end))
}

/// Take the gate for a served call (`mcp::call`); immediate and empty unless `serve --muscle on`.
#[must_use]
pub fn busy() -> Busy {
    IN_FLIGHT.fetch_add(1, Ordering::SeqCst);
    if !IDLE.load(Ordering::Relaxed) {
        return Busy(None);
    }
    WAITING.fetch_add(1, Ordering::SeqCst);
    let g = GATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    WAITING.fetch_sub(1, Ordering::SeqCst);
    Busy(Some(g))
}

/// The scorer of a cycle: a call's id, the choices of a replay and its reply; the reply's
/// reciprocal place and the count of answer places it was scored on.
pub type Score<'a> = dyn FnMut(&str, &[(Site, String)], &str) -> Option<(f64, usize)> + 'a;

/// One replay through the served call: (reply, ms). `None` when a served call is waiting.
fn run_replay(input: &Value, choices: &[(Site, String)]) -> Option<(String, u64)> {
    let idle = IDLE.load(Ordering::Relaxed);
    if idle && WAITING.load(Ordering::SeqCst) > 0 {
        return None;
    }
    let _g = idle.then(|| GATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner));
    let api = crate::tools::api();
    let t = Instant::now();
    let out = forced_with(choices, || api.dispatch("nm_context_pack", input));
    let ms = u64::try_from(t.elapsed().as_millis()).unwrap_or(u64::MAX);
    if idle {
        // Nothing of a replay reaches a served call's audit row.
        let _ = crate::tools::take_diagnostics();
        let _ = crate::tools::take_steps();
    }
    match out {
        Ok(Value::String(s)) => Some((s, ms)),
        Ok(v) => Some((v.encode().unwrap_or_default(), ms)),
        Err(_) => None,
    }
}

/// One idle cycle of G3 (module note, step 4). Replays run through the served call; each is scored
/// by `score` with time as a cost against a replay of the call's own choices (so both are timed
/// alike). The state is taken out while replays run, so a replayed call never waits on it.
/// # Errors
/// G3 is off; store faults.
pub fn cycle(score: &mut Score<'_>) -> Result<Cycle, Fault> {
    let mut st = STATE.lock().map_err(|_| bad("muscle lock poisoned"))?.take().ok_or_else(|| bad("G3 is off"))?;
    // Roadmap Phase 4.5 (2026-11-24): the muscle's learning cycle in the live view.
    let mut live = crate::live::stage_as("learn", "learn.g3", "", "cycle");
    let out = run_cycle(&mut st, score);
    match &out {
        Ok(c) => {
            live.count("candidates", c.candidates);
            live.count("replays", c.replays);
            live.count("promoted", c.promoted.len());
            live.count("withdrawn", c.withdrawn.len());
        }
        Err(_) => live.label("failed"),
    }
    drop(live);
    st.fresh = 0;
    st.cycles += 1;
    st.last_cycle = out.as_ref().ok().cloned();
    if let Ok(mut s) = STATE.lock() {
        *s = Some(st);
    }
    out
}

/// Default wall-clock budget per G3 idle replay cycle (seconds); override via `NANO_G3_BUDGET_S`.
pub const CYCLE_BUDGET_S: f64 = 20.0;
/// Default maximum replays per G3 idle cycle; override via `NANO_G3_MAX_REPLAYS`.
pub const MAX_REPLAYS_PER_CYCLE: usize = 64;

fn cycle_budget_s() -> f64 {
    std::env::var("NANO_G3_BUDGET_S").ok().and_then(|v| v.parse::<f64>().ok()).filter(|x| x.is_finite() && *x > 0.0).unwrap_or(CYCLE_BUDGET_S)
}
fn max_replays_per_cycle() -> usize {
    std::env::var("NANO_G3_MAX_REPLAYS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|x| *x > 0).unwrap_or(MAX_REPLAYS_PER_CYCLE)
}

fn run_cycle(st: &mut MuscleState, score: &mut Score<'_>) -> Result<Cycle, Fault> {
    let inputs = st.inputs.clone();
    let records: BTreeMap<String, Vec<(Site, String)>> = st.g3.records().iter().map(|r| (r.id.clone(), r.choices.clone())).collect();
    let chain_open: BTreeSet<String> = records.iter().filter(|(_, c)| c.iter().any(|(s, _)| *s == Site::Chain)).map(|(id, _)| id.clone()).collect();
    let alts = |s: Site| match s {
        Site::Chain => CHAINS.iter().map(|x| (*x).to_owned()).collect::<Vec<String>>(),
        Site::TeamMode | Site::IntakeChain | Site::Integrations | Site::SecondLook | Site::RerankCap => nmmusc::sites::alternatives(s).iter().map(|x| (*x).to_owned()).collect(),
        Site::LookArm => look_arm_labels(),
        _ => ALLOCATIONS.iter().map(|x| (*x).to_owned()).collect(),
    };
    let started = std::time::Instant::now();
    let budget_s = cycle_budget_s();
    let max_replays = max_replays_per_cycle();
    let mut replays_done: usize = 0;
    // The time of each call's own choices, replayed as the others are.
    let mut reference: BTreeMap<String, u64> = BTreeMap::new();
    let mut replay = |id: &str, site: Site, choice: &str| -> Option<f64> {
        // Yield immediately if a live user call arrived or if the idle replay budget is spent.
        if WAITING.load(Ordering::SeqCst) > 0 || replays_done >= max_replays || started.elapsed().as_secs_f64() >= budget_s {
            return None;
        }
        if site == Site::Chain && !chain_open.contains(id) {
            return None;
        }
        // A flow site is replayed only on calls whose record took it (the flow reached it).
        if matches!(site, Site::TeamMode | Site::IntakeChain | Site::Integrations | Site::SecondLook | Site::RerankCap | Site::LookArm) && !records.get(id).is_some_and(|c| c.iter().any(|(s, _)| *s == site)) {
            return None;
        }
        let input = inputs.get(id)?;
        let own = records.get(id)?;
        if !reference.contains_key(id) {
            let (_, ms) = run_replay(input, own)?;
            reference.insert(id.to_owned(), ms);
        }
        replays_done += 1;
        let mut choices = own.clone();
        match choices.iter_mut().find(|(s, _)| *s == site) {
            Some(slot) => slot.1 = choice.to_owned(),
            None => choices.push((site, choice.to_owned())),
        }
        let (reply, ms) = run_replay(input, &choices)?;
        let (places, answers) = score(id, &choices, &reply)?;
        Some(outcome(places, answers, cited(&reply).len(), ms, reference.get(id).copied().unwrap_or(ms)))
    };
    let store = st.store.clone();
    let mut hook = |r: &Record, s: Site, m: &str| replay(&r.id, s, m);
    st.g3.cycle(store.as_ref(), &alts, &mut hook).map_err(|e| musc(&e))
}

// ---------------------------------------------------------------------------------------------
// Verdicts from the evidence ledger, and the idle cycle in `serve`.

/// Join the evidence ledger's verified `answered` outcomes to the recorded calls they judge: the
/// audit row (`audit`, the Nano's `memory/audit.jsonl`) names the call (`config.call`) under the
/// pack's audit id; the ledger (`<project memory>/evidence.jsonl`) names the pack
/// (`answered:<audit id>`). Returns the verdicts joined.
#[must_use]
pub fn ledger_verdicts(audit: &Path, memory: &Path) -> usize {
    let verified: BTreeMap<String, Vec<(String, usize, usize)>> = nmmem::outcomes::from_ledger(memory)
        .into_iter()
        .filter(|o| o.kind == "answered" && o.verified && !o.places.is_empty())
        .filter_map(|o| Some((o.id.strip_prefix("answered:")?.to_owned(), o.places.iter().map(|p| (p.path.replace('\\', "/"), p.start, p.end)).collect())))
        .collect();
    if verified.is_empty() {
        return 0;
    }
    let waiting: BTreeSet<String> = with(|st| st.served.keys().filter(|id| !st.verified.contains_key(*id)).cloned().collect()).unwrap_or_default();
    if waiting.is_empty() {
        return 0;
    }
    let audit = std::fs::read_to_string(audit).unwrap_or_default();
    let mut joined: Vec<(String, Places)> = Vec::new();
    for l in audit.lines().filter(|l| l.contains("\"config\"") && l.contains("\"state\":\"done\"")) {
        let Ok(v) = nmjson::parse(l.as_bytes()) else { continue };
        let (Ok(aid), Ok(call)) = (v.get("id").and_then(Value::string), v.get("config").and_then(|c| c.get("call")).and_then(Value::string)) else { continue };
        if waiting.contains(call)
            && let Some(places) = verified.get(aid)
        {
            joined.push((call.to_owned(), places.clone()));
        }
    }
    let mut n = 0;
    for (call, places) in joined {
        let rr = with(|st| st.served.get(&call).map(|s| reciprocal_place(&s.cited, &places))).flatten();
        if let Some(rr) = rr
            && verdict(&call, rr).is_ok()
        {
            // The call's replays were scored against its own reply; they are scored again.
            let _ = with(|st| {
                st.g3.forget_replays(&call);
                st.verified.insert(call, places)
            });
            n += 1;
        }
    }
    n
}

/// The organism's own measure of a replay's reply (module note, step 3): against the verified places
/// when Claude Code's verdict joined the call, else against the served reply's own places,
/// scaled so the served reply scores 1. `None` when the call left no places to compare.
#[must_use]
pub fn own_score(served: &Served, verified: Option<&Places>, reply: &str) -> Option<(f64, usize)> {
    let got = cited(reply);
    if let Some(v) = verified {
        return Some((reciprocal_place(&got, v), v.len()));
    }
    let full = reciprocal_place(&served.cited, &served.cited);
    (full > 0.0).then(|| ((reciprocal_place(&got, &served.cited) / full).min(1.0), served.cited.len()))
}

/// How many calls Claude Code's verdicts (the ledger) have judged.
#[must_use]
pub fn verified_calls() -> usize {
    with(|st| st.verified.len()).unwrap_or(0)
}

/// What the idle loop last looked at: the memory folder, the newer of the ledger's and the audit
/// log's times, and the calls recorded.
type Seen = (PathBuf, std::time::SystemTime, u64);

/// Start the idle cycle in the serving process (`serve --muscle on`), on the Nano's audit log; see
/// `serve_idle_on`.
#[must_use]
pub fn serve_idle() -> bool {
    serve_idle_on(None)
}

/// Start the idle cycle: when no call has run for the longest call seen (at least two seconds), new
/// ledger verdicts are joined (`ledger_verdicts` on `audit`, the Nano's `memory/audit.jsonl` when
/// `None`) and, if any, G3's cycle runs, its replays scored by the verified places. A served call
/// that arrives waits for at most the replay in progress. Returns `false` (and starts nothing)
/// when G3 is off or the cycle already runs.
#[must_use]
pub fn serve_idle_on(audit: Option<PathBuf>) -> bool {
    if !is_on() || IDLE.swap(true, Ordering::SeqCst) {
        return false;
    }
    LAST_ACTIVE.store(nmtype::now_ms(), Ordering::Relaxed);
    std::thread::spawn(move || {
        let mut seen: Option<Seen> = None;
        loop {
            std::thread::park_timeout(std::time::Duration::from_secs(1));
            if !is_on() {
                continue;
            }
            let quiet = nmtype::now_ms().saturating_sub(LAST_ACTIVE.load(Ordering::Relaxed));
            if quiet < LONGEST.load(Ordering::Relaxed).max(2000) || WAITING.load(Ordering::SeqCst) > 0 {
                continue;
            }
            let Some(o) = crate::tools::installed() else { continue };
            let Some(memory) = o.memory.lock().ok().map(|m| m.dir.clone()) else { continue };
            let ledger = memory.join(nmmem::outcomes::LEDGER);
            let audit = audit.clone().unwrap_or_else(|| o.home.join("memory").join("audit.jsonl"));
            let stamp = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
            let changed = stamp(&ledger).max(stamp(&audit)).unwrap_or(std::time::UNIX_EPOCH);
            let calls = with(|st| st.next).unwrap_or(0);
            let now = (memory.clone(), changed, calls);
            if seen.as_ref() == Some(&now) {
                continue;
            }
            seen = Some(now);
            let _ = ledger_verdicts(&audit, &memory);
            if with(|st| st.fresh).unwrap_or(0) == 0 {
                continue;
            }
            let (served, verified) = with(|st| (st.served.clone(), st.verified.clone())).unwrap_or_default();
            let mut score = |id: &str, _: &[(Site, String)], reply: &str| own_score(served.get(id)?, verified.get(id), reply);
            match cycle(&mut score) {
                Ok(cy) => eprintln!(
                    "muscle: cycle: {} replays, {} candidates, promoted {:?}, provisional {:?}, refused by gain {:?}, refused by negative selection {}, withdrawn {:?}",
                    cy.replays,
                    cy.candidates,
                    cy.promoted.iter().map(|p| p.1.clone()).collect::<Vec<_>>(),
                    cy.provisional,
                    cy.refused_gain,
                    cy.refused_self.len(),
                    cy.withdrawn
                ),
                Err(e) => eprintln!("muscle: cycle: {e}"),
            }
        }
    });
    true
}

// ---------------------------------------------------------------------------------------------
// G3's prior replay as a learner of the idle learning cycle (`nmlearn::idle`).

/// G3's counterfactual prior replay as the first learner of the idle learning cycle. Its gate is
/// G3's own: a pack is promoted only when its gain holds on held-out records (`nmmusc::g3`). Its
/// state is the muscle store's top-level files (the pack files and `index.json`), snapshotted inside
/// the store's own folder (`rollback/`), since muscle memory is its own store; a rollback puts them
/// back and reloads the packs acting as priors.
pub struct G3Learner {
    /// The Nano's audit log (`memory/audit.jsonl`) and the project memory folder, for the ledger's
    /// verified verdicts.
    pub audit: PathBuf,
    pub memory: PathBuf,
}

impl nmlearn::idle::Learner for G3Learner {
    fn name(&self) -> &str {
        "g3-priors"
    }
    fn state_paths(&self) -> Vec<PathBuf> {
        store_dir().into_iter().collect()
    }
    fn snapshot_root(&self) -> Option<PathBuf> {
        store_dir().map(|d| d.join("rollback"))
    }
    fn run(&mut self) -> Result<nmlearn::idle::LearnerReport, String> {
        if !is_on() {
            return Ok(nmlearn::idle::LearnerReport::not_run(self.name(), "not run: G3 is off in this process (serve --muscle on)"));
        }
        let joined = ledger_verdicts(&self.audit, &self.memory);
        let (served, verified) = with(|st| (st.served.clone(), st.verified.clone())).unwrap_or_default();
        let held = with(|st| st.g3.records().iter().filter(|r| r.outcome.is_some()).count()).unwrap_or(0);
        let mut score = |id: &str, _: &[(Site, String)], reply: &str| own_score(served.get(id)?, verified.get(id), reply);
        let cy = cycle(&mut score).map_err(|e| e.to_string())?;
        Ok(nmlearn::idle::LearnerReport {
            name: self.name().to_owned(),
            ran: true,
            gate: "G3 held-out gain gate (nmmusc::g3): a pack is promoted only when its gain holds on held-out records".into(),
            held_out: held,
            current: None,
            candidate: None,
            promoted: !cy.promoted.is_empty(),
            note: format!(
                "{joined} ledger verdicts joined; {} replays, {} candidates, promoted {:?}, refused by gain {}, withdrawn {}",
                cy.replays,
                cy.candidates,
                cy.promoted.iter().map(|p| p.1.clone()).collect::<Vec<_>>(),
                cy.refused_gain.len(),
                cy.withdrawn.len()
            ),
        })
    }
    fn after_restore(&mut self) -> Result<(), String> {
        if is_on() { reload_packs().map(|_| ()).map_err(|e| e.to_string()) } else { Ok(()) }
    }
}

/// Section check: with learning off `begin` touches nothing; with it on, a call is recorded with
/// the rule's choice and the mode is put back after it; time costs less than one place.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let _ = disable();
    let dir = nmtype::check_dir("nmapi-g3").map_err(|e| e.to_string())?;
    let home = nmtype::nano_root();
    let root = dir.join("tree");
    std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    std::fs::write(root.join("lib.rs"), "pub fn tide_level(hour: u64) -> u64 {\n    hour * 2\n}\n").map_err(|e| e.to_string())?;
    let o = Organism::open_with(&home, &root, &[], false, &dir.join("memory")).map_err(|e| e.to_string())?;
    let input = Value::Object([("question".to_owned(), Value::String("Where is tide_level defined?".into()))].into_iter().collect());
    let _ = note_site(nmmusc::sites::SiteKind::Mode, "g1", "-", "relation", None);
    if begin(&o, &input).is_some() || !take_sites().events.is_empty() {
        return Err("nmapi.g3: begin or a decision site acted with learning off".into());
    }
    enable(&home, Some(&dir.join("muscle"))).map_err(|e| e.to_string())?;
    let c = begin(&o, &input).ok_or("nmapi.g3: no call with learning on")?;
    note_site(nmmusc::sites::SiteKind::Order, "g2", "g0_t1", "focus", Some(true)).map_err(|e| e.to_string())?;
    note_site(nmmusc::sites::SiteKind::Chain, "g2", "-", "code-close:embedder", None).map_err(|e| e.to_string())?;
    if note_site(nmmusc::sites::SiteKind::Pose, "g1", "g0_t1", "src/lib.rs", None).is_ok() {
        let _ = disable();
        return Err("nmapi.g3: a path entered a decision site".into());
    }
    let cfg = end(&o, c, Some("@@ lib.rs:1-3\npub fn tide_level(hour: u64) -> u64 {\n    hour * 2\n}\n"), None);
    let ok = cfg.get("by").and_then(Value::string).ok() == Some("rule")
        && cfg.get("allocation").and_then(Value::string).ok() == Some("nbv")
        && cfg.get("chain").is_err()
        && o.refine.read().is_ok_and(|r| *r == "nbv")
        && with(|st| (st.g3.records().len(), st.served.values().map(|s| s.cited.clone()).collect::<Vec<_>>()))
            == Some((1, vec![vec![("lib.rs".to_owned(), 1, 3)]]))
        && with(|st| st.g3.records().first().map(|r| r.observed.iter().filter(|x| x.starts_with(nmmusc::sites::STEP)).count())) == Some(Some(2));
    if !ok {
        let _ = disable();
        return Err(format!("nmapi.g3: record {cfg:?}"));
    }
    // The chain site opens only where the stack serves and the call names no mode; a forced chain
    // changes the call's input, a named mode is never overridden.
    if let Ok(mut w) = o.whole.write() {
        *w = Some(std::sync::Arc::new(crate::whole::WholePath::new(&home)));
    }
    let rule = chain_rule(&o, &input);
    let c = begin(&o, &input).ok_or("nmapi.g3: no call with the stack")?;
    let cfg = end(&o, c, None, None);
    let named = Value::Object([("question".to_owned(), Value::String("Where is tide_level defined?".into())), ("mode".to_owned(), Value::String("full".into()))].into_iter().collect());
    let other = CHAINS.iter().find(|c| Some(**c) != rule.as_deref()).copied().unwrap_or("flow");
    let f = forced_with(&[(Site::Chain, other.to_owned())], || begin(&o, &input));
    let g = forced_with(&[(Site::Chain, other.to_owned())], || begin(&o, &named));
    let _ = disable();
    let chain_ok = rule.is_some()
        && cfg.get("chain").and_then(Value::string).ok() == rule.as_deref()
        && f.as_ref().and_then(|c| c.input.as_ref()).and_then(|i| i.get("mode").ok()).and_then(|m| m.string().ok()) == Some(other)
        && g.as_ref().is_some_and(|c| c.chain.is_none() && c.input.is_none());
    if !chain_ok {
        return Err(format!("nmapi.g3: chain site: rule {rule:?}, record {cfg:?}, forced {f:?}, named {g:?}"));
    }
    // Time is a cost below one place: equal places, faster gains; a place moved outweighs any time.
    let a = |p: usize| vec![("a.rs".to_owned(), p, p)];
    let cites: Vec<(String, usize, usize)> = (1..=5).map(|i| ("a.rs".to_owned(), i, i)).collect();
    let (first, second) = (reciprocal_place(&cites, &a(1)), reciprocal_place(&cites, &a(2)));
    let fast = outcome(second, 1, 5, 100, 1000);
    let slow = outcome(second, 1, 5, 1900, 1000);
    let worst_first = outcome(first, 1, 5, 100_000, 1000);
    let worst_second = outcome(second, 1, 5, 0, 1000);
    if !(fast > slow && worst_first > worst_second && (first - 1.0).abs() < 1e-12 && (second - 0.5).abs() < 1e-12) {
        return Err(format!("nmapi.g3: time as a cost: fast {fast}, slow {slow}, first {worst_first}, second {worst_second}"));
    }
    Ok(())
}

#[cfg(test)]
mod look_arm_site {
    use super::{NOTED, forced_look_arm, forced_with, look_arm_label, look_arm_labels, note_look_arm, quiet_ms};
    use nmmusc::priors::Site;

    /// The look arm is replayed like the other sites: its labels are the tuner's arms, a forced
    /// label reads back as its arm, and with G3 off and no replay nothing is noted.
    #[test]
    fn look_arm_labels_round_trip_and_replays_force_them() {
        let labels = look_arm_labels();
        assert_eq!(labels.len(), nmlearn::tuner::ARMS.len());
        assert!(labels.contains(&"0.205".to_owned()) && labels.contains(&"never".to_owned()));
        assert_eq!(look_arm_label(0.0), "0");
        for (label, arm) in labels.iter().zip(nmlearn::tuner::ARMS) {
            let got = forced_with(&[(Site::LookArm, label.clone())], forced_look_arm);
            assert!(got.is_some_and(|g| g.to_bits() == arm.to_bits()), "{label}");
        }
        assert_eq!(forced_look_arm(), None);
        note_look_arm(0.1);
        assert!(NOTED.with(|n| n.borrow().is_empty()), "nothing is noted with G3 off");
        // Inside a replay the choice is noted for the record.
        forced_with(&[(Site::Allocation, "nbv".to_owned())], || note_look_arm(0.35));
        assert_eq!(NOTED.with(|n| n.borrow().clone()), vec![(Site::LookArm, "0.35".to_owned())]);
    }

    #[test]
    fn quiet_counts_from_the_last_call() {
        let before = nmtype::now_ms();
        let b = super::busy();
        assert_eq!(quiet_ms(), None, "a call in flight is not quiet");
        drop(b);
        assert!(super::SERVED_END.load(std::sync::atomic::Ordering::Relaxed) >= before, "the call's end is the start of quiet");
    }
}

#[cfg(test)]
mod stage5_replay_sites {
    use super::{decide_site, forced_with};
    use nmmusc::priors::Site;

    /// Stage 5 WP-C2: G3's replays (`run_replay` runs the served call under `forced_with`) reach the
    /// rerank cap, second look and allocation sites: a forced choice is what the site returns, and
    /// without one (G3 off, no prior) each returns its deterministic rule.
    #[test]
    fn replay_forces_the_new_sites_and_the_rule_holds_otherwise() {
        let forced = forced_with(&[(Site::RerankCap, "tight".to_owned()), (Site::SecondLook, "skip".to_owned())], || {
            (decide_site(Site::RerankCap, "base", false), decide_site(Site::SecondLook, "gated", false))
        });
        assert_eq!(forced, ("tight".to_owned(), "skip".to_owned()));
        assert_eq!(decide_site(Site::RerankCap, "base", false), "base");
        assert_eq!(decide_site(Site::SecondLook, "gated", false), "gated");
        // Each new site offers its rule among the alternatives G3 replays.
        for (site, rule) in [(Site::RerankCap, "base"), (Site::SecondLook, "gated")] {
            assert!(nmmusc::sites::alternatives(site).contains(&rule), "{site:?}");
        }
    }
}
