//! Core Brain side of the served question path, first form (2026-11-10). Gavriel
//! Technologies; architecture by Luis Jake Gabriel III. Specification
//! `docs/2026-11-10-neumor-guide-how-it-works.md` sections 3 (G2, G3), 4.1, 5.1 to 5.3, 6, 7.
//!
//! Off unless `serve --brain-chains` (or `trace --brain-chains`) is given: with it off the
//! organism's path is the one it was before this module, untouched, and serves as the control
//! and fallback. With it on, after the existing reduce step:
//! 1. G2 (`nmbrain::g2::Selector`) selects the chain for the question and the parts handed up;
//!    the chain runs in the Core Brain over exact nodes (`nmbrain::chain`); its final output
//!    becomes the answer text; the chain log and hops go to the trace and the episode record
//!    only. On G2's escalate outcome the decision node forms the chain. The parts pass the
//!    sensory boundary `CHAIN_INPUT`; G2's instructions and the hop reports are behavioural
//!    `nmtype::Envelope`s on `MESSAGE_CHANNEL`.
//! 2. Every behavioural step is a symbol in the session's `nmmusc::BehaviourLog` (RAM), one
//!    sequence per question; the outcome is attached through `on_feedback`. Packs valid for the
//!    loaded base are loaded at session start; a pack whose description names the task kind is
//!    reloaded and offered to G2 as proposed control messages, and G2's decision is recorded.
//!    Mining and promotion run only on request (`mine_and_promote`).
//! 3. Learning units are presented as cartridges (`nmlearn::cartridge`); the active set is
//!    selected per request; the disagreement rule runs where two cartridge answers exist; each
//!    use is recorded in `units/cartridge-uses.jsonl`.
//! 4. Preferences: `on_feedback` feeds the preference store; recalled rules for the
//!    circumstance enter the controlled input of every writer and decision hop.
//! 5. The session holds the one broadcast `Settings` record; an edit the system applies sends
//!    an action copy to G1 and G2, whose `ActionLedger`s then tell own edits from new input.
//! 6. `trace_lines` renders each behavioural step back to text through the envelope renderer.
//! 7. Phase I1 (lane M, 2026-11-19): `intake` runs the Core Brain's evidence chain over G2's gate
//!    and per-piece evidence (`nmbrain::chain::run_evidence`) with the card's MicroModels behind
//!    `JudgeReader` and `ValueHop`; each hop is a behavioural envelope and a G3 decision site.
use crate::forage::Organism;
use nmbrain::chain::{self, ChainBook, DecisionFormer, Former, HopRecord, Mode, NodeMap, Run};
use nmbrain::g2::{BuiltinCases, Outcome, Selector};
use nmjson::Value;
use nmlearn::cartridge as cart;
use nmlearn::preference as pref;
use nmmusc::{BehaviourLog, BehaviouralSymbol, GainEstimator, MuscleStore, ReplayLayer};
use nmtype::class::boundaries as bnd;
use nmtype::{
    ActionLedger, Classed, DataClass, Envelope, Fault, FaultKind, MessageKind, Pack, Packet, Reference, Resolver, SettingValue, Settings,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// The served option.
pub const FLAG: &str = "--brain-chains";
/// Unit ids used as senders and targets of the in-house messages.
pub const G1: &str = "nmbrain.g1#0";
pub const G2: &str = "nmbrain.g2#0";
pub const G3: &str = "nmbrain.g3#0";
/// The confidence gain G2 sets on the reflexes when it is unsure, and for how long.
pub const GAIN_UNSURE: f32 = 0.75;
pub const GAIN_TTL_MS: u64 = 60_000;
pub const CHAIN: &str = "nmbrain.chain#0";
pub const SESSION: &str = "nmapi.session#0";
/// The base population whose hash packs and cartridges must match.
pub const BASE: &str = "qwen3-0.6b";
/// The seed field that carries recalled preference rules into a hop's controlled input.
pub const PREF_FIELD: &str = "preferences";
/// Roles whose hop makes a choice of style or approach (they receive the recalled rules).
pub const CHOICE_ROLES: [&str; 2] = ["writer", "decision"];
/// The cartridge track-record file, relative to the mini root (P9 note, "Not done").
pub const USES_FILE: &str = "units/cartridge-uses.jsonl";
/// The chain-log file beside the episodes in the session's memory folder (memory on only).
pub const CHAIN_LOG_FILE: &str = "chains.jsonl";

fn bad(e: impl std::fmt::Display) -> Fault {
    Fault::new("nmapi.brain_side", FaultKind::Config, e.to_string())
}
fn musc(e: &nmmusc::Fault) -> Fault {
    let kind = if e.code == "nmmusc-sensory-refused" { FaultKind::Policy } else { FaultKind::Config };
    Fault::new("nmapi.brain_side", kind, e.to_string())
}
fn s(t: &str) -> Value {
    Value::String(t.to_owned())
}
fn n(x: usize) -> Value {
    Value::Number(x.to_string())
}
fn f(x: f64) -> Value {
    Value::Number(format!("{x:.4}"))
}
fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

/// Where the session keeps what it writes. `live` is the served layout; checks pass folders
/// under `tests/section-checks/`.
#[derive(Clone, Debug)]
pub struct Config {
    /// Muscle-memory store folder (`None`: the live store, `designs/muscle.json`).
    pub muscle_dir: Option<PathBuf>,
    /// Home of the preference store (`<home>/preferences/records.jsonl`).
    pub pref_home: PathBuf,
    /// Cartridge track-record file.
    pub uses_path: PathBuf,
    /// First-form ordering (G2 and the chain's ranking nodes reorder freely, no order
    /// contract); off in service, kept so the comparison is repeatable.
    pub first_form_order: bool,
}
impl Config {
    #[must_use]
    pub fn live(home: &Path) -> Self {
        Self { muscle_dir: None, pref_home: home.to_path_buf(), uses_path: home.join(USES_FILE), first_form_order: false }
    }
}

/// Whether a muscle-memory pack executes as a prior (2026-11-12, L3 of continuous learning;
/// analysis E 12.4, research file Solution 6 L3): only when G2 is unsure (it escalated), no chain
/// was forced, and a loaded pack for the task kind had steps G2 accepted. Then the pack's order
/// (run the chain of that kind) is carried out in place of forming a new chain; when G2 is sure,
/// its own choice stands (G2's veto). Withdrawn packs are never loaded, so only packs whose gain
/// held can act.
#[must_use]
pub fn pack_executes(escalated: bool, forced: bool, offers: &[Offer]) -> bool {
    escalated && !forced && offers.iter().any(|o| o.accepted > 0)
}

/// One muscle-memory pack offered to G2 for a question, and what G2 did with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Offer {
    pub pack: String,
    pub description: String,
    pub proposed: usize,
    pub accepted: usize,
    pub declined: usize,
}

/// What the Core Brain side did for one question (trace and episode record; never the answer).
#[derive(Clone, Debug)]
pub struct BrainRecord {
    pub question_id: String,
    /// The chain that ran.
    pub kind: String,
    /// Formation mode: fixed (G2 chose) or decision (G2 escalated).
    pub mode: &'static str,
    pub escalated: bool,
    /// G2's calibrated confidence and the derived threshold.
    pub confidence: f32,
    pub threshold: f32,
    /// G2's first choice (the best guess when it escalated).
    pub g2_choice: String,
    pub upstream: usize,
    pub held_back: usize,
    pub final_output: Vec<String>,
    /// The parts the final output rests on (`chain::Run::citations`); the answer text carries
    /// those the output does not already hold.
    pub citations: Vec<String>,
    /// True when the final output became the answer text (not withheld, output not empty).
    pub answer_replaced: bool,
    /// The decision node's own hops (escalation) followed by the chain's hops.
    pub log: Vec<HopRecord>,
    /// Every behavioural step of the question, in order (G2's instruction, pack offers and
    /// G2's decisions, hop reports, the settings edit).
    pub steps: Vec<Envelope>,
    /// Symbols this question added to the behaviour log.
    pub symbols: usize,
    pub offers: Vec<Offer>,
    /// Cartridge selection (`nmlearn::cartridge::Selection::to_value`).
    pub cartridges: Value,
    /// The disagreement rule's resolution, when two cartridge answers existed.
    pub resolution: Option<Value>,
    /// Recalled preference rules given to the writer and decision hops.
    pub preferences: Vec<String>,
    pub settings_version: u64,
    /// Milliseconds: whole side, G2 selection, chain.
    pub ms: f64,
    pub g2_ms: f64,
    pub chain_ms: f64,
}

/// Session counters (first measurements).
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub questions: usize,
    pub escalations: usize,
    pub ms_total: f64,
    pub by_kind: BTreeMap<String, usize>,
    pub own_edits_recognised: usize,
    pub offers: usize,
    pub accepted_steps: usize,
    /// Packs executed as priors (2026-11-12).
    pub executed: usize,
}

/// Facts about one answered question kept for the feedback path.
#[derive(Clone, Debug)]
struct Asked {
    question_id: String,
    question: String,
    answer: String,
    kind: String,
    file_kind: String,
}

/// RAM state of the session.
struct Session {
    log: BehaviourLog,
    settings: Settings,
    g1: ActionLedger,
    g2: ActionLedger,
    stats: Stats,
    kinds: BTreeMap<String, String>,
    asked: BTreeMap<String, Asked>,
    episodes: BTreeMap<String, String>,
    prefs: pref::Prefs,
    pref_index: pref::Index,
    uses: Vec<(String, bool)>,
    open_uses: BTreeMap<String, Vec<String>>,
    next_question: u64,
}

/// A behavioural item as a G3 symbol: kind, target and a stable action name, never content.
/// The class comes from the item's own tag, so a sensory item planted here is refused.
pub struct Symbol<'a, T: Classed + ?Sized> {
    pub item: &'a T,
    pub kind: nmmusc::MessageKind,
    pub target: String,
    pub action: String,
}
impl<T: Classed + ?Sized> BehaviouralSymbol for Symbol<'_, T> {
    fn data_class(&self) -> nmmusc::DataClass {
        match self.item.data_class() {
            DataClass::Behavioural => nmmusc::DataClass::Behavioural,
            DataClass::Sensory => nmmusc::DataClass::Sensory,
        }
    }
    fn kind(&self) -> nmmusc::MessageKind {
        self.kind
    }
    fn descriptor(&self) -> String {
        format!("{} {} {}", self.kind.as_str(), self.target, self.action)
    }
}

/// Resolver over the studied substrate: spans read from the files as held; token ids decode
/// byte-level (first-form stand-in for the base tokenizer, as in the P5 checks).
pub struct SubResolver<'a>(pub &'a nmsubs::Substrate);
impl Resolver for SubResolver<'_> {
    fn span(&self, path: &str, start: u32, end: u32) -> Result<Option<String>, Fault> {
        let Some(file) = self.0.files.iter().find(|x| x.path == path) else { return Ok(None) };
        let (a, b) = (usize::try_from(start).unwrap_or(1).max(1), usize::try_from(end).unwrap_or(0));
        let lines: Vec<&str> = file.text.lines().skip(a - 1).take(b.saturating_sub(a) + 1).collect();
        Ok(if lines.is_empty() { None } else { Some(lines.join("\n")) })
    }
    fn tokens(&self, ids: &[u32]) -> Result<Option<String>, Fault> {
        Ok(ids.iter().map(|i| u8::try_from(*i).ok()).collect::<Option<Vec<u8>>>().and_then(|b| String::from_utf8(b).ok()))
    }
}

/// Byte-level label tokens (stand-in until the base tokenizer is passed in).
#[must_use]
pub fn label_tokens(text: &str) -> Vec<u32> {
    nmtype::message::TableResolver::encode(text)
}

/// A part handed up, as one line: its first evidence line (`path:line text`), else key and text.
#[must_use]
pub fn part_line(p: &Packet) -> String {
    p.get_evidence().first().cloned().unwrap_or_else(|| format!("{} {}", p.key(), p.text()))
}

/// Span references of lines of the form `path:line text` (or `path:a-b text`).
#[must_use]
pub fn spans_of(lines: &[String]) -> Vec<Reference> {
    let mut out: Vec<Reference> = Vec::new();
    for l in lines {
        let loc = l.split_once(' ').map_or(l.as_str(), |(a, _)| a);
        let Some((path, range)) = loc.rsplit_once(':') else { continue };
        let (a, b) = range.split_once('-').unwrap_or((range, range));
        let (Ok(start), Ok(end)) = (a.parse::<u32>(), b.parse::<u32>()) else { continue };
        if path.is_empty() || start == 0 || start > end {
            continue;
        }
        let r = Reference::Span { path: path.to_owned(), start, end };
        if !out.contains(&r) {
            out.push(r);
        }
    }
    out
}

/// The chain book with the preference seed added to every choice-making hop.
#[must_use]
pub fn with_preferences(book: &ChainBook) -> ChainBook {
    let mut b = ChainBook { seeds: book.seeds.clone(), roles: book.roles.clone(), chains: book.chains.clone() };
    if !b.seeds.iter().any(|x| x == PREF_FIELD) {
        b.seeds.push(PREF_FIELD.to_owned());
    }
    for c in &mut b.chains {
        for node in &mut c.nodes {
            if CHOICE_ROLES.contains(&node.role.as_str()) && !node.takes.iter().any(|t| t == PREF_FIELD) {
                // Appended last: the exact nodes read their inputs by position from the front
                // (writer) or by name (decision), so the rules never displace a part.
                node.takes.push(PREF_FIELD.to_owned());
            }
        }
    }
    b
}

/// The same packet with another text (id recomputed as `Packet::new` would).
/// # Errors
/// The packet's JSON form did not read back.
pub fn with_text(p: &Packet, text: &str) -> Result<Packet, Fault> {
    let id = Packet::new(p.kind(), p.key(), text, p.provenance().clone()).to_value().get("id").cloned().map_err(bad)?;
    let mut v = p.to_value();
    if let Value::Object(m) = &mut v {
        m.insert("text".into(), s(text));
        m.insert("id".into(), id);
    }
    Packet::from_value(&v)
}

/// The answer text: the head line, the chain's final output, then each cited part the output
/// does not already hold.
#[must_use]
pub fn answer_text(head: &str, output: &[String], citations: &[String]) -> String {
    let mut lines: Vec<&str> = vec![head];
    lines.extend(output.iter().map(String::as_str));
    lines.extend(citations.iter().filter(|c| !output.contains(c)).map(String::as_str));
    lines.join("\n")
}

fn file_kind(parts: &[String]) -> String {
    parts
        .first()
        .and_then(|l| l.split_once(' ').map_or(Some(l.as_str()), |(a, _)| Some(a)))
        .and_then(|loc| loc.split(':').next())
        .and_then(|p| p.rsplit_once('.'))
        .map(|(_, e)| e.to_owned())
        .unwrap_or_default()
}
fn show(v: &SettingValue) -> String {
    match v {
        SettingValue::Flag(b) => b.to_string(),
        SettingValue::Count(c) => c.to_string(),
        SettingValue::Level(x) => format!("{x}"),
        SettingValue::Word(w) => w.clone(),
    }
}
fn short(fp: &str) -> &str {
    fp.get(..12).unwrap_or(fp)
}

/// The Core Brain side of one served session.
pub struct BrainSide {
    pub home: PathBuf,
    /// Where the behavioural chain log is written (2026-11-12, direction of Luis Jake Gabriel III:
    /// memory is a different thing from motor data): the learning store `units/` (learning data
    /// with outcomes, like the decision ledger), or beside a test session's muscle folder; not the
    /// memory folder and not the muscle-memory store, which holds packs only.
    pub chain_log: PathBuf,
    pub book: ChainBook,
    book_pref: ChainBook,
    pub selector: Selector,
    nodes: NodeMap,
    store: Option<MuscleStore>,
    /// Packs loaded at session start: (fingerprint, description).
    pub packs: Vec<(String, String)>,
    pub base_hash: String,
    pub cart_policy: cart::Policy,
    pub cartridges: Vec<cart::Label>,
    pub loaded: cart::Loaded,
    pub repo: String,
    pub cfg: Config,
    state: Mutex<Session>,
    /// One line on what was loaded (for `serve` and `trace`).
    pub note: String,
}

impl BrainSide {
    /// Session start: chain book, G2 fitted on its cases, exact nodes, packs valid for the
    /// loaded base, cartridges, preference store, the one settings record.
    /// # Errors
    /// The chain book or G2 could not be built, or the settings record was refused.
    pub fn open(org: &Organism, cfg: Config) -> Result<Self, Fault> {
        let home = org.home.clone();
        let book = ChainBook::load(&home.join("designs").join("chains.json"))?;
        let book_pref = with_preferences(&book);
        book_pref.chains.iter().try_for_each(|c| book_pref.validate(c))?;
        // Stage 5 WP-C2: G2's past cases are the built-in set plus the verified closed calls G3 holds.
        let cases = RecordCases::from_records(&crate::g3::with(|st| st.g3.records().to_vec()).unwrap_or_default(), &book.kinds());
        let mut selector = Selector::fit(Arc::clone(&org.slots), &cases)?;
        selector.first_form_order = cfg.first_form_order;
        let mut nodes = NodeMap::exact(&org.slots, &book.roles);
        nodes.first_form_order = cfg.first_form_order;
        let components: Vec<String> = org.slots.by_name.values().filter(|x| x.engine_bound()).map(|x| format!("{}*", x.name)).collect();
        let loaded = cart::Loaded::population(&home, BASE, components.clone()).unwrap_or(cart::Loaded {
            base: BASE.into(),
            base_sha256: String::new(),
            components,
        });
        let base_hash = loaded.base_sha256.clone();
        let cart_policy = cart::Policy::load(&home);
        let cartridges = cart::units_as_cartridges(&nmlearn::Units::open(&home), &cart_policy);
        let store = match &cfg.muscle_dir {
            Some(d) => MuscleStore::open(d),
            None => MuscleStore::live(),
        }
        .ok();
        let packs: Vec<(String, String)> = store
            .as_ref()
            .and_then(|st| st.index().ok().map(|rows| (st, rows)))
            .map(|(st, rows)| {
                rows.into_iter()
                    .filter(|r| !r.withdrawn && r.base_hash == base_hash && st.load(&r.fingerprint, &base_hash).is_ok())
                    .map(|r| (r.fingerprint, r.description))
                    .collect()
            })
            .unwrap_or_default();
        let prefs = pref::Prefs::open(&cfg.pref_home).unwrap_or_default();
        let pref_index = pref::Index::build(&prefs, &pref::ExactEmbedder);
        let uses = read_uses(&cfg.uses_path);
        let repo = org.sub().root.file_name().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default();
        let refine = org.refine.read().map(|r| r.clone()).unwrap_or_default();
        let settings = Settings::default()
            .set("comprehension_first", SettingValue::Flag(true))
            .set("tentacles", SettingValue::Count(org.tentacles as u64))
            .set("refine", SettingValue::Word(refine))
            .set("mode", SettingValue::Word(if crate::tools::installed().is_some() { "extension".into() } else { "standalone".into() }))
            .set("brain_chains", SettingValue::Flag(true));
        bnd::BROADCAST.admit(&settings)?;
        let note = format!(
            "brain chains on: {} chains, G2 k {} threshold {:.3} on {} built-in cases, {} packs valid for {BASE}, {} cartridges ({} active-capable), {} preference records, settings v{}",
            book.chains.len(),
            selector.k,
            selector.threshold,
            BuiltinCases.cases_len(),
            packs.len(),
            cartridges.len(),
            cartridges.iter().filter(|l| cart::slot_check(l, &loaded).is_ok()).count(),
            prefs.records.len(),
            settings.version()
        );
        let chain_log = match &cfg.muscle_dir {
            Some(d) => d.with_file_name(format!("{}-{CHAIN_LOG_FILE}", d.file_name().and_then(|n| n.to_str()).unwrap_or("session"))),
            None => home.join("units").join(CHAIN_LOG_FILE),
        };
        Ok(Self {
            chain_log,
            home,
            book,
            book_pref,
            selector,
            nodes,
            store,
            packs,
            base_hash,
            cart_policy,
            cartridges,
            loaded,
            repo,
            cfg,
            state: Mutex::new(Session {
                log: BehaviourLog::default(),
                settings,
                g1: ActionLedger::default(),
                g2: ActionLedger::default(),
                stats: Stats::default(),
                kinds: BTreeMap::new(),
                asked: BTreeMap::new(),
                episodes: BTreeMap::new(),
                prefs,
                pref_index,
                uses,
                open_uses: BTreeMap::new(),
                next_question: 0,
            }),
            note,
        })
    }

    fn envelope(kind: MessageKind, from: &str, to: &str, confidence: f32, desc: &str, refs: Vec<Reference>) -> Result<Envelope, Fault> {
        let env = Envelope::new(kind, from, to, confidence, desc, refs, label_tokens(desc))?;
        bnd::MESSAGE_CHANNEL.admit(&env)?;
        Ok(env)
    }

    /// Record one behavioural item in the behaviour log under question `qid`. The item must
    /// pass the behavioural message boundary first; the codebook refuses sensory items too.
    /// # Errors
    /// `nm-class-mismatch` at the boundary, or `nmmusc-sensory-refused` in the codebook.
    pub fn record<T: Classed + ?Sized>(
        &self,
        qid: &str,
        item: &T,
        kind: nmmusc::MessageKind,
        target: &str,
        action: &str,
    ) -> Result<u32, Fault> {
        let mut st = self.state.lock().map_err(|_| bad("brain side lock"))?;
        Self::record_in(&mut st, qid, item, kind, target, action)
    }
    fn record_in<T: Classed + ?Sized>(
        st: &mut Session,
        qid: &str,
        item: &T,
        kind: nmmusc::MessageKind,
        target: &str,
        action: &str,
    ) -> Result<u32, Fault> {
        bnd::MESSAGE_CHANNEL.admit(item)?;
        st.log.record(qid, &Symbol { item, kind, target: target.to_owned(), action: action.to_owned() }).map_err(|e| musc(&e))
    }

    /// An edit the system applies to the broadcast settings: a new record, the instruction
    /// that made it, and its action copy to G1 and G2 (recorded in both ledgers).
    fn apply_setting(st: &mut Session, key: &str, value: SettingValue) -> Result<Envelope, Fault> {
        let next = st.settings.set(key, value.clone());
        bnd::BROADCAST.admit(&next)?;
        let env = Self::envelope(
            MessageKind::Instruction,
            SESSION,
            "broadcast",
            1.0,
            &format!("set {key} to {} (settings v{})", show(&value), next.version()),
            Vec::new(),
        )?;
        for (observer, ledger) in [(G1, &mut st.g1), (G2, &mut st.g2)] {
            let copy = env.action_copy(observer);
            bnd::ACTION_COPIES.admit(&copy)?;
            ledger.record(&copy)?;
        }
        if st.g1.is_own(&env) && st.g2.is_own(&env) {
            st.stats.own_edits_recognised += 1;
        }
        st.settings = next;
        Ok(env)
    }

    /// Whether G1 and G2 both recognise `msg` as the system's own action.
    #[must_use]
    pub fn is_own(&self, msg: &Envelope) -> bool {
        self.state.lock().is_ok_and(|st| st.g1.is_own(msg) && st.g2.is_own(msg))
    }
    /// Whether a changed line is one the system itself acted on (G1's ledger). Where it will
    /// matter: `Organism::sync` and the G1 loop, once applied patches send action copies.
    #[must_use]
    pub fn is_self_caused(&self, path: &str, line: u32) -> bool {
        self.state.lock().is_ok_and(|st| st.g1.is_self_caused(path, line))
    }
    /// An action the system applied to source lines (a patch the coding agent applied from a
    /// proposal): its action copy goes to G1 and G2.
    /// # Errors
    /// An invalid span or the ledger refused the copy.
    pub fn applied_edit(&self, desc: &str, spans: Vec<Reference>) -> Result<Envelope, Fault> {
        let env = Self::envelope(MessageKind::Instruction, SESSION, "substrate", 1.0, desc, spans)?;
        let mut guard = self.state.lock().map_err(|_| bad("brain side lock"))?;
        let st = &mut *guard;
        for (observer, ledger) in [(G1, &mut st.g1), (G2, &mut st.g2)] {
            let copy = env.action_copy(observer);
            bnd::ACTION_COPIES.admit(&copy)?;
            ledger.record(&copy)?;
        }
        Ok(env)
    }
    /// The current broadcast settings record.
    #[must_use]
    pub fn settings(&self) -> Settings {
        self.state.lock().map(|st| st.settings.clone()).unwrap_or_default()
    }
    /// Session counters.
    #[must_use]
    pub fn stats(&self) -> Stats {
        self.state.lock().map(|st| st.stats.clone()).unwrap_or_default()
    }
    /// The behaviour log's traces: (question id, symbol count, outcome).
    #[must_use]
    pub fn traces(&self) -> Vec<(String, usize, Option<f64>)> {
        self.state.lock().map(|st| st.log.traces().iter().map(|t| (t.id.clone(), t.symbols.len(), t.outcome)).collect()).unwrap_or_default()
    }

    /// Recalled preference rules for the circumstance (empty when the store has no records).
    fn recall_rules(st: &Session, kind: &str, file_kind: &str, question: &str) -> Vec<String> {
        if st.prefs.records.is_empty() {
            return Vec::new();
        }
        let c = pref::Circumstance::new(kind, "writer", file_kind, question);
        pref::recall(&st.pref_index, &c, &pref::ExactEmbedder, None, None)
            .into_iter()
            .filter_map(|(i, sim)| {
                st.prefs.records.get(i).map(|r| {
                    format!(
                        "when {} (similarity {sim:.2}): prefer {}; avoid {}",
                        r.rule.when,
                        r.rule.prefer.join(", "),
                        r.rule.avoid.join(", ")
                    )
                })
            })
            .collect()
    }

    /// The decision node (2026-11-10, first form, `crate::decide`): over the union of the
    /// tentacles' shortlists, the released reranker scores every candidate, then a tournament of
    /// full-depth joint reads with the base weights picks the winner. `None` while the step is
    /// off (`--brain-decide` not given). Weights are the central copies, used by reference.
    /// # Errors
    /// The models cannot be opened, or an engine fault.
    pub fn decide(
        &self,
        question: &str,
        candidates: &[crate::joint_read::Piece],
        budget_s: Option<f64>,
    ) -> Result<Option<crate::decide::Decision>, Fault> {
        if !crate::decide::enabled() {
            return Ok(None);
        }
        let base = crate::neural::model(&self.home)?;
        let rr = crate::neural::reranker_model(&self.home)?;
        let set = crate::tentacle_pass::drones(&self.home);
        Ok(Some(crate::decide::decide(&base, &rr, question, candidates, &set.heads, set.last_layer(), budget_s)?))
    }

    /// Run one listed chain with the preference seed.
    /// # Errors
    /// An unknown chain, a validation fault or a node failure.
    pub fn run_chain(&self, kind: &str, task: &str, parts: &[String], rules: &[String], mode: Mode) -> Result<Run, Fault> {
        let def = self.book_pref.get(kind).ok_or_else(|| bad(format!("no chain {kind}")))?;
        self.book_pref.validate(def)?;
        Ok(chain::run_with(
            def,
            task,
            vec![("task".into(), vec![task.to_owned()]), ("parts".into(), parts.to_vec()), (PREF_FIELD.into(), rules.to_vec())],
            &self.nodes,
            mode,
        )?)
    }

    /// Offer packs whose description names the task kind to G2, as proposed control messages
    /// through the reload path; G2 accepts a step that names the chain or one of its roles and
    /// declines the rest. Accepted steps are recorded, not executed (no receiving part takes
    /// pack control yet).
    fn offer(&self, st: &mut Session, qid: &str, kind: &str, steps: &mut Vec<Envelope>) -> Result<Vec<Offer>, Fault> {
        let Some(store) = self.store.as_ref() else { return Ok(Vec::new()) };
        let roles: Vec<String> = self.book.get(kind).map(nmbrain::chain::ChainDef::roles).unwrap_or_default();
        let mut offers = Vec::new();
        for (fp, desc) in self.packs.iter().filter(|(_, d)| d.split(|c: char| !(c.is_alphanumeric() || c == '-')).any(|w| w == kind)) {
            let msgs = match nmmusc::reload(store, fp, &self.base_hash, &mut ReplayLayer) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("brain chains: pack {} not reloaded: {e}", short(fp));
                    continue;
                }
            };
            let (mut accepted, mut declined) = (0, 0);
            for m in &msgs {
                let fits = m.descriptor.contains(kind) || roles.iter().any(|r| m.descriptor.contains(&format!("hop:{r}:")));
                let proposal = Self::envelope(
                    MessageKind::Instruction,
                    G3,
                    G2,
                    0.5,
                    &format!("proposed by pack {} step {}: {}", short(fp), m.step, m.descriptor),
                    Vec::new(),
                )?;
                steps.push(proposal);
                let verdict = if fits && m.proposed { "accept" } else { "decline" };
                if verdict == "accept" {
                    accepted += 1;
                } else {
                    declined += 1;
                }
                steps.push(Self::envelope(
                    MessageKind::Report,
                    G2,
                    G3,
                    1.0,
                    &format!("{verdict} pack {} step {}", short(fp), m.step),
                    Vec::new(),
                )?);
            }
            let decision = if accepted > 0 { "accept" } else { "decline" };
            let env = Self::envelope(
                MessageKind::Report,
                G2,
                G3,
                1.0,
                &format!("{decision} pack {}: {accepted} of {} steps fit chain {kind}", short(fp), msgs.len()),
                Vec::new(),
            )?;
            Self::record_in(st, qid, &env, nmmusc::MessageKind::Route, G3, &format!("pack:{}:{decision}", short(fp)))?;
            steps.push(env);
            st.stats.offers += 1;
            st.stats.accepted_steps += accepted;
            offers.push(Offer { pack: fp.clone(), description: desc.clone(), proposed: msgs.len(), accepted, declined });
        }
        Ok(offers)
    }

    /// Cartridge answers over the same candidates. First form: no runner for unit answers in
    /// the served path (no unit is active on the processor), so none.
    fn cartridge_answers(_: &cart::Selection) -> Vec<cart::Answer> {
        Vec::new()
    }

    /// The Core Brain side for one question, after the existing reduce and emit. Returns the
    /// answer packet (the chain's final output as text, or `emitted` unchanged on any fault)
    /// and the record.
    pub fn after_reduce(&self, org: &Organism, question: &str, pack: &Pack, emitted: Packet) -> (Packet, Option<BrainRecord>) {
        match self.question(org, question, pack, &emitted, None) {
            Ok((p, r)) => (p, Some(r)),
            Err(e) => {
                eprintln!("brain chains: fallback to the existing answer: {e}");
                (emitted, None)
            }
        }
    }

    /// One question; `forced` names the chain to run instead of G2's choice (checks of every
    /// task kind), G2 still selects the parts and records its own choice.
    /// # Errors
    /// A boundary refused an item, or the chain could not run.
    #[expect(clippy::too_many_lines, reason = "one question's steps in order, each short")]
    pub fn question(
        &self,
        org: &Organism,
        question: &str,
        pack: &Pack,
        emitted: &Packet,
        forced: Option<&str>,
    ) -> Result<(Packet, BrainRecord), Fault> {
        let t0 = Instant::now();
        // Step 1: the parts are sensory; nothing behavioural enters the chain's input.
        bnd::CHAIN_INPUT.admit_all(&pack.packets)?;
        let parts: Vec<String> = pack.packets.iter().map(part_line).collect();
        // Refinement step 1 (item 2): on the tentacle core path G1's hand-over reaches G2 as its
        // evidence set and G2 selects on it (no part dropped by word overlap; NEXT when short).
        let stack = org.stack.lock().ok().and_then(|t| t.as_ref().filter(|(q, _)| q == question).map(|(_, e)| Arc::clone(e)));
        let sel = match stack.as_ref().map(|e| e.lock()) {
            Some(Ok(mut set)) => {
                bnd::CHAIN_INPUT.admit(&*set)?;
                self.selector.select_evidence(question, &parts, &self.book, &mut set)
            }
            _ => self.selector.select(question, &parts, &self.book),
        };
        let (g2_choice, escalated, confidence, threshold) = match &sel.outcome {
            Outcome::Chosen { kind, confidence, .. } => (kind.clone(), false, *confidence, self.selector.threshold),
            Outcome::Escalate { best_guess, confidence, threshold } => (best_guess.clone(), true, *confidence, *threshold),
        };
        let mut st = self.state.lock().map_err(|_| bad("brain side lock"))?;
        // One behaviour sequence per question: the session numbers its questions (the pack's
        // task number is the same for every question of the reduce step).
        st.next_question += 1;
        let qid = format!("q{}", st.next_question);
        let mut steps: Vec<Envelope> = Vec::new();
        let up = sel.upstream.len();
        let (desc, kind_msg, target) = if escalated {
            (
                format!(
                    "escalate: confidence {confidence:.3} below {threshold:.3}, best guess {g2_choice}; the decision node forms the chain over {up} parts"
                ),
                MessageKind::Request,
                "nmbrain.chain#decision".to_owned(),
            )
        } else {
            (
                format!("run chain {g2_choice} at confidence {confidence:.3} over {up} parts, {} held back", sel.held_back),
                MessageKind::Instruction,
                format!("nmbrain.chain#{g2_choice}"),
            )
        };
        let g2_env = Self::envelope(kind_msg, G2, &target, confidence, &desc, spans_of(&sel.upstream))?;
        Self::record_in(
            &mut st,
            &qid,
            &g2_env,
            nmmusc::MessageKind::Route,
            &target,
            &format!("select:{}", if escalated { "escalate" } else { g2_choice.as_str() }),
        )?;
        steps.push(g2_env);
        // 2026-11-12 (motor, gains from above): when G2 is unsure it lowers the reflexes'
        // confidence bounds for one minute (gain 0.75), so the units drop fewer weak findings and
        // the next reads bring in more evidence; the gain expires by itself.
        if escalated {
            org.reflexes.set_gain(nmpod::pipeline::CONFIDENCE_GAIN, GAIN_UNSURE, G2, GAIN_TTL_MS);
            let env = Self::envelope(
                MessageKind::Reflex,
                G2,
                "nmpod.unit#*",
                1.0,
                &format!(
                    "gain {} to {GAIN_UNSURE} for {} s (G2 unsure at {confidence:.3})",
                    nmpod::pipeline::CONFIDENCE_GAIN,
                    GAIN_TTL_MS / 1000
                ),
                Vec::new(),
            )?;
            Self::record_in(&mut st, &qid, &env, nmmusc::MessageKind::Reflex, "nmpod.unit#*", "gain:confidence:widen")?;
            steps.push(env);
        }
        // Step 2: packs for this task kind, offered through the reload path.
        let offers = self.offer(&mut st, &qid, forced.unwrap_or(&g2_choice), &mut steps)?;
        // Step 3: cartridges for this request.
        let scope = format!("reranker:{}:{}", forced.unwrap_or(&g2_choice), self.repo);
        let req = cart::Request { scope: scope.clone(), data_class: self.cart_policy.unit_class };
        let tracks = cart::tracks_from(&st.uses);
        let csel = cart::select(&req, &self.cartridges, &self.loaded, &tracks, &self.cart_policy);
        let answers = Self::cartridge_answers(&csel);
        let resolution = match (answers.first(), answers.get(1)) {
            (Some(a), Some(b)) => {
                let bar = cart::threshold(&nmlearn::outcomes::since(&self.home, 0), "reranker", &self.cart_policy);
                Some(cart::resolve(a, b, &tracks, &bar, &self.cart_policy).to_value())
            }
            _ => None,
        };
        for a in &csel.active {
            let line = obj(vec![
                ("at", Value::Number(nmtype::now_ms().to_string())),
                ("cartridge", s(&a.id)),
                ("scope", s(&scope)),
                ("question", s(&qid)),
                ("right", Value::Null),
                ("class", s("behavioural")),
            ]);
            append_line(&self.cfg.uses_path, &line);
            st.open_uses.entry(qid.clone()).or_default().push(a.id.clone());
        }
        // Step 4: recalled preference rules for the choice-making hops.
        let fk = file_kind(&sel.upstream);
        let rules = Self::recall_rules(&st, forced.unwrap_or(&g2_choice), &fk, question);
        // The chain in the Core Brain over exact nodes.
        let tc = Instant::now();
        let executed = pack_executes(escalated, forced.is_some(), &offers);
        if executed {
            let env = Self::envelope(
                MessageKind::Instruction,
                G3,
                &format!("nmbrain.chain#{g2_choice}"),
                0.5,
                &format!("pack prior executed: run chain {g2_choice} (G2 unsure at {confidence:.3})"),
                Vec::new(),
            )?;
            Self::record_in(&mut st, &qid, &env, nmmusc::MessageKind::Route, G3, &format!("execute:{g2_choice}"))?;
            steps.push(env);
            st.stats.executed += 1;
        }
        let (run, mut log) = match forced {
            Some(k) => (self.run_chain(k, question, &sel.upstream, &rules, Mode::Fixed)?, Vec::new()),
            None if executed => (self.run_chain(&g2_choice, question, &sel.upstream, &rules, Mode::Fixed)?, Vec::new()),
            None if escalated => {
                let formed = DecisionFormer.form(question, &sel.upstream, &self.book, &self.nodes)?;
                (self.run_chain(&formed.chain.kind, question, &sel.upstream, &rules, Mode::Decided)?, formed.log)
            }
            None => (self.run_chain(&g2_choice, question, &sel.upstream, &rules, Mode::Fixed)?, Vec::new()),
        };
        let chain_ms = tc.elapsed().as_secs_f64() * 1000.0;
        log.extend(run.log.iter().cloned());
        for h in &log {
            let env = Self::envelope(
                MessageKind::Report,
                &format!("nmbrain.chain#{}", h.role),
                CHAIN,
                1.0,
                &format!(
                    "hop {} {} ({}) took {} and emitted {}: {} lines in {:.3} ms, order {}",
                    h.index,
                    h.role,
                    h.label,
                    h.takes.join("+"),
                    h.emits,
                    h.output.len(),
                    h.ms,
                    h.order.as_str()
                ),
                spans_of(&h.output),
            )?;
            Self::record_in(&mut st, &qid, &env, nmmusc::MessageKind::Report, CHAIN, &format!("hop:{}:{}", h.role, h.emits))?;
            steps.push(env);
        }
        // Step 5: the system's own settings edit, with action copies to G1 and G2.
        let set_env = Self::apply_setting(&mut st, "chain_in_use", SettingValue::Word(run.kind.clone()))?;
        Self::record_in(&mut st, &qid, &set_env, nmmusc::MessageKind::Instruction, "broadcast", "set:chain_in_use")?;
        steps.push(set_env);
        // The final output becomes the answer text (law 6: a withheld pack stays withheld),
        // followed by the citations of the parts it rests on that it does not already hold.
        // 2026-11-12: an output that rests on no part (the chain was given none) does not replace
        // the evidenced answer.
        let answer_replaced = !pack.withheld && !run.final_output.is_empty() && !run.citations.is_empty();
        let out = if answer_replaced {
            let head = emitted.text().lines().next().unwrap_or(&pack.reason).to_owned();
            with_text(emitted, &answer_text(&head, &run.final_output, &run.citations))?
        } else {
            emitted.clone()
        };
        let symbols = st.log.traces().iter().find(|t| t.id == qid).map_or(0, |t| t.symbols.len());
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        st.stats.questions += 1;
        st.stats.escalations += usize::from(escalated);
        st.stats.ms_total += ms;
        *st.stats.by_kind.entry(run.kind.clone()).or_default() += 1;
        st.kinds.insert(qid.clone(), run.kind.clone());
        st.asked.insert(
            qid.clone(),
            Asked {
                question_id: qid.clone(),
                question: question.to_owned(),
                answer: out.text().to_owned(),
                kind: run.kind.clone(),
                file_kind: fk,
            },
        );
        let settings_version = st.settings.version();
        drop(st);
        let _ = org;
        Ok((
            out,
            BrainRecord {
                question_id: qid,
                kind: run.kind.clone(),
                mode: run.mode.as_str(),
                escalated,
                confidence,
                threshold,
                g2_choice,
                upstream: up,
                held_back: sel.held_back,
                final_output: run.final_output,
                citations: run.citations,
                answer_replaced,
                log,
                steps,
                symbols,
                offers,
                cartridges: csel.to_value(),
                resolution,
                preferences: rules,
                settings_version,
                ms,
                g2_ms: sel.ms,
                chain_ms,
            },
        ))
    }

    /// After the episode is written: bind it to the question for the feedback path, and (memory
    /// on) add the chain log line for the episode (`units/chains.jsonl` since 2026-11-12; earlier
    /// it sat beside `episodes.jsonl` in the memory folder).
    pub fn after_remember(&self, org: &Organism, rec: &BrainRecord, episode: Option<&str>) {
        if let (Some(id), Ok(mut st)) = (episode, self.state.lock()) {
            st.episodes.insert(id.to_owned(), rec.question_id.clone());
        }
        if !org.memory_on {
            return;
        }
        let hops: Vec<Value> = rec
            .log
            .iter()
            .map(|h| {
                obj(vec![
                    ("hop", n(h.index)),
                    ("role", s(&h.role)),
                    ("label", s(&h.label)),
                    ("emits", s(&h.emits)),
                    ("lines", n(h.output.len())),
                    ("ms", f(h.ms)),
                ])
            })
            .collect();
        let line = obj(vec![
            ("at", Value::Number(nmtype::now_ms().to_string())),
            ("episode", episode.map_or(Value::Null, s)),
            ("question", s(&rec.question_id)),
            ("class", s("behavioural")),
            ("chain", s(&rec.kind)),
            ("mode", s(rec.mode)),
            ("escalated", Value::Bool(rec.escalated)),
            ("confidence", f(f64::from(rec.confidence))),
            ("hops", Value::Array(hops)),
        ]);
        append_line(&self.chain_log, &line);
    }

    /// Bind an episode id to a question id directly (checks).
    /// The whole path's decisions for one question as one behaviour sequence (2026-11-12, motor
    /// MO5: G1 and G2 decision events logged, so G3 mines decisions that varied): each event is a
    /// report from its level (`G1`, `G2`, the Core Brain) with a discrete action. When `episode` is
    /// given it is bound, so a verdict becomes the sequence's outcome. Returns the question id.
    /// # Errors
    /// A boundary refusal or a log fault.
    pub fn record_whole(&self, events: &[(&str, String)], episode: Option<&str>) -> Result<String, Fault> {
        let mut st = self.state.lock().map_err(|_| bad("brain side lock"))?;
        st.next_question += 1;
        let qid = format!("q{}", st.next_question);
        // Whole-path questions are one kind for the gain baseline (MO7).
        st.kinds.insert(qid.clone(), "whole".to_owned());
        for (level, action) in events {
            let env = Self::envelope(MessageKind::Report, level, G3, 1.0, &format!("{level} decided {action}"), Vec::new())?;
            Self::record_in(&mut st, &qid, &env, nmmusc::MessageKind::Report, G3, action)?;
        }
        if let Some(e) = episode {
            st.episodes.insert(e.to_owned(), qid.clone());
        }
        Ok(qid)
    }

    /// MO8 (2026-11-13, `nmmusc::priors`): the priors of the strongest live pack learned on the
    /// whole path (description names the `whole` kind; not withdrawn; evidence its promotion gain
    /// over its error less the opposing running margin, above 0). `None` with no store or no pack.
    #[must_use]
    pub fn whole_path_priors(&self) -> Option<Vec<nmmusc::priors::Prior>> {
        let store = self.store.as_ref()?;
        let z = nmmusc::Policy::load().gate_z;
        let mut best: Option<(f64, Vec<nmmusc::priors::Prior>)> = None;
        for row in store.index().ok()? {
            let Ok(p) = store.load(&row.fingerprint, &self.base_hash) else { continue };
            if p.header.withdrawn || !p.header.description.contains("chain whole") {
                continue;
            }
            let opposed = nmmusc::withdraw::margin(&p.header.promotion, &p.header.running, z).map_or(0.0, |m| (-m).max(0.0));
            let ev = nmmusc::priors::evidence(&p.header.promotion, opposed);
            let steps: Vec<String> = p.steps().into_iter().map(|(_, d)| d).collect();
            let pr = nmmusc::priors::priors(&p.header.fingerprint, &steps, ev);
            if ev > 0.0 && !pr.is_empty() && best.as_ref().is_none_or(|(e, _)| ev > *e) {
                best = Some((ev, pr));
            }
        }
        best.map(|(_, p)| p)
    }

    pub fn bind_episode(&self, episode: &str, question_id: &str) {
        if let Ok(mut st) = self.state.lock() {
            st.episodes.insert(episode.to_owned(), question_id.to_owned());
        }
    }

    /// The taste path (2026-11-12, signal router): an accept or reject of the offered answer's
    /// form, given as taste, feeds the preference store, and nothing else.
    #[must_use]
    pub fn on_taste(&self, episode: &str, accept: bool) -> Value {
        if nmlearn::route::admit(nmlearn::route::SignalKind::Taste, nmlearn::route::Layer::L6Preference).is_err() {
            return obj(vec![("preference_logged", Value::Bool(false))]);
        }
        let Ok(mut st) = self.state.lock() else { return Value::Null };
        let Some(qid) = st.episodes.get(episode).cloned() else { return obj(vec![("taste", s("episode not asked in this session"))]) };
        let logged = match st.asked.get(&qid).cloned() {
            Some(a) => {
                let fb = pref::Feedback {
                    circumstance: pref::Circumstance::new(&a.kind, "writer", &a.file_kind, &a.question),
                    offered: a.answer.clone(),
                    alternative: None,
                    signal: if accept { pref::Signal::Accept } else { pref::Signal::Reject },
                    source: "nm_feedback taste".into(),
                };
                match pref::feed(&self.cfg.pref_home, &fb) {
                    Ok(r) => {
                        st.prefs.records.push(r);
                        st.pref_index = pref::Index::build(&st.prefs, &pref::ExactEmbedder);
                        true
                    }
                    Err(e) => {
                        eprintln!("brain chains: preference feed: {e}");
                        false
                    }
                }
            }
            None => false,
        };
        obj(vec![("preference_logged", Value::Bool(logged))])
    }

    /// The feedback path (after `nm_feedback`'s outcome line): the outcome is attached to the
    /// question's behaviour trace, the preference feed gets the verdict, the question's
    /// cartridge uses get their verdict.
    #[must_use]
    pub fn on_feedback(&self, episode: &str, correct: bool) -> Value {
        let Ok(mut st) = self.state.lock() else { return Value::Null };
        let Some(qid) = st.episodes.get(episode).cloned() else {
            return obj(vec![("brain_chains", s("episode not asked in this session"))]);
        };
        let attached = st.log.set_outcome(&qid, if correct { 1.0 } else { 0.0 }).is_ok();
        // 2026-11-12 (signal router, Solution 6 step 1): a right or wrong verdict is not taste, so
        // it no longer reaches the preference store; taste arrives by its own signal.
        let verdict_to_pref = nmlearn::route::admit(nmlearn::route::SignalKind::Verdict, nmlearn::route::Layer::L6Preference).is_ok();
        let pref_logged = match st.asked.get(&qid).cloned().filter(|_| verdict_to_pref) {
            Some(a) => match pref::feed_verdict(
                &self.cfg.pref_home,
                pref::Circumstance::new(&a.kind, "writer", &a.file_kind, &a.question),
                &a.answer,
                correct,
            ) {
                Ok(r) => {
                    st.prefs.records.push(r);
                    st.pref_index = pref::Index::build(&st.prefs, &pref::ExactEmbedder);
                    let _ = a.question_id;
                    true
                }
                Err(e) => {
                    eprintln!("brain chains: preference feed: {e}");
                    false
                }
            },
            None => false,
        };
        let used = st.open_uses.remove(&qid).unwrap_or_default();
        for id in &used {
            append_line(
                &self.cfg.uses_path,
                &obj(vec![
                    ("at", Value::Number(nmtype::now_ms().to_string())),
                    ("cartridge", s(id)),
                    ("question", s(&qid)),
                    ("right", Value::Bool(correct)),
                    ("class", s("behavioural")),
                ]),
            );
            st.uses.push((id.clone(), correct));
        }
        obj(vec![
            ("outcome_attached", Value::Bool(attached)),
            ("preference_logged", Value::Bool(pref_logged)),
            ("cartridge_verdicts", n(used.len())),
        ])
    }

    /// G3 mining and promotion, on request only (not on every question): mine the behaviour
    /// log; each recurring chunk with a promotable gain record becomes a pack in the store,
    /// described by the chain kind it served most.
    /// # Errors
    /// The lock, or a pack could not be built or written.
    pub fn mine_and_promote(&self) -> Result<Value, Fault> {
        let st = self.state.lock().map_err(|_| bad("brain side lock"))?;
        let mined = nmmusc::mine(st.log.traces());
        let z = nmmusc::Policy::load().gate_z;
        let mut written = Vec::new();
        let mut promotable = 0;
        for c in &mined.candidates {
            // Gain by the kind baseline (MO7): effect = outcome minus the mean of its question kind.
            let kind_of = |id: &str| st.kinds.get(id).cloned().unwrap_or_else(|| "any".to_owned());
            let Some(rec) = nmmusc::KindMeans { kind_of: &kind_of }.estimate(st.log.traces(), &c.symbols) else { continue };
            if !rec.promotable(z) {
                continue;
            }
            promotable += 1;
            let mut votes: BTreeMap<&str, usize> = BTreeMap::new();
            for t in st.log.traces().iter().filter(|t| t.symbols.windows(c.symbols.len()).any(|w| w == c.symbols.as_slice())) {
                if let Some(k) = st.kinds.get(&t.id) {
                    *votes.entry(k.as_str()).or_default() += 1;
                }
            }
            let kind = votes.iter().max_by_key(|(_, v)| **v).map_or("any", |(k, _)| *k);
            let p = nmmusc::Pack::build(
                &c.symbols,
                &st.log.codebook,
                &format!("chain {kind}: {} behavioural messages that recurred with a measured gain", c.symbols.len()),
                &self.base_hash,
                rec,
            )
            .map_err(|e| musc(&e))?;
            if let Some(store) = self.store.as_ref() {
                store.write(&p).map_err(|e| musc(&e))?;
                written.push(s(&p.header.fingerprint));
            }
        }
        Ok(obj(vec![
            ("traces", n(mined.stats.traces)),
            ("candidates", n(mined.candidates.len())),
            ("promotable", n(promotable)),
            ("written", Value::Array(written)),
        ]))
    }

    /// Each behavioural step of a record rendered back to text through the envelope renderer
    /// (spans read from the substrate; labels byte-level).
    #[must_use]
    pub fn trace_lines(&self, sub: &nmsubs::Substrate, rec: &BrainRecord) -> Vec<String> {
        let r = SubResolver(sub);
        let mut out = vec![format!(
            "brain chains: chain {} ({}), G2 choice {} confidence {:.3} threshold {:.3}{}, {} parts up, {} held back, {} hops, {} symbols, {} packs offered, {} cartridges active, {} preference rules, {:.2} ms (G2 {:.3}, chain {:.3})",
            rec.kind,
            rec.mode,
            rec.g2_choice,
            rec.confidence,
            rec.threshold,
            if rec.escalated { " escalated" } else { "" },
            rec.upstream,
            rec.held_back,
            rec.log.len(),
            rec.symbols,
            rec.offers.len(),
            rec.cartridges.get("active").and_then(Value::array).map_or(0, <[Value]>::len),
            rec.preferences.len(),
            rec.ms,
            rec.g2_ms,
            rec.chain_ms
        )];
        for (i, env) in rec.steps.iter().enumerate() {
            out.push(format!("--- behavioural step {i}"));
            out.push(env.render(&r).unwrap_or_else(|_| env.render_bare()));
        }
        out
    }
}

/// The least verified outcome a closed call needs to become one of G2's past cases.
pub const CASE_OUTCOME_MIN: f64 = 0.5;

/// G2's past cases (Stage 5 WP-C2): the built-in set plus each verified closed call of the G3 store
/// whose G2 chain site chose a chain kind of the chain book (written in brain mode) and whose
/// outcome reached `CASE_OUTCOME_MIN`. A case's task description holds only the call's labels (its
/// request kind and area), never the question's words, as the configuration records hold no words
/// (`nmtype::class` boundaries).
#[derive(Clone, Debug, Default)]
pub struct RecordCases {
    pub extra: Vec<nmbrain::g2::Case>,
}

impl RecordCases {
    /// The closed calls of `records` that teach G2 a chain choice.
    #[must_use]
    pub fn from_records(records: &[nmmusc::g3::Record], kinds: &[String]) -> Self {
        let extra = records
            .iter()
            .filter(|r| r.outcome.is_some_and(|o| o >= CASE_OUTCOME_MIN))
            .filter_map(|r| {
                let (_, chain) = r.choices.iter().find(|(site, choice)| *site == nmmusc::priors::Site::Chain && kinds.contains(choice))?;
                Some(nmbrain::g2::Case { text: format!("{} request in {}", r.kind, r.area), label: chain.clone() })
            })
            .collect();
        Self { extra }
    }
}

impl nmbrain::g2::CaseSource for RecordCases {
    fn cases(&self) -> Vec<nmbrain::g2::Case> {
        let mut c = nmbrain::g2::CaseSource::cases(&BuiltinCases);
        c.extend(self.extra.iter().cloned());
        c
    }
}

/// `BuiltinCases` size, for the session note.
trait CasesLen {
    fn cases_len(&self) -> usize;
}
impl CasesLen for BuiltinCases {
    fn cases_len(&self) -> usize {
        nmbrain::g2::CaseSource::cases(self).len()
    }
}

fn append_line(path: &Path, v: &Value) {
    let Ok(line) = v.encode() else { return };
    if let Some(p) = path.parent() {
        let _ = std::fs::create_dir_all(p);
    }
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut fh| std::io::Write::write_all(&mut fh, format!("{line}\n").as_bytes()));
}
fn read_uses(path: &Path) -> Vec<(String, bool)> {
    std::fs::read_to_string(path)
        .map(|t| {
            t.lines()
                .filter_map(|l| nmjson::parse(l.as_bytes()).ok())
                .filter_map(|v| {
                    Some((
                        v.get("cartridge").and_then(Value::string).ok()?.to_owned(),
                        matches!(v.get("right"), Ok(Value::Bool(true))),
                        matches!(v.get("right"), Ok(Value::Bool(_))),
                    ))
                })
                .filter(|(_, _, judged)| *judged)
                .map(|(id, ok, _)| (id, ok))
                .collect()
        })
        .unwrap_or_default()
}

/// Whether the served option is given. With it, `--brain-decide` also switches the Core Brain's
/// decision step on (`crate::decide`, off by default; 2026-11-10).
#[must_use]
pub fn from_args(args: &[String]) -> bool {
    let on = args.iter().any(|a| a == FLAG);
    if on && args.iter().any(|a| a == crate::decide::FLAG) {
        crate::decide::set_enabled(true);
    }
    on
}

/// Switch the Core Brain side on for this organism (live layout).
/// # Errors
/// As `BrainSide::open`.
pub fn enable(org: &Organism) -> Result<String, Fault> {
    enable_with(org, Config::live(&org.home))
}
/// As `enable`, with the given folders.
/// # Errors
/// As `BrainSide::open`.
pub fn enable_with(org: &Organism, cfg: Config) -> Result<String, Fault> {
    let side = BrainSide::open(org, cfg)?;
    let note = side.note.clone();
    if let Ok(mut g) = org.brain_side.write() {
        *g = Some(Arc::new(side));
    }
    Ok(note)
}
/// The session's Core Brain side, when on.
#[must_use]
pub fn side(org: &Organism) -> Option<Arc<BrainSide>> {
    org.brain_side.read().ok().and_then(|s| s.clone())
}

// ---------------------------------------------------------------------------------------------
// Phase I1: the Core Brain's intake on the served side (tentacle master plan 3.9; lane M,
// 2026-11-19). `nmbrain::chain::run_evidence` runs the chain over G2's gate and per-piece evidence;
// here the card's MicroModels are put behind its reader traits, each hop becomes a behavioural
// envelope (chain to G2) admitted at the message channel, and the chain chosen is logged as a G3
// decision site. Nothing here is called by the served flow until lane T1 calls `intake`.
// ---------------------------------------------------------------------------------------------

fn logit(p: f32) -> f32 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

/// The judge's duel reads on the card, one pair at a time (the packed path is
/// `drone_path::judge_pairs`): the posed read of `A` against `B` for the question, and the mean
/// content-free read over `neural::CONTENT_FREE`, each as label log-odds. `text` gives a piece's
/// posed and request-free texts by its content key.
pub struct JudgeReader<'a> {
    pub model: &'a nmengn::model::Model,
    pub question: String,
    pub text: &'a dyn Fn(&str) -> Option<(String, String)>,
}
impl JudgeReader<'_> {
    fn score(&self, q: &str, a: &str, b: &str) -> Result<f32, String> {
        let dev = if self.model.pop.gpu.is_some() { nmengn::placement::Device::Gpu } else { nmengn::placement::Device::Cpu };
        let doc = format!("Passage A:\n{a}\n\nPassage B:\n{b}");
        let p = nmengn::roles::card_rerank_pairs(self.model, dev, crate::drone_path::PAIR_INSTRUCTION, &[(q, doc.as_str())]).map_err(|f| f.to_string())?;
        p.first().copied().map(logit).ok_or_else(|| "no score".to_owned())
    }
}
impl nmbrain::chain::DuelReader for JudgeReader<'_> {
    fn posed(&mut self, a: &str, b: &str) -> Result<f32, String> {
        let (Some((ta, _)), Some((tb, _))) = ((self.text)(a), (self.text)(b)) else { return Err(format!("no text for {a} or {b}")) };
        let q = self.question.clone();
        self.score(&q, &ta, &tb)
    }
    fn content_free(&mut self, a: &str, b: &str) -> Result<f32, String> {
        let (Some((_, ua)), Some((_, ub))) = ((self.text)(a), (self.text)(b)) else { return Err(format!("no text for {a} or {b}")) };
        let mut sum = 0.0;
        for f in crate::neural::CONTENT_FREE {
            sum += self.score(f, &ua, &ub)?;
        }
        #[expect(clippy::cast_precision_loss, reason = "a count of forms")]
        let n = crate::neural::CONTENT_FREE.len() as f32;
        Ok(sum / n)
    }
    fn forms(&self) -> usize {
        crate::neural::CONTENT_FREE.len()
    }
}

/// A hop over values already read (the embedder's meaning read less its content-free mean, the
/// reader's joint or one-word read as label log-odds): each named piece's value less the named
/// pieces' mean, over `scale` (the values' noise band), so the hop adds calibrated evidence on
/// the log-odds scale. A piece with no value adds nothing.
pub struct ValueHop<'a> {
    pub model: nmbrain::chain::MicroModel,
    pub value: &'a dyn Fn(nmbrain::chain::Job, &str) -> Option<f32>,
    pub scale: f32,
}
impl nmbrain::chain::HopReader for ValueHop<'_> {
    fn model(&self) -> nmbrain::chain::MicroModel {
        self.model
    }
    fn read(&mut self, job: nmbrain::chain::Job, pieces: &[&nmbrain::chain::PieceEvidence]) -> Result<Vec<f32>, String> {
        let v: Vec<Option<f32>> = pieces.iter().map(|p| (self.value)(job, &p.key).filter(|x| x.is_finite())).collect();
        let have: Vec<f32> = v.iter().flatten().copied().collect();
        if have.is_empty() {
            return Err("no values".into());
        }
        #[expect(clippy::cast_precision_loss, reason = "a count of pieces")]
        let mean = have.iter().sum::<f32>() / have.len() as f32;
        let s = if self.scale.is_finite() && self.scale > 0.0 { self.scale } else { 1.0 };
        Ok(v.into_iter().map(|x| x.map_or(0.0, |x| (x - mean) / s)).collect())
    }
}

/// The reader as the chain's joint and one-word hops (as `joint_read` serves G2's ties): the joint
/// hop reads the named pieces together in one sequence, both orders, the label readout (logit of
/// "yes" less "no" that piece `[k]` answers the question); the one-word hop reads each piece alone
/// with the same yes or no judgement. Each hop's values less their mean, as label log-odds.
pub struct ReaderHop<'a> {
    pub model: &'a nmengn::model::Model,
    pub question: String,
    pub heads: Vec<(usize, usize)>,
    pub last: usize,
    pub piece: &'a dyn Fn(&str) -> Option<crate::joint_read::Piece>,
}
impl nmbrain::chain::HopReader for ReaderHop<'_> {
    fn model(&self) -> nmbrain::chain::MicroModel {
        nmbrain::chain::MicroModel::Reader
    }
    fn read(&mut self, job: nmbrain::chain::Job, pieces: &[&nmbrain::chain::PieceEvidence]) -> Result<Vec<f32>, String> {
        let dev = if self.model.pop.gpu.is_some() { nmengn::placement::Device::Gpu } else { nmengn::placement::Device::Cpu };
        let ps: Vec<crate::joint_read::Piece> = pieces.iter().filter_map(|p| (self.piece)(&p.key)).collect();
        if ps.len() != pieces.len() || ps.is_empty() {
            return Err("a piece has no text".into());
        }
        let read = |set: &[crate::joint_read::Piece]| -> Result<Vec<f32>, String> {
            crate::joint_read::read_with(self.model, dev, &self.question, set, &self.heads, self.last, usize::MAX, true, true, false)
                .map(|o| o.yes_no.group_score)
                .map_err(|f| f.to_string())
        };
        let v: Vec<f32> = if job == nmbrain::chain::Job::OneWord {
            ps.iter().map(|p| read(std::slice::from_ref(p)).map(|s| s.first().copied().unwrap_or(0.0))).collect::<Result<_, _>>()?
        } else {
            read(&ps)?
        };
        #[expect(clippy::cast_precision_loss, reason = "a count of pieces")]
        let mean = v.iter().filter(|x| x.is_finite()).sum::<f32>() / v.len().max(1) as f32;
        Ok(v.into_iter().map(|x| if x.is_finite() { x - mean } else { 0.0 }).collect())
    }
}

/// The evidence bound when none is fitted: twice the noise band (median absolute deviation) of the
/// pieces' log-odds, never under one log-odds unit (stated fallback until H16's replay fits it).
#[must_use]
pub fn evidence_bound(pieces: &[nmbrain::chain::PieceEvidence]) -> f32 {
    let mut v: Vec<f32> = pieces.iter().map(|p| p.logodds).filter(|x| x.is_finite()).collect();
    if v.len() < 2 {
        return 1.0;
    }
    v.sort_by(f32::total_cmp);
    let med = v[v.len() / 2];
    let mut d: Vec<f32> = v.iter().map(|x| (x - med).abs()).collect();
    d.sort_by(f32::total_cmp);
    (2.0 * d[d.len() / 2]).max(1.0)
}

/// What the intake did on one call.
#[derive(Debug)]
pub struct Intake {
    pub out: nmbrain::chain::ChainOut,
    /// One behavioural envelope per hop, chain to G2, as admitted at the message channel.
    pub envelopes: Vec<Envelope>,
    pub bound: f32,
}

/// Phase I1: run the Core Brain's intake over G2's gate line and its pieces (in G2's order, each
/// with its calibrated evidence): the plan from the gate, the routing table's choice per job
/// (`priors` from past configurations or muscle memory), each hop's evidence added, stopping at the
/// bound (`None`: `evidence_bound`) or `deadline`. Each hop becomes a behavioural envelope admitted
/// at the message channel and a G3 decision site (`core-brain.chain`, `<job>:<model>`).
/// # Errors
/// A hop envelope refused (it never is for a well-formed hop).
pub fn intake(
    gate: &str,
    pieces: Vec<nmbrain::chain::PieceEvidence>,
    readers: &mut [&mut dyn nmbrain::chain::HopReader],
    priors: &dyn Fn(nmbrain::chain::Job) -> Option<(nmbrain::chain::MicroModel, f64)>,
    bound: Option<f32>,
    deadline: Instant,
) -> Result<Intake, Fault> {
    use nmbrain::chain::{IntakeIn, Routing, plan_of, run_evidence};
    let i = IntakeIn { gate: nmbrain::chain::GateStatus::of(gate), pieces };
    let bound = bound.unwrap_or_else(|| evidence_bound(&i.pieces));
    // Stage 5 WP-C2: the job map with the measured routing record laid over it when one is in
    // force for the serving base cartridges (`drone_path::measured_record`), else the job map.
    let routing = crate::drone_path::measured_record().map_or_else(Routing::job_map, Routing::from_record);
    let out = run_evidence(&i, &plan_of(&i), &routing, readers, priors, bound, deadline);
    let mut envelopes = Vec::with_capacity(out.hops.len());
    for h in &out.hops {
        let model = h.model.map_or("none", nmbrain::chain::MicroModel::name);
        let line = format!("intake hop {} by {model}: {} read, margin {:.2}{}", h.job.name(), h.read, h.margin, if h.note.is_empty() { String::new() } else { format!(", {}", h.note) });
        let env = Envelope::new(MessageKind::Report, CHAIN, G2, 1.0, &line, Vec::new(), Vec::new())?;
        bnd::MESSAGE_CHANNEL.admit(&env)?;
        envelopes.push(env);
        if let Some(m) = h.model {
            let _ = crate::g3::note_site(nmmusc::sites::SiteKind::Chain, "g2", "-", &format!("{}:{}", h.job.name(), m.name()), None);
        }
    }
    Ok(Intake { out, envelopes, bound })
}

/// Section check of the served intake: a tie on code goes to the embedder's values, the hop is an
/// envelope on the message channel, the bound falls back to one unit, and a withheld gate reads
/// nothing.
/// # Errors
/// One fault line.
pub fn check_intake() -> Result<(), String> {
    use nmbrain::chain::{Job, MicroModel, PieceEvidence, Stop};
    let p = |k: &str, l: f32| PieceEvidence { key: k.into(), prose: false, logodds: l };
    let vals = |_j: Job, k: &str| -> Option<f32> { Some(if k == "right" { 0.9 } else { 0.1 }) };
    let mut emb = ValueHop { model: MicroModel::Embedder, value: &vals, scale: 0.2 };
    let far = Instant::now() + std::time::Duration::from_secs(30);
    let r = intake("tie: two within the band", vec![p("wrong", 1.0), p("right", 0.8)], &mut [&mut emb], &|_| None, None, far).map_err(|f| f.to_string())?;
    if r.out.order.first() != Some(&1) || r.envelopes.len() != 1 || r.out.stop != Stop::Bound || (r.bound - 1.0).abs() > 1e-6 {
        return Err(format!("nmapi.brain_side.intake: tie {:?} {}", r.out, r.envelopes.len()));
    }
    let w = intake("withheld", vec![p("a", 1.0)], &mut [&mut emb], &|_| None, None, far).map_err(|f| f.to_string())?;
    if w.out.stop != Stop::Withheld || !w.envelopes.is_empty() {
        return Err("nmapi.brain_side.intake: withheld".into());
    }
    Ok(())
}

/// Section check without an organism: the chain book with the preference seed validates; a
/// behavioural envelope is refused at the sensory boundaries and a sensory packet at the
/// behavioural ones and by the behaviour log; a span reference is parsed from an evidence line.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let o = |a: usize| Offer { pack: "p".into(), description: "d".into(), proposed: 2, accepted: a, declined: 2 - a };
    if !pack_executes(true, false, &[o(1)])
        || pack_executes(false, false, &[o(2)])
        || pack_executes(true, true, &[o(2)])
        || pack_executes(true, false, &[o(0)])
    {
        return Err("nmapi.brain_side: pack execution rule".into());
    }
    let book = ChainBook::from_mini().map_err(|e| e.to_string())?;
    let bp = with_preferences(&book);
    for c in &bp.chains {
        bp.validate(c).map_err(|e| format!("{}: {e}", c.kind))?;
    }
    let env =
        Envelope::new(MessageKind::Instruction, G2, CHAIN, 0.5, "run chain locate-code", Vec::new(), label_tokens("run chain locate-code"))
            .map_err(|e| e.to_string())?;
    let sensory = Packet::new("grep", "k", "text", nmtype::Provenance::new("a.rs:1", "check", 1, 0));
    for b in [bnd::CHAIN_INPUT, bnd::ANSWER_EVIDENCE] {
        match b.admit(&env) {
            Err(e) if e.message.starts_with(nmtype::WRONG_CLASS) => {}
            other => return Err(format!("{} admitted a behavioural envelope: {other:?}", b.name)),
        }
    }
    for b in [bnd::MESSAGE_CHANNEL, bnd::MUSCLE_MEMORY] {
        match b.admit(&sensory) {
            Err(e) if e.message.starts_with(nmtype::WRONG_CLASS) => {}
            other => return Err(format!("{} admitted a sensory packet: {other:?}", b.name)),
        }
    }
    let mut log = BehaviourLog::default();
    match log.record("q", &Symbol { item: &sensory, kind: nmmusc::MessageKind::Report, target: "x".into(), action: "y".into() }) {
        Err(e) if e.code == "nmmusc-sensory-refused" => {}
        other => return Err(format!("behaviour log took a sensory item: {other:?}")),
    }
    if log.traces().iter().any(|t| !t.symbols.is_empty()) {
        return Err("a refused item left a symbol".into());
    }
    let spans = spans_of(&["crates/a.rs:12 fn x()".to_owned(), "notes".to_owned()]);
    if spans != vec![Reference::Span { path: "crates/a.rs".into(), start: 12, end: 12 }] {
        return Err(format!("span parse: {spans:?}"));
    }
    Ok(())
}
