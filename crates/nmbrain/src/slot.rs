//! MicroModel slots of the Core Brain (guide 3; PLAN 4c, rule 8): each slot runs its exact
//! control always, and its Qwen3 engine when one is bound (S6, S7), recording both; the
//! comparison is the qualification record. Exact controls, first written in the earlier
//! `nmcore::brain` pass: hashed unigram and bigram embedder, key-term reranker, keyword
//! router, extractive writer (returns the best evidence lines, never generated text).
use nmjson::Value;
use nmtype::{Fault, FaultKind, SlotView};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

/// Embedding width of the exact embedder.
pub const DIM: usize = 256;

/// The exact control behind every slot.
pub trait Exact: Send + Sync {
    fn route(&self, q: &str) -> Vec<String>;
    fn embed(&self, t: &str) -> Vec<f32>;
    fn rerank(&self, q: &str, d: &str) -> f32;
    /// Scores for many documents against one question; the same numbers as `rerank` per
    /// document, with the question's side worked out once (2026-11-08).
    fn rerank_all(&self, q: &str, docs: &[String]) -> Vec<f32> {
        docs.iter().map(|d| self.rerank(q, d)).collect()
    }
    /// `rerank_all` for passages of documents (prose, tables, records): the question's document
    /// terms (`nmlex::key_terms_doc`) instead of its code terms. The same as `rerank_all` unless
    /// a control says otherwise.
    fn rerank_all_doc(&self, q: &str, docs: &[String]) -> Vec<f32> {
        self.rerank_all(q, docs)
    }
    /// `rerank_all` (or `rerank_all_doc` when `doc`) measured against what the studied material
    /// says about the question (`Material`). The same as those unless a control says otherwise.
    fn rerank_all_in(&self, q: &str, docs: &[String], doc: bool, m: &Material) -> Vec<f32> {
        let _ = m;
        if doc { self.rerank_all_doc(q, docs) } else { self.rerank_all(q, docs) }
    }
    fn write(&self, prompt: &str, evidence: &[String]) -> String;
}

/// The exact controls of the mini.
#[derive(Clone, Copy, Debug, Default)]
pub struct Controls;

/// Exact embedder: hashed unigrams (weight 1) and bigrams (0.5), L2-normalised.
#[must_use]
pub fn embed(text: &str) -> Vec<f32> {
    let w = nmlex::words(text);
    let mut v = vec![0.0_f32; DIM];
    let mut add = |t: &str, wt: f32| {
        let h = t.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3));
        let i = usize::try_from(h % DIM as u64).unwrap_or(0);
        if let Some(x) = v.get_mut(i) {
            *x += if h & (1 << 40) == 0 { wt } else { -wt };
        }
    };
    for (i, t) in w.iter().enumerate() {
        add(t, 1.0);
        if let Some(n) = w.get(i + 1) {
            add(&format!("{t} {n}"), 0.5);
        }
    }
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
    v
}
/// Share of the question's names and key terms a text contains.
fn coverage(q: &str, d: &str) -> f32 {
    coverage_of(&coverage_terms(q), d)
}
/// The question's identifiers (lower-cased) and key terms, the terms `coverage` counts.
pub(crate) fn coverage_terms(q: &str) -> Vec<String> {
    let mut terms: Vec<String> = nmlex::names(q).iter().map(|n| n.to_lowercase()).collect();
    for t in nmlex::key_terms(q) {
        if !terms.contains(&t) {
            terms.push(t);
        }
    }
    terms
}
/// `coverage_terms` for document passages: names, then the document terms.
pub(crate) fn coverage_terms_doc(q: &str) -> Vec<String> {
    let mut terms: Vec<String> = nmlex::names(q).iter().map(|n| n.to_lowercase()).collect();
    for t in nmlex::key_terms_doc(q) {
        if !terms.contains(&t) {
            terms.push(t);
        }
    }
    terms
}
pub(crate) fn coverage_of(terms: &[String], d: &str) -> f32 {
    coverage_in(terms, d, 0.0)
}
/// `coverage_of` where a term the text holds only as its stem (one inflection ending off, the stem
/// four letters or more: `scales` in `scale_kernel`, `pushed` in `ringbuf_push`) counts
/// `form_weight` of a term held as written.
pub(crate) fn coverage_in(terms: &[String], d: &str, form_weight: f32) -> f32 {
    if terms.is_empty() {
        return 0.0;
    }
    let dl = d.to_lowercase();
    let held: f32 = terms
        .iter()
        .map(|t| {
            if dl.contains(t.as_str()) {
                1.0
            } else if form_weight > 0.0 && ENDINGS.iter().filter_map(|e| t.strip_suffix(e)).any(|s| s.len() >= 4 && dl.contains(s)) {
                form_weight
            } else {
                0.0
            }
        })
        .sum();
    #[expect(clippy::cast_precision_loss, reason = "term counts are small")]
    let c = held / terms.len() as f32;
    c
}
/// Inflection endings a term held by its stem may differ by.
const ENDINGS: [&str; 5] = ["ing", "ed", "es", "er", "s"];

/// What the studied material says about the question being reduced, for the exact control's
/// coverage: the question terms it holds in no form (no packet can hold them, so they are left
/// out), and how much a term held only by its stem counts (one minus the material's expected
/// presence of a word: in a small material the exact form of a word is rarely there even where
/// the answer is, so its stem is good evidence; in a large one the exact form would be, so a stem
/// alone counts little). The default changes nothing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Material {
    pub unmatched: Vec<String>,
    pub form_weight: f32,
}
impl Material {
    /// True when it changes nothing.
    #[must_use]
    pub fn is_plain(&self) -> bool {
        self.unmatched.is_empty() && self.form_weight <= 0.0
    }
}
fn cos(a: &[f32], b: &[f32]) -> f32 {
    let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    d.max(0.0)
}

impl Exact for Controls {
    fn route(&self, q: &str) -> Vec<String> {
        let l = q.to_lowercase();
        let has = |xs: &[&str]| xs.iter().any(|x| l.contains(x));
        let mut j: Vec<&str> = Vec::new();
        // Jobs picked here from the question's words become focus jobs that every tentacle
        // carries (2026-11-09); the rest are the territory jobs.
        // A call question asks for call sites whatever its opening word ("what calls", "which code
        // calls", "where is it called").
        if has(&["depend", "used by", "deps", " used", " uses", "callers", " calls ", "called from", "called by", "is called", "call site"]) {
            j.push("dependency");
        }
        // History phrases only (2026-11-09): a bare "changed", "when" or "why" also appears in
        // questions about behaviour ("how is the state changed", "when the server starts").
        if has(&["history", "commit", "recent", "what changed", "files changed", "when did", "when was", "last change", "modified"]) {
            j.push("historical");
        }
        // Not "step check": in this project it names a component, so "how is a Step Check
        // specified" is not a failure question (M2 revision 2, 2026-11-09).
        if has(&["build", "error", "compile", "fail", "clippy", "broke"]) {
            j.push("execution");
        }
        if has(&["test", "verif", "covered", "coverage"]) {
            j.push("verification");
        }
        // 2026-11-10: rule and interface questions.
        // Phrases that ask about the project's rules, not a bare "rule" ("the rule layer",
        // "IEEE rules" are about code).
        if has(&[
            "which rule",
            "what rule",
            "rules apply",
            "rules say",
            "rule covers",
            "is it allowed",
            "allowed to",
            "may i",
            "convention",
            "policy",
            "lint",
        ]) {
            j.push("constraint");
        }
        if has(&["public api", "public interface", "public items", "signature", "interface of", "exports"]) {
            j.push("functional");
        }
        j.extend(["structural", "retrieval", "semantic", "reasoning"]);
        let mut out: Vec<String> = Vec::new();
        for x in j {
            if !out.iter().any(|y| y == x) {
                out.push(x.to_owned());
            }
        }
        out
    }
    fn embed(&self, t: &str) -> Vec<f32> {
        embed(t)
    }
    /// Key-term coverage of the question's rare-looking names and terms, blended with
    /// embedding similarity. Orders only; never decides what is true.
    fn rerank(&self, q: &str, d: &str) -> f32 {
        0.8 * coverage(q, d) + 0.2 * cos(&embed(q), &embed(d))
    }
    fn rerank_all(&self, q: &str, docs: &[String]) -> Vec<f32> {
        let (terms, qv) = (coverage_terms(q), embed(q));
        docs.iter().map(|d| 0.8 * coverage_of(&terms, d) + 0.2 * cos(&qv, &embed(d))).collect()
    }
    fn rerank_all_doc(&self, q: &str, docs: &[String]) -> Vec<f32> {
        let (terms, qv) = (coverage_terms_doc(q), embed(q));
        docs.iter().map(|d| 0.8 * coverage_of(&terms, d) + 0.2 * cos(&qv, &embed(d))).collect()
    }
    fn rerank_all_in(&self, q: &str, docs: &[String], doc: bool, m: &Material) -> Vec<f32> {
        let mut terms = if doc { coverage_terms_doc(q) } else { coverage_terms(q) };
        terms.retain(|t| !m.unmatched.contains(t));
        let qv = embed(q);
        docs.iter().map(|d| 0.8 * coverage_in(&terms, d, m.form_weight) + 0.2 * cos(&qv, &embed(d))).collect()
    }
    /// Extractive: the evidence lines that cover the most question terms, keeping the given
    /// order among equals (the pack is already ranked), at most three.
    fn write(&self, prompt: &str, evidence: &[String]) -> String {
        let mut scored: Vec<(f32, &String)> = evidence.iter().map(|e| (coverage(prompt, e), e)).collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.into_iter().take(3).filter(|(s, _)| *s > 0.0).map(|(_, e)| e.clone()).collect::<Vec<_>>().join("\n")
    }
}

/// A neural engine bound to a slot (S6: our embed and rerank components over the one
/// Qwen3-0.6B population; nmapi wires it from nmengn, so nmbrain does not depend on the
/// engine). Orders and similarities only; the exact control decides admission (law 6).
pub trait Engine: Send + Sync {
    /// L2-normalised embedding.
    /// # Errors
    /// The engine failed (tokenizer, provisioning or a device fault).
    fn embed(&self, text: &str) -> Result<Vec<f32>, Fault>;
    /// Relevance score for ordering (higher first).
    /// # Errors
    /// As `embed`.
    fn rerank(&self, query: &str, doc: &str) -> Result<f32, Fault>;
    /// Scores for one whole set of documents, when the engine orders sets (a learned readout
    /// behind its scope and benefit gates); `None` keeps the per-pair `rerank`.
    /// # Errors
    /// As `embed`.
    fn rerank_set(&self, query: &str, docs: &[String]) -> Result<Option<Vec<f32>>, Fault> {
        let _ = (query, docs);
        Ok(None)
    }
    /// RAM MB and GPU MB held by the engine's parameter views.
    fn resident_mb(&self) -> (f32, f32);
    /// Release device views (the population stays in RAM).
    fn release_device(&self);
    /// What is bound, for reports (population hash prefix, assembly, device).
    fn label(&self) -> String;
}

/// Qualification record of one slot (rule 8): engine calls, time and faults beside the
/// control, which always runs.
#[derive(Clone, Debug, Default)]
pub struct Record {
    pub control_calls: u64,
    pub engine_calls: u64,
    pub engine_ms: f64,
    pub engine_faults: u64,
    /// Shadow comparisons (engine order beside control order) and how often their first
    /// candidates agreed.
    pub shadow: u64,
    pub shadow_agree: u64,
}

/// One slot: a name, its weight card id, its exact control, an optional neural engine,
/// homeostasis flags (suspended, throttled) and its qualification record.
pub struct Slot {
    pub name: String,
    pub card: String,
    pub control: Box<dyn Exact>,
    engine: RwLock<Option<Arc<dyn Engine>>>,
    suspended: AtomicBool,
    throttled: AtomicBool,
    qualified: AtomicBool,
    record: Mutex<Record>,
}
impl Slot {
    #[must_use]
    pub fn new(name: &str, card: &str) -> Self {
        Self {
            name: name.into(),
            card: card.into(),
            control: Box::new(Controls),
            engine: RwLock::new(None),
            suspended: AtomicBool::new(false),
            throttled: AtomicBool::new(false),
            qualified: AtomicBool::new(false),
            record: Mutex::new(Record::default()),
        }
    }
    #[must_use]
    pub fn calls(&self) -> u64 {
        self.record.lock().map_or(0, |r| r.control_calls)
    }
    /// The engine, when bound and not suspended.
    #[must_use]
    pub fn engine(&self) -> Option<Arc<dyn Engine>> {
        if self.suspended.load(Ordering::SeqCst) {
            return None;
        }
        self.engine.read().ok().and_then(|e| e.clone())
    }
    #[must_use]
    pub fn engine_bound(&self) -> bool {
        self.engine.read().is_ok_and(|e| e.is_some())
    }
    #[must_use]
    pub fn is_suspended(&self) -> bool {
        self.suspended.load(Ordering::SeqCst)
    }
    #[must_use]
    pub fn is_throttled(&self) -> bool {
        self.throttled.load(Ordering::SeqCst)
    }
    #[must_use]
    pub fn record(&self) -> Record {
        self.record.lock().map(|r| r.clone()).unwrap_or_default()
    }
    fn timed<T>(&self, f: impl FnOnce(&dyn Engine) -> Result<T, Fault>) -> Option<T> {
        let e = self.engine()?;
        let t = Instant::now();
        let out = f(e.as_ref());
        if let Ok(mut r) = self.record.lock() {
            r.engine_calls += 1;
            r.engine_ms += t.elapsed().as_secs_f64() * 1000.0;
            if out.is_err() {
                r.engine_faults += 1;
            }
        }
        out.ok()
    }
}

/// The four standard slots (router, writer, embedder, reranker): roles over the one
/// Qwen3-0.6B population (law 9; Luis Jake Gabriel III, 23:39 Finnish time, 2026-11-07).
pub struct Slots {
    pub by_name: BTreeMap<String, Slot>,
    /// Candidates the neural reranker may order, as set per question by the Map Coordinator's
    /// model-side budget (CM1, 2026-11-09); `usize::MAX` leaves the slot's own cap.
    cap: std::sync::atomic::AtomicUsize,
    /// Withhold floor on the qualified reranker's best score (its P(yes) for the released
    /// reranker), as f32 bits; 0 means none (2026-11-10, `serve --rerank-withhold`).
    withhold: std::sync::atomic::AtomicU32,
    /// The best neural score of the last reduce, as f32 bits (for reports).
    last_best: std::sync::atomic::AtomicU32,
    /// What the studied material says about the question being reduced (`Material`).
    material: std::sync::Mutex<Material>,
    /// The drones' judgment of each packet of the question being reduced, by packet id: its
    /// relevance from their calibrated evidence (`set_judged`); empty: the exact control judges.
    judged: std::sync::Mutex<BTreeMap<u64, f32>>,
}
impl Default for Slots {
    fn default() -> Self {
        let by_name =
            ["router", "writer", "embedder", "reranker"].into_iter().map(|n| (n.to_owned(), Slot::new(n, "qwen3-0.6b"))).collect();
        Self {
            by_name,
            cap: std::sync::atomic::AtomicUsize::new(usize::MAX),
            withhold: std::sync::atomic::AtomicU32::new(0),
            last_best: std::sync::atomic::AtomicU32::new(0),
            material: std::sync::Mutex::new(Material::default()),
            judged: std::sync::Mutex::new(BTreeMap::new()),
        }
    }
}
/// The slot name inside a node or slot id (`nmbrain.slot#reranker` or `reranker`).
fn slot_name(id: &str) -> &str {
    id.rsplit('#').next().unwrap_or(id)
}
impl Slots {
    fn slot(&self, name: &str) -> Result<&Slot, Fault> {
        let s = self.by_name.get(name).ok_or_else(|| Fault::new("nmbrain.slot", FaultKind::Config, format!("no slot {name}")))?;
        if let Ok(mut r) = s.record.lock() {
            r.control_calls += 1;
        }
        Ok(s)
    }
    /// Bind a neural engine to a slot.
    /// # Errors
    /// Unknown slot.
    pub fn bind(&self, name: &str, engine: Arc<dyn Engine>) -> Result<(), Fault> {
        let s = self.by_name.get(name).ok_or_else(|| Fault::new("nmbrain.slot", FaultKind::Config, format!("no slot {name}")))?;
        if let Ok(mut e) = s.engine.write() {
            *e = Some(engine);
        }
        s.suspended.store(false, Ordering::SeqCst);
        Ok(())
    }
    /// Homeostasis: suspend a slot's engine (the control keeps answering) and release its
    /// device views. Returns false for an id that names no slot.
    pub fn suspend(&self, id: &str) -> bool {
        let Some(s) = self.by_name.get(slot_name(id)) else { return false };
        s.suspended.store(true, Ordering::SeqCst);
        if let Some(e) = s.engine.read().ok().and_then(|e| e.clone()) {
            e.release_device();
        }
        true
    }
    /// Resume a suspended slot's engine.
    pub fn resume(&self, id: &str) -> bool {
        self.by_name.get(slot_name(id)).is_some_and(|s| {
            s.suspended.store(false, Ordering::SeqCst);
            true
        })
    }
    /// Homeostasis: throttle a slow slot (its engine sees half as many candidates).
    pub fn throttle(&self, id: &str) -> bool {
        self.by_name.get(slot_name(id)).is_some_and(|s| {
            s.throttled.store(true, Ordering::SeqCst);
            true
        })
    }
    /// Law 8: an engine decides only once qualified (it beat its control in a recorded run).
    /// Unqualified engines run beside the control and are recorded (shadow).
    #[must_use]
    pub fn qualified(&self, name: &str) -> bool {
        self.by_name.get(name).is_some_and(|s| s.qualified.load(Ordering::SeqCst))
    }
    /// Mark a slot's engine qualified or not (a Step Check result, never a default).
    pub fn set_qualified(&self, name: &str, q: bool) {
        if let Some(s) = self.by_name.get(name) {
            s.qualified.store(q, Ordering::SeqCst);
        }
    }
    /// Record one shadow comparison.
    pub fn note_shadow(&self, name: &str, agree: bool) {
        if let Some(Ok(mut r)) = self.by_name.get(name).map(|s| s.record.lock()) {
            r.shadow += 1;
            r.shadow_agree += u64::from(agree);
        }
    }
    /// Resident memory per bound engine: (slot id, RAM MB, GPU MB).
    #[must_use]
    pub fn resident(&self) -> Vec<(String, f32, f32)> {
        self.by_name
            .values()
            .filter_map(|s| {
                s.engine().map(|e| {
                    let (r, g) = e.resident_mb();
                    (format!("nmbrain.slot#{}", s.name), r, g)
                })
            })
            .collect()
    }
    /// Rerank score through the reranker slot's exact control (orders and admits).
    #[must_use]
    pub fn rerank(&self, q: &str, d: &str) -> f32 {
        self.slot("reranker").map_or(0.0, |s| s.control.rerank(q, d))
    }
    /// `rerank` for many documents in one control call (each still counts as a control call).
    /// Measured against the material set by `set_material`.
    #[must_use]
    pub fn rerank_all(&self, q: &str, docs: &[String]) -> Vec<f32> {
        self.rerank_all_as(q, docs, false)
    }
    /// `rerank_all` for document passages (`Exact::rerank_all_doc`).
    #[must_use]
    pub fn rerank_all_doc(&self, q: &str, docs: &[String]) -> Vec<f32> {
        self.rerank_all_as(q, docs, true)
    }
    fn rerank_all_as(&self, q: &str, docs: &[String], doc: bool) -> Vec<f32> {
        self.rerank_all_with(q, docs, doc, &self.material())
    }
    /// `rerank_all` (or `rerank_all_doc` when `doc`) measured against `m`, whatever
    /// `set_material` holds.
    #[must_use]
    pub fn rerank_all_with(&self, q: &str, docs: &[String], doc: bool, m: &Material) -> Vec<f32> {
        match self.by_name.get("reranker") {
            Some(s) => {
                if let Ok(mut r) = s.record.lock() {
                    r.control_calls += docs.len() as u64;
                }
                match (m.is_plain(), doc) {
                    (true, false) => s.control.rerank_all(q, docs),
                    (true, true) => s.control.rerank_all_doc(q, docs),
                    (false, _) => s.control.rerank_all_in(q, docs, doc, m),
                }
            }
            None => vec![0.0; docs.len()],
        }
    }
    /// Set what the studied material says about the question being reduced (the default clears
    /// it); the exact control's coverage is measured against it until it is cleared.
    pub fn set_material(&self, m: Material) {
        if let Ok(mut u) = self.material.lock() {
            *u = m;
        }
    }
    /// The material set by `set_material`.
    #[must_use]
    pub fn material(&self) -> Material {
        self.material.lock().map(|u| u.clone()).unwrap_or_default()
    }
    /// Set the drones' judgment of the packets of the question being reduced (empty clears it).
    pub fn set_judged(&self, j: BTreeMap<u64, f32>) {
        if let Ok(mut u) = self.judged.lock() {
            *u = j;
        }
    }
    /// The drones' judgment of the packets set by `set_judged`.
    #[must_use]
    pub fn judged(&self) -> BTreeMap<u64, f32> {
        self.judged.lock().map(|u| u.clone()).unwrap_or_default()
    }
    /// Set scores from the reranker engine, when it orders this set (see `Engine::rerank_set`).
    #[must_use]
    pub fn rerank_set_neural(&self, q: &str, docs: &[String]) -> Option<Vec<f32>> {
        self.by_name.get("reranker")?.timed(|e| e.rerank_set(q, docs)).flatten().filter(|v| v.len() == docs.len())
    }
    /// Neural rerank score when the reranker engine is bound and not suspended.
    #[must_use]
    pub fn rerank_neural(&self, q: &str, d: &str) -> Option<f32> {
        self.by_name.get("reranker")?.timed(|e| e.rerank(q, d))
    }
    /// How many candidates the neural reranker may order (halved when throttled).
    #[must_use]
    pub fn rerank_cap(&self, cap: usize) -> usize {
        let cap = cap.min(self.cap.load(Ordering::SeqCst));
        if self.by_name.get("reranker").is_some_and(Slot::is_throttled) { cap / 2 } else { cap }
    }
    /// Set the withhold floor on the qualified reranker's best score (0: none).
    pub fn set_withhold(&self, floor: f32) {
        self.withhold.store(floor.to_bits(), Ordering::SeqCst);
    }
    /// The withhold floor, when one is set.
    #[must_use]
    pub fn withhold_floor(&self) -> Option<f32> {
        let f = f32::from_bits(self.withhold.load(Ordering::SeqCst));
        (f > 0.0).then_some(f)
    }
    /// Record and read the best neural score of the last reduce.
    pub fn note_best(&self, s: f32) {
        self.last_best.store(s.to_bits(), Ordering::SeqCst);
    }
    #[must_use]
    pub fn last_best(&self) -> f32 {
        f32::from_bits(self.last_best.load(Ordering::SeqCst))
    }
    /// Set the coordinator's per-question cap on neural reranking (`usize::MAX`: none).
    pub fn set_rerank_cap(&self, cap: usize) {
        self.cap.store(cap, Ordering::SeqCst);
    }
    /// Jobs through the router slot.
    #[must_use]
    pub fn route(&self, q: &str) -> Vec<String> {
        self.slot("router").map(|s| s.control.route(q)).unwrap_or_default()
    }
    /// Vector through the embedder slot's exact control.
    #[must_use]
    pub fn embed(&self, t: &str) -> Vec<f32> {
        self.slot("embedder").map(|s| s.control.embed(t)).unwrap_or_default()
    }
    /// Neural embedding when the embedder engine is bound and not suspended.
    #[must_use]
    pub fn embed_neural(&self, t: &str) -> Option<Vec<f32>> {
        self.by_name.get("embedder")?.timed(|e| e.embed(t))
    }
    /// Engine milliseconds so far per slot (`nmbrain.slot#name`), for the homeostasis latency
    /// map (the slots run in the Core Brain, outside the forage frame's hops).
    #[must_use]
    pub fn engine_ms(&self) -> BTreeMap<String, f64> {
        self.by_name.values().map(|s| (format!("nmbrain.slot#{}", s.name), s.record().engine_ms)).collect()
    }
    /// The qualification record of every slot, for `nm_status` and run reports.
    #[must_use]
    pub fn qualification(&self) -> Value {
        Value::Object(
            self.by_name
                .values()
                .map(|s| {
                    let r = s.record();
                    let label =
                        s.engine.read().ok().and_then(|e| e.as_ref().map(|e| e.label())).unwrap_or_else(|| "exact control only".into());
                    let m: BTreeMap<String, Value> = [
                        ("engine".to_owned(), Value::String(label)),
                        ("suspended".to_owned(), Value::Bool(s.is_suspended())),
                        ("throttled".to_owned(), Value::Bool(s.is_throttled())),
                        ("qualified".to_owned(), Value::Bool(s.qualified.load(Ordering::SeqCst))),
                        ("shadow".to_owned(), Value::Number(r.shadow.to_string())),
                        ("shadow_top1_agree".to_owned(), Value::Number(r.shadow_agree.to_string())),
                        ("control_calls".to_owned(), Value::Number(r.control_calls.to_string())),
                        ("engine_calls".to_owned(), Value::Number(r.engine_calls.to_string())),
                        ("engine_ms".to_owned(), Value::Number(format!("{:.1}", r.engine_ms))),
                        ("engine_faults".to_owned(), Value::Number(r.engine_faults.to_string())),
                    ]
                    .into_iter()
                    .collect();
                    (s.name.clone(), Value::Object(m))
                })
                .collect(),
        )
    }
}
impl SlotView for Slots {
    fn call(&self, slot: &str, input: &Value) -> Result<Value, Fault> {
        let s = self.slot(slot)?;
        let text = |k: &str| input.get(k).and_then(Value::string).unwrap_or("").to_owned();
        Ok(match slot {
            "router" => Value::Array(s.control.route(&text("query")).into_iter().map(Value::String).collect()),
            "embedder" => Value::Array(s.control.embed(&text("text")).into_iter().map(|x| Value::Number(x.to_string())).collect()),
            "reranker" => Value::Number(s.control.rerank(&text("query"), &text("doc")).to_string()),
            _ => {
                let ev: Vec<String> = input
                    .get("evidence")
                    .and_then(Value::array)
                    .map(|a| a.iter().filter_map(|e| e.string().ok().map(str::to_owned)).collect())
                    .unwrap_or_default();
                Value::String(s.control.write(&text("prompt"), &ev))
            }
        })
    }
}

/// Self-check of the `slot` sub-component: the embedder is normalised and stable, the
/// reranker prefers the line naming the subject, the writer is extractive (it returns
/// given evidence, never new text), and the four slots answer through `SlotView`.
pub(crate) fn check() -> Result<(), String> {
    let v = embed("ParMap observe edits");
    let n: f32 = v.iter().map(|x| x * x).sum();
    if (n - 1.0).abs() > 1e-4 || v != embed("ParMap observe edits") {
        return Err("nmbrain.slot: embedder not normalised or not stable".into());
    }
    let c = Controls;
    if c.rerank("Where is ParMap?", "pub struct ParMap {") <= c.rerank("Where is ParMap?", "fn other() {}") {
        return Err("nmbrain.slot: reranker order".into());
    }
    let ev = vec!["a.rs:1 pub struct ParMap {".to_owned(), "b.rs:2 fn other() {}".to_owned()];
    let w = c.write("Where is ParMap?", &ev);
    if !w.lines().all(|l| ev.iter().any(|e| e == l)) || !w.contains("ParMap") {
        return Err("nmbrain.slot: writer is not extractive".into());
    }
    let slots = Slots::default();
    for name in ["router", "writer", "embedder", "reranker"] {
        slots.call(name, &Value::Null).map_err(|e| e.to_string())?;
    }
    Ok(())
}
