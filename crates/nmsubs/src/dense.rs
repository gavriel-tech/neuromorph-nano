//! The dense lane (2026-11-10, approved by Luis Jake Gabriel III): chunk vectors from the
//! released Qwen3-Embedding-0.6B, kept by chunk content hash in one file under `scratch/dense/`
//! (model outputs, not weights; a changed chunk gets a new hash and a new vector, the rest are
//! reused), and one fused ranking per question shared by every tentacle. The ranking joins the
//! embedding's order with the substrate's own lexical order by reciprocal rank (Cormack et al.
//! 2009); each tentacle then reports the ranked chunks inside its territory, so all territories
//! read by meaning in parallel and the GCX context map sees where the lane found code. The vectors come from a
//! function the Core Brain passes in (the embedder slot's engine); this crate holds no model.
//! Probe on the CX1 development half: `runs/2026-11-10-neumor-note-embedding-lane-probe.md`.
//!
//! RAM only on the served path (2026-11-10, refinement step 1, item 7; direction of Luis Jake
//! Gabriel III of 08:12 UTC that encodings are never written to disk, settled by
//! `docs/2026-11-10-neumor-analysis-g-vectors-and-g1-placement.md` question 1, option O3): the
//! served lane (`DenseLane::in_ram`) neither reads nor writes a store file. Three safeguards:
//! (a) its meaning ranking is used only once every current chunk has a vector; until then it
//! says so (`Stats::ready`, `coverage`) and the path runs without it, so no partial index ranks
//! silently; (b) the background build goes through the caller's device gate as a background job
//! (questions go first) and embeds first the files the memory system cited most recently
//! (`set_priority`); (c) vectors of chunks that no longer exist are dropped (`retain`) and a
//! changed chunk, having a new content key, is embedded again on sync. The file-backed form
//! (`open` with a path) is kept unchanged for the locked runner that uses it.
//!
//! Kept on disk (`keep_on_disk`, module `store`): when the caller gives the lane a project
//! memory folder, the vectors are also written there (content hash and vector only, at 16 bits
//! per value, under a header naming the embedder) and loaded at the next start, so only chunks
//! without a stored vector are embedded. Vectors held in RAM are then the stored values, so a
//! session that embedded a chunk and one that loaded it rank alike.
use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

#[path = "dense_store.rs"]
pub mod store;

/// A batch vector function: texts in, one vector per text out; `None` when the engine failed.
pub type BatchVectorize = dyn Fn(&[String]) -> Option<Vec<Vec<f32>>>;
/// A vector function (document or question); `None` when the engine is unavailable.
pub type Vectorize = Arc<dyn Fn(&str) -> Option<Vec<f32>> + Send + Sync>;

/// The embedding model card's query instruction for this lane (the handover form).
pub const QUERY_TASK: &str = "Given a question about a code repository, retrieve the code that answers it";
/// The instruction that says what the answer is (step 1l, form A1): the code or text that
/// defines, implements, configures or documents what the question describes.
pub const ANSWER_TASK: &str =
    "Given a question about a code repository, retrieve the code or text that defines, implements, configures or documents what the question describes";
/// The instruction for a request of `kind` (the request's drone: `locate`, `value`, `explain`,
/// `trace`, `absent-check`; step 1l, form A2). Any other kind gets `ANSWER_TASK`.
#[must_use]
pub fn kind_task(kind: &str) -> &'static str {
    match kind {
        "locate" => "Given a question about a code repository, retrieve the code or text that defines or implements what the question names",
        "value" => "Given a question about a code repository, retrieve the line that sets the value, default, limit or setting the question asks for",
        "explain" => "Given a question about a code repository, retrieve the code or documentation that shows how or why what the question describes works",
        "trace" => "Given a question about a code repository, retrieve the code along the path the question follows: where it starts, what it calls, where it ends",
        "absent-check" => "Given a question about a code repository, retrieve the code or text that would implement or document what the question asks about",
        _ => ANSWER_TASK,
    }
}
/// Which instruction the question is encoded under: 0 the handover's `QUERY_TASK`, 1
/// `ANSWER_TASK`, 2 by the request's kind (`kind_task` of `KIND_OF`). Default 2 (step 1l, change
/// A, kept: pooled over the four lane sets, pack 120 to 127 and every range 101 to 108 of 142,
/// p 0.016 each, nothing fell); `NANO_DENSE_TASK=handover` restores the handover form.
pub static TASK_FORM: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
/// The request's kind as its drone reads it (bound by the caller, which holds the drone).
pub static KIND_OF: std::sync::OnceLock<fn(&str) -> &'static str> = std::sync::OnceLock::new();
/// The instruction `question` is encoded under in the form `TASK_FORM` holds.
#[must_use]
pub fn task_for(question: &str) -> &'static str {
    match TASK_FORM.load(std::sync::atomic::Ordering::Relaxed) {
        1 => ANSWER_TASK,
        2 => KIND_OF.get().map_or(ANSWER_TASK, |k| kind_task(k(question))),
        _ => QUERY_TASK,
    }
}
/// The text the embedder encodes for `question` under `task`.
fn posed(task: &str, question: &str) -> String {
    format!("Instruct: {task}\nQuery:{question}")
}
/// Provenance origin of chunks the lane put in a pack (the Core Brain admits them by rank).
pub const ORIGIN: &str = "nmsubs.dense#g0";
/// The meaning lane's form in the readers' views. Off (the first form, 2026-11-10): the lane's
/// fused place orders its chunks (`dense:<place>:` keys, confidence from the place). On (2026-11-17,
/// master design: ranking stays with the evidence and G2): the nearest chunks by meaning alone are
/// added as candidates (`chunk:` keys with their lexical score), so the drones' evidence and the
/// Core Brain judge and order them as any other candidate.
pub static CANDIDATES_ONLY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The meaning signal in the drones' evidence (`cosines`): each piece's cosine with the question,
/// calibrated against the content-free questions', is one more signal of the evidence's meaning
/// family. Off by default (it did not pass its measurement); on for a measurement.
pub static MEANING_SIGNAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The content-free questions the meaning signal is calibrated against (the judge's forms).
pub const CONTENT_FREE: [&str; 3] = ["N/A", "", "[MASK]"];
/// The background build runs (default). Off only for a measurement that holds the lane at the
/// coverage it loaded (`NANO_DENSE_FILL=off`).
pub static FILL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
/// Vectors loaded from the store at start, at most (a measurement of the first start at a given
/// coverage, `NANO_DENSE_LOAD=<share>`); taken in build order. `usize::MAX`: all.
pub static LOAD_LIMIT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);
/// At partial coverage the fused form is used for a question whose lexical head has every vector
/// (`DenseLane::partial_now`); off by default, `NANO_DENSE_FUSED=head` for a measurement.
pub static HEAD_RULE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Candidates each order contributes to the fusion.
pub const DEPTH: usize = 50;
/// Reciprocal rank constant.
pub const RRF_K: f32 = 60.0;
/// Weight of the lexical order in the fusion (the embedding order weighs 1).
pub const LEXICAL_WEIGHT: f32 = 0.3;
/// Fused places a tentacle may report from (the lane's whole-tree shortlist).
pub const TOP: usize = 6;

/// The fusion's numbers for a measurement (step 1l, change B): each fixed value may be named, or
/// derived per request. `None` keeps the constant.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tune {
    /// `LEXICAL_WEIGHT` named.
    pub lexical_weight: Option<f32>,
    /// `RRF_K` named.
    pub rrf_k: Option<f32>,
    /// `TOP` named.
    pub top: Option<usize>,
    /// `views::DENSE_K` named.
    pub dense_k: Option<usize>,
    /// The lexical weight derived per request from how strongly the material holds the request's
    /// words (`word_hold`).
    pub derive_weight: bool,
    /// The shortlist derived per request: the pieces whose cosine with the question stands above
    /// the lane's own content-free level (`above_level`); each territory reports every shortlisted
    /// piece it holds.
    pub derive_top: bool,
}
/// The fusion's numbers in force (`Tune`).
pub static TUNE: RwLock<Tune> = RwLock::new(Tune {
    lexical_weight: None,
    rrf_k: None,
    top: None,
    dense_k: None,
    derive_weight: false,
    derive_top: false,
});
/// The numbers in force.
#[must_use]
pub fn tune() -> Tune {
    TUNE.read().map(|t| *t).unwrap_or_default()
}
/// File the derived numbers of each request are appended to, for a measurement (`NANO_DENSE_TRACE`).
pub static TRACE: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// The recipe of the text a chunk is embedded from (`doc_text_with`), named in the header of
/// stored vectors; change it whenever that text changes.
pub const RECIPE: &str = "path line with heading path, then the chunk's lines; version 1";

/// The recipe of a code window embedded with its lead (step 1l, change C), named in the header
/// so vectors stored under the first recipe are never mixed with these. Roadmap Phase 4
/// (2026-11-24): each stored row names its recipe (`store`), so switching between the two embeds
/// again only the rows of pieces that have a lead.
pub const RECIPE_LEAD: &str =
    "path line with heading path; a code window's lead (doc comment and first line of the definition it sits in); then the chunk's lines; version 2";
/// Code windows are embedded with their lead (`RECIPE_LEAD`).
pub static WITH_LEAD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The recipe in force.
#[must_use]
pub fn recipe() -> &'static str {
    if WITH_LEAD.load(std::sync::atomic::Ordering::Relaxed) { RECIPE_LEAD } else { RECIPE }
}
/// The text the lane embeds for `c` of the file at `path`, under the recipe in force.
#[must_use]
pub fn embed_text(path: &str, c: &crate::Chunk) -> String {
    if WITH_LEAD.load(std::sync::atomic::Ordering::Relaxed) && !c.lead.is_empty() {
        format!("{path}
{}
{}", c.lead, c.text)
    } else {
        doc_text_with(path, &c.heading, &c.text)
    }
}

/// Keys of the pieces of `sub` that have a lead (Roadmap Phase 4 (2026-11-24)): only their text
/// differs between `RECIPE` and `RECIPE_LEAD`, so only their stored rows depend on that switch.
fn with_lead(sub: &crate::Substrate) -> std::collections::BTreeSet<&str> {
    sub.chunks.iter().filter(|c| !c.lead.is_empty()).map(|c| c.key.as_str()).collect()
}

/// The text a chunk is embedded from: its path line, then its lines (as in the probe).
#[must_use]
pub fn doc_text(path: &str, text: &str) -> String {
    format!("{path}\n{text}")
}

/// The text embedded for a chunk with its heading path (the document title and heading path are
/// the chunk's context, known when it is read): `path | heading` then the text. With no heading
/// (every code chunk) it is `doc_text` exactly.
#[must_use]
pub fn doc_text_with(path: &str, heading: &str, text: &str) -> String {
    if heading.is_empty() { doc_text(path, text) } else { format!("{path} | {heading}\n{text}") }
}

/// What the lane has done, for `nm_status` and the trace.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stats {
    pub vectors: usize,
    pub embedded_now: usize,
    pub index_ms: f64,
    pub queries: usize,
    pub query_ms: f64,
    pub faults: usize,
    /// RAM-only lane (refinement step 1): every current chunk has a vector (the ranking is used
    /// only then); chunks covered of the current total; questions answered without the ranking
    /// because it was not ready; vectors dropped for chunks that no longer exist; wall time from
    /// the start of the build to full coverage.
    pub ready: bool,
    pub covered: usize,
    pub total: usize,
    pub not_ready: usize,
    pub dropped: usize,
    pub full_ms: Option<f64>,
    /// Questions ranked at partial coverage (`PARTIAL`), and the wall time of the first.
    pub partial: usize,
    pub first_rank_ms: Option<f64>,
    /// Kept on disk: vectors loaded from the store at start, rows in the store, and what the
    /// store reported (empty when the lane keeps nothing on disk).
    pub loaded: usize,
    pub stored: usize,
    pub store_note: String,
    /// Dead-skin barrier layer (H32): vectors revived for free on undo/revert and rows shed.
    pub revived: usize,
    pub shed: usize,
    /// The embedder's served values moved away from those the vectors were made under
    /// (`DenseLane::mark_stale`): the reason, empty while they are current; and the questions
    /// answered without the meaning ranking because of it.
    pub stale: String,
    pub stale_skips: usize,
}

/// The fused ranking of one question on one substrate version: chunk index to fused place.
#[derive(Clone, Debug, Default)]
pub struct Ranks {
    pub version: u64,
    pub question: String,
    pub place: BTreeMap<usize, usize>,
}

/// One question's cosines, best first, with the version, the question and the vectors held then.
type Scored = (u64, String, usize, Arc<Vec<(f32, usize)>>);
/// Every chunk's cosine with each of several questions, in chunk order.
type Cosines = Arc<Vec<Vec<f32>>>;

/// Stage 5 WP-C4: the most auxiliary vectors held per key.
pub const AUX_PER_KEY: usize = 3;

/// The lane: vectors by chunk hash, the store file, the two vector functions, the last ranking.
pub struct DenseLane {
    vecs: RwLock<BTreeMap<String, Arc<Vec<f32>>>>,
    file: Option<PathBuf>,
    doc: Vectorize,
    query: Vectorize,
    ranks: Mutex<Option<Ranks>>,
    /// The last question's order by embedding cosine alone (`meaning_ranks`).
    meaning: Mutex<Option<Ranks>>,
    /// The last question's cosines, best first, with its version and text: the question is
    /// encoded once per call however many readers ask.
    scores: Mutex<Option<Scored>>,
    /// Every chunk's cosine with each content-free question, in chunk order, per version and
    /// instruction.
    content_free: Mutex<Vec<(u64, String, Cosines)>>,
    /// The content-free questions' vectors per instruction (they depend on neither the request nor
    /// the material): encoded once, at the lane's first build, for every instruction.
    null_queries: Mutex<BTreeMap<String, Arc<Vec<Vec<f32>>>>>,
    /// One embedding job at a time (indexing and a question share the engine).
    jobs: Mutex<()>,
    /// Roadmap Phase 0 item 3 (2026-11-23): the epoch of the transform the query side passes
    /// through (the project's lens, `set_query_epoch`), and the epoch the query-side caches were
    /// made under. When they differ, the content-free vectors, their cosines, the last question's
    /// vector and scores are made again: the lens seated after the note band had cached the
    /// content-free vectors unflattened, so every question was calibrated against vectors of
    /// another transform (large code first 27 to 23 with the lens on; drones' meaning signal
    /// val08 28 to 61).
    epoch_of: RwLock<Option<Arc<dyn Fn() -> u64 + Send + Sync>>>,
    /// Roadmap Phase 0 item 3 (2026-11-23): the project's lens on the question's vector, applied
    /// here rather than inside `query` so the lane holds both readings: the lensed cosines give the
    /// order, mapped onto the plain cosines' mean and spread for the same question (the scale every
    /// reader downstream was built on). Applied inside `query`, the flattening lowered every cosine
    /// by about 0.1 (0.6711 to 0.5533 at the top) with the order nearly the same, and the served
    /// flow lost (large code first 28 to 23) though the lane's order gained offline.
    lens_fn: RwLock<Option<QueryLens>>,
    /// The lens's flattening on a piece's vector (both sides flattened; `set_doc_lens`), taken
    /// once a question.
    doc_lens: RwLock<Option<DocLens>>,
    /// The pieces' flattened vectors under the lens of one epoch (made once an epoch, about a
    /// second for 11,126 pieces, then read by every question).
    flat_cache: RwLock<Option<(u64, usize, FlatPieces)>>,
    epoch_seen: std::sync::atomic::AtomicU64,
    /// The last question's vector, by the text encoded: a question is encoded once however often
    /// it is scored while the build adds vectors.
    qvec: Mutex<Option<(String, Arc<Vec<f32>>)>>,
    /// Stage 5 WP-C4 E1/E4: auxiliary vectors per piece key (the writer's checked notes and likely
    /// names, encoded as documents through `doc`, so on the pieces' own scale) and per structure node
    /// (`module:<path>`). A piece's cosine with a question is the largest of its own and its notes'
    /// (`NANO_AUX_NOTES=off` reads the pieces' own alone).
    aux: RwLock<BTreeMap<String, Vec<Arc<Vec<f32>>>>>,
    pub stats: Mutex<Stats>,
    /// RAM-only lane: the ranking waits for full coverage; build order from `priority`.
    ram_only: bool,
    priority: RwLock<Vec<String>>,
    /// Bumped whenever `priority` changes, so a running build re-orders what is left.
    priority_gen: std::sync::atomic::AtomicU64,
    /// The first start (`set_partial`).
    partial: std::sync::atomic::AtomicBool,
    started: std::time::Instant,
    /// The on-disk store, when the caller gave one (`keep_on_disk`).
    store: Mutex<Option<store::Store>>,
    /// What made the vectors held, in the store's header form (`keep_on_disk`, `set_made_by`);
    /// empty when the caller named nothing.
    made_by: RwLock<store::Header>,
    /// Set by `mark_stale`: the reason, and whether the question side still encodes with the values
    /// that made the vectors (only a change of the piece-side recipe leaves it so).
    stale: RwLock<Option<(String, bool)>>,
    /// The texts behind `aux`, per key (at most `AUX_PER_KEY`, newest last), so a lane built for
    /// new embedder values encodes them again (`aux_texts`). RAM only.
    aux_src: RwLock<BTreeMap<String, Vec<String>>>,
}

impl std::fmt::Debug for DenseLane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DenseLane").field("file", &self.file).field("vectors", &self.len()).finish_non_exhaustive()
    }
}

impl DenseLane {
    /// Open the lane over a store file (read now when it exists; `None`: memory only).
    #[must_use]
    pub fn open(file: Option<PathBuf>, doc: Vectorize, query: Vectorize) -> Self {
        let mut vecs = BTreeMap::new();
        if let Some(bytes) = file.as_ref().and_then(|f| std::fs::read(f).ok()) {
            for (k, v) in decode(&bytes) {
                vecs.insert(k, Arc::new(v));
            }
        }
        let n = vecs.len();
        Self {
            vecs: RwLock::new(vecs),
            file,
            doc,
            query,
            ranks: Mutex::new(None),
            meaning: Mutex::new(None),
            scores: Mutex::new(None),
            content_free: Mutex::new(Vec::new()),
            null_queries: Mutex::new(BTreeMap::new()),
            jobs: Mutex::new(()),
            epoch_of: RwLock::new(None),
            lens_fn: RwLock::new(None),
            doc_lens: RwLock::new(None),
            flat_cache: RwLock::new(None),
            epoch_seen: std::sync::atomic::AtomicU64::new(0),
            qvec: Mutex::new(None),
            aux: RwLock::new(BTreeMap::new()),
            stats: Mutex::new(Stats { vectors: n, ..Stats::default() }),
            ram_only: false,
            priority: RwLock::new(Vec::new()),
            priority_gen: std::sync::atomic::AtomicU64::new(0),
            partial: std::sync::atomic::AtomicBool::new(false),
            started: std::time::Instant::now(),
            store: Mutex::new(None),
            made_by: RwLock::new(store::Header::new()),
            stale: RwLock::new(None),
            aux_src: RwLock::new(BTreeMap::new()),
        }
    }
    /// Names what made the vectors held (the store's header form: weights, values, precision,
    /// recipe). `keep_on_disk` names it too.
    pub fn set_made_by(&self, header: store::Header) {
        if let Ok(mut h) = self.made_by.write() {
            *h = header;
        }
    }
    /// What made the vectors held (empty when never named).
    #[must_use]
    pub fn made_by(&self) -> store::Header {
        self.made_by.read().map(|h| h.clone()).unwrap_or_default()
    }
    /// The first field in which `running` (the embedder serving now) differs from what made the
    /// vectors held, as `field: held -> running`; `None` when they are equal or nothing was named.
    #[must_use]
    pub fn differs_from(&self, running: &store::Header) -> Option<String> {
        let held = self.made_by();
        if held.is_empty() {
            return None;
        }
        held.keys()
            .chain(running.keys())
            .find(|k| held.get(*k) != running.get(*k))
            .map(|k| format!("{k}: {} -> {}", held.get(k).map_or("absent", String::as_str), running.get(k).map_or("absent", String::as_str)))
    }
    /// Whether a change from what made the vectors to `running` leaves the question side as it
    /// was: only the recipe of the piece text differs (`store::RECIPE_FIELDS`), so a question
    /// encoded now still lies in the space of the vectors held.
    #[must_use]
    pub fn query_side_holds(&self, running: &store::Header) -> bool {
        let held = self.made_by();
        held.keys().chain(running.keys()).filter(|k| held.get(*k) != running.get(*k)).all(|k| store::RECIPE_FIELDS.contains(&k.as_str()))
    }
    /// Marks the vectors held stale: the embedder's served values moved since they were made
    /// (`reason` says how). The lane is never filled again (a lane built for the new values
    /// replaces it); its store is written once more and let go, so that lane owns the folder.
    /// Unless `query_side_holds` (the question side still encodes in the vectors' space), the lane
    /// gives no ranking, order, cosines or piece vectors and `ready` is false, so every reader runs
    /// on the lexical and structural paths alone. Idempotent; the first reason is kept.
    pub fn mark_stale(&self, reason: &str, query_side_holds: bool) {
        let Ok(mut s) = self.stale.write() else { return };
        if s.is_some() {
            return;
        }
        *s = Some((reason.to_owned(), query_side_holds));
        drop(s);
        self.flush_store();
        if let Ok(mut st) = self.store.lock() {
            *st = None;
        }
        if let Ok(mut c) = self.scores.lock() {
            *c = None;
        }
        if let Ok(mut c) = self.ranks.lock() {
            *c = None;
        }
        if let Ok(mut c) = self.meaning.lock() {
            *c = None;
        }
        if let Ok(mut st) = self.stats.lock() {
            st.stale = reason.to_owned();
        }
    }
    /// The reason the vectors held are stale (`mark_stale`), if they are.
    #[must_use]
    pub fn stale(&self) -> Option<String> {
        self.stale.read().ok().and_then(|s| s.as_ref().map(|(r, _)| r.clone()))
    }
    /// The lane is stale and its vectors are not in the question side's space: it serves nothing.
    fn withheld(&self) -> bool {
        self.stale.read().is_ok_and(|s| s.as_ref().is_some_and(|(_, holds)| !holds))
    }
    /// The texts of the auxiliary vectors held, per key (`add_aux_text`), for a lane built for
    /// new embedder values to encode again.
    #[must_use]
    pub fn aux_texts(&self) -> Vec<(String, String)> {
        self.aux_src.read().map(|a| a.iter().flat_map(|(k, v)| v.iter().map(move |t| (k.clone(), t.clone()))).collect()).unwrap_or_default()
    }
    /// The RAM-only lane (refinement step 1, item 7): no store file is read or written; the
    /// ranking is used only at full coverage of the current chunks (module notes).
    #[must_use]
    pub fn in_ram(doc: Vectorize, query: Vectorize) -> Self {
        Self { ram_only: true, ..Self::open(None, doc, query) }
    }
    /// The epoch of the transform the query side passes through (it changes when that transform
    /// does); the query-side caches follow it.
    pub fn set_query_epoch(&self, f: Arc<dyn Fn() -> u64 + Send + Sync>) {
        if let Ok(mut e) = self.epoch_of.write() {
            *e = Some(f);
        }
    }
    /// The lens on the question's vector (`lens_fn`): `None` from `f` when no lens is in force.
    pub fn set_query_lens(&self, f: QueryLens) {
        if let Ok(mut l) = self.lens_fn.write() {
            *l = Some(f);
        }
    }
    /// The lens's flattening on the pieces' vectors (`doc_lens`).
    pub fn set_doc_lens(&self, f: DocLens) {
        if let Ok(mut l) = self.doc_lens.write() {
            *l = Some(f);
        }
    }
    /// Every piece's vector flattened by the lens in force (index by chunk), made once per lens
    /// epoch and chunk count; `None` without a piece-side lens.
    fn flat_pieces(&self, sub: &crate::Substrate) -> Option<FlatPieces> {
        let e = self.epoch_of.read().ok().and_then(|f| f.as_ref().map(|f| f())).unwrap_or(0);
        let n = sub.chunks.len();
        if let Some((ce, cn, v)) = self.flat_cache.read().ok().and_then(|c| c.clone())
            && ce == e
            && cn == n
        {
            return Some(v);
        }
        let f = self.doc_lens.read().ok().and_then(|d| d.as_ref().and_then(|d| d()))?;
        let v: FlatPieces = Arc::new(sub.chunks.iter().map(|c| self.vector(&c.key).map(|x| f(&x))).collect());
        if let Ok(mut c) = self.flat_cache.write() {
            *c = Some((e, n, Arc::clone(&v)));
        }
        Some(v)
    }
    /// Clears the query-side caches when the query transform's epoch moved since they were made.
    /// The content-free caches are taken (never held by this thread at a sync point); the per-question
    /// caches only if free (`try_lock`): they (ranks, meaning, scores,
    /// the question's vector) can be held by this very thread at a sync point (the ranks are built
    /// under their lock and score through here: a blocking lock deadlocked the first question after
    /// the lens seated, 2026-11-23); they are made again with the next question. The content-free
    /// vectors and cosines, which the calibration reads, are never held at a sync point.
    fn sync_epoch(&self) {
        let Some(e) = self.epoch_of.read().ok().and_then(|f| f.as_ref().map(|f| f())) else { return };
        if self.epoch_seen.swap(e, std::sync::atomic::Ordering::SeqCst) == e {
            return;
        }
        if let Ok(mut c) = self.null_queries.lock() {
            c.clear();
        }
        if let Ok(mut c) = self.content_free.lock() {
            c.clear();
        }
        if let Ok(mut c) = self.qvec.try_lock() {
            *c = None;
        }
        if let Ok(mut c) = self.scores.try_lock() {
            *c = None;
        }
        if let Ok(mut c) = self.ranks.try_lock() {
            *c = None;
        }
        if let Ok(mut c) = self.meaning.try_lock() {
            *c = None;
        }
    }
    /// Build order: these files first, in this order (the memory system's most recent
    /// citations), then every other chunk in substrate order.
    pub fn set_priority(&self, files: Vec<String>) {
        if let Ok(mut p) = self.priority.write() {
            *p = files;
        }
        self.priority_gen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    /// The first start (off by default): the RAM-only lane ranks with the vectors it has (pieces
    /// without one are reached by words alone, through the lexical order and the other evidence),
    /// and each question moves the files its lexical order names to the front of the build order.
    /// The stats count the questions ranked at partial coverage.
    pub fn set_partial(&self, on: bool) {
        self.partial.store(on, std::sync::atomic::Ordering::Relaxed);
    }
    /// The first start is on and the lane's places are not yet comparable for `question`: the lane
    /// does not hold every current piece's vector and (with `HEAD_RULE`) some piece of the
    /// question's lexical head (the fusion's first `DEPTH` places by words) has none.
    #[must_use]
    pub fn partial_now(&self, sub: &crate::Substrate, question: &str) -> bool {
        if !(self.ram_only && self.partial.load(std::sync::atomic::Ordering::Relaxed) && !self.ready(sub)) {
            return false;
        }
        if !HEAD_RULE.load(std::sync::atomic::Ordering::Relaxed) {
            return true;
        }
        let Ok(v) = self.vecs.read() else { return true };
        !sub.lexical_order(question).into_iter().take(DEPTH).all(|i| sub.chunks.get(i).is_some_and(|c| v.contains_key(&c.key)))
    }
    /// Move `files` to the front of the build order (a question's likely files), keeping the rest
    /// in their order.
    pub fn prefer(&self, files: &[String]) {
        let Ok(mut p) = self.priority.write() else { return };
        if p.len() >= files.len() && p.iter().zip(files).all(|(a, b)| a == b) {
            return;
        }
        let mut next: Vec<String> = Vec::with_capacity(p.len() + files.len());
        for f in files.iter().chain(p.iter()) {
            if !next.contains(f) {
                next.push(f.clone());
            }
        }
        *p = next;
        drop(p);
        self.priority_gen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    /// Sort `todo` (key, text, path) by the build order in force.
    fn order_todo(&self, todo: &mut [(String, String, String)]) {
        let pri = self.priority.read().map(|p| p.clone()).unwrap_or_default();
        if !pri.is_empty() {
            let at: BTreeMap<&str, usize> = pri.iter().enumerate().rev().map(|(i, p)| (p.as_str(), i)).collect();
            todo.sort_by_key(|(_, _, path)| at.get(path.as_str()).copied().unwrap_or(usize::MAX));
        }
    }
    /// Keep the vectors on disk in `dir` (a project's memory folder's `dense/`): read the store
    /// written for an embedder described by `header`, load the vectors of `sub`'s chunks, and
    /// write every vector embedded from now on. A store made by another embedder or a damaged
    /// file is not used (the line says why) and is replaced at the first write. Returns one line
    /// saying what was found.
    /// Roadmap Phase 4 (2026-11-24): each row names its recipe; a row whose recipe gives another
    /// text for its piece than the running recipe (`store::Store::current`) is not loaded, so the
    /// fill embeds that piece again and the store appends its new row. Rows of the running
    /// recipe, and rows of a piece with no lead under either text recipe, are loaded as they are.
    pub fn keep_on_disk(&self, dir: &std::path::Path, header: store::Header, sub: &crate::Substrate) -> String {
        let live: std::collections::BTreeSet<&str> = sub.chunks.iter().map(|c| c.key.as_str()).collect();
        let t = std::time::Instant::now();
        self.set_made_by(header.clone());
        let (mut st, mut got, found) = store::Store::open(dir, header, &live);
        let leads = with_lead(sub);
        got.retain(|(k, _)| st.current(k, leads.contains(k.as_str())));
        let again = live.iter().filter(|k| st.row_recipe(k).is_some() && !st.current(k, leads.contains(*k))).count();
        st.note_file_tags(
            sub.chunks
                .iter()
                .filter_map(|c| sub.files.get(c.file).map(|f| (c.key.as_str(), store::file_tag(&f.path)))),
        );
        let limit = LOAD_LIMIT.load(std::sync::atomic::Ordering::Relaxed);
        if limit < got.len() {
            let mut todo: Vec<(String, String, String)> =
                sub.chunks.iter().filter_map(|c| sub.files.get(c.file).map(|f| (c.key.clone(), String::new(), f.path.replace('\\', "/")))).collect();
            self.order_todo(&mut todo);
            let place: BTreeMap<&str, usize> = todo.iter().enumerate().rev().map(|(i, t)| (t.0.as_str(), i)).collect();
            got.sort_by_key(|(k, _)| place.get(k.as_str()).copied().unwrap_or(usize::MAX));
            got.truncate(limit);
        }
        let n = got.len();
        if let Ok(mut m) = self.vecs.write() {
            for (k, v) in got {
                m.insert(k, Arc::new(v));
            }
        }
        let mut line = format!("{} in {:.0} ms ({})", found.line(), t.elapsed().as_secs_f64() * 1000.0, dir.display());
        if again > 0 {
            line.push_str(&format!("; {again} rows of current pieces made under another recipe than the running one, to be embedded again; {n} vectors loaded"));
        }
        let rows = st.rows();
        if let Ok(mut s) = self.store.lock() {
            *s = Some(st);
        }
        if let Ok(mut s) = self.stats.lock() {
            s.loaded = n;
            s.stored = rows;
            s.vectors = self.len();
            s.store_note.clone_from(&line);
        }
        self.note_coverage(sub);
        line
    }
    /// The lane keeps its vectors on disk.
    #[must_use]
    pub fn keeps_on_disk(&self) -> bool {
        self.store.lock().is_ok_and(|s| s.is_some())
    }
    /// A new vector for RAM: the stored value when the lane keeps vectors on disk (the store
    /// then holds the row, written with the next batch).
    fn admit(&self, key: &str, v: Vec<f32>, path: Option<&str>) -> Vec<f32> {
        let Ok(mut s) = self.store.lock() else { return v };
        match s.as_mut() {
            Some(st) => {
                let tag = path.map(store::file_tag);
                st.add_tagged(key, &v, tag.as_deref());
                store::rounded(&v)
            }
            None => v,
        }
    }
    /// Write the rows waiting in the store, and note the store's state.
    fn flush_store(&self) {
        let Ok(mut s) = self.store.lock() else { return };
        let Some(st) = s.as_mut() else { return };
        st.flush();
        let (rows, fault) = (st.rows(), st.fault.clone());
        drop(s);
        if let Ok(mut s) = self.stats.lock() {
            s.stored = rows;
            if let Some(f) = fault {
                s.store_note = format!("store write failed: {f}");
            }
        }
    }
    /// (chunks of `sub` with a vector, chunks of `sub`).
    #[must_use]
    pub fn coverage(&self, sub: &crate::Substrate) -> (usize, usize) {
        let v = self.vecs.read();
        let have = v.as_ref().map_or(0, |m| sub.chunks.iter().filter(|c| m.contains_key(&c.key)).count());
        (have, sub.chunks.len())
    }
    /// The ranking may be used: always for the file-backed form; for the RAM-only lane once every
    /// current chunk has a vector.
    #[must_use]
    pub fn ready(&self, sub: &crate::Substrate) -> bool {
        if self.withheld() {
            return false;
        }
        if !self.ram_only {
            return true;
        }
        let (have, total) = self.coverage(sub);
        total > 0 && have == total
    }
    /// Drop vectors of chunks `sub` no longer holds (RAM-only lane), move them into the
    /// dead-skin barrier layer on disk (H32), revive any barrier vectors that reappeared in `sub`
    /// without GPU re-embedding, and run one bounded micro-shed step; returns how many vectors
    /// were dropped from RAM.
    pub fn retain(&self, sub: &crate::Substrate) -> usize {
        if !self.ram_only {
            return 0;
        }
        let live: std::collections::BTreeSet<&str> = sub.chunks.iter().map(|c| c.key.as_str()).collect();
        let live_file_tags: std::collections::BTreeSet<String> = sub.files.iter().map(|f| store::file_tag(&f.path)).collect();
        let dropped = self.vecs.write().map_or(0, |mut m| {
            let before = m.len();
            m.retain(|k, _| live.contains(k.as_str()));
            before - m.len()
        });
        let (revived_vecs, shed_rows, stored_rows) = self
            .store
            .lock()
            .ok()
            .and_then(|mut s| {
                let st = s.as_mut()?;
                let rep = st.shed(&live, Some(&live_file_tags), nmtype::now_ms());
                // Roadmap Phase 4 (2026-11-24): a revived row of another recipe whose text differs
                // for its piece is embedded again by the fill, not revived.
                let leads = with_lead(sub);
                let revived: Vec<(String, Vec<f32>)> = rep.revived.into_iter().filter(|(k, _)| st.current(k, leads.contains(k.as_str()))).collect();
                Some((revived, rep.shed_rows, st.rows()))
            })
            .unwrap_or_default();
        let revived_n = revived_vecs.len();
        if !revived_vecs.is_empty()
            && let Ok(mut m) = self.vecs.write()
        {
            for (k, v) in revived_vecs {
                m.entry(k).or_insert_with(|| Arc::new(v));
            }
        }
        if let Ok(mut s) = self.stats.lock() {
            s.dropped += dropped;
            s.revived += revived_n;
            s.shed += shed_rows;
            if stored_rows > 0 || shed_rows > 0 {
                s.stored = stored_rows;
            }
            s.vectors = self.len();
        }
        dropped
    }
    /// Bytes of vectors held in RAM (counted in the `content` memory area as `dense_vectors`).
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.vecs.read().map_or(0, |m| m.iter().map(|(k, v)| k.len() + v.len() * 4).sum())
    }
    /// Vectors held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.vecs.read().map(|v| v.len()).unwrap_or(0)
    }
    /// No vectors yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The vector of a chunk hash, when indexed.
    #[must_use]
    pub fn vector(&self, key: &str) -> Option<Arc<Vec<f32>>> {
        if self.withheld() {
            return None;
        }
        self.vecs.read().ok()?.get(key).cloned()
    }
    /// Embed the chunks of `sub` that have no vector yet, one at a time, appending each to the
    /// store file; returns how many were embedded. Stops early when `keep_going` says so.
    pub fn fill(&self, sub: &crate::Substrate, keep_going: &dyn Fn() -> bool) -> usize {
        if self.stale().is_some() {
            return 0;
        }
        self.warm_instructions();
        if !FILL.load(std::sync::atomic::Ordering::Relaxed) {
            return 0;
        }
        let mut todo: Vec<(String, String, String)> = sub
            .chunks
            .iter()
            .filter(|c| self.vector(&c.key).is_none())
            .filter_map(|c| sub.files.get(c.file).map(|f| (c.key.clone(), embed_text(&f.path, c), f.path.replace('\\', "/"))))
            .collect();
        // RAM-only lane: the memory system's most recently cited files first (stable otherwise).
        if self.ram_only {
            let pri = self.priority.read().map(|p| p.clone()).unwrap_or_default();
            if !pri.is_empty() {
                todo.sort_by_key(|(_, _, path)| pri.iter().position(|p| p == path).unwrap_or(usize::MAX));
            }
        }
        let mut done = 0;
        let mut seen = std::collections::BTreeSet::new();
        let mut held = self.bytes();
        for (key, text, path) in todo {
            if !keep_going() || content_area_full(held) {
                break;
            }
            if !seen.insert(key.clone()) || self.vector(&key).is_some() {
                continue;
            }
            let t = std::time::Instant::now();
            let v = {
                let _g = self.jobs.lock();
                (self.doc)(&text)
            };
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let Some(v) = v.filter(|v| !v.is_empty()) else {
                if let Ok(mut s) = self.stats.lock() {
                    s.faults += 1;
                }
                continue;
            };
            if let Some(f) = &self.file {
                let _ = std::fs::OpenOptions::new().create(true).append(true).open(f).and_then(|mut h| h.write_all(&encode(&key, &v)));
            }
            let v = self.admit(&key, v, Some(&path));
            held += key.len() + v.len() * 4;
            if let Ok(mut m) = self.vecs.write() {
                m.insert(key, Arc::new(v));
            }
            done += 1;
            if let Ok(mut s) = self.stats.lock() {
                s.vectors = self.len();
                s.embedded_now += 1;
                s.index_ms += ms;
            }
        }
        nmtype::limits::set_area_used("content", "dense_vectors", held as u64);
        self.flush_store();
        self.note_coverage(sub);
        done
    }
    fn note_coverage(&self, sub: &crate::Substrate) {
        if self.ram_only {
            let (covered, total) = self.coverage(sub);
            if let Ok(mut s) = self.stats.lock() {
                s.covered = covered;
                s.total = total;
                s.ready = total > 0 && covered == total;
                if s.ready && s.full_ms.is_none() {
                    s.full_ms = Some(self.started.elapsed().as_secs_f64() * 1000.0);
                }
            }
        }
    }
    /// As `fill`, `batch` chunks per call of `embed` (texts in, one vector per text out, `None`
    /// when the engine failed); the batched embedding is the same vectors within binary16 rounding
    /// and much less card time per chunk. No store file of the `open` form is written; a lane that
    /// keeps its vectors on disk (`keep_on_disk`) writes them to its store in batches.
    pub fn fill_batched(&self, sub: &crate::Substrate, embed: &BatchVectorize, batch: usize, keep_going: &dyn Fn() -> bool) -> usize {
        if self.stale().is_some() {
            return 0;
        }
        self.warm_instructions();
        if !FILL.load(std::sync::atomic::Ordering::Relaxed) {
            return 0;
        }
        let mut todo: Vec<(String, String, String)> = sub
            .chunks
            .iter()
            .filter(|c| self.vector(&c.key).is_none())
            .filter_map(|c| sub.files.get(c.file).map(|f| (c.key.clone(), embed_text(&f.path, c), f.path.replace('\\', "/"))))
            .collect();
        let mut seen = std::collections::BTreeSet::new();
        todo.retain(|(k, _, _)| seen.insert(k.clone()));
        let mut order_gen = self.priority_gen.load(std::sync::atomic::Ordering::Relaxed);
        self.order_todo(&mut todo);
        let mut done = 0;
        let mut at = 0;
        let mut held = self.bytes();
        while at < todo.len() {
            if !keep_going() || content_area_full(held) {
                break;
            }
            // A question moved its likely files forward: what is left is ordered again.
            let now = self.priority_gen.load(std::sync::atomic::Ordering::Relaxed);
            if now != order_gen {
                order_gen = now;
                if let Some(rest) = todo.get_mut(at..) {
                    self.order_todo(rest);
                }
            }
            let end = (at + batch.max(1)).min(todo.len());
            let Some(group) = todo.get(at..end) else { break };
            at = end;
            let texts: Vec<String> = group.iter().map(|g| g.1.clone()).collect();
            let t = std::time::Instant::now();
            let vs = {
                let _g = self.jobs.lock();
                embed(&texts)
            };
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let Some(vs) = vs.filter(|v| v.len() == group.len() && v.iter().all(|x| !x.is_empty())) else {
                if let Ok(mut s) = self.stats.lock() {
                    s.faults += group.len();
                }
                continue;
            };
            let vs: Vec<Vec<f32>> = group.iter().zip(vs).map(|(g, v)| self.admit(&g.0, v, Some(&g.2))).collect();
            held += group.iter().zip(&vs).map(|(g, v)| g.0.len() + v.len() * 4).sum::<usize>();
            if let Ok(mut m) = self.vecs.write() {
                for (g, v) in group.iter().zip(vs) {
                    m.insert(g.0.clone(), Arc::new(v));
                }
            }
            done += group.len();
            if let Ok(mut s) = self.stats.lock() {
                s.vectors = self.len();
                s.embedded_now += group.len();
                s.index_ms += ms;
            }
        }
        nmtype::limits::set_area_used("content", "dense_vectors", held as u64);
        self.flush_store();
        self.note_coverage(sub);
        done
    }
    /// The fused ranking of `question` on `sub`, computed once per question and version and
    /// shared by every tentacle; `None` when the question cannot be embedded or no chunk has a
    /// vector yet (the tentacles then read as before).
    pub fn ranks(&self, sub: &crate::Substrate, question: &str) -> Option<Ranks> {
        let mut cell = self.ranks.lock().ok()?;
        if let Some(r) = cell.as_ref().filter(|r| r.version == sub.version && r.question == question) {
            return Some(r.clone());
        }
        let t = tune();
        let top = if t.derive_top { self.above_level(sub, question)? } else { t.top.unwrap_or(TOP) };
        let order = self.order(sub, question, top)?;
        let r = Ranks {
            version: sub.version,
            question: question.to_owned(),
            place: order.into_iter().enumerate().map(|(p, i)| (i, p)).collect(),
        };
        *cell = Some(r.clone());
        Some(r)
    }
    /// The order of `question` on `sub` by embedding cosine alone (no lexical list fused in),
    /// computed once per question and version and shared by every tentacle: chunk index to place,
    /// the first `DEPTH` places. The candidates-only form (`CANDIDATES_ONLY`) reads it. `None` as
    /// `ranks`.
    pub fn meaning_ranks(&self, sub: &crate::Substrate, question: &str) -> Option<Ranks> {
        let mut cell = self.meaning.lock().ok()?;
        if let Some(r) = cell.as_ref().filter(|r| r.version == sub.version && r.question == question) {
            return Some(r.clone());
        }
        let order = self.dense_scores(sub, question)?;
        let r = Ranks {
            version: sub.version,
            question: question.to_owned(),
            place: order.iter().take(DEPTH).enumerate().map(|(p, (_, i))| (*i, p)).collect(),
        };
        *cell = Some(r.clone());
        Some(r)
    }
    /// The fused order of `question` on `sub`: chunk indices, best first, at most `depth` of
    /// them (`ranks` keeps the first `TOP`; the E2 head probe reads deeper). `None` as `ranks`.
    pub fn order(&self, sub: &crate::Substrate, question: &str, depth: usize) -> Option<Vec<usize>> {
        Some(self.fuse(sub, question, DEPTH)?.into_iter().take(depth).collect())
    }
    /// The fused order over every chunk (each list read to the end, not to `DEPTH`): the one
    /// big window E3.2 compares the lenses against.
    pub fn order_full(&self, sub: &crate::Substrate, question: &str) -> Option<Vec<usize>> {
        self.fuse(sub, question, usize::MAX)
    }
    /// Chunk indices of the whole tree by embedding cosine with `question`, best first (the
    /// meaning lens, `crate::lens`).
    pub fn embedding_order(&self, sub: &crate::Substrate, question: &str) -> Option<Vec<usize>> {
        Some(self.dense_scores(sub, question)?.iter().map(|x| x.1).collect())
    }
    /// The meaning signal of the drones' evidence: every chunk's embedding cosine with `question`,
    /// then with each content-free question (`CONTENT_FREE`, encoded and scored once per version),
    /// each in chunk order (`NaN` for a chunk with no vector). `None` when the signal is off, the
    /// lane cannot rank (as `ranks`) or fewer than two content-free questions encode.
    pub fn cosines(&self, sub: &crate::Substrate, question: &str) -> Option<Vec<Vec<f32>>> {
        if !MEANING_SIGNAL.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        self.cosines_now(sub, question)
    }
    /// `cosines` whatever `MEANING_SIGNAL` holds (a caller with its own switch, the stack).
    pub fn cosines_now(&self, sub: &crate::Substrate, question: &str) -> Option<Vec<Vec<f32>>> {
        let scores = self.dense_scores(sub, question)?;
        let mut posed = vec![f32::NAN; sub.chunks.len()];
        for (c, i) in scores.iter() {
            if let Some(x) = posed.get_mut(*i) {
                *x = *c;
            }
        }
        let nulls = self.content_free_cosines(sub, task_for(question))?;
        Some(std::iter::once(posed).chain(nulls.iter().cloned()).collect())
    }
    /// Stage 5 WP-C4 E1: `own`, the cosine of a piece's own vector `ov` with `q`, raised by how much the
    /// best of the piece's auxiliary vectors beats it on the calibrated scale: each source's cosine less
    /// its own mean cosine with the content-free questions `nulls`. The content-free rows stay the
    /// piece's own, so the downstream calibration (posed less content-free) reads
    /// max(own calibrated, note calibrated). A plain maximum of raw cosines (the first form) lifted
    /// noted pieces across the board: general notes sit close to many questions, content-free ones
    /// too (eight sets: large code pack 47 to 43, two decoys lost). Without `nulls`, or with
    /// `NANO_AUX_NOTES=off`, `own`.
    ///
    /// Second form (2026-11-23, the large code dip diagnosed): the gain is returned, not applied.
    /// The caller raises a piece only by the part of its gain above the question's band
    /// (`aux_band`): a noted piece takes the best of up to four draws (its own text and up to three
    /// notes), so about half the noted pieces gained on every question whatever it asked (large
    /// code: pieces at place 7,576, 4,276 and 2,749 lifted into the first 20 on exp05, the decoy
    /// dec16 raised 762 to 7). `None` for a piece without notes, without `nulls`, or with
    /// `NANO_AUX_NOTES=off`.
    /// Roadmap Phase 1 (2026-11-24): the piece's folder note (`module`, `module:<path>`, the L3 level
    /// of the world model) is read with the piece's own notes, through the same band.
    #[expect(clippy::too_many_arguments, reason = "the piece, its folder, the question and the calibration")]
    fn aux_gain(&self, key: &str, module: Option<&str>, q: &[f32], ov: &[f32], own: f32, nulls: Option<&[Vec<f32>]>) -> Option<f32> {
        if std::env::var("NANO_AUX_NOTES").is_ok_and(|v| v == "off") {
            return None;
        }
        let nulls = nulls.filter(|n| !n.is_empty())?;
        let vs: Vec<Arc<Vec<f32>>> = {
            let a = self.aux.read().ok()?;
            let mut v: Vec<Arc<Vec<f32>>> = a.get(key).cloned().unwrap_or_default();
            if let Some(m) = module.and_then(|m| a.get(m)) {
                v.extend(m.iter().cloned());
            }
            v
        };
        if vs.is_empty() {
            return None;
        }
        #[expect(clippy::cast_precision_loss, reason = "a few content-free questions")]
        let k = nulls.len() as f32;
        let level = |v: &[f32]| nulls.iter().map(|n| dot(n, v)).sum::<f32>() / k;
        let own_cal = own - level(ov);
        let best = vs.iter().map(|v| dot(q, v) - level(v)).fold(f32::NEG_INFINITY, f32::max);
        best.is_finite().then_some(best - own_cal)
    }
    /// Stage 5 WP-C4 E1/E4: encode `text` (a checked note or likely names of the piece `key`, or a
    /// structure node's note under `module:<path>`) as a document and hold it beside the piece, at
    /// most `AUX_PER_KEY` a key (the newest kept). The question scores and content-free cosines are
    /// recomputed on their next use. False when the encoder gives no vector.
    pub fn add_aux_text(&self, key: &str, text: &str) -> bool {
        // The text is kept whatever happens next, so a lane built for new embedder values encodes
        // it (`aux_texts`); a stale lane encodes nothing.
        if let Ok(mut a) = self.aux_src.write() {
            let e = a.entry(key.to_owned()).or_default();
            if !e.iter().any(|t| t == text) {
                e.push(text.to_owned());
                if e.len() > AUX_PER_KEY {
                    e.remove(0);
                }
            }
        }
        if self.stale().is_some() {
            return false;
        }
        // Not under `jobs`: a served call holds the card gate the encoder waits for and then takes
        // `jobs` to encode its question, so holding `jobs` here while waiting would deadlock (seen
        // 2026-11-23); the card gate already serialises the encoding.
        let Some(v) = (self.doc)(text) else { return false };
        if let Ok(mut a) = self.aux.write() {
            let e = a.entry(key.to_owned()).or_default();
            e.push(Arc::new(v));
            if e.len() > AUX_PER_KEY {
                e.remove(0);
            }
        }
        if let Ok(mut s) = self.scores.lock() {
            *s = None;
        }
        if let Ok(mut c) = self.content_free.lock() {
            c.clear();
        }
        true
    }
    /// Stage 5 WP-C4: the auxiliary vectors of `key` (structure nodes, `module:<path>`, included).
    pub fn aux_vectors(&self, key: &str) -> Vec<Arc<Vec<f32>>> {
        self.aux.read().ok().and_then(|a| a.get(key).cloned()).unwrap_or_default()
    }
    /// Stage 5 WP-C4: keys holding auxiliary vectors.
    pub fn aux_len(&self) -> usize {
        self.aux.read().map_or(0, |a| a.len())
    }
    /// Every chunk's cosine with each content-free question encoded under `task`, in chunk order;
    /// once per version and instruction.
    fn content_free_cosines(&self, sub: &crate::Substrate, task: &str) -> Option<Cosines> {
        self.sync_epoch();
        if let Some((_, _, v)) = self.content_free.lock().ok()?.iter().find(|(ver, t, _)| *ver == sub.version && t == task) {
            return Some(Arc::clone(v));
        }
        let qs = self.null_queries_for(task)?;
        if qs.len() < 2 {
            return None;
        }
        let v: Arc<Vec<Vec<f32>>> = Arc::new(qs.iter().map(|q| sub.chunks.iter().map(|c| self.vector(&c.key).map_or(f32::NAN, |v| dot(q, &v))).collect()).collect());
        if let Ok(mut cell) = self.content_free.lock() {
            cell.retain(|(ver, _, _)| *ver == sub.version);
            cell.push((sub.version, task.to_owned(), Arc::clone(&v)));
        }
        Some(v)
    }
    /// The content-free questions encoded under `task`; once per instruction. `None` when the
    /// encoder gives none.
    fn null_queries_for(&self, task: &str) -> Option<Arc<Vec<Vec<f32>>>> {
        self.sync_epoch();
        if let Some(v) = self.null_queries.lock().ok()?.get(task) {
            return Some(Arc::clone(v));
        }
        let qs: Vec<Vec<f32>> = {
            let _g = self.jobs.lock();
            CONTENT_FREE.iter().filter_map(|q| (self.query)(&posed(task, q))).collect()
        };
        if qs.is_empty() {
            return None;
        }
        let v = Arc::new(qs);
        if let Ok(mut cell) = self.null_queries.lock() {
            cell.insert(task.to_owned(), Arc::clone(&v));
        }
        Some(v)
    }
    /// Encode the content-free questions under every instruction a question can be posed with
    /// (`QUERY_TASK`, `ANSWER_TASK`, each kind's), so no request waits for them.
    pub fn warm_instructions(&self) {
        let kinds = ["locate", "value", "explain", "trace", "absent-check"];
        for task in [QUERY_TASK, ANSWER_TASK].into_iter().chain(kinds.iter().map(|k| kind_task(k))) {
            let _ = self.null_queries_for(task);
        }
    }
    /// The derived shortlist (step 1l, change B): how many pieces have a cosine with the question
    /// above the lane's own content-free level, the highest cosine any piece reaches with a
    /// content-free question under the same instruction (what an empty question already finds).
    /// Fallback: `TOP` when fewer than two content-free questions encode; at least one piece.
    fn above_level(&self, sub: &crate::Substrate, question: &str) -> Option<usize> {
        let scores = self.dense_scores(sub, question)?;
        let Some(nulls) = self.content_free_cosines(sub, task_for(question)) else { return Some(TOP) };
        let level = nulls.iter().flat_map(|v| v.iter().copied()).filter(|x| !x.is_nan()).fold(f32::MIN, f32::max);
        let n = scores.iter().take_while(|(c, _)| *c > level).count().max(1);
        if let Some(path) = TRACE.get() {
            let best = scores.first().map_or(f32::NAN, |x| x.0);
            let line = format!("above {n} level {level:.4} best {best:.4} question {}\n", question.replace('\n', " "));
            let _ = std::fs::OpenOptions::new().create(true).append(true).open(path).and_then(|mut h| h.write_all(line.as_bytes()));
        }
        Some(n)
    }
    /// Embedding cosine of every chunk with a vector, best first; computed once per question and
    /// version.
    fn dense_scores(&self, sub: &crate::Substrate, question: &str) -> Option<Arc<Vec<(f32, usize)>>> {
        self.sync_epoch();
        let held = self.len();
        if let Some((_, _, _, v)) = self.scores.lock().ok()?.as_ref().filter(|(ver, q, n, _)| *ver == sub.version && q == question && *n == held) {
            return Some(Arc::clone(v));
        }
        let v = Arc::new(self.dense_scores_now(sub, question)?);
        if let Ok(mut cell) = self.scores.lock() {
            *cell = Some((sub.version, question.to_owned(), held, Arc::clone(&v)));
        }
        Some(v)
    }
    fn dense_scores_now(&self, sub: &crate::Substrate, question: &str) -> Option<Vec<(f32, usize)>> {
        if self.withheld() {
            if let Ok(mut s) = self.stats.lock() {
                s.stale_skips += 1;
            }
            return None;
        }
        if self.is_empty() {
            return None;
        }
        // Safeguard (a): a RAM-only lane ranks only at full coverage; it counts the questions
        // that ran without it.
        let partial = self.ram_only && self.partial.load(std::sync::atomic::Ordering::Relaxed);
        if partial && !self.ready(sub) {
            // The first start: rank with the vectors there are, and build this question's likely
            // files next (the files of its lexical order's first places).
            let mut files: Vec<String> = Vec::new();
            for i in sub.lexical_order(question).into_iter().take(DEPTH) {
                if let Some(f) = sub.chunks.get(i).and_then(|c| sub.files.get(c.file)) {
                    let p = f.path.replace('\\', "/");
                    if !files.contains(&p) {
                        files.push(p);
                    }
                }
            }
            self.prefer(&files);
            if let Ok(mut s) = self.stats.lock() {
                s.partial += 1;
                if s.first_rank_ms.is_none() {
                    s.first_rank_ms = Some(self.started.elapsed().as_secs_f64() * 1000.0);
                }
            }
        } else if !self.ready(sub) {
            if let Ok(mut s) = self.stats.lock() {
                s.not_ready += 1;
                let (covered, total) = self.coverage(sub);
                s.covered = covered;
                s.total = total;
                s.ready = false;
            }
            return None;
        }
        let t = std::time::Instant::now();
        let text = posed(task_for(question), question);
        let held = self.qvec.lock().ok().and_then(|c| c.as_ref().filter(|(t, _)| *t == text).map(|(_, v)| Arc::clone(v)));
        let qv = match held {
            Some(v) => v,
            None => {
                // During the first start a served question never waits for the build: when the
                // engine is busy embedding pieces, this call runs without the lane.
                let v = if partial && !self.ready(sub) {
                    let Ok(_g) = self.jobs.try_lock() else {
                        if let Ok(mut s) = self.stats.lock() {
                            s.not_ready += 1;
                        }
                        return None;
                    };
                    (self.query)(&text)
                } else {
                    let _g = self.jobs.lock();
                    (self.query)(&text)
                };
                let v = Arc::new(v?);
                if let Ok(mut c) = self.qvec.lock() {
                    *c = Some((text, Arc::clone(&v)));
                }
                v
            }
        };
        if let Ok(mut s) = self.stats.lock() {
            s.queries += 1;
            s.query_ms += t.elapsed().as_secs_f64() * 1000.0;
        }
        let nulls = self.null_queries_for(task_for(question));
        // Own cosine and, for a noted piece, its note gain (`aux_gain`).
        let rows: Vec<(f32, Option<f32>, usize)> = sub
            .chunks
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                self.vector(&c.key).map(|v| {
                    let own = dot(&qv, &v);
                    let module = sub.files.get(c.file).map(|f| f.path.replace('\\', "/")).and_then(|p| p.rsplit_once('/').map(|(d, _)| format!("module:{d}")));
                    (own, self.aux_gain(&c.key, module.as_deref(), &qv, &v, own, nulls.as_deref().map(Vec::as_slice)), i)
                })
            })
            .collect();
        // The lens's order on the plain scale (`lens_fn`): each piece's lensed cosine mapped by the
        // affine map that gives the lensed cosines the plain cosines' mean and spread.
        let lq = self.lens_fn.read().ok().and_then(|f| f.as_ref().and_then(|f| f(&qv)));
        let rows: Vec<(f32, Option<f32>, usize)> = match lq {
            Some(lq) => {
                let flat = self.flat_pieces(sub);
                let lensed: Vec<f32> = rows
                    .iter()
                    .map(|(_, _, i)| match flat.as_ref().and_then(|f| f.get(*i)).and_then(Option::as_ref) {
                        Some(fv) => dot(&lq, fv),
                        None => sub.chunks.get(*i).and_then(|c| self.vector(&c.key)).map_or(f32::NAN, |v| dot(&lq, &v)),
                    })
                    .collect();
                let stats = |xs: &mut dyn Iterator<Item = f32>| {
                    let v: Vec<f32> = xs.filter(|x| x.is_finite()).collect();
                    #[expect(clippy::cast_precision_loss, reason = "piece counts are small")]
                    let n = v.len().max(1) as f32;
                    let m = v.iter().sum::<f32>() / n;
                    (m, (v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / n).sqrt())
                };
                let (mb, sb) = stats(&mut rows.iter().map(|r| r.0));
                let (ml, sl) = stats(&mut lensed.iter().copied());
                let a = if sl > 0.0 { sb / sl } else { 1.0 };
                rows.iter().zip(&lensed).map(|((own, g, i), l)| (if l.is_finite() { mb + a * (l - ml) } else { *own }, *g, *i)).collect()
            }
            None => rows,
        };
        let gains: Vec<f32> = rows.iter().filter_map(|r| r.1).collect();
        let band = aux_band(&gains);
        let mut dense: Vec<(f32, usize)> = rows.iter().map(|(own, g, i)| (own + g.map_or(0.0, |g| (g - band).max(0.0)), *i)).collect();
        dense.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        // Diagnosis (`NANO_DENSE_TRACE`): the served lane's first ten pieces with their scores.
        if let Some(path) = TRACE.get() {
            let top: Vec<String> = dense
                .iter()
                .take(10)
                .map(|(sc, i)| sub.chunks.get(*i).map_or_else(String::new, |c| format!("{}:{}-{} {sc:.4}", sub.files.get(c.file).map_or("?", |f| f.path.as_str()), c.start, c.end)))
                .collect();
            let line = format!("dense top q {} | {}\n", question.replace('\n', " "), top.join("; "));
            let _ = std::fs::OpenOptions::new().create(true).append(true).open(path).and_then(|mut h| h.write_all(line.as_bytes()));
        }
        if let Some(path) = std::env::var_os("NANO_AUX_TRACE")
            && let Some(nulls) = nulls.as_deref()
        {
            self.aux_trace(sub, question, &qv, nulls, &dense, band, std::path::Path::new(&path));
        }
        Some(dense)
    }
    /// Stage 5 WP-C4 diagnosis (`NANO_AUX_TRACE=<file>`): per question, the calibrated own and best
    /// note cosines over the noted pieces (mean and spread of each), how many pieces the notes
    /// lift, and the most raised pieces with their places without and with the notes.
    #[expect(clippy::too_many_arguments, reason = "diagnostic trace helper")]
    fn aux_trace(&self, sub: &crate::Substrate, question: &str, qv: &[f32], nulls: &[Vec<f32>], dense: &[(f32, usize)], band: f32, path: &std::path::Path) {
        if nulls.is_empty() {
            return;
        }
        #[expect(clippy::cast_precision_loss, reason = "a few content-free questions")]
        let k = nulls.len() as f32;
        let level = |v: &[f32]| nulls.iter().map(|n| dot(n, v)).sum::<f32>() / k;
        let Ok(aux) = self.aux.read() else { return };
        let mut own_all: Vec<(f32, usize)> = Vec::new();
        let mut rows: Vec<(usize, f32, f32)> = Vec::new();
        let mut own_cal_all: Vec<f32> = Vec::new();
        for (i, c) in sub.chunks.iter().enumerate() {
            let Some(v) = self.vector(&c.key) else { continue };
            let own = dot(qv, &v);
            own_all.push((own, i));
            let oc = own - level(&v);
            own_cal_all.push(oc);
            if let Some(vs) = aux.get(&c.key) {
                let best = vs.iter().map(|n| dot(qv, n) - level(n)).fold(f32::NEG_INFINITY, f32::max);
                rows.push((i, oc, best));
            }
        }
        own_all.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let place = |order: &[(f32, usize)], i: usize| order.iter().position(|x| x.1 == i).map_or(0, |p| p + 1);
        let stats = |xs: &[f32]| {
            #[expect(clippy::cast_precision_loss, reason = "piece counts are small")]
            let n = xs.len().max(1) as f32;
            let m = xs.iter().sum::<f32>() / n;
            let sd = (xs.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / n).sqrt();
            (m, sd)
        };
        let (om, osd) = stats(&own_cal_all);
        let noted_own: Vec<f32> = rows.iter().map(|r| r.1).collect();
        let noted_best: Vec<f32> = rows.iter().map(|r| r.2).collect();
        let (nom, nosd) = stats(&noted_own);
        let (nbm, nbsd) = stats(&noted_best);
        let lifted = rows.iter().filter(|r| r.2 - r.1 > band).count();
        let mut raised: Vec<(usize, usize, usize, f32, f32)> =
            rows.iter().filter(|r| r.2 - r.1 > band).map(|r| (r.0, place(&own_all, r.0), place(dense, r.0), r.1, r.2)).collect();
        raised.sort_by_key(|r| r.2);
        let mut line = format!(
            "aux q {} | all own cal {om:.4}+-{osd:.4} | noted {} own cal {nom:.4}+-{nosd:.4} note cal {nbm:.4}+-{nbsd:.4} | band {band:.4} lifted {lifted} |",
            question.replace('\n', " "),
            rows.len()
        );
        for (i, before, after, oc, nc) in raised.iter().take(8) {
            let key = sub.chunks.get(*i).map_or_else(|| "?".to_owned(), |c| format!("{}:{}-{}", sub.files.get(c.file).map_or("?", |f| f.path.as_str()), c.start, c.end));
            line.push_str(&format!(" {key} {before}->{after} ({oc:.3}/{nc:.3});"));
        }
        line.push('\n');
        let _ = std::fs::OpenOptions::new().create(true).append(true).open(path).and_then(|mut h| h.write_all(line.as_bytes()));
    }
    /// Reciprocal-rank fusion of the embedding order and the lexical order, each read to
    /// `list_depth`.
    fn fuse(&self, sub: &crate::Substrate, question: &str, list_depth: usize) -> Option<Vec<usize>> {
        let dense = self.dense_scores(sub, question)?;
        let lexical = sub.lexical_order(question);
        let t = tune();
        let k = t.rrf_k.unwrap_or(RRF_K);
        let weight = if t.derive_weight { word_hold(sub, question, &lexical, list_depth.min(DEPTH)) } else { t.lexical_weight.unwrap_or(LEXICAL_WEIGHT) };
        if let Some(path) = TRACE.get() {
            let line = format!("weight {weight:.4} question {}\n", question.replace('\n', " "));
            let _ = std::fs::OpenOptions::new().create(true).append(true).open(path).and_then(|mut h| h.write_all(line.as_bytes()));
        }
        let mut fused: BTreeMap<usize, f32> = BTreeMap::new();
        #[expect(clippy::cast_precision_loss, reason = "ranks are small")]
        for (r, i) in dense.iter().map(|x| x.1).take(list_depth).enumerate() {
            *fused.entry(i).or_default() += 1.0 / (k + r as f32);
        }
        #[expect(clippy::cast_precision_loss, reason = "ranks are small")]
        for (r, i) in lexical.into_iter().take(list_depth).enumerate() {
            *fused.entry(i).or_default() += weight / (k + r as f32);
        }
        let mut order: Vec<(usize, f32)> = fused.into_iter().collect();
        order.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        Some(order.into_iter().map(|x| x.0).collect())
    }
}

/// The project's lens on a question's vector: the lensed vector, or `None` when no lens is in force.
pub type QueryLens = Arc<dyn Fn(&[f32]) -> Option<Vec<f32>> + Send + Sync>;
/// The lens's flattening for the pieces' vectors, taken once a question (`None`: no lens in force).
pub type DocLens = Arc<dyn Fn() -> Option<PieceFlat> + Send + Sync>;
/// Every piece's flattened vector, by chunk index (`None`: the piece has no vector).
pub type FlatPieces = Arc<Vec<Option<Vec<f32>>>>;
/// One question's flattening of a piece's vector.
pub type PieceFlat = Arc<dyn Fn(&[f32]) -> Vec<f32> + Send + Sync>;

/// Stage 5 WP-C4: the note gain a noted piece takes from chance alone on one question: the median
/// gain over the question's noted pieces plus the largest deviation noise gives among that many,
/// sqrt(2 ln n) robust spreads (1.4826 times the median absolute deviation, the form G1 uses). Only
/// the part of a gain above it raises a piece. Derived per question, not fitted; 0 for fewer than
/// two noted pieces, and never below 0.
fn aux_band(gains: &[f32]) -> f32 {
    let mut g: Vec<f32> = gains.iter().copied().filter(|x| x.is_finite()).collect();
    if g.len() < 2 {
        return 0.0;
    }
    let mid = |w: &mut Vec<f32>| -> f32 {
        w.sort_by(f32::total_cmp);
        let n = w.len();
        if n % 2 == 1 { w.get(n / 2).copied().unwrap_or(0.0) } else { (w.get(n / 2 - 1).copied().unwrap_or(0.0) + w.get(n / 2).copied().unwrap_or(0.0)) / 2.0 }
    };
    let m = mid(&mut g);
    let mut dev: Vec<f32> = g.iter().map(|x| (x - m).abs()).collect();
    let sigma = 1.4826 * mid(&mut dev);
    #[expect(clippy::cast_precision_loss, reason = "piece counts are small")]
    let n = g.len() as f32;
    (m + sigma * (2.0 * n.ln()).sqrt()).max(0.0)
}

/// The derived lexical weight (step 1l, change B): how strongly the material holds the request's
/// words, as the rarity-weighted share of the request's key terms the best holder among the
/// lexical order's first `depth` pieces contains (a rare word held is high surprise; a word no
/// piece holds, the word-mismatch case, adds nothing). One for a request the words settle fully
/// (words weigh as much as meaning), near zero where the material does not hold its words (meaning
/// decides). Fallback: `LEXICAL_WEIGHT` when the request has no key term.
fn word_hold(sub: &crate::Substrate, question: &str, lexical: &[usize], depth: usize) -> f32 {
    if nmlex::key_terms(question).is_empty() {
        return LEXICAL_WEIGHT;
    }
    lexical
        .iter()
        .take(depth)
        .filter_map(|i| sub.chunks.get(*i))
        .map(|c| {
            let path = sub.files.get(c.file).map_or("", |f| f.path.as_str());
            sub.coverage(question, &format!("{path} {} {}", c.heading, c.text))
        })
        .fold(0.0_f32, f32::max)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// One store record: 64-byte hex hash, little-endian u32 width, the f32 values.
fn encode(key: &str, v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(68 + v.len() * 4);
    out.extend_from_slice(key.as_bytes());
    out.extend_from_slice(&u32::try_from(v.len()).unwrap_or(0).to_le_bytes());
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Every whole record in a store file; a cut record at the end is ignored.
fn decode(bytes: &[u8]) -> Vec<(String, Vec<f32>)> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(head) = bytes.get(at..at + 68) {
        let (key, w) = head.split_at(64);
        let w = u32::from_le_bytes([w[0], w[1], w[2], w[3]]) as usize;
        let Some(body) = bytes.get(at + 68..at + 68 + w * 4) else { break };
        let Ok(key) = std::str::from_utf8(key) else { break };
        out.push((key.to_owned(), body.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()));
        at += 68 + w * 4;
    }
    out
}

/// Self-check of the `dense` sub-component: records round-trip, a cut record is ignored, the
/// fused ranking puts a chunk both orders like first, and a question with no engine gives none.
pub(crate) fn check() -> Result<(), String> {
    let key = "a".repeat(64);
    let mut bytes = encode(&key, &[0.5, -1.0]);
    bytes.extend_from_slice(&encode(&"b".repeat(64), &[1.0])[..10]);
    if decode(&bytes) != vec![(key, vec![0.5, -1.0])] {
        return Err("nmsubs.dense: store records do not round-trip".into());
    }
    let embed: crate::Embed = Arc::new(|_: &str| vec![1.0]);
    let sub = crate::from_files(&[("a.rs", "fn parse_header() { read the header bytes }"), ("b.rs", "fn unrelated() {}")], embed);
    let vec_of = |t: &str| Some(if t.contains("header") { vec![1.0, 0.0] } else { vec![0.0, 1.0] });
    let lane = DenseLane::open(None, Arc::new(vec_of), Arc::new(move |_: &str| Some(vec![1.0, 0.0])));
    lane.fill(&sub, &|| true);
    let r = lane.ranks(&sub, "where is the header parsed").ok_or("nmsubs.dense: no ranking")?;
    let first = r.place.iter().find(|(_, p)| **p == 0).map(|(i, _)| *i);
    if lane.len() != 2 || first.and_then(|i| sub.chunks.get(i)).map(|c| c.file) != Some(0) {
        return Err("nmsubs.dense: fused ranking does not put the matching chunk first".into());
    }
    let none = DenseLane::open(None, Arc::new(|_: &str| None), Arc::new(|_: &str| None));
    none.fill(&sub, &|| true);
    if none.ranks(&sub, "x").is_some() {
        return Err("nmsubs.dense: a lane without vectors ranked".into());
    }
    // RAM-only lane (refinement step 1, item 7): no ranking at partial coverage; priority files
    // first; full coverage ranks as the file-backed lane does; a vanished chunk's vector dropped.
    let ram = DenseLane::in_ram(Arc::new(vec_of), Arc::new(move |_: &str| Some(vec![1.0, 0.0])));
    ram.set_priority(vec!["b.rs".into()]);
    let mut seen = 0;
    let one = std::cell::Cell::new(0);
    ram.fill(&sub, &|| {
        one.set(one.get() + 1);
        one.get() <= 1
    });
    let first_embedded =
        sub.chunks.iter().find(|c| ram.vector(&c.key).is_some()).and_then(|c| sub.files.get(c.file)).map(|f| f.path.clone());
    if ram.ranks(&sub, "where is the header parsed").is_some()
        || ram.stats.lock().map_or(true, |s| s.ready || s.not_ready != 1)
        || first_embedded.as_deref() != Some("b.rs")
    {
        return Err(format!("nmsubs.dense: RAM lane ranked at partial coverage or built out of order (first {first_embedded:?})"));
    }
    ram.fill(&sub, &|| true);
    let r2 = ram.ranks(&sub, "where is the header parsed").ok_or("nmsubs.dense: RAM lane did not rank at full coverage")?;
    if r2.place != r.place {
        return Err("nmsubs.dense: RAM lane ranks differently from the file-backed lane".into());
    }
    let smaller = crate::from_files(&[("a.rs", "fn parse_header() { read the header bytes }")], Arc::new(|_: &str| vec![1.0]));
    seen += ram.retain(&smaller);
    if seen != 1 || ram.len() != 1 || ram.file.is_some() || ram.bytes() == 0 {
        return Err(format!("nmsubs.dense: RAM lane retained {} vectors after dropping {seen}", ram.len()));
    }
    check_partial(vec_of)?;
    store::check()?;
    check_kept(&sub, vec_of, &r.place)
}

/// The first start: a RAM-only lane with `set_partial` ranks with the vectors it has, counts the
/// question, and builds the question's likely file next.
fn check_partial<F>(vec_of: F) -> Result<(), String>
where
    F: Fn(&str) -> Option<Vec<f32>> + Send + Sync + Copy + 'static,
{
    let embed: crate::Embed = Arc::new(|_: &str| vec![1.0]);
    let sub = crate::from_files(
        &[("a.rs", "fn unrelated() {}"), ("b.rs", "fn other_thing() {}"), ("c.rs", "fn parse_header() { read the header bytes }")],
        embed,
    );
    let lane = DenseLane::in_ram(Arc::new(vec_of), Arc::new(move |_: &str| Some(vec![1.0, 0.0])));
    lane.set_partial(true);
    let one = std::cell::Cell::new(0);
    lane.fill(&sub, &|| {
        one.set(one.get() + 1);
        one.get() <= 1
    });
    if lane.ready(&sub) || lane.ranks(&sub, "where is the header parsed").is_none() {
        return Err("nmsubs.dense: the first start did not rank at partial coverage".into());
    }
    if lane.stats.lock().map_or(true, |s| s.partial != 1 || s.first_rank_ms.is_none()) {
        return Err("nmsubs.dense: the first start did not count its partial ranking".into());
    }
    let embed_batch = move |t: &[String]| Some(t.iter().map(|x| vec_of(x).unwrap_or_default()).collect::<Vec<_>>());
    let two = std::cell::Cell::new(0);
    lane.fill_batched(&sub, &embed_batch, 1, &|| {
        two.set(two.get() + 1);
        two.get() <= 1
    });
    let c = sub.files.iter().position(|f| f.path == "c.rs");
    if !sub.chunks.iter().filter(|ch| Some(ch.file) == c).all(|ch| lane.vector(&ch.key).is_some()) {
        return Err("nmsubs.dense: the question's likely file was not built next".into());
    }
    Ok(())
}

/// A lane kept on disk: a second lane over the same folder loads every vector, embeds none, and
/// ranks as the first did.
fn check_kept<F>(sub: &crate::Substrate, vec_of: F, place: &BTreeMap<usize, usize>) -> Result<(), String>
where
    F: Fn(&str) -> Option<Vec<f32>> + Send + Sync + Copy + 'static,
{
    let dir = std::env::temp_dir().join(format!("nano-dense-kept-check-{}-{}", std::process::id(), nmtype::now_ms()));
    let header: store::Header = [("weights".to_owned(), "check".to_owned())].into();
    let lane = || DenseLane::in_ram(Arc::new(vec_of), Arc::new(move |_: &str| Some(vec![1.0, 0.0])));
    let r = (|| {
        let first = lane();
        first.keep_on_disk(&dir, header.clone(), sub);
        first.fill(sub, &|| true);
        let second = lane();
        second.keep_on_disk(&dir, header.clone(), sub);
        let embedded = second.fill(sub, &|| true);
        let r = second.ranks(sub, "where is the header parsed").ok_or("nmsubs.dense: kept lane did not rank")?;
        if embedded != 0 || !second.ready(sub) || &r.place != place {
            return Err(format!("nmsubs.dense: kept lane embedded {embedded} after a restart or ranked differently"));
        }
        // H32 live-session undo barrier: edit b.rs -> old chunk enters dead-skin barrier, then undo -> revived with 0 embeds.
        let edited = crate::from_files(
            &[("a.rs", "fn parse_header() { read the header bytes }"), ("b.rs", "fn edited_unrelated() {}")],
            Arc::new(|_: &str| vec![1.0]),
        );
        second.retain(&edited);
        let embedded_edit = second.fill(&edited, &|| true);
        if embedded_edit != 1 {
            return Err(format!("nmsubs.dense(H32): edited file embedded {embedded_edit}, want 1"));
        }
        second.retain(sub);
        let embedded_undo = second.fill(sub, &|| true);
        let revived = second.stats.lock().map_or(0, |s| s.revived);
        if embedded_undo != 0 || revived != 1 || !second.ready(sub) {
            return Err(format!("nmsubs.dense(H32): undo embedded {embedded_undo} (revived {revived}), want 0 embeds and 1 revival"));
        }
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&dir);
    r
}

/// Memory areas (2026-11-27): the dense lane's vectors count in the `content` area as
/// `dense_vectors`; a fill stops when the area holds its budget (the chunks not reached keep no
/// vector, and the lane ranks with the vectors it holds).
fn content_area_full(held: usize) -> bool {
    nmtype::limits::set_area_used("content", "dense_vectors", held as u64);
    let full = nmtype::limits::area_over("content");
    if full {
        nmtype::limits::set_area_full("content", true);
    }
    full
}

/// Puts `next`, a lane built for the embedder serving now, in place of the lane in `cell` in one
/// write: every reader that takes the lane after it reads `next`, and a reader already holding the
/// old lane finishes on it. Refused, with the old lane left in place, unless `next` is not stale
/// itself and holds a vector for every current piece of `sub`. Returns the lane it replaced.
/// # Errors
/// `next` is stale or does not cover `sub`; the lock is poisoned.
pub fn switch(cell: &RwLock<Option<Arc<DenseLane>>>, next: &Arc<DenseLane>, sub: &crate::Substrate) -> Result<Option<Arc<DenseLane>>, String> {
    if let Some(why) = next.stale() {
        return Err(format!("the new lane is stale itself ({why})"));
    }
    let (have, total) = next.coverage(sub);
    if total == 0 || have < total {
        return Err(format!("the new lane covers {have} of {total} pieces"));
    }
    let mut g = cell.write().map_err(|_| "the lane's lock is poisoned".to_owned())?;
    Ok(g.replace(Arc::clone(next)))
}

#[cfg(test)]
mod follow_tests {
    use super::*;

    fn sub() -> crate::Substrate {
        crate::from_files(&[("a.rs", "fn parse_header() { read the header bytes }"), ("b.rs", "fn unrelated() {}")], Arc::new(|_: &str| vec![1.0]))
    }
    /// Values A put the header piece on the first axis; values B on the second.
    fn lane(axis: usize) -> DenseLane {
        let doc = move |t: &str| {
            let hit = t.contains("header");
            Some(if hit == (axis == 0) { vec![1.0, 0.0] } else { vec![0.0, 1.0] })
        };
        let q = move |_: &str| Some(if axis == 0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] });
        DenseLane::in_ram(Arc::new(doc), Arc::new(q))
    }
    fn header(values: &str) -> store::Header {
        [("parameters".to_owned(), values.to_owned()), ("recipe".to_owned(), RECIPE.to_owned())].into()
    }

    #[test]
    fn stale_mark_withholds_the_lane() {
        let sub = sub();
        let dir = std::env::temp_dir().join(format!("nano-dense-stale-{}-{}", std::process::id(), nmtype::now_ms()));
        let l = lane(0);
        l.keep_on_disk(&dir, header("open weights a"), &sub);
        l.fill(&sub, &|| true);
        assert!(l.ready(&sub) && l.ranks(&sub, "where is the header parsed").is_some());
        assert_eq!(l.differs_from(&header("open weights a")), None, "equal values are not a change");
        let why = l.differs_from(&header("open weights a with overlay x")).expect("a change is named");
        assert!(why.starts_with("parameters: open weights a -> "), "{why}");
        assert!(!l.query_side_holds(&header("open weights a with overlay x")));
        l.mark_stale(&why, false);
        assert_eq!(l.stale().as_deref(), Some(why.as_str()));
        assert!(!l.ready(&sub));
        assert!(l.ranks(&sub, "where is the header parsed").is_none() && l.embedding_order(&sub, "x").is_none());
        assert!(sub.chunks.iter().all(|c| l.vector(&c.key).is_none()), "no stale piece vector is handed out");
        assert!(!l.keeps_on_disk(), "the store is let go for the new lane");
        assert_eq!(l.fill(&sub, &|| true), 0, "a stale lane is never filled");
        assert!(!l.add_aux_text("k", "a note"), "nothing is encoded into a stale lane");
        assert_eq!(l.aux_texts(), vec![("k".to_owned(), "a note".to_owned())], "the note's text is kept for the new lane");
        let st = l.stats.lock().map(|s| s.clone()).unwrap_or_default();
        assert_eq!(st.stale, why);
        assert!(st.stale_skips >= 1);
        // The rows written before the mark are on disk under the old values.
        let (_, got, _) = store::Store::open(&dir, header("open weights a"), &sub.chunks.iter().map(|c| c.key.as_str()).collect());
        assert_eq!(got.len(), sub.chunks.len());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn recipe_change_keeps_the_question_side() {
        let sub = sub();
        let l = lane(0);
        l.set_made_by(header("open weights a"));
        l.fill(&sub, &|| true);
        let mut lead = header("open weights a");
        lead.insert("recipe".to_owned(), RECIPE_LEAD.to_owned());
        assert!(l.query_side_holds(&lead));
        l.mark_stale(&l.differs_from(&lead).unwrap_or_default(), true);
        assert!(l.stale().is_some() && l.ranks(&sub, "where is the header parsed").is_some(), "old vectors still serve the question side");
        assert_eq!(l.fill(&sub, &|| true), 0);
    }

    #[test]
    fn switch_waits_for_full_coverage_then_swaps() {
        let sub = sub();
        let old = Arc::new(lane(0));
        old.set_made_by(header("open weights a"));
        old.fill(&sub, &|| true);
        let cell = RwLock::new(Some(Arc::clone(&old)));
        old.mark_stale("parameters: open weights a -> b", false);
        let next = Arc::new(lane(1));
        next.set_made_by(header("b"));
        let once = std::cell::Cell::new(0);
        next.fill(&sub, &|| {
            once.set(once.get() + 1);
            once.get() <= 1
        });
        assert!(switch(&cell, &next, &sub).is_err_and(|e| e.contains("covers 1 of 2")), "a partial lane is not switched in");
        assert!(cell.read().is_ok_and(|g| g.as_ref().is_some_and(|l| Arc::ptr_eq(l, &old))));
        assert!(switch(&cell, &old, &sub).is_err_and(|e| e.contains("stale")));
        next.fill(&sub, &|| true);
        let held_before = cell.read().ok().and_then(|g| g.clone()).expect("a lane");
        let replaced = switch(&cell, &next, &sub).expect("full coverage switches").expect("the old lane");
        assert!(Arc::ptr_eq(&replaced, &old) && Arc::ptr_eq(&held_before, &old));
        let now = cell.read().ok().and_then(|g| g.clone()).expect("a lane");
        assert!(Arc::ptr_eq(&now, &next) && now.stale().is_none() && now.ready(&sub));
        let r = now.ranks(&sub, "where is the header parsed").expect("the new lane ranks");
        let first = r.place.iter().find(|(_, p)| **p == 0).and_then(|(i, _)| sub.chunks.get(*i)).map(|c| c.file);
        assert_eq!(first, Some(0), "the new values rank the header piece first");
        assert!(held_before.ranks(&sub, "x").is_none(), "a reader holding the old lane gets nothing stale");
    }
}
