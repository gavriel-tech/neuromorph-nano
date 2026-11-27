//! Binds the Core Brain's embedder and reranker slots (law 9: nothing fine-tuned). Since
//! 2026-11-09 each slot runs the released model made for its job through its own assembly:
//! the embedder Qwen3-Embedding-0.6B through `qwen3-embedding` (ending in `lastpool`), the
//! reranker Qwen3-Reranker-0.6B through `qwen3-reranker` (ending in `labelsoftmax`, P(yes)).
//! When a learned readout unit is switched on, the reranker stays on the Qwen3-0.6B base
//! population (`qwen3-reference`), the features that unit was fitted on. The exact controls
//! keep running beside the engines (rule 8); the reranker orders only, the control's floor
//! decides admission (law 6). Also the population read for `nm_weights`.
use crate::forage::Organism;
use nmbrain::slot::Engine;
use nmengn::model::{Model, Population};
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::path::Path;
use std::sync::{Arc, Mutex};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmapi.neural", FaultKind::Config, m)
}

/// Withhold floor on the released reranker's best P(yes), qualified in R1 (2026-11-10).
pub const WITHHOLD_FLOOR: f32 = 0.5;

/// The population id every role uses.
pub const POPULATION: &str = "qwen3-0.6b";

/// A served model a call boundary can replace: opened once, and bound again with a parameter overlay
/// applied when one changes (`settle_overlays`). Readers take their own `Arc`, so a call already
/// running keeps the model it started with.
///
/// 2026-11-27 (MCP stability fix): `opening` is held while the model is opened, so a second
/// caller (the start's background load and a first call, or two threads of one call) waits for
/// that one load and takes the same model. Before, each caller opened its own copy (the open
/// was evaluated before `get_or_init`), and the copy that lost was read into RAM for nothing.
struct Cell(std::sync::RwLock<Option<Arc<Model>>>, Mutex<()>);
impl Cell {
    const fn new() -> Self {
        Self(std::sync::RwLock::new(None), Mutex::new(()))
    }
    /// Holds the cell's opening lock (a poisoned lock is taken as it is: the open is retried).
    fn opening(&self) -> std::sync::MutexGuard<'_, ()> {
        self.1.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    fn get(&self) -> Option<Arc<Model>> {
        self.0.read().ok().and_then(|g| g.clone())
    }
    /// The model held, or `m` held from now on when none is.
    fn get_or_init(&self, m: Arc<Model>) -> Arc<Model> {
        let Ok(mut g) = self.0.write() else { return m };
        Arc::clone(g.get_or_insert(m))
    }
    /// Holds `m` when none is held; false when one is.
    fn set(&self, m: Arc<Model>) -> bool {
        let Ok(mut g) = self.0.write() else { return false };
        if g.is_some() {
            return false;
        }
        *g = Some(m);
        true
    }
    fn replace(&self, m: Arc<Model>) {
        if let Ok(mut g) = self.0.write() {
            *g = Some(m);
        }
    }
}

static MODEL: Cell = Cell::new();
static RERANK_MODEL: Cell = Cell::new();
static EMBED_MODEL: Cell = Cell::new();
static LAST: Mutex<String> = Mutex::new(String::new());

/// One role over a model.
struct Role {
    model: Arc<Model>,
    rerank: bool,
    /// The model is the released role model with its own assembly (2026-11-09), not the base.
    card: bool,
    /// A learned readout unit Luis switched on (S11 revision 4); `None` changes nothing.
    readout: Option<nmlearn::readout::Active>,
}
impl Role {
    /// (hidden state at the last position of `final_norm`, base yes-minus-no score) of a pair.
    fn readout_features(&self, query: &str, doc: &str) -> Result<(Vec<f32>, f32), Fault> {
        let m = &self.model;
        let tok = |s: &str| {
            m.pop
                .tokenizer
                .encode(s)
                .ok()
                .and_then(|v| v.first().and_then(|x| usize::try_from(*x).ok()))
                .ok_or_else(|| bad(format!("no token for {s:?}")))
        };
        let (yes, no) = (tok(" yes")?, tok(" no")?);
        let mut d_ids = m.pop.tokenizer.encode(doc).map_err(|e| bad(e.to_string()))?;
        d_ids.truncate(nmengn::roles::MAX_TOKENS);
        let ids = m.pop.tokenizer.encode(&nmengn::roles::prompt(query, &m.pop.tokenizer.decode(&d_ids))).map_err(|e| bad(e.to_string()))?;
        let targets = [(m.role_node("final_norm")?, "y".to_owned()), (m.role_node("head")?, "logits".to_owned())];
        let dev = if m.pop.gpu.is_some() { nmengn::placement::Device::Gpu } else { nmengn::placement::Device::Cpu };
        let out = m.forward(dev, &ids, &targets, &nmengn::model::RunOpts::default())?;
        let y = nmengn::model::values(out.get(&targets[0]).ok_or_else(|| bad("no hidden state"))?)?;
        let l = nmengn::model::values(out.get(&targets[1]).ok_or_else(|| bad("no logits"))?)?;
        let width = 1024.min(y.len());
        Ok((
            y.get(y.len() - width..).map(<[f32]>::to_vec).unwrap_or_default(),
            l.get(yes).copied().unwrap_or(0.0) - l.get(no).copied().unwrap_or(0.0),
        ))
    }
}
impl Engine for Role {
    fn embed(&self, text: &str) -> Result<Vec<f32>, Fault> {
        let (v, c, ms) = if self.card { nmengn::roles::card_embed(&self.model, text)? } else { nmengn::roles::embed(&self.model, text)? };
        note(&format!("embed on {} in {ms:.0} ms ({})", c.device.name(), c.why));
        Ok(v)
    }
    fn rerank(&self, query: &str, doc: &str) -> Result<f32, Fault> {
        let (s, c, ms) = if self.card {
            nmengn::roles::card_rerank(&self.model, nmengn::roles::CODE_INSTRUCTION, query, doc)?
        } else {
            nmengn::roles::rerank(&self.model, query, doc)?
        };
        note(&format!("rerank on {} in {ms:.0} ms ({})", c.device.name(), c.why));
        Ok(s)
    }
    fn rerank_set(&self, query: &str, docs: &[String]) -> Result<Option<Vec<f32>>, Fault> {
        // The released reranker scores a set with one prefilled head (2026-11-10).
        if self.card && self.readout.is_none() {
            let dev = if self.model.pop.gpu.is_some() { nmengn::placement::Device::Gpu } else { nmengn::placement::Device::Cpu };
            let t = std::time::Instant::now();
            let v = nmengn::roles::card_rerank_set(&self.model, dev, nmengn::roles::CODE_INSTRUCTION, query, docs)?;
            note(&format!(
                "rerank set of {} on {} in {:.0} ms (one prefilled head)",
                docs.len(),
                dev.name(),
                t.elapsed().as_secs_f64() * 1000.0
            ));
            return Ok(Some(v));
        }
        let Some(r) = self.readout.as_ref().filter(|r| self.rerank && r.serves(query)) else { return Ok(None) };
        let (mut unit, mut base) = (Vec::new(), Vec::new());
        for d in docs {
            let (h, b) = self.readout_features(query, d)?;
            unit.push(r.score(&h, b));
            base.push(f64::from(b));
        }
        note(&format!("rerank set of {} through readout {}", docs.len(), r.unit));
        #[expect(clippy::cast_possible_truncation, reason = "scores for ordering")]
        Ok(Some(r.order(&unit, &base).into_iter().map(|x| x as f32).collect()))
    }
    fn resident_mb(&self) -> (f32, f32) {
        let (ram, gpu) = self.model.pop.prov.totals();
        #[expect(clippy::cast_precision_loss, reason = "megabytes for a report")]
        let mb = |b: u64| (b >> 20) as f32;
        // A released role model is this role's own; the base population is shared by both
        // roles, so each reports half and the total is counted once.
        if self.card { (mb(ram), mb(gpu)) } else { (mb(ram) / 2.0, mb(gpu) / 2.0) }
    }
    fn release_device(&self) {
        self.model.pop.release_device();
    }
    fn label(&self) -> String {
        format!(
            "{} {} over {} ({}), {}",
            if self.rerank { "rerank" } else { "embed" },
            self.model.asm.name,
            self.model.pop.card.id,
            self.model.pop.card.sha_main.get(..12).unwrap_or(""),
            self.model.pop.gpu_note
        )
    }
}
fn note(s: &str) {
    if let Ok(mut l) = LAST.lock() {
        s.clone_into(&mut l);
    }
}
/// The last placement note of an engine call.
#[must_use]
pub fn last_note() -> String {
    LAST.lock().map(|l| l.clone()).unwrap_or_default()
}

/// Bind an assembly after applying the active precision setting and prefetching any `q8`
/// tensors across CPU worker threads, holding the `Q8` views alive until `Model::bind` finishes.
fn bind_prefetched(pop: &Arc<Population>, mut asm: nmengn::Assembly) -> Result<Model, Fault> {
    // Roadmap Phase 4.5 (2026-11-24): every forward of a bound MicroModel tells the live view.
    crate::live::install();
    let _ = nmengn::model::apply_precision(&mut asm, &pop.card.id);
    let _held = if asm.nodes.iter().any(|n| n.precision == "q8") {
        pop.prefetch_q8(&asm, nmengn::cpu::threads())?
    } else {
        Vec::new()
    };
    Model::bind(pop, asm)
}

/// The shared model, opened on first use.
/// # Errors
/// The population or the assembly cannot be opened.
pub fn model(home: &Path) -> Result<Arc<Model>, Fault> {
    if let Some(m) = MODEL.get() {
        return Ok(m);
    }
    let _opening = MODEL.opening();
    if let Some(m) = MODEL.get() {
        return Ok(m);
    }
    let m = MODEL.get_or_init(open_reader(home)?);
    note_bound("reader", home, &m);
    seat_reader(home, &m);
    Ok(m)
}

/// The Reader's model opened and bound on its base source (the open weights or a base cartridge,
/// with its parameter overlay), not yet held.
fn open_reader(home: &Path) -> Result<Arc<Model>, Fault> {
    let mut asm = nmengn::Assembly::load(&home.join("designs/assemblies"), "qwen3-reference")?;
    served_width(&mut asm, POPULATION, false);
    if !open_weights_present(home, POPULATION) {
        return Ok(Arc::new(open_standalone_base(home, POPULATION, &asm, true, false)?));
    }
    // The model alone holds its population, so seating a base cartridge releases the open weights.
    let pop = open_population(home, POPULATION, true)?;
    let open = bind_prefetched(&pop, asm)?;
    Ok(Arc::new(on_base(open)))
}

/// What the Reader's model carries once held: the distilled cartridge and the request read.
fn seat_reader(home: &Path, m: &Arc<Model>) {
    // The distilled cartridge plugged in for the Reader, on its final norm (`distil`).
    let line = crate::distil::seat(home, m, "reader");
    if !line.is_empty() {
        eprintln!("neural: reader: {line}");
    }
    // Lane R: the request's drone reads its kind and relations with this reader when its model
    // form is on (`NANO_REQ_MODEL=on`, off by default).
    let _ = crate::request_read::install_with(m);
}

/// The released reranker model (Qwen3-Reranker-0.6B, its own assembly), the one copy the reranker
/// slot uses, opened on first use (2026-11-10, for the Core Brain's decision step).
/// # Errors
/// The population or the assembly cannot be opened.
pub fn reranker_model(home: &Path) -> Result<Arc<Model>, Fault> {
    // 2026-11-25: the model the reranker role runs on (`nmtype::role_models`), Qwen3-Reranker-0.6B
    // unless the user chose another installed reranker.
    role_model(home, &RERANK_MODEL, "qwen3-reranker", &nmtype::role_models::model("reranker"))
}

/// The released embedder (Qwen3-Embedding-0.6B, its own assembly), the one copy the embedder slot
/// uses, opened on first use (2026-11-10, for the judge test over the drones' shortlist).
/// # Errors
/// The population or the assembly cannot be opened.
pub fn embedder_model(home: &Path) -> Result<Arc<Model>, Fault> {
    role_model(home, &EMBED_MODEL, "qwen3-embedding", &nmtype::role_models::model("embedder"))
}

/// The writer MicroModel's open weights (Qwen3-1.7B).
pub const WRITER: &str = "qwen3-1.7b";

/// The writer (Qwen3-1.7B on the reference assembly), opened on demand and not held here: the
/// caller keeps it while it writes and drops it to free the card (the writer loads on demand,
/// master design 3.8 rule 6). It binds at the selector's width for the writer and seats its base
/// cartridge as the other MicroModels do (`on_base`): complete, connected and checked, the open
/// weights released; otherwise the open weights serve with the reason logged.
/// # Errors
/// The population or the assembly cannot be opened.
pub fn writer_model(home: &Path, gpu: bool) -> Result<Arc<Model>, Fault> {
    let mut asm = nmengn::Assembly::load(&home.join("designs/assemblies"), "qwen3-reference")?;
    // 2026-11-25: the model the writer role runs on (`nmtype::role_models`), WRITER unless chosen.
    let id = nmtype::role_models::model("writer");
    served_width(&mut asm, &id, false);
    let m = if !open_weights_present(home, &id) {
        Arc::new(open_standalone_base(home, &id, &asm, gpu, false)?)
    } else {
        // The model alone holds its population, so seating a base cartridge releases the open weights.
        let pop = open_population(home, &id, gpu)?;
        let open = bind_prefetched(&pop, asm)?;
        Arc::new(on_base(open))
    };
    // The distilled cartridge plugged in for the Writer, on its final norm (`distil`).
    let line = crate::distil::seat_writer(home, &m);
    if !line.is_empty() {
        eprintln!("neural: writer: {line}");
    }
    Ok(m)
}

/// Every model bound in this process, by the slot or role that uses it (anatomy zoom).
#[must_use]
pub fn bound_models() -> Vec<(&'static str, Arc<Model>)> {
    let mut v = Vec::new();
    if let Some(m) = MODEL.get() {
        v.push(("base", m));
    }
    if let Some(m) = EMBED_MODEL.get() {
        v.push(("embedder", m));
    }
    if let Some(m) = RERANK_MODEL.get() {
        v.push(("reranker", m));
    }
    v
}

/// The shared model when it has been opened in this process (units attach to it).
#[must_use]
pub fn bound_model() -> Option<Arc<Model>> {
    MODEL.get()
}
/// Bind the embedder and reranker slots; returns what was bound.
/// # Errors
/// As `model`.
pub fn bind(org: &Organism) -> Result<String, Fault> {
    let e = embedder_model(&org.home)?;
    org.slots.bind("embedder", Arc::new(Role { model: Arc::clone(&e), rerank: false, card: true, readout: None }))?;
    let readout = nmlearn::readout::Active::open(&org.home);
    let (r, card) = if readout.is_some() {
        (model(&org.home)?, false)
    } else {
        (reranker_model(&org.home)?, true)
    };
    org.slots.bind("reranker", Arc::new(Role { model: Arc::clone(&r), rerank: true, card, readout }))?;
    // Qualified 2026-11-10 (R1, `runs/2026-11-10-R1-reranker.md`): with the released reranker
    // bound, a pack whose best candidate scores below P(yes) 0.5 is withheld. On by default;
    // `serve --rerank-withhold 0` switches it off. Ordering did not qualify and stays off.
    if card {
        org.slots.set_withhold(WITHHOLD_FLOOR);
    }
    Ok(format!(
        "embedder bound to {} over {}; reranker bound to {} over {}; {}",
        e.asm.name, e.pop.card.id, r.asm.name, r.pop.card.id, e.pop.gpu_note
    ))
}

/// Bind only the embedder slot to the released Qwen3-Embedding-0.6B (`serve --dense`, 2026-11-10):
/// the dense lane without the reranker's cost per pack.
/// # Errors
/// As `model`.
pub fn bind_embedder(org: &Organism) -> Result<String, Fault> {
    let e = embedder_model(&org.home)?;
    org.slots.bind("embedder", Arc::new(Role { model: Arc::clone(&e), rerank: false, card: true, readout: None }))?;
    Ok(format!("embedder bound to {} over {}; {}", e.asm.name, e.pop.card.id, e.pop.gpu_note))
}

/// Chunks per batched embedding call of the dense lane's index (Roadmap Phase 4.6, 2026-11-24:
/// 64, was 16, so the parts inside a call can grow).
pub const DENSE_BATCH: usize = 64;
/// Chunks the index embeds per entry into the device gate at the start: a question that arrives
/// waits for at most one part of background work to end before it has the card to itself.
pub const DENSE_PART: usize = 4;
/// Roadmap Phase 4.6 (2026-11-24): the part adapts to the card. It doubles while a part takes
/// under `PART_GROW_MS` and halves when one takes over `PART_SHRINK_MS`, between 1 and
/// `PART_MAX` chunks, so a question still waits about a quarter second at most while the lane is
/// no longer held to 4 chunks a forward (cold start: 2,999 pieces at about 6.5 a second).
static PART: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(DENSE_PART);
const PART_MAX: usize = 64;
const PART_GROW_MS: u128 = 150;
const PART_SHRINK_MS: u128 = 300;

/// The next part size after a part of `k` chunks took `ms` milliseconds.
fn next_part(k: usize, ms: u128) -> usize {
    if ms > PART_SHRINK_MS {
        (k / 2).max(1)
    } else if ms < PART_GROW_MS {
        (k * 2).min(PART_MAX)
    } else {
        k
    }
}

/// One part of the index on the card, under the device gate as a background job; a part that
/// fails (card memory) is split in halves and tried again, down to one chunk.
fn index_part(m: &Model, texts: &[String], cap: usize) -> Option<Vec<Vec<f32>>> {
    let got = {
        let _g = nmengn::placement::gpu_gate().enter("dense lane indexing", false);
        nmengn::roles::card_embed_batch(m, texts, cap, nmengn::roles::batch_tokens(m, cap + 1)).ok()
    };
    match got {
        Some(v) => Some(v),
        None if texts.len() > 1 => {
            let (a, b) = texts.split_at(texts.len() / 2);
            let mut v = index_part(m, a, cap)?;
            v.extend(index_part(m, b, cap)?);
            Some(v)
        }
        None => None,
    }
}

/// Long pieces are embedded in parts (`NANO_DENSE_PARTS=on`; off by default).
pub static PARTS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The store's header names the parts' length (always, except for a measurement). Roadmap Phase 4
/// (2026-11-24): the store records it per row (`nmsubs::dense::store`, row recipes), so switching
/// parts on or off embeds again only the rows made the other way; the store is not set aside.
static PARTS_NAMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The input length, in tokens, at which a layer's attention (two products over the input per head)
/// costs as much as the rest of the layer (the projections and the feed-forward), from the model's
/// own dimensions; past it the cost of each token grows with the input's length.
#[must_use]
pub fn attention_crossover(m: &Model) -> usize {
    let d = m.pop.dims;
    let q = d.heads * d.head_dim;
    let rest = d.hidden * (q + 2 * d.kv_heads * d.head_dim) + q * d.hidden + 3 * d.hidden * d.inter;
    (rest / (2 * q).max(1)).max(1)
}

/// `text` (a path line, then lines) in parts of at most `limit` tokens, each part with the path
/// line; one part when it fits. A single line longer than `limit` is a part of its own.
fn split_long(m: &Model, text: &str, limit: usize) -> Vec<String> {
    let count = |t: &str| m.pop.tokenizer.encode(t).map_or(t.len() / 4, |v| v.len());
    if count(text) <= limit {
        return vec![text.to_owned()];
    }
    let (head, body) = text.split_once('\n').unwrap_or(("", text));
    let room = limit.saturating_sub(count(head) + 1).max(1);
    let mut parts = Vec::new();
    let (mut cur, mut used) = (String::new(), 0_usize);
    for line in body.split('\n') {
        let n = count(line) + 1;
        if used > 0 && used + n > room {
            parts.push(format!("{head}\n{cur}"));
            (cur, used) = (String::new(), 0);
        }
        if used > 0 {
            cur.push('\n');
        }
        cur.push_str(line);
        used += n;
    }
    if used > 0 || parts.is_empty() {
        parts.push(format!("{head}\n{cur}"));
    }
    parts
}

/// One vector per owner: the normalised mean of its parts' normalised vectors (a piece in one part
/// keeps its vector).
fn merge_parts(vs: Vec<Vec<f32>>, owner: &[usize]) -> Vec<Vec<f32>> {
    let norm = |v: &mut Vec<f32>| {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            v.iter_mut().for_each(|x| *x /= n);
        }
    };
    if owner.windows(2).all(|w| w.first() != w.last()) {
        return vs;
    }
    let mut out: Vec<Vec<f32>> = Vec::new();
    let mut last = usize::MAX;
    for (mut v, &o) in vs.into_iter().zip(owner) {
        norm(&mut v);
        if o == last {
            if let Some(acc) = out.last_mut() {
                acc.iter_mut().zip(&v).for_each(|(a, b)| *a += b);
            }
        } else {
            out.push(v);
            last = o;
        }
    }
    out.iter_mut().for_each(norm);
    out
}

/// Files the memory system cited most recently, newest first, each once (the RAM-only dense
/// lane builds these first; refinement step 1, item 7, safeguard b).
#[must_use]
pub fn recent_cited_files(org: &Organism) -> Vec<String> {
    let Ok(m) = org.memory.lock() else { return Vec::new() };
    let mut out: Vec<String> = Vec::new();
    for e in m.episodic.latest(usize::MAX) {
        for c in e.evidence.iter().filter(|c| !c.starts_with("note:")) {
            let loc = c.split_once(' ').map_or(c.as_str(), |(a, _)| a);
            let path = loc.rsplit_once(':').map_or(loc, |(p, _)| p).replace('\\', "/");
            if !path.is_empty() && !out.contains(&path) {
                out.push(path);
            }
        }
    }
    out
}

/// The material's files, most recently changed first (by the file system's modified time; files
/// it cannot read go last).
fn recently_changed(sub: &nmsubs::Substrate) -> Vec<String> {
    let mut files: Vec<(std::time::SystemTime, String)> = sub
        .files
        .iter()
        .map(|f| {
            let at = std::fs::metadata(sub.root.join(&f.path)).and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
            (at, f.path.replace('\\', "/"))
        })
        .collect();
    files.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    files.into_iter().map(|(_, p)| p).collect()
}

/// Start the dense lane: chunk vectors from the bound embedder, keyed by chunk content hash and
/// held in RAM only (2026-11-10, refinement step 1, item 7: direction of Luis Jake Gabriel III of
/// 08:12 UTC that encodings are never written to disk; analysis
/// `docs/2026-11-10-neumor-analysis-g-vectors-and-g1-placement.md` question 1, option O3; before
/// this date they were stored in `scratch/dense/<model>-<root>.vec`, which is now neither read
/// nor written and is left in place). Built in the background through the device gate (questions
/// go first), files the memory system cited most recently first; the meaning ranking is used
/// only once every current chunk has a vector (until then the path runs without it, and the
/// lane's statistics say so); after each sync the vectors of vanished chunks are dropped and
/// changed chunks embedded again. Every question then gets one fused ranking that all tentacles
/// share (`nmsubs::dense`).
/// # Errors
/// No embedder engine bound.
pub fn start_dense(org: &Organism) -> Result<String, Fault> {
    let model = EMBED_MODEL.get().ok_or_else(|| bad("dense lane: the embedder is not bound"))?;
    let sub = org.sub();
    // The lane's form in the readers' views: `NANO_DENSE_FORM` `candidates` or `fused` names one
    // for a measurement; otherwise the form `nmsubs::dense::CANDIDATES_ONLY` holds.
    match std::env::var("NANO_DENSE_FORM").as_deref() {
        Ok("candidates") => nmsubs::dense::CANDIDATES_ONLY.store(true, std::sync::atomic::Ordering::Relaxed),
        Ok("fused") => nmsubs::dense::CANDIDATES_ONLY.store(false, std::sync::atomic::Ordering::Relaxed),
        _ => {}
    }
    // `NANO_DENSE_SIGNAL=on`: the meaning signal in the drones' evidence, for a measurement.
    if std::env::var("NANO_DENSE_SIGNAL").as_deref() == Ok("on") {
        nmsubs::dense::MEANING_SIGNAL.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    // The question's instruction (step 1l): the request's drone gives its kind; `NANO_DENSE_TASK`
    // (`handover`, `answer`, `kind`) names the form for a measurement.
    // `NANO_DENSE_LEAD=on`: code windows embedded with their lead (step 1l, change C), for a
    // measurement; the store's header names the recipe, so no vector of the other recipe is used.
    // Roadmap Phase 4 (2026-11-24): the store records each row's recipe, so a store made without
    // the lead stays in use; only rows of pieces that have a lead are embedded again (through the
    // lane's usual fill), and switching back embeds only those again.
    // Roadmap Phase 4.6 (2026-11-25): on by default (`NANO_DENSE_LEAD=off` for diagnosis). Full
    // sets on a copy of the bench store: large code 23, 37, 44, 41 to 25, 39, 47, 44; dev-large-2
    // 61, 81, 91, 82 to 61, 80, 89, 83, decoys 20/20 (the gate holds: first 61, pack 88, every 79);
    // the dev-large-2 pack give-back of 2 is its next climb.
    if std::env::var("NANO_DENSE_LEAD").as_deref() != Ok("off") {
        nmsubs::dense::WITH_LEAD.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let _ = nmsubs::dense::KIND_OF.set(|q| nmpod::intake::Request::read(q, &[]).kind);
    match std::env::var("NANO_DENSE_TASK").as_deref() {
        Ok("handover") => nmsubs::dense::TASK_FORM.store(0, std::sync::atomic::Ordering::Relaxed),
        Ok("answer") => nmsubs::dense::TASK_FORM.store(1, std::sync::atomic::Ordering::Relaxed),
        Ok("kind") => nmsubs::dense::TASK_FORM.store(2, std::sync::atomic::Ordering::Relaxed),
        _ => {}
    }
    // The fusion's numbers (step 1l, change B), for a measurement: `NANO_DENSE_LEXW`,
    // `NANO_DENSE_RRFK`, `NANO_DENSE_TOP`, `NANO_DENSE_K` name a value; `derived` in
    // `NANO_DENSE_LEXW` or `NANO_DENSE_TOP` derives it per request; `NANO_DENSE_TRACE` names a file
    // the derived numbers are appended to.
    if let Ok(mut t) = nmsubs::dense::TUNE.write() {
        let var = |k: &str| std::env::var(k).ok();
        match var("NANO_DENSE_LEXW").as_deref() {
            Some("derived") => t.derive_weight = true,
            Some(v) => t.lexical_weight = v.parse().ok(),
            None => {}
        }
        t.rrf_k = var("NANO_DENSE_RRFK").and_then(|v| v.parse().ok()).or(t.rrf_k);
        match var("NANO_DENSE_TOP").as_deref() {
            Some("derived") => t.derive_top = true,
            Some(v) => t.top = v.parse().ok(),
            None => {}
        }
        t.dense_k = var("NANO_DENSE_K").and_then(|v| v.parse().ok()).or(t.dense_k);
    }
    if let Some(p) = std::env::var_os("NANO_DENSE_TRACE") {
        let _ = nmsubs::dense::TRACE.set(std::path::PathBuf::from(p));
    }
    let lane = new_lane(org, &model);
    // The first start (on by default since 2026-11-18 07:53 UTC, T-055 verified; `NANO_DENSE_PARTIAL=off`
    // waits for full coverage): rank at
    // partial coverage, each question's likely files built next, and after memory's files the most
    // recently changed ones.
    // `NANO_DENSE_FILL=off` and `NANO_DENSE_LOAD=<share>` hold the lane at a loaded coverage, for
    // measuring the first start without the card.
    // `NANO_DENSE_PARTS=on`: long pieces in parts, named in the store's header. `unnamed`: the same
    // without the header's name, for measuring over a copy of a store whose long pieces are missing.
    // Roadmap Phase 4 (2026-11-24): with `on`, rows record the parts' length; rows of a store made
    // whole are embedded again in parts one by one through the fill (the store stays in use, and
    // with `NANO_DENSE_LEAD=on` too, one pass covers both). Rows embedded by the one-at-a-time
    // fallback (`fill`) are recorded under the same recipe although that path embeds them whole.
    match std::env::var("NANO_DENSE_PARTS").as_deref() {
        Ok("on") => {
            PARTS.store(true, std::sync::atomic::Ordering::Relaxed);
            PARTS_NAMED.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        Ok("unnamed") => PARTS.store(true, std::sync::atomic::Ordering::Relaxed),
        _ => {}
    }
    let partial = std::env::var("NANO_DENSE_PARTIAL").as_deref() != Ok("off");
    if std::env::var("NANO_DENSE_FUSED").as_deref() == Ok("head") {
        nmsubs::dense::HEAD_RULE.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    lane.set_partial(partial);
    if std::env::var("NANO_DENSE_FILL").as_deref() == Ok("off") {
        nmsubs::dense::FILL.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    let mut first_files = recent_cited_files(org);
    if partial {
        for f in recently_changed(&sub) {
            if !first_files.contains(&f) {
                first_files.push(f);
            }
        }
    }
    let prioritised = first_files.len();
    if let Some(share) = std::env::var("NANO_DENSE_LOAD").ok().and_then(|v| v.parse::<f64>().ok()).filter(|s| (0.0..1.0).contains(s)) {
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss, reason = "a share of the piece count")]
        let n = (share * sub.chunks.len() as f64) as usize;
        nmsubs::dense::LOAD_LIMIT.store(n, std::sync::atomic::Ordering::Relaxed);
    }
    lane.set_priority(first_files);
    // Kept on disk (`nmsubs::dense::store`): with memory on and the `dense_cache` setting on, the
    // vectors are written to the project's memory folder and loaded at the next start.
    // With memory off, `NANO_DENSE_DIR` may name a folder for them (a bench or test run over a
    // frozen tree: the material's vectors built once, then only the question encoded per call).
    // What made the vectors, named whether or not they are kept on disk, so a change of the
    // embedder's served values is seen (`follow_embedder`).
    lane.set_made_by(dense_header(&model));
    let kept = kept_dir(org).map(|dir| lane.keep_on_disk(&dir, dense_header(&model), &sub));
    if let Ok(mut d) = sub.dense.write() {
        *d = Some(Arc::clone(&lane));
    }
    let (tx, rx) = std::sync::mpsc::channel::<Arc<nmsubs::Substrate>>();
    if let Ok(mut f) = org.dense_feed.lock() {
        *f = Some(tx);
    }
    let first = Arc::clone(&sub);
    let cell = Arc::clone(&sub.dense);
    drop(lane);
    std::thread::spawn(move || {
        let mut cur = first;
        loop {
            // The lane in place now, on the embedder bound now: a lane built for new embedder
            // values replaces a stale one (`follow_embedder`), and a stale lane is never filled.
            // The lock is let go before the build, so the switch never waits on it.
            let lane = cell.read().ok().and_then(|d| d.clone());
            if let Some(lane) = lane.filter(|l| l.stale().is_none()) {
                // Safeguard (c): vectors of chunks this version no longer holds are dropped;
                // changed chunks have new keys and are embedded by the fill.
                lane.retain(&cur);
                // On the card, chunks are embedded whole, many per forward; what fails falls back
                // to one at a time.
                if let Some(m) = EMBED_MODEL.get().filter(|m| m.pop.gpu.is_some()) {
                    let embed = batch_embed(m);
                    // Roadmap Phase 4.5 (2026-11-24): the dense lane's build progress in the live
                    // view, only when there is something to build.
                    let (have, total) = lane.coverage(&cur);
                    let live = (have < total).then(|| {
                        crate::live::progress_start("dense", have, total);
                        crate::live::stage("dense", "embedder")
                    });
                    lane.fill_batched(&cur, &embed, DENSE_BATCH, &|| true);
                    if let Some(mut st) = live {
                        let (have, total) = lane.coverage(&cur);
                        st.count("vectors", have);
                        st.count("pieces", total);
                        crate::live::progress_end("dense", have, total);
                    }
                }
                lane.fill(&cur, &|| true);
            }
            match rx.recv() {
                Ok(mut next) => {
                    while let Ok(n) = rx.try_recv() {
                        next = n;
                    }
                    cur = next;
                }
                Err(_) => break,
            }
        }
    });
    let held = kept.map_or_else(|| "vectors held in RAM only (no store file read or written)".to_owned(), |k| format!("vectors kept on disk: {k}"));
    Ok(format!(
        "dense lane over {} chunks, {held}; built in the background through the device gate, {prioritised} files first; {}",
        sub.chunks.len(),
        if partial {
            "the meaning ranking is used from the first vector, each question's likely files built next"
        } else {
            "the meaning ranking is used once every chunk has a vector"
        }
    ))
}

/// The dense lane over the embedder `model`: pieces encoded through the embedder slot as background
/// work, questions on the device the engine's placement chooses (or the one `NANO_DENSE_QUERY`
/// names), with the project's lens and the distilled request lens applied by the lane.
fn new_lane(org: &Organism, model: &Arc<Model>) -> Arc<nmsubs::dense::DenseLane> {
    let (s1, s2) = (Arc::clone(&org.slots), Arc::clone(&org.slots));
    // Step P2: each background chunk embedding enters the device gate as a background job, so
    // it never shares the card with the tentacle pass's encoding and reading (it waits while the
    // pass holds the gate, and the wait is reported in the pass's note). Query embeddings (the
    // second function) run in the asking thread and are not gated.
    let index = move |t: &str| {
        let _bg = nmengn::residency::background();
        let _g = nmengn::placement::gpu_gate().enter("dense lane indexing", false);
        s1.embed_neural(t)
    };
    // The question's vector: on the device the engine's placement chooses, or on the one
    // `NANO_DENSE_QUERY` (`cpu` or `gpu`) names, for measuring the two.
    let qdev = match std::env::var("NANO_DENSE_QUERY").as_deref() {
        Ok("cpu") => Some(nmengn::placement::Device::Cpu),
        Ok("gpu") if model.pop.gpu.is_some() => Some(nmengn::placement::Device::Gpu),
        _ => None,
    };
    let qm = Arc::clone(model);
    let query = move |t: &str| match qdev {
        Some(d) => {
            let at = std::time::Instant::now();
            let v = nmengn::roles::card_embed_on(&qm, d, t).ok();
            note(&format!("embed on {} in {:.0} ms (named by NANO_DENSE_QUERY)", d.name(), at.elapsed().as_secs_f64() * 1000.0));
            v
        }
        None => s2.embed_neural(t),
    };
    // Lane B (2026-11-19): the project's lens on the request's vector when it is in force (on by
    // default once a fitted file passed its control, `NANO_ADAPTER_LENS=off` for diagnosis, and its
    // file fitted on this embedder); otherwise the vector untouched.
    // Roadmap Phase 0 item 3 (2026-11-23): the lens is applied by the lane (`set_query_lens`), which
    // keeps its order and the plain scale; `query` gives the plain vector (content-free questions
    // included, so the calibration stays on the plain scale).
    let lane = Arc::new(nmsubs::dense::DenseLane::in_ram(Arc::new(index), Arc::new(query)));
    // With a distilled request lens plugged in, it maps the request first (`distil::query_lens`);
    // stored piece vectors are never mapped by it.
    lane.set_query_lens(Arc::new(|v: &[f32]| crate::distil::query_lens(v)));
    lane.set_doc_lens(Arc::new(crate::adapter::lens_doc));
    // The lane's query-side caches follow the lens.
    lane.set_query_epoch(Arc::new(crate::adapter::lens_epoch));
    lane
}

/// The folder the dense lane keeps its vectors in: with memory on and the `dense_cache` setting
/// on, the project's memory folder's `dense/`; with memory off, the folder `NANO_DENSE_DIR` names
/// (a bench or test run over a frozen tree); otherwise none (RAM only).
fn kept_dir(org: &Organism) -> Option<std::path::PathBuf> {
    let test_dir = (!org.memory_on).then(|| std::env::var_os("NANO_DENSE_DIR")).flatten().map(std::path::PathBuf::from);
    match test_dir {
        Some(dir) => Some(dir),
        None => (org.memory_on && nmsubs::dense::store::cache_on(&org.home))
            .then(|| org.memory.lock().ok().map(|m| m.dir.join(nmsubs::dense::store::DIR)))
            .flatten(),
    }
}

/// The lane's batched piece encoder on the card over `m`: background work that enters the device
/// gate per part (`index_part`), so it waits while a question is answered, and room made for it
/// never releases a served model's weights.
fn batch_embed(m: Arc<Model>) -> impl Fn(&[String]) -> Option<Vec<Vec<f32>>> {
    move |texts: &[String]| {
        let _bg = nmengn::residency::background();
        let cap = nmengn::roles::input_cap(&m, nmengn::placement::Device::Gpu, 1);
        // Long pieces in parts (`PARTS`): each part within the length where attention starts to
        // cost more than the rest of a layer, the piece's vector the mean of its parts' vectors;
        // nothing of the piece is left out.
        let (texts, owner) = if PARTS.load(std::sync::atomic::Ordering::Relaxed) {
            let limit = attention_crossover(&m).min(cap);
            let mut all = Vec::new();
            let mut owner = Vec::new();
            for (i, t) in texts.iter().enumerate() {
                for p in split_long(&m, t, limit) {
                    all.push(p);
                    owner.push(i);
                }
            }
            (all, owner)
        } else {
            (texts.to_vec(), (0..texts.len()).collect())
        };
        let mut out = Vec::with_capacity(texts.len());
        let mut rest: &[String] = &texts;
        while !rest.is_empty() {
            let k = PART.load(std::sync::atomic::Ordering::Relaxed).clamp(1, PART_MAX).min(rest.len());
            let (part, tail) = rest.split_at(k);
            let t = std::time::Instant::now();
            out.extend(index_part(&m, part, cap)?);
            // A part cut short by the end of the group says nothing about growing.
            let ms = t.elapsed().as_millis();
            if k == PART.load(std::sync::atomic::Ordering::Relaxed) || ms > PART_SHRINK_MS {
                PART.store(next_part(k, ms), std::sync::atomic::Ordering::Relaxed);
            }
            rest = tail;
        }
        crate::live::progress_add("dense", owner.last().map_or(0, |o| o + 1));
        Some(merge_parts(out, &owner))
    }
}

/// Bumped by every change of the embedder's served values the dense lane follows, so a rebuild
/// under way starts again for the newest values.
static FOLLOW_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// A rebuild thread runs (`follow_embedder`).
static FOLLOW_RUNNING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Calls dispatched in this process (`settle_installed`, at every call's start), and when the last
/// one started (ms): the rebuild runs only after the idle Writer's quiet time with no call.
static CALLS_SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LAST_CALL_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// The header the rebuild under way builds for (`rebuild`), `None` when none runs.
static BUILDING: Mutex<Option<nmsubs::dense::store::Header>> = Mutex::new(None);
/// What the dense lane's follow of the embedder did last (empty before the first change).
static FOLLOW_LINE: Mutex<String> = Mutex::new(String::new());

/// What the dense lane's follow of the embedder did last: the change seen, the rebuild's progress
/// or its result; empty while the lane is on the values it started with.
#[must_use]
pub fn dense_follow_line() -> String {
    FOLLOW_LINE.lock().map(|l| l.clone()).unwrap_or_default()
}

fn set_follow_line(line: &str) {
    if let Ok(mut l) = FOLLOW_LINE.lock() {
        line.clone_into(&mut l);
    }
    eprintln!("neural: {line}");
}

/// No call in flight and none started for the idle Writer's quiet time (`writer_idle::quiet_ms`).
fn rebuild_idle() -> bool {
    crate::writer_idle::calls_in_flight() == 0
        && nmtype::now_ms().saturating_sub(LAST_CALL_MS.load(std::sync::atomic::Ordering::SeqCst)) >= crate::writer_idle::quiet_ms()
}

/// The dense lane follows a change of the embedder's served values (a parameter overlay or plug
/// change that bound the embedder again, `settle_overlays`). The vectors held are compared with
/// the embedder now bound by their header (`dense_header`): equal (a lens or request-lens change,
/// which acts on the question side only and moves the lens epoch the lane already follows, or an
/// overlay on another MicroModel), nothing changes. Otherwise the lane is marked stale: its
/// meaning ranking is withheld (unless only the piece-side recipe moved, so questions still encode
/// in the vectors' space) and every question runs on the lexical and structural paths; a lane for
/// the new values is built in the background at idle (`rebuild`), preempted by every call, and put
/// in place in one write once it covers every current piece. Returns the line it logged, `None`
/// when nothing changed or no lane runs.
pub fn follow_embedder(org: &Organism) -> Option<String> {
    let lane = org.sub().dense.read().ok().and_then(|d| d.clone())?;
    let model = EMBED_MODEL.get()?;
    let running = dense_header(&model);
    // A lane already stale is being rebuilt: other values than those it is built for restart it;
    // the same values (a change to another MicroModel) leave it running.
    if lane.stale().is_some() && BUILDING.lock().is_ok_and(|b| b.as_ref() == Some(&running)) {
        return None;
    }
    let why = match lane.stale() {
        Some(prev) => format!("{prev}; then {}", lane.differs_from(&running).unwrap_or_else(|| "values bound again".to_owned())),
        None => lane.differs_from(&running)?,
    };
    let holds = lane.stale().is_none() && lane.query_side_holds(&running);
    lane.mark_stale(&why, holds);
    FOLLOW_GEN.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let line = format!(
        "dense lane: the embedder's served values changed ({why}); {}; a lane for the new values is built in the background at idle",
        if holds { "the vectors held still serve questions" } else { "its meaning ranking is withheld and questions run on the lexical and structural paths" }
    );
    // Stored only: the caller (`settle_overlays`) logs the lines it returns.
    if let Ok(mut l) = FOLLOW_LINE.lock() {
        line.clone_into(&mut l);
    }
    if !FOLLOW_RUNNING.swap(true, std::sync::atomic::Ordering::SeqCst)
        && std::thread::Builder::new().name("nano-dense-rebuild".into()).spawn(rebuild).is_err()
    {
        FOLLOW_RUNNING.store(false, std::sync::atomic::Ordering::SeqCst);
        set_follow_line("dense lane: the rebuild thread did not start; the lane stays stale until the next start");
    }
    Some(line)
}

/// The rebuild thread of `follow_embedder`: for the newest values, a lane over the embedder bound
/// now loads what its store already holds under the new header (a rebuild cut short by a restart
/// resumes), then embeds the rest whenever the process is idle (`rebuild_idle`), stopping at the
/// next batch when a call starts; the old lane's note and name texts are encoded again; once every
/// current piece has a vector the new lane is switched in (`nmsubs::dense::switch`). A newer
/// change starts it again.
fn rebuild() {
    use std::sync::atomic::Ordering::SeqCst;
    loop {
        let epoch = FOLLOW_GEN.load(SeqCst);
        let (Some(org), Some(model)) = (crate::tools::installed(), EMBED_MODEL.get()) else {
            set_follow_line("dense lane: no embedder bound or no project served; the lane stays stale until the next start");
            FOLLOW_RUNNING.store(false, SeqCst);
            break;
        };
        let t = std::time::Instant::now();
        let header = dense_header(&model);
        if let Ok(mut b) = BUILDING.lock() {
            *b = Some(header.clone());
        }
        let sub = org.sub();
        let next = new_lane(org, &model);
        next.set_made_by(header.clone());
        next.set_priority(recent_cited_files(org));
        let kept = kept_dir(org).map(|d| next.keep_on_disk(&d, header.clone(), &sub));
        let loaded = next.len();
        let (mut embedded, mut preempted, mut rounds) = (0_usize, 0_usize, 0_usize);
        let done = loop {
            if FOLLOW_GEN.load(SeqCst) != epoch {
                // Superseded: its pending rows are written and its store let go for the next one.
                next.mark_stale("superseded by a newer change of the embedder's values", false);
                break None;
            }
            if !rebuild_idle() {
                std::thread::park_timeout(std::time::Duration::from_millis(250));
                continue;
            }
            let calls = CALLS_SEEN.load(SeqCst);
            let go = || CALLS_SEEN.load(SeqCst) == calls && crate::writer_idle::calls_in_flight() == 0 && FOLLOW_GEN.load(SeqCst) == epoch;
            let cur = org.sub();
            next.retain(&cur);
            let (have, total) = next.coverage(&cur);
            crate::live::progress_start("dense", have, total);
            if model.pop.gpu.is_some() {
                embedded += next.fill_batched(&cur, &batch_embed(Arc::clone(&model)), DENSE_BATCH, &go);
            }
            embedded += next.fill(&cur, &go);
            let old = cur.dense.read().ok().and_then(|d| d.clone());
            let have_aux: std::collections::BTreeSet<(String, String)> = next.aux_texts().into_iter().collect();
            for (k, text) in old.map(|o| o.aux_texts()).unwrap_or_default() {
                if !go() {
                    break;
                }
                if !have_aux.contains(&(k.clone(), text.clone())) {
                    let _ = next.add_aux_text(&k, &text);
                }
            }
            let (have, total) = next.coverage(&cur);
            crate::live::progress_end("dense", have, total);
            rounds += 1;
            if !go() {
                preempted += 1;
                continue;
            }
            match nmsubs::dense::switch(&cur.dense, &next, &cur) {
                Ok(_) => break Some((have, total)),
                Err(e) => {
                    // Pieces the engine failed on are tried again at the next idle round.
                    if rounds % 20 == 1 {
                        set_follow_line(&format!("dense lane: rebuild not switched in yet: {e}"));
                    }
                    std::thread::park_timeout(std::time::Duration::from_secs(1));
                }
            }
        };
        if let Some((have, total)) = done {
            set_follow_line(&format!(
                "dense lane: rebuilt for the embedder's new values and switched in, {have} of {total} pieces, in {:.1} s ({embedded} embedded, {loaded} loaded from the store{}; preempted by calls {preempted} times)",
                t.elapsed().as_secs_f64(),
                kept.map_or_else(|| ", vectors in RAM only".to_owned(), |k| format!(": {k}"))
            ));
        }
        if FOLLOW_GEN.load(SeqCst) == epoch {
            if let Ok(mut b) = BUILDING.lock() {
                *b = None;
            }
            FOLLOW_RUNNING.store(false, SeqCst);
            // A change seen between the check and the store would find the flag still set.
            if FOLLOW_GEN.load(SeqCst) == epoch || FOLLOW_RUNNING.swap(true, SeqCst) {
                break;
            }
        }
    }
}

/// What makes the embedder's vectors what they are, for the header of stored vectors: its weight
/// hash, its precision, the arithmetic switches in force, its pooling and input ceiling, and the
/// recipe of the embedded text. Stored vectors are used only under an equal header.
#[must_use]
pub fn dense_header(m: &Model) -> nmsubs::dense::store::Header {
    use std::sync::atomic::Ordering::Relaxed;
    let mut widths: Vec<&str> = m.asm.nodes.iter().filter(|n| n.op == "linear").map(|n| n.precision.as_str()).collect();
    widths.sort_unstable();
    widths.dedup();
    let keeps_f32 = m.no_half.load(Relaxed);
    // The integer path for 8-bit linears changes the vectors within rounding: named only when it
    // is in force, so stores made without it keep their header.
    let int8_dot = widths.contains(&"q8") && m.int8_dot.load(Relaxed) && nmengn::gpu::HALF_MATH.load(Relaxed) && nmengn::gpu::dp4a_served();
    let ceiling = nmengn::roles::REFERENCE_MAX_TOKENS.min(m.pop.card.int("max_position_embeddings").unwrap_or(nmengn::roles::REFERENCE_MAX_TOKENS));
    [
        ("model", m.pop.card.id.clone()),
        ("weights", m.pop.card.sha_main.clone()),
        ("parameters", parameters_key(m)),
        ("assembly", m.asm.name.clone()),
        ("precision", widths.join(",")),
        ("binary32_math", keeps_f32.to_string()),
        ("half_math", (nmengn::gpu::HALF_MATH.load(Relaxed) && !keeps_f32).to_string()),
        ("segment_attention", nmengn::gpu::SEGMENT_ATTENTION.load(Relaxed).to_string()),
        ("pooling", m.asm.nodes.last().map(|n| n.op.clone()).unwrap_or_default()),
        ("input_ceiling", ceiling.saturating_sub(1).to_string()),
        ("recipe", nmsubs::dense::recipe().to_owned()),
    ]
    .into_iter()
    .chain(int8_dot.then(|| ("int8_dot", "true".to_owned())))
    .chain(PARTS_NAMED.load(Relaxed).then(|| ("parts_within", attention_crossover(m).to_string())))
    .map(|(k, v)| (k.to_owned(), v))
    .collect()
}

/// The dense lane for a run outside a server (E3 runs, 2026-11-09): its own embedder (on the
/// GPU when `gpu` and one is present), the same store file `start_dense` uses for this tree,
/// and every missing chunk embedded first, so the lane covers the whole tree. Returns the lane
/// and a line saying what was held and embedded.
/// # Errors
/// The embedder cannot be opened.
pub fn open_lane(home: &Path, sub: &nmsubs::Substrate, gpu: bool) -> Result<(nmsubs::dense::DenseLane, String), Fault> {
    let pop = Population::open(home, &nmtype::role_models::model("embedder"), true, gpu)?;
    let dev = if gpu && pop.gpu.is_some() { nmengn::placement::Device::Gpu } else { nmengn::placement::Device::Cpu };
    let asm = nmengn::Assembly::load(&home.join("designs/assemblies"), "qwen3-embedding")?;
    let m = Arc::new(bind_prefetched(&pop, asm)?);
    // 2026-11-10 (refinement step 1, item 7): held in RAM only, every chunk embedded in this run.
    let (m1, m2) = (Arc::clone(&m), Arc::clone(&m));
    let lane = nmsubs::dense::DenseLane::in_ram(
        Arc::new(move |t: &str| nmengn::roles::card_embed_on(&m1, dev, t).ok()),
        Arc::new(move |t: &str| nmengn::roles::card_embed_on(&m2, dev, t).ok()),
    );
    let filled = lane.fill(sub, &|| true);
    Ok((lane, format!("lane: {} chunks, vectors in RAM only, {filled} embedded now ({})", sub.chunks.len(), dev.name())))
}

/// A released role model with its own assembly, opened once per process.
/// # Errors
/// The population or the assembly cannot be opened.
/// The embedder binds at 8 bits (`serve --embedder-q8`); set before the first bind.
pub static EMBED_Q8: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The reranker (the Core Brain's judge) binds at 8 bits (`serve --reranker-q8`, 2026-11-13; its
/// closeness check passed at 63% of the bytes, cartridge step F); set before the first bind.
pub static RERANK_Q8: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Precision per part (2026-11-13): the engine's setting, applied by `Model::bind` to every model.
pub use nmengn::model::{PRECISION, Precision};

/// The base cartridges to offer per MicroModel (`qwen3-reranker-0.6b`), in the order to try them,
/// set by the serving program before the models are opened. The first that seats is the
/// MicroModel's base source; when none seats, or none is listed, it runs on its open weights.
pub static BASE_CARTRIDGES: std::sync::RwLock<std::collections::BTreeMap<String, Vec<std::path::PathBuf>>> =
    std::sync::RwLock::new(std::collections::BTreeMap::new());

/// The catalogue's name (`id@version`) of each base cartridge file offered, where it differs from
/// the name its label carries (a cartridge renamed after it was written), for the seat lines.
pub static BASE_KEYS: std::sync::RwLock<std::collections::BTreeMap<std::path::PathBuf, String>> =
    std::sync::RwLock::new(std::collections::BTreeMap::new());

/// The base cartridges of each MicroModel left out before the start (a file missing, damaged or
/// made for other open weights), with their reasons, set by the serving program beside
/// `BASE_CARTRIDGES`, so a MicroModel left on its open weights says why.
pub static BASE_LEFT_OUT: std::sync::RwLock<std::collections::BTreeMap<String, Vec<String>>> =
    std::sync::RwLock::new(std::collections::BTreeMap::new());

/// What the serving program does with each seating result: (MicroModel, the cartridge seated or
/// none, the line, every refusal with its reason). Set by the serving program; none in other runs.
pub type SeatHook = fn(&str, Option<&str>, &str, &[String]);
pub static ON_SEAT: std::sync::RwLock<Option<SeatHook>> = std::sync::RwLock::new(None);

/// What seating did for each MicroModel in this process (seated, with its check; or refused and
/// why, with the open weights kept), for the audit and the reports.
static SEATS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// The seating lines of this process.
#[must_use]
pub fn seat_notes() -> Vec<String> {
    SEATS.lock().map(|v| v.clone()).unwrap_or_default()
}

/// The parameter values each seated MicroModel runs on, by population id: a cartridge whose
/// label says its values are the open weights' own and whose check came back bit-identical runs on
/// the open weights' values; any other seated cartridge on its own parameters.
static SEATED_VALUES: std::sync::RwLock<std::collections::BTreeMap<String, String>> = std::sync::RwLock::new(std::collections::BTreeMap::new());

/// What a model's outputs are made from, for keys of encoded reads kept on disk: `open weights
/// <sha>` (the open weights, or an exact base cartridge proven bit-identical to them at its seat), or
/// `cartridge <parameters sha>` (a base cartridge with values of its own).
#[must_use]
pub fn parameters_key(m: &Model) -> String {
    SEATED_VALUES
        .read()
        .ok()
        .filter(|_| m.pop.source != nmengn::model::OPEN_WEIGHTS)
        .and_then(|g| g.get(&m.pop.card.id).cloned())
        .unwrap_or_else(|| values_key(&m.pop.card.sha_main, None))
}

/// The key of the values a model runs on: the open weights' hash, or, for a seated cartridge
/// (`(parameters sha, exact and proven bit-identical)`), the open weights' hash when it is exact
/// and the cartridge's parameters hash when it is not.
#[must_use]
pub fn values_key(open_sha: &str, seated: Option<(&str, bool)>) -> String {
    match seated {
        Some((params, false)) => format!("cartridge {params}"),
        _ => format!("open weights {open_sha}"),
    }
}

/// A cartridge file as an offer to be a MicroModel's base source: its parameter sets (checked
/// against its label) with their places and addresses, and what its label says of it.
/// # Errors
/// A file that is not a labelled cartridge, or parameters that do not match the label.
pub fn offer_of(file: &Path) -> Result<nmengn::base::Offer, Fault> {
    let parts = nmlearn::library::tensor_parts_of(file).map_err(|e| bad(e.to_string()))?;
    let label = nmlearn::library::label_of(file).map_err(|e| bad(e.to_string()))?;
    Ok(offer_from(&label, parts))
}

/// The offer a label and its parameter sets make (`offer_of` without the file).
#[must_use]
pub fn offer_from(label: &Value, parts: Vec<nmlearn::library::TensorPart>) -> nmengn::base::Offer {
    let text = |k: &str| label.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default();
    let notes = label.get("notes").cloned().unwrap_or(Value::Null);
    let bound = match notes.get("check_bound") {
        Ok(Value::Number(t)) => t.parse::<f32>().ok(),
        _ => None,
    };
    let open: std::collections::BTreeMap<String, Option<nmengn::base::Recipe>> = parts
        .iter()
        .filter(|p| p.open)
        .map(|p| (p.name.clone(), p.keep.as_ref().map(|k| nmengn::base::Recipe { cols: k.cols, ranges: k.ranges.clone() })))
        .collect();
    nmengn::base::Offer {
        key: format!("{}@{}", text("id"), label.get("version").and_then(Value::integer).unwrap_or(0)),
        weights: text("weights"),
        parameters: text("parameters_sha256"),
        stated: parts.iter().map(|p| (p.name.clone(), (p.layer, p.block.clone()))).collect(),
        parts: parts
            .into_iter()
            .map(|p| nmnpcs::TensorDesc {
                name: p.name,
                dtype: p.dtype,
                shape: p.shape,
                file: p.file.display().to_string(),
                start: p.offset,
                end: p.offset + p.len,
                sha256: p.sha256,
            })
            .collect(),
        exact: matches!(notes.get("exact"), Ok(Value::Bool(true))),
        bound,
        open,
    }
}

/// The population of a MicroModel's open weights as a served model opens it: with the parameter
/// overlay of `open:<id>` applied at once when no base cartridge is offered to it (the open weights
/// will serve); when base cartridges are offered, unpatched, so each is checked against the open
/// weights as released, and `on_base` applies the overlay only if the open weights end up serving.
/// Opening a population reads no tensor, so the unpatched one costs its header alone.
fn open_population(home: &Path, id: &str, gpu: bool) -> Result<Arc<Population>, Fault> {
    let pop = Population::open(home, id, true, gpu)?;
    let offered = BASE_CARTRIDGES.read().ok().is_some_and(|g| g.get(id).is_some_and(|v| !v.is_empty()));
    // A lab run opens populations with their overlays already (`nmengn::overlay::AT_OPEN`).
    if offered || pop.source != nmengn::model::OPEN_WEIGHTS {
        return Ok(pop);
    }
    Ok(overlaid_population(&pop, &format!("open:{id}")).map_or(pop, |(p, _)| p))
}

/// `pop` with the parameter overlay of `key` applied (touched tensors patched in RAM, the files never
/// written), and the overlay's identity; `None` when it has no overlay with an effect, or the
/// overlay cannot be applied (said on standard error; the population serves without it). Records
/// what the outputs are made from, so vectors kept on disk are not mixed across the change.
fn overlaid_population(pop: &Arc<Population>, key: &str) -> Option<(Arc<Population>, String)> {
    let id = pop.card.id.clone();
    let p = match nmengn::overlay::patch(&pop.home, key, &pop.tensors, &pop.card.dir) {
        Ok(Some(p)) => p,
        Ok(None) => return None,
        Err(e) => {
            eprintln!("neural: {id}: parameter overlay of {key} not applied, served without it: {e}");
            return None;
        }
    };
    let open = pop.source == nmengn::model::OPEN_WEIGHTS;
    let source = if open { format!("{} with overlay {}", nmengn::model::OPEN_WEIGHTS, p.identity) } else { pop.source.clone() };
    match Population::open_parts(&pop.home, &id, pop.gpu.is_some(), source, p.descs) {
        Ok(np) => {
            nmengn::overlay::hold(&id, p.held);
            if open && let Ok(mut g) = SEATED_VALUES.write() {
                g.insert(id.clone(), format!("{} with overlay {}", values_key(&pop.card.sha_main, None), p.identity));
            }
            eprintln!("neural: {id}: parameter overlay of {key} ({}) applied to {} tensors", p.identity, p.tensors.len());
            Some((np, p.identity))
        }
        Err(e) => {
            eprintln!("neural: {id}: parameter overlay of {key} not applied, served without it: {e}");
            None
        }
    }
}

/// A bound model bound again over its population with the overlay of `key` applied, and the overlay's
/// identity; the model as it was when there is nothing to apply or the bind fails.
fn rebind_overlaid(m: Model, key: &str) -> (Model, Option<String>) {
    let Some((pop, ident)) = overlaid_population(&m.pop, key) else { return (m, None) };
    // The live view shows the rebind on the MicroModel it binds again, for as long as it takes.
    let mut live = crate::live::stage_as("rebind", "overlay.rebind", crate::live::role_of(&m.pop.card.id), "failed");
    let rows = m.bound.iter().any(|b| !b.rows.is_empty());
    match Model::bind(&pop, m.asm.clone()) {
        Ok(n) => {
            live.label("applied");
            drop(m);
            (if rows { n.label_rows() } else { n }, Some(ident))
        }
        Err(e) => {
            eprintln!("neural: {}: parameter overlay of {key} not applied, the bind failed: {e}", pop.card.id);
            (m, None)
        }
    }
}

/// The seat signature each held model was bound under, by its use (`reader`, `embedder`,
/// `reranker`): the overlay identities of its open weights and of every base cartridge offered to it.
static BOUND_SIGS: Mutex<std::collections::BTreeMap<&'static str, String>> = Mutex::new(std::collections::BTreeMap::new());

/// The identity of an object's overlay with an effect, empty when it has none.
fn overlay_ident(home: &Path, key: &str) -> String {
    nmengn::overlay::effective(home, key).map(|o| nmengn::overlay::identity(&o)).unwrap_or_default()
}

/// What decides a MicroModel's seat as far as overlays go: the overlay identity of its open weights
/// and of each base cartridge offered to it (a whole mask there changes which one seats).
fn seat_signature(home: &Path, id: &str) -> String {
    let mut parts = vec![format!("open:{id}={}", overlay_ident(home, &format!("open:{id}")))];
    for file in BASE_CARTRIDGES.read().ok().and_then(|g| g.get(id).cloned()).unwrap_or_default() {
        let key = BASE_KEYS.read().ok().and_then(|g| g.get(&file).cloned()).or_else(|| {
            let l = nmlearn::library::label_of(&file).ok()?;
            Some(format!("{}@{}", l.get("id").and_then(Value::string).unwrap_or(""), l.get("version").and_then(Value::integer).unwrap_or(0)))
        });
        if let Some(k) = key {
            parts.push(format!("{k}={}", overlay_ident(home, &k)));
        }
    }
    parts.join(";")
}

/// Records the seat signature a held model was bound under.
fn note_bound(name: &'static str, home: &Path, m: &Arc<Model>) {
    let sig = seat_signature(home, &m.pop.card.id);
    if let Ok(mut g) = BOUND_SIGS.lock() {
        g.insert(name, sig);
    }
}

/// The overlay stamp the last settle saw; also serializes settles.
static SETTLED: Mutex<u64> = Mutex::new(0);

/// Applies parameter overlay changes to this process's held models where a call starts (the API's
/// dispatch calls it before every operation). When the seat signature of a held model changed (its
/// open weights' overlay, or the overlay of a base cartridge offered to it), that MicroModel is
/// opened and bound again with the overlay applied, the slots that use it are bound to the new
/// model, and the old one is dropped once the calls running on it end. The bound model holds
/// packed and device copies of every tensor, made at bind for its precision, so a touched tensor
/// is not rewritten in place: the whole MicroModel is bound again, and the time it took is
/// recorded (`nmengn::overlay::record_reload`). Learning units loaded on the Reader take their
/// overlays live (`learn::settle_overlays`). When the embedder's served values changed, the dense
/// lane follows (`follow_embedder`): its vectors are marked stale and a lane for the new values is
/// built at idle and switched in. Returns one line per change.
pub fn settle_overlays(home: &Path, org: Option<&Organism>) -> Vec<String> {
    let st = nmengn::overlay::stamp_cached(home);
    let Ok(mut last) = SETTLED.lock() else { return Vec::new() };
    if *last == st {
        return Vec::new();
    }
    *last = st;
    let mut lines = Vec::new();
    let mut rebound = false;
    for (name, cell) in [("reader", &MODEL), ("embedder", &EMBED_MODEL), ("reranker", &RERANK_MODEL)] {
        let Some(m) = cell.get() else { continue };
        let id = m.pop.card.id.clone();
        let now = seat_signature(home, &id);
        if BOUND_SIGS.lock().ok().and_then(|g| g.get(name).cloned()).as_deref() == Some(now.as_str()) {
            continue;
        }
        // The card copies of the old model are released first (when no run holds them), so the
        // new model's uploads fit beside it on a small card.
        let _ = m.pop.try_release_device();
        drop(m);
        let t = std::time::Instant::now();
        let opened = match name {
            "reader" => open_reader(home),
            "embedder" => open_role(home, "qwen3-embedding", &id),
            _ => open_role(home, "qwen3-reranker", &id),
        };
        match opened {
            Ok(n) => {
                cell.replace(Arc::clone(&n));
                // Weight placement (2026-11-27): the new model's weights go to the card in the
                // background, as at start, and its host copies of the linear maps are dropped
                // once there (`preload::stage`); a call before that uploads what it uses.
                let staged = Arc::clone(&n);
                std::thread::spawn(move || {
                    if let Err(e) = crate::preload::stage(&staged) {
                        eprintln!("neural: {}: not staged on the card after the overlay rebind ({e}); it uploads at its first run", staged.pop.card.id);
                    }
                });
                note_bound(name, home, &n);
                if name == "reader" {
                    seat_reader(home, &n);
                    lines.extend(crate::learn::apply_units(&n, home, true));
                } else {
                    seat_role(home, &id, &n);
                }
                let secs = t.elapsed().as_secs_f64();
                nmengn::overlay::record_reload(home, &id, secs);
                rebound = true;
                lines.push(format!("{id}: bound again with its parameter overlays at the call boundary in {secs:.2} s ({})", n.pop.source));
            }
            Err(e) => lines.push(format!("{id}: bound again with its parameter overlays refused ({e}); the model bound before serves")),
        }
    }
    if rebound && let Some(o) = org {
        let bound = |s: &str| o.slots.by_name.get(s).is_some_and(|x| x.engine_bound());
        let r = if bound("reranker") { bind(o).map(|_| ()) } else if bound("embedder") { bind_embedder(o).map(|_| ()) } else { Ok(()) };
        if let Err(e) = r {
            lines.push(format!("slots not bound again: {e}"));
        }
        // After the slots: until they are bound again, questions still encode on the old values.
        if let Some(l) = follow_embedder(o) {
            lines.push(l);
        }
    }
    if let Some(m) = MODEL.get() {
        lines.extend(crate::learn::apply_units(&m, home, false));
    }
    for l in &lines {
        eprintln!("neural: {l}");
    }
    lines
}

/// `settle_overlays` for the organism this process serves, if any.
pub fn settle_installed() {
    // Every call's start passes here: the dense lane's rebuild gives way to it (`rebuild_idle`).
    CALLS_SEEN.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    LAST_CALL_MS.store(nmtype::now_ms(), std::sync::atomic::Ordering::SeqCst);
    if let Some(o) = crate::tools::installed() {
        let _ = settle_overlays(&o.home, Some(o));
    }
}

/// True when at least one upstream `.safetensors` file is present in `weights/<id>/files/`.
fn open_weights_present(home: &Path, id: &str) -> bool {
    nmnpcs::WeightCard::load(home, id).ok().is_some_and(|c| !c.shards().is_empty())
}

/// Reads `model.embed_tokens.weight` directly from an offered standalone base cartridge for `id`,
/// so drone intake (`nmpod::intake::Bf16Table`) works without upstream `.safetensors` on disk.
/// # Errors
/// No offered standalone base cartridge contains a BF16 `model.embed_tokens.weight` tensor.
pub fn embed_tokens_from_base(id: &str) -> Result<Arc<nmpod::intake::Bf16Table>, Fault> {
    let files = BASE_CARTRIDGES.read().ok().and_then(|g| g.get(id).cloned()).unwrap_or_default();
    let mut last_err = format!("{id}: no base cartridge offered for standalone token table");
    for file in &files {
        let offer = match offer_of(file) {
            Ok(o) => o,
            Err(e) => {
                last_err = e.to_string();
                continue;
            }
        };
        if let Some(t) = offer.parts.iter().find(|p| (p.name == "model.embed_tokens.weight" || p.name == "embed_tokens.weight") && p.dtype == "BF16" && p.shape.len() == 2) {
            let rows = t.shape.first().copied().unwrap_or(0);
            let width = t.shape.get(1).copied().unwrap_or(0);
            return nmpod::intake::Bf16Table::open_range(Path::new(&t.file), t.start, rows, width);
        }
    }
    Err(bad(last_err))
}

/// Seats a self-contained base cartridge directly when the optional upstream open-weight
/// `.safetensors` are not installed on disk. Upstream open weights remain optional for fallback,
/// reference parity checks, parameter mapping, and surgery.
fn open_standalone_base(home: &Path, id: &str, asm: &nmengn::Assembly, gpu: bool, label_rows: bool) -> Result<Model, Fault> {
    let files = BASE_CARTRIDGES.read().ok().and_then(|g| g.get(id).cloned()).unwrap_or_default();
    let mut refused: Vec<String> = BASE_LEFT_OUT.read().ok().and_then(|g| g.get(id).cloned()).unwrap_or_default();
    let hook = ON_SEAT.read().ok().and_then(|h| *h);
    let mut seated = None;
    let mut seated_file = None;
    for file in &files {
        let name = file.file_name().map_or_else(String::new, |f| f.to_string_lossy().into_owned());
        let named = |mut o: nmengn::base::Offer| {
            if let Some(k) = BASE_KEYS.read().ok().and_then(|g| g.get(file).cloned()) {
                o.key = k;
            }
            o
        };
        let unmasked = |o: nmengn::base::Offer| {
            if nmengn::overlay::effective(home, &o.key).is_some_and(|x| x.whole_masked()) {
                Err(format!("{}: masked whole in its parameter overlay; it is left out while it stays offered", o.key))
            } else {
                Ok(o)
            }
        };
        match offer_of(file)
            .map(named)
            .map_err(|e| e.to_string())
            .and_then(unmasked)
            .and_then(|o| nmengn::base::seat_standalone(home, id, asm.clone(), gpu, label_rows, &o).map(|r| (o, r)).map_err(|e| e.to_string()))
        {
            Ok(x) => {
                seated = Some(x);
                seated_file = Some(file.clone());
                break;
            }
            Err(e) => refused.push(format!("{name}: {e}")),
        }
    }
    let Some((o, (cart, s))) = seated else {
        let why = if refused.is_empty() {
            "no base cartridge offered".to_owned()
        } else {
            format!("refused: {}", refused.join("; "))
        };
        let line = format!("{id}: no base cartridge seated and optional upstream open weights are not installed ({why}); download the base cartridges (or optional upstream open weights for fallback)");
        eprintln!("neural: {line}");
        if let Ok(mut v) = SEATS.lock() {
            v.push(line.clone());
        }
        if let Some(hook) = hook {
            hook(id, None, &line, &refused);
        }
        return Err(bad(line));
    };
    let (cart, overlay) = if nmengn::overlay::AT_OPEN.load(std::sync::atomic::Ordering::Relaxed) { (cart, None) } else { rebind_overlaid(cart, &o.key) };
    let mut comp_sha = None;
    let comp = match seated_file.as_deref().map(|f| compensation_of(home, f, &o.key, &cart)) {
        Some(Some(Ok((key, sha, plugged)))) if comp_on(home, id) => {
            cart.set_filters(Some(Arc::new(plugged)));
            comp_sha = Some(sha);
            format!("; compensation {key} plugged on it")
        }
        Some(Some(Ok((key, _, _)))) => format!("; compensation {key} named, not plugged (NANO_BASE_COMP=on, or settings.json base_compensation, plugs it)"),
        Some(Some(Err(e))) => format!("; compensation refused, the cartridge serves without it ({e})"),
        _ => String::new(),
    };
    if let Ok(mut g) = SEATED_VALUES.write() {
        let k = values_key(&cart.pop.card.sha_main, Some((&o.parameters, o.exact)));
        let k = overlay.as_ref().map_or_else(|| k.clone(), |x| format!("{k} with overlay {x}"));
        g.insert(id.to_owned(), comp_sha.map_or_else(|| k.clone(), |c| format!("{k} with compensation {c}")));
    }
    let comp = match &overlay {
        Some(x) => format!("{comp}; parameter overlay {x} applied"),
        None => comp,
    };
    let line = format!(
        "{id}: base cartridge {} seated standalone in {:.2} s (upstream open weights not installed; optional for fallback, reference, parameter mapping and surgery){comp}",
        o.key, s.secs
    );
    let full = if refused.is_empty() { line } else { format!("{line} (refused: {})", refused.join("; ")) };
    eprintln!("neural: {full}");
    if let Ok(mut v) = SEATS.lock() {
        v.push(full.clone());
    }
    if let Some(hook) = hook {
        hook(id, Some(&o.key), &full, &refused);
    }
    Ok(cart)
}

/// The job's model on its base source: the cartridges offered for this MicroModel are tried in
/// order, each seated over `open` (complete, connected, checked against the open weights); the first
/// that seats becomes the source and `open` is dropped, releasing the open weights from RAM and the
/// card. When every one is refused, or none is offered, `open` serves. The line and each refusal's
/// reason are kept in `seat_notes` and handed to `ON_SEAT`.
fn on_base(open: Model) -> Model {
    let id = open.pop.card.id.clone();
    let files = BASE_CARTRIDGES.read().ok().and_then(|g| g.get(&id).cloned()).unwrap_or_default();
    let mut refused: Vec<String> = BASE_LEFT_OUT.read().ok().and_then(|g| g.get(&id).cloned()).unwrap_or_default();
    let hook = ON_SEAT.read().ok().and_then(|h| *h);
    // Nothing offered: quiet in runs that do not serve and have nothing left out; otherwise the open
    // weights serve and the line says why.
    if files.is_empty() && refused.is_empty() && hook.is_none() {
        return open;
    }
    let mut seated = None;
    let mut seated_file = None;
    for file in &files {
        let name = file.file_name().map_or_else(String::new, |f| f.to_string_lossy().into_owned());
        let named = |mut o: nmengn::base::Offer| {
            if let Some(k) = BASE_KEYS.read().ok().and_then(|g| g.get(file).cloned()) {
                o.key = k;
            }
            o
        };
        // A base cartridge masked whole in its parameter overlay is left out while it stays offered:
        // its effect over the open weights is off, so the next one (or the open weights) serves.
        let unmasked = |o: nmengn::base::Offer| {
            if nmengn::overlay::effective(&open.pop.home, &o.key).is_some_and(|x| x.whole_masked()) {
                Err(format!("{}: masked whole in its parameter overlay; it is left out while it stays offered", o.key))
            } else {
                Ok(o)
            }
        };
        match offer_of(file).map(named).map_err(|e| e.to_string()).and_then(unmasked).and_then(|o| nmengn::base::seat(&open, &o).map(|r| (o, r)).map_err(|e| e.to_string())) {
            Ok(x) => {
                seated = Some(x);
                seated_file = Some(file.clone());
                break;
            }
            Err(e) => refused.push(format!("{name}: {e}")),
        }
    }
    let (line, key, m) = match seated {
        Some((o, (cart, s))) => {
            // The cartridge's parameter overlay, applied over the cartridge that passed its check
            // against the open weights (the check compares the cartridge as written).
            // (A lab run opened the cartridge with its overlay already, `nmengn::overlay::AT_OPEN`.)
            let (cart, overlay) = if nmengn::overlay::AT_OPEN.load(std::sync::atomic::Ordering::Relaxed) { (cart, None) } else { rebind_overlaid(cart, &o.key) };
            // The compensation file the seated cartridge names, plugged on it alone (its own
            // filters), only with `NANO_BASE_COMP=on` (the lead's decision 2026-11-19: the served
            // measures held without it, so it is a stackable piece, not the default).
            let mut comp_sha = None;
            let comp = match seated_file.as_deref().map(|f| compensation_of(&open.pop.home, f, &o.key, &cart)) {
                Some(Some(Ok((key, sha, plugged)))) if comp_on(&open.pop.home, &id) => {
                    cart.set_filters(Some(Arc::new(plugged)));
                    comp_sha = Some(sha);
                    format!("; compensation {key} plugged on it")
                }
                Some(Some(Ok((key, _, _)))) => format!("; compensation {key} named, not plugged (NANO_BASE_COMP=on, or settings.json base_compensation, plugs it)"),
                Some(Some(Err(e))) => format!("; compensation refused, the cartridge serves without it ({e})"),
                _ => String::new(),
            };
            if let Ok(mut g) = SEATED_VALUES.write() {
                // Vectors kept on disk are keyed by what made them: the compensation is part of it.
                let k = values_key(&open.pop.card.sha_main, Some((&o.parameters, o.exact && s.identical == s.outputs)));
                let k = overlay.as_ref().map_or_else(|| k.clone(), |x| format!("{k} with overlay {x}"));
                g.insert(id.clone(), comp_sha.map_or_else(|| k.clone(), |c| format!("{k} with compensation {c}")));
            }
            let comp = match &overlay {
                Some(x) => format!("{comp}; parameter overlay {x} applied"),
                None => comp,
            };
            drop(open);
            let left = nmengn::residency::get().parameters().iter().filter(|r| r.model == id && r.source == nmengn::model::OPEN_WEIGHTS && !r.tensor.starts_with('(')).count();
            let line = format!(
                "{id}: base cartridge {} seated in {:.2} s; check {} of {} outputs bit-identical to the open weights, largest change {:e}; open weights released ({left} of their tensors resident){comp}",
                o.key, s.secs, s.identical, s.outputs, s.largest
            );
            (line, Some(o.key), cart)
        }
        None => {
            // The open weights serve: their parameter overlay applies now, unless the population was
            // opened with it already (no base cartridge was offered).
            let (open, overlay) = if open.pop.source == nmengn::model::OPEN_WEIGHTS { rebind_overlaid(open, &format!("open:{id}")) } else { (open, None) };
            let ov = overlay.map_or_else(String::new, |x| format!("; parameter overlay {x} applied"));
            if files.is_empty() && refused.is_empty() {
                (format!("{id}: no base cartridge offered to it; the open weights serve{ov}"), None, open)
            } else {
                (format!("{id}: no base cartridge seated; the open weights stay{ov}"), None, open)
            }
        }
    };
    let full = if refused.is_empty() { line } else { format!("{line} (refused: {})", refused.join("; ")) };
    eprintln!("neural: {full}");
    if let Ok(mut v) = SEATS.lock() {
        v.push(full.clone());
    }
    if let Some(hook) = hook {
        hook(&id, key.as_deref(), &full, &refused);
    }
    m
}

/// Whether a seated base cartridge of `model` serves with the compensation file it names:
/// `NANO_BASE_COMP=on` or `=off` for every MicroModel when set; `NANO_BASE_COMP_ON=<models>` on for
/// those named; else `settings.json`
/// `base_compensation` (`{"<model>": "on"}`, the default per MicroModel); else off.
fn comp_on(home: &Path, model: &str) -> bool {
    match std::env::var("NANO_BASE_COMP").as_deref() {
        Ok("on") => return true,
        Ok("off") => return false,
        _ => {}
    }
    // `NANO_BASE_COMP_ON=<model>[,<model>...]`: on for the MicroModels named (a measurement arm).
    if std::env::var("NANO_BASE_COMP_ON").is_ok_and(|v| v.split(',').any(|m| m == model)) {
        return true;
    }
    std::fs::read(home.join("settings.json"))
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .and_then(|v| v.get("base_compensation").and_then(|c| c.get(model)).and_then(Value::string).ok().map(|s| s == "on"))
        .unwrap_or(false)
}

/// The compensation a base cartridge's label names (`notes.compensation`, a key on the library's
/// shelf): its own small file of gain-bias parts (gain 1, a bias per channel at nodes of the
/// MicroModel's assembly) that restores the mean contribution of what the base cartridge left out.
/// Built as the seated model's own filters (never the rack's, so no other run sees it). `None`
/// when the label names none; refused (the cartridge then serves without it) when the file is not
/// on the shelf, was made for another cartridge or other open weights, or a part does not fit a
/// node. Returns its key, its file's SHA-256 and the filters.
pub fn compensation_of(home: &Path, file: &Path, base_key: &str, cart: &Model) -> Option<Result<(String, String, nmengn::filter::Plugged), String>> {
    let label = nmlearn::library::label_of(file).ok()?;
    let key = label.get("notes").and_then(|n| n.get("compensation")).and_then(Value::string).ok()?.to_owned();
    Some((|| {
        let lib = nmlearn::library::Library::open(home).map_err(|e| e.to_string())?;
        let c = lib.cartridge(&key).ok_or_else(|| format!("{key} is not in the cartridge library"))?;
        if c.model != cart.pop.card.id || c.weights != cart.pop.card.sha_main {
            return Err(format!("{key} was made for {} over open weights {:.12}", c.model, c.weights));
        }
        let made_for = c.notes.get("for").and_then(Value::string).map(str::to_owned).unwrap_or_default();
        if made_for != base_key {
            return Err(format!("{key} was made for {made_for}, the seated base cartridge is {base_key}"));
        }
        let points = match lib.filter(c).map_err(|e| e.to_string())? {
            nmlearn::library::Filter::Points { points, .. } => points,
            _ => return Err(format!("{key} holds no gain-bias parts")),
        };
        let width = |t: &str| -> Option<usize> {
            // A point at a feed-forward block's output (`L<n>.down`) is the stream's width.
            let n = cart.asm.nodes.iter().find(|n| n.id.rsplit('#').next() == Some(t))?;
            (n.op == "linear" && t.ends_with(".down")).then_some(cart.pop.dims.hidden)
        };
        let mut parts = Vec::with_capacity(points.len());
        for p in points {
            if width(&p.target) != Some(p.gain.len()) {
                return Err(format!("{key}: part {} of width {} fits no feed-forward output of the model", p.target, p.gain.len()));
            }
            parts.push(nmengn::filter::Part { target: p.target, gain: p.gain, bias: p.bias });
        }
        let sha = c.files.first().map(|f| f.sha256.clone()).unwrap_or_default();
        let plugged = nmengn::filter::build(&cart.pop.card.id, &[nmengn::filter::CartridgeFilter { cartridge: key.clone(), parts }])?;
        Ok((key.clone(), sha, plugged))
    })())
}

/// The precision a served MicroModel binds at, over its assembly's linears: the selector's width
/// when one is selected (applied by `Model::bind`), else 8 bits for a short form, else the built-in
/// default (`nmengn::model::DEFAULT_BITS`, 16).
pub fn served_width(asm: &mut nmengn::Assembly, pop_id: &str, short_q8: bool) -> u8 {
    if let Some(b) = PRECISION.read().ok().and_then(|p| p.of(pop_id)) {
        return b;
    }
    let bits = if short_q8 { 8 } else { nmengn::model::DEFAULT_BITS };
    let form = match bits {
        4 => "q4",
        8 => "q8",
        _ => "f16",
    };
    for n in asm.nodes.iter_mut().filter(|n| n.op == "linear") {
        form.clone_into(&mut n.precision);
    }
    bits
}

fn role_model(home: &Path, cell: &Cell, asm_name: &str, pop_id: &str) -> Result<Arc<Model>, Fault> {
    if let Some(m) = cell.get() {
        return Ok(m);
    }
    let _opening = cell.opening();
    if let Some(m) = cell.get() {
        return Ok(m);
    }
    let m = cell.get_or_init(open_role(home, asm_name, pop_id)?);
    note_bound(if asm_name == "qwen3-embedding" { "embedder" } else { "reranker" }, home, &m);
    seat_role(home, pop_id, &m);
    Ok(m)
}

/// A role model opened and bound on its base source (with its parameter overlay), not yet held.
fn open_role(home: &Path, asm_name: &str, pop_id: &str) -> Result<Arc<Model>, Fault> {
    let mut asm = nmengn::Assembly::load(&home.join("designs/assemblies"), asm_name)?;
    // The embedder or the judge at 8 bits when asked by a short form (`serve --embedder-q8`,
    // `--reranker-q8`); `--precision` overrides the short forms.
    let role = nmtype::role_models::role_of(pop_id);
    let short = (role == Some("embedder") && EMBED_Q8.load(std::sync::atomic::Ordering::Relaxed))
        || (role == Some("reranker") && RERANK_Q8.load(std::sync::atomic::Ordering::Relaxed));
    served_width(&mut asm, pop_id, short);
    // Parameter loader step 4 (2026-11-10): a role model whose output head feeds only a label
    // read (the reranker) loads the label rows of the head, not the whole vocabulary. Exact:
    // P(yes) bit-identical on 72 of 72 pairs with 296 MB less on the device
    // (`teardown_roles::p7_label_rows`, with the host token lookup). A model without such a head
    // is unchanged by this.
    // The model alone holds its population, so seating a base cartridge releases the open weights.
    if !open_weights_present(home, pop_id) {
        return Ok(Arc::new(open_standalone_base(home, pop_id, &asm, true, true)?));
    }
    let pop = open_population(home, pop_id, true)?;
    let open = bind_prefetched(&pop, asm)?.label_rows();
    let m = Arc::new(on_base(open));
    Ok(m)
}

/// What a role model carries once held: its distilled cartridge or lens and the request read.
fn seat_role(home: &Path, pop_id: &str, m: &Arc<Model>) {
    let role = nmtype::role_models::role_of(pop_id);
    // The distilled cartridge plugged in for the Reranker, on its final norm, or the distilled
    // request lens for the embedder (`distil`).
    if role == Some("embedder") {
        let _ = crate::distil::seat_embedder(home, m);
    }
    if role == Some("reranker") {
        let line = crate::distil::seat(home, m, "reranker");
        if !line.is_empty() {
            eprintln!("neural: reranker: {line}");
        }
    }
    // Lane R: the request's kind read by the embedder when it is on (`NANO_REQ_KIND=embed`, off).
    if role == Some("embedder") {
        let _ = crate::request_read::install_embedder_with(m);
    }
}

/// `nm_weights`: the population read only (tensors, shapes, precision, clusters, groups,
/// where each cluster lives).
/// # Errors
/// The population cannot be opened.
pub fn weights(home: &Path, limit: usize) -> Result<Value, Fault> {
    let (pop, live) = match MODEL.get() {
        Some(m) => (Arc::clone(&m.pop), true),
        None => (Population::open(home, POPULATION, false, false)?, false),
    };
    let mut v = nmnpcs::inspect::inspect(&pop.card, &pop.tensors, &pop.clusters, if live { Some(pop.prov.as_ref()) } else { None }, limit);
    if let Value::Object(m) = &mut v {
        m.insert("bound".into(), Value::Bool(live));
        m.insert("unused_populations".into(), Value::Array(Vec::new()));
    }
    Ok(v)
}

/// `nm_weights` for any installed MicroModel (2026-11-26, the interface's `/api/models/weights`):
/// the bound population when `model` is the one this process runs, else the model's weight card
/// and tensor header opened read only (no hash pass, no graphics card; nothing is written). Each
/// tensor with its shape and element type, the clusters and, for a bound model, what is resident.
/// # Errors
/// A model id that is not a plain folder name, or a missing or corrupt population.
pub fn weights_of(home: &Path, model: &str, limit: usize) -> Result<Value, Fault> {
    if model.is_empty() || model.starts_with('.') || !model.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')) {
        return Err(Fault::new("nmapi.neural", FaultKind::Config, format!("{model}: not a model id")));
    }
    let (pop, live) = match MODEL.get().filter(|m| m.pop.card.id == model) {
        Some(m) => (Arc::clone(&m.pop), true),
        None => (Population::open(home, model, false, false)?, false),
    };
    let mut v = nmnpcs::inspect::inspect(&pop.card, &pop.tensors, &pop.clusters, if live { Some(pop.prov.as_ref()) } else { None }, limit);
    if let Value::Object(m) = &mut v {
        m.insert("model".into(), Value::String(model.to_owned()));
        m.insert("bound".into(), Value::Bool(live));
        // The tensors by name (the first `limit`), each with its element type, shape and bytes.
        let mut names: Vec<&nmnpcs::decompose::TensorDesc> = pop.tensors.iter().collect();
        names.sort_by(|a, b| a.name.cmp(&b.name));
        let num = |x: u64| Value::Number(x.to_string());
        let list = names
            .iter()
            .take(limit)
            .map(|t| {
                Value::Object(
                    [
                        ("name".to_owned(), Value::String(t.name.clone())),
                        ("dtype".to_owned(), Value::String(t.dtype.clone())),
                        ("shape".to_owned(), Value::Array(t.shape.iter().map(|d| num(*d as u64)).collect())),
                        ("bytes".to_owned(), num(t.end.saturating_sub(t.start))),
                        ("file".to_owned(), Value::String(t.file.clone())),
                    ]
                    .into_iter()
                    .collect(),
                )
            })
            .collect();
        m.insert("tensor_list".into(), Value::Array(list));
        m.insert("limit".into(), num(limit as u64));
    }
    Ok(v)
}

/// The content-free questions the judge's label score is calibrated against at G2's edge (the
/// tentacle core design: "N/A", an empty question and a placeholder).
pub const CONTENT_FREE: [&str; 3] = ["N/A", "", "[MASK]"];

/// Each piece's content-free label (mean log-odds over `CONTENT_FREE`), by the values the judge
/// runs on (`parameters_key`) and the piece's content key: it depends on neither the request nor
/// the window the request picks, so it is read once per piece while the material and the judge are
/// unchanged (once per material version), and held in RAM.
static EDGE_NULL: Mutex<std::collections::BTreeMap<(String, String), f32>> = Mutex::new(std::collections::BTreeMap::new());

/// One contested piece for `judge_edge`: its content key (`nmsubs::Chunk::key`), the text the judge
/// reads for the request, and the request-free text its content-free label is read on (the piece
/// as the judge would read it with no question terms).
pub struct EdgePiece {
    pub key: String,
    pub posed: String,
    pub plain: String,
}

/// What one `judge_edge` call did.
#[derive(Clone, Copy, Debug, Default)]
pub struct EdgeRead {
    /// Pairs read in the one packed call (posed, plus three per piece seen first).
    pub pairs: usize,
    /// Pieces whose content-free label came from RAM.
    pub held: usize,
    /// Prompt tokens of the pairs read.
    pub tokens: usize,
}

fn edge_logit(p: f32) -> f32 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

/// Pieces whose content-free label is not yet held for the judge `m`.
#[must_use]
pub fn edge_unheld(m: &Model, keys: &[&str]) -> usize {
    let values = parameters_key(m);
    EDGE_NULL.lock().map_or(keys.len(), |h| keys.iter().filter(|k| !h.contains_key(&(values.clone(), (**k).to_owned()))).count())
}

/// The judge at G2's inclusion edge in one packed read (master design 3.8 rule 6, the judge's
/// role): every contested piece's posed pair and, for each piece seen first, its three
/// content-free pairs go into one `card_rerank_pairs` call on the card. Returns each piece's label
/// log-odds minus its content-free mean (the caller then subtracts the mean over the pieces judged
/// together, as the design writes), and what was read.
/// # Errors
/// Tokenizer or engine failure.
pub fn judge_edge(m: &Model, question: &str, pieces: &[EdgePiece]) -> Result<(Vec<f32>, EdgeRead), Fault> {
    let values = parameters_key(m);
    let held: Vec<Option<f32>> = {
        let h = EDGE_NULL.lock().map_err(|_| bad("content-free labels: lock poisoned"))?;
        pieces.iter().map(|p| h.get(&(values.clone(), p.key.clone())).copied()).collect()
    };
    let mut pairs: Vec<(&str, &str)> = pieces.iter().map(|p| (question, p.posed.as_str())).collect();
    for (p, h) in pieces.iter().zip(&held) {
        if h.is_none() {
            pairs.extend(CONTENT_FREE.iter().map(|f| (*f, p.plain.as_str())));
        }
    }
    let dev = if m.pop.gpu.is_some() { nmengn::placement::Device::Gpu } else { nmengn::placement::Device::Cpu };
    let tokens = pairs
        .iter()
        .map(|(q, d)| m.pop.tokenizer.encode(&nmengn::roles::card_prompt(nmengn::roles::CODE_INSTRUCTION, q, d)).map_or(0, |v| v.len()))
        .sum();
    let p = nmengn::roles::card_rerank_pairs(m, dev, nmengn::roles::CODE_INSTRUCTION, &pairs)?;
    let mut fresh = p.get(pieces.len()..).unwrap_or(&[]).chunks(CONTENT_FREE.len());
    let mut out = Vec::with_capacity(pieces.len());
    let mut h = EDGE_NULL.lock().map_err(|_| bad("content-free labels: lock poisoned"))?;
    for ((piece, posed), was) in pieces.iter().zip(&p).zip(&held) {
        let null = match was {
            Some(n) => *n,
            None => {
                #[expect(clippy::cast_precision_loss, reason = "a count of forms")]
                let n = fresh.next().unwrap_or(&[]).iter().map(|x| edge_logit(*x)).sum::<f32>() / CONTENT_FREE.len() as f32;
                h.insert((values.clone(), piece.key.clone()), n);
                n
            }
        };
        out.push(edge_logit(*posed) - null);
    }
    Ok((out, EdgeRead { pairs: pairs.len(), held: held.iter().filter(|x| x.is_some()).count(), tokens }))
}

/// Tokens of a candidate the close-call arm's judge reads (step J, C119: signature, doc comment,
/// first lines; A3 T2 cut at about 128 tokens).
pub const CLOSE_CUT_TOKENS: usize = 128;
/// The judge's log-odds are clamped where A3 T2 measured them (P(yes) clamped to 1e-6 and 1 - 1e-6).
const CLOSE_Z_CAP: f32 = 13.815_51;

/// A candidate as the close-call arm's judge reads it: its place line (`path:first-last`), then
/// its own text, the whole cut at `CLOSE_CUT_TOKENS` tokens of the judge's tokenizer with the place
/// line counted in (lane G package 3, 2026-11-18: the cut is at the token, not after the line that
/// passes it, which overshot to about 150 to 170 tokens and cost card time, T-079). The kept tokens are decoded back to their bytes;
/// a character split by the cut is dropped.
#[must_use]
pub fn close_doc(m: &Model, place: &str, text: &str) -> String {
    let head = format!("{place}\n");
    // The place line counts in the cut (at least half the cut is left for the piece's own text).
    let room = CLOSE_CUT_TOKENS.saturating_sub(m.pop.tokenizer.encode(&head).map_or(0, |v| v.len())).max(CLOSE_CUT_TOKENS / 2);
    let body: String = text.lines().flat_map(|l| [l, "\n"]).collect();
    let out = match m.pop.tokenizer.encode(&body) {
        Ok(ids) if ids.len() > room => {
            let cut = m.pop.tokenizer.decode(ids.get(..room).unwrap_or(&ids));
            cut.trim_end_matches('\u{FFFD}').to_owned()
        }
        _ => body,
    };
    format!("{head}{out}")
}

/// The close-call arm's judge (step J, C119): every candidate of one question in one packed read
/// on the card, the shared prompt head (instruction and question) prefilled once and the
/// documents packed side by side (`nmengn::roles::card_rerank_batch_z`, lane S's packed path);
/// off the card one at a time. Returns each candidate's label log-odds, clamped as measured.
/// # Errors
/// Tokenizer or engine failure.
pub fn judge_close(m: &Model, question: &str, docs: &[String]) -> Result<Vec<f32>, Fault> {
    // Roadmap Phase 4.5 (2026-11-24): the close-call judge in the live view.
    let mut live = crate::live::stage("G2.close", crate::live::role_of(&m.pop.card.id));
    live.count("candidates", docs.len());
    // The code read-out's states come from the judge's own packed run when it applies (lane T1
    // 2026-11-19, speed: a second unpacked forward per document before).
    let mut states: Vec<Option<Vec<f32>>> = Vec::new();
    let z = if m.pop.gpu.is_some() {
        if crate::adapter::judge_applies(m, docs) && !std::env::var("NANO_JUDGE_STATES").is_ok_and(|v| v == "off") {
            let (z, s) = nmengn::roles::card_rerank_batch_z_states(m, nmengn::roles::CODE_INSTRUCTION, question, docs, None)?;
            states = s;
            z
        } else {
            nmengn::roles::card_rerank_batch_z(m, nmengn::roles::CODE_INSTRUCTION, question, docs, None)?
        }
    } else {
        nmengn::roles::card_rerank_batch(m, nmengn::placement::Device::Cpu, nmengn::roles::CODE_INSTRUCTION, question, docs, None)?.into_iter().map(edge_logit).collect()
    };
    let z = z.into_iter().map(|x| if x.is_finite() { x.clamp(-CLOSE_Z_CAP, CLOSE_Z_CAP) } else { 0.0 }).collect();
    // Lane B (2026-11-19): the judge's code read-out in a close call of code pieces only, when it is
    // in force (`NANO_ADAPTER_JUDGE=on`); otherwise `z` untouched.
    Ok(crate::adapter::judge_code_states(m, question, docs, z, &states))
}

/// Seats `m` as this process's reranker before anything binds one (a measurement arm: the judge
/// run from a cartridge in place of the bare weights). False when a reranker is already bound.
pub fn seat_reranker(m: Arc<Model>) -> bool {
    RERANK_MODEL.set(m)
}

fn active_card_reranker(org: &Organism) -> Option<Arc<Model>> {
    let eng = org.slots.by_name.get("reranker")?.engine()?;
    eng.label().starts_with("rerank qwen3-reranker").then(|| RERANK_MODEL.get()).flatten()
}

fn active_card_embedder(org: &Organism) -> Option<Arc<Model>> {
    let eng = org.slots.by_name.get("embedder")?.engine()?;
    eng.label().starts_with("embed qwen3-embedding").then(|| EMBED_MODEL.get()).flatten()
}

/// Header describing what produced persisted episode query vectors under `<mem>/meaning/vectors.bin`.
#[must_use]
pub fn meaning_embed_header(org: &Organism) -> Option<nmsubs::dense::store::Header> {
    let eng = org.slots.by_name.get("embedder")?.engine()?;
    let task_sha = nmhash::sha256_bytes(crate::session::MEANING_TASK.as_bytes());
    if let Some(m) = active_card_embedder(org) {
        let mut h = dense_header(&m);
        h.remove("recipe");
        h.remove("parts_within");
        h.insert("kind".into(), "episode_queries".into());
        h.insert("task_sha256".into(), task_sha);
        return Some(h);
    }
    Some(
        [("kind", "episode_queries".to_owned()), ("engine", eng.label()), ("task_sha256", task_sha)]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect(),
    )
}

/// Content-free key identifying the active reranker model, weights, precision, math switches, and device.
#[must_use]
pub fn reranker_values_key(org: &Organism) -> Option<String> {
    use std::sync::atomic::Ordering::Relaxed;
    let eng = org.slots.by_name.get("reranker")?.engine()?;
    if let Some(m) = active_card_reranker(org) {
        let mut widths: Vec<&str> = m.asm.nodes.iter().filter(|n| n.op == "linear").map(|n| n.precision.as_str()).collect();
        widths.sort_unstable();
        widths.dedup();
        let keeps_f32 = m.no_half.load(Relaxed);
        let int8_dot = widths.contains(&"q8") && m.int8_dot.load(Relaxed) && nmengn::gpu::HALF_MATH.load(Relaxed) && nmengn::gpu::dp4a_served();
        let fields: std::collections::BTreeMap<&str, String> = [
            ("model", m.pop.card.id.clone()),
            ("weights", m.pop.card.sha_main.clone()),
            ("parameters", parameters_key(&m)),
            ("assembly", m.asm.name.clone()),
            ("precision", widths.join(",")),
            ("binary32_math", keeps_f32.to_string()),
            ("half_math", (nmengn::gpu::HALF_MATH.load(Relaxed) && !keeps_f32).to_string()),
            ("segment_attention", nmengn::gpu::SEGMENT_ATTENTION.load(Relaxed).to_string()),
            ("int8_dot", int8_dot.to_string()),
            ("judge_q8_half", nmengn::roles::JUDGE_Q8_HALF.load(Relaxed).to_string()),
            ("instruction", nmengn::roles::CODE_INSTRUCTION.to_owned()),
            ("device", if m.pop.gpu.is_some() { "gpu" } else { "cpu" }.to_owned()),
        ]
        .into_iter()
        .collect();
        let text: String = fields.iter().map(|(k, v)| format!("{k}={}\n", v.replace('\n', " "))).collect();
        return Some(nmhash::sha256_bytes(text.as_bytes()));
    }
    Some(nmhash::sha256_bytes(eng.label().as_bytes()))
}

/// Embed a batch of texts, packing on GPU via `card_embed_batch` when the released card embedder is active.
#[must_use]
pub fn embed_batch(org: &Organism, texts: &[String]) -> Option<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Some(Vec::new());
    }
    if let Some(m) = active_card_embedder(org)
        && m.pop.gpu.is_some()
        && texts.len() >= 2
    {
        let cap = nmengn::roles::input_cap(&m, nmengn::placement::Device::Gpu, 1);
        let max_tokens = nmengn::roles::batch_tokens(&m, cap + 1);
        if let Ok(vs) = nmengn::roles::card_embed_batch(&m, texts, cap, max_tokens)
            && vs.len() == texts.len()
        {
            return Some(vs);
        }
    }
    texts.iter().map(|t| org.slots.embed_neural(t)).collect()
}

/// Rerank many documents against one query, packing on GPU via `card_rerank_batch` when the card reranker is active.
#[must_use]
pub fn rerank_batch(org: &Organism, query: &str, docs: &[String]) -> Option<Vec<f32>> {
    if docs.is_empty() {
        return Some(Vec::new());
    }
    if let Some(m) = active_card_reranker(org)
        && m.pop.gpu.is_some()
        && docs.len() >= 2
        && let Ok(vs) = nmengn::roles::card_rerank_batch(&m, nmengn::placement::Device::Gpu, nmengn::roles::CODE_INSTRUCTION, query, docs, None)
        && vs.len() == docs.len()
    {
        return Some(vs);
    }
    if let Some(v) = org.slots.rerank_set_neural(query, docs) {
        return Some(v);
    }
    docs.iter().map(|d| org.slots.rerank_neural(query, d)).collect()
}

/// Rerank arbitrary `(query, doc)` pairs, packing on GPU via `card_rerank_pairs` when the card reranker is active.
#[must_use]
pub fn rerank_pairs(org: &Organism, pairs: &[(&str, &str)]) -> Option<Vec<f32>> {
    if pairs.is_empty() {
        return Some(Vec::new());
    }
    if let Some(m) = active_card_reranker(org)
        && m.pop.gpu.is_some()
        && pairs.len() >= 2
        && let Ok(vs) = nmengn::roles::card_rerank_pairs(&m, nmengn::placement::Device::Gpu, nmengn::roles::CODE_INSTRUCTION, pairs)
        && vs.len() == pairs.len()
    {
        return Some(vs);
    }
    pairs.iter().map(|(q, d)| org.slots.rerank_neural(q, d)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    static PRECISION_TEST_LOCK: Mutex<()> = Mutex::new(());

    struct PrecisionGuard(Precision);
    impl Drop for PrecisionGuard {
        fn drop(&mut self) {
            if let Ok(mut g) = PRECISION.write() {
                *g = self.0;
            }
        }
    }

    #[test]
    fn apply_precision_sets_q8_before_prefetch() {
        let _lock = PRECISION_TEST_LOCK.lock().unwrap();
        let prev = *PRECISION.read().unwrap();
        let _guard = PrecisionGuard(prev);

        let home = nmtype::nano_root();
        let mut asm_q8 = nmengn::Assembly::load(&home.join("designs/assemblies"), "qwen3-embedding").unwrap();
        *PRECISION.write().unwrap() = Precision { base: None, writer: None, reranker: None, embedder: Some(8) };
        served_width(&mut asm_q8, "qwen3-embedding-0.6b", false);
        let full16 = nmengn::model::apply_precision(&mut asm_q8, "qwen3-embedding-0.6b");
        assert!(!full16);
        assert!(asm_q8.nodes.iter().any(|n| n.op == "linear" && n.precision == "q8"));

        let mut asm_short = nmengn::Assembly::load(&home.join("designs/assemblies"), "qwen3-embedding").unwrap();
        *PRECISION.write().unwrap() = Precision { base: None, writer: None, reranker: None, embedder: None };
        served_width(&mut asm_short, "qwen3-embedding-0.6b", true);
        let full16_short = nmengn::model::apply_precision(&mut asm_short, "qwen3-embedding-0.6b");
        assert!(!full16_short);
        assert!(asm_short.nodes.iter().any(|n| n.op == "linear" && n.precision == "q8"));

        let mut asm_16 = nmengn::Assembly::load(&home.join("designs/assemblies"), "qwen3-embedding").unwrap();
        *PRECISION.write().unwrap() = Precision { base: None, writer: None, reranker: None, embedder: Some(16) };
        served_width(&mut asm_16, "qwen3-embedding-0.6b", true);
        let full16_only = nmengn::model::apply_precision(&mut asm_16, "qwen3-embedding-0.6b");
        assert!(full16_only);
        assert!(asm_16.nodes.iter().filter(|n| n.op == "linear").all(|n| n.precision == "f16"));
    }

    #[test]
    #[ignore = "needs weights"]
    fn prefetch_q8_matches_cold_bind() {
        let _lock = PRECISION_TEST_LOCK.lock().unwrap();
        let prev = *PRECISION.read().unwrap();
        let _guard = PrecisionGuard(prev);
        *PRECISION.write().unwrap() = Precision { base: None, writer: None, reranker: Some(8), embedder: None };

        let home = nmtype::nano_root();
        let pop = Population::open(&home, "qwen3-reranker-0.6b", true, false).unwrap();
        let mut asm = nmengn::Assembly::load(&home.join("designs/assemblies"), "qwen3-reranker").unwrap();
        served_width(&mut asm, "qwen3-reranker-0.6b", true);
        let prefetched = bind_prefetched(&pop, asm.clone()).unwrap();
        let cold = Model::bind(&pop, asm).unwrap();
        let (a, _, _) = nmengn::roles::card_rerank(&prefetched, nmengn::roles::CODE_INSTRUCTION, "where is parse", "fn parse() {}").unwrap();
        let (b, _, _) = nmengn::roles::card_rerank(&cold, nmengn::roles::CODE_INSTRUCTION, "where is parse", "fn parse() {}").unwrap();
        assert!((a - b).abs() < 1e-6);
    }
}

#[cfg(test)]
mod part_tests {
    use super::next_part;

    /// Roadmap Phase 4.6 (2026-11-24): the dense part grows while fast, shrinks when slow, and
    /// stays within 1 and 64 chunks.
    #[test]
    fn dense_part_adapts_within_bounds() {
        assert_eq!(next_part(4, 50), 8);
        assert_eq!(next_part(64, 10), 64);
        assert_eq!(next_part(8, 200), 8);
        assert_eq!(next_part(8, 400), 4);
        assert_eq!(next_part(1, 900), 1);
    }
}
