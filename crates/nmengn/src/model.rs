//! The population and its bound assemblies (PLAN 4c `model`, guide 9.4, 9.5): one
//! `Population` per checkpoint (card, tensor descriptors, clusters, one NPCS provisioner,
//! tokenizer, the GPU if present, the placement table, device copies of bound weights); a
//! `Model` is an assembly whose nodes have bound their clusters through NPCS. Two models over
//! one population share every cluster they both bind (one `Arc<View>` per cluster) and every
//! device copy with the same binding. A model runs on the device placement chooses.
use crate::adapter::Adapter;
use crate::assembly::{Assembly, Node};
use crate::cpu::{self, Linear};
use crate::gpu::{DevT, Ops};
use crate::placement::{Device, Placement};
use nmnpcs::{Provisioner, TensorDesc, View, WeightCard, compat};
use nmtype::{Fault, FaultKind};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmengn.model", FaultKind::Execution, m)
}

/// Population dimensions from config.json.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dims {
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub vocab: usize,
    pub eps: f32,
    pub theta: f32,
}

/// The source label of a population read from the released files.
pub const OPEN_WEIGHTS: &str = "open weights";

/// The overlay of `key` applied as a population opens, when `overlay::AT_OPEN` asks for it (lab
/// runs): the patched descriptors and the source label (`open weights with overlay <id>` for the
/// open weights). Otherwise, or with no overlay, the descriptors as they are. A failure is said and
/// the released values are read.
fn at_open(home: &Path, key: &str, descs: Vec<TensorDesc>, dir: &Path, source: String) -> (Vec<TensorDesc>, String) {
    if !crate::overlay::AT_OPEN.load(Ordering::Relaxed) {
        return (descs, source);
    }
    match crate::overlay::patch(home, key, &descs, dir) {
        Ok(Some(p)) => {
            eprintln!("engine: parameter overlay of {key} ({}) applied to {} tensors as the population opened", p.identity, p.tensors.len());
            let model = key.strip_prefix("open:").unwrap_or(key).to_owned();
            crate::overlay::hold(&model, p.held);
            let label = if source == OPEN_WEIGHTS { format!("{OPEN_WEIGHTS} with overlay {}", p.identity) } else { source };
            (p.descs, label)
        }
        Ok(None) => (descs, source),
        Err(e) => {
            eprintln!("engine: parameter overlay of {key} not applied, the released values are read: {e}");
            (descs, source)
        }
    }
}

/// One MicroModel's parameters as a population: from the open weights, or from one complete base
/// cartridge (`source` says which; never a mixture).
pub struct Population {
    pub home: std::path::PathBuf,
    /// `OPEN_WEIGHTS`, or `cartridge <id@version>`.
    pub source: String,
    pub card: WeightCard,
    pub tensors: Vec<TensorDesc>,
    pub clusters: Vec<nmnpcs::ParameterCluster>,
    index: BTreeMap<String, usize>,
    pub prov: Arc<Provisioner>,
    pub tokenizer: nmtokz::Tokenizer,
    pub dims: Dims,
    pub gpu: Option<Ops>,
    pub gpu_note: String,
    pub placement: Placement,
    dev: Mutex<BTreeMap<String, DevT>>,
    q8: Mutex<BTreeMap<String, std::sync::Weak<cpu::Q8>>>,
    /// Weight placement (2026-11-27): the 8-bit packets kept resident by tensor, while the host
    /// copies are (`Model::drop_host` lets them go with the tensors they were made from).
    q8_keep: Mutex<BTreeMap<String, Arc<cpu::Q8>>>,
    /// Why the host copies are where they are, in words (`host_state`).
    host_note: Mutex<String>,
    /// GPU runs in progress on this population (step P2: its weights stay while it is used).
    active: AtomicUsize,
    /// Use counter of its last GPU run or weight upload (`residency::tick`).
    last_use: AtomicU64,
    /// Weight uploads measured: (seconds, bytes) in all (the reload cost per byte).
    uploads: Mutex<(f64, u64)>,
}
impl std::fmt::Debug for Population {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Population").field("card", &self.card.id).field("clusters", &self.clusters.len()).finish_non_exhaustive()
    }
}
impl Population {
    /// Open `weights/<id>` under `home`: card, header, clusters, provisioner, tokenizer, GPU.
    /// `verify` recomputes the main file's hash (stamped, so only once per file version).
    /// # Errors
    /// A missing or corrupt population.
    pub fn open(home: &Path, id: &str, verify: bool, gpu: bool) -> Result<Arc<Self>, Fault> {
        let card = WeightCard::load(home, id)?;
        if verify {
            card.verify(home)?;
        }
        let index = nmsafe::TensorIndex::open(&card.dir).map_err(|e| bad(format!("{id}: header: {e}")))?;
        // Canonical names (2026-11-09): a headless release (Qwen3-Embedding) stores `layers.N...`
        // and `embed_tokens.weight` without the `model.` prefix; clusters and assemblies use the
        // prefixed names. Only the name in the index changes; the bytes are read by offset.
        let canon = |n: &str| if n.starts_with("model.") || n == "lm_head.weight" { n.to_owned() } else { format!("model.{n}") };
        let tensors: Vec<TensorDesc> = index
            .tensors
            .values()
            .map(|t| TensorDesc {
                name: canon(&t.name),
                dtype: t.dtype.clone(),
                shape: t.shape.iter().map(|x| usize::try_from(*x).unwrap_or(0)).collect(),
                file: t.file.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default(),
                start: t.start,
                end: t.end,
                sha256: String::new(),
            })
            .collect();
        let (tensors, source) = at_open(home, &format!("open:{id}"), tensors, &card.dir, OPEN_WEIGHTS.to_owned());
        Self::build(home, card, tensors, source, gpu)
    }
    /// A population whose parameters all come from one base cartridge: each descriptor in `parts`
    /// names a parameter set by the tensor it loads onto (its layer and block), its element type,
    /// shape and byte range in the cartridge file (an absolute path). The open weights' files are
    /// not read; the weight card gives only the configuration and the tokenizer. Whether the parts
    /// are complete and connect is checked by `base::seat` before anything binds.
    /// # Errors
    /// A missing card or tokenizer, or a parameter set the cut into clusters does not know.
    pub fn open_cartridge(home: &Path, id: &str, gpu: bool, key: &str, parts: Vec<TensorDesc>) -> Result<Arc<Self>, Fault> {
        let card = WeightCard::load(home, id)?;
        let (parts, source) = at_open(home, key, parts, &card.dir, format!("cartridge {key}"));
        Self::build(home, card, parts, source, gpu)
    }
    /// A population over `parts` labelled `source` (the open weights or a base cartridge with a
    /// parameter overlay applied, `overlay::patch`): relative file names are read in the population
    /// folder, absolute ones where they are, and a part with a content hash held in RAM is taken
    /// from there. The completeness of the parts is the caller's (they are the descriptors of a
    /// population already opened, with touched tensors offered under new hashes).
    /// # Errors
    /// As `open_cartridge`.
    pub fn open_parts(home: &Path, id: &str, gpu: bool, source: String, parts: Vec<TensorDesc>) -> Result<Arc<Self>, Fault> {
        let card = WeightCard::load(home, id)?;
        Self::build(home, card, parts, source, gpu)
    }
    fn build(home: &Path, card: WeightCard, tensors: Vec<TensorDesc>, source: String, gpu: bool) -> Result<Arc<Self>, Fault> {
        let id = card.id.clone();
        let id = id.as_str();
        let dims = Dims {
            layers: card.int("num_hidden_layers")?,
            hidden: card.int("hidden_size")?,
            heads: card.int("num_attention_heads")?,
            kv_heads: card.int("num_key_value_heads")?,
            head_dim: card.int("head_dim")?,
            inter: card.int("intermediate_size")?,
            vocab: card.int("vocab_size")?,
            eps: card.float("rms_norm_eps")?,
            theta: card.float("rope_theta")?,
        };
        let clusters = nmnpcs::decompose::decompose(id, dims.heads, dims.kv_heads, dims.head_dim, &tensors)?;
        let idx = clusters.iter().enumerate().map(|(i, c)| (c.id.clone(), i)).collect();
        let prov = Arc::new(Provisioner::new(id, &card.dir, &clusters));
        let tokenizer = nmtokz::Tokenizer::open(&card.dir.join("tokenizer.json")).map_err(|e| bad(format!("tokenizer: {e}")))?;
        // Roadmap Phase 4.5 (2026-11-25): `limits.vram_mb` of 0 means processor only, latched
        // when the first MicroModel of this process opens (a change applies at the next start).
        let (gpu, gpu_note) = if gpu && crate::limits::cpu_only_at_start() {
            (None, "processor only: limits.vram_mb is 0 in settings.json (set it to auto or a size and restart to use the card)".to_owned())
        } else if gpu {
            // Roadmap Phase 4.6 (2026-11-25): the process's one card (`device`: `NANO_GPU`,
            // `CUDA_VISIBLE_DEVICES`, else the largest card the kernels run on); every way of not
            // getting one leaves a plain-English processor line here.
            match crate::device::open() {
                Ok(g) => {
                    let note =
                        format!("{} (compute {}.{}, {} MB)", g.info().name, g.info().major, g.info().minor, g.info().total_bytes >> 20);
                    // Roadmap Phase 4.6 (2026-11-25): the per-card dp4a kernel picks, measured once
                    // per card and kept in machine.json (`NANO_AUTOTUNE=off` keeps the defaults).
                    // All four variants are bit-identical (`gpu::check_dp4a_tile`), so a pick
                    // changes speed only; a failed measurement keeps the defaults.
                    if let Err(e) = crate::autotune::tune_on_device(home, &g) {
                        eprintln!("autotune: kept the default dp4a kernels: {e}");
                    }
                    (Some(Ops { gpu: g }), note)
                }
                Err(e) => (None, e),
            }
        } else {
            (None, "GPU not requested".into())
        };
        // Roadmap Phase 4.6 (2026-11-25): the measured placement of this machine's card only; a
        // different card starts from neutral and measures its own.
        let machine = gpu.as_ref().map(|o| crate::capacity::machine_key(Some((o.gpu.info().name.as_str(), o.gpu.info().total_bytes >> 20))));
        let placement = Placement::load_for(&home.join("designs/placement.json"), machine.as_deref());
        let pop = Arc::new(Self {
            home: home.to_owned(),
            source,
            card,
            tensors,
            clusters,
            index: idx,
            prov,
            tokenizer,
            dims,
            gpu,
            gpu_note,
            placement,
            dev: Mutex::new(BTreeMap::new()),
            q8: Mutex::new(BTreeMap::new()),
            q8_keep: Mutex::new(BTreeMap::new()),
            host_note: Mutex::new(String::new()),
            active: AtomicUsize::new(0),
            last_use: AtomicU64::new(0),
            uploads: Mutex::new((0.0, 0)),
        });
        // Step P2: the device residency manager knows every population with device weights.
        crate::residency::get().register(&pop);
        Ok(pop)
    }
    /// GPU runs in progress on this population.
    #[must_use]
    pub fn active(&self) -> usize {
        self.active.load(Ordering::SeqCst)
    }
    /// Mark a GPU run starting (true) or ending (false); a start also marks the last use.
    pub fn mark_active(&self, on: bool) {
        if on {
            self.active.fetch_add(1, Ordering::SeqCst);
            self.last_use.store(crate::residency::tick(), Ordering::Relaxed);
        } else {
            let _ = self.active.try_update(Ordering::SeqCst, Ordering::SeqCst, |n| Some(n.saturating_sub(1)));
        }
    }
    /// The use counter of the last GPU run or weight upload.
    #[must_use]
    pub fn last_use(&self) -> u64 {
        self.last_use.load(Ordering::Relaxed)
    }
    /// Weight bytes uploaded to the device so far (every upload counted).
    #[must_use]
    pub fn uploaded_bytes(&self) -> u64 {
        self.uploads.lock().map_or(0, |u| u.1)
    }
    /// Measured weight upload time per byte (0 before any upload).
    #[must_use]
    pub fn reload_secs_per_byte(&self) -> f64 {
        #[expect(clippy::cast_precision_loss, reason = "a cost ratio")]
        self.uploads.lock().map_or(0.0, |u| if u.1 == 0 { 0.0 } else { u.0 / u.1 as f64 })
    }
    /// Release every device copy unless another thread is uploading or reading the table now
    /// (step P2, for the residency manager); false when nothing was released.
    pub fn try_release_device(&self) -> bool {
        let Ok(mut d) = self.dev.try_lock() else { return false };
        if d.is_empty() {
            return false;
        }
        d.clear();
        drop(d);
        for c in &self.clusters {
            self.prov.set_device(&c.id, 0);
        }
        true
    }
    /// A cluster by id.
    /// # Errors
    /// Unknown id.
    pub fn cluster(&self, id: &str) -> Result<nmnpcs::ParameterCluster, Fault> {
        if let Some(c) = self.index.get(id).and_then(|i| self.clusters.get(*i)) {
            return Ok(c.clone());
        }
        // A row or column range of a tensor cut whole (2026-11-11, cartridge build step C, the
        // channel-block mask): `<model>/<tensor>/rows<a>-<b>` or `cols<a>-<b>`. It reads the same
        // tensor bytes as the whole cluster; only the region differs. Registered on first use.
        let unknown = || bad(format!("unknown cluster {id}"));
        let (stem, region) = id.rsplit_once('/').ok_or_else(unknown)?;
        let whole = self.index.get(&format!("{stem}/whole")).and_then(|i| self.clusters.get(*i)).ok_or_else(unknown)?;
        let [r, k] = whole.shape.as_slice() else { return Err(unknown()) };
        let range = |p: &str| {
            region
                .strip_prefix(p)
                .and_then(|x| x.split_once('-'))
                .and_then(|(a, b)| Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?)))
        };
        let (region, n) = match (range("rows"), range("cols")) {
            (Some((a, b)), _) if a < b && b <= *r => (nmnpcs::Region::Rows(a..b), (b - a) * k),
            (_, Some((a, b))) if a < b && b <= *k => (nmnpcs::Region::Cols(a..b), (b - a) * r),
            _ => return Err(unknown()),
        };
        let width = whole.bytes / u64::try_from(r * k).map_err(|e| bad(e.to_string()))?.max(1);
        let mut c = whole.clone();
        id.clone_into(&mut c.id);
        c.region = region;
        c.bytes = u64::try_from(n).map_err(|e| bad(e.to_string()))? * width;
        self.prov.lifecycle.register(id);
        Ok(c)
    }
    /// The int8 precision of a cluster's whole tensor, quantized once and shared while used.
    /// # Errors
    /// A malformed tensor.
    pub fn q8(&self, v: &View) -> Result<Arc<cpu::Q8>, Fault> {
        if let Some(q) = self.q8.lock().map_err(|_| bad("q8 table lock"))?.get(&v.cluster.tensor).and_then(std::sync::Weak::upgrade) {
            return Ok(q);
        }
        // 2026-11-12: quantized outside the lock, so several tensors quantize at once
        // (`prefetch_q8`); if another thread finished the same tensor first, its copy is kept.
        // 2026-11-27: from the host copy, read again from its file when it was dropped.
        let k = v.cluster.shape.get(1).copied().unwrap_or(1);
        let bytes = self.prov.bytes(v)?;
        let q = Arc::new(cpu::Q8::from_bf16(&bytes, k).map_err(|e| bad(format!("{}: {e}", v.cluster.tensor)))?);
        drop(bytes);
        let mut m = self.q8.lock().map_err(|_| bad("q8 table lock"))?;
        if let Some(other) = m.get(&v.cluster.tensor).and_then(std::sync::Weak::upgrade) {
            return Ok(other);
        }
        m.insert(v.cluster.tensor.clone(), Arc::downgrade(&q));
        Ok(q)
    }
    /// Cartridge loading step L (2026-11-12): prepare an assembly's 8-bit packets ahead of its
    /// job (file pages read and every `q8` linear tensor quantized), on up to `threads` processor
    /// threads, while another job runs. The returned handles keep the packets alive; a `bind` of
    /// the assembly made while they are held quantizes nothing.
    /// # Errors
    /// An unknown cluster or a malformed tensor.
    pub fn prefetch_q8(&self, asm: &crate::assembly::Assembly, threads: usize) -> Result<Vec<Arc<cpu::Q8>>, Fault> {
        // Every cluster the 8-bit maps bind, grouped by tensor: one thread takes a tensor's
        // clusters together, so the tensor is read once and every view of it is dropped (a bind
        // after this finds every view and reads nothing).
        let mut by_tensor: BTreeMap<String, Vec<nmnpcs::ParameterCluster>> = BTreeMap::new();
        let mut seen = std::collections::BTreeSet::new();
        for n in asm.nodes.iter().filter(|n| n.precision == "q8" && n.op == "linear") {
            for r in &n.clusters {
                let c = self.cluster(&format!("{}/{r}", self.card.id))?;
                if seen.insert(c.id.clone()) {
                    by_tensor.entry(c.tensor.clone()).or_default().push(c);
                }
            }
        }
        let clusters: Vec<Vec<nmnpcs::ParameterCluster>> = by_tensor.into_values().collect();
        // Weight placement (2026-11-27): each tensor is read, quantized, and its 16-bit host copy
        // dropped at once (an 8-bit map reads only its packets; a 16-bit reader reads the file
        // range again on demand), so a load holds one 16-bit tensor per thread instead of the
        // whole model beside its 8-bit copy. With a card the packets are not made here: staging
        // makes them in batches and copies them straight to the card (`stage_parts`).
        let quantize = self.gpu.is_none();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let out: Mutex<Vec<Arc<cpu::Q8>>> = Mutex::new(Vec::new());
        let failed: Mutex<Option<Fault>> = Mutex::new(None);
        std::thread::scope(|s| {
            for _ in 0..threads.max(1) {
                s.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(group) = clusters.get(i) else { break };
                        match group.iter().map(|c| self.prov.acquire(c)).collect::<Result<Vec<_>, Fault>>().and_then(|views| {
                            let q = match views.first() {
                                Some(v) if quantize => Some(self.q8(v)?),
                                _ => None,
                            };
                            for v in &views {
                                self.prov.drop_host(v);
                            }
                            Ok(q)
                        }) {
                            Ok(q) => {
                                if let (Some(q), Ok(mut o)) = (q, out.lock()) {
                                    o.push(q);
                                }
                            }
                            Err(e) => {
                                if let Ok(mut f) = failed.lock() {
                                    *f = Some(e);
                                }
                            }
                        }
                    }
                });
            }
        });
        if let Some(e) = failed.into_inner().ok().flatten() {
            return Err(e);
        }
        out.into_inner().map_err(|_| bad("prefetch lock"))
    }
    /// The 8-bit packets of a tensor when they are in RAM now (no read, no quantization).
    #[must_use]
    pub fn q8_held(&self, v: &View) -> Option<Arc<cpu::Q8>> {
        self.q8.lock().ok()?.get(&v.cluster.tensor).and_then(std::sync::Weak::upgrade)
    }
    /// The 8-bit packets of `views` (one per distinct tensor), quantized on up to `threads`
    /// processor threads; the returned handles keep them alive.
    /// # Errors
    /// A host copy that cannot be read, or a malformed tensor.
    pub fn q8_views(&self, views: &[Arc<View>], threads: usize) -> Result<Vec<Arc<cpu::Q8>>, Fault> {
        // One view per tensor: the packets are a whole tensor's, shared by its views.
        let mut seen = BTreeSet::new();
        let views: Vec<&Arc<View>> = views.iter().filter(|v| seen.insert(v.cluster.tensor.clone())).collect();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let out: Mutex<Vec<Arc<cpu::Q8>>> = Mutex::new(Vec::new());
        let failed: Mutex<Option<Fault>> = Mutex::new(None);
        std::thread::scope(|s| {
            for _ in 0..threads.max(1) {
                s.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(v) = views.get(i) else { break };
                        match self.q8(v) {
                            Ok(q) => {
                                if let Ok(mut o) = out.lock() {
                                    o.push(q);
                                }
                            }
                            Err(e) => {
                                if let Ok(mut f) = failed.lock() {
                                    *f = Some(e);
                                }
                            }
                        }
                    }
                });
            }
        });
        if let Some(e) = failed.into_inner().ok().flatten() {
            return Err(e);
        }
        out.into_inner().map_err(|_| bad("prefetch lock"))
    }
    /// Host memory of the weights (2026-11-27, weight placement): the tensor bytes in RAM now
    /// (kept, or read again and still held by a reader) and the 8-bit packets kept resident.
    #[must_use]
    pub fn host_bytes(&self) -> (u64, u64) {
        let q8 = self.q8_keep.lock().map_or(0, |m| m.values().map(|q| (q.values.len() + q.scales.len() * 4) as u64).sum());
        (self.prov.totals().0, q8)
    }
    /// Where the host copies are: `dropped` (the large tensors read again from their files on
    /// demand) or `resident`, and why, in words.
    #[must_use]
    pub fn host_state(&self) -> (&'static str, String) {
        // A tensor whose 16-bit copy is dropped but whose 8-bit packets are kept is still resident
        // for its 8-bit map (the form its runs read).
        let keep = self.q8_keep.lock().map(|k| k.keys().cloned().collect::<BTreeSet<String>>()).unwrap_or_default();
        let views = self.prov.views();
        let dropped = views.iter().filter(|v| !v.resident() && !keep.contains(&v.cluster.tensor)).count();
        let packets = views.iter().filter(|v| !v.resident() && keep.contains(&v.cluster.tensor)).count();
        let note = self.host_note.lock().map(|n| n.clone()).unwrap_or_default();
        if dropped > 0 {
            let note = if note.is_empty() { "the card holds them; read again from their files on demand".to_owned() } else { note };
            return ("dropped", format!("{dropped} tensors: {note}"));
        }
        let why = if self.gpu.is_none() {
            "processor only: the processor runs read them".to_owned()
        } else if !note.is_empty() {
            note
        } else {
            "not on the card yet: they upload at the first run, then the host copy is dropped at the next staging".to_owned()
        };
        let why = if packets > 0 { format!("{why} ({packets} tensors held as 8-bit packets only)") } else { why };
        ("resident", why)
    }
    fn set_host_note(&self, note: String) {
        if let Ok(mut n) = self.host_note.lock() {
            *n = note;
        }
    }
    /// Free device memory, when a GPU is open. Roadmap Phase 4.5 (2026-11-25): the effective
    /// figure under the user's card limit (`residency::Residency::effective_free`), so the batch
    /// and input caps sized from it (`roles::batch_tokens`, `input_cap`, `card_doc_cap`, the
    /// drone and reader groups) tile within the limit; with `auto` the driver's figure.
    #[must_use]
    pub fn gpu_free(&self) -> Option<u64> {
        let o = self.gpu.as_ref()?;
        let (free, _) = o.gpu.memory().ok()?;
        Some(crate::residency::get().effective_free(&o.gpu, free) as u64)
    }
    /// Device copy for a binding key, uploaded once and shared.
    fn device(&self, key: &str, views: &[Arc<View>], make: impl FnOnce() -> Result<DevT, String>) -> Result<DevT, Fault> {
        let mut d = self.dev.lock().map_err(|_| bad("device table lock"))?;
        if let Some(t) = d.get(key) {
            return Ok(t.clone());
        }
        let t = {
            let _p = nmffi::prof::Span::new("weights.upload");
            // Step P2: room is made in the shared device budget first (idle pool buffers, kept
            // windows, then idle weights of other populations), the copy is a long-lived
            // allocation, and its upload time is recorded as this population's reload cost.
            let need: u64 = views.iter().map(|v| v.cluster.bytes).sum();
            crate::residency::get().make_room(usize::try_from(need).unwrap_or(usize::MAX), &self.card.id);
            let _k = nmffi::KeptScope::new();
            let started = std::time::Instant::now();
            let t = make().map_err(bad)?;
            if let Ok(mut u) = self.uploads.lock() {
                u.0 += started.elapsed().as_secs_f64();
                u.1 += t.buf.bytes() as u64;
            }
            self.last_use.store(crate::residency::tick(), Ordering::Relaxed);
            t
        };
        // Device bytes are attributed to the bound clusters in proportion to their size.
        let total: u64 = views.iter().map(|v| v.cluster.bytes).sum::<u64>().max(1);
        for v in views {
            let share = (t.buf.bytes() as u64).saturating_mul(v.cluster.bytes) / total;
            self.prov.set_device(&v.cluster.id, v.device_bytes() + usize::try_from(share).unwrap_or(0));
        }
        d.insert(key.to_owned(), t.clone());
        Ok(t)
    }
    /// Put a device copy made elsewhere (the copy stream, `Model::stage`) under `key`, with the same
    /// room making, reload-cost record and attribution as `device`. Returns false when the key is
    /// already held (the given copy is then dropped).
    fn device_staged(&self, key: &str, views: &[Arc<View>], t: DevT, secs: f64) -> bool {
        let Ok(mut d) = self.dev.lock() else { return false };
        if d.contains_key(key) {
            return false;
        }
        if let Ok(mut u) = self.uploads.lock() {
            u.0 += secs;
            u.1 += t.buf.bytes() as u64;
        }
        self.last_use.store(crate::residency::tick(), Ordering::Relaxed);
        let total: u64 = views.iter().map(|v| v.cluster.bytes).sum::<u64>().max(1);
        for v in views {
            let share = (t.buf.bytes() as u64).saturating_mul(v.cluster.bytes) / total;
            self.prov.set_device(&v.cluster.id, v.device_bytes() + usize::try_from(share).unwrap_or(0));
        }
        d.insert(key.to_owned(), t);
        true
    }
    /// Whether a device copy is held under `key`.
    #[must_use]
    pub fn holds_device(&self, key: &str) -> bool {
        self.dev.lock().is_ok_and(|d| d.contains_key(key))
    }
    /// Drop the device copy held under `key` (bound to `views`) with its share of the
    /// attribution; a later reader uploads it again. Returns the bytes freed.
    fn release_key(&self, key: &str, views: &[Arc<View>]) -> usize {
        let Some(t) = self.dev.lock().ok().and_then(|mut d| d.remove(key)) else { return 0 };
        let total: u64 = views.iter().map(|v| v.cluster.bytes).sum::<u64>().max(1);
        for v in views {
            let share = (t.buf.bytes() as u64).saturating_mul(v.cluster.bytes) / total;
            self.prov.set_device(&v.cluster.id, v.device_bytes().saturating_sub(usize::try_from(share).unwrap_or(0)));
        }
        t.buf.bytes()
    }
    /// Release every device copy (the population stays in RAM); NPCS residency follows.
    pub fn release_device(&self) {
        let _p = nmffi::prof::Span::new("weights.release");
        if let Ok(mut d) = self.dev.lock() {
            d.clear();
        }
        for c in &self.clusters {
            self.prov.set_device(&c.id, 0);
        }
    }
    /// Device bytes held by weight copies.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.dev.lock().map(|d| d.values().map(|t| t.buf.bytes()).sum()).unwrap_or(0)
    }
    /// `device_bytes` without waiting: `None` while the device table is locked (an upload in
    /// progress). Roadmap Phase 4.5 (2026-11-25): the card limit reads it during room-making,
    /// which an upload asks for while it holds that lock.
    #[must_use]
    pub fn device_bytes_now(&self) -> Option<usize> {
        self.dev.try_lock().ok().map(|d| d.values().map(|t| t.buf.bytes()).sum())
    }
}

impl Bound {
    /// Width of the logits a `labelsoftmax` node reads: the vocabulary, or its label rows.
    fn label_width(&self, vocab: usize) -> usize {
        if self.rows.is_empty() { vocab } else { self.rows.len() }
    }
    /// Where a `labelsoftmax` node finds its labels in those logits: the token ids, or the
    /// positions of the label rows.
    fn label_read(&self) -> Vec<usize> {
        if self.rows.is_empty() { self.label_ids.clone() } else { (0..self.rows.len()).collect() }
    }
}

/// One bound node: its assembly node, the cluster views it holds, its weights in usable form.
pub struct Bound {
    pub node: Node,
    pub views: Vec<Arc<View>>,
    pub lin: Option<Linear>,
    pub vec: Option<Vec<f32>>,
    /// Token ids of a `labelsoftmax` node's labels, resolved at bind.
    pub label_ids: Vec<usize>,
    /// Output rows read (2026-11-10, teardown patch 7, `Model::label_rows`): on a `head` node
    /// whose logits feed only a `labelsoftmax` node, the label tokens' rows of the output head
    /// (logits one per row, in this order); on that `labelsoftmax` node the same list (its logits
    /// are then those rows). Empty: the full head over the vocabulary.
    pub rows: Vec<usize>,
    key: String,
    /// The global key-value heads this node's keys and values hold (2026-11-10, step P0):
    /// a `score` node's own range; a `rope` node's is that of the `score` node its keys feed.
    kv: Option<(usize, usize)>,
}

/// An activation: CPU values, a device tensor, or integers (ids, positions).
#[derive(Clone, Debug)]
pub enum Act {
    Cpu(Vec<f32>),
    Gpu(DevT),
    Ids(Vec<i32>),
}

/// Run options: query heads switched off per layer (golden heads-off case) and whether the
/// head computes only the last position.
#[derive(Clone, Debug, Default)]
pub struct RunOpts {
    pub heads_off: BTreeMap<usize, Vec<usize>>,
    /// Feed-forward channel blocks switched off per layer (2026-11-11, cartridge build step E,
    /// block lesions): blocks of 64 channels of the layer's `swigate` output, numbered over the
    /// output as bound, are set to zero. Empty (the default) leaves every run exactly as before.
    pub blocks_off: BTreeMap<usize, Vec<usize>>,
    /// Steering vectors per layer (2026-11-12, continuous learning L5, a mapped in-model
    /// cartridge): the vector, times its scale, is added to every position of the layer's output
    /// (its `res2` residual), rounded to binary16 as the engine's activations are. Empty (the
    /// default) leaves every run exactly as before; unloading is removing the entry.
    pub steer: BTreeMap<usize, (Arc<Vec<f32>>, f32)>,
    pub all_positions: bool,
    /// Key and value cache for decoding (one entry per attention component).
    pub cache: Option<Arc<Cache>>,
    /// A read-only key and value prefix shared by reference (E1, 2026-11-10): positions
    /// `0..window.len` come from it; a decode cache, when also given, follows it.
    pub window: Option<Arc<crate::kvstore::Window>>,
    /// Records keys before rope and values of the new tokens per layer (chunk encoding).
    pub capture: Option<Arc<crate::kvstore::Capture>>,
    /// Records per-head attention mass on the window for the listed layers (drone reading).
    pub probe: Option<Arc<crate::kvstore::Probe>>,
    /// 2026-11-10 (diagnostic A of `docs/2026-11-10-neumor-design-fundamentals-of-the-read.md`):
    /// every query, after rope, multiplied by this factor (an attention temperature of 1 / factor
    /// at every layer the read passes through). Processor only; `None` (the default) leaves every
    /// run exactly as before.
    pub q_scale: Option<f32>,
    /// Packed independent sequences (2026-11-13, batched embedding): their lengths, in order,
    /// summing to the tokens of the run. Positions restart at the run's start for each; attention
    /// runs within each sequence only; `lastpool` gives one pooled vector per sequence, `head`
    /// and `labelsoftmax` one row per sequence (its last token). With a `window` or a `cache`,
    /// every sequence reads that shared prefix (window first) as its own first positions, starts
    /// right after it, and the cache is never written (one prefilled prompt head or one encoded
    /// context, many continuations in one run). GPU only; no capture or probe with it. `None`
    /// (the default) leaves every run as before.
    pub segments: Option<Arc<Vec<usize>>>,
    /// Targets stay on the card (`Act::Gpu`) instead of coming back as host values, for a run
    /// that continues from them. False (the default) leaves every run as before.
    pub device_targets: bool,
    /// Output rows this run reads from a full `head` node (for example the yes and no rows of a
    /// label tail): the head's logits are those rows only, in this order, from a device copy of
    /// those rows. Empty (the default) gives the whole vocabulary. A node already bound to label
    /// rows keeps its own.
    pub head_rows: Arc<Vec<usize>>,
    /// Relevance-head reads of packed groups (lane S, 2026-11-18, T-074): with `segments`, one
    /// probe per sequence and the number of its last rows that query (`kvstore::SegmentProbe`,
    /// `kvstore::read_groups`). GPU only; no decode cache with it. `None` (the default) leaves every
    /// run as before.
    pub segment_probe: Option<Arc<crate::kvstore::SegmentProbe>>,
    /// Tails over a group's body (lane S package 5, 2026-11-18): with `segments`, entry `i` =
    /// `Some(j)` makes sequence `i` read sequence `j` (an earlier one, itself no tail) after the
    /// shared prefix and before its own rows, its positions continuing after `j`'s; two questions
    /// over one group's pieces in one run, the pieces computed once. Needs the one-launch packed
    /// attention. `None` (the default) leaves every run as before.
    pub segment_bodies: Option<Arc<Vec<Option<usize>>>>,
    /// Part scores reduced on the card for tail sequences (lane S package 5,
    /// `kvstore::SegmentParts`, `kvstore::read_group_parts`). With `segments` and
    /// `segment_bodies`. `None` (the default) leaves every run as before.
    pub segment_parts: Option<Arc<crate::kvstore::SegmentParts>>,
    /// Per-hop `gain-bias` cartridge filter stack (WP-A2, 2026-11-22): composed on top of
    /// `Model::filters_in_use()` for this forward call only, without mutating global `RACK`.
    /// `None` (the default) leaves every run as before.
    pub filters: Option<Arc<crate::filter::Plugged>>,
    /// When true, `final_norm` targets only need the last token row of each segment (or the
    /// last token overall), allowing the X3 last-layer prefill row gather to prune dead tokens
    /// right after the last layer's attention score. False (the default) preserves full rows.
    pub last_rows_only: bool,
}

/// Keys and values kept per attention component between decode steps (session state; the
/// components stay stateless). CPU entries are binary16 bits (lossless: every activation is a
/// binary16 value; half the RAM of f32, E1 2026-11-10), GPU entries device tensors.
#[derive(Debug, Default)]
pub struct Cache {
    entries: Mutex<BTreeMap<String, Kv>>,
}
#[derive(Debug, Clone)]
enum Kv {
    Cpu(Vec<u16>, Vec<u16>),
    Gpu(DevT, DevT),
}
impl Cache {
    /// Positions held (from any entry), 0 when empty.
    #[must_use]
    pub fn len(&self, kv_width: usize) -> usize {
        self.entries
            .lock()
            .ok()
            .and_then(|e| {
                e.values().next().map(|k| match k {
                    Kv::Cpu(k, _) => k.len() / kv_width.max(1),
                    Kv::Gpu(k, _) => k.shape.first().copied().unwrap_or(0),
                })
            })
            .unwrap_or(0)
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.lock().map_or(true, |e| e.is_empty())
    }
    /// A copy of this cache to continue from (2026-11-10: one prefilled prompt head, many
    /// continuations). Entries are shared until a continuation appends, which always writes
    /// new buffers, so the original is never changed.
    #[must_use]
    pub fn fork(&self) -> Self {
        Self { entries: Mutex::new(self.entries.lock().map(|e| e.clone()).unwrap_or_default()) }
    }
    /// Rewind every cached attention entry to at most `keep` token positions (zero-copy on GPU,
    /// truncates `Vec` on CPU), used by speculative verification to discard rejected draft tokens.
    pub fn truncate(&self, keep: usize, kv_width: usize) {
        let Ok(mut map) = self.entries.lock() else { return };
        for kv in map.values_mut() {
            match kv {
                Kv::Cpu(k, v) => {
                    let n = keep.saturating_mul(kv_width);
                    if k.len() > n {
                        k.truncate(n);
                    }
                    if v.len() > n {
                        v.truncate(n);
                    }
                }
                Kv::Gpu(k, v) => {
                    if let Some(r) = k.shape.first_mut()
                        && *r > keep
                    {
                        *r = keep;
                    }
                    if let Some(r) = v.shape.first_mut()
                        && *r > keep
                    {
                        *r = keep;
                    }
                }
            }
        }
    }
}

/// An assembly bound to a population. `units` holds the learning-weight units loaded on its
/// `linear` nodes (by node id); a node's own clusters are never written (guide 10.1).
/// One staged upload: device key, the bound node it copies, shape, element type, the views it serves,
/// and whether it is an 8-bit map (its host bytes are made when its copy starts).
type StageJob = (String, usize, Vec<usize>, i32, Vec<Arc<View>>, bool);

/// Chunk size of a staged copy (two pinned buffers of this size are used): 32 MiB.
pub const STAGE_CHUNK: usize = 32 << 20;

/// Elements of 8-bit maps whose packets one staging batch makes at a time (2026-11-27, weight
/// placement): 256 Mi, about 256 MiB of packets.
pub const STAGE_BATCH: usize = 256 << 20;

/// What `Model::stage` did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Staged {
    pub tensors: usize,
    pub bytes: u64,
    /// Host time preparing the bytes, and time the copies took on the copy stream.
    pub host_s: f64,
    pub copy_s: f64,
}

/// Precision per part (2026-11-13, directions of Luis Jake Gabriel III: every part switchable to
/// 8 or 16, all at once or one by one): `base` (the 0.6B of the drones, tentacles and second look),
/// `writer` (the 1.7B), `reranker`, `embedder`; `None` is no selection: the product serves the
/// part at `DEFAULT_BITS` (8 since 2026-11-24, the measured configuration).
/// An explicit 16 also turns half math off for the part (full precision); 8 and 4 bind its
/// linears at 8 or 4 bits. The selector governs the whole MicroModel, whatever its source of
/// parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Precision {
    pub base: Option<u8>,
    pub writer: Option<u8>,
    pub reranker: Option<u8>,
    pub embedder: Option<u8>,
}
impl Precision {
    /// From `--precision 8|16` or `--precision base=8,writer=16,...`.
    /// # Errors
    /// An unknown part or a width other than 8 or 16.
    pub fn parse(s: &str) -> Result<Self, String> {
        let width = |w: &str| match w {
            "4" => Ok(4_u8),
            "8" => Ok(8_u8),
            "16" => Ok(16_u8),
            o => Err(format!("--precision: {o} is not 4, 8 or 16")),
        };
        if !s.contains('=') {
            let w = Some(width(s)?);
            return Ok(Self { base: w, writer: w, reranker: w, embedder: w });
        }
        let mut p = Self::default();
        for kv in s.split(',').filter(|x| !x.is_empty()) {
            let (k, v) = kv.split_once('=').ok_or_else(|| format!("--precision: {kv} is not part=width"))?;
            let w = Some(width(v)?);
            match k {
                "base" => p.base = w,
                "writer" => p.writer = w,
                "reranker" => p.reranker = w,
                "embedder" => p.embedder = w,
                o => return Err(format!("--precision: unknown part {o} (base, writer, reranker, embedder)")),
            }
        }
        Ok(p)
    }
    /// The width asked for a population, if any.
    #[must_use]
    pub fn of(&self, pop_id: &str) -> Option<u8> {
        // 2026-11-25: the role a chosen model runs as (`nmtype::role_models`) sets its precision.
        match nmtype::role_models::role_of(pop_id) {
            Some("reranker") => return self.reranker,
            Some("embedder") => return self.embedder,
            Some("writer") => return self.writer,
            Some("reader") => return self.base,
            _ => {}
        }
        // Card ids may carry a prefix (`weight-opnwgt-basewt-qwen3-reranker-0.6b`): match the end.
        if pop_id.ends_with("qwen3-reranker-0.6b") {
            self.reranker
        } else if pop_id.ends_with("qwen3-embedding-0.6b") {
            self.embedder
        } else if pop_id.ends_with("qwen3-1.7b") {
            self.writer
        } else {
            self.base
        }
    }
}
/// Per-piece widths per part (`--precision-map <file>`, 2026-11-14): part (`base`, `writer`,
/// `reranker`, `embedder`) to piece (`default`, `L<n>`, `L<n>.attn`, `L<n>.ffn`) to 4, 8 or 16.
pub static PRECISION_MAP: std::sync::RwLock<Option<BTreeMap<String, BTreeMap<String, u8>>>> = std::sync::RwLock::new(None);

/// Read a precision map file (JSON object of part to piece to width).
/// # Errors
/// An unreadable file or a width other than 4, 8 or 16.
pub fn read_precision_map(path: &Path) -> Result<BTreeMap<String, BTreeMap<String, u8>>, String> {
    let v = nmjson::parse(&std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?).map_err(|e| e.to_string())?;
    let mut out = BTreeMap::new();
    for (part, pieces) in v.object().map_err(|e| e.to_string())? {
        let mut m = BTreeMap::new();
        for (k, w) in pieces.object().map_err(|e| e.to_string())? {
            let b = w.integer().map_err(|e| e.to_string())?;
            if !matches!(b, 4 | 8 | 16) {
                return Err(format!("--precision-map: {part}.{k} = {b}, not 4, 8 or 16"));
            }
            m.insert(k.clone(), u8::try_from(b).unwrap_or(16));
        }
        out.insert(part.clone(), m);
    }
    Ok(out)
}

/// The precision asked for this process (set once, before the first bind).
pub static PRECISION: std::sync::RwLock<Precision> =
    std::sync::RwLock::new(Precision { base: None, writer: None, reranker: None, embedder: None });

/// Apply the asked precision to an assembly before binding (`Model::bind` does, for every model):
/// 8 sets its linears to 8 bits, 16 to binary16 (over any short form); returns whether the model
/// must keep binary32 math (16 asked).
#[must_use]
pub fn apply_precision(asm: &mut Assembly, pop_id: &str) -> bool {
    // Per-piece mixes (2026-11-14, `--precision-map`): the part's default width, then the widths of
    // named pieces (`L<n>`, `L<n>.attn`, `L<n>.ffn`), over anything set above.
    if let Some(map) = PRECISION_MAP.read().ok().and_then(|m| m.clone()) {
        let part = ["qwen3-reranker-0.6b", "qwen3-embedding-0.6b", "qwen3-1.7b"].iter().find(|p| pop_id.ends_with(**p)).map_or(
            "base",
            |p| match *p {
                "qwen3-reranker-0.6b" => "reranker",
                "qwen3-embedding-0.6b" => "embedder",
                _ => "writer",
            },
        );
        if let Some(cfg) = map.get(part) {
            let bits = |n: &Node| -> Option<u8> {
                let short = n.id.rsplit('#').next().unwrap_or("");
                let (layer, piece) = short.split_once('.').unwrap_or((short, ""));
                let group = if matches!(piece, "q" | "k" | "v" | "o") { "attn" } else { "ffn" };
                [format!("{layer}.{group}"), layer.to_owned(), "default".to_owned()].iter().find_map(|k| cfg.get(k).copied())
            };
            let mut any16 = false;
            for n in asm.nodes.iter_mut().filter(|n| n.op == "linear") {
                match bits(n) {
                    Some(4) => "q4".clone_into(&mut n.precision),
                    Some(8) => "q8".clone_into(&mut n.precision),
                    Some(16) => {
                        "f16".clone_into(&mut n.precision);
                        any16 = true;
                    }
                    _ => {}
                }
            }
            return any16 && asm.nodes.iter().filter(|n| n.op == "linear").all(|n| n.precision == "f16");
        }
    }
    let asked = PRECISION.read().map(|p| p.of(pop_id)).unwrap_or(None);
    if asked == Some(4) {
        for n in asm.nodes.iter_mut().filter(|n| n.op == "linear") {
            "q4".clone_into(&mut n.precision);
        }
    } else if asked == Some(8) {
        for n in asm.nodes.iter_mut().filter(|n| n.op == "linear") {
            "q8".clone_into(&mut n.precision);
        }
    } else if asked == Some(16) {
        for n in asm.nodes.iter_mut().filter(|n| n.op == "linear") {
            "f16".clone_into(&mut n.precision);
        }
    }
    asked == Some(16)
}

/// The precision a served MicroModel's stored parameters take when none is selected for it
/// (applied where the product binds its MicroModels; an assembly bound directly keeps the widths
/// it carries). Roadmap Phase 4.6 (2026-11-24): 8 bits, the configuration every board was
/// measured at. Before, 8 bits came only from the GTX 1650's settings cartridge, so any other card
/// served at an unmeasured 16 bits (twice the card memory, so a small card could not seat the four).
/// 16 stays selectable per part (`--precision`, the precision controls); serving 16 bits on large
/// cards is in EVALUATION until a board measures it there.
pub const DEFAULT_BITS: u8 = 8;

/// Steering vectors by layer: the vector and its scale (`RunOpts::steer`, `Model::set_steer`).
pub type Steer = BTreeMap<usize, (Arc<Vec<f32>>, f32)>;

pub struct Model {
    pub pop: Arc<Population>,
    pub asm: Assembly,
    pub bound: Vec<Bound>,
    units: RwLock<BTreeMap<String, Arc<Adapter>>>,
    /// Cartridge filters of this model alone, in place of the rack's for its MicroModel.
    filters: RwLock<Option<Arc<crate::filter::Plugged>>>,
    /// Bound at 16 by request (`--precision`, 2026-11-13): its forwards never use half math.
    pub no_half: AtomicBool,
    /// Its role takes the integer path for 8-bit linears (`gpu::int8_dot_part`).
    pub int8_dot: AtomicBool,
    /// Steering vectors seated on this model (a distilled steer cartridge): used by every run that
    /// gives none of its own (`RunOpts::steer`). `steering` says whether any is seated, so a model
    /// without one pays one atomic load per run and nothing else.
    steer_seat: RwLock<Steer>,
    steering: AtomicBool,
}
/// Roadmap Phase 4.5 (2026-11-24): told when a forward of a MicroModel starts (`None`) and when it
/// ends (`Some(start)`), with the model's weight card id, for the live view (`nmapi::live`). Set
/// once by the serving process; unset it costs one load and a branch per forward.
pub type ExecHook = fn(&str, Option<std::time::Instant>);
/// The installed `ExecHook`, if any.
pub static EXEC_HOOK: std::sync::OnceLock<ExecHook> = std::sync::OnceLock::new();
fn exec_start(card: &str) -> Option<(ExecHook, std::time::Instant)> {
    EXEC_HOOK.get().map(|h| {
        h(card, None);
        (*h, std::time::Instant::now())
    })
}
fn exec_end(card: &str, live: Option<(ExecHook, std::time::Instant)>) {
    if let Some((h, t)) = live {
        h(card, Some(t));
    }
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model").field("assembly", &self.asm.name).field("nodes", &self.bound.len()).finish_non_exhaustive()
    }
}
impl Drop for Model {
    fn drop(&mut self) {
        for b in &self.bound {
            for v in &b.views {
                let _ = self.pop.prov.release(&v.cluster.id);
            }
        }
    }
}

/// A read over a window may not pass the last layer its chunks hold (2026-11-10: chunks
/// encoded through the drone heads' last layer only): such a layer would otherwise attend to
/// the new tokens alone, silently.
fn window_holds(opts: &RunOpts, n: &Node) -> Result<(), Fault> {
    match (&opts.window, n.layer) {
        (Some(w), Some(l)) if w.len > 0 && l >= w.layers() => Err(bad(format!("{}: the window holds layers 0..{} only", n.id, w.layers()))),
        _ => Ok(()),
    }
}

/// Why a `score` node over global query heads `h` and key-value heads `g` does not line up
/// with the shared key-value groups of a population with `heads` query and `kv` key-value
/// heads (`None`: it does). Aligned: a non-empty head range inside the population whose ends
/// fall on group boundaries, and exactly the key-value heads of those groups (2026-11-10,
/// step P0; for example heads 8..16 with key-value heads 4..8 in Qwen3-0.6B).
#[must_use]
pub fn kv_alignment(h: (usize, usize), g: (usize, usize), heads: usize, kv: usize) -> Option<String> {
    if kv == 0 || !heads.is_multiple_of(kv) {
        return Some(format!("{heads} query heads do not divide into {kv} key-value groups"));
    }
    let group = heads / kv;
    if h.0 >= h.1 || h.1 > heads || g.0 >= g.1 || g.1 > kv {
        return Some(format!("heads {h:?} or kv {g:?} empty or outside {heads} query and {kv} key-value heads"));
    }
    if !h.0.is_multiple_of(group) || !h.1.is_multiple_of(group) || g != (h.0 / group, h.1 / group) {
        return Some(format!(
            "heads {}..{} with kv {}..{} do not line up with the key-value groups ({group} query heads per key-value head): these heads need kv {}..{}{}",
            h.0,
            h.1,
            g.0,
            g.1,
            h.0 / group,
            h.1.div_ceil(group),
            if h.0.is_multiple_of(group) && h.1.is_multiple_of(group) {
                ""
            } else {
                ", and the head range must start and end on a group boundary"
            }
        ));
    }
    None
}

/// The last token's row of `x` (`width` values per token), L2-normalised (`lastpool`).
#[must_use]
pub fn last_pool(x: &[f32], width: usize) -> Vec<f32> {
    let n = x.len() / width.max(1);
    let mut v = x.get(n.saturating_sub(1) * width..n * width).map(<[f32]>::to_vec).unwrap_or_default();
    let norm = v.iter().map(|a| a * a).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|a| *a /= norm);
    }
    v
}

/// Softmax over the logits of `ids` at the last position (`labelsoftmax`): the probability of
/// each label, in order. `vocab` is the logits width per position.
#[must_use]
pub fn label_softmax(logits: &[f32], vocab: usize, ids: &[usize]) -> Vec<f32> {
    let n = logits.len() / vocab.max(1);
    let last = logits.get(n.saturating_sub(1) * vocab..).unwrap_or(&[]);
    let xs: Vec<f64> = ids.iter().map(|i| f64::from(last.get(*i).copied().unwrap_or(f32::MIN))).collect();
    let m = xs.iter().copied().fold(f64::MIN, f64::max);
    let es: Vec<f64> = xs.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = es.iter().sum();
    #[expect(clippy::cast_possible_truncation, reason = "probabilities")]
    es.iter().map(|e| (e / s) as f32).collect()
}

/// Add `scale * v` to every row of `y` (rows of `v.len()` values), rounded to binary16.
fn steer_rows(y: &mut [f32], v: &[f32], scale: f32) {
    for row in y.chunks_mut(v.len().max(1)) {
        for (x, d) in row.iter_mut().zip(v) {
            *x = crate::half::r16(*x + scale * d);
        }
    }
}

/// Adds a low-rank unit's delta to a norm node's `y`: `r16(y + delta(y))` per row, the unit's
/// small matrices on the host as on a linear's unit; on the card the rows come down, are shifted
/// and go up again in the same shape (one row a step while decoding).
fn lowrank_into(ops: Option<&Ops>, u: &Adapter, outs: &mut [(&str, Act)], tokens: usize) -> Result<(), String> {
    for (port, a) in outs.iter_mut() {
        if *port != "y" {
            continue;
        }
        let (mut y, shape) = match (&*a, ops) {
            (Act::Cpu(y), _) => (y.clone(), None),
            (Act::Gpu(t), Some(o)) => (o.download(t)?, Some(t.shape.clone())),
            _ => return Err(format!("{}: a low-rank unit reads host or card values", u.unit)),
        };
        let rows = y.len().checked_div(u.k).unwrap_or(tokens);
        let d = u.delta(&y, rows);
        for (x, dx) in y.iter_mut().zip(&d) {
            *x = crate::half::r16(*x + dx);
        }
        *a = match (shape, ops) {
            (Some(s), Some(o)) => Act::Gpu(o.upload_f32(&y, s)?),
            _ => Act::Cpu(y),
        };
    }
    Ok(())
}

/// Set the listed 64-channel blocks of every token row of `y` (`tokens` rows) to `zero`.
fn zero_blocks<T: Copy>(y: &mut [T], tokens: usize, blocks: &[usize], zero: T) {
    let width = y.len() / tokens.max(1);
    for row in y.chunks_mut(width.max(1)) {
        for &b in blocks {
            if let Some(s) = row.get_mut(b * 64..((b + 1) * 64).min(width)) {
                s.fill(zero);
            }
        }
    }
}

static HOST_EMBED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
/// Whether the token-table lookup is done on the host (default) or on the device.
#[must_use]
pub fn host_embed() -> bool {
    HOST_EMBED.load(std::sync::atomic::Ordering::Relaxed)
}
/// Switch the host token-table lookup on or off for this process (on by default; off gives the
/// device lookup over a whole-table upload, the earlier form, for comparison).
pub fn set_host_embed(on: bool) {
    HOST_EMBED.store(on, std::sync::atomic::Ordering::Relaxed);
}

impl Model {
    /// Bind an assembly: the port check, then every node acquires its clusters through NPCS
    /// (compatibility checked), then tensor widths are checked along every edge. Nothing runs
    /// before all three pass.
    /// # Errors
    /// The first failing check, named.
    pub fn bind(pop: &Arc<Population>, asm: Assembly) -> Result<Self, Fault> {
        let mut asm = asm;
        let full16 = apply_precision(&mut asm, &pop.card.id);
        asm.check_ports()?;
        let d = pop.dims;
        let want = |rows: Option<usize>, cols: Option<usize>| compat::Want {
            family: "qwen3".into(),
            dtypes: vec!["BF16".into(), "F16".into()],
            rows,
            cols,
        };
        let mut bound = Vec::with_capacity(asm.nodes.len());
        for n in &asm.nodes {
            let mut views = Vec::new();
            for r in &n.clusters {
                // Assemblies name tensor and region; the population being bound supplies the rest.
                let c = pop.cluster(&format!("{}/{r}", pop.card.id))?;
                let w = match n.op.as_str() {
                    "norm" => want(Some(1), Some(d.hidden)),
                    "headnorm" => want(Some(1), Some(d.head_dim)),
                    "embed" | "head" => want(Some(d.vocab), Some(d.hidden)),
                    _ => want(None, None),
                };
                compat::check_cluster(&c, &pop.card.family, &w)?;
                views.push(pop.prov.acquire(&c)?);
            }
            let (lin, vec) = match n.op.as_str() {
                "linear" | "head" | "embed" => {
                    (Some(Linear::new(views.clone(), n.stack == "cols").map_err(|e| bad(format!("{}: {e}", n.id)))?), None)
                }
                "norm" | "headnorm" => (
                    None,
                    match views.first() {
                        Some(v) => Some(cpu::vector_from(v, &pop.prov.bytes(v)?)),
                        None => None,
                    },
                ),
                _ => (None, None),
            };
            // Weight placement (2026-11-27): the map keeps no host weights of its own. Its 8-bit
            // packets are kept by the population while the host copies are resident (and made on
            // demand once they are dropped, `held`); a 4-bit map is rounded from the bytes read now.
            let lin = match lin {
                Some(mut l) if n.precision == "q4" => {
                    l.pins = views.iter().map(|v| pop.prov.bytes(v)).collect::<Result<Vec<_>, Fault>>()?;
                    let mut l = l.with_q4();
                    l.pins.clear();
                    Some(l)
                }
                Some(mut l) if n.precision == "q8" => {
                    l.int8 = true;
                    // Kept when they are at hand (prefetched, or made from a resident copy); a
                    // bind over host copies already dropped makes them on demand instead.
                    let mut kept = Vec::new();
                    for v in &views {
                        match pop.q8_held(v) {
                            Some(q) => kept.push((v.cluster.tensor.clone(), q)),
                            None if v.resident() => kept.push((v.cluster.tensor.clone(), pop.q8(v)?)),
                            None => {}
                        }
                    }
                    if let Ok(mut keep) = pop.q8_keep.lock() {
                        keep.extend(kept);
                    }
                    Some(l)
                }
                other => other,
            };
            let key = format!("{}:{}:{}", n.precision, n.stack, n.clusters.join(","));
            let label_ids = n
                .labels
                .iter()
                .map(|l| match pop.tokenizer.encode(l).map_err(|e| bad(e.to_string()))?.as_slice() {
                    [id] => usize::try_from(*id).map_err(|e| bad(e.to_string())),
                    _ => Err(bad(format!("{}: label {l:?} is not one token", n.id))),
                })
                .collect::<Result<Vec<_>, Fault>>()?;
            let kv = match n.op.as_str() {
                "score" => Some(n.kv.unwrap_or((0, d.kv_heads))),
                "rope" => asm
                    .edges
                    .iter()
                    .filter(|e| e.from == n.id && e.from_port == "k")
                    .find_map(|e| asm.node(&e.to).filter(|t| t.op == "score").map(|t| t.kv.unwrap_or((0, d.kv_heads)))),
                _ => None,
            };
            bound.push(Bound { node: n.clone(), views, lin, vec, label_ids, rows: Vec::new(), key, kv });
        }
        let m = Self { pop: Arc::clone(pop), asm, bound, units: RwLock::new(BTreeMap::new()), filters: RwLock::new(None), no_half: AtomicBool::new(full16), int8_dot: AtomicBool::new(crate::gpu::int8_dot_part(&pop.card.id)), steer_seat: RwLock::new(BTreeMap::new()), steering: AtomicBool::new(false) };
        m.check_widths()?;
        // Weight placement (2026-11-27): an 8-bit map reads only its packets, so the 16-bit host
        // copy of a tensor whose packets are kept and that no other node of this model reads at
        // 16 bits is dropped now (read again from its file range on demand).
        let sixteen: BTreeSet<&str> =
            m.bound.iter().filter(|b| !b.lin.as_ref().is_some_and(|l| l.int8)).flat_map(|b| b.views.iter().map(|v| v.cluster.tensor.as_str())).collect();
        let keep = pop.q8_keep.lock().map(|k| k.keys().cloned().collect::<BTreeSet<String>>()).unwrap_or_default();
        for b in m.bound.iter().filter(|b| b.node.op == "linear" && b.lin.as_ref().is_some_and(|l| l.int8)) {
            for v in b.views.iter().filter(|v| keep.contains(&v.cluster.tensor) && !sixteen.contains(v.cluster.tensor.as_str())) {
                pop.prov.drop_host(v);
            }
        }
        Ok(m)
    }
    /// Drop the device copies of full output heads (a head read whole, as the base check does):
    /// the jobs read their rows (`label_rows`, `RunOpts::head_rows`), as on the open weights. Off
    /// with the device token lookup, whose table may share the head's key. Returns bytes freed.
    pub fn release_full_heads(&self) -> usize {
        if !host_embed() {
            return 0;
        }
        self.bound.iter().filter(|b| b.node.op == "head" && b.rows.is_empty()).map(|b| self.pop.release_key(&b.key, &b.views)).sum()
    }
    /// A copy of `b`'s map holding its weights for one reader (2026-11-27, weight placement): the
    /// 8-bit packets of an 8-bit map (kept ones, or made again from the host copies read again
    /// from their files), else the blocks' tensor bytes. Dropped by the reader, it frees whatever
    /// was read again for it.
    fn held(&self, b: &Bound) -> Result<Linear, Fault> {
        let mut l = b.lin.clone().ok_or_else(|| bad(format!("{}: no weights", b.node.id)))?;
        if l.int8 {
            l.q8 = l.blocks.iter().map(|v| self.pop.q8(v)).collect::<Result<Vec<_>, Fault>>()?;
        } else if l.q4.is_none() {
            // One read per tensor: the blocks of one tensor share its bytes.
            let mut got: BTreeMap<String, Arc<[u8]>> = BTreeMap::new();
            let mut pins = Vec::with_capacity(l.blocks.len());
            for v in &l.blocks {
                let b = match got.get(&v.cluster.tensor) {
                    Some(b) => Arc::clone(b),
                    None => {
                        let b = self.pop.prov.bytes(v)?;
                        got.insert(v.cluster.tensor.clone(), Arc::clone(&b));
                        b
                    }
                };
                pins.push(b);
            }
            l.pins = pins;
        }
        Ok(l)
    }
    /// Whether one of `b`'s weight copies is held on the card (binary16, 8-bit, 8-bit bitmask or
    /// 4-bit form).
    fn on_card(&self, b: &Bound) -> bool {
        [b.key.clone(), format!("{}#v", b.key), format!("{}#bm_v", b.key), format!("{}#4v", b.key)].iter().any(|k| self.pop.holds_device(k))
    }
    /// Weight placement (2026-11-27, Luis Jake Gabriel III: "Not needlessly take up ram if it's on
    /// the GPU"): drop the host copies of every linear map whose weights the card holds (the
    /// tensor bytes and their 8-bit packets). The token table, the output head and the norms stay
    /// resident (the host token lookup and the head-row reads use them on every call). A path that
    /// needs a dropped tensor later (an upload after an eviction, a restore, a processor run) reads
    /// it again from its file range. Nothing is dropped without a card. Returns the RAM bytes freed.
    pub fn drop_host(&self) -> u64 {
        if self.pop.gpu.is_none() {
            return 0;
        }
        let before = self.pop.host_bytes();
        let mut n = 0;
        for b in self.bound.iter().filter(|b| b.node.op == "linear" && self.on_card(b)) {
            for v in &b.views {
                let bytes = self.pop.prov.drop_host(v) > 0;
                let packets = self.pop.q8_keep.lock().is_ok_and(|mut keep| keep.remove(&v.cluster.tensor).is_some());
                if bytes || packets {
                    n += 1;
                }
            }
        }
        let after = self.pop.host_bytes();
        let freed = (before.0 + before.1).saturating_sub(after.0 + after.1);
        if n > 0 {
            self.pop.set_host_note("the card holds them; read again from their files on demand (an upload after an eviction, a restore, a processor run)".to_owned());
            eprintln!(
                "engine: {}: host copies of {n} linear tensors dropped ({} MiB freed; {} MiB of weights stay in RAM: token table, head, norms)",
                self.pop.card.id,
                freed >> 20,
                (after.0 + after.1) >> 20
            );
        }
        freed
    }
    /// Keep the host copies of this model's linear maps resident again (read again from their
    /// files where dropped, with their 8-bit packets), for a processor run. They are dropped again
    /// at the next staging on the card.
    /// # Errors
    /// A file that cannot be read or changed since it was loaded.
    fn restore_host(&self) -> Result<(), Fault> {
        // What a processor run reads: an 8-bit map's packets, a 16-bit map's tensor bytes.
        let keep = self.pop.q8_keep.lock().map(|k| k.keys().cloned().collect::<BTreeSet<String>>()).unwrap_or_default();
        let ready = |b: &Bound| {
            if b.lin.as_ref().is_some_and(|l| l.int8) { b.views.iter().all(|v| keep.contains(&v.cluster.tensor)) } else { b.views.iter().all(|v| v.resident()) }
        };
        let lins: Vec<&Bound> = self.bound.iter().filter(|b| b.node.op == "linear" && b.lin.as_ref().is_some_and(|l| l.q4.is_none()) && !ready(b)).collect();
        if lins.is_empty() {
            return Ok(());
        }
        for b in lins.iter().filter(|b| !b.lin.as_ref().is_some_and(|l| l.int8)) {
            for v in &b.views {
                self.pop.prov.keep(v)?;
            }
        }
        let int8: Vec<Arc<View>> = lins.iter().filter(|b| b.lin.as_ref().is_some_and(|l| l.int8)).flat_map(|b| b.views.iter().cloned()).collect();
        // Quantized on the processor threads first; `q8` then finds each packet held.
        let _made = self.pop.q8_views(&int8, cpu::threads())?;
        let kept = int8.iter().map(|v| Ok((v.cluster.tensor.clone(), self.pop.q8(v)?))).collect::<Result<Vec<_>, Fault>>()?;
        if let Ok(mut keep) = self.pop.q8_keep.lock() {
            keep.extend(kept);
        }
        self.pop.set_host_note("a processor run needed them: read again from their files and kept; dropped again at the next staging on the card".to_owned());
        eprintln!("engine: {}: host copies read again and kept for a processor run", self.pop.card.id);
        Ok(())
    }
    /// Read only the label rows of the output head (2026-11-10, teardown patch 7): every `head`
    /// node whose logits go to exactly one consumer, a `labelsoftmax` node, computes the logits
    /// of that node's label tokens only (two rows of the released head for the reranker, instead
    /// of the whole vocabulary); the device holds a copy of those rows only. The released weights
    /// are read, never altered; the label probabilities are unchanged.
    #[must_use]
    pub fn label_rows(mut self) -> Self {
        let mut pairs = Vec::new();
        for (li, l) in self.bound.iter().enumerate().filter(|(_, b)| b.node.op == "labelsoftmax") {
            let into: Vec<&crate::assembly::Edge> = self.asm.edges.iter().filter(|e| e.to == l.node.id && e.to_port == "logits").collect();
            let [e] = into.as_slice() else { continue };
            let Some(hi) = self.bound.iter().position(|b| b.node.id == e.from && b.node.op == "head") else { continue };
            if self.asm.edges.iter().filter(|x| x.from == e.from).count() == 1 && !l.label_ids.is_empty() {
                pairs.push((hi, li, l.label_ids.clone()));
            }
        }
        for (hi, li, rows) in pairs {
            if let Some(h) = self.bound.get_mut(hi) {
                h.rows.clone_from(&rows);
            }
            if let Some(l) = self.bound.get_mut(li) {
                l.rows = rows;
            }
        }
        self
    }
    /// The full output head on every node (undoes `label_rows`; for comparison tests).
    #[must_use]
    pub fn full_head(mut self) -> Self {
        for b in &mut self.bound {
            b.rows.clear();
        }
        self
    }
    /// What this bound model holds (2026-11-10, tentacle bodies): the distinct parameter clusters
    /// its components acquired, their bytes on the host (views shared with every model of the
    /// population, counted once here), and the device bytes of its weight copies resident now
    /// (copies are keyed by cluster, so a copy another model of the population uploaded counts for
    /// both).
    #[must_use]
    pub fn holdings(&self) -> (usize, u64, usize) {
        let mut ids = BTreeSet::new();
        let mut host = 0_u64;
        for b in &self.bound {
            for v in &b.views {
                if ids.insert(v.cluster.id.clone()) {
                    host += v.cluster.bytes;
                }
            }
        }
        let keys: BTreeSet<String> = self
            .bound
            .iter()
            .filter(|b| !b.views.is_empty())
            .flat_map(|b| [b.key.clone(), format!("{}#v", b.key), format!("{}#s", b.key)])
            .collect();
        let dev = self.pop.dev.lock().map(|d| d.iter().filter(|(k, _)| keys.contains(*k)).map(|(_, t)| t.buf.bytes()).sum()).unwrap_or(0);
        (ids.len(), host, dev)
    }
    /// Every bound component: (id, operation, clusters held).
    #[must_use]
    pub fn components(&self) -> Vec<(String, String, Vec<String>)> {
        self.bound.iter().map(|b| (b.node.id.clone(), b.node.op.clone(), b.node.clusters.clone())).collect()
    }
    /// Load a unit on its node (replacing one already there) or unload it (`None`). The node
    /// must be a `linear` whose widths the unit's matrices fit.
    /// # Errors
    /// No such node, not a linear, or widths that do not fit.
    pub fn set_unit(&self, node: &str, unit: Option<Arc<Adapter>>) -> Result<(), Fault> {
        let b = self.bound.iter().find(|b| b.node.id == node).ok_or_else(|| bad(format!("no node {node}")))?;
        // The low-rank socket at a `norm` node (the Writer's distilled cartridges at `final_norm`):
        // `y += scale * up(down(y))` on the node's output after its cartridge filters, square at
        // the norm's width.
        if let Some(u) = unit.as_ref().filter(|_| b.node.op == "norm") {
            let width = b.vec.as_ref().map_or(0, Vec::len);
            if u.k != width || u.n != width {
                return Err(bad(format!("{node}: unit {} takes {}x{}, the norm is {width} wide", u.unit, u.n, u.k)));
            }
        } else if let Some(u) = &unit {
            let l = b
                .lin
                .as_ref()
                .filter(|_| b.node.op == "linear")
                .ok_or_else(|| bad(format!("{node}: units bind only to linear components")))?;
            if u.k != l.k || u.n != l.n {
                return Err(bad(format!("{node}: unit {} takes {}x{}, the component is {}x{}", u.unit, u.n, u.k, l.n, l.k)));
            }
        }
        let mut m = self.units.write().map_err(|_| bad("units lock"))?;
        match unit {
            Some(u) => {
                m.insert(node.to_owned(), u);
            }
            None => {
                m.remove(node);
            }
        }
        Ok(())
    }
    /// Gives this model its own cartridge filters (`Some`), used in place of the rack's for its
    /// MicroModel, or returns it to the rack (`None`).
    pub fn set_filters(&self, f: Option<Arc<crate::filter::Plugged>>) {
        if let Ok(mut g) = self.filters.write() {
            *g = f;
        }
    }
    /// The cartridge filters a run of this model applies now: its own, else the rack's.
    #[must_use]
    pub fn filters_in_use(&self) -> Option<Arc<crate::filter::Plugged>> {
        self.filters.read().ok().and_then(|g| g.clone()).or_else(|| crate::filter::for_pop(&self.pop.card.id))
    }
    /// The units loaded, as (node id, unit id).
    #[must_use]
    pub fn units_loaded(&self) -> Vec<(String, String)> {
        self.units.read().map(|m| m.iter().map(|(k, v)| (k.clone(), v.unit.clone())).collect()).unwrap_or_default()
    }
    /// Seats steering vectors on this model (`layer -> (vector, scale)`, each vector as wide as
    /// the model), used by every run that gives none of its own; an empty map takes them out, and
    /// runs are then exactly as without.
    /// # Errors
    /// A layer the model does not have or a vector of another width.
    pub fn set_steer(&self, steer: Steer) -> Result<(), Fault> {
        let d = self.pop.dims.hidden;
        if let Some((l, (v, _))) = steer.iter().find(|(l, (v, s))| **l >= self.pop.dims.layers || v.len() != d || !s.is_finite()) {
            return Err(bad(format!("steer at layer {l}: {} values (the model has {} layers of width {d})", v.len(), self.pop.dims.layers)));
        }
        let on = !steer.is_empty();
        let mut g = self.steer_seat.write().map_err(|_| bad("steer lock"))?;
        *g = steer;
        self.steering.store(on, Ordering::Relaxed);
        Ok(())
    }
    /// The steering vectors seated on this model (empty when none).
    #[must_use]
    pub fn steer_seated(&self) -> Steer {
        if !self.steering.load(Ordering::Relaxed) {
            return BTreeMap::new();
        }
        self.steer_seat.read().map(|g| g.clone()).unwrap_or_default()
    }
    /// The node id that plays `role` in this assembly (`final_norm`, `head`, `pool`, `label`,
    /// `learn_target`, ...): the one the assembly names under `names` in its file, else the
    /// stated default (2026-11-10, step P0). A role with no such node is refused by name, never
    /// answered by a missing target or a wrong node.
    /// # Errors
    /// The assembly has no node for the role.
    pub fn role_node(&self, role: &str) -> Result<String, Fault> {
        let id = self.asm.named(role);
        if self.asm.node(&id).is_none() {
            return Err(Fault::new(
                "nmengn.model",
                FaultKind::Interface,
                format!(
                    "assembly {} has no {role} node (looked for {id}); an assembly names its own under \"names\" in its file",
                    self.asm.name
                ),
            ));
        }
        Ok(id)
    }
    /// The key-value heads a `rope` node's keys hold: those of the `score` node it feeds, or
    /// every head when its keys are the full width (step P0).
    fn rope_kv(&self, b: &Bound, width: usize) -> Result<(usize, usize), Fault> {
        let d = self.pop.dims;
        match b.kv {
            Some(g) if (g.1 - g.0) * d.head_dim == width => Ok(g),
            None if width == d.kv_heads * d.head_dim => Ok((0, d.kv_heads)),
            _ => Err(bad(format!(
                "{}: its keys ({width} values per token) cannot be matched to key-value heads: a rope node whose keys are captured must feed one score node",
                b.node.id
            ))),
        }
    }
    fn unit_on(&self, node: &str) -> Option<Arc<Adapter>> {
        self.units.read().ok().and_then(|m| m.get(node).cloned())
    }
    fn width_of(&self, w: &BTreeMap<(String, String), usize>, node: &str, port: &str) -> Result<usize, Fault> {
        let (f, fp) = self.asm.producer(node, port).ok_or_else(|| bad(format!("{node}.{port} unfed")))?;
        w.get(&(f.to_owned(), fp.to_owned())).copied().ok_or_else(|| bad(format!("{f}.{fp} has no width")))
    }
    /// Width check along every edge (the second half of the port check).
    /// # Errors
    /// The first edge whose widths disagree.
    pub fn check_widths(&self) -> Result<(), Fault> {
        let d = self.pop.dims;
        let mut w: BTreeMap<(String, String), usize> = BTreeMap::new();
        let f = |n: &Node, m: String| Fault::new("nmengn.assembly", FaultKind::Interface, format!("{}: {m}", n.id));
        for b in &self.bound {
            let n = &b.node;
            let i = |p: &str| self.width_of(&w, &n.id, p);
            let out = match n.op.as_str() {
                "source" => 0,
                "embed" => d.hidden,
                "pass" => i("x")?,
                "norm" | "headnorm" => {
                    let (x, g) = (i("x")?, b.vec.as_ref().map_or(0, Vec::len));
                    if g == 0 || x % g != 0 || (n.op == "norm" && x != g) {
                        return Err(f(n, format!("input width {x} does not fit norm width {g}")));
                    }
                    x
                }
                "linear" => {
                    let (x, l) = (i("x")?, b.lin.as_ref().ok_or_else(|| f(n, "no weights".into()))?);
                    if x != l.k {
                        return Err(f(n, format!("input width {x}, weights take {}", l.k)));
                    }
                    l.n
                }
                "rope" => {
                    let (q, k) = (i("q")?, i("k")?);
                    w.insert((n.id.clone(), "k".into()), k);
                    q
                }
                "score" => {
                    let (h, g) = (n.heads.unwrap_or((0, d.heads)), n.kv.unwrap_or((0, d.kv_heads)));
                    // 2026-11-10 (step P0): the query heads must line up with the shared
                    // key-value groups (`heads / kv_heads` query heads per key-value head) and
                    // the key-value range must be exactly the groups those heads use. Both
                    // kernels map a node's local query head `j` to its local key-value head
                    // `j / group`; that is the right global head only for such a range.
                    if let Some(why) = kv_alignment(h, g, d.heads, d.kv_heads) {
                        return Err(f(n, why));
                    }
                    let (q, k, v) = (i("q")?, i("k")?, i("v")?);
                    if q != (h.1 - h.0) * d.head_dim || k != (g.1 - g.0) * d.head_dim || v != k {
                        return Err(f(n, format!("q {q}, k {k}, v {v} do not fit heads {h:?} and kv {g:?}")));
                    }
                    q
                }
                "swigate" | "add" => {
                    let (p0, p1) = if n.op == "add" { ("a", "b") } else { ("gate", "up") };
                    let (a, c) = (i(p0)?, i(p1)?);
                    if a != c {
                        return Err(f(n, format!("{p0} width {a} differs from {p1} width {c}")));
                    }
                    a
                }
                "head" => {
                    if i("x")? != d.hidden {
                        return Err(f(n, "input width".into()));
                    }
                    if b.rows.is_empty() { d.vocab } else { b.rows.len() }
                }
                // Output components (2026-11-09): the pooled vector keeps the hidden width; the
                // label probabilities are one per label.
                "lastpool" => {
                    if i("x")? != d.hidden {
                        return Err(f(n, "input width".into()));
                    }
                    d.hidden
                }
                "labelsoftmax" => {
                    let want = if b.rows.is_empty() { d.vocab } else { b.rows.len() };
                    if i("logits")? != want || b.label_ids.is_empty() || b.label_ids.iter().any(|x| *x >= d.vocab) {
                        return Err(f(n, "logits width or labels".into()));
                    }
                    b.label_ids.len()
                }
                _ => return Err(f(n, "unknown operation".into())),
            };
            let port = if n.op == "head" {
                "logits"
            } else if n.op == "rope" {
                "q"
            } else {
                "y"
            };
            w.insert((n.id.clone(), port.into()), out);
        }
        Ok(())
    }
    /// Device bytes this model's weights need (for placement).
    #[must_use]
    pub fn device_need(&self) -> u64 {
        let mut keys = BTreeSet::new();
        // Weight placement (2026-11-27): an 8-bit map holds one byte per weight on the card (its
        // binary16 row scales aside), half its 16-bit cluster bytes; counted at 16 bits, a model
        // that fits the card at 8 bits was placed on the processor.
        self.bound
            .iter()
            .filter(|b| keys.insert(b.key.clone()))
            .map(|b| {
                let bytes = b.views.iter().map(|v| v.cluster.bytes).sum::<u64>();
                if b.lin.as_ref().is_some_and(|l| l.int8) { bytes / 2 + b.lin.as_ref().map_or(0, |l| l.n as u64 * 2) } else { bytes }
            })
            .sum()
    }

    /// Run the nodes needed for `targets` from `seeds` on `device`. `tokens` is the number of
    /// positions in the activations.
    /// # Errors
    /// A missing input, a device failure or an unknown operation.
    pub fn run(
        &self,
        device: Device,
        seeds: Vec<(String, String, Act)>,
        targets: &[(String, String)],
        tokens: usize,
        opts: &RunOpts,
    ) -> Result<BTreeMap<(String, String), Act>, Fault> {
        // A steer seated on the model applies where the run gives none of its own; without one
        // the run's options are used as given.
        let merged;
        let opts = if opts.steer.is_empty() && self.steering.load(Ordering::Relaxed) {
            merged = RunOpts { steer: self.steer_seated(), ..opts.clone() };
            &merged
        } else {
            opts
        };
        let ops = match device {
            Device::Gpu => Some(self.pop.gpu.as_ref().ok_or_else(|| bad("GPU chosen but none is open"))?),
            Device::Cpu => {
                // Weight placement (2026-11-27): a processor run reads every weight row on the
                // host, so dropped host copies are read again once and kept (not per run).
                self.restore_host()?;
                None
            }
        };
        // Stage timing (2026-11-10, step P1; off unless a part test switches it on): each node
        // is a span named by its operation, with the device synchronised at both edges so its
        // queued kernels are charged to it; nothing here changes what is computed.
        let _run = nmffi::prof::Span::new("run");
        let prof_gpu = ops.filter(|_| nmffi::prof::on());
        let edge = |o: Option<&Ops>| {
            if let Some(o) = o {
                let _p = nmffi::prof::Span::new("sync.edge");
                let _ = o.gpu.sync();
            }
        };
        let mut acts: BTreeMap<(String, String), Act> = BTreeMap::new();
        let seeded: BTreeSet<String> = seeds.iter().map(|s| s.0.clone()).collect();
        let seed_span = nmffi::prof::Span::new("run.seeds");
        for (n, p, a) in seeds {
            let a = match (ops, a) {
                (Some(o), Act::Cpu(x)) => {
                    let width = x.len() / tokens.max(1);
                    Act::Gpu(o.upload_f32(&x, vec![tokens, width]).map_err(bad)?)
                }
                (_, a) => a,
            };
            acts.insert((n, p), a);
        }
        drop(seed_span);
        let ids: Vec<String> = targets.iter().map(|t| t.0.clone()).collect();
        let plan_span = nmffi::prof::Span::new("run.plan");
        let order = self.asm.needed(&ids, &seeded);
        // Step P2: each activation is released right after its last reader runs (targets are
        // kept), so its device buffer goes back to the pool for the next component instead of
        // staying held until the run ends. Nothing computed changes.
        let dies = last_readers(&self.asm, &self.bound, &order);
        let keep: BTreeSet<&(String, String)> = targets.iter().collect();
        let read: BTreeSet<&(String, String)> = dies.values().flatten().collect();
        let read: BTreeSet<(String, String)> = read.into_iter().cloned().collect();
        drop(plan_span);
        // Step P2: device bookkeeping of a GPU run (the population is in use, so the residency
        // manager does not release its weights meanwhile; the run's working peak is measured).
        let _active = ops.map(|o| crate::residency::RunMark::new(&self.pop, o));
        // Cartridge filters plugged on this MicroModel, composed with any opt-in hot-swappable
        // `NANO_JOB_FILTERS=on` job-path filters and any per-hop `opts.filters` (WP-A2 / Stage 6B:
        // zero global RACK mutation, zero cost when none are active).
        let auto_job = if !JOB_FILTERS_HELD_OFF.with(std::cell::Cell::get) && job_filters_auto_on(&self.pop.card.id) {
            crate::filter::load_enabled_job_filters(&self.pop.home, &self.pop.card.id, &self.pop.source).map_err(bad)?
        } else {
            None
        };
        let mut filters = self.filters_in_use();
        for extra in [auto_job, opts.filters.clone()].into_iter().flatten() {
            filters = Some(match filters {
                None => extra,
                Some(base_f) => crate::filter::stack_plugged_cached(&base_f, &extra).map_err(bad)?,
            });
        }
        // WP-A1 X3: dead prefill last-layer row pruning. When a GPU run only reads last-token
        // outputs (`head`, `labelsoftmax`, `lastpool`, or `final_norm` with `opts.last_rows_only`)
        // and no cache/probe/capture/lesion needs non-final rows, gather only the M last-token
        // rows right after the final attention `score` node so the final layer's o_proj, FFN,
        // and final_norm run on M rows instead of all `tokens`.
        let gather_plan = ops.and_then(|_| self.last_layer_gather_plan(&order, targets, tokens, opts));
        let mut active_tokens = tokens;
        for (k, i) in order.into_iter().enumerate() {
            let b = self.bound.get(i).ok_or_else(|| bad("node index"))?;
            let n = &b.node;
            edge(prof_gpu);
            let node_span = nmffi::prof::Span::new(op_span(&n.op));
            let get = |p: &str| -> Result<&Act, Fault> {
                let (f, fp) = self.asm.producer(&n.id, p).ok_or_else(|| bad(format!("{}.{p} unfed", n.id)))?;
                acts.get(&(f.to_owned(), fp.to_owned())).ok_or_else(|| bad(format!("{}.{p}: {f}.{fp} not computed", n.id)))
            };
            // Parameter-level job usage recording (`pmap`, off by default; zero cost when off).
            if crate::pmap::on(&self.pop.card.id) {
                match n.op.as_str() {
                    "embed" => {
                        if let Ok(Act::Ids(ids)) = get("ids") {
                            crate::pmap::note_tokens(&self.pop.card.id, ids, false);
                        }
                    }
                    "linear" => {
                        if let Some((l, stream)) = crate::pmap::Stream::from_node_id(&n.id)
                            && let Ok(act) = get("x")
                        {
                            match (act, ops) {
                                (Act::Cpu(x), _) => {
                                    let w = x.len() / active_tokens.max(1);
                                    crate::pmap::note_input(&self.pop.card.id, l, stream, w, x);
                                }
                                (Act::Gpu(x), Some(o)) => {
                                    if let Ok(h) = o.download(x) {
                                        let w = x.shape.last().copied().unwrap_or(0);
                                        crate::pmap::note_input(&self.pop.card.id, l, stream, w, &h);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    "head" => {
                        if !opts.head_rows.is_empty() {
                            let ids: Vec<i32> = opts.head_rows.iter().filter_map(|r| i32::try_from(*r).ok()).collect();
                            crate::pmap::note_tokens(&self.pop.card.id, &ids, true);
                        } else if !b.rows.is_empty() {
                            let ids: Vec<i32> = b.rows.iter().filter_map(|r| i32::try_from(*r).ok()).collect();
                            crate::pmap::note_tokens(&self.pop.card.id, &ids, true);
                        }
                    }
                    _ => {}
                }
            }
            let mut outs: Vec<(&str, Act)> = match ops {
                None => self.cpu_node(b, &get, active_tokens, opts)?,
                Some(o) => self.gpu_node(o, b, &get, active_tokens, opts)?,
            };
            if let Some(f) = &filters {
                crate::filter::apply(f, ops, &n.id, &mut outs).map_err(bad)?;
            }
            // A low-rank unit at a norm node (`set_unit`), after the filters, so what serves is what
            // the unit was fitted on: the filtered output plus the unit's delta of it.
            if n.op == "norm"
                && let Some(u) = self.unit_on(&n.id)
            {
                lowrank_into(ops, &u, &mut outs, active_tokens).map_err(bad)?;
            }
            // Channel recording for a map from the real jobs (`chanrec`, off by default).
            if n.op == "swigate"
                && let Some(l) = n.layer
                && crate::chanrec::on(&self.pop.card.id)
            {
                for (_, a) in outs.iter().filter(|(p, _)| *p == "y") {
                    match (a, ops) {
                        (Act::Cpu(y), _) => crate::chanrec::note(l, self.pop.dims.inter, y),
                        (Act::Gpu(y), Some(o)) => {
                            if let Ok(h) = o.download(y) {
                                crate::chanrec::note(l, y.shape.last().copied().unwrap_or(0), &h);
                            }
                        }
                        _ => {}
                    }
                }
            }
            edge(prof_gpu);
            if nmffi::prof::on() {
                nmffi::prof::add_side(&n.layer.map_or_else(|| "layer -".to_owned(), |l| format!("layer {l:02}")), node_span.elapsed());
            }
            drop(node_span);
            for (p, a) in outs {
                let key = (n.id.clone(), p.to_owned());
                // An output nothing reads and no target names is not kept (step P2).
                if keep.contains(&key) || read.contains(&key) {
                    acts.insert(key, a);
                }
            }
            if let Some(gone) = dies.get(&k) {
                for key in gone.iter().filter(|x| !keep.contains(x)) {
                    acts.remove(key);
                }
                crate::gpu::release_q8_blocks_if_dead();
            }
            if let (Some(o), Some(plan)) = (ops, &gather_plan)
                && k == plan.score_pos
                && plan.crossing.iter().all(|key| {
                    matches!(acts.get(key), Some(Act::Gpu(dev)) if dev.shape.first().copied() == Some(tokens))
                })
            {
                for key in &plan.crossing {
                    if let Some(Act::Gpu(dev)) = acts.get(key).cloned() {
                        let gathered = gather_gpu_rows(o, &dev, &plan.rows).map_err(bad)?;
                        acts.insert(key.clone(), Act::Gpu(gathered));
                    }
                }
                crate::gpu::release_q8_blocks_if_dead();
                active_tokens = plan.rows.len();
            }
        }
        let target_span = nmffi::prof::Span::new("run.targets");
        let mut out = BTreeMap::new();
        for t in targets {
            let a = acts.remove(t).ok_or_else(|| bad(format!("target {}.{} not produced", t.0, t.1)))?;
            let a = match (ops, a) {
                (Some(o), Act::Gpu(x)) if !opts.device_targets => Act::Cpu(o.download(&x).map_err(bad)?),
                (_, a) => a,
            };
            out.insert(t.clone(), a);
        }
        drop(target_span);
        let drop_span = nmffi::prof::Span::new("run.drop");
        crate::gpu::clear_run_holds();
        drop(acts);
        drop(drop_span);
        Ok(out)
    }

    fn heads_off(&self, n: &Node, opts: &RunOpts) -> Vec<bool> {
        let h = n.heads.unwrap_or((0, self.pop.dims.heads));
        let off = n.layer.and_then(|l| opts.heads_off.get(&l)).cloned().unwrap_or_default();
        (h.0..h.1).map(|g| off.contains(&g)).collect()
    }

    fn cpu_node<'a>(
        &self,
        b: &Bound,
        get: &dyn Fn(&str) -> Result<&'a Act, Fault>,
        tokens: usize,
        opts: &RunOpts,
    ) -> Result<Vec<(&'static str, Act)>, Fault> {
        let d = self.pop.dims;
        let f = |p: &str| -> Result<&'a [f32], Fault> {
            match get(p)? {
                Act::Cpu(x) => Ok(x.as_slice()),
                _ => Err(bad(format!("{}.{p}: not a CPU activation", b.node.id))),
            }
        };
        let ints = |p: &str| -> Result<&'a [i32], Fault> {
            match get(p)? {
                Act::Ids(x) => Ok(x.as_slice()),
                _ => Err(bad(format!("{}.{p}: not integers", b.node.id))),
            }
        };
        let n = &b.node;
        Ok(match n.op.as_str() {
            "source" => return Err(bad(format!("{}: source not seeded", n.id))),
            "pass" => vec![("y", Act::Cpu(f("x")?.to_vec()))],
            "embed" => {
                let v = b.views.first().ok_or_else(|| bad("embed table"))?;
                vec![("y", Act::Cpu(cpu::embed_from(v, &self.pop.prov.bytes(v)?, ints("ids")?)))]
            }
            "norm" | "headnorm" => vec![("y", Act::Cpu(cpu::rmsnorm(f("x")?, b.vec.as_deref().unwrap_or(&[]), d.eps)))],
            "linear" => {
                let x = f("x")?;
                let mut y = cpu::linear(x, tokens, &self.held(b)?);
                if let Some(u) = self.unit_on(&n.id) {
                    y = cpu::add(&y, &u.delta(x, tokens));
                }
                vec![("y", Act::Cpu(y))]
            }
            "rope" => {
                let (mut q, mut k) = (f("q")?.to_vec(), f("k")?.to_vec());
                if let (Some(c), Some(l)) = (&opts.capture, n.layer) {
                    c.keys_at(l, self.rope_kv(b, k.len() / tokens.max(1))?, d.kv_heads, d.head_dim, &k);
                }
                let pos = ints("pos")?;
                let hd = d.head_dim;
                let (hq, hk) = (q.len() / (tokens * hd).max(1), k.len() / (tokens * hd).max(1));
                cpu::rope(&mut q, hq, hd, pos, d.theta);
                cpu::rope(&mut k, hk, hd, pos, d.theta);
                if let Some(f) = opts.q_scale {
                    q.iter_mut().for_each(|x| *x = crate::half::r16(*x * f));
                }
                vec![("q", Act::Cpu(q)), ("k", Act::Cpu(k))]
            }
            "score" => {
                if opts.segments.is_some() {
                    return Err(bad(format!("{}: packed sequences run on the GPU only", n.id)));
                }
                let (h, g) = (n.heads.unwrap_or((0, d.heads)), n.kv.unwrap_or((0, d.kv_heads)));
                let (k, v, q) = (f("k")?, f("v")?, f("q")?);
                let width = (g.1 - g.0) * d.head_dim;
                if let (Some(c), Some(l)) = (&opts.capture, n.layer) {
                    c.values_at(l, g, d.kv_heads, d.head_dim, v);
                }
                window_holds(opts, n)?;
                // Cached rows: the shared window first, then the decode cache's earlier steps.
                // A window row holds every key-value head of its layer; this node reads its own
                // heads' columns of it (2026-11-10, step P0).
                let win = opts.window.as_ref().zip(n.layer).map(|(w, l)| (w.layer_k(l), w.layer_v(l)));
                let (win_width, win_off) = (d.kv_heads * d.head_dim, g.0 * d.head_dim);
                let prev = match &opts.cache {
                    Some(c) => c.entries.lock().map_err(|_| bad("cache lock"))?.get(&n.id).cloned(),
                    None => None,
                };
                let (pk, pv): (&[u16], &[u16]) = match &prev {
                    Some(Kv::Cpu(a, b)) => (a, b),
                    _ => (&[], &[]),
                };
                let mut ck: Vec<&[u16]> = Vec::new();
                let mut cv: Vec<&[u16]> = Vec::new();
                if let Some((wk, wv)) = win {
                    ck.extend(wk);
                    cv.extend(wv);
                }
                // The window's key segments come first and hold keys before rope (E3.1).
                let pre = ck.len();
                ck.push(pk);
                cv.push(pv);
                let table = opts.window.as_ref().map_or(&[][..], |w| w.table());
                let kr = cpu::Rows { cached: ck, new: k, width, pre, table, head: d.head_dim, win: pre, win_width, win_off };
                let vr = cpu::Rows { cached: cv, new: v, width, win: pre, win_width, win_off, ..cpu::Rows::default() };
                if let (Some(p), Some(l), Some(w)) = (&opts.probe, n.layer, &opts.window)
                    && p.layers.contains(&l)
                    && p.accepts(&n.id)
                {
                    // The probe's head numbers are global; this node's are local (step P0).
                    let only = p.local_heads(l, h);
                    // Tentacle core A5: the raw scores too when the probe asks (same mass).
                    let (mass, lg) =
                        cpu::attention_read(q, &kr, tokens, h.1 - h.0, g.1 - g.0, d.head_dim, w.len, only.as_deref(), p.want_logits);
                    p.record(&n.id, l, h, w.len, mass);
                    if p.want_logits {
                        p.record_logits(&n.id, l, h, w.len, tokens, lg);
                    }
                }
                let y = cpu::attention_rows(q, &kr, &vr, tokens, h.1 - h.0, g.1 - g.0, d.head_dim, &self.heads_off(n, opts));
                if let Some(c) = &opts.cache {
                    let nk: Vec<u16> = pk.iter().copied().chain(k.iter().map(|x| crate::half::bits16(*x))).collect();
                    let nv: Vec<u16> = pv.iter().copied().chain(v.iter().map(|x| crate::half::bits16(*x))).collect();
                    c.entries.lock().map_err(|_| bad("cache lock"))?.insert(n.id.clone(), Kv::Cpu(nk, nv));
                }
                vec![("y", Act::Cpu(y))]
            }
            "swigate" => {
                let mut y = cpu::swigate(f("gate")?, f("up")?);
                if let Some(off) = n.layer.and_then(|l| opts.blocks_off.get(&l)) {
                    zero_blocks(&mut y, tokens, off, 0.0);
                }
                vec![("y", Act::Cpu(y))]
            }
            "add" => {
                let mut y = cpu::add(f("a")?, f("b")?);
                if let Some((v, s)) = n.layer.filter(|_| n.id.ends_with(".res2")).and_then(|l| opts.steer.get(&l)) {
                    steer_rows(&mut y, v, *s);
                }
                vec![("y", Act::Cpu(y))]
            }
            "head" => {
                let x = f("x")?;
                let held = self.held(b)?;
                let lin = &held;
                let (x, t) = if opts.all_positions { (x, tokens) } else { (x.get((tokens - 1) * d.hidden..).unwrap_or(&[]), 1) };
                if b.rows.is_empty() && !opts.head_rows.is_empty() {
                    // The asked rows out of the full product, so each value equals the full head's.
                    let full = cpu::linear(x, t, lin);
                    let y: Vec<f32> = full
                        .chunks_exact(d.vocab.max(1))
                        .flat_map(|row| opts.head_rows.iter().map(|r| row.get(*r).copied().unwrap_or(0.0)))
                        .collect();
                    vec![("logits", Act::Cpu(y))]
                } else if b.rows.is_empty() {
                    vec![("logits", Act::Cpu(cpu::linear(x, t, lin)))]
                } else {
                    // Label rows only (teardown patch 7): one dot product per position and row.
                    let mut w = vec![0.0_f32; lin.k];
                    let mut y = vec![0.0_f32; t * b.rows.len()];
                    for (j, r) in b.rows.iter().enumerate() {
                        lin.row(*r, &mut w);
                        for (ti, xr) in x.chunks_exact(d.hidden).take(t).enumerate() {
                            if let Some(o) = y.get_mut(ti * b.rows.len() + j) {
                                *o = cpu::dot(xr, &w);
                            }
                        }
                    }
                    vec![("logits", Act::Cpu(y))]
                }
            }
            "lastpool" => match &opts.segments {
                // One pooled vector per packed sequence, side by side.
                Some(segs) => {
                    let x = f("x")?;
                    let mut out = Vec::with_capacity(segs.len() * d.hidden);
                    let mut end = 0;
                    for n in segs.iter() {
                        end += n;
                        out.extend(last_pool(x.get(end.saturating_sub(1) * d.hidden..end * d.hidden).unwrap_or(&[]), d.hidden));
                    }
                    vec![("y", Act::Cpu(out))]
                }
                None => vec![("y", Act::Cpu(last_pool(f("x")?, d.hidden)))],
            },
            "labelsoftmax" => vec![("y", Act::Cpu(label_softmax(f("logits")?, b.label_width(d.vocab), &b.label_read())))],
            other => return Err(bad(format!("{}: unknown operation {other}", n.id))),
        })
    }

    /// Automatic `bitmask-q8` GPU compression for sparse 8-bit linears (`>= 20%` exact zeros,
    /// `NANO_BMQ8=off` disables): caches its 5 compressed buffers in `Population::dev` so residency
    /// tracks the real compressed VRAM footprint and dense layers fall back to `weight_q8` in O(1).
    fn weight_bmq8(&self, o: &Ops, b: &Bound) -> Result<Option<crate::gpu::DevBitmaskQ8>, Fault> {
        if std::env::var("NANO_BMQ8").is_ok_and(|v| v == "off") {
            return Ok(None);
        }
        let l = b.lin.as_ref().ok_or_else(|| bad("no weights"))?;
        let (km, kg, kr, kv, ks) = (
            format!("{}#bm_m", b.key),
            format!("{}#bm_g", b.key),
            format!("{}#bm_r", b.key),
            format!("{}#bm_v", b.key),
            format!("{}#bm_s", b.key),
        );
        if let Ok(dev) = self.pop.dev.lock() {
            if dev.contains_key(&format!("{}#v", b.key)) {
                return Ok(None);
            }
            if let (Some(m), Some(g), Some(r), Some(v), Some(s)) = (dev.get(&km), dev.get(&kg), dev.get(&kr), dev.get(&kv), dev.get(&ks)) {
                return Ok(Some(crate::gpu::DevBitmaskQ8 {
                    rows: l.n,
                    cols: l.k,
                    groups_per_row: l.k.div_ceil(64),
                    mask_words: m.clone(),
                    group_offs: g.clone(),
                    row_offs: r.clone(),
                    packed_i8: v.clone(),
                    row_scales: s.clone(),
                }));
            }
        }
        let (vals, scales) = self.held(b)?.packed_q8();
        let Some(bm) = crate::pmap::GpuBitmaskQ8::from_dense_q8(l.n, l.k, &vals, &scales, 20) else {
            return Ok(None);
        };
        let uploaded = {
            let _k = nmffi::KeptScope::new();
            o.upload_bmq8(&bm).map_err(|e| bad(format!("{}: {e}", b.node.id)))?
        };
        let m = self.pop.device(&km, &[], || Ok(uploaded.mask_words.clone()))?;
        let g = self.pop.device(&kg, &[], || Ok(uploaded.group_offs.clone()))?;
        let r = self.pop.device(&kr, &[], || Ok(uploaded.row_offs.clone()))?;
        let v = self.pop.device(&kv, &b.views, || Ok(uploaded.packed_i8.clone()))?;
        let s = self.pop.device(&ks, &[], || Ok(uploaded.row_scales.clone()))?;
        Ok(Some(crate::gpu::DevBitmaskQ8 {
            rows: bm.rows,
            cols: bm.cols,
            groups_per_row: bm.groups_per_row,
            mask_words: m,
            group_offs: g,
            row_offs: r,
            packed_i8: v,
            row_scales: s,
        }))
    }
    /// Int8 weights and their row scales on the device (a q8 linear).
    fn weight_q8(&self, o: &Ops, b: &Bound) -> Result<(DevT, DevT), Fault> {
        let (kv, ks) = (format!("{}#v", b.key), format!("{}#s", b.key));
        // Weight placement (2026-11-27): the 8-bit packets are held for the uploads only when one
        // of the two copies is missing (made again from the files when the host copy was dropped).
        let l = if self.pop.holds_device(&kv) && self.pop.holds_device(&ks) { b.lin.clone().ok_or_else(|| bad("no weights"))? } else { self.held(b)? };
        // Values and scales are two device copies under two keys; each upload runs once.
        let v = self.pop.device(&kv, &b.views, || {
            let vt = o.alloc(vec![l.n, l.k], 4)?;
            o.gpu.upload(&vt.buf, &l.packed_q8().0)?;
            Ok(vt)
        })?;
        let s = self.pop.device(&ks, &[], || o.upload16(&l.scales16(), vec![l.n]))?;
        Ok((v, s))
    }
    fn weight_q4(&self, o: &Ops, b: &Bound) -> Result<(DevT, DevT), Fault> {
        let l = b.lin.as_ref().ok_or_else(|| bad("no weights"))?;
        let q = l.q4.as_ref().ok_or_else(|| bad("no 4-bit weights"))?;
        let v = self.pop.device(&format!("{}#4v", b.key), &b.views, || {
            let vt = o.alloc(vec![l.n, l.k / 2], 4)?;
            o.gpu.upload(&vt.buf, &q.packed)?;
            Ok(vt)
        })?;
        let s = self.pop.device(&format!("{}#4s", b.key), &[], || {
            let bytes: Vec<u8> = q.scales.iter().flat_map(|x| x.to_le_bytes()).collect();
            o.upload16(&bytes, vec![l.n, l.k / crate::cpu::Q4_GROUP])
        })?;
        Ok((v, s))
    }
    fn weight(&self, o: &Ops, b: &Bound) -> Result<DevT, Fault> {
        self.pop.device(&b.key, &b.views, || match (&b.lin, &b.vec) {
            (Some(_), _) => {
                let l = self.held(b).map_err(|e| e.to_string())?;
                o.upload16(&l.packed16(), vec![l.n, l.k])
            }
            (None, Some(v)) => o.upload_f32(v, vec![v.len()]),
            _ => Err("no weights to upload".into()),
        })
    }

    /// Stage this model's large weights on the card through the copy stream `stream` (2026-11-13,
    /// cartridge loading step L, second part): every linear (8-bit values, or binary16), the token
    /// table and a full output head not yet held are allocated first, the default stream is
    /// synchronised once (fresh allocations are zeroed there), then copied in chunks of
    /// `STAGE_CHUNK` through two pinned buffers: one fills on the host while the other copies.
    /// The copies land in the same device cache the forward reads, so a job bound to this model
    /// finds them held; another model's job can run on the default stream meanwhile. Small
    /// tensors (norms, 8-bit scales) stay on the ordinary path.
    /// # Errors
    /// No GPU, device memory, or a copy fault.
    pub fn stage(&self, stream: &nmffi::cuda::CopyStream) -> Result<Staged, Fault> {
        self.stage_parts(stream, true)
    }
    /// As `stage`; with `full_head` false a full output head is left off the card (a job that
    /// reads only some of its rows, through `RunOpts::head_rows`, uploads those rows alone, and
    /// a job that reads the whole head uploads it on first use). The token table is staged only
    /// when the lookup runs on the device (`host_embed` off): the host lookup never reads a
    /// device copy.
    /// # Errors
    /// As `stage`.
    pub fn stage_parts(&self, stream: &nmffi::cuda::CopyStream, full_head: bool) -> Result<Staged, Fault> {
        let o = self.pop.gpu.as_ref().ok_or_else(|| bad("stage: no GPU"))?;
        let t0 = std::time::Instant::now();
        let wanted = |b: &Bound| {
            b.node.op == "linear" || (b.node.op == "embed" && !host_embed()) || (b.node.op == "head" && b.rows.is_empty() && full_head)
        };
        // Jobs: key, bound node, shape, dtype, views, 8-bit. Weight placement (2026-11-27): each
        // job's host bytes are made when its copy starts (one tensor's bytes at a time), not all
        // before the first copy.
        let mut jobs: Vec<StageJob> = Vec::new();
        for (i, b) in self.bound.iter().enumerate() {
            let Some(l) = b.lin.as_ref() else { continue };
            if !wanted(b) {
                continue;
            }
            if b.node.op == "linear" && l.q4.is_some() {
                // 4-bit maps are uploaded packed on first use, not staged as binary16.
                continue;
            }
            if b.node.op == "linear" && l.int8 {
                let key = format!("{}#v", b.key);
                let bm_key = format!("{}#bm_v", b.key);
                // Do not stage a duplicate dense copy when this sparse linear is already resident in `bitmask-q8` form.
                if !self.pop.holds_device(&key) && !self.pop.holds_device(&bm_key) && !jobs.iter().any(|j| j.0 == key) {
                    jobs.push((key, i, vec![l.n, l.k], 4, b.views.clone(), true));
                }
            } else if !self.pop.holds_device(&b.key) && !jobs.iter().any(|j| j.0 == b.key) {
                jobs.push((b.key.clone(), i, vec![l.n, l.k], crate::gpu::F16, b.views.clone(), false));
            }
        }
        // Bytes per job: one per element at 8 bits, two at binary16.
        let need: usize = jobs.iter().map(|j| j.2.iter().product::<usize>() * if j.5 { 1 } else { 2 }).sum();
        crate::residency::get().make_room(need, &self.pop.card.id);
        let dsts: Vec<DevT> = {
            let _k = nmffi::KeptScope::new();
            jobs.iter().map(|j| o.alloc(j.2.clone(), j.3)).collect::<Result<_, _>>().map_err(bad)?
        };
        o.gpu.sync().map_err(bad)?;
        let chunk = STAGE_CHUNK.min(need.max(1));
        let mut pins = [o.gpu.pinned(chunk).map_err(bad)?, o.gpu.pinned(chunk).map_err(bad)?];
        let t1 = std::time::Instant::now();
        let mut host_s = t1.duration_since(t0).as_secs_f64();
        let mut which = 0_usize;
        // The 8-bit packets of the batch being copied (made on the processor threads from the files
        // when not at hand; at most `STAGE_BATCH` elements' worth at a time).
        let mut batch: Vec<Arc<cpu::Q8>> = Vec::new();
        for (j, ((_, i, _, _, _, int8), dst)) in jobs.iter().zip(&dsts).enumerate() {
            let made_at = std::time::Instant::now();
            let b = self.bound.get(*i).ok_or_else(|| bad("stage: node"))?;
            if *int8 && b.views.iter().any(|v| self.pop.q8_held(v).is_none()) {
                drop(std::mem::take(&mut batch));
                let (mut views, mut size) = (Vec::new(), 0_usize);
                for (_, k, shape, _, _, _) in jobs.iter().skip(j).filter(|x| x.5) {
                    if size >= STAGE_BATCH {
                        break;
                    }
                    for v in self.bound.get(*k).map(|nb| nb.views.as_slice()).unwrap_or(&[]) {
                        if self.pop.q8_held(v).is_none() {
                            views.push(Arc::clone(v));
                        }
                    }
                    size += shape.iter().product::<usize>();
                }
                batch = self.pop.q8_views(&views, cpu::threads())?;
            }
            let bytes = if *int8 { self.held(b)?.packed_q8().0 } else { self.held(b)?.packed16() };
            host_s += made_at.elapsed().as_secs_f64();
            let mut at = 0;
            while at < bytes.len() {
                let n = chunk.min(bytes.len() - at);
                let p = &mut pins[which];
                // Fill this buffer while the other one's copy runs; then wait for that copy (the
                // stream runs copies in order, so after the wait this buffer's last copy is done too).
                p.bytes_mut()[..n].copy_from_slice(&bytes[at..at + n]);
                stream.wait().map_err(bad)?;
                o.gpu.upload_async(stream, &dst.buf, at, &pins[which], n).map_err(bad)?;
                at += n;
                which ^= 1;
            }
            // The stream still reads the pinned buffers, never `bytes`: they go now.
            drop(bytes);
        }
        stream.wait().map_err(bad)?;
        drop(batch);
        let copy_s = t1.elapsed().as_secs_f64();
        let mut held = 0;
        for ((key, _, _, _, views, _), dst) in jobs.into_iter().zip(dsts) {
            held += usize::from(self.pop.device_staged(&key, &views, dst, copy_s / f64::from(u32::try_from(held + 1).unwrap_or(1))));
        }
        // The card holds the linear maps now: their host copies go (`drop_host`).
        self.drop_host();
        Ok(Staged { tensors: held, bytes: need as u64, host_s, copy_s })
    }

    fn gpu_node<'a>(
        &self,
        o: &Ops,
        b: &Bound,
        get: &dyn Fn(&str) -> Result<&'a Act, Fault>,
        tokens: usize,
        opts: &RunOpts,
    ) -> Result<Vec<(&'static str, Act)>, Fault> {
        let d = self.pop.dims;
        let e = |m: String| bad(format!("{}: {m}", b.node.id));
        let t = |p: &str| -> Result<DevT, Fault> {
            match get(p)? {
                Act::Gpu(x) => Ok(x.clone()),
                // Lane S (2026-11-18): uploaded once per run, not once per reading node.
                Act::Ids(v) => o.ids_cached(v).map_err(bad),
                Act::Cpu(_) => Err(bad(format!("{}.{p}: CPU activation on the GPU path", b.node.id))),
            }
        };
        let n = &b.node;
        Ok(match n.op.as_str() {
            "source" => return Err(e("source not seeded".into())),
            "pass" => vec![("y", Act::Gpu(t("x")?))],
            "embed" => {
                // Parameter loader step 2 (2026-11-10): the token table stays on the host. The
                // rows of this call's ids are gathered there and only they are uploaded (the same
                // binary16 bytes the device kernel would copy out of a whole-table upload), so the
                // device holds no copy of the table for a lookup (311 MB per 0.6B population).
                // `set_host_embed(false)` gives the device lookup back, for comparison.
                match (get("ids")?, host_embed(), b.lin.as_ref()) {
                    (Act::Ids(v), true, Some(_)) => {
                        let rows: Vec<usize> = v.iter().map(|i| usize::try_from(*i).unwrap_or(usize::MAX)).collect();
                        let l = self.held(b)?;
                        let bytes = l.packed16_rows(&rows).ok_or_else(|| e("a token id is outside the table".into()))?;
                        vec![("y", Act::Gpu(o.upload16(&bytes, vec![rows.len(), l.k]).map_err(e)?))]
                    }
                    _ => {
                        let table = self.weight(o, b)?;
                        vec![("y", Act::Gpu(o.embed(&t("ids")?, &table).map_err(e)?))]
                    }
                }
            }
            "norm" | "headnorm" => {
                let x = t("x")?;
                let w = self.weight(o, b)?;
                let rows = x.elements() / w.elements().max(1);
                let y = o.norm(&x.reshape(vec![rows, w.elements()]), &w, d.eps).map_err(e)?;
                vec![("y", Act::Gpu(y.reshape(x.shape.clone())))]
            }
            "linear" => {
                let x = t("x")?;
                let y = if let Some(l) = b.lin.as_ref().filter(|l| l.q4.is_some()) {
                    let (w, s) = self.weight_q4(o, b)?;
                    o.linear_q4(&x, &w, &s, l.n, l.k).map_err(e)?
                } else if b.lin.as_ref().is_some_and(|l| l.int8) {
                    // Weight placement (2026-11-27): after an eviction, one held copy of the 8-bit
                    // packets serves both the bitmask attempt and the dense upload.
                    let _held = if self.on_card(b) { None } else { Some(self.held(b)?) };
                    if let Some(bm) = self.weight_bmq8(o, b)? {
                        o.linear_bmq8(&x, &bm).map_err(e)?
                    } else {
                        let (w, s) = self.weight_q8(o, b)?;
                        o.linear_q8(&x, &w, &s).map_err(e)?
                    }
                } else {
                    o.linear(&x, &self.weight(o, b)?).map_err(e)?
                };
                // A loaded unit: its small matrices run on the host, the delta is added here.
                let y = match self.unit_on(&n.id) {
                    Some(u) => {
                        let xh = o.download(&x).map_err(e)?;
                        let d = o.upload_f32(&u.delta(&xh, tokens), y.shape.clone()).map_err(e)?;
                        o.binary(5, &y, &d).map_err(e)?
                    }
                    None => y,
                };
                vec![("y", Act::Gpu(y))]
            }
            "rope" => {
                if opts.q_scale.is_some() {
                    return Err(e("a query scale is read on the processor only".into()));
                }
                let (q, k, pos) = (t("q")?, t("k")?, t("pos")?);
                if let (Some(c), Some(l)) = (&opts.capture, n.layer) {
                    let kv = self.rope_kv(b, k.elements() / tokens.max(1))?;
                    // Step P2: the bits come down as the device holds them and are stored as
                    // they are (was a download through f32 and a narrowing back, same bits).
                    let host = {
                        let _p = nmffi::prof::Span::new("capture.download");
                        o.download16(&k).map_err(e)?
                    };
                    let _p = nmffi::prof::Span::new("capture.store");
                    c.keys_at16(l, kv, d.kv_heads, d.head_dim, host);
                }
                let hd = d.head_dim;
                let q3 = q.reshape(vec![tokens, q.elements() / (tokens * hd).max(1), hd]);
                let k3 = k.reshape(vec![tokens, k.elements() / (tokens * hd).max(1), hd]);
                let (qo, ko) = o.rope(&q3, &k3, &pos, d.theta).map_err(e)?;
                vec![("q", Act::Gpu(qo.reshape(q.shape.clone()))), ("k", Act::Gpu(ko.reshape(k.shape.clone())))]
            }
            "score" => {
                let (h, g) = (n.heads.unwrap_or((0, d.heads)), n.kv.unwrap_or((0, d.kv_heads)));
                let hd = d.head_dim;
                let mask: Vec<u8> = self.heads_off(n, opts).iter().map(|off| u8::from(!off)).collect();
                let mask_span = nmffi::prof::Span::new("score.mask");
                let m = o.mask_cached(&mask).map_err(e)?;
                drop(mask_span);
                let q = t("q")?.reshape(vec![tokens, h.1 - h.0, hd]);
                let mut k = t("k")?.reshape(vec![tokens, g.1 - g.0, hd]);
                let mut v = t("v")?.reshape(vec![tokens, g.1 - g.0, hd]);
                if let (Some(c), Some(l)) = (&opts.capture, n.layer) {
                    let host = {
                        let _p = nmffi::prof::Span::new("capture.download");
                        o.download16(&v).map_err(e)?
                    };
                    let _p = nmffi::prof::Span::new("capture.store");
                    c.values_at16(l, g, d.kv_heads, hd, host);
                }
                window_holds(opts, n)?;
                // Rows before the new tokens joined on the device: the shared window, then the
                // decode cache's earlier steps (each part copied once into one buffer).
                let cat = |parts: &[&DevT]| -> Result<DevT, String> {
                    let rows: usize = parts.iter().map(|x| x.shape.first().copied().unwrap_or(0)).sum();
                    let out = o.alloc(vec![rows, g.1 - g.0, hd], 2)?;
                    let mut at = 0;
                    for x in parts {
                        let b = x.elements() * 2;
                        o.gpu.copy(&x.buf, 0, &out.buf, at, b)?;
                        at += b;
                    }
                    Ok(out)
                };
                let (new_k, new_v) = (k.clone(), v.clone());
                let mut prev_k: Vec<DevT> = Vec::new();
                let mut prev_v: Vec<DevT> = Vec::new();
                if let (Some(w), Some(l)) = (&opts.window, n.layer) {
                    let dev_span = nmffi::prof::Span::new("window.device");
                    let dev = w.device_range(o, g, hd, d.theta).map_err(e)?;
                    drop(dev_span);
                    if let Some((wk, wv)) = dev.get(l) {
                        prev_k.push(wk.clone());
                        prev_v.push(wv.clone());
                    }
                }
                if let Some(c) = &opts.cache
                    && let Some(Kv::Gpu(pk, pv)) = c.entries.lock().map_err(|_| bad("cache lock"))?.get(&n.id)
                {
                    prev_k.push(pk.clone());
                    prev_v.push(pv.clone());
                }
                if !prev_k.is_empty() && opts.segments.is_none() {
                    let _p = nmffi::prof::Span::new("score.concat");
                    k = cat(&prev_k.iter().chain(std::iter::once(&new_k)).collect::<Vec<_>>()).map_err(e)?;
                    v = cat(&prev_v.iter().chain(std::iter::once(&new_v)).collect::<Vec<_>>()).map_err(e)?;
                }
                if let Some(c) = opts.cache.as_ref().filter(|_| opts.segments.is_none()) {
                    // The cache keeps only what follows the window (the window is shared).
                    let wrows = opts.window.as_ref().map_or(0, |w| w.len);
                    let keep = |x: &DevT| -> Result<DevT, String> {
                        if wrows == 0 {
                            return Ok(x.clone());
                        }
                        let rows = x.shape.first().copied().unwrap_or(0) - wrows;
                        let out = o.alloc(vec![rows, g.1 - g.0, hd], 2)?;
                        o.gpu.copy(&x.buf, wrows * (g.1 - g.0) * hd * 2, &out.buf, 0, rows * (g.1 - g.0) * hd * 2)?;
                        Ok(out)
                    };
                    let (ck, cv) = (keep(&k).map_err(e)?, keep(&v).map_err(e)?);
                    c.entries.lock().map_err(|_| bad("cache lock"))?.insert(n.id.clone(), Kv::Gpu(ck, cv));
                }
                // The drone reading when the model runs on the GPU: the query rows come back (a few kilobytes)
                // and the mass is scored on the CPU over the window's RAM keys.
                if let (Some(p), Some(l), Some(w)) = (&opts.probe, n.layer, &opts.window)
                    && p.layers.contains(&l)
                    && p.accepts(&n.id)
                {
                    let dl_span = nmffi::prof::Span::new("probe.download");
                    let (qh, kh) = (o.download(&q).map_err(e)?, o.download(&new_k).map_err(e)?);
                    drop(dl_span);
                    let _p = nmffi::prof::Span::new("probe.mass");
                    let wk = w.layer_k(l);
                    let pre = wk.len();
                    let kr = cpu::Rows {
                        pre,
                        cached: wk,
                        new: &kh,
                        width: (g.1 - g.0) * hd,
                        table: w.table(),
                        head: hd,
                        win: pre,
                        win_width: d.kv_heads * hd,
                        win_off: g.0 * hd,
                    };
                    let only = p.local_heads(l, h);
                    // Tentacle core A5: the raw scores too when the probe asks (same mass).
                    let (mass, lg) = cpu::attention_read(&qh, &kr, tokens, h.1 - h.0, g.1 - g.0, hd, w.len, only.as_deref(), p.want_logits);
                    p.record(&n.id, l, h, w.len, mass);
                    if p.want_logits {
                        p.record_logits(&n.id, l, h, w.len, tokens, lg);
                    }
                }
                // Lane S package 5: part scores of tail sequences reduced on the card.
                if let (Some(sp), Some(segs), Some(l)) = (&opts.segment_parts, &opts.segments, n.layer)
                    && sp.groups.iter().any(|(hs, _)| hs.iter().any(|(hl, hg)| *hl == l && (h.0..h.1).contains(hg)))
                {
                    let _p = nmffi::prof::Span::new("parts.card");
                    let starts: Vec<usize> = segs.iter().scan(0, |a, n| {
                        let s = *a;
                        *a += n;
                        Some(s)
                    }).collect();
                    let mut part_first = Vec::with_capacity(sp.groups.len());
                    let mut parts: Vec<i32> = Vec::new();
                    for (_, ps) in &sp.groups {
                        part_first.push(parts.len() / 2);
                        parts.extend(ps.iter().flat_map(|(a, b)| [i32::try_from(*a).unwrap_or(0), i32::try_from(*b).unwrap_or(0)]));
                    }
                    // place: (group, question, head index, parts, first output).
                    let mut items: Vec<i32> = Vec::new();
                    let mut place: Vec<[usize; 5]> = Vec::new();
                    let mut outputs = 0;
                    for (i, tail) in sp.tails.iter().enumerate() {
                        let Some((gi, qi)) = *tail else { continue };
                        let Some((hs, ps)) = sp.groups.get(gi) else { continue };
                        let body = opts.segment_bodies.as_ref().and_then(|b| b.get(i).copied().flatten());
                        let (bs, bl) = body.map_or((0, 0), |j| (starts.get(j).copied().unwrap_or(0), segs.get(j).copied().unwrap_or(0)));
                        for (hi, &(hl, hg)) in hs.iter().enumerate() {
                            if hl != l || !(h.0..h.1).contains(&hg) {
                                continue;
                            }
                            let c = |x: usize| i32::try_from(x).unwrap_or(0);
                            items.extend([c(starts[i]), c(segs[i]), c(bs), c(bl), c(hg - h.0), c(part_first[gi]), c(ps.len()), c(outputs)]);
                            place.push([gi, qi, hi, ps.len(), outputs]);
                            outputs += ps.len();
                        }
                    }
                    let kvw = (g.1 - g.0) * hd;
                    let pre = match (prev_k.as_slice(), opts.window.is_some()) {
                        ([], _) => None,
                        ([pk], _) => Some(pk.reshape(vec![pk.elements() / kvw, g.1 - g.0, hd])),
                        _ => Some(cat(&prev_k.iter().collect::<Vec<_>>()).map_err(e)?),
                    };
                    let q3 = q.reshape(vec![tokens, h.1 - h.0, hd]);
                    let k3 = new_k.reshape(vec![tokens, g.1 - g.0, hd]);
                    let vals = o.part_lse(&q3, &k3, pre.as_ref(), &items, &parts, outputs).map_err(e)?;
                    let mut out = sp.out.lock().map_err(|_| bad("part scores lock"))?;
                    for [gi, qi, hi, np, at] in place {
                        if let Some(row) = out.get_mut(gi).and_then(|g| g.get_mut(qi)) {
                            for p in 0..np {
                                if let (Some(d), Some(v)) = (row.get_mut(hi * np + p), vals.get(at + p)) {
                                    *d = *v;
                                }
                            }
                        }
                    }
                }
                // Lane S (2026-11-18, T-074): the relevance-head reads of packed groups, each group's
                // question rows against the window and the group's own earlier rows, recorded as a
                // plain read of that question over the window and the group's pieces records them.
                if let (Some(sp), Some(segs), Some(l)) = (&opts.segment_probe, &opts.segments, n.layer)
                    && sp.probes.first().is_some_and(|p| p.layers.contains(&l))
                {
                    let dl_span = nmffi::prof::Span::new("probe.download");
                    let (qh, kh) = (o.download(&q).map_err(e)?, o.download(&new_k).map_err(e)?);
                    drop(dl_span);
                    let _p = nmffi::prof::Span::new("probe.mass");
                    let (hq, kvw) = ((h.1 - h.0) * hd, (g.1 - g.0) * hd);
                    let wk: Vec<&[u16]> = opts.window.as_ref().map(|w| w.layer_k(l)).unwrap_or_default();
                    let table: &[(f32, f32)] = opts.window.as_ref().map_or(&[], |w| w.table());
                    let wlen = opts.window.as_ref().map_or(0, |w| w.len);
                    let pre = wk.len();
                    let starts: Vec<usize> = segs.iter().scan(0, |a, n| {
                        let s = *a;
                        *a += n;
                        Some(s)
                    }).collect();
                    let mut at = 0;
                    for (i, ((len, probe), rows)) in segs.iter().zip(&sp.probes).zip(&sp.rows).enumerate() {
                        let rows = (*rows).min(*len);
                        if rows > 0 && probe.accepts(&n.id) {
                            // A tail reads its body's keys before its own (package 5).
                            let body = opts.segment_bodies.as_ref().and_then(|b| b.get(i).copied().flatten());
                            let own = kh.get(at * kvw..(at + len) * kvw).unwrap_or(&[]);
                            let joined: Vec<f32>;
                            let (new, blen): (&[f32], usize) = match body.and_then(|j| Some((starts.get(j)?, segs.get(j)?))) {
                                Some((bs, bl)) => {
                                    joined = kh.get(bs * kvw..(bs + bl) * kvw).unwrap_or(&[]).iter().chain(own).copied().collect();
                                    (&joined, *bl)
                                }
                                None => (own, 0),
                            };
                            let kr = cpu::Rows {
                                pre,
                                cached: wk.clone(),
                                new,
                                width: kvw,
                                table,
                                head: hd,
                                win: pre,
                                win_width: d.kv_heads * hd,
                                win_off: g.0 * hd,
                            };
                            let qs = qh.get((at + len - rows) * hq..(at + len) * hq).unwrap_or(&[]);
                            let only = probe.local_heads(l, h);
                            let span = wlen + blen + len - rows;
                            let (mass, lg) = cpu::attention_read(qs, &kr, rows, h.1 - h.0, g.1 - g.0, hd, span, only.as_deref(), probe.want_logits);
                            probe.record(&n.id, l, h, span, mass);
                            if probe.want_logits {
                                probe.record_logits(&n.id, l, h, span, rows, lg);
                            }
                        }
                        at += len;
                    }
                }
                let y = match &opts.segments {
                    // Attention within each packed sequence: its rows of q, k and v copied out,
                    // read on their own, the output copied back in place.
                    // Lane S (2026-11-18): every sequence in one launch, the shared prefix joined
                    // once (bit-identical to the per-sequence reads below).
                    Some(segs)
                        if crate::gpu::segments_ready()
                            && let Some(y) = {
                            let kvw = (g.1 - g.0) * hd;
                            let joined = match (prev_k.as_slice(), prev_v.as_slice()) {
                                ([], _) => None,
                                ([pk], [pv]) => Some((pk.reshape(vec![pk.elements() / kvw, g.1 - g.0, hd]), pv.reshape(vec![pv.elements() / kvw, g.1 - g.0, hd]))),
                                _ => Some((
                                    cat(&prev_k.iter().collect::<Vec<_>>()).map_err(e)?,
                                    cat(&prev_v.iter().collect::<Vec<_>>()).map_err(e)?,
                                )),
                            };
                            o.attention_segments(&q, &k, &v, &m, joined.as_ref().map(|(a, b)| (a, b)), segs, opts.segment_bodies.as_ref().map(|b| b.as_slice())).map_err(e)?
                        } =>
                    {
                        y
                    }
                    Some(_) if opts.segment_bodies.is_some() => return Err(e("segment bodies need the one-launch packed attention".into())),
                    Some(segs) => {
                        let (qw, kw) = ((h.1 - h.0) * hd, (g.1 - g.0) * hd);
                        let out = o.alloc(vec![tokens, h.1 - h.0, hd], 2).map_err(e)?;
                        let mut at = 0;
                        for &len in segs.iter() {
                            let take = |x: &DevT, w: usize, heads: usize| -> Result<DevT, String> {
                                let s = o.alloc(vec![len, heads, hd], 2)?;
                                o.gpu.copy(&x.buf, at * w * 2, &s.buf, 0, len * w * 2)?;
                                Ok(s)
                            };
                            let (qs, mut ks, mut vs) = (
                                take(&q, qw, h.1 - h.0).map_err(e)?,
                                take(&k, kw, g.1 - g.0).map_err(e)?,
                                take(&v, kw, g.1 - g.0).map_err(e)?,
                            );
                            // The shared prefix (a prefilled head) comes first in every sequence.
                            if !prev_k.is_empty() {
                                ks = cat(&prev_k.iter().chain(std::iter::once(&ks)).collect::<Vec<_>>()).map_err(e)?;
                                vs = cat(&prev_v.iter().chain(std::iter::once(&vs)).collect::<Vec<_>>()).map_err(e)?;
                            }
                            let ys = o.attention_segment(&qs, &ks, &vs, &m).map_err(e)?;
                            o.gpu.copy(&ys.buf, 0, &out.buf, at * qw * 2, len * qw * 2).map_err(e)?;
                            at += len;
                        }
                        out
                    }
                    None => o.attention(&q, &k, &v, &m).map_err(e)?,
                };
                vec![("y", Act::Gpu(y.reshape(vec![tokens, (h.1 - h.0) * hd])))]
            }
            "swigate" => {
                let y = o.binary(4, &t("gate")?, &t("up")?).map_err(e)?;
                match n.layer.and_then(|l| opts.blocks_off.get(&l)) {
                    // Block lesion on the device: the binary16 output comes back as its bits, the
                    // blocks are zeroed and it goes back up (an exact round trip, no conversion).
                    Some(off) => {
                        let mut h = o.download16(&y).map_err(e)?;
                        zero_blocks(&mut h, tokens, off, 0_u16);
                        vec![("y", Act::Gpu(o.upload_parts16(&[&h], y.shape.clone()).map_err(e)?))]
                    }
                    None => vec![("y", Act::Gpu(y))],
                }
            }
            "add" => {
                let y = o.binary(5, &t("a")?, &t("b")?).map_err(e)?;
                match n.layer.filter(|_| n.id.ends_with(".res2")).and_then(|l| opts.steer.get(&l)) {
                    // The steered residual comes back, is shifted and rounded, and goes up again.
                    Some((v, s)) => {
                        let mut h = o.download(&y).map_err(e)?;
                        steer_rows(&mut h, v, *s);
                        vec![("y", Act::Gpu(o.upload_f32(&h, y.shape.clone()).map_err(e)?))]
                    }
                    None => vec![("y", Act::Gpu(y))],
                }
            }
            "head" => {
                let x = t("x")?;
                let rows: &[usize] = if b.rows.is_empty() { &opts.head_rows } else { &b.rows };
                let w = if rows.is_empty() {
                    self.weight(o, b)?
                } else {
                    // Label rows or active-vocabulary subset (`opts.head_rows`): upload only those
                    // rows to VRAM, hashing large row lists into a fixed-length cache key.
                    let cache_key = if rows.len() <= 16 {
                        format!("{}:rows{:?}", b.key, rows)
                    } else if b.rows.is_empty() {
                        head_rows_cache_key(&b.key, &opts.head_rows)
                    } else {
                        let raw: Vec<u8> = rows.iter().flat_map(|r| (*r as u64).to_le_bytes()).collect();
                        let h = nmhash::sha256_bytes(&raw);
                        format!("{}:rows{}:{}", b.key, rows.len(), &h[..16])
                    };
                    self.pop.device(&cache_key, &b.views, || {
                        let held = self.held(b).map_err(|e| e.to_string())?;
                        let lin = &held;
                        if let Some(packed) = lin.packed16_rows(rows) {
                            return o.upload16(&packed, vec![rows.len(), lin.k]);
                        }
                        let mut buf = vec![0.0_f32; lin.k];
                        let mut packed = Vec::with_capacity(rows.len() * lin.k * 2);
                        for r in rows {
                            lin.row(*r, &mut buf);
                            packed.extend(buf.iter().flat_map(|v| crate::half::bits16(*v).to_le_bytes()));
                        }
                        o.upload16(&packed, vec![rows.len(), lin.k])
                    })?
                };
                let y = if opts.all_positions {
                    o.linear(&x, &w).map_err(e)?
                } else if let Some(segs) = &opts.segments {
                    if x.shape.first().copied() == Some(segs.len()) {
                        // Already gathered to one last-token row per segment (X3).
                        o.linear(&x, &w).map_err(e)?
                    } else {
                        // The last token of each packed sequence, one row each.
                        let last = o.alloc(vec![segs.len(), d.hidden], 2).map_err(e)?;
                        let mut end = 0;
                        for (r, n) in segs.iter().enumerate() {
                            end += n;
                            o.gpu.copy(&x.buf, (end - 1) * d.hidden * 2, &last.buf, r * d.hidden * 2, d.hidden * 2).map_err(e)?;
                        }
                        o.linear(&last, &w).map_err(e)?
                    }
                } else if x.shape.first().copied() == Some(1) {
                    o.linear(&x, &w).map_err(e)?
                } else {
                    let last = o.alloc(vec![1, d.hidden], 2).map_err(e)?;
                    o.gpu.copy(&x.buf, (tokens - 1) * d.hidden * 2, &last.buf, 0, d.hidden * 2).map_err(e)?;
                    o.linear(&last, &w).map_err(e)?
                };
                vec![("logits", Act::Gpu(y))]
            }
            // Output components gather only the needed last-token rows on GPU before downloading
            // (cuts PCIe transfer 100x-250x on packed embedding batches, bit-identical).
            "lastpool" => {
                let gx = t("x")?;
                match &opts.segments {
                    // One pooled vector per packed sequence, side by side.
                    Some(segs) => {
                        let compact = if gx.shape.first().copied() == Some(segs.len()) {
                            gx
                        } else {
                            let rows: Vec<usize> = segs
                                .iter()
                                .scan(0_usize, |end, n| {
                                    *end += *n;
                                    Some(end.saturating_sub(1))
                                })
                                .collect();
                            gather_gpu_rows(o, &gx, &rows).map_err(e)?
                        };
                        let x = o.download(&compact).map_err(e)?;
                        let mut out = Vec::with_capacity(segs.len() * d.hidden);
                        for k in 0..segs.len() {
                            out.extend(last_pool(x.get(k * d.hidden..(k + 1) * d.hidden).unwrap_or(&[]), d.hidden));
                        }
                        vec![("y", Act::Cpu(out))]
                    }
                    None => {
                        let compact = if gx.shape.first().copied() == Some(1) {
                            gx
                        } else {
                            gather_gpu_rows(o, &gx, &[tokens.saturating_sub(1)]).map_err(e)?
                        };
                        let x = o.download(&compact).map_err(e)?;
                        vec![("y", Act::Cpu(last_pool(&x, d.hidden)))]
                    }
                }
            }
            "labelsoftmax" => {
                let logits = o.download(&t("logits")?).map_err(e)?;
                let width = b.label_width(d.vocab);
                match &opts.segments {
                    // One row of logits per packed sequence: its labels' probabilities, side by side.
                    Some(_) => vec![(
                        "y",
                        Act::Cpu(logits.chunks_exact(width.max(1)).flat_map(|row| label_softmax(row, width, &b.label_read())).collect()),
                    )],
                    None => vec![("y", Act::Cpu(label_softmax(&logits, width, &b.label_read())))],
                }
            }
            other => return Err(e(format!("unknown operation {other}"))),
        })
    }

    /// Prefill `ids` at positions 0.. and return the requested targets (on `device`).
    /// # Errors
    /// As `run`.
    pub fn forward(
        &self,
        device: Device,
        ids: &[i32],
        targets: &[(String, String)],
        opts: &RunOpts,
    ) -> Result<BTreeMap<(String, String), Act>, Fault> {
        self.forward_at(device, ids, 0, targets, opts)
    }
    /// Prefill or step `ids` at positions `start..` (decoding with `opts.cache`).
    /// # Errors
    /// As `run`.
    pub fn forward_at(
        &self,
        device: Device,
        ids: &[i32],
        start: usize,
        targets: &[(String, String)],
        opts: &RunOpts,
    ) -> Result<BTreeMap<(String, String), Act>, Fault> {
        let before = crate::gpu::NO_HALF.with(|c| c.replace(self.no_half.load(Ordering::Relaxed)));
        let dot = crate::gpu::INT8_DOT.with(|c| c.replace(self.int8_dot.load(Ordering::Relaxed)));
        let live = exec_start(&self.pop.card.id);
        let r = self.forward_at_inner(device, ids, start, targets, opts);
        exec_end(&self.pop.card.id, live);
        crate::gpu::NO_HALF.with(|c| c.set(before));
        crate::gpu::INT8_DOT.with(|c| c.set(dot));
        r
    }
    /// Run from given activations (`seeds`, for example a middle layer's residual and the
    /// positions) to `targets`, with the model's own math settings, as `forward_at` does from ids.
    /// # Errors
    /// As `run`.
    pub fn run_from(
        &self,
        device: Device,
        seeds: Vec<(String, String, Act)>,
        targets: &[(String, String)],
        tokens: usize,
        opts: &RunOpts,
    ) -> Result<BTreeMap<(String, String), Act>, Fault> {
        let before = crate::gpu::NO_HALF.with(|c| c.replace(self.no_half.load(Ordering::Relaxed)));
        let dot = crate::gpu::INT8_DOT.with(|c| c.replace(self.int8_dot.load(Ordering::Relaxed)));
        let live = exec_start(&self.pop.card.id);
        let r = self.run(device, seeds, targets, tokens, opts);
        exec_end(&self.pop.card.id, live);
        crate::gpu::NO_HALF.with(|c| c.set(before));
        crate::gpu::INT8_DOT.with(|c| c.set(dot));
        r
    }
    fn forward_at_inner(
        &self,
        device: Device,
        ids: &[i32],
        start: usize,
        targets: &[(String, String)],
        opts: &RunOpts,
    ) -> Result<BTreeMap<(String, String), Act>, Fault> {
        let s = i32::try_from(start).unwrap_or(0);
        let pos: Vec<i32> = match &opts.segments {
            Some(segs) => {
                let held = opts.window.as_ref().map_or(0, |w| w.len)
                    + opts.cache.as_ref().map_or(0, |c| c.len(self.pop.dims.kv_heads * self.pop.dims.head_dim));
                if start != held
                    || device != Device::Gpu
                    || segs.contains(&0)
                    || segs.iter().sum::<usize>() != ids.len()
                    || opts.capture.is_some()
                    || opts.probe.is_some()
                    || opts.segment_probe.as_ref().is_some_and(|p| p.probes.len() != segs.len() || p.rows.len() != segs.len() || opts.cache.is_some())
                    || opts.segment_bodies.as_ref().is_some_and(|b| {
                        b.len() != segs.len() || b.iter().enumerate().any(|(i, x)| x.is_some_and(|j| j >= i || b.get(j).copied().flatten().is_some()))
                    })
                {
                    return Err(bad(
                        "segments: on the GPU, start equal to the shared prefix (window and cache), non-empty lengths summing to the tokens, and no capture or probe",
                    ));
                }
                let body_len = |i: usize| -> i32 {
                    opts.segment_bodies.as_ref().and_then(|b| b.get(i).copied().flatten()).and_then(|j| segs.get(j)).map_or(0, |n| i32::try_from(*n).unwrap_or(0))
                };
                segs.iter().enumerate().flat_map(|(i, n)| s + body_len(i)..s + body_len(i) + i32::try_from(*n).unwrap_or(0)).collect()
            }
            None => (s..s + i32::try_from(ids.len()).unwrap_or(0)).collect(),
        };
        let seeds = vec![
            (crate::assembly::nid("ids"), "y".to_owned(), Act::Ids(ids.to_vec())),
            (crate::assembly::nid("pos"), "y".to_owned(), Act::Ids(pos)),
        ];
        self.run(device, seeds, targets, ids.len(), opts)
    }

    /// WP-A1 X3: plan dead prefill row elimination right after the final `score` node in `order`.
    /// Returns a `LastGatherPlan` when safe: every node strictly after `score_pos` is
    /// token-row-independent and every requested target only observes the last token row of each
    /// segment (or the single last token). Works for any layer count.
    fn last_layer_gather_plan(
        &self,
        order: &[usize],
        targets: &[(String, String)],
        tokens: usize,
        opts: &RunOpts,
    ) -> Option<LastGatherPlan> {
        if !last_layer_prune_on()
            || self.pop.dims.layers == 0
            || tokens <= 2
            || opts.all_positions
            || opts.capture.is_some()
            || opts.probe.is_some()
            || opts.segment_probe.is_some()
            || opts.segment_parts.is_some()
            || !opts.blocks_off.is_empty()
            || !opts.steer.is_empty()
            || (opts.cache.is_some() && opts.segments.is_none())
            || crate::pmap::on(&self.pop.card.id)
            || crate::chanrec::on(&self.pop.card.id)
        {
            return None;
        }
        let gather_rows: Vec<usize> = match &opts.segments {
            Some(segs) => {
                if segs.is_empty() || segs.contains(&0) || segs.iter().sum::<usize>() != tokens {
                    return None;
                }
                segs.iter()
                    .scan(0_usize, |end, n| {
                        *end += *n;
                        Some(*end - 1)
                    })
                    .collect()
            }
            None => vec![tokens - 1],
        };
        if gather_rows.len() * 2 >= tokens {
            return None;
        }
        let fin_id = self.asm.named("final_norm");
        for (tid, _) in targets {
            let op = self.asm.node(tid).map(|n| n.op.as_str())?;
            let allowed = matches!(op, "head" | "labelsoftmax" | "lastpool") || (opts.last_rows_only && *tid == fin_id);
            if !allowed {
                return None;
            }
        }
        let score_pos = order.iter().rposition(|&i| self.bound.get(i).is_some_and(|b| b.node.op == "score"))?;
        if score_pos + 1 >= order.len() {
            return None;
        }
        let mut post_nodes: BTreeSet<&str> = BTreeSet::new();
        for &i in order.get(score_pos + 1..)? {
            let b = self.bound.get(i)?;
            let op = b.node.op.as_str();
            if !matches!(op, "linear" | "add" | "norm" | "headnorm" | "swigate" | "pass" | "head" | "labelsoftmax" | "lastpool") {
                return None;
            }
            if self.unit_on(&b.node.id).is_some() {
                return None;
            }
            post_nodes.insert(b.node.id.as_str());
        }
        let mut crossing_set: BTreeSet<(String, String)> = BTreeSet::new();
        for e in &self.asm.edges {
            if post_nodes.contains(e.to.as_str()) && !post_nodes.contains(e.from.as_str()) {
                crossing_set.insert((e.from.clone(), e.from_port.clone()));
            }
        }
        if crossing_set.is_empty() {
            return None;
        }
        Some(LastGatherPlan { score_pos, rows: gather_rows, crossing: crossing_set.into_iter().collect() })
    }
}

/// Plan for WP-A1 X3 dead prefill last-layer row pruning.
struct LastGatherPlan {
    score_pos: usize,
    rows: Vec<usize>,
    crossing: Vec<(String, String)>,
}

/// Whether WP-A1 X3 dead prefill last-layer pruning is active (`NANO_LAST_LAYER_PRUNE=off` disables
/// it for paired control-vs-candidate comparisons).
fn last_layer_prune_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("NANO_LAST_LAYER_PRUNE").as_deref(), Ok("off" | "0" | "false")))
}

type HeadRowsKeyEntry = (std::sync::Weak<Vec<usize>>, String, String);
static HEAD_ROWS_KEY_CACHE: Mutex<Vec<HeadRowsKeyEntry>> = Mutex::new(Vec::new());

/// Caches the SHA-256 device cache key for a large `Arc<Vec<usize>>` `opts.head_rows` subset
/// keyed by live `Weak` identity, avoiding re-hashing ~24K row indices on every decode token.
fn head_rows_cache_key(base_key: &str, rows: &Arc<Vec<usize>>) -> String {
    let weak = Arc::downgrade(rows);
    if let Ok(mut g) = HEAD_ROWS_KEY_CACHE.lock() {
        g.retain(|(w, _, _)| w.strong_count() > 0);
        if let Some((_, _, cached)) = g.iter().find(|(w, bk, _)| w.ptr_eq(&weak) && bk == base_key) {
            return cached.clone();
        }
    }
    let raw: Vec<u8> = rows.iter().flat_map(|r| (*r as u64).to_le_bytes()).collect();
    let h = nmhash::sha256_bytes(&raw);
    let key = format!("{base_key}:rows{}:{}", rows.len(), &h[..16]);
    if let Ok(mut g) = HEAD_ROWS_KEY_CACHE.lock() {
        if g.len() >= 16 {
            g.remove(0);
        }
        g.push((weak, base_key.to_owned(), key.clone()));
    }
    key
}

thread_local! {
    /// Set while a `JobFiltersHeldOff` is alive on this thread.
    static JOB_FILTERS_HELD_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// While alive, forwards on this thread leave the automatic job filters out (2026-11-24): a
/// section check that measures its own sockets against a float64 reference (`filter::check_model`)
/// needs the MicroModel bare, as it needs the rack empty. Serving never holds one.
#[derive(Debug)]
pub struct JobFiltersHeldOff {
    before: bool,
}
impl JobFiltersHeldOff {
    /// Holds the job filters off on this thread.
    #[must_use]
    pub fn new() -> Self {
        Self { before: JOB_FILTERS_HELD_OFF.with(|c| c.replace(true)) }
    }
}
impl Default for JobFiltersHeldOff {
    fn default() -> Self {
        Self::new()
    }
}
impl Drop for JobFiltersHeldOff {
    fn drop(&mut self) {
        JOB_FILTERS_HELD_OFF.with(|c| c.set(self.before));
    }
}

/// Whether Stage 6B hot-swappable per-hop job filters (`library/parameter-maps/<role>/job-filter-*.json`)
/// load automatically on every forward of the MicroModel `card` (`NANO_JOB_FILTERS`). Roadmap
/// Phase 0 item 6 (2026-11-24): per-role isolation. `on` (or `1`, `true`) loads every role's
/// filters; a comma list of roles (`embedder`, `reranker`, `reader`, `writer`) loads only those
/// roles' filters, so each role can be measured and promoted on its own; unset, the embedder's
/// (measured, below); `off` (or `0`, `false`), none, for diagnosis.
fn job_filters_auto_on(card: &str) -> bool {
    static SET: std::sync::OnceLock<Option<Vec<String>>> = std::sync::OnceLock::new();
    // Roadmap Phase 0 item 6 (2026-11-24): the embedder's filters serve by default. Full sets with
    // dp4a on: dev-large-2 first 60 to 62, top 3 79 to 81, pack 89 to 91, every 81 to 82, decoys
    // 20/20 (the gate: first 61 or more, pack 88, every 79); large code tuning within one. Other
    // roles join the default when their own files are measured; `NANO_JOB_FILTERS=off` for diagnosis.
    // The reader's (held-out cos 0.9811 to 0.9837) gave the same answers on both full sets and a
    // lower median and worst time: ON. The reranker's (0.9819 to 0.9836) gave the same answers, 1.6
    // percent slower: NEEDS_HILLCLIMB, which stays on. The writer's acts on idle notes and is
    // measured with them (EVALUATION).
    let set = SET.get_or_init(|| match std::env::var("NANO_JOB_FILTERS").as_deref() {
        Err(_) => Some(vec!["embedder".to_owned(), "reranker".to_owned(), "reader".to_owned()]),
        Ok("off" | "0" | "false") => None,
        Ok("on" | "1" | "true") => Some(Vec::new()),
        Ok(list) => {
            // `,` or `+` between roles (`+` passes through the stage bench's comma-split `--env`).
            let roles: Vec<String> = list.split([',', '+']).map(|r| r.trim().to_ascii_lowercase()).filter(|r| !r.is_empty()).collect();
            (!roles.is_empty()).then_some(roles)
        }
    });
    match set {
        None => false,
        Some(roles) if roles.is_empty() => true,
        Some(roles) => roles.iter().any(|r| role_matches(r, card)),
    }
}

/// Whether the role name `role` names the MicroModel `card` (its card id): embedder (the embedding
/// model), reranker (the judge), writer (the 1.7B), reader (the 0.6B base).
fn role_matches(role: &str, card: &str) -> bool {
    // 2026-11-25: the role a chosen model runs as (`nmtype::role_models`) first.
    if let Some(r) = nmtype::role_models::role_of(card) {
        return r == role || (role == "judge" && r == "reranker");
    }
    let c = card.to_ascii_lowercase();
    match role {
        "embedder" => c.contains("embedding"),
        "reranker" | "judge" => c.contains("reranker"),
        "writer" => c.contains("1.7b"),
        "reader" => c.contains("0.6b") && !c.contains("embedding") && !c.contains("reranker"),
        other => c.contains(other),
    }
}

/// Gathers `rows` along leading dimension 0 of device tensor `x` into a compact `[rows.len(), ...]`
/// device tensor of identical dtype.
fn gather_gpu_rows(o: &Ops, x: &DevT, rows: &[usize]) -> Result<DevT, String> {
    let total_rows = x.shape.first().copied().unwrap_or(0);
    if total_rows == 0 || rows.iter().any(|&r| r >= total_rows) {
        return Err(format!("gather_gpu_rows: index out of bounds for shape {:?}", x.shape));
    }
    let elem_bytes: usize = match x.dtype {
        crate::gpu::F16 => 2,
        1 | 5 => 4,
        _ => 1,
    };
    let row_elems = x.elements() / total_rows;
    let row_bytes = row_elems * elem_bytes;
    let mut shape = x.shape.clone();
    if let Some(first) = shape.first_mut() {
        *first = rows.len();
    }
    let out = o.alloc(shape, x.dtype)?;
    for (dst_r, &src_r) in rows.iter().enumerate() {
        o.gpu.copy(&x.buf, src_r * row_bytes, &out.buf, dst_r * row_bytes, row_bytes)?;
    }
    Ok(out)
}

/// For each step of `order` (its position), the activations whose last reader is that step
/// (step P2, 2026-11-10): an edge into a node of the order marks its source port as read there;
/// the largest position wins.
fn last_readers(asm: &Assembly, bound: &[Bound], order: &[usize]) -> BTreeMap<usize, Vec<(String, String)>> {
    let at: BTreeMap<&str, usize> = order.iter().enumerate().filter_map(|(k, i)| bound.get(*i).map(|b| (b.node.id.as_str(), k))).collect();
    let mut last: BTreeMap<(String, String), usize> = BTreeMap::new();
    for e in &asm.edges {
        if let Some(&k) = at.get(e.to.as_str()) {
            let x = last.entry((e.from.clone(), e.from_port.clone())).or_insert(k);
            *x = (*x).max(k);
        }
    }
    let mut dies: BTreeMap<usize, Vec<(String, String)>> = BTreeMap::new();
    for (key, k) in last {
        dies.entry(k).or_default().push(key);
    }
    dies
}

/// The stage-timing span name of an operation (step P1, 2026-11-10).
fn op_span(op: &str) -> &'static str {
    match op {
        "source" => "op.source",
        "pass" => "op.pass",
        "embed" => "op.embed",
        "norm" => "op.norm",
        "headnorm" => "op.headnorm",
        "linear" => "op.linear",
        "rope" => "op.rope",
        "score" => "op.score",
        "swigate" => "op.swigate",
        "add" => "op.add",
        "head" => "op.head",
        "lastpool" => "op.lastpool",
        "labelsoftmax" => "op.labelsoftmax",
        _ => "op.other",
    }
}

/// Values of a CPU activation (after `run` every target is on the host).
/// # Errors
/// The activation is not f32 values.
pub fn values(a: &Act) -> Result<&[f32], Fault> {
    match a {
        Act::Cpu(x) => Ok(x),
        _ => Err(bad("not host values")),
    }
}

/// Text for the P0 checks (written for them; not from any test).
const CHECK_ANCHOR: &str = "Repository neuromorph-mini: a Rust workspace (engine checks).\n";
const CHECK_CHUNK: &str = "crates/x/src/ring.rs\npub struct Ring { items: Vec<u32>, head: usize }\nimpl Ring {\n    pub fn push(&mut self, x: u32) { let n = self.items.len(); self.items[self.head % n] = x; self.head += 1; }\n}\n";
const CHECK_QUESTION: &str = "\nQuestion: where does the ring write a new item";

/// The devices a weight-bound check runs on: the CPU, and the GPU when one opened.
fn check_devices(pop: &Population) -> Vec<Device> {
    if pop.gpu.is_some() { vec![Device::Cpu, Device::Gpu] } else { vec![Device::Cpu] }
}
fn worst(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Section check of step P0 item 1 (2026-11-10), on the released Qwen3-0.6B: a `score` node
/// whose query heads do not line up with the shared key-value groups is refused when the
/// assembly binds (two cases that pass the width check: a shifted head range, a moved
/// key-value range), and every aligned sub-range (halves, among them heads 8..16 with
/// key-value heads 4..8, and quarters, for example heads 4..8 with key-value heads 2..4) gives
/// exactly the numbers of the same heads inside the one all-heads node, on the CPU and on the
/// GPU when one is open. Measured values: `tests/section-checks/nmengn-kv-groups/report.txt`.
pub(crate) fn check_kv_groups() -> Result<(), String> {
    use std::fmt::Write as _;
    let mut rep = String::from("nmengn.model.kv_groups (step P0 item 1)\n");
    let r = kv_groups(&mut rep);
    let _ = writeln!(rep, "result: {}", r.as_ref().map_or_else(|e| format!("FAIL {e}"), |()| "ok".to_owned()));
    let dir = nmtype::check_dir("nmengn-kv-groups").map_err(|e| e.to_string())?;
    std::fs::write(dir.join("report.txt"), rep).map_err(|e| e.to_string())?;
    r
}
fn kv_groups(rep: &mut String) -> Result<(), String> {
    use crate::assembly::{Shape, generate, nid};
    use std::fmt::Write as _;
    // The rule alone (Qwen3-0.6B geometry: 16 query heads over 8 key-value heads).
    for (h, g, ok) in [
        ((0, 16), (0, 8), true),
        ((8, 16), (4, 8), true),
        ((4, 8), (2, 4), true),
        ((2, 4), (1, 2), true),
        ((1, 9), (0, 4), false),
        ((0, 8), (1, 5), false),
        ((1, 3), (0, 2), false),
        ((0, 3), (0, 2), false),
    ] {
        if kv_alignment(h, g, 16, 8).is_none() != ok {
            return Err(format!("alignment rule: heads {h:?} kv {g:?} should be {}", if ok { "accepted" } else { "refused" }));
        }
    }
    let home = nmtype::nano_root();
    let pop = Population::open(&home, "qwen3-0.6b", true, true).map_err(|e| e.to_string())?;
    let d = pop.dims;
    let one = Shape { layers: 1, heads: d.heads, kv_heads: d.kv_heads, head_dim: d.head_dim };
    // Refusals: the split assembly with one path's heads shifted, or its kv range moved.
    for (node, field, value) in [("L0.scorea", "heads", (1, 9)), ("L0.scoreb", "kv", (3, 7))] {
        let mut a = generate("p0-misaligned", one, 2);
        let n = a.nodes.iter_mut().find(|n| n.id == nid(node)).ok_or("no node to shift")?;
        if field == "heads" {
            n.heads = Some(value);
        } else {
            n.kv = Some(value);
        }
        match Model::bind(&pop, a) {
            Ok(_) => return Err(format!("{node} with {field} {value:?} was bound")),
            Err(e) if e.to_string().contains("line up") => {
                let _ = writeln!(rep, "refused: {node} {field} {value:?}: {e}");
            }
            Err(e) => return Err(format!("{node} with {field} {value:?} refused for another reason: {e}")),
        }
    }
    // Aligned sub-ranges against the all-heads node.
    let full = Model::bind(&pop, generate("p0-full", one, 1)).map_err(|e| e.to_string())?;
    let halves = Model::bind(&pop, generate("p0-halves", one, 2)).map_err(|e| e.to_string())?;
    let quarters = Model::bind(&pop, generate("p0-quarters", one, 4)).map_err(|e| e.to_string())?;
    let ids = pop.tokenizer.encode(&format!("{CHECK_ANCHOR}{CHECK_CHUNK}")).map_err(|e| e.to_string())?;
    let hd = d.head_dim;
    for dev in check_devices(&pop) {
        let t = (nid("L0.score"), "y".to_owned());
        let all = full.forward(dev, &ids, std::slice::from_ref(&t), &RunOpts::default()).map_err(|e| e.to_string())?;
        let all = values(all.get(&t).ok_or("no full output")?).map_err(|e| e.to_string())?.to_vec();
        let width = d.heads * hd;
        for (m, tags) in [(&halves, &["a", "b"][..]), (&quarters, &["a", "b", "c", "d"][..])] {
            for tag in tags {
                let id = nid(&format!("L0.score{tag}"));
                let node = m.asm.node(&id).ok_or("no path node")?;
                let (h, g) = (node.heads.ok_or("no heads")?, node.kv.ok_or("no kv")?);
                let tp = (id.clone(), "y".to_owned());
                let got = m.forward(dev, &ids, std::slice::from_ref(&tp), &RunOpts::default()).map_err(|e| e.to_string())?;
                let got = values(got.get(&tp).ok_or("no path output")?).map_err(|e| e.to_string())?.to_vec();
                let want: Vec<f32> =
                    all.chunks_exact(width).flat_map(|row| row.get(h.0 * hd..h.1 * hd).unwrap_or(&[]).iter().copied()).collect();
                let diff = worst(&got, &want);
                let _ = writeln!(
                    rep,
                    "{} heads {}..{} kv {}..{}: {} values ({} tokens), max abs against the same heads of the all-heads node {diff}",
                    dev.name(),
                    h.0,
                    h.1,
                    g.0,
                    g.1,
                    got.len(),
                    ids.len()
                );
                if diff != 0.0 || got.is_empty() {
                    return Err(format!("{}: heads {h:?} kv {g:?} differ from the all-heads node by {diff}", dev.name()));
                }
            }
        }
    }
    pop.release_device();
    Ok(())
}

/// Section check of step P0 item 2 (2026-11-10), on the released Qwen3-0.6B, CPU and GPU when
/// open: with `qwen3-split-attention` (two attention components per layer, heads 0..8 and
/// 8..16, mathematically the reference) the captured keys and values of an encoded chunk, the
/// drone heads' attention mass and every head's mass through the last drone layer equal those
/// of `qwen3-reference` within the tolerance the split's existing parity check uses (S6 gate 5:
/// the recorded prompt0 logits tolerance); the split assembly reading the reference's own
/// window (each component reading its key-value columns) agrees the same way; layer 0 keys and
/// values are equal bit for bit (no output projection precedes them). Mass is compared per
/// question token. Measured values: `tests/section-checks/nmengn-by-node/report.txt`.
pub(crate) fn check_by_node() -> Result<(), String> {
    use std::fmt::Write as _;
    let mut rep = String::from("nmengn.model.by_node (step P0 item 2)\n");
    let r = by_node(&mut rep);
    let _ = writeln!(rep, "result: {}", r.as_ref().map_or_else(|e| format!("FAIL {e}"), |()| "ok".to_owned()));
    let dir = nmtype::check_dir("nmengn-by-node").map_err(|e| e.to_string())?;
    std::fs::write(dir.join("report.txt"), rep).map_err(|e| e.to_string())?;
    r
}
fn by_node(rep: &mut String) -> Result<(), String> {
    use crate::kvstore::{ChunkKv, Probe, Window, drone_set, encode_through, score_targets};
    use std::fmt::Write as _;
    let home = nmtype::nano_root();
    // Tolerances (2026-11-10): the S6 role parity limits, the engine's existing parity rules for
    // vectors and probabilities: every key and value row (one token, every key-value head) has
    // cosine at least 0.999 with its reference row; attention mass per question token (a
    // probability) differs by at most 0.003. A first limit tried, the prompt0 logits atol
    // (0.0454) as an absolute limit on keys and values, does not fit them: they reach several
    // hundred, where one binary16 step is 0.25, and the split rounds its output projection in
    // two halves (measured: cosine above 0.99999 on every layer, largest difference 0.42).
    // 2026-11-10, measured while building this check: on the GPU the split assembly differs from
    // the reference by more than that on single token rows even with no window at all (one
    // sequence, plain encoding: worst row cosine 0.9962), so the key and value limit on a device
    // is 0.999 or that device's no-window control, whichever is lower; layer 0 is exact.
    let (c_tol, p_tol) = (0.999_f64, 0.003_f32);
    let _ = writeln!(
        rep,
        "tolerances: key and value rows cosine at least {c_tol}, or the same device's no-window control when lower; mass per question token at most {p_tol} (S6 role parity limits); layer 0 keys and values equal bit for bit"
    );
    let pop = Population::open(&home, "qwen3-0.6b", true, true).map_err(|e| e.to_string())?;
    let d = pop.dims;
    let dir = home.join("designs/assemblies");
    let reference = Model::bind(&pop, Assembly::load(&dir, "qwen3-reference").map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let split = Model::bind(&pop, Assembly::load(&dir, "qwen3-split-attention").map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let drones = drone_set(&home, &pop.card.id).map_err(|e| e.to_string())?;
    let last = drones.last_layer();
    let _ = writeln!(rep, "drone set: {} ({} pairs, last layer {last})", drones.source, drones.heads.len());
    let tok = |s: &str| pop.tokenizer.encode(s).map_err(|e| e.to_string());
    let (a_ids, c_ids, q) = (tok(CHECK_ANCHOR)?, tok(CHECK_CHUNK)?, tok(CHECK_QUESTION)?);
    #[expect(clippy::cast_precision_loss, reason = "a token count")]
    let per = 1.0 / q.len().max(1) as f32;
    let f16 = |v: &[u16]| v.iter().map(|b| crate::half::from16(*b)).collect::<Vec<f32>>();
    type Kept<'a> = Vec<(&'a str, Arc<ChunkKv>, Arc<ChunkKv>)>;
    let (mut fails, mut refs, mut splits): (Vec<String>, Kept<'_>, Kept<'_>) = (Vec::new(), Vec::new(), Vec::new());
    let row = d.kv_heads * d.head_dim;
    let cos = |a: &[f32], b: &[f32]| {
        let (ab, aa, bb) = a.iter().zip(b).fold((0.0_f64, 0.0_f64, 0.0_f64), |t, (x, y)| {
            (t.0 + f64::from(*x) * f64::from(*y), t.1 + f64::from(*x) * f64::from(*x), t.2 + f64::from(*y) * f64::from(*y))
        });
        if aa == 0.0 && bb == 0.0 { 1.0 } else { ab / (aa.sqrt() * bb.sqrt()).max(1e-30) }
    };
    for dev in check_devices(&pop) {
        let n = dev.name();
        type Enc = (Arc<ChunkKv>, Arc<ChunkKv>, Arc<Window>);
        let enc = |m: &Model| -> Result<Enc, String> {
            let ak = Arc::new(encode_through(m, dev, None, &a_ids, last).map_err(|e| e.to_string())?);
            let aw = Arc::new(Window::build(m, &[Arc::clone(&ak)]).map_err(|e| e.to_string())?);
            let ck = Arc::new(encode_through(m, dev, Some(&aw), &c_ids, last).map_err(|e| e.to_string())?);
            let w = Arc::new(Window::build(m, &[Arc::clone(&ak), Arc::clone(&ck)]).map_err(|e| e.to_string())?);
            Ok((ak, ck, w))
        };
        let (ra, rc, rw) = enc(&reference)?;
        let (sa, sc, sw) = enc(&split)?;
        refs.push((n, Arc::clone(&ra), Arc::clone(&rc)));
        splits.push((n, Arc::clone(&sa), Arc::clone(&sc)));
        let (mut kv_abs, mut kv_cos, mut l0) = (0.0_f32, 1.0_f64, 0.0_f32);
        for l in 0..=last {
            let (mut la, mut lc) = (0.0_f32, 1.0_f64);
            for (r, s) in [(&ra, &sa), (&rc, &sc)] {
                for keys in [true, false] {
                    let pick = |c: &ChunkKv| -> Result<Vec<f32>, String> {
                        (if keys { c.k_pre.get(l) } else { c.v.get(l) }).map(|x| f16(x)).ok_or_else(|| format!("layer {l} missing"))
                    };
                    let (x, y) = (pick(r)?, pick(s)?);
                    la = la.max(worst(&x, &y));
                    lc = x.chunks(row).zip(y.chunks(row)).map(|(p, q)| cos(p, q)).fold(lc, f64::min);
                    // Rows below 0.9995: per half of the key-value heads (a column fault shows
                    // in one half on every row; rounding shows in both halves of a few rows).
                    let h = row / 2;
                    for (i, (p, q)) in x.chunks(row).zip(y.chunks(row)).enumerate() {
                        let (c0, c1) = (
                            cos(p.get(..h).unwrap_or(&[]), q.get(..h).unwrap_or(&[])),
                            cos(p.get(h..).unwrap_or(&[]), q.get(h..).unwrap_or(&[])),
                        );
                        if c0.min(c1) < 0.9995 {
                            let _ = writeln!(
                                rep,
                                "    {n} layer {l} {} row {i}: first half cosine {c0:.6}, second half {c1:.6}",
                                if keys { "keys" } else { "values" }
                            );
                        }
                    }
                }
            }
            let _ = writeln!(rep, "  {n} layer {l}: keys and values max abs {la}, worst row cosine {lc:.7}");
            kv_abs = kv_abs.max(la);
            kv_cos = kv_cos.min(lc);
            if l == 0 {
                l0 = la;
            }
        }
        let _ = writeln!(
            rep,
            "{n} captured keys and values, layers 0..={last}, anchor and chunk ({} + {} tokens): layer 0 max abs {l0}; all layers max abs {kv_abs}, worst row cosine {kv_cos:.7}",
            a_ids.len(),
            c_ids.len()
        );
        // Control with no window: anchor and chunk encoded as one sequence by each assembly
        // (no window, no capture assembled across a window): the split's own rounding.
        let whole = [a_ids.clone(), c_ids.clone()].concat();
        let (wr, ws) = (
            encode_through(&reference, dev, None, &whole, last).map_err(|e| e.to_string())?,
            encode_through(&split, dev, None, &whole, last).map_err(|e| e.to_string())?,
        );
        let mut ctl = 1.0_f64;
        let mut ctl_at = (0, 0);
        for l in 0..=last {
            for keys in [true, false] {
                let pick = |c: &ChunkKv| (if keys { c.k_pre.get(l) } else { c.v.get(l) }).map(|v| f16(v)).unwrap_or_default();
                let (x, y) = (pick(&wr), pick(&ws));
                for (i, (p, q)) in x.chunks(row).zip(y.chunks(row)).enumerate() {
                    let c = cos(p, q);
                    if c < ctl {
                        ctl = c;
                        ctl_at = (l, i);
                    }
                }
            }
        }
        let _ = writeln!(
            rep,
            "{n} control, one sequence of {} tokens with no window: split against reference worst key or value row cosine {ctl:.7} (layer {}, row {})",
            whole.len(),
            ctl_at.0,
            ctl_at.1
        );
        // The key and value limit on this device: 0.999, or the split's own rounding measured by
        // the control when that is lower (the window and capture path may add nothing to it).
        let limit = c_tol.min(ctl);
        let _ = writeln!(
            rep,
            "{n} key and value row limit {limit:.7}: window path worst {kv_cos:.7} -> {}",
            if l0 == 0.0 && kv_cos >= limit { "within" } else { "OUTSIDE" }
        );
        if l0 != 0.0 || kv_cos < limit {
            fails.push(format!(
                "{n}: split keys or values differ from the reference (layer 0 max abs {l0}, worst row cosine {kv_cos}, limit {limit})"
            ));
        }
        // Drone mass, and every head through the last drone layer.
        let read = |m: &Model, w: &Arc<Window>, p: Probe| -> Result<Arc<Probe>, String> {
            let p = Arc::new(p);
            let stop = score_targets(m, last).map_err(|e| e.to_string())?;
            m.forward_at(
                dev,
                &q,
                w.len,
                &stop,
                &RunOpts { window: Some(Arc::clone(w)), probe: Some(Arc::clone(&p)), ..RunOpts::default() },
            )
            .map_err(|e| e.to_string())?;
            Ok(p)
        };
        let drone_mass = |p: &Probe, len: usize| -> Result<Vec<f32>, String> {
            let mut v = Vec::new();
            for &(l, h) in &drones.heads {
                let m = p.get(l).ok_or_else(|| format!("layer {l} not read"))?;
                v.extend(m.get(h * len..(h + 1) * len).ok_or_else(|| format!("layer {l} head {h} missing"))?.iter().map(|x| x * per));
            }
            Ok(v)
        };
        let (pr, ps, px) = (
            read(&reference, &rw, Probe::of_heads(&drones.heads))?,
            read(&split, &sw, Probe::of_heads(&drones.heads))?,
            read(&split, &rw, Probe::of_heads(&drones.heads))?,
        );
        let (mr, ms, mx) = (drone_mass(&pr, rw.len)?, drone_mass(&ps, sw.len)?, drone_mass(&px, rw.len)?);
        let (d_own, d_cross) = (worst(&mr, &ms), worst(&mr, &mx));
        let high = mr.iter().copied().fold(0.0_f32, f32::max);
        let _ = writeln!(
            rep,
            "{n} drone mass per question token ({} values, largest {high:.4}): split on its own window max abs {d_own}; split reading the reference window max abs {d_cross}",
            mr.len()
        );
        if mr.iter().all(|x| *x == 0.0) || d_own > p_tol || d_cross > p_tol {
            fails.push(format!("{n}: drone mass differs (own window {d_own}, reference window {d_cross}, tolerance {p_tol})"));
        }
        // The component that starts at head 8 holds its own record.
        let b = split
            .asm
            .nodes
            .iter()
            .find(|x| x.op == "score" && x.layer == Some(last) && x.heads.is_some_and(|h| h.0 > 0))
            .ok_or("no second attention component")?;
        let nm = ps.get_node(&b.id).ok_or("second component recorded nothing")?;
        if nm.heads.0 == 0 || nm.mass.len() != (nm.heads.1 - nm.heads.0) * nm.span || nm.span != sw.len {
            return Err(format!("{n}: second component record {:?} over {} values", nm.heads, nm.mass.len()));
        }
        let _ = writeln!(rep, "{n} second component {} recorded heads {}..{} over {} positions", b.id, nm.heads.0, nm.heads.1, nm.span);
        let layers: Vec<usize> = (0..=last).collect();
        let (ar, asp) = (read(&reference, &rw, Probe::new(layers.clone()))?, read(&split, &sw, Probe::new(layers.clone()))?);
        let mut all_worst = 0.0_f32;
        for &l in &layers {
            let (x, y) = (ar.get(l).ok_or("reference layer not read")?, asp.get(l).ok_or("split layer not read")?);
            if x.len() != y.len() || x.len() != d.heads * rw.len {
                return Err(format!("{n}: layer {l} mass layout {} against {}", y.len(), x.len()));
            }
            all_worst = all_worst.max(worst(&x, &y) * per);
        }
        let _ = writeln!(rep, "{n} every head, layers 0..={last}, per question token: max abs {all_worst}");
        if all_worst > p_tol {
            fails.push(format!("{n}: all-head mass differs by {all_worst}"));
        }
    }
    // The reference's own CPU against GPU difference (both pass the S6 reference parity): the
    // scale of device rounding, reported beside the split's difference.
    for (what, set) in [("reference", &refs), ("split", &splits)] {
        if let [(an, ra, rc), (bn, ga, gc)] = set.as_slice() {
            for l in 0..=last {
                let (mut la, mut lc) = (0.0_f32, 1.0_f64);
                for (x, y) in [(ra, ga), (rc, gc)] {
                    for keys in [true, false] {
                        let pick = |c: &ChunkKv| (if keys { c.k_pre.get(l) } else { c.v.get(l) }).map(|v| f16(v)).unwrap_or_default();
                        let (p, q) = (pick(x), pick(y));
                        la = la.max(worst(&p, &q));
                        lc = p.chunks(row).zip(q.chunks(row)).map(|(u, v)| cos(u, v)).fold(lc, f64::min);
                    }
                }
                let _ =
                    writeln!(rep, "  {what} {an} against {what} {bn}, layer {l}: keys and values max abs {la}, worst row cosine {lc:.7}");
            }
        }
    }
    pop.release_device();
    if fails.is_empty() { Ok(()) } else { Err(fails.join("; ")) }
}

#[cfg(test)]
mod job_filter_roles {
    use super::role_matches;

    #[test]
    fn roles_name_their_micromodels() {
        assert!(role_matches("embedder", "qwen3-embedding-0.6b"));
        assert!(role_matches("reranker", "qwen3-reranker-0.6b"));
        assert!(role_matches("writer", "qwen3-1.7b"));
        assert!(role_matches("reader", "qwen3-0.6b"));
        assert!(!role_matches("reader", "qwen3-embedding-0.6b"));
        assert!(!role_matches("reader", "qwen3-reranker-0.6b"));
        assert!(!role_matches("embedder", "qwen3-reranker-0.6b"));
        assert!(!role_matches("writer", "qwen3-0.6b"));
    }
}
