//! Keys-only scan, first form (2026-11-10; tentacle core design
//! `docs/2026-11-10-neumor-design-tentacle-core-v1.md` section 2.2 and engine additions A6 and A7,
//! architecture by Luis Jake Gabriel III; specification `docs/2026-11-10-neumor-guide-how-it-works.md`
//! sections 2.3, 2.4 and 6, selective execution of the released weights). Three separable parts:
//! - `Keys`: the drone-key store for one population and drone set (`nmengn::drone::KeyStore`
//!   behind a lock), with its byte limit and bytes per token recorded, and the background pass
//!   that encodes the code base once through the drone layers and keeps only the drone-key
//!   packets. The pass enters the device gate per chunk as a background job, so a question (a
//!   foreground job) goes first; a later pass encodes only chunks whose content key it lacks
//!   (changed or new chunks) and drops packets of chunks that left the tree.
//! - `cal_rows`: one tentacle's raw-score table from its window's drone keys and the question
//!   path's queries: per drone head and per part (anchor first), the log-sum-exp of the raw scores
//!   over the part's positions averaged over the question tokens (`tentacle_pass::part_lse`), the
//!   same table the exact scan gives the calibrated score. `score_windows` runs one thread per
//!   tentacle.
//! - The pass's keys-only mode (`TentaclePass` with `Settings::keys`): the question path once per
//!   window for the posed and the content-free text, scoring on the processor, windows as outlines
//!   whose missing encodings a tentacle core's re-read fills on demand.
//!
//! The exact scan stays the default and the fallback; nothing here changes it.
use crate::tentacle_pass::part_lse;
use nmengn::drone::{self, KeyStore, Queries, RelevanceKeys};
use nmengn::kvstore::{self, Window};
use nmengn::model::Model;
use nmengn::placement::Device;
use nmtype::{Fault, FaultKind};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmapi.key_scan", FaultKind::Execution, m)
}

/// What the background pass has done.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PassStatus {
    /// A pass is running now.
    pub running: bool,
    /// Passes finished, and the substrate version of the last one started.
    pub passes: usize,
    pub version: u64,
    /// Chunks of the tree in the last pass, chunks already held, chunks encoded, tokens encoded.
    pub chunks: usize,
    pub held: usize,
    pub encoded: usize,
    pub tokens: usize,
    /// Packets dropped because their chunk left the tree or changed.
    pub dropped: usize,
    /// Milliseconds encoding, seconds of wall time of the last pass, seconds waited at the gate.
    pub enc_ms: f64,
    pub wall_s: f64,
    pub waited_s: f64,
    /// Faults (the anchored pass stops at the first; the intake pass skips the piece and goes on).
    pub faults: Vec<String>,
    /// Intake pass (2026-11-18, Phase C2): pieces encoded alone (no anchor), the longest single
    /// kernel (one piece's encoding) in milliseconds, and pieces skipped after a fault.
    pub alone: bool,
    pub kernel_max_ms: f64,
    pub skipped: usize,
    /// Segments encoded (each one kernel, T-069).
    pub segments: usize,
    /// Drone readings (2026-11-19, lane DR; `drone_dendrite::signals_on`): pieces read through
    /// layers 0 to k, pieces read lexically only after a fault, milliseconds of those passes.
    pub readings: usize,
    pub reading_faults: usize,
    pub reading_ms: f64,
}

/// The intake's target time for one kernel (master plan 3.2: about 50 ms, so a question waits
/// behind the background pass at most that long) and the segment length's bounds in tokens.
pub const KERNEL_MS: f64 = 50.0;
pub const SEGMENT: (usize, usize) = (64, 128);

/// A piece's request-independent signals as the drones took it in (`drone_windows::Keep`): its
/// kind, tokens, the share of distinct tokens (a low share marks repetitive or generated text, the
/// anomaly signal) and whether it is a document.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PieceSignals {
    pub kind: String,
    pub tokens: usize,
    pub distinct_share: f32,
    pub document: bool,
}

/// One piece for the intake pass: its content key, its identity across versions (path and first
/// line), its document text and its signals.
#[derive(Clone, Debug, Default)]
pub struct IntakePiece {
    pub key: String,
    pub identity: String,
    pub doc: String,
    pub signals: PieceSignals,
    /// The piece's file and own text (for its epistemic class, lane DR).
    pub path: String,
    pub text: String,
}

/// A piece's local memory (master plan 3.2): its last reading (content key, material version,
/// signals) and how many times its content changed since it was first read.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PieceMemory {
    pub key: String,
    pub version: u64,
    pub changes: u32,
    pub signals: PieceSignals,
    /// The drones' last reading of the piece's content (`drone_dendrite::Reading`: kind,
    /// epistemic class, standout flags, anomaly, confidence), made from its pass through layers 0
    /// to k when `drone_dendrite::signals_on`; cleared when the content changes.
    pub reading: Option<crate::drone_dendrite::Reading>,
}

/// The relevance-key store of one drone set (master plan 3.2: the tentacles' periphery data,
/// named drone keys until 2026-11-18).
#[derive(Debug)]
pub struct Keys {
    /// The drone heads read and the (layer, key-value head) pairs they use.
    pub heads: Vec<(usize, usize)>,
    pub pairs: Vec<(usize, usize)>,
    /// Bytes per token of a packet, derived from the pairs.
    pub bytes_per_token: usize,
    /// The packets are pieces encoded alone (the intake pass, Phase C2), not behind the anchor.
    pub alone: bool,
    store: Mutex<KeyStore>,
    status: Mutex<PassStatus>,
    /// Every piece's signals and local memory by identity (the intake pass).
    memory: Mutex<std::collections::BTreeMap<String, PieceMemory>>,
    /// The material's running background of layer-k states (the drone readings' standardisation).
    background: Mutex<crate::drone_dendrite::Background>,
}

impl Keys {
    /// A store for `heads` of a model with `group` query heads per key-value head and head width
    /// `head_dim`, holding at most `cap` bytes.
    #[must_use]
    pub fn new(heads: &[(usize, usize)], group: usize, head_dim: usize, cap: usize) -> Self {
        let pairs = drone::pairs(heads, group);
        let bytes_per_token = drone::bytes_per_token(&pairs, head_dim);
        Self {
            heads: heads.to_vec(),
            pairs,
            bytes_per_token,
            alone: false,
            store: Mutex::new(KeyStore::new(cap)),
            status: Mutex::new(PassStatus::default()),
            memory: Mutex::new(std::collections::BTreeMap::new()),
            background: Mutex::new(crate::drone_dendrite::Background::default()),
        }
    }
    /// A store for the intake pass (pieces encoded alone, Phase C2); as `new` otherwise.
    #[must_use]
    pub fn new_alone(heads: &[(usize, usize)], group: usize, head_dim: usize, cap: usize) -> Self {
        Self { alone: true, ..Self::new(heads, group, head_dim, cap) }
    }
    /// The packet of a chunk, marked as used.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<Arc<RelevanceKeys>> {
        self.store.lock().ok().and_then(|mut s| s.get(key))
    }
    /// True when the chunk's packet is held.
    #[must_use]
    pub fn has(&self, key: &str) -> bool {
        self.store.lock().is_ok_and(|s| s.contains(key))
    }
    /// Hold a packet.
    pub fn put(&self, key: &str, p: Arc<RelevanceKeys>) {
        if let Ok(mut s) = self.store.lock() {
            s.put(key, p);
        }
    }
    /// (bytes held, packets held, byte limit).
    #[must_use]
    pub fn held(&self) -> (usize, usize, usize) {
        self.store.lock().map(|s| (s.bytes(), s.len(), s.cap())).unwrap_or((0, 0, 0))
    }
    /// Change the byte limit; returns the packets dropped.
    pub fn set_cap(&self, cap: usize) -> usize {
        self.store.lock().map(|mut s| s.set_cap(cap)).unwrap_or(0)
    }
    /// The background pass's status.
    #[must_use]
    pub fn status(&self) -> PassStatus {
        self.status.lock().map(|s| s.clone()).unwrap_or_default()
    }
    /// The packet of one chunk, encoded now behind `anchor` through the deepest drone layer
    /// (the caller holds the device gate as it needs). Returns the packet and the full encoding.
    /// # Errors
    /// Engine faults.
    pub fn encode(
        &self,
        model: &Model,
        dev: Device,
        anchor: &Arc<Window>,
        ids: &[i32],
    ) -> Result<(Arc<RelevanceKeys>, Arc<kvstore::ChunkKv>), Fault> {
        let last = self.heads.iter().map(|p| p.0).max().unwrap_or(0);
        let kv = Arc::new(kvstore::encode_through(model, dev, Some(anchor), ids, last)?);
        let d = model.pop.dims;
        let p = Arc::new(RelevanceKeys::from_chunk(&kv, &self.pairs, d.kv_heads, d.head_dim)?);
        Ok((p, kv))
    }
    /// One pass over `docs` (chunk content key, document text), in order: packets for the chunks
    /// not held, each encoded under the device gate as a background job; packets of chunks not in
    /// `docs` dropped first. Synchronous (the background thread and the measurement call it).
    pub fn pass(&self, model: &Model, dev: Device, anchor: &Arc<Window>, docs: &[(String, String)], version: u64) {
        let started = Instant::now();
        let live: std::collections::BTreeSet<String> = docs.iter().map(|d| d.0.clone()).collect();
        let dropped = self.store.lock().map(|mut s| s.retain(&live)).unwrap_or(0);
        if let Ok(mut st) = self.status.lock() {
            *st = PassStatus { running: true, passes: st.passes, version, chunks: docs.len(), dropped, ..PassStatus::default() };
        }
        for (key, doc) in docs {
            if self.has(key) {
                if let Ok(mut st) = self.status.lock() {
                    st.held += 1;
                }
                continue;
            }
            let gate = (dev == Device::Gpu).then(|| nmengn::placement::gpu_gate().enter("drone-key pass (background)", false));
            let waited = gate.as_ref().map_or(0.0, |g| g.waited);
            let t = Instant::now();
            let r = model
                .pop
                .tokenizer
                .encode(doc)
                .map_err(|e| bad(e.to_string()))
                .and_then(|ids| self.encode(model, dev, anchor, &ids).map(|(p, _)| (p, ids.len())));
            drop(gate);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            match r {
                Ok((p, n)) => {
                    self.put(key, p);
                    if let Ok(mut st) = self.status.lock() {
                        st.encoded += 1;
                        st.tokens += n;
                        st.enc_ms += ms;
                        st.waited_s += waited;
                    }
                }
                Err(e) => {
                    if let Ok(mut st) = self.status.lock() {
                        st.faults.push(format!("{}: {}", key.get(..12).unwrap_or(key), e.message));
                    }
                    break;
                }
            }
        }
        if let Ok(mut st) = self.status.lock() {
            st.running = false;
            st.passes += 1;
            st.wall_s = started.elapsed().as_secs_f64();
        }
    }
    /// Start a background pass on its own thread unless one is running or the last finished pass
    /// was over the same substrate version. Returns whether one started.
    pub fn start(self: &Arc<Self>, model: Arc<Model>, dev: Device, anchor: Arc<Window>, docs: Vec<(String, String)>, version: u64) -> bool {
        {
            let Ok(mut st) = self.status.lock() else { return false };
            if st.running || (st.passes > 0 && st.version == version && st.faults.is_empty()) {
                return false;
            }
            st.running = true;
            st.version = version;
        }
        let me = Arc::clone(self);
        std::thread::spawn(move || me.pass(&model, dev, &anchor, &docs, version));
        true
    }
}

impl Keys {
    /// One piece's relevance keys, the piece encoded alone (positions from 0, no anchor) through
    /// the deepest relevance layer (master plan 3.2: the drone pass over layers 0 to k continued
    /// on the piece alone to L_r; the layer-k states are not kept). The caller holds the device
    /// gate as it needs.
    /// # Errors
    /// Engine faults.
    pub fn encode_alone(&self, model: &Model, dev: Device, ids: &[i32]) -> Result<Arc<RelevanceKeys>, Fault> {
        let last = self.heads.iter().map(|p| p.0).max().unwrap_or(0);
        let kv = kvstore::encode_through(model, dev, None, ids, last)?;
        let d = model.pop.dims;
        Ok(Arc::new(RelevanceKeys::from_chunk(&kv, &self.pairs, d.kv_heads, d.head_dim)?))
    }
    /// One piece's relevance keys encoded alone in segments (T-069, 2026-11-18): the piece's
    /// tokens cut into runs of `seg` tokens, each run encoded behind the runs before it (their
    /// keys and values as its prefix, positions continuing), so the keys equal the whole-piece
    /// encode within float tolerance while each device job is one segment. `each` is called around
    /// every segment (it enters the device gate and times the kernel); returns the packet and each
    /// segment's milliseconds.
    /// # Errors
    /// Engine faults.
    pub fn encode_segmented(
        &self,
        model: &Model,
        dev: Device,
        ids: &[i32],
        seg: usize,
        each: &mut dyn FnMut() -> Option<nmengn::placement::GateGuard>,
    ) -> Result<(Arc<RelevanceKeys>, Vec<f64>), Fault> {
        let last = self.heads.iter().map(|p| p.0).max().unwrap_or(0);
        let mut done: Vec<Arc<kvstore::ChunkKv>> = Vec::new();
        let mut times = Vec::new();
        for run in ids.chunks(seg.max(1)) {
            let prefix = if done.is_empty() { None } else { Some(Arc::new(Window::build(model, &done)?)) };
            let gate = each();
            let t = Instant::now();
            let kv = kvstore::encode_through(model, dev, prefix.as_ref(), run, last);
            drop(gate);
            times.push(t.elapsed().as_secs_f64() * 1000.0);
            done.push(Arc::new(kv?));
        }
        let d = model.pop.dims;
        let layers = last + 1;
        let whole = kvstore::ChunkKv {
            tokens: ids.len(),
            anchor: 0,
            k_pre: (0..layers).map(|l| done.iter().flat_map(|c| c.k_pre.get(l).map_or(&[][..], Vec::as_slice).iter().copied()).collect()).collect(),
            v: Vec::new(),
        };
        Ok((Arc::new(RelevanceKeys::from_chunk(&whole, &self.pairs, d.kv_heads, d.head_dim)?), times))
    }
    /// The intake pass (Phase C2, 2026-11-18): `pieces` in the given order. Every piece's signals
    /// and local memory are kept (its last reading and how often its content changed); each piece
    /// whose keys are not held is encoded alone in segments (`encode_segmented`), each segment one
    /// kernel at the lowest class (the device gate entered as a background job per segment, and on
    /// the processor the budget's background class), so a question waits behind the pass at most
    /// one segment; the segment length follows the measured time a token toward `KERNEL_MS`,
    /// within `SEGMENT`; a piece that faults is skipped and the pass goes on; packets of pieces
    /// not in `pieces` dropped first.
    pub fn intake(&self, model: &Model, dev: Device, pieces: &[IntakePiece], version: u64) {
        nmtype::budget::background(|| self.intake_now(model, dev, pieces, version));
    }
    fn intake_now(&self, model: &Model, dev: Device, pieces: &[IntakePiece], version: u64) {
        let started = Instant::now();
        let live: std::collections::BTreeSet<String> = pieces.iter().map(|d| d.key.clone()).collect();
        let dropped = self.store.lock().map(|mut s| s.retain(&live)).unwrap_or(0);
        if let Ok(mut st) = self.status.lock() {
            *st = PassStatus { running: true, passes: st.passes, version, chunks: pieces.len(), dropped, alone: true, ..PassStatus::default() };
        }
        self.remember(pieces, version);
        // Milliseconds a token, measured as the pass runs (first guess: the card's 0.67 ms).
        let mut per_token = 0.67_f64;
        let reading = crate::drone_dendrite::signals_on();
        for piece in pieces {
            // Shutdown (lane T1, 2026-11-19): a pass stops between pieces when the process is
            // ending (`stop_passes`), so no device job runs while the card is torn down (the
            // eight-set bench hung at exit with the pass mid-run).
            if STOP_PASSES.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            // A call in flight has the processor and the card: the pass waits between pieces
            // (lane T1, 2026-11-19: with the pass running beside calls the drones' depth read of a
            // call took 170 s on the Mini's tree, 1.6 s without it).
            while CALLS.load(std::sync::atomic::Ordering::Relaxed) > 0 && !STOP_PASSES.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::park_timeout(std::time::Duration::from_millis(10));
            }
            // Memory areas (2026-11-27): the keys held count in the `content` area, and the pass
            // stops when it is full (the pieces not reached stay unencoded; `nm_status` says so).
            nmengn::limits::set_area_used("content", "relevance_keys", self.held().0 as u64);
            if crate::drone_windows::content_full() {
                nmengn::limits::set_area_full("content", true);
                if let Ok(mut st) = self.status.lock() {
                    st.faults.push("stopped: the content memory area is full (limits.memory.content_mb)".to_owned());
                }
                break;
            }
            if reading {
                self.read_piece(model, dev, piece);
            }
            if self.has(&piece.key) {
                if let Ok(mut st) = self.status.lock() {
                    st.held += 1;
                }
                continue;
            }
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a small token count")]
            let seg = ((KERNEL_MS / per_token.max(1e-3)) as usize).clamp(SEGMENT.0, SEGMENT.1);
            let mut waited = 0.0;
            let mut each = || {
                let g = (dev == Device::Gpu).then(|| nmengn::placement::gpu_gate().enter("relevance-key intake (background)", false));
                waited += g.as_ref().map_or(0.0, |g| g.waited);
                g
            };
            let r = model
                .pop
                .tokenizer
                .encode(&piece.doc)
                .map_err(|e| bad(e.to_string()))
                .and_then(|ids| self.encode_segmented(model, dev, &ids, seg, &mut each).map(|(p, t)| (p, ids.len(), t)));
            let Ok(mut st) = self.status.lock() else { continue };
            st.waited_s += waited;
            match r {
                Ok((p, n, times)) => {
                    let ms: f64 = times.iter().sum();
                    st.kernel_max_ms = times.iter().copied().fold(st.kernel_max_ms, f64::max);
                    st.segments += times.len();
                    st.encoded += 1;
                    st.tokens += n;
                    st.enc_ms += ms;
                    drop(st);
                    #[expect(clippy::cast_precision_loss, reason = "a token count")]
                    if n > 0 {
                        per_token = ms / n as f64;
                    }
                    self.put(&piece.key, p);
                }
                Err(e) => {
                    st.skipped += 1;
                    st.faults.push(format!("{}: {}", piece.key.get(..12).unwrap_or(&piece.key), e.message));
                }
            }
        }
        nmengn::limits::set_area_used("content", "relevance_keys", self.held().0 as u64);
        if let Ok(mut st) = self.status.lock() {
            st.running = false;
            st.passes += 1;
            st.wall_s = started.elapsed().as_secs_f64();
        }
    }
    /// Keep every piece's signals and local memory (master plan 3.2): the reading of this version
    /// replaces the last; a piece whose content key differs from its last reading counts one
    /// change. Pieces gone from the material are forgotten.
    fn remember(&self, pieces: &[IntakePiece], version: u64) {
        let Ok(mut mem) = self.memory.lock() else { return };
        let live: std::collections::BTreeSet<&str> = pieces.iter().map(|p| p.identity.as_str()).collect();
        mem.retain(|k, _| live.contains(k.as_str()));
        for p in pieces {
            let e = mem.entry(p.identity.clone()).or_default();
            if !e.key.is_empty() && e.key != p.key {
                e.changes += 1;
                e.reading = None;
            }
            p.key.clone_into(&mut e.key);
            e.version = version;
            e.signals = p.signals.clone();
        }
    }
    /// The drones' reading of one piece (lane DR, 2026-11-19): unless its local memory already holds
    /// a reading of this content at this depth, the piece runs alone through layers 0 to k
    /// (`drone_dendrite::sense_states`, one device job at the lowest class) and its signals are
    /// kept in its local memory; after a fault it is read lexically only (cold) and counted.
    fn read_piece(&self, model: &Model, dev: Device, piece: &IntakePiece) {
        let k = crate::drone_dendrite::depth();
        let fresh = self.memory.lock().is_ok_and(|m| {
            m.get(&piece.identity).is_some_and(|e| e.key == piece.key && e.reading.as_ref().is_some_and(|r| r.sensed && r.depth == k))
        });
        if fresh {
            return;
        }
        let t = Instant::now();
        // One device job a piece holds the card until it ends: a piece over `READ_PIECE_MAX`
        // tokens is read cold (lexically), so a call never waits behind one long background job
        // (two calls of about 150 s on the Mini's tree, 2.4 s and 1.9 s alone; lane T1 2026-11-19).
        let ids = model.pop.tokenizer.encode(&piece.doc).map_err(|e| bad(e.to_string())).and_then(|v| {
            if v.len() > READ_PIECE_MAX { Err(bad(format!("{} tokens, over {READ_PIECE_MAX}: read cold", v.len()))) } else { Ok(v) }
        });
        let gate = (dev == Device::Gpu && ids.is_ok()).then(|| nmengn::placement::gpu_gate().enter("drone reading (background)", false));
        let local = ids.as_ref().map_err(Clone::clone).and_then(|ids| crate::drone_dendrite::sense_states(model, dev, ids, k));
        drop(gate);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let facts = crate::drone_dendrite::PieceFacts {
            path: &piece.path,
            text: &piece.text,
            kind: &piece.signals.kind,
            distinct_share: piece.signals.distinct_share,
            document: piece.signals.document,
            tokens: ids.as_ref().map_or(piece.signals.tokens, Vec::len),
        };
        let r = {
            let Ok(mut bg) = self.background.lock() else { return };
            crate::drone_dendrite::Reading::read(local.as_ref().ok(), &mut bg, k, &facts)
        };
        if let Ok(mut st) = self.status.lock() {
            if local.is_ok() {
                st.readings += 1;
            } else {
                st.reading_faults += 1;
            }
            st.reading_ms += ms;
            st.kernel_max_ms = st.kernel_max_ms.max(ms);
        }
        if let Ok(mut m) = self.memory.lock()
            && let Some(e) = m.get_mut(&piece.identity)
        {
            e.reading = Some(r);
        }
    }
    /// A piece's signals and local memory, by its identity (path and first line).
    #[must_use]
    pub fn piece(&self, identity: &str) -> Option<PieceMemory> {
        self.memory.lock().ok().and_then(|m| m.get(identity).cloned())
    }
    /// Start the intake pass on its own thread unless one is running or the last finished intake
    /// was over the same material version. Returns whether one started.
    pub fn start_intake(self: &Arc<Self>, model: Arc<Model>, dev: Device, pieces: Vec<IntakePiece>, version: u64) -> bool {
        {
            let Ok(mut st) = self.status.lock() else { return false };
            if st.running || (st.passes > 0 && st.version == version && st.alone && st.skipped == 0) {
                return false;
            }
            st.running = true;
            st.version = version;
        }
        let me = Arc::clone(self);
        std::thread::spawn(move || me.intake(&model, dev, &pieces, version));
        true
    }
    /// Bytes held per token held (0 with nothing held).
    #[must_use]
    pub fn bytes_per_token_held(&self) -> f64 {
        let (bytes, _, _) = self.held();
        let tokens: usize = self.store.lock().map(|s| s.tokens()).unwrap_or(0);
        #[expect(clippy::cast_precision_loss, reason = "a ratio for the record")]
        if tokens == 0 { 0.0 } else { bytes as f64 / tokens as f64 }
    }
}

/// The longest piece the background drone reading runs on the device (tokens); longer pieces are
/// read cold.
pub const READ_PIECE_MAX: usize = 2_048;
/// Calls in flight (`CallMark`): the background intake waits between pieces while any runs.
static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Held while a served call runs (`call_mark`); the intake pass waits meanwhile.
#[derive(Debug)]
pub struct CallMark(());
impl Drop for CallMark {
    fn drop(&mut self) {
        CALLS.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}
/// Mark a served call as running until the mark drops.
#[must_use]
pub fn call_mark() -> CallMark {
    CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    CallMark(())
}
/// Whether the background passes are stopping (shutdown, `stop_passes`).
#[must_use]
pub fn stopping() -> bool {
    STOP_PASSES.load(std::sync::atomic::Ordering::Relaxed)
}
/// Wait while any served call runs (a background pass yields between its steps), or until the
/// passes stop.
pub fn wait_for_calls() {
    while CALLS.load(std::sync::atomic::Ordering::Relaxed) > 0 && !stopping() {
        std::thread::park_timeout(std::time::Duration::from_millis(10));
    }
}
/// Set at shutdown: every intake pass stops at its next piece.
static STOP_PASSES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Stop the background intake passes of this process at their next piece and wait (at most
/// `wait`) until none runs on `keys`.
pub fn stop_passes(keys: Option<&Keys>, wait: std::time::Duration) {
    STOP_PASSES.store(true, std::sync::atomic::Ordering::Relaxed);
    let Some(k) = keys else { return };
    let t = Instant::now();
    while t.elapsed() < wait && k.status.lock().is_ok_and(|s| s.running) {
        std::thread::park_timeout(std::time::Duration::from_millis(20));
    }
}

/// The intake pass's switch: on by default since 2026-11-19 (lead's word: the drone kind is a design
/// part on a loose gate, measured in lane T1's final run); `NANO_INTAKE_KEYS=off` restores.
#[must_use]
pub fn intake_on() -> bool {
    !std::env::var("NANO_INTAKE_KEYS").is_ok_and(|v| v == "off")
}


/// The intake pieces of `sub` in `order`: content key, identity (path and first line), document
/// text, and the signals of the drones' record of each piece (`keep`, when it holds that piece).
#[must_use]
pub fn intake_pieces(sub: &nmsubs::Substrate, keep: Option<&crate::drone_windows::Keep>, order: &[usize]) -> Vec<IntakePiece> {
    order
        .iter()
        .filter_map(|&c| {
            let ch = sub.chunks.get(c)?;
            let f = sub.files.get(ch.file)?;
            let signals = keep.and_then(|k| k.record(c)).map_or_else(PieceSignals::default, |r| {
                #[expect(clippy::cast_precision_loss, reason = "a share")]
                let share = if r.tokens.is_empty() { 0.0 } else { r.distinct.len() as f32 / r.tokens.len() as f32 };
                PieceSignals { kind: ch.kind.to_owned(), tokens: r.tokens.len(), distinct_share: share, document: nmtype::is_document(&f.path) }
            });
            Some(IntakePiece {
                key: ch.key.clone(),
                identity: format!("{}:{}", f.path, ch.start),
                doc: nmsubs::dense::doc_text(&f.path, &ch.text),
                signals,
                path: f.path.clone(),
                text: ch.text.clone(),
            })
        })
        .collect()
}

/// The intake order (master plan 3.2): first the pieces the current questions' windows touch
/// (`touched`, in their order), then the pieces not yet held (new or changed: their content key is
/// not in the store), most related first (degree on the specific reference graph), then the rest
/// in material order (held pieces cost nothing). Indices into `sub.chunks`.
#[must_use]
pub fn intake_order(sub: &nmsubs::Substrate, keys: &Keys, touched: &[usize]) -> Vec<usize> {
    let n = sub.chunks.len();
    let mut seen = vec![false; n];
    let mut out = Vec::with_capacity(n);
    for &c in touched {
        if let Some(s) = seen.get_mut(c).filter(|s| !**s) {
            *s = true;
            out.push(c);
        }
    }
    let graph = sub.refs();
    let mut fresh: Vec<(usize, usize)> = (0..n)
        .filter(|c| !seen.get(*c).copied().unwrap_or(true) && sub.chunks.get(*c).is_some_and(|ch| !keys.has(&ch.key)))
        .map(|c| (graph.specific.degree(c), c))
        .collect();
    fresh.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    for (_, c) in fresh {
        if let Some(s) = seen.get_mut(c) {
            *s = true;
        }
        out.push(c);
    }
    out.extend((0..n).filter(|c| !seen.get(*c).copied().unwrap_or(true)));
    out
}

/// One tentacle's raw-score table from keys only: per head of `q.heads` (rows) and per part of
/// `parts_at` (columns, anchor first), the log-sum-exp of the raw scores over the part's
/// positions averaged over the question tokens. `packets` are the parts' drone keys in window
/// order; `len` the window length; `table` its rope table; `geometry` (head width, query heads per
/// key-value head).
/// # Errors
/// As `nmengn::drone::scan_keys`.
pub fn cal_rows(
    q: &Queries,
    packets: &[Arc<RelevanceKeys>],
    parts_at: &[(usize, usize)],
    len: usize,
    table: &[(f32, f32)],
    geometry: (usize, usize),
) -> Result<Vec<f32>, Fault> {
    let logits = drone::scan_keys(q, packets, len, table, geometry.0, geometry.1)?;
    let mut out = Vec::with_capacity(q.heads.len() * parts_at.len());
    for lg in &logits {
        out.extend(part_lse(lg, q.tokens, len, parts_at));
    }
    Ok(out)
}

/// One window's keys-only job: its packets (anchor first), part positions, length, and the
/// posed and content-free queries.
#[derive(Debug)]
pub struct Job {
    pub packets: Vec<Arc<RelevanceKeys>>,
    pub parts: Vec<(usize, usize)>,
    pub len: usize,
    pub posed: Queries,
    pub null: Queries,
}

/// (posed rows, content-free rows) of one window.
pub type Rows = (Vec<f32>, Vec<f32>);

/// Score every window on its own thread (one per tentacle): (posed rows, content-free rows) per
/// job, and the milliseconds of the whole step.
#[must_use]
pub fn score_windows(jobs: &[Job], head_dim: usize, group: usize, theta: f32) -> (Vec<Result<Rows, Fault>>, f64) {
    let t = Instant::now();
    // One partition per tentacle; the workers lent by the one processor budget (2026-11-18 Phase
    // C1), so tentacles beyond the thread count queue on it.
    let out = nmtype::budget::run(jobs.len(), |i| -> Result<Rows, Fault> {
        let j = jobs.get(i).ok_or_else(|| bad("a scoring job was lost"))?;
        let table = nmengn::cpu::rope_table(j.len, head_dim, theta);
        let a = cal_rows(&j.posed, &j.packets, &j.parts, j.len, &table, (head_dim, group))?;
        let b = cal_rows(&j.null, &j.packets, &j.parts, j.len, &table, (head_dim, group))?;
        Ok((a, b))
    })
    .into_iter()
    .map(|r| r.unwrap_or_else(|| Err(bad("a scoring thread stopped"))))
    .collect();
    (out, t.elapsed().as_secs_f64() * 1000.0)
}

/// Spearman rank correlation of two equally long score lists (average ranks for ties); `None`
/// below two values or with no spread.
#[must_use]
pub fn spearman(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() || a.len() < 2 {
        return None;
    }
    let rank = |v: &[f32]| -> Vec<f64> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|x, y| v.get(*x).copied().unwrap_or(0.0).total_cmp(&v.get(*y).copied().unwrap_or(0.0)));
        let mut r = vec![0.0; v.len()];
        let mut i = 0;
        while i < idx.len() {
            let mut j = i;
            while j + 1 < idx.len()
                && v.get(idx.get(j + 1).copied().unwrap_or(0)).map(|x| x.to_bits())
                    == v.get(idx.get(i).copied().unwrap_or(0)).map(|x| x.to_bits())
            {
                j += 1;
            }
            #[expect(clippy::cast_precision_loss, reason = "ranks are small")]
            let avg = (i + j) as f64 / 2.0;
            for k in i..=j {
                if let Some(x) = idx.get(k).and_then(|p| r.get_mut(*p)) {
                    *x = avg;
                }
            }
            i = j + 1;
        }
        r
    };
    let (ra, rb) = (rank(a), rank(b));
    #[expect(clippy::cast_precision_loss, reason = "counts are small")]
    let n = ra.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let cov: f64 = ra.iter().zip(&rb).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = ra.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = rb.iter().map(|y| (y - mb).powi(2)).sum();
    (va > 0.0 && vb > 0.0).then(|| cov / (va * vb).sqrt())
}

/// Section check (processor only): the store's bytes per token follow the drone set; Spearman on
/// hand values; `cal_rows` on hand packets equals the log-sum-exp computed directly; the
/// background pass rule (no second start while running or after a finished pass of the same
/// version). Values: `tests/section-checks/nmapi-key-scan/report.txt`.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    use std::fmt::Write as _;
    let mut rep = String::from("nmapi.key_scan (keys-only scan first form)\n");
    let r = key_check(&mut rep);
    let _ = writeln!(rep, "result: {}", r.as_ref().map_or_else(|e| format!("FAIL {e}"), |()| "ok".to_owned()));
    let dir = nmtype::check_dir("nmapi-key-scan").map_err(|e| e.to_string())?;
    std::fs::write(dir.join("report.txt"), rep).map_err(|e| e.to_string())?;
    r
}

fn key_check(rep: &mut String) -> Result<(), String> {
    use std::fmt::Write as _;
    let home = nmtype::nano_root();
    let set = crate::tentacle_pass::drones(&home);
    let k = Keys::new(&set.heads, 2, 128, 1 << 20);
    let _ = writeln!(rep, "(a) drone set {} heads: {} pairs, {} bytes per token", set.heads.len(), k.pairs.len(), k.bytes_per_token);
    if k.bytes_per_token != k.pairs.len() * 256 {
        return Err("bytes per token".into());
    }
    let s1 = spearman(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0]);
    let s2 = spearman(&[1.0, 2.0, 3.0, 4.0], &[4.0, 3.0, 2.0, 1.0]);
    let s3 = spearman(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]);
    let _ = writeln!(rep, "(b) spearman: same order {s1:?}; reversed {s2:?}; no spread {s3:?}");
    if s1.is_none_or(|x| (x - 1.0).abs() > 1e-12) || s2.is_none_or(|x| (x + 1.0).abs() > 1e-12) || s3.is_some() {
        return Err("spearman on hand values".into());
    }
    // cal_rows on hand packets: one pair, head width 4, two parts.
    let d = 4;
    let pk = |vals: &[f32]| {
        Arc::new(RelevanceKeys {
            tokens: vals.len() / d,
            pairs: vec![(0, 0)],
            rows: vec![vals.iter().map(|x| nmengn::half::bits16(*x)).collect()],
        })
    };
    let packets = vec![pk(&[0.5, -0.25, 1.0, 0.0]), pk(&[1.0, 0.0, 0.0, 0.5, -1.0, 0.5, 0.25, 0.0])];
    let parts = vec![(0, 1), (1, 2)];
    let q = Queries { tokens: 2, heads: vec![(0, 0)], q: vec![vec![1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, -1.0]] };
    let table = nmengn::cpu::rope_table(3, d, 1_000_000.0);
    let rows = cal_rows(&q, &packets, &parts, 3, &table, (d, 2)).map_err(|e| e.to_string())?;
    let logits = drone::scan_keys(&q, &packets, 3, &table, d, 2).map_err(|e| e.to_string())?;
    let want = part_lse(logits.first().ok_or("no logits")?, 2, 3, &parts);
    let _ = writeln!(rep, "(c) cal_rows on hand packets {rows:?} equal the direct log-sum-exp {want:?}: {}", rows == want);
    if rows != want || rows.iter().any(|x| !x.is_finite()) {
        return Err("cal_rows".into());
    }
    // Background pass rule.
    let k = Arc::new(Keys::new(&set.heads, 2, 128, 1 << 20));
    if let Ok(mut st) = k.status.lock() {
        st.passes = 1;
        st.version = 7;
    }
    let again = k.status().passes == 1 && k.status().version == 7;
    let _ = writeln!(rep, "(d) a finished pass over version 7 is recorded: {again}");
    if !again {
        return Err("pass status".into());
    }
    Ok(())
}
