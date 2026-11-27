//! SET ASIDE 2026-11-10 (correction from the planning session on the direction of Luis Jake
//! Gabriel III: only the weights' ownership moves to one shared copy; the drone reading stays in
//! the tentacle's tip and is not moved into a Core Brain service). Compiled, checked, called by
//! nothing in the served path; kept for reference, not deleted.
//!
//! The reader MicroModel of the Core Brain (2026-11-10; direction of Luis Jake Gabriel III, 08:00
//! UTC, specification section 10: the weights are not placed in the tentacles; sensory processing
//! runs in parallel and flows upstream into the Core Brain, where the data is processed with the
//! weights). One loaded reader, one copy of the weights (the base population, run as quantized
//! packets through the layers the drone heads read), shared by every sensory stream.
//!
//! The tentacles prepare their windows and cut them into spans on the processor (no weights) and
//! send the spans upstream by reference (kept encodings, never re-encoded); this reader does the
//! drone attention-head reading of each span with the weights (`designs/drones.json` holds the
//! reader's head sets) and returns per head and per part the log-sum-exp rows of the posed and the
//! content-free read. Spans are read in groups under one hold of the device gate; the group size is
//! derived from the free device memory now and the bytes one span's key and value copy needs on
//! the device (fallback: one span at a time). The engine reads one window per forward pass, so a
//! group shares the gate and the resident weights, not one kernel launch; the spans per second are
//! recorded.
use crate::tentacle_pass;
use nmengn::kvstore::{self, ChunkKv, Window};
use nmengn::model::Model;
use nmengn::placement::Device;
use nmtype::Fault;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// One span sent upstream: which tentacle and span it is, its parts by reference (anchor first),
/// the heads to read it with and the posed and content-free texts.
#[derive(Clone, Debug)]
pub struct SpanRequest {
    pub tentacle: usize,
    pub span: usize,
    pub parts: Vec<Arc<ChunkKv>>,
    pub heads: Vec<(usize, usize)>,
    pub posed: String,
    pub null: String,
}

/// What one read returned: per head and per part (anchor first) the posed and content-free rows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpanRows {
    pub posed: Vec<f32>,
    pub null: Vec<f32>,
    pub parts: usize,
}
impl SpanRows {
    /// Per head, the reading of every part above the anchor (posed minus content-free, minus the
    /// anchor's).
    #[must_use]
    pub fn above(&self, heads: usize) -> Vec<Vec<f32>> {
        let m = self.parts;
        let at = |v: &[f32], h: usize, k: usize| v.get(h * m + k).copied().unwrap_or(0.0);
        (0..heads)
            .map(|h| {
                (1..m).map(|k| (at(&self.posed, h, k) - at(&self.null, h, k)) - (at(&self.posed, h, 0) - at(&self.null, h, 0))).collect()
            })
            .collect()
    }
}

/// Counters of the reader.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stats {
    pub reads: usize,
    pub spans: usize,
    pub groups: usize,
    pub ms: f64,
    /// The last group size used and how it was derived.
    pub group: usize,
    pub group_why: String,
}

/// The reader.
pub struct Reader {
    pub model: Arc<Model>,
    pub dev: Device,
    pub stop: Vec<(String, String)>,
    stats: Mutex<Stats>,
}
impl std::fmt::Debug for Reader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reader").field("dev", &self.dev).finish_non_exhaustive()
    }
}

static READER: OnceLock<Arc<Reader>> = OnceLock::new();

impl Reader {
    /// The Core Brain's reader over the base population (one per process).
    /// # Errors
    /// The model cannot be loaded, or the assembly has no scoring component on the last drone layer.
    pub fn shared(home: &Path) -> Result<Arc<Self>, Fault> {
        if let Some(r) = READER.get() {
            return Ok(Arc::clone(r));
        }
        let model = crate::neural::model(home)?;
        let dev = if model.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
        let stop = kvstore::score_targets(&model, tentacle_pass::drones(home).last_layer())?;
        let r = Arc::new(Self { model, dev, stop, stats: Mutex::new(Stats::default()) });
        Ok(Arc::clone(READER.get_or_init(|| r)))
    }
    /// The reader over `model`: the process's reader when it reads with that model (pointer
    /// identity), else a reader made for it.
    #[must_use]
    pub fn of(model: &Arc<Model>, dev: Device, stop: &[(String, String)]) -> Arc<Self> {
        if let Some(r) = READER.get().filter(|r| Arc::ptr_eq(&r.model, model)) {
            return Arc::clone(r);
        }
        let r = Arc::new(Self { model: Arc::clone(model), dev, stop: stop.to_vec(), stats: Mutex::new(Stats::default()) });
        Arc::clone(READER.get_or_init(|| r))
    }
    /// The counters so far.
    #[must_use]
    pub fn stats(&self) -> Stats {
        self.stats.lock().map(|s| s.clone()).unwrap_or_default()
    }
    /// One read without taking the device gate (the caller holds it): the posed and the content-free
    /// read of a window made of `parts` (anchor first).
    fn read_ungated(&self, parts: &[Arc<ChunkKv>], heads: &[(usize, usize)], posed: &str, null: &str) -> Result<SpanRows, Fault> {
        let w = Arc::new(Window::build(&self.model, parts)?);
        let (_, pc) = tentacle_pass::drone_read_logits(&self.model, self.dev, &w, &self.stop, heads, posed)?;
        let (_, nc) = tentacle_pass::drone_read_logits(&self.model, self.dev, &w, &self.stop, heads, null)?;
        w.release_device();
        Ok(SpanRows { posed: pc, null: nc, parts: w.parts.len() })
    }
    /// One read under the device gate (a tentacle core's re-read or answer).
    /// # Errors
    /// Engine faults.
    pub fn read(&self, parts: &[Arc<ChunkKv>], heads: &[(usize, usize)], posed: &str, null: &str) -> Result<SpanRows, Fault> {
        let t = Instant::now();
        let gate = (self.dev == Device::Gpu).then(|| nmengn::placement::gpu_gate().enter("Core Brain reader", true));
        let r = self.read_ungated(parts, heads, posed, null);
        drop(gate);
        if let Ok(mut s) = self.stats.lock() {
            s.reads += 2;
            s.ms += t.elapsed().as_secs_f64() * 1000.0;
        }
        r
    }
    /// The group size for spans of about `tokens` tokens: free device memory over the bytes one
    /// span's key and value copy takes on the device (every kept layer), at least 1; on the
    /// processor, every span in one group.
    #[must_use]
    pub fn group_size(&self, tokens: usize, spans: usize) -> (usize, String) {
        let d = self.model.pop.dims;
        let layers = self.stop.len().max(1).max(tentacle_pass::drone_last_layer() + 1);
        let per = tokens.max(1) * layers * 2 * d.kv_heads * d.head_dim * 2;
        match (self.dev, self.model.pop.gpu_free()) {
            (Device::Gpu, Some(free)) => {
                let reserve = nmengn::residency::get().reserve().0 as u64;
                let usable = free.saturating_sub(reserve);
                let k = usize::try_from(usable / per.max(1) as u64).unwrap_or(1).clamp(1, spans.max(1));
                (
                    k,
                    format!(
                        "{k} spans: free device memory {} MB less the measured reserve {} MB over {} MB per span of {tokens} tokens",
                        free >> 20,
                        reserve >> 20,
                        per >> 20
                    ),
                )
            }
            (Device::Gpu, None) => (1, "1 span (fallback: free device memory not read)".to_owned()),
            (Device::Cpu, _) => (spans.max(1), "every span in one group (processor)".to_owned()),
        }
    }
}

/// Spans of one window (tentacle side, no weights): consecutive chunks of `parts` (anchor first)
/// grouped so that a span holds at most `span` tokens (a longer chunk alone); indices into `parts`.
#[must_use]
pub fn cut(parts: &[Arc<ChunkKv>], span: usize) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut used = 0;
    for (i, c) in parts.iter().enumerate().skip(1) {
        if !cur.is_empty() && used + c.tokens > span {
            groups.push(std::mem::take(&mut cur));
            used = 0;
        }
        cur.push(i);
        used += c.tokens;
    }
    if !cur.is_empty() {
        groups.push(cur);
    }
    groups
}

/// Section check (processor only): the span cut on hand values (every chunk once, in order, no span
/// over the size unless a chunk alone is larger); `SpanRows::above` on hand values.
pub fn check() -> Result<(), String> {
    let mk = |t: usize| Arc::new(ChunkKv { tokens: t, ..ChunkKv::default() });
    let parts = vec![mk(5), mk(400), mk(700), mk(300), mk(2000), mk(100)];
    let g = cut(&parts, 1000);
    // 400 + 700 passes 1,000, so chunk 1 stands alone; 700 + 300 fits; 2,000 alone; then 100 (hand value corrected 2026-11-10: the earlier expectation put 400 and 700 in one span).
    if g != vec![vec![1], vec![2, 3], vec![4], vec![5]] {
        return Err(format!("span cut {g:?}"));
    }
    let r = SpanRows { posed: vec![1.0, 3.0, 2.0, 0.0, 1.0, 1.0], null: vec![0.5, 0.5, 0.5, 0.0, 0.0, 0.0], parts: 3 };
    let a = r.above(2);
    if a != vec![vec![2.0, 1.0], vec![1.0, 1.0]] {
        return Err(format!("rows above the anchor {a:?}"));
    }
    Ok(())
}
