//! Encoded context passed by reference (E1, 2026-11-10; design
//! `docs/2026-11-10-neumor-design-context-architecture.md`, Luis Jake Gabriel III): code is
//! encoded once through the released Qwen3 layers and its keys and values are kept in RAM (and
//! on the GPU when a window runs there), never decoded to text and re-encoded along the way.
//! - `ChunkKv`: one chunk's keys (before rope) and values for every layer, binary16 bits,
//!   captured while the chunk is encoded after an anchor (Star Attention's anchor block,
//!   arXiv 2411.17116). Keys are kept before rope so a chunk can be placed at any position by
//!   the engine's own rope, exactly.
//! - `Window`: a read-only key and value prefix over chunks placed side by side; it holds the
//!   chunks by reference and owns only a rope table of its positions (E3.1, 2026-11-09: keys
//!   get their window position's rope as attention reads them, so no placed copy exists); any
//!   number of questions (tentacles) read it by reference (`RunOpts::window`) without copying.
//! - `Capture` records keys and values while encoding; `Probe` records per-head attention mass
//!   on a window (the drone reading).

use crate::gpu::{DevT, Ops};
use crate::half::bits16;
use crate::model::{Model, RunOpts};
use crate::placement::Device;
use nmtype::{Fault, FaultKind};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmengn.kvstore", FaultKind::Execution, m)
}

/// The drone tips of the released Qwen3-0.6B: 16 (layer, head) pairs chosen on the CX1
/// development half by the E2 runbook's rule (`runs/2026-11-10-neumor-note-e2-head-probe.md`),
/// all in layers 8 to 17, so a drone read can stop after layer 17.
pub const DRONE_HEADS_QWEN3_0_6B: [(usize, usize); 16] = [
    (13, 6),
    (14, 6),
    (13, 14),
    (16, 8),
    (13, 15),
    (14, 14),
    (14, 13),
    (17, 1),
    (13, 7),
    (16, 13),
    (8, 4),
    (17, 12),
    (12, 0),
    (11, 11),
    (17, 13),
    (17, 11),
];

/// The drone reader sets as data (2026-11-10, step P0): `designs/drones.json`, one default set
/// per population and optional sets per tentacle (by lens name). `DRONE_HEADS_QWEN3_0_6B`
/// stays as the stated fallback for `qwen3-0.6b` (and what the locked E3.3 and G1 runners
/// read).
pub const DRONE_FILE: &str = "designs/drones.json";
/// The population the fallback constant belongs to.
pub const DRONE_FALLBACK_POPULATION: &str = "qwen3-0.6b";

/// One population's drone readers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DroneSet {
    pub population: String,
    /// The default (layer, head) pairs, global head numbers, in the file's order.
    pub heads: Vec<(usize, usize)>,
    /// Sets of named tentacles (lens names), used instead of the default for that tentacle.
    pub tentacles: BTreeMap<String, Vec<(usize, usize)>>,
    /// Where the set came from: the file, or the fallback constant and why.
    pub source: String,
    /// True when read from the file.
    pub from_file: bool,
    /// Version 2 (2026-11-10, tentacle core A8): how the set reads (`exact`: the question posed
    /// after the window; `scan`: keys only, not built yet), the record its selection came from,
    /// and the verification set the core's re-read uses (empty: the default set).
    pub read_mode: String,
    pub selection: String,
    pub verification: Vec<(usize, usize)>,
}
impl DroneSet {
    /// The set a tentacle reads with: its own when the file names one, else the default.
    #[must_use]
    pub fn for_tentacle(&self, name: &str) -> &[(usize, usize)] {
        self.tentacles.get(name).map_or(self.heads.as_slice(), Vec::as_slice)
    }
    /// The highest layer any set reads (chunks are encoded and kept through it).
    #[must_use]
    pub fn last_layer(&self) -> usize {
        self.heads.iter().chain(self.tentacles.values().flatten()).map(|p| p.0).max().unwrap_or(0)
    }
    /// The verification set (the core's re-read): its own when the file names one, else the
    /// default set.
    #[must_use]
    pub fn verification_set(&self) -> &[(usize, usize)] {
        if self.verification.is_empty() { &self.heads } else { &self.verification }
    }
}

/// Parse and validate `designs/drones.json`: every population names a weight folder under
/// `home` and a non-empty default set; every pair is `[layer, head]` inside that population's
/// layer and head counts (read from its weight card), with no pair twice in one set; tentacle
/// sets follow the same rules. Returns the sets by population.
/// # Errors
/// The first rule broken, named.
pub fn drone_sets(home: &std::path::Path, v: &nmjson::Value) -> Result<BTreeMap<String, DroneSet>, String> {
    use nmjson::Value;
    let pops = match v.get("populations") {
        Ok(Value::Object(m)) if !m.is_empty() => m,
        _ => return Err(format!("{DRONE_FILE}: \"populations\" must be a non-empty object")),
    };
    let mut out = BTreeMap::new();
    for (id, p) in pops {
        let card = nmnpcs::WeightCard::load(home, id).map_err(|e| format!("{DRONE_FILE}: population {id}: {e}"))?;
        let (layers, heads) = (
            card.int("num_hidden_layers").map_err(|e| format!("{DRONE_FILE}: {id}: {e}"))?,
            card.int("num_attention_heads").map_err(|e| format!("{DRONE_FILE}: {id}: {e}"))?,
        );
        let pairs = |what: &str, x: Result<&Value, nmjson::Error>| -> Result<Vec<(usize, usize)>, String> {
            let a = x.and_then(Value::array).map_err(|e| format!("{DRONE_FILE}: {id}: {what}: {e}"))?;
            if a.is_empty() {
                return Err(format!("{DRONE_FILE}: {id}: {what} is empty"));
            }
            let mut got: Vec<(usize, usize)> = Vec::with_capacity(a.len());
            for (i, e) in a.iter().enumerate() {
                let n =
                    |j: usize| e.array().ok().and_then(|q| q.get(j)).and_then(|x| x.integer().ok()).and_then(|x| usize::try_from(x).ok());
                let (Some(l), Some(h), true) = (n(0), n(1), e.array().is_ok_and(|q| q.len() == 2)) else {
                    return Err(format!("{DRONE_FILE}: {id}: {what} entry {i} is not [layer, head]"));
                };
                if l >= layers || h >= heads {
                    return Err(format!("{DRONE_FILE}: {id}: {what} entry {i} [{l}, {h}] is outside {layers} layers and {heads} heads"));
                }
                if got.contains(&(l, h)) {
                    return Err(format!("{DRONE_FILE}: {id}: {what} lists [{l}, {h}] twice"));
                }
                got.push((l, h));
            }
            Ok(got)
        };
        let default = pairs("default", p.get("default"))?;
        // Version 2 fields (tentacle core A8): optional, checked when present.
        let read_mode = match p.get("read_mode") {
            Ok(Value::String(m)) if m == "exact" || m == "scan" => m.clone(),
            Ok(other) => return Err(format!("{DRONE_FILE}: {id}: read_mode must be \"exact\" or \"scan\", got {other:?}")),
            Err(_) => "exact".to_owned(),
        };
        let selection = match p.get("source") {
            Ok(Value::String(s)) => s.clone(),
            _ => String::new(),
        };
        let verification = match p.get("verification") {
            Ok(Value::String(s)) if s == "default" => Vec::new(),
            Ok(Value::Array(_)) => pairs("verification", p.get("verification"))?,
            Ok(other) => {
                return Err(format!("{DRONE_FILE}: {id}: verification must be \"default\" or a list of [layer, head], got {other:?}"));
            }
            Err(_) => Vec::new(),
        };
        let mut tentacles = BTreeMap::new();
        match p.get("tentacles") {
            Ok(Value::Object(m)) => {
                for (name, s) in m {
                    tentacles.insert(name.clone(), pairs(&format!("tentacle {name}"), Ok(s))?);
                }
            }
            Ok(_) => return Err(format!("{DRONE_FILE}: {id}: \"tentacles\" must be an object")),
            Err(_) => {}
        }
        out.insert(
            id.clone(),
            DroneSet {
                population: id.clone(),
                heads: default,
                tentacles,
                source: format!("{DRONE_FILE} ({id})"),
                from_file: true,
                read_mode,
                selection,
                verification,
            },
        );
    }
    Ok(out)
}

/// The drone set of `population`: from `designs/drones.json` under `home`; for `qwen3-0.6b`
/// the fallback constant when the file is missing, invalid or has no entry for it (the reason
/// is kept in `source`).
/// # Errors
/// Another population with no valid entry in the file.
pub fn drone_set(home: &std::path::Path, population: &str) -> Result<DroneSet, Fault> {
    let read = std::fs::read(home.join(DRONE_FILE))
        .map_err(|e| format!("{DRONE_FILE}: {e}"))
        .and_then(|b| nmjson::parse(&b).map_err(|e| format!("{DRONE_FILE}: {e}")))
        .and_then(|v| drone_sets(home, &v));
    let why = match read {
        Ok(mut m) => match m.remove(population) {
            Some(s) => return Ok(s),
            None => format!("{DRONE_FILE} has no entry for {population}"),
        },
        Err(e) => e,
    };
    if population == DRONE_FALLBACK_POPULATION {
        return Ok(DroneSet {
            population: population.to_owned(),
            heads: DRONE_HEADS_QWEN3_0_6B.to_vec(),
            tentacles: BTreeMap::new(),
            source: format!("fallback constant DRONE_HEADS_QWEN3_0_6B ({why})"),
            from_file: false,
            read_mode: "exact".to_owned(),
            ..DroneSet::default()
        });
    }
    Err(bad(format!("no drone set for {population}: {why}")))
}

/// Keys before rope and values of the new tokens, per layer, recorded while encoding.
/// 2026-11-10 (step P0): an attention component records only its own key-value heads
/// (`keys_at`, `values_at`), so an assembly with more than one attention component per layer
/// fills one layer's row from all of them; `uncovered` names heads recorded other than once,
/// which `encode_through` refuses.
#[derive(Debug, Default)]
pub struct Capture {
    pub k_pre: Mutex<BTreeMap<usize, Vec<u16>>>,
    pub v: Mutex<BTreeMap<usize, Vec<u16>>>,
    /// Per layer, (key, value) recordings per key-value head (`keys_at`, `values_at` only).
    cover: Mutex<BTreeMap<usize, Marks>>,
}
/// Recordings per key-value head of one layer: (keys, values).
type Marks = (Vec<u8>, Vec<u8>);
impl Capture {
    /// Record a layer's keys (before rope), every key-value head.
    pub fn keys(&self, layer: usize, k: &[f32]) {
        if let Ok(mut m) = self.k_pre.lock() {
            m.insert(layer, k.iter().map(|x| bits16(*x)).collect());
        }
    }
    /// Record a layer's values, every key-value head.
    pub fn values(&self, layer: usize, v: &[f32]) {
        if let Ok(mut m) = self.v.lock() {
            m.insert(layer, v.iter().map(|x| bits16(*x)).collect());
        }
    }
    /// Record keys (before rope) of key-value heads `kv.0..kv.1` of `total`, `head` values
    /// each, into the layer's row of every head. A component over every head writes the row
    /// exactly as `keys` does.
    pub fn keys_at(&self, layer: usize, kv: (usize, usize), total: usize, head: usize, k: &[f32]) {
        self.put(true, layer, kv, total, head, k);
    }
    /// `keys_at` for values.
    pub fn values_at(&self, layer: usize, kv: (usize, usize), total: usize, head: usize, v: &[f32]) {
        self.put(false, layer, kv, total, head, v);
    }
    #[expect(clippy::too_many_arguments, reason = "a layer, a head range and its geometry name one recording")]
    fn put(&self, keys: bool, layer: usize, kv: (usize, usize), total: usize, head: usize, x: &[f32]) {
        let store = if keys { &self.k_pre } else { &self.v };
        if let Ok(mut c) = self.cover.lock() {
            let e = c.entry(layer).or_insert_with(|| (vec![0; total], vec![0; total]));
            let marks = if keys { &mut e.0 } else { &mut e.1 };
            for g in kv.0..kv.1.min(total) {
                if let Some(m) = marks.get_mut(g) {
                    *m = m.saturating_add(1);
                }
            }
        }
        let Ok(mut m) = store.lock() else { return };
        if kv == (0, total) {
            m.insert(layer, x.iter().map(|v| bits16(*v)).collect());
            return;
        }
        let (part, row) = ((kv.1 - kv.0) * head, total * head);
        let tokens = x.len() / part.max(1);
        let dst = m.entry(layer).or_insert_with(|| vec![0; tokens * row]);
        if dst.len() != tokens * row {
            *dst = vec![0; tokens * row];
        }
        for t in 0..tokens {
            if let (Some(d), Some(s)) = (dst.get_mut(t * row + kv.0 * head..t * row + kv.1 * head), x.get(t * part..(t + 1) * part)) {
                for (a, b) in d.iter_mut().zip(s) {
                    *a = bits16(*b);
                }
            }
        }
    }
    /// `keys_at` for binary16 bits as the device holds them (step P2: the GPU capture downloads
    /// u16 rows and stores them as they are; the old path widened them to f32 and narrowed them
    /// back, which gives the same bits for every non-NaN value, `check_capture16`).
    pub fn keys_at16(&self, layer: usize, kv: (usize, usize), total: usize, head: usize, k: Vec<u16>) {
        self.put16(true, layer, kv, total, head, k);
    }
    /// `keys_at16` for values.
    pub fn values_at16(&self, layer: usize, kv: (usize, usize), total: usize, head: usize, v: Vec<u16>) {
        self.put16(false, layer, kv, total, head, v);
    }
    #[expect(clippy::too_many_arguments, reason = "a layer, a head range and its geometry name one recording")]
    fn put16(&self, keys: bool, layer: usize, kv: (usize, usize), total: usize, head: usize, x: Vec<u16>) {
        let store = if keys { &self.k_pre } else { &self.v };
        if let Ok(mut c) = self.cover.lock() {
            let e = c.entry(layer).or_insert_with(|| (vec![0; total], vec![0; total]));
            let marks = if keys { &mut e.0 } else { &mut e.1 };
            for g in kv.0..kv.1.min(total) {
                if let Some(m) = marks.get_mut(g) {
                    *m = m.saturating_add(1);
                }
            }
        }
        let Ok(mut m) = store.lock() else { return };
        if kv == (0, total) {
            // Every head: the downloaded rows are the stored rows; moved in, not copied.
            m.insert(layer, x);
            return;
        }
        let (part, row) = ((kv.1 - kv.0) * head, total * head);
        let tokens = x.len() / part.max(1);
        let dst = m.entry(layer).or_insert_with(|| vec![0; tokens * row]);
        if dst.len() != tokens * row {
            *dst = vec![0; tokens * row];
        }
        for t in 0..tokens {
            if let (Some(d), Some(s)) = (dst.get_mut(t * row + kv.0 * head..t * row + kv.1 * head), x.get(t * part..(t + 1) * part)) {
                d.copy_from_slice(s);
            }
        }
    }
    /// Key-value heads of `layer` whose keys or values were not recorded exactly once
    /// (missing, or recorded by two components), when components recorded their own heads;
    /// `None` when the layer was recorded whole (`keys`, `values`) or not at all.
    #[must_use]
    pub fn uncovered(&self, layer: usize) -> Option<(Vec<usize>, Vec<usize>)> {
        let c = self.cover.lock().ok()?;
        let (k, v) = c.get(&layer)?;
        let odd = |m: &[u8]| m.iter().enumerate().filter(|x| *x.1 != 1).map(|x| x.0).collect::<Vec<_>>();
        Some((odd(k), odd(v)))
    }
}

/// Relevance-head reads of packed groups (lane S, 2026-11-18, for the tentacles' integration reads,
/// T-074): one `Probe` per sequence of `RunOpts::segments`, and how many of its last rows query
/// (the group's question tokens). At every probed layer each group's question rows are read against
/// the shared window (when given) and the group's own earlier rows, exactly as a plain read of that
/// question over a window of the window and the group's other rows records them (`Probe::record`
/// and, with `with_logits`, `record_logits`; span = window length plus the group's rows before
/// its question). See `read_groups`.
#[derive(Debug, Default)]
pub struct SegmentProbe {
    pub probes: Vec<Arc<Probe>>,
    pub rows: Vec<usize>,
}

/// The integration reads of several groups in one packed card run (lane S, 2026-11-18, T-074):
/// each group is its pieces' tokens then its question's tokens, one sequence of
/// `RunOpts::segments` (attention block-diagonal by group, positions restarting after `window`,
/// which every group reads as its first positions); the run stops at `stop` (the score components
/// of the last read layer, `score_targets`). Returns one probe per group, filled as
/// `Probe::of_heads(heads).with_logits()` would be by a plain read of that group's question over
/// a window of `window` and the group's pieces: `probe.get(layer)` gives mass `[heads, span]`,
/// `probe.logits(layer, head)` the raw scores `[question tokens, span]` and their log normalisers,
/// with span = window length plus the group's piece tokens. The pieces' boundaries are the caller's
/// (window parts first, then each piece's token count). On the card only.
/// # Errors
/// Engine faults, an empty group or question, or a device other than the card.
pub fn read_groups(
    m: &Model,
    window: Option<&Arc<Window>>,
    stop: &[(String, String)],
    heads: &[(usize, usize)],
    groups: &[(Vec<i32>, Vec<i32>)],
) -> Result<Vec<Arc<Probe>>, Fault> {
    if groups.is_empty() || groups.iter().any(|(p, q)| p.is_empty() || q.is_empty()) {
        return Err(bad("read_groups: every group needs pieces and a question"));
    }
    let probes: Vec<Arc<Probe>> = groups.iter().map(|_| Arc::new(Probe::of_heads(heads).with_logits())).collect();
    let rows: Vec<usize> = groups.iter().map(|(_, q)| q.len()).collect();
    let lens: Vec<usize> = groups.iter().map(|(p, q)| p.len() + q.len()).collect();
    let packed: Vec<i32> = groups.iter().flat_map(|(p, q)| p.iter().chain(q).copied()).collect();
    let opts = RunOpts {
        window: window.map(Arc::clone),
        segments: Some(Arc::new(lens)),
        segment_probe: Some(Arc::new(SegmentProbe { probes: probes.clone(), rows })),
        device_targets: true,
        ..RunOpts::default()
    };
    m.forward_at(Device::Gpu, &packed, window.map_or(0, |w| w.len), stop, &opts)?;
    Ok(probes)
}

/// Label reads of packed groups (lane S, 2026-11-19, for the tentacles' deep check): each group is
/// (its span's tokens, its tails' tokens); every span runs once as one sequence (after `window`,
/// when given) and each tail as a sequence that reads the window, then its group's span
/// (`RunOpts::segment_bodies`), then itself, never another tail or group; the output head reads
/// only `rows` (for example the yes and no tokens) at each tail's last token. Returns
/// `[group][tail]` rows of `rows.len()` logits, the values a separate read of span then tail gives
/// within the card's rounding (`joint_read::pass_with`'s label tails over its encoding). One card
/// run for every group; on the card only.
/// # Errors
/// Engine faults, an empty span or tail, or no rows.
pub fn read_group_labels(m: &Model, window: Option<&Arc<Window>>, groups: &[(Vec<i32>, Vec<Vec<i32>>)], rows: &[usize]) -> Result<Vec<Vec<Vec<f32>>>, Fault> {
    let _job = crate::chanrec::job("joint");
    if rows.is_empty() || groups.is_empty() || groups.iter().any(|(s, ts)| s.is_empty() || ts.is_empty() || ts.iter().any(Vec::is_empty)) {
        return Err(bad("read_group_labels: every group needs a span and tails, and the head rows"));
    }
    let (mut lens, mut bodies, mut packed, mut slot) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (gi, (span, tails)) in groups.iter().enumerate() {
        let body = lens.len();
        lens.push(span.len());
        bodies.push(None);
        slot.push(None);
        packed.extend_from_slice(span);
        for (ti, t) in tails.iter().enumerate() {
            lens.push(t.len());
            bodies.push(Some(body));
            slot.push(Some((gi, ti)));
            packed.extend_from_slice(t);
        }
    }
    let target = (m.role_node("head")?, "logits".to_owned());
    let opts = RunOpts {
        window: window.map(Arc::clone),
        segments: Some(Arc::new(lens)),
        segment_bodies: Some(Arc::new(bodies)),
        head_rows: Arc::new(rows.to_vec()),
        ..RunOpts::default()
    };
    let out = m.forward_at(Device::Gpu, &packed, window.map_or(0, |w| w.len), std::slice::from_ref(&target), &opts)?;
    let v = crate::model::values(out.get(&target).ok_or_else(|| bad("no logits"))?)?;
    let w = rows.len();
    let mut res: Vec<Vec<Vec<f32>>> = groups.iter().map(|(_, ts)| vec![Vec::new(); ts.len()]).collect();
    for (k, s) in slot.iter().enumerate() {
        if let Some((gi, ti)) = s
            && let Some(dst) = res.get_mut(*gi).and_then(|g| g.get_mut(*ti))
        {
            *dst = v.get(k * w..(k + 1) * w).ok_or_else(|| bad("short logits"))?.to_vec();
        }
    }
    Ok(res)
}

/// `read_groups` with several questions over each group's pieces (lane S package 5, 2026-11-18):
/// each group is (its pieces' tokens, its questions' tokens); the pieces run once as one sequence,
/// and each question as a tail sequence that reads the window, then that group's pieces
/// (`RunOpts::segment_bodies`), then its own tokens, never another question. Returns, per group,
/// one probe per question, filled as `read_groups` fills a probe for that (pieces, question) pair
/// (the same span: window length plus the group's piece tokens). The tentacles' posed and
/// content-free reads cost one pass over the pieces instead of two. On the card only.
/// # Errors
/// As `read_groups`, or a group with no question.
pub fn read_groups_tails(
    m: &Model,
    window: Option<&Arc<Window>>,
    stop: &[(String, String)],
    heads: &[(usize, usize)],
    groups: &[(Vec<i32>, Vec<Vec<i32>>)],
) -> Result<Vec<Vec<Arc<Probe>>>, Fault> {
    if groups.is_empty() || groups.iter().any(|(p, qs)| p.is_empty() || qs.is_empty() || qs.iter().any(Vec::is_empty)) {
        return Err(bad("read_groups_tails: every group needs pieces and at least one question"));
    }
    let (mut lens, mut bodies, mut rows, mut packed, mut probes, mut out) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (pieces, questions) in groups {
        let body = lens.len();
        lens.push(pieces.len());
        bodies.push(None);
        rows.push(0);
        probes.push(Arc::new(Probe::of_heads(heads).with_logits()));
        packed.extend_from_slice(pieces);
        let mut mine = Vec::with_capacity(questions.len());
        for q in questions {
            let p = Arc::new(Probe::of_heads(heads).with_logits());
            lens.push(q.len());
            bodies.push(Some(body));
            rows.push(q.len());
            probes.push(Arc::clone(&p));
            packed.extend_from_slice(q);
            mine.push(p);
        }
        out.push(mine);
    }
    let opts = RunOpts {
        window: window.map(Arc::clone),
        segments: Some(Arc::new(lens)),
        segment_bodies: Some(Arc::new(bodies)),
        segment_probe: Some(Arc::new(SegmentProbe { probes, rows })),
        device_targets: true,
        ..RunOpts::default()
    };
    m.forward_at(Device::Gpu, &packed, window.map_or(0, |w| w.len), stop, &opts)?;
    Ok(out)
}

/// One group of `read_group_parts`: its pieces' tokens, its questions' tokens (for example the
/// posed and the content-free question), the heads `(layer, global head)` it reads, and its parts
/// `(first position, positions)` in the span (window positions first, then its pieces').
#[derive(Clone, Debug, Default)]
pub struct PartGroup {
    pub pieces: Vec<i32>,
    pub questions: Vec<Vec<i32>>,
    pub heads: Vec<(usize, usize)>,
    pub parts: Vec<(usize, usize)>,
}

/// Part scores of packed groups reduced on the card (lane S package 5, 2026-11-18): for each tail
/// sequence of `RunOpts::segments` (`tails[i] = Some((group, question))`), at each of its group's
/// heads, per part the log-sum-exp of the question rows' raw scores over the part's positions,
/// averaged over the rows (`tentacle_pass::part_lse` of the processor read). `out[group][question]`
/// is `[heads x parts]` in the group's head order.
#[derive(Debug, Default)]
pub struct SegmentParts {
    pub tails: Vec<Option<(usize, usize)>>,
    /// Per group: its heads `(layer, global head)` and its parts `(first position, positions)`.
    pub groups: Vec<GroupShape>,
    pub out: Mutex<Vec<Vec<Vec<f32>>>>,
}
/// A group's heads and parts (`SegmentParts::groups`).
pub type GroupShape = (Vec<(usize, usize)>, Vec<(usize, usize)>);

/// The tentacles' integration reads as part scores, in one packed card run (lane S package 5,
/// 2026-11-18, T-074): each group's pieces run once (after the shared `window`), each of its
/// questions as a tail over them (`RunOpts::segment_bodies`); at the layers of the groups' heads the
/// raw scores of each question are reduced on the card to per-part log-sum-exp averaged over the
/// question's tokens (`part_lse_kernel`), so only `heads x parts` numbers a question come back; the
/// run stops at `stop` (`score_targets` of the last layer read). Returns `[group][question]` rows of
/// `heads x parts` (group's head order, then part order), the numbers `tentacle_pass::part_lse`
/// gives over the logits of a plain read. On the card only.
/// # Errors
/// Engine faults, or a group with no pieces, question, head or part.
pub fn read_group_parts(m: &Model, window: Option<&Arc<Window>>, stop: &[(String, String)], groups: &[PartGroup]) -> Result<Vec<Vec<Vec<f32>>>, Fault> {
    let _job = crate::chanrec::job("integration");
    if groups.is_empty() || groups.iter().any(|g| g.pieces.is_empty() || g.questions.is_empty() || g.questions.iter().any(Vec::is_empty) || g.heads.is_empty() || g.parts.is_empty()) {
        return Err(bad("read_group_parts: every group needs pieces, questions, heads and parts"));
    }
    let (mut lens, mut bodies, mut tails, mut packed) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (gi, g) in groups.iter().enumerate() {
        let body = lens.len();
        lens.push(g.pieces.len());
        bodies.push(None);
        tails.push(None);
        packed.extend_from_slice(&g.pieces);
        for (qi, q) in g.questions.iter().enumerate() {
            lens.push(q.len());
            bodies.push(Some(body));
            tails.push(Some((gi, qi)));
            packed.extend_from_slice(q);
        }
    }
    let out: Vec<Vec<Vec<f32>>> = groups.iter().map(|g| vec![vec![0.0; g.heads.len() * g.parts.len()]; g.questions.len()]).collect();
    let sp = Arc::new(SegmentParts { tails, groups: groups.iter().map(|g| (g.heads.clone(), g.parts.clone())).collect(), out: Mutex::new(out) });
    let opts = RunOpts {
        window: window.map(Arc::clone),
        segments: Some(Arc::new(lens)),
        segment_bodies: Some(Arc::new(bodies)),
        segment_parts: Some(Arc::clone(&sp)),
        device_targets: true,
        ..RunOpts::default()
    };
    m.forward_at(Device::Gpu, &packed, window.map_or(0, |w| w.len), stop, &opts)?;
    let r = sp.out.lock().map_err(|_| bad("part scores lock"))?.clone();
    Ok(r)
}

/// Per-head attention mass on a window: `[heads, window length]`. A layer listed in `heads`
/// reads only those heads (the others stay zero): the drone tips E2 chose, read without the
/// rest of the layer. 2026-11-10 (step P0): head numbers in `heads` are global (the
/// population's); mass is recorded per attention component (`record`) with the global head
/// range it covers, and `get(layer)` lays a layer's components side by side at their global
/// head rows (one component over every head: its mass exactly as recorded).
#[derive(Debug, Default)]
pub struct Probe {
    pub layers: Vec<usize>,
    pub heads: BTreeMap<usize, Vec<usize>>,
    /// Per attention component id: its layer, global head range and mass.
    pub nodes: Mutex<BTreeMap<String, NodeMass>>,
    /// Tentacle core A5 (2026-11-10): also record the raw attention scores (before softmax) of
    /// the listed heads (`with_logits`); off by default, so every existing reader is unchanged.
    pub want_logits: bool,
    /// Per attention component id: the raw scores of its read heads (`NodeLogits`).
    pub logit_nodes: Mutex<BTreeMap<String, NodeLogits>>,
    /// 2026-11-10 (tentacle assemblies): record only attention components whose id holds this
    /// text (a tentacle's drone tip), so a tentacle's own sensory components on the same layer
    /// are not read twice; `None` (the default) records every component, as before.
    pub node_filter: Option<String>,
}
/// One attention component's raw scores (tentacle core A5): the read heads in global numbers,
/// each with `crate::cpu::HeadLogits` (`[tokens, span]` scores before softmax and the log
/// normaliser per token).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NodeLogits {
    pub layer: usize,
    pub heads: (usize, usize),
    pub span: usize,
    pub tokens: usize,
    /// (global head, its scores).
    pub per_head: Vec<(usize, crate::cpu::HeadLogits)>,
}
/// One attention component's recorded mass: `[heads.1 - heads.0, span]`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NodeMass {
    pub layer: usize,
    pub heads: (usize, usize),
    pub span: usize,
    pub mass: Vec<f32>,
}
impl Probe {
    /// A probe of these layers, every head.
    #[must_use]
    pub fn new(layers: Vec<usize>) -> Self {
        Self {
            layers,
            heads: BTreeMap::new(),
            nodes: Mutex::new(BTreeMap::new()),
            want_logits: false,
            logit_nodes: Mutex::new(BTreeMap::new()),
            node_filter: None,
        }
    }
    /// A probe of these (layer, head) pairs only (global head numbers).
    #[must_use]
    pub fn of_heads(pairs: &[(usize, usize)]) -> Self {
        let mut heads: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for &(l, h) in pairs {
            heads.entry(l).or_default().push(h);
        }
        Self {
            layers: heads.keys().copied().collect(),
            heads,
            nodes: Mutex::new(BTreeMap::new()),
            want_logits: false,
            logit_nodes: Mutex::new(BTreeMap::new()),
            node_filter: None,
        }
    }
    /// The same probe, reading only attention components whose id holds `text` (2026-11-10).
    #[must_use]
    pub fn with_node_filter(mut self, text: &str) -> Self {
        self.node_filter = Some(text.to_owned());
        self
    }
    /// True when this probe reads the attention component `id`.
    #[must_use]
    pub fn accepts(&self, id: &str) -> bool {
        self.node_filter.as_ref().is_none_or(|f| id.contains(f.as_str()))
    }
    /// The heads read on layer `l` (`None`: all), global head numbers.
    #[must_use]
    pub fn heads_of(&self, l: usize) -> Option<&[usize]> {
        self.heads.get(&l).map(Vec::as_slice)
    }
    /// The heads an attention component over global heads `range` reads on layer `l`, as its
    /// own (local) head indices (`None`: all of them).
    #[must_use]
    pub fn local_heads(&self, l: usize, range: (usize, usize)) -> Option<Vec<usize>> {
        self.heads_of(l).map(|hs| hs.iter().filter(|g| (range.0..range.1).contains(*g)).map(|g| g - range.0).collect())
    }
    /// The last layer read (a run can stop after it).
    #[must_use]
    pub fn last_layer(&self) -> Option<usize> {
        self.layers.iter().copied().max()
    }
    /// Record one attention component's mass.
    pub fn record(&self, node: &str, layer: usize, heads: (usize, usize), span: usize, mass: Vec<f32>) {
        if let Ok(mut m) = self.nodes.lock() {
            m.insert(node.to_owned(), NodeMass { layer, heads, span, mass });
        }
    }
    /// The same probe, also recording raw scores (tentacle core A5).
    #[must_use]
    pub fn with_logits(mut self) -> Self {
        self.want_logits = true;
        self
    }
    /// Record one attention component's raw scores (local heads turned global here).
    #[expect(clippy::too_many_arguments, reason = "a component id, its layer, head range and geometry name one recording")]
    pub fn record_logits(
        &self,
        node: &str,
        layer: usize,
        heads: (usize, usize),
        span: usize,
        tokens: usize,
        per: Vec<crate::cpu::HeadLogits>,
    ) {
        if let Ok(mut m) = self.logit_nodes.lock() {
            m.insert(
                node.to_owned(),
                NodeLogits { layer, heads, span, tokens, per_head: per.into_iter().map(|h| (h.head + heads.0, h)).collect() },
            );
        }
    }
    /// The raw scores of global head `head` on `layer` (from whichever component covers it).
    #[must_use]
    pub fn logits(&self, layer: usize, head: usize) -> Option<crate::cpu::HeadLogits> {
        let m = self.logit_nodes.lock().ok()?;
        m.values().filter(|n| n.layer == layer).find_map(|n| n.per_head.iter().find(|(g, _)| *g == head).map(|(_, h)| h.clone()))
    }
    /// The mass one attention component recorded.
    #[must_use]
    pub fn get_node(&self, node: &str) -> Option<NodeMass> {
        self.nodes.lock().ok()?.get(node).cloned()
    }
    /// The mass recorded for a layer, `[heads, span]` by global head (heads no component of the
    /// layer covers stay zero). `None` when nothing was recorded on the layer, or when two of
    /// its components cover one head (read each with `get_node` then).
    #[must_use]
    pub fn get(&self, layer: usize) -> Option<Vec<f32>> {
        let m = self.nodes.lock().ok()?;
        let parts: Vec<&NodeMass> = m.values().filter(|n| n.layer == layer).collect();
        match parts.as_slice() {
            [] => None,
            [one] if one.heads.0 == 0 => Some(one.mass.clone()),
            _ => {
                let span = parts.first().map_or(0, |p| p.span);
                let top = parts.iter().map(|p| p.heads.1).max().unwrap_or(0);
                let mut seen = vec![false; top];
                let mut out = vec![0.0_f32; top * span];
                for p in &parts {
                    if p.span != span {
                        return None;
                    }
                    for h in p.heads.0..p.heads.1 {
                        if std::mem::replace(seen.get_mut(h)?, true) {
                            return None;
                        }
                        let src = p.mass.get((h - p.heads.0) * span..(h - p.heads.0 + 1) * span)?;
                        out.get_mut(h * span..(h + 1) * span)?.copy_from_slice(src);
                    }
                }
                Some(out)
            }
        }
    }
}

/// One chunk encoded once: its token count, keys before rope and values per layer
/// (`[tokens, kv heads, head width]`, binary16 bits), and the anchor length it followed.
/// Layers `0..k_pre.len()` are held: every layer from `encode`, layers through the last one
/// read from `encode_through` (2026-11-10).
#[derive(Clone, Debug, Default)]
pub struct ChunkKv {
    pub tokens: usize,
    pub anchor: usize,
    pub k_pre: Vec<Vec<u16>>,
    pub v: Vec<Vec<u16>>,
}
impl ChunkKv {
    /// Bytes held.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.k_pre.iter().chain(&self.v).map(|x| x.len() * 2).sum()
    }
    /// Layers held (`0..layers()`).
    #[must_use]
    pub fn layers(&self) -> usize {
        self.k_pre.len()
    }
}

/// Bytes one token holds when keys and values are kept for `layers` layers of model `m`
/// (binary16 keys and values: 2 x layers x kv heads x head width x 2 bytes).
#[must_use]
pub fn bytes_per_token(m: &Model, layers: usize) -> usize {
    let d = m.pop.dims;
    2 * layers * d.kv_heads * d.head_dim * 2
}

/// The RAM limit for kept encoded chunks, derived when the cache is made (2026-11-10,
/// direction of Luis Jake Gabriel III: operating numbers are derived at run time).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamLimit {
    /// The limit in bytes.
    pub bytes: usize,
    /// Physical memory available when it was read (0 when the reading failed).
    pub free: u64,
    /// Physical memory of the machine (0 when the reading failed).
    pub total: u64,
    /// The reading failed and the stated fallback is used.
    pub fallback: bool,
}

/// The fallback limit when the machine's memory cannot be read: the earlier fixed value
/// (8 GiB), used only then.
pub const FALLBACK_CACHE_BYTES: usize = 8 << 30;

/// Derive the limit from physical memory available now (`nmffi::dynlib::system_memory`).
/// Rule: half of the memory available at this moment is left free for the operating system,
/// other programs and the parts of the mini that load after the cache is made (the judge's
/// reranker, the embedder, the Core Brain); the kept chunks may use the other half. It is
/// read after the base model is loaded, so the model's own weights are already counted. When
/// the reading fails the limit is `FALLBACK_CACHE_BYTES`. Roadmap Phase 4.5 (2026-11-25): under
/// the user's host memory limit (`limits.ram_mb`) the cache takes at most half of it
/// (`limits::cache_ceiling`).
#[must_use]
pub fn derive_ram_limit() -> RamLimit {
    let mut l = ram_limit_from(nmffi::dynlib::system_memory().ok());
    l.bytes = crate::limits::cache_ceiling(l.bytes);
    l
}

/// The rule of `derive_ram_limit` on a reading (`None`: the reading failed).
#[must_use]
pub fn ram_limit_from(reading: Option<(u64, u64)>) -> RamLimit {
    match reading {
        Some((free, total)) if total > 0 && free <= total => {
            RamLimit { bytes: usize::try_from(free / 2).unwrap_or(usize::MAX), free, total, fallback: false }
        }
        _ => RamLimit { bytes: FALLBACK_CACHE_BYTES, free: 0, total: 0, fallback: true },
    }
}

/// A read-only key and value prefix: the chunks it holds (shared by reference, never copied),
/// the rope cosines and sines of window positions `0..len` (the one part a window owns), and
/// device copies made once, on first use on the GPU.
#[derive(Debug, Default)]
pub struct Window {
    pub len: usize,
    pub chunks: Vec<Arc<ChunkKv>>,
    /// Where each chunk starts: (first position, tokens), in order.
    pub parts: Vec<(usize, usize)>,
    table: Vec<(f32, f32)>,
    /// Device copies per key-value head range (2026-11-10, step P0: an attention component
    /// over some key-value heads gets a copy of those heads only).
    dev: Mutex<BTreeMap<(usize, usize), DevLayers>>,
    /// The window's content key (step P2): the chunk store's content keys combined in window
    /// order (`with_key`). A keyed window's device copies are offered to the residency manager
    /// and kept between questions while the device budget allows.
    key: Option<String>,
}
/// Device keys and values per layer.
type DevLayers = Vec<(DevT, DevT)>;
impl Window {
    /// Bytes this window owns in RAM (its rope table; the chunks are shared).
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.table.len() * 8
    }
    /// The rope cosines and sines of the window's positions (`cpu::rope_table`).
    #[must_use]
    pub fn table(&self) -> &[(f32, f32)] {
        &self.table
    }
    /// Bytes of the chunks it references.
    #[must_use]
    pub fn chunk_bytes(&self) -> usize {
        self.chunks.iter().map(|c| c.bytes()).sum()
    }
    /// Keys before rope of a layer, one segment per chunk, read in place (attention gives each
    /// row its window position's rope as it reads it).
    #[must_use]
    pub fn layer_k(&self, l: usize) -> Vec<&[u16]> {
        self.chunks.iter().filter_map(|c| c.k_pre.get(l).map(Vec::as_slice)).collect()
    }
    /// Values of a layer, one segment per chunk, read in place.
    #[must_use]
    pub fn layer_v(&self, l: usize) -> Vec<&[u16]> {
        self.chunks.iter().filter_map(|c| c.v.get(l).map(Vec::as_slice)).collect()
    }
    /// Layers the window's chunks hold (0 for a window with no chunks).
    #[must_use]
    pub fn layers(&self) -> usize {
        self.chunks.first().map_or(0, |c| c.k_pre.len())
    }
    /// Device copies of every layer (`[len, kv, d]` binary16): values uploaded as they are,
    /// keys uploaded before rope and given the GPU's own rope at window positions, so a window
    /// on the GPU matches the GPU's direct encoding. Made once.
    /// # Errors
    /// An upload or kernel failure.
    pub fn device(&self, o: &Ops, kv: usize, d: usize, theta: f32) -> Result<Vec<(DevT, DevT)>, String> {
        self.device_range(o, (0, kv), d, theta)
    }
    /// `device` for key-value heads `kv.0..kv.1` only (`[len, kv.1 - kv.0, d]`), for an
    /// attention component over those heads (2026-11-10, step P0). The window's rows hold every
    /// key-value head of their layer; the range `(0, all)` uploads them exactly as `device`
    /// always did. Made once per range.
    ///
    /// Step P2: the rows are copied to the device as they are, one plain copy per chunk (the
    /// chunks already hold `[tokens, kv heads, head width]` binary16, the device layout), and
    /// keys get the same rope kernel as before; a range of some heads gathers its columns into
    /// one u16 buffer first. A keyed window (`with_key`) takes a kept copy from the residency
    /// manager when one is held and offers a new copy to it afterwards. Bit for bit the same as
    /// `device_range_reference` (the path before step P2), check `nmengn.kvstore.upload`.
    /// # Errors
    /// A range past the heads the chunks hold, an upload or kernel failure.
    pub fn device_range(&self, o: &Ops, kv: (usize, usize), d: usize, theta: f32) -> Result<Vec<(DevT, DevT)>, String> {
        let mut g = self.dev.lock().map_err(|_| "window device lock".to_owned())?;
        if let Some(t) = g.get(&kv) {
            return Ok(t.clone());
        }
        let rkey = self.key.as_ref().map(|k| format!("{k}|kv {}..{}|d {d}|theta {theta}", kv.0, kv.1));
        if let Some(t) = rkey.as_deref().and_then(|k| crate::residency::get().window_get(k)) {
            g.insert(kv, t.clone());
            return Ok(t);
        }
        // Room in the shared device budget first, as for weights (step P2): a copy of a full
        // window is close to a gigabyte, so idle weights of other populations are released
        // before it is allocated rather than the card being oversubscribed (found in the first
        // `run kept` of step P2: reads of 45 s with the device full).
        let need = self.layers() * 2 * self.len * (kv.1.saturating_sub(kv.0)) * d * 2;
        crate::residency::get().make_room(need, "");
        let t = self.upload_range(o, kv, d, theta)?;
        if let Some(k) = rkey {
            let bytes = t.iter().map(|(a, b)| (a.elements() + b.elements()) * 2).sum();
            crate::residency::get().window_offer(&k, t.clone(), bytes);
        }
        g.insert(kv, t.clone());
        Ok(t)
    }
    /// Drop this window object's own device copies (tentacle core, 2026-11-10): a window kept
    /// in RAM for later re-reads must not pin its device copies (close to a gigabyte each). A
    /// keyed copy the residency manager kept stays there under its budget and is taken again on
    /// the next device read; otherwise the next device read uploads again. Returns the copies
    /// dropped.
    pub fn release_device(&self) -> usize {
        self.dev
            .lock()
            .map(|mut g| {
                let n = g.len();
                g.clear();
                n
            })
            .unwrap_or(0)
    }
    /// Keys (given the GPU's rope at window positions) and values of every layer on the
    /// device, heads `kv.0..kv.1` (step P2). Each layer's keys before rope are a working buffer
    /// freed after its rope, so the upload holds one layer's spare copy at a time.
    fn upload_range(&self, o: &Ops, kv: (usize, usize), d: usize, theta: f32) -> Result<DevLayers, String> {
        let layers = self.layers();
        let total = self.total_heads(d).unwrap_or(kv.1);
        if kv.0 >= kv.1 || kv.1 > total {
            return Err(format!("window rows hold {total} key-value heads; a component asked for heads {}..{}", kv.0, kv.1));
        }
        let n = kv.1 - kv.0;
        let shape = vec![self.len, n, d];
        let pos: Vec<i32> = (0..self.len).map(|p| i32::try_from(p).unwrap_or(i32::MAX)).collect();
        let pos = o.upload_i32(&pos)?;
        let mut out = Vec::with_capacity(layers);
        for l in 0..layers {
            let mut pair = Vec::with_capacity(2);
            for keys in [true, false] {
                let rows: Vec<&[u16]> =
                    self.chunks.iter().map(|c| if keys { c.k_pre.get(l) } else { c.v.get(l) }.map_or(&[][..], Vec::as_slice)).collect();
                // Values are kept as uploaded; keys before rope are a working buffer.
                let _k = (!keys).then(nmffi::KeptScope::new);
                let t = if n == total {
                    o.upload_parts16(&rows, shape.clone())?
                } else {
                    let mut cols: Vec<u16> = Vec::with_capacity(self.len * n * d);
                    for r in rows.iter().flat_map(|r| r.chunks(total * d)) {
                        cols.extend_from_slice(r.get(kv.0 * d..kv.1 * d).unwrap_or(&[]));
                    }
                    o.upload_parts16(&[&cols], shape.clone())?
                };
                pair.push(t);
            }
            let v = pair.pop().ok_or("window upload")?;
            let kp = pair.pop().ok_or("window upload")?;
            let k = {
                let _k = nmffi::KeptScope::new();
                o.rope_k(&kp, &pos, theta)?
            };
            out.push((k, v));
        }
        Ok(out)
    }
    /// Key-value heads per row the chunks hold (`None` with no tokens).
    fn total_heads(&self, d: usize) -> Option<usize> {
        self.chunks.iter().find(|c| c.tokens > 0).and_then(|c| c.k_pre.first().map(|x| x.len() / (c.tokens * d).max(1)))
    }
    /// The same window under its content key (step P2; `window_key` makes one).
    #[must_use]
    pub fn with_key(mut self, key: String) -> Self {
        self.key = Some(key);
        self
    }
    /// The content key, when the window has one.
    #[must_use]
    pub fn key(&self) -> Option<&str> {
        self.key.as_deref()
    }
    /// The upload path before step P2 (host repacking of every row into bytes, rope on keys
    /// and queries), kept for the equality check of the new path. Not cached.
    /// # Errors
    /// As `device_range`.
    pub fn device_range_reference(&self, o: &Ops, kv: (usize, usize), d: usize, theta: f32) -> Result<Vec<(DevT, DevT)>, String> {
        let layers = self.chunks.first().map_or(0, |c| c.k_pre.len());
        let total = self
            .chunks
            .iter()
            .find(|c| c.tokens > 0)
            .and_then(|c| c.k_pre.first().map(|x| x.len() / (c.tokens * d).max(1)))
            .unwrap_or(kv.1);
        if kv.0 >= kv.1 || kv.1 > total {
            return Err(format!("window rows hold {total} key-value heads; a component asked for heads {}..{}", kv.0, kv.1));
        }
        let pos: Vec<i32> = (0..self.len).map(|p| i32::try_from(p).unwrap_or(i32::MAX)).collect();
        let pos = o.upload_i32(&pos)?;
        let n = kv.1 - kv.0;
        let up = |l: usize, keys: bool| -> Result<DevT, String> {
            let rows = self.chunks.iter().flat_map(|c| if keys { &c.k_pre[l] } else { &c.v[l] });
            let bytes: Vec<u8> = if n == total {
                rows.flat_map(|b| b.to_le_bytes()).collect()
            } else {
                let row: Vec<&u16> = rows.collect();
                row.chunks(total * d).flat_map(|r| r.get(kv.0 * d..kv.1 * d).unwrap_or(&[]).iter().flat_map(|b| b.to_le_bytes())).collect()
            };
            o.upload16(&bytes, vec![self.len, n, d])
        };
        let mut t = Vec::with_capacity(layers);
        for l in 0..layers {
            let kp = up(l, true)?;
            let (_, k) = o.rope(&kp, &kp, &pos, theta)?;
            t.push((k, up(l, false)?));
        }
        Ok(t)
    }
    /// Place chunks side by side (anchor first when given) into one window by reference: no
    /// key is copied or rotated here; the window keeps the rope table of its positions. Every
    /// chunk holds the same layers: all of the model's, or the first ones through the last
    /// layer a read uses (`encode_through`, 2026-11-10); a read past them is refused by the
    /// engine.
    /// # Errors
    /// Chunks encoded with different layer counts or widths.
    pub fn build(m: &Model, parts: &[Arc<ChunkKv>]) -> Result<Self, Fault> {
        let _p = nmffi::prof::Span::new("window.build");
        let d = m.pop.dims;
        let width = d.kv_heads * d.head_dim;
        let layers = parts.first().map_or(d.layers, |c| c.k_pre.len());
        if layers == 0 || layers > d.layers {
            return Err(bad("a chunk holds no layers, or more than the model has"));
        }
        let mut at = 0;
        let mut placed = Vec::new();
        for c in parts {
            if c.k_pre.len() != layers || c.v.len() != layers {
                return Err(bad("a chunk was encoded with a different layer count"));
            }
            // 2026-11-10 (step P0): a window row holds every key-value head of its layer; a
            // chunk that holds fewer (or a token count that does not fit) is refused by name.
            if let Some((l, x)) = c.k_pre.iter().chain(&c.v).enumerate().find(|(_, x)| x.len() != c.tokens * width) {
                let per = x.len().checked_div(c.tokens).unwrap_or(0);
                return Err(bad(format!(
                    "a chunk of {} tokens holds {per} values per token in its {} {} of layer {}; a window needs every key-value head ({} x {} = {width} values per token)",
                    c.tokens,
                    if l < c.k_pre.len() { "keys" } else { "values" },
                    if per < width {
                        "(too few: encoded by attention components over some key-value heads only?)"
                    } else {
                        "(does not fit the token count)"
                    },
                    l % c.k_pre.len().max(1),
                    d.kv_heads,
                    d.head_dim
                )));
            }
            placed.push((at, c.tokens));
            at += c.tokens;
        }
        Ok(Self {
            len: at,
            chunks: parts.to_vec(),
            parts: placed,
            table: crate::cpu::rope_table(at, d.head_dim, d.theta),
            dev: Mutex::new(BTreeMap::new()),
            key: None,
        })
    }
    /// The placement of a window without a model (2026-11-10, scan dump replay): parts side by
    /// side by reference, positions and lengths only, no rope table. A window made this way
    /// cannot be read by the engine (a read needs `build`); it carries the chunk references and
    /// positions for code that only needs those, such as a replay of the tentacle cores on the
    /// processor from a scan dump.
    #[must_use]
    pub fn outline(parts: &[Arc<ChunkKv>]) -> Self {
        let mut at = 0;
        let placed = parts
            .iter()
            .map(|c| {
                let p = (at, c.tokens);
                at += c.tokens;
                p
            })
            .collect();
        Self { len: at, chunks: parts.to_vec(), parts: placed, ..Self::default() }
    }
}

/// The encoded working set (E3, 2026-11-09): chunks encoded once, held by content key and
/// shared by reference, least recently used out past `cap` bytes.
#[derive(Debug, Default)]
pub struct ChunkCache {
    map: BTreeMap<String, Arc<ChunkKv>>,
    order: std::collections::VecDeque<String>,
    bytes: usize,
    cap: usize,
}
impl ChunkCache {
    /// An empty cache holding at most `cap` bytes.
    #[must_use]
    pub fn new(cap: usize) -> Self {
        Self { cap, ..Self::default() }
    }
    /// Bytes held.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    /// The byte limit.
    #[must_use]
    pub fn cap(&self) -> usize {
        self.cap
    }
    /// Chunks held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }
    /// No chunk held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    /// The chunk under `key`, marked as just used.
    pub fn get(&mut self, key: &str) -> Option<Arc<ChunkKv>> {
        let c = self.map.get(key).cloned()?;
        self.order.retain(|k| k != key);
        self.order.push_back(key.to_owned());
        Some(c)
    }
    /// Hold `c` under `key`, dropping the least recently used past the cap. Roadmap Phase 4.5
    /// (2026-11-25): the cap is lowered live to half the user's host memory limit when one is
    /// set (`limits::cache_ceiling`), so a changed limit applies at the next chunk kept.
    pub fn put(&mut self, key: &str, c: Arc<ChunkKv>) {
        if let Some(old) = self.map.insert(key.to_owned(), Arc::clone(&c)) {
            self.bytes -= old.bytes();
            self.order.retain(|k| k != key);
        }
        self.bytes += c.bytes();
        self.order.push_back(key.to_owned());
        // 2026-11-27: and live under the machine's memory (`limits::host_cache_cap`).
        let cap = crate::limits::host_cache_cap(self.cap, self.bytes);
        while self.bytes > cap && self.order.len() > 1 {
            let Some(k) = self.order.pop_front() else { break };
            if let Some(old) = self.map.remove(&k) {
                self.bytes -= old.bytes();
            }
        }
    }
    /// Shared RAM budget (2026-11-10): encodings that something else keeps alive (`pinned`:
    /// content key to bytes, for example the windows a tentacle core keeps for its re-reads)
    /// count against the same limit. Least recently used entries that are not pinned are dropped
    /// until the bytes alive (this cache's unpinned entries plus every pinned encoding) fit
    /// `cap`; pinned entries are never dropped here (dropping them would free nothing). Returns
    /// (entries dropped, bytes alive after).
    pub fn fit_with(&mut self, pinned: &BTreeMap<String, usize>, cap: usize) -> (usize, usize) {
        // Roadmap Phase 4.5 (2026-11-25): never above half the user's host memory limit.
        let pinned_bytes: usize = pinned.values().sum();
        // 2026-11-27: and live under the machine's memory (`limits::host_cache_cap`).
        let cap = crate::limits::host_cache_cap(cap, self.bytes.saturating_add(pinned_bytes));
        let unpinned =
            |m: &BTreeMap<String, Arc<ChunkKv>>| m.iter().filter(|(k, _)| !pinned.contains_key(*k)).map(|(_, c)| c.bytes()).sum::<usize>();
        let mut alive = unpinned(&self.map) + pinned_bytes;
        let mut dropped = 0;
        let mut keep = std::collections::VecDeque::new();
        while alive > cap {
            let Some(k) = self.order.pop_front() else { break };
            if pinned.contains_key(&k) {
                keep.push_back(k);
                continue;
            }
            if let Some(old) = self.map.remove(&k) {
                self.bytes -= old.bytes();
                alive -= old.bytes();
                dropped += 1;
            }
        }
        while let Some(k) = keep.pop_back() {
            self.order.push_front(k);
        }
        (dropped, alive)
    }
    /// Change the byte limit (2026-11-10: the limit shared with the drone-key store), dropping
    /// the least recently used past it.
    pub fn set_cap(&mut self, cap: usize) {
        self.cap = cap;
        while self.bytes > self.cap && !self.order.is_empty() {
            let Some(k) = self.order.pop_front() else { break };
            if let Some(old) = self.map.remove(&k) {
                self.bytes -= old.bytes();
            }
        }
    }
}

/// The content key of a window (step P2): the chunk store's content keys of its parts (anchor
/// first) combined in window order, with what else fixes the encoded rows (population, last
/// layer kept). Two windows with the same key hold the same rows at the same positions.
#[must_use]
pub fn window_key(population: &str, last_layer: usize, part_keys: &[&str]) -> String {
    let mut text = format!("{population}\nthrough layer {last_layer}\n");
    for k in part_keys {
        text.push_str(k);
        text.push('\n');
    }
    nmhash::sha256_bytes(text.as_bytes())
}

/// Reads kept by key (step P2): a read that depends only on what its key names (for the
/// tentacle pass, the no-question read of a window: the window's content key, the pose text,
/// the drone set and the stop layer) is computed once and reused. `retain` drops every entry
/// whose key is not in use any more, so an entry goes when its window's content changes.
#[derive(Debug, Default)]
pub struct KeptReads {
    map: BTreeMap<String, Arc<Vec<f32>>>,
    hits: u64,
    misses: u64,
}
impl KeptReads {
    /// The read under `key`, counted as a hit or a miss.
    pub fn get(&mut self, key: &str) -> Option<Arc<Vec<f32>>> {
        let v = self.map.get(key).cloned();
        if v.is_some() {
            self.hits += 1;
        } else {
            self.misses += 1;
        }
        v
    }
    /// Keep `v` under `key`.
    pub fn put(&mut self, key: &str, v: Arc<Vec<f32>>) {
        self.map.insert(key.to_owned(), v);
    }
    /// Keep only the entries whose keys are in `live`; returns how many were dropped.
    pub fn retain(&mut self, live: &std::collections::BTreeSet<String>) -> usize {
        let before = self.map.len();
        self.map.retain(|k, _| live.contains(k));
        before - self.map.len()
    }
    /// Entries held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }
    /// No entry held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    /// Bytes of the values held.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.map.values().map(|v| v.len() * 4).sum()
    }
    /// (hits, misses) so far.
    #[must_use]
    pub fn counts(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }
}

/// Encode `ids` once after `anchor` (or alone), capturing every layer's keys before rope and
/// values. The chunk sits at positions `anchor.len..` while it is encoded.
/// # Errors
/// An engine fault, or a layer that recorded nothing.
pub fn encode(m: &Model, device: Device, anchor: Option<&Arc<Window>>, ids: &[i32]) -> Result<ChunkKv, Fault> {
    encode_through(m, device, anchor, ids, m.pop.dims.layers.saturating_sub(1))
}

/// `encode` stopped at layer `last` (2026-11-10): nothing above it is computed, and keys
/// before rope and values are kept for layers `0..=last` only. A read that stops at `last`
/// (the drone heads) sees exactly what the full encoding gives on those layers.
/// # Errors
/// An engine fault, a layer that recorded nothing, `last` past the model's layers, or an
/// anchor that holds fewer layers than the encoding needs.
pub fn encode_through(m: &Model, device: Device, anchor: Option<&Arc<Window>>, ids: &[i32], last: usize) -> Result<ChunkKv, Fault> {
    let _job = crate::chanrec::job("encode");
    let d = m.pop.dims;
    if last >= d.layers {
        return Err(bad(format!("last layer {last} past the model's {} layers", d.layers)));
    }
    if let Some(a) = anchor.filter(|a| a.len > 0 && a.layers() <= last) {
        return Err(bad(format!("the anchor holds {} layers, the encoding needs {}", a.layers(), last + 1)));
    }
    // Stage timing (2026-11-10, step P1; off unless a part test switches it on).
    let _enc = nmffi::prof::Span::new("encode");
    let setup = nmffi::prof::Span::new("encode.setup");
    let cap = Arc::new(Capture::default());
    let opts = RunOpts { window: anchor.cloned(), capture: Some(Arc::clone(&cap)), ..RunOpts::default() };
    let start = anchor.map_or(0, |a| a.len);
    // 2026-11-10 (step P0): the run stops at every attention component of layer `last`, found
    // in the assembly by operation and layer (was the fixed id `L{last}.score`, which a split or
    // renamed attention component does not have; the reference assembly gives that one id).
    let targets = score_targets(m, last)?;
    drop(setup);
    m.forward_at(device, ids, start, &targets, &opts)?;
    let _collect = nmffi::prof::Span::new("encode.collect");
    for l in 0..=last {
        if let Some((k, v)) = cap.uncovered(l)
            && !(k.is_empty() && v.is_empty())
        {
            return Err(bad(format!(
                "layer {l}: key-value heads {k:?} (keys) and {v:?} (values) were not recorded exactly once; the assembly's attention components must cover every key-value head of a layer once for an encoded chunk"
            )));
        }
    }
    let take = |x: &Mutex<BTreeMap<usize, Vec<u16>>>| -> Result<Vec<Vec<u16>>, Fault> {
        let mut g = x.lock().map_err(|_| bad("capture lock"))?;
        (0..=last).map(|l| g.remove(&l).ok_or_else(|| bad(format!("layer {l} recorded nothing")))).collect()
    };
    Ok(ChunkKv { tokens: ids.len(), anchor: start, k_pre: take(&cap.k_pre)?, v: take(&cap.v)? })
}

/// Partial recompute of a join (2026-11-13, sensory SN4; the CacheBlend idea, our selection rule):
/// every chunk after the first gets its first `frac` share of tokens (at least one) re-encoded
/// with the real preceding chunks of the window as context, and those tokens' keys and values
/// spliced into a copy of the chunk (keys before rope, so the splice is exact); chunks are given
/// with their token ids. Layers `0..=last`. Measured (`run cart sn4`, 5 windows): 15% brings the
/// drone heads' per-chunk mass to rank agreement 0.95 with the whole window encoded at once
/// (plain join 0.92) and the key-value error at layer 17 from 0.299 to 0.252.
/// # Errors
/// An engine fault, or a chunk whose token ids do not match its token count.
pub fn recompute(
    m: &Model,
    device: Device,
    parts: &[(Arc<ChunkKv>, Vec<i32>)],
    frac: f32,
    last: usize,
) -> Result<Vec<Arc<ChunkKv>>, Fault> {
    let d = m.pop.dims;
    let width = d.kv_heads * d.head_dim;
    let mut out: Vec<Arc<ChunkKv>> = Vec::with_capacity(parts.len());
    for (k, (c, ids)) in parts.iter().enumerate() {
        if ids.len() != c.tokens || c.k_pre.len() <= last {
            return Err(bad(format!(
                "recompute: chunk {k} holds {} tokens and {} layers, given {} ids for layer {last}",
                c.tokens,
                c.k_pre.len(),
                ids.len()
            )));
        }
        if k == 0 {
            out.push(Arc::clone(c));
            continue;
        }
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss, reason = "a token count")]
        let r = ((ids.len() as f32 * frac).ceil() as usize).clamp(1, ids.len());
        let prefix = Arc::new(Window::build(m, &out)?);
        let fresh = encode_through(m, device, Some(&prefix), &ids[..r], last)?;
        let mut copy = (**c).clone();
        for l in 0..=last {
            copy.k_pre[l][..r * width].copy_from_slice(&fresh.k_pre[l]);
            copy.v[l][..r * width].copy_from_slice(&fresh.v[l]);
        }
        out.push(Arc::new(copy));
    }
    Ok(out)
}

/// The run targets that end at layer `layer`: every `score` component the assembly lists on
/// that layer, in list order (2026-11-10, step P0). The reference assembly gives one,
/// `L{layer}.score`.
/// # Errors
/// The assembly has no `score` component on that layer.
pub fn score_targets(m: &Model, layer: usize) -> Result<Vec<(String, String)>, Fault> {
    let t: Vec<(String, String)> =
        m.asm.nodes.iter().filter(|n| n.op == "score" && n.layer == Some(layer)).map(|n| (n.id.clone(), "y".to_owned())).collect();
    if t.is_empty() {
        return Err(bad(format!("assembly {} has no score component on layer {layer}", m.asm.name)));
    }
    Ok(t)
}

/// Self-check of `kvstore`: rope placement of keys kept before rope equals rope applied at the
/// same positions directly, and segment reads see cached and new rows in order.
pub(crate) fn check() -> Result<(), String> {
    let pre = [0.5_f32, -0.25, 1.0, 2.0, 0.125, 3.0, -1.5, 0.75];
    let mut direct = pre.to_vec();
    crate::cpu::rope(&mut direct, 1, 4, &[7, 8], 1_000_000.0);
    let mut placed: Vec<f32> = pre.iter().map(|x| crate::cpu::wide(bits16(*x))).collect();
    crate::cpu::rope(&mut placed, 1, 4, &[7, 8], 1_000_000.0);
    if direct != placed {
        return Err("nmengn.kvstore: placement differs from direct rope".into());
    }
    // A row read with its position's rope equals the row placed by `rope` ahead of time.
    let table = crate::cpu::rope_table(9, 4, 1_000_000.0);
    let pre_bits: Vec<u16> = pre.iter().map(|x| bits16(*x)).collect();
    let rows = crate::cpu::Rows { cached: vec![&pre_bits], width: 4, pre: 1, table: &table, head: 4, ..Default::default() };
    let mut got = [0.0_f32; 8];
    rows.read(0, 0, &mut got[..4]);
    rows.read(1, 0, &mut got[4..]);
    let mut want: Vec<f32> = pre_bits.iter().map(|b| crate::cpu::wide(*b)).collect();
    crate::cpu::rope(&mut want, 1, 4, &[0, 1], 1_000_000.0);
    if got.as_slice() != want.as_slice() {
        return Err("nmengn.kvstore: rope on read differs from placed rope".into());
    }
    let mut cache = ChunkCache::new(40);
    let chunk = |n: usize| Arc::new(ChunkKv { tokens: 1, anchor: 0, k_pre: vec![vec![0; n]], v: vec![vec![0; n]] });
    cache.put("a", chunk(4));
    cache.put("b", chunk(4));
    let _ = cache.get("a");
    cache.put("c", chunk(4));
    if cache.get("b").is_some() || cache.get("a").is_none() || cache.bytes() > 40 {
        return Err("nmengn.kvstore: chunk cache order".into());
    }
    if cache.cap() != 40 || cache.len() != 2 || cache.is_empty() {
        return Err("nmengn.kvstore: chunk cache size figures".into());
    }
    // Shared budget (2026-11-10): pinned encodings count, are never dropped, and the rest go
    // least recently used first. Each chunk(4) holds 16 bytes; "a" and "c" are held.
    let mut pinned = BTreeMap::new();
    pinned.insert("a".to_owned(), 16);
    pinned.insert("w".to_owned(), 16);
    let (dropped, alive) = cache.fit_with(&pinned, 40);
    if dropped != 1 || alive != 32 || cache.get("a").is_none() || cache.get("c").is_some() {
        return Err(format!("nmengn.kvstore: shared budget with pinned encodings (dropped {dropped}, alive {alive})"));
    }
    cache.put("c", chunk(4));
    cache.set_cap(16);
    if cache.len() != 1 || cache.bytes() != 16 || cache.get("c").is_none() {
        return Err("nmengn.kvstore: lowered limit".into());
    }
    // The RAM limit rule: half of the available memory; the fallback only when the reading fails.
    let l = ram_limit_from(Some((10 << 30, 32 << 30)));
    if l.bytes != 5 << 30 || l.fallback || l.free != 10 << 30 {
        return Err(format!("nmengn.kvstore: RAM limit rule {l:?}"));
    }
    if ram_limit_from(None) != (RamLimit { bytes: FALLBACK_CACHE_BYTES, free: 0, total: 0, fallback: true })
        || !ram_limit_from(Some((5, 0))).fallback
    {
        return Err("nmengn.kvstore: RAM limit fallback".into());
    }
    let live = derive_ram_limit();
    if !live.fallback && (live.free == 0 || live.bytes == 0) {
        return Err(format!("nmengn.kvstore: RAM limit read {live:?}"));
    }
    let cached: Vec<u16> = [1.0_f32, 2.0].iter().map(|x| bits16(*x)).collect();
    let rows = crate::cpu::Rows { cached: vec![&cached], new: &[3.0, 4.0], width: 2, ..Default::default() };
    let mut out = [0.0_f32; 2];
    rows.read(1, 0, &mut out);
    if rows.len() != 2 || out != [3.0, 4.0] {
        return Err("nmengn.kvstore: segment reads".into());
    }
    Ok(())
}

/// Texts for the step P2 checks (written for them; not from any test).
pub(crate) const P2_ANCHOR: &str = "Repository of the mini, read in parts.\n";
pub(crate) const P2_CHUNKS: [&str; 3] = [
    "crates/x/src/ring.rs\npub struct Ring { items: Vec<u32>, head: usize }\nimpl Ring {\n    pub fn push(&mut self, x: u32) { let n = self.items.len(); self.items[self.head % n] = x; self.head += 1; }\n}\n",
    "crates/x/src/clock.rs\npub fn tick(now: u64, step: u64) -> u64 { now.wrapping_add(step) }\npub fn late(now: u64, due: u64) -> bool { now > due }\n",
    "crates/x/src/limit.rs\npub struct Limit { bytes: usize }\nimpl Limit {\n    pub fn room(&self, used: usize) -> usize { self.bytes.saturating_sub(used) }\n}\n",
];

/// Write a check's report to its section-check folder and pass on its result.
pub(crate) fn p2_report(name: &str, title: &str, run: fn(&mut String) -> Result<(), String>) -> Result<(), String> {
    use std::fmt::Write as _;
    let mut rep = format!("{title}\n");
    let r = run(&mut rep);
    let _ = writeln!(rep, "result: {}", r.as_ref().map_or_else(|e| format!("FAIL {e}"), |()| "ok".to_owned()));
    let dir = nmtype::check_dir(name).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("report.txt"), rep).map_err(|e| e.to_string())?;
    r
}

/// The released base, bound to the reference assembly, with its device (`None`: no GPU).
fn p2_model() -> Result<Option<(Arc<crate::model::Population>, Model, Ops)>, String> {
    let home = nmtype::nano_root();
    let pop = crate::model::Population::open(&home, DRONE_FALLBACK_POPULATION, true, true).map_err(|e| e.to_string())?;
    let Some(o) = pop.gpu.clone() else { return Ok(None) };
    let asm = crate::assembly::Assembly::load(&home.join("designs/assemblies"), "qwen3-reference").map_err(|e| e.to_string())?;
    let m = Model::bind(&pop, asm).map_err(|e| e.to_string())?;
    Ok(Some((pop, m, o)))
}

/// Anchor and chunks encoded on the GPU through the last drone layer, as the pass keeps them.
fn p2_parts(m: &Model, last: usize) -> Result<Vec<Arc<ChunkKv>>, String> {
    let tok = |s: &str| m.pop.tokenizer.encode(s).map_err(|e| e.to_string());
    let ak = Arc::new(encode_through(m, Device::Gpu, None, &tok(P2_ANCHOR)?, last).map_err(|e| e.to_string())?);
    let aw = Arc::new(Window::build(m, &[Arc::clone(&ak)]).map_err(|e| e.to_string())?);
    let mut parts = vec![ak];
    for c in P2_CHUNKS {
        parts.push(Arc::new(encode_through(m, Device::Gpu, Some(&aw), &tok(c)?, last).map_err(|e| e.to_string())?));
    }
    Ok(parts)
}

/// Drone heads' mass on a window for a posed text, every probed layer laid out in order.
fn p2_read(m: &Model, w: &Arc<Window>, heads: &[(usize, usize)], last: usize, text: &str) -> Result<Vec<f32>, String> {
    let q = m.pop.tokenizer.encode(text).map_err(|e| e.to_string())?;
    let probe = Arc::new(Probe::of_heads(heads));
    let stop = score_targets(m, last).map_err(|e| e.to_string())?;
    m.forward_at(
        Device::Gpu,
        &q,
        w.len,
        &stop,
        &RunOpts { window: Some(Arc::clone(w)), probe: Some(Arc::clone(&probe)), ..RunOpts::default() },
    )
    .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for l in &probe.layers {
        out.extend(probe.get(*l).ok_or_else(|| format!("layer {l} not read"))?);
    }
    Ok(out)
}

pub(crate) fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Section check of step P2 fix 1 (2026-11-10), on the released Qwen3-0.6B on the GPU: a
/// window's device copy made the new way (one plain copy per chunk, keys given the key-only rope
/// launch) equals the copy of the path before step P2 (`device_range_reference`: host repacking
/// into bytes, rope on keys and queries) bit for bit, for every key-value head and for a range
/// of some heads; a keyed window's copy is kept by the residency manager, a second window object
/// with the same content key takes that kept copy without uploading, and the drone read on it
/// equals the read on a fresh unkeyed window bit for bit. Without a GPU the check reports that.
/// # Errors
/// A difference, or an engine fault.
pub(crate) fn check_upload() -> Result<(), String> {
    p2_report("nmengn-upload", "nmengn.kvstore.upload (step P2 fix 1)", upload_check)
}
fn upload_check(rep: &mut String) -> Result<(), String> {
    use std::fmt::Write as _;
    let Some((pop, m, o)) = p2_model()? else {
        let _ = writeln!(rep, "no GPU open: skipped (the CPU path reads the chunks in place)");
        return Ok(());
    };
    let d = pop.dims;
    let set = drone_set(&nmtype::nano_root(), DRONE_FALLBACK_POPULATION).map_err(|e| e.to_string())?;
    let last = set.last_layer();
    let parts = p2_parts(&m, last)?;
    let w = Window::build(&m, &parts).map_err(|e| e.to_string())?;
    let host_per_token = w.chunk_bytes() / w.len.max(1);
    for kv in [(0, d.kv_heads), (2, 6)] {
        let new = w.device_range(&o, kv, d.head_dim, d.theta)?;
        let old = w.device_range_reference(&o, kv, d.head_dim, d.theta)?;
        if new.len() != old.len() || new.len() != last + 1 {
            return Err(format!("heads {kv:?}: {} layers new, {} old", new.len(), old.len()));
        }
        let mut values = 0;
        for (l, ((nk, nv), (ok, ov))) in new.iter().zip(&old).enumerate() {
            for (a, b, what) in [(nk, ok, "keys"), (nv, ov, "values")] {
                let (x, y) = (o.download16(a)?, o.download16(b)?);
                if x != y || a.shape != b.shape {
                    return Err(format!("heads {kv:?} layer {l} {what}: new copy differs from the reference path"));
                }
                values += x.len();
            }
        }
        let dev_bytes: usize = new.iter().map(|(a, b)| (a.elements() + b.elements()) * 2).sum();
        let _ = writeln!(
            rep,
            "heads {}..{}: {} layers, {values} binary16 values, new copy equal to the reference path bit for bit; device bytes per token {}",
            kv.0,
            kv.1,
            new.len(),
            dev_bytes / w.len.max(1)
        );
    }
    let _ = writeln!(
        rep,
        "window {} positions; host bytes per token {host_per_token} (the chunks as kept; unchanged by step P2, which copies them as they are)",
        w.len
    );
    // Keyed windows and the residency manager.
    let keys: Vec<String> = (0..parts.len()).map(|i| format!("p2-check part {i}")).collect();
    let key = window_key(DRONE_FALLBACK_POPULATION, last, &keys.iter().map(String::as_str).collect::<Vec<_>>());
    let res = crate::residency::get();
    let s0 = res.stats();
    let w1 = Arc::new(Window::build(&m, &parts).map_err(|e| e.to_string())?.with_key(key.clone()));
    let text = "\nQuestion: where is the ring buffer pushed?";
    let r1 = p2_read(&m, &w1, &set.heads, last, text)?;
    let s1 = res.stats();
    let kept = s1.windows_kept > s0.windows_kept;
    let w2 = Arc::new(Window::build(&m, &parts).map_err(|e| e.to_string())?.with_key(key));
    let r2 = p2_read(&m, &w2, &set.heads, last, text)?;
    let s2 = res.stats();
    let plain = Arc::new(Window::build(&m, &parts).map_err(|e| e.to_string())?);
    let r0 = p2_read(&m, &plain, &set.heads, last, text)?;
    let _ = writeln!(
        rep,
        "keyed window: kept {kept} (windows kept {} -> {}), second window object took the kept copy: {} (hits {} -> {}); reads of {} values: kept copy equals fresh window {}, first keyed read equals fresh window {}",
        s0.windows_kept,
        s1.windows_kept,
        s2.window_hits > s1.window_hits,
        s1.window_hits,
        s2.window_hits,
        r0.len(),
        same_bits(&r2, &r0),
        same_bits(&r1, &r0)
    );
    let _ = writeln!(rep, "{}", res.report());
    res.release_windows();
    pop.release_device();
    if !kept {
        return Err("the keyed window was not kept (device budget refused a window of a few megabytes)".into());
    }
    if s2.window_hits <= s1.window_hits || !same_bits(&r2, &r0) || !same_bits(&r1, &r0) || r0.is_empty() {
        return Err("a kept window copy did not give the fresh window's read".into());
    }
    Ok(())
}

/// Section check of step P2 fix 4 (2026-11-10): (a) every binary16 bit pattern that is not a
/// NaN comes back unchanged from a widening to f32 and a narrowing back (the capture's old
/// path), so storing the downloaded bits as they are gives the same stored bytes; NaN patterns
/// are counted; (b) on the GPU, a tensor holding every pattern downloads to the same bits by
/// `download16` as by the old download and narrowing, NaN excepted; (c) a chunk encoded with the
/// pool on (twice, the second time from reused buffers) and with the pool off holds the same
/// keys and values bit for bit.
/// # Errors
/// A difference, or an engine fault.
pub(crate) fn check_capture16() -> Result<(), String> {
    p2_report("nmengn-capture16", "nmengn.kvstore.capture16 (step P2 fix 4)", capture16_check)
}
fn capture16_check(rep: &mut String) -> Result<(), String> {
    use std::fmt::Write as _;
    let all: Vec<u16> = (0..=u16::MAX).collect();
    let is_nan = |b: u16| (b & 0x7c00) == 0x7c00 && (b & 0x03ff) != 0;
    let changed: Vec<u16> = all.iter().copied().filter(|b| bits16(crate::half::from16(*b)) != *b).collect();
    let nan = all.iter().filter(|b| is_nan(**b)).count();
    let _ = writeln!(
        rep,
        "(a) {} of 65536 patterns change on the old round trip, all NaN: {} ({nan} NaN patterns exist)",
        changed.len(),
        changed.iter().all(|b| is_nan(*b))
    );
    if !changed.iter().all(|b| is_nan(*b)) {
        return Err("a non-NaN pattern changes on the round trip".into());
    }
    let Some((pop, m, o)) = p2_model()? else {
        let _ = writeln!(rep, "no GPU open: (b) and (c) skipped");
        return Ok(());
    };
    let t = o.upload_parts16(&[&all], vec![all.len()])?;
    let direct = o.download16(&t)?;
    let old: Vec<u16> = o.download(&t)?.iter().map(|x| bits16(*x)).collect();
    let differ = direct.iter().zip(&old).filter(|(a, b)| a != b).count();
    let bad_differ = direct.iter().zip(&old).filter(|(a, b)| a != b && !is_nan(**a)).count();
    let _ = writeln!(
        rep,
        "(b) device tensor of every pattern: download16 equals the device bits {}; differs from the old path at {differ} patterns, non-NaN {bad_differ}",
        direct == all
    );
    if direct != all || bad_differ != 0 {
        return Err("download16 is not the device's bits, or the old path differs off NaN".into());
    }
    let last = drone_set(&nmtype::nano_root(), DRONE_FALLBACK_POPULATION).map_err(|e| e.to_string())?.last_layer();
    let ids = pop.tokenizer.encode(P2_CHUNKS[0]).map_err(|e| e.to_string())?;
    let enc = || encode_through(&m, Device::Gpu, None, &ids, last).map_err(|e| e.to_string());
    let p0 = o.gpu.pool_stats();
    let a = enc()?;
    let b = enc()?;
    let p1 = o.gpu.pool_stats();
    o.gpu.set_pool(false);
    let c = enc();
    o.gpu.set_pool(true);
    let c = c?;
    let same = |x: &ChunkKv, y: &ChunkKv| x.k_pre == y.k_pre && x.v == y.v && x.tokens == y.tokens;
    let _ = writeln!(
        rep,
        "(c) chunk of {} tokens through layer {last}: pool on equals pool off {}, second pooled run equals first {}; pooled runs reused {} buffers and allocated {} new on the device",
        ids.len(),
        same(&a, &c),
        same(&a, &b),
        p1.hits - p0.hits,
        p1.device_allocs - p0.device_allocs
    );
    pop.release_device();
    if !same(&a, &c) || !same(&a, &b) {
        return Err("pooled encoding differs from unpooled encoding".into());
    }
    Ok(())
}

/// One probed read for the logit check: the probe after the run.
#[expect(clippy::too_many_arguments, reason = "one probed read: model, device, window, heads, stop layer, text and the logits switch")]
pub(crate) fn logit_probe(
    m: &Model,
    dev: Device,
    w: &Arc<Window>,
    heads: &[(usize, usize)],
    last: usize,
    text: &str,
    logits: bool,
) -> Result<Arc<Probe>, String> {
    let q = m.pop.tokenizer.encode(text).map_err(|e| e.to_string())?;
    let p = Probe::of_heads(heads);
    let probe = Arc::new(if logits { p.with_logits() } else { p });
    let stop = score_targets(m, last).map_err(|e| e.to_string())?;
    m.forward_at(dev, &q, w.len, &stop, &RunOpts { window: Some(Arc::clone(w)), probe: Some(Arc::clone(&probe)), ..RunOpts::default() })
        .map_err(|e| e.to_string())?;
    Ok(probe)
}

/// Section check of tentacle core A5 (2026-11-10), on the released Qwen3-0.6B, CPU and GPU: a
/// probe that also records raw attention scores gives the same mass, bit for bit, as a probe
/// that does not (so no existing output changes), and the recorded scores reproduce that mass:
/// for every drone head and window position, the sum over question tokens of
/// `exp(logit - lse)` equals the recorded mass (largest difference reported; limit 1e-5, a few
/// f32 roundings of weights that sum to the token count). Without a GPU the check reports that.
/// # Errors
/// A difference, or an engine fault.
pub(crate) fn check_logits() -> Result<(), String> {
    p2_report("nmengn-logits", "nmengn.kvstore.logits (tentacle core A5)", logits_check)
}
fn logits_check(rep: &mut String) -> Result<(), String> {
    use std::fmt::Write as _;
    let Some((pop, m, _o)) = p2_model()? else {
        let _ = writeln!(rep, "no GPU open: skipped (the chunks are encoded on the GPU here)");
        return Ok(());
    };
    let set = drone_set(&nmtype::nano_root(), DRONE_FALLBACK_POPULATION).map_err(|e| e.to_string())?;
    let last = set.last_layer();
    let parts = p2_parts(&m, last)?;
    let w = Arc::new(Window::build(&m, &parts).map_err(|e| e.to_string())?);
    let text = "\nQuestion: where is the ring buffer pushed?";
    let mut worst = 0.0_f32;
    for dev in [Device::Cpu, Device::Gpu] {
        let plain = logit_probe(&m, dev, &w, &set.heads, last, text, false)?;
        let with = logit_probe(&m, dev, &w, &set.heads, last, text, true)?;
        let (mut values, mut dmax, mut heads) = (0_usize, 0.0_f32, 0_usize);
        for &l in &plain.layers {
            let (a, b) = (plain.get(l).ok_or("layer not read")?, with.get(l).ok_or("layer not read")?);
            if !same_bits(&a, &b) {
                return Err(format!("{}: layer {l}: mass with logits recorded differs from mass without", dev.name()));
            }
            values += a.len();
            for &h in set.heads.iter().filter(|p| p.0 == l).map(|(_, h)| h) {
                let lg = with.logits(l, h).ok_or_else(|| format!("{}: no logits for layer {l} head {h}", dev.name()))?;
                let tokens = lg.lse.len();
                if tokens == 0 || lg.logits.len() != tokens * w.len {
                    return Err(format!(
                        "{}: layer {l} head {h}: {} logits for {tokens} tokens and {} positions",
                        dev.name(),
                        lg.logits.len(),
                        w.len
                    ));
                }
                for p in 0..w.len {
                    let back: f32 = (0..tokens)
                        .map(|t| {
                            (lg.logits.get(t * w.len + p).copied().unwrap_or(f32::NEG_INFINITY) - lg.lse.get(t).copied().unwrap_or(0.0))
                                .exp()
                        })
                        .sum();
                    let rec = a.get(h * w.len + p).copied().unwrap_or(0.0);
                    dmax = dmax.max((back - rec).abs());
                }
                heads += 1;
            }
        }
        if plain.logit_nodes.lock().map(|n| !n.is_empty()).unwrap_or(true) {
            return Err(format!("{}: a probe without logits recorded logits", dev.name()));
        }
        let _ = writeln!(
            rep,
            "{}: window {} positions, {} drone heads; mass with and without logits equal bit for bit ({values} values); mass rebuilt from the logits: largest difference {dmax:.2e}",
            dev.name(),
            w.len,
            heads
        );
        worst = worst.max(dmax);
    }
    pop.release_device();
    if worst > 1e-5 {
        return Err(format!("mass rebuilt from the logits differs by {worst:.2e} (limit 1e-5)"));
    }
    Ok(())
}
