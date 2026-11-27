//! The tentacle core, first form (2026-11-10; design `docs/2026-11-10-neumor-design-tentacle-core-v1.md`,
//! architecture by Luis Jake Gabriel III; specification `docs/2026-11-10-neumor-guide-how-it-works.md`
//! sections 2.2 to 2.4 and 3). Three separable parts, each with its own check:
//! - `Spec`: the one module specification applied to every tentacle (`designs/tentacle-core.json`),
//!   validated, and expanded per tentacle into the weight clusters each module holds
//!   (`holdings`, `anatomy_table`; A8 and A10 in first form).
//! - `TentacleCore`: one core per tentacle. It owns its drone set and its window (chunks by
//!   reference into the kept chunk store), holds a `nmbrain::g0::G0` for its state and inbox, and
//!   implements `nmbrain::g0::TentacleSide`: a report with a confidence that means the same in
//!   every tentacle and a no-information flag judged against the content-free baseline, a re-read
//!   with a new focus as a new pose over the kept encoded chunks, and an answer about one span
//!   (A9).
//! - `judge`: the G1 loop wired in behind `Settings::core`: the cores' `g0_tx` states go to
//!   `nmbrain::g1::G1::run` with a derived budget, and G1's hand-over is forwarded in place of the
//!   per-tentacle cut. Instructions travel as `nmtype` envelopes (behavioural); the forwarded chunks
//!   are sensory packets; both are checked at their boundaries, and the isolation gate is recorded.
//!
//! Scores. A chunk's calibrated score (`tentacle_pass::Score::Calibrated`) is, per drone head, the
//! log-sum-exp of the raw attention scores over the chunk's positions, averaged over the question
//! tokens, posed minus content-free at the same positions, summed over the heads. The core reports
//! it minus the anchor's score: the anchor is the same text at the same positions in every window,
//! so it is each window's content-free baseline. Confidence and gate 1 follow `nmbrain::g1::rules` (first form: `logistic(score / heads)`, one nat
//! per head as the unit, the same in every tentacle); no information when no chunk scores above the
//! anchor. Method source: the content-free calibration of In-Context Reranking (Chen, Jimenez
//! Gutierrez, Su, arXiv 2410.02642; code github.com/OSU-NLP-Group/In-Context-Reranking, MIT licence):
//! the subtraction of a content-free query's attention at the same positions was taken; it is
//! applied here to raw scores rather than normalised attention. No code was copied.
//!
//! Parallel rule (stated, recorded in each note): processor phases run on one thread per tentacle
//! (G1 starts them); device reads take the device gate one at a time, because the card is one
//! device with one stream and 4 GB, and on the processor one read already spreads over every core.
use crate::core_read::{self, ReadOptions};
use crate::tentacle_pass::{self, Cand, Pass, Scan, Score, Settings, WindowScan};
use nmbrain::g0::{self, Answer, Focus, G0, HeldChunk, Inbox, Report, SpanRef, SpanReport, TentacleSide};
use nmbrain::g1::{Budget, G1, ItemSig, Outcome, WholeCodebase};
use nmengn::kvstore::{self, ChunkKv, DroneSet, Window};
use nmengn::model::Model;
use nmengn::placement::Device;
use nmjson::Value;
use nmsubs::Substrate;
use nmtype::{Fault, FaultKind};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmapi.tentacle_core", FaultKind::Execution, m)
}

/// The module specification file.
pub const SPEC_FILE: &str = "designs/tentacle-core.json";
/// Module kinds the specification may name (specification 2.2: attention, feed-forward, gating,
/// memory, routing, parameter clusters; plus the core's reasoning; and since 2026-11-19 the
/// recurrent path, master plan 3.3: the core's state carried across rounds through its g0).
pub const KINDS: [&str; 8] = ["attention", "feed-forward", "gating", "memory", "routing", "parameter-cluster", "reasoning", "recurrent"];
/// The kinds every tentacle's anatomy must hold (as `kind` or `also`); the recurrent path since
/// 2026-11-19 (master plan 3.3, the core's components).
pub const REQUIRED_KINDS: [&str; 7] = ["attention", "feed-forward", "gating", "memory", "routing", "parameter-cluster", "recurrent"];
/// Layer range keywords.
pub const LAYERS: [&str; 4] = ["sense", "drone", "deep", "none"];
/// Cluster group keywords (NPCS groups of `nmnpcs::decompose`).
pub const GROUPS: [&str; 10] =
    ["attn", "head:all", "head:drone", "head:verification", "kv:all", "kv:drone", "kv:verification", "ffn", "embed", "final"];
/// Module states in a first form.
pub const STATES: [&str; 3] = ["built", "first form", "not built"];
/// The stated fallback shortlist when no chunk scores above the anchor (G1 rule D2, the served
/// shortlist).
pub const SHORTLIST_FALLBACK: usize = tentacle_pass::SERVED_SHORTLIST;

/// One module of the specification.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Module {
    pub name: String,
    pub kind: String,
    pub also: String,
    pub layers: String,
    pub groups: Vec<String>,
    pub shared: String,
    pub state: String,
    pub rule: String,
    /// Gating modules: (fallback value, why).
    pub fallback: Option<(f64, String)>,
    /// The component of the assembled core this module binds to (`Component::parse`; empty: a
    /// holding with no step of its own), 2026-11-19.
    pub component: String,
}

/// The tentacle core specification.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Spec {
    pub population: String,
    pub modules: Vec<Module>,
}

/// Parse and validate a specification: version 1, a population with a weight card under `home`,
/// `applies_to` all, unique module names, known kinds, layer ranges, cluster groups, sharing and
/// states; gating modules with a rule, what it is derived from and a fallback with its reason;
/// every required kind present. Named faults.
/// # Errors
/// The first rule broken.
pub fn validate(home: &Path, v: &Value) -> Result<Spec, String> {
    let f = SPEC_FILE;
    let s = |x: &Value, k: &str| -> Option<String> { x.get(k).ok().and_then(|y| y.string().ok()).map(str::to_owned) };
    match v.get("version").ok().and_then(|x| x.integer().ok()) {
        Some(1) => {}
        other => return Err(format!("{f}: version must be 1, got {other:?}")),
    }
    let population = s(v, "population").ok_or_else(|| format!("{f}: population missing"))?;
    nmnpcs::WeightCard::load(home, &population).map_err(|e| format!("{f}: population {population}: {e}"))?;
    if s(v, "applies_to").as_deref() != Some("all") {
        return Err(format!("{f}: applies_to must be \"all\" (one specification for every tentacle)"));
    }
    let mods = v.get("modules").and_then(Value::array).map_err(|e| format!("{f}: modules: {e}"))?;
    if mods.is_empty() {
        return Err(format!("{f}: no modules"));
    }
    let mut out = Vec::new();
    let mut names = BTreeSet::new();
    for (i, m) in mods.iter().enumerate() {
        let name = s(m, "name").ok_or_else(|| format!("{f}: module {i} has no name"))?;
        if !names.insert(name.clone()) {
            return Err(format!("{f}: module {name} named twice"));
        }
        let field = |k: &str| s(m, k).ok_or_else(|| format!("{f}: {name}: {k} missing"));
        let kind = field("kind")?;
        let also = s(m, "also").unwrap_or_default();
        for k in std::iter::once(&kind).chain((!also.is_empty()).then_some(&also)) {
            if !KINDS.contains(&k.as_str()) {
                return Err(format!("{f}: {name}: kind {k} is not one of {}", KINDS.join(", ")));
            }
        }
        let layers = field("layers")?;
        if !LAYERS.contains(&layers.as_str()) {
            return Err(format!("{f}: {name}: layers {layers} is not one of {}", LAYERS.join(", ")));
        }
        let groups: Vec<String> = m
            .get("groups")
            .and_then(Value::array)
            .map_err(|e| format!("{f}: {name}: groups: {e}"))?
            .iter()
            .map(|g| g.string().map(str::to_owned).map_err(|e| format!("{f}: {name}: group: {e}")))
            .collect::<Result<_, _>>()?;
        if let Some(g) = groups.iter().find(|g| !GROUPS.contains(&g.as_str())) {
            return Err(format!("{f}: {name}: group {g} is not one of {}", GROUPS.join(", ")));
        }
        if layers == "none" && groups.iter().any(|g| !matches!(g.as_str(), "embed" | "final")) {
            return Err(format!("{f}: {name}: layer groups named with layers none"));
        }
        let shared = field("shared")?;
        if shared != "reference" && shared != "per-tentacle" {
            return Err(format!("{f}: {name}: shared must be reference or per-tentacle, got {shared}"));
        }
        let state = field("state")?;
        if !STATES.contains(&state.as_str()) {
            return Err(format!("{f}: {name}: state {state} is not one of {}", STATES.join(", ")));
        }
        let rule = s(m, "rule").unwrap_or_default();
        let component = s(m, "component").unwrap_or_default();
        if !component.is_empty() && Component::parse(&component).is_none() {
            return Err(format!("{f}: {name}: component {component} is not one of {}", Component::ALL.iter().map(|c| c.name()).collect::<Vec<_>>().join(", ")));
        }
        let mut fallback = None;
        if kind == "gating" {
            if rule.is_empty() || s(m, "derived_from").is_none_or(|d| d.is_empty()) {
                return Err(format!("{f}: {name}: a gating module needs a rule and what it is derived from"));
            }
            let fb = m.get("fallback").map_err(|_| format!("{f}: {name}: a gating module needs a stated fallback"))?;
            let value = match fb.get("value") {
                Ok(Value::Number(n)) => n.parse::<f64>().map_err(|e| format!("{f}: {name}: fallback value: {e}"))?,
                _ => return Err(format!("{f}: {name}: fallback value must be a number")),
            };
            let why = s(fb, "why").filter(|w| !w.is_empty()).ok_or_else(|| format!("{f}: {name}: fallback needs its reason (why)"))?;
            fallback = Some((value, why));
        }
        out.push(Module { name, kind, also, layers, groups, shared, state, rule, fallback, component });
    }
    for k in REQUIRED_KINDS {
        if !out.iter().any(|m| m.kind == k || m.also == k) {
            return Err(format!("{f}: no module of kind {k} (specification 2.2)"));
        }
    }
    Ok(Spec { population, modules: out })
}

/// Load and validate `designs/tentacle-core.json` under `home`.
/// # Errors
/// Unreadable, not JSON, or invalid.
pub fn load_spec(home: &Path) -> Result<Spec, String> {
    let b = std::fs::read(home.join(SPEC_FILE)).map_err(|e| format!("{SPEC_FILE}: {e}"))?;
    let v = nmjson::parse(&b).map_err(|e| format!("{SPEC_FILE}: {e}"))?;
    validate(home, &v)
}

/// What one module of one tentacle holds.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Holding {
    pub tentacle: String,
    pub module: String,
    pub kind: String,
    pub state: String,
    pub shared: String,
    /// NPCS cluster groups provisioned into the module.
    pub clusters: Vec<String>,
    /// Groups the population's decomposition does not hold (must be empty).
    pub missing: Vec<String>,
}

/// Geometry for expanding cluster groups.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Geometry {
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
}

/// Expand the specification for every tentacle (lens names): per module the cluster groups it
/// holds, with the tentacle's drone set (and verification set) and its derived L_d (the highest
/// layer of its drone set). `known` is the population's set of cluster groups.
#[must_use]
pub fn holdings(spec: &Spec, drones: &DroneSet, g: Geometry, known: &BTreeSet<String>) -> Vec<Holding> {
    let group = (g.heads / g.kv_heads.max(1)).max(1);
    let mut out = Vec::new();
    for t in nmsubs::lens::NAMES {
        let set = drones.for_tentacle(t);
        let ver = drones.verification_set();
        let ld = set.iter().map(|p| p.0).max().unwrap_or(0);
        for m in &spec.modules {
            let layers: Vec<usize> = match m.layers.as_str() {
                "sense" => (0..=ld).collect(),
                "drone" => set.iter().map(|p| p.0).collect::<BTreeSet<_>>().into_iter().collect(),
                "deep" => (ld + 1..g.layers).collect(),
                _ => Vec::new(),
            };
            let mut cs: BTreeSet<String> = BTreeSet::new();
            for gname in &m.groups {
                match gname.as_str() {
                    "embed" => {
                        cs.insert("embed".into());
                    }
                    "final" => {
                        cs.insert("final".into());
                    }
                    _ => {
                        for &l in &layers {
                            let pick = |s: &[(usize, usize)]| s.iter().filter(|p| p.0 == l).map(|p| p.1).collect::<Vec<usize>>();
                            let names: Vec<String> = match gname.as_str() {
                                "attn" => vec![format!("layer{l}.attn")],
                                "ffn" => vec![format!("layer{l}.ffn")],
                                "head:all" => (0..g.heads).map(|h| format!("layer{l}.head{h}")).collect(),
                                "kv:all" => (0..g.kv_heads).map(|k| format!("layer{l}.kv{k}")).collect(),
                                "head:drone" => pick(set).into_iter().map(|h| format!("layer{l}.head{h}")).collect(),
                                "head:verification" => pick(ver).into_iter().map(|h| format!("layer{l}.head{h}")).collect(),
                                "kv:drone" => pick(set).into_iter().map(|h| format!("layer{l}.kv{}", h / group)).collect(),
                                "kv:verification" => pick(ver).into_iter().map(|h| format!("layer{l}.kv{}", h / group)).collect(),
                                _ => Vec::new(),
                            };
                            cs.extend(names);
                        }
                    }
                }
            }
            let clusters: Vec<String> = cs.into_iter().collect();
            let missing = clusters.iter().filter(|c| !known.contains(*c)).cloned().collect();
            out.push(Holding {
                tentacle: t.to_owned(),
                module: m.name.clone(),
                kind: if m.also.is_empty() { m.kind.clone() } else { format!("{} + {}", m.kind, m.also) },
                state: m.state.clone(),
                shared: m.shared.clone(),
                clusters,
                missing,
            });
        }
    }
    out
}

/// The population's geometry and cluster groups, read from its weight card and file header only
/// (no weights loaded, no device opened).
/// # Errors
/// The population cannot be opened.
pub fn population_groups(home: &Path, id: &str) -> Result<(Geometry, BTreeSet<String>), String> {
    let pop = nmengn::model::Population::open(home, id, false, false).map_err(|e| e.to_string())?;
    let g = Geometry { layers: pop.dims.layers, heads: pop.dims.heads, kv_heads: pop.dims.kv_heads };
    Ok((g, pop.clusters.iter().map(|c| c.group.clone()).collect()))
}

/// Validate the specification and resolve every held cluster group for every tentacle (what
/// `check designs` runs). Returns (modules, tentacles, distinct groups held).
/// # Errors
/// Invalid specification, or a group the population does not hold.
pub fn check_spec_file(home: &Path, v: &Value) -> Result<(usize, usize, usize), String> {
    let spec = validate(home, v)?;
    let drones = kvstore::drone_set(home, &spec.population).map_err(|e| e.to_string())?;
    let (g, known) = population_groups(home, &spec.population)?;
    let hs = holdings(&spec, &drones, g, &known);
    if let Some(h) = hs.iter().find(|h| !h.missing.is_empty()) {
        return Err(format!(
            "{SPEC_FILE}: tentacle {} module {}: clusters not in the population: {}",
            h.tentacle,
            h.module,
            h.missing.join(", ")
        ));
    }
    let distinct: BTreeSet<&String> = hs.iter().flat_map(|h| &h.clusters).collect();
    Ok((spec.modules.len(), nmsubs::lens::NAMES.len(), distinct.len()))
}

/// The anatomy view of the tentacle cores in first form (A10): per tentacle, every module with its
/// kind, state, sharing and the cluster groups provisioned into it; per tentacle its drone heads;
/// and the totals (one copy of each cluster, referenced by every tentacle that holds it).
/// # Errors
/// As `check_spec_file`.
pub fn anatomy_table(home: &Path) -> Result<String, String> {
    use std::fmt::Write as _;
    let spec = load_spec(home)?;
    let drones = kvstore::drone_set(home, &spec.population).map_err(|e| e.to_string())?;
    let (g, known) = population_groups(home, &spec.population)?;
    let hs = holdings(&spec, &drones, g, &known);
    let mut out = String::new();
    let _ = writeln!(
        out,
        "Tentacle cores, first form: {} modules per tentacle from {SPEC_FILE}, population {} ({} layers, {} heads, {} key-value heads); drone sets from {} (read mode {}).",
        spec.modules.len(),
        spec.population,
        g.layers,
        g.heads,
        g.kv_heads,
        drones.source,
        drones.read_mode
    );
    let short = |v: &[String]| match v {
        [] => "none".to_owned(),
        [a] => a.clone(),
        [a, .., z] => format!("{a} .. {z} ({} groups)", v.len()),
    };
    for t in nmsubs::lens::NAMES {
        let set = drones.for_tentacle(t);
        let own = drones.tentacles.contains_key(t);
        let heads: Vec<String> = set.iter().map(|(l, h)| format!("L{l}.h{h}")).collect();
        let _ = writeln!(
            out,
            "\ntentacle {t} (g0_t{}): drone set {} ({} heads: {}); L_d = {}",
            nmsubs::lens::NAMES.iter().position(|x| *x == t).unwrap_or(0) + 1,
            if own { "own" } else { "default" },
            set.len(),
            heads.join(" "),
            set.iter().map(|p| p.0).max().unwrap_or(0)
        );
        let _ = writeln!(out, "| module | kind | state | shared | clusters held |");
        let _ = writeln!(out, "|---|---|---|---|---|");
        for h in hs.iter().filter(|h| h.tentacle == t) {
            let _ = writeln!(out, "| {} | {} | {} | {} | {} |", h.module, h.kind, h.state, h.shared, short(&h.clusters));
        }
    }
    let distinct: BTreeSet<&String> = hs.iter().flat_map(|h| &h.clusters).collect();
    let refs: usize = hs.iter().map(|h| h.clusters.len()).sum();
    let missing: usize = hs.iter().map(|h| h.missing.len()).sum();
    let _ = writeln!(
        out,
        "\ntotals: {} cluster references over {} tentacles resolve to {} distinct cluster groups of the population, each held once and shared by reference; groups not in the population: {missing}",
        refs,
        nmsubs::lens::NAMES.len(),
        distinct.len()
    );
    Ok(out)
}

/// Shortlist size from a tentacle's scores (gate 1): the effective number of chunks carrying the
/// positive score, `ceil(exp(H))` with `H` the entropy of `p_i = max(0, s_i) / sum max(0, s)`, at
/// most the chunk count; `SHORTLIST_FALLBACK` (at most the chunk count) when no score is positive.
/// Returns the size and how it was set.
#[must_use]
pub fn shortlist(scores: &[f32]) -> (usize, String) {
    let n = scores.len();
    let pos: Vec<f64> = scores.iter().map(|s| f64::from(s.max(0.0))).filter(|x| x.is_finite()).collect();
    let total: f64 = pos.iter().sum();
    if n == 0 || total <= 0.0 {
        let k = SHORTLIST_FALLBACK.min(n.max(1));
        return (k, format!("fallback {k}: no chunk above the anchor"));
    }
    let h: f64 = pos.iter().filter(|x| **x > 0.0).map(|x| -(x / total) * (x / total).ln()).sum();
    let eff = h.exp();
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a small positive count")]
    let k = (eff.ceil() as usize).clamp(1, n);
    (k, format!("{k} = ceil(exp(entropy)) of the positive scores ({eff:.2} effective chunks of {n})"))
}

/// What every core of one question shares by reference: the model (one copy of the weights), the
/// device, the stop targets, the anchor's encoding, the question, the verification set, and the
/// counters of device reads.
pub struct CoreShared {
    /// The shared model: one copy of the weights, held centrally and used by reference by every
    /// tentacle's core and drone tip (2026-11-10, direction of Luis Jake Gabriel III: the weights
    /// are not placed in the tentacles; no tentacle owns, stores or loads a copy). `None` in a
    /// replay from a scan dump (no weights, no device: every read comes from the tape).
    pub model: Option<Arc<Model>>,
    pub dev: Device,
    pub stop: Vec<(String, String)>,
    pub anchor: Arc<ChunkKv>,
    pub question: String,
    pub verify: Vec<(usize, usize)>,
    /// (device reads, milliseconds, faults) over all cores.
    pub reads: Mutex<(usize, f64, Vec<String>)>,
    /// Scan dump (2026-11-10, `crate::scan_dump`): content key of each kept encoding by
    /// address, and the tape that records every read (live) or answers it (replay).
    pub names: BTreeMap<usize, String>,
    pub tape: Option<Arc<crate::scan_dump::Tape>>,
    /// Keys-only scan (2026-11-10): fills outlined chunks before a read.
    pub fill: Option<tentacle_pass::Fill>,
    /// Refinement step 2a (2026-11-10, `crate::core_read`): the components of the core's read
    /// that are on (all off: the first form).
    pub read: ReadOptions,
}
impl std::fmt::Debug for CoreShared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoreShared").field("question", &self.question).field("dev", &self.dev).finish_non_exhaustive()
    }
}
impl CoreShared {
    /// One posed read and its content-free read over a short window (the anchor and `chunks` by
    /// reference), under the device gate; the calibrated score of every chunk above the anchor.
    fn read(&self, chunks: &[Arc<ChunkKv>], heads: &[(usize, usize)], posed: &str, null: &str) -> Result<Vec<f32>, Fault> {
        self.read_with(chunks, heads, posed, null, false)
    }
    /// The content-free pose for a posed text ending in `suffix`: the first form's plain pose, or
    /// with component 3 on the pose of the posed text's token count (needs the model's tokenizer;
    /// a replay without a model keeps the plain pose).
    fn null_pose(&self, posed: &str, suffix: &str) -> String {
        match self.model.as_ref().filter(|_| self.read.matched_pose) {
            Some(m) => core_read::matched_null(&|s: &str| m.pop.tokenizer.encode(s).ok().map(|v| v.len()), posed, suffix).0,
            None => core_read::plain_null(suffix),
        }
    }
    /// `read`, and with `votes` the score is taken from the raw scores of the two reads by
    /// attention-weighted token votes with the low-signal positions of the content-free read left
    /// out (`core_read::chunk_scores`, component 2), in the same unit (nats summed over the heads,
    /// above the anchor).
    fn read_with(
        &self,
        chunks: &[Arc<ChunkKv>],
        heads: &[(usize, usize)],
        posed: &str,
        null: &str,
        votes: bool,
    ) -> Result<Vec<f32>, Fault> {
        let t = Instant::now();
        // Scan dump: the read is named by content, heads and texts (a hash, no text kept).
        let key = self.tape.as_ref().map(|_| {
            crate::scan_dump::read_key(&self.names, chunks, heads, posed, &if votes { format!("{null}\nvotes") } else { null.to_owned() })
        });
        if let (Some(tape), Some(k)) = (self.tape.as_ref().filter(|t| t.replaying()), key.as_ref()) {
            let v = tape.get(k).ok_or_else(|| bad("read not in the scan dump"))?;
            if let Ok(mut r) = self.reads.lock() {
                r.0 += 2;
            }
            return Ok(v);
        }
        let model = self.model.as_ref().ok_or_else(|| bad("no model for a device read"))?;
        let gate = (self.dev == Device::Gpu).then(|| nmengn::placement::gpu_gate().enter("tentacle core read", true));
        let mut parts = vec![Arc::clone(&self.anchor)];
        match self.fill.as_ref() {
            Some(f) => parts.extend((f.0)(chunks)?),
            None => parts.extend(chunks.iter().cloned()),
        }
        let w = Arc::new(Window::build(model, &parts)?);
        let out: Vec<f32> = if votes {
            let pr = core_read::raw_read(model, self.dev, &w, &self.stop, heads, posed)?;
            let nr = core_read::raw_read(model, self.dev, &w, &self.stop, heads, null)?;
            drop(gate);
            let mask = core_read::low_signal(&nr, &w.parts);
            core_read::chunk_scores(&pr, &nr, &w.parts, core_read::Kind::Weighted, Some(&mask))
        } else {
            let (_, pc) = tentacle_pass::drone_read_logits(model, self.dev, &w, &self.stop, heads, posed)?;
            let (_, nc) = tentacle_pass::drone_read_logits(model, self.dev, &w, &self.stop, heads, null)?;
            drop(gate);
            let cal = tentacle_pass::calibrated(&pc, &nc, w.parts.len());
            let anchor = cal.first().copied().unwrap_or(0.0);
            cal.into_iter().skip(1).map(|x| x - anchor).collect()
        };
        if let Ok(mut r) = self.reads.lock() {
            r.0 += 2;
            r.1 += t.elapsed().as_secs_f64() * 1000.0;
        }
        if let (Some(tape), Some(k)) = (self.tape.as_ref(), key.as_ref()) {
            tape.put(k, out.clone());
        }
        Ok(out)
    }
    fn fault(&self, m: String) {
        if let Ok(mut r) = self.reads.lock() {
            r.2.push(m);
        }
    }
}

/// One tentacle core.
#[derive(Debug)]
pub struct TentacleCore {
    /// The tentacle's G0 state (`g0_t{n}`): held spans, drone table, focus, report, inbox.
    pub g0: G0,
    shared: Arc<CoreShared>,
    /// The tentacle's drone set (its own when the file names one, else the default).
    pub drones: Vec<(usize, usize)>,
    /// The window the scan read (chunks by reference).
    pub window: Arc<Window>,
    /// Kept encodings of the held chunks, in window order (references into the store).
    chunks: Vec<Arc<ChunkKv>>,
    /// Reported score per held chunk: calibrated, above the anchor.
    pub scores: Vec<f32>,
    /// Shortlist size and how it was set (gate 1).
    pub depth_why: String,
    rereads: u32,
    reads: u32,
    /// Spread (standard deviation over the window's parts, anchor included) of the content-free
    /// read summed over the drone heads: the noise floor of the `NullMargin` normalisation.
    pub null_spread: Option<f32>,
    /// Refinement step 1: the attention-mass score per held chunk, the window's content-free
    /// baseline (the anchor's calibrated score), the lens and each chunk's rank in its lens's
    /// assignment order (when the scan recorded it), for the evidence records.
    mass: Vec<f32>,
    anchor: f32,
    lens: String,
    lens_rank: Vec<Option<usize>>,
    /// Refinement step 2a: the full read's reported score per held chunk, kept unchanged beside
    /// the second look's; and the second look (`None` until one is taken).
    full: Vec<f32>,
    look: Option<Look>,
}

/// A second look (refinement step 2a, component 1): the held chunks it read (window order), their
/// scores from the short read, unscaled, and how the shortlist size was set.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Look {
    pub pick: Vec<usize>,
    pub scores: Vec<f32>,
    pub why: String,
}

/// Re-reads each core can make for one question (first form: one, a short-window re-read of its
/// shortlist and focus).
pub const REREADS: u32 = 1;

impl TentacleCore {
    /// A core over one window of a scan made with logits and kept windows (`Settings::core`).
    /// # Errors
    /// The scan holds no window or no calibrated rows for this tentacle.
    pub fn from_scan(
        sub: &Substrate,
        scan: &Scan,
        w: &WindowScan,
        drones: Vec<(usize, usize)>,
        shared: Arc<CoreShared>,
    ) -> Result<Self, Fault> {
        let window = w
            .window
            .clone()
            .ok_or_else(|| bad(format!("tentacle {}: the scan kept no window (the core needs Settings::core)", w.tentacle)))?;
        let cal = w.calibrated_parts();
        if cal.len() != w.parts.len() || cal.is_empty() {
            return Err(bad(format!("tentacle {}: no calibrated rows (the scan recorded no logits)", w.tentacle)));
        }
        let anchor = cal.first().copied().unwrap_or(0.0);
        let scores: Vec<f32> = cal.iter().skip(1).map(|x| x - anchor).collect();
        let held: Vec<HeldChunk> = w
            .place
            .iter()
            .zip(w.parts.iter().skip(1))
            .filter_map(|(c, &(_, tokens))| {
                let ch = sub.chunks.get(*c)?;
                let file = sub.files.get(ch.file)?.path.replace('\\', "/");
                Some(HeldChunk { span: SpanRef { file, start: ch.start, end: ch.end, key: ch.key.clone() }, chunk: *c, tokens })
            })
            .collect();
        if held.len() != scores.len() {
            return Err(bad(format!("tentacle {}: {} held chunks for {} scores", w.tentacle, held.len(), scores.len())));
        }
        // Drone table: per head, the calibrated reading of each chunk above the anchor's.
        let heads = w.cal_heads();
        let n = w.parts.len();
        let rows: Vec<Vec<f32>> = (0..heads)
            .map(|h| {
                let at = |k: usize| w.cal_posed.get(h * n + k).copied().unwrap_or(0.0) - w.cal_null.get(h * n + k).copied().unwrap_or(0.0);
                (1..n).map(|k| at(k) - at(0)).collect()
            })
            .collect();
        // Gate 1 by the rule in service (`nmbrain::g1::rules`; first form: `shortlist`).
        let (depth, depth_why) = nmbrain::g1::shortlist_by(nmbrain::g1::rules().shortlist, &scores, SHORTLIST_FALLBACK);
        let mut rec = scan.records.iter().find(|r| usize::from(r.tentacle) == w.tentacle).cloned().unwrap_or_default();
        rec.baseline = 0.0;
        let g0 = G0::from_window(
            &rec,
            held,
            vec![(Score::Calibrated.name().to_owned(), scores.clone()), (Score::Mass.name().to_owned(), w.scores(Score::Mass))],
            &rows,
            depth,
        );
        let chunks = window.chunks.iter().skip(1).cloned().collect();
        let null_parts: Vec<f32> =
            (0..n).map(|k| (0..heads).map(|h| w.cal_null.get(h * n + k).copied().unwrap_or(0.0)).sum::<f32>()).collect();
        #[expect(clippy::cast_precision_loss, reason = "a part count")]
        let mean = null_parts.iter().sum::<f32>() / n.max(1) as f32;
        #[expect(clippy::cast_precision_loss, reason = "a part count")]
        let sd = (null_parts.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n.max(1) as f32).sqrt();
        let null_spread = (n > 1 && sd.is_finite() && sd > 0.0).then_some(sd);
        let mass = w.scores(Score::Mass);
        let lens = nmsubs::lens::NAMES.get(w.tentacle).copied().unwrap_or("lens").to_owned();
        let lens_rank = (0..scores.len()).map(|k| w.lens_rank.get(k).copied()).collect();
        let full = scores.clone();
        Ok(Self {
            g0,
            shared,
            drones,
            window,
            chunks,
            scores,
            depth_why,
            rereads: 0,
            reads: 0,
            null_spread,
            mass,
            anchor,
            lens,
            lens_rank,
            full,
            look: None,
        })
    }
    /// Drone heads behind the scores.
    #[must_use]
    pub fn heads(&self) -> usize {
        self.g0.readings.drones.max(1)
    }
    /// Held chunk order by score (focus files first), ties in window order.
    fn order(&self) -> Vec<usize> {
        let aimed = |i: usize| self.g0.held.get(i).is_some_and(|h| self.g0.focus.files.contains(&h.span.file));
        let mut idx: Vec<usize> = (0..self.scores.len()).collect();
        // Refinement step 2a: the chunks of a second look come before the others (first place
        // inside the tentacle is decided by the short read); without one this term is equal for
        // every chunk and the order is the first form's.
        let looked = |i: usize| self.look.as_ref().is_some_and(|l| l.pick.contains(&i));
        idx.sort_by(|&a, &b| {
            aimed(b)
                .cmp(&aimed(a))
                .then(looked(b).cmp(&looked(a)))
                .then(
                    self.scores
                        .get(b)
                        .copied()
                        .unwrap_or(f32::NEG_INFINITY)
                        .total_cmp(&self.scores.get(a).copied().unwrap_or(f32::NEG_INFINITY)),
                )
                .then(a.cmp(&b))
        });
        idx
    }
    /// The second look (component 1): the shortlist by the full read's mass (size from those
    /// scores, `core_read::look_size`), read once more as one short window with the question from
    /// the encodings already held (verification set; nothing encoded), its scores taken unscaled.
    /// The full read's scores stay in `full`. A fault leaves the first form's order.
    fn second_look(&mut self) {
        let (k, why) = core_read::look_size(&self.mass);
        let pick = core_read::look_pick(&self.mass, k);
        let chunks: Vec<Arc<ChunkKv>> = pick.iter().filter_map(|i| self.chunks.get(*i).cloned()).collect();
        if chunks.len() != pick.len() || pick.is_empty() {
            return;
        }
        let posed = format!("\nQuestion: {}", self.shared.question);
        let null = self.shared.null_pose(&posed, "");
        match self.shared.read_with(&chunks, &self.shared.verify, &posed, &null, self.shared.read.votes) {
            Ok(s) if s.len() == pick.len() => self.take_look(pick, s, why),
            Ok(s) => self.shared.fault(format!("{} second look: {} scores for {} chunks", self.g0.id, s.len(), pick.len())),
            Err(e) => self.shared.fault(format!("{} second look: {}", self.g0.id, e.message)),
        }
    }
    /// Take a short read's scores for the chunks `pick`, unscaled.
    fn take_look(&mut self, pick: Vec<usize>, s: Vec<f32>, why: String) {
        for (i, v) in pick.iter().zip(&s) {
            if let Some(x) = self.scores.get_mut(*i) {
                *x = *v;
            }
        }
        self.look = Some(Look { pick, scores: s, why });
    }
    fn make_report(&mut self) -> Report {
        if self.reads == 0 && self.shared.read.second_look && self.look.is_none() {
            self.second_look();
        }
        self.reads += 1;
        // Confidence by the rule in service (first form: `logistic(score / heads)`).
        let conf = nmbrain::g1::confidences_with(nmbrain::g1::rules().confidence, &self.scores, self.heads(), self.null_spread);
        let order = self.order();
        let depth = self.g0.depth.max(1);
        let best_score = self.scores.iter().copied().filter(|x| x.is_finite()).fold(f32::NEG_INFINITY, f32::max);
        let no_information = best_score.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater);
        let best: Vec<SpanReport> = order
            .iter()
            .take(depth)
            .enumerate()
            .filter_map(|(place, &i)| {
                let h = self.g0.held.get(i)?;
                let s = self.scores.get(i).copied().unwrap_or(f32::NEG_INFINITY);
                Some(SpanReport { span: h.span.clone(), score: s, confidence: conf.get(i).copied().unwrap_or(0.0), place })
            })
            .collect();
        let next = order.iter().skip(depth).take(depth).filter_map(|&i| self.g0.held.get(i).map(|h| h.span.clone())).collect();
        let confidence = if no_information { 0.0 } else { best.first().map_or(0.0, |b| b.confidence) };
        let facts = self.facts(&conf);
        self.g0.report = Report {
            from: self.g0.id.clone(),
            tentacle: self.g0.tentacle,
            best,
            no_information,
            best_score: if best_score.is_finite() { best_score } else { 0.0 },
            baseline: 0.0,
            confidence,
            next,
            focus: self.g0.focus.clone(),
            reads: self.reads,
            facts,
            read: self.read_text(),
            ..Report::default()
        };
        self.g0.report.clone()
    }
    /// The evidence of every held chunk (refinement step 1): tokens, rank and margin by the
    /// reported score, every score kind (calibrated above the anchor, mass, confidence), the
    /// window's content-free baseline and spread, rank by mass, lens and lens rank.
    /// What the report and the evidence records say about the read: the components on, and the
    /// second look's shortlist (empty for the first form).
    fn read_text(&self) -> String {
        let on = self.shared.read.text();
        match self.look.as_ref() {
            Some(l) if !on.is_empty() => format!("{on}; shortlist {}", l.why),
            _ => on,
        }
    }
    fn facts(&self, conf: &[f32]) -> BTreeMap<String, nmbrain::evidence::SpanFacts> {
        let by = |v: &[f32]| {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|&a, &b| {
                v.get(b).copied().unwrap_or(f32::NEG_INFINITY).total_cmp(&v.get(a).copied().unwrap_or(f32::NEG_INFINITY)).then(a.cmp(&b))
            });
            idx
        };
        let order = by(&self.scores);
        let mass_order = by(&self.mass);
        let mut out = BTreeMap::new();
        for (rank, &i) in order.iter().enumerate() {
            let Some(h) = self.g0.held.get(i) else { continue };
            let s = self.scores.get(i).copied().unwrap_or(f32::NEG_INFINITY);
            let below = order.get(rank + 1).and_then(|j| self.scores.get(*j)).copied();
            // With a second look the full read's score stays under `calibrated` and the short
            // read's unscaled score is added beside it for the chunks it read.
            let full = if self.look.is_some() { self.full.get(i).copied().unwrap_or(s) } else { s };
            let mut scores = vec![
                ("calibrated".to_owned(), full),
                ("mass".to_owned(), self.mass.get(i).copied().unwrap_or(f32::NEG_INFINITY)),
                ("confidence".to_owned(), conf.get(i).copied().unwrap_or(0.0)),
            ];
            if self.look.as_ref().is_some_and(|l| l.pick.contains(&i)) {
                scores.push(("second_look".to_owned(), s));
            }
            out.insert(
                h.span.key.clone(),
                nmbrain::evidence::SpanFacts {
                    tokens: h.tokens,
                    window_rank: rank,
                    margin: below.map_or(0.0, |b| if s.is_finite() && b.is_finite() { s - b } else { 0.0 }),
                    scores,
                    baseline: self.anchor,
                    null_spread: self.null_spread,
                    mass_rank: mass_order.iter().position(|&k| k == i).unwrap_or(usize::MAX),
                    lens: self.lens.clone(),
                    lens_rank: self.lens_rank.get(i).copied().flatten(),
                    read: self.read_text(),
                    ..nmbrain::evidence::SpanFacts::default()
                },
            );
        }
        out
    }
    /// The focus line of a re-read pose (empty without files or terms).
    fn focus_line(f: &Focus) -> String {
        let mut parts = Vec::new();
        if !f.files.is_empty() {
            parts.push(format!("files {}", f.files.join(", ")));
        }
        if !f.terms.is_empty() {
            parts.push(format!("terms {}", f.terms.join(" ")));
        }
        if parts.is_empty() { String::new() } else { format!("\nFocus: {}", parts.join("; ")) }
    }
}

impl TentacleSide for TentacleCore {
    fn id(&self) -> &str {
        &self.g0.id
    }
    fn held(&self) -> Vec<SpanRef> {
        self.g0.held.iter().map(|h| h.span.clone()).collect()
    }
    fn options(&self) -> usize {
        usize::try_from(REREADS.saturating_sub(self.rereads)).unwrap_or(0)
    }
    fn report(&mut self) -> Report {
        if self.reads == 0 { self.make_report() } else { self.g0.report.clone() }
    }
    /// A new pose over the kept encoded chunks: a short window of the anchor, the shortlist and
    /// the held chunks of the focus files (window order, by reference, nothing encoded), read with
    /// the verification set under the question plus the focus, and its content-free read on the
    /// same window. Those chunks take the new order at their full-window mean; the others keep theirs.
    fn reread(&mut self, focus: &Focus) -> Report {
        if self.rereads >= REREADS {
            return self.report();
        }
        self.rereads += 1;
        self.g0.focus = Focus {
            lens: self.g0.focus.lens.clone(),
            pose: self.g0.focus.pose.clone(),
            statistic: Score::Calibrated.name().to_owned(),
            files: focus.files.clone(),
            terms: focus.terms.clone(),
        };
        let depth = self.g0.depth.max(1);
        let top: BTreeSet<usize> = self.order().into_iter().take(depth).collect();
        let pick: Vec<usize> = (0..self.chunks.len())
            .filter(|i| top.contains(i) || self.g0.held.get(*i).is_some_and(|h| focus.files.contains(&h.span.file)))
            .collect();
        let chunks: Vec<Arc<ChunkKv>> = pick.iter().filter_map(|i| self.chunks.get(*i).cloned()).collect();
        let line = Self::focus_line(focus);
        let posed = format!("\nQuestion: {}{line}", self.shared.question);
        let null = self.shared.null_pose(&posed, &line);
        match self.shared.read_with(&chunks, &self.shared.verify, &posed, &null, self.shared.read.votes) {
            // Refinement step 2a (F13): with the second look on, a re-read's scores are taken
            // unscaled with their own normaliser and its chunks lead the order; the full read's
            // values stay beside them. The first form below (mean set back) stays selectable.
            Ok(s) if self.shared.read.second_look && s.len() == pick.len() => {
                let why = format!("re-read on instruction: {} chunks (shortlist and focus files)", pick.len());
                self.take_look(pick, s, why);
            }
            Ok(s) => {
                // The focus line changes the pose, which moves every score of the short window by
                // a common amount: the re-read gives the order among the picked chunks, and their
                // mean is set back to their full-window mean so they stay on the scale every other
                // chunk and every other tentacle is on.
                #[expect(clippy::cast_precision_loss, reason = "a chunk count")]
                let n = s.len().max(1) as f32;
                let old = pick.iter().filter_map(|i| self.scores.get(*i)).sum::<f32>() / n;
                let new = s.iter().sum::<f32>() / n;
                for (i, v) in pick.iter().zip(s) {
                    if let Some(x) = self.scores.get_mut(*i) {
                        *x = v - new + old;
                    }
                }
            }
            Err(e) => self.shared.fault(format!("{} re-read: {}", self.g0.id, e.message)),
        }
        self.make_report()
    }
    /// A focused read of one span: the anchor and that chunk alone, posed with `about` and read
    /// content-free; its score above the anchor, its confidence, and its rank in the window.
    fn answer(&mut self, span: &SpanRef, about: &str) -> Answer {
        let Some(i) = self.g0.held.iter().position(|h| h.span.key == span.key || h.span.overlaps(span)) else {
            return Answer {
                from: self.g0.id.clone(),
                span: span.clone(),
                held: false,
                score: 0.0,
                confidence: 0.0,
                rank: None,
                no_information: true,
            };
        };
        let rank = self.order().iter().position(|&k| k == i);
        let held_span = self.g0.held.get(i).map_or_else(|| span.clone(), |h| h.span.clone());
        let chunk: Vec<Arc<ChunkKv>> = self.chunks.get(i).cloned().into_iter().collect();
        let (score, ok) = match self.shared.read(&chunk, &self.drones, &format!("\nQuestion: {about}"), "\nQuestion: N/A") {
            Ok(v) => (v.first().copied().unwrap_or(0.0), true),
            Err(e) => {
                self.shared.fault(format!("{} answer: {}", self.g0.id, e.message));
                (self.scores.get(i).copied().unwrap_or(0.0), false)
            }
        };
        let _ = ok;
        Answer {
            from: self.g0.id.clone(),
            span: held_span,
            held: true,
            score,
            confidence: nmbrain::g1::confidence_of(nmbrain::g1::rules().confidence, &self.scores, i, score, self.heads(), self.null_spread),
            rank,
            no_information: score.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater),
        }
    }
    fn inbox_mut(&mut self) -> &mut Inbox {
        &mut self.g0.inbox
    }
    fn inbox(&self) -> &Inbox {
        &self.g0.inbox
    }
}

/// The reference graph of the substrate (`nmsubs::refgraph`, held in RAM, rebuilt on sync) as
/// G1 sees it (refinement step 1, item 4): ranks and relations by content key.
pub struct SubGraph(pub Arc<nmsubs::refgraph::RefGraph>);

/// The reference graph plus co-citation links (2026-11-12, memory design: "co-citation links act
/// as priors" at G1): two chunks cited together in past answers are linked, by content key, with
/// a weight (how often). A span's structure value is the reference graph's, plus what it receives
/// from the seeds it was co-cited with (seed weight times link weight over the strongest link).
pub struct WithCoCitation {
    pub base: SubGraph,
    pub links: BTreeMap<String, Vec<(String, f32)>>,
}
impl nmbrain::evidence::Structure for WithCoCitation {
    fn rank(&self, seeds: &[(String, f32)], ask: &[String]) -> Vec<f32> {
        let base = self.base.rank(seeds, ask);
        let top = self.links.values().flatten().map(|(_, w)| *w).fold(0.0_f32, f32::max).max(1e-9);
        ask.iter()
            .zip(base)
            .map(|(k, b)| {
                let co: f32 = seeds
                    .iter()
                    .filter(|(s, _)| s != k)
                    .map(|(s, w)| {
                        w.max(0.0) * self.links.get(s).and_then(|l| l.iter().find(|(x, _)| x == k)).map_or(0.0, |(_, lw)| lw / top)
                    })
                    .sum();
                b + co
            })
            .collect()
    }
    fn related(&self, key: &str) -> Vec<(String, String, f32)> {
        let mut r = self.base.related(key);
        r.extend(self.links.get(key).into_iter().flatten().map(|(k, w)| (k.clone(), "co-cited".to_owned(), *w)));
        r.sort_by(|a, b| b.2.total_cmp(&a.2));
        r
    }
    fn describe(&self) -> String {
        format!("{}; {} co-cited chunks", self.base.describe(), self.links.len())
    }
}
impl nmbrain::evidence::Structure for SubGraph {
    /// Personalised rank from the seeds, less each seed's own restart share, so a span's value is
    /// what it receives from the other seeds through references (not its own seeding).
    fn rank(&self, seeds: &[(String, f32)], ask: &[String]) -> Vec<f32> {
        let s: Vec<(usize, f32)> = seeds.iter().filter_map(|(k, w)| self.0.index_of(k).map(|i| (i, *w))).collect();
        let total: f32 = s.iter().map(|x| x.1.max(0.0)).sum();
        let (r, _) = self.0.rank(&s);
        ask.iter()
            .map(|k| {
                let Some(i) = self.0.index_of(k) else { return 0.0 };
                let own: f32 = s.iter().filter(|x| x.0 == i).map(|x| x.1.max(0.0)).sum();
                let restart = if total > 0.0 { (1.0 - nmsubs::refgraph::DAMPING) * own / total } else { 0.0 };
                (r.get(i).copied().unwrap_or(0.0) - restart).max(0.0)
            })
            .collect()
    }
    fn related(&self, key: &str) -> Vec<(String, String, f32)> {
        let Some(i) = self.0.index_of(key) else { return Vec::new() };
        self.0.neighbours(i).into_iter().filter_map(|(j, r, w)| self.0.keys.get(j).map(|k| (k.clone(), r.name().to_owned(), w))).collect()
    }
    fn describe(&self) -> String {
        format!(
            "reference graph: substrate version {}, {} chunks, {} item names, {} edges, {} bytes in RAM, built in {:.0} ms",
            self.0.version,
            self.0.keys.len(),
            self.0.names,
            self.0.edges,
            self.0.bytes(),
            self.0.build_ms
        )
    }
}

/// G1's whole-code-base layer from the substrate (files and item signatures).
#[must_use]
pub fn codebase(sub: &Substrate) -> WholeCodebase {
    let files: Vec<String> = sub.files.iter().map(|f| f.path.replace('\\', "/")).collect();
    let items: Vec<ItemSig> = sub
        .symbols
        .iter()
        .map(|s| ItemSig {
            name: s.name.clone(),
            kind: s.kind.clone(),
            file: s.file.replace('\\', "/"),
            line: s.line,
            signature: s.text.clone(),
        })
        .collect();
    WholeCodebase::build(files, items)
}

/// What one core run produced, for the pass's note and the measurement.
#[derive(Debug, Default)]
pub struct CoreRun {
    pub outcome: Outcome,
    pub budget: Option<Budget>,
    /// Per tentacle: (g0 id, shortlist and how it was set, G0 bytes in RAM).
    pub cores: Vec<(String, String, usize)>,
    /// Device reads the loop made, their milliseconds, and faults.
    pub reads: usize,
    pub read_ms: f64,
    pub faults: Vec<String>,
    /// Loop time.
    pub ms: f64,
    /// Envelopes checked at the message channel (behavioural) and packets at the chain input
    /// (sensory); wrong-class hand-overs refused (must be 0).
    pub envelopes: usize,
    pub packets: usize,
    pub refused: usize,
    /// First reports reporting no information, and per core its first ranking (chunk index,
    /// reported score), best first.
    pub no_info: usize,
    pub ranked: Vec<(String, Vec<(usize, f32)>)>,
    /// Per core (g0 id): its first report's no-information flag.
    pub no_info_by: BTreeMap<String, bool>,
}

/// Build the cores of a scan, run the G1 loop and return its outcome. The budget: forward count
/// from the pack's token room over the mean held chunk tokens, where the room is what today's cut
/// forwards (`per_tentacle` chunks of mean size per tentacle) unless `pack_tokens` names one; no
/// time limit (the loop's derived ceiling bounds it). Values recorded in `CoreRun::budget`.
/// # Errors
/// No model, no windows, or a scan without logits and kept windows.
pub fn run(home: &Path, sub: &Substrate, scan: &Scan, settings: &Settings, pack_tokens: Option<usize>) -> Result<CoreRun, Fault> {
    run_taped(home, sub, scan, settings, pack_tokens, None)
}

/// `run` with a scan dump tape (2026-11-10, `crate::scan_dump`): a recording tape keeps every
/// device read of the loop; a replay tape answers them, and then no model is loaded and no device
/// is used.
/// # Errors
/// As `run`.
pub fn run_taped(
    home: &Path,
    sub: &Substrate,
    scan: &Scan,
    settings: &Settings,
    pack_tokens: Option<usize>,
    tape: Option<Arc<crate::scan_dump::Tape>>,
) -> Result<CoreRun, Fault> {
    let started = Instant::now();
    let replay = tape.as_ref().is_some_and(|t| t.replaying());
    let model = if replay { None } else { Some(crate::neural::model(home)?) };
    let dev = if model.as_ref().is_some_and(|m| m.pop.gpu.is_some()) { Device::Gpu } else { Device::Cpu };
    let sets = tentacle_pass::drones(home);
    let stop = match model.as_ref() {
        Some(m) => kvstore::score_targets(m, sets.last_layer())?,
        None => Vec::new(),
    };
    let anchor = scan
        .windows
        .iter()
        .find_map(|w| w.window.as_ref().and_then(|x| x.chunks.first().cloned()))
        .ok_or_else(|| bad("no window with an anchor in the scan"))?;
    let names = if tape.is_some() { crate::scan_dump::names_of(sub, scan) } else { BTreeMap::new() };
    let shared = Arc::new(CoreShared {
        model,
        dev,
        stop,
        anchor,
        question: scan.question.clone(),
        verify: sets.verification_set().to_vec(),
        reads: Mutex::new((0, 0.0, Vec::new())),
        names,
        tape,
        fill: scan.fill.clone(),
        read: settings.read,
    });
    let mut cores: Vec<TentacleCore> = Vec::new();
    for w in &scan.windows {
        let lens = nmsubs::lens::NAMES.get(w.tentacle).copied().unwrap_or("lens");
        cores.push(TentacleCore::from_scan(sub, scan, w, sets.for_tentacle(lens).to_vec(), Arc::clone(&shared))?);
    }
    let facts: Vec<(String, String, usize)> = cores.iter().map(|c| (c.g0.id.clone(), c.depth_why.clone(), c.g0.bytes())).collect();
    // First reports (no device read: the scan's scores); G1's first round takes these.
    let mut no_info = 0;
    let mut ranked = Vec::new();
    let mut no_info_by = BTreeMap::new();
    for c in &mut cores {
        let flag = c.report().no_information;
        no_info += usize::from(flag);
        no_info_by.insert(c.g0.id.clone(), flag);
        let order = c.order();
        ranked.push((c.g0.id.clone(), order.iter().filter_map(|&i| Some((c.g0.held.get(i)?.chunk, c.scores.get(i).copied()?))).collect()));
    }
    let tokens: Vec<usize> = scan.windows.iter().flat_map(|w| w.parts.iter().skip(1).map(|p| p.1)).collect();
    let mean = tokens.iter().sum::<usize>().checked_div(tokens.len()).unwrap_or(0);
    let room = pack_tokens.unwrap_or(settings.per_tentacle.max(1) * scan.windows.len() * mean);
    // Refinement step 1: under the queue rule the round capacity is the receiving room in tokens
    // (`nmbrain::evidence::Capacity`), and G1 holds the reference graph over items.
    let budget = Budget::derive(scan.windows.len(), Some(room), Some(mean), None).with_room(pack_tokens);
    let mut boxes: Vec<Box<dyn TentacleSide>> = cores.into_iter().map(|c| Box::new(c) as Box<dyn TentacleSide>).collect();
    let structure = (nmbrain::g1::rules().forward == nmbrain::g1::ForwardRule::Queue)
        .then(|| nmbrain::g1::StructureRef(Arc::new(SubGraph(sub.refs())) as Arc<dyn nmbrain::evidence::Structure>));
    let mut g1 = G1::new(codebase(sub)).with_structure(structure);
    let outcome = g1.run(&scan.question, &mut boxes, &budget);
    let (reads, read_ms, faults) = shared.reads.lock().map(|r| r.clone()).unwrap_or_default();
    let mut refused = 0;
    let envs = outcome.handover.messages();
    for e in &envs {
        refused += usize::from(nmtype::class::boundaries::MESSAGE_CHANNEL.admit(e).is_err());
    }
    Ok(CoreRun {
        no_info,
        ranked,
        no_info_by,
        envelopes: envs.len(),
        refused,
        budget: Some(budget),
        cores: facts,
        reads,
        read_ms,
        faults,
        ms: started.elapsed().as_secs_f64() * 1000.0,
        outcome,
        ..CoreRun::default()
    })
}

/// The pass's judge when the core is on: the G1 loop's hand-over becomes the forwarded chunks
/// (sensory `chunk` packets through the bridge, in G1's order), and each tentacle's window record
/// carries its core's report. Returns the pass and the run's facts.
/// # Errors
/// As `run`.
pub fn judge(home: &Path, sub: &Substrate, scan: &Scan, settings: &Settings) -> Result<(Pass, CoreRun), Fault> {
    judge_taped(home, sub, scan, settings, None)
}

/// `judge` with a scan dump tape (`run_taped`).
/// # Errors
/// As `run`.
pub fn judge_taped(
    home: &Path,
    sub: &Substrate,
    scan: &Scan,
    settings: &Settings,
    tape: Option<Arc<crate::scan_dump::Tape>>,
) -> Result<(Pass, CoreRun), Fault> {
    let mut cr = run_taped(home, sub, scan, settings, None, tape)?;
    let h = &cr.outcome.handover;
    let by_key: BTreeMap<&str, usize> = sub.chunks.iter().enumerate().map(|(i, c)| (c.key.as_str(), i)).collect();
    let mut cands = Vec::new();
    for (place, f) in h.items.iter().enumerate() {
        let Some(&chunk) = by_key.get(f.span.key.as_str()) else { continue };
        let tentacle = f.from.strip_prefix("g0_t").and_then(|n| n.parse::<usize>().ok()).map_or(0, |n| n.saturating_sub(1));
        cands.push(Cand { tentacle, chunk, score: f.combined, place, lines: Vec::new() });
    }
    let order: Vec<usize> = (0..cands.len()).collect();
    // Refinement step 1: under the queue rule the hand-over travels as its evidence set and the
    // packets carry their records' relevance (the bridge no longer reduces them to a place).
    let stack = nmbrain::g1::rules().forward == nmbrain::g1::ForwardRule::Queue;
    let packets = if stack {
        tentacle_pass::bridge_evidence(sub, &cands, &cr.outcome.evidence, settings)
    } else {
        tentacle_pass::bridge(sub, &cands, &order, settings)
    };
    if stack {
        cr.refused += usize::from(nmtype::class::boundaries::CHAIN_INPUT.admit(&cr.outcome.evidence).is_err());
    }
    for p in &packets {
        cr.refused += usize::from(nmtype::class::boundaries::CHAIN_INPUT.admit(p).is_err());
    }
    cr.packets = packets.len();
    let mut records = scan.records.clone();
    for r in &mut records {
        let id = g0::g0_id(r.tentacle);
        let found = h.items.iter().filter(|f| f.from == id).count();
        let mine = h.items.iter().filter(|f| f.from == id).map(|f| f.confidence).fold(0.0_f32, f32::max);
        r.findings = found;
        r.best = mine;
        // A window with no chunks has no core: nothing to read, so no information.
        r.no_information = cr.no_info_by.get(&id).copied().unwrap_or(true);
    }
    let rounds = h.rounds;
    let note = format!(
        "tentacle core on {} (first form): {} windows, {} tokens held, score calibrated (above the anchor), {} first reports with no information; G1 loop: {} rounds (ceiling {}), stop: {}; forwarded {} spans ({}); isolation gate {}; device reads in the loop {} ({:.0} ms); faults {}; envelopes checked behavioural {} and packets checked sensory {}, refused {}; shortlists: {}; parallel rule: processor phases one thread per tentacle, device reads one at a time at the device gate; loop {:.0} ms; scan: encoded {} new tokens in {:.0} ms, reads {:.0} ms",
        scan.device,
        scan.windows.len(),
        scan.window_tokens,
        cr.no_info,
        rounds,
        cr.outcome.ceiling,
        h.stop,
        h.items.len(),
        cr.budget.as_ref().map_or_else(String::new, |b| b.why.clone()),
        h.gate_direct,
        cr.reads,
        cr.read_ms,
        if cr.faults.is_empty() { "none".to_owned() } else { cr.faults.join("; ") },
        cr.envelopes,
        cr.packets,
        cr.refused,
        cr.cores.iter().map(|(id, why, _)| format!("{id} {why}")).collect::<Vec<_>>().join("; "),
        cr.ms,
        scan.enc_tokens,
        scan.enc_ms,
        scan.read_ms,
    );
    let note = if stack {
        let (ft, qt) = cr.outcome.evidence.tokens();
        format!(
            "{note}; context stack: evidence set ({}), {} forwarded ({ft} tokens) and {} queued ({qt} tokens), {}",
            cr.outcome.evidence.rule,
            cr.outcome.evidence.forwarded.len(),
            cr.outcome.evidence.queue.len(),
            cr.outcome.evidence.capacity.why
        )
    } else {
        note
    };
    let evidence = stack.then(|| Arc::new(Mutex::new(cr.outcome.evidence.clone())));
    let pass = Pass { packets, windows: records, order: cands.iter().map(|c| c.chunk).collect(), note, evidence };
    Ok((pass, cr))
}

// ---------------------------------------------------------------------------------------------
// The assembled core (2026-11-19, lane DR; master plan 3.3 steps 4, 5, 7, 8 and "the core's
// components"; Phase 0 audit rules 3.3.1 own core, 3.3.3 components from the anatomy, 3.3.13 read
// in steps, 3.3.16 one tentacle system). The core is built from the components the specification
// names (`designs/tentacle-core.json`, each module's `component`), in the order listed: attention
// (the periphery read, the self-started integration, the deep check), feed-forward (the shared
// deep layers the integration and the deep check run), gating and routing (gate 1 by posterior
// mass, the deep gate, the local decisions, the route to g0), memory (the kept pieces and the
// report) and the recurrent path (`Carried`, the core's state across rounds through its g0). The
// device reads come through `CoreIo`, so one core serves the Team's tentacles whatever reader the
// flow gives them. Constants are stated, not fitted.
// ---------------------------------------------------------------------------------------------

/// Share of the window's posterior mass the reported pieces hold (gate 1's rule, the design's
/// stated fallback for `T.decide`).
pub const REPORT_MASS: f32 = 0.9;
/// A reported piece whose drone anomaly is at least this is flagged in the next look.
pub const ANOMALY_FLAG: f32 = 0.8;
/// Pieces the deep check reads at most (design section 3.4 item 5).
pub const DEEP_MAX: usize = 2;
/// The three content-free forms of the deep check's calibration (Zhao et al. 2021, contextual
/// calibration: the label bias under content-free inputs is removed; three forms averaged).
pub const NULL_FORMS: [&str; 3] = ["N/A", "", "[MASK]"];

/// A component of the assembled core (what a specification module binds to).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Component {
    /// The drones' pass over each piece alone (held, not run per round).
    Sense,
    /// The kept pieces, their local memory and the kept content-free reads (held).
    Memory,
    /// The periphery read of the whole window under the pose (its scores arrive in `Piece`).
    Periphery,
    /// Calibration: posed minus content-free (the scores arrive calibrated; the step checks them).
    Calibrate,
    /// Gate 1: report depth by posterior mass, cut to the token grant.
    Shortlist,
    /// The self-started integration read when the margin lies in the noise band.
    SelfRead,
    /// The deep gate (T.gate2).
    DeepGate,
    /// The deep check: a calibrated yes or no by the reader on the top one or two pieces.
    DeepCheck,
    /// The two label rows the deep check reads (held with the deep check).
    LabelRows,
    /// The local decisions.
    Decide,
    /// The route of spans by reference to g0 and of wants to G1.
    Route,
    /// The report.
    Report,
    /// The recurrent path: the state carried to the next round.
    Carry,
}
impl Component {
    /// Every component.
    pub const ALL: [Self; 13] = [
        Self::Sense,
        Self::Memory,
        Self::Periphery,
        Self::Calibrate,
        Self::Shortlist,
        Self::SelfRead,
        Self::DeepGate,
        Self::DeepCheck,
        Self::LabelRows,
        Self::Decide,
        Self::Route,
        Self::Report,
        Self::Carry,
    ];
    /// The steps an assembled core must hold.
    pub const REQUIRED: [Self; 6] = [Self::Periphery, Self::Shortlist, Self::Decide, Self::Route, Self::Report, Self::Carry];
    /// The name in the specification.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Sense => "sense",
            Self::Memory => "memory",
            Self::Periphery => "periphery",
            Self::Calibrate => "calibrate",
            Self::Shortlist => "shortlist",
            Self::SelfRead => "self_read",
            Self::DeepGate => "deep_gate",
            Self::DeepCheck => "deep_check",
            Self::LabelRows => "label_rows",
            Self::Decide => "decide",
            Self::Route => "route",
            Self::Report => "report",
            Self::Carry => "carry",
        }
    }
    /// The component of a specification name.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|c| c.name() == s)
    }
}

/// One piece of a tentacle's window for a round: the substrate chunk, its tokens, its calibrated
/// log-odds from the periphery read (or the fused value), the per-head posed and content-free
/// values behind it (empty when the read gives none), and what its drones read of it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Piece {
    pub chunk: usize,
    pub tokens: usize,
    pub log_odds: f32,
    pub posed_heads: Vec<f32>,
    pub null_heads: Vec<f32>,
    /// The drones' epistemic class (`drone_dendrite::Epistemic`), anomaly (0 to 1).
    pub class: crate::drone_dendrite::Epistemic,
    pub anomaly: f32,
    /// The piece a definition pulls into the next look (its use, caller or test on the specific
    /// reference graph), when the flow knows one.
    pub companion: Option<usize>,
}

/// What a round gives the core: its window's pieces, the card tokens granted (None: no cut), the
/// noise band of the margin (None: derived from the window), and the pose read under.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RoundInput {
    pub pieces: Vec<Piece>,
    pub grant: Option<usize>,
    pub band: Option<f32>,
    pub pose: String,
}

/// The device reads a round may make, supplied by the flow (lane T1). Each returns per piece of
/// `chunks` a score (None when the read is not available, which the core records and goes on).
pub trait CoreIo {
    /// The integration read of `chunks` joined under `pose` (calibrated log-odds per piece).
    fn integrate(&mut self, chunks: &[usize], pose: &str) -> Option<Vec<f32>>;
    /// The deep check of `chunks` (calibrated yes-minus-no log-odds per piece).
    fn deep_check(&mut self, chunks: &[usize]) -> Option<Vec<f32>>;
}

/// No device reads (a replay or the processor without a reader): every read is unavailable.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoReads;
impl CoreIo for NoReads {
    fn integrate(&mut self, _: &[usize], _: &str) -> Option<Vec<f32>> {
        None
    }
    fn deep_check(&mut self, _: &[usize]) -> Option<Vec<f32>> {
        None
    }
}

/// The flow's integration read: per chunk of the list, its calibrated log-odds under the pose.
pub type IntegrateFn<'a> = dyn FnMut(&[usize], &str) -> Option<Vec<f32>> + 'a;

/// The deep check on the card (or the processor) by the shared reader: `pieces` by chunk, their
/// texts in the joint read's piece form; `integrate` the flow's integration read when it has one.
pub struct ReaderIo<'a> {
    pub model: &'a Model,
    pub dev: Device,
    pub question: &'a str,
    pub pieces: BTreeMap<usize, crate::joint_read::Piece>,
    pub heads: Vec<(usize, usize)>,
    pub last: usize,
    pub integrate: Option<Box<IntegrateFn<'a>>>,
    /// Faults met, one line each.
    pub faults: Vec<String>,
}
impl CoreIo for ReaderIo<'_> {
    fn integrate(&mut self, chunks: &[usize], pose: &str) -> Option<Vec<f32>> {
        self.integrate.as_mut().and_then(|f| f(chunks, pose))
    }
    fn deep_check(&mut self, chunks: &[usize]) -> Option<Vec<f32>> {
        let ps: Vec<&crate::joint_read::Piece> = chunks.iter().filter_map(|c| self.pieces.get(c)).collect();
        if ps.len() != chunks.len() {
            self.faults.push(format!("deep check: {} of {} pieces have no text", chunks.len() - ps.len(), chunks.len()));
            return None;
        }
        let gate = (self.dev == Device::Gpu).then(|| nmengn::placement::gpu_gate().enter("tentacle deep check", true));
        let r = deep_check(self.model, self.dev, self.question, &ps, &self.heads, self.last);
        drop(gate);
        match r {
            Ok(v) => Some(v),
            Err(e) => {
                self.faults.push(format!("deep check: {}", e.message));
                None
            }
        }
    }
}

/// The deep check (master plan 3.3 step 7; Phase 0 audit P1-5): the reader reads `pieces` (one or
/// two) whole after the question and answers yes or no for each (`joint_read::pass_with`'s label
/// tails, the yes and no rows of the output head); the same is read under each of the three
/// content-free forms `NULL_FORMS`, and a piece's check is its yes-minus-no log-odds less the mean
/// over the content-free forms (contextual calibration). Positive: the reader says the piece
/// answers.
/// # Errors
/// Tokenizer or engine faults.
pub fn deep_check(
    model: &Model,
    dev: Device,
    question: &str,
    pieces: &[&crate::joint_read::Piece],
    heads: &[(usize, usize)],
    last: usize,
) -> Result<Vec<f32>, Fault> {
    if pieces.is_empty() {
        return Ok(Vec::new());
    }
    let order: Vec<usize> = (0..pieces.len()).collect();
    let posed = crate::joint_read::pass_with(model, dev, question, pieces, &order, heads, last, true, false)?.yes_no;
    // The content-free reads depend only on the pieces: held in RAM for the process (lane T1,
    // 2026-11-19: with every deep check run, the three content-free passes were three quarters of
    // its time), keyed by the pieces' places and texts' lengths.
    let key: String = pieces.iter().map(|p| format!("{}:{}-{}#{}|", p.path, p.start, p.end, p.text.len())).collect();
    let held = NULL_HELD.lock().ok().and_then(|m| m.get(&key).cloned());
    let null = if let Some(v) = held.filter(|v| v.len() == pieces.len()) {
        v
    } else {
        let mut null = vec![0.0_f32; pieces.len()];
        for form in NULL_FORMS {
            let v = crate::joint_read::pass_with(model, dev, form, pieces, &order, heads, last, true, false)?.yes_no;
            for (n, x) in null.iter_mut().zip(v) {
                *n += x;
            }
        }
        if let Ok(mut m) = NULL_HELD.lock() {
            m.insert(key, null.clone());
        }
        null
    };
    #[expect(clippy::cast_precision_loss, reason = "three forms")]
    let k = NULL_FORMS.len() as f32;
    Ok(posed.iter().zip(&null).map(|(p, n)| p - n / k).collect())
}

/// The deep check's content-free sums, by the pieces read (RAM, for the process).
static NULL_HELD: std::sync::Mutex<BTreeMap<String, Vec<f32>>> = std::sync::Mutex::new(BTreeMap::new());
/// Stage 5 WP-C3 R2: the shared form's content-free label sums per piece set (the span reads no
/// question, so they do not change with it), keyed by the pieces' content and the model's parameters
/// key, bounded (least recently used), held in RAM for the process.
static NULL_HELD_SHARED: std::sync::Mutex<Option<crate::cache_reads::Lru<Vec<f32>>>> = std::sync::Mutex::new(None);

/// Softmax of `log_odds` (non-finite values get no mass).
#[must_use]
pub fn posterior(log_odds: &[f32]) -> Vec<f32> {
    let top = log_odds.iter().copied().filter(|x| x.is_finite()).fold(f32::NEG_INFINITY, f32::max);
    if !top.is_finite() {
        return vec![0.0; log_odds.len()];
    }
    let e: Vec<f32> = log_odds.iter().map(|x| if x.is_finite() { (x - top).exp() } else { 0.0 }).collect();
    let z: f32 = e.iter().sum();
    e.iter().map(|x| if z > 0.0 { x / z } else { 0.0 }).collect()
}

/// Gate 1 (report depth by posterior mass): over `order` (pieces best first) the smallest prefix
/// whose posterior holds `REPORT_MASS`, then cut so its tokens fit `grant` (at least one piece).
/// Returns the depth and the mass it holds. With no finite score: `SHORTLIST_FALLBACK`.
#[must_use]
pub fn posterior_depth(pieces: &[Piece], order: &[usize], grant: Option<usize>) -> (usize, f32) {
    let lo: Vec<f32> = pieces.iter().map(|p| p.log_odds).collect();
    let post = posterior(&lo);
    if post.iter().all(|p| *p <= 0.0) {
        return (SHORTLIST_FALLBACK.min(order.len()).max(usize::from(!order.is_empty())), 0.0);
    }
    let (mut depth, mut mass, mut tokens) = (0_usize, 0.0_f32, 0_usize);
    for &i in order {
        if mass >= REPORT_MASS {
            break;
        }
        let t = pieces.get(i).map_or(0, |p| p.tokens);
        if depth > 0 && grant.is_some_and(|g| tokens + t > g) {
            break;
        }
        depth += 1;
        tokens += t;
        mass += post.get(i).copied().unwrap_or(0.0);
    }
    (depth.max(1).min(order.len().max(1)), mass)
}

/// Median absolute deviation of `xs` (finite values), the mean absolute deviation where that is
/// 0, and 1 where they do not spread (the g0 report's spread rule).
#[must_use]
pub fn spread_of(xs: &[f32]) -> f32 {
    let mut v: Vec<f32> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    if v.len() < 2 {
        return 1.0;
    }
    v.sort_by(f32::total_cmp);
    let med = v.get(v.len() / 2).copied().unwrap_or(0.0);
    let mut dev: Vec<f32> = v.iter().map(|x| (x - med).abs()).collect();
    dev.sort_by(f32::total_cmp);
    let mad = dev.get(dev.len() / 2).copied().unwrap_or(0.0);
    if mad > 0.0 {
        return mad;
    }
    #[expect(clippy::cast_precision_loss, reason = "a count")]
    let mean = dev.iter().sum::<f32>() / dev.len() as f32;
    if mean > 0.0 { mean } else { 1.0 }
}

/// The core's local decisions for one round (master plan 3.3 step 7).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Decisions {
    /// Pieces (indices into the round's pieces) best first, and the report depth with the
    /// posterior mass it holds.
    pub order: Vec<usize>,
    pub depth: usize,
    pub mass: f32,
    /// The top piece's posed-to-null ratio over its heads (geometric mean; from its log-odds when
    /// the read gives no per-head values), and the no-information verdict.
    pub ratio: f32,
    pub no_information: bool,
    /// Spread (MAD of the shortlist), margin of the first over the second in spreads, and the noise
    /// band the margin is judged against (in spreads).
    pub spread: f32,
    pub margin: f32,
    pub band: f32,
    /// Self-started integration (the margin in the band), and the deep gate's verdict: pieces to
    /// check.
    pub self_read: bool,
    pub deep: Vec<usize>,
    /// Next look: chunks a reported definition pulls (its companion, not reported), and the
    /// reported pieces flagged anomalous.
    pub next: Vec<usize>,
    pub anomalies: Vec<usize>,
    /// Why, one line per decision.
    pub why: Vec<String>,
}

/// The local decisions over `input` (pieces with their current log-odds), with the deep check's
/// verdicts (`deep`: per piece index, calibrated log-odds) when it ran. Rules: gate 1 by posterior
/// mass; no information when the top piece's posed-to-null ratio is at most 1 over its heads and
/// the deep check, when run, is at most 0 on it; self-started read when the margin (in spreads)
/// lies within the band (given, else 1 spread); deep gate as `T.gate2` (margin below one spread or
/// the top's log-odds at most 0: the top two when the margin is in the band, else the top one, at
/// most `DEEP_MAX`); next look: a reported definition pulls its companion, a reported piece with
/// anomaly at least `ANOMALY_FLAG` is flagged.
#[must_use]
pub fn decide(input: &RoundInput, deep: &BTreeMap<usize, f32>) -> Decisions {
    let ps = &input.pieces;
    let mut order: Vec<usize> = (0..ps.len()).collect();
    order.sort_by(|&a, &b| {
        let (x, y) = (ps.get(a).map_or(f32::NEG_INFINITY, |p| p.log_odds), ps.get(b).map_or(f32::NEG_INFINITY, |p| p.log_odds));
        y.total_cmp(&x).then(a.cmp(&b))
    });
    let (depth, mass) = posterior_depth(ps, &order, input.grant);
    let lo = |i: usize| ps.get(i).map_or(f32::NEG_INFINITY, |p| p.log_odds);
    let short: Vec<f32> = order.iter().take(depth.max(2)).map(|&i| lo(i)).collect();
    let spread = spread_of(&short);
    let (t1, t2) = (order.first().map_or(f32::NEG_INFINITY, |&i| lo(i)), order.get(1).map_or(f32::NEG_INFINITY, |&i| lo(i)));
    let margin = if t1.is_finite() && t2.is_finite() { (t1 - t2) / spread } else if t1.is_finite() { f32::INFINITY } else { 0.0 };
    let band = input.band.unwrap_or(1.0);
    let mut why = vec![format!("depth {depth} by posterior mass {mass:.2} (rule {REPORT_MASS}, grant {:?})", input.grant)];
    // No information: the top piece's posed-to-null ratio over its heads.
    let top = order.first().and_then(|&i| ps.get(i));
    let log_ratio = top.map_or(f32::NEG_INFINITY, |p| {
        let d: Vec<f32> = p.posed_heads.iter().zip(&p.null_heads).map(|(a, b)| a - b).filter(|x| x.is_finite()).collect();
        #[expect(clippy::cast_precision_loss, reason = "a head count")]
        if d.is_empty() { p.log_odds } else { d.iter().sum::<f32>() / d.len() as f32 }
    });
    let ratio = if log_ratio.is_finite() { log_ratio.exp() } else { 0.0 };
    let deep_top = order.first().and_then(|i| deep.get(i)).copied();
    let no_information = ratio <= 1.0 && deep_top.is_none_or(|d| d <= 0.0);
    why.push(format!(
        "no information {no_information}: posed-to-null ratio {ratio:.3} over the heads, deep check {}",
        deep_top.map_or_else(|| "not run".to_owned(), |d| format!("{d:.2}"))
    ));
    let in_band = margin.is_finite() && margin <= band;
    let self_read = in_band && depth > 1;
    why.push(format!("margin {margin:.2} spreads (spread {spread:.3}, band {band:.2}): self-started read {self_read}"));
    let want_deep = (margin.is_finite() && margin <= 1.0) || t1 <= 0.0;
    let deep_n = if want_deep { if in_band { DEEP_MAX } else { 1 } } else { 0 };
    let deep_pick: Vec<usize> = order.iter().copied().take(deep_n.min(order.len())).collect();
    why.push(format!("deep gate: {} piece(s)", deep_pick.len()));
    let reported: BTreeSet<usize> = order.iter().take(depth).filter_map(|&i| ps.get(i).map(|p| p.chunk)).collect();
    let mut next = Vec::new();
    let mut anomalies = Vec::new();
    for &i in order.iter().take(depth) {
        let Some(p) = ps.get(i) else { continue };
        if p.class == crate::drone_dendrite::Epistemic::Definition
            && let Some(c) = p.companion
            && !reported.contains(&c)
            && !next.contains(&c)
        {
            next.push(c);
        }
        if p.anomaly >= ANOMALY_FLAG {
            anomalies.push(p.chunk);
        }
    }
    why.push(format!("next look: {} companion(s) of reported definitions; {} anomaly flag(s)", next.len(), anomalies.len()));
    Decisions { order, depth, mass, ratio, no_information, spread, margin, band, self_read, deep: deep_pick, next, anomalies, why }
}

/// The core's state across rounds (the recurrent path through its g0): pieces it has read with
/// the integration (never read again), the last report's best chunks, margin and no-information,
/// rounds and device reads so far.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Carried {
    pub rounds: u32,
    pub reads: u32,
    pub integrated: BTreeSet<usize>,
    pub checked: BTreeMap<usize, f32>,
    pub last_best: Vec<usize>,
    pub last_margin: f32,
    pub last_no_information: bool,
}

/// One round's report (master plan 3.3 step 8), by chunk: best pieces with log-odds and posterior
/// share, their epistemic tags, spread, margin, no-information, next look, anomalies, the deep
/// check's verdicts, the decisions' reasons, the round and reads.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CoreReport {
    pub best: Vec<(usize, f32, f32)>,
    pub tags: Vec<String>,
    pub spread: f32,
    pub margin: f32,
    pub no_information: bool,
    pub next: Vec<usize>,
    pub anomalies: Vec<usize>,
    pub deep: Vec<(usize, f32)>,
    pub self_read: bool,
    pub why: Vec<String>,
    pub round: u32,
    pub reads: u32,
    /// The components run, in order.
    pub steps: Vec<&'static str>,
}

/// The g0 report's epistemic tag of a class.
#[must_use]
pub fn tag_of(c: crate::drone_dendrite::Epistemic) -> &'static str {
    use crate::drone_dendrite::Epistemic as E;
    match c {
        E::Definition => "defines",
        E::Use | E::Unknown | E::Generated => "uses",
        E::Test => "tests",
        E::Documentation | E::Configuration => "documents",
    }
}

impl CoreReport {
    /// The report in g0's form for tentacle `id` (`tentacle` its number): spans by reference
    /// (`span_of` a chunk's span), confidence the posterior share, the next look's companions as
    /// wants to G1, anomalies named in the read text.
    #[must_use]
    pub fn to_g0(&self, id: &str, tentacle: u8, focus: &Focus, span_of: &dyn Fn(usize) -> Option<SpanRef>) -> Report {
        let best: Vec<SpanReport> = self
            .best
            .iter()
            .enumerate()
            .filter_map(|(place, (c, s, p))| span_of(*c).map(|span| SpanReport { span, score: *s, confidence: *p, place }))
            .collect();
        let anomalies: Vec<String> = self.anomalies.iter().filter_map(|c| span_of(*c).map(|s| s.label())).collect();
        let mut read = format!("core round {} ({}): {}", self.round, self.steps.join(", "), self.why.join("; "));
        if !anomalies.is_empty() {
            read = format!("{read}; anomaly flagged: {}", anomalies.join(", "));
        }
        Report {
            from: id.to_owned(),
            tentacle,
            best_score: self.best.first().map_or(0.0, |b| b.1),
            confidence: if self.no_information { 0.0 } else { self.best.first().map_or(0.0, |b| b.2) },
            best,
            no_information: self.no_information,
            baseline: 0.0,
            next: Vec::new(),
            focus: focus.clone(),
            reads: self.reads,
            read,
            spread: self.spread,
            margin: self.margin,
            tags: self.tags.clone(),
            wants: self.next.iter().filter_map(|c| span_of(*c)).collect(),
            ..Report::default()
        }
    }
}

/// The core assembled from the specification's components (audit rule 3.3.3): the steps in the
/// order the specification lists them, the module bound to each, and the carried state.
#[derive(Clone, Debug, Default)]
pub struct Core {
    pub steps: Vec<(Component, String)>,
    pub carried: Carried,
}
impl Core {
    /// Assemble the core from `spec`: every module naming a component binds it, in the listed
    /// order; refused when a required step (`Component::REQUIRED`) or a required kind
    /// (`REQUIRED_KINDS`) is missing, or a component is bound twice.
    /// # Errors
    /// The first missing or doubled part.
    pub fn assemble(spec: &Spec) -> Result<Self, String> {
        let mut steps: Vec<(Component, String)> = Vec::new();
        for m in &spec.modules {
            if m.component.is_empty() {
                continue;
            }
            let c = Component::parse(&m.component).ok_or_else(|| format!("{}: unknown component {}", m.name, m.component))?;
            if steps.iter().any(|(x, _)| *x == c) {
                return Err(format!("{}: component {} bound twice", m.name, c.name()));
            }
            steps.push((c, m.name.clone()));
        }
        if let Some(c) = Component::REQUIRED.iter().find(|c| !steps.iter().any(|(x, _)| x == *c)) {
            return Err(format!("the core lacks the step {}", c.name()));
        }
        let bound: Vec<&Module> = spec.modules.iter().filter(|m| !m.component.is_empty()).collect();
        if let Some(k) = REQUIRED_KINDS.iter().find(|k| !bound.iter().any(|m| m.kind == **k || m.also == **k)) {
            return Err(format!("the core's components hold no {k} module"));
        }
        Ok(Self { steps, carried: Carried::default() })
    }
    /// The core of `designs/tentacle-core.json` under `home`.
    /// # Errors
    /// As `load_spec` and `assemble`.
    pub fn from_design(home: &Path) -> Result<Self, String> {
        Self::assemble(&load_spec(home)?)
    }
    /// One round: the components run in the assembled order over `input`; device reads through
    /// `io`. The self-started read replaces the read pieces' log-odds by the integration's (only
    /// pieces never integrated before are read); the deep check's verdicts enter the decisions;
    /// the carried state is updated last.
    pub fn round(&mut self, mut input: RoundInput, io: &mut dyn CoreIo) -> CoreReport {
        let mut deep: BTreeMap<usize, f32> = BTreeMap::new();
        // Verdicts carried from earlier rounds count again (the recurrent path).
        for (i, p) in input.pieces.iter().enumerate() {
            if let Some(v) = self.carried.checked.get(&p.chunk) {
                deep.insert(i, *v);
            }
        }
        let mut d = decide(&input, &deep);
        let mut rep = CoreReport { round: self.carried.rounds + 1, ..CoreReport::default() };
        for (c, _) in self.steps.clone() {
            rep.steps.push(c.name());
            match c {
                Component::Sense
                | Component::Memory
                | Component::LabelRows
                | Component::Periphery
                | Component::Route
                | Component::Report
                | Component::DeepGate => {}
                Component::Calibrate => {
                    let bad = input.pieces.iter().filter(|p| !p.log_odds.is_finite()).count();
                    if bad > 0 {
                        d.why.push(format!("calibration: {bad} piece(s) without a finite score"));
                    }
                }
                Component::Shortlist => d = decide(&input, &deep),
                Component::SelfRead => {
                    if d.self_read {
                        let pick: Vec<usize> = d
                            .order
                            .iter()
                            .take(d.depth)
                            .copied()
                            .filter(|i| input.pieces.get(*i).is_some_and(|p| !self.carried.integrated.contains(&p.chunk)))
                            .collect();
                        let chunks: Vec<usize> = pick.iter().filter_map(|i| input.pieces.get(*i).map(|p| p.chunk)).collect();
                        if chunks.is_empty() {
                            d.why.push("self-started read: every shortlisted piece already integrated".into());
                        } else if let Some(v) = io.integrate(&chunks, &input.pose).filter(|v| v.len() == chunks.len()) {
                            self.carried.reads += 1;
                            for (i, x) in pick.iter().zip(v) {
                                if let Some(p) = input.pieces.get_mut(*i) {
                                    p.log_odds = x;
                                    self.carried.integrated.insert(p.chunk);
                                }
                            }
                            rep.self_read = true;
                            d = decide(&input, &deep);
                        } else {
                            d.why.push("self-started read: no integration read available".into());
                        }
                    }
                }
                Component::DeepCheck => {
                    let pick: Vec<usize> = d.deep.iter().copied().filter(|i| !deep.contains_key(i)).collect();
                    let chunks: Vec<usize> = pick.iter().filter_map(|i| input.pieces.get(*i).map(|p| p.chunk)).collect();
                    if !chunks.is_empty() {
                        if let Some(v) = io.deep_check(&chunks).filter(|v| v.len() == chunks.len()) {
                            self.carried.reads += 1;
                            for ((i, c), x) in pick.iter().zip(&chunks).zip(v) {
                                deep.insert(*i, x);
                                self.carried.checked.insert(*c, x);
                            }
                        } else {
                            d.why.push("deep check: not available".into());
                        }
                    }
                }
                Component::Decide => d = decide(&input, &deep),
                Component::Carry => {
                    self.carried.rounds += 1;
                    self.carried.last_best = d.order.iter().take(d.depth).filter_map(|i| input.pieces.get(*i).map(|p| p.chunk)).collect();
                    self.carried.last_margin = d.margin;
                    self.carried.last_no_information = d.no_information;
                }
            }
        }
        let lo: Vec<f32> = input.pieces.iter().map(|p| p.log_odds).collect();
        let post = posterior(&lo);
        rep.best = d
            .order
            .iter()
            .take(d.depth)
            .filter_map(|&i| input.pieces.get(i).map(|p| (p.chunk, p.log_odds, post.get(i).copied().unwrap_or(0.0))))
            .collect();
        rep.tags = d.order.iter().take(d.depth).filter_map(|&i| input.pieces.get(i).map(|p| tag_of(p.class).to_owned())).collect();
        rep.spread = d.spread;
        rep.margin = d.margin;
        rep.no_information = d.no_information;
        rep.next = d.next.clone();
        rep.anomalies = d.anomalies.clone();
        rep.deep = deep.iter().filter_map(|(i, v)| input.pieces.get(*i).map(|p| (p.chunk, *v))).collect();
        rep.why = d.why;
        rep.reads = self.carried.reads;
        rep
    }
}

/// The assembled core's part of the section check (2026-11-19): the design assembles with every
/// required step in order; a design without the recurrent path is refused; gate 1 by posterior
/// mass and its token cut; the decisions (no information both ways, the self-started read in the
/// band, the deep gate, a definition pulling its companion, an anomaly flagged); a round through
/// stub reads integrates only pieces never read, takes the deep check, and carries its state.
fn assembled_check(rep: &mut String, home: &Path) -> Result<(), String> {
    use crate::drone_dendrite::Epistemic;
    use std::fmt::Write as _;
    let spec = load_spec(home)?;
    let core = Core::assemble(&spec)?;
    let names: Vec<&str> = core.steps.iter().map(|(c, _)| c.name()).collect();
    let _ = writeln!(rep, "(z) assembled core: {}", names.join(" > "));
    let pos = |c: Component| core.steps.iter().position(|(x, _)| *x == c);
    if !(pos(Component::Shortlist) < pos(Component::Decide) && pos(Component::Decide) < pos(Component::Carry)) {
        return Err("the assembled core's steps are out of order".into());
    }
    let mut no_recur = spec.clone();
    no_recur.modules.retain(|m| m.kind != "recurrent");
    if Core::assemble(&no_recur).is_ok() {
        return Err("a core without the recurrent path assembled".into());
    }
    let piece = |chunk: usize, lo: f32| Piece { chunk, tokens: 100, log_odds: lo, ..Piece::default() };
    let ps = vec![piece(10, 5.0), piece(11, 1.0), piece(12, 0.5), piece(13, -2.0)];
    let order = vec![0, 1, 2, 3];
    let (d1, m1) = posterior_depth(&ps, &order, None);
    let (d2, _) = posterior_depth(&ps, &order, Some(150));
    let _ = writeln!(rep, "(z) posterior depth {d1} (mass {m1:.3}); with a 150-token grant {d2}");
    if d1 != 1 || d2 != 1 || m1 < REPORT_MASS {
        return Err(format!("posterior depth {d1} {m1} {d2}"));
    }
    let close = vec![piece(10, 2.0), piece(11, 1.95), piece(12, 1.9), piece(13, -3.0)];
    let mut def = close.clone();
    if let Some(p) = def.first_mut() {
        p.class = Epistemic::Definition;
        p.companion = Some(99);
        p.anomaly = 0.9;
    }
    let inp = RoundInput { pieces: def, grant: None, band: None, pose: String::new() };
    let d = decide(&inp, &BTreeMap::new());
    if d.no_information || !d.self_read || d.deep.len() != DEEP_MAX || d.next != vec![99] || d.anomalies != vec![10] {
        return Err(format!("decisions on close pieces: {d:?}"));
    }
    let neg = RoundInput { pieces: vec![piece(1, -0.5), piece(2, -1.0)], ..RoundInput::default() };
    let dn = decide(&neg, &BTreeMap::new());
    let dy = decide(&neg, &BTreeMap::from([(0, 2.0)]));
    if !dn.no_information || dy.no_information {
        return Err("no information: a negative top without a check is information, or a positive check did not lift it".into());
    }
    struct Stub(u32);
    impl CoreIo for Stub {
        fn integrate(&mut self, chunks: &[usize], _: &str) -> Option<Vec<f32>> {
            self.0 += 1;
            #[expect(clippy::cast_precision_loss, reason = "small test values")]
            Some(chunks.iter().map(|c| *c as f32 / 10.0).collect())
        }
        fn deep_check(&mut self, chunks: &[usize]) -> Option<Vec<f32>> {
            Some(chunks.iter().map(|_| 1.5).collect())
        }
    }
    let mut c = core.clone();
    let mut io = Stub(0);
    let r1 = c.round(RoundInput { pieces: close.clone(), ..RoundInput::default() }, &mut io);
    let r2 = c.round(RoundInput { pieces: close, ..RoundInput::default() }, &mut io);
    let _ = writeln!(rep, "(z) round 1: best {:?}, read {}, deep {:?}; round 2 integrations {}", r1.best.first(), r1.self_read, r1.deep, io.0);
    if !r1.self_read || r1.deep.is_empty() || c.carried.rounds != 2 || r2.round != 2 || io.0 != 1 || r1.no_information {
        return Err(format!("rounds: read {} deep {:?} rounds {} integrations {}", r1.self_read, r1.deep, c.carried.rounds, io.0));
    }
    let g = r1.to_g0("g0_t1", 1, &Focus::default(), &|ch| Some(SpanRef { file: "a.rs".into(), start: ch, end: ch, key: format!("k{ch}") }));
    if g.best.len() != r1.best.len() || g.spread != r1.spread || g.tags.len() != g.best.len() {
        return Err("the g0 report does not carry the round's report".into());
    }
    Ok(())
}

/// Section check of the tentacle core (2026-11-10): (a) the specification file validates, every
/// tentacle's clusters resolve in the population, and malformed specifications are refused by
/// name; (b) gate 1's shortlist rule on hand values; (c) on the released base (GPU when open) over
/// a three-file tree: with the core on, eight cores' reports reach their `g0_tx` and G1, every
/// report field is filled, a repeat gives the same hand-over bit for bit, the isolation gate reads
/// 0, every instruction and report is behavioural and every forwarded packet sensory, the cores
/// share one model (one weight copy), and the answerable and decoy questions' outcomes are
/// reported. Values: `tests/section-checks/nmapi-tentacle-core/report.txt`.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    use std::fmt::Write as _;
    let mut rep = String::from("nmapi.tentacle_core (tentacle core A8, A9, A10 first form)\n");
    let r = core_check(&mut rep);
    let _ = writeln!(rep, "result: {}", r.as_ref().map_or_else(|e| format!("FAIL {e}"), |()| "ok".to_owned()));
    let dir = nmtype::check_dir("nmapi-tentacle-core").map_err(|e| e.to_string())?;
    std::fs::write(dir.join("report.txt"), rep).map_err(|e| e.to_string())?;
    r
}

fn core_check(rep: &mut String) -> Result<(), String> {
    use std::fmt::Write as _;
    let home = nmtype::nano_root();
    // (a) The specification.
    let v = nmjson::parse(&std::fs::read(home.join(SPEC_FILE)).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let (mods, tents, groups) = check_spec_file(&home, &v)?;
    assembled_check(rep, &home)?;
    let _ = writeln!(
        rep,
        "(a) {SPEC_FILE}: {mods} modules, {tents} tentacles, {groups} distinct cluster groups held, all resolve in the population"
    );
    let base = r#"{"version":1,"population":"qwen3-0.6b","applies_to":"all","modules":[M]}"#;
    let ok_mods = r#"{"name":"a","kind":"attention","also":"feed-forward","layers":"sense","groups":["attn"],"shared":"reference","state":"built"},{"name":"m","kind":"memory","layers":"none","groups":[],"shared":"reference","state":"built"},{"name":"r","kind":"routing","layers":"none","groups":[],"shared":"per-tentacle","state":"built"},{"name":"p","kind":"parameter-cluster","layers":"none","groups":["final"],"shared":"reference","state":"built"},{"name":"g","kind":"gating","layers":"none","groups":[],"shared":"per-tentacle","state":"built","rule":"x","derived_from":"y","fallback":{"value":2,"why":"z"}},{"name":"c","kind":"recurrent","layers":"none","groups":[],"shared":"per-tentacle","state":"first form"}"#;
    let good = base.replace('M', ok_mods);
    validate(&home, &nmjson::parse(good.as_bytes()).map_err(|e| e.to_string())?)?;
    let bads = [
        ("an unknown kind", good.replace("\"kind\":\"routing\"", "\"kind\":\"wiring\"")),
        ("a gating module without its fallback reason", good.replace(",\"why\":\"z\"", "")),
        ("a module named twice", good.replace("\"name\":\"r\"", "\"name\":\"m\"")),
        ("an unknown cluster group", good.replace("[\"attn\"]", "[\"attn\",\"block\"]")),
        ("a missing required kind", good.replace("\"kind\":\"routing\"", "\"kind\":\"reasoning\"")),
        ("another version", good.replace("\"version\":1", "\"version\":3")),
    ];
    for (what, text) in bads {
        let x = nmjson::parse(text.as_bytes()).map_err(|e| e.to_string())?;
        match validate(&home, &x) {
            Ok(_) => return Err(format!("the validator accepted {what}")),
            Err(e) => {
                let _ = writeln!(rep, "(a) refused {what}: {e}");
            }
        }
    }
    // (b) Gate 1.
    let (k1, w1) = shortlist(&[4.0, 0.0, -1.0, 0.0]);
    let (k2, _) = shortlist(&[1.0, 1.0, 1.0, 1.0]);
    let (k3, w3) = shortlist(&[-1.0, -2.0]);
    let (k4, _) = shortlist(&[3.0, 1.0, 0.5, -1.0, 0.0, 0.0, 0.0, 0.0]);
    let _ =
        writeln!(rep, "(b) shortlist: one positive -> {k1} ({w1}); four equal -> {k2}; none positive -> {k3} ({w3}); 3, 1, 0.5 -> {k4}");
    if k1 != 1 || k2 != 4 || k3 != 2 || !(2..=3).contains(&k4) {
        return Err(format!("shortlist rule: {k1}, {k2}, {k3}, {k4}"));
    }
    // (c) The core on a small tree.
    let embed: nmsubs::Embed = Arc::new(|_: &str| vec![0.0; 4]);
    let files = [
        (
            "a/ring.rs",
            "pub struct Ring { items: Vec<u32>, head: usize }\nimpl Ring {\n    pub fn push(&mut self, x: u32) { let n = self.items.len(); self.items[self.head % n] = x; self.head += 1; }\n}\n",
        ),
        (
            "a/clock.rs",
            "pub fn tick(now: u64, step: u64) -> u64 { now.wrapping_add(step) }\npub fn late(now: u64, due: u64) -> bool { now > due }\n",
        ),
        (
            "a/limit.rs",
            "pub struct Limit { bytes: usize }\nimpl Limit {\n    pub fn room(&self, used: usize) -> usize { self.bytes.saturating_sub(used) }\n}\n",
        ),
    ];
    let sub = nmsubs::from_files(&files, embed);
    let settings = Settings { core: true, ..Settings::served() };
    let pass = tentacle_pass::TentaclePass::new(&home, settings);
    let ask = |q: &str| -> Result<(Arc<Scan>, Pass, CoreRun), String> {
        let (scan, _) = pass.scan(&sub, q, &settings).map_err(|e| e.to_string())?;
        let (p, cr) = judge(&home, &sub, &scan, &settings).map_err(|e| e.to_string())?;
        Ok((scan, p, cr))
    };
    let q = "where is the free room of a limit computed?";
    let (scan, p1, c1) = ask(q)?;
    let (_, p2, c2) = ask(q)?;
    let label = |p: &Pass| p.packets.iter().map(|k| k.key().to_owned()).collect::<Vec<_>>();
    let same = label(&p1) == label(&p2)
        && format!("{:?}", c1.outcome.handover.to_value().encode()) == format!("{:?}", c2.outcome.handover.to_value().encode());
    let _ = writeln!(
        rep,
        "(c) {}: {} windows; hand-over {}; packets {:?}; repeat identical: {same}",
        scan.device,
        scan.windows.len(),
        c1.outcome.handover.to_value().encode().unwrap_or_default(),
        label(&p1)
    );
    let _ = writeln!(rep, "(c) note: {}", p1.note);
    // Report fields: a fresh set of cores, report each.
    let model = crate::neural::model(&home).map_err(|e| e.to_string())?;
    let sets = tentacle_pass::drones(&home);
    let stop = kvstore::score_targets(&model, sets.last_layer()).map_err(|e| e.to_string())?;
    let anchor = scan.windows.iter().find_map(|w| w.window.as_ref().and_then(|x| x.chunks.first().cloned())).ok_or("no anchor")?;
    let dev = if model.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let shared = Arc::new(CoreShared {
        model: Some(Arc::clone(&model)),
        dev,
        stop,
        anchor,
        question: q.to_owned(),
        verify: sets.verification_set().to_vec(),
        reads: Mutex::new((0, 0.0, Vec::new())),
        names: BTreeMap::new(),
        tape: None,
        fill: None,
        read: ReadOptions::default(),
    });
    let mut fields_ok = true;
    let mut one_copy = true;
    for w in &scan.windows {
        let mut core = TentacleCore::from_scan(&sub, &scan, w, sets.heads.clone(), Arc::clone(&shared)).map_err(|e| e.to_string())?;
        one_copy &= core.shared.model.as_ref().is_some_and(|m| Arc::ptr_eq(m, &model));
        let r = core.report();
        let filled = !r.from.is_empty()
            && !r.best.is_empty()
            && r.best.iter().all(|b| b.confidence.is_finite() && !b.span.key.is_empty())
            && r.reads == 1;
        fields_ok &= filled;
        let a = core.answer(
            &r.best.first().map_or_else(|| SpanRef { file: String::new(), start: 0, end: 0, key: String::new() }, |b| b.span.clone()),
            q,
        );
        let rr = core.reread(&Focus { files: vec!["a/limit.rs".into()], ..Focus::default() });
        let _ = writeln!(
            rep,
            "(c) {}: shortlist {}; report: best {:?} confidence {:.3} no information {}; answer about its first span: score {:.3} confidence {:.3} rank {:?}; after a focused re-read: best {:?}, reads {}",
            r.from,
            core.depth_why,
            r.best.iter().map(|b| (b.span.label(), b.score)).collect::<Vec<_>>(),
            r.confidence,
            r.no_information,
            a.score,
            a.confidence,
            a.rank,
            rr.best.iter().map(|b| (b.span.label(), b.score)).collect::<Vec<_>>(),
            rr.reads
        );
    }
    let gate = c1.outcome.handover.gate_direct;
    let _ = writeln!(
        rep,
        "(c) report fields filled: {fields_ok}; cores share one model by reference (one weight copy, pointer identity; no tentacle owns a copy): {one_copy}; isolation gate {gate}; envelopes behavioural {} and packets sensory {}, refused {}",
        c1.envelopes, c1.packets, c1.refused
    );
    // Decoy: content absent from the tree.
    let (_, _, cd) = ask("where is the video codec's motion vector table decoded?")?;
    let _ = writeln!(
        rep,
        "(c) answerable question: first reports with no information {} of {}; decoy question: {} of {}, forwarded {} spans, overall confidence {:.3} (answerable {:.3}); a measurement, the decoy rule is set at refinement",
        c1.no_info,
        c1.ranked.len(),
        cd.no_info,
        cd.ranked.len(),
        cd.outcome.handover.items.len(),
        cd.outcome.handover.confidence,
        c1.outcome.handover.confidence
    );
    // (d) Refinement step 2a: the read components on (second look, token votes, length-matched
    // content-free pose). On one scan made with the matched pose, a first-form core and a core
    // with every component on: the report and every evidence record name what was on, the chunks
    // of the second look carry its unscaled score beside the full read's (which equals the
    // first-form core's, bit for bit), they lead the order, and a repeat gives the same hand-over.
    let on = Settings { core: true, read: ReadOptions::all(), ..Settings::served() };
    let (scan_on, _) = pass.scan(&sub, q, &on).map_err(|e| e.to_string())?;
    let (pa, ca) = judge(&home, &sub, &scan_on, &on).map_err(|e| e.to_string())?;
    let (pb, cb) = judge(&home, &sub, &scan_on, &on).map_err(|e| e.to_string())?;
    let same_on = label(&pa) == label(&pb)
        && ca.outcome.handover.to_value().encode().unwrap_or_default() == cb.outcome.handover.to_value().encode().unwrap_or_default();
    let anchor_on = scan_on.windows.iter().find_map(|w| w.window.as_ref().and_then(|x| x.chunks.first().cloned())).ok_or("no anchor")?;
    let share = |read: ReadOptions| {
        Arc::new(CoreShared {
            model: Some(Arc::clone(&model)),
            dev,
            stop: shared.stop.clone(),
            anchor: Arc::clone(&anchor_on),
            question: q.to_owned(),
            verify: sets.verification_set().to_vec(),
            reads: Mutex::new((0, 0.0, Vec::new())),
            names: BTreeMap::new(),
            tape: None,
            fill: None,
            read,
        })
    };
    let (first_form, all_on) = (share(ReadOptions::default()), share(ReadOptions::all()));
    let mut look_ok = true;
    let mut looked = 0;
    for w in &scan_on.windows {
        let r0 =
            TentacleCore::from_scan(&sub, &scan_on, w, sets.heads.clone(), Arc::clone(&first_form)).map_err(|e| e.to_string())?.report();
        let mut core = TentacleCore::from_scan(&sub, &scan_on, w, sets.heads.clone(), Arc::clone(&all_on)).map_err(|e| e.to_string())?;
        let r1 = core.report();
        let Some(l) = core.look.clone() else {
            look_ok = false;
            continue;
        };
        looked += l.pick.len();
        let named = !r1.read.is_empty()
            && r0.read.is_empty()
            && r1.facts.values().all(|f| !f.read.is_empty())
            && r0.facts.values().all(|f| f.read.is_empty());
        let beside = r1.facts.iter().all(|(k, f)| {
            r0.facts.get(k).is_some_and(|f0| {
                f0.score("calibrated").map(f32::to_bits) == f.score("calibrated").map(f32::to_bits)
                    && f0.score("mass").map(f32::to_bits) == f.score("mass").map(f32::to_bits)
            })
        });
        let unscaled = l.pick.iter().zip(&l.scores).all(|(i, s)| {
            core.g0.held.get(*i).and_then(|h| r1.facts.get(&h.span.key)).and_then(|f| f.score("second_look")).map(f32::to_bits)
                == Some(s.to_bits())
        });
        let lead = core.order().iter().take(l.pick.len()).all(|i| l.pick.contains(i));
        look_ok &=
            named && beside && unscaled && lead && r1.facts.values().filter(|f| f.score("second_look").is_some()).count() == l.pick.len();
        let _ = writeln!(
            rep,
            "(d) {}: read: {}; second look over {} of {} chunks, scores {:?} (full read {:?}); named {named}, full read beside {beside}, unscaled {unscaled}, lead the order {lead}",
            r1.from,
            r1.read,
            l.pick.len(),
            core.scores.len(),
            l.scores,
            l.pick.iter().filter_map(|i| core.full.get(*i)).collect::<Vec<_>>()
        );
    }
    let faults_on = all_on.reads.lock().map(|r| r.2.clone()).unwrap_or_default();
    let _ = writeln!(
        rep,
        "(d) components on: content-free pose: {}; hand-over {}; repeat identical: {same_on}; chunks given a second look {looked}; faults {faults_on:?}; note: {}",
        scan_on.pose,
        ca.outcome.handover.to_value().encode().unwrap_or_default(),
        pa.note
    );
    nmengn::residency::get().release_windows();
    if !same_on || !look_ok || looked == 0 || !faults_on.is_empty() || scan_on.pose.is_empty() {
        return Err("tentacle core with the read components on: repeat differs, a report or record does not name them, the full read's scores are not kept beside the second look's, or no second look was taken".into());
    }
    if !same || !fields_ok || !one_copy || gate != 0 || c1.refused != 0 || p1.packets.is_empty() {
        return Err("tentacle core: repeat differs, report fields missing, more than one weight copy, gate above 0, a wrong-class hand-over, or nothing forwarded".into());
    }
    Ok(())
}

/// The deep checks of every tentacle due in a round as one packed card run (lane S's
/// `kvstore::read_group_labels`; lane T1 2026-11-19). Each request is one tentacle's top one or two
/// pieces, each cut to its first `DEEP_SPAN_CHARS` characters (the design reads the top spans, not
/// whole pieces). Per request and form (the question, then the content-free forms not yet held in
/// RAM) one group: the span (system prompt, the form, the pieces) and one tail a piece. Returns per
/// request the yes-minus-no log-odds less the content-free mean, per piece; empty on a fault.
#[must_use]
pub fn deep_check_packed(model: &Model, question: &str, requests: &[Vec<crate::joint_read::Piece>]) -> Vec<Vec<f32>> {
    deep_check_packed_with(model, question, requests, !std::env::var("NANO_DEEP_SHARED").is_ok_and(|v| v == "off"))
}

/// `deep_check_packed` in the shared-span form (`shared`) or one span a form.
#[must_use]
pub fn deep_check_packed_with(model: &Model, question: &str, requests: &[Vec<crate::joint_read::Piece>], shared: bool) -> Vec<Vec<f32>> {
    use crate::joint_read::{ASSISTANT, SYSTEM, piece_text};
    let tok = |s: &str| model.pop.tokenizer.encode(s).ok();
    let first = |s: &str| tok(s).and_then(|v| v.first().and_then(|x| usize::try_from(*x).ok()));
    let (Some(yes), Some(no)) = (first("yes"), first("no")) else { return Vec::new() };
    let cut = |p: &crate::joint_read::Piece| {
        let mut q = p.clone();
        if q.text.len() > DEEP_SPAN_CHARS {
            let mut end = DEEP_SPAN_CHARS;
            while !q.text.is_char_boundary(end) {
                end -= 1;
            }
            q.text.truncate(end);
            q.end = q.start + q.text.lines().count().saturating_sub(1);
        }
        q
    };
    let reqs: Vec<Vec<crate::joint_read::Piece>> = requests.iter().map(|r| r.iter().map(cut).collect()).collect();
    // Lane S's shared-span form (default; `NANO_DEEP_SHARED=off` keeps one span a form): one span a
    // request (the system prompt and the pieces), and per piece one tail for the question and one
    // for each content-free form; about three times faster, the content-free question then only in
    // its tail (the calibration shifts; its order is the ladder's to judge).
    if shared {
        // Stage 5 WP-C3 R2: in the shared form the content-free tails read the span without the
        // question, so their labels depend on the pieces alone: held per piece set
        // (`NULL_HELD_SHARED`), a held request's group carries only its posed tails
        // (`NANO_DEEP_NULL_HELD=off` reads them every call, as before).
        let hold = !std::env::var("NANO_DEEP_NULL_HELD").is_ok_and(|v| v == "off");
        // Keyed by the pieces' content and the model's parameters key, so an edited file or a swapped
        // cartridge never meets a stale sum; bounded (`cache_reads::Lru`).
        let params = crate::neural::parameters_key(model);
        let key = |pieces: &[crate::joint_read::Piece]| -> u64 {
            let body: String = pieces.iter().map(|p| format!("{}:{}-{}#{}#{}|", p.path, p.start, p.end, p.text.len(), p.text)).collect();
            crate::cache_reads::key(&["deep-null-shared", &params, &body])
        };
        let held_of = |r: &[crate::joint_read::Piece]| -> Option<Vec<f32>> {
            if !hold {
                return None;
            }
            NULL_HELD_SHARED.lock().ok().and_then(|mut m| m.get_or_insert_with(|| crate::cache_reads::Lru::new(crate::cache_reads::CAP)).get(key(r))).filter(|v| v.len() == r.len())
        };
        let mut groups: Vec<(Vec<i32>, Vec<Vec<i32>>)> = Vec::new();
        let mut idx: Vec<usize> = Vec::new();
        for (ri, r) in reqs.iter().enumerate() {
            if r.is_empty() {
                continue;
            }
            let held = held_of(r).is_some();
            let Some(mut span) = tok(SYSTEM) else { return Vec::new() };
            for (k, p) in r.iter().enumerate() {
                let Some(t) = tok(&piece_text(k + 1, p)) else { return Vec::new() };
                span.extend(t);
            }
            let mut tails: Vec<Vec<i32>> = Vec::new();
            let forms: &[&str] = if held { &[] } else { &NULL_FORMS };
            for form in std::iter::once(question).chain(forms.iter().copied()) {
                for k in 0..r.len() {
                    let Some(t) = tok(&format!("
Question: {form}
Does piece [{}] answer the question? Answer yes or no.{ASSISTANT}", k + 1)) else { return Vec::new() };
                    tails.push(t);
                }
            }
            groups.push((span, tails));
            idx.push(ri);
        }
        if groups.is_empty() {
            return vec![Vec::new(); requests.len()];
        }
        let _served = nmengn::gpu::ServedScope::new();
        // The packed read needs the segment attention (lane S's served switch); on for this read.
        let seg = nmengn::gpu::SEGMENT_ATTENTION.swap(true, std::sync::atomic::Ordering::Relaxed);
        let res = nmengn::kvstore::read_group_labels(model, None, &groups, &[yes, no]);
        nmengn::gpu::SEGMENT_ATTENTION.store(seg, std::sync::atomic::Ordering::Relaxed);
        let out = match res {
            Ok(o) => o,
            Err(e) => {
                eprintln!("deep check packed: {}", e.message);
                return Vec::new();
            }
        };
        let mut res: Vec<Vec<f32>> = vec![Vec::new(); reqs.len()];
        #[expect(clippy::cast_precision_loss, reason = "three forms")]
        let kf = NULL_FORMS.len() as f32;
        for (ri, g) in idx.iter().zip(&out) {
            let Some(r) = reqs.get(*ri) else { continue };
            let n = r.len();
            let yn: Vec<f32> = g.iter().map(|t| t.first().copied().unwrap_or(0.0) - t.get(1).copied().unwrap_or(0.0)).collect();
            // The content-free sum per piece: held, or read now (then held).
            let nulls: Vec<f32> = match held_of(r) {
                Some(v) => v,
                None => {
                    let v: Vec<f32> = (0..n).map(|k| (1..=NULL_FORMS.len()).map(|f| yn.get(f * n + k).copied().unwrap_or(0.0)).sum()).collect();
                    if hold && let Ok(mut m) = NULL_HELD_SHARED.lock() {
                        m.get_or_insert_with(|| crate::cache_reads::Lru::new(crate::cache_reads::CAP)).put(key(r), v.clone());
                    }
                    v
                }
            };
            res[*ri] = (0..n).map(|k| yn.get(k).copied().unwrap_or(0.0) - nulls.get(k).copied().unwrap_or(0.0) / kf).collect();
        }
        return res;
    }
    let group = |form: &str, pieces: &[crate::joint_read::Piece]| -> Option<(Vec<i32>, Vec<Vec<i32>>)> {
        let mut span = tok(&format!("{SYSTEM}Question: {form}\n\n"))?;
        for (k, p) in pieces.iter().enumerate() {
            span.extend(tok(&piece_text(k + 1, p))?);
        }
        let tails: Vec<Vec<i32>> = (0..pieces.len()).map(|k| tok(&format!("\nQuestion: {form}\nDoes piece [{}] answer the question? Answer yes or no.{ASSISTANT}", k + 1))).collect::<Option<_>>()?;
        Some((span, tails))
    };
    let key = |pieces: &[crate::joint_read::Piece]| -> String { pieces.iter().map(|p| format!("{}:{}-{}#{}|", p.path, p.start, p.end, p.text.len())).collect() };
    // The groups: posed for every request; the content-free forms only for requests not held.
    let mut groups: Vec<(Vec<i32>, Vec<Vec<i32>>)> = Vec::new();
    let mut slots: Vec<(usize, Option<usize>)> = Vec::new(); // (request, form index or None = posed)
    for (ri, r) in reqs.iter().enumerate() {
        if r.is_empty() {
            continue;
        }
        let Some(g) = group(question, r) else { return Vec::new() };
        groups.push(g);
        slots.push((ri, None));
        let held = NULL_HELD.lock().ok().is_some_and(|m| m.get(&key(r)).is_some_and(|v| v.len() == r.len()));
        if !held {
            for (fi, form) in NULL_FORMS.iter().enumerate() {
                let Some(g) = group(form, r) else { return Vec::new() };
                groups.push(g);
                slots.push((ri, Some(fi)));
            }
        }
    }
    if groups.is_empty() {
        return vec![Vec::new(); requests.len()];
    }
    let _served = nmengn::gpu::ServedScope::new();
    let seg = nmengn::gpu::SEGMENT_ATTENTION.swap(true, std::sync::atomic::Ordering::Relaxed);
    let res = nmengn::kvstore::read_group_labels(model, None, &groups, &[yes, no]);
    nmengn::gpu::SEGMENT_ATTENTION.store(seg, std::sync::atomic::Ordering::Relaxed);
    let out = match res {
            Ok(o) => o,
            Err(e) => {
                eprintln!("deep check packed: {}", e.message);
                return Vec::new();
            }
        };
    let mut posed: Vec<Vec<f32>> = vec![Vec::new(); reqs.len()];
    let mut nulls: Vec<Vec<f32>> = reqs.iter().map(|r| vec![0.0; r.len()]).collect();
    for ((ri, form), g) in slots.iter().zip(&out) {
        let yn: Vec<f32> = g.iter().map(|t| t.first().copied().unwrap_or(0.0) - t.get(1).copied().unwrap_or(0.0)).collect();
        match form {
            None => posed[*ri] = yn,
            Some(_) => {
                for (n, x) in nulls[*ri].iter_mut().zip(yn) {
                    *n += x;
                }
            }
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "three forms")]
    let k = NULL_FORMS.len() as f32;
    reqs.iter()
        .enumerate()
        .map(|(ri, r)| {
            if r.is_empty() {
                return Vec::new();
            }
            let held = NULL_HELD.lock().ok().and_then(|m| m.get(&key(r)).cloned()).filter(|v| v.len() == r.len());
            let null = match held {
                Some(v) => v,
                None => {
                    let v = nulls[ri].clone();
                    if let Ok(mut m) = NULL_HELD.lock() {
                        m.insert(key(r), v.clone());
                    }
                    v
                }
            };
            posed[ri].iter().zip(&null).map(|(p, n)| p - n / k).collect()
        })
        .collect()
}

/// The span the packed deep check reads of each piece: its first characters (about 250 tokens).
pub const DEEP_SPAN_CHARS: usize = 1_000;
