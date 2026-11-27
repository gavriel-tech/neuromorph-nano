//! Sockets: the places in a MicroModel's layers where a cartridge plugs in and filters it.
//!
//! A cartridge of learned parameters filters a MicroModel at named points of its layers.
//! The form is a per-channel gain and bias over one activation, `y = x * gain + bias`, where a
//! point is a node of the model's assembly by its short id (`L27.swigate`, the gated feed-forward
//! of the last layer; `L20.o`, an attention output; `final_norm`). This is the representation
//! editing form of (IA)3 and RED: a gain on a linear's input equals scaling that linear's input
//! columns, so a cartridge acts as a filter over the released weights without writing them; the
//! released weights stay read-only and the filter is applied to the activation as it passes.
//!
//! Stacking. Cartridges stacked in a defined order compose per point: cartridge A then cartridge B gives
//! `(x * gA + bA) * gB + bB`, folded once in float64 into one gain and bias
//! (`gA * gB`, `bA * gB + bB`), so a point costs one element-wise pass however many cartridges filter
//! it. Order matters when a cartridge has a bias, and the fold keeps it.
//!
//! Arithmetic. One fused multiply-add in float32 and one rounding to binary16 per value, on the
//! processor and on the card alike (`filter_kernel`), so both give the same bits for the same
//! input. A zero bias is stored as negative zero, so a channel with gain 1 and no bias returns
//! its input bit for bit (positive zero plus negative zero is positive zero; negative zero plus
//! negative zero is negative zero).
//!
//! Plugging. The rack holds the filters plugged per MicroModel (by its name, for example
//! `qwen3-reranker-0.6b`); every run of any assembly over that MicroModel reads it, so a stack
//! switch reaches the next call without a restart, and unplugging returns the exact base: with
//! nothing plugged, no node is touched. A model can also hold its own filters, which take the
//! rack's place for that model alone (used by tests that must not touch other runs).
use crate::gpu::{DevT, Ops};
use crate::model::Act;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

/// One cartridge's filter at one point: the point's short node id, a gain and a bias of its width.
#[derive(Clone, Debug, PartialEq)]
pub struct Part {
    pub target: String,
    pub gain: Vec<f32>,
    pub bias: Vec<f32>,
}

/// One cartridge's parts, by the cartridge's key.
#[derive(Clone, Debug, PartialEq)]
pub struct CartridgeFilter {
    pub cartridge: String,
    pub parts: Vec<Part>,
}

/// The composed filter at one point: the cartridges that filter it, in order, folded into one gain
/// and bias; the card copies are made on first use and kept while plugged.
#[derive(Debug)]
pub struct NodeFilter {
    pub target: String,
    pub cartridges: Vec<String>,
    pub gain: Vec<f32>,
    pub bias: Vec<f32>,
    dev: Mutex<Option<(DevT, DevT)>>,
}
impl NodeFilter {
    /// Bytes this point holds on the card once used there (gain and bias, float32).
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        if self.dev.lock().map(|d| d.is_some()).unwrap_or(false) { self.gain.len() * 8 } else { 0 }
    }
}

/// The filters plugged on one MicroModel: its cartridges in order and the composed filter per point.
#[derive(Debug, Default)]
pub struct Plugged {
    pub model: String,
    pub cartridges: Vec<String>,
    pub nodes: BTreeMap<String, Arc<NodeFilter>>,
}
impl Plugged {
    /// Values held (gain and bias) over every point.
    #[must_use]
    pub fn values(&self) -> usize {
        self.nodes.values().map(|n| n.gain.len() * 2).sum()
    }
    /// Bytes held on the card now.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.nodes.values().map(|n| n.device_bytes()).sum()
    }
}

/// Folds gain and bias pairs, applied in order, into one pair: in float64, rounded once to
/// float32; a zero bias becomes negative zero (see the module note).
/// # Errors
/// Pairs of different widths, or a gain and bias of different lengths.
pub fn compose(parts: &[(&[f32], &[f32])]) -> Result<(Vec<f32>, Vec<f32>), String> {
    let width = parts.first().map_or(0, |p| p.0.len());
    if parts.iter().any(|(g, b)| g.len() != width || b.len() != width) {
        return Err(format!("filters at one point must share one width ({width})"));
    }
    let mut g = vec![1.0_f64; width];
    let mut b = vec![0.0_f64; width];
    for (pg, pb) in parts {
        for i in 0..width {
            let (gi, bi) = (f64::from(pg.get(i).copied().unwrap_or(1.0)), f64::from(pb.get(i).copied().unwrap_or(0.0)));
            if let (Some(gv), Some(bv)) = (g.get_mut(i), b.get_mut(i)) {
                *bv = *bv * gi + bi;
                *gv *= gi;
            }
        }
    }
    #[expect(clippy::cast_possible_truncation, reason = "the composed filter is stored in float32 by design")]
    let to32 = |v: f64| v as f32;
    let bias = b.iter().map(|v| if *v == 0.0 { -0.0 } else { to32(*v) }).collect();
    Ok((g.iter().map(|v| to32(*v)).collect(), bias))
}

/// One part's gain and bias, borrowed.
type Pair<'a> = (&'a [f32], &'a [f32]);

/// Builds the plugged set for `model` from `cartridges` in order: parts at the same point compose.
/// # Errors
/// A part whose gain and bias differ in length, or parts of different widths at one point.
pub fn build(model: &str, cartridges: &[CartridgeFilter]) -> Result<Plugged, String> {
    let mut by: BTreeMap<String, (Vec<String>, Vec<Pair<'_>>)> = BTreeMap::new();
    for b in cartridges {
        for p in &b.parts {
            if p.gain.len() != p.bias.len() || p.gain.is_empty() {
                return Err(format!("{}: {}: gain of {} and bias of {}", b.cartridge, p.target, p.gain.len(), p.bias.len()));
            }
            let e = by.entry(p.target.clone()).or_default();
            e.0.push(b.cartridge.clone());
            e.1.push((&p.gain, &p.bias));
        }
    }
    let mut nodes = BTreeMap::new();
    for (t, (names, pairs)) in by {
        let (gain, bias) = compose(&pairs).map_err(|e| format!("{t}: {e}"))?;
        nodes.insert(t.clone(), Arc::new(NodeFilter { target: t, cartridges: names, gain, bias, dev: Mutex::new(None) }));
    }
    Ok(Plugged { model: model.to_owned(), cartridges: cartridges.iter().map(|b| b.cartridge.clone()).collect(), nodes })
}

/// Composes two already-built `Plugged` filter stacks (`base` applied first, then `extra`)
/// per target node in float64 rounded once to float32, without touching global `RACK`.
/// Used by per-hop `RunOpts::filters` (WP-A2) so a job-path filter can layer cleanly over
/// a model's rack/base filters for a single forward call.
/// # Errors
/// Width mismatch when both stacks filter the same node.
pub fn stack_plugged(base: &Plugged, extra: &Plugged) -> Result<Arc<Plugged>, String> {
    let mut cartridges = base.cartridges.clone();
    cartridges.extend(extra.cartridges.iter().cloned());
    let mut nodes: BTreeMap<String, Arc<NodeFilter>> = base.nodes.clone();
    for (target, ef) in &extra.nodes {
        if let Some(bf) = nodes.get(target) {
            let (gain, bias) = compose(&[(&bf.gain, &bf.bias), (&ef.gain, &ef.bias)]).map_err(|e| format!("{target}: {e}"))?;
            let mut names = bf.cartridges.clone();
            names.extend(ef.cartridges.iter().cloned());
            nodes.insert(target.clone(), Arc::new(NodeFilter { target: target.clone(), cartridges: names, gain, bias, dev: Mutex::new(None) }));
        } else {
            nodes.insert(target.clone(), Arc::clone(ef));
        }
    }
    Ok(Arc::new(Plugged { model: base.model.clone(), cartridges, nodes }))
}

type StackedCacheEntry = (std::sync::Weak<Plugged>, std::sync::Weak<Plugged>, Arc<Plugged>);
static STACKED_CACHE: Mutex<Vec<StackedCacheEntry>> = Mutex::new(Vec::new());

/// Cached version of [`stack_plugged`] keyed by live `Arc` identity (`Weak::ptr_eq`),
/// keeping composed `NodeFilter` GPU tensors resident across repeated forward calls while
/// automatically invalidating whenever either underlying filter stack is hot-swapped.
/// # Errors
/// Width mismatch when both stacks filter the same node.
pub fn stack_plugged_cached(base: &Arc<Plugged>, extra: &Arc<Plugged>) -> Result<Arc<Plugged>, String> {
    if let Ok(mut g) = STACKED_CACHE.lock() {
        g.retain(|(wb, we, _)| wb.strong_count() > 0 && we.strong_count() > 0);
        if let Some((_, _, cached)) = g.iter().find(|(wb, we, _)| wb.ptr_eq(&Arc::downgrade(base)) && we.ptr_eq(&Arc::downgrade(extra))) {
            return Ok(Arc::clone(cached));
        }
    }
    let built = stack_plugged(base, extra)?;
    if let Ok(mut g) = STACKED_CACHE.lock() {
        if g.len() >= 32 {
            g.remove(0);
        }
        g.push((Arc::downgrade(base), Arc::downgrade(extra), Arc::clone(&built)));
    }
    Ok(built)
}

/// Dynamic per-hop socket compatibility check across ALL 4+ MicroModels (no hardcoded
/// `SERVED_MODELS` restriction): validates socket contract, active base weight hash, and
/// that every target node exists in the model's assembly.
/// # Errors
/// The first reason the filter does not fit.
pub fn fits_per_hop(home: &std::path::Path, model: &str, weights: &str, contract: &str, targets: &[&str]) -> Result<(), String> {
    if contract != SOCKETS {
        return Err(format!("written for socket contract {contract:?}, this engine has {SOCKETS}"));
    }
    let card = nmtype::models_dir(home).join(model).join("part.json");
    let open = std::fs::read(&card)
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .and_then(|v| v.get("weights").and_then(|w| w.get("sha256")).and_then(nmjson::Value::string).ok().map(str::to_owned));
    match crate::base::base_of(model).or(open) {
        Some(h) if h == weights => {}
        Some(h) => return Err(format!("made for base {weights:.12}, this {model} runs on base {h:.12}")),
        None => return Err(format!("{}: no weight hash to check against", card.display())),
    }
    let asm_name = model.strip_suffix("-0.6b").or_else(|| model.strip_suffix("-1.7b")).unwrap_or(model);
    let asm = crate::Assembly::load(&home.join("designs").join("assemblies"), asm_name).map_err(|e| e.to_string())?;
    if let Some(t) = targets.iter().find(|t| !asm.nodes.iter().any(|n| n.id == crate::assembly::nid(t))) {
        return Err(format!("{model} has no socket {t}"));
    }
    Ok(())
}

/// Parses one `library/parameter-maps/<role>/job-filter-<job>.json` document written by
/// `nano-lab run pmap surgery job-filter` into a `CartridgeFilter`. Returns `Ok(None)` when
/// `"enabled": false` is set (toggled off in `/cartridges`).
/// # Errors
/// Malformed JSON, socket contract mismatch, model mismatch, or invalid gain/bias arrays.
pub fn parse_job_filter_json(bytes: &[u8], expected_pop: &str) -> Result<Option<CartridgeFilter>, String> {
    let val = nmjson::parse(bytes).map_err(|e| format!("job-filter json: {e}"))?;
    let obj = val.object().map_err(|e| format!("job-filter root: {e}"))?;
    if let Some(nmjson::Value::Bool(false)) = obj.get("enabled") {
        return Ok(None);
    }
    let contract = obj.get("contract").and_then(|v| v.string().ok()).unwrap_or("");
    if contract != SOCKETS {
        return Err(format!("job-filter contract {contract:?} != {SOCKETS}"));
    }
    let model = obj.get("model").and_then(|v| v.string().ok()).unwrap_or("");
    if !expected_pop.is_empty() && !model.is_empty() && !names_model(expected_pop, model) {
        return Err(format!("job-filter built for {model}, requested for {expected_pop}"));
    }
    let cartridge = obj.get("cartridge").and_then(|v| v.string().ok()).unwrap_or("job-filter").to_owned();
    let arr = obj.get("parts").and_then(|v| v.array().ok()).ok_or("job-filter missing parts array")?;
    let parse_vec = |v: &nmjson::Value, label: &str| -> Result<Vec<f32>, String> {
        let items = v.array().map_err(|e| format!("{label}: {e}"))?;
        let mut out = Vec::with_capacity(items.len());
        for it in items {
            match it {
                nmjson::Value::Number(s) => out.push(s.parse::<f32>().map_err(|e| format!("{label} num {s}: {e}"))?),
                _ => return Err(format!("{label}: non-numeric entry")),
            }
        }
        Ok(out)
    };
    let mut parts = Vec::with_capacity(arr.len());
    for p in arr {
        let po = p.object().map_err(|e| format!("part: {e}"))?;
        let target = po.get("target").and_then(|v| v.string().ok()).ok_or("part missing target")?.to_owned();
        let gain = parse_vec(po.get("gain").ok_or("part missing gain")?, "gain")?;
        let bias = parse_vec(po.get("bias").ok_or("part missing bias")?, "bias")?;
        parts.push(Part { target, gain, bias });
    }
    if parts.is_empty() {
        return Ok(None);
    }
    Ok(Some(CartridgeFilter { cartridge, parts }))
}

/// Resolves the short role directory name (`reranker`, `embedder`, `reader`, `writer`, or any
/// future role registered in `designs/base-cartridges.json`) for a population id.
#[must_use]
pub fn role_for_pop(home: &std::path::Path, pop_id: &str) -> String {
    // 2026-11-25: the role a chosen model runs as (`nmtype::role_models`) first.
    if let Some(r) = nmtype::role_models::role_of(pop_id) {
        return r.to_owned();
    }
    if let Ok(text) = std::fs::read_to_string(home.join("designs/base-cartridges.json"))
        && let Ok(nmjson::Value::Object(root)) = nmjson::parse(text.as_bytes())
        && let Some(nmjson::Value::Object(serving)) = root.get("serving")
    {
        for (pid, entry) in serving {
            if names_model(pop_id, pid)
                && let Ok(cart) = entry.get("cartridge").and_then(nmjson::Value::string)
            {
                return cart.strip_suffix("-base").unwrap_or(cart).to_owned();
            }
        }
    }
    if pop_id.contains("reranker") {
        "reranker".to_owned()
    } else if pop_id.contains("embedding") {
        "embedder".to_owned()
    } else if pop_id.contains("1.7b") {
        "writer".to_owned()
    } else {
        "reader".to_owned()
    }
}

/// 2026-11-26 (v0.1.0): a base key as it is compared: `cartridge writer-base@4` (a population's
/// source label) and `writer-base@4` are the same key; `open weights`, `open` and `open:<model>`
/// all name the open weights.
fn base_key_norm(k: &str) -> Option<String> {
    let k = k.trim();
    let k = k.strip_prefix("cartridge ").unwrap_or(k).trim();
    if k.is_empty() {
        return None;
    }
    // `open weights with overlay <id>`: the open weights with a parameter overlay are still the
    // open weights a filter was built on.
    if k.starts_with(crate::model::OPEN_WEIGHTS) || k == "open" || k.starts_with("open:") {
        return Some("open".to_owned());
    }
    Some(k.to_owned())
}

/// 2026-11-26 (v0.1.0): whether a job filter was built on the base that serves its MicroModel now.
/// `built_on` is the `made_on` its lineage records (a base cartridge key such as `writer-base@4`,
/// or `open:<model>`), `serving` the base serving that MicroModel (a key, `cartridge <key>` or
/// `open weights`). `Some(true)` the same base, `Some(false)` another one, `None` when either is
/// not known. This only reports: a filter that does not match still loads.
#[must_use]
pub fn job_filter_base_match(built_on: Option<&str>, serving: Option<&str>) -> Option<bool> {
    let b = built_on.and_then(base_key_norm)?;
    let s = serving.and_then(base_key_norm)?;
    Some(b == s)
}

/// 2026-11-26 (v0.1.0): the base a job filter file in a role folder was built on, as the folder's
/// `lineage.json` records it (`maps.<file>.made_on`, the same record `nmapi::lineage` writes, read
/// here directly because this crate sits below it). `None` when not recorded.
#[must_use]
pub fn job_filter_built_on(role_dir: &std::path::Path, file: &str) -> Option<String> {
    let bytes = std::fs::read(role_dir.join("lineage.json")).ok()?;
    let v = nmjson::parse(&bytes).ok()?;
    let made_on = v.get("maps").ok()?.get(file).ok()?.get("made_on").ok()?.string().ok()?.trim().to_owned();
    (!made_on.is_empty()).then_some(made_on)
}

type JobFilterStamp = Vec<(String, u64, u64)>;
struct JobFilterCacheEntry {
    checked_at: std::time::Instant,
    stamp: JobFilterStamp,
    plugged: Option<Arc<Plugged>>,
}
static JOB_FILTER_CACHE: Mutex<BTreeMap<String, JobFilterCacheEntry>> = Mutex::new(BTreeMap::new());
const JOB_FILTER_POLL_MS: u64 = 1000;

/// Loads and composes all currently enabled per-hop job filters for `pop_id` from
/// `library/parameter-maps/<role>/job-filter-*.json`, cached in memory with a 1-second
/// `(name, mtime, len)` refresh so hot-swapping On/Off/Remove in `/cartridges` takes effect
/// without a restart while keeping per-forward-pass filesystem overhead at zero.
/// Returns `Ok(None)` when no enabled job filters exist. `serving_base` is the base the
/// MicroModel runs on now (its population's source label); a filter built on another base still
/// loads, with one warning line (2026-11-26).
/// # Errors
/// Invalid filter file or composition mismatch.
pub fn load_enabled_job_filters(home: &std::path::Path, pop_id: &str, serving_base: &str) -> Result<Option<Arc<Plugged>>, String> {
    let cache_key = format!("{}|{pop_id}", home.display());
    let now = std::time::Instant::now();
    if let Ok(g) = JOB_FILTER_CACHE.lock()
        && let Some(entry) = g.get(&cache_key)
        && now.duration_since(entry.checked_at) < std::time::Duration::from_millis(JOB_FILTER_POLL_MS)
    {
        return Ok(entry.plugged.clone());
    }
    let role = role_for_pop(home, pop_id);
    let dir = home.join("library/parameter-maps").join(&role);
    let mut paths: Vec<std::path::PathBuf> = match std::fs::read_dir(&dir) {
        Ok(rd) => rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|n| n.starts_with("job-filter-") && n.ends_with(".json"))
            })
            .collect(),
        Err(_) => {
            if let Ok(mut g) = JOB_FILTER_CACHE.lock() {
                g.insert(cache_key, JobFilterCacheEntry { checked_at: now, stamp: Vec::new(), plugged: None });
            }
            return Ok(None);
        }
    };
    paths.sort();
    let mut stamp: JobFilterStamp = Vec::with_capacity(paths.len());
    for p in &paths {
        if let Ok(meta) = std::fs::metadata(p) {
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX));
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("").to_owned();
            stamp.push((name, mtime, meta.len()));
        }
    }
    // The parameter overlays are part of what is loaded: an overlay written for a job filter reaches
    // the next call as a plug change does.
    stamp.push(("overlays".to_owned(), crate::overlay::stamp(home), 0));
    if let Ok(mut g) = JOB_FILTER_CACHE.lock()
        && let Some(entry) = g.get_mut(&cache_key)
        && entry.stamp == stamp
    {
        entry.checked_at = now;
        return Ok(entry.plugged.clone());
    }
    let mut filters = Vec::new();
    for p in &paths {
        let bytes = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
        // 2026-11-25: a job filter made for another model (the role now runs on a model the user
        // chose) is not loaded on it, with one line, not an error on every call: it stays with the
        // model it was made for.
        match parse_job_filter_json(&bytes, pop_id) {
            Ok(Some(mut cf)) => {
                // The job filter's parameter overlay (`filter:<role>/<job>`): masked whole, it is left
                // out while it stays plugged in; otherwise its edits and masks apply to its points.
                let job = p.file_name().and_then(|s| s.to_str()).and_then(|n| n.strip_prefix("job-filter-")?.strip_suffix(".json")).unwrap_or("");
                if let Some(ov) = crate::overlay::effective(home, &format!("filter:{role}/{job}")) {
                    if ov.whole_masked() {
                        continue;
                    }
                    ov.apply_parts(&mut cf.parts);
                }
                // 2026-11-26 (v0.1.0): a job filter built on another base cartridge version than
                // the one serving is still loaded (nothing is switched off without the owner's
                // confirmation); one warning line per filter, built-on and serving base.
                let file = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                let built_on = job_filter_built_on(&dir, file);
                if job_filter_base_match(built_on.as_deref(), Some(serving_base)) == Some(false) {
                    static WARNED: Mutex<Vec<String>> = Mutex::new(Vec::new());
                    let built = built_on.as_deref().unwrap_or("");
                    let key = format!("{}|{built}|{serving_base}", p.display());
                    if let Ok(mut w) = WARNED.lock()
                        && !w.contains(&key)
                    {
                        eprintln!("job filters: warning: {role}/{file} was built on {built}, {pop_id} serves on {serving_base}; it stays loaded, rebuild it on the serving base to match");
                        w.push(key);
                    }
                }
                filters.push(cf);
            }
            Ok(None) => {}
            Err(e) if e.starts_with("job-filter built for") => {
                static SAID: Mutex<Vec<String>> = Mutex::new(Vec::new());
                let name = p.display().to_string();
                if let Ok(mut said) = SAID.lock()
                    && !said.contains(&name)
                {
                    eprintln!("job filters: {} not loaded here, it stays with its own model: {e}", p.file_name().and_then(|s| s.to_str()).unwrap_or(""));
                    said.push(name);
                }
            }
            Err(e) => return Err(e),
        }
    }
    let built = if filters.is_empty() { None } else { Some(Arc::new(build(pop_id, &filters)?)) };
    if let Ok(mut g) = JOB_FILTER_CACHE.lock() {
        g.insert(cache_key, JobFilterCacheEntry { checked_at: now, stamp, plugged: built.clone() });
    }
    // 2026-11-24: a new set of job filters is a new filter generation, so every output stored
    // under the old set (the reranker's prefix state, stored reads, stored replies) is not reused.
    GENERATION.fetch_add(1, Ordering::SeqCst);
    Ok(built)
}

static RACK: RwLock<BTreeMap<String, Arc<Plugged>>> = RwLock::new(BTreeMap::new());
static GENERATION: AtomicU64 = AtomicU64::new(0);

/// Plugs `cartridges` on `model` in order, replacing what was plugged there; an empty list unplugs.
/// Returns the plugged set.
/// # Errors
/// As `build`.
pub fn plug(model: &str, cartridges: &[CartridgeFilter]) -> Result<Arc<Plugged>, String> {
    let p = Arc::new(build(model, cartridges)?);
    let mut g = RACK.write().unwrap_or_else(std::sync::PoisonError::into_inner);
    if cartridges.is_empty() {
        g.remove(model);
    } else {
        g.insert(model.to_owned(), Arc::clone(&p));
    }
    GENERATION.fetch_add(1, Ordering::SeqCst);
    Ok(p)
}
/// Unplugs every cartridge of `model`. Returns whether anything was plugged.
pub fn unplug(model: &str) -> bool {
    let had = RACK.write().unwrap_or_else(std::sync::PoisonError::into_inner).remove(model).is_some();
    GENERATION.fetch_add(1, Ordering::SeqCst);
    had
}
/// Unplugs every cartridge of every MicroModel.
pub fn unplug_all() {
    RACK.write().unwrap_or_else(std::sync::PoisonError::into_inner).clear();
    GENERATION.fetch_add(1, Ordering::SeqCst);
}
/// The MicroModels whose sockets take cartridges in a serving process. The reranker keeps no model
/// output across calls (the reply cache keys on `generation`); the base model's encoded chunks and
/// the embedder's index on disk are kept across calls, so a cartridge there would mix filtered and
/// unfiltered numbers until those are rebuilt, and they are refused until that is wired.
pub const SERVED_MODELS: &[&str] = &["qwen3-reranker-0.6b"];

/// The socket contract: what a cartridge's `gain-bias` parts mean to this engine (a node's `y`
/// output by short id, `x * gain + bias` per channel, one rounding to the activation's type). A
/// cartridge written for another contract is refused at the socket.
pub const SOCKETS: &str = "nmengn-sockets/1";

/// Whether an additional cartridge fits this engine's sockets on `model`: a MicroModel with served
/// sockets, made for the base the MicroModel runs on now (`weights`, the SHA-256 on the label,
/// against `base::base_of`, else the open weights' SHA-256 on the weight card), for this socket
/// contract, every point a node of the model's assembly. `home` holds the assemblies and the
/// weight cards.
/// # Errors
/// The first reason it does not fit, by name.
pub fn fits(home: &std::path::Path, model: &str, weights: &str, contract: &str, targets: &[&str]) -> Result<(), String> {
    if !SERVED_MODELS.contains(&model) {
        return Err(format!("no served socket on {model}"));
    }
    if contract != SOCKETS {
        return Err(format!("written for socket contract {contract:?}, this engine has {SOCKETS}"));
    }
    let card = nmtype::models_dir(home).join(model).join("part.json");
    let open = std::fs::read(&card).ok().and_then(|b| nmjson::parse(&b).ok()).and_then(|v| v.get("weights").and_then(|w| w.get("sha256")).and_then(nmjson::Value::string).ok().map(str::to_owned));
    match crate::base::base_of(model).or(open) {
        Some(h) if h == weights => {}
        Some(h) => return Err(format!("made for base {weights:.12}, this {model} runs on base {h:.12}")),
        None => return Err(format!("{}: no weight hash to check against", card.display())),
    }
    let asm_name = model.strip_suffix("-0.6b").unwrap_or(model);
    let asm = crate::Assembly::load(&home.join("designs").join("assemblies"), asm_name).map_err(|e| e.to_string())?;
    if let Some(t) = targets.iter().find(|t| !asm.nodes.iter().any(|n| n.id == crate::assembly::nid(t))) {
        return Err(format!("{model} has no socket {t}"));
    }
    Ok(())
}

/// The whole rack wanted next (MicroModel to its cartridges in order), applied at the next call
/// boundary by `settle`, so one call never reads two different stacks.
static PENDING: Mutex<Option<BTreeMap<String, Vec<CartridgeFilter>>>> = Mutex::new(None);

/// Checks and holds the rack wanted next (every MicroModel absent from `all` is unplugged then).
/// # Errors
/// A MicroModel not in `SERVED_MODELS`, or a set that does not build.
pub fn stage(all: BTreeMap<String, Vec<CartridgeFilter>>) -> Result<(), String> {
    for (m, cs) in &all {
        if !SERVED_MODELS.contains(&m.as_str()) {
            return Err(format!("{m}: no served socket (sockets serve {})", SERVED_MODELS.join(", ")));
        }
        build(m, cs)?;
    }
    if let Ok(mut g) = PENDING.lock() {
        *g = Some(all);
    }
    Ok(())
}

/// Applies the rack `stage` held, if any, replacing the whole rack. Called where a call starts.
/// Returns whether anything changed.
/// # Errors
/// A set that does not build (it was checked when staged, so only on a fault).
pub fn settle() -> Result<bool, String> {
    let Some(all) = PENDING.lock().ok().and_then(|mut g| g.take()) else { return Ok(false) };
    replace_all(&all)?;
    Ok(true)
}

/// Replaces the whole rack with `all` at once (MicroModels absent from it are unplugged).
/// # Errors
/// As `build`; nothing changes then.
pub fn replace_all(all: &BTreeMap<String, Vec<CartridgeFilter>>) -> Result<(), String> {
    let mut next = BTreeMap::new();
    for (m, cs) in all.iter().filter(|(_, cs)| !cs.is_empty()) {
        next.insert(m.clone(), Arc::new(build(m, cs)?));
    }
    *RACK.write().unwrap_or_else(std::sync::PoisonError::into_inner) = next;
    GENERATION.fetch_add(1, Ordering::SeqCst);
    Ok(())
}

/// The job filters each MicroModel's forward passes have loaded (the per-hop cache), so the live
/// view reports them as plugged in (2026-11-25: every part shows it is properly connected; the
/// live view listed only the rack and never the job filters).
#[must_use]
pub fn job_filters_loaded() -> Vec<Arc<Plugged>> {
    JOB_FILTER_CACHE.lock().map(|g| g.values().filter_map(|e| e.plugged.clone()).collect()).unwrap_or_default()
}

/// What is plugged, per MicroModel.
#[must_use]
pub fn plugged() -> Vec<Arc<Plugged>> {
    RACK.read().map(|g| g.values().cloned().collect()).unwrap_or_default()
}
/// Changes on every plug and unplug (a cache of model outputs keys on it with `signature`).
#[must_use]
pub fn generation() -> u64 {
    GENERATION.load(Ordering::SeqCst)
}
/// The cartridges plugged on the MicroModel of `pop_id`, joined (empty when none): what a stored
/// output of that model must record to be reused.
#[must_use]
pub fn signature(pop_id: &str) -> String {
    for_pop(pop_id).map(|p| p.cartridges.join("+")).unwrap_or_default()
}

/// Does a weight card id (`weight-opnwgt-basewt-qwen3-reranker-0.6b`) name MicroModel `model`?
#[must_use]
pub fn names_model(pop_id: &str, model: &str) -> bool {
    pop_id == model || pop_id.strip_suffix(model).is_some_and(|p| p.ends_with('-'))
}
/// The rack's filters for the population `pop_id`.
#[must_use]
pub fn for_pop(pop_id: &str) -> Option<Arc<Plugged>> {
    let g = RACK.read().ok()?;
    if g.is_empty() {
        return None;
    }
    g.iter().find(|(m, _)| names_model(pop_id, m)).map(|(_, p)| Arc::clone(p))
}

/// `r16(fma(x, gain, bias))` per value, in place; `x` is rows of the gain's width.
/// # Errors
/// An activation whose length is not a whole number of rows.
pub fn apply_cpu(x: &mut [f32], f: &NodeFilter) -> Result<(), String> {
    let w = f.gain.len();
    if w == 0 || !x.len().is_multiple_of(w) {
        return Err(format!("{}: filter of width {w} over {} values", f.target, x.len()));
    }
    for row in x.chunks_exact_mut(w) {
        for ((v, g), b) in row.iter_mut().zip(&f.gain).zip(&f.bias) {
            *v = crate::half::r16(v.mul_add(*g, *b));
        }
    }
    Ok(())
}
/// The same on the card: the filtered activation as a new tensor.
/// # Errors
/// A width that does not fit, an activation that is not binary16, or a device failure.
pub fn apply_gpu(o: &Ops, x: &DevT, f: &NodeFilter) -> Result<DevT, String> {
    let w = f.gain.len();
    if w == 0 || !x.elements().is_multiple_of(w) || x.shape.last().copied().unwrap_or(0) % w != 0 {
        return Err(format!("{}: filter of width {w} over a tensor of {:?}", f.target, x.shape));
    }
    if x.dtype != crate::gpu::F16 {
        return Err(format!("{}: filters read binary16 activations", f.target));
    }
    let (g, b) = {
        let mut d = f.dev.lock().map_err(|_| "filter lock".to_owned())?;
        if d.is_none() {
            *d = Some((o.upload_vec32(&f.gain)?, o.upload_vec32(&f.bias)?));
        }
        d.clone().ok_or("filter upload")?
    };
    o.filter(x, &g, &b)
}

/// The layer name of a tentacle body's component: `t<k>.S<n>.<op>` as `L<n>.<op>`.
fn body_alias(short: &str) -> Option<String> {
    let (t, rest) = short.split_once('.')?;
    let k = t.strip_prefix('t')?;
    let (s, op) = rest.split_once('.')?;
    let n = s.strip_prefix('S')?;
    (!k.is_empty() && k.bytes().all(|b| b.is_ascii_digit()) && !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())).then(|| format!("L{n}.{op}"))
}

#[cfg(test)]
mod body_alias_tests {
    #[test]
    fn body_components_take_their_layers_filter() {
        assert_eq!(super::body_alias("t1.S13.swigate").as_deref(), Some("L13.swigate"));
        assert_eq!(super::body_alias("t8.S0.attn_norm").as_deref(), Some("L0.attn_norm"));
        assert_eq!(super::body_alias("L13.swigate"), None);
        assert_eq!(super::body_alias("t1.X3.swigate"), None);
        assert_eq!(super::body_alias("tx.S3.swigate"), None);
    }
}

/// Applies the plugged filter of node `node_id` (if any) to its `y` output.
/// # Errors
/// As `apply_cpu` and `apply_gpu`, or an output that is not an activation.
pub(crate) fn apply(p: &Plugged, ops: Option<&Ops>, node_id: &str, outs: &mut [(&'static str, Act)]) -> Result<(), String> {
    let short = node_id.rsplit('#').next().unwrap_or(node_id);
    // 2026-11-24: a tentacle body names layer n's components `t<k>.S<n>.<op>`; a filter names
    // them `L<n>.<op>`, so a body's component takes the filter of its layer (before, job filters
    // never reached the bodies and their readings left the reference path's).
    let alias = if p.nodes.contains_key(short) { None } else { body_alias(short) };
    let Some(f) = p.nodes.get(alias.as_deref().unwrap_or(short)) else { return Ok(()) };
    let _span = nmffi::prof::Span::new("filter");
    for (port, a) in outs.iter_mut().filter(|(port, _)| *port == "y") {
        match (a, ops) {
            (Act::Cpu(x), _) => apply_cpu(x, f)?,
            (Act::Gpu(x), Some(o)) => *x = apply_gpu(o, x, f)?,
            _ => return Err(format!("{node_id}.{port}: a filter reads activations only")),
        }
    }
    Ok(())
}

/// The float64 reference of cartridges applied one after another: `((x * g1 + b1) * g2 + b2) ...` per
/// value, with no rounding until the end (tests compare the engine against it).
#[must_use]
pub fn reference(x: &[f32], cartridges: &[(&[f32], &[f32])]) -> Vec<f64> {
    let w = cartridges.first().map_or(1, |b| b.0.len()).max(1);
    x.iter()
        .enumerate()
        .map(|(i, v)| {
            cartridges.iter().fold(f64::from(*v), |acc, (g, b)| {
                acc * f64::from(g.get(i % w).copied().unwrap_or(1.0)) + f64::from(b.get(i % w).copied().unwrap_or(0.0))
            })
        })
        .collect()
}

/// The tolerance of the engine's value against `reference`: half a binary16 step at the
/// reference, plus 4 float32 steps of the magnitude of its terms (the reference over absolute
/// values), which covers the float32 fold of the gains and biases and the rounding to float32
/// before binary16.
#[must_use]
pub fn tolerance(x: &[f32], cartridges: &[(&[f32], &[f32])]) -> Vec<f64> {
    let abs: Vec<(Vec<f32>, Vec<f32>)> = cartridges.iter().map(|(g, b)| (g.iter().map(|v| v.abs()).collect(), b.iter().map(|v| v.abs()).collect())).collect();
    let refs: Vec<(&[f32], &[f32])> = abs.iter().map(|(g, b)| (&g[..], &b[..])).collect();
    let ax: Vec<f32> = x.iter().map(|v| v.abs()).collect();
    let want = reference(x, cartridges);
    reference(&ax, &refs).iter().zip(&want).map(|(m, w)| 0.5 * ulp16(*w) + m * 4.0 * f64::from(f32::EPSILON)).collect()
}

/// The spacing of binary16 values at `x` (one unit in the last place; subnormal spacing below).
#[must_use]
pub fn ulp16(x: f64) -> f64 {
    let a = x.abs();
    if a < 6.103_515_625e-5 {
        return 5.960_464_477_539_063e-8;
    }
    2.0_f64.powi(a.log2().floor() as i32 - 10)
}

/// Self-check of the sockets on synthetic activations, on the processor and on the card when one
/// is present: an identity cartridge returns every value bit for bit (negative zero included); a known
/// cartridge equals the float64 reference within half a binary16 step plus the float32 fold; two cartridges
/// stacked equal the reference of their composition in both orders, and the orders differ; the
/// processor and the card give the same bits; unplugging leaves the rack empty.
pub(crate) fn check() -> Result<(), String> {
    let width = 96_usize;
    let rows = 7_usize;
    let mut seed = 0x9e37_79b9_u64;
    let mut rnd = |lo: f32, hi: f32| -> f32 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        #[expect(clippy::cast_precision_loss, reason = "a test value")]
        let u = (seed % 1_000_000) as f32 / 1_000_000.0;
        lo + (hi - lo) * u
    };
    let mut x: Vec<f32> = (0..rows * width).map(|_| crate::half::r16(rnd(-40.0, 40.0))).collect();
    if let Some(v) = x.get_mut(3) {
        *v = -0.0;
    }
    if let Some(v) = x.get_mut(5) {
        *v = 0.0;
    }
    let ga: Vec<f32> = (0..width).map(|_| rnd(0.5, 1.5)).collect();
    let ba: Vec<f32> = (0..width).map(|_| rnd(-0.3, 0.3)).collect();
    let gb: Vec<f32> = (0..width).map(|_| rnd(0.7, 1.3)).collect();
    let bb: Vec<f32> = (0..width).map(|_| rnd(-0.5, 0.5)).collect();
    let id = CartridgeFilter { cartridge: "identity".into(), parts: vec![Part { target: "t".into(), gain: vec![1.0; width], bias: vec![0.0; width] }] };
    let a = CartridgeFilter { cartridge: "a".into(), parts: vec![Part { target: "t".into(), gain: ga.clone(), bias: ba.clone() }] };
    let b = CartridgeFilter { cartridge: "b".into(), parts: vec![Part { target: "t".into(), gain: gb.clone(), bias: bb.clone() }] };
    let gpu = match nmffi::Gpu::open(0) {
        Ok(g) => Some(Ops { gpu: g }),
        Err(e) if e.contains("nvcuda.dll not found") || e.contains(nmffi::cuda::NO_USABLE) => None,
        Err(e) => return Err(e),
    };
    let run = |cartridges: &[CartridgeFilter]| -> Result<(Vec<f32>, Option<Vec<f32>>), String> {
        let p = build("m", cartridges)?;
        let f = p.nodes.get("t").ok_or("nmengn.filter: no point t")?;
        let mut c = x.clone();
        apply_cpu(&mut c, f)?;
        let g = match &gpu {
            Some(o) => Some(o.download(&apply_gpu(o, &o.upload_f32(&x, vec![rows, width])?, f)?)?),
            None => None,
        };
        Ok((c, g))
    };
    let bits = |u: &[f32], v: &[f32]| u.len() == v.len() && u.iter().zip(v).all(|(p, q)| p.to_bits() == q.to_bits());
    let (ic, ig) = run(std::slice::from_ref(&id))?;
    if !bits(&ic, &x) || ig.as_ref().is_some_and(|g| !bits(g, &x)) {
        return Err("nmengn.filter: an identity cartridge changed a value".into());
    }
    let within = |got: &[f32], refs: &[(&[f32], &[f32])]| -> Option<String> {
        let (want, tol) = (reference(&x, refs), tolerance(&x, refs));
        got.iter().zip(want.iter().zip(&tol)).enumerate().find_map(|(i, (g, (w, t)))| {
            ((f64::from(*g) - w).abs() > *t).then(|| format!("value {i}: {g} against {w} (tolerance {t:e})"))
        })
    };
    let (pa, pb): (Pair<'_>, Pair<'_>) = ((&ga, &ba), (&gb, &bb));
    for (cartridges, refs, name) in
        [(vec![a.clone()], vec![pa], "a"), (vec![a.clone(), b.clone()], vec![pa, pb], "a then b"), (vec![b.clone(), a.clone()], vec![pb, pa], "b then a")]
    {
        let (c, g) = run(&cartridges)?;
        if let Some(m) = within(&c, &refs) {
            return Err(format!("nmengn.filter: processor, {name}: {m}"));
        }
        if let Some(g) = g
            && !bits(&g, &c)
        {
            return Err(format!("nmengn.filter: {name}: card and processor differ"));
        }
    }
    if run(&[a.clone(), b.clone()])?.0 == run(&[b, a])?.0 {
        return Err("nmengn.filter: the two orders gave the same result".into());
    }
    if plug("nmengn-filter-check", std::slice::from_ref(&id)).is_err() || for_pop("weight-x-nmengn-filter-check").is_none() {
        return Err("nmengn.filter: plugged cartridge not found by its MicroModel".into());
    }
    if !unplug("nmengn-filter-check") || for_pop("weight-x-nmengn-filter-check").is_some() {
        return Err("nmengn.filter: unplug left a filter".into());
    }
    if names_model("weight-opnwgt-basewt-qwen3-reranker-0.6b", "qwen3-0.6b") || !names_model("weight-opnwgt-basewt-qwen3-0.6b", "qwen3-0.6b") {
        return Err("nmengn.filter: MicroModel names matched wrongly".into());
    }
    Ok(())
}

/// One forward of `m` over `ids` on `dev`, returning each target's values.
fn values_of(m: &crate::Model, dev: crate::Device, ids: &[i32], targets: &[(String, String)]) -> Result<Vec<Vec<f32>>, String> {
    let out = m.forward(dev, ids, targets, &crate::RunOpts::default()).map_err(|e| e.to_string())?;
    targets
        .iter()
        .map(|t| {
            out.get(t)
                .ok_or_else(|| format!("no {}.{}", t.0, t.1))
                .and_then(|a| crate::model::values(a).map(<[f32]>::to_vec).map_err(|e| e.to_string()))
        })
        .collect()
}

/// Section check of the sockets on the released Qwen3-Reranker-0.6B, on the processor and on the
/// card: a zero-effect cartridge at four points (an attention value, two gated feed-forwards, the
/// final norm) gives bit-identical outputs; a known synthetic cartridge equals a float64 reference
/// of the same arithmetic, at the socket (a middle layer's gated feed-forward against its
/// unfiltered value) and at the score (the final norm's filter carried through the head's yes and
/// no rows); two cartridges stacked equal the reference of their composition in both orders; the
/// rack and a model's own filters give the same bits; unplugging returns the base bit for bit.
/// Also measures the cost per plugged cartridge. Report:
/// `tests/section-checks/nmengn-filter-model/report.txt`.
pub(crate) fn check_model() -> Result<(), String> {
    use std::fmt::Write as _;
    let mut rep = String::from("nmengn.filter.model: cartridge filters on qwen3-reranker-0.6b\n");
    let r = model_parity(&mut rep);
    let _ = writeln!(rep, "result: {}", r.as_ref().map_or_else(|e| format!("FAIL {e}"), |()| "ok".to_owned()));
    let dir = nmtype::check_dir("nmengn-filter-model").map_err(|e| e.to_string())?;
    std::fs::write(dir.join("report.txt"), rep).map_err(|e| e.to_string())?;
    r
}

/// Checks `got` against the float64 reference `want` value by value; returns the worst error in
/// binary16 steps.
fn socket_within(got: &[f32], x: &[f32], refs: &[(&[f32], &[f32])], what: &str) -> Result<f64, String> {
    let (want, tol) = (reference(x, refs), tolerance(x, refs));
    let mut worst = 0.0_f64;
    for ((g, w), t) in got.iter().zip(&want).zip(&tol) {
        let e = (f64::from(*g) - w).abs();
        if e > *t {
            return Err(format!("{what}: socket value {g} against reference {w}"));
        }
        worst = worst.max(e / ulp16(*w));
    }
    Ok(worst)
}

#[expect(clippy::too_many_lines, reason = "one check, its cases in order")]
fn model_parity(rep: &mut String) -> Result<(), String> {
    use crate::{Assembly, Device, Model};
    use std::fmt::Write as _;
    const MODEL: &str = "qwen3-reranker-0.6b";
    // The check's sockets are measured on the bare MicroModel: the automatic job filters (on by
    // default for the reranker since 2026-11-24) would stack under them, outside the reference.
    let _bare = crate::model::JobFiltersHeldOff::new();
    let home = nmtype::nano_root();
    let pop = crate::model::Population::open(&home, MODEL, true, true).map_err(|e| e.to_string())?;
    let d = pop.dims;
    let m = Model::bind(&pop, Assembly::load(&home.join("designs/assemblies"), "qwen3-reranker").map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?
        .label_rows();
    let head = m.bound.iter().find(|b| b.node.op == "head").ok_or("no head")?;
    let lin = head.lin.as_ref().ok_or("no head weights")?;
    let (ry, rn) = (head.rows.first().copied().ok_or("no label rows")?, head.rows.get(1).copied().ok_or("no label rows")?);
    let (mut wy, mut wn) = (vec![0.0_f32; lin.k], vec![0.0_f32; lin.k]);
    lin.row(ry, &mut wy);
    lin.row(rn, &mut wn);
    let dw: Vec<f64> = wy.iter().zip(&wn).map(|(a, b)| f64::from(*a) - f64::from(*b)).collect();
    let pairs = [
        (
            "Where is the function that parses a configuration file?",
            "fn parse_config(path: &Path) -> Result<Config, Error> {\n    let text = std::fs::read_to_string(path)?;\n    toml::from_str(&text)\n}",
        ),
        ("How is the ring buffer's head advanced?", "pub struct Ring { items: Vec<u32>, head: usize }\nimpl Ring { pub fn push(&mut self, x: u32) { self.head += 1; } }"),
    ];
    let ids: Vec<Vec<i32>> = pairs
        .iter()
        .map(|(q, doc)| pop.tokenizer.encode(&crate::roles::card_prompt(crate::roles::CODE_INSTRUCTION, q, doc)).map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    let nid = crate::assembly::nid;
    let targets = [(m.role_node("head").map_err(|e| e.to_string())?, "logits".to_owned()), (nid("final_norm"), "y".to_owned()), (nid("L13.swigate"), "y".to_owned())];
    let mut seed = 0x517c_c1b7_2722_0a95_u64;
    let mut rnd = |n: usize, lo: f32, hi: f32| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                #[expect(clippy::cast_precision_loss, reason = "a test value")]
                let u = (seed % 1_000_000) as f32 / 1_000_000.0;
                lo + (hi - lo) * u
            })
            .collect()
    };
    let part = |t: &str, g: Vec<f32>, b: Vec<f32>| Part { target: t.into(), gain: g, bias: b };
    let one = |name: &str, parts: Vec<Part>| CartridgeFilter { cartridge: name.into(), parts };
    let kvw = d.kv_heads * d.head_dim;
    let zero = one(
        "zero",
        vec![
            part("L5.v", vec![1.0; kvw], vec![0.0; kvw]),
            part("L13.swigate", vec![1.0; d.inter], vec![0.0; d.inter]),
            part("L27.swigate", vec![1.0; d.inter], vec![0.0; d.inter]),
            part("final_norm", vec![1.0; d.hidden], vec![0.0; d.hidden]),
        ],
    );
    let (na_g, na_b, nb_g, nb_b) = (rnd(d.hidden, 0.8, 1.2), rnd(d.hidden, -0.05, 0.05), rnd(d.hidden, 0.9, 1.1), rnd(d.hidden, -0.05, 0.05));
    let (ma_g, ma_b, mb_g, mb_b) = (rnd(d.inter, 0.7, 1.3), rnd(d.inter, -0.02, 0.02), rnd(d.inter, 0.8, 1.2), rnd(d.inter, -0.02, 0.02));
    let norm_a = one("norm-a", vec![part("final_norm", na_g.clone(), na_b.clone())]);
    let norm_b = one("norm-b", vec![part("final_norm", nb_g.clone(), nb_b.clone())]);
    let mid_a = one("mid-a", vec![part("L13.swigate", ma_g.clone(), ma_b.clone())]);
    let mid_b = one("mid-b", vec![part("L13.swigate", mb_g.clone(), mb_b.clone())]);
    let local = |cs: &[CartridgeFilter]| -> Result<(), String> {
        m.set_filters(Some(Arc::new(build(MODEL, cs)?)));
        Ok(())
    };
    let bits = |u: &[f32], v: &[f32]| u.len() == v.len() && u.iter().zip(v).all(|(p, q)| p.to_bits() == q.to_bits());
    let same3 = |a: &[Vec<f32>], b: &[Vec<f32>]| a.iter().zip(b).all(|(x, y)| bits(x, y));
    let z_of = |v: &[Vec<f32>]| v.first().map_or(0.0, |l| f64::from(l.first().copied().unwrap_or(0.0)) - f64::from(l.get(1).copied().unwrap_or(0.0)));
    let _ = writeln!(
        rep,
        "tolerances: socket values within half a binary16 step of the float64 reference plus 4 float32 steps of the magnitude of its terms (the fold and the rounding to float32 first); score z = logit(yes) - logit(no) within the sum over channels of |w_yes - w_no| times half a binary16 step of the reference value, plus 1e-4 of the sum of |terms| (the head's float32 sums)"
    );
    if for_pop(&pop.card.id).is_some() {
        return Err("a cartridge is plugged on the rack for this MicroModel; the check needs it empty".into());
    }
    let devices: Vec<Device> = if pop.gpu.is_some() { vec![Device::Cpu, Device::Gpu] } else { vec![Device::Cpu] };
    for dev in devices {
        let dn = dev.name();
        for (pi, p) in ids.iter().enumerate() {
            m.set_filters(None);
            let base = values_of(&m, dev, p, &targets)?;
            let (b_norm, b_mid) = (base.get(1).ok_or("no norm")?, base.get(2).ok_or("no mid")?);
            local(std::slice::from_ref(&zero))?;
            if !same3(&values_of(&m, dev, p, &targets)?, &base) {
                return Err(format!("{dn} pair {pi}: a zero-effect cartridge changed an output"));
            }
            let last = b_norm.get(b_norm.len() - d.hidden..).ok_or("short norm output")?;
            for (cs, refs, name) in [
                (vec![norm_a.clone()], vec![(&na_g[..], &na_b[..])], "norm-a"),
                (vec![norm_a.clone(), norm_b.clone()], vec![(&na_g[..], &na_b[..]), (&nb_g[..], &nb_b[..])], "norm-a then norm-b"),
                (vec![norm_b.clone(), norm_a.clone()], vec![(&nb_g[..], &nb_b[..]), (&na_g[..], &na_b[..])], "norm-b then norm-a"),
            ] {
                local(&cs)?;
                let got = values_of(&m, dev, p, &targets)?;
                if !got.get(2).is_some_and(|g| bits(g, b_mid)) {
                    return Err(format!("{dn} pair {pi} {name}: a layer before the filter changed"));
                }
                let want = reference(last, &refs);
                let g_norm = got.get(1).ok_or("no norm")?;
                let worst = socket_within(g_norm.get(g_norm.len() - d.hidden..).unwrap_or(&[]), last, &refs, &format!("{dn} pair {pi} {name}"))?;
                let zr: f64 = want.iter().zip(&dw).map(|(n, w)| n * w).sum();
                let tol: f64 = want.iter().zip(&dw).map(|(n, w)| w.abs() * 0.5 * ulp16(*n)).sum::<f64>()
                    + 1e-4 * want.iter().zip(&dw).map(|(n, w)| (n * w).abs()).sum::<f64>();
                let ze = z_of(&got);
                if (ze - zr).abs() > tol {
                    return Err(format!("{dn} pair {pi} {name}: score {ze:.5} against reference {zr:.5} (tolerance {tol:.5})"));
                }
                let _ = writeln!(
                    rep,
                    "{dn} pair {pi} {name}: socket worst {worst:.3} binary16 steps; score {ze:.5}, reference {zr:.5}, base {:.5}; difference {:.2e}, tolerance {tol:.2e}",
                    z_of(&base),
                    (ze - zr).abs()
                );
            }
            for (cs, refs, name) in [
                (vec![mid_a.clone()], vec![(&ma_g[..], &ma_b[..])], "mid-a"),
                (vec![mid_a.clone(), mid_b.clone()], vec![(&ma_g[..], &ma_b[..]), (&mb_g[..], &mb_b[..])], "mid-a then mid-b"),
                (vec![mid_b.clone(), mid_a.clone()], vec![(&mb_g[..], &mb_b[..]), (&ma_g[..], &ma_b[..])], "mid-b then mid-a"),
            ] {
                local(&cs)?;
                let got = values_of(&m, dev, p, &targets)?;
                let worst = socket_within(got.get(2).ok_or("no mid")?, b_mid, &refs, &format!("{dn} pair {pi} {name}"))?;
                let _ = writeln!(rep, "{dn} pair {pi} {name}: {} socket values, worst {worst:.3} binary16 steps; score {:.5}, base {:.5}", b_mid.len(), z_of(&got), z_of(&base));
            }
            local(&[norm_a.clone(), mid_b.clone()])?;
            let own = values_of(&m, dev, p, &targets)?;
            m.set_filters(None);
            plug(MODEL, &[norm_a.clone(), mid_b.clone()])?;
            let racked = values_of(&m, dev, p, &targets);
            unplug(MODEL);
            if !same3(&own, &racked?) {
                return Err(format!("{dn} pair {pi}: the rack and the model's own filters differ"));
            }
            if !same3(&values_of(&m, dev, p, &targets)?, &base) {
                return Err(format!("{dn} pair {pi}: unplugging did not return the base"));
            }
            let _ = writeln!(rep, "{dn} pair {pi}: zero-effect cartridge bit-identical; rack equals own filters; unplugged equals base bit for bit");
        }
    }
    m.set_filters(None);
    cost(&m, rep)
}

/// Time of one forward of a reranker-sized input with 0, 1, 2 and 4 cartridges plugged, each
/// filtering every layer's gated feed-forward and the final norm, on the card (and the processor,
/// once), and the card bytes the filters hold.
fn cost(m: &crate::Model, rep: &mut String) -> Result<(), String> {
    use crate::Device;
    use std::fmt::Write as _;
    let d = m.pop.dims;
    let text: String = (0..60).map(|i| format!("let value_{i} = compute(input, {i});\n")).collect();
    let mut ids = m
        .pop
        .tokenizer
        .encode(&crate::roles::card_prompt(crate::roles::CODE_INSTRUCTION, "Where is the value computed?", &text))
        .map_err(|e| e.to_string())?;
    ids.truncate(512);
    let target = vec![(m.role_node("label").map_err(|e| e.to_string())?, "y".to_owned())];
    let full = |name: &str, g: f32| CartridgeFilter {
        cartridge: name.into(),
        parts: (0..d.layers)
            .map(|l| Part { target: format!("L{l}.swigate"), gain: vec![g; d.inter], bias: vec![0.001; d.inter] })
            .chain(std::iter::once(Part { target: "final_norm".into(), gain: vec![g; d.hidden], bias: vec![0.0; d.hidden] }))
            .collect(),
    };
    let devices: Vec<Device> = if m.pop.gpu.is_some() { vec![Device::Gpu, Device::Cpu] } else { vec![Device::Cpu] };
    let _ = writeln!(
        rep,
        "cost: {} tokens; each cartridge filters {} points ({} values, {} bytes as float32)",
        ids.len(),
        d.layers + 1,
        (d.layers * d.inter + d.hidden) * 2,
        (d.layers * d.inter + d.hidden) * 8
    );
    for dev in devices {
        let reps = if dev == Device::Gpu { 7 } else { 1 };
        for n in [0_usize, 1, 2, 4] {
            #[expect(clippy::cast_precision_loss, reason = "a test gain")]
            let cs: Vec<CartridgeFilter> = (0..n).map(|i| full(&format!("c{i}"), 1.0 + 0.001 * i as f32)).collect();
            let p = Arc::new(build(&m.pop.card.id, &cs)?);
            m.set_filters(Some(Arc::clone(&p)));
            let _ = values_of(m, dev, &ids, &target)?;
            let mut ts = Vec::new();
            for _ in 0..reps {
                let t = std::time::Instant::now();
                let _ = values_of(m, dev, &ids, &target)?;
                ts.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            ts.sort_by(f64::total_cmp);
            let med = ts.get(ts.len() / 2).copied().unwrap_or(0.0);
            let _ = writeln!(rep, "cost {} {n} cartridges: median {med:.2} ms per forward; card bytes held by the filters {}", dev.name(), p.device_bytes());
        }
    }
    m.set_filters(None);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-11-26 (v0.1.0): a job filter's built-on base against the serving base: the same key
    /// matches (also as a population's `cartridge <key>` label), another version does not, and a
    /// base not recorded on either side is unknown.
    #[test]
    fn job_filter_base_match_matching_mismatching_not_recorded() {
        assert_eq!(job_filter_base_match(Some("writer-base@4"), Some("writer-base@4")), Some(true));
        assert_eq!(job_filter_base_match(Some("writer-base@4"), Some("cartridge writer-base@4")), Some(true));
        assert_eq!(job_filter_base_match(Some("open:qwen3-1.7b"), Some(crate::model::OPEN_WEIGHTS)), Some(true));
        assert_eq!(job_filter_base_match(Some("writer-base@3"), Some("writer-base@4")), Some(false));
        assert_eq!(job_filter_base_match(Some("writer-base@4"), Some("open weights")), Some(false));
        assert_eq!(job_filter_base_match(None, Some("writer-base@4")), None);
        assert_eq!(job_filter_base_match(Some(""), Some("writer-base@4")), None);
        assert_eq!(job_filter_base_match(Some("writer-base@4"), None), None);

        let d = nmtype::nano_root().join("scratch").join("unit-tests").join(format!("nmengn-jf-base-{}-{}", std::process::id(), nmtype::now_ms()));
        std::fs::create_dir_all(&d).expect("test dir");
        assert_eq!(job_filter_built_on(&d, "job-filter-write.json"), None, "no lineage file: not recorded");
        std::fs::write(
            d.join("lineage.json"),
            r#"{"maps":{"job-filter-write.json":{"made_on":"writer-base@4"},"job-filter-old.json":{"made_on":""}}}"#,
        )
        .expect("lineage");
        assert_eq!(job_filter_built_on(&d, "job-filter-write.json").as_deref(), Some("writer-base@4"));
        assert_eq!(job_filter_built_on(&d, "job-filter-old.json"), None, "an empty made_on is not recorded");
        assert_eq!(job_filter_built_on(&d, "job-filter-none.json"), None);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn stack_plugged_composes_order_and_unions_disjoint_nodes() {
        let base_c = CartridgeFilter {
            cartridge: "base-f".into(),
            parts: vec![
                Part { target: "L0.swigate".into(), gain: vec![1.2, 0.8], bias: vec![0.1, -0.2] },
                Part { target: "L1.swigate".into(), gain: vec![0.9, 1.1], bias: vec![0.0, 0.05] },
            ],
        };
        let extra_c = CartridgeFilter {
            cartridge: "hop-job".into(),
            parts: vec![
                Part { target: "L0.swigate".into(), gain: vec![0.5, 1.5], bias: vec![0.25, 0.3] },
                Part { target: "final_norm".into(), gain: vec![1.05, 0.95], bias: vec![-0.01, 0.02] },
            ],
        };
        let base_p = build("qwen3-0.6b", &[base_c]).unwrap_or_default();
        let extra_p = build("qwen3-0.6b", &[extra_c]).unwrap_or_default();
        let stacked = stack_plugged(&base_p, &extra_p).unwrap_or_else(|_| Arc::new(Plugged::default()));

        assert_eq!(stacked.cartridges, vec!["base-f", "hop-job"]);
        assert_eq!(stacked.nodes.len(), 3);
        let l0 = stacked.nodes.get("L0.swigate").expect("L0.swigate present");
        assert_eq!(l0.cartridges, vec!["base-f", "hop-job"]);
        // (x * 1.2 + 0.1) * 0.5 + 0.25 = x * 0.6 + 0.30
        assert!((l0.gain[0] - 0.6).abs() < 1e-6);
        assert!((l0.bias[0] - 0.30).abs() < 1e-6);
        // (x * 0.8 - 0.2) * 1.5 + 0.3 = x * 1.2 + 0.0
        assert!((l0.gain[1] - 1.2).abs() < 1e-6);
        assert!(l0.bias[1].abs() < 1e-6);
        assert!(stacked.nodes.contains_key("L1.swigate"));
        assert!(stacked.nodes.contains_key("final_norm"));
    }

    #[test]
    fn parse_job_filter_respects_enabled_contract_and_model() {
        let enabled_json = br#"{
          "contract": "nmengn-sockets/1",
          "cartridge": "reranker-code-slice@1",
          "model": "qwen3-reranker-0.6b",
          "enabled": true,
          "parts": [{"target": "L2.swigate", "gain": [1.05, 0.95], "bias": [0.01, -0.02]}]
        }"#;
        let parsed = parse_job_filter_json(enabled_json, "qwen3-reranker-0.6b").expect("valid enabled filter");
        let cf = parsed.expect("enabled filter returns Some");
        assert_eq!(cf.cartridge, "reranker-code-slice@1");
        assert_eq!(cf.parts.len(), 1);
        assert_eq!(cf.parts[0].target, "L2.swigate");

        let disabled_json = br#"{
          "contract": "nmengn-sockets/1",
          "cartridge": "reranker-code-slice@1",
          "model": "qwen3-reranker-0.6b",
          "enabled": false,
          "parts": [{"target": "L2.swigate", "gain": [1.05, 0.95], "bias": [0.01, -0.02]}]
        }"#;
        assert!(parse_job_filter_json(disabled_json, "qwen3-reranker-0.6b").expect("valid disabled filter").is_none());

        assert!(parse_job_filter_json(enabled_json, "qwen3-1.7b").is_err(), "model mismatch rejected");
    }
}
