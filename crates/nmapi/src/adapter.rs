//! Fitted adapter cartridges of the user's project: the loading path (Luis Jake Gabriel III,
//! 2026-11-18 05:56 UTC: a piece fitted once from the user's material is a fitted adapter
//! cartridge, not continuous learning; 08:38 UTC item 2: the embedder's per-project lens first,
//! then the judge's code adapter, closed form first). Fitting lives in `adapter_fit`.
//!
//! Two adapters, each its own labelled, versioned file on the adapter shelf
//! (`library/cartridges/adapter/<id>/<id>-v<n>.cartridge`, the library's one-file form: mark,
//! label, parameters checked against the label's `parameters_sha256`), loaded beside the base
//! cartridge it was fitted on and never merged into it:
//! - `embedder-lens-<project>`: a map on the request's vector only (orchestration study 9.2 D: an
//!   adapter that changed item vectors would force every store to be re-embedded). Two parts: the
//!   material's strongest shared directions flattened (`flat` `[k, d]` with shrink `beta` `[k]`; the
//!   two-sided flattening folds into one matrix on the request's side), then the identity plus a
//!   low-rank map (`down` `[r, d]`, `up` `[d, r]`) fitted by reduced-rank ridge from the project's
//!   own questions (commit messages, doc comments hidden from their definitions) to their items.
//!   `q' = normalize(qf + up (down qf))`, `qf = q - sum beta_i u_i (u_i . q)`.
//! - `judge-code-<project>`: a read-out at the judge's final norm (`w` `[d]`, with `alpha` and
//!   `bias` in the label's notes), fitted on the difference of the judge's states between the true
//!   item and the rival that beats it (study 9.2 A1). Added to the judge's label log-odds only
//!   inside a close call whose candidates are all code (C102: the judge is below the embedder on
//!   code); prose and mixed calls are untouched.
//!
//! Each adapter carries the values key of the MicroModel it was fitted on (`notes.values`,
//! `neural::parameters_key`): it loads only on that same base (open weights, or a base cartridge
//! with its compensation), otherwise it is left out with the reason. Pulled (`pull`, its switch
//! off, or its file gone) the base's outputs come back bit for bit: the hooks return their input
//! untouched.
//!
//! Switches: the lens and the judge's read-out load and act by default once a fitted file passed its
//! own control (`NANO_ADAPTER_LENS=off`, `NANO_ADAPTER_JUDGE=off` for diagnosis); the fit runs at
//! idle by default (`adapter_fit`; `NANO_ADAPTER_FIT=off`). Gavriel Technologies; architecture by
//! Luis Jake Gabriel III.
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmapi.adapter", FaultKind::Interface, m)
}

/// The lens's id before the project's mark.
pub const LENS_PREFIX: &str = "embedder-lens";
/// The judge's code adapter's id before the project's mark.
pub const JUDGE_PREFIX: &str = "judge-code";
/// The embedder's open weights the lens is fitted on.
pub const EMBEDDER: &str = "qwen3-embedding-0.6b";
/// The judge's open weights the read-out is fitted on.
pub const JUDGE: &str = "qwen3-reranker-0.6b";
/// The label's `kind` of the lens.
pub const LENS_KIND: &str = "lens";
/// The label's `kind` of the read-out.
pub const READOUT_KIND: &str = "readout";
/// The judge's log-odds are clamped where the close-call arm clamps them.
const Z_CAP: f32 = 13.815_51;

/// One part of an adapter file: socket name, shape, numbers.
pub type Part = (String, Vec<usize>, Vec<f32>);

/// Whether the lens loads and acts: on by default since 2026-11-19 (it serves only once a fitted
/// file passed its own control); `NANO_ADAPTER_LENS=off`.
#[must_use]
pub fn lens_on() -> bool {
    !std::env::var("NANO_ADAPTER_LENS").is_ok_and(|v| v == "off")
}
/// Whether the judge's code adapter loads and acts: on by default since 2026-11-19 (as the lens);
/// `NANO_ADAPTER_JUDGE=off`.
#[must_use]
pub fn judge_on() -> bool {
    !std::env::var("NANO_ADAPTER_JUDGE").is_ok_and(|v| v == "off")
}
/// Whether the adapters are fitted at idle: on by default since 2026-11-19; `NANO_ADAPTER_FIT=off`.
#[must_use]
pub fn fit_on() -> bool {
    !std::env::var("NANO_ADAPTER_FIT").is_ok_and(|v| v == "off")
}


/// The project's mark in an adapter's id: the first 12 hex digits of the SHA-256 of the material's
/// root as one canonical path (forward slashes, lower case), so each project has its own adapters.
#[must_use]
pub fn project_mark(root: &Path) -> String {
    let p = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let t = p.display().to_string().replace('\\', "/").trim_start_matches("//?/").to_lowercase();
    nmhash::sha256_bytes(t.as_bytes()).chars().take(12).collect()
}

/// An adapter's id for the project at `root`.
#[must_use]
pub fn adapter_id(prefix: &str, root: &Path) -> String {
    format!("{prefix}-{}", project_mark(root))
}

/// The file of one adapter version on the shelf (`nmlearn::library::shelf_path`:
/// `adapter/<id>/<id>-v<n>.cartridge`).
#[must_use]
pub fn shelf_file(home: &Path, id: &str, version: u64) -> PathBuf {
    home.join("library/cartridges").join(nmlearn::library::shelf_path(nmlearn::library::ADAPTER, "", "", "", id, version))
}

/// Every version file of `id` on the shelf, oldest first: (version, file).
#[must_use]
pub fn versions(home: &Path, id: &str) -> Vec<(u64, PathBuf)> {
    let Some(dir) = shelf_file(home, id, 1).parent().map(Path::to_path_buf) else { return Vec::new() };
    let pre = format!("{id}-v");
    let post = format!(".{}", nmlearn::library::CARTRIDGE_EXT);
    let mut out: Vec<(u64, PathBuf)> = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .filter_map(|e| {
                    let name = e.file_name().to_string_lossy().into_owned();
                    let v = name.strip_prefix(&pre)?.strip_suffix(&post)?.parse::<u64>().ok()?;
                    Some((v, e.path()))
                })
                .collect()
        })
        .unwrap_or_default();
    out.sort_by_key(|(v, _)| *v);
    out
}

/// The per-project lens on the embedder's request vectors.
#[derive(Clone, Debug, PartialEq)]
pub struct Lens {
    /// `id@version`.
    pub key: String,
    /// The embedder's values key it was fitted on.
    pub values: String,
    pub dim: usize,
    /// Flattened directions, `[k, d]` row-major (unit vectors), and their shrink.
    pub flat: Vec<f32>,
    pub beta: Vec<f32>,
    /// Low-rank map: `down` `[r, d]`, `up` `[d, r]`.
    pub rank: usize,
    pub down: Vec<f32>,
    pub up: Vec<f32>,
}
impl Lens {
    /// The flattening alone (no map), normalised: a piece's side of the lens (`lens_doc`).
    #[must_use]
    pub fn apply_flat(&self, v: &[f32]) -> Vec<f32> {
        let d = self.dim;
        if v.len() != d || d == 0 {
            return v.to_vec();
        }
        let mut out = v.to_vec();
        for (i, b) in self.beta.iter().enumerate() {
            let Some(u) = self.flat.get(i * d..(i + 1) * d) else { break };
            let c = b * dot(u, v);
            for (x, ui) in out.iter_mut().zip(u) {
                *x -= c * ui;
            }
        }
        let n = dot(&out, &out).sqrt();
        if n > 0.0 && n.is_finite() {
            out.iter_mut().for_each(|x| *x /= n);
        }
        out
    }
    /// The lens applied to one request vector, unit length again. A vector of another width is
    /// returned untouched.
    #[must_use]
    pub fn apply(&self, q: &[f32]) -> Vec<f32> {
        let d = self.dim;
        if q.len() != d || d == 0 {
            return q.to_vec();
        }
        let mut qf = q.to_vec();
        for (i, b) in self.beta.iter().enumerate() {
            let Some(u) = self.flat.get(i * d..(i + 1) * d) else { break };
            let c = b * dot(u, q);
            for (x, ui) in qf.iter_mut().zip(u) {
                *x -= c * ui;
            }
        }
        let mut out = qf.clone();
        if self.rank > 0 {
            let h: Vec<f32> = (0..self.rank).map(|j| self.down.get(j * d..(j + 1) * d).map_or(0.0, |row| dot(row, &qf))).collect();
            for (m, o) in out.iter_mut().enumerate() {
                if let Some(row) = self.up.get(m * self.rank..(m + 1) * self.rank) {
                    *o += dot(row, &h);
                }
            }
        }
        let n = dot(&out, &out).sqrt();
        if n > 0.0 && n.is_finite() {
            out.iter_mut().for_each(|x| *x /= n);
            out
        } else {
            q.to_vec()
        }
    }
}

/// The judge's code read-out.
#[derive(Clone, Debug, PartialEq)]
pub struct Readout {
    pub key: String,
    pub values: String,
    pub w: Vec<f32>,
    pub bias: f32,
    pub alpha: f32,
}
impl Readout {
    /// What it adds to one candidate's log-odds from the judge's final-norm state there.
    #[must_use]
    pub fn term(&self, h: &[f32]) -> f32 {
        if h.len() != self.w.len() {
            return 0.0;
        }
        self.alpha * (dot(&self.w, h) + self.bias)
    }
}

/// Inner product of two vectors (the shorter length).
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// One adapter's state: tried for which file, what loaded, and why not.
#[derive(Debug)]
struct Slot<T> {
    tried: Option<PathBuf>,
    loaded: Option<Arc<T>>,
    note: String,
}
impl<T> Slot<T> {
    const fn empty() -> Self {
        Self { tried: None, loaded: None, note: String::new() }
    }
}
static LENS: RwLock<Slot<Lens>> = RwLock::new(Slot::empty());
static READOUT: RwLock<Slot<Readout>> = RwLock::new(Slot::empty());
/// Serialises lazy loads.
static LOADING: Mutex<()> = Mutex::new(());

fn text(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default()
}
fn num(v: &Value, k: &str) -> Option<f64> {
    match v.get(k) {
        Ok(Value::Number(s)) => s.parse().ok(),
        _ => None,
    }
}
fn shape(p: &Value) -> Vec<usize> {
    p.get("shape").and_then(Value::array).unwrap_or_default().iter().filter_map(|x| x.integer().ok().and_then(|v| usize::try_from(v).ok())).collect()
}

/// An adapter file's label and its parts as f32 (parts are stored at 16 bits, `F16`, or as `F32`),
/// the parameters checked against the label.
/// # Errors
/// Unreadable, not a cartridge, parameters that do not match, a part outside the parameters.
pub fn read_file(file: &Path) -> Result<(Value, Vec<Part>), Fault> {
    let bytes = std::fs::read(file).map_err(|e| bad(format!("{}: {e}", file.display())))?;
    read_bytes(&bytes).map_err(|e| bad(format!("{}: {}", file.display(), e.message)))
}

/// `read_file` on the file's bytes.
/// # Errors
/// As `read_file`.
pub fn read_bytes(bytes: &[u8]) -> Result<(Value, Vec<Part>), Fault> {
    let (label, body) = nmlearn::library::read_cartridge(bytes).map_err(|e| bad(format!("{e:?}")))?;
    if text(&label, "family") != nmlearn::library::ADAPTER {
        return Err(bad(format!("not an adapter for the cartridge library (family {})", text(&label, "family"))));
    }
    let mut parts = Vec::new();
    for p in label.get("parts").and_then(Value::array).unwrap_or_default() {
        let at = usize::try_from(p.get("at").and_then(Value::integer).unwrap_or(0)).unwrap_or(usize::MAX);
        let len = usize::try_from(p.get("bytes").and_then(Value::integer).unwrap_or(0)).unwrap_or(0);
        let raw = body.get(at..at.saturating_add(len)).ok_or_else(|| bad("a part lies outside the parameters"))?;
        let sh = shape(p);
        let count: usize = sh.iter().product();
        let xs: Vec<f32> = match text(p, "encoding").as_str() {
            "F16" if raw.len() == 2 * count => raw.as_chunks::<2>().0.iter().map(|c| nmengn::half::from16(u16::from_le_bytes(*c))).collect(),
            "F32" if raw.len() == 4 * count => raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
            e => return Err(bad(format!("{}: {e} part of {} bytes for shape {sh:?}", text(p, "socket"), raw.len()))),
        };
        parts.push((text(p, "socket"), sh, xs));
    }
    Ok((label, parts))
}

fn part<'a>(parts: &'a [Part], name: &str) -> Option<&'a Part> {
    parts.iter().find(|p| p.0 == name)
}

fn fitted_on(label: &Value, values: &str) -> Result<(String, String), Fault> {
    let notes = label.get("notes").cloned().unwrap_or(Value::Null);
    let fitted = text(&notes, "values");
    if fitted != values {
        return Err(bad(format!("fitted on {fitted}, the MicroModel runs on {values}")));
    }
    Ok((format!("{}@{}", text(label, "id"), label.get("version").and_then(Value::integer).unwrap_or(0)), fitted))
}

/// The lens in a file, checked against the embedder's values key.
/// # Errors
/// As `read_file`; a label of another kind or MicroModel; a values key that is not `values`;
/// parts of the wrong shapes.
pub fn lens_of(file: &Path, values: &str) -> Result<Lens, Fault> {
    let (label, mut parts) = read_file(file)?;
    overlaid(&label, &mut parts);
    lens_from(&label, &parts, values).map_err(|e| bad(format!("{}: {}", file.display(), e.message)))
}

/// The lens of a label and its parts (`read_bytes`), checked against `values`.
/// # Errors
/// As `lens_of`.
pub fn lens_from(label: &Value, parts: &[Part], values: &str) -> Result<Lens, Fault> {
    if text(label, "kind") != LENS_KIND || text(label, "model") != EMBEDDER {
        return Err(bad("not an embedder lens"));
    }
    let (key, fitted) = fitted_on(label, values)?;
    let (flat, beta) = match (part(parts, "flat"), part(parts, "beta")) {
        (Some(f), Some(b)) => (f.2.clone(), b.2.clone()),
        _ => (Vec::new(), Vec::new()),
    };
    let (rank, down, up) = match (part(parts, "down"), part(parts, "up")) {
        (Some(d), Some(u)) => (d.1.first().copied().unwrap_or(0), d.2.clone(), u.2.clone()),
        _ => (0, Vec::new(), Vec::new()),
    };
    let dim = part(parts, "down").and_then(|d| d.1.get(1).copied()).or_else(|| part(parts, "flat").and_then(|f| f.1.get(1).copied())).unwrap_or(0);
    if dim == 0 || flat.len() != beta.len() * dim || down.len() != rank * dim || up.len() != dim * rank {
        return Err(bad(format!("{key}: shapes do not fit (d {dim}, k {}, r {rank})", beta.len())));
    }
    Ok(Lens { key, values: fitted, dim, flat, beta, rank, down, up })
}

/// The judge's read-out in a file, checked against the judge's values key.
/// # Errors
/// As `lens_of`.
pub fn readout_of(file: &Path, values: &str) -> Result<Readout, Fault> {
    let (label, mut parts) = read_file(file)?;
    overlaid(&label, &mut parts);
    readout_from(&label, &parts, values).map_err(|e| bad(format!("{}: {}", file.display(), e.message)))
}

/// The read-out of a label and its parts, checked against `values`.
/// # Errors
/// As `readout_of`.
pub fn readout_from(label: &Value, parts: &[Part], values: &str) -> Result<Readout, Fault> {
    if text(label, "kind") != READOUT_KIND || text(label, "model") != JUDGE {
        return Err(bad("not a judge read-out"));
    }
    let (key, fitted) = fitted_on(label, values)?;
    let notes = label.get("notes").cloned().unwrap_or(Value::Null);
    let w = part(parts, "w").map(|p| p.2.clone()).ok_or_else(|| bad(format!("{key}: no read-out vector")))?;
    #[expect(clippy::cast_possible_truncation, reason = "a scale and an offset stored as text")]
    let (alpha, bias) = (num(&notes, "alpha").unwrap_or(0.0) as f32, num(&notes, "bias").unwrap_or(0.0) as f32);
    if w.is_empty() || !alpha.is_finite() || !bias.is_finite() {
        return Err(bad(format!("{key}: empty read-out or a non-finite scale")));
    }
    Ok(Readout { key, values: fitted, w, bias, alpha })
}

/// The version key (`id@version`) of an adapter file's label.
fn key_of(label: &Value) -> String {
    format!("{}@{}", text(label, "id"), label.get("version").and_then(Value::integer).unwrap_or(0))
}

/// Applies the adapter version's parameter overlay (`memory/overlays/`) to its parts as they load;
/// the file is never written.
fn overlaid(label: &Value, parts: &mut [Part]) {
    if let Some(ov) = nmengn::overlay::effective(&nmtype::nano_root(), &key_of(label)) {
        for (name, _, xs) in parts.iter_mut() {
            ov.apply(name, "F32", xs);
        }
    }
}

/// The key of an adapter file whose version is masked whole in its parameter overlay: it is left out
/// while it stays plugged (no earlier version takes its place).
fn masked_whole(file: &Path) -> Option<String> {
    let k = key_of(&nmlearn::library::label_of(file).ok()?);
    nmengn::overlay::effective(&nmtype::nano_root(), &k).is_some_and(|o| o.whole_masked()).then_some(k)
}

/// The overlay stamp each slot was loaded under; a change loads the file again at the next look.
static LENS_OVERLAY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static READOUT_OVERLAY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The shelf is looked at again at most this often (ms), so a call never lists a folder.
const LOOK_EVERY_MS: u64 = 30_000;
/// The last look per prefix: (when, newest file).
static LOOKED: Mutex<Vec<(String, u64, Option<PathBuf>)>> = Mutex::new(Vec::new());

/// The newest version file of `prefix` for the installed material, if any (the shelf listed at most
/// every `LOOK_EVERY_MS`).
fn newest(home: &Path, prefix: &str) -> Option<PathBuf> {
    let now = nmtype::now_ms();
    if let Ok(g) = LOOKED.lock()
        && let Some((_, at, f)) = g.iter().find(|(p, _, _)| p == prefix)
        && now.saturating_sub(*at) < LOOK_EVERY_MS
    {
        return f.clone();
    }
    let o = crate::tools::installed()?;
    let id = adapter_id(prefix, &o.sub().root);
    let f = versions(home, &id).pop().map(|(_, f)| f);
    if let Ok(mut g) = LOOKED.lock() {
        g.retain(|(p, _, _)| p != prefix);
        g.push((prefix.to_owned(), now, f.clone()));
    }
    f
}

/// Forgets the last look at the shelf (after a fit writes a new version).
pub fn look_again() {
    if let Ok(mut g) = LOOKED.lock() {
        g.clear();
    }
}

/// Whether the installed project's adapter `prefix` is plugged in and its compartment is on
/// (`plugs`, 2026-11-26: "plug in and unplug the learned units"). No installed project: true (no
/// adapter loads then anyway).
#[must_use]
pub fn plugged(home: &Path, prefix: &str) -> bool {
    let Some(o) = crate::tools::installed() else { return true };
    let root = &o.sub().root;
    crate::plugs::acts(home, &crate::plugs::compartment_of_root(home, root), &adapter_id(prefix, root))
}

/// Takes a seated adapter out when it is unplugged, so the base serves alone bit for bit and the
/// next call after it is plugged in again loads the file afresh. True when something was seated.
fn unseat<T>(slot: &RwLock<Slot<T>>) -> bool {
    let Ok(mut s) = slot.write() else { return false };
    if s.tried.is_none() && s.loaded.is_none() {
        return false;
    }
    *s = Slot { note: "unplugged".into(), ..Slot::empty() };
    true
}

/// The values key of a bound role model ("embedder" or "reranker").
fn bound_values(role: &str) -> Option<String> {
    crate::neural::bound_models().into_iter().find(|(r, _)| *r == role).map(|(_, m)| crate::neural::parameters_key(&m))
}

/// Seats a freshly fitted lens (after `adapter_fit` writes it); it acts only with its switch on.
pub fn seat_lens(l: Lens, file: PathBuf) {
    look_again();
    LENS_EPOCH.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    if let Ok(mut s) = LENS.write() {
        s.note = format!("lens {} seated on the embedder ({})", l.key, l.values);
        s.tried = Some(file);
        s.loaded = Some(Arc::new(l));
    }
}
/// Seats a freshly fitted read-out; it acts only with its switch on.
pub fn seat_readout(r: Readout, file: PathBuf) {
    look_again();
    if let Ok(mut s) = READOUT.write() {
        s.note = format!("judge code read-out {} seated ({})", r.key, r.values);
        s.tried = Some(file);
        s.loaded = Some(Arc::new(r));
    }
}

/// Pulls both adapters: the base serves alone again, bit for bit, until a file is tried afresh
/// (the next use, while the switch is on and the file is there).
pub fn pull() {
    if let Ok(mut g) = LENS.write() {
        *g = Slot { note: "pulled".into(), ..Slot::empty() };
    }
    LENS_EPOCH.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    if let Ok(mut g) = READOUT.write() {
        *g = Slot { note: "pulled".into(), ..Slot::empty() };
    }
}

/// The epoch of the lens in force: it moves whenever the lens is loaded, replaced or pulled, so
/// caches of lensed vectors (the meaning lane's content-free vectors) are made again.
static LENS_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The lens epoch (`LENS_EPOCH`).
#[must_use]
pub fn lens_epoch() -> u64 {
    LENS_EPOCH.load(std::sync::atomic::Ordering::SeqCst)
}

/// Moves the lens epoch when another request-side map changed (a distilled request lens seated
/// or taken out, `distil::seat_embedder`), so caches of lensed request vectors are made again.
pub fn lens_changed() {
    LENS_EPOCH.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

/// The lens in force, loading the newest file of the installed project on first use (and again
/// when a newer file appears). None when switched off, absent, or fitted on other values.
#[must_use]
pub fn lens() -> Option<Arc<Lens>> {
    if !lens_on() {
        return None;
    }
    let home = nmtype::nano_root();
    if !plugged(&home, LENS_PREFIX) {
        if unseat(&LENS) {
            LENS_EPOCH.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            eprintln!("adapter: lens unplugged; the embedder serves alone");
        }
        return None;
    }
    let file = newest(&home, LENS_PREFIX)?;
    let ov = nmengn::overlay::stamp_cached(&home);
    if let Ok(s) = LENS.read()
        && s.tried.as_ref() == Some(&file)
        && LENS_OVERLAY.load(std::sync::atomic::Ordering::SeqCst) == ov
    {
        return s.loaded.clone();
    }
    LENS_OVERLAY.store(ov, std::sync::atomic::Ordering::SeqCst);
    if let Some(k) = masked_whole(&file) {
        let mut s = LENS.write().ok()?;
        s.tried = Some(file);
        s.loaded = None;
        s.note = format!("lens {k} masked whole in its parameter overlay; the embedder serves alone");
        LENS_EPOCH.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        eprintln!("adapter: {}", s.note);
        return None;
    }
    let _g = LOADING.lock();
    let values = bound_values("embedder")?;
    let mut got = lens_of(&file, &values);
    if got.is_err()
        && let Some(o) = crate::tools::installed()
    {
        let id = adapter_id(LENS_PREFIX, &o.sub().root);
        for (_, cand) in versions(&home, &id).into_iter().rev() {
            if cand == file {
                continue;
            }
            if let Ok(l) = lens_of(&cand, &values) {
                got = Ok(l);
                break;
            }
        }
    }
    let mut s = LENS.write().ok()?;
    s.tried = Some(file);
    LENS_EPOCH.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    match got {
        Ok(mut l) => {
            // Diagnosis (`NANO_LENS_BETA`): the flattening's strength for a measurement.
            if let Some(b) = std::env::var("NANO_LENS_BETA").ok().and_then(|v| v.parse::<f32>().ok()) {
                l.beta = vec![b; l.beta.len()];
            }
            s.note = format!("lens {} seated on the embedder ({values})", l.key);
            s.loaded = Some(Arc::new(l));
        }
        Err(e) => {
            s.note = format!("lens left out: {e}");
            s.loaded = None;
        }
    }
    eprintln!("adapter: {}", s.note);
    s.loaded.clone()
}

/// The judge's code read-out in force (as `lens`).
#[must_use]
pub fn readout() -> Option<Arc<Readout>> {
    if !judge_on() {
        return None;
    }
    let home = nmtype::nano_root();
    if !plugged(&home, JUDGE_PREFIX) {
        if unseat(&READOUT) {
            eprintln!("adapter: judge code read-out unplugged; the judge serves alone");
        }
        return None;
    }
    let file = newest(&home, JUDGE_PREFIX)?;
    let ov = nmengn::overlay::stamp_cached(&home);
    if let Ok(s) = READOUT.read()
        && s.tried.as_ref() == Some(&file)
        && READOUT_OVERLAY.load(std::sync::atomic::Ordering::SeqCst) == ov
    {
        return s.loaded.clone();
    }
    READOUT_OVERLAY.store(ov, std::sync::atomic::Ordering::SeqCst);
    if let Some(k) = masked_whole(&file) {
        let mut s = READOUT.write().ok()?;
        s.tried = Some(file);
        s.loaded = None;
        s.note = format!("judge code read-out {k} masked whole in its parameter overlay; the judge serves alone");
        eprintln!("adapter: {}", s.note);
        return None;
    }
    let _g = LOADING.lock();
    let values = bound_values("reranker").or_else(|| crate::neural::reranker_model(&home).ok().map(|m| crate::neural::parameters_key(&m)))?;
    let mut got = readout_of(&file, &values);
    if got.is_err()
        && let Some(o) = crate::tools::installed()
    {
        let id = adapter_id(JUDGE_PREFIX, &o.sub().root);
        for (_, cand) in versions(&home, &id).into_iter().rev() {
            if cand == file {
                continue;
            }
            if let Ok(r) = readout_of(&cand, &values) {
                got = Ok(r);
                break;
            }
        }
    }
    let mut s = READOUT.write().ok()?;
    s.tried = Some(file);
    match got {
        Ok(r) => {
            s.note = format!("judge code read-out {} seated ({values})", r.key);
            s.loaded = Some(Arc::new(r));
        }
        Err(e) => {
            s.note = format!("judge code read-out left out: {e}");
            s.loaded = None;
        }
    }
    eprintln!("adapter: {}", s.note);
    s.loaded.clone()
}

/// Roadmap Phase 4.5 (2026-11-24): the keys (`id@version`) of the embedder's lens and the judge's
/// code read-out seated and switched on now, read without loading anything (for the live view).
#[must_use]
pub fn seated() -> (Option<String>, Option<String>) {
    let lens = if lens_on() { LENS.read().ok().and_then(|s| s.loaded.as_ref().map(|l| l.key.clone())) } else { None };
    let readout = if judge_on() { READOUT.read().ok().and_then(|s| s.loaded.as_ref().map(|r| r.key.clone())) } else { None };
    (lens, readout)
}

/// What the two adapters are doing, for the status and the trace.
#[must_use]
pub fn status() -> String {
    let l = LENS.read().map(|s| s.note.clone()).unwrap_or_default();
    let r = READOUT.read().map(|s| s.note.clone()).unwrap_or_default();
    format!(
        "lens {} ({}); judge code {} ({})",
        if lens_on() { "on" } else { "off" },
        if l.is_empty() { "not loaded" } else { l.as_str() },
        if judge_on() { "on" } else { "off" },
        if r.is_empty() { "not loaded" } else { r.as_str() }
    )
}

/// Hook on the meaning lane's request vector (`neural::start_dense`, the lane's query function):
/// the lens applied when it is in force; otherwise the vector untouched.
#[must_use]
pub fn lens_query(v: Option<Vec<f32>>) -> Option<Vec<f32>> {
    let v = v?;
    // Every request-side map in force: the distilled request lens, then the project's lens.
    Some(crate::distil::query_lens(&v).unwrap_or(v))
}

/// Roadmap Phase 0 item 3 (2026-11-23): the lens's flattening for the pieces' vectors, taken once a
/// question (the lens is looked up once, not per piece): the flattening on a piece's vector (the shared
/// directions taken out of the pieces as well as the question, as all-but-the-top does for every
/// vector of the space; Mu and Viswanath, ICLR 2018); its question-side map is not applied. `None`
/// when no lens is in force.
#[must_use]
pub fn lens_doc() -> Option<nmsubs::dense::PieceFlat> {
    let l = lens()?;
    Some(Arc::new(move |v: &[f32]| l.apply_flat(v)))
}

/// Whether a close-call document (its first line `path:first-last`, `neural::close_doc`) is code:
/// the material's file by that path is source code; a path the material does not hold is judged
/// by its file name's ending.
#[must_use]
pub fn doc_is_code(doc: &str) -> bool {
    let head = doc.lines().next().unwrap_or("");
    let path = head.rsplit_once(':').map_or(head, |(p, _)| p).replace('\\', "/");
    if let Some(o) = crate::tools::installed() {
        let sub = o.sub();
        if let Some(f) = sub.files.iter().find(|f| {
            let fp = f.path.replace('\\', "/");
            fp == path || path.ends_with(&format!("/{fp}"))
        }) {
            return f.is_code();
        }
    }
    code_name(path.rsplit('/').next().unwrap_or(&path))
}

/// Whether a file name names source code by its ending.
#[must_use]
pub fn code_name(name: &str) -> bool {
    let Some(ext) = name.rsplit_once('.').map(|(_, e)| e.to_lowercase()) else { return false };
    matches!(
        ext.as_str(),
        "rs" | "py" | "js" | "jsx" | "ts" | "tsx" | "go" | "java" | "kt" | "c" | "h" | "cc" | "cpp" | "hpp" | "cs" | "rb" | "php" | "swift" | "scala" | "lua" | "sh" | "toml"
    )
}

/// The judge's final-norm state at the last position of one close-call pair (the prompt the
/// close-call arm reads, `CODE_INSTRUCTION`), and its label log-odds from the same pass.
/// # Errors
/// Tokenizer or engine failure.
pub fn judge_state(m: &nmengn::Model, question: &str, doc: &str) -> Result<(Vec<f32>, f32), Fault> {
    let ids = m.pop.tokenizer.encode(&nmengn::roles::card_prompt(nmengn::roles::CODE_INSTRUCTION, question, doc)).map_err(|e| bad(e.to_string()))?;
    let targets = [(m.role_node("final_norm")?, "y".to_owned()), (m.role_node("label")?, "y".to_owned())];
    let dev = if m.pop.gpu.is_some() { nmengn::placement::Device::Gpu } else { nmengn::placement::Device::Cpu };
    let out = m.forward(dev, &ids, &targets, &nmengn::model::RunOpts::default())?;
    let y = nmengn::model::values(out.get(&targets[0]).ok_or_else(|| bad("no hidden state"))?)?;
    let p = nmengn::model::values(out.get(&targets[1]).ok_or_else(|| bad("no label output"))?)?.first().copied().unwrap_or(0.5);
    let d = m.pop.dims.hidden;
    let h = y.get(y.len().saturating_sub(d)..).map(<[f32]>::to_vec).unwrap_or_default();
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    Ok((h, (p / (1.0 - p)).ln()))
}

/// Hook on the close-call judge (`neural::judge_close`): inside a close call whose candidates are
/// all code (two or more), each candidate's log-odds gain the read-out's term; any other call, or
/// with the read-out not in force, `z` comes back untouched. A candidate whose state cannot be read
/// keeps its log-odds.
#[must_use]
pub fn judge_code(m: &nmengn::Model, question: &str, docs: &[String], z: Vec<f32>) -> Vec<f32> {
    judge_code_states(m, question, docs, z, &[])
}

/// Whether the code read-out applies to a close call over `docs` (two or more, all code, the
/// read-out in force and fitted on these parameters).
#[must_use]
pub fn judge_applies(m: &nmengn::Model, docs: &[String]) -> bool {
    docs.len() >= 2 && judge_on() && readout().is_some_and(|r| r.values == crate::neural::parameters_key(m)) && docs.iter().all(|d| doc_is_code(d))
}

/// `judge_code` with each document's last hidden state when the judge's packed run gave it
/// (`states`, by document; lane T1 2026-11-19, speed): a document without one is read alone.
#[must_use]
pub fn judge_code_states(m: &nmengn::Model, question: &str, docs: &[String], z: Vec<f32>, states: &[Option<Vec<f32>>]) -> Vec<f32> {
    if docs.len() != z.len() || !judge_applies(m, docs) {
        return z;
    }
    let Some(r) = readout() else { return z };
    // A measurement (`NANO_JUDGE_STATE_CHECK`): the packed run's terms against the separate read's.
    if std::env::var_os("NANO_JUDGE_STATE_CHECK").is_some() {
        let worst = docs
            .iter()
            .enumerate()
            .filter_map(|(i, d)| Some((r.term(states.get(i)?.as_ref()?) - r.term(&judge_state(m, question, d).ok()?.0)).abs()))
            .fold(0.0_f32, f32::max);
        eprintln!("judge state check: {} documents, largest term difference {worst:.4}", docs.len());
    }
    z.into_iter()
        .zip(docs)
        .enumerate()
        .map(|(i, (x, d))| match states.get(i).cloned().flatten().map_or_else(|| judge_state(m, question, d).map(|s| s.0), Ok) {
            Ok(h) => (x + r.term(&h)).clamp(-Z_CAP, Z_CAP),
            Err(_) => x,
        })
        .collect()
}

/// Self-check: an empty lens returns the vector untouched; a lens with one flattened direction
/// at full shrink removes that direction; a rank-one map adds its hand-computed delta; the file
/// form written and read back (in memory) gives the same numbers, refuses other values and
/// refuses parameters that do not match the label; a read-out of another width adds nothing.
/// # Errors
/// The first of those that fails.
pub fn check() -> Result<(), String> {
    let q = vec![0.6_f32, 0.8, 0.0];
    let id = Lens { key: "t@1".into(), values: "v".into(), dim: 3, flat: Vec::new(), beta: Vec::new(), rank: 0, down: Vec::new(), up: Vec::new() };
    if id.apply(&q) != q {
        return Err("nmapi.adapter: an empty lens changed the vector".into());
    }
    let flat = Lens { flat: vec![1.0, 0.0, 0.0], beta: vec![1.0], ..id.clone() };
    let f = flat.apply(&q);
    if f.first().is_none_or(|x| x.abs() > 1e-6) || f.get(1).is_none_or(|x| (x - 1.0).abs() > 1e-6) {
        return Err(format!("nmapi.adapter: flattening gave {f:?}"));
    }
    // down = [0, 0, 1] (r 1), up = [1, 0, 0]^T: q = [0, 0, 1] -> [1, 0, 1] / sqrt 2.
    let low = Lens { rank: 1, down: vec![0.0, 0.0, 1.0], up: vec![1.0, 0.0, 0.0], ..id };
    let g = low.apply(&[0.0, 0.0, 1.0]);
    let s = std::f32::consts::FRAC_1_SQRT_2;
    if g.iter().zip([s, 0.0, s]).any(|(a, b)| (a - b).abs() > 1e-6) {
        return Err(format!("nmapi.adapter: low-rank map gave {g:?}"));
    }
    let r = Readout { key: "r@1".into(), values: "v".into(), w: vec![1.0, 2.0], bias: 0.5, alpha: 2.0 };
    if (r.term(&[1.0, 1.0]) - 7.0).abs() > 1e-6 || r.term(&[1.0]) != 0.0 {
        return Err("nmapi.adapter: read-out term".into());
    }
    let label = crate::adapter_fit::label(&crate::adapter_fit::LabelIn {
        id: "check-lens",
        version: 1,
        model: EMBEDDER,
        weights: "w",
        kind: LENS_KIND,
        does: "self-check",
        made_from: "self-check",
        notes: Value::Object(std::iter::once(("values".to_owned(), Value::String("v".into()))).collect()),
    });
    let parts: Vec<Part> = vec![
        ("flat".to_owned(), vec![1, 3], vec![1.0_f32, 0.0, 0.0]),
        ("beta".to_owned(), vec![1], vec![0.5]),
        ("down".to_owned(), vec![1, 3], vec![0.0, 0.0, 1.0]),
        ("up".to_owned(), vec![3, 1], vec![1.0, 0.0, 0.0]),
    ];
    let bytes = crate::adapter_fit::encode_parts(label, &parts);
    let (l, p) = read_bytes(&bytes).map_err(|e| e.to_string())?;
    let back = lens_from(&l, &p, "v").map_err(|e| e.to_string())?;
    if back.beta != [0.5] || back.down != [0.0, 0.0, 1.0] || back.rank != 1 || back.dim != 3 {
        return Err(format!("nmapi.adapter: file round trip gave {back:?}"));
    }
    if lens_from(&l, &p, "other").is_ok() {
        return Err("nmapi.adapter: a lens fitted on other values loaded".into());
    }
    let mut broken = bytes;
    if let Some(b) = broken.last_mut() {
        *b ^= 1;
    }
    if read_bytes(&broken).is_ok() {
        return Err("nmapi.adapter: parameters that do not match the label were read".into());
    }
    Ok(())
}
