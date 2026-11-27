//! Downloads from the interface (2026-11-25, directions of Luis Jake Gabriel III: the base models
//! are downloadable from the interface and kept separate from the base cartridges).
//!
//! Two kinds, each its own list and its own action: the released models (every file the models
//! folder's weight cards and `INTEGRITY.md` name, from the Hugging Face repository and revision each
//! card pins) and the base cartridges (the serving versions `designs/base-cartridges.json` names,
//! from the cartridge repository). A file is placed only once its SHA-256 matches; a partial file
//! resumes; a file already present is kept. The engine is standard library only, so the transfer is
//! Windows' own HTTP service (`nmffi::http`, 2026-11-26; `curl.exe` only as the fallback), run in a
//! background thread; one download runs at a time.
//! Nothing is fetched while the air gap is on. A private cartridge repository needs the user's own
//! Hugging Face token (`HF_TOKEN`, else the `hf` login file), sent only to huggingface.co from a
//! header file that is removed afterwards.

use nmjson::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}
fn obj(f: Vec<(&str, Value)>) -> Value {
    Value::Object(f.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

/// The official cartridge library on Hugging Face (C1, 2026-11-26, Luis Jake Gabriel III: "Okay fix
/// everything"): the base cartridges each MicroModel serves, under the PolyForm Noncommercial
/// licence, each file checked against its SHA-256 in `designs/base-cartridges.json`.
pub const CARTRIDGE_REPO: &str = "gavriel-tech/neuromorph-nano-cartridges";

/// What happens to a fetched file once it is on disk (2026-11-26: cartridge sources).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Then {
    /// Kept where it landed (models, base cartridges).
    Keep,
    /// A library stack file: imported through the library's own checks (it arrives on trial),
    /// then the downloaded copy is removed.
    ImportStack,
    /// A parameter map for a MicroModel role: imported through the map checks, then removed.
    ImportMap(String),
}

/// One file to fetch: where it goes, where it comes from, what it must hash to (empty: the source
/// publishes no SHA-256 for it; the import step's own checks then decide), and what follows.
#[derive(Clone, Debug)]
pub struct Want {
    pub dest: PathBuf,
    pub url: String,
    pub sha: String,
    pub label: String,
    pub auth: bool,
    pub then: Then,
}

/// The state of the download in progress (or the last one). 2026-11-25 (nothing pseudo, every
/// process sees the same): it is also written to `memory/downloads.json` under the home, so a
/// download started from MCP or the command line shows in the interface and the other way round;
/// `updated_ms` is refreshed every few seconds while it runs, so a download whose process ended
/// reads as stopped, never as running forever.
#[derive(Clone, Debug, Default)]
struct Job {
    what: String,
    by: String,
    file: PathBuf,
    updated_ms: u64,
    running: bool,
    done: usize,
    total: usize,
    current: String,
    current_part: Option<PathBuf>,
    failed: Vec<String>,
    finished_ms: u64,
}

static JOB: Mutex<Option<Job>> = Mutex::new(None);

/// A running download whose state file has not been refreshed for this long belongs to a process
/// that ended.
const STALE_MS: u64 = 30_000;

fn job_file(home: &Path) -> PathBuf {
    home.join("memory").join("downloads.json")
}

fn job_value(j: &Job) -> Value {
    obj(vec![
        ("what", s(&j.what)),
        ("by", s(&j.by)),
        ("running", Value::Bool(j.running)),
        ("done", n(j.done as u64)),
        ("total", n(j.total as u64)),
        ("current", s(&j.current)),
        ("current_bytes", n(j.current_part.as_deref().map_or(0, size_of))),
        ("failed", Value::Array(j.failed.iter().map(|f| s(f)).collect())),
        ("finished_ms", n(j.finished_ms)),
        ("updated_ms", n(j.updated_ms)),
    ])
}

fn write_job(j: &Job) {
    if j.file.as_os_str().is_empty() {
        return;
    }
    if let Some(d) = j.file.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let tmp = j.file.with_extension("json.tmp");
    if std::fs::write(&tmp, nmjson::pretty::pretty(&job_value(j))).is_ok() {
        let _ = std::fs::rename(&tmp, &j.file);
    }
}

/// The download as any process sees it: this process's own job, else the shared state file (a
/// running one whose file went stale reads as stopped, with why).
fn current_job(home: &Path) -> Value {
    if let Some(j) = JOB.lock().ok().and_then(|g| g.clone()) {
        return job_value(&j);
    }
    let Some(mut v) = std::fs::read(job_file(home)).ok().and_then(|b| nmjson::parse(&b).ok()) else {
        return Value::Null;
    };
    let running = matches!(v.get("running"), Ok(Value::Bool(true)));
    let updated = v.get("updated_ms").and_then(Value::integer).unwrap_or(0);
    if running && nmtype::now_ms().saturating_sub(updated) > STALE_MS
        && let Value::Object(m) = &mut v
    {
        m.insert("running".into(), Value::Bool(false));
        m.insert("stopped".into(), s("the process that ran this download ended before it finished; start it again to resume"));
    }
    v
}

/// Whether a download is running in any process.
#[must_use]
pub fn running(home: &Path) -> bool {
    matches!(current_job(home).get("running"), Ok(Value::Bool(true)))
}

/// Every model file `INTEGRITY.md` names, by model: (file path under `files/`, SHA-256).
fn integrity_rows(home: &Path) -> BTreeMap<String, Vec<(String, String)>> {
    let mut out: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    let text = std::fs::read_to_string(home.join("INTEGRITY.md")).unwrap_or_default();
    for line in text.lines() {
        // | `models/<id>/files/<file>` | `<sha>` |
        let Some(rest) = line.strip_prefix("| `models/") else { continue };
        let Some((path, tail)) = rest.split_once('`') else { continue };
        let Some((id, file)) = path.split_once("/files/") else { continue };
        let sha = tail.split('`').nth(1).unwrap_or("").to_owned();
        if sha.len() == 64 {
            out.entry(id.to_owned()).or_default().push((file.to_owned(), sha));
        }
    }
    out
}

/// A weight card's origin repository (owner/name) and revision.
fn origin_of(home: &Path, id: &str) -> Option<(String, String, u64)> {
    let v = nmjson::parse(&std::fs::read(nmtype::models_dir(home).join(id).join("part.json")).ok()?).ok()?;
    let w = v.get("weights").ok()?;
    let origin = w.get("origin").ok()?.string().ok()?.trim_start_matches("https://huggingface.co/").to_owned();
    let rev = w.get("revision").ok()?.string().ok()?.to_owned();
    let bytes = w.get("bytes").ok().and_then(|b| b.integer().ok()).unwrap_or(0);
    Some((origin, rev, bytes))
}

fn size_of(p: &Path) -> u64 {
    std::fs::metadata(p).map_or(0, |m| m.len())
}

fn models_wants(home: &Path, id: &str) -> Vec<Want> {
    let Some((repo, rev, _)) = origin_of(home, id) else { return Vec::new() };
    let rows = integrity_rows(home);
    let dir = nmtype::models_dir(home).join(id).join("files");
    rows.get(id)
        .map(|files| {
            files
                .iter()
                .map(|(f, sha)| Want { dest: dir.join(f), url: format!("https://huggingface.co/{repo}/resolve/{rev}/{f}"), sha: sha.clone(), label: format!("{id}/{f}"), auth: false, then: Then::Keep })
                .collect()
        })
        .unwrap_or_default()
}

/// The base cartridge files the library serves (`designs/base-cartridges.json`), where they go on
/// the shelf and where each comes from in the cartridge repository.
fn cartridge_wants(home: &Path) -> Vec<Want> {
    let Some(v) = std::fs::read(home.join("designs/base-cartridges.json")).ok().and_then(|b| nmjson::parse(&b).ok()) else { return Vec::new() };
    let Ok(serving) = v.get("serving").and_then(Value::object) else { return Vec::new() };
    serving
        .values()
        .filter_map(|e| {
            let id = e.get("cartridge").ok()?.string().ok()?;
            let version = e.get("version").ok()?.integer().ok()?;
            let sha = e.get("sha256").ok()?.string().ok()?.to_owned();
            let role = id.strip_suffix("-base").unwrap_or(id);
            Some(Want {
                dest: home.join("library/cartridges/base").join(role).join(format!("{id}-v{version}.cartridge")),
                url: format!("https://huggingface.co/{CARTRIDGE_REPO}/resolve/main/base/{role}/{id}.cartridge"),
                sha,
                label: format!("{id}@{version}"),
                auth: false,
                then: Then::Keep,
            })
        })
        .collect()
}

/// Summary of local parameter maps and serving job filters across the four MicroModel roles,
/// including per-file descriptions for expandable UI info drawers.
fn parameter_maps_view(home: &Path) -> Value {
    let roles: Vec<Value> = ["reader", "writer", "embedder", "reranker"]
        .iter()
        .map(|role| {
            let dir = home.join("library/parameter-maps").join(role);
            let mut total: u64 = 0;
            let mut job_filters: u64 = 0;
            let mut bytes: u64 = 0;
            let mut has_super = false;
            let mut entries: Vec<(String, u64)> = std::fs::read_dir(&dir)
                .ok()
                .into_iter()
                .flatten()
                .flatten()
                .filter_map(|entry| {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    let md = entry.metadata().ok()?;
                    if !md.is_file() || !name.ends_with(".json") || name == nmapi::lineage::FILE || name.starts_with('.') {
                        return None;
                    }
                    Some((name, md.len()))
                })
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            let mut items = Vec::with_capacity(entries.len());
            for (name, len) in entries {
                total += 1;
                bytes += len;
                if name.starts_with("job-filter-") {
                    job_filters += 1;
                }
                if name == "super-weights.json" {
                    has_super = true;
                }
                let kind = if name.starts_with("job-filter-") {
                    "job filter"
                } else if name == "super-weights.json" {
                    "super-weights"
                } else if name.starts_with("pmap-") {
                    "parameter map"
                } else {
                    "map"
                };
                let desc = crate::sources::describe_file(&name, "parameter map", Some(role));
                items.push(obj(vec![
                    ("file", s(&name)),
                    ("kind", s(kind)),
                    ("bytes", n(len)),
                    ("description", s(&desc)),
                ]));
            }
            obj(vec![
                ("role", s(role)),
                ("files", n(total)),
                ("job_filters", n(job_filters)),
                ("super_weights", Value::Bool(has_super)),
                ("on_disk", n(bytes)),
                ("items", Value::Array(items)),
            ])
        })
        .collect();
    Value::Array(roles)
}

/// What is on disk for each model, each base cartridge and parameter maps, and the download in progress.
pub fn view(home: &Path) -> Value {
    let models: Vec<Value> = nmtype::role_models::installed(home)
        .iter()
        .map(|id| {
            let wants = models_wants(home, id);
            let (repo, rev, bytes) = origin_of(home, id).unwrap_or_default();
            let present = wants.iter().filter(|w| w.dest.is_file()).count();
            let partial = wants.iter().filter(|w| !w.dest.is_file() && w.dest.with_extension(part_ext(&w.dest)).is_file()).count();
            obj(vec![
                ("id", s(id)),
                ("repo", s(&repo)),
                ("revision", s(&rev)),
                ("bytes", n(bytes)),
                ("files", n(wants.len() as u64)),
                ("present", n(present as u64)),
                ("partial", n(partial as u64)),
                ("on_disk", n(wants.iter().map(|w| size_of(&w.dest)).sum())),
                ("state", s(if wants.is_empty() { "unknown" } else if present == wants.len() { "installed" } else if present + partial > 0 { "partial" } else { "missing" })),
            ])
        })
        .collect();
    let carts: Vec<Value> = cartridge_wants(home)
        .iter()
        .map(|w| {
            let rel = w.dest.strip_prefix(home).unwrap_or(&w.dest).display().to_string();
            let desc = crate::sources::describe_file(&rel, "base cartridge", None);
            obj(vec![
                ("key", s(&w.label)),
                ("present", Value::Bool(w.dest.is_file())),
                ("on_disk", n(size_of(&w.dest))),
                ("path", s(&rel)),
                ("description", s(&desc)),
            ])
        })
        .collect();
    let job = current_job(home);
    let token = token().is_some();
    obj(vec![
        ("models", Value::Array(models)),
        ("cartridges", Value::Array(carts)),
        ("parameter_maps", parameter_maps_view(home)),
        ("cartridge_repo", s(CARTRIDGE_REPO)),
        ("token", Value::Bool(token)),
        ("air_gap", Value::Bool(crate::controls::air_gap_on(home))),
        ("job", job),
    ])
}

fn part_ext(dest: &Path) -> String {
    format!("{}.part", dest.extension().and_then(|e| e.to_str()).unwrap_or("bin"))
}

/// Whether a Hugging Face token is on this machine (never its value).
#[must_use]
pub fn token_found() -> bool {
    token().is_some()
}

pub fn token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok())
        .or_else(|| std::env::var_os("USERPROFILE").and_then(|h| std::fs::read_to_string(PathBuf::from(h).join(".cache/huggingface/token")).ok()))
        .map(|t| t.trim().to_owned())
        .filter(|t| !t.is_empty())
}

/// Only the small runtime tokenizer and configuration JSONs (`config.json`, `tokenizer.json`) each
/// MicroModel needs to boot directly from its standalone base cartridge without downloading the
/// optional 8 GB upstream `.safetensors` open weights.
fn runtime_config_wants(home: &Path) -> Vec<Want> {
    nmtype::role_models::installed(home)
        .iter()
        .flat_map(|id| {
            models_wants(home, id)
                .into_iter()
                .filter(|w| w.label.ends_with("/config.json") || w.label.ends_with("/tokenizer.json"))
        })
        .collect()
}

/// Starts a download in the background: `models:<id>` (one model's files), `cartridges` (the
/// serving base cartridges plus the small tokenizer/config JSONs needed to run standalone), or
/// `parameter-maps` (official parameter maps and job filters from the public cartridge library;
/// pass `only: "serving"` for job filters, super-weights and surgery probes only). `only` for
/// models/cartridges keeps files whose name ends with it (for example `.json`).
/// # Errors
/// The air gap is on, a download is running, or there is nothing to fetch.
pub fn start(home: &Path, what: &str, only: Option<&str>, by: &str) -> Result<String, String> {
    if crate::controls::air_gap_on(home) {
        return Err("the air gap is on: nothing is downloaded until it is turned off".into());
    }
    if what == "parameter-maps" {
        let filter = match only.map(str::trim).filter(|s| !s.is_empty()) {
            Some("serving") => Some("serving"),
            _ => Some("parameter-maps"),
        };
        return crate::sources::fetch_all(home, crate::sources::OFFICIAL, filter, by);
    }
    let mut wants = if let Some(id) = what.strip_prefix("models:") {
        if !nmtype::role_models::installed(home).iter().any(|m| m == id) {
            return Err(format!("{id}: no weight card in the models folder"));
        }
        models_wants(home, id)
    } else if what == "cartridges" {
        let mut all = cartridge_wants(home);
        all.extend(runtime_config_wants(home));
        all
    } else {
        return Err(format!("{what}: download models:<id>, cartridges or parameter-maps"));
    };
    if let Some(o) = only.filter(|o| !o.is_empty()) {
        wants.retain(|w| w.label.ends_with(o));
    }
    wants.retain(|w| !w.dest.is_file());
    if wants.is_empty() {
        return Err("every file is already on disk".into());
    }
    let into = if what == "cartridges" { "the cartridge library" } else { "the models folder" };
    start_wants(home, what, wants, into, by)
}

/// Starts the download of `wants` in the background, as one job named `what`.
/// # Errors
/// The air gap is on, a download is running, or `wants` is empty.
pub fn start_wants(home: &Path, what: &str, wants: Vec<Want>, into: &str, by: &str) -> Result<String, String> {
    if crate::controls::air_gap_on(home) {
        return Err("the air gap is on: nothing is downloaded until it is turned off".into());
    }
    if wants.is_empty() {
        return Err("nothing to fetch".into());
    }
    // 2026-11-26: the shared state file is read before the lock is taken (`running` takes it too;
    // asked while holding it, a first download in a fresh process waited on itself forever).
    let elsewhere = running(home);
    {
        let mut g = JOB.lock().map_err(|_| "download state poisoned".to_owned())?;
        if g.as_ref().is_some_and(|j| j.running) || elsewhere {
            return Err("a download is already running".into());
        }
        let j = Job { what: what.to_owned(), by: by.to_owned(), file: job_file(home), updated_ms: nmtype::now_ms(), running: true, total: wants.len(), ..Job::default() };
        write_job(&j);
        *g = Some(j);
    }
    let line = format!("downloading {} {} to {into}", wants.len(), if wants.len() == 1 { "file" } else { "files" });
    let home = home.to_owned();
    std::thread::spawn(move || run(&home, &wants));
    Ok(line)
}

fn set(f: impl FnOnce(&mut Job)) {
    if let Ok(mut g) = JOB.lock()
        && let Some(j) = g.as_mut()
    {
        f(j);
        j.updated_ms = nmtype::now_ms();
        write_job(j);
    }
}

/// A header file carrying the user's Hugging Face token for `curl.exe` (so the token is never on a
/// command line), or None without a token. The caller removes it when done.
#[must_use]
pub fn token_header() -> Option<PathBuf> {
    token().and_then(|t| {
        let p = std::env::temp_dir().join(format!("nano-hf-{}-{}.txt", std::process::id(), nmtype::now_ms()));
        std::fs::write(&p, format!("Authorization: Bearer {t}")).ok().map(|()| p)
    })
}

fn run(home: &Path, wants: &[Want]) {
    let header = if wants.iter().any(|w| w.auth) { token_header() } else { None };
    // The heartbeat: the shared state (with the bytes of the file in progress) refreshed every few
    // seconds while the download runs.
    let beat = std::thread::spawn(|| {
        while JOB.lock().ok().and_then(|g| g.as_ref().map(|j| j.running)).unwrap_or(false) {
            set(|_| {});
            std::thread::park_timeout(std::time::Duration::from_secs(3));
        }
    });
    for w in wants {
        let part = w.dest.with_extension(part_ext(&w.dest));
        set(|j| {
            j.current = w.label.clone();
            j.current_part = Some(part.clone());
        });
        let by = JOB.lock().ok().and_then(|g| g.as_ref().map(|j| j.by.clone())).unwrap_or_default();
        let r = fetch_one(w, &part, header.as_deref()).and_then(|()| after(home, w, &by));
        set(|j| {
            j.done += 1;
            if let Err(e) = &r {
                j.failed.push(format!("{}: {e}", w.label));
            }
        });
    }
    if let Some(h) = header {
        let _ = std::fs::remove_file(h);
    }
    let by = JOB.lock().ok().and_then(|g| g.as_ref().map(|j| j.by.clone())).unwrap_or_default();
    let _ = crate::controls::audit(home, "download", vec![("files", n(wants.len() as u64)), ("by", s(&by))]);
    set(|j| {
        j.running = false;
        j.current_part = None;
        j.finished_ms = nmtype::now_ms();
    });
    let _ = beat.join();
}

/// The step after a file is placed: an import through the library's or the maps' own checks, then
/// the downloaded copy goes (the library keeps its own). A refused import leaves nothing behind.
fn after(home: &Path, w: &Want, by: &str) -> Result<(), String> {
    let r = match &w.then {
        Then::Keep => return Ok(()),
        Then::ImportStack => crate::library::import_api(home, &w.dest.display().to_string(), by).map(|_| ()),
        Then::ImportMap(role) => crate::pmaps::import(home, role, &w.dest.display().to_string(), None, false, by).map(|_| ()),
    };
    if let Ok(m) = std::fs::metadata(&w.dest) {
        let mut p = m.permissions();
        #[expect(clippy::permissions_set_readonly_false, reason = "the downloaded copy is removed right after; the read-only mark set on placing it would refuse that on Windows")]
        p.set_readonly(false);
        let _ = std::fs::set_permissions(&w.dest, p);
    }
    let _ = std::fs::remove_file(&w.dest);
    r.map_err(|e| format!("not imported: {e}"))
}

/// The refusal text for a private or gated repository.
const REFUSED: &str = "the repository refused the download (private: set HF_TOKEN or log in with the hf command line)";

/// One file through Windows' own HTTP service (nmffi::http; 2026-11-26, Luis Jake Gabriel III: "Any
/// dependencies should be rebuilt in rust"): resumes a partial file with a Range request, starts
/// over when the server sends the whole file, retries a network failure three times, and sends the
/// token only to huggingface.co (never to the file servers a download is redirected to).
fn fetch_rust(w: &Want, part: &Path) -> Result<(), String> {
    use std::io::Write as _;
    let token = if w.auth { token() } else { None };
    let mut last = String::new();
    for attempt in 0..4_u32 {
        if attempt > 0 {
            std::thread::park_timeout(std::time::Duration::from_secs(2));
        }
        let have = std::fs::metadata(part).map_or(0, |m| m.len());
        let mut file: Option<std::fs::File> = None;
        let mut refused: Option<String> = None;
        let r = nmffi::http::get(&w.url, token.as_deref(), "huggingface.co", have, 120_000, &mut |ev| match ev {
            nmffi::http::Event::Status(206) => {
                file = Some(std::fs::OpenOptions::new().create(true).append(true).open(part).map_err(|e| format!("{}: {e}", part.display()))?);
                Ok(true)
            }
            nmffi::http::Event::Status(200) => {
                file = Some(std::fs::File::create(part).map_err(|e| format!("{}: {e}", part.display()))?);
                Ok(true)
            }
            // The partial file already holds every byte.
            nmffi::http::Event::Status(416) => Ok(false),
            nmffi::http::Event::Status(401 | 403) => {
                refused = Some(REFUSED.to_owned());
                Ok(false)
            }
            nmffi::http::Event::Status(s) => {
                refused = Some(format!("the server answered {s}"));
                Ok(false)
            }
            nmffi::http::Event::Data(bytes) => {
                if let Some(f) = file.as_mut() {
                    f.write_all(bytes).map_err(|e| format!("{}: {e}", part.display()))?;
                }
                Ok(true)
            }
        });
        if let Some(why) = refused {
            return Err(why);
        }
        match r {
            Ok(_) => {
                if let Some(mut f) = file {
                    f.flush().map_err(|e| format!("{}: {e}", part.display()))?;
                }
                return Ok(());
            }
            Err(e) => last = e,
        }
    }
    Err(last)
}

fn fetch_one(w: &Want, part: &Path, header: Option<&Path>) -> Result<(), String> {
    if let Some(d) = w.dest.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    if nmffi::http::available() {
        fetch_rust(w, part)?;
    } else {
        fetch_curl(w, part, header)?;
    }
    placed(w, part)
}

/// The same download through the `curl.exe` program, kept as the fallback for a Windows whose
/// HTTP service cannot be loaded.
fn fetch_curl(w: &Want, part: &Path, header: Option<&Path>) -> Result<(), String> {
    let mut cmd = std::process::Command::new("curl.exe");
    cmd.args(["-L", "--fail", "--silent", "--show-error", "--retry", "3", "--retry-delay", "2", "-C", "-", "-o"]).arg(part);
    if w.auth
        && let Some(h) = header
    {
        cmd.arg("-H").arg(format!("@{}", h.display()));
    }
    let out = cmd.arg(&w.url).output().map_err(|e| format!("curl.exe could not start: {e}"))?;
    if !out.status.success() {
        let why = String::from_utf8_lossy(&out.stderr).trim().to_owned();
        return Err(if why.contains("401") || why.contains("403") { REFUSED.to_owned() } else { why });
    }
    Ok(())
}

/// After either download path: the file's SHA-256 checked, then the part file placed read-only.
fn placed(w: &Want, part: &Path) -> Result<(), String> {
    let got = std::fs::File::open(part).and_then(nmhash::sha256).map_err(|e| format!("{}: {e}", part.display()))?;
    // A source that publishes no SHA-256 for a file (a small file outside Git LFS): the import
    // step's own checks decide (a stack file carries its own hashes; a map is checked by its kind).
    if !w.sha.is_empty() && got != w.sha {
        let _ = std::fs::remove_file(part);
        return Err(format!("SHA-256 {got:.12} is not the expected {:.12}; nothing was placed", w.sha));
    }
    std::fs::rename(part, &w.dest).map_err(|e| format!("{}: {e}", w.dest.display()))?;
    if let Ok(m) = std::fs::metadata(&w.dest) {
        let mut p = m.permissions();
        p.set_readonly(true);
        let _ = std::fs::set_permissions(&w.dest, p);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-11-26: the first download of a fresh process starts (it once waited on its own lock).
    #[test]
    fn a_first_download_starts_without_waiting_on_itself() {
        let home = std::env::temp_dir().join(format!("nm-fetch-start-{}", std::process::id()));
        std::fs::create_dir_all(home.join("memory")).unwrap();
        let want = Want { dest: home.join("got.bin"), url: format!("file:///{}", home.join("absent.bin").display()), sha: String::new(), label: "probe".into(), auth: false, then: Then::Keep };
        let (tx, rx) = std::sync::mpsc::channel();
        let h = home.clone();
        std::thread::spawn(move || {
            let _ = tx.send(start_wants(&h, "probe", vec![want], "a test folder", "test"));
        });
        let r = rx.recv_timeout(std::time::Duration::from_secs(10)).expect("start_wants returned within 10 s");
        assert!(r.is_ok(), "{r:?}");
        for _ in 0..100 {
            if !JOB.lock().unwrap().as_ref().is_some_and(|j| j.running) {
                break;
            }
            std::thread::park_timeout(std::time::Duration::from_millis(100));
        }
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn integrity_rows_read() {
        let home = std::env::temp_dir().join(format!("nm-fetch-{}", std::process::id()));
        std::fs::create_dir_all(&home).unwrap();
        let sha = "a".repeat(64);
        std::fs::write(home.join("INTEGRITY.md"), format!("| `models/qwen3-0.6b/files/config.json` | `{sha}` |\n| `kernels/x` | `{sha}` |\n")).unwrap();
        let r = integrity_rows(&home);
        assert_eq!(r.get("qwen3-0.6b").map(Vec::len), Some(1));
        assert_eq!(r.len(), 1);
        let _ = std::fs::remove_dir_all(&home);
    }
}
