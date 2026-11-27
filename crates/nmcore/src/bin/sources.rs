//! Cartridge sources (2026-11-26, Luis Jake Gabriel III, his words: "Users should be able to connect
//! form our official hugging face cartridge libraries or 3rd parties along with the open weightsz
//! obviously our fallback openweights are the Qwen open weights.").
//!
//! A source is a Hugging Face repository that holds cartridges: the official Neuromorph library
//! (built in, `fetch::CARTRIDGE_REPO`), and any third-party library the user adds by its repository
//! name. The Qwen3 open weights stay the fallback, from the downloads (`fetch`), unchanged.
//!
//! Nothing is fetched until the user asks: browsing lists a source's files (one request to
//! huggingface.co, kept as the last listing), fetching downloads one file through the downloader
//! (`fetch::start_wants`: the published SHA-256 checked when the source publishes one, partial files
//! resumed) and then imports it through the library's own checks, so a third party's file is held
//! to the same tests as one of ours:
//! - a library stack file (the library's export, every file and the whole body hashed; under
//!   `stacks/` or named `<name>.stack.json`) is imported on trial (`library::import_api`), so it
//!   serves only once its evidence allows;
//! - a parameter map under `parameter-maps/<role>/` is imported through the map checks
//!   (`pmaps::import`), never over an existing map;
//! - a base cartridge is listed but not installed from a third party: base cartridges come from the
//!   official library and the serving list (`designs/base-cartridges.json`) only, in v0.1.0.
//!
//! The air gap blocks browsing, fetching and publishing. A token is sent only where the user said
//! so (a third-party source with `token: true`, or an explicit publish to Hugging Face; the
//! official public library sends no token on browse or fetch), only to huggingface.co. The list of
//! sources is `<home>/cartridge-sources.json`; every change is an audit row (kind
//! `cartridge-source`); removing a source goes through the two-step delete gate and keeps whatever
//! was already imported from it.
use crate::fetch::{Then, Want};
use nmjson::Value;
use std::path::{Path, PathBuf};

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}
fn obj(f: Vec<(&str, Value)>) -> Value {
    Value::Object(f.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn text(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default()
}

/// The id of the official library.
pub const OFFICIAL: &str = "official";
/// The roles a parameter map can be imported for.
const ROLES: [&str; 4] = ["reader", "writer", "embedder", "reranker"];
/// The listing keeps at most this many files.
const MAX_FILES: usize = 5000;
/// Direct Hugging Face commit uploads (NDJSON base64) stay at or below 10 MiB; larger files use
/// `hf upload` (Xet/LFS).
pub const PUBLISH_DIRECT_LIMIT: u64 = 10 * 1024 * 1024;

/// One source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Source {
    pub id: String,
    pub repo: String,
    /// `model` or `dataset` (Hugging Face keeps both kinds of repository).
    pub kind: String,
    pub revision: String,
    pub official: bool,
    /// Whether the user's Hugging Face token is sent to huggingface.co for this source.
    pub token: bool,
    pub added_ms: u64,
}

fn file(home: &Path) -> PathBuf {
    home.join("cartridge-sources.json")
}
fn listing_file(home: &Path, id: &str) -> PathBuf {
    home.join("memory").join("cartridge-sources").join(format!("{id}.json"))
}

/// The official public library, as the downloads use it.
#[must_use]
pub fn official() -> Source {
    Source { id: OFFICIAL.into(), repo: crate::fetch::CARTRIDGE_REPO.into(), kind: "model".into(), revision: "main".into(), official: true, token: false, added_ms: 0 }
}

/// A repository name as Hugging Face writes it: `owner/name`, letters, digits, `-`, `_`, `.`.
fn repo_ok(r: &str) -> bool {
    let mut parts = r.split('/');
    let ok = |p: &str| !p.is_empty() && p.len() <= 96 && !p.starts_with('.') && p.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    matches!((parts.next(), parts.next(), parts.next()), (Some(a), Some(b), None) if ok(a) && ok(b))
}

/// The id of a third-party source: its repository with `/` as `--`.
#[must_use]
pub fn id_of(repo: &str) -> String {
    repo.replace('/', "--").to_ascii_lowercase()
}

/// Every source: the official one first, then the third parties in the order added.
#[must_use]
pub fn list(home: &Path) -> Vec<Source> {
    let mut out = vec![official()];
    let saved = std::fs::read(file(home)).ok().and_then(|b| nmjson::parse(&b).ok());
    for v in saved.as_ref().and_then(|v| v.get("sources").ok()).and_then(|a| a.array().ok()).unwrap_or_default() {
        let repo = text(v, "repo");
        if !repo_ok(&repo) {
            continue;
        }
        out.push(Source {
            id: id_of(&repo),
            kind: if text(v, "kind") == "dataset" { "dataset".into() } else { "model".into() },
            revision: Some(text(v, "revision")).filter(|r| !r.is_empty()).unwrap_or_else(|| "main".into()),
            token: matches!(v.get("token"), Ok(Value::Bool(true))),
            added_ms: v.get("added_ms").and_then(Value::integer).unwrap_or(0),
            official: false,
            repo,
        });
    }
    out
}

fn write(home: &Path, third: &[Source]) -> Result<(), String> {
    let rows: Vec<Value> = third
        .iter()
        .map(|x| obj(vec![("repo", s(&x.repo)), ("kind", s(&x.kind)), ("revision", s(&x.revision)), ("token", Value::Bool(x.token)), ("added_ms", n(x.added_ms))]))
        .collect();
    let v = obj(vec![("note", s("Third-party cartridge libraries added in the interface or over MCP; the official library is built in. Every change is in memory/audit.jsonl.")), ("sources", Value::Array(rows))]);
    let p = file(home);
    let tmp = p.with_extension("json.tmp");
    std::fs::write(&tmp, nmjson::pretty::pretty(&v)).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &p).map_err(|e| format!("{}: {e}", p.display()))
}

/// Finds a source by id.
/// # Errors
/// No such source.
pub fn find(home: &Path, id: &str) -> Result<Source, String> {
    list(home).into_iter().find(|x| x.id == id).ok_or_else(|| format!("no cartridge source {id}"))
}

fn base_url(src: &Source) -> String {
    if src.kind == "dataset" { format!("https://huggingface.co/datasets/{}", src.repo) } else { format!("https://huggingface.co/{}", src.repo) }
}

/// True when any path segment is `archived` or `archive` (decommissioned versions kept on Hugging
/// Face in an archive tree mirroring `base/`, `parameter-maps/`, and `stacks/`, invisible to the
/// downloader so users only ever see healthy current content).
fn is_archived_path(path: &str) -> bool {
    path.split('/').any(|seg| {
        let s = seg.to_ascii_lowercase();
        s == "archived" || s == "archive"
    })
}

/// What a listed file is to the Nano, and whether it installs from this source: (kind, role for a
/// map, why not when it does not).
fn class_of(src: &Source, path: &str) -> (&'static str, Option<String>, Option<String>) {
    if is_archived_path(path) {
        return ("archived", None, Some("archived (decommissioned version; hidden from downloads)".into()));
    }
    let lower = path.to_ascii_lowercase();
    if lower.ends_with("/lineage.json") {
        return ("lineage record", None, Some("the library's record of where its maps came from; read with the maps, not installed on its own".into()));
    }
    let role = path.split('/').find(|seg| ROLES.contains(&seg.to_ascii_lowercase().as_str())).map(str::to_ascii_lowercase);
    if lower.ends_with(".cartridge") {
        if lower.starts_with("base/") || lower.contains("/base/") {
            let why = if src.official { "the base cartridges the MicroModels serve come through Downloads (the serving list)" } else { "base cartridges come only from the official library in v0.1.0" };
            return ("base cartridge", None, Some(why.into()));
        }
        return ("cartridge file", None, Some("a single cartridge file is not installed on its own; libraries share stacks as stack files".into()));
    }
    if lower.ends_with(".json") && (lower.contains("parameter-maps/") || lower.contains("maps/")) {
        return match role {
            Some(r) => ("parameter map", Some(r), None),
            None => ("parameter map", None, Some("its path names no MicroModel role (reader, writer, embedder, reranker)".into())),
        };
    }
    // A library shares stacks under `stacks/` or as `<name>.stack.json` (the library's export).
    if lower.ends_with(".stack.json") || (lower.ends_with(".json") && (lower.starts_with("stacks/") || lower.contains("/stacks/"))) {
        return ("stack file", None, None);
    }
    ("other", None, Some("not a kind of file the Nano installs (stacks go under stacks/ or end in .stack.json; maps under parameter-maps/<role>/)".into()))
}

/// The listing kept for a source, or null.
fn cached(home: &Path, id: &str) -> Value {
    std::fs::read(listing_file(home, id)).ok().and_then(|b| nmjson::parse(&b).ok()).unwrap_or(Value::Null)
}

fn source_value(home: &Path, x: &Source) -> Value {
    obj(vec![
        ("id", s(&x.id)),
        ("repo", s(&x.repo)),
        ("kind", s(&x.kind)),
        ("revision", s(&x.revision)),
        ("official", Value::Bool(x.official)),
        ("token", Value::Bool(x.token)),
        ("added_ms", n(x.added_ms)),
        ("url", s(&base_url(x))),
        ("listing", cached(home, &x.id)),
    ])
}

fn is_base_cartridge(c: &nmlearn::library::Cartridge) -> bool {
    c.family == "base" || c.kind == "tensors" || c.files.iter().any(|f| f.name.starts_with("base/"))
}

/// Plain-English description of what a cartridge, parameter map, job filter or stack file does,
/// for expandable UI info drawers and listings.
#[must_use]
pub fn describe_file(path: &str, kind: &str, role: Option<&str>) -> String {
    let lower = path.to_ascii_lowercase();
    let name = path.rsplit('/').next().unwrap_or(path);
    let r = role.unwrap_or("MicroModel");
    if kind == "base cartridge" || lower.ends_with(".cartridge") {
        if lower.contains("reader-base") {
            return "Standalone base cartridge for the Reader MicroModel: reads workspace files and extracts cited spans without needing upstream .safetensors.".into();
        }
        if lower.contains("writer-base") {
            return "Standalone base cartridge for the Writer MicroModel: writes idle L1-L4 world-model comprehension notes without needing upstream .safetensors.".into();
        }
        if lower.contains("embedder-base") {
            return "Standalone base cartridge for the Embedder MicroModel: builds and queries dense semantic embeddings across your workspace.".into();
        }
        if lower.contains("reranker-base") {
            return "Standalone base cartridge for the Reranker MicroModel: ranks and judges close candidate spans and enforces decoy withholding.".into();
        }
        return "Neuromorph cartridge file holding verified parameter tensors or hardware/tuning settings.".into();
    }
    if name.starts_with("job-filter-") && name.ends_with(".json") {
        return match (role, name) {
            (Some("reader"), "job-filter-read.json") => "Maps per-layer affine gain and bias sockets across Reader layers for workspace reading and cited span extraction (active while plugged into reader-base).".into(),
            (Some("writer"), "job-filter-write.json") => "Maps per-layer affine gain and bias sockets across Writer layers for idle L1-L4 world-model note and outline writing (active while plugged into writer-base).".into(),
            (Some("reranker"), "job-filter-judge-code.json") => "Maps per-layer affine gain and bias sockets across Reranker layers for close-candidate code judging and decoy withholding (active while plugged into reranker-base).".into(),
            (_, "job-filter-code-slice.json") => format!("Maps per-layer affine gain and bias sockets across {r} layers for code-slice representation and retrieval (active while plugged into {r}-base)."),
            _ => {
                let job = name.strip_prefix("job-filter-").and_then(|x| x.strip_suffix(".json")).unwrap_or("job");
                format!("Maps per-layer affine gain and bias sockets across {r} layers for the '{job}' job (active while plugged in without altering base weights).")
            }
        };
    }
    if name == "super-weights.json" {
        return format!("Maps exact [row, col] high-leverage super-weight coordinates across {r} attention and MLP projections pinned so surgery and pruning preserve them.");
    }
    if name.starts_with("pmap-") && name.ends_with(".json") {
        if let Some(pct) = name.strip_prefix("pmap-owl-").and_then(|x| x.strip_suffix(".json")).filter(|s| s.chars().all(|c| c.is_ascii_digit())) {
            return format!("Maps per-weight Wanda sensitivity (|w| * ||x||) and OWL outlier-weighted layer budgets across {r} attention and MLP projections ({pct}% target sparsity), used to prune and build {r}-base.");
        }
        return format!("Maps per-weight sensitivity and layer sparsity budgets across {r} attention and MLP projections, used to prune and build {r}-base from upstream open weights.");
    }
    if name.starts_with("act-") && name.ends_with("-bias.json") {
        return format!("Maps per-output mean-activation bias compensation offsets paired with an activation channel mask across {r} layers.");
    }
    if name.starts_with("act-") && name.ends_with(".json") {
        return format!("Maps per-channel feed-forward activation magnitudes across {r} layers to score low-activity channels.");
    }
    if name.starts_with("role") && name.ends_with(".json") {
        return format!("Maps per-layer role activation energy and sensitivity separation across {r} layers.");
    }
    if name.starts_with("tol") && name.ends_with(".json") {
        return format!("Maps per-layer pruning tolerance thresholds measuring hidden-state drift per layer for {r}.");
    }
    if name.starts_with("swigate") && name.ends_with(".json") {
        return format!("Maps mean SwiGLU gate activations per channel across {r} MLP layers.");
    }
    if name == "surgery-probe.json" {
        return format!("Maps layer-by-layer hidden-state cosine similarity and max drift comparing {r}-base against upstream open weights.");
    }
    if kind == "stack file" || lower.ends_with(".stack.json") {
        if lower.contains("gtx-1650") {
            return "Shared hardware tuning stack for NVIDIA GeForce GTX 1650 (4 GB VRAM): bundles measured precision, batching, and co-residency settings for a 4 GB Turing GPU. Imports on trial.".into();
        }
        return "Exported cartridge stack bundling verified non-base cartridges and their SHA-256 hashes. Imports on trial so you can evaluate before activating.".into();
    }
    if kind == "parameter map" {
        let stem = name.strip_suffix(".json").unwrap_or(name);
        return format!("Maps parameter measurements ({stem}) across {r} layers.");
    }
    if name.eq_ignore_ascii_case("MANIFEST.md") {
        return "SHA-256 integrity manifest and provenance log describing every file in this folder.".into();
    }
    format!("Repository file ({kind}).")
}

/// Local parameter maps and exportable non-base stacks the user can publish to Hugging Face.
fn publishable(home: &Path) -> Value {
    let mut maps = Vec::new();
    for role in ROLES {
        let dir = home.join("library/parameter-maps").join(role);
        let mut names: Vec<(String, u64)> = std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                let md = e.metadata().ok()?;
                if !md.is_file() || !name.ends_with(".json") || name == nmapi::lineage::FILE || name.starts_with('.') {
                    return None;
                }
                Some((name, md.len()))
            })
            .collect();
        names.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, len) in names {
            let desc = describe_file(&name, "parameter map", Some(role));
            maps.push(obj(vec![
                ("role", s(role)),
                ("file", s(&name)),
                ("bytes", n(len)),
                ("direct_ok", Value::Bool(len <= PUBLISH_DIRECT_LIMIT)),
                ("description", s(&desc)),
            ]));
        }
    }
    let mut stacks = Vec::new();
    if let Ok(lib) = nmlearn::library::Library::open(home) {
        for st in &lib.stacks {
            let has_base = st.cartridges.iter().any(|k| lib.cartridge(k).is_some_and(is_base_cartridge));
            if has_base {
                continue;
            }
            stacks.push(obj(vec![
                ("key", s(&st.key())),
                ("use", s(&st.use_for)),
                ("description", s(&st.description)),
            ]));
        }
    }
    obj(vec![("maps", Value::Array(maps)), ("stacks", Value::Array(stacks))])
}

/// `GET /api/cartridge-sources`: every source with its last listing, the air gap, publishable
/// local items, and whether a token is on this machine.
#[must_use]
pub fn view(home: &Path) -> Value {
    obj(vec![
        ("sources", Value::Array(list(home).iter().map(|x| source_value(home, x)).collect())),
        ("air_gap", Value::Bool(crate::controls::air_gap_on(home))),
        ("token_found", Value::Bool(crate::fetch::token_found())),
        ("fallback", s("the Qwen3 open weights (Apache-2.0), from Downloads: the optional upstream fallback, unchanged")),
        ("file", s(&file(home).display().to_string())),
        ("publishable", publishable(home)),
        ("publish_limit_bytes", n(PUBLISH_DIRECT_LIMIT)),
    ])
}

/// Adds a third-party source. Returns the audit id.
/// # Errors
/// A repository name that does not read, one already added, or a write failure.
pub fn add(home: &Path, repo: &str, kind: Option<&str>, revision: Option<&str>, token: bool, by: &str) -> Result<String, String> {
    let repo = repo.trim().trim_start_matches("https://huggingface.co/").trim_start_matches("datasets/").trim_end_matches('/').to_owned();
    if !repo_ok(&repo) {
        return Err(format!("{repo}: give a Hugging Face repository as owner/name"));
    }
    if repo.eq_ignore_ascii_case(crate::fetch::CARTRIDGE_REPO) {
        return Err("that is the official library; it is built in".into());
    }
    let rev = revision.map(str::trim).filter(|r| !r.is_empty()).unwrap_or("main");
    if !rev.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')) || rev.len() > 64 {
        return Err(format!("revision {rev}: a branch, tag or commit"));
    }
    let mut third: Vec<Source> = list(home).into_iter().filter(|x| !x.official).collect();
    if third.iter().any(|x| x.id == id_of(&repo)) {
        return Err(format!("{repo} is already a source"));
    }
    let src = Source { id: id_of(&repo), repo: repo.clone(), kind: if kind == Some("dataset") { "dataset".into() } else { "model".into() }, revision: rev.into(), official: false, token, added_ms: nmtype::now_ms() };
    third.push(src.clone());
    write(home, &third)?;
    crate::controls::audit(
        home,
        "cartridge-source",
        vec![("action", s("add")), ("source", s(&src.id)), ("repo", s(&repo)), ("kind", s(&src.kind)), ("revision", s(rev)), ("token", Value::Bool(token)), ("by", s(by)), ("label", s(&format!("Cartridge source {repo} added")))],
    )
}

/// What removing a source takes away: (the source, its kept listing if any).
/// # Errors
/// The official source, or no such source.
pub fn remove_plan(home: &Path, id: &str) -> Result<(Source, Option<PathBuf>), String> {
    if id == OFFICIAL {
        return Err("the official library is built in; it is not removed".into());
    }
    let src = find(home, id)?;
    let l = listing_file(home, id);
    Ok((src, l.is_file().then_some(l)))
}

/// Removes a third-party source (after its confirmation) and its kept listing. What was imported from
/// it stays in the library. Returns the audit id; Audit can add it again.
/// # Errors
/// As `remove_plan`, or a write failure.
pub fn remove(home: &Path, id: &str, by: &str) -> Result<String, String> {
    let (src, l) = remove_plan(home, id)?;
    let third: Vec<Source> = list(home).into_iter().filter(|x| !x.official && x.id != id).collect();
    write(home, &third)?;
    if let Some(l) = l {
        let _ = std::fs::remove_file(l);
    }
    crate::controls::audit(
        home,
        "cartridge-source",
        vec![
            ("action", s("remove")),
            ("source", s(id)),
            ("repo", s(&src.repo)),
            ("kind", s(&src.kind)),
            ("revision", s(&src.revision)),
            ("token", Value::Bool(src.token)),
            ("by", s(by)),
            ("label", s(&format!("Cartridge source {} removed (what was imported from it stays)", src.repo))),
        ],
    )
}

/// Undoes an add (removes the source) or a remove (adds it again), by its audit row.
/// # Errors
/// A row that is not a source change, or the change failing.
pub fn undo(home: &Path, row: &Value, id: &str, by: &str) -> Result<Option<String>, String> {
    let repo = text(row, "repo");
    let audit = match text(row, "action").as_str() {
        "add" => remove(home, &id_of(&repo), by)?,
        "remove" => add(home, &repo, Some(&text(row, "kind")), Some(&text(row, "revision")), matches!(row.get("token"), Ok(Value::Bool(true))), by)?,
        a => return Err(format!("{id}: a source {a} is not undone")),
    };
    Ok(Some(audit))
}

/// Reads one listing page of the repository tree through `curl.exe` (the only outbound path, as the
/// downloads use). Returns the body.
fn tree(src: &Source) -> Result<Vec<u8>, String> {
    let api = if src.kind == "dataset" { "datasets" } else { "models" };
    let url = format!("https://huggingface.co/api/{api}/{}/tree/{}?recursive=true", src.repo, src.revision);
    // 2026-11-26: through Windows' own HTTP service (nmffi::http); curl.exe below is the fallback.
    if nmffi::http::available() {
        let token = if src.token { crate::fetch::token() } else { None };
        let mut body = Vec::new();
        let mut status = 0_u16;
        nmffi::http::get(&url, token.as_deref(), "huggingface.co", 0, 60_000, &mut |ev| {
            match ev {
                nmffi::http::Event::Status(s) => status = s,
                nmffi::http::Event::Data(b) => body.extend_from_slice(b),
            }
            Ok(true)
        })?;
        return match status {
            200 => Ok(body),
            401 | 403 => Err(format!("{} refused the listing (private or gated: tick 'send my token' for this source, with HF_TOKEN set or the hf login done)", src.repo)),
            404 => Err(format!("{} at {} was not found on Hugging Face (check the name, the revision and whether it is a dataset)", src.repo, src.revision)),
            s => Err(format!("{} answered {s} to the listing", src.repo)),
        };
    }
    let mut cmd = std::process::Command::new("curl.exe");
    cmd.args(["-L", "--fail", "--silent", "--show-error", "--max-time", "60"]);
    let header = if src.token {
        crate::fetch::token_header()
    } else {
        None
    };
    if let Some(h) = &header {
        cmd.arg("-H").arg(format!("@{}", h.display()));
    }
    let out = cmd.arg(&url).output().map_err(|e| format!("curl.exe could not start: {e}"));
    if let Some(h) = header {
        let _ = std::fs::remove_file(h);
    }
    let out = out?;
    if !out.status.success() {
        let why = String::from_utf8_lossy(&out.stderr).trim().to_owned();
        return Err(if why.contains("401") || why.contains("403") {
            format!("{} refused the listing (private or gated: tick 'send my token' for this source, with HF_TOKEN set or the hf login done)", src.repo)
        } else if why.contains("404") {
            format!("{} at {} was not found on Hugging Face (check the name, the revision and whether it is a dataset)", src.repo, src.revision)
        } else {
            why
        });
    }
    Ok(out.stdout)
}

/// Lists a source's files (after the user asks), keeps the listing, and returns it.
/// # Errors
/// The air gap is on, no such source, the request failing, or a reply that does not read.
pub fn browse(home: &Path, id: &str, by: &str) -> Result<Value, String> {
    if crate::controls::air_gap_on(home) {
        return Err("the air gap is on: no source is browsed until it is turned off".into());
    }
    let src = find(home, id)?;
    let body = tree(&src)?;
    let v = nmjson::parse(&body).map_err(|e| format!("the listing did not read: {e}"))?;
    let entries = v.array().map_err(|_| "the listing is not a list of files".to_owned())?;
    let mut files = Vec::new();
    for e in entries.iter().filter(|e| text(e, "type") == "file").take(MAX_FILES) {
        let path = text(e, "path");
        if is_archived_path(&path) {
            continue;
        }
        let lfs = e.get("lfs").ok();
        let sha = lfs.map(|l| text(l, "oid")).filter(|o| o.len() == 64).unwrap_or_default();
        let size = lfs.and_then(|l| l.get("size").and_then(Value::integer).ok()).or_else(|| e.get("size").and_then(Value::integer).ok()).unwrap_or(0);
        let (kind, role, mut why) = class_of(&src, &path);
        // A map already among yours is not fetched again (an import never replaces one).
        let name = path.rsplit('/').next().unwrap_or(&path);
        let present = role.as_deref().is_some_and(|r| home.join("library/parameter-maps").join(r).join(name).is_file());
        if present && why.is_none() {
            why = Some(format!("already among your {} maps (to swap it, import with replace on Parameters)", role.as_deref().unwrap_or("")));
        }
        let desc = describe_file(&path, kind, role.as_deref());
        files.push(obj(vec![
            ("present", Value::Bool(present)),
            ("path", s(&path)),
            ("bytes", n(size)),
            ("sha256", if sha.is_empty() { Value::Null } else { s(&sha) }),
            ("kind", s(kind)),
            ("role", role.as_deref().map_or(Value::Null, s)),
            ("description", s(&desc)),
            ("installable", Value::Bool(why.is_none())),
            ("why_not", why.as_deref().map_or(Value::Null, s)),
        ]));
    }
    let listing = obj(vec![("at_ms", n(nmtype::now_ms())), ("revision", s(&src.revision)), ("files", Value::Array(files.clone())), ("truncated", Value::Bool(entries.len() > MAX_FILES))]);
    let p = listing_file(home, id);
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    std::fs::write(&p, nmjson::pretty::pretty(&listing)).map_err(|e| format!("{}: {e}", p.display()))?;
    let _ = crate::controls::audit(home, "cartridge-source", vec![("action", s("browse")), ("source", s(id)), ("files", n(files.len() as u64)), ("by", s(by)), ("label", s(&format!("Cartridge source {} listed: {} files", src.repo, files.len())))]);
    Ok(listing)
}

fn want_for_entry(home: &Path, src: &Source, entry: &Value) -> Result<Want, String> {
    let path = text(entry, "path");
    if path.split('/').any(|seg| seg == ".." || seg.is_empty()) || path.contains('\\') {
        return Err(format!("{path}: not a plain path in the repository"));
    }
    let name = path.rsplit('/').next().unwrap_or(&path);
    let (sub, then) = match text(entry, "kind").as_str() {
        "parameter map" => {
            let role = text(entry, "role");
            (format!("maps-{role}"), Then::ImportMap(role))
        }
        _ => ("stacks".to_owned(), Then::ImportStack),
    };
    let dest = home.join("incoming").join(&src.id).join(sub).join(name);
    if dest.is_file() {
        let _ = std::fs::remove_file(&dest);
    }
    Ok(Want {
        dest,
        url: format!("{}/resolve/{}/{path}", base_url(src), src.revision),
        sha: text(entry, "sha256"),
        label: format!("{}:{path}", src.repo),
        auth: src.token,
        then,
    })
}

/// Fetches one listed file of a source and installs it through its checks, in the background.
/// Returns the download's line.
/// # Errors
/// The air gap, no such source or listing, a file not listed or not installable, or a download running.
pub fn fetch(home: &Path, id: &str, path: &str, by: &str) -> Result<String, String> {
    let src = find(home, id)?;
    let listing = cached(home, id);
    let entry = listing
        .get("files")
        .and_then(Value::array)
        .map_err(|_| "browse the source first: its files are listed before one is fetched".to_owned())?
        .iter()
        .find(|f| text(f, "path") == path)
        .cloned()
        .ok_or_else(|| format!("{path}: not in the source's last listing (browse it again)"))?;
    if !matches!(entry.get("installable"), Ok(Value::Bool(true))) {
        return Err(format!("{path}: {}", text(&entry, "why_not")));
    }
    let want = want_for_entry(home, &src, &entry)?;
    crate::fetch::start_wants(home, &format!("source:{id}"), vec![want], "the libraries on this machine, through their checks", by)
}

fn is_serving_map_name(name: &str) -> bool {
    name.starts_with("job-filter-") || name == "super-weights.json"
}

/// Fetches all missing installable files from a source (browsing it first if needed) and installs
/// each through its checks in the background. `only` optionally restricts the batch: `"serving"`
/// (job filters, super-weights and surgery probes), `"parameter-maps"`, or `"stacks"`.
/// # Errors
/// The air gap, no such source, browse failing, nothing new to fetch, or a download running.
pub fn fetch_all(home: &Path, id: &str, only: Option<&str>, by: &str) -> Result<String, String> {
    if crate::controls::air_gap_on(home) {
        return Err("the air gap is on: nothing is downloaded until it is turned off".into());
    }
    let src = find(home, id)?;
    let listing = browse(home, id, by)?;
    let entries = listing.get("files").and_then(Value::array).map_err(|_| "the listing has no files".to_owned())?;
    let mut wants = Vec::new();
    for entry in entries {
        if !matches!(entry.get("installable"), Ok(Value::Bool(true))) {
            continue;
        }
        let kind = text(entry, "kind");
        let path = text(entry, "path");
        let name = path.rsplit('/').next().unwrap_or(&path);
        let keep = match only.unwrap_or("") {
            "" => true,
            "serving" => kind == "parameter map" && is_serving_map_name(name),
            "parameter-maps" => kind == "parameter map",
            "stacks" => kind == "stack file",
            other => return Err(format!("only {other}: use serving, parameter-maps or stacks")),
        };
        if keep {
            wants.push(want_for_entry(home, &src, entry)?);
        }
    }
    if wants.is_empty() {
        return Err("every matching file from this source is already on disk".into());
    }
    let label = match only.unwrap_or("") {
        "serving" => format!("source:{id}:serving"),
        "parameter-maps" => format!("source:{id}:maps"),
        "stacks" => format!("source:{id}:stacks"),
        _ => format!("source:{id}:all"),
    };
    crate::fetch::start_wants(home, &label, wants, "the libraries on this machine, through their checks", by)
}

/// Pure-Rust RFC 4648 standard base64 encoding (no outside crates).
fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = u32::from(chunk[0]);
        let b1 = u32::from(*chunk.get(1).unwrap_or(&0));
        let b2 = u32::from(*chunk.get(2).unwrap_or(&0));
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(char::from(TABLE[((n >> 18) & 0x3f) as usize]));
        out.push(char::from(TABLE[((n >> 12) & 0x3f) as usize]));
        out.push(if chunk.len() > 1 { char::from(TABLE[((n >> 6) & 0x3f) as usize]) } else { '=' });
        out.push(if chunk.len() > 2 { char::from(TABLE[(n & 0x3f) as usize]) } else { '=' });
    }
    out
}

fn slug_of(raw: &str) -> String {
    let mut out = String::new();
    let mut dash = false;
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            dash = false;
        } else if !dash && !out.is_empty() {
            out.push('-');
            dash = true;
        }
    }
    out.trim_end_matches('-').to_owned()
}

fn post_ndjson(url: &str, token: &str, body: &[u8]) -> Result<(u16, Vec<u8>), String> {
    if nmffi::http::available() {
        return nmffi::http::post(url, Some(token), "huggingface.co", "application/x-ndjson", body, 120_000);
    }
    let stamp = format!("{}-{}", std::process::id(), nmtype::now_ms());
    let hdr = std::env::temp_dir().join(format!("nano-hf-pub-h-{stamp}.txt"));
    let dat = std::env::temp_dir().join(format!("nano-hf-pub-b-{stamp}.ndjson"));
    std::fs::write(&hdr, format!("Authorization: Bearer {token}")).map_err(|e| format!("{}: {e}", hdr.display()))?;
    std::fs::write(&dat, body).map_err(|e| {
        let _ = std::fs::remove_file(&hdr);
        format!("{}: {e}", dat.display())
    })?;
    let out = std::process::Command::new("curl.exe")
        .args(["--silent", "--show-error", "--max-time", "120", "-X", "POST", "-H", "Content-Type: application/x-ndjson", "-H"])
        .arg(format!("@{}", hdr.display()))
        .arg("--data-binary")
        .arg(format!("@{}", dat.display()))
        .args(["-w", "\n%{http_code}", url])
        .output();
    let _ = std::fs::remove_file(&hdr);
    let _ = std::fs::remove_file(&dat);
    let out = out.map_err(|e| format!("curl.exe could not start: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_owned());
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let (body_part, code_part) = text.rsplit_once('\n').unwrap_or(("", text.as_ref()));
    let code = code_part.trim().parse::<u16>().map_err(|_| format!("unexpected HTTP status from Hugging Face: {}", code_part.trim()))?;
    Ok((code, body_part.as_bytes().to_vec()))
}

/// Parameters for publishing a local parameter map or stack file to a Hugging Face repository.
pub struct PublishRequest<'a> {
    pub repo: &'a str,
    pub kind: Option<&'a str>,
    pub revision: Option<&'a str>,
    pub create_pr: bool,
    pub item_kind: &'a str,
    pub role: Option<&'a str>,
    pub item: &'a str,
    pub summary: Option<&'a str>,
}

/// Publishes one local parameter map or non-base stack to a Hugging Face repository (directly on a
/// branch or as a Pull Request when `create_pr` is true). Blocked by the air gap; requires the
/// user's Hugging Face token (`HF_TOKEN` or `hf auth login`), sent only to huggingface.co.
/// # Errors
/// Air gap on, missing token, invalid repo/item, base cartridge refused, file over 10 MiB direct
/// upload limit, or Hugging Face refusing the commit.
pub fn publish(home: &Path, req: &PublishRequest<'_>, by: &str) -> Result<Value, String> {
    if crate::controls::air_gap_on(home) {
        return Err("the air gap is on: nothing is published until it is turned off".into());
    }
    let repo = req.repo.trim().trim_start_matches("https://huggingface.co/").trim_start_matches("datasets/").trim_end_matches('/').to_owned();
    if !repo_ok(&repo) {
        return Err(format!("{repo}: give owner/name as Hugging Face writes it"));
    }
    let repo_kind = if req.kind == Some("dataset") { "dataset" } else { "model" };
    let rev = req.revision.map(str::trim).filter(|r| !r.is_empty()).unwrap_or("main");
    if rev.len() > 96 || !rev.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/')) {
        return Err(format!("{rev}: not a revision name"));
    }
    let token = crate::fetch::token().ok_or_else(|| "no Hugging Face token found: set HF_TOKEN or run `hf auth login` first".to_owned())?;

    let (bytes, remote_path, local_hint, default_summary) = match req.item_kind {
        "parameter-map" => {
            let role = req.role.map(str::trim).unwrap_or("");
            if !ROLES.contains(&role) {
                return Err(format!("{role}: not a role (reader, writer, embedder or reranker)"));
            }
            let file = req.item.trim();
            if file.is_empty() || file.contains(['/', '\\']) || file.starts_with('.') || file == nmapi::lineage::FILE || !file.ends_with(".json") {
                return Err(format!("{file}: give a .json parameter map file name in library/parameter-maps/{role}/"));
            }
            let local = home.join("library/parameter-maps").join(role).join(file);
            let md = std::fs::metadata(&local).map_err(|e| format!("{}: {e}", local.display()))?;
            if !md.is_file() {
                return Err(format!("{}: not a file", local.display()));
            }
            let remote = format!("parameter-maps/{role}/{file}");
            if md.len() > PUBLISH_DIRECT_LIMIT {
                let pr_flag = if req.create_pr { " --create-pr" } else { "" };
                return Err(format!(
                    "{role}/{file} is {} bytes (over the 10 MiB direct browser/API upload limit). Upload it with Xet/LFS instead:\n  hf upload {repo} \"{}\" \"{remote}\"{pr_flag}",
                    md.len(),
                    local.display()
                ));
            }
            let data = std::fs::read(&local).map_err(|e| format!("{}: {e}", local.display()))?;
            crate::pmaps::check_import(file, &data, None)?;
            (data, remote, local.display().to_string(), format!("Add {role} parameter map {file}"))
        }
        "stack" => {
            let key = req.item.trim();
            if key.is_empty() {
                return Err("item: give a stack key (id@version)".into());
            }
            let lib = nmlearn::library::Library::open(home).map_err(|e| e.to_string())?;
            let st = lib.stack(key).ok_or_else(|| format!("no stack {key}"))?;
            if st.cartridges.iter().any(|k| lib.cartridge(k).is_some_and(is_base_cartridge)) {
                return Err(format!("{key}: contains a base cartridge; upload base cartridges via `hf upload`"));
            }
            let data = lib.export_stack(&st.key()).map_err(|e| e.to_string())?;
            let slug = slug_of(&st.key());
            if slug.is_empty() {
                return Err(format!("{key}: stack name cannot be turned into a file name"));
            }
            let remote = format!("stacks/{slug}.stack.json");
            if data.len() as u64 > PUBLISH_DIRECT_LIMIT {
                return Err(format!("{key} export is {} bytes (over the 10 MiB direct upload limit); run `neuromorph-nano library export {key} --out {slug}.stack.json` and upload with `hf upload`", data.len()));
            }
            (data, remote, key.to_owned(), format!("Add stack {}", st.key()))
        }
        other => return Err(format!("item_kind {other}: use parameter-map or stack")),
    };

    let summary = req.summary.map(str::trim).filter(|x| !x.is_empty()).map_or(default_summary, str::to_owned);
    let sha = nmhash::sha256_bytes(&bytes);
    let header_line = obj(vec![
        ("key", s("header")),
        ("value", obj(vec![("summary", s(&summary)), ("description", s(&format!("Published from Neuromorph Nano (SHA-256: {sha})")))])),
    ])
    .encode()
    .map_err(|e| format!("encode header: {e}"))?;
    let file_line = obj(vec![
        ("key", s("file")),
        ("value", obj(vec![("content", s(&base64_encode(&bytes))), ("path", s(&remote_path)), ("encoding", s("base64"))])),
    ])
    .encode()
    .map_err(|e| format!("encode file: {e}"))?;
    let payload = format!("{header_line}\n{file_line}\n");

    let api_kind = if repo_kind == "dataset" { "datasets" } else { "models" };
    let query = if req.create_pr { "?create_pr=1" } else { "" };
    let url = format!("https://huggingface.co/api/{api_kind}/{repo}/commit/{rev}{query}");
    let (status, reply) = post_ndjson(&url, &token, payload.as_bytes())?;
    let reply_text = String::from_utf8_lossy(&reply);
    if !(200..300).contains(&status) {
        let hint = match status {
            401 => " (check HF_TOKEN or run `hf auth login` with a write token)",
            403 => " (no write permission on this branch; enable 'open as Pull Request' or publish to your own repo)",
            404 => " (repository or revision not found on Hugging Face)",
            _ => "",
        };
        let detail = reply_text.trim();
        let detail = if detail.len() > 240 { &detail[..240] } else { detail };
        return Err(format!("Hugging Face returned HTTP {status}{hint}: {detail}"));
    }
    let parsed = nmjson::parse(&reply).ok();
    let published_url = parsed
        .as_ref()
        .and_then(|v| {
            v.get("pullRequestUrl")
                .or_else(|_| v.get("commitUrl"))
                .and_then(Value::string)
                .ok()
                .map(str::to_owned)
        })
        .unwrap_or_else(|| format!("https://huggingface.co/{repo}/blob/{rev}/{remote_path}"));

    let audit = crate::controls::audit(
        home,
        "cartridge-source",
        vec![
            ("action", s("publish")),
            ("repo", s(&repo)),
            ("kind", s(repo_kind)),
            ("revision", s(rev)),
            ("create_pr", Value::Bool(req.create_pr)),
            ("item_kind", s(req.item_kind)),
            ("source", s(&local_hint)),
            ("path", s(&remote_path)),
            ("bytes", n(bytes.len() as u64)),
            ("sha256", s(&sha)),
            ("url", s(&published_url)),
            ("by", s(by)),
            ("label", s(&format!("Published {remote_path} to {repo}{}", if req.create_pr { " (PR)" } else { "" }))),
        ],
    )?;
    Ok(obj(vec![
        ("repo", s(&repo)),
        ("path", s(&remote_path)),
        ("bytes", n(bytes.len() as u64)),
        ("sha256", s(&sha)),
        ("create_pr", Value::Bool(req.create_pr)),
        ("url", s(&published_url)),
        ("audit_id", s(&audit)),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Home(PathBuf);
    impl Drop for Home {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn sources_add_list_and_remove_with_undo() {
        let home = Home(std::env::temp_dir().join(format!("nm-sources-{}", std::process::id())));
        std::fs::create_dir_all(home.0.join("memory")).unwrap();
        assert_eq!(list(&home.0).len(), 1, "the official library is always there");
        assert!(add(&home.0, "not a repo", None, None, false, "test").is_err());
        assert!(add(&home.0, crate::fetch::CARTRIDGE_REPO, None, None, false, "test").is_err(), "the official one is built in");
        let a = add(&home.0, "https://huggingface.co/someone/their-cartridges/", None, Some("v2"), false, "test").unwrap();
        assert!(!a.is_empty());
        assert!(add(&home.0, "someone/their-cartridges", None, None, false, "test").is_err(), "added once");
        let l = list(&home.0);
        assert_eq!(l.len(), 2);
        assert_eq!((l[1].id.as_str(), l[1].revision.as_str(), l[1].token), ("someone--their-cartridges", "v2", false));
        assert!(remove(&home.0, OFFICIAL, "test").is_err());
        remove(&home.0, "someone--their-cartridges", "test").unwrap();
        assert_eq!(list(&home.0).len(), 1);
        let log = std::fs::read_to_string(home.0.join("memory/audit.jsonl")).unwrap();
        let row = nmjson::parse(log.lines().last().unwrap().as_bytes()).unwrap();
        undo(&home.0, &row, "x", "test").unwrap();
        assert_eq!(list(&home.0).len(), 2, "the remove is undone");
    }

    #[test]
    fn files_are_classed_by_what_installs() {
        let third = Source { id: "a--b".into(), repo: "a/b".into(), kind: "model".into(), revision: "main".into(), official: false, token: false, added_ms: 0 };
        assert_eq!(class_of(&third, "base/reader/reader-base.cartridge").0, "base cartridge");
        assert!(class_of(&third, "base/reader/reader-base.cartridge").2.is_some(), "no third-party base cartridges");
        assert_eq!(class_of(&third, "parameter-maps/reranker/job-filter-x.json").1.as_deref(), Some("reranker"));
        assert!(class_of(&third, "parameter-maps/job-filter-x.json").2.is_some(), "a map needs its role");
        assert!(class_of(&third, "stacks/fast-code.json").2.is_none());
        assert!(class_of(&third, "README.md").2.is_some());
        assert!(class_of(&third, "config.json").2.is_some(), "a model's config is not a stack");
        assert!(class_of(&third, "my-fast-code.stack.json").2.is_none());
        assert!(is_archived_path("archived/base/reader/reader-base-v2.cartridge"));
        assert!(is_archived_path("archived/parameter-maps/reader/job-filter-read.json"));
        assert!(is_archived_path("archived/stacks/gpu-old-v1.stack.json"));
        assert!(!is_archived_path("parameter-maps/reader/job-filter-read.json"));
        assert_eq!(class_of(&third, "archived/parameter-maps/reranker/job-filter-x.json").0, "archived");
        assert!(class_of(&third, "archived/stacks/fast-code.stack.json").2.is_some());
    }

    #[test]
    fn official_source_is_public_and_base64_matches_rfc4648() {
        assert!(!official().token, "official public library sends no token on browse/fetch");
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        assert_eq!(slug_of("gpu-nvidia-geforce-gtx-1650-4096mib@3"), "gpu-nvidia-geforce-gtx-1650-4096mib-3");
    }
}
