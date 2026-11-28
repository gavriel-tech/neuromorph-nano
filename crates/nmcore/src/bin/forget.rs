//! Deleting with care (2026-11-26, Luis Jake Gabriel III: "make sure the deleting anything has
//! verifications for multi steps to prevent accidental deleting"), and the data controls asked for
//! with it: forget a project, delete a single memory, trim the audit log.
//!
//! Every destructive request of the interface's API (and of MCP, which borrows the same routes)
//! goes through `gate` in two steps. The first request deletes nothing: it answers with a preview
//! of exactly what would go (items, sizes, whether it can be undone), a single-use confirmation
//! token that expires after two minutes, and the word the interface asks the user to type. Only
//! the same request sent again with that token deletes. A token is bound to its request (path and
//! body), so it cannot confirm anything else.

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
fn text(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::string).ok().map(str::to_owned)
}

/// How long a confirmation token stays good.
pub const TOKEN_MS: u64 = 120_000;
/// Requests waiting for their confirmation: token, then (the request's fingerprint, expiry).
static PENDING: Mutex<BTreeMap<String, (String, u64)>> = Mutex::new(BTreeMap::new());
/// Memory deletions a running server applies (it holds the memory journal).
pub const REQUESTS: &str = "forget-requests.jsonl";

/// Whether a write request deletes something.
#[must_use]
pub fn destructive(path: &str, v: &Value) -> bool {
    matches!(
        path,
        "/api/storage/delete"
            | "/api/storage/prune"
            | "/api/dense/clear"
            | "/api/ledger/clear"
            | "/api/examples/clear"
            | "/api/cartridges/bases/remove"
            | "/api/parameter-maps/remove"
            | "/api/projects/forget"
            | "/api/memory/forget"
            | "/api/audit/retention"
            | "/api/comprehension/refresh"
            | "/api/preferences/remove"
    ) || (path == "/api/surgery/job-filter" && text(v, "action").as_deref() == Some("delete"))
        // 2026-11-26 (Phase B): unregistering a learning unit and removing a project lens.
        || (path == "/api/units" && text(v, "action").as_deref() == Some("unregister"))
        || (path == "/api/lenses" && text(v, "action").as_deref() == Some("remove"))
        // 2026-11-26: removing a cartridge source.
        || (path == "/api/cartridge-sources" && text(v, "action").as_deref() == Some("remove"))
        // Removing a reminder, loop or step of the agent's guidance.
        || (path == "/api/guidance" && text(v, "action").as_deref() == Some("remove"))
        // Removing a custom prime or deleting a user-saved prime preset.
        || (path == "/api/primes" && matches!(text(v, "action").as_deref(), Some("remove" | "preset_delete")))
        // Clearing a parameter overlay (every entry out; undone from Audit). Its preview passes.
        || (path == "/api/parameters/overlay" && text(v, "action").as_deref() == Some("clear") && !matches!(v.get("preview"), Ok(Value::Bool(true))))
}

/// The request as the token binds it: the path and the body without its confirmation fields.
fn fingerprint(path: &str, v: &Value) -> String {
    let mut m = match v {
        Value::Object(m) => m.clone(),
        _ => BTreeMap::new(),
    };
    m.remove("confirm");
    m.remove("confirm_token");
    format!("{path} {}", Value::Object(m).encode().unwrap_or_default())
}

/// The two-step gate. A destructive request without `confirm_token` gets `Some(preview)` and
/// nothing happens; with a good token it gets `None` (go ahead) and `confirm: true` is set in the
/// body. A request that deletes nothing passes untouched.
/// # Errors
/// The preview cannot be made (a refused or unknown target), or the token is unknown, used,
/// expired or made for another request.
pub fn gate(home: &Path, path: &str, v: &mut Value) -> Result<Option<Value>, String> {
    if !destructive(path, v) {
        return Ok(None);
    }
    let fp = fingerprint(path, v);
    let now = nmtype::now_ms();
    let Some(token) = text(v, "confirm_token") else {
        let (preview, word) = preview_of(home, path, v)?;
        let token = crate::network::random_hex();
        let expires = now + TOKEN_MS;
        if let Ok(mut p) = PENDING.lock() {
            p.retain(|_, (_, e)| *e >= now);
            p.insert(token.clone(), (fp, expires));
        }
        return Ok(Some(obj(vec![
            ("step", s("preview")),
            ("deleted", Value::Bool(false)),
            ("preview", preview),
            ("confirm_token", s(&token)),
            ("expires_ms", n(expires)),
            ("confirm_word", s(&word)),
            ("next", s("Nothing has been deleted. To delete, send the same request again with this confirm_token within two minutes.")),
        ])));
    };
    let held = PENDING.lock().map_err(|_| "confirmation state poisoned".to_owned())?.remove(&token);
    match held {
        Some((f, e)) if f == fp && e >= now => {}
        Some((f, _)) if f != fp => return Err("confirm_token: made for another request; nothing was deleted; ask for a new preview".into()),
        Some(_) => return Err("confirm_token: expired; nothing was deleted; ask for a new preview".into()),
        None => return Err("confirm_token: unknown or already used; nothing was deleted; ask for a new preview".into()),
    }
    if let Value::Object(m) = v {
        m.insert("confirm".into(), Value::Bool(true));
    }
    Ok(None)
}

fn item(label: &str, path: Option<&Path>, bytes: Option<u64>, files: Option<u64>) -> Value {
    obj(vec![
        ("label", s(label)),
        ("path", path.map_or(Value::Null, |p| s(&p.display().to_string()))),
        ("bytes", bytes.map_or(Value::Null, n)),
        ("files", files.map_or(Value::Null, n)),
    ])
}

fn preview(title: &str, items: Vec<Value>, bytes: Option<u64>, files: Option<u64>, undo: &str, note: Option<&str>) -> Value {
    obj(vec![
        ("title", s(title)),
        ("items", Value::Array(items)),
        ("bytes", bytes.map_or(Value::Null, n)),
        ("files", files.map_or(Value::Null, n)),
        ("undo", s(undo)),
        ("note", note.map_or(Value::Null, s)),
    ])
}

/// Bytes and files under a path (a file counts as one).
fn size_of(p: &Path) -> (u64, u64) {
    let Ok(md) = std::fs::metadata(p) else { return (0, 0) };
    if md.is_file() {
        return (md.len(), 1);
    }
    let (mut b, mut f) = (0, 0);
    if let Ok(rd) = std::fs::read_dir(p) {
        for e in rd.flatten() {
            let (x, y) = size_of(&e.path());
            b += x;
            f += y;
        }
    }
    (b, f)
}

fn field<'a>(v: &'a Value, k: &str) -> Result<&'a str, String> {
    v.get(k).and_then(Value::string).map_err(|e| format!("{k}: {e}"))
}

/// A project's memory folder by its key (`nano` for the Nano's own).
fn memory_of(home: &Path, key: &str) -> Result<PathBuf, String> {
    crate::ledger::project_memory(home, &format!("project={key}")).map(|(_, m)| m).ok_or_else(|| format!("project: no project {key}"))
}

const CANNOT: &str = "cannot be undone";

/// The preview of one destructive request and the word the user types to confirm it.
fn preview_of(home: &Path, path: &str, v: &Value) -> Result<(Value, String), String> {
    let delete = "delete".to_owned();
    Ok(match path {
        "/api/storage/delete" => {
            let p = nmapi::storage::delete_plan(home, field(v, "path")?)?;
            let shown = text(&p, "path").unwrap_or_default();
            let label = text(&p, "label").unwrap_or_default();
            let (b, f) = (p.get("bytes").and_then(Value::integer).ok(), p.get("files").and_then(Value::integer).ok());
            let word = Path::new(&shown).file_name().map_or_else(|| delete.clone(), |x| x.to_string_lossy().into_owned());
            (preview(&format!("Delete {label}"), vec![item(&label, Some(Path::new(&shown)), b, f)], b, f, CANNOT, None), word)
        }
        "/api/storage/prune" => {
            let p = nmapi::storage::prune_plan(home, field(v, "category")?)?;
            let items: Vec<Value> = p.get("items").and_then(Value::array).map(<[Value]>::to_vec).unwrap_or_default();
            let skipped = p.get("skipped_in_use").and_then(Value::array).map_or(0, <[Value]>::len);
            let note = (skipped > 0).then(|| format!("{skipped} items changed in the last ten minutes may be in use and are kept"));
            let junk = text(&p, "category").as_deref() == Some("junk");
            let title = if junk { "Delete the junk".to_owned() } else { format!("Prune the rebuildable {}", text(&p, "name").unwrap_or_default()) };
            let undo = if junk { "cannot be undone; nothing uses these files" } else { "cannot be undone; the Nano makes these files again when it needs them" };
            let (b, f) = (p.get("bytes").and_then(Value::integer).ok(), p.get("files").and_then(Value::integer).ok());
            (preview(&title, items, b, f, undo, note.as_deref()), delete)
        }
        "/api/dense/clear" => {
            let key = field(v, "project")?;
            let dir = memory_of(home, key)?.join(nmsubs::dense::store::DIR);
            let (rows, bytes) = nmsubs::dense::store::disk_usage(&dir).unwrap_or((0, 0));
            let label = format!("{rows} vectors of the meaning index");
            (preview(&format!("Clear the meaning index of {key}"), vec![item(&label, Some(&dir), Some(bytes), None)], Some(bytes), None, "cannot be undone; the next start embeds the material again", None), delete)
        }
        "/api/ledger/clear" => {
            let key = field(v, "project")?;
            let file = crate::ledger::ledger_file(&memory_of(home, key)?);
            let rows = std::fs::read_to_string(&file).map_or(0, |t| t.lines().filter(|l| !l.trim().is_empty()).count() as u64);
            let (b, _) = size_of(&file);
            (preview(&format!("Clear the evidence ledger of {key}"), vec![item(&format!("{rows} ledger rows"), Some(&file), Some(b), Some(1))], Some(b), Some(1), CANNOT, Some("packs served before now are not judged again")), delete)
        }
        "/api/examples/clear" => {
            let key = field(v, "project")?;
            let memory = memory_of(home, key)?;
            let ex = nmtype::examples::dir(&memory);
            let rows: usize = [nmtype::examples::EXAMPLES, nmtype::examples::PREFERENCES, nmtype::examples::JUDGED, nmtype::examples::PASSAGES]
                .iter()
                .map(|f| nmtype::examples::read_rows(&memory, f).map_or(0, |r| r.len()))
                .sum();
            let (b, f) = size_of(&ex);
            (preview(&format!("Clear the training store of {key}"), vec![item(&format!("{rows} training rows"), Some(&ex), Some(b), Some(f))], Some(b), Some(f), CANNOT, Some("export it first (Your training data) if you may want it back")), delete)
        }
        "/api/cartridges/bases/remove" => {
            let key = field(v, "key")?;
            let all = crate::library::base_versions(home);
            let found = all
                .get("models")
                .and_then(Value::array)
                .unwrap_or_default()
                .iter()
                .flat_map(|m| m.get("versions").and_then(Value::array).map(<[Value]>::to_vec).unwrap_or_default())
                .find(|x| text(x, "key").as_deref() == Some(key))
                .ok_or_else(|| format!("{key}: not a base cartridge in the cartridge library"))?;
            if matches!(found.get("serving"), Ok(Value::Bool(true))) {
                return Err(format!("{key} serves now; choose another version before removing it"));
            }
            let b = found.get("bytes").and_then(Value::integer).ok();
            (preview(&format!("Remove the base cartridge {key}"), vec![item(key, None, b, None)], b, None, "cannot be undone; download it again to have it back", None), key.to_owned())
        }
        "/api/parameter-maps/remove" if v.get("files").is_ok() || matches!(v.get("archived"), Ok(Value::Bool(true))) => {
            let role = field(v, "role")?;
            let files = crate::pmaps::files_of(v)?;
            let archived = matches!(v.get("archived"), Ok(Value::Bool(true)));
            let plan = crate::pmaps::remove_plan(home, role, &files, archived)?;
            let b: u64 = plan.iter().map(|(_, x)| x).sum();
            let items = plan.iter().map(|(p, x)| item(&p.file_name().map_or_else(String::new, |f| f.to_string_lossy().into_owned()), Some(p), Some(*x), Some(1))).collect();
            (preview(&format!("Delete {} {role} {}{}", files.len(), if files.len() == 1 { "map" } else { "maps" }, if archived { " from the archive" } else { "" }), items, Some(b), Some(files.len() as u64), CANNOT, Some("archiving keeps them out of the way and can be undone; deleting cannot")), format!("{role}:{}", files.len()))
        }
        "/api/parameter-maps/remove" => {
            let (role, file) = (field(v, "role")?, field(v, "file")?);
            if !["reader", "writer", "embedder", "reranker"].contains(&role) || file.contains(['/', '\\']) || file.starts_with('.') || file.is_empty() {
                return Err(format!("{role}/{file}: not a parameter map"));
            }
            let p = home.join("library/parameter-maps").join(role).join(file);
            let (b, _) = size_of(&p);
            if !p.is_file() {
                return Err(format!("{file}: no such map"));
            }
            (preview(&format!("Remove the {role} parameter map {file}"), vec![item(file, Some(&p), Some(b), Some(1))], Some(b), Some(1), CANNOT, None), file.to_owned())
        }
        "/api/surgery/job-filter" => {
            let (role, job) = (field(v, "role")?, field(v, "job")?);
            let p = home.join("library/parameter-maps").join(role).join(format!("job-filter-{job}.json"));
            let (b, _) = size_of(&p);
            (preview(&format!("Remove the {role} job filter {job}"), vec![item(&format!("{role}/{job}"), Some(&p), Some(b), Some(1))], Some(b), Some(1), "can be undone in Audit (the filter is kept aside)", None), delete)
        }
        "/api/projects/forget" => {
            let key = field(v, "project")?;
            let (dir, b, f) = forget_project_plan(home, key)?;
            (
                preview(
                    &format!("Forget the project {key}"),
                    vec![item("everything the Nano keeps for it: memory, world-model notes, meaning index, evidence ledger, training store, learning state", Some(&dir), Some(b), Some(f))],
                    Some(b),
                    Some(f),
                    CANNOT,
                    Some("the project's own files are not touched; the Nano learns it from scratch if you use it again"),
                ),
                key.to_owned(),
            )
        }
        "/api/memory/forget" => {
            let (key, id) = (field(v, "project")?, field(v, "id")?);
            let found = memory_items_of(&memory_of(home, key)?).into_iter().find(|x| text(x, "id").as_deref() == Some(id)).ok_or_else(|| format!("{id}: no such memory in {key}"))?;
            let shown = text(&found, "text").unwrap_or_default();
            (
                preview(
                    "Delete this memory",
                    vec![item(&shown, None, None, None)],
                    None,
                    None,
                    "the memory leaves recall and every view at once; the memory journal keeps a record that it existed",
                    Some("to erase a project's memory completely, forget the project"),
                ),
                delete,
            )
        }
        "/api/audit/retention" => {
            let days = crate::controls::audit_keep_days(home);
            let r = nmmem::keep::pass_days(home, nmtype::now_ms(), false, days).map_err(|e| e.to_string())?;
            let file = home.join("memory").join("audit.jsonl");
            (
                preview(
                    &format!("Trim the audit log to {days} days of use"),
                    vec![item(&format!("{} tool rows older than {days} days of use", r.rows()), Some(&file), Some(r.bytes()), None)],
                    Some(r.bytes()),
                    None,
                    CANNOT,
                    Some("setting changes are always kept, so Undo keeps working"),
                ),
                delete,
            )
        }
        "/api/units" => {
            let id = field(v, "id")?;
            let units = nmlearn::Units::open(home);
            let card = units.get(id).ok_or_else(|| format!("{id}: no such learning unit"))?;
            if !card.files.is_empty() {
                return Err(format!("{id} has weight files; a fitted unit is retired, never removed"));
            }
            let dir = units.unit_dir(id);
            let (b, f) = size_of(&dir);
            (
                preview(&format!("Unregister the learning unit {id}"), vec![item(&format!("{id} (target {}, rank {}, status {})", card.target, card.rank, card.status), Some(&dir), Some(b), Some(f))], Some(b), Some(f), "can be undone in Audit (its card is registered again)", Some("an unfitted unit has no weights; only its registry entry goes")),
                id.to_owned(),
            )
        }
        "/api/guidance" => {
            let (title, line, word) = crate::guidance::remove_plan(home, v)?;
            (preview(&title, vec![item(&line, None, None, None)], None, None, "can be undone in Audit (the item is put back as it was)", None), word)
        }
        "/api/primes" => {
            let (title, word) = crate::primes_ctl::delete_preview(home, v)?;
            (preview(&title, vec![item(&title, None, None, None)], None, None, "can be undone in Audit (previous primes configuration and presets are restored)", None), word)
        }
        "/api/cartridge-sources" => {
            let id = field(v, "id")?;
            let (src, listing) = crate::sources::remove_plan(home, id)?;
            let mut items = vec![item(&format!("the source {}", src.repo), None, None, None)];
            if let Some(l) = &listing {
                items.push(item("its last listing", Some(l), std::fs::metadata(l).ok().map(|m| m.len()), Some(1)));
            }
            (preview(&format!("Remove the cartridge source {}", src.repo), items, None, Some(u64::from(listing.is_some())), "Audit can add it again; what was imported from it stays in the cartridge library on this machine", None), id.to_owned())
        }
        "/api/lenses" => {
            let id = field(v, "id")?;
            let version = v.get("version").and_then(Value::integer).ok();
            let files = crate::learned_units::lens_remove_plan(home, id, version)?;
            let b: u64 = files.iter().map(|(_, x)| x).sum();
            let items = files.iter().map(|(p, x)| item(&p.file_name().map_or_else(String::new, |f| f.to_string_lossy().into_owned()), Some(p), Some(*x), Some(1))).collect();
            let title = match version {
                Some(ver) => format!("Remove version {ver} of the lens {id}"),
                None => format!("Remove the lens {id} (every version)"),
            };
            (preview(&title, items, Some(b), Some(files.len() as u64), "cannot be undone; the idle fit makes a lens again from the project's material", None), id.to_owned())
        }
        "/api/comprehension/refresh" => {
            let (key, files, notes, served) = crate::memory_api::refresh_plan(home, field(v, "project")?)?;
            let b: u64 = files.iter().map(|(_, x)| x).sum();
            let items = files
                .iter()
                .enumerate()
                .map(|(i, (p, x))| item(&if i == 0 { format!("{notes} world-model notes (L1 to L4)") } else { "the older note store a start carries over".to_owned() }, Some(p), Some(*x), Some(1)))
                .collect();
            let note = if served { "a running Nano serves this project: it empties its notes and its idle writer writes them again" } else { "the idle writer writes the notes again the next time the project is served" };
            (preview(&format!("Refresh the world model of {key}"), items, Some(b), Some(files.len() as u64), "cannot be undone; the notes are written again from the material", Some(note)), delete)
        }
        "/api/preferences/remove" => crate::preferences::remove_preview(home, field(v, "id")?)?,
        "/api/parameters/overlay" => crate::overlay::clear_preview(home, field(v, "key")?)?,
        _ => return Err(format!("{path}: no preview")),
    })
}

/// A served project (its live file names it): forgetting it now would pull its memory away from a
/// running server.
fn served(home: &Path, key: &str) -> bool {
    crate::live::read_all(home).iter().any(|p| text(p, "project_key").as_deref() == Some(key) || text(p, "project").as_deref() == Some(key))
}

/// The folder a forget would remove, with its bytes and files, after every refusal.
fn forget_project_plan(home: &Path, key: &str) -> Result<(PathBuf, u64, u64), String> {
    if key == "nano" {
        return Err("project: the Nano's own memory holds the audit log and every setting's history; delete its items one by one instead".into());
    }
    let dir = memory_of(home, key)?;
    if served(home, key) {
        return Err(format!("{key} is being served by a running Nano; stop it first, then forget the project"));
    }
    let (b, f) = size_of(&dir);
    Ok((dir, b, f))
}

/// `POST /api/projects/forget {project}` (after its confirmation): removes everything the Nano
/// keeps for a project (its whole memory folder); audited.
/// # Errors
/// A refusal of `forget_project_plan`, the removal or the audit log failing.
pub fn forget_project(home: &Path, key: &str, by: &str) -> Result<Value, String> {
    let (dir, bytes, files) = forget_project_plan(home, key)?;
    std::fs::remove_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let audit = crate::controls::audit(
        home,
        "project-forget",
        vec![("project", s(key)), ("bytes", n(bytes)), ("files", n(files)), ("by", s(by)), ("label", s(&format!("Forgot project {key}: {files} files")))],
    )?;
    Ok(obj(vec![("project", s(key)), ("bytes", n(bytes)), ("files", n(files)), ("audit_id", s(&audit))]))
}

/// The memories of one memory folder as the tier views show them (`1-core`, `2-active`,
/// `3-archive`, one `items.json` per topic): read only, newest first.
fn memory_items_of(memory: &Path) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    for tier in ["1-core", "2-active", "3-archive"] {
        let Ok(rd) = std::fs::read_dir(memory.join(tier)) else { continue };
        for topic in rd.flatten().filter(|e| e.path().is_dir()) {
            let Some(v) = std::fs::read(topic.path().join("items.json")).ok().and_then(|b| nmjson::parse(&b).ok()) else { continue };
            for x in v.get("items").and_then(Value::array).unwrap_or_default() {
                let Some(id) = text(x, "id") else { continue };
                if text(x, "state").is_some_and(|st| st != "active") {
                    continue;
                }
                let full = text(x, "text").unwrap_or_default();
                let shown: String = full.chars().take(400).collect();
                out.push(obj(vec![
                    ("id", s(&id)),
                    ("tier", s(tier)),
                    ("topic", s(&text(x, "topic").unwrap_or_else(|| topic.file_name().to_string_lossy().into_owned()))),
                    ("text", s(&shown)),
                    ("state", s(&text(x, "state").unwrap_or_else(|| "active".into()))),
                    ("class", s(&text(x, "class").unwrap_or_default())),
                    ("recorded", x.get("recorded").and_then(Value::integer).map_or(Value::Null, n)),
                ]));
            }
        }
    }
    out.sort_by_key(|x| std::cmp::Reverse(x.get("recorded").and_then(Value::integer).unwrap_or(0)));
    out
}

/// `GET /api/memory/items?project=<key>[&limit=n][&q=text]`: the project's memories (read only).
#[must_use]
pub fn memory_items(home: &Path, query: &str) -> Value {
    let Some((key, memory)) = crate::ledger::project_memory(home, query) else {
        return obj(vec![("error", s("project: give the key of a project as /api/projects lists it, or nano"))]);
    };
    let limit = crate::project::query_arg(query, "limit").and_then(|l| l.parse::<usize>().ok()).unwrap_or(200).min(2000);
    let q = crate::project::query_arg(query, "q").map(|x| x.to_lowercase()).filter(|x| !x.is_empty());
    let all = memory_items_of(&memory);
    let matching: Vec<Value> = all.into_iter().filter(|x| q.as_ref().is_none_or(|q| text(x, "text").is_some_and(|t| t.to_lowercase().contains(q.as_str())))).collect();
    let total = matching.len() as u64;
    obj(vec![("project", s(&key)), ("items", Value::Array(matching.into_iter().take(limit).collect())), ("total", n(total))])
}

/// `POST /api/memory/forget {project, id}` (after its confirmation): when no server holds the
/// project's memory journal, the memory is deleted now (`Memory2::forget`); otherwise the request
/// waits in `forget-requests.jsonl` and the running server applies it within seconds
/// (`apply_requests`). Audited either way.
/// # Errors
/// An unknown project or memory, or a write failing.
pub fn forget_memory(home: &Path, key: &str, id: &str, by: &str) -> Result<Value, String> {
    let memory = memory_of(home, key)?;
    if !memory_items_of(&memory).iter().any(|x| text(x, "id").as_deref() == Some(id)) {
        return Err(format!("{id}: no such memory in {key}"));
    }
    let root = std::fs::read(memory.join("project.json")).ok().and_then(|b| nmjson::parse(&b).ok()).and_then(|v| text(&v, "root")).map_or_else(|| home.to_path_buf(), PathBuf::from);
    let applied = match nmmem::mem2::Memory2::open(&memory, &root) {
        Ok(mut m) => {
            m.forget(id).map_err(|e| e.to_string())?;
            "now"
        }
        Err(_) => {
            let line = obj(vec![("id", s(id)), ("at", n(nmtype::now_ms())), ("by", s(by))]).encode().map_err(|e| e.to_string())?;
            let path = memory.join(REQUESTS);
            let mut old = std::fs::read_to_string(&path).unwrap_or_default();
            old.push_str(&line);
            old.push('\n');
            std::fs::write(&path, old).map_err(|e| format!("{}: {e}", path.display()))?;
            "queued"
        }
    };
    let audit = crate::controls::audit(home, "memory-forget", vec![("project", s(key)), ("id", s(id)), ("applied", s(applied)), ("by", s(by)), ("label", s(&format!("Deleted a memory of {key}")))])?;
    let note = if applied == "now" {
        "Removed from what the Nano recalls and shows; the memory journal keeps a record that it existed."
    } else {
        "A running Nano holds this project's memory; it removes the memory within seconds. The memory journal keeps a record that it existed."
    };
    Ok(obj(vec![("project", s(key)), ("id", s(id)), ("applied", s(applied)), ("audit_id", s(&audit)), ("note", s(note))]))
}

/// The serving process applies waiting memory deletions with the journal it holds (its idle loop
/// calls this every few seconds). Returns how many were applied.
pub fn apply_requests(memory: &Path, m: &mut nmmem::mem2::Memory2) -> usize {
    let path = memory.join(REQUESTS);
    let Ok(raw) = std::fs::read_to_string(&path) else { return 0 };
    let _ = std::fs::remove_file(&path);
    let mut done = 0;
    for line in raw.lines().filter(|l| !l.trim().is_empty()) {
        if let Some(id) = nmjson::parse(line.as_bytes()).ok().and_then(|v| text(&v, "id"))
            && m.forget(&id).is_ok()
        {
            done += 1;
        }
    }
    done
}

/// `POST /api/audit/retention` (after its confirmation): trims the audit log's tool rows to the
/// `audit.keep_days` days of use; audited.
/// # Errors
/// The audit log cannot be rewritten.
pub fn retention(home: &Path, by: &str) -> Result<Value, String> {
    let days = crate::controls::audit_keep_days(home);
    let r = nmmem::keep::pass_days(home, nmtype::now_ms(), true, days).map_err(|e| e.to_string())?;
    let audit = crate::controls::audit(
        home,
        "retention",
        vec![
            ("rows", n(r.rows() as u64)),
            ("bytes_before", n(r.before)),
            ("bytes_after", n(r.after)),
            ("days", n(days)),
            ("by", s(by)),
            ("label", s(&format!("Audit log trimmed to {days} days of use: {} rows removed", r.rows()))),
        ],
    )?;
    Ok(obj(vec![("rows", n(r.rows() as u64)), ("bytes_before", n(r.before)), ("bytes_after", n(r.after)), ("days", n(days)), ("audit_id", s(&audit))]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(pairs: &[(&str, &str)]) -> Value {
        obj(pairs.iter().map(|(k, v)| (*k, s(v))).collect())
    }

    /// The gate: a destructive request first gets a preview and deletes nothing; its token
    /// confirms only that request, once; a request that deletes nothing passes.
    #[test]
    fn deletes_need_a_preview_then_their_own_token_once() {
        let home = std::env::temp_dir().join(format!("nano-gate-{}", std::process::id()));
        let maps = home.join("library/parameter-maps/writer");
        std::fs::create_dir_all(&maps).unwrap();
        std::fs::write(maps.join("pmap-x.json"), "{}").unwrap();
        let path = "/api/parameter-maps/remove";
        let mut first = body(&[("role", "writer"), ("file", "pmap-x.json")]);
        let reply = gate(&home, path, &mut first).unwrap().expect("a preview");
        assert_eq!(text(&reply, "step").as_deref(), Some("preview"));
        assert_eq!(text(&reply, "confirm_word").as_deref(), Some("pmap-x.json"));
        assert!(maps.join("pmap-x.json").is_file(), "the preview deletes nothing");
        let token = text(&reply, "confirm_token").unwrap();
        // The token does not confirm another request.
        let mut other = body(&[("role", "writer"), ("file", "pmap-y.json"), ("confirm_token", &token)]);
        assert!(gate(&home, path, &mut other).is_err());
        // Used once (even by the wrong request), it is gone.
        let mut again = body(&[("role", "writer"), ("file", "pmap-x.json"), ("confirm_token", &token)]);
        assert!(gate(&home, path, &mut again).is_err());
        // A fresh preview and its own token go ahead and set confirm.
        let mut second = body(&[("role", "writer"), ("file", "pmap-x.json")]);
        let token = text(&gate(&home, path, &mut second).unwrap().unwrap(), "confirm_token").unwrap();
        let mut ok = body(&[("role", "writer"), ("file", "pmap-x.json"), ("confirm_token", &token)]);
        assert!(gate(&home, path, &mut ok).unwrap().is_none());
        assert!(matches!(ok.get("confirm"), Ok(Value::Bool(true))));
        // A write that deletes nothing is not gated.
        let mut plain = body(&[("role", "writer"), ("job", "x"), ("action", "enable")]);
        assert!(gate(&home, "/api/surgery/job-filter", &mut plain).unwrap().is_none());
        assert!(destructive("/api/surgery/job-filter", &body(&[("action", "delete")])));
        let _ = std::fs::remove_dir_all(&home);
    }

    /// 2026-11-26 (Phase B): unregistering a unit, removing a lens and refreshing the world model
    /// are gated the same way: a preview naming what goes, then their own token; a unit with weight
    /// files and an unknown lens are refused at the preview; plugging a lens is not gated.
    #[test]
    fn phase_b_deletes_preview_then_confirm() {
        let home = std::env::temp_dir().join(format!("nano-gate-b-{}", std::process::id()));
        std::fs::create_dir_all(home.join("memory/world-model")).unwrap();
        std::fs::write(home.join("memory/world-model/notes.jsonl"), "{\"k\":\"note:a\",\"by\":\"w\",\"t\":\"x\"}\n").unwrap();
        let card = nmlearn::UnitCard::new("u-e2e", "qwen3-0.6b", "abc", "reranker", "nmengn.components#L27.down", 8, 8);
        nmlearn::Units::open(&home).register(card).unwrap();
        let lens = nmapi::adapter::shelf_file(&home, "embedder-lens-abc123", 1);
        std::fs::create_dir_all(lens.parent().unwrap()).unwrap();
        std::fs::write(&lens, b"lens bytes").unwrap();
        for (path, b) in [
            ("/api/units", body(&[("action", "unregister"), ("id", "u-e2e")])),
            ("/api/lenses", body(&[("action", "remove"), ("id", "embedder-lens-abc123")])),
            ("/api/comprehension/refresh", body(&[("project", "nano")])),
        ] {
            let mut first = b.clone();
            let reply = gate(&home, path, &mut first).unwrap().unwrap_or_else(|| panic!("{path}: a preview"));
            assert_eq!(text(&reply, "step").as_deref(), Some("preview"), "{path}");
            let items = reply.get("preview").and_then(|p| p.get("items")).and_then(Value::array).map_or(0, <[Value]>::len);
            assert!(items >= 1, "{path}: the preview names what goes");
            let token = text(&reply, "confirm_token").unwrap();
            let mut ok = b.clone();
            if let Value::Object(m) = &mut ok {
                m.insert("confirm_token".into(), s(&token));
            }
            assert!(gate(&home, path, &mut ok).unwrap().is_none(), "{path}: its own token goes ahead");
        }
        assert!(lens.is_file() && home.join("memory/world-model/notes.jsonl").is_file(), "the gate itself deletes nothing");
        let mut unknown = body(&[("action", "remove"), ("id", "embedder-lens-ffff")]);
        assert!(gate(&home, "/api/lenses", &mut unknown).is_err());
        let mut plug = body(&[("action", "disable"), ("id", "embedder-lens-abc123")]);
        assert!(gate(&home, "/api/lenses", &mut plug).unwrap().is_none(), "a plug is not a delete");
        let _ = std::fs::remove_dir_all(&home);
    }

    /// Forgetting is refused for the Nano's own memory and for an unknown project.
    #[test]
    fn forget_project_refusals() {
        let home = std::env::temp_dir().join(format!("nano-forget-{}", std::process::id()));
        std::fs::create_dir_all(home.join("memory")).unwrap();
        assert!(forget_project_plan(&home, "nano").is_err());
        assert!(forget_project_plan(&home, "no-such-project").is_err());
        let _ = std::fs::remove_dir_all(&home);
    }
}
