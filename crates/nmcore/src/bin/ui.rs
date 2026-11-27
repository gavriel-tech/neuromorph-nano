//! The interface server of the Neuromorph Nano: serves the Next.js static export in
//! `interface/out/` and a JSON API, on 127.0.0.1. `neuromorph-nano ui [--port <n>] [--dev]`
//! (default port 4870). Standard library only.
//!
//! Reads (GET, HEAD) are open to this machine. A write (POST to `/api/controls`, `/api/undo`,
//! `/api/cartridges`, `/api/dense/clear`, `/api/ledger/clear`, `/api/examples/clear`, `/api/pair/code` or `/api/devices/revoke`) must carry the server's local
//! token in the `X-Nano-Token` header, be JSON, and come from the interface's own origin (or no
//! origin, as from a local program). The token is made at start, held in memory only, and given
//! to the interface inside every HTML page it serves to this machine. With `--dev`, the Next.js
//! development server's origin (port 3000) is also accepted and may fetch the token from
//! `GET /api/token`; without it that endpoint answers 404. Requests naming a host other than this
//! machine are refused, so a page that rebinds its own name to 127.0.0.1 cannot read the token.
//! Every accepted write is audited.
//!
//! Local network (setting `network`, read at start; `off` is the default and changes nothing):
//! with `view` or `control` the server also listens on this machine's private addresses. A
//! request from another device is served only from a private or link-local address, only with a
//! Host header naming the address it arrived on and the port, and only with a paired device's
//! token (`X-Nano-Token` or the `nano_device` cookie); without one it gets the pairing page.
//! Such a device never receives the local token: its pages carry its own token. In `view` it can
//! read and change nothing; in `control` it can also change settings, audited with its name.
//! Switching the setting off refuses other devices at once. There is no encryption on the wire.
use crate::network;
use nmjson::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;

const DEFAULT_PORT: u16 = 4870;
/// Version of the JSON API (`/api/status` field `api`): 2 adds tool steps and states, models held
/// now, cartridge sizes and switching, the projects served now, devices and pairing; 3 adds step
/// clocks for every part (`timed`, `us`, `inside`, `cells`, `counts`), `/api/activity`,
/// `/api/memory/tiers` and each project's `tiers`; 4 adds `/api/session`, `/api/ledger` and
/// `/api/ledger/clear`; 5 adds the cartridge library: its cartridges as further rows of
/// `/api/cartridges` (with `library`), `/api/cartridges/library`, and library switches on
/// `POST /api/cartridges`; 6 adds `/api/outcomes`, nudges in `/api/session` (`nudges`) and
/// `/api/ledger` (`nudges`, `outcomes`, rows `nudged` and `outcome`), and the setting `nudges`, named primes, `primed` and `priming` since 7 (the earlier names still read); 7
/// adds `/api/examples`, `POST /api/examples/clear` and the setting `examples`. Within 7,
/// `/api/comprehension` gained `levels`, `routing` and `heads`, and `/api/stage-times` was added
/// (a page meets an older build as an absent field or a 404). Also within 7 (2026-11-26, Phase
/// B): `/api/models/weights`, `/api/parameter-maps/read|save|import`, `/api/cartridges/inspect|
/// values|parameters|trial|stack|import`, `/api/units`, `/api/lenses`, `/api/memory/add|pin|sweep`,
/// `/api/goals`, `/api/comprehension/refresh` and `/api/connect`; then `/api/guidance` (reminders,
/// loops and the step graph for the coding agent).
pub const API_VERSION: u64 = 7;
/// The request header that carries the local token (or a paired device's token) on a write.
const TOKEN_HEADER: &str = "x-nano-token";
/// The cookie that carries a paired device's token on page loads.
const DEVICE_COOKIE: &str = "nano_device";
/// Origins of the Next.js development server, accepted only with `--dev`.
const DEV_ORIGINS: [&str; 2] = ["http://localhost:3000", "http://127.0.0.1:3000"];

/// What every request handler needs: the Nano folder, the port, the token, the dev switch, the
/// private addresses listened on, and the devices already seen this run.
struct Server {
    home: PathBuf,
    port: u16,
    token: String,
    dev: bool,
    addrs: Vec<IpAddr>,
    seen: Mutex<BTreeSet<String>>,
}

/// A fresh 256-bit token as 64 hex digits (`network::random_hex`).
fn new_token() -> String {
    network::random_hex()
}

/// The page with the token placed in a `<meta name="nano-token">` right after `<head>`; a page
/// without a head is returned unchanged.
fn with_token(page: &[u8], token: &str) -> Vec<u8> {
    let Some(at) = page.windows(6).position(|w| w == b"<head>").map(|i| i + 6) else { return page.to_vec() };
    let meta = format!("<meta name=\"nano-token\" content=\"{token}\">");
    let mut out = Vec::with_capacity(page.len() + meta.len());
    out.extend_from_slice(page.get(..at).unwrap_or_default());
    out.extend_from_slice(meta.as_bytes());
    out.extend_from_slice(page.get(at..).unwrap_or_default());
    out
}

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}
fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect::<BTreeMap<_, _>>())
}
fn read_json(p: &Path) -> Option<Value> {
    std::fs::read(p).ok().and_then(|b| nmjson::parse(&b).ok())
}
fn json_files(dir: &Path) -> Vec<(String, Value)> {
    let mut out: Vec<(String, Value)> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
                .filter_map(|e| {
                    let name = e.path().file_stem()?.to_string_lossy().into_owned();
                    Some((name, read_json(&e.path())?))
                })
                .collect()
        })
        .unwrap_or_default();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}
fn dir_bytes(p: &Path) -> u64 {
    std::fs::read_dir(p)
        .map(|rd| {
            rd.flatten()
                .map(|e| match e.metadata() {
                    Ok(m) if m.is_dir() => dir_bytes(&e.path()),
                    Ok(m) => m.len(),
                    Err(_) => 0,
                })
                .sum()
        })
        .unwrap_or(0)
}

/// A precision width as JSON: the number of bits, or `"default"` when none is set.
fn width(w: Option<u8>) -> Value {
    w.map_or_else(|| s("default"), |b| n(u64::from(b)))
}

/// The first layer a component id names (`nmengn.components#L27.down` is 27).
fn layer_of(target: &str) -> Option<u64> {
    let rest = target.rsplit_once("#L")?.1;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// The learned adapter cartridges: the learning units in `units/` shown as cartridges, each by
/// its label, with its size, the model it is attached to, its class, the layers it touches and
/// whether its files match their recorded hashes. The released models are loaded whole and carry
/// no cartridge.
fn learned_cartridges(home: &Path) -> Value {
    let policy = nmlearn::cartridge::Policy::load(home);
    let units = nmlearn::Units::open(home);
    Value::Array(
        units
            .cards
            .iter()
            .map(|card| {
                let dir = units.unit_dir(&card.id);
                let label = nmlearn::cartridge::from_unit(card, &dir, &policy);
                let mut v = label.to_value();
                let bytes = dir_bytes(&dir);
                let verified = !card.files.is_empty()
                    && card.files.iter().all(|(p, h)| std::fs::read(dir.join(p)).is_ok_and(|b| nmhash::sha256_bytes(&b) == *h));
                if let Value::Object(o) = &mut v {
                    o.insert("bytes".into(), n(bytes));
                    // Loads happen in the serving process; this server reports loaded or not.
                    o.insert("loaded_bytes".into(), n(if card.loaded { bytes } else { 0 }));
                    o.insert("model".into(), s(&card.base));
                    o.insert("class".into(), s("learning"));
                    o.insert("layers".into(), layer_of(&card.target).map_or(Value::Null, |l| Value::Array(vec![n(l), n(l)])));
                    o.insert("verified".into(), Value::Bool(verified));
                    o.insert("switch_key".into(), s(&format!("{}{}", crate::controls::CARTRIDGE_KEY, card.id)));
                }
                v
            })
            .collect(),
    )
}

/// The learned adapter cartridges, then the cartridge library's cartridges (each with `library`).
fn all_cartridges(home: &Path) -> Value {
    let mut v = learned_cartridges(home);
    if let Value::Array(a) = &mut v {
        a.extend(crate::library::api_rows(home));
    }
    v
}

/// Layer, head and head-group counts of a model from its `files/config.json`.
fn model_shape(dir: &Path) -> Vec<(&'static str, Value)> {
    let Some(c) = read_json(&dir.join("files").join("config.json")) else { return Vec::new() };
    let int = |k: &str| c.get(k).and_then(Value::integer).ok().map_or(Value::Null, n);
    vec![
        ("layers", int("num_hidden_layers")),
        ("heads", int("num_attention_heads")),
        ("head_groups", int("num_key_value_heads")),
        ("kv_heads", int("num_key_value_heads")),
        ("hidden", int("hidden_size")),
    ]
}

/// `/api/limits` (2026-11-25, nothing pseudo): the saved limits, the recommendation and the
/// machine profile, with the limits in effect and the residency mode as the newest serving process
/// reports them in its live file. With no serving process running there is nothing in effect: the
/// limits apply when one starts, and the view says so instead of describing this process.
fn limits_view(home: &Path) -> Value {
    let mut v = nmapi::limits::view(home);
    let serving = crate::live::read_all(home).into_iter().filter(|p| p.get("limits").is_ok()).max_by_key(|p| p.get("beat_ms").and_then(Value::integer).unwrap_or(0));
    if let Value::Object(m) = &mut v {
        match serving {
            Some(p) => {
                if let Ok(l) = p.get("limits") {
                    m.insert("effective".into(), l.clone());
                }
                m.insert("effective_from".into(), s(&format!("serving process {}", p.get("pid").and_then(Value::integer).unwrap_or(0))));
            }
            None => {
                if let Some(Value::Object(e)) = m.get_mut("effective") {
                    e.insert("mode".into(), s("not serving"));
                }
                m.insert("effective_from".into(), s("no serving process is running: these are the limits a serving process will start with"));
            }
        }
    }
    v
}

fn status(home: &Path) -> Value {
    let models = nmtype::models_dir(home);
    // The precision saved in the controls: what a serving process starts with. The precision in
    // use is the one each serving process reports for the models it holds (`loaded`).
    let precision = nmengn::model::PRECISION.read().map(|p| *p).unwrap_or_default();
    // Models held now by the serving processes (their live files), one row per process and model.
    let live = crate::live::read_all(home);
    let mut loaded: Vec<Value> = Vec::new();
    for p in &live {
        for m in p.get("loaded").and_then(Value::array).unwrap_or_default() {
            let mut m = m.clone();
            if let Value::Object(o) = &mut m {
                for k in ["pid", "project"] {
                    if let Ok(x) = p.get(k) {
                        o.insert(k.to_owned(), x.clone());
                    }
                }
            }
            loaded.push(m);
        }
    }
    let sum = |k: &str| loaded.iter().filter_map(|m| m.get(k).and_then(Value::integer).ok()).sum::<u64>();
    let (ram, gpu) = (sum("ram_bytes"), sum("gpu_bytes"));
    let held: BTreeSet<String> = loaded.iter().filter_map(|m| m.get("id").and_then(Value::string).ok().map(str::to_owned)).collect();
    // 2026-11-25: per model, the bits in use as a serving process reports them (a settings
    // cartridge can change them inside that process), else the saved setting.
    let in_use: BTreeMap<String, Value> = loaded
        .iter()
        .filter_map(|m| {
            let id = m.get("id").and_then(Value::string).ok()?.to_owned();
            let bits = m.get("effective_bits").ok().filter(|b| !matches!(b, Value::Null)).or_else(|| m.get("precision").ok())?;
            Some((id, bits.clone()))
        })
        .collect();
    let mut list = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&models) {
        let mut dirs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
        dirs.sort();
        for d in dirs {
            let id = d.file_name().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default();
            let card = read_json(&d.join("part.json"));
            let main = card
                .as_ref()
                .and_then(|c| c.get("weights").ok())
                .and_then(|w| w.get("main_file").ok())
                .and_then(|m| m.string().ok())
                .map(str::to_owned);
            let mut row = vec![
                ("id", s(&id)),
                ("bytes", n(dir_bytes(&d.join("files")))),
                ("card", Value::Bool(card.is_some())),
                ("main_file", main.map_or(Value::Null, |m| s(&m))),
                ("precision", in_use.get(&id).cloned().unwrap_or_else(|| width(precision.of(&id)))),
                ("precision_source", s(if in_use.contains_key(&id) { "serving process" } else { "saved setting (applies at the next start)" })),
                ("loaded", Value::Bool(held.contains(&id))),
            ];
            row.extend(model_shape(&d));
            list.push(obj(row));
        }
    }
    let crates: Vec<Value> = std::fs::read_dir(home.join("crates"))
        .map(|rd| {
            let mut v: Vec<String> = rd
                .flatten()
                .filter(|e| e.path().join("part.json").is_file())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect();
            v.sort();
            v.into_iter().map(|c| s(&c)).collect()
        })
        .unwrap_or_default();
    obj(vec![
        ("name", s("Neuromorph Nano")),
        ("version", s(env!("CARGO_PKG_VERSION"))),
        ("api", n(API_VERSION)),
        ("root", s(&home.display().to_string())),
        ("models_dir", s(&models.display().to_string())),
        ("models", Value::Array(list)),
        ("loaded", Value::Array(loaded)),
        ("resident_bytes", n(ram)),
        ("gpu_bytes", n(gpu)),
        ("serving", n(live.len() as u64)),
        ("substrate", live.first().and_then(|p| p.get("substrate").ok().cloned()).unwrap_or(Value::Null)),
        ("parts", Value::Array(crates)),
        (
            "precision",
            obj(vec![
                ("base", width(precision.base)),
                ("writer", width(precision.writer)),
                ("reranker", width(precision.reranker)),
                ("embedder", width(precision.embedder)),
            ]),
        ),
        ("memory_dir", s(&home.join("memory").display().to_string())),
        ("time_ms", n(nmtype::now_ms())),
    ])
}

fn api(home: &Path, path: &str, query: &str) -> Option<Value> {
    let named = |dir: &Path| Value::Object(json_files(dir).into_iter().collect());
    match path {
        "/api/status" => Some(status(home)),
        "/api/controls" => Some(crate::controls::view(home)),
        // `settings.json` key by key: the keys beside the controls, each entry with its choices.
        "/api/controls/saved" => Some(crate::controls::saved_view(home)),
        "/api/parts" => Some(Value::Object(
            std::fs::read_dir(home.join("crates"))
                .map(|rd| {
                    rd.flatten()
                        .filter_map(|e| Some((e.file_name().to_string_lossy().into_owned(), read_json(&e.path().join("part.json"))?)))
                        .collect()
                })
                .unwrap_or_default(),
        )),
        "/api/settings" => Some(named(&home.join("designs"))),
        "/api/cartridges" => Some(all_cartridges(home)),
        // 2026-11-26: every learned unit by compartment, each plugged in or out (`learned_units`).
        "/api/learned-units" => Some(crate::learned_units::view(home)),
        "/api/cartridges/library" => Some(crate::library::api_view(home)),
        "/api/cartridges/bases" => Some(crate::library::base_versions(home)),
        // 2026-11-25: which MicroModels each cartridge fits; the model each role runs on.
        "/api/cartridges/compat" => Some(crate::library::compat(home)),
        "/api/models/roles" => Some(crate::library::roles_view(home)),
        // 2026-11-25: the released models and the base cartridges on disk, and the download in
        // progress (each downloaded on its own; `crate::fetch`).
        "/api/downloads" => Some(crate::fetch::view(home)),
        // 2026-11-26 (Luis Jake Gabriel III: the official cartridge library or third parties, with
        // the Qwen open weights as the fallback): cartridge sources (`sources`).
        "/api/cartridge-sources" => Some(crate::sources::view(home)),
        // 2026-11-25: every parameter map of every MicroModel (`crate::pmaps`).
        "/api/parameter-maps" => Some(crate::pmaps::view(home)),
        // 2026-11-25: for the MCP tools (`nm_settings op=devices`); the interface's own server
        // answers this path first with the addresses it listens on.
        "/api/devices" => Some(network::view(home, &network_mode(home), &[], 0)),
        // Project memory folders (E1, 2026-11-15): one per studied repository, and which is served now.
        "/api/projects" => Some(crate::project::list(home)),
        "/api/memory" => Some(read_json(&home.join("memory").join("index.json")).unwrap_or(Value::Null)),
        "/api/memory/tiers" => Some(crate::activity::memory_tiers(home)),
        // Per-component activity over a window, from the audit rows (`window`, `project`).
        "/api/activity" => Some(crate::activity::view(home, query)),
        // Filters `kind`, `project` and `limit` (newest 200 when absent); same shape either way.
        "/api/audit" => Some(crate::project::audit_view(home, query)),
        // A Claude Code session's state, and the evidence ledger (`project` is a key of `/api/projects`).
        "/api/session" => Some(crate::sessions::api_view(home, query)),
        "/api/ledger" => Some(crate::ledger::api_view(home, query)),
        // What memory knows of verified and refuted outcomes in a project, each place read fresh.
        "/api/outcomes" => Some(crate::outcomes::api_view(home, query)),
        // The training store: counts by family, domain, verdict, label, strength and split; newest examples.
        "/api/examples" => Some(crate::examples::api_view(home, query)),
        "/api/training/references" => Some(crate::examples::references_view(home)),
        "/api/training/distil" => Some(crate::distil::view(home, query)),
        // Teaching sets for a live teacher (`teach.rs`).
        "/api/training/distil/teach" => Some(crate::teach::list(home)),
        "/api/memory/items" => Some(crate::forget::memory_items(home, query)),
        // The 3-level Continuous Learning Engine & Supervisor state for a project (`project=<key>`).
        "/api/learning" => Some(crate::examples::learning_api_view(home, query)),
        // Precision all-layer parameter surgery maps, super-weights, and ledger across all base cartridges.
        "/api/surgery" => Some(surgery_view(home)),
        // Stage 6C: 4-level comprehension and world model: notes per level (L1 piece, L2 file,
        // L3 module, L4 system), outline preview, the measured routing record, per-lens heads and
        // the serving base cartridges.
        "/api/comprehension" => Some(comprehension_view(home, query)),
        // The per-stage breakdown of a warm question (median and p95 ms per stage, per set) from
        // the newest stage-bench report (`file=` names another).
        "/api/stage-times" => Some(stage_times_view(home, query)),
        // Roadmap Phase 4.5 (2026-11-24): disk use by category with exact paths, and the
        // storage paths and limits (`nmapi::storage`, which `nm_storage` calls too).
        "/api/storage" => Some(nmapi::storage::view(home)),
        "/api/storage/settings" => Some(nmapi::storage::settings(home)),
        // Roadmap Phase 4.5 (2026-11-25): the hardware limits: saved, in effect, recommended for
        // this machine, and its profile (`nmapi::limits`, which `nm_settings` calls too).
        "/api/limits" => Some(limits_view(home)),
        "/api/airgap" => Some(crate::controls::air_gap_view(home, 0)),
        // 2026-11-26 (Phase B, every feature in the interface, MCP and the API): a MicroModel's
        // tensors and clusters (`nm_weights`, read only), a map's contents, a cartridge's values,
        // the learning units, the project lenses, the objectives, the MCP clients' connection.
        "/api/models/weights" => Some(weights_view(home, query)),
        "/api/parameter-maps/read" => Some(crate::pmaps::read(home, query).unwrap_or_else(|e| obj(vec![("error", s(&e))]))),
        "/api/cartridges/inspect" => Some(crate::library::inspect(home, query)),
        // Stated preference cartridges (`preferences`).
        "/api/preferences" => Some(crate::preferences::list(home)),
        // The user's own repositories for backup and rollback (`backups.rs`).
        "/api/backups" => Some(crate::backups::view(home)),
        // Cartridge parameters (`cparams`): every object holding parameters; one object's
        // parameters section (`key`); one page of a tensor's values (`key`, `tensor`).
        "/api/cartridges/parameters" => Some(parameters_view(home, query)),
        // The whole-set overview of an opened object (`poverview`): its tensors in layout order
        // (`key`), or one tensor's region as a grid of cell summaries, streamed (`tensor`, `w`,
        // `h`, `r0`, `r1`, `c0`, `c1`).
        "/api/parameters/overview" => Some(crate::poverview::view(home, query)),
        // One object's parameter overlay (`overlay`): edits, masks and freezes held beside it.
        "/api/parameters/overlay" => Some(match crate::project::query_arg(query, "key").filter(|k| !k.is_empty()) {
            Some(k) => crate::overlay::view(home, &k),
            None => obj(vec![("error", s("key: the object's key (open:<model>, id@version, filter:<role>/<job>, <adapter>@<n> or a unit's id)"))]),
        }),
        "/api/units" => Some(crate::learned_units::units_view(home)),
        "/api/lenses" => Some(crate::learned_units::lenses_view(home, query)),
        "/api/goals" => Some(crate::memory_api::goals_view(home, query)),
        // Reminders, loops and the step graph for the coding agent (`guidance`).
        "/api/guidance" => Some(crate::guidance::view(home, query)),
        "/api/connect" => Some(crate::onboard::status_view(home, query)),
        _ => None,
    }
}

/// `GET /api/models/weights?model=<id>&limit=<n>` (2026-11-26, Phase B item A1): what `nm_weights`
/// reads for an installed MicroModel (default the base model): its weight card, tensor and cluster
/// counts, element types, the first `limit` tensors (name, element type, shape, bytes) and clusters,
/// and, when this process runs that model, what is resident. Nothing is written.
fn weights_view(home: &Path, query: &str) -> Value {
    let model = crate::project::query_arg(query, "model").filter(|m| !m.is_empty()).unwrap_or_else(|| nmapi::neural::POPULATION.to_owned());
    let limit = crate::project::query_arg(query, "limit").and_then(|l| l.parse::<usize>().ok()).unwrap_or(40).min(2000);
    if !nmtype::models_dir(home).join(&model).join("part.json").is_file() {
        return obj(vec![("error", s(&format!("{model}: not an installed model (see /api/status models)")))]);
    }
    nmapi::neural::weights_of(home, &model, limit).unwrap_or_else(|e| obj(vec![("error", s(&e.to_string()))]))
}

/// `GET /api/cartridges/parameters[?key=<key>[&tensor=<name>&offset=<n>&limit=<n>]]`: the list of
/// every object that holds parameters, one object's parameters section, or one page of a tensor's
/// values (`crate::cparams`); with `path`, the values a selection names and the map values at the
/// same place; with `maps=1`, the parameter maps beside the object.
fn parameters_view(home: &Path, query: &str) -> Value {
    let arg = |k: &str| crate::project::query_arg(query, k).filter(|x| !x.is_empty());
    let num = |name: &str, d: usize| arg(name).and_then(|x| x.parse::<usize>().ok()).unwrap_or(d);
    let fail = |e: String| obj(vec![("error", s(&e))]);
    match (arg("key"), arg("tensor")) {
        (None, _) => crate::cparams::list(home),
        (Some(k), _) if arg("head").is_some() => crate::cparams::head_count(home, &k),
        (Some(k), _) if arg("maps").is_some() => crate::cparams::maps(home, &k),
        (Some(k), _) if arg("path").is_some() => crate::cparams::region(home, &k, &arg("path").unwrap_or_default(), num("offset", 0)).unwrap_or_else(fail),
        (Some(k), None) => crate::cparams::section(home, &k),
        (Some(k), Some(t)) => crate::cparams::page(home, &k, &t, num("offset", 0), num("limit", 256)).unwrap_or_else(fail),
    }
}

/// Roadmap Phase 4.5 (2026-11-24): the interface's routes lent to the MCP tools
/// (`nmapi::parity::lend_routes`), so a tool and a page run the same function: a body of `None`
/// reads a route, `Some` writes one.
/// # Errors
/// An unknown route, or the route's own failure.
pub fn lent_route(home: &Path, path: &str, query: &str, body: Option<&[u8]>, by: &str) -> Result<Value, String> {
    match body {
        None => api(home, path, query).ok_or_else(|| format!("unknown route {path}")),
        Some(b) => write_api(home, path, b, by),
    }
}

/// 2026-11-26 (v0.1.0): the base serving each MicroModel, by model id, as `/api/cartridges/bases`
/// reports it (`serving`: a base cartridge key, or `open weights`). Empty when the library cannot
/// be read.
fn serving_bases(home: &Path) -> BTreeMap<String, String> {
    let v = crate::library::base_versions(home);
    let mut out = BTreeMap::new();
    if let Ok(Value::Array(models)) = v.get("models") {
        for m in models {
            if let (Ok(model), Ok(serving)) = (m.get("model").and_then(Value::string), m.get("serving").and_then(Value::string)) {
                out.insert(model.to_owned(), serving.to_owned());
            }
        }
    }
    out
}

fn surgery_view(home: &Path) -> Value {
    let base = home.join("library/parameter-maps");
    let mut roles: BTreeMap<String, Value> = BTreeMap::new();
    let serving = serving_bases(home);
    let mut ledger_all: Vec<Value> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&base) {
        let mut dirs: Vec<std::path::PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
        dirs.sort();
        for dir in dirs {
            let Some(name) = dir.file_name().and_then(|s| s.to_str()).map(str::to_owned) else { continue };
            let probe = read_json(&dir.join("surgery-probe.json"));
            let mut sw_count: usize = 0;
            if let Some(Value::Object(sw)) = read_json(&dir.join("super-weights.json"))
                && let Some(Value::Object(ts)) = sw.get("tensors")
            {
                for v in ts.values() {
                    if let Value::Array(arr) = v {
                        sw_count += arr.len();
                    }
                }
            }
            let mut role_ledger: Vec<Value> = Vec::new();
            if let Ok(text) = std::fs::read_to_string(dir.join("surgery-ledger.jsonl")) {
                for line in text.lines().filter(|l| !l.trim().is_empty()) {
                    if let Ok(v) = nmjson::parse(line.as_bytes()) {
                        role_ledger.push(v.clone());
                        ledger_all.push(v);
                    }
                }
            }
            let mut job_filters: Vec<Value> = Vec::new();
            if let Ok(frd) = std::fs::read_dir(&dir) {
                let mut fpaths: Vec<std::path::PathBuf> = frd
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| {
                        p.file_name()
                            .and_then(|s| s.to_str())
                            .is_some_and(|n| n.starts_with("job-filter-") && n.ends_with(".json"))
                    })
                    .collect();
                fpaths.sort();
                for fp in fpaths {
                    if let Some(Value::Object(jf)) = read_json(&fp) {
                        let mut summary = BTreeMap::new();
                        for k in ["cartridge", "job", "base", "model", "base_cos", "filtered_cos", "base_max_diff", "filtered_max_diff", "enabled"] {
                            if let Some(v) = jf.get(k) {
                                summary.insert(k.to_owned(), v.clone());
                            }
                        }
                        if !summary.contains_key("enabled") {
                            summary.insert("enabled".to_owned(), Value::Bool(true));
                        }
                        if let Some(Value::Array(parts)) = jf.get("parts") {
                            summary.insert("layers".to_owned(), Value::Number(parts.len().to_string()));
                        }
                        // 2026-11-26 (v0.1.0): the base cartridge version this filter was built
                        // on (its lineage `made_on`, null when not recorded), the base serving its
                        // MicroModel now, and whether they match (null when either is unknown).
                        // Only reported: the filter loads either way.
                        let file = fp.file_name().and_then(|s| s.to_str()).unwrap_or("");
                        let built_on = nmengn::filter::job_filter_built_on(&dir, file);
                        let serving_base = jf.get("model").and_then(|m| m.string().ok()).and_then(|m| serving.get(m)).cloned();
                        let base_match = nmengn::filter::job_filter_base_match(built_on.as_deref(), serving_base.as_deref());
                        summary.insert("built_on".to_owned(), built_on.map_or(Value::Null, |b| s(&b)));
                        summary.insert("serving_base".to_owned(), serving_base.map_or(Value::Null, |b| s(&b)));
                        summary.insert("base_match".to_owned(), base_match.map_or(Value::Null, Value::Bool));
                        job_filters.push(Value::Object(summary));
                    }
                }
            }
            if probe.is_some() || sw_count > 0 || !role_ledger.is_empty() || !job_filters.is_empty() {
                let mut m = BTreeMap::new();
                if let Some(p) = probe {
                    m.insert("probe".to_owned(), p);
                }
                m.insert("super_weights".to_owned(), Value::Number(sw_count.to_string()));
                m.insert("ledger".to_owned(), Value::Array(role_ledger));
                m.insert("job_filters".to_owned(), Value::Array(job_filters));
                roles.insert(name, Value::Object(m));
            }
        }
    }
    let mut root = BTreeMap::new();
    root.insert("roles".to_owned(), Value::Object(roles));
    root.insert("ledger".to_owned(), Value::Array(ledger_all));
    root.insert("job_filter_build".to_owned(), job_filter_build_json(home));
    root.insert("lab_build".to_owned(), job_filter_build_json(home));
    Value::Object(root)
}

/// A job filter build (2026-11-25, nothing pseudo, every process sees the same): kept in this
/// process and in `memory/job-filter-build.json` under the home, refreshed every few seconds while
/// it runs, so a build started from MCP shows in the interface and the other way round, and a
/// build whose process ended reads as stopped.
#[derive(Clone, Default)]
struct JobFilterBuildState {
    status: String,
    /// What the lab builds (2026-11-26): `filter` (a job filter), `map` (a parameter map and the
    /// base cartridge version cut from it) or `base` (a base cartridge version by restoration).
    kind: String,
    role: String,
    job: String,
    by: String,
    started_ms: u64,
    finished_ms: u64,
    updated_ms: u64,
    message: String,
}

static JOB_FILTER_BUILD: Mutex<Option<JobFilterBuildState>> = Mutex::new(None);

fn build_file(home: &Path) -> PathBuf {
    home.join("memory").join("job-filter-build.json")
}

fn build_value(st: &JobFilterBuildState) -> Value {
    obj(vec![
        ("status", s(&st.status)),
        ("kind", s(if st.kind.is_empty() { "filter" } else { &st.kind })),
        ("role", s(&st.role)),
        ("job", s(&st.job)),
        ("by", s(&st.by)),
        ("started_ms", n(st.started_ms)),
        ("finished_ms", n(st.finished_ms)),
        ("updated_ms", n(st.updated_ms)),
        ("message", s(&st.message)),
    ])
}

fn write_build(home: &Path, st: &JobFilterBuildState) {
    let p = build_file(home);
    if let Some(d) = p.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let tmp = p.with_extension("json.tmp");
    if std::fs::write(&tmp, nmjson::pretty::pretty(&build_value(st))).is_ok() {
        let _ = std::fs::rename(&tmp, &p);
    }
}

fn job_filter_build_json(home: &Path) -> Value {
    if let Some(st) = JOB_FILTER_BUILD.lock().ok().and_then(|g| g.clone()) {
        return build_value(&st);
    }
    let Some(mut v) = std::fs::read(build_file(home)).ok().and_then(|b| nmjson::parse(&b).ok()) else {
        return build_value(&JobFilterBuildState { status: "idle".to_owned(), ..JobFilterBuildState::default() });
    };
    let running = v.get("status").and_then(Value::string).is_ok_and(|x| x == "running");
    let updated = v.get("updated_ms").and_then(Value::integer).unwrap_or(0);
    if running
        && nmtype::now_ms().saturating_sub(updated) > 30_000
        && let Value::Object(m) = &mut v
    {
        m.insert("status".into(), s("stopped"));
        m.insert("message".into(), s("the process that ran this build ended before it finished; start it again"));
    }
    v
}

/// Takes the one lab build slot for `st`, or refuses while a build runs here or in another process.
/// 2026-11-26: whether a build runs elsewhere is read BEFORE the state lock is taken; reading it
/// under the lock (`build_running` takes the same lock) deadlocked the request, and every later
/// `/api/surgery` read waited on it (found by the first real build started from the interface).
fn claim_build(home: &Path, st: JobFilterBuildState) -> Result<(), String> {
    let elsewhere = build_running(home);
    let mut g = JOB_FILTER_BUILD.lock().map_err(|_| "build state lock poisoned".to_owned())?;
    if g.as_ref().is_some_and(|x| x.status == "running") || elsewhere {
        return Err("a lab build is already running".to_owned());
    }
    write_build(home, &st);
    *g = Some(st);
    Ok(())
}

/// Whether a job filter build runs in any process.
fn build_running(home: &Path) -> bool {
    job_filter_build_json(home).get("status").and_then(Value::string).is_ok_and(|x| x == "running")
}

fn find_nano_lab(home: &Path) -> Option<PathBuf> {
    let name = if cfg!(windows) { "nano-lab.exe" } else { "nano-lab" };
    let mut candidates = Vec::new();
    if let Ok(cur) = std::env::current_exe()
        && let Some(dir) = cur.parent()
    {
        candidates.push(dir.join(name));
    }
    candidates.push(home.join("bin").join(name));
    candidates.push(home.join("target/release").join(name));
    candidates.into_iter().find(|p| p.is_file())
}

fn safe_layers(spec: &str) -> bool {
    spec == "all" || (!spec.is_empty() && spec.len() <= 32 && spec.chars().all(|c| c.is_ascii_digit() || matches!(c, ',' | '-')))
}

fn start_job_filter_build(home: &Path, role: &str, job: &str, layers: &str, by: &str) -> Result<Value, String> {
    start_lab_build(home, &LabBuild { kind: "filter", role, job, layers, share: 0.0, device: "gpu", target: "", set: "", technique: "", rounds: 0, budget_s: 0 }, by)
}

/// One lab build the interface, MCP or the API asks for.
struct LabBuild<'a> {
    kind: &'a str,
    role: &'a str,
    job: &'a str,
    layers: &'a str,
    share: f64,
    device: &'a str,
    /// Distillation: the kind of learned cartridge it fills (`adapter`, `filter`, ...), the teacher
    /// data's set (`all` for every set), the technique and its rounds.
    target: &'a str,
    set: &'a str,
    technique: &'a str,
    rounds: u64,
    /// A distillation by `unit`: the wall budget of each forward-only fit, seconds (0: the default).
    budget_s: u64,
}

/// The lab's arguments and the first status line for a build. A build that makes a base cartridge
/// version keeps it beside the others when `library.upgrade` is keep (the default), so what serves
/// now is never taken away; the user chooses on Base versions.
fn lab_args(home: &Path, b: &LabBuild<'_>) -> Result<(Vec<String>, String), String> {
    let role = b.role;
    let keep = crate::controls::current(home).get(crate::library::UPGRADE_KEY).is_none_or(|p| p == "keep");
    let mut args: Vec<String> = match b.kind {
        "filter" => ["run", "pmap", "surgery", "job-filter", role, "--job", b.job, "--layers", b.layers, "--split-gate", "--write"].into_iter().map(str::to_owned).collect(),
        "map" => {
            if !(0.05..=0.60).contains(&b.share) {
                return Err("share must be between 0.05 and 0.60 (the part of each matrix mapped out)".to_owned());
            }
            if !matches!(b.device, "gpu" | "cpu") {
                return Err("device must be gpu or cpu".to_owned());
            }
            vec!["run".to_owned(), "pmap".to_owned(), "build-base".to_owned(), role.to_owned(), format!("{:.2}", b.share), b.device.to_owned()]
        }
        "base" => ["run", "pmap", "surgery", "cascade", role, "--anchor-budget", "8", "--budget", "5", "--write"].into_iter().map(str::to_owned).collect(),
        "distil" => {
            let technique = if b.technique.is_empty() { "lens" } else { b.technique };
            let target = if !b.target.is_empty() {
                b.target
            } else if technique == "channel" {
                "filter"
            } else {
                "adapter"
            };
            if let Some(why) = nmapi::distil::not_built(role, target, technique) {
                return Err(why);
            }
            let set = if b.set.is_empty() { "all" } else { b.set };
            if !safe_slug(set) {
                return Err("set must be a set name as the teacher data lists it (letters, digits, '-' or '_'), or all".to_owned());
            }
            let mut a: Vec<String> = ["run", "distil", "build", role, "--target", target, "--set", set, "--technique", technique].into_iter().map(str::to_owned).collect();
            if technique == "unit" && b.budget_s > 0 {
                a.extend(["--budget-s".to_owned(), b.budget_s.min(120).to_string()]);
            }
            if technique == "teacher-loop" && b.rounds > 0 {
                a.extend(["--rounds".to_owned(), b.rounds.min(8).to_string()]);
            }
            // The build runs on the base the MicroModel serves now (the user's choice on Base
            // versions), not on the newest version in the cartridge library.
            match serving_bases(home).get(&nmtype::role_models::model(role)).map(String::as_str) {
                Some("open weights") => a.push("--open".to_owned()),
                Some(key) if !key.is_empty() => {
                    let file = nmlearn::library::Library::open(home).ok().and_then(|lib| lib.cartridge(key).and_then(|c| c.files.first().map(|f| lib.cartridge_dir(c).join(&f.name))));
                    if let Some(f) = file {
                        a.extend(["--base".to_owned(), f.display().to_string()]);
                    }
                }
                _ => {}
            }
            a
        }
        // A held-out check of an existing job filter (an edited version, say): measured on the
        // held-out prompts against the open weights, nothing fitted or written.
        "check" => {
            if !safe_slug(b.job) {
                return Err("job: the job filter's job name, as Parameters lists it".to_owned());
            }
            let file = home.join("library/parameter-maps").join(role).join(format!("job-filter-{}.json", b.job));
            if !file.is_file() {
                return Err(format!("no job filter {} for the {role}", b.job));
            }
            ["run", "pmap", "surgery", "job-filter", role, "--layers", "all", "--split-gate"].into_iter().map(str::to_owned).chain(["--check-file".to_owned(), file.display().to_string()]).collect()
        }
        // A local open-weights model answers a teaching set's open prompts (`teach.rs`); its
        // answers are imported into the teacher data when the run ends.
        "teach" => {
            let (pf, of, instruction, _) = crate::teach::local_files(home, b.set)?;
            let teacher = if b.technique.is_empty() { nmapi::neural::WRITER } else { b.technique };
            let mut a: Vec<String> = ["run", "distil", "teach"].into_iter().map(str::to_owned).collect();
            a.extend(["--prompts".to_owned(), pf.display().to_string(), "--out".to_owned(), of.display().to_string(), "--teacher".to_owned(), teacher.to_owned()]);
            if !instruction.is_empty() {
                a.extend(["--instruction".to_owned(), instruction]);
            }
            a
        }
        other => return Err(format!("{other}: build a filter, a map, a base, a distil or a teach")),
    };
    if !matches!(b.kind, "filter" | "distil") && keep {
        args.push("--keep".to_owned());
    }
    let beside = if keep { "kept beside the versions you have; choose what serves on Base versions" } else { "replacing the base cartridge in place (library.upgrade: replace)" };
    let msg = match b.kind {
        "filter" => format!("Running held-out split-gate fit for {role}/{} (layers: {})...", b.job, b.layers),
        "check" => format!("Measuring the job filter {role}/{} on the held-out prompts against the open weights; nothing is written...", b.job),
        "teach" => format!("A local open-weights model is answering the open prompts of teaching set {}...", b.set),
        "distil" => format!(
            "Distilling the teacher pairs of set {} into the {role} ({}); fit, tune and held-out gate, a cartridge written only if it passes...",
            if b.set.is_empty() { "all" } else { b.set },
            if b.technique.is_empty() { "lens" } else { b.technique }
        ),
        "map" => format!("Mapping the {role}'s parameters ({:.0}% share, {}) and cutting a base cartridge version from the map, {beside}...", b.share * 100.0, b.device),
        _ => format!("Restoring the weights that matter most into a new {role} base cartridge version, {beside}..."),
    };
    Ok((args, msg))
}

/// Starts a lab build in the background: one at a time, in the live view while it runs, audited
/// when it ends. 2026-11-26 (Luis Jake Gabriel III: "This should all be part of the neuromorph nano
/// and integrated fully into the mcp the api and the interface").
fn start_lab_build(home: &Path, b: &LabBuild<'_>, by: &str) -> Result<Value, String> {
    let (role, job, layers) = (b.role, b.job, b.layers);
    if b.kind == "filter" && (!safe_slug(role) || !safe_slug(job)) {
        return Err("role and job must be alphanumeric with '-' or '_'".to_owned());
    }
    if b.kind == "filter" && !safe_layers(layers) {
        return Err("layers must be 'all' or a numeric range/list like '12-27'".to_owned());
    }
    // The name and layer checks answer first, so a malformed request gets the same message from
    // every build route; a well-formed name that is not a MicroModel role is refused here.
    if !matches!(role, "reader" | "writer" | "embedder" | "reranker") {
        return Err("role must be reader, writer, embedder or reranker".to_owned());
    }
    let (lab_argv, first_message) = lab_args(home, b)?;
    // Roadmap Phase 4.5 (2026-11-24): no build while the drive is under the disk floor.
    nmapi::storage::disk_guard(home)?;
    // 2026-11-26 (Luis Jake Gabriel III: "Nano lab should be a part of Neuromorph nano"): the lab runs
    // inside this same program (`neuromorph-nano lab ...`); a separate nano-lab program is only the
    // fallback when this program's own path cannot be read.
    let (lab, lead): (PathBuf, Option<&str>) = match std::env::current_exe() {
        Ok(me) => (me, Some("lab")),
        Err(_) => match find_nano_lab(home) {
            Some(p) => (p, None),
            None => return Err("the lab is not reachable: this program's path is unreadable and no nano-lab program was found".to_owned()),
        },
    };
    claim_build(
        home,
        JobFilterBuildState {
            status: "running".to_owned(),
            kind: b.kind.to_owned(),
            role: role.to_owned(),
            job: if b.kind == "distil" { format!("{} ({})", if b.set.is_empty() { "all" } else { b.set }, if b.technique.is_empty() { "lens" } else { b.technique }) } else { job.to_owned() },
            by: by.to_owned(),
            started_ms: nmtype::now_ms(),
            finished_ms: 0,
            updated_ms: nmtype::now_ms(),
            message: first_message,
        },
    )?;
    let home_owned = home.to_path_buf();
    let role_owned = role.to_owned();
    let job_owned = match b.kind {
        "filter" => job.to_owned(),
        "distil" => format!("{} ({})", if b.set.is_empty() { "all" } else { b.set }, if b.technique.is_empty() { "lens" } else { b.technique }),
        _ => b.kind.to_owned(),
    };
    let kind_owned = b.kind.to_owned();
    let set_owned = b.set.to_owned();
    let teacher_owned = if b.technique.is_empty() { nmapi::neural::WRITER.to_owned() } else { b.technique.to_owned() };
    let beat_home = home.to_path_buf();
    std::thread::spawn(move || {
        while let Some(st) = JOB_FILTER_BUILD.lock().ok().and_then(|mut g| {
            g.as_mut().filter(|st| st.status == "running").map(|st| {
                st.updated_ms = nmtype::now_ms();
                st.clone()
            })
        }) {
            write_build(&beat_home, &st);
            std::thread::park_timeout(std::time::Duration::from_secs(3));
        }
    });
    std::thread::spawn(move || {
        // Roadmap Phase 4.5 (2026-11-24): the build in the live view while it runs.
        let mut live = nmapi::live::stage_as("filter", if kind_owned == "filter" { "filter.build" } else { "lab.build" }, nmapi::live::role_name(&role_owned), "");
        let tmp_dir = home_owned.join("scratch/tmp");
        let _ = std::fs::create_dir_all(&tmp_dir);
        let out = std::process::Command::new(&lab)
            .args(lead)
            .current_dir(&home_owned)
            .env("TEMP", &tmp_dir)
            .env("TMP", &tmp_dir)
            // `nano-lab run pmap surgery ...` (2026-11-24: without `run` the command fell through to
            // nano-lab's usage text and no build ever ran).
            .args(&lab_argv)
            .output();
        let (status, message) = match out {
            Ok(o) => {
                let combined = format!("{}\n{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
                let summary: Vec<&str> = combined
                    .lines()
                    .map(str::trim)
                    .filter(|l| {
                        l.starts_with("WRITTEN:")
                            || l.starts_with("REFUSED:")
                            || l.starts_with("held-out")
                            || l.starts_with("final_cos")
                            || l.starts_with("kept")
                            || l.starts_with("elapsed:")
                            || l.contains("PASS")
                            || l.contains("FAIL")
                            || l.contains("error")
                    })
                    .collect();
                let tail = if summary.is_empty() {
                    combined.lines().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join(" | ")
                } else {
                    summary.join(" | ")
                };
                let clipped: String = tail.chars().take(480).collect();
                if o.status.success() {
                    ("done".to_owned(), if clipped.is_empty() { "Completed.".to_owned() } else { clipped })
                } else {
                    ("error".to_owned(), if clipped.is_empty() { format!("Exited with status {}", o.status) } else { clipped })
                }
            }
            Err(e) => ("error".to_owned(), format!("spawn {}: {e}", lab.display())),
        };
        live.label(if status == "done" { "done" } else { "error" });
        drop(live);
        let what = match kind_owned.as_str() {
            "map" => "Parameter map and base cartridge build",
            "base" => "Base cartridge build",
            "distil" => "Distillation build",
            "teach" => "Local teacher run",
            "check" => "Held-out check",
            _ => "Job filter build",
        };
        // A local teacher's answers join the teacher data when its run ends.
        if kind_owned == "teach" && status == "done" {
            let newest = std::fs::read_dir(nmtype::examples::dir(&home_owned.join("memory")).join("teach")).ok().and_then(|rd| {
                rd.flatten()
                    .map(|e| e.path())
                    .filter(|p| p.file_name().and_then(|x| x.to_str()).is_some_and(|x| x.starts_with(&format!("{}-local-", nmapi::distil::slug(&set_owned))) && x.ends_with(".jsonl")))
                    .max()
            });
            if let Some(out) = newest {
                let _ = crate::teach::local_done(&home_owned, &set_owned, &out, &format!("{} (local open weights)", teacher_owned), "lab");
            }
        }
        let label = if kind_owned == "filter" { format!("{what} {role_owned}/{job_owned}: {status}") } else { format!("{what} for the {role_owned}: {status}") };
        let _ = crate::controls::audit(&home_owned, if kind_owned == "filter" { "job-filter-build" } else { "lab-build" }, vec![("kind", s(&kind_owned)), ("role", s(&role_owned)), ("job", s(&job_owned)), ("status", s(&status)), ("label", s(&label))]);
        if let Ok(mut g) = JOB_FILTER_BUILD.lock()
            && let Some(st) = g.as_mut()
        {
            st.status = status;
            st.finished_ms = nmtype::now_ms();
            st.updated_ms = st.finished_ms;
            st.message = message;
            write_build(&home_owned, st);
        }
    });
    Ok(obj(vec![
        ("kind", s(b.kind)),
        ("role", s(role)),
        ("job", s(job)),
        ("layers", s(layers)),
        ("job_filter_build", job_filter_build_json(home)),
        ("lab_build", job_filter_build_json(home)),
    ]))
}

fn safe_slug(s: &str) -> bool {
    !s.is_empty() && s.len() <= 64 && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
}

/// Where a removed job filter is kept so its removal can be undone (`<role>/removed/`; the
/// scanners read only the files directly in the role folder).
fn removed_job_filter(home: &Path, role: &str, job: &str) -> PathBuf {
    home.join("library/parameter-maps").join(role).join("removed").join(format!("job-filter-{job}.json"))
}

fn set_job_filter_enabled(path: &Path, on: bool) -> Result<bool, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut val = nmjson::parse(text.as_bytes()).map_err(|e| format!("parse: {e}"))?;
    let Value::Object(ref mut m) = val else {
        return Err("job filter JSON is not an object".to_owned());
    };
    let was = !matches!(m.get("enabled"), Some(Value::Bool(false)));
    m.insert("enabled".to_owned(), Value::Bool(on));
    let encoded = val.encode().map_err(|e| format!("encode: {e}"))?;
    std::fs::write(path, encoded).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(was)
}

/// Plugs a job filter in or out, or removes it (kept under `removed/`), with an audit row the
/// Audit page can undo (2026-11-25).
fn update_job_filter(home: &Path, role: &str, job: &str, action: &str, by: &str) -> Result<Value, String> {
    if !safe_slug(role) || !safe_slug(job) {
        return Err("role and job must be alphanumeric with '-' or '_'".to_owned());
    }
    let path = home.join("library/parameter-maps").join(role).join(format!("job-filter-{job}.json"));
    if !path.is_file() {
        return Err(format!("job filter {role}/{job} not found"));
    }
    let from = match action {
        "delete" => {
            let keep = removed_job_filter(home, role, job);
            if let Some(d) = keep.parent() {
                std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
            }
            std::fs::rename(&path, &keep).map_err(|e| format!("{}: {e}", path.display()))?;
            "present"
        }
        "enable" | "disable" => {
            if set_job_filter_enabled(&path, action == "enable")? {
                "enabled"
            } else {
                "disabled"
            }
        }
        other => return Err(format!("unknown action {other} (use enable, disable, or delete)")),
    };
    let label = match action {
        "delete" => format!("Job filter {role}/{job} removed (kept for undo)"),
        "enable" => format!("Job filter {role}/{job} plugged in"),
        _ => format!("Job filter {role}/{job} plugged out"),
    };
    let audit_id = crate::controls::audit(home, "job-filter", vec![("role", s(role)), ("job", s(job)), ("action", s(action)), ("from", s(from)), ("by", s(by)), ("label", s(&label))])?;
    Ok(obj(vec![
        ("role", s(role)),
        ("job", s(job)),
        ("action", s(action)),
        ("audit_id", s(&audit_id)),
        ("surgery", surgery_view(home)),
    ]))
}

/// Undoes a `job-filter` audit row: puts a removed filter back, or sets it back to how it was.
/// # Errors
/// The row is not a job filter change, or the filter is no longer where the undo needs it.
pub fn undo_job_filter(home: &Path, row: &Value, id: &str, by: &str) -> Result<Option<String>, String> {
    let field = |k: &str| row.get(k).and_then(Value::string).map(str::to_owned).map_err(|e| e.to_string());
    let (role, job, action, from) = (field("role")?, field("job")?, field("action")?, field("from")?);
    if !safe_slug(&role) || !safe_slug(&job) {
        return Err("bad job filter in the audit row".to_owned());
    }
    let path = home.join("library/parameter-maps").join(&role).join(format!("job-filter-{job}.json"));
    if action == "delete" {
        let keep = removed_job_filter(home, &role, &job);
        if path.exists() {
            return Err(format!("job filter {role}/{job} exists again; nothing was put back"));
        }
        std::fs::rename(&keep, &path).map_err(|e| format!("{}: {e}", keep.display()))?;
    } else {
        if !path.is_file() {
            return Err(format!("job filter {role}/{job} is no longer there"));
        }
        set_job_filter_enabled(&path, from == "enabled")?;
    }
    let label = format!("Undone: job filter {role}/{job} {action}");
    let aid = crate::controls::audit(home, "job-filter", vec![("role", s(&role)), ("job", s(&job)), ("action", s("undo")), ("undo_of", s(id)), ("by", s(by)), ("label", s(&label))])?;
    Ok(Some(aid))
}

/// The four levels of the world model and the note tags (the idle writer's `Job::tag`) each
/// level holds: L1 piece (piece notes, likely names and piece summaries), L2 file, L3 module
/// (folder or crate), L4 system (the project outline).
const LEVELS: [(u64, &str, &[&str]); 4] = [(1, "piece", &["note", "names", "sum"]), (2, "file", &["file"]), (3, "module", &["module"]), (4, "system", &["outline"])];

/// The world-model level (1 to 4) a note tag belongs to; none for a tag of no level.
fn level_of(tag: &str) -> Option<u64> {
    LEVELS.iter().find(|(_, _, tags)| tags.contains(&tag)).map(|(l, _, _)| *l)
}

/// What the note store holds, read the way the idle writer loads it (`writer_idle::load_store`):
/// one JSON object per line with `k` (the key, `<tag>:<hash or path>`), `by` (the writer's values
/// key) and `t` (the text); a later line for the same key replaces the earlier one, and the
/// newest outline is the last outline line. Returns the notes per tag, the number of distinct
/// keys and the newest outline as (key, text).
fn read_notes(text: &str) -> (BTreeMap<String, u64>, u64, Option<(String, String)>) {
    let mut by_key: BTreeSet<String> = BTreeSet::new();
    let mut outline: Option<(String, String)> = None;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(v) = nmjson::parse(line.as_bytes()) else { continue };
        let get = |k: &str| v.get(k).and_then(Value::string).ok();
        let (Some(k), Some(_), Some(t)) = (get("k"), get("by"), get("t")) else { continue };
        if k.starts_with("outline:") {
            outline = Some((k.to_owned(), t.to_owned()));
        }
        by_key.insert(k.to_owned());
    }
    let mut tags: BTreeMap<String, u64> = BTreeMap::new();
    for k in &by_key {
        let tag = k.split_once(':').map_or("other", |(t, _)| t);
        *tags.entry(tag.to_owned()).or_insert(0) += 1;
    }
    (tags, by_key.len() as u64, outline)
}

/// The note store of a memory folder: `<memory>/world-model/notes.jsonl` first, else the older
/// `<memory>/writer/notes.jsonl`. Returns the place read (relative to the memory folder) and its
/// text; none when neither exists.
fn note_store(memory: &Path) -> Option<(&'static str, String)> {
    ["world-model/notes.jsonl", "writer/notes.jsonl"].into_iter().find_map(|rel| std::fs::read_to_string(memory.join(rel)).ok().map(|t| (rel, t)))
}

/// The fields of `v` named in `keys` that it has, as a new object.
fn pick(v: &Value, keys: &[&str]) -> Value {
    Value::Object(keys.iter().filter_map(|k| v.get(k).ok().map(|x| ((*k).to_owned(), x.clone()))).collect())
}

/// The measured routing record (`designs/routing-record.json`): its rows (job, model, accuracy,
/// slope, n and the check fold), its combine flags with their decisive thresholds, and its lens
/// list. `present` is false when the file is missing or unreadable (the strengths table then
/// falls back to its fixed values).
fn routing_view(home: &Path) -> Value {
    let Some(rec) = read_json(&home.join("designs/routing-record.json")) else {
        return obj(vec![("present", Value::Bool(false))]);
    };
    let rows_in = rec.get("rows").and_then(Value::array).unwrap_or_default();
    let mut rows = Vec::new();
    let mut combine = Vec::new();
    for r in rows_in {
        if r.get("model").is_ok() {
            rows.push(pick(r, &["job", "kind", "model", "stack", "accuracy", "slope", "n", "correct", "check_accuracy", "check_n", "strength"]));
        } else if r.get("combine").is_ok() {
            let mut c = pick(r, &["job", "kind", "combine", "decisive"]);
            if let Value::Object(o) = &mut c {
                o.entry("decisive".to_owned()).or_insert(Value::Null);
            }
            combine.push(c);
        }
    }
    let lens = rec.get("lens").and_then(Value::string).unwrap_or("");
    let lenses: Vec<Value> = lens.split(',').map(str::trim).filter(|l| !l.is_empty()).map(s).collect();
    let bases: Vec<Value> = match rec.get("base_cartridges") {
        Ok(Value::Object(b)) => b.keys().map(|k| s(k)).collect(),
        _ => Vec::new(),
    };
    obj(vec![
        ("present", Value::Bool(true)),
        ("made_utc", rec.get("made_utc").ok().cloned().unwrap_or(Value::Null)),
        ("version", rec.get("version").ok().cloned().unwrap_or(Value::Null)),
        ("rows", Value::Array(rows)),
        ("combine", Value::Array(combine)),
        ("lens", s(lens)),
        ("lenses", Value::Array(lenses)),
        ("base_cartridges", Value::Array(bases)),
    ])
}

/// The drone reader sets (`designs/drones.json`): per population the `default` set and each
/// lens's own set under `tentacles`, as rows `{population, lens, heads, count}` with `heads` the
/// list of `[layer, head]`. A lens with no own set reads with the default one.
fn heads_view(home: &Path) -> Vec<Value> {
    let mut out = Vec::new();
    let Some(drones) = read_json(&home.join("designs/drones.json")) else { return out };
    let Ok(Value::Object(pops)) = drones.get("populations") else { return out };
    for (pop, p) in pops {
        let row = |lens: &str, heads: &Value| {
            let count = heads.array().map(|a| a.len() as u64).unwrap_or(0);
            obj(vec![("population", s(pop)), ("lens", s(lens)), ("heads", heads.clone()), ("count", n(count))])
        };
        if let Ok(d) = p.get("default") {
            out.push(row("default", d));
        }
        if let Ok(Value::Object(tents)) = p.get("tentacles") {
            for (lens, heads) in tents {
                out.push(row(lens, heads));
            }
        }
    }
    out
}

fn comprehension_view(home: &Path, query: &str) -> Value {
    // 2026-11-25: an unknown project is an error, never another folder's notes shown in its place;
    // with no project named, the Nano's own memory.
    let named = query.split('&').any(|kv| kv.split_once('=').is_some_and(|(k, v)| k == "project" && !v.is_empty()));
    let memory_dir = match crate::ledger::project_memory(home, query) {
        Some((_, m)) => m,
        None if named => return obj(vec![("error", s("project: give the key of a project as /api/projects lists it"))]),
        None => home.join("memory"),
    };
    let store = note_store(&memory_dir);
    let (tags, total_notes, outline) = store.as_ref().map(|(_, t)| read_notes(t)).unwrap_or_default();

    let levels: Vec<Value> = LEVELS
        .iter()
        .map(|(l, name, names)| {
            let count: u64 = names.iter().filter_map(|t| tags.get(*t)).sum();
            let per: Vec<Value> = names.iter().map(|t| obj(vec![("tag", s(t)), ("count", n(tags.get(*t).copied().unwrap_or(0)))])).collect();
            obj(vec![("level", n(*l)), ("name", s(name)), ("count", n(count)), ("tags", Value::Array(per))])
        })
        .collect();
    let unlevelled: u64 = tags.iter().filter(|(t, _)| level_of(t).is_none()).map(|(_, c)| c).sum();
    let counts_obj: BTreeMap<String, Value> = tags.into_iter().map(|(k, v)| (k, n(v))).collect();

    let routing = routing_view(home);
    let routing_present = matches!(routing.get("present"), Ok(Value::Bool(true)));
    let heads = heads_view(home);
    let default_heads = heads
        .iter()
        .filter(|h| h.get("lens").and_then(Value::string).is_ok_and(|l| l == "default"))
        .filter_map(|h| h.get("count").and_then(Value::integer).ok())
        .max()
        .unwrap_or(0);
    let lens_sets = heads.iter().filter(|h| h.get("lens").and_then(Value::string).is_ok_and(|l| l != "default")).count() as u64;

    let mut base_count: u64 = 0;
    let mut backed_up: u64 = 0;
    if let Some(Value::Object(bc)) = read_json(&home.join("designs/base-cartridges.json"))
        && let Some(Value::Object(serving)) = bc.get("serving")
    {
        base_count = serving.len() as u64;
        for entry in serving.values() {
            if let Ok(c) = entry.get("backup_commit").and_then(Value::string)
                && !c.is_empty()
                && !c.contains("pending")
                && !c.contains("uploading")
            {
                backed_up += 1;
            }
        }
    }

    let (outline_key, outline_text) = outline.map_or((Value::Null, Value::Null), |(k, t)| (s(&k), s(&t.chars().take(600).collect::<String>())));
    obj(vec![
        ("total_notes", n(total_notes)),
        ("note_counts", Value::Object(counts_obj)),
        ("notes_store", store.map_or(Value::Null, |(rel, _)| s(rel))),
        ("levels", Value::Array(levels)),
        ("unlevelled_notes", n(unlevelled)),
        ("outline", outline_text),
        ("outline_key", outline_key),
        ("routing_record", Value::Bool(routing_present)),
        ("routing", routing),
        ("drone_default_heads", n(default_heads)),
        ("drone_lens_sets", n(lens_sets)),
        ("heads", Value::Array(heads)),
        ("base_cartridges", n(base_count)),
        ("base_backed_up", n(backed_up)),
    ])
}

/// The hook that would give live per-call stage times, reported with the bench tables.
const STAGE_HOOK: &str = "live per-call stage times come from the serving process's live file (the newest 32 calls, while a serving process runs); the tables are the newest runs/STAGE-BENCH-*.txt measurement";

/// One parsed stage table: the set, the set's result line, its warm-question line and the stage
/// rows (stage, median ms, p95 ms, n).
struct StageSet {
    set: String,
    summary: String,
    warm: String,
    stages: Vec<(String, String, String, u64)>,
}

/// The stage tables of a `nano-lab stage-bench` report: each `stages, <set> (...)` block and the
/// rows under it (`<stage> <median> <p95> <n>`), with the set's result line (the unindented line
/// before) and its `warm question ms` line. Also the report's first line up to `program`.
fn parse_stage_bench(text: &str) -> (String, Vec<StageSet>) {
    let header = text.lines().next().unwrap_or("").split("  program").next().unwrap_or("").trim().to_owned();
    let mut sets: Vec<StageSet> = Vec::new();
    let mut summary = String::new();
    let mut warm = String::new();
    let mut open = false;
    for line in text.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("stages, ") {
            let set = rest.split(" (").next().unwrap_or(rest).trim().to_owned();
            sets.push(StageSet { set, summary: summary.clone(), warm: warm.clone(), stages: Vec::new() });
            open = true;
            continue;
        }
        if open {
            let f: Vec<&str> = t.split_whitespace().collect();
            if let [stage, med, p95, cnt] = f.as_slice()
                && med.parse::<f64>().is_ok()
                && p95.parse::<f64>().is_ok()
                && let Ok(c) = cnt.parse::<u64>()
                && let Some(cur) = sets.last_mut()
            {
                cur.stages.push(((*stage).to_owned(), (*med).to_owned(), (*p95).to_owned(), c));
                continue;
            }
            open = false;
        }
        if !line.starts_with(' ') && !t.is_empty() {
            summary = t.to_owned();
            warm.clear();
        } else if let Some(w) = t.strip_prefix("warm question ms:") {
            warm = w.trim().to_owned();
        }
    }
    (header, sets)
}

/// The stamp of a `STAGE-BENCH-<stamp>.txt` file name.
fn bench_stamp(name: &str) -> Option<u64> {
    name.strip_prefix("STAGE-BENCH-")?.strip_suffix(".txt")?.parse().ok()
}

/// `GET /api/stage-times[?file=STAGE-BENCH-<stamp>.txt]`: the per-stage breakdown of a warm
/// question (median and p95 ms per stage, per set) from the newest stage-bench report that has
/// stage tables, or the named one. The serving process prints its `NANO_STAGE_TIMES` lines to its
/// own standard error only, out of this server's reach; `hook` says what would make it live. When
/// a serving process does publish recent calls in its live file (`stage_times`, a list of
/// objects of stage to ms), they are passed on as `live_calls` (newest last) and `live` is true.
fn stage_times_view(home: &Path, query: &str) -> Value {
    let runs = home.join("runs");
    let mut files: Vec<(u64, String)> = std::fs::read_dir(&runs)
        .map(|rd| rd.flatten().filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            bench_stamp(&name).map(|st| (st, name))
        }).collect())
        .unwrap_or_default();
    files.sort_by_key(|a| std::cmp::Reverse(a.0));
    let want = crate::project::query_arg(query, "file").filter(|f| bench_stamp(f).is_some());
    let mut chosen: Option<(String, u64, String, Vec<StageSet>)> = None;
    for (stamp, name) in &files {
        if want.as_deref().is_some_and(|w| w != name) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(runs.join(name)) else { continue };
        let (header, sets) = parse_stage_bench(&text);
        if sets.iter().any(|st| !st.stages.is_empty()) {
            chosen = Some((name.clone(), *stamp, header, sets));
            break;
        }
    }
    let recent: Vec<Value> = files.iter().take(20).map(|(st, name)| obj(vec![("file", s(name)), ("stamp", n(*st))])).collect();
    let live_calls: Vec<Value> = crate::live::read_all(home)
        .iter()
        .filter_map(|p| p.get("stage_times").and_then(Value::array).ok().map(<[Value]>::to_vec))
        .flatten()
        .collect();
    let live_calls: Vec<Value> = live_calls.iter().skip(live_calls.len().saturating_sub(32)).cloned().collect();
    let num = |x: &str| Value::Number(x.to_owned());
    let (file, stamp, header, sets) = chosen.map_or((Value::Null, Value::Null, Value::Null, Vec::new()), |(f, st, h, sets)| {
        let sets = sets
            .into_iter()
            .map(|st| {
                let rows = st.stages.iter().map(|(name, med, p95, c)| obj(vec![("stage", s(name)), ("median_ms", num(med)), ("p95_ms", num(p95)), ("n", n(*c))])).collect();
                obj(vec![("set", s(&st.set)), ("summary", s(&st.summary)), ("warm", s(&st.warm)), ("stages", Value::Array(rows))])
            })
            .collect();
        (s(&f), n(st), s(&h), sets)
    });
    obj(vec![
        ("source", s("stage-bench")),
        ("live", Value::Bool(!live_calls.is_empty())),
        ("live_calls", Value::Array(live_calls)),
        ("file", file),
        ("stamp", stamp),
        ("header", header),
        ("sets", Value::Array(sets)),
        ("files", Value::Array(recent)),
        ("hook", s(STAGE_HOOK)),
    ])
}

fn content_type(p: &Path) -> &'static str {
    match p.extension().and_then(|x| x.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        "txt" => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

/// A file under `out/` for a URL path, refusing anything that climbs out of it.
fn static_file(out: &Path, url: &str) -> Option<PathBuf> {
    let rel = Path::new(url.trim_start_matches('/'));
    if rel.components().any(|c| !matches!(c, Component::Normal(_))) {
        return None;
    }
    let p = out.join(rel);
    let p = if p.is_dir() { p.join("index.html") } else { p };
    if p.is_file() {
        return Some(p);
    }
    // Next.js prefetch asks for `__next.<segment>.__PAGE__.txt`; the export writes it as
    // `__next.<segment>/__PAGE__.txt`.
    let name = p.file_name()?.to_str()?;
    if name.starts_with("__next.") {
        let alt = p.with_file_name(name.replacen(".__", "/__", 1));
        if alt.is_file() {
            return Some(alt);
        }
    }
    None
}

/// Writes one response. `cors` names the development origin allowed to read it, if any.
fn send(stream: &mut TcpStream, head_only: bool, code: &str, ctype: &str, body: &[u8], cors: Option<&str>) {
    send_with(stream, head_only, code, ctype, body, cors, "");
}
/// `send` with extra header lines (each ending in CRLF).
#[expect(clippy::too_many_arguments, reason = "the response line, its body and its headers")]
fn send_with(stream: &mut TcpStream, head_only: bool, code: &str, ctype: &str, body: &[u8], cors: Option<&str>, extra: &str) {
    let allow = cors.map_or_else(String::new, |o| format!("Access-Control-Allow-Origin: {o}\r\nVary: Origin\r\nAccess-Control-Allow-Methods: GET, POST\r\nAccess-Control-Allow-Headers: Content-Type, X-Nano-Token\r\n"));
    let head = format!(
        "HTTP/1.1 {code}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n{allow}{extra}Connection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    if !head_only {
        let _ = stream.write_all(body);
    }
}
fn send_json_error(stream: &mut TcpStream, code: &str, msg: &str, cors: Option<&str>) {
    let body = obj(vec![("error", s(msg))]).encode().unwrap_or_default();
    send(stream, false, code, "application/json", body.as_bytes(), cors);
}

const MAX_BODY: usize = 64 * 1024;
/// 2026-11-26 (Phase B): a parameter map's edit carries the map (a job filter or the role weights
/// run to about 2 MB), so that one route takes a larger body.
const MAX_MAP_BODY: usize = 8 * 1024 * 1024;

/// The largest body a request to `target` may carry.
fn body_limit(target: &str) -> usize {
    if target.split('?').next() == Some("/api/parameter-maps/save") { MAX_MAP_BODY } else { MAX_BODY }
}

impl Server {
    /// The interface's own origins on this port.
    fn own_origin(&self, o: &str) -> bool {
        o == format!("http://127.0.0.1:{}", self.port) || o == format!("http://localhost:{}", self.port)
    }
    /// The development origin a response may be shared with: only with `--dev`.
    fn dev_origin<'a>(&self, origin: Option<&'a str>) -> Option<&'a str> {
        origin.filter(|o| self.dev && DEV_ORIGINS.contains(o))
    }
    /// A request must name this machine as its host (or name none, as HTTP/1.0 clients may):
    /// a page that rebinds its own name to 127.0.0.1 still sends that name, and is refused.
    fn local_host(&self, host: Option<&str>) -> bool {
        host.is_none_or(|h| h == format!("127.0.0.1:{}", self.port) || h == format!("localhost:{}", self.port))
    }
    /// A write is accepted only with the local token, as JSON, and from the interface's own
    /// origin, the development origin under `--dev`, or no origin (a local program).
    fn write_allowed(&self, origin: Option<&str>, ctype: Option<&str>, token: Option<&str>) -> bool {
        let origin_ok = origin.is_none_or(|o| self.own_origin(o) || self.dev_origin(Some(o)).is_some());
        let token_ok = token.is_some_and(|t| network::same_secret(t.as_bytes(), self.token.as_bytes()));
        origin_ok && token_ok && ctype.is_some_and(|c| c.starts_with("application/json"))
    }
    /// A request from another device must name, as its host, the address it arrived on and this
    /// port (exactly; IPv4 only).
    fn remote_host(&self, host: Option<&str>, local: &SocketAddr) -> bool {
        self.addrs.contains(&local.ip()) && host == Some(format!("{}:{}", local.ip(), self.port).as_str())
    }
}

/// The current `network` setting.
fn network_mode(home: &Path) -> String {
    if crate::controls::air_gap_on(home) {
        return "off".to_owned();
    }
    crate::controls::current(home).get("network").cloned().unwrap_or_else(|| "off".to_owned())
}

fn write_api(home: &Path, path: &str, body: &[u8], by: &str) -> Result<Value, String> {
    let mut v = nmjson::parse(body).map_err(|e| format!("body: {e}"))?;
    // 2026-11-26 (Luis Jake Gabriel III: every delete verified in more than one step): a destructive
    // request first gets a preview and a confirmation token; only the same request with the token
    // deletes (`forget::gate`). MCP borrows these routes, so its deletes are gated the same way.
    if let Some(preview) = crate::forget::gate(home, path, &mut v)? {
        return Ok(preview);
    }
    let v = v;
    let field = |k: &str| v.get(k).and_then(Value::string).map_err(|e| format!("{k}: {e}"));
    if path == "/api/projects/forget" {
        return crate::forget::forget_project(home, field("project")?, by);
    }
    if path == "/api/memory/forget" {
        return crate::forget::forget_memory(home, field("project")?, field("id")?, by);
    }
    if path == "/api/audit/retention" {
        return crate::forget::retention(home, by);
    }
    // Roadmap Phase 4.5 (2026-11-24): storage controls. Prune and delete need `confirm: true`
    // and are audited in `nmapi::storage`; a storage setting is written by `set_by`.
    if let Some(op) = path.strip_prefix("/api/storage/") {
        let confirm = matches!(v.get("confirm"), Ok(Value::Bool(true)));
        return match op {
            "prune" => nmapi::storage::prune(home, field("category")?, confirm, by),
            "delete" => nmapi::storage::delete(home, field("path")?, confirm, by),
            "settings" => {
                let key = nmapi::storage::setting_key(field("key")?)?;
                let audit = crate::controls::set_by(home, key, field("value")?, None, by)?;
                Ok(obj(vec![("audit_id", audit.map_or(Value::Null, |i| s(&i))), ("settings", nmapi::storage::settings(home))]))
            }
            _ => Err("unknown endpoint".to_owned()),
        };
    }
    // Roadmap Phase 4.5 (2026-11-25): set one hardware limit (`{key, value}`), or apply this
    // machine's recommended values (`{confirm: true}`); each change is a logged setting.
    if path == "/api/limits" || path == "/api/limits/apply-recommended" {
        let mut set = |k: &str, x: &str| crate::controls::set_by(home, k, x, None, by);
        return if path == "/api/limits" {
            nmapi::limits::set_one(home, field("key")?, field("value")?, &mut set)
        } else {
            nmapi::limits::apply_recommended(home, matches!(v.get("confirm"), Ok(Value::Bool(true))), &mut set)
        };
    }
    // 2026-11-25: base cartridge versions: `{"model", "choice": key|auto|open}` chooses what serves
    // from the next start; `/remove` with `{"key", "confirm": true}` deletes one version.
    if path == "/api/cartridges/bases" {
        let audit = {
            // The choice is a key of `settings.json`: written under the controls' lock.
            let _g = crate::controls::write_guard()?;
            crate::library::choose_base(home, field("model")?, field("choice")?, by)?
        };
        return Ok(obj(vec![("audit_id", s(&audit)), ("bases", crate::library::base_versions(home))]));
    }
    // 2026-11-25: revoking a paired device from the MCP tools (`nm_settings op=device_revoke`);
    // the interface's server answers this path first.
    if path == "/api/devices/revoke" {
        network::revoke(home, field("id")?, by)?;
        return Ok(network::view(home, &network_mode(home), &[], 0));
    }
    if path == "/api/parameter-maps/remove" {
        if !matches!(v.get("confirm"), Ok(Value::Bool(true))) {
            return Err("confirm: removing a parameter map deletes its file; send confirm: true".into());
        }
        // 2026-11-26: several maps at once, or archived ones (`files`, `archived`).
        let audit = if v.get("files").is_ok() || matches!(v.get("archived"), Ok(Value::Bool(true))) {
            crate::pmaps::remove_many(home, field("role")?, &crate::pmaps::files_of(&v)?, matches!(v.get("archived"), Ok(Value::Bool(true))), by)?
        } else {
            crate::pmaps::remove(home, field("role")?, field("file")?, by)?
        };
        return Ok(obj(vec![("audit_id", s(&audit)), ("maps", crate::pmaps::view(home))]));
    }
    // 2026-11-26 (Luis Jake Gabriel III: "We have to have a way to clean up old maps too"): old maps
    // moved into the role's archive and back (reversible, audited with undo).
    if path == "/api/parameter-maps/archive" || path == "/api/parameter-maps/restore" {
        let files = crate::pmaps::files_of(&v)?;
        let role = field("role")?;
        let r = if path.ends_with("archive") { crate::pmaps::archive(home, role, &files, by)? } else { crate::pmaps::restore(home, role, &files, by)? };
        let mut out = r;
        if let Value::Object(m) = &mut out {
            m.insert("maps".into(), crate::pmaps::view(home));
        }
        return Ok(out);
    }
    // 2026-11-26 (Phase B items A2 to A6, B1 to B3): editing and importing maps, cartridge values,
    // trial, stacks and imports, learning units, lenses, memory notes, pins, sweeps, objectives and
    // the world-model refresh. Each runs the function its MCP tool runs.
    if let Some(r) = phase_b_write(home, path, &v, by) {
        return r;
    }
    if path == "/api/backups" {
        return crate::backups::api(home, &v, by);
    }
    if path == "/api/cartridge-sources" {
        let action = field("action")?;
        let id = v.get("id").and_then(Value::string).ok();
        let need_id = || id.ok_or_else(|| "id: give the source's id as /api/cartridge-sources lists it".to_owned());
        return match action {
            "add" => {
                let token = matches!(v.get("token"), Ok(Value::Bool(true)));
                let audit = crate::sources::add(home, field("repo")?, v.get("kind").and_then(Value::string).ok(), v.get("revision").and_then(Value::string).ok(), token, by)?;
                Ok(obj(vec![("audit_id", s(&audit)), ("sources", crate::sources::view(home))]))
            }
            "remove" => {
                let audit = crate::sources::remove(home, need_id()?, by)?;
                Ok(obj(vec![("audit_id", s(&audit)), ("sources", crate::sources::view(home))]))
            }
            "browse" => {
                let listing = crate::sources::browse(home, need_id()?, by)?;
                Ok(obj(vec![("listing", listing), ("sources", crate::sources::view(home))]))
            }
            "fetch" => {
                let line = crate::sources::fetch(home, need_id()?, field("path")?, by)?;
                Ok(obj(vec![("line", s(&line)), ("downloads", crate::fetch::view(home))]))
            }
            "fetch_all" => {
                let only = v.get("only").and_then(Value::string).ok();
                let line = crate::sources::fetch_all(home, need_id()?, only, by)?;
                Ok(obj(vec![("line", s(&line)), ("downloads", crate::fetch::view(home))]))
            }
            "publish" => {
                if !matches!(v.get("confirm"), Ok(Value::Bool(true))) {
                    return Err("confirm: publishing uploads a local file to Hugging Face; send confirm: true".into());
                }
                let req = crate::sources::PublishRequest {
                    repo: field("repo")?,
                    kind: v.get("kind").and_then(Value::string).ok(),
                    revision: v.get("revision").and_then(Value::string).ok(),
                    create_pr: matches!(v.get("create_pr"), Ok(Value::Bool(true))),
                    item_kind: field("item_kind")?,
                    role: v.get("role").and_then(Value::string).ok(),
                    item: field("item")?,
                    summary: v.get("summary").and_then(Value::string).ok(),
                };
                let published = crate::sources::publish(home, &req, by)?;
                Ok(obj(vec![("published", published), ("sources", crate::sources::view(home))]))
            }
            other => Err(format!("action {other}: use add, remove, browse, fetch, fetch_all or publish")),
        };
    }
    if path == "/api/downloads" {
        let line = crate::fetch::start(home, field("what")?, v.get("only").and_then(Value::string).ok(), by)?;
        return Ok(obj(vec![("line", s(&line)), ("downloads", crate::fetch::view(home))]));
    }
    if path == "/api/models/roles" {
        let (audit, line) = {
            let _g = crate::controls::write_guard()?;
            crate::library::choose_role(home, field("role")?, field("model")?, by)?
        };
        return Ok(obj(vec![("audit_id", s(&audit)), ("line", s(&line)), ("roles", crate::library::roles_view(home))]));
    }
    if path == "/api/cartridges/bases/remove" {
        if !matches!(v.get("confirm"), Ok(Value::Bool(true))) {
            return Err("confirm: removing a base cartridge deletes its file; send confirm: true".into());
        }
        let audit = crate::library::remove_base(home, field("key")?, by)?;
        return Ok(obj(vec![("audit_id", s(&audit)), ("bases", crate::library::base_versions(home))]));
    }
    if path == "/api/cartridges" && v.get("library").is_ok() {
        // The stack of cartridges for a use: `{"library": "<use>", "stack": "<stack id>"}`, or
        // `"stack": null` to switch the use off.
        let use_for = field("library")?;
        let stack = match v.get("stack") {
            Ok(Value::Null) => None,
            Ok(Value::String(x)) => Some(x.as_str()),
            _ => return Err("stack: give a stack id, or null to switch the use off".into()),
        };
        let audit = crate::library::switch_use(home, use_for, stack, by)?;
        return Ok(obj(vec![("audit_id", audit.map_or(Value::Null, |i| s(&i))), ("cartridges", all_cartridges(home))]));
    }
    // 2026-11-26 (Luis Jake Gabriel III: "We should be able to control and compartmentalized and
    // plug in and unplug the learned units"): `{"unit": "<compartment>/<id>", "on": bool}` or
    // `{"compartment": "<key>", "on": bool}`. A registered learning unit (`shared/<id>`) is plugged
    // through its registry, as on the cartridge rack.
    if path == "/api/learned-units" {
        let value = match (v.get("on"), v.get("value").and_then(Value::string)) {
            (Ok(Value::Bool(b)), _) => if *b { "on" } else { "off" },
            (_, Ok(x)) => x,
            _ => return Err("on: give on (true or false) or value (on or off)".into()),
        };
        let key = if let Ok(Value::String(unit)) = v.get("unit") {
            match unit.strip_prefix("shared/") {
                Some(id) if nmlearn::Units::open(home).get(id).is_some() => format!("{}{id}", crate::controls::CARTRIDGE_KEY),
                _ => format!("{}{unit}", crate::learned_units::LEARNED_KEY),
            }
        } else if let Ok(Value::String(c)) = v.get("compartment") {
            format!("{}{c}", crate::learned_units::COMPARTMENT_KEY)
        } else {
            return Err("give unit (<compartment>/<id>) or compartment, with on".into());
        };
        let audit = crate::controls::set_by(home, &key, value, None, by)?;
        return Ok(obj(vec![("audit_id", audit.map_or(Value::Null, |i| s(&i))), ("key", s(&key)), ("learned_units", crate::learned_units::view(home))]));
    }
    if path == "/api/cartridges" {
        let id = field("id")?;
        let value = match (v.get("on"), v.get("value").and_then(Value::string)) {
            (Ok(Value::Bool(b)), _) => if *b { "on" } else { "off" },
            (_, Ok(x)) => x,
            _ => return Err("on: give on (true or false) or value (on or off)".into()),
        };
        let audit = crate::controls::set_by(home, &format!("{}{id}", crate::controls::CARTRIDGE_KEY), value, None, by)?;
        return Ok(obj(vec![("audit_id", audit.map_or(Value::Null, |i| s(&i))), ("cartridges", all_cartridges(home))]));
    }
    if path == "/api/ledger/clear" {
        let key = field("project")?;
        let (key, memory) = crate::ledger::project_memory(home, &format!("project={key}")).ok_or_else(|| format!("project: no project {key}"))?;
        let (rows, audit) = crate::ledger::clear(home, &memory, &key, by)?;
        return Ok(obj(vec![("audit_id", s(&audit)), ("project", s(&key)), ("rows", n(rows as u64))]));
    }
    // 2026-11-25 (Luis Jake Gabriel III: "Allow users to prep their own air gapped data for their own
    // training"): the training store exported to a file (to carry to an air-gapped machine) or
    // imported from one (hash checked, nothing is added twice), and the user's own reference material
    // listed with its licence and hash (it stays where it is; nothing is copied or sent anywhere).
    if path == "/api/training/export" {
        let key = field("project")?;
        let (key, memory) = crate::ledger::project_memory(home, &format!("project={key}")).ok_or_else(|| format!("project: no project {key}"))?;
        let (file, rows) = crate::examples::export_to_exports(home, &memory, &key, v.get("name").and_then(Value::string).ok())?;
        return Ok(obj(vec![("project", s(&key)), ("file", s(&file.display().to_string())), ("rows", n(rows as u64))]));
    }
    if path == "/api/training/import" {
        let key = field("project")?;
        let file = field("file")?;
        let (key, memory) = crate::ledger::project_memory(home, &format!("project={key}")).ok_or_else(|| format!("project: no project {key}"))?;
        let rows = crate::examples::import(home, &memory, &key, Path::new(file))?;
        return Ok(obj(vec![("project", s(&key)), ("file", s(file)), ("rows", n(rows as u64))]));
    }
    // Distillation's teacher data: curated teacher pairs imported from a local file, and their
    // outcome labels (only healthy rows are read by a fit). Audited; Audit undoes both.
    if path == "/api/training/distil/teach" {
        return crate::teach::api(home, &v, by);
    }
    if path == "/api/training/distil/import" {
        let file = field("file")?;
        let opt = |k: &str| v.get(k).and_then(Value::string).ok().filter(|x| !x.trim().is_empty());
        return crate::distil::import(home, Path::new(file), opt("set"), opt("licence"), opt("teacher"), by);
    }
    if path == "/api/training/distil/label" {
        let outcome = field("outcome")?;
        let ids: Vec<String> = v.get("ids").and_then(Value::array).unwrap_or_default().iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect();
        let set = v.get("set").and_then(Value::string).ok();
        let why = v.get("why").and_then(Value::string).unwrap_or("");
        return crate::distil::label(home, &ids, set, outcome, why, by);
    }
    if path == "/api/training/references" {
        let p = field("path")?;
        let licence = field("licence")?.trim();
        if licence.is_empty() {
            return Err("licence: name the licence of the material (for example MIT, CC-BY-4.0, or own)".into());
        }
        let source = v.get("source").and_then(Value::string).unwrap_or("own material");
        let domain = v.get("domain").and_then(Value::string).unwrap_or("");
        let id = crate::examples::reference_add(home, Path::new(p), licence, source, domain)?;
        return Ok(obj(vec![("id", s(&id)), ("references", crate::examples::references_view(home))]));
    }
    if path == "/api/examples/clear" {
        let key = field("project")?;
        let (key, memory) = crate::ledger::project_memory(home, &format!("project={key}")).ok_or_else(|| format!("project: no project {key}"))?;
        let (rows, audit) = crate::examples::clear(home, &memory, &key, by)?;
        return Ok(obj(vec![("audit_id", s(&audit)), ("project", s(&key)), ("rows", n(rows as u64))]));
    }
    if path == "/api/surgery/job-filter" {
        let role = field("role")?;
        let job = field("job")?;
        let action = field("action")?;
        return update_job_filter(home, role, job, action, by);
    }
    // 2026-11-26: every lab build from one route: a job filter, a parameter map (with the base
    // cartridge version cut from it) or a base cartridge version by restoration.
    if path == "/api/lab/build" {
        let kind = field("kind")?;
        let role = field("role")?;
        let job = v.get("job").and_then(Value::string).unwrap_or("");
        let layers = v.get("layers").and_then(Value::string).unwrap_or("all");
        let share = match v.get("share") {
            Ok(Value::Number(x) | Value::String(x)) => x.trim().parse::<f64>().map_err(|_| format!("share {x}: a number such as 0.40"))?,
            _ => 0.40,
        };
        let device = v.get("device").and_then(Value::string).unwrap_or("gpu");
        let set = v.get("set").and_then(Value::string).unwrap_or("all");
        // A local teacher run names its teacher (an installed open-weights model) in `teacher`.
        let technique = if kind == "teach" { v.get("teacher").and_then(Value::string).unwrap_or("") } else { v.get("technique").and_then(Value::string).unwrap_or("lens") };
        let target = v.get("target").and_then(Value::string).unwrap_or("");
        let rounds = match v.get("rounds") {
            Ok(Value::Number(x) | Value::String(x)) => x.trim().parse::<u64>().map_err(|_| format!("rounds {x}: a whole number from 1 to 8"))?,
            _ => 0,
        };
        let budget_s = match v.get("budget_s") {
            Ok(Value::Number(x) | Value::String(x)) => x.trim().parse::<u64>().map_err(|_| format!("budget_s {x}: whole seconds from 1 to 120"))?,
            _ => 0,
        };
        return start_lab_build(home, &LabBuild { kind, role, job, layers, share, device, target, set, technique, rounds, budget_s }, by);
    }
    if path == "/api/surgery/job-filter/build" {
        let role = field("role")?;
        let job = field("job")?;
        let layers = v.get("layers").and_then(Value::string).unwrap_or("all");
        return start_job_filter_build(home, role, job, layers, by);
    }
    // Roadmap Phase 2: a rollback may name the unified cycle (`cycle`, its audit id).
    if path == "/api/learning/cycle" {
        let key = field("project")?;
        let action = v.get("action").and_then(Value::string).unwrap_or("cycle");
        let target_ver = v.get("version").and_then(Value::integer).ok();
        let (key, memory) = crate::ledger::project_memory(home, &format!("project={key}")).ok_or_else(|| format!("project: no project {key}"))?;
        return crate::examples::learning_cycle_api_at(home, &memory, &key, &crate::examples::CycleAsk::of(action, target_ver, v.get("cycle").and_then(Value::string).ok()), by);
    }
    if path == "/api/dense/clear" {
        // The project by its key as `/api/projects` lists it (`nano` for the Nano's own folder).
        let key = field("project")?;
        let memory = if key == "nano" {
            home.join("memory")
        } else {
            let dir = nmtype::project::projects_dir(home);
            let ok = !key.is_empty() && key.chars().all(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.')) && !key.starts_with('.');
            if !ok || !dir.join(key).is_dir() {
                return Err(format!("project: no project {key}"));
            }
            dir.join(key)
        };
        let (rows, bytes, audit) = crate::controls::dense_clear(home, &memory, key, by)?;
        return Ok(obj(vec![("audit_id", s(&audit)), ("project", s(key)), ("rows", n(rows)), ("bytes", n(bytes))]));
    }
    let id = match path {
        "/api/controls" => crate::controls::set_by(home, field("key")?, field("value")?, None, by)?,
        "/api/undo" => crate::controls::undo_by(home, field("id")?, by)?,
        _ => return Err("unknown endpoint".to_owned()),
    };
    Ok(obj(vec![("audit_id", id.map_or(Value::Null, |i| s(&i))), ("controls", crate::controls::view(home))]))
}

/// The Phase B writes (2026-11-26); `None` for any other path.
fn phase_b_write(home: &Path, path: &str, v: &Value, by: &str) -> Option<Result<Value, String>> {
    let field = |k: &str| v.get(k).and_then(Value::string).map_err(|e| format!("{k}: {e}"));
    let opt = |k: &str| v.get(k).and_then(Value::string).ok().filter(|x| !x.is_empty());
    let flag = |k: &str| matches!(v.get(k), Ok(Value::Bool(true)));
    let audit_reply = |a: Option<String>, extra: Vec<(&str, Value)>| {
        let mut f = vec![("audit_id", a.map_or(Value::Null, |i| s(&i)))];
        f.extend(extra);
        obj(f)
    };
    Some(match path {
        "/api/parameter-maps/save" => (|| {
            let (role, file) = (field("role")?, field("file")?);
            if file.starts_with("job-filter-") && build_running(home) {
                return Err("a job filter build is running; save when it ends".to_owned());
            }
            let content = v.get("content").map_err(|_| "content: give the map's new value as JSON".to_owned())?;
            let layer = match v.get("layer") {
                Ok(Value::Number(x)) => Some(x.clone()),
                Ok(Value::String(x)) if !x.is_empty() => Some(x.clone()),
                _ => None,
            };
            crate::pmaps::save(home, role, file, opt("section"), layer.as_deref(), content, by)
        })(),
        // 2026-11-26: where a map came from and what it is for (nmapi::lineage).
        "/api/parameter-maps/lineage" => (|| crate::pmaps::set_lineage(home, field("role")?, field("file")?, v, by))(),
        "/api/parameter-maps/import" => (|| crate::pmaps::import(home, field("role")?, field("path")?, opt("name"), flag("replace"), by))(),
        "/api/cartridges/values" => (|| {
            let key = field("key")?;
            let Ok(Value::Object(changes)) = v.get("values") else { return Err("values: give an object of name to value (text, or null to take a preference key out)".to_owned()) };
            let mut r = crate::library::set_values(home, key, changes, by)?;
            if let Value::Object(m) = &mut r {
                m.insert("inspect".into(), crate::library::inspect(home, &format!("key={}", m.get("key").and_then(|k| k.string().ok()).unwrap_or(key))));
            }
            Ok(r)
        })(),
        // A cartridge's parameters edited into a new version (`cparams::edit`): a strength, layers
        // switched or scaled, values set; the source is never written; audited with undo.
        "/api/cartridges/parameters" => (|| {
            let key = field("key")?;
            crate::cparams::edit(home, key, &crate::cparams::parse_edit(v)?, by)
        })(),
        // A selection's values written to exports/parameters/ (nothing of the object is written).
        "/api/cartridges/parameters/export" => (|| crate::cparams::export(home, field("key")?, field("path")?))(),
        // A change of an object's parameter overlay (`overlay::post`): edit, mask, unmask, freeze,
        // unfreeze, clear (through the delete gate) or undo, with a preview; audited with undo.
        "/api/parameters/overlay" => crate::overlay::post(home, v, by),
        "/api/cartridges/trial" => (|| {
            let name = field("name")?;
            let trial = match v.get("trial") {
                Ok(Value::Bool(b)) => *b,
                _ => return Err("trial: true puts the stack on trial, false serves it (needs its evidence)".to_owned()),
            };
            let a = crate::library::trial_api(home, name, trial, by)?;
            Ok(audit_reply(a, vec![("stack", s(name)), ("trial", Value::Bool(trial)), ("library", crate::library::api_view(home))]))
        })(),
        "/api/cartridges/stack" => (|| {
            let carts: Vec<String> = v.get("cartridges").and_then(Value::array).map_err(|_| "cartridges: give the cartridge keys in order".to_owned())?.iter().map(|c| c.string().map(str::to_owned).map_err(|_| "cartridges: keys are text".to_owned())).collect::<Result<_, _>>()?;
            let switch = match v.get("switch") {
                Ok(Value::Bool(b)) => Some(*b),
                _ => None,
            };
            crate::library::stack_save(home, field("id")?, field("use")?, &carts, opt("description"), switch, by)
        })(),
        "/api/cartridges/import" => (|| crate::library::import_api(home, field("path")?, by))(),
        "/api/units" => (|| {
            let id = field("id")?;
            match field("action")? {
                "load" | "unload" => {
                    let a = crate::controls::set_by(home, &format!("{}{id}", crate::controls::CARTRIDGE_KEY), if field("action")? == "load" { "on" } else { "off" }, None, by)?;
                    Ok(audit_reply(a, vec![("unit", s(id)), ("units", crate::learned_units::units_view(home))]))
                }
                "unregister" => {
                    let a = crate::learned_units::unregister(home, id, by)?;
                    Ok(audit_reply(Some(a), vec![("unit", s(id)), ("units", crate::learned_units::units_view(home))]))
                }
                "register" => {
                    // Binding the MicroModel checks the target's widths: only a serving process with
                    // the engine bound can (an MCP call lands here in that process).
                    let o = nmapi::tools::installed().ok_or("register binds the MicroModel to check the target node's widths, so it runs in a serving process with the engine bound: use nm_unit op=register there")?;
                    let mut req = v.clone();
                    if let Value::Object(m) = &mut req {
                        m.insert("op".into(), s("register"));
                    }
                    let card = nmapi::learn::tool(o, &req).map_err(|e| e.to_string())?;
                    let a = crate::controls::audit(home, "unit", vec![("action", s("register")), ("unit", s(id)), ("by", s(by)), ("label", s(&format!("Learning unit {id} registered")))])?;
                    Ok(audit_reply(Some(a), vec![("unit", s(id)), ("card", card), ("units", crate::learned_units::units_view(home))]))
                }
                other => Err(format!("action {other}: use load, unload, register or unregister")),
            }
        })(),
        "/api/lenses" => (|| {
            let action = field("action")?;
            let id = opt("id").or_else(|| opt("unit")).ok_or("id: give the lens id (embedder-lens-<mark>)")?;
            match action {
                "enable" | "disable" => {
                    let key = crate::learned_units::lens_key(home, id)?;
                    let a = crate::controls::set_by(home, &format!("{}{key}", crate::learned_units::LEARNED_KEY), if action == "enable" { "on" } else { "off" }, None, by)?;
                    Ok(audit_reply(a, vec![("key", s(&key)), ("lenses", crate::learned_units::lenses_view(home, ""))]))
                }
                "remove" => {
                    let id = id.rsplit('/').next().unwrap_or(id);
                    crate::learned_units::lens_remove(home, id, v.get("version").and_then(Value::integer).ok(), by)
                }
                other => Err(format!("action {other}: use enable, disable or remove")),
            }
        })(),
        "/api/memory/add" => crate::memory_api::add(home, v, by),
        "/api/memory/pin" => crate::memory_api::pin(home, v, by),
        "/api/memory/sweep" => crate::memory_api::sweep(home, v, by),
        "/api/goals" => crate::memory_api::goals_set(home, v, by),
        "/api/guidance" => crate::guidance::write(home, v, by),
        "/api/comprehension/refresh" => (|| crate::memory_api::refresh(home, field("project")?, by))(),
        // Stated preference cartridges: the user's words to rules (a preview, then confirm), edits,
        // the gated removal, plug and unplug (a library use switch), and a draft checked.
        "/api/preferences/say" => crate::preferences::say(home, v, by),
        "/api/preferences/edit" => crate::preferences::edit(home, v, by),
        "/api/preferences/remove" => (|| crate::preferences::remove(home, field("id")?, by))(),
        "/api/preferences/plug" => (|| {
            let on = match v.get("on") {
                Ok(Value::Bool(b)) => *b,
                _ => return Err("on: true plugs the cartridge in, false unplugs it".to_owned()),
            };
            crate::preferences::plug(home, field("id")?, on, by)
        })(),
        "/api/preferences/apply" => crate::preferences::apply(home, v),
        _ => return None,
    })
}

/// One parsed request.
#[derive(Default)]
struct Req {
    method: String,
    target: String,
    origin: Option<String>,
    host: Option<String>,
    token: Option<String>,
    cookie: Option<String>,
    ctype: Option<String>,
    length: usize,
    body: Vec<u8>,
}
impl Req {
    fn path_query(&self) -> (&str, &str) {
        self.target.split_once('?').unwrap_or((self.target.as_str(), ""))
    }
    /// The device token in the `nano_device` cookie, if any.
    fn cookie_token(&self) -> Option<&str> {
        self.cookie.as_deref()?.split(';').find_map(|kv| kv.trim().strip_prefix(DEVICE_COOKIE)?.strip_prefix('='))
    }
    fn json(&self) -> bool {
        self.ctype.as_deref().is_some_and(|c| c.starts_with("application/json"))
    }
}

fn read_request(stream: &TcpStream) -> Option<Req> {
    let mut req = Req::default();
    let mut line = String::new();
    let mut r = BufReader::new(stream);
    r.read_line(&mut line).ok()?;
    let mut h = String::new();
    while r.read_line(&mut h).is_ok_and(|k| k > 2) {
        if let Some((k, v)) = h.split_once(':') {
            let (k, v) = (k.trim().to_ascii_lowercase(), v.trim().to_owned());
            match k.as_str() {
                "origin" => req.origin = Some(v),
                "host" => req.host = Some(v.to_ascii_lowercase()),
                TOKEN_HEADER => req.token = Some(v),
                "cookie" => req.cookie = Some(v),
                "content-type" => req.ctype = Some(v.to_ascii_lowercase()),
                "content-length" => req.length = v.parse().unwrap_or(0),
                _ => {}
            }
        }
        h.clear();
    }
    let mut it = line.split_whitespace();
    req.method = it.next().unwrap_or("").to_owned();
    req.target = it.next().unwrap_or("/").to_owned();
    if req.length > 0 && req.length <= body_limit(&req.target) {
        req.body.resize(req.length, 0);
        std::io::Read::read_exact(&mut r, &mut req.body).ok()?;
    }
    Some(req)
}

fn handle(srv: &Server, mut stream: TcpStream) {
    let Some(req) = read_request(&stream) else { return };
    match (stream.peer_addr(), stream.local_addr()) {
        (Ok(peer), Ok(local)) if !peer.ip().is_loopback() => handle_remote(srv, &mut stream, &req, peer.ip(), &local),
        (Ok(_), Ok(_)) => handle_local(srv, &mut stream, &req),
        _ => {}
    }
}

/// A static file or page; HTML pages carry `token` in their head.
fn serve_static(srv: &Server, stream: &mut TcpStream, path: &str, target: &str, head_only: bool, token: &str) {
    let out = srv.home.join("interface").join("out");
    let page = |p: &Path, body: Vec<u8>| if content_type(p).starts_with("text/html") { with_token(&body, token) } else { body };
    match static_file(&out, path).and_then(|p| Some((std::fs::read(&p).ok()?, p))) {
        Some((body, p)) => send(stream, head_only, "200 OK", content_type(&p), &page(&p, body), None),
        None => {
            eprintln!("not found: {target}");
            let body = std::fs::read(out.join("404.html"))
                .map_or_else(|_| b"not found; build the interface with npm run build in interface/".to_vec(), |b| with_token(&b, token));
            send(stream, head_only, "404 Not Found", "text/html; charset=utf-8", &body, None);
        }
    }
}

/// A GET or HEAD on `/api/...`, the same for this machine and paired devices.
fn serve_api(srv: &Server, stream: &mut TcpStream, path: &str, query: &str, head_only: bool, cors: Option<&str>) {
    let home = srv.home.as_path();
    // Roadmap Phase 4.5 (2026-11-24): live activity streamed from the engine (`crate::live`):
    // `/api/live` the whole state, `/api/live/stream` its new timeline events as Server-Sent
    // Events on this connection's own thread.
    if path == "/api/live/stream" && !head_only {
        crate::live::stream(home, stream, cors);
        return;
    }
    let v = if path == "/api/devices" {
        Some(network::view(home, &network_mode(home), &srv.addrs, srv.port))
    } else if path == "/api/airgap" {
        Some(crate::controls::air_gap_view(home, srv.addrs.len()))
    } else if path == "/api/live" {
        Some(crate::live::api_view(home))
    } else {
        api(home, path, query)
    };
    match v.map(|v| v.encode()) {
        Some(Ok(body)) => send(stream, head_only, "200 OK", "application/json", body.as_bytes(), cors),
        Some(Err(e)) => send(stream, head_only, "500 Internal Server Error", "text/plain", e.to_string().as_bytes(), cors),
        None => send(stream, head_only, "404 Not Found", "application/json", b"{\"error\":\"unknown endpoint\"}", cors),
    }
}

/// A request from this machine (the loopback listener).
fn handle_local(srv: &Server, stream: &mut TcpStream, req: &Req) {
    let home = srv.home.as_path();
    let (path, query) = req.path_query();
    let head_only = req.method == "HEAD";
    let cors = srv.dev_origin(req.origin.as_deref());
    if !srv.local_host(req.host.as_deref()) {
        send(stream, false, "403 Forbidden", "text/plain", b"this server answers only requests addressed to 127.0.0.1 or localhost", None);
        return;
    }
    match req.method.as_str() {
        "OPTIONS" => {
            send(stream, true, "204 No Content", "text/plain", b"", cors);
            return;
        }
        "POST" => {
            if !srv.write_allowed(req.origin.as_deref(), req.ctype.as_deref(), req.token.as_deref()) || req.length > body_limit(&req.target) {
                send(
                    stream,
                    false,
                    "403 Forbidden",
                    "text/plain",
                    b"writes are accepted only from this machine's interface, as JSON, with its local token",
                    cors,
                );
                return;
            }
            let out = match path {
                "/api/pair/code" => {
                    if crate::controls::air_gap_on(home) {
                        Err("air_gap is on: local network pairing is disabled".to_owned())
                    } else {
                        network::new_code(home, "interface")
                            .map(|(code, expires)| obj(vec![("code", s(&code)), ("expires_ms", n(expires))]))
                            .map_err(|e| e.to_string())
                    }
                }
                "/api/devices/revoke" => nmjson::parse(&req.body)
                    .map_err(|e| format!("body: {e}"))
                    .and_then(|v| v.get("id").and_then(Value::string).map(str::to_owned).map_err(|e| format!("id: {e}")))
                    .and_then(|id| network::revoke(home, &id, "interface"))
                    .map(|()| network::view(home, &network_mode(home), &srv.addrs, srv.port)),
                _ => write_api(home, path, &req.body, "interface"),
            };
            match out.and_then(|v| v.encode().map_err(|e| e.to_string())) {
                Ok(body) => send(stream, false, "200 OK", "application/json", body.as_bytes(), cors),
                Err(e) => send_json_error(stream, "400 Bad Request", &e, cors),
            }
            return;
        }
        "GET" | "HEAD" => {}
        _ => {
            send(stream, false, "405 Method Not Allowed", "text/plain", b"GET, HEAD or POST", cors);
            return;
        }
    }
    if path == "/api/token" {
        // Only the development server's pages need this (the served pages carry the token);
        // it is answered to that origin alone, and only under `--dev`.
        match cors {
            Some(_) => {
                let body = obj(vec![("token", s(&srv.token))]).encode().unwrap_or_default();
                send(stream, head_only, "200 OK", "application/json", body.as_bytes(), cors);
            }
            None => send(stream, head_only, "404 Not Found", "application/json", b"{\"error\":\"unknown endpoint\"}", None),
        }
        return;
    }
    if path.starts_with("/api/") {
        serve_api(srv, stream, path, query, head_only, cors);
        return;
    }
    serve_static(srv, stream, path, &req.target, head_only, &srv.token);
}

/// A request from another device on the local network (see the module note for the rules).
fn handle_remote(srv: &Server, stream: &mut TcpStream, req: &Req, peer: IpAddr, local: &SocketAddr) {
    let home = srv.home.as_path();
    let mode = network_mode(home);
    if mode == "off" || !network::is_private(&peer) {
        send(stream, false, "403 Forbidden", "text/plain", b"local network access is off, or this address is not on the local network", None);
        return;
    }
    if !srv.remote_host(req.host.as_deref(), local) {
        send(stream, false, "403 Forbidden", "text/plain", b"this server answers only requests addressed to its own address and port", None);
        return;
    }
    let (path, query) = req.path_query();
    let head_only = req.method == "HEAD";
    let own = format!("http://{}:{}", local.ip(), srv.port);
    let origin_ok = req.origin.as_deref().is_none_or(|o| o == own);
    if req.method == "OPTIONS" {
        send(stream, true, "204 No Content", "text/plain", b"", None);
        return;
    }
    if req.method == "POST" && path == "/api/pair" {
        if !origin_ok || !req.json() || req.length > body_limit(&req.target) {
            send_json_error(stream, "403 Forbidden", "pairing takes JSON from this server's own page", None);
            return;
        }
        let v = nmjson::parse(&req.body).unwrap_or(Value::Null);
        let code = v.get("code").and_then(Value::string).unwrap_or("");
        let name = v.get("name").and_then(Value::string).unwrap_or("tablet");
        match network::pair(home, code, name, &peer.to_string()) {
            Ok((d, token)) => {
                let body = obj(vec![("device_id", s(&d.id)), ("name", s(&d.name))]).encode().unwrap_or_default();
                let cookie = format!("Set-Cookie: {DEVICE_COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=31536000\r\n");
                send_with(stream, false, "200 OK", "application/json", body.as_bytes(), None, &cookie);
            }
            Err(network::PairError::Locked(t)) => {
                send_json_error(stream, "429 Too Many Requests", &network::PairError::Locked(t).to_string(), None);
            }
            Err(e) => send_json_error(stream, "403 Forbidden", &e.to_string(), None),
        }
        return;
    }
    let header_token = req.token.as_deref();
    let device = header_token.or_else(|| req.cookie_token()).and_then(|t| network::authenticate(home, t));
    let Some(dev) = device else {
        if matches!(req.method.as_str(), "GET" | "HEAD") && !path.starts_with("/api/") {
            send(stream, head_only, "200 OK", "text/html; charset=utf-8", network::PAIR_PAGE.as_bytes(), None);
        } else {
            send_json_error(stream, "401 Unauthorized", "pair this device first", None);
        }
        return;
    };
    // The device's first request of this server's run is audited.
    if srv.seen.lock().is_ok_and(|mut seen| seen.insert(dev.id.clone())) {
        let _ = crate::controls::audit(
            home,
            "device",
            vec![
                ("label", s(&format!("Device connected: {}", dev.name))),
                ("device", s(&dev.id)),
                ("device_name", s(&dev.name)),
                ("peer", s(&peer.to_string())),
                ("network", s(&mode)),
            ],
        );
    }
    match req.method.as_str() {
        "POST" => {
            if mode != "control" {
                send_json_error(stream, "403 Forbidden", "this device may read only (network: view)", None);
                return;
            }
            // A write needs the device's token in the header (a cookie alone is not enough).
            let header_ok = header_token.and_then(|t| network::authenticate(home, t)).is_some_and(|d| d.id == dev.id);
            if !origin_ok || !req.json() || !header_ok || req.length > body_limit(&req.target) {
                send_json_error(stream, "403 Forbidden", "writes need JSON from this server's own page with the device's token", None);
                return;
            }
            if matches!(path, "/api/pair/code" | "/api/devices/revoke") {
                send_json_error(stream, "403 Forbidden", "pairing and revoking are done on the computer itself", None);
                return;
            }
            match write_api(home, path, &req.body, &dev.name).and_then(|v| v.encode().map_err(|e| e.to_string())) {
                Ok(body) => send(stream, false, "200 OK", "application/json", body.as_bytes(), None),
                Err(e) => send_json_error(stream, "400 Bad Request", &e, None),
            }
        }
        "GET" | "HEAD" => {
            if path == "/api/token" {
                send(stream, head_only, "404 Not Found", "application/json", b"{\"error\":\"unknown endpoint\"}", None);
            } else if path.starts_with("/api/") {
                serve_api(srv, stream, path, query, head_only, None);
            } else {
                // The device's pages carry its own token, never the local one.
                let token = header_token.or_else(|| req.cookie_token()).unwrap_or("");
                serve_static(srv, stream, path, &req.target, head_only, token);
            }
        }
        _ => send(stream, false, "405 Method Not Allowed", "text/plain", b"GET, HEAD or POST", None),
    }
}

/// Runs the interface server until the process ends.
/// # Errors
/// A bad `--port` value, or the port could not be bound on 127.0.0.1.
pub fn run(home: &Path, args: &[String]) -> Result<(), String> {
    let port = args
        .windows(2)
        .find(|w| w.first().map(String::as_str) == Some("--port"))
        .and_then(|w| w.get(1))
        .map_or(Ok(DEFAULT_PORT), |p| p.parse::<u16>().map_err(|e| format!("--port: {e}")))?;
    let dev = args.iter().any(|a| a == "--dev");
    let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|e| format!("127.0.0.1:{port}: {e}"))?;
    let mode = network_mode(home);
    let mut lan: Vec<(IpAddr, TcpListener)> = Vec::new();
    if mode != "off" {
        for a in network::local_addresses() {
            match TcpListener::bind((a, port)) {
                Ok(l) => lan.push((a, l)),
                Err(e) => eprintln!("local network: {a}:{port}: {e}"),
            }
        }
    }
    let dev_note = if dev { "; development origin http://localhost:3000 accepted" } else { "" };
    eprintln!("Neuromorph Nano interface: http://127.0.0.1:{port}/ (local only{dev_note}; models {})", nmtype::models_dir(home).display());
    if mode != "off" {
        let list: Vec<String> = lan.iter().map(|(a, _)| format!("http://{a}:{port}/")).collect();
        eprintln!(
            "local network ({mode}): {}. No encryption on the wire. Pair a tablet with `neuromorph-nano pair`.",
            if list.is_empty() { "no private address found".to_owned() } else { list.join(", ") }
        );
    }
    let srv = std::sync::Arc::new(Server {
        home: home.to_path_buf(),
        port,
        token: new_token(),
        dev,
        addrs: lan.iter().map(|(a, _)| *a).collect(),
        seen: Mutex::new(BTreeSet::new()),
    });
    for (_, l) in lan {
        let srv = std::sync::Arc::clone(&srv);
        std::thread::spawn(move || {
            for stream in l.incoming().flatten() {
                let srv = std::sync::Arc::clone(&srv);
                std::thread::spawn(move || handle(&srv, stream));
            }
        });
    }
    for stream in listener.incoming().flatten() {
        let srv = std::sync::Arc::clone(&srv);
        std::thread::spawn(move || handle(&srv, stream));
    }
    Ok(())
}

/// `neuromorph-nano pair`: makes a pairing code for a tablet (valid five minutes, once) and prints
/// it with the addresses to open and the setting in force.
pub fn pair_command(home: &Path) -> std::process::ExitCode {
    use std::process::ExitCode;
    if crate::controls::air_gap_on(home) {
        eprintln!("pair: air_gap is on (local network access and pairing are disabled)");
        return ExitCode::FAILURE;
    }
    let mode = network_mode(home);
    match network::new_code(home, "pair command") {
        Ok((code, _)) => {
            println!("pairing code: {code} (valid five minutes, once)");
            let addrs = network::local_addresses();
            let port = DEFAULT_PORT;
            for a in &addrs {
                println!("open on the tablet: http://{a}:{port}/ (or the port the interface server was started with)");
            }
            if mode == "off" {
                println!("the setting network is off: set it to view or control and restart the interface server first");
            } else {
                println!("network: {mode}");
            }
            println!("no encryption on the wire: anyone who can listen on this local network can read the traffic, including the tablet's token");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("pair: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Section check (no network): tokens are 64 hex digits and differ between servers; the
/// comparison accepts only the exact token; a write needs the token, JSON and an allowed origin;
/// the development origin counts only under `--dev`; foreign hosts are refused, also for other
/// devices, whose host must be the address they reached and the port; pages get the token after
/// `<head>`; a cartridge's layer comes from its target.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let (a, b) = (new_token(), new_token());
    if a.len() != 64 || !a.chars().all(|c| c.is_ascii_hexdigit()) || a == b {
        return Err("ui: tokens must be 64 hex digits and fresh per server".into());
    }
    if !network::same_secret(a.as_bytes(), a.as_bytes())
        || network::same_secret(a.as_bytes(), b.as_bytes())
        || network::same_secret(a.as_bytes(), a.get(..63).unwrap_or_default().as_bytes())
    {
        return Err("ui: token comparison".into());
    }
    let lan: IpAddr = "192.168.1.20".parse().map_err(|e: std::net::AddrParseError| e.to_string())?;
    let server = |dev: bool| Server {
        home: PathBuf::from("/x/nano"),
        port: 4870,
        token: a.clone(),
        dev,
        addrs: vec![lan],
        seen: Mutex::new(BTreeSet::new()),
    };
    let (srv, dev) = (server(false), server(true));
    let json = Some("application/json");
    let at = SocketAddr::new(lan, 4870);
    let other = SocketAddr::new("10.0.0.9".parse().map_err(|e: std::net::AddrParseError| e.to_string())?, 4870);
    let cases = [
        (srv.write_allowed(None, json, Some(&a)), true, "local program with the token"),
        (srv.write_allowed(Some("http://127.0.0.1:4870"), json, Some(&a)), true, "own page with the token"),
        (srv.write_allowed(None, json, None), false, "no token"),
        (srv.write_allowed(None, json, Some(&b)), false, "wrong token"),
        (srv.write_allowed(None, Some("text/plain"), Some(&a)), false, "not JSON"),
        (srv.write_allowed(Some("http://evil.example"), json, Some(&a)), false, "foreign origin"),
        (srv.write_allowed(Some("http://localhost:3000"), json, Some(&a)), false, "dev origin without --dev"),
        (dev.write_allowed(Some("http://localhost:3000"), json, Some(&a)), true, "dev origin with --dev"),
        (srv.local_host(Some("evil.example:4870")), false, "foreign host"),
        (srv.local_host(Some("127.0.0.1:4870")), true, "own host"),
        (srv.dev_origin(Some("http://localhost:3000")).is_some(), false, "token endpoint without --dev"),
        (srv.remote_host(Some("192.168.1.20:4870"), &at), true, "device naming the address it reached"),
        (srv.remote_host(Some("evil.example:4870"), &at), false, "device with a foreign host"),
        (srv.remote_host(Some("192.168.1.20:80"), &at), false, "device naming another port"),
        (srv.remote_host(Some("10.0.0.9:4870"), &other), false, "an address not listened on"),
        (srv.remote_host(None, &at), false, "device naming no host"),
    ];
    if let Some((_, _, what)) = cases.iter().find(|(got, want, _)| got != want) {
        return Err(format!("ui: write rule: {what}"));
    }
    let page = with_token(b"<!DOCTYPE html><html><head><title>x</title></head></html>", &a);
    if !String::from_utf8_lossy(&page).contains(&format!("<head><meta name=\"nano-token\" content=\"{a}\"><title>")) {
        return Err("ui: token not placed in the page head".into());
    }
    if layer_of("nmengn.components#L27.down") != Some(27) || layer_of("x#final_norm").is_some() {
        return Err("ui: cartridge layer from its target".into());
    }
    let req = Req { cookie: Some(format!("a=1; {DEVICE_COOKIE}={a}; b=2")), ..Req::default() };
    if req.cookie_token() != Some(a.as_str()) {
        return Err("ui: device cookie".into());
    }
    if !safe_slug("code-slice") || safe_slug("../etc") || !safe_layers("all") || !safe_layers("12-27") || safe_layers("12;rm") {
        return Err("ui: surgery slug/layers validation".into());
    }
    check_comprehension()?;
    check_stage_bench()
}

/// The note store is read by the writer's field names (`k`, `by`, `t`), a repeated key counts
/// once, tags map to the four levels, and the newest outline is the last one written.
fn check_comprehension() -> Result<(), String> {
    let lines = [
        r#"{"k":"note:a1","by":"w","t":"piece a"}"#,
        r#"{"k":"note:a1","by":"w","t":"piece a again"}"#,
        r#"{"k":"names:a1","by":"w","t":"q"}"#,
        r#"{"k":"sum:a1","by":"w","t":"s"}"#,
        r#"{"k":"file:f1","by":"w","t":"f"}"#,
        r#"{"k":"module:crates/x","by":"w","t":"m"}"#,
        r#"{"k":"outline:old","by":"w","t":"first"}"#,
        r#"{"k":"outline:new","by":"w","t":"second"}"#,
        r#"{"key":"note:zz","text":"old field names are not the store"}"#,
        "not json",
    ];
    let store = lines.join("\n");
    let (tags, total, outline) = read_notes(&store);
    let want: [(&str, u64); 6] = [("note", 1), ("names", 1), ("sum", 1), ("file", 1), ("module", 1), ("outline", 2)];
    if total != 7 || want.iter().any(|(t, c)| tags.get(*t) != Some(c)) {
        return Err(format!("ui: note store counts {tags:?} total {total}"));
    }
    if outline != Some(("outline:new".to_owned(), "second".to_owned())) {
        return Err("ui: the newest outline is the last one written".into());
    }
    let levels = [("note", Some(1)), ("names", Some(1)), ("sum", Some(1)), ("file", Some(2)), ("module", Some(3)), ("outline", Some(4)), ("other", None)];
    if let Some((t, _)) = levels.iter().find(|(t, l)| level_of(t) != *l) {
        return Err(format!("ui: note tag {t} on the wrong level"));
    }
    Ok(())
}

/// A stage-bench report's stage tables parse to their sets and rows, and the program path is
/// left out of the header.
fn check_stage_bench() -> Result<(), String> {
    let text = [
        r"stage-bench base  2026-11-24T03:05:19Z  program C:\x\nano.exe  env []",
        "",
        "large-code   first  27 top3  39",
        "  warm question ms: median 3349 p95 4933 min 2370 max 6056 (n 70)",
        "  stages, large-code (ms per warm question: median / p95 / n)",
        "    sense               916.0   1173.0    70",
        "    total              3319.0   4902.0    70",
        r"  raw: C:\x\a.txt",
        "",
        "all sets: first 88",
    ]
    .join("\n");
    let (header, sets) = parse_stage_bench(&text);
    let ok = header == "stage-bench base  2026-11-24T03:05:19Z"
        && sets.len() == 1
        && sets.first().is_some_and(|st| {
            st.set == "large-code"
                && st.summary.starts_with("large-code")
                && st.warm.starts_with("median 3349")
                && st.stages.len() == 2
                && st.stages.first().is_some_and(|r| r.0 == "sense" && r.1 == "916.0" && r.2 == "1173.0" && r.3 == 70)
        });
    if !ok || bench_stamp("STAGE-BENCH-17.txt") != Some(17) || bench_stamp("STAGE-BENCH-../x.txt").is_some() {
        return Err("ui: stage-bench tables".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-11-26: claiming the build slot never waits on itself (the state lock is not
    /// re-entrant), a second claim while one runs is refused, and the status read answers while a
    /// build runs. Each step runs on its own thread with a deadline, so a deadlock fails the test
    /// instead of hanging it.
    #[test]
    fn a_build_claim_does_not_wait_on_itself() {
        let home = nmtype::check_dir("nmcore-ui-claim").expect("test dir");
        let within = |f: Box<dyn FnOnce() -> bool + Send>| {
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let _ = tx.send(f());
            });
            rx.recv_timeout(std::time::Duration::from_secs(10)).unwrap_or(false)
        };
        let st = |status: &str| JobFilterBuildState { status: status.to_owned(), kind: "filter".to_owned(), role: "reranker".to_owned(), job: "t".to_owned(), ..JobFilterBuildState::default() };
        if let Ok(mut g) = JOB_FILTER_BUILD.lock() {
            *g = Some(st("done"));
        }
        let (h1, s1) = (home.clone(), st("running"));
        assert!(within(Box::new(move || claim_build(&h1, s1).is_ok())), "a claim after a finished build must succeed without waiting on itself");
        let (h2, s2) = (home.clone(), st("running"));
        assert!(within(Box::new(move || claim_build(&h2, s2).is_err())), "a second claim while one runs must be refused");
        let h3 = home.clone();
        assert!(within(Box::new(move || build_running(&h3))), "the status read must answer while a build runs");
        if let Ok(mut g) = JOB_FILTER_BUILD.lock() {
            *g = None;
        }
        let _ = std::fs::remove_dir_all(&home);
    }
}
