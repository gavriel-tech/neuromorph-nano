//! Every learned unit by compartment, each plugged in or out (2026-11-26, Luis Jake Gabriel III,
//! his words: "We should be able to control and compartmentalized and plug in and unplug the
//! learned units").
//!
//! Three kinds, each listed from its own store with exact paths, sizes and dates:
//! - `adapter`: the project adapters on the adapter shelf (`library/cartridges/adapter/<id>/`), the
//!   embedder's lens and the judge's code read-out, fitted at idle from the project's own material;
//!   compartment: the project named in the label (`notes.project`).
//! - `piece`: the learned pieces of `nm_unit` (`units/learned/` for the Nano's own folder,
//!   `memory/projects/<key>/learned/` for a project): the L4 calibrator (the judge for asking more)
//!   and the L1 settings.
//! - `unit`: the registered learning units (`units/registry.json`), compartment `shared`; plugged in
//!   and out through their registry (`cartridge:<id>`, hash-checked load), loaded by `serve --unit`.
//!
//! The plug state of adapters and pieces, and every compartment switch, live in `units/plugs.json`
//! (`nmapi::plugs`), read again by the serving process when the file changes. Each change is a
//! `setting` row in the audit log under `learned:<compartment>/<id>` or `compartment:<key>`.
use nmjson::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The key prefix of a learned unit's plug in the audit log (`learned:<compartment>/<id>`).
pub const LEARNED_KEY: &str = "learned:";
/// The key prefix of a compartment switch in the audit log (`compartment:<key>`).
pub const COMPARTMENT_KEY: &str = "compartment:";

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
fn read_json(p: &Path) -> Option<Value> {
    std::fs::read(p).ok().and_then(|b| nmjson::parse(&b).ok())
}
fn file_bytes(p: &Path) -> u64 {
    std::fs::metadata(p).map_or(0, |m| m.len())
}
fn dir_bytes(p: &Path) -> u64 {
    std::fs::read_dir(p).map(|rd| rd.flatten().map(|e| if e.path().is_dir() { dir_bytes(&e.path()) } else { file_bytes(&e.path()) }).sum()).unwrap_or(0)
}
fn modified_ms(p: &Path) -> u64 {
    std::fs::metadata(p)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// One learned unit, as listed.
struct Unit {
    compartment: String,
    id: String,
    kind: &'static str,
    fields: Vec<(&'static str, Value)>,
    plugged: bool,
    /// Why it would not act even when plugged in and its compartment is on (empty: it acts).
    held: String,
}

/// Projects known to the Nano: key to root.
fn projects(home: &Path) -> BTreeMap<String, String> {
    std::fs::read_dir(nmtype::project::projects_dir(home))
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().is_dir())
                .filter_map(|e| {
                    let key = e.file_name().to_str()?.to_owned();
                    let root = read_json(&e.path().join("project.json")).map(|v| text(&v, "root")).unwrap_or_default();
                    Some((key, root))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The compartment of an adapter: the project its label names, else the project whose key carries
/// the first 8 hex digits of the adapter's mark, else `project:<mark>`.
fn adapter_compartment(home: &Path, id: &str, label: &Value, known: &BTreeMap<String, String>) -> String {
    // A distilled cartridge serves its MicroModel in every project: the Nano's own compartment.
    if nmapi::distil::role_of(id).is_some() {
        return nmapi::plugs::NANO.to_owned();
    }
    let root = label.get("notes").map(|nt| text(nt, "project")).unwrap_or_default();
    if !root.is_empty() {
        let c = nmapi::plugs::compartment_of_root(home, Path::new(&root));
        if c == nmapi::plugs::NANO || known.contains_key(&c) {
            return c;
        }
    }
    let mark = id.rsplit('-').next().unwrap_or("");
    let short = mark.get(..8).unwrap_or(mark);
    known.keys().find(|k| k.rsplit('-').next() == Some(short)).cloned().unwrap_or_else(|| format!("project:{mark}"))
}

fn adapters(home: &Path, known: &BTreeMap<String, String>, plugs: &nmapi::plugs::State, out: &mut Vec<Unit>) {
    let shelf = home.join("library/cartridges/adapter");
    let Ok(rd) = std::fs::read_dir(&shelf) else { return };
    let mut ids: Vec<String> = rd.flatten().filter(|e| e.path().is_dir()).filter_map(|e| e.file_name().to_str().map(str::to_owned)).collect();
    ids.sort();
    for id in ids {
        let versions = nmapi::adapter::versions(home, &id);
        let Some((newest, file)) = versions.last().cloned() else { continue };
        let label = nmlearn::library::label_of(&file).unwrap_or(Value::Null);
        let compartment = adapter_compartment(home, &id, &label, known);
        let (role, switch_on, switch) = if id.starts_with(nmapi::adapter::LENS_PREFIX) {
            ("embedder", nmapi::adapter::lens_on(), "NANO_ADAPTER_LENS")
        } else if id.starts_with(nmapi::adapter::JUDGE_PREFIX) {
            ("reranker", nmapi::adapter::judge_on(), "NANO_ADAPTER_JUDGE")
        } else if let Some(r) = nmapi::distil::role_of(&id).filter(|r| matches!(*r, "writer" | "reader" | "reranker" | "embedder")) {
            // A distilled cartridge: it loads on its MicroModel as the MicroModel opens (`nmapi::distil`).
            (r, nmapi::distil::on(), "NANO_DISTIL")
        } else {
            ("", false, "")
        };
        let notes = label.get("notes").cloned().unwrap_or(Value::Null);
        let held = if role.is_empty() {
            "no serving path loads this kind of adapter".to_owned()
        } else if switch_on {
            String::new()
        } else {
            format!("{switch}=off in this server's environment")
        };
        out.push(Unit {
            plugged: plugs.plugged(&nmapi::plugs::key(&compartment, &id)),
            compartment,
            id: id.clone(),
            kind: "adapter",
            fields: vec![
                ("name", s(&text(&label, "kind"))),
                ("does", s(&text(&label, "does"))),
                ("role", s(role)),
                ("model", s(&text(&label, "model"))),
                ("acts_on", s(&text(&notes, "acts_on"))),
                ("made_from", s(&text(&label, "made_from"))),
                ("measured", s(&text(&notes, "measured"))),
                ("made", s(&text(&label, "made"))),
                ("version", n(newest)),
                ("versions", n(versions.len() as u64)),
                ("bytes", n(versions.iter().map(|(_, f)| file_bytes(f)).sum())),
                ("path", s(&file.display().to_string())),
                ("modified_ms", n(modified_ms(&file))),
                ("loads", s(if nmapi::distil::role_of(&id).is_some() { "on its MicroModel in every project, only on the base it was fitted on, as the MicroModel opens" } else { "only while its own project is served, and only on the base it was fitted on" })),
            ],
            held,
        });
    }
}

fn pieces(home: &Path, known: &BTreeMap<String, String>, plugs: &nmapi::plugs::State, out: &mut Vec<Unit>) {
    let mut stores: Vec<(String, PathBuf)> = vec![(nmapi::plugs::NANO.to_owned(), nmlearn::learned::store(home))];
    stores.extend(known.keys().map(|k| (k.clone(), nmtype::project::projects_dir(home).join(k).join("learned"))));
    for (compartment, store) in stores {
        let ids = nmlearn::learned::list(&store);
        // The judge uses the newest calibrator that is plugged in (`nmapi::ask::judge`).
        let in_use = ids.iter().rfind(|x| x.starts_with("l4-calibrator-") && plugs.acts(&compartment, x)).cloned();
        for id in ids {
            let at = store.join(&id);
            let label = read_json(&at.join("cartridge.json")).unwrap_or(Value::Null);
            let recipe = read_json(&at.join("recipe.json")).unwrap_or(Value::Null);
            let calibrator = id.starts_with("l4-calibrator-");
            let held = if calibrator {
                if plugs.acts(&compartment, &id) && in_use.as_deref() != Some(id.as_str()) {
                    format!("a newer calibrator ({}) is the one in use", in_use.clone().unwrap_or_default())
                } else {
                    String::new()
                }
            } else {
                "saved with its gate report; no serving path loads L1 settings yet (next version)".to_owned()
            };
            out.push(Unit {
                plugged: plugs.plugged(&nmapi::plugs::key(&compartment, &id)),
                compartment: compartment.clone(),
                id: id.clone(),
                kind: "piece",
                fields: vec![
                    ("name", s(if calibrator { "L4 calibrator" } else { "L1 settings" })),
                    ("does", s(&text(&label, "description"))),
                    ("role", s("core brain")),
                    ("model", s(&text(&label, "base"))),
                    ("acts_on", s(if calibrator { "the judge for asking more (whether a pack needs another look)" } else { "the Core Brain's second-look threshold" })),
                    ("made_from", s(&text(&recipe, "signal"))),
                    ("measured", s(&text(&recipe, "method"))),
                    ("made", s(&text(&label, "date"))),
                    ("bytes", n(dir_bytes(&at))),
                    ("path", s(&at.display().to_string())),
                    ("modified_ms", n(modified_ms(&at))),
                    ("loads", s("only for the memory it was fitted in, and only on the base it was fitted on")),
                ],
                held,
            });
        }
    }
}

fn registered(home: &Path, out: &mut Vec<Unit>) {
    let units = nmlearn::Units::open(home);
    for card in &units.cards {
        let dir = units.unit_dir(&card.id);
        let held = match card.status.as_str() {
            "rejected" | "rolled-back" | "retired" => format!("status {}: it does not load", card.status),
            _ if card.files.is_empty() => "unfitted: no weights yet".to_owned(),
            _ => String::new(),
        };
        out.push(Unit {
            compartment: nmapi::plugs::SHARED.to_owned(),
            id: card.id.clone(),
            kind: "unit",
            plugged: card.loaded,
            fields: vec![
                ("name", s("learning unit")),
                ("does", s(&format!("low-rank adapter beside {} (rank {}, scale {})", card.target, card.rank, card.scale))),
                ("role", s(&card.slot)),
                ("model", s(&card.base)),
                ("acts_on", s(&card.target)),
                ("made_from", s("checked outcomes (forward-only fit)")),
                ("measured", s(&format!("status {}", card.status))),
                ("made", s(&nmlearn::cartridge::date_of(card.registered_at))),
                ("bytes", n(dir_bytes(&dir))),
                ("path", s(&dir.display().to_string())),
                ("modified_ms", n(modified_ms(&dir))),
                ("loads", s("when a server starts with serve --unit; plugged through its registry with a hash check")),
            ],
            held,
        });
    }
}

/// `GET /api/learned-units`: every compartment with its units.
#[must_use]
pub fn view(home: &Path) -> Value {
    let known = projects(home);
    let plugs = nmapi::plugs::read_now(home);
    let mut units = Vec::new();
    adapters(home, &known, &plugs, &mut units);
    pieces(home, &known, &plugs, &mut units);
    registered(home, &mut units);
    let mut comps: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    comps.entry(nmapi::plugs::NANO.to_owned()).or_default();
    comps.entry(nmapi::plugs::SHARED.to_owned()).or_default();
    for k in known.keys() {
        comps.entry(k.clone()).or_default();
    }
    let (mut total, mut acting) = (0u64, 0u64);
    for u in units {
        let on = plugs.compartment_on(&u.compartment);
        let acts = on && u.plugged && u.held.is_empty();
        total += 1;
        acting += u64::from(acts);
        let why = if !on {
            "its compartment is switched off".to_owned()
        } else if !u.plugged {
            "unplugged".to_owned()
        } else {
            u.held.clone()
        };
        let switch_key = if u.kind == "unit" { format!("{}{}", crate::controls::CARTRIDGE_KEY, u.id) } else { format!("{LEARNED_KEY}{}", nmapi::plugs::key(&u.compartment, &u.id)) };
        let mut f = vec![
            ("key", s(&nmapi::plugs::key(&u.compartment, &u.id))),
            ("id", s(&u.id)),
            ("kind", s(u.kind)),
            ("compartment", s(&u.compartment)),
            ("plugged", Value::Bool(u.plugged)),
            ("acts", Value::Bool(acts)),
            ("why_not", if acts { Value::Null } else { s(&why) }),
            ("switch_key", s(&switch_key)),
        ];
        f.extend(u.fields);
        comps.entry(u.compartment).or_default().push(obj(f));
    }
    let compartments: Vec<Value> = comps
        .into_iter()
        .map(|(key, list)| {
            let (label, root, what) = if key == nmapi::plugs::NANO {
                ("The Nano's own folder".to_owned(), nmtype::nano_root().display().to_string(), "pieces learned while the Nano studies its own folder")
            } else if key == nmapi::plugs::SHARED {
                ("Shared learning units".to_owned(), home.join("units").display().to_string(), "registered learning units, loaded on request with serve --unit")
            } else if let Some(root) = known.get(&key) {
                (Path::new(root).file_name().and_then(|x| x.to_str()).unwrap_or(&key).to_owned(), root.clone(), "adapters and pieces fitted on this project; they never load for another")
            } else {
                (key.clone(), String::new(), "adapters whose project is no longer known to the Nano")
            };
            let on = plugs.compartment_on(&key);
            obj(vec![
                ("key", s(&key)),
                ("label", s(&label)),
                ("root", s(&root)),
                ("holds", s(what)),
                ("on", Value::Bool(on)),
                ("switch_key", s(&format!("{COMPARTMENT_KEY}{key}"))),
                ("units", Value::Array(list)),
            ])
        })
        .collect();
    obj(vec![
        ("file", s(&nmapi::plugs::path(home).display().to_string())),
        ("total", n(total)),
        ("acting", n(acting)),
        ("applies", s("at the next call of the serving process (it reads the plug file again when it changes); outputs of an unplugged unit are the base's bit for bit")),
        ("compartments", Value::Array(compartments)),
    ])
}

fn on_off(key: &str, value: &str) -> Result<bool, String> {
    match value {
        "on" => Ok(true),
        "off" => Ok(false),
        v => Err(format!("{key}: {v} is not one of on, off")),
    }
}

fn log(home: &Path, key: &str, label: &str, on: bool, undo_of: Option<&str>, by: &str) -> Result<Option<String>, String> {
    let (from, to) = if on { ("off", "on") } else { ("on", "off") };
    let mut fields = vec![
        ("key", s(key)),
        ("label", s(label)),
        ("from", s(from)),
        ("to", s(to)),
        ("by", s(by)),
        ("applies", s("next call of the serving process")),
    ];
    if let Some(u) = undo_of {
        fields.push(("undo_of", s(u)));
    }
    crate::controls::audit(home, "setting", fields).map(Some)
}

/// Plugs a learned adapter or piece in or out (`learned:<compartment>/<id>`). Returns the audit id,
/// or `None` when it was already so.
/// # Errors
/// A value other than on or off, a unit that does not exist, or a write failure.
pub fn switch_unit(home: &Path, unit: &str, value: &str, undo_of: Option<&str>, by: &str) -> Result<Option<String>, String> {
    let key = format!("{LEARNED_KEY}{unit}");
    let on = on_off(&key, value)?;
    let (compartment, id) = unit.split_once('/').ok_or_else(|| format!("{key}: give <compartment>/<id>"))?;
    let exists = view(home)
        .get("compartments")
        .ok()
        .and_then(|c| c.array().ok())
        .is_some_and(|cs| cs.iter().filter(|c| text(c, "key") == compartment).flat_map(|c| c.get("units").ok().and_then(|u| u.array().ok()).unwrap_or_default()).any(|u| text(u, "id") == id && text(u, "kind") != "unit"));
    if !exists {
        return Err(format!("no learned adapter or piece {id} in compartment {compartment}"));
    }
    // A distilled cartridge plugs in only where the compatibility table says it fits.
    if on
        && nmapi::distil::role_of(id).is_some()
        && let Some(why) = crate::library::distil_misfit(home, id)
    {
        return Err(format!("refused: {why}"));
    }
    let (before, after) = nmapi::plugs::change(home, |st| {
        if on {
            st.units_off.remove(unit);
        } else {
            st.units_off.insert(unit.to_owned());
        }
    })?;
    if before == after {
        return Ok(None);
    }
    // A distilled cartridge of a MicroModel already open in this process takes the plug change at
    // once (the Writer takes it again at its next idle session either way).
    if let Some(role) = nmapi::distil::role_of(id) {
        for (slot, m) in nmapi::neural::bound_models() {
            if (slot == "base" && role == "reader") || slot == role {
                let _ = nmapi::distil::seat(home, &m, role);
            }
        }
    }
    log(home, &key, &format!("Learned unit {id} ({compartment})"), on, undo_of, by)
}

/// Switches a compartment on or off (`compartment:<key>`): off takes every unit in it out, each
/// keeping its own plug, so switching it on again brings back exactly what was plugged in.
/// # Errors
/// A value other than on or off, an unknown compartment, or a write failure.
pub fn switch_compartment(home: &Path, compartment: &str, value: &str, undo_of: Option<&str>, by: &str) -> Result<Option<String>, String> {
    let key = format!("{COMPARTMENT_KEY}{compartment}");
    let on = on_off(&key, value)?;
    let known = compartment == nmapi::plugs::NANO
        || compartment == nmapi::plugs::SHARED
        || view(home).get("compartments").ok().and_then(|c| c.array().ok()).is_some_and(|cs| cs.iter().any(|c| text(c, "key") == compartment));
    if !known {
        return Err(format!("no compartment {compartment}"));
    }
    let (before, after) = nmapi::plugs::change(home, |st| {
        if on {
            st.compartments_off.remove(compartment);
        } else {
            st.compartments_off.insert(compartment.to_owned());
        }
    })?;
    if before == after {
        return Ok(None);
    }
    log(home, &key, &format!("Learned units of {compartment}"), on, undo_of, by)
}

// ---------------------------------------------------------------------------------------------
// Registered learning units and project lenses over the API (2026-11-26, Phase B items A5, A6).

/// `GET /api/units`: every registered learning unit (`units/registry.json`, compartment `shared`)
/// with its target node, rank, scale, slot, status, lineage, whether it is loaded, the MicroModel
/// it was made over and whether that model's recorded hash is the one the unit was made for, and
/// whether its files match their recorded hashes. Loading and unloading go through the cartridge
/// switch (`cartridge:<id>`, the same audit and undo as the cartridge rack).
#[must_use]
pub fn units_view(home: &Path) -> Value {
    let units = nmlearn::Units::open(home);
    let plugs = nmapi::plugs::read_now(home);
    let rows: Vec<Value> = units
        .cards
        .iter()
        .map(|card| {
            let dir = units.unit_dir(&card.id);
            let model_sha = nmnpcs::WeightCard::load(home, &card.base).ok().map(|w| w.sha_main);
            let verified = !card.files.is_empty() && card.files.iter().all(|(p, h)| std::fs::read(dir.join(p)).is_ok_and(|b| nmhash::sha256_bytes(&b) == *h));
            obj(vec![
                ("id", s(&card.id)),
                ("base", s(&card.base)),
                ("base_sha256", s(&card.base_sha256)),
                ("model_sha256", model_sha.as_deref().map_or(Value::Null, s)),
                ("base_compatible", model_sha.as_deref().map_or(Value::Null, |m| Value::Bool(m == card.base_sha256))),
                ("slot", s(&card.slot)),
                ("target", s(&card.target)),
                ("rank", n(card.rank as u64)),
                ("scale", Value::Number(format!("{}", card.scale))),
                ("status", s(&card.status)),
                ("lineage", Value::Array(card.lineage.iter().map(|l| s(l)).collect())),
                ("loaded", Value::Bool(card.loaded)),
                ("files", n(card.files.len() as u64)),
                ("files_verified", Value::Bool(verified)),
                ("bytes", n(dir_bytes(&dir))),
                ("path", s(&dir.display().to_string())),
                ("registered_at", n(card.registered_at)),
                ("switch_key", s(&format!("{}{}", crate::controls::CARTRIDGE_KEY, card.id))),
                ("unregister", s(if card.files.is_empty() { "allowed (unfitted: no files)" } else { "refused: a unit with files is retired, never removed" })),
            ])
        })
        .collect();
    obj(vec![
        ("units", Value::Array(rows)),
        ("registry", s(&units.dir.join("registry.json").display().to_string())),
        ("shared_on", Value::Bool(plugs.compartment_on(nmapi::plugs::SHARED))),
        ("register", s("register binds the MicroModel to check the target node's widths, so it runs in a serving process with the engine bound (nm_unit op=register)")),
        ("applies", s("a unit loads when a server starts with serve --unit; plugged through its registry with a hash check")),
    ])
}

/// `POST /api/units {action: unregister, id}` (after its confirmation): takes an unfitted unit (no
/// files) out of the registry. The card is kept in the audit row so the Audit page can register it
/// again (`undo_unregister`).
/// # Errors
/// An unknown unit, one with files, or a write failure.
pub fn unregister(home: &Path, id: &str, by: &str) -> Result<String, String> {
    let mut units = nmlearn::Units::open(home);
    let card = units.get(id).cloned().ok_or_else(|| format!("no unit {id}"))?;
    match nmapi::tools::installed() {
        // In a serving process the unit leaves through `nm_unit`'s own path (its organism log too).
        Some(o) => {
            let v = obj(vec![("op", s("unregister")), ("id", s(id))]);
            nmapi::learn::tool(o, &v).map_err(|e| e.to_string())?;
        }
        None => units.unregister(id).map_err(|e| e.to_string())?,
    }
    crate::controls::audit(home, "unit", vec![("action", s("unregister")), ("unit", s(id)), ("card", card.to_value()), ("by", s(by)), ("label", s(&format!("Learning unit {id} unregistered (its card kept for undo)")))])
}

/// Undoes a unit's unregistering: its card, as the audit row kept it, is registered again.
/// # Errors
/// A row without a card, or the registry refusing it (an id taken again, an open unit on the same
/// target).
pub fn undo_unregister(home: &Path, row: &Value, id: &str, by: &str) -> Result<Option<String>, String> {
    let card = nmlearn::UnitCard::from_value(row.get("card").map_err(|e| format!("card: {e}"))?).map_err(|e| e.to_string())?;
    let unit = card.id.clone();
    nmlearn::Units::open(home).register(card).map_err(|e| e.to_string())?;
    crate::controls::audit(home, "unit", vec![("action", s("register")), ("unit", s(&unit)), ("undo_of", s(id)), ("by", s(by)), ("label", s(&format!("Undone: learning unit {unit} registered again")))]).map(Some)
}

/// Every project lens on the adapter shelf (`library/cartridges/adapter/embedder-lens-<mark>/`),
/// as the learned units list shows them, each with its version files.
fn lens_rows(home: &Path, project: Option<&str>) -> Vec<Value> {
    let v = view(home);
    let mut out = Vec::new();
    for c in v.get("compartments").and_then(Value::array).unwrap_or_default() {
        let comp = text(c, "key");
        if project.is_some_and(|p| p != comp) {
            continue;
        }
        for u in c.get("units").and_then(Value::array).unwrap_or_default() {
            let id = text(u, "id");
            if text(u, "kind") != "adapter" || !id.starts_with(nmapi::adapter::LENS_PREFIX) {
                continue;
            }
            let versions: Vec<Value> = nmapi::adapter::versions(home, &id).iter().map(|(ver, f)| obj(vec![("version", n(*ver)), ("path", s(&f.display().to_string())), ("bytes", n(file_bytes(f)))])).collect();
            let mut row = u.clone();
            if let Value::Object(m) = &mut row {
                m.insert("version_files".into(), Value::Array(versions));
            }
            out.push(row);
        }
    }
    out
}

/// `GET /api/lenses[?project=<key>]`: the project lenses (the embedder's per-project adapter), each
/// with its compartment, plug state, whether it acts and why not, its versions and their files.
/// Plugging one in or out is the learned-unit plug (`learned:<compartment>/<id>`).
#[must_use]
pub fn lenses_view(home: &Path, query: &str) -> Value {
    let project = crate::project::query_arg(query, "project").filter(|p| !p.is_empty());
    obj(vec![
        ("project", project.as_deref().map_or(Value::Null, s)),
        ("lenses", Value::Array(lens_rows(home, project.as_deref()))),
        ("serving_switch", s("NANO_ADAPTER_LENS")),
        ("lens_on", Value::Bool(nmapi::adapter::lens_on())),
        ("applies", s("at the next call of the serving process (it reads the plug file again when it changes)")),
    ])
}

/// A lens by id (and the compartment it is in), from the list.
fn find_lens(home: &Path, id: &str) -> Result<Value, String> {
    // A distilled cartridge is removed through the same route (its versions on the adapter shelf).
    if nmapi::distil::role_of(id).is_some() && !id.contains(['/', '\\', '.']) {
        return crate::distil::cartridges(home).array().unwrap_or_default().iter().find(|r| text(r, "id") == id).cloned().ok_or_else(|| format!("no distilled cartridge {id} on the adapter shelf"));
    }
    if !id.starts_with(nmapi::adapter::LENS_PREFIX) || id.contains(['/', '\\', '.']) {
        return Err(format!("{id}: not a lens id (embedder-lens-<mark>)"));
    }
    lens_rows(home, None).into_iter().find(|r| text(r, "id") == id).ok_or_else(|| format!("no lens {id} among the adapters in the cartridge library"))
}

/// The plug key of a lens (`<compartment>/<id>`), given the id or the key itself.
/// # Errors
/// No such lens.
pub fn lens_key(home: &Path, unit: &str) -> Result<String, String> {
    let id = unit.rsplit('/').next().unwrap_or(unit);
    find_lens(home, id).map(|r| text(&r, "key"))
}

/// The files a lens removal deletes (`version` one, else every version) with their bytes, after the
/// refusals: a lens that a serving process names in its live file is in use.
/// # Errors
/// No such lens or version, or the lens in use.
pub fn lens_remove_plan(home: &Path, id: &str, version: Option<u64>) -> Result<Vec<(PathBuf, u64)>, String> {
    find_lens(home, id)?;
    if crate::live::read_all(home).iter().any(|p| p.encode().is_ok_and(|t| t.contains(id))) {
        return Err(format!("{id} is named by a running Nano; unplug it (or stop the server) first, then remove it"));
    }
    let files: Vec<(PathBuf, u64)> = nmapi::adapter::versions(home, id).into_iter().filter(|(v, _)| version.is_none_or(|w| w == *v)).map(|(_, f)| {
        let b = file_bytes(&f);
        (f, b)
    }).collect();
    if files.is_empty() {
        return Err(format!("{id}: no version {} in the cartridge library", version.map_or_else(|| "files".to_owned(), |v| v.to_string())));
    }
    Ok(files)
}

/// `POST /api/lenses {action: remove, id[, version]}` (after its confirmation): deletes the lens's
/// version files (one version, or all; the empty folder goes too). Audited; it cannot be undone
/// (the idle fit makes a lens again from the project's material).
/// # Errors
/// As `lens_remove_plan`, or a removal failing.
pub fn lens_remove(home: &Path, id: &str, version: Option<u64>, by: &str) -> Result<Value, String> {
    let files = lens_remove_plan(home, id, version)?;
    let mut bytes = 0;
    for (f, b) in &files {
        std::fs::remove_file(f).map_err(|e| format!("{}: {e}", f.display()))?;
        bytes += b;
    }
    let dir = nmapi::adapter::shelf_file(home, id, 1).parent().map(Path::to_path_buf);
    if let Some(d) = dir.filter(|d| std::fs::read_dir(d).is_ok_and(|mut rd| rd.next().is_none())) {
        let _ = std::fs::remove_dir(d);
    }
    let shown: Vec<Value> = files.iter().map(|(f, _)| s(&f.display().to_string())).collect();
    let audit = crate::controls::audit(
        home,
        "lens-remove",
        vec![("lens", s(id)), ("files", Value::Array(shown.clone())), ("bytes", n(bytes)), ("by", s(by)), ("label", s(&format!("Lens {id} removed: {} files", files.len())))],
    )?;
    Ok(obj(vec![("lens", s(id)), ("files", Value::Array(shown)), ("bytes", n(bytes)), ("audit_id", s(&audit)), ("lenses", lenses_view(home, ""))]))
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

    fn piece(store: &Path, id: &str) {
        let at = store.join(id);
        std::fs::create_dir_all(&at).unwrap();
        std::fs::write(at.join("cartridge.json"), r#"{"description":"L4 calibrator","base":"qwen3-reranker-0.6b","date":"2026-11-26"}"#).unwrap();
        std::fs::write(at.join("recipe.json"), r#"{"signal":"verdicts","method":"logistic"}"#).unwrap();
    }

    #[test]
    fn units_list_by_compartment_and_plug_with_an_audit_row() {
        let home = Home(std::env::temp_dir().join(format!("nm-learned-units-{}", std::process::id())));
        let proj = nmtype::project::projects_dir(&home.0).join("shop-1234abcd");
        std::fs::create_dir_all(&proj).unwrap();
        std::fs::write(proj.join("project.json"), r#"{"key":"shop-1234abcd","root":"/w/shop"}"#).unwrap();
        piece(&proj.join("learned"), "l4-calibrator-001");
        piece(&proj.join("learned"), "l4-calibrator-002");
        let v = view(&home.0);
        let comp = v.get("compartments").unwrap().array().unwrap().iter().find(|c| text(c, "key") == "shop-1234abcd").cloned().unwrap();
        let units = comp.get("units").unwrap().array().unwrap();
        assert_eq!(units.len(), 2);
        assert_eq!(units.iter().filter(|u| u.get("acts").is_ok_and(|a| *a == Value::Bool(true))).count(), 1, "only the newest calibrator is in use");
        let audit = switch_unit(&home.0, "shop-1234abcd/l4-calibrator-002", "off", None, "test").unwrap();
        assert!(audit.is_some());
        assert!(switch_unit(&home.0, "shop-1234abcd/l4-calibrator-002", "off", None, "test").unwrap().is_none(), "already off");
        assert!(!nmapi::plugs::acts(&home.0, "shop-1234abcd", "l4-calibrator-002"));
        let v = view(&home.0);
        let acts: Vec<String> = v.get("compartments").unwrap().array().unwrap().iter().flat_map(|c| c.get("units").unwrap().array().unwrap().iter()).filter(|u| u.get("acts").is_ok_and(|a| *a == Value::Bool(true))).map(|u| text(u, "id")).collect();
        assert_eq!(acts, vec!["l4-calibrator-001".to_owned()], "unplugging the newest brings the one before it in");
        assert!(switch_compartment(&home.0, "shop-1234abcd", "off", None, "test").unwrap().is_some());
        assert!(!nmapi::plugs::acts(&home.0, "shop-1234abcd", "l4-calibrator-001"));
        assert!(switch_unit(&home.0, "shop-1234abcd/nope", "off", None, "test").is_err());
        assert!(switch_compartment(&home.0, "no-such-00000000", "off", None, "test").is_err());
        let log = std::fs::read_to_string(home.0.join("memory/audit.jsonl")).unwrap();
        assert_eq!(log.lines().count(), 2);
        assert!(log.contains("learned:shop-1234abcd/l4-calibrator-002") && log.contains("compartment:shop-1234abcd"));
    }
}
