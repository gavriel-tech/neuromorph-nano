//! The MCP tools that match the web interface's inspections and controls one for one.
//!
//! Roadmap Phase 4.5 (2026-11-24), item 4 (MCP and web UI parity). Every shipped inspection and
//! control of the interface is reachable from MCP too, through the same functions:
//!
//! - storage (`nm_storage`) calls `crate::storage`, as the interface's `/api/storage` routes do;
//! - the agent's guidance (reminders, loops and the step graph, `nm_goal` with an `op`) calls the
//!   interface's `/api/guidance` (`guidance`), so a reminder or loop the agent sets for itself is the
//!   one the Memory page shows and the hooks deliver;
//! - cartridges and job filters (`nm_cartridges`), the learning cycle and its rollback
//!   (`nm_learning`), comprehension and the 4-level world model (`nm_comprehension`) and the
//!   controls, status and audit (`nm_settings`) call the interface's own route functions, which
//!   the program lends this library at start (`lend_routes`, from `nmcore`'s `ui.rs`), so a tool
//!   and a page run the same code and write the same audit rows.
//!
//! Only the listed routes are reachable (no pairing, device or token route: those stay with the
//! person at the machine), and every reply is a behavioural report (counts, paths, settings,
//! verdicts, the writer's own notes), never raw material of the studied folder (`nmtype::class`).
//! Destructive actions need `confirm: true`.
use crate::{ApiError, Field, FieldKind};
use nmjson::Value;
use std::path::Path;
use std::sync::OnceLock;

/// The interface's route function lent by the program: (home, path, query, body for a write,
/// who asks) to the route's reply. A body of `None` reads (`GET`), `Some` writes (`POST`).
pub type Routes = fn(&Path, &str, &str, Option<&[u8]>, &str) -> Result<Value, String>;
static ROUTES: OnceLock<Routes> = OnceLock::new();

/// Lends this library the interface's routes (first call wins).
pub fn lend_routes(f: Routes) {
    let _ = ROUTES.set(f);
}

/// Who the audit rows name for a tool call.
const BY: &str = "mcp";

fn bad(m: impl Into<String>) -> ApiError {
    ApiError::new("nm-400", m)
}
fn failed(m: String) -> ApiError {
    ApiError::new("nm-500", m)
}
fn arg<'a>(v: &'a Value, k: &str) -> Option<&'a str> {
    v.get(k).and_then(Value::string).ok()
}
fn need<'a>(v: &'a Value, k: &str, op: &str) -> Result<&'a str, ApiError> {
    arg(v, k).filter(|x| !x.is_empty()).ok_or_else(|| bad(format!("{op} needs {k}")))
}
fn confirmed(v: &Value, op: &str) -> Result<(), ApiError> {
    if matches!(v.get("confirm"), Ok(Value::Bool(true))) { Ok(()) } else { Err(bad(format!("{op} changes or removes data; send confirm: true"))) }
}
fn home() -> std::path::PathBuf {
    nmtype::nano_root()
}
fn s(t: &str) -> Value {
    Value::String(t.to_owned())
}
/// A delete's body (2026-11-26, Luis Jake Gabriel III: every delete verified in more than one
/// step): the first call, without `confirm_token`, gets a preview and a token from the route and
/// deletes nothing; the same call with that `confirm_token` deletes.
fn gated(v: &Value, mut pairs: Vec<(&str, Value)>) -> Result<Vec<u8>, ApiError> {
    if let Some(tok) = arg(v, "confirm_token").filter(|x| !x.is_empty()) {
        pairs.push(("confirm_token", s(tok)));
    }
    body(pairs)
}
fn body(pairs: Vec<(&str, Value)>) -> Result<Vec<u8>, ApiError> {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect()).encode().map(String::into_bytes).map_err(|e| failed(e.to_string()))
}
/// A query of the named fields that are given, each kept to safe characters.
fn query(v: &Value, keys: &[&str]) -> String {
    let mut q = Vec::new();
    for k in keys {
        if let Some(x) = arg(v, k).filter(|x| !x.is_empty() && x.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))) {
            q.push(format!("{k}={x}"));
        }
    }
    if let Ok(l) = v.get("limit").and_then(Value::integer) {
        q.push(format!("limit={l}"));
    }
    q.join("&")
}
/// A project key as `/api/projects` lists it (`nano` for the Nano's own folder), safe in a query.
fn project(v: &Value) -> Result<String, ApiError> {
    let key = arg(v, "project").unwrap_or("nano");
    let ok = !key.is_empty() && !key.starts_with('.') && key.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if ok { Ok(key.to_owned()) } else { Err(bad("project: give a key as nm_settings op projects lists it")) }
}

/// A query value percent-encoded (paths and keys hold `/`, `@`, `:`, backslashes and spaces).
fn enc(x: &str) -> String {
    x.bytes().map(|b| if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') { char::from(b).to_string() } else { format!("%{b:02X}") }).collect()
}
/// A query of the named fields that are given, each percent-encoded, and the integers named.
fn query_enc(v: &Value, keys: &[&str], ints: &[&str]) -> String {
    let mut q: Vec<String> = keys.iter().filter_map(|k| arg(v, k).filter(|x| !x.is_empty()).map(|x| format!("{k}={}", enc(x)))).collect();
    q.extend(ints.iter().filter_map(|k| v.get(k).and_then(Value::integer).ok().map(|x| format!("{k}={x}"))));
    q.join("&")
}
/// The given optional fields, copied as they are.
fn copied<'a>(v: &Value, keys: &[&'a str]) -> Vec<(&'a str, Value)> {
    keys.iter().filter_map(|k| v.get(k).ok().map(|x| (*k, x.clone()))).collect()
}

/// Reads or writes one of the interface's routes through the lent function.
fn route(path: &str, query: &str, write: Option<Vec<u8>>) -> Result<Value, ApiError> {
    let f = ROUTES.get().ok_or_else(|| ApiError::new("nm-503", "the interface's routes are not lent to this process (start it with `neuromorph-nano serve`)"))?;
    let v = f(&home(), path, query, write.as_deref(), BY).map_err(failed)?;
    // 2026-11-25: a view that answers `{"error": "..."}` (a bad project key) is a failure here too,
    // never a success with an error inside.
    if let Ok(e) = v.get("error").and_then(Value::string) {
        return Err(ApiError::new("nm-400", e));
    }
    Ok(v)
}

/// The `nm_goal` ops that reach the agent's guidance.
pub const GUIDANCE_OPS: &[&str] = &["guidance", "reminder_add", "reminder_list", "reminder_edit", "loop_add", "loop_edit", "loop_end", "loop_outcomes", "step_add", "step_done", "step_edit", "remove", "enable", "disable", "reset"];

/// `nm_goal` with a guidance `op`: the reminders, loops and step graph of a project (`served` by
/// default), through `/api/guidance`. The route names the caller `mcp`, so what the agent sets is
/// kept as set by the agent. A remove takes two calls (a preview with a `confirm_token`, then the
/// same call with it).
/// # Errors
/// A missing field, a bad project key, or the route refusing the change.
pub fn guidance(v: &Value, served: &str) -> Result<Value, ApiError> {
    const REMINDER: &[&str] = &["text", "trigger", "every", "glob", "repeat", "max", "files", "error", "count", "enabled"];
    const LOOP: &[&str] = &["prompt", "until", "parts", "all_of", "any_of", "goal", "max", "stall", "budget_ms", "budget_tools", "enabled"];
    const STEP: &[&str] = &["title", "depends_on", "depends_any", "goal", "check", "paths", "probe"];
    let op = arg(v, "op").unwrap_or("guidance");
    let key = if arg(v, "project").is_some() { project(v)? } else { served.to_owned() };
    let change = |action: &str, kind: &str, keys: &[&str], id: Option<&str>| -> Result<Value, ApiError> {
        let mut b = vec![("project", s(&key)), ("action", s(action)), ("kind", s(kind))];
        if let Some(i) = id {
            b.push(("id", s(i)));
        }
        b.extend(copied(v, keys));
        route("/api/guidance", "", Some(body(b)?))
    };
    let kind = || need(v, "kind", op);
    let item = || need(v, "item", op);
    match op {
        "guidance" => route("/api/guidance", &format!("project={key}"), None),
        "reminder_list" => {
            let all = route("/api/guidance", &format!("project={key}"), None)?;
            Ok(Value::Object(
                [("project".to_owned(), s(&key)), ("reminders".to_owned(), all.get("reminders").cloned().unwrap_or(Value::Array(Vec::new()))), ("settings".to_owned(), all.get("settings").cloned().unwrap_or(Value::Null))]
                    .into_iter()
                    .collect(),
            ))
        }
        "reminder_add" => change("add", "reminder", REMINDER, None),
        "reminder_edit" => change("edit", "reminder", REMINDER, Some(item()?)),
        "loop_add" => change("add", "loop", LOOP, None),
        "loop_edit" => change("edit", "loop", LOOP, Some(item()?)),
        "loop_end" => change("done", "loop", &[], Some(item()?)),
        // 2026-11-26: the outcome rows of ended loops, newest first (`limit`, default 20, at most 200).
        "loop_outcomes" => {
            let limit = v.get("limit").ok().and_then(|x| x.integer().ok()).unwrap_or(20).clamp(1, 200);
            let all = route("/api/guidance", &format!("project={key}&outcomes={limit}"), None)?;
            Ok(Value::Object(
                [("project".to_owned(), s(&key)), ("outcomes".to_owned(), all.get("outcomes").cloned().unwrap_or(Value::Array(Vec::new()))), ("outcomes_total".to_owned(), all.get("outcomes_total").cloned().unwrap_or(Value::Null))]
                    .into_iter()
                    .collect(),
            ))
        }
        "step_add" => change("add", "step", STEP, arg(v, "item")),
        "step_edit" => change("edit", "step", STEP, Some(item()?)),
        "step_done" => change("done", "step", &[], Some(item()?)),
        "remove" => route("/api/guidance", "", Some(gated(v, vec![("project", s(&key)), ("action", s("remove")), ("kind", s(kind()?)), ("id", s(item()?))])?)),
        "enable" | "disable" | "reset" => change(op, kind()?, &[], Some(item()?)),
        other => Err(bad(format!("op {other}: use {}", GUIDANCE_OPS.join(", ")))),
    }
}

/// `nm_storage`: the hard-drive storage view and its controls (`crate::storage`).
/// # Errors
/// A missing field, no `confirm` on a prune, delete or index clear, or the action failing.
pub fn storage(v: &Value) -> Result<Value, ApiError> {
    let op = arg(v, "op").unwrap_or("view");
    let h = home();
    match op {
        "view" => Ok(crate::storage::view(&h)),
        "settings" => Ok(crate::storage::settings(&h)),
        "prune" => route("/api/storage/prune", "", Some(gated(v, vec![("category", s(need(v, "category", op)?))])?)),
        "delete" => route("/api/storage/delete", "", Some(gated(v, vec![("path", s(need(v, "path", op)?))])?)),
        // 2026-11-26: forget a project, the memories of a project, delete one memory, trim the audit log.
        "forget_project" => route("/api/projects/forget", "", Some(gated(v, vec![("project", s(&project(v)?))])?)),
        "memories" => route("/api/memory/items", &format!("project={}&{}", project(v)?, query(v, &["limit", "q"])), None),
        "forget_memory" => route("/api/memory/forget", "", Some(gated(v, vec![("project", s(&project(v)?)), ("id", s(need(v, "id", op)?))])?)),
        "audit_retention" => route("/api/audit/retention", "", Some(gated(v, vec![])?)),
        "set" => route("/api/storage/settings", "", Some(body(vec![("key", s(need(v, "key", op)?)), ("value", s(need(v, "value", op)?))])?)),
        "dense_clear" => route("/api/dense/clear", "", Some(gated(v, vec![("project", s(&project(v)?))])?)),
        other => Err(bad(format!("op {other}: use view, settings, prune, delete, set, dense_clear, forget_project, memories, forget_memory or audit_retention"))),
    }
}

/// `nm_cartridges`: the cartridge list and library, switching a cartridge or a use, and job
/// filters (list, on, off, build through the held-out split gate, remove). `filters` borrows
/// `/api/surgery`, so each job filter row carries the same `built_on`, `serving_base` and
/// `base_match` the interface shows (2026-11-26: the base cartridge version it was built on
/// against the one serving; reported, never switched off).
/// # Errors
/// A missing field, no `confirm` on a removal, or the route failing.
pub fn cartridges(v: &Value) -> Result<Value, ApiError> {
    let op = arg(v, "op").unwrap_or("list");
    let filter = |action: &str| -> Result<Value, ApiError> {
        route("/api/surgery/job-filter", "", Some(body(vec![("role", s(need(v, "role", op)?)), ("job", s(need(v, "job", op)?)), ("action", s(action))])?))
    };
    match op {
        "list" => route("/api/cartridges", "", None),
        "library" => route("/api/cartridges/library", "", None),
        "filters" => route("/api/surgery", "", None),
        "on" | "off" => route("/api/cartridges", "", Some(body(vec![("id", s(need(v, "id", op)?)), ("on", Value::Bool(op == "on"))])?)),
        "use" => {
            let stack = arg(v, "stack").filter(|x| !x.is_empty()).map_or(Value::Null, s);
            route("/api/cartridges", "", Some(body(vec![("library", s(need(v, "use", op)?)), ("stack", stack)])?))
        }
        // 2026-11-25: base cartridge versions (the Cartridges page's base panel).
        "bases" => route("/api/cartridges/bases", "", None),
        "compat" => route("/api/cartridges/compat", "", None),
        // 2026-11-25: the whole parameter mapping (the Parameters page).
        "maps" => route("/api/parameter-maps", "", None),
        // 2026-11-26 (Luis Jake Gabriel III: "a way to clean up old maps"): several maps at once, or
        // archived ones; archiving and restoring are reversible (Audit undoes them).
        "map_remove" => {
            let mut b = vec![("role", s(need(v, "role", op)?)), ("confirm", Value::Bool(true))];
            if v.get("files").is_ok() || matches!(v.get("archived"), Ok(Value::Bool(true))) {
                b.extend(copied(v, &["files", "file", "archived"]));
            } else {
                b.push(("file", s(need(v, "file", op)?)));
            }
            route("/api/parameter-maps/remove", "", Some(gated(v, b)?))
        }
        "map_archive" | "map_restore" => {
            let mut b = vec![("role", s(need(v, "role", op)?))];
            b.extend(copied(v, &["files", "file"]));
            route(if op == "map_archive" { "/api/parameter-maps/archive" } else { "/api/parameter-maps/restore" }, "", Some(body(b)?))
        }
        "base_serve" => route("/api/cartridges/bases", "", Some(body(vec![("model", s(need(v, "model", op)?)), ("choice", s(need(v, "choice", op)?))])?)),
        "base_remove" => route("/api/cartridges/bases/remove", "", Some(gated(v, vec![("key", s(need(v, "key", op)?))])?)),
        "filter_on" => filter("enable"),
        "filter_off" => filter("disable"),
        "filter_remove" => route(
            "/api/surgery/job-filter",
            "",
            Some(gated(v, vec![("role", s(need(v, "role", op)?)), ("job", s(need(v, "job", op)?)), ("action", s("delete"))])?),
        ),
        "filter_build" => route(
            "/api/surgery/job-filter/build",
            "",
            Some(body(vec![("role", s(need(v, "role", op)?)), ("job", s(need(v, "job", op)?)), ("layers", s(arg(v, "layers").unwrap_or("all")))])?),
        ),
        // 2026-11-26 (Luis Jake Gabriel III: "This should all be part of the neuromorph nano and
        // integrated fully into the mcp the api and the interface"): the lab's other builds, the
        // same route as the interface's Build tab. A new base cartridge version is kept beside the
        // others when library.upgrade is keep (the default).
        "map_build" => route(
            "/api/lab/build",
            "",
            Some(body(vec![("kind", s("map")), ("role", s(need(v, "role", op)?)), ("share", s(arg(v, "share").unwrap_or("0.40"))), ("device", s(arg(v, "device").unwrap_or("gpu")))])?),
        ),
        "base_build" => route("/api/lab/build", "", Some(body(vec![("kind", s("base")), ("role", s(need(v, "role", op)?))])?)),
        // Distillation: curated teacher pairs trained into a learned adapter cartridge for the chosen
        // MicroModel by the chosen technique, through the held-out gate (the same route as the
        // interface). The cartridges it made are listed by `distils`, inspected and edited by
        // `inspect` and `set_values`, plugged by `lens_switch` and removed by `lens_remove`.
        "distil_build" => {
            let mut b = vec![("kind", s("distil")), ("role", s(arg(v, "role").filter(|x| !x.is_empty()).unwrap_or("writer"))), ("set", s(arg(v, "set").filter(|x| !x.is_empty()).unwrap_or("all"))), ("technique", s(arg(v, "technique").filter(|x| !x.is_empty()).unwrap_or("lens")))];
            b.extend(copied(v, &["rounds", "target", "budget_s"]));
            route("/api/lab/build", "", Some(body(b)?))
        }
        // A held-out check of an existing job filter (an edited one, say); nothing written.
        "filter_check" => route("/api/lab/build", "", Some(body(vec![("kind", s("check")), ("role", s(need(v, "role", op)?)), ("job", s(need(v, "job", op)?))])?)),
        "distils" => route("/api/training/distil", "limit=0", None).map(|r| r.get("cartridges").cloned().unwrap_or(Value::Array(Vec::new()))),
        // 2026-11-26 (Phase B): a map's contents, editing and importing maps, a cartridge's values,
        // trial, stacks and imports, and the project lenses.
        "map_read" => route("/api/parameter-maps/read", &query_enc(v, &["role", "file", "section", "layer"], &["offset", "limit"]), None),
        "map_save" => {
            let text = need(v, "content", op)?;
            let content = nmjson::parse(text.as_bytes()).map_err(|e| bad(format!("content: the new value as JSON text ({e})")))?;
            let mut b = vec![("role", s(need(v, "role", op)?)), ("file", s(need(v, "file", op)?)), ("content", content)];
            b.extend(copied(v, &["section", "layer"]));
            route("/api/parameter-maps/save", "", Some(body(b)?))
        }
        // 2026-11-26: state where a map came from and what it is for.
        "map_lineage" => {
            let mut b = vec![("role", s(need(v, "role", op)?)), ("file", s(need(v, "file", op)?))];
            b.extend(copied(v, &["made_on", "made_by", "purpose", "note"]));
            route("/api/parameter-maps/lineage", "", Some(body(b)?))
        }
        "map_import" => {
            let mut b = vec![("role", s(need(v, "role", op)?)), ("path", s(need(v, "path", op)?))];
            b.extend(copied(v, &["name", "replace"]));
            route("/api/parameter-maps/import", "", Some(body(b)?))
        }
        "inspect" => route("/api/cartridges/inspect", &query_enc(v, &["key"], &[]), None),
        "set_values" => {
            let values = v.get("values").ok().filter(|x| matches!(x, Value::Object(_))).cloned().ok_or_else(|| bad("set_values needs values (an object of name to value)"))?;
            route("/api/cartridges/values", "", Some(body(vec![("key", s(need(v, "key", op)?)), ("values", values)])?))
        }
        "trial" => {
            let trial = match v.get("trial") {
                Ok(Value::Bool(b)) => *b,
                _ => return Err(bad("trial needs trial (true or false)")),
            };
            route("/api/cartridges/trial", "", Some(body(vec![("name", s(need(v, "stack", op)?)), ("trial", Value::Bool(trial))])?))
        }
        "stack_save" => {
            let carts = v.get("cartridges").ok().filter(|x| matches!(x, Value::Array(_))).cloned().ok_or_else(|| bad("stack_save needs cartridges (keys in order)"))?;
            let mut b = vec![("id", s(need(v, "stack", op)?)), ("use", s(need(v, "use", op)?)), ("cartridges", carts)];
            b.extend(copied(v, &["description", "switch"]));
            route("/api/cartridges/stack", "", Some(body(b)?))
        }
        "cartridge_import" => route("/api/cartridges/import", "", Some(body(vec![("path", s(need(v, "path", op)?))])?)),
        // Cartridge parameters (the Parameters page's cartridge lane): every object holding
        // parameters, one object's parameters section, a page of a tensor's values; and an edit
        // into a new version (strength, layers, values), the same route as the interface's Apply.
        "parameters" => {
            let mut q = query_enc(v, &["key", "tensor", "path"], &["offset", "limit"]);
            if matches!(v.get("maps"), Ok(Value::Bool(true))) {
                q.push_str("&maps=1");
            }
            route("/api/cartridges/parameters", &q, None)
        }
        "parameters_edit" => {
            let mut b = vec![("key", s(need(v, "key", op)?))];
            b.extend(copied(v, &["scale", "layers", "values", "edit", "selection", "path", "value", "lo", "hi", "source", "source_b", "t", "target", "width", "rank", "file", "ops", "name", "description", "preview"]));
            // The operation's own name travels as `edit` here (`op` names the tool's operation).
            let b: Vec<(&str, Value)> = b.into_iter().map(|(k, x)| if k == "edit" { ("op", x) } else { (k, x) }).collect();
            route("/api/cartridges/parameters", "", Some(body(b)?))
        }
        "parameters_export" => route("/api/cartridges/parameters/export", "", Some(body(vec![("key", s(need(v, "key", op)?)), ("path", s(need(v, "path", op)?))])?)),
        // The parameter overlay of any object (open weights included): held beside it, never in its
        // files, applied when the engine seats it; the same route as the Parameters page's overlay
        // controls. Clearing goes through the delete gate (two calls, confirm_token).
        "overlay" => route("/api/parameters/overlay", &query_enc(v, &["key"], &[]), None),
        "overlay_edit" | "mask" | "unmask" | "freeze" | "unfreeze" | "overlay_undo" | "overlay_clear" => {
            let action = match op {
                "overlay_edit" => "edit",
                "overlay_undo" => "undo",
                "overlay_clear" => "clear",
                x => x,
            };
            let mut b = vec![("key", s(need(v, "key", op)?)), ("action", s(action))];
            b.extend(copied(v, &["edit", "selection", "path", "value", "lo", "hi", "source", "source_b", "t", "file", "ops", "scale", "layers", "values", "id", "ids", "whole", "preview"]));
            let b: Vec<(&str, Value)> = b.into_iter().map(|(k, x)| if k == "edit" { ("op", x) } else { (k, x) }).collect();
            route("/api/parameters/overlay", "", Some(if action == "clear" { gated(v, b)? } else { body(b)? }))
        }
        // The whole-set overview of an opened object: its tensors in layout order (key alone), or
        // one tensor's region as a w x h grid of cell min, max, mean and RMS, streamed from the file.
        "parameters_overview" => {
            need(v, "key", op)?;
            route("/api/parameters/overview", &query_enc(v, &["key", "tensor"], &["w", "h", "r0", "r1", "c0", "c1"]), None)
        }
        "lenses" => route("/api/lenses", &query_enc(v, &["project"], &[]), None),
        "lens_switch" => {
            let on = match v.get("on") {
                Ok(Value::Bool(b)) => *b,
                _ => return Err(bad("lens_switch needs on (true or false)")),
            };
            route("/api/lenses", "", Some(body(vec![("action", s(if on { "enable" } else { "disable" })), ("id", s(need(v, "id", op)?))])?))
        }
        "lens_remove" => {
            let mut b = vec![("action", s("remove")), ("id", s(need(v, "id", op)?))];
            b.extend(copied(v, &["version"]));
            route("/api/lenses", "", Some(gated(v, b)?))
        }
        other => Err(bad(format!("op {other}: use list, library, filters, on, off, use, bases, compat, maps, map_read, map_save, map_import, map_lineage, map_archive, map_restore, map_remove, map_build, base_serve, base_remove, base_build, distil_build, distils, filter_check, filter_on, filter_off, filter_build, filter_remove, inspect, set_values, parameters, parameters_edit, parameters_export, parameters_overview, overlay, overlay_edit, mask, unmask, freeze, unfreeze, overlay_undo, overlay_clear, trial, stack_save, cartridge_import, lenses, lens_switch or lens_remove"))),
    }
}

/// `nm_learning`: the learning engine's state for a project, one unified learning cycle, a
/// rollback, the training store, the evidence ledger and the outcomes, and clearing the store or
/// the ledger.
/// # Errors
/// A bad project, no `confirm` on a rollback or a clear, or the route failing.
pub fn learning(v: &Value) -> Result<Value, ApiError> {
    let op = arg(v, "op").unwrap_or("view");
    let key = project(v)?;
    let q = format!("project={key}");
    match op {
        "view" => route("/api/learning", &q, None),
        "examples" => route("/api/examples", &q, None),
        "ledger" => route("/api/ledger", &q, None),
        "outcomes" => route("/api/outcomes", &q, None),
        "cycle" | "sync" => route("/api/learning/cycle", "", Some(body(vec![("project", s(&key)), ("action", s(op))])?)),
        // 2026-11-25: the training store to and from a file, and the user's own reference material.
        "export" => {
            let mut b = vec![("project", s(&key))];
            if let Some(nm) = arg(v, "name") {
                b.push(("name", s(nm)));
            }
            route("/api/training/export", "", Some(body(b)?))
        }
        "import" => route("/api/training/import", "", Some(body(vec![("project", s(&key)), ("file", s(need(v, "file", op)?))])?)),
        "references" => route("/api/training/references", "", None),
        // Distillation's teacher data: listed by set, imported from a local JSON lines file, labelled.
        "distil" => route("/api/training/distil", &query(v, &["set", "limit"]), None),
        // A live teacher: the agent opens a teaching set, takes its open prompts and sends its
        // answers, which join the teacher data; or a local open-weights model answers them (a lab
        // run through the interface or the API, kind teach).
        "teach_list" => route("/api/training/distil/teach", "", None),
        "teach_start" => {
            let mut b = vec![("action", s("start")), ("set", s(need(v, "set", op)?))];
            if let Ok(p) = v.get("prompts") {
                b.push(("prompts", p.clone()));
            }
            b.extend(copied(v, &["instruction", "licence"]));
            route("/api/training/distil/teach", "", Some(body(b)?))
        }
        "teach_next" => {
            let mut b = vec![("action", s("next")), ("set", s(need(v, "set", op)?))];
            if let Ok(c) = v.get("count") {
                b.push(("count", c.clone()));
            }
            route("/api/training/distil/teach", "", Some(body(b)?))
        }
        "teach_local" => {
            let mut b = vec![("kind", s("teach")), ("role", s("writer")), ("set", s(need(v, "set", op)?))];
            b.extend(copied(v, &["teacher"]));
            route("/api/lab/build", "", Some(body(b)?))
        }
        "teach_answer" => {
            let mut b = vec![("action", s("answer")), ("set", s(need(v, "set", op)?)), ("answers", v.get("answers").cloned().map_err(|_| bad("teach_answer needs answers: [{prompt_id or prompt, completion}]"))?)];
            b.extend(copied(v, &["teacher", "licence"]));
            route("/api/training/distil/teach", "", Some(body(b)?))
        }
        "distil_import" => {
            let mut b = vec![("file", s(need(v, "file", op)?))];
            for k in ["set", "licence", "teacher"] {
                if let Some(x) = arg(v, k).filter(|x| !x.is_empty()) {
                    b.push((k, s(x)));
                }
            }
            route("/api/training/distil/import", "", Some(body(b)?))
        }
        "distil_label" => {
            let mut b = vec![("outcome", s(need(v, "outcome", op)?))];
            b.extend(copied(v, &["ids", "set", "why"]));
            route("/api/training/distil/label", "", Some(body(b)?))
        }
        "reference_add" => {
            let mut b = vec![("path", s(need(v, "path", op)?)), ("licence", s(need(v, "licence", op)?))];
            for k in ["source", "domain"] {
                if let Some(x) = arg(v, k) {
                    b.push((k, s(x)));
                }
            }
            route("/api/training/references", "", Some(body(b)?))
        }
        "rollback" => {
            confirmed(v, op)?;
            let mut b = vec![("project", s(&key)), ("action", s("rollback"))];
            if let Ok(ver) = v.get("version").and_then(Value::integer) {
                b.push(("version", Value::Number(ver.to_string())));
            }
            if let Some(c) = arg(v, "cycle").filter(|c| !c.is_empty()) {
                b.push(("cycle", s(c)));
            }
            route("/api/learning/cycle", "", Some(body(b)?))
        }
        "clear_examples" | "clear_ledger" => {
            let path = if op == "clear_examples" { "/api/examples/clear" } else { "/api/ledger/clear" };
            route(path, "", Some(gated(v, vec![("project", s(&key))])?))
        }
        // 2026-11-26 (Luis Jake Gabriel III: "plug in and unplug the learned units"): every learned
        // unit by compartment, one plugged in or out, a whole compartment switched on or off.
        "units" => route("/api/learned-units", "", None),
        "unit_on" | "unit_off" => {
            route("/api/learned-units", "", Some(body(vec![("unit", s(need(v, "unit", op)?)), ("on", Value::Bool(op == "unit_on"))])?))
        }
        "compartment_on" | "compartment_off" => {
            let c = arg(v, "compartment").map_or_else(|| key.clone(), str::to_owned);
            route("/api/learned-units", "", Some(body(vec![("compartment", s(&c)), ("on", Value::Bool(op == "compartment_on"))])?))
        }
        // Stated preference cartridges: the user's words, passed by their agent, turned into rules
        // (a preview first, then confirm: true writes a library version), edited, removed (two
        // calls), plugged in or out, and a draft checked against the rules in force.
        "preference_list" => route("/api/preferences", "", None),
        "preference_say" => {
            let mut b = vec![("text", s(need(v, "text", op)?)), ("confirm", Value::Bool(matches!(v.get("confirm"), Ok(Value::Bool(true)))))];
            b.extend(copied(v, &["name", "scope", "cartridge", "plug", "trial"]));
            if arg(v, "project").is_some() {
                b.push(("project", s(&key)));
            }
            route("/api/preferences/say", "", Some(body(b)?))
        }
        "preference_edit" => {
            let mut b = vec![("id", s(need(v, "id", op)?)), ("confirm", Value::Bool(matches!(v.get("confirm"), Ok(Value::Bool(true)))))];
            b.extend(copied(v, &["add", "rule", "text", "strength", "drop", "name", "scope"]));
            route("/api/preferences/edit", "", Some(body(b)?))
        }
        "preference_remove" => route("/api/preferences/remove", "", Some(gated(v, vec![("id", s(need(v, "id", op)?))])?)),
        "preference_plug" | "preference_unplug" => route("/api/preferences/plug", "", Some(body(vec![("id", s(need(v, "id", op)?)), ("on", Value::Bool(op == "preference_plug"))])?)),
        "preference_apply" => {
            let mut b = vec![("text", s(arg(v, "text").ok_or_else(|| bad("preference_apply needs text (the draft)"))?))];
            b.extend(copied(v, &["path"]));
            if arg(v, "project").is_some() {
                b.push(("project", s(&key)));
            }
            route("/api/preferences/apply", "", Some(body(b)?))
        }
        other => Err(bad(format!("op {other}: use view, cycle, sync, rollback, examples, ledger, outcomes, export, import, references, reference_add, distil, distil_import, distil_label, teach_list, teach_start, teach_next, teach_answer, teach_local, clear_examples, clear_ledger, units, unit_on, unit_off, compartment_on, compartment_off, preference_list, preference_say, preference_edit, preference_remove, preference_plug, preference_unplug or preference_apply"))),
    }
}

/// `nm_comprehension`: the 4-level world model of a project (L1 piece, L2 file, L3 module, L4
/// system notes), the routing record, per-lens heads and the serving base cartridges, and the
/// memory tiers. `refresh` (2026-11-26) removes the world-model notes in two calls (a preview with
/// a confirm_token, then the same call with it) so the idle writer writes L1 to L4 again.
/// # Errors
/// A bad project, or the route failing.
pub fn comprehension(v: &Value) -> Result<Value, ApiError> {
    let q = format!("project={}", project(v)?);
    match arg(v, "op").unwrap_or("view") {
        "view" => route("/api/comprehension", &q, None),
        "tiers" => route("/api/memory/tiers", &q, None),
        "memory" => route("/api/memory", "", None),
        "refresh" => route("/api/comprehension/refresh", "", Some(gated(v, vec![("project", s(&project(v)?))])?)),
        other => Err(bad(format!("op {other}: use view, tiers, memory or refresh"))),
    }
}

/// `nm_settings`: the user controls with their values, setting or undoing one (logged), the
/// status of the MicroModels (precision, serving processes, card memory), the projects, the audit
/// log, per-part activity, the stage-time tables, a Claude Code session's state and the parts.
/// Roadmap Phase 4.5 (2026-11-25): the hardware limits (`limits`: saved, in effect, recommended
/// and the profile; `recommend`: the recommendation alone; `set_limit`; `apply_recommended`, with
/// `confirm`), through the interface's `/api/limits` routes.
/// # Errors
/// A missing field, no `confirm` on applying the recommendation, or the route failing.
pub fn settings(v: &Value) -> Result<Value, ApiError> {
    let op = arg(v, "op").unwrap_or("view");
    match op {
        "limits" => route("/api/limits", "", None),
        "recommend" => {
            let all = route("/api/limits", "", None)?;
            Ok(all.get("recommendation").cloned().unwrap_or(Value::Null))
        }
        "set_limit" => route("/api/limits", "", Some(body(vec![("key", s(need(v, "key", op)?)), ("value", s(need(v, "value", op)?))])?)),
        // 2026-11-25: paired devices (the Settings page's devices panel); a pairing code is made
        // only from the interface on this machine.
        "devices" => route("/api/devices", "", None),
        // 2026-11-25: the model each role runs on (the Models page's roles panel).
        "roles" => route("/api/models/roles", "", None),
        // 2026-11-25: downloads of the released models and the base cartridges, each on its own.
        "downloads" => route("/api/downloads", "", None),
        "download" => {
            confirmed(v, op)?;
            let mut b = vec![("what", s(need(v, "what", op)?))];
            if let Some(only) = arg(v, "only").filter(|x| !x.is_empty()) {
                b.push(("only", s(only)));
            }
            route("/api/downloads", "", Some(body(b)?))
        }
        "role_set" => route("/api/models/roles", "", Some(body(vec![("role", s(need(v, "role", op)?)), ("model", s(need(v, "model", op)?))])?)),
        "device_revoke" => {
            confirmed(v, op)?;
            route("/api/devices/revoke", "", Some(body(vec![("id", s(need(v, "id", op)?))])?))
        }
        "apply_recommended" => {
            confirmed(v, op)?;
            route("/api/limits/apply-recommended", "", Some(body(vec![("confirm", Value::Bool(true))])?))
        }
        "view" => route("/api/controls", "", None),
        // `settings.json` key by key: the keys beside the controls (the base cartridge served, the
        // base cartridge compensation, the model each role runs on), each entry with its choices
        // and the key `set` takes for it.
        "saved" => route("/api/controls/saved", "", None),
        "set" => route("/api/controls", "", Some(body(vec![("key", s(need(v, "key", op)?)), ("value", s(need(v, "value", op)?))])?)),
        "undo" => route("/api/undo", "", Some(body(vec![("id", s(need(v, "id", op)?))])?)),
        "status" => route("/api/status", "", None),
        "projects" => route("/api/projects", "", None),
        "designs" => route("/api/settings", "", None),
        "audit" => route("/api/audit", &query(v, &["kind", "project"]), None),
        "activity" => route("/api/activity", &query(v, &["window", "project"]), None),
        "stage_times" => route("/api/stage-times", &query(v, &["file"]), None),
        "session" => route("/api/session", &query(v, &["project"]), None),
        "parts" => route("/api/parts", "", None),
        "airgap" => route("/api/airgap", "", None),
        // 2026-11-26 (Phase B): which MCP clients are connected for a project, with the commands.
        "connect" => route("/api/connect", &format!("project={}", enc(need(v, "project", op)?)), None),
        // 2026-11-26 (Luis Jake Gabriel III: the official cartridge library or third parties): the
        // cartridge sources; browsing and fetching reach huggingface.co only when asked.
        "sources" => route("/api/cartridge-sources", "", None),
        // The user's own repositories for backing up and rolling back their data: nothing is sent
        // unless backup_now is called.
        "backups" => route("/api/backups", "", None),
        "backup_add" => {
            let mut b = vec![("action", s("add")), ("url", s(need(v, "url", op)?)), ("token", Value::Bool(matches!(v.get("token"), Ok(Value::Bool(true)))))];
            if let Ok(p) = v.get("parts") {
                b.push(("parts", p.clone()));
            }
            route("/api/backups", "", Some(body(b)?))
        }
        "backup_remove" => route("/api/backups", "", Some(body(vec![("action", s("remove")), ("id", s(need(v, "id", op)?))])?)),
        "backup_now" => route("/api/backups", "", Some(body(vec![("action", s("backup")), ("id", s(need(v, "id", op)?))])?)),
        "backup_snapshots" => route("/api/backups", "", Some(body(vec![("action", s("snapshots")), ("id", s(need(v, "id", op)?))])?)),
        "backup_preview" => route("/api/backups", "", Some(body(vec![("action", s("preview")), ("id", s(need(v, "id", op)?)), ("commit", s(need(v, "commit", op)?))])?)),
        "backup_restore" => {
            confirmed(v, op)?;
            route("/api/backups", "", Some(body(vec![("action", s("restore")), ("id", s(need(v, "id", op)?)), ("commit", s(need(v, "commit", op)?)), ("confirm", Value::Bool(true))])?))
        }
        "source_add" => {
            let mut b = vec![("action", s("add")), ("repo", s(need(v, "repo", op)?)), ("token", Value::Bool(matches!(v.get("token"), Ok(Value::Bool(true)))))];
            for k in ["kind", "revision"] {
                if let Some(x) = arg(v, k) {
                    b.push((k, s(x)));
                }
            }
            route("/api/cartridge-sources", "", Some(body(b)?))
        }
        "source_remove" => route("/api/cartridge-sources", "", Some(gated(v, vec![("action", s("remove")), ("id", s(need(v, "id", op)?))])?)),
        "source_browse" => route("/api/cartridge-sources", "", Some(body(vec![("action", s("browse")), ("id", s(need(v, "id", op)?))])?)),
        "source_fetch" => {
            confirmed(v, op)?;
            route("/api/cartridge-sources", "", Some(body(vec![("action", s("fetch")), ("id", s(need(v, "id", op)?)), ("path", s(need(v, "path", op)?))])?))
        }
        other => Err(bad(format!(
            "op {other}: use view, saved, set, undo, status, projects, designs, audit, activity, stage_times, session, parts, airgap, connect, devices, device_revoke, roles, role_set, downloads, download, sources, source_add, source_remove, source_browse, source_fetch, backups, backup_add, backup_remove, backup_now, backup_snapshots, backup_preview, backup_restore, limits, recommend, set_limit or apply_recommended"
        ))),
    }
}

/// `nm_weights` with a `model`: the weight view of any installed model through the interface's
/// `/api/models/weights` (without `model`, the tool reads the served base model itself).
/// # Errors
/// The route failing, or not lent.
pub fn weights_of(v: &Value) -> Result<Value, ApiError> {
    route("/api/models/weights", &query_enc(v, &["model"], &["limit"]), None)
}

/// The parity table: every op of the tools in this module with the interface route it reads
/// (`GET`) or writes (`POST`). It is the one list the checks read: this module's tests hold it
/// against the ops each tool answers and against `docs/mcp-tools.md`; the program's `check
/// nmcore.parity` holds it against the routes the interface server answers, in both directions.
/// An op or a route added without its row here fails those checks.
pub const OPS: &[(&str, &str, &str, &str)] = &[
    ("nm_goal", "guidance", "GET", "/api/guidance"),
    ("nm_goal", "reminder_list", "GET", "/api/guidance"),
    ("nm_goal", "loop_outcomes", "GET", "/api/guidance"),
    ("nm_goal", "reminder_add", "POST", "/api/guidance"),
    ("nm_goal", "reminder_edit", "POST", "/api/guidance"),
    ("nm_goal", "loop_add", "POST", "/api/guidance"),
    ("nm_goal", "loop_edit", "POST", "/api/guidance"),
    ("nm_goal", "loop_end", "POST", "/api/guidance"),
    ("nm_goal", "step_add", "POST", "/api/guidance"),
    ("nm_goal", "step_done", "POST", "/api/guidance"),
    ("nm_goal", "step_edit", "POST", "/api/guidance"),
    ("nm_goal", "remove", "POST", "/api/guidance"),
    ("nm_goal", "enable", "POST", "/api/guidance"),
    ("nm_goal", "disable", "POST", "/api/guidance"),
    ("nm_goal", "reset", "POST", "/api/guidance"),
    ("nm_storage", "view", "GET", "/api/storage"),
    ("nm_storage", "settings", "GET", "/api/storage/settings"),
    ("nm_storage", "memories", "GET", "/api/memory/items"),
    ("nm_storage", "prune", "POST", "/api/storage/prune"),
    ("nm_storage", "delete", "POST", "/api/storage/delete"),
    ("nm_storage", "set", "POST", "/api/storage/settings"),
    ("nm_storage", "dense_clear", "POST", "/api/dense/clear"),
    ("nm_storage", "forget_project", "POST", "/api/projects/forget"),
    ("nm_storage", "forget_memory", "POST", "/api/memory/forget"),
    ("nm_storage", "audit_retention", "POST", "/api/audit/retention"),
    ("nm_cartridges", "list", "GET", "/api/cartridges"),
    ("nm_cartridges", "library", "GET", "/api/cartridges/library"),
    ("nm_cartridges", "filters", "GET", "/api/surgery"),
    ("nm_cartridges", "bases", "GET", "/api/cartridges/bases"),
    ("nm_cartridges", "compat", "GET", "/api/cartridges/compat"),
    ("nm_cartridges", "maps", "GET", "/api/parameter-maps"),
    ("nm_cartridges", "map_read", "GET", "/api/parameter-maps/read"),
    ("nm_cartridges", "distils", "GET", "/api/training/distil"),
    ("nm_cartridges", "inspect", "GET", "/api/cartridges/inspect"),
    ("nm_cartridges", "parameters", "GET", "/api/cartridges/parameters"),
    ("nm_cartridges", "parameters_overview", "GET", "/api/parameters/overview"),
    ("nm_cartridges", "overlay", "GET", "/api/parameters/overlay"),
    ("nm_cartridges", "lenses", "GET", "/api/lenses"),
    ("nm_cartridges", "on", "POST", "/api/cartridges"),
    ("nm_cartridges", "off", "POST", "/api/cartridges"),
    ("nm_cartridges", "use", "POST", "/api/cartridges"),
    ("nm_cartridges", "base_serve", "POST", "/api/cartridges/bases"),
    ("nm_cartridges", "base_remove", "POST", "/api/cartridges/bases/remove"),
    ("nm_cartridges", "map_save", "POST", "/api/parameter-maps/save"),
    ("nm_cartridges", "map_import", "POST", "/api/parameter-maps/import"),
    ("nm_cartridges", "map_lineage", "POST", "/api/parameter-maps/lineage"),
    ("nm_cartridges", "map_archive", "POST", "/api/parameter-maps/archive"),
    ("nm_cartridges", "map_restore", "POST", "/api/parameter-maps/restore"),
    ("nm_cartridges", "map_remove", "POST", "/api/parameter-maps/remove"),
    ("nm_cartridges", "map_build", "POST", "/api/lab/build"),
    ("nm_cartridges", "base_build", "POST", "/api/lab/build"),
    ("nm_cartridges", "distil_build", "POST", "/api/lab/build"),
    ("nm_cartridges", "filter_check", "POST", "/api/lab/build"),
    ("nm_cartridges", "filter_on", "POST", "/api/surgery/job-filter"),
    ("nm_cartridges", "filter_off", "POST", "/api/surgery/job-filter"),
    ("nm_cartridges", "filter_remove", "POST", "/api/surgery/job-filter"),
    ("nm_cartridges", "filter_build", "POST", "/api/surgery/job-filter/build"),
    ("nm_cartridges", "set_values", "POST", "/api/cartridges/values"),
    ("nm_cartridges", "parameters_edit", "POST", "/api/cartridges/parameters"),
    ("nm_cartridges", "parameters_export", "POST", "/api/cartridges/parameters/export"),
    ("nm_cartridges", "overlay_edit", "POST", "/api/parameters/overlay"),
    ("nm_cartridges", "mask", "POST", "/api/parameters/overlay"),
    ("nm_cartridges", "unmask", "POST", "/api/parameters/overlay"),
    ("nm_cartridges", "freeze", "POST", "/api/parameters/overlay"),
    ("nm_cartridges", "unfreeze", "POST", "/api/parameters/overlay"),
    ("nm_cartridges", "overlay_undo", "POST", "/api/parameters/overlay"),
    ("nm_cartridges", "overlay_clear", "POST", "/api/parameters/overlay"),
    ("nm_cartridges", "trial", "POST", "/api/cartridges/trial"),
    ("nm_cartridges", "stack_save", "POST", "/api/cartridges/stack"),
    ("nm_cartridges", "cartridge_import", "POST", "/api/cartridges/import"),
    ("nm_cartridges", "lens_switch", "POST", "/api/lenses"),
    ("nm_cartridges", "lens_remove", "POST", "/api/lenses"),
    ("nm_learning", "view", "GET", "/api/learning"),
    ("nm_learning", "examples", "GET", "/api/examples"),
    ("nm_learning", "ledger", "GET", "/api/ledger"),
    ("nm_learning", "outcomes", "GET", "/api/outcomes"),
    ("nm_learning", "references", "GET", "/api/training/references"),
    ("nm_learning", "distil", "GET", "/api/training/distil"),
    ("nm_learning", "teach_list", "GET", "/api/training/distil/teach"),
    ("nm_learning", "units", "GET", "/api/learned-units"),
    ("nm_learning", "preference_list", "GET", "/api/preferences"),
    ("nm_learning", "cycle", "POST", "/api/learning/cycle"),
    ("nm_learning", "sync", "POST", "/api/learning/cycle"),
    ("nm_learning", "rollback", "POST", "/api/learning/cycle"),
    ("nm_learning", "export", "POST", "/api/training/export"),
    ("nm_learning", "import", "POST", "/api/training/import"),
    ("nm_learning", "reference_add", "POST", "/api/training/references"),
    ("nm_learning", "distil_import", "POST", "/api/training/distil/import"),
    ("nm_learning", "distil_label", "POST", "/api/training/distil/label"),
    ("nm_learning", "teach_start", "POST", "/api/training/distil/teach"),
    ("nm_learning", "teach_next", "POST", "/api/training/distil/teach"),
    ("nm_learning", "teach_answer", "POST", "/api/training/distil/teach"),
    ("nm_learning", "teach_local", "POST", "/api/lab/build"),
    ("nm_learning", "clear_examples", "POST", "/api/examples/clear"),
    ("nm_learning", "clear_ledger", "POST", "/api/ledger/clear"),
    ("nm_learning", "unit_on", "POST", "/api/learned-units"),
    ("nm_learning", "unit_off", "POST", "/api/learned-units"),
    ("nm_learning", "compartment_on", "POST", "/api/learned-units"),
    ("nm_learning", "compartment_off", "POST", "/api/learned-units"),
    ("nm_learning", "preference_say", "POST", "/api/preferences/say"),
    ("nm_learning", "preference_edit", "POST", "/api/preferences/edit"),
    ("nm_learning", "preference_remove", "POST", "/api/preferences/remove"),
    ("nm_learning", "preference_plug", "POST", "/api/preferences/plug"),
    ("nm_learning", "preference_unplug", "POST", "/api/preferences/plug"),
    ("nm_learning", "preference_apply", "POST", "/api/preferences/apply"),
    ("nm_comprehension", "view", "GET", "/api/comprehension"),
    ("nm_comprehension", "tiers", "GET", "/api/memory/tiers"),
    ("nm_comprehension", "memory", "GET", "/api/memory"),
    ("nm_comprehension", "refresh", "POST", "/api/comprehension/refresh"),
    ("nm_settings", "view", "GET", "/api/controls"),
    ("nm_settings", "saved", "GET", "/api/controls/saved"),
    ("nm_settings", "status", "GET", "/api/status"),
    ("nm_settings", "projects", "GET", "/api/projects"),
    ("nm_settings", "designs", "GET", "/api/settings"),
    ("nm_settings", "audit", "GET", "/api/audit"),
    ("nm_settings", "activity", "GET", "/api/activity"),
    ("nm_settings", "stage_times", "GET", "/api/stage-times"),
    ("nm_settings", "session", "GET", "/api/session"),
    ("nm_settings", "parts", "GET", "/api/parts"),
    ("nm_settings", "airgap", "GET", "/api/airgap"),
    ("nm_settings", "connect", "GET", "/api/connect"),
    ("nm_settings", "devices", "GET", "/api/devices"),
    ("nm_settings", "roles", "GET", "/api/models/roles"),
    ("nm_settings", "downloads", "GET", "/api/downloads"),
    ("nm_settings", "sources", "GET", "/api/cartridge-sources"),
    ("nm_settings", "backups", "GET", "/api/backups"),
    ("nm_settings", "limits", "GET", "/api/limits"),
    ("nm_settings", "recommend", "GET", "/api/limits"),
    ("nm_settings", "set", "POST", "/api/controls"),
    ("nm_settings", "undo", "POST", "/api/undo"),
    ("nm_settings", "device_revoke", "POST", "/api/devices/revoke"),
    ("nm_settings", "role_set", "POST", "/api/models/roles"),
    ("nm_settings", "download", "POST", "/api/downloads"),
    ("nm_settings", "source_add", "POST", "/api/cartridge-sources"),
    ("nm_settings", "source_remove", "POST", "/api/cartridge-sources"),
    ("nm_settings", "source_browse", "POST", "/api/cartridge-sources"),
    ("nm_settings", "source_fetch", "POST", "/api/cartridge-sources"),
    ("nm_settings", "backup_add", "POST", "/api/backups"),
    ("nm_settings", "backup_remove", "POST", "/api/backups"),
    ("nm_settings", "backup_now", "POST", "/api/backups"),
    ("nm_settings", "backup_snapshots", "POST", "/api/backups"),
    ("nm_settings", "backup_preview", "POST", "/api/backups"),
    ("nm_settings", "backup_restore", "POST", "/api/backups"),
    ("nm_settings", "set_limit", "POST", "/api/limits"),
    ("nm_settings", "apply_recommended", "POST", "/api/limits/apply-recommended"),
];

/// Interface routes whose feature a tool outside this module serves with its own function (the
/// same data, read or written by the engine process the agent talks to): (method, route, tool).
pub const OTHER_TOOL_ROUTES: &[(&str, &str, &str)] = &[
    ("GET", "/api/live", "nm_live"),
    ("GET", "/api/models/weights", "nm_weights"),
    ("GET", "/api/units", "nm_unit op=list"),
    ("POST", "/api/units", "nm_unit op=register, unregister, load, unload"),
    ("GET", "/api/goals", "nm_goal"),
    ("POST", "/api/goals", "nm_goal objective, or id with status"),
    ("POST", "/api/memory/add", "nm_memory op=note"),
    ("POST", "/api/memory/pin", "nm_memory op=pin, unpin"),
    ("POST", "/api/memory/sweep", "nm_memory op=sweep"),
];

/// Interface routes no tool reaches, each with the reason: (method, route, why).
pub const NOT_IN_MCP: &[(&str, &str, &str)] = &[
    ("POST", "/api/pair/code", "a pairing code is made only by the person at the machine, from the interface or `neuromorph-nano pair`"),
    ("POST", "/api/pair", "a device pairs itself from its own browser with the code"),
    ("GET", "/api/token", "the local write token is given to the development server's pages only, under --dev"),
    ("GET", "/api/live/stream", "a Server-Sent Events stream for a browser; nm_live returns the same state on request"),
];

const fn field(name: &'static str, kind: FieldKind, doc: &'static str) -> Field {
    Field { name, kind, required: false, doc, default: None }
}
const CONFIRM: Field = field("confirm", FieldKind::Bool, "true to carry out an action that changes data");
/// 2026-11-26: every delete takes two calls: the first returns a preview and this token; the same
/// call again with it deletes.
const CONFIRM_TOKEN: Field = field("confirm_token", FieldKind::String, "a delete's second call: the confirm_token its first call (the preview) returned");

/// Inputs of `nm_storage`.
pub const IN_STORAGE: &[Field] = &[
    field("op", FieldKind::String, "view (default), settings, prune, delete, set, dense_clear, forget_project, memories, forget_memory or audit_retention; every delete is two calls: a preview with a confirm_token, then the same call with it"),
    field("category", FieldKind::String, "prune: memory, vectors, compression, cartridges or junk (only rebuildable items and junk go)"),
    field("path", FieldKind::String, "delete: exact path of one removable or junk item from the view"),
    field("id", FieldKind::String, "forget_memory: the memory's id from memories"),
    field("limit", FieldKind::Integer, "memories: how many (default 200)"),
    field("q", FieldKind::String, "memories: only those whose text holds this"),
    CONFIRM_TOKEN,
    field("key", FieldKind::String, "set: storage.disk_floor_gb"),
    field("value", FieldKind::String, "set: the new value"),
    field("project", FieldKind::String, "dense_clear: project key (nano for the Nano itself)"),
    CONFIRM,
];
/// Inputs of `nm_cartridges`.
pub const IN_CARTRIDGES: &[Field] = &[
    field("op", FieldKind::String, "list (default), library, filters, on, off, use, bases, compat, maps, map_read, map_save, map_import, map_lineage, map_archive, map_restore, map_remove, map_build, base_serve, base_remove, base_build, distil_build (train teacher pairs into a learned cartridge of the chosen kind through the held-out gate), distils (the distilled cartridges with their gate figures), filter_on, filter_off, filter_build, filter_remove, inspect, set_values, parameters, parameters_edit, parameters_export, parameters_overview (an opened object's tensors in layout order, or one tensor's region as a w x h grid of cell min, max, mean and RMS, exact values at one weight per cell), overlay (an object's parameter overlay: edits, masks and freezes held beside it, applied when the engine seats it; the open weights are edited here), overlay_edit, mask, unmask, freeze, unfreeze, overlay_undo, overlay_clear (two calls: a preview with a confirm_token, then the same call with it), trial, stack_save, cartridge_import, lenses, lens_switch or lens_remove"),
    field("id", FieldKind::String, "on, off: cartridge id; lens_switch, lens_remove: the lens id (embedder-lens-<mark>); unmask, unfreeze: one overlay entry id"),
    field("model", FieldKind::String, "base_serve: the MicroModel (qwen3-0.6b, qwen3-1.7b, qwen3-embedding-0.6b or qwen3-reranker-0.6b)"),
    field("choice", FieldKind::String, "base_serve: a base cartridge key in the cartridge library (for example writer-base@4), auto (newest usable) or open (open weights)"),
    field("key", FieldKind::String, "base_remove: the base cartridge version to delete; inspect, set_values: a cartridge's id@version (or id); parameters, parameters_edit, overlay and its ops: open:<model> (overlay only), a cartridge's id@version, a job filter as filter:<role>/<job>, an adapter version <id>@<n> or a learning unit's id (parameters without key lists every object holding parameters)"),
    field("whole", FieldKind::Bool, "mask, freeze: true for the whole object (also when no path or selection is given): a whole mask leaves its effect out while it stays plugged; of the open weights, every layer an exact identity"),
    field("ids", FieldKind::Array, "unmask, unfreeze: overlay entry ids (as overlay lists them) to take out; id for one; or a path or selection meeting them"),
    field("tensor", FieldKind::String, "parameters: one tensor's name, for a page of its values (offset, limit); parameters_overview: the tensor to summarise as a grid (without it, the set's tensors in layout order)"),
    field("w", FieldKind::Integer, "parameters_overview: grid columns (default 128, at most 1024, at most the region's columns; w x h at most 65536)"),
    field("h", FieldKind::Integer, "parameters_overview: grid rows (default 128, at most 1024, at most the region's rows)"),
    field("r0", FieldKind::Integer, "parameters_overview: the region's first row (a vector is one row; default 0)"),
    field("r1", FieldKind::Integer, "parameters_overview: the region's end row, exclusive (default the tensor's rows)"),
    field("c0", FieldKind::Integer, "parameters_overview: the region's first column (default 0)"),
    field("c1", FieldKind::Integer, "parameters_overview: the region's end column, exclusive (default the tensor's columns)"),
    field("maps", FieldKind::Bool, "parameters: true lists the parameter maps beside the object, with their values on its layers and points"),
    field("scale", FieldKind::String, "parameters_edit: a strength from 0 to 4 for every scalable tensor (a gain moves toward 1, the rest toward 0)"),
    field("layers", FieldKind::Object, "parameters_edit: layer number to on, off or a scale (off drops a layer's points, or zeroes its change)"),
    field("edit", FieldKind::String, "parameters_edit, overlay_edit (no structural ones there): the operation on the selection: set, add, mul, scale, clamp, zero, identity, negate, paste, copy, interpolate, import, drop, add_point or rank"),
    field("selection", FieldKind::Object, "parameters_edit: tensor, layer, target, part, range [a, b), index, rows, cols, head with head_dim, channel"),
    field("value", FieldKind::String, "parameters_edit: the number for set, add, mul or scale"),
    field("lo", FieldKind::String, "parameters_edit: clamp's lower bound"),
    field("hi", FieldKind::String, "parameters_edit: clamp's upper bound"),
    field("source", FieldKind::Object, "parameters_edit: copy, interpolate: {key} of another version or cartridge, or {map: <role>/job-filter-<job>.json}"),
    field("source_b", FieldKind::Object, "parameters_edit, overlay_edit: interpolate between two versions: the second source (as source), the result source + t * (source_b - source)"),
    field("t", FieldKind::String, "parameters_edit: interpolate's weight toward the source, 0 to 1"),
    field("target", FieldKind::String, "parameters_edit: add_point's node (for example L12.o)"),
    field("width", FieldKind::Integer, "parameters_edit: add_point's width"),
    field("rank", FieldKind::Integer, "parameters_edit: rank's new rank (the best approximation of that rank of a low-rank pair, by truncated singular value decomposition)"),
    field("ops", FieldKind::Array, "parameters_edit: several operations in order, each {op, selection or path, ...}"),
    field("description", FieldKind::String, "parameters_edit: the new version's description"),
    field("preview", FieldKind::Bool, "parameters_edit and the overlay ops: true returns the difference (old and new values, norms) and writes nothing"),
    field("use", FieldKind::String, "use, stack_save: the use (general, hardware:<piece>, project:<key>, job:<name>, preference)"),
    field("stack", FieldKind::String, "use: stack id, empty switches the use off; trial, stack_save: the stack id"),
    field("file", FieldKind::String, "map_read, map_save, map_remove: the map's file name in the role's folder; parameters_edit import: a values file (nm-parameter-values/1, as parameters_export writes)"),
    field("files", FieldKind::Array, "map_archive, map_restore, map_remove: several map file names of one role"),
    field("archived", FieldKind::Bool, "map_remove: true deletes from the role's archive"),
    field("made_on", FieldKind::String, "map_lineage: what the map was measured on, a base cartridge id@version or open:<model id>"),
    field("made_by", FieldKind::String, "map_lineage: the command that made it"),
    field("purpose", FieldKind::String, "map_lineage: serving, base-build, job-filter, probe, measurement or experiment"),
    field("note", FieldKind::String, "map_lineage: a note"),
    field("section", FieldKind::String, "map_read, map_save: a path of keys or list indexes split by / (for example jobs/drone/layers)"),
    field("layer", FieldKind::String, "map_read, map_save: one more step, matched as L<n>, then <n>, then index n"),
    field("offset", FieldKind::Integer, "map_read: the first key, element or row of a page; parameters: the first value of a page"),
    field("limit", FieldKind::Integer, "map_read: how many keys, elements or rows; parameters: how many values (at most 4096)"),
    field("content", FieldKind::String, "map_save: the new value as JSON text (the whole map, or the value section and layer name); it keeps the map's shape"),
    field("path", FieldKind::String, "map_import: a map file on this machine; cartridge_import: a stack file from library export; parameters, parameters_edit, parameters_export: a parameter path (L3, L3.o.gain, L3.o.gain[0:16], down[2:4, 0:8], *)"),
    field("role", FieldKind::String, "job filters, maps and distil_build: embedder, reader, reranker or writer (distil_build: writer by default; built: the Writer and the Reader from teacher completions, adapter or filter; the Reranker from teacher scores, adapter, filter or readout; the embedder answers with what it still needs)"),
    field("set", FieldKind::String, "distil_build: the teacher data set to train on (all by default)"),
    field("target", FieldKind::String, "distil_build: the kind of learned cartridge to fill: adapter (lens, teacher-loop, unit, steer), filter (a job filter: channel, teacher-loop), preference (readout) or readout (readout, teacher-loop); adapter by default, filter for channel"),
    field("technique", FieldKind::String, "distil_build: lens (default), teacher-loop, channel, unit, readout or steer; the ones not built yet answer with what they need"),
    field("rounds", FieldKind::String, "distil_build with teacher-loop: rounds at most, 1 to 8 (default 4)"),
    field("budget_s", FieldKind::String, "distil_build with unit: the wall budget of each forward-only fit in seconds, 1 to 120 (default 20)"),
    field("name", FieldKind::String, "map_import: the file name in the role's folder (default the file's own); parameters_edit: the new version's name"),
    field("replace", FieldKind::Bool, "map_import: true replaces a map of that name (kept under backups/ first)"),
    field("values", FieldKind::Object, "set_values: name to value (text; null takes a pref.key.* out); parameters_edit: tensor name to its values (a list of every value, or {offset, values}), shape checked"),
    field("trial", FieldKind::Bool, "trial: true puts the stack on trial, false serves it (needs its evidence)"),
    field("cartridges", FieldKind::Array, "stack_save: cartridge keys in order"),
    field("description", FieldKind::String, "stack_save: what the stack is for"),
    field("switch", FieldKind::Bool, "stack_save: switch the use to it (default: when the use runs an earlier version)"),
    field("project", FieldKind::String, "lenses: only this project's lenses"),
    field("on", FieldKind::Bool, "lens_switch: true plugs the lens in, false unplugs it"),
    field("version", FieldKind::Integer, "lens_remove: one version (default every version)"),
    field("job", FieldKind::String, "job filters: the job name"),
    field("layers", FieldKind::String, "filter_build: all (default) or a range like 12-27"),
    field("share", FieldKind::String, "map_build: the part of each matrix mapped out, 0.05 to 0.60 (default 0.40)"),
    field("device", FieldKind::String, "map_build: gpu (default) or cpu"),
    CONFIRM,
    CONFIRM_TOKEN,
];
/// Inputs of `nm_learning`.
pub const IN_LEARNING: &[Field] = &[
    field("op", FieldKind::String, "view (default), cycle, sync (bring stored examples into the engine), rollback, examples, ledger, outcomes, export (the training store to a file under exports/), import (a training store file, hash checked), references (your listed reference material), reference_add (list material: path and licence), distil (the teacher data for distillation by set, with the distilled cartridges), distil_import (curated teacher pairs from a JSON lines file on this machine: file, set, licence, teacher), teach_list, teach_start (open a teaching set for a live teacher: prompts, instruction, licence), teach_next (its open prompts, for you to answer as the teacher), teach_answer (your answers; they join the teacher data), teach_local (an installed open-weights model answers the open prompts, on the graphics card), distil_label (an outcome for teacher pairs: ids or set, outcome, why; only used and superseded pairs are trained on), clear_examples, clear_ledger, units (every learned unit by compartment), unit_on, unit_off (plug one in or out), compartment_on or compartment_off (all of a compartment's units); preference_list, preference_say (the user's words to rules: a preview, then confirm: true writes the cartridge), preference_edit, preference_remove (two calls: a preview with a confirm_token, then the same call with it), preference_plug, preference_unplug, preference_apply (a draft checked against the rules in force)"),
    field("text", FieldKind::String, "preference_say: the user's words, as they said them; preference_edit with rule: the rule's new words; preference_apply: the draft"),
    field("name", FieldKind::String, "preference_say, preference_edit: the cartridge's name (a new name makes a new cartridge)"),
    field("scope", FieldKind::String, "preference_say, preference_edit: global, project:<key>, kinds:<ext>,<ext>, folders:<path>,<path> (joined by ;)"),
    field("cartridge", FieldKind::String, "preference_say: add to this cartridge id"),
    field("plug", FieldKind::Bool, "preference_say: plug the cartridge in once written (default true)"),
    field("trial", FieldKind::Bool, "preference_say: write it on trial (recorded, never applied)"),
    field("id", FieldKind::String, "preference_edit, preference_remove, preference_plug, preference_unplug: the cartridge id"),
    field("add", FieldKind::String, "preference_edit: new words to add as rules"),
    field("rule", FieldKind::String, "preference_edit: the rule id (r1, r2, ...) that text rewrites or strength sets"),
    field("strength", FieldKind::String, "preference_edit with rule: a want's strength, 0.1 to 2.0 (0.5 by default; strongly prefer gives 1.0, slightly prefer 0.25)"),
    field("drop", FieldKind::Array, "preference_edit: rule ids to take out"),
    field("unit", FieldKind::String, "unit_on, unit_off: <compartment>/<id> as units lists it"),
    field("compartment", FieldKind::String, "compartment_on, compartment_off: nano, shared or a project key (the project by default)"),
    field("name", FieldKind::String, "export: file name ending in .jsonl (optional)"),
    field("file", FieldKind::String, "import: path of a training store export on this machine; distil_import: a JSON lines file of teacher pairs (prompt, completion, teacher, licence, optional topk; for the Reranker a score from 0 to 1, the prompt a query and the completion a document)"),
    field("set", FieldKind::String, "teach_start, teach_next, teach_answer: the teaching set's name; distil_import: the set's name (the file's name by default); distil, distil_label: one set"),
    field("prompts", FieldKind::Array, "teach_start: the prompts the teacher answers (questions or tasks, as text)"),
    field("instruction", FieldKind::String, "teach_start: what the teacher is to do in every answer (shown to the teacher only, so the student learns to do it unasked)"),
    field("count", FieldKind::Integer, "teach_next: how many open prompts (10 by default, at most 50)"),
    field("answers", FieldKind::Array, "teach_answer: the teacher's answers, each {prompt_id (from teach_next) or prompt, completion}, and for a preference cartridge verdict (accept or reject), paths and tool, and for the Reranker score (0 to 1, how well the completion, a document, answers the prompt, a query)"),
    field("teacher", FieldKind::String, "distil_import: the teacher, for rows that do not name one; teach_answer: who answered (your name as the agent and its model, for example claude-code with its model, antigravity, or a local model)"),
    field("ids", FieldKind::Array, "distil_label: the teacher pairs' ids"),
    field("outcome", FieldKind::String, "distil_label: used, superseded, failed, duplicate, control or unused"),
    field("why", FieldKind::String, "distil_label: the reason, in words"),
    field("limit", FieldKind::Integer, "distil: rows listed at most (50 by default)"),
    field("path", FieldKind::String, "reference_add: a file or folder on this machine; preference_apply: the draft's file path (a code path is checked, never rewritten)"),
    field("licence", FieldKind::String, "reference_add: the material's licence (for example MIT, CC-BY-4.0, own); distil_import: the teacher outputs' licence, for rows that do not name one"),
    field("source", FieldKind::String, "reference_add: where it came from (optional)"),
    field("domain", FieldKind::String, "reference_add: a domain tag (optional)"),
    field("project", FieldKind::String, "project key (nano by default)"),
    field("version", FieldKind::Integer, "rollback: engine snapshot version"),
    field("cycle", FieldKind::String, "rollback: audit id of the cycle to roll back"),
    CONFIRM,
    CONFIRM_TOKEN,
];
/// Inputs of `nm_comprehension`.
pub const IN_COMPREHENSION: &[Field] = &[
    field("op", FieldKind::String, "view (default), tiers, memory or refresh (the world-model notes removed so the idle writer writes them again; two calls: a preview with a confirm_token, then the same call with it)"),
    field("project", FieldKind::String, "project key (nano by default)"),
    CONFIRM_TOKEN,
];
/// Inputs of `nm_settings`.
pub const IN_SETTINGS: &[Field] = &[
    field("op", FieldKind::String, "view (default), saved (settings.json key by key: the keys beside the controls, each entry with its choices and the key set takes for it), set, undo, status, projects, designs, audit, activity, stage_times, session, parts, airgap, connect, devices, device_revoke, roles, role_set, downloads, download, sources (the official and third-party cartridge libraries), source_add, source_remove (two calls: a preview with a confirm_token, then the same call with it), source_browse (lists a source's files from huggingface.co), source_fetch (fetches one listed file and imports it through the cartridge library's checks; needs confirm), backups (your own repositories for backup and rollback), backup_add, backup_remove, backup_now (sends a snapshot of the chosen parts to your repository), backup_snapshots, backup_preview (what a restore would write; writes nothing), backup_restore (needs confirm; Audit undoes it), limits, recommend, set_limit or apply_recommended"),
    field("repo", FieldKind::String, "source_add: a Hugging Face repository, owner/name"),
    field("url", FieldKind::String, "backup_add: your repository's git URL (GitHub or any git host, or a Hugging Face repository) or a local folder"),
    field("parts", FieldKind::Array, "backup_add: what it holds: memory, maps, cartridges, units, settings (all but cartridges by default)"),
    field("commit", FieldKind::String, "backup_preview, backup_restore: a snapshot's commit id as backup_snapshots lists it"),
    field("revision", FieldKind::String, "source_add: a branch, tag or commit (main by default)"),
    field("token", FieldKind::Bool, "source_add: true sends your Hugging Face token to huggingface.co for this source (private or gated libraries); backup_add: true sends your token for that host (HF_TOKEN for Hugging Face, GITHUB_TOKEN or GH_TOKEN for other hosts) to that host only"),
    field("path", FieldKind::String, "source_fetch: the file's path in the source, as source_browse lists it"),
    field("key", FieldKind::String, "set: a control key as view lists it, or one entry of a settings.json map as saved lists it (base_cartridges:<model id>, base_compensation:<model id>, role_models:<role>); set_limit: vram_mb, ram_mb, cpu_threads, or a memory area: content_mb, caches_mb, weights_mb"),
    field("value", FieldKind::String, "set: the new value; set_limit: auto or a number (vram_mb 0 is processor only from the next start)"),
    field("role", FieldKind::String, "role_set: reader, writer, embedder or reranker"),
    field("model", FieldKind::String, "role_set: an installed model id, or default"),
    field("what", FieldKind::String, "download: models:<model id> (that model's files) or cartridges (the serving base cartridges); needs confirm"),
    field("only", FieldKind::String, "download: only the files whose name ends with this (for example .json)"),
    field("id", FieldKind::String, "undo: audit id of the change; device_revoke: the paired device id; source_remove, source_browse, source_fetch: the source's id as sources lists it; backup_remove, backup_now, backup_snapshots, backup_preview, backup_restore: the repository's id as backups lists it"),
    field("kind", FieldKind::String, "audit: row kind; source_add: model (default) or dataset"),
    field("project", FieldKind::String, "audit, activity, session: project key; connect: the project's folder (or key)"),
    field("window", FieldKind::String, "activity: time window, as the interface names it"),
    field("file", FieldKind::String, "stage_times: another stage-bench report"),
    field("limit", FieldKind::Integer, "audit: rows, newest first"),
    CONFIRM,
    CONFIRM_TOKEN,
];

#[cfg(test)]
mod tests {
    use super::*;

    fn input(pairs: &[(&str, Value)]) -> Value {
        Value::Object(pairs.iter().map(|(k, v)| ((*k).to_owned(), v.clone())).collect())
    }

    #[test]
    fn parity_destructive_ops_need_confirm() {
        // Changes that delete nothing still need confirm: true.
        let no = [
            (learning as fn(&Value) -> Result<Value, ApiError>, input(&[("op", s("rollback"))])),
            (settings, input(&[("op", s("apply_recommended"))])),
        ];
        for (f, v) in no {
            let e = f(&v).unwrap_err();
            assert!(e.message.contains("confirm"), "{v:?}: {}", e.message);
        }
        // 2026-11-26: every delete goes to the interface's routes, where the two-step gate
        // (preview with a confirm_token, then the same call with it) applies; no storage function
        // is called directly. Without lent routes (this test) they stop there.
        let gated = [
            (storage as fn(&Value) -> Result<Value, ApiError>, input(&[("op", s("prune")), ("category", s("vectors"))])),
            (storage, input(&[("op", s("delete")), ("path", s("memory/world-model/notes.jsonl"))])),
            (storage, input(&[("op", s("dense_clear"))])),
            (storage, input(&[("op", s("forget_project")), ("project", s("some-project"))])),
            (storage, input(&[("op", s("forget_memory")), ("id", s("ep:1"))])),
            (storage, input(&[("op", s("audit_retention"))])),
            (cartridges, input(&[("op", s("filter_remove")), ("role", s("reader")), ("job", s("read"))])),
            (learning, input(&[("op", s("clear_examples"))])),
            (learning, input(&[("op", s("clear_ledger"))])),
            (cartridges, input(&[("op", s("lens_remove")), ("id", s("embedder-lens-0000"))])),
            (cartridges, input(&[("op", s("overlay_clear")), ("key", s("open:qwen3-0.6b"))])),
            (comprehension, input(&[("op", s("refresh"))])),
            (learning, input(&[("op", s("preference_remove")), ("id", s("stated-x"))])),
            (|v: &Value| guidance(v, "nano"), input(&[("op", s("remove")), ("kind", s("reminder")), ("item", s("r1"))])),
        ];
        for (f, v) in gated {
            let e = f(&v).unwrap_err();
            assert!(e.message.contains("not lent"), "{v:?}: {}", e.message);
        }
    }

    #[test]
    fn parity_refuses_bad_ops_and_project_keys() {
        assert!(storage(&input(&[("op", s("wipe"))])).is_err());
        assert!(cartridges(&input(&[("op", s("burn"))])).is_err());
        assert!(learning(&input(&[("project", s("../etc"))])).unwrap_err().message.contains("project"));
        assert!(comprehension(&input(&[("project", s(".hidden"))])).is_err());
        assert!(storage(&input(&[("op", s("prune")), ("confirm", Value::Bool(true))])).unwrap_err().message.contains("category"));
        assert!(settings(&input(&[("op", s("set_limit")), ("value", s("auto"))])).unwrap_err().message.contains("key"));
        assert!(settings(&input(&[("op", s("overclock"))])).unwrap_err().message.contains("apply_recommended"));
        assert!(learning(&input(&[("op", s("preference_say"))])).unwrap_err().message.contains("text"));
        assert!(learning(&input(&[("op", s("preference_edit"))])).unwrap_err().message.contains("id"));
        assert!(learning(&input(&[("op", s("preference_apply"))])).unwrap_err().message.contains("text"));
        // The agent's guidance: an op is known and asks for the fields it needs, and a bad key is
        // refused before any route.
        assert!(guidance(&input(&[("op", s("loop_end"))]), "nano").unwrap_err().message.contains("item"));
        assert!(guidance(&input(&[("op", s("enable")), ("item", s("r1"))]), "nano").unwrap_err().message.contains("kind"));
        assert!(guidance(&input(&[("op", s("juggle"))]), "nano").unwrap_err().message.contains("reminder_add"));
        assert!(guidance(&input(&[("op", s("guidance")), ("project", s("../x"))]), "nano").unwrap_err().message.contains("project"));
        assert!(guidance(&input(&[("op", s("reminder_add")), ("text", s("x")), ("trigger", s("stop"))]), "nano").unwrap_err().message.contains("not lent"));
    }

    #[test]
    fn parity_tools_are_in_the_table_and_on_demand() {
        // The MCP contract check runs first, in this one test, as both touch the session's
        // dynamically activated tools.
        crate::mcp::check().unwrap();
        let api = crate::tools::api();
        for name in ["nm_storage", "nm_cartridges", "nm_learning", "nm_comprehension", "nm_settings", "nm_live"] {
            assert!(api.operations().iter().any(|o| o.name == name), "{name} missing from the tool table");
            assert!(crate::mcp::PRIORITY_TOOLS.contains(&name), "{name} missing from the priority list");
            assert!(!crate::mcp::LEAN_TOOLS.contains(&name), "{name} must not be lean");
            assert!(crate::mcp::INSTRUCTIONS.contains(name), "{name} missing from the on-demand list");
        }
        let added = crate::mcp::trigger_dynamic_tools(5, true, "nm_context_pack", Some("how much disk storage do the cartridges use"));
        crate::mcp::reset_dynamic_tools();
        assert!(added.contains(&"nm_storage") && added.contains(&"nm_cartridges"), "{added:?}");
    }

    type Tool = fn(&Value) -> Result<Value, ApiError>;

    /// The tools of the parity table, by name.
    fn tools() -> Vec<(&'static str, Tool)> {
        vec![("nm_goal", (|v: &Value| guidance(v, "nano")) as Tool), ("nm_storage", storage), ("nm_cartridges", cartridges), ("nm_learning", learning), ("nm_comprehension", comprehension), ("nm_settings", settings)]
    }

    /// The ops a tool's answer to an unknown op lists.
    fn listed(f: Tool) -> std::collections::BTreeSet<String> {
        let m = f(&input(&[("op", s("no_such_op"))])).unwrap_err().message;
        let list = m.split_once(": use ").map(|x| x.1.to_owned()).unwrap_or_default();
        list.replace(" or ", ", ").split(", ").map(|x| x.trim().to_owned()).filter(|x| !x.is_empty()).collect()
    }

    /// A project document of the development tree; `None` in a tree without the docs.
    fn doc(name: &str) -> Option<String> {
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("docs").join(name)).ok()
    }

    /// The parity table and each tool's own op list are the same set, with no row twice.
    #[test]
    fn parity_table_is_every_op() {
        for (tool, f) in tools() {
            let table: std::collections::BTreeSet<String> = OPS.iter().filter(|r| r.0 == tool).map(|r| r.1.to_owned()).collect();
            assert_eq!(table, listed(f), "{tool}: the parity table (OPS) and the tool's op list differ");
            assert_eq!(table.len(), OPS.iter().filter(|r| r.0 == tool).count(), "{tool}: an op is in the parity table twice");
        }
        let known: Vec<&str> = tools().iter().map(|t| t.0).collect();
        assert!(OPS.iter().all(|r| known.contains(&r.0) && matches!(r.2, "GET" | "POST") && r.3.starts_with("/api/")), "a parity row names an unknown tool, method or route");
    }

    /// Every op the table names is answered by its tool: it asks for a field, a confirmation or
    /// the routes (not lent in a test), and never falls through to the unknown op answer.
    #[test]
    fn parity_table_ops_dispatch() {
        for (tool, f) in tools() {
            for r in OPS.iter().filter(|r| r.0 == tool) {
                // These two read the disk through `crate::storage` without a route.
                if tool == "nm_storage" && matches!(r.1, "view" | "settings") {
                    continue;
                }
                if let Err(e) = f(&input(&[("op", s(r.1))])) {
                    assert!(!e.message.starts_with(&format!("op {}: use", r.1)), "{tool} {}: named in the parity table, not answered by the tool", r.1);
                }
            }
        }
    }

    /// The documents name what the table names: every op in its tool's row of `docs/mcp-tools.md`,
    /// every route (the ones with another tool or none included) in `docs/interface-api.md`.
    #[test]
    fn parity_table_is_documented() {
        if let Some(text) = doc("mcp-tools.md") {
            for (tool, _) in tools() {
                let row = text.lines().find(|l| l.starts_with(&format!("| `{tool}` |"))).unwrap_or_else(|| panic!("docs/mcp-tools.md has no row for {tool}"));
                let words: std::collections::BTreeSet<&str> = row.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).collect();
                let missing: Vec<&str> = OPS.iter().filter(|r| r.0 == tool && !words.contains(r.1)).map(|r| r.1).collect();
                assert!(missing.is_empty(), "docs/mcp-tools.md, row {tool}: ops not named: {missing:?}");
            }
        }
        if let Some(text) = doc("interface-api.md") {
            // A route is named when its path is followed by something that does not continue it.
            let named = |path: &str| text.match_indices(path).any(|(i, _)| !text[i + path.len()..].starts_with(|c: char| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '/')));
            let routes = OPS.iter().map(|r| r.3).chain(OTHER_TOOL_ROUTES.iter().map(|r| r.1)).chain(NOT_IN_MCP.iter().map(|r| r.1));
            let missing: std::collections::BTreeSet<&str> = routes.filter(|p| !named(p)).collect();
            assert!(missing.is_empty(), "docs/interface-api.md does not name these routes: {missing:?}");
        }
    }
}
