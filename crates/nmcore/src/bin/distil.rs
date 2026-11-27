//! Teacher data for distillation, and the distilled cartridges on the shelf: importing curated
//! teacher pairs from a local JSON lines file into the distil store (`nmtype::examples::DISTIL`,
//! the Nano's own `memory/training/distil.jsonl`), listing them by set, labelling their outcome,
//! undoing an import or a label, and the view of the distilled cartridges with their gate figures.
//! The same functions back the interface (`/api/training/distil...`) and MCP (`nm_learning`).
//!
//! An input line is a JSON object: `prompt`, `completion` (both required), `teacher`, `licence`
//! (required here or for the whole import), optionally `topk` (per completion token, the teacher's
//! top `[token id, log-probability]` pairs under the Qwen3 tokenizer) and `template` (`chat`, the
//! default, or `raw`). The split is fixed by the prompt's hash; a row already in the store (the same
//! prompt, completion and teacher) is passed over. Every import lists its file as reference material
//! (`references.json`, licence and hash) and labels its rows `used` (healthy, read by a fit); a row
//! labelled `failed`, `duplicate`, `control` or `unused` is never read by a fit. No network call.
use nmjson::Value;
use nmtype::examples::{self as ex, DistilRow};
use std::collections::{BTreeMap, BTreeSet};
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
fn text(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default()
}
fn strs<I: IntoIterator<Item = String>>(i: I) -> Value {
    Value::Array(i.into_iter().map(|x| s(&x)).collect())
}

fn memory(home: &Path) -> PathBuf {
    home.join("memory")
}
fn store(home: &Path) -> PathBuf {
    ex::dir(&memory(home)).join(ex::DISTIL)
}

static WRITE: Mutex<()> = Mutex::new(());

fn write_rows(home: &Path, rows: &[DistilRow]) -> Result<(), String> {
    let f = store(home);
    if let Some(d) = f.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let mut t = String::new();
    for r in rows {
        t.push_str(&r.to_value().encode().map_err(|e| e.to_string())?);
        t.push('\n');
    }
    let tmp = f.with_extension("jsonl.tmp");
    std::fs::write(&tmp, t).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &f).map_err(|e| format!("{}: {e}", f.display()))
}

/// The set name an import takes: the one asked for, else the file's stem, as letters, digits and
/// `-` (`nmapi::distil::slug`).
fn set_name(asked: Option<&str>, file: &Path) -> String {
    nmapi::distil::slug(asked.filter(|x| !x.trim().is_empty()).unwrap_or_else(|| file.file_stem().and_then(|x| x.to_str()).unwrap_or("set")))
}

/// Imports curated teacher pairs from `file` into set `set` (the file's name by default); audited,
/// undone by removing the rows it added. Returns (set, rows added, rows passed over, reference id,
/// audit id).
/// # Errors
/// An unreadable file, a row without prompt or completion, a row without a licence when the import
/// names none, a row over the size limits, or the store not writable.
pub fn import(home: &Path, file: &Path, set: Option<&str>, licence: Option<&str>, teacher: Option<&str>, by: &str) -> Result<Value, String> {
    let set = set_name(set, file);
    let body = std::fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?;
    let _g = WRITE.lock().map_err(|_| "distil store lock poisoned".to_owned())?;
    let mut rows = ex::read_distil(&memory(home))?;
    let have: BTreeSet<String> = rows.iter().map(|r| r.id.clone()).collect();
    let now = nmtype::now_ms();
    let mut fresh: Vec<DistilRow> = Vec::new();
    let mut passed = 0_u64;
    for (i, line) in body.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let v = nmjson::parse(line.trim().as_bytes()).map_err(|e| format!("line {}: not JSON ({e})", i + 1))?;
        let prompt = text(&v, "prompt");
        let completion = text(&v, "completion");
        if prompt.trim().is_empty() || completion.trim().is_empty() {
            return Err(format!("line {}: a pair needs a prompt and a completion", i + 1));
        }
        if prompt.chars().count() > ex::DISTIL_PROMPT_MAX || completion.chars().count() > ex::DISTIL_COMPLETION_MAX {
            return Err(format!("line {}: longer than {} characters of prompt or {} of completion", i + 1, ex::DISTIL_PROMPT_MAX, ex::DISTIL_COMPLETION_MAX));
        }
        let lic = Some(text(&v, "licence")).filter(|x| !x.trim().is_empty()).or_else(|| licence.map(str::to_owned)).filter(|x| !x.trim().is_empty());
        let Some(lic) = lic else { return Err(format!("line {}: no licence; name the teacher outputs' licence for the import (for example Apache-2.0, or own)", i + 1)) };
        let tch = Some(text(&v, "teacher")).filter(|x| !x.trim().is_empty()).or_else(|| teacher.map(str::to_owned)).unwrap_or_else(|| "not stated".to_owned());
        let mut row = DistilRow::from_value(&v).ok_or_else(|| format!("line {}: not a teacher pair", i + 1))?;
        row.id = ex::distil_id(&prompt, &completion, &tch);
        if have.contains(&row.id) || fresh.iter().any(|r| r.id == row.id) {
            passed += 1;
            continue;
        }
        row.set.clone_from(&set);
        row.teacher = tch;
        row.licence = lic;
        row.split = ex::distil_split(&prompt);
        row.outcome = "used".into();
        row.at = now;
        fresh.push(row);
    }
    let teachers: BTreeSet<String> = fresh.iter().map(|r| r.teacher.clone()).collect();
    let licences: BTreeSet<String> = fresh.iter().map(|r| r.licence.clone()).collect();
    let reference = crate::examples::reference_add(
        home,
        file,
        &licences.iter().cloned().collect::<Vec<_>>().join("; "),
        &format!("teacher pairs: {}", teachers.iter().cloned().collect::<Vec<_>>().join(", ")),
        &format!("distil:{set}"),
    )?;
    for r in &mut fresh {
        r.reference.clone_from(&reference);
    }
    let ids: Vec<String> = fresh.iter().map(|r| r.id.clone()).collect();
    let added = fresh.len() as u64;
    rows.extend(fresh);
    write_rows(home, &rows)?;
    let audit = crate::controls::audit(
        home,
        "distil",
        vec![
            ("action", s("import")),
            ("set", s(&set)),
            ("file", s(&file.display().to_string())),
            ("rows", n(added)),
            ("passed_over", n(passed)),
            ("reference", s(&reference)),
            ("ids", strs(ids)),
            ("by", s(by)),
            ("label", s(&format!("Teacher pairs imported into set {set}: {added} rows ({passed} already here)"))),
        ],
    )?;
    Ok(obj(vec![("set", s(&set)), ("rows", n(added)), ("passed_over", n(passed)), ("reference", s(&reference)), ("audit_id", s(&audit))]))
}

/// Labels rows (`ids`, else every row of `set`) with an outcome of `nmapi::lineage::OUTCOMES`;
/// audited, undone by putting each row's earlier label back.
/// # Errors
/// An outcome outside the list, nothing named, no row matching, or the store not writable.
pub fn label(home: &Path, ids: &[String], set: Option<&str>, outcome: &str, why: &str, by: &str) -> Result<Value, String> {
    if !nmapi::lineage::OUTCOMES.contains(&outcome) {
        return Err(format!("outcome {outcome}: use one of {}", nmapi::lineage::OUTCOMES.join(", ")));
    }
    if ids.is_empty() && set.is_none_or(str::is_empty) {
        return Err("name the rows (ids) or a set".into());
    }
    let _g = WRITE.lock().map_err(|_| "distil store lock poisoned".to_owned())?;
    let mut rows = ex::read_distil(&memory(home))?;
    let mut before: BTreeMap<String, String> = BTreeMap::new();
    for r in rows.iter_mut().filter(|r| if ids.is_empty() { set == Some(r.set.as_str()) } else { ids.contains(&r.id) }) {
        before.insert(r.id.clone(), r.outcome.clone());
        r.outcome = outcome.to_owned();
    }
    if before.is_empty() {
        return Err("no teacher pair matches".into());
    }
    write_rows(home, &rows)?;
    let count = before.len() as u64;
    let audit = crate::controls::audit(
        home,
        "distil",
        vec![
            ("action", s("label")),
            ("outcome", s(outcome)),
            ("why", s(why)),
            ("before", Value::Object(before.into_iter().map(|(k, v)| (k, s(&v))).collect())),
            ("by", s(by)),
            ("label", s(&format!("{count} teacher pairs labelled {outcome}{}", if why.is_empty() { String::new() } else { format!(": {why}") }))),
        ],
    )?;
    Ok(obj(vec![("rows", n(count)), ("outcome", s(outcome)), ("audit_id", s(&audit))]))
}

/// Undoes a `distil` audit row: an import's rows are taken out again; a label's rows get their
/// earlier labels back. Returns the undo's audit id.
/// # Errors
/// Not an import or a label row, or the store not writable.
pub fn undo(home: &Path, row: &Value, id: &str, by: &str) -> Result<Option<String>, String> {
    let action = text(row, "action");
    let _g = WRITE.lock().map_err(|_| "distil store lock poisoned".to_owned())?;
    let mut rows = ex::read_distil(&memory(home))?;
    let what = match action.as_str() {
        "import" => {
            let ids: BTreeSet<String> = row.get("ids").and_then(Value::array).unwrap_or_default().iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect();
            let before = rows.len();
            rows.retain(|r| !ids.contains(&r.id));
            format!("{} imported teacher pairs taken out again", before - rows.len())
        }
        "label" => {
            let prior: BTreeMap<String, String> = match row.get("before") {
                Ok(Value::Object(m)) => m.iter().map(|(k, v)| (k.clone(), v.string().unwrap_or_default().to_owned())).collect(),
                _ => BTreeMap::new(),
            };
            for r in rows.iter_mut() {
                if let Some(o) = prior.get(&r.id) {
                    r.outcome.clone_from(o);
                }
            }
            format!("{} teacher pairs given their earlier labels back", prior.len())
        }
        "values" => {
            let file = PathBuf::from(text(row, "file"));
            let cart = text(row, "cartridge");
            if nmapi::distil::role_of(&cart).is_none() || nmapi::adapter::versions(home, &cart).last().map(|(_, f)| f) != Some(&file) {
                return Err(format!("{id}: the version it wrote is no longer the newest of {cart}; remove versions on Cartridges"));
            }
            std::fs::remove_file(&file).map_err(|e| format!("{}: {e}", file.display()))?;
            format!("{} taken out again (the edit is undone)", file.display())
        }
        other => return Err(format!("{id}: a distil {other} row is not undone here")),
    };
    write_rows(home, &rows)?;
    crate::controls::audit(home, "distil", vec![("action", s("undo")), ("undo_of", s(id)), ("by", s(by)), ("label", s(&what))]).map(Some)
}

/// `GET /api/training/distil[?set=<name>&limit=<n>]`: the teacher data by set (rows, splits,
/// outcomes, teachers, licences, pairs with teacher log-probabilities, healthy rows), a page of
/// rows with short previews, the techniques and roles the builder takes, and the distilled
/// cartridges on the shelf with their gate figures.
#[must_use]
pub fn view(home: &Path, query: &str) -> Value {
    let want = crate::project::query_arg(query, "set").filter(|x| !x.is_empty());
    let limit = crate::project::query_arg(query, "limit").and_then(|x| x.parse::<usize>().ok()).unwrap_or(50).min(500);
    let rows = ex::read_distil(&memory(home)).unwrap_or_default();
    let mut sets: BTreeMap<String, Vec<&DistilRow>> = BTreeMap::new();
    for r in &rows {
        sets.entry(r.set.clone()).or_default().push(r);
    }
    let count = |rs: &[&DistilRow], f: &dyn Fn(&DistilRow) -> String| -> Value {
        let mut m: BTreeMap<String, u64> = BTreeMap::new();
        for r in rs {
            *m.entry(f(r)).or_default() += 1;
        }
        Value::Object(m.into_iter().map(|(k, v)| (k, n(v))).collect())
    };
    let set_rows: Vec<Value> = sets
        .iter()
        .map(|(name, rs)| {
            obj(vec![
                ("set", s(name)),
                ("rows", n(rs.len() as u64)),
                ("healthy", n(rs.iter().filter(|r| r.healthy()).count() as u64)),
                ("splits", count(rs, &|r| r.split.name().to_owned())),
                ("outcomes", count(rs, &|r| if r.outcome.is_empty() { "not stated".to_owned() } else { r.outcome.clone() })),
                ("teachers", strs(rs.iter().map(|r| r.teacher.clone()).collect::<BTreeSet<_>>())),
                ("licences", strs(rs.iter().map(|r| r.licence.clone()).collect::<BTreeSet<_>>())),
                ("with_topk", n(rs.iter().filter(|r| !r.topk.is_empty()).count() as u64)),
                ("with_score", n(rs.iter().filter(|r| r.score.is_some()).count() as u64)),
                ("cartridge", s(&nmapi::distil::cartridge_id("writer", name))),
            ])
        })
        .collect();
    let preview = |t: &str| -> String {
        let one: String = t.split_whitespace().collect::<Vec<_>>().join(" ");
        if one.chars().count() > 140 { format!("{}...", one.chars().take(140).collect::<String>()) } else { one }
    };
    let page: Vec<Value> = rows
        .iter()
        .filter(|r| want.as_deref().is_none_or(|w| r.set == w))
        .rev()
        .take(limit)
        .map(|r| {
            obj(vec![
                ("id", s(&r.id)),
                ("set", s(&r.set)),
                ("split", s(r.split.name())),
                ("outcome", s(&r.outcome)),
                ("teacher", s(&r.teacher)),
                ("licence", s(&r.licence)),
                ("topk", Value::Bool(!r.topk.is_empty())),
                ("score", r.score.map_or(Value::Null, |x| Value::Number(format!("{x:.4}")))),
                ("prompt", s(&preview(&r.prompt))),
                ("completion", s(&preview(&r.completion))),
            ])
        })
        .collect();
    obj(vec![
        ("file", s(&store(home).display().to_string())),
        ("rows_total", n(rows.len() as u64)),
        ("sets", Value::Array(set_rows)),
        ("rows", Value::Array(page)),
        ("outcomes", strs(nmapi::lineage::OUTCOMES.iter().map(|x| (*x).to_owned()))),
        ("techniques", strs(nmapi::distil::TECHNIQUES.iter().map(|x| (*x).to_owned()))),
        ("targets", strs(nmapi::distil::TARGETS.iter().map(|x| (*x).to_owned()))),
        ("fills", Value::Array(nmapi::distil::FILLS.iter().map(|(k, t)| obj(vec![("target", s(k)), ("technique", s(t))])).collect())),
        // A preference cartridge holds engine-wide values, so it is built for every role.
        (
            "built",
            Value::Array(
                nmapi::distil::BUILT
                    .iter()
                    .copied()
                    .chain(nmapi::distil::ROLES.iter().map(|r| (*r, "preference", "readout")))
                    .map(|(r, k, t)| obj(vec![("role", s(r)), ("target", s(k)), ("technique", s(t))]))
                    .collect(),
            ),
        ),
        ("roles", strs(nmapi::distil::ROLES.iter().map(|x| (*x).to_owned()))),
        // Every allowed pair that is not built, with what it still needs (`not_built`).
        (
            "unavailable",
            Value::Array(
                nmapi::distil::ROLES
                    .iter()
                    .flat_map(|r| nmapi::distil::FILLS.iter().map(move |(k, t)| (*r, *k, *t)))
                    .filter_map(|(r, k, t)| nmapi::distil::not_built(r, k, t).map(|why| obj(vec![("role", s(r)), ("target", s(k)), ("technique", s(t)), ("why", s(&why))])))
                    .collect(),
            ),
        ),
        ("cartridges", cartridges(home)),
        ("gate", s("held-out split: teacher-token loss falls and top-1 rises; a shuffled-target control gains at most a quarter of that; neutral drift within 0.10 nats per token; the note check not worse (the Writer); stable over three repeated trials")),
        ("gate_scores", s("the Reranker, from teacher scores, on the held-out split: relative entropy from the teacher's scores falls and verdict agreement does not fall; a shuffled-target control gains at most a quarter of that; neutral drift within 0.05 nats per pair; stable over three repeated trials")),
        ("gate_similarity", s("the Embedder, from teacher similarity scores, on the held-out split: the error of its cosines from the teacher's scores (on its own cosine scale) falls and same-query order agreement does not fall; a shuffled-target control gains at most a quarter of that; neutral requests move by at most 0.02 in cosine distance; stable over three repeated trials; only request vectors are mapped")),
    ])
}

/// The distilled cartridges on the adapter shelf, every role: id, role, versions, the newest
/// label's technique, set, base, gate figures and measured line, and whether it is plugged in.
#[must_use]
pub fn cartridges(home: &Path) -> Value {
    let plugs = nmapi::plugs::read_now(home);
    let mut out = Vec::new();
    for role in nmapi::distil::ROLES {
        for (id, file, ms) in nmapi::distil::on_shelf(home, role) {
            let label = nmlearn::library::label_of(&file).unwrap_or(Value::Null);
            let notes = label.get("notes").cloned().unwrap_or(Value::Null);
            let versions = nmapi::adapter::versions(home, &id);
            out.push(obj(vec![
                ("id", s(&id)),
                ("key", s(&nmapi::plugs::key(nmapi::plugs::NANO, &id))),
                ("role", s(role)),
                ("model", s(&text(&label, "model"))),
                ("kind", s(&text(&label, "kind"))),
                ("version", n(versions.last().map_or(0, |(v, _)| *v))),
                ("versions", n(versions.len() as u64)),
                ("technique", s(&text(&notes, "technique"))),
                ("set", s(&text(&notes, "set"))),
                ("base", s(&text(&notes, "base"))),
                ("weights_sha256", s(&text(&label, "weights"))),
                ("socket", notes.get("socket").cloned().unwrap_or(Value::Null)),
                ("scale", notes.get("scale").cloned().unwrap_or(Value::Null)),
                ("gate", notes.get("gate").cloned().unwrap_or(Value::Null)),
                ("measured", s(&text(&notes, "measured"))),
                ("teachers", notes.get("teachers").cloned().unwrap_or(Value::Array(Vec::new()))),
                ("made", s(&text(&label, "made"))),
                ("modified_ms", n(ms)),
                ("plugged", Value::Bool(plugs.acts(nmapi::plugs::NANO, &id))),
                ("path", s(&file.display().to_string())),
            ]));
        }
    }
    Value::Array(out)
}

/// A distilled cartridge's id and version file from a key (`id` for the newest, or `id@version`);
/// None when the key names no distilled cartridge on the shelf.
fn find(home: &Path, key: &str) -> Option<(String, u64, PathBuf)> {
    let (id, ver) = key.split_once('@').map_or((key, None), |(i, v)| (i, v.parse::<u64>().ok()));
    nmapi::distil::role_of(id)?;
    let versions = nmapi::adapter::versions(home, id);
    let (v, f) = match ver {
        Some(w) => versions.into_iter().find(|(v, _)| *v == w)?,
        None => versions.into_iter().last()?,
    };
    Some((id.to_owned(), v, f))
}

fn stat(xs: &[f32]) -> Value {
    let norm = xs.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
    let mean = xs.iter().map(|x| f64::from(x.abs())).sum::<f64>() / xs.len().max(1) as f64;
    let max = xs.iter().map(|x| f64::from(x.abs())).fold(0.0, f64::max);
    obj(vec![("norm", Value::Number(format!("{norm:.6}"))), ("mean_abs", Value::Number(format!("{mean:.6}"))), ("max_abs", Value::Number(format!("{max:.6}")))])
}

/// `GET /api/cartridges/inspect?key=<id>[@version]` for a distilled cartridge: its label (model,
/// open weights, base, socket, technique, teacher data, gate figures, lineage), its tensors with
/// shapes and value statistics (norm, mean and largest magnitude, the norm of each direction of the
/// map), and its editable values (`scale`, and `rank`: how many leading directions act). None when
/// the key is not a distilled cartridge.
#[must_use]
pub fn inspect(home: &Path, key: &str) -> Option<Value> {
    let (id, version, file) = find(home, key)?;
    let (label, parts) = match nmapi::adapter::read_file(&file) {
        Ok(x) => x,
        Err(e) => return Some(obj(vec![("key", s(&format!("{id}@{version}"))), ("source", s("distil")), ("error", s(&e.message))])),
    };
    // The `parameters` section in the shape the cartridge editor reads for low-rank factors: name,
    // target node, layer, shape, dtype, size, L2 and largest magnitude, and a value summary.
    let target = label.get("notes").ok().and_then(|nt| nt.get("socket").ok()).map(|sk| text(sk, "node")).unwrap_or_default();
    let tensors: Vec<Value> = parts
        .iter()
        .map(|(name, shape, xs)| {
            let st = stat(xs);
            let mut fields = vec![
                ("name", s(name)),
                ("target", s(&target)),
                ("layer", s("final norm (after the last layer)")),
                ("shape", Value::Array(shape.iter().map(|x| n(*x as u64)).collect())),
                ("dtype", s("F16")),
                ("size", n(xs.len() as u64)),
                ("l2", st.get("norm").cloned().unwrap_or(Value::Null)),
                ("max_abs", st.get("max_abs").cloned().unwrap_or(Value::Null)),
                ("summary", st),
            ];
            if name == "down"
                && let Some(w) = shape.get(1).copied().filter(|w| *w > 0)
            {
                fields.push(("direction_norms", Value::Array(xs.chunks(w).map(|r| Value::Number(format!("{:.6}", r.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt()))).collect())));
            }
            obj(fields)
        })
        .collect();
    let notes = label.get("notes").cloned().unwrap_or(Value::Null);
    let rank = parts.iter().find(|p| p.0 == "down").and_then(|p| p.1.first().copied()).unwrap_or(0);
    let scale = obj(vec![("key", s("scale")), ("value", notes.get("scale").cloned().unwrap_or(Value::Null)), ("code_value", s("1")), ("kind", s("strength of the map: 0 adds nothing, 1 as fitted"))]);
    // A judge read-out acts along one direction: its strength is its only value.
    let values = if [nmapi::adapter::READOUT_KIND, nmapi::distil::STEER_KIND].contains(&text(&label, "kind").as_str()) {
        Value::Array(vec![scale])
    } else {
        Value::Array(vec![scale, obj(vec![("key", s("rank")), ("value", n(rank as u64)), ("code_value", n(rank as u64)), ("kind", s("leading directions that act (at most the fitted rank)"))])])
    };
    Some(obj(vec![
        ("key", s(&format!("{id}@{version}"))),
        ("source", s("distil")),
        ("cartridge", label),
        ("parameters", Value::Array(tensors)),
        ("versions", n(nmapi::adapter::versions(home, &id).len() as u64)),
        ("plugged", Value::Bool(nmapi::plugs::read_now(home).acts(nmapi::plugs::NANO, &id))),
        ("misfit", crate::library::distil_misfit(home, &id).map_or(Value::Null, |w| s(&w))),
        ("values", values),
        ("values_editable", Value::Bool(true)),
        ("note", s("an edit writes a new version of the cartridge (the file is never written over), keeps the gate figures of the version it came from and says it was not gated itself; Audit undoes it")),
    ]))
}

fn number(v: Option<&Value>, k: &str) -> Result<Option<f64>, String> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(x) | Value::String(x)) => x.trim().parse::<f64>().map(Some).map_err(|_| format!("{k} {x}: a number")),
        Some(_) => Err(format!("{k}: a number")),
    }
}

/// `POST /api/cartridges/values {key, values: {scale?, rank?}}` for a distilled cartridge: a new
/// version with the map's strength changed and/or only its leading directions kept. Audited; undo
/// removes the version it wrote. None when the key is not a distilled cartridge.
/// # Errors
/// A value that does not read, a rank above the fitted one, or a write failure.
pub fn set_values(home: &Path, key: &str, changes: &BTreeMap<String, Value>, by: &str) -> Option<Result<Value, String>> {
    let (id, version, file) = find(home, key)?;
    Some(edit_values(home, &id, version, &file, changes, by))
}

fn edit_values(home: &Path, id: &str, version: u64, file: &Path, changes: &BTreeMap<String, Value>, by: &str) -> Result<Value, String> {
    let (label, parts) = nmapi::adapter::read_file(file).map_err(|e| e.message)?;
    if let Some(k) = changes.keys().find(|k| !matches!(k.as_str(), "scale" | "rank")) {
        return Err(format!("{k}: a distilled cartridge's values are scale and rank"));
    }
    let mut notes = label.get("notes").cloned().unwrap_or(Value::Null);
    let old_scale = number(notes.get("scale").ok(), "scale").ok().flatten().unwrap_or(1.0);
    let scale = number(changes.get("scale"), "scale")?.unwrap_or(old_scale);
    if !(0.0..=4.0).contains(&scale) {
        return Err(format!("scale {scale}: from 0 to 4"));
    }
    let kind = text(&label, "kind");
    if kind == nmapi::adapter::READOUT_KIND || kind == nmapi::distil::STEER_KIND {
        return edit_readout(home, id, version, &label, &parts, notes, old_scale, scale, changes, by);
    }
    // A request lens holds its strength in `up` (the serving lens has no scale of its own).
    let lens_gain = if kind == nmapi::adapter::LENS_KIND && old_scale > 0.0 { scale / old_scale } else { 1.0 };
    let dn = parts.iter().find(|p| p.0 == "down").ok_or("no down part")?;
    let up = parts.iter().find(|p| p.0 == "up").ok_or("no up part")?;
    let (r0, d) = (dn.1.first().copied().unwrap_or(0), dn.1.get(1).copied().unwrap_or(0));
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "a whole rank given as a number")]
    let r = number(changes.get("rank"), "rank")?.map_or(r0, |x| x.max(1.0) as usize);
    if r > r0 {
        return Err(format!("rank {r}: at most the fitted rank {r0}"));
    }
    let down: Vec<f32> = dn.2.iter().take(r * d).copied().collect();
    #[expect(clippy::cast_possible_truncation, reason = "parameters kept at 16 bits")]
    let upk: Vec<f32> = (0..d).flat_map(|m| up.2.get(m * r0..m * r0 + r).unwrap_or(&[]).iter().map(|x| (f64::from(*x) * lens_gain) as f32)).collect();
    let edit = format!("edited from {id}@{version}: scale {old_scale} to {scale}, rank {r0} to {r}; this version was not gated itself");
    if let Value::Object(o) = &mut notes {
        o.insert("scale".into(), Value::Number(format!("{scale}")));
        o.insert("edited".into(), s(&edit));
        if let Some(Value::Object(sock)) = o.get_mut("socket") {
            sock.insert("rank".into(), n(r as u64));
            sock.insert("down".into(), s(&format!("[{r}, {d}]")));
            sock.insert("up".into(), s(&format!("[{d}, {r}]")));
        }
    }
    let (model, weights, does, made_from) = (text(&label, "model"), text(&label, "weights"), text(&label, "does"), text(&label, "made_from"));
    let l = nmapi::adapter_fit::LabelIn { id, version: 0, model: &model, weights: &weights, kind: if kind == nmapi::adapter::LENS_KIND { nmapi::adapter::LENS_KIND } else { nmapi::distil::KIND }, does: &does, made_from: &made_from, notes };
    let new_parts = vec![("down".to_owned(), vec![r, d], down), ("up".to_owned(), vec![d, r], upk)];
    let (path, v) = nmapi::adapter_fit::write_next(home, l, &new_parts).map_err(|e| e.message)?;
    let audit = crate::controls::audit(
        home,
        "distil",
        vec![("action", s("values")), ("cartridge", s(id)), ("version", n(v)), ("from", s(&format!("{id}@{version}"))), ("file", s(&path.display().to_string())), ("by", s(by)), ("label", s(&format!("{id}@{v} written: {edit}")))],
    )?;
    Ok(obj(vec![("key", s(&format!("{id}@{v}"))), ("audit_id", s(&audit)), ("note", s(&edit)), ("misfit", crate::library::distil_misfit(home, id).map_or(Value::Null, |w| s(&w)))]))
}

/// A new version of a distilled judge read-out with its strength changed (its `alpha`, and `scale`
/// beside it); the read-out vector and its direction are kept.
#[expect(clippy::too_many_arguments, reason = "an edit is its cartridge, its version, its label and parts, both strengths, the changes and who asked")]
fn edit_readout(home: &Path, id: &str, version: u64, label: &Value, parts: &[nmapi::adapter::Part], mut notes: Value, old_scale: f64, scale: f64, changes: &BTreeMap<String, Value>, by: &str) -> Result<Value, String> {
    if changes.contains_key("rank") {
        return Err("rank: a judge read-out acts along one direction; its value is scale".into());
    }
    let edit = format!("edited from {id}@{version}: scale {old_scale} to {scale}; this version was not gated itself");
    if let Value::Object(o) = &mut notes {
        o.insert("scale".into(), Value::Number(format!("{scale}")));
        o.insert("alpha".into(), Value::Number(format!("{scale}")));
        o.insert("edited".into(), s(&edit));
    }
    let (model, weights, does, made_from) = (text(label, "model"), text(label, "weights"), text(label, "does"), text(label, "made_from"));
    let kind = text(label, "kind");
    let l = nmapi::adapter_fit::LabelIn { id, version: 0, model: &model, weights: &weights, kind: &kind, does: &does, made_from: &made_from, notes };
    let (path, v) = nmapi::adapter_fit::write_next(home, l, parts).map_err(|e| e.message)?;
    let audit = crate::controls::audit(
        home,
        "distil",
        vec![("action", s("values")), ("cartridge", s(id)), ("version", n(v)), ("from", s(&format!("{id}@{version}"))), ("file", s(&path.display().to_string())), ("by", s(by)), ("label", s(&format!("{id}@{v} written: {edit}")))],
    )?;
    Ok(obj(vec![("key", s(&format!("{id}@{v}"))), ("audit_id", s(&audit)), ("note", s(&edit)), ("misfit", crate::library::distil_misfit(home, id).map_or(Value::Null, |w| s(&w)))]))
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
    fn import_label_and_undo_round_trip() {
        let home = Home(std::env::temp_dir().join(format!("nm-distil-{}-{}", std::process::id(), nmtype::now_ms())));
        std::fs::create_dir_all(home.0.join("memory")).unwrap();
        let file = home.0.join("pairs.jsonl");
        std::fs::write(&file, "{\"prompt\":\"Say hi\",\"completion\":\"Hi.\",\"teacher\":\"t\"}\n{\"prompt\":\"Say bye\",\"completion\":\"Bye.\",\"teacher\":\"t\",\"licence\":\"own\"}\n").unwrap();
        assert!(import(&home.0, &file, Some("Greetings"), None, None, "test").is_err(), "a row without a licence is refused");
        let r = import(&home.0, &file, Some("Greetings"), Some("Apache-2.0"), None, "test").unwrap();
        assert_eq!(text(&r, "set"), "greetings");
        assert_eq!(r.get("rows").and_then(Value::integer).unwrap(), 2);
        let again = import(&home.0, &file, Some("Greetings"), Some("Apache-2.0"), None, "test").unwrap();
        assert_eq!(again.get("rows").and_then(Value::integer).unwrap(), 0, "nothing is added twice");
        let rows = ex::read_distil(&home.0.join("memory")).unwrap();
        assert!(rows.iter().all(|r| r.healthy() && r.set == "greetings" && !r.reference.is_empty()));
        let l = label(&home.0, &[], Some("greetings"), "unused", "a test", "test").unwrap();
        assert!(ex::read_distil(&home.0.join("memory")).unwrap().iter().all(|r| !r.healthy()));
        assert!(label(&home.0, &[], Some("greetings"), "great", "", "test").is_err());
        let log = std::fs::read_to_string(home.0.join("memory/audit.jsonl")).unwrap();
        let find = |id: &str| log.lines().filter_map(|x| nmjson::parse(x.as_bytes()).ok()).find(|v| text(v, "id") == id).unwrap();
        undo(&home.0, &find(&text(&l, "audit_id")), &text(&l, "audit_id"), "test").unwrap();
        assert!(ex::read_distil(&home.0.join("memory")).unwrap().iter().all(DistilRow::healthy));
        undo(&home.0, &find(&text(&r, "audit_id")), &text(&r, "audit_id"), "test").unwrap();
        assert!(ex::read_distil(&home.0.join("memory")).unwrap().is_empty());
        let v = view(&home.0, "");
        assert_eq!(v.get("rows_total").and_then(Value::integer).unwrap(), 0);
        // What is not built says why; what is built is not listed as unavailable.
        let un = v.get("unavailable").and_then(Value::array).unwrap();
        assert!(un.iter().any(|r| text(r, "role") == "embedder" && !text(r, "why").is_empty()));
        assert!(!un.iter().any(|r| text(r, "role") == "reranker" && text(r, "target") == "readout"));
        // A scored pair keeps its teacher score through the import.
        std::fs::write(&file, "{\"prompt\":\"where is run\",\"completion\":\"fn run() {}\",\"teacher\":\"t\",\"licence\":\"own\",\"score\":0.9}\n").unwrap();
        import(&home.0, &file, Some("judged"), None, None, "test").unwrap();
        let rows = ex::read_distil(&home.0.join("memory")).unwrap();
        assert_eq!(rows.iter().find(|r| r.set == "judged").and_then(|r| r.score), Some(0.9));
    }
}
