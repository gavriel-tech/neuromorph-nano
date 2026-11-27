//! Every parameter map, for the interface and MCP (2026-11-25, directions of Luis Jake Gabriel
//! III: the interface shows and controls the entire parameter mapping, including adding and editing
//! maps).
//!
//! `library/parameter-maps/<role>/` holds each MicroModel's maps: the parameter maps a base
//! cartridge is built from (`pmap-*`, per job, layer and socket), the activation maps (`act-*`,
//! channels left out per layer), the separation maps (`zsep-map-*`), the tolerances (`tol-*`), the
//! block map (`map.json`), the role weights (`role.json`), the energy record (`record.json`), the
//! pinned super-weights, the surgery probe and its ledger, and the job filters. Each is listed with
//! its kind, what it covers (jobs, layers, sockets), its size and date; the summaries are cached by
//! file size and time, so a 55 MB map is read once. Job filters are added (built through the
//! held-out split gate), plugged in or out and removed through the surgery routes; any other map
//! can be removed here (the base cartridges already built from it keep serving; building them again
//! needs it).

use nmjson::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn text(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default()
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}
fn obj(f: Vec<(&str, Value)>) -> Value {
    Value::Object(f.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

/// The kind of a map file, by its name, with the words the interface shows.
fn kind_of(name: &str) -> (&'static str, &'static str) {
    if name.starts_with("job-filter-") {
        ("job filter", "a per-hop filter over some layers for one job; plugs in and out live")
    } else if name.starts_with("pmap-") {
        ("parameter map", "per job, layer and socket: how much each part of the weights matters; the base cartridge is cut from it")
    } else if name.starts_with("act-") {
        ("activation map", "the channels left out in each layer, measured on the job's activations")
    } else if name.starts_with("zsep-map") {
        ("separation map", "how well each layer's values separate right from wrong answers")
    } else if name.starts_with("tol-") {
        ("tolerance", "how much each layer can be cut before its outputs change")
    } else if name == "map.json" {
        ("block map", "the blocks of each matrix kept at each budget")
    } else if name == "role.json" {
        ("role weights", "the weight of each part for this MicroModel's role")
    } else if name == "record.json" {
        ("energy record", "the measured energy of each part over the job's inputs")
    } else if name == "super-weights.json" {
        ("super-weights", "the few weights pinned so no cut ever touches them")
    } else if name == "surgery-probe.json" {
        ("surgery probe", "layer by layer drift of the base cartridge against the open weights")
    } else if name == "surgery-ledger.jsonl" {
        ("surgery ledger", "every surgical change made to this MicroModel's base cartridge")
    } else {
        ("other", "a file of this MicroModel's parameter mapping")
    }
}

/// Layer keys (`L0`, `L12`, or bare numbers) of an object, as numbers.
fn layers_of(v: &Value) -> Vec<u64> {
    let mut out: Vec<u64> = v.object().map(|o| o.keys().filter_map(|k| k.trim_start_matches('L').parse::<u64>().ok()).collect()).unwrap_or_default();
    out.sort_unstable();
    out.dedup();
    out
}

/// What one file covers: jobs, layers and sockets, as far as its shape says.
/// The object keys of a JSON document by depth, read in one pass over its bytes without building
/// the document (2026-11-25: a 55 MB map parsed whole took gigabytes, and with a serving engine
/// beside it the interface process ran out of memory). Also counts the values of arrays at
/// `count_depth` (array elements, not keys). Keys longer than 64 bytes are skipped.
fn scan_keys(bytes: &[u8], max_depth: usize, count_depth: usize) -> (Vec<Vec<String>>, u64) {
    let mut keys: Vec<Vec<String>> = vec![Vec::new(); max_depth + 1];
    let mut stack: Vec<u8> = Vec::new();
    let mut counted = 0_u64;
    let mut i = 0;
    let mut expect_key = false;
    let mut seen_value = false;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            b'{' => {
                stack.push(b'{');
                expect_key = true;
                seen_value = false;
            }
            b'[' => {
                stack.push(b'[');
                seen_value = false;
            }
            b'}' | b']' => {
                if c == b']' && stack.len() == count_depth && seen_value {
                    counted += 1;
                }
                stack.pop();
                expect_key = false;
                seen_value = true;
            }
            b',' => {
                if stack.last() == Some(&b'[') && stack.len() == count_depth {
                    counted += 1;
                }
                expect_key = stack.last() == Some(&b'{');
            }
            b'"' => {
                let start = i + 1;
                let mut j = start;
                while j < bytes.len() && bytes[j] != b'"' {
                    j += if bytes[j] == b'\\' { 2 } else { 1 };
                }
                if expect_key && stack.last() == Some(&b'{') {
                    let depth = stack.len();
                    if depth <= max_depth && j - start <= 64 && keys[depth].len() < 4096 {
                        let k = String::from_utf8_lossy(&bytes[start..j.min(bytes.len())]).into_owned();
                        if !keys[depth].contains(&k) {
                            keys[depth].push(k);
                        }
                    }
                    expect_key = false;
                } else {
                    seen_value = true;
                }
                i = j;
            }
            b' ' | b'\n' | b'\r' | b'\t' | b':' => {}
            _ => seen_value = true,
        }
        i += 1;
    }
    (keys, counted)
}

/// A summary of a large map from its keys alone (`scan_keys`), in the same shape as `summary`.
fn summary_scanned(path: &Path, name: &str) -> Value {
    let Ok(bytes) = std::fs::read(path) else { return obj(vec![("unreadable", Value::Bool(true))]) };
    let layers_in = |ks: &[String]| -> Vec<u64> {
        let mut v: Vec<u64> = ks.iter().filter_map(|k| k.trim_start_matches('L').parse::<u64>().ok()).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let arr = |v: &[u64]| Value::Array(v.iter().map(|l| n(*l)).collect());
    if name.starts_with("pmap-") {
        let (k, _) = scan_keys(&bytes, 5, 0);
        return obj(vec![("jobs", Value::Array(k[2].iter().map(|j| s(j)).collect())), ("layers", arr(&layers_in(&k[4]))), ("sockets", Value::Array(k[5].iter().map(|x| s(x)).collect()))]);
    }
    if name.starts_with("act-") {
        let (k, channels) = scan_keys(&bytes, 2, 3);
        return obj(vec![("layers", arr(&layers_in(&k[2]))), ("channels_off", n(channels))]);
    }
    let (k, _) = scan_keys(&bytes, 2, 0);
    let layers = layers_in(&k[1]);
    if !layers.is_empty() {
        return obj(vec![("layers", arr(&layers))]);
    }
    obj(vec![("entries", n(k[1].len() as u64))])
}

/// Files above this size are summarised from their keys (`summary_scanned`), never parsed whole.
const PARSE_LIMIT: u64 = 2 * 1024 * 1024;

fn summary(path: &Path, name: &str) -> Value {
    if !name.ends_with(".jsonl") && !name.starts_with("job-filter-") && std::fs::metadata(path).is_ok_and(|m| m.len() > PARSE_LIMIT) {
        return summary_scanned(path, name);
    }
    if name.ends_with(".jsonl") {
        let rows = std::fs::read_to_string(path).map_or(0, |t| t.lines().filter(|l| !l.trim().is_empty()).count());
        return obj(vec![("rows", n(rows as u64))]);
    }
    let Some(v) = std::fs::read(path).ok().and_then(|b| nmjson::parse(&b).ok()) else { return obj(vec![("unreadable", Value::Bool(true))]) };
    if name.starts_with("job-filter-") {
        let text = |k: &str| v.get(k).ok().and_then(|x| x.string().ok()).unwrap_or("").to_owned();
        let num = |k: &str| match v.get(k) {
            Ok(Value::Number(x)) => x.parse::<f64>().ok(),
            _ => None,
        };
        return obj(vec![
            ("job", s(&text("job"))),
            ("cartridge", s(&text("cartridge"))),
            ("model", s(&text("model"))),
            ("base", s(&text("base"))),
            ("parts", n(v.get("parts").and_then(Value::array).map_or(0, |a| a.len() as u64))),
            ("cos_before", num("base_cos").map_or(Value::Null, |x| s(&format!("{x:.4}")))),
            ("cos_after", num("filtered_cos").map_or(Value::Null, |x| s(&format!("{x:.4}")))),
        ]);
    }
    if name.starts_with("pmap-") {
        let jobs = v.get("jobs").and_then(Value::object).map(|o| o.keys().cloned().collect::<Vec<_>>()).unwrap_or_default();
        let first = v.get("jobs").ok().and_then(|j| j.object().ok()).and_then(|o| o.values().next().cloned());
        let layers = first.as_ref().and_then(|j| j.get("layers").ok()).map_or_else(Vec::new, layers_of);
        let sockets: Vec<String> = first
            .as_ref()
            .and_then(|j| j.get("layers").ok())
            .and_then(|l| l.object().ok())
            .and_then(|o| o.values().next().and_then(|x| x.object().ok()).map(|m| m.keys().cloned().collect()))
            .unwrap_or_default();
        return obj(vec![("jobs", Value::Array(jobs.iter().map(|j| s(j)).collect())), ("layers", Value::Array(layers.iter().map(|l| n(*l)).collect())), ("sockets", Value::Array(sockets.iter().map(|x| s(x)).collect()))]);
    }
    if name.starts_with("act-") {
        let off = v.get("off").ok();
        let layers = off.map_or_else(Vec::new, layers_of);
        let channels: u64 = off.and_then(|o| o.object().ok()).map_or(0, |o| o.values().map(|x| x.array().map_or(0, |a| a.len() as u64)).sum());
        return obj(vec![("layers", Value::Array(layers.iter().map(|l| n(*l)).collect())), ("channels_off", n(channels))]);
    }
    if name == "super-weights.json" {
        let tensors = v.get("tensors").and_then(Value::object).map_or(0, |o| o.len() as u64);
        return obj(vec![("tensors", n(tensors))]);
    }
    let layers = layers_of(&v);
    if !layers.is_empty() {
        return obj(vec![("layers", Value::Array(layers.iter().map(|l| n(*l)).collect()))]);
    }
    obj(vec![("entries", n(v.object().map(|o| o.len()).or_else(|_| v.array().map(<[Value]>::len)).unwrap_or(0) as u64))])
}

type Cache = BTreeMap<PathBuf, (u64, u64, Value)>;
static CACHE: Mutex<Cache> = Mutex::new(BTreeMap::new());

fn cached_summary(path: &Path, name: &str, len: u64, mtime: u64) -> Value {
    if let Ok(g) = CACHE.lock()
        && let Some((l, t, v)) = g.get(path)
        && *l == len
        && *t == mtime
    {
        return v.clone();
    }
    let v = summary(path, name);
    if let Ok(mut g) = CACHE.lock() {
        g.insert(path.to_owned(), (len, mtime, v.clone()));
    }
    v
}

/// Every map of every MicroModel, by role.
pub fn view(home: &Path) -> Value {
    // One view is computed at a time; a request that arrives meanwhile waits and then finds the
    // summaries cached (requests in parallel each read every map before the cache filled).
    static ONE: Mutex<()> = Mutex::new(());
    let _one = ONE.lock();
    let base = home.join("library/parameter-maps");
    backfill(home);
    let mut roles = Vec::new();
    for role in ["reader", "writer", "embedder", "reranker"] {
        let dir = base.join(role);
        // 2026-11-26 (Luis Jake Gabriel III: "keep track of what they belong to"): each map's
        // lineage, and the base cartridge its MicroModel serves now, to say whether it is current.
        let lineage = nmapi::lineage::read(&dir);
        let model = nmtype::role_models::resolve(home).get(role).cloned().unwrap_or_default();
        let serving = serving_base(home, &model);
        let mut files: Vec<(String, PathBuf)> = std::fs::read_dir(&dir)
            .map(|rd| rd.flatten().filter(|e| e.path().is_file()).filter_map(|e| e.file_name().to_str().map(|n| (n.to_owned(), e.path()))).collect())
            .unwrap_or_default();
        files.sort();
        let maps: Vec<Value> = files
            .iter()
            .filter(|(name, _)| !name.ends_with(".tmp") && !name.ends_with(".lock") && name != nmapi::lineage::FILE)
            .map(|(name, p)| {
                let meta = std::fs::metadata(p).ok();
                let len = meta.as_ref().map_or(0, std::fs::Metadata::len);
                let mtime = meta.and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0));
                let (kind, about) = kind_of(name);
                let rec = lineage.get(name.as_str());
                let (state, why) = state_of(name, rec, &model, &serving);
                obj(vec![
                    ("file", s(name)),
                    ("kind", s(kind)),
                    ("about", s(about)),
                    ("bytes", n(len)),
                    ("modified_ms", n(mtime)),
                    ("summary", cached_summary(p, name, len, mtime)),
                    ("purpose", s(&purpose_of(name, rec))),
                    ("lineage", rec.map_or(Value::Null, nmapi::lineage::Record::to_value)),
                    ("state", s(state)),
                    ("state_why", s(&why)),
                ])
            })
            .collect();
        // 2026-11-26 (Luis Jake Gabriel III: "We have to have a way to clean up old maps too"): the
        // maps archived out of the folder, and how much each kind of old map takes.
        let archived: Vec<Value> = std::fs::read_dir(dir.join(ARCHIVE))
            .map(|rd| {
                let mut v: Vec<(String, u64, u64)> = rd
                    .flatten()
                    .filter(|e| e.path().is_file())
                    .filter_map(|e| {
                        let m = e.metadata().ok()?;
                        let t = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0));
                        Some((e.file_name().to_str()?.to_owned(), m.len(), t))
                    })
                    .collect();
                v.sort();
                v.into_iter()
                    .map(|(f, b, t)| {
                        let (kind, about) = kind_of(&f);
                        obj(vec![("file", s(&f)), ("kind", s(kind)), ("about", s(about)), ("bytes", n(b)), ("modified_ms", n(t)), ("lineage", lineage.get(f.as_str()).map_or(Value::Null, nmapi::lineage::Record::to_value))])
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut cleanup = BTreeMap::new();
        for m in &maps {
            let st = m.get("state").and_then(Value::string).unwrap_or("");
            if matches!(st, "experiment" | "history" | "stale") {
                let e = cleanup.entry(st.to_owned()).or_insert((0u64, 0u64));
                e.0 += 1;
                e.1 += m.get("bytes").and_then(Value::integer).unwrap_or(0);
            }
        }
        let cleanup = Value::Object(cleanup.into_iter().map(|(k, (c, b))| (k, obj(vec![("maps", n(c)), ("bytes", n(b))]))).collect());
        roles.push(obj(vec![
            ("role", s(role)),
            ("model", s(&model)),
            ("serving_base", if serving.is_empty() { Value::Null } else { s(&serving) }),
            ("maps", Value::Array(maps)),
            ("archived", Value::Array(archived)),
            ("cleanup", cleanup),
        ]));
    }
    obj(vec![("roles", Value::Array(roles))])
}

/// The base cartridge a model serves now (`id@version` from `designs/base-cartridges.json`), or empty.
#[must_use]
pub fn serving_base(home: &Path, model: &str) -> String {
    std::fs::read(home.join("designs/base-cartridges.json"))
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .and_then(|v| v.get("serving").ok().and_then(|sv| sv.get(model).ok()).map(|e| (text(e, "cartridge"), e.get("version").and_then(Value::integer).unwrap_or(0))))
        .filter(|(c, _)| !c.is_empty())
        .map_or_else(String::new, |(c, ver)| format!("{c}@{ver}"))
}

/// What a map is for: its recorded purpose, else what its kind of file is always for (a job filter
/// is read while serving; a surgery probe and ledger are probes); empty when neither says.
fn purpose_of(name: &str, rec: Option<&nmapi::lineage::Record>) -> String {
    if let Some(r) = rec.filter(|r| !r.purpose.is_empty()) {
        return r.purpose.clone();
    }
    if name.starts_with("job-filter-") {
        "serving".into()
    } else if name.starts_with("surgery-") {
        "probe".into()
    } else {
        String::new()
    }
}

/// Whether a map is current for what its MicroModel serves now: `experiment`, `current`, `stale`
/// (made on another base cartridge version than the one serving), or `origin not recorded`; with why.
fn state_of(name: &str, rec: Option<&nmapi::lineage::Record>, model: &str, serving: &str) -> (&'static str, String) {
    if purpose_of(name, rec) == "experiment" {
        return ("experiment", "an experiment or comparison run; nothing that serves or builds reads it".into());
    }
    // A map that built cartridges: current when the serving base (or what it carries) is among
    // them, else history (it built an earlier version only).
    if let Some(r) = rec.filter(|r| !r.used_by.is_empty() && matches!(r.purpose.as_str(), "base-build" | "measurement"))
        && !serving.is_empty()
    {
        return if r.used_by.iter().any(|u| u == serving) {
            ("current", format!("{serving}, serving now, is built from it"))
        } else {
            ("history", format!("built {}; {serving} serves now", r.used_by.join(", ")))
        };
    }
    if name.ends_with(".jsonl") && purpose_of(name, rec) == "probe" {
        let empty = std::path::Path::new(name).extension().is_some() && rec.is_some_and(|r| r.made_on.is_empty());
        return ("log", if empty { "the log of every surgery write to this MicroModel's base cartridge".into() } else { "a log".into() });
    }
    let Some(r) = rec.filter(|r| !r.made_on.is_empty()) else {
        return ("origin not recorded", "this map does not say what it was measured on; state it, or make it again with a tool that records it".into());
    };
    if let Some(m) = r.made_on.strip_prefix("open:") {
        return if m == model {
            ("current", format!("measured on the open weights of {m}, which do not change"))
        } else {
            ("stale", format!("measured on the open weights of {m}; this MicroModel runs {model}"))
        };
    }
    if serving.is_empty() {
        return ("origin not recorded", format!("made on {}; no base cartridge is named as serving for {model}", r.made_on));
    }
    if r.made_on == serving {
        ("current", format!("made on {serving}, the base cartridge serving now"))
    } else {
        ("stale", format!("made on {}; {serving} serves now", r.made_on))
    }
}

/// The base cartridge a base cartridge was made from, read from its label's `made_from` (the text
/// before " + "), or empty.
fn parent_of(home: &Path, serving: &str) -> String {
    crate::library::open(home)
        .ok()
        .and_then(|lib| lib.cartridge(serving).map(|c| c.made_from.clone()))
        .and_then(|m| m.split(" + ").next().map(|x| x.trim().to_owned()))
        .filter(|x| x.contains('@'))
        .unwrap_or_default()
}

/// One backfilled record (`recorded: backfill`).
fn proven(made_on: &str, made_by: &str, purpose: &str, used_by: &[&str], note: &str) -> nmapi::lineage::Record {
    nmapi::lineage::Record {
        made_on: made_on.into(),
        made_by: made_by.into(),
        made_ms: 0,
        purpose: purpose.into(),
        used_by: used_by.iter().map(|u| (*u).to_owned()).collect(),
        recorded: "backfill".into(),
        note: note.into(),
        ..nmapi::lineage::Record::default()
    }
}

/// The lineage of a map made before lineage was recorded, from what the map itself and the lab's
/// code prove (traced 2026-11-26: which `nano-lab` command writes each kind of map, which build read
/// it, and what the serving base cartridges were made from). None when nothing proves it.
fn backfilled(role: &str, file: &str, model: &str, serving: &str, parent: &str, map: &Path) -> Option<nmapi::lineage::Record> {
    let open = format!("open:{model}");
    let head = || std::fs::read(map).ok().and_then(|b| nmjson::parse(&b).ok());
    // `parent` is the base cartridge the serving one was made from, as its own label says
    // ("embedder-base@3 + open weights via `nano-lab run pmap surgery` ...").
    let surgery = format!("nano-lab run pmap surgery cascade {role} --anchor-budget 8 --budget 5 --write");
    if let Some(job) = file.strip_prefix("job-filter-").and_then(|x| x.strip_suffix(".json")) {
        let v = head()?;
        let base = text(&v, "base");
        let cart = text(&v, "cartridge");
        let layers = text(&v, "layers");
        let by = format!("nano-lab run pmap surgery job-filter {role} --job {job} --layers {} --split-gate --write", if layers.is_empty() { "all" } else { &layers });
        return Some(proven(&base, &by, "serving", &[cart.as_str()], "read by the serving engine while it is plugged in; serving checks its model, not its base version"));
    }
    let r = match file {
        "surgery-probe.json" => {
            let v = head()?;
            proven(&text(&v, "cartridge"), "nano-lab run pmap surgery (every operation but job-filter, dry runs too)", "probe", &[], "layer-by-layer drift of a base cartridge against the open weights; this one was not rewritten after the last surgery write, so it describes the base before the serving one")
        }
        "surgery-ledger.jsonl" => proven("", &surgery, "probe", &[], "one line per surgery write: the parent, the new version, what was added back"),
        "super-weights.json" => proven(parent, &surgery, "base-build", &[serving], "the coordinates surgery restored, merged run by run; read by every surgery run (a cascade only adds weights back, so it did not change what was cut)"),
        "map.json" => proven(&open, "nano-lab run bcart map solve", "base-build", &["reranker-base@3"], "the energy map solved from record.json and tol-*; built reranker-base@3"),
        "role.json" => proven(&open, "nano-lab run bcart map role", "base-build", &["reranker-base@3"], "the reranker's role map; built reranker-base@3"),
        "role-20-bias.json" => proven(&open, "nano-lab run emap judge <set> role+bias:20", "base-build", &["reranker-base-comp@1"], "built the reranker's compensation cartridge (made for reranker-base@3)"),
        "swigate-mean.json" => proven(&open, "nano-lab run emap judge-mean", "measurement", &["role-20-bias.json"], "mean feed-forward gate outputs of the open weights"),
        "scores.json" => proven(&open, "nano-lab run bcart record map", "experiment", &[], "per-head and per-channel scores read only by trial cartridges"),
        "tol-a.json" | "tol-b.json" => proven(&open, "nano-lab run bcart map tol <a|b> <first> <last>", "measurement", &["map.json"], "tolerance of each layer to lesions, one half of the question set"),
        "union.json" | "union-bias.json" => proven(&open, "nano-lab run ecart map model=reader rec=scratch/lane-d/reader/rec/<set>.json", "experiment", &[], "a union of the reader's channel records; no build ever read it"),
        "record.json" => match role {
            "reranker" => proven(&open, "nano-lab run bcart map record", "measurement", &["tol-a.json", "tol-b.json", "map.json"], "channel records of the open weights over the map part of the question set"),
            "writer" => proven(&open, "nano-lab run ecart record model=writer", "measurement", &["act-20.json", "act-20-bias.json"], "channel records of the open weights"),
            "embedder" => proven(&open, "nano-lab run emap embed-record lane-d-tune-code.json", "measurement", &["act-20.json", "act-20-bias.json"], "channel records of the open weights, even half of the set"),
            _ => return None,
        },
        "act-20.json" => match role {
            "embedder" => proven(&open, "nano-lab run emap embed <set> act:20", "base-build", &["embedder-base@2"], "channels left out per layer at 20 percent; built embedder-base@2"),
            "writer" => proven(&open, "nano-lab run ecart map model=writer share=20", "base-build", &["writer-base@2"], "channels left out per layer at 20 percent; built writer-base@2"),
            _ => return None,
        },
        "act-20-bias.json" => match role {
            "embedder" => proven(&open, "nano-lab run emap embed <set> act+bias:20", "base-build", &["embedder-base-comp@1"], "built the embedder's compensation cartridge (made for embedder-base@2)"),
            "writer" => proven(&open, "nano-lab run ecart map model=writer share=20", "base-build", &["writer-base-comp@1"], "built the writer's compensation cartridge (made for writer-base@2)"),
            _ => return None,
        },
        "pmap-v1.json" => proven(&open, &format!("nano-lab run pmap sweep {role}"), "experiment", &[], "a sweep; byte-identical to this MicroModel's pmap-owl map, which is the one on record"),
        _ if file.starts_with("pmap-owl-") => {
            let share = file.trim_start_matches("pmap-owl-").trim_end_matches(".json");
            let by = format!("nano-lab run pmap build-base {role} 0.{share} gpu");
            proven(&open, &by, "base-build", &[parent, serving], &format!("the cut that made {parent}; it carries into {serving}, since surgery since then only added weights back{}", if matches!(role, "embedder" | "reranker") { " (the share is read from the file name)" } else { "" }))
        }
        _ if file.starts_with("act-20-next") => proven(&open, "nano-lab run ecart addback model=embed base=act-20 out=act-20-next[l]", "experiment", &[], "an add-back variant of act-20.json; read only by emap experiments, no build used it"),
        _ if file.starts_with("zsep-") => proven(&open, "nano-lab run bcart zsep run <arm>", "experiment", &[], "one arm of a lesion comparison; read only by bcart zsep report"),
        _ => return None,
    };
    Some(r)
}

/// Fills in, once per process and only where a map has no record yet, the lineage the maps and the
/// lab's code prove (`backfilled`). Never overwrites a record a tool or the user wrote.
fn backfill(home: &Path) {
    static DONE: Mutex<bool> = Mutex::new(false);
    let Ok(mut done) = DONE.lock() else { return };
    if *done {
        return;
    }
    *done = true;
    let roles = nmtype::role_models::resolve(home);
    for role in ROLES {
        let dir = home.join("library/parameter-maps").join(role);
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        let model = roles.get(role).cloned().unwrap_or_default();
        let serving = serving_base(home, &model);
        let parent = parent_of(home, &serving);
        let have = nmapi::lineage::read(&dir);
        let mut add = Vec::new();
        for e in rd.flatten().filter(|e| e.path().is_file()) {
            let Some(name) = e.file_name().to_str().map(str::to_owned) else { continue };
            if name == nmapi::lineage::FILE || have.contains_key(&name) || name.ends_with(".tmp") {
                continue;
            }
            if let Some(r) = backfilled(role, &name, &model, &serving, &parent, &e.path()) {
                add.push((name, r));
            }
        }
        if !add.is_empty() {
            let _ = nmapi::lineage::change(&dir, |all| {
                for (k, r) in add {
                    all.entry(k).or_insert(r);
                }
            });
        }
    }
}

/// `POST /api/parameter-maps/lineage {role, file, made_on?, made_by?, purpose?, note?, outcome?,
/// outcome_why?}` (a map in the role's folder or its archive): states
/// where a map came from or what it is for (`recorded: user`). Audited with the record as it was,
/// so the Audit page undoes it.
/// # Errors
/// A bad role or file, a purpose that is not one of `nmapi::lineage::PURPOSES`, or a write failure.
pub fn set_lineage(home: &Path, role: &str, file: &str, v: &Value, by: &str) -> Result<Value, String> {
    // 2026-11-26: an archived map keeps its record too, so its outcome can be stated.
    let archived = ROLES.contains(&role) && !file.contains(['/', '\\']) && !file.starts_with('.') && home.join("library/parameter-maps").join(role).join("archive").join(file).is_file();
    if !archived {
        map_path(home, role, file)?;
    }
    let purpose = text(v, "purpose");
    let outcome = text(v, "outcome");
    if !outcome.is_empty() && !nmapi::lineage::OUTCOMES.contains(&outcome.as_str()) {
        return Err(format!("outcome {outcome}: use one of {}", nmapi::lineage::OUTCOMES.join(", ")));
    }
    if matches!(outcome.as_str(), "failed" | "duplicate" | "control") && text(v, "outcome_why").trim().is_empty() {
        return Err(format!("outcome {outcome} needs its reason (outcome_why): the gate, the dip, the measurement or the fact"));
    }
    // 2026-11-26 (Luis Jake Gabriel III: "Successful data from failure data needs to be kept
    // separate"): a map labelled anything but healthy leaves the library for its archive (the
    // quarantine) in the same step. A map in use is never quarantined: one that does not work is
    // IN_TROUBLE and is fixed in place (the five part states).
    let quarantine = !archived && !outcome.is_empty() && !matches!(outcome.as_str(), "used" | "superseded");
    if quarantine && let Some(why) = protected(home, role, file) {
        return Err(format!("{file} is in use ({why}): a map in use that does not work is IN_TROUBLE and is fixed in place, not labelled {outcome} or quarantined"));
    }
    if !purpose.is_empty() && !nmapi::lineage::PURPOSES.contains(&purpose.as_str()) {
        return Err(format!("purpose {purpose}: use one of {}", nmapi::lineage::PURPOSES.join(", ")));
    }
    let dir = home.join("library/parameter-maps").join(role);
    let before = nmapi::lineage::read(&dir).get(file).cloned();
    let after = nmapi::lineage::change(&dir, |all| {
        let r = all.entry(file.to_owned()).or_default();
        for (k, slot) in [("made_on", &mut r.made_on), ("made_by", &mut r.made_by), ("note", &mut r.note), ("outcome_why", &mut r.outcome_why)] {
            if let Ok(x) = v.get(k).and_then(Value::string) {
                *slot = x.trim().to_owned();
            }
        }
        if !purpose.is_empty() {
            r.purpose.clone_from(&purpose);
        }
        if !outcome.is_empty() {
            r.outcome.clone_from(&outcome);
        }
        r.recorded = "user".into();
        r.clone()
    })?;
    let audit = crate::controls::audit(
        home,
        "parameter-map",
        vec![
            ("action", s("lineage")),
            ("role", s(role)),
            ("file", s(file)),
            ("from", before.as_ref().map_or(Value::Null, nmapi::lineage::Record::to_value)),
            ("to", after.to_value()),
            ("by", s(by)),
            ("label", s(&format!("Lineage of {role}/{file} stated"))),
        ],
    )?;
    let mut out = vec![("role", s(role)), ("file", s(file)), ("lineage", after.to_value()), ("audit_id", s(&audit))];
    if quarantine {
        let moved = archive(home, role, &[file.to_owned()], by)?;
        out.push(("quarantined", Value::Bool(true)));
        out.push(("quarantine_audit_id", moved.get("audit_id").cloned().unwrap_or(Value::Null)));
    }
    Ok(obj(out))
}

/// Undoes a stated lineage: the record goes back to what it was (or away, when there was none).
/// # Errors
/// A row without its role or file, or a write failure.
pub fn undo_lineage(home: &Path, row: &Value, id: &str, by: &str) -> Result<Option<String>, String> {
    let (role, file) = (text(row, "role"), text(row, "file"));
    if role.is_empty() || file.is_empty() {
        return Err(format!("{id}: not a lineage row"));
    }
    let dir = home.join("library/parameter-maps").join(&role);
    let back = row.get("from").ok().filter(|f| !matches!(f, Value::Null)).map(nmapi::lineage::Record::from_value);
    nmapi::lineage::change(&dir, |all| match &back {
        Some(r) => {
            all.insert(file.clone(), r.clone());
        }
        None => {
            all.remove(&file);
        }
    })?;
    crate::controls::audit(home, "parameter-map", vec![("action", s("lineage-undo")), ("role", s(&role)), ("file", s(&file)), ("undo_of", s(id)), ("by", s(by)), ("label", s(&format!("Lineage of {role}/{file} put back")))]).map(Some)
}

/// The folder old maps are moved into, under each role's folder. Nothing that serves or builds reads it.
pub const ARCHIVE: &str = "archive";

/// Why a map is kept where it is (moving or deleting it would change what serves, or what the
/// serving base is built from), or None when it can be cleaned up.
fn protected(home: &Path, role: &str, file: &str) -> Option<String> {
    if file.starts_with("job-filter-") {
        return Some("a job filter is read while serving; unplug or remove it with its own controls".into());
    }
    if matches!(file, "super-weights.json" | "surgery-ledger.jsonl") {
        return Some("surgery reads and writes it at a fixed place".into());
    }
    let dir = home.join("library/parameter-maps").join(role);
    let model = nmtype::role_models::resolve(home).get(role).cloned().unwrap_or_default();
    let serving = serving_base(home, &model);
    let rec = nmapi::lineage::read(&dir).remove(file);
    let (state, why) = state_of(file, rec.as_ref(), &model, &serving);
    (state == "current" && purpose_of(file, rec.as_ref()) == "base-build").then(|| format!("current: {why}"))
}

/// The map files a cleanup names: `files` (a list) or `file` (one), each a plain name.
/// # Errors
/// None given, or a name that is not a plain file name.
pub fn files_of(v: &Value) -> Result<Vec<String>, String> {
    let list: Vec<String> = match (v.get("files").and_then(Value::array), v.get("file").and_then(Value::string)) {
        (Ok(a), _) => a.iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect(),
        (_, Ok(f)) => vec![f.to_owned()],
        _ => return Err("files: give the map file names (or file for one)".into()),
    };
    if list.is_empty() {
        return Err("files: give at least one map".into());
    }
    if let Some(bad) = list.iter().find(|f| f.contains(['/', '\\']) || f.starts_with('.') || f.is_empty() || f.as_str() == nmapi::lineage::FILE) {
        return Err(format!("{bad}: a map file name in the role's folder"));
    }
    Ok(list)
}

/// Moves old maps into `<role>/archive/` (nothing serves or builds from there; their lineage stays
/// with them), all or none: a map that serves, or that the serving base is built from, is refused.
/// Audited; the Audit page undoes it (the maps go back).
/// # Errors
/// A bad role, a map missing or protected, one already archived under that name, or a move failing.
pub fn archive(home: &Path, role: &str, files: &[String], by: &str) -> Result<Value, String> {
    move_maps(home, role, files, true, None, by)
}

/// Brings archived maps back into the role's folder, all or none (refused where a map of that name is
/// there). Audited; the Audit page undoes it (they go back to the archive).
/// # Errors
/// A bad role, a map not in the archive, a name taken, or a move failing.
pub fn restore(home: &Path, role: &str, files: &[String], by: &str) -> Result<Value, String> {
    move_maps(home, role, files, false, None, by)
}

fn move_maps(home: &Path, role: &str, files: &[String], to_archive: bool, undo_of: Option<&str>, by: &str) -> Result<Value, String> {
    if !ROLES.contains(&role) {
        return Err(format!("{role}: not a role (reader, writer, embedder or reranker)"));
    }
    let dir = home.join("library/parameter-maps").join(role);
    let arch = dir.join(ARCHIVE);
    let (from, to) = if to_archive { (dir.clone(), arch.clone()) } else { (arch.clone(), dir.clone()) };
    let mut bytes = 0;
    for f in files {
        let p = from.join(f);
        if !p.is_file() {
            return Err(format!("{f}: not {}", if to_archive { "in the role's folder" } else { "in the archive" }));
        }
        if to_archive && let Some(why) = protected(home, role, f) {
            return Err(format!("{f} is kept: {why}"));
        }
        // 2026-11-26: failure data stays apart from healthy data; a map labelled other than used
        // or superseded comes back into the library only once it is labelled healthy again.
        if !to_archive
            && let Some(o) = nmapi::lineage::read(&dir).get(f.as_str()).map(|r| r.outcome.clone()).filter(|o| !o.is_empty() && !matches!(o.as_str(), "used" | "superseded"))
        {
            return Err(format!("{f} is labelled {o}: it stays out of the parameter map library, apart from healthy maps; label it used or superseded first"));
        }
        if to.join(f).exists() {
            return Err(format!("{f}: a map of that name is already {}", if to_archive { "in the archive" } else { "in the role's folder" }));
        }
        bytes += std::fs::metadata(&p).map_or(0, |m| m.len());
    }
    std::fs::create_dir_all(&to).map_err(|e| format!("{}: {e}", to.display()))?;
    for f in files {
        std::fs::rename(from.join(f), to.join(f)).map_err(|e| format!("{f}: {e}"))?;
    }
    let action = if to_archive { "archive" } else { "restore" };
    let mut fields = vec![
        ("action", s(action)),
        ("role", s(role)),
        ("files", Value::Array(files.iter().map(|f| s(f)).collect())),
        ("bytes", n(bytes)),
        ("by", s(by)),
        ("label", s(&format!("{} {} {role} {} {} the archive", if to_archive { "Archived" } else { "Restored" }, files.len(), if files.len() == 1 { "map" } else { "maps" }, if to_archive { "into" } else { "from" }))),
    ];
    if let Some(u) = undo_of {
        fields.push(("undo_of", s(u)));
    }
    let audit = crate::controls::audit(home, "parameter-map", fields)?;
    Ok(obj(vec![("audit_id", s(&audit)), ("role", s(role)), ("files", Value::Array(files.iter().map(|f| s(f)).collect())), ("bytes", n(bytes))]))
}

/// Undoes an archive (the maps come back) or a restore (they go back to the archive).
/// # Errors
/// A row without its maps, or the move failing.
pub fn undo_move(home: &Path, row: &Value, id: &str, by: &str) -> Result<Option<String>, String> {
    let role = text(row, "role");
    let files: Vec<String> = row.get("files").and_then(Value::array).map(|a| a.iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect()).unwrap_or_default();
    let back_to_archive = text(row, "action") == "restore";
    let v = move_maps(home, &role, &files, back_to_archive, Some(id), by)?;
    Ok(v.get("audit_id").and_then(Value::string).ok().map(str::to_owned))
}

/// The files a removal deletes, with their bytes, after the refusals (a protected map in the role's
/// folder; an archived map is never protected).
/// # Errors
/// A bad role, a map missing, or a protected one.
pub fn remove_plan(home: &Path, role: &str, files: &[String], archived: bool) -> Result<Vec<(PathBuf, u64)>, String> {
    if !ROLES.contains(&role) {
        return Err(format!("{role}: not a role"));
    }
    let dir = home.join("library/parameter-maps").join(role);
    let at = if archived { dir.join(ARCHIVE) } else { dir };
    files
        .iter()
        .map(|f| {
            if !archived && f.starts_with("job-filter-") {
                return Err("job filters are removed with the job filter's own remove action".to_owned());
            }
            if !archived && let Some(why) = protected(home, role, f) {
                return Err(format!("{f} is kept: {why}"));
            }
            let p = at.join(f);
            let b = std::fs::metadata(&p).map_err(|e| format!("{f}: {e}"))?.len();
            Ok((p, b))
        })
        .collect()
}

/// Deletes maps (after their confirmation), from the role's folder or its archive, and their lineage.
/// Returns the audit id.
/// # Errors
/// As `remove_plan`, or a removal failing.
pub fn remove_many(home: &Path, role: &str, files: &[String], archived: bool, by: &str) -> Result<String, String> {
    let plan = remove_plan(home, role, files, archived)?;
    let mut bytes = 0;
    for (p, b) in &plan {
        std::fs::remove_file(p).map_err(|e| format!("{}: {e}", p.display()))?;
        bytes += b;
    }
    let _ = nmapi::lineage::change(&home.join("library/parameter-maps").join(role), |all| {
        for f in files {
            all.remove(f);
        }
    });
    crate::controls::audit(
        home,
        "parameter-map-remove",
        vec![("role", s(role)), ("files", Value::Array(files.iter().map(|f| s(f)).collect())), ("archived", Value::Bool(archived)), ("bytes", n(bytes)), ("by", s(by))],
    )
}

/// Removes one map file (not a job filter, which the surgery routes remove). Returns the audit id.
/// # Errors
/// An unknown role or file, a job filter, or a failed removal.
pub fn remove(home: &Path, role: &str, file: &str, by: &str) -> Result<String, String> {
    if !["reader", "writer", "embedder", "reranker"].contains(&role) {
        return Err(format!("{role}: not a role"));
    }
    if file.contains(['/', '\\']) || file.starts_with('.') || file.is_empty() {
        return Err(format!("{file}: a file name in the role's folder"));
    }
    if file.starts_with("job-filter-") {
        return Err("job filters are removed with the job filter's own remove action".into());
    }
    let p = home.join("library/parameter-maps").join(role).join(file);
    let bytes = std::fs::metadata(&p).map_err(|e| format!("{file}: {e}"))?.len();
    std::fs::remove_file(&p).map_err(|e| format!("{file}: {e}"))?;
    let _ = nmapi::lineage::change(&home.join("library/parameter-maps").join(role), |all| all.remove(file));
    crate::controls::audit(home, "parameter-map-remove", vec![("role", s(role)), ("file", s(file)), ("bytes", n(bytes)), ("by", s(by))])
}

// ---------------------------------------------------------------------------------------------
// Reading and editing a map's contents (2026-11-26, Phase B items A2 and A4; directions of Luis Jake
// Gabriel III: "The whole parameter mapping can be seen and controlled, including adding and
// editing maps").

/// The most bytes of map values one read returns: a value larger than this is listed by its keys
/// or paged by its elements, never sent whole.
pub const READ_CAP: usize = 2 * 1024 * 1024;
/// The largest editable map the save parses whole.
const SAVE_LIMIT: u64 = 8 * 1024 * 1024;
/// Where the file before each save is kept, under the role's folder.
const BACKUPS: &str = "backups";
const ROLES: [&str; 4] = ["reader", "writer", "embedder", "reranker"];

/// The maps a save may change: the role weights, the pinned super-weights, the tolerances and the
/// job filters. The parameter, activation and separation maps, the block map and the energy record
/// are measurements a build reads; they are read here, not edited.
#[must_use]
pub fn editable(file: &str) -> bool {
    file.ends_with(".json") && (file == "role.json" || file == "super-weights.json" || file.starts_with("tol-") || file.starts_with("job-filter-"))
}

/// A map file directly in a role's folder, after the name checks.
fn map_path(home: &Path, role: &str, file: &str) -> Result<PathBuf, String> {
    if !ROLES.contains(&role) {
        return Err(format!("{role}: not a role (reader, writer, embedder or reranker)"));
    }
    if file.is_empty() || file.contains(['/', '\\']) || file.starts_with('.') || file.ends_with(".tmp") || file.ends_with(".lock") {
        return Err(format!("{file}: a map's file name in the role's folder"));
    }
    let p = home.join("library/parameter-maps").join(role).join(file);
    if !p.is_file() {
        return Err(format!("{role}/{file}: no such map"));
    }
    Ok(p)
}

fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while b.get(i).is_some_and(|c| matches!(c, b' ' | b'\n' | b'\r' | b'\t')) {
        i += 1;
    }
    i
}

/// The end (exclusive) of the JSON string starting at `i` (its opening quote).
fn string_end(b: &[u8], i: usize) -> Option<usize> {
    let mut j = i + 1;
    while let Some(&c) = b.get(j) {
        match c {
            b'\\' => j += 2,
            b'"' => return Some(j + 1),
            _ => j += 1,
        }
    }
    None
}

/// The end (exclusive) of the JSON value starting at `i`, read over its bytes without building it.
fn value_end(b: &[u8], i: usize) -> Option<usize> {
    match b.get(i)? {
        b'"' => string_end(b, i),
        b'{' | b'[' => {
            let mut depth = 0_usize;
            let mut j = i;
            while let Some(&c) = b.get(j) {
                match c {
                    b'"' => {
                        j = string_end(b, j)?;
                        continue;
                    }
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth = depth.checked_sub(1)?;
                        if depth == 0 {
                            return Some(j + 1);
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            None
        }
        _ => {
            let mut j = i;
            while b.get(j).is_some_and(|c| !matches!(c, b',' | b'}' | b']' | b' ' | b'\n' | b'\r' | b'\t')) {
                j += 1;
            }
            (j > i).then_some(j)
        }
    }
}

/// The members of the object at `i`: (key, value start, value end).
fn members(b: &[u8], i: usize) -> Option<Vec<(String, usize, usize)>> {
    if b.get(i) != Some(&b'{') {
        return None;
    }
    let mut out = Vec::new();
    let mut j = skip_ws(b, i + 1);
    if b.get(j) == Some(&b'}') {
        return Some(out);
    }
    loop {
        let ke = string_end(b, j)?;
        let key = match nmjson::parse(b.get(j..ke)?).ok()? {
            Value::String(k) => k,
            _ => return None,
        };
        j = skip_ws(b, ke);
        if b.get(j) != Some(&b':') {
            return None;
        }
        let vs = skip_ws(b, j + 1);
        let ve = value_end(b, vs)?;
        out.push((key, vs, ve));
        j = skip_ws(b, ve);
        match b.get(j)? {
            b',' => j = skip_ws(b, j + 1),
            b'}' => return Some(out),
            _ => return None,
        }
    }
}

/// The elements of the array at `i`: (start, end).
fn elements(b: &[u8], i: usize) -> Option<Vec<(usize, usize)>> {
    if b.get(i) != Some(&b'[') {
        return None;
    }
    let mut out = Vec::new();
    let mut j = skip_ws(b, i + 1);
    if b.get(j) == Some(&b']') {
        return Some(out);
    }
    loop {
        let e = value_end(b, j)?;
        out.push((j, e));
        j = skip_ws(b, e);
        match b.get(j)? {
            b',' => j = skip_ws(b, j + 1),
            b']' => return Some(out),
            _ => return None,
        }
    }
}

/// The path a read or a save names: the `section` split on `/`, then the `layer`.
fn path_of(section: Option<&str>, layer: Option<&str>) -> Vec<String> {
    let mut p: Vec<String> = section.unwrap_or("").split('/').map(str::trim).filter(|x| !x.is_empty()).map(str::to_owned).collect();
    if let Some(l) = layer.map(str::trim).filter(|x| !x.is_empty()) {
        p.push(format!("\u{1}{l}"));
    }
    p
}

/// The key a path step names in an object: a `layer` step (marked) is `L<n>`, else `<n>`.
fn step_key<'a>(keys: impl Iterator<Item = &'a str> + Clone, step: &str) -> Option<String> {
    match step.strip_prefix('\u{1}') {
        Some(l) => {
            let want = [format!("L{l}"), l.to_owned()];
            want.into_iter().find(|w| keys.clone().any(|k| k == w))
        }
        None => keys.clone().find(|k| *k == step).map(str::to_owned),
    }
}

fn step_index(step: &str) -> Option<usize> {
    step.trim_start_matches('\u{1}').parse().ok()
}

fn shown_step(step: &str) -> String {
    step.trim_start_matches('\u{1}').to_owned()
}

/// The byte span of the value a path names, and the path as resolved.
fn locate(b: &[u8], path: &[String]) -> Result<(usize, usize, Vec<String>), String> {
    let start = skip_ws(b, 0);
    let mut span = (start, value_end(b, start).ok_or("the map is not one JSON value")?);
    let mut walked = Vec::new();
    for step in path {
        let next = match b.get(span.0) {
            Some(b'{') => {
                let m = members(b, span.0).ok_or("the map's JSON does not read")?;
                let key = step_key(m.iter().map(|(k, _, _)| k.as_str()), step).ok_or_else(|| format!("no key {} under /{}", shown_step(step), walked.join("/")))?;
                walked.push(key.clone());
                m.into_iter().find(|(k, _, _)| *k == key).map(|(_, s, e)| (s, e))
            }
            Some(b'[') => {
                let i = step_index(step).ok_or_else(|| format!("{} is not an index of the list at /{}", shown_step(step), walked.join("/")))?;
                walked.push(i.to_string());
                elements(b, span.0).ok_or("the map's JSON does not read")?.get(i).copied()
            }
            _ => None,
        };
        span = next.ok_or_else(|| format!("nothing at /{}", walked.join("/")))?;
    }
    Ok((span.0, span.1, walked))
}

fn kind_of_value(c: Option<&u8>) -> &'static str {
    match c {
        Some(b'{') => "object",
        Some(b'[') => "array",
        Some(b'"') => "string",
        Some(b't' | b'f') => "bool",
        Some(b'n') => "null",
        _ => "number",
    }
}

/// `GET /api/parameter-maps/read?role=<r>&file=<f>[&section=<a/b>][&layer=<n>][&offset=<n>][&limit=<n>]`:
/// a map's contents. A value of at most `READ_CAP` bytes comes whole (`mode` `full` for the whole
/// map, `value` for a part); a larger object is listed by its keys with each value's size (`keys`),
/// a larger list by a page of its elements (`items`); a `.jsonl` ledger by a page of its rows. The
/// file's bytes are read once and only the value named is parsed, so a 55 MB map is never built
/// whole. `section` is a path of keys (or list indexes) split by `/`; `layer` adds a step that
/// matches `L<n>`, then `<n>`, then index `n`.
/// # Errors
/// A bad role or file name, no such map, or a path that names nothing.
pub fn read(home: &Path, query: &str) -> Result<Value, String> {
    let arg = |k: &str| crate::project::query_arg(query, k).filter(|x| !x.is_empty());
    let (role, file) = (arg("role").ok_or("role: give the MicroModel's role")?, arg("file").ok_or("file: give the map's file name")?);
    let p = map_path(home, &role, &file)?;
    let num = |k: &str, d: usize| arg(k).and_then(|x| x.parse::<usize>().ok()).unwrap_or(d);
    let (offset, limit) = (num("offset", 0), num("limit", 0));
    let meta = std::fs::metadata(&p).map_err(|e| format!("{file}: {e}"))?;
    let modified = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0));
    let (kind, about) = kind_of(&file);
    let mut out = vec![
        ("role", s(&role)),
        ("file", s(&file)),
        ("kind", s(kind)),
        ("about", s(about)),
        ("bytes", n(meta.len())),
        ("modified_ms", n(modified)),
        ("editable", Value::Bool(editable(&file))),
        ("cap_bytes", n(READ_CAP as u64)),
    ];
    let b = std::fs::read(&p).map_err(|e| format!("{file}: {e}"))?;
    if file.ends_with(".jsonl") {
        let lines: Vec<&[u8]> = b.split(|c| *c == b'\n').filter(|l| l.iter().any(|c| !c.is_ascii_whitespace())).collect();
        let limit = if limit == 0 { 200 } else { limit.min(2000) };
        let rows: Vec<Value> = lines.iter().skip(offset).take(limit).map(|l| nmjson::parse(l).unwrap_or_else(|_| s("(a line that is not JSON)"))).collect();
        let more = offset + rows.len() < lines.len();
        out.extend([("mode", s("rows")), ("offset", n(offset as u64)), ("limit", n(limit as u64)), ("total", n(lines.len() as u64)), ("rows", Value::Array(rows)), ("truncated", Value::Bool(more))]);
        return Ok(obj(out));
    }
    let path = path_of(arg("section").as_deref(), arg("layer").as_deref());
    let (vs, ve, walked) = locate(&b, &path)?;
    out.push(("path", Value::Array(walked.iter().map(|w| s(w)).collect())));
    out.push(("value_bytes", n((ve - vs) as u64)));
    out.push(("type", s(kind_of_value(b.get(vs)))));
    let slice = b.get(vs..ve).unwrap_or_default();
    if ve - vs <= READ_CAP {
        let v = nmjson::parse(slice).map_err(|e| format!("{file}: {e}"))?;
        out.extend([("mode", s(if path.is_empty() { "full" } else { "value" })), ("content", v), ("truncated", Value::Bool(false))]);
        return Ok(obj(out));
    }
    let note = format!("this value is {} bytes, over the {READ_CAP} bytes one read returns; ", ve - vs);
    match b.get(vs) {
        Some(b'{') => {
            let m = members(&b, vs).ok_or("the map's JSON does not read")?;
            let limit = if limit == 0 { 500 } else { limit.min(4096) };
            let keys: Vec<Value> = m.iter().skip(offset).take(limit).map(|(k, s0, e0)| obj(vec![("key", s(k)), ("bytes", n((e0 - s0) as u64)), ("type", s(kind_of_value(b.get(*s0))))])).collect();
            let more = offset + keys.len() < m.len();
            out.extend([
                ("mode", s("keys")),
                ("offset", n(offset as u64)),
                ("limit", n(limit as u64)),
                ("total", n(m.len() as u64)),
                ("keys", Value::Array(keys)),
                ("truncated", Value::Bool(true)),
                ("more_keys", Value::Bool(more)),
                ("note", s(&format!("{note}its keys are listed; name one in section (keys split by /) or layer to read it"))),
            ]);
        }
        Some(b'[') => {
            let e = elements(&b, vs).ok_or("the map's JSON does not read")?;
            let limit = if limit == 0 { 100 } else { limit.min(10_000) };
            let (mut items, mut used) = (Vec::new(), 0_usize);
            for (s0, e0) in e.iter().skip(offset).take(limit) {
                if used + (e0 - s0) > READ_CAP && !items.is_empty() {
                    break;
                }
                if e0 - s0 > READ_CAP {
                    items.push(obj(vec![("bytes", n((e0 - s0) as u64)), ("type", s(kind_of_value(b.get(*s0)))), ("note", s("this element is over the cap; read it by its index"))]));
                } else {
                    items.push(nmjson::parse(b.get(*s0..*e0).unwrap_or_default()).map_err(|x| format!("{file}: {x}"))?);
                    used += e0 - s0;
                }
            }
            let more = offset + items.len() < e.len();
            out.extend([
                ("mode", s("items")),
                ("offset", n(offset as u64)),
                ("limit", n(limit as u64)),
                ("returned", n(items.len() as u64)),
                ("total", n(e.len() as u64)),
                ("items", Value::Array(items)),
                ("truncated", Value::Bool(more)),
                ("note", s(&format!("{note}a page of its elements from offset is returned"))),
            ]);
        }
        _ => return Err(format!("{note}a single value this large is not returned")),
    }
    Ok(obj(out))
}

fn variant(v: &Value) -> &'static str {
    match v {
        Value::Object(_) => "object",
        Value::Array(_) => "array",
        Value::String(_) => "string",
        Value::Number(_) => "number",
        Value::Bool(_) => "bool",
        Value::Null => "null",
    }
}

/// Every number in `v` is finite.
fn finite(v: &Value, at: &str) -> Result<(), String> {
    match v {
        Value::Number(x) => {
            if x.parse::<f64>().is_ok_and(f64::is_finite) {
                Ok(())
            } else {
                Err(format!("{at}: {x} is not a finite number"))
            }
        }
        Value::Array(a) => a.iter().enumerate().try_for_each(|(i, x)| finite(x, &format!("{at}/{i}"))),
        Value::Object(o) => o.iter().try_for_each(|(k, x)| finite(x, &format!("{at}/{k}"))),
        _ => Ok(()),
    }
}

/// The edited map keeps the map's shape: the same kind of value at the top, the same keys with
/// values of the same kinds (a list keeps its length and the kinds of its elements); a job filter
/// keeps every part, each part's target, and the lengths of its `gain` and `bias`; every number
/// is finite.
fn same_shape(file: &str, old: &Value, new: &Value) -> Result<(), String> {
    if variant(old) != variant(new) {
        return Err(format!("the map is a JSON {}; the edit is a JSON {}", variant(old), variant(new)));
    }
    match (old, new) {
        (Value::Object(a), Value::Object(b)) => {
            if let Some(k) = a.keys().find(|k| !b.contains_key(*k)) {
                return Err(format!("key {k} is missing from the edit"));
            }
            if let Some(k) = b.keys().find(|k| !a.contains_key(*k)) {
                return Err(format!("key {k} is not in the map"));
            }
            if let Some((k, x)) = a.iter().find(|(k, x)| b.get(*k).is_some_and(|y| variant(x) != variant(y))) {
                return Err(format!("{k}: was a JSON {}, the edit makes it another kind", variant(x)));
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                return Err(format!("the list has {} elements; the edit has {}", a.len(), b.len()));
            }
            if let Some(i) = a.iter().zip(b).position(|(x, y)| variant(x) != variant(y)) {
                return Err(format!("element {i} changes kind"));
            }
        }
        _ => {}
    }
    nested_shape(old, new, "")?;
    if file.starts_with("job-filter-") {
        let parts = |v: &Value| v.get("parts").and_then(Value::array).map(<[Value]>::to_vec).unwrap_or_default();
        let (pa, pb) = (parts(old), parts(new));
        if pa.len() != pb.len() {
            return Err(format!("the filter has {} parts; the edit has {}", pa.len(), pb.len()));
        }
        for (i, (x, y)) in pa.iter().zip(&pb).enumerate() {
            let t = |v: &Value| v.get("target").and_then(Value::string).map(str::to_owned).unwrap_or_default();
            if t(x) != t(y) {
                return Err(format!("part {i}: its target {} cannot change", t(x)));
            }
            for k in ["gain", "bias"] {
                let len = |v: &Value| v.get(k).and_then(Value::array).map(<[Value]>::len).ok();
                if len(x) != len(y) {
                    return Err(format!("part {i} ({}): {k} has {} values; the edit has {}", t(x), len(x).unwrap_or(0), len(y).unwrap_or(0)));
                }
                if let Ok(a) = y.get(k).and_then(Value::array)
                    && a.iter().any(|z| !matches!(z, Value::Number(_)))
                {
                    return Err(format!("part {i} ({}): {k} holds something other than numbers", t(x)));
                }
            }
        }
    }
    finite(new, "")
}

/// Below the top: every object keeps its keys, and every value (and every list element the old
/// and new lists both have) keeps its kind. Nested lists may change length (pinned super-weights
/// can be added or taken out); a job filter's gains and biases are held to their lengths apart.
fn nested_shape(old: &Value, new: &Value, at: &str) -> Result<(), String> {
    if variant(old) != variant(new) {
        return Err(format!("{at}: was a JSON {}, the edit makes it a JSON {}", variant(old), variant(new)));
    }
    match (old, new) {
        (Value::Object(a), Value::Object(b)) => {
            if let Some(k) = a.keys().find(|k| !b.contains_key(*k)) {
                return Err(format!("{at}/{k}: missing from the edit"));
            }
            if let Some(k) = b.keys().find(|k| !a.contains_key(*k)) {
                return Err(format!("{at}/{k}: not in the map"));
            }
            a.iter().try_for_each(|(k, x)| b.get(k).map_or(Ok(()), |y| nested_shape(x, y, &format!("{at}/{k}"))))
        }
        (Value::Array(a), Value::Array(b)) => a.iter().zip(b).enumerate().try_for_each(|(i, (x, y))| nested_shape(x, y, &format!("{at}/{i}"))),
        _ => Ok(()),
    }
}

/// The value a path names in a parsed map, to replace.
fn at_path<'a>(v: &'a mut Value, path: &[String]) -> Result<&'a mut Value, String> {
    let mut cur = v;
    let mut walked = Vec::new();
    for step in path {
        cur = match cur {
            Value::Object(o) => {
                let key = step_key(o.keys().map(String::as_str), step).ok_or_else(|| format!("no key {} under /{}", shown_step(step), walked.join("/")))?;
                walked.push(key.clone());
                o.get_mut(&key).ok_or("vanished key")?
            }
            Value::Array(a) => {
                let i = step_index(step).ok_or_else(|| format!("{} is not an index", shown_step(step)))?;
                walked.push(i.to_string());
                a.get_mut(i).ok_or_else(|| format!("nothing at /{}", walked.join("/")))?
            }
            _ => return Err(format!("nothing at /{}/{}", walked.join("/"), shown_step(step))),
        };
    }
    Ok(cur)
}

static MAP_WRITE: Mutex<()> = Mutex::new(());

/// Writes a map: the file as it was copied to `<role>/backups/<file>.<ms>.json`, then the new text
/// through a temporary file and a rename. Returns the backup's path relative to the home.
fn write_map(home: &Path, role: &str, file: &str, p: &Path, text: &[u8]) -> Result<String, String> {
    let rel = format!("library/parameter-maps/{role}/{BACKUPS}/{file}.{}.json", nmtype::now_ms());
    let bak = home.join(&rel);
    if let Some(d) = bak.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    std::fs::copy(p, &bak).map_err(|e| format!("{}: {e}", bak.display()))?;
    let tmp = p.with_extension("json.tmp");
    std::fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, p).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok(rel)
}

/// `POST /api/parameter-maps/save {role, file, content[, section][, layer]}`: replaces an editable
/// map (or, with `section`/`layer`, the one value they name in it) with `content` after checking it
/// keeps the map's shape (`same_shape`). The file as it was is kept under `<role>/backups/` and the
/// save is a `parameter-map` audit row the Audit page undoes (`undo_save`). A base cartridge
/// already built from a map keeps serving; a job filter is read again when a serving process
/// loads it.
/// # Errors
/// A map that is not editable, a path that names nothing, content that changes the shape, or a
/// write failure.
#[expect(clippy::too_many_arguments, reason = "a save names the map, the place in it, the value and who asked")]
pub fn save(home: &Path, role: &str, file: &str, section: Option<&str>, layer: Option<&str>, content: &Value, by: &str) -> Result<Value, String> {
    if !editable(file) {
        return Err(format!("{file}: only role.json, super-weights.json, tol-*.json and job-filter-*.json are edited; the other maps are measurements a build reads"));
    }
    let p = map_path(home, role, file)?;
    let _g = MAP_WRITE.lock().map_err(|_| "map write lock poisoned".to_owned())?;
    let before = std::fs::metadata(&p).map_err(|e| format!("{file}: {e}"))?.len();
    if before > SAVE_LIMIT {
        return Err(format!("{file}: {before} bytes, over the {SAVE_LIMIT} bytes an edit reads whole"));
    }
    let old = std::fs::read(&p).ok().and_then(|b| nmjson::parse(&b).ok()).ok_or_else(|| format!("{file}: the map's JSON does not read"))?;
    let path = path_of(section, layer);
    let mut new = old.clone();
    let place = at_path(&mut new, &path)?;
    if variant(place) != variant(content) {
        return Err(format!("the value at /{} is a JSON {}; the edit is a JSON {}", path.iter().map(|x| shown_step(x)).collect::<Vec<_>>().join("/"), variant(place), variant(content)));
    }
    *place = content.clone();
    same_shape(file, &old, &new)?;
    if new == old {
        return Ok(obj(vec![("role", s(role)), ("file", s(file)), ("changed", Value::Bool(false)), ("audit_id", Value::Null)]));
    }
    let text = new.encode().map_err(|e| format!("encode: {e}"))?;
    let backup = write_map(home, role, file, &p, text.as_bytes())?;
    let shown: Vec<String> = path.iter().map(|x| shown_step(x)).collect();
    let audit = crate::controls::audit(
        home,
        "parameter-map",
        vec![
            ("action", s("save")),
            ("role", s(role)),
            ("file", s(file)),
            ("section", s(&shown.join("/"))),
            ("backup", s(&backup)),
            ("bytes_before", n(before)),
            ("bytes_after", n(text.len() as u64)),
            ("by", s(by)),
            ("label", s(&format!("Parameter map {role}/{file} edited{}", if shown.is_empty() { String::new() } else { format!(" at /{}", shown.join("/")) }))),
        ],
    )?;
    Ok(obj(vec![
        ("role", s(role)),
        ("file", s(file)),
        ("changed", Value::Bool(true)),
        ("audit_id", s(&audit)),
        ("backup", s(&backup)),
        ("bytes", n(text.len() as u64)),
        ("undo", s("undo it in Audit (the file as it was is put back)")),
    ]))
}

/// Undoes a `parameter-map` save: the file as it is now is kept under `backups/` too, then the
/// backup the save made is put back. The undo is itself an audit row.
/// # Errors
/// A row that is not a save, a backup that is gone, or a write failure.
pub fn undo_save(home: &Path, row: &Value, id: &str, by: &str) -> Result<Option<String>, String> {
    let field = |k: &str| row.get(k).and_then(Value::string).map(str::to_owned).map_err(|e| format!("{k}: {e}"));
    let (role, file, backup) = (field("role")?, field("file")?, field("backup")?);
    if row.get("action").and_then(Value::string).is_ok_and(|a| a == "save") && !editable(&file) {
        return Err(format!("{file}: not an editable map"));
    }
    let prefix = format!("library/parameter-maps/{role}/{BACKUPS}/{file}.");
    if !backup.starts_with(&prefix) || backup.contains("..") {
        return Err("the audit row's backup is not a map backup".into());
    }
    let p = map_path(home, &role, &file)?;
    let _g = MAP_WRITE.lock().map_err(|_| "map write lock poisoned".to_owned())?;
    let text = std::fs::read(home.join(&backup)).map_err(|e| format!("{backup}: {e} (the backup is gone; nothing was put back)"))?;
    let kept = write_map(home, &role, &file, &p, &text)?;
    let aid = crate::controls::audit(
        home,
        "parameter-map",
        vec![
            ("action", s("undo")),
            ("role", s(&role)),
            ("file", s(&file)),
            ("backup", s(&kept)),
            ("undo_of", s(id)),
            ("by", s(by)),
            ("label", s(&format!("Undone: parameter map {role}/{file} edit"))),
        ],
    )?;
    Ok(Some(aid))
}

/// The largest map file an import copies.
const IMPORT_LIMIT: u64 = 128 * 1024 * 1024;

/// A file on this machine to import: a local path (a network share is refused: nothing is read
/// from the network), an existing regular file, at most `limit` bytes.
/// # Errors
/// A network path, a missing file, a folder, or a file over the limit.
pub fn local_file(path: &str, limit: u64) -> Result<PathBuf, String> {
    let t = path.trim();
    if t.is_empty() {
        return Err("path: give a file on this machine".into());
    }
    if t.starts_with("\\\\") || t.starts_with("//") || t.contains("://") {
        return Err(format!("{t}: a network path; only a file on this machine is imported"));
    }
    let p = PathBuf::from(t);
    let md = std::fs::metadata(&p).map_err(|e| format!("{t}: {e}"))?;
    if !md.is_file() {
        return Err(format!("{t}: not a file"));
    }
    if md.len() > limit {
        return Err(format!("{t}: {} bytes, over the {limit} bytes an import takes", md.len()));
    }
    Ok(p)
}

/// The checks an imported map passes for its kind (by its name): JSON throughout (a ledger: every
/// line), the top-level keys its kind is read by (`jobs` for a parameter map, `off` for an
/// activation map, `tensors` for the super-weights, `parts` with each part's target and equal-length
/// numeric gain and bias for a job filter), finite numbers where it is parsed; a map over the size
/// a save parses is checked by its structure without being built. Replacing a map keeps its shape.
fn check_import(file: &str, bytes: &[u8], old: Option<&Value>) -> Result<(), String> {
    if file.ends_with(".jsonl") {
        for (i, l) in bytes.split(|c| *c == b'\n').enumerate().filter(|(_, l)| l.iter().any(|c| !c.is_ascii_whitespace())) {
            nmjson::parse(l).map_err(|e| format!("line {}: not JSON ({e})", i + 1))?;
        }
        return Ok(());
    }
    if !file.ends_with(".json") {
        return Err(format!("{file}: a map is a .json file (or a .jsonl ledger)"));
    }
    let start = skip_ws(bytes, 0);
    let end = value_end(bytes, start).ok_or("not one JSON value")?;
    if skip_ws(bytes, end) != bytes.len() {
        return Err("text after the JSON value".into());
    }
    let need = |k: &str| -> Result<(), String> {
        let m = members(bytes, start).ok_or("the map must be a JSON object")?;
        if m.iter().any(|(x, _, _)| x == k) { Ok(()) } else { Err(format!("{file}: a {} needs the key {k}", kind_of(file).0)) }
    };
    if file.starts_with("pmap-") {
        need("jobs")?;
    } else if file.starts_with("act-") {
        need("off")?;
    } else if file == "super-weights.json" {
        need("tensors")?;
    } else if file.starts_with("job-filter-") {
        need("parts")?;
    }
    if bytes.len() as u64 > SAVE_LIMIT {
        return Ok(());
    }
    let v = nmjson::parse(bytes).map_err(|e| format!("not JSON: {e}"))?;
    finite(&v, "")?;
    if file.starts_with("job-filter-") {
        for (i, p) in v.get("parts").and_then(Value::array).map_err(|e| format!("parts: {e}"))?.iter().enumerate() {
            let target = p.get("target").and_then(Value::string).map_err(|_| format!("part {i}: no target"))?;
            let nums = |k: &str| p.get(k).and_then(Value::array).ok().filter(|a| a.iter().all(|z| matches!(z, Value::Number(_)))).map(<[Value]>::len);
            match (nums("gain"), nums("bias")) {
                (Some(g), Some(b)) if g == b && g > 0 => {}
                _ => return Err(format!("part {i} ({target}): gain and bias must be number lists of the same length")),
            }
        }
    }
    if let Some(o) = old {
        same_shape(file, o, &v)?;
    }
    Ok(())
}

/// `POST /api/parameter-maps/import {role, path[, name][, replace]}`: copies a map file from a path
/// on this machine into the role's folder under its own name (or `name`), after the checks of its
/// kind (`check_import`). An existing map is replaced only with `replace: true`, and then keeps its
/// shape and is kept under `<role>/backups/` first. Audited as a `parameter-map` `import` row; undo
/// puts the replaced map back, or moves an added map to `backups/`.
/// # Errors
/// A bad role or name, a network or missing path, a file that fails its checks, an existing map
/// without `replace`, or a write failure.
pub fn import(home: &Path, role: &str, path: &str, name: Option<&str>, replace: bool, by: &str) -> Result<Value, String> {
    if !ROLES.contains(&role) {
        return Err(format!("{role}: not a role (reader, writer, embedder or reranker)"));
    }
    let src = local_file(path, IMPORT_LIMIT)?;
    let file = name.map_or_else(|| src.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default(), str::to_owned);
    if file.is_empty() || file.contains(['/', '\\']) || file.starts_with('.') || file.ends_with(".tmp") || file.ends_with(".lock") {
        return Err(format!("{file}: give a plain file name for the map"));
    }
    let dir = home.join("library/parameter-maps").join(role);
    let dest = dir.join(&file);
    if std::fs::canonicalize(&src).ok() == std::fs::canonicalize(&dest).ok() && dest.exists() {
        return Err("the file is that map already".into());
    }
    let bytes = std::fs::read(&src).map_err(|e| format!("{}: {e}", src.display()))?;
    let _g = MAP_WRITE.lock().map_err(|_| "map write lock poisoned".to_owned())?;
    let exists = dest.is_file();
    if exists && !replace {
        return Err(format!("{role}/{file} exists; send replace: true to replace it (the map as it is is kept under backups/)"));
    }
    let old = if exists && std::fs::metadata(&dest).is_ok_and(|m| m.len() <= SAVE_LIMIT) && file.ends_with(".json") { std::fs::read(&dest).ok().and_then(|b| nmjson::parse(&b).ok()) } else { None };
    check_import(&file, &bytes, old.as_ref())?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let backup = if exists {
        Some(write_map(home, role, &file, &dest, &bytes)?)
    } else {
        let tmp = dir.join(format!("{file}.tmp"));
        std::fs::write(&tmp, &bytes).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &dest).map_err(|e| format!("{}: {e}", dest.display()))?;
        None
    };
    // 2026-11-26: an imported map's lineage: where it came from (its file name only, never a path
    // of this machine, since lineage travels with the maps) and, for a job filter, its base.
    let base_of = nmjson::parse(&bytes).ok().map(|v| text(&v, "base")).unwrap_or_default();
    let from = src.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
    let _ = nmapi::lineage::record_made(&dir, &file, &base_of, &format!("imported from {from}"), if file.starts_with("job-filter-") { "serving" } else { "" });
    let audit = crate::controls::audit(
        home,
        "parameter-map",
        vec![
            ("action", s("import")),
            ("role", s(role)),
            ("file", s(&file)),
            ("source", s(&src.display().to_string())),
            ("backup", backup.as_deref().map_or(Value::Null, s)),
            ("bytes", n(bytes.len() as u64)),
            ("sha256", s(&nmhash::sha256_bytes(&bytes))),
            ("by", s(by)),
            ("label", s(&format!("Parameter map {role}/{file} {}", if exists { "replaced by an import" } else { "imported" }))),
        ],
    )?;
    Ok(obj(vec![
        ("role", s(role)),
        ("file", s(&file)),
        ("replaced", Value::Bool(exists)),
        ("backup", backup.as_deref().map_or(Value::Null, s)),
        ("bytes", n(bytes.len() as u64)),
        ("audit_id", s(&audit)),
        ("undo", s(if exists { "undo it in Audit (the map as it was is put back)" } else { "undo it in Audit (the imported map is moved to backups/)" })),
    ]))
}

/// Undoes a map import: a replaced map is put back from its backup; an added one is moved to
/// `<role>/backups/` (kept, out of the maps the scanners read).
/// # Errors
/// The map is gone, or a write failure.
pub fn undo_import(home: &Path, row: &Value, id: &str, by: &str) -> Result<Option<String>, String> {
    if row.get("backup").and_then(Value::string).is_ok() {
        return undo_save(home, row, id, by);
    }
    let field = |k: &str| row.get(k).and_then(Value::string).map(str::to_owned).map_err(|e| format!("{k}: {e}"));
    let (role, file) = (field("role")?, field("file")?);
    let p = map_path(home, &role, &file)?;
    let _g = MAP_WRITE.lock().map_err(|_| "map write lock poisoned".to_owned())?;
    let rel = format!("library/parameter-maps/{role}/{BACKUPS}/{file}.{}.json", nmtype::now_ms());
    let kept = home.join(&rel);
    if let Some(d) = kept.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    std::fs::rename(&p, &kept).map_err(|e| format!("{}: {e}", p.display()))?;
    let aid = crate::controls::audit(
        home,
        "parameter-map",
        vec![("action", s("undo")), ("role", s(&role)), ("file", s(&file)), ("backup", s(&rel)), ("undo_of", s(id)), ("by", s(by)), ("label", s(&format!("Undone: parameter map {role}/{file} import (moved to backups)")))],
    )?;
    Ok(Some(aid))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-11-26: old maps go into the archive and back (undoable), a serving map is kept, and an
    /// archived map is deleted with its lineage.
    #[test]
    fn old_maps_archive_restore_and_delete() {
        let home = std::env::temp_dir().join(format!("nm-pmaps-arch-{}", std::process::id()));
        let d = home.join("library/parameter-maps/reranker");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::create_dir_all(home.join("memory")).unwrap();
        for f in ["zsep-rand-1.json", "zsep-rand-2.json", "job-filter-x.json"] {
            std::fs::write(d.join(f), "{}").unwrap();
        }
        let files = |v: &[&str]| v.iter().map(|x| (*x).to_owned()).collect::<Vec<_>>();
        assert!(archive(&home, "reranker", &files(&["zsep-rand-1.json", "job-filter-x.json"]), "test").is_err(), "a job filter is kept, and nothing moves");
        assert!(d.join("zsep-rand-1.json").is_file(), "all or none");
        let r = archive(&home, "reranker", &files(&["zsep-rand-1.json", "zsep-rand-2.json"]), "test").unwrap();
        assert!(d.join(ARCHIVE).join("zsep-rand-1.json").is_file() && !d.join("zsep-rand-1.json").exists());
        let log = std::fs::read_to_string(home.join("memory/audit.jsonl")).unwrap();
        let row = log.lines().filter_map(|l| nmjson::parse(l.as_bytes()).ok()).find(|v| v.get("id").and_then(Value::string).ok() == r.get("audit_id").and_then(Value::string).ok()).unwrap();
        undo_move(&home, &row, "a1", "test").unwrap();
        assert!(d.join("zsep-rand-2.json").is_file(), "the archive is undone");
        archive(&home, "reranker", &files(&["zsep-rand-2.json"]), "test").unwrap();
        restore(&home, "reranker", &files(&["zsep-rand-2.json"]), "test").unwrap();
        assert!(d.join("zsep-rand-2.json").is_file());
        archive(&home, "reranker", &files(&["zsep-rand-1.json"]), "test").unwrap();
        assert!(remove_plan(&home, "reranker", &files(&["job-filter-x.json"]), false).is_err(), "a job filter is never removed here");
        remove_many(&home, "reranker", &files(&["zsep-rand-1.json"]), true, "test").unwrap();
        assert!(!d.join(ARCHIVE).join("zsep-rand-1.json").exists());
        let _ = std::fs::remove_dir_all(&home);
    }

    /// An import copies a checked map in, refuses a bad one and an existing one without replace,
    /// replaces with a backup, and each is undone.
    #[test]
    fn import_checks_replaces_and_undoes() {
        let home = std::env::temp_dir().join(format!("nm-pmaps-imp-{}", std::process::id()));
        let d = home.join("library/parameter-maps/writer");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::create_dir_all(home.join("memory")).unwrap();
        let src = home.join("tol-mine.json");
        std::fs::write(&src, r#"{"L0":{"low25":0.5}}"#).unwrap();
        let bad = home.join("job-filter-bad.json");
        std::fs::write(&bad, r#"{"parts":[{"target":"L1.x","gain":[1.0],"bias":[]}]}"#).unwrap();
        let r = import(&home, "writer", &src.display().to_string(), None, false, "test").unwrap();
        assert!(d.join("tol-mine.json").is_file() && matches!(r.get("replaced"), Ok(Value::Bool(false))));
        assert!(import(&home, "writer", &bad.display().to_string(), None, false, "test").is_err(), "gain and bias lengths differ");
        assert!(import(&home, "writer", &src.display().to_string(), None, false, "test").is_err(), "exists without replace");
        assert!(import(&home, "writer", "\\\\server\\share\\x.json", None, false, "test").is_err(), "no network paths");
        std::fs::write(&src, r#"{"L0":{"low25":0.75}}"#).unwrap();
        let r2 = import(&home, "writer", &src.display().to_string(), None, true, "test").unwrap();
        assert!(r2.get("backup").and_then(Value::string).is_ok());
        let log = std::fs::read_to_string(home.join("memory/audit.jsonl")).unwrap();
        let rows: Vec<Value> = log.lines().filter_map(|l| nmjson::parse(l.as_bytes()).ok()).collect();
        let id_of = |v: &Value| v.get("id").and_then(Value::string).unwrap().to_owned();
        undo_import(&home, &rows[1], &id_of(&rows[1]), "test").unwrap();
        assert!(std::fs::read_to_string(d.join("tol-mine.json")).unwrap().contains("0.5"), "the replaced map is back");
        undo_import(&home, &rows[0], &id_of(&rows[0]), "test").unwrap();
        assert!(!d.join("tol-mine.json").exists(), "the added map is moved aside");
        let _ = std::fs::remove_dir_all(&home);
    }

    /// A large value is listed by keys or paged, a small one comes whole, a path finds a layer;
    /// a save keeps the shape, refuses a changed one, keeps a backup and is undone from it.
    #[test]
    fn read_slices_and_save_with_backup_and_undo() {
        let home = std::env::temp_dir().join(format!("nm-pmaps-rw-{}", std::process::id()));
        let d = home.join("library/parameter-maps/reranker");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::create_dir_all(home.join("memory")).unwrap();
        std::fs::write(d.join("pmap-x.json"), r#"{"jobs": {"drone": {"layers": {"L0": {"attn_in": {"sq": [1.5, 2]}}, "L3": {"ffn": {"sq": [3]}}}}}}"#).unwrap();
        std::fs::write(d.join("job-filter-x.json"), r#"{"job":"x","enabled":true,"parts":[{"target":"L3.swigate","gain":[1.0,1.0],"bias":[0.0,0.0]}]}"#).unwrap();
        std::fs::write(d.join("surgery-ledger.jsonl"), "{\"op\":\"a\"}\n{\"op\":\"b\"}\n").unwrap();
        let r = read(&home, "role=reranker&file=pmap-x.json&section=jobs/drone/layers&layer=3").unwrap();
        assert_eq!(r.get("mode").and_then(Value::string).unwrap(), "value");
        assert!(nmjson::pretty::pretty(&r).contains("\"ffn\""));
        let full = read(&home, "role=reranker&file=job-filter-x.json").unwrap();
        assert_eq!(full.get("mode").and_then(Value::string).unwrap(), "full");
        assert!(matches!(full.get("editable"), Ok(Value::Bool(true))));
        let rows = read(&home, "role=reranker&file=surgery-ledger.jsonl&offset=1").unwrap();
        assert_eq!(rows.get("rows").and_then(Value::array).unwrap().len(), 1);
        assert!(read(&home, "role=reranker&file=pmap-x.json&section=jobs/nope").is_err());
        assert!(read(&home, "role=reranker&file=../x").is_err());
        // A spliced span finds each member and element.
        let b = br#" {"a": [1, {"b": "x}]"}], "c": null} "#;
        let m = members(b, 1).unwrap();
        assert_eq!(m.iter().map(|x| x.0.as_str()).collect::<Vec<_>>(), ["a", "c"]);
        assert_eq!(elements(b, m[0].1).unwrap().len(), 2);
        // Saves.
        let gain = nmjson::parse(b"[1.25, 0.5]").unwrap();
        let ok = save(&home, "reranker", "job-filter-x.json", Some("parts/0/gain"), None, &gain, "test").unwrap();
        let backup = ok.get("backup").and_then(Value::string).unwrap().to_owned();
        assert!(home.join(&backup).is_file());
        assert!(std::fs::read_to_string(d.join("job-filter-x.json")).unwrap().contains("1.25"));
        let short = nmjson::parse(b"[1.0]").unwrap();
        assert!(save(&home, "reranker", "job-filter-x.json", Some("parts/0/gain"), None, &short, "test").is_err(), "a gain keeps its length");
        let text = nmjson::parse(b"[\"a\", 1]").unwrap();
        assert!(save(&home, "reranker", "job-filter-x.json", Some("parts/0/gain"), None, &text, "test").is_err(), "numbers only");
        let target = nmjson::parse(br#"{"target":"L4.swigate","gain":[1.0,1.0],"bias":[0.0,0.0]}"#).unwrap();
        assert!(save(&home, "reranker", "job-filter-x.json", Some("parts/0"), None, &target, "test").is_err(), "a target stays");
        assert!(save(&home, "reranker", "pmap-x.json", None, None, &gain, "test").is_err(), "measurements are not edited");
        let log = std::fs::read_to_string(home.join("memory/audit.jsonl")).unwrap();
        let row = log.lines().filter_map(|l| nmjson::parse(l.as_bytes()).ok()).find(|v| v.get("kind").and_then(Value::string).is_ok_and(|k| k == "parameter-map")).unwrap();
        let id = row.get("id").and_then(Value::string).unwrap().to_owned();
        undo_save(&home, &row, &id, "test").unwrap();
        assert!(!std::fs::read_to_string(d.join("job-filter-x.json")).unwrap().contains("1.25"), "the backup is back");
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn kinds_and_summaries() {
        assert_eq!(kind_of("pmap-owl-45.json").0, "parameter map");
        assert_eq!(kind_of("job-filter-read.json").0, "job filter");
        let home = std::env::temp_dir().join(format!("nm-pmaps-{}", std::process::id()));
        let d = home.join("library/parameter-maps/writer");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("act-20.json"), r#"{"off": {"L0": [1, 2], "L3": [4]}}"#).unwrap();
        std::fs::write(d.join("pmap-x.json"), r#"{"jobs": {"note": {"layers": {"L0": {"attn_in": {}}, "L1": {"attn_in": {}}}}}}"#).unwrap();
        // The key scan a large map gets gives what the full parse gives.
        for f in ["act-20.json", "pmap-x.json"] {
            let full = nmjson::pretty::pretty(&summary(&d.join(f), f));
            let scanned = nmjson::pretty::pretty(&summary_scanned(&d.join(f), f));
            assert_eq!(full, scanned, "{f}");
        }
        let v = view(&home);
        let text = nmjson::pretty::pretty(&v);
        assert!(text.contains("\"channels_off\": 3"), "{text}");
        assert!(text.contains("\"attn_in\""), "{text}");
        assert!(remove(&home, "writer", "job-filter-x.json", "test").is_err());
        assert!(remove(&home, "writer", "../x", "test").is_err());
        assert!(remove(&home, "writer", "act-20.json", "test").is_ok() || !d.join("act-20.json").exists());
        let _ = std::fs::remove_dir_all(&home);
    }
}
