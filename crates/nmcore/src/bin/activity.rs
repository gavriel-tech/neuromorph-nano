//! Real activity per component, for the interface (`/api/activity`): built from the tool rows the
//! engine already writes to `memory/audit.jsonl`, nothing else. For a time window it gives, for
//! every anatomy component, how many calls used it, how many steps it ran, the time its timed
//! steps spent (total and median per call), how many steps had no clock, and when it last ran;
//! and the same per sub-part where the steps name one (a tentacle's drone, pipeline and working
//! memory, a MicroModel's layers, the cells a step touched, the anatomy children of a component).
//! Nothing is estimated: a step without a clock counts as a use with no time. The memory tier
//! counts of each memory folder (`tiers`) are read from the tier files. Standard library and
//! nmjson only.
use nmjson::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;

/// The window when none (or a value that does not read) is given.
const DEFAULT_WINDOW: &str = "1h";
/// The longest window served.
const MAX_WINDOW_MS: u64 = 7 * 24 * 3600 * 1000;
/// Bytes read from the end of the audit log per block when looking back over a window.
const BLOCK: u64 = 256 * 1024;
/// A call's rows carry its start time and may be written up to this long after it (the longest
/// call), so the log is read this much further back than the window.
const LATE_MS: u64 = 10 * 60 * 1000;
/// A result is reused while the log is unchanged and it is younger than this.
const REUSE_MS: u64 = 3000;

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}
/// Microseconds as milliseconds with three decimals.
fn ms_of(us: u64) -> Value {
    Value::Number(format!("{}.{:03}", us / 1000, us % 1000))
}

/// A window such as `15m`, `1h`, `24h` or `7d`, in milliseconds.
fn window_ms(w: &str) -> Option<u64> {
    let (num, unit) = w.split_at(w.len().checked_sub(1)?);
    let k: u64 = num.parse().ok().filter(|k| *k > 0)?;
    let per = match unit {
        "m" => 60_000,
        "h" => 3_600_000,
        "d" => 86_400_000,
        _ => return None,
    };
    k.checked_mul(per).filter(|m| *m <= MAX_WINDOW_MS)
}

/// One step as read back from an audit row.
struct StepRow {
    part: String,
    /// Absolute start (call start plus `start_ms`), or the call's start when not measured.
    at: u64,
    /// Microseconds spent (`us`, else `ms` times 1000), when the step was timed.
    us: Option<u64>,
    layers: Option<(u64, u64)>,
    job: Option<String>,
    cells: Vec<String>,
}

fn steps_of(row: &Value, time: u64) -> Vec<StepRow> {
    row.get("steps")
        .and_then(Value::array)
        .unwrap_or_default()
        .iter()
        .filter_map(|st| {
            let part = st.get("part").and_then(Value::string).ok()?.to_owned();
            let start = st.get("start_ms").and_then(Value::integer).ok();
            let us = st.get("us").and_then(Value::integer).ok().or_else(|| st.get("ms").and_then(Value::integer).ok().map(|m| m.saturating_mul(1000)));
            let layers = st.get("layers").and_then(Value::array).ok().and_then(|l| {
                let a = l.first()?.integer().ok()?;
                let b = l.get(1)?.integer().ok()?;
                Some((a, b))
            });
            let cells = st
                .get("cells")
                .and_then(Value::array)
                .unwrap_or_default()
                .iter()
                .filter_map(|c| c.string().ok().map(str::to_owned))
                .collect();
            Some(StepRow {
                part,
                at: time.saturating_add(start.unwrap_or(0)),
                us,
                layers,
                job: st.get("job").and_then(Value::string).ok().map(str::to_owned),
                cells,
            })
        })
        .collect()
}

/// Counts for one component, part or sub-part.
#[derive(Default, Clone)]
struct Acc {
    /// Per call (audit id): microseconds of its timed steps here.
    per_call: BTreeMap<String, u64>,
    calls: BTreeSet<String>,
    steps: u64,
    untimed: u64,
    total_us: u64,
    last_ms: u64,
    layers: BTreeMap<u64, Box<Acc>>,
    sub: BTreeMap<String, Acc>,
    jobs: BTreeMap<String, u64>,
}
impl Acc {
    fn add(&mut self, call: &str, st: &StepRow) {
        self.calls.insert(call.to_owned());
        self.steps += 1;
        match st.us {
            Some(u) => {
                self.total_us = self.total_us.saturating_add(u);
                *self.per_call.entry(call.to_owned()).or_default() += u;
            }
            None => self.untimed += 1,
        }
        self.last_ms = self.last_ms.max(st.at);
    }
    fn median_us(&self) -> Option<u64> {
        let mut v: Vec<u64> = self.per_call.values().copied().collect();
        v.sort_unstable();
        let k = v.len();
        if k == 0 {
            return None;
        }
        let (a, b) = (v.get((k - 1) / 2).copied().unwrap_or(0), v.get(k / 2).copied().unwrap_or(0));
        Some(a / 2 + b / 2 + (a % 2 + b % 2) / 2)
    }
    fn value(&self, deep: bool) -> Value {
        let mut o = BTreeMap::new();
        o.insert("calls".to_owned(), n(self.calls.len() as u64));
        o.insert("steps".to_owned(), n(self.steps));
        o.insert("timed_steps".to_owned(), n(self.steps - self.untimed));
        o.insert("untimed_steps".to_owned(), n(self.untimed));
        let timed = self.steps > self.untimed;
        o.insert("total_ms".to_owned(), if timed { ms_of(self.total_us) } else { Value::Null });
        o.insert("median_ms".to_owned(), self.median_us().map_or(Value::Null, ms_of));
        o.insert("last_ms".to_owned(), if self.steps > 0 { n(self.last_ms) } else { Value::Null });
        if deep {
            if !self.layers.is_empty() {
                o.insert("layers".to_owned(), Value::Object(self.layers.iter().map(|(l, a)| (l.to_string(), a.value(false))).collect()));
            }
            if !self.sub.is_empty() {
                o.insert("sub".to_owned(), Value::Object(self.sub.iter().map(|(k, a)| (k.clone(), a.value(false))).collect()));
            }
            if !self.jobs.is_empty() {
                o.insert("jobs".to_owned(), Value::Object(self.jobs.iter().map(|(k, c)| (k.clone(), n(*c))).collect()));
            }
        }
        Value::Object(o)
    }
}

/// The anatomy's nodes as (id, parent).
fn anatomy(home: &Path) -> Vec<(String, Option<String>)> {
    std::fs::read(home.join("designs").join("anatomy.json"))
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .map(|v| {
            v.get("nodes")
                .and_then(Value::array)
                .unwrap_or_default()
                .iter()
                .filter_map(|x| {
                    let id = x.get("id").and_then(Value::string).ok()?.to_owned();
                    Some((id, x.get("parent").and_then(Value::string).ok().map(str::to_owned)))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The anatomy component a step's part belongs to, and the sub-part it names when it is not the
/// component itself: the component itself; a part of a tentacle or drone instance
/// (`nmpod.pipeline#t0`, `nmmem.working#t0`, `nmpod.pipeline#t0-emit`) under that tentacle or
/// drone; a part without instance (`nmbrain.synthesis`) under the one component of that name.
fn component_of(part: &str, ids: &BTreeSet<String>) -> Option<(String, Option<String>)> {
    if ids.contains(part) {
        return Some((part.to_owned(), None));
    }
    let (base, inst) = match part.split_once('#') {
        Some((b, i)) => (b, Some(i)),
        None => (part, None),
    };
    if let Some(i) = inst {
        let head = i.split('-').next().unwrap_or(i);
        let unit = format!("nmpod.unit#{head}");
        let numbered = head.len() > 1 && head.starts_with(['t', 'd']) && head.get(1..).is_some_and(|d| d.bytes().all(|b| b.is_ascii_digit()));
        if numbered && ids.contains(&unit) {
            return Some((unit, Some(part.to_owned())));
        }
    }
    let prefix = format!("{base}#");
    let mut named = ids.iter().filter(|id| id.starts_with(&prefix));
    match (named.next(), named.next()) {
        (Some(one), None) => Some((one.clone(), inst.is_some().then(|| part.to_owned()))),
        _ => None,
    }
}

/// The audit log's lines that may hold rows of calls started at or after `from`: read back from
/// the end in blocks until a whole line older than `from - LATE_MS` is reached. Also the log's
/// length, for reuse.
fn tail_lines(path: &Path, from: u64) -> (Vec<String>, u64) {
    let Ok(mut f) = std::fs::File::open(path) else { return (Vec::new(), 0) };
    let len = f.metadata().map_or(0, |m| m.len());
    let floor = from.saturating_sub(LATE_MS);
    let mut pos = len;
    let mut buf: Vec<u8> = Vec::new();
    loop {
        let start = pos.saturating_sub(BLOCK);
        let mut block = vec![0_u8; usize::try_from(pos - start).unwrap_or(0)];
        if f.seek(SeekFrom::Start(start)).is_err() || f.read_exact(&mut block).is_err() {
            break;
        }
        block.extend_from_slice(&buf);
        buf = block;
        pos = start;
        if pos == 0 {
            break;
        }
        // The first line may be cut; the next whole line's time decides whether to read on.
        let text = String::from_utf8_lossy(&buf);
        let older = text.lines().skip(1).filter_map(time_of).next().is_some_and(|t| t < floor);
        if older {
            break;
        }
    }
    let text = String::from_utf8_lossy(&buf).into_owned();
    let skip = usize::from(pos > 0);
    (text.lines().skip(skip).map(str::to_owned).collect(), len)
}

/// The `time_ms` of a log line, read without parsing the whole row.
fn time_of(line: &str) -> Option<u64> {
    let at = line.find("\"time_ms\":")? + "\"time_ms\":".len();
    let digits: String = line.get(at..)?.chars().skip_while(|c| *c == ' ').take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

type Cached = (u64, String, u64, Value);
static LAST: Mutex<Option<Cached>> = Mutex::new(None);

/// `/api/activity[?window=1h|24h|<n>m|<n>h|<n>d][&project=<id>]`.
#[must_use]
pub fn view(home: &Path, query: &str) -> Value {
    let asked = crate::project::query_arg(query, "window").filter(|w| window_ms(w).is_some()).unwrap_or_else(|| DEFAULT_WINDOW.to_owned());
    let project = crate::project::query_arg(query, "project").filter(|p| !p.is_empty());
    let path = home.join("memory").join("audit.jsonl");
    let len = std::fs::metadata(&path).map_or(0, |m| m.len());
    let key = format!("{asked}|{}", project.as_deref().unwrap_or(""));
    let now = nmtype::now_ms();
    if let Ok(g) = LAST.lock()
        && let Some((l, k, at, v)) = g.as_ref()
        && *l == len
        && *k == key
        && now.saturating_sub(*at) < REUSE_MS
    {
        return v.clone();
    }
    let span = window_ms(&asked).unwrap_or(3_600_000);
    let from = now.saturating_sub(span);
    let (lines, len) = tail_lines(&path, from);
    let v = build(&anatomy(home), &lines, &asked, span, from, now, project.as_deref());
    if let Ok(mut g) = LAST.lock() {
        *g = Some((len, key, now, v.clone()));
    }
    v
}

#[expect(clippy::too_many_arguments, reason = "the anatomy, the log lines, the window and its bounds, the time now and the project filter")]
fn build(nodes: &[(String, Option<String>)], lines: &[String], window: &str, span: u64, from: u64, now: u64, project: Option<&str>) -> Value {
    // Tool rows by id, the last writing of each kept (a running row, then its end).
    let mut rows: BTreeMap<String, Value> = BTreeMap::new();
    for l in lines.iter().filter(|l| l.contains("\"kind\":\"tool\"")) {
        let Ok(v) = nmjson::parse(l.as_bytes()) else { continue };
        let Ok(id) = v.get("id").and_then(Value::string).map(str::to_owned) else { continue };
        rows.insert(id, v);
    }
    let ids: BTreeSet<String> = nodes.iter().map(|(i, _)| i.clone()).collect();
    let mut comps: BTreeMap<String, Acc> = ids.iter().map(|i| (i.clone(), Acc::default())).collect();
    let mut parts: BTreeMap<String, Acc> = BTreeMap::new();
    let (mut done, mut running, mut stale, mut timed, mut untimed) = (0_u64, 0_u64, 0_u64, 0_u64, 0_u64);
    let mut by_tool: BTreeMap<String, u64> = BTreeMap::new();
    for (id, row) in &rows {
        let time = row.get("time_ms").and_then(Value::integer).unwrap_or(0);
        if time < from || project.is_some_and(|p| row.get("project").and_then(Value::string).ok() != Some(p)) {
            continue;
        }
        let state = row.get("state").and_then(Value::string).unwrap_or("done");
        let old = now.saturating_sub(time) >= LATE_MS;
        match state {
            "running" if !old => {
                running += 1;
                continue;
            }
            "running" | "stale" => stale += 1,
            _ => done += 1,
        }
        if let Ok(t) = row.get("tool").and_then(Value::string) {
            *by_tool.entry(t.to_owned()).or_default() += 1;
        }
        for st in steps_of(row, time) {
            if st.us.is_some() {
                timed += 1;
            } else {
                untimed += 1;
            }
            let p = parts.entry(st.part.clone()).or_default();
            p.add(id, &st);
            if let Some((a, b)) = st.layers {
                for l in a..=b.min(a + 1024) {
                    p.layers.entry(l).or_default().add(id, &st);
                }
            }
            let Some((comp, sub)) = component_of(&st.part, &ids) else { continue };
            let c = comps.entry(comp).or_default();
            c.add(id, &st);
            if let Some(sp) = sub {
                c.sub.entry(sp).or_default().add(id, &st);
            }
            for cell in &st.cells {
                c.sub.entry(cell.clone()).or_default().add(id, &st);
            }
            if let Some((a, b)) = st.layers {
                for l in a..=b.min(a + 1024) {
                    c.layers.entry(l).or_default().add(id, &st);
                }
            }
            if let Some(j) = &st.job {
                *c.jobs.entry(j.clone()).or_default() += 1;
            }
        }
    }
    // Each component also lists its anatomy children that ran, with their own counts.
    let own: BTreeMap<String, Acc> = comps.clone();
    for (child, parent) in nodes {
        if let (Some(p), Some(a)) = (parent, own.get(child))
            && a.steps > 0
            && let Some(pa) = comps.get_mut(p)
        {
            let mut copy = a.clone();
            copy.layers.clear();
            copy.sub.clear();
            pa.sub.insert(child.clone(), copy);
        }
    }
    let mut o = BTreeMap::new();
    o.insert("window".to_owned(), s(window));
    o.insert("window_ms".to_owned(), n(span));
    o.insert("from_ms".to_owned(), n(from));
    o.insert("to_ms".to_owned(), n(now));
    o.insert("project".to_owned(), project.map_or(Value::Null, s));
    o.insert("calls".to_owned(), n(done + stale));
    o.insert("done".to_owned(), n(done));
    o.insert("stale".to_owned(), n(stale));
    o.insert("running".to_owned(), n(running));
    o.insert("timed_steps".to_owned(), n(timed));
    o.insert("untimed_steps".to_owned(), n(untimed));
    o.insert("tools".to_owned(), Value::Object(by_tool.into_iter().map(|(k, c)| (k, n(c))).collect()));
    o.insert("components".to_owned(), Value::Object(comps.iter().map(|(k, a)| (k.clone(), a.value(true))).collect()));
    o.insert("parts".to_owned(), Value::Object(parts.iter().map(|(k, a)| (k.clone(), a.value(true))).collect()));
    Value::Object(o)
}

/// Records held per tier in one memory folder, read from its tier files: `core` (items in
/// `1-core/items.json`), `active` and `archive` (items in every topic folder's `items.json` under
/// `2-active/` and `3-archive/`), `temporary` (notes in `temp/notes.json`). A tier with no file
/// counts 0.
#[must_use]
pub fn tiers(dir: &Path) -> Value {
    let count = |f: &Path, key: &str| -> u64 {
        std::fs::read(f)
            .ok()
            .and_then(|b| nmjson::parse(&b).ok())
            .map_or(0, |v| v.get(key).and_then(Value::array).map_or(0, |a| a.len() as u64))
    };
    let level = |name: &str| -> u64 {
        let base = dir.join(name);
        let mut c = count(&base.join("items.json"), "items");
        if let Ok(rd) = std::fs::read_dir(&base) {
            c += rd.flatten().filter(|e| e.path().is_dir()).map(|e| count(&e.path().join("items.json"), "items")).sum::<u64>();
        }
        c
    };
    let o: BTreeMap<String, Value> = [
        ("core", n(count(&dir.join("1-core").join("items.json"), "items"))),
        ("active", n(level("2-active"))),
        ("archive", n(level("3-archive"))),
        ("temporary", n(count(&dir.join("temp").join("notes.json"), "notes"))),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v))
    .collect();
    Value::Object(o)
}

/// Item ids listed in more than one tier file of one memory folder (memory revisions 1 and 2 both
/// write the tier files; an item they place in different tiers is listed in both).
#[must_use]
pub fn tiers_overlap(dir: &Path) -> u64 {
    let ids = |f: &Path| -> BTreeSet<String> {
        std::fs::read(f)
            .ok()
            .and_then(|b| nmjson::parse(&b).ok())
            .map(|v| {
                v.get("items")
                    .and_then(Value::array)
                    .unwrap_or_default()
                    .iter()
                    .filter_map(|x| x.get("id").and_then(Value::string).ok().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut per_tier: Vec<BTreeSet<String>> = Vec::new();
    for name in ["1-core", "2-active", "3-archive"] {
        let base = dir.join(name);
        let mut set = ids(&base.join("items.json"));
        if let Ok(rd) = std::fs::read_dir(&base) {
            for e in rd.flatten().filter(|e| e.path().is_dir()) {
                set.extend(ids(&e.path().join("items.json")));
            }
        }
        per_tier.push(set);
    }
    let mut seen: BTreeMap<String, u8> = BTreeMap::new();
    for set in &per_tier {
        for id in set {
            *seen.entry(id.clone()).or_default() += 1;
        }
    }
    seen.values().filter(|c| **c > 1).count() as u64
}

/// Topic folders held in the active and archive tiers of one memory folder.
#[must_use]
pub fn tier_topics(dir: &Path) -> Value {
    let topics = |name: &str| std::fs::read_dir(dir.join(name)).map_or(0, |rd| rd.flatten().filter(|e| e.path().is_dir()).count() as u64);
    Value::Object([("active".to_owned(), n(topics("2-active"))), ("archive".to_owned(), n(topics("3-archive")))].into_iter().collect())
}

/// `/api/memory/tiers`: the tier counts of the Nano's own memory folder and of every project
/// memory folder, and their sums.
#[must_use]
pub fn memory_tiers(home: &Path) -> Value {
    let own = home.join("memory");
    let mut projects = BTreeMap::new();
    let mut overlap = BTreeMap::new();
    if let Ok(rd) = std::fs::read_dir(nmtype::project::projects_dir(home)) {
        for e in rd.flatten().filter(|e| e.path().is_dir()) {
            projects.insert(e.file_name().to_string_lossy().into_owned(), tiers(&e.path()));
            overlap.insert(e.file_name().to_string_lossy().into_owned(), n(tiers_overlap(&e.path())));
        }
    }
    let nano = tiers(&own);
    let mut total: BTreeMap<String, u64> = BTreeMap::new();
    for v in std::iter::once(&nano).chain(projects.values()) {
        for k in ["core", "active", "archive", "temporary"] {
            *total.entry(k.to_owned()).or_default() += v.get(k).and_then(Value::integer).unwrap_or(0);
        }
    }
    let mut o = BTreeMap::new();
    o.insert("nano".to_owned(), nano);
    o.insert("nano_topics".to_owned(), tier_topics(&own));
    o.insert("nano_overlap".to_owned(), n(tiers_overlap(&own)));
    o.insert("projects".to_owned(), Value::Object(projects));
    o.insert("projects_overlap".to_owned(), Value::Object(overlap));
    o.insert("total".to_owned(), Value::Object(total.into_iter().map(|(k, c)| (k, n(c))).collect()));
    Value::Object(o)
}

/// Section check (processor only): windows read, parts map to their components, and a small log
/// gives the right counts, medians, untimed steps, layers, cells, jobs and children.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    if window_ms("1h") != Some(3_600_000) || window_ms("24h") != Some(86_400_000) || window_ms("15m") != Some(900_000) {
        return Err("activity: window".into());
    }
    if window_ms("0h").is_some() || window_ms("9d").is_some() || window_ms("h").is_some() || window_ms("").is_some() {
        return Err("activity: a bad window must not read".into());
    }
    let ids: BTreeSet<String> =
        ["nmpod.unit#t0", "nmpod.unit#d0", "nmbrain.synthesis#b0", "nmbrain.slot#reranker", "nmmem.gate#m0"].map(str::to_owned).into_iter().collect();
    let cases = [
        ("nmpod.unit#t0", Some(("nmpod.unit#t0", None))),
        ("nmpod.pipeline#t0-emit", Some(("nmpod.unit#t0", Some("nmpod.pipeline#t0-emit")))),
        ("nmmem.working#t0", Some(("nmpod.unit#t0", Some("nmmem.working#t0")))),
        ("nmbrain.synthesis", Some(("nmbrain.synthesis#b0", None))),
        ("nmapi.assemble", None),
        ("nmpod.unit#t9", None),
    ];
    for (part, want) in cases {
        let got = component_of(part, &ids);
        let want = want.map(|(c, s)| (c.to_owned(), s.map(str::to_owned)));
        if got != want {
            return Err(format!("activity: {part} maps to {got:?}"));
        }
    }
    if time_of(r#"{"id":"a","kind":"tool","time_ms":1794777406738,"x":1}"#) != Some(1_794_777_406_738) {
        return Err("activity: time of a line".into());
    }
    let now = 10_000_000_u64;
    let lines: Vec<String> = [
        format!(r#"{{"id":"a","kind":"tool","tool":"nm_context_pack","time_ms":{},"state":"running","steps":[]}}"#, now - 5000),
        format!(
            r#"{{"id":"a","kind":"tool","tool":"nm_context_pack","time_ms":{},"state":"done","steps":[{{"part":"nmpod.unit#t0","start_ms":1,"ms":0,"us":300,"job":"semantic"}},{{"part":"nmpod.pipeline#t0","start_ms":2,"ms":1,"us":1200}},{{"part":"nmbrain.synthesis","start_ms":3,"ms":2,"cells":["nmbrain.arbiter#b0"]}},{{"part":"nmbrain.slot#reranker","ms":5,"layers":[0,1]}},{{"part":"nmapi.assemble","ms":null,"timed":false}}]}}"#,
            now - 5000
        ),
        format!(r#"{{"id":"b","kind":"tool","tool":"nm_status","time_ms":{},"state":"done","steps":[{{"part":"nmbrain.synthesis","start_ms":0,"ms":4}}]}}"#, now - 1000),
        format!(r#"{{"id":"c","kind":"tool","tool":"nm_status","time_ms":{},"state":"running","steps":[]}}"#, now - 10),
        format!(r#"{{"id":"d","kind":"tool","tool":"nm_status","time_ms":{},"state":"done","steps":[{{"part":"nmbrain.synthesis","ms":9}}]}}"#, now - 7_200_000),
    ]
    .into_iter()
    .collect();
    let nodes: Vec<(String, Option<String>)> = [
        ("nmpod.pod#p0", None),
        ("nmpod.unit#t0", Some("nmpod.pod#p0")),
        ("nmpod.unit#d0", Some("nmpod.unit#t0")),
        ("nmbrain.synthesis#b0", None),
        ("nmbrain.slot#reranker", None),
    ]
    .into_iter()
    .map(|(a, b)| (a.to_owned(), b.map(str::to_owned)))
    .collect();
    let v = build(&nodes, &lines, "1h", 3_600_000, now - 3_600_000, now, None);
    let get = |path: &[&str]| -> String {
        let mut x = &v;
        for k in path {
            match x.get(k) {
                Ok(y) => x = y,
                Err(_) => return "-".into(),
            }
        }
        match x {
            Value::Number(t) | Value::String(t) => t.clone(),
            Value::Null => "null".into(),
            other => other.encode().unwrap_or_default(),
        }
    };
    let checks = [
        (get(&["components", "nmpod.unit#t0", "sub", "nmpod.pipeline#t0", "total_ms"]), "1.200"),
        (get(&["components", "nmpod.unit#t0", "total_ms"]), "1.500"),
        (get(&["components", "nmpod.unit#t0", "jobs", "semantic"]), "1"),
        (get(&["components", "nmpod.unit#d0", "calls"]), "0"),
        (get(&["components", "nmpod.unit#d0", "last_ms"]), "null"),
        (get(&["components", "nmpod.pod#p0", "sub", "nmpod.unit#t0", "calls"]), "1"),
        (get(&["components", "nmbrain.synthesis#b0", "sub", "nmbrain.arbiter#b0", "calls"]), "1"),
        (get(&["components", "nmbrain.slot#reranker", "layers", "0", "steps"]), "1"),
        (get(&["calls"]), "2"),
        (get(&["running"]), "1"),
        (get(&["untimed_steps"]), "1"),
        (get(&["parts", "nmbrain.synthesis", "calls"]), "2"),
        (get(&["parts", "nmbrain.synthesis", "total_ms"]), "6.000"),
        (get(&["parts", "nmbrain.synthesis", "median_ms"]), "3.000"),
        (get(&["parts", "nmbrain.synthesis", "last_ms"]), &(now - 1000).to_string()),
        (get(&["parts", "nmpod.unit#t0", "total_ms"]), "0.300"),
        (get(&["parts", "nmapi.assemble", "total_ms"]), "null"),
        (get(&["parts", "nmapi.assemble", "untimed_steps"]), "1"),
        (get(&["parts", "nmbrain.slot#reranker", "layers", "1", "calls"]), "1"),
        (get(&["tools", "nm_status"]), "1"),
    ];
    if let Some((got, want)) = checks.iter().find(|(g, w)| g != w) {
        return Err(format!("activity: counts gave {got} where {want} was expected: {}", v.encode().unwrap_or_default()));
    }
    Ok(())
}
