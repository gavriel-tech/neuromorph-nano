//! Persistent identity (guide 7.1; PLAN 4c): one record per component and one for the
//! organism, with the ten fields the guide lists. It survives restart and reassembly: a
//! component keeps its identity when it is detached, remounted or moved to another structure.
use crate::store::{self, obj, s, u};
use nmjson::Value;
use nmtype::Fault;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    pub id: String,
    pub origin: String,
    pub created: u64,
    pub developmental_history: Vec<(u64, String)>,
    pub previous_configurations: Vec<u64>,
    pub parameter_associations: Vec<String>,
    pub training_history: Vec<String>,
    pub contributions: Vec<(String, String)>,
    pub dependencies: Vec<String>,
    pub version_history: Vec<u64>,
    pub state: String,
}

fn file(dir: &Path, id: &str) -> PathBuf {
    let safe: String = id.chars().map(|c| if c.is_alphanumeric() || c == '.' || c == '-' { c } else { '_' }).collect();
    dir.join("identity").join(format!("{safe}.json"))
}

impl Identity {
    /// Load the record for `id`, or start one with its origin.
    #[must_use]
    pub fn load_or_new(dir: &Path, id: &str, origin: &str) -> Self {
        store::get(&file(dir, id)).map_or_else(
            || Self {
                id: id.into(),
                origin: origin.into(),
                created: nmtype::now_ms(),
                developmental_history: Vec::new(),
                previous_configurations: Vec::new(),
                parameter_associations: Vec::new(),
                training_history: Vec::new(),
                contributions: Vec::new(),
                dependencies: Vec::new(),
                version_history: Vec::new(),
                state: "new".into(),
            },
            |v| Self::from_value(&v),
        )
    }
    /// Record a developmental event (mount, change, restart) at a snapshot version.
    pub fn event(&mut self, what: &str, version: u64) {
        self.developmental_history.push((nmtype::now_ms(), what.to_owned()));
        if self.version_history.last() != Some(&version) {
            if let Some(prev) = self.version_history.last() {
                self.previous_configurations.push(*prev);
            }
            self.version_history.push(version);
        }
    }
    /// Compact the developmental history (2026-11-12, memory revision 2, section 13): the session
    /// open and close lines of each UTC day become one line ("<n> sessions on run-day"), every other
    /// event is kept as it was, in order. Returns how many lines were folded away.
    pub fn compact_history(&mut self) -> usize {
        const DAY: u64 = 86_400_000;
        let before = self.developmental_history.len();
        let mut out: Vec<(u64, String)> = Vec::new();
        let mut day: Option<(u64, u64, u64)> = None; // (day, first time, sessions)
        let flush = |out: &mut Vec<(u64, String)>, d: Option<(u64, u64, u64)>| {
            if let Some((_, at, n)) = d {
                out.push((at, format!("{n} session{} on this run-day", if n == 1 { "" } else { "s" })));
            }
        };
        for (at, what) in std::mem::take(&mut self.developmental_history) {
            let session = what == "session open" || what == "session close" || what.ends_with("on this run-day");
            if !session {
                flush(&mut out, day.take());
                out.push((at, what));
                continue;
            }
            let opens = if what == "session open" {
                1
            } else if what == "session close" {
                0
            } else {
                what.split_whitespace().next().and_then(|n| n.parse().ok()).unwrap_or(0)
            };
            match day {
                Some((d, first, n)) if d == at / DAY => day = Some((d, first, n + opens)),
                _ => {
                    flush(&mut out, day.take());
                    day = Some((at / DAY, at, opens));
                }
            }
        }
        flush(&mut out, day);
        self.developmental_history = out;
        before.saturating_sub(self.developmental_history.len())
    }
    /// Fill a field from a real event, once per distinct value: `parameters` (populations and
    /// cartridges used), `training` (learned pieces kept, with their gate result), `dependencies`
    /// (parts it runs on).
    pub fn note(&mut self, field: &str, value: &str) {
        let list = match field {
            "parameters" => &mut self.parameter_associations,
            "training" => &mut self.training_history,
            _ => &mut self.dependencies,
        };
        if !list.iter().any(|x| x == value) {
            list.push(value.to_owned());
        }
    }
    /// A contribution (what, where): a memory verified right, for example.
    pub fn contribute(&mut self, what: &str, where_: &str) {
        if !self.contributions.iter().any(|(a, b)| a == what && b == where_) {
            self.contributions.push((what.to_owned(), where_.to_owned()));
        }
    }
    /// Save (the record for this id is replaced by its newer state; its history is inside it).
    /// # Errors
    /// The file could not be written.
    pub fn save(&self, dir: &Path) -> Result<(), Fault> {
        store::put_safe(&file(dir, &self.id), &self.to_value())
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let pairs = |v: &[(u64, String)]| Value::Array(v.iter().map(|(a, b)| Value::Array(vec![u(*a), s(b)])).collect());
        let strs = |v: &[String]| Value::Array(v.iter().map(|x| s(x)).collect());
        let nums = |v: &[u64]| Value::Array(v.iter().map(|x| u(*x)).collect());
        obj(vec![
            ("id", s(&self.id)),
            ("origin", s(&self.origin)),
            ("created", u(self.created)),
            ("developmental_history", pairs(&self.developmental_history)),
            ("previous_configurations", nums(&self.previous_configurations)),
            ("parameter_associations", strs(&self.parameter_associations)),
            ("training_history", strs(&self.training_history)),
            ("contributions", Value::Array(self.contributions.iter().map(|(a, b)| Value::Array(vec![s(a), s(b)])).collect())),
            ("dependencies", strs(&self.dependencies)),
            ("version_history", nums(&self.version_history)),
            ("state", s(&self.state)),
        ])
    }
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        let arr = |k: &str| v.get(k).and_then(Value::array).map(<[Value]>::to_vec).unwrap_or_default();
        let pair = |x: &Value| -> Option<(Value, Value)> {
            if let Value::Array(p) = x { Some((p.first()?.clone(), p.get(1)?.clone())) } else { None }
        };
        Self {
            id: store::text(v, "id"),
            origin: store::text(v, "origin"),
            created: store::num(v, "created"),
            developmental_history: arr("developmental_history")
                .iter()
                .filter_map(pair)
                .map(|(a, b)| (a.integer().unwrap_or(0), b.string().unwrap_or("").to_owned()))
                .collect(),
            previous_configurations: arr("previous_configurations").iter().filter_map(|x| x.integer().ok()).collect(),
            parameter_associations: store::strings(v, "parameter_associations"),
            training_history: store::strings(v, "training_history"),
            contributions: arr("contributions")
                .iter()
                .filter_map(pair)
                .map(|(a, b)| (a.string().unwrap_or("").to_owned(), b.string().unwrap_or("").to_owned()))
                .collect(),
            dependencies: store::strings(v, "dependencies"),
            version_history: arr("version_history").iter().filter_map(|x| x.integer().ok()).collect(),
            state: store::text(v, "state"),
        }
    }
}

/// Self-check of the `identity` sub-component: a record with events survives save and load
/// unchanged, and a version change keeps the previous configuration.
pub(crate) fn check() -> Result<(), String> {
    let dir = crate::scratch_dir("identity")?;
    let mut id = Identity::load_or_new(&dir, "nmpod.unit#t3", "designs/assemblies/forage-s2.json");
    id.event("mounted", 1);
    id.event("snapshot replaced", 2);
    id.contributions.push(("task 1".into(), "4 packets".into()));
    id.save(&dir).map_err(|e| e.to_string())?;
    let back = Identity::load_or_new(&dir, "nmpod.unit#t3", "other");
    if back != id || back.previous_configurations != [1] {
        return Err("nmmem.identity: record changed across save and load".into());
    }
    // Compaction: two days of session lines become two lines; a real event between them stays.
    let mut c = back;
    let day = 86_400_000_u64;
    c.developmental_history = vec![
        (10 * day + 1, "session open".into()),
        (10 * day + 2, "session close".into()),
        (10 * day + 3, "session open".into()),
        (10 * day + 4, "cartridge kept".into()),
        (11 * day + 1, "session open".into()),
        (11 * day + 2, "session close".into()),
    ];
    let folded = c.compact_history();
    let lines: Vec<&str> = c.developmental_history.iter().map(|(_, w)| w.as_str()).collect();
    if folded != 3 || lines != ["2 sessions on this run-day", "cartridge kept", "1 session on this run-day"] {
        return Err(format!("nmmem.identity: compaction gave {lines:?}"));
    }
    c.note("training", "judge cartridge q8 kept");
    c.note("training", "judge cartridge q8 kept");
    if c.training_history.len() != 1 {
        return Err("nmmem.identity: a field note was repeated".into());
    }
    Ok(())
}
