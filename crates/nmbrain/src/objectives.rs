//! Objectives (guide 3, 11.2, 18; PLAN 4c): an objective tree with sub-goals, statuses and
//! evidence; a rule decomposition (one sub-goal per clause); the objective kind that picks
//! the composition row; `next` gives the first open leaf. Persisted by the organism.
use nmjson::Value;
use nmtype::TaskKind;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Objective {
    pub id: u64,
    pub text: String,
    pub parent: Option<u64>,
    pub status: String,
    pub kind: TaskKind,
    pub evidence: Vec<String>,
}

/// The objective kind by rule (the composition table's key).
#[must_use]
pub fn classify(text: &str) -> TaskKind {
    let l = text.to_lowercase();
    let has = |xs: &[&str]| xs.iter().any(|x| l.contains(x));
    if has(&["build", "compile", "cargo error", "fails to build"]) {
        TaskKind::Build
    } else if has(&["verify", "check that", "test that", "regression"]) {
        TaskKind::Verify
    } else if has(&["improve", "refactor", "clean up", "simplify"]) {
        TaskKind::Improve
    } else if has(&["anatomy", "tentacle", "drone", "snapshot", "rewire"]) {
        TaskKind::Anatomy
    } else if has(&["understand", "how does", "explain", "architecture"]) {
        TaskKind::Understand
    } else {
        TaskKind::Context
    }
}

/// The tree.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Objectives {
    pub items: Vec<Objective>,
}
impl Objectives {
    /// Add an objective and its sub-goals (one per clause split on ";" or " and then ").
    pub fn add(&mut self, text: &str) -> u64 {
        let id = self.items.iter().map(|o| o.id).max().unwrap_or(0) + 1;
        self.items.push(Objective {
            id,
            text: text.to_owned(),
            parent: None,
            status: "open".into(),
            kind: classify(text),
            evidence: Vec::new(),
        });
        let clauses: Vec<&str> = text.split([';']).flat_map(|c| c.split(" and then ")).map(str::trim).filter(|c| !c.is_empty()).collect();
        if clauses.len() > 1 {
            for (i, c) in clauses.iter().enumerate() {
                let sid = id * 100 + i as u64 + 1;
                self.items.push(Objective {
                    id: sid,
                    text: (*c).to_owned(),
                    parent: Some(id),
                    status: "open".into(),
                    kind: classify(c),
                    evidence: Vec::new(),
                });
            }
        }
        id
    }
    /// The first open leaf, depth first in insertion order.
    #[must_use]
    pub fn next(&self) -> Option<&Objective> {
        self.items.iter().find(|o| o.status == "open" && !self.items.iter().any(|c| c.parent == Some(o.id) && c.status == "open"))
    }
    /// Mark an objective done with evidence; a parent closes when all its sub-goals have.
    pub fn done(&mut self, id: u64, evidence: &str) {
        if let Some(o) = self.items.iter_mut().find(|o| o.id == id) {
            o.status = "done".into();
            o.evidence.push(evidence.to_owned());
        }
        let parent = self.items.iter().find(|o| o.id == id).and_then(|o| o.parent);
        if let Some(p) = parent
            && self.items.iter().filter(|o| o.parent == Some(p)).all(|o| o.status == "done")
            && let Some(po) = self.items.iter_mut().find(|o| o.id == p)
        {
            po.status = "done".into();
        }
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let s = |t: &str| Value::String(t.to_owned());
        Value::Array(
            self.items
                .iter()
                .map(|o| {
                    Value::Object(
                        [
                            ("id".to_owned(), Value::Number(o.id.to_string())),
                            ("objective".to_owned(), s(&o.text)),
                            ("parent".to_owned(), o.parent.map_or(Value::Null, |p| Value::Number(p.to_string()))),
                            ("status".to_owned(), s(&o.status)),
                            ("kind".to_owned(), s(o.kind.name())),
                            ("evidence".to_owned(), Value::Array(o.evidence.iter().map(|e| s(e)).collect())),
                        ]
                        .into_iter()
                        .collect(),
                    )
                })
                .collect(),
        )
    }
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        let items = v
            .array()
            .map(|a| {
                a.iter()
                    .map(|o| {
                        let text = o.get("objective").and_then(Value::string).unwrap_or("").to_owned();
                        Objective {
                            id: o.get("id").and_then(Value::integer).unwrap_or(0),
                            parent: o.get("parent").and_then(Value::integer).ok(),
                            status: o.get("status").and_then(Value::string).unwrap_or("open").to_owned(),
                            kind: o
                                .get("kind")
                                .and_then(Value::string)
                                .ok()
                                .and_then(TaskKind::from_name)
                                .unwrap_or_else(|| classify(&text)),
                            evidence: o
                                .get("evidence")
                                .and_then(Value::array)
                                .map(|e| e.iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect())
                                .unwrap_or_default(),
                            text,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self { items }
    }
}

/// Self-check of the `objectives` sub-component: decomposition, kinds, next, closing a parent,
/// round trip.
pub(crate) fn check() -> Result<(), String> {
    let mut t = Objectives::default();
    let id = t.add("map the context crates; then verify that the drone crates build");
    if t.items.len() != 3 || t.items.get(2).map(|o| o.kind) != Some(TaskKind::Build) {
        return Err(format!("nmbrain.objectives: decomposition {:?}", t.items.iter().map(|o| (&o.text, o.kind)).collect::<Vec<_>>()));
    }
    let first = t.next().map(|o| o.id).ok_or("nmbrain.objectives: no next")?;
    t.done(first, "a.rs:1");
    let second = t.next().map(|o| o.id).ok_or("nmbrain.objectives: no second")?;
    t.done(second, "b.rs:2");
    if t.items.iter().find(|o| o.id == id).map(|o| o.status.as_str()) != Some("done") || Objectives::from_value(&t.to_value()) != t {
        return Err("nmbrain.objectives: parent not closed or round trip".into());
    }
    Ok(())
}
