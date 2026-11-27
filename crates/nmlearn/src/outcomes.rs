//! The checked-output log (PLAN 4c `outcomes`; guide 10.6): `units/outcomes.jsonl`, one line
//! per outcome `{at, slot, task, query, answer, correct, tokens, verified}`, append only. Only
//! outcomes marked correct (and, for a fit, verified) are training data; nothing else is.
use crate::unit::{n, obj, s};
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::io::Write as _;
use std::path::Path;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmlearn.outcomes", FaultKind::Resource, m)
}

/// One checked outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    pub at: u64,
    pub slot: String,
    pub task: String,
    pub query: String,
    pub answer: String,
    pub correct: bool,
    pub tokens: u64,
    /// True when the answer is ground truth by construction (a definition line for its
    /// constant), false when it is a verdict given later through `nm_feedback`.
    pub verified: bool,
}
impl Outcome {
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("at", n(self.at)),
            ("slot", s(&self.slot)),
            ("task", s(&self.task)),
            ("query", s(&self.query)),
            ("answer", s(&self.answer)),
            ("correct", Value::Bool(self.correct)),
            ("tokens", n(self.tokens)),
            ("verified", Value::Bool(self.verified)),
        ])
    }
    /// # Errors
    /// A line without the required fields.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let g = |k: &str| v.get(k).and_then(Value::string).map(str::to_owned).map_err(|e| bad(format!("{k}: {e}")));
        let u = |k: &str| v.get(k).and_then(Value::integer).unwrap_or(0);
        Ok(Self {
            at: u("at"),
            slot: g("slot")?,
            task: g("task")?,
            query: g("query")?,
            answer: g("answer")?,
            correct: matches!(v.get("correct"), Ok(Value::Bool(true))),
            tokens: u("tokens"),
            verified: matches!(v.get("verified"), Ok(Value::Bool(true))),
        })
    }
}

fn path(home: &Path) -> std::path::PathBuf {
    home.join("units").join("outcomes.jsonl")
}

/// Append one outcome.
/// # Errors
/// The log cannot be written.
pub fn append(home: &Path, o: &Outcome) -> Result<(), Fault> {
    let p = path(home);
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).map_err(|e| bad(e.to_string()))?;
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&p).map_err(|e| bad(e.to_string()))?;
    writeln!(f, "{}", o.to_value().encode().map_err(|e| bad(e.to_string()))?).map_err(|e| bad(e.to_string()))
}

/// Every outcome logged at or after `since` (0 for all), in log order.
#[must_use]
pub fn since(home: &Path, since: u64) -> Vec<Outcome> {
    std::fs::read_to_string(path(home))
        .map(|t| {
            t.lines()
                .filter_map(|l| nmjson::parse(l.as_bytes()).ok())
                .filter_map(|v| Outcome::from_value(&v).ok())
                .filter(|o| o.at >= since)
                .collect()
        })
        .unwrap_or_default()
}

/// The training candidates among `outcomes`: correct and, when `verified_only`, verified.
#[must_use]
pub fn training(outcomes: &[Outcome], verified_only: bool) -> Vec<&Outcome> {
    outcomes.iter().filter(|o| o.correct && (!verified_only || o.verified)).collect()
}

/// Self-check of `outcomes`: append two lines, read them back since a time, and only the
/// correct verified one is training data.
pub(crate) fn check() -> Result<(), String> {
    let home = nmtype::check_dir("nmlearn-outcomes").map_err(|e| e.to_string())?;
    let a = Outcome {
        at: 10,
        slot: "reranker".into(),
        task: "t1".into(),
        query: "q".into(),
        answer: "a".into(),
        correct: true,
        tokens: 5,
        verified: true,
    };
    let b = Outcome {
        at: 20,
        slot: "reranker".into(),
        task: "t2".into(),
        query: "q2".into(),
        answer: "b".into(),
        correct: false,
        tokens: 6,
        verified: false,
    };
    append(&home, &a).map_err(|e| e.to_string())?;
    append(&home, &b).map_err(|e| e.to_string())?;
    let all = since(&home, 0);
    if all != vec![a.clone(), b] || since(&home, 15).len() != 1 || training(&all, true) != vec![&a] {
        return Err("nmlearn.outcomes: log round trip".into());
    }
    Ok(())
}
