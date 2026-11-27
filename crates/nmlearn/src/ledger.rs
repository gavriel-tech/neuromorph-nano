//! The decision ledger (2026-11-12, build step 2 of research file Solution 6): every choice the
//! Core Brain makes on the whole path is written with what it chose from, so that any setting can
//! be replayed against later verdicts ("every logged choice replays"). One line per decision in
//! `decisions.jsonl`; a verdict arrives later as its own line (`{"at","episode","correct"}`) and
//! is joined at read time by episode id. Append only; ids and numbers only (the question is kept
//! as its SHA-256, never its words).
//!
//! A decision records: the candidates in arrival order (chunk ids), the reranker's P(yes) per
//! candidate, the second look's read set and yes-or-no scores, the threshold used, the chunk
//! chosen, and the propensity of that choice under the policy that made it (1 for a fixed rule;
//! below 1 when a tuner explores), so replays can weight them correctly.
//!
//! Standalone signals (2026-11-12, the design's L1 signals, research file Solution 6: re-read
//! verification and cost, which need no human verdict): `look_verified`, one flag per chunk of
//! the second look, set when a fresh read of that chunk from the tree holds the question's
//! distinctive terms; `cost_ms`, the whole path's time to this decision. Lines written before
//! these fields read back with them empty and zero.
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::Path;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmlearn.ledger", FaultKind::Execution, m)
}

/// One Core Brain decision.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Decision {
    pub at: u64,
    pub episode: Option<String>,
    pub question_sha: String,
    pub version: u64,
    /// Candidates (chunk ids) in arrival order, and the reranker's P(yes) for each.
    pub candidates: Vec<usize>,
    pub rerank: Vec<f32>,
    /// The second look: chunks read together (winner first) and their yes-or-no scores.
    pub look: Vec<usize>,
    pub look_scores: Vec<f32>,
    pub threshold: f32,
    pub chosen: usize,
    pub propensity: f32,
    /// Re-read verification of each `look` chunk (same order); empty on older lines.
    pub look_verified: Vec<bool>,
    /// Time from the question to this decision, in milliseconds (0 on older lines).
    pub cost_ms: u64,
    /// Joined at read time from `use` lines: the agent went on to use the chosen chunk's location
    /// (`Some(true)`); `None` when no use was seen (not evidence of non-use).
    pub agent_used: Option<bool>,
    /// Joined at read time from verdict lines; `None` until a verdict arrives.
    pub correct: Option<bool>,
}

fn nums<T: ToString>(v: &[T]) -> Value {
    Value::Array(v.iter().map(|x| Value::Number(x.to_string())).collect())
}
fn read_us(v: &Value, k: &str) -> Vec<usize> {
    v.get(k)
        .and_then(Value::array)
        .map(|a| a.iter().filter_map(|x| x.integer().ok().and_then(|i| usize::try_from(i).ok())).collect())
        .unwrap_or_default()
}
fn read_fs(v: &Value, k: &str) -> Vec<f32> {
    v.get(k)
        .and_then(Value::array)
        .map(|a| a.iter().filter_map(|x| if let Value::Number(n) = x { n.parse().ok() } else { None }).collect())
        .unwrap_or_default()
}
fn read_f(v: &Value, k: &str) -> f32 {
    match v.get(k) {
        Ok(Value::Number(n)) => n.parse().unwrap_or(0.0),
        _ => 0.0,
    }
}

impl Decision {
    #[must_use]
    pub fn to_value(&self) -> Value {
        let s = |t: &str| Value::String(t.to_owned());
        Value::Object(
            [
                ("at", Value::Number(self.at.to_string())),
                ("kind", s("decision")),
                ("episode", self.episode.as_deref().map_or(Value::Null, s)),
                ("question_sha", s(&self.question_sha)),
                ("version", Value::Number(self.version.to_string())),
                ("candidates", nums(&self.candidates)),
                ("rerank", nums(&self.rerank)),
                ("look", nums(&self.look)),
                ("look_scores", nums(&self.look_scores)),
                // An infinite threshold (never move first place) is written as 1e9.
                ("threshold", Value::Number(if self.threshold.is_finite() { self.threshold.to_string() } else { "1e9".into() })),
                ("chosen", Value::Number(self.chosen.to_string())),
                ("propensity", Value::Number(self.propensity.to_string())),
                ("look_verified", Value::Array(self.look_verified.iter().map(|b| Value::Bool(*b)).collect())),
                ("cost_ms", Value::Number(self.cost_ms.to_string())),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect(),
        )
    }
    /// # Errors
    /// Not a decision line.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        if v.get("kind").and_then(Value::string).ok() != Some("decision") {
            return Err(bad("not a decision line"));
        }
        Ok(Self {
            at: v.get("at").and_then(Value::integer).ok().unwrap_or(0),
            episode: v.get("episode").and_then(Value::string).ok().map(str::to_owned),
            question_sha: v.get("question_sha").and_then(Value::string).map(str::to_owned).unwrap_or_default(),
            version: v.get("version").and_then(Value::integer).ok().unwrap_or(0),
            candidates: read_us(v, "candidates"),
            rerank: read_fs(v, "rerank"),
            look: read_us(v, "look"),
            look_scores: read_fs(v, "look_scores"),
            threshold: {
                let t = read_f(v, "threshold");
                if t >= 1e8 { f32::INFINITY } else { t }
            },
            chosen: v.get("chosen").and_then(Value::integer).ok().and_then(|x| usize::try_from(x).ok()).unwrap_or(0),
            propensity: read_f(v, "propensity"),
            look_verified: v
                .get("look_verified")
                .and_then(Value::array)
                .map(|a| a.iter().filter_map(|x| if let Value::Bool(b) = x { Some(*b) } else { None }).collect())
                .unwrap_or_default(),
            agent_used: None,
            cost_ms: v.get("cost_ms").and_then(Value::integer).ok().unwrap_or(0),
            correct: None,
        })
    }
    /// The re-read verification of a chunk of the second look (`None` when not recorded).
    #[must_use]
    pub fn verified(&self, chunk: usize) -> Option<bool> {
        if self.look_verified.len() != self.look.len() {
            return None;
        }
        self.look.iter().position(|c| *c == chunk).and_then(|i| self.look_verified.get(i).copied())
    }
    /// The chunk a different second-look threshold would have chosen (the replay of this one
    /// setting): the reranker's best, moved to the look's best only when its margin over the
    /// reranker's best reaches `t`.
    #[must_use]
    pub fn replay_look(&self, t: f32) -> usize {
        let first = self.look.first().copied().unwrap_or(self.chosen);
        let Some((bi, &bs)) = self.look_scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0))) else {
            return first;
        };
        let margin = bs - self.look_scores.first().copied().unwrap_or(bs);
        if bi != 0 && margin >= t { self.look.get(bi).copied().unwrap_or(first) } else { first }
    }
}

/// Append one decision.
/// # Errors
/// The file cannot be written.
pub fn append(path: &Path, d: &Decision) -> Result<(), Fault> {
    append_value(path, &d.to_value())
}
/// Append an agent-use line: the agent used a location the decision of `episode` cited, through
/// `tool` (2026-11-13, L1 signal "agent use").
/// # Errors
/// The file cannot be written.
pub fn append_use(path: &Path, episode: &str, location: &str, tool: &str, at: u64) -> Result<(), Fault> {
    let s = |t: &str| Value::String(t.to_owned());
    let v = Value::Object(
        [
            ("at".to_owned(), Value::Number(at.to_string())),
            ("kind".to_owned(), s("use")),
            ("episode".to_owned(), s(episode)),
            ("location".to_owned(), s(location)),
            ("tool".to_owned(), s(tool)),
        ]
        .into_iter()
        .collect(),
    );
    append_value(path, &v)
}
/// Append a verdict for an episode (joined to its decisions at read time).
/// # Errors
/// The file cannot be written.
pub fn append_verdict(path: &Path, episode: &str, correct: bool, at: u64) -> Result<(), Fault> {
    let v = Value::Object(
        [
            ("at".to_owned(), Value::Number(at.to_string())),
            ("kind".to_owned(), Value::String("verdict".into())),
            ("episode".to_owned(), Value::String(episode.to_owned())),
            ("correct".to_owned(), Value::Bool(correct)),
        ]
        .into_iter()
        .collect(),
    );
    append_value(path, &v)
}
fn append_value(path: &Path, v: &Value) -> Result<(), Fault> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| bad(e.to_string()))?;
    }
    let line = v.encode().map_err(|e| bad(e.to_string()))?;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
    f.write_all(format!("{line}\n").as_bytes()).map_err(|e| bad(e.to_string()))
}

/// Every decision of a ledger with its verdict joined (the last verdict for an episode wins).
/// A missing file is an empty ledger.
/// # Errors
/// An unreadable file.
pub fn load(path: &Path) -> Result<Vec<Decision>, Fault> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(bad(format!("{}: {e}", path.display()))),
    };
    let mut out = Vec::new();
    let mut verdicts: BTreeMap<String, bool> = BTreeMap::new();
    let mut uses: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(v) = nmjson::parse(line.as_bytes()) else { continue };
        match v.get("kind").and_then(Value::string).ok() {
            Some("decision") => out.push(Decision::from_value(&v)?),
            Some("use") => {
                if let Ok(ep) = v.get("episode").and_then(Value::string) {
                    uses.insert(ep.to_owned());
                }
            }
            Some("verdict") => {
                if let (Ok(ep), Ok(Value::Bool(c))) = (v.get("episode").and_then(Value::string), v.get("correct")) {
                    verdicts.insert(ep.to_owned(), *c);
                }
            }
            _ => {}
        }
    }
    for d in &mut out {
        d.correct = d.episode.as_ref().and_then(|e| verdicts.get(e).copied());
        d.agent_used = d.episode.as_ref().filter(|e| uses.contains(*e)).map(|_| true);
    }
    Ok(out)
}

/// Section check: a decision round-trips through its line; a verdict line joins by episode; the
/// replay of the threshold reproduces the stored choice at the stored threshold and moves or
/// keeps first place as the margin says.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let d = Decision {
        at: 5,
        episode: Some("ep1".into()),
        question_sha: "ab".into(),
        version: 3,
        candidates: vec![10, 11, 12],
        rerank: vec![0.9, 0.5, 0.25],
        look: vec![10, 12],
        look_scores: vec![5.0, 5.5],
        threshold: 0.205,
        chosen: 12,
        propensity: 1.0,
        look_verified: vec![false, true],
        cost_ms: 1234,
        agent_used: None,
        correct: None,
    };
    let back = Decision::from_value(&d.to_value()).map_err(|e| e.to_string())?;
    if back != d {
        return Err(format!("nmlearn.ledger: round trip {back:?}"));
    }
    let never = Decision { threshold: f32::INFINITY, ..d.clone() };
    if Decision::from_value(&never.to_value()).map(|x| x.threshold.is_infinite()) != Ok(true) {
        return Err("nmlearn.ledger: infinite threshold".into());
    }
    if d.replay_look(0.205) != 12 || d.replay_look(0.6) != 10 || d.verified(12) != Some(true) || d.verified(11).is_some() {
        return Err("nmlearn.ledger: replay of the threshold".into());
    }
    let dir = nmtype::check_dir("nmlearn-ledger").map_err(|e| e.to_string())?;
    let path = dir.join("decisions.jsonl");
    append(&path, &d).map_err(|e| e.to_string())?;
    append_verdict(&path, "ep1", true, 6).map_err(|e| e.to_string())?;
    append_use(&path, "ep1", "a.rs:3", "nm_view", 7).map_err(|e| e.to_string())?;
    let got = load(&path).map_err(|e| e.to_string())?;
    if got.len() != 1 || got.first().and_then(|x| x.correct) != Some(true) || got.first().and_then(|x| x.agent_used) != Some(true) {
        return Err(format!("nmlearn.ledger: join {got:?}"));
    }
    Ok(())
}
