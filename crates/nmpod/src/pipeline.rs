//! Local processing in every unit (guide 4.2, 12): reflex check, confidence by reader
//! agreement, anomaly flag, a working-memory note, and distillation into one cited packet
//! per claim key. The reflex table is data (`designs/reflexes.json`, PLAN 4d item 2):
//! four-field rows, in order, first match wins, no match means keep.
use crate::tip::Finding;
use nmjson::Value;
use nmtype::{Category, ComponentId, Fault, FaultKind, Packet, Provenance, Stance, Status};
use std::collections::BTreeMap;

/// What a reflex decides about one finding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reflex {
    Keep,
    Drop,
    Escalate(String),
    Hold(String),
}

/// One reflex row.
#[derive(Clone, Debug, PartialEq)]
pub struct ReflexRow {
    pub level: String,
    pub field: String,
    pub op: String,
    pub value: Value,
    pub then: String,
    pub reason: String,
}

/// The reflex table shared by drones, tentacles and the brain.
#[derive(Clone, Debug, Default)]
pub struct ReflexTable {
    pub rows: Vec<ReflexRow>,
    /// Gains set from above (2026-11-12, motor: G2 sets them with an expiry); every unit sharing
    /// the table reads the gain in force each time a reflex fires.
    pub gains: std::sync::Arc<std::sync::Mutex<nmtype::order::Gains>>,
}
impl PartialEq for ReflexTable {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
    }
}
/// The gain name that scales the reflex table's confidence bounds.
pub const CONFIDENCE_GAIN: &str = "reflex.confidence";

/// Signals about the whole territory read, for the `territory_empty` and
/// `readers_disagree` fields.
#[derive(Clone, Copy, Debug, Default)]
pub struct Signals {
    pub territory_empty: bool,
    pub readers_disagree: bool,
}

fn glob(pattern: &str, text: &str) -> bool {
    match pattern.split_once('*') {
        None => pattern == text,
        Some((head, tail)) => {
            let tail = tail.trim_start_matches('*').trim_start_matches('/');
            text.starts_with(head) && (tail.is_empty() || text.ends_with(tail))
        }
    }
}

impl ReflexTable {
    /// Read `{"rows": [...]}`.
    /// # Errors
    /// A row missing a field.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let bad = |e: String| Fault::new("nmpod.pipeline", FaultKind::Config, e);
        let rows = v.get("rows").and_then(Value::array).map_err(|e| bad(e.to_string()))?;
        let text = |r: &Value, k: &str| r.get(k).and_then(Value::string).map(str::to_owned).map_err(|e| bad(format!("{k}: {e}")));
        let rows = rows
            .iter()
            .map(|r| {
                Ok(ReflexRow {
                    level: text(r, "level")?,
                    field: text(r, "field")?,
                    op: text(r, "op")?,
                    value: r.get("value").cloned().map_err(|e| bad(e.to_string()))?,
                    then: text(r, "then")?,
                    reason: text(r, "reason")?,
                })
            })
            .collect::<Result<_, Fault>>()?;
        Ok(Self { rows, gains: std::sync::Arc::default() })
    }
    /// The decision for one finding at one level (`drone`, `tentacle`, `brain`).
    #[must_use]
    pub fn decide(&self, level: &str, f: &Finding, s: Signals) -> Reflex {
        self.decide_with_gain(level, f, s, 1.0)
    }
    /// Set a gain from above for `ttl_ms` (motor division).
    pub fn set_gain(&self, name: &str, value: f32, set_by: &str, ttl_ms: u64) {
        if let Ok(mut g) = self.gains.lock() {
            g.set(name, value, set_by, nmtype::now_ms(), ttl_ms);
        }
    }
    /// The decision with the confidence gain in force now (1 when none is set or it expired).
    #[must_use]
    pub fn decide_now(&self, level: &str, f: &Finding, s: Signals) -> Reflex {
        let gain = self.gains.lock().map_or(1.0, |g| g.get(CONFIDENCE_GAIN, nmtype::now_ms()));
        self.decide_with_gain(level, f, s, gain)
    }
    /// The same with a confidence gain set from above (2026-11-12, motor division: G2 sets the
    /// gain, with an expiry, through `nmtype::order::Gains`): every confidence bound of the table
    /// is multiplied by `gain`, so a gain above 1 makes "drop below" and "hold below" fire more
    /// readily and a gain below 1 less. A gain of 1 is exactly `decide`.
    #[must_use]
    pub fn decide_with_gain(&self, level: &str, f: &Finding, s: Signals, gain: f32) -> Reflex {
        for r in self.rows.iter().filter(|r| r.level == "any" || r.level == level) {
            let hit = match (r.field.as_str(), &r.value) {
                ("territory_empty", Value::Bool(b)) => s.territory_empty == *b,
                ("readers_disagree", Value::Bool(b)) => s.readers_disagree == *b,
                ("confidence", Value::Number(n)) => {
                    let n: f32 = n.parse::<f32>().unwrap_or(0.0) * gain;
                    match r.op.as_str() {
                        "lt" => f.score < n,
                        "gt" => f.score > n,
                        _ => false,
                    }
                }
                (field, Value::String(want)) => {
                    let got = match field {
                        "kind" => f.kind.as_str(),
                        "status" => f.status.name(),
                        "source" => f.source.as_str(),
                        "text" => f.text.as_str(),
                        _ => "",
                    };
                    match r.op.as_str() {
                        "eq" => got == want,
                        "ne" => got != want,
                        "contains" => got.contains(want.as_str()),
                        "matches_glob" => glob(want, got),
                        _ => false,
                    }
                }
                _ => false,
            };
            if hit {
                return match r.then.as_str() {
                    "drop" => Reflex::Drop,
                    "escalate" => Reflex::Escalate(r.reason.clone()),
                    "hold" => Reflex::Hold(r.reason.clone()),
                    _ => Reflex::Keep,
                };
            }
        }
        Reflex::Keep
    }
}

/// Readers disagree when two readers report different texts for one key.
#[must_use]
pub fn disagree(findings: &[Finding]) -> bool {
    let mut by: BTreeMap<&str, Vec<(&str, &str)>> = BTreeMap::new();
    for f in findings {
        by.entry(f.key.as_str()).or_default().push((f.reader.as_str(), f.text.as_str()));
    }
    by.values().any(|v| v.iter().any(|a| v.iter().any(|b| a.0 != b.0 && a.1 != b.1)))
}
/// Confidence: the share of the readers that ran which found anything, weighted by the
/// best score (0 when nothing was found).
#[must_use]
pub fn confidence(findings: &[Finding], readers_run: usize) -> f32 {
    if findings.is_empty() || readers_run == 0 {
        return 0.0;
    }
    let mut readers: Vec<&str> = findings.iter().map(|f| f.reader.as_str()).collect();
    readers.sort_unstable();
    readers.dedup();
    #[expect(clippy::cast_precision_loss, reason = "reader counts are single digits")]
    let share = readers.len() as f32 / readers_run as f32;
    let best = findings.iter().map(|f| f.score).fold(0.0_f32, f32::max).max(0.5);
    (share * best).clamp(0.0, 1.0)
}
/// One packet per claim key: agreeing findings merge their evidence; different texts under
/// one key become separate packets (conflicts are kept, never merged away).
#[must_use]
pub fn distill(findings: &[Finding], origin: &ComponentId, version: u64, conf: f32, anomaly: bool, scope: &str) -> Vec<Packet> {
    let mut groups: BTreeMap<(String, String), Vec<&Finding>> = BTreeMap::new();
    for f in findings {
        groups.entry((f.key.clone(), f.text.clone())).or_default().push(f);
    }
    groups
        .into_iter()
        .filter_map(|((key, text), fs)| {
            let first = fs.first()?;
            let mut evidence: Vec<String> = fs.iter().flat_map(|f| f.evidence.clone()).collect();
            evidence.sort();
            evidence.dedup();
            let category = if first.status == Status::Hypothesized { Category::Speculation } else { Category::Fact };
            Some(
                Packet::new(&first.kind, &key, &text, Provenance::new(&first.source, origin, version, nmtype::now_ms()))
                    .status(first.status)
                    .stance(if first.reader == "slot" { Stance::Unknown } else { Stance::Objective })
                    .category(category)
                    .confidence(conf.max(first.score.min(1.0)))
                    .evidence(evidence)
                    .scope(scope)
                    .anomaly(anomaly),
            )
        })
        .collect()
}

/// Self-check of the `pipeline` sub-component: the designs reflex rows decide in order,
/// disagreement is seen, distillation keeps two values under one key apart.
pub(crate) fn check() -> Result<(), String> {
    let table = br#"{"rows":[{"level":"any","field":"source","op":"matches_glob","value":"target/**","then":"drop","reason":"build output"},{"level":"drone","field":"confidence","op":"lt","value":0.2,"then":"drop","reason":"floor"},{"level":"drone","field":"readers_disagree","op":"eq","value":true,"then":"escalate","reason":"disagree"}]}"#;
    let t = ReflexTable::from_value(&nmjson::parse(table).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let f = |key: &str, text: &str, reader: &str, src: &str, score| Finding {
        key: key.into(),
        kind: "symbol".into(),
        text: text.into(),
        source: src.into(),
        evidence: vec![src.into()],
        reader: reader.to_owned(),
        score,
        status: Status::Observed,
    };
    let a = f("value:LIMIT", "LIMIT = 4", "symbols", "a/x.rs:1", 0.9);
    let b = f("value:LIMIT", "LIMIT = 9", "lexical", "b/y.rs:1", 0.9);
    if t.decide("drone", &f("k", "t", "lexical", "target/debug/x.rs:1", 0.9), Signals::default()) != Reflex::Drop {
        return Err("nmpod.pipeline: glob drop".into());
    }
    if t.decide("drone", &a, Signals { readers_disagree: true, ..Signals::default() }) != Reflex::Escalate("disagree".into()) {
        return Err("nmpod.pipeline: escalate on disagreement".into());
    }
    // Gain from above (2026-11-12): 1 is exactly `decide`; 2 doubles the floor, so a 0.3 finding
    // that the plain floor of 0.2 keeps is dropped.
    let mid = f("k", "t", "lexical", "a.rs:1", 0.3);
    if t.decide_with_gain("drone", &mid, Signals::default(), 1.0) != t.decide("drone", &mid, Signals::default())
        || t.decide("drone", &mid, Signals::default()) != Reflex::Keep
        || t.decide_with_gain("drone", &mid, Signals::default(), 2.0) != Reflex::Drop
    {
        return Err("nmpod.pipeline: gain from above".into());
    }
    // A gain set from above reaches `decide_now` and expires.
    t.set_gain(CONFIDENCE_GAIN, 2.0, "nmbrain.g2", 60_000);
    if t.decide_now("drone", &mid, Signals::default()) != Reflex::Drop {
        return Err("nmpod.pipeline: a gain set from above was not in force".into());
    }
    t.set_gain(CONFIDENCE_GAIN, 2.0, "nmbrain.g2", 0);
    let until = std::time::Instant::now() + std::time::Duration::from_millis(2);
    while std::time::Instant::now() < until {
        std::thread::park_timeout(until.saturating_duration_since(std::time::Instant::now()));
    }
    if t.decide_now("drone", &mid, Signals::default()) != Reflex::Keep {
        return Err("nmpod.pipeline: an expired gain stayed in force".into());
    }
    if t.decide("tentacle", &f("k", "t", "lexical", "a.rs:1", 0.1), Signals::default()) != Reflex::Keep {
        return Err("nmpod.pipeline: drone-level row applied at tentacle level".into());
    }
    let both = [a, b];
    if !disagree(&both) || distill(&both, &"nmpod.unit#d0".to_owned(), 1, 0.5, true, "t0").len() != 2 {
        return Err("nmpod.pipeline: conflicting values merged or not seen".into());
    }
    Ok(())
}
