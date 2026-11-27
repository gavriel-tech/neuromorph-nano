//! Preference layer, first form (step P9; guide 6 "Preference learning"; guide 9 item 5;
//! research note on the open gaps, gap 5). Learns the taste and preferences of Luis Jake
//! Gabriel III under specific circumstances, without training: on each accept, reject or
//! correction one record is stored with its circumstance and a short rule derived from the
//! change, and decisions recall the rules of the most similar past circumstances. Method after
//! CIPHER (Gao et al. 2024, arXiv 2404.15269: a preference inferred per edit, stored with the
//! context embedding, the k nearest by cosine recalled); here the rule is a deterministic diff
//! into structured fields instead of a model-written sentence. A small pairwise head on frozen
//! features comes later, once enough pairs exist.
//!
//! The store is behavioural only (guide 5.1), lives under `preferences/`, and is never written
//! under `memory/`. Separable components: `Embedder` (trait; the Core Brain's exact hashed
//! embedder by default), `Feedback` and `PrefRecord` (the record), `extract_rule`, `Prefs`
//! (append-only store), `Index` with `recall` and `predict`, `feed` and `feed_verdict` (the
//! entry the served feedback tool calls), `leave_one_out` (the first measurement).
use crate::unit::{n, obj, s};
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::collections::BTreeSet;
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// Folder of the store under the mini (or a test home).
pub const DIR: &str = "preferences";
/// The append-only record file inside it.
pub const FILE: &str = "records.jsonl";
/// The only data class a preference record carries.
pub const CLASS: &str = "behavioural";

fn bad(kind: FaultKind, m: impl Into<String>) -> Fault {
    Fault::new("nmlearn.preference", kind, m)
}

/// Text to vector (L2-normalised), for circumstance similarity.
pub trait Embedder {
    fn embed(&self, text: &str) -> Vec<f32>;
}
/// The Core Brain's exact embedder (hashed unigrams and bigrams, `nmbrain::slot::embed`).
#[derive(Clone, Copy, Debug, Default)]
pub struct ExactEmbedder;
impl Embedder for ExactEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        nmbrain::slot::embed(text)
    }
}
/// Any function from text to a vector (a neural embedder slot passed in by its caller).
impl<F: Fn(&str) -> Vec<f32>> Embedder for F {
    fn embed(&self, text: &str) -> Vec<f32> {
        self(text)
    }
}
fn cos(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum()
}

/// The kind of signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Signal {
    /// The offered form was accepted (an alternative, when given, is the form not chosen).
    Accept,
    /// The offered form was rejected (an alternative, when given, is the form wanted).
    Reject,
    /// The offered form was changed into the alternative (required).
    Correction,
}
impl Signal {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Accept => "accept",
            Self::Reject => "reject",
            Self::Correction => "correction",
        }
    }
    /// # Errors
    /// An unknown signal.
    pub fn parse(t: &str) -> Result<Self, Fault> {
        match t {
            "accept" => Ok(Self::Accept),
            "reject" => Ok(Self::Reject),
            "correction" => Ok(Self::Correction),
            _ => Err(bad(FaultKind::Config, format!("unknown signal {t:?}"))),
        }
    }
}

/// Where a preference was shown.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Circumstance {
    /// Task kind (answer, report, record, code, run).
    pub task: String,
    /// Part or tool involved.
    pub part: String,
    /// File kind touched, when any.
    pub file_kind: String,
    /// The request in his words or a summary.
    pub text: String,
}
impl Circumstance {
    #[must_use]
    pub fn new(task: &str, part: &str, file_kind: &str, text: &str) -> Self {
        Self { task: task.into(), part: part.into(), file_kind: file_kind.into(), text: text.into() }
    }
    /// The structured key a rule is tied to.
    #[must_use]
    pub fn key(&self) -> String {
        format!("task={}; part={}; file={}", self.task, self.part, self.file_kind)
    }
    fn embed_text(&self) -> String {
        format!("{} {} {} {}", self.task, self.part, self.file_kind, self.text)
    }
    fn to_value(&self) -> Value {
        obj(vec![("task", s(&self.task)), ("part", s(&self.part)), ("file_kind", s(&self.file_kind)), ("text", s(&self.text))])
    }
    fn from_value(v: &Value) -> Self {
        let g = |k: &str| v.get(k).and_then(Value::string).unwrap_or_default().to_owned();
        Self { task: g("task"), part: g("part"), file_kind: g("file_kind"), text: g("text") }
    }
}

/// One signal as the feed receives it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Feedback {
    pub circumstance: Circumstance,
    pub offered: String,
    pub alternative: Option<String>,
    pub signal: Signal,
    /// Where it came from (`nm_feedback`, `constructed`, ...).
    pub source: String,
}

/// A rule in plain structured form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rule {
    pub when: String,
    pub prefer: Vec<String>,
    pub avoid: Vec<String>,
}
impl Rule {
    fn to_value(&self) -> Value {
        let l = |x: &[String]| Value::Array(x.iter().map(|t| s(t)).collect());
        obj(vec![("when", s(&self.when)), ("prefer", l(&self.prefer)), ("avoid", l(&self.avoid))])
    }
}

/// Lower-case word terms of a text (letters and digits; underscores, dashes, dots and slashes
/// split words but are kept inside the whole term too, as path and name terms).
#[must_use]
pub fn terms(text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for w in text.split(|c: char| !(c.is_alphanumeric() || "_-./".contains(c))) {
        let w = w.trim_matches(|c: char| "_-./".contains(c)).to_lowercase();
        if w.is_empty() {
            continue;
        }
        for p in w.split(|c: char| "_-./".contains(c)).filter(|p| !p.is_empty()) {
            out.insert(p.to_owned());
        }
        out.insert(w);
    }
    out
}

/// The rule of one signal: prefer what only the preferred form has, avoid what only the other
/// form has. Accept without an alternative prefers the offered form's terms; reject without
/// one avoids them.
#[must_use]
pub fn extract_rule(c: &Circumstance, signal: Signal, offered: &str, alternative: Option<&str>) -> Rule {
    let (fav, dis) = match signal {
        Signal::Accept => (Some(offered), alternative),
        Signal::Reject | Signal::Correction => (alternative, Some(offered)),
    };
    let (tf, td) = (fav.map(terms).unwrap_or_default(), dis.map(terms).unwrap_or_default());
    Rule { when: c.key(), prefer: tf.difference(&td).cloned().collect(), avoid: td.difference(&tf).cloned().collect() }
}

/// One stored record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefRecord {
    pub at: u64,
    pub signal: Signal,
    pub circumstance: Circumstance,
    pub offered: String,
    pub alternative: Option<String>,
    pub rule: Rule,
    pub source: String,
}
impl PrefRecord {
    /// A record from a signal (the rule is derived here).
    /// # Errors
    /// A correction without its corrected form, or an empty offered form.
    pub fn from_feedback(fb: &Feedback, at: u64) -> Result<Self, Fault> {
        if fb.offered.trim().is_empty() {
            return Err(bad(FaultKind::Config, "offered form is empty"));
        }
        if fb.signal == Signal::Correction && fb.alternative.as_deref().is_none_or(|a| a.trim().is_empty()) {
            return Err(bad(FaultKind::Config, "a correction needs its corrected form"));
        }
        let rule = extract_rule(&fb.circumstance, fb.signal, &fb.offered, fb.alternative.as_deref());
        Ok(Self {
            at,
            signal: fb.signal,
            circumstance: fb.circumstance.clone(),
            offered: fb.offered.clone(),
            alternative: fb.alternative.clone(),
            rule,
            source: fb.source.clone(),
        })
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("at", n(self.at)),
            ("date", s(&crate::cartridge::date_of(self.at))),
            ("class", s(CLASS)),
            ("signal", s(self.signal.name())),
            ("circumstance", self.circumstance.to_value()),
            ("offered", s(&self.offered)),
            ("alternative", self.alternative.as_deref().map_or(Value::Null, s)),
            ("rule", self.rule.to_value()),
            ("source", s(&self.source)),
        ])
    }
    /// # Errors
    /// A line that is not a behavioural preference record.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let g = |k: &str| v.get(k).and_then(Value::string).map(str::to_owned).map_err(|e| bad(FaultKind::Config, format!("{k}: {e}")));
        if g("class")? != CLASS {
            return Err(bad(FaultKind::Policy, "a preference record must be behavioural"));
        }
        let fb = Feedback {
            circumstance: Circumstance::from_value(v.get("circumstance").map_err(|e| bad(FaultKind::Config, e.to_string()))?),
            offered: g("offered")?,
            alternative: v.get("alternative").ok().and_then(|a| a.string().ok()).map(str::to_owned),
            signal: Signal::parse(&g("signal")?)?,
            source: g("source").unwrap_or_default(),
        };
        Self::from_feedback(&fb, v.get("at").and_then(Value::integer).unwrap_or(0))
    }
    /// Was the offered form the preferred one?
    #[must_use]
    pub fn offered_preferred(&self) -> bool {
        self.signal == Signal::Accept
    }
}

/// The store: `<home>/preferences/records.jsonl`, append only.
#[derive(Clone, Debug, Default)]
pub struct Prefs {
    pub dir: PathBuf,
    pub records: Vec<PrefRecord>,
}
impl Prefs {
    /// Open the store of `home` (an absent file is an empty store).
    /// # Errors
    /// A home whose store would sit under a `memory` folder.
    pub fn open(home: &Path) -> Result<Self, Fault> {
        let dir = home.join(DIR);
        if dir.components().any(|c| c.as_os_str() == "memory") {
            return Err(bad(FaultKind::Policy, format!("{}: preferences are never stored under memory/", dir.display())));
        }
        let records = std::fs::read_to_string(dir.join(FILE))
            .map(|t| t.lines().filter_map(|l| nmjson::parse(l.as_bytes()).ok()).filter_map(|v| PrefRecord::from_value(&v).ok()).collect())
            .unwrap_or_default();
        Ok(Self { dir, records })
    }
    /// An in-memory store (no file), for leave-one-out and tests.
    #[must_use]
    pub fn in_memory(records: Vec<PrefRecord>) -> Self {
        Self { dir: PathBuf::new(), records }
    }
    /// Append one record from a signal.
    /// # Errors
    /// An invalid signal or a write failure.
    pub fn add(&mut self, fb: &Feedback) -> Result<PrefRecord, Fault> {
        let r = PrefRecord::from_feedback(fb, nmtype::now_ms())?;
        std::fs::create_dir_all(&self.dir).map_err(|e| bad(FaultKind::Resource, e.to_string()))?;
        let line = r.to_value().encode().map_err(|e| bad(FaultKind::Resource, e.to_string()))?;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir.join(FILE))
            .map_err(|e| bad(FaultKind::Resource, e.to_string()))?;
        writeln!(f, "{line}").map_err(|e| bad(FaultKind::Resource, e.to_string()))?;
        self.records.push(r.clone());
        Ok(r)
    }
}

/// The feed the served feedback tool calls: store one signal under `home`.
/// # Errors
/// As `Prefs::open` and `Prefs::add`.
pub fn feed(home: &Path, fb: &Feedback) -> Result<PrefRecord, Fault> {
    Prefs::open(home)?.add(fb)
}
/// The feed for a plain right or wrong verdict (`nm_feedback` today): right is accept, wrong
/// is reject, with the episode's question as the circumstance.
/// # Errors
/// As `feed`.
pub fn feed_verdict(home: &Path, circumstance: Circumstance, answer: &str, correct: bool) -> Result<PrefRecord, Fault> {
    feed(
        home,
        &Feedback {
            circumstance,
            offered: answer.into(),
            alternative: None,
            signal: if correct { Signal::Accept } else { Signal::Reject },
            source: "nm_feedback".into(),
        },
    )
}

/// Circumstance vectors of a store, computed once.
#[derive(Clone, Debug)]
pub struct Index {
    pub vectors: Vec<Vec<f32>>,
}
impl Index {
    #[must_use]
    pub fn build(prefs: &Prefs, emb: &dyn Embedder) -> Self {
        Self { vectors: prefs.records.iter().map(|r| emb.embed(&r.circumstance.embed_text())).collect() }
    }
}
/// Recall depth derived from the store size: ceil(sqrt(n)), at least 1.
#[must_use]
pub fn recall_k(n: usize) -> usize {
    (1..=n.max(1)).find(|k| k * k >= n).unwrap_or(1)
}
/// The `k` (derived when `None`) nearest records to `c`, as (record index, cosine), best first,
/// skipping `exclude` (the held-out record in leave-one-out).
#[must_use]
pub fn recall(index: &Index, c: &Circumstance, emb: &dyn Embedder, k: Option<usize>, exclude: Option<usize>) -> Vec<(usize, f64)> {
    let q = emb.embed(&c.embed_text());
    let mut sims: Vec<(usize, f64)> =
        index.vectors.iter().enumerate().filter(|(i, _)| Some(*i) != exclude).map(|(i, v)| (i, cos(&q, v))).collect();
    let k = k.unwrap_or_else(|| recall_k(sims.len()));
    sims.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    sims.truncate(k);
    sims
}

/// Which of two options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
    A,
    B,
}
/// A prediction with its evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    pub choice: Choice,
    pub score_a: f64,
    pub score_b: f64,
    /// Record indices recalled.
    pub used: Vec<usize>,
    /// True when the scores tied and `default` decided.
    pub by_default: bool,
}
fn score(rule: &Rule, option: &BTreeSet<String>) -> f64 {
    let hit = |l: &[String]| l.iter().filter(|t| option.contains(*t)).count();
    #[expect(clippy::cast_precision_loss, reason = "counts are small")]
    let x = hit(&rule.prefer) as f64 - hit(&rule.avoid) as f64;
    x
}
/// Which of `a` and `b` he would prefer under `c`: each recalled rule adds its similarity
/// times (preferred terms present minus avoided terms present) to each option; ties go to
/// `default`.
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "a prediction names the store, index, circumstance, both options, embedder, default and held-out record"
)]
pub fn predict(
    prefs: &Prefs,
    index: &Index,
    c: &Circumstance,
    a: &str,
    b: &str,
    emb: &dyn Embedder,
    default: Choice,
    exclude: Option<usize>,
) -> Prediction {
    let near = recall(index, c, emb, None, exclude);
    let (ta, tb) = (terms(a), terms(b));
    let (mut sa, mut sb) = (0.0, 0.0);
    for (i, sim) in &near {
        if let Some(r) = prefs.records.get(*i) {
            sa += sim.max(0.0) * score(&r.rule, &ta);
            sb += sim.max(0.0) * score(&r.rule, &tb);
        }
    }
    let tie = (sa - sb).abs() < 1e-12;
    let choice = if tie {
        default
    } else if sa > sb {
        Choice::A
    } else {
        Choice::B
    };
    Prediction { choice, score_a: sa, score_b: sb, used: near.into_iter().map(|(i, _)| i).collect(), by_default: tie }
}

/// Leave-one-out result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Loo {
    pub n: usize,
    pub recall_right: usize,
    pub majority_right: usize,
    pub ties: usize,
}
/// Leave-one-out over cases with an alternative: predict offered (A) against alternative (B)
/// from the other records, against always choosing the more common answer among them.
/// Ties in recall fall back to that same majority answer.
#[must_use]
pub fn leave_one_out(cases: &[Feedback], emb: &dyn Embedder) -> Loo {
    let records: Vec<PrefRecord> =
        cases.iter().filter(|c| c.alternative.is_some()).filter_map(|c| PrefRecord::from_feedback(c, 0).ok()).collect();
    let prefs = Prefs::in_memory(records);
    let index = Index::build(&prefs, emb);
    let mut out = Loo { n: prefs.records.len(), recall_right: 0, majority_right: 0, ties: 0 };
    for (i, r) in prefs.records.iter().enumerate() {
        let others = prefs.records.iter().enumerate().filter(|(j, _)| *j != i);
        let accepts = others.clone().filter(|(_, o)| o.offered_preferred()).count();
        let majority = if 2 * accepts > out.n - 1 { Choice::A } else { Choice::B };
        let truth = if r.offered_preferred() { Choice::A } else { Choice::B };
        let p = predict(&prefs, &index, &r.circumstance, &r.offered, r.alternative.as_deref().unwrap_or(""), emb, majority, Some(i));
        out.recall_right += usize::from(p.choice == truth);
        out.majority_right += usize::from(majority == truth);
        out.ties += usize::from(p.by_default);
    }
    out
}

/// A constructed set from the project's own working rules (`CLAUDE.md`, `rules.md`): eight
/// circumstances, three signals each, offered form first and the other form second. Used by
/// the check and the first measurement; not a record of real feedback.
#[must_use]
pub fn working_rule_cases() -> Vec<Feedback> {
    let rows: [(&str, &str, &str, &str, &str, &str, &str); 24] = [
        (
            "record",
            "run note",
            "md",
            "write the time stamp of a new run note",
            "Written 14:05 local time",
            "Written 2026-11-10 12:05 UTC",
            "correction",
        ),
        (
            "record",
            "run note",
            "md",
            "stamp the result record",
            "Result recorded 2026-11-10 09:40 UTC",
            "Result recorded 11:40 local time",
            "accept",
        ),
        (
            "record",
            "design note",
            "md",
            "time of a dated addition",
            "Addition at 03:15 local time",
            "Addition 2026-11-10 01:15 UTC",
            "reject",
        ),
        (
            "code",
            "new folder",
            "dir",
            "create a folder for preference records",
            "folder created with a README.md guide saying what goes in it",
            "folder created without guide",
            "accept",
        ),
        (
            "code",
            "new folder",
            "dir",
            "make a folder for case sets",
            "folder created without guide",
            "folder created with a README.md guide saying what goes in it and what writes it",
            "correction",
        ),
        (
            "code",
            "new folder",
            "dir",
            "add an output folder for a stage",
            "new folder made, guide left for later",
            "new folder made with its README.md guide",
            "reject",
        ),
        (
            "code",
            "temporary file",
            "tmp",
            "where to put a temporary copy for a harness",
            "temporary copy written to the repository root",
            "temporary copy written under scratch/",
            "correction",
        ),
        (
            "code",
            "temporary file",
            "tmp",
            "intermediate output of a probe",
            "intermediate output kept under scratch/ only",
            "intermediate output written next to the source",
            "accept",
        ),
        (
            "code",
            "temporary file",
            "tmp",
            "disposable gate copy",
            "disposable gate copy placed in runs/",
            "disposable gate copy placed in scratch/gates/",
            "reject",
        ),
        (
            "code",
            "operating limit",
            "rs",
            "set the RAM limit for kept encodings",
            "fixed limit of 8 GiB",
            "limit derived at run time as half of free RAM, fallback 8 GiB stated",
            "correction",
        ),
        (
            "code",
            "operating limit",
            "rs",
            "choose the batch size",
            "batch size derived at run time from free memory and measured cost, fallback stated",
            "fixed batch size of 16",
            "accept",
        ),
        (
            "code",
            "operating limit",
            "rs",
            "pick the shortlist length",
            "hard-coded shortlist of 4",
            "shortlist derived at run time from measurements, fixed only as stated fallback",
            "reject",
        ),
        (
            "finish",
            "git",
            "repo",
            "work is done and builds clean",
            "committed and pushed to the remote",
            "asked before commit and push, nothing committed",
            "correction",
        ),
        (
            "finish",
            "git",
            "repo",
            "a fix passes its check",
            "reported the fix and asked before commit",
            "committed the fix directly",
            "accept",
        ),
        ("finish", "git", "repo", "after a refactor", "pushed the branch without asking", "asked before push", "reject"),
        (
            "tidy",
            "logs",
            "jsonl",
            "the run logs are long",
            "deleted old logs to save space",
            "kept every log and added a new dated note",
            "correction",
        ),
        (
            "tidy",
            "logs",
            "md",
            "journal is getting large",
            "kept the journal whole and added a new dated note",
            "condensed the journal into a summary",
            "accept",
        ),
        (
            "tidy",
            "logs",
            "txt",
            "old reports clutter runs",
            "removed superseded reports",
            "kept the superseded reports and named the new one in the run note",
            "reject",
        ),
        (
            "answer",
            "reply",
            "chat",
            "report progress",
            "long elaborate reply with many headings and background",
            "plain short reply with the numbers",
            "correction",
        ),
        (
            "answer",
            "reply",
            "chat",
            "say what changed",
            "plain short reply naming the files",
            "long elaborate reply with background sections",
            "accept",
        ),
        (
            "answer",
            "reply",
            "chat",
            "explain a failure",
            "lengthy narrative with every detail repeated",
            "plain short reply with cause and fix",
            "reject",
        ),
        (
            "train",
            "weights",
            "safetensors",
            "improve the reranker",
            "edit the released weights in place",
            "fit a separate unit beside the released weights, unchanged base",
            "correction",
        ),
        (
            "train",
            "weights",
            "safetensors",
            "add a judgment",
            "fit a separate unit beside the released weights, base unchanged",
            "fine-tune the released weights",
            "accept",
        ),
        (
            "train",
            "weights",
            "safetensors",
            "speed up a model",
            "overwrite the released weights with a pruned copy",
            "keep the released weights unchanged and add a separate piece",
            "reject",
        ),
    ];
    rows.iter()
        .map(|(task, part, fk, text, offered, other, sig)| Feedback {
            circumstance: Circumstance::new(task, part, fk, text),
            offered: (*offered).into(),
            alternative: Some((*other).into()),
            signal: match *sig {
                "accept" => Signal::Accept,
                "reject" => Signal::Reject,
                _ => Signal::Correction,
            },
            source: "constructed".into(),
        })
        .collect()
}

/// Self-check of `preference`: a rule is the diff of the two forms; records round-trip and
/// are behavioural; the feed writes under `preferences/` and refuses a `memory/` home and a
/// correction with no corrected form; recall finds the same circumstance first; on the
/// constructed working-rule set, leave-one-out recall beats always choosing the more common
/// answer.
pub(crate) fn check() -> Result<(), String> {
    let f = |m: String| Err(format!("nmlearn.preference: {m}"));
    let c = Circumstance::new("record", "run note", "md", "stamp a note");
    let r = extract_rule(&c, Signal::Correction, "Written 14:05 local time", Some("Written 12:05 UTC"));
    if r.prefer != ["12", "utc"]
        || !r.avoid.contains(&"local".to_owned())
        || r.avoid.contains(&"written".to_owned())
        || r.when != "task=record; part=run note; file=md"
    {
        return f(format!("rule {r:?}"));
    }
    let home = nmtype::check_dir("nmlearn-preference").map_err(|e| e.to_string())?;
    let fb = Feedback {
        circumstance: c.clone(),
        offered: "Written 14:05 local time".into(),
        alternative: Some("Written 12:05 UTC".into()),
        signal: Signal::Correction,
        source: "check".into(),
    };
    let rec = feed(&home, &fb).map_err(|e| e.to_string())?;
    feed_verdict(&home, c.clone(), "an answer", true).map_err(|e| e.to_string())?;
    let back = Prefs::open(&home).map_err(|e| e.to_string())?;
    let text = std::fs::read_to_string(home.join(DIR).join(FILE)).map_err(|e| e.to_string())?;
    if back.records.len() != 2
        || back.records.first() != Some(&rec)
        || !text.lines().all(|l| l.contains("\"class\":\"behavioural\""))
        || back.records.get(1).map(|x| x.signal) != Some(Signal::Accept)
    {
        return f(format!("store round trip: {} records", back.records.len()));
    }
    if Prefs::open(&home.join("memory")).is_ok() {
        return f("a store under memory/ was opened".into());
    }
    let mut no_fix = fb.clone();
    no_fix.alternative = None;
    if feed(&home, &no_fix).is_ok() {
        return f("a correction with no corrected form was stored".into());
    }
    let cases = working_rule_cases();
    let all = Prefs::in_memory(cases.iter().filter_map(|x| PrefRecord::from_feedback(x, 0).ok()).collect());
    let idx = Index::build(&all, &ExactEmbedder);
    let first = cases.first().ok_or("no cases")?;
    if recall(&idx, &first.circumstance, &ExactEmbedder, Some(1), None).first().map(|x| x.0) != Some(0) || recall_k(23) != 5 {
        return f("recall order or depth".into());
    }
    let loo = leave_one_out(&cases, &ExactEmbedder);
    if loo.n != 24 || loo.recall_right <= loo.majority_right {
        return f(format!("leave-one-out recall {} of {} against majority {}", loo.recall_right, loo.n, loo.majority_right));
    }
    Ok(())
}
