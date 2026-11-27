//! Verified outcomes as memories. The evidence ledger (`evidence.jsonl` in a project's memory
//! folder) judges what followed each context pack and each prime, and writes an `outcome` row when
//! something was verified or refuted: which places answered which question (`answered`), and which
//! lines fixed which failing check (`fixed`). This module turns those rows into memory records
//! through the journal, which stays the only writer of every tier file, and reads them back for
//! recall.
//!
//! Rules:
//! - One record per kind of outcome, question (or failing check) and set of places: a second
//!   verification of the same thing adds a verdict and a use to the same record.
//! - Each ledger outcome is imported once: its id is kept as the `by` of the verdict it gives, so
//!   importing again finds it and adds nothing.
//! - The newest verdict decides the record's state: verified is active, refuted is invalid. A
//!   refuted record is kept and shown as refuted (class protected, so it never leaves the views by
//!   itself); nothing is dropped.
//! - Records keep paths, line ranges and the hash of the lines at verification, never the lines.
//!   Recall reads the lines fresh and compares the hash; a place whose lines changed is stale.
//! - Memory changes no weights and no settings; it keeps what happened, with its evidence.
use crate::book::Book;
use crate::record::{Anchor, Class, Kind, Part, Record, Retention, Source, State, Tier, Validity};
use crate::store::{obj, s, u};
use nmjson::Value;
use nmtype::Fault;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// The topic outcome memories are kept under (`2-active/outcomes/items.json` and so on).
pub const TOPIC: &str = "outcomes";
/// The ledger file in a project memory folder (written by the ledger, only read here).
pub const LEDGER: &str = "evidence.jsonl";

fn text(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::string).ok().map(str::to_owned)
}
fn sha16(t: &str) -> String {
    nmhash::sha256_bytes(t.as_bytes()).chars().take(16).collect()
}
/// Strength as a rank: direct 3, indirect 2, weak 1, anything else 0.
#[must_use]
pub fn rank(strength: &str) -> u8 {
    match strength {
        "direct" => 3,
        "indirect" => 2,
        "weak" => 1,
        _ => 0,
    }
}

/// A place an outcome rests on: a file, its lines, and the hash of those lines when verified.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Place {
    pub path: String,
    pub start: usize,
    pub end: usize,
    pub hash: String,
}

impl Place {
    /// `path:a-b` (or `path:a`).
    #[must_use]
    pub fn label(&self) -> String {
        if self.start == self.end { format!("{}:{}", self.path, self.start) } else { format!("{}:{}-{}", self.path, self.start, self.end) }
    }
    fn to_value(&self) -> Value {
        obj(vec![("path", s(&self.path)), ("lines", Value::Array(vec![u(self.start as u64), u(self.end as u64)])), ("hash", s(&self.hash))])
    }
    fn from_value(v: &Value) -> Option<Self> {
        let l = v.get("lines").ok()?.array().ok()?;
        Some(Self {
            path: text(v, "path")?,
            start: usize::try_from(l.first()?.integer().ok()?).ok()?,
            end: usize::try_from(l.get(1)?.integer().ok()?).ok()?,
            hash: text(v, "hash")?,
        })
    }
}

/// The hash of lines `start..=end` (1-based) of a text, as anchors keep it; `None` when the text
/// has fewer lines.
#[must_use]
pub fn lines_hash_of(body: &str, start: usize, end: usize) -> Option<String> {
    if start == 0 || end < start {
        return None;
    }
    let lines: Vec<&str> = body.lines().skip(start - 1).take(end - start + 1).collect();
    (lines.len() == end - start + 1).then(|| crate::record::lines_hash(&lines))
}

/// Reads a place fresh under `root`: whether its lines still hash as they did.
#[must_use]
pub fn fresh(root: &Path, p: &Place) -> bool {
    let file = p.path.split('/').fold(root.to_path_buf(), |a, c| a.join(c));
    std::fs::read_to_string(file).ok().and_then(|b| lines_hash_of(&b, p.start, p.end)).is_some_and(|h| h == p.hash)
}

/// A failing check's signature: check class and package, first error's code, file, line, message.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Failure {
    pub check: String,
    pub pkg: String,
    pub code: String,
    pub file: String,
    pub line: u64,
    pub msg: String,
}

impl Failure {
    /// `cargo-test shop`.
    #[must_use]
    pub fn check_label(&self) -> String {
        if self.pkg.is_empty() { self.check.clone() } else { format!("{} {}", self.check, self.pkg) }
    }
    fn key(&self) -> String {
        format!("{}|{}|{}|{}", self.check, self.pkg, self.code, self.file)
    }
}

/// One outcome row of the ledger.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    /// The ledger's id of the outcome (`answered:<pack>`, `fixed:<16 hex>`).
    pub id: String,
    /// `answered` or `fixed`.
    pub kind: String,
    pub verified: bool,
    pub strength: String,
    /// The rules that decided it, for example `check` or `revert`.
    pub rule: String,
    /// `answered`: the question asked of the Nano (filled from the situation record when read
    /// back from the ledger).
    pub question: Option<String>,
    /// `answered`: the question's hash as the ledger keeps it.
    pub ask_sha: Option<String>,
    /// `fixed`: the failure that was fixed.
    pub failure: Option<Failure>,
    pub places: Vec<Place>,
    pub session: String,
    pub at: u64,
}

impl Outcome {
    /// The ledger row's own fields (the ledger adds its sequence, time and hashes).
    #[must_use]
    pub fn fields(&self) -> Vec<(&'static str, Value)> {
        let mut f = vec![
            ("row", s("outcome")),
            ("outcome", s(&self.id)),
            ("kind", s(&self.kind)),
            ("state", s(if self.verified { "verified" } else { "refuted" })),
            ("strength", s(&self.strength)),
            ("rule", s(&self.rule)),
            ("session", s(&self.session)),
            ("at", u(self.at)),
            ("places", Value::Array(self.places.iter().map(Place::to_value).collect())),
            ("memory", s(&record_id(self))),
        ];
        // The ledger keeps the question as its hash only; the text stays in the situation record,
        // under that record's own rules, and is looked up there (`questions`).
        if let Some(q) = &self.question {
            f.push(("ask_sha", s(&ask_sha(q))));
        }
        if let Some(x) = &self.failure {
            f.push((
                "failure",
                obj(vec![("check", s(&x.check)), ("pkg", s(&x.pkg)), ("code", s(&x.code)), ("file", s(&x.file)), ("line", u(x.line)), ("msg", s(&x.msg))]),
            ));
        }
        f
    }
    /// An outcome from a ledger row (`row: outcome`); `None` for any other row.
    #[must_use]
    pub fn from_value(v: &Value) -> Option<Self> {
        if text(v, "row").as_deref() != Some("outcome") {
            return None;
        }
        let failure = v.get("failure").ok().map(|f| Failure {
            check: text(f, "check").unwrap_or_default(),
            pkg: text(f, "pkg").unwrap_or_default(),
            code: text(f, "code").unwrap_or_default(),
            file: text(f, "file").unwrap_or_default(),
            line: f.get("line").and_then(Value::integer).unwrap_or(0),
            msg: text(f, "msg").unwrap_or_default(),
        });
        Some(Self {
            id: text(v, "outcome")?,
            kind: text(v, "kind")?,
            verified: text(v, "state").as_deref() == Some("verified"),
            strength: text(v, "strength").unwrap_or_default(),
            rule: text(v, "rule").unwrap_or_default(),
            question: None,
            ask_sha: text(v, "ask_sha"),
            failure,
            places: v.get("places").and_then(Value::array).unwrap_or_default().iter().filter_map(Place::from_value).collect(),
            session: text(v, "session").unwrap_or_default(),
            at: v.get("at").and_then(Value::integer).unwrap_or(0),
        })
    }
}

/// Lower-cased words of three letters or more, without common words: the terms recall matches.
#[must_use]
pub fn terms(t: &str) -> BTreeSet<String> {
    const COMMON: &[&str] = &[
        "the", "and", "for", "are", "was", "were", "with", "from", "into", "onto", "this", "that", "these", "those", "what", "where", "which",
        "when", "who", "whom", "why", "how", "does", "did", "done", "doing", "can", "could", "should", "would", "will", "shall", "may", "might",
        "must", "has", "have", "had", "not", "but", "you", "your", "our", "its", "they", "them", "their", "there", "here", "then", "than", "also",
        "any", "all", "some", "about", "please", "make", "let", "get", "got", "use", "used", "using", "now", "just", "only", "out", "one",
        "code", "file", "files", "function", "find", "show", "tell", "look", "need", "want", "like", "way",
    ];
    t.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .map(str::to_lowercase)
        .filter(|w| w.chars().count() >= 3 && !COMMON.contains(&w.as_str()))
        .collect()
}

/// The memory record an outcome lands in: the same kind, question (or failure) and places (path,
/// lines and the lines' content) always give the same record; the same lines with other content
/// are another record.
#[must_use]
pub fn record_id(o: &Outcome) -> String {
    let key = match (&o.question, &o.failure) {
        (Some(q), _) => terms(q).into_iter().collect::<Vec<_>>().join(" "),
        (None, Some(f)) => f.key(),
        (None, None) => String::new(),
    };
    let places: Vec<String> = o.places.iter().map(|p| format!("{} {}", p.label(), p.hash)).collect();
    format!("oc:{}", sha16(&format!("{}\n{key}\n{}", o.kind, places.join("\n"))))
}

/// The verdict's `by` that marks a ledger outcome as imported.
fn by_of(o: &Outcome) -> String {
    format!("ledger {} {} {}", o.id, o.rule, o.strength)
}

fn summary(o: &Outcome) -> String {
    let places: Vec<String> = o.places.iter().map(Place::label).collect();
    match (&o.question, &o.failure) {
        (Some(q), _) => format!("question \"{q}\" answered at {}", places.join(", ")),
        (None, Some(f)) => format!(
            "{} failed{} at {}:{}{}; fixed by the change at {}",
            f.check_label(),
            if f.code.is_empty() { String::new() } else { format!(" with {}", f.code) },
            f.file,
            f.line,
            if f.msg.is_empty() { String::new() } else { format!(" ({})", f.msg) },
            places.join(", ")
        ),
        (None, None) => format!("{} at {}", o.kind, places.join(", ")),
    }
}

/// The memory record of an outcome (before any verdict).
#[must_use]
pub fn record_of(o: &Outcome, day: u64) -> Record {
    let anchors: Vec<Anchor> = o
        .places
        .iter()
        .map(|p| Anchor {
            source: Source::File,
            location: p.path.clone(),
            part: Part::Lines { start: p.start, end: p.end },
            part_hash: p.hash.clone(),
            name: String::new(),
            excerpt: None,
        })
        .collect();
    let text = summary(o);
    let kind = if o.kind == "fixed" { Kind::Failure } else { Kind::Fact };
    let mut keys: Vec<String> = vec![format!("outcome:{}", o.kind)];
    keys.extend(o.places.iter().map(|p| p.path.clone()));
    if let Some(q) = &o.question {
        keys.push(format!("q:{q}"));
    }
    if let Some(f) = &o.failure {
        keys.push(format!("check:{}", f.check));
        keys.push(format!("pkg:{}", f.pkg));
        keys.push(format!("error:{}|{}|{}", f.code, f.file, f.line));
        if !f.msg.is_empty() {
            keys.push(format!("msg:{}", f.msg));
        }
    }
    keys.dedup();
    Record {
        id: record_id(o),
        content: Record::content_hash(kind, &text, &anchors),
        kind,
        class: Class::Protected,
        tier: Tier::Active,
        state: State::Active,
        topic: TOPIC.to_owned(),
        text,
        anchors,
        keys,
        links: Vec::new(),
        validity: Validity { from_day: day, recorded_day: day, ..Validity::default() },
        retention: Retention { stability: crate::retention::S0_ACTIVE, difficulty: crate::retention::d0(), last_use_day: day, ..Retention::default() },
        verdicts: Vec::new(),
    }
}

/// The journal events that bring outcomes into the book: per outcome not yet imported, the record
/// when it is new, its verdict (`by` naming the ledger outcome, its rule and strength), a use
/// (useful when verified), and a state change when the verdict turns it over. Text that must never
/// be saved (secrets) is refused, named in the second list.
#[must_use]
pub fn import_events(book: &Book, outcomes: &[Outcome], day: u64, locked: &[String]) -> (Vec<(String, Value)>, Vec<String>) {
    let mut seen: BTreeSet<String> = book.records.values().flat_map(|r| r.verdicts.iter().map(|v| v.1.clone())).collect();
    let mut state: BTreeMap<String, State> = BTreeMap::new();
    let mut out = Vec::new();
    let mut refused = Vec::new();
    for o in outcomes {
        let by = by_of(o);
        if seen.contains(&by) || !usable(o) {
            continue;
        }
        let r = record_of(o, day);
        if let Some(why) = std::iter::once(r.text.as_str()).chain(r.keys.iter().map(String::as_str)).find_map(|t| crate::classes::never_save(t, locked)) {
            refused.push(format!("{}: {why}", o.id));
            continue;
        }
        let id = r.id.clone();
        let now = match state.get(&id).copied().or_else(|| book.records.get(&id).map(|x| x.state)) {
            Some(st) => st,
            None => {
                out.push(("record".to_owned(), r.to_value()));
                State::Active
            }
        };
        out.push(("verdict".to_owned(), obj(vec![("id", s(&id)), ("right", Value::Bool(o.verified)), ("by", s(&by))])));
        out.push(("use".to_owned(), obj(vec![("id", s(&id)), ("useful", Value::Bool(o.verified))])));
        let want = if o.verified { State::Active } else { State::Invalid };
        if now != want {
            let why = if o.verified { "verified again by the evidence ledger" } else { "refuted by the evidence ledger" };
            out.push(("state".to_owned(), obj(vec![("id", s(&id)), ("to", s(want.label())), ("from", s(now.label())), ("why", s(why))])));
        }
        state.insert(id, want);
        seen.insert(by);
    }
    (out, refused)
}

/// The hash a question is kept under: the first 16 hex digits of the SHA-256 of its words, lower
/// case, single spaced (the same hash the ledger's `served` rows keep).
#[must_use]
pub fn ask_sha(q: &str) -> String {
    sha16(&q.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase())
}

/// The situation record's file in a project memory folder (written by the hook program).
pub const SITUATION: &str = "situation.jsonl";

/// The questions asked of the Nano's tools in a project's situation record, by `ask_sha`.
#[must_use]
pub fn questions(memory: &Path) -> BTreeMap<String, String> {
    let Ok(t) = std::fs::read_to_string(memory.join(SITUATION)) else { return BTreeMap::new() };
    t.lines()
        .filter(|l| l.contains("\"nano\":"))
        .filter_map(|l| nmjson::parse(l.as_bytes()).ok())
        .filter_map(|v| v.get("nano").ok().and_then(|n| text(n, "ask")))
        .map(|q| (ask_sha(&q), q))
        .collect()
}

/// Every outcome row of a project's ledger, in ledger order, each answered one with its question
/// looked up in the situation record (left `None` when the record no longer holds it). Rows are
/// read as written; the ledger itself checks its chain.
#[must_use]
pub fn from_ledger(memory: &Path) -> Vec<Outcome> {
    let Ok(t) = std::fs::read_to_string(memory.join(LEDGER)) else { return Vec::new() };
    let mut out: Vec<Outcome> =
        t.lines().filter(|l| l.contains("\"row\":\"outcome\"")).filter_map(|l| nmjson::parse(l.as_bytes()).ok()).filter_map(|v| Outcome::from_value(&v)).collect();
    if out.iter().any(|o| o.ask_sha.is_some()) {
        let qs = questions(memory);
        for o in &mut out {
            if let Some(h) = &o.ask_sha {
                o.question = qs.get(h).cloned();
            }
        }
    }
    out
}

/// An outcome memory can take in: it has places, and an answered one has its question.
fn usable(o: &Outcome) -> bool {
    !o.places.is_empty() && (o.kind != "answered" || o.question.is_some())
}

/// Imports the outcomes of a project's ledger into the open memory. Returns the events written
/// and the outcomes refused.
/// # Errors
/// A journal or view write fault.
pub fn import_ledger(m: &mut crate::mem2::Memory2, memory: &Path) -> Result<(usize, Vec<String>), Fault> {
    let outcomes = from_ledger(memory);
    if outcomes.is_empty() {
        return Ok((0, Vec::new()));
    }
    let locked = crate::classes::locked_questions(&nmtype::nano_root());
    let (events, refused) = import_events(&m.book, &outcomes, m.day, &locked);
    let n = events.len();
    m.commit_events(events)?;
    Ok((n, refused))
}

/// What memory knows of one outcome record: from the journal's views, from ledger outcomes not yet
/// imported, or both.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Known {
    pub record: String,
    pub kind: String,
    pub question: Option<String>,
    pub failure: Option<Failure>,
    pub places: Vec<Place>,
    pub verified: u32,
    pub refuted: u32,
    /// The strongest strength among the verifications.
    pub strength: String,
    /// The newest verdict: true verified, false refuted.
    pub standing: bool,
    /// Whether the journal holds it (else it waits in the ledger for the next import).
    pub in_memory: bool,
}

impl Known {
    fn from_record(r: &Record) -> Option<Self> {
        if r.topic != TOPIC {
            return None;
        }
        let key = |p: &str| r.keys.iter().find_map(|k| k.strip_prefix(p).map(str::to_owned));
        let kind = key("outcome:")?;
        let failure = key("check:").map(|check| {
            let e = key("error:").unwrap_or_default();
            let mut it = e.splitn(3, '|');
            Failure {
                check,
                pkg: key("pkg:").unwrap_or_default(),
                code: it.next().unwrap_or_default().to_owned(),
                file: it.next().unwrap_or_default().to_owned(),
                line: it.next().and_then(|x| x.parse().ok()).unwrap_or(0),
                msg: key("msg:").unwrap_or_default(),
            }
        });
        let places = r
            .anchors
            .iter()
            .filter_map(|a| match a.part {
                Part::Lines { start, end } => Some(Place { path: a.location.clone(), start, end, hash: a.part_hash.clone() }),
                _ => None,
            })
            .collect();
        let strength = r
            .verdicts
            .iter()
            .filter(|v| v.0)
            .filter_map(|v| v.1.rsplit(' ').next().map(str::to_owned))
            .max_by_key(|x| rank(x))
            .unwrap_or_default();
        Some(Self {
            record: r.id.clone(),
            kind,
            question: key("q:"),
            failure,
            places,
            verified: r.verdicts.iter().filter(|v| v.0).count() as u32,
            refuted: r.verdicts.iter().filter(|v| !v.0).count() as u32,
            strength,
            standing: r.verdicts.last().map_or(r.state == State::Active, |v| v.0),
            in_memory: true,
        })
    }
}

/// The outcome records in the journal's views of a memory folder, read only: `1-core/items.json`,
/// and `items.json` of the `outcomes` topic in `2-active` and `3-archive`. Each record's verdict
/// `by` values are returned too, so ledger outcomes already imported are known.
#[must_use]
pub fn from_views(dir: &Path) -> (Vec<Record>, BTreeSet<String>) {
    let files = [dir.join("1-core").join("items.json"), dir.join("2-active").join(TOPIC).join("items.json"), dir.join("3-archive").join(TOPIC).join("items.json")];
    let mut out: Vec<Record> = Vec::new();
    let mut by = BTreeSet::new();
    for f in files {
        let Some(v) = crate::store::get(&f) else { continue };
        for item in v.get("items").and_then(Value::array).unwrap_or_default() {
            if let Ok(r) = Record::from_value(item)
                && r.topic == TOPIC
                && !out.iter().any(|x| x.id == r.id)
            {
                by.extend(r.verdicts.iter().map(|x| x.1.clone()));
                out.push(r);
            }
        }
    }
    (out, by)
}

/// The outcome records of an open journal's book, in RAM (what a serving process recalls from).
#[must_use]
pub fn known_in(book: &Book) -> Vec<Known> {
    book.records.values().filter_map(Known::from_record).collect()
}

/// Everything memory knows of outcomes for a project: the journal's outcome records, with the
/// ledger's outcomes not yet imported folded in (newest last, so the newest verdict stands).
#[must_use]
pub fn known(dir: &Path) -> Vec<Known> {
    let (records, imported) = from_views(dir);
    let mut out: Vec<Known> = records.iter().filter_map(Known::from_record).collect();
    for o in from_ledger(dir) {
        if imported.contains(&by_of(&o)) || !usable(&o) {
            continue;
        }
        let id = record_id(&o);
        let k = match out.iter().position(|k| k.record == id) {
            Some(i) => out.get_mut(i),
            None => {
                out.push(Known {
                    record: id,
                    kind: o.kind.clone(),
                    question: o.question.clone(),
                    failure: o.failure.clone(),
                    places: o.places.clone(),
                    verified: 0,
                    refuted: 0,
                    strength: String::new(),
                    standing: o.verified,
                    in_memory: false,
                });
                out.last_mut()
            }
        };
        if let Some(k) = k {
            if o.verified {
                k.verified += 1;
                if rank(&o.strength) > rank(&k.strength) {
                    k.strength.clone_from(&o.strength);
                }
            } else {
                k.refuted += 1;
            }
            k.standing = o.verified;
        }
    }
    out
}

/// Answered outcomes whose question matches `question`, best first: the share of the stored
/// question's terms found in `question`, each term weighted by its rarity among the stored
/// questions. A match is kept when its share stands out from the shares of the other stored
/// questions (`evidence::recalled`, at the resolution of the stored question's lightest term).
#[must_use]
pub fn recall<'a>(known: &'a [Known], question: &str) -> Vec<(f64, &'a Known)> {
    let asked = terms(question);
    let stored: Vec<(BTreeSet<String>, &Known)> = known.iter().filter_map(|k| Some((terms(k.question.as_deref()?), k))).collect();
    let n = stored.len().max(1) as f64;
    let idf = |t: &str| {
        let df = stored.iter().filter(|(ts, _)| ts.contains(t)).count() as f64;
        (1.0 + n / df.max(1.0)).ln()
    };
    // Each stored question: its share and its resolution (the share its lightest term carries).
    let scored: Vec<(f64, f64, &Known)> = stored
        .iter()
        .map(|(ts, k)| {
            let all: f64 = ts.iter().map(|t| idf(t)).sum();
            if all <= 0.0 {
                return (0.0, 1.0, *k);
            }
            let share = ts.iter().filter(|t| asked.contains(*t)).map(|t| idf(t)).sum::<f64>() / all;
            let lightest = ts.iter().map(|t| idf(t)).fold(f64::INFINITY, f64::min) / all;
            (share, lightest, *k)
        })
        .collect();
    #[expect(clippy::cast_possible_truncation, reason = "shares between 0 and 1")]
    let mut out: Vec<(f64, &Known)> = scored
        .iter()
        .enumerate()
        .filter(|(i, (share, lightest, _))| {
            let others: Vec<f32> = scored.iter().enumerate().filter(|(j, _)| j != i).map(|(_, x)| x.0 as f32).collect();
            *share > 0.0 && crate::evidence::recalled(*share as f32, &others, *lightest as f32)
        })
        .map(|(_, (share, _, k))| (*share, *k))
        .collect();
    out.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.record.cmp(&b.1.record)));
    out
}

/// Section check: an answered and a fixed outcome become two records; importing again adds
/// nothing; a refuting outcome turns the answered record invalid and keeps it; a second
/// verification adds to the same record; the views read back as known outcomes with their counts
/// and standing; a ledger outcome not yet imported is folded in; recall finds a similar question
/// and not an unrelated one; a place whose lines changed is not fresh.
/// # Errors
/// One fault line.
pub(crate) fn check() -> Result<(), String> {
    let root = nmtype::check_dir("nmmem-outcomes-root").map_err(|e| e.to_string())?;
    let body = "fn a() {}\nfn total(x: u64) -> u64 {\n    x * 2\n}\n";
    std::fs::create_dir_all(root.join("src")).map_err(|e| e.to_string())?;
    std::fs::write(root.join("src/lib.rs"), body).map_err(|e| e.to_string())?;
    let h = lines_hash_of(body, 2, 4).ok_or("outcomes: hash of lines 2-4")?;
    let place = Place { path: "src/lib.rs".into(), start: 2, end: 4, hash: h };
    let answered = |id: &str, ok: bool, strength: &str| Outcome {
        id: id.into(),
        kind: "answered".into(),
        verified: ok,
        strength: strength.into(),
        rule: if ok { "check".into() } else { "revert".into() },
        question: Some("where is the order total computed".into()),
        ask_sha: None,
        failure: None,
        places: vec![place.clone()],
        session: "s1".into(),
        at: 1,
    };
    let fixed = Outcome {
        id: "fixed:1".into(),
        kind: "fixed".into(),
        verified: true,
        strength: "direct".into(),
        rule: "check".into(),
        question: None,
        ask_sha: None,
        failure: Some(Failure { check: "cargo-test".into(), pkg: "shop".into(), code: "E0308".into(), file: "src/lib.rs".into(), line: 3, msg: "mismatched types".into() }),
        places: vec![place.clone()],
        session: "s1".into(),
        at: 2,
    };
    if Outcome::from_value(&Value::Object(fixed.fields().into_iter().map(|(k, v)| (k.to_owned(), v)).collect())).as_ref() != Some(&fixed) {
        return Err("outcomes: a row does not read back as the outcome it was written from".into());
    }
    let dir = nmtype::check_dir("nmmem-outcomes").map_err(|e| e.to_string())?;
    let mut m = crate::mem2::Memory2::open(&dir, &root).map_err(|e| e.to_string())?;
    let first = vec![answered("answered:p1", true, "indirect"), fixed.clone()];
    let (ev, refused) = import_events(&m.book, &first, m.day, &[]);
    m.commit_events(ev).map_err(|e| e.to_string())?;
    let rid = record_id(&first[0]);
    if !refused.is_empty() || m.book.records.get(&rid).map(|r| (r.state, r.verdicts.len())) != Some((State::Active, 1)) || m.book.records.len() < 2 {
        return Err(format!("outcomes: first import gave {:?}", m.book.records.get(&rid).map(|r| (r.state, r.verdicts.len()))));
    }
    if !import_events(&m.book, &first, m.day, &[]).0.is_empty() {
        return Err("outcomes: importing the same outcomes again wrote events".into());
    }
    let more = vec![answered("answered:p2", true, "weak"), answered("answered:p3", false, "direct")];
    let (ev, _) = import_events(&m.book, &more, m.day, &[]);
    m.commit_events(ev).map_err(|e| e.to_string())?;
    let r = m.book.records.get(&rid).ok_or("outcomes: answered record missing")?;
    if r.state != State::Invalid || r.verdicts.len() != 3 || r.class != Class::Protected || r.retention.uses != 3 {
        return Err(format!("outcomes: after a refutation state {:?}, verdicts {}, uses {}", r.state, r.verdicts.len(), r.retention.uses));
    }
    drop(m);
    // A ledger with one outcome not yet imported: folded in, and it turns the record over again.
    let row = Value::Object(answered("answered:p4", true, "direct").fields().into_iter().map(|(k, v)| (k.to_owned(), v)).collect());
    std::fs::write(dir.join(LEDGER), format!("{}\n", row.encode().map_err(|e| e.to_string())?)).map_err(|e| e.to_string())?;
    // The ledger keeps the question's hash; its text is in the situation record.
    std::fs::write(dir.join(SITUATION), "{\"kind\":\"nano\",\"nano\":{\"tool\":\"nm_context_pack\",\"ask\":\"where is the order total computed\"}}\n")
        .map_err(|e| e.to_string())?;
    if row.get("question").is_ok() || row.get("ask_sha").is_err() {
        return Err("outcomes: the ledger row keeps the question's text".into());
    }
    let k = known(&dir);
    let a = k.iter().find(|x| x.record == rid).ok_or("outcomes: answered not known")?;
    let f = k.iter().find(|x| x.kind == "fixed").ok_or("outcomes: fixed not known")?;
    if (a.verified, a.refuted, a.standing, a.strength.as_str()) != (3, 1, true, "direct") || f.failure.as_ref().map(|x| x.code.as_str()) != Some("E0308") || !f.standing {
        return Err(format!("outcomes: known {a:?}"));
    }
    if recall(&k, "where does the order total get computed?").first().map(|x| x.1.record.as_str()) != Some(rid.as_str())
        || !recall(&k, "rename the config loader").is_empty()
    {
        return Err("outcomes: recall of a similar and an unrelated question".into());
    }
    if !fresh(&root, &place) {
        return Err("outcomes: unchanged lines read as stale".into());
    }
    std::fs::write(root.join("src/lib.rs"), body.replace("x * 2", "x * 3")).map_err(|e| e.to_string())?;
    if fresh(&root, &place) {
        return Err("outcomes: changed lines read as fresh".into());
    }
    let mut m = crate::mem2::Memory2::open(&dir, &root).map_err(|e| e.to_string())?;
    let (n, _) = import_ledger(&mut m, &dir).map_err(|e| e.to_string())?;
    if n == 0 || import_ledger(&mut m, &dir).map_err(|e| e.to_string())?.0 != 0 {
        return Err(format!("outcomes: ledger import wrote {n} events, then wrote again"));
    }
    Ok(())
}
