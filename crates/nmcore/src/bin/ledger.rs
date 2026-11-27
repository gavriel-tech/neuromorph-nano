//! The evidence ledger: what followed each context pack the Nano served, joined to it by stated
//! rules, kept per project in `<project memory>/evidence.jsonl`.
//!
//! Two inputs are joined. The Nano's audit log (`memory/audit.jsonl`) holds one row per context
//! pack it served: its id, the places it cited, the steps that ran, the serving mode. The
//! situation record (`situation.jsonl`, written by the hook intake) holds what Claude Code did:
//! prompts, reads, edits, commands, and its own call of the Nano's tool. A pack is found in the
//! situation record by the episode id of its reply, else by its question and time, else by time.
//!
//! A pack's window is what happened after it in the same session: the rest of its turn (up to the
//! next prompt), the next prompt, and the next turn (up to its end). The window closes when the
//! next turn ends, when the session ends or is cleared, after an hour with no record in the
//! session, or a day after the pack. Each rule then gives one row, with a state from the design's
//! states (used, passed-over, verified, refuted) and a strength (direct, indirect, weak), or
//! `none` (the rule found nothing) or `unknown` (the rule could not be judged, with the reason).
//!
//! The file is append-only. Each pack gets a `served` row as soon as it is seen and its evidence
//! rows once its window closes, all at once. Each line carries the previous line's hash and its
//! own (SHA-256 of the line without `h`), as the memory journal does, so a changed or removed
//! line is found. Only paths, line ranges, check classes, outcome flags and hashes are kept: never
//! file contents, command output, command text or prompt text.
//!
//! Primes and the agent's guidance are judged the same way. A prime is a pack whose cited places
//! are the prime's places (`PRIME_RULES`). Each guidance delivery the hook records name (a
//! reminder, a step nudge, a loop continuation) gets a `guided` row at once and one evidence row
//! by its kind's rule (`GUIDANCE_RULES`) once its window, the rest of its session, closes.
use crate::sessions::{Rec, kept_prompts, records};
use nmjson::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// The ledger's file in a project memory folder.
pub const FILE: &str = "evidence.jsonl";
const LOCK: &str = "evidence.lock";
const CLEARED: &str = "evidence.cleared";
/// A window closes after this long with no record in its session.
const IDLE_MS: u64 = 60 * 60 * 1000;
/// A window closes this long after its pack whatever else happens.
const MAX_WINDOW_MS: u64 = 24 * 60 * 60 * 1000;
/// A lock older than this was left by a process that stopped; it is taken over.
const STALE_LOCK_MS: u64 = 60 * 1000;
/// The hardware description is read again after this long.
const HARDWARE_MS: u64 = 24 * 60 * 60 * 1000;
const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";
/// The rules, in the order their rows are written.
pub const RULES: [&str; 7] = ["read", "edit", "elsewhere", "check", "revert", "next-prompt", "asked-again"];

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}
fn obj(f: Vec<(&str, Value)>) -> Value {
    Value::Object(f.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn opt(v: Option<&str>) -> Value {
    v.map_or(Value::Null, s)
}
fn text(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::string).ok().map(str::to_owned)
}
fn sha16(t: &str) -> String {
    nmhash::sha256_bytes(t.as_bytes()).chars().take(16).collect()
}

/// The ledger file of a project memory folder.
#[must_use]
pub fn ledger_file(memory: &Path) -> PathBuf {
    memory.join(FILE)
}

// ---------------------------------------------------------------------------------------------
// What the audit log says the Nano served.

/// A context pack as the audit log recorded it.
#[derive(Clone, Debug, Default)]
pub struct Served {
    pub id: String,
    pub time_ms: u64,
    pub ms: u64,
    pub mode: String,
    pub via: String,
    /// The serving process (`<pid>-<start ms>`).
    pub server: String,
    /// The parts that ran, each once, in run order (`nmpod.unit#t3` counts as `nmpod.unit`).
    pub steps: Vec<String>,
    pub sources: Vec<String>,
    pub episode: Option<String>,
    pub argument: Option<String>,
    pub withheld: Option<bool>,
}

impl Served {
    fn from_value(v: &Value) -> Option<Self> {
        let mut steps: Vec<String> = Vec::new();
        for st in v.get("steps").and_then(Value::array).unwrap_or_default() {
            if let Some(p) = text(st, "part") {
                let base = p.split('#').next().unwrap_or(&p).to_owned();
                if !steps.contains(&base) {
                    steps.push(base);
                }
            }
        }
        Some(Self {
            id: text(v, "id")?,
            time_ms: v.get("time_ms").and_then(Value::integer).ok()?,
            ms: v.get("ms").and_then(Value::integer).unwrap_or(0),
            mode: text(v, "mode").unwrap_or_default(),
            via: text(v, "via").unwrap_or_default(),
            server: text(v, "session").unwrap_or_default(),
            steps,
            sources: v.get("sources").and_then(Value::array).unwrap_or_default().iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect(),
            episode: text(v, "episode"),
            argument: text(v, "argument"),
            withheld: match v.get("withheld") {
                Ok(Value::Bool(b)) => Some(*b),
                _ => None,
            },
        })
    }
    /// When the reply was ready.
    #[must_use]
    pub fn end(&self) -> u64 {
        self.time_ms.saturating_add(self.ms)
    }
}

/// The audit log's context packs of one project and every setting change.
#[derive(Default)]
pub struct AuditIndex {
    pub packs: Vec<Served>,
    /// Setting changes: time, key, new value.
    pub settings: Vec<(u64, String, String)>,
}

/// Reads the audit log of the Nano folder `home` for the project labelled `label`. Only lines
/// that can matter are parsed (setting changes, and context pack rows of this project).
#[must_use]
pub fn audit_index(home: &Path, label: &str) -> AuditIndex {
    let text_all = std::fs::read_to_string(home.join("memory").join("audit.jsonl")).unwrap_or_default();
    let project = format!("\"project\":\"{label}\"");
    let mut idx = AuditIndex::default();
    let mut at: BTreeMap<String, usize> = BTreeMap::new();
    for l in text_all.lines() {
        if l.contains("\"kind\":\"setting\"") {
            if let Ok(v) = nmjson::parse(l.as_bytes())
                && let (Some(k), Some(to), Ok(t)) = (text(&v, "key"), text(&v, "to"), v.get("time_ms").and_then(Value::integer))
            {
                idx.settings.push((t, k, to));
            }
            continue;
        }
        if !l.contains("\"tool\":\"nm_context_pack\"") || !l.contains(&project) {
            continue;
        }
        let Ok(v) = nmjson::parse(l.as_bytes()) else { continue };
        if v.get("kind").and_then(Value::string).ok() != Some("tool") || v.get("state").and_then(Value::string).is_ok_and(|x| x != "done") {
            continue;
        }
        let Some(p) = Served::from_value(&v) else { continue };
        match at.get(&p.id) {
            Some(&i) => {
                if let Some(slot) = idx.packs.get_mut(i) {
                    *slot = p;
                }
            }
            None => {
                at.insert(p.id.clone(), idx.packs.len());
                idx.packs.push(p);
            }
        }
    }
    idx
}

/// The settings in effect for a pack, replayed from the audit log's setting changes: controls
/// that apply at the next start of the server as of that server's start, the answer mode as of
/// the call; cartridges switched on as of the server's start; and the serving path.
#[must_use]
pub fn settings_at(idx: &AuditIndex, p: &Served) -> Value {
    let start = p.server.rsplit('-').next().and_then(|x| x.parse::<u64>().ok()).unwrap_or(p.time_ms);
    let keys = ["precision.base", "precision.writer", "precision.reranker", "precision.embedder", "memory", "dense_cache", "mode"];
    let mut vals: BTreeMap<String, String> =
        crate::controls::CONTROLS.iter().filter(|c| keys.contains(&c.key)).map(|c| (c.key.to_owned(), c.default.to_owned())).collect();
    let mut carts: BTreeMap<String, bool> = BTreeMap::new();
    for (t, key, to) in &idx.settings {
        let cutoff = if key == "mode" { p.time_ms } else { start };
        if *t > cutoff {
            continue;
        }
        if let Some(id) = key.strip_prefix(crate::controls::CARTRIDGE_KEY) {
            carts.insert(id.to_owned(), to == "on");
        } else if vals.contains_key(key) {
            vals.insert(key.clone(), to.clone());
        }
    }
    let mut o: BTreeMap<String, Value> = vals.into_iter().map(|(k, v)| (k, s(&v))).collect();
    o.insert("path".into(), s(&p.mode));
    o.insert("via".into(), s(&p.via));
    o.insert("cartridges".into(), Value::Array(carts.into_iter().filter(|c| c.1).map(|c| s(&c.0)).collect()));
    Value::Object(o)
}

/// A cited place, `path:a-b`, `path:a` or `path`, as a path and its lines (none: the whole file).
#[must_use]
pub fn place(src: &str) -> (String, Option<(u64, u64)>) {
    let src = src.trim_end_matches(';');
    if let Some((p, r)) = src.rsplit_once(':') {
        let span = match r.split_once('-') {
            Some((a, b)) => a.parse().ok().zip(b.parse().ok()),
            None => r.parse().ok().map(|a| (a, a)),
        };
        if let Some(sp) = span {
            return (p.to_owned(), Some(sp));
        }
    }
    (src.to_owned(), None)
}

fn place_text(p: &str, span: Option<(u64, u64)>) -> String {
    match span {
        Some((a, b)) if a == b => format!("{p}:{a}"),
        Some((a, b)) => format!("{p}:{a}-{b}"),
        None => p.to_owned(),
    }
}

// ---------------------------------------------------------------------------------------------
// Packs in the situation record.

/// A context pack Claude Code received, as the hook record saw it, joined to its audit row.
#[derive(Clone, Debug)]
pub struct Pack {
    pub rec: Rec,
    /// When the pack was handed over: the audit row's end when joined, else the hook's time.
    pub at: u64,
    /// The audit id when joined, else `hook:` and the call's id.
    pub id: String,
    pub served: Option<Served>,
    /// How the hook record was joined to the audit row: episode, question and time, time, none.
    pub joined_by: &'static str,
    pub cited: Vec<(String, Option<(u64, u64)>)>,
}

fn norm(t: &str) -> String {
    t.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

/// The context packs in a project's records, each joined to its audit row where one is found.
/// Each audit row is joined at most once.
#[must_use]
pub fn packs(recs: &[Rec], idx: &AuditIndex) -> Vec<Pack> {
    let mut claimed: BTreeSet<String> = BTreeSet::new();
    let mut out = Vec::new();
    for r in recs.iter().filter(|r| r.done() && r.nano.as_ref().is_some_and(|x| x.0 == "nm_context_pack")) {
        let (_, ask, episode) = r.nano.clone().unwrap_or_default();
        let free = |p: &&Served| !claimed.contains(&p.id);
        let near = |p: &&Served| p.end() <= r.t.saturating_add(2000);
        let by_episode = episode.as_ref().and_then(|e| idx.packs.iter().filter(free).find(|p| p.episode.as_ref() == Some(e)));
        let by_ask = || {
            let a = norm(ask.as_deref()?);
            idx.packs
                .iter()
                .filter(free)
                .filter(near)
                .filter(|p| p.end().saturating_add(600_000) >= r.t && p.argument.as_deref().map(norm).as_deref() == Some(a.as_str()))
                .min_by_key(|p| r.t.abs_diff(p.end()))
        };
        let by_time = || idx.packs.iter().filter(free).filter(near).filter(|p| p.end().saturating_add(120_000) >= r.t).min_by_key(|p| r.t.abs_diff(p.end()));
        let (served, joined_by) = match (by_episode, by_ask()) {
            (Some(p), _) => (Some(p.clone()), "episode"),
            (None, Some(p)) => (Some(p.clone()), "question and time"),
            (None, None) => match by_time() {
                Some(p) => (Some(p.clone()), "time"),
                None => (None, "none"),
            },
        };
        if let Some(p) = &served {
            claimed.insert(p.id.clone());
        }
        let id = served.as_ref().map_or_else(
            || match &r.use_id {
                Some(u) => format!("hook:{u}"),
                None => format!("hook:{}-{}", r.session.chars().take(8).collect::<String>(), r.t),
            },
            |p| p.id.clone(),
        );
        let cited = served.as_ref().map(|p| p.sources.iter().map(|x| place(x)).collect()).unwrap_or_default();
        out.push(Pack { at: served.as_ref().map_or(r.t, |p| p.end().min(r.t)), rec: r.clone(), id, served, joined_by, cited });
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Windows and rules.

/// What followed a pack in its session.
pub struct Window<'a> {
    /// The rest of the pack's turn, up to the next prompt.
    pub turn: Vec<&'a Rec>,
    pub next_prompt: Option<&'a Rec>,
    /// The next turn, up to its end.
    pub next_turn: Vec<&'a Rec>,
    /// The prompt of the pack's own turn.
    pub own_prompt: Option<&'a Rec>,
    /// Why the window is closed, or `None` while it is open.
    pub closed: Option<&'static str>,
}

fn closing(r: &Rec) -> Option<&'static str> {
    if r.event == "SessionEnd" {
        return Some("session ended");
    }
    if r.kind == "session" && matches!(r.source.as_deref(), Some("clear" | "startup")) {
        return Some("conversation cleared or started again");
    }
    None
}

/// The window of `p` among the records of a project, as of `now`.
#[must_use]
pub fn window<'a>(recs: &'a [Rec], p: &Pack, now: u64) -> Window<'a> {
    let mine: Vec<&Rec> = recs.iter().filter(|r| r.session == p.rec.session).collect();
    let own = |r: &Rec| r.i == p.rec.i || (r.use_id.is_some() && r.use_id == p.rec.use_id);
    let mut w = Window {
        turn: Vec::new(),
        next_prompt: None,
        next_turn: Vec::new(),
        own_prompt: mine.iter().copied().rev().find(|r| r.kind == "prompt" && (r.t, r.i) < (p.rec.t, p.rec.i)),
        closed: None,
    };
    for r in mine.iter().copied().filter(|r| r.t > p.at && !own(r)) {
        if let Some(why) = closing(r) {
            w.closed = Some(why);
            break;
        }
        match (w.next_prompt.is_some(), r.kind.as_str()) {
            (false, "prompt") => w.next_prompt = Some(r),
            (false, _) => w.turn.push(r),
            (true, "prompt") => {
                w.closed = Some("next turn ended");
                break;
            }
            (true, "stop") if r.event == "Stop" => {
                w.next_turn.push(r);
                w.closed = Some("next turn ended");
                break;
            }
            (true, _) => w.next_turn.push(r),
        }
    }
    let last = mine.last().map_or(p.at, |r| r.t);
    if w.closed.is_none() {
        if now >= last.saturating_add(IDLE_MS) {
            w.closed = Some("an hour with no activity in the session");
        } else if now >= p.at.saturating_add(MAX_WINDOW_MS) {
            w.closed = Some("a day after the pack");
        }
    }
    w
}

/// One join: a rule's finding for a pack.
#[derive(Clone, Debug)]
pub struct Ev {
    pub rule: &'static str,
    /// used, passed-over, verified, refuted, none, unknown.
    pub state: &'static str,
    /// direct, indirect, weak; none for `none` and `unknown`.
    pub strength: Option<&'static str>,
    /// The record that is the evidence, and the place in it that matched.
    pub event: Option<(Rec, Option<String>)>,
    /// Completed tool calls between the pack and the evidence.
    pub steps_after: Option<usize>,
    pub why: Option<String>,
    pub detail: Vec<(&'static str, Value)>,
}

impl Ev {
    fn found(rule: &'static str, state: &'static str, strength: &'static str, r: &Rec, at: Option<String>, steps: usize) -> Self {
        Self { rule, state, strength: Some(strength), event: Some((r.clone(), at)), steps_after: Some(steps), why: None, detail: Vec::new() }
    }
    fn not(rule: &'static str, state: &'static str, why: &str) -> Self {
        Self { rule, state, strength: None, event: None, steps_after: None, why: Some(why.to_owned()), detail: Vec::new() }
    }
    fn with(mut self, k: &'static str, v: Value) -> Self {
        self.detail.push((k, v));
        self
    }
}

/// Line ranges overlap.
fn overlap(a: (u64, u64), b: (u64, u64)) -> bool {
    a.0 <= b.1 && b.0 <= a.1
}

/// Where a record of kind `kinds` touched a cited place: (strength, cited path, place touched).
/// Direct when the lines touched overlap the cited lines (a cited whole file overlaps any lines);
/// indirect when only the file is the same (no lines recorded, or other lines of it).
fn touch(r: &Rec, cited: &[(String, Option<(u64, u64)>)], extra: &[(String, u64, u64)]) -> Option<(&'static str, String, String)> {
    let mut best: Option<(&'static str, String, String)> = None;
    for (path, span) in cited {
        if !r.paths.contains(path) {
            continue;
        }
        let mut spans: Vec<(u64, u64)> = r.span_of(path).into_iter().collect();
        spans.extend(extra.iter().filter(|e| &e.0 == path).map(|e| (e.1, e.2)));
        let hit = spans.iter().copied().find(|sp| span.is_none_or(|c| overlap(c, *sp)));
        match (hit, &best) {
            (Some(sp), _) => return Some(("direct", path.clone(), place_text(path, Some(sp)))),
            (None, None) => best = Some(("indirect", path.clone(), place_text(path, spans.first().copied()))),
            _ => {}
        }
    }
    best
}

/// Words at the start, and phrases anywhere, of a next prompt that say the user is correcting
/// course or accepting the result. Matching is on lower-cased text; a prompt matching both is
/// neither.
const CORRECTION_START: &[&str] = &["no", "nope", "wrong", "stop", "actually", "wait", "undo", "revert"];
const CORRECTION_ANY: &[&str] = &[
    "that's not", "that is not", "not what i", "not that", "wrong", "incorrect", "undo", "revert", "go back", "you broke", "didn't work",
    "doesn't work", "does not work", "still fails", "still failing", "still broken", "not right",
];
const ACCEPT_START: &[&str] = &["thanks", "thank you", "great", "perfect", "nice", "lgtm", "looks good", "good", "works"];
const ACCEPT_ANY: &[&str] = &["that works", "it works", "works now", "looks good", "well done", "thank you", "thanks"];

/// The first marker of a correction and of an acceptance found in a prompt.
#[must_use]
pub fn markers(prompt: &str) -> (Option<&'static str>, Option<&'static str>) {
    let t = prompt.trim().to_lowercase();
    let first: String = t.chars().take_while(|c| c.is_alphanumeric() || *c == '\'').collect();
    let starts = |list: &[&'static str]| list.iter().copied().find(|m| if m.contains(' ') { t.starts_with(m) } else { first == *m });
    let any = |list: &[&'static str]| list.iter().copied().find(|m| t.contains(m));
    (starts(CORRECTION_START).or_else(|| any(CORRECTION_ANY)), starts(ACCEPT_START).or_else(|| any(ACCEPT_ANY)))
}

/// The joins of one pack over its window, one per rule in `RULES` order. Kept prompt text by hash
/// is needed only for the next-prompt rule.
#[must_use]
pub fn judge(recs: &[Rec], p: &Pack, w: &Window, prompts: &BTreeMap<String, String>) -> Vec<Ev> {
    // Before-the-edit lines of each edit, by tool call id (the after-the-call record holds the lines
    // where the new text is; the cited lines refer to the file before the edit).
    let pre: BTreeMap<&str, Vec<(String, u64, u64)>> = recs
        .iter()
        .filter(|r| r.event == "PreToolUse" && r.session == p.rec.session)
        .filter_map(|r| Some((r.use_id.as_deref()?, r.lines.clone())))
        .collect();
    let steps_before = |seq: &[&Rec], r: &Rec| seq.iter().take_while(|x| x.i != r.i).filter(|x| x.done()).count();
    let turn: Vec<&Rec> = w.turn.iter().copied().filter(|r| r.done() && r.ok == Some(true)).collect();
    let no_cited = if p.served.is_none() { "the pack was not found in the audit log" } else { "the pack cited no places" };
    let used = |kinds: &[&str], rule: &'static str| -> Ev {
        if p.cited.is_empty() {
            return Ev::not(rule, "unknown", no_cited);
        }
        let mut found: Option<Ev> = None;
        for r in turn.iter().filter(|r| kinds.contains(&r.kind.as_str())) {
            let extra = r.use_id.as_deref().and_then(|u| pre.get(u)).cloned().unwrap_or_default();
            if let Some((strength, _, at)) = touch(r, &p.cited, &extra) {
                let ev = Ev::found(rule, "used", strength, r, Some(at), steps_before(&w.turn, r));
                if strength == "direct" {
                    return ev;
                }
                found.get_or_insert(ev);
            }
        }
        found.unwrap_or_else(|| Ev::not(rule, "none", "no cited place was touched this way in the turn"))
    };
    let read = used(&["read"], "read");
    let edit = used(&["edit", "write"], "edit");
    let elsewhere = if p.cited.is_empty() {
        Ev::not("elsewhere", "unknown", no_cited)
    } else if read.state == "used" || edit.state == "used" {
        Ev::not("elsewhere", "none", "a cited place was used")
    } else {
        let cited: BTreeSet<&str> = p.cited.iter().map(|c| c.0.as_str()).collect();
        let mut went: Vec<String> = Vec::new();
        let mut first: Option<&Rec> = None;
        for pass in [&["edit", "write"][..], &["read"][..]] {
            for r in turn.iter().filter(|r| pass.contains(&r.kind.as_str())) {
                for x in r.paths.iter().filter(|x| !cited.contains(x.as_str()) && !matches!(x.as_str(), "." | "<outside>")) {
                    if !went.contains(x) && went.len() < 8 {
                        went.push(x.clone());
                        first = Some(first.map_or(*r, |f| if (r.t, r.i) < (f.t, f.i) { *r } else { f }));
                    }
                }
            }
        }
        match first {
            Some(r) => Ev::found("elsewhere", "passed-over", "indirect", r, None, steps_before(&w.turn, r))
                .with("went", Value::Array(went.iter().map(|x| s(x)).collect())),
            None => Ev::not("elsewhere", "unknown", "nothing was read or edited in the turn"),
        }
    };
    let checks: Vec<&Rec> = w.turn.iter().copied().filter(|r| r.done() && matches!(r.kind.as_str(), "build" | "test")).collect();
    let edited_at = edit.event.as_ref().filter(|_| edit.state == "used").map(|e| (e.0.t, e.0.i));
    let check = match checks.last() {
        None => Ev::not("check", "unknown", "no build or test ran in the turn"),
        Some(last) => {
            let after_edit = edited_at.is_some_and(|e| e < (last.t, last.i));
            let state = match last.ok {
                Some(true) => "verified",
                Some(false) => "refuted",
                None => "unknown",
            };
            let passed = checks.iter().filter(|c| c.ok == Some(true)).count() as u64;
            Ev::found("check", state, if after_edit { "indirect" } else { "weak" }, last, None, steps_before(&w.turn, last))
                .with("runs", Value::Array(vec![n(passed), n(checks.len() as u64 - passed)]))
                .with("after_cited_edit", Value::Bool(after_edit))
        }
    };
    let revert = match edited_at {
        None => Ev::not("revert", "unknown", "no cited place was edited"),
        Some(e) => {
            let edited: BTreeSet<String> = turn
                .iter()
                .filter(|r| matches!(r.kind.as_str(), "edit" | "write"))
                .flat_map(|r| r.paths.iter().filter(|x| p.cited.iter().any(|c| &c.0 == *x)).cloned())
                .collect();
            let undo: Vec<&Rec> = w
                .turn
                .iter()
                .copied()
                .filter(|r| r.done() && (r.t, r.i) > e && matches!(r.cmd_class.as_deref(), Some("git-revert" | "git-restore" | "git-checkout" | "git-reset" | "git-stash")))
                .collect();
            if let Some(r) = undo.iter().find(|r| r.paths.iter().any(|x| edited.contains(x))) {
                Ev::found("revert", "refuted", "direct", r, r.paths.iter().find(|x| edited.contains(*x)).cloned(), steps_before(&w.turn, r))
            } else if let Some(r) = undo.iter().find(|r| r.paths.is_empty() || r.paths.iter().any(|x| x == ".")) {
                Ev::found("revert", "refuted", "weak", r, None, steps_before(&w.turn, r))
            } else {
                Ev::not("revert", "none", "no undo of the edited cited files in the turn")
            }
        }
    };
    let next_prompt = next_prompt_rule(w, prompts);
    let asked_again = asked_again_rule(p, w);
    vec![read, edit, elsewhere, check, revert, next_prompt, asked_again]
}

fn next_prompt_rule(w: &Window, prompts: &BTreeMap<String, String>) -> Ev {
    let Some(np) = w.next_prompt else {
        return Ev::not("next-prompt", "unknown", "no next prompt before the window closed");
    };
    let steps = w.turn.iter().filter(|x| x.done()).count();
    let Some(t) = np.prompt.as_ref().and_then(|x| prompts.get(&x.0)) else {
        return Ev::not("next-prompt", "unknown", "the next prompt's text is not kept");
    };
    match markers(t) {
        (Some(m), None) => Ev::found("next-prompt", "refuted", "weak", np, None, steps).with("marker", s(m)),
        (None, Some(m)) => Ev::found("next-prompt", "verified", "weak", np, None, steps).with("marker", s(m)),
        (Some(_), Some(_)) => Ev::not("next-prompt", "none", "signs of both correction and acceptance"),
        (None, None) => Ev::not("next-prompt", "none", "no sign of correction or acceptance"),
    }
}

fn asked_again_rule(p: &Pack, w: &Window) -> Ev {
    let Some(ask) = p.rec.nano.as_ref().and_then(|x| x.1.as_deref()).map(norm) else {
        return Ev::not("asked-again", "unknown", "the question was not recorded");
    };
    let seq: Vec<&Rec> = w.turn.iter().copied().chain(w.next_prompt).chain(w.next_turn.iter().copied()).collect();
    let again = seq.iter().find(|r| r.done() && r.nano.as_ref().is_some_and(|x| x.0 == "nm_context_pack" && x.1.as_deref().map(norm).as_deref() == Some(ask.as_str())));
    let steps_before = |r: &Rec| seq.iter().take_while(|x| x.i != r.i).filter(|x| x.done()).count();
    if let Some(r) = again {
        return Ev::found("asked-again", "refuted", "indirect", r, None, steps_before(r)).with("by", s("the same question to the Nano"));
    }
    let sha = |r: Option<&Rec>| r.and_then(|x| x.prompt.as_ref().map(|q| q.0.clone()));
    if let (Some(a), Some(b)) = (sha(w.own_prompt), sha(w.next_prompt))
        && a == b
        && let Some(np) = w.next_prompt
    {
        return Ev::found("asked-again", "refuted", "weak", np, None, steps_before(np)).with("by", s("the same prompt from the user"));
    }
    if w.closed.is_some() || w.next_prompt.is_some() {
        Ev::not("asked-again", "none", "the question was not asked again")
    } else {
        Ev::not("asked-again", "none", "not asked again so far")
    }
}

// ---------------------------------------------------------------------------------------------
// Hardware.

fn run_quiet(cmd: &str, args: &[&str]) -> Option<String> {
    let mut c = std::process::Command::new(cmd);
    c.args(args).stdin(std::process::Stdio::null()).stderr(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt as _;
        // No console window for a program started from a hook.
        c.creation_flags(0x0800_0000);
    }
    let o = c.output().ok()?;
    o.status.success().then(|| String::from_utf8_lossy(&o.stdout).trim().to_owned())
}

/// This machine's hardware pieces: processor, logical threads, graphics card and its memory, RAM.
/// Read without opening the graphics card (the driver's `nvidia-smi` names it), cached in
/// `memory/hardware.json` for a day. `slow` allows reading the RAM size where that needs a slow
/// system query (Windows); without it a cached size is used, or none. `id` names the processor,
/// threads and graphics card together, so a change of RAM size alone does not change it.
#[must_use]
pub fn hardware(home: &Path, slow: bool) -> Value {
    let file = home.join("memory").join("hardware.json");
    let cached = std::fs::read(&file).ok().and_then(|b| nmjson::parse(&b).ok());
    let now = nmtype::now_ms();
    if let Some(c) = &cached {
        let fresh = c.get("t").and_then(Value::integer).is_ok_and(|t| now.saturating_sub(t) < HARDWARE_MS);
        let ram_known = !matches!(c.get("ram_mib"), Ok(Value::Null) | Err(_));
        if fresh && (ram_known || !slow) {
            return c.clone();
        }
    }
    let cpu = std::env::var("PROCESSOR_IDENTIFIER").ok().filter(|x| !x.is_empty()).or_else(|| {
        std::fs::read_to_string("/proc/cpuinfo").ok()?.lines().find(|l| l.starts_with("model name"))?.split_once(':').map(|x| x.1.trim().to_owned())
    });
    let threads = std::thread::available_parallelism().map_or(0, std::num::NonZero::get) as u64;
    let gpu_line = run_quiet("nvidia-smi", &["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]);
    let (gpu, gpu_mib) = gpu_line.as_deref().and_then(|l| l.lines().next()).map_or((None, None), |l| {
        let mut it = l.split(',').map(str::trim);
        (it.next().map(str::to_owned), it.next().and_then(|m| m.parse::<u64>().ok()))
    });
    let ram_mib = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|t| t.lines().find(|l| l.starts_with("MemTotal:"))?.split_whitespace().nth(1)?.parse::<u64>().ok())
        .map(|kb| kb / 1024)
        .or_else(|| {
            if cfg!(windows) && slow {
                run_quiet("powershell", &["-NoProfile", "-NonInteractive", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])?
                    .parse::<u64>()
                    .ok()
                    .map(|b| b / (1024 * 1024))
            } else {
                cached.as_ref().and_then(|c| c.get("ram_mib").and_then(Value::integer).ok())
            }
        });
    let id = sha16(&format!("{}|{threads}|{}|{}", cpu.as_deref().unwrap_or("?"), gpu.as_deref().unwrap_or("none"), gpu_mib.unwrap_or(0)));
    let v = obj(vec![
        ("id", s(&id)),
        ("cpu", opt(cpu.as_deref())),
        ("threads", n(threads)),
        ("gpu", opt(gpu.as_deref())),
        ("gpu_mib", gpu_mib.map_or(Value::Null, n)),
        ("ram_mib", ram_mib.map_or(Value::Null, n)),
        ("t", n(now)),
        ("note", s("This machine's hardware as the evidence ledger tags it; read again after a day. Written by the Neuromorph Nano.")),
    ]);
    if let Some(d) = file.parent()
        && std::fs::create_dir_all(d).is_ok()
    {
        let _ = std::fs::write(&file, nmjson::pretty::pretty(&v));
    }
    v
}

/// The hardware a pack ran on: the machine's pieces and which of them the serving path uses (the
/// plain path runs on the processor; the neural and whole paths also on the graphics card when
/// there is one). Inferred from the serving mode, as the audit row does not name a device.
fn hardware_of(hw: &Value, mode: &str) -> Value {
    let has_gpu = !matches!(hw.get("gpu"), Ok(Value::Null) | Err(_));
    let mut used = vec![s("cpu")];
    if has_gpu && mode != "plain" && !mode.is_empty() {
        used.push(s("gpu"));
    }
    let mut o = match hw {
        Value::Object(m) => m.iter().filter(|(k, _)| !matches!(k.as_str(), "t" | "note")).map(|(k, v)| (k.clone(), v.clone())).collect(),
        _ => BTreeMap::new(),
    };
    o.insert("used".into(), Value::Array(used));
    o.insert("used_from".into(), s("serving mode"));
    Value::Object(o)
}

// ---------------------------------------------------------------------------------------------
// Rows and the file.

fn event_value(r: &Rec, at: Option<&str>) -> Value {
    let mut f = vec![("at", n(r.t)), ("event", s(&r.event)), ("kind", s(&r.kind)), ("tool", s(&r.tool))];
    if let Some(a) = at {
        f.push(("place", s(a)));
    }
    if let Some(c) = &r.cmd_class {
        f.push(("check", s(c)));
    }
    if let Some(p) = &r.pkg {
        f.push(("pkg", s(p)));
    }
    if let Some(ok) = r.ok {
        f.push(("ok", Value::Bool(ok)));
    }
    if let Some(e) = r.exit {
        f.push(("exit", Value::Number(e.to_string())));
    }
    if let Some((a, b)) = r.tests {
        f.push(("tests", Value::Array(vec![n(a), n(b)])));
    }
    obj(f)
}

fn served_row(p: &Pack, idx: &AuditIndex, hw: &Value) -> Vec<(&'static str, Value)> {
    let sv = p.served.as_ref();
    vec![
        ("row", s("served")),
        ("pack", s(&p.id)),
        ("session", s(&p.rec.session)),
        ("at", n(p.at)),
        ("tool", s("nm_context_pack")),
        ("joined_by", s(p.joined_by)),
        ("episode", opt(p.rec.nano.as_ref().and_then(|x| x.2.as_deref()).or_else(|| sv.and_then(|x| x.episode.as_deref())))),
        ("ask_sha", opt(p.rec.nano.as_ref().and_then(|x| x.1.as_deref()).map(|a| sha16(&norm(a))).as_deref())),
        ("cited", Value::Array(sv.map(|x| x.sources.iter().map(|y| s(y)).collect()).unwrap_or_default())),
        ("steps", Value::Array(sv.map(|x| x.steps.iter().map(|y| s(y)).collect()).unwrap_or_default())),
        ("settings", sv.map_or(Value::Null, |x| settings_at(idx, x))),
        ("withheld", sv.and_then(|x| x.withheld).map_or(Value::Null, Value::Bool)),
        ("hardware", hardware_of(hw, sv.map_or("", |x| x.mode.as_str()))),
    ]
}

fn evidence_row(p: &Pack, e: &Ev, closed: &str, hw_id: &str) -> Vec<(&'static str, Value)> {
    let mut f = vec![
        ("row", s("evidence")),
        ("pack", s(&p.id)),
        ("session", s(&p.rec.session)),
        ("rule", s(e.rule)),
        ("state", s(e.state)),
        ("strength", opt(e.strength)),
        ("event", e.event.as_ref().map_or(Value::Null, |(r, at)| event_value(r, at.as_deref()))),
        ("steps_after", e.steps_after.map_or(Value::Null, |x| n(x as u64))),
        ("closed", s(closed)),
        ("hardware", s(hw_id)),
    ];
    if let Some(w) = &e.why {
        f.push(("why", s(w)));
    }
    if !e.detail.is_empty() {
        f.push(("detail", obj(e.detail.clone())));
    }
    f
}

/// One evaluated pack: its joins as of now, and whether its window is closed.
pub struct PackState {
    pub pack: Pack,
    pub closed: Option<&'static str>,
    pub evs: Vec<Ev>,
}

/// The ledger's rows as read, and every pack's state.
pub struct View {
    pub rows: Vec<Value>,
    /// Rows in the file, and a fault when the chain is broken.
    pub chain: (usize, Option<String>),
    pub torn: Option<String>,
    pub packs: Vec<PackState>,
    /// Every prime's state, judged as packs are (rules `PRIME_RULES`).
    pub primes: Vec<PackState>,
    /// Every guidance delivery's state (rules `GUIDANCE_RULES`).
    pub guided: Vec<GuideState>,
    pub written: usize,
}

/// Reads the ledger and checks its chain: the rows, the last hash, and a torn last line (an
/// incomplete line left by a stop, which is passed over). A damaged line elsewhere is an error.
/// # Errors
/// A line before the last does not parse, or its hashes do not chain.
pub fn read_ledger(file: &Path) -> Result<(Vec<Value>, String, Option<String>), String> {
    let text_all = match std::fs::read_to_string(file) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), GENESIS.to_owned(), None)),
        Err(e) => return Err(format!("{}: {e}", file.display())),
    };
    let mut prev = GENESIS.to_owned();
    let mut rows = Vec::new();
    let mut lines = text_all.split_inclusive('\n').peekable();
    let mut k = 0;
    while let Some(line) = lines.next() {
        k += 1;
        let last = lines.peek().is_none();
        let checked = (|| -> Result<Value, String> {
            if !line.ends_with('\n') {
                return Err("no line end".into());
            }
            let v = nmjson::parse(line.trim_end().as_bytes()).map_err(|e| format!("not JSON: {e}"))?;
            let h = text(&v, "h").ok_or("no hash")?;
            if text(&v, "prev").as_deref() != Some(prev.as_str()) {
                return Err("its previous hash is not the hash of the line before".into());
            }
            let mut body = v.clone();
            if let Value::Object(m) = &mut body {
                m.remove("h");
            }
            if nmhash::sha256_bytes(body.encode().map_err(|e| e.to_string())?.as_bytes()) != h {
                return Err("its hash does not match its content".into());
            }
            Ok(v)
        })();
        match checked {
            Ok(v) => {
                prev = text(&v, "h").unwrap_or_default();
                rows.push(v);
            }
            Err(why) if last => return Ok((rows, prev, Some(format!("line {k}: {why}")))),
            Err(why) => return Err(format!("{}: line {k}: {why}; the ledger was changed after it was written", file.display())),
        }
    }
    Ok((rows, prev, None))
}

/// Holds the ledger's lock file while it lives; removed on drop.
struct Lock(PathBuf);
impl Drop for Lock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Takes the ledger's lock, waiting up to `wait_ms`; `None` when another writer holds it. A lock
/// older than a minute was left by a process that stopped and is taken over.
fn lock(memory: &Path, wait_ms: u64) -> Result<Option<Lock>, String> {
    let p = memory.join(LOCK);
    let start = std::time::Instant::now();
    loop {
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&p) {
            Ok(mut f) => {
                let _ = std::io::Write::write_all(&mut f, format!("{}\n", std::process::id()).as_bytes());
                return Ok(Some(Lock(p)));
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let age = std::fs::metadata(&p).and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).map_or(0, |d| d.as_millis());
                if age > u128::from(STALE_LOCK_MS) {
                    let _ = std::fs::remove_file(&p);
                    continue;
                }
                if start.elapsed().as_millis() >= u128::from(wait_ms) {
                    return Ok(None);
                }
                std::thread::park_timeout(std::time::Duration::from_millis(15));
            }
            Err(e) => return Err(format!("{}: {e}", p.display())),
        }
    }
}

/// Appends rows to the ledger, each chained to the one before. A torn last line is first set
/// aside in `evidence.torn` and cut off, as the memory journal does.
fn append(file: &Path, prev: &str, torn: bool, rows: Vec<Vec<(&'static str, Value)>>, seq0: usize) -> Result<usize, String> {
    if torn {
        let all = std::fs::read(file).map_err(|e| format!("{}: {e}", file.display()))?;
        let keep = all.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
        let side = file.with_extension("torn");
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&side).map_err(|e| format!("{}: {e}", side.display()))?;
        std::io::Write::write_all(&mut f, all.get(keep..).unwrap_or_default()).map_err(|e| e.to_string())?;
        std::io::Write::write_all(&mut f, b"\n").map_err(|e| e.to_string())?;
        let w = std::fs::OpenOptions::new().write(true).open(file).map_err(|e| format!("{}: {e}", file.display()))?;
        w.set_len(keep as u64).map_err(|e| e.to_string())?;
    }
    let mut prev = prev.to_owned();
    let mut out = String::new();
    let count = rows.len();
    let now = nmtype::now_ms();
    for (k, fields) in rows.into_iter().enumerate() {
        let mut m: BTreeMap<String, Value> = fields.into_iter().map(|(a, b)| (a.to_owned(), b)).collect();
        m.insert("v".into(), n(1));
        m.insert("seq".into(), n((seq0 + k + 1) as u64));
        m.insert("t".into(), n(now));
        m.insert("prev".into(), s(&prev));
        let body = Value::Object(m.clone());
        let h = nmhash::sha256_bytes(body.encode().map_err(|e| e.to_string())?.as_bytes());
        m.insert("h".into(), s(&h));
        out.push_str(&Value::Object(m).encode().map_err(|e| e.to_string())?);
        out.push('\n');
        prev = h;
    }
    if count == 0 {
        return Ok(0);
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(file).map_err(|e| format!("{}: {e}", file.display()))?;
    std::io::Write::write_all(&mut f, out.as_bytes()).map_err(|e| format!("{}: {e}", file.display()))?;
    f.sync_all().map_err(|e| format!("{}: {e}", file.display()))?;
    Ok(count)
}

fn cleared_at(memory: &Path) -> u64 {
    std::fs::read_to_string(memory.join(CLEARED)).ok().and_then(|t| t.lines().next().and_then(|l| l.trim().parse().ok())).unwrap_or(0)
}

/// Evaluates every pack of a project (records `recs`) as of `now`, without writing.
#[must_use]
pub fn evaluate(home: &Path, memory: &Path, label: &str, recs: &[Rec], now: u64) -> Vec<PackState> {
    let idx = audit_index(home, label);
    let prompts = kept_prompts(home, memory);
    let cleared = cleared_at(memory);
    packs(recs, &idx)
        .into_iter()
        .filter(|p| p.rec.t > cleared)
        .map(|p| {
            let w = window(recs, &p, now);
            let evs = judge(recs, &p, &w, &prompts);
            PackState { closed: w.closed, evs, pack: p }
        })
        .collect()
}

/// The rules of a prime's window, in the order their rows are written: those of a pack without
/// `asked-again` (a prime asks nothing), and `followed`, which says in one row whether the prime
/// was followed (a primed place read or edited) or passed over.
pub const PRIME_RULES: [&str; 7] = ["read", "edit", "elsewhere", "check", "revert", "next-prompt", "followed"];

/// The primes in a project's records, each as a pack whose cited places are the prime's places.
#[must_use]
pub fn prime_packs(recs: &[Rec]) -> Vec<Pack> {
    recs.iter()
        .filter_map(|r| {
            let x = r.prime.as_ref()?;
            Some(Pack { rec: r.clone(), at: r.t, id: format!("prime:{}", x.id), served: None, joined_by: "prime", cited: x.places.iter().map(|p| place(p)).collect() })
        })
        .collect()
}

/// The `followed` row of a prime: used (with the strength of the read or edit) when a primed place
/// was read or edited in the window; passed over once the window has closed without that (indirect
/// when Claude Code went elsewhere, weak when it read and edited nothing); `none` while it is open.
fn followed_rule(evs: &[Ev], w: &Window) -> Ev {
    let used: Vec<&Ev> = evs.iter().filter(|e| matches!(e.rule, "read" | "edit") && e.state == "used").collect();
    if let Some(u) = used.iter().find(|e| e.strength == Some("direct")).or_else(|| used.first()) {
        return Ev { rule: "followed", ..(*u).clone() };
    }
    if w.closed.is_none() {
        return Ev::not("followed", "none", "no primed place touched so far");
    }
    match evs.iter().find(|e| e.rule == "elsewhere" && e.state == "passed-over") {
        Some(e) => Ev { rule: "followed", ..e.clone() },
        None => Ev {
            rule: "followed",
            state: "passed-over",
            strength: Some("weak"),
            event: None,
            steps_after: None,
            why: Some("nothing was read or edited before the window closed".into()),
            detail: Vec::new(),
        },
    }
}

/// Evaluates every prime of a project as of `now`, without writing.
#[must_use]
pub fn evaluate_primes(home: &Path, memory: &Path, recs: &[Rec], now: u64) -> Vec<PackState> {
    let prompts = kept_prompts(home, memory);
    let cleared = cleared_at(memory);
    prime_packs(recs)
        .into_iter()
        .filter(|p| p.rec.t > cleared)
        .map(|p| {
            let w = window(recs, &p, now);
            let mut evs: Vec<Ev> = judge(recs, &p, &w, &prompts).into_iter().filter(|e| e.rule != "asked-again").collect();
            evs.push(followed_rule(&evs, &w));
            PackState { closed: w.closed, evs, pack: p }
        })
        .collect()
}

fn primed_row(p: &Pack, hw_id: &str) -> Vec<(&'static str, Value)> {
    let x = p.rec.prime.clone().unwrap_or_default();
    vec![
        ("row", s("primed")),
        ("pack", s(&p.id)),
        ("session", s(&p.rec.session)),
        ("at", n(p.at)),
        ("kind", s(&x.kind)),
        ("hook", s(&p.rec.event)),
        ("places", Value::Array(x.places.iter().map(|y| s(y)).collect())),
        ("from", Value::Array(x.from.iter().map(|y| s(y)).collect())),
        ("mode", s(&x.mode)),
        ("hardware", s(hw_id)),
    ]
}

// ---------------------------------------------------------------------------------------------
// The agent's guidance (`guidance.rs`): reminders, step nudges and loop continuations.

/// The rules of guidance deliveries, one per delivery kind (step, loop, reminder):
/// - `step-followed` (a step nudge): used when the step is marked done later in the session (by
///   the Nano after its check passed, direct; or by the user or the agent, direct) or when its
///   check passes later in the session (indirect when an edit under the step's paths came between
///   the nudge and the check, weak otherwise); passed over once the session's window closed
///   without that; `none` while it is open.
/// - `loop-met` (a loop continuation): verified when the loop's condition was met before its cap
///   (a later stop of the session ends it as done), refuted when it reached its cap without the
///   condition (ended as exhausted), stalled (the same unmet reason past its stall threshold) or
///   spent its time or tool budget (ended as budget-spent), all direct; `unknown` when it was ended by hand or the
///   session ended before it was judged again.
/// - `glob-check` (a reminder tied to an edit glob, its `edit` trigger's glob or else its file
///   scope): verified when the last build or test after the last edit under that glob, later in
///   the session, passed, refuted when it failed, both indirect; `unknown` for a reminder with no
///   glob, a reminder no longer set, or no edit under the glob or no check after it before the
///   window closed.
pub const GUIDANCE_RULES: [&str; 3] = ["step-followed", "loop-met", "glob-check"];

/// One delivery of the agent's guidance, as a pack with no cited places (`id` is
/// `guide:<kind>:<item>:<session>:<time>`), its kind and the item delivered.
#[derive(Clone, Debug)]
pub struct Delivery {
    pub pack: Pack,
    /// `step`, `loop` or `reminder`.
    pub kind: &'static str,
    pub item: String,
    /// A loop continuation's iteration and maximum.
    pub iteration: Option<(u64, u64)>,
}

/// One delivery and its judgment as of now.
pub struct GuideState {
    pub d: Delivery,
    /// Why its window is closed, or `None` while it is open.
    pub closed: Option<&'static str>,
    pub ev: Ev,
}

/// The guidance deliveries in a project's records: each reminder delivered, each step nudge and
/// each loop continuation (a Stop the loop blocked), in record order.
#[must_use]
pub fn deliveries(recs: &[Rec]) -> Vec<Delivery> {
    let mut out = Vec::new();
    for r in recs {
        let Some(g) = &r.guidance else { continue };
        let mk = |kind: &'static str, item: &str, iteration: Option<(u64, u64)>| {
            let id = format!("guide:{kind}:{item}:{}:{}", r.session.chars().take(8).collect::<String>(), r.t);
            Delivery { pack: Pack { rec: r.clone(), at: r.t, id, served: None, joined_by: "guidance", cited: Vec::new() }, kind, item: item.to_owned(), iteration }
        };
        out.extend(g.reminders.iter().map(|x| mk("reminder", x, None)));
        if let Some(st) = &g.step {
            out.push(mk("step", st, None));
        }
        out.extend(g.loops.iter().filter(|l| l.1 == "block").map(|l| mk("loop", &l.0, Some((l.2, l.3)))));
    }
    out
}

/// What followed a delivery in its session: every record after it, up to the session's end or
/// clearing. Closed then, or after an hour with no record in the session, or a day after the
/// delivery. With the time the window ends (`now` while it is open).
fn session_after<'a>(recs: &'a [Rec], d: &Delivery, now: u64) -> (Vec<&'a Rec>, Option<&'static str>, u64) {
    let r0 = &d.pack.rec;
    let mine: Vec<&Rec> = recs.iter().filter(|r| r.session == r0.session).collect();
    let mut after = Vec::new();
    let mut closed = None;
    let mut until = now;
    for r in mine.iter().copied().filter(|r| (r.t, r.i) > (r0.t, r0.i)) {
        if let Some(why) = closing(r) {
            closed = Some(why);
            until = r.t;
            break;
        }
        after.push(r);
    }
    let last = mine.last().map_or(r0.t, |r| r.t);
    if closed.is_none() {
        if now >= last.saturating_add(IDLE_MS) {
            closed = Some("an hour with no activity in the session");
            until = last.saturating_add(IDLE_MS);
        } else if now >= r0.t.saturating_add(MAX_WINDOW_MS) {
            closed = Some("a day after the delivery");
            until = r0.t.saturating_add(MAX_WINDOW_MS);
        }
    }
    (after, closed, until)
}

/// Whether a build or test record counts for a step's check (`test` takes a test; `build` takes a
/// build or a test, which builds first).
fn check_fits(check: &str, kind: &str) -> bool {
    match check {
        "test" => kind == "test",
        "build" => matches!(kind, "build" | "test"),
        _ => false,
    }
}

/// The `step-followed` row of a step nudge (`GUIDANCE_RULES`).
fn step_rule(d: &Delivery, after: &[&Rec], (closed, until): (Option<&str>, u64), g: Option<&crate::guidance::Guidance>) -> Ev {
    const RULE: &str = "step-followed";
    let steps_before = |r: &Rec| after.iter().take_while(|x| x.i != r.i).filter(|x| x.done()).count();
    if let Some(r) = after.iter().find(|r| r.guidance.as_ref().is_some_and(|x| x.auto_done.contains(&d.item))) {
        return Ev::found(RULE, "used", "direct", r, None, steps_before(r)).with("by", s("the Nano marked the step done: its check passed after edits under its paths"));
    }
    let step = g.and_then(|g| g.steps.iter().find(|x| x.id == d.item));
    let Some(step) = step else {
        return if closed.is_some() { Ev::not(RULE, "unknown", "the step is no longer in the step graph") } else { Ev::not(RULE, "none", "the step is no longer in the step graph") };
    };
    let same_session = step.evidence.as_ref().and_then(|e| text(e, "session")).is_none_or(|x| x == d.pack.rec.session);
    if step.done
        && let Some(at) = step.done_ms
        && at > d.pack.at
        && at <= until
        && same_session
    {
        return Ev {
            rule: RULE,
            state: "used",
            strength: Some("direct"),
            event: None,
            steps_after: None,
            why: None,
            detail: vec![("by", s(&format!("marked done by the {}", step.done_by.as_deref().unwrap_or("user")))), ("done_ms", n(at))],
        };
    }
    if let Some(check) = step.check.as_deref() {
        let fits = |r: &&&Rec| r.done() && check_fits(check, &r.kind);
        if let Some(c) = after.iter().find(|r| fits(r) && r.ok == Some(true)) {
            let under = |r: &Rec| step.paths.is_empty() || r.paths.iter().any(|p| step.paths.iter().any(|gl| crate::guidance::glob_match(gl, p)));
            let edited = after.iter().take_while(|x| x.i != c.i).any(|r| r.done() && matches!(r.kind.as_str(), "edit" | "write") && r.ok == Some(true) && under(r));
            return Ev::found(RULE, "used", if edited { "indirect" } else { "weak" }, c, None, steps_before(c)).with("by", s(&format!("its check ({check}) passed")));
        }
        if closed.is_some()
            && let Some(c) = after.iter().rev().find(|r| fits(r))
        {
            return Ev::found(RULE, "passed-over", "indirect", c, None, steps_before(c)).with("by", s(&format!("its check ({check}) ran and did not pass")));
        }
    }
    if closed.is_none() {
        return Ev::not(RULE, "none", "the step is not done so far");
    }
    Ev { rule: RULE, state: "passed-over", strength: Some("weak"), event: None, steps_after: None, why: Some("the step was not marked done and its check did not pass before the window closed".into()), detail: Vec::new() }
}

/// The `loop-met` row of a loop continuation (`GUIDANCE_RULES`).
fn loop_rule(d: &Delivery, after: &[&Rec], closed: Option<&str>, g: Option<&crate::guidance::Guidance>) -> Ev {
    const RULE: &str = "loop-met";
    let steps_before = |r: &Rec| after.iter().take_while(|x| x.i != r.i).filter(|x| x.done()).count();
    let (iteration, max) = d.iteration.unwrap_or((0, 0));
    let ended = after.iter().find_map(|r| r.guidance.as_ref()?.loops.iter().find(|l| l.0 == d.item && matches!(l.1.as_str(), "done" | "exhausted" | "stalled" | "budget-spent")).map(|l| (*r, l.clone())));
    if let Some((r, l)) = ended {
        let (state, by) = match l.1.as_str() {
            "done" => ("verified", "its condition was met before the cap"),
            "stalled" => ("refuted", "it stalled: the same unmet reason past its stall threshold"),
            "budget-spent" => ("refuted", "its time or tool budget was used without its condition met"),
            _ => ("refuted", "it reached its cap without its condition met"),
        };
        return Ev::found(RULE, state, "direct", r, None, steps_before(r)).with("by", s(by)).with("iteration", n(iteration)).with("ended_at", n(l.2)).with("max", n(if l.3 > 0 { l.3 } else { max }));
    }
    let l = g.and_then(|g| g.loops.iter().find(|x| x.id == d.item));
    match l {
        Some(l) if l.state == "ended" && l.ended_ms.is_some_and(|t| t > d.pack.at) => Ev::not(RULE, "unknown", "the loop was ended by hand before its condition was judged again").with("iteration", n(iteration)),
        Some(l) if l.state == "active" || closed.is_none() => {
            if closed.is_some() {
                Ev::not(RULE, "unknown", "the session ended before the loop was judged again").with("iteration", n(iteration))
            } else {
                Ev::not(RULE, "none", "the loop has not ended so far").with("iteration", n(iteration)).with("max", n(max))
            }
        }
        Some(l) => Ev::not(RULE, "unknown", &format!("the loop is {} with no stop of this session recording it", l.state)),
        None => Ev::not(RULE, if closed.is_some() { "unknown" } else { "none" }, "the loop is no longer set"),
    }
}

/// The glob a reminder is tied to: its `edit` trigger's glob, else its file scope.
fn reminder_glob(r: &crate::guidance::Reminder) -> Option<String> {
    if r.trigger == "edit" && r.glob.is_some() { r.glob.clone() } else { r.files.clone() }
}

/// The `glob-check` row of a reminder delivery (`GUIDANCE_RULES`).
fn reminder_rule(d: &Delivery, after: &[&Rec], closed: Option<&str>, g: Option<&crate::guidance::Guidance>) -> Ev {
    const RULE: &str = "glob-check";
    let steps_before = |r: &Rec| after.iter().take_while(|x| x.i != r.i).filter(|x| x.done()).count();
    let Some(rem) = g.and_then(|g| g.reminders.iter().find(|x| x.id == d.item)) else {
        return Ev::not(RULE, "unknown", "the reminder is no longer set");
    };
    let Some(glob) = reminder_glob(rem) else {
        return Ev::not(RULE, "unknown", "the reminder names no edit glob; only reminders tied to an edit glob are judged");
    };
    let open = |why: &str| if closed.is_some() { Ev::not(RULE, "unknown", why) } else { Ev::not(RULE, "none", &format!("{why} so far")) };
    let under: Vec<&&Rec> = after
        .iter()
        .filter(|r| r.done() && matches!(r.kind.as_str(), "edit" | "write") && r.ok == Some(true) && r.paths.iter().any(|p| crate::guidance::glob_match(&glob, p)))
        .collect();
    let Some(last) = under.last() else {
        return open(&format!("no edit under {glob} followed")).with("glob", s(&glob));
    };
    let place = last.paths.iter().find(|p| crate::guidance::glob_match(&glob, p)).cloned();
    let check = after.iter().rev().take_while(|r| r.i != last.i).find(|r| r.done() && matches!(r.kind.as_str(), "build" | "test"));
    match check {
        Some(c) if c.ok == Some(true) => Ev::found(RULE, "verified", "indirect", c, place, steps_before(c)).with("glob", s(&glob)).with("edits", n(under.len() as u64)),
        Some(c) if c.ok == Some(false) => Ev::found(RULE, "refuted", "indirect", c, place, steps_before(c)).with("glob", s(&glob)).with("edits", n(under.len() as u64)),
        _ => open(&format!("no build or test after the last edit under {glob}")).with("glob", s(&glob)).with("edits", n(under.len() as u64)),
    }
}

/// The judgment of one delivery over what followed it in its session, as of `now`, with the
/// guidance as it is now (`g`; `None` when it cannot be read).
#[must_use]
pub fn judge_delivery(recs: &[Rec], d: &Delivery, g: Option<&crate::guidance::Guidance>, now: u64) -> (Option<&'static str>, Ev) {
    let (after, closed, until) = session_after(recs, d, now);
    let ev = match d.kind {
        "step" => step_rule(d, &after, (closed, until), g),
        "loop" => loop_rule(d, &after, closed, g),
        _ => reminder_rule(d, &after, closed, g),
    };
    (closed, ev)
}

/// Evaluates every guidance delivery of a project as of `now`, without writing.
#[must_use]
pub fn evaluate_guidance(memory: &Path, recs: &[Rec], now: u64) -> Vec<GuideState> {
    let cleared = cleared_at(memory);
    let g = crate::guidance::load(memory).ok();
    deliveries(recs)
        .into_iter()
        .filter(|d| d.pack.rec.t > cleared)
        .map(|d| {
            let (closed, ev) = judge_delivery(recs, &d, g.as_ref(), now);
            GuideState { d, closed, ev }
        })
        .collect()
}

fn guided_row(st: &GuideState, hw_id: &str) -> Vec<(&'static str, Value)> {
    let p = &st.d.pack;
    let mut f = vec![
        ("row", s("guided")),
        ("pack", s(&p.id)),
        ("session", s(&p.rec.session)),
        ("at", n(p.at)),
        ("kind", s(st.d.kind)),
        ("item", s(&st.d.item)),
        ("hook", s(&p.rec.event)),
        ("channel", s(p.rec.guidance.as_ref().map_or("", |g| g.channel.as_str()))),
        ("hardware", s(hw_id)),
    ];
    if let Some((i, m)) = st.d.iteration {
        f.push(("iteration", Value::Array(vec![n(i), n(m)])));
    }
    f
}

impl GuideState {
    /// The delivery and its judgment as JSON: the ledger id, kind, item, hook, session, time,
    /// window and the rule's finding.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let p = &self.d.pack;
        obj(vec![
            ("delivery", s(&p.id)),
            ("kind", s(self.d.kind)),
            ("item", s(&self.d.item)),
            ("event", s(&p.rec.event)),
            ("session", s(&p.rec.session)),
            ("at", n(p.at)),
            ("window", s(if self.closed.is_some() { "closed" } else { "open" })),
            ("closed", opt(self.closed)),
            ("verdict", self.ev.to_value()),
        ])
    }
}

/// Counts of guidance judgments by delivery kind, then by state.
#[must_use]
pub fn guidance_counts(states: &[GuideState]) -> Value {
    let mut c: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
    for st in states {
        *c.entry(st.d.kind.to_owned()).or_default().entry(st.ev.state.to_owned()).or_default() += 1;
    }
    Value::Object(c.into_iter().map(|(k, m)| (k, Value::Object(m.into_iter().map(|(a, b)| (a, n(b))).collect()))).collect())
}

/// The ledger and the packs' states, read only.
#[must_use]
pub fn view(home: &Path, memory: &Path, label: &str, recs: &[Rec], now: u64) -> View {
    let (rows, chain_fault, torn) = match read_ledger(&ledger_file(memory)) {
        Ok((r, _, t)) => (r, None, t),
        Err(e) => (Vec::new(), Some(e), None),
    };
    View {
        chain: (rows.len(), chain_fault),
        rows,
        torn,
        packs: evaluate(home, memory, label, recs, now),
        primes: evaluate_primes(home, memory, recs, now),
        guided: evaluate_guidance(memory, recs, now),
        written: 0,
    }
}

/// Brings the ledger up to date as of `now`: a `served` row for every pack not yet in it, and the
/// evidence rows of every pack whose window has closed and that has none yet. Nothing is written
/// with memory off. Primes get a `primed` row at once and their evidence rows when their window
/// closes, as packs do; guidance deliveries get a `guided` row at once and their evidence row when
/// their session's window closes. With the project's `root` known, verified and refuted outcomes
/// (`outcomes::all`) get one `outcome` row each, for the memory system to take in. `wait_ms` is
/// how long to wait for another writer's lock (0 from a hook: the next settle catches up).
/// # Errors
/// The ledger's chain is broken, or a write fails.
#[allow(clippy::too_many_arguments)]
pub fn settle(home: &Path, memory: &Path, label: &str, root: Option<&Path>, now: u64, wait_ms: u64, slow_hardware: bool) -> Result<View, String> {
    let recs = records(memory);
    if crate::controls::memory_off(home) {
        return Ok(view(home, memory, label, &recs, now));
    }
    let states = evaluate(home, memory, label, &recs, now);
    let primes = evaluate_primes(home, memory, &recs, now);
    let guided = evaluate_guidance(memory, &recs, now);
    let file = ledger_file(memory);
    let mut written = 0;
    if !(states.is_empty() && primes.is_empty() && guided.is_empty() && root.is_none())
        && let Some(_g) = lock(memory, wait_ms)?
    {
        let (rows, prev, torn) = read_ledger(&file)?;
        let rows_of = |kinds: &[&str], key: &str| -> BTreeSet<String> {
            // Rows written before the rename name a prime's pack `nudge:<id>` and its row `nudged`.
            rows.iter()
                .filter(|r| text(r, "row").is_some_and(|k| kinds.contains(&k.as_str())))
                .filter_map(|r| text(r, key).map(|p| p.strip_prefix("nudge:").map_or(p.clone(), |x| format!("prime:{x}"))))
                .collect()
        };
        let served = rows_of(&["served", "primed", "nudged", "guided"], "pack");
        let settled = rows_of(&["evidence"], "pack");
        let had = rows_of(&["outcome"], "outcome");
        let outcomes: Vec<nmmem::outcomes::Outcome> =
            root.map(|r| crate::outcomes::all(r, &recs, &states)).unwrap_or_default().into_iter().filter(|o| !had.contains(&o.id)).collect();
        let todo = states.iter().chain(&primes).any(|st| !served.contains(&st.pack.id) || (st.closed.is_some() && !settled.contains(&st.pack.id)))
            || guided.iter().any(|st| !served.contains(&st.d.pack.id) || (st.closed.is_some() && !settled.contains(&st.d.pack.id)))
            || !outcomes.is_empty();
        if todo {
            let idx = audit_index(home, label);
            let hw = hardware(home, slow_hardware);
            let hw_id = text(&hw, "id").unwrap_or_default();
            let mut new: Vec<Vec<(&'static str, Value)>> = Vec::new();
            for st in states.iter().chain(&primes) {
                if !served.contains(&st.pack.id) {
                    new.push(if st.pack.rec.prime.is_some() { primed_row(&st.pack, &hw_id) } else { served_row(&st.pack, &idx, &hw) });
                }
                if let Some(why) = st.closed
                    && !settled.contains(&st.pack.id)
                {
                    new.extend(st.evs.iter().map(|e| evidence_row(&st.pack, e, why, &hw_id)));
                }
            }
            for st in &guided {
                if !served.contains(&st.d.pack.id) {
                    new.push(guided_row(st, &hw_id));
                }
                if let Some(why) = st.closed
                    && !settled.contains(&st.d.pack.id)
                {
                    new.push(evidence_row(&st.d.pack, &st.ev, why, &hw_id));
                }
            }
            new.extend(outcomes.iter().map(|o| {
                let mut f = o.fields();
                f.push(("hardware", s(&hw_id)));
                f
            }));
            written = append(&file, &prev, torn.is_some(), new, rows.len())?;
        }
    }
    // The training store takes its examples from the packs whose windows are now closed.
    if root.is_some() {
        let _ = crate::examples::settle(home, memory, label, root, &recs, &states, now, wait_ms);
    }
    let mut v = view(home, memory, label, &recs, now);
    v.packs = states;
    v.primes = primes;
    v.guided = guided;
    v.written = written;
    Ok(v)
}

/// Called by the hook intake after it appended a record: at a stop, a session start or a session
/// end, the ledger is brought up to date. Cheap when the project's records hold no context pack,
/// no prime, no guidance delivery and no failed check.
/// # Errors
/// As `settle`.
pub fn after_hook(home: &Path, memory: &Path, root: &Path, event: &str) -> Result<(), String> {
    if !matches!(event, "Stop" | "SessionStart" | "SessionEnd") {
        return Ok(());
    }
    let worth = std::fs::read_to_string(crate::hook::situation_file(memory))
        .is_ok_and(|t| t.contains("\"tool\":\"nm_context_pack\"") || t.contains("\"prime\":") || t.contains("\"guidance\":") || t.contains("\"ok\":false"));
    if !worth {
        return Ok(());
    }
    settle(home, memory, &nmtype::project::label(home, root), Some(root), nmtype::now_ms(), 0, false).map(|_| ())
}

// ---------------------------------------------------------------------------------------------
// Views.

impl Ev {
    /// The finding as JSON (the shape of an evidence row's own fields).
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut f = vec![
            ("rule", s(self.rule)),
            ("state", s(self.state)),
            ("strength", opt(self.strength)),
            ("event", self.event.as_ref().map_or(Value::Null, |(r, at)| event_value(r, at.as_deref()))),
            ("steps_after", self.steps_after.map_or(Value::Null, |x| n(x as u64))),
        ];
        if let Some(w) = &self.why {
            f.push(("why", s(w)));
        }
        if !self.detail.is_empty() {
            f.push(("detail", obj(self.detail.clone())));
        }
        obj(f)
    }
    /// The finding as one short plain phrase.
    #[must_use]
    pub fn phrase(&self) -> String {
        let what = match (self.state, self.strength) {
            (st, Some(sg)) => format!("{st} {sg}"),
            (st, None) => st.to_owned(),
        };
        let at = self.event.as_ref().map(|(r, place)| {
            let thing = place.clone().or_else(|| r.cmd_class.clone()).unwrap_or_else(|| r.kind.clone());
            format!(" ({} {thing}, {} steps after)", if r.tool.is_empty() { r.event.as_str() } else { r.tool.as_str() }, self.steps_after.unwrap_or(0))
        });
        let why = self.why.as_ref().map(|w| format!(": {w}")).unwrap_or_default();
        format!("{} {what}{}{why}", self.rule, at.unwrap_or_default())
    }
}

impl View {
    /// Per served pack of `session`: its id, joining, window state and joins, keyed by the place of
    /// its hook record.
    #[must_use]
    pub fn pack_values(&self, session: &str) -> BTreeMap<usize, Value> {
        self.packs
            .iter()
            .chain(&self.primes)
            .filter(|p| p.pack.rec.session == session)
            .map(|p| {
                let written = self.rows.iter().any(|r| text(r, "row").as_deref() == Some("evidence") && text(r, "pack").as_deref() == Some(p.pack.id.as_str()));
                (
                    p.pack.rec.i,
                    obj(vec![
                        ("id", s(&p.pack.id)),
                        ("at", n(p.pack.at)),
                        ("joined_by", s(p.pack.joined_by)),
                        ("cited", Value::Array(p.pack.cited.iter().map(|(a, b)| s(&place_text(a, *b))).collect())),
                        ("window", s(if p.closed.is_some() { "closed" } else { "open" })),
                        ("closed", opt(p.closed)),
                        ("in_ledger", Value::Bool(written)),
                        ("joins", Value::Array(p.evs.iter().map(Ev::to_value).collect())),
                    ]),
                )
            })
            .collect()
    }
    /// Per served pack of `session`, one plain line.
    #[must_use]
    pub fn pack_lines(&self, session: &str) -> BTreeMap<usize, String> {
        self.packs
            .iter()
            .chain(&self.primes)
            .filter(|p| p.pack.rec.session == session)
            .map(|p| {
                let cited: Vec<String> = p.pack.cited.iter().map(|(a, b)| place_text(a, *b)).collect();
                let state = p.closed.map_or_else(|| "window open, so far".to_owned(), |w| format!("window closed ({w})"));
                let joins: Vec<String> = p.evs.iter().filter(|e| e.state != "none" && e.state != "unknown").map(Ev::phrase).collect();
                (
                    p.pack.rec.i,
                    format!(
                        "pack {} (joined by {}), cited {}; {state}: {}",
                        p.pack.id,
                        p.pack.joined_by,
                        if cited.is_empty() { "nothing known".into() } else { cited.join(" ") },
                        if joins.is_empty() { "no evidence yet".into() } else { joins.join("; ") }
                    ),
                )
            })
            .collect()
    }
}

/// Counts of evidence rows by rule, then by `state/strength` (or the state alone).
#[must_use]
pub fn counts(rows: &[Value]) -> BTreeMap<String, BTreeMap<String, u64>> {
    let mut c: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
    for r in rows.iter().filter(|r| text(r, "row").as_deref() == Some("evidence")) {
        let key = match text(r, "strength") {
            Some(sg) => format!("{}/{sg}", text(r, "state").unwrap_or_default()),
            None => text(r, "state").unwrap_or_default(),
        };
        *c.entry(text(r, "rule").unwrap_or_default()).or_default().entry(key).or_default() += 1;
    }
    c
}

/// The rows of one pack: its served row and its evidence rows.
#[must_use]
pub fn rows_of_pack<'a>(rows: &'a [Value], pack: &str) -> Vec<&'a Value> {
    rows.iter().filter(|r| text(r, "pack").as_deref() == Some(pack)).collect()
}

/// The rows of one file: served rows citing it, the evidence rows of those packs, and evidence
/// rows whose event touched it.
#[must_use]
pub fn rows_of_file<'a>(rows: &'a [Value], file: &str) -> Vec<&'a Value> {
    let file = file.replace('\\', "/");
    let cites = |r: &Value| r.get("cited").and_then(Value::array).unwrap_or_default().iter().any(|c| c.string().is_ok_and(|x| place(x).0 == file));
    let packs: BTreeSet<String> = rows.iter().filter(|r| cites(r)).filter_map(|r| text(r, "pack")).collect();
    let touched = |r: &Value| r.get("event").ok().and_then(|e| text(e, "place")).is_some_and(|p| place(&p).0 == file);
    rows.iter().filter(|r| text(r, "pack").is_some_and(|p| packs.contains(&p)) || touched(r)).collect()
}

/// Primes whose `followed` row has the state `state`.
fn followed_count(rows: &[Value], state: &str) -> u64 {
    rows.iter().filter(|r| text(r, "rule").as_deref() == Some("followed") && text(r, "state").as_deref() == Some(state)).count() as u64
}

/// The ledger view as JSON: chain state, counts, the selected rows, the packs, primes and guidance
/// deliveries whose window is still open with their joins so far (never written until the window
/// closes), and the guidance deliveries' judgments by kind (`guidance`, the newest first).
#[must_use]
pub fn view_value(v: &View, key: &str, file: &Path, pack: Option<&str>, path: Option<&str>, limit: usize) -> Value {
    let picked: Vec<&Value> = match (pack, path) {
        (Some(p), _) => rows_of_pack(&v.rows, p),
        (None, Some(f)) => rows_of_file(&v.rows, f),
        (None, None) => v.rows.iter().collect(),
    };
    let skip = picked.len().saturating_sub(limit);
    let mut states: BTreeMap<String, u64> = BTreeMap::new();
    for r in v.rows.iter().filter(|r| text(r, "row").as_deref() == Some("evidence")) {
        *states.entry(text(r, "state").unwrap_or_default()).or_default() += 1;
    }
    let counts_v = Value::Object(counts(&v.rows).into_iter().map(|(k, m)| (k, Value::Object(m.into_iter().map(|(a, b)| (a, n(b))).collect()))).collect());
    let pending: Vec<Value> = v
        .packs
        .iter()
        .chain(&v.primes)
        .filter(|p| p.closed.is_none())
        .filter(|p| pack.is_none_or(|x| x == p.pack.id) && path.is_none_or(|f| p.pack.cited.iter().any(|c| c.0 == f.replace('\\', "/"))))
        .map(|p| {
            obj(vec![
                ("pack", s(&p.pack.id)),
                ("session", s(&p.pack.rec.session)),
                ("at", n(p.pack.at)),
                ("joined_by", s(p.pack.joined_by)),
                ("joins", Value::Array(p.evs.iter().map(Ev::to_value).collect())),
            ])
        })
        .collect();
    let guided_pending: Vec<Value> = v
        .guided
        .iter()
        .filter(|g| g.closed.is_none() && pack.is_none_or(|x| x == g.d.pack.id) && path.is_none())
        .map(|g| {
            obj(vec![
                ("pack", s(&g.d.pack.id)),
                ("session", s(&g.d.pack.rec.session)),
                ("at", n(g.d.pack.at)),
                ("joined_by", s("guidance")),
                ("joins", Value::Array(vec![g.ev.to_value()])),
            ])
        })
        .collect();
    let mut pending = pending;
    pending.extend(guided_pending);
    let guidance = obj(vec![
        ("given", n(v.guided.len() as u64)),
        ("window_open", n(v.guided.iter().filter(|g| g.closed.is_none()).count() as u64)),
        ("rules", Value::Array(GUIDANCE_RULES.iter().map(|r| s(r)).collect())),
        ("by_kind", guidance_counts(&v.guided)),
        ("deliveries", Value::Array(v.guided.iter().rev().take(limit.min(200)).map(GuideState::to_value).collect())),
    ]);
    let primes = obj(vec![
        ("given", n(v.primes.len() as u64)),
        ("window_open", n(v.primes.iter().filter(|p| p.closed.is_none()).count() as u64)),
        ("followed", n(followed_count(&v.rows, "used"))),
        ("passed_over", n(followed_count(&v.rows, "passed-over"))),
    ]);
    obj(vec![
        ("project", s(key)),
        ("file", s(&file.display().to_string())),
        ("chain", obj(vec![("rows", n(v.chain.0 as u64)), ("intact", Value::Bool(v.chain.1.is_none())), ("fault", opt(v.chain.1.as_deref())), ("torn", opt(v.torn.as_deref()))])),
        ("packs", obj(vec![("served", n(v.packs.len() as u64)), ("window_open", n(v.packs.iter().filter(|p| p.closed.is_none()).count() as u64))])),
        // The same counts under their earlier name, for an interface that has not moved to `primes`.
        ("nudges", primes.clone()),
        ("primes", primes),
        ("guidance", guidance),
        ("outcomes", n(v.rows.iter().filter(|r| text(r, "row").as_deref() == Some("outcome")).count() as u64)),
        ("states", Value::Object(states.into_iter().map(|(a, b)| (a, n(b))).collect())),
        ("counts", counts_v),
        ("rows", Value::Array(picked.into_iter().skip(skip).cloned().collect())),
        ("pending", Value::Array(pending)),
        ("written", n(v.written as u64)),
    ])
}

/// The project named by `project=` in a query, and its memory folder: `nano` for the Nano's own
/// folder, else a key as `/api/projects` lists it. Anything else, or a folder that does not exist,
/// is refused.
#[must_use]
pub fn project_memory(home: &Path, query: &str) -> Option<(String, PathBuf)> {
    let key = crate::project::query_arg(query, "project")?;
    if key == "nano" {
        return Some((key, home.join("memory")));
    }
    let ok = !key.is_empty() && !key.starts_with('.') && key.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    let dir = nmtype::project::projects_dir(home).join(&key);
    (ok && dir.is_dir()).then_some((key, dir))
}

/// `/api/ledger?project=<key>[&pack=<id>][&file=<path>][&limit=<n>]`: read only (the ledger is
/// written by the hooks and the `ledger` command).
#[must_use]
pub fn api_view(home: &Path, query: &str) -> Value {
    let Some((key, memory)) = project_memory(home, query) else {
        return obj(vec![("error", s("project: give the key of a project as /api/projects lists it"))]);
    };
    let recs = records(&memory);
    let v = view(home, &memory, &key, &recs, nmtype::now_ms());
    let limit = crate::project::query_arg(query, "limit").and_then(|l| l.parse::<usize>().ok()).unwrap_or(500).min(10_000);
    let pack = crate::project::query_arg(query, "pack").filter(|x| !x.is_empty());
    let file = crate::project::query_arg(query, "file").filter(|x| !x.is_empty());
    view_value(&v, &key, &ledger_file(&memory), pack.as_deref(), file.as_deref(), limit)
}

/// Removes a project's ledger and marks the time, so packs served before it are not written again
/// from the records that remain. Logged as a `privacy` audit row. Returns the rows removed and the
/// audit id.
/// # Errors
/// Another writer holds the ledger, or a file cannot be removed or written.
pub fn clear(home: &Path, memory: &Path, label: &str, by: &str) -> Result<(usize, String), String> {
    let _g = lock(memory, 3000)?.ok_or("the ledger is being written; try again")?;
    let file = ledger_file(memory);
    let rows = std::fs::read_to_string(&file).map(|t| t.lines().filter(|l| !l.trim().is_empty()).count()).unwrap_or(0);
    for f in [file.clone(), file.with_extension("torn")] {
        if f.exists() {
            std::fs::remove_file(&f).map_err(|e| format!("{}: {e}", f.display()))?;
        }
    }
    if memory.is_dir() {
        std::fs::write(memory.join(CLEARED), format!("{}\nEvidence ledger cleared at this time (ms); packs served before it are not written again.\n", nmtype::now_ms()))
            .map_err(|e| e.to_string())?;
    }
    let id = crate::controls::audit(
        home,
        "privacy",
        vec![("label", s(&format!("Evidence ledger cleared: {rows} rows"))), ("project", s(label)), ("cleared", n(rows as u64)), ("by", s(by))],
    )?;
    Ok((rows, id))
}

fn row_line(r: &Value) -> String {
    let t = r.get("t").and_then(Value::integer).unwrap_or(0);
    let stamp = nmmusc::pack::utc_stamp(t).replace('T', " ").trim_end_matches('Z').to_owned();
    match text(r, "row").as_deref() {
        Some("served") => {
            let cited: Vec<String> = r.get("cited").and_then(Value::array).unwrap_or_default().iter().filter_map(|c| c.string().ok().map(str::to_owned)).collect();
            let path = r.get("settings").ok().and_then(|x| text(x, "path")).unwrap_or_else(|| "?".into());
            let gpu = r.get("hardware").ok().and_then(|h| text(h, "gpu")).unwrap_or_else(|| "no graphics card".into());
            format!(
                "{stamp}  served {} session {} ({} path, joined by {}) cited {}  [{gpu}]",
                text(r, "pack").unwrap_or_default(),
                text(r, "session").unwrap_or_default(),
                path,
                text(r, "joined_by").unwrap_or_default(),
                if cited.is_empty() { "nothing known".into() } else { cited.join(" ") }
            )
        }
        Some("primed" | "nudged") => {
            let places: Vec<String> = r.get("places").and_then(Value::array).unwrap_or_default().iter().filter_map(|c| c.string().ok().map(str::to_owned)).collect();
            format!(
                "{stamp}  primed {} session {} ({} at {}, {}) cited {}",
                text(r, "pack").unwrap_or_default(),
                text(r, "session").unwrap_or_default(),
                text(r, "kind").unwrap_or_default(),
                text(r, "hook").unwrap_or_default(),
                text(r, "mode").unwrap_or_default(),
                places.join(" ")
            )
        }
        Some("guided") => format!(
            "{stamp}  guided {} session {} ({} {} at {}, {})",
            text(r, "pack").unwrap_or_default(),
            text(r, "session").unwrap_or_default(),
            text(r, "kind").unwrap_or_default(),
            text(r, "item").unwrap_or_default(),
            text(r, "hook").unwrap_or_default(),
            text(r, "channel").unwrap_or_default(),
        ),
        Some("outcome") => {
            let places: Vec<String> = r
                .get("places")
                .and_then(Value::array)
                .unwrap_or_default()
                .iter()
                .filter_map(|p| {
                    let l = p.get("lines").ok()?.array().ok()?;
                    Some(place_text(&text(p, "path")?, Some((l.first()?.integer().ok()?, l.get(1)?.integer().ok()?))))
                })
                .collect();
            format!(
                "{stamp}  outcome {} {} {} by {} at {} (memory {})",
                text(r, "outcome").unwrap_or_default(),
                text(r, "state").unwrap_or_default(),
                text(r, "strength").unwrap_or_default(),
                text(r, "rule").unwrap_or_default(),
                places.join(" "),
                text(r, "memory").unwrap_or_default()
            )
        }
        _ => {
            let what = match text(r, "strength") {
                Some(sg) => format!("{} {sg}", text(r, "state").unwrap_or_default()),
                None => text(r, "state").unwrap_or_default(),
            };
            let ev = r.get("event").ok().filter(|e| !matches!(e, Value::Null)).map(|e| {
                let thing = text(e, "place").or_else(|| text(e, "check")).unwrap_or_else(|| text(e, "kind").unwrap_or_default());
                format!("  ({} {thing}, {} steps after)", text(e, "tool").filter(|x| !x.is_empty()).or_else(|| text(e, "event")).unwrap_or_default(), r.get("steps_after").and_then(Value::integer).unwrap_or(0))
            });
            let why = text(r, "why").map(|w| format!(": {w}")).unwrap_or_default();
            format!("    {:<12} {what}{}{why}", text(r, "rule").unwrap_or_default(), ev.unwrap_or_default())
        }
    }
}

/// `neuromorph-nano ledger --project <repo> [--pack <id>] [--file <path>] [--json] [--as-of <ms>]
/// [--clear]`: brings the ledger up to date (not with memory off) and prints it.
pub fn run(home: &Path, args: &[String]) -> ExitCode {
    let arg = |k: &str| args.windows(2).find(|w| w.first().map(String::as_str) == Some(k)).and_then(|w| w.get(1)).cloned();
    let Some(repo) = arg("--project") else {
        eprintln!("usage: neuromorph-nano ledger --project <repo> [--pack <id>] [--file <path>] [--json] [--as-of <ms>] [--clear]");
        return ExitCode::FAILURE;
    };
    let root = nmsubs::repo::canon(Path::new(&repo));
    let memory = nmtype::project::memory_dir(home, &root);
    let label = nmtype::project::label(home, &root);
    if args.iter().any(|a| a == "--clear") {
        return match clear(home, &memory, &label, "command line") {
            Ok((rows, id)) => {
                println!("cleared the evidence ledger of {label}: {rows} rows removed from {}; audit {id}", ledger_file(&memory).display());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("ledger: {e}");
                ExitCode::FAILURE
            }
        };
    }
    let now = match arg("--as-of") {
        Some(t) => match t.parse::<u64>() {
            Ok(t) => t,
            Err(_) => {
                eprintln!("ledger: --as-of takes a time in milliseconds since 1970");
                return ExitCode::FAILURE;
            }
        },
        None => nmtype::now_ms(),
    };
    // The command may take half a second to read the RAM size, which a hook does not; the cached
    // description then serves the rows the hooks write.
    if !crate::controls::memory_off(home) {
        let _ = hardware(home, true);
    }
    let v = match settle(home, &memory, &label, Some(&root), now, 3000, true) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("ledger: {e}");
            return ExitCode::FAILURE;
        }
    };
    let file = ledger_file(&memory);
    if args.iter().any(|a| a == "--json") {
        println!("{}", view_value(&v, &label, &file, arg("--pack").as_deref(), arg("--file").as_deref(), usize::MAX).encode().unwrap_or_default());
        return ExitCode::SUCCESS;
    }
    print_plain(&v, &file, arg("--pack").as_deref(), arg("--file").as_deref(), crate::controls::memory_off(home));
    ExitCode::SUCCESS
}

fn print_plain(v: &View, file: &Path, pack: Option<&str>, path: Option<&str>, memory_off: bool) {
    let chain = match &v.chain.1 {
        None => "chain intact".to_owned(),
        Some(f) => format!("CHAIN BROKEN: {f}"),
    };
    println!("evidence ledger: {} ({} rows, {chain}; {} written now{})", file.display(), v.chain.0, v.written, if memory_off { "; memory is off, nothing is written" } else { "" });
    if let Some(t) = &v.torn {
        println!("torn last line passed over: {t}");
    }
    let open = v.packs.iter().filter(|p| p.closed.is_none()).count();
    println!("packs: {} served, {} with the window closed, {open} with the window open", v.packs.len(), v.packs.len() - open);
    let nopen = v.primes.iter().filter(|p| p.closed.is_none()).count();
    let outcomes = v.rows.iter().filter(|r| text(r, "row").as_deref() == Some("outcome")).count();
    println!("primes: {} given, {} with the window closed, {nopen} with the window open; outcomes for memory: {outcomes}", v.primes.len(), v.primes.len() - nopen);
    let gopen = v.guided.iter().filter(|g| g.closed.is_none()).count();
    println!("guidance deliveries: {} given, {} with the window closed, {gopen} with the window open", v.guided.len(), v.guided.len() - gopen);
    println!("evidence by rule (state/strength: rows; packs, primes and guidance together):");
    let c = counts(&v.rows);
    for rule in RULES.iter().chain(PRIME_RULES.iter().filter(|r| !RULES.contains(r))).chain(GUIDANCE_RULES.iter()).copied() {
        if let Some(m) = c.get(rule) {
            println!("  {rule:<12} {}", m.iter().map(|(k, x)| format!("{k} {x}")).collect::<Vec<_>>().join(", "));
        }
    }
    let picked: Vec<&Value> = match (pack, path) {
        (Some(p), _) => rows_of_pack(&v.rows, p),
        (None, Some(f)) => rows_of_file(&v.rows, f),
        (None, None) => {
            let skip = v.rows.len().saturating_sub(80);
            v.rows.iter().skip(skip).collect()
        }
    };
    println!("rows ({} shown):", picked.len());
    for r in picked {
        println!("{}", row_line(r));
    }
    for p in v.packs.iter().chain(&v.primes).filter(|p| p.closed.is_none() && pack.is_none_or(|x| x == p.pack.id)) {
        println!("window open: {} session {} (so far, not written)", p.pack.id, p.pack.rec.session);
        for e in &p.evs {
            println!("    {}", e.phrase());
        }
    }
    for g in v.guided.iter().filter(|g| g.closed.is_none() && pack.is_none_or(|x| x == g.d.pack.id)) {
        println!("window open: {} session {} (so far, not written)", g.d.pack.id, g.d.pack.rec.session);
        println!("    {}", g.ev.phrase());
    }
}

/// Section check (processor only): scripted records and audit rows give the expected joins for a
/// pack whose cited lines are read and edited and whose test passes; a pack passed over; a pack
/// with no following events (unknown); markers of a corrected course; place parsing; the chain
/// finds a changed line.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    if place("src/a.rs:3-4") != ("src/a.rs".into(), Some((3, 4)))
        || place("src/a.rs:3-4;") != ("src/a.rs".into(), Some((3, 4)))
        || place("src/b.rs:9") != ("src/b.rs".into(), Some((9, 9)))
        || place("Cargo.toml") != ("Cargo.toml".into(), None)
    {
        return Err("ledger: place parsing".into());
    }
    if markers("No, that's the wrong file").0.is_none() || markers("thanks, works now").1.is_none() || markers("now add a test").0.is_some() {
        return Err("ledger: prompt markers".into());
    }
    let lines = [
        r#"{"t":10,"event":"UserPromptSubmit","kind":"prompt","session":"s1","prompt":{"sha":"p1","chars":5,"kept":true}}"#,
        r#"{"t":20,"event":"PostToolUse","kind":"nano","session":"s1","tool":"mcp__neuromorph-nano__nm_context_pack","use":"u1","nano":{"tool":"nm_context_pack","ask":"where is refund","episode":"e1"},"ok":true}"#,
        r#"{"t":21,"event":"PostToolUse","kind":"read","session":"s1","tool":"Read","paths":["a.rs"],"lines":[["a.rs",1,40]],"ok":true}"#,
        r#"{"t":22,"event":"PreToolUse","kind":"edit","session":"s1","tool":"Edit","use":"u2","paths":["a.rs"],"lines":[["a.rs",12,13]]}"#,
        r#"{"t":23,"event":"PostToolUse","kind":"edit","session":"s1","tool":"Edit","use":"u2","paths":["a.rs"],"lines":[["a.rs",12,14]],"ok":true}"#,
        r#"{"t":24,"event":"PostToolUse","kind":"test","session":"s1","tool":"Bash","cmd_class":"cargo-test","ok":true,"exit":0,"tests":[4,0]}"#,
        r#"{"t":30,"event":"UserPromptSubmit","kind":"prompt","session":"s1","prompt":{"sha":"p2","chars":6,"kept":true}}"#,
        r#"{"t":31,"event":"Stop","kind":"stop","session":"s1"}"#,
    ];
    let recs: Vec<Rec> = lines.iter().enumerate().filter_map(|(i, l)| nmjson::parse(l.as_bytes()).ok().map(|v| Rec::from_value(i, &v))).collect();
    let idx = AuditIndex {
        packs: vec![Served { id: "a1".into(), time_ms: 15, ms: 3, episode: Some("e1".into()), sources: vec!["a.rs:10-13".into(), "b.rs".into()], ..Served::default() }],
        settings: Vec::new(),
    };
    let ps = packs(&recs, &idx);
    let p = ps.first().ok_or("ledger: no pack found")?;
    if p.id != "a1" || p.joined_by != "episode" || p.at != 18 {
        return Err(format!("ledger: pack joined as {} by {} at {}", p.id, p.joined_by, p.at));
    }
    let prompts: BTreeMap<String, String> = [("p2".to_owned(), "thanks, that works".to_owned())].into_iter().collect();
    let w = window(&recs, p, 32);
    let got: Vec<String> = judge(&recs, p, &w, &prompts).iter().map(|e| format!("{}:{}:{}", e.rule, e.state, e.strength.unwrap_or("-"))).collect();
    let want = ["read:used:direct", "edit:used:direct", "elsewhere:none:-", "check:verified:indirect", "revert:none:-", "next-prompt:verified:weak", "asked-again:none:-"];
    if w.closed != Some("next turn ended") || got != want {
        return Err(format!("ledger: joins {got:?}, window {:?}", w.closed));
    }
    // The same pack with nothing after it: open now, unknown once an hour has passed.
    let alone: Vec<Rec> = recs.iter().take(2).cloned().collect();
    let pa = packs(&alone, &idx);
    let pa = pa.first().ok_or("ledger: no pack alone")?;
    if window(&alone, pa, 100).closed.is_some() {
        return Err("ledger: a window closed with no reason".into());
    }
    let wa = window(&alone, pa, 20 + IDLE_MS);
    let got: Vec<String> = judge(&alone, pa, &wa, &prompts).iter().map(|e| format!("{}:{}", e.rule, e.state)).collect();
    if got != ["read:none", "edit:none", "elsewhere:unknown", "check:unknown", "revert:unknown", "next-prompt:unknown", "asked-again:none"] {
        return Err(format!("ledger: joins of a pack alone {got:?}"));
    }
    // The chain: written rows read back; a changed line is found.
    let dir = std::env::temp_dir().join(format!("nano-ledger-check-{}-{}", std::process::id(), nmtype::now_ms()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let file = dir.join(FILE);
    let r = (|| -> Result<(), String> {
        append(&file, GENESIS, false, vec![vec![("row", s("served")), ("pack", s("a1"))], vec![("row", s("evidence")), ("pack", s("a1"))]], 0)?;
        let (rows, _, torn) = read_ledger(&file)?;
        if rows.len() != 2 || torn.is_some() {
            return Err(format!("ledger: read back {} rows", rows.len()));
        }
        let t = std::fs::read_to_string(&file).map_err(|e| e.to_string())?;
        std::fs::write(&file, t.replacen("\"a1\"", "\"a2\"", 1)).map_err(|e| e.to_string())?;
        if read_ledger(&file).is_ok() {
            return Err("ledger: a changed line was not found".into());
        }
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&dir);
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guidance::{Guidance, Loop, Reminder, Step};

    fn recs(lines: &[&str]) -> Vec<Rec> {
        lines.iter().enumerate().map(|(i, l)| Rec::from_value(i, &nmjson::parse(l.as_bytes()).expect("json"))).collect()
    }
    fn verdict(lines: &[&str], g: &Guidance, now: u64) -> Vec<String> {
        let r = recs(lines);
        deliveries(&r)
            .iter()
            .map(|d| {
                let (closed, ev) = judge_delivery(&r, d, Some(g), now);
                format!("{} {}:{}:{}:{}", d.item, ev.rule, ev.state, ev.strength.unwrap_or("-"), if closed.is_some() { "closed" } else { "open" })
            })
            .collect()
    }
    const END: &str = r#"{"t":90,"event":"SessionEnd","kind":"session","session":"s1"}"#;
    const NUDGE: &str = r#"{"t":10,"event":"UserPromptSubmit","kind":"prompt","session":"s1","guidance":{"reminders":[],"step":"a","auto_done":[],"loops":[],"chars":40,"channel":"context"}}"#;
    const EDIT: &str = r#"{"t":11,"event":"PostToolUse","kind":"edit","session":"s1","tool":"Edit","paths":["src/a.rs"],"ok":true}"#;
    const TEST_OK: &str = r#"{"t":12,"event":"PostToolUse","kind":"test","session":"s1","tool":"Bash","cmd_class":"cargo-test","ok":true}"#;
    const TEST_FAIL: &str = r#"{"t":12,"event":"PostToolUse","kind":"test","session":"s1","tool":"Bash","cmd_class":"cargo-test","ok":false}"#;

    fn step(check: Option<&str>) -> Guidance {
        Guidance { steps: vec![Step { id: "a".into(), title: "A".into(), check: check.map(str::to_owned), paths: vec!["src/**".into()], ..Step::default() }], ..Guidance::default() }
    }

    #[test]
    fn deliveries_are_read_from_the_hook_records() {
        let r = recs(&[r#"{"t":5,"event":"Stop","kind":"stop","session":"s1","guidance":{"reminders":["r1","r2"],"step":"a","auto_done":["b"],"loops":[{"id":"l1","action":"block","iteration":1,"max":3},{"id":"l2","action":"done","iteration":2,"max":3}],"channel":"block"}}"#]);
        let d = deliveries(&r);
        let got: Vec<(&str, &str)> = d.iter().map(|x| (x.kind, x.item.as_str())).collect();
        assert_eq!(got, [("reminder", "r1"), ("reminder", "r2"), ("step", "a"), ("loop", "l1")], "a final note and a step the Nano marked done are not deliveries to judge");
        assert_eq!(d.get(3).and_then(|x| x.iteration), Some((1, 3)));
        assert!(d.iter().all(|x| x.pack.id.starts_with(&format!("guide:{}:{}:s1:5", x.kind, x.item))));
        assert!(deliveries(&recs(&[EDIT])).is_empty());
    }

    #[test]
    fn a_step_nudge_is_followed_when_the_step_is_marked_done_or_its_check_passes() {
        let auto = r#"{"t":12,"event":"PostToolUse","kind":"test","session":"s1","ok":true,"guidance":{"reminders":[],"auto_done":["a"],"loops":[],"channel":"context"}}"#;
        assert_eq!(verdict(&[NUDGE, EDIT, auto, END], &step(Some("test")), 100), ["a step-followed:used:direct:closed"]);
        assert_eq!(verdict(&[NUDGE, EDIT, TEST_OK, END], &step(Some("test")), 100), ["a step-followed:used:indirect:closed"], "the check passed after an edit under the step's paths");
        assert_eq!(verdict(&[NUDGE, TEST_OK, END], &step(Some("test")), 100), ["a step-followed:used:weak:closed"], "the check passed with no edit under the paths");
        let by_agent = Guidance { steps: vec![Step { id: "a".into(), done: true, done_by: Some("agent".into()), done_ms: Some(20), ..Step::default() }], ..Guidance::default() };
        assert_eq!(verdict(&[NUDGE, EDIT, END], &by_agent, 100), ["a step-followed:used:direct:closed"]);
        let before = Guidance { steps: vec![Step { id: "a".into(), done: true, done_ms: Some(5), ..Step::default() }], ..Guidance::default() };
        assert_eq!(verdict(&[NUDGE, EDIT, END], &before, 100), ["a step-followed:passed-over:weak:closed"], "done before the nudge is not following it");
        assert_eq!(verdict(&[NUDGE, EDIT, TEST_FAIL, END], &step(Some("test")), 100), ["a step-followed:passed-over:indirect:closed"]);
        assert_eq!(verdict(&[NUDGE, EDIT], &step(Some("test")), 20), ["a step-followed:none:-:open"], "not judged while the session goes on");
        assert_eq!(verdict(&[NUDGE, EDIT], &step(None), 11 + IDLE_MS), ["a step-followed:passed-over:weak:closed"], "an hour with no activity closes the window");
        assert_eq!(verdict(&[NUDGE, END], &Guidance::default(), 100), ["a step-followed:unknown:-:closed"], "a step no longer in the graph");
    }

    #[test]
    fn a_loop_continuation_is_judged_by_its_condition_before_the_cap() {
        let block = |t: u64, k: u64| format!(r#"{{"t":{t},"event":"Stop","kind":"stop","session":"s1","guidance":{{"loops":[{{"id":"l1","action":"block","iteration":{k},"max":3}}],"channel":"block"}}}}"#);
        let end = |t: u64, action: &str| format!(r#"{{"t":{t},"event":"Stop","kind":"stop","session":"s1","guidance":{{"loops":[{{"id":"l1","action":"{action}","iteration":2,"max":3}}],"channel":"context"}}}}"#);
        let (b1, b2) = (block(10, 1), block(20, 2));
        let active = Guidance { loops: vec![Loop { id: "l1".into(), state: "active".into(), max: 3, ..Loop::default() }], ..Guidance::default() };
        let done = end(30, "done");
        assert_eq!(verdict(&[b1.as_str(), EDIT, b2.as_str(), done.as_str(), END], &active, 100), ["l1 loop-met:verified:direct:closed", "l1 loop-met:verified:direct:closed"]);
        let out = end(30, "exhausted");
        assert_eq!(verdict(&[b1.as_str(), b2.as_str(), out.as_str(), END], &active, 100), ["l1 loop-met:refuted:direct:closed", "l1 loop-met:refuted:direct:closed"]);
        for early in ["stalled", "budget-spent"] {
            let e = end(30, early);
            assert_eq!(verdict(&[b1.as_str(), e.as_str(), END], &active, 100), ["l1 loop-met:refuted:direct:closed"], "a loop that ended {early}");
        }
        assert_eq!(verdict(&[b1.as_str()], &active, 11), ["l1 loop-met:none:-:open"]);
        assert_eq!(verdict(&[b1.as_str(), END], &active, 100), ["l1 loop-met:unknown:-:closed"], "the session ended before the loop was judged again");
        let ended = Guidance { loops: vec![Loop { id: "l1".into(), state: "ended".into(), ended_ms: Some(15), ..Loop::default() }], ..Guidance::default() };
        assert_eq!(verdict(&[b1.as_str(), END], &ended, 100), ["l1 loop-met:unknown:-:closed"]);
        let r = recs(&[b1.as_str(), b2.as_str(), done.as_str()]);
        let d = deliveries(&r);
        let (_, ev) = judge_delivery(&r, d.first().expect("delivery"), Some(&active), 40);
        assert_eq!(ev.to_value().get("detail").and_then(|x| x.get("ended_at")).and_then(Value::integer).ok(), Some(2), "the iteration the condition was met at");
    }

    #[test]
    fn a_reminder_tied_to_an_edit_glob_is_judged_by_the_check_after_its_edits() {
        let pre = r#"{"t":10,"event":"PreToolUse","kind":"edit","session":"s1","tool":"Edit","use":"u1","paths":["src/a.rs"],"guidance":{"reminders":["r1"],"loops":[],"channel":"context"}}"#;
        let edit = r#"{"t":11,"event":"PostToolUse","kind":"edit","session":"s1","tool":"Edit","use":"u1","paths":["src/a.rs"],"ok":true}"#;
        let other = r#"{"t":11,"event":"PostToolUse","kind":"edit","session":"s1","tool":"Edit","paths":["docs/a.md"],"ok":true}"#;
        let tied = |glob: Option<&str>, files: Option<&str>| Guidance {
            reminders: vec![Reminder { id: "r1".into(), trigger: "edit".into(), glob: glob.map(str::to_owned), files: files.map(str::to_owned), enabled: true, ..Reminder::default() }],
            ..Guidance::default()
        };
        let g = tied(Some("src/**"), None);
        assert_eq!(verdict(&[pre, edit, TEST_OK, END], &g, 100), ["r1 glob-check:verified:indirect:closed"]);
        assert_eq!(verdict(&[pre, edit, TEST_FAIL, END], &g, 100), ["r1 glob-check:refuted:indirect:closed"]);
        assert_eq!(verdict(&[pre, edit, TEST_FAIL, TEST_OK.replace("\"t\":12", "\"t\":13").as_str(), END], &g, 100), ["r1 glob-check:verified:indirect:closed"], "the last check after the last edit counts");
        assert_eq!(verdict(&[pre, TEST_OK, edit.replace("\"t\":11", "\"t\":14").as_str(), END], &g, 100), ["r1 glob-check:unknown:-:closed"], "a check before the edit does not count");
        assert_eq!(verdict(&[pre, other, TEST_OK, END], &g, 100), ["r1 glob-check:unknown:-:closed"], "no edit under the glob");
        assert_eq!(verdict(&[pre, edit], &g, 20), ["r1 glob-check:none:-:open"]);
        assert_eq!(verdict(&[pre, edit, TEST_OK, END], &tied(None, Some("src/**")), 100), ["r1 glob-check:verified:indirect:closed"], "the file scope is the glob when the trigger has none");
        assert_eq!(verdict(&[pre, edit, TEST_OK, END], &tied(None, None), 100), ["r1 glob-check:unknown:-:closed"], "a reminder with no glob is not judged");
        assert_eq!(verdict(&[pre, edit, TEST_OK, END], &Guidance::default(), 100), ["r1 glob-check:unknown:-:closed"], "a reminder no longer set");
    }

    #[test]
    fn the_view_counts_guidance_deliveries_and_lists_the_open_ones() {
        let r = recs(&[NUDGE, EDIT]);
        let g = step(Some("test"));
        let guided: Vec<GuideState> = deliveries(&r)
            .into_iter()
            .map(|d| {
                let (closed, ev) = judge_delivery(&r, &d, Some(&g), 20);
                GuideState { d, closed, ev }
            })
            .collect();
        let v = View { rows: Vec::new(), chain: (0, None), torn: None, packs: Vec::new(), primes: Vec::new(), guided, written: 0 };
        let out = view_value(&v, "k", Path::new("evidence.jsonl"), None, None, 50);
        let gv = out.get("guidance").expect("guidance");
        assert_eq!(gv.get("given").and_then(Value::integer).ok(), Some(1));
        assert_eq!(gv.get("window_open").and_then(Value::integer).ok(), Some(1));
        assert_eq!(gv.get("by_kind").and_then(|x| x.get("step")).and_then(|x| x.get("none")).and_then(Value::integer).ok(), Some(1));
        let d = gv.get("deliveries").and_then(Value::array).expect("deliveries");
        assert_eq!(d.first().and_then(|x| text(x, "kind")).as_deref(), Some("step"));
        assert_eq!(d.first().and_then(|x| x.get("verdict").ok()).and_then(|x| text(x, "rule")).as_deref(), Some("step-followed"));
        let pending = out.get("pending").and_then(Value::array).expect("pending");
        assert!(pending.iter().any(|p| text(p, "pack").is_some_and(|x| x.starts_with("guide:step:a:"))));
    }
}
