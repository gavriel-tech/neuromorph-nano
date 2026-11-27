//! Agent guidance: reminders, loops and a step graph that the user or the coding agent sets for a
//! project, delivered to the agent through the hooks (`hook.rs`) on the channel the primes use.
//!
//! - A **reminder** is a note with a trigger (session start, every prompt or every Nth prompt, after
//!   a compaction, at a stop, before an edit whose path matches a glob, after a failed build or
//!   test, after every Nth tool call, or when the same recorded error, by its code or a piece of
//!   its message, comes back in N of the agent's tool results, 2 to 5: `error-repeats`), a
//!   repetition (once, every time, at most N times per
//!   session), an optional file scope (a glob) and a switch. It is delivered whole, prefixed
//!   `Neuromorph Nano reminder (set by the user|set by the agent): `, and may ride with a prime in the
//!   same `additionalContext`. Reminders never count against the prime's never-twice rule and are
//!   never cut.
//! - A **loop** keeps the agent going at Stop until its condition holds: the last build or test
//!   after the last edit passed (`check-passes`), an objective is closed (`goal-done`), every step
//!   is done (`steps-done`), or it is ended by hand (`manual`). While the condition is unmet and the
//!   loop is under its maximum (default 3, hard cap 20), the Stop hook replies
//!   `{"decision":"block","reason":...}`, Claude Code's documented way to make the agent continue.
//!   A condition may combine 2 to 4 of these (`all-of`, `any-of`, with `parts`); its unmet reason
//!   names the parts not met. Each loop also holds a stall threshold (`stall`, default 2, 1 to
//!   10): when the same unmet reason comes back that many times in a row the loop ends `stalled`
//!   (a manual loop has no progress reason and does not stall); and optional budgets, wall time
//!   (`budget_ms`) and tool calls (`budget_tools`) since it was set, which end it `budget-spent`.
//!   At the maximum, or when the condition holds, the loop is marked exhausted or done; every end
//!   is said once in a final non-blocking note naming the reason, and appends one outcome row to
//!   `guidance-outcomes.jsonl` (the condition, iterations, budget used, the end, the distinct
//!   unmet reasons in order, the ledger's verdict, start and end), kept as evidence for Learning
//!   (which does not read it yet in v0.1.0). This is the one deliberate exception to "hooks never
//!   block": it
//!   continues the agent rather than stopping it, it acts only when the user or the agent set a
//!   loop and left it on, it is capped, and the `loops` setting switches every loop off.
//!   Antigravity has no documented stop continuation the Nano relies on, so a loop does not act
//!   there; each such stop is recorded and the view says so.
//! - A **step graph** holds steps (id, title, dependencies, done, an optional check that must pass
//!   after edits, optional path globs), standalone or attached to an objective. Dependencies are an
//!   AND-join (`depends_on`, every one done) and an optional OR-join (`depends_any`, at least one
//!   done). Ready and blocked are computed from both; cycles over both are refused on write. A
//!   probe step (`probe: true`, which needs a check) is marked done by its check passing while its
//!   dependencies are met, with no edit needed. With reminders on, the next
//!   ready step is given at prompts and at stops, once per step per session unless it changes. A
//!   step with a check is marked done by the Nano when that check passes after an edit under its
//!   globs, with the evidence kept on the step.
//!
//! Storage: `<project memory>/guidance.json`, written whole and atomically (a temporary file
//! renamed over it), versioned by a counter; a write that finds the version moved under it is
//! applied again on the newer file. Every change (by the user, the agent or the Nano) is an audit
//! row (`kind: "guidance"`, ids and no text) with its before and after kept in
//! `guidance-history.jsonl`, so the Audit page undoes it. Per-session counters live in
//! `guidance/<session>.json`; what was sent to the agent is appended to
//! `guidance-deliveries.jsonl` (the text included, so the interface can show it) and named on the
//! hook's situation record (`guidance`), where the evidence ledger reads it and judges each
//! delivery by its kind's rule (`ledger::GUIDANCE_RULES`): a step nudge followed, a loop's
//! condition met before its cap, a reminder's edits under its glob keeping a passing check. The
//! view gives each delivery its verdict.
use crate::sessions::Rec;
use nmjson::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Mutex;

/// The guidance file in a project memory folder.
pub const FILE: &str = "guidance.json";
/// Before and after of every change, by audit id, for undo.
const HISTORY: &str = "guidance-history.jsonl";
/// What was sent to the agent, newest last.
const DELIVERIES: &str = "guidance-deliveries.jsonl";
/// One row per loop that ended, for any reason, newest last (`guidance-outcomes.jsonl`).
pub const OUTCOMES: &str = "guidance-outcomes.jsonl";
/// Outcome rows the view gives when the query names no number, and the most it gives.
const OUTCOMES_SHOWN: usize = 20;
const OUTCOMES_MAX: usize = 200;
/// The folder of per-session counters.
const MEMO_DIR: &str = "guidance";
/// A loop's maximum iterations when none is given.
pub const LOOP_MAX_DEFAULT: u64 = 3;
/// The most iterations any loop may have.
pub const LOOP_CAP: u64 = 20;
/// A loop's stall threshold when none is given, and its bounds: the loop ends `stalled` when the
/// same unmet reason comes back this many times in a row.
pub const STALL_DEFAULT: u64 = 2;
pub const STALL_MAX: u64 = 10;
/// Bounds of a loop's wall-time budget (milliseconds since it was set or restarted).
pub const BUDGET_MS_MIN: u64 = 1000;
pub const BUDGET_MS_MAX: u64 = 86_400_000;
/// Most tool calls a loop's tool budget may name.
pub const BUDGET_TOOLS_MAX: u64 = 10_000;
/// The parts of a combined condition (`all-of`, `any-of`).
const PARTS_MIN: usize = 2;
const PARTS_MAX: usize = 4;
/// The repeats an `error-repeats` reminder may wait for.
pub const ERROR_COUNT_MIN: u64 = 2;
pub const ERROR_COUNT_MAX: u64 = 5;
/// Longest error signature of an `error-repeats` reminder.
const MAX_SIGNATURE: usize = 80;
/// Distinct unmet reasons a loop keeps, and the characters kept of each.
const UNMET_KEEP: usize = 12;
const UNMET_CHARS: usize = 300;
/// Longest reminder text, loop prompt or step title.
const MAX_TEXT: usize = 2000;
/// Longest glob.
const MAX_GLOB: usize = 200;
/// Most items of one kind in a project.
const MAX_ITEMS: usize = 200;
/// Most files remembered per session for a file scope.
const MAX_TOUCHED: usize = 200;
/// Deliveries the view shows.
const SHOWN: usize = 40;
/// Past this size the deliveries file keeps its newest `KEEP` lines.
const TRIM_BYTES: u64 = 512 * 1024;
const KEEP: usize = 1000;

pub const TRIGGERS: &[&str] = &["session-start", "prompt", "compact", "stop", "edit", "failure", "tools", "error-repeats"];
pub const REPEATS: &[&str] = &["once", "always", "max"];
pub const CONDITIONS: &[&str] = &["check-passes", "goal-done", "steps-done", "manual", "all-of", "any-of"];
/// The conditions a combined condition (`all-of`, `any-of`) is made of.
pub const PART_CONDITIONS: &[&str] = &["check-passes", "goal-done", "steps-done", "manual"];
/// The states a loop ends in, besides `active`.
pub const LOOP_ENDS: &[&str] = &["done", "exhausted", "stalled", "budget-spent", "ended"];
pub const CHECKS: &[&str] = &["build", "test"];
pub const KINDS: &[&str] = &["reminder", "loop", "step"];
pub const ACTIONS: &[&str] = &["add", "edit", "remove", "enable", "disable", "done", "reset"];

/// What a Claude Code session reads when a host cannot take a loop's continuation.
const AGY_LOOP: &str = "not acted on Antigravity: it has no documented stop continuation the Nano relies on; loops act on Claude Code only";

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}
fn obj(f: Vec<(&str, Value)>) -> Value {
    Value::Object(f.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn text(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::string).ok().map(str::to_owned)
}
fn int(v: &Value, k: &str) -> Option<u64> {
    v.get(k).and_then(Value::integer).ok()
}
fn flag(v: &Value, k: &str) -> Option<bool> {
    match v.get(k) {
        Ok(Value::Bool(b)) => Some(*b),
        _ => None,
    }
}
fn opt_s(x: Option<&String>) -> Value {
    x.map_or(Value::Null, |t| s(t))
}
fn opt_n(x: Option<u64>) -> Value {
    x.map_or(Value::Null, n)
}
/// A list given as an array of strings or as one comma-separated string.
fn list(v: &Value, k: &str) -> Option<Vec<String>> {
    match v.get(k) {
        Ok(Value::Array(a)) => Some(a.iter().filter_map(|x| x.string().ok()).map(|x| x.trim().to_owned()).filter(|x| !x.is_empty()).collect()),
        Ok(Value::String(t)) => Some(t.split(',').map(str::trim).filter(|x| !x.is_empty()).map(str::to_owned).collect()),
        _ => None,
    }
}
fn strings(a: &[String]) -> Value {
    Value::Array(a.iter().map(|x| s(x)).collect())
}

/// Who set an item, from the route's caller: the agent over MCP, the user otherwise.
#[must_use]
pub fn who_of(by: &str) -> &'static str {
    if by == "mcp" { "agent" } else { "user" }
}
fn who_label(who: &str) -> &'static str {
    match who {
        "agent" => "set by the agent",
        "nano" => "set by the Nano",
        _ => "set by the user",
    }
}

// ---------------------------------------------------------------------------------------------
// The items.

/// A reminder: a note delivered to the agent at its trigger.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Reminder {
    pub id: String,
    pub text: String,
    pub by: String,
    pub via: String,
    pub trigger: String,
    /// For `prompt` and `tools`: every Nth one (1 is every one).
    pub every: u64,
    /// For `edit`: the paths it fires on (every edit when absent).
    pub glob: Option<String>,
    pub repeat: String,
    /// For `repeat: max`: at most this many deliveries per session.
    pub max: u64,
    /// The file scope: only when the event's paths, or a file the session touched, match.
    pub files: Option<String>,
    /// For `error-repeats`: the error code (`E0308`) or a piece of its message the hook recorded.
    pub error: Option<String>,
    /// For `error-repeats`: how many tool results must carry that error (2 to 5).
    pub count: u64,
    pub enabled: bool,
    pub created_ms: u64,
    pub updated_ms: u64,
    /// Deliveries before this time do not count toward the repetition (a reset).
    pub reset_ms: u64,
}

/// A loop: at Stop, keeps the agent going until its condition holds.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Loop {
    pub id: String,
    pub prompt: String,
    pub by: String,
    pub via: String,
    pub until: String,
    /// For `all-of` and `any-of`: the conditions combined (2 to 4 of `PART_CONDITIONS`).
    pub parts: Vec<String>,
    pub goal: Option<u64>,
    pub max: u64,
    pub iterations: u64,
    /// The same unmet reason this many times in a row ends the loop `stalled` (1 to 10).
    pub stall: u64,
    /// Wall time since the loop was set (or restarted) after which it ends `budget-spent`.
    pub budget_ms: Option<u64>,
    /// Tool calls since the loop was set (or restarted) after which it ends `budget-spent`.
    pub budget_tools: Option<u64>,
    /// How many stops in a row gave the same unmet reason as the one before.
    pub repeats: u64,
    /// The distinct unmet reasons in the order they came (the newest `UNMET_KEEP`).
    pub unmet_seq: Vec<String>,
    /// `active`, `done` (condition met), `exhausted` (at its maximum), `stalled` (the same unmet
    /// reason past its stall threshold), `budget-spent` (its time or tool budget used) or `ended`
    /// (by hand).
    pub state: String,
    pub enabled: bool,
    pub created_ms: u64,
    pub updated_ms: u64,
    /// Records before this time do not count for `check-passes` (set at creation and reset).
    pub since_ms: u64,
    pub ended_ms: Option<u64>,
    pub end_note: Option<String>,
    pub last_unmet: Option<String>,
}

/// A step of the step graph.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Step {
    pub id: String,
    pub title: String,
    pub by: String,
    pub via: String,
    pub goal: Option<u64>,
    pub depends_on: Vec<String>,
    /// An OR-join: when not empty, at least one of these steps must be done as well.
    pub depends_any: Vec<String>,
    /// A probe: its check passing while it is ready marks it done, with no edit needed.
    pub probe: bool,
    pub done: bool,
    pub done_by: Option<String>,
    pub done_ms: Option<u64>,
    /// Why it is done: who marked it, or the passing check and the edits it followed.
    pub evidence: Option<Value>,
    /// `build` or `test`: the check that must pass after edits (under `paths`) for the Nano to mark
    /// it done.
    pub check: Option<String>,
    pub paths: Vec<String>,
    /// Bumped when the step changes, so it is nudged again.
    pub rev: u64,
    pub created_ms: u64,
    pub updated_ms: u64,
}

/// The guidance of one project.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Guidance {
    pub version: u64,
    pub next: u64,
    pub reminders: Vec<Reminder>,
    pub loops: Vec<Loop>,
    pub steps: Vec<Step>,
}

impl Reminder {
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("id", s(&self.id)),
            ("text", s(&self.text)),
            ("by", s(&self.by)),
            ("via", s(&self.via)),
            ("trigger", s(&self.trigger)),
            ("every", n(self.every)),
            ("glob", opt_s(self.glob.as_ref())),
            ("repeat", s(&self.repeat)),
            ("max", n(self.max)),
            ("files", opt_s(self.files.as_ref())),
            ("error", opt_s(self.error.as_ref())),
            ("count", n(self.count)),
            ("enabled", Value::Bool(self.enabled)),
            ("created_ms", n(self.created_ms)),
            ("updated_ms", n(self.updated_ms)),
            ("reset_ms", n(self.reset_ms)),
        ])
    }
    fn from_value(v: &Value) -> Option<Self> {
        Some(Self {
            id: text(v, "id")?,
            text: text(v, "text").unwrap_or_default(),
            by: text(v, "by").unwrap_or_else(|| "user".into()),
            via: text(v, "via").unwrap_or_default(),
            trigger: text(v, "trigger").unwrap_or_else(|| "prompt".into()),
            every: int(v, "every").unwrap_or(1).max(1),
            glob: text(v, "glob"),
            repeat: text(v, "repeat").unwrap_or_else(|| "always".into()),
            max: int(v, "max").unwrap_or(1),
            files: text(v, "files"),
            error: text(v, "error"),
            count: int(v, "count").unwrap_or(ERROR_COUNT_MIN).clamp(ERROR_COUNT_MIN, ERROR_COUNT_MAX),
            enabled: flag(v, "enabled").unwrap_or(true),
            created_ms: int(v, "created_ms").unwrap_or(0),
            updated_ms: int(v, "updated_ms").unwrap_or(0),
            reset_ms: int(v, "reset_ms").unwrap_or(0),
        })
    }
}

impl Loop {
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("id", s(&self.id)),
            ("prompt", s(&self.prompt)),
            ("by", s(&self.by)),
            ("via", s(&self.via)),
            ("until", s(&self.until)),
            ("parts", strings(&self.parts)),
            ("goal", opt_n(self.goal)),
            ("max", n(self.max)),
            ("iterations", n(self.iterations)),
            ("stall", n(self.stall)),
            ("budget_ms", opt_n(self.budget_ms)),
            ("budget_tools", opt_n(self.budget_tools)),
            ("repeats", n(self.repeats)),
            ("unmet_seq", strings(&self.unmet_seq)),
            ("state", s(&self.state)),
            ("enabled", Value::Bool(self.enabled)),
            ("created_ms", n(self.created_ms)),
            ("updated_ms", n(self.updated_ms)),
            ("since_ms", n(self.since_ms)),
            ("ended_ms", opt_n(self.ended_ms)),
            ("end_note", opt_s(self.end_note.as_ref())),
            ("last_unmet", opt_s(self.last_unmet.as_ref())),
        ])
    }
    fn from_value(v: &Value) -> Option<Self> {
        Some(Self {
            id: text(v, "id")?,
            prompt: text(v, "prompt").unwrap_or_default(),
            by: text(v, "by").unwrap_or_else(|| "user".into()),
            via: text(v, "via").unwrap_or_default(),
            until: text(v, "until").unwrap_or_else(|| "manual".into()),
            parts: list(v, "parts").unwrap_or_default(),
            goal: int(v, "goal"),
            max: int(v, "max").unwrap_or(LOOP_MAX_DEFAULT).clamp(1, LOOP_CAP),
            iterations: int(v, "iterations").unwrap_or(0),
            stall: int(v, "stall").unwrap_or(STALL_DEFAULT).clamp(1, STALL_MAX),
            budget_ms: int(v, "budget_ms"),
            budget_tools: int(v, "budget_tools"),
            repeats: int(v, "repeats").unwrap_or(0),
            unmet_seq: list(v, "unmet_seq").unwrap_or_default(),
            state: text(v, "state").unwrap_or_else(|| "active".into()),
            enabled: flag(v, "enabled").unwrap_or(true),
            created_ms: int(v, "created_ms").unwrap_or(0),
            updated_ms: int(v, "updated_ms").unwrap_or(0),
            since_ms: int(v, "since_ms").unwrap_or(0),
            ended_ms: int(v, "ended_ms"),
            end_note: text(v, "end_note"),
            last_unmet: text(v, "last_unmet"),
        })
    }
}

impl Step {
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("id", s(&self.id)),
            ("title", s(&self.title)),
            ("by", s(&self.by)),
            ("via", s(&self.via)),
            ("goal", opt_n(self.goal)),
            ("depends_on", strings(&self.depends_on)),
            ("depends_any", strings(&self.depends_any)),
            ("probe", Value::Bool(self.probe)),
            ("done", Value::Bool(self.done)),
            ("done_by", opt_s(self.done_by.as_ref())),
            ("done_ms", opt_n(self.done_ms)),
            ("evidence", self.evidence.clone().unwrap_or(Value::Null)),
            ("check", opt_s(self.check.as_ref())),
            ("paths", strings(&self.paths)),
            ("rev", n(self.rev)),
            ("created_ms", n(self.created_ms)),
            ("updated_ms", n(self.updated_ms)),
        ])
    }
    fn from_value(v: &Value) -> Option<Self> {
        Some(Self {
            id: text(v, "id")?,
            title: text(v, "title").unwrap_or_default(),
            by: text(v, "by").unwrap_or_else(|| "user".into()),
            via: text(v, "via").unwrap_or_default(),
            goal: int(v, "goal"),
            depends_on: list(v, "depends_on").unwrap_or_default(),
            depends_any: list(v, "depends_any").unwrap_or_default(),
            probe: flag(v, "probe").unwrap_or(false),
            done: flag(v, "done").unwrap_or(false),
            done_by: text(v, "done_by"),
            done_ms: int(v, "done_ms"),
            evidence: v.get("evidence").ok().filter(|e| !matches!(e, Value::Null)).cloned(),
            check: text(v, "check"),
            paths: list(v, "paths").unwrap_or_default(),
            rev: int(v, "rev").unwrap_or(0),
            created_ms: int(v, "created_ms").unwrap_or(0),
            updated_ms: int(v, "updated_ms").unwrap_or(0),
        })
    }
}

impl Guidance {
    #[must_use]
    pub fn to_value(&self) -> Value {
        obj(vec![
            ("schema", s("nm-guidance/1")),
            ("version", n(self.version)),
            ("next", n(self.next)),
            ("reminders", Value::Array(self.reminders.iter().map(Reminder::to_value).collect())),
            ("loops", Value::Array(self.loops.iter().map(Loop::to_value).collect())),
            ("steps", Value::Array(self.steps.iter().map(Step::to_value).collect())),
            ("note", s("Reminders, loops and steps for the coding agent, written by the Neuromorph Nano. Every change is in memory/audit.jsonl and can be undone there.")),
        ])
    }
    /// # Errors
    /// Not a guidance object.
    pub fn from_value(v: &Value) -> Result<Self, String> {
        if !matches!(v, Value::Object(_)) {
            return Err("guidance.json: not an object".into());
        }
        let items = |k: &str| v.get(k).and_then(Value::array).map(<[Value]>::to_vec).unwrap_or_default();
        Ok(Self {
            version: int(v, "version").unwrap_or(0),
            next: int(v, "next").unwrap_or(0),
            reminders: items("reminders").iter().filter_map(Reminder::from_value).collect(),
            loops: items("loops").iter().filter_map(Loop::from_value).collect(),
            steps: items("steps").iter().filter_map(Step::from_value).collect(),
        })
    }
    fn is_empty(&self) -> bool {
        self.reminders.is_empty() && self.loops.is_empty() && self.steps.is_empty()
    }
    fn fresh_id(&mut self, prefix: &str) -> String {
        loop {
            self.next += 1;
            let id = format!("{prefix}{}", self.next);
            if !self.reminders.iter().any(|r| r.id == id) && !self.loops.iter().any(|l| l.id == id) && !self.steps.iter().any(|x| x.id == id) {
                return id;
            }
        }
    }
    /// One item as a value, by kind and id.
    fn item(&self, kind: &str, id: &str) -> Option<Value> {
        match kind {
            "reminder" => self.reminders.iter().find(|r| r.id == id).map(Reminder::to_value),
            "loop" => self.loops.iter().find(|l| l.id == id).map(Loop::to_value),
            "step" => self.steps.iter().find(|x| x.id == id).map(Step::to_value),
            _ => None,
        }
    }
    /// Puts an item back as `v` (or takes it out when `None`), keeping its place in the list.
    fn set_item(&mut self, kind: &str, id: &str, v: Option<&Value>) -> Result<(), String> {
        fn put<T>(list: &mut Vec<T>, at: Option<usize>, x: Option<T>) {
            match (at, x) {
                (Some(i), Some(x)) => {
                    if let Some(slot) = list.get_mut(i) {
                        *slot = x;
                    }
                }
                (Some(i), None) => {
                    list.remove(i);
                }
                (None, Some(x)) => list.push(x),
                (None, None) => {}
            }
        }
        match kind {
            "reminder" => {
                let at = self.reminders.iter().position(|r| r.id == id);
                put(&mut self.reminders, at, v.and_then(Reminder::from_value));
            }
            "loop" => {
                let at = self.loops.iter().position(|l| l.id == id);
                put(&mut self.loops, at, v.and_then(Loop::from_value));
            }
            "step" => {
                let at = self.steps.iter().position(|x| x.id == id);
                put(&mut self.steps, at, v.and_then(Step::from_value));
                graph_ok(&self.steps)?;
            }
            other => return Err(format!("kind {other}: use reminder, loop or step")),
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// The step graph.

/// `done`, `ready` (every dependency done) or `blocked`.
#[must_use]
pub fn status_of(step: &Step, steps: &[Step]) -> &'static str {
    if step.done {
        "done"
    } else if step.depends_on.iter().all(|d| steps.iter().any(|x| x.id == *d && x.done)) && (step.depends_any.is_empty() || step.depends_any.iter().any(|d| steps.iter().any(|x| x.id == *d && x.done))) {
        "ready"
    } else {
        "blocked"
    }
}

/// The first ready step, in the order the steps were added.
#[must_use]
pub fn next_ready(steps: &[Step]) -> Option<&Step> {
    steps.iter().find(|x| status_of(x, steps) == "ready")
}

/// A cycle among the steps' dependencies, as the ids along it, if there is one.
#[must_use]
pub fn cycle(steps: &[Step]) -> Option<Vec<String>> {
    // 0 unseen, 1 on the path, 2 finished.
    fn visit<'a>(id: &'a str, steps: &'a [Step], mark: &mut BTreeMap<&'a str, u8>, path: &mut Vec<&'a str>) -> Option<Vec<String>> {
        match mark.get(id).copied().unwrap_or(0) {
            1 => {
                let from = path.iter().position(|p| *p == id).unwrap_or(0);
                let mut c: Vec<String> = path.get(from..).unwrap_or_default().iter().map(|x| (*x).to_owned()).collect();
                c.push(id.to_owned());
                return Some(c);
            }
            2 => return None,
            _ => {}
        }
        mark.insert(id, 1);
        path.push(id);
        if let Some(st) = steps.iter().find(|x| x.id == id) {
            for d in st.depends_on.iter().chain(st.depends_any.iter()) {
                if let Some(c) = visit(d, steps, mark, path) {
                    return Some(c);
                }
            }
        }
        path.pop();
        mark.insert(id, 2);
        None
    }
    let mut mark = BTreeMap::new();
    for st in steps {
        let mut path = Vec::new();
        if let Some(c) = visit(&st.id, steps, &mut mark, &mut path) {
            return Some(c);
        }
    }
    None
}

/// Every dependency names a step, no step depends on itself, and there is no cycle.
fn graph_ok(steps: &[Step]) -> Result<(), String> {
    for st in steps {
        if let Some(d) = st.depends_on.iter().find(|d| st.depends_any.contains(d)) {
            return Err(format!("step {}: {d} is in both depends_on and depends_any; keep it in one", st.id));
        }
        if st.probe && st.check.is_none() {
            return Err(format!("step {}: a probe needs a check (build or test)", st.id));
        }
        for d in st.depends_on.iter().chain(st.depends_any.iter()) {
            if d == &st.id {
                return Err(format!("step {} cannot depend on itself", st.id));
            }
            if !steps.iter().any(|x| x.id == *d) {
                return Err(format!("step {} depends on {d}, which is not a step", st.id));
            }
        }
    }
    match cycle(steps) {
        Some(c) => Err(format!("refused: the dependencies would make a cycle ({})", c.join(" -> "))),
        None => Ok(()),
    }
}

// ---------------------------------------------------------------------------------------------
// Globs.

/// Whether `path` (relative, either slash) matches `pat`: `*` within one folder, `**` across
/// folders, `?` one character. A pattern without a slash also matches a file's name at any depth
/// (`*.rs`). Case is ignored on Windows.
#[must_use]
pub fn glob_match(pat: &str, path: &str) -> bool {
    let norm = |t: &str| {
        let t = t.trim().replace('\\', "/");
        let t = t.trim_start_matches("./").to_owned();
        if cfg!(windows) { t.to_ascii_lowercase() } else { t }
    };
    let (p, x) = (norm(pat), norm(path));
    if p.is_empty() {
        return false;
    }
    if !p.contains('/') {
        let base = x.rsplit('/').next().unwrap_or(&x);
        if wild(p.as_bytes(), base.as_bytes()) {
            return true;
        }
    }
    wild(p.as_bytes(), x.as_bytes())
}

fn wild(p: &[u8], t: &[u8]) -> bool {
    let rest = |k: usize| p.get(k..).unwrap_or_default();
    let tail = |k: usize| t.get(k..).unwrap_or_default();
    match p.first() {
        None => t.is_empty(),
        Some(b'*') if p.get(1) == Some(&b'*') => {
            let after = rest(2);
            // `**/` also matches no folder at all.
            if after.first() == Some(&b'/') && wild(after.get(1..).unwrap_or_default(), t) {
                return true;
            }
            (0..=t.len()).any(|i| wild(after, tail(i)))
        }
        Some(b'*') => {
            for i in 0..=t.len() {
                if wild(rest(1), tail(i)) {
                    return true;
                }
                if t.get(i) == Some(&b'/') {
                    break;
                }
            }
            false
        }
        Some(b'?') => t.first().is_some_and(|c| *c != b'/') && wild(rest(1), tail(1)),
        Some(c) => t.first() == Some(c) && wild(rest(1), tail(1)),
    }
}

fn valid_glob(g: &str, what: &str) -> Result<String, String> {
    let g = g.trim();
    if g.is_empty() || g.chars().count() > MAX_GLOB || g.matches('*').count() > 8 {
        return Err(format!("{what}: a glob of 1 to {MAX_GLOB} characters with at most 8 stars"));
    }
    Ok(g.to_owned())
}

// ---------------------------------------------------------------------------------------------
// Changes.

/// One change to the guidance: what was done to which item, and the item before and after.
#[derive(Clone, Debug, PartialEq)]
pub struct Change {
    pub action: String,
    pub kind: String,
    pub id: String,
    pub before: Option<Value>,
    pub after: Option<Value>,
    pub label: String,
}

/// Who asks and when, for a change.
#[derive(Clone, Copy)]
pub struct Asker<'a> {
    /// `user`, `agent` or `nano`.
    pub who: &'a str,
    /// The route's caller: `interface`, `mcp`, `cli`, `hook` or a paired device.
    pub via: &'a str,
    pub now: u64,
}

fn body_text(v: &Value, k: &str, what: &str) -> Result<Option<String>, String> {
    match text(v, k) {
        None => Ok(None),
        Some(t) => {
            let t = t.trim().to_owned();
            if t.is_empty() || t.chars().count() > MAX_TEXT {
                return Err(format!("{what}: 1 to {MAX_TEXT} characters"));
            }
            Ok(Some(t))
        }
    }
}
fn count_in(v: &Value, k: &str, lo: u64, hi: u64) -> Result<Option<u64>, String> {
    match v.get(k) {
        Err(_) | Ok(Value::Null) => Ok(None),
        Ok(x) => {
            let c = x.integer().ok().or_else(|| x.string().ok().and_then(|t| t.trim().parse().ok())).ok_or_else(|| format!("{k}: a whole number from {lo} to {hi}"))?;
            if c < lo || c > hi {
                return Err(format!("{k}: a whole number from {lo} to {hi}"));
            }
            Ok(Some(c))
        }
    }
}
fn one_of(v: &Value, k: &str, allowed: &[&str]) -> Result<Option<String>, String> {
    match text(v, k) {
        None => Ok(None),
        Some(t) if allowed.contains(&t.as_str()) => Ok(Some(t)),
        Some(t) => Err(format!("{k} {t}: use {}", allowed.join(", "))),
    }
}
fn goal_known(goal: Option<u64>, goals: &[(u64, String, String)]) -> Result<(), String> {
    match goal {
        Some(g) if !goals.iter().any(|(i, _, _)| *i == g) => Err(format!("goal {g}: no such objective in this project (nm_goal lists them)")),
        _ => Ok(()),
    }
}

/// A reminder from a request: a new one (`base` absent) or `base` with the given fields changed.
fn reminder_from(v: &Value, base: Option<&Reminder>) -> Result<Reminder, String> {
    let mut r = base.cloned().unwrap_or_else(|| Reminder { every: 1, repeat: "always".into(), max: 1, count: ERROR_COUNT_MIN, enabled: true, ..Reminder::default() });
    if let Some(t) = body_text(v, "text", "text")? {
        r.text = t;
    } else if base.is_none() {
        return Err("text: give the reminder's words".into());
    }
    if let Some(t) = one_of(v, "trigger", TRIGGERS)? {
        r.trigger = t;
    } else if base.is_none() {
        return Err(format!("trigger: use {}", TRIGGERS.join(", ")));
    }
    if let Some(e) = count_in(v, "every", 1, 1000)? {
        r.every = e;
    }
    if !matches!(r.trigger.as_str(), "prompt" | "tools") {
        r.every = 1;
    }
    if let Some(g) = text(v, "glob") {
        r.glob = if g.trim().is_empty() { None } else { Some(valid_glob(&g, "glob")?) };
    }
    if r.trigger != "edit" {
        r.glob = None;
    }
    if let Some(x) = one_of(v, "repeat", REPEATS)? {
        r.repeat = x;
    }
    if let Some(m) = count_in(v, "max", 1, 1000)? {
        r.max = m;
    } else if r.repeat == "max" && base.is_none_or(|b| b.repeat != "max") && v.get("max").is_err() {
        return Err("max: with repeat max, give how many times per session (1 to 1000)".into());
    }
    if let Some(f) = text(v, "files") {
        r.files = if f.trim().is_empty() { None } else { Some(valid_glob(&f, "files")?) };
    }
    if let Some(x) = text(v, "error") {
        let x = x.trim().to_owned();
        if x.is_empty() || x.chars().count() > MAX_SIGNATURE {
            return Err(format!("error: an error code or a piece of its message, 1 to {MAX_SIGNATURE} characters"));
        }
        r.error = Some(x);
    }
    if let Some(c) = count_in(v, "count", ERROR_COUNT_MIN, ERROR_COUNT_MAX)? {
        r.count = c;
    }
    if r.trigger == "error-repeats" {
        if r.error.is_none() {
            return Err("error: an error-repeats reminder needs the error code or a piece of its message (as the hook records it, for example E0308)".into());
        }
        r.count = r.count.clamp(ERROR_COUNT_MIN, ERROR_COUNT_MAX);
    } else {
        r.error = None;
    }
    if let Some(e) = flag(v, "enabled") {
        r.enabled = e;
    }
    Ok(r)
}

/// A budget field: absent keeps it (`None`), null or 0 clears it (`Some(None)`), else a whole
/// number from `lo` to `hi`.
fn budget_in(v: &Value, k: &str, lo: u64, hi: u64) -> Result<Option<Option<u64>>, String> {
    match v.get(k) {
        Err(_) => Ok(None),
        Ok(Value::Null) => Ok(Some(None)),
        Ok(x) if x.integer().ok() == Some(0) || x.string().ok().map(str::trim) == Some("0") => Ok(Some(None)),
        Ok(_) => count_in(v, k, lo, hi).map(Some),
    }
}

/// The condition named by `until`, written with a hyphen (`all_of` is taken as `all-of`).
fn condition_of(v: &Value) -> Option<String> {
    if v.get("all_of").is_ok_and(|x| !matches!(x, Value::Null)) {
        return Some("all-of".into());
    }
    if v.get("any_of").is_ok_and(|x| !matches!(x, Value::Null)) {
        return Some("any-of".into());
    }
    text(v, "until").map(|u| u.trim().replace('_', "-"))
}

/// Whether a loop's condition, or one of its parts, is `c`.
fn uses(l: &Loop, c: &str) -> bool {
    l.until == c || l.parts.iter().any(|p| p == c)
}

fn loop_from(v: &Value, base: Option<&Loop>, goals: &[(u64, String, String)]) -> Result<Loop, String> {
    let mut l = base.cloned().unwrap_or_else(|| Loop { max: LOOP_MAX_DEFAULT, stall: STALL_DEFAULT, state: "active".into(), enabled: true, ..Loop::default() });
    if let Some(t) = body_text(v, "prompt", "prompt")? {
        l.prompt = t;
    } else if base.is_none() {
        return Err("prompt: give what the agent is to do while the condition is unmet".into());
    }
    match condition_of(v) {
        Some(u) if CONDITIONS.contains(&u.as_str()) => l.until = u,
        Some(u) => return Err(format!("until {u}: use {}", CONDITIONS.join(", "))),
        None if base.is_none() => return Err(format!("until: use {}", CONDITIONS.join(", "))),
        None => {}
    }
    let given = ["all_of", "any_of", "parts"].iter().find_map(|k| list(v, k).filter(|_| v.get(k).is_ok_and(|x| !matches!(x, Value::Null))));
    if let Some(p) = given {
        let mut seen = BTreeSet::new();
        let p: Vec<String> = p.into_iter().map(|x| x.replace('_', "-")).collect();
        if let Some(bad) = p.iter().find(|x| !PART_CONDITIONS.contains(&x.as_str())) {
            return Err(format!("parts: {bad} is not a condition; use {}", PART_CONDITIONS.join(", ")));
        }
        if !p.iter().all(|x| seen.insert(x.clone())) {
            return Err("parts: each condition once".into());
        }
        l.parts = p;
    }
    if matches!(l.until.as_str(), "all-of" | "any-of") {
        if l.parts.len() < PARTS_MIN || l.parts.len() > PARTS_MAX {
            return Err(format!("parts: {} takes {PARTS_MIN} to {PARTS_MAX} distinct conditions of {}", l.until, PART_CONDITIONS.join(", ")));
        }
    } else {
        l.parts.clear();
    }
    if v.get("goal").is_ok() {
        l.goal = count_in(v, "goal", 0, u64::MAX)?;
    }
    if uses(&l, "goal-done") && l.goal.is_none() {
        return Err("goal: a goal-done condition needs the objective's id".into());
    }
    if !uses(&l, "goal-done") && !uses(&l, "steps-done") {
        l.goal = None;
    }
    goal_known(l.goal, goals)?;
    if let Some(m) = count_in(v, "max", 1, LOOP_CAP)? {
        l.max = m;
    }
    if let Some(x) = count_in(v, "stall", 1, STALL_MAX)? {
        l.stall = x;
    }
    if let Some(b) = budget_in(v, "budget_ms", BUDGET_MS_MIN, BUDGET_MS_MAX)? {
        l.budget_ms = b;
    }
    if let Some(b) = budget_in(v, "budget_tools", 1, BUDGET_TOOLS_MAX)? {
        l.budget_tools = b;
    }
    if let Some(e) = flag(v, "enabled") {
        l.enabled = e;
    }
    Ok(l)
}

fn step_from(v: &Value, base: Option<&Step>, goals: &[(u64, String, String)]) -> Result<Step, String> {
    let mut st = base.cloned().unwrap_or_default();
    if let Some(t) = body_text(v, "title", "title")? {
        st.title = t;
    } else if base.is_none() {
        return Err("title: give what the step is".into());
    }
    if let Some(d) = list(v, "depends_on").or_else(|| list(v, "depends")) {
        let mut seen = BTreeSet::new();
        st.depends_on = d.into_iter().filter(|x| seen.insert(x.clone())).collect();
    }
    if let Some(d) = list(v, "depends_any") {
        let mut seen = BTreeSet::new();
        st.depends_any = d.into_iter().filter(|x| seen.insert(x.clone())).collect();
    }
    match v.get("probe") {
        Ok(Value::Bool(b)) => st.probe = *b,
        Ok(Value::String(t)) if matches!(t.trim(), "true" | "false") => st.probe = t.trim() == "true",
        Ok(Value::Null) | Err(_) => {}
        Ok(_) => return Err("probe: true or false".into()),
    }
    if let Some(c) = text(v, "check") {
        st.check = match c.as_str() {
            "" | "none" => None,
            x if CHECKS.contains(&x) => Some(x.to_owned()),
            x => return Err(format!("check {x}: use build, test or none")),
        };
    }
    if let Some(p) = list(v, "paths") {
        st.paths = p.iter().map(|g| valid_glob(g, "paths")).collect::<Result<_, _>>()?;
    }
    if v.get("goal").is_ok() {
        st.goal = count_in(v, "goal", 0, u64::MAX)?;
    }
    goal_known(st.goal, goals)?;
    Ok(st)
}

fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 40 && id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

fn noun(kind: &str) -> &'static str {
    match kind {
        "reminder" => "Reminder",
        "loop" => "Loop",
        _ => "Step",
    }
}

/// Applies one request (`action`, `kind`, `id` and the item's fields) to `g`. `goals` are the
/// project's objectives, for the ids loops and steps name.
/// # Errors
/// An unknown action, kind or item, a field out of bounds, a step graph that would be invalid
/// (an unknown dependency, a cycle, a step others depend on removed), or nothing to change.
pub fn apply(g: &mut Guidance, v: &Value, a: Asker, goals: &[(u64, String, String)]) -> Result<Change, String> {
    // Worked on a copy, so a refused change leaves `g` exactly as it was.
    let mut work = g.clone();
    let ch = apply_in(&mut work, v, a, goals)?;
    *g = work;
    Ok(ch)
}

fn apply_in(g: &mut Guidance, v: &Value, a: Asker, goals: &[(u64, String, String)]) -> Result<Change, String> {
    let action = text(v, "action").ok_or_else(|| format!("action: use {}", ACTIONS.join(", ")))?;
    if !ACTIONS.contains(&action.as_str()) {
        return Err(format!("action {action}: use {}", ACTIONS.join(", ")));
    }
    let kind = text(v, "kind").ok_or("kind: use reminder, loop or step")?;
    if !KINDS.contains(&kind.as_str()) {
        return Err(format!("kind {kind}: use reminder, loop or step"));
    }
    let now = a.now;
    if action == "add" {
        return add(g, &kind, v, a, goals);
    }
    let id = text(v, "id").filter(|x| !x.is_empty()).ok_or_else(|| format!("id: give the {kind}'s id"))?;
    let before = g.item(&kind, &id).ok_or_else(|| format!("no {kind} {id}"))?;
    let label_of = |what: &str| format!("{} {id} {what}", noun(&kind));
    let label = match (action.as_str(), kind.as_str()) {
        ("edit", "reminder") => {
            let i = g.reminders.iter().position(|r| r.id == id).ok_or("reminder")?;
            let cur = g.reminders.get(i).cloned().ok_or("reminder")?;
            let mut r = reminder_from(v, Some(&cur))?;
            r.updated_ms = now;
            if let Some(slot) = g.reminders.get_mut(i) {
                *slot = r;
            }
            label_of("edited")
        }
        ("edit", "loop") => {
            let i = g.loops.iter().position(|l| l.id == id).ok_or("loop")?;
            let cur = g.loops.get(i).cloned().ok_or("loop")?;
            let mut l = loop_from(v, Some(&cur), goals)?;
            l.updated_ms = now;
            if let Some(slot) = g.loops.get_mut(i) {
                *slot = l;
            }
            label_of("edited")
        }
        ("edit", _) => {
            let i = g.steps.iter().position(|x| x.id == id).ok_or("step")?;
            let cur = g.steps.get(i).cloned().ok_or("step")?;
            let mut st = step_from(v, Some(&cur), goals)?;
            if st != cur {
                st.rev += 1;
            }
            st.updated_ms = now;
            if let Some(slot) = g.steps.get_mut(i) {
                *slot = st;
            }
            graph_ok(&g.steps)?;
            label_of("edited")
        }
        ("enable" | "disable", "step") => return Err("a step is not switched; mark it done, or reset it".into()),
        ("enable" | "disable", _) => {
            let on = action == "enable";
            let changed = match kind.as_str() {
                "reminder" => g.reminders.iter_mut().find(|r| r.id == id).map(|r| std::mem::replace(&mut r.enabled, on) != on),
                _ => g.loops.iter_mut().find(|l| l.id == id).map(|l| std::mem::replace(&mut l.enabled, on) != on),
            };
            if changed != Some(true) {
                return Err(format!("{kind} {id} is already {}", if on { "on" } else { "off" }));
            }
            label_of(if on { "switched on" } else { "switched off" })
        }
        ("done", "reminder") => return Err("a reminder is not done; switch it off or remove it".into()),
        ("done", "loop") => {
            let l = g.loops.iter_mut().find(|l| l.id == id).ok_or("loop")?;
            if l.state != "active" {
                return Err(format!("loop {id} is already {}", l.state));
            }
            l.state = "ended".into();
            l.ended_ms = Some(now);
            l.end_note = Some(format!("ended by {}", if a.who == "agent" { "the agent" } else { "you" }));
            l.updated_ms = now;
            label_of("ended")
        }
        ("done", _) => {
            let all = g.steps.clone();
            let st = g.steps.iter_mut().find(|x| x.id == id).ok_or("step")?;
            if st.done {
                return Err(format!("step {id} is already done"));
            }
            st.done = true;
            st.done_by = Some(a.who.to_owned());
            st.done_ms = Some(now);
            st.evidence = Some(obj(vec![("marked_by", s(a.who)), ("via", s(a.via)), ("was", s(status_of(st, &all)))]));
            st.updated_ms = now;
            label_of("marked done")
        }
        ("reset", "reminder") => {
            let r = g.reminders.iter_mut().find(|r| r.id == id).ok_or("reminder")?;
            r.reset_ms = now;
            r.updated_ms = now;
            label_of("reset (its deliveries count again from now)")
        }
        ("reset", "loop") => {
            let l = g.loops.iter_mut().find(|l| l.id == id).ok_or("loop")?;
            l.iterations = 0;
            l.state = "active".into();
            l.since_ms = now;
            l.ended_ms = None;
            l.end_note = None;
            l.last_unmet = None;
            l.repeats = 0;
            l.unmet_seq.clear();
            l.updated_ms = now;
            label_of("reset")
        }
        ("reset", _) => {
            let st = g.steps.iter_mut().find(|x| x.id == id).ok_or("step")?;
            st.done = false;
            st.done_by = None;
            st.done_ms = None;
            st.evidence = None;
            st.rev += 1;
            st.updated_ms = now;
            label_of("opened again")
        }
        ("remove", _) => {
            if kind == "step" {
                let users: Vec<&str> = g.steps.iter().filter(|x| x.depends_on.contains(&id) || x.depends_any.contains(&id)).map(|x| x.id.as_str()).collect();
                if !users.is_empty() {
                    return Err(format!("step {id} is a dependency of {}; change or remove those first", users.join(", ")));
                }
            }
            g.set_item(&kind, &id, None)?;
            label_of("removed")
        }
        _ => return Err(format!("action {action}: use {}", ACTIONS.join(", "))),
    };
    let after = g.item(&kind, &id);
    if after.as_ref() == Some(&before) {
        return Err(format!("{kind} {id}: nothing to change"));
    }
    Ok(Change { action, kind, id, before: Some(before), after, label })
}

fn add(g: &mut Guidance, kind: &str, v: &Value, a: Asker, goals: &[(u64, String, String)]) -> Result<Change, String> {
    let now = a.now;
    let (id, after) = match kind {
        "reminder" => {
            if g.reminders.len() >= MAX_ITEMS {
                return Err(format!("at most {MAX_ITEMS} reminders in a project"));
            }
            let mut r = reminder_from(v, None)?;
            r.id = g.fresh_id("r");
            (r.by, r.via, r.created_ms, r.updated_ms) = (a.who.to_owned(), a.via.to_owned(), now, now);
            g.reminders.push(r.clone());
            (r.id.clone(), r.to_value())
        }
        "loop" => {
            if g.loops.len() >= MAX_ITEMS {
                return Err(format!("at most {MAX_ITEMS} loops in a project"));
            }
            let mut l = loop_from(v, None, goals)?;
            l.id = g.fresh_id("l");
            (l.by, l.via, l.created_ms, l.updated_ms, l.since_ms) = (a.who.to_owned(), a.via.to_owned(), now, now, now);
            g.loops.push(l.clone());
            (l.id.clone(), l.to_value())
        }
        _ => {
            if g.steps.len() >= MAX_ITEMS {
                return Err(format!("at most {MAX_ITEMS} steps in a project"));
            }
            let mut st = step_from(v, None, goals)?;
            st.id = match text(v, "id").filter(|x| !x.is_empty()) {
                Some(id) if !valid_id(&id) => return Err(format!("id {id}: 1 to 40 letters, digits, '-', '_' or '.'")),
                Some(id) if g.item("step", &id).is_some() || g.item("reminder", &id).is_some() || g.item("loop", &id).is_some() => return Err(format!("id {id} is taken")),
                Some(id) => id,
                None => g.fresh_id("s"),
            };
            (st.by, st.via, st.created_ms, st.updated_ms) = (a.who.to_owned(), a.via.to_owned(), now, now);
            g.steps.push(st.clone());
            if let Err(e) = graph_ok(&g.steps) {
                g.steps.pop();
                return Err(e);
            }
            (st.id.clone(), st.to_value())
        }
    };
    let label = format!("{} {id} added", noun(kind));
    Ok(Change { action: "add".into(), kind: kind.to_owned(), id, before: None, after: Some(after), label })
}

// ---------------------------------------------------------------------------------------------
// Storage.

/// The guidance file of a project memory folder.
#[must_use]
pub fn file_of(memory: &Path) -> PathBuf {
    memory.join(FILE)
}

/// The guidance of a project memory folder (empty when there is none yet).
/// # Errors
/// The file exists and cannot be read or is not guidance (it is then never written over).
pub fn load(memory: &Path) -> Result<Guidance, String> {
    let f = file_of(memory);
    match std::fs::read(&f) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Guidance::default()),
        Err(e) => Err(format!("{}: {e}", f.display())),
        Ok(b) => Guidance::from_value(&nmjson::parse(&b).map_err(|e| format!("{}: {e}", f.display()))?),
    }
}

/// Writes the whole file through a temporary file renamed over it, the version counter raised.
fn save(memory: &Path, g: &mut Guidance) -> Result<(), String> {
    g.version += 1;
    std::fs::create_dir_all(memory).map_err(|e| format!("{}: {e}", memory.display()))?;
    let f = file_of(memory);
    let tmp = memory.join(format!("{FILE}.{}-{}.tmp", std::process::id(), nmtype::now_ms()));
    std::fs::write(&tmp, nmjson::pretty::pretty(&g.to_value())).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &f).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("{}: {e}", f.display())
    })
}

static WRITE: Mutex<()> = Mutex::new(());

/// Runs `f` on the newest guidance and saves it. When another process wrote the file meanwhile
/// (its version moved), `f` runs again on that newer file, so neither write is lost.
fn update<T>(memory: &Path, mut f: impl FnMut(&mut Guidance) -> Result<T, String>) -> Result<(T, Guidance), String> {
    let _g = WRITE.lock().map_err(|_| "guidance lock poisoned".to_owned())?;
    for _ in 0..5 {
        let mut g = load(memory)?;
        let seen = g.version;
        let r = f(&mut g)?;
        if load(memory)?.version != seen {
            continue;
        }
        save(memory, &mut g)?;
        return Ok((r, g));
    }
    Err("guidance.json kept changing under this write; nothing was written, try again".into())
}

fn append_line(file: &Path, v: &Value) -> Result<(), String> {
    use std::io::Write;
    let mut line = v.encode().map_err(|e| e.to_string())?;
    line.push('\n');
    if let Some(d) = file.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(file).map_err(|e| format!("{}: {e}", file.display()))?;
    f.write_all(line.as_bytes()).map_err(|e| format!("{}: {e}", file.display()))
}

fn lines_of(file: &Path) -> Vec<Value> {
    std::fs::read_to_string(file).unwrap_or_default().lines().filter(|l| !l.trim().is_empty()).filter_map(|l| nmjson::parse(l.as_bytes()).ok()).collect()
}

/// The audit row of a change, and its before and after kept for undo.
#[expect(clippy::too_many_arguments, reason = "an audit row names its home, project folder and key, the change, who asked, the version written and the row it undoes")]
fn audit_change(home: &Path, memory: &Path, key: &str, ch: &Change, via: &str, version: u64, undo_of: Option<&str>) -> Result<String, String> {
    let mut fields = vec![
        ("action", s(&ch.action)),
        ("item", s(&ch.kind)),
        ("guidance_id", s(&ch.id)),
        ("project", s(key)),
        ("by", s(via)),
        ("version", n(version)),
        ("label", s(&format!("{} in {key}", ch.label))),
    ];
    if let Some(u) = undo_of {
        fields.push(("undo_of", s(u)));
    }
    let id = crate::controls::audit(home, "guidance", fields)?;
    append_line(
        &memory.join(HISTORY),
        &obj(vec![
            ("audit", s(&id)),
            ("t", n(nmtype::now_ms())),
            ("item", s(&ch.kind)),
            ("id", s(&ch.id)),
            ("before", ch.before.clone().unwrap_or(Value::Null)),
            ("after", ch.after.clone().unwrap_or(Value::Null)),
        ]),
    )?;
    Ok(id)
}

/// A project's key and memory folder.
fn memory_of(home: &Path, key: &str) -> Result<(String, PathBuf), String> {
    crate::ledger::project_memory(home, &format!("project={key}")).ok_or_else(|| format!("project: no project {key} (give a key as /api/projects lists it, or nano)"))
}

/// The key `/api/projects` names a project memory folder by: `nano` for the Nano's own, else the
/// folder's name.
fn key_of(home: &Path, memory: &Path) -> String {
    if memory == home.join("memory") {
        return "nano".into();
    }
    memory.file_name().map_or_else(|| "nano".into(), |f| f.to_string_lossy().into_owned())
}

/// The objectives of a project: as the serving process holds them when it serves the project here,
/// else as saved.
fn objectives(memory: &Path) -> Vec<(u64, String, String)> {
    match nmapi::memory_write::served_here(memory) {
        Some(o) => o.objectives.lock().map(|g| g.clone()).unwrap_or_default(),
        None => nmapi::memory_write::saved_objectives(memory),
    }
}

/// `POST /api/guidance {project, action, kind, id, ...}`: one change, audited with undo. The
/// caller `by` names who set an item: `mcp` is the agent, anything else the user. A remove has
/// passed the delete gate (`forget::gate`, which sets `confirm`).
/// # Errors
/// An unknown project, a refused change, or the write failing.
pub fn write(home: &Path, v: &Value, by: &str) -> Result<Value, String> {
    let key = text(v, "project").ok_or("project: give a project key")?;
    let (key, memory) = memory_of(home, &key)?;
    if text(v, "action").as_deref() == Some("remove") && flag(v, "confirm") != Some(true) {
        return Err("remove goes through the delete gate: send it without a token for the preview, then again with its confirm_token".into());
    }
    let goals = objectives(&memory);
    let asker = Asker { who: who_of(by), via: by, now: nmtype::now_ms() };
    let (ch, g) = update(&memory, |g| apply(g, v, asker, &goals))?;
    let audit = audit_change(home, &memory, &key, &ch, by, g.version, None)?;
    record_outcomes(&memory, std::slice::from_ref(&ch), asker.now, &|| crate::sessions::records(&memory))?;
    Ok(obj(vec![
        ("project", s(&key)),
        ("action", s(&ch.action)),
        ("kind", s(&ch.kind)),
        ("id", s(&ch.id)),
        ("version", n(g.version)),
        ("audit_id", s(&audit)),
        ("note", s(&format!("{}. Audit undoes it.", ch.label))),
        ("guidance", view_of(home, &key, &memory)),
    ]))
}

/// What a remove would take, for the delete gate's preview: (title, item line, confirm word).
/// # Errors
/// An unknown project, kind or item, or a step others depend on.
pub fn remove_plan(home: &Path, v: &Value) -> Result<(String, String, String), String> {
    let key = text(v, "project").ok_or("project: give a project key")?;
    let (key, memory) = memory_of(home, &key)?;
    let kind = text(v, "kind").filter(|k| KINDS.contains(&k.as_str())).ok_or("kind: use reminder, loop or step")?;
    let id = text(v, "id").filter(|x| !x.is_empty()).ok_or("id: give the item's id")?;
    let g = load(&memory)?;
    let item = g.item(&kind, &id).ok_or_else(|| format!("no {kind} {id} in {key}"))?;
    if kind == "step" && let Some(u) = g.steps.iter().find(|x| x.depends_on.contains(&id) || x.depends_any.contains(&id)) {
        return Err(format!("step {id} is a dependency of {}; change or remove that first", u.id));
    }
    let words = ["text", "prompt", "title"].iter().find_map(|k| text(&item, k)).unwrap_or_default();
    Ok((format!("Remove the {kind} {id} from {key}"), format!("{kind} {id}: {words}"), id))
}

/// Undoes a guidance change by its audit row: the item is put back as it was before (an added item
/// is taken out again). The undo is itself a change, logged and undoable.
/// # Errors
/// No history for the row, or the restored step graph would be invalid.
pub fn undo(home: &Path, row: &Value, audit_id: &str, by: &str) -> Result<Option<String>, String> {
    let key = text(row, "project").unwrap_or_default();
    let (key, memory) = memory_of(home, &key)?;
    let h = lines_of(&memory.join(HISTORY)).into_iter().rfind(|x| text(x, "audit").as_deref() == Some(audit_id)).ok_or_else(|| format!("{audit_id}: no guidance history kept for it; it cannot be undone"))?;
    let kind = text(&h, "item").unwrap_or_default();
    let id = text(&h, "id").unwrap_or_default();
    let before = h.get("before").ok().filter(|b| !matches!(b, Value::Null)).cloned();
    let (ch, g) = update(&memory, |g| {
        let current = g.item(&kind, &id);
        g.set_item(&kind, &id, before.as_ref())?;
        Ok(Change { action: "undo".into(), kind: kind.clone(), id: id.clone(), before: current, after: before.clone(), label: format!("Undone: {} {id} put back as before {audit_id}", noun(&kind)) })
    })?;
    let id = audit_change(home, &memory, &key, &ch, by, g.version, Some(audit_id))?;
    record_outcomes(&memory, std::slice::from_ref(&ch), nmtype::now_ms(), &|| crate::sessions::records(&memory))?;
    Ok(Some(id))
}

// ---------------------------------------------------------------------------------------------
// Loop conditions.

fn check_label(r: &Rec) -> String {
    format!("{}{}", r.cmd_class.as_deref().unwrap_or(&r.kind), r.pkg.as_deref().map(|p| format!(" {p}")).unwrap_or_default())
}

/// `check-passes`: in `session`, from `since`, the last build or test after the last edit passed.
#[must_use]
pub fn check_passes(recs: &[Rec], session: &str, since: u64) -> (bool, String) {
    let mine: Vec<&Rec> = recs.iter().filter(|r| r.session == session && r.t >= since && r.done()).collect();
    let last_edit = mine.iter().rposition(|r| matches!(r.kind.as_str(), "edit" | "write") && r.ok == Some(true));
    let last_check = mine.iter().enumerate().rev().find(|(i, r)| matches!(r.kind.as_str(), "build" | "test") && last_edit.is_none_or(|e| *i > e)).map(|(_, r)| *r);
    match (last_check, last_edit) {
        (Some(c), _) if c.ok == Some(true) => (true, format!("the last check after the last edit, {}, passed", check_label(c))),
        (Some(c), _) => {
            let at = c.errors.first().map(|(code, f, l)| format!(" ({}{f}:{l})", if code.is_empty() { String::new() } else { format!("{code} at ") })).unwrap_or_default();
            (false, format!("the last check after the last edit, {}, failed{at}", check_label(c)))
        }
        (None, Some(_)) => (false, "no build or test ran after the last edit".into()),
        (None, None) => (false, "no build or test has run since the loop was set".into()),
    }
}

/// Whether a loop's condition holds now, and in words why or why not. A combined condition names
/// the parts not yet met (`all-of`) or every part, none met (`any-of`).
fn evaluate(l: &Loop, g: &Guidance, goals: &[(u64, String, String)], recs: &dyn Fn() -> Vec<Rec>, session: &str) -> (bool, String) {
    if !matches!(l.until.as_str(), "all-of" | "any-of") {
        return evaluate_one(&l.until, l, g, goals, recs, session);
    }
    let parts: Vec<(&str, bool, String)> = l.parts.iter().map(|p| {
        let (met, why) = evaluate_one(p, l, g, goals, recs, session);
        (p.as_str(), met, why)
    }).collect();
    let said = |list: &[&(&str, bool, String)]| list.iter().map(|(p, _, why)| format!("{p} ({why})")).collect::<Vec<_>>().join("; ");
    let met: Vec<&(&str, bool, String)> = parts.iter().filter(|x| x.1).collect();
    let unmet: Vec<&(&str, bool, String)> = parts.iter().filter(|x| !x.1).collect();
    if l.until == "all-of" {
        if unmet.is_empty() { (true, format!("all of {} met: {}", parts.len(), said(&met))) } else { (false, format!("all-of, not yet met: {}", said(&unmet))) }
    } else if met.is_empty() {
        (false, format!("any-of, none met: {}", said(&unmet)))
    } else {
        (true, format!("any-of met by {}", said(&met)))
    }
}

/// One condition of a loop, by name.
fn evaluate_one(cond: &str, l: &Loop, g: &Guidance, goals: &[(u64, String, String)], recs: &dyn Fn() -> Vec<Rec>, session: &str) -> (bool, String) {
    match cond {
        "check-passes" => check_passes(&recs(), session, l.since_ms),
        "goal-done" => {
            let gid = l.goal.unwrap_or(0);
            match goals.iter().find(|(i, _, _)| *i == gid) {
                Some((_, _, st)) if st != "open" => (true, format!("goal {gid} is {st}")),
                Some(_) => (false, format!("goal {gid} is still open")),
                None => (false, format!("goal {gid} is not among this project's objectives")),
            }
        }
        "steps-done" => {
            let mine: Vec<&Step> = g.steps.iter().filter(|x| l.goal.is_none_or(|gid| x.goal == Some(gid))).collect();
            if mine.is_empty() {
                return (false, "no steps yet".into());
            }
            let open: Vec<String> = mine.iter().filter(|x| !x.done).map(|x| format!("{} ({})", x.id, status_of(x, &g.steps))).collect();
            if open.is_empty() {
                (true, format!("all {} steps are done", mine.len()))
            } else {
                let shown: Vec<&String> = open.iter().take(6).collect();
                let more = open.len().saturating_sub(shown.len());
                (false, format!("steps not done: {}{}", shown.iter().map(|x| x.as_str()).collect::<Vec<_>>().join(", "), if more > 0 { format!(" and {more} more") } else { String::new() }))
            }
        }
        _ => (false, "this loop runs until you or the agent end it".into()),
    }
}

// ---------------------------------------------------------------------------------------------
// Per-session counters.

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Memo {
    prompts: u64,
    tools: u64,
    /// (reminder id, time) of each delivery.
    delivered: Vec<(String, u64)>,
    /// `step@rev` of each step nudged.
    nudged: Vec<String>,
    touched: Vec<String>,
    /// (reminder id, tool results carrying its error since it was last delivered), for
    /// `error-repeats`.
    errors: Vec<(String, u64)>,
}

fn memo_file(memory: &Path, session: &str) -> PathBuf {
    let name: String = session.chars().filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_')).take(80).collect();
    let name = if name.is_empty() || name.len() != session.len() { nmhash::sha256_bytes(session.as_bytes()).chars().take(16).collect() } else { name };
    memory.join(MEMO_DIR).join(format!("{name}.json"))
}

impl Memo {
    fn load(memory: &Path, session: &str) -> Self {
        let Some(v) = std::fs::read(memo_file(memory, session)).ok().and_then(|b| nmjson::parse(&b).ok()) else { return Self::default() };
        let delivered = v
            .get("delivered")
            .and_then(Value::array)
            .unwrap_or_default()
            .iter()
            .filter_map(|x| {
                let a = x.array().ok()?;
                Some((a.first()?.string().ok()?.to_owned(), a.get(1)?.integer().ok()?))
            })
            .collect();
        let errors = v
            .get("errors")
            .and_then(Value::array)
            .unwrap_or_default()
            .iter()
            .filter_map(|x| {
                let a = x.array().ok()?;
                Some((a.first()?.string().ok()?.to_owned(), a.get(1)?.integer().ok()?))
            })
            .collect();
        Self { prompts: int(&v, "prompts").unwrap_or(0), tools: int(&v, "tools").unwrap_or(0), delivered, nudged: list(&v, "nudged").unwrap_or_default(), touched: list(&v, "touched").unwrap_or_default(), errors }
    }
    fn save(&self, memory: &Path, session: &str) -> Result<(), String> {
        let f = memo_file(memory, session);
        if let Some(d) = f.parent() {
            std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        let v = obj(vec![
            ("prompts", n(self.prompts)),
            ("tools", n(self.tools)),
            ("delivered", Value::Array(self.delivered.iter().map(|(id, t)| Value::Array(vec![s(id), n(*t)])).collect())),
            ("nudged", strings(&self.nudged)),
            ("touched", strings(&self.touched)),
            ("errors", Value::Array(self.errors.iter().map(|(id, k)| Value::Array(vec![s(id), n(*k)])).collect())),
        ]);
        let tmp = f.with_extension(format!("{}.tmp", std::process::id()));
        std::fs::write(&tmp, v.encode().map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &f).map_err(|e| format!("{}: {e}", f.display()))
    }
    fn count(&self, id: &str, since: u64) -> u64 {
        self.delivered.iter().filter(|(x, t)| x == id && *t >= since).count() as u64
    }
    /// Tool results that carried an `error-repeats` reminder's error since it was last delivered.
    fn hits(&self, id: &str) -> u64 {
        self.errors.iter().find(|(x, _)| x == id).map_or(0, |(_, k)| *k)
    }
    fn hit(&mut self, id: &str) {
        match self.errors.iter_mut().find(|(x, _)| x == id) {
            Some((_, k)) => *k += 1,
            None => self.errors.push((id.to_owned(), 1)),
        }
    }
    fn clear_hits(&mut self, id: &str) {
        self.errors.retain(|(x, _)| x != id);
    }
}

/// Whether a hook record carries the error `sig`: a located error whose code or message holds it,
/// or a failed tool's message that holds it (case ignored).
fn error_in(rec: &BTreeMap<String, Value>, sig: &str) -> bool {
    let want = sig.trim().to_ascii_lowercase();
    if want.is_empty() {
        return false;
    }
    let has = |t: &str| t.to_ascii_lowercase().contains(&want);
    rec.get("errors").and_then(|v| v.array().ok()).unwrap_or_default().iter().any(|e| ["code", "msg"].iter().any(|k| text(e, k).is_some_and(|x| has(&x)))) || field(rec, "msg").is_some_and(has)
}

/// Tool calls completed from `since` up to `until`, in every session of the project.
fn tools_since(recs: &[Rec], since: u64, until: u64) -> u64 {
    recs.iter().filter(|r| r.done() && r.t >= since && r.t <= until).count() as u64
}

/// Keeps an unmet reason on a loop: the count of stops in a row with the same reason as the one
/// before, and the sequence of distinct reasons.
fn note_unmet(l: &mut Loop, detail: &str) {
    if l.last_unmet.as_deref() == Some(detail) {
        l.repeats += 1;
    } else {
        l.repeats = 0;
    }
    let d: String = detail.chars().take(UNMET_CHARS).collect();
    if l.unmet_seq.last() != Some(&d) {
        l.unmet_seq.push(d);
        if l.unmet_seq.len() > UNMET_KEEP {
            l.unmet_seq.remove(0);
        }
    }
    l.last_unmet = Some(detail.to_owned());
}

/// Which budget of a loop is used, in words, if one is.
fn budget_spent(l: &Loop, used_ms: u64, used_tools: u64) -> Option<String> {
    if let Some(b) = l.budget_tools
        && used_tools >= b
    {
        return Some(format!("its tool budget is used ({used_tools} of {b} tool calls since it was set)"));
    }
    if let Some(b) = l.budget_ms
        && used_ms >= b
    {
        return Some(format!("its time budget is used ({} s of {} s since it was set)", used_ms / 1000, b / 1000));
    }
    None
}

/// The end reason an ended loop's state stands for.
fn end_reason(state: &str) -> &'static str {
    match state {
        "done" => "met",
        "exhausted" => "cap",
        "stalled" => "stalled",
        "budget-spent" => "budget-spent",
        _ => "ended",
    }
}

/// The outcome row of a change that ended an active loop (its condition met, its cap, stalled,
/// its budget spent, ended by hand, or removed), else `None`. The verdict is the one the evidence
/// ledger's `loop-met` rule gives each of the loop's continuations for that end.
fn outcome_of(ch: &Change, now: u64, recs: &[Rec]) -> Option<Value> {
    if ch.kind != "loop" {
        return None;
    }
    let before = ch.before.as_ref().and_then(Loop::from_value).filter(|l| l.state == "active")?;
    let (l, end) = match ch.after.as_ref().and_then(Loop::from_value) {
        None => (before, "removed"),
        Some(x) if x.state == "active" => return None,
        Some(x) => {
            let e = end_reason(&x.state);
            (x, e)
        }
    };
    let ended = l.ended_ms.unwrap_or(now);
    let verdict = match end {
        "met" => "verified",
        "cap" | "stalled" | "budget-spent" => "refuted",
        _ => "unknown",
    };
    Some(obj(vec![
        ("id", s(&l.id)),
        ("by", s(&l.by)),
        ("prompt", s(&l.prompt)),
        ("condition", obj(vec![("until", s(&l.until)), ("parts", strings(&l.parts)), ("goal", opt_n(l.goal))])),
        ("iterations", n(l.iterations)),
        ("max", n(l.max)),
        ("stall", n(l.stall)),
        ("budget", obj(vec![("ms", opt_n(l.budget_ms)), ("tools", opt_n(l.budget_tools)), ("used_ms", n(ended.saturating_sub(l.since_ms))), ("used_tools", n(tools_since(recs, l.since_ms, ended)))])),
        ("end", s(end)),
        ("end_note", opt_s(l.end_note.as_ref())),
        ("unmet", strings(&l.unmet_seq)),
        ("verdicts", obj(vec![("rule", s("loop-met")), ("state", s(verdict)), ("continuations", n(l.iterations))])),
        ("started_ms", n(l.since_ms)),
        ("ended_ms", n(ended)),
    ]))
}

/// Appends the outcome rows of the changes that ended a loop.
fn record_outcomes(memory: &Path, changes: &[Change], now: u64, recs: &dyn Fn() -> Vec<Rec>) -> Result<(), String> {
    if !changes.iter().any(|c| c.kind == "loop") {
        return Ok(());
    }
    let recs = recs();
    let file = memory.join(OUTCOMES);
    let mut wrote = false;
    for row in changes.iter().filter_map(|c| outcome_of(c, now, &recs)) {
        append_line(&file, &row)?;
        wrote = true;
    }
    if wrote {
        trim_deliveries(&file);
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The hook.

/// What the hook knows when guidance is looked at.
pub struct At<'a> {
    pub home: &'a Path,
    pub memory: &'a Path,
    /// The event, normalised (`UserPromptSubmit` for Antigravity's first `PreInvocation`).
    pub event: &'a str,
    /// The hook's own record of this event (not yet appended).
    pub rec: &'a BTreeMap<String, Value>,
    /// Antigravity, and whether this event takes injected text there.
    pub agy: bool,
    pub inject: bool,
    pub now: u64,
}

/// What guidance gives at one hook.
#[derive(Clone, Debug, PartialEq)]
pub struct Out {
    /// Texts for the agent's context, each whole.
    pub texts: Vec<String>,
    /// A loop's continuation: the Stop hook's `reason`.
    pub block: Option<String>,
    /// Loop ends, also shown to the user (`systemMessage`).
    pub notice: Vec<String>,
    /// The `guidance` field of the hook's situation record.
    pub record: Value,
}

impl Default for Out {
    fn default() -> Self {
        Self { texts: Vec::new(), block: None, notice: Vec::new(), record: Value::Null }
    }
}

fn field<'a>(rec: &'a BTreeMap<String, Value>, k: &str) -> Option<&'a str> {
    rec.get(k).and_then(|v| v.string().ok())
}

/// Whether a reminder's trigger fires at this event.
fn fires(r: &Reminder, event: &str, rec: &BTreeMap<String, Value>, memo: &Memo, paths: &[String]) -> bool {
    let post = matches!(event, "PostToolUse" | "PostToolUseFailure");
    match r.trigger.as_str() {
        "session-start" => event == "SessionStart" && field(rec, "source") != Some("compact"),
        "compact" => event == "SessionStart" && field(rec, "source") == Some("compact"),
        "prompt" => event == "UserPromptSubmit" && memo.prompts.is_multiple_of(r.every.max(1)),
        "stop" => event == "Stop",
        "edit" => {
            event == "PreToolUse"
                && matches!(field(rec, "tool"), Some("Edit" | "MultiEdit" | "Write" | "NotebookEdit"))
                && r.glob.as_deref().is_none_or(|g| paths.iter().any(|p| glob_match(g, p)))
        }
        "failure" => post && matches!(field(rec, "kind"), Some("build" | "test")) && matches!(rec.get("ok"), Some(Value::Bool(false))),
        "tools" => post && memo.tools.is_multiple_of(r.every.max(1)),
        "error-repeats" => post && r.error.as_deref().is_some_and(|sig| error_in(rec, sig)) && memo.hits(&r.id) >= r.count.max(ERROR_COUNT_MIN),
        _ => false,
    }
}

/// Whether the event is inside a reminder's file scope.
fn in_scope(r: &Reminder, paths: &[String], memo: &Memo) -> bool {
    let Some(f) = r.files.as_deref() else { return true };
    paths.iter().any(|p| glob_match(f, p)) || (r.trigger != "edit" && memo.touched.iter().any(|p| glob_match(f, p)))
}

fn repetition_allows(r: &Reminder, memo: &Memo) -> bool {
    let c = memo.count(&r.id, r.reset_ms);
    match r.repeat.as_str() {
        "once" => c == 0,
        "max" => c < r.max.max(1),
        _ => true,
    }
}

/// The steps a passing check marks done: ready steps with that check whose globs (any path when
/// none) hold an edit of this session made after the step was last changed; and probe steps with
/// that check that were ready before this check (their dependencies met), with no edit needed.
fn auto_done(g: &mut Guidance, rec: &BTreeMap<String, Value>, recs: &[Rec], session: &str, now: u64) -> Vec<Change> {
    let kind = field(rec, "kind").unwrap_or_default();
    let mut out = Vec::new();
    let before_check = g.steps.clone();
    for i in 0..g.steps.len() {
        let Some(st) = g.steps.get(i).cloned() else { continue };
        let fits = match st.check.as_deref() {
            Some("test") => kind == "test",
            Some("build") => matches!(kind, "build" | "test"),
            _ => false,
        };
        if st.done || !fits || status_of(&st, &g.steps) != "ready" {
            continue;
        }
        if st.probe && status_of(&st, &before_check) != "ready" {
            continue;
        }
        let edits: Vec<&Rec> = recs
            .iter()
            .filter(|r| r.session == session && r.done() && matches!(r.kind.as_str(), "edit" | "write") && r.ok == Some(true) && r.t >= st.updated_ms)
            .filter(|r| st.paths.is_empty() || r.paths.iter().any(|p| st.paths.iter().any(|gl| glob_match(gl, p))))
            .collect();
        if edits.is_empty() && !st.probe {
            continue;
        }
        let mut files: Vec<String> = Vec::new();
        for p in edits.iter().flat_map(|r| r.paths.iter()) {
            if files.len() < 8 && !files.contains(p) {
                files.push(p.clone());
            }
        }
        let before = st.to_value();
        let mut ev = vec![("marked_by", s("nano")), ("check", s(kind)), ("session", s(session)), ("at", n(now)), ("edits", strings(&files)), ("probe", Value::Bool(st.probe))];
        for k in ["cmd_class", "pkg", "cmd"] {
            if let Some(x) = field(rec, k) {
                ev.push((k, s(x)));
            }
        }
        if let Some(t) = rec.get("tests") {
            ev.push(("tests", t.clone()));
        }
        if let Some(slot) = g.steps.get_mut(i) {
            slot.done = true;
            slot.done_by = Some("nano".into());
            slot.done_ms = Some(now);
            slot.evidence = Some(obj(ev));
            slot.updated_ms = now;
            let how = if st.probe { "a probe, its check passed with its dependencies met" } else { "passed after edits under its paths" };
            out.push(Change { action: "auto-done".into(), kind: "step".into(), id: st.id.clone(), before: Some(before), after: Some(slot.to_value()), label: format!("Step {} marked done by the Nano: {kind} {how}", st.id) });
        }
    }
    out
}

fn iterations(k: u64) -> String {
    if k == 1 { "1 iteration".into() } else { format!("{k} iterations") }
}

fn step_text(st: &Step, g: &Guidance) -> String {
    let done = g.steps.iter().filter(|x| x.done).count();
    let mut after: Vec<String> = Vec::new();
    if !st.depends_on.is_empty() {
        after.push(st.depends_on.join(", "));
    }
    if !st.depends_any.is_empty() {
        after.push(format!("one of {}", st.depends_any.join(", ")));
    }
    let deps = if after.is_empty() { String::new() } else { format!(" (after {}, done)", after.join(" and ")) };
    let check = match st.check.as_deref() {
        Some(c) if st.probe => format!(" It is a probe: it is done when a {c} passes, with no edit needed."),
        Some(c) if st.paths.is_empty() => format!(" It is done when a {c} passes after your edits."),
        Some(c) => format!(" It is done when a {c} passes after edits under {}.", st.paths.join(", ")),
        None => " Mark it done with nm_goal op step_done when it is.".to_owned(),
    };
    format!("Neuromorph Nano step graph: the next ready step is {} \"{}\"{deps}; {done} of {} steps done.{check}", st.id, st.title, g.steps.len())
}

/// Reminders, step nudges and loops at one hook. Returns `None` when the project has no guidance
/// or nothing applies. A fault costs the guidance only, never the hook's record.
/// # Errors
/// The guidance or a counter file cannot be read or written.
pub fn at_hook(a: &At) -> Result<Option<Out>, String> {
    if !file_of(a.memory).is_file() {
        return Ok(None);
    }
    let session = field(a.rec, "session").unwrap_or_default().to_owned();
    if session.is_empty() {
        return Ok(None);
    }
    let g0 = load(a.memory)?;
    if g0.is_empty() {
        return Ok(None);
    }
    let controls = crate::controls::current(a.home);
    let reminders_on = controls.get("reminders").is_none_or(|v| v != "off");
    let loops_on = controls.get("loops").is_none_or(|v| v != "off");
    let mut memo = Memo::load(a.memory, &session);
    match a.event {
        "UserPromptSubmit" => memo.prompts += 1,
        "PostToolUse" | "PostToolUseFailure" => memo.tools += 1,
        _ => {}
    }
    let paths: Vec<String> = a.rec.get("paths").and_then(|v| v.array().ok()).unwrap_or_default().iter().filter_map(|p| p.string().ok().map(str::to_owned)).collect();
    for p in &paths {
        if memo.touched.len() < MAX_TOUCHED && !memo.touched.contains(p) {
            memo.touched.push(p.clone());
        }
    }
    if matches!(a.event, "PostToolUse" | "PostToolUseFailure") {
        for r in g0.reminders.iter().filter(|r| r.enabled && r.trigger == "error-repeats") {
            if r.error.as_deref().is_some_and(|sig| error_in(a.rec, sig)) {
                memo.hit(&r.id);
            }
        }
    }
    let can_give = reminders_on && (!a.agy || a.inject);
    let mut out = Out::default();
    let mut rows: Vec<Value> = Vec::new();
    let channel = if a.agy { "inject" } else { "context" };
    let row = |kind: &str, id: &str, channel: &str, t: &str| {
        obj(vec![("t", n(a.now)), ("session", s(&session)), ("event", s(a.event)), ("kind", s(kind)), ("id", s(id)), ("channel", s(channel)), ("host", s(if a.agy { "antigravity" } else { "claude-code" })), ("text", s(t))])
    };
    let mut given: Vec<String> = Vec::new();
    if can_give {
        for r in g0.reminders.iter().filter(|r| r.enabled) {
            if fires(r, a.event, a.rec, &memo, &paths) && in_scope(r, &paths, &memo) && repetition_allows(r, &memo) {
                let t = format!("Neuromorph Nano reminder ({}): {}", who_label(&r.by), r.text);
                memo.delivered.push((r.id.clone(), a.now));
                if r.trigger == "error-repeats" {
                    memo.clear_hits(&r.id);
                }
                rows.push(row("reminder", &r.id, channel, &t));
                given.push(r.id.clone());
                out.texts.push(t);
            }
        }
    }
    // Writes (a step the Nano marks done, a loop's progress) run on the newest file.
    let post_pass = matches!(a.event, "PostToolUse" | "PostToolUseFailure") && matches!(field(a.rec, "kind"), Some("build" | "test")) && matches!(a.rec.get("ok"), Some(Value::Bool(true)));
    let wants_steps = post_pass && g0.steps.iter().any(|x| !x.done && x.check.is_some());
    let wants_loops = a.event == "Stop" && loops_on && g0.loops.iter().any(|l| l.enabled && l.state == "active");
    let mut changes: Vec<Change> = Vec::new();
    let mut loop_rec: Vec<Value> = Vec::new();
    let mut block: Vec<String> = Vec::new();
    let mut g = g0.clone();
    if wants_steps || (wants_loops && !a.agy) {
        let recs = crate::sessions::records(a.memory);
        let goals = objectives(a.memory);
        let records = || recs.clone();
        let result = update(a.memory, |gx| {
            let mut ch = Vec::new();
            let (mut lr, mut bl, mut notes) = (Vec::new(), Vec::new(), Vec::new());
            if wants_steps {
                ch.extend(auto_done(gx, a.rec, &recs, &session, a.now));
            }
            if wants_loops && !a.agy {
                let snapshot = gx.clone();
                for l in gx.loops.iter_mut().filter(|l| l.enabled && l.state == "active") {
                    let before = l.to_value();
                    let (met, detail) = evaluate(l, &snapshot, &goals, &records, &session);
                    if !met {
                        note_unmet(l, &detail);
                    }
                    let spent = if met { None } else { budget_spent(l, a.now.saturating_sub(l.since_ms), tools_since(&recs, l.since_ms, u64::MAX)) };
                    // A manual loop has no progress reason, so the stall threshold does not apply to it.
                    let stalled = !met && l.until != "manual" && l.repeats >= l.stall;
                    let (action, label) = if met {
                        l.state = "done".into();
                        l.ended_ms = Some(a.now);
                        l.end_note = Some(format!("condition met: {detail}"));
                        notes.push((l.id.clone(), format!("Neuromorph Nano loop {} ended: its condition is met ({detail}) after {}.", l.id, iterations(l.iterations))));
                        ("loop-done", format!("Loop {} done: {detail}", l.id))
                    } else if stalled {
                        l.state = "stalled".into();
                        l.ended_ms = Some(a.now);
                        l.end_note = Some(format!("stalled: the same unmet reason at {} stops in a row: {detail}", l.repeats + 1));
                        notes.push((l.id.clone(), format!("Neuromorph Nano loop {} ended as stalled after {}: the same unmet reason came back at {} stops in a row ({detail}). It no longer continues you; change the approach or ask the user.", l.id, iterations(l.iterations), l.repeats + 1)));
                        ("loop-stalled", format!("Loop {} stalled after {}", l.id, iterations(l.iterations)))
                    } else if let Some(what) = spent {
                        l.state = "budget-spent".into();
                        l.ended_ms = Some(a.now);
                        l.end_note = Some(format!("budget spent: {what}; not met: {detail}"));
                        notes.push((l.id.clone(), format!("Neuromorph Nano loop {} ended as budget-spent after {}: {what}. Not met: {detail}.", l.id, iterations(l.iterations))));
                        ("loop-budget-spent", format!("Loop {} budget spent after {}", l.id, iterations(l.iterations)))
                    } else if l.iterations < l.max {
                        l.iterations += 1;
                        bl.push((l.id.clone(), format!("Neuromorph Nano loop {} ({}; continuing, {} of {}): {} Not met yet: {detail}.", l.id, who_label(&l.by), l.iterations, l.max, l.prompt)));
                        lr.push(obj(vec![("id", s(&l.id)), ("action", s("block")), ("iteration", n(l.iterations)), ("max", n(l.max))]));
                        continue;
                    } else {
                        l.state = "exhausted".into();
                        l.ended_ms = Some(a.now);
                        l.end_note = Some(format!("stopped at its maximum of {}: {detail}", iterations(l.max)));
                        notes.push((l.id.clone(), format!("Neuromorph Nano loop {} stopped after {} without its condition met: {detail}.", l.id, iterations(l.max))));
                        ("loop-exhausted", format!("Loop {} exhausted after {}", l.id, iterations(l.max)))
                    };
                    lr.push(obj(vec![("id", s(&l.id)), ("action", s(&l.state)), ("iteration", n(l.iterations)), ("max", n(l.max))]));
                    l.updated_ms = a.now;
                    ch.push(Change { action: action.into(), kind: "loop".into(), id: l.id.clone(), before: Some(before), after: Some(l.to_value()), label });
                }
            }
            if ch.is_empty() && bl.is_empty() {
                return Err(String::new());
            }
            Ok((ch, lr, bl, notes))
        });
        match result {
            Ok(((ch, lr, bl, notes), saved)) => {
                for c in &ch {
                    audit_change(a.home, a.memory, &key_of(a.home, a.memory), c, "hook", saved.version, None)?;
                }
                record_outcomes(a.memory, &ch, a.now, &|| recs.clone())?;
                for (id, t) in &bl {
                    rows.push(row("loop", id, "block", t));
                }
                for (id, t) in &notes {
                    rows.push(row("loop", id, "note", t));
                    out.notice.push(t.clone());
                }
                changes = ch;
                loop_rec = lr;
                block = bl.into_iter().map(|(_, t)| t).collect();
                g = saved;
            }
            Err(e) if e.is_empty() => {}
            Err(e) => return Err(e),
        }
    }
    if wants_loops && a.agy {
        for l in g0.loops.iter().filter(|l| l.enabled && l.state == "active") {
            rows.push(row("loop", &l.id, "none", AGY_LOOP));
            loop_rec.push(obj(vec![("id", s(&l.id)), ("action", s("not-acted")), ("host", s("antigravity"))]));
        }
    }
    let mut nudged = None;
    if can_give && matches!(a.event, "UserPromptSubmit" | "Stop")
        && let Some(st) = next_ready(&g.steps)
    {
        let mark = format!("{}@{}", st.id, st.rev);
        if !memo.nudged.contains(&mark) {
            let t = step_text(st, &g);
            memo.nudged.push(mark);
            rows.push(row("step", &st.id, channel, &t));
            out.texts.push(t);
            nudged = Some(st.id.clone());
        }
    }
    let auto: Vec<String> = changes.iter().filter(|c| c.action == "auto-done").map(|c| c.id.clone()).collect();
    for c in changes.iter().filter(|c| c.action == "auto-done") {
        if can_give {
            let (title, probe) = g.steps.iter().find(|x| x.id == c.id).map(|x| (x.title.clone(), x.probe)).unwrap_or_default();
            let how = if probe { "passed, a probe with its dependencies met" } else { "passed after edits under its paths" };
            let t = format!("Neuromorph Nano step graph: step {} \"{title}\" is marked done ({} {how}).", c.id, field(a.rec, "cmd_class").or_else(|| field(a.rec, "kind")).unwrap_or("the check"));
            rows.push(row("step", &c.id, channel, &t));
            out.texts.push(t);
        } else {
            rows.push(row("step", &c.id, "none", &c.label));
        }
    }
    memo.save(a.memory, &session)?;
    if rows.is_empty() {
        return Ok(None);
    }
    if !block.is_empty() {
        // Whatever else is due rides in the reason, so it reaches the agent with the continuation.
        let mut parts = block;
        parts.extend(out.notice.iter().cloned());
        parts.append(&mut out.texts);
        for r in rows.iter_mut().filter(|r| text(r, "channel").as_deref() == Some(channel)) {
            if let Value::Object(m) = r {
                m.insert("channel".into(), s("block"));
            }
        }
        out.block = Some(parts.join("\n\n"));
    }
    let chars: usize = out.texts.iter().chain(out.block.iter()).map(|t| t.chars().count()).sum();
    out.record = obj(vec![
        ("reminders", strings(&given)),
        ("step", nudged.as_deref().map_or(Value::Null, s)),
        ("auto_done", strings(&auto)),
        ("loops", Value::Array(loop_rec)),
        ("chars", n(chars as u64)),
        ("channel", s(if out.block.is_some() { "block" } else { channel })),
    ]);
    let file = a.memory.join(DELIVERIES);
    for r in &rows {
        append_line(&file, r)?;
    }
    trim_deliveries(&file);
    // 2026-11-27 (C1): one live reading per delivery that reaches the agent (`GET /api/live`), with
    // fixed names and counts only. A loop's continuation carries whatever else was due.
    if out.block.is_some() || !out.texts.is_empty() {
        let (stage, label) = if out.block.is_some() {
            ("guidance.loop", "continuation")
        } else if nudged.is_some() || !auto.is_empty() {
            ("guidance.step", "nudge")
        } else {
            ("guidance.reminder", "reminder")
        };
        let steps = u64::from(nudged.is_some()) + if can_give { auto.len() as u64 } else { 0 };
        crate::live::delivered(a.home, stage, label, &[("reminders", given.len() as u64), ("steps", steps), ("loops", u64::from(out.block.is_some())), ("chars", chars as u64)]);
    }
    Ok(Some(out))
}

fn trim_deliveries(file: &Path) {
    if std::fs::metadata(file).map_or(0, |m| m.len()) <= TRIM_BYTES {
        return;
    }
    let all = std::fs::read_to_string(file).unwrap_or_default();
    let lines: Vec<&str> = all.lines().filter(|l| !l.trim().is_empty()).collect();
    let keep = lines.get(lines.len().saturating_sub(KEEP)..).unwrap_or_default();
    let tmp = file.with_extension(format!("jsonl.{}.tmp", std::process::id()));
    if std::fs::write(&tmp, format!("{}\n", keep.join("\n"))).is_ok() && std::fs::rename(&tmp, file).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// The hook's reply for Claude Code: a loop's continuation (`decision: block`, the prime riding in
/// the reason), else the prime and the guidance texts in one `additionalContext`, with a loop's end
/// also in `systemMessage`. `None` when there is nothing to say.
#[must_use]
pub fn reply(event: &str, prime: Option<&str>, out: Option<&Out>) -> Option<String> {
    if let Some(reason) = out.and_then(|o| o.block.as_ref()) {
        let reason = match prime {
            Some(p) => format!("{reason}\n\n{p}"),
            None => reason.clone(),
        };
        return obj(vec![("decision", s("block")), ("reason", s(&reason))]).encode().ok();
    }
    let mut parts: Vec<&str> = prime.into_iter().collect();
    if let Some(o) = out {
        parts.extend(o.notice.iter().map(String::as_str));
        parts.extend(o.texts.iter().map(String::as_str));
    }
    if parts.is_empty() {
        return None;
    }
    let inner = obj(vec![("hookEventName", s(event)), ("additionalContext", s(&parts.join("\n\n")))]);
    let mut top = vec![("hookSpecificOutput", inner)];
    if let Some(o) = out.filter(|o| !o.notice.is_empty()) {
        top.push(("systemMessage", s(&o.notice.join(" "))));
    }
    obj(top).encode().ok()
}

/// The text Antigravity is given (`injectSteps`): the prime, then the guidance texts.
#[must_use]
pub fn agy_text(prime: Option<&str>, out: Option<&Out>) -> Option<String> {
    let mut parts: Vec<&str> = prime.into_iter().collect();
    if let Some(o) = out {
        parts.extend(o.texts.iter().map(String::as_str));
    }
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

// ---------------------------------------------------------------------------------------------
// The view.

/// `GET /api/guidance?project=<key>`: reminders (with how often each was delivered), loops (state,
/// iterations, and whether the condition holds now in the newest session), steps (done, ready or
/// blocked), the settings in force, what each host can take, and the newest deliveries, each with
/// the evidence ledger's verdict (`verdict`, null for a delivery no rule judges: a loop's final
/// note, a step the Nano marked done, a stop not acted on) and the verdicts counted by kind.
#[must_use]
pub fn view(home: &Path, query: &str) -> Value {
    let shown = query.split('&').find_map(|kv| kv.strip_prefix("outcomes=")).and_then(|x| x.trim().parse::<usize>().ok()).map_or(OUTCOMES_SHOWN, |k| k.clamp(1, OUTCOMES_MAX));
    match crate::ledger::project_memory(home, query) {
        Some((key, memory)) => view_with(home, &key, &memory, shown),
        None => obj(vec![("error", s("project: give the key of a project as /api/projects lists it, or nano"))]),
    }
}

/// The outcome rows of a project's ended loops, newest first, at most `k`, and how many there are.
#[must_use]
pub fn outcomes(memory: &Path, k: usize) -> (Vec<Value>, u64) {
    let all = lines_of(&memory.join(OUTCOMES));
    let total = all.len() as u64;
    (all.into_iter().rev().take(k).collect(), total)
}

fn view_of(home: &Path, key: &str, memory: &Path) -> Value {
    view_with(home, key, memory, OUTCOMES_SHOWN)
}

fn view_with(home: &Path, key: &str, memory: &Path, shown_outcomes: usize) -> Value {
    let g = match load(memory) {
        Ok(g) => g,
        Err(e) => return obj(vec![("error", s(&e))]),
    };
    let controls = crate::controls::current(home);
    let setting = |k: &str| controls.get(k).cloned().unwrap_or_else(|| "on".into());
    let all = lines_of(&memory.join(DELIVERIES));
    let count = |kind: &str, id: &str, channel: Option<&str>| all.iter().filter(|r| text(r, "kind").as_deref() == Some(kind) && text(r, "id").as_deref() == Some(id) && channel.is_none_or(|c| text(r, "channel").as_deref() == Some(c))).count() as u64;
    let goals = objectives(memory);
    let needs_recs = !all.is_empty() || g.loops.iter().any(|l| l.state == "active" && l.until == "check-passes");
    let recs = if needs_recs { crate::sessions::records(memory) } else { Vec::new() };
    let judged = crate::ledger::evaluate_guidance(memory, &recs, nmtype::now_ms());
    let newest = crate::sessions::sessions(&recs).last().map(|x| x.0.clone()).unwrap_or_default();
    let records = || recs.clone();
    let reminders: Vec<Value> = g
        .reminders
        .iter()
        .map(|r| {
            let mut v = r.to_value();
            if let Value::Object(m) = &mut v {
                m.insert("delivered".into(), n(count("reminder", &r.id, None)));
            }
            v
        })
        .collect();
    let loops: Vec<Value> = g
        .loops
        .iter()
        .map(|l| {
            let mut v = l.to_value();
            if let Value::Object(m) = &mut v {
                if l.state == "active" {
                    let (met, detail) = evaluate(l, &g, &goals, &records, &newest);
                    m.insert("now".into(), obj(vec![("met", Value::Bool(met)), ("detail", s(&detail)), ("session", s(&newest))]));
                }
                m.insert("blocks".into(), n(count("loop", &l.id, Some("block"))));
                m.insert("not_acted_antigravity".into(), n(count("loop", &l.id, Some("none"))));
            }
            v
        })
        .collect();
    let steps: Vec<Value> = g
        .steps
        .iter()
        .map(|x| {
            let mut v = x.to_value();
            if let Value::Object(m) = &mut v {
                m.insert("status".into(), s(status_of(x, &g.steps)));
            }
            v
        })
        .collect();
    let by = |st: &str| g.steps.iter().filter(|x| status_of(x, &g.steps) == st).count() as u64;
    let shown: Vec<Value> = all.iter().rev().take(SHOWN).map(|r| with_verdict(r, &judged)).collect();
    let (ended, ended_total) = outcomes(memory, shown_outcomes);
    let words = |a: &[&str]| strings(&a.iter().map(|x| (*x).to_owned()).collect::<Vec<_>>());
    obj(vec![
        ("project", s(key)),
        ("file", s(&file_of(memory).display().to_string())),
        ("version", n(g.version)),
        ("settings", obj(vec![("reminders", s(&setting("reminders"))), ("loops", s(&setting("loops"))), ("memory", s(&setting("memory")))])),
        (
            "limits",
            obj(vec![
                ("loop_max_default", n(LOOP_MAX_DEFAULT)),
                ("loop_cap", n(LOOP_CAP)),
                ("text_chars", n(MAX_TEXT as u64)),
                ("stall_default", n(STALL_DEFAULT)),
                ("stall_max", n(STALL_MAX)),
                ("budget_ms_min", n(BUDGET_MS_MIN)),
                ("budget_ms_max", n(BUDGET_MS_MAX)),
                ("budget_tools_max", n(BUDGET_TOOLS_MAX)),
                ("parts_min", n(PARTS_MIN as u64)),
                ("parts_max", n(PARTS_MAX as u64)),
                ("error_count_min", n(ERROR_COUNT_MIN)),
                ("error_count_max", n(ERROR_COUNT_MAX)),
            ]),
        ),
        ("vocabulary", obj(vec![("triggers", words(TRIGGERS)), ("repeats", words(REPEATS)), ("conditions", words(CONDITIONS)), ("parts", words(PART_CONDITIONS)), ("loop_ends", words(LOOP_ENDS)), ("checks", words(CHECKS))])),
        (
            "hosts",
            Value::Array(vec![
                obj(vec![("host", s("claude-code")), ("reminders", s("every trigger")), ("loops", s("act at Stop through decision block, capped"))]),
                obj(vec![("host", s("antigravity")), ("reminders", s("session start and prompts only (the events that take injected steps)")), ("loops", s(AGY_LOOP))]),
            ]),
        ),
        ("reminders", Value::Array(reminders)),
        ("loops", Value::Array(loops)),
        ("steps", Value::Array(steps)),
        ("counts", obj(vec![("steps", n(g.steps.len() as u64)), ("done", n(by("done"))), ("ready", n(by("ready"))), ("blocked", n(by("blocked")))])),
        ("next_step", next_ready(&g.steps).map_or(Value::Null, |x| s(&x.id))),
        ("deliveries", Value::Array(shown)),
        ("delivered_total", n(all.len() as u64)),
        ("outcomes", Value::Array(ended)),
        ("outcomes_total", n(ended_total)),
        ("outcomes_file", s(&memory.join(OUTCOMES).display().to_string())),
        ("verdicts", obj(vec![("rules", strings(&crate::ledger::GUIDANCE_RULES.iter().map(|x| (*x).to_owned()).collect::<Vec<_>>())), ("judged", n(judged.len() as u64)), ("by_kind", crate::ledger::guidance_counts(&judged))])),
    ])
}

/// The channels on which a delivery reached the agent (a final note, a not-acted stop and a
/// recorded-only row are not judged).
const JUDGED_CHANNELS: &[&str] = &["context", "inject", "block"];
/// A deliveries row and its hook record are written in the same hook run; their times differ by
/// the time the hook took.
const MATCH_MS: u64 = 5000;

/// A deliveries row with the ledger's verdict of the delivery it records (`verdict`, null when no
/// rule judges it): the judged delivery of the same session, hook event, kind and item nearest in
/// time, within `MATCH_MS`.
fn with_verdict(row: &Value, judged: &[crate::ledger::GuideState]) -> Value {
    let field = |k: &str| text(row, k).unwrap_or_default();
    let t = int(row, "t").unwrap_or(0);
    let found = JUDGED_CHANNELS
        .contains(&field("channel").as_str())
        .then(|| {
            judged
                .iter()
                .filter(|g| g.d.kind == field("kind") && g.d.item == field("id") && g.d.pack.rec.session == field("session") && g.d.pack.rec.event == field("event") && g.d.pack.at.abs_diff(t) <= MATCH_MS)
                .min_by_key(|g| g.d.pack.at.abs_diff(t))
        })
        .flatten();
    let mut v = row.clone();
    if let Value::Object(m) = &mut v {
        m.insert("verdict".into(), found.map_or(Value::Null, crate::ledger::GuideState::to_value));
    }
    v
}

// ---------------------------------------------------------------------------------------------
// The command.

const USAGE: &str = "usage: neuromorph-nano guidance --project <repo or key> [view] [--json]
  add reminder --text T --trigger session-start|prompt|compact|stop|edit|failure|tools|error-repeats
               [--every N] [--glob G] [--repeat once|always|max] [--max N] [--files G]
               [--error CODE --count 2..5]   with error-repeats: the error and its repeats
  add loop --prompt T --until check-passes|goal-done|steps-done|manual|all-of|any-of
           [--parts c1,c2] [--goal N] [--max N] [--stall 1..10] [--budget-ms MS] [--budget-tools N]
  add step --title T [--id ID] [--after a,b] [--after-any c,d] [--goal N] [--check build|test|none]
           [--paths g1,g2] [--probe true|false]   a probe is done when its check passes, no edit
  edit <reminder|loop|step> <id> [the fields above]
  enable|disable <reminder|loop> <id>
  done <loop|step> <id>                 end a loop, or mark a step done
  reset <reminder|loop|step> <id>       count a reminder again, restart a loop, open a step again
  remove <reminder|loop|step> <id> [--confirm <id>]   a preview; with the id typed again, removes
  undo <audit id>                       as the Audit page";

fn arg<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2).find(|w| w.first().map(String::as_str) == Some(flag)).and_then(|w| w.get(1)).map(String::as_str)
}

/// The project key of `--project`: a folder is turned into its key (its memory folder made when
/// missing), anything else is taken as a key.
fn project_key(home: &Path, p: &str) -> Result<String, String> {
    let path = Path::new(p);
    if path.is_dir() {
        let root = nmsubs::repo::canon(path);
        let memory = nmtype::project::memory_dir(home, &root);
        std::fs::create_dir_all(&memory).map_err(|e| format!("{}: {e}", memory.display()))?;
        return Ok(nmtype::project::label(home, &root));
    }
    Ok(p.to_owned())
}

fn fields_of(rest: &[String], pairs: &mut Vec<(&'static str, Value)>) {
    for (k, f) in [("text", "--text"), ("trigger", "--trigger"), ("glob", "--glob"), ("repeat", "--repeat"), ("files", "--files"), ("prompt", "--prompt"), ("until", "--until"), ("title", "--title"), ("check", "--check"), ("error", "--error"), ("parts", "--parts")] {
        if let Some(x) = arg(rest, f) {
            pairs.push((k, s(x)));
        }
    }
    for (k, f) in [("every", "--every"), ("max", "--max"), ("goal", "--goal"), ("count", "--count"), ("stall", "--stall"), ("budget_ms", "--budget-ms"), ("budget_tools", "--budget-tools")] {
        if let Some(x) = arg(rest, f) {
            pairs.push((k, s(x)));
        }
    }
    if let Some(x) = arg(rest, "--after").or_else(|| arg(rest, "--depends-on")) {
        pairs.push(("depends_on", s(x)));
    }
    if let Some(x) = arg(rest, "--after-any").or_else(|| arg(rest, "--depends-any")) {
        pairs.push(("depends_any", s(x)));
    }
    if let Some(x) = arg(rest, "--probe") {
        pairs.push(("probe", s(x)));
    }
    if let Some(x) = arg(rest, "--paths") {
        pairs.push(("paths", s(x)));
    }
}

fn post(home: &Path, pairs: Vec<(&str, Value)>) -> Result<Value, String> {
    let body = obj(pairs).encode().map_err(|e| e.to_string())?.into_bytes();
    crate::ui::lent_route(home, "/api/guidance", "", Some(&body), "cli")
}

fn run_inner(home: &Path, args: &[String]) -> Result<Value, String> {
    let rest = args.get(1..).unwrap_or_default();
    let pos: Vec<&str> = {
        let mut out = Vec::new();
        let mut skip = false;
        for a in rest {
            if skip {
                skip = false;
            } else if a.starts_with("--") {
                skip = a != "--json";
            } else {
                out.push(a.as_str());
            }
        }
        out
    };
    if pos.first() == Some(&"undo") {
        let id = pos.get(1).ok_or(USAGE)?;
        let body = obj(vec![("id", s(id))]).encode().map_err(|e| e.to_string())?.into_bytes();
        return crate::ui::lent_route(home, "/api/undo", "", Some(&body), "cli");
    }
    let key = project_key(home, arg(rest, "--project").ok_or(USAGE)?)?;
    match pos.as_slice() {
        [] | ["view"] => crate::ui::lent_route(home, "/api/guidance", &format!("project={key}"), None, "cli"),
        ["add", kind] => {
            let mut p = vec![("project", s(&key)), ("action", s("add")), ("kind", s(kind))];
            if let Some(id) = arg(rest, "--id") {
                p.push(("id", s(id)));
            }
            fields_of(rest, &mut p);
            post(home, p)
        }
        ["remove", kind, id] => {
            let p = || vec![("project", s(&key)), ("action", s("remove")), ("kind", s(kind)), ("id", s(id))];
            let first = post(home, p())?;
            let Some(word) = arg(rest, "--confirm") else { return Ok(first) };
            if word != *id {
                return Err(format!("--confirm: type the id ({id}) to remove it; nothing was removed"));
            }
            let token = text(&first, "confirm_token").ok_or("the preview gave no confirmation token; nothing was removed")?;
            let mut q = p();
            q.push(("confirm_token", s(&token)));
            post(home, q)
        }
        [action @ ("edit" | "enable" | "disable" | "done" | "reset"), kind, id] => {
            let mut p = vec![("project", s(&key)), ("action", s(action)), ("kind", s(kind)), ("id", s(id))];
            fields_of(rest, &mut p);
            post(home, p)
        }
        _ => Err(USAGE.into()),
    }
}

/// `neuromorph-nano guidance ...`: runs the interface's own route (`ui::lent_route`), so the
/// command, MCP and the interface do the same thing.
pub fn run(home: &Path, args: &[String]) -> ExitCode {
    match run_inner(home, args) {
        Ok(v) => {
            if args.iter().any(|a| a == "--json") {
                println!("{}", v.encode().unwrap_or_default());
            } else {
                println!("{}", nmjson::pretty::pretty(&v));
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("guidance: {e}");
            ExitCode::FAILURE
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Section check.

/// Section check (a temporary folder, processor only): triggers and repetition, a loop that blocks
/// then exhausts at its cap and one that ends when its check passes, the block reply's shape, step
/// readiness, cycle refusal, a step marked done by a passing check, the storage round trip, and an
/// audited change undone.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let dir = std::env::temp_dir().join(format!("nano-guidance-check-{}-{}", std::process::id(), nmtype::now_ms()));
    let r = check_in(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    r
}

fn req(pairs: &[(&str, Value)]) -> Value {
    obj(pairs.iter().map(|(k, v)| (*k, v.clone())).collect())
}

fn rec_of(j: &str) -> Result<BTreeMap<String, Value>, String> {
    match nmjson::parse(j.as_bytes()).map_err(|e| e.to_string())? {
        Value::Object(m) => Ok(m),
        _ => Err("guidance: record".into()),
    }
}

fn check_in(dir: &Path) -> Result<(), String> {
    let e = |x: std::io::Error| x.to_string();
    let home = dir.join("home");
    let memory = home.join("memory");
    std::fs::create_dir_all(&memory).map_err(e)?;
    let user = Asker { who: "user", via: "interface", now: 1000 };
    let mut g = Guidance::default();
    // Storage round trip.
    apply(&mut g, &req(&[("action", s("add")), ("kind", s("reminder")), ("text", s("run the tests")), ("trigger", s("prompt")), ("every", n(2)), ("repeat", s("max")), ("max", n(2))]), user, &[])?;
    apply(&mut g, &req(&[("action", s("add")), ("kind", s("step")), ("title", s("write the parser")), ("id", s("parse"))]), user, &[])?;
    apply(&mut g, &req(&[("action", s("add")), ("kind", s("step")), ("title", s("test it")), ("depends_on", s("parse")), ("check", s("test")), ("paths", s("src/**"))]), user, &[])?;
    let mut saved = g.clone();
    save(&memory, &mut saved)?;
    let back = load(&memory)?;
    if back.reminders != g.reminders || back.steps != g.steps || back.version != 1 {
        return Err(format!("guidance: round trip {back:?}"));
    }
    // Readiness and cycles.
    let second = g.steps.get(1).map(|x| x.id.clone()).unwrap_or_default();
    let st = |g: &Guidance, id: &str| g.steps.iter().find(|x| x.id == id).cloned().unwrap_or_default();
    if status_of(&st(&g, "parse"), &g.steps) != "ready" || status_of(&st(&g, &second), &g.steps) != "blocked" || next_ready(&g.steps).map(|x| x.id.as_str()) != Some("parse") {
        return Err("guidance: readiness".into());
    }
    let cyc = apply(&mut g, &req(&[("action", s("edit")), ("kind", s("step")), ("id", s("parse")), ("depends_on", s(&second))]), user, &[]);
    if !cyc.as_ref().is_err_and(|m| m.contains("cycle")) || st(&g, "parse").depends_on.contains(&second) {
        return Err(format!("guidance: a cycle was not refused: {cyc:?}"));
    }
    // Triggers and repetition.
    let r = g.reminders.first().cloned().ok_or("guidance: reminder")?;
    let prompt = rec_of(r#"{"session":"s1","kind":"prompt"}"#)?;
    let mut memo = Memo { prompts: 1, ..Memo::default() };
    if fires(&r, "UserPromptSubmit", &prompt, &memo, &[]) {
        return Err("guidance: every 2nd prompt fired on the 1st".into());
    }
    memo.prompts = 2;
    if !fires(&r, "UserPromptSubmit", &prompt, &memo, &[]) || !repetition_allows(&r, &memo) {
        return Err("guidance: every 2nd prompt did not fire on the 2nd".into());
    }
    memo.delivered = vec![("r1".into(), 5), ("r1".into(), 6)];
    if repetition_allows(&r, &memo) {
        return Err("guidance: at most 2 per session allowed a 3rd".into());
    }
    let edit_r = Reminder { trigger: "edit".into(), glob: Some("*.rs".into()), ..r.clone() };
    let pre = rec_of(r#"{"session":"s1","tool":"Edit","kind":"edit"}"#)?;
    if !fires(&edit_r, "PreToolUse", &pre, &memo, &["crates/a/src/lib.rs".into()]) || fires(&edit_r, "PreToolUse", &pre, &memo, &["README.md".into()]) {
        return Err("guidance: edit glob".into());
    }
    if !glob_match("src/**", "src/a/b.rs") || glob_match("src/*", "src/a/b.rs") || !glob_match("**/lib.rs", "lib.rs") || !glob_match("docs/?.md", "docs/a.md") {
        return Err("guidance: glob".into());
    }
    // Loops at Stop: a manual loop blocks up to its cap, then is exhausted with a note.
    let mut lg = Guidance::default();
    apply(&mut lg, &req(&[("action", s("add")), ("kind", s("loop")), ("prompt", s("keep polishing.")), ("until", s("manual")), ("max", n(2))]), user, &[])?;
    save(&memory, &mut lg)?;
    let stop = rec_of(r#"{"session":"s1","kind":"stop","event":"Stop"}"#)?;
    let at = |rec: &BTreeMap<String, Value>, event: &'static str| at_hook(&At { home: &home, memory: &memory, event, rec, agy: false, inject: true, now: nmtype::now_ms() });
    for i in 1..=2 {
        let o = at(&stop, "Stop")?.ok_or("guidance: a loop gave nothing at Stop")?;
        let line = reply("Stop", None, Some(&o)).ok_or("guidance: no reply")?;
        let v = nmjson::parse(line.as_bytes()).map_err(|x| x.to_string())?;
        if text(&v, "decision").as_deref() != Some("block") || !text(&v, "reason").unwrap_or_default().contains(&format!("continuing, {i} of 2")) || v.get("hookSpecificOutput").is_ok() {
            return Err(format!("guidance: block reply {line}"));
        }
    }
    let o = at(&stop, "Stop")?.ok_or("guidance: no final note")?;
    let l = load(&memory)?.loops.first().cloned().ok_or("guidance: loop")?;
    if o.block.is_some() || l.state != "exhausted" || l.iterations != 2 || o.notice.is_empty() || reply("Stop", None, Some(&o)).is_some_and(|x| x.contains("\"decision\"")) {
        return Err(format!("guidance: the cap did not end the loop ({l:?}, {o:?})"));
    }
    if at(&stop, "Stop")?.is_some() {
        return Err("guidance: an exhausted loop acted again".into());
    }
    // A check-passes loop ends once a test passes after the last edit; a step with a test check is
    // then marked done by the Nano, with its evidence.
    let sit = crate::hook::situation_file(&memory);
    let now = nmtype::now_ms();
    let line = |t: u64, body: &str| format!("{{\"t\":{t},\"session\":\"s2\",{body}}}\n");
    std::fs::write(&sit, line(now + 1, r#""event":"PostToolUse","kind":"edit","tool":"Edit","ok":true,"paths":["src/lib.rs"]"#)).map_err(e)?;
    update(&memory, |gx| {
        gx.loops.clear();
        gx.steps = g.steps.clone();
        if let Some(p) = gx.steps.iter_mut().find(|x| x.id == "parse") {
            p.done = true;
        }
        for x in &mut gx.steps {
            x.updated_ms = 0;
        }
        apply(gx, &req(&[("action", s("add")), ("kind", s("loop")), ("prompt", s("fix and test.")), ("until", s("check-passes"))]), Asker { now: 0, ..user }, &[])
    })?;
    let stop2 = rec_of(r#"{"session":"s2","kind":"stop","event":"Stop"}"#)?;
    let o = at(&stop2, "Stop")?.ok_or("guidance: check-passes loop gave nothing")?;
    if !o.block.as_deref().is_some_and(|b| b.contains("no build or test ran after the last edit")) {
        return Err(format!("guidance: check-passes before a test {o:?}"));
    }
    let pass = rec_of(r#"{"session":"s2","event":"PostToolUse","kind":"test","tool":"Bash","ok":true,"cmd_class":"cargo-test","tests":[3,0]}"#)?;
    let mut text_file = std::fs::read_to_string(&sit).map_err(e)?;
    text_file.push_str(&line(now + 2, r#""event":"PostToolUse","kind":"test","tool":"Bash","ok":true,"cmd_class":"cargo-test""#));
    std::fs::write(&sit, text_file).map_err(e)?;
    let o = at(&pass, "PostToolUse")?.ok_or("guidance: a passing test marked nothing")?;
    let after = load(&memory)?;
    let s3 = after.steps.iter().find(|x| x.id == second).cloned().unwrap_or_default();
    if !s3.done || s3.done_by.as_deref() != Some("nano") || !s3.evidence.as_ref().is_some_and(|ev| text(ev, "check").as_deref() == Some("test")) || !o.record.get("auto_done").and_then(Value::array).is_ok_and(|a| a.len() == 1) {
        return Err(format!("guidance: auto-done {s3:?}"));
    }
    let o = at(&stop2, "Stop")?.ok_or("guidance: no loop end")?;
    if o.block.is_some() || load(&memory)?.loops.first().map(|l| l.state.as_str()) != Some("done") {
        return Err(format!("guidance: check-passes loop did not end ({o:?})"));
    }
    let rows = lines_of(&memory.join(OUTCOMES));
    if !rows.last().is_some_and(|r| text(r, "end").as_deref() == Some("met") && r.get("verdicts").ok().and_then(|v| text(v, "state")).as_deref() == Some("verified")) {
        return Err(format!("guidance: the met loop left no outcome row ({rows:?})"));
    }
    // Stall: a check-passes loop with stall 1 in a session with no check ends stalled at its 2nd
    // stop, says so once, and leaves an outcome row with its one distinct unmet reason.
    update(&memory, |gx| {
        gx.loops.clear();
        apply(gx, &req(&[("action", s("add")), ("kind", s("loop")), ("prompt", s("fix and test.")), ("until", s("check-passes")), ("stall", n(1))]), Asker { now: nmtype::now_ms(), ..user }, &[])
    })?;
    let stop3 = rec_of(r#"{"session":"s3","kind":"stop","event":"Stop"}"#)?;
    let first = at(&stop3, "Stop")?.ok_or("guidance: stall loop gave nothing")?;
    let o = at(&stop3, "Stop")?.ok_or("guidance: stall loop did not end")?;
    let l = load(&memory)?.loops.first().cloned().ok_or("guidance: loop")?;
    if first.block.is_none() || o.block.is_some() || l.state != "stalled" || !o.notice.iter().any(|t| t.contains("stalled")) || at(&stop3, "Stop")?.is_some_and(|x| x.notice.iter().any(|t| t.contains("stalled"))) {
        return Err(format!("guidance: stall ({l:?}, {o:?})"));
    }
    let row = lines_of(&memory.join(OUTCOMES)).pop().ok_or("guidance: no stall outcome")?;
    if text(&row, "end").as_deref() != Some("stalled") || row.get("unmet").and_then(Value::array).map(<[Value]>::len).ok() != Some(1) || int(&row, "iterations") != Some(1) {
        return Err(format!("guidance: stall outcome {row:?}"));
    }
    // Budget: a tool budget of 1, set before the two tool calls on record, ends the loop
    // budget-spent at its first stop.
    update(&memory, |gx| {
        gx.loops.clear();
        apply(gx, &req(&[("action", s("add")), ("kind", s("loop")), ("prompt", s("fix and test.")), ("until", s("manual")), ("budget_tools", n(1))]), Asker { now: 0, ..user }, &[])
    })?;
    let o = at(&stop3, "Stop")?.ok_or("guidance: budget loop gave nothing")?;
    let row = lines_of(&memory.join(OUTCOMES)).pop().ok_or("guidance: no budget outcome")?;
    if o.block.is_some() || load(&memory)?.loops.first().map(|l| l.state.as_str()) != Some("budget-spent") || text(&row, "end").as_deref() != Some("budget-spent") || row.get("budget").ok().and_then(|b| int(b, "used_tools")).is_none_or(|k| k == 0) {
        return Err(format!("guidance: budget ({o:?}, {row:?})"));
    }
    // An audited change through the route function, then undone.
    std::fs::create_dir_all(home.join("memory")).map_err(e)?;
    let w = write(&home, &req(&[("project", s("nano")), ("action", s("add")), ("kind", s("reminder")), ("text", s("mind the docs")), ("trigger", s("stop"))]), "mcp")?;
    let rid = text(&w, "id").ok_or("guidance: id")?;
    let audit = text(&w, "audit_id").ok_or("guidance: audit id")?;
    if !load(&memory)?.reminders.iter().any(|r| r.id == rid && r.by == "agent") {
        return Err("guidance: an agent's reminder was not kept as the agent's".into());
    }
    if write(&home, &req(&[("project", s("nano")), ("action", s("remove")), ("kind", s("reminder")), ("id", s(&rid))]), "interface").is_ok() {
        return Err("guidance: a remove went past the delete gate".into());
    }
    let row = lines_of(&home.join("memory/audit.jsonl")).into_iter().find(|r| text(r, "id").as_deref() == Some(&audit)).ok_or("guidance: audit row")?;
    if row.get("text").is_ok() || text(&row, "kind").as_deref() != Some("guidance") {
        return Err("guidance: audit row shape".into());
    }
    undo(&home, &row, &audit, "interface")?;
    if load(&memory)?.reminders.iter().any(|r| r.id == rid) {
        return Err("guidance: undo did not take the added reminder out".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user() -> Asker<'static> {
        Asker { who: "user", via: "interface", now: 10 }
    }

    #[test]
    fn section_check() {
        check().unwrap();
    }

    #[test]
    fn triggers_match_their_events() {
        let r = |trigger: &str| Reminder { id: "r1".into(), trigger: trigger.into(), every: 1, repeat: "always".into(), enabled: true, ..Reminder::default() };
        let memo = Memo { prompts: 1, tools: 3, ..Memo::default() };
        let start = rec_of(r#"{"source":"startup"}"#).unwrap();
        let compact = rec_of(r#"{"source":"compact"}"#).unwrap();
        let fail = rec_of(r#"{"kind":"test","ok":false}"#).unwrap();
        let pass = rec_of(r#"{"kind":"test","ok":true}"#).unwrap();
        assert!(fires(&r("session-start"), "SessionStart", &start, &memo, &[]));
        assert!(!fires(&r("session-start"), "SessionStart", &compact, &memo, &[]));
        assert!(fires(&r("compact"), "SessionStart", &compact, &memo, &[]));
        assert!(fires(&r("stop"), "Stop", &start, &memo, &[]));
        assert!(fires(&r("failure"), "PostToolUseFailure", &fail, &memo, &[]));
        assert!(!fires(&r("failure"), "PostToolUse", &pass, &memo, &[]));
        let every3 = Reminder { every: 3, ..r("tools") };
        assert!(fires(&every3, "PostToolUse", &pass, &memo, &[]));
        assert!(!fires(&every3, "PostToolUse", &pass, &Memo { tools: 4, ..memo.clone() }, &[]));
        let once = Reminder { repeat: "once".into(), ..r("prompt") };
        assert!(repetition_allows(&once, &memo));
        assert!(!repetition_allows(&once, &Memo { delivered: vec![("r1".into(), 1)], ..memo.clone() }));
        // A reset counts again from its time.
        assert!(repetition_allows(&Reminder { reset_ms: 5, ..once }, &Memo { delivered: vec![("r1".into(), 1)], ..memo.clone() }));
        let scoped = Reminder { files: Some("docs/**".into()), ..r("prompt") };
        assert!(!in_scope(&scoped, &[], &memo));
        assert!(in_scope(&scoped, &[], &Memo { touched: vec!["docs/a.md".into()], ..memo }));
    }

    #[test]
    fn fields_are_checked() {
        let mut g = Guidance::default();
        let bad = |g: &mut Guidance, pairs: &[(&str, Value)]| apply(g, &req(pairs), user(), &[(1, "x".into(), "open".into())]).unwrap_err();
        assert!(bad(&mut g, &[("action", s("add")), ("kind", s("reminder")), ("trigger", s("prompt"))]).contains("text"));
        assert!(bad(&mut g, &[("action", s("add")), ("kind", s("reminder")), ("text", s("x")), ("trigger", s("lunch"))]).contains("trigger"));
        assert!(bad(&mut g, &[("action", s("add")), ("kind", s("reminder")), ("text", s("x")), ("trigger", s("stop")), ("repeat", s("max"))]).contains("max"));
        assert!(bad(&mut g, &[("action", s("add")), ("kind", s("loop")), ("prompt", s("x")), ("until", s("manual")), ("max", n(21))]).contains("1 to 20"));
        assert!(bad(&mut g, &[("action", s("add")), ("kind", s("loop")), ("prompt", s("x")), ("until", s("goal-done"))]).contains("goal"));
        assert!(bad(&mut g, &[("action", s("add")), ("kind", s("loop")), ("prompt", s("x")), ("until", s("goal-done")), ("goal", n(9))]).contains("no such objective"));
        assert!(bad(&mut g, &[("action", s("add")), ("kind", s("step")), ("title", s("x")), ("depends_on", s("nope"))]).contains("not a step"));
        assert!(bad(&mut g, &[("action", s("add")), ("kind", s("step")), ("title", s("x")), ("id", s("a b"))]).contains("id"));
        let l = apply(&mut g, &req(&[("action", s("add")), ("kind", s("loop")), ("prompt", s("x")), ("until", s("goal-done")), ("goal", n(1))]), user(), &[(1, "x".into(), "open".into())]).unwrap();
        assert_eq!(g.loops[0].max, LOOP_MAX_DEFAULT);
        assert!(bad(&mut g, &[("action", s("enable")), ("kind", s("loop")), ("id", s(&l.id))]).contains("already on"));
        apply(&mut g, &req(&[("action", s("add")), ("kind", s("step")), ("title", s("a")), ("id", s("a"))]), user(), &[]).unwrap();
        apply(&mut g, &req(&[("action", s("add")), ("kind", s("step")), ("title", s("b")), ("depends_on", Value::Array(vec![s("a")]))]), user(), &[]).unwrap();
        assert!(bad(&mut g, &[("action", s("remove")), ("kind", s("step")), ("id", s("a"))]).contains("dependency"));
    }

    #[test]
    fn loop_conditions() {
        let rec = |t: u64, kind: &str, ok: bool| Rec { t, i: t as usize, event: "PostToolUse".into(), kind: kind.into(), session: "s".into(), ok: Some(ok), ..Rec::default() };
        assert!(!check_passes(&[], "s", 0).0);
        assert!(!check_passes(&[rec(1, "edit", true)], "s", 0).0);
        assert!(!check_passes(&[rec(1, "edit", true), rec(2, "test", false)], "s", 0).0);
        assert!(check_passes(&[rec(1, "edit", true), rec(2, "test", false), rec(3, "test", true)], "s", 0).0);
        assert!(!check_passes(&[rec(1, "test", true), rec(2, "edit", true)], "s", 0).0, "a pass before the last edit does not count");
        assert!(!check_passes(&[rec(1, "edit", true), rec(2, "test", true)], "s", 5).0, "records before the loop do not count");
        let g = Guidance { steps: vec![Step { id: "a".into(), done: true, ..Step::default() }, Step { id: "b".into(), depends_on: vec!["a".into()], ..Step::default() }], ..Guidance::default() };
        let l = |until: &str, goal: Option<u64>| Loop { until: until.into(), goal, ..Loop::default() };
        let none = || Vec::new();
        let goals = [(1, "x".to_owned(), "done".to_owned()), (2, "y".to_owned(), "open".to_owned())];
        assert!(evaluate(&l("goal-done", Some(1)), &g, &goals, &none, "s").0);
        assert!(!evaluate(&l("goal-done", Some(2)), &g, &goals, &none, "s").0);
        let (met, why) = evaluate(&l("steps-done", None), &g, &goals, &none, "s");
        assert!(!met && why.contains("b (ready)"), "{why}");
        assert!(!evaluate(&l("manual", None), &g, &goals, &none, "s").0);
    }

    #[test]
    fn loop_bounds_and_combined_conditions() {
        let mut g = Guidance::default();
        let goals = [(1, "x".to_owned(), "open".to_owned())];
        let add = |g: &mut Guidance, pairs: &[(&str, Value)]| apply(g, &req(pairs), user(), &goals);
        let base = |extra: &[(&'static str, Value)]| -> Vec<(&'static str, Value)> {
            let mut p = vec![("action", s("add")), ("kind", s("loop")), ("prompt", s("x"))];
            p.extend(extra.iter().cloned());
            p
        };
        assert!(add(&mut g, &base(&[("until", s("check-passes")), ("stall", n(11))])).unwrap_err().contains("1 to 10"));
        assert!(add(&mut g, &base(&[("until", s("check-passes")), ("budget_ms", n(10))])).unwrap_err().contains("budget_ms"));
        assert!(add(&mut g, &base(&[("until", s("check-passes")), ("budget_tools", n(BUDGET_TOOLS_MAX + 1))])).unwrap_err().contains("budget_tools"));
        assert!(add(&mut g, &base(&[("until", s("all-of")), ("parts", s("check-passes"))])).unwrap_err().contains("2 to 4"));
        assert!(add(&mut g, &base(&[("until", s("any-of")), ("parts", s("check-passes,lunch"))])).unwrap_err().contains("not a condition"));
        assert!(add(&mut g, &base(&[("until", s("all-of")), ("parts", s("check-passes,check-passes"))])).unwrap_err().contains("once"));
        assert!(add(&mut g, &base(&[("all_of", Value::Array(vec![s("check-passes"), s("goal-done")]))])).unwrap_err().contains("goal"));
        let c = add(&mut g, &base(&[("all_of", Value::Array(vec![s("steps_done"), s("goal-done")])), ("goal", n(1)), ("budget_ms", n(60_000))])).unwrap();
        let l = g.loops.iter().find(|x| x.id == c.id).cloned().unwrap();
        assert_eq!((l.until.as_str(), l.parts.clone(), l.stall, l.budget_ms, l.budget_tools), ("all-of", vec!["steps-done".to_owned(), "goal-done".to_owned()], STALL_DEFAULT, Some(60_000), None));
        // Clearing a budget with 0, and the loop's fields surviving the round trip.
        apply(&mut g, &req(&[("action", s("edit")), ("kind", s("loop")), ("id", s(&c.id)), ("budget_ms", n(0)), ("stall", n(4))]), user(), &goals).unwrap();
        let l = g.loops.iter().find(|x| x.id == c.id).cloned().unwrap();
        assert_eq!((l.budget_ms, l.stall), (None, 4));
        assert_eq!(Loop::from_value(&l.to_value()), Some(l.clone()));
        // Evaluation: all-of names the parts not met; any-of is met by one.
        let steps = Guidance { steps: vec![Step { id: "a".into(), done: true, goal: Some(1), ..Step::default() }], ..Guidance::default() };
        let none = || Vec::new();
        let (met, why) = evaluate(&l, &steps, &goals, &none, "s");
        assert!(!met && why.contains("goal-done (goal 1 is still open)") && !why.contains("steps-done"), "{why}");
        let any = Loop { until: "any-of".into(), ..l.clone() };
        let (met, why) = evaluate(&any, &steps, &goals, &none, "s");
        assert!(met && why.contains("met by steps-done"), "{why}");
        let (met, why) = evaluate(&any, &Guidance::default(), &goals, &none, "s");
        assert!(!met && why.contains("none met") && why.contains("steps-done (no steps yet)"), "{why}");
    }

    #[test]
    fn stall_and_budget_bookkeeping() {
        let mut l = Loop { stall: 2, ..Loop::default() };
        note_unmet(&mut l, "a");
        note_unmet(&mut l, "a");
        assert_eq!((l.repeats, l.unmet_seq.len()), (1, 1));
        note_unmet(&mut l, "b");
        note_unmet(&mut l, "b");
        note_unmet(&mut l, "b");
        assert_eq!((l.repeats, l.unmet_seq.clone()), (2, vec!["a".to_owned(), "b".to_owned()]));
        for i in 0..20 {
            note_unmet(&mut l, &format!("r{i}"));
        }
        assert_eq!(l.unmet_seq.len(), UNMET_KEEP);
        assert!(budget_spent(&Loop { budget_tools: Some(3), ..Loop::default() }, 0, 3).is_some_and(|w| w.contains("tool budget")));
        assert!(budget_spent(&Loop { budget_ms: Some(5000), ..Loop::default() }, 4999, 100).is_none());
        assert!(budget_spent(&Loop { budget_ms: Some(5000), ..Loop::default() }, 5000, 0).is_some_and(|w| w.contains("time budget")));
        // An outcome row for an end, and none for a change that leaves the loop active.
        let active = Loop { id: "l1".into(), state: "active".into(), since_ms: 100, iterations: 2, unmet_seq: vec!["a".into()], ..Loop::default() };
        let ended = Loop { state: "ended".into(), ended_ms: Some(400), ..active.clone() };
        let ch = |before: &Loop, after: Option<&Loop>| Change { action: "x".into(), kind: "loop".into(), id: "l1".into(), before: Some(before.to_value()), after: after.map(Loop::to_value), label: String::new() };
        let recs = vec![Rec { t: 150, event: "PostToolUse".into(), ..Rec::default() }, Rec { t: 500, event: "PostToolUse".into(), ..Rec::default() }];
        let row = outcome_of(&ch(&active, Some(&ended)), 999, &recs).unwrap();
        assert_eq!((text(&row, "end").as_deref(), int(&row, "ended_ms"), row.get("budget").ok().and_then(|b| int(b, "used_tools"))), (Some("ended"), Some(400), Some(1)));
        assert_eq!(text(row.get("verdicts").unwrap(), "state").as_deref(), Some("unknown"));
        assert_eq!(outcome_of(&ch(&active, None), 999, &recs).and_then(|r| text(&r, "end")).as_deref(), Some("removed"));
        assert!(outcome_of(&ch(&active, Some(&active)), 999, &recs).is_none());
        assert!(outcome_of(&ch(&ended, None), 999, &recs).is_none(), "removing an ended loop adds no second row");
    }

    #[test]
    fn or_joins_and_probe_steps() {
        let mut g = Guidance::default();
        let add = |g: &mut Guidance, pairs: &[(&'static str, Value)]| {
            let mut p = vec![("action", s("add")), ("kind", s("step"))];
            p.extend(pairs.iter().cloned());
            apply(g, &req(&p), user(), &[])
        };
        add(&mut g, &[("id", s("a")), ("title", s("a"))]).unwrap();
        add(&mut g, &[("id", s("b")), ("title", s("b"))]).unwrap();
        add(&mut g, &[("id", s("c")), ("title", s("c")), ("depends_any", s("a,b"))]).unwrap();
        add(&mut g, &[("id", s("d")), ("title", s("d")), ("depends", s("c")), ("depends_any", s("a,b"))]).unwrap();
        let status = |g: &Guidance, id: &str| status_of(g.steps.iter().find(|x| x.id == id).unwrap(), &g.steps);
        assert_eq!((status(&g, "c"), status(&g, "d")), ("blocked", "blocked"));
        apply(&mut g, &req(&[("action", s("done")), ("kind", s("step")), ("id", s("b"))]), user(), &[]).unwrap();
        assert_eq!((status(&g, "c"), status(&g, "d")), ("ready", "blocked"), "one of the OR-join is enough; d still waits for c");
        // A cycle through an OR edge, a step in both lists, a probe with no check, an unknown OR
        // dependency, and removing a step an OR-join names are refused.
        let edit = |g: &mut Guidance, pairs: &[(&'static str, Value)]| {
            let mut p = vec![("action", s("edit")), ("kind", s("step"))];
            p.extend(pairs.iter().cloned());
            apply(g, &req(&p), user(), &[])
        };
        assert!(edit(&mut g, &[("id", s("a")), ("depends_any", s("d,b"))]).unwrap_err().contains("cycle"));
        assert!(edit(&mut g, &[("id", s("d")), ("depends_on", s("c,a"))]).unwrap_err().contains("both"));
        assert!(add(&mut g, &[("title", s("p")), ("probe", Value::Bool(true))]).unwrap_err().contains("probe needs a check"));
        assert!(add(&mut g, &[("title", s("p")), ("depends_any", s("zz"))]).unwrap_err().contains("not a step"));
        assert!(apply(&mut g, &req(&[("action", s("remove")), ("kind", s("step")), ("id", s("a"))]), user(), &[]).unwrap_err().contains("dependency"));
        assert!(add(&mut g, &[("title", s("p")), ("probe", s("maybe"))]).unwrap_err().contains("probe"));
        let p = add(&mut g, &[("id", s("p")), ("title", s("p")), ("depends_any", s("a,b")), ("check", s("test")), ("probe", s("true"))]).unwrap();
        let step = g.steps.iter().find(|x| x.id == p.id).cloned().unwrap();
        assert!(step.probe && step.depends_any == ["a", "b"]);
        assert_eq!(Step::from_value(&step.to_value()), Some(step.clone()));
        assert!(step_text(&step, &g).contains("one of a, b") && step_text(&step, &g).contains("probe"));
        // A passing test marks the ready probe done with no edit; a check step with no edit stays
        // open; a probe whose dependency the same check marks done waits for a later check.
        let mut h = Guidance {
            steps: vec![
                Step { id: "base".into(), done: true, ..Step::default() },
                Step { id: "p1".into(), depends_on: vec!["base".into()], check: Some("test".into()), probe: true, ..Step::default() },
                Step { id: "q".into(), check: Some("test".into()), paths: vec!["src/**".into()], ..Step::default() },
                Step { id: "a2".into(), check: Some("test".into()), ..Step::default() },
                Step { id: "p2".into(), depends_on: vec!["a2".into()], check: Some("test".into()), probe: true, ..Step::default() },
            ],
            ..Guidance::default()
        };
        let pass = rec_of(r#"{"kind":"test","ok":true,"cmd_class":"cargo-test"}"#).unwrap();
        let recs = vec![Rec { t: 5, session: "s".into(), event: "PostToolUse".into(), kind: "edit".into(), ok: Some(true), paths: vec!["docs/a.md".into()], ..Rec::default() }];
        let done: Vec<String> = auto_done(&mut h, &pass, &recs, "s", 10).into_iter().map(|c| c.id).collect();
        assert_eq!(done, ["p1", "a2"]);
        let p1 = h.steps.iter().find(|x| x.id == "p1").unwrap();
        assert!(p1.evidence.as_ref().is_some_and(|e| matches!(e.get("probe"), Ok(Value::Bool(true)))));
        let again: Vec<String> = auto_done(&mut h, &pass, &recs, "s", 20).into_iter().map(|c| c.id).collect();
        assert_eq!(again, ["p2"]);
    }

    #[test]
    fn error_repeats_reminder() {
        let mut g = Guidance::default();
        let add = |g: &mut Guidance, pairs: &[(&str, Value)]| apply(g, &req(pairs), user(), &[]);
        assert!(add(&mut g, &[("action", s("add")), ("kind", s("reminder")), ("text", s("x")), ("trigger", s("error-repeats"))]).unwrap_err().contains("error"));
        assert!(add(&mut g, &[("action", s("add")), ("kind", s("reminder")), ("text", s("x")), ("trigger", s("error-repeats")), ("error", s("E0308")), ("count", n(6))]).unwrap_err().contains("2 to 5"));
        add(&mut g, &[("action", s("add")), ("kind", s("reminder")), ("text", s("read the types first")), ("trigger", s("error-repeats")), ("error", s("e0308")), ("count", n(2))]).unwrap();
        let r = g.reminders[0].clone();
        assert_eq!((r.error.as_deref(), r.count), (Some("e0308"), 2));
        let fail = rec_of(r#"{"kind":"test","ok":false,"errors":[{"code":"E0308","file":"a.rs","line":5,"msg":"mismatched types"}]}"#).unwrap();
        let other = rec_of(r#"{"kind":"build","ok":false,"errors":[{"code":"E0425","file":"a.rs","line":9}]}"#).unwrap();
        let tool = rec_of(r#"{"kind":"read","ok":false,"msg":"File does not exist: mismatched TYPES"}"#).unwrap();
        assert!(error_in(&fail, "E0308") && error_in(&fail, "mismatched") && !error_in(&other, "E0308") && error_in(&tool, "mismatched types"));
        let mut memo = Memo::default();
        memo.hit(&r.id);
        assert!(!fires(&r, "PostToolUseFailure", &fail, &memo, &[]), "one result with the error is not a repeat");
        memo.hit(&r.id);
        assert!(fires(&r, "PostToolUseFailure", &fail, &memo, &[]));
        assert!(!fires(&r, "PostToolUseFailure", &other, &memo, &[]), "the event itself must carry the error");
        assert!(!fires(&r, "Stop", &fail, &memo, &[]));
        memo.clear_hits(&r.id);
        assert_eq!(memo.hits(&r.id), 0);
        // Another trigger drops the error fields.
        apply(&mut g, &req(&[("action", s("edit")), ("kind", s("reminder")), ("id", s(&r.id)), ("trigger", s("stop"))]), user(), &[]).unwrap();
        assert!(g.reminders[0].error.is_none());
    }

    #[test]
    fn replies_carry_the_prime_and_the_reminders() {
        let o = Out { texts: vec!["Neuromorph Nano reminder (set by the user): x".into()], ..Out::default() };
        let line = reply("UserPromptSubmit", Some("prime"), Some(&o)).unwrap();
        let v = nmjson::parse(line.as_bytes()).unwrap();
        let ctx = v.get("hookSpecificOutput").and_then(|h| h.get("additionalContext")).and_then(Value::string).unwrap().to_owned();
        assert_eq!(ctx, "prime\n\nNeuromorph Nano reminder (set by the user): x");
        assert!(reply("Stop", None, None).is_none());
        let b = Out { block: Some("loop".into()), ..Out::default() };
        let v = nmjson::parse(reply("Stop", Some("p"), Some(&b)).unwrap().as_bytes()).unwrap();
        assert_eq!(text(&v, "decision").as_deref(), Some("block"));
        assert_eq!(text(&v, "reason").as_deref(), Some("loop\n\np"));
        assert_eq!(agy_text(Some("p"), Some(&o)).as_deref(), Some("p\n\nNeuromorph Nano reminder (set by the user): x"));
    }

    #[test]
    fn undo_puts_a_removed_step_back() {
        let mut g = Guidance::default();
        let c = apply(&mut g, &req(&[("action", s("add")), ("kind", s("step")), ("title", s("a"))]), user(), &[]).unwrap();
        let r = apply(&mut g, &req(&[("action", s("remove")), ("kind", s("step")), ("id", s(&c.id))]), user(), &[]).unwrap();
        assert!(g.steps.is_empty());
        g.set_item("step", &c.id, r.before.as_ref()).unwrap();
        assert_eq!(g.steps.len(), 1);
        // A restore that would leave a dangling dependency is refused.
        let mut h = Guidance { steps: vec![Step { id: "b".into(), depends_on: vec!["a".into()], ..Step::default() }], ..Guidance::default() };
        assert!(h.set_item("step", "b", Some(&Step { id: "b".into(), depends_on: vec!["zz".into()], ..Step::default() }.to_value())).is_err());
    }

    #[test]
    fn deliveries_carry_the_ledgers_verdict() {
        let line = r#"{"t":10,"event":"UserPromptSubmit","kind":"prompt","session":"s1","guidance":{"step":"a","channel":"context"}}"#;
        let recs = vec![Rec::from_value(0, &nmjson::parse(line.as_bytes()).unwrap())];
        let g = Guidance { steps: vec![Step { id: "a".into(), ..Step::default() }], ..Guidance::default() };
        let judged: Vec<crate::ledger::GuideState> = crate::ledger::deliveries(&recs)
            .into_iter()
            .map(|d| {
                let (closed, ev) = crate::ledger::judge_delivery(&recs, &d, Some(&g), 20);
                crate::ledger::GuideState { d, closed, ev }
            })
            .collect();
        let row = |t: u64, channel: &str, event: &str| obj(vec![("t", n(t)), ("session", s("s1")), ("event", s(event)), ("kind", s("step")), ("id", s("a")), ("channel", s(channel)), ("text", s("x"))]);
        let v = with_verdict(&row(12, "context", "UserPromptSubmit"), &judged);
        let verdict = v.get("verdict").unwrap();
        assert_eq!(text(verdict, "delivery").as_deref(), Some("guide:step:a:s1:10"));
        assert_eq!(verdict.get("verdict").ok().and_then(|x| text(x, "rule")).as_deref(), Some("step-followed"));
        assert_eq!(text(verdict, "window").as_deref(), Some("open"));
        // A final note, another hook event and a row far in time carry no verdict.
        for r in [row(12, "note", "UserPromptSubmit"), row(12, "context", "PostToolUse"), row(10 + MATCH_MS + 1, "context", "UserPromptSubmit")] {
            assert!(matches!(with_verdict(&r, &judged).get("verdict"), Ok(Value::Null)), "{r:?}");
        }
    }
}
