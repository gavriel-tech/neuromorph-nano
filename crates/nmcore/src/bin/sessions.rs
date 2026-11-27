//! Session state: a per-session view of the situation record, built from the hook records of one
//! Claude Code session. It holds the task in play (the newest prompt, its kept text when the
//! `prompts` control keeps it), the files read and edited with their line ranges, the commands and
//! checks with pass or fail, what is still open (failing checks, files changed and not checked
//! since), and what the Nano served in the session and when. It is computed from the records each
//! time it is asked for, so it is current as soon as a hook has appended its line; nothing of it is
//! stored apart from the situation record itself.
//!
//! The `session` command prints it plainly; `/api/session` serves it to the interface.
use nmjson::Value;
use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;

/// One hook record, typed. `i` is its place in the file, kept so records with the same time keep
/// their file order.
#[derive(Clone, Debug, Default)]
pub struct Rec {
    pub i: usize,
    pub t: u64,
    pub event: String,
    pub kind: String,
    pub session: String,
    pub tool: String,
    pub use_id: Option<String>,
    pub source: Option<String>,
    pub model: Option<String>,
    pub paths: Vec<String>,
    pub lines: Vec<(String, u64, u64)>,
    pub ok: Option<bool>,
    pub exit: Option<i64>,
    pub tests: Option<(u64, u64)>,
    pub cmd: Option<String>,
    pub cmd_class: Option<String>,
    pub pkg: Option<String>,
    pub errors: Vec<(String, String, u64)>,
    pub msg: Option<String>,
    /// A prompt: its hash, its length in characters, whether its text was kept.
    pub prompt: Option<(String, u64, bool)>,
    /// A call of one of the Nano's tools: the tool, its question, the episode of its reply.
    pub nano: Option<(String, Option<String>, Option<String>)>,
    pub sensitive: bool,
    /// The message of the first located error, as the hook kept it (secrets dropped, 120 characters).
    pub error_msg: Option<String>,
    /// A prime the Nano gave at this hook.
    pub prime: Option<PrimeRec>,
    /// The agent's guidance delivered at this hook (`guidance.rs`).
    pub guidance: Option<GuidanceRec>,
}

/// The agent's guidance as its hook record keeps it: the reminders delivered, the step nudged,
/// the steps the Nano marked done, and each loop's action (id, `block`, `done`, `exhausted`, `stalled`, `budget-spent` or
/// `not-acted`, iteration, maximum).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GuidanceRec {
    pub reminders: Vec<String>,
    pub step: Option<String>,
    pub auto_done: Vec<String>,
    pub loops: Vec<(String, String, u64, u64)>,
    pub channel: String,
}

/// A prime as its hook record keeps it: kind, id, the places it cited, the memories it came from,
/// and the `priming` setting it was given under.
#[derive(Clone, Debug, Default)]
pub struct PrimeRec {
    pub kind: String,
    pub id: String,
    pub places: Vec<String>,
    pub from: Vec<String>,
    pub mode: String,
}

fn text(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::string).ok().map(str::to_owned)
}
fn int(v: &Value, k: &str) -> Option<u64> {
    v.get(k).and_then(Value::integer).ok()
}

impl Rec {
    /// The typed form of one record line.
    #[must_use]
    pub fn from_value(i: usize, v: &Value) -> Self {
        let lines = v
            .get("lines")
            .and_then(Value::array)
            .unwrap_or_default()
            .iter()
            .filter_map(|l| {
                let a = l.array().ok()?;
                Some((a.first()?.string().ok()?.to_owned(), a.get(1)?.integer().ok()?, a.get(2)?.integer().ok()?))
            })
            .collect();
        let pair = |k: &str| {
            let a = v.get(k).and_then(Value::array).ok()?;
            Some((a.first()?.integer().ok()?, a.get(1)?.integer().ok()?))
        };
        let errors = v
            .get("errors")
            .and_then(Value::array)
            .unwrap_or_default()
            .iter()
            .map(|e| (text(e, "code").unwrap_or_default(), text(e, "file").unwrap_or_default(), int(e, "line").unwrap_or(0)))
            .collect();
        Self {
            i,
            t: int(v, "t").unwrap_or(0),
            event: text(v, "event").unwrap_or_default(),
            kind: text(v, "kind").unwrap_or_default(),
            session: text(v, "session").unwrap_or_default(),
            tool: text(v, "tool").unwrap_or_default(),
            use_id: text(v, "use"),
            source: text(v, "source"),
            model: text(v, "model"),
            paths: v.get("paths").and_then(Value::array).unwrap_or_default().iter().filter_map(|p| p.string().ok().map(str::to_owned)).collect(),
            lines,
            ok: match v.get("ok") {
                Ok(Value::Bool(b)) => Some(*b),
                _ => None,
            },
            exit: v.get("exit").ok().and_then(|x| match x {
                Value::Number(n) => n.parse().ok(),
                _ => None,
            }),
            tests: pair("tests"),
            cmd: text(v, "cmd"),
            cmd_class: text(v, "cmd_class"),
            pkg: text(v, "pkg"),
            errors,
            msg: text(v, "msg"),
            prompt: v.get("prompt").ok().map(|p| (text(p, "sha").unwrap_or_default(), int(p, "chars").unwrap_or(0), matches!(p.get("kept"), Ok(Value::Bool(true))))),
            nano: v.get("nano").ok().map(|n| (text(n, "tool").unwrap_or_default(), text(n, "ask"), text(n, "episode"))),
            sensitive: matches!(v.get("sensitive"), Ok(Value::Bool(true))),
            error_msg: v.get("errors").and_then(Value::array).ok().and_then(|e| e.first()).and_then(|e| text(e, "msg")),
            // Records written before the rename keep the prime under `nudge`.
            prime: v.get("prime").or_else(|_| v.get("nudge")).ok().and_then(|x| {
                let list = |k: &str| x.get(k).and_then(Value::array).unwrap_or_default().iter().filter_map(|p| p.string().ok().map(str::to_owned)).collect();
                Some(PrimeRec { kind: text(x, "kind")?, id: text(x, "id")?, places: list("places"), from: list("from"), mode: text(x, "mode").unwrap_or_default() })
            }),
            guidance: v.get("guidance").ok().filter(|x| matches!(x, Value::Object(_))).map(|x| {
                let list = |k: &str| x.get(k).and_then(Value::array).unwrap_or_default().iter().filter_map(|p| p.string().ok().map(str::to_owned)).collect();
                GuidanceRec {
                    reminders: list("reminders"),
                    step: text(x, "step"),
                    auto_done: list("auto_done"),
                    loops: x
                        .get("loops")
                        .and_then(Value::array)
                        .unwrap_or_default()
                        .iter()
                        .filter_map(|l| Some((text(l, "id")?, text(l, "action")?, int(l, "iteration").unwrap_or(0), int(l, "max").unwrap_or(0))))
                        .collect(),
                    channel: text(x, "channel").unwrap_or_default(),
                }
            }),
        }
    }
    /// A completed tool call (its after-the-call hook), as opposed to its before-the-call hook.
    #[must_use]
    pub fn done(&self) -> bool {
        matches!(self.event.as_str(), "PostToolUse" | "PostToolUseFailure")
    }
    /// The lines this record names in `path`, if any.
    #[must_use]
    pub fn span_of(&self, path: &str) -> Option<(u64, u64)> {
        self.lines.iter().find(|l| l.0 == path).map(|l| (l.1, l.2))
    }
}

/// Every record of a project memory folder, in time order (file order within one millisecond).
#[must_use]
pub fn records(memory: &Path) -> Vec<Rec> {
    let mut v: Vec<Rec> = crate::situation::read(memory, None, None).iter().enumerate().map(|(i, r)| Rec::from_value(i, r)).collect();
    v.sort_by_key(|r| (r.t, r.i));
    v
}

/// The sessions of a set of records: id, first and last time, record count; newest last.
#[must_use]
pub fn sessions(recs: &[Rec]) -> Vec<(String, u64, u64, usize)> {
    let mut m: BTreeMap<String, (u64, u64, usize)> = BTreeMap::new();
    for r in recs.iter().filter(|r| !r.session.is_empty()) {
        let e = m.entry(r.session.clone()).or_insert((r.t, r.t, 0));
        e.0 = e.0.min(r.t);
        e.1 = e.1.max(r.t);
        e.2 += 1;
    }
    let mut v: Vec<(String, u64, u64, usize)> = m.into_iter().map(|(k, (a, b, n))| (k, a, b, n)).collect();
    v.sort_by_key(|s| (s.2, s.1));
    v
}

/// Kept prompt text by hash, after removing text older than the retention set in `prompts.days`.
/// Empty when the `prompts` control is `off`: kept text is then not used, even before it is cleared.
#[must_use]
pub fn kept_prompts(home: &Path, memory: &Path) -> BTreeMap<String, String> {
    let c = crate::controls::current(home);
    if c.get("prompts").is_some_and(|v| v == "off") {
        return BTreeMap::new();
    }
    let days = c.get("prompts.days").and_then(|d| d.parse::<u64>().ok()).unwrap_or(30);
    let _ = crate::hook::prune(&crate::hook::prompts_file(memory), days);
    std::fs::read_to_string(crate::hook::prompts_file(memory))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| nmjson::parse(l.as_bytes()).ok())
        .filter_map(|v| Some((text(&v, "sha")?, text(&v, "text")?)))
        .collect()
}

/// Line ranges merged where they overlap or touch, in order.
#[must_use]
pub fn merge(mut spans: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    spans.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::new();
    for (a, b) in spans {
        match out.last_mut() {
            Some(last) if a <= last.1.saturating_add(1) => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

fn spans_text(spans: &[(u64, u64)]) -> String {
    spans.iter().map(|(a, b)| if a == b { a.to_string() } else { format!("{a}-{b}") }).collect::<Vec<_>>().join(",")
}

/// What one file saw in a session.
#[derive(Clone, Debug, Default)]
pub struct FileUse {
    pub reads: usize,
    pub read_lines: Vec<(u64, u64)>,
    pub edits: usize,
    pub edit_lines: Vec<(u64, u64)>,
    pub writes: usize,
    pub last: u64,
    pub sensitive: bool,
}

/// The state of one session.
#[derive(Clone, Debug, Default)]
pub struct State {
    pub session: String,
    pub first: u64,
    pub last: u64,
    pub records: usize,
    pub model: Option<String>,
    /// Session starts: time and source (startup, resume, clear, compact).
    pub starts: Vec<(u64, String)>,
    pub compactions: usize,
    /// Prompts: time, hash, characters, kept text if any.
    pub prompts: Vec<(u64, String, u64, Option<String>)>,
    pub files: BTreeMap<String, FileUse>,
    /// Shell commands, in order.
    pub commands: Vec<Rec>,
    /// Checks still failing: the newest run of each check (class and package) that failed.
    pub failing: Vec<Rec>,
    /// Tool calls that failed and were not followed by a success of the same tool on the same path.
    pub tool_failures: Vec<Rec>,
    /// Files edited after the newest passing build or test (or with none since).
    pub unchecked: Vec<String>,
    /// Calls of the Nano's tools.
    pub served: Vec<Rec>,
    /// Primes the Nano gave in the session, each on the record of the hook that gave it.
    pub primes: Vec<Rec>,
}

/// The state of `session` from the records of a project (any order; they are sorted here).
#[must_use]
pub fn state(recs: &[Rec], session: &str, prompts: &BTreeMap<String, String>) -> State {
    let mut st = State { session: session.to_owned(), ..State::default() };
    let mut mine: Vec<&Rec> = recs.iter().filter(|r| r.session == session).collect();
    mine.sort_by_key(|r| (r.t, r.i));
    st.records = mine.len();
    st.first = mine.first().map_or(0, |r| r.t);
    st.last = mine.last().map_or(0, |r| r.t);
    let mut checks: BTreeMap<String, Rec> = BTreeMap::new();
    let mut last_pass: u64 = 0;
    let mut edited_at: BTreeMap<String, u64> = BTreeMap::new();
    for r in &mine {
        match r.kind.as_str() {
            "session" => {
                st.starts.push((r.t, r.source.clone().unwrap_or_default()));
                if r.model.is_some() {
                    st.model.clone_from(&r.model);
                }
            }
            "compact" if r.event == "PreCompact" => st.compactions += 1,
            "prompt" => {
                if let Some((sha, chars, _)) = &r.prompt {
                    st.prompts.push((r.t, sha.clone(), *chars, prompts.get(sha).cloned()));
                }
            }
            "nano" if r.done() => st.served.push((*r).clone()),
            _ => {}
        }
        if r.prime.is_some() {
            st.primes.push((*r).clone());
        }
        if !r.done() {
            continue;
        }
        if matches!(r.kind.as_str(), "read" | "edit" | "write") && r.ok == Some(true) {
            for p in r.paths.iter().filter(|p| p.as_str() != "." && p.as_str() != "<outside>") {
                let f = st.files.entry(p.clone()).or_default();
                f.last = r.t;
                f.sensitive |= r.sensitive;
                let span = r.span_of(p);
                match r.kind.as_str() {
                    "read" => {
                        f.reads += 1;
                        f.read_lines.extend(span);
                    }
                    k => {
                        if k == "write" {
                            f.writes += 1;
                        } else {
                            f.edits += 1;
                        }
                        f.edit_lines.extend(span);
                        edited_at.insert(p.clone(), r.t);
                    }
                }
            }
        }
        if matches!(r.tool.as_str(), "Bash" | "PowerShell") {
            st.commands.push((*r).clone());
        }
        if matches!(r.kind.as_str(), "build" | "test") {
            let key = format!("{} {}", r.cmd_class.as_deref().unwrap_or("check"), r.pkg.as_deref().unwrap_or(""));
            if r.ok == Some(true) {
                last_pass = last_pass.max(r.t);
            }
            checks.insert(key, (*r).clone());
        }
        if r.event == "PostToolUseFailure" && !matches!(r.tool.as_str(), "Bash" | "PowerShell") {
            st.tool_failures.push((*r).clone());
        } else if r.ok == Some(true) {
            st.tool_failures.retain(|f| !(f.tool == r.tool && f.paths == r.paths));
        }
    }
    for f in st.files.values_mut() {
        f.read_lines = merge(std::mem::take(&mut f.read_lines));
        f.edit_lines = merge(std::mem::take(&mut f.edit_lines));
    }
    st.failing = checks.into_values().filter(|c| c.ok == Some(false)).collect();
    st.unchecked = edited_at.into_iter().filter(|(_, t)| *t > last_pass).map(|(p, _)| p).collect();
    st
}

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}
fn spans_value(spans: &[(u64, u64)]) -> Value {
    Value::Array(spans.iter().map(|(a, b)| Value::Array(vec![n(*a), n(*b)])).collect())
}
fn obj(f: Vec<(&str, Value)>) -> Value {
    Value::Object(f.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

/// One command as the interface sees it: time, class, package, outcome, exit code, test counts,
/// located errors, and the command text as the hook kept it (secrets already dropped).
fn command_value(r: &Rec) -> Value {
    let mut f = vec![("t", n(r.t)), ("kind", s(&r.kind)), ("class", s(r.cmd_class.as_deref().unwrap_or("other")))];
    if let Some(c) = &r.cmd {
        f.push(("cmd", s(c)));
    }
    if let Some(p) = &r.pkg {
        f.push(("pkg", s(p)));
    }
    f.push(("ok", r.ok.map_or(Value::Null, Value::Bool)));
    if let Some(e) = r.exit {
        f.push(("exit", Value::Number(e.to_string())));
    }
    if let Some((p, x)) = r.tests {
        f.push(("tests", Value::Array(vec![n(p), n(x)])));
    }
    if !r.errors.is_empty() {
        f.push((
            "errors",
            Value::Array(r.errors.iter().map(|(c, file, l)| obj(vec![("code", s(c)), ("file", s(file)), ("line", n(*l))])).collect()),
        ));
    }
    obj(f)
}

/// One prime as the interface sees it: time, the hook event, kind, id, places, and what the
/// evidence ledger knows of it (keyed by the record's place, as packs are).
fn prime_value(r: &Rec, packs: &BTreeMap<usize, Value>) -> Value {
    let x = r.prime.clone().unwrap_or_default();
    let mut f = vec![
        ("t", n(r.t)),
        ("event", s(&r.event)),
        ("kind", s(&x.kind)),
        ("id", s(&x.id)),
        ("places", Value::Array(x.places.iter().map(|p| s(p)).collect())),
        ("from", Value::Array(x.from.iter().map(|p| s(p)).collect())),
    ];
    if let Some(p) = packs.get(&r.i) {
        f.push(("ledger", p.clone()));
    }
    obj(f)
}

impl State {
    /// The state as JSON for the interface. `packs` adds what the evidence ledger knows of each
    /// pack served in the session (`ledger::pack_value` rows), keyed by the record's place.
    #[must_use]
    pub fn to_value(&self, packs: &BTreeMap<usize, Value>) -> Value {
        let task = self.prompts.last().map_or(Value::Null, |(t, sha, chars, kept)| {
            obj(vec![("t", n(*t)), ("sha", s(sha)), ("chars", n(*chars)), ("text", kept.as_deref().map_or(Value::Null, s))])
        });
        let files = self
            .files
            .iter()
            .map(|(p, f)| {
                obj(vec![
                    ("path", s(p)),
                    ("reads", n(f.reads as u64)),
                    ("read_lines", spans_value(&f.read_lines)),
                    ("edits", n(f.edits as u64)),
                    ("writes", n(f.writes as u64)),
                    ("edit_lines", spans_value(&f.edit_lines)),
                    ("last", n(f.last)),
                    ("sensitive", Value::Bool(f.sensitive)),
                ])
            })
            .collect();
        let served = self
            .served
            .iter()
            .map(|r| {
                let (tool, ask, episode) = r.nano.clone().unwrap_or_default();
                let mut f = vec![("t", n(r.t)), ("tool", s(&tool)), ("ask", ask.as_deref().map_or(Value::Null, s)), ("episode", episode.as_deref().map_or(Value::Null, s))];
                if let Some(p) = packs.get(&r.i) {
                    f.push(("pack", p.clone()));
                }
                obj(f)
            })
            .collect();
        let failures = self
            .tool_failures
            .iter()
            .map(|r| obj(vec![("t", n(r.t)), ("tool", s(&r.tool)), ("paths", Value::Array(r.paths.iter().map(|p| s(p)).collect())), ("msg", r.msg.as_deref().map_or(Value::Null, s))]))
            .collect();
        obj(vec![
            ("session", s(&self.session)),
            ("first_ms", n(self.first)),
            ("last_ms", n(self.last)),
            ("records", n(self.records as u64)),
            ("model", self.model.as_deref().map_or(Value::Null, s)),
            ("starts", Value::Array(self.starts.iter().map(|(t, src)| obj(vec![("t", n(*t)), ("source", s(src))])).collect())),
            ("compactions", n(self.compactions as u64)),
            ("task", task),
            ("prompts", n(self.prompts.len() as u64)),
            ("files", Value::Array(files)),
            ("commands", Value::Array(self.commands.iter().map(command_value).collect())),
            (
                "open",
                obj(vec![
                    ("failing_checks", Value::Array(self.failing.iter().map(command_value).collect())),
                    ("tool_failures", Value::Array(failures)),
                    ("unchecked_changes", Value::Array(self.unchecked.iter().map(|p| s(p)).collect())),
                ]),
            ),
            ("served", Value::Array(served)),
            ("primes", Value::Array(self.primes.iter().map(|r| prime_value(r, packs)).collect())),
            // The same list under its earlier name, for an interface that has not moved to `primes`.
            ("nudges", Value::Array(self.primes.iter().map(|r| prime_value(r, packs)).collect())),
        ])
    }

    /// The state as plain lines. `packs` gives one plain line per served pack, keyed as in
    /// `to_value`.
    #[must_use]
    pub fn lines(&self, packs: &BTreeMap<usize, String>) -> Vec<String> {
        let at = |t: u64| nmmusc::pack::utc_stamp(t).replace('T', " ").trim_end_matches('Z').to_owned();
        let clock = |t: u64| at(t).split(' ').nth(1).unwrap_or_default().to_owned();
        let mut out = vec![format!(
            "session {}: {} to {}, {} records{}{}",
            self.session,
            at(self.first),
            clock(self.last),
            self.records,
            self.model.as_deref().map(|m| format!(", model {m}")).unwrap_or_default(),
            if self.starts.len() > 1 { format!(", started {} times ({})", self.starts.len(), self.starts.iter().map(|x| x.1.as_str()).collect::<Vec<_>>().join(", ")) } else { String::new() }
        )];
        match self.prompts.last() {
            Some((_, _, chars, Some(t))) => out.push(format!("task (prompt {} of {}): \"{}\" ({chars} characters)", self.prompts.len(), self.prompts.len(), t.replace('\n', " "))),
            Some((_, _, chars, None)) => out.push(format!("task (prompt {} of {}): {chars} characters, text not kept", self.prompts.len(), self.prompts.len())),
            None => out.push("task: no prompt recorded".into()),
        }
        out.push(format!("files ({}):", self.files.len()));
        for (p, f) in &self.files {
            let mut parts = Vec::new();
            if f.reads > 0 {
                parts.push(if f.read_lines.is_empty() { format!("read {}x", f.reads) } else { format!("read {}", spans_text(&f.read_lines)) });
            }
            if f.edits + f.writes > 0 {
                let what = if f.writes > 0 && f.edits == 0 { "written" } else { "edited" };
                parts.push(if f.edit_lines.is_empty() { format!("{what} {}x", f.edits + f.writes) } else { format!("{what} {} ({}x)", spans_text(&f.edit_lines), f.edits + f.writes) });
            }
            out.push(format!("  {p}  {}{}", parts.join("; "), if f.sensitive { "  (sensitive name)" } else { "" }));
        }
        out.push(format!("commands ({}):", self.commands.len()));
        for c in &self.commands {
            let outcome = match c.ok {
                Some(true) => "ok".to_owned(),
                Some(false) => "FAILED".to_owned(),
                None => String::new(),
            };
            let tests = c.tests.map(|(p, f)| format!("  tests {p} passed {f} failed")).unwrap_or_default();
            let exit = c.exit.map(|e| format!(" exit {e}")).unwrap_or_default();
            out.push(format!("  {} {}  {}{exit}{tests}  {}", clock(c.t), c.cmd_class.as_deref().unwrap_or("other"), outcome, c.cmd.as_deref().unwrap_or("")));
        }
        out.push("open:".into());
        if self.failing.is_empty() && self.tool_failures.is_empty() && self.unchecked.is_empty() {
            out.push("  nothing open".into());
        }
        for c in &self.failing {
            let errs: Vec<String> = c.errors.iter().map(|(code, f, l)| format!("{code} {f}:{l}")).collect();
            out.push(format!("  failing check {} {} (last run {}){}", c.cmd_class.as_deref().unwrap_or("check"), c.pkg.as_deref().unwrap_or(""), clock(c.t), if errs.is_empty() { String::new() } else { format!(": {}", errs.join(", ")) }));
        }
        for f in &self.tool_failures {
            out.push(format!("  failed {} {} ({}): {}", f.tool, f.paths.join(" "), clock(f.t), f.msg.as_deref().unwrap_or("")));
        }
        if !self.unchecked.is_empty() {
            out.push(format!("  changed and not checked since: {}", self.unchecked.join(" ")));
        }
        out.push(format!("served by the Nano ({}):", self.served.len()));
        for r in &self.served {
            let (tool, ask, _) = r.nano.clone().unwrap_or_default();
            out.push(format!("  {} {tool} \"{}\"{}", clock(r.t), ask.unwrap_or_default(), packs.get(&r.i).map(|p| format!("\n    {p}")).unwrap_or_default()));
        }
        if !self.primes.is_empty() {
            out.push(format!("primes given by the Nano ({}):", self.primes.len()));
        }
        for r in &self.primes {
            let x = r.prime.clone().unwrap_or_default();
            out.push(format!("  {} {} at {} ({}): {}{}", clock(r.t), x.kind, r.event, x.id, x.places.join(" "), packs.get(&r.i).map(|p| format!("\n    {p}")).unwrap_or_default()));
        }
        out
    }
}

/// `neuromorph-nano session --project <repo> [--session <id>] [--list] [--json]`: the state of one
/// session (the newest when none is named), or the list of sessions. Read only.
pub fn run(home: &Path, args: &[String]) -> ExitCode {
    let arg = |k: &str| args.windows(2).find(|w| w.first().map(String::as_str) == Some(k)).and_then(|w| w.get(1)).cloned();
    let Some(repo) = arg("--project") else {
        eprintln!("usage: neuromorph-nano session --project <repo> [--session <id>] [--list] [--json]");
        return ExitCode::FAILURE;
    };
    let root = nmsubs::repo::canon(Path::new(&repo));
    let memory = nmtype::project::memory_dir(home, &root);
    let label = nmtype::project::label(home, &root);
    let recs = records(&memory);
    let all = sessions(&recs);
    if args.iter().any(|a| a == "--list") {
        println!("sessions in {} ({}):", crate::hook::situation_file(&memory).display(), all.len());
        for (id, a, b, c) in &all {
            println!("  {id}  {} to {}  {c} records", nmmusc::pack::utc_stamp(*a), nmmusc::pack::utc_stamp(*b));
        }
        return ExitCode::SUCCESS;
    }
    let Some(id) = arg("--session").or_else(|| all.last().map(|x| x.0.clone())) else {
        println!("no session recorded in {}", crate::hook::situation_file(&memory).display());
        return ExitCode::SUCCESS;
    };
    if !all.iter().any(|x| x.0 == id) {
        eprintln!("session: no session {id} in {}", crate::hook::situation_file(&memory).display());
        return ExitCode::FAILURE;
    }
    let st = state(&recs, &id, &kept_prompts(home, &memory));
    let view = crate::ledger::view(home, &memory, &label, &recs, nmtype::now_ms());
    if args.iter().any(|a| a == "--json") {
        println!("{}", st.to_value(&view.pack_values(&id)).encode().unwrap_or_default());
        return ExitCode::SUCCESS;
    }
    for l in st.lines(&view.pack_lines(&id)) {
        println!("{l}");
    }
    ExitCode::SUCCESS
}

/// `/api/session?project=<key>[&session=<id>]`: the sessions of a project and the state of one
/// (the newest when none is named). `project` is the key as `/api/projects` lists it, `nano` for
/// the Nano's own folder.
#[must_use]
pub fn api_view(home: &Path, query: &str) -> Value {
    let Some((key, memory)) = crate::ledger::project_memory(home, query) else {
        return obj(vec![("error", s("project: give the key of a project as /api/projects lists it"))]);
    };
    let recs = records(&memory);
    let all = sessions(&recs);
    let want = crate::project::query_arg(query, "session").filter(|x| !x.is_empty()).or_else(|| all.last().map(|x| x.0.clone()));
    let list = all.iter().map(|(id, a, b, c)| obj(vec![("session", s(id)), ("first_ms", n(*a)), ("last_ms", n(*b)), ("records", n(*c as u64))])).collect();
    let session = match want.filter(|w| all.iter().any(|x| &x.0 == w)) {
        Some(id) => {
            let view = crate::ledger::view(home, &memory, &key, &recs, nmtype::now_ms());
            state(&recs, &id, &kept_prompts(home, &memory)).to_value(&view.pack_values(&id))
        }
        None => Value::Null,
    };
    obj(vec![("project", s(&key)), ("sessions", Value::Array(list)), ("session", session)])
}

/// Section check (processor only): a scripted set of records gives the expected files, line
/// ranges, failing check, unchecked change and served call.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let lines = [
        r#"{"t":1,"event":"SessionStart","kind":"session","session":"s1","source":"startup"}"#,
        r#"{"t":2,"event":"UserPromptSubmit","kind":"prompt","session":"s1","prompt":{"sha":"aa","chars":9,"kept":true}}"#,
        r#"{"t":3,"event":"PostToolUse","kind":"nano","session":"s1","tool":"mcp__neuromorph-nano__nm_context_pack","nano":{"tool":"nm_context_pack","ask":"where"},"ok":true}"#,
        r#"{"t":4,"event":"PostToolUse","kind":"read","session":"s1","tool":"Read","paths":["a.rs"],"lines":[["a.rs",1,20]],"ok":true}"#,
        r#"{"t":5,"event":"PostToolUse","kind":"edit","session":"s1","tool":"Edit","paths":["a.rs"],"lines":[["a.rs",4,6]],"ok":true}"#,
        r#"{"t":6,"event":"PostToolUse","kind":"read","session":"s2","tool":"Read","paths":["b.rs"],"ok":true}"#,
        r#"{"t":7,"event":"PostToolUseFailure","kind":"test","session":"s1","tool":"Bash","cmd":"cargo test","cmd_class":"cargo-test","ok":false,"exit":101,"errors":[{"code":"E0308","file":"a.rs","line":5}]}"#,
    ];
    let recs: Vec<Rec> = lines.iter().enumerate().filter_map(|(i, l)| nmjson::parse(l.as_bytes()).ok().map(|v| Rec::from_value(i, &v))).collect();
    let prompts: BTreeMap<String, String> = [("aa".to_owned(), "fix it now".to_owned())].into_iter().collect();
    let st = state(&recs, "s1", &prompts);
    let a = st.files.get("a.rs").ok_or("sessions: a.rs missing")?;
    if st.files.len() != 1 || a.read_lines != [(1, 20)] || a.edit_lines != [(4, 6)] || a.edits != 1 {
        return Err(format!("sessions: files {:?}", st.files));
    }
    if st.failing.len() != 1 || st.unchecked != ["a.rs"] || st.served.len() != 1 || st.prompts.last().and_then(|p| p.3.as_deref()) != Some("fix it now") {
        return Err(format!("sessions: failing {} unchecked {:?} served {}", st.failing.len(), st.unchecked, st.served.len()));
    }
    if merge(vec![(5, 9), (1, 3), (4, 4), (12, 14)]) != [(1, 9), (12, 14)] {
        return Err("sessions: merge of line ranges".into());
    }
    if sessions(&recs).iter().map(|x| x.0.as_str()).collect::<Vec<_>>() != ["s2", "s1"] {
        return Err("sessions: order of sessions".into());
    }
    Ok(())
}
