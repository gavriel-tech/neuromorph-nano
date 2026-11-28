//! Primes Studio backend: per-project built-in prime toggles, custom primes (event triggers,
//! interval timing triggers, clock-window schedules, file globs, repetition caps, and loop links),
//! saved prime presets, atomic writes, Audit logging with Undo, and live hook evaluation.
use nmjson::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Atomic write using a process- and sequence-unique temporary sibling file so concurrent MCP,
/// HTTP API, and UI writers never collide on the same `.tmp` filename.
fn write_atomic(target: &Path, text: &str) -> Result<(), String> {
    if let Some(dir) = target.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let name = target.file_name().and_then(|x| x.to_str()).unwrap_or("file");
    let tmp = target.with_file_name(format!("{name}.{pid}.{seq}.tmp"));
    std::fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
    if let Err(e) = std::fs::rename(&tmp, target) {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("{}: {e}", target.display()));
    }
    Ok(())
}

/// Per-project primes configuration file inside `<project memory>/`.
pub const CONFIG_FILE: &str = "primes-config.json";
/// Per-project runtime firing state for custom primes (counts, timestamps, session caps).
const STATE_FILE: &str = "primes/custom-state.json";
/// Global user-saved presets inside `<home>/memory/`.
const PRESETS_FILE: &str = "prime-presets.json";
/// Audit kind for prime configuration changes (undoable via `/api/undo`).
pub const AUDIT_KIND: &str = "prime-config";

/// The 6 built-in automatic prime rules: `(id, label, trigger, summary)`.
pub const BUILTINS: &[(&str, &str, &str, &str)] = &[
    ("prompt-recall", "Prompt recall", "UserPromptSubmit / PreInvocation", "Surfaces verified past answers matching the prompt, with cited lines re-read fresh."),
    ("failure-fix", "Failure fix", "PostToolUseFailure / failed check", "Surfaces the verified fix lines when the same error code and file signature recur."),
    ("edit-impact", "Edit impact", "PreToolUse (Edit / MultiEdit / Write)", "Warns before modifying lines holding a verified fix, refuted outcome, or a signature with live callers."),
    ("resume", "Compaction & startup resume", "SessionStart (compact / startup)", "Restores active edited files, failing checks, and open work from the session record."),
    ("core", "Core memories", "SessionStart (1-core)", "Surfaces the repository's most-used 1-core memories at session open when lines are unchanged."),
    ("open-loop", "Open loop check", "Stop (priming = on)", "Flags files modified during the turn that were not built or tested since, or are still failing."),
];

/// Valid triggers for custom primes.
pub const TRIGGERS: &[&str] = &[
    "session-start",
    "prompt",
    "compact",
    "stop",
    "edit",
    "failure",
    "tools",
    "error-repeats",
    "interval",
    "schedule",
];

/// Valid repetition policies.
pub const REPEATS: &[&str] = &["always", "once", "max"];

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}
fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn opt_s(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::string).ok().map(str::trim).filter(|x| !x.is_empty()).map(str::to_owned)
}
fn opt_u(v: &Value, k: &str) -> Option<u64> {
    v.get(k).and_then(Value::integer).ok()
}
fn opt_b(v: &Value, k: &str) -> Option<bool> {
    match v.get(k) {
        Ok(Value::Bool(b)) => Some(*b),
        _ => None,
    }
}

/// Optional clock/day schedule window for a custom prime (UTC 24-hour clock).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Schedule {
    pub days: Vec<String>,
    pub after: String,
    pub before: String,
}

impl Schedule {
    fn is_empty(&self) -> bool {
        self.days.is_empty() && self.after.is_empty() && self.before.is_empty()
    }
    fn to_value(&self) -> Value {
        if self.is_empty() {
            return Value::Null;
        }
        obj(vec![
            ("days", Value::Array(self.days.iter().map(|d| s(d)).collect())),
            ("after", s(&self.after)),
            ("before", s(&self.before)),
        ])
    }
    fn from_value(v: &Value) -> Self {
        let days: Vec<String> = v
            .get("days")
            .ok()
            .and_then(|x| x.array().ok())
            .map(|a| {
                a.iter()
                    .filter_map(|i| i.string().ok())
                    .map(|x| x.trim().to_ascii_lowercase())
                    .filter(|d| matches!(d.as_str(), "mon" | "tue" | "wed" | "thu" | "fri" | "sat" | "sun"))
                    .collect()
            })
            .unwrap_or_default();
        let after = opt_s(v, "after").filter(|x| valid_hhmm(x)).unwrap_or_default();
        let before = opt_s(v, "before").filter(|x| valid_hhmm(x)).unwrap_or_default();
        Self { days, after, before }
    }
    fn matches_now(&self, now_ms: u64) -> bool {
        if self.is_empty() {
            return true;
        }
        let secs = now_ms / 1000;
        let day_idx = ((secs / 86_400) + 3) % 7; // 1970-01-01 was Thursday (index 3, 0=mon)
        let day_names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];
        let today = day_names.get(day_idx as usize).copied().unwrap_or("mon");
        if !self.days.is_empty() && !self.days.iter().any(|d| d == today) {
            return false;
        }
        let mins = ((secs % 86_400) / 60) as u32;
        if let Some(a) = parse_hhmm(&self.after)
            && mins < a
        {
            return false;
        }
        if let Some(b) = parse_hhmm(&self.before)
            && mins > b
        {
            return false;
        }
        true
    }
}

fn parse_hhmm(t: &str) -> Option<u32> {
    let (h, m) = t.split_once(':')?;
    let hh: u32 = h.parse().ok()?;
    let mm: u32 = m.parse().ok()?;
    (hh < 24 && mm < 60).then_some(hh * 60 + mm)
}
fn valid_hhmm(t: &str) -> bool {
    parse_hhmm(t).is_some()
}

/// One user- or agent-defined custom prime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CustomPrime {
    pub id: String,
    pub name: String,
    pub text: String,
    pub enabled: bool,
    pub trigger: String,
    pub every: u64,
    pub interval_min: u64,
    pub schedule: Schedule,
    pub repeat: String,
    pub max: u64,
    pub glob: Option<String>,
    pub error: Option<String>,
    pub error_count: u64,
    pub loop_id: Option<String>,
    pub by: String,
}

impl CustomPrime {
    fn to_value(&self) -> Value {
        obj(vec![
            ("id", s(&self.id)),
            ("name", s(&self.name)),
            ("text", s(&self.text)),
            ("enabled", Value::Bool(self.enabled)),
            ("trigger", s(&self.trigger)),
            ("every", n(self.every)),
            ("interval_min", n(self.interval_min)),
            ("schedule", self.schedule.to_value()),
            ("repeat", s(&self.repeat)),
            ("max", n(self.max)),
            ("glob", self.glob.as_deref().map_or(Value::Null, s)),
            ("error", self.error.as_deref().map_or(Value::Null, s)),
            ("error_count", n(self.error_count)),
            ("loop_id", self.loop_id.as_deref().map_or(Value::Null, s)),
            ("by", s(&self.by)),
        ])
    }
    fn from_value(v: &Value) -> Option<Self> {
        let id = opt_s(v, "id")?;
        let name = opt_s(v, "name")?;
        let text = opt_s(v, "text")?;
        let trigger = opt_s(v, "trigger").filter(|t| TRIGGERS.contains(&t.as_str())).unwrap_or_else(|| "prompt".to_owned());
        let repeat = opt_s(v, "repeat").filter(|r| REPEATS.contains(&r.as_str())).unwrap_or_else(|| "always".to_owned());
        let schedule = v.get("schedule").map(Schedule::from_value).unwrap_or_default();
        Some(Self {
            id,
            name,
            text,
            enabled: opt_b(v, "enabled").unwrap_or(true),
            trigger,
            every: opt_u(v, "every").unwrap_or(1).clamp(1, 100),
            interval_min: opt_u(v, "interval_min").unwrap_or(0).min(1440),
            schedule,
            repeat,
            max: opt_u(v, "max").unwrap_or(3).clamp(1, 100),
            glob: opt_s(v, "glob"),
            error: opt_s(v, "error"),
            error_count: opt_u(v, "error_count").unwrap_or(2).clamp(2, 5),
            loop_id: opt_s(v, "loop_id"),
            by: opt_s(v, "by").unwrap_or_else(|| "user".to_owned()),
        })
    }
}

/// Per-project primes configuration.
#[derive(Clone, Debug)]
pub struct Config {
    pub builtins: BTreeMap<String, bool>,
    pub custom: Vec<CustomPrime>,
}

impl Default for Config {
    fn default() -> Self {
        let mut builtins = BTreeMap::new();
        for (id, _, _, _) in BUILTINS {
            builtins.insert((*id).to_owned(), false);
        }
        Self { builtins, custom: Vec::new() }
    }
}

impl Config {
    fn load(memory: &Path) -> Self {
        let mut cfg = Self::default();
        let Ok(bytes) = std::fs::read(memory.join(CONFIG_FILE)) else { return cfg };
        let Ok(v) = nmjson::parse(&bytes) else { return cfg };
        if let Ok(Value::Object(m)) = v.get("builtins") {
            for (k, val) in m {
                if let Value::Bool(b) = val {
                    cfg.builtins.insert(k.clone(), *b);
                }
            }
        }
        if let Ok(arr) = v.get("custom").and_then(Value::array) {
            cfg.custom = arr.iter().filter_map(CustomPrime::from_value).collect();
        }
        cfg
    }
    fn save(&self, memory: &Path) -> Result<(), String> {
        let mut bm = BTreeMap::new();
        for (k, v) in &self.builtins {
            bm.insert(k.clone(), Value::Bool(*v));
        }
        let body = obj(vec![
            ("version", n(1)),
            ("builtins", Value::Object(bm)),
            ("custom", Value::Array(self.custom.iter().map(CustomPrime::to_value).collect())),
        ]);
        write_atomic(&memory.join(CONFIG_FILE), &nmjson::pretty::pretty(&body))
    }
}

/// True when built-in prime rule `rule` is enabled in `memory` (all 6 default to false, opt-in).
#[must_use]
pub fn builtin_enabled(memory: &Path, rule: &str) -> bool {
    let cfg = Config::load(memory);
    cfg.builtins.get(rule).copied().unwrap_or(false)
}

#[derive(Clone, Debug)]
struct Preset {
    name: String,
    description: String,
    builtin_preset: bool,
    mode: String,
    builtins: BTreeMap<String, bool>,
    custom: Vec<CustomPrime>,
}

impl Preset {
    fn to_value(&self) -> Value {
        let mut bm = BTreeMap::new();
        for (k, v) in &self.builtins {
            bm.insert(k.clone(), Value::Bool(*v));
        }
        obj(vec![
            ("name", s(&self.name)),
            ("description", s(&self.description)),
            ("builtin", Value::Bool(self.builtin_preset)),
            ("mode", s(&self.mode)),
            ("builtins", Value::Object(bm)),
            ("custom", Value::Array(self.custom.iter().map(CustomPrime::to_value).collect())),
        ])
    }
    fn from_value(v: &Value) -> Option<Self> {
        let name = opt_s(v, "name")?;
        let description = opt_s(v, "description").unwrap_or_default();
        let mode = opt_s(v, "mode").filter(|m| matches!(m.as_str(), "off" | "quiet" | "on")).unwrap_or_else(|| "quiet".to_owned());
        let mut builtins = Config::default().builtins;
        if let Ok(Value::Object(m)) = v.get("builtins") {
            for (k, val) in m {
                if let Value::Bool(b) = val {
                    builtins.insert(k.clone(), *b);
                }
            }
        }
        let custom = v.get("custom").and_then(Value::array).map(|a| a.iter().filter_map(CustomPrime::from_value).collect()).unwrap_or_default();
        Some(Self { name, description, builtin_preset: false, mode, builtins, custom })
    }
}

fn builtin_presets() -> Vec<Preset> {
    let all_off = Config::default().builtins;

    vec![
        Preset {
            name: "quiet-verified".into(),
            description: "Prompts the agent to check verified Neuromorph Nano recall and fresh citations once per session.".into(),
            builtin_preset: true,
            mode: "off".into(),
            builtins: all_off.clone(),
            custom: vec![CustomPrime {
                id: "p-verified-recall".into(),
                name: "Verified recall check".into(),
                text: "Before broad repo searches or edits, call nm_context_pack to check verified past answers and confirm cited lines are still fresh on disk.".into(),
                enabled: false,
                trigger: "prompt".into(),
                every: 1,
                interval_min: 0,
                schedule: Schedule::default(),
                repeat: "once".into(),
                max: 1,
                glob: None,
                error: None,
                error_count: 2,
                loop_id: None,
                by: "preset".into(),
            }],
        },
        Preset {
            name: "active-guardrails".into(),
            description: "Timed 30-minute check-in reminding the agent to compile, run tests, and review modified files.".into(),
            builtin_preset: true,
            mode: "off".into(),
            builtins: all_off.clone(),
            custom: vec![CustomPrime {
                id: "p-guardrail-30m".into(),
                name: "30m verification check-in".into(),
                text: "30-minute check-in: verify all modified files compile cleanly, run targeted tests for changed modules, and confirm no temporary debug code remains.".into(),
                enabled: false,
                trigger: "interval".into(),
                every: 1,
                interval_min: 30,
                schedule: Schedule::default(),
                repeat: "always".into(),
                max: 10,
                glob: None,
                error: None,
                error_count: 2,
                loop_id: None,
                by: "preset".into(),
            }],
        },
        Preset {
            name: "build-discipline".into(),
            description: "Triggers after 2 repeated build or test errors to pause and inspect root cause and callers.".into(),
            builtin_preset: true,
            mode: "off".into(),
            builtins: all_off.clone(),
            custom: vec![CustomPrime {
                id: "p-build-repeat".into(),
                name: "Repeated build failure pause".into(),
                text: "The same build or test error has recurred twice. Pause editing, inspect full error context and callers with nm_context_pack or nm_symbol_find, and fix the underlying root cause.".into(),
                enabled: false,
                trigger: "error-repeats".into(),
                every: 1,
                interval_min: 0,
                schedule: Schedule::default(),
                repeat: "max".into(),
                max: 3,
                glob: None,
                error: Some("error".into()),
                error_count: 2,
                loop_id: None,
                by: "preset".into(),
            }],
        },
        Preset {
            name: "compaction-shield".into(),
            description: "Restores active goals, edited files, and test status right after conversation compaction.".into(),
            builtin_preset: true,
            mode: "off".into(),
            builtins: all_off.clone(),
            custom: vec![CustomPrime {
                id: "p-compact-restore".into(),
                name: "Post-compaction continuity".into(),
                text: "Context was just compacted. Re-anchor on the active goal, verify the current git diff and last test result, and continue without repeating completed work.".into(),
                enabled: false,
                trigger: "compact".into(),
                every: 1,
                interval_min: 0,
                schedule: Schedule::default(),
                repeat: "always".into(),
                max: 5,
                glob: None,
                error: None,
                error_count: 2,
                loop_id: None,
                by: "preset".into(),
            }],
        },
        Preset {
            name: "workday-checkin".into(),
            description: "Scheduled weekday (Mon-Fri 09:00-18:00 UTC) hourly diff & scope sanity check.".into(),
            builtin_preset: true,
            mode: "off".into(),
            builtins: all_off.clone(),
            custom: vec![CustomPrime {
                id: "p-workday-schedule".into(),
                name: "Weekday hourly diff review".into(),
                text: "Scheduled check-in: review current changes for scope creep, verify public contracts and docs stay aligned, and run lint/tests before moving on.".into(),
                enabled: false,
                trigger: "schedule".into(),
                every: 1,
                interval_min: 60,
                schedule: Schedule {
                    days: vec!["mon".into(), "tue".into(), "wed".into(), "thu".into(), "fri".into()],
                    after: "09:00".into(),
                    before: "18:00".into(),
                },
                repeat: "max".into(),
                max: 4,
                glob: None,
                error: None,
                error_count: 2,
                loop_id: None,
                by: "preset".into(),
            }],
        },
        Preset {
            name: "pre-edit-guard".into(),
            description: "Before editing core source files, checks downstream callers and existing invariants.".into(),
            builtin_preset: true,
            mode: "off".into(),
            builtins: all_off,
            custom: vec![CustomPrime {
                id: "p-pre-edit-guard".into(),
                name: "Pre-edit caller & contract guard".into(),
                text: "Before editing this source file, check callers with nm_deps or nm_symbol_find, confirm any API/schema parity requirements, and preserve existing test invariants.".into(),
                enabled: false,
                trigger: "edit".into(),
                every: 1,
                interval_min: 5,
                schedule: Schedule::default(),
                repeat: "max".into(),
                max: 5,
                glob: Some("src/**".into()),
                error: None,
                error_count: 2,
                loop_id: None,
                by: "preset".into(),
            }],
        },
    ]
}

fn load_user_presets(home: &Path) -> Vec<Preset> {
    let Ok(bytes) = std::fs::read(home.join("memory").join(PRESETS_FILE)) else { return Vec::new() };
    let Ok(v) = nmjson::parse(&bytes) else { return Vec::new() };
    v.get("presets")
        .and_then(Value::array)
        .map(|a| a.iter().filter_map(Preset::from_value).collect())
        .unwrap_or_default()
}

fn save_user_presets(home: &Path, presets: &[Preset]) -> Result<(), String> {
    let body = obj(vec![("version", n(1)), ("presets", Value::Array(presets.iter().map(Preset::to_value).collect()))]);
    write_atomic(&home.join("memory").join(PRESETS_FILE), &nmjson::pretty::pretty(&body))
}

fn all_presets(home: &Path) -> Vec<Preset> {
    let mut out = builtin_presets();
    for u in load_user_presets(home) {
        if let Some(pos) = out.iter().position(|p| p.name == u.name) {
            out[pos] = u;
        } else {
            out.push(u);
        }
    }
    out
}

/// Runtime firing state per custom prime (`{id: {fired, last_ms, session, session_fired, ticks}}`).
fn load_state(memory: &Path) -> BTreeMap<String, Value> {
    let Ok(bytes) = std::fs::read(memory.join(STATE_FILE)) else { return BTreeMap::new() };
    match nmjson::parse(&bytes) {
        Ok(Value::Object(m)) => m,
        _ => BTreeMap::new(),
    }
}

fn save_state(memory: &Path, state: &BTreeMap<String, Value>) {
    let _ = write_atomic(&memory.join(STATE_FILE), &nmjson::pretty::pretty(&Value::Object(state.clone())));
}

fn resolve_memory(home: &Path, project: Option<&str>) -> Result<(String, PathBuf), String> {
    let q = project.map(str::trim).filter(|x| !x.is_empty()).unwrap_or("nano");
    crate::ledger::project_memory(home, &format!("project={q}")).ok_or_else(|| format!("project: no project {q}"))
}

fn history_path(home: &Path, audit_id: &str) -> PathBuf {
    home.join("memory").join("primes-history").join(format!("{audit_id}.json"))
}

fn simple_glob(pat: &str, path: &str) -> bool {
    let p: Vec<char> = pat.replace('\\', "/").chars().collect();
    let t: Vec<char> = path.replace('\\', "/").chars().collect();
    fn go(p: &[char], t: &[char]) -> bool {
        match p.first() {
            None => t.is_empty(),
            Some('*') if p.get(1) == Some(&'*') => {
                let rest = p.get(2..).unwrap_or(&[]);
                let rest = rest.strip_prefix(&['/']).unwrap_or(rest);
                (0..=t.len()).any(|k| go(rest, t.get(k..).unwrap_or(&[])))
            }
            Some('*') => {
                let rest = p.get(1..).unwrap_or(&[]);
                for k in 0..=t.len() {
                    if go(rest, t.get(k..).unwrap_or(&[])) {
                        return true;
                    }
                    if t.get(k) == Some(&'/') {
                        break;
                    }
                }
                false
            }
            Some('?') => t.first().is_some_and(|c| *c != '/') && go(p.get(1..).unwrap_or(&[]), t.get(1..).unwrap_or(&[])),
            Some(c) => t.first().is_some_and(|x| x.eq_ignore_ascii_case(c)) && go(p.get(1..).unwrap_or(&[]), t.get(1..).unwrap_or(&[])),
        }
    }
    go(&p, &t)
}

/// Hook invocation context for evaluating custom primes (`hook.rs`).
pub struct HookCtx<'a> {
    pub home: &'a Path,
    pub memory: &'a Path,
    pub root: &'a Path,
    pub event: &'a str,
    pub rec: &'a BTreeMap<String, Value>,
    pub now_ms: u64,
    pub agy: bool,
}

/// Evaluates enabled custom primes during a lifecycle hook (`hook.rs`). Returns any triggered
/// custom primes (name, id, text) ready to inject alongside built-in primes and guidance.
pub fn at_hook(ctx: &HookCtx<'_>) -> Vec<(String, String, String)> {
    let (home, memory, root, event, rec, now_ms, agy) = (ctx.home, ctx.memory, ctx.root, ctx.event, ctx.rec, ctx.now_ms, ctx.agy);
    let mode = crate::prime::Mode::parse(crate::controls::current(home).get("priming"));
    if mode == crate::prime::Mode::Off {
        return Vec::new();
    }
    let cfg = Config::load(memory);
    if cfg.custom.is_empty() {
        return Vec::new();
    }
    let session = rec.get("session").and_then(|v| v.string().ok()).unwrap_or("default");
    let kind = rec.get("kind").and_then(|v| v.string().ok()).unwrap_or("");
    let ok_false = matches!(rec.get("ok"), Some(Value::Bool(false)));
    let rec_paths: Vec<String> = rec
        .get("paths")
        .and_then(|v| v.array().ok())
        .map(|a| a.iter().filter_map(|v| v.string().ok().map(str::to_owned)).collect())
        .unwrap_or_default();
    if !rec_paths.is_empty() && rec_paths.iter().all(|p| nmsubs::repo::is_nanoignored(root, p)) {
        return Vec::new();
    }
    let mut state = load_state(memory);
    let mut dirty = false;
    let mut fired = Vec::new();

    for cp in &cfg.custom {
        if !cp.enabled {
            continue;
        }
        let trigger_match = match cp.trigger.as_str() {
            "session-start" => event == "SessionStart",
            "prompt" => event == "UserPromptSubmit",
            "compact" => event == "SessionStart" && rec.get("source").and_then(|v| v.string().ok()) == Some("compact"),
            "stop" => event == "Stop",
            "edit" => event == "PreToolUse" && matches!(rec.get("tool").and_then(|v| v.string().ok()), Some("Edit" | "MultiEdit" | "Write" | "NotebookEdit")),
            "failure" => matches!(event, "PostToolUse" | "PostToolUseFailure") && ok_false,
            "tools" => event == "PostToolUse",
            "error-repeats" => {
                if !matches!(event, "PostToolUse" | "PostToolUseFailure") || !ok_false {
                    false
                } else if let Some(sig) = &cp.error {
                    let needle = sig.to_ascii_lowercase();
                    let err_blob = rec.get("errors").map(|v| v.encode().unwrap_or_default()).unwrap_or_default().to_ascii_lowercase()
                        + &rec.get("error").and_then(|v| v.string().ok()).unwrap_or("").to_ascii_lowercase();
                    err_blob.contains(&needle)
                } else {
                    true
                }
            }
            "interval" => matches!(event, "UserPromptSubmit" | "SessionStart" | "Stop" | "PostToolUse"),
            "schedule" => matches!(event, "UserPromptSubmit" | "SessionStart"),
            _ => false,
        };
        if !trigger_match {
            continue;
        }
        if let Some(g) = &cp.glob
            && !rec_paths.iter().any(|p| simple_glob(g, p))
        {
            continue;
        }
        if !cp.schedule.matches_now(now_ms) {
            continue;
        }

        let prev = state.get(&cp.id);
        let prev_session = prev.and_then(|v| v.get("session").and_then(Value::string).ok()).unwrap_or("");
        let mut sess_fired = if prev_session == session { prev.and_then(|v| v.get("session_fired").and_then(Value::integer).ok()).unwrap_or(0) } else { 0 };
        let mut ticks = if prev_session == session { prev.and_then(|v| v.get("ticks").and_then(Value::integer).ok()).unwrap_or(0) } else { 0 };
        let last_ms = prev.and_then(|v| v.get("last_ms").and_then(Value::integer).ok()).unwrap_or(0);
        let total_fired = prev.and_then(|v| v.get("fired").and_then(Value::integer).ok()).unwrap_or(0);

        ticks += 1;
        let stride = if cp.trigger == "error-repeats" { cp.error_count } else { cp.every };
        if stride > 1 && !ticks.is_multiple_of(stride) {
            state.insert(
                cp.id.clone(),
                obj(vec![
                    ("fired", n(total_fired)),
                    ("last_ms", n(last_ms)),
                    ("session", s(session)),
                    ("session_fired", n(sess_fired)),
                    ("ticks", n(ticks)),
                ]),
            );
            dirty = true;
            continue;
        }

        // Interval timing trigger / cooldown check.
        let min_gap_min = if cp.trigger == "interval" { cp.interval_min.max(1) } else { cp.interval_min };
        if min_gap_min > 0 && last_ms > 0 && now_ms.saturating_sub(last_ms) < min_gap_min.saturating_mul(60_000) {
            continue;
        }

        // Repetition cap per session.
        let blocked = match cp.repeat.as_str() {
            "once" => sess_fired >= 1,
            "max" => sess_fired >= cp.max,
            _ => false,
        };
        if blocked {
            continue;
        }

        sess_fired += 1;
        state.insert(
            cp.id.clone(),
            obj(vec![
                ("fired", n(total_fired + 1)),
                ("last_ms", n(now_ms)),
                ("session", s(session)),
                ("session_fired", n(sess_fired)),
                ("ticks", n(ticks)),
            ]),
        );
        dirty = true;
        let line = if cp.text.starts_with("Neuromorph Nano") { cp.text.clone() } else { format!("Neuromorph Nano prime ({}): {}", cp.name, cp.text) };
        let proj_key = memory.file_name().and_then(|x| x.to_str()).unwrap_or("nano");
        let _ = crate::controls::audit(
            home,
            "prime",
            vec![
                ("project", s(proj_key)),
                ("prime_kind", s("custom")),
                ("prime_id", s(&cp.id)),
                ("name", s(&cp.name)),
                ("trigger", s(&cp.trigger)),
                ("event", s(event)),
                ("kind_event", s(kind)),
                ("via", s(if agy { "antigravity" } else { "claude-code" })),
                ("ok", Value::Bool(true)),
                ("label", s(&format!("Custom prime '{}' ({}) at {}", cp.name, cp.trigger, event))),
            ],
        );
        fired.push((cp.id.clone(), cp.name.clone(), line));
    }

    if dirty {
        save_state(memory, &state);
    }
    fired
}

/// `GET /api/primes?project=<key>` and `nm_settings op=primes`.
#[must_use]
pub fn view(home: &Path, query: &str) -> Value {
    let proj_arg = crate::project::query_arg(query, "project");
    let (key, memory) = match resolve_memory(home, proj_arg.as_deref()) {
        Ok(x) => x,
        Err(e) => return obj(vec![("error", s(&e))]),
    };
    let cfg = Config::load(&memory);
    let state = load_state(&memory);
    let controls = crate::controls::current(home);
    let mode = crate::prime::Mode::parse(controls.get("priming")).label();
    let ledger = crate::ledger::api_view(home, &format!("project={key}"));
    let guidance = crate::guidance::view(home, &format!("project={key}"));

    let builtins: Vec<Value> = BUILTINS
        .iter()
        .map(|(id, label, trigger, summary)| {
            obj(vec![
                ("id", s(id)),
                ("label", s(label)),
                ("trigger", s(trigger)),
                ("summary", s(summary)),
                ("enabled", Value::Bool(cfg.builtins.get(*id).copied().unwrap_or(false))),
            ])
        })
        .collect();

    let custom: Vec<Value> = cfg
        .custom
        .iter()
        .map(|c| {
            let st = state.get(&c.id);
            let mut val = c.to_value();
            if let Value::Object(m) = &mut val {
                m.insert("fired".into(), n(st.and_then(|x| x.get("fired").and_then(Value::integer).ok()).unwrap_or(0)));
                m.insert("last_ms".into(), n(st.and_then(|x| x.get("last_ms").and_then(Value::integer).ok()).unwrap_or(0)));
            }
            val
        })
        .collect();

    let presets: Vec<Value> = all_presets(home).iter().map(Preset::to_value).collect();
    let loops = guidance.get("loops").cloned().unwrap_or_else(|_| Value::Array(Vec::new()));
    let steps = guidance.get("steps").cloned().unwrap_or_else(|_| Value::Array(Vec::new()));
    let reminders = guidance.get("reminders").cloned().unwrap_or_else(|_| Value::Array(Vec::new()));
    let counts = ledger.get("primes").cloned().unwrap_or(Value::Null);

    obj(vec![
        ("project", s(&key)),
        ("mode", s(mode)),
        ("builtins", Value::Array(builtins)),
        ("custom", Value::Array(custom)),
        ("presets", Value::Array(presets)),
        ("triggers", Value::Array(TRIGGERS.iter().map(|t| s(t)).collect())),
        ("repeats", Value::Array(REPEATS.iter().map(|r| s(r)).collect())),
        ("counts", counts),
        ("loops", loops),
        ("steps", steps),
        ("reminders", reminders),
    ])
}

/// Preview text for two-step delete of a custom prime or saved preset (`forget.rs`).
pub fn delete_preview(home: &Path, v: &Value) -> Result<(String, String), String> {
    let action = opt_s(v, "action").unwrap_or_default();
    match action.as_str() {
        "remove" => {
            let id = opt_s(v, "id").ok_or_else(|| "id: give the custom prime id".to_owned())?;
            let (key, memory) = resolve_memory(home, opt_s(v, "project").as_deref())?;
            let cfg = Config::load(&memory);
            let found = cfg.custom.iter().find(|c| c.id == id).ok_or_else(|| format!("id: no custom prime {id} in {key}"))?;
            Ok((format!("Remove custom prime '{}' ({id}) from {key}", found.name), id))
        }
        "preset_delete" => {
            let name = opt_s(v, "name").ok_or_else(|| "name: give the preset name".to_owned())?;
            let user = load_user_presets(home);
            if !user.iter().any(|p| p.name == name) {
                return Err(format!("name: no user-saved preset '{name}' (built-in presets cannot be deleted)"));
            }
            Ok((format!("Delete saved prime preset '{name}'"), name))
        }
        other => Err(format!("action {other}: not a delete action")),
    }
}

/// `POST /api/primes` and `nm_settings` `prime_*` ops.
/// # Errors
/// Validation or I/O failure.
pub fn write(home: &Path, v: &Value, by: &str) -> Result<Value, String> {
    let action = opt_s(v, "action").ok_or_else(|| "action: use toggle_builtin, add, edit, toggle, remove, preset_save, preset_apply or preset_delete".to_owned())?;
    let (key, memory) = resolve_memory(home, opt_s(v, "project").as_deref())?;
    let mut cfg = Config::load(&memory);
    let prev_cfg_json = std::fs::read_to_string(memory.join(CONFIG_FILE)).unwrap_or_default();
    let prev_presets_json = std::fs::read_to_string(home.join("memory").join(PRESETS_FILE)).unwrap_or_default();
    let prev_mode = crate::prime::Mode::parse(crate::controls::current(home).get("priming")).label().to_owned();

    let label: String = match action.as_str() {
        "toggle_builtin" => {
            let id = opt_s(v, "id").ok_or_else(|| "id: give a built-in prime rule id".to_owned())?;
            if !BUILTINS.iter().any(|(b, _, _, _)| *b == id) {
                return Err(format!("id: unknown built-in prime '{id}'"));
            }
            let enabled = opt_b(v, "enabled").ok_or_else(|| "enabled: true or false".to_owned())?;
            cfg.builtins.insert(id.clone(), enabled);
            cfg.save(&memory)?;
            format!("Built-in prime '{id}' {} ({key})", if enabled { "enabled" } else { "disabled" })
        }
        "add" => {
            let name = opt_s(v, "name").ok_or_else(|| "name: give a short name for this prime".to_owned())?;
            let text = opt_s(v, "text").ok_or_else(|| "text: give the nudge text for this prime".to_owned())?;
            if name.chars().count() > 80 || text.chars().count() > 2000 {
                return Err("name (<=80) or text (<=2000) is too long".to_owned());
            }
            let trigger = opt_s(v, "trigger").unwrap_or_else(|| "prompt".to_owned());
            if !TRIGGERS.contains(&trigger.as_str()) {
                return Err(format!("trigger: unknown trigger '{trigger}'"));
            }
            let repeat = opt_s(v, "repeat").unwrap_or_else(|| "always".to_owned());
            if !REPEATS.contains(&repeat.as_str()) {
                return Err(format!("repeat: use always, once or max (got '{repeat}')"));
            }
            if cfg.custom.len() >= 100 {
                return Err("at most 100 custom primes per project".to_owned());
            }
            let hash = nmhash::sha256_bytes(format!("{key}:{name}:{}:{}", nmtype::now_ms(), cfg.custom.len()).as_bytes());
            let id = format!("p-{}", hash.get(..8).unwrap_or("00000000"));
            let schedule = v.get("schedule").map(Schedule::from_value).unwrap_or_default();
            let cp = CustomPrime {
                id: id.clone(),
                name: name.clone(),
                text,
                enabled: opt_b(v, "enabled").unwrap_or(true),
                trigger: trigger.clone(),
                every: opt_u(v, "every").unwrap_or(1).clamp(1, 100),
                interval_min: opt_u(v, "interval_min").unwrap_or(if trigger == "interval" { 15 } else { 0 }).min(1440),
                schedule,
                repeat,
                max: opt_u(v, "max").unwrap_or(3).clamp(1, 100),
                glob: opt_s(v, "glob"),
                error: opt_s(v, "error"),
                error_count: opt_u(v, "error_count").unwrap_or(2).clamp(2, 5),
                loop_id: opt_s(v, "loop_id"),
                by: if by == "mcp" { "agent".to_owned() } else { "user".to_owned() },
            };
            cfg.custom.push(cp);
            cfg.save(&memory)?;
            format!("Added custom prime '{name}' ({trigger}) to {key}")
        }
        "edit" => {
            let id = opt_s(v, "id").ok_or_else(|| "id: give the custom prime id".to_owned())?;
            let cp = cfg.custom.iter_mut().find(|c| c.id == id).ok_or_else(|| format!("id: no custom prime {id}"))?;
            if let Some(name) = opt_s(v, "name") {
                if name.chars().count() > 80 {
                    return Err("name: at most 80 characters".to_owned());
                }
                cp.name = name;
            }
            if let Some(text) = opt_s(v, "text") {
                if text.chars().count() > 2000 {
                    return Err("text: at most 2000 characters".to_owned());
                }
                cp.text = text;
            }
            if let Some(trigger) = opt_s(v, "trigger") {
                if !TRIGGERS.contains(&trigger.as_str()) {
                    return Err(format!("trigger: unknown trigger '{trigger}'"));
                }
                cp.trigger = trigger;
            }
            if let Some(repeat) = opt_s(v, "repeat") {
                if !REPEATS.contains(&repeat.as_str()) {
                    return Err(format!("repeat: use always, once or max (got '{repeat}')"));
                }
                cp.repeat = repeat;
            }
            if let Some(en) = opt_b(v, "enabled") {
                cp.enabled = en;
            }
            if let Some(ev) = opt_u(v, "every") {
                cp.every = ev.clamp(1, 100);
            }
            if let Some(im) = opt_u(v, "interval_min") {
                cp.interval_min = im.min(1440);
            }
            if let Some(mx) = opt_u(v, "max") {
                cp.max = mx.clamp(1, 100);
            }
            if let Ok(sc) = v.get("schedule") {
                cp.schedule = Schedule::from_value(sc);
            }
            if v.get("glob").is_ok() {
                cp.glob = opt_s(v, "glob");
            }
            if v.get("error").is_ok() {
                cp.error = opt_s(v, "error");
            }
            if let Some(ec) = opt_u(v, "error_count") {
                cp.error_count = ec.clamp(2, 5);
            }
            if v.get("loop_id").is_ok() {
                cp.loop_id = opt_s(v, "loop_id");
            }
            let nm = cp.name.clone();
            cfg.save(&memory)?;
            format!("Updated custom prime '{nm}' ({id}) in {key}")
        }
        "toggle" => {
            let id = opt_s(v, "id").ok_or_else(|| "id: give the custom prime id".to_owned())?;
            let enabled = opt_b(v, "enabled").ok_or_else(|| "enabled: true or false".to_owned())?;
            let cp = cfg.custom.iter_mut().find(|c| c.id == id).ok_or_else(|| format!("id: no custom prime {id}"))?;
            cp.enabled = enabled;
            let nm = cp.name.clone();
            cfg.save(&memory)?;
            format!("Custom prime '{nm}' {} ({key})", if enabled { "enabled" } else { "disabled" })
        }
        "remove" => {
            let id = opt_s(v, "id").ok_or_else(|| "id: give the custom prime id".to_owned())?;
            let pos = cfg.custom.iter().position(|c| c.id == id).ok_or_else(|| format!("id: no custom prime {id}"))?;
            let removed = cfg.custom.remove(pos);
            cfg.save(&memory)?;
            format!("Removed custom prime '{}' ({id}) from {key}", removed.name)
        }
        "preset_save" => {
            let name = opt_s(v, "name").ok_or_else(|| "name: give a name for this preset".to_owned())?;
            if name.chars().count() > 60 {
                return Err("name: at most 60 characters".to_owned());
            }
            let description = opt_s(v, "description").unwrap_or_else(|| format!("Saved from project {key}"));
            let mut user = load_user_presets(home);
            let preset = Preset {
                name: name.clone(),
                description,
                builtin_preset: false,
                mode: prev_mode.clone(),
                builtins: cfg.builtins.clone(),
                custom: cfg.custom.clone(),
            };
            if let Some(pos) = user.iter().position(|p| p.name == name) {
                user[pos] = preset;
            } else {
                user.push(preset);
            }
            save_user_presets(home, &user)?;
            format!("Saved prime preset '{name}' from {key}")
        }
        "all_off" => {
            for (id, _, _, _) in BUILTINS {
                cfg.builtins.insert((*id).to_owned(), false);
            }
            for cp in &mut cfg.custom {
                cp.enabled = false;
            }
            cfg.save(&memory)?;
            let _ = crate::controls::set_by(home, "priming", "off", None, by);
            format!("Turned all primes off ({key})")
        }
        "preset_apply" => {
            let name = opt_s(v, "name").ok_or_else(|| "name: give the preset name to apply".to_owned())?;
            let preset = all_presets(home).into_iter().find(|p| p.name == name).ok_or_else(|| format!("name: no prime preset '{name}'"))?;
            if preset.builtin_preset {
                // Preloaded presets add their primes into the project's list switched OFF
                // (iPhone alarm style) so nothing fires until the user toggles each one on,
                // and any added prime can be individually toggled, edited, or deleted.
                for mut cp in preset.custom {
                    cp.enabled = false;
                    if !cfg.custom.iter().any(|existing| existing.id == cp.id) {
                        cfg.custom.push(cp);
                    }
                }
                cfg.save(&memory)?;
                format!("Added preloaded preset '{name}' to {key} (switched off until enabled)")
            } else {
                cfg.builtins = preset.builtins;
                cfg.custom = preset.custom;
                cfg.save(&memory)?;
                let _ = crate::controls::set_by(home, "priming", &preset.mode, None, by);
                format!("Applied saved prime preset '{name}' to {key}")
            }
        }
        "preset_delete" => {
            let name = opt_s(v, "name").ok_or_else(|| "name: give the preset name to delete".to_owned())?;
            let mut user = load_user_presets(home);
            let before = user.len();
            user.retain(|p| p.name != name);
            if user.len() == before {
                return Err(format!("name: no user-saved preset '{name}'"));
            }
            save_user_presets(home, &user)?;
            format!("Deleted prime preset '{name}'")
        }
        other => return Err(format!("action {other}: use toggle_builtin, add, edit, toggle, remove, all_off, preset_save, preset_apply or preset_delete")),
    };

    let audit_id = crate::controls::audit(
        home,
        AUDIT_KIND,
        vec![
            ("action", s(&action)),
            ("project", s(&key)),
            ("by", s(by)),
            ("label", s(&label)),
        ],
    )?;
    let hp = history_path(home, &audit_id);
    if let Some(dir) = hp.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let snap = obj(vec![
        ("project", s(&key)),
        ("prev_cfg", s(&prev_cfg_json)),
        ("prev_presets", s(&prev_presets_json)),
        ("prev_mode", s(&prev_mode)),
        ("action", s(&action)),
    ]);
    if let Ok(enc) = snap.encode() {
        let _ = std::fs::write(&hp, enc.as_bytes());
    }
    Ok(obj(vec![("audit_id", s(&audit_id)), ("primes", view(home, &format!("project={key}")))]))
}

/// Undoes one `prime-config` audit row by restoring the saved pre-change snapshot.
pub fn undo(home: &Path, _row: &Value, id: &str, by: &str) -> Result<Option<String>, String> {
    let bytes = std::fs::read(history_path(home, id)).map_err(|_| format!("prime-config undo: no saved snapshot for {id}"))?;
    let snap = nmjson::parse(&bytes).map_err(|e| format!("prime-config undo: {e}"))?;
    let key = snap.get("project").and_then(Value::string).unwrap_or("nano");
    let (_, memory) = resolve_memory(home, Some(key))?;
    let prev_cfg = snap.get("prev_cfg").and_then(Value::string).unwrap_or("");
    let prev_presets = snap.get("prev_presets").and_then(Value::string).unwrap_or("");
    let prev_mode = snap.get("prev_mode").and_then(Value::string).unwrap_or("quiet");
    let action = snap.get("action").and_then(Value::string).unwrap_or("");

    let cfg_path = memory.join(CONFIG_FILE);
    if prev_cfg.is_empty() {
        let _ = std::fs::remove_file(&cfg_path);
    } else {
        std::fs::create_dir_all(&memory).map_err(|e| format!("{}: {e}", memory.display()))?;
        std::fs::write(&cfg_path, prev_cfg.as_bytes()).map_err(|e| format!("{}: {e}", cfg_path.display()))?;
    }
    if matches!(action, "preset_save" | "preset_delete") {
        let pp = home.join("memory").join(PRESETS_FILE);
        if prev_presets.is_empty() {
            let _ = std::fs::remove_file(&pp);
        } else {
            let _ = std::fs::write(&pp, prev_presets.as_bytes());
        }
    }
    if matches!(action, "preset_apply" | "all_off") {
        let _ = crate::controls::set_by(home, "priming", prev_mode, None, by);
    }
    let undo_id = crate::controls::audit(
        home,
        AUDIT_KIND,
        vec![
            ("action", s("undo")),
            ("undo_of", s(id)),
            ("project", s(key)),
            ("by", s(by)),
            ("label", s(&format!("Undid prime configuration change ({key})"))),
        ],
    )?;
    Ok(Some(undo_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primes_ctl_builtins_custom_interval_presets_and_undo() {
        let base = std::env::temp_dir().join(format!("nmcore-primes-ctl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let home = base.join("home");
        let memory = home.join("memory");
        std::fs::create_dir_all(&memory).unwrap();

        // 1. Built-ins default to off (opt-in); toggle one on and verify builtin_enabled().
        assert!(!builtin_enabled(&memory, "prompt-recall"));
        let r1 = write(&home, &obj(vec![("project", s("nano")), ("action", s("toggle_builtin")), ("id", s("prompt-recall")), ("enabled", Value::Bool(true))]), "test").unwrap();
        assert!(builtin_enabled(&memory, "prompt-recall"));
        let aid1 = r1.get("audit_id").and_then(Value::string).unwrap().to_owned();

        // 2. Enable priming mode, add a custom interval prime, fire it once, confirm cooldown holds.
        let _ = crate::controls::set_by(&home, "priming", "on", None, "test").unwrap();
        let _ = write(
            &home,
            &obj(vec![
                ("project", s("nano")),
                ("action", s("add")),
                ("name", s("15m check")),
                ("text", s("Run tests now")),
                ("trigger", s("interval")),
                ("interval_min", n(15)),
            ]),
            "test",
        )
        .unwrap();
        let mut rec = BTreeMap::new();
        rec.insert("session".into(), s("s1"));
        let t0 = 1_700_000_000_000u64;
        let hit1 = at_hook(&HookCtx { home: &home, memory: &memory, root: &home, event: "UserPromptSubmit", rec: &rec, now_ms: t0, agy: false });
        assert_eq!(hit1.len(), 1);
        let hit2 = at_hook(&HookCtx { home: &home, memory: &memory, root: &home, event: "UserPromptSubmit", rec: &rec, now_ms: t0 + 60_000, agy: false });
        assert!(hit2.is_empty(), "cooldown must hold for 15 minutes");
        let hit3 = at_hook(&HookCtx { home: &home, memory: &memory, root: &home, event: "UserPromptSubmit", rec: &rec, now_ms: t0 + 16 * 60_000, agy: false });
        assert_eq!(hit3.len(), 1);

        // 3. Adding a preloaded preset adds its prime switched OFF (iPhone alarm model).
        let _ = write(&home, &obj(vec![("project", s("nano")), ("action", s("preset_apply")), ("name", s("active-guardrails"))]), "test").unwrap();
        let cfg_after = Config::load(&memory);
        let added = cfg_after.custom.iter().find(|c| c.id == "p-guardrail-30m").unwrap();
        assert!(!added.enabled, "preloaded preset primes must start OFF");

        // 4. Save user preset, turn all off, then re-apply saved preset.
        let _ = write(&home, &obj(vec![("project", s("nano")), ("action", s("preset_save")), ("name", s("my-preset"))]), "test").unwrap();
        let _ = write(&home, &obj(vec![("project", s("nano")), ("action", s("all_off"))]), "test").unwrap();
        assert!(!builtin_enabled(&memory, "prompt-recall"));
        let _ = write(&home, &obj(vec![("project", s("nano")), ("action", s("preset_apply")), ("name", s("my-preset"))]), "test").unwrap();
        assert!(builtin_enabled(&memory, "prompt-recall"));

        // 5. Undo the original toggle cleanly -> back to default off.
        let _ = crate::controls::undo_by(&home, &aid1, "test").unwrap();
        assert!(!builtin_enabled(&memory, "prompt-recall"));

        let _ = std::fs::remove_dir_all(&base);
    }
}
