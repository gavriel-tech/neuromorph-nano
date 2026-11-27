//! Conformance check, first form (2026-11-10; Gavriel Technologies; direction of Luis Jake
//! Gabriel III: the work must not go in circles or revert to conventional architectures, and
//! every step is judged by whether performance improved and whether the system moved closer
//! to the specification). The rules are data, `designs/conformance.json`: each names the
//! specification section it protects, how it is detected from the repository itself (source
//! text, design files, assemblies), the detector that runs it, and the known violations it
//! tolerates (stand-ins), each with a recorded reason and the plan step that removes it.
//!
//! Components: `text` (walks and text helpers), `anatomy` (C-BLOCK, C-CORE, C-DRONE,
//! C-ONE-TENTACLE, C-CROSS), `stores` (C-CLASS, C-MUSCLE, C-WEIGHTS, C-NUMBERS, C-DOWNLOAD).
//! `run` evaluates every rule under a root; `table` prints one line per rule; `cli` is the
//! command face (registration in the binary is one line: `Some("conform") =>
//! nmtool::conformance::cli(&nano_root())`). The run fails only when a rule marked
//! `must_pass_now` finds a violation that no stand-in covers.
mod anatomy;
mod stores;
mod text;

use nmjson::Value;
use std::path::Path;
use std::time::Instant;

/// The rule file, relative to the mini folder.
pub const RULE_FILE: &str = "designs/conformance.json";

/// One violation a detector found; `key` is what a stand-in names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Violation {
    pub key: String,
    pub text: String,
}

/// What a detector returns: violations, notes, or why it cannot check yet.
#[derive(Clone, Debug, Default)]
pub struct Finding {
    pub violations: Vec<Violation>,
    pub notes: Vec<String>,
    pub not_yet: Option<String>,
}
impl Finding {
    fn add(&mut self, key: &str, text: impl Into<String>) {
        self.violations.push(Violation { key: key.to_owned(), text: text.into() });
    }
}

/// A known violation tolerated with a recorded reason and the plan step that removes it.
#[derive(Clone, Debug)]
pub struct StandIn {
    pub key: String,
    pub reason: String,
    pub step: String,
}

/// One rule as data.
#[derive(Clone, Debug)]
pub struct Rule {
    pub id: String,
    pub protects: String,
    pub detect: String,
    pub detector: String,
    pub must_pass_now: bool,
    pub recorded: String,
    pub stand_ins: Vec<StandIn>,
    pub params: Value,
}

/// A rule's live state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Pass,
    StandIn,
    Fail,
    NotYet,
}
impl State {
    /// The words the table and `recorded_state` use.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::StandIn => "fail (stand-in)",
            Self::Fail => "fail",
            Self::NotYet => "not yet checkable",
        }
    }
}

/// One evaluated rule.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub id: String,
    pub state: State,
    pub reason: String,
    pub step: String,
    pub blocking: bool,
    pub recorded: String,
    pub notes: Vec<String>,
    pub violations: Vec<Violation>,
    pub micros: u128,
}

/// Parse the rule list.
/// # Errors
/// A missing field, a duplicate id, or a stand-in without a reason or step.
pub fn parse_rules(v: &Value) -> Result<Vec<Rule>, String> {
    let mut out: Vec<Rule> = Vec::new();
    for r in text::list(v, "rules") {
        let id = text::s(r, "id")?;
        if out.iter().any(|x| x.id == id) {
            return Err(format!("rule {id} twice"));
        }
        let mut stand_ins = Vec::new();
        for si in text::list(r, "stand_ins") {
            let st = StandIn { key: text::s(si, "key")?, reason: text::s(si, "reason")?, step: text::s(si, "step")? };
            if st.reason.trim().is_empty() || st.step.trim().is_empty() {
                return Err(format!("{id}: stand-in {} needs a reason and a plan step", st.key));
            }
            stand_ins.push(st);
        }
        let must_pass_now = matches!(r.get("must_pass_now"), Ok(Value::Bool(true)));
        let params = r.get("params").cloned().unwrap_or(Value::Null);
        out.push(Rule {
            protects: text::s(r, "protects")?,
            detect: text::s(r, "detect")?,
            detector: text::s(r, "detector")?,
            recorded: text::so(r, "recorded_state"),
            id,
            must_pass_now,
            stand_ins,
            params,
        });
    }
    if out.is_empty() {
        return Err("no rules".into());
    }
    Ok(out)
}

/// Load `designs/conformance.json` under `root`.
/// # Errors
/// Unreadable or invalid rule file.
pub fn load(root: &Path) -> Result<Vec<Rule>, String> {
    let p = root.join(RULE_FILE);
    let bytes = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    parse_rules(&nmjson::parse(&bytes).map_err(|e| format!("{RULE_FILE}: {e}"))?)
}

fn detect(root: &Path, rule: &Rule) -> Result<Finding, String> {
    let p = &rule.params;
    match rule.detector.as_str() {
        "block" => anatomy::block(root, p),
        "core" => anatomy::core(root, p),
        "drone" => anatomy::drone(root, p),
        "one_tentacle" => anatomy::one_tentacle(root, p),
        "cross" => anatomy::cross(root, p),
        "class" => stores::class(root, p),
        "muscle" => stores::muscle(root, p),
        "weights" => stores::weights(root, p),
        "numbers" => stores::numbers(root, p),
        "download" => stores::download(root, p),
        other => Err(format!("unknown detector {other}")),
    }
}

/// Evaluate one rule under `root`.
/// # Errors
/// The detector faulted (a design file it needs is unreadable, a parameter is missing).
pub fn evaluate(root: &Path, rule: &Rule) -> Result<Outcome, String> {
    let t = Instant::now();
    let f = detect(root, rule).map_err(|e| format!("{}: {e}", rule.id))?;
    let uncovered: Vec<&Violation> = f.violations.iter().filter(|v| !rule.stand_ins.iter().any(|s| s.key == v.key)).collect();
    let covering: Vec<&StandIn> = rule.stand_ins.iter().filter(|s| f.violations.iter().any(|v| v.key == s.key)).collect();
    let join = |xs: Vec<String>| {
        let mut d: Vec<String> = Vec::new();
        for x in xs {
            if !d.contains(&x) {
                d.push(x);
            }
        }
        d.join("; ")
    };
    let (state, reason, step) = if let Some(n) = &f.not_yet {
        (State::NotYet, n.clone(), String::new())
    } else if f.violations.is_empty() {
        (State::Pass, f.notes.first().cloned().unwrap_or_default(), String::new())
    } else if uncovered.is_empty() {
        (State::StandIn, join(covering.iter().map(|s| s.reason.clone()).collect()), join(covering.iter().map(|s| s.step.clone()).collect()))
    } else {
        let mut r: Vec<String> = uncovered.iter().take(3).map(|v| v.text.clone()).collect();
        if uncovered.len() > 3 {
            r.push(format!("and {} more", uncovered.len() - 3));
        }
        (State::Fail, format!("new violation: {}", r.join("; ")), String::new())
    };
    Ok(Outcome {
        id: rule.id.clone(),
        blocking: state == State::Fail && rule.must_pass_now,
        state,
        reason,
        step,
        recorded: rule.recorded.clone(),
        notes: f.notes,
        violations: f.violations,
        micros: t.elapsed().as_micros(),
    })
}

/// Evaluate every rule of the rule file under `root`: outcomes and total milliseconds.
/// # Errors
/// The rule file or a detector faulted.
pub fn run(root: &Path) -> Result<(Vec<Outcome>, f64), String> {
    let t = Instant::now();
    let rules = load(root)?;
    let out = rules.iter().map(|r| evaluate(root, r)).collect::<Result<Vec<_>, _>>()?;
    Ok((out, t.elapsed().as_secs_f64() * 1000.0))
}

/// Does any outcome block (a new violation of a must-pass rule)?
#[must_use]
pub fn blocking(outcomes: &[Outcome]) -> bool {
    outcomes.iter().any(|o| o.blocking)
}

/// One line per rule: id, state, whether it blocks, reason, plan step, change since record.
#[must_use]
pub fn table(outcomes: &[Outcome]) -> String {
    let mut s = String::new();
    for o in outcomes {
        let step = if o.step.is_empty() { String::new() } else { format!(" [step {}]", o.step) };
        let moved =
            if o.recorded.is_empty() || o.recorded == o.state.as_str() { String::new() } else { format!(" (recorded: {})", o.recorded) };
        let blocks = if o.blocking { "BLOCKS" } else { "ok" };
        s.push_str(&format!(
            "{:<15} {:<18} {blocks:<6} {}{step}{moved} ({:.1} ms)\n",
            o.id,
            o.state.as_str(),
            o.reason,
            o.micros as f64 / 1000.0
        ));
    }
    s
}

/// The command face: print the table and the time; failure only when a rule blocks.
#[must_use]
pub fn cli(root: &Path) -> std::process::ExitCode {
    match run(root) {
        Ok((out, ms)) => {
            print!("{}", table(&out));
            let n = |st: State| out.iter().filter(|o| o.state == st).count();
            println!(
                "conformance: {} rules in {ms:.1} ms: {} pass, {} stand-in, {} fail, {} not yet checkable; {}",
                out.len(),
                n(State::Pass),
                n(State::StandIn),
                n(State::Fail),
                n(State::NotYet),
                if blocking(&out) { "BLOCKED by a new violation" } else { "nothing blocks" }
            );
            if blocking(&out) { std::process::ExitCode::FAILURE } else { std::process::ExitCode::SUCCESS }
        }
        Err(e) => {
            println!("conformance: fault: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn write(p: &Path, body: &str) -> Result<(), String> {
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    std::fs::write(p, body).map_err(|e| format!("{}: {e}", p.display()))
}

/// A small clean fixture: two tentacles reporting to the Core Brain, one unit type, the three
/// gate files, and a muscle store that refuses sensory items.
fn fixture(dir: &Path) -> Result<(), String> {
    let node = |id: &str, kind: &str, parent: &str| {
        format!(
            "{{\"id\": \"{id}\", \"kind\": \"{kind}\", \"parent\": {}}}",
            if parent.is_empty() { "null".into() } else { format!("\"{parent}\"") }
        )
    };
    let edge = |a: &str, b: &str| format!("{{\"from\": \"{a}\", \"from_port\": \"o\", \"to\": \"{b}\", \"to_port\": \"i\"}}");
    let nodes = [
        node("b#0", "brain", ""),
        node("r#0", "brain", "b#0"),
        node("p#0", "neuropod", ""),
        node("t#0", "tentacle", "p#0"),
        node("d#0", "drone", "t#0"),
        node("t#1", "tentacle", "p#0"),
        node("d#1", "drone", "t#1"),
    ]
    .join(", ");
    let edges = [edge("t#0", "d#0"), edge("d#0", "r#0"), edge("d#1", "r#0")].join(", ");
    write(&dir.join("designs/assemblies/a.json"), &format!("{{\"nodes\": [{nodes}], \"edges\": [{edges}]}}\n"))?;
    write(
        &dir.join("crates/nmpod/src/unit.rs"),
        "pub struct Unit {\n    pub level: u8,\n}\nimpl Unit {\n    pub fn new() -> Self {\n        Self { level: 0 }\n    }\n}\n",
    )?;
    write(&dir.join("crates/nmnte/src/topology.rs"), "pub fn bypasses_brain() {}\n")?;
    write(&dir.join("crates/nmbrain/src/gmap.rs"), "pub const GATE: &str = \"crossover\";\n")?;
    write(&dir.join("crates/nmsubs/src/lens.rs"), "pub const RULE: &str = \"no crossover\";\n")?;
    write(
        &dir.join("crates/nmmusc/src/store.rs"),
        "pub fn put(x: &str) -> Result<(), String> {\n    if x.starts_with(\"sensory\") {\n        return Err(\"sensory item refused\".into());\n    }\n    Ok(())\n}\n",
    )?;
    write(&dir.join("memory/README.md"), "memory\n")
}

/// Plant one violation of each kind in the copy: an edge between tentacles, a unit holding a
/// handle to another and a function taking two, muscle memory under `memory/`, and a store
/// with no sensory refusal.
fn plant(dir: &Path) -> Result<(), String> {
    let a = dir.join("designs/assemblies/a.json");
    let t = text::read(&a)
        .replace("\"edges\": [", "\"edges\": [{\"from\": \"d#0\", \"from_port\": \"o\", \"to\": \"d#1\", \"to_port\": \"i\"}, ");
    write(&a, &t)?;
    write(
        &dir.join("crates/nmpod/src/unit.rs"),
        "pub struct Unit {\n    pub level: u8,\n    pub peer: std::sync::Arc<Unit>,\n}\nimpl Unit {\n    pub fn merge(&mut self, other: &Unit) {\n        self.level = other.level;\n    }\n}\n",
    )?;
    write(&dir.join("memory/muscle/pack.json"), "{}\n")?;
    write(&dir.join("crates/nmmusc/src/store.rs"), "pub fn put(_x: &str) -> Result<(), String> {\n    Ok(())\n}\n")
}

/// Section check: the rule file loads and every rule evaluates on the mini without a fault;
/// on a fixture under `scratch/`, C-CROSS and C-MUSCLE pass clean, and a planted violation in
/// a copy is detected by both as a new, blocking violation.
/// # Errors
/// One line naming what failed.
pub fn check() -> Result<(), String> {
    let root = nmtype::nano_root();
    let rules = load(&root)?;
    for r in &rules {
        evaluate(&root, r)?;
    }
    let dir = root.join("scratch/nmtool-conformance");
    if dir.exists() {
        std::fs::remove_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let (clean, copy) = (dir.join("clean"), dir.join("copy"));
    fixture(&clean)?;
    crate::stepcheck::copy_tree(&clean, &copy).map_err(|e| e.to_string())?;
    plant(&copy)?;
    for id in ["C-CROSS", "C-MUSCLE"] {
        let rule = rules.iter().find(|r| r.id == id).ok_or(format!("{id} missing from the rule file"))?;
        let ok = evaluate(&clean, rule)?;
        if ok.state != State::Pass {
            return Err(format!("{id} on the clean fixture: {} {} {:?}", ok.state.as_str(), ok.reason, ok.violations));
        }
        let bad = evaluate(&copy, rule)?;
        let want: &[&str] =
            if id == "C-CROSS" { &["assembly:", "Unit.peer", "fn merge"] } else { &["placed:memory/muscle", "no-sensory-refusal:nmmusc"] };
        let keys: Vec<&str> = bad.violations.iter().map(|v| v.key.as_str()).collect();
        let missed: Vec<&&str> = want.iter().filter(|w| !keys.iter().any(|k| k.contains(**w))).collect();
        if bad.state != State::Fail || !bad.blocking || !missed.is_empty() {
            return Err(format!(
                "{id} on the planted copy: {} blocking {} missed {missed:?} found {keys:?}",
                bad.state.as_str(),
                bad.blocking
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// The section check, and the first measurement on the mini (`--ignored --nocapture`).
    #[test]
    fn section_check() {
        super::check().unwrap();
    }
    #[test]
    #[ignore = "first measurement: prints the table and the time on the Nano folder"]
    fn first_measurement() {
        let (out, ms) = super::run(&nmtype::nano_root()).unwrap();
        print!("{}", super::table(&out));
        println!("total {ms:.1} ms");
        for o in &out {
            println!("{}: notes {:?}", o.id, o.notes);
        }
    }
}
