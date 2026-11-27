//! Product tests of primes and of memory expansion, run through the real program on copies of the
//! small fixture repository under `scratch/priming-test/`, with a test Nano folder there
//! (`NANO_HOME`). Each scripted session is fed to `neuromorph-nano hook <Event>` one event at a
//! time. What each hook printed is compared exactly with the replies written down beforehand in
//! `tests/fixtures/priming/` (`-` is silence), and so are the ledger's prime and outcome rows (as one
//! line each: kind, state, strength, steps, the event's tool and place; hashes, times and ids left
//! out). Claude Code itself is not run, no real settings are touched, and nothing outside
//! `scratch/` is written.
#![allow(clippy::unwrap_used, clippy::too_many_arguments)]
use nmjson::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

const EXE: &str = env!("CARGO_BIN_EXE_neuromorph-nano");
const ASK: &str = "where is the checkout total computed";
const LIB: &str = "crates/shop/src/lib.rs";
const LINE18: &str = "    Some(unit_price_cents(&item.code) * u64::from(n))";
const LINE18_FIXED: &str = "    Some(unit_price_cents(&item.code) * u64::from(n) + 0)";
const E0308_18: &str = "Exit code 101\nerror[E0308]: mismatched types\n  --> crates/shop/src/lib.rs:18:5\ntest result: FAILED. 2 passed; 1 failed";
const PASS: &str = "test result: ok. 3 passed; 0 failed; 0 ignored";

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").canonicalize().unwrap()
}
fn now_ms() -> u64 {
    u64::try_from(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()).unwrap()
}
fn pause(ms: u64) {
    let t = std::time::Instant::now();
    while t.elapsed().as_millis() < u128::from(ms) {
        std::thread::yield_now();
    }
}
fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap().flatten() {
        let p = e.path();
        if p.is_dir() {
            copy_dir(&p, &to.join(e.file_name()));
        } else {
            std::fs::copy(&p, to.join(e.file_name())).unwrap();
        }
    }
}
fn text(o: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}
fn s(t: &str) -> Value {
    Value::String(t.to_owned())
}
fn obj(f: Vec<(&str, Value)>) -> Value {
    Value::Object(f.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn get(v: &Value, k: &str) -> String {
    match v.get(k) {
        Ok(Value::String(x)) => x.clone(),
        Ok(Value::Number(x)) => x.clone(),
        _ => "-".into(),
    }
}

/// The expected replies of a fixture: label to text (`-` for silence), and the expected ledger lines.
fn expected(name: &str) -> (BTreeMap<String, String>, Vec<String>) {
    let t = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("priming").join(name)).unwrap();
    let mut replies = BTreeMap::new();
    let mut ledger = Vec::new();
    for l in t.lines().filter(|l| !l.trim().is_empty()) {
        let (k, v) = l.split_once('\t').unwrap();
        if k == "ledger" {
            ledger.push(v.to_owned());
        } else {
            replies.insert(k.to_owned(), v.to_owned());
        }
    }
    (replies, ledger)
}

struct Place {
    dir: PathBuf,
    home: PathBuf,
    repo: PathBuf,
    want: BTreeMap<String, String>,
    /// Every reply printed, by label.
    got: Vec<(String, String)>,
    /// The prime deadline given to the hook (`NANO_PRIME_DEADLINE_MS`); `None` keeps the program's
    /// own. Scripted sessions get a long one so a loaded test machine does not drop a prime whose
    /// text is being checked; the timing test and the drop test set their own.
    deadline: Option<u64>,
}

impl Place {
    fn new(name: &str, fixture: Option<&str>) -> Self {
        let dir = workspace().join("scratch").join("priming-test").join(format!("{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let home = dir.join("nano");
        let repo = dir.join("small-repo");
        std::fs::create_dir_all(home.join("memory")).unwrap();
        copy_dir(&workspace().join("tests").join("fixtures").join("small-repo"), &repo);
        let want = fixture.map(|f| expected(f).0).unwrap_or_default();
        Self { dir, home, repo, want, got: Vec::new(), deadline: Some(5000) }
    }
    fn set(&self, key: &str, value: &str) {
        let p = self.home.join("settings.json");
        let mut m: BTreeMap<String, Value> = std::fs::read(&p).ok().and_then(|b| nmjson::parse(&b).ok()).and_then(|v| if let Value::Object(m) = v { Some(m) } else { None }).unwrap_or_default();
        m.insert(key.into(), s(value));
        std::fs::write(&p, Value::Object(m).encode().unwrap()).unwrap();
    }
    fn cmd(&self, args: &[&str]) -> Command {
        let mut c = Command::new(EXE);
        c.args(args).env("NANO_HOME", &self.home).env_remove("CLAUDE_PROJECT_DIR").env_remove("NANO_PRIME_DEADLINE_MS").current_dir(&self.dir);
        if let Some(d) = self.deadline {
            c.env("NANO_PRIME_DEADLINE_MS", d.to_string());
        }
        c
    }
    fn run(&self, args: &[&str]) -> String {
        let repo = self.repo.display().to_string();
        let mut a: Vec<&str> = args.to_vec();
        a.extend(["--project", repo.as_str()]);
        text(&self.cmd(&a).output().unwrap())
    }
    fn audit(&self, line: &str) {
        let p = self.home.join("memory").join("audit.jsonl");
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(p).unwrap();
        std::io::Write::write_all(&mut f, format!("{line}\n").as_bytes()).unwrap();
    }
    /// Feeds one hook input and returns what it printed (empty for silence), with its time.
    fn raw_hook(&self, input: &Value) -> (String, u128) {
        let event = get(input, "hook_event_name");
        let mut c = self.cmd(&["hook", &event]);
        c.env("CLAUDE_PROJECT_DIR", &self.repo).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        let t = std::time::Instant::now();
        let mut child = c.spawn().unwrap();
        std::io::Write::write_all(child.stdin.as_mut().unwrap(), input.encode().unwrap().as_bytes()).unwrap();
        let o = child.wait_with_output().unwrap();
        let ms = t.elapsed().as_millis();
        assert!(o.status.success() && o.stderr.is_empty(), "{event}: {}", text(&o));
        pause(2);
        (String::from_utf8_lossy(&o.stdout).trim().to_owned(), ms)
    }
    /// Feeds one hook input under a label and compares what it printed with the expected reply.
    fn hook(&mut self, label: &str, input: &Value) {
        let event = get(input, "hook_event_name");
        let (out, _) = self.raw_hook(input);
        if let Some(want) = self.want.get(label) {
            let want_out = if want == "-" {
                String::new()
            } else {
                obj(vec![("hookSpecificOutput", obj(vec![("hookEventName", s(&event)), ("additionalContext", s(want))]))]).encode().unwrap()
            };
            assert_eq!(out, want_out, "{label}: the hook printed something other than expected");
        }
        self.got.push((label.to_owned(), out));
    }
    fn base(&self, event: &str, session: &str) -> Vec<(&'static str, Value)> {
        vec![("hook_event_name", s(event)), ("session_id", s(session)), ("cwd", s(&self.repo.display().to_string()))]
    }
    fn file(&self, rel: &str) -> String {
        rel.split('/').fold(self.repo.clone(), |p, c| p.join(c)).display().to_string()
    }
    fn start(&mut self, label: &str, sess: &str, source: &str) {
        let mut f = self.base("SessionStart", sess);
        f.push(("source", s(source)));
        self.hook(label, &obj(f));
    }
    fn prompt(&mut self, label: &str, sess: &str, p: &str) {
        let mut f = self.base("UserPromptSubmit", sess);
        f.push(("prompt", s(p)));
        f.push(("prompt_id", s(&format!("{label}-{}", now_ms()))));
        self.hook(label, &obj(f));
    }
    fn fail(&mut self, label: &str, sess: &str, error: &str) {
        let mut f = self.base("PostToolUseFailure", sess);
        f.extend([("tool_name", s("Bash")), ("tool_input", obj(vec![("command", s("cargo test -p shop"))])), ("error", s(error))]);
        self.hook(label, &obj(f));
    }
    fn bash(&mut self, label: &str, sess: &str, cmd: &str, stdout: &str) {
        let mut f = self.base("PostToolUse", sess);
        f.extend([("tool_name", s("Bash")), ("tool_input", obj(vec![("command", s(cmd))])), ("tool_response", obj(vec![("stdout", s(stdout)), ("stderr", s(""))]))]);
        self.hook(label, &obj(f));
    }
    fn read(&mut self, label: &str, sess: &str, rel: &str, start: u64, count: u64) {
        let mut f = self.base("PostToolUse", sess);
        let file = self.file(rel);
        f.extend([
            ("tool_name", s("Read")),
            ("tool_input", obj(vec![("file_path", s(&file))])),
            ("tool_response", obj(vec![("type", s("text")), ("file", obj(vec![("filePath", s(&file)), ("startLine", Value::Number(start.to_string())), ("numLines", Value::Number(count.to_string()))]))])),
        ]);
        self.hook(label, &obj(f));
    }
    fn edit_input(&self, event: &str, sess: &str, use_id: &str, rel: &str, old: &str, new: &str) -> Value {
        let mut f = self.base(event, sess);
        f.extend([
            ("tool_name", s("Edit")),
            ("tool_use_id", s(use_id)),
            ("tool_input", obj(vec![("file_path", s(&self.file(rel))), ("old_string", s(old)), ("new_string", s(new))])),
        ]);
        obj(f)
    }
    /// An edit as Claude Code makes it: the before hook, the change, the after hook.
    fn edit(&mut self, label: &str, sess: &str, use_id: &str, rel: &str, old: &str, new: &str) {
        self.hook(&format!("{label}.pre"), &self.edit_input("PreToolUse", sess, use_id, rel, old, new));
        self.change(rel, old, new);
        self.hook(&format!("{label}.post"), &self.edit_input("PostToolUse", sess, use_id, rel, old, new));
    }
    fn change(&self, rel: &str, old: &str, new: &str) {
        let p = self.file(rel);
        let t = std::fs::read_to_string(&p).unwrap();
        assert!(t.contains(old), "{rel} lacks {old}");
        std::fs::write(&p, t.replacen(old, new, 1)).unwrap();
    }
    fn stop(&mut self, label: &str, sess: &str, again: bool) {
        let mut f = self.base("Stop", sess);
        f.push(("stop_hook_active", Value::Bool(again)));
        self.hook(label, &obj(f));
    }
    fn memory(&self) -> PathBuf {
        let projects = self.home.join("memory").join("projects");
        std::fs::read_dir(&projects).unwrap().flatten().map(|e| e.path()).find(|p| p.is_dir()).unwrap()
    }
    fn ledger_rows(&self) -> Vec<Value> {
        std::fs::read_to_string(self.memory().join("evidence.jsonl")).unwrap_or_default().lines().map(|l| nmjson::parse(l.as_bytes()).unwrap()).collect()
    }
    fn situation(&self) -> String {
        std::fs::read_to_string(self.memory().join("situation.jsonl")).unwrap_or_default()
    }
    /// The prime and outcome rows of the ledger, one line each.
    fn ledger_lines(&self) -> Vec<String> {
        let rows = self.ledger_rows();
        let mut primes: Vec<String> = Vec::new();
        let mut out = Vec::new();
        for r in &rows {
            let place_of = |p: &Value| {
                let l = p.get("lines").unwrap().array().unwrap();
                let (a, b) = (l[0].integer().unwrap(), l[1].integer().unwrap());
                if a == b { format!("{}:{a}", get(p, "path")) } else { format!("{}:{a}-{b}", get(p, "path")) }
            };
            match get(r, "row").as_str() {
                "outcome" => {
                    let places: Vec<String> = r.get("places").unwrap().array().unwrap().iter().map(place_of).collect();
                    let tail = match r.get("failure") {
                        Ok(f) => format!("failure={} {} {} {}:{} {}", get(f, "check"), get(f, "pkg"), get(f, "code"), get(f, "file"), get(f, "line"), get(f, "msg")),
                        Err(_) => format!("question={} ask={}", get(r, "question"), if get(r, "ask_sha").len() == 16 { "hashed" } else { "missing" }),
                    };
                    out.push(format!("outcome {} {} {} {} {} {tail}", get(r, "kind"), get(r, "state"), get(r, "strength"), get(r, "rule"), places.join(",")));
                }
                "primed" => {
                    primes.push(get(r, "pack"));
                    let places: Vec<String> = r.get("places").unwrap().array().unwrap().iter().map(|p| p.string().unwrap().to_owned()).collect();
                    let from: Vec<String> = r.get("from").unwrap().array().unwrap().iter().map(|p| p.string().unwrap().to_owned()).map(|f| if f.starts_with("oc:") { "memory".into() } else { f }).collect();
                    out.push(format!("primed {} {} {} {} from={}", get(r, "kind"), get(r, "hook"), get(r, "mode"), places.join(","), from.join(",")));
                }
                "evidence" if get(r, "pack").starts_with("prime:") => {
                    let k = primes.iter().position(|p| *p == get(r, "pack")).map_or(0, |i| i + 1);
                    let (tool, at) = match r.get("event") {
                        Ok(e @ Value::Object(_)) => {
                            let tool = get(e, "tool");
                            let tool = if tool.is_empty() || tool == "-" { get(e, "event") } else { tool };
                            let at = match get(e, "place").as_str() {
                                "-" => get(e, "check"),
                                p => p.to_owned(),
                            };
                            (tool, at)
                        }
                        _ => ("-".into(), "-".into()),
                    };
                    out.push(format!(
                        "evidence n{k} {} {} {} steps={} {tool} {at} closed={}",
                        get(r, "rule"),
                        get(r, "state"),
                        get(r, "strength"),
                        get(r, "steps_after"),
                        get(r, "closed")
                    ));
                }
                _ => {}
            }
        }
        out
    }
    fn compare_ledger(&self, fixture: &str) {
        let (_, want) = expected(fixture);
        let got = self.ledger_lines();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g, w, "{fixture}: ledger line {} differs", i + 1);
        }
        assert_eq!(got.len(), want.len(), "{fixture}: {} ledger lines, expected {}:\n{}", got.len(), want.len(), got.join("\n"));
        let j = self.run(&["ledger", "--json"]);
        let v = nmjson::parse(j.trim().as_bytes()).unwrap();
        assert!(matches!(v.get("chain").unwrap().get("intact"), Ok(Value::Bool(true))), "{j}");
    }
    fn done(self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// A session in which `cargo test -p shop` fails with E0308 at line 18, line 18 is edited, and the
/// test passes: the ledger gets one verified `fixed` outcome at its stop.
fn fix_session(p: &mut Place, sess: &str) {
    p.start("a.start", sess, "startup");
    p.prompt("a.prompt", sess, "the checkout test fails, fix it");
    p.fail("a.fail", sess, E0308_18);
    p.edit("a.edit", sess, "u-a1", LIB, LINE18, LINE18_FIXED);
    p.bash("a.pass", sess, "cargo test -p shop", PASS);
    p.stop("a.stop", sess, false);
}

#[test]
fn failure_fix_fires_on_its_trigger_only_and_once() {
    let mut p = Place::new("failure-fix", Some("failure-fix.expected.txt"));
    fix_session(&mut p, "s-a");
    p.start("b.start", "s-b", "startup");
    p.fail("b.fail", "s-b", E0308_18);
    p.fail("b.other", "s-b", "Exit code 101\nerror[E0599]: no method named `total` found\n  --> crates/shop/src/lib.rs:18:9");
    p.fail("b.again", "s-b", E0308_18);
    p.bash("b.build", "s-b", "cargo build -p shop", "Finished `release` profile");
    p.stop("b.stop", "s-b", false);
    p.compare_ledger("failure-fix.expected.txt");
    // The prime rides on its hook's record: kind, places, sources; never its text.
    let sit = p.situation();
    assert_eq!(sit.matches("\"prime\":").count(), 1, "{sit}");
    assert!(!sit.contains("Those lines are unchanged"), "the prime text was stored");
    let memo = std::fs::read_to_string(p.memory().join("primes").join("s-b.jsonl")).unwrap();
    assert_eq!(memo.lines().count(), 1, "{memo}");
    p.done();
}

#[test]
fn prime_followed_and_verified_shows_in_the_ledger() {
    let mut p = Place::new("followed", Some("followed.expected.txt"));
    fix_session(&mut p, "s-a");
    p.start("b.start", "s-b", "startup");
    p.prompt("b.prompt", "s-b", "the checkout test fails again");
    p.fail("b.fail", "s-b", E0308_18);
    p.read("b.read", "s-b", LIB, 13, 7);
    p.edit("b.edit", "s-b", "u-b2", LIB, LINE18_FIXED, LINE18);
    p.bash("b.pass", "s-b", "cargo test -p shop", PASS);
    p.stop("b.stop", "s-b", false);
    p.prompt("b.thanks", "s-b", "thanks, that works");
    p.stop("b.stop2", "s-b", false);
    p.compare_ledger("followed.expected.txt");
    let session = p.run(&["session", "--session", "s-b"]);
    assert!(session.contains("primes given by the Nano (2):") && session.contains("failure-fix at PostToolUseFailure"), "{session}");
    p.done();
}

#[test]
fn prime_ignored_shows_as_passed_over() {
    let mut p = Place::new("passed-over", Some("passed-over.expected.txt"));
    fix_session(&mut p, "s-a");
    p.start("c.start", "s-c", "startup");
    p.fail("c.fail", "s-c", E0308_18);
    p.read("c.read", "s-c", "crates/inventory/src/lib.rs", 1, 21);
    p.stop("c.stop", "s-c", false);
    p.prompt("c.ok", "s-c", "ok, leave it");
    p.stop("c.stop2", "s-c", false);
    p.compare_ledger("passed-over.expected.txt");
    p.done();
}

#[test]
fn verified_answer_becomes_a_memory_and_is_recalled() {
    let mut p = Place::new("prompt-recall", Some("prompt-recall.expected.txt"));
    let (want, _) = expected("prompt-recall.expected.txt");
    p.start("a.start", "s-a", "startup");
    p.prompt("a.prompt", "s-a", "checkout charges the wrong total");
    // The context pack as the serving process audits it, then Claude Code's call as its hook sees it.
    let label = {
        let root = nmsubs_key(&p.repo);
        p.audit(&format!(
            r#"{{"id":"a1","kind":"tool","tool":"nm_context_pack","project":"{root}","mode":"plain","via":"serve","session":"9999-{}","state":"done","ms":1,"time_ms":{},"ok":true,"episode":"e-1","argument":"{ASK}","sources":["crates/shop/src/lib.rs:13-19","crates/inventory/src/lib.rs:1-21"],"steps":[]}}"#,
            now_ms() - 10_000,
            now_ms()
        ));
        root
    };
    pause(3);
    let mut f = p.base("PostToolUse", "s-a");
    f.extend([
        ("tool_name", s("mcp__neuromorph-nano__nm_context_pack")),
        ("tool_use_id", s("u-pack")),
        ("tool_input", obj(vec![("question", s(ASK))])),
        ("tool_response", Value::Array(vec![obj(vec![("type", s("text")), ("text", s(r#"{"episode":"e-1","text":"pack"}"#))])])),
    ]);
    p.hook("a.pack", &obj(f));
    p.read("a.read", "s-a", LIB, 13, 7);
    p.edit("a.edit", "s-a", "u-a1", LIB, LINE18, LINE18_FIXED);
    p.bash("a.pass", "s-a", "cargo test -p shop", PASS);
    p.stop("a.stop", "s-a", false);
    p.prompt("a.thanks", "s-a", "thanks, that works");
    p.stop("a.stop2", "s-a", false);
    let outcome = p.ledger_lines().into_iter().find(|l| l.starts_with("outcome answered")).unwrap_or_default();
    assert_eq!(outcome, want["outcome"], "the answered outcome");
    // The ledger keeps the question as a hash only, and no file content.
    let ledger = std::fs::read_to_string(p.memory().join("evidence.jsonl")).unwrap();
    assert!(!ledger.contains(ASK) && !ledger.contains("unit_price_cents"), "{ledger}");
    assert!(p.memory().ends_with(&label));
    p.start("b.start", "s-b", "startup");
    p.prompt("b.similar", "s-b", "where is the checkout total computed in the shop crate?");
    p.prompt("b.unrelated", "s-b", "add a licensing section to the README");
    p.prompt("b.similar.again", "s-b", "where is the checkout total computed in the shop crate?");
    // Recall by command, before and after the journal takes the outcome in.
    let before = p.run(&["outcomes", "--recall", "how is the checkout total computed"]);
    assert!(before.contains("recalled for") && before.contains(&want["recall.before"]), "{before}");
    let imported = p.run(&["outcomes", "--import"]);
    assert!(imported.contains(&want["import"]), "{imported}");
    let again = p.run(&["outcomes", "--import"]);
    assert!(again.contains(&want["import.again"]), "{again}");
    let after = p.run(&["outcomes", "--recall", "how is the checkout total computed"]);
    assert!(after.contains(&want["recall.after"]) && !after.contains("not yet in the journal"), "{after}");
    // The journal's view holds the memory under its own topic; nothing else writes that file.
    let view = std::fs::read_to_string(p.memory().join("2-active").join("outcomes").join("items.json")).unwrap();
    assert!(view.contains("\"topic\": \"outcomes\"") && view.contains("where is the checkout total computed"), "{view}");
    assert!(!view.contains("unit_price_cents"), "file contents reached memory");
    p.start("c.start", "s-c", "startup");
    p.prompt("c.similar", "s-c", "how is the checkout total computed?");
    p.done();
}

fn nmsubs_key(repo: &Path) -> String {
    nmtype::project::key(&nmsubs::repo::canon(repo))
}

#[test]
fn stale_source_gives_no_prime() {
    let mut p = Place::new("stale", None);
    fix_session(&mut p, "s-a");
    // The fixed lines change after the fix was verified.
    p.change(LIB, LINE18_FIXED, "    Some(unit_price_cents(&item.code) * u64::from(n) + 1)");
    p.start("b.start", "s-b", "startup");
    p.fail("b.fail", "s-b", E0308_18);
    assert!(p.got.iter().all(|(_, o)| o.is_empty()), "{:?}", p.got);
    assert!(!p.situation().contains("\"prime\":"));
    p.done();
}

#[test]
fn priming_off_gives_nothing() {
    let mut p = Place::new("off", None);
    p.set("priming", "off");
    fix_session(&mut p, "s-a");
    p.start("b.start", "s-b", "startup");
    p.fail("b.fail", "s-b", E0308_18);
    p.edit("b.edit", "s-b", "u-b1", LIB, LINE18_FIXED, LINE18);
    p.start("b.compact", "s-b", "compact");
    assert!(p.got.iter().all(|(_, o)| o.is_empty()), "{:?}", p.got);
    assert!(!p.situation().contains("\"prime\":"));
    // Memory still keeps the verified fix: priming off silences, it does not forget.
    assert!(p.ledger_lines().iter().any(|l| l.starts_with("outcome fixed verified")));
    p.done();
}

#[test]
fn each_prime_once_over_a_long_session() {
    let mut p = Place::new("budget", None);
    // Five failures, each fixed by an edit of its own line.
    let fixes = [
        ("E0001", 7, "        \"APPLE\" => 40,", "        \"APPLE\" => 41,"),
        ("E0002", 8, "        \"PEAR\" => 55,", "        \"PEAR\" => 56,"),
        ("E0003", 9, "        _ => 100,", "        _ => 101,"),
        ("E0004", 16, "        return None;", "        return None ;"),
        ("E0005", 18, LINE18, LINE18_FIXED),
    ];
    let err = |code: &str, line: u32| format!("Exit code 101\nerror[{code}]: broken\n  --> crates/shop/src/lib.rs:{line}:5");
    p.start("a.start", "s-a", "startup");
    for (i, (code, line, old, new)) in fixes.iter().enumerate() {
        p.fail(&format!("a.fail{i}"), "s-a", &err(code, *line));
        p.edit(&format!("a.edit{i}"), "s-a", &format!("u-a{i}"), LIB, old, new);
        p.bash(&format!("a.pass{i}"), "s-a", "cargo test -p shop", PASS);
    }
    p.stop("a.stop", "s-a", false);
    assert_eq!(p.ledger_lines().iter().filter(|l| l.starts_with("outcome fixed verified direct")).count(), 5, "{:?}", p.ledger_lines());
    assert!(p.got.iter().all(|(_, o)| o.is_empty()), "no prime without memory: {:?}", p.got);
    // A long session under quiet: every failure again, with reads, prompts and passes between.
    p.got.clear();
    p.start("b.start", "s-b", "startup");
    for round in 0..3 {
        for (i, (code, line, _, _)) in fixes.iter().enumerate() {
            p.prompt(&format!("b.prompt{round}{i}"), "s-b", &format!("round {round} step {i}: keep going"));
            p.read(&format!("b.read{round}{i}"), "s-b", "crates/inventory/src/lib.rs", 1, 21);
            p.fail(&format!("b.fail{round}{i}"), "s-b", &err(code, *line));
            p.bash(&format!("b.build{round}{i}"), "s-b", "cargo build -p shop", "Finished");
        }
    }
    p.stop("b.stop", "s-b", false);
    let given: Vec<&(String, String)> = p.got.iter().filter(|(_, o)| !o.is_empty()).collect();
    let labels: Vec<&str> = given.iter().map(|(l, _)| l.as_str()).collect();
    assert_eq!(labels, ["b.fail00", "b.fail01", "b.fail02", "b.fail03", "b.fail04"], "quiet gives each failure's prime once in a session, no quota");
    assert_eq!(p.got.len(), 1 + 3 * 5 * 4 + 1);
    // Under `on`: all five, each once, in a session of the same length.
    p.set("priming", "on");
    p.got.clear();
    p.start("c.start", "s-c", "startup");
    for round in 0..3 {
        for (i, (code, line, _, _)) in fixes.iter().enumerate() {
            p.fail(&format!("c.fail{round}{i}"), "s-c", &err(code, *line));
        }
    }
    // Under `on` the start also says what the previous session left open (its failing test).
    let labels: Vec<String> = p.got.iter().filter(|(_, o)| !o.is_empty()).map(|(l, _)| l.clone()).collect();
    assert_eq!(labels, ["c.start", "c.fail00", "c.fail01", "c.fail02", "c.fail03", "c.fail04"], "on gives the start's note and each of the five once");
    assert!(p.got[0].1.contains("the previous Claude Code session in this project left open: failing cargo-test shop"), "{}", p.got[0].1);
    for (_, o) in p.got.iter().filter(|(_, o)| !o.is_empty()) {
        let t = nmjson::parse(o.as_bytes()).unwrap().get("hookSpecificOutput").unwrap().get("additionalContext").unwrap().string().unwrap().to_owned();
        assert!(!t.ends_with("..."), "a prime is given whole: {t}");
    }
    p.done();
}

#[test]
fn edit_impact_resume_and_open_loop() {
    let mut p = Place::new("loops", Some("loops.expected.txt"));
    // A signature change finds its caller; a body edit with nothing in memory is silent.
    let sig = "pub fn checkout(item: &mut StockItem, n: u32) -> Option<u64> {";
    let input = p.edit_input("PreToolUse", "s-e", "u-e1", LIB, sig, "pub fn checkout(item: &mut StockItem, n: u32, discount: u64) -> Option<u64> {");
    p.hook("caller", &input);
    let input = p.edit_input("PreToolUse", "s-e", "u-e2", LIB, "        return None;", "        return None ;");
    p.hook("body-edit", &input);
    // Resume after compaction: what was in play.
    p.start("r.start", "s-r", "startup");
    p.prompt("r.prompt", "s-r", "make checkout reject zero units");
    p.edit("r.edit", "s-r", "u-r1", LIB, "    if !take_from_shelf(item, n) {", "    if n == 0 || !take_from_shelf(item, n) {");
    p.fail("r.fail", "s-r", "Exit code 101\nerror[E0308]: mismatched types\n  --> crates/shop/src/lib.rs:15:5");
    let mut pre = p.base("PreCompact", "s-r");
    pre.push(("trigger", s("auto")));
    p.hook("r.precompact", &obj(pre));
    p.start("compact", "s-r", "compact");
    p.start("resume", "s-r", "resume");
    p.start("clear", "s-r", "clear");
    // A new session: silent under quiet; under on, what the previous session left open.
    p.start("startup.quiet", "s-q", "startup");
    p.set("priming", "on");
    p.start("startup.on", "s-o", "startup");
    // Open loop at a stop: quiet says nothing; on names the change; never twice; silent once tested.
    p.set("priming", "quiet");
    p.prompt("l.prompt", "s-l", "tidy the price table");
    p.edit("l.edit", "s-l", "u-l1", LIB, "        \"APPLE\" => 40,", "        \"APPLE\" => 45,");
    p.stop("open-loop.quiet", "s-l", false);
    p.set("priming", "on");
    p.stop("open-loop.on", "s-l", false);
    p.stop("open-loop.again", "s-l", true);
    p.bash("l.pass", "s-l", "cargo test -p shop", PASS);
    p.stop("open-loop.passed", "s-l", false);
    p.prompt("l.prompt2", "s-l", "now the pear price");
    p.edit("l.edit2", "s-l", "u-l2", LIB, "        \"PEAR\" => 55,", "        \"PEAR\" => 56,");
    p.fail("l.fail", "s-l", "Exit code 101\nerror[E0308]: mismatched types\n  --> crates/shop/src/lib.rs:8:5");
    p.stop("open-loop.failed", "s-l", false);
    for (k, _) in expected("loops.expected.txt").0 {
        assert!(p.got.iter().any(|(l, _)| *l == k), "step {k} was not run");
    }
    p.done();
}

#[test]
fn a_prime_not_ready_in_time_is_dropped() {
    let mut p = Place::new("late", None);
    // A deadline already passed when the hook starts: the caller search is not waited for.
    p.deadline = Some(0);
    let sig = "pub fn checkout(item: &mut StockItem, n: u32) -> Option<u64> {";
    let input = p.edit_input("PreToolUse", "s-l", "u-l1", LIB, sig, "pub fn checkout(item: &mut StockItem) -> Option<u64> {");
    p.hook("late", &input);
    assert!(p.got.iter().all(|(_, o)| o.is_empty()), "{:?}", p.got);
    let sit = p.situation();
    assert!(sit.contains("\"prime_late\":\"edit-impact\"") && !sit.contains("\"prime\":"), "{sit}");
    p.done();
}

#[test]
fn hook_time_with_priming_on() {
    let mut q = Place::new("timing", None);
    fix_session(&mut q, "s-a");
    // The program's own deadline from here on.
    q.deadline = None;
    q.set("priming", "on");
    let mut kinds: BTreeMap<&str, Vec<u128>> = BTreeMap::new();
    let mut given = 0;
    for i in 0..30 {
        let sess = format!("t-{i}");
        let mut f = q.base("PostToolUseFailure", &sess);
        f.extend([("tool_name", s("Bash")), ("tool_input", obj(vec![("command", s("cargo test -p shop"))])), ("error", s(E0308_18))]);
        let (o, ms) = q.raw_hook(&obj(f));
        given += usize::from(!o.is_empty());
        kinds.entry("failure (prime given)").or_default().push(ms);
        let mut f = q.base("UserPromptSubmit", &sess);
        f.push(("prompt", s("where is the checkout total computed?")));
        kinds.entry("prompt (recall looked up)").or_default().push(q.raw_hook(&obj(f)).1);
        let input = q.edit_input("PreToolUse", &sess, "u-t", LIB, LINE18_FIXED, LINE18);
        kinds.entry("before an edit (memory looked up)").or_default().push(q.raw_hook(&input).1);
        let file = q.file(LIB);
        let mut f = q.base("PostToolUse", &sess);
        f.extend([("tool_name", s("Read")), ("tool_input", obj(vec![("file_path", s(&file))]))]);
        kinds.entry("read (no prime rule)").or_default().push(q.raw_hook(&obj(f)).1);
    }
    // Each new session gets the failure-fix prime once, unless it was not ready in time.
    let late = q.situation().matches("\"prime_late\":\"failure-fix\"").count();
    assert_eq!(given + late, 30, "{given} given, {late} late");
    let mut report = format!("hook time with priming on, 30 runs each, process start included (ms); failure-fix given {given}, dropped as late {late}:\n");
    for (k, mut v) in kinds {
        v.sort_unstable();
        report.push_str(&format!("  {k}: median {}, worst {}\n", v[v.len() / 2], v[v.len() - 1]));
    }
    println!("{report}");
    let _ = std::fs::write(workspace().join("scratch").join("priming-test").join("hook-times.txt"), &report);
    q.done();
}
