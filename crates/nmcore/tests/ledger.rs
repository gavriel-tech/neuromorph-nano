//! Product tests of the session state and the evidence ledger, run through the real program on
//! copies of the small fixture repository under `scratch/ledger-test/`, with a test Nano folder
//! there (`NANO_HOME`). Each scripted session is fed to `neuromorph-nano hook <Event>` one event
//! at a time; the audit rows of the context packs the Nano served are written into the test Nano
//! folder's audit log in the shape a serving process writes them. The ledger the program then
//! holds is compared exactly with the rows written down beforehand in `tests/fixtures/ledger/`
//! (times, hashes and the machine's hardware left out). Claude Code itself is not run, and
//! nothing outside `scratch/` is written.
#![allow(clippy::unwrap_used)]
use nmjson::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

const EXE: &str = env!("CARGO_BIN_EXE_neuromorph-nano");
const ASK: &str = "where is the checkout total computed";
const LIB: &str = "crates/shop/src/lib.rs";
const OLD: &str = "    Some(unit_price_cents(&item.code) * u64::from(n))";
const NEW: &str = "    Some(unit_price_cents(&item.code) * u64::from(n) + 0)";

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").canonicalize().unwrap()
}
fn now_ms() -> u64 {
    u64::try_from(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()).unwrap()
}
/// Waits a few milliseconds, so two writes never share a millisecond.
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
fn esc(p: &Path) -> String {
    p.display().to_string().replace('\\', "\\\\")
}
fn text(o: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}

struct Place {
    dir: PathBuf,
    home: PathBuf,
    repo: PathBuf,
    /// The serving process's start time in the audit rows.
    start: u64,
}

impl Place {
    fn new(name: &str) -> Self {
        let dir = workspace().join("scratch").join("ledger-test").join(format!("{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let home = dir.join("nano");
        let repo = dir.join("small-repo");
        std::fs::create_dir_all(home.join("memory")).unwrap();
        copy_dir(&workspace().join("tests").join("fixtures").join("small-repo"), &repo);
        let p = Self { dir, home, repo, start: now_ms() - 10_000 };
        // Settings changed before the server started apply; a precision change after its start
        // does not; the answer mode applies from the next call.
        p.audit(&format!(r#"{{"id":"s1","kind":"setting","key":"precision.reranker","from":"default","to":"8","time_ms":{}}}"#, p.start - 20_000));
        p.audit(&format!(r#"{{"id":"s2","kind":"setting","key":"precision.base","from":"default","to":"4","time_ms":{}}}"#, p.start + 1000));
        p.audit(&format!(r#"{{"id":"s3","kind":"setting","key":"mode","from":"full","to":"fast","time_ms":{}}}"#, p.start + 2000));
        p
    }
    fn cmd(&self, args: &[&str]) -> Command {
        let mut c = Command::new(EXE);
        c.args(args).env("NANO_HOME", &self.home).env_remove("CLAUDE_PROJECT_DIR").current_dir(&self.dir);
        c
    }
    fn run(&self, args: &[&str]) -> Output {
        let repo = self.repo.display().to_string();
        let mut a: Vec<&str> = args.to_vec();
        a.extend(["--project", repo.as_str()]);
        self.cmd(&a).output().unwrap()
    }
    fn audit(&self, line: &str) {
        let p = self.home.join("memory").join("audit.jsonl");
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(p).unwrap();
        std::io::Write::write_all(&mut f, format!("{line}\n").as_bytes()).unwrap();
    }
    fn hook(&self, event: &str, input: &str) {
        let mut c = self.cmd(&["hook", event]);
        c.env("CLAUDE_PROJECT_DIR", &self.repo).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut child = c.spawn().unwrap();
        std::io::Write::write_all(child.stdin.as_mut().unwrap(), input.as_bytes()).unwrap();
        let o = child.wait_with_output().unwrap();
        assert!(o.status.success() && o.stdout.is_empty() && o.stderr.is_empty(), "{event}: {}", text(&o));
        pause(2);
    }
    fn memory(&self) -> PathBuf {
        let projects = self.home.join("memory").join("projects");
        std::fs::read_dir(&projects).unwrap().flatten().map(|e| e.path()).find(|p| p.is_dir()).unwrap()
    }
    fn label(&self) -> String {
        self.memory().file_name().unwrap().to_string_lossy().into_owned()
    }
    fn cwd(&self) -> String {
        esc(&self.repo)
    }
    fn file(&self, rel: &str) -> String {
        esc(&rel.split('/').fold(self.repo.clone(), |p, c| p.join(c)))
    }
    fn start_session(&self, s: &str, source: &str) {
        self.hook("SessionStart", &format!(r#"{{"hook_event_name":"SessionStart","session_id":"{s}","source":"{source}","cwd":"{}"}}"#, self.cwd()));
    }
    fn prompt(&self, s: &str, t: &str) {
        self.hook("UserPromptSubmit", &format!(r#"{{"hook_event_name":"UserPromptSubmit","session_id":"{s}","prompt":"{t}","cwd":"{}"}}"#, self.cwd()));
    }
    /// The audit row of a context pack, as the serving process writes it at the end of the call.
    fn served(&self, id: &str, episode: &str) {
        let label = self.label();
        let start = self.start;
        let time = now_ms();
        self.audit(&format!(
            r#"{{"id":"{id}","kind":"tool","tool":"nm_context_pack","project":"{label}","mode":"whole","via":"serve","session":"9999-{start}","state":"done","ms":1,"time_ms":{time},"ok":true,"episode":"{episode}","argument":"{ASK}","argument_field":"question","withheld":false,"sources":["crates/shop/src/lib.rs:13-19","crates/inventory/src/lib.rs:1-21"],"steps":[{{"part":"nmapi.mcp","ms":1}},{{"part":"nmapi.whole","ms":1}},{{"part":"nmbrain.slot#reranker","ms":1}},{{"part":"nmbrain.slot#embedder","ms":1}},{{"part":"nmapi.assemble","ms":0}}]}}"#
        ));
        pause(3);
    }
    /// Claude Code's call of the Nano's context pack tool, as its after-the-call hook reports it.
    fn pack(&self, s: &str, use_id: &str, episode: &str) {
        let reply = format!(r#"{{\"episode\":\"{episode}\",\"text\":\"pack\"}}"#);
        self.hook(
            "PostToolUse",
            &format!(
                r#"{{"hook_event_name":"PostToolUse","session_id":"{s}","tool_name":"mcp__neuromorph-nano__nm_context_pack","tool_use_id":"{use_id}","tool_input":{{"question":"{ASK}"}},"tool_response":[{{"type":"text","text":"{reply}"}}],"cwd":"{}"}}"#,
                self.cwd()
            ),
        );
    }
    fn read(&self, s: &str, rel: &str, span: Option<(u64, u64)>) {
        let file = self.file(rel);
        let resp = span.map_or_else(String::new, |(a, k)| format!(r#","tool_response":{{"type":"text","file":{{"filePath":"{file}","startLine":{a},"numLines":{k}}}}}"#));
        self.hook("PostToolUse", &format!(r#"{{"hook_event_name":"PostToolUse","session_id":"{s}","tool_name":"Read","tool_input":{{"file_path":"{file}"}}{resp},"cwd":"{}"}}"#, self.cwd()));
    }
    fn edit(&self, event: &str, s: &str, use_id: &str) {
        self.hook(
            event,
            &format!(
                r#"{{"hook_event_name":"{event}","session_id":"{s}","tool_name":"Edit","tool_use_id":"{use_id}","tool_input":{{"file_path":"{}","old_string":"{OLD}","new_string":"{NEW}"}},"cwd":"{}"}}"#,
                self.file(LIB),
                self.cwd()
            ),
        );
    }
    /// The edit itself, as Claude Code would make it between its two hooks.
    fn change_file(&self) {
        let p = self.repo.join("crates").join("shop").join("src").join("lib.rs");
        let t = std::fs::read_to_string(&p).unwrap();
        assert!(t.contains(OLD));
        std::fs::write(&p, t.replace(OLD, NEW)).unwrap();
    }
    fn bash(&self, s: &str, cmd: &str, stdout: &str) {
        self.hook(
            "PostToolUse",
            &format!(r#"{{"hook_event_name":"PostToolUse","session_id":"{s}","tool_name":"Bash","tool_input":{{"command":"{cmd}"}},"tool_response":{{"stdout":"{stdout}","stderr":""}},"cwd":"{}"}}"#, self.cwd()),
        );
    }
    fn bash_fail(&self, s: &str, cmd: &str, error: &str) {
        self.hook(
            "PostToolUseFailure",
            &format!(r#"{{"hook_event_name":"PostToolUseFailure","session_id":"{s}","tool_name":"Bash","tool_input":{{"command":"{cmd}"}},"error":"{error}","cwd":"{}"}}"#, self.cwd()),
        );
    }
    fn stop(&self, s: &str) {
        self.hook("Stop", &format!(r#"{{"hook_event_name":"Stop","session_id":"{s}","stop_hook_active":false,"cwd":"{}"}}"#, self.cwd()));
    }
    fn ledger_text(&self) -> String {
        std::fs::read_to_string(self.memory().join("evidence.jsonl")).unwrap_or_default()
    }
    fn ledger_rows(&self) -> usize {
        self.ledger_text().lines().count()
    }
    fn done(self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// A ledger row without what changes from run to run: the time written, the times of events, the
/// hashes, and the hardware (checked here to be present and the same across rows, then named HW).
fn normal(v: &Value, hw: &mut Option<String>) -> String {
    let Value::Object(m) = v else { panic!("row is not an object: {v:?}") };
    let mut m = m.clone();
    for k in ["t", "at", "prev", "h"] {
        m.remove(k);
    }
    let id = match m.get("hardware") {
        Some(Value::String(id)) => id.clone(),
        Some(h @ Value::Object(_)) => {
            assert!(h.get("used").is_ok() && h.get("cpu").is_ok(), "served hardware {h:?}");
            h.get("id").unwrap().string().unwrap().to_owned()
        }
        other => panic!("hardware {other:?}"),
    };
    assert!(!id.is_empty());
    assert_eq!(hw.get_or_insert_with(|| id.clone()), &id, "hardware differs between rows");
    m.insert("hardware".into(), Value::String("HW".into()));
    if let Some(Value::Object(e)) = m.get_mut("event") {
        e.remove("at");
    }
    Value::Object(m).encode().unwrap()
}

/// The ledger the program holds equals the expected rows exactly, and its chain is intact.
fn compare(place: &Place, expected: &str) {
    let got = place.ledger_text();
    let want_text = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("ledger").join(expected)).unwrap();
    let mut hw = None;
    let got: Vec<String> = got.lines().map(|l| normal(&nmjson::parse(l.as_bytes()).unwrap(), &mut hw)).collect();
    let want: Vec<String> = want_text.lines().map(|l| nmjson::parse(l.as_bytes()).unwrap().encode().unwrap()).collect();
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "{expected}: row {} differs", i + 1);
    }
    assert_eq!(got.len(), want.len(), "{expected}: {} rows, expected {}", got.len(), want.len());
    let j = place.run(&["ledger", "--json"]);
    let v = nmjson::parse(&j.stdout).unwrap();
    assert!(matches!(v.get("chain").unwrap().get("intact"), Ok(Value::Bool(true))), "{}", text(&j));
}

/// Session start, the prompt, and the pack served and received.
fn opening(place: &Place, s: &str, tag: &str, first: &str) {
    place.start_session(s, "startup");
    place.prompt(s, first);
    place.served(&format!("a1-{tag}"), &format!("e-{tag}"));
    place.pack(s, "u-pack", &format!("e-{tag}"));
}

#[test]
fn edit_then_test_passes() {
    let place = Place::new("edit-pass");
    let s = "s-edit-pass";
    opening(&place, s, "edit-pass", "checkout charges the wrong total");
    place.read(s, LIB, Some((13, 7)));
    place.edit("PreToolUse", s, "u-e1");
    place.change_file();
    place.edit("PostToolUse", s, "u-e1");
    place.bash(s, "cargo test -p shop", "test result: ok. 3 passed; 0 failed; 0 ignored");
    place.stop(s);
    // The stop wrote the pack's served row at once; its evidence waits for its window to close.
    assert_eq!(place.ledger_rows(), 1, "{}", place.ledger_text());
    place.prompt(s, "thanks, that works");
    place.stop(s);
    compare(&place, "edit-pass.expected.jsonl");
    // The session state shows the task, the files with their lines, the check and the pack.
    let o = place.run(&["session"]);
    let out = text(&o);
    for want in ["task (prompt 2 of 2): \"thanks, that works\"", "crates/shop/src/lib.rs  read 13-19; edited 18 (1x)", "cargo-test  ok", "served by the Nano (1)", "pack a1-edit-pass (joined by episode)", "window closed (next turn ended)", "nothing open"] {
        assert!(out.contains(want), "no {want} in:\n{out}");
    }
    let j = nmjson::parse(&place.run(&["session", "--json"]).stdout).unwrap();
    assert_eq!(j.get("served").unwrap().array().unwrap().len(), 1);
    // Nothing of the prompt text, the command text or the file's content is in the ledger.
    let l = place.ledger_text();
    for planted in ["thanks, that works", "cargo test -p", "unit_price_cents", ASK] {
        assert!(!l.contains(planted), "{planted} is in the ledger");
    }
    place.done();
}

#[test]
fn edit_then_test_fails_and_is_undone() {
    let place = Place::new("edit-fail");
    let s = "s-edit-fail";
    opening(&place, s, "edit-fail", "checkout charges the wrong total");
    place.read(s, LIB, Some((13, 7)));
    place.edit("PreToolUse", s, "u-e1");
    place.change_file();
    place.edit("PostToolUse", s, "u-e1");
    place.bash_fail(s, "cargo test -p shop", "Exit code 101\\ntest result: FAILED. 2 passed; 1 failed; 0 ignored");
    place.bash(s, "git checkout crates/shop/src/lib.rs", "");
    place.stop(s);
    place.prompt(s, "still failing, look again");
    place.stop(s);
    compare(&place, "edit-fail.expected.jsonl");
    let out = text(&place.run(&["session"]));
    assert!(out.contains("failing check cargo-test shop"), "{out}");
    place.done();
}

#[test]
fn cited_lines_never_touched() {
    let place = Place::new("untouched");
    let s = "s-untouched";
    opening(&place, s, "untouched", "checkout charges the wrong total");
    place.read(s, "README.md", None);
    place.read(s, "crates/shop/tests/checkout_refund.rs", None);
    place.stop(s);
    place.prompt(s, "now add a test for refunds");
    place.stop(s);
    compare(&place, "untouched.expected.jsonl");
    place.done();
}

#[test]
fn next_prompt_corrects_course() {
    let place = Place::new("correction");
    let s = "s-correction";
    opening(&place, s, "correction", "checkout charges the wrong total");
    place.read(s, LIB, Some((1, 19)));
    place.stop(s);
    place.prompt(s, "No, that's the wrong place. The total is in inventory.");
    place.served("a2-correction", "e2-correction");
    place.pack(s, "u-pack2", "e2-correction");
    place.stop(s);
    compare(&place, "correction.expected.jsonl");
    place.done();
}

#[test]
fn no_following_events() {
    let place = Place::new("no-follow");
    let s = "s-no-follow";
    opening(&place, s, "no-follow", "checkout charges the wrong total");
    // No hook settles (no stop); the command writes the served row and leaves the window open.
    let o = place.run(&["ledger"]);
    assert!(o.status.success() && text(&o).contains("window open: a1-no-follow"), "{}", text(&o));
    assert_eq!(place.ledger_rows(), 1);
    // An hour later the window closes with what could not be judged recorded as unknown.
    let later = (now_ms() + 2 * 60 * 60 * 1000).to_string();
    let o = place.run(&["ledger", "--as-of", &later]);
    assert!(o.status.success(), "{}", text(&o));
    compare(&place, "no-follow.expected.jsonl");
    place.done();
}

#[test]
fn hooks_out_of_order() {
    let place = Place::new("out-of-order");
    let s = "s-out-of-order";
    place.start_session(s, "startup");
    place.prompt(s, "checkout charges the wrong total");
    place.served("a1-out-of-order", "e-out-of-order");
    // The Read's hook lands before the pack's own hook (the pack's hook process was slower).
    place.read(s, LIB, Some((13, 7)));
    place.pack(s, "u-pack", "e-out-of-order");
    // The edit's after-hook lands before its before-hook, which then finds the file changed.
    place.change_file();
    place.edit("PostToolUse", s, "u-e1");
    place.edit("PreToolUse", s, "u-e1");
    place.bash(s, "cargo test -p shop", "test result: ok. 3 passed; 0 failed; 0 ignored");
    place.stop(s);
    place.prompt(s, "great, thanks");
    place.stop(s);
    compare(&place, "out-of-order.expected.jsonl");
    place.done();
}

#[test]
fn two_sessions_interleaved() {
    let place = Place::new("interleaved");
    let (a, b) = ("sa-interleaved", "sb-interleaved");
    place.start_session(a, "startup");
    place.start_session(b, "startup");
    place.prompt(a, "checkout charges the wrong total");
    place.prompt(b, "tidy the shop crate");
    place.served("a1-interleaved", "e-interleaved");
    place.pack(a, "u-pack", "e-interleaved");
    // Session b reads, edits and tests the file session a was given; that is not a's evidence.
    place.read(b, LIB, Some((13, 7)));
    place.change_file();
    place.edit("PostToolUse", b, "u-b1");
    place.bash(b, "cargo test -p shop", "test result: ok. 3 passed; 0 failed; 0 ignored");
    place.read(a, "crates/shop/tests/checkout_refund.rs", None);
    place.stop(a);
    place.stop(b);
    place.prompt(a, "ok, next one");
    place.stop(a);
    compare(&place, "interleaved.expected.jsonl");
    let o = text(&place.run(&["session", "--session", b]));
    assert!(o.contains("served by the Nano (0)") && o.contains("edited 18"), "{o}");
    place.done();
}

#[test]
fn restart_in_the_middle() {
    let place = Place::new("restart");
    let s = "s-restart";
    opening(&place, s, "restart", "checkout charges the wrong total");
    place.read(s, LIB, Some((13, 7)));
    place.stop(s);
    assert_eq!(place.ledger_rows(), 1);
    // A ledger write cut off by a stop leaves half a line; Claude Code then resumes the session.
    let file = place.memory().join("evidence.jsonl");
    let mut f = std::fs::OpenOptions::new().append(true).open(&file).unwrap();
    std::io::Write::write_all(&mut f, b"{\"v\":1,\"seq\":2,\"row\":\"evid").unwrap();
    drop(f);
    place.start_session(s, "resume");
    place.change_file();
    place.edit("PostToolUse", s, "u-e1");
    place.bash(s, "cargo test -p shop", "test result: ok. 3 passed; 0 failed; 0 ignored");
    place.prompt(s, "perfect");
    place.stop(s);
    compare(&place, "restart.expected.jsonl");
    let torn = std::fs::read_to_string(place.memory().join("evidence.torn")).unwrap();
    assert!(torn.contains("\"row\":\"evid"), "{torn}");
    // Settling again writes nothing more (served, seven evidence rows, the outcome for memory).
    place.run(&["ledger"]);
    assert_eq!(place.ledger_rows(), 9);
    place.done();
}

#[test]
fn privacy_prompts_off_memory_off_and_clear() {
    let place = Place::new("privacy");
    let s = "s-privacy";
    std::fs::write(place.home.join("settings.json"), b"{\"prompts\":\"off\"}").unwrap();
    opening(&place, s, "privacy", "checkout charges the wrong total");
    place.read(s, LIB, Some((13, 7)));
    place.stop(s);
    place.prompt(s, "thanks, that works");
    place.stop(s);
    let l = place.ledger_text();
    assert!(l.contains("\"why\":\"the next prompt's text is not kept\""), "{l}");
    // A chain that was changed after writing is found and nothing more is written.
    std::fs::write(place.memory().join("evidence.jsonl"), l.replacen("\"read\"", "\"edit\"", 1)).unwrap();
    let o = place.run(&["ledger"]);
    assert!(!o.status.success() && text(&o).contains("changed after it was written"), "{}", text(&o));
    std::fs::write(place.memory().join("evidence.jsonl"), &l).unwrap();
    // Memory off: the command reads but writes nothing.
    place.prompt(s, "one more");
    place.served("a2-privacy", "e2-privacy");
    place.pack(s, "u-pack2", "e2-privacy");
    std::fs::write(place.home.join("settings.json"), b"{\"memory\":\"off\"}").unwrap();
    let before = place.ledger_text();
    let later = (now_ms() + 2 * 60 * 60 * 1000).to_string();
    let o = place.run(&["ledger", "--as-of", &later]);
    assert!(o.status.success() && text(&o).contains("memory is off, nothing is written"), "{}", text(&o));
    assert_eq!(place.ledger_text(), before);
    // Clearing removes the ledger, is audited, and what was cleared is not written again.
    std::fs::write(place.home.join("settings.json"), b"{}").unwrap();
    let o = place.run(&["ledger", "--clear"]);
    assert!(o.status.success() && text(&o).contains("rows removed"), "{}", text(&o));
    assert!(!place.memory().join("evidence.jsonl").exists());
    let audit = std::fs::read_to_string(place.home.join("memory").join("audit.jsonl")).unwrap();
    assert!(audit.contains("Evidence ledger cleared"), "{audit}");
    let o = place.run(&["ledger", "--as-of", &later]);
    assert!(o.status.success(), "{}", text(&o));
    assert_eq!(place.ledger_rows(), 0, "{}", place.ledger_text());
    place.done();
}
