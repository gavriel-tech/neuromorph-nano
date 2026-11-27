//! Product tests of connecting a repository to Claude Code and disconnecting it again, and of the
//! hook intake, run through the real program on copies of the small fixture repository under
//! `scratch/hooks-test/`, with a test Nano folder there (`NANO_HOME`). Nothing outside `scratch/`
//! is written; Claude Code itself is not run.
#![allow(clippy::unwrap_used)]
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

const EXE: &str = env!("CARGO_BIN_EXE_neuromorph-nano");

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").canonicalize().unwrap()
}

/// A fresh test place: a Nano folder and a copy of the fixture repository.
struct Place {
    dir: PathBuf,
    home: PathBuf,
    repo: PathBuf,
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

impl Place {
    fn new(name: &str) -> Self {
        let dir = workspace().join("scratch").join("hooks-test").join(format!("{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let home = dir.join("nano");
        let repo = dir.join("small-repo");
        std::fs::create_dir_all(&home).unwrap();
        copy_dir(&workspace().join("tests").join("fixtures").join("small-repo"), &repo);
        Self { dir, home, repo }
    }
    fn cmd(&self, args: &[&str]) -> Command {
        let mut c = Command::new(EXE);
        c.args(args).env("NANO_HOME", &self.home).env_remove("CLAUDE_PROJECT_DIR").current_dir(&self.dir);
        c
    }
    fn setup(&self, flags: &[&str]) -> Output {
        let repo = self.repo.display().to_string();
        let mut a = vec!["setup", "--project", repo.as_str()];
        a.extend_from_slice(flags);
        self.cmd(&a).output().unwrap()
    }
    fn hook(&self, event: &str, input: &str) -> Output {
        let mut c = self.cmd(&["hook", event]);
        c.env("CLAUDE_PROJECT_DIR", &self.repo).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut child = c.spawn().unwrap();
        std::io::Write::write_all(child.stdin.as_mut().unwrap(), input.as_bytes()).unwrap();
        child.wait_with_output().unwrap()
    }
    fn settings(&self) -> PathBuf {
        self.repo.join(".claude").join("settings.local.json")
    }
    /// The project memory folder the test Nano uses for the repository.
    fn memory(&self) -> PathBuf {
        let projects = self.home.join("memory").join("projects");
        std::fs::read_dir(&projects).unwrap().flatten().map(|e| e.path()).find(|p| p.is_dir()).unwrap()
    }
    fn done(self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn text(o: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("settings").join(name)).unwrap()
}

const SHAPES: &[&str] = &["others-and-comments.json", "crlf-tabs-empty-hooks.json", "one-line-empty-event.json", "bom-four-spaces.json", "empty-object.json"];

#[test]
fn write_twice_then_remove_gives_back_every_file_byte_for_byte() {
    let place = Place::new("reversible");
    std::fs::create_dir_all(place.repo.join(".claude")).unwrap();
    for name in SHAPES {
        let original = fixture(name);
        std::fs::write(place.settings(), &original).unwrap();
        let w = place.setup(&["--hooks", "--write"]);
        assert!(w.status.success(), "{name}: write failed: {}", text(&w));
        let written = std::fs::read(place.settings()).unwrap();
        assert_ne!(written, original, "{name}: write changed nothing");
        let s = String::from_utf8_lossy(&written);
        assert_eq!(s.matches("\"hook\"").count(), 9, "{name}: nine handlers expected in {s}");
        // The other entries are still there, byte for byte, in order.
        if *name == "others-and-comments.json" {
            for keep in ["/* a colleague's formatter, kept as is */", "prettier --write", "C:\\\\Tools\\\\neuromorph-nano.exe hook stop", "// trailing comma after this object"] {
                assert!(s.contains(keep), "{name}: lost {keep}");
            }
        }
        let again = place.setup(&["--hooks", "--write"]);
        assert!(again.status.success(), "{name}: second write failed: {}", text(&again));
        assert_eq!(std::fs::read(place.settings()).unwrap(), written, "{name}: a second write changed the file");
        assert!(text(&again).contains("already holds these entries"), "{name}: {}", text(&again));
        let st = place.setup(&["--status"]);
        assert!(text(&st).contains("9 handlers"), "{name}: status {}", text(&st));
        let r = place.setup(&["--remove"]);
        assert!(r.status.success(), "{name}: remove failed: {}", text(&r));
        assert!(text(&r).contains("removed 9 hook handlers"), "{name}: {}", text(&r));
        assert_eq!(std::fs::read(place.settings()).unwrap(), original, "{name}: not the original after remove");
    }
    // A backup of the file as it was is kept for every write and removal.
    let backups = std::fs::read_dir(place.memory().join("setup").join("backups")).unwrap().count();
    assert_eq!(backups, 2 * SHAPES.len());
    let audit = std::fs::read_to_string(place.home.join("memory").join("audit.jsonl")).unwrap();
    assert_eq!(audit.matches("\"kind\":\"setup\"").count(), 2 * SHAPES.len());
    place.done();
}

#[test]
fn no_file_before_means_no_file_after() {
    let place = Place::new("absent");
    let w = place.setup(&["--hooks", "--write"]);
    assert!(w.status.success(), "{}", text(&w));
    assert!(place.settings().is_file());
    let r = place.setup(&["--remove"]);
    assert!(r.status.success(), "{}", text(&r));
    assert!(!place.settings().exists(), "file left behind");
    assert!(!place.repo.join(".claude").exists(), "folder left behind");
    // A folder setup did not create stays, and the file goes.
    std::fs::create_dir_all(place.repo.join(".claude")).unwrap();
    std::fs::write(place.repo.join(".claude").join("settings.json"), b"{\"shared\": true}").unwrap();
    assert!(place.setup(&["--hooks", "--write"]).status.success());
    assert!(place.setup(&["--remove"]).status.success());
    assert!(!place.settings().exists());
    assert_eq!(std::fs::read(place.repo.join(".claude").join("settings.json")).unwrap(), b"{\"shared\": true}");
    place.done();
}

#[test]
fn a_file_that_is_not_json_is_refused_and_left_alone() {
    let place = Place::new("broken");
    std::fs::create_dir_all(place.repo.join(".claude")).unwrap();
    let original = fixture("broken.json.txt");
    std::fs::write(place.settings(), &original).unwrap();
    for flags in [&["--hooks", "--write"][..], &["--remove"][..]] {
        let o = place.setup(flags);
        assert!(!o.status.success(), "{flags:?} accepted a broken file");
        assert!(text(&o).contains("is not valid JSON"), "{flags:?}: {}", text(&o));
        assert!(text(&o).contains("nothing changed"), "{flags:?}: {}", text(&o));
        assert_eq!(std::fs::read(place.settings()).unwrap(), original);
    }
    assert!(!place.home.join("memory").join("projects").exists() || std::fs::read_dir(place.memory().join("setup").join("backups")).is_err());
    place.done();
}

#[test]
fn mcp_entry_goes_in_and_out_byte_for_byte() {
    let place = Place::new("mcp");
    let original = b"{\n  // shared servers\n  \"mcpServers\": {\n    \"other\": { \"command\": \"other-server\", \"args\": [] }\n  }\n}\n".to_vec();
    let f = place.repo.join(".mcp.json");
    std::fs::write(&f, &original).unwrap();
    assert!(place.setup(&["--write"]).status.success());
    let w = std::fs::read_to_string(&f).unwrap();
    assert!(w.contains("\"neuromorph-nano\"") && w.contains("// shared servers") && w.contains("other-server"));
    assert!(place.setup(&["--hooks", "--write"]).status.success());
    let r = place.setup(&["--remove"]);
    assert!(text(&r).contains("removed the MCP server entry"), "{}", text(&r));
    assert_eq!(std::fs::read(&f).unwrap(), original);
    assert!(!place.settings().exists());
    // With no .mcp.json before, none after.
    std::fs::remove_file(&f).unwrap();
    assert!(place.setup(&["--write"]).status.success());
    assert!(place.setup(&["--remove"]).status.success());
    assert!(!f.exists());
    place.done();
}

#[test]
fn intake_records_events_and_drops_secrets() {
    let place = Place::new("intake");
    let repo = place.repo.display().to_string().replace('\\', "\\\\");
    let lib = place.repo.join("crates").join("shop").join("src").join("lib.rs").display().to_string().replace('\\', "\\\\");
    let inputs = [
        ("SessionStart", r#"{"hook_event_name":"SessionStart","session_id":"s1","source":"startup","cwd":"CWD"}"#.to_owned()),
        ("UserPromptSubmit", r#"{"hook_event_name":"UserPromptSubmit","session_id":"s1","prompt_id":"p1","prompt":"refund fails; my key is sk-PLANTEDabcdefghijklmnop123456","cwd":"CWD"}"#.to_owned()),
        ("PreToolUse", format!(r#"{{"hook_event_name":"PreToolUse","session_id":"s1","prompt_id":"p1","tool_name":"Edit","tool_input":{{"file_path":"{lib}","old_string":"pub fn","new_string":"PLANTED"}},"cwd":"CWD"}}"#)),
        ("PostToolUseFailure", r#"{"hook_event_name":"PostToolUseFailure","session_id":"s1","prompt_id":"p1","tool_name":"Bash","tool_input":{"command":"TOKEN=PLANTEDsecret99 cargo test -p shop"},"error":"Exit code 101\nerror[E0425]: cannot find value\n --> crates/shop/src/lib.rs:3:5","cwd":"CWD"}"#.to_owned()),
        ("Stop", r#"{"hook_event_name":"Stop","session_id":"s1","stop_hook_active":false,"last_assistant_message":"PLANTED reply","cwd":"CWD"}"#.to_owned()),
        ("PreCompact", r#"{"hook_event_name":"PreCompact","session_id":"s1","trigger":"manual","custom_instructions":"PLANTED","cwd":"CWD"}"#.to_owned()),
    ];
    for (ev, input) in &inputs {
        let o = place.hook(ev, &input.replace("CWD", &repo));
        assert!(o.status.success(), "{ev}: exit {:?}", o.status);
        assert!(o.stdout.is_empty() && o.stderr.is_empty(), "{ev}: printed {}", text(&o));
    }
    let mem = place.memory();
    let rec = std::fs::read_to_string(mem.join("situation.jsonl")).unwrap();
    assert_eq!(rec.lines().count(), inputs.len(), "{rec}");
    assert!(!rec.contains("PLANTED"), "planted text in the record: {rec}");
    for want in ["\"kind\":\"session\"", "\"kind\":\"prompt\"", "\"kind\":\"edit\"", "\"kind\":\"test\"", "\"exit\":101", "\"code\":\"E0425\"", "\"kind\":\"stop\"", "\"kind\":\"compact\"", "crates/shop/src/lib.rs"] {
        assert!(rec.contains(want), "no {want} in {rec}");
    }
    let kept = std::fs::read_to_string(mem.join("prompts.jsonl")).unwrap();
    assert!(kept.contains("refund fails") && !kept.contains("PLANTED") && kept.contains("[secret dropped: API key]"), "{kept}");
    let shown = place.cmd(&["situation", "--project", &place.repo.display().to_string()]).output().unwrap();
    assert!(text(&shown).contains("refund fails") && text(&shown).contains("exit 101"), "{}", text(&shown));
    // Switching prompts off keeps only the length and hash; clearing removes the kept text.
    std::fs::write(place.home.join("settings.json"), b"{\"prompts\":\"off\"}").unwrap();
    place.hook("UserPromptSubmit", &inputs[1].1.replace("CWD", &repo));
    assert_eq!(std::fs::read_to_string(mem.join("prompts.jsonl")).unwrap(), kept);
    let c = place.cmd(&["situation", "--project", &place.repo.display().to_string(), "--clear-prompts"]).output().unwrap();
    assert!(c.status.success() && text(&c).contains("cleared the kept text of 1 prompts"), "{}", text(&c));
    assert!(!mem.join("prompts.jsonl").exists());
    // Bad input: exit 0, nothing printed, the fault in the Nano's own audit.
    let bad = place.hook("PostToolUse", "not json");
    assert!(bad.status.success() && bad.stdout.is_empty() && bad.stderr.is_empty());
    let audit = std::fs::read_to_string(place.home.join("memory").join("audit.jsonl")).unwrap();
    assert!(audit.contains("\"kind\":\"hook-fault\""), "{audit}");
    // Memory off: nothing recorded.
    std::fs::write(place.home.join("settings.json"), b"{\"memory\":\"off\"}").unwrap();
    let before = std::fs::read_to_string(mem.join("situation.jsonl")).unwrap();
    place.hook("Stop", &inputs[4].1.replace("CWD", &repo));
    assert_eq!(std::fs::read_to_string(mem.join("situation.jsonl")).unwrap(), before);
    place.done();
}

#[test]
fn parallel_hooks_keep_every_line_whole() {
    let place = Place::new("parallel");
    let repo = place.repo.display().to_string().replace('\\', "\\\\");
    let input = format!(r#"{{"hook_event_name":"PostToolUse","session_id":"s1","tool_name":"Bash","tool_input":{{"command":"cargo check"}},"tool_response":{{"stdout":"{}","stderr":""}},"cwd":"{repo}"}}"#, "x".repeat(3000));
    let children: Vec<_> = (0..40)
        .map(|_| {
            let mut c = place.cmd(&["hook", "PostToolUse"]);
            c.env("CLAUDE_PROJECT_DIR", &place.repo).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null());
            let mut ch = c.spawn().unwrap();
            std::io::Write::write_all(ch.stdin.as_mut().unwrap(), input.as_bytes()).unwrap();
            drop(ch.stdin.take());
            ch
        })
        .collect();
    for mut ch in children {
        assert!(ch.wait().unwrap().success());
    }
    let rec = std::fs::read_to_string(place.memory().join("situation.jsonl")).unwrap();
    assert_eq!(rec.lines().count(), 40);
    assert!(rec.lines().all(|l| l.starts_with('{') && l.ends_with('}') && l.contains("\"cmd_class\":\"cargo-check\"")));
    place.done();
}
