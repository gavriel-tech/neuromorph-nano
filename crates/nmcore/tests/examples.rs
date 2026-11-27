//! Product tests of the training store, run through the real program on copies of the small fixture
//! repository under `scratch/examples-test/`, with a test Nano folder there (`NANO_HOME`). Each
//! scripted session is fed to `neuromorph-nano hook <Event>` one event at a time; the audit row of
//! each context pack is written in the shape a serving process writes it, and the judged set the
//! serving path hands over is recorded with `examples --capture` (the same capture code the serving
//! process runs after a reply). Claude Code itself is not run, no real settings or memory are
//! touched, and nothing outside `scratch/` is written.
#![allow(clippy::unwrap_used)]
use nmjson::Value;
use nmtype::examples as ex;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

const EXE: &str = env!("CARGO_BIN_EXE_neuromorph-nano");
const ASK: &str = "where is the checkout total computed";
const LIB: &str = "crates/shop/src/lib.rs";
const INV: &str = "crates/inventory/src/lib.rs";
const OLD: &str = "    Some(unit_price_cents(&item.code) * u64::from(n))";
const NEW: &str = "    Some(unit_price_cents(&item.code) * u64::from(n) + 0)";
const INV_OLD: &str = "    if item.count < n {";
const INV_NEW: &str = "    if item.count < n || n == 0 {";

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
fn esc(p: &Path) -> String {
    p.display().to_string().replace('\\', "\\\\")
}
fn text(o: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}
fn get(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).unwrap_or_default().to_owned()
}

struct Place {
    dir: PathBuf,
    home: PathBuf,
    repo: PathBuf,
    start: u64,
    env: Vec<(String, String)>,
}

impl Place {
    fn new(name: &str) -> Self {
        let dir = workspace().join("scratch").join("examples-test").join(format!("{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let home = dir.join("nano");
        let repo = dir.join("small-repo");
        std::fs::create_dir_all(home.join("memory")).unwrap();
        copy_dir(&workspace().join("tests").join("fixtures").join("small-repo"), &repo);
        Self { dir, home, repo, start: now_ms() - 10_000, env: Vec::new() }
    }
    fn set(&self, key: &str, value: &str) {
        let p = self.home.join("settings.json");
        let mut m = std::fs::read(&p).ok().and_then(|b| nmjson::parse(&b).ok()).and_then(|v| v.object().ok().cloned()).unwrap_or_default();
        m.insert(key.into(), Value::String(value.into()));
        std::fs::write(p, Value::Object(m).encode().unwrap()).unwrap();
    }
    fn cmd(&self, args: &[&str]) -> Command {
        let mut c = Command::new(EXE);
        c.args(args).env("NANO_HOME", &self.home).env_remove("CLAUDE_PROJECT_DIR").current_dir(&self.dir);
        for (k, v) in &self.env {
            c.env(k, v);
        }
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
    fn hook(&self, event: &str, input: &str) -> u128 {
        let mut c = self.cmd(&["hook", event]);
        c.env("CLAUDE_PROJECT_DIR", &self.repo).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        let t = std::time::Instant::now();
        let mut child = c.spawn().unwrap();
        std::io::Write::write_all(child.stdin.as_mut().unwrap(), input.as_bytes()).unwrap();
        let o = child.wait_with_output().unwrap();
        let us = t.elapsed().as_micros();
        assert!(o.status.success() && o.stderr.is_empty(), "{event}: {}", text(&o));
        pause(2);
        us
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
    fn start_session(&self, s: &str) {
        self.hook("SessionStart", &format!(r#"{{"hook_event_name":"SessionStart","session_id":"{s}","source":"startup","cwd":"{}"}}"#, self.cwd()));
    }
    fn prompt(&self, s: &str, t: &str) {
        self.hook("UserPromptSubmit", &format!(r#"{{"hook_event_name":"UserPromptSubmit","session_id":"{s}","prompt":"{t}","cwd":"{}"}}"#, self.cwd()));
    }
    /// The audit row of a context pack, as a serving process writes it at the end of the call, with
    /// the reranker's scores in its context map as the whole path records them.
    fn served(&self, id: &str, episode: &str, sources: &[&str]) {
        let label = self.label();
        let time = now_ms();
        let src: Vec<String> = sources.iter().map(|s| format!("\"{s}\"")).collect();
        let scores: Vec<String> = sources.iter().map(|s| format!("\"rr {s} 0.500000\"")).collect();
        self.audit(&format!(
            r#"{{"id":"{id}","kind":"tool","tool":"nm_context_pack","project":"{label}","mode":"whole","via":"serve","session":"9999-{}","state":"done","ms":1,"time_ms":{time},"ok":true,"episode":"{episode}","argument":"{ASK}","argument_field":"question","withheld":false,"sources":[{}],"context_map":{{"mode":"full","scores":[{}],"stages":{{"reranker":0.900,"joint read":1.700}},"judge_tokens":553,"rerank_docs":3}},"steps":[{{"part":"nmapi.mcp","ms":1,"us":1200}},{{"part":"nmbrain.slot#reranker","ms":900,"us":900000}}]}}"#,
            self.start,
            src.join(","),
            scores.join(",")
        ));
        pause(3);
    }
    /// The judged set the serving path handed over for pack `id`, recorded through the capture code.
    fn capture(&self, id: &str, cands: &[(&str, f64, &str)], weights: &str, views: bool) {
        let c: Vec<String> = cands.iter().map(|(p, sc, t)| format!(r#"{{"place":"{p}","score":{sc},"text":"{t}"}}"#)).collect();
        let v: Vec<String> = cands.iter().enumerate().map(|(i, _)| format!("[{}, -0.25, 1.5]", i as f64 * 0.5)).collect();
        let body = format!(
            r#"{{"pack":"{id}","question":"{ASK}","scored":true,"candidates":[{}],"engine":{{"model":"qwen3-reranker-0.6b","weights":"{weights}","precision":"q8","half_math":"true"}},"views":[{}]}}"#,
            c.join(","),
            if views { v.join(",") } else { String::new() }
        );
        let f = self.dir.join(format!("capture-{id}.json"));
        std::fs::write(&f, body).unwrap();
        let o = self.run(&["examples", "--capture", f.to_str().unwrap()]);
        assert!(o.status.success(), "{}", text(&o));
    }
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
    fn read(&self, s: &str, rel: &str, span: (u64, u64)) {
        let file = self.file(rel);
        self.hook(
            "PostToolUse",
            &format!(
                r#"{{"hook_event_name":"PostToolUse","session_id":"{s}","tool_name":"Read","tool_input":{{"file_path":"{file}"}},"tool_response":{{"type":"text","file":{{"filePath":"{file}","startLine":{},"numLines":{}}}}},"cwd":"{}"}}"#,
                span.0,
                span.1,
                self.cwd()
            ),
        );
    }
    fn edit(&self, s: &str, use_id: &str, rel: &str, old: &str, new: &str) {
        for event in ["PreToolUse", "PostToolUse"] {
            if event == "PostToolUse" {
                let p = rel.split('/').fold(self.repo.clone(), |p, c| p.join(c));
                let t = std::fs::read_to_string(&p).unwrap();
                assert!(t.contains(old));
                std::fs::write(&p, t.replace(old, new)).unwrap();
            }
            self.hook(
                event,
                &format!(
                    r#"{{"hook_event_name":"{event}","session_id":"{s}","tool_name":"Edit","tool_use_id":"{use_id}","tool_input":{{"file_path":"{}","old_string":"{old}","new_string":"{new}"}},"cwd":"{}"}}"#,
                    self.file(rel),
                    self.cwd()
                ),
            );
        }
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
    fn stop(&self, s: &str) -> u128 {
        self.hook("Stop", &format!(r#"{{"hook_event_name":"Stop","session_id":"{s}","stop_hook_active":false,"cwd":"{}"}}"#, self.cwd()))
    }
    fn tdir(&self) -> PathBuf {
        ex::dir(&self.memory())
    }
    fn examples(&self) -> Vec<Value> {
        ex::read_rows(&self.memory(), ex::EXAMPLES).unwrap()
    }
    /// Every byte the store holds, as text.
    fn store_text(&self) -> String {
        let mut out = String::new();
        let mut stack = vec![self.tdir()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
                if e.path().is_dir() {
                    stack.push(e.path());
                } else {
                    out.push_str(&String::from_utf8_lossy(&std::fs::read(e.path()).unwrap()));
                }
            }
        }
        out
    }
    fn done(self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Each candidate of an example as `label/strength place [served]`.
fn labels(e: &Value) -> Vec<String> {
    e.get("candidates")
        .unwrap()
        .array()
        .unwrap()
        .iter()
        .map(|c| {
            format!(
                "{}/{} {}{}",
                get(c, "label"),
                c.get("strength").ok().and_then(|x| x.string().ok()).unwrap_or("-"),
                get(c, "place"),
                if matches!(c.get("served"), Ok(Value::Bool(true))) { " served" } else { "" }
            )
        })
        .collect()
}

/// The pack, then the edit of its first cited place, a passing test, an accepting next prompt.
fn positive(p: &Place, s: &str, weights: &str) {
    p.start_session(s);
    p.prompt(s, "checkout charges the wrong total");
    p.served("a1-pos", "e-pos", &["crates/shop/src/lib.rs:13-19", "crates/inventory/src/lib.rs:1-21"]);
    p.capture("a1-pos", &[(&format!("{LIB}:13-19"), 0.989_83, ""), (&format!("{INV}:1-21"), 0.4, ""), ("crates/shop/tests/checkout_refund.rs:1-10", 0.140_336, "")], weights, true);
    p.pack(s, "u-pack", "e-pos");
    p.read(s, LIB, (13, 7));
    p.edit(s, "u-e1", LIB, OLD, NEW);
    p.bash(s, "cargo test -p shop", "test result: ok. 3 passed; 0 failed; 0 ignored");
    p.stop(s);
    p.prompt(s, "thanks, that works");
    p.stop(s);
}

#[test]
fn pack_edit_pass_gives_a_positive_example() {
    let p = Place::new("positive");
    positive(&p, "s-pos", "w1");
    let ex_rows = p.examples();
    assert_eq!(ex_rows.len(), 1, "{ex_rows:?}");
    let e = &ex_rows[0];
    assert_eq!((get(e, "family"), get(e, "verdict"), get(e, "strength")), ("learning".into(), "verified".into(), "indirect".into()));
    assert_eq!(
        labels(e),
        [
            "verified/indirect crates/shop/src/lib.rs:13-19 served",
            "unused/- crates/inventory/src/lib.rs:1-21 served",
            "judged/- crates/shop/tests/checkout_refund.rs:1-10"
        ]
    );
    // The score is exact (its bits), the hash is the lines' hash before the edit.
    let first = &e.get("candidates").unwrap().array().unwrap()[0];
    assert_eq!(get(first, "bits"), format!("{:08x}", 0.989_83_f32.to_bits()));
    let before = std::fs::read_to_string(workspace().join("tests/fixtures/small-repo").join(LIB)).unwrap();
    let lines: Vec<&str> = before.lines().skip(12).take(7).collect();
    assert_eq!(get(first, "hash"), ex::content_hash(&lines));
    // Provenance: path taken, settings, hardware, engine; domain tags; a split fixed by hash.
    assert_eq!(get(e, "path"), "full");
    assert!(e.get("settings").unwrap().get("precision.reranker").is_ok() && e.get("hardware").unwrap().get("id").is_ok());
    assert!(get(e, "engine").starts_with('e'), "{e:?}");
    let domains: Vec<String> = e.get("domains").unwrap().array().unwrap().iter().map(|d| d.string().unwrap().to_owned()).collect();
    assert!(domains.contains(&"code:rust".to_owned()) && domains.iter().any(|d| d.starts_with("project:")), "{domains:?}");
    assert_eq!(get(e, "split"), ex::split_of(&get(e, "ask_sha"), LIB).name());
    // The reader a fitting job calls: positives, views for its engine, preferences, hardware.
    let got = ex::read_examples(&p.memory(), &ex::Filter { family: Some("learning".into()), min_strength: 2, ..ex::Filter::default() }).unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].positives().iter().map(|c| c.path.clone()).collect::<Vec<_>>(), [LIB]);
    assert_eq!(got[0].candidates[0].score, Some(0.989_83));
    let views = ex::read_views(&p.memory(), &get(e, "engine")).unwrap();
    assert_eq!(views.get(&("a1-pos".to_owned(), 1)), Some(&vec![0.5, -0.25, 1.5]));
    let prefs: Vec<String> = ex::read_preferences(&p.memory()).unwrap().iter().map(|r| format!("{} {}", get(r, "kind"), get(r, "strength"))).collect();
    assert!(prefs.contains(&"edit-kept indirect".to_owned()) && prefs.contains(&"acceptance weak".to_owned()), "{prefs:?}");
    let hw = ex::read_hardware(&p.memory()).unwrap().unwrap();
    assert_eq!(get(&hw, "family"), "fundamentals");
    assert!(hw.get("rows").unwrap().array().unwrap().iter().any(|r| get(r, "part") == "stage:reranker" && get(r, "piece").starts_with(if hw.get("machine").unwrap().get("gpu").and_then(Value::string).is_ok() { "gpu:" } else { "cpu:" })));
    // The command's counts, one example shown, the API's shape.
    let o = p.run(&["examples"]);
    let out = text(&o);
    assert!(o.status.success() && out.contains("verified 1") && out.contains("learning 1") && out.contains("split"), "{out}");
    let shown = text(&p.run(&["examples", "--show", "x-a1-pos"]));
    assert!(shown.contains("x-a1-pos verified indirect") && shown.contains("crates/shop/src/lib.rs:13-19 [served]"), "{shown}");
    let j = nmjson::parse(&p.run(&["examples", "--json"]).stdout).unwrap();
    assert_eq!(get(&j, "setting"), "features");
    let size = std::fs::read_to_string(p.tdir().join(ex::EXAMPLES)).unwrap().len();
    println!("one example: {size} bytes; row: {}", std::fs::read_to_string(p.tdir().join(ex::EXAMPLES)).unwrap().trim());
    p.done();
}

#[test]
fn went_elsewhere_gives_the_true_place_and_hard_negatives() {
    let p = Place::new("elsewhere");
    let s = "s-else";
    p.start_session(s);
    p.prompt(s, "zero units should not be taken from the shelf");
    p.served("a1-else", "e-else", &["crates/shop/src/lib.rs:13-19", "README.md:1-4"]);
    p.capture("a1-else", &[(&format!("{LIB}:13-19"), 0.95, ""), ("README.md:1-4", 0.7, ""), (&format!("{INV}:9-16"), 0.2, "")], "w1", false);
    p.pack(s, "u-pack", "e-else");
    p.read(s, INV, (9, 8));
    p.edit(s, "u-e1", INV, INV_OLD, INV_NEW);
    p.bash(s, "cargo test -p inventory", "test result: ok. 2 passed; 0 failed; 0 ignored");
    p.stop(s);
    p.prompt(s, "next one");
    p.stop(s);
    let rows = p.examples();
    assert_eq!(rows.len(), 1, "{rows:?}");
    let e = &rows[0];
    assert_eq!((get(e, "verdict"), get(e, "strength"), get(e, "anchor")), ("corrected".into(), "indirect".into(), INV.into()));
    assert_eq!(
        labels(e),
        ["passed-over/indirect crates/shop/src/lib.rs:13-19 served", "passed-over/indirect README.md:1-4 served", "went/indirect crates/inventory/src/lib.rs:9-16"]
    );
    let got = ex::read_examples(&p.memory(), &ex::Filter::default()).unwrap();
    assert_eq!(got[0].positives().iter().map(|c| c.path.clone()).collect::<Vec<_>>(), [INV]);
    assert_eq!(got[0].hard_negatives().len(), 2);
    p.done();
}

#[test]
fn refuted_is_kept_as_refuted() {
    let p = Place::new("refuted");
    let s = "s-ref";
    p.start_session(s);
    p.prompt(s, "checkout charges the wrong total");
    p.served("a1-ref", "e-ref", &["crates/shop/src/lib.rs:13-19"]);
    p.capture("a1-ref", &[(&format!("{LIB}:13-19"), 0.9, "")], "w1", false);
    p.pack(s, "u-pack", "e-ref");
    p.read(s, LIB, (13, 7));
    p.edit(s, "u-e1", LIB, OLD, NEW);
    p.bash_fail(s, "cargo test -p shop", "Exit code 101\\ntest result: FAILED. 2 passed; 1 failed; 0 ignored");
    p.bash(s, "git checkout crates/shop/src/lib.rs", "");
    p.stop(s);
    p.prompt(s, "still failing, look again");
    p.stop(s);
    let rows = p.examples();
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!((get(&rows[0], "verdict"), get(&rows[0], "strength")), ("refuted".into(), "direct".into()));
    assert_eq!(labels(&rows[0]), ["refuted/direct crates/shop/src/lib.rs:13-19 served"]);
    let prefs: Vec<String> = ex::read_preferences(&p.memory()).unwrap().iter().map(|r| format!("{} {}", get(r, "kind"), get(r, "strength"))).collect();
    assert!(prefs.contains(&"edit-reverted direct".to_owned()) && prefs.contains(&"correction weak".to_owned()), "{prefs:?}");
    // Preferences live in their own file with their own family, never among learning examples.
    assert!(ex::read_preferences(&p.memory()).unwrap().iter().all(|r| get(r, "family") == "preference"));
    assert!(p.examples().iter().all(|r| get(r, "family") == "learning"));
    p.done();
}

#[test]
fn features_keeps_no_text_and_full_keeps_passages_without_secrets() {
    let p = Place::new("text");
    positive(&p, "s-feat", "w1");
    let all = p.store_text();
    for planted in [ASK, "unit_price_cents", "take_from_shelf", "thanks, that works", "checkout charges"] {
        assert!(!all.contains(planted), "{planted} is in the store at features");
    }
    // Full: a second pack whose judged passages carry a planted secret.
    p.set("examples", "full");
    let s = "s-full";
    p.start_session(s);
    p.prompt(s, "the refund path");
    p.served("a2-full", "e-full", &["crates/shop/src/lib.rs:1-3"]);
    let secret = "sk-live-4f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c";
    p.capture("a2-full", &[("crates/shop/src/lib.rs:1-3", 0.8, &format!("let api_key = \\\"{secret}\\\"; // checkout total")), ("README.md:1-4", 0.3, "")], "w1", false);
    p.pack(s, "u-pack", "e-full");
    p.read(s, LIB, (1, 3));
    p.edit(s, "u-e1", INV, INV_OLD, INV_NEW);
    p.bash(s, "cargo test -p shop", "test result: ok. 3 passed; 0 failed; 0 ignored");
    p.stop(s);
    p.prompt(s, "great");
    p.stop(s);
    let passages = std::fs::read_to_string(p.tdir().join(ex::PASSAGES)).unwrap();
    assert!(!passages.contains(secret) && passages.contains("[secret dropped") && passages.contains("checkout total"), "{passages}");
    assert!(passages.contains("small-repo"), "the README passage read from the file: {passages}");
    let rows = p.examples();
    assert_eq!(rows.len(), 2, "{rows:?}");
    assert_eq!(get(&rows[1], "question"), ASK);
    assert!(rows[0].get("question").is_err(), "the features example gained text");
    let got = ex::read_examples(&p.memory(), &ex::Filter::default()).unwrap();
    assert!(got[1].candidates.iter().any(|c| c.text.as_deref().is_some_and(|t| t.contains("checkout total"))));
    p.done();
}

#[test]
fn split_never_changes_and_engines_are_kept_apart() {
    let a = Place::new("split-a");
    positive(&a, "s-split", "w1");
    let b = Place::new("split-b");
    positive(&b, "s-split", "w2");
    let (ea, eb) = (&a.examples()[0], &b.examples()[0]);
    assert_eq!((get(ea, "split"), get(ea, "ask_sha"), get(ea, "anchor")), (get(eb, "split"), get(eb, "ask_sha"), get(eb, "anchor")));
    assert_eq!(get(ea, "split"), ex::split_of(&get(ea, "ask_sha"), &get(ea, "anchor")).name());
    // Settling again rewrites nothing.
    let before = std::fs::read_to_string(a.tdir().join(ex::EXAMPLES)).unwrap();
    a.run(&["examples"]);
    assert_eq!(std::fs::read_to_string(a.tdir().join(ex::EXAMPLES)).unwrap(), before);
    // A second pack in place a under other weights: another engine id, its own views file, and a
    // fitting job asking for one engine gets only its examples.
    let s = "s-w2";
    a.start_session(s);
    a.prompt(s, "inventory total");
    a.served("a2-w2", "e-w2", &["crates/inventory/src/lib.rs:18-21"]);
    a.capture("a2-w2", &[(&format!("{INV}:18-21"), 0.7, "")], "w2", true);
    a.pack(s, "u-pack", "e-w2");
    a.read(s, INV, (18, 4));
    a.edit(s, "u-e1", INV, INV_OLD, INV_NEW);
    a.bash(s, "cargo test -p inventory", "test result: ok. 2 passed; 0 failed; 0 ignored");
    a.stop(s);
    a.prompt(s, "thanks");
    a.stop(s);
    let rows = a.examples();
    assert_eq!(rows.len(), 2, "{rows:?}");
    let (e1, e2) = (get(&rows[0], "engine"), get(&rows[1], "engine"));
    assert!(!e1.is_empty() && !e2.is_empty() && e1 != e2);
    assert_eq!(e2, get(eb, "engine"), "the same weights give the same engine id in another run");
    let only = ex::read_examples(&a.memory(), &ex::Filter { engine: Some(e1.clone()), ..ex::Filter::default() }).unwrap();
    assert_eq!(only.iter().map(|e| e.pack.clone()).collect::<Vec<_>>(), ["a1-pos"]);
    let engines = ex::read_engines(&a.memory()).unwrap();
    assert_eq!(get(engines.get(&e1).unwrap(), "weights"), "w1");
    assert!(a.tdir().join("views").join(format!("{e1}.bin")).exists() && a.tdir().join("views").join(format!("{e2}.bin")).exists());
    assert!(ex::read_views(&a.memory(), &e1).unwrap().keys().all(|k| k.0 == "a1-pos"));
    a.done();
    b.done();
}

/// Retention counts run-days and removes nothing for size: a fresh passed-over example stays beside
/// a verified one (the rule itself is checked in `nmmem.keep` and `nmcore.examples`).
#[test]
fn retention_keeps_fresh_rows() {
    let p = Place::new("retention");
    positive(&p, "s-keep", "w1");
    // A newer pack whose cited lines are never touched: passed over, weak.
    let s = "s-weak";
    p.start_session(s);
    p.prompt(s, "tidy the readme");
    p.served("a2-weak", "e-weak", &["crates/shop/src/lib.rs:13-19"]);
    p.capture("a2-weak", &[(&format!("{LIB}:13-19"), 0.6, "")], "w1", false);
    p.pack(s, "u-pack", "e-weak");
    p.read(s, "README.md", (1, 4));
    p.stop(s);
    p.prompt(s, "ok");
    p.stop(s);
    let rows = p.examples();
    assert_eq!(rows.iter().map(|r| get(r, "verdict")).collect::<Vec<_>>(), ["verified", "passed-over"]);
    p.done();
}

#[test]
fn off_writes_nothing_and_clear_removes_everything() {
    let p = Place::new("off");
    p.set("examples", "off");
    positive(&p, "s-off", "w1");
    assert!(!p.tdir().exists(), "the store was written with examples off");
    p.set("examples", "features");
    // Memory off writes nothing either.
    p.set("memory", "off");
    p.run(&["examples"]);
    assert!(!p.tdir().exists(), "the store was written with memory off");
    p.set("memory", "on");
    // Switched on later: the closed pack's example is written at the next settle.
    p.run(&["examples"]);
    assert_eq!(p.examples().len(), 1);
    let o = p.run(&["examples", "--clear"]);
    assert!(o.status.success() && text(&o).contains("rows removed"), "{}", text(&o));
    assert!(!p.tdir().exists());
    let audit = std::fs::read_to_string(p.home.join("memory").join("audit.jsonl")).unwrap();
    assert!(audit.contains("Training store cleared"), "{audit}");
    // Nothing served before the clear comes back.
    p.run(&["examples"]);
    assert!(p.examples().is_empty());
    // Export and import between two stores: the same rows arrive, a tampered file is refused.
    let q = Place::new("move");
    positive(&q, "s-move", "w1");
    let file = q.dir.join("export.jsonl");
    assert!(q.run(&["examples", "--export", file.to_str().unwrap()]).status.success());
    let r = Place::new("move-to");
    r.start_session("s-x");
    let o = r.run(&["examples", "--import", file.to_str().unwrap()]);
    assert!(o.status.success() && text(&o).contains("imported"), "{}", text(&o));
    assert_eq!(r.examples().len(), 1);
    let t = std::fs::read_to_string(&file).unwrap();
    std::fs::write(&file, t.replacen("verified", "refuted", 1)).unwrap();
    assert!(!r.run(&["examples", "--import", file.to_str().unwrap()]).status.success());
    p.done();
    q.done();
    r.done();
}

#[test]
fn capture_and_settle_cost() {
    // On the serving thread: the tap in the whole path, taking it, starting the capture thread.
    let paths: Vec<String> = (0..40).map(|i| format!("crates/x/src/m{i}.rs")).collect();
    let text40 = "fn a() {}\n".repeat(30);
    let scores = vec![0.5_f32; 40];
    ex::set_level(ex::FEATURES);
    let t = std::time::Instant::now();
    for _ in 0..200 {
        ex::tap(ASK, paths.iter().map(|p| (p.as_str(), 1, 30, text40.as_str())), &scores, true);
        let tap = ex::take_tap();
        std::thread::spawn(move || drop(tap)).join().unwrap();
    }
    let per_call = t.elapsed().as_micros() / 200;
    // A Stop hook that settles, with the store at features and off, on the same session.
    let p = Place::new("cost");
    positive(&p, "s-cost", "w1");
    let mut on = Vec::new();
    let mut off = Vec::new();
    for i in 0..15 {
        p.set("examples", "features");
        on.push(p.stop("s-cost"));
        p.set("examples", "off");
        off.push(p.stop("s-cost"));
        let _ = i;
    }
    on.sort_unstable();
    off.sort_unstable();
    // The capture thread's own work (finding the audit row, hashing lines, writing), measured as a
    // capture command against a command that starts the program and writes nothing.
    let (mut cap, mut base) = (Vec::new(), Vec::new());
    let cands: Vec<(String, f64, &str)> = (1..=19).map(|i| (format!("{INV}:{i}-{}", i + 2), 0.5, "")).collect();
    let cands: Vec<(&str, f64, &str)> = cands.iter().map(|(a, b, c)| (a.as_str(), *b, *c)).collect();
    for _ in 0..10 {
        let t = std::time::Instant::now();
        p.set("examples", "features");
        p.capture("a1-pos", &cands, "w1", false);
        cap.push(t.elapsed().as_micros());
        let t = std::time::Instant::now();
        p.set("examples", "features");
        p.run(&["examples", "--references"]);
        base.push(t.elapsed().as_micros());
    }
    cap.sort_unstable();
    base.sort_unstable();
    println!("capture of 19 candidates: command median {} ms, program start alone {} ms", cap[5] / 1000, base[5] / 1000);
    println!("serving thread, 40 candidates: {per_call} us per call (tap, take, thread start)");
    println!("Stop hook median: features {} ms, off {} ms", on[7] / 1000, off[7] / 1000);
    assert!(per_call < 5_000, "{per_call} us per call");
    p.done();
}
