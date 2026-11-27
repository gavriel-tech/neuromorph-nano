//! Product tests of the cartridge library through the real program, with test Nano folders under
//! `scratch/library-test/` (`NANO_HOME`): the stacks in use on this machine, a graphics card the
//! library has not seen (its identity handed in), switching the stack for a use and back, export
//! and import as one file, tampered files refused by name, trial cartridges never changing a served
//! reply, and evidence from another piece never taking a stack off trial. Nothing outside
//! `scratch/` is written; the models folder is only read.
#![allow(clippy::unwrap_used)]
use nmjson::Value;
use nmlearn::hardware::{Machine, cpu_piece, gpu_piece};
use nmlearn::library::{CartridgeSpec, EvidenceRow, Filter, Library, Stack, Strength};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const EXE: &str = env!("CARGO_BIN_EXE_neuromorph-nano");
const GTX_STACK: &str = "gpu-nvidia-geforce-gtx-1650-4096mib";
const GTX: &str = "gpu:nvidia:geforce-gtx-1650:4096mib:driver-581";

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").canonicalize().unwrap()
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
fn out(o: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}
fn st(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default()
}
fn is_true(v: &Value, k: &str) -> bool {
    matches!(v.get(k), Ok(Value::Bool(true)))
}

struct Home {
    dir: PathBuf,
}
// 2026-11-26 (Luis Jake Gabriel III: "We should always delete test and junk stuff"): a test's home
// goes when the test ends.
impl Drop for Home {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}
impl Home {
    fn new(name: &str) -> Self {
        let dir = workspace().join("scratch").join("library-test").join(format!("{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("memory")).unwrap();
        for f in ["nano.json", "rules.md"] {
            std::fs::copy(workspace().join(f), dir.join(f)).unwrap();
        }
        copy_dir(&workspace().join("designs"), &dir.join("designs"));
        Self { dir }
    }
    /// The part cards the server reads (`crates/*/part.json`), for commands that open the organism.
    fn with_cards(self) -> Self {
        for e in std::fs::read_dir(workspace().join("crates")).unwrap().flatten() {
            let card = e.path().join("part.json");
            if card.is_file() {
                let to = self.dir.join("crates").join(e.file_name());
                std::fs::create_dir_all(&to).unwrap();
                std::fs::copy(&card, to.join("part.json")).unwrap();
            }
        }
        self
    }
    fn run(&self, args: &[&str]) -> Output {
        Command::new(EXE)
            .args(args)
            .env("NANO_HOME", &self.dir)
            .env("NANO_MODELS", nmtype::models_dir(workspace()))
            .env_remove("CLAUDE_PROJECT_DIR")
            .current_dir(&self.dir)
            .output()
            .unwrap()
    }
    fn ok(&self, args: &[&str]) -> String {
        let o = self.run(args);
        assert!(o.status.success(), "{args:?} failed: {}", out(&o));
        out(&o)
    }
    fn json(&self, args: &[&str]) -> Value {
        let o = self.run(args);
        assert!(o.status.success(), "{args:?} failed: {}", out(&o));
        nmjson::parse(&o.stdout).unwrap()
    }
    fn audit(&self) -> Vec<Value> {
        std::fs::read_to_string(self.dir.join("memory").join("audit.jsonl")).unwrap_or_default().lines().filter_map(|l| nmjson::parse(l.as_bytes()).ok()).collect()
    }
    fn machine(&self, name: &str, gpu: nmlearn::hardware::Piece) -> String {
        let m = Machine { pieces: vec![cpu_piece("12th Gen Intel(R) Core(TM) i5-12450H"), gpu], probed_ms: 1, how: "given".into() };
        let p = self.dir.join(name);
        std::fs::write(&p, m.to_value().encode().unwrap()).unwrap();
        p.to_str().unwrap().to_owned()
    }
    fn stacks(&self, machine: Option<&str>) -> Value {
        match machine {
            Some(m) => self.json(&["library", "stack", "--json", "--machine", m]),
            None => self.json(&["library", "stack", "--json"]),
        }
    }
}

fn use_of<'a>(v: &'a Value, u: &str) -> &'a Value {
    v.get("uses").unwrap().array().unwrap().iter().find(|x| st(x, "use") == u).unwrap()
}
fn options(v: &Value) -> Vec<String> {
    v.get("options").unwrap().array().unwrap().iter().map(|x| x.string().unwrap().to_owned()).collect()
}

#[test]
fn stacks_in_use_on_this_machine() {
    let h = Home::new("here");
    let v = h.stacks(None);
    let m = Machine::from_value(v.get("machine").unwrap()).unwrap();
    assert!(h.audit().iter().any(|r| st(r, "kind") == "library" && st(r, "action") == "create"));
    let Some(gpu) = m.piece(nmlearn::hardware::Kind::Gpu) else { return };
    let g = use_of(&v, "hardware:gpu");
    if gpu.identity() == GTX {
        assert_eq!(st(g, "stack"), format!("{GTX_STACK}@3"));
        assert!(!is_true(g, "trial"));
        assert!(options(&v).windows(2).any(|w| w == ["--judge-window", "384"]));
    } else {
        assert!(is_true(g, "trial") || st(g, "stack").is_empty());
    }
}

#[test]
fn unknown_graphics_card_gets_the_nearest_relative_on_trial() {
    let h = Home::new("unknown");
    let f = h.machine("m1660.json", gpu_piece("NVIDIA GeForce GTX 1660 SUPER", 6144, "581.83", "7.5"));
    let v = h.stacks(Some(&f));
    let g = use_of(&v, "hardware:gpu");
    assert_eq!(st(g, "stack"), format!("{GTX_STACK}@3"));
    assert!(is_true(g, "trial"));
    assert!(st(g, "why").contains("nearest"), "{}", st(g, "why"));
    assert!(v.get("set").unwrap().object().unwrap().is_empty(), "a relative served settings");
    assert!(options(&v).is_empty());
    // The same card on another driver range stands in on trial too; another family gets nothing.
    let f2 = h.machine("mdrv.json", gpu_piece("NVIDIA GeForce GTX 1650", 4096, "572.10", "7.5"));
    assert!(is_true(use_of(&h.stacks(Some(&f2)), "hardware:gpu"), "trial"));
    let f3 = h.machine("mamd.json", gpu_piece("AMD Radeon RX 7600", 8192, "", ""));
    assert!(st(use_of(&h.stacks(Some(&f3)), "hardware:gpu"), "stack").is_empty());
}

#[test]
fn switch_a_use_off_and_back() {
    let h = Home::new("switch");
    let f = h.machine("m.json", gpu_piece("NVIDIA GeForce GTX 1650", 4096, "581.83", "7.5"));
    let before = h.stacks(Some(&f));
    assert!(!options(&before).is_empty());
    h.ok(&["library", "switch", "hardware:gpu", "off"]);
    let off = h.stacks(Some(&f));
    assert!(st(use_of(&off, "hardware:gpu"), "stack").is_empty());
    assert!(options(&off).is_empty(), "unplugged, the code's own values remain");
    h.ok(&["library", "switch", "hardware:gpu", GTX_STACK]);
    assert_eq!(options(&h.stacks(Some(&f))), options(&before));
    assert!(h.audit().iter().filter(|r| st(r, "action") == "use").count() == 2);
    // A stack for another use, or one that does not exist, is refused.
    assert!(!h.run(&["library", "switch", "job:coding", GTX_STACK]).status.success());
    assert!(!h.run(&["library", "switch", "hardware:gpu", "nonexistent"]).status.success());
}

#[test]
fn export_import_round_trip_and_tamper() {
    let a = Home::new("export-a");
    let b = Home::new("export-b");
    a.ok(&["library", "list"]);
    let file = a.dir.join("gtx.nmstack");
    a.ok(&["library", "export", GTX_STACK, "--out", file.to_str().unwrap()]);
    let first = std::fs::read(&file).unwrap();
    // B starts with an empty library, so the import is real.
    Library::open(&b.dir).unwrap().save().unwrap();
    let said = b.ok(&["library", "import", file.to_str().unwrap()]);
    assert!(said.contains("on trial"), "{said}");
    assert!(is_true(&b.json(&["library", "show", GTX_STACK]), "trial"));
    let again = b.dir.join("again.nmstack");
    b.ok(&["library", "export", GTX_STACK, "--out", again.to_str().unwrap()]);
    assert_eq!(std::fs::read(&again).unwrap(), first, "round trip not byte-equal");
    let (la, lb) = (Library::open(&a.dir).unwrap(), Library::open(&b.dir).unwrap());
    let (ea, eb) = (la.cartridge("gtx-1650-4gb").unwrap(), lb.cartridge("gtx-1650-4gb").unwrap());
    for f in &ea.files {
        assert_eq!(std::fs::read(la.cartridge_dir(ea).join(&f.name)).unwrap(), std::fs::read(lb.cartridge_dir(eb).join(&f.name)).unwrap());
    }
    // Tampered files are refused, each with its reason, and leave nothing behind.
    let text = String::from_utf8(first).unwrap();
    let c = Home::new("export-c");
    Library::open(&c.dir).unwrap().save().unwrap();
    let refuse = |body: &str, name: &str| {
        let p = c.dir.join(name);
        std::fs::write(&p, body).unwrap();
        let o = c.run(&["library", "import", p.to_str().unwrap()]);
        assert!(!o.status.success(), "{name} was accepted");
        out(&o)
    };
    let at = text.find("\"hex\":\"").unwrap() + 7;
    let mut t1 = text.clone();
    let ch = if &t1[at..=at] == "7" { "6" } else { "7" };
    t1.replace_range(at..=at, ch);
    let r1 = refuse(&t1, "t1.nmstack");
    assert!(r1.contains("(file-hash)"), "{r1}");
    let r2 = refuse(&text.replace("What was learned about this graphics card.", "Something else."), "t2.nmstack");
    assert!(r2.contains("(stack-hash)"), "{r2}");
    let r3 = refuse(&text.replacen("bit-identical", "bit-identicaI", 1), "t3.nmstack");
    assert!(r3.contains("(export-hash)"), "{r3}");
    assert!(refuse(&text[..text.len() / 2], "t4.nmstack").contains("(schema)"));
    assert!(Library::open(&c.dir).unwrap().stacks.is_empty(), "a refused file left a stack");
}

/// A stack for the general use, switched on, holding one cartridge on trial whose settings would
/// change what the whole path reads.
fn add_trial_cartridge(home: &Path) {
    let mut lib = Library::open(home).unwrap();
    let filter = Filter::Settings([("judge_window", "128"), ("look_gate", "1"), ("half_math", "off")].iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect());
    let spec = CartridgeSpec { id: "trial-settings".into(), version: 1, family: "fundamentals".into(), domain: "nano-system".into(), scope: "general".into(), model: "settings".into(), weights: String::new(), layers: None, name: "trial settings".into(), description: "Settings that would change the whole path.".into(), made_from: "values written for the test".into(), effect: String::new() };
    lib.add_cartridge(&spec, &filter, Value::Null, Vec::new(), true).unwrap();
    lib.add_stack(Stack { id: "general-trial".into(), version: 1, use_for: "general".into(), piece: None, cartridges: vec!["trial-settings@1".into()], description: String::new(), hash: String::new(), trial: false }).unwrap();
    lib.switch_use("general", Some("general-trial")).unwrap();
    lib.save().unwrap();
}

/// What a server started now would put in use (bench runs record nothing).
fn last_start(h: &Home) -> Value {
    h.stacks(None)
}

fn dump(h: &Home, repo: &Path, qfile: &Path, name: &str) -> Vec<(String, Vec<u8>)> {
    let d = h.dir.join(name);
    h.ok(&["bench", "--mode", "plain", "--root", repo.to_str().unwrap(), "--questions", qfile.to_str().unwrap(), "--dump", d.to_str().unwrap(), "--quiet"]);
    let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(&d).unwrap().flatten().map(|e| (e.file_name().to_string_lossy().into_owned(), std::fs::read(e.path()).unwrap())).collect();
    files.sort();
    files
}

#[test]
fn a_trial_cartridge_never_changes_a_served_reply() {
    let h = Home::new("trial").with_cards();
    let repo = h.dir.join("small-repo");
    copy_dir(&workspace().join("tests").join("fixtures").join("small-repo"), &repo);
    let q = h.dir.join("questions.json");
    std::fs::write(
        &q,
        r#"{"questions":[{"id":"q1","kind":"name","question":"where is the checkout total computed","answers":[{"file":"crates/shop/src/lib.rs","anchor":"pub fn checkout","lines":[13,19]}]},
            {"id":"q2","kind":"name","question":"how is the unit price of an item found","answers":[{"file":"crates/shop/src/lib.rs","anchor":"pub fn unit_price_cents","lines":[4,11]}]}]}"#,
    )
    .unwrap();
    h.ok(&["library", "list"]);
    let without = dump(&h, &repo, &q, "dump-a");
    let a = last_start(&h);
    add_trial_cartridge(&h.dir);
    let with = dump(&h, &repo, &q, "dump-b");
    let b = last_start(&h);
    assert!(!without.is_empty());
    assert_eq!(without, with, "a trial cartridge changed a served reply");
    assert_eq!(a.get("options").unwrap(), b.get("options").unwrap(), "a trial cartridge changed the served options");
    assert_eq!(a.get("set").unwrap(), b.get("set").unwrap());
    assert!(b.get("trial").unwrap().array().unwrap().iter().any(|x| st(x, "key") == "judge_window" && st(x, "value") == "128"), "trial not recorded");
    assert!(b.get("loaded").unwrap().array().unwrap().iter().any(|x| st(x, "key") == "trial-settings@1" && is_true(x, "trial")));
}

#[test]
fn evidence_from_another_piece_never_takes_a_stack_off_trial() {
    let h = Home::new("ready");
    h.ok(&["library", "list"]);
    let gtx = gpu_piece("NVIDIA GeForce GTX 1650", 4096, "581.83", "7.5");
    let other = gpu_piece("NVIDIA GeForce GTX 1660 SUPER", 6144, "581.83", "7.5");
    let rows = |subject: &str, hw: &str, k: usize| -> Vec<EvidenceRow> {
        (0..k).map(|_| EvidenceRow { subject: subject.into(), hardware: hw.into(), outcome: "verified".into(), strength: Some(Strength::Direct), measure: "m".into(), source: "test".into(), at: 1 }).collect()
    };
    let mut lib = Library::open(&h.dir).unwrap();
    let filter = Filter::Settings([("fast_k".to_owned(), "4".to_owned())].into());
    let spec = |id: &str| CartridgeSpec { id: id.into(), version: 1, family: "fundamentals".into(), domain: "hardware:gpu".into(), scope: String::new(), model: "settings".into(), weights: String::new(), layers: None, name: id.into(), description: "Settings for the test.".into(), made_from: "values written for the test".into(), effect: String::new() };
    lib.add_cartridge(&spec("x"), &filter, Value::Null, rows("x@1", &other.identity(), 10), true).unwrap();
    lib.add_cartridge(&spec("y"), &filter, Value::Null, rows("y@1", &gtx.identity(), 6), true).unwrap();
    for (id, b) in [("s-x", "x@1"), ("s-y", "y@1")] {
        lib.add_stack(Stack { id: id.into(), version: 1, use_for: "hardware:gpu".into(), piece: Some(gtx.clone()), cartridges: vec![b.into()], description: String::new(), hash: String::new(), trial: true }).unwrap();
    }
    lib.save().unwrap();
    let o = h.run(&["library", "switch", "s-x", "serve"]);
    assert!(!o.status.success(), "served on evidence from another piece");
    assert!(out(&o).contains("0 verified"), "{}", out(&o));
    assert!(is_true(&h.json(&["library", "show", "s-x"]), "trial"));
    h.ok(&["library", "switch", "s-y", "serve"]);
    assert!(!is_true(&h.json(&["library", "show", "s-y"]), "trial"));
}

/// A bench run and a `check` run write nothing into the Nano's own audit log, its project memory
/// or the library's record of the last server start: only a serving process records its start.
#[test]
fn bench_and_check_leave_the_audit_untouched() {
    let h = Home::new("quiet").with_cards();
    let repo = h.dir.join("small-repo");
    copy_dir(&workspace().join("tests").join("fixtures").join("small-repo"), &repo);
    let q = h.dir.join("questions.json");
    std::fs::write(
        &q,
        r#"{"questions":[{"id":"q1","kind":"name","question":"where is the checkout total computed","answers":[{"file":"crates/shop/src/lib.rs","anchor":"pub fn checkout","lines":[13,19]}]}]}"#,
    )
    .unwrap();
    h.ok(&["library", "list"]);
    let audit = h.dir.join("memory").join("audit.jsonl");
    let before = std::fs::read(&audit).unwrap_or_default();
    h.ok(&["bench", "--mode", "plain", "--root", repo.to_str().unwrap(), "--questions", q.to_str().unwrap(), "--quiet"]);
    h.ok(&["check", "nmcore.library", "nmlearn.library"]);
    let after = std::fs::read(&audit).unwrap_or_default();
    assert_eq!(String::from_utf8_lossy(&before), String::from_utf8_lossy(&after), "bench or check wrote to the audit log");
    assert!(!h.dir.join("memory").join("stacks").join("in-use.json").exists(), "bench recorded a server start");
    assert!(!h.dir.join("memory").join("projects").exists(), "bench wrote into project memory");
}

/// A server over MCP on standard input and output: one call at a time, the reply's text.
struct Server {
    child: std::process::Child,
    out: std::io::BufReader<std::process::ChildStdout>,
    id: u64,
}
impl Server {
    fn start(h: &Home, args: &[&str]) -> Self {
        use std::process::Stdio;
        let mut child = Command::new(EXE)
            .args(args)
            .env("NANO_HOME", &h.dir)
            .env("NANO_MODELS", nmtype::models_dir(workspace()))
            .env_remove("CLAUDE_PROJECT_DIR")
            .current_dir(&h.dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(std::fs::File::create(h.dir.join("serve.err")).unwrap())
            .spawn()
            .unwrap();
        let out = std::io::BufReader::new(child.stdout.take().unwrap());
        let mut s = Self { child, out, id: 0 };
        s.send(r#"{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}"#);
        s.line();
        s.send(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#);
        s
    }
    fn send(&mut self, m: &str) {
        use std::io::Write;
        let stdin = self.child.stdin.as_mut().unwrap();
        stdin.write_all(m.as_bytes()).unwrap();
        stdin.write_all(b"\n").unwrap();
        stdin.flush().unwrap();
    }
    fn line(&mut self) -> Value {
        use std::io::BufRead;
        let mut l = String::new();
        self.out.read_line(&mut l).unwrap();
        nmjson::parse(l.as_bytes()).unwrap()
    }
    fn pack(&mut self, question: &str) -> String {
        self.id += 1;
        let q = Value::String(question.to_owned()).encode().unwrap();
        self.send(&format!(r#"{{"jsonrpc":"2.0","id":{},"method":"tools/call","params":{{"name":"nm_context_pack","arguments":{{"question":{q},"mode":"fast"}}}}}}"#, self.id));
        let r = self.line();
        r.get("result").unwrap().get("content").unwrap().array().unwrap()[0].get("text").unwrap().string().unwrap().to_owned()
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        drop(self.child.stdin.take());
        let _ = self.child.wait();
    }
}

fn wait_s(secs: u64) {
    let t = std::time::Instant::now();
    while t.elapsed().as_secs() < secs {
        std::thread::park_timeout(std::time::Duration::from_millis(200));
    }
}

/// Through a running server: a stack holding a socket cartridge on the reranker, switched on from
/// the command line, reaches the next call with no restart and changes the reply; a stack holding a
/// zero-effect cartridge gives the base reply byte for byte; switching the use off gives the base
/// reply again; the server's start row names the sockets.
#[test]
fn a_socket_cartridge_switches_in_a_running_server_and_unplugs_to_the_base() {
    let h = Home::new("sockets").with_cards();
    let repo = h.dir.join("small-repo");
    copy_dir(&workspace().join("tests").join("fixtures").join("small-repo"), &repo);
    h.ok(&["library", "list"]);
    let mut lib = Library::open(&h.dir).unwrap();
    let spec = |id: &str| CartridgeSpec {
        id: id.into(),
        version: 1,
        family: "fundamentals".into(),
        domain: "rust-fundamentals".into(),
        scope: "general".into(),
        model: "qwen3-reranker-0.6b".into(),
        weights: "27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b".into(),
        layers: Some((27, 27)),
        name: id.into(),
        description: "A test filter at the reranker's final norm.".into(),
        made_from: "numbers written for the test".into(),
        effect: String::new(),
    };
    let zero = Filter::point("final_norm", vec![1.0; 1024], vec![0.0; 1024]);
    // Every other channel silenced: the reranker's scores move.
    let strong = Filter::point("final_norm", (0..1024).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect(), vec![0.0; 1024]);
    // Made for other bare weights: refused at the socket, by name.
    let mut other = spec("other-weights");
    other.weights = "0".repeat(64);
    lib.add_cartridge(&spec("zero-effect"), &zero, Value::Null, Vec::new(), false).unwrap();
    lib.add_cartridge(&spec("strong"), &strong, Value::Null, Vec::new(), false).unwrap();
    lib.add_cartridge(&other, &strong, Value::Null, Vec::new(), false).unwrap();
    for (id, c) in [("zero-stack", "zero-effect"), ("strong-stack", "strong"), ("other-stack", "other-weights")] {
        lib.add_stack(Stack { id: id.into(), version: 1, use_for: "job:test".into(), piece: None, cartridges: vec![c.into()], description: String::new(), hash: String::new(), trial: false }).unwrap();
    }
    lib.save().unwrap();
    let mut srv = Server::start(&h, &["serve", "--root", repo.to_str().unwrap(), "--whole", "--neural", "--memory", "off", "--dense", "off"]);
    let q = "where is the checkout total computed";
    let base = srv.pack(q);
    assert!(!base.is_empty());
    h.ok(&["library", "switch", "job:test", "strong-stack"]);
    wait_s(3);
    let strong_reply = srv.pack(q);
    h.ok(&["library", "switch", "job:test", "zero-stack"]);
    wait_s(3);
    let zero_reply = srv.pack(q);
    h.ok(&["library", "switch", "job:test", "off"]);
    wait_s(3);
    let off_reply = srv.pack(q);
    h.ok(&["library", "switch", "job:test", "other-stack"]);
    wait_s(3);
    let other_reply = srv.pack(q);
    drop(srv);
    assert_eq!(other_reply, base, "a cartridge made for other bare weights was seated");
    assert_ne!(strong_reply, base, "the switched stack did not reach the running server");
    assert_eq!(zero_reply, base, "a zero-effect cartridge changed the reply");
    assert_eq!(off_reply, base, "unplugging did not give the base reply");
    let rows = h.audit();
    assert_eq!(rows.iter().filter(|r| st(r, "kind") == "library-stacks").count(), 1, "one start row per serving process");
    let switches: Vec<&Value> = rows.iter().filter(|r| st(r, "kind") == "library-sockets").collect();
    assert!(switches.len() >= 4, "each switch audited");
    let refused = switches.last().unwrap().get("refused").unwrap().encode().unwrap();
    assert!(refused.contains("other-weights@1") && refused.contains("made for base"), "{refused}");
}
