//! S10 run (runbook `runs/2026-11-08-S10-runbook.md`, gates 4 and 5): `run s10 delta` keeps
//! a new scratch tree in step through the poll thread, then closes and reopens the session.
//! Writes a new file under `runs/`.
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::fmt::Write as _;
use std::path::Path;
use std::process::ExitCode;
use std::time::{Duration, Instant};

const QUESTION: &str = "What value is the constant POLL_LIMIT set to, and where is it defined?";

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("s10", FaultKind::Execution, m)
}
fn line(out: &mut String, s: impl AsRef<str>) {
    let _ = writeln!(out, "{}", s.as_ref());
}
fn wait(d: Duration) {
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let _ = rx.recv_timeout(d);
    drop(tx);
}
fn packets(v: &Value) -> Vec<Value> {
    v.get("packets").and_then(Value::array).map(<[Value]>::to_vec).unwrap_or_default()
}
fn field(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default()
}
fn observed(v: &Value, text: &str) -> bool {
    packets(v).iter().any(|p| field(p, "text").contains(text) && matches!(field(p, "status").as_str(), "" | "observed"))
}

fn delta(home: &Path) -> Result<String, Fault> {
    let t0 = Instant::now();
    let mut out = String::new();
    let stamp = nmtype::now_ms();
    let tree = home.join(format!("tests/stage-gates/s10/s10-tree-{stamp}"));
    let mem = home.join(format!("tests/stage-gates/s10/s10-mem-{stamp}"));
    std::fs::create_dir_all(tree.join("src")).map_err(|e| bad(e.to_string()))?;
    let file = tree.join("src/lib.rs");
    let w = |p: &Path, t: &str| std::fs::write(p, t).map_err(|e| bad(e.to_string()));
    w(&tree.join("Cargo.toml"), "[package]\nname = \"s10-probe\"\nversion = \"0.1.0\"\nedition = \"2024\"\n")?;
    w(&file, "//! Probe.\n/// How many polls a reader waits.\npub const POLL_LIMIT: u32 = 4;\n")?;
    w(&tree.join("src/other.rs"), "pub fn other() -> u32 {\n    7\n}\n")?;
    let org = nmapi::forage::Organism::open_with(home, &tree, &[], true, &mem)?;
    nmapi::tools::install(org).map_err(|e| bad(e.message))?;
    let org = nmapi::tools::installed().ok_or_else(|| bad("organism not installed"))?;
    let poller = nmapi::delta::start(org, Duration::from_secs(1));
    let api = nmapi::tools::api();
    let ask = || {
        api.dispatch(
            "nm_context_pack",
            &Value::Object(
                [("question".to_owned(), Value::String(QUESTION.into())), ("format".to_owned(), Value::String("json".into()))]
                    .into_iter()
                    .collect(),
            ),
        )
        .map_err(|e| bad(e.message))
    };
    let r1 = ask()?;
    line(&mut out, format!("reply 1 (version {}): {}", field(&r1, "version"), r1.encode().unwrap_or_default()));
    let v1 = observed(&r1, "POLL_LIMIT = 4");
    w(&file, "//! Probe.\n/// How many polls a reader waits.\npub const POLL_LIMIT: u32 = 9;\n")?;
    let changed_at = Instant::now();
    wait(Duration::from_millis(2500));
    let r2 = ask()?;
    line(&mut out, format!("reply 2, {} ms after the edit: {}", changed_at.elapsed().as_millis(), r2.encode().unwrap_or_default()));
    let status = api.dispatch("nm_status", &Value::Object(std::collections::BTreeMap::new())).map_err(|e| bad(e.message))?;
    line(&mut out, format!("nm_status delta: {}", status.get("delta").map(|d| d.encode().unwrap_or_default()).unwrap_or_default()));
    poller.stop();
    let new_seen = observed(&r2, "POLL_LIMIT = 9");
    let old_gone = !observed(&r2, "POLL_LIMIT = 4");
    let obsolete = packets(&r2).iter().any(|p| field(p, "obsolete").contains("src/lib.rs"));
    line(&mut out, format!("first reply observed V1: {v1}"));
    line(
        &mut out,
        format!(
            "GATE 4: V2 observed {new_seen}, no observed V1 {old_gone}, earlier episode marked obsolete {obsolete}: {}",
            if new_seen && old_gone && obsolete { "PASS" } else { "MISS" }
        ),
    );
    // Gate 5: close, reopen on the same memory folder.
    nmapi::session::close(org);
    let org2 = nmapi::forage::Organism::open_with(home, &tree, &[], true, &mem)?;
    let reopen = nmapi::session::open_budget(&org2, 600);
    let text = reopen.encode().unwrap_or_default();
    let tokens = text.len() / 4;
    let keys = ["identity", "objectives", "latest", "structural_summary_stale"].iter().all(|k| reopen.get(k).is_ok());
    let episodes = reopen.get("latest").and_then(Value::array).map_or(0, <[Value]>::len);
    line(&mut out, format!("reopen reply ({tokens} tokens, characters / 4): {text}"));
    line(
        &mut out,
        format!(
            "GATE 5: at most 600 tokens {}, identity, objectives, latest episodes ({episodes}) and stale flag present {keys}: {}",
            tokens <= 600,
            if tokens <= 600 && keys && episodes > 0 { "PASS" } else { "MISS" }
        ),
    );
    line(&mut out, format!("wall {} ms", t0.elapsed().as_millis()));
    Ok(out)
}

/// `run s10 delta`.
pub fn run(args: &[String], home: &Path) -> ExitCode {
    let part = args.get(2).map_or("delta", String::as_str);
    let stamp = nmtype::now_ms();
    let result = match part {
        "delta" => delta(home),
        other => Err(bad(format!("unknown S10 part {other:?}: delta"))),
    };
    match result {
        Ok(text) => {
            let p = home.join(format!("runs/S10-{part}-{stamp}.txt"));
            let _ = std::fs::write(&p, &text);
            print!("{text}");
            println!("report: {}", p.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            let _ = std::fs::write(home.join(format!("runs/S10-{part}-{stamp}.txt")), format!("FAULT {e}\n"));
            eprintln!("s10 {part}: {e}");
            ExitCode::FAILURE
        }
    }
}
