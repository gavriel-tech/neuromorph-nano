//! `neuromorph-nano coldstart --repo <path>`: the cold-start first-run measure on a repository the
//! Nano has not seen (completion roadmap Phase 5 item 2). It starts a warm daemon on the
//! repository as a child process with a fresh test memory (`tests/mcp-launcher/coldstart-<ms>`,
//! so nothing is cached from before; the repository itself is only read), and times from the
//! start: the material read and the tools ready (indexing), the first `nm_context_pack` (its own
//! latency and when it answered), the meaning index kept on disk (`dense/vectors.bin` first seen,
//! then settled: no growth for the settle time), and the idle writer's first notes. Then it stops
//! the daemon and writes `report.json` and `report.md` into `runs/coldstart-<ms>/`. The test memory
//! is removed at the end unless `--keep`.
use crate::harness::{self, Daemon, int, num, obj, s};
use nmjson::Value;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

pub const USAGE: &str = "usage: neuromorph-nano coldstart --repo <path> [--question \"<text>\"] [--timeout 30m] [--settle 20s]
                              [--port 7532] [--keep] [-- <serve options>]
  --repo      the unseen repository (only read; its memory goes to a fresh test folder)
  --question  the first question (default: where the main entry point is)
  --timeout   how long to wait for the dense index and the writer's first notes
  --settle    how long the dense index file must stay the same size to count as built
  --keep      keep the test memory (the dense index and notes) after the run";

/// The cold-start harness's settings.
#[derive(Debug)]
pub struct Config {
    pub repo: PathBuf,
    pub question: String,
    pub timeout: Duration,
    pub settle: Duration,
    pub port: u16,
    pub keep: bool,
    pub serve: Vec<String>,
}

/// Reads the harness's arguments (without the word `coldstart`).
/// # Errors
/// No `--repo`, or a malformed value, as one line.
pub fn parse_config(args: &[String]) -> Result<Config, String> {
    let own: Vec<String> = args.iter().take_while(|a| a.as_str() != "--").cloned().collect();
    let repo = harness::opt(&own, "--repo").or_else(|| harness::opt(&own, "--root")).ok_or("--repo <path> is needed")?;
    let port = harness::opt(&own, "--port").map_or(Ok(7532), |p| p.parse::<u16>().map_err(|_| format!("--port takes a port number, got {p}")))?;
    Ok(Config {
        repo: PathBuf::from(repo),
        question: harness::opt(&own, "--question").unwrap_or("Where is the main entry point of this project and what does it start?").to_owned(),
        timeout: harness::parse_duration(harness::opt(&own, "--timeout").unwrap_or("30m"))?,
        settle: harness::parse_duration(harness::opt(&own, "--settle").unwrap_or("20s"))?,
        port,
        keep: harness::has(&own, "--keep"),
        serve: harness::passthrough(args),
    })
}

/// The moments a cold start is measured by, in seconds since the daemon was started.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Moments {
    pub listening: Option<f64>,
    pub ready: Option<f64>,
    pub ready_reported: Option<f64>,
    pub drones_done: Option<f64>,
    pub first_pack_answered: Option<f64>,
    pub first_pack_ms: Option<f64>,
    pub dense_started: Option<f64>,
    pub dense_file_seen: Option<f64>,
    pub dense_settled: Option<f64>,
    pub dense_bytes: u64,
    pub writer_first_notes: Option<f64>,
    pub writer_line: Option<String>,
}

/// Reads the daemon's own time-stamped lines into the moments they mark: the start's `ready at`
/// (the daemon's own clock), the drones' intake `done at`, the dense lane's start, and the writer's
/// first `results written at idle`.
pub fn read_lines(lines: &[harness::Line], m: &mut Moments) {
    for (at, l) in lines {
        if m.ready_reported.is_none() && l.starts_with("start in the background") {
            m.ready_reported = harness::secs_after(l, "ready at");
        }
        if m.drones_done.is_none() && (l.contains("took in") || l.contains("taken in")) && l.contains("done at") {
            m.drones_done = harness::secs_after(l, "done at");
        }
        if m.dense_started.is_none() && l.starts_with("dense") && !l.contains("not started") {
            m.dense_started = Some(*at);
        }
        if m.writer_first_notes.is_none() && l.starts_with("writer:") && l.contains("written at idle") {
            m.writer_first_notes = Some(*at);
            m.writer_line = Some(l.clone());
        }
    }
}

/// `neuromorph-nano coldstart ...`.
pub fn run(home: &Path, args: &[String]) -> ExitCode {
    if harness::has(args, "--help") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    match coldstart(home, args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("coldstart: {e}\n{USAGE}");
            ExitCode::FAILURE
        }
    }
}

fn coldstart(home: &Path, args: &[String]) -> Result<bool, String> {
    let cfg = parse_config(args)?;
    if !cfg.repo.is_dir() {
        return Err(format!("--repo {} is not a folder", cfg.repo.display()));
    }
    let repo = nmsubs::repo::canon(&cfg.repo);
    let stamp = nmtype::now_ms();
    let memory_name = format!("mcp-launcher/coldstart-{stamp}");
    let memory = nmtype::tests_dir(&memory_name);
    let report = harness::report_dir(home, "coldstart")?;
    let mut serve = vec!["--root".to_owned(), repo.display().to_string(), "--memory-dir".to_owned(), memory_name, "--writer-idle".to_owned(), "on".to_owned()];
    serve.extend(cfg.serve.iter().cloned());
    println!("coldstart: {} with a fresh memory in {}", repo.display(), memory.display());
    let mut d = Daemon::spawn(&serve, cfg.port, &report.join("daemon.port"), &report.join("daemon.log"))?;
    let mut m = Moments::default();
    let mut fault = None;
    if d.wait_listening(Duration::from_secs(600)) {
        m.listening = Some(d.secs());
    } else {
        fault = Some("the daemon did not listen within 600 s".to_owned());
    }
    // The tools answer once the start has read the material: the first light call times indexing.
    let empty = Value::Object(std::collections::BTreeMap::new());
    if fault.is_none() {
        let st = harness::call(cfg.port, "nm_status", &empty, cfg.timeout);
        match st.result {
            Ok(_) => m.ready = Some(d.secs()),
            Err(e) => fault = Some(format!("nm_status: {e}")),
        }
    }
    let mut status_text = String::new();
    if fault.is_none() {
        let c = harness::call(cfg.port, "nm_context_pack", &harness::question_args(&cfg.question), cfg.timeout);
        match c.result {
            Ok(t) => {
                m.first_pack_answered = Some(d.secs());
                m.first_pack_ms = Some(c.ms);
                status_text = t.chars().take(600).collect();
            }
            Err(e) => fault = Some(format!("first nm_context_pack: {e}")),
        }
    }
    // Then quiet: the dense index builds and the writer gets its idle time; nothing is called.
    let vectors = memory.join("dense").join("vectors.bin");
    let started = Instant::now();
    let (mut last_len, mut same_since) = (0u64, Instant::now());
    while fault.is_none() && started.elapsed() < cfg.timeout && d.alive() {
        if let Ok(lines) = d.lines.lock() {
            read_lines(&lines, &mut m);
        }
        let len = std::fs::metadata(&vectors).map(|x| x.len()).unwrap_or(0);
        if len > 0 && m.dense_file_seen.is_none() {
            m.dense_file_seen = Some(d.secs());
        }
        if len != last_len {
            last_len = len;
            same_since = Instant::now();
            m.dense_settled = None;
        } else if len > 0 && m.dense_settled.is_none() && same_since.elapsed() >= cfg.settle {
            m.dense_settled = Some(d.secs() - cfg.settle.as_secs_f64());
        }
        m.dense_bytes = len;
        if m.dense_settled.is_some() && m.writer_first_notes.is_some() {
            break;
        }
        harness::park(Duration::from_secs(1));
    }
    if !d.alive() && fault.is_none() {
        fault = Some("the daemon ended during the run".to_owned());
    }
    let (clean, stop_ms, code) = d.stop(Duration::from_secs(60));
    if let Ok(lines) = d.lines.lock() {
        read_lines(&lines, &mut m);
    }
    let tail = d.tail(40);
    if !cfg.keep {
        let _ = std::fs::remove_dir_all(&memory);
    }
    let passed = fault.is_none() && m.first_pack_answered.is_some() && m.dense_settled.is_some() && m.writer_first_notes.is_some() && clean;
    let json = report_json(&cfg, &repo, &m, passed, fault.as_deref(), (clean, stop_ms, code), &tail, &status_text, stamp);
    let md = report_md(&repo, &m, passed, fault.as_deref(), stop_ms);
    harness::write_report(&report, &json, &md)?;
    println!("{md}");
    println!("coldstart: report in {}", report.display());
    Ok(passed)
}

fn opt_num(x: Option<f64>) -> Value {
    x.map_or(Value::Null, num)
}

#[allow(clippy::too_many_arguments)]
fn report_json(
    cfg: &Config,
    repo: &Path,
    m: &Moments,
    passed: bool,
    fault: Option<&str>,
    stop: (bool, u128, Option<i32>),
    tail: &[String],
    first_answer: &str,
    stamp: u64,
) -> Value {
    obj(vec![
        ("kind", s("coldstart")),
        ("started_ms", int(stamp)),
        ("passed", Value::Bool(passed)),
        ("fault", fault.map_or(Value::Null, s)),
        ("repo", s(&repo.display().to_string())),
        ("question", s(&cfg.question)),
        ("serve_options", Value::Array(cfg.serve.iter().map(|x| s(x)).collect())),
        (
            "seconds_since_start",
            obj(vec![
                ("listening", opt_num(m.listening)),
                ("tools_ready", opt_num(m.ready)),
                ("ready_reported_by_daemon", opt_num(m.ready_reported)),
                ("drones_intake_done", opt_num(m.drones_done)),
                ("first_pack_answered", opt_num(m.first_pack_answered)),
                ("dense_lane_started", opt_num(m.dense_started)),
                ("dense_file_seen", opt_num(m.dense_file_seen)),
                ("dense_built", opt_num(m.dense_settled)),
                ("writer_first_notes", opt_num(m.writer_first_notes)),
            ]),
        ),
        ("first_pack_ms", opt_num(m.first_pack_ms)),
        ("first_pack_text_head", s(first_answer)),
        ("dense_bytes", int(m.dense_bytes)),
        ("writer_line", m.writer_line.as_deref().map_or(Value::Null, s)),
        (
            "shutdown",
            obj(vec![
                ("clean", Value::Bool(stop.0)),
                ("ms", int(u64::try_from(stop.1).unwrap_or(u64::MAX))),
                ("exit_code", stop.2.map_or(Value::Null, |c| Value::Number(c.to_string()))),
            ]),
        ),
        ("daemon_log_tail", Value::Array(tail.iter().map(|x| s(x)).collect())),
    ])
}

fn report_md(repo: &Path, m: &Moments, passed: bool, fault: Option<&str>, stop_ms: u128) -> String {
    let f = |x: Option<f64>| x.map_or_else(|| "not reached".to_owned(), |y| format!("{y:.1} s"));
    let mut t = format!("# Cold-start report\n\nResult: {}\n\nRepository: {}\n", if passed { "PASS" } else { "FAIL" }, repo.display());
    if let Some(e) = fault {
        t.push_str(&format!("\nFault: {e}\n"));
    }
    t.push_str(&format!(
        "\n| Moment (since the daemon started) | Time |\n|---|---|\n| Listening | {} |\n| Indexing done, tools ready | {} (daemon's own clock: {}) |\n| Drones' intake done | {} |\n| First nm_context_pack answered | {} ({} for the call) |\n| Dense lane started | {} |\n| Dense index built ({} bytes on disk) | {} |\n| Idle writer's first notes | {} |\n| Shutdown | {stop_ms} ms |\n",
        f(m.listening),
        f(m.ready),
        f(m.ready_reported),
        f(m.drones_done),
        f(m.first_pack_answered),
        m.first_pack_ms.map_or_else(|| "n/a".to_owned(), |x| format!("{x:.0} ms")),
        f(m.dense_started),
        m.dense_bytes,
        f(m.dense_settled),
        f(m.writer_first_notes),
    ));
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config() {
        let a: Vec<String> = ["--repo", "D:\\other\\repo", "--timeout", "10m", "--keep", "--", "--neural"].iter().map(|x| (*x).to_owned()).collect();
        let c = parse_config(&a).unwrap();
        assert_eq!(c.repo, PathBuf::from("D:\\other\\repo"));
        assert_eq!(c.timeout.as_secs(), 600);
        assert_eq!(c.settle.as_secs(), 20);
        assert_eq!(c.port, 7532);
        assert!(c.keep);
        assert_eq!(c.serve, vec!["--neural".to_owned()]);
        assert!(parse_config(&[]).is_err());
        assert!(parse_config(&["--repo".to_owned(), "x".to_owned(), "--settle".to_owned(), "never".to_owned()]).is_err());
    }

    #[test]
    fn reads_daemon_lines() {
        let lines: Vec<harness::Line> = vec![
            (0.4, "memory: D:\\Nano\\tests\\mcp-launcher\\coldstart-1".to_owned()),
            (6.1, "start in the background: open (material read, memory opened) 5.20 s, configure 0.40 s, session open 0.10 s; ready at 5.70 s".to_owned()),
            (6.3, "dense (plain path default): embedder bound; at 6.20 s".to_owned()),
            (9.0, "flow: drones took in 812 pieces; done at 8.90 s".to_owned()),
            (80.0, "writer: waits for 7000 MiB of free commit memory".to_owned()),
            (95.5, "writer: 14 results written at idle (14 held)".to_owned()),
            (130.0, "writer: 9 results written at idle (23 held)".to_owned()),
        ];
        let mut m = Moments::default();
        read_lines(&lines, &mut m);
        assert_eq!(m.ready_reported, Some(5.7));
        assert_eq!(m.drones_done, Some(8.9));
        assert_eq!(m.dense_started, Some(6.3));
        assert_eq!(m.writer_first_notes, Some(95.5));
        assert!(m.writer_line.as_deref().unwrap().contains("14 results"));
        let mut n = Moments::default();
        read_lines(&[(1.0, "dense (plain path default): not started (no card); the path runs without it".to_owned())], &mut n);
        assert_eq!(n.dense_started, None);
        assert!(report_md(Path::new("D:\\r"), &m, false, Some("x"), 5).contains("95.5 s"));
    }
}
