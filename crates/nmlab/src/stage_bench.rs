//! `nano-lab stage-bench`: the Stage 5 measuring harness (WP-C0, M1 and M2 of
//! the 2026-11-22 action plan).
//!
//! It runs `neuromorph-nano bench --mode board` as a child process under the shared graphics card
//! lock (`scratch/LOCKS/card.lock`) and reads what the bench prints: the board's score line, the
//! seconds of every question, and the `NANO_STAGE_TIMES` lines on standard error. Two commands:
//!
//! - `base`: one board run per named set, in order, under one lock: the baseline (M1).
//! - `pair`: a paired control and candidate comparison (M2). One warm-up run (discarded), then
//!   control, candidate, control, ..., control, with at least six candidate runs. Each candidate
//!   run is judged against the two control runs beside it (the measuring rule of the
//!   inference-optimisation method: beat both neighbours, nothing worse by more than 3 percent).
//!
//! Every report goes to `runs/STAGE-BENCH-<ms>.txt`: the plain text table followed by one JSON
//! object with every number, and the raw outputs of every run under `runs/stage-bench-<ms>/`.
//! Child processes get `TEMP` and `TMP` inside `scratch/` so nothing is written to C:.
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

type R<T> = Result<T, String>;

const USAGE: &str = "usage:
  nano-lab stage-bench base [--exe <path>] [--n <count>] [--wait <s>] [--env K=V,K=V] --set <label>|<root>|<questions>|<dense dir or ->|<on|off> [--set ...] [-- <serve options>]
  nano-lab stage-bench pair [--exe <path>] [--n <count>] [--samples <k>] [--wait <s>] --set <label>|<root>|<questions>|<dense dir or ->|<on|off>
                            [--control K=V,K=V] [--candidate K=V,K=V] [-- <serve options>]
  nano-lab stage-bench lock-status

The default program is scratch/target-claude/release/neuromorph-nano.exe. Every run sets
NANO_STAGE_TIMES=1. --wait is how long to wait for a held card lock (default 3600 s).";

/// Milliseconds since the epoch.
fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

// ---------------------------------------------------------------------------------------------
// The graphics card lock.

/// The shared graphics card lock: a file created only if it does not exist, holding
/// `{"pid": .., "started_utc": "..", "cmd": ".."}`. A lock whose process is gone is stale and is
/// cleared. Dropping the guard removes the file if it is still this process's lock.
#[derive(Debug)]
pub struct CardLock {
    path: PathBuf,
    pid: u32,
}

impl CardLock {
    /// The project's card lock file.
    #[must_use]
    pub fn default_path(home: &Path) -> PathBuf {
        home.join("scratch").join("LOCKS").join("card.lock")
    }

    /// Take the lock at `path`, waiting up to `wait` while a live process holds it.
    /// # Errors
    /// When the lock is still held by a live process after `wait`, or the file cannot be written.
    pub fn acquire_at(path: &Path, cmd: &str, wait: Duration) -> R<Self> {
        let pid = std::process::id();
        let body = format!("{{\"pid\": {pid}, \"started_utc\": \"{}\", \"cmd\": \"{}\"}}\n", nmmusc::pack::utc_stamp(now_ms()), json_escape(cmd));
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let start = Instant::now();
        let mut told = false;
        loop {
            match OpenOptions::new().write(true).create_new(true).open(path) {
                Ok(mut f) => {
                    f.write_all(body.as_bytes()).map_err(|e| format!("{}: {e}", path.display()))?;
                    return Ok(Self { path: path.to_path_buf(), pid });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    let holder = holder_pid(path);
                    match holder {
                        Some(h) if h != pid && process_alive(h) => {}
                        Some(h) if h == pid => return Err(format!("{}: already held by this process", path.display())),
                        // A lock with no readable pid, or whose process has ended, is stale only when
                        // no other card program is running: another agent may write the lock from a
                        // short-lived shell while its bench runs under other processes (seen
                        // 2026-11-22: a lock naming an ended pid while that agent's benches ran).
                        _ if !other_card_programs(pid).is_empty() => {}
                        // Another agent may run a sequence of benches under one lock written from a
                        // short-lived shell, with gaps between its processes (seen twice on
                        // 2026-11-22): its lock is honoured until it is older than `STALE_AFTER`.
                        _ if lock_age(path).is_some_and(|a| a < STALE_AFTER) => {}
                        _ => {
                            let _ = fs::remove_file(path);
                            continue;
                        }
                    }
                    if start.elapsed() >= wait {
                        return Err(format!("{}: held by live process {}", path.display(), holder.unwrap_or(0)));
                    }
                    if !told {
                        eprintln!("stage-bench: card lock held by process {}, waiting", holder.unwrap_or(0));
                        told = true;
                    }
                    std::thread::park_timeout(Duration::from_secs(2).min(wait.saturating_sub(start.elapsed()).max(Duration::from_millis(50))));
                }
                Err(e) => return Err(format!("{}: {e}", path.display())),
            }
        }
    }
}

impl Drop for CardLock {
    fn drop(&mut self) {
        if holder_pid(&self.path) == Some(self.pid) {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// The pid written in a lock file, if it can be read.
fn holder_pid(path: &Path) -> Option<u32> {
    let text = fs::read_to_string(path).ok()?;
    let at = text.find("\"pid\"")?;
    let rest = text.get(at + 5..)?;
    let rest = rest.trim_start_matches([' ', ':', '\t']);
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// Whether a process with this id is running.
fn process_alive(pid: u32) -> bool {
    if pid == std::process::id() {
        return true;
    }
    #[cfg(windows)]
    {
        Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"])
            .output()
            .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains(&format!("\"{pid}\"")))
    }
    #[cfg(not(windows))]
    {
        Command::new("kill").args(["-0", &pid.to_string()]).output().is_ok_and(|o| o.status.success())
    }
}

/// How old a lock whose process has ended must be before it is cleared as stale.
const STALE_AFTER: Duration = Duration::from_secs(2 * 3600);

/// How long ago the lock file was written.
fn lock_age(path: &Path) -> Option<Duration> {
    fs::metadata(path).ok()?.modified().ok()?.elapsed().ok()
}

/// The programs that use the graphics card for the organism.
const CARD_PROGRAMS: [&str; 2] = ["neuromorph-nano.exe", "nano-lab.exe"];

/// The ids of running card programs other than this process.
fn other_card_programs(own: u32) -> Vec<u32> {
    let mut out = Vec::new();
    #[cfg(windows)]
    for image in CARD_PROGRAMS {
        let Ok(o) = Command::new("tasklist").args(["/FI", &format!("IMAGENAME eq {image}"), "/NH", "/FO", "CSV"]).output() else { continue };
        for line in String::from_utf8_lossy(&o.stdout).lines() {
            // "name","pid","session","n","mem"
            if let Some(pid) = line.split(',').nth(1).map(|f| f.trim_matches('"')).and_then(|f| f.parse::<u32>().ok())
                && pid != own
            {
                out.push(pid);
            }
        }
    }
    #[cfg(not(windows))]
    {
        let _ = (own, CARD_PROGRAMS);
    }
    out
}

fn json_escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o
}

// ---------------------------------------------------------------------------------------------
// Parsing what the bench prints.

/// One named question set.
#[derive(Debug, Clone)]
struct SetSpec {
    label: String,
    root: String,
    questions: String,
    dense_dir: Option<String>,
    dense: String,
}

fn parse_set(s: &str) -> R<SetSpec> {
    let parts: Vec<&str> = s.split('|').collect();
    let [label, root, questions, dense_dir, dense] = parts.as_slice() else {
        return Err(format!("--set takes label|root|questions|dense dir or -|on or off, got {s}"));
    };
    if *dense != "on" && *dense != "off" {
        return Err(format!("--set {label}: the meaning lane is on or off, got {dense}"));
    }
    Ok(SetSpec {
        label: (*label).to_owned(),
        root: (*root).to_owned(),
        questions: (*questions).to_owned(),
        dense_dir: (*dense_dir != "-").then(|| (*dense_dir).to_owned()),
        dense: (*dense).to_owned(),
    })
}

/// The board's score line of the served flow.
#[derive(Debug, Clone, Default, PartialEq)]
struct Score {
    first: u32,
    top3: u32,
    pack: u32,
    every: u32,
    cited: u32,
    s_med: f64,
    s_worst: f64,
    decoys: String,
}

/// The stage times of one question (`NANO_STAGE_TIMES`), in milliseconds.
#[derive(Debug, Clone, Default)]
struct Stages {
    signals: Option<f64>,
    evidence: Option<f64>,
    drone_kind: Option<f64>,
    sense: Option<f64>,
    compose: Option<f64>,
    g1: Option<f64>,
    pack: Option<f64>,
    total: Option<f64>,
    deep: Option<f64>,
    tentacle_read: Option<f64>,
    g2_own: Option<f64>,
    g2_order_reads: Option<f64>,
    judge_duels: Option<f64>,
    judge_close: Option<f64>,
}

/// The stage names in report order, with how to read each from `Stages`.
const STAGE_NAMES: [&str; 14] = [
    "signals_meaning",
    "evidence",
    "drone_kind",
    "sense",
    "compose",
    "g1",
    "tentacle_read",
    "deep_check",
    "g2_own",
    "g2_order_reads",
    "judge_duels",
    "judge_close",
    "pack",
    "total",
];

impl Stages {
    fn get(&self, name: &str) -> Option<f64> {
        match name {
            "signals_meaning" => self.signals,
            "evidence" => self.evidence,
            "drone_kind" => self.drone_kind,
            "sense" => self.sense,
            "compose" => self.compose,
            "g1" => self.g1,
            "tentacle_read" => self.tentacle_read,
            "deep_check" => self.deep,
            "g2_own" => self.g2_own,
            "g2_order_reads" => self.g2_order_reads,
            "judge_duels" => self.judge_duels,
            "judge_close" => self.judge_close,
            "pack" => self.pack,
            "total" => self.total,
            _ => None,
        }
    }
}

/// The number after `key` in `line` (the first token that parses as a number).
fn number_after(line: &str, key: &str) -> Option<f64> {
    let at = line.find(key)?;
    line.get(at + key.len()..)?
        .split(|c: char| c.is_whitespace() || c == ',' || c == ';' || c == '(' || c == ')')
        .filter(|t| !t.is_empty())
        .find_map(|t| t.parse::<f64>().ok())
}

/// The stage times of every question, in the order printed. A question's record starts at its
/// `signals and meaning` line and ends at the summary line that holds `total`; the drone kind line
/// of the background thread may come anywhere in between.
fn parse_stages(err: &str) -> Vec<Stages> {
    let mut out = Vec::new();
    let mut cur = Stages::default();
    let ms = |s: Option<f64>| s.map(|v| v * 1000.0);
    for line in err.lines().filter(|l| l.contains("STAGE_TIMES")) {
        if line.contains("signals and meaning") {
            cur.signals = ms(number_after(line, "signals and meaning"));
            cur.evidence = ms(number_after(line, "evidence"));
        } else if line.contains("drone kind") {
            cur.drone_kind = ms(number_after(line, "(async bg)").or_else(|| number_after(line, "drone kind")));
        } else if line.contains(" total ") {
            cur.sense = ms(number_after(line, "sense"));
            cur.compose = ms(number_after(line, "compose"));
            cur.g1 = ms(number_after(line, " g1 "));
            cur.pack = ms(number_after(line, " pack "));
            cur.total = ms(number_after(line, " total "));
            cur.deep = ms(number_after(line, "; deep"));
            cur.tentacle_read = ms(number_after(line, ") read"));
            cur.g2_own = number_after(line, "g2 own");
            cur.g2_order_reads = number_after(line, "order reads");
            cur.judge_duels = number_after(line, "duels judge");
            cur.judge_close = number_after(line, "close");
            out.push(std::mem::take(&mut cur));
        }
    }
    out
}

/// The served flow's score line, from the board's text.
fn parse_score(text: &str) -> Option<Score> {
    let line = text.lines().find(|l| l.split_whitespace().next() == Some("flow") && l.contains('|'))?;
    let t: Vec<&str> = line.split_whitespace().collect();
    // flow first top3 pack every cited tokmed toktotal smed sworst decoys | ...
    let n = |i: usize| t.get(i).and_then(|v| v.parse::<u32>().ok());
    let f = |i: usize| t.get(i).and_then(|v| v.parse::<f64>().ok());
    Some(Score {
        first: n(1)?,
        top3: n(2)?,
        pack: n(3)?,
        every: n(4)?,
        cited: n(5)?,
        s_med: f(8)?,
        s_worst: f(9)?,
        decoys: t.get(10).map_or_else(String::new, |s| (*s).to_owned()),
    })
}

/// The seconds of every question, in order (per-question lines end with `s <seconds>`).
fn parse_question_seconds(text: &str) -> Vec<f64> {
    text.lines()
        .filter(|l| l.contains(" flow ") && l.contains(" tok "))
        .filter_map(|l| {
            let t: Vec<&str> = l.split_whitespace().collect();
            let i = t.iter().rposition(|w| *w == "s")?;
            t.get(i + 1)?.parse::<f64>().ok()
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Statistics.

fn quantile(v: &[f64], q: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss, reason = "an index into a short sorted list")]
    let i = ((s.len() - 1) as f64 * q).round() as usize;
    s.get(i.min(s.len() - 1)).copied()
}

#[derive(Debug, Clone, Copy, Default)]
struct Summary {
    n: usize,
    median: f64,
    p95: f64,
    min: f64,
    max: f64,
}

fn summarise(v: &[f64]) -> Option<Summary> {
    Some(Summary { n: v.len(), median: quantile(v, 0.5)?, p95: quantile(v, 0.95)?, min: quantile(v, 0.0)?, max: quantile(v, 1.0)? })
}

// ---------------------------------------------------------------------------------------------
// Running the bench.

/// The result of one board run.
#[derive(Debug, Clone)]
struct RunOut {
    arm: String,
    set: String,
    score: Option<Score>,
    /// Seconds of every question after the first, in milliseconds.
    warm_ms: Vec<f64>,
    stages: Vec<Stages>,
    wall_s: f64,
    out_file: PathBuf,
}

struct Ctx<'a> {
    home: &'a Path,
    exe: PathBuf,
    n: Option<String>,
    serve: Vec<String>,
    raw: PathBuf,
}

fn run_once(ctx: &Ctx<'_>, set: &SetSpec, arm: &str, idx: usize, env: &[(String, String)]) -> R<RunOut> {
    let tag = format!("{}-{arm}-{idx:02}", set.label);
    let out_file = ctx.raw.join(format!("{tag}.txt"));
    let err_file = ctx.raw.join(format!("{tag}.err"));
    let tmp = ctx.home.join("scratch").join("stage-bench-tmp");
    fs::create_dir_all(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    let mut args: Vec<String> = vec!["bench".into(), "--mode".into(), "board".into(), "--root".into(), set.root.clone(), "--questions".into(), set.questions.clone(), "--paths".into(), "flow".into()];
    if let Some(n) = &ctx.n {
        args.extend(["--n".into(), n.clone()]);
    }
    args.push("--".into());
    args.extend(["--dense".into(), set.dense.clone()]);
    args.extend(ctx.serve.iter().cloned());
    let mut cmd = Command::new(&ctx.exe);
    cmd.args(&args)
        .current_dir(ctx.home)
        .env("NANO_STAGE_TIMES", "1")
        .env("TEMP", &tmp)
        .env("TMP", &tmp)
        .stdin(Stdio::null())
        .stdout(File::create(&out_file).map_err(|e| format!("{}: {e}", out_file.display()))?)
        .stderr(File::create(&err_file).map_err(|e| format!("{}: {e}", err_file.display()))?);
    if let Some(d) = &set.dense_dir {
        cmd.env("NANO_DENSE_DIR", d);
    }
    for (k, v) in env {
        cmd.env(k, v);
    }
    eprintln!("stage-bench: {tag} start");
    let t0 = Instant::now();
    let mut child = cmd.spawn().map_err(|e| format!("{}: {e}", ctx.exe.display()))?;
    // A watchdog for the exit hang seen before T-086 was fixed: once the board is written, the
    // process gets 60 s to end on its own.
    let mut written_at: Option<Instant> = None;
    loop {
        if child.try_wait().map_err(|e| e.to_string())?.is_some() {
            break;
        }
        if written_at.is_none() && fs::read_to_string(&out_file).is_ok_and(|t| t.contains("written:")) {
            written_at = Some(Instant::now());
        }
        if written_at.is_some_and(|w| w.elapsed() > Duration::from_secs(60)) {
            eprintln!("stage-bench: {tag} did not exit 60 s after its board, ended");
            let _ = child.kill();
            let _ = child.wait();
            break;
        }
        std::thread::park_timeout(Duration::from_millis(500));
    }
    let wall_s = t0.elapsed().as_secs_f64();
    let text = fs::read_to_string(&out_file).unwrap_or_default();
    let err = fs::read_to_string(&err_file).unwrap_or_default();
    let score = parse_score(&text);
    let secs = parse_question_seconds(&text);
    let warm_ms: Vec<f64> = secs.iter().skip(1).map(|s| s * 1000.0).collect();
    let stages = parse_stages(&err);
    eprintln!(
        "stage-bench: {tag} done in {wall_s:.0} s: {}",
        score.as_ref().map_or_else(|| "no score line (see the .err file)".to_owned(), |s| format!("first {} top3 {} pack {} every {} decoys {} med {:.3} s", s.first, s.top3, s.pack, s.every, s.decoys, s.s_med))
    );
    Ok(RunOut { arm: arm.to_owned(), set: set.label.clone(), score, warm_ms, stages: stages.into_iter().skip(1).collect(), wall_s, out_file })
}

fn parse_env(s: &str) -> R<Vec<(String, String)>> {
    s.split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| p.split_once('=').map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned())).ok_or_else(|| format!("environment setting {p}: expected K=V")))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Reports.

fn stage_table(out: &mut String, label: &str, runs: &[&RunOut]) {
    let _ = writeln!(out, "  stages, {label} (ms per warm question: median / p95 / n)");
    for name in STAGE_NAMES {
        let v: Vec<f64> = runs.iter().flat_map(|r| r.stages.iter().filter_map(|s| s.get(name))).collect();
        if let Some(s) = summarise(&v) {
            let _ = writeln!(out, "    {name:<16} {:>8.1} {:>8.1} {:>5}", s.median, s.p95, s.n);
        }
    }
}

fn score_line(s: Option<&Score>) -> String {
    s.map_or_else(
        || "no score line".to_owned(),
        |s| format!("first {:>3} top3 {:>3} pack {:>3} every {:>3} cited {:>3} decoys {:>6} s med {:.3} worst {:.3}", s.first, s.top3, s.pack, s.every, s.cited, s.decoys, s.s_med, s.s_worst),
    )
}

fn json_runs(runs: &[RunOut]) -> String {
    let mut j = String::from("[");
    for (i, r) in runs.iter().enumerate() {
        if i > 0 {
            j.push(',');
        }
        let _ = write!(j, "{{\"arm\":\"{}\",\"set\":\"{}\",\"wall_s\":{:.1},\"out\":\"{}\"", json_escape(&r.arm), json_escape(&r.set), r.wall_s, json_escape(&r.out_file.display().to_string()));
        if let Some(s) = &r.score {
            let _ = write!(
                j,
                ",\"first\":{},\"top3\":{},\"pack\":{},\"every\":{},\"cited\":{},\"decoys\":\"{}\",\"s_med\":{:.3},\"s_worst\":{:.3}",
                s.first,
                s.top3,
                s.pack,
                s.every,
                s.cited,
                json_escape(&s.decoys),
                s.s_med,
                s.s_worst
            );
        }
        if let Some(s) = summarise(&r.warm_ms) {
            let _ = write!(j, ",\"warm_ms\":{{\"n\":{},\"median\":{:.1},\"p95\":{:.1},\"min\":{:.1},\"max\":{:.1}}}", s.n, s.median, s.p95, s.min, s.max);
        }
        j.push_str(",\"stages_median_ms\":{");
        let mut first = true;
        for name in STAGE_NAMES {
            let v: Vec<f64> = r.stages.iter().filter_map(|s| s.get(name)).collect();
            if let Some(m) = quantile(&v, 0.5) {
                if !first {
                    j.push(',');
                }
                first = false;
                let _ = write!(j, "\"{name}\":{m:.1}");
            }
        }
        j.push_str("}}");
    }
    j.push(']');
    j
}

fn write_report(home: &Path, stamp: u64, text: &str, json: &str) -> R<PathBuf> {
    let path = home.join("runs").join(format!("STAGE-BENCH-{stamp}.txt"));
    let body = format!("{text}\njson:\n{json}\n");
    fs::write(&path, body).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}

// ---------------------------------------------------------------------------------------------
// Commands.

struct Opts {
    exe: PathBuf,
    n: Option<String>,
    samples: usize,
    wait: Duration,
    sets: Vec<SetSpec>,
    control: Vec<(String, String)>,
    candidate: Vec<(String, String)>,
    serve: Vec<String>,
}

fn parse_opts(home: &Path, args: &[String]) -> R<Opts> {
    let mut o = Opts {
        exe: home.join("scratch").join("target-claude").join("release").join(if cfg!(windows) { "neuromorph-nano.exe" } else { "neuromorph-nano" }),
        n: None,
        samples: 6,
        wait: Duration::from_secs(3600),
        sets: Vec::new(),
        control: Vec::new(),
        candidate: Vec::new(),
        serve: Vec::new(),
    };
    let mut i = 0;
    while let Some(a) = args.get(i) {
        let val = |i: usize| args.get(i + 1).cloned().ok_or_else(|| format!("{a} takes a value"));
        match a.as_str() {
            "--" => {
                o.serve = args.get(i + 1..).map(<[String]>::to_vec).unwrap_or_default();
                break;
            }
            "--exe" => o.exe = PathBuf::from(val(i)?),
            "--n" => o.n = Some(val(i)?),
            "--samples" => o.samples = val(i)?.parse().map_err(|_| "--samples takes a count".to_owned())?,
            "--wait" => o.wait = Duration::from_secs(val(i)?.parse().map_err(|_| "--wait takes seconds".to_owned())?),
            "--set" => o.sets.push(parse_set(&val(i)?)?),
            "--control" | "--env" => o.control = parse_env(&val(i)?)?,
            "--candidate" => o.candidate = parse_env(&val(i)?)?,
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
        i += 2;
    }
    if o.sets.is_empty() {
        return Err(format!("no --set given\n{USAGE}"));
    }
    Ok(o)
}

fn cmd_base(home: &Path, args: &[String]) -> R<PathBuf> {
    let o = parse_opts(home, args)?;
    let stamp = now_ms();
    let raw = home.join("runs").join(format!("stage-bench-{stamp}"));
    fs::create_dir_all(&raw).map_err(|e| format!("{}: {e}", raw.display()))?;
    let _lock = CardLock::acquire_at(&CardLock::default_path(home), &format!("nano-lab stage-bench base ({} sets)", o.sets.len()), o.wait)?;
    let ctx = Ctx { home, exe: o.exe.clone(), n: o.n.clone(), serve: o.serve.clone(), raw };
    let mut runs = Vec::new();
    for set in &o.sets {
        runs.push(run_once(&ctx, set, "base", 0, &o.control)?);
    }
    let mut text = String::new();
    let _ = writeln!(text, "stage-bench base  {}  program {}  env {:?}  serve options: --dense <per set> {}", nmmusc::pack::utc_stamp(stamp), o.exe.display(), o.control, o.serve.join(" "));
    for r in &runs {
        let _ = writeln!(text, "\n{:<12} {}", r.set, score_line(r.score.as_ref()));
        if let Some(s) = summarise(&r.warm_ms) {
            let _ = writeln!(text, "  warm question ms: median {:.0} p95 {:.0} min {:.0} max {:.0} (n {})", s.median, s.p95, s.min, s.max, s.n);
        }
        stage_table(&mut text, &r.set, &[r]);
        let _ = writeln!(text, "  raw: {}", r.out_file.display());
    }
    let tot = |f: fn(&Score) -> u32| runs.iter().filter_map(|r| r.score.as_ref().map(f)).sum::<u32>();
    let _ = writeln!(text, "\nall sets: first {} top3 {} pack {} every {} cited {}", tot(|s| s.first), tot(|s| s.top3), tot(|s| s.pack), tot(|s| s.every), tot(|s| s.cited));
    let json = format!("{{\"kind\":\"base\",\"stamp\":{stamp},\"exe\":\"{}\",\"runs\":{}}}", json_escape(&o.exe.display().to_string()), json_runs(&runs));
    print!("{text}");
    write_report(home, stamp, &text, &json)
}

fn cmd_pair(home: &Path, args: &[String]) -> R<PathBuf> {
    let o = parse_opts(home, args)?;
    let [set] = o.sets.as_slice() else {
        return Err("pair takes exactly one --set".to_owned());
    };
    if o.samples < 6 {
        return Err("pair needs at least 6 candidate samples".to_owned());
    }
    let stamp = now_ms();
    let raw = home.join("runs").join(format!("stage-bench-{stamp}"));
    fs::create_dir_all(&raw).map_err(|e| format!("{}: {e}", raw.display()))?;
    let _lock = CardLock::acquire_at(&CardLock::default_path(home), &format!("nano-lab stage-bench pair {}", set.label), o.wait)?;
    let ctx = Ctx { home, exe: o.exe.clone(), n: o.n.clone(), serve: o.serve.clone(), raw };
    let _warm = run_once(&ctx, set, "warmup", 0, &o.control)?;
    let mut controls = vec![run_once(&ctx, set, "control", 0, &o.control)?];
    let mut cands = Vec::new();
    for k in 0..o.samples {
        cands.push(run_once(&ctx, set, "candidate", k, &o.candidate)?);
        controls.push(run_once(&ctx, set, "control", k + 1, &o.control)?);
    }
    let med = |r: &RunOut| quantile(&r.warm_ms, 0.5).unwrap_or(f64::NAN);
    // Each candidate against the two controls beside it.
    let mut wins = 0;
    let mut pair_lines = String::new();
    for (k, c) in cands.iter().enumerate() {
        let (a, b) = (controls.get(k).map_or(f64::NAN, med), controls.get(k + 1).map_or(f64::NAN, med));
        let cm = med(c);
        let beat = cm < a && cm < b;
        wins += usize::from(beat);
        let _ = writeln!(pair_lines, "  sample {k}: control {a:.0} ms, candidate {cm:.0} ms, control {b:.0} ms: {}", if beat { "beats both" } else { "does not beat both" });
    }
    let pool = |rs: &[RunOut]| rs.iter().flat_map(|r| r.warm_ms.iter().copied()).collect::<Vec<f64>>();
    let (cs, xs) = (summarise(&pool(&controls)), summarise(&pool(&cands)));
    let mut text = String::new();
    let _ = writeln!(text, "stage-bench pair  {}  set {}  program {}", nmmusc::pack::utc_stamp(stamp), set.label, o.exe.display());
    let _ = writeln!(text, "control env: {:?}\ncandidate env: {:?}\nserve options: --dense {} {}", o.control, o.candidate, set.dense, o.serve.join(" "));
    let _ = writeln!(text, "\npaired runs (median warm question ms per run):\n{pair_lines}");
    let _ = writeln!(text, "candidate beats both neighbouring controls on {wins} of {} samples", cands.len());
    if let (Some(c), Some(x)) = (cs, xs) {
        let d = x.median - c.median;
        let _ = writeln!(text, "pooled warm question ms: control median {:.0} p95 {:.0} min {:.0} max {:.0} (n {})", c.median, c.p95, c.min, c.max, c.n);
        let _ = writeln!(text, "                         candidate median {:.0} p95 {:.0} min {:.0} max {:.0} (n {})", x.median, x.p95, x.min, x.max, x.n);
        let _ = writeln!(text, "delta median {d:+.0} ms ({:+.1} percent)", 100.0 * d / c.median);
    }
    // Stages where the candidate is worse by more than 3 percent.
    let mut worse = Vec::new();
    for name in STAGE_NAMES {
        let c: Vec<f64> = controls.iter().flat_map(|r| r.stages.iter().filter_map(|s| s.get(name))).collect();
        let x: Vec<f64> = cands.iter().flat_map(|r| r.stages.iter().filter_map(|s| s.get(name))).collect();
        if let (Some(cm), Some(xm)) = (quantile(&c, 0.5), quantile(&x, 0.5))
            && cm > 1.0
            && xm > cm * 1.03
        {
            worse.push(format!("{name} {cm:.0} to {xm:.0} ms"));
        }
    }
    let _ = writeln!(text, "stages worse by more than 3 percent: {}", if worse.is_empty() { "none".to_owned() } else { worse.join(", ") });
    let same_quality = cands.iter().chain(controls.iter()).filter_map(|r| r.score.as_ref()).map(|s| (s.first, s.top3, s.pack, s.every, s.cited, s.decoys.clone())).collect::<std::collections::BTreeSet<_>>();
    let _ = writeln!(text, "score lines across all runs: {} distinct", same_quality.len());
    let _ = writeln!(text, "control score:   {}", score_line(controls.first().and_then(|r| r.score.as_ref())));
    let _ = writeln!(text, "candidate score: {}", score_line(cands.first().and_then(|r| r.score.as_ref())));
    let ctl: Vec<&RunOut> = controls.iter().collect();
    let cnd: Vec<&RunOut> = cands.iter().collect();
    stage_table(&mut text, "control", &ctl);
    stage_table(&mut text, "candidate", &cnd);
    let accepted = wins == cands.len() && worse.is_empty();
    let _ = writeln!(text, "\nacceptance (beat both neighbours on every sample, no stage worse by more than 3 percent): {}", if accepted { "accepted" } else { "not accepted" });
    let mut all = controls.clone();
    all.extend(cands.iter().cloned());
    let json = format!(
        "{{\"kind\":\"pair\",\"stamp\":{stamp},\"set\":\"{}\",\"wins\":{wins},\"samples\":{},\"accepted\":{accepted},\"runs\":{}}}",
        json_escape(&set.label),
        cands.len(),
        json_runs(&all)
    );
    print!("{text}");
    write_report(home, stamp, &text, &json)
}


// ---------------------------------------------------------------------------------------------
// WP-C2: the measured routing record from the duel ability audit.

/// One question with its resolved answer lines.
struct AnsQ {
    fold: Option<u64>,
    /// (file, first line, last line) of each answer range.
    ranges: Vec<(String, usize, usize)>,
}

fn jstr(v: &nmjson::Value, k: &str) -> Option<String> {
    v.get(k).ok()?.string().ok().map(str::to_owned)
}

fn jnum(v: &nmjson::Value, k: &str) -> Option<f64> {
    match v.get(k).ok()? {
        nmjson::Value::Number(n) => n.parse().ok(),
        _ => None,
    }
}

/// The questions of one set, by question text, with answer ranges resolved (an answer's `lines`
/// when given, else the line of the root file holding its anchor) and its fold from `folds`.
fn load_answers(home: &Path, root: &str, qfile: &str, folds: Option<&nmjson::Value>) -> R<Vec<(String, AnsQ)>> {
    let text = fs::read(home.join(qfile)).map_err(|e| format!("{qfile}: {e}"))?;
    let v = nmjson::parse(&text).map_err(|e| format!("{qfile}: {e:?}"))?;
    let qs = v.get("questions").and_then(nmjson::Value::array).map_err(|e| format!("{qfile}: {e:?}"))?;
    let mut out = Vec::new();
    for q in qs {
        let (Some(id), Some(question)) = (jstr(q, "id"), jstr(q, "question")) else { continue };
        let fold = folds.and_then(|f| f.get("folds").ok()).and_then(|f| f.get(&id).ok()).and_then(|x| x.integer().ok());
        let mut ranges = Vec::new();
        for a in q.get("answers").and_then(nmjson::Value::array).unwrap_or(&[]) {
            let Some(file) = jstr(a, "file") else { continue };
            let lines: Vec<usize> =
                a.get("lines").and_then(nmjson::Value::array).map(|l| l.iter().filter_map(|x| x.integer().ok()).filter_map(|x| usize::try_from(x).ok()).collect()).unwrap_or_default();
            if let [s0, e0] = lines.as_slice() {
                ranges.push((file, *s0, *e0));
                continue;
            }
            let Some(anchor) = jstr(a, "anchor") else { continue };
            let body = fs::read_to_string(home.join(root).join(&file)).unwrap_or_default();
            if let Some(i) = body.lines().position(|l| l.contains(anchor.as_str())) {
                ranges.push((file, i + 1, i + 1));
            }
        }
        if !ranges.is_empty() {
            out.push((question, AnsQ { fold, ranges }));
        }
    }
    Ok(out)
}

/// Whether a piece reference `file:start-end` holds any answer range.
fn holds(piece: &str, ranges: &[(String, usize, usize)]) -> bool {
    let Some((file, rng)) = piece.rsplit_once(':') else { return false };
    let Some((a, b)) = rng.split_once('-') else { return false };
    let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) else { return false };
    let file = file.replace('\\', "/");
    ranges.iter().any(|(f, s0, e0)| file.ends_with(f.as_str()) && a <= *e0 && *s0 <= b)
}

/// The slope `k` of P(entrant better) = sigmoid(k x), fitted by Newton's method on (x, truth).
fn fit_slope(points: &[(f64, bool)]) -> f64 {
    let mut k = 0.0_f64;
    for _ in 0..50 {
        let (mut g, mut h) = (0.0_f64, 0.0_f64);
        for (x, y) in points {
            let p = 1.0 / (1.0 + (-k * x).exp());
            g += (f64::from(u8::from(*y)) - p) * x;
            h += p * (1.0 - p) * x * x;
        }
        if h <= 1e-12 {
            break;
        }
        let step = g / h;
        k += step;
        if step.abs() < 1e-9 {
            break;
        }
    }
    k
}

/// Wilson 95 percent interval of a share.
fn wilson(correct: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let (p, nn) = (correct as f64 / n as f64, n as f64);
    let z = 1.96_f64;
    let d = 1.0 + z * z / nn;
    let c = (p + z * z / (2.0 * nn)) / d;
    let h = z * (p * (1.0 - p) / nn + z * z / (4.0 * nn * nn)).sqrt() / d;
    (c - h, c + h)
}

fn num_or_null(x: f64) -> String {
    if x.is_finite() { format!("{x:.4}") } else { "null".to_owned() }
}

/// The key of one audit point set: (piece kind, MicroModel, from the fit folds).
type PointKey = (String, &'static str, bool);

/// One MicroModel's scores on one kind of close call.
struct Scored {
    model: &'static str,
    fit_n: usize,
    fit_right: usize,
    check_n: usize,
    check_right: usize,
    slope: f64,
}

impl Scored {
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    fn acc(right: usize, n: usize) -> f64 {
        if n == 0 { f64::NAN } else { right as f64 / n as f64 }
    }
    fn row(&self, kind: &str) -> String {
        let (lo, hi) = wilson(self.fit_right, self.fit_n);
        let fa = Self::acc(self.fit_right, self.fit_n);
        format!(
            "{{\"job\":\"{}\",\"kind\":\"{kind}\",\"model\":\"{}\",\"stack\":\"{}\",\"precision\":\"served\",\"scope\":\"tuning sets fit folds 0-2\",\"n\":{},\"correct\":{},\"accuracy\":{},\"interval\":[{lo:.4},{hi:.4}],\"check_n\":{},\"check_correct\":{},\"check_accuracy\":{},\"slope\":{:.6},\"strength\":{}}}",
            if kind == "code" { "code-close" } else { "prose-close" },
            self.model,
            if self.model == "embedder" { "embedder-base+project lens" } else { "reranker-base+judge-code adapter" },
            self.fit_n,
            self.fit_right,
            num_or_null(fa),
            self.check_n,
            self.check_right,
            num_or_null(Self::acc(self.check_right, self.check_n)),
            self.slope,
            num_or_null(fa * 100.0)
        )
    }
}

/// The fewest informative fit duels a judge skip threshold is derived from.
const RECORD_MIN_FIT: usize = 30;

/// `nano-lab stage-bench routing-record --audit <jsonl> --set ... [--out designs/routing-record.json]`
fn cmd_routing_record(home: &Path, args: &[String]) -> R<PathBuf> {
    let mut audit: Option<String> = None;
    let mut out_path = home.join("designs").join("routing-record.json");
    let mut sets: Vec<SetSpec> = Vec::new();
    let mut i = 0;
    while let Some(a) = args.get(i) {
        let val = |i: usize| args.get(i + 1).cloned().ok_or_else(|| format!("{a} takes a value"));
        match a.as_str() {
            "--audit" => audit = Some(val(i)?),
            "--out" => out_path = home.join(val(i)?),
            "--set" => sets.push(parse_set(&val(i)?)?),
            other => return Err(format!("routing-record: unknown argument {other}")),
        }
        i += 2;
    }
    let audit = audit.ok_or("routing-record needs --audit <file>")?;
    let read_folds = |p: &str| fs::read(home.join(p)).ok().and_then(|b| nmjson::parse(&b).ok());
    let (folds_dev, folds_large) = (read_folds("tests/dev/folds.json"), read_folds("tests/dev-large/folds.json"));
    let mut answers: BTreeMap<String, AnsQ> = BTreeMap::new();
    for s in &sets {
        if s.questions.contains("dev-large-2") {
            return Err("routing-record: dev-large-2 is the unseen verification set and is never fitted on".to_owned());
        }
        let folds = if s.questions.contains("dev-large") { folds_large.as_ref() } else { folds_dev.as_ref() };
        for (q, a) in load_answers(home, &s.root, &s.questions, folds)? {
            answers.insert(q, a);
        }
    }
    // (kind, model, fit) -> (difference, entrant holds the answer).
    let mut pts: BTreeMap<PointKey, Vec<(f64, bool)>> = BTreeMap::new();
    // Duels read by both MicroModels: (kind, fit, embedder, judge, entrant holds the answer).
    let mut paired: Vec<(String, bool, f64, f64, bool)> = Vec::new();
    let (mut lines_n, mut informative) = (0_usize, 0_usize);
    // The lens in force when the duels were read (each line names it; `none` without one).
    let mut lens_seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let text = fs::read_to_string(home.join(&audit)).map_err(|e| format!("{audit}: {e}"))?;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        lines_n += 1;
        let Ok(v) = nmjson::parse(line.as_bytes()) else { continue };
        lens_seen.insert(jstr(&v, "lens").unwrap_or_else(|| "none".to_owned()));
        let (Some(q), Some(a), Some(b), Some(kind)) = (jstr(&v, "question"), jstr(&v, "a"), jstr(&v, "b"), jstr(&v, "kind")) else { continue };
        let Some(ans) = answers.get(&q) else { continue };
        let Some(fold) = ans.fold else { continue };
        let (ha, hb) = (holds(&a, &ans.ranges), holds(&b, &ans.ranges));
        if ha == hb {
            continue;
        }
        informative += 1;
        if let (Some(e), Some(j)) = (jnum(&v, "embedder").filter(|x| x.is_finite()), jnum(&v, "judge").filter(|x| x.is_finite())) {
            paired.push((kind.clone(), fold <= 2, e, j, ha));
        }
        for model in ["embedder", "judge"] {
            if let Some(x) = jnum(&v, model).filter(|x| x.is_finite()) {
                pts.entry((kind.clone(), model, fold <= 2)).or_default().push((x, ha));
            }
        }
    }
    let mut report = String::new();
    let _ = writeln!(report, "routing record from {audit}: {lines_n} duel lines, {informative} informative (exactly one side holds an answer)");
    let mut rows = Vec::new();
    for kind in ["code", "prose"] {
        let mut scored = Vec::new();
        for model in ["embedder", "judge"] {
            let f = pts.get(&(kind.to_owned(), model, true)).cloned().unwrap_or_default();
            let c = pts.get(&(kind.to_owned(), model, false)).cloned().unwrap_or_default();
            // Scored by the calibrated verdict: the sign of slope times difference (a MicroModel whose
            // raw reading runs against the truth, the judge's pair prompt on code, is read the other
            // way round, as the served duel reads it through the record's slope). The raw sign's
            // share is reported beside it.
            let slope = fit_slope(&f);
            let right = |v: &[(f64, bool)]| v.iter().filter(|(x, y)| *x != 0.0 && slope != 0.0 && (slope * x > 0.0) == *y).count();
            let raw = |v: &[(f64, bool)]| v.iter().filter(|(x, y)| *x != 0.0 && (*x > 0.0) == *y).count();
            let _ = writeln!(report, "  {kind:<5} {model:<8} raw sign: fit {}/{}, check {}/{}", raw(&f), f.len(), raw(&c), c.len());
            let s = Scored { model, fit_n: f.len(), fit_right: right(&f), check_n: c.len(), check_right: right(&c), slope };
            let (lo, hi) = wilson(s.fit_right, s.fit_n);
            let _ = writeln!(
                report,
                "  {kind:<5} {model:<8} fit {}/{} = {} [{lo:.3}, {hi:.3}]  check {}/{} = {}  slope {:.3}",
                s.fit_right,
                s.fit_n,
                num_or_null(Scored::acc(s.fit_right, s.fit_n)),
                s.check_right,
                s.check_n,
                num_or_null(Scored::acc(s.check_right, s.check_n)),
                s.slope
            );
            rows.push(s.row(kind));
            scored.push(s);
        }
        // Dual-strong: both measured and their fit intervals overlap: their evidence is combined.
        let overlap = match scored.as_slice() {
            [a, b] if a.fit_n > 0 && b.fit_n > 0 => {
                let ((alo, ahi), (blo, bhi)) = (wilson(a.fit_right, a.fit_n), wilson(b.fit_right, b.fit_n));
                alo <= bhi && blo <= ahi
            }
            _ => false,
        };
        let _ = writeln!(report, "  {kind:<5} dual-strong (fit intervals overlap): {overlap}");
        // The combined verdict (slopes fitted on the fit folds) scored on both folds.
        let mut decisive: Option<f64> = None;
        if let [a, b] = scored.as_slice() {
            for (label, fit) in [("fit", true), ("check", false)] {
                let both: Vec<&(String, bool, f64, f64, bool)> = paired.iter().filter(|p| p.0 == kind && p.1 == fit).collect();
                let right = both.iter().filter(|p| (a.slope * p.2 + b.slope * p.3 > 0.0) == p.4).count();
                let _ = writeln!(report, "  {kind:<5} combined {label}: {right}/{}", both.len());
            }
            // Roadmap Phase 0 item 4 (2026-11-23): the judge's read is skipped where the embedder's
            // calibrated evidence alone decides as well as the combined verdict. On the fit folds:
            // the smallest |embedder log-odds| from which, in every informative duel at or above it,
            // the combined verdict is right no more often than the embedder's alone (the judge would
            // add GPU time and nothing else). Reported on the check folds; the serving default ln 9
            // stands where too few duels decide it.
            let fit_d: Vec<&(String, bool, f64, f64, bool)> = paired.iter().filter(|p| p.0 == kind && p.1).collect();
            let gain = |t: f64, set: &[&(String, bool, f64, f64, bool)]| -> (i64, usize) {
                let mut g = 0_i64;
                let mut n = 0;
                for p in set.iter().filter(|p| (a.slope * p.2).abs() >= t) {
                    let comb = (a.slope * p.2 + b.slope * p.3 > 0.0) == p.4;
                    let emb = (a.slope * p.2 > 0.0) == p.4;
                    g += i64::from(comb) - i64::from(emb);
                    n += 1;
                }
                (g, n)
            };
            // The threshold sits just above the most confident duel the judge ever corrected (the
            // combined verdict right where the embedder's alone was wrong) on the fit folds: no
            // duel the judge fixed is skipped. A net-gain rule (fixes less breaks) chose 0.538 on a
            // partial wide audit and skipped duels the judge fixed on the check folds (3 of 40).
            if fit_d.len() >= RECORD_MIN_FIT {
                let fixed = fit_d.iter().filter(|p| ((a.slope * p.2 + b.slope * p.3 > 0.0) == p.4) && ((a.slope * p.2 > 0.0) != p.4)).map(|p| (a.slope * p.2).abs()).fold(0.0_f64, f64::max);
                decisive = Some(fixed * 1.000_001 + 1e-6);
            }
            if let Some(t) = decisive {
                let chk: Vec<&(String, bool, f64, f64, bool)> = paired.iter().filter(|p| p.0 == kind && !p.1).collect();
                let (gf, nf) = gain(t, &fit_d);
                let (gc, nc) = gain(t, &chk);
                let all_e: Vec<f64> = paired.iter().filter(|p| p.0 == kind).map(|p| (a.slope * p.2).abs()).collect();
                #[expect(clippy::cast_precision_loss, reason = "duel counts")]
                let share = all_e.iter().filter(|x| **x >= t).count() as f64 / all_e.len().max(1) as f64;
                let _ = writeln!(report, "  {kind:<5} judge skipped from |embedder log-odds| {t:.3}: fit {nf} duels, judge gain {gf}; check {nc} duels, judge gain {gc}; {:.0} percent of informative duels skip the judge", share * 100.0);
                // The threshold must hold on the check folds: if a duel above it was one the judge
                // corrected there, it is not written and serving keeps ln 9 (prose, 37 fit duels:
                // 0.710 skipped 3 check duels the judge fixed).
                let check_fixed = chk.iter().any(|p| (a.slope * p.2).abs() >= t && ((a.slope * p.2 + b.slope * p.3 > 0.0) == p.4) && ((a.slope * p.2 > 0.0) != p.4));
                if check_fixed {
                    let _ = writeln!(report, "  {kind:<5} threshold {t:.3} not written: the judge corrected a check duel above it; serving keeps ln 9");
                    decisive = None;
                }
            } else {
                let _ = writeln!(report, "  {kind:<5} judge skip threshold: too few informative fit duels ({}, need {RECORD_MIN_FIT}); serving keeps ln 9", fit_d.len());
            }
        }
        let dec = decisive.map_or_else(String::new, |t| format!(",\"decisive\":{t:.6}"));
        rows.push(format!("{{\"job\":\"{}\",\"kind\":\"{kind}\",\"combine\":{overlap}{dec}}}", if kind == "code" { "code-close" } else { "prose-close" }));
    }
    // Every lens the duels were read with (projects differ; the lane keeps the plain scale with a
    // lens, so the embedder's differences share one scale and one record holds for all of them).
    let lens = if lens_seen.is_empty() { "none".to_owned() } else { lens_seen.into_iter().collect::<Vec<_>>().join(",") };
    let served = fs::read(home.join("designs/base-cartridges.json")).ok().and_then(|b| nmjson::parse(&b).ok());
    let sha = |m: &str| served.as_ref().and_then(|s| s.get("serving").ok()).and_then(|s| s.get(m).ok()).and_then(|x| jstr(x, "sha256")).unwrap_or_default();
    let body = format!(
        "{{\n \"note\": \"Measured routing record (Stage 5 WP-C2): each MicroModel's accuracy on the close calls G2 routes, read by both MicroModels on the same duels (NANO_DUEL_AUDIT), scored against the tuning sets' answers; fit on folds 0 to 2, checked on folds 3 and 4; never fitted on dev-large-2. Valid only while the serving base cartridges below are unchanged; otherwise the strengths table falls back to STRENGTHS. Written by nano-lab stage-bench routing-record.\",\n \"version\": 1,\n \"made_utc\": \"{}\",\n \"audit\": \"{}\",\n \"base_cartridges\": {{\"qwen3-embedding-0.6b\": \"{}\", \"qwen3-reranker-0.6b\": \"{}\"}},\n \"lens\": \"{}\",\n \"rows\": [\n  {}\n ]\n}}\n",
        nmmusc::pack::utc_stamp(now_ms()),
        json_escape(&audit),
        sha("qwen3-embedding-0.6b"),
        sha("qwen3-reranker-0.6b"),
        json_escape(&lens),
        rows.join(",\n  ")
    );
    fs::write(&out_path, body).map_err(|e| format!("{}: {e}", out_path.display()))?;
    print!("{report}");
    write_report(home, now_ms(), &report, "{}")?;
    Ok(out_path)
}

/// `nano-lab stage-bench ...`
pub fn run(args: &[String], home: &Path) -> std::process::ExitCode {
    let sub = args.get(1).map(String::as_str);
    let rest = args.get(2..).map(<[String]>::to_vec).unwrap_or_default();
    let result = match sub {
        Some("base") => cmd_base(home, &rest).map(|p| format!("written: {}", p.display())),
        Some("pair") => cmd_pair(home, &rest).map(|p| format!("written: {}", p.display())),
        Some("routing-record") => cmd_routing_record(home, &rest).map(|p| format!("written: {}", p.display())),
        Some("lock-status") => {
            let p = CardLock::default_path(home);
            Ok(match holder_pid(&p) {
                Some(pid) => format!("card lock held by process {pid} ({})", if process_alive(pid) { "running" } else { "ended: stale" }),
                None if p.exists() => "card lock file present without a readable pid (stale)".to_owned(),
                None => "card lock free".to_owned(),
            })
        }
        _ => Err(USAGE.to_owned()),
    };
    match result {
        Ok(msg) => {
            println!("{msg}");
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("stage-bench: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_lock_path(name: &str) -> PathBuf {
        let dir = nmtype::nano_root().join("scratch").join("stage-bench-tmp").join(format!("locktest-{}-{name}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        dir.join("card.lock")
    }

    #[test]
    fn card_lock_is_exclusive_and_released_on_drop() {
        let p = test_lock_path("excl");
        let _ = fs::remove_file(&p);
        let first = CardLock::acquire_at(&p, "first", Duration::ZERO).expect("first take");
        assert_eq!(holder_pid(&p), Some(std::process::id()));
        // A second take by the same process is refused, never shared.
        assert!(CardLock::acquire_at(&p, "second", Duration::ZERO).is_err());
        drop(first);
        assert!(!p.exists(), "the lock file is removed when the guard drops");
        let again = CardLock::acquire_at(&p, "again", Duration::ZERO).expect("take after release");
        drop(again);
        assert!(!p.exists());
    }

    #[test]
    fn stale_lock_is_cleared() {
        let p = test_lock_path("stale");
        // With another card program running, an ended holder is waited for, not cleared (by
        // design); the clearing path is checked only when the card is idle.
        if !other_card_programs(std::process::id()).is_empty() {
            return;
        }
        // A pid far above any running process: the lock is stale.
        fs::write(&p, "{\"pid\": 4294967290, \"started_utc\": \"2026-01-01T00:00:00Z\", \"cmd\": \"gone\"}\n").expect("write stale lock");
        // A fresh lock of an ended process is still honoured (another agent's bench sequence).
        assert!(CardLock::acquire_at(&p, "early", Duration::ZERO).is_err(), "a fresh lock is honoured");
        // Older than STALE_AFTER: cleared.
        let old = SystemTime::now() - STALE_AFTER - Duration::from_secs(60);
        File::options().write(true).open(&p).expect("open lock").set_modified(old).expect("age the lock");
        let g = CardLock::acquire_at(&p, "fresh", Duration::ZERO).expect("stale lock cleared");
        assert_eq!(holder_pid(&p), Some(std::process::id()));
        drop(g);
        assert!(!p.exists());
    }

    #[test]
    fn a_lock_held_by_a_live_process_is_waited_for_then_refused() {
        let p = test_lock_path("live");
        // The parent of this test (the cargo test runner's shell) is alive; use this process's
        // own pid written by another guard's file but pretend it is foreign by writing pid 4
        // (the Windows System process) or 1 (init) which always run.
        let live = if cfg!(windows) { 4 } else { 1 };
        fs::write(&p, format!("{{\"pid\": {live}, \"started_utc\": \"x\", \"cmd\": \"other\"}}\n")).expect("write live lock");
        let t = Instant::now();
        let r = CardLock::acquire_at(&p, "waiter", Duration::from_millis(300));
        assert!(r.is_err(), "a live holder is not displaced");
        assert!(t.elapsed() >= Duration::from_millis(250), "it waited before refusing");
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn stage_lines_are_parsed() {
        let err = "STAGE_TIMES signals and meaning 0.441 s, evidence 0.012 s\n\
                   STAGE_TIMES drone kind (async bg) 0.301 s\n\
                   STAGE_TIMES sense 0.460 compose 0.010 g1 1.240 pack 0.002 total 2.900 rounds 3; deep 0.420 (fallbacks 0) read 0.700; g2 own 410 ms, order reads 424 ms, duels judge 12 ms, close 300 ms; g2 parts: sufficiency 1 ms, own picks 0 ms, intake 2 ms\n";
        let s = parse_stages(err);
        assert_eq!(s.len(), 1);
        let q = s.first().expect("one question");
        assert_eq!(q.drone_kind.map(f64::round), Some(301.0));
        assert_eq!(q.g1.map(f64::round), Some(1240.0));
        assert_eq!(q.tentacle_read.map(f64::round), Some(700.0));
        assert_eq!(q.deep.map(f64::round), Some(420.0));
        assert_eq!(q.total.map(f64::round), Some(2900.0));
        assert_eq!(q.judge_close.map(f64::round), Some(300.0));
        assert_eq!(q.g2_order_reads.map(f64::round), Some(424.0));
    }

    #[test]
    fn score_and_question_lines_are_parsed() {
        let text = "path    first  top3  pack every cited   tok med tok total   s med s worst  decoys |    up    f4 every-up aspects\n\
                    flow       60    77    84    74    80      4074    447329   2.891   5.283   20/20 |     -     -        -       -\n\
                    val101 code     flow 11111 rr 1.0000 tok 4100 s 3.100\n\
                    dec101 decoy     flow 00000l rr 0.0000 tok 4149 s 2.771\n";
        let s = parse_score(text).expect("score line");
        assert_eq!((s.first, s.top3, s.pack, s.every, s.cited), (60, 77, 84, 74, 80));
        assert!((s.s_med - 2.891).abs() < 1e-9);
        assert_eq!(s.decoys, "20/20");
        assert_eq!(parse_question_seconds(text), vec![3.1, 2.771]);
    }

    #[test]
    fn quantiles() {
        let v = [5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(quantile(&v, 0.5), Some(3.0));
        assert_eq!(quantile(&v, 0.0), Some(1.0));
        assert_eq!(quantile(&v, 1.0), Some(5.0));
        assert_eq!(quantile(&[], 0.5), None);
    }
}
