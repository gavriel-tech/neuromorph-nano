//! `neuromorph-nano soak`: the long-running daemon soak (completion roadmap Phase 5 item 1). It
//! copies a project into `scratch/soak-<ms>/repo`, starts a warm daemon on that copy as a child
//! process (its own port file and its own test memory under `tests/mcp-launcher/soak-<ms>`, the
//! idle writer on, the tree polled), and drives a mixed workload for the given duration:
//! questions (`nm_context_pack`), idle time for the writer, file edits in the copy, and a light
//! call (`nm_status`) arriving right after idle time, timed as the call preemption. It samples the
//! daemon's resident memory and the graphics card's memory over time (`nvidia-smi` when present,
//! else skipped), checks that no file was written outside D: and the project and memory folders,
//! stops the daemon and times a clean shutdown, then writes `report.json` and `report.md` into
//! `runs/soak-<ms>/`. The copy and the test memory are removed at the end unless `--keep`.
use crate::harness::{self, Daemon, Rng, int, num, obj, s};
use nmjson::Value;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant, SystemTime};

pub const USAGE: &str = "usage: neuromorph-nano soak [--duration 4h] [--project <repo>] [--questions <file>] [--port 7531]
                          [--idle 60s] [--sample 30s] [--seed <n>] [--preempt-ms 300] [--allow <folder>]...
                          [--keep] [-- <serve options>]
       neuromorph-nano soak --self-test     check the harness's parsers and verdicts only (starts nothing)
  --project    the project copied into scratch/ and served (default tests/fixtures/small-repo)
  --questions  questions file (tests/dev/questions.json form, or a list of strings); default: made
               from the project's file names
  --idle       quiet time given to the idle writer before each preemption probe
  --allow      a further folder of the Nano folder that may be written during the run
  --keep       keep the scratch copy and the test memory after the run";

/// The soak's settings.
#[derive(Debug)]
pub struct Config {
    pub duration: Duration,
    pub project: PathBuf,
    pub questions: Option<PathBuf>,
    pub port: u16,
    pub idle: Duration,
    pub sample: Duration,
    pub seed: u64,
    pub preempt_ms: f64,
    pub allow: Vec<String>,
    pub keep: bool,
    pub serve: Vec<String>,
}

/// Folders of the Nano folder a run may write: memory, test memories, reports, scratch, muscle
/// memory and the cartridge library (the idle adapter fit writes there).
pub const ALLOWED_IN_HOME: &[&str] = &["memory", "tests", "runs", "scratch", "muscle", "library"];

/// Reads the soak's arguments (without the word `soak`).
/// # Errors
/// A malformed value, as one line.
pub fn parse_config(home: &Path, args: &[String]) -> Result<Config, String> {
    let own: Vec<String> = args.iter().take_while(|a| a.as_str() != "--").cloned().collect();
    let dur = |f: &str, d: &str| harness::parse_duration(harness::opt(&own, f).unwrap_or(d));
    let port = harness::opt(&own, "--port").map_or(Ok(7531), |p| p.parse::<u16>().map_err(|_| format!("--port takes a port number, got {p}")))?;
    let preempt_ms = harness::opt(&own, "--preempt-ms")
        .map_or(Ok(300.0), |p| p.parse::<f64>().ok().filter(|x| *x > 0.0).ok_or_else(|| format!("--preempt-ms takes milliseconds, got {p}")))?;
    let seed = harness::opt(&own, "--seed").map_or(Ok(nmtype::now_ms()), |p| p.parse::<u64>().map_err(|_| format!("--seed takes a number, got {p}")))?;
    let project = harness::opt(&own, "--project").map_or_else(|| home.join("tests").join("fixtures").join("small-repo"), PathBuf::from);
    let allow: Vec<String> = own.windows(2).filter(|w| w.first().map(String::as_str) == Some("--allow")).filter_map(|w| w.get(1).cloned()).collect();
    Ok(Config {
        duration: dur("--duration", "4h")?,
        project,
        questions: harness::opt(&own, "--questions").map(PathBuf::from),
        port,
        idle: dur("--idle", "60s")?,
        sample: dur("--sample", "30s")?,
        seed,
        preempt_ms,
        allow,
        keep: harness::has(&own, "--keep"),
        serve: harness::passthrough(args),
    })
}

/// Memory growth over a run: (early median, late median, least-squares slope per hour), in the
/// samples' unit, after the first tenth (warm-up) is left out; `None` under 8 samples.
pub fn growth(samples: &[(f64, f64)]) -> Option<(f64, f64, f64)> {
    if samples.len() < 8 {
        return None;
    }
    let warm = samples.len() / 10;
    let body = samples.get(warm..)?;
    let q = (body.len() / 4).max(1);
    let med = |v: &[(f64, f64)]| harness::percentile(&v.iter().map(|x| x.1).collect::<Vec<_>>(), 50.0);
    let early = med(body.get(..q)?)?;
    let late = med(body.get(body.len() - q..)?)?;
    #[allow(clippy::cast_precision_loss)]
    let n = body.len() as f64;
    let (mx, my) = (body.iter().map(|x| x.0).sum::<f64>() / n, body.iter().map(|x| x.1).sum::<f64>() / n);
    let sxx: f64 = body.iter().map(|x| (x.0 - mx).powi(2)).sum();
    let sxy: f64 = body.iter().map(|x| (x.0 - mx) * (x.1 - my)).sum();
    let slope_per_s = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    Some((early, late, slope_per_s * 3600.0))
}

/// The leak verdict on a growth measure: a leak when the late median is above the early one by
/// more than `floor` and by more than a tenth of the early median.
pub fn leaked(g: (f64, f64, f64), floor: f64) -> bool {
    let (early, late, _) = g;
    late - early > floor.max(early * 0.10)
}

/// The comment line an edit appends to a source file of this extension.
pub fn comment_for(ext: &str, n: usize) -> String {
    let lead = if ["py", "rb"].contains(&ext) { "#" } else { "//" };
    format!("\n{lead} soak edit {n}\n")
}

/// A changed file found by the write check, and why it counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Written {
    pub path: String,
    pub why: &'static str,
}

/// Whether a path below the Nano folder lies in a folder a run may write (`ALLOWED_IN_HOME` and
/// the `--allow` folders); `rel` is relative to the Nano folder with either slash.
pub fn allowed_in_home(rel: &str, extra: &[String]) -> bool {
    let first = rel.trim_start_matches(['/', '\\']).split(['/', '\\']).next().unwrap_or_default();
    ALLOWED_IN_HOME.contains(&first) || extra.iter().any(|a| a.trim_matches(['/', '\\']).split(['/', '\\']).next() == Some(first))
}

/// Whether a changed file outside the Nano folder looks like the Nano's own (its name or folder
/// names the Nano).
pub fn names_nano(p: &str) -> bool {
    let l = p.to_ascii_lowercase();
    l.contains("neuromorph") || l.contains("nano-") || l.contains("nano_") || l.contains("\\nano\\") || l.contains("/nano/")
}

fn modified_since(p: &Path, t0: SystemTime) -> bool {
    std::fs::metadata(p).and_then(|m| m.modified()).is_ok_and(|m| m >= t0)
}

/// Files under `dir` (to `depth` folders down, at most `budget` entries visited) changed since
/// `t0`, folders named in `skip` left out.
fn changed_files(dir: &Path, t0: SystemTime, depth: usize, skip: &[&str], budget: &mut usize, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        if *budget == 0 {
            return;
        }
        *budget -= 1;
        let p = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        match e.file_type() {
            Ok(t) if t.is_dir() => {
                if depth > 0 && !skip.contains(&name.as_str()) {
                    changed_files(&p, t0, depth - 1, skip, budget, out);
                }
            }
            Ok(t) if t.is_file() && modified_since(&p, t0) => out.push(p),
            _ => {}
        }
    }
}

/// The write check after a run: files changed since `t0` in the Nano folder outside the allowed
/// folders, files naming the Nano changed in the user's profile folders off D:, and any run folder
/// that is not on D:. Returns (violations, notes).
pub fn write_check(home: &Path, run_dirs: &[&Path], t0: SystemTime, extra: &[String]) -> (Vec<Written>, Vec<String>) {
    let mut bad = Vec::new();
    let mut notes = Vec::new();
    for d in run_dirs {
        if !harness::on_d_drive(d) {
            bad.push(Written { path: d.display().to_string(), why: "a run folder is not on D:" });
        }
    }
    // The Nano folder: walk everything but the allowed folders and the build and weight folders.
    let mut budget = 400_000;
    let mut changed = Vec::new();
    for e in std::fs::read_dir(home).map(|rd| rd.flatten().collect::<Vec<_>>()).unwrap_or_default() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        match e.file_type() {
            Ok(t) if t.is_dir() => {
                if !allowed_in_home(&name, extra) && !["target", ".git", "weights", "bin"].contains(&name.as_str()) {
                    changed_files(&p, t0, 12, &["target", ".git", "node_modules"], &mut budget, &mut changed);
                }
            }
            Ok(t) if t.is_file() && modified_since(&p, t0) => changed.push(p),
            _ => {}
        }
    }
    for p in changed {
        let rel = p.strip_prefix(home).map_or_else(|_| p.display().to_string(), |r| r.display().to_string());
        if !allowed_in_home(&rel, extra) {
            bad.push(Written { path: p.display().to_string(), why: "changed in the Nano folder outside memory, tests, runs, scratch, muscle and library" });
        }
    }
    // The executable folder must not change while a daemon runs from it.
    let mut b2 = 10_000;
    let mut bins = Vec::new();
    changed_files(&home.join("bin"), t0, 2, &[], &mut b2, &mut bins);
    bad.extend(bins.into_iter().map(|p| Written { path: p.display().to_string(), why: "bin/ changed during the run" }));
    // The user's profile folders off D: (temporary files, application data).
    let watched: Vec<(String, usize)> = [("TEMP", 3), ("TMP", 3), ("LOCALAPPDATA", 3), ("APPDATA", 3), ("USERPROFILE", 1), ("ProgramData", 1)]
        .iter()
        .filter_map(|(k, depth)| std::env::var(k).ok().filter(|v| !v.is_empty()).map(|v| (v, *depth)))
        .collect();
    let mut seen: Vec<String> = Vec::new();
    for (dir, depth) in watched {
        let key = dir.to_ascii_lowercase();
        if seen.contains(&key) || harness::on_d_drive(Path::new(&dir)) {
            continue;
        }
        seen.push(key);
        let mut budget = 200_000;
        let mut found = Vec::new();
        changed_files(Path::new(&dir), t0, depth, &["node_modules", "Packages", "Microsoft"], &mut budget, &mut found);
        let ours: Vec<&PathBuf> = found.iter().filter(|p| names_nano(&p.display().to_string())).collect();
        notes.push(format!("{dir}: {} files changed during the run, {} naming the Nano", found.len(), ours.len()));
        bad.extend(ours.into_iter().map(|p| Written { path: p.display().to_string(), why: "a file naming the Nano was written off D:" }));
        if found.iter().any(|p| p.display().to_string().contains("ComputeCache")) {
            notes.push(format!("{dir}: the graphics driver's compute cache changed (kernel compilation at run time)"));
        }
    }
    (bad, notes)
}

/// Copies a project into `dst` (build, version-control and dependency folders left out, files over
/// 4 MiB skipped): (files, bytes, source files that edits may touch).
/// # Errors
/// A folder or file cannot be read or written.
pub fn copy_tree(src: &Path, dst: &Path) -> Result<(usize, u64, Vec<PathBuf>), String> {
    let mut stack = vec![(src.to_path_buf(), dst.to_path_buf())];
    let (mut files, mut bytes, mut sources) = (0usize, 0u64, Vec::new());
    while let Some((from, to)) = stack.pop() {
        std::fs::create_dir_all(&to).map_err(|e| format!("{}: {e}", to.display()))?;
        for e in std::fs::read_dir(&from).map_err(|e| format!("{}: {e}", from.display()))?.flatten() {
            let name = e.file_name();
            let n = name.to_string_lossy();
            let (p, q) = (e.path(), to.join(&name));
            match e.file_type() {
                Ok(t) if t.is_dir() && !harness::SKIP_DIRS.contains(&n.as_ref()) => stack.push((p, q)),
                Ok(t) if t.is_file() => {
                    let len = e.metadata().map(|m| m.len()).unwrap_or(0);
                    if len > 4 << 20 || files >= 50_000 {
                        continue;
                    }
                    std::fs::copy(&p, &q).map_err(|e| format!("{}: {e}", p.display()))?;
                    files += 1;
                    bytes += len;
                    if q.extension().and_then(|x| x.to_str()).is_some_and(|x| harness::SOURCE_EXT.contains(&x)) {
                        sources.push(q);
                    }
                }
                _ => {}
            }
        }
    }
    sources.sort();
    Ok((files, bytes, sources))
}

/// One edit in the copy: append a comment to a source file, add a new small file beside one, or
/// delete a file an earlier edit added. Returns what was done.
fn edit(rng: &mut Rng, sources: &[PathBuf], added: &mut Vec<PathBuf>, n: usize) -> Result<String, String> {
    let kind = rng.below(3);
    if kind == 2
        && let Some(p) = (!added.is_empty()).then(|| added.swap_remove(rng.below(added.len())))
    {
        std::fs::remove_file(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        return Ok(format!("deleted {}", p.display()));
    }
    let Some(base) = sources.get(rng.below(sources.len())) else { return Err("the project has no source files to edit".into()) };
    let ext = base.extension().and_then(|x| x.to_str()).unwrap_or("rs").to_owned();
    if kind == 1 {
        let p = base.with_file_name(format!("soak_added_{n}.{ext}"));
        let body = if ext == "py" { format!("def soak_added_{n}():\n    return {n}\n") } else { format!("// soak edit {n}\n") };
        std::fs::write(&p, body).map_err(|e| format!("{}: {e}", p.display()))?;
        added.push(p.clone());
        return Ok(format!("added {}", p.display()));
    }
    let mut f = std::fs::OpenOptions::new().append(true).open(base).map_err(|e| format!("{}: {e}", base.display()))?;
    std::io::Write::write_all(&mut f, comment_for(&ext, n).as_bytes()).map_err(|e| format!("{}: {e}", base.display()))?;
    Ok(format!("appended to {}", base.display()))
}

/// Everything a run measured.
#[derive(Default)]
struct Measures {
    questions: Vec<f64>,
    question_faults: Vec<String>,
    preempt: Vec<f64>,
    baseline: Vec<f64>,
    /// A question asked right after each idle phase (2026-11-25: what the first question pays
    /// after idle time, for the idle KV window release).
    after_idle: Vec<f64>,
    /// Times the daemon reported an idle release of card memory.
    idle_releases: usize,
    probe_faults: Vec<String>,
    writer_busy_probes: usize,
    edits: usize,
    edit_faults: Vec<String>,
    idles: usize,
    rss: Vec<(f64, f64)>,
    gpu_used: Vec<(f64, f64)>,
    gpu_proc: Vec<(f64, f64)>,
    gpu_skipped: bool,
    /// Calls whose connection reset after the full answer had arrived (answer kept, counted).
    late_resets: usize,
}

impl Measures {
    fn counted(&mut self, c: harness::Called) -> harness::Called {
        self.late_resets += usize::from(c.late_reset);
        c
    }

    fn sample(&mut self, d: &Daemon) {
        let pid = d.child.id();
        let t = d.secs();
        if let Some(k) = harness::rss_kib(pid) {
            #[allow(clippy::cast_precision_loss)]
            self.rss.push((t, k as f64 / 1024.0));
        }
        match harness::gpu_sample(pid) {
            Some(g) => {
                #[allow(clippy::cast_precision_loss)]
                self.gpu_used.push((t, g.used_mib as f64));
                if let Some(p) = g.process_mib {
                    #[allow(clippy::cast_precision_loss)]
                    self.gpu_proc.push((t, p as f64));
                }
            }
            None => self.gpu_skipped = true,
        }
    }
}

/// Waits `d`, sampling memory whenever a sample is due; stops early when the daemon ends.
fn wait_sampling(daemon: &mut Daemon, m: &mut Measures, d: Duration, every: Duration, next: &mut Instant) {
    let until = Instant::now() + d;
    while Instant::now() < until && daemon.alive() {
        if Instant::now() >= *next {
            m.sample(daemon);
            *next = Instant::now() + every;
        }
        harness::park(Duration::from_millis(500).min(until.saturating_duration_since(Instant::now())));
    }
}

/// `neuromorph-nano soak ...`.
pub fn run(home: &Path, args: &[String]) -> ExitCode {
    if harness::has(args, "--help") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    if harness::has(args, "--self-test") {
        return match self_test() {
            Ok(n) => {
                println!("soak self-test ok: {n} parser and verdict cases");
                ExitCode::SUCCESS
            }
            Err(e) => {
                println!("soak self-test FAIL: {e}");
                ExitCode::FAILURE
            }
        };
    }
    match soak(home, args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("soak: {e}");
            ExitCode::FAILURE
        }
    }
}

fn soak(home: &Path, args: &[String]) -> Result<bool, String> {
    let cfg = parse_config(home, args)?;
    if !cfg.project.is_dir() {
        return Err(format!("--project {} is not a folder", cfg.project.display()));
    }
    let stamp = nmtype::now_ms();
    let work = home.join("scratch").join(format!("soak-{stamp}"));
    let copy = work.join("repo");
    let memory_name = format!("mcp-launcher/soak-{stamp}");
    let memory = nmtype::tests_dir(&memory_name);
    let report = harness::report_dir(home, "soak")?;
    let t0 = SystemTime::now();
    println!("soak: copying {} into {}", cfg.project.display(), copy.display());
    let (files, bytes, sources) = copy_tree(&cfg.project, &copy)?;
    let questions = harness::load_questions(cfg.questions.as_deref(), &copy)?;
    println!("soak: {files} files ({bytes} bytes), {} source files, {} questions; running {:.0} s", sources.len(), questions.len(), cfg.duration.as_secs_f64());
    let mut serve = vec![
        "--root".to_owned(),
        copy.display().to_string(),
        "--memory-dir".to_owned(),
        memory_name.clone(),
        "--writer-idle".to_owned(),
        "on".to_owned(),
        "--poll".to_owned(),
        "2".to_owned(),
    ];
    serve.extend(cfg.serve.iter().cloned());
    let mut d = Daemon::spawn(&serve, cfg.port, &work.join("daemon.port"), &work.join("daemon.log"))?;
    let mut m = Measures::default();
    let mut fault: Option<String> = None;
    if !d.wait_listening(Duration::from_secs(600)) {
        fault = Some(format!("the daemon did not listen on 127.0.0.1:{} within 600 s", cfg.port));
    }
    let ready = harness::call(cfg.port, "nm_status", &Value::Object(std::collections::BTreeMap::new()), Duration::from_secs(900));
    if let Err(e) = &ready.result {
        fault.get_or_insert(format!("the first nm_status failed: {e}"));
    }
    let mut rng = Rng::new(cfg.seed);
    let (mut added, mut n) = (Vec::new(), 0usize);
    let started = Instant::now();
    let mut next_sample = Instant::now();
    let limit = Duration::from_secs(600);
    let empty = Value::Object(std::collections::BTreeMap::new());
    while fault.is_none() && started.elapsed() < cfg.duration {
        if !d.alive() {
            fault = Some("the daemon ended during the run".into());
            break;
        }
        if Instant::now() >= next_sample {
            m.sample(&d);
            next_sample = Instant::now() + cfg.sample;
        }
        n += 1;
        match rng.below(100) {
            0..45 => {
                let q = questions.get(rng.below(questions.len())).cloned().unwrap_or_default();
                let c = m.counted(harness::call(cfg.port, "nm_context_pack", &harness::question_args(&q), limit));
                match c.result {
                    Ok(_) => m.questions.push(c.ms),
                    Err(e) => m.question_faults.push(format!("{q}: {e}")),
                }
                // Now and then the same light call right after a question, when the writer has had
                // no quiet time: the baseline the preemption is read against.
                if n % 4 == 0 {
                    let b = m.counted(harness::call(cfg.port, "nm_status", &empty, limit));
                    if b.result.is_ok() {
                        m.baseline.push(b.ms);
                    }
                }
            }
            45..70 => {
                m.idles += 1;
                let before = d.count_lines("writer:");
                wait_sampling(&mut d, &mut m, cfg.idle, cfg.sample, &mut next_sample);
                let p = m.counted(harness::call(cfg.port, "nm_status", &empty, limit));
                match p.result {
                    Ok(_) => m.preempt.push(p.ms),
                    Err(e) => m.probe_faults.push(e),
                }
                if d.count_lines("writer:") > before {
                    m.writer_busy_probes += 1;
                }
                let q = questions.get(rng.below(questions.len())).cloned().unwrap_or_default();
                let c = m.counted(harness::call(cfg.port, "nm_context_pack", &harness::question_args(&q), limit));
                match c.result {
                    Ok(_) => {
                        m.after_idle.push(c.ms);
                        m.questions.push(c.ms);
                    }
                    Err(e) => m.question_faults.push(format!("{q}: {e}")),
                }
            }
            70..90 => match edit(&mut rng, &sources, &mut added, n) {
                Ok(_) => m.edits += 1,
                Err(e) => m.edit_faults.push(e),
            },
            _ => {
                for _ in 0..3 {
                    let q = questions.get(rng.below(questions.len())).cloned().unwrap_or_default();
                    let c = m.counted(harness::call(cfg.port, "nm_context_pack", &harness::question_args(&q), limit));
                    match c.result {
                        Ok(_) => m.questions.push(c.ms),
                        Err(e) => m.question_faults.push(format!("{q}: {e}")),
                    }
                }
            }
        }
    }
    m.sample(&d);
    let ran_s = started.elapsed().as_secs_f64();
    let (clean, stop_ms, code) = d.stop(Duration::from_secs(60));
    let port_file_left = d.port_file.exists();
    let (writes, notes) = write_check(home, &[home, &copy, &memory, &report], t0, &cfg.allow);
    m.idle_releases = d.count_lines("residency: idle release");
    let tail = d.tail(40);
    if !cfg.keep {
        let _ = std::fs::remove_dir_all(&work);
        let _ = std::fs::remove_dir_all(&memory);
        // A test session's decision ledger sits beside its memory folder (`ledger_path`).
        let _ = std::fs::remove_file(nmapi::whole::ledger_path(home, &memory));
    }
    let v = Verdicts::of(&m, &cfg, clean && !port_file_left, &writes, fault.as_deref());
    let json = report_json(&cfg, &m, &v, ran_s, (clean, stop_ms, code, port_file_left), &writes, &notes, &tail, fault.as_deref(), stamp);
    let md = report_md(&cfg, &m, &v, ran_s, stop_ms, &writes, &notes, fault.as_deref());
    harness::write_report(&report, &json, &md)?;
    println!("{md}");
    println!("soak: report in {}", report.display());
    Ok(v.all())
}

/// The run's verdicts: `None` where there was too little to judge.
struct Verdicts {
    preempt: Option<bool>,
    ram: Option<bool>,
    gpu: Option<bool>,
    calls: bool,
    shutdown: bool,
    writes: bool,
    fault: bool,
}

impl Verdicts {
    fn of(m: &Measures, cfg: &Config, shutdown: bool, writes: &[Written], fault: Option<&str>) -> Self {
        let p95 = harness::percentile(&m.preempt, 95.0);
        let gpu_series = if m.gpu_proc.len() >= 8 { &m.gpu_proc } else { &m.gpu_used };
        Self {
            preempt: p95.map(|p| p <= cfg.preempt_ms),
            ram: growth(&m.rss).map(|g| !leaked(g, 64.0)),
            gpu: growth(gpu_series).map(|g| !leaked(g, 32.0)),
            calls: m.question_faults.is_empty() && m.probe_faults.is_empty(),
            shutdown,
            writes: writes.is_empty(),
            fault: fault.is_none(),
        }
    }
    fn all(&self) -> bool {
        self.preempt != Some(false) && self.ram != Some(false) && self.gpu != Some(false) && self.calls && self.shutdown && self.writes && self.fault
    }
}

fn verdict(v: Option<bool>) -> Value {
    s(match v {
        Some(true) => "pass",
        Some(false) => "FAIL",
        None => "not judged",
    })
}

fn stats(v: &[f64]) -> Value {
    obj(vec![
        ("count", int(v.len() as u64)),
        ("p50_ms", harness::percentile(v, 50.0).map_or(Value::Null, num)),
        ("p95_ms", harness::percentile(v, 95.0).map_or(Value::Null, num)),
        ("max_ms", harness::percentile(v, 100.0).map_or(Value::Null, num)),
    ])
}

fn growth_value(g: Option<(f64, f64, f64)>) -> Value {
    g.map_or(Value::Null, |(e, l, sl)| obj(vec![("early_median_mib", num(e)), ("late_median_mib", num(l)), ("slope_mib_per_hour", num(sl))]))
}

fn series(v: &[(f64, f64)]) -> Value {
    Value::Array(v.iter().map(|(t, x)| Value::Array(vec![num(*t), num(*x)])).collect())
}

#[allow(clippy::too_many_arguments)]
fn report_json(
    cfg: &Config,
    m: &Measures,
    v: &Verdicts,
    ran_s: f64,
    stop: (bool, u128, Option<i32>, bool),
    writes: &[Written],
    notes: &[String],
    tail: &[String],
    fault: Option<&str>,
    stamp: u64,
) -> Value {
    let strs = |x: &[String]| Value::Array(x.iter().take(50).map(|t| s(t)).collect());
    let gpu_series = if m.gpu_proc.len() >= 8 { &m.gpu_proc } else { &m.gpu_used };
    obj(vec![
        ("kind", s("soak")),
        ("started_ms", int(stamp)),
        ("passed", Value::Bool(v.all())),
        ("fault", fault.map_or(Value::Null, s)),
        (
            "config",
            obj(vec![
                ("duration_s", num(cfg.duration.as_secs_f64())),
                ("ran_s", num(ran_s)),
                ("project", s(&cfg.project.display().to_string())),
                ("port", int(u64::from(cfg.port))),
                ("idle_s", num(cfg.idle.as_secs_f64())),
                ("sample_s", num(cfg.sample.as_secs_f64())),
                ("seed", int(cfg.seed)),
                ("preempt_target_ms", num(cfg.preempt_ms)),
                ("serve_options", strs(&cfg.serve)),
            ]),
        ),
        (
            "verdicts",
            obj(vec![
                ("preemption", verdict(v.preempt)),
                ("ram_leak", verdict(v.ram)),
                ("gpu_leak", verdict(v.gpu)),
                ("calls", verdict(Some(v.calls))),
                ("shutdown", verdict(Some(v.shutdown))),
                ("writes", verdict(Some(v.writes))),
            ]),
        ),
        ("questions", stats(&m.questions)),
        ("question_faults", strs(&m.question_faults)),
        ("late_resets", int(m.late_resets as u64)),
        ("preemption_probes", stats(&m.preempt)),
        ("preemption_baseline", stats(&m.baseline)),
        ("questions_after_idle", stats(&m.after_idle)),
        ("idle_releases", int(m.idle_releases as u64)),
        ("probe_faults", strs(&m.probe_faults)),
        ("idle_phases", int(m.idles as u64)),
        ("probes_with_writer_activity", int(m.writer_busy_probes as u64)),
        ("edits", int(m.edits as u64)),
        ("edit_faults", strs(&m.edit_faults)),
        ("ram", obj(vec![("growth", growth_value(growth(&m.rss))), ("samples", series(&m.rss))])),
        (
            "gpu",
            if m.gpu_used.is_empty() {
                s("skipped: nvidia-smi not found or failed")
            } else {
                obj(vec![
                    ("judged_on", s(if m.gpu_proc.len() >= 8 { "process" } else { "card total" })),
                    ("growth", growth_value(growth(gpu_series))),
                    ("card_used_samples", series(&m.gpu_used)),
                    ("process_samples", series(&m.gpu_proc)),
                ])
            },
        ),
        (
            "shutdown",
            obj(vec![
                ("clean", Value::Bool(stop.0)),
                ("ms", int(u64::try_from(stop.1).unwrap_or(u64::MAX))),
                ("exit_code", stop.2.map_or(Value::Null, |c| Value::Number(c.to_string()))),
                ("port_file_left", Value::Bool(stop.3)),
            ]),
        ),
        ("writes_outside", Value::Array(writes.iter().take(200).map(|w| obj(vec![("path", s(&w.path)), ("why", s(w.why))])).collect())),
        ("write_check_notes", strs(notes)),
        ("daemon_log_tail", strs(tail)),
    ])
}

#[allow(clippy::too_many_arguments)]
fn report_md(cfg: &Config, m: &Measures, v: &Verdicts, ran_s: f64, stop_ms: u128, writes: &[Written], notes: &[String], fault: Option<&str>) -> String {
    let f = |x: Option<f64>| x.map_or_else(|| "n/a".to_owned(), |y| format!("{y:.0}"));
    let word = |x: Option<bool>| match x {
        Some(true) => "pass",
        Some(false) => "FAIL",
        None => "not judged",
    };
    let g = |x: Option<(f64, f64, f64)>| x.map_or_else(|| "too few samples".to_owned(), |(e, l, sl)| format!("{e:.0} MiB early, {l:.0} MiB late, {sl:+.1} MiB per hour"));
    let gpu_series = if m.gpu_proc.len() >= 8 { &m.gpu_proc } else { &m.gpu_used };
    let mut t = format!(
        "# Soak report\n\nResult: {}\n\nRan {ran_s:.0} s of {:.0} s on a copy of {} (seed {}).\n",
        if v.all() { "PASS" } else { "FAIL" },
        cfg.duration.as_secs_f64(),
        cfg.project.display(),
        cfg.seed
    );
    if let Some(e) = fault {
        t.push_str(&format!("\nFault: {e}\n"));
    }
    t.push_str(&format!(
        "\n| Measure | Value | Verdict |\n|---|---|---|\n| Call preemption (nm_status after idle) | p50 {} ms, p95 {} ms, max {} ms over {} probes (target {:.0} ms; baseline p50 {} ms) | {} |\n| Questions | {} answered, p50 {} ms, p95 {} ms; {} faults | {} |\n| Process memory | {} | {} |\n| Graphics-card memory | {} | {} |\n| Shutdown | {} ms | {} |\n| Writes outside D:, the project and memory | {} found | {} |\n",
        f(harness::percentile(&m.preempt, 50.0)),
        f(harness::percentile(&m.preempt, 95.0)),
        f(harness::percentile(&m.preempt, 100.0)),
        m.preempt.len(),
        cfg.preempt_ms,
        f(harness::percentile(&m.baseline, 50.0)),
        word(v.preempt),
        m.questions.len(),
        f(harness::percentile(&m.questions, 50.0)),
        f(harness::percentile(&m.questions, 95.0)),
        m.question_faults.len() + m.probe_faults.len(),
        word(Some(v.calls)),
        g(growth(&m.rss)),
        word(v.ram),
        if m.gpu_skipped && m.gpu_used.is_empty() { "skipped (no nvidia-smi)".to_owned() } else { g(growth(gpu_series)) },
        word(v.gpu),
        stop_ms,
        word(Some(v.shutdown)),
        writes.len(),
        word(Some(v.writes)),
    ));
    t.push_str(&format!("\nIdle phases {}, of which {} saw the writer report work; edits {} ({} faults).\n", m.idles, m.writer_busy_probes, m.edits, m.edit_faults.len()));
    t.push_str(&format!(
        "Questions right after idle: p50 {} ms, p95 {} ms over {}; idle releases of card memory reported: {}.\n",
        f(harness::percentile(&m.after_idle, 50.0)),
        f(harness::percentile(&m.after_idle, 95.0)),
        m.after_idle.len(),
        m.idle_releases
    ));
    if m.late_resets > 0 {
        t.push_str(&format!("Connections reset after the full answer had arrived: {} (answers kept).\n", m.late_resets));
    }
    for w in writes.iter().take(20) {
        t.push_str(&format!("- {}: {}\n", w.why, w.path));
    }
    if !notes.is_empty() {
        t.push_str("\nWrite check notes (other programs on this machine also write these folders):\n");
        for n in notes {
            t.push_str(&format!("- {n}\n"));
        }
    }
    t
}

/// The harness's parsers and verdicts on fixed samples (no process is started): the cases count.
/// # Errors
/// The first case that fails, as one line.
pub fn self_test() -> Result<usize, String> {
    let mut n = 0;
    let mut ok = |c: bool, what: &str| -> Result<(), String> {
        n += 1;
        if c { Ok(()) } else { Err(what.to_owned()) }
    };
    ok(harness::parse_tasklist("\"neuromorph-nano.exe\",\"4242\",\"Console\",\"1\",\"812,004 K\"") == Some(812_004), "tasklist row")?;
    ok(harness::parse_compute_apps("4242, 2743\n", 4242) == Some(2743), "nvidia-smi compute apps")?;
    ok(harness::parse_compute_apps("4242, [N/A]\n", 4242).is_none(), "nvidia-smi N/A")?;
    ok(harness::parse_gpu_used("2795\n") == Some(2795), "nvidia-smi memory.used")?;
    ok(harness::parse_duration("4h").is_ok_and(|d| d.as_secs() == 14_400), "duration 4h")?;
    ok(harness::secs_after("...; ready at 8.25 s", "ready at") == Some(8.25), "ready line")?;
    let flat: Vec<(f64, f64)> = (0..40).map(|i| (f64::from(i) * 30.0, 900.0 + f64::from(i % 3))).collect();
    let rising: Vec<(f64, f64)> = (0..40).map(|i| (f64::from(i) * 30.0, 900.0 + f64::from(i) * 10.0)).collect();
    ok(growth(&flat).is_some_and(|g| !leaked(g, 64.0)), "flat memory is no leak")?;
    ok(growth(&rising).is_some_and(|g| leaked(g, 64.0)), "rising memory is a leak")?;
    ok(growth(&flat[..5]).is_none(), "too few samples")?;
    ok(allowed_in_home("memory/projects/x/notes.json", &[]) && !allowed_in_home("crates/nmapi/src/lib.rs", &[]), "allowed folders")?;
    ok(allowed_in_home("designs/x.json", &["designs".to_owned()]), "an --allow folder")?;
    ok(names_nano("C:\\Users\\u\\AppData\\Local\\Temp\\neuromorph-123.tmp") && !names_nano("C:\\Users\\u\\AppData\\Local\\Temp\\other.tmp"), "names the Nano")?;
    ok(comment_for("py", 3) == "\n# soak edit 3\n" && comment_for("rs", 3) == "\n// soak edit 3\n", "edit comments")?;
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_test_passes() {
        assert!(self_test().unwrap() >= 13);
    }

    #[test]
    fn config_defaults_and_values() {
        let home = Path::new("D:\\Nano");
        let a: Vec<String> = ["--duration", "90m", "--port", "7600", "--idle", "20s", "--allow", "designs", "--keep", "--", "--neural"]
            .iter()
            .map(|x| (*x).to_owned())
            .collect();
        let c = parse_config(home, &a).unwrap();
        assert_eq!(c.duration.as_secs(), 5400);
        assert_eq!(c.port, 7600);
        assert_eq!(c.idle.as_secs(), 20);
        assert!(c.keep);
        assert_eq!(c.allow, vec!["designs".to_owned()]);
        assert_eq!(c.serve, vec!["--neural".to_owned()]);
        assert!(c.project.ends_with("small-repo"));
        let d = parse_config(home, &[]).unwrap();
        assert_eq!(d.duration.as_secs(), 14_400);
        assert!((d.preempt_ms - 300.0).abs() < f64::EPSILON);
        assert!(parse_config(home, &["--port".to_owned(), "x".to_owned()]).is_err());
        assert!(parse_config(home, &["--duration".to_owned(), "soon".to_owned()]).is_err());
    }

    #[test]
    fn copies_and_edits_a_tree() {
        let dir = nmtype::check_dir("nmcore-soak-copy").unwrap();
        let src = dir.join("src-repo");
        std::fs::create_dir_all(src.join("src")).unwrap();
        std::fs::create_dir_all(src.join("target")).unwrap();
        std::fs::write(src.join("src").join("lib.rs"), "pub fn a() {}\n").unwrap();
        std::fs::write(src.join("target").join("big.bin"), "x").unwrap();
        std::fs::write(src.join("README.md"), "# r\n").unwrap();
        let (files, _, sources) = copy_tree(&src, &dir.join("copy")).unwrap();
        assert_eq!(files, 2);
        assert_eq!(sources.len(), 1);
        assert!(!dir.join("copy").join("target").exists());
        let mut rng = Rng::new(3);
        let mut added = Vec::new();
        for i in 0..12 {
            edit(&mut rng, &sources, &mut added, i).unwrap();
        }
        assert!(std::fs::read_to_string(&sources[0]).unwrap().starts_with("pub fn a() {}"));
        let t0 = SystemTime::now() - Duration::from_secs(60);
        let mut budget = 1000;
        let mut changed = Vec::new();
        changed_files(&dir, t0, 4, &[], &mut budget, &mut changed);
        assert!(!changed.is_empty());
    }
}
