//! Shared pieces of the long-run harnesses (`soak`, `coldstart`): a warm daemon started as a child
//! of this program with its standard error captured line by line and time-stamped, tool calls over
//! its loopback port timed from the client side, a clean stop, process memory and graphics-card
//! memory sampled from outside (`tasklist` or `/proc`, `nvidia-smi` when present, else skipped),
//! and the report folder under `runs/`. Nothing here talks to any host but 127.0.0.1.
use nmjson::Value;
use std::collections::BTreeMap;
use std::io::{BufRead as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Waits without `std::thread::sleep` (refused by the workspace's clippy rules): parks until the
/// time has passed, across spurious wake-ups.
pub fn park(d: Duration) {
    let until = Instant::now() + d;
    while let Some(left) = until.checked_duration_since(Instant::now()).filter(|l| !l.is_zero()) {
        std::thread::park_timeout(left);
    }
}

/// The value after `flag`, if given.
pub fn opt<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2).find(|w| w.first().map(String::as_str) == Some(flag)).and_then(|w| w.get(1)).map(String::as_str)
}

/// Whether `flag` is present.
pub fn has(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

/// The arguments after a lone `--` (serve options passed through to the daemon).
pub fn passthrough(args: &[String]) -> Vec<String> {
    args.iter().position(|a| a == "--").map(|i| args.get(i + 1..).unwrap_or_default().to_vec()).unwrap_or_default()
}

/// A length of time: `90`, `90s`, `45m`, `4h`, `1.5h`.
/// # Errors
/// Not a number with an optional s, m or h.
pub fn parse_duration(t: &str) -> Result<Duration, String> {
    let t = t.trim();
    let (num, mul) = match t.chars().last() {
        Some('h') => (&t[..t.len() - 1], 3600.0),
        Some('m') => (&t[..t.len() - 1], 60.0),
        Some('s') => (&t[..t.len() - 1], 1.0),
        _ => (t, 1.0),
    };
    let v: f64 = num.trim().parse().map_err(|_| format!("not a length of time: {t} (use 90s, 45m or 4h)"))?;
    if !v.is_finite() || v <= 0.0 {
        return Err(format!("not a positive length of time: {t}"));
    }
    Ok(Duration::from_secs_f64(v * mul))
}

/// One time-stamped line of the daemon's standard error (seconds since it was started).
pub type Line = (f64, String);

/// A warm daemon run as a child process for a harness.
pub struct Daemon {
    pub child: std::process::Child,
    pub port: u16,
    pub port_file: PathBuf,
    pub started: Instant,
    pub lines: Arc<Mutex<Vec<Line>>>,
}

impl Daemon {
    /// Starts `<this program> daemon <args>` with standard error captured into `log` and kept in
    /// memory with the time each line arrived.
    /// # Errors
    /// The program cannot be found or started, or the log cannot be created.
    pub fn spawn(args: &[String], port: u16, port_file: &Path, log: &Path) -> Result<Self, String> {
        let exe = std::env::current_exe().map_err(|e| format!("this program's path: {e}"))?;
        let mut cmd = std::process::Command::new(exe);
        cmd.arg("daemon").args(args).arg("--port").arg(port.to_string()).arg("--port-file").arg(port_file);
        cmd.stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::piped());
        let started = Instant::now();
        let mut child = cmd.spawn().map_err(|e| format!("start the daemon: {e}"))?;
        let lines: Arc<Mutex<Vec<Line>>> = Arc::new(Mutex::new(Vec::new()));
        if let Some(err) = child.stderr.take() {
            let mut file = std::fs::File::create(log).map_err(|e| format!("{}: {e}", log.display()))?;
            let keep = Arc::clone(&lines);
            std::thread::spawn(move || {
                for line in std::io::BufReader::new(err).lines().map_while(Result::ok) {
                    let at = started.elapsed().as_secs_f64();
                    let _ = writeln!(file, "[{at:9.2}] {line}");
                    if let Ok(mut l) = keep.lock() {
                        l.push((at, line));
                    }
                }
            });
        }
        Ok(Self { child, port, port_file: port_file.to_path_buf(), started, lines })
    }

    /// Seconds since the daemon was started.
    pub fn secs(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    /// Captured lines containing `needle`.
    pub fn count_lines(&self, needle: &str) -> usize {
        self.lines.lock().map(|l| l.iter().filter(|(_, x)| x.contains(needle)).count()).unwrap_or(0)
    }

    /// The last `n` captured lines.
    pub fn tail(&self, n: usize) -> Vec<String> {
        self.lines.lock().map(|l| l.iter().rev().take(n).rev().map(|(at, x)| format!("[{at:.2}] {x}")).collect()).unwrap_or_default()
    }

    /// Whether the child is still running.
    pub fn alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// Waits until the port accepts a connection (`true`) or the child ends or the time is up.
    pub fn wait_listening(&mut self, limit: Duration) -> bool {
        let until = Instant::now() + limit;
        while Instant::now() < until {
            if std::net::TcpStream::connect_timeout(&([127, 0, 0, 1], self.port).into(), Duration::from_millis(500)).is_ok() {
                return true;
            }
            if !self.alive() {
                return false;
            }
            park(Duration::from_millis(200));
        }
        false
    }

    /// Asks the daemon to stop (`STOP` on its port) and waits for the process to end: (whether it
    /// ended by itself within `limit` with a success status, milliseconds from the request to the
    /// end, its exit code). A daemon that does not end in time is killed.
    pub fn stop(&mut self, limit: Duration) -> (bool, u128, Option<i32>) {
        let asked = Instant::now();
        if let Ok(mut s) = std::net::TcpStream::connect_timeout(&([127, 0, 0, 1], self.port).into(), Duration::from_secs(2)) {
            let _ = s.set_read_timeout(Some(Duration::from_secs(5)));
            let _ = writeln!(s, "STOP");
            let _ = s.flush();
            let mut resp = String::new();
            let _ = std::io::BufReader::new(s).read_line(&mut resp);
        }
        while asked.elapsed() < limit {
            if let Ok(Some(st)) = self.child.try_wait() {
                return (st.success(), asked.elapsed().as_millis(), st.code());
            }
            park(Duration::from_millis(100));
        }
        let _ = self.child.kill();
        let code = self.child.wait().ok().and_then(|s| s.code());
        (false, asked.elapsed().as_millis(), code)
    }
}

/// One tool call's outcome seen from the client: milliseconds from connect to the full answer, and
/// the answer's text or the fault.
#[derive(Clone, Debug)]
pub struct Called {
    pub ms: f64,
    pub result: Result<String, String>,
    /// The connection ended with a read error after the full answer had arrived (the answer is
    /// kept and the event counted, not hidden).
    pub late_reset: bool,
}

/// The JSON-RPC lines a client sends for one tool call: initialize, then the call.
pub fn call_lines(tool: &str, arguments: &Value) -> String {
    let init = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{},\"clientInfo\":{\"name\":\"nano-harness\",\"version\":\"1.0\"}}}";
    let mut params = BTreeMap::new();
    params.insert("name".to_owned(), Value::String(tool.to_owned()));
    params.insert("arguments".to_owned(), arguments.clone());
    let mut call = BTreeMap::new();
    call.insert("jsonrpc".to_owned(), Value::String("2.0".into()));
    call.insert("id".to_owned(), Value::Number("2".into()));
    call.insert("method".to_owned(), Value::String("tools/call".into()));
    call.insert("params".to_owned(), Value::Object(params));
    format!("{init}\n{}\n", Value::Object(call).encode().unwrap_or_default())
}

/// Whether a complete answer line of call id 2 (ending in a newline) is in `lines`.
pub fn has_answer(lines: &str) -> bool {
    lines.split_inclusive('\n').any(|l| l.ends_with('\n') && l.contains("\"id\":2"))
}

/// Reads the answer line of call id 2: the first text content, or the fault (a JSON-RPC error, or
/// a result marked `isError`).
/// # Errors
/// No answer line, unreadable JSON, an error object or an `isError` result.
pub fn read_answer(lines: &str) -> Result<String, String> {
    let line = lines.lines().find(|l| l.contains("\"id\":2")).ok_or("no answer to the call")?;
    let v = nmjson::parse(line.as_bytes()).map_err(|e| format!("answer is not JSON: {e}"))?;
    if let Ok(e) = v.get("error") {
        return Err(format!("error: {}", e.encode().unwrap_or_default()));
    }
    let r = v.get("result").map_err(|_| "answer has no result".to_owned())?;
    let text = r
        .get("content")
        .ok()
        .and_then(|c| c.array().ok())
        .and_then(|a| a.first())
        .and_then(|x| x.get("text").ok())
        .and_then(|t| t.string().ok())
        .map_or_else(|| r.encode().unwrap_or_default(), str::to_owned);
    if matches!(r.get("isError"), Ok(Value::Bool(true))) {
        return Err(format!("tool error: {}", text.chars().take(300).collect::<String>()));
    }
    Ok(text)
}

/// One tool call on the daemon's port, timed from the connect to the last byte of the answer.
pub fn call(port: u16, tool: &str, arguments: &Value, limit: Duration) -> Called {
    let t0 = Instant::now();
    let mut late_reset = false;
    let result = (|| {
        let mut s = std::net::TcpStream::connect_timeout(&([127, 0, 0, 1], port).into(), Duration::from_secs(5)).map_err(|e| format!("connect: {e}"))?;
        s.set_read_timeout(Some(limit)).map_err(|e| e.to_string())?;
        s.write_all(call_lines(tool, arguments).as_bytes()).map_err(|e| format!("write: {e}"))?;
        s.flush().map_err(|e| format!("write: {e}"))?;
        let _ = s.shutdown(std::net::Shutdown::Write);
        // 2026-11-25: read as bytes; a read error keeps what arrived before it. A reset after the
        // full answer line is counted (`late_reset`) and the answer kept; a reset before it is a
        // fault that says how many bytes had arrived.
        let mut raw = Vec::new();
        let read = std::io::Read::read_to_end(&mut s, &mut raw);
        let out = String::from_utf8_lossy(&raw);
        if let Err(e) = read {
            if !has_answer(&out) {
                return Err(format!("read: {e} (after {} bytes)", raw.len()));
            }
            late_reset = true;
        }
        read_answer(&out)
    })();
    Called { ms: t0.elapsed().as_secs_f64() * 1000.0, result, late_reset }
}

/// The arguments of an `nm_context_pack` call for a question.
pub fn question_args(q: &str) -> Value {
    let mut m = BTreeMap::new();
    m.insert("question".to_owned(), Value::String(q.to_owned()));
    Value::Object(m)
}

/// The resident memory of a process in KiB (`tasklist` on Windows, `/proc/<pid>/status`
/// elsewhere), or `None` when it cannot be read.
pub fn rss_kib(pid: u32) -> Option<u64> {
    if cfg!(windows) {
        let out = std::process::Command::new("tasklist").args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"]).output().ok()?;
        parse_tasklist(&String::from_utf8_lossy(&out.stdout))
    } else {
        parse_proc_status(&std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?)
    }
}

/// The memory column of one `tasklist /FO CSV /NH` row (`"name","pid","session","n","123,456 K"`),
/// in KiB. Digit group separators of any locale (comma, dot, space, no-break space) are dropped.
pub fn parse_tasklist(text: &str) -> Option<u64> {
    let line = text.lines().find(|l| l.trim_start().starts_with('"'))?;
    let last = line.trim().rsplit("\",\"").next()?.trim_matches('"');
    let digits: String = last.chars().filter(char::is_ascii_digit).collect();
    let kib: u64 = digits.parse().ok()?;
    let unit = last.trim_end().chars().last()?;
    match unit.to_ascii_uppercase() {
        'K' => Some(kib),
        'M' => Some(kib * 1024),
        _ => None,
    }
}

/// `VmRSS` of a `/proc/<pid>/status` text, in KiB.
pub fn parse_proc_status(text: &str) -> Option<u64> {
    text.lines().find_map(|l| l.strip_prefix("VmRSS:")).and_then(|r| r.split_whitespace().next()).and_then(|n| n.parse().ok())
}

/// A graphics-card memory sample: the process's own use when the driver reports it (it does not
/// under the Windows display driver model, which prints `[N/A]`), and the card's total in use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Gpu {
    pub process_mib: Option<u64>,
    pub used_mib: u64,
}

/// Samples the graphics card with `nvidia-smi`; `None` when it is not installed or fails (the
/// harness then records the graphics-card samples as skipped).
pub fn gpu_sample(pid: u32) -> Option<Gpu> {
    let run = |q: &[&str]| -> Option<String> {
        let o = std::process::Command::new("nvidia-smi").args(q).output().ok()?;
        o.status.success().then(|| String::from_utf8_lossy(&o.stdout).into_owned())
    };
    let used = parse_gpu_used(&run(&["--query-gpu=memory.used", "--format=csv,noheader,nounits"])?)?;
    let apps = run(&["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).unwrap_or_default();
    Some(Gpu { process_mib: parse_compute_apps(&apps, pid), used_mib: used })
}

/// The first card's memory in use (MiB) from `--query-gpu=memory.used --format=csv,noheader,nounits`.
pub fn parse_gpu_used(text: &str) -> Option<u64> {
    text.lines().map(str::trim).find(|l| !l.is_empty()).and_then(|l| l.split(',').next()).and_then(|n| n.trim().parse().ok())
}

/// One process's memory (MiB) from `--query-compute-apps=pid,used_memory --format=csv,noheader,nounits`
/// rows (`1234, 2743`); `None` when the process is absent or the driver prints `[N/A]`.
pub fn parse_compute_apps(text: &str, pid: u32) -> Option<u64> {
    text.lines().find_map(|l| {
        let mut parts = l.split(',').map(str::trim);
        let p: u32 = parts.next()?.parse().ok()?;
        (p == pid).then(|| parts.next()?.parse().ok()).flatten()
    })
}

/// The seconds in a `... ready at 12.34 s` or `...; done at 3.5 s` line after `marker`.
pub fn secs_after(line: &str, marker: &str) -> Option<f64> {
    let rest = &line[line.find(marker)? + marker.len()..];
    rest.split_whitespace().next()?.trim_end_matches(';').parse().ok()
}

/// The p-th percentile (0 to 100, nearest rank) of a list.
pub fn percentile(v: &[f64], p: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
    let rank = ((p / 100.0) * s.len() as f64).ceil().max(1.0) as usize;
    s.get(rank.min(s.len()) - 1).copied()
}

/// A number as a JSON value (finite only; otherwise null).
pub fn num(x: f64) -> Value {
    if x.is_finite() { Value::Number(format!("{:.3}", x).trim_end_matches('0').trim_end_matches('.').to_owned()) } else { Value::Null }
}

/// A whole number as a JSON value.
pub fn int(x: u64) -> Value {
    Value::Number(x.to_string())
}

/// A string as a JSON value.
pub fn s(x: &str) -> Value {
    Value::String(x.to_owned())
}

/// An object from pairs.
pub fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

/// A new report folder `runs/<kind>-<ms>/`.
/// # Errors
/// The folder cannot be created.
pub fn report_dir(home: &Path, kind: &str) -> Result<PathBuf, String> {
    let d = home.join("runs").join(format!("{kind}-{}", nmtype::now_ms()));
    std::fs::create_dir_all(&d).map_err(|e| format!("{}: {e}", d.display()))?;
    Ok(d)
}

/// Writes `report.json` and `report.md` into a report folder.
/// # Errors
/// A file cannot be written.
pub fn write_report(dir: &Path, json: &Value, md: &str) -> Result<(), String> {
    let j = dir.join("report.json");
    std::fs::write(&j, nmjson::pretty::pretty(json)).map_err(|e| format!("{}: {e}", j.display()))?;
    let m = dir.join("report.md");
    std::fs::write(&m, md).map_err(|e| format!("{}: {e}", m.display()))
}

/// Whether a path is on the D: drive (Windows); on other systems every path counts as local.
pub fn on_d_drive(p: &Path) -> bool {
    if !cfg!(windows) {
        return true;
    }
    let t = p.display().to_string();
    let t = t.trim_start_matches(r"\\?\");
    t.len() >= 2 && t.as_bytes().first().is_some_and(|b| b.eq_ignore_ascii_case(&b'd')) && t.as_bytes().get(1) == Some(&b':')
}

/// Questions for a run: from a file (`{"questions": [{"question": ...}]}` as in
/// `tests/dev/questions.json`, or a plain list of strings), else made from the names of the
/// project's source files.
/// # Errors
/// The file cannot be read or holds no question.
pub fn load_questions(file: Option<&Path>, project: &Path) -> Result<Vec<String>, String> {
    if let Some(f) = file {
        let b = std::fs::read(f).map_err(|e| format!("{}: {e}", f.display()))?;
        let v = nmjson::parse(&b).map_err(|e| format!("{}: {e}", f.display()))?;
        let qs = questions_of(&v);
        if qs.is_empty() {
            return Err(format!("{}: no questions found", f.display()));
        }
        return Ok(qs);
    }
    let mut stems = Vec::new();
    collect_stems(project, 0, &mut stems);
    stems.sort();
    stems.dedup();
    let mut qs: Vec<String> = stems
        .iter()
        .take(40)
        .flat_map(|st| [format!("Where is {st} defined and what does it do?"), format!("What calls into {st}?")])
        .collect();
    if qs.is_empty() {
        qs.push("Where is the main entry point of this project?".to_owned());
    }
    Ok(qs)
}

/// The question texts of a parsed questions file.
pub fn questions_of(v: &Value) -> Vec<String> {
    let list = v.get("questions").ok().and_then(|q| q.array().ok()).or_else(|| v.array().ok()).unwrap_or_default();
    list.iter()
        .filter_map(|q| q.string().ok().map(str::to_owned).or_else(|| q.get("question").ok().and_then(|t| t.string().ok()).map(str::to_owned)))
        .filter(|q| !q.trim().is_empty())
        .collect()
}

/// Folders a harness never copies or walks.
pub const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", "scratch", "weights", ".venv", "__pycache__", "dist", "build"];

/// Source file extensions a harness edits and names questions after.
pub const SOURCE_EXT: &[&str] = &["rs", "py", "ts", "tsx", "js", "go", "java", "c", "cc", "cpp", "h", "hpp", "cs", "rb", "kt", "swift"];

fn collect_stems(dir: &Path, depth: usize, out: &mut Vec<String>) {
    if depth > 6 || out.len() > 400 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().into_owned();
        if p.is_dir() {
            if !SKIP_DIRS.contains(&name.as_str()) && !name.starts_with('.') {
                collect_stems(&p, depth + 1, out);
            }
        } else if p.extension().and_then(|x| x.to_str()).is_some_and(|x| SOURCE_EXT.contains(&x))
            && let Some(st) = p.file_stem().and_then(|x| x.to_str())
            && !["mod", "lib", "main", "index", "__init__"].contains(&st)
        {
            out.push(st.to_owned());
        }
    }
}

/// A small deterministic random sequence (xorshift), so a run can be repeated with its seed.
pub struct Rng(u64);

impl Rng {
    pub const fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed })
    }
    pub const fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// A number below `n` (0 when `n` is 0).
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        usize::try_from(self.next() % (n as u64)).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answer_line_complete() {
        assert!(has_answer("{\"id\":1}\n{\"id\":2,\"result\":{}}\n"));
        assert!(!has_answer("{\"id\":1}\n{\"id\":2,\"res"));
        assert!(!has_answer("{\"id\":1}\n"));
    }

    #[test]
    fn durations() {
        assert_eq!(parse_duration("90").unwrap(), Duration::from_secs(90));
        assert_eq!(parse_duration("45m").unwrap(), Duration::from_secs(2700));
        assert_eq!(parse_duration("4h").unwrap(), Duration::from_secs(14400));
        assert_eq!(parse_duration("1.5h").unwrap(), Duration::from_secs(5400));
        assert!(parse_duration("0s").is_err());
        assert!(parse_duration("soon").is_err());
        assert!(parse_duration("-3m").is_err());
    }

    #[test]
    fn memory_parsers() {
        let row = "\"neuromorph-nano.exe\",\"11240\",\"Console\",\"1\",\"1,234,568 K\"\r\n";
        assert_eq!(parse_tasklist(row), Some(1_234_568));
        assert_eq!(parse_tasklist("\"x.exe\",\"1\",\"Services\",\"0\",\"98.304 K\""), Some(98_304));
        assert_eq!(parse_tasklist("INFO: No tasks are running which match the specified criteria."), None);
        assert_eq!(parse_proc_status("Name:\tx\nVmRSS:\t  51200 kB\nThreads: 4\n"), Some(51_200));
        assert_eq!(parse_proc_status("Name: x\n"), None);
        assert_eq!(parse_gpu_used("2743\n"), Some(2743));
        assert_eq!(parse_gpu_used("  812, 4096\n"), Some(812));
        assert_eq!(parse_gpu_used(""), None);
        assert_eq!(parse_compute_apps("1200, 300\n11240, 2743\n", 11240), Some(2743));
        assert_eq!(parse_compute_apps("11240, [N/A]\n", 11240), None);
        assert_eq!(parse_compute_apps("", 1), None);
    }

    #[test]
    fn answers_and_lines() {
        let ok = "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"pack\"}]}}\n";
        assert_eq!(read_answer(ok).unwrap(), "pack");
        let bad = "{\"jsonrpc\":\"2.0\",\"id\":2,\"error\":{\"code\":-32601,\"message\":\"no\"}}";
        assert!(read_answer(bad).is_err());
        let tool_err = "{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"isError\":true,\"content\":[{\"type\":\"text\",\"text\":\"boom\"}]}}";
        assert!(read_answer(tool_err).unwrap_err().contains("boom"));
        assert!(read_answer("").is_err());
        let l = call_lines("nm_context_pack", &question_args("Where is \"x\"?"));
        assert_eq!(l.lines().count(), 2);
        let second = nmjson::parse(l.lines().nth(1).unwrap().as_bytes()).unwrap();
        assert_eq!(second.get("params").unwrap().get("name").unwrap().string().unwrap(), "nm_context_pack");
        assert_eq!(secs_after("start in the background: open 1.00 s; ready at 12.34 s", "ready at"), Some(12.34));
        assert_eq!(secs_after("flow: drones took in 9 pieces; done at 3.5 s", "done at"), Some(3.5));
        assert_eq!(secs_after("nothing here", "ready at"), None);
    }

    #[test]
    fn numbers_args_and_paths() {
        assert_eq!(percentile(&[5.0, 1.0, 3.0, 2.0, 4.0], 50.0), Some(3.0));
        assert_eq!(percentile(&[5.0, 1.0, 3.0, 2.0, 4.0], 95.0), Some(5.0));
        assert_eq!(percentile(&[], 50.0), None);
        let a: Vec<String> = ["--duration", "2h", "--keep", "--", "--neural"].iter().map(|x| (*x).to_owned()).collect();
        assert_eq!(opt(&a, "--duration"), Some("2h"));
        assert!(has(&a, "--keep"));
        assert_eq!(passthrough(&a), vec!["--neural".to_owned()]);
        if cfg!(windows) {
            assert!(on_d_drive(Path::new("D:\\x")));
            assert!(on_d_drive(Path::new("\\\\?\\d:\\x")));
            assert!(!on_d_drive(Path::new("C:\\Users")));
        }
        let mut r = Rng::new(7);
        let first: Vec<usize> = (0..5).map(|_| r.below(10)).collect();
        let mut r2 = Rng::new(7);
        assert_eq!(first, (0..5).map(|_| r2.below(10)).collect::<Vec<_>>());
        assert_eq!(num(1.5).encode().unwrap(), "1.5");
        assert_eq!(num(2.0).encode().unwrap(), "2");
        let qs = questions_of(&nmjson::parse(b"{\"questions\":[{\"question\":\"a?\"},{\"id\":\"x\"}]}").unwrap());
        assert_eq!(qs, vec!["a?".to_owned()]);
        assert_eq!(questions_of(&nmjson::parse(b"[\"b?\", \"\"]").unwrap()), vec!["b?".to_owned()]);
    }
}
