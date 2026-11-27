//! One child process with a deadline (law 7: every run under one minute). Both pipes are
//! drained on their own threads so a chatty child never blocks; at the deadline the child
//! is killed and the result says so. A timeout is a finding, never retried longer here.
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc::{RecvTimeoutError, Sender, channel};
use std::time::{Duration, Instant};

/// Default deadline for one command: under the 60 s run ceiling with room for parsing.
pub const DEADLINE: Duration = Duration::from_secs(55);

/// What one command did.
#[derive(Clone, Debug, Default)]
pub struct Output {
    /// Exit status was success.
    pub ok: bool,
    /// Standard output, lossy UTF-8.
    pub stdout: String,
    /// Standard error, lossy UTF-8.
    pub stderr: String,
    /// Wall milliseconds.
    pub ms: u64,
    /// Killed at the deadline.
    pub timed_out: bool,
}

/// Read a pipe to its end on its own thread and send (which, bytes) when done.
fn drain(which: usize, r: Option<impl Read + Send + 'static>, tx: Sender<(usize, Vec<u8>)>) {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut r) = r {
            let _ = r.read_to_end(&mut buf);
        }
        let _ = tx.send((which, buf));
    });
}

/// Run `prog args` in `dir` with extra environment `envs`, killed at `deadline`. Both pipes
/// closing marks the end; a child still holding them at the deadline is killed, and pipes
/// held open by its own children afterwards are given up after two seconds.
#[must_use]
pub fn run(dir: &Path, prog: &str, args: &[&str], envs: &[(&str, &str)], deadline: Duration) -> Output {
    let t0 = Instant::now();
    let mut c = Command::new(prog);
    c.args(args).current_dir(dir).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    for (k, v) in envs {
        c.env(k, v);
    }
    let mut child = match c.spawn() {
        Ok(ch) => ch,
        Err(e) => return Output { stderr: format!("{prog}: {e}"), ..Output::default() },
    };
    let (tx, rx) = channel();
    drain(0, child.stdout.take(), tx.clone());
    drain(1, child.stderr.take(), tx);
    let mut bufs: [Option<Vec<u8>>; 2] = [None, None];
    let mut timed_out = false;
    while bufs.iter().any(Option::is_none) {
        let wait = if timed_out { Duration::from_secs(2) } else { deadline.saturating_sub(t0.elapsed()) };
        match rx.recv_timeout(wait) {
            Ok((w, b)) => {
                if let Some(slot) = bufs.get_mut(w) {
                    *slot = Some(b);
                }
            }
            Err(RecvTimeoutError::Timeout) if !timed_out => {
                let _ = child.kill();
                timed_out = true;
            }
            Err(_) => break,
        }
    }
    let status = child.wait().ok();
    let [out, err] = bufs;
    let stdout = String::from_utf8_lossy(&out.unwrap_or_default()).into_owned();
    let stderr = String::from_utf8_lossy(&err.unwrap_or_default()).into_owned();
    Output {
        ok: !timed_out && status.is_some_and(|s| s.success()),
        stdout,
        stderr,
        ms: u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX),
        timed_out,
    }
}

/// Section check: `cargo --version` answers; a missing program fails cleanly.
/// # Errors
/// One line naming what failed.
pub fn check() -> Result<(), String> {
    let here = nmtype::nano_root();
    let here = here.as_path();
    let o = run(here, "cargo", &["--version"], &[], Duration::from_secs(20));
    if !o.ok || !o.stdout.starts_with("cargo ") || o.timed_out {
        return Err(format!("cargo --version: ok {} out {:?} err {:?}", o.ok, o.stdout, o.stderr));
    }
    let m = run(here, "nmtool-no-such-program", &[], &[], Duration::from_secs(5));
    if m.ok || m.stderr.is_empty() {
        return Err("a missing program did not fail".into());
    }
    Ok(())
}
